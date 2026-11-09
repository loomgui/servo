/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! GPU-rasterized 2D canvas backend (Loom prototype).
//!
//! Canvas commands are recorded into a `vello_hybrid` scene, exactly like the `vello_cpu`
//! backend records them, but the scene is rasterized on the GPU into a surfman surface that lives
//! on the same adapter as WebRender. The surface is handed to WebRender as an external texture
//! through a swap chain, so no canvas pixels are copied through CPU memory per update.
//!
//! All GPU canvases on a canvas thread share one GLES 3.0 context, one `vello_hybrid` renderer
//! (its shader programs take about half a second to compile under ANGLE) and one image atlas. Each
//! canvas owns a swap chain on that context and a retained texture. Content persists across
//! updates: every rendered frame is copied on the GPU into the retained texture, which is drawn
//! back as the background when later draws are not preceded by a full-canvas clear.
//!
//! Images (`drawImage`, `putImageData`, patterns) are uploaded into `vello_hybrid`'s image atlas
//! once per distinct content: uploads are cached per thread by a content hash, pinned while a
//! pending scene samples them, and evicted when idle or over budget.

#![allow(
    unsafe_code,
    reason = "GL calls on the canvas's own context, which this module keeps current"
)]

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::sync::Arc;

use euclid::default::{Point2D, Rect, Size2D, Transform2D};
use fonts::FontIdentifier;
use glow::HasContext;
use kurbo::Shape;
use log::warn;
use paint_api::{GpuCanvasShared, SerializableImageData, WebRenderImageHandlerType};
use pixels::{Snapshot, SnapshotAlphaMode, SnapshotPixelFormat};
use profile_traits::mem::ReportKind;
use servo_canvas_traits::canvas::{
    CompositionOptions, CompositionOrBlending, CompositionStyle, FillOrStrokeStyle, FillRule,
    LineOptions, Path, ShadowOptions, TextRun,
};
use surfman::chains::SwapChainsAPI;
use surfman::{
    Context, ContextAttributeFlags, ContextAttributes, Device, GLVersion, SurfaceAccess,
    SurfaceType,
};
use vello_cpu::{kurbo, peniko};
use vello_hybrid::{
    ImageId, RenderSize, Resources, SampleRect, Scene, TextureId, WebGl2RenderingContext,
    WebGlRenderer, WebGlTextureBindings,
};
use webrender_api::{
    ExternalImageData, ExternalImageId, ExternalImageType, ImageBufferKind, ImageDescriptor,
    ImageDescriptorFlags, ImageFormat,
};

use crate::backend::{CanvasStoreSizesPerType, Convert, GenericDrawTarget};
use crate::canvas_data::Filter;
use crate::vello_cpu_backend::{paint, shadow_filter, snapshot_as_pixmap};

/// Texture binding id used to draw retained content back into a scene.
const RETAINED_TEXTURE: TextureId = TextureId(1);
/// Largest image side the renderer's atlas pages hold (`AtlasConfig::default().atlas_size`).
const MAX_ATLAS_IMAGE_SIDE: u16 = 4096;
/// Atlas bytes cached uploads may hold before unpinned ones are evicted, oldest first.
const UPLOAD_CACHE_BUDGET: usize = 64 << 20;
/// Rasterizations (on any canvas of the thread) after which an unused upload is evicted.
const UPLOAD_CACHE_MAX_IDLE: u64 = 600;

/// Identifies uploaded image content: its size and a hash of its premultiplied pixels.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct UploadKey {
    width: u16,
    height: u16,
    hash: u64,
}

impl UploadKey {
    fn of(pixmap: &vello_cpu::Pixmap) -> Self {
        let mut hasher = rustc_hash::FxHasher::default();
        pixmap.data_as_u8_slice().hash(&mut hasher);
        Self {
            width: pixmap.width(),
            height: pixmap.height(),
            hash: hasher.finish(),
        }
    }
}

struct Upload {
    id: ImageId,
    /// Pending scenes that sample this upload; pinned uploads are never evicted.
    pins: u32,
    last_used: u64,
    bytes: usize,
}

/// Atlas uploads shared by every canvas of a thread.
#[derive(Default)]
struct ImageUploads {
    entries: HashMap<UploadKey, Upload>,
    bytes: usize,
    /// Counts rasterizations; drives idle eviction.
    epoch: u64,
}

fn warn_once(flag: &'static std::thread::LocalKey<Cell<bool>>, message: &str) {
    flag.with(|warned| {
        if !warned.replace(true) {
            warn!("GPU canvas: {message}");
        }
    });
}

thread_local! {
    static WARNED_IMAGE: Cell<bool> = const { Cell::new(false) };
    /// Font data shared by all GPU canvases on this thread, as in the CPU backend.
    static FONT_CACHE: RefCell<HashMap<FontIdentifier, peniko::FontData>> = RefCell::default();
}

/// Run `f` with the `peniko` font for `text_run`, loading it into the thread's cache once.
fn with_font(text_run: &TextRun, f: impl FnOnce(&peniko::FontData)) {
    FONT_CACHE.with(|cache| {
        let identifier = &text_run.font.identifier;
        if !cache.borrow().contains_key(identifier) {
            let Some(font_data_and_index) = text_run.font.font_data_and_index() else {
                return;
            };
            cache
                .borrow_mut()
                .insert(identifier.clone(), font_data_and_index.convert());
        }
        if let Some(font) = cache.borrow().get(identifier) {
            f(font);
        }
    });
}

fn glyphs(text_run: &TextRun) -> impl Iterator<Item = vello_cpu::Glyph> + Clone + '_ {
    text_run
        .glyphs_and_positions
        .iter()
        .map(|glyph| vello_cpu::Glyph {
            id: glyph.id,
            x: glyph.point.x,
            y: glyph.point.y,
        })
}

/// GPU memory shared by every canvas of this thread, as `(name, bytes)`: the renderer's image
/// atlas and the cached uploads in it. Empty when no GPU canvas was created on this thread.
pub(crate) fn shared_memory_report() -> Vec<(&'static str, usize)> {
    GPU_THREAD.with(|slot| {
        let Some(thread) = slot.borrow().clone() else {
            return Vec::new();
        };
        let atlas = thread.renderer.borrow().atlas_info();
        let atlas_bytes =
            atlas.width as usize * atlas.height as usize * 4 * atlas.layer_count as usize;
        vec![
            ("gpu-image-atlas", atlas_bytes),
            ("gpu-cached-uploads", thread.uploads.borrow().bytes),
        ]
    })
}

/// Whether GPU canvases can be created on this thread. Selecting the backend calls this first so
/// that any failure falls back to the CPU backend instead of panicking later.
pub(crate) fn available() -> bool {
    GpuThread::get().is_some()
}

fn shared() -> Option<&'static GpuCanvasShared> {
    paint_api::gpu_canvas_shared()
}

/// GL state shared by every GPU canvas on a canvas thread: a surfman device and context on
/// WebRender's adapter, one `vello_hybrid` renderer and its resources (image and glyph atlases,
/// gradient ramps). Canvases own only a detached swap chain on this context and a retained
/// texture. A canvas renders into its swap chain's back buffer as a texture, so the context's own
/// (1×1) surface never changes: rebinding surfaces makes ANGLE wait on every switch.
struct GpuThread {
    device: Device,
    /// Never dropped: this lives until the canvas thread exits, and tearing down EGL objects from
    /// thread-exit callbacks can deadlock ANGLE.
    context: std::mem::ManuallyDrop<RefCell<Context>>,
    gl: WebGl2RenderingContext,
    renderer: RefCell<WebGlRenderer>,
    resources: RefCell<Resources>,
    uploads: RefCell<ImageUploads>,
    shared: &'static GpuCanvasShared,
    /// Reads a canvas's back buffer into its retained texture.
    copy_framebuffer: glow::NativeFramebuffer,
}

thread_local! {
    static GPU_THREAD: RefCell<Option<Rc<GpuThread>>> = const { RefCell::new(None) };
    /// GPU setup failed on this thread; canvases use the CPU backend.
    static GPU_FAILED: Cell<bool> = const { Cell::new(false) };
}

impl GpuThread {
    fn get() -> Option<Rc<Self>> {
        if GPU_FAILED.with(Cell::get) {
            return None;
        }
        GPU_THREAD.with(|slot| {
            if let Some(gpu) = slot.borrow().as_ref() {
                return Some(gpu.clone());
            }
            match Self::new() {
                Ok(gpu) => {
                    let gpu = Rc::new(gpu);
                    *slot.borrow_mut() = Some(gpu.clone());
                    Some(gpu)
                },
                Err(error) => {
                    warn!("GPU canvas: using the CPU backend: {error}");
                    GPU_FAILED.with(|failed| failed.set(true));
                    None
                },
            }
        })
    }

    fn new() -> Result<Self, String> {
        let shared = shared().ok_or("GPU canvas state was not published")?;
        let details = shared
            .surfman_details
            .any()
            .ok_or("no painter surfman details")?;
        let device = details
            .connection
            .create_device(&details.adapter)
            .map_err(|e| format!("device: {e:?}"))?;
        let attributes = ContextAttributes {
            version: GLVersion::new(3, 0),
            flags: ContextAttributeFlags::ALPHA |
                ContextAttributeFlags::DEPTH |
                ContextAttributeFlags::STENCIL,
        };
        let descriptor = device
            .create_context_descriptor(&attributes)
            .map_err(|e| format!("context descriptor: {e:?}"))?;
        let mut context = device
            .create_context(&descriptor, None)
            .map_err(|e| format!("context: {e:?}"))?;
        let setup = (|| {
            let surface = device
                .create_surface(
                    &context,
                    SurfaceAccess::GPUOnly,
                    SurfaceType::Generic {
                        size: Size2D::new(1, 1),
                    },
                )
                .map_err(|e| format!("surface: {e:?}"))?;
            device
                .bind_surface_to_context(&mut context, surface)
                .map_err(|(e, _)| format!("bind surface: {e:?}"))?;
            device
                .make_context_current(&context)
                .map_err(|e| format!("make current: {e:?}"))
        })();
        if let Err(error) = setup {
            let _ = device.destroy_context(&mut context);
            return Err(error);
        }
        let glow = unsafe {
            glow::Context::from_loader_function(|symbol| device.get_proc_address(&context, symbol))
        };
        let copy_framebuffer = unsafe { glow.create_framebuffer()? };
        let gl = WebGl2RenderingContext::from_glow(glow, 1, 1);
        let (renderer, resources) = WebGlRenderer::new(gl.clone());
        Ok(Self {
            device,
            context: std::mem::ManuallyDrop::new(RefCell::new(context)),
            gl,
            renderer: RefCell::new(renderer),
            resources: RefCell::new(resources),
            uploads: RefCell::default(),
            shared,
            copy_framebuffer,
        })
    }

    fn make_current(&self) {
        if let Err(error) = self.device.make_context_current(&self.context.borrow()) {
            warn!("GPU canvas: make current failed: {error:?}");
        }
    }

    /// The atlas image for `pixmap`, uploading it if this content is not cached, pinned once.
    fn pin_upload(&self, pixmap: &vello_cpu::Pixmap) -> Option<(UploadKey, ImageId)> {
        let key = UploadKey::of(pixmap);
        let mut uploads = self.uploads.borrow_mut();
        let epoch = uploads.epoch;
        if let Some(upload) = uploads.entries.get_mut(&key) {
            upload.pins += 1;
            upload.last_used = epoch;
            return Some((key, upload.id));
        }
        self.make_current();
        let bytes = usize::from(key.width) * usize::from(key.height) * 4;
        let mut id = self.try_upload(pixmap);
        if id.is_none() {
            // The atlas is full: drop every unpinned upload and retry once.
            self.evict(&mut uploads, |_, _| true);
            id = self.try_upload(pixmap);
        }
        let Some(id) = id else {
            warn_once(
                &WARNED_IMAGE,
                "the image atlas is full; skipping an image draw",
            );
            return None;
        };
        uploads.bytes += bytes;
        uploads.entries.insert(
            key,
            Upload {
                id,
                pins: 1,
                last_used: epoch,
                bytes,
            },
        );
        Some((key, id))
    }

    fn try_upload(&self, pixmap: &vello_cpu::Pixmap) -> Option<ImageId> {
        self.renderer
            .borrow_mut()
            .try_upload_image(&mut self.resources.borrow_mut(), pixmap)
    }

    fn unpin_uploads(&self, keys: impl Iterator<Item = UploadKey>) {
        let mut uploads = self.uploads.borrow_mut();
        for key in keys {
            if let Some(upload) = uploads.entries.get_mut(&key) {
                upload.pins = upload.pins.saturating_sub(1);
            }
        }
    }

    /// Advance the upload epoch after a rasterization and evict idle or over-budget uploads.
    fn end_rasterization(&self) {
        let mut uploads = self.uploads.borrow_mut();
        uploads.epoch += 1;
        let epoch = uploads.epoch;
        if uploads.bytes > UPLOAD_CACHE_BUDGET {
            // Oldest first until under budget.
            let mut candidates: Vec<(u64, UploadKey)> = uploads
                .entries
                .iter()
                .filter(|(_, upload)| upload.pins == 0)
                .map(|(key, upload)| (upload.last_used, *key))
                .collect();
            candidates.sort_unstable_by_key(|(last_used, _)| *last_used);
            let mut excess = uploads.bytes - UPLOAD_CACHE_BUDGET;
            let mut doomed = Vec::new();
            for (_, key) in candidates {
                if excess == 0 {
                    break;
                }
                let bytes = uploads.entries[&key].bytes;
                excess = excess.saturating_sub(bytes);
                doomed.push(key);
            }
            self.evict(&mut uploads, |key, _| doomed.contains(key));
        }
        // Scan for idle uploads only occasionally; the cache is small and this is cheap anyway.
        if epoch % 64 == 0 {
            self.evict(&mut uploads, |_, upload| {
                epoch.saturating_sub(upload.last_used) > UPLOAD_CACHE_MAX_IDLE
            });
        }
    }

    /// Destroy the unpinned uploads `doomed` selects.
    fn evict(&self, uploads: &mut ImageUploads, doomed: impl Fn(&UploadKey, &Upload) -> bool) {
        let keys: Vec<UploadKey> = uploads
            .entries
            .iter()
            .filter(|(key, upload)| upload.pins == 0 && doomed(key, upload))
            .map(|(key, _)| *key)
            .collect();
        if keys.is_empty() {
            return;
        }
        self.make_current();
        let (mut renderer, mut resources) =
            (self.renderer.borrow_mut(), self.resources.borrow_mut());
        for key in keys {
            if let Some(upload) = uploads.entries.remove(&key) {
                uploads.bytes -= upload.bytes;
                renderer.destroy_image(&mut resources, upload.id);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum State {
    /// The scene holds commands that have not been rasterized yet.
    Drawing,
    /// The scene was rasterized; the retained texture holds the canvas content.
    Rendered,
}

/// Per-canvas GL resources: a swap chain on the thread's context and the retained frame.
struct Gpu {
    thread: Rc<GpuThread>,
    image_id: ExternalImageId,
    /// Copy of the last rendered frame, drawn back when content must be retained.
    retained: glow::NativeTexture,
}

impl Gpu {
    fn new(size: Size2D<u32>) -> Result<Self, String> {
        let thread = GpuThread::get().ok_or("GPU canvases are unavailable on this thread")?;
        thread.make_current();
        let image_id = thread
            .shared
            .external_image_id_manager
            .clone()
            .next_id(WebRenderImageHandlerType::Canvas2d);
        if let Err(error) = thread.shared.swap_chains.create_detached_swap_chain(
            image_id,
            size.cast(),
            &thread.device,
            &mut thread.context.borrow_mut(),
            SurfaceAccess::GPUOnly,
        ) {
            thread
                .shared
                .external_image_id_manager
                .clone()
                .remove(&image_id);
            return Err(format!("swap chain: {error:?}"));
        }
        let glow = thread.gl.glow();
        let retained = unsafe {
            let texture = glow.create_texture()?;
            glow.bind_texture(glow::TEXTURE_2D, Some(texture));
            glow.tex_storage_2d(
                glow::TEXTURE_2D,
                1,
                glow::RGBA8,
                size.width as i32,
                size.height as i32,
            );
            glow.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::NEAREST as i32,
            );
            glow.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::NEAREST as i32,
            );
            glow.bind_texture(glow::TEXTURE_2D, None);
            texture
        };
        Ok(Self {
            thread,
            image_id,
            retained,
        })
    }

    /// Rasterize `scene` into this canvas's back buffer, keep a GPU copy of the result, and
    /// publish the buffer to WebRender.
    fn render(&mut self, scene: &Scene, size: Size2D<u32>, clear: bool, draw_retained: bool) {
        let thread = &self.thread;
        thread.make_current();
        let Some(swap_chain) = thread.shared.swap_chains.get(self.image_id) else {
            return;
        };
        let surface_texture = match swap_chain
            .take_surface_texture(&thread.device, &mut thread.context.borrow_mut())
        {
            Ok(surface_texture) => surface_texture,
            Err(error) => {
                warn!("GPU canvas: could not take the back buffer: {error:?}");
                return;
            },
        };
        let target = thread.device.surface_gl_texture_target();
        if let Some(texture) = thread.device.surface_texture_object(&surface_texture) {
            thread
                .gl
                .set_drawing_buffer_size(size.width as i32, size.height as i32);
            let mut bindings = WebGlTextureBindings::new();
            if draw_retained {
                bindings.insert(RETAINED_TEXTURE, self.retained);
            }
            let render_size = RenderSize {
                width: size.width,
                height: size.height,
            };
            if let Err(error) = thread.renderer.borrow_mut().render_to_texture(
                scene,
                &mut thread.resources.borrow_mut(),
                &render_size,
                &bindings,
                clear,
                texture,
                target,
            ) {
                warn!("GPU canvas: render failed: {error:?}");
            }
            let glow = thread.gl.glow();
            unsafe {
                // Retain this frame on the GPU before the buffer moves to WebRender.
                glow.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(thread.copy_framebuffer));
                glow.framebuffer_texture_2d(
                    glow::READ_FRAMEBUFFER,
                    glow::COLOR_ATTACHMENT0,
                    target,
                    Some(texture),
                    0,
                );
                glow.bind_texture(glow::TEXTURE_2D, Some(self.retained));
                glow.copy_tex_sub_image_2d(
                    glow::TEXTURE_2D,
                    0,
                    0,
                    0,
                    0,
                    0,
                    size.width as i32,
                    size.height as i32,
                );
                glow.bind_texture(glow::TEXTURE_2D, None);
                glow.framebuffer_texture_2d(
                    glow::READ_FRAMEBUFFER,
                    glow::COLOR_ATTACHMENT0,
                    target,
                    None,
                    0,
                );
                glow.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
                glow.flush();
            }
        }
        let mut context = thread.context.borrow_mut();
        if let Err(error) =
            swap_chain.recycle_surface_texture(&thread.device, &mut context, surface_texture)
        {
            warn!("GPU canvas: could not return the back buffer: {error:?}");
        }
        if let Err(error) = swap_chain.swap_buffers(
            &thread.device,
            &mut context,
            surfman::chains::PreserveBuffer::No,
        ) {
            warn!("GPU canvas: swap failed: {error:?}");
        }
    }

    /// Read the retained copy back as top-down premultiplied RGBA.
    fn read_retained(&self, size: Size2D<u32>) -> Vec<u8> {
        self.thread.make_current();
        let glow = self.thread.gl.glow();
        let (width, height) = (size.width as usize, size.height as usize);
        let mut pixels = vec![0_u8; width * height * 4];
        unsafe {
            let framebuffer = glow.create_framebuffer().ok();
            glow.bind_framebuffer(glow::READ_FRAMEBUFFER, framebuffer);
            glow.framebuffer_texture_2d(
                glow::READ_FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(self.retained),
                0,
            );
            glow.read_pixels(
                0,
                0,
                width as i32,
                height as i32,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(Some(&mut pixels)),
            );
            glow.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
            if let Some(framebuffer) = framebuffer {
                glow.delete_framebuffer(framebuffer);
            }
        }
        // GL rows are bottom-up.
        let stride = width * 4;
        let mut flipped = vec![0_u8; pixels.len()];
        for (row, chunk) in pixels.chunks_exact(stride).enumerate() {
            let dest = (height - 1 - row) * stride;
            flipped[dest..dest + stride].copy_from_slice(chunk);
        }
        flipped
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        let thread = &self.thread;
        thread.make_current();
        unsafe { thread.gl.glow().delete_texture(self.retained) };
        let _ = thread.shared.swap_chains.destroy(
            self.image_id,
            &thread.device,
            &mut thread.context.borrow_mut(),
        );
        thread
            .shared
            .external_image_id_manager
            .clone()
            .remove(&self.image_id);
    }
}

pub(crate) struct VelloGpuDrawTarget {
    scene: Scene,
    gpu: Gpu,
    size: Size2D<u32>,
    clips: Vec<(Path, kurbo::Affine)>,
    state: State,
    /// The next rasterization starts from a transparent target (nothing to retain).
    clear_next: bool,
    /// The scene's first command draws the retained texture back.
    draws_retained: bool,
    /// Cached uploads the pending scene samples; unpinned after the next rasterization.
    pinned_uploads: Vec<UploadKey>,
    /// Open shadow (filter) layers: draws inside them do not target the canvas directly.
    shadow_depth: u32,
}

impl VelloGpuDrawTarget {
    fn with_composition(
        &mut self,
        composition_operation: CompositionOrBlending,
        f: impl FnOnce(&mut Self),
    ) {
        if composition_operation == CompositionOrBlending::Composition(CompositionStyle::SourceOver)
        {
            f(self);
            return;
        }
        // `lighter` is plain additive blending. Outside clip layers it can blend straight into
        // the canvas instead of through a layer per draw (two GPU passes each).
        if composition_operation == CompositionOrBlending::Composition(CompositionStyle::Lighter) &&
            self.clips.is_empty() &&
            self.shadow_depth == 0
        {
            self.scene.set_additive(true);
            f(self);
            self.scene.set_additive(false);
            return;
        }
        self.scene.push_blend_layer(composition_operation.convert());
        f(self);
        self.scene.pop_layer();
    }

    fn ignore_clips(&mut self, f: impl FnOnce(&mut Self)) {
        for _ in &self.clips {
            self.scene.pop_layer();
        }
        f(self);
        for (path, affine) in &self.clips {
            self.scene.set_transform(*affine);
            self.scene.push_clip_layer(&path.0);
        }
    }

    /// Before new commands after a rasterization, draw the retained content back as background.
    fn ensure_drawing(&mut self) {
        if self.state == State::Drawing {
            return;
        }
        let size = self.size;
        self.ignore_clips(|self_| {
            self_.scene.set_transform(kurbo::Affine::IDENTITY);
            // Retained texel rows are bottom-up; flip them into scene space.
            let flip = kurbo::Affine::new([1.0, 0.0, 0.0, -1.0, 0.0, f64::from(size.height)]);
            self_.scene.draw_texture_rects(
                RETAINED_TEXTURE,
                peniko::ImageQuality::Low,
                [SampleRect {
                    source_region: vello_hybrid::RectU16::new(
                        0,
                        0,
                        size.width as u16,
                        size.height as u16,
                    ),
                    transform: flip,
                }],
            );
        });
        self.draws_retained = true;
        self.clear_next = true;
        self.state = State::Drawing;
    }

    /// Rasterize pending commands, if any.
    fn flush(&mut self) {
        if self.state != State::Drawing {
            return;
        }
        let size = self.size;
        let (clear, draw_retained) = (self.clear_next, self.draws_retained);
        self.ignore_clips(|self_| {
            self_.gpu.render(&self_.scene, size, clear, draw_retained);
            self_.scene.reset();
        });
        self.unpin_uploads();
        self.gpu.thread.end_rasterization();
        self.state = State::Rendered;
        self.clear_next = true;
        self.draws_retained = false;
    }

    fn is_viewport_cleared(&mut self, rect: &Rect<f32>, transform: Transform2D<f64>) -> bool {
        let transformed_rect = transform.outer_transformed_rect(&rect.cast());
        if transformed_rect.is_empty() {
            return false;
        }
        let viewport: Rect<f64> = Rect::from_size(self.size.cast());
        let Some(clip) = self.clips.iter().try_fold(viewport, |acc, (path, affine)| {
            let mut bez = path.0.clone();
            bez.apply_affine(*affine);
            acc.intersection(&bez.bounding_box().into())
        }) else {
            return false;
        };
        transformed_rect.cast().contains_rect(&viewport) && clip.contains_rect(&viewport)
    }

    /// A GPU canvas of `size`, or `None` (logged) when its GPU resources cannot be created;
    /// callers fall back to the CPU backend.
    pub(crate) fn try_new(size: Size2D<u32>) -> Option<Self> {
        let size = size.max(Size2D::new(1, 1));
        let gpu = match Gpu::new(size) {
            Ok(gpu) => gpu,
            Err(error) => {
                warn!("GPU canvas: using the CPU backend for a {size:?} canvas: {error}");
                return None;
            },
        };
        let mut target = Self {
            scene: Scene::new(size.width as u16, size.height as u16),
            gpu,
            size,
            clips: Vec::new(),
            state: State::Drawing,
            clear_next: true,
            draws_retained: false,
            pinned_uploads: Vec::new(),
            shadow_depth: 0,
        };
        // Publish a transparent first frame so WebRender always has a surface to sample.
        target.flush();
        Some(target)
    }

    /// The atlas image holding `pixmap`, pinned until the pending scene is rasterized. Returns
    /// `None` for images the atlas cannot hold.
    fn upload_image(&mut self, pixmap: &vello_cpu::Pixmap) -> Option<ImageId> {
        if pixmap.width() > MAX_ATLAS_IMAGE_SIDE || pixmap.height() > MAX_ATLAS_IMAGE_SIDE {
            warn_once(
                &WARNED_IMAGE,
                "images larger than 4096 px are not drawn yet",
            );
            return None;
        }
        let (key, id) = self.gpu.thread.pin_upload(pixmap)?;
        self.pinned_uploads.push(key);
        Some(id)
    }

    fn unpin_uploads(&mut self) {
        if !self.pinned_uploads.is_empty() {
            self.gpu.thread.unpin_uploads(self.pinned_uploads.drain(..));
        }
    }

    /// The `vello_cpu` paint for `style`, with any pattern image moved into the atlas.
    fn paint(&mut self, style: FillOrStrokeStyle, alpha: f64) -> Option<vello_cpu::PaintType> {
        match paint(style, alpha) {
            vello_cpu::PaintType::Image(mut image) => {
                if let vello_cpu::ImageSource::Pixmap(pixmap) = &image.image {
                    let id = self.upload_image(pixmap)?;
                    image.image = vello_cpu::ImageSource::opaque_id(id);
                }
                Some(vello_cpu::PaintType::Image(image))
            },
            other => Some(other),
        }
    }

    /// Block until the GPU has executed every submitted command (benchmarks).
    #[cfg(test)]
    pub(crate) fn finish_gpu(&self) {
        self.gpu.thread.make_current();
        unsafe { self.gpu.thread.gl.glow().finish() };
    }

    /// The external image id WebRender samples this canvas through.
    #[cfg(test)]
    pub(crate) fn external_image_id(&self) -> ExternalImageId {
        self.gpu.image_id
    }

    fn external_image_data(&self) -> SerializableImageData {
        SerializableImageData::External(ExternalImageData {
            id: self.gpu.image_id,
            channel_index: 0,
            image_type: ExternalImageType::TextureHandle(ImageBufferKind::Texture2D),
            normalized_uvs: false,
        })
    }
}

impl Drop for VelloGpuDrawTarget {
    fn drop(&mut self) {
        self.unpin_uploads();
    }
}

impl GenericDrawTarget for VelloGpuDrawTarget {
    type SourceSurface = Arc<vello_cpu::Pixmap>;

    fn new(size: Size2D<u32>) -> Self {
        Self::try_new(size).expect("GPU canvas init failed")
    }

    fn canvas_store_sizes(
        &self,
        _ops: &mut malloc_size_of::MallocSizeOfOps,
    ) -> Option<Vec<CanvasStoreSizesPerType>> {
        // The retained copy plus the swap chain's buffers: the back buffer, the front buffer
        // WebRender samples, and at most one recycled buffer.
        let frame = (self.size.width * self.size.height * 4) as usize;
        Some(vec![
            CanvasStoreSizesPerType {
                name: "gpu-retained-texture",
                size: frame,
                kind: ReportKind::ExplicitNonHeapSize,
            },
            CanvasStoreSizesPerType {
                name: "gpu-swap-chain",
                size: frame * 3,
                kind: ReportKind::ExplicitNonHeapSize,
            },
        ])
    }

    fn clear_rect(&mut self, rect: &Rect<f32>, transform: Transform2D<f64>) {
        if self.is_viewport_cleared(rect, transform) {
            self.scene.reset();
            self.unpin_uploads();
            self.clips.clear();
            self.state = State::Drawing;
            self.clear_next = true;
            self.draws_retained = false;
            return;
        }
        self.ensure_drawing();
        let rect: kurbo::Rect = rect.cast().into();
        let mut clip_path = rect.to_path(0.1);
        clip_path.apply_affine(transform.cast().into());
        self.scene.push_layer(
            Some(&clip_path.to_path(0.1)),
            Some(peniko::Compose::Clear.into()),
            None,
            None,
            None,
        );
        self.scene.pop_layer();
    }

    fn copy_surface(
        &mut self,
        surface: Self::SourceSurface,
        source: Rect<i32>,
        destination: Point2D<i32>,
    ) {
        // Mirrors the CPU backend, including its unclipped source-over fill.
        self.ensure_drawing();
        let Some(id) = self.upload_image(&surface) else {
            return;
        };
        let destination: kurbo::Point = destination.cast::<f64>().into();
        let rect = kurbo::Rect::from_origin_size(destination, source.size.cast());
        self.scene.set_transform(kurbo::Affine::IDENTITY);
        self.ignore_clips(|self_| {
            self_.scene.set_paint(vello_cpu::Image {
                image: vello_cpu::ImageSource::opaque_id(id),
                sampler: peniko::ImageSampler {
                    x_extend: peniko::Extend::Pad,
                    y_extend: peniko::Extend::Pad,
                    quality: peniko::ImageQuality::Low,
                    alpha: 1.0,
                },
            });
            self_.scene.fill_rect(&rect);
        });
    }

    fn create_similar_draw_target(&self, size: &Size2D<i32>) -> Self {
        Self::new(size.cast())
    }

    fn draw_surface(
        &mut self,
        surface: Self::SourceSurface,
        dest: Rect<f64>,
        source: Rect<f64>,
        filter: Filter,
        composition_options: CompositionOptions,
        transform: Transform2D<f64>,
    ) {
        self.ensure_drawing();
        let scale_up = dest.size.width > source.size.width || dest.size.height > source.size.height;
        let Some(id) = self.upload_image(&surface) else {
            return;
        };
        // Global alpha as a multiply tint, so the cached upload is reused at any alpha. Quantized
        // to 8 bits like the CPU backend's `multiply_alpha`.
        let alpha = (composition_options.alpha * 255.0) as u8;
        let tint = (alpha != u8::MAX).then(|| vello_hybrid::Tint {
            color: peniko::Color::new([1.0, 1.0, 1.0, f32::from(alpha) / 255.0]),
            mode: vello_hybrid::TintMode::Multiply,
        });
        self.with_composition(composition_options.composition_operation, move |self_| {
            self_.scene.set_tint(tint);
            self_.scene.set_transform(transform.cast().into());
            self_.scene.set_paint(vello_cpu::Image {
                image: vello_cpu::ImageSource::opaque_id(id),
                sampler: peniko::ImageSampler {
                    x_extend: peniko::Extend::Pad,
                    y_extend: peniko::Extend::Pad,
                    // Bicubic/bilinear only when scaling up, as in the CPU backend.
                    quality: if scale_up {
                        filter.convert()
                    } else {
                        peniko::ImageQuality::Low
                    },
                    alpha: 1.0,
                },
            });
            self_.scene.set_paint_transform(
                kurbo::Affine::translate((dest.origin.x, dest.origin.y)).pre_scale_non_uniform(
                    dest.size.width / source.size.width,
                    dest.size.height / source.size.height,
                ),
            );
            self_.scene.fill_rect(&dest.into());
            self_.scene.reset_paint_transform();
            self_.scene.reset_tint();
        })
    }

    fn push_shadow(&mut self, shadow: &ShadowOptions) -> bool {
        self.ensure_drawing();
        self.scene.set_transform(kurbo::Affine::IDENTITY);
        self.scene
            .push_layer(None, None, None, None, Some(shadow_filter(shadow)));
        self.shadow_depth += 1;
        true
    }

    fn pop_shadow(&mut self) {
        self.scene.pop_layer();
        self.shadow_depth -= 1;
    }

    fn draw_surface_with_shadow(
        &self,
        _surface: Self::SourceSurface,
        _dest: &Point2D<f32>,
        _shadow_options: ShadowOptions,
        _composition_options: CompositionOptions,
    ) {
        warn!("no support for drawing shadows");
    }

    fn fill(
        &mut self,
        path: &Path,
        fill_rule: FillRule,
        style: FillOrStrokeStyle,
        composition_options: CompositionOptions,
        transform: Transform2D<f64>,
    ) {
        self.ensure_drawing();
        let Some(paint) = self.paint(style, composition_options.alpha) else {
            return;
        };
        self.with_composition(composition_options.composition_operation, |self_| {
            self_.scene.set_transform(transform.cast().into());
            self_.scene.set_fill_rule(fill_rule.convert());
            self_.scene.set_paint(paint);
            self_.scene.fill_path(&path.0);
        });
        self.scene.set_fill_rule(peniko::Fill::NonZero);
    }

    fn fill_text(
        &mut self,
        text_runs: Vec<TextRun>,
        style: FillOrStrokeStyle,
        composition_options: CompositionOptions,
        transform: Transform2D<f64>,
    ) {
        self.ensure_drawing();
        let Some(paint) = self.paint(style, composition_options.alpha) else {
            return;
        };
        self.scene.set_paint(paint);
        self.scene.set_transform(transform.cast().into());
        let thread = self.gpu.thread.clone();
        self.with_composition(composition_options.composition_operation, |self_| {
            for text_run in &text_runs {
                with_font(text_run, |font| {
                    self_
                        .scene
                        .glyph_run(&mut thread.resources.borrow_mut(), font)
                        .font_size(text_run.pt_size)
                        .fill_glyphs(glyphs(text_run));
                });
            }
        })
    }

    fn fill_rect(
        &mut self,
        rect: &Rect<f32>,
        style: FillOrStrokeStyle,
        composition_options: CompositionOptions,
        transform: Transform2D<f64>,
    ) {
        self.ensure_drawing();
        let Some(paint) = self.paint(style, composition_options.alpha) else {
            return;
        };
        self.with_composition(composition_options.composition_operation, |self_| {
            self_.scene.set_transform(transform.cast().into());
            self_.scene.set_paint(paint);
            self_.scene.fill_rect(&rect.cast().into());
        })
    }

    fn get_size(&self) -> Size2D<i32> {
        self.size.cast()
    }

    fn pop_clip(&mut self) {
        if self.clips.pop().is_some() {
            self.scene.pop_layer();
        }
    }

    fn push_clip(&mut self, path: &Path, fill_rule: FillRule, transform: Transform2D<f64>) {
        let affine = transform.cast().into();
        self.scene.set_transform(affine);
        self.scene.set_fill_rule(fill_rule.convert());
        self.scene.push_clip_layer(&path.0);
        self.clips.push((path.clone(), affine));
        self.scene.set_fill_rule(peniko::Fill::NonZero);
    }

    fn push_clip_rect(&mut self, rect: &Rect<i32>) {
        let mut path = Path::new();
        let rect = rect.cast();
        path.rect(
            rect.origin.x,
            rect.origin.y,
            rect.size.width,
            rect.size.height,
        );
        self.push_clip(&path, FillRule::Nonzero, Transform2D::identity());
    }

    fn stroke(
        &mut self,
        path: &Path,
        style: FillOrStrokeStyle,
        line_options: LineOptions,
        composition_options: CompositionOptions,
        transform: Transform2D<f64>,
    ) {
        self.ensure_drawing();
        let Some(paint) = self.paint(style, composition_options.alpha) else {
            return;
        };
        self.with_composition(composition_options.composition_operation, |self_| {
            self_.scene.set_transform(transform.cast().into());
            self_.scene.set_paint(paint);
            self_.scene.set_stroke(line_options.convert());
            self_.scene.stroke_path(&path.0);
        })
    }

    fn stroke_text(
        &mut self,
        text_runs: Vec<TextRun>,
        style: FillOrStrokeStyle,
        line_options: LineOptions,
        composition_options: CompositionOptions,
        transform: Transform2D<f64>,
    ) {
        self.ensure_drawing();
        let Some(paint) = self.paint(style, composition_options.alpha) else {
            return;
        };
        self.scene.set_paint(paint);
        self.scene.set_stroke(line_options.convert());
        self.scene.set_transform(transform.cast().into());
        let thread = self.gpu.thread.clone();
        self.with_composition(composition_options.composition_operation, |self_| {
            for text_run in &text_runs {
                with_font(text_run, |font| {
                    self_
                        .scene
                        .glyph_run(&mut thread.resources.borrow_mut(), font)
                        .font_size(text_run.pt_size)
                        .stroke_glyphs(glyphs(text_run));
                });
            }
        })
    }

    fn stroke_rect(
        &mut self,
        rect: &Rect<f32>,
        style: FillOrStrokeStyle,
        line_options: LineOptions,
        composition_options: CompositionOptions,
        transform: Transform2D<f64>,
    ) {
        self.ensure_drawing();
        let Some(paint) = self.paint(style, composition_options.alpha) else {
            return;
        };
        self.with_composition(composition_options.composition_operation, |self_| {
            self_.scene.set_transform(transform.cast().into());
            self_.scene.set_paint(paint);
            self_.scene.set_stroke(line_options.convert());
            self_.scene.stroke_rect(&rect.cast().into());
        })
    }

    fn image_descriptor_and_serializable_data(
        &mut self,
    ) -> (ImageDescriptor, SerializableImageData) {
        self.flush();
        let descriptor = ImageDescriptor {
            format: ImageFormat::BGRA8,
            size: self.size.cast().cast_unit(),
            stride: None,
            offset: 0,
            flags: ImageDescriptorFlags::empty(),
        };
        (descriptor, self.external_image_data())
    }

    fn snapshot(&mut self) -> Snapshot {
        self.flush();
        Snapshot::from_vec(
            self.size.cast(),
            SnapshotPixelFormat::RGBA,
            SnapshotAlphaMode::Transparent {
                premultiplied: true,
            },
            self.gpu.read_retained(self.size),
        )
    }

    fn surface(&mut self) -> Self::SourceSurface {
        snapshot_as_pixmap(self.snapshot())
    }

    fn create_source_surface_from_data(&self, data: Snapshot) -> Option<Self::SourceSurface> {
        Some(snapshot_as_pixmap(data))
    }
}
