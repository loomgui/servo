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
//! for the next rasterization and released right after it.

#![allow(
    unsafe_code,
    reason = "GL calls on the canvas's own context, which this module keeps current"
)]

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
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
use crate::vello_cpu_backend::{paint, snapshot_as_pixmap};

/// Texture binding id used to draw retained content back into a scene.
const RETAINED_TEXTURE: TextureId = TextureId(1);
/// Largest image side the renderer's atlas pages hold (`AtlasConfig::default().atlas_size`).
const MAX_ATLAS_IMAGE_SIDE: u16 = 4096;

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
/// gradient ramps). Canvases own only a swap chain on this context and a retained texture. Before
/// a canvas renders, its swap chain takes the context's surface attachment; a 1×1 anchor swap
/// chain holds the attachment while no canvas does.
struct GpuThread {
    device: Device,
    /// Never dropped: this lives until the canvas thread exits, and tearing down EGL objects from
    /// thread-exit callbacks can deadlock ANGLE.
    context: std::mem::ManuallyDrop<RefCell<Context>>,
    gl: WebGl2RenderingContext,
    renderer: RefCell<WebGlRenderer>,
    resources: RefCell<Resources>,
    shared: &'static GpuCanvasShared,
    anchor: ExternalImageId,
    /// The swap chain whose back buffer is bound to the context.
    attached: Cell<ExternalImageId>,
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
        let anchor = shared
            .external_image_id_manager
            .clone()
            .next_id(WebRenderImageHandlerType::Canvas2d);
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
                .map_err(|e| format!("make current: {e:?}"))?;
            shared
                .swap_chains
                .create_attached_swap_chain(anchor, &device, &mut context, SurfaceAccess::GPUOnly)
                .map_err(|e| format!("swap chain: {e:?}"))
        })();
        if let Err(error) = setup {
            let _ = device.destroy_context(&mut context);
            shared.external_image_id_manager.clone().remove(&anchor);
            return Err(error);
        }
        let glow = unsafe {
            glow::Context::from_loader_function(|symbol| device.get_proc_address(&context, symbol))
        };
        let gl = WebGl2RenderingContext::from_glow(glow, 1, 1);
        let (renderer, resources) = WebGlRenderer::new(gl.clone());
        Ok(Self {
            device,
            context: std::mem::ManuallyDrop::new(RefCell::new(context)),
            gl,
            renderer: RefCell::new(renderer),
            resources: RefCell::new(resources),
            shared,
            anchor,
            attached: Cell::new(anchor),
        })
    }

    fn make_current(&self) {
        if let Err(error) = self.device.make_context_current(&self.context.borrow()) {
            warn!("GPU canvas: make current failed: {error:?}");
        }
    }

    /// Bind the back buffer of `id`'s swap chain to the context.
    fn attach(&self, id: ExternalImageId) -> bool {
        let current = self.attached.get();
        if current == id {
            return true;
        }
        let (Some(chain), Some(current_chain)) = (
            self.shared.swap_chains.get(id),
            self.shared.swap_chains.get(current),
        ) else {
            warn!("GPU canvas: missing swap chain for {id:?} or {current:?}");
            return false;
        };
        match chain.take_attachment_from(
            &self.device,
            &mut self.context.borrow_mut(),
            &current_chain,
        ) {
            Ok(()) => {
                self.attached.set(id);
                true
            },
            Err(error) => {
                warn!("GPU canvas: could not attach {id:?}: {error:?}");
                false
            },
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
        if !thread.attach(self.image_id) {
            return;
        }
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
        if let Err(error) = thread.renderer.borrow_mut().render_with_clear(
            scene,
            &mut thread.resources.borrow_mut(),
            &render_size,
            &bindings,
            clear,
        ) {
            warn!("GPU canvas: render failed: {error:?}");
        }
        let glow = thread.gl.glow();
        unsafe {
            // Retain this frame on the GPU before the surface moves to WebRender.
            glow.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
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
            glow.flush();
        }
        if let Some(swap_chain) = thread.shared.swap_chains.get(self.image_id) &&
            let Err(error) = swap_chain.swap_buffers(
                &thread.device,
                &mut thread.context.borrow_mut(),
                surfman::chains::PreserveBuffer::No,
            )
        {
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
        if thread.attached.get() == self.image_id {
            thread.attach(thread.anchor);
        }
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
    /// Atlas images the pending scene samples; released after the next rasterization.
    uploaded_images: Vec<ImageId>,
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
        self.release_images();
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

    /// Upload `pixmap` into the atlas for the pending scene. Returns `None` for images the atlas
    /// cannot hold.
    fn upload_image(&mut self, pixmap: &vello_cpu::Pixmap) -> Option<ImageId> {
        if pixmap.width() > MAX_ATLAS_IMAGE_SIDE || pixmap.height() > MAX_ATLAS_IMAGE_SIDE {
            warn_once(&WARNED_IMAGE, "images larger than 4096 px are not drawn yet");
            return None;
        }
        let thread = &self.gpu.thread;
        thread.make_current();
        let id = thread
            .renderer
            .borrow_mut()
            .upload_image(&mut thread.resources.borrow_mut(), pixmap);
        self.uploaded_images.push(id);
        Some(id)
    }

    fn release_images(&mut self) {
        if self.uploaded_images.is_empty() {
            return;
        }
        let thread = &self.gpu.thread;
        thread.make_current();
        let (mut renderer, mut resources) =
            (thread.renderer.borrow_mut(), thread.resources.borrow_mut());
        for id in self.uploaded_images.drain(..) {
            renderer.destroy_image(&mut resources, id);
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

impl GenericDrawTarget for VelloGpuDrawTarget {
    type SourceSurface = Arc<vello_cpu::Pixmap>;

    fn new(size: Size2D<u32>) -> Self {
        let size = size.max(Size2D::new(1, 1));
        let gpu = Gpu::new(size).unwrap_or_else(|error| panic!("GPU canvas init failed: {error}"));
        let mut target = Self {
            scene: Scene::new(size.width as u16, size.height as u16),
            gpu,
            size,
            clips: Vec::new(),
            state: State::Drawing,
            clear_next: true,
            draws_retained: false,
            uploaded_images: Vec::new(),
        };
        // Publish a transparent first frame so WebRender always has a surface to sample.
        target.flush();
        target
    }

    fn canvas_store_sizes(
        &self,
        _ops: &mut malloc_size_of::MallocSizeOfOps,
    ) -> Option<Vec<CanvasStoreSizesPerType>> {
        Some(vec![CanvasStoreSizesPerType {
            name: "gpu-backing-texture",
            size: (self.size.width * self.size.height * 4) as usize,
            kind: ReportKind::ExplicitNonHeapSize,
        }])
    }

    fn clear_rect(&mut self, rect: &Rect<f32>, transform: Transform2D<f64>) {
        if self.is_viewport_cleared(rect, transform) {
            self.scene.reset();
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
        mut surface: Self::SourceSurface,
        dest: Rect<f64>,
        source: Rect<f64>,
        filter: Filter,
        composition_options: CompositionOptions,
        transform: Transform2D<f64>,
    ) {
        self.ensure_drawing();
        let scale_up = dest.size.width > source.size.width || dest.size.height > source.size.height;
        if composition_options.alpha != 1.0 {
            Arc::make_mut(&mut surface).multiply_alpha((composition_options.alpha * 255.0) as u8);
        }
        let Some(id) = self.upload_image(&surface) else {
            return;
        };
        self.with_composition(composition_options.composition_operation, move |self_| {
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
        })
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
        self.scene
            .set_paint(paint);
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
        self.scene
            .set_paint(paint);
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
