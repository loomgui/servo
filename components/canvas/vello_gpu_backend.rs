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
//! The retained texture is also how one GPU canvas is drawn into another (`drawImage` with a
//! canvas source): the destination's scene samples the source's frame directly, so no pixels
//! are read back or uploaded. A frame that another canvas still has to draw is never
//! overwritten; the source keeps its next frame in a new texture instead.
//!
//! Recording a path means turning it into strips, which is nearly all the canvas thread does
//! for a path-heavy canvas; for text, preparing the glyphs comes first. Path fills and strokes
//! and source-over text runs are therefore not recorded as they arrive: consecutive ones are
//! collected and handed to a few worker threads in small batches, which generate their strips
//! while the page goes on drawing (the canvas thread helps whenever it has no command to
//! handle), and they are recorded in order when anything else has to happen to the scene.
//!
//! Composite operations that are fixed-function blends (`lighter`, `destination-out`, `xor`,
//! `screen`, `source-atop`, `destination-over`) blend each draw straight into its target instead
//! of through a blend layer. Every blend layer costs rounds of render passes, and on a tiling
//! GPU (Apple's) each render pass loads and stores whole intermediate textures.
//!
//! On desktop OpenGL (CGL on macOS), data textures are uploaded and canvas pixels read back
//! through pixel buffers: macOS's OpenGL otherwise makes the CPU wait for the GPU on every
//! render that rewrites a texture still in use, and untiles textures on the CPU for readback.
//!
//! If the thread's GL context is lost, or the renderer fails in a way it reports by panicking,
//! the thread stops using the GPU for good: GPU canvases draw nothing from then on and
//! [`take_context_loss`] tells the canvas thread to rebuild them on the CPU backend. Their
//! content is gone; pages that redraw every frame recover on their next frame.
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
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

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
    Context, ContextAttributeFlags, ContextAttributes, Device, GLApi, GLVersion, SurfaceAccess,
    SurfaceType,
};
use vello_cpu::{kurbo, peniko};
use vello_hybrid::{
    DirectBlend, GlyphRunCaches, GlyphRunJob, GlyphRunStrips, ImageId, PathStripGenerator,
    PathStripSettings, PathStrips, RenderSize, Resources, SampleRect, Scene, TextureId,
    WebGl2RenderingContext, WebGlRenderer, WebGlTextureBindings,
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
/// Texture binding ids from here on are the other canvases' frames a scene draws.
const FIRST_SOURCE_TEXTURE: u64 = 2;
/// Largest image side the renderer's atlas pages hold (`AtlasConfig::default().atlas_size`).
const MAX_ATLAS_IMAGE_SIDE: u16 = 4096;
/// Atlas bytes cached uploads may hold before unpinned ones are evicted, oldest first.
const UPLOAD_CACHE_BUDGET: usize = 64 << 20;
/// Rasterizations (on any canvas of the thread) after which an unused upload is evicted.
const UPLOAD_CACHE_MAX_IDLE: u64 = 600;

/// Consecutive path draws from which generating their strips on the worker threads beats
/// doing it on the canvas thread. A path takes around 20 µs; waking the workers, about as long.
const MIN_PARALLEL_PATHS: usize = 16;
/// How many paths a thread takes from a batch at a time.
const PATHS_PER_CLAIM: usize = 1;
/// Glyphs of a text run prepared together, on one thread.
const GLYPHS_PER_JOB: usize = 4;

/// What generating the strips of a pending path draw needs. Shared with the worker threads.
struct PathJob {
    path: kurbo::BezPath,
    /// The stroke of a stroked path; a filled path has none.
    stroke: Option<kurbo::Stroke>,
    fill_rule: peniko::Fill,
    settings: PathStripSettings,
}

/// The strips of a pending draw to generate: a path's, or every glyph outline's of a text run.
enum StripJob {
    Path(PathJob),
    Text(GlyphRunJob),
}

impl StripJob {
    fn run(&self) -> JobStrips {
        match self {
            Self::Path(job) => JobStrips::Path(path_strips(job)),
            Self::Text(job) => JobStrips::Text(Box::new(
                GLYPH_RUN_CACHES.with(|caches| job.strips(&mut caches.borrow_mut())),
            )),
        }
    }
}

enum JobStrips {
    Path(PathStrips),
    Text(Box<GlyphRunStrips>),
}

/// How a pending draw goes into the scene once its strips exist.
enum PendingDraw {
    /// A canvas fill or stroke.
    Canvas {
        transform: kurbo::Affine,
        paint: vello_cpu::PaintType,
        composite: Composite,
    },
    /// A text run of `fillText` or `strokeText`, drawn source-over with `paint`.
    Text(vello_cpu::PaintType),
}

/// Paths and text a page drew that are not in the scene yet: canvas fills and strokes, and
/// text runs. They go into the scene together and in order, before anything else does; see
/// [`VelloGpuDrawTarget::draw_pending_paths`].
///
/// Generating their strips (and, for text, preparing the glyphs first) is nearly all the work
/// of recording them. Every [`DISPATCH_PATHS`] of them are handed to the worker threads at once,
/// so their strips are generated while the page goes on drawing, as the CPU backend rasterizes
/// on its worker threads while the page draws. When they are recorded, this thread generates
/// what is left, alongside the workers.
#[derive(Default)]
struct PendingPaths {
    /// How to draw each pending path, in order.
    draws: Vec<PendingDraw>,
    /// The strips of the first pending paths, in order, in batches handed to the workers.
    dispatched: Vec<Arc<StripBatch>>,
    /// The draws after those.
    jobs: Vec<StripJob>,
}

/// Paths the worker threads take at once while the page draws.
const DISPATCH_PATHS: usize = 16;

impl PendingPaths {
    fn is_empty(&self) -> bool {
        self.draws.is_empty()
    }

    fn push(&mut self, thread: &GpuThread, job: StripJob, draw: PendingDraw) {
        self.jobs.push(job);
        self.draws.push(draw);
        if self.jobs.len() >= DISPATCH_PATHS &&
            let Some(workers) = thread.path_workers()
        {
            let jobs = std::mem::replace(&mut self.jobs, Vec::with_capacity(DISPATCH_PATHS));
            let batch = StripBatch::new(jobs);
            batch.start(workers);
            self.dispatched.push(batch);
        }
    }

    /// Generate strips of the pending draws on this thread while `idle` holds. Draws not handed
    /// to the workers yet go to them first.
    fn work_while(&mut self, thread: &GpuThread, idle: &dyn Fn() -> bool) {
        if !self.jobs.is_empty() {
            let jobs = std::mem::replace(&mut self.jobs, Vec::with_capacity(DISPATCH_PATHS));
            let batch = StripBatch::new(jobs);
            if let Some(workers) = thread.path_workers() {
                batch.start(workers);
            }
            self.dispatched.push(batch);
        }
        for batch in &self.dispatched {
            while idle() {
                if !batch.work_one() {
                    break;
                }
            }
        }
    }

    /// Forget every pending path: the canvas was cleared. Batches the workers are on finish
    /// and are dropped.
    fn clear(&mut self) {
        *self = Self::default();
    }

    /// Record every pending path into `scene`, in order.
    fn record(&mut self, scene: &mut Scene, thread: &GpuThread) {
        if self.is_empty() {
            return;
        }
        let mut batches = std::mem::take(&mut self.dispatched);
        if !self.jobs.is_empty() {
            let rest = StripBatch::new(std::mem::take(&mut self.jobs));
            if rest.jobs.len() >= MIN_PARALLEL_PATHS &&
                let Some(workers) = thread.path_workers()
            {
                rest.start(workers);
            }
            batches.push(rest);
        }
        // This thread works on every batch alongside the workers instead of waiting for them.
        for batch in &batches {
            batch.work();
        }
        let strips: Vec<JobStrips> = batches.iter().flat_map(|batch| batch.finish()).collect();
        for (draw, strips) in self.draws.drain(..).zip(&strips) {
            match (draw, strips) {
                (
                    PendingDraw::Canvas {
                        transform,
                        paint,
                        composite,
                    },
                    JobStrips::Path(strips),
                ) => composite.apply(scene, |scene| {
                    scene.set_transform(transform);
                    scene.set_paint(paint);
                    scene.draw_path_strips(strips);
                }),
                (PendingDraw::Text(paint), JobStrips::Text(strips)) => {
                    scene.set_paint(paint);
                    scene.draw_glyph_run_strips(&mut thread.resources.borrow_mut(), strips);
                },
                _ => unreachable!("every pending draw has strips of its kind"),
            }
        }
    }
}

/// Paths whose strips any number of threads generate together. Each claims a few paths at a
/// time until none are left, which keeps them balanced whatever the paths cost.
struct StripBatch {
    jobs: Vec<StripJob>,
    next: AtomicUsize,
    strips: Vec<Mutex<Option<JobStrips>>>,
    /// Paths done, and a wake-up for [`Self::finish`] when that reaches all of them.
    done: Mutex<usize>,
    all_done: std::sync::Condvar,
}

impl StripBatch {
    fn new(jobs: Vec<StripJob>) -> Arc<Self> {
        Arc::new(Self {
            strips: jobs.iter().map(|_| Mutex::new(None)).collect(),
            jobs,
            next: AtomicUsize::new(0),
            done: Mutex::new(0),
            all_done: std::sync::Condvar::new(),
        })
    }

    /// Have every worker thread help.
    fn start(self: &Arc<Self>, workers: &rayon::ThreadPool) {
        for _ in 0..workers.current_num_threads() {
            let batch = self.clone();
            workers.spawn(move || batch.work());
        }
    }

    /// Generate the strips of one path if one is left to claim; returns whether one was.
    fn work_one(&self) -> bool {
        let index = self.next.fetch_add(1, Ordering::Relaxed);
        if index >= self.jobs.len() {
            return false;
        }
        let strips = self.jobs[index].run();
        *lock(&self.strips[index]) = Some(strips);
        let mut done = lock(&self.done);
        *done += 1;
        if *done == self.jobs.len() {
            self.all_done.notify_all();
        }
        true
    }

    /// Generate strips until no path is left to claim.
    fn work(&self) {
        loop {
            let start = self.next.fetch_add(PATHS_PER_CLAIM, Ordering::Relaxed);
            if start >= self.jobs.len() {
                return;
            }
            let end = (start + PATHS_PER_CLAIM).min(self.jobs.len());
            for index in start..end {
                let strips = self.jobs[index].run();
                *lock(&self.strips[index]) = Some(strips);
            }
            let mut done = lock(&self.done);
            *done += end - start;
            if *done == self.jobs.len() {
                self.all_done.notify_all();
            }
        }
    }

    /// The strips of every path, in order, once the threads on them are done.
    fn finish(&self) -> Vec<JobStrips> {
        let mut done = lock(&self.done);
        while *done < self.jobs.len() {
            done = self
                .all_done
                .wait(done)
                .unwrap_or_else(|error| error.into_inner());
        }
        drop(done);
        self.strips
            .iter()
            .map(|strips| lock(strips).take().expect("every path was claimed"))
            .collect()
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|error| error.into_inner())
}

/// How one draw composites into its target.
#[derive(Clone, Copy)]
enum Composite {
    SourceOver,
    /// Straight into the target with a fixed-function blend.
    Direct(DirectBlend),
    /// Through a blend layer around the draw.
    Layer(peniko::BlendMode),
}

impl Composite {
    /// How one path, rectangle or image is drawn with `composition_operation`.
    ///
    /// Where the operation is a fixed-function blend ([`DirectBlend`]), the draw blends straight
    /// into its target. That is exactly what a blend layer around the one draw gives, inside clip
    /// and shadow layers too, since the draw then lands in that layer's texture just as the
    /// blend layer would. Each blend layer costs its own rounds of render passes, which on a
    /// tiling GPU (Apple's) each load and store whole intermediate textures.
    fn of_one_draw(composition_operation: CompositionOrBlending) -> Self {
        if composition_operation == CompositionOrBlending::Composition(CompositionStyle::SourceOver)
        {
            return Self::SourceOver;
        }
        let mode: peniko::BlendMode = composition_operation.convert();
        match DirectBlend::for_mode(mode) {
            Some(blend) if !forced_blend_layers() => Self::Direct(blend),
            _ => Self::Layer(mode),
        }
    }

    /// Run `f`, which records the draw into `scene`, composited this way.
    fn apply(self, scene: &mut Scene, f: impl FnOnce(&mut Scene)) {
        match self {
            Self::SourceOver => f(scene),
            Self::Direct(blend) => {
                scene.set_direct_blend(blend);
                f(scene);
                scene.set_direct_blend(DirectBlend::SourceOver);
            },
            Self::Layer(mode) => {
                scene.push_blend_layer(mode);
                f(scene);
                scene.pop_layer();
            },
        }
    }
}

thread_local! {
    /// This thread's strip generator: the canvas thread's, or a path worker's.
    static PATH_STRIP_GENERATOR: RefCell<PathStripGenerator> = RefCell::default();
    /// This thread's glyph caches and strip generator for text runs.
    static GLYPH_RUN_CACHES: RefCell<GlyphRunCaches> = RefCell::default();
}

fn path_strips(job: &PathJob) -> PathStrips {
    PATH_STRIP_GENERATOR.with(|generator| {
        let mut generator = generator.borrow_mut();
        match &job.stroke {
            Some(stroke) => generator.stroke(&job.path, stroke, &job.settings),
            None => generator.fill(&job.path, job.fill_rule, &job.settings),
        }
    })
}

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

/// Whether this thread lost its GL context since the last call. The canvas thread then
/// rebuilds its GPU canvases, which [`available`] now sends to the CPU backend.
pub(crate) fn take_context_loss() -> bool {
    CONTEXT_LOSS_PENDING.with(|pending| pending.replace(false))
}

/// Behave as if the GL context had just been lost (tests).
#[cfg(test)]
pub(crate) fn simulate_context_loss() {
    if let Some(thread) = GpuThread::get() {
        thread.lose("simulated");
    }
}

/// Have the driver mark this thread's context lost without telling this module (tests).
/// Returns false when the driver has no `GL_CHROMIUM_lose_context`; ANGLE has it.
///
/// ANGLE then still makes the context current and answers queries, and reports neither a
/// reset status nor `GL_CONTEXT_LOST`, but it drops every draw with a GL error.
#[cfg(test)]
pub(crate) fn lose_context_in_driver() -> bool {
    const GUILTY_CONTEXT_RESET: u32 = 0x8253;
    const INNOCENT_CONTEXT_RESET: u32 = 0x8254;
    let Some(thread) = GpuThread::get() else {
        return false;
    };
    thread.make_current();
    let pointer = thread
        .device
        .get_proc_address(&thread.context.borrow(), "glLoseContextCHROMIUM");
    if pointer.is_null() {
        return false;
    }
    let lose_context: extern "system" fn(u32, u32) = unsafe { std::mem::transmute(pointer) };
    lose_context(GUILTY_CONTEXT_RESET, INNOCENT_CONTEXT_RESET);
    true
}

/// Give this thread the state it shares with the paint thread. The canvas thread calls this
/// once, before it creates any canvas.
pub(crate) fn install_shared(shared: GpuCanvasShared) {
    GPU_SHARED.with(|slot| *slot.borrow_mut() = Some(shared));
}

fn shared() -> Option<GpuCanvasShared> {
    GPU_SHARED.with(|slot| slot.borrow().clone())
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
    shared: GpuCanvasShared,
    /// Reads a canvas's back buffer into its retained texture.
    copy_framebuffer: glow::NativeFramebuffer,
    /// Where canvas pixels are read back to on desktop OpenGL; see [`Gpu::read_retained`].
    /// Created with the first readback.
    readback_buffer: Cell<Option<glow::NativeBuffer>>,
    /// Threads that generate path strips, started with the first batch that is worth it.
    /// `None` inside if they could not be started; paths are then prepared on this thread.
    path_workers: std::cell::OnceCell<Option<rayon::ThreadPool>>,
    /// A texture that only exists to be asked about. A context whose calls have stopped
    /// having any effect answers false to `glIsTexture`, without raising an error or a reset
    /// status (surfman creates contexts without reset notification).
    canary: glow::NativeTexture,
    /// The GL context is gone, or the renderer is in an unknown state after a panic. Nothing
    /// touches GL or the renderer again.
    lost: Cell<bool>,
}

/// Whether every composite operation goes through a blend layer (tests).
#[cfg(test)]
pub(crate) fn forced_blend_layers() -> bool {
    FORCE_BLEND_LAYERS.with(Cell::get)
}

#[cfg(not(test))]
fn forced_blend_layers() -> bool {
    false
}

/// Send every composite operation through a blend layer on this thread (tests), to check that
/// direct blending gives the same pixels.
#[cfg(test)]
pub(crate) fn force_blend_layers(force: bool) {
    FORCE_BLEND_LAYERS.with(|forced| forced.set(force));
}

#[cfg(test)]
thread_local! {
    static FORCE_BLEND_LAYERS: Cell<bool> = const { Cell::new(false) };
}

thread_local! {
    /// What this thread shares with the paint thread; see [`install_shared`].
    static GPU_SHARED: RefCell<Option<GpuCanvasShared>> = const { RefCell::new(None) };
    static GPU_THREAD: RefCell<Option<Rc<GpuThread>>> = const { RefCell::new(None) };
    /// GPU setup failed on this thread, or its context was lost; canvases use the CPU backend.
    static GPU_FAILED: Cell<bool> = const { Cell::new(false) };
    /// The context was lost and the canvas thread has not rebuilt its GPU canvases yet.
    static CONTEXT_LOSS_PENDING: Cell<bool> = const { Cell::new(false) };
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
        let shared = shared().ok_or("this thread was not given the GPU canvas state")?;
        let details = shared
            .surfman_details
            .any()
            .ok_or("no painter surfman details")?;
        // vello_hybrid's WebGL renderer runs on OpenGL ES 3.0 (ANGLE on Windows) or on a desktop
        // OpenGL 3.3+ core profile (CGL on macOS), translating its shaders to GLSL 3.30 there.
        // Older desktop GL is reported unavailable below, so canvases use the CPU backend.
        let version = match details.connection.gl_api() {
            GLApi::GLES => GLVersion::new(3, 0),
            // Surfman creates a core profile for 3.2 and up. CGL then returns the newest core
            // version the GPU supports (4.1).
            GLApi::GL => GLVersion::new(3, 2),
        };
        let device = details
            .connection
            .create_device(&details.adapter)
            .map_err(|e| format!("device: {e:?}"))?;
        let attributes = ContextAttributes {
            version,
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
        let copy_framebuffer = match unsafe { glow.create_framebuffer() } {
            Ok(framebuffer) => framebuffer,
            Err(error) => {
                let _ = device.destroy_context(&mut context);
                return Err(error);
            },
        };
        // A texture name becomes a texture when it is first bound.
        let canary = unsafe {
            glow.create_texture().inspect(|texture| {
                glow.bind_texture(glow::TEXTURE_2D, Some(*texture));
                glow.bind_texture(glow::TEXTURE_2D, None);
            })
        };
        let canary = match canary {
            Ok(canary) => canary,
            Err(error) => {
                unsafe { glow.delete_framebuffer(copy_framebuffer) };
                let _ = device.destroy_context(&mut context);
                return Err(error);
            },
        };
        let gl_version = glow.version();
        if !gl_version.is_embedded && (gl_version.major, gl_version.minor) < (3, 3) {
            let version = format!("{}.{}", gl_version.major, gl_version.minor);
            unsafe {
                glow.delete_texture(canary);
                glow.delete_framebuffer(copy_framebuffer);
            }
            let _ = device.destroy_context(&mut context);
            return Err(format!(
                "the GPU canvas needs OpenGL ES 3.0 or OpenGL 3.3; this context is OpenGL {version}"
            ));
        }
        let gl = WebGl2RenderingContext::from_glow(glow, 1, 1);
        // The renderer panics rather than returning errors (shader compilation, link and
        // capability checks). Contain that here so a driver that cannot run it falls back to the
        // CPU backend instead of killing the canvas thread.
        let renderer = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            WebGlRenderer::new(gl.clone())
        }));
        let (renderer, resources) = match renderer {
            Ok(renderer) => renderer,
            Err(payload) => {
                let message = payload
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| payload.downcast_ref::<&str>().copied())
                    .unwrap_or("unknown panic");
                unsafe { gl.glow().delete_framebuffer(copy_framebuffer) };
                let _ = device.destroy_context(&mut context);
                return Err(format!("renderer: {message}"));
            },
        };
        Ok(Self {
            device,
            context: std::mem::ManuallyDrop::new(RefCell::new(context)),
            gl,
            renderer: RefCell::new(renderer),
            resources: RefCell::new(resources),
            uploads: RefCell::default(),
            shared,
            copy_framebuffer,
            readback_buffer: Cell::new(None),
            path_workers: Default::default(),
            canary,
            lost: Cell::new(false),
        })
    }

    /// Make the thread's context current and notice if it was lost. Callers must check
    /// [`Self::lost`] afterwards before they touch GL or the renderer.
    fn make_current(&self) {
        if self.lost.get() {
            return;
        }
        if let Err(error) = self.device.make_context_current(&self.context.borrow()) {
            self.lose(&format!("make current failed: {error:?}"));
            return;
        }
        if !unsafe { self.gl.glow().is_texture(self.canary) } {
            self.lose("its GL calls have no effect");
        }
    }

    /// The threads that generate path strips, started the first time they are asked for.
    /// `None` if they could not be started; paths are then prepared on this thread.
    fn path_workers(&self) -> Option<&rayon::ThreadPool> {
        self.path_workers
            .get_or_init(|| {
                let threads = servo_config::pref!(thread_pool_canvas_workers).max(1) as usize;
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .thread_name(|index| format!("CanvasPaths#{index}"))
                    .build()
                    .inspect_err(|error| warn!("GPU canvas: no path workers: {error}"))
                    .ok()
            })
            .as_ref()
    }

    /// Stop using the GPU on this thread.
    fn lose(&self, why: &str) {
        if self.lost.replace(true) {
            return;
        }
        warn!("GPU canvas: lost the GL context ({why}); canvases move to the CPU backend");
        GPU_FAILED.with(|failed| failed.set(true));
        CONTEXT_LOSS_PENDING.with(|pending| pending.set(true));
    }


    /// The atlas image for `pixmap`, uploading it if this content is not cached, pinned once.
    fn pin_upload(&self, pixmap: &vello_cpu::Pixmap) -> Option<(UploadKey, ImageId)> {
        if self.lost.get() {
            return None;
        }
        let key = UploadKey::of(pixmap);
        let mut uploads = self.uploads.borrow_mut();
        let epoch = uploads.epoch;
        if let Some(upload) = uploads.entries.get_mut(&key) {
            upload.pins += 1;
            upload.last_used = epoch;
            return Some((key, upload.id));
        }
        self.make_current();
        if self.lost.get() {
            return None;
        }
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
        if self.lost.get() {
            return None;
        }
        // As in `Gpu::render`: the renderer panics when GL fails under it.
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.renderer
                .borrow_mut()
                .try_upload_image(&mut self.resources.borrow_mut(), pixmap)
        }))
        .unwrap_or_else(|_| {
            self.lose("the renderer panicked");
            None
        })
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
        if self.lost.get() {
            return;
        }
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

/// A canvas frame kept as a GL texture, with bottom-up rows.
///
/// The canvas that rendered it holds it as its retained content. Other canvases of the thread
/// hold it while a scene of theirs that draws it is pending.
pub(crate) struct GpuFrame {
    thread: Rc<GpuThread>,
    texture: glow::NativeTexture,
    size: Size2D<u32>,
}

impl GpuFrame {
    /// An uninitialized frame texture of `size`. The thread's context must be current.
    fn new(thread: &Rc<GpuThread>, size: Size2D<u32>) -> Result<Self, String> {
        if thread.lost.get() {
            return Err("the GL context was lost".into());
        }
        let glow = thread.gl.glow();
        let texture = unsafe {
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
            thread: thread.clone(),
            texture,
            size,
        })
    }
}

impl Drop for GpuFrame {
    fn drop(&mut self) {
        self.thread.make_current();
        if !self.thread.lost.get() {
            unsafe { self.thread.gl.glow().delete_texture(self.texture) };
        }
    }
}

/// Per-canvas GL resources: a swap chain on the thread's context and the retained frame.
struct Gpu {
    thread: Rc<GpuThread>,
    image_id: ExternalImageId,
    /// Copy of the last rendered frame, drawn back when content must be retained.
    retained: Rc<GpuFrame>,
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
        thread
            .shared
            .canvas_image_ids
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .push(image_id);
        let retained = match GpuFrame::new(&thread, size) {
            Ok(frame) => Rc::new(frame),
            Err(error) => {
                Self::release(&thread, image_id);
                return Err(error);
            },
        };
        Ok(Self {
            thread,
            image_id,
            retained,
        })
    }

    /// Give back the swap chain and image id of a canvas.
    fn release(thread: &Rc<GpuThread>, image_id: ExternalImageId) {
        thread
            .shared
            .canvas_image_ids
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .retain(|id| *id != image_id);
        let _ = thread.shared.swap_chains.destroy(
            image_id,
            &thread.device,
            &mut thread.context.borrow_mut(),
        );
        thread
            .shared
            .external_image_id_manager
            .clone()
            .remove(&image_id);
    }

    /// Rasterize `scene` into this canvas's back buffer, keep a GPU copy of the result, and
    /// publish the buffer to WebRender. `sources` are the other canvases' frames the scene
    /// draws, in the order of their texture ids.
    fn render(
        &mut self,
        scene: &Scene,
        size: Size2D<u32>,
        clear: bool,
        draw_retained: bool,
        sources: &[Rc<GpuFrame>],
    ) {
        let thread = self.thread.clone();
        let thread = &thread;
        thread.make_current();
        if thread.lost.get() {
            return;
        }
        let Some(swap_chain) = thread.shared.swap_chains.get(self.image_id) else {
            return;
        };
        let surface_texture = match swap_chain
            .take_surface_texture(&thread.device, &mut thread.context.borrow_mut())
        {
            Ok(surface_texture) => surface_texture,
            Err(error) => {
                // Without a back buffer this canvas would silently stop updating.
                thread.lose(&format!("could not take the back buffer: {error:?}"));
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
                bindings.insert(RETAINED_TEXTURE, self.retained.texture);
            }
            for (index, source) in sources.iter().enumerate() {
                bindings.insert(
                    TextureId(FIRST_SOURCE_TEXTURE + index as u64),
                    source.texture,
                );
            }
            let render_size = RenderSize {
                width: size.width,
                height: size.height,
            };
            // The renderer reports GL failures by panicking, and a lost context makes object
            // creation fail. Contain that: a panic here would take down the canvas thread and
            // with it every canvas of the page.
            let rendered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                thread.renderer.borrow_mut().render_to_texture(
                    scene,
                    &mut thread.resources.borrow_mut(),
                    &render_size,
                    &bindings,
                    clear,
                    texture,
                    target,
                )
            }));
            match rendered {
                Ok(Ok(())) => {},
                Ok(Err(error)) => warn!("GPU canvas: render failed: {error:?}"),
                Err(_) => thread.lose("the renderer panicked"),
            }
            // Rendering raises no GL error when it works. A context that is lost keeps
            // answering queries, so the canary passes, but it drops every draw with an error.
            let error = unsafe { thread.gl.glow().get_error() };
            if error != glow::NO_ERROR {
                thread.lose(&format!("GL error {error:#x} while rendering"));
            }
            // Another canvas may still have to draw the frame this one replaces. Leave that
            // texture to it and retain the new frame in a fresh one.
            if !thread.lost.get() && Rc::strong_count(&self.retained) > 1 {
                match GpuFrame::new(thread, size) {
                    Ok(frame) => self.retained = Rc::new(frame),
                    Err(error) => warn!("GPU canvas: could not allocate a frame: {error}"),
                }
            }
            if !thread.lost.get() {
                self.retain_and_flush(target, texture, size);
            }
        }
        let mut context = thread.context.borrow_mut();
        if let Err(error) =
            swap_chain.recycle_surface_texture(&thread.device, &mut context, surface_texture)
        {
            thread.lose(&format!("could not return the back buffer: {error:?}"));
            return;
        }
        if let Err(error) = swap_chain.swap_buffers(
            &thread.device,
            &mut context,
            surfman::chains::PreserveBuffer::No,
        ) {
            thread.lose(&format!("could not present a frame: {error:?}"));
        }
    }

    /// Copy the back buffer, bound as `texture`, into the retained frame and flush, so that
    /// the buffer is complete when it moves to WebRender.
    fn retain_and_flush(&self, target: u32, texture: glow::NativeTexture, size: Size2D<u32>) {
        let glow = self.thread.gl.glow();
        unsafe {
            glow.bind_framebuffer(
                glow::READ_FRAMEBUFFER,
                Some(self.thread.copy_framebuffer),
            );
            glow.framebuffer_texture_2d(
                glow::READ_FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                target,
                Some(texture),
                0,
            );
            glow.bind_texture(glow::TEXTURE_2D, Some(self.retained.texture));
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
            // This is the synchronization with WebRender, not a hint. The painter samples
            // the buffer on its own device. Surfaces are shared between devices by share
            // handle (Direct3D) or IOSurface (macOS), where the producer has to flush
            // before another device reads; GL sync objects cannot cross devices. Without
            // this flush the other device reads every frame incomplete, which
            // `published_frame_is_complete_on_another_device` checks.
            glow.flush();
        }
    }

    /// Read `rect` of the retained copy back as top-down premultiplied RGBA. `rect` lies
    /// within the canvas, whose size is `size`.
    fn read_retained(&self, size: Size2D<u32>, rect: Rect<u32>) -> Vec<u8> {
        self.thread.make_current();
        let glow = self.thread.gl.glow();
        let (width, height) = (rect.size.width as usize, rect.size.height as usize);
        let len = width * height * 4;
        if self.thread.lost.get() || len == 0 {
            // The content is gone with the context.
            return vec![0; len];
        }
        let framebuffer = unsafe { glow.create_framebuffer().ok() };
        unsafe {
            glow.bind_framebuffer(glow::READ_FRAMEBUFFER, framebuffer);
            glow.framebuffer_texture_2d(
                glow::READ_FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(self.retained.texture),
                0,
            );
        }
        let (x, y) = (
            rect.origin.x as i32,
            (size.height - rect.origin.y - rect.size.height) as i32,
        );
        let pixels = if glow.version().is_embedded {
            self.read_pixels_gles(x, y, width, height)
        } else {
            self.read_pixels_desktop(x, y, width, height)
        };
        unsafe {
            glow.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
            if let Some(framebuffer) = framebuffer {
                glow.delete_framebuffer(framebuffer);
            }
        }
        pixels
    }

    /// Read the bound read framebuffer's `width`×`height` pixels at `x`, `y` on OpenGL ES
    /// (ANGLE), top row first.
    fn read_pixels_gles(&self, x: i32, y: i32, width: usize, height: usize) -> Vec<u8> {
        let glow = self.thread.gl.glow();
        let mut pixels = vec![0_u8; width * height * 4];
        // GL rows are bottom-up. ANGLE can hand them over top-down; elsewhere they are
        // turned around afterwards.
        let top_down = glow
            .supported_extensions()
            .contains("GL_ANGLE_pack_reverse_row_order");
        unsafe {
            if top_down {
                glow.pixel_store_i32(PACK_REVERSE_ROW_ORDER_ANGLE, 1);
            }
            glow.read_pixels(
                x,
                y,
                width as i32,
                height as i32,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(Some(&mut pixels)),
            );
            if top_down {
                glow.pixel_store_i32(PACK_REVERSE_ROW_ORDER_ANGLE, 0);
            }
        }
        if !top_down {
            flip_rows(&mut pixels, width * 4);
        }
        pixels
    }

    /// Read the bound read framebuffer's `width`×`height` pixels at `x`, `y` on desktop OpenGL
    /// (CGL on macOS), top row first.
    ///
    /// Reading into client memory there makes the driver wait for the GPU and then untile the
    /// texture on the CPU, into a buffer that would first have to be zeroed, with the rows
    /// turned around afterwards. Instead the GPU copies the pixels into a pixel pack buffer, in
    /// linear rows, and they are copied from there bottom row first: one pass over the pixels.
    fn read_pixels_desktop(&self, x: i32, y: i32, width: usize, height: usize) -> Vec<u8> {
        let glow = self.thread.gl.glow();
        let stride = width * 4;
        let len = stride * height;
        let buffer = match self.thread.readback_buffer.get() {
            Some(buffer) => Some(buffer),
            None => unsafe { glow.create_buffer().ok() }
                .inspect(|buffer| self.thread.readback_buffer.set(Some(*buffer))),
        };
        let Some(buffer) = buffer else {
            let mut pixels = vec![0_u8; len];
            unsafe {
                glow.read_pixels(
                    x,
                    y,
                    width as i32,
                    height as i32,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    glow::PixelPackData::Slice(Some(&mut pixels)),
                );
            }
            flip_rows(&mut pixels, stride);
            return pixels;
        };
        let mut pixels = Vec::with_capacity(len);
        unsafe {
            glow.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(buffer));
            glow.buffer_data_size(glow::PIXEL_PACK_BUFFER, len as i32, glow::STREAM_READ);
            glow.read_pixels(
                x,
                y,
                width as i32,
                height as i32,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelPackData::BufferOffset(0),
            );
            let mapped =
                glow.map_buffer_range(glow::PIXEL_PACK_BUFFER, 0, len as i32, glow::MAP_READ_BIT);
            if mapped.is_null() {
                pixels.resize(len, 0);
            } else {
                let rows = std::slice::from_raw_parts(mapped, len);
                for row in rows.chunks_exact(stride).rev() {
                    pixels.extend_from_slice(row);
                }
                glow.unmap_buffer(glow::PIXEL_PACK_BUFFER);
            }
            glow.bind_buffer(glow::PIXEL_PACK_BUFFER, None);
        }
        pixels
    }
}

/// `GL_PACK_REVERSE_ROW_ORDER_ANGLE` of `ANGLE_pack_reverse_row_order`: `glReadPixels`
/// returns the top row first.
const PACK_REVERSE_ROW_ORDER_ANGLE: u32 = 0x93A4;

/// Reverse the order of the `stride`-byte rows of `pixels` in place.
pub(crate) fn flip_rows(pixels: &mut [u8], stride: usize) {
    let rows = pixels.len() / stride;
    let (top, bottom) = pixels.split_at_mut(rows / 2 * stride);
    for (upper, lower) in top
        .chunks_exact_mut(stride)
        .zip(bottom.chunks_exact_mut(stride).rev())
    {
        upper.swap_with_slice(lower);
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        self.thread.make_current();
        Self::release(&self.thread, self.image_id);
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
    /// Other canvases' frames the pending scene draws; released after the next rasterization.
    /// The frame at index `i` is bound as texture `FIRST_SOURCE_TEXTURE + i`.
    source_frames: Vec<Rc<GpuFrame>>,
    /// Path and glyph draws that are not in the scene yet.
    pending_paths: PendingPaths,
    /// Open shadow (filter) layers: draws inside them do not target the canvas directly.
    shadow_depth: u32,
}

impl VelloGpuDrawTarget {
    /// Run `f`, which draws one path, rectangle or image, composited with
    /// `composition_operation`; see [`Composite::of_one_draw`].
    fn with_composition(
        &mut self,
        composition_operation: CompositionOrBlending,
        f: impl FnOnce(&mut Self),
    ) {
        match Composite::of_one_draw(composition_operation) {
            Composite::SourceOver => f(self),
            Composite::Direct(blend) => {
                self.scene.set_direct_blend(blend);
                f(self);
                self.scene.set_direct_blend(DirectBlend::SourceOver);
            },
            Composite::Layer(mode) => {
                self.scene.push_blend_layer(mode);
                f(self);
                self.scene.pop_layer();
            },
        }
    }

    /// [`Self::with_composition`] for `f` drawing several glyphs. A blend layer composites them
    /// together, which differs from blending each glyph where glyphs overlap, so only `lighter`
    /// blends directly here, and only outside clip and shadow layers, as it always did.
    fn with_text_composition(
        &mut self,
        composition_operation: CompositionOrBlending,
        f: impl FnOnce(&mut Self),
    ) {
        if composition_operation == CompositionOrBlending::Composition(CompositionStyle::SourceOver)
        {
            f(self);
            return;
        }
        if composition_operation == CompositionOrBlending::Composition(CompositionStyle::Lighter) &&
            self.clips.is_empty() &&
            self.shadow_depth == 0
        {
            self.scene.set_direct_blend(DirectBlend::Plus);
            f(self);
            self.scene.set_direct_blend(DirectBlend::SourceOver);
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

    /// Record the pending path draws, in the order they were made. Everything that changes
    /// the scene other than another path draw calls this first.
    fn draw_pending_paths(&mut self) {
        self.pending_paths.record(&mut self.scene, &self.gpu.thread);
    }

    /// Queue a path fill (`stroke` is `None`) or stroke. Its strips are generated together
    /// with those of the path draws around it.
    fn draw_path(
        &mut self,
        path: &Path,
        stroke: Option<kurbo::Stroke>,
        fill_rule: peniko::Fill,
        style: FillOrStrokeStyle,
        composition_options: CompositionOptions,
        transform: Transform2D<f64>,
    ) {
        // Not `ensure_drawing`: that would record the path draws before this one first.
        self.start_drawing();
        let Some(paint) = self.paint(style, composition_options.alpha) else {
            return;
        };
        let transform: kurbo::Affine = transform.cast().into();
        self.scene.set_transform(transform);
        let Some(settings) = self.scene.path_strip_settings() else {
            // The scene has to generate these strips itself.
            self.draw_pending_paths();
            self.with_composition(composition_options.composition_operation, |self_| {
                self_.scene.set_paint(paint);
                match stroke {
                    Some(stroke) => {
                        self_.scene.set_stroke(stroke);
                        self_.scene.stroke_path(&path.0);
                    },
                    None => {
                        self_.scene.set_fill_rule(fill_rule);
                        self_.scene.fill_path(&path.0);
                        self_.scene.set_fill_rule(peniko::Fill::NonZero);
                    },
                }
            });
            return;
        };
        let job = PathJob {
            path: path.0.clone(),
            stroke,
            fill_rule,
            settings,
        };
        let draw = PendingDraw::Canvas {
            transform,
            paint,
            composite: Composite::of_one_draw(composition_options.composition_operation),
        };
        self.pending_paths
            .push(&self.gpu.thread, StripJob::Path(job), draw);
    }

    /// Fill (`line_options` is `None`) or stroke `text_runs`.
    ///
    /// Generating the strips of the glyph outlines is nearly all the work. Source-over text
    /// therefore joins the pending path draws, whose strips are generated together on the worker
    /// threads, as the CPU backend rasterizes on its worker threads. The strips are the ones the
    /// scene would have generated.
    fn draw_text(
        &mut self,
        text_runs: Vec<TextRun>,
        style: FillOrStrokeStyle,
        line_options: Option<LineOptions>,
        composition_options: CompositionOptions,
        transform: Transform2D<f64>,
    ) {
        let deferred = composition_options.composition_operation
            == CompositionOrBlending::Composition(CompositionStyle::SourceOver);
        if deferred {
            // Not `ensure_drawing`: that would record the pending draws first.
            self.start_drawing();
        } else {
            self.ensure_drawing();
        }
        let Some(paint) = self.paint(style, composition_options.alpha) else {
            return;
        };
        self.scene.set_paint(paint.clone());
        let stroked = line_options.is_some();
        if let Some(line_options) = line_options {
            self.scene.set_stroke(line_options.convert());
        }
        self.scene.set_transform(transform.cast().into());
        let thread = self.gpu.thread.clone();
        if deferred {
            let visible =
                !matches!(&paint, vello_cpu::PaintType::Solid(color) if color.components[3] == 0.0);
            for text_run in &text_runs {
                if !visible {
                    break;
                }
                with_font(text_run, |font| {
                    // A few glyphs per job, so that a long or costly run (strokes) spreads over
                    // the threads too.
                    let all_glyphs: Vec<_> = glyphs(text_run).collect();
                    let mut jobs = all_glyphs.chunks(GLYPHS_PER_JOB).map(|chunk| {
                        self.scene.glyph_run_job(
                            font,
                            text_run.pt_size,
                            chunk.iter().copied(),
                            stroked,
                        )
                    });
                    match jobs.by_ref().collect::<Option<Vec<_>>>() {
                        Some(jobs) => {
                            for job in jobs {
                                self.pending_paths.push(
                                    &thread,
                                    StripJob::Text(job),
                                    PendingDraw::Text(paint.clone()),
                                );
                            }
                        },
                        None => {
                            self.pending_paths.record(&mut self.scene, &thread);
                            self.scene.set_paint(paint.clone());
                            let mut resources = thread.resources.borrow_mut();
                            let run = self
                                .scene
                                .glyph_run(&mut resources, font)
                                .font_size(text_run.pt_size);
                            if stroked {
                                run.stroke_glyphs(glyphs(text_run));
                            } else {
                                run.fill_glyphs(glyphs(text_run));
                            }
                        },
                    }
                });
            }
            return;
        }
        self.with_text_composition(composition_options.composition_operation, |self_| {
            for text_run in &text_runs {
                with_font(text_run, |font| {
                    let mut resources = thread.resources.borrow_mut();
                    let run = self_
                        .scene
                        .glyph_run(&mut resources, font)
                        .font_size(text_run.pt_size);
                    if stroked {
                        run.stroke_glyphs(glyphs(text_run));
                    } else {
                        run.fill_glyphs(glyphs(text_run));
                    }
                });
            }
        })
    }

    /// Before new commands after a rasterization, draw the retained content back as
    /// background. Records the pending path draws first.
    fn ensure_drawing(&mut self) {
        self.draw_pending_paths();
        self.start_drawing();
    }

    /// [`Self::ensure_drawing`] for a path draw, which joins the pending ones.
    fn start_drawing(&mut self) {
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
        self.draw_pending_paths();
        if self.state != State::Drawing {
            return;
        }
        let size = self.size;
        let (clear, draw_retained) = (self.clear_next, self.draws_retained);
        self.ignore_clips(|self_| {
            self_.gpu.render(
                &self_.scene,
                size,
                clear,
                draw_retained,
                &self_.source_frames,
            );
            self_.scene.reset();
        });
        self.source_frames.clear();
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
            source_frames: Vec::new(),
            pending_paths: PendingPaths::default(),
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

    /// This canvas's current content as a frame another canvas of the thread can draw with
    /// [`Self::draw_frame`]. Rasterizes pending commands; copies nothing.
    pub(crate) fn frame(&mut self) -> Rc<GpuFrame> {
        self.flush();
        self.gpu.retained.clone()
    }

    /// Generate strips of pending draws on this thread while `idle` holds; see
    /// [`PendingPaths`].
    pub(crate) fn work_while(&mut self, idle: &dyn Fn() -> bool) {
        let thread = self.gpu.thread.clone();
        self.pending_paths.work_while(&thread, idle);
    }

    /// Draw the `source` region of another canvas's `frame`, in canvas pixels from its top
    /// left, into `dest`. The frame is sampled on the GPU: nothing is read back or uploaded.
    /// What is drawn is the frame as it was when [`Self::frame`] returned it, whatever its
    /// canvas draws afterwards.
    pub(crate) fn draw_frame(
        &mut self,
        frame: &Rc<GpuFrame>,
        source: Rect<u32>,
        dest: Rect<f64>,
        filter: Filter,
        composition_options: CompositionOptions,
        transform: Transform2D<f64>,
    ) {
        let Some(source) = source.intersection(&Rect::from_size(frame.size)) else {
            return;
        };
        if source.is_empty() {
            return;
        }
        self.ensure_drawing();
        let index = match self
            .source_frames
            .iter()
            .position(|held| Rc::ptr_eq(held, frame))
        {
            Some(index) => index,
            None => {
                self.source_frames.push(frame.clone());
                self.source_frames.len() - 1
            },
        };
        let texture = TextureId(FIRST_SOURCE_TEXTURE + index as u64);
        let (width, height) = (f64::from(source.size.width), f64::from(source.size.height));
        let scale_up = dest.size.width > width || dest.size.height > height;
        // Global alpha as a multiply tint, quantized to 8 bits like `draw_surface`.
        let alpha = (composition_options.alpha * 255.0) as u8;
        let tint = (alpha != u8::MAX).then(|| vello_hybrid::Tint {
            color: peniko::Color::new([1.0, 1.0, 1.0, f32::from(alpha) / 255.0]),
            mode: vello_hybrid::TintMode::Multiply,
        });
        // Frame rows are bottom-up: take the region from the bottom and flip it into place.
        let bottom = frame.size.height - source.max_y();
        let source_region = vello_hybrid::RectU16::new(
            source.min_x() as u16,
            bottom as u16,
            source.max_x() as u16,
            (bottom + source.size.height) as u16,
        );
        let placement = kurbo::Affine::translate((dest.origin.x, dest.origin.y))
            .pre_scale_non_uniform(dest.size.width / width, dest.size.height / height) *
            kurbo::Affine::new([1.0, 0.0, 0.0, -1.0, 0.0, height]);
        self.with_composition(composition_options.composition_operation, move |self_| {
            self_.scene.set_tint(tint);
            self_.scene.set_transform(transform.cast().into());
            self_.scene.draw_texture_rects(
                texture,
                // Bilinear only when scaling up, as for image sources.
                if scale_up {
                    filter.convert()
                } else {
                    peniko::ImageQuality::Low
                },
                [SampleRect {
                    source_region,
                    transform: placement,
                }],
            );
            self_.scene.reset_tint();
        })
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
            // The swap chain's surfaces are rectangle textures (IOSurfaces) on macOS.
            image_type: ExternalImageType::TextureHandle(
                match self.gpu.thread.device.surface_gl_texture_target() {
                    glow::TEXTURE_RECTANGLE => ImageBufferKind::TextureRect,
                    _ => ImageBufferKind::Texture2D,
                },
            ),
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
            // The pending path draws would be cleared too.
            self.pending_paths.clear();
            self.scene.reset();
            self.source_frames.clear();
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
        self.draw_pending_paths();
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
        self.draw_path(
            path,
            None,
            fill_rule.convert(),
            style,
            composition_options,
            transform,
        );
    }

    fn fill_text(
        &mut self,
        text_runs: Vec<TextRun>,
        style: FillOrStrokeStyle,
        composition_options: CompositionOptions,
        transform: Transform2D<f64>,
    ) {
        self.draw_text(text_runs, style, None, composition_options, transform);
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
        self.draw_pending_paths();
        if self.clips.pop().is_some() {
            self.scene.pop_layer();
        }
    }

    fn push_clip(&mut self, path: &Path, fill_rule: FillRule, transform: Transform2D<f64>) {
        self.draw_pending_paths();
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
        self.draw_path(
            path,
            Some(line_options.convert()),
            peniko::Fill::NonZero,
            style,
            composition_options,
            transform,
        );
    }

    fn stroke_text(
        &mut self,
        text_runs: Vec<TextRun>,
        style: FillOrStrokeStyle,
        line_options: LineOptions,
        composition_options: CompositionOptions,
        transform: Transform2D<f64>,
    ) {
        self.draw_text(
            text_runs,
            style,
            Some(line_options),
            composition_options,
            transform,
        );
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
        self.snapshot_rect(Rect::from_size(self.size))
    }

    fn snapshot_rect(&mut self, rect: Rect<u32>) -> Snapshot {
        self.flush();
        Snapshot::from_vec(
            rect.size,
            SnapshotPixelFormat::RGBA,
            SnapshotAlphaMode::Transparent {
                premultiplied: true,
            },
            self.gpu.read_retained(self.size, rect),
        )
    }

    fn surface(&mut self) -> Self::SourceSurface {
        snapshot_as_pixmap(self.snapshot())
    }

    fn create_source_surface_from_data(&self, data: Snapshot) -> Option<Self::SourceSurface> {
        Some(snapshot_as_pixmap(data))
    }
}
