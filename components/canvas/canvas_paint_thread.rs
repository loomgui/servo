/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::borrow::ToOwned;
use std::{f32, thread};

use crossbeam_channel::{Sender, select, unbounded};
use euclid::default::{Rect, Size2D, Transform2D};
use log::warn;
use paint_api::{CrossProcessPaintApi, GpuCanvasShared};
use pixels::Snapshot;
use profile_traits::mem::{
    ProcessReports, ProfilerChan, Report, ReportsChan, perform_memory_report,
};
use rustc_hash::FxHashMap;
use servo_base::generic_channel::GenericSender;
use servo_base::{Epoch, generic_channel};
use servo_canvas_traits::ConstellationCanvasMsg;
use servo_canvas_traits::canvas::*;
use webrender_api::ImageKey;

use crate::canvas_data::*;

pub struct CanvasPaintThread {
    canvases: FxHashMap<CanvasId, Canvas>,
    next_canvas_id: CanvasId,
    paint_api: CrossProcessPaintApi,
}

impl CanvasPaintThread {
    fn new(paint_api: CrossProcessPaintApi) -> CanvasPaintThread {
        CanvasPaintThread {
            canvases: FxHashMap::default(),
            next_canvas_id: CanvasId(0),
            paint_api,
        }
    }

    /// Creates a new `CanvasPaintThread` and returns an `IpcSender` to
    /// communicate with it.
    ///
    /// `gpu_canvas_shared` is what the thread needs to rasterize canvases on the GPU under the
    /// `gpu` canvas backend. Without it every canvas uses the CPU backend.
    pub fn start(
        paint_api: CrossProcessPaintApi,
        mem_profiler_chan: ProfilerChan,
        gpu_canvas_shared: Option<GpuCanvasShared>,
    ) -> (Sender<ConstellationCanvasMsg>, GenericSender<CanvasMsg>) {
        let (ipc_sender, ipc_receiver) = generic_channel::channel::<CanvasMsg>().unwrap();
        let msg_receiver = ipc_receiver.route_preserving_errors();
        let (create_sender, create_receiver) = unbounded();
        let registration = mem_profiler_chan.prepare_memory_reporting(
            "canvas".into(),
            create_sender.clone(),
            ConstellationCanvasMsg::CollectMemoryReport,
        );
        thread::Builder::new()
            .name("Canvas".to_owned())
            .spawn(move || {
                let _registration = registration;
                if let Some(gpu_canvas_shared) = gpu_canvas_shared {
                    crate::vello_gpu_backend::install_shared(gpu_canvas_shared);
                }
                let mut canvas_paint_thread = CanvasPaintThread::new(
                    paint_api);
                loop {
                    select! {
                        recv(msg_receiver) -> msg => {
                            match msg {
                                Ok(Ok((canvas_id, command))) => {
                                    canvas_paint_thread.process_command(command, canvas_id);
                                    canvas_paint_thread.recover_from_gpu_context_loss();
                                },
                                Ok(Err(e)) => {
                                    warn!("CanvasPaintThread message deserialization error: {e:?}");
                                }
                                Err(_disconnected) => {
                                    warn!("CanvasMsg receiver disconnected");
                                    break;
                                },
                            }
                        }
                        recv(create_receiver) -> msg => {
                            match msg {
                                Ok(ConstellationCanvasMsg::Create { sender: creator, size }) => {
                                    if let Err(error) = creator.send(canvas_paint_thread.create_canvas(size)) {
                                        warn!("Create canvas response failed ({error})");
                                    }
                                },
                                Ok(ConstellationCanvasMsg::CollectMemoryReport(sender)) => {
                                    canvas_paint_thread.collect_memory_reports(sender);
                                },
                                Ok(ConstellationCanvasMsg::Exit(exit_sender)) => {
                                    let _ = exit_sender.send(());
                                    break;
                                },
                                Err(e) => {
                                    warn!("Error on CanvasPaintThread receive ({})", e);
                                    break;
                                },
                            }
                        }
                    }
                }
            })
            .expect("Thread spawning failed");

        (create_sender, ipc_sender)
    }

    /// A lost GL context leaves this thread's GPU canvases unable to draw. Rebuild them on the
    /// CPU backend. Their content is gone; they show again with the page's next drawing.
    fn recover_from_gpu_context_loss(&mut self) {
        if !crate::vello_gpu_backend::take_context_loss() {
            return;
        }
        for canvas in self.canvases.values_mut() {
            if canvas.is_gpu() {
                canvas.recreate_with_backend_choice(None);
            }
        }
    }

    #[servo_tracing::instrument(skip_all)]
    pub fn create_canvas(&mut self, size: Size2D<u64>) -> Option<CanvasId> {
        let canvas_id = self.next_canvas_id;
        self.next_canvas_id.0 += 1;

        let canvas = Canvas::new(size, self.paint_api.clone())?;
        self.canvases.insert(canvas_id, canvas);

        Some(canvas_id)
    }

    fn collect_memory_reports(&self, sender: ReportsChan) {
        perform_memory_report(|ops| {
            let mut reports: Vec<Report> = self
                .canvases
                .iter()
                .flat_map(|(canvas_id, canvas)| canvas.collect_memory_report(*canvas_id, ops))
                .collect();
            // Loom: GPU resources shared by all GPU canvases (image atlas, cached uploads).
            reports.extend(
                crate::vello_gpu_backend::shared_memory_report()
                    .into_iter()
                    .map(|(name, size)| Report {
                        path: profile_traits::path!["canvas", "gpu-shared", name],
                        kind: profile_traits::mem::ReportKind::ExplicitNonHeapSize,
                        size,
                    }),
            );
            sender.send(ProcessReports::new(reports));
        });
    }

    #[servo_tracing::instrument(
        skip_all,
        fields(message = message.to_string())
    )]
    fn process_command(&mut self, message: CanvasCommand, canvas_id: CanvasId) {
        match message {
            CanvasCommand::Recreate(size) => self.canvas(canvas_id).recreate(size),
            CanvasCommand::Destroy => {
                self.canvases.remove(&canvas_id);
            },
            CanvasCommand::SetImageKey(image_key) => {
                self.canvas(canvas_id).set_image_key(image_key);
            },
            CanvasCommand::FillText(
                text_bounds,
                text_runs,
                fill_or_stroke_style,
                shadow_options,
                composition_options,
                transform,
            ) => {
                self.canvas(canvas_id).fill_text(
                    text_bounds,
                    text_runs,
                    fill_or_stroke_style,
                    shadow_options,
                    composition_options,
                    transform,
                );
            },
            CanvasCommand::StrokeText(
                text_bounds,
                text_runs,
                fill_or_stroke_style,
                line_options,
                shadow_options,
                composition_options,
                transform,
            ) => {
                self.canvas(canvas_id).stroke_text(
                    text_bounds,
                    text_runs,
                    fill_or_stroke_style,
                    line_options,
                    shadow_options,
                    composition_options,
                    transform,
                );
            },
            CanvasCommand::FillRect(
                rect,
                style,
                shadow_options,
                composition_options,
                transform,
            ) => {
                self.canvas(canvas_id).fill_rect(
                    &rect,
                    style,
                    shadow_options,
                    composition_options,
                    transform,
                );
            },
            CanvasCommand::StrokeRect(
                rect,
                style,
                line_options,
                shadow_options,
                composition_options,
                transform,
            ) => {
                self.canvas(canvas_id).stroke_rect(
                    &rect,
                    style,
                    line_options,
                    shadow_options,
                    composition_options,
                    transform,
                );
            },
            CanvasCommand::ClearRect(ref rect, transform) => {
                self.canvas(canvas_id).clear_rect(rect, transform)
            },
            CanvasCommand::FillPath(
                style,
                path,
                fill_rule,
                shadow_options,
                composition_options,
                transform,
            ) => {
                self.canvas(canvas_id).fill_path(
                    &path,
                    fill_rule,
                    style,
                    shadow_options,
                    composition_options,
                    transform,
                );
            },
            CanvasCommand::StrokePath(
                path,
                style,
                line_options,
                shadow_options,
                composition_options,
                transform,
            ) => {
                self.canvas(canvas_id).stroke_path(
                    &path,
                    style,
                    line_options,
                    shadow_options,
                    composition_options,
                    transform,
                );
            },
            CanvasCommand::ClipPath(path, fill_rule, transform) => {
                self.canvas(canvas_id)
                    .clip_path(&path, fill_rule, transform);
            },
            CanvasCommand::DrawImage(
                snapshot,
                dest_rect,
                source_rect,
                smoothing_enabled,
                shadow_options,
                composition_options,
                transform,
            ) => self.canvas(canvas_id).draw_image(
                snapshot.to_owned(),
                dest_rect,
                source_rect,
                smoothing_enabled,
                shadow_options,
                composition_options,
                transform,
            ),
            CanvasCommand::DrawEmptyImage(
                image_size,
                dest_rect,
                source_rect,
                shadow_options,
                composition_options,
                transform,
            ) => self.canvas(canvas_id).draw_image(
                Snapshot::cleared(image_size),
                dest_rect,
                source_rect,
                false,
                shadow_options,
                composition_options,
                transform,
            ),
            CanvasCommand::DrawImageInOther(
                other_canvas_id,
                dest_rect,
                source_rect,
                smoothing,
                shadow_options,
                composition_options,
                transform,
            ) => {
                // Between two GPU canvases the destination samples the source's frame on the
                // GPU. Any other pair goes through the source's pixels.
                let frame = if self.canvas(other_canvas_id).is_gpu() {
                    self.canvas(canvas_id).gpu_frame()
                } else {
                    None
                };
                if let Some(frame) = frame {
                    self.canvas(other_canvas_id).draw_gpu_frame(
                        frame,
                        dest_rect,
                        source_rect,
                        smoothing,
                        shadow_options,
                        composition_options,
                        transform,
                    );
                } else {
                    let snapshot = self
                        .canvas(canvas_id)
                        .read_pixels(Some(source_rect.to_u32()));
                    self.canvas(other_canvas_id).draw_image(
                        snapshot,
                        dest_rect,
                        source_rect,
                        smoothing,
                        shadow_options,
                        composition_options,
                        transform,
                    );
                }
            },
            CanvasCommand::GetImageData(dest_rect, sender) => {
                let snapshot = self.canvas(canvas_id).read_pixels(dest_rect);
                if let Err(error) = sender.send(snapshot.to_shared()) {
                    warn!("GetImageData response failed ({error})");
                }
            },
            CanvasCommand::PutImageData(rect, snapshot) => {
                self.canvas(canvas_id)
                    .put_image_data(snapshot.to_owned(), rect);
            },
            CanvasCommand::UpdateImage(canvas_epoch) => {
                self.canvas(canvas_id).update_image_rendering(canvas_epoch);
            },
            CanvasCommand::PopClips(clips) => self.canvas(canvas_id).pop_clips(clips),
            CanvasCommand::ProcessBatchMessages(messages) => {
                for message in messages {
                    self.process_command(message, canvas_id);
                }
            },
        }
    }

    fn canvas(&mut self, canvas_id: CanvasId) -> &mut Canvas {
        self.canvases.get_mut(&canvas_id).expect("Bogus canvas id")
    }
}

#[cfg_attr(
    feature = "vello",
    expect(
        clippy::large_enum_variant,
        reason = "Current consensus is on keeping enum instead of boxing it. https://github.com/servo/servo/pull/46863"
    )
)]
enum Canvas {
    #[cfg(feature = "vello")]
    Vello(CanvasData<crate::vello_backend::VelloDrawTarget>),
    VelloCPU(CanvasData<crate::vello_cpu_backend::VelloCPUDrawTarget>),
    /// GPU-rasterized canvas (Loom prototype), selected with `dom_canvas_backend = "gpu"`.
    VelloGpu(CanvasData<crate::vello_gpu_backend::VelloGpuDrawTarget>),
}

impl Canvas {
    fn new(size: Size2D<u64>, paint_api: CrossProcessPaintApi) -> Option<Self> {
        match servo_config::pref!(dom_canvas_backend)
            .to_lowercase()
            .as_str()
        {
            #[cfg(feature = "vello")]
            "vello" => Some(Self::Vello(CanvasData::new(size, paint_api))),
            "gpu" if Self::wants_gpu(size) => Some(
                Self::try_gpu(size, paint_api.clone())
                    .unwrap_or_else(|| Self::VelloCPU(CanvasData::new(size, paint_api))),
            ),
            _ => Some(Self::VelloCPU(CanvasData::new(size, paint_api))),
        }
    }

    /// Loom: under the `gpu` backend, whether a canvas of `size` belongs on the GPU.
    fn wants_gpu(size: Size2D<u64>) -> bool {
        let min_pixels = servo_config::pref!(dom_canvas_gpu_min_pixels).max(0) as u64;
        size.area() >= min_pixels && crate::vello_gpu_backend::available()
    }

    /// Loom: a GPU canvas of `size`, or `None` when its GPU resources cannot be created.
    fn try_gpu(size: Size2D<u64>, paint_api: CrossProcessPaintApi) -> Option<Self> {
        let target = crate::vello_gpu_backend::VelloGpuDrawTarget::try_new(
            size.max(Size2D::new(1, 1)).cast(),
        )?;
        Some(Self::VelloGpu(CanvasData::with_draw_target(
            target, paint_api,
        )))
    }

    /// Loom: recreate (resize or reset) under the `gpu` backend. GPU canvases are rebuilt here,
    /// so a failed GPU allocation falls back to the CPU backend instead of panicking, and canvases
    /// move between backends when their size crosses `dom_canvas_gpu_min_pixels`. Returns whether
    /// the canvas was recreated.
    fn recreate_with_backend_choice(&mut self, size: Option<Size2D<u64>>) -> bool {
        if servo_config::pref!(dom_canvas_backend).to_lowercase() != "gpu" {
            return false;
        }
        let (current_size, is_gpu) = match self {
            #[cfg(feature = "vello")]
            Canvas::Vello(_) => return false,
            Canvas::VelloCPU(canvas_data) => (canvas_data.size(), false),
            Canvas::VelloGpu(canvas_data) => (canvas_data.size(), true),
        };
        let size = size.unwrap_or(current_size);
        let wants_gpu = Self::wants_gpu(size);
        if !wants_gpu && !is_gpu {
            return false;
        }
        let (image_key, paint_api) = match self {
            #[cfg(feature = "vello")]
            Canvas::Vello(_) => unreachable!(),
            Canvas::VelloCPU(canvas_data) => {
                (canvas_data.take_image_key(), canvas_data.paint_api())
            },
            Canvas::VelloGpu(canvas_data) => {
                (canvas_data.take_image_key(), canvas_data.paint_api())
            },
        };
        *self = wants_gpu
            .then(|| Self::try_gpu(size, paint_api.clone()))
            .flatten()
            .unwrap_or_else(|| Canvas::VelloCPU(CanvasData::new(size, paint_api)));
        if let Some(image_key) = image_key {
            match self {
                #[cfg(feature = "vello")]
                Canvas::Vello(_) => unreachable!(),
                Canvas::VelloCPU(canvas_data) => {
                    canvas_data.adopt_image_key(image_key);
                    canvas_data.update_image_rendering(None);
                },
                Canvas::VelloGpu(canvas_data) => {
                    canvas_data.adopt_image_key(image_key);
                    canvas_data.update_image_rendering(None);
                },
            }
        }
        true
    }

    fn collect_memory_report(
        &self,
        canvas_id: CanvasId,
        ops: &mut malloc_size_of::MallocSizeOfOps,
    ) -> Vec<Report> {
        match self {
            #[cfg(feature = "vello")]
            Canvas::Vello(canvas_data) => canvas_data.collect_memory_report(canvas_id, ops),
            Canvas::VelloCPU(canvas_data) => canvas_data.collect_memory_report(canvas_id, ops),
            Canvas::VelloGpu(canvas_data) => canvas_data.collect_memory_report(canvas_id, ops),
        }
    }

    fn set_image_key(&mut self, image_key: ImageKey) {
        match self {
            #[cfg(feature = "vello")]
            Canvas::Vello(canvas_data) => canvas_data.set_image_key(image_key),
            Canvas::VelloCPU(canvas_data) => canvas_data.set_image_key(image_key),
            Canvas::VelloGpu(canvas_data) => canvas_data.set_image_key(image_key),
        }
    }

    fn pop_clips(&mut self, clips: usize) {
        match self {
            #[cfg(feature = "vello")]
            Canvas::Vello(canvas_data) => canvas_data.pop_clips(clips),
            Canvas::VelloCPU(canvas_data) => canvas_data.pop_clips(clips),
            Canvas::VelloGpu(canvas_data) => canvas_data.pop_clips(clips),
        }
    }

    fn stroke_text(
        &mut self,
        text_bounds: Rect<f64>,
        text_runs: Vec<TextRun>,
        fill_or_stroke_style: FillOrStrokeStyle,
        line_options: LineOptions,
        shadow_options: ShadowOptions,
        composition_options: CompositionOptions,
        transform: Transform2D<f64>,
    ) {
        match self {
            #[cfg(feature = "vello")]
            Canvas::Vello(canvas_data) => canvas_data.stroke_text(
                text_bounds,
                text_runs,
                fill_or_stroke_style,
                line_options,
                shadow_options,
                composition_options,
                transform,
            ),
            Canvas::VelloCPU(canvas_data) => canvas_data.stroke_text(
                text_bounds,
                text_runs,
                fill_or_stroke_style,
                line_options,
                shadow_options,
                composition_options,
                transform,
            ),
            Canvas::VelloGpu(canvas_data) => canvas_data.stroke_text(
                text_bounds,
                text_runs,
                fill_or_stroke_style,
                line_options,
                shadow_options,
                composition_options,
                transform,
            ),
        }
    }

    fn fill_text(
        &mut self,
        text_bounds: Rect<f64>,
        text_runs: Vec<TextRun>,
        fill_or_stroke_style: FillOrStrokeStyle,
        shadow_options: ShadowOptions,
        composition_options: CompositionOptions,
        transform: Transform2D<f64>,
    ) {
        match self {
            #[cfg(feature = "vello")]
            Canvas::Vello(canvas_data) => canvas_data.fill_text(
                text_bounds,
                text_runs,
                fill_or_stroke_style,
                shadow_options,
                composition_options,
                transform,
            ),
            Canvas::VelloCPU(canvas_data) => canvas_data.fill_text(
                text_bounds,
                text_runs,
                fill_or_stroke_style,
                shadow_options,
                composition_options,
                transform,
            ),
            Canvas::VelloGpu(canvas_data) => canvas_data.fill_text(
                text_bounds,
                text_runs,
                fill_or_stroke_style,
                shadow_options,
                composition_options,
                transform,
            ),
        }
    }

    fn fill_rect(
        &mut self,
        rect: &Rect<f32>,
        style: FillOrStrokeStyle,
        shadow_options: ShadowOptions,
        composition_options: CompositionOptions,
        transform: Transform2D<f64>,
    ) {
        match self {
            #[cfg(feature = "vello")]
            Canvas::Vello(canvas_data) => {
                canvas_data.fill_rect(rect, style, shadow_options, composition_options, transform)
            },
            Canvas::VelloCPU(canvas_data) => {
                canvas_data.fill_rect(rect, style, shadow_options, composition_options, transform)
            },
            Canvas::VelloGpu(canvas_data) => {
                canvas_data.fill_rect(rect, style, shadow_options, composition_options, transform)
            },
        }
    }

    fn stroke_rect(
        &mut self,
        rect: &Rect<f32>,
        style: FillOrStrokeStyle,
        line_options: LineOptions,
        shadow_options: ShadowOptions,
        composition_options: CompositionOptions,
        transform: Transform2D<f64>,
    ) {
        match self {
            #[cfg(feature = "vello")]
            Canvas::Vello(canvas_data) => canvas_data.stroke_rect(
                rect,
                style,
                line_options,
                shadow_options,
                composition_options,
                transform,
            ),
            Canvas::VelloCPU(canvas_data) => canvas_data.stroke_rect(
                rect,
                style,
                line_options,
                shadow_options,
                composition_options,
                transform,
            ),
            Canvas::VelloGpu(canvas_data) => canvas_data.stroke_rect(
                rect,
                style,
                line_options,
                shadow_options,
                composition_options,
                transform,
            ),
        }
    }

    fn fill_path(
        &mut self,
        path: &Path,
        fill_rule: FillRule,
        style: FillOrStrokeStyle,
        shadow_options: ShadowOptions,
        composition_options: CompositionOptions,
        transform: Transform2D<f64>,
    ) {
        match self {
            #[cfg(feature = "vello")]
            Canvas::Vello(canvas_data) => canvas_data.fill_path(
                path,
                fill_rule,
                style,
                shadow_options,
                composition_options,
                transform,
            ),
            Canvas::VelloCPU(canvas_data) => canvas_data.fill_path(
                path,
                fill_rule,
                style,
                shadow_options,
                composition_options,
                transform,
            ),
            Canvas::VelloGpu(canvas_data) => canvas_data.fill_path(
                path,
                fill_rule,
                style,
                shadow_options,
                composition_options,
                transform,
            ),
        }
    }

    fn stroke_path(
        &mut self,
        path: &Path,
        style: FillOrStrokeStyle,
        line_options: LineOptions,
        shadow_options: ShadowOptions,
        composition_options: CompositionOptions,
        transform: Transform2D<f64>,
    ) {
        match self {
            #[cfg(feature = "vello")]
            Canvas::Vello(canvas_data) => canvas_data.stroke_path(
                path,
                style,
                line_options,
                shadow_options,
                composition_options,
                transform,
            ),
            Canvas::VelloCPU(canvas_data) => canvas_data.stroke_path(
                path,
                style,
                line_options,
                shadow_options,
                composition_options,
                transform,
            ),
            Canvas::VelloGpu(canvas_data) => canvas_data.stroke_path(
                path,
                style,
                line_options,
                shadow_options,
                composition_options,
                transform,
            ),
        }
    }

    fn clear_rect(&mut self, rect: &Rect<f32>, transform: Transform2D<f64>) {
        match self {
            #[cfg(feature = "vello")]
            Canvas::Vello(canvas_data) => canvas_data.clear_rect(rect, transform),
            Canvas::VelloCPU(canvas_data) => canvas_data.clear_rect(rect, transform),
            Canvas::VelloGpu(canvas_data) => canvas_data.clear_rect(rect, transform),
        }
    }

    #[expect(clippy::too_many_arguments)]
    fn draw_image(
        &mut self,
        snapshot: Snapshot,
        dest_rect: Rect<f64>,
        source_rect: Rect<f64>,
        smoothing_enabled: bool,
        shadow_options: ShadowOptions,
        composition_options: CompositionOptions,
        transform: Transform2D<f64>,
    ) {
        match self {
            #[cfg(feature = "vello")]
            Canvas::Vello(canvas_data) => canvas_data.draw_image(
                snapshot,
                dest_rect,
                source_rect,
                smoothing_enabled,
                shadow_options,
                composition_options,
                transform,
            ),
            Canvas::VelloCPU(canvas_data) => canvas_data.draw_image(
                snapshot,
                dest_rect,
                source_rect,
                smoothing_enabled,
                shadow_options,
                composition_options,
                transform,
            ),
            Canvas::VelloGpu(canvas_data) => canvas_data.draw_image(
                snapshot,
                dest_rect,
                source_rect,
                smoothing_enabled,
                shadow_options,
                composition_options,
                transform,
            ),
        }
    }

    fn is_gpu(&self) -> bool {
        matches!(self, Canvas::VelloGpu(_))
    }

    /// This canvas's current content as a GPU frame, if it is a GPU canvas.
    fn gpu_frame(&mut self) -> Option<std::rc::Rc<crate::vello_gpu_backend::GpuFrame>> {
        match self {
            Canvas::VelloGpu(canvas_data) => Some(canvas_data.gpu_frame()),
            _ => None,
        }
    }

    /// Draw another GPU canvas's frame. Only called on GPU canvases.
    #[expect(clippy::too_many_arguments)]
    fn draw_gpu_frame(
        &mut self,
        frame: std::rc::Rc<crate::vello_gpu_backend::GpuFrame>,
        dest_rect: Rect<f64>,
        source_rect: Rect<f64>,
        smoothing_enabled: bool,
        shadow_options: ShadowOptions,
        composition_options: CompositionOptions,
        transform: Transform2D<f64>,
    ) {
        if let Canvas::VelloGpu(canvas_data) = self {
            canvas_data.draw_gpu_frame(
                frame,
                dest_rect,
                source_rect,
                smoothing_enabled,
                shadow_options,
                composition_options,
                transform,
            );
        }
    }

    fn read_pixels(&mut self, read_rect: Option<Rect<u32>>) -> Snapshot {
        match self {
            #[cfg(feature = "vello")]
            Canvas::Vello(canvas_data) => canvas_data.read_pixels(read_rect),
            Canvas::VelloCPU(canvas_data) => canvas_data.read_pixels(read_rect),
            Canvas::VelloGpu(canvas_data) => canvas_data.read_pixels(read_rect),
        }
    }

    fn clip_path(&mut self, path: &Path, fill_rule: FillRule, transform: Transform2D<f64>) {
        match self {
            #[cfg(feature = "vello")]
            Canvas::Vello(canvas_data) => canvas_data.clip_path(path, fill_rule, transform),
            Canvas::VelloCPU(canvas_data) => canvas_data.clip_path(path, fill_rule, transform),
            Canvas::VelloGpu(canvas_data) => canvas_data.clip_path(path, fill_rule, transform),
        }
    }

    fn put_image_data(&mut self, snapshot: Snapshot, rect: Rect<u32>) {
        match self {
            #[cfg(feature = "vello")]
            Canvas::Vello(canvas_data) => canvas_data.put_image_data(snapshot, rect),
            Canvas::VelloCPU(canvas_data) => canvas_data.put_image_data(snapshot, rect),
            Canvas::VelloGpu(canvas_data) => canvas_data.put_image_data(snapshot, rect),
        }
    }

    fn update_image_rendering(&mut self, canvas_epoch: Option<Epoch>) {
        match self {
            #[cfg(feature = "vello")]
            Canvas::Vello(canvas_data) => canvas_data.update_image_rendering(canvas_epoch),
            Canvas::VelloCPU(canvas_data) => canvas_data.update_image_rendering(canvas_epoch),
            Canvas::VelloGpu(canvas_data) => canvas_data.update_image_rendering(canvas_epoch),
        }
    }

    fn recreate(&mut self, size: Option<Size2D<u64>>) {
        if self.recreate_with_backend_choice(size) {
            return;
        }
        match self {
            #[cfg(feature = "vello")]
            Canvas::Vello(canvas_data) => canvas_data.recreate(size),
            Canvas::VelloCPU(canvas_data) => canvas_data.recreate(size),
            Canvas::VelloGpu(canvas_data) => canvas_data.recreate(size),
        }
    }
}

#[cfg(test)]
mod canvas_tests {
    use euclid::default::{Rect, Size2D, Transform2D};
    use paint_api::CrossProcessPaintApi;
    use servo_canvas_traits::canvas::{
        CompositionOptions, CompositionOrBlending, CompositionStyle, FillOrStrokeStyle,
        ShadowOptions,
    };
    use style::color::AbsoluteColor;

    use super::Canvas;
    use crate::canvas_data::CanvasData;
    use crate::gpu_parity_tests::gpu_shared;

    fn fill(canvas: &mut Canvas, color: (u8, u8, u8)) {
        let (r, g, b) = color;
        canvas.fill_rect(
            &Rect::from_size(Size2D::new(64.0, 64.0)),
            FillOrStrokeStyle::Color(AbsoluteColor::srgb_legacy(r, g, b, 1.0)),
            ShadowOptions {
                offset_x: 0.0,
                offset_y: 0.0,
                blur: 0.0,
                color: AbsoluteColor::TRANSPARENT_BLACK,
            },
            CompositionOptions {
                alpha: 1.0,
                composition_operation: CompositionOrBlending::Composition(
                    CompositionStyle::SourceOver,
                ),
            },
            Transform2D::identity(),
        );
    }

    fn center(canvas: &mut Canvas) -> [u8; 4] {
        let pixels = canvas.read_pixels(None);
        let bytes = pixels.as_raw_bytes();
        let offset = (32 * 64 + 32) * 4;
        bytes[offset..offset + 4].try_into().unwrap()
    }

    /// A CPU canvas of 512x512 or more rasterizes on worker threads, where `vello_cpu` has no
    /// filter layers and panics on one. A shadowed draw there must still draw its shape.
    #[test]
    fn shadow_on_a_large_cpu_canvas_draws_the_shape() {
        let size = Size2D::new(600, 600);
        let mut canvas = Canvas::VelloCPU(CanvasData::new(size, CrossProcessPaintApi::dummy()));
        canvas.fill_rect(
            &Rect::new((200.0, 200.0).into(), Size2D::new(200.0, 200.0)),
            FillOrStrokeStyle::Color(AbsoluteColor::srgb_legacy(40, 120, 220, 1.0)),
            ShadowOptions {
                offset_x: 8.0,
                offset_y: 8.0,
                blur: 6.0,
                color: AbsoluteColor::srgb_legacy(0, 0, 0, 0.6),
            },
            CompositionOptions {
                alpha: 1.0,
                composition_operation: CompositionOrBlending::Composition(
                    CompositionStyle::SourceOver,
                ),
            },
            Transform2D::identity(),
        );
        let pixels = canvas.read_pixels(None);
        let offset = (300 * 600 + 300) * 4;
        assert_eq!(&pixels.as_raw_bytes()[offset..offset + 4], &[40, 120, 220, 255]);
    }

    /// A GPU canvas whose GL context is lost must neither panic nor take the canvas thread
    /// down. It moves to the CPU backend, where it draws again, and new canvases start there.
    #[test]
    fn lost_gpu_context_moves_canvases_to_the_cpu_backend() {
        lose_context_and_recover("gpu-context-loss", false);
    }

    /// The same when the driver loses the context and nothing tells the backend: it has to
    /// notice that its draws stopped working.
    #[test]
    fn context_lost_in_the_driver_is_noticed() {
        lose_context_and_recover("gpu-context-loss-driver", true);
    }

    fn lose_context_and_recover(thread_name: &'static str, in_driver: bool) {
        let supported = surfman::Connection::new()
            .map(|connection| connection.gl_api() == surfman::GLApi::GLES)
            .unwrap_or(false);
        if !supported {
            eprintln!("{thread_name}: skipped; the GPU canvas backend needs OpenGL ES");
            return;
        }
        let (sender, receiver) = std::sync::mpsc::channel();
        // A thread of its own: the loss is permanent for the thread, and the other GPU tests
        // share theirs. It never exits, because ANGLE can deadlock when a thread that made a
        // context current does.
        std::thread::Builder::new()
            .name(thread_name.into())
            .spawn(move || {
                let result = std::panic::catch_unwind(|| {
                    // Not in one expression: `set` would wait for the guard `get` returns.
                    let mut preferences = servo_config::prefs::get().clone();
                    preferences.dom_canvas_backend = "gpu".into();
                    preferences.dom_canvas_gpu_min_pixels = 0;
                    servo_config::prefs::set(preferences);
                    crate::vello_gpu_backend::install_shared(gpu_shared());
                    let size = Size2D::new(64, 64);
                    let mut canvas =
                        Canvas::new(size, CrossProcessPaintApi::dummy()).expect("canvas");
                    assert!(canvas.is_gpu(), "the canvas starts on the GPU");
                    fill(&mut canvas, (10, 200, 30));
                    assert_eq!(center(&mut canvas), [10, 200, 30, 255]);

                    if !in_driver {
                        crate::vello_gpu_backend::simulate_context_loss();
                    } else if !crate::vello_gpu_backend::lose_context_in_driver() {
                        eprintln!("{thread_name}: skipped; no GL_CHROMIUM_lose_context");
                        return;
                    }
                    // Still a GPU canvas until the thread reacts: drawing and reading must
                    // be harmless.
                    fill(&mut canvas, (200, 10, 30));
                    let after_loss = center(&mut canvas);
                    if !in_driver {
                        assert_eq!(after_loss, [0, 0, 0, 0], "the content is gone");
                    }

                    assert!(
                        crate::vello_gpu_backend::take_context_loss(),
                        "the loss was noticed"
                    );
                    assert!(
                        !crate::vello_gpu_backend::take_context_loss(),
                        "reported once"
                    );
                    assert!(canvas.recreate_with_backend_choice(None));
                    assert!(!canvas.is_gpu(), "the canvas moved to the CPU backend");
                    let other = Canvas::new(size, CrossProcessPaintApi::dummy()).expect("canvas");
                    assert!(!other.is_gpu(), "new canvases start on the CPU backend");
                    fill(&mut canvas, (30, 10, 200));
                    assert_eq!(center(&mut canvas), [30, 10, 200, 255]);
                });
                let _ = sender.send(result.is_ok());
                loop {
                    std::thread::park();
                }
            })
            .expect("spawn");
        assert!(receiver.recv().expect("the thread reported"), "see the panic above");
    }
}
