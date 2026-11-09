/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::borrow::ToOwned;
use std::{f32, thread};

use crossbeam_channel::{Sender, select, unbounded};
use euclid::default::{Rect, Size2D, Transform2D};
use log::warn;
use paint_api::CrossProcessPaintApi;
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
    pub fn start(
        paint_api: CrossProcessPaintApi,
        mem_profiler_chan: ProfilerChan,
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
                let mut canvas_paint_thread = CanvasPaintThread::new(
                    paint_api);
                loop {
                    select! {
                        recv(msg_receiver) -> msg => {
                            match msg {
                                Ok(Ok((canvas_id, command))) => {
                                    canvas_paint_thread.process_command(command, canvas_id);
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
