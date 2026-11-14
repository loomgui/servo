/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! CPU-vs-GPU canvas benchmarks (Loom). Ignored by default; run optimized:
//!
//! `cargo test --release -p servo-canvas --features surfman/sm-angle-default gpu_bench -- --ignored --nocapture`
//!
//! Each workload replays the same per-frame command stream on both backends and presents every
//! frame the way `CanvasData::update_image_rendering` does. Reported per frame:
//!
//! - `cpu`: canvas-thread time on the `vello_cpu` backend, including the copy of the frame into
//!   shared memory for WebRender.
//! - `gpu submit`: canvas-thread time on the GPU backend (recording, upload, draw submission).
//! - `gpu+finish`: the same with `glFinish` after every frame, so it includes GPU execution.

use std::time::{Duration, Instant};

use super::*;

const WARMUP_FRAMES: u32 = 30;

/// Timed frames per run; `GPU_BENCH_FRAMES` overrides (long runs for profiling).
fn frames() -> u32 {
    std::env::var("GPU_BENCH_FRAMES")
        .ok()
        .and_then(|frames| frames.parse().ok())
        .unwrap_or(240)
}

/// A deterministic xorshift stream in `[0, 1)`.
struct Rng(u32);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 % 100_000) as f64 / 100_000.0
    }
}

/// The hands-on main menu's ember field at 1920×1080: ~116 embers, each a radial-gradient glow
/// with frame-dependent alpha plus a solid core, composited with `lighter` after a full clear.
fn embers<T: GenericDrawTarget>(t: &mut T, frame: u32) {
    let (w, h) = (1920.0, 1080.0);
    t.clear_rect(&rect(0.0, 0.0, w as f32, h as f32), identity());
    let lighter = with_op(CompositionOrBlending::Composition(
        CompositionStyle::Lighter,
    ));
    let mut rng = Rng(0x2545_f491);
    for i in 0..116 {
        let phase = (frame as f64 * 0.01 + rng.next()) % 1.0;
        let x = rng.next() * w + (frame as f64 * 0.3 + i as f64).sin() * 20.0;
        let y = h - phase * h;
        let heat = 1.0 - phase;
        let size = (1.0 + rng.next() * 3.0) * (0.6 + heat * 0.6);
        let alpha = ((phase * std::f64::consts::PI).sin() * 0.95) as f32;
        let r = size * 3.2;
        let glow = FillOrStrokeStyle::RadialGradient(RadialGradientStyle::new(
            x,
            y,
            0.0,
            x,
            y,
            r,
            vec![
                stop(0.0, 255, (150.0 + 70.0 * heat) as u8, 60, alpha),
                stop(0.4, 244, 150, 60, alpha * 0.4),
                stop(1.0, 150, 50, 20, 0.0),
            ],
        ));
        t.fill(
            &circle(x, y, r),
            FillRule::Nonzero,
            glow,
            lighter,
            identity(),
        );
        t.fill(
            &circle(x, y, size * 0.7),
            FillRule::Nonzero,
            color(255, 240, 220, alpha),
            lighter,
            identity(),
        );
    }
}

/// 260 curved shapes on a 1920×1080 canvas, redrawn every frame: a third filled, all stroked
/// with varying widths and joins, a quarter dashed. This is the `paths` workload of Loom's
/// canvas benchmark, where preparing the paths dominates.
fn paths<T: GenericDrawTarget>(t: &mut T, frame: u32) {
    t.clear_rect(&rect(0.0, 0.0, 1920.0, 1080.0), identity());
    let mut rng = Rng(0x9e37_79b9);
    let time = f64::from(frame) / 60.0;
    for i in 0..260_u32 {
        let (x, y) = (rng.next() * 1920.0, rng.next() * 1080.0);
        let wobble = (time * 2.0 + f64::from(i)).sin() * 30.0;
        let mut path = Path::new();
        path.move_to(x, y);
        path.bezier_curve_to(
            x + 80.0,
            y - 60.0 + wobble,
            x + 160.0,
            y + 60.0 - wobble,
            x + 240.0,
            y,
        );
        path.line_to(x + 200.0, y + 70.0);
        path.quadratic_curve_to(x + 120.0, y + 110.0 + wobble, x + 20.0, y + 60.0);
        path.close_path();
        if i % 3 == 0 {
            t.fill(
                &path,
                FillRule::Nonzero,
                color((i * 37 % 255) as u8, 150, (255 - i * 53 % 255) as u8, 0.55),
                normal(),
                identity(),
            );
        }
        let join = [
            LineJoinStyle::Miter,
            LineJoinStyle::Round,
            LineJoinStyle::Bevel,
        ][(i % 3) as usize];
        let dash = if i % 4 == 0 { vec![12.0, 6.0] } else { vec![] };
        t.stroke(
            &path,
            color((i * 53 % 255) as u8, 200, (i * 91 % 255) as u8, 0.9),
            line(1.0 + f64::from(i % 5), LineCapStyle::Butt, join, dash),
            normal(),
            identity(),
        );
    }
}

/// One player's HUD panel (400×140), redrawn every frame: background, health and energy bars
/// with gradients, a score and labels, and a row of item icons.
fn hud_panel<T: GenericDrawTarget>(t: &mut T, frame: u32, player: u32, icon: &pixels::Snapshot) {
    t.clear_rect(&rect(0.0, 0.0, 400.0, 140.0), identity());
    t.fill_rect(
        &rect(0.0, 0.0, 400.0, 140.0),
        color(12, 14, 22, 0.85),
        normal(),
        identity(),
    );
    let health = 0.5 + 0.5 * ((frame + player * 40) as f64 * 0.05).sin();
    let energy = 0.5 + 0.5 * ((frame + player * 17) as f64 * 0.08).cos();
    for (row, value, from, to) in [
        (0, health, (220, 40, 40), (250, 160, 60)),
        (1, energy, (40, 120, 230), (90, 220, 250)),
    ] {
        let y = 52.0 + row as f32 * 26.0;
        t.fill_rect(
            &rect(12.0, y, 376.0, 18.0),
            color(40, 44, 60, 1.0),
            normal(),
            identity(),
        );
        let gradient = FillOrStrokeStyle::LinearGradient(LinearGradientStyle::new(
            12.0,
            0.0,
            388.0,
            0.0,
            vec![
                stop(0.0, from.0, from.1, from.2, 1.0),
                stop(1.0, to.0, to.1, to.2, 1.0),
            ],
        ));
        t.fill_rect(
            &rect(12.0, y, (376.0 * value) as f32, 18.0),
            gradient,
            normal(),
            identity(),
        );
    }
    let score = format!("P{} {:06}", player + 1, frame * 37 + player * 1000);
    t.fill_text(
        vec![text_run(&score, 12.0, 34.0, 24.0)],
        color(240, 230, 200, 1.0),
        normal(),
        identity(),
    );
    t.fill_text(
        vec![
            text_run("HP", 340.0, 66.0, 13.0),
            text_run("EN", 340.0, 92.0, 13.0),
        ],
        color(255, 255, 255, 0.9),
        normal(),
        identity(),
    );
    for slot in 0..6 {
        let x = 12.0 + slot as f64 * 34.0;
        let surface = t
            .create_source_surface_from_data(icon.clone())
            .expect("icon");
        t.draw_surface(
            surface,
            frect(x, 106.0, 28.0, 28.0),
            frect(0.0, 0.0, 32.0, 32.0),
            Filter::Bilinear,
            normal(),
            identity(),
        );
    }
}

/// 500 sprites (four 32×32 images) moving over a 1920×1080 canvas.
fn sprites<T: GenericDrawTarget>(t: &mut T, frame: u32, images: &[pixels::Snapshot]) {
    t.clear_rect(&rect(0.0, 0.0, 1920.0, 1080.0), identity());
    let mut rng = Rng(0x9e37_79b9);
    for i in 0..500 {
        let x = (rng.next() * 1920.0 + frame as f64 * (1.0 + rng.next() * 3.0)) % 1920.0;
        let y = rng.next() * 1050.0;
        let image = images[i % images.len()].clone();
        let surface = t.create_source_surface_from_data(image).expect("sprite");
        t.draw_surface(
            surface,
            frect(x, y, 32.0, 32.0),
            frect(0.0, 0.0, 32.0, 32.0),
            Filter::Bilinear,
            normal(),
            identity(),
        );
    }
}

/// Loom canvas benchmark `compositing`: a background, 90 rectangles in six blend and composite
/// modes inside a circular clip, and 12 rectangles with shadows.
fn compositing<T: GenericDrawTarget>(t: &mut T, frame: u32) {
    let (w, h) = (1920.0_f64, 1080.0_f64);
    let time = f64::from(frame) / 60.0;
    t.clear_rect(&rect(0.0, 0.0, w as f32, h as f32), identity());
    t.fill_rect(
        &rect(0.0, 0.0, w as f32, h as f32),
        color(0x24, 0x34, 0x47, 1.0),
        normal(),
        identity(),
    );
    t.push_clip(
        &circle(w / 2.0, h / 2.0, 500.0),
        FillRule::Nonzero,
        identity(),
    );
    let modes = [
        CompositionOrBlending::Blending(BlendingStyle::Multiply),
        CompositionOrBlending::Blending(BlendingStyle::Screen),
        CompositionOrBlending::Blending(BlendingStyle::Overlay),
        CompositionOrBlending::Composition(CompositionStyle::Lighter),
        CompositionOrBlending::Composition(CompositionStyle::DestinationOut),
        CompositionOrBlending::Composition(CompositionStyle::Xor),
    ];
    for i in 0..90_u32 {
        let x = w / 2.0 + (time * 0.7 + f64::from(i) * 0.5).cos() * 380.0;
        let y = h / 2.0 + (time * 0.9 + f64::from(i) * 0.8).sin() * 300.0;
        let hue = (i * 47 + frame) % 360;
        t.fill_rect(
            &rect((x - 120.0) as f32, (y - 90.0) as f32, 240.0, 180.0),
            color((hue % 256) as u8, (255 - hue % 200) as u8, 120, 0.7),
            with_op(modes[i as usize % modes.len()]),
            identity(),
        );
    }
    t.pop_clip();
    let shadow_options = ShadowOptions {
        offset_x: 0.0,
        offset_y: 8.0,
        blur: 16.0,
        color: match color(0, 0, 0, 0.6) {
            FillOrStrokeStyle::Color(color) => color,
            _ => unreachable!(),
        },
    };
    for i in 0..12_u32 {
        let y = 60.0 + (time + f64::from(i)).sin() * 20.0;
        let shadowed = t.push_shadow(&shadow_options);
        t.fill_rect(
            &rect((80 + i * 150) as f32, y as f32, 110.0, 70.0),
            color(0xff, 0xd6, 0x78, 1.0),
            normal(),
            identity(),
        );
        if shadowed {
            t.pop_shadow();
        }
    }
}

/// Loom canvas benchmark `text`: 216 short strings in three sizes that change every frame, and
/// one stroked line.
fn text<T: GenericDrawTarget>(t: &mut T, frame: u32) {
    t.clear_rect(&rect(0.0, 0.0, 1920.0, 1080.0), identity());
    for row in 0..36_u32 {
        for column in 0..6_u32 {
            let size = 14.0 + (column % 3) as f32 * 6.0;
            let label = format!("Score {}", (frame * (row + 1) + column * 977) % 100_000);
            t.fill_text(
                vec![text_run(
                    &label,
                    20.0 + column as f32 * 316.0,
                    28.0 + row as f32 * 29.0,
                    size,
                )],
                color(0xe8, 0xea, 0xed, 1.0),
                normal(),
                identity(),
            );
        }
    }
    t.stroke_text(
        vec![text_run(&format!("Frame {frame}"), 640.0, 560.0, 64.0)],
        color(0xff, 0xd6, 0x78, 1.0),
        line(1.5, LineCapStyle::Butt, LineJoinStyle::Miter, vec![]),
        normal(),
        identity(),
    );
}

/// Loom canvas benchmark `particles` content on a `width`×`height` canvas: `count` radial
/// glows added together.
fn glows<T: GenericDrawTarget>(t: &mut T, time: f64, count: u32, width: f64, height: f64) {
    t.clear_rect(&rect(0.0, 0.0, width as f32, height as f32), identity());
    let lighter = with_op(CompositionOrBlending::Composition(
        CompositionStyle::Lighter,
    ));
    for i in 0..count {
        let fi = f64::from(i);
        let phase = (time * (0.05 + f64::from(i % 7) * 0.01) + fi * 0.137) % 1.0;
        let x = ((i * 97) as f64 % width) + (time + fi).sin() * 24.0;
        let y = height - phase * height;
        let radius = 18.0 + f64::from(i % 5) * 9.0;
        let alpha = ((phase * std::f64::consts::PI).sin() * 0.9) as f32;
        let glow = FillOrStrokeStyle::RadialGradient(RadialGradientStyle::new(
            x,
            y,
            0.0,
            x,
            y,
            radius,
            vec![
                stop(0.0, 255, 220, 150, alpha),
                stop(0.4, 244, 150, 60, alpha * 0.4),
                stop(1.0, 150, 50, 20, 0.0),
            ],
        ));
        t.fill(
            &circle(x, y, radius),
            FillRule::Nonzero,
            glow,
            lighter,
            identity(),
        );
    }
}

struct Timing {
    per_frame: Duration,
}

/// Run `frames` frames of `workload` against `targets`, presenting each target every frame.
fn run<T: GenericDrawTarget>(
    targets: &mut [T],
    mut workload: impl FnMut(&mut [T], u32),
    mut after_frame: impl FnMut(&mut [T]),
) -> Timing {
    for frame in 0..WARMUP_FRAMES {
        workload(targets, frame);
        for target in targets.iter_mut() {
            let _ = target.image_descriptor_and_serializable_data();
        }
        after_frame(targets);
    }
    let frames = frames();
    let start = Instant::now();
    let mut recording = Duration::ZERO;
    for frame in WARMUP_FRAMES..WARMUP_FRAMES + frames {
        let frame_start = Instant::now();
        workload(targets, frame);
        recording += frame_start.elapsed();
        for target in targets.iter_mut() {
            let _ = target.image_descriptor_and_serializable_data();
        }
        after_frame(targets);
    }
    // `GPU_BENCH_SPLIT=1`: how much of a frame is the page's drawing calls, the rest being the
    // presentation (rasterization on the CPU backend, submission on the GPU backend).
    if std::env::var_os("GPU_BENCH_SPLIT").is_some() {
        eprintln!(
            "    drawing calls {:.2} ms of {:.2} ms per frame",
            ms(recording / frames),
            ms(start.elapsed() / frames)
        );
    }
    Timing {
        per_frame: start.elapsed() / frames,
    }
}

#[derive(Clone, Copy)]
enum Workload {
    Embers,
    Paths,
    Hud,
    Sprites,
    Compositing,
    Text,
    Many,
}

impl Workload {
    fn key(self) -> &'static str {
        match self {
            Self::Embers => "embers",
            Self::Paths => "paths",
            Self::Hud => "hud",
            Self::Sprites => "sprites",
            Self::Compositing => "compositing",
            Self::Text => "text",
            Self::Many => "many",
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Embers => "embers 1920x1080 (116 gradient embers, lighter)",
            Self::Paths => "paths 1920x1080 (260 curves, filled and stroked)",
            Self::Hud => "hud 4x 400x140 (bars, text, 6 icons each)",
            Self::Sprites => "sprites 1920x1080 (500 drawImage)",
            Self::Compositing => "compositing 1920x1080 (90 blends in a clip, 12 shadows)",
            Self::Text => "text 1920x1080 (216 strings, 1 stroked)",
            Self::Many => "many 6x 640x540 (20 glows each)",
        }
    }

    fn targets<T: GenericDrawTarget>(self) -> Vec<T> {
        match self {
            Self::Embers | Self::Paths | Self::Sprites | Self::Compositing | Self::Text => {
                vec![T::new(Size2D::new(1920, 1080))]
            },
            Self::Many => (0..6).map(|_| T::new(Size2D::new(640, 540))).collect(),
            Self::Hud => (0..4).map(|_| T::new(Size2D::new(400, 140))).collect(),
        }
    }

    fn frame<T: GenericDrawTarget>(
        self,
        targets: &mut [T],
        frame: u32,
        images: &[pixels::Snapshot],
    ) {
        match self {
            Self::Embers => embers(&mut targets[0], frame),
            Self::Paths => paths(&mut targets[0], frame),
            Self::Hud => {
                for (player, target) in targets.iter_mut().enumerate() {
                    hud_panel(target, frame, player as u32, &images[0]);
                }
            },
            Self::Sprites => sprites(&mut targets[0], frame, images),
            Self::Compositing => compositing(&mut targets[0], frame),
            Self::Text => text(&mut targets[0], frame),
            Self::Many => {
                for (i, target) in targets.iter_mut().enumerate() {
                    glows(target, f64::from(frame) / 60.0 + i as f64, 20, 640.0, 540.0);
                }
            },
        }
    }
}

fn images() -> Vec<pixels::Snapshot> {
    (0..4)
        .map(|i| test_image(32 - i, 32 - i))
        .map(|image| {
            // Normalize every sprite to 32×32 so draw rects are identical across images.
            let mut data = image.as_raw_bytes().to_vec();
            data.resize(32 * 32 * 4, 0);
            pixels::Snapshot::from_vec(
                Size2D::new(32, 32),
                pixels::SnapshotPixelFormat::RGBA,
                pixels::SnapshotAlphaMode::Transparent {
                    premultiplied: true,
                },
                data,
            )
        })
        .collect()
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

/// One frame of a page that composes an offscreen canvas into a visible one: redraw the
/// 1920×1080 source, then draw all of it into the 1920×1080 destination and present that.
fn layer_frame<T: GenericDrawTarget>(
    source: &mut T,
    dest: &mut T,
    frame: u32,
    draw_source: impl FnOnce(&mut T, &mut T),
) {
    source.clear_rect(&rect(0.0, 0.0, 1920.0, 1080.0), identity());
    for i in 0..40_u32 {
        let x = ((i * 131 + frame * 7) % 1800) as f32;
        let y = ((i * 71 + frame * 3) % 1000) as f32;
        source.fill_rect(
            &rect(x, y, 120.0, 80.0),
            color((i * 6) as u8, 200, (255 - i * 5) as u8, 0.8),
            normal(),
            identity(),
        );
    }
    dest.clear_rect(&rect(0.0, 0.0, 1920.0, 1080.0), identity());
    draw_source(dest, source);
    let _ = dest.image_descriptor_and_serializable_data();
}

/// What drawing one full-screen canvas into another costs per frame on the canvas thread.
#[test]
#[ignore = "benchmark; run with --release -- --ignored --nocapture"]
fn gpu_bench_canvas_to_canvas() {
    let size = Size2D::new(1920, 1080);
    let whole = urect(0, 0, 1920, 1080);
    let place = frect(0.0, 0.0, 1920.0, 1080.0);
    let frames = frames();

    let mut source = VelloCPUDrawTarget::new(size);
    let mut dest = VelloCPUDrawTarget::new(size);
    let mut run = |frame| {
        layer_frame(&mut source, &mut dest, frame, |dest, source| {
            dest.draw_canvas(source, whole, place, Filter::Nearest, normal(), identity());
        })
    };
    (0..WARMUP_FRAMES).for_each(&mut run);
    let start = Instant::now();
    (WARMUP_FRAMES..WARMUP_FRAMES + frames).for_each(&mut run);
    let cpu = start.elapsed() / frames;

    let gpu = on_gpu_thread(move || {
        let mut timings = Vec::new();
        for through_pixels in [true, false] {
            let mut source = VelloGpuDrawTarget::new(size);
            let mut dest = VelloGpuDrawTarget::new(size);
            let mut run = |frame| {
                layer_frame(&mut source, &mut dest, frame, |dest, source| {
                    if through_pixels {
                        // What a canvas source cost before: read back, hash and upload.
                        let pixels = source.snapshot();
                        draw_image(dest, pixels, place, Filter::Nearest, normal(), identity());
                    } else {
                        dest.draw_canvas(
                            source,
                            whole,
                            place,
                            Filter::Nearest,
                            normal(),
                            identity(),
                        );
                    }
                });
                dest.finish_gpu();
            };
            (0..WARMUP_FRAMES).for_each(&mut run);
            let start = Instant::now();
            (WARMUP_FRAMES..WARMUP_FRAMES + frames).for_each(&mut run);
            timings.extend_from_slice(&(start.elapsed() / frames).as_nanos().to_le_bytes());
        }
        timings
    });
    let nanos = |index: usize| {
        let bytes: [u8; 16] = gpu[index * 16..][..16].try_into().unwrap();
        Duration::from_nanos(u128::from_le_bytes(bytes) as u64)
    };
    println!(
        "\n1920x1080 canvas drawn into a 1920x1080 canvas, per frame, GPU times include glFinish\n\
         cpu backend                      {:>7.2} ms\n\
         gpu backend, through pixels      {:>7.2} ms\n\
         gpu backend, through the frame   {:>7.2} ms\n",
        ms(cpu),
        ms(nanos(0)),
        ms(nanos(1)),
    );
}

/// One frame of a page that reads its canvas back: a cheap scene, the presentation, and
/// `getImageData` of `read`.
fn readback_frame<T: GenericDrawTarget>(target: &mut T, frame: u32, read: Rect<u32>) -> usize {
    target.fill_rect(
        &rect(0.0, 0.0, 1920.0, 1080.0),
        color(16, 24, 32, 1.0),
        normal(),
        identity(),
    );
    for i in 0..40_u32 {
        let x = ((i * 131 + frame * 7) % 1800) as f32;
        let y = ((i * 71 + frame * 3) % 1000) as f32;
        target.fill_rect(
            &rect(x, y, 90.0, 60.0),
            color((i * 6) as u8, 200, (255 - i * 5) as u8, 1.0),
            normal(),
            identity(),
        );
    }
    let _ = target.image_descriptor_and_serializable_data();
    if read.is_empty() {
        return 0;
    }
    target.snapshot_rect(read).as_raw_bytes().len()
}

/// What `getImageData` costs per frame on the canvas thread: a 1920×1080 canvas read back
/// as a 64×64 region and whole, against the same frame without a read.
#[test]
#[ignore = "benchmark; run with --release -- --ignored --nocapture"]
fn gpu_bench_readback() {
    let size = Size2D::new(1920, 1080);
    let reads = [
        urect(0, 0, 0, 0),
        urect(928, 508, 64, 64),
        urect(0, 0, 1920, 1080),
    ];
    let frames = frames();
    fn measure<T: GenericDrawTarget>(target: &mut T, read: Rect<u32>, frames: u32) -> Duration {
        for frame in 0..WARMUP_FRAMES {
            readback_frame(target, frame, read);
        }
        let start = Instant::now();
        for frame in WARMUP_FRAMES..WARMUP_FRAMES + frames {
            readback_frame(target, frame, read);
        }
        start.elapsed() / frames
    }
    let cpu: Vec<Duration> = reads
        .iter()
        .map(|read| measure(&mut VelloCPUDrawTarget::new(size), *read, frames))
        .collect();
    let gpu = on_gpu_thread(move || {
        let mut timings = Vec::new();
        for read in reads {
            let elapsed = measure(&mut VelloGpuDrawTarget::new(size), read, frames);
            timings.extend_from_slice(&elapsed.as_nanos().to_le_bytes());
        }
        timings
    });
    let nanos = |index: usize| {
        let bytes: [u8; 16] = gpu[index * 16..][..16].try_into().unwrap();
        Duration::from_nanos(u128::from_le_bytes(bytes) as u64)
    };
    let mut report = String::from("\n1920x1080 canvas, per frame      cpu ms    gpu ms\n");
    let names = ["no read", "read 64x64", "read 1920x1080"];
    for (index, name) in names.iter().enumerate() {
        report += &format!(
            "{:<28} {:>9.2} {:>9.2}\n",
            name,
            ms(cpu[index]),
            ms(nanos(index))
        );
    }
    eprintln!("{report}");
}

/// One canvas of `side`×`side` showing a HUD panel scaled to fill it: finds the size where the
/// GPU backend's fixed per-update cost stops dominating.
#[test]
#[ignore = "benchmark; run with --release -- --ignored --nocapture"]
fn gpu_bench_crossover() {
    let mut report = String::from(
        "
side    pixels    cpu ms   gpu+finish ms
",
    );
    for side in [128_u32, 192, 256, 320, 384, 512, 768, 1024] {
        let icon = images().remove(0);
        let scale = f64::from(side) / 400.0;
        let cpu = {
            let icon = icon.clone();
            let mut targets = vec![VelloCPUDrawTarget::new(Size2D::new(side, side))];
            run(
                &mut targets,
                |t, frame| scaled_panel(&mut t[0], frame, &icon, scale),
                |_| {},
            )
        };
        let gpu = on_gpu_thread(move || {
            let mut targets = vec![VelloGpuDrawTarget::new(Size2D::new(side, side))];
            let timing = run(
                &mut targets,
                |t, frame| scaled_panel(&mut t[0], frame, &icon, scale),
                |t| t[0].finish_gpu(),
            );
            timing.per_frame.as_nanos().to_string().into_bytes()
        });
        let gpu = Duration::from_nanos(String::from_utf8(gpu).unwrap().parse().unwrap());
        report += &format!(
            "{:<6} {:>7} {:>9.3} {:>15.3}
",
            side,
            side * side,
            ms(cpu.per_frame),
            ms(gpu)
        );
    }
    eprintln!("{report}");
}

/// The HUD panel content, stretched over a square canvas (`scale` = side / 400).
fn scaled_panel<T: GenericDrawTarget>(t: &mut T, frame: u32, icon: &pixels::Snapshot, scale: f64) {
    let side = (400.0 * scale) as f32;
    t.clear_rect(&rect(0.0, 0.0, side, side), identity());
    let s = Transform2D::scale(scale, scale * 400.0 / 140.0);
    t.fill_rect(
        &rect(0.0, 0.0, 400.0, 140.0),
        color(12, 14, 22, 0.85),
        normal(),
        s,
    );
    let health = 0.5 + 0.5 * (frame as f64 * 0.05).sin();
    t.fill_rect(
        &rect(12.0, 52.0, 376.0, 18.0),
        color(40, 44, 60, 1.0),
        normal(),
        s,
    );
    let gradient = FillOrStrokeStyle::LinearGradient(LinearGradientStyle::new(
        12.0,
        0.0,
        388.0,
        0.0,
        vec![stop(0.0, 220, 40, 40, 1.0), stop(1.0, 250, 160, 60, 1.0)],
    ));
    t.fill_rect(
        &rect(12.0, 52.0, (376.0 * health) as f32, 18.0),
        gradient,
        normal(),
        s,
    );
    let score = format!("P1 {:06}", frame * 37);
    t.fill_text(
        vec![text_run(&score, 12.0, 34.0, 24.0)],
        color(240, 230, 200, 1.0),
        normal(),
        s,
    );
    for slot in 0..6 {
        let surface = t
            .create_source_surface_from_data(icon.clone())
            .expect("icon");
        t.draw_surface(
            surface,
            frect(12.0 + slot as f64 * 34.0, 106.0, 28.0, 28.0),
            frect(0.0, 0.0, 32.0, 32.0),
            Filter::Bilinear,
            normal(),
            s,
        );
    }
}

#[test]
#[ignore = "benchmark; run with --release -- --ignored --nocapture"]
fn gpu_bench() {
    // `GPU_BENCH_ONLY=embers|hud|sprites` runs one workload; `GPU_BENCH_GPU_ONLY=1` skips the
    // CPU backend (for profiling the GPU path).
    let only = std::env::var("GPU_BENCH_ONLY").ok();
    let gpu_only = std::env::var_os("GPU_BENCH_GPU_ONLY").is_some();
    let workloads = [
        Workload::Embers,
        Workload::Paths,
        Workload::Hud,
        Workload::Sprites,
        Workload::Compositing,
        Workload::Text,
        Workload::Many,
    ]
        .into_iter()
        .filter(|workload| only.as_deref().is_none_or(|only| only == workload.key()));
    let mut report = String::from(
        "\nworkload                                           cpu ms   gpu submit ms   gpu+finish ms   speedup\n",
    );
    for workload in workloads {
        let images = images();
        let cpu = if gpu_only {
            Timing {
                per_frame: Duration::ZERO,
            }
        } else {
            let images = images.clone();
            let mut targets = workload.targets::<VelloCPUDrawTarget>();
            run(
                &mut targets,
                |targets, frame| workload.frame(targets, frame, &images),
                |_| {},
            )
        };
        let gpu = on_gpu_thread(move || {
            let mut targets = workload.targets::<VelloGpuDrawTarget>();
            let submit = run(
                &mut targets,
                |targets, frame| workload.frame(targets, frame, &images),
                |_| {},
            );
            let finish = run(
                &mut targets,
                |targets, frame| workload.frame(targets, frame, &images),
                |targets| targets[0].finish_gpu(),
            );
            format!(
                "{} {}",
                submit.per_frame.as_nanos(),
                finish.per_frame.as_nanos()
            )
            .into_bytes()
        });
        let gpu = String::from_utf8(gpu).unwrap();
        let mut parts = gpu
            .split(' ')
            .map(|n| Duration::from_nanos(n.parse().unwrap()));
        let (submit, finish) = (parts.next().unwrap(), parts.next().unwrap());
        report += &format!(
            "{:<50} {:>7.2} {:>15.2} {:>15.2} {:>8.1}x\n",
            workload.name(),
            ms(cpu.per_frame),
            ms(submit),
            ms(finish),
            ms(cpu.per_frame) / ms(finish)
        );
    }
    eprintln!("{report}");
}
