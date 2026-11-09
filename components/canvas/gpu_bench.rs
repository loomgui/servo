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
    let lighter = with_op(CompositionOrBlending::Composition(CompositionStyle::Lighter));
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
        t.fill(&circle(x, y, r), FillRule::Nonzero, glow, lighter, identity());
        t.fill(&circle(x, y, size * 0.7), FillRule::Nonzero, color(255, 240, 220, alpha), lighter, identity());
    }
}

/// One player's HUD panel (400×140), redrawn every frame: background, health and energy bars
/// with gradients, a score and labels, and a row of item icons.
fn hud_panel<T: GenericDrawTarget>(t: &mut T, frame: u32, player: u32, icon: &pixels::Snapshot) {
    t.clear_rect(&rect(0.0, 0.0, 400.0, 140.0), identity());
    t.fill_rect(&rect(0.0, 0.0, 400.0, 140.0), color(12, 14, 22, 0.85), normal(), identity());
    let health = 0.5 + 0.5 * ((frame + player * 40) as f64 * 0.05).sin();
    let energy = 0.5 + 0.5 * ((frame + player * 17) as f64 * 0.08).cos();
    for (row, value, from, to) in [(0, health, (220, 40, 40), (250, 160, 60)), (1, energy, (40, 120, 230), (90, 220, 250))] {
        let y = 52.0 + row as f32 * 26.0;
        t.fill_rect(&rect(12.0, y, 376.0, 18.0), color(40, 44, 60, 1.0), normal(), identity());
        let gradient = FillOrStrokeStyle::LinearGradient(LinearGradientStyle::new(
            12.0,
            0.0,
            388.0,
            0.0,
            vec![stop(0.0, from.0, from.1, from.2, 1.0), stop(1.0, to.0, to.1, to.2, 1.0)],
        ));
        t.fill_rect(&rect(12.0, y, (376.0 * value) as f32, 18.0), gradient, normal(), identity());
    }
    let score = format!("P{} {:06}", player + 1, frame * 37 + player * 1000);
    t.fill_text(vec![text_run(&score, 12.0, 34.0, 24.0)], color(240, 230, 200, 1.0), normal(), identity());
    t.fill_text(vec![text_run("HP", 340.0, 66.0, 13.0), text_run("EN", 340.0, 92.0, 13.0)], color(255, 255, 255, 0.9), normal(), identity());
    for slot in 0..6 {
        let x = 12.0 + slot as f64 * 34.0;
        let surface = t.create_source_surface_from_data(icon.clone()).expect("icon");
        t.draw_surface(surface, frect(x, 106.0, 28.0, 28.0), frect(0.0, 0.0, 32.0, 32.0), Filter::Bilinear, normal(), identity());
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
        t.draw_surface(surface, frect(x, y, 32.0, 32.0), frect(0.0, 0.0, 32.0, 32.0), Filter::Bilinear, normal(), identity());
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
    for frame in WARMUP_FRAMES..WARMUP_FRAMES + frames {
        workload(targets, frame);
        for target in targets.iter_mut() {
            let _ = target.image_descriptor_and_serializable_data();
        }
        after_frame(targets);
    }
    Timing {
        per_frame: start.elapsed() / frames,
    }
}

#[derive(Clone, Copy)]
enum Workload {
    Embers,
    Hud,
    Sprites,
}

impl Workload {
    fn key(self) -> &'static str {
        match self {
            Self::Embers => "embers",
            Self::Hud => "hud",
            Self::Sprites => "sprites",
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Embers => "embers 1920x1080 (116 gradient embers, lighter)",
            Self::Hud => "hud 4x 400x140 (bars, text, 6 icons each)",
            Self::Sprites => "sprites 1920x1080 (500 drawImage)",
        }
    }

    fn targets<T: GenericDrawTarget>(self) -> Vec<T> {
        match self {
            Self::Embers | Self::Sprites => vec![T::new(Size2D::new(1920, 1080))],
            Self::Hud => (0..4).map(|_| T::new(Size2D::new(400, 140))).collect(),
        }
    }

    fn frame<T: GenericDrawTarget>(self, targets: &mut [T], frame: u32, images: &[pixels::Snapshot]) {
        match self {
            Self::Embers => embers(&mut targets[0], frame),
            Self::Hud => {
                for (player, target) in targets.iter_mut().enumerate() {
                    hud_panel(target, frame, player as u32, &images[0]);
                }
            },
            Self::Sprites => sprites(&mut targets[0], frame, images),
        }
    }
}

fn images() -> Vec<pixels::Snapshot> {
    (0..4).map(|i| test_image(32 - i, 32 - i)).map(|image| {
        // Normalize every sprite to 32×32 so draw rects are identical across images.
        let mut data = image.as_raw_bytes().to_vec();
        data.resize(32 * 32 * 4, 0);
        pixels::Snapshot::from_vec(
            Size2D::new(32, 32),
            pixels::SnapshotPixelFormat::RGBA,
            pixels::SnapshotAlphaMode::Transparent { premultiplied: true },
            data,
        )
    }).collect()
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

#[test]
#[ignore = "benchmark; run with --release -- --ignored --nocapture"]
fn gpu_bench() {
    // `GPU_BENCH_ONLY=embers|hud|sprites` runs one workload; `GPU_BENCH_GPU_ONLY=1` skips the
    // CPU backend (for profiling the GPU path).
    let only = std::env::var("GPU_BENCH_ONLY").ok();
    let gpu_only = std::env::var_os("GPU_BENCH_GPU_ONLY").is_some();
    let workloads = [Workload::Embers, Workload::Hud, Workload::Sprites]
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
            run(&mut targets, |targets, frame| workload.frame(targets, frame, &images), |_| {})
        };
        let gpu = on_gpu_thread(move || {
            let mut targets = workload.targets::<VelloGpuDrawTarget>();
            let submit = run(&mut targets, |targets, frame| workload.frame(targets, frame, &images), |_| {});
            let finish = run(
                &mut targets,
                |targets, frame| workload.frame(targets, frame, &images),
                |targets| targets[0].finish_gpu(),
            );
            format!("{} {}", submit.per_frame.as_nanos(), finish.per_frame.as_nanos()).into_bytes()
        });
        let gpu = String::from_utf8(gpu).unwrap();
        let mut parts = gpu.split(' ').map(|n| Duration::from_nanos(n.parse().unwrap()));
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
