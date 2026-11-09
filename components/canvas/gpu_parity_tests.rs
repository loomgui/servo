/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! CPU-vs-GPU parity tests for the 2D canvas backends (Loom).
//!
//! Every scene runs the same command sequence through `VelloCPUDrawTarget` and
//! `VelloGpuDrawTarget`, snapshots both, and compares premultiplied RGBA pixels. A pixel matches
//! when every channel differs by at most `CHANNEL_TOLERANCE`; a scene passes when at most
//! `MISMATCH_BUDGET` of its pixels miss that bound (anti-aliasing and gradient dithering are not
//! bit-identical between the CPU and GPU fine rasterizers). Failing scenes write
//! `<name>-cpu.png`, `<name>-gpu.png` and `<name>-diff.png` to `target/gpu-parity/`; set
//! `GPU_PARITY_SAVE=1` to write the CPU and GPU images of every scene.
//!
//! The GPU half needs a real GPU. On Windows run with ANGLE, as Loom ships it:
//! `cargo test -p servo-canvas --features surfman/sm-angle-default gpu_parity`
//! with `libEGL.dll` / `libGLESv2.dll` on `PATH`.

use std::path::PathBuf;
use std::sync::Once;

use euclid::default::{Point2D, Rect, Size2D, Transform2D};
use fonts::{FontData, FontDataAndIndex, FontIdentifier};
use paint_api::{
    GpuCanvasShared, PainterSurfmanDetails, PainterSurfmanDetailsMap,
    WebRenderExternalImageIdManager,
};
use servo_base::id::PainterId;
use servo_canvas_traits::canvas::{
    BlendingStyle, CanvasGradientStop, CompositionOptions, CompositionOrBlending,
    CompositionStyle, FillOrStrokeStyle, FillRule, LineCapStyle, LineJoinStyle, LineOptions,
    CanvasFont, GlyphAndPosition, LinearGradientStyle, Path, RadialGradientStyle, TextRun,
};
use style::color::AbsoluteColor;

use crate::backend::GenericDrawTarget;
use crate::vello_cpu_backend::VelloCPUDrawTarget;
use crate::vello_gpu_backend::VelloGpuDrawTarget;

const CHANNEL_TOLERANCE: u8 = 3;
const MISMATCH_BUDGET: f64 = 0.002;

static INIT: Once = Once::new();

/// Publish a GPU canvas context the way the paint thread does in a running Servo.
fn init_gpu() {
    INIT.call_once(|| {
        let connection = surfman::Connection::new().expect("surfman connection");
        let adapter = connection
            .create_hardware_adapter()
            .expect("hardware adapter");
        let details = PainterSurfmanDetailsMap::default();
        details.insert(
            PainterId::next(),
            PainterSurfmanDetails {
                connection,
                adapter,
            },
        );
        paint_api::set_gpu_canvas_shared(GpuCanvasShared {
            surfman_details: details,
            swap_chains: surfman::chains::SwapChains::new(),
            external_image_id_manager: WebRenderExternalImageIdManager::default(),
        });
    });
}

type GpuJob = Box<dyn FnOnce() -> Vec<u8> + Send>;

/// Run GPU work on one long-lived thread, like Servo's canvas thread. ANGLE deadlocks when a
/// thread that made EGL contexts current exits, which test threads do after every test.
fn on_gpu_thread(job: impl FnOnce() -> Vec<u8> + Send + 'static) -> Vec<u8> {
    static SENDER: std::sync::OnceLock<
        std::sync::Mutex<std::sync::mpsc::Sender<(GpuJob, std::sync::mpsc::Sender<Vec<u8>>)>>,
    > = std::sync::OnceLock::new();
    let sender = SENDER.get_or_init(|| {
        init_gpu();
        let (sender, receiver) = std::sync::mpsc::channel::<(GpuJob, std::sync::mpsc::Sender<Vec<u8>>)>();
        std::thread::Builder::new()
            .name("gpu-parity".into())
            .spawn(move || {
                for (job, reply) in receiver {
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job))
                        .unwrap_or_default();
                    let _ = reply.send(result);
                }
            })
            .expect("spawn GPU thread");
        std::sync::Mutex::new(sender)
    });
    let (reply, result) = std::sync::mpsc::channel();
    sender
        .lock()
        .unwrap()
        .send((Box::new(job), reply))
        .expect("GPU thread alive");
    let pixels = result.recv().expect("GPU job finished");
    assert!(!pixels.is_empty(), "GPU scene panicked");
    pixels
}

fn out_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/gpu-parity");
    std::fs::create_dir_all(&dir).ok();
    dir.canonicalize().unwrap_or(dir)
}

fn save(name: &str, suffix: &str, size: Size2D<u32>, rgba: &[u8]) {
    let path = out_dir().join(format!("{name}-{suffix}.png"));
    if let Some(image) = image::RgbaImage::from_raw(size.width, size.height, rgba.to_vec()) {
        let _ = image.save(path);
    }
}

pub(crate) struct Comparison {
    pub(crate) mismatched: usize,
    pub(crate) total: usize,
    pub(crate) max_delta: u8,
}

pub(crate) fn compare_pixels(cpu: &[u8], gpu: &[u8]) -> Comparison {
    assert_eq!(cpu.len(), gpu.len(), "snapshot sizes differ");
    let mut mismatched = 0;
    let mut max_delta = 0;
    for (a, b) in cpu.chunks_exact(4).zip(gpu.chunks_exact(4)) {
        let delta = a
            .iter()
            .zip(b)
            .map(|(x, y)| x.abs_diff(*y))
            .max()
            .unwrap_or(0);
        max_delta = max_delta.max(delta);
        if delta > CHANNEL_TOLERANCE {
            mismatched += 1;
        }
    }
    Comparison {
        mismatched,
        total: cpu.len() / 4,
        max_delta,
    }
}

fn run_parity(
    name: &str,
    size: Size2D<u32>,
    cpu_scene: fn(&mut VelloCPUDrawTarget),
    gpu_scene: fn(&mut VelloGpuDrawTarget),
) {
    let mut cpu = VelloCPUDrawTarget::new(size);
    cpu_scene(&mut cpu);
    let cpu = cpu.snapshot();
    let gpu = on_gpu_thread(move || {
        let mut gpu = VelloGpuDrawTarget::new(size);
        gpu_scene(&mut gpu);
        gpu.snapshot().as_raw_bytes().to_vec()
    });
    let gpu = pixels::Snapshot::from_vec(
        size,
        pixels::SnapshotPixelFormat::RGBA,
        pixels::SnapshotAlphaMode::Transparent {
            premultiplied: true,
        },
        gpu,
    );

    // A scene that drew nothing would match trivially; `clear` legitimately wipes the canvas.
    assert!(
        name == "composite_clear" || cpu.as_raw_bytes().chunks_exact(4).any(|pixel| pixel[3] != 0),
        "{name}: the CPU reference is blank"
    );

    let result = compare_pixels(cpu.as_raw_bytes(), gpu.as_raw_bytes());
    let fraction = result.mismatched as f64 / result.total as f64;
    let save_all = std::env::var_os("GPU_PARITY_SAVE").is_some();
    if save_all || fraction > MISMATCH_BUDGET {
        save(name, "cpu", size, cpu.as_raw_bytes());
        save(name, "gpu", size, gpu.as_raw_bytes());
    }
    if fraction > MISMATCH_BUDGET {
        let diff: Vec<u8> = cpu
            .as_raw_bytes()
            .chunks_exact(4)
            .zip(gpu.as_raw_bytes().chunks_exact(4))
            .flat_map(|(a, b)| {
                let d = a.iter().zip(b).map(|(x, y)| x.abs_diff(*y)).max().unwrap_or(0);
                let v = if d > CHANNEL_TOLERANCE { 255 } else { d.saturating_mul(16) };
                [v, 0, 0, 255]
            })
            .collect();
        save(name, "diff", size, &diff);
        panic!(
            "{name}: {} of {} pixels ({:.3}%) differ by more than {CHANNEL_TOLERANCE} \
             (max channel delta {}); images in {}",
            result.mismatched,
            result.total,
            fraction * 100.0,
            result.max_delta,
            out_dir().display()
        );
    }
}

/// Declare a parity test whose body draws into `$t`, a `GenericDrawTarget`.
macro_rules! parity {
    ($name:ident, $w:expr, $h:expr, |$t:ident| $body:block) => {
        #[test]
        fn $name() {
            fn scene<T: GenericDrawTarget>($t: &mut T) $body
            run_parity(
                stringify!($name),
                Size2D::new($w, $h),
                scene::<VelloCPUDrawTarget>,
                scene::<VelloGpuDrawTarget>,
            );
        }
    };
}

// ---- helpers -------------------------------------------------------------------------------

fn color(r: u8, g: u8, b: u8, a: f32) -> FillOrStrokeStyle {
    FillOrStrokeStyle::Color(AbsoluteColor::srgb_legacy(r, g, b, a))
}

fn stop(offset: f64, r: u8, g: u8, b: u8, a: f32) -> CanvasGradientStop {
    CanvasGradientStop {
        offset,
        color: AbsoluteColor::srgb_legacy(r, g, b, a),
    }
}

fn normal() -> CompositionOptions {
    CompositionOptions {
        alpha: 1.0,
        composition_operation: CompositionOrBlending::Composition(CompositionStyle::SourceOver),
    }
}

fn with_op(operation: CompositionOrBlending) -> CompositionOptions {
    CompositionOptions {
        alpha: 1.0,
        composition_operation: operation,
    }
}

fn line(width: f64, cap: LineCapStyle, join: LineJoinStyle, dash: Vec<f32>) -> LineOptions {
    LineOptions {
        width,
        cap_style: cap,
        join_style: join,
        miter_limit: 10.0,
        dash,
        dash_offset: 0.0,
    }
}

fn identity() -> Transform2D<f64> {
    Transform2D::identity()
}

fn rect(x: f32, y: f32, w: f32, h: f32) -> Rect<f32> {
    Rect::new(euclid::default::Point2D::new(x, y), Size2D::new(w, h))
}

fn circle(cx: f64, cy: f64, r: f64) -> Path {
    let mut path = Path::new();
    path.move_to(cx + r, cy);
    let _ = path.arc(cx, cy, r, 0.0, std::f64::consts::TAU, false);
    path.close_path();
    path
}

fn star(cx: f64, cy: f64, r: f64) -> Path {
    let mut path = Path::new();
    for i in 0..5 {
        let angle = -std::f64::consts::FRAC_PI_2 + i as f64 * 4.0 * std::f64::consts::PI / 5.0;
        let (x, y) = (cx + r * angle.cos(), cy + r * angle.sin());
        if i == 0 {
            path.move_to(x, y);
        } else {
            path.line_to(x, y);
        }
    }
    path.close_path();
    path
}

fn backdrop<T: GenericDrawTarget>(t: &mut T) {
    t.fill_rect(&rect(8.0, 8.0, 72.0, 72.0), color(220, 40, 40, 1.0), normal(), identity());
}

fn composite_scene<T: GenericDrawTarget>(t: &mut T, operation: CompositionOrBlending) {
    backdrop(t);
    t.fill(
        &circle(70.0, 70.0, 40.0),
        FillRule::Nonzero,
        color(40, 90, 230, 0.8),
        with_op(operation),
        identity(),
    );
}

// ---- geometry ------------------------------------------------------------------------------

parity!(solid_rects, 128, 96, |t| {
    t.fill_rect(&rect(4.0, 4.0, 60.0, 40.0), color(255, 0, 0, 1.0), normal(), identity());
    t.fill_rect(&rect(30.5, 20.25, 70.0, 50.0), color(0, 160, 255, 0.6), normal(), identity());
});

parity!(nonzero_star, 120, 120, |t| {
    t.fill(&star(60.0, 60.0, 50.0), FillRule::Nonzero, color(30, 160, 60, 1.0), normal(), identity());
});

parity!(evenodd_star, 120, 120, |t| {
    t.fill(&star(60.0, 60.0, 50.0), FillRule::Evenodd, color(30, 160, 60, 1.0), normal(), identity());
});

parity!(circles_aa, 160, 120, |t| {
    for i in 0..12 {
        let r = 1.5 + i as f64 * 2.3;
        t.fill(
            &circle(14.0 + i as f64 * 11.0, 60.0, r),
            FillRule::Nonzero,
            color(250, 200, 40, 0.9),
            normal(),
            identity(),
        );
    }
});

parity!(curves_and_transforms, 160, 160, |t| {
    let mut path = Path::new();
    path.move_to(10.0, 80.0);
    path.bezier_curve_to(40.0, 0.0, 120.0, 160.0, 150.0, 80.0);
    path.quadratic_curve_to(80.0, 150.0, 10.0, 80.0);
    path.close_path();
    let transform = Transform2D::rotation(euclid::Angle::degrees(17.0))
        .then_translate(euclid::default::Vector2D::new(12.0, -18.0));
    t.fill(&path, FillRule::Nonzero, color(120, 60, 200, 1.0), normal(), transform);
});

parity!(global_alpha, 120, 80, |t| {
    backdrop(t);
    let options = CompositionOptions {
        alpha: 0.35,
        ..normal()
    };
    t.fill_rect(&rect(30.0, 20.0, 80.0, 50.0), color(0, 0, 255, 1.0), options, identity());
});

// ---- gradients -----------------------------------------------------------------------------

parity!(linear_gradient, 160, 80, |t| {
    let style = FillOrStrokeStyle::LinearGradient(LinearGradientStyle::new(
        10.0,
        0.0,
        150.0,
        0.0,
        vec![stop(0.0, 255, 0, 0, 1.0), stop(0.5, 255, 255, 0, 0.5), stop(1.0, 0, 0, 255, 1.0)],
    ));
    t.fill_rect(&rect(0.0, 0.0, 160.0, 80.0), style, normal(), identity());
});

parity!(radial_gradient, 128, 128, |t| {
    let style = FillOrStrokeStyle::RadialGradient(RadialGradientStyle::new(
        64.0,
        64.0,
        0.0,
        64.0,
        64.0,
        60.0,
        vec![stop(0.0, 255, 240, 200, 1.0), stop(0.4, 244, 150, 60, 0.4), stop(1.0, 150, 50, 20, 0.0)],
    ));
    t.fill(&circle(64.0, 64.0, 60.0), FillRule::Nonzero, style, normal(), identity());
});

parity!(two_point_radial_gradient, 128, 128, |t| {
    let style = FillOrStrokeStyle::RadialGradient(RadialGradientStyle::new(
        40.0,
        40.0,
        5.0,
        70.0,
        70.0,
        55.0,
        vec![stop(0.0, 0, 200, 120, 1.0), stop(1.0, 20, 20, 80, 1.0)],
    ));
    t.fill_rect(&rect(0.0, 0.0, 128.0, 128.0), style, normal(), identity());
});

// ---- strokes -------------------------------------------------------------------------------

fn zigzag() -> Path {
    let mut path = Path::new();
    path.move_to(20.0, 90.0);
    path.line_to(60.0, 20.0);
    path.line_to(100.0, 90.0);
    path.line_to(140.0, 30.0);
    path
}

parity!(stroke_butt_miter, 160, 110, |t| {
    t.stroke(&zigzag(), color(0, 0, 0, 1.0), line(9.0, LineCapStyle::Butt, LineJoinStyle::Miter, vec![]), normal(), identity());
});

parity!(stroke_round_round, 160, 110, |t| {
    t.stroke(&zigzag(), color(200, 0, 100, 1.0), line(9.0, LineCapStyle::Round, LineJoinStyle::Round, vec![]), normal(), identity());
});

parity!(stroke_square_bevel, 160, 110, |t| {
    t.stroke(&zigzag(), color(0, 120, 200, 1.0), line(9.0, LineCapStyle::Square, LineJoinStyle::Bevel, vec![]), normal(), identity());
});

parity!(stroke_hairline_and_dashes, 160, 110, |t| {
    t.stroke(&zigzag(), color(0, 0, 0, 1.0), line(1.0, LineCapStyle::Butt, LineJoinStyle::Miter, vec![]), normal(), identity());
    t.stroke(&circle(80.0, 55.0, 40.0), color(220, 80, 0, 1.0), line(3.0, LineCapStyle::Round, LineJoinStyle::Round, vec![8.0, 5.0]), normal(), identity());
});

parity!(stroke_rect, 120, 80, |t| {
    t.stroke_rect(&rect(10.5, 10.5, 99.0, 59.0), color(10, 10, 10, 1.0), line(2.0, LineCapStyle::Butt, LineJoinStyle::Miter, vec![]), normal(), identity());
});

// ---- composite and blend modes -------------------------------------------------------------

macro_rules! composite {
    ($name:ident, $op:expr) => {
        parity!($name, 120, 120, |t| {
            composite_scene(t, $op);
        });
    };
}

composite!(composite_clear, CompositionOrBlending::Composition(CompositionStyle::Clear));
composite!(composite_copy, CompositionOrBlending::Composition(CompositionStyle::Copy));
composite!(composite_source_over, CompositionOrBlending::Composition(CompositionStyle::SourceOver));
composite!(composite_destination_over, CompositionOrBlending::Composition(CompositionStyle::DestinationOver));
composite!(composite_source_in, CompositionOrBlending::Composition(CompositionStyle::SourceIn));
composite!(composite_destination_in, CompositionOrBlending::Composition(CompositionStyle::DestinationIn));
composite!(composite_source_out, CompositionOrBlending::Composition(CompositionStyle::SourceOut));
composite!(composite_destination_out, CompositionOrBlending::Composition(CompositionStyle::DestinationOut));
composite!(composite_source_atop, CompositionOrBlending::Composition(CompositionStyle::SourceAtop));
composite!(composite_destination_atop, CompositionOrBlending::Composition(CompositionStyle::DestinationAtop));
composite!(composite_xor, CompositionOrBlending::Composition(CompositionStyle::Xor));
composite!(composite_lighter, CompositionOrBlending::Composition(CompositionStyle::Lighter));
composite!(blend_multiply, CompositionOrBlending::Blending(BlendingStyle::Multiply));
composite!(blend_screen, CompositionOrBlending::Blending(BlendingStyle::Screen));
composite!(blend_overlay, CompositionOrBlending::Blending(BlendingStyle::Overlay));
composite!(blend_darken, CompositionOrBlending::Blending(BlendingStyle::Darken));
composite!(blend_lighten, CompositionOrBlending::Blending(BlendingStyle::Lighten));
composite!(blend_color_dodge, CompositionOrBlending::Blending(BlendingStyle::ColorDodge));
composite!(blend_color_burn, CompositionOrBlending::Blending(BlendingStyle::ColorBurn));
composite!(blend_hard_light, CompositionOrBlending::Blending(BlendingStyle::HardLight));
composite!(blend_soft_light, CompositionOrBlending::Blending(BlendingStyle::SoftLight));
composite!(blend_difference, CompositionOrBlending::Blending(BlendingStyle::Difference));
composite!(blend_exclusion, CompositionOrBlending::Blending(BlendingStyle::Exclusion));
composite!(blend_hue, CompositionOrBlending::Blending(BlendingStyle::Hue));
composite!(blend_saturation, CompositionOrBlending::Blending(BlendingStyle::Saturation));
composite!(blend_color, CompositionOrBlending::Blending(BlendingStyle::Color));
composite!(blend_luminosity, CompositionOrBlending::Blending(BlendingStyle::Luminosity));

// ---- clips, clears, retained content -------------------------------------------------------

parity!(clip_path, 120, 120, |t| {
    t.push_clip(&circle(60.0, 60.0, 45.0), FillRule::Nonzero, identity());
    t.fill_rect(&rect(0.0, 0.0, 120.0, 120.0), color(0, 150, 90, 1.0), normal(), identity());
    t.pop_clip();
    t.fill_rect(&rect(0.0, 0.0, 30.0, 30.0), color(250, 0, 0, 1.0), normal(), identity());
});

parity!(nested_clips, 120, 120, |t| {
    t.push_clip(&circle(50.0, 60.0, 40.0), FillRule::Nonzero, identity());
    t.push_clip_rect(&Rect::new(euclid::default::Point2D::new(40, 0), Size2D::new(80, 120)));
    t.fill_rect(&rect(0.0, 0.0, 120.0, 120.0), color(80, 40, 200, 1.0), normal(), identity());
    t.pop_clip();
    t.fill_rect(&rect(0.0, 50.0, 120.0, 20.0), color(250, 200, 0, 1.0), normal(), identity());
    t.pop_clip();
});

parity!(partial_clear_rect, 120, 80, |t| {
    t.fill_rect(&rect(0.0, 0.0, 120.0, 80.0), color(20, 20, 160, 1.0), normal(), identity());
    t.clear_rect(&rect(30.0, 20.0, 50.0, 30.0), identity());
});

parity!(full_clear_then_draw, 120, 80, |t| {
    t.fill_rect(&rect(0.0, 0.0, 120.0, 80.0), color(20, 20, 160, 1.0), normal(), identity());
    t.clear_rect(&rect(0.0, 0.0, 120.0, 80.0), identity());
    t.fill_rect(&rect(10.0, 10.0, 40.0, 40.0), color(200, 20, 20, 1.0), normal(), identity());
});

parity!(retained_across_updates, 120, 80, |t| {
    t.fill_rect(&rect(0.0, 0.0, 80.0, 50.0), color(20, 140, 20, 1.0), normal(), identity());
    // Simulate a presented update: the next draws must land on top of retained content.
    let _ = t.snapshot();
    t.fill_rect(&rect(40.0, 25.0, 80.0, 55.0), color(200, 20, 120, 0.7), normal(), identity());
    let _ = t.snapshot();
    t.clear_rect(&rect(0.0, 0.0, 20.0, 20.0), identity());
});

// ---- text ----------------------------------------------------------------------------------

/// Lato Medium from the WPT fonts, shared by every text scene (one identifier, so both backends
/// exercise their font caches).
fn lato() -> &'static (uuid::Uuid, Vec<u8>) {
    static LATO: std::sync::OnceLock<(uuid::Uuid, Vec<u8>)> = std::sync::OnceLock::new();
    LATO.get_or_init(|| {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/wpt/tests/fonts/Lato-Medium.ttf");
        (uuid::Uuid::new_v4(), std::fs::read(path).expect("Lato-Medium.ttf"))
    })
}

/// Lay `text` out on one line with the font's own advances (no shaping; enough for parity).
fn text_run(text: &str, x: f32, y: f32, pt_size: f32) -> TextRun {
    use skrifa::MetadataProvider;
    let (id, bytes) = lato();
    let font = skrifa::FontRef::new(bytes).expect("parse Lato");
    let charmap = font.charmap();
    let metrics = font.glyph_metrics(skrifa::instance::Size::new(pt_size), skrifa::instance::LocationRef::default());
    let mut pen = x;
    let glyphs_and_positions = text
        .chars()
        .map(|ch| {
            let glyph = charmap.map(ch).unwrap_or_default();
            let positioned = GlyphAndPosition {
                id: glyph.to_u32(),
                point: Point2D::new(pen, y),
            };
            pen += metrics.advance_width(glyph).unwrap_or(pt_size * 0.5);
            positioned
        })
        .collect();
    TextRun {
        font: CanvasFont {
            identifier: FontIdentifier::ArrayBuffer(*id),
            data: Some(FontDataAndIndex {
                data: FontData::from_bytes(bytes),
                index: 0,
            }),
        },
        pt_size,
        glyphs_and_positions,
        advance: pen - x,
        bounds: euclid::default::Rect::new(
            euclid::default::Point2D::new(x as f64, (y - pt_size) as f64),
            Size2D::new((pen - x) as f64, pt_size as f64 * 1.3),
        ),
    }
}

parity!(fill_text_sizes, 220, 120, |t| {
    for (y, size) in [(16.0, 12.0), (40.0, 20.0), (78.0, 34.0), (114.0, 30.0)] {
        t.fill_text(vec![text_run("Loom GPU canvas 0123", 6.0, y, size)], color(30, 30, 40, 1.0), normal(), identity());
    }
});

parity!(stroke_text, 220, 80, |t| {
    let options = line(2.0, LineCapStyle::Round, LineJoinStyle::Round, vec![]);
    t.stroke_text(vec![text_run("Outline", 8.0, 56.0, 48.0)], color(200, 40, 40, 1.0), options, normal(), identity());
});

parity!(gradient_text, 220, 80, |t| {
    let gradient = FillOrStrokeStyle::LinearGradient(LinearGradientStyle::new(
        0.0,
        0.0,
        220.0,
        0.0,
        vec![stop(0.0, 255, 80, 0, 1.0), stop(1.0, 40, 80, 255, 1.0)],
    ));
    t.fill_text(vec![text_run("Gradient", 6.0, 58.0, 52.0)], gradient, normal(), identity());
});

parity!(transformed_text, 200, 200, |t| {
    let transform = Transform2D::rotation(euclid::Angle::degrees(30.0))
        .then_scale(1.2, 0.9)
        .then_translate(euclid::default::Vector2D::new(40.0, 30.0));
    t.fill_text(vec![text_run("Rotated text", 0.0, 24.0, 22.0)], color(20, 120, 60, 0.85), normal(), transform);
});

parity!(text_composite_and_clip, 200, 100, |t| {
    t.fill_rect(&rect(0.0, 0.0, 200.0, 100.0), color(30, 40, 90, 1.0), normal(), identity());
    t.push_clip(&circle(100.0, 50.0, 46.0), FillRule::Nonzero, identity());
    let lighter = with_op(CompositionOrBlending::Composition(CompositionStyle::Lighter));
    t.fill_text(vec![text_run("Lighter", 10.0, 62.0, 44.0)], color(200, 120, 40, 0.8), lighter, identity());
    t.pop_clip();
    let xor = with_op(CompositionOrBlending::Composition(CompositionStyle::Xor));
    t.fill_text(vec![text_run("xor", 140.0, 90.0, 24.0)], color(250, 250, 250, 1.0), xor, identity());
});

parity!(text_runs_hud, 256, 96, |t| {
    // A HUD-style label: several runs, mixed sizes and alpha, drawn every frame.
    t.fill_rect(&rect(0.0, 0.0, 256.0, 96.0), color(12, 14, 20, 0.8), normal(), identity());
    let runs = vec![
        text_run("P1 1250", 8.0, 24.0, 18.0),
        text_run("P2 980", 136.0, 24.0, 18.0),
        text_run("HP 74 / 100", 8.0, 56.0, 14.0),
        text_run("x3 combo!", 136.0, 56.0, 14.0),
        text_run("wave 12 - 00:41.6", 8.0, 86.0, 11.0),
    ];
    t.fill_text(runs, color(240, 230, 200, 0.95), normal(), identity());
});

// ---- realistic workloads -------------------------------------------------------------------

// The hands-on main menu's ember particles: a radial-gradient glow plus a solid core per
// particle, composited with `lighter`.
parity!(ember_particles, 256, 192, |t| {
    let lighter = with_op(CompositionOrBlending::Composition(CompositionStyle::Lighter));
    let mut seed = 0x2545_f491_u32;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed % 10_000) as f64 / 10_000.0
    };
    for _ in 0..90 {
        let (x, y) = (next() * 256.0, next() * 192.0);
        let size = 1.0 + next() * 3.0;
        let alpha = 0.3 + next() as f32 * 0.65;
        let glow = FillOrStrokeStyle::RadialGradient(RadialGradientStyle::new(
            x,
            y,
            0.0,
            x,
            y,
            size * 3.2,
            vec![stop(0.0, 255, 220, 150, alpha), stop(0.4, 244, 150, 60, alpha * 0.4), stop(1.0, 150, 50, 20, 0.0)],
        ));
        t.fill(&circle(x, y, size * 3.2), FillRule::Nonzero, glow, lighter, identity());
        t.fill(&circle(x, y, size * 0.7), FillRule::Nonzero, color(255, 240, 220, alpha), lighter, identity());
    }
});
