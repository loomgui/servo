/* Diagnostic overlay only. Never a rendering/validity decision. */
use std::cell::{Cell, RefCell};
use std::sync::{Once, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

fn gate(oracle: &str, probe: &str, profile: &str) -> bool {
    oracle == "1"
        && probe == "1"
        && matches!(
            profile,
            "app"
                | "oracle-retained-backgrounds"
                | "oracle-selected-highlight"
                | "oracle-certified-backgrounds"
        )
}

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        gate(
            &std::env::var("LOOM_PACING_ORACLE").unwrap_or_default(),
            &std::env::var("LOOM_PREPARATION_PROBE").unwrap_or_default(),
            &std::env::var("LOOM_PACING_PROFILE").unwrap_or_default(),
        )
    })
}

#[derive(Clone, Copy, Default)]
struct ShaderTotals {
    count: u64,
    nanoseconds: u128,
}
#[derive(Default)]
struct ShaderState {
    starts: Vec<Instant>,
    skipped_depth: usize,
    totals: ShaderTotals,
}
thread_local! {
    static SHADERS: RefCell<ShaderState> = RefCell::new(ShaderState::default());
    static LOST: Cell<u64> = const { Cell::new(0) };
    static COMPLETED: RefCell<Vec<Record>> = const { RefCell::new(Vec::new()) };
    static AGGREGATE: RefCell<Aggregate> = RefCell::new(Aggregate::default());
}
fn lost() {
    LOST.with(|n| n.set(n.get().saturating_add(1)));
}
fn shader_totals() -> ShaderTotals {
    SHADERS.with(|state| match state.try_borrow() {
        Ok(state) => state.totals,
        Err(_) => {
            lost();
            ShaderTotals::default()
        },
    })
}
struct Hooks;
static HOOKS: Hooks = Hooks;
impl webrender::ProfilerHooks for Hooks {
    fn register_thread(&self, _: &str) {}
    fn unregister_thread(&self) {}
    fn event_marker(&self, _: &str) {}
    fn add_text_marker(&self, _: &str, _: &str, _: Duration) {}
    fn thread_is_being_profiled(&self) -> bool {
        false
    }
    fn begin_marker(&self, label: &str) {
        if label != "loom shader cache compile link" {
            return;
        }
        // No TLS borrow survives into GL/WR work. The stack handles nesting.
        SHADERS.with(|state| match state.try_borrow_mut() {
            Ok(mut state) => {
                if state.skipped_depth != 0 || state.starts.len() >= 16 {
                    state.skipped_depth = state.skipped_depth.saturating_add(1);
                    lost();
                } else {
                    state.starts.push(Instant::now());
                }
            },
            Err(_) => lost(),
        });
    }
    fn end_marker(&self, label: &str) {
        if label != "loom shader cache compile link" {
            return;
        }
        SHADERS.with(|state| match state.try_borrow_mut() {
            Ok(mut state) => {
                if state.skipped_depth != 0 {
                    state.skipped_depth -= 1;
                } else if let Some(start) = state.starts.pop() {
                    let elapsed = start.elapsed();
                    state.totals.count = state.totals.count.saturating_add(1);
                    state.totals.nanoseconds =
                        state.totals.nanoseconds.saturating_add(elapsed.as_nanos());
                } else {
                    lost();
                }
            },
            Err(_) => lost(),
        });
    }
}

/// Called at the very start of Paint::new, before this embedder creates any
/// WebView/Painter/WebRender instance. WR silently ignores a late installation.
pub(crate) fn install_hooks_before_painters() {
    if !enabled() {
        return;
    }
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| webrender::set_profiler_hooks(Some(&HOOKS)));
}

struct Record {
    frame_tag: u64,
    utc_us: u128,
    success: bool,
    total_ns: u128,
    stages_ns: [u128; 4],
    shaders_before: ShaderTotals,
    shaders_after: ShaderTotals,
    lost_markers: u64,
}
struct Running {
    frame_tag: u64,
    utc_us: u128,
    start: Instant,
    last: Instant,
    stages_ns: [u128; 4],
    next: usize,
    shaders_before: ShaderTotals,
}
pub(crate) struct PreparationSpan(Option<Running>);
impl PreparationSpan {
    pub(crate) fn begin(frame_tag: u64) -> Self {
        if !enabled() {
            return Self(None);
        }
        let start = Instant::now();
        Self(Some(Running {
            frame_tag,
            utc_us: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_micros())
                .unwrap_or(0),
            start,
            last: start,
            stages_ns: [0; 4],
            next: 0,
            shaders_before: shader_totals(),
        }))
    }
    // The four sequential checkpoints are notify, make-current+GL-check,
    // framebuffer lookup/bind, renderer.update. They include waits/descheduling.
    pub(crate) fn checkpoint(&mut self) {
        let Some(run) = self.0.as_mut() else {
            return;
        };
        let now = Instant::now();
        if run.next < run.stages_ns.len() {
            run.stages_ns[run.next] = now.duration_since(run.last).as_nanos();
            run.next += 1;
        } else {
            lost();
        }
        run.last = now;
    }
    pub(crate) fn finish(&mut self, success: bool) {
        let Some(run) = self.0.take() else {
            return;
        };
        let record = Record {
            frame_tag: run.frame_tag,
            utc_us: run.utc_us,
            success,
            total_ns: run.start.elapsed().as_nanos(),
            stages_ns: run.stages_ns,
            shaders_before: run.shaders_before,
            shaders_after: shader_totals(),
            lost_markers: LOST.with(Cell::get),
        };
        // Only retain owned data. Emission is outside the painter RefMut.
        COMPLETED.with(|records| match records.try_borrow_mut() {
            Ok(mut records) if records.len() < 16 => records.push(record),
            _ => lost(),
        });
    }
}
impl Drop for PreparationSpan {
    fn drop(&mut self) {
        self.finish(false);
    }
}
#[derive(Default)]
struct Aggregate {
    count: u64,
    total_ns: u128,
    stages_ns: [u128; 4],
    max_ns: u128,
    outliers: u64,
}
fn ms(ns: u128) -> f64 {
    ns as f64 / 1_000_000.0
}
/// Paint calls this only AFTER its painter RefMut has dropped. No GL/TLS borrow
/// survives into println. I/O still perturbs this excluded diagnostic process.
pub(crate) fn flush_completed() {
    if !enabled() {
        return;
    }
    let records = COMPLETED.with(|records| match records.try_borrow_mut() {
        Ok(mut records) => std::mem::take(&mut *records),
        Err(_) => {
            lost();
            Vec::new()
        },
    });
    for record in records {
        let (emit, summary) = AGGREGATE.with(|aggregate| {
            let Ok(mut a) = aggregate.try_borrow_mut() else {
                lost();
                return (false, None);
            };
            a.count = a.count.saturating_add(1);
            a.total_ns = a.total_ns.saturating_add(record.total_ns);
            a.max_ns = a.max_ns.max(record.total_ns);
            for (total, value) in a.stages_ns.iter_mut().zip(record.stages_ns) {
                *total = total.saturating_add(value);
            }
            let outlier = record.total_ns >= 10_000_000 && a.outliers < 64;
            if outlier {
                a.outliers += 1;
            }
            let emit = a.count <= 8 || outlier;
            let summary =
                (a.count % 240 == 0).then(|| (a.count, a.total_ns, a.stages_ns, a.max_ns));
            (emit, summary)
        });
        if emit {
            println!(
                "[PREP-PROBE] painter frame_tag={} utc_us={} success={} total_ms={:.6} notify_ms={:.6} make_current_glcheck_ms={:.6} framebuffer_bind_ms={:.6} renderer_update_ms={:.6} shader_cache_compile_link_count={} shader_cache_compile_link_ms={:.6} shader_before_count={} shader_before_ms={:.6} lost_markers={} thread={:?}",
                record.frame_tag,
                record.utc_us,
                record.success,
                ms(record.total_ns),
                ms(record.stages_ns[0]),
                ms(record.stages_ns[1]),
                ms(record.stages_ns[2]),
                ms(record.stages_ns[3]),
                record
                    .shaders_after
                    .count
                    .saturating_sub(record.shaders_before.count),
                ms(record
                    .shaders_after
                    .nanoseconds
                    .saturating_sub(record.shaders_before.nanoseconds)),
                record.shaders_before.count,
                ms(record.shaders_before.nanoseconds),
                record.lost_markers,
                std::thread::current().id()
            );
        }
        if let Some((count, total, stages, max)) = summary {
            println!(
                "[PREP-PROBE] painter_aggregate count={} total_ms={:.6} notify_ms={:.6} make_current_glcheck_ms={:.6} framebuffer_bind_ms={:.6} renderer_update_ms={:.6} max_ms={:.6}",
                count,
                ms(total),
                ms(stages[0]),
                ms(stages[1]),
                ms(stages[2]),
                ms(stages[3]),
                ms(max)
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use webrender::ProfilerHooks as _;
    #[test]
    fn preparation_probe_requires_exact_opt_in_and_known_profile() {
        assert!(gate("1", "1", "oracle-retained-backgrounds"));
        assert!(!gate("1", "0", "oracle-retained-backgrounds"));
        assert!(!gate("0", "1", "oracle-retained-backgrounds"));
        assert!(!gate("1", "1", "unknown"));
    }
    #[test]
    fn shader_hook_ignores_unrelated_labels_and_retains_no_borrow() {
        let before = shader_totals();
        HOOKS.begin_marker("unrelated");
        HOOKS.end_marker("unrelated");
        assert_eq!(shader_totals().count, before.count);
        HOOKS.begin_marker("loom shader cache compile link");
        assert!(SHADERS.with(|state| state.try_borrow_mut().is_ok()));
        HOOKS.end_marker("loom shader cache compile link");
        assert_eq!(shader_totals().count, before.count + 1);
    }
}
