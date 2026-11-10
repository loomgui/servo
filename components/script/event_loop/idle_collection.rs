/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Deciding when to collect JS garbage between two frames of a seal-driven embedder.
//!
//! Collections are not incremental in Servo, so each one pauses the script thread for a
//! fixed cost (marking the live heap) plus a cost that grows with the garbage it frees.
//! The engine's own trigger waits for tens of megabytes and then pauses for over 100 ms.
//!
//! After a seal-tagged rendering update the frame's display list has been sent and the
//! script thread has nothing to do until the embedder seals its next frame. This type
//! measures that gap and how long collections take, and asks for a collection:
//!
//! - when the garbage has reached the configured bound, whatever the gap is; and
//! - earlier, when the gap is too short for a collection of the bound's size but long
//!   enough for a smaller one, at the last update where the collection still fits.

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::time::Duration;

/// How many recent gaps and collections the estimates are made from.
const SAMPLES: usize = 8;

/// The share of the shortest recent gap that a collection is allowed to fill.
const GAP_SHARE: f64 = 0.8;

/// A collection that leaves at least this much of the gap unused did not delay the next
/// frame, so the gap was the idle time plus the collection.
const UNUSED_GAP: Duration = Duration::from_micros(200);

/// Garbage below this is not worth a collection's fixed cost.
const MINIMUM_GARBAGE_KB: f64 = 16.0;

/// Without collections of different sizes to compare, the share of a collection's pause
/// that is assumed not to depend on the amount of garbage.
const ASSUMED_FIXED_SHARE: f64 = 0.65;

#[derive(Default)]
pub(crate) struct IdleCollection {
    /// The JS heap size right after the last collection.
    heap_bytes_after_collection: Cell<u32>,
    /// The JS heap size at the previous seal-tagged update.
    heap_bytes_at_previous_update: Cell<u32>,
    /// A running average of how much the JS heap grows from one update to the next.
    growth_kb_per_update: Cell<f64>,
    /// How long the script thread has waited for messages since the last seal-tagged
    /// update completed. `None` while the next frame is being produced.
    idle_in_gap: Cell<Option<Duration>>,
    /// How long a collection started in the current gap took.
    collection_in_gap: Cell<Duration>,
    /// The most recent gaps between an update and the next seal.
    gaps: RefCell<VecDeque<Duration>>,
    /// The most recent collections, as (garbage in KB, pause in ms).
    collections: RefCell<VecDeque<(f64, f64)>>,
}

/// Why a collection was requested, for logging.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Reason {
    /// The garbage reached the configured bound.
    Bound,
    /// A collection of this size is the largest that still fits the idle gap.
    FitsGap,
}

impl IdleCollection {
    /// The script thread waited this long for a message.
    pub(crate) fn waited(&self, duration: Duration) {
        if let Some(idle) = self.idle_in_gap.get() {
            self.idle_in_gap.set(Some(idle + duration));
        }
    }

    /// The embedder sealed its next frame: the gap that started with the last update is over.
    pub(crate) fn next_frame_sealed(&self) {
        let Some(idle) = self.idle_in_gap.take() else {
            return;
        };
        let collection = self.collection_in_gap.replace(Duration::ZERO);
        // A collection that used up the whole gap hides how long the gap was. The idle time
        // alone is then a lower bound, which only makes the next decisions more careful.
        let gap = if idle >= UNUSED_GAP {
            idle + collection
        } else {
            idle
        };
        push_sample(&mut self.gaps.borrow_mut(), gap);
    }

    /// A seal-tagged rendering update has completed with the JS heap at `heap_bytes`.
    /// Returns why a collection should run now, if one should.
    pub(crate) fn update_completed(
        &self,
        heap_bytes: u32,
        bound_kb: f64,
        fit_gap: bool,
    ) -> Option<Reason> {
        self.idle_in_gap.set(Some(Duration::ZERO));
        self.collection_in_gap.set(Duration::ZERO);

        let after_collection = self.heap_bytes_after_collection.get();
        let previous = self.heap_bytes_at_previous_update.replace(heap_bytes);
        if heap_bytes < after_collection {
            // The engine collected on its own in the meantime.
            self.heap_bytes_after_collection.set(heap_bytes);
            return None;
        }
        let grown_kb = f64::from(heap_bytes.saturating_sub(previous)) / 1024.0;
        self.growth_kb_per_update
            .set(self.growth_kb_per_update.get() * 0.8 + grown_kb * 0.2);

        let garbage_kb = f64::from(heap_bytes - after_collection) / 1024.0;
        if garbage_kb >= bound_kb {
            return Some(Reason::Bound);
        }
        if !fit_gap || garbage_kb < MINIMUM_GARBAGE_KB {
            return None;
        }
        let budget_ms = self.budget_ms()?;
        let (fixed_ms, ms_per_kb) = self.pause_model()?;
        let pause_now = fixed_ms + ms_per_kb * garbage_kb;
        let pause_at_bound = fixed_ms + ms_per_kb * bound_kb;
        if pause_at_bound <= budget_ms || pause_now > budget_ms {
            // Either the bound's collection fits anyway, or nothing fits any more and the
            // bound limits how much builds up.
            return None;
        }
        let garbage_at_next_update = garbage_kb + 1.5 * self.growth_kb_per_update.get();
        let pause_at_next_update = fixed_ms + ms_per_kb * garbage_at_next_update;
        (pause_at_next_update > budget_ms).then_some(Reason::FitsGap)
    }

    /// A collection requested by [`Self::update_completed`] has run.
    pub(crate) fn collected(&self, heap_bytes_before: u32, heap_bytes_after: u32, pause: Duration) {
        let garbage_kb =
            f64::from(heap_bytes_before.saturating_sub(self.heap_bytes_after_collection.get()))
                / 1024.0;
        self.heap_bytes_after_collection.set(heap_bytes_after);
        self.heap_bytes_at_previous_update.set(heap_bytes_after);
        self.collection_in_gap.set(pause);
        push_sample(
            &mut self.collections.borrow_mut(),
            (garbage_kb, pause.as_secs_f64() * 1000.0),
        );
    }

    /// The pause that a collection may take, in milliseconds: a share of the shortest
    /// recent gap. `None` until enough gaps have been measured.
    pub(crate) fn budget_ms(&self) -> Option<f64> {
        let gaps = self.gaps.borrow();
        if gaps.len() < SAMPLES {
            return None;
        }
        let shortest = gaps.iter().min()?;
        Some(shortest.as_secs_f64() * 1000.0 * GAP_SHARE)
    }

    /// The pause of a collection as (fixed ms, ms per KB of garbage), fitted to the recent
    /// collections. `None` before the first collection.
    pub(crate) fn pause_model(&self) -> Option<(f64, f64)> {
        let collections = self.collections.borrow();
        if collections.is_empty() {
            return None;
        }
        let count = collections.len() as f64;
        let mean_kb = collections.iter().map(|(kb, _)| kb).sum::<f64>() / count;
        let mean_ms = collections.iter().map(|(_, ms)| ms).sum::<f64>() / count;
        let smallest_kb = collections
            .iter()
            .map(|(kb, _)| *kb)
            .fold(f64::MAX, f64::min);
        let largest_kb = collections.iter().map(|(kb, _)| *kb).fold(0.0, f64::max);
        if smallest_kb > 0.0 && largest_kb >= 1.5 * smallest_kb {
            let covariance: f64 = collections
                .iter()
                .map(|(kb, ms)| (kb - mean_kb) * (ms - mean_ms))
                .sum();
            let variance: f64 = collections
                .iter()
                .map(|(kb, _)| (kb - mean_kb).powi(2))
                .sum();
            let ms_per_kb = (covariance / variance).max(0.0);
            let fixed_ms = (mean_ms - ms_per_kb * mean_kb).max(0.0);
            return Some((fixed_ms, ms_per_kb));
        }
        if mean_kb <= 0.0 {
            return Some((mean_ms, 0.0));
        }
        Some((
            mean_ms * ASSUMED_FIXED_SHARE,
            mean_ms * (1.0 - ASSUMED_FIXED_SHARE) / mean_kb,
        ))
    }
}

fn push_sample<T>(samples: &mut VecDeque<T>, sample: T) {
    if samples.len() == SAMPLES {
        samples.pop_front();
    }
    samples.push_back(sample);
}

#[cfg(test)]
mod tests {
    use super::*;

    const KB: u32 = 1024;

    /// Runs `updates` frames that each add `growth_kb` of garbage and leave `gap_ms` idle,
    /// with collections that take `fixed_ms + ms_per_kb * garbage`. Returns every
    /// collection as (reason, garbage in KB).
    fn run(
        collector: &IdleCollection,
        updates: usize,
        growth_kb: u32,
        gap_ms: f64,
        bound_kb: f64,
        fit_gap: bool,
    ) -> Vec<(Reason, f64)> {
        let (fixed_ms, ms_per_kb) = (2.0, 0.005);
        let mut heap = collector.heap_bytes_at_previous_update.get();
        let mut collected = Vec::new();
        for _ in 0..updates {
            heap += growth_kb * KB;
            let mut used_ms = 0.0;
            if let Some(reason) = collector.update_completed(heap, bound_kb, fit_gap) {
                let garbage_kb =
                    f64::from(heap - collector.heap_bytes_after_collection.get()) / 1024.0;
                used_ms = fixed_ms + ms_per_kb * garbage_kb;
                let after = collector.heap_bytes_after_collection.get();
                collector.collected(heap, after, Duration::from_secs_f64(used_ms / 1000.0));
                heap = after;
                collected.push((reason, garbage_kb));
            }
            collector.waited(Duration::from_secs_f64(
                (gap_ms - used_ms).max(0.0) / 1000.0,
            ));
            collector.next_frame_sealed();
        }
        collected
    }

    #[test]
    fn collects_at_the_bound_when_the_gap_is_long() {
        let collector = IdleCollection::default();
        let collected = run(&collector, 400, 4, 12.0, 256.0, true);
        assert!(collected.len() >= 5);
        assert!(
            collected
                .iter()
                .all(|(reason, kb)| *reason == Reason::Bound && *kb >= 256.0)
        );
    }

    #[test]
    fn collects_at_the_bound_when_nothing_fits_the_gap() {
        let collector = IdleCollection::default();
        let collected = run(&collector, 400, 4, 0.05, 256.0, true);
        assert!(collected.len() >= 5);
        assert!(collected.iter().all(|(reason, _)| *reason == Reason::Bound));
    }

    #[test]
    fn collects_less_when_only_a_smaller_collection_fits_the_gap() {
        let collector = IdleCollection::default();
        // A 256 KB collection takes 3.28 ms. A 3.4 ms gap allows 2.72 ms: about 144 KB.
        let collected = run(&collector, 2000, 4, 3.4, 256.0, true);
        let settled = &collected[collected.len() - 10..];
        for (reason, kb) in settled {
            assert_eq!(*reason, Reason::FitsGap);
            assert!((100.0..=144.0).contains(kb), "collected {kb} KB");
        }
    }

    #[test]
    fn ignores_the_gap_when_fitting_is_off() {
        let collector = IdleCollection::default();
        let collected = run(&collector, 400, 4, 3.4, 256.0, false);
        assert!(collected.iter().all(|(reason, _)| *reason == Reason::Bound));
    }

    #[test]
    fn follows_a_collection_by_the_engine() {
        let collector = IdleCollection::default();
        assert_eq!(
            collector.update_completed(300 * KB, 256.0, true),
            Some(Reason::Bound)
        );
        collector.collected(300 * KB, 100 * KB, Duration::from_millis(3));
        // The engine collected: the heap is now smaller than after our last collection.
        assert_eq!(collector.update_completed(50 * KB, 256.0, true), None);
        assert_eq!(collector.heap_bytes_after_collection.get(), 50 * KB);
        assert_eq!(collector.update_completed(60 * KB, 256.0, true), None);
    }
}
