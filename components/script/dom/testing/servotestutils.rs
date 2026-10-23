/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

// check-tidy: no specs after this line

use backtrace::Backtrace;
use dom_struct::dom_struct;
use js::context::{JSContext, NoGC};
use layout_api::{ReflowPhasesRun, ReflowStatistics};
use script_bindings::codegen::GenericBindings::WindowBinding::WindowMethods;
use script_bindings::domstring::DOMString;
use script_bindings::reflector::Reflector;
use script_bindings::root::DomRoot;
use servo_base::Epoch;
use time::Duration;

use crate::dom::abortsignal::AbortSignal;
use crate::dom::bindings::codegen::Bindings::ServoTestUtilsBinding::ServoTestUtilsMethods;
use crate::dom::globalscope::GlobalScope;
use crate::dom::layoutresult::LayoutResult;
use crate::dom::types::AccessibilityUpdateResult;

/// One actual ScriptThread automatic update, copied only for test bindings.
/// forceLayout and queries never record or advance this observation.
#[derive(Clone, Copy, Default)]
pub(crate) struct AutomaticRenderingObservation {
    sequence: u64,
    phases: ReflowPhasesRun,
    statistics: AutomaticRenderingStatistics,
}

#[derive(Clone, Copy, Default)]
struct AutomaticRenderingStatistics {
    inline_background_color_recascade_count: u32,
    certified_background_color_count: u32,
    certified_background_cached_leaf_count: u32,
    certified_background_cached_family_count: u32,
    rebuilt_fragment_count: u32,
    restyle_fragment_count: u32,
    only_descendants_changed_count: u32,
}

impl AutomaticRenderingObservation {
    pub(crate) fn next(
        previous: Option<Self>,
        phases: ReflowPhasesRun,
        statistics: &ReflowStatistics,
    ) -> Option<Self> {
        Some(Self {
            sequence: previous
                .map_or(0, |previous| previous.sequence)
                .checked_add(1)?,
            phases,
            statistics: AutomaticRenderingStatistics::from(statistics),
        })
    }

    fn result(&self, cx: &mut JSContext, global: &GlobalScope) -> DomRoot<LayoutResult> {
        let mut phases = Vec::new();
        if self.phases.contains(ReflowPhasesRun::RanLayout) {
            phases.push(DOMString::from_static("RanLayout"))
        }
        if self
            .phases
            .contains(ReflowPhasesRun::BuiltStackingContextTree)
        {
            phases.push(DOMString::from_static("BuiltStackingContextTree"))
        }
        if self.phases.contains(ReflowPhasesRun::BuiltDisplayList) {
            phases.push(DOMString::from_static("BuiltDisplayList"))
        }
        if self
            .phases
            .contains(ReflowPhasesRun::UpdatedBackgroundColors)
        {
            phases.push(DOMString::from_static("UpdatedBackgroundColors"))
        }
        if self
            .phases
            .contains(ReflowPhasesRun::UpdatedScrollNodeOffset)
        {
            phases.push(DOMString::from_static("UpdatedScrollNodeOffset"))
        }
        if self.phases.contains(ReflowPhasesRun::UpdatedImageData) {
            phases.push(DOMString::from_static("UpdatedImageData"))
        }

        LayoutResult::new(
            cx,
            global,
            phases,
            self.sequence,
            self.statistics.inline_background_color_recascade_count,
            self.statistics.certified_background_color_count,
            self.statistics.certified_background_cached_leaf_count,
            self.statistics.certified_background_cached_family_count,
            self.statistics.rebuilt_fragment_count,
            self.statistics.restyle_fragment_count,
            self.statistics.only_descendants_changed_count,
        )
    }
}

impl From<&ReflowStatistics> for AutomaticRenderingStatistics {
    fn from(statistics: &ReflowStatistics) -> Self {
        Self {
            inline_background_color_recascade_count: statistics
                .inline_background_color_recascade_count,
            certified_background_color_count: statistics.certified_background_color_count,
            certified_background_cached_leaf_count: statistics
                .certified_background_cached_leaf_count,
            certified_background_cached_family_count: statistics
                .certified_background_cached_family_count,
            rebuilt_fragment_count: statistics.rebuilt_fragment_count,
            restyle_fragment_count: statistics.restyle_fragment_count,
            only_descendants_changed_count: statistics.only_descendants_changed_count,
        }
    }
}

#[dom_struct]
pub(crate) struct ServoTestUtils {
    reflector_: Reflector,
}

impl ServoTestUtilsMethods<crate::DomTypeHolder> for ServoTestUtils {
    fn AdvanceClock(no_gc: &NoGC, global: &GlobalScope, ms: i32) {
        global
            .as_window()
            .advance_animation_clock(no_gc, Duration::milliseconds(ms as i64));
    }

    #[expect(unsafe_code)]
    fn CrashHard(_: &GlobalScope) {
        unsafe { std::ptr::null_mut::<i32>().write(42) }
    }

    fn ForceLayout(cx: &mut JSContext, global: &GlobalScope) -> DomRoot<LayoutResult> {
        let (phases_run, statistics) = global.as_window().Document().update_the_rendering(cx);

        AutomaticRenderingObservation {
            sequence: 0,
            phases: phases_run,
            statistics: AutomaticRenderingStatistics::from(&statistics),
        }
        .result(cx, global)
    }

    fn AutomaticRenderingObservation(
        cx: &mut JSContext,
        global: &GlobalScope,
    ) -> Option<DomRoot<LayoutResult>> {
        global
            .as_window()
            .Document()
            .automatic_rendering_observation()
            .map(|observation| observation.result(cx, global))
    }

    fn AbortSignalFollowerCount(_: &GlobalScope, signal: &AbortSignal) -> u32 {
        signal.dependent_signal_count() as u32
    }

    fn DependentAbortSignalCount(global: &GlobalScope) -> u32 {
        global.dependent_abort_signal_count() as u32
    }

    fn InlineStylePreservationCloneCount(_: &GlobalScope) -> u32 {
        crate::dom::cssstyledeclaration::inline_style_preservation_clone_count()
    }

    fn CompactBackgroundMutationCount(_: &GlobalScope) -> u32 {
        crate::dom::cssstyledeclaration::compact_background_mutation_count()
    }

    fn BackgroundParseAttemptCount(_: &GlobalScope) -> u32 {
        crate::dom::cssstyledeclaration::background_parse_attempt_count()
    }

    fn Js_backtrace(_: &GlobalScope) {
        println!("Current JS stack:");
        let rust_stack = Backtrace::new();
        println!("Current Rust stack:\n{:?}", rust_stack);
    }

    fn Panic(_: &GlobalScope) {
        panic!("explicit panic from script")
    }

    fn EnsureAccessibilityActive(global: &GlobalScope) {
        let window = global.as_window();
        window
            .layout()
            .set_accessibility_active(true, Epoch::default());
    }

    fn ForceAccessibilityUpdate(
        cx: &mut JSContext,
        global: &GlobalScope,
    ) -> DomRoot<AccessibilityUpdateResult> {
        let window = global.as_window();
        window.layout().set_force_accessibility_update();
        let (_, statistics) = window.Document().update_the_rendering(cx);

        AccessibilityUpdateResult::new(
            cx,
            global,
            statistics.nodes_updated_from_dom,
            statistics.nodes_updated_from_tree,
            statistics.nodes_updated_bounds,
            statistics.nodes_in_tree_update,
        )
    }
}

#[cfg(test)]
mod automatic_rendering_observation_tests {
    use super::*;

    #[test]
    fn automatic_observation_is_a_passive_monotonic_copy_of_supplied_updates() {
        let statistics = ReflowStatistics {
            certified_background_color_count: 2,
            certified_background_cached_leaf_count: 2,
            certified_background_cached_family_count: 2,
            ..Default::default()
        };
        let first = AutomaticRenderingObservation::next(
            None,
            ReflowPhasesRun::UpdatedBackgroundColors,
            &statistics,
        )
        .unwrap();
        let copy = first;
        let second = AutomaticRenderingObservation::next(
            Some(first),
            ReflowPhasesRun::empty(),
            &ReflowStatistics::default(),
        )
        .unwrap();
        assert_eq!(first.sequence, 1);
        assert_eq!(second.sequence, 2);
        assert_eq!(copy.statistics.certified_background_color_count, 2);
        assert_eq!(copy.statistics.certified_background_cached_leaf_count, 2);
        assert_eq!(copy.statistics.certified_background_cached_family_count, 2);
        assert_eq!(second.statistics.certified_background_color_count, 0);
        assert!(second.phases.is_empty());
        assert_eq!(copy.sequence, 1);
        assert!(
            copy.phases
                .contains(ReflowPhasesRun::UpdatedBackgroundColors)
        );
    }

    #[test]
    fn automatic_observation_sequence_exhaustion_never_wraps_or_fabricates_freshness() {
        let previous = AutomaticRenderingObservation {
            sequence: u64::MAX,
            ..Default::default()
        };
        assert!(
            AutomaticRenderingObservation::next(
                Some(previous),
                ReflowPhasesRun::BuiltDisplayList,
                &ReflowStatistics::default(),
            )
            .is_none()
        );
        assert_eq!(previous.sequence, u64::MAX);
    }
}
