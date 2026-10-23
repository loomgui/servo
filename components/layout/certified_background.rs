/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! An all-or-nothing eager repair of the complete certified pending list.
//! Preparation borrows authority but changes none. The owned result outlives
//! every style/cache/layout guard and commits only after positive painter ACK.

#![expect(unsafe_code)]

use layout_api::{
    LayoutElement, LayoutNode, ReflowRequestRestyle, RestyleReason, TrustedNodeAddress,
};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use script::layout_dom::{ServoDangerousStyleElement, ServoLayoutElement, ServoLayoutNode};
use servo_config::pref;
use style::context::{SharedStyleContext, StyleContext, ThreadLocalStyleContext};
use style::data::ElementDataRef;
use style::dom::{OpaqueNode, TElement, TNode};
use style::inline_background::{
    BackgroundBatchProof, BackgroundParentProof, BackgroundStyleReuse, CapturedBackgroundStyle,
    PreparedBackgroundStyle, capture_background_style_in_batch, commit_background_style,
    prepare_background_style_in_batch,
};
use style::invalidation::element::restyle_hints::RestyleHint;
use style::shared_lock::StylesheetGuards;
use web_atoms::local_name;

use crate::background_color_cache::{
    BackgroundColorCache, BackgroundColorUpdates, BackgroundLeafReuse, BackgroundStyleReuseStore,
    CertifiedBackgroundBinding,
};
use crate::dom::PreparedEmptyBackgroundLeaf;

pub(crate) fn omission_tag_is_current(
    pending: &layout_api::PendingRestyle,
    epoch: Option<u64>,
) -> bool {
    pending.style_snapshot_omitted.is_some()
        && pending.style_snapshot_omitted == epoch
        && pending.style_attribute_only
        && pending.snapshot.is_none()
        && pending.damage.is_empty()
        && pending.hint.bits() == RestyleHint::RESTYLE_STYLE_ATTRIBUTE.bits()
}

pub(crate) struct CertifiedBackgroundInput {
    targets: Vec<TrustedNodeAddress>,
    ancestors: Vec<TrustedNodeAddress>,
}

impl CertifiedBackgroundInput {
    /// Capture before SnapshotSetter consumes the original pending list. The
    /// script certificate proves these flags were clean before ordinary drain;
    /// the pending metadata proves no other invalidation was coalesced with it.
    pub(crate) fn capture(restyle: &ReflowRequestRestyle) -> Option<Self> {
        if restyle.reason.bits() != RestyleReason::PendingRestyles.bits()
            || restyle.stylesheets_changed
            || restyle.pending_restyles.is_empty()
        {
            return None;
        }
        let ancestors = restyle.certified_background_paths.as_ref()?;
        let mut targets = Vec::with_capacity(restyle.pending_restyles.len());
        let mut identities = FxHashSet::default();
        for (address, pending) in &restyle.pending_restyles {
            if !pending.style_attribute_only
                || !pending.damage.is_empty()
                || pending.hint.bits() != RestyleHint::RESTYLE_STYLE_ATTRIBUTE.bits()
            {
                return None;
            }
            match (&pending.snapshot, pending.style_snapshot_omitted) {
                (Some(snapshot), None)
                    if snapshot.state.is_none()
                        && !snapshot.class_changed
                        && !snapshot.id_changed
                        && snapshot.changed_attrs.as_slice()
                            == [style::LocalName::new(local_name!("style"))] => {},
                // The Layout environment barrier revalidates this exact epoch
                // before this captured input is allowed to prepare anything.
                (None, Some(_)) => {},
                _ => return None,
            }
            let node = unsafe { ServoLayoutNode::new(address) };
            let element = node.as_element()?;
            let data = element.style_data()?.element_data.borrow();
            if !data.hint.is_empty() || !data.damage.is_empty() || !identities.insert(node.opaque())
            {
                return None;
            }
            targets.push(*address);
        }
        // An admitted target is never another target's ancestor. No unrelated
        // path may be cleared by the accepted commit.
        if ancestors
            .iter()
            .any(|address| identities.contains(&unsafe { ServoLayoutNode::new(address) }.opaque()))
        {
            return None;
        }
        Some(Self {
            targets,
            ancestors: ancestors.clone(),
        })
    }

    pub(crate) fn prepare<'dom>(
        self,
        shared: &SharedStyleContext,
        cache: &mut BackgroundColorCache,
        pool: Option<&rayon::ThreadPool>,
        document: OpaqueNode,
    ) -> Option<PreparedBackgroundBatch<'dom>> {
        let mut reusable_styles = cache.detach_reusable_styles();
        if !cache.can_prepare_certified_updates() {
            return None;
        }
        cache.scope_target_leases(
            &mut reusable_styles,
            document,
            shared.inline_background_epoch,
        );
        // Ordinary environment invalidation has already run. Certify every
        // ancestor again before staging; its fresh descendant flag alone must
        // never hide a pending hint, damage or animation-only traversal.
        for address in &self.ancestors {
            let node = unsafe { ServoLayoutNode::new(address) };
            let element = node.as_element()?;
            let dangerous = unsafe { element.dangerous_style_element() };
            let data = element.style_data()?.element_data.borrow();
            if !data.hint.is_empty()
                || !data.damage.is_empty()
                || dangerous.has_animation_only_dirty_descendants()
            {
                return None;
            }
        }
        let mut local = ThreadLocalStyleContext::new();
        let context = StyleContext::<ServoDangerousStyleElement> {
            shared,
            thread_local: &mut local,
        };
        let batch = BackgroundBatchProof::prepare(&context)?;
        let mut parents = FxHashMap::default();
        let collect = shared.inline_background_count.is_some();
        let mut cached_leaf_count = 0usize;
        let mut cached_family_count = 0usize;
        // Only this serial capture touches DOM, layout ownership or cache slots.
        // Retain each ElementData read guard until all pure workers have joined.
        let mut targets = Vec::with_capacity(self.targets.len());
        for address in self.targets {
            let node = unsafe { ServoLayoutNode::new(&address) };
            let element = node.as_element()?;
            let dangerous = unsafe { element.dangerous_style_element() };
            let data = element.style_data()?.element_data.borrow();
            let primary = data.styles.primary.as_ref()?;
            // A warm slot carries owned handles, never style authority. Every
            // current data/Selection/parent/PDB proof below remains fresh.
            let selection = data
                .styles
                .pseudos
                .get(&style::selector_parser::PseudoElement::Selection)?;
            let mut reuse = None;
            let warm = cache
                .take_target_leaf(&mut reusable_styles, node.opaque())
                .and_then(|retired| {
                    let BackgroundLeafReuse { style, layout } = retired;
                    let same = style.matches_primary_selection(
                        primary,
                        selection,
                        shared.inline_background_epoch,
                    );
                    reuse = Some(style);
                    same.then(|| layout.revalidate_owned(primary, selection))
                        .flatten()
                });
            let layout = match warm {
                Some(layout) => layout,
                None => PreparedEmptyBackgroundLeaf::prepare_with_cached(
                    node,
                    primary,
                    selection,
                    |fragment| {
                        cache
                            .bound_fragment(fragment)
                            .map(|binding| (binding, None))
                    },
                )?,
            };
            let binding = layout.binding()?;
            let parent_id = dangerous.traversal_parent()?.as_node().opaque();
            let parent = match parents.entry(parent_id) {
                std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    *entry.insert(BackgroundParentProof::prepare(dangerous)?)
                },
            };
            targets.push(BackgroundPreparationTarget {
                target: node.opaque(),
                element,
                data,
                layout,
                binding,
                parent,
                reuse,
            });
        }
        let parallel = should_parallelize_background_styles(
            pool.is_some(),
            targets.len(),
            pref!(layout_parallelism_job_count_minimum) as usize,
            pref!(layout_parallelism_job_size_minimum) as usize,
        );
        let styles = if parallel {
            PreparedBackgroundStyle::layout_attribution_mark("parallel_prepare");
            // Tokens borrow only immutable style data, never the target's DOM
            // or layout fields. Author and ElementData read guards cover join.
            let captured = targets
                .iter_mut()
                .map(|target| {
                    capture_background_style_in_batch(
                        &context,
                        unsafe { target.element.dangerous_style_element() },
                        &target.data,
                        RestyleHint::RESTYLE_STYLE_ATTRIBUTE,
                        target.reuse.take(),
                        &batch,
                        &target.parent,
                    )
                })
                .collect::<Option<Vec<_>>>()?;
            prepare_captured_background_styles(captured, &shared.guards, pool.unwrap())?
        } else {
            // Preserve the original fused proof/materialization and its caller
            // TLS diagnostic sampling for small batches and missing pools.
            targets
                .iter_mut()
                .map(|target| {
                    prepare_background_style_in_batch(
                        &context,
                        unsafe { target.element.dangerous_style_element() },
                        &target.data,
                        RestyleHint::RESTYLE_STYLE_ATTRIBUTE,
                        target.reuse.take(),
                        &batch,
                        &target.parent,
                    )
                })
                .collect::<Option<Vec<_>>>()?
        };
        let mut entries = Vec::with_capacity(targets.len());
        let mut colors = Vec::with_capacity(targets.len());
        let mut changed_fragments = Vec::with_capacity(targets.len());
        for (target, style) in targets.into_iter().zip(styles) {
            let BackgroundPreparationTarget {
                target,
                element,
                data,
                layout,
                binding,
                ..
            } = target;
            // No read guard may survive into the ACK/atomic mutable commit.
            drop(data);
            if collect {
                cached_leaf_count += usize::from(layout.used_cached_proof());
                cached_family_count += usize::from(style.used_cached_family());
            }
            colors.push(cache.stage_bound_color(
                binding,
                layout.fragment(),
                style.old_primary(),
                style.new_primary(),
            )?);
            changed_fragments.push(layout.fragment().clone());
            entries.push(PreparedEntry {
                target,
                element,
                style,
                layout,
            });
        }
        let updates = cache.finish_certified_updates(colors, changed_fragments)?;
        reusable_styles.reserve_for_commit();
        Some(PreparedBackgroundBatch {
            reusable_styles,
            entries,
            ancestors: self.ancestors,
            updates,
            cached_leaf_count,
            cached_family_count,
        })
    }
}

/// DOM/layout/cache-bearing state stays on the calling Layout thread.
struct BackgroundPreparationTarget<'dom> {
    target: OpaqueNode,
    element: ServoLayoutElement<'dom>,
    data: ElementDataRef<'dom>,
    layout: PreparedEmptyBackgroundLeaf,
    binding: CertifiedBackgroundBinding,
    parent: BackgroundParentProof,
    reuse: Option<BackgroundStyleReuse>,
}

fn should_parallelize_background_styles(
    has_pool: bool,
    count: usize,
    minimum_jobs: usize,
    minimum_job_size: usize,
) -> bool {
    has_pool && count >= minimum_jobs.saturating_mul(minimum_job_size.max(1))
}

fn prepare_captured_background_styles(
    captured: Vec<CapturedBackgroundStyle<'_>>,
    guards: &StylesheetGuards,
    pool: &rayon::ThreadPool,
) -> Option<Vec<PreparedBackgroundStyle>> {
    // Indexed Rayon collection retains target order; a single failed proof
    // drops every candidate and takes the original whole-batch fallback.
    pool.install(|| {
        captured
            .into_par_iter()
            .map(|input| input.prepare(guards))
            .collect()
    })
}

struct PreparedEntry<'dom> {
    target: OpaqueNode,
    element: ServoLayoutElement<'dom>,
    style: PreparedBackgroundStyle,
    layout: PreparedEmptyBackgroundLeaf,
}

pub(crate) struct PreparedBackgroundBatch<'dom> {
    reusable_styles: BackgroundStyleReuseStore,
    entries: Vec<PreparedEntry<'dom>>,
    ancestors: Vec<TrustedNodeAddress>,
    updates: BackgroundColorUpdates,
    cached_leaf_count: usize,
    cached_family_count: usize,
}

impl PreparedBackgroundBatch<'_> {
    pub(crate) fn submit(
        self,
        statistics: &mut layout_api::ReflowStatistics,
        submit: impl FnOnce(u64, Vec<webrender_api::PropertyValue<webrender_api::ColorF>>) -> bool,
        publish: impl FnOnce(BackgroundStyleReuseStore),
    ) -> Option<u32> {
        let Self {
            reusable_styles,
            entries,
            ancestors,
            updates,
            cached_leaf_count,
            cached_family_count,
        } = self;
        let count = u32::try_from(entries.len()).ok()?;
        let accepted = updates.submit_with_reuse(
            statistics,
            submit,
            reusable_styles,
            |reusable_styles| {
                for entry in entries {
                    let reuse =
                        commit_background_style(&mut entry.element.element_data_mut(), entry.style);
                    let layout = entry
                        .layout
                        .commit_for_reuse(reuse.new_primary(), reuse.selection().unwrap());
                    reusable_styles.insert_target(
                        entry.target,
                        BackgroundLeafReuse {
                            style: reuse,
                            layout,
                        },
                    );
                    unsafe {
                        entry
                            .element
                            .dangerous_style_element()
                            .unset_dirty_descendants();
                    }
                }
                for address in ancestors {
                    let node = unsafe { ServoLayoutNode::new(&address) };
                    // Addresses came from the pre-drain normal-ancestor certificate.
                    unsafe {
                        node.as_element()
                            .unwrap()
                            .dangerous_style_element()
                            .unset_dirty_descendants();
                    }
                }
            },
            publish,
        );
        if accepted {
            statistics.certified_background_cached_leaf_count = cached_leaf_count as u32;
            statistics.certified_background_cached_family_count = cached_family_count as u32;
            Some(count)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod unified_background_tests {
    use super::*;
    #[test]
    fn parallel_background_preparation_respects_pool_and_coarsened_threshold() {
        assert!(!should_parallelize_background_styles(false, 10_000, 4, 16));
        assert!(!should_parallelize_background_styles(true, 63, 4, 16));
        assert!(should_parallelize_background_styles(true, 64, 4, 16));
        assert!(should_parallelize_background_styles(true, 10_000, 4, 16));
    }

    #[test]
    fn parallel_background_preparation_handles_zero_and_overflow_preferences() {
        assert!(!should_parallelize_background_styles(false, 0, 0, 0));
        assert!(should_parallelize_background_styles(true, 0, 0, 16));
        assert!(!should_parallelize_background_styles(true, 3, 4, 0));
        assert!(should_parallelize_background_styles(true, 4, 4, 0));
        assert!(!should_parallelize_background_styles(
            true,
            10_000,
            usize::MAX,
            16
        ));
    }

    #[test]
    fn unified_background_omission_tag_rejects_stale_epoch_mixed_history_and_unknown_provenance() {
        let mut pending = layout_api::PendingRestyle {
            style_attribute_only: true,
            style_snapshot_omitted: Some(17),
            hint: RestyleHint::RESTYLE_STYLE_ATTRIBUTE,
            ..Default::default()
        };
        assert!(omission_tag_is_current(&pending, Some(17)));
        assert!(!omission_tag_is_current(&pending, Some(18)));
        assert!(!omission_tag_is_current(&pending, None));
        pending.snapshot = Some(style::selector_parser::Snapshot::new());
        assert!(!omission_tag_is_current(&pending, Some(17)));
        pending.snapshot = None;
        pending.style_attribute_only = false;
        assert!(!omission_tag_is_current(&pending, Some(17)));
        pending.style_attribute_only = true;
        pending.hint.insert(RestyleHint::RESTYLE_SELF);
        assert!(!omission_tag_is_current(&pending, Some(17)));
        pending.hint = RestyleHint::RESTYLE_STYLE_ATTRIBUTE;
        pending.style_snapshot_omitted = None;
        assert!(
            !omission_tag_is_current(&pending, Some(17)),
            "snapshot=None alone is never an omission"
        );
    }
}
