/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Retain the identities of background primitives, never rasterized pixels.
//! The damage walk or a complete certified empty-leaf batch may add deltas.
//! Any unrecognized repaint makes
//! the whole batch ineligible until a complete display list replaces it.

use std::sync::Arc;

use layout_api::ReflowStatistics;
use rustc_hash::FxHashMap;
use servo_base::id::ColorBindingId;
use style::data::BackgroundColorChangeWitness;
use style::inline_background::BackgroundStyleReuse;
use style::properties::ComputedValues;
use webrender_api::{ColorF, PropertyBindingKey, PropertyValue};

use crate::display_list::rgba;
use crate::fragment_tree::{BoxFragment, FragmentFlags, FragmentStatus};
use crate::layout_box_base::LayoutBoxBase;

#[derive(Default)]
pub(crate) struct BackgroundColorCache {
    // Holding the Arc ensures that an address cannot be recycled within this
    // display list's lifetime, even if a later reflow reconstructs a subtree.
    bindings: FxHashMap<usize, (Arc<BoxFragment>, PropertyBindingKey<ColorF>)>,
    keys: Vec<PropertyBindingKey<ColorF>>,
    next_key: usize,
    generation: u64,
    pending: Option<Vec<PropertyValue<ColorF>>>,
    changed_fragments: Vec<Arc<BoxFragment>>,
    reusable_styles: BackgroundStyleReuseStore,
}

/// Non-authoritative, bounded by this display list's retained bindings. A slot
/// retains one retired CV/Background plus its exact-current/family references.
/// Clearing a warmed population synchronously drops O(population) references,
/// including on a sparse fallback; no deferred cleanup or history is retained.
pub(crate) struct BackgroundLeafReuse {
    pub(crate) style: BackgroundStyleReuse,
    pub(crate) proof: crate::dom::EmptyBackgroundLeafProof,
}

/// One live binding checked under the exclusive prepare guard, never GPU authority.
#[derive(Clone, Copy)]
pub(crate) struct CertifiedBackgroundBinding {
    generation: u64,
    fragment: usize,
    key: PropertyBindingKey<ColorF>,
}

pub(crate) struct BackgroundStyleReuseStore<T = BackgroundLeafReuse> {
    generation: u64,
    entries: FxHashMap<usize, Option<T>>,
    additional_slots: usize,
}

impl<T> Default for BackgroundStyleReuseStore<T> {
    fn default() -> Self {
        Self {
            generation: 0,
            entries: FxHashMap::default(),
            additional_slots: 0,
        }
    }
}

impl<T> BackgroundStyleReuseStore<T> {
    fn clear(&mut self) {
        self.entries.clear();
    }

    fn detach(&mut self, generation: u64) -> Self {
        if self.generation != generation {
            self.clear();
        }
        Self {
            generation,
            entries: std::mem::take(&mut self.entries),
            additional_slots: 0,
        }
    }

    fn publish(&mut self, generation: u64, eligible: bool, prepared: Self) {
        if eligible && prepared.generation == generation {
            *self = prepared;
        }
    }

    fn take(&mut self, identity: usize) -> Option<T> {
        match self.entries.get_mut(&identity) {
            Some(slot) => slot.take(),
            None => {
                self.additional_slots += 1;
                None
            },
        }
    }

    /// Before ACK, reserve only genuinely new binding slots. Existing keys stay
    /// in place with an empty value while detached, avoiding tombstone/rehash
    /// allocation and capacity growth on steady dense or sparse updates.
    pub(crate) fn reserve_for_commit(&mut self) {
        self.entries.reserve(self.additional_slots);
    }

    pub(crate) fn insert(&mut self, fragment: &Arc<BoxFragment>, value: T) {
        self.entries
            .insert(Arc::as_ptr(fragment) as usize, Some(value));
    }
}

impl BackgroundColorCache {
    pub(crate) fn invalidate(&mut self) {
        self.pending = None;
        self.changed_fragments.clear();
        self.clear_reusable_styles();
    }

    /// Selection highlights use static display-list colors, including inherited
    /// `::selection` colors. Keep this entire display list ineligible for color
    /// deltas until a full rebuild no longer paints a selection highlight.
    pub(crate) fn record_selection_highlight(&mut self) {
        self.invalidate();
    }

    pub(crate) fn begin_display_list(&mut self) {
        self.generation = self
            .generation
            .checked_add(1)
            .expect("Background display-list generation exhausted");
        self.clear_reusable_styles();
        self.bindings.clear();
        self.next_key = 0;
        self.changed_fragments.clear();
        self.pending = Some(Vec::new());
    }

    pub(crate) fn clear_reusable_styles(&mut self) {
        self.reusable_styles.clear();
    }

    /// Detach all scratch before preparing. Failure/drop destroys the detached
    /// population without reacquiring this cache or retaining a guard over ACK.
    pub(crate) fn detach_reusable_styles(&mut self) -> BackgroundStyleReuseStore {
        self.reusable_styles.detach(self.generation)
    }

    pub(crate) fn take_bound_leaf(
        &self,
        prepared: &mut BackgroundStyleReuseStore,
        fragment: &Arc<BoxFragment>,
    ) -> Option<(CertifiedBackgroundBinding, Option<BackgroundLeafReuse>)> {
        if prepared.generation != self.generation || !self.can_prepare_certified_updates() {
            return None;
        }
        let identity = Arc::as_ptr(fragment) as usize;
        let (bound, key) = self.bindings.get(&identity)?;
        if !Arc::ptr_eq(bound, fragment) {
            return None;
        }
        Some((
            CertifiedBackgroundBinding {
                generation: self.generation,
                fragment: identity,
                key: *key,
            },
            prepared.take(identity),
        ))
    }

    /// The same exclusive prepare validated current base/fragment/style/status.
    /// This fuses color staging with that target; no second hash/style-borrow pass.
    pub(crate) fn stage_bound_color(
        &self,
        binding: CertifiedBackgroundBinding,
        fragment: &Arc<BoxFragment>,
        old: &ComputedValues,
        new: &ComputedValues,
    ) -> Option<PropertyValue<ColorF>> {
        if binding.generation != self.generation
            || binding.fragment != Arc::as_ptr(fragment) as usize
        {
            return None;
        }
        let old_color = old.resolve_color(&old.get_background().background_color);
        let new_color = new.resolve_color(&new.get_background().background_color);
        if old_color.alpha != 1.0 || new_color.alpha != 1.0 {
            return None;
        }
        Some(binding.key.with(rgba(new_color)))
    }

    pub(crate) fn finish_certified_updates(
        &self,
        colors: Vec<PropertyValue<ColorF>>,
        changed_fragments: Vec<Arc<BoxFragment>>,
    ) -> Option<BackgroundColorUpdates> {
        if !self.can_prepare_certified_updates()
            || colors.is_empty()
            || colors.len() != changed_fragments.len()
        {
            return None;
        }
        Some(BackgroundColorUpdates {
            generation: self.generation,
            colors,
            changed_fragments,
        })
    }

    /// Only after every style/leaf repair and changed-fragment cleanup. A
    /// generation/eligibility mismatch destroys scratch, never republishes it.
    pub(crate) fn publish_reusable_styles(&mut self, prepared: BackgroundStyleReuseStore) {
        let eligible = self.can_prepare_certified_updates();
        self.reusable_styles
            .publish(self.generation, eligible, prepared);
    }

    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    pub(crate) fn next_key(&mut self) -> PropertyBindingKey<ColorF> {
        if self.next_key == self.keys.len() {
            self.keys
                .push(PropertyBindingKey::new(ColorBindingId::new().into()));
        }
        let key = self.keys[self.next_key];
        self.next_key += 1;
        key
    }

    pub(crate) fn bind(&mut self, fragment: &Arc<BoxFragment>) -> PropertyBindingKey<ColorF> {
        // A fragment may be painted more than once; all of its primitives must
        // share a binding rather than leave earlier paints on an obsolete key.
        if let Some((_, key)) = self.bindings.get(&(Arc::as_ptr(fragment) as usize)) {
            return *key;
        }
        let key = self.next_key();
        self.bindings
            .insert(Arc::as_ptr(fragment) as usize, (fragment.clone(), key));
        key
    }

    pub(crate) fn record_style_change(
        &mut self,
        base: &LayoutBoxBase,
        new: &ComputedValues,
        witness: Option<&BackgroundColorChangeWitness>,
    ) {
        if self.pending.is_none() || std::ptr::eq(&*base.style, new) {
            return;
        }
        let old = &base.style;
        if !background_color_only(old, new, witness) {
            self.invalidate();
            return;
        }
        if old.get_background().background_color == new.get_background().background_color {
            return;
        }
        let old_color = old.resolve_color(&old.get_background().background_color);
        let new_color = new.resolve_color(&new.get_background().background_color);
        if old_color.alpha != 1.0
            || new_color.alpha != 1.0
            || base.base_fragment_info.flags.intersects(
                FragmentFlags::IS_ROOT_ELEMENT
                    | FragmentFlags::IS_BODY_ELEMENT_OF_HTML_ELEMENT_ROOT,
            )
        {
            self.invalidate();
            return;
        }
        let fragments = base.fragments();
        if fragments.is_empty() {
            self.invalidate();
            return;
        }
        for fragment in fragments.iter() {
            let Some(fragment) = fragment.retrieve_box_fragment() else {
                self.invalidate();
                return;
            };
            let Some((bound_fragment, key)) = self.bindings.get(&(Arc::as_ptr(&fragment) as usize))
            else {
                self.invalidate();
                return;
            };
            self.pending
                .as_mut()
                .unwrap()
                .push(key.with(rgba(new_color)));
            self.changed_fragments.push(bound_fragment.clone());
        }
    }

    pub(crate) fn can_prepare_certified_updates(&self) -> bool {
        self.pending.as_ref().is_some_and(Vec::is_empty) && self.changed_fragments.is_empty()
    }

    /// Prepare an owned batch without consuming cache or fragment authority.
    /// The style producer certifies the exact pair; every current binding and
    /// fragment identity is independently checked here before any submission.
    pub(crate) fn prepare_certified_updates<'a>(
        &self,
        changes: impl IntoIterator<
            Item = (
                &'a style::inline_background::PreparedBackgroundStyle,
                &'a Arc<BoxFragment>,
            ),
        >,
    ) -> Option<BackgroundColorUpdates> {
        self.prepare_certified_pairs(
            changes
                .into_iter()
                .map(|(style, fragment)| (style.old_primary(), style.new_primary(), fragment)),
        )
    }

    fn prepare_certified_pairs<'a>(
        &self,
        changes: impl IntoIterator<
            Item = (
                &'a servo_arc::Arc<ComputedValues>,
                &'a servo_arc::Arc<ComputedValues>,
                &'a Arc<BoxFragment>,
            ),
        >,
    ) -> Option<BackgroundColorUpdates> {
        if !self.can_prepare_certified_updates() {
            return None;
        }
        let mut colors = Vec::new();
        let mut changed_fragments = Vec::new();
        for (old, new, fragment) in changes {
            let old_color = old.resolve_color(&old.get_background().background_color);
            let new_color = new.resolve_color(&new.get_background().background_color);
            if old_color.alpha != 1.0
                || new_color.alpha != 1.0
                || fragment.base.flags.intersects(
                    FragmentFlags::IS_ROOT_ELEMENT
                        | FragmentFlags::IS_BODY_ELEMENT_OF_HTML_ELEMENT_ROOT,
                )
                || fragment.base.status() != FragmentStatus::Clean
                || !servo_arc::Arc::ptr_eq(&fragment.style(), old)
            {
                return None;
            }
            let (bound, key) = self.bindings.get(&(Arc::as_ptr(fragment) as usize))?;
            if !Arc::ptr_eq(bound, fragment) {
                return None;
            }
            colors.push(key.with(rgba(new_color)));
            changed_fragments.push(fragment.clone());
        }
        if colors.is_empty() {
            return None;
        }
        Some(BackgroundColorUpdates {
            generation: self.generation,
            colors,
            changed_fragments,
        })
    }

    /// Taking a batch does not commit fragment status or accounting.
    pub(crate) fn take_updates(&mut self) -> Option<BackgroundColorUpdates> {
        let updates = self.pending.as_mut()?;
        if updates.is_empty() {
            return None;
        }
        Some(BackgroundColorUpdates {
            generation: self.generation,
            colors: std::mem::take(updates),
            changed_fragments: std::mem::take(&mut self.changed_fragments),
        })
    }
}

/// A producer proof applies only to its exact final style pair. A stale or
/// unrelated proof follows the ordinary comparison path in this same call.
fn background_color_only(
    old: &ComputedValues,
    new: &ComputedValues,
    witness: Option<&BackgroundColorChangeWitness>,
) -> bool {
    if witness
        .is_some_and(|witness| std::ptr::eq(&*witness.old, old) && std::ptr::eq(&*witness.new, new))
    {
        return true;
    }
    #[cfg(test)]
    STRICT_BACKGROUND_COMPARISONS.with(|count| count.set(count.get() + 1));
    old.same_except_background_color(new)
}

#[cfg(test)]
std::thread_local! {
    static STRICT_BACKGROUND_COMPARISONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// An owned transaction batch, so no layout cache mutex crosses the reply wait.
pub(crate) struct BackgroundColorUpdates {
    generation: u64,
    colors: Vec<PropertyValue<ColorF>>,
    changed_fragments: Vec<Arc<BoxFragment>>,
}

impl BackgroundColorUpdates {
    pub(crate) fn submit(
        self,
        statistics: &mut ReflowStatistics,
        submit: impl FnOnce(u64, Vec<PropertyValue<ColorF>>) -> bool,
    ) -> bool {
        self.submit_with_commit(statistics, submit, || {})
    }

    /// The same transaction with detached scratch. Publication occurs only
    /// after submit_with_commit has cleaned every repaired fragment. Rejection,
    /// abandoned preparation, or unwind drops owned scratch without a cache lock.
    pub(crate) fn submit_with_reuse<R>(
        self,
        statistics: &mut ReflowStatistics,
        submit: impl FnOnce(u64, Vec<PropertyValue<ColorF>>) -> bool,
        mut reuse: R,
        commit: impl FnOnce(&mut R),
        publish: impl FnOnce(R),
    ) -> bool {
        if !self.submit_with_commit(statistics, submit, || commit(&mut reuse)) {
            return false;
        }
        publish(reuse);
        true
    }

    /// Commit staged style/layout references only after positive painter ACK,
    /// then clean the fragments repaired by that commit. A rejected or dropped
    /// reply leaves all staged authority untouched for ordinary fallback.
    pub(crate) fn submit_with_commit(
        self,
        statistics: &mut ReflowStatistics,
        submit: impl FnOnce(u64, Vec<PropertyValue<ColorF>>) -> bool,
        commit: impl FnOnce(),
    ) -> bool {
        if !submit(self.generation, self.colors) {
            return false;
        }
        commit();
        // Commit only after the painter accepted and queued these exact values.
        for fragment in self.changed_fragments {
            if fragment.base.status() == FragmentStatus::StyleChanged {
                statistics.restyle_fragment_count += 1;
                fragment.base.set_status(FragmentStatus::Clean);
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fragment_tree::{BaseFragmentInfo, Fragment};
    use crate::geom::PhysicalRect;
    use servo_arc::Arc as ServoArc;
    use style::color::AbsoluteColor;
    use style::properties::style_structs::Font;
    use style::values::computed::color::Color;

    fn collect_updates(
        cache: &mut BackgroundColorCache,
        statistics: &mut ReflowStatistics,
    ) -> Option<Vec<PropertyValue<ColorF>>> {
        let batch = cache.take_updates()?;
        let mut values = None;
        assert!(batch.submit(statistics, |_, colors| {
            values = Some(colors);
            true
        }));
        values
    }

    fn style(color: Color) -> ServoArc<ComputedValues> {
        let mut style =
            (*ComputedValues::initial_values_with_font_override(Font::initial_values())).clone();
        style.mutate_background().background_color = color;
        ServoArc::new(style)
    }

    fn bound_box(
        cache: &mut BackgroundColorCache,
        id: usize,
        style: ServoArc<ComputedValues>,
    ) -> LayoutBoxBase {
        let info = BaseFragmentInfo::new_for_testing(id);
        let fragment = Arc::new(BoxFragment::new(
            info,
            style.clone(),
            Vec::new(),
            PhysicalRect::zero(),
            Default::default(),
            Default::default(),
            Default::default(),
            None,
        ));
        fragment.base.set_status(FragmentStatus::Clean);
        cache.bindings.insert(
            Arc::as_ptr(&fragment) as usize,
            (fragment.clone(), PropertyBindingKey::new(id as u64)),
        );
        let base = LayoutBoxBase::new(info, style);
        base.set_fragment(Fragment::Box(fragment));
        base
    }

    fn change(
        cache: &mut BackgroundColorCache,
        base: &mut LayoutBoxBase,
        style: &ServoArc<ComputedValues>,
    ) {
        cache.record_style_change(base, style, None);
        base.repair_style(style);
    }

    #[test]
    fn unified_background_fused_binding_stages_exact_colors_and_rejects_stale_generation() {
        let old = style(Color::BLACK);
        let new = style(Color::WHITE);
        let mut cache = BackgroundColorCache::default();
        cache.begin_display_list();
        let base = bound_box(&mut cache, 81, old.clone());
        let fragments = base.fragments();
        let fragment = fragments[0].retrieve_box_fragment().unwrap();
        let mut retired = cache.detach_reusable_styles();
        let (binding, reuse) = cache.take_bound_leaf(&mut retired, &fragment).unwrap();
        assert!(reuse.is_none());
        let value = cache
            .stage_bound_color(binding, &fragment, &old, &new)
            .unwrap();
        let batch = cache
            .finish_certified_updates(vec![value], vec![fragment.clone()])
            .unwrap();
        assert!(!batch.submit(&mut ReflowStatistics::default(), |_, _| false));
        assert!(ServoArc::ptr_eq(&fragment.style(), &old));
        assert_eq!(fragment.base.status(), FragmentStatus::Clean);
        assert!(
            cache
                .finish_certified_updates(Vec::new(), Vec::new())
                .is_none()
        );
        cache.begin_display_list();
        assert!(
            cache
                .stage_bound_color(binding, &fragment, &old, &new)
                .is_none()
        );
        assert!(cache.take_bound_leaf(&mut retired, &fragment).is_none());
    }

    #[test]
    fn key_pool_reuses_full_list_ids_and_grows_without_caret_collisions() {
        std::thread::spawn(|| {
            use servo_base::id::{PipelineId, PipelineNamespace, PipelineNamespaceId};
            PipelineNamespace::install(PipelineNamespaceId(10_003));
            let mut cache = BackgroundColorCache::default();
            cache.begin_display_list();
            let first: Vec<_> = (0..100).map(|_| cache.next_key().id).collect();
            let caret: PropertyBindingKey<ColorF> =
                PropertyBindingKey::new(PipelineId::new().into());
            assert!(!first.contains(&caret.id));
            cache.begin_display_list();
            let reused: Vec<_> = (0..100).map(|_| cache.next_key().id).collect();
            assert_eq!(first, reused);
            let grown = cache.next_key();
            assert!(!first.contains(&grown.id));
            assert_ne!(grown.id, caret.id);
            assert_eq!(cache.generation(), 2);
            assert_eq!(cache.keys.len(), 101);
            let base = bound_box(&mut cache, 7, style(Color::BLACK));
            let fragments = base.fragments();
            let fragment = fragments[0].retrieve_box_fragment().unwrap();
            cache.bindings.clear();
            let first = cache.bind(&fragment);
            let used = cache.next_key;
            assert_eq!(cache.bind(&fragment), first);
            assert_eq!(cache.next_key, used);
        })
        .join()
        .unwrap();
    }

    #[test]
    fn certified_batch_stages_without_mutation_and_commits_only_after_positive_ack() {
        let black = style(Color::BLACK);
        let white = style(Color::WHITE);
        for accepted in [false, true] {
            let mut cache = BackgroundColorCache::default();
            cache.begin_display_list();
            let base = std::cell::RefCell::new(bound_box(&mut cache, 81, black.clone()));
            let fragment = {
                let base = base.borrow();
                let fragments = base.fragments();
                Arc::clone(&fragments[0].retrieve_box_fragment().unwrap())
            };
            let mut statistics = ReflowStatistics::default();
            let batch = cache
                .prepare_certified_pairs([(&black, &white, &fragment)])
                .unwrap();
            assert!(cache.pending.as_ref().unwrap().is_empty());
            assert!(cache.changed_fragments.is_empty());
            assert!(servo_arc::Arc::ptr_eq(&base.borrow().style, &black));
            assert!(
                batch.submit_with_commit(
                    &mut statistics,
                    |generation, values| {
                        assert_eq!(generation, cache.generation());
                        assert_eq!(values.len(), 1);
                        assert_eq!(values[0].value, ColorF::WHITE);
                        assert!(servo_arc::Arc::ptr_eq(&fragment.style(), &black));
                        assert_eq!(fragment.base.status(), FragmentStatus::Clean);
                        accepted
                    },
                    || {
                        base.borrow_mut().repair_style(&white);
                        assert_eq!(fragment.base.status(), FragmentStatus::StyleChanged);
                    }
                ) == accepted
            );
            assert!(servo_arc::Arc::ptr_eq(
                &base.borrow().style,
                if accepted { &white } else { &black }
            ));
            assert_eq!(fragment.base.status(), FragmentStatus::Clean);
            assert_eq!(statistics.restyle_fragment_count, u32::from(accepted));
        }
    }

    #[test]
    fn certified_batch_missing_identity_or_registration_rejects_whole_unchanged_batch() {
        let black = style(Color::BLACK);
        let white = style(Color::WHITE);
        let mut cache = BackgroundColorCache::default();
        cache.begin_display_list();
        let first = bound_box(&mut cache, 91, black.clone());
        let second = bound_box(&mut cache, 92, black.clone());
        let first_fragments = first.fragments();
        let second_fragments = second.fragments();
        let first_fragment = first_fragments[0].retrieve_box_fragment().unwrap();
        let second_fragment = second_fragments[0].retrieve_box_fragment().unwrap();
        cache
            .bindings
            .remove(&(Arc::as_ptr(&second_fragment) as usize));
        assert!(
            cache
                .prepare_certified_pairs([
                    (&black, &white, &*first_fragment),
                    (&black, &white, &*second_fragment),
                ])
                .is_none()
        );
        let equal_but_different_arc = style(Color::BLACK);
        assert!(
            cache
                .prepare_certified_pairs([(&equal_but_different_arc, &white, &*first_fragment)])
                .is_none()
        );
        assert!(cache.pending.as_ref().unwrap().is_empty());
        assert!(cache.changed_fragments.is_empty());
        assert!(servo_arc::Arc::ptr_eq(&first.style, &black));
        assert!(servo_arc::Arc::ptr_eq(&second.style, &black));
        assert_eq!(first_fragment.base.status(), FragmentStatus::Clean);
        assert_eq!(second_fragment.base.status(), FragmentStatus::Clean);
    }

    fn comparisons_during(run: impl FnOnce()) -> usize {
        STRICT_BACKGROUND_COMPARISONS.with(|count| count.set(0));
        run();
        STRICT_BACKGROUND_COMPARISONS.with(|count| count.get())
    }

    #[test]
    fn exact_witness_skips_comparison_but_keeps_repair_and_ack_commit() {
        let black = style(Color::BLACK);
        let white = style(Color::WHITE);
        let witness = BackgroundColorChangeWitness {
            old: black.clone(),
            new: white.clone(),
        };
        let mut cache = BackgroundColorCache::default();
        cache.begin_display_list();
        let mut base = bound_box(&mut cache, 1, black);
        assert_eq!(
            comparisons_during(|| {
                cache.record_style_change(&base, &white, Some(&witness));
                // The producer proof does not replace ordinary fragment repair.
                base.repair_style(&white);
            }),
            0,
            "an exact producer proof must remove the strict comparison"
        );
        assert!(ServoArc::ptr_eq(&base.style, &white));
        let fragments = base.fragments();
        let fragment = fragments[0].retrieve_box_fragment().unwrap();
        assert!(ServoArc::ptr_eq(&fragment.style(), &white));
        assert_eq!(fragment.base.status(), FragmentStatus::StyleChanged);
        let mut statistics = ReflowStatistics::default();
        let batch = cache.take_updates().unwrap();
        assert!(batch.submit(&mut statistics, |_, colors| {
            assert_eq!(colors.len(), 1);
            assert_eq!(colors[0].value, ColorF::WHITE);
            true
        }));
        assert_eq!(fragment.base.status(), FragmentStatus::Clean);
        assert_eq!(statistics.restyle_fragment_count, 1);
    }

    #[test]
    fn mismatched_witnesses_use_strict_comparison_and_preserve_fallback() {
        let black = style(Color::BLACK);
        let white = style(Color::WHITE);
        for mismatch_old in [false, true] {
            let original = if mismatch_old { &black } else { &white };
            let mut changed = (**original).clone();
            changed.mutate_effects().opacity = 0.5;
            let changed = ServoArc::new(changed);
            let (actual_old, actual_new) = if mismatch_old {
                (changed, white.clone())
            } else {
                (black.clone(), changed)
            };
            let witness = BackgroundColorChangeWitness {
                old: black.clone(),
                new: white.clone(),
            };
            let mut cache = BackgroundColorCache::default();
            cache.begin_display_list();
            let base = bound_box(&mut cache, 1, actual_old);
            assert_eq!(
                comparisons_during(|| {
                    cache.record_style_change(&base, &actual_new, Some(&witness));
                }),
                1
            );
            assert!(
                cache.pending.is_none(),
                "a stale proof must not admit an opacity change"
            );
        }
        // Equal values with different identities are also unproven, but the
        // normal comparator may still safely accept their background change.
        let witness = BackgroundColorChangeWitness {
            old: style(Color::BLACK),
            new: style(Color::WHITE),
        };
        let mut cache = BackgroundColorCache::default();
        cache.begin_display_list();
        let base = bound_box(&mut cache, 1, black);
        assert_eq!(
            comparisons_during(|| {
                cache.record_style_change(&base, &white, Some(&witness));
            }),
            1
        );
        assert_eq!(cache.take_updates().unwrap().colors.len(), 1);
    }

    #[test]
    fn exact_witness_keeps_root_body_alpha_and_binding_guards() {
        let black = style(Color::BLACK);
        let white = style(Color::WHITE);
        for flags in [
            FragmentFlags::IS_ROOT_ELEMENT,
            FragmentFlags::IS_BODY_ELEMENT_OF_HTML_ELEMENT_ROOT,
            FragmentFlags::empty(),
        ] {
            let mut cache = BackgroundColorCache::default();
            cache.begin_display_list();
            let mut base = bound_box(&mut cache, 1, black.clone());
            base.base_fragment_info.flags.insert(flags);
            if flags.is_empty() {
                cache.bindings.clear();
            }
            let witness = BackgroundColorChangeWitness {
                old: black.clone(),
                new: white.clone(),
            };
            assert_eq!(
                comparisons_during(|| {
                    cache.record_style_change(&base, &white, Some(&witness));
                }),
                0
            );
            assert!(cache.pending.is_none());
        }
        let translucent = style(Color::Absolute(AbsoluteColor::srgb_legacy(255, 0, 0, 0.5)));
        for (old, new) in [(black.clone(), translucent.clone()), (translucent, white)] {
            let witness = BackgroundColorChangeWitness {
                old: old.clone(),
                new: new.clone(),
            };
            let mut cache = BackgroundColorCache::default();
            cache.begin_display_list();
            let base = bound_box(&mut cache, 1, old);
            cache.record_style_change(&base, &new, Some(&witness));
            assert!(cache.pending.is_none());
        }
    }

    #[test]
    fn sparse_and_dense_batches_collect_only_changed_boxes() {
        let black = style(Color::BLACK);
        let white = style(Color::WHITE);
        let mut cache = BackgroundColorCache::default();
        cache.begin_display_list();
        let mut boxes: Vec<_> = (1..=10_000)
            .map(|id| bound_box(&mut cache, id, black.clone()))
            .collect();
        for base in &mut boxes[..100] {
            change(&mut cache, base, &white);
        }
        let mut statistics = ReflowStatistics::default();
        assert_eq!(
            collect_updates(&mut cache, &mut statistics).unwrap().len(),
            100
        );
        assert_eq!(statistics.restyle_fragment_count, 100);
        assert!(collect_updates(&mut cache, &mut statistics).is_none());
        for base in &mut boxes[..100] {
            change(&mut cache, base, &black);
        }
        assert_eq!(
            collect_updates(&mut cache, &mut statistics).unwrap().len(),
            100
        );
        for base in &mut boxes {
            change(&mut cache, base, &white);
        }
        assert_eq!(
            collect_updates(&mut cache, &mut statistics).unwrap().len(),
            10_000
        );
        assert_eq!(statistics.restyle_fragment_count, 10_200);
    }

    #[test]
    fn any_other_style_change_invalidates_the_whole_batch() {
        let black = style(Color::BLACK);
        let white = style(Color::WHITE);
        let mut cache = BackgroundColorCache::default();
        cache.begin_display_list();
        let mut first = bound_box(&mut cache, 1, black.clone());
        let mut second = bound_box(&mut cache, 2, black.clone());
        change(&mut cache, &mut first, &white);
        let mut changed = (*white).clone();
        changed.mutate_effects().opacity = 0.5;
        assert!(!black.same_except_background_color(&changed));
        change(&mut cache, &mut second, &ServoArc::new(changed));
        assert!(cache.take_updates().is_none());
        assert!(cache.pending.is_none());
    }

    #[test]
    fn root_body_alpha_and_missing_binding_force_full_display_list() {
        let black = style(Color::BLACK);
        let white = style(Color::WHITE);
        for flag in [
            FragmentFlags::IS_ROOT_ELEMENT,
            FragmentFlags::IS_BODY_ELEMENT_OF_HTML_ELEMENT_ROOT,
        ] {
            let mut cache = BackgroundColorCache::default();
            cache.begin_display_list();
            let mut base = bound_box(&mut cache, 1, black.clone());
            base.base_fragment_info.flags.insert(flag);
            change(&mut cache, &mut base, &white);
            assert!(cache.pending.is_none());
        }
        let mut cache = BackgroundColorCache::default();
        cache.begin_display_list();
        let mut base = bound_box(&mut cache, 1, black.clone());
        let translucent = style(Color::Absolute(AbsoluteColor::srgb_legacy(255, 0, 0, 0.5)));
        change(&mut cache, &mut base, &translucent);
        assert!(cache.pending.is_none());
        cache.begin_display_list();
        change(&mut cache, &mut base, &white);
        assert!(cache.pending.is_none());
    }

    #[test]
    fn full_display_list_retires_fragments_and_recovers_eligibility() {
        let black = style(Color::BLACK);
        let white = style(Color::WHITE);
        let mut cache = BackgroundColorCache::default();
        cache.begin_display_list();
        let mut base = bound_box(&mut cache, 1, black.clone());
        change(&mut cache, &mut base, &white);
        cache.invalidate();
        assert!(cache.take_updates().is_none());
        cache.begin_display_list();
        assert!(cache.bindings.is_empty());
        assert!(cache.changed_fragments.is_empty());
        let mut replacement = bound_box(&mut cache, 2, black);
        change(&mut cache, &mut replacement, &white);
        let values = collect_updates(&mut cache, &mut ReflowStatistics::default()).unwrap();
        assert_eq!(values.len(), 1);
        assert_eq!(values[0].key.id.uid, 2);
    }

    #[test]
    fn selection_highlight_keeps_later_bindings_ineligible_until_full_rebuild() {
        std::thread::spawn(|| {
            use servo_base::id::{PipelineNamespace, PipelineNamespaceId};
            PipelineNamespace::install(PipelineNamespaceId(10_004));
            for selection_before_first_binding in [false, true] {
                let black = style(Color::BLACK);
                let white = style(Color::WHITE);
                let mut cache = BackgroundColorCache::default();
                cache.begin_display_list();
                let mut first = bound_box(&mut cache, 1, black.clone());
                let mut second = bound_box(&mut cache, 2, black.clone());
                let first_fragment = {
                    let fragments = first.fragments();
                    let fragment = fragments[0].retrieve_box_fragment().unwrap();
                    Arc::clone(&fragment)
                };
                let second_fragment = {
                    let fragments = second.fragments();
                    let fragment = fragments[0].retrieve_box_fragment().unwrap();
                    Arc::clone(&fragment)
                };
                cache.bindings.clear();

                if selection_before_first_binding {
                    cache.record_selection_highlight();
                }
                cache.bind(&first_fragment);
                if !selection_before_first_binding {
                    cache.record_selection_highlight();
                }
                // Display-list traversal can register more primary background
                // bindings after the static highlight, but cannot revive deltas.
                cache.bind(&second_fragment);
                change(&mut cache, &mut first, &white);
                change(&mut cache, &mut second, &white);
                assert!(
                    cache.take_updates().is_none(),
                    "a selection highlight must keep later background bindings ineligible"
                );
                assert!(cache.pending.is_none());
                assert_eq!(first_fragment.base.status(), FragmentStatus::StyleChanged);
                assert_eq!(second_fragment.base.status(), FragmentStatus::StyleChanged);

                // A subsequent full list with no selected highlight restores
                // ordinary bindings and commits the next exact color changes.
                cache.begin_display_list();
                cache.bind(&first_fragment);
                cache.bind(&second_fragment);
                change(&mut cache, &mut first, &black);
                change(&mut cache, &mut second, &black);
                let mut statistics = ReflowStatistics::default();
                assert_eq!(
                    collect_updates(&mut cache, &mut statistics).unwrap().len(),
                    2
                );
                assert_eq!(statistics.restyle_fragment_count, 2);
                assert_eq!(first_fragment.base.status(), FragmentStatus::Clean);
                assert_eq!(second_fragment.base.status(), FragmentStatus::Clean);
            }
        })
        .join()
        .unwrap();
    }

    #[test]
    fn generated_comparator_checks_repaint_and_nonproperty_metadata() {
        let black = style(Color::BLACK);
        let mut changed = (*style(Color::WHITE)).clone();
        assert!(black.same_except_background_color(&changed));
        changed.mutate_inherited_text().color = AbsoluteColor::WHITE;
        assert!(!black.same_except_background_color(&changed));
        let mut changed = (*black).clone();
        changed
            .flags
            .insert(style::computed_value_flags::ComputedValueFlags::HAS_TEXT_DECORATION_LINES);
        assert!(!black.same_except_background_color(&changed));
    }
}

#[cfg(test)]
mod reuse_tests {
    use super::*;
    use crate::fragment_tree::BaseFragmentInfo;
    use crate::geom::PhysicalRect;
    use paint_api::{CrossProcessPaintApi, PaintMessage};
    use servo_base::Epoch;
    use servo_base::id::{TEST_PIPELINE_ID, TEST_WEBVIEW_ID};
    use std::cell::Cell;
    use std::rc::Rc;
    use style::properties::style_structs::Font;

    struct Retired(Rc<Cell<usize>>);
    impl Drop for Retired {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    #[test]
    fn certified_background_reuse_dense_store_detach_reserve_invalidate_and_generation_drop() {
        let dropped = Rc::new(Cell::new(0));
        let mut store = BackgroundStyleReuseStore::<Retired>::default();
        // Use the actual production container: invalidation must drop the whole
        // warmed population even when the rejected next opportunity is sparse.
        store.generation = 7;
        for key in 0..10_000 {
            store.entries.insert(key, Some(Retired(dropped.clone())));
        }
        let capacity = store.entries.capacity();
        let mut detached = store.detach(7);
        assert!(store.entries.is_empty());
        for key in 0..10_000 {
            drop(detached.take(key).unwrap());
        }
        assert_eq!(detached.additional_slots, 0);
        detached.reserve_for_commit();
        assert_eq!(
            detached.entries.capacity(),
            capacity,
            "steady selected slots must not grow capacity"
        );
        for key in 0..10_000 {
            detached.entries.insert(key, Some(Retired(dropped.clone())));
        }
        store.publish(7, true, detached);
        assert_eq!(store.entries.len(), 10_000);
        assert_eq!(dropped.get(), 10_000);
        store.clear(); // Exact operation called by fallback, invalidation and full DL.
        assert_eq!(dropped.get(), 20_000);
        assert!(store.entries.is_empty());
        store.entries.insert(1, Some(Retired(dropped.clone())));
        let detached = store.detach(7);
        store.publish(7, false, detached);
        assert_eq!(dropped.get(), 20_001);
        store.entries.insert(1, Some(Retired(dropped.clone())));
        let detached = store.detach(7);
        store.publish(8, true, detached);
        assert_eq!(dropped.get(), 20_002);
        store.entries.insert(1, Some(Retired(dropped.clone())));
        drop(store.detach(9)); // Changed display-list/session generation retires old entries.
        assert_eq!(dropped.get(), 20_003);
        store.entries.insert(1, Some(Retired(dropped.clone())));
        drop(store); // Session teardown owns and retires its complete population.
        assert_eq!(dropped.get(), 20_004);
    }

    fn fragment() -> Arc<BoxFragment> {
        Arc::new(BoxFragment::new(
            BaseFragmentInfo::new_for_testing(902),
            ComputedValues::initial_values_with_font_override(Font::initial_values()),
            Vec::new(),
            PhysicalRect::zero(),
            Default::default(),
            Default::default(),
            Default::default(),
            None,
        ))
    }

    fn batch(fragment: &Arc<BoxFragment>) -> BackgroundColorUpdates {
        BackgroundColorUpdates {
            generation: 7,
            colors: vec![PropertyBindingKey::new(1).with(ColorF::WHITE)],
            changed_fragments: vec![fragment.clone()],
        }
    }

    #[test]
    fn certified_background_reuse_false_and_disconnected_ack_drop_without_publication_then_retry() {
        for reply in [Some(false), None] {
            let fragment = fragment();
            fragment.base.set_status(FragmentStatus::Clean);
            let dropped = Rc::new(Cell::new(0));
            let published = Cell::new(false);
            let mut statistics = ReflowStatistics::default();
            let api = CrossProcessPaintApi::dummy_with_callback(Some(Box::new(move |message| {
                if let PaintMessage::UpdateBackgroundColors { response, .. } = message {
                    if let Some(reply) = reply {
                        let _ = response.send(reply);
                    }
                }
            })));
            assert!(!batch(&fragment).submit_with_reuse(
                &mut statistics,
                |generation, colors| api.update_background_colors(
                    TEST_WEBVIEW_ID,
                    TEST_PIPELINE_ID,
                    Epoch(7),
                    generation,
                    colors
                ),
                Retired(dropped.clone()),
                |_| panic!("rejected ACK cannot commit"),
                |_| published.set(true)
            ));
            assert!(!published.get());
            assert_eq!(dropped.get(), 1);
            assert_eq!(fragment.base.status(), FragmentStatus::Clean);
            assert_eq!(statistics.restyle_fragment_count, 0);
            assert!(batch(&fragment).submit_with_reuse(
                &mut statistics,
                |_, _| true,
                Retired(dropped.clone()),
                |_| fragment.base.set_status(FragmentStatus::StyleChanged),
                |retired| {
                    assert_eq!(fragment.base.status(), FragmentStatus::Clean);
                    published.set(true);
                    drop(retired);
                }
            ));
            assert!(published.get());
            assert_eq!(dropped.get(), 2);
            assert_eq!(statistics.restyle_fragment_count, 1);
        }
    }

    #[test]
    fn certified_background_reuse_abandoned_batch_and_commit_unwind_never_publish() {
        let fragment = fragment();
        fragment.base.set_status(FragmentStatus::Clean);
        let dropped = Rc::new(Cell::new(0));
        let mut store = BackgroundStyleReuseStore::default();
        store.entries.insert(1, Some(Retired(dropped.clone())));
        drop((batch(&fragment), store));
        assert_eq!(dropped.get(), 1);
        let published = Cell::new(false);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            batch(&fragment).submit_with_reuse(
                &mut ReflowStatistics::default(),
                |_, _| true,
                Retired(dropped.clone()),
                |_| panic!("injected owner repair unwind"),
                |_| published.set(true),
            );
        }));
        assert!(result.is_err());
        assert!(!published.get());
        assert_eq!(dropped.get(), 2);
        assert_eq!(fragment.base.status(), FragmentStatus::Clean);
    }
}

#[cfg(test)]
mod rejected_batch_tests {
    use super::*;
    use crate::fragment_tree::BaseFragmentInfo;
    use crate::geom::PhysicalRect;
    use paint_api::{CrossProcessPaintApi, PaintMessage};
    use servo_base::Epoch;
    use servo_base::id::{TEST_PIPELINE_ID, TEST_WEBVIEW_ID};
    use style::properties::style_structs::Font;

    fn api(reply: Option<bool>) -> CrossProcessPaintApi {
        CrossProcessPaintApi::dummy_with_callback(Some(Box::new(move |message| {
            if let PaintMessage::UpdateBackgroundColors { response, .. } = message {
                if let Some(reply) = reply {
                    let _ = response.send(reply);
                }
                // None deliberately drops the owned reply sender.
            }
        })))
    }

    #[test]
    fn rejected_or_disconnected_ack_preserves_dirty_state_then_retry_commits() {
        for (reply, eager) in [Some(false), None]
            .into_iter()
            .flat_map(|reply| [false, true].map(|eager| (reply, eager)))
        {
            let style = ComputedValues::initial_values_with_font_override(Font::initial_values());
            let fragment = Arc::new(BoxFragment::new(
                BaseFragmentInfo::new_for_testing(1),
                style,
                Vec::new(),
                PhysicalRect::zero(),
                Default::default(),
                Default::default(),
                Default::default(),
                None,
            ));
            fragment.base.set_status(FragmentStatus::StyleChanged);
            let batch = || BackgroundColorUpdates {
                generation: 1,
                colors: vec![PropertyBindingKey::new(1).with(ColorF::WHITE)],
                changed_fragments: vec![fragment.clone()],
            };
            let mut statistics = ReflowStatistics::default();
            let (sent, received) = std::sync::mpsc::channel();
            let paint_api = CrossProcessPaintApi::dummy_with_callback(Some(Box::new(
                move |message| match message {
                    PaintMessage::UpdateBackgroundColors { response, .. } => {
                        if let Some(reply) = reply {
                            let _ = response.send(reply);
                        }
                    },
                    message @ PaintMessage::SendDisplayList { .. } => {
                        sent.send(message).unwrap();
                    },
                    _ => {},
                },
            )));
            let dirty = std::cell::Cell::new(!eager);
            let cache = parking_lot::Mutex::new(BackgroundColorCache::default());
            if eager {
                cache.lock().begin_display_list();
                fragment.base.set_status(FragmentStatus::Clean);
                let accepted = crate::layout_impl::LayoutThread::submit_certified_background_update(
                    &cache,
                    &dirty,
                    &mut statistics,
                    |statistics| {
                        batch()
                            .submit_with_commit(
                                statistics,
                                |generation, colors| {
                                    paint_api.update_background_colors(
                                        TEST_WEBVIEW_ID,
                                        TEST_PIPELINE_ID,
                                        Epoch(7),
                                        generation,
                                        colors,
                                    )
                                },
                                || fragment.base.set_status(FragmentStatus::StyleChanged),
                            )
                            .then_some(1)
                    },
                );
                assert!(accepted.is_none());
                assert!(
                    dirty.get(),
                    "early rejected ACK must force a fresh full list in this reflow"
                );
                assert!(cache.lock().pending.is_none());
                assert_eq!(fragment.base.status(), FragmentStatus::Clean);
                assert_eq!(statistics.restyle_fragment_count, 0);
                // The unchanged ordinary damage/repair continuation still owns
                // the rejected style mutation; no eager state was committed.
                fragment.base.set_status(FragmentStatus::StyleChanged);
            }
            let phase = crate::layout_impl::LayoutThread::submit_render_update(
                (!eager).then(batch),
                &cache,
                &dirty,
                &mut statistics,
                |generation, colors| {
                    paint_api.update_background_colors(
                        TEST_WEBVIEW_ID,
                        TEST_PIPELINE_ID,
                        Epoch(7),
                        generation,
                        colors,
                    )
                },
                |_| {
                    // The production fallback still sees uncommitted damage.
                    assert_eq!(fragment.base.status(), FragmentStatus::StyleChanged);
                    use paint_api::display_list::{
                        AxesScrollSensitivity, PaintDisplayListInfo, ScrollType,
                    };
                    use webrender_api::units::{LayoutRect, LayoutSize};
                    use webrender_api::{
                        CommonItemProperties, DisplayListBuilder, SpaceAndClipInfo,
                    };
                    let pipeline = TEST_PIPELINE_ID.into();
                    let bounds = LayoutRect::from_size(LayoutSize::new(1.0, 1.0));
                    let mut builder = DisplayListBuilder::new(pipeline);
                    builder.begin();
                    builder.push_rect(
                        &CommonItemProperties::new(bounds, SpaceAndClipInfo::root_scroll(pipeline)),
                        bounds,
                        ColorF::WHITE,
                    );
                    let info = PaintDisplayListInfo::new(
                        Default::default(),
                        LayoutSize::new(1.0, 1.0),
                        pipeline,
                        Epoch(7),
                        AxesScrollSensitivity {
                            x: ScrollType::empty(),
                            y: ScrollType::empty(),
                        },
                        false,
                    );
                    paint_api.send_display_list(TEST_WEBVIEW_ID, &info, builder.end().1);
                    layout_api::ReflowPhasesRun::BuiltDisplayList
                },
            );
            assert_eq!(phase, layout_api::ReflowPhasesRun::BuiltDisplayList);
            assert!(!phase.contains(layout_api::ReflowPhasesRun::UpdatedBackgroundColors));
            assert!(!dirty.get());
            let message = received
                .recv_timeout(std::time::Duration::from_secs(1))
                .unwrap();
            let PaintMessage::SendDisplayList {
                display_list_descriptor,
                display_list_info_receiver,
                display_list_data_receiver,
                ..
            } = message
            else {
                panic!("expected full list in same call");
            };
            assert_eq!(display_list_info_receiver.recv().unwrap().epoch, Epoch(7));
            let data = display_list_data_receiver.recv().unwrap();
            let list = webrender_api::BuiltDisplayList::from_data(
                webrender_api::DisplayListPayload {
                    items_data: data.items_data,
                    spatial_tree: data.spatial_tree,
                },
                display_list_descriptor,
            );
            let mut iter = list.iter();
            let item = iter.next().unwrap();
            let webrender_api::DisplayItem::Rectangle(rect) = item.item() else {
                panic!("expected changed rectangle");
            };
            assert_eq!(
                rect.color,
                webrender_api::PropertyBinding::Value(ColorF::WHITE)
            );
            assert_eq!(fragment.base.status(), FragmentStatus::StyleChanged);
            assert_eq!(statistics.restyle_fragment_count, 0);
            // Full display-list fallback still observes StyleChanged. A later
            // acknowledged batch may commit normally after the cache is reset.
            dirty.set(true);
            let phase = if eager {
                cache.lock().begin_display_list();
                fragment.base.set_status(FragmentStatus::Clean);
                let count = crate::layout_impl::LayoutThread::submit_certified_background_update(
                    &cache,
                    &dirty,
                    &mut statistics,
                    |statistics| {
                        batch()
                            .submit_with_commit(
                                statistics,
                                |generation, colors| {
                                    api(Some(true)).update_background_colors(
                                        TEST_WEBVIEW_ID,
                                        TEST_PIPELINE_ID,
                                        Epoch(8),
                                        generation,
                                        colors,
                                    )
                                },
                                || fragment.base.set_status(FragmentStatus::StyleChanged),
                            )
                            .then_some(1)
                    },
                );
                assert_eq!(count, Some(1));
                layout_api::ReflowPhasesRun::UpdatedBackgroundColors
            } else {
                let phase = crate::layout_impl::LayoutThread::submit_render_update(
                    Some(batch()),
                    &cache,
                    &dirty,
                    &mut statistics,
                    |generation, colors| {
                        api(Some(true)).update_background_colors(
                            TEST_WEBVIEW_ID,
                            TEST_PIPELINE_ID,
                            Epoch(8),
                            generation,
                            colors,
                        )
                    },
                    |_| panic!("accepted retry must not rebuild"),
                );
                phase
            };
            assert_eq!(phase, layout_api::ReflowPhasesRun::UpdatedBackgroundColors);
            assert!(!dirty.get());
            assert_eq!(fragment.base.status(), FragmentStatus::Clean);
            assert_eq!(statistics.restyle_fragment_count, 1);
        }
    }
}
