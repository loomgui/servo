/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Retain the identities of background primitives, never rasterized pixels.
//! Only the normal damage walk may add deltas. Any unrecognized repaint makes
//! the whole batch ineligible until a complete display list replaces it.

use std::sync::Arc;

use layout_api::ReflowStatistics;
use rustc_hash::FxHashMap;
use servo_base::id::ColorBindingId;
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
}

impl BackgroundColorCache {
    pub(crate) fn invalidate(&mut self) {
        self.pending = None;
        self.changed_fragments.clear();
    }

    pub(crate) fn begin_display_list(&mut self) {
        self.generation = self
            .generation
            .checked_add(1)
            .expect("Background display-list generation exhausted");
        self.bindings.clear();
        self.next_key = 0;
        self.changed_fragments.clear();
        self.pending = Some(Vec::new());
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

    pub(crate) fn record_style_change(&mut self, base: &LayoutBoxBase, new: &ComputedValues) {
        if self.pending.is_none() || std::ptr::eq(&*base.style, new) {
            return;
        }
        let old = &base.style;
        if !old.same_except_background_color(new) {
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
        if !submit(self.generation, self.colors) {
            return false;
        }
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
        cache.record_style_change(base, style);
        base.repair_style(style);
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
        for reply in [Some(false), None] {
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
            let dirty = std::cell::Cell::new(true);
            let cache = parking_lot::Mutex::new(BackgroundColorCache::default());
            let phase = crate::layout_impl::LayoutThread::submit_render_update(
                Some(batch()),
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
            assert_eq!(phase, layout_api::ReflowPhasesRun::UpdatedBackgroundColors);
            assert!(!dirty.get());
            assert_eq!(fragment.base.status(), FragmentStatus::Clean);
            assert_eq!(statistics.restyle_fragment_count, 1);
        }
    }
}
