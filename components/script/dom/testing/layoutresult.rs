/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */
use dom_struct::dom_struct;
use js::context::JSContext;
use js::gc::MutableHandleValue;
use script_bindings::domstring::DOMString;
use script_bindings::reflector::{Reflector, reflect_dom_object};

use crate::dom::bindings::codegen::Bindings::ServoTestUtilsBinding::LayoutResultMethods;
use crate::dom::bindings::root::DomRoot;
use crate::dom::bindings::utils::to_frozen_array;
use crate::dom::globalscope::GlobalScope;

#[dom_struct]
pub(crate) struct LayoutResult {
    reflector_: Reflector,
    phases: Vec<DOMString>,
    inline_background_color_recascade_count: u32,
    certified_background_color_count: u32,
    certified_background_cached_leaf_count: u32,
    certified_background_cached_family_count: u32,
    rebuilt_fragment_count: u32,
    restyle_fragment_count: u32,
    only_descendants_changed_count: u32,
}

impl LayoutResult {
    pub(crate) fn new_inherited(
        phases: Vec<DOMString>,
        inline_background_color_recascade_count: u32,
        certified_background_color_count: u32,
        certified_background_cached_leaf_count: u32,
        certified_background_cached_family_count: u32,
        rebuilt_fragment_count: u32,
        restyle_fragment_count: u32,
        only_descendants_changed_count: u32,
    ) -> Self {
        Self {
            reflector_: Reflector::new(),
            phases,
            inline_background_color_recascade_count,
            certified_background_color_count,
            certified_background_cached_leaf_count,
            certified_background_cached_family_count,
            rebuilt_fragment_count,
            restyle_fragment_count,
            only_descendants_changed_count,
        }
    }

    pub(crate) fn new(
        cx: &mut JSContext,
        global: &GlobalScope,
        phases: Vec<DOMString>,
        inline_background_color_recascade_count: u32,
        certified_background_color_count: u32,
        certified_background_cached_leaf_count: u32,
        certified_background_cached_family_count: u32,
        rebuilt_fragment_count: u32,
        restyle_fragment_count: u32,
        only_descendants_changed_count: u32,
    ) -> DomRoot<Self> {
        reflect_dom_object(
            cx,
            Box::new(Self::new_inherited(
                phases,
                inline_background_color_recascade_count,
                certified_background_color_count,
                certified_background_cached_leaf_count,
                certified_background_cached_family_count,
                rebuilt_fragment_count,
                restyle_fragment_count,
                only_descendants_changed_count,
            )),
            global,
        )
    }
}

impl LayoutResultMethods<crate::DomTypeHolder> for LayoutResult {
    fn Phases(&self, cx: &mut JSContext, return_value: MutableHandleValue) {
        to_frozen_array(cx, &self.phases, return_value);
    }

    fn InlineBackgroundColorRecascadeCount(&self) -> u32 {
        self.inline_background_color_recascade_count
    }

    fn CertifiedBackgroundColorCount(&self) -> u32 {
        self.certified_background_color_count
    }

    fn CertifiedBackgroundCachedLeafCount(&self) -> u32 {
        self.certified_background_cached_leaf_count
    }

    fn CertifiedBackgroundCachedFamilyCount(&self) -> u32 {
        self.certified_background_cached_family_count
    }

    fn RebuiltFragmentCount(&self) -> u32 {
        self.rebuilt_fragment_count
    }

    fn RestyleFragmentCount(&self) -> u32 {
        self.restyle_fragment_count
    }

    fn OnlyDescendantsChangedCount(&self) -> u32 {
        self.only_descendants_changed_count
    }
}
