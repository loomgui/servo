/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::sync::Arc;

use app_units::Au;
use script::layout_dom::ServoLayoutNode;
use style::properties::ComputedValues;

use crate::context::LayoutContext;
use crate::dom::{LayoutBox, NodeExt};
use crate::flexbox::FlexLevelBox;
use crate::flow::BlockLevelBox;
use crate::flow::inline::InlineItem;
use crate::fragment_tree::{BoxFragment, Fragment};
use crate::geom::PhysicalVec;
use crate::positioned::AbsolutelyPositionedBox;

/// A data structure to track a layout root.
///
/// Layout roots are places in the fragment tree where fragment tree layout can start.
/// These are fragments that isolate fragment damage from their ancestors sufficiently
/// that their ancestors can be preserved from one fragment tree layout to the next.
/// **Currently, this only includes absolutely-or-fixed-positioned fragments that do not
/// have any escaping fixed position fragments.**.
///
/// During damage propagation, upward flowing fragment tree `Relayout` damage can be
/// converted into `DescendantCollectedAsLayoutRoot` damage. When an ancestor of a layout
/// root has fragment tree layout damage, `DescendantCollectedAsLayoutRoot` damage is
/// converted back into `Relayout` and propagated up to the next layout root or the root
/// of the DOM.
///
/// This is only possible because the damage propagation traversal ensures that damage
/// above the absolute (which might shift its static position) is converted back into
/// `Relayout` damage. This means that the absolute should be placed with the fully
/// adjusted static positioning rectangle between two layouts.
///
/// It's possible that upon laying out a layout root again, a new fixed position element
/// has started to escape from the layout root. In that case, a full fragment tree layout
/// becomes necessary to rebuild the fragment properly.
pub(crate) struct LayoutRoot<'dom> {
    node: ServoLayoutNode<'dom>,
}

impl<'dom> TryFrom<ServoLayoutNode<'dom>> for LayoutRoot<'dom> {
    type Error = ();

    fn try_from(node: ServoLayoutNode<'dom>) -> Result<Self, Self::Error> {
        if !node.is_absolutely_positioned() {
            return Err(());
        }

        // Only accept previously known layout roots without fragmentation as compatible layout roots.
        let mut is_layout_root = false;
        node.with_layout_box_base(|base| {
            let fragments = base.fragments();
            is_layout_root =
                fragments.len() == 1 && matches!(fragments.first(), Some(Fragment::LayoutRoot(..)));
        });
        if !is_layout_root {
            return Err(());
        }

        Ok(Self { node })
    }
}

impl<'dom> LayoutRoot<'dom> {
    /// The absolutely positioned box of this layout root.
    fn positioned_box(&self) -> Option<crate::cell::ArcRefCell<AbsolutelyPositionedBox>> {
        let inner_layout_data = self.node.inner_layout_data()?;
        let layout_box = inner_layout_data.self_box.clone();

        let layout_box = layout_box.borrow();
        let layout_box = layout_box.as_ref()?;

        Some(match layout_box {
            LayoutBox::BlockLevel(block_level) => {
                let mut block_level = block_level.borrow_mut();
                match &mut *block_level {
                    BlockLevelBox::OutOfFlowAbsolutelyPositionedBox(positioned_box) => {
                        positioned_box.clone()
                    },
                    _ => return None,
                }
            },
            LayoutBox::InlineLevel(InlineItem::OutOfFlowAbsolutelyPositionedBox(
                positioned_box,
                _,
            )) => positioned_box.clone(),
            LayoutBox::FlexLevel(flex_level_box) => {
                let mut flex_level_box = flex_level_box.borrow_mut();
                match &mut *flex_level_box {
                    FlexLevelBox::OutOfFlowAbsolutelyPositionedBox(positioned_box) => {
                        positioned_box.clone()
                    },
                    _ => return None,
                }
            },
            _ => return None,
        })
    }

    /// The style of this layout root changes to `new_style`, which differs from its
    /// current style only in box offsets. If that only moves the box, return its
    /// fragment and how far it moves; the fragment and its contents are otherwise
    /// what a new layout would produce. Nothing is changed here.
    pub(crate) fn offset_for_box_offsets_change(
        &self,
        new_style: &ComputedValues,
    ) -> Option<(Arc<BoxFragment>, PhysicalVec<Au>)> {
        let positioned_box = self.positioned_box()?;
        let positioned_box = positioned_box.borrow();
        let formatting_context = &positioned_box.context;
        let fragments = formatting_context.base.fragments();
        let Some(Fragment::LayoutRoot(layout_root_fragment)) = fragments.first() else {
            return None;
        };
        let layout_inputs = formatting_context.layout_root_layout_inputs.borrow();
        let offset = layout_inputs.as_ref()?.offset_for_box_offsets_change(
            formatting_context,
            formatting_context.style(),
            new_style,
        )?;
        let box_fragment = Arc::clone(&*layout_root_fragment.inner_box_fragment());
        Some((box_fragment, offset))
    }

    pub(crate) fn try_layout(&self, layout_context: &LayoutContext) -> bool {
        let Some(positioned_box) = self.positioned_box() else {
            return false;
        };

        let positioned_box = positioned_box.borrow();
        let formatting_context = &positioned_box.context;
        let fragments = formatting_context.base.fragments();
        let Some(Fragment::LayoutRoot(layout_root_fragment)) = fragments.first() else {
            return false;
        };

        let layout_inputs = formatting_context.layout_root_layout_inputs.borrow();
        let Some(layout_inputs) = &*layout_inputs else {
            return false;
        };

        // If an `Err` is returned here, that means that this layout root is no longer
        // a viable layout root and a full fragment tree layout is necessary.
        layout_inputs
            .layout(
                layout_context,
                formatting_context,
                &layout_root_fragment.fragment,
            )
            .is_ok()
    }

    pub(crate) fn handle_failed_layout_root_layout(&self) {
        self.node
            .clear_fragments_and_dirty_fragment_caches_recursively();
    }

    pub(crate) fn node(&self) -> ServoLayoutNode<'dom> {
        self.node
    }
}
