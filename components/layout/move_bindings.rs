/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Move a box without building a new display list.
//!
//! A box that only changes position keeps its contents: the same fragments, in the same
//! painting order, under the same clips. The stacking context tree gives such a box a
//! reference frame whose transform is a WebRender property binding. A later move then
//! only changes that property, and WebRender builds a frame without a new scene.
//!
//! Two kinds of change count as a move:
//!
//! - **Box offsets** (`top`, `right`, `bottom`, `left`) of an absolutely positioned box
//!   whose size does not depend on them. The box gets an extra reference frame
//!   ([`BoundFrameKind::Origin`]) that is the identity until the box moves, and then
//!   translates it by as far as it moved.
//! - **Transform** (`transform`, `translate`, `rotate`, `scale`) of a box that already
//!   has a reference frame ([`BoundFrameKind::Transform`]).
//!
//! Every other change, and every case this module cannot prove to be a move, takes the
//! ordinary path: a new stacking context tree and a new display list.
//!
//! Only boxes that are likely to move get a bound frame, because each one adds a
//! spatial node to the scene and work to every display list. A node is *promoted*:
//!
//! - when a restyle only moved boxes and its box was one of them, or
//! - when its box is created with a `style` attribute that positions it, which is how
//!   script places a box it is about to move (a tooltip, a dragged item).
//!
//! The second is a guess, and a document can prove it wrong: the markers of a minimap
//! are placed the same way, but move in every frame together with other changes, so
//! their frames would never carry a move. When display list after display list is
//! built with bound frames and no box moves through one, the frames are dropped and
//! the guess is no longer made, until a restyle that only moves boxes shows that
//! frames pay off again.

use std::sync::Arc;

use app_units::Au;
use rustc_hash::{FxHashMap, FxHashSet};
use servo_base::id::{ColorBindingId, ScrollTreeNodeId};
use style::dom::OpaqueNode;
use webrender_api::units::LayoutTransform;
use webrender_api::{PropertyBindingKey, PropertyValue};

use crate::fragment_tree::BoxFragment;
use crate::geom::PhysicalVec;

/// The most bound frames one stacking context tree may have. Boxes past this limit are
/// drawn and moved the ordinary way.
const MAX_BOUND_FRAMES: usize = 64;

/// The most promoted nodes to remember. The set is cleared when it grows past this, so
/// that nodes that have gone away do not accumulate.
const MAX_PROMOTED_NODES: usize = 1024;

/// How many display lists in a row may be built with bound frames that no box moves
/// through. One more, and the frames are dropped.
const MAX_UNUSED_DISPLAY_LISTS: u32 = 16;

/// Whether `scale` is a positive power of two: 1, 2, 4, or a half, a quarter. Scaling
/// by one and back is exact in floats.
pub(crate) fn is_power_of_two(scale: f32) -> bool {
    const MANTISSA: u32 = (1 << 23) - 1;
    scale.is_normal() && scale > 0.0 && scale.to_bits() & MANTISSA == 0
}

/// Whether `distance`, at `scale` device pixels per CSS pixel, is a whole number of
/// device pixels, at a scale that is a power of two. A box moved by its offsets only
/// goes through its frame by such distances (see [`BoundFrameKind::Origin`]).
pub(crate) fn is_whole_device_pixels(distance: Au, scale: f32) -> bool {
    let on_screen = distance.to_f64_px() * f64::from(scale);
    is_power_of_two(scale) && (on_screen - on_screen.round()).abs() < 1.0 / 1024.0
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum BoundFrameKind {
    /// A frame added around an absolutely positioned box, to translate it.
    ///
    /// The frame is the identity when its display list is built: the box and its
    /// contents are described in the coordinates of the frame's parent, exactly as
    /// without the frame, so they are drawn exactly as without it. A move makes the
    /// frame translate them, and only by whole device pixels: anything else could
    /// snap text and edges to other pixels than a new display list would.
    Origin {
        /// How far the box has moved since the display list was built.
        offset: PhysicalVec<Au>,
    },
    /// The frame of a box with a CSS transform.
    Transform {
        /// Whether the transform is a 2d scale and translation. WebRender draws such a
        /// frame in its parent's coordinate system, so an update must not change this.
        is_2d_scale_translation: bool,
    },
}

/// A move recorded during damage propagation.
#[derive(Clone, Copy, Debug)]
pub(crate) enum PendingMove {
    /// The box moved by this offset. Its fragment already has the new position.
    Offset(PhysicalVec<Au>),
    /// The box has this new transform.
    Transform(LayoutTransform),
}

pub(crate) struct BoundFrame {
    /// The fragment the frame was made for. Holding it keeps its address from being
    /// reused while this frame is registered.
    pub(crate) fragment: Arc<BoxFragment>,
    pub(crate) node: ScrollTreeNodeId,
    pub(crate) kind: BoundFrameKind,
    /// Whether something inside the frame would not follow it. A tainted frame has no
    /// binding and never moves through one.
    tainted: bool,
    pending: Option<PendingMove>,
    /// The value to send to the painter with the next property update.
    staged: Option<PropertyValue<LayoutTransform>>,
}

/// The moves of one restyle, taken once damage propagation is done.
pub(crate) struct MoveBatch {
    /// Indices into the registered frames, with the move for each.
    pub(crate) moves: Vec<(usize, PendingMove)>,
    /// Scroll containers whose scrollable overflow was invalidated by this restyle.
    pub(crate) scroll_containers: Vec<Arc<BoxFragment>>,
    /// Whether any element had damage that was not handled as a move.
    pub(crate) other_damage: bool,
}

/// Keys that one stacking context tree after another hands out in the same order.
#[derive(Default)]
struct KeyPool {
    keys: Vec<PropertyBindingKey<LayoutTransform>>,
    next: usize,
}

impl KeyPool {
    fn next_key(&mut self) -> PropertyBindingKey<LayoutTransform> {
        if self.next == self.keys.len() {
            // Transform and color properties are separate in WebRender, so the ids only
            // need to be unique among transform bindings.
            self.keys
                .push(PropertyBindingKey::new(ColorBindingId::new().into()));
        }
        let key = self.keys[self.next];
        self.next += 1;
        key
    }
}

#[derive(Default)]
pub(crate) struct MoveBindings {
    /// Nodes whose boxes get a bound frame when the stacking context tree is built.
    promoted: FxHashSet<OpaqueNode>,
    /// Binding keys, reused by every stacking context tree in order, one pool for
    /// each kind of frame. See [`Self::next_key`] for why they are kept apart.
    origin_keys: KeyPool,
    transform_keys: KeyPool,
    /// The bound frames of the current stacking context tree.
    frames: Vec<BoundFrame>,
    by_fragment: FxHashMap<usize, usize>,
    /// The frames that enclose the fragment being added to the stacking context tree.
    open: Vec<usize>,
    /// Whether this restyle may move boxes in place.
    enabled: bool,
    moved: Vec<usize>,
    scroll_containers: Vec<Arc<BoxFragment>>,
    other_damage: bool,
    /// Nodes whose style change in this restyle is a move, but whose box has no bound
    /// frame to move through.
    unbound_movers: Vec<OpaqueNode>,
    staged: Vec<usize>,
    /// The display lists built with bound frames since a box last moved through one.
    unused_display_lists: u32,
    /// Whether bound frames went unused for too long. While this is set, a box gets
    /// no frame just because script positions it.
    suspended: bool,
}

impl MoveBindings {
    /// Remember that the box of `node` should get a bound frame. This is only a hint:
    /// whether a change is a move is decided again each time.
    pub(crate) fn promote(&mut self, node: OpaqueNode) {
        if self.promoted.len() >= MAX_PROMOTED_NODES && !self.promoted.contains(&node) {
            self.promoted.clear();
        }
        self.promoted.insert(node);
    }

    pub(crate) fn is_promoted(&self, node: OpaqueNode) -> bool {
        self.promoted.contains(&node)
    }

    /// The box of `node` was created with a `style` attribute that positions it.
    /// Promote the node on the guess that script will move the box, unless such
    /// frames have been going unused.
    pub(crate) fn promote_positioned_by_script(&mut self, node: OpaqueNode) {
        if !self.suspended {
            self.promote(node);
        }
    }

    /// The style change of `node` in this restyle is a move, but its box has no bound
    /// frame, so it takes the ordinary path. If the restyle turns out to move boxes
    /// and nothing else, the node is promoted when it finishes. Returns false for a
    /// node that is promoted already: its box cannot have a frame, and its change
    /// counts as any other damage.
    pub(crate) fn note_unbound_move(&mut self, node: OpaqueNode) -> bool {
        if self.promoted.contains(&node) {
            return false;
        }
        self.unbound_movers.push(node);
        true
    }

    /// A full display list is built from the current stacking context tree.
    pub(crate) fn note_display_list(&mut self) {
        if !self.frames.is_empty() {
            self.unused_display_lists = self.unused_display_lists.saturating_add(1);
        }
    }

    /// Boxes moved on screen through their frames, without a new display list.
    pub(crate) fn note_binding_update(&mut self) {
        self.unused_display_lists = 0;
    }

    /// Whether the bound frames of the current stacking context tree have gone unused
    /// for so many display lists that the next one should be built without them. If
    /// so, every node loses its promotion; the caller has a new tree built.
    pub(crate) fn drop_unused_frames(&mut self) -> bool {
        if self.frames.is_empty() || self.unused_display_lists < MAX_UNUSED_DISPLAY_LISTS {
            return false;
        }
        self.promoted.clear();
        self.suspended = true;
        self.unused_display_lists = 0;
        true
    }

    /// Forget the frames of the previous stacking context tree.
    pub(crate) fn begin_stacking_context_tree(&mut self) {
        self.frames.clear();
        self.by_fragment.clear();
        self.open.clear();
        self.origin_keys.next = 0;
        self.transform_keys.next = 0;
        self.moved.clear();
        self.staged.clear();
    }

    pub(crate) fn can_add_frame(&self) -> bool {
        self.frames.len() < MAX_BOUND_FRAMES
    }

    /// The binding key for the next frame of the given kind.
    ///
    /// A key never changes kind. The painter keeps the latest value of every key and
    /// can hand it to WebRender while WebRender still draws the previous display
    /// list, in which the same key may belong to another frame. WebRender requires the
    /// transform of a frame that only translates to stay a translation, so a key of
    /// such a frame must never have carried the matrix of a transformed box.
    pub(crate) fn next_key(
        &mut self,
        kind: &BoundFrameKind,
    ) -> PropertyBindingKey<LayoutTransform> {
        match kind {
            BoundFrameKind::Origin { .. } => self.origin_keys.next_key(),
            BoundFrameKind::Transform { .. } => self.transform_keys.next_key(),
        }
    }

    /// Register a frame and mark it as enclosing the fragments added until
    /// [`Self::leave_frame`].
    pub(crate) fn enter_frame(
        &mut self,
        fragment: Arc<BoxFragment>,
        node: ScrollTreeNodeId,
        kind: BoundFrameKind,
    ) {
        let index = self.frames.len();
        self.by_fragment
            .insert(Arc::as_ptr(&fragment) as usize, index);
        self.frames.push(BoundFrame {
            fragment,
            node,
            kind,
            tainted: false,
            pending: None,
            staged: None,
        });
        self.open.push(index);
    }

    pub(crate) fn leave_frame(&mut self) {
        self.open.pop();
    }

    pub(crate) fn has_open_frames(&self) -> bool {
        !self.open.is_empty()
    }

    /// Taint the frames that enclose the current fragment, because it would not follow
    /// them. Returns the spatial nodes that must lose their binding.
    pub(crate) fn taint_open_frames(&mut self, only_origin_frames: bool) -> Vec<ScrollTreeNodeId> {
        let mut nodes = Vec::new();
        for index in &self.open {
            let frame = &mut self.frames[*index];
            if frame.tainted {
                continue;
            }
            if only_origin_frames && !matches!(frame.kind, BoundFrameKind::Origin { .. }) {
                continue;
            }
            frame.tainted = true;
            nodes.push(frame.node);
        }
        nodes
    }

    /// Start a restyle. `enabled` is false when a move must take the ordinary path
    /// whatever its style change looks like.
    pub(crate) fn begin_restyle(&mut self, enabled: bool) {
        self.enabled = enabled;
        self.other_damage = false;
        self.unbound_movers.clear();
        self.scroll_containers.clear();
        for index in self.moved.drain(..) {
            self.frames[index].pending = None;
        }
    }

    pub(crate) fn enabled(&self) -> bool {
        self.enabled
    }

    /// The bound frame of `fragment`, if it has one that can move.
    pub(crate) fn frame_for(&self, fragment: &Arc<BoxFragment>) -> Option<(usize, &BoundFrame)> {
        let index = *self.by_fragment.get(&(Arc::as_ptr(fragment) as usize))?;
        let frame = &self.frames[index];
        (!frame.tainted && Arc::ptr_eq(&frame.fragment, fragment)).then_some((index, frame))
    }

    pub(crate) fn record_move(&mut self, index: usize, pending: PendingMove) {
        let frame = &mut self.frames[index];
        if frame.pending.is_none() {
            self.moved.push(index);
        }
        frame.pending = Some(pending);
    }

    /// An element had damage that is not a move.
    pub(crate) fn note_other_damage(&mut self) {
        self.other_damage = true;
    }

    pub(crate) fn has_moves(&self) -> bool {
        !self.moved.is_empty()
    }

    /// The scrollable overflow of `fragment` was invalidated after a box moved.
    pub(crate) fn note_scroll_container(&mut self, fragment: Arc<BoxFragment>) {
        self.scroll_containers.push(fragment);
    }

    /// Take the moves recorded since [`Self::begin_restyle`].
    pub(crate) fn finish_restyle(&mut self) -> Option<MoveBatch> {
        self.enabled = false;
        let mut other_damage = std::mem::take(&mut self.other_damage);
        let scroll_containers = std::mem::take(&mut self.scroll_containers);
        let unbound_movers = std::mem::take(&mut self.unbound_movers);
        if !unbound_movers.is_empty() {
            // A restyle that only moved boxes would have kept its display list had
            // they all been bound. Bind them in the next one.
            if !other_damage {
                for node in unbound_movers {
                    self.promote(node);
                }
                self.suspended = false;
                self.unused_display_lists = 0;
            }
            // For the boxes that did move through a frame, a move without one is a
            // change their bindings cannot carry.
            other_damage = true;
        }
        if self.moved.is_empty() {
            return None;
        }
        let moves = std::mem::take(&mut self.moved)
            .into_iter()
            .filter_map(|index| Some((index, self.frames[index].pending.take()?)))
            .collect();
        Some(MoveBatch {
            moves,
            scroll_containers,
            other_damage,
        })
    }

    pub(crate) fn frame(&self, index: usize) -> &BoundFrame {
        &self.frames[index]
    }

    /// The stacking context tree now has `frame` at its new place; `value` is what its
    /// bound property must become.
    pub(crate) fn stage(
        &mut self,
        index: usize,
        kind: BoundFrameKind,
        value: PropertyValue<LayoutTransform>,
    ) {
        let frame = &mut self.frames[index];
        frame.kind = kind;
        if frame.staged.is_none() {
            self.staged.push(index);
        }
        frame.staged = Some(value);
    }

    /// Take the values waiting for the painter, with the fragments they belong to.
    pub(crate) fn take_staged(
        &mut self,
    ) -> (Vec<PropertyValue<LayoutTransform>>, Vec<Arc<BoxFragment>>) {
        let mut values = Vec::with_capacity(self.staged.len());
        let mut fragments = Vec::with_capacity(self.staged.len());
        for index in self.staged.drain(..) {
            let frame = &mut self.frames[index];
            if let Some(value) = frame.staged.take() {
                values.push(value);
                fragments.push(frame.fragment.clone());
            }
        }
        (values, fragments)
    }

    pub(crate) fn has_staged(&self) -> bool {
        !self.staged.is_empty()
    }

    /// Drop staged values. The next display list is a full one, which carries the
    /// current place of every frame.
    pub(crate) fn clear_staged(&mut self) {
        for index in self.staged.drain(..) {
            self.frames[index].staged = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use style::properties::ComputedValues;
    use style::properties::style_structs::Font;

    use super::*;
    use crate::fragment_tree::BaseFragmentInfo;
    use crate::geom::PhysicalRect;

    fn fragment() -> Arc<BoxFragment> {
        Arc::new(BoxFragment::new(
            BaseFragmentInfo::new_for_testing(1),
            ComputedValues::initial_values_with_font_override(Font::initial_values()),
            Vec::new(),
            PhysicalRect::zero(),
            Default::default(),
            Default::default(),
            Default::default(),
            None,
        ))
    }

    fn origin_kind() -> BoundFrameKind {
        BoundFrameKind::Origin {
            offset: PhysicalVec::zero(),
        }
    }

    #[test]
    fn keys_are_reused_by_the_next_stacking_context_tree_and_never_change_kind() {
        let transform_kind = BoundFrameKind::Transform {
            is_2d_scale_translation: false,
        };
        let mut bindings = MoveBindings::default();
        bindings.origin_keys.keys = vec![PropertyBindingKey::new(1), PropertyBindingKey::new(2)];
        bindings.transform_keys.keys = vec![PropertyBindingKey::new(3)];
        let first = bindings.next_key(&origin_kind());
        let transform = bindings.next_key(&transform_kind);
        let second = bindings.next_key(&origin_kind());
        assert_eq!(first, PropertyBindingKey::new(1));
        assert_eq!(second, PropertyBindingKey::new(2));
        assert_eq!(transform, PropertyBindingKey::new(3));

        // The next tree asks in another order and gets the same key for each kind.
        bindings.begin_stacking_context_tree();
        assert_eq!(bindings.next_key(&transform_kind), transform);
        assert_eq!(bindings.next_key(&origin_kind()), first);
        assert_eq!(bindings.next_key(&origin_kind()), second);
    }

    #[test]
    fn a_tainted_frame_cannot_move() {
        let mut bindings = MoveBindings::default();
        let outer = fragment();
        let inner = fragment();
        bindings.enter_frame(outer.clone(), ScrollTreeNodeId { index: 2 }, origin_kind());
        bindings.enter_frame(
            inner.clone(),
            ScrollTreeNodeId { index: 3 },
            BoundFrameKind::Transform {
                is_2d_scale_translation: true,
            },
        );
        // Something that only escapes origin frames leaves the transform frame alone.
        assert_eq!(
            bindings.taint_open_frames(true),
            vec![ScrollTreeNodeId { index: 2 }]
        );
        bindings.leave_frame();
        bindings.leave_frame();
        assert!(bindings.frame_for(&outer).is_none());
        assert!(bindings.frame_for(&inner).is_some());
        // A fragment that was never registered has no frame.
        assert!(bindings.frame_for(&fragment()).is_none());
    }

    #[test]
    fn a_restyle_reports_its_moves_once_and_forgets_abandoned_ones() {
        let mut bindings = MoveBindings::default();
        let moved = fragment();
        bindings.enter_frame(moved.clone(), ScrollTreeNodeId { index: 2 }, origin_kind());
        bindings.leave_frame();
        let (index, _) = bindings.frame_for(&moved).unwrap();

        bindings.begin_restyle(true);
        assert!(bindings.finish_restyle().is_none());

        bindings.begin_restyle(true);
        bindings.record_move(index, PendingMove::Offset(PhysicalVec::new(Au(60), Au(0))));
        bindings.note_other_damage();
        let batch = bindings.finish_restyle().unwrap();
        assert_eq!(batch.moves.len(), 1);
        assert!(batch.other_damage);
        assert!(bindings.finish_restyle().is_none());

        // A move whose restyle never finished is not reported by the next one.
        bindings.begin_restyle(true);
        bindings.record_move(index, PendingMove::Offset(PhysicalVec::new(Au(60), Au(0))));
        bindings.begin_restyle(true);
        assert!(bindings.finish_restyle().is_none());
    }

    #[test]
    fn staging_keeps_only_the_latest_value_for_a_frame() {
        let mut bindings = MoveBindings::default();
        let moved = fragment();
        bindings.enter_frame(moved.clone(), ScrollTreeNodeId { index: 2 }, origin_kind());
        bindings.leave_frame();
        let key = PropertyBindingKey::new(9);
        bindings.stage(
            0,
            origin_kind(),
            key.with(LayoutTransform::translation(1.0, 0.0, 0.0)),
        );
        bindings.stage(
            0,
            origin_kind(),
            key.with(LayoutTransform::translation(2.0, 0.0, 0.0)),
        );
        let (values, fragments) = bindings.take_staged();
        assert_eq!(values.len(), 1);
        assert_eq!(values[0].value, LayoutTransform::translation(2.0, 0.0, 0.0));
        assert!(Arc::ptr_eq(&fragments[0], &moved));
        assert!(!bindings.has_staged());

        bindings.stage(
            0,
            origin_kind(),
            key.with(LayoutTransform::translation(3.0, 0.0, 0.0)),
        );
        bindings.clear_staged();
        assert!(bindings.take_staged().0.is_empty());
    }

    #[test]
    fn a_restyle_that_only_moves_unbound_boxes_promotes_them() {
        let mut bindings = MoveBindings::default();

        // A move together with another change promotes nothing.
        bindings.begin_restyle(true);
        assert!(bindings.note_unbound_move(OpaqueNode(1)));
        bindings.note_other_damage();
        assert!(bindings.finish_restyle().is_none());
        assert!(!bindings.is_promoted(OpaqueNode(1)));

        // The same move on its own does.
        bindings.begin_restyle(true);
        assert!(bindings.note_unbound_move(OpaqueNode(1)));
        assert!(bindings.finish_restyle().is_none());
        assert!(bindings.is_promoted(OpaqueNode(1)));

        // A promoted node whose box still has no frame is not a mover without one.
        bindings.begin_restyle(true);
        assert!(!bindings.note_unbound_move(OpaqueNode(1)));
        assert!(bindings.finish_restyle().is_none());
    }

    #[test]
    fn a_move_without_a_frame_keeps_bound_moves_off_their_bindings() {
        let mut bindings = MoveBindings::default();
        let moved = fragment();
        bindings.enter_frame(moved.clone(), ScrollTreeNodeId { index: 2 }, origin_kind());
        bindings.leave_frame();
        let (index, _) = bindings.frame_for(&moved).unwrap();

        bindings.begin_restyle(true);
        bindings.record_move(index, PendingMove::Offset(PhysicalVec::new(Au(60), Au(0))));
        assert!(bindings.note_unbound_move(OpaqueNode(7)));
        let batch = bindings.finish_restyle().unwrap();
        assert!(batch.other_damage);
        // The restyle only moved boxes, so the one without a frame gets one.
        assert!(bindings.is_promoted(OpaqueNode(7)));
    }

    #[test]
    fn unused_frames_are_dropped_until_a_restyle_only_moves_boxes() {
        let mut bindings = MoveBindings::default();
        bindings.promote_positioned_by_script(OpaqueNode(1));
        assert!(bindings.is_promoted(OpaqueNode(1)));

        // Display lists without bound frames do not count.
        for _ in 0..MAX_UNUSED_DISPLAY_LISTS * 2 {
            bindings.note_display_list();
        }
        assert!(!bindings.drop_unused_frames());

        bindings.enter_frame(fragment(), ScrollTreeNodeId { index: 2 }, origin_kind());
        bindings.leave_frame();
        for _ in 0..MAX_UNUSED_DISPLAY_LISTS - 1 {
            bindings.note_display_list();
        }
        assert!(!bindings.drop_unused_frames());

        // A move through a frame starts the count again.
        bindings.note_binding_update();
        for _ in 0..MAX_UNUSED_DISPLAY_LISTS - 1 {
            bindings.note_display_list();
        }
        assert!(!bindings.drop_unused_frames());
        bindings.note_display_list();
        assert!(bindings.drop_unused_frames());
        assert!(!bindings.is_promoted(OpaqueNode(1)));

        // From here on a box that script positions gets no frame.
        bindings.promote_positioned_by_script(OpaqueNode(2));
        assert!(!bindings.is_promoted(OpaqueNode(2)));

        // Until a restyle only moves boxes.
        bindings.begin_restyle(true);
        assert!(bindings.note_unbound_move(OpaqueNode(3)));
        assert!(bindings.finish_restyle().is_none());
        assert!(bindings.is_promoted(OpaqueNode(3)));
        bindings.promote_positioned_by_script(OpaqueNode(2));
        assert!(bindings.is_promoted(OpaqueNode(2)));
    }

    #[test]
    fn the_promoted_set_is_bounded() {
        let mut bindings = MoveBindings::default();
        for id in 0..MAX_PROMOTED_NODES {
            bindings.promote(OpaqueNode(id));
        }
        assert!(bindings.is_promoted(OpaqueNode(0)));
        bindings.promote(OpaqueNode(MAX_PROMOTED_NODES));
        assert!(!bindings.is_promoted(OpaqueNode(0)));
        assert!(bindings.is_promoted(OpaqueNode(MAX_PROMOTED_NODES)));
    }

    #[test]
    fn only_powers_of_two_scale_exactly() {
        for scale in [1.0, 2.0, 4.0, 0.5, 0.25] {
            assert!(is_power_of_two(scale), "{scale}");
        }
        for scale in [1.25, 1.5, 3.0, 0.75, 0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(!is_power_of_two(scale), "{scale}");
        }
    }

    #[test]
    fn whole_device_pixels() {
        assert!(is_whole_device_pixels(Au::from_px(3), 1.0));
        assert!(is_whole_device_pixels(Au::from_f32_px(-1.5), 2.0));
        assert!(!is_whole_device_pixels(Au::from_f32_px(2.5), 0.5));
        assert!(!is_whole_device_pixels(Au::from_f32_px(0.4), 1.0));
        assert!(!is_whole_device_pixels(Au::from_f32_px(1.5), 1.0));
        // 4px is 5 device pixels at 125%, but that scale is not exact.
        assert!(!is_whole_device_pixels(Au::from_px(4), 1.25));
    }
}
