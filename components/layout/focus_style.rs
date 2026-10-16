/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Non-consuming current-style proof for the two focusability style fields.
//! Nothing here is style-commit, cache-reuse, paint or frame authority.

#![expect(unsafe_code)]

use layout_api::{BackgroundFocusStyleRequest, LayoutElement, LayoutNode, PendingRestyle};
use rustc_hash::{FxHashMap, FxHashSet};
use script::layout_dom::ServoLayoutNode;
use servo_arc::Arc;
use style::context::{SharedStyleContext, StyleContext, ThreadLocalStyleContext};
use style::dom::{TElement, TNode};
use style::inline_background::{
    BackgroundBatchProof, BackgroundParentProof, validate_background_style_before_drain,
};
use style::invalidation::element::restyle_hints::RestyleHint;
use style::properties::ComputedValues;
use web_atoms::local_name;

fn exact_pending_background_style(
    pending: &PendingRestyle,
    lease: Option<u64>,
    epoch: Option<u64>,
) -> bool {
    if !pending.style_attribute_only
        || !pending.damage.is_empty()
        || pending.hint.bits() != RestyleHint::RESTYLE_STYLE_ATTRIBUTE.bits()
    {
        return false;
    }
    match (&pending.snapshot, pending.style_snapshot_omitted) {
        (Some(snapshot), None) => {
            snapshot.state.is_none()
                && !snapshot.class_changed
                && !snapshot.id_changed
                && snapshot.changed_attrs.as_slice()
                    == [style::LocalName::new(local_name!("style"))]
        },
        (None, Some(tag)) => Some(tag) == lease && Some(tag) == epoch,
        _ => false,
    }
}

pub(crate) fn validate(
    request: BackgroundFocusStyleRequest<'_>,
    shared: &SharedStyleContext,
) -> Option<Arc<ComputedValues>> {
    if request.pending_restyles.is_empty() {
        return None;
    }
    let mut identities = FxHashSet::default();
    for (address, pending) in &request.pending_restyles {
        if !exact_pending_background_style(
            pending,
            request.snapshot_lease,
            shared.inline_background_epoch,
        ) || !identities.insert(unsafe { ServoLayoutNode::new(address) }.opaque())
        {
            return None;
        }
    }
    for address in &request.ancestors {
        let node = unsafe { ServoLayoutNode::new(address) };
        if identities.contains(&node.opaque()) {
            return None;
        }
        let element = node.as_element()?;
        let dangerous = unsafe { element.dangerous_style_element() };
        let data = element.style_data()?.element_data.borrow();
        if !data.hint.is_empty()
            || !data.damage.is_empty()
            || dangerous.has_dirty_descendants()
            || dangerous.has_animation_only_dirty_descendants()
        {
            return None;
        }
    }
    let mut local = ThreadLocalStyleContext::new();
    let context = StyleContext {
        shared,
        thread_local: &mut local,
    };
    let batch = BackgroundBatchProof::prepare(&context)?;
    let mut parents = FxHashMap::default();
    for (address, _) in &request.pending_restyles {
        let element = unsafe { ServoLayoutNode::new(address) }.as_element()?;
        let dangerous = unsafe { element.dangerous_style_element() };
        if dangerous.has_dirty_descendants() {
            return None;
        }
        let data = element.style_data()?.element_data.borrow();
        let parent_id = dangerous.traversal_parent()?.as_node().opaque();
        let parent = match parents.entry(parent_id) {
            std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                *entry.insert(BackgroundParentProof::prepare(dangerous)?)
            },
        };
        validate_background_style_before_drain(&context, dangerous, &data, &batch, &parent)?;
    }
    // This Arc is usable only for display/visibility in the same synchronous
    // focus predicate. Public style queries and the pending list are unchanged.
    let focused = unsafe { ServoLayoutNode::new(&request.focused) }.as_element()?;
    let dangerous = unsafe { focused.dangerous_style_element() };
    let data = focused.style_data()?.element_data.borrow();
    if !data.hint.is_empty()
        || !data.damage.is_empty()
        || dangerous.has_dirty_descendants()
        || dangerous.has_animation_only_dirty_descendants()
    {
        return None;
    }
    data.styles.primary.as_ref().cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use style::selector_parser::Snapshot;

    fn intact_style_pending() -> PendingRestyle {
        let mut snapshot = Snapshot::new();
        snapshot
            .changed_attrs
            .push(style::LocalName::new(local_name!("style")));
        PendingRestyle {
            style_attribute_only: true,
            snapshot: Some(snapshot),
            hint: RestyleHint::RESTYLE_STYLE_ATTRIBUTE,
            ..Default::default()
        }
    }

    #[test]
    fn focus_background_intact_history_needs_no_omission_lease_but_rejects_mixed_provenance() {
        let mut pending = intact_style_pending();
        assert!(exact_pending_background_style(&pending, None, Some(7)));
        pending.snapshot.as_mut().unwrap().class_changed = true;
        assert!(!exact_pending_background_style(&pending, None, Some(7)));
        pending.snapshot.as_mut().unwrap().class_changed = false;
        pending.hint.insert(RestyleHint::RESTYLE_SELF);
        assert!(!exact_pending_background_style(&pending, None, Some(7)));
        pending.hint = RestyleHint::RESTYLE_STYLE_ATTRIBUTE;
        pending.style_attribute_only = false;
        assert!(!exact_pending_background_style(&pending, None, Some(7)));
    }

    #[test]
    fn focus_background_omitted_history_requires_exact_live_epoch_and_never_guesses_mixed_debt() {
        let mut pending = PendingRestyle {
            style_attribute_only: true,
            style_snapshot_omitted: Some(7),
            hint: RestyleHint::RESTYLE_STYLE_ATTRIBUTE,
            ..Default::default()
        };
        assert!(exact_pending_background_style(&pending, Some(7), Some(7)));
        assert!(!exact_pending_background_style(&pending, None, Some(7)));
        assert!(!exact_pending_background_style(&pending, Some(7), Some(8)));
        assert!(!exact_pending_background_style(&pending, Some(8), Some(7)));
        pending.snapshot = intact_style_pending().snapshot;
        assert!(!exact_pending_background_style(&pending, Some(7), Some(7)));
        pending.snapshot = None;
        pending.style_snapshot_omitted = None;
        assert!(!exact_pending_background_style(&pending, Some(7), Some(7)));
    }
}
