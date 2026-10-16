/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::cell::Cell;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use html5ever::LocalName;

use js::context::JSContext;
use script_bindings::callback::ExceptionHandling;
use script_bindings::cell::DomRefCell;
use script_bindings::inheritance::Castable;
use script_bindings::root::{Dom, DomRoot};

use crate::dom::bindings::str::DOMString;
use crate::dom::types::{EventTarget, HTMLSlotElement, MutationObserver, MutationRecord};
use crate::runtime::job_queue::NotifyMutationObserversMicrotask;

/// A monotonic, agent-wide negative proof, never an observer routing table.
/// Collisions, saturation, re-observation, and disconnect can only retain false
/// positives. Empty filters deliberately mirror ObserverOptions' current
/// unfiltered matching, including an explicitly supplied empty filter.
#[derive(JSTraceable, Default)]
struct AttributeObserverInterest {
    bits: Cell<[u64; 4]>,
    unfiltered_attributes: Cell<bool>,
}

impl AttributeObserverInterest {
    fn positions(name: &str) -> [usize; 2] {
        // The matching path compares case-sensitive attribute-name strings.
        // Hash that exact spelling; namespaces are deliberately not excluded.
        let mut hash = DefaultHasher::new();
        name.hash(&mut hash);
        let hash = hash.finish();
        [(hash & 255) as usize, ((hash >> 32) & 255) as usize]
    }

    fn record<'a>(&self, attributes: bool, filter: impl IntoIterator<Item = &'a str>) {
        if !attributes {
            return;
        }
        let mut filtered = false;
        let mut bits = self.bits.get();
        for name in filter {
            filtered = true;
            for bit in Self::positions(name) {
                bits[bit / 64] |= 1u64 << (bit % 64);
            }
        }
        self.bits.set(bits);
        if !filtered {
            self.unfiltered_attributes.set(true);
        }
    }

    fn cannot_match(&self, name: &str) -> bool {
        if self.unfiltered_attributes.get() {
            return false;
        }
        let bits = self.bits.get();
        Self::positions(name)
            .into_iter()
            .any(|bit| bits[bit / 64] & (1u64 << (bit % 64)) == 0)
    }
}

/// A helper struct for mutation observers used in `ScriptThread`
/// Since the Rc is always stored in ScriptThread, it's always reachable by the GC.
#[derive(JSTraceable, Default)]
#[cfg_attr(crown, crown::unrooted_must_root_lint::allow_unrooted_in_rc)]
#[cfg_attr(crown, crown::unrooted_must_root_lint::must_root)]
pub(crate) struct ScriptMutationObservers {
    /// Union of every valid attribute registration ever published in this agent.
    /// Never reset when pending notifications are drained or observers disconnect.
    attribute_interest: AttributeObserverInterest,

    /// Microtask Queue for adding support for mutation observer microtasks
    mutation_observer_microtask_queued: Cell<bool>,

    /// The unit of related similar-origin browsing contexts' list of MutationObserver objects
    mutation_observers: DomRefCell<Vec<Dom<MutationObserver>>>,

    /// <https://dom.spec.whatwg.org/#signal-slot-list>
    signal_slots: DomRefCell<Vec<Dom<HTMLSlotElement>>>,
}

impl ScriptMutationObservers {
    pub(crate) fn record_attribute_observer(&self, attributes: bool, filter: &[DOMString]) {
        let filter_names: Vec<_> = filter.iter().map(DOMString::str).collect();
        self.attribute_interest
            .record(attributes, filter_names.iter().map(|name| &**name));
    }

    pub(crate) fn attribute_observers_cannot_match(&self, name: &LocalName) -> bool {
        self.attribute_interest.cannot_match(name.as_ref())
    }

    pub(crate) fn add_mutation_observer(&self, observer: &MutationObserver) {
        self.mutation_observers
            .borrow_mut()
            .push(Dom::from_ref(observer));
    }

    /// <https://dom.spec.whatwg.org/#notify-mutation-observers>
    pub(crate) fn notify_mutation_observers(&self, cx: &mut JSContext) {
        // Step 1. Set the surrounding agent’s mutation observer microtask queued to false.
        self.mutation_observer_microtask_queued.set(false);

        // Step 2. Let notifySet be a clone of the surrounding agent’s pending mutation observers.
        // Step 3. Empty the surrounding agent’s pending mutation observers.
        let notify_list = self.take_mutation_observers();

        // Step 4. Let signalSet be a clone of the surrounding agent’s signal slots.
        // Step 5. Empty the surrounding agent’s signal slots.
        let signal_set: Vec<DomRoot<HTMLSlotElement>> = self.take_signal_slots();

        // Step 6. For each mo of notifySet:
        for mo in notify_list.iter() {
            let record_queue = mo.record_queue();

            // Step 6.1 Let records be a clone of mo’s record queue.
            let queue: Vec<DomRoot<MutationRecord>> = record_queue
                .borrow()
                .iter()
                .map(|record| record.as_rooted())
                .collect();

            // Step 6.2 Empty mo’s record queue.
            record_queue.borrow_mut().clear();

            // TODO Step 6.3 For each node of mo’s node list, remove all transient registered observers
            // whose observer is mo from node’s registered observer list.

            // Step 6.4 If records is not empty, then invoke mo’s callback with « records,
            // mo » and "report", and with callback this value mo.
            if !queue.is_empty() {
                let _ = mo
                    .callback()
                    .Call_(cx, &**mo, queue, mo, ExceptionHandling::Report);
            }
        }

        // Step 6. For each slot of signalSet, fire an event named slotchange,
        // with its bubbles attribute set to true, at slot.
        for slot in signal_set {
            slot.upcast::<EventTarget>()
                .fire_bubbling_event(cx, atom!("slotchange"));
        }
    }

    /// <https://dom.spec.whatwg.org/#queue-a-mutation-observer-compound-microtask>
    pub(crate) fn queue_mutation_observer_microtask(&self, cx: &JSContext) {
        // Step 1. If the surrounding agent’s mutation observer microtask queued is true, then return.
        if self.mutation_observer_microtask_queued.get() {
            return;
        }

        // Step 2. Set the surrounding agent’s mutation observer microtask queued to true.
        self.mutation_observer_microtask_queued.set(true);

        // Step 3. Queue a microtask to notify mutation observers.
        crate::runtime::job_queue::enqueue(cx, Box::new(NotifyMutationObserversMicrotask::new()));
    }

    pub(crate) fn add_signal_slot(&self, observer: &HTMLSlotElement) {
        self.signal_slots.borrow_mut().push(Dom::from_ref(observer));
    }

    pub(crate) fn take_signal_slots(&self) -> Vec<DomRoot<HTMLSlotElement>> {
        self.signal_slots
            .take()
            .into_iter()
            .inspect(|slot| {
                slot.remove_from_signal_slots();
            })
            .map(|slot| slot.as_rooted())
            .collect()
    }

    pub(crate) fn take_mutation_observers(&self) -> Vec<DomRoot<MutationObserver>> {
        self.mutation_observers
            .take()
            .iter()
            .map(|mo| mo.as_rooted())
            .collect()
    }
}

#[cfg(test)]
mod attribute_interest_tests {
    use std::collections::HashMap;

    use super::AttributeObserverInterest;

    #[test]
    fn filtered_names_use_exact_case_sensitive_spelling() {
        let interest = AttributeObserverInterest::default();
        let name = String::from("style");
        assert!(interest.cannot_match(&name));
        interest.record(true, [name.as_str()]);
        assert!(!interest.cannot_match(&String::from("style")));
        assert_eq!(
            AttributeObserverInterest::positions(&name),
            AttributeObserverInterest::positions("style")
        );
        // Atom/filter matching uses exact spellings. Bloom collisions between
        // distinct spellings are allowed and must never become a negative.
        assert_ne!(name, "STYLE");
        interest.record(true, ["STYLE"]);
        assert!(!interest.cannot_match("STYLE"));
    }

    #[test]
    fn non_attribute_options_do_not_disable_the_negative_proof() {
        let interest = AttributeObserverInterest::default();
        interest.record(false, std::iter::empty());
        interest.record(false, ["style"]);
        assert!(interest.cannot_match("style"));
    }

    #[test]
    fn empty_filter_is_conservatively_unfiltered_and_sticky() {
        let interest = AttributeObserverInterest::default();
        interest.record(true, std::iter::empty());
        interest.record(false, ["style"]);
        interest.record(true, ["data-loom-image-kind"]);
        assert!(!interest.cannot_match("style"));
        assert!(!interest.cannot_match("arbitrary-later-attribute"));
    }

    #[test]
    fn replacement_and_disconnect_cannot_remove_prior_interest() {
        let interest = AttributeObserverInterest::default();
        interest.record(true, ["first"]);
        let before = interest.bits.get();
        interest.record(true, ["second"]);
        interest.record(false, std::iter::empty());
        assert!(
            before
                .into_iter()
                .zip(interest.bits.get())
                .all(|(before, after)| before & after == before)
        );
        assert!(!interest.cannot_match("first"));
        assert!(!interest.cannot_match("second"));
        // There is intentionally no disconnect/notification reset operation.
    }

    #[test]
    fn actual_hash_collision_only_causes_a_false_positive() {
        let mut seen = HashMap::new();
        let mut collision = None;
        // Two 8-bit positions have at most 65536 ordered combinations.
        for index in 0..=65536 {
            let name = format!("attribute-{index}");
            let bits = AttributeObserverInterest::positions(&name);
            if let Some(previous) = seen.insert(bits, name.clone()) {
                collision = Some((previous, name));
                break;
            }
        }
        let (registered, other) = collision.expect("bounded pigeonhole collision");
        assert_ne!(registered, other);
        let interest = AttributeObserverInterest::default();
        interest.record(true, [registered.as_str()]);
        assert!(!interest.cannot_match(&registered));
        assert!(!interest.cannot_match(&other));
    }

    #[test]
    fn saturated_filter_always_falls_back_to_exact_matching() {
        let interest = AttributeObserverInterest::default();
        interest.bits.set([u64::MAX; 4]);
        assert!(!interest.cannot_match("style"));
        assert!(!interest.cannot_match("different-namespace-name"));
    }
}
