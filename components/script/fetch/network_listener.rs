/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use content_security_policy::Violation;
use js::context::JSContext;
use net_traits::request::RequestId;
use net_traits::{
    BoxedFetchCallback, FetchMetadata, FetchResponseMsg, NetworkError, ResourceFetchTiming,
    ResourceTimingType,
};
use servo_base::id::PipelineId;
use servo_url::ServoUrl;

use crate::dom::bindings::inheritance::Castable;
use crate::dom::bindings::refcounted::Trusted;
use crate::dom::bindings::root::DomRoot;
use crate::dom::globalscope::GlobalScope;
use crate::dom::performance::performanceentry::PerformanceEntry;
use crate::dom::performance::performanceresourcetiming::{
    InitiatorType, PerformanceResourceTiming,
};
use crate::tasks::task_source::SendableTaskSource;

pub(crate) trait ResourceTimingListener {
    fn resource_timing_information(&self) -> (InitiatorType, ServoUrl);
    fn resource_timing_global(&self) -> DomRoot<GlobalScope>;
}

pub(crate) fn submit_timing<T: ResourceTimingListener>(
    cx: &mut JSContext,
    listener: &T,
    result: &Result<(), NetworkError>,
    resource_timing: &ResourceFetchTiming,
) {
    // https://www.w3.org/TR/resource-timing/#resources-included-in-the-performanceresourcetiming-interface
    // If a resource fetch is aborted because it failed a fetch precondition
    // (e.g. mixed content, CORS restriction, CSP policy, etc), then this resource
    // will not be included as a PerformanceResourceTiming object in
    // the Performance Timeline.
    if let Err(error) = &result
        && error.is_permanent_failure()
    {
        return;
    }

    // Resource timings should only be submitted for the initial preload request,
    // not for the request that consumes the preload: https://github.com/whatwg/html/issues/12047
    if resource_timing.preloaded {
        return;
    }
    // TODO Resources for which the fetch was initiated, but was later aborted
    // (e.g. due to a network error) MAY be included as PerformanceResourceTiming
    // objects in the Performance Timeline and MUST contain initialized attribute
    // values for processed substeps of the processing model.
    if resource_timing.timing_type != ResourceTimingType::Resource
        && resource_timing.timing_type != ResourceTimingType::Error
    {
        warn!(
            "Submitting non-resource ({:?}) timing as resource",
            resource_timing.timing_type
        );
        return;
    }

    let (initiator_type, url) = listener.resource_timing_information();
    if initiator_type == InitiatorType::Other {
        warn!("Ignoring InitiatorType::Other resource {:?}", url);
        return;
    }

    submit_timing_data(
        cx,
        &listener.resource_timing_global(),
        url,
        initiator_type,
        resource_timing,
    );
}

pub(crate) fn submit_timing_data(
    cx: &mut JSContext,
    global: &GlobalScope,
    url: ServoUrl,
    initiator_type: InitiatorType,
    resource_timing: &ResourceFetchTiming,
) {
    let performance_entry =
        PerformanceResourceTiming::new(cx, global, url, initiator_type, resource_timing);
    global
        .performance(cx)
        .queue_entry(performance_entry.upcast::<PerformanceEntry>());
}

pub(crate) trait FetchResponseListener: Send + 'static {
    /// A gating mechanism that runs before invoking the listener methods on the target
    /// thread. If the `should_invoke` method returns false, the listener does not receive
    /// the notification.
    fn should_invoke(&self) -> bool {
        true
    }

    fn process_request_body(&mut self, request_id: RequestId);
    fn process_response(
        &mut self,
        cx: &mut JSContext,
        request_id: RequestId,
        metadata: Result<FetchMetadata, NetworkError>,
    );
    fn process_response_chunk(&mut self, cx: &mut JSContext, request_id: RequestId, chunk: Bytes);
    fn process_response_eof(
        self,
        cx: &mut JSContext,
        request_id: RequestId,
        response: Result<(), NetworkError>,
        timing: ResourceFetchTiming,
    );
    fn process_csp_violations(
        &mut self,
        cx: &mut js::context::JSContext,
        request_id: RequestId,
        violations: Vec<Violation>,
    );

    fn process_content_length(&mut self, _request_id: RequestId, _size: usize) {}
}

/// An off-thread sink for async network event tasks. All such events are forwarded to
/// a target thread, where they are invoked on the provided context object.
pub(crate) struct NetworkListener<Listener: FetchResponseListener> {
    pub(crate) context: Arc<Mutex<Option<Listener>>>,
    pub(crate) task_source: SendableTaskSource,
    /// The [`GlobalScope`] this [`NetworkListener`] was created for.
    pub(crate) global_scope: Trusted<GlobalScope>,
    certified_arm_response: Option<CertifiedArmResponse>,
}

impl<Listener: FetchResponseListener> NetworkListener<Listener> {
    pub(crate) fn new(
        context: Listener,
        task_source: SendableTaskSource,
        global_scope: &GlobalScope,
    ) -> Self {
        Self {
            context: Arc::new(Mutex::new(Some(context))),
            task_source,
            global_scope: Trusted::new(global_scope),
            certified_arm_response: None,
        }
    }

    /// Attach only to the ordinary fetch() arm request, before starting its fetch.
    pub(crate) fn observe_certified_arm_response(&mut self, url: &ServoUrl) {
        self.certified_arm_response = CertifiedArmResponse::from_url(url);
    }

    pub(crate) fn notify(&mut self, message: FetchResponseMsg) {
        let context = self.context.clone();
        let global_scope = self.global_scope.clone();
        let arm_task = self.certified_arm_response.and_then(|arm| {
            if let FetchResponseMsg::ProcessResponse(request_id, ..) = &message {
                CertifiedArmResponseTask::capture(arm, *request_id, self.task_source.pipeline_id)
            } else {
                None
            }
        });
        self.task_source
            .queue(task!(network_listener_response: move |cx| {
                if let Some(arm_task) = arm_task {
                    arm_task.started();
                }
                if let FetchResponseMsg::ProcessResponseEOF(request_id, ..) = &message {
                    global_scope
                        .root()
                        .fetch_group_mut()
                        .mark_fetch_request_as_done(request_id);
                }

                let mut context = context.lock().unwrap();
                let Some(fetch_listener) = &mut *context else {
                    return;
                };

                if !fetch_listener.should_invoke() {
                    return;
                }

                match message {
                    FetchResponseMsg::ProcessRequestBody(request_id) => {
                        fetch_listener.process_request_body(request_id)
                    },
                    FetchResponseMsg::ProcessResponse(request_id, meta) => {
                        fetch_listener.process_response(cx, request_id, meta)
                    },
                    FetchResponseMsg::ProcessResponseChunk(request_id, data) => {
                        fetch_listener.process_response_chunk(cx, request_id, data)
                    },
                    FetchResponseMsg::ProcessResponseEOF(request_id, result, timing) => {
                        if let Some(fetch_listener) = context.take() {
                            fetch_listener.process_response_eof(cx, request_id, result, timing);
                        };
                    },
                    FetchResponseMsg::ProcessCspViolations(request_id, violations) => {
                        fetch_listener.process_csp_violations(cx, request_id, violations)
                    },
                    FetchResponseMsg::ProcessContentLength(request_id, size) => { fetch_listener.process_content_length(request_id, size) },
                }
            }));
        // queue() may cancel/drop the task. This records an attempt, never delivery
        // or promise completion. Logging is after queue() and outside its guards.
        if let Some(arm_task) = arm_task {
            arm_task.enqueue_attempted();
        }
    }

    pub(crate) fn into_callback(mut self) -> BoxedFetchCallback {
        Box::new(move |response_msg| self.notify(response_msg))
    }
}

// Disposable scheduling attribution, never a frame/readiness/validity channel.
// Selected arm response tasks only: first64 attempts/process, at most128 rows.
// START may precede ENQUEUE emission on another thread; use captured UTC fields.
const CERTIFIED_ARM_TASK_LIMIT: u64 = 64;

pub(crate) fn certified_layout_probe_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        certified_layout_probe_gate(
            std::env::var("LOOM_CERTIFIED_LAYOUT_PROBE")
                .as_deref()
                .unwrap_or(""),
            std::env::var("LOOM_PACING_ORACLE").as_deref().unwrap_or(""),
            std::env::var("LOOM_PACING_PROFILE")
                .as_deref()
                .unwrap_or(""),
        )
    })
}

fn certified_layout_probe_gate(probe: &str, oracle: &str, profile: &str) -> bool {
    probe == "1" && oracle == "1" && profile == "app"
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct CertifiedArmResponse {
    frame: u64,
    sequence: u64,
}

impl CertifiedArmResponse {
    fn from_url(url: &ServoUrl) -> Option<Self> {
        let url = url.as_url();
        if url.scheme() != "loom"
            || url.host_str() != Some("frame-arm")
            || url.fragment().is_some()
            || url.port().is_some()
            || !url.username().is_empty()
            || url.password().is_some()
        {
            return None;
        }
        let frame = url.path().strip_prefix('/')?.parse().ok()?;
        let mut query = url.query_pairs();
        let (key, value) = query.next()?;
        if key != "s" || query.next().is_some() || frame == 0 {
            return None;
        }
        Some(Self {
            frame,
            sequence: value.parse().ok()?,
        })
    }
}

fn certified_arm_task_ordinal(counter: &AtomicU64) -> Option<u64> {
    let ordinal = counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .ok()?
        + 1;
    (ordinal <= CERTIFIED_ARM_TASK_LIMIT).then_some(ordinal)
}

#[derive(Clone, Copy)]
struct CertifiedArmResponseTask {
    arm: CertifiedArmResponse,
    request: RequestId,
    pipeline: PipelineId,
    ordinal: u64,
    enqueue_utc_us: u128,
    enqueue_clock: Instant,
}

impl CertifiedArmResponseTask {
    fn capture(
        arm: CertifiedArmResponse,
        request: RequestId,
        pipeline: PipelineId,
    ) -> Option<Self> {
        static OBSERVED: AtomicU64 = AtomicU64::new(0);
        let ordinal = certified_arm_task_ordinal(&OBSERVED)?;
        Some(Self {
            arm,
            request,
            pipeline,
            ordinal,
            enqueue_utc_us: certified_arm_utc_us(),
            enqueue_clock: Instant::now(),
        })
    }

    fn enqueue_attempted(self) {
        info!(
            "[CERTIFIED-LAYOUT-NET] phase=enqueue_attempt pipeline={:?} frame={} sequence={} request={:?} ordinal={} enqueue_utc_us={} emit_utc_us={} thread={:?}",
            self.pipeline,
            self.arm.frame,
            self.arm.sequence,
            self.request,
            self.ordinal,
            self.enqueue_utc_us,
            certified_arm_utc_us(),
            std::thread::current().id()
        );
    }

    fn started(self) {
        let start_utc_us = certified_arm_utc_us();
        let queue_to_start_ms = self.enqueue_clock.elapsed().as_secs_f64() * 1000.0;
        info!(
            "[CERTIFIED-LAYOUT-NET] phase=start pipeline={:?} frame={} sequence={} request={:?} ordinal={} enqueue_utc_us={} start_utc_us={} queue_to_start_ms={:.6} thread={:?}",
            self.pipeline,
            self.arm.frame,
            self.arm.sequence,
            self.request,
            self.ordinal,
            self.enqueue_utc_us,
            start_utc_us,
            queue_to_start_ms,
            std::thread::current().id()
        );
    }
}

fn certified_arm_utc_us() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|v| v.as_micros())
        .unwrap_or(0)
}

#[cfg(test)]
mod certified_arm_response_probe_tests {
    use super::*;

    #[test]
    fn exact_gate_and_arm_url_join_only() {
        assert!(certified_layout_probe_gate("1", "1", "app"));
        for args in [
            ("", "1", "app"),
            ("1", "", "app"),
            ("1", "1", "oracle-retained-backgrounds"),
        ] {
            assert!(!certified_layout_probe_gate(args.0, args.1, args.2));
        }
        assert_eq!(
            CertifiedArmResponse::from_url(&ServoUrl::parse("loom://frame-arm/2662?s=7").unwrap()),
            Some(CertifiedArmResponse {
                frame: 2662,
                sequence: 7
            })
        );
        for url in [
            "http://frame-arm/2662?s=7",
            "loom://frame-ack/2662?s=7",
            "loom://frame-arm/0?s=7",
            "loom://frame-arm/2662",
            "loom://frame-arm/2662?s=7&s=8",
            "loom://frame-arm/2662?s=7#other",
            "loom://frame-arm/x?s=7",
        ] {
            assert_eq!(
                CertifiedArmResponse::from_url(&ServoUrl::parse(url).unwrap()),
                None,
                "{url}"
            );
        }
    }

    #[test]
    fn actual_task_selection_is_bounded_and_does_not_wrap() {
        let counter = AtomicU64::new(0);
        for ordinal in 1..=CERTIFIED_ARM_TASK_LIMIT {
            assert_eq!(certified_arm_task_ordinal(&counter), Some(ordinal));
        }
        assert_eq!(certified_arm_task_ordinal(&counter), None);
        counter.store(u64::MAX, Ordering::Relaxed);
        assert_eq!(certified_arm_task_ordinal(&counter), None);
    }
}
