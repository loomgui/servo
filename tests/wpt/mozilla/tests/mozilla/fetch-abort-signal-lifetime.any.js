// META: title=Completed fetches release their dependent AbortSignal registrations
// META: global=window,worker
// META: timeout=long
// META: script=/common/utils.js

function dependentSignalCount() {
  // Also runs the DOM abort-signal checkpoint so the result does not depend on
  // when the rendering or worker event loop last performed its usual cleanup.
  return ServoTestUtils.dependentAbortSignalCount();
}

async function completeFetches(signal, count) {
  for (let i = 0; i < count; ++i) {
    const response = await fetch('/fetch/api/resources/empty.txt', { signal });
    // EOF, not just response headers, is the cancellation-registration boundary.
    await response.arrayBuffer();
  }
}

promise_test(async t => {
  const controller = new AbortController();
  t.add_cleanup(() => controller.abort());
  await TestUtils.gc();
  const baseline = dependentSignalCount();

  for (const count of [8, 128]) {
    await completeFetches(controller.signal, count);
    await TestUtils.gc();
    assert_equals(dependentSignalCount(), baseline,
                  `completed empty fetch registrations are released after ${count} requests`);
    assert_false(controller.signal.aborted, 'the shared source remains usable');
  }
}, 'Repeated completed fetches with a long-lived source signal have bounded retention');

promise_test(async t => {
  const controller = new AbortController();
  const otherController = new AbortController();
  t.add_cleanup(() => controller.abort());
  t.add_cleanup(() => otherController.abort());

  // Keep an actual JS-visible follower alive across collection. Cleanup must
  // discard dead WeakRefs without disconnecting live signals or any() sources.
  const live = new Request('/fetch/api/resources/empty.txt', { signal: controller.signal });
  let latest;
  for (let batch = 0; batch < 8; ++batch) {
    await completeFetches(controller.signal, 16);
    dependentSignalCount();
    await TestUtils.gc();
    // Use the flattened source path too. This registration prunes only dead
    // entries accumulated on the non-dependent controller signal. The bound
    // allows amortized sweeps instead of scanning all followers on each fetch.
    const intermediate = AbortSignal.any([controller.signal, otherController.signal]);
    latest = AbortSignal.any([intermediate]);
    assert_less_than_equal(ServoTestUtils.abortSignalFollowerCount(controller.signal), 40,
                           'weak follower registry is bounded after collection and registration');
  }
  const reason = new Error('live follower survives cleanup');
  controller.abort(reason);
  assert_true(live.signal.aborted);
  assert_equals(live.signal.reason, reason);
  assert_true(latest.aborted);
  assert_equals(latest.reason, reason);
}, 'Source signals prune dead weak followers while preserving live followers');

promise_test(async t => {
  const controller = new AbortController();
  const otherController = new AbortController();
  const abortKey = token();
  t.add_cleanup(async () => {
    controller.abort();
    otherController.abort();
    await fetch(`/fetch/api/resources/stash-put.py?key=${abortKey}&value=close`);
  });
  await TestUtils.gc();
  const baseline = dependentSignalCount();

  // Each Request creates a dependent signal. AbortSignal.any() and clone()
  // must continue to propagate cancellation while the body is still streaming.
  const signal = AbortSignal.any([controller.signal, otherController.signal]);
  const request = new Request(
    `/fetch/api/resources/infinite-slow-response.py?abortKey=${abortKey}`, { signal });
  const clonedRequest = request.clone();
  const response = await fetch(clonedRequest);
  const reader = response.body.getReader();
  await reader.read();
  await TestUtils.gc();
  assert_equals(dependentSignalCount(), baseline + 1,
                'the active body keeps its cancellation registration after headers and GC');

  await completeFetches(controller.signal, 32);
  await TestUtils.gc();
  assert_equals(dependentSignalCount(), baseline + 1,
                'completed sibling fetches release only their own registrations');

  const reason = new Error('cancel active streamed body');
  controller.abort(reason);
  assert_true(signal.aborted, 'AbortSignal.any() follows the source');
  assert_equals(signal.reason, reason);
  assert_true(request.signal.aborted, 'the original Request signal follows the source');
  assert_true(clonedRequest.signal.aborted, 'the cloned Request signal follows the source');
  assert_equals(clonedRequest.signal.reason, reason);
  await promise_rejects_exactly(t, reason, reader.read());
  await promise_rejects_exactly(t, reason, reader.closed);
  await TestUtils.gc();
  assert_equals(dependentSignalCount(), baseline, 'aborted registrations are released');
}, 'EOF cleanup preserves active body cancellation through cloned Requests and AbortSignal.any()');

promise_test(async t => {
  const controller = new AbortController();
  t.add_cleanup(() => controller.abort());
  await TestUtils.gc();
  const baseline = dependentSignalCount();
  const signal = AbortSignal.any([controller.signal]);
  let calls = 0;
  signal.addEventListener('abort', () => ++calls, { once: true });

  await completeFetches(signal, 32);
  await TestUtils.gc();
  assert_equals(dependentSignalCount(), baseline + 1,
                'fetch cleanup preserves unrelated abort listeners');
  controller.abort();
  assert_equals(calls, 1);
  assert_equals(dependentSignalCount(), baseline);
}, 'Completing fetches does not remove other dependent signal listeners');
