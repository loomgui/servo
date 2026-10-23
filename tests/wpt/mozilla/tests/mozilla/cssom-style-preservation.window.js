// META: title=Inline style preservation follows consumers of the old attribute value

function preservationCloneCount() {
  return ServoTestUtils.inlineStylePreservationCloneCount();
}

function styleTarget(t, parent = document.body) {
  const target = document.createElement('div');
  target.style.cssText = 'position: absolute; left: 1px; top: 2px; width: 10px; height: 20px; background-color: red';
  parent.append(target);
  t.add_cleanup(() => target.remove());
  return target;
}

function observe(t, target, options) {
  const observer = new MutationObserver(() => {});
  observer.observe(target, options);
  t.add_cleanup(() => observer.disconnect());
  return observer;
}

test(t => {
  // Construction alone makes the window's observer flag sticky.
  const observer = new MutationObserver(() => {});
  t.add_cleanup(() => observer.disconnect());
  const target = styleTarget(t);
  const before = preservationCloneCount();
  target.style.backgroundColor = 'green';
  target.style.backgroundColor = 'blue';
  assert_equals(preservationCloneCount() - before, 0);
  assert_equals(target.style.backgroundColor, 'blue');
}, 'An unregistered MutationObserver does not require declaration-block preservation');

test(t => {
  const observer = observe(t, document, {
    attributes: true, subtree: true, attributeOldValue: true,
    attributeFilter: ['data-loom-image-kind'],
  });
  const target = styleTarget(t);
  const before = preservationCloneCount();
  target.style.backgroundColor = 'green';
  target.style.backgroundColor = 'blue';
  assert_equals(preservationCloneCount() - before, 0);
  assert_array_equals(observer.takeRecords(), []);
  assert_equals(target.style.width, '10px');
  assert_equals(target.style.height, '20px');
  assert_equals(target.getAttribute('style'), target.style.cssText);
}, 'An unrelated attribute filter avoids full declaration-block copies');

test(t => {
  const target = styleTarget(t);
  const observer = observe(t, target, { attributes: true, attributeFilter: ['style'] });
  const before = preservationCloneCount();
  target.style.backgroundColor = 'green';
  target.style.backgroundColor = 'blue';
  assert_equals(preservationCloneCount() - before, 0);
  const records = observer.takeRecords();
  assert_equals(records.length, 2);
  assert_array_equals(records.map(record => record.oldValue), [null, null]);
  assert_true(records.every(record => record.attributeName === 'style' && record.target === target));
}, 'Style observers without attributeOldValue still receive records without preservation');

test(t => {
  const target = styleTarget(t);
  const observer = observe(t, target, { attributes: true, attributeOldValue: true });
  // cssText serializes the declaration block without filling the Attr's lazy
  // serialization cache, so old-value correctness cannot rely on a cached Attr.
  const initial = target.style.cssText;
  const before = preservationCloneCount();
  target.style.backgroundColor = 'green';
  const middle = target.style.cssText;
  target.style.backgroundColor = 'blue';
  assert_equals(preservationCloneCount() - before, 2);
  assert_array_equals(observer.takeRecords().map(record => record.oldValue), [initial, middle]);
}, 'Matching old-value observers retain each successive old declaration block');

test(t => {
  const parent = document.createElement('section');
  document.body.append(parent);
  t.add_cleanup(() => parent.remove());
  const target = styleTarget(t, parent);
  const ancestor = observe(t, parent, { attributes: true, attributeOldValue: true });
  const before = preservationCloneCount();
  target.style.backgroundColor = 'green';
  assert_equals(preservationCloneCount() - before, 0);
  assert_array_equals(ancestor.takeRecords(), []);

  ancestor.observe(parent, { attributes: true, attributeOldValue: true, subtree: true });
  const second = observe(t, target, { attributes: true, attributeOldValue: true });
  const withoutOldValue = observe(t, target, { attributes: true });
  const old = target.style.cssText;
  target.style.setProperty('background-color', 'blue', 'important');
  assert_equals(preservationCloneCount() - before, 1, 'one copy serves multiple matching observers');
  assert_equals(ancestor.takeRecords()[0].oldValue, old);
  assert_equals(second.takeRecords()[0].oldValue, old);
  assert_equals(withoutOldValue.takeRecords()[0].oldValue, null);
  assert_equals(target.style.getPropertyPriority('background-color'), 'important');
}, 'Ancestor subtree filtering and multiple observer old-value requests agree');

test(t => {
  const target = styleTarget(t);
  const observer = observe(t, target, {
    attributes: true, attributeOldValue: true, attributeFilter: ['STYLE'],
  });
  const before = preservationCloneCount();
  target.style.backgroundColor = 'green';
  assert_equals(preservationCloneCount() - before, 0);
  assert_array_equals(observer.takeRecords(), []);
}, 'Attribute-filter names retain their case-sensitive matching behavior');

test(t => {
  const target = styleTarget(t);
  const observer = observe(t, target, {
    attributes: true, attributeOldValue: true, attributeFilter: ['data-kind'],
  });
  target.setAttributeNS('urn:preservation', 'p:data-kind', 'one');
  target.setAttributeNS('urn:preservation', 'p:data-kind', 'two');
  assert_array_equals(observer.takeRecords(), [], 'a filter excludes namespaced attributes');
  observer.observe(target, { attributes: true, attributeOldValue: true });
  target.setAttributeNS('urn:preservation', 'p:data-kind', 'three');
  const record = observer.takeRecords()[0];
  assert_equals(record.attributeName, 'data-kind');
  assert_equals(record.attributeNamespace, 'urn:preservation');
  assert_equals(record.oldValue, 'two');
}, 'Shared observer matching preserves namespace filtering and unfiltered old values');

test(t => {
  const documentObserver = observe(t, document, {
    attributes: true, subtree: true, attributeOldValue: true, attributeFilter: ['style'],
  });
  const host = document.createElement('div');
  document.body.append(host);
  t.add_cleanup(() => host.remove());
  const shadow = host.attachShadow({ mode: 'open' });
  const target = styleTarget(t, shadow);
  const before = preservationCloneCount();
  target.style.backgroundColor = 'green';
  assert_equals(preservationCloneCount() - before, 0);
  assert_array_equals(documentObserver.takeRecords(), []);

  const shadowObserver = observe(t, shadow, { attributes: true, subtree: true, attributeOldValue: true });
  const old = target.style.cssText;
  target.style.backgroundColor = 'blue';
  assert_equals(preservationCloneCount() - before, 1);
  assert_equals(shadowObserver.takeRecords()[0].oldValue, old);
}, 'Observer preservation respects shadow boundaries and observers inside shadow roots');

test(t => {
  const root = document.createElement('section');
  const target = styleTarget(t, root);
  const observer = observe(t, root, { attributes: true, subtree: true, attributeOldValue: true });
  const old = target.style.cssText;
  const before = preservationCloneCount();
  target.style.backgroundColor = 'green';
  assert_equals(preservationCloneCount() - before, 1);
  assert_equals(observer.takeRecords()[0].oldValue, old);
}, 'Old-value observers registered on a detached subtree retain their values');

test(t => {
  observe(t, document, { childList: true, subtree: true });
  const target = styleTarget(t);
  target.remove();
  const before = preservationCloneCount();
  target.style.backgroundColor = 'green';
  assert_equals(preservationCloneCount() - before, 1,
                'detached-node fallback remains conservative for pending transient observers');
}, 'Detached nodes keep conservative preservation across subtree removal');

promise_test(async t => {
  const calls = [];
  customElements.define('preserved-style-observed', class extends HTMLElement {
    static observedAttributes = ['style'];
    attributeChangedCallback(name, oldValue, value) { calls.push([name, oldValue, value]); }
  });
  const target = document.createElement('preserved-style-observed');
  document.body.append(target);
  t.add_cleanup(() => target.remove());
  target.style.cssText = 'background-color: red; width: 10px';
  await Promise.resolve();
  calls.length = 0;
  const initial = target.style.cssText;
  const before = preservationCloneCount();
  target.style.backgroundColor = 'green';
  const middle = target.style.cssText;
  target.style.backgroundColor = 'blue';
  const last = target.style.cssText;
  await Promise.resolve();
  assert_equals(preservationCloneCount() - before, 2);
  assert_array_equals(calls.map(call => call[0]), ['style', 'style']);
  assert_array_equals(calls.map(call => call[1]), [initial, middle]);
  assert_array_equals(calls.map(call => call[2]), [middle, last]);
}, 'Custom elements observing style retain old/new values and callback order');

promise_test(async t => {
  const calls = [];
  customElements.define('preserved-style-unobserved', class extends HTMLElement {
    static observedAttributes = ['data-other'];
    attributeChangedCallback(name, oldValue, value) { calls.push([name, oldValue, value]); }
  });
  const target = document.createElement('preserved-style-unobserved');
  document.body.append(target);
  t.add_cleanup(() => target.remove());
  target.style.cssText = 'background-color: red; width: 10px';
  await Promise.resolve();
  const before = preservationCloneCount();
  target.style.backgroundColor = 'green';
  target.style.backgroundColor = 'blue';
  target.setAttribute('data-other', 'current');
  await Promise.resolve();
  assert_equals(preservationCloneCount() - before, 0);
  assert_equals(calls.length, 1);
  assert_array_equals(calls[0], ['data-other', null, 'current']);
}, 'Custom elements with an unrelated observed attribute avoid style preservation');

test(t => {
  const target = styleTarget(t);
  observe(t, target, { attributes: true, attributeFilter: ['data-other'] });
  const before = preservationCloneCount();
  target.style.backgroundColor = 'green';
  assert_equals(preservationCloneCount() - before, 0);
  const observer = observe(t, target, { attributes: true, attributeOldValue: true });
  const old = target.style.cssText;
  target.style.backgroundColor = 'blue';
  assert_equals(preservationCloneCount() - before, 1);
  assert_equals(observer.takeRecords()[0].oldValue, old);
}, 'Registering an old-value observer after a no-copy mutation sees the current old value');

test(t => {
  observe(t, document, { attributes: true, subtree: true, attributeFilter: ['data-other'] });
  const target = styleTarget(t);
  const clone = target.cloneNode(false);
  document.body.append(clone);
  t.add_cleanup(() => clone.remove());
  const cloneInitial = clone.style.cssText;
  const attr = target.getAttributeNode('style');
  const before = preservationCloneCount();
  target.style.backgroundColor = 'blue';
  assert_equals(clone.style.cssText, cloneInitial, 'cloned elements keep independent declarations');
  clone.style.backgroundColor = 'green';
  assert_equals(target.style.backgroundColor, 'blue');
  assert_equals(attr, target.getAttributeNode('style'), 'the exposed Attr keeps its identity');
  assert_equals(attr.value, target.style.cssText, 'the exposed Attr sees the latest serialization');
  assert_equals(preservationCloneCount() - before, 0);
}, 'No-copy CSSOM updates preserve cloned element independence and exposed Attr values');

test(t => {
  const sheet = document.createElement('style');
  sheet.textContent = `
    .preserve-scope .preserve-target[style*="--paint-token: 1;"] + .preserve-peer { color: rgb(255, 0, 0); }
    .preserve-scope .preserve-peer { color: rgb(0, 0, 0); }
  `;
  document.head.append(sheet);
  t.add_cleanup(() => sheet.remove());
  const scope = document.createElement('section');
  scope.className = 'preserve-scope';
  document.body.append(scope);
  t.add_cleanup(() => scope.remove());
  const target = styleTarget(t, scope);
  target.className = 'preserve-target';
  const peer = document.createElement('div');
  peer.className = 'preserve-peer';
  scope.append(peer);
  observe(t, target, { attributes: true });
  target.style.backgroundColor = 'var(--shade)';
  target.style.setProperty('--shade', 'rgb(0, 0, 255)');
  assert_equals(getComputedStyle(target).backgroundColor, 'rgb(0, 0, 255)');
  assert_equals(getComputedStyle(peer).color, 'rgb(0, 0, 0)');
  const before = preservationCloneCount();
  target.style.setProperty('--shade', 'rgb(0, 255, 0)');
  target.style.setProperty('--paint-token', '1');
  assert_equals(getComputedStyle(target).backgroundColor, 'rgb(0, 255, 0)');
  assert_equals(getComputedStyle(peer).color, 'rgb(255, 0, 0)');
  target.style.setProperty('--paint-token', '2');
  assert_equals(getComputedStyle(peer).color, 'rgb(0, 0, 0)');
  assert_equals(preservationCloneCount() - before, 0);
  assert_equals(target.getAttribute('style'), target.style.cssText);
}, 'No-copy CSSOM mutations preserve custom properties, attribute selectors, and serialization');
