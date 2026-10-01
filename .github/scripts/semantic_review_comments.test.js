// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const test = require('node:test');
const assert = require('node:assert/strict');
const {NAME, notice, requests, reviewState, publish} = require('./semantic_review');

const repo = {owner: 'NVIDIA', repo: 'TensorRT-LLM'};
const service = {login: 'trtllm-agent', id: 296075020, type: 'User'};
const reviewer = {login: 'coderabbitai[bot]', id: 136622811, type: 'Bot'};
const publisher = {login: 'github-actions[bot]', id: 41898282, type: 'Bot'};
const human = {login: 'reviewer', id: 1, type: 'User'};
const marker = '<!-- semantic-review-summary:v1 NVIDIA/TensorRT-LLM#1 -->';
const pending = '<!-- semantic-review-cleanup-pending -->';
const request = (id = 1) => ({id: `00000000-0000-0000-0000-${String(id).padStart(12, '0')}`,
  head: 'a'.repeat(40), target: 'b'.repeat(40), mergeBase: 'c'.repeat(40), branch: 'main'});
const comment = (id, body, user = reviewer) => ({id, node_id: `IC_${id}`, body, user,
  created_at: new Date(Date.UTC(2026, 8, 30) + id * 1000).toISOString()});
const trigger = (id, value = request(), user = service) =>
  comment(id, `@coderabbitai\n<!-- semantic-review-request:${JSON.stringify(value)} -->`, user);
const reply = (id, value = request(), verdict = 'PASS', evidence = true, user = reviewer) =>
  comment(id, `## SEMANTIC_REVIEW\n${notice}\n` +
    `SEMANTIC_RESULT request_id=${value.id} head=${value.head} target=${value.target} ` +
    `merge_base=${value.mergeBase} verdict=${verdict}\n` +
    (evidence ? [value.head, value.target].map(sha =>
      `https://github.com/NVIDIA/TensorRT-LLM/blob/${sha}/file.cpp#L1`).join('\n') : ''), user);

function fixture(comments) {
  const state = {comments, statuses: [], checks: [], hidden: new Set(), mutations: [],
    queries: [], creates: [], patches: [], warnings: [], denied: new Set(), missing: new Set()};
  const github = {rest: {issues: {
    listComments: async () => ({data: state.comments}),
    createComment: async args => {
      assert.equal(args.request.retries, 0);
      state.creates.push(args);
      if (state.createError === 'before') throw Object.assign(new Error('private remote detail'), {status: 503});
      const value = comment(10000 + state.creates.length, args.body, publisher);
      state.comments.push(value);
      if (state.createError === 'after') throw Object.assign(new Error('private remote detail'), {status: 503});
      return {data: value};
    },
    updateComment: async args => {
      assert.equal(args.request.retries, 0);
      state.patches.push(args);
      if (state.patchError) throw Object.assign(new Error('private remote detail'), {status: 403});
      const value = state.comments.find(item => item.id === args.comment_id);
      value.body = args.body;
      return {data: value};
    },
  }, checks: {
    listForRef: async () => ({data: state.checks}),
    update: async () => { if (state.cleanupError) throw new Error('cleanup failed'); },
  }, repos: {
    listCommitStatusesForRef: async () => ({data: state.statuses}),
    createCommitStatus: async args => {
      if (state.statusError) throw new Error('status failed');
      state.statuses.push({...args, id: state.statuses.length + 1, creator: publisher});
    },
  }}, paginate: async (method, args) => (await method(args)).data,
  graphql: async (query, args) => {
    if (args.ids) {
      state.queries.push(args.ids);
      assert.ok(args.ids.length <= 100);
      if (state.queryError) throw Object.assign(new Error('private remote detail'), {status: 403});
      return {nodes: args.ids.map(id => state.missing.has(id) ? null : {id,
        isMinimized: state.hidden.has(id), viewerCanMinimize: !state.denied.has(id),
        viewerCanUnminimize: state.hidden.has(id) && !state.denied.has(id)})};
    }
    assert.equal(args.request.retries, 0);
    const action = query.includes('unminimizeComment') ? 'show' : 'hide';
    const classifier = query.match(/classifier: (\w+)/)?.[1];
    state.mutations.push({id: args.id, action, classifier});
    if (state.mutationError) throw Object.assign(new Error('private remote detail'), {status: 403});
    if (action === 'show') state.hidden.delete(args.id);
    else state.hidden.add(args.id);
    if (state.unconfirmedMutation) return {};
    return action === 'show' ? {unminimizeComment: {unminimizedComment: {isMinimized: false}}} :
      {minimizeComment: {minimizedComment: {isMinimized: true}}};
  }};
  const core = {info() {}, warning(value) { state.warnings.push(value); },
    summary: {addRaw() { return this; }, async write() {}}};
  return {state,
    inspect: () => reviewState({github, repo, number: 1}),
    publish: () => publish({github, context: {repo}, core, number: 1}),
    summary: () => state.comments.find(item => item.user === publisher && item.body.startsWith(marker)),
  };
}

test('waiting summary records fixed refs and notice, restores current trigger, then becomes a no-op', async () => {
  const value = request();
  value.branch = 'release/test@reviewer<&';
  const f = fixture([trigger(1, value)]);
  f.state.hidden.add('IC_1');
  assert.equal((await f.inspect()).commentUpdate, true);
  await f.publish();
  assert.equal(f.state.statuses.at(-1).state, 'pending');
  assert.match(f.summary().body, /\*\*WAITING: Waiting for CodeRabbit response\*\*/);
  assert.ok(f.summary().body.includes(notice));
  assert.ok(f.summary().body.includes('Branch: <code>release/test&#64;reviewer&lt;&amp;</code>'));
  assert.ok(!f.summary().body.includes('@'));
  for (const sha of [value.head, value.target, value.mergeBase]) {
    assert.ok(f.summary().body.includes(`/commit/${sha}`));
  }
  assert.ok(f.summary().body.includes('/pull/1#issuecomment-1'));
  assert.ok(!f.summary().body.includes(pending));
  assert.deepEqual(f.state.mutations, [{id: 'IC_1', action: 'show', classifier: undefined}]);
  assert.equal((await f.inspect()).commentUpdate, false);
  const calls = [f.state.creates.length, f.state.patches.length, f.state.queries.length, f.state.mutations.length];
  await f.publish();
  assert.deepEqual([f.state.creates.length, f.state.patches.length, f.state.queries.length,
    f.state.mutations.length], calls);
});

for (const [verdict, evidence, expected] of [
  ['PASS', true, 'PASS'], ['FAIL', true, 'FAIL'],
  ['INCONCLUSIVE', false, 'INCONCLUSIVE'], ['PASS', false, 'INCONCLUSIVE'],
]) {
  test(`${verdict} with evidence=${evidence} completes trigger and keeps accepted ${expected} reply visible`, async () => {
    const f = fixture([trigger(1), reply(2, request(), verdict, evidence)]);
    f.state.hidden.add('IC_2');
    await f.publish();
    assert.equal(f.state.statuses.at(-1).state,
      {PASS: 'success', FAIL: 'failure', INCONCLUSIVE: 'pending'}[expected]);
    assert.ok(f.summary().body.includes(`**${expected}:`));
    assert.ok(f.summary().body.includes('[CodeRabbit analysis](https://github.com/NVIDIA/TensorRT-LLM/pull/1#issuecomment-2)'));
    assert.deepEqual(f.state.mutations, [
      {id: 'IC_2', action: 'show', classifier: undefined},
      {id: 'IC_1', action: 'hide', classifier: 'RESOLVED'},
    ]);
    assert.deepEqual(f.state.queries[0], ['IC_2', 'IC_1']);
  });
}

test('a late reply for an older request cannot hide the current accepted result', async () => {
  const f = fixture([trigger(1), reply(2), trigger(3, request(2)), reply(4, request(2)), reply(5)]);
  await f.publish();
  assert.equal((await f.inspect()).result.comment.id, 4);
  assert.deepEqual([...f.state.hidden].sort(), ['IC_1', 'IC_2', 'IC_3', 'IC_5']);
  assert.ok(!f.state.hidden.has('IC_4'));
  assert.ok(f.summary().body.includes('#issuecomment-4'));
});

test('waiting protects the latest previous request result without carrying its verdict forward', async () => {
  const f = fixture([trigger(1), reply(2), trigger(3, request(2)), reply(4, request(2)),
    trigger(5, request(3)), reply(6)]);
  await f.publish();
  assert.equal((await f.inspect()).result, undefined);
  assert.match(f.summary().body, /\*\*WAITING:/);
  assert.equal(f.state.statuses.at(-1).state, 'pending');
  assert.ok(!f.state.hidden.has('IC_4'));
  assert.ok(!f.state.hidden.has('IC_5'));
  assert.ok(f.state.hidden.has('IC_6'));
});

test('a new waiting request does not restore a previous result hidden after invalidation', async () => {
  const f = fixture([trigger(1), reply(2)]);
  await f.publish();
  f.state.comments.push(comment(3, `SEMANTIC_REVIEW\n${request().id}\nInvalid result`));
  await f.publish();
  assert.ok(f.state.hidden.has('IC_2'));
  f.state.comments.push(trigger(4, request(2)));
  await f.publish();
  assert.ok(f.state.hidden.has('IC_2'));
  assert.match(f.summary().body, /\*\*WAITING:/);
});

for (const action of ['show', 'hide']) {
  test(`an unconfirmed ${action} mutation retains the cleanup marker for read-back on a later scan`, async () => {
    const f = fixture([trigger(1), reply(2)]);
    if (action === 'show') f.state.hidden.add('IC_2');
    f.state.unconfirmedMutation = true;
    await f.publish();
    assert.equal(f.state.mutations.length, 1);
    assert.ok(f.summary().body.includes(pending));
    assert.match(f.state.warnings[0], /was not confirmed/);
    assert.equal(f.state.statuses.at(-1).state, 'success');
    f.state.unconfirmedMutation = false;
    await f.publish();
    assert.equal((await f.inspect()).commentUpdate, false);
    assert.equal(f.state.statuses.length, 1);
  });
}

test('only authenticated production requests and valid bound replies enter moderation', async () => {
  const replay = request(2);
  const malformed = reply(7);
  malformed.body += '\nSEMANTIC_RESULT duplicate';
  const f = fixture([trigger(1), reply(2),
    comment(3, `@coderabbitai replay ${replay.id}`, service), reply(4, replay),
    trigger(5, request(3), {...service, id: 123}), reply(6, request(3)), malformed,
    reply(8, request(), 'PASS', true, {...reviewer, id: 123}),
    comment(9, `Unrelated UUID mention ${request().id}`),
    comment(10, marker, {...publisher, id: 123}),
    comment(11, marker, human), comment(12, marker.replace('#1', '#2'), publisher),
    comment(13, reply(14).body, human)]);
  // Malformed bound replies retain existing waiting authority; presentation must
  // not turn them or unrelated lookalikes into moderation candidates.
  f.state.hidden.add('IC_9');
  await f.publish();
  assert.equal(f.state.creates.length, 1);
  assert.deepEqual([...f.state.hidden], ['IC_9', 'IC_2']);
  assert.ok(f.state.mutations.every(item => ['IC_1', 'IC_2'].includes(item.id)));
  assert.match(f.summary().body, /\*\*WAITING:/);
});

test('multiple trusted summaries update the oldest exact PR marker and never create another', async () => {
  const f = fixture([trigger(1), reply(2), comment(30, marker, publisher), comment(20, marker, publisher)]);
  await f.publish();
  assert.equal(f.state.creates.length, 0);
  assert.ok(f.state.patches.every(item => item.comment_id === 20));
  assert.equal(f.state.comments.find(item => item.id === 30).body, marker);
});

for (const change of ['edited', 'deleted']) {
  test(`${change} accepted source resets summary to waiting and restores trigger without reviving older verdict`, async () => {
    const f = fixture([trigger(1), reply(2), reply(3)]);
    await f.publish();
    assert.ok(f.state.hidden.has('IC_1'));
    if (change === 'edited') f.state.comments.find(item => item.id === 3).body = 'edited away';
    else f.state.comments = f.state.comments.filter(item => item.id !== 3);
    await f.publish();
    assert.equal((await f.inspect()).result, undefined);
    assert.equal(f.state.statuses.at(-1).state, 'pending');
    assert.match(f.summary().body, /\*\*WAITING:/);
    assert.ok(!f.state.hidden.has('IC_1'));
    assert.ok(f.state.hidden.has('IC_2'));
    assert.ok(!f.state.hidden.has('IC_3'));
  });
}

test('an ambiguously delivered summary POST is not retried and a fresh scan repairs the same comment', async () => {
  const f = fixture([trigger(1), reply(2)]);
  f.state.createError = 'after';
  await f.publish();
  assert.equal(f.state.creates.length, 1);
  assert.equal(f.state.mutations.length, 0);
  assert.ok(f.summary().body.includes(pending));
  assert.equal((await f.inspect()).commentUpdate, true);
  assert.equal(f.state.statuses.at(-1).state, 'success');
  assert.match(f.state.warnings[0], /summary publication incomplete \(HTTP 503\)/);
  assert.ok(!f.state.warnings[0].includes('private remote detail'));
  f.state.createError = undefined;
  await f.publish();
  assert.equal(f.state.creates.length, 1);
  assert.equal(f.state.statuses.length, 1);
  assert.equal((await f.inspect()).commentUpdate, false);
  assert.deepEqual([...f.state.hidden], ['IC_1']);
});

test('failed summary PATCH preserves the new status and prevents moderation', async () => {
  const f = fixture([trigger(1)]);
  await f.publish();
  f.state.comments.push(reply(2));
  f.state.patchError = true;
  await f.publish();
  assert.equal(f.state.statuses.at(-1).state, 'success');
  assert.equal(f.state.mutations.length, 0);
  assert.equal((await f.inspect()).commentUpdate, true);
  assert.match(f.state.warnings[0], /summary publication incomplete \(HTTP 403\)/);
});

for (const failure of ['query', 'mutation', 'denied', 'missing', 'node-id']) {
  test(`${failure} moderation failure leaves visible summary and retry marker without changing verdict`, async () => {
    const f = fixture([trigger(1), reply(2)]);
    if (failure === 'query') f.state.queryError = true;
    if (failure === 'mutation') f.state.mutationError = true;
    if (failure === 'denied') f.state.denied.add('IC_1');
    if (failure === 'missing') f.state.missing.add('IC_1');
    if (failure === 'node-id') delete f.state.comments[0].node_id;
    await f.publish();
    assert.equal(f.state.statuses.at(-1).state, 'success');
    assert.ok(f.summary().body.includes('**PASS:'));
    assert.ok(f.summary().body.includes(pending));
    assert.equal((await f.inspect()).commentUpdate, true);
    assert.ok(!f.state.warnings[0].includes('private remote detail'));
    if (failure === 'denied') assert.match(f.state.warnings[0], /token cannot minimize/);
    if (failure === 'missing') assert.match(f.state.warnings[0], /deleted or unavailable/);
    if (failure === 'node-id') assert.match(f.state.warnings[0], /missing comment node ID/);
    f.state.queryError = f.state.mutationError = false;
    f.state.denied.clear();
    f.state.missing.clear();
    f.state.comments[0].node_id = 'IC_1';
    await f.publish();
    assert.equal(f.state.statuses.length, 1);
    assert.equal((await f.inspect()).commentUpdate, false);
  });
}

test('failure to restore the latest reply stops all minimization', async () => {
  const f = fixture([trigger(1), reply(2), reply(3)]);
  f.state.hidden.add('IC_3');
  f.state.denied.add('IC_3');
  await f.publish();
  assert.equal(f.state.mutations.length, 0);
  assert.ok(f.summary().body.includes(pending));
  assert.match(f.state.warnings[0], /token cannot restore the current comment/);
});

test('cleanup progresses by at most twenty minimizations per publish and converges to a no-op', async () => {
  const f = fixture([trigger(1), ...Array.from({length: 25}, (_, index) => reply(index + 2))]);
  await f.publish();
  assert.equal(f.state.mutations.length, 20);
  assert.ok(f.summary().body.includes(pending));
  assert.equal((await f.inspect()).commentUpdate, true);
  await f.publish();
  assert.equal(f.state.mutations.length, 25);
  assert.equal(f.state.statuses.length, 1);
  assert.equal((await f.inspect()).commentUpdate, false);
  assert.ok(!f.state.hidden.has('IC_26'));
  await f.publish();
  assert.equal(f.state.mutations.length, 25);
  assert.equal(f.state.creates.length, 1);
});

test('visibility lookups are bounded and query the protected reply before cleanup candidates', async () => {
  const f = fixture([trigger(1), ...Array.from({length: 110}, (_, index) => reply(index + 2))]);
  await f.publish();
  assert.equal(f.state.queries[0][0], 'IC_111');
  assert.deepEqual(f.state.queries.map(ids => ids.length), [100, 11]);
  assert.equal(f.state.mutations.length, 20);
  assert.ok(f.summary().body.includes(pending));
});

test('a late superseded reply changes the cleanup digest and repairs presentation without another status', async () => {
  const f = fixture([trigger(1), reply(2), trigger(3, request(2)), reply(4, request(2))]);
  await f.publish();
  const before = f.summary().body;
  f.state.comments.push(reply(5));
  const state = await f.inspect();
  assert.equal(state.update, undefined);
  assert.equal(state.commentUpdate, true);
  await f.publish();
  assert.notEqual(f.summary().body, before);
  assert.equal(f.state.statuses.length, 1);
  assert.ok(f.state.hidden.has('IC_5'));
  assert.ok(!f.state.hidden.has('IC_4'));
});

test('status failure prevents comment publication and moderation', async () => {
  const f = fixture([trigger(1), reply(2)]);
  f.state.statusError = true;
  await assert.rejects(f.publish(), /status failed/);
  assert.equal(f.state.creates.length, 0);
  assert.equal(f.state.mutations.length, 0);
});

test('legacy cleanup failure prevents comment publication and moderation', async () => {
  const f = fixture([trigger(1), reply(2)]);
  f.state.checks.push({id: 3, name: NAME, head_sha: request().head,
    app: {slug: 'github-actions'}, external_id: `semantic-review:1:${request().id}`,
    status: 'in_progress'});
  f.state.cleanupError = true;
  await assert.rejects(f.publish(), /cleanup failed/);
  assert.equal(f.state.statuses.at(-1).state, 'success');
  assert.equal(f.state.creates.length, 0);
  assert.equal(f.state.mutations.length, 0);
});

test('non-string branch metadata cannot create a request or crash presentation', async () => {
  for (const branch of [['release/x'], {branch: 'main'}, null, 1]) {
    const value = {...request(), branch};
    const f = fixture([trigger(1, value), reply(2, value)]);
    assert.deepEqual(requests(f.state.comments), []);
    await f.publish();
    assert.equal(f.state.creates.length, 0);
  }
});

test('a replay without a trusted production request has no presentation side effects', async () => {
  const f = fixture([comment(1, `@coderabbitai ${request().id}`, service), reply(2)]);
  await f.publish();
  assert.equal(f.state.statuses.length, 0);
  assert.equal(f.state.creates.length, 0);
  assert.equal(f.state.mutations.length, 0);
});
