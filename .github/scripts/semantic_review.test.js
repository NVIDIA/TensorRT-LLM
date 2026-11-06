// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const test = require('node:test');
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {join} = require('node:path');
const {NAME, identity, statusContext, requests, parseResult, command, awaiting, reviewState, publish,
  STICKY_MARKER, tidy} = require('./semantic_review');
const cases = require('./semantic_review_cases');

const repo = {owner: 'NVIDIA', repo: 'TensorRT-LLM'};
const service = {login: 'trtllm-agent', id: 296075020, type: 'User'};
const bot = {login: 'coderabbitai[bot]', id: 136622811, type: 'Bot'};
const id = n => `00000000-0000-4000-8000-${String(n).padStart(12, '0')}`;
const request = (n = 1, extra = {}) => ({id: id(n), head: 'a'.repeat(40),
  target: 'b'.repeat(40), mergeBase: 'c'.repeat(40), branch: 'main', ...extra});
const comment = (n, body, user = bot) => ({id: n, body, user, node_id: `IC_${n}`,
  created_at: new Date(1700000000000 + n * 1000).toISOString(),
  html_url: `https://github.com/NVIDIA/TensorRT-LLM/pull/1#issuecomment-${n}`});
const record = (n, r) => comment(n, `<!-- semantic-review-request:${JSON.stringify(r)} -->`, service);
function reply(n, r, verdict = 'PASS', evidence = true) {
  const body = `Analysis before the result.\nSEMANTIC_REVIEW\n` +
    (evidence ? [r.head, r.target].map(sha =>
      `https://github.com/NVIDIA/TensorRT-LLM/blob/${sha}/path.py#L12`).join('\n') + '\n' : '') +
    `SEMANTIC_RESULT request_id=${r.id} head=${r.head} target=${r.target} merge_base=${r.mergeBase} verdict=${verdict}\n`;
  return comment(n, body);
}
const publisher = {login: 'github-actions[bot]', id: 41898282, type: 'Bot'};
const legacy = (r = request(), extra = {}) => ({id: 100, name: NAME, head_sha: r.head,
  external_id: identity(1, r), app: {slug: 'github-actions'}, status: 'in_progress',
  conclusion: null, output: awaiting(r), ...extra});
function harness(r = request()) {
  const state = {comments: [record(10, r)], writes: [], updates: [], summaries: [], reads: 0,
    refs: [], updateFailures: new Set(), statusFailures: 0, statusErrorCode: 503, statuses: [], checks: [],
    commentWrites: [], commentEdits: [], minimized: [], restored: [], graphqlFailures: 0,
    warnings: []};
  const github = {
    graphql: async (query, variables) => {
      if (state.graphqlFailures-- > 0) {
        throw Object.assign(new Error('GraphQL failed'), {status: 502});
      }
      if (query.includes('unminimizeComment')) {
        state.minimized.splice(state.minimized.indexOf(variables.id), 1);
        state.restored.push(variables.id);
        return {unminimizeComment: {unminimizedComment: {isMinimized: false}}};
      }
      if (query.includes('minimizeComment')) {
        state.minimized.push(variables.id);
        return {minimizeComment: {minimizedComment: {isMinimized: true}}};
      }
      return {nodes: variables.ids.map(id => ({id, isMinimized: state.minimized.includes(id)}))};
    },
    paginate: async (method, args) => {
      if (method === github.rest.issues.listComments) {
        state.reads += 1;
        return structuredClone(state.comments);
      }
      const {data} = await method(args);
      return data.check_runs || data;
    },
    rest: {issues: {listComments() {},
      createComment: async ({issue_number, body}) => {
        const id = Math.max(0, ...state.comments.map(comment => comment.id)) + 1;
        state.commentWrites.push({issue_number, body});
        state.comments.push({id, body, user: publisher, node_id: `IC_${id}`,
          created_at: new Date(1700000000000 + id * 1000).toISOString()});
        return {data: {id}};
      },
      updateComment: async ({comment_id, body}) => {
        state.commentEdits.push({comment_id, body});
        Object.assign(state.comments.find(comment => comment.id === comment_id), {body});
      },
    }, repos: {
      listCommitStatusesForRef: async ({ref}) => ({data: structuredClone(
        state.statuses.filter(status => status.sha === ref))}),
      createCommitStatus: async args => {
        if (state.statusFailures-- > 0) {
          throw Object.assign(new Error('Status failed'), {status: state.statusErrorCode});
        }
        state.writes.push(args);
        const status = {...args, id: Math.max(0, ...state.statuses.map(item => item.id)) + 1,
          creator: publisher};
        state.statuses.push(status);
        state.status = status;
        return {data: structuredClone(status)};
      },
    }, checks: {
      listForRef: async ({ref, check_name: name, filter}) => {
        assert.equal(filter, 'all');
        state.refs.push(ref);
        return {data: {check_runs: structuredClone(state.checks.filter(check =>
          check.head_sha === ref && check.name === name))}};
      },
      update: async update => {
        assert.equal(update.conclusion, 'cancelled');
        if (state.updateFailures.delete(update.check_run_id)) {
          throw Object.assign(new Error('Check update failed'), {status: 503});
        }
        state.updates.push(update);
        Object.assign(state.checks.find(item => item.id === update.check_run_id), update);
      },
    }},
  };
  const core = {summary: {addRaw(text) {state.summaries.push(text); return this;}, async write() {}},
    warning(message) {state.warnings.push(message);},
    setFailed() {throw new Error('AI verdict must not fail the orchestration job');}};
  const deliver = async event => publish({github, core, context: {
    repo, eventName: 'issue_comment', payload: {issue: {number: 1, pull_request: {}}, comment: event},
  }});
  const sticky = () => state.comments.find(comment =>
    comment.user?.login === publisher.login && comment.body?.includes(STICKY_MARKER));
  return {state, deliver, sticky,
    inspect: extra => reviewState({github, repo, number: 1, ...extra}),
    repair: (comments, extra) => publish({github, core, context: {repo}, number: 1, comments, ...extra}),
    tidy: extra => tidy({github, core, context: {repo}, number: 1, ...extra})};
}

test('request records require the pinned account and valid immutable metadata', () => {
  const r = request();
  assert.equal(requests([record(10, r)]).length, 1);
  assert.equal(requests([record(10, {...r, checkId: 100})]).length, 1);
  assert.equal(requests([record(10, {...r, automaticRetryOf: id(2)})])[0].automaticRetryOf, id(2));
  for (const user of [{...service, id: 1}, {...service, type: 'Bot'}, bot]) {
    assert.deepEqual(requests([{...record(10, r), user}]), []);
  }
  for (const extra of [{id: 'unknown'}, {head: 'main'}, {branch: 'feature/x'}, {checkId: 0},
    ...[null, false, 3, 'unknown', [id(2)], {id: id(2)}].map(automaticRetryOf => ({automaticRetryOf}))]) {
    assert.deepEqual(requests([record(10, {...r, ...extra})]), []);
  }
  assert.deepEqual(requests([comment(10, '<!-- semantic-review-request:{invalid} -->', service)]), []);
});

test('protocol requires matching request ID, all revisions and the real reviewer', () => {
  const r = request();
  assert.equal(parseResult(reply(20, r), r, repo).verdict, 'PASS');
  for (const extra of [{id: id(2)}, {head: 'd'.repeat(40)}, {target: 'd'.repeat(40)},
    {mergeBase: 'd'.repeat(40)}]) {
    assert.equal(parseResult(reply(20, {...r, ...extra}), r, repo), undefined);
  }
  for (const user of [{...bot, id: 1}, {...bot, type: 'User'}, service]) {
    assert.equal(parseResult({...reply(20, r), user}, r, repo), undefined);
  }
});

test('missing or wrong-repository evidence is inconclusive, never PASS', () => {
  const r = request();
  assert.equal(parseResult(reply(20, r, 'PASS', false), r, repo).verdict, 'INCONCLUSIVE');
  assert.equal(parseResult(reply(20, r, 'FAIL', false), r, repo).verdict, 'INCONCLUSIVE');
  const wrong = reply(20, r);
  wrong.body = wrong.body.replaceAll('NVIDIA/TensorRT-LLM/blob', 'elsewhere/project/blob');
  assert.equal(parseResult(wrong, r, repo).verdict, 'INCONCLUSIVE');
  assert.equal(parseResult(reply(20, r, 'INCONCLUSIVE', false), r, repo).verdict, 'INCONCLUSIVE');
});

test('Markdown result headings publish verdicts without accepting quoted or duplicate sections', async () => {
  const r = request();
  for (const heading of ['# SEMANTIC_REVIEW', '## SEMANTIC_REVIEW', '###### SEMANTIC_REVIEW']) {
    const message = reply(20, r, 'FAIL');
    message.body = message.body.replace('SEMANTIC_REVIEW', heading);
    const {state, deliver} = harness(r);
    state.comments.push(message);
    await deliver(message);
    assert.equal(state.status.state, 'failure');
    assert.equal(parseResult({...message, body: message.body.replace(heading, `> ${heading}`)}, r, repo), undefined);
    assert.equal(parseResult({...message, body: `${message.body}\nSEMANTIC_REVIEW\n`}, r, repo), undefined);
  }
});

test('conflicting records and missing protocol markers are rejected', () => {
  const r = request();
  const message = reply(20, r);
  message.body += reply(21, r, 'FAIL').body.split('SEMANTIC_REVIEW\n')[1];
  assert.equal(parseResult(message, r, repo), undefined);
  assert.equal(parseResult(comment(20, 'PASS'), r, repo), undefined);
});

test('waiting is pending with request link and no invented result', async () => {
  const {state, repair} = harness();
  const current = await repair();
  assert.equal(current.result, undefined);
  assert.equal(state.status.state, 'pending');
  assert.equal(state.status.context, statusContext(1));
  assert.equal(state.status.description, 'Waiting for CodeRabbit response');
  assert.match(state.status.target_url, /#issuecomment-10$/);
  assert.equal(Object.hasOwn(state.status, 'conclusion'), false);
  assert.equal((await repair()).update, undefined);
  assert.equal(state.writes.length, 1);
});

test('all valid verdicts finish internally, with distinct descriptions and reply links', async () => {
  for (const [verdict, evidence, expected, description] of [
    ['PASS', true, 'success', 'No semantic conflict found (best effort)'],
    ['FAIL', true, 'failure', 'Possible semantic conflict'],
    ['INCONCLUSIVE', false, 'pending', 'Review completed: inconclusive'],
    ['PASS', false, 'pending', 'Review completed: inconclusive'],
    ['FAIL', false, 'pending', 'Review completed: inconclusive'],
  ]) {
    const r = request();
    const {state, repair} = harness(r);
    state.comments.push(reply(20, r, verdict, evidence));
    const current = await repair();
    assert.equal(state.status.state, expected);
    assert.equal(state.status.description, description);
    assert.ok(current.result);
    assert.equal(current.result.verdict, expected === 'pending' ? 'INCONCLUSIVE' : verdict);
    assert.match(state.status.target_url, /semantic_review_source=20#issuecomment-20$/);
    assert.match(state.summaries[0], new RegExp(r.target));
    assert.equal((await repair()).update, undefined);
  }
});

test('late old replies cannot override a new request even with the same head and target', async () => {
  for (const changed of [{}, {target: 'd'.repeat(40)}, {head: 'd'.repeat(40)}]) {
    const old = request();
    const current = request(2, changed);
    const {state, repair} = harness(old);
    state.comments.push(reply(20, old));
    await repair();
    state.comments.push(record(30, current), reply(40, old));
    await repair();
    assert.equal(state.status.state, 'pending');
    assert.match(state.status.target_url, new RegExp(current.id));
    assert.equal(state.status.sha, current.head);
    state.comments.push(reply(50, current, 'FAIL'), reply(60, old));
    await repair();
    assert.equal(state.status.state, 'failure');
    assert.match(state.status.target_url, /#issuecomment-50$/);
  }
});

test('same-head PRs use independent contexts and source watermarks', async () => {
  const r = request();
  const {state, repair} = harness(r);
  const firstComments = [record(10, r), reply(30, r, 'FAIL')];
  await repair(firstComments);
  const first = structuredClone(state.status);
  await repair([record(10, r), reply(20, r)], {number: 2});
  assert.equal(state.status.state, 'success');
  assert.equal(state.status.context, statusContext(2));
  assert.notEqual(state.status.context, first.context);
  assert.equal((await repair(firstComments)).update, undefined);
  assert.equal(state.writes.length, 2);
});

test('edited or deleted published replies revoke a verdict without falling back to older PASS', async () => {
  for (const change of ['delete', 'invalid', 'wrong revision']) {
    const r = request();
    const {state, repair} = harness(r);
    const latest = reply(30, r, 'FAIL');
    state.comments.push(reply(20, r), latest);
    await repair();
    if (change === 'delete') state.comments.pop();
    else latest.body = change === 'invalid' ? 'Correcting my analysis.' :
      reply(30, {...r, target: 'd'.repeat(40)}).body;
    const current = await repair();
    assert.equal(current.result, undefined);
    assert.equal(state.status.state, 'pending');
    assert.equal(state.status.description, 'Waiting for CodeRabbit response');
    assert.match(state.status.target_url, /semantic_review_source=30#issuecomment-10$/);
    assert.equal((await repair()).update, undefined);
    state.comments.push(reply(40, r));
    assert.equal((await repair()).result.verdict, 'PASS');
    assert.equal(state.status.state, 'success');
  }
});

test('malformed new replies invalidate older results and retain their watermark after deletion', async () => {
  const r = request();
  for (const body of [`Request ${r.id}: still investigating.`,
    reply(30, r).body.replace('verdict=PASS', 'verdict=UNKNOWN'),
    reply(30, r).body.replace('SEMANTIC_REVIEW', 'Missing heading'),
    reply(30, r).body + `SEMANTIC_RESULT request_id=${r.id} malformed\n`]) {
    const {state, repair} = harness(r);
    state.comments.push(reply(20, r));
    await repair();
    state.comments.push(comment(30, body));
    assert.equal((await repair()).result, undefined);
    assert.match(state.status.target_url, /semantic_review_source=30#issuecomment-10$/);
    state.comments.pop();
    assert.equal((await repair()).result, undefined);
    assert.equal(state.writes.length, 2);
  }
});

test('wrong UUID or revision replies do not invalidate an earlier valid result', async () => {
  for (const extra of [{id: id(2)}, {head: 'd'.repeat(40)}, {target: 'd'.repeat(40)},
    {mergeBase: 'd'.repeat(40)}]) {
    const r = request();
    const {state, repair} = harness(r);
    state.comments.push(reply(20, r));
    await repair();
    state.comments.push(reply(30, {...r, ...extra}, 'FAIL'));
    const current = await repair();
    assert.equal(current.result.comment.id, 20);
    assert.equal(current.update, undefined);
    assert.equal(state.writes.length, 1);
  }
});

test('a bot-looking user cannot publish or revoke results', async () => {
  const r = request();
  const {state, deliver} = harness(r);
  const forged = {...reply(20, r), user: {...bot, id: 123}};
  state.comments.push(forged);
  await deliver(forged);
  assert.equal(state.writes.length, 0);
  assert.equal(state.updates.length, 0);
});

test('status watermarks require the pinned publisher, exact context, head, repository and request', async () => {
  const r = request();
  for (const extra of [
    {creator: {...publisher, id: 1}}, {creator: {...publisher, type: 'User'}},
    {creator: {...publisher, login: 'other[bot]'}}, {context: statusContext(2)},
    {sha: 'd'.repeat(40)}, {target_url: 'not a URL'},
    {target_url: `https://github.com/elsewhere/project/pull/1?semantic_review_request=${r.id}&semantic_review_source=30`},
    {target_url: `https://github.com/NVIDIA/TensorRT-LLM/pull/2?semantic_review_request=${r.id}&semantic_review_source=30`},
    {target_url: `https://github.com/NVIDIA/TensorRT-LLM/pull/1?semantic_review_request=${id(2)}&semantic_review_source=30`},
  ]) {
    const {state, repair} = harness(r);
    state.comments.push(reply(30, r, 'FAIL'));
    await repair();
    Object.assign(state.status, extra);
    state.comments = [record(10, r), reply(20, r)];
    assert.equal((await repair()).result.verdict, 'PASS');
    assert.equal(state.status.state, 'success');
    assert.deepEqual(state.status.creator, publisher);
  }
});

test('a missing status never prevents result parsing and publication failure is recoverable', async () => {
  const r = request();
  const {state, inspect, repair} = harness(r);
  const comments = [record(10, r), reply(20, r, 'FAIL')];
  const current = await inspect({comments});
  assert.equal(current.result.verdict, 'FAIL');
  assert.equal(current.update.state, 'failure');
  assert.equal(state.reads, 0);
  assert.equal(state.writes.length, 0);
  state.statusFailures = 1;
  await assert.rejects(repair(comments), {status: 503});
  await repair(comments);
  assert.equal((await repair(comments)).update, undefined);
  assert.equal(state.writes.length, 1);
  assert.equal(state.summaries.length, 1);
});

test('status API errors, including the per-SHA context cap, are reported without alternate contexts', async () => {
  for (const status of [503, 422]) {
    const {state, repair} = harness();
    state.statusFailures = 1;
    state.statusErrorCode = status;
    await assert.rejects(repair(), {status});
    assert.equal(state.writes.length, 0);
  }
});

test('unchanged pending scans do not append statuses; stale descriptions are repaired', async () => {
  const {state, repair} = harness();
  await repair();
  for (let n = 0; n < 5; n++) assert.equal((await repair()).update, undefined);
  assert.equal(state.writes.length, 1);
  state.status.description = 'Stale';
  await repair();
  assert.equal(state.writes.length, 2);
  assert.equal(state.status.description, 'Waiting for CodeRabbit response');
});

test('migration inherits a deleted source from the newest matching legacy check', async () => {
  const r = request(1, {checkId: 100});
  const {state, repair} = harness(r);
  state.comments.push(reply(20, r));
  state.checks.push(legacy(r), legacy(r, {id: 101, status: 'completed', conclusion: 'failure',
    output: {summary: '<!-- semantic-review-source:30 -->'}}));
  const current = await repair();
  assert.equal(current.result, undefined);
  assert.match(state.status.target_url, /semantic_review_source=30#issuecomment-10$/);
  assert.equal(state.checks[0].conclusion, 'cancelled');
  assert.equal(state.checks[1].conclusion, 'failure');
  assert.equal((await repair()).update, undefined);
});

test('legacy neutral without a valid reply becomes waiting, not a completed analysis', async () => {
  const r = request(1, {checkId: 100});
  const {state, repair} = harness(r);
  state.checks.push(legacy(r, {status: 'completed', conclusion: 'neutral'}));
  assert.equal((await repair()).result, undefined);
  assert.equal(state.status.description, 'Waiting for CodeRabbit response');
  assert.equal(state.updates.length, 0);
});

test('new requests cannot inherit an old legacy reply watermark', async () => {
  const old = request();
  const r = request(2);
  const {state, repair} = harness(r);
  state.comments.push(reply(20, r));
  state.checks.push(legacy(old, {output: {summary: '<!-- semantic-review-source:30 -->'}}));
  assert.equal((await repair()).result.verdict, 'PASS');
});

test('legacy pending cleanup happens after status publication and retries idempotently', async () => {
  const r = request();
  const {state, repair} = harness(r);
  state.checks.push(legacy(r), legacy(r, {id: 101}));
  state.statusFailures = 1;
  await assert.rejects(repair(), {status: 503});
  assert.equal(state.updates.length, 0);
  state.updateFailures.add(101);
  await assert.rejects(repair(), {status: 503});
  assert.equal(state.writes.length, 1);
  assert.deepEqual(state.updates.map(update => update.check_run_id), [100]);
  await repair();
  assert.equal(state.writes.length, 1);
  assert.deepEqual(state.updates.map(update => update.check_run_id), [100, 101]);
  assert.deepEqual((await repair()).cleanup, []);
});

test('legacy cleanup excludes completed checks and other PRs, apps, names or heads', async () => {
  const r = request();
  const {state, repair} = harness(r);
  const excluded = [
    legacy(r, {id: 102, external_id: identity(2, r)}),
    legacy(r, {id: 103, app: {slug: 'untrusted'}}),
    legacy(r, {id: 104, name: 'Another check'}),
    legacy(r, {id: 105, head_sha: 'd'.repeat(40)}),
    legacy(r, {id: 106, status: 'completed', conclusion: 'success'}),
  ];
  state.checks.push(legacy(r), ...excluded);
  await repair();
  assert.deepEqual(state.updates.map(update => update.check_run_id), [100]);
  assert.equal(excluded.slice(0, 4).every(check => check.status === 'in_progress'), true);
  assert.equal(excluded[4].conclusion, 'success');
});

test('explicit head allows scoped legacy cleanup without a trusted request', async () => {
  const r = request();
  const {state, inspect, repair} = harness(r);
  state.checks.push(legacy(r));
  assert.equal(await inspect({comments: []}), undefined);
  await repair([], {head: r.head});
  assert.equal(state.checks[0].conclusion, 'cancelled');
  assert.equal(state.writes.length, 0);
});

test('legacy cleanup reads only recorded and explicit heads, deduplicating identical refs', async () => {
  const r = request();
  const {state, inspect, repair} = harness(r);
  await inspect({head: r.head});
  assert.deepEqual(state.refs, [r.head]);
  state.refs = [];
  const newHead = 'd'.repeat(40);
  state.checks.push(legacy(r), legacy(r, {id: 101, head_sha: newHead}),
    legacy(r, {id: 102, head_sha: 'e'.repeat(40)}));
  await repair(undefined, {head: newHead});
  assert.deepEqual(state.refs, [r.head, newHead]);
  assert.deepEqual(state.updates.map(update => update.check_run_id), [100, 101]);
  assert.equal(state.checks[2].status, 'in_progress');
});

test('fixed historical inputs round-trip request identity and all three result verdicts', () => {
  assert.equal(cases.length, 12);
  for (const [index, fixture] of cases.entries()) {
    const r = request(index + 1, fixture);
    const body = command(r);
    for (const value of [r.id, r.head, r.target, r.mergeBase]) assert.ok(body.includes(value));
    assert.ok(body.startsWith('@coderabbitai\n'));
    const [recorded] = requests([record(10, r)]);
    assert.equal(identity(fixture.pr, recorded), `semantic-review:${fixture.pr}:${r.id}`);
    for (const verdict of ['PASS', 'FAIL', 'INCONCLUSIVE']) {
      assert.equal(parseResult(reply(20, r, verdict), recorded, repo).verdict, verdict);
    }
  }
});

test('sticky summary is created once, edited in place, and never pings the reviewer', async () => {
  const r = request();
  const {state, sticky, tidy} = harness(r);
  await tidy();
  assert.equal(state.commentWrites.length, 1);
  assert.ok(sticky().body.startsWith(STICKY_MARKER));
  assert.match(sticky().body, /Waiting for CodeRabbit response/);
  assert.doesNotMatch(sticky().body, /@coderabbitai/);
  assert.deepEqual(state.minimized, []);
  await tidy();
  assert.equal(state.commentWrites.length, 1);
  assert.equal(state.commentEdits.length, 0);
  state.comments.push(reply(20, r, 'FAIL'));
  await tidy();
  assert.equal(state.commentWrites.length, 1);
  assert.equal(state.commentEdits.length, 1);
  assert.match(sticky().body, /Possible semantic conflict/);
  assert.match(sticky().body, /#issuecomment-20\)/);
  assert.deepEqual(state.minimized.sort(), ['IC_10', 'IC_20']);
  await tidy();
  assert.equal(state.commentEdits.length, 1);
  assert.deepEqual(state.minimized.sort(), ['IC_10', 'IC_20']);
});

test('superseded request pairs are minimized; the active pair waits for its verdict', async () => {
  const old = request();
  const {state, sticky, tidy} = harness(old);
  state.comments.push(reply(20, old));
  await tidy();
  assert.deepEqual(state.minimized.sort(), ['IC_10', 'IC_20']);
  const next = request(2, {target: 'd'.repeat(40)});
  state.comments.push(record(30, next));
  await tidy();
  assert.deepEqual(state.minimized.sort(), ['IC_10', 'IC_20']);
  assert.match(sticky().body, /Waiting for CodeRabbit response/);
  assert.match(sticky().body, /\| PASS \|/);
  state.comments.push(reply(40, next, 'FAIL'));
  await tidy();
  assert.deepEqual(state.minimized.sort(), ['IC_10', 'IC_20', 'IC_30', 'IC_40']);
  assert.match(sticky().body, /Possible semantic conflict/);
});

test('a superseded request without a reply is minimized and listed without a result', async () => {
  const old = request();
  const {state, sticky, tidy} = harness(old);
  state.comments.push(record(30, request(2)));
  await tidy();
  assert.deepEqual(state.minimized, ['IC_10']);
  assert.match(sticky().body, /\| NO RESULT \|/);
  assert.match(sticky().body, /\| WAITING \|/);
});

test('the sticky summary follows a revoked verdict instead of an older PASS', async () => {
  const r = request();
  const {state, sticky, repair, tidy} = harness(r);
  const latest = reply(30, r, 'FAIL');
  state.comments.push(reply(20, r), latest);
  await repair();
  state.comments.splice(state.comments.indexOf(latest), 1);
  await repair();
  assert.equal(state.status.description, 'Waiting for CodeRabbit response');
  await tidy();
  assert.match(sticky().body, /Waiting for CodeRabbit response/);
  assert.doesNotMatch(sticky().body, /No semantic conflict found/);
});

test('history rows follow the newest reply, and all bound replies are minimized', async () => {
  const a = request();
  const {state, sticky, repair, tidy} = harness(a);
  state.comments.push(reply(20, a));
  await repair();
  state.comments.push(reply(30, a, 'FAIL'));
  await repair();
  state.comments.push(record(40, request(2, {target: 'd'.repeat(40)})));
  await tidy();
  assert.match(sticky().body, /\| FAIL \| \[reply\]\([^)]*#issuecomment-30\)/);
  assert.doesNotMatch(sticky().body, /\| PASS \|/);
  assert.deepEqual(state.minimized.sort(), ['IC_10', 'IC_20', 'IC_30']);
});

test('history rows do not resurrect a verdict revoked before supersession', async () => {
  const a = request();
  const {state, sticky, repair, tidy} = harness(a);
  state.comments.push(reply(20, a));
  await repair();
  const revoked = reply(30, a, 'FAIL');
  state.comments.push(revoked);
  await repair();
  state.comments.splice(state.comments.indexOf(revoked), 1);
  await repair();
  state.comments.push(record(40, request(2, {target: 'd'.repeat(40)})));
  await tidy();
  assert.match(sticky().body, /\| NO RESULT \|/);
  assert.doesNotMatch(sticky().body, /\| PASS \|/);
  assert.deepEqual(state.minimized.sort(), ['IC_10', 'IC_20']);
});

test('a revoked verdict restores the minimized active request to visible', async () => {
  const r = request();
  const {state, sticky, tidy} = harness(r);
  const verdict = reply(20, r);
  state.comments.push(verdict);
  await tidy();
  assert.deepEqual(state.minimized.sort(), ['IC_10', 'IC_20']);
  state.comments.splice(state.comments.indexOf(verdict), 1);
  state.graphqlFailures = 1;
  await tidy();
  assert.match(sticky().body, /Waiting for CodeRabbit response/);
  assert.deepEqual(state.restored, []);
  assert.equal(state.warnings.length, 1);
  await tidy();
  assert.deepEqual(state.restored, ['IC_10']);
  assert.deepEqual(state.minimized, ['IC_20']);
  await tidy();
  assert.deepEqual(state.restored, ['IC_10']);
});

test('minimization failures warn without blocking the sticky summary, then heal', async () => {
  const r = request();
  const {state, tidy} = harness(r);
  state.comments.push(reply(20, r));
  state.graphqlFailures = 1;
  await tidy();
  assert.equal(state.commentWrites.length, 1);
  assert.deepEqual(state.minimized, []);
  assert.equal(state.warnings.length, 1);
  state.graphqlFailures = 0;
  await tidy();
  assert.deepEqual(state.minimized.sort(), ['IC_10', 'IC_20']);
});

test('tidy without a semantic review request or trusted event is a no-op', async () => {
  const {state, tidy} = harness();
  state.comments = [comment(5, 'Ordinary review feedback.')];
  await tidy();
  assert.equal(state.commentWrites.length + state.commentEdits.length, 0);
  assert.deepEqual(state.minimized, []);
  assert.equal(await tidy({number: undefined, context: {repo, payload: {issue: {number: 1}}}}), undefined);
});

test('disabled semantic workflow cannot request or process bot replies and retains trusted job isolation', () => {
  const workflow = readFileSync(join(__dirname, '../workflows/semantic-review.yml'), 'utf8');
  assert.doesNotMatch(workflow, /pull_request_target:|pull_request:/);
  assert.doesNotMatch(workflow, /^  (schedule|issue_comment|pull_request|pull_request_target):/m);
  assert.match(workflow, /^  workflow_dispatch:/m);
  const discoverJob = workflow.split('  discover:')[1].split('  request:')[0];
  const requestJob = workflow.split('  request:')[1].split('  publish:')[0];
  assert.match(discoverJob, /if: \$\{\{ false \}\}/);
  assert.match(requestJob, /needs: discover/);
  assert.match(workflow, /group: semantic-review-pr-\$\{\{ matrix.number \}\}/);
  assert.match(workflow, /group: semantic-review-pr-\$\{\{ github.event.issue.number \}\}/);
  assert.equal(workflow.match(/      cancel-in-progress: false\n      queue: max/g).length, 3);
  assert.equal(workflow.match(/ref: \$\{\{ github.event.repository.default_branch \}\}/g).length, 4);
  assert.equal(workflow.match(/secrets\./g).length, 1);
  const publishJob = workflow.split('  publish:')[1].split('  tidy:')[0];
  const tidyJob = workflow.split('  tidy:')[1];
  assert.match(publishJob, /if: \$\{\{ false \}\}/);
  assert.doesNotMatch(publishJob, /SEMANTIC_COMMAND_TOKEN|issues: write|pull-requests: write/);
  assert.doesNotMatch(tidyJob, /SEMANTIC_COMMAND_TOKEN|statuses: write|checks: write/);
  assert.match(tidyJob, /needs: publish/);
  assert.match(tidyJob, /if: needs.publish.outputs.active == 'true'/);
  assert.match(tidyJob, /issues: write/);
  assert.equal(workflow.match(/statuses: write/g).length, 2);
  assert.match(workflow, /statuses: read/);
});
