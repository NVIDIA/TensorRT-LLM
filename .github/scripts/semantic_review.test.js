// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const test = require('node:test');
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {join} = require('node:path');
const {NAME, identity, requests, parseResult, command, publish} = require('./semantic_review');
const cases = require('./semantic_review_cases');

const repo = {owner: 'NVIDIA', repo: 'TensorRT-LLM'};
const service = {login: 'trtllm-agent', id: 296075020, type: 'User'};
const bot = {login: 'coderabbitai[bot]', id: 136622811, type: 'Bot'};
const id = n => `00000000-0000-4000-8000-${String(n).padStart(12, '0')}`;
const request = (n = 1, extra = {}) => ({id: id(n), head: 'a'.repeat(40),
  target: 'b'.repeat(40), mergeBase: 'c'.repeat(40), branch: 'main', checkId: 100, ...extra});
const comment = (n, body, user = bot) => ({id: n, body, user,
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
function harness(r = request()) {
  const state = {comments: [record(10, r)], updates: [], summaries: [],
    check: {id: r.checkId, name: NAME, head_sha: r.head, external_id: identity(1, r),
      app: {slug: 'github-actions'}, status: 'completed', conclusion: 'neutral'}};
  const github = {
    paginate: async () => structuredClone(state.comments),
    rest: {issues: {listComments() {}}, checks: {
      get: async () => ({data: structuredClone(state.check)}),
      update: async update => { state.updates.push(update); Object.assign(state.check, update); },
    }},
  };
  const core = {summary: {addRaw(text) {state.summaries.push(text); return this;}, async write() {}},
    setFailed() {throw new Error('AI verdict must not fail the orchestration job');}};
  const deliver = async event => publish({github, core, context: {
    repo, eventName: 'issue_comment', payload: {issue: {number: 1, pull_request: {}}, comment: event},
  }});
  return {state, deliver};
}

test('request records require the pinned account and valid immutable metadata', () => {
  const r = request();
  assert.equal(requests([record(10, r)]).length, 1);
  for (const user of [{...service, id: 1}, {...service, type: 'Bot'}, bot]) {
    assert.deepEqual(requests([{...record(10, r), user}]), []);
  }
  for (const extra of [{id: 'unknown'}, {head: 'main'}, {branch: 'feature/x'}, {checkId: 0}]) {
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

test('conflicting records and missing protocol markers are rejected', () => {
  const r = request();
  const message = reply(20, r);
  message.body += reply(21, r, 'FAIL').body.split('SEMANTIC_REVIEW\n')[1];
  assert.equal(parseResult(message, r, repo), undefined);
  assert.equal(parseResult(comment(20, 'PASS'), r, repo), undefined);
});

test('publishes an exact-version result without requiring the live main SHA', async () => {
  const r = request();
  const {state, deliver} = harness(r);
  state.comments.push(reply(20, r, 'FAIL'));
  await deliver(state.comments.at(-1));
  assert.equal(state.check.conclusion, 'failure');
  assert.match(state.check.output.summary, new RegExp(r.target));
  assert.match(state.check.details_url, /issuecomment-20$/);
});

test('late old PASS cannot overwrite newer FAIL when only target changed', async () => {
  const old = request();
  const current = request(2, {target: 'd'.repeat(40)});
  const {state, deliver} = harness(current);
  state.comments = [record(10, old), record(30, current), reply(40, current, 'FAIL'), reply(50, old)];
  await deliver(state.comments[2]);
  await deliver(state.comments[3]);
  assert.equal(state.check.conclusion, 'failure');
  assert.match(state.check.output.summary, new RegExp(current.id));
  assert.match(state.check.details_url, /issuecomment-40$/);
});

test('old reply cannot temporarily approve an awaiting newer request', async () => {
  const old = request();
  const current = request(2);
  const {state, deliver} = harness(current);
  state.comments = [record(10, old), record(30, current), reply(40, old)];
  await deliver(state.comments.at(-1));
  assert.equal(state.check.conclusion, 'neutral');
  assert.equal(state.updates.length, 0);
});

test('same-version explicit retry requires its own request ID', async () => {
  const old = request();
  const current = request(2);
  const {state, deliver} = harness(current);
  state.comments = [record(10, old), record(30, current), reply(40, old), reply(50, current, 'FAIL')];
  await deliver(state.comments.at(-1));
  assert.equal(state.check.conclusion, 'failure');
  assert.match(state.check.details_url, /issuecomment-50$/);
});

test('new head and current check identity are enforced', async () => {
  for (const extra of [{head_sha: 'd'.repeat(40)}, {external_id: identity(1, request(2))},
    {app: {slug: 'untrusted'}}, {name: 'Different check'}]) {
    const r = request();
    const {state, deliver} = harness(r);
    Object.assign(state.check, extra);
    state.comments.push(reply(20, r));
    await deliver(state.comments.at(-1));
    assert.equal(state.updates.length, 0);
  }
});

test('editing a published PASS into invalid text revokes the green check', async () => {
  for (const body of ['Cannot verify the revisions.', 'SEMANTIC_REVIEW\nINCONCLUSIVE']) {
    const r = request();
    const {state, deliver} = harness(r);
    const result = reply(20, r);
    state.comments.push(result);
    await deliver(result);
    assert.equal(state.check.conclusion, 'success');
    result.body = body;
    await deliver(result);
    assert.equal(state.check.conclusion, 'neutral');
  }
});

test('deleting the published result does not fall back to an earlier PASS', async () => {
  const r = request();
  const {state, deliver} = harness(r);
  const removed = reply(30, r, 'FAIL');
  state.comments.push(reply(20, r), removed);
  await deliver(removed);
  state.comments = state.comments.filter(c => c.id !== removed.id);
  await deliver(removed);
  assert.equal(state.check.conclusion, 'neutral');
});

test('a newer malformed reply for the current request invalidates an older PASS', async () => {
  const r = request();
  const {state, deliver} = harness(r);
  state.comments.push(reply(20, r));
  await deliver(state.comments.at(-1));
  const invalid = reply(30, r);
  invalid.body += reply(31, r, 'FAIL').body.split('SEMANTIC_REVIEW\n')[1];
  state.comments.push(invalid);
  await deliver(invalid);
  assert.equal(state.check.conclusion, 'neutral');
  assert.match(state.check.details_url, /issuecomment-30$/);
});

test('new valid reply can supersede an invalidated source', async () => {
  const r = request();
  const {state, deliver} = harness(r);
  const first = reply(20, r);
  state.comments.push(first);
  await deliver(first);
  first.body = 'Cannot verify.';
  state.comments.push(reply(30, r, 'FAIL'));
  await deliver(first);
  assert.equal(state.check.conclusion, 'failure');
});

test('a bot-looking user cannot publish or revoke results', async () => {
  const r = request();
  const {state, deliver} = harness(r);
  const forged = {...reply(20, r), user: {...bot, id: 123}};
  state.comments.push(forged);
  await deliver(forged);
  assert.equal(state.updates.length, 0);
});

test('all three real histories produce the same protocol without ground-truth hints', () => {
  assert.equal(cases.length, 3);
  for (const fixture of cases) {
    const r = request(1, fixture);
    const body = command(r);
    for (const value of [r.id, r.head, r.target, r.mergeBase]) assert.ok(body.includes(value));
    assert.ok(body.startsWith('@coderabbitai\n'));
    assert.notEqual(r.mergeBase, r.target);
    assert.doesNotMatch(body, /llm_build_stats|routed_output_is_global|get_steady_clock_now_in_seconds/);
    assert.equal(parseResult(reply(20, r, 'FAIL'), r, repo).verdict, 'FAIL');
  }
});

test('privileged jobs run trusted code and serialize request switches with publication', () => {
  const workflow = readFileSync(join(__dirname, '../workflows/semantic-review.yml'), 'utf8');
  assert.doesNotMatch(workflow, /pull_request_target:|pull_request:/);
  assert.match(workflow, /cron: '23 \*\/2 \* \* \*'/);
  assert.equal(workflow.match(/concurrency:\n      group: semantic-review-state\n      cancel-in-progress: false\n      queue: max/g).length, 2);
  assert.equal(workflow.match(/ref: \$\{\{ github.event.repository.default_branch \}\}/g).length, 2);
  assert.match(workflow, /types: \[created, edited, deleted\]/);
  assert.equal(workflow.match(/secrets\./g).length, 1);
  assert.doesNotMatch(workflow.split('  publish:')[1], /SEMANTIC_COMMAND_TOKEN|issues: write/);
});
