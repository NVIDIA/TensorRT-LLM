// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const test = require('node:test');
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {join} = require('node:path');
const {NAME, identity, requests, parseResult, command, awaiting, reviewState, publish} = require('./semantic_review');
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
  const state = {comments: [record(10, r)], updates: [], creates: [], summaries: [], reads: 0, refs: [], updateFailures: new Set(),
    check: {id: r.checkId, name: NAME, head_sha: r.head, external_id: identity(1, r),
      app: {slug: 'github-actions'}, status: 'in_progress', conclusion: null,
      output: awaiting(r), details_url: 'https://github.com/NVIDIA/TensorRT-LLM/pull/1'}};
  state.checks = [state.check];
  const github = {
    paginate: async (method, args) => {
      if (method === github.rest.issues.listComments) {
        state.reads += 1;
        return structuredClone(state.comments);
      }
      const {data} = await method(args);
      return data.check_runs;
    },
    rest: {issues: {listComments() {}}, checks: {
      listForRef: async ({ref, check_name: name, filter}) => {
        assert.equal(filter, 'all');
        state.refs.push(ref);
        return {data: {check_runs: structuredClone(state.checks.filter(check =>
          check.head_sha === ref && check.name === name))}};
      },
      create: async args => {
        assert.equal(Object.hasOwn(args, 'conclusion'), false);
        assert.equal(Object.hasOwn(args, 'completed_at'), false);
        state.creates.push(args);
        const created = {...args, id: Math.max(...state.checks.map(check => check.id)) + 1,
          app: {slug: 'github-actions'}, conclusion: null};
        created.details_url = `https://github.com/NVIDIA/TensorRT-LLM/runs/${created.id}`;
        state.checks.push(created);
        state.check = created;
        return {data: structuredClone(created)};
      },
      update: async update => {
        assert.notEqual(update.conclusion, null);
        assert.notEqual(update.completed_at, null);
        if (state.updateFailures.delete(update.check_run_id)) {
          throw Object.assign(new Error('Check update failed'), {status: 503});
        }
        state.updates.push(update);
        const check = state.checks.find(item => item.id === update.check_run_id);
        const previous = {status: check.status, conclusion: check.conclusion};
        Object.assign(check, update);
        if (previous.status === 'completed' && update.status === 'in_progress') {
          Object.assign(check, previous);
        }
        check.details_url = `https://github.com/NVIDIA/TensorRT-LLM/runs/${check.id}`;
        if (update.conclusion !== 'cancelled') state.check = check;
        return {data: structuredClone(check)};
      },
    }},
  };
  const core = {summary: {addRaw(text) {state.summaries.push(text); return this;}, async write() {}},
    setFailed() {throw new Error('AI verdict must not fail the orchestration job');}};
  const deliver = async event => publish({github, core, context: {
    repo, eventName: 'issue_comment', payload: {issue: {number: 1, pull_request: {}}, comment: event},
  }});
  return {state, deliver,
    inspect: extra => reviewState({github, repo, number: 1, ...extra}),
    repair: (comments, extra) => publish({github, core, context: {repo}, number: 1, comments, ...extra})};
}

test('request records require the pinned account and valid immutable metadata', () => {
  const r = request();
  assert.equal(requests([record(10, r)]).length, 1);
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
    assert.equal(state.check.conclusion, 'failure');
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

test('publishes an exact-version result without requiring the live main SHA', async () => {
  const r = request();
  const {state, deliver} = harness(r);
  state.comments.push(reply(20, r, 'FAIL'));
  await deliver(state.comments.at(-1));
  assert.equal(state.check.conclusion, 'failure');
  assert.match(state.check.output.summary, new RegExp(r.target));
  assert.match(state.check.output.summary, /#issuecomment-20/);
  assert.match(state.check.output.summary, /<!-- semantic-review-source:20 -->/);
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
  assert.match(state.check.output.summary, /#issuecomment-40/);
});

test('old reply cannot temporarily approve an awaiting newer request', async () => {
  const old = request();
  const current = request(2);
  const {state, deliver} = harness(current);
  state.comments = [record(10, old), record(30, current), reply(40, old)];
  await deliver(state.comments.at(-1));
  assert.equal(state.check.status, 'in_progress');
  assert.equal(state.check.conclusion, null);
  assert.equal(state.updates.length, 0);
});

test('same-version explicit retry requires its own request ID', async () => {
  const old = request();
  const current = request(2);
  const {state, deliver} = harness(current);
  state.comments = [record(10, old), record(30, current), reply(40, old), reply(50, current, 'FAIL')];
  await deliver(state.comments.at(-1));
  assert.equal(state.check.conclusion, 'failure');
  assert.match(state.check.output.summary, /#issuecomment-50/);
});

test('new head and current check identity are enforced', async () => {
  for (const extra of [{head_sha: 'd'.repeat(40)}, {external_id: identity(1, request(2))},
    {app: {slug: 'untrusted'}}, {name: 'Different check'}]) {
    const r = request();
    const {state, deliver} = harness(r);
    Object.assign(state.check, extra);
    state.comments.push(reply(20, r));
    const current = await deliver(state.comments.at(-1));
    assert.equal(current.result, undefined);
    assert.equal(state.creates.length, 0);
    assert.equal(state.updates.every(update => update.conclusion === 'cancelled'), true);
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
    assert.equal(state.check.status, 'in_progress');
    assert.equal(state.check.conclusion, null);
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
  assert.equal(state.check.status, 'in_progress');
  assert.equal(state.check.conclusion, null);
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
  assert.equal(state.check.status, 'in_progress');
  assert.equal(state.check.conclusion, null);
  assert.match(state.check.output.summary, /#issuecomment-30/);
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

test('each valid verdict completes the check, including explicit or downgraded INCONCLUSIVE', async () => {
  for (const [verdict, evidence, conclusion] of [['PASS', true, 'success'],
    ['FAIL', true, 'failure'], ['INCONCLUSIVE', false, 'neutral'],
    ['PASS', false, 'neutral'], ['FAIL', false, 'neutral']]) {
    const r = request();
    const {state, deliver} = harness(r);
    const message = reply(20, r, verdict, evidence);
    state.comments.push(message);
    const resolved = await deliver(message);
    assert.equal(state.check.status, 'completed');
    assert.equal(state.check.conclusion, conclusion);
    assert.ok(resolved.result);
    assert.equal(resolved.result.verdict, conclusion === 'neutral' ? 'INCONCLUSIVE' : verdict);
  }
});

test('no valid reply means waiting, even when an existing check is completed neutral', async () => {
  const {state, inspect, repair} = harness();
  state.check.status = 'completed';
  state.check.conclusion = 'neutral';
  state.check.completed_at = '2026-09-28T00:00:00Z';
  const current = await inspect();
  assert.equal(current.result, undefined);
  assert.equal(current.update.status, 'in_progress');
  assert.equal(current.create, true);
  assert.equal(current.update.head_sha, current.request.head);
  assert.equal(current.update.external_id, identity(1, current.request));
  assert.equal(Object.hasOwn(current.update, 'conclusion'), false);
  assert.equal(Object.hasOwn(current.update, 'completed_at'), false);
  assert.equal(state.updates.length, 0);
  await repair();
  assert.equal(state.check.status, 'in_progress');
  assert.equal(state.check.conclusion, null);
  assert.equal((await inspect()).update, undefined);
});

test('wrong UUID or revision replies are ignored without invalidating an earlier valid result', async () => {
  for (const extra of [{id: id(2)}, {head: 'd'.repeat(40)}, {target: 'd'.repeat(40)},
    {mergeBase: 'd'.repeat(40)}]) {
    const r = request();
    const {state, deliver} = harness(r);
    const first = reply(20, r);
    state.comments.push(first);
    await deliver(first);
    const wrong = reply(30, {...r, ...extra}, 'FAIL');
    state.comments.push(wrong);
    const current = await deliver(wrong);
    assert.equal(current.result.comment.id, first.id);
    assert.equal(current.update, undefined);
    assert.equal(state.check.conclusion, 'success');
    assert.equal(state.updates.length, 1);
  }
});

test('malformed current-request replies wait instead of manufacturing an INCONCLUSIVE result', async () => {
  const r = request();
  for (const body of [`Request ${r.id}: still investigating.`,
    reply(30, r).body.replace('verdict=PASS', 'verdict=UNKNOWN'),
    reply(30, r).body.replace('SEMANTIC_REVIEW', 'Missing heading'),
    reply(30, r).body + `SEMANTIC_RESULT request_id=${r.id} malformed\n`]) {
    const {state, deliver, inspect} = harness(r);
    const first = reply(20, r);
    state.comments.push(first);
    await deliver(first);
    const invalid = comment(30, body);
    state.comments.push(invalid);
    const current = await deliver(invalid);
    assert.equal(current.result, undefined);
    assert.equal(state.check.status, 'in_progress');
    assert.equal(state.check.conclusion, null);
    assert.match(state.check.output.summary, /#issuecomment-30/);
    assert.equal((await inspect()).update, undefined);
    state.comments = state.comments.filter(item => item.id !== invalid.id);
    const deleted = await deliver(invalid);
    assert.equal(deleted.result, undefined);
    assert.equal(deleted.update, undefined);
  }
});

test('editing the published source to a mismatched revision returns to waiting', async () => {
  const r = request();
  const {state, deliver} = harness(r);
  const message = reply(20, r);
  state.comments.push(message);
  await deliver(message);
  message.body = reply(20, {...r, target: 'd'.repeat(40)}).body;
  const current = await deliver(message);
  assert.equal(current.result, undefined);
  assert.equal(state.check.status, 'in_progress');
  assert.equal(state.check.conclusion, null);
});

test('scheduled repair shares read-only state computation and avoids redundant writes', async () => {
  const r = request();
  const {state, inspect, repair} = harness(r);
  const comments = [record(10, r), reply(20, r, 'FAIL')];
  const current = await inspect({comments});
  assert.equal(current.result.verdict, 'FAIL');
  assert.equal(current.update.conclusion, 'failure');
  assert.equal(state.reads, 0);
  assert.equal(state.updates.length, 0);
  await repair(comments);
  const again = await repair(comments);
  assert.equal(again.result.verdict, 'FAIL');
  assert.equal(again.update, undefined);
  assert.equal(state.updates.length, 1);
  assert.equal(state.summaries.length, 1);
  assert.equal(state.reads, 0);
  state.check.output.summary = 'Stale output';
  await repair(comments);
  assert.equal(state.updates.length, 2);
  state.check.details_url = 'https://github.com/NVIDIA/TensorRT-LLM/pull/1';
  await repair(comments);
  assert.equal(state.updates.length, 2);
});

test('the summary source marker survives platform details URL rewriting and blocks stale fallback', async () => {
  const r = request();
  const {state, deliver, inspect} = harness(r);
  const latest = reply(30, r, 'FAIL');
  state.comments.push(reply(20, r), latest);
  await deliver(latest);
  assert.equal(state.check.details_url, `https://github.com/NVIDIA/TensorRT-LLM/runs/${r.checkId}`);
  assert.match(state.check.output.summary, /<!-- semantic-review-source:30 -->/);
  assert.equal((await inspect()).update, undefined);
  state.comments = state.comments.filter(item => item.id !== latest.id);
  const missing = await deliver(latest);
  assert.equal(missing.result, undefined);
  assert.match(state.check.output.summary, /<!-- semantic-review-source:30 -->/);
  assert.equal((await inspect()).update, undefined);
  assert.equal(state.updates.every(update => !Object.hasOwn(update, 'details_url')), true);
});

test('an invalidated completed verdict gets a replacement check which later receives the valid reply', async () => {
  for (const verdict of ['PASS', 'FAIL', 'INCONCLUSIVE']) {
    const r = request();
    const {state, deliver, inspect} = harness(r);
    const message = reply(20, r, verdict);
    state.comments.push(message);
    await deliver(message);
    const original = state.check;
    message.body = 'The analysis is being corrected.';
    const invalid = await deliver(message);
    const replacement = state.check;
    assert.equal(invalid.create, true);
    assert.notEqual(replacement.id, original.id);
    assert.equal(original.status, 'completed');
    assert.equal(replacement.status, 'in_progress');
    assert.equal(replacement.conclusion, null);
    assert.equal(replacement.external_id, original.external_id);
    assert.equal(replacement.head_sha, original.head_sha);
    assert.equal((await inspect()).request.checkId, original.id);
    assert.equal((await inspect()).check.id, replacement.id);
    assert.equal((await inspect()).update, undefined);
    message.body = reply(20, r, 'FAIL').body;
    const repaired = await deliver(message);
    assert.equal(repaired.create, false);
    assert.equal(state.check.id, replacement.id);
    assert.equal(state.check.conclusion, 'failure');
    assert.equal(state.creates.length, 1);
    assert.equal(state.updates.at(-1).check_run_id, replacement.id);
  }
});

test('only the newest native check for the exact request identity receives publication', async () => {
  const r = request();
  const {state, deliver, inspect} = harness(r);
  const replacement = {...structuredClone(state.check), id: 101};
  state.checks.push(replacement,
    {...structuredClone(replacement), id: 102, external_id: identity(1, request(2))},
    {...structuredClone(replacement), id: 103, external_id: identity(2, r)},
    {...structuredClone(replacement), id: 104, app: {slug: 'untrusted'}},
    {...structuredClone(replacement), id: 105, head_sha: 'd'.repeat(40)},
    {...structuredClone(replacement), id: 106, name: 'Different check'});
  state.comments.push(reply(20, r, 'FAIL'));
  assert.equal((await inspect()).check.id, replacement.id);
  await deliver(state.comments.at(-1));
  assert.equal(state.updates[0].check_run_id, replacement.id);
  assert.equal(state.checks.find(check => check.id === r.checkId).conclusion, 'cancelled');
  assert.equal(state.checks.find(check => check.id === 102).conclusion, 'cancelled');
  assert.equal(state.checks.filter(check => check.conclusion === 'failure').length, 1);
});

test('without a current matching check, known refs permit only scoped cleanup', async () => {
  const {state, inspect} = harness();
  assert.equal(await inspect({comments: []}), undefined);
  assert.equal(state.reads, 0);
  state.check.external_id = identity(1, request(2));
  const current = await inspect();
  assert.equal(current.check, undefined);
  assert.equal(current.result, undefined);
  assert.equal(current.update, undefined);
  assert.deepEqual(current.cleanup.map(check => check.id), [state.check.id]);
  assert.equal(state.updates.length, 0);
});

test('a completed current result still cleans superseded and unrecorded pending checks idempotently', async () => {
  const r = request();
  const {state, deliver, inspect, repair} = harness(r);
  state.comments.push(reply(20, r));
  await deliver(state.comments.at(-1));
  for (const number of [98, 101]) state.checks.push({...structuredClone(state.check),
    id: number, external_id: identity(1, request(number)), status: 'in_progress', conclusion: null});
  const current = await inspect();
  assert.equal(current.result.verdict, 'PASS');
  assert.equal(current.update, undefined);
  assert.deepEqual(current.cleanup.map(check => check.id), [98, 101]);
  await repair();
  assert.equal(state.check.conclusion, 'success');
  for (const number of [98, 101]) {
    const inactive = state.checks.find(check => check.id === number);
    assert.equal(inactive.conclusion, 'cancelled');
    assert.match(inactive.output.summary, /does not assign an AI verdict/);
  }
  const writes = state.updates.length;
  assert.deepEqual((await repair()).cleanup, []);
  assert.equal(state.updates.length, writes);
});

test('failed cancellation is retried without republishing the valid result or repeating completed cleanup', async () => {
  const r = request();
  const {state, deliver, inspect, repair} = harness(r);
  state.comments.push(reply(20, r, 'FAIL'));
  await deliver(state.comments.at(-1));
  for (const number of [101, 102]) state.checks.push({...structuredClone(state.check),
    id: number, external_id: identity(1, request(number)), status: 'in_progress', conclusion: null});
  state.updateFailures.add(102);
  await assert.rejects(repair(), {status: 503});
  assert.equal(state.check.conclusion, 'failure');
  assert.deepEqual((await inspect()).cleanup.map(check => check.id), [102]);
  await repair();
  assert.deepEqual((await inspect()).cleanup, []);
  assert.equal(state.updates.filter(update => update.check_run_id === r.checkId).length, 1);
  assert.equal(state.updates.filter(update => update.check_run_id === 101).length, 1);
  assert.equal(state.updates.filter(update => update.check_run_id === 102).length, 1);
});

test('cleanup excludes the selected check, completed checks and other PRs, apps, names or heads', async () => {
  const r = request();
  const {state, inspect, repair} = harness(r);
  const orphan = {...structuredClone(state.check), id: 101, external_id: identity(1, request(2))};
  const excluded = [
    {...orphan, id: 102, external_id: identity(2, request(2))},
    {...orphan, id: 103, app: {slug: 'untrusted'}},
    {...orphan, id: 104, name: 'Another check'},
    {...orphan, id: 105, head_sha: 'd'.repeat(40)},
    {...orphan, id: 106, status: 'completed', conclusion: 'success'},
  ];
  state.checks.push(orphan, ...excluded);
  assert.deepEqual((await inspect()).cleanup.map(check => check.id), [101]);
  await repair();
  assert.deepEqual(state.updates.map(update => update.check_run_id), [101]);
  assert.equal(state.check.status, 'in_progress');
  assert.equal(state.creates.length, 0);
  assert.equal(excluded.slice(0, 4).every(check => check.status === 'in_progress'), true);
  assert.equal(excluded[4].conclusion, 'success');
});

test('an explicit head permits orphan cleanup without any trusted request record', async () => {
  const r = request();
  const {state, inspect, repair} = harness(r);
  const current = await inspect({comments: [], head: r.head});
  assert.equal(current.request, undefined);
  assert.equal(current.check, undefined);
  assert.equal(current.result, undefined);
  assert.equal(current.update, undefined);
  assert.deepEqual(current.cleanup.map(check => check.id), [r.checkId]);
  await repair([], {head: r.head});
  assert.equal(state.check.conclusion, 'cancelled');
  assert.equal(state.creates.length, 0);
  assert.deepEqual((await inspect({comments: [], head: r.head})).cleanup, []);
});

test('cleanup reads only the recorded and explicit heads, deduplicating identical refs', async () => {
  const r = request();
  const {state, inspect, repair} = harness(r);
  await inspect({head: r.head});
  assert.deepEqual(state.refs, [r.head]);
  state.refs = [];
  const newHead = 'd'.repeat(40);
  state.checks.push({...structuredClone(state.check), id: 101, head_sha: newHead},
    {...structuredClone(state.check), id: 102, head_sha: 'e'.repeat(40)});
  const current = await repair(undefined, {head: newHead});
  assert.deepEqual(state.refs, [r.head, newHead]);
  assert.deepEqual(current.cleanup.map(check => check.id), [101]);
  assert.equal(state.checks.find(check => check.id === 101).conclusion, 'cancelled');
  assert.equal(state.checks.find(check => check.id === 102).status, 'in_progress');
  assert.equal(state.checks.find(check => check.id === r.checkId).status, 'in_progress');
});

test('real divergent, integrated and repaired histories use the same result protocol', () => {
  assert.equal(cases.length, 9);
  for (const fixture of cases) {
    const r = request(1, fixture);
    const body = command(r);
    for (const value of [r.id, r.head, r.target, r.mergeBase]) assert.ok(body.includes(value));
    assert.ok(body.startsWith('@coderabbitai\n'));
    assert.doesNotMatch(body, /llm_build_stats|routed_output_is_global|get_steady_clock_now_in_seconds/);
    assert.equal(parseResult(reply(20, r, 'FAIL'), r, repo).verdict, 'FAIL');
  }
});

test('privileged jobs run trusted code and serialize request switches with publication', () => {
  const workflow = readFileSync(join(__dirname, '../workflows/semantic-review.yml'), 'utf8');
  assert.doesNotMatch(workflow, /pull_request_target:|pull_request:/);
  assert.match(workflow, /cron: '23 \*\/2 \* \* \*'/);
  assert.match(workflow, /group: semantic-review-pr-\$\{\{ matrix.number \}\}/);
  assert.match(workflow, /group: semantic-review-pr-\$\{\{ github.event.issue.number \}\}/);
  assert.equal(workflow.match(/      cancel-in-progress: false\n      queue: max/g).length, 2);
  assert.equal(workflow.match(/ref: \$\{\{ github.event.repository.default_branch \}\}/g).length, 3);
  assert.match(workflow, /types: \[created, edited, deleted\]/);
  assert.equal(workflow.match(/secrets\./g).length, 1);
  assert.doesNotMatch(workflow.split('  publish:')[1], /SEMANTIC_COMMAND_TOKEN|issues: write/);
});
