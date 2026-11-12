// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const test = require('node:test');
const assert = require('node:assert/strict');
const {MARKER, bodyFor, publish, validate} = require('./pyrefly_report');

const head = 'a'.repeat(40);
const finding = {path: 'tensorrt_llm/module.py', line: 12, name: 'unknown-name',
  message: 'Unknown name `example`'};

function harness({resolved = false, existing = true, diagnostics = [finding], error = null} = {}) {
  const report = {pr_number: 42, base_ref: 'release/test', base_sha: 'b'.repeat(40),
    head_sha: head, diagnostics, error};
  const run = {id: 123, run_attempt: 1, event: 'pull_request',
    path: '.github/workflows/pyrefly-diff.yml', head_sha: head, head_branch: 'feature',
    repository: {full_name: 'NVIDIA/TensorRT-LLM'},
    head_repository: {full_name: 'contributor/TensorRT-LLM'}, pull_requests: []};
  const pr = {state: 'open', base: {ref: report.base_ref},
    head: {sha: head, repo: {full_name: run.head_repository.full_name}}};
  // The author is what the GraphQL API returns for a comment made with the Actions token.
  const thread = {id: 'thread-1', isResolved: resolved, comments: {nodes: [{fullDatabaseId: '99',
    body: `${MARKER}\nold report`, author: {__typename: 'Bot', login: 'github-actions'}}]}};
  const state = {report, run, pr, thread, writes: [], existing,
    files: [{filename: 'tensorrt_llm/module.py', status: 'modified'}]};
  const github = {
    rest: {pulls: {
      get: async () => ({data: state.pr}),
      listFiles: async () => ({data: state.files}),
      createReviewComment: async args => state.writes.push(['create', args]),
      updateReviewComment: async args => state.writes.push(['update', args]),
    }},
    graphql: async (query, variables) => {
      if (!query.startsWith('query')) {
        state.writes.push([query.includes('unresolveReviewThread') ? 'unresolve' : 'resolve', variables]);
        return {};
      }
      return {repository: {pullRequest: {reviewThreads: {
        nodes: state.existing ? [state.thread] : [], pageInfo: {hasNextPage: false},
      }}}};
    },
  };
  state.publish = () => publish({github, report, context: {
    repo: {owner: 'NVIDIA', repo: 'TensorRT-LLM'}, payload: {workflow_run: run},
  }, core: {info() {}, warning() {}}});
  return state;
}

test('fork run with an empty pull_requests list can create one file-level thread', async () => {
  const state = harness({existing: false});
  await state.publish();
  assert.equal(state.writes.length, 1);
  const [operation, args] = state.writes[0];
  assert.equal(operation, 'create');
  assert.equal(args.subject_type, 'file');
  assert.equal(args.commit_id, head);
  assert.equal(args.path, 'tensorrt_llm/module.py');
  assert.ok(args.body.includes('release/test'));
  assert.ok(args.body.includes('unknown-name'));
  assert.ok(args.body.includes('resolve this conversation'));
});

test('each check with findings reopens a manually resolved thread', async () => {
  const state = harness({resolved: true});
  await state.publish();
  assert.deepEqual(state.writes.map(item => item[0]), ['update', 'unresolve']);
});

test('open thread is updated without adding another comment', async () => {
  const state = harness();
  await state.publish();
  assert.deepEqual(state.writes.map(item => item[0]), ['update']);
});

test('64-bit comment IDs are preserved when updating the existing report', async () => {
  const state = harness();
  state.thread.comments.nodes[0].fullDatabaseId = '9007199254740993';
  await state.publish();
  assert.equal(state.writes[0][1].comment_id, '9007199254740993');
});

test('clean comparison updates and resolves an existing thread', async () => {
  const state = harness({diagnostics: []});
  await state.publish();
  assert.deepEqual(state.writes.map(item => item[0]), ['update', 'resolve']);
  assert.ok(state.writes[0][1].body.includes('no new diagnostics'));
  assert.ok(!state.writes[0][1].body.includes('resolve this conversation'));
});

test('clean comparison does not create a thread', async () => {
  const state = harness({existing: false, diagnostics: []});
  await state.publish();
  assert.deepEqual(state.writes, []);
});

test('incomplete check does not create a thread', async () => {
  const state = harness({existing: false, diagnostics: [], error: 'Scanner failed'});
  await state.publish();
  assert.deepEqual(state.writes, []);
});

test('incomplete check never changes whether an existing thread is resolved', async () => {
  for (const resolved of [false, true]) {
    const state = harness({resolved, diagnostics: [], error: 'Scanner failed'});
    await state.publish();
    assert.deepEqual(state.writes.map(item => item[0]), ['update']);
    assert.ok(state.writes[0][1].body.includes('check incomplete'));
    assert.ok(state.writes[0][1].body.includes('Scanner failed'));
  }
});

test('deletion-only PR still gets a file-level thread', async () => {
  const state = harness({existing: false});
  state.files = [{filename: 'deleted.py', status: 'removed'}];
  await state.publish();
  assert.equal(state.writes[0][1].path, 'deleted.py');
  assert.equal(state.writes[0][1].subject_type, 'file');
});

test('force-push, retargeting, closing, and mismatched head repositories discard old results', async () => {
  for (const change of [
    state => {state.pr.head.sha = 'c'.repeat(40);},
    state => {state.pr.base.ref = 'main';},
    state => {state.pr.state = 'closed';},
    state => {state.pr.head.repo.full_name = 'another/TensorRT-LLM';},
  ]) {
    const state = harness();
    change(state);
    await state.publish();
    assert.deepEqual(state.writes, []);
  }
});

test('report cannot claim another source head or workflow', async () => {
  for (const change of [
    state => {state.run.head_sha = 'c'.repeat(40);},
    state => {state.run.path = '.github/workflows/another.yml';},
    state => {state.run.event = 'push';},
  ]) {
    const state = harness();
    change(state);
    await assert.rejects(state.publish, /does not match/);
    assert.deepEqual(state.writes, []);
  }
});

test('only a marked comment from the Actions bot is reused as the report', async () => {
  for (const author of [
    {__typename: 'User', login: 'human'},
    {__typename: 'User', login: 'github-actions'},
    {__typename: 'Bot', login: 'another-app'},
    null,
  ]) {
    const state = harness();
    state.thread.comments.nodes[0].author = author;
    await state.publish();
    assert.deepEqual(state.writes.map(item => item[0]), ['create']);
  }
});

test('malformed reports are rejected before any API call', async () => {
  for (const change of [
    report => {report.pr_number = '42';},
    report => {report.head_sha = 'main';},
    report => {report.base_sha = '';},
    report => {report.error = '';},
    report => {report.diagnostics = [{...finding, line: 0}];},
    report => {report.diagnostics = [{...finding, message: undefined}];},
  ]) {
    const state = harness();
    change(state.report);
    await assert.rejects(state.publish, /Invalid Pyrefly report/);
    assert.deepEqual(state.writes, []);
  }
});

test('incomplete report may lack the target branch SHA', () => {
  const state = harness({diagnostics: [], error: 'Could not merge'});
  state.report.base_sha = '';
  validate(state.report);
  assert.ok(bodyFor(state.report, state.run, 'https://github.com').includes('(unavailable)'));
});

test('rendered diagnostics escape markup and keep long reports within the comment limit', () => {
  const state = harness();
  state.report.diagnostics = Array.from({length: 200}, () => ({...finding,
    message: '<script>example()</script>\n' + 'x'.repeat(2000)}));
  const body = bodyFor(state.report, state.run, 'https://github.com');
  assert.ok(body.includes('&lt;script&gt;'));
  assert.ok(!body.includes('<script>'));
  assert.ok(body.includes('of 200 diagnostics'));
  assert.ok(body.length < 65536);
});
