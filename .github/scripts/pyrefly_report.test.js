// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const test = require('node:test');
const assert = require('node:assert/strict');
const {MARKER, bodyFor, publish} = require('./pyrefly_report');

const head = 'a'.repeat(40);
const finding = {path: 'tensorrt_llm/module.py', line: 12, name: 'unknown-name',
  description: 'Unknown name `example`'};

function harness({resolved = false, existing = true, errors = [finding], error = null} = {}) {
  const report = {pr_number: 42, base_ref: 'release/test', base_sha: 'b'.repeat(40),
    head_sha: head, errors, error};
  const run = {id: 123, run_attempt: 1, event: 'pull_request',
    path: '.github/workflows/pyrefly-diff.yml', head_sha: head, head_branch: 'feature',
    repository: {full_name: 'NVIDIA/TensorRT-LLM'},
    head_repository: {full_name: 'contributor/TensorRT-LLM'}, pull_requests: []};
  const pr = {state: 'open', base: {ref: report.base_ref},
    head: {sha: head, repo: {full_name: run.head_repository.full_name}}};
  const thread = {id: 'thread-1', isResolved: resolved, comments: {nodes: [{fullDatabaseId: '99',
    body: `${MARKER}\nold report`, author: {login: 'github-actions[bot]'}}]}};
  const state = {report, run, pr, thread, writes: [], existing, queryCount: 0,
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
      state.queryCount++;
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
  assert.ok(args.body.includes('release/test'));
  assert.ok(args.body.includes('unknown-name'));
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
  const state = harness({errors: []});
  await state.publish();
  assert.deepEqual(state.writes.map(item => item[0]), ['update', 'resolve']);
  assert.ok(state.writes[0][1].body.includes('no new diagnostics'));
});

test('clean comparison does not create a thread', async () => {
  const state = harness({existing: false, errors: []});
  await state.publish();
  assert.deepEqual(state.writes, []);
});

test('scanner failure never resolves an existing thread', async () => {
  const state = harness({errors: [], error: 'Scanner failed'});
  await state.publish();
  assert.deepEqual(state.writes.map(item => item[0]), ['update']);
  assert.ok(state.writes[0][1].body.includes('check incomplete'));
});

test('retry can finish a thread state change after the body was already updated', async () => {
  const state = harness({resolved: true});
  state.thread.comments.nodes[0].body += '\n<!-- pyrefly-run:123:1 -->';
  await state.publish();
  assert.deepEqual(state.writes.map(item => item[0]), ['update', 'unresolve']);
});

test('an earlier attempt of the same run cannot overwrite a later attempt', async () => {
  const state = harness();
  state.thread.comments.nodes[0].body += '\n<!-- pyrefly-run:123:2 -->';
  await state.publish();
  assert.deepEqual(state.writes, []);
});

test('deletion-only PR anchors a new thread on the left side', async () => {
  const state = harness({existing: false});
  state.files = [{filename: 'deleted.py', status: 'removed'}];
  await state.publish();
  assert.equal(state.writes[0][1].side, 'LEFT');
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

test('older completed runs cannot overwrite a newer report for the same head', async () => {
  const state = harness();
  state.thread.comments.nodes[0].body += '\n<!-- pyrefly-run:124:1 -->';
  await state.publish();
  assert.deepEqual(state.writes, []);
});

test('human comments with a copied marker are not edited', async () => {
  const state = harness();
  state.thread.comments.nodes[0].author.login = 'human';
  await state.publish();
  assert.deepEqual(state.writes.map(item => item[0]), ['create']);
});

test('rendered diagnostics escape markup and keep long reports within the comment limit', () => {
  const state = harness();
  state.report.errors = Array.from({length: 200}, () => ({...finding,
    description: '<script>example()</script>\n' + 'x'.repeat(2000)}));
  const body = bodyFor(state.report, state.run, 'https://github.com');
  assert.ok(body.includes('&lt;script&gt;'));
  assert.ok(!body.includes('<script>'));
  assert.ok(body.includes('of 200 diagnostics'));
  assert.ok(body.length < 65536);
});
