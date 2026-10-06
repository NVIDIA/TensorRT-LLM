// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

const assert = require('node:assert/strict');
const { readdirSync } = require('node:fs');
const { tmpdir } = require('node:os');
const test = require('node:test');
const { findCursorArtifact, parseCursor, restoreCursor } = require('./semantic_review_cursor');

const ARTIFACT = 'semantic-review-cursor';
const NEXT_ARTIFACT = 'semantic-review-cursor-next';
const archives = {
  valid: 'UEsDBBQAAAAAAOlbPF0CF26zEQAAABEAAAALAAAAY3Vyc29yLmpzb257Imxhc3RfcHIiOjE5NjIxfVBLAQIUAxQAAAAAAOlbPF0CF26zEQAAABEAAAALAAAAAAAAAAAAAACAAQAAAABjdXJzb3IuanNvblBLBQYAAAAAAQABADkAAAA6AAAAAAA=',
  missing: 'UEsDBBQAAAAAAOlbPF0CF26zEQAAABEAAAAKAAAAb3RoZXIuanNvbnsibGFzdF9wciI6MTk2MjF9UEsBAhQDFAAAAAAA6Vs8XQIXbrMRAAAAEQAAAAoAAAAAAAAAAAAAAIABAAAAAG90aGVyLmpzb25QSwUGAAAAAAEAAQA4AAAAOQAAAAAA',
  invalid: 'UEsDBBQAAAAAAOlbPF14VmogBwAAAAcAAAALAAAAY3Vyc29yLmpzb257YnJva2VuUEsBAhQDFAAAAAAA6Vs8XXhWaiAHAAAABwAAAAsAAAAAAAAAAAAAAIABAAAAAGN1cnNvci5qc29uUEsFBgAAAAABAAEAOQAAADAAAAAAAA==',
  schema: 'UEsDBBQAAAAAAOlbPF2FugLXEwAAABMAAAALAAAAY3Vyc29yLmpzb257Imxhc3RfcHIiOiIxOTYyMSJ9UEsBAhQDFAAAAAAA6Vs8XYW6AtcTAAAAEwAAAAsAAAAAAAAAAAAAAIABAAAAAGN1cnNvci5qc29uUEsFBgAAAAABAAEAOQAAADwAAAAAAA==',
};
const zip = name => Uint8Array.from(Buffer.from(archives[name], 'base64')).buffer;
const temporaryCursors = () => readdirSync(tmpdir()).filter(name => name.startsWith('semantic-review-cursor-')).sort();

const repo = { owner: 'NVIDIA', repo: 'TensorRT-LLM' };
const artifact = (id, extra = {}) => ({ id, name: ARTIFACT, expired: false,
  created_at: new Date(1700000000000 + id * 1000).toISOString(),
  workflow_run: { id: id + 1000 }, ...extra });
const trustedRun = (extra = {}) => ({ event: 'schedule',
  path: '.github/workflows/semantic-review.yml',
  repository: { full_name: 'NVIDIA/TensorRT-LLM' },
  head_repository: { full_name: 'NVIDIA/TensorRT-LLM' }, ...extra });

function fixture(pages = [[]], options = {}) {
  const state = { calls: [], rateReads: 0, before: [], after: [],
    remaining: options.remaining ?? 5000 };
  const api = (kind, operation) => async (args) => {
    for (const hook of state.before) await hook();
    state.calls.push({ kind, ...args });
    state.remaining -= 1;
    if (options.errors?.[kind] && (!options.errorName || options.errorName === args.name)) {
      throw options.errors[kind];
    }
    const response = { data: operation(args),
      headers: { 'x-ratelimit-remaining': String(state.remaining) } };
    for (const hook of state.after) await hook(response);
    return response;
  };
  const github = {
    hook: {
      before: (_, callback) => state.before.push(callback),
      after: (_, callback) => state.after.push(callback),
      remove: (_, callback) => {
        state.before = state.before.filter(hook => hook !== callback);
        state.after = state.after.filter(hook => hook !== callback);
      },
    },
    paginate: async (method, args) => {
      assert.equal(method, github.rest.actions.listArtifactsForRepo);
      const all = [];
      for (let page = 1; page <= pages.length; page += 1) {
        const { data } = await method({ ...args, page });
        all.push(...data.artifacts);
      }
      return all;
    },
    rest: {
      rateLimit: { get: async () => {
        state.rateReads += 1;
        return { data: { resources: { core: { remaining: state.remaining } } } };
      } },
      actions: {
        listArtifactsForRepo: api('list', ({ page, name }) =>
          ({ artifacts: pages[page - 1].filter(item => item.name === name) })),
        getWorkflowRun: api('run', ({ run_id: id }) => options.runs?.[id] || trustedRun()),
        downloadArtifact: api('download', () => options.archive ?? zip('valid')),
      },
    },
  };
  const context = { repo, eventName: 'schedule' };
  return { state, find: () => findCursorArtifact({ github, context }),
    restore: () => restoreCursor({ github, context }) };
}

test('manual and publisher invocations return without touching API or cursor files', async () => {
  const github = new Proxy({}, { get() { throw new Error('Unexpected API access'); } });
  for (const eventName of ['workflow_dispatch', 'issue_comment', 'pull_request']) {
    assert.equal(await findCursorArtifact({ github, context: { repo, eventName } }), undefined);
    assert.equal(await restoreCursor({ github, context: { repo, eventName } }), undefined);
  }
});

test('lookup paginates both exact artifact names sequentially and sorts their combined results', async () => {
  const older = artifact(900, { created_at: '2026-09-01T00:00:00Z' });
  const firstTie = artifact(1, { created_at: '2026-09-03T00:00:00Z' });
  const lastTie = artifact(2, { name: NEXT_ARTIFACT, created_at: firstTie.created_at });
  const unrelated = artifact(9999, { name: 'semantic-review-cursor-other',
    created_at: '2026-09-04T00:00:00Z' });
  const f = fixture([[older, firstTie], [lastTie, unrelated]]);
  assert.equal((await f.find()).id, lastTie.id);
  const lists = f.state.calls.filter(call => call.kind === 'list');
  assert.equal(lists.length, 4);
  assert.deepEqual(lists.map(call => call.name), [ARTIFACT, ARTIFACT, NEXT_ARTIFACT, NEXT_ARTIFACT]);
  for (const call of lists) {
    assert.equal(call.per_page, 100);
    assert.equal(call.owner, repo.owner);
    assert.equal(call.repo, repo.repo);
  }
  assert.deepEqual(f.state.calls.filter(call => call.kind === 'run').map(call => call.run_id),
    [lastTie.workflow_run.id]);
});

test('trusted runs accept a workflow ref suffix and case-insensitive repository identities', async () => {
  const selected = artifact(1);
  const f = fixture([[selected]], { runs: { [selected.workflow_run.id]: trustedRun({
    path: '.github/workflows/semantic-review.yml@refs/heads/main',
    repository: { full_name: 'nvidia/tensorrt-llm' },
    head_repository: { full_name: 'NVIDIA/TENSORRT-LLM' },
  }) } });
  assert.equal((await f.find()).id, selected.id);
});

test('manual, publisher, fork and wrong-workflow artifacts cannot supply the cursor', async () => {
  const changes = [
    { event: 'workflow_dispatch' },
    { event: 'issue_comment' },
    { repository: { full_name: 'other/TensorRT-LLM' } },
    { head_repository: { full_name: 'other/TensorRT-LLM' } },
    { path: '.github/workflows/unrelated.yml' },
  ];
  const artifacts = changes.map((_, index) => artifact(index + 2));
  const runs = Object.fromEntries(artifacts.map((item, index) =>
    [item.workflow_run.id, trustedRun(changes[index])]));
  const oldest = artifact(1);
  const f = fixture([[oldest, ...artifacts]], { runs });
  assert.equal((await f.find()).id, oldest.id);
  assert.equal(f.state.calls.filter(call => call.kind === 'run').length, artifacts.length + 1);
  const onlyUntrusted = fixture([artifacts], { runs });
  assert.equal(await onlyUntrusted.find(), undefined);
});

test('the latest trusted artifact being expired fails instead of falling back', async () => {
  const older = artifact(1);
  const expired = artifact(2, { name: NEXT_ARTIFACT, expired: true });
  const f = fixture([[older, expired]]);
  await assert.rejects(f.find(), /artifact has expired/);
  assert.deepEqual(f.state.calls.filter(call => call.kind === 'run').map(call => call.run_id),
    [expired.workflow_run.id]);
  const untrusted = fixture([[older, expired]], {
    runs: { [expired.workflow_run.id]: trustedRun({ event: 'workflow_dispatch' }) },
  });
  assert.equal((await untrusted.find()).id, older.id);
});

test('absent artifacts return no cursor without reading a workflow run', async () => {
  const f = fixture();
  assert.equal(await f.find(), undefined);
  assert.equal(f.state.calls.some(call => call.kind === 'run'), false);
});

test('artifact-list and workflow-run API failures propagate rather than resetting the cursor', async () => {
  for (const kind of ['list', 'run']) {
    for (const status of [403, 404, 429, 500, 503]) {
      const failure = Object.assign(new Error('API unavailable'), { status });
      const f = fixture([[artifact(1), artifact(2)]], { errors: { [kind]: failure } });
      await assert.rejects(f.find(), error => error === failure);
      assert.equal(f.state.calls.filter(call => call.kind === kind).length, 1);
      assert.equal(f.state.before.length + f.state.after.length, 0);
    }
  }
});

test('quota reserve stops lookup before spending the last 1000 requests', async () => {
  for (const remaining of [999, 1000]) {
    const f = fixture([[artifact(1)]], { remaining });
    await assert.rejects(f.find(), { code: 'SEMANTIC_REVIEW_QUOTA' });
    assert.equal(f.state.rateReads, 1);
    assert.deepEqual(f.state.calls, []);
    assert.equal(f.state.before.length + f.state.after.length, 0);
  }
  const during = fixture([[artifact(1)]], { remaining: 1001 });
  await assert.rejects(during.find(), { code: 'SEMANTIC_REVIEW_QUOTA' });
  assert.equal(during.state.remaining, 1000);
  assert.deepEqual(during.state.calls.map(call => call.kind), ['list']);
  const beforeRun = fixture([[artifact(1)]], { remaining: 1002 });
  await assert.rejects(beforeRun.find(), { code: 'SEMANTIC_REVIEW_QUOTA' });
  assert.deepEqual(beforeRun.state.calls.map(call => call.kind), ['list', 'list']);
  const enough = fixture([[artifact(1)]], { remaining: 1003 });
  assert.equal((await enough.find()).id, 1);
  assert.equal(enough.state.remaining, 1000);
});

test('cursor JSON accepts only a positive safe integer last_pr', () => {
  for (const last_pr of [1, 19621, Number.MAX_SAFE_INTEGER]) {
    assert.equal(parseCursor(JSON.stringify({ last_pr })), last_pr);
  }
  for (const value of [null, {}, [], { last_pr: null }, { last_pr: '19621' },
    { last_pr: 0 }, { last_pr: -1 }, { last_pr: 1.5 }, { last_pr: true },
    { last_pr: [19621] }, { last_pr: Number.MAX_SAFE_INTEGER + 1 }]) {
    assert.throws(() => parseCursor(JSON.stringify(value)), /positive PR number/);
  }
  assert.throws(() => parseCursor('{broken'), SyntaxError);
});

test('restore downloads the trusted artifact as ZIP and reads its exact cursor.json entry', async () => {
  const before = temporaryCursors();
  const f = fixture([[artifact(1)]]);
  assert.deepEqual(await f.restore(), { cursor: 19621, nextArtifact: NEXT_ARTIFACT });
  assert.deepEqual(f.state.calls.filter(call => call.kind === 'download'),
    [{ kind: 'download', ...repo, artifact_id: 1, archive_format: 'zip' }]);
  assert.deepEqual(temporaryCursors(), before);
  const absent = fixture();
  assert.deepEqual(await absent.restore(), { nextArtifact: ARTIFACT });
  assert.equal(absent.state.calls.some(call => call.kind === 'download'), false);
});

test('invalid ZIPs, missing entries, broken JSON and invalid cursors fail without older-artifact fallback', async () => {
  const before = temporaryCursors();
  const nonZip = Uint8Array.from(Buffer.from('Not an archive')).buffer;
  for (const archive of [nonZip, zip('missing'), zip('invalid'), zip('schema')]) {
    const f = fixture([[artifact(1), artifact(2)]], { archive });
    await assert.rejects(f.restore());
    assert.deepEqual(f.state.calls.filter(call => call.kind === 'download').map(call => call.artifact_id), [2]);
    assert.deepEqual(temporaryCursors(), before);
  }
});

test('restore propagates download API errors and releases its quota hooks', async () => {
  for (const status of [403, 404, 429, 500, 503]) {
    const failure = Object.assign(new Error('Download failed'), { status });
    const f = fixture([[artifact(1)]], { errors: { download: failure } });
    await assert.rejects(f.restore(), error => error === failure);
    assert.equal(f.state.calls.filter(call => call.kind === 'download').length, 1);
    assert.equal(f.state.before.length + f.state.after.length, 0);
  }
});

test('restore rechecks quota after lookup and does not download at the 1000-request reserve', async () => {
  const f = fixture([[artifact(1)]], { remaining: 1003 });
  await assert.rejects(f.restore(), { code: 'SEMANTIC_REVIEW_QUOTA' });
  assert.equal(f.state.remaining, 1000);
  assert.equal(f.state.rateReads, 2);
  assert.deepEqual(f.state.calls.map(call => call.kind), ['list', 'list', 'run']);
  assert.equal(f.state.before.length + f.state.after.length, 0);
  const enough = fixture([[artifact(1)]], { remaining: 1004 });
  assert.deepEqual(await enough.restore(), { cursor: 19621, nextArtifact: NEXT_ARTIFACT });
  assert.equal(enough.state.remaining, 1000);
});

test('a failure listing the second slot propagates even when the first has a valid cursor', async () => {
  const failure = Object.assign(new Error('Second slot unavailable'), { status: 503 });
  const f = fixture([[artifact(1)]], { errors: { list: failure }, errorName: NEXT_ARTIFACT });
  await assert.rejects(f.restore(), error => error === failure);
  assert.deepEqual(f.state.calls.map(call => call.name), [ARTIFACT, NEXT_ARTIFACT]);
});

test('either slot can be newest and successful subsequent saves alternate the target name', async () => {
  for (const latestName of [ARTIFACT, NEXT_ARTIFACT]) {
    const otherName = latestName === ARTIFACT ? NEXT_ARTIFACT : ARTIFACT;
    const pages = [[artifact(1, { name: otherName }), artifact(2, { name: latestName })]];
    const f = fixture(pages);
    const restored = await f.restore();
    assert.deepEqual(restored, { cursor: 19621, nextArtifact: otherName });
    assert.equal((await f.find()).id, 2);
    pages[0] = pages[0].filter(item => item.name !== restored.nextArtifact);
    pages[0].push(artifact(3, { name: restored.nextArtifact }));
    assert.deepEqual(await f.restore(), { cursor: 19621, nextArtifact: latestName });
    assert.equal((await f.find()).id, 3);
  }
});

test('a same-run save failure after deleting the older target slot preserves the restored latest cursor', async () => {
  const run = { id: 9001 };
  const older = artifact(1, { name: ARTIFACT, workflow_run: run });
  const latest = artifact(2, { name: NEXT_ARTIFACT, workflow_run: run });
  const pages = [[older, latest]];
  const f = fixture(pages);
  const restored = await f.restore();
  assert.deepEqual(restored, { cursor: 19621, nextArtifact: ARTIFACT });
  pages[0] = pages[0].filter(item => item.name !== restored.nextArtifact);
  assert.deepEqual(await f.restore(), restored);
  assert.equal((await f.find()).id, latest.id);
  assert.deepEqual(f.state.calls.filter(call => call.kind === 'download').map(call => call.artifact_id),
    [latest.id, latest.id]);
});
