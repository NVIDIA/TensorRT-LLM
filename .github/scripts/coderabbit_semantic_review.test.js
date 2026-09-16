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
const {test} = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const workflow = fs.readFileSync(
  path.join(__dirname, '../workflows/coderabbit-semantic-review.yml'), 'utf8');
const marker = '          script: |\n';
assert.ok(workflow.includes(marker));
const script = workflow.slice(workflow.indexOf(marker) + marker.length)
  .split('\n').map(line => line.replace(/^ {12}/, '')).join('\n');
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const execute = new AsyncFunction('github', 'context', 'core', 'process', script);

const HEAD = 'a'.repeat(40);
const BASE = 'b'.repeat(40);
const NEW_BASE = 'c'.repeat(40);

function harness(overrides = {}) {
  const pr = {number: 12, state: 'open', draft: false,
    head: {sha: HEAD}, base: {ref: 'main', sha: 'outdated-event-sha'},
    labels: [{name: 'ai: semantic-conflict'}], ...overrides};
  const comments = [];
  const posted = [];
  let target = BASE;
  const calls = [];
  const github = {
    rest: {
      pulls: {
        list: 'list-pulls',
        get: async args => {calls.push(['get-pr', args]); return {data: pr};},
      },
      git: {getRef: async args => {
        calls.push(['get-ref', args]);
        return {data: {object: {sha: target}}};
      }},
      issues: {
        listComments: 'list-comments',
        createComment: async args => {
          posted.push(args);
          comments.push({body: args.body, user: {login: 'github-actions[bot]'}});
          return {data: {html_url: `https://github.com/example/repo/pull/12#${posted.length}`}};
        },
      },
    },
    paginate: async (method, args) => {
      calls.push([method, args]);
      if (method === 'list-pulls') return [pr];
      assert.equal(method, 'list-comments');
      return comments;
    },
  };
  const summaries = [];
  const core = {info: () => {}, summary: {
    addRaw(text) {summaries.push(text); return this;},
    async write() {},
  }};
  const context = {repo: {owner: 'example', repo: 'repo'},
    eventName: 'pull_request_target', payload: {pull_request: {number: 12}}};
  return {pr, comments, posted, calls, summaries,
    setTarget: sha => {target = sha;},
    run: (eventName = 'pull_request_target', pullNumber = '') =>
      execute(github, {...context, eventName}, core, {env: {DISPATCH_PULL_NUMBER: pullNumber}}),
  };
}

test('requests the live main SHA and never treats dispatch as an AI verdict', async () => {
  const h = harness();
  await h.run();
  assert.equal(h.posted.length, 1);
  assert.match(h.posted[0].body, /^@coderabbitai run pre-merge checks/);
  assert.ok(h.posted[0].body.includes(HEAD));
  assert.ok(h.posted[0].body.includes(BASE));
  assert.ok(!h.posted[0].body.includes('outdated-event-sha'));
  assert.match(h.posted[0].body, /return Inconclusive/);
  assert.match(h.summaries[0], /No AI verdict is asserted/);
  assert.deepEqual(h.calls.find(c => c[0] === 'get-ref')[1],
    {owner: 'example', repo: 'repo', ref: 'heads/main'});
});

test('a main-only update triggers another review with unchanged PR head', async () => {
  const h = harness();
  await h.run('push');
  h.setTarget(NEW_BASE);
  await h.run('push');
  assert.equal(h.posted.length, 2);
  assert.ok(h.posted[1].body.includes(`${HEAD}:${NEW_BASE}`));
  assert.ok(!h.posted[1].body.includes(BASE));
});

test('a PR-only update triggers another review', async () => {
  const h = harness();
  await h.run();
  h.pr.head.sha = 'd'.repeat(40);
  await h.run();
  assert.equal(h.posted.length, 2);
  assert.ok(h.posted[1].body.includes(h.pr.head.sha));
});

test('duplicates are suppressed, but workflow dispatch retries the same pair', async () => {
  const h = harness();
  await h.run();
  await h.run('push');
  assert.equal(h.posted.length, 1);
  await h.run('workflow_dispatch', '12');
  assert.equal(h.posted.length, 2);
});

test('a marker posted by another user cannot suppress requests', async () => {
  const h = harness();
  h.comments.push({body: `<!-- coderabbit-semantic-request:12:${HEAD}:${BASE} -->`,
    user: {login: 'someone-else'}});
  await h.run();
  assert.equal(h.posted.length, 1);
});

test('closed, draft, unlabelled, and release PRs are not reviewed', async () => {
  for (const overrides of [{state: 'closed'}, {draft: true}, {labels: []},
    {base: {ref: 'release/1.0'}}]) {
    const h = harness(overrides);
    await h.run();
    await h.run('workflow_dispatch', '12');
    assert.equal(h.posted.length, 0);
    assert.ok(!h.calls.some(c => c[0] === 'get-ref'));
  }
});

test('invalid dispatch inputs and unsupported events cannot post comments', async () => {
  for (const value of ['', '0', '-1', '1.2', '12x', '9007199254740992']) {
    const h = harness();
    await assert.rejects(h.run('workflow_dispatch', value), /positive integer/);
    assert.equal(h.posted.length, 0);
  }
  await assert.rejects(harness().run('issue_comment'), /Unsupported event/);
});
