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
const fs = require('node:fs');
const { test } = require('node:test');
const workflow = fs.readFileSync(`${__dirname}/../workflows/pr-rate-limit.yml`, 'utf8');
const script = workflow.split('          script: |\n')[1]
  .split('\n').map((line) => line.replace(/^ {12}/, '')).join('\n');
const execute = new (Object.getPrototypeOf(async function () {}).constructor)(
  'github', 'context', 'core', 'process', 'Date', script,
);
const now = Date.parse('2026-09-09T12:00:00Z');
const hour = 3600000;
function pr(number, overrides = {}) {
  return { number, created_at: new Date(now - hour + number * 1000).toISOString(),
    state: 'open', labels: [], user: { id: 42, login: 'new-user', type: 'User' },
    pull_request: {}, ...overrides };
}
async function run(options = {}) {
  const current = options.current || pr(6);
  const writes = options.writes || [];
  const calls = [];
  const github = { rest: {
    pulls: {
      get: async ({ pull_number }) => {
        calls.push(pull_number);
        if (options.readError) throw new Error('API failure');
        return { data: pull_number === current.number
          ? (calls.filter((n) => n === current.number).length > 1 && options.recheck || current)
          : { ...pr(pull_number), merged_at: options.merged ? '2026-09-08T00:00:00Z' : null } };
      },
      update: async (args) => {
        if (options.closeError) throw new Error('close failed');
        writes.push({ kind: 'close', ...args });
      },
    },
    repos: { getCollaboratorPermissionLevel: async () => {
      if (options.permissionError) throw new Error('permission failed');
      return { data: { permission: options.permission || 'read', user: { permissions: { push: options.push } } } };
    } },
    issues: {
      listForRepo: async (args) => {
        assert.equal(args.creator, current.user.login);
        assert.equal(args.state, 'all');
        if (options.historyError || args.page === options.failedPage) throw new Error('history failed');
        return { data: options.pages ? (options.pages[args.page - 1] || [])
          : options.history || Array.from({ length: 6 }, (_, i) => pr(i + 1)) };
      },
      listComments: () => {},
      createComment: async (args) => {
        if (options.commentError) throw new Error('comment failed');
        writes.push({ kind: 'comment', ...args });
      },
    },
  }, paginate: async () => {
    if (options.commentsError) throw new Error('comments failed');
    return options.comments || [];
  } };
  const context = { repo: { owner: 'NVIDIA', repo: 'TensorRT-LLM' }, payload: { pull_request: current } };
  await execute(github, context, { info: () => {} }, { env: options.env || {} },
    class extends Date { static now() { return now; } });
  return { writes, calls };
}
test('first five pass; sixth receives explanation before closure', async () => {
  for (let n = 1; n <= 5; n++) assert.deepEqual((await run({ current: pr(n) })).writes, []);
  const { writes } = await run();
  assert.deepEqual(writes.map((w) => w.kind), ['comment', 'close']);
  assert.match(writes[0].body, /submitted 6 PRs/);
  assert.match(writes[0].body, /no merged PRs/);
  assert.match(writes[0].body, /2026-09-10T11:00:06.000Z/);
  assert.equal(writes[1].state, 'closed');
});
test('counts drafts and closed unmerged PRs, but not ordinary issues or another author', async () => {
  const history = Array.from({ length: 5 }, (_, i) => pr(i + 1, { draft: true, state: 'closed' }));
  history.push(pr(0, { pull_request: undefined }), pr(-1, { user: { id: 99 } }));
  assert.equal((await run({ history })).writes.length, 2);
});
test('exact rolling boundary excluded; later PRs cannot penalize earlier submissions', async () => {
  const current = pr(6);
  const boundary = new Date(Date.parse(current.created_at) - 24 * hour).toISOString();
  const history = [pr(1, { created_at: boundary }), ...[2, 3, 4, 5].map((n) => pr(n)), pr(7)];
  assert.deepEqual((await run({ current, history })).writes, []);
});
test('same-second bursts ordered by PR number, regardless of execution order', async () => {
  const history = Array.from({ length: 10 }, (_, i) => pr(i + 1, { created_at: pr(1).created_at }));
  assert.deepEqual((await run({ current: history[4], history })).writes, []);
  assert.equal((await run({ current: history[5], history })).writes.length, 2);
});
test('history pagination and duplicate entries do not change quota', async () => {
  const first = Array.from({ length: 100 }, () => pr(1));
  const { writes } = await run({ pages: [first, [pr(2), pr(3), pr(4), pr(5)]] });
  assert.equal(writes.length, 2);
  assert.match(writes[0].body, /submitted 6 PRs/);
});
test('actual merged history exempts; author association alone does not', async () => {
  const history = Array.from({ length: 6 }, (_, i) => pr(i + 1));
  history.push(pr(0, { state: 'closed' }));
  assert.deepEqual((await run({ history, merged: true })).writes, []);
  assert.equal((await run({ current: pr(6, { author_association: 'CONTRIBUTOR' }) })).writes.length, 2);
});
test('maintainers, bots, allowlisted users and labeled PRs are exempt', async () => {
  for (const permission of ['write', 'admin', 'maintain'])
    assert.deepEqual((await run({ permission })).writes, []);
  assert.deepEqual((await run({ env: { EXEMPT_USERS: 'other, NEW-USER ' } })).writes, []);
  assert.deepEqual((await run({ push: true })).writes, []);
  assert.deepEqual((await run({ current: pr(6, { user: { id: 42, login: 'bot', type: 'Bot' } }) })).writes, []);
  assert.deepEqual((await run({ current: pr(6, { labels: [{ name: 'pr-rate-limit-exempt' }] }) })).writes, []);
});
test('dry run, closed PRs, and exemptions added during evaluation make no writes', async () => {
  assert.deepEqual((await run({ env: { DRY_RUN: 'true' } })).writes, []);
  assert.deepEqual((await run({ current: pr(6, { state: 'closed' }) })).writes, []);
  assert.deepEqual((await run({ recheck: pr(6, { labels: [{ name: 'pr-rate-limit-exempt' }] }) })).writes, []);
  assert.deepEqual((await run({ recheck: pr(6, { state: 'closed' }) })).writes, []);
});
test('reopening after cooldown is allowed; reopening before it is still limited', async () => {
  assert.deepEqual((await run({ current: pr(6, { created_at: new Date(now - 24 * hour).toISOString() }) })).writes, []);
  assert.equal((await run()).writes.length, 2);
});
test('reruns reuse only bot-authored comments and retry closure', async () => {
  const body = '<!-- new-contributor-pr-rate-limit:v1 -->';
  const trusted = { user: { login: 'github-actions[bot]' }, body };
  assert.deepEqual((await run({ comments: [trusted] })).writes.map((w) => w.kind), ['close']);
  assert.equal((await run({ comments: [{ user: { login: 'new-user' }, body }] })).writes.length, 2);
});
test('API failures, incomplete history and invalid timestamps abort without moderation', async () => {
  for (const options of [{ readError: true }, { historyError: true }, { commentError: true },
    { permissionError: true }, { commentsError: true },
    { pages: [Array.from({ length: 100 }, () => pr(1))], failedPage: 2 },
    { history: [pr(1, { created_at: 'invalid' })] },
    { current: pr(6, { created_at: 'invalid' }) },
    { pages: Array.from({ length: 20 }, () => Array.from({ length: 100 }, () => pr(1))) }]) {
    const writes = [];
    await assert.rejects(run({ ...options, writes }));
    assert.deepEqual(writes, []);
  }
});
test('successful comment survives a close failure and is not duplicated on retry', async () => {
  const writes = [];
  await assert.rejects(run({ closeError: true, writes }), /close failed/);
  assert.deepEqual(writes.map((w) => w.kind), ['comment']);
  const comments = [{ user: { login: 'github-actions[bot]' }, body: writes[0].body }];
  assert.deepEqual((await run({ comments })).writes.map((w) => w.kind), ['close']);
});
test('workflow uses trusted inline code, minimal permissions and per-PR concurrency', () => {
  assert.match(workflow, /pull_request_target:/);
  assert.match(workflow, /types: \[opened, reopened, ready_for_review\]/);
  assert.match(workflow, /pull-requests: write/);
  assert.match(workflow, /issues: read/);
  assert.match(workflow, /group: pr-rate-limit-\$\{\{ github.event.pull_request.number \}\}/);
  assert.doesNotMatch(workflow, /actions\/checkout|head.sha|secrets\./);
});
