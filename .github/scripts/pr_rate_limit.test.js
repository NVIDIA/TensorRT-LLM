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
function extractScript(source) {
  const parts = source.split('          script: |\n');
  assert.equal(parts.length, 2, 'Expected exactly one inline script block');
  const lines = parts[1].split('\n');
  const end = lines.findIndex((line) => line.trim() && !line.startsWith('            '));
  return lines.slice(0, end < 0 ? lines.length : end)
    .map((line) => line.replace(/^ {12}/, '')).join('\n');
}
const script = extractScript(workflow);
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
  const warnings = [];
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
      if (options.permissionError) throw Object.assign(new Error('permission failed'), { status: options.permissionStatus });
      return { data: { permission: options.permission || 'read', user: { permissions: { push: options.push } } } };
    } },
    issues: {
      listForRepo: async (args) => {
        assert.equal(args.creator, current.user.login);
        assert.ok(['all', 'open'].includes(args.state));
        if (options.historyError || args.page === options.failedPage) throw new Error('history failed');
        if (args.state === 'open' && options.openHistoryError) throw new Error('open history failed');
        if (args.state === 'open') return { data: args.page === 1
          ? options.openHistory || (options.history || Array.from({ length: 6 }, (_, i) => pr(i + 1)))
            .filter((item) => item.state === 'open') : [] };
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
  const context = { repo: { owner: 'NVIDIA', repo: 'TensorRT-LLM' }, payload: options.payload || { pull_request: current } };
  await execute(github, context, { info: () => {}, warning: (message) => warnings.push(message) }, { env: options.env || {} },
    class extends Date { static now() { return now; } });
  return { writes, calls, warnings };
}
test('first five pass; sixth receives explanation before closure', async () => {
  for (let n = 1; n <= 5; n++) assert.deepEqual((await run({ current: pr(n), history: Array.from({ length: n }, (_, i) => pr(i + 1)) })).writes, []);
  const { writes } = await run();
  assert.deepEqual(writes.map((w) => w.kind), ['comment', 'close']);
  assert.match(writes[0].body, /currently have 6 open PRs/);
  assert.match(writes[0].body, /no merged PRs/);
  assert.match(writes[0].body, /wait until your existing PRs are reviewed or merged/);
  assert.doesNotMatch(writes[0].body, /24|cooldown|consolidate|rolling/);
  assert.equal(writes[1].state, 'closed');
});
test('counts drafts but excludes closed PRs, ordinary issues and another author', async () => {
  const history = Array.from({ length: 5 }, (_, i) => pr(i + 1, { draft: true }));
  history.push(pr(-2, { state: 'closed' }), pr(0, { pull_request: undefined }), pr(-1, { user: { id: 99 } }));
  assert.equal((await run({ history })).writes.length, 2);
});


test('history pagination and duplicate entries do not change quota', async () => {
  const first = Array.from({ length: 100 }, () => pr(1));
  const { writes } = await run({ pages: [first, [pr(2), pr(3), pr(4), pr(5)]] });
  assert.equal(writes.length, 2);
  assert.match(writes[0].body, /currently have 6 open PRs/);
});
test('actual merged history exempts; author association alone does not', async () => {
  const history = Array.from({ length: 6 }, (_, i) => pr(i + 1));
  history.push(pr(0, { state: 'closed', pull_request: { merged_at: '2026-09-08T00:00:00Z' } }));
  const result = await run({ history });
  assert.deepEqual(result.writes, []);
  assert.deepEqual(result.calls, [6]);
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

test('reruns reuse only bot-authored comments and retry closure', async () => {
  const body = '<!-- new-contributor-pr-rate-limit:v2 -->';
  const trusted = { user: { login: 'github-actions[bot]' }, body };
  assert.deepEqual((await run({ comments: [trusted] })).writes.map((w) => w.kind), ['close']);
  assert.equal((await run({ comments: [{ user: { login: 'new-user' }, body }] })).writes.length, 2);
});
test('unexpected API failures abort without moderation', async () => {
  for (const options of [{ readError: true }, { historyError: true }, { commentError: true },
    { permissionError: true }, { commentsError: true }, { openHistoryError: true },
    { pages: [Array.from({ length: 100 }, () => pr(1))], failedPage: 2 }]) {
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
  assert.match(workflow, /group: pr-rate-limit-\$\{\{ github.event.pull_request.number \|\| inputs.pr_number \}\}/);
  assert.doesNotMatch(workflow, /actions\/checkout|head.sha|secrets\./);
});

test('manual recovery evaluates the requested PR and rejects invalid input before API calls', async () => {
  const { writes, calls } = await run({ payload: { inputs: { pr_number: '6' } } });
  assert.deepEqual(writes.map((w) => w.kind), ['comment', 'close']);
  assert.ok(calls.every((number) => number === 6));
  for (const pr_number of ['', '0', '-1', '1.5', '06', ' 6', '6x', '9007199254740992']) {
    await assert.rejects(run({ payload: { inputs: { pr_number } }, readError: true }), /positive integer PR number/);
  }
  await assert.rejects(run({ payload: {}, readError: true }), /positive integer PR number/);
});

test('manual recovery preserves dry run, exemptions, capacity and comment reconciliation', async () => {
  const payload = { inputs: { pr_number: '6' } };
  assert.deepEqual((await run({ payload, env: { DRY_RUN: 'true' } })).writes, []);
  assert.deepEqual((await run({ payload, permission: 'write' })).writes, []);
  assert.deepEqual((await run({ payload, current: pr(6, { created_at: new Date(now - 24 * hour).toISOString() }), history: [] })).writes, []);
  const comments = [{ user: { login: 'github-actions[bot]' }, body: '<!-- new-contributor-pr-rate-limit:v2 -->' }];
  assert.deepEqual((await run({ payload, comments })).writes.map((w) => w.kind), ['close']);
});

test('privileged action is pinned and manual runs are restricted to the default branch', () => {
  assert.match(workflow, /uses: actions\/github-script@[0-9a-f]{40} # v8/);
  assert.match(workflow, /workflow_dispatch:/);
  assert.match(workflow, /github.event_name != 'workflow_dispatch' \|\|/);
  assert.ok(workflow.includes("github.ref == format('refs/heads/{0}', github.event.repository.default_branch)"));
});

test('open-PR limit prevents reopening an old backlog', async () => {
  const current = pr(6, { created_at: new Date(now - 48 * hour).toISOString() });
  const history = Array.from({ length: 5 }, (_, i) => pr(i + 1, { draft: true }));
  const { writes } = await run({ current, history });
  assert.deepEqual(writes.map((w) => w.kind), ['comment', 'close']);
  assert.match(writes[0].body, /currently have 6 open PRs/);
  assert.match(writes[0].body, /ask a maintainer to reopen this PR when fewer than 5 of your other PRs are open/);
  assert.doesNotMatch(writes[0].body, /cooldown ends/);
  assert.deepEqual((await run({ current, history: history.slice(0, 4) })).writes, []);
});

test('old reopened PR is allowed when capacity becomes available during evaluation', async () => {
  const current = pr(6, { created_at: new Date(now - 48 * hour).toISOString() });
  assert.deepEqual((await run({ current, openHistory: [pr(1), pr(2)] })).writes, []);
});

test('open-PR limit applies even when recent submissions are below five', async () => {
  const history = Array.from({ length: 5 }, (_, i) => pr(i + 1, { created_at: new Date(now - 48 * hour).toISOString() }));
  const { writes } = await run({ history });
  assert.match(writes[0].body, /currently have 6 open PRs/);
  assert.doesNotMatch(writes[0].body, /cooldown ends/);
});

test('a reopened backlog of old PRs stays subject to the open cap during manual recovery', async () => {
  const history = Array.from({ length: 100 }, (_, i) => pr(i + 1, { created_at: new Date(now - 48 * hour).toISOString() }));
  const current = history[99];
  const { writes } = await run({ current, pages: [history, []], openHistory: history, payload: { inputs: { pr_number: '100' } } });
  assert.match(writes[0].body, /currently have 100 open PRs/);
  assert.equal(writes[1].pull_number, 100);
});

test('closed submissions never consume capacity, regardless of timestamps', async () => {
  const history = Array.from({ length: 30 }, (_, i) => pr(i + 1, { state: 'closed', created_at: 'invalid' }));
  assert.deepEqual((await run({ history })).writes, []);
});
test('permission denials and incomplete history warn and skip', async () => {
  for (const options of [
    { permissionError: true, permissionStatus: 403 },
    { permissionError: true, permissionStatus: 404 },
    { pages: Array.from({ length: 20 }, () => Array.from({ length: 100 }, () => pr(1))) },
  ]) {
    const result = await run(options);
    assert.deepEqual(result.writes, []);
    assert.equal(result.warnings.length, 1);
  }
});
test('configured open cap is enforced and invalid configuration skips all API calls', async () => {
  assert.deepEqual((await run({ env: { MAX_OPEN: '6' } })).writes, []);
  const { writes } = await run({ env: { MAX_OPEN: '3' } });
  assert.match(writes[0].body, /limit: 3/);
  for (const value of ['0', '-1', '1.5', 'bad', 'Infinity', '9007199254740992']) {
    const result = await run({ env: { MAX_OPEN: value } });
    assert.deepEqual(result.calls, []);
    assert.deepEqual(result.writes, []);
    assert.equal(result.warnings.length, 1);
  }
});
test('script extraction stops at a subsequent step and rejects ambiguous blocks', () => {
  assert.equal(extractScript(workflow + '\n      - run: echo later\n'), script);
  assert.throws(() => extractScript(''), /exactly one/);
  assert.throws(() => extractScript(workflow + '          script: |\n'), /exactly one/);
});
