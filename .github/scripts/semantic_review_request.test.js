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
const test = require('node:test');
const { discover, run, requestOne } = require('./semantic_review_request');
const { NAME, requests } = require('./semantic_review');

const HEAD = '1'.repeat(40);
const TARGET = '2'.repeat(40);
const BASE = '3'.repeat(40);
const OTHER = '4'.repeat(40);
const SERVICE = { login: 'trtllm-agent', id: 296075020, type: 'User' };
const APPROVED = [{ name: 'ci: full pre-merge approved' }];

function pull(number = 1, changes = {}) {
  return { number, state: 'open', draft: false, base: { ref: 'main' },
    head: { sha: HEAD }, labels: APPROVED, auto_merge: null, ...changes };
}

function fixture(prs = [pull()], options = {}) {
  const state = {
    prs, checks: [], comments: new Map(), posts: [], updates: [], warnings: [],
    failures: [], remaining: options.remaining ?? 5000, target: TARGET,
    mergeBase: BASE, service: SERVICE, readCounts: new Map(), refReads: 0,
    before: [], after: [], commandBefore: [], commandAfter: [],
    commandRemaining: options.commandRemaining ?? 5000, postErrors: new Map(), comparisons: 0,
  };
  const api = (method, commandToken = false) => async (args) => {
    const key = commandToken ? 'commandRemaining' : 'remaining';
    for (const hook of commandToken ? state.commandBefore : state.before) await hook();
    state[key] -= 1;
    const response = { data: await method(args),
      headers: { 'x-ratelimit-remaining': String(state[key]) } };
    for (const hook of commandToken ? state.commandAfter : state.after) await hook(response);
    return response;
  };
  const github = {
    hook: {
      before: (_, callback) => state.before.push(callback),
      after: (_, callback) => state.after.push(callback),
      remove: (_, callback) => {
        state.before = state.before.filter((hook) => hook !== callback);
        state.after = state.after.filter((hook) => hook !== callback);
      },
    },
    paginate: async (method, args) => {
      const { data } = await method(args);
      return data.check_runs || data;
    },
    rest: {
      rateLimit: { get: async () => ({ data: { resources: { core: { remaining: state.remaining } } } }) },
      pulls: {
        list: api(() => structuredClone(state.prs)),
        get: api(({ pull_number: number }) => {
          const count = (state.readCounts.get(number) || 0) + 1;
          state.readCounts.set(number, count);
          const pr = structuredClone(state.prs.find((item) => item.number === number));
          return options.readPR ? options.readPR(pr, count) : pr;
        }),
      },
      git: { getRef: api(() => {
        state.refReads += 1;
        return { object: { sha: options.readTarget ?
          options.readTarget(state.refReads) : state.target } };
      }) },
      repos: { compareCommits: api(() => {
        state.comparisons += 1;
        return { merge_base_commit: { sha: state.mergeBase } };
      }) },
      issues: { listComments: api(({ issue_number: number }) => state.comments.get(number) || []) },
      checks: {
        listForRef: api(({ ref, check_name: name }) => ({ check_runs: state.checks.filter(
          (check) => check.head_sha === ref && check.name === name) })),
        create: api((args) => {
          const check = { ...args, id: state.checks.length + 100,
            app: { slug: 'github-actions' } };
          state.checks.push(check);
          return check;
        }),
        update: api((args) => {
          state.updates.push(args);
          const check = state.checks.find((item) => item.id === args.check_run_id);
          Object.assign(check, args);
          return check;
        }),
      },
    },
  };
  const commandGithub = { hook: {
    before: (_, callback) => state.commandBefore.push(callback),
    after: (_, callback) => state.commandAfter.push(callback),
    remove: (_, callback) => {
      state.commandBefore = state.commandBefore.filter((hook) => hook !== callback);
      state.commandAfter = state.commandAfter.filter((hook) => hook !== callback);
    },
  }, rest: {
    rateLimit: { get: async () => ({ data: { resources: { core: { remaining: state.commandRemaining } } } }) },
    users: { getAuthenticated: api(() => state.service, true) },
    issues: { createComment: api((args) => {
      assert.equal(args.request.retries, 0);
      const errorMode = state.postErrors.get(args.issue_number);
      state.postErrors.delete(args.issue_number);
      const failure = () => Object.assign(new Error('Delivery unavailable'), { status: 502 });
      if (errorMode === 'before') throw failure();
      state.posts.push(args);
      const comments = state.comments.get(args.issue_number) || [];
      const comment = { id: state.posts.length + 1000, user: SERVICE, body: args.body,
        created_at: new Date().toISOString() };
      state.comments.set(args.issue_number, comments.concat(comment));
      if (errorMode === 'after') throw failure();
      return comment;
    }, true) },
  } };
  const context = { repo: { owner: 'NVIDIA', repo: 'TensorRT-LLM' }, eventName: 'schedule' };
  const core = {
    warning: (message) => state.warnings.push(message), info: () => {},
    setFailed: (message) => state.failures.push(message),
  };
  const args = { github, commandGithub, context, core };
  return { ...args, state, one: (changes = {}) => requestOne({ ...args, number: 1, ...changes }),
    scan: (changes = {}) => discover({ ...args, ...changes }),
    worker: (changes = {}) => run({ ...args, number: 1, ...changes }) };
}

test('eligibility accepts either approval or auto-merge and requires an open supported non-draft PR', async () => {
  for (const pr of [pull(), pull(1, { labels: [], auto_merge: { enabled_by: {} } }),
    pull(1, { base: { ref: 'release/1.2' } })]) {
    const f = fixture([pr]);
    assert.equal((await f.one()).status, 'requested');
  }
  for (const changes of [{ labels: [] }, { state: 'closed' }, { draft: true },
    { base: { ref: 'feature/experimental' } }]) {
    const f = fixture([pull(1, changes)]);
    assert.equal((await f.one()).status, 'ineligible');
    assert.equal(f.state.posts.length, 0);
    assert.equal(f.state.checks.length, 0);
  }
});

test('first request creates a neutral check and records the exact analysis identity', async () => {
  const f = fixture();
  const result = await f.one();
  const recorded = requests(f.state.comments.get(1))[0];
  assert.equal(result.status, 'requested');
  assert.deepEqual([recorded.head, recorded.target, recorded.mergeBase, recorded.branch],
    [HEAD, TARGET, BASE, 'main']);
  assert.equal(recorded.checkId, f.state.checks[0].id);
  assert.equal(f.state.checks[0].external_id, `semantic-review:1:${recorded.id}`);
  assert.equal(f.state.checks[0].conclusion, 'neutral');
  assert.match(f.state.posts[0].body, /^@coderabbitai/);
});

test('an already requested pair is skipped even without a reply; force issues a new request', async () => {
  const f = fixture();
  const first = await f.one();
  assert.equal((await f.one()).status, 'unchanged');
  assert.equal(f.state.posts.length, 1);
  const forced = await f.one({ manual: true });
  assert.equal(forced.status, 'requested');
  assert.notEqual(first.request.id, forced.request.id);
  assert.equal(first.request.checkId, forced.request.checkId);
  assert.equal(f.state.checks.length, 1);
});

test('same SHA pair on a different target branch is a distinct request', async () => {
  const f = fixture();
  await f.one();
  f.state.prs[0].base.ref = 'release/1.2';
  assert.equal((await f.one()).status, 'requested');
  assert.equal(f.state.posts.length, 2);
});

test('a forged request comment cannot suppress analysis', async () => {
  const f = fixture();
  await f.one();
  f.state.comments.get(1)[0].user = { ...SERVICE, id: 999 };
  assert.equal((await f.one()).status, 'requested');
});

test('a changed target reuses the head check and revokes the old verdict; a changed head gets a new check', async () => {
  const f = fixture();
  const first = await f.one();
  f.state.checks[0].conclusion = 'success';
  f.state.target = OTHER;
  const second = await f.one();
  assert.equal(first.request.checkId, second.request.checkId);
  assert.equal(f.state.checks[0].conclusion, 'neutral');
  assert.match(f.state.checks[0].external_id, new RegExp(second.request.id));
  f.state.prs[0].head.sha = '5'.repeat(40);
  const third = await f.one();
  assert.notEqual(third.request.checkId, second.request.checkId);
  assert.equal(f.state.checks.length, 2);
});

test('a check from a different app or PR cannot be reused', async () => {
  const f = fixture();
  f.state.checks.push({ id: 77, name: NAME, head_sha: HEAD,
    app: { slug: 'another-app' }, external_id: 'semantic-review:1:unrelated' });
  f.state.checks.push({ id: 78, name: NAME, head_sha: HEAD,
    app: { slug: 'github-actions' }, external_id: 'semantic-review:2:unrelated' });
  const result = await f.one();
  assert.notEqual(result.request.checkId, 77);
  assert.notEqual(result.request.checkId, 78);
});

test('a target already contained in the PR is analyzed with its real merge base', async () => {
  const f = fixture();
  f.state.mergeBase = TARGET;
  assert.equal((await f.one()).status, 'requested');
  assert.equal((await f.one({ manual: true })).status, 'requested');
  assert.equal(f.state.posts.length, 2);
  assert.equal(requests(f.state.comments.get(1))[0].mergeBase, TARGET);
});

test('live head changes, branch changes and approval removal stop stale requests before mutation', async () => {
  for (const change of [(pr) => { pr.head.sha = OTHER; },
    (pr) => { pr.base.ref = 'release/1.2'; }, (pr) => { pr.labels = []; }]) {
    const f = fixture([pull()], { readPR: (pr, count) => {
      if (count === 2) change(pr);
      return pr;
    } });
    assert.equal((await f.one()).status, 'moved');
    assert.equal(f.state.posts.length, 0);
    assert.equal(f.state.checks.length, 0);
  }
  const f = fixture([pull()], { readTarget: (count) => count === 1 ? TARGET : OTHER });
  assert.equal((await f.one()).status, 'moved');
  assert.equal(f.state.checks.length, 0);
});

test('a token with the right login but wrong immutable service ID is rejected', async () => {
  const f = fixture();
  f.state.service = { ...SERVICE, id: 999 };
  await assert.rejects(f.one(), { code: 'SEMANTIC_REVIEW_COMMAND_USER' });
  assert.equal(f.state.checks.length, 0);
  assert.equal(f.state.posts.length, 0);
});

test('failed posting leaves a neutral check and allows retry without a false dedup record', async () => {
  const f = fixture();
  await f.one();
  f.state.checks[0].conclusion = 'success';
  f.state.target = OTHER;
  f.state.postErrors.set(1, 'before');
  await assert.rejects(f.one(), { status: 502 });
  assert.equal(f.state.checks[0].conclusion, 'neutral');
  assert.match(f.state.checks[0].output.title, /could not be confirmed/);
  assert.equal((await f.one()).status, 'requested');
  assert.equal(f.state.posts.length, 2);
  assert.equal(f.state.checks.length, 1);
});

test('ambiguous POST acceptance is recovered from the trusted comment without a second request', async () => {
  const f = fixture();
  f.state.postErrors.set(1, 'after');
  await assert.rejects(f.one(), { status: 502 });
  assert.equal(f.state.posts.length, 1);
  assert.equal((await f.one()).status, 'unchanged');
  assert.equal(f.state.posts.length, 1);
  assert.equal(f.state.checks[0].conclusion, 'neutral');
});

test('discovery selects at most 30 new requests, newest PR first on every scan', async () => {
  const candidates = Array.from({ length: 45 }, (_, index) => pull(index + 1));
  const f = fixture(candidates);
  const expected = Array.from({ length: 30 }, (_, index) => 45 - index);
  assert.deepEqual((await f.scan()).numbers, expected);
  assert.deepEqual((await f.scan()).numbers, expected);
  assert.equal(f.state.comparisons, 0);
  assert.equal(f.state.posts.length, 0);
  assert.equal(f.state.checks.length, 0);
});

test('already requested and newly ineligible candidates do not consume discovery slots', async () => {
  const f = fixture(Array.from({ length: 45 }, (_, index) => pull(index + 1)), {
    readPR: (pr, count) => {
      if (pr.number === 45 && count > 2) pr.labels = [];
      return pr;
    },
  });
  for (let number = 36; number <= 45; number += 1) await f.one({ number });
  const result = await f.scan();
  assert.deepEqual(result.numbers, Array.from({ length: 30 }, (_, index) => 35 - index));
  assert.equal(result.skipped, 10);
  assert.equal(f.state.posts.length, 10);
});

test('discovery read errors consume slots and report failure while preserving other selected PRs', async () => {
  const f = fixture(Array.from({ length: 40 }, (_, index) => pull(index + 1)), {
    readPR: (pr) => {
      if (pr.number > 34) throw Object.assign(new Error('Unavailable'), { status: 502 });
      return pr;
    },
  });
  const result = await f.scan();
  assert.equal(result.failed, 6);
  assert.deepEqual(result.numbers, Array.from({ length: 24 }, (_, index) => 34 - index));
  assert.equal(f.state.warnings.length, 5);
  assert.equal(f.state.failures.length, 1);
  assert.equal(f.state.checks.length, 0);
});

test('at most 30 workers run per discovery even when a POST is accepted ambiguously', async () => {
  const f = fixture(Array.from({ length: 40 }, (_, index) => pull(index + 1)));
  const { numbers } = await f.scan();
  for (const number of numbers.slice(0, 3)) f.state.postErrors.set(number, 'after');
  const results = [];
  for (const number of numbers) results.push(await f.worker({ number }));
  assert.equal(results.filter((result) => result.status === 'requested').length, 27);
  assert.equal(results.filter((result) => result.status === 'failed').length, 3);
  assert.equal(f.state.posts.length, 30);
  assert.equal(f.state.failures.length, 3);
  assert.equal(f.state.checks.every((check) => check.conclusion === 'neutral'), true);
  assert.equal((await f.worker({ number: numbers[0] })).status, 'unchanged');
  assert.equal(f.state.posts.length, 30);
});

test('worker rechecks approval, refs and request comments after discovery', async () => {
  const f = fixture([pull(1), pull(2), pull(3)]);
  assert.deepEqual((await f.scan()).numbers, [3, 2, 1]);
  f.state.prs[0].labels = [];
  assert.equal((await f.worker({ number: 1 })).status, 'ineligible');
  await f.one({ number: 2 });
  assert.equal((await f.worker({ number: 2 })).status, 'unchanged');
  f.state.prs[2].head.sha = OTHER;
  f.state.target = '5'.repeat(40);
  const result = await f.worker({ number: 3 });
  assert.equal(result.status, 'requested');
  assert.equal(result.request.head, OTHER);
  assert.equal(result.request.target, f.state.target);
  assert.equal(f.state.posts.length, 2);
});

test('discovery preserves 1000 REST requests and returns candidates already found', async () => {
  const f = fixture([pull(1), pull(2)], { remaining: 1005 });
  const result = await f.scan();
  assert.equal(result.limited, true);
  assert.deepEqual(result.numbers, [2]);
  assert.equal(f.state.remaining, 1000);
  assert.equal(f.state.failures.length, 0);
  assert.equal(f.state.before.length, 0);
  assert.equal(f.state.after.length, 0);
});

test('worker preserves the 1000-request reserve independently for both tokens', async () => {
  for (const options of [{ remaining: 1000 }, { remaining: 1004 },
    { commandRemaining: 1000 }, { commandRemaining: 1001 }]) {
    const f = fixture([pull()], options);
    assert.equal((await f.worker()).status, 'limited');
    assert.equal(f.state.posts.length, 0);
    assert.ok(f.state.remaining >= 1000);
    assert.ok(f.state.commandRemaining >= 1000);
    assert.equal(f.state.checks.every((check) => check.conclusion === 'neutral'), true);
    assert.equal(f.state.failures.length, 0);
    assert.equal(f.state.before.length + f.state.after.length +
      f.state.commandBefore.length + f.state.commandAfter.length, 0);
  }
  const f = fixture([pull()], { commandRemaining: 1002 });
  assert.equal((await f.worker()).status, 'requested');
  assert.equal(f.state.commandRemaining, 1000);
});

test('rate-limit responses stop discovery without marking an AI failure', async () => {
  for (const error of [Object.assign(new Error('Retry later'), { status: 429 }),
    Object.assign(new Error('Secondary rate limit'), { status: 403 })]) {
    const f = fixture([pull()], { readPR: () => { throw error; } });
    const result = await f.scan();
    assert.equal(result.limited, true);
    assert.equal(result.failed, 0);
    assert.equal(f.state.failures.length, 0);
    assert.equal(f.state.checks.length, 0);
  }
});

test('worker operational errors fail its job, release quota hooks and never pass the AI check', async () => {
  const f = fixture();
  f.state.service = { ...SERVICE, id: 999 };
  assert.equal((await f.worker()).status, 'failed');
  assert.equal(f.state.failures.length, 1);
  assert.equal(f.state.posts.length, 0);
  assert.equal(f.state.checks.length, 0);
  assert.equal(f.state.before.length + f.state.after.length +
    f.state.commandBefore.length + f.state.commandAfter.length, 0);
});

test('manual dispatch validates input and forces any open supported PR, including drafts', async () => {
  const previous = process.env.INPUT_PULL_NUMBER;
  try {
    const f = fixture([pull(1, { labels: [], draft: true })]);
    f.context.eventName = 'workflow_dispatch';
    for (const input of ['', '0', '-1', '1.5', '1oops', '9007199254740992']) {
      process.env.INPUT_PULL_NUMBER = input;
      await assert.rejects(f.scan(), /positive pull request number/);
    }
    process.env.INPUT_PULL_NUMBER = '1';
    f.state.mergeBase = TARGET;
    assert.deepEqual((await f.scan()).numbers, [1]);
    assert.equal((await f.worker()).status, 'requested');
    assert.deepEqual((await f.scan()).numbers, [1]);
    assert.equal((await f.worker()).status, 'requested');
    assert.equal(f.state.posts.length, 2);
    for (const change of [{ state: 'closed' }, { base: { ref: 'feature/experimental' } }]) {
      f.state.prs[0] = pull(1, change);
      assert.deepEqual((await f.scan()).numbers, []);
      assert.equal((await f.worker()).status, 'ineligible');
    }
    const limited = fixture([pull(1, { labels: [], draft: true })], { commandRemaining: 1000 });
    limited.context.eventName = 'workflow_dispatch';
    assert.equal((await limited.worker()).status, 'limited');
    assert.equal(limited.state.posts.length, 0);
  } finally {
    if (previous === undefined) delete process.env.INPUT_PULL_NUMBER;
    else process.env.INPUT_PULL_NUMBER = previous;
  }
});
