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
const { NAME, requests, publish } = require('./semantic_review');

const HEAD = '1'.repeat(40);
const TARGET = '2'.repeat(40);
const BASE = '3'.repeat(40);
const OTHER = '4'.repeat(40);
const HOUR = 60 * 60 * 1000;
const NOW = Date.parse('2026-09-28T00:00:00Z');
const REVIEWER = { login: 'coderabbitai[bot]', id: 136622811, type: 'Bot' };
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
    now: NOW, nextCommentId: 1000, commentReads: new Map(), checkUpdateFailures: 0,
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
      issues: { listComments: api(({ issue_number: number }) => {
        const count = (state.commentReads.get(number) || 0) + 1;
        state.commentReads.set(number, count);
        if (state.onListComments) state.onListComments(number, count);
        return state.comments.get(number) || [];
      }) },
      checks: {
        listForRef: api(({ ref, check_name: name }) => ({ check_runs: structuredClone(state.checks.filter(
          (check) => check.head_sha === ref && check.name === name)) })),
        create: api((args) => {
          const check = { conclusion: null, completed_at: null, ...args, id: state.checks.length + 100,
            app: { slug: 'github-actions' } };
          state.checks.push(check);
          return check;
        }),
        update: api((args) => {
          state.updates.push(args);
          if (state.onCheckUpdate) state.onCheckUpdate(args);
          if (args.conclusion === null || args.completed_at === null) {
            throw Object.assign(new Error('Completed Check fields cannot be cleared'), { status: 422 });
          }
          if (state.checkUpdateFailures > 0) {
            state.checkUpdateFailures -= 1;
            throw Object.assign(new Error('Check update unavailable'), { status: 502 });
          }
          const check = state.checks.find((item) => item.id === args.check_run_id);
          const wasCompleted = check.status === 'completed';
          Object.assign(check, args);
          // GitHub does not reopen a completed run when conclusion is omitted.
          if (wasCompleted && args.status === 'in_progress') check.status = 'completed';
          if (check.status === 'completed' && check.conclusion) {
            check.completed_at = new Date(state.now).toISOString();
          }
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
      const comment = { id: state.nextCommentId++, user: SERVICE, body: args.body,
        created_at: new Date(state.now).toISOString() };
      state.comments.set(args.issue_number, comments.concat(comment));
      if (errorMode === 'after') throw failure();
      return comment;
    }, true) },
  } };
  const context = { repo: { owner: 'NVIDIA', repo: 'TensorRT-LLM' }, eventName: 'schedule' };
  const core = {
    warning: (message) => state.warnings.push(message), info: () => {},
    setFailed: (message) => state.failures.push(message),
    summary: { addRaw: () => ({ write: async () => {} }) },
  };
  const args = { github, commandGithub, context, core };
  return { ...args, state,
    one: (changes = {}) => requestOne({ ...args, number: 1, now: state.now, ...changes }),
    scan: (changes = {}) => discover({ ...args, now: state.now, ...changes }),
    worker: (changes = {}) => run({ ...args, number: 1, now: state.now, ...changes }),
    publish: (number = 1) => publish({ ...args, number }),
    reply: (request, verdict = 'PASS', changes = {}, number = 1) => {
      const id = state.nextCommentId++;
      const body = `SEMANTIC_REVIEW\nSEMANTIC_RESULT request_id=${request.id} ` +
        `head=${request.head} target=${request.target} merge_base=${request.mergeBase} verdict=${verdict}\n` +
        `https://github.com/NVIDIA/TensorRT-LLM/blob/${request.head}/head.py#L1\n` +
        `https://github.com/NVIDIA/TensorRT-LLM/blob/${request.target}/target.py#L1`;
      const comment = { id, user: REVIEWER, body, created_at: new Date(state.now).toISOString(),
        html_url: `https://github.com/NVIDIA/TensorRT-LLM/pull/${number}#issuecomment-${id}`, ...changes };
      state.comments.set(number, [...(state.comments.get(number) || []), comment]);
      return comment;
    },
  };
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

test('first request creates an in-progress check and records the exact analysis identity', async () => {
  const f = fixture();
  const result = await f.one();
  const recorded = requests(f.state.comments.get(1))[0];
  assert.equal(result.status, 'requested');
  assert.deepEqual([recorded.head, recorded.target, recorded.mergeBase, recorded.branch],
    [HEAD, TARGET, BASE, 'main']);
  assert.equal(recorded.checkId, f.state.checks.at(-1).id);
  assert.equal(f.state.checks.at(-1).external_id, `semantic-review:1:${recorded.id}`);
  assert.equal(f.state.checks.at(-1).status, 'in_progress');
  assert.equal(f.state.checks.at(-1).conclusion, null);
  assert.match(f.state.posts[0].body, /^@coderabbitai/);
});

test('an unanswered request younger than two hours is skipped; manual dispatch forces a new request', async () => {
  const f = fixture();
  const first = await f.one();
  assert.equal((await f.one()).status, 'unchanged');
  assert.equal(f.state.posts.length, 1);
  const forced = await f.one({ manual: true });
  assert.equal(forced.status, 'requested');
  assert.notEqual(first.request.id, forced.request.id);
  assert.notEqual(first.request.checkId, forced.request.checkId);
  assert.equal(f.state.checks.length, 2);
  const oldCheck = f.state.checks.find((check) => check.id === first.request.checkId);
  assert.equal(oldCheck.status, 'completed');
  assert.equal(oldCheck.conclusion, 'cancelled');
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

test('each new request creates a Check while retaining completed results as history', async () => {
  const f = fixture();
  const first = await f.one();
  f.reply(first.request);
  await f.publish();
  f.state.target = OTHER;
  const second = await f.one();
  assert.notEqual(first.request.checkId, second.request.checkId);
  assert.equal(f.state.checks.find((check) => check.id === first.request.checkId).conclusion, 'success');
  assert.equal(f.state.checks.at(-1).status, 'in_progress');
  assert.equal(f.state.checks.at(-1).conclusion, null);
  assert.match(f.state.checks.at(-1).external_id, new RegExp(second.request.id));
  f.state.prs[0].head.sha = '5'.repeat(40);
  const third = await f.one();
  assert.notEqual(third.request.checkId, second.request.checkId);
  assert.equal(f.state.checks.length, 3);
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
    assert.ok(['moved', 'ineligible'].includes((await f.one()).status));
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

test('an unaccepted POST cancels only its new Check and remains eligible for another request', async () => {
  for (const hasCompletedRequest of [false, true]) {
    const f = fixture();
    if (hasCompletedRequest) {
      const first = await f.one();
      f.reply(first.request);
      await f.publish();
      f.state.target = OTHER;
    }
    f.state.postErrors.set(1, 'before');
    await assert.rejects(f.one(), { status: 502 });
    const undelivered = f.state.checks.at(-1);
    assert.equal(undelivered.status, 'completed');
    assert.equal(undelivered.conclusion, 'cancelled');
    assert.match(undelivered.output.title, /could not be confirmed/);
    assert.equal(requests(f.state.comments.get(1) || []).length, Number(hasCompletedRequest));
    assert.deepEqual((await f.scan()).jobs, [{ number: 1, allowRequest: true }]);
    assert.equal((await f.one()).status, 'requested');
    assert.equal(f.state.posts.length, Number(hasCompletedRequest) + 1);
    assert.equal(f.state.checks.length, Number(hasCompletedRequest) + 2);
    assert.equal(f.state.checks.at(-1).status, 'in_progress');
    assert.equal(undelivered.conclusion, 'cancelled');
    if (hasCompletedRequest) assert.equal(f.state.checks[0].conclusion, 'success');
  }
});

test('ambiguous POST acceptance is recovered from the trusted comment without a second request', async () => {
  const f = fixture();
  f.state.postErrors.set(1, 'after');
  await assert.rejects(f.one(), { status: 502 });
  assert.equal(f.state.posts.length, 1);
  assert.equal((await f.one()).status, 'unchanged');
  assert.equal(f.state.posts.length, 1);
  assert.equal(f.state.checks.at(-1).status, 'in_progress');
  assert.equal(f.state.checks.at(-1).conclusion, null);
});

test('discovery selects at most 30 new requests, newest PR first on every scan', async () => {
  const candidates = Array.from({ length: 45 }, (_, index) => pull(index + 1));
  const f = fixture(candidates);
  const expected = Array.from({ length: 30 }, (_, index) => 45 - index);
  assert.deepEqual((await f.scan()).jobs, expected.map((number) => ({ number, allowRequest: true })));
  assert.deepEqual((await f.scan()).jobs, expected.map((number) => ({ number, allowRequest: true })));
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
  assert.deepEqual(result.jobs, Array.from({ length: 30 }, (_, index) => ({ number: 35 - index, allowRequest: true })));
  assert.equal(result.skipped, 15);
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
  assert.deepEqual(result.jobs, Array.from({ length: 24 }, (_, index) => ({ number: 34 - index, allowRequest: true })));
  assert.equal(f.state.warnings.length, 5);
  assert.equal(f.state.failures.length, 1);
  assert.equal(f.state.checks.length, 0);
});

test('at most 30 workers run per discovery even when a POST is accepted ambiguously', async () => {
  const f = fixture(Array.from({ length: 40 }, (_, index) => pull(index + 1)));
  const { jobs } = await f.scan();
  const numbers = jobs.map((job) => job.number);
  for (const number of numbers.slice(0, 3)) f.state.postErrors.set(number, 'after');
  const results = [];
  for (const number of numbers) results.push(await f.worker({ number }));
  assert.equal(results.filter((result) => result.status === 'requested').length, 27);
  assert.equal(results.filter((result) => result.status === 'failed').length, 3);
  assert.equal(f.state.posts.length, 30);
  assert.equal(f.state.failures.length, 3);
  assert.equal(f.state.checks.every((check) => check.status === 'in_progress' && check.conclusion === null), true);
  assert.equal((await f.worker({ number: numbers[0] })).status, 'unchanged');
  assert.equal(f.state.posts.length, 30);
});

test('worker rechecks approval, refs and request comments after discovery', async () => {
  const f = fixture([pull(1), pull(2), pull(3)]);
  assert.deepEqual((await f.scan()).jobs, [3, 2, 1].map((number) => ({ number, allowRequest: true })));
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
  assert.deepEqual(result.jobs, [{ number: 2, allowRequest: true }]);
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
    assert.equal(f.state.checks.every((check) =>
      (check.status === 'in_progress' && check.conclusion === null) ||
      (check.status === 'completed' && check.conclusion === 'cancelled')), true);
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
    assert.deepEqual((await f.scan()).jobs, [{ number: 1, allowRequest: true }]);
    assert.equal((await f.worker()).status, 'requested');
    assert.deepEqual((await f.scan()).jobs, [{ number: 1, allowRequest: true }]);
    assert.equal((await f.worker()).status, 'requested');
    assert.equal(f.state.posts.length, 2);
    for (const change of [{ state: 'closed' }, { base: { ref: 'feature/experimental' } }]) {
      f.state.prs[0] = pull(1, change);
      assert.deepEqual((await f.scan()).jobs, []);
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

test('an unanswered pair gets one automatic retry at two hours and ignores the late first reply', async () => {
  const f = fixture();
  const first = await f.one();
  f.state.now += 2 * HOUR - 1;
  assert.deepEqual((await f.scan()).jobs, []);
  f.state.now += 1;
  assert.deepEqual((await f.scan()).jobs, [{ number: 1, allowRequest: true }]);
  const retried = await f.worker();
  assert.equal(retried.status, 'requested');
  assert.notEqual(retried.request.id, first.request.id);
  assert.equal(retried.request.automaticRetryOf, first.request.id);
  assert.equal(requests(f.state.comments.get(1))[0].automaticRetryOf, first.request.id);
  f.reply(first.request);
  await f.publish();
  assert.equal(f.state.checks.at(-1).status, 'in_progress');
  assert.equal(f.state.checks.at(-1).conclusion, null);
  f.state.now += 4 * HOUR;
  assert.deepEqual((await f.scan()).jobs, []);
  assert.equal((await f.one()).status, 'unchanged');
  assert.equal(f.state.posts.length, 2);
  f.reply(retried.request, 'FAIL');
  await f.publish();
  assert.equal(f.state.checks.at(-1).status, 'completed');
  assert.equal(f.state.checks.at(-1).conclusion, 'failure');
});

test('only a valid bound reply suppresses the automatic retry', async () => {
  for (const alter of [
    (comment) => { comment.user = { ...REVIEWER, id: 999 }; },
    (comment) => { comment.body = comment.body.replace(/request_id=\S+/, 'request_id=aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'); },
    (comment) => { comment.body = comment.body.replace(`head=${HEAD}`, `head=${OTHER}`); },
    (comment) => { comment.body = comment.body.replace('verdict=PASS', 'verdict=UNKNOWN'); },
  ]) {
    const f = fixture();
    const first = await f.one();
    alter(f.reply(first.request));
    f.state.now += 2 * HOUR;
    const result = await f.worker();
    assert.equal(result.status, 'requested');
    assert.equal(result.request.automaticRetryOf, first.request.id);
    assert.equal(f.state.posts.length, 2);
    assert.equal(f.state.checks.at(-1).status, 'in_progress');
    assert.equal(f.state.checks.at(-1).conclusion, null);
  }
});

test('valid PASS, FAIL, INCONCLUSIVE and missing-evidence replies are repaired without another AI request', async () => {
  for (const [verdict, expected, stripEvidence] of [
    ['PASS', 'success', false], ['FAIL', 'failure', false],
    ['INCONCLUSIVE', 'neutral', false], ['PASS', 'neutral', true],
  ]) {
    const f = fixture();
    const first = await f.one();
    const reply = f.reply(first.request, verdict);
    if (stripEvidence) reply.body = reply.body.split('\n').slice(0, 2).join('\n');
    f.state.now += 3 * HOUR;
    const scan = await f.scan();
    assert.equal(scan.requested, 0);
    assert.deepEqual(scan.jobs, [{ number: 1, allowRequest: false }]);
    assert.equal((await f.worker({ allowRequest: false, commandGithub: undefined })).status, 'reconciled');
    assert.equal(f.state.checks.at(-1).status, 'completed');
    assert.equal(f.state.checks.at(-1).conclusion, expected);
    assert.equal(f.state.posts.length, 1);
    assert.deepEqual((await f.scan()).jobs, []);
  }
});

test('a failed Check write is retried from the reply rather than asking AI again', async () => {
  const f = fixture();
  const first = await f.one();
  f.reply(first.request, 'FAIL');
  f.state.checkUpdateFailures = 1;
  await assert.rejects(f.publish(), { status: 502 });
  assert.equal(f.state.checks.at(-1).status, 'in_progress');
  f.state.now += 3 * HOUR;
  assert.deepEqual((await f.scan()).jobs, [{ number: 1, allowRequest: false }]);
  assert.equal((await f.worker({ commandGithub: undefined })).status, 'unchanged');
  assert.equal(f.state.checks.at(-1).conclusion, 'failure');
  assert.equal(f.state.posts.length, 1);
});

test('a reply arriving after discovery or during the worker cancels the planned retry', async () => {
  for (const duringWorker of [false, true]) {
    const f = fixture();
    const first = await f.one();
    f.state.now += 3 * HOUR;
    assert.deepEqual((await f.scan()).jobs, [{ number: 1, allowRequest: true }]);
    if (duringWorker) {
      const finalRead = f.state.commentReads.get(1) + 2;
      f.state.onListComments = (number, count) => {
        if (number === 1 && count === finalRead) f.reply(first.request);
      };
    } else f.reply(first.request);
    assert.equal((await f.worker()).status, 'unchanged');
    assert.equal(f.state.posts.length, 1);
    assert.equal(f.state.checks.at(-1).conclusion, 'success');
  }
});

test('manual requests neither consume nor replenish the single automatic retry allowance', async () => {
  const f = fixture();
  const first = await f.one({ manual: true });
  assert.equal(first.request.automaticRetryOf, undefined);
  f.state.now += 2 * HOUR;
  const retry = await f.one();
  assert.equal(retry.request.automaticRetryOf, first.request.id);
  f.reply(retry.request);
  await f.publish();
  assert.equal(f.state.checks.at(-1).conclusion, 'success');
  const manual = await f.one({ manual: true });
  assert.equal(manual.request.automaticRetryOf, undefined);
  assert.notEqual(manual.request.id, retry.request.id);
  assert.notEqual(manual.request.checkId, retry.request.checkId);
  assert.equal(f.state.checks.find((check) => check.id === retry.request.checkId).conclusion, 'success');
  assert.equal(f.state.checks.at(-1).status, 'in_progress');
  assert.equal(f.state.checks.at(-1).conclusion, null);
  f.state.now += 3 * HOUR;
  assert.deepEqual((await f.scan()).jobs, []);
  assert.equal((await f.one()).status, 'unchanged');
  assert.equal(f.state.posts.length, 3);
});

test('returning to a previously superseded pair does not automatically retry its old request', async () => {
  const f = fixture();
  await f.one();
  f.state.target = OTHER;
  await f.one();
  f.state.target = TARGET;
  f.state.now += 3 * HOUR;
  assert.deepEqual((await f.scan()).jobs, []);
  assert.equal((await f.one()).status, 'unchanged');
  assert.equal(f.state.posts.length, 2);
});

test('repair recovers the recorded revisions after the current head, target or eligibility changes', async () => {
  for (const change of [
    (state) => { state.prs[0].head.sha = OTHER; },
    (state) => { state.target = OTHER; },
    (state) => { state.prs[0].labels = []; },
    (state) => { state.prs[0].draft = true; },
  ]) {
    const f = fixture();
    const first = await f.one();
    f.reply(first.request, 'FAIL');
    change(f.state);
    const scan = await f.scan();
    assert.equal(scan.jobs.length, 1);
    assert.equal(scan.jobs[0].number, 1);
    if (!f.state.prs[0].labels.length || f.state.prs[0].draft) {
      assert.equal(scan.jobs[0].allowRequest, false);
      assert.equal(scan.requested, 0);
    }
    const result = await f.worker({ allowRequest: false, commandGithub: undefined });
    assert.equal(result.status, 'reconciled');
    assert.equal(f.state.checks.at(-1).head_sha, HEAD);
    assert.equal(f.state.checks.at(-1).conclusion, 'failure');
    assert.match(f.state.checks.at(-1).output.summary, new RegExp(first.request.id));
    assert.equal(f.state.posts.length, 1);
  }
});

test('repair jobs survive a full 30-request budget and cannot upgrade to new requests', async () => {
  const f = fixture(Array.from({ length: 32 }, (_, index) => pull(index + 1)));
  for (const number of [1, 2]) {
    const first = await f.one({ number });
    f.reply(first.request, 'PASS', {}, number);
  }
  const scan = await f.scan();
  assert.equal(scan.requested, 30);
  assert.equal(scan.jobs.length, 32);
  assert.deepEqual(scan.jobs.slice(-2), [
    { number: 2, allowRequest: false }, { number: 1, allowRequest: false },
  ]);
  f.state.prs[0].head.sha = OTHER;
  for (const job of scan.jobs) {
    const result = await f.worker({ ...job, ...(!job.allowRequest ? { commandGithub: undefined } : {}) });
    assert.equal(result.status, job.allowRequest ? 'requested' : 'reconciled');
  }
  assert.equal(f.state.posts.length, 32);
  assert.equal(requests(f.state.comments.get(1)).length, 1);
  assert.equal(f.state.checks.find((check) => check.external_id.startsWith('semantic-review:1:')).conclusion, 'success');
});

test('automatic retries and new pairs share the same 30-request budget', async () => {
  const f = fixture(Array.from({ length: 40 }, (_, index) => pull(index + 1)));
  for (let number = 31; number <= 40; number += 1) await f.one({ number });
  f.state.now += 2 * HOUR;
  const scan = await f.scan();
  assert.equal(scan.requested, 30);
  assert.equal(scan.jobs.length, 30);
  for (const job of scan.jobs) assert.equal((await f.worker(job)).status, 'requested');
  assert.equal(f.state.posts.length, 40);
  assert.equal([...f.state.comments.values()].flatMap(requests)
    .filter((request) => request.automaticRetryOf).length, 10);
});

test('repair-only workers do not need a command token or consume its reserved quota', async () => {
  const f = fixture();
  const first = await f.one();
  f.reply(first.request);
  f.state.commandRemaining = 1000;
  f.state.remaining = 1000;
  const result = await f.worker({ allowRequest: false, commandGithub: undefined });
  assert.equal(result.status, 'reconciled');
  assert.equal(f.state.checks.at(-1).conclusion, 'success');
  assert.equal(f.state.posts.length, 1);
  assert.equal(f.state.commandRemaining, 1000);
});

test('the Actions matrix remains within 256 jobs when many old replies need repair', async () => {
  const f = fixture(Array.from({ length: 260 }, (_, index) => pull(index + 1)), {
    remaining: 30000, commandRemaining: 30000,
  });
  for (let number = 1; number <= 260; number += 1) {
    const first = await f.one({ number });
    f.reply(first.request, 'PASS', {}, number);
  }
  const scan = await f.scan();
  assert.equal(scan.requested, 0);
  assert.equal(scan.jobs.length, 256);
  assert.equal(scan.jobs.every((job) => job.allowRequest === false), true);
  assert.equal(scan.jobs[0].number, 260);
  assert.equal(scan.jobs.at(-1).number, 5);
});

test('a due automatic retry still preserves both token reserves', async () => {
  for (const quota of [{ remaining: 1003 }, { commandRemaining: 1001 }]) {
    const f = fixture();
    await f.one();
    f.state.now += 3 * HOUR;
    Object.assign(f.state, quota);
    assert.equal((await f.worker()).status, 'limited');
    assert.equal(f.state.posts.length, 1);
    assert.ok(f.state.remaining >= 1000);
    assert.ok(f.state.commandRemaining >= 1000);
    assert.equal(requests(f.state.comments.get(1)).some((request) => request.automaticRetryOf), false);
  }
});

test('an edited invalid result becomes retryable even though the original Check completed', async () => {
  const f = fixture();
  const first = await f.one();
  const reply = f.reply(first.request);
  await f.publish();
  assert.equal(f.state.checks.at(-1).conclusion, 'success');
  reply.body = reply.body.replace('verdict=PASS', 'verdict=UNKNOWN');
  f.state.now += 3 * HOUR;
  assert.deepEqual((await f.scan()).jobs, [{ number: 1, allowRequest: true }]);
  const retry = await f.worker();
  assert.equal(retry.status, 'requested');
  assert.equal(retry.request.automaticRetryOf, first.request.id);
  assert.equal(f.state.checks.at(-1).status, 'in_progress');
  assert.equal(f.state.checks.at(-1).conclusion, null);
  assert.equal(f.state.posts.length, 2);
});

test('a failed manual same-pair POST preserves the previous pending request and cancels the orphan Check', async () => {
  const f = fixture();
  const first = await f.one();
  const originalCheck = f.state.checks.at(-1);
  f.state.postErrors.set(1, 'before');
  await assert.rejects(f.one({ manual: true }), { status: 502 });
  const undelivered = f.state.checks.at(-1);
  assert.notEqual(undelivered.id, originalCheck.id);
  assert.equal(undelivered.status, 'completed');
  assert.equal(undelivered.conclusion, 'cancelled');
  assert.equal(originalCheck.status, 'in_progress');
  assert.equal(originalCheck.conclusion, null);
  assert.equal(requests(f.state.comments.get(1)).length, 1);
  assert.equal((await f.one()).status, 'unchanged');
  assert.equal(f.state.posts.length, 1);
  f.reply(first.request, 'FAIL');
  await f.publish();
  assert.equal(originalCheck.conclusion, 'failure');
  assert.equal(undelivered.conclusion, 'cancelled');
});

test('an ambiguously accepted automatic retry remains pending and consumes its one retry allowance', async () => {
  const f = fixture();
  const first = await f.one();
  f.state.now += 2 * HOUR;
  f.state.postErrors.set(1, 'after');
  await assert.rejects(f.one(), { status: 502 });
  const accepted = requests(f.state.comments.get(1))[0];
  assert.notEqual(accepted.id, first.request.id);
  assert.equal(accepted.automaticRetryOf, first.request.id);
  const retryCheck = f.state.checks.find((check) => check.id === accepted.checkId);
  assert.equal(retryCheck.status, 'in_progress');
  assert.equal(retryCheck.conclusion, null);
  f.state.now += 3 * HOUR;
  assert.equal((await f.one()).status, 'unchanged');
  assert.equal(f.state.posts.length, 2);
  assert.equal(requests(f.state.comments.get(1)).filter((request) => request.automaticRetryOf).length, 1);
  f.reply(first.request);
  await f.publish();
  assert.equal(retryCheck.status, 'in_progress');
  assert.equal(retryCheck.conclusion, null);
  f.reply(accepted);
  await f.publish();
  assert.equal(retryCheck.conclusion, 'success');
});

test('failed delivery readback keeps the unknown request pending until a later comment read recovers it', async () => {
  const f = fixture();
  f.state.postErrors.set(1, 'after');
  f.state.onListComments = (_number, count) => {
    if (count === 3) throw Object.assign(new Error('Readback unavailable'), { status: 503 });
  };
  await assert.rejects(f.one(), { status: 502 });
  assert.equal(f.state.posts.length, 1);
  assert.equal(f.state.checks.at(-1).status, 'in_progress');
  assert.equal(f.state.checks.at(-1).conclusion, null);
  assert.equal(f.state.warnings.length, 1);
  assert.match(f.state.warnings[0], /delivery remains unknown/);
  assert.equal((await f.one()).status, 'unchanged');
  assert.equal(f.state.posts.length, 1);
  assert.equal(f.state.checks.length, 1);
});

test('a scan repairs missed cancellation after an accepted retry finishes without requesting AI again', async () => {
  const f = fixture();
  const first = await f.one();
  const oldCheck = f.state.checks.at(-1);
  f.state.now += 2 * HOUR;
  f.state.postErrors.set(1, 'after');
  await assert.rejects(f.one(), { status: 502 });
  const retried = requests(f.state.comments.get(1))[0];
  assert.equal(retried.automaticRetryOf, first.request.id);
  assert.equal(oldCheck.status, 'in_progress');
  f.reply(retried);
  f.state.onCheckUpdate = ({ check_run_id: id }) => {
    if (id === oldCheck.id) {
      f.state.onCheckUpdate = undefined;
      throw Object.assign(new Error('Cancellation unavailable'), { status: 502 });
    }
  };
  await assert.rejects(f.publish(), { status: 502 });
  const currentCheck = f.state.checks.find((check) => check.id === retried.checkId);
  assert.equal(currentCheck.conclusion, 'success');
  assert.equal(oldCheck.status, 'in_progress');
  const scan = await f.scan();
  assert.equal(scan.requested, 0);
  assert.deepEqual(scan.jobs, [{ number: 1, allowRequest: false }]);
  assert.equal((await f.worker({ ...scan.jobs[0], commandGithub: undefined })).status, 'reconciled');
  assert.equal(oldCheck.status, 'completed');
  assert.equal(oldCheck.conclusion, 'cancelled');
  assert.equal(currentCheck.conclusion, 'success');
  assert.equal(f.state.posts.length, 2);
  assert.deepEqual((await f.scan()).jobs, []);
});

test('a scan cancels an unrecorded manual attempt without replacing the existing PASS', async () => {
  const f = fixture();
  const first = await f.one();
  f.reply(first.request);
  await f.publish();
  const completed = f.state.checks.at(-1);
  const readback = f.state.commentReads.get(1) + 3;
  f.state.postErrors.set(1, 'before');
  f.state.onListComments = (_number, count) => {
    if (count === readback) throw Object.assign(new Error('Readback unavailable'), { status: 503 });
  };
  await assert.rejects(f.one({ manual: true }), { status: 502 });
  const orphan = f.state.checks.at(-1);
  assert.notEqual(orphan.id, completed.id);
  assert.equal(orphan.status, 'in_progress');
  assert.equal(requests(f.state.comments.get(1)).length, 1);
  const scan = await f.scan();
  assert.equal(scan.requested, 0);
  assert.deepEqual(scan.jobs, [{ number: 1, allowRequest: false }]);
  await f.worker({ ...scan.jobs[0], commandGithub: undefined });
  assert.equal(orphan.conclusion, 'cancelled');
  assert.equal(completed.conclusion, 'success');
  assert.equal(f.state.posts.length, 1);
  assert.deepEqual((await f.scan()).jobs, []);
});

test('a scan finds and cancels an orphan on the current head even without a request record or approval', async () => {
  const f = fixture();
  f.state.postErrors.set(1, 'before');
  f.state.onListComments = (_number, count) => {
    if (count === 3) throw Object.assign(new Error('Readback unavailable'), { status: 503 });
  };
  await assert.rejects(f.one(), { status: 502 });
  const orphan = f.state.checks.at(-1);
  assert.equal(orphan.status, 'in_progress');
  assert.equal(requests(f.state.comments.get(1) || []).length, 0);
  f.state.prs[0].labels = [];
  const scan = await f.scan();
  assert.equal(scan.requested, 0);
  assert.deepEqual(scan.jobs, [{ number: 1, allowRequest: false }]);
  await f.worker({ ...scan.jobs[0], commandGithub: undefined });
  assert.equal(orphan.status, 'completed');
  assert.equal(orphan.conclusion, 'cancelled');
  assert.equal(f.state.posts.length, 0);
  assert.deepEqual((await f.scan()).jobs, []);
});

test('a FAIL received at the final recheck remains completed when a manual new request is sent', async () => {
  const f = fixture();
  const first = await f.one();
  const previousCheck = f.state.checks.find((check) => check.id === first.request.checkId);
  const finalRead = f.state.commentReads.get(1) + 2;
  f.state.onListComments = (number, count) => {
    if (number === 1 && count === finalRead) f.reply(first.request, 'FAIL');
  };
  const manual = await f.one({ manual: true });
  assert.equal(manual.status, 'requested');
  assert.notEqual(manual.request.id, first.request.id);
  assert.notEqual(manual.request.checkId, first.request.checkId);
  assert.equal(previousCheck.status, 'completed');
  assert.equal(previousCheck.conclusion, 'failure');
  assert.equal(f.state.checks.at(-1).status, 'in_progress');
  assert.equal(f.state.checks.at(-1).conclusion, null);
  assert.equal(f.state.posts.length, 2);
});
