// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const test = require('node:test');
const assert = require('node:assert/strict');
const {decide, isBot, isBotCommitter, run} = require('./stale_pr_feedback');
const DAY = 86400000;
const at = days => new Date(Date.UTC(2026, 8, 1) + days * DAY).toISOString();
const user = {login: 'author', type: 'User'};
const bot = {login: 'svc-trtllm-gh-bot', type: 'User'};
const maintainer = {login: 'reviewer', type: 'User'};
function snapshot() {
  return {
    issue: {number: 14488, state: 'open', pull_request: {}, user: user, created_at: at(0),
      updated_at: at(25), labels: [{name: 'stale'}, {name: 'waiting for feedback'}]},
    timeline: [{event: 'labeled', label: {name: 'stale'}, actor: bot, created_at: at(10)}],
    comments: [{user: bot, created_at: at(10), body: '<!-- assessment-head:abc -->'}],
    reviews: [], reviewComments: [], head: 'abc',
  };
}
for (const login of ['coderabbitai[bot]', 'trtllm-agent', 'svc-trtllm-gh-bot', 'tensorrt-cicd', 'github-actions[bot]']) {
  test(`ignore ${login} comments and label changes`, () => {
    const s = snapshot();
    s.comments.push({user: {login}, created_at: at(12), updated_at: at(13)});
    s.timeline.push({event: 'labeled', label: {name: 'Community want to contribute'}, actor: {login}, created_at: at(12)});
    assert.equal(decide(s, Date.parse(at(13))), 'skip');
    assert.equal(decide(s, Date.parse(at(24))), 'close');
  });
}
test('GitHub Bot type and case-insensitive service logins', () => {
  assert.ok(isBot({type: 'Bot', login: 'new-app'}));
  assert.ok(isBot({login: 'TRTLLM-AGENT'}));
  assert.equal(isBot(user), false);
});
for (const collection of ['comments', 'reviews', 'reviewComments']) {
  test(`${collection}: human response lifts stale`, () => {
    const s = snapshot();
    s[collection].push({user, created_at: at(12), submitted_at: at(12)});
    assert.equal(decide(s, Date.parse(at(30))), 'unstale');
  });
}
test('human commit lifts stale', () => {
  const s = snapshot();
  s.timeline.push({event: 'committed', author: user, committer: {date: at(12)}});
  assert.equal(decide(s, Date.parse(at(30))), 'unstale');
});
test('human edit to an older comment lifts stale', () => {
  const s = snapshot();
  s.comments.push({user, created_at: at(3), updated_at: at(11)});
  assert.equal(decide(s, Date.parse(at(12))), 'unstale');
});
test('restored stale label starts a new full two-week window', () => {
  const s = snapshot();
  s.timeline.push({event: 'unlabeled', label: {name: 'stale'}, created_at: at(11)},
    {event: 'labeled', label: {name: 'stale'}, created_at: at(20)});
  assert.equal(decide(s, Date.parse(at(33))), 'skip');
  assert.equal(decide(s, Date.parse(at(34))), 'close');
});
test('missing stale timestamp fails safely', () => {
  const s = snapshot(); s.timeline = [];
  assert.throws(() => decide(s, Date.parse(at(30))), /no corresponding/);
});
test('mark inactive waiting PR, ignoring bot chatter', () => {
  const s = snapshot(); s.issue.labels = [{name: 'waiting for feedback'}]; s.timeline = [];
  s.comments.push({user: bot, created_at: at(19)});
  assert.equal(decide(s, Date.parse(at(20))), 'mark');
  s.comments.push({user, created_at: at(19)});
  assert.equal(decide(s, Date.parse(at(20))), 'skip');
});
test('closed PRs and issues are excluded', () => {
  const s = snapshot(); s.issue.state = 'closed';
  assert.equal(decide(s, Date.parse(at(40))), 'skip');
  s.issue.state = 'open'; delete s.issue.pull_request;
  assert.equal(decide(s, Date.parse(at(40))), 'skip');
});
function client(s, change = false) {
  const writes = []; let gets = 0;
  const issues = {
    listForRepo: async () => [s.issue], get: async () => ({data: {...s.issue, updated_at: change && gets++ ? at(31) : at(25)}}),
    listEventsForTimeline: async () => s.timeline, listComments: async () => s.comments,
  };
  const pulls = {get: async () => ({data: {head: {sha: s.head}}}),
    listReviews: async () => s.reviews, listReviewComments: async () => s.reviewComments};
  for (const name of ['removeLabel', 'addLabels', 'createComment']) issues[name] = async args => writes.push([name, args]);
  pulls.update = async args => writes.push(['close', args]);
  return {github: {rest: {issues, pulls}, paginate: async (fn, args) => fn(args)}, writes};
}
for (const change of [false, true]) {
  test(`publication rechecks activity; concurrent change=${change}`, async () => {
    const {github, writes} = client(snapshot(), change);
    const errors = [];
    await run({github, context: {repo: {owner: 'NVIDIA', repo: 'TensorRT-LLM'}},
      core: {info() {}, setFailed: e => errors.push(e)}, now: Date.parse(at(24))});
    assert.deepEqual(errors, []);
    assert.deepEqual(writes.map(w => w[0]), change ? [] : ['close', 'createComment']);
  });
}
test('API read failure produces no writes', async () => {
  const {github, writes} = client(snapshot()); const errors = [];
  github.rest.issues.listEventsForTimeline = async () => {throw new Error('API failure');};
  await run({github, context: {repo: {}}, core: {info() {}, setFailed: e => errors.push(e)}, now: Date.parse(at(30))});
  assert.equal(errors.length, 1); assert.deepEqual(writes, []);
});

test('new feedback request gets 14 days before marking stale', () => {
  const s = snapshot(); s.issue.labels = [{name: 'waiting for feedback'}];
  s.timeline = [{event: 'labeled', label: {name: 'waiting for feedback'}, actor: user, created_at: at(20)}];
  assert.equal(decide(s, Date.parse(at(21))), 'skip');
  assert.equal(decide(s, Date.parse(at(34))), 'mark');
});
for (const action of ['mark', 'unstale']) {
  test(`publication applies ${action} transition`, async () => {
    const s = snapshot();
    if (action === 'mark') {s.issue.labels = [{name: 'waiting for feedback'}]; s.timeline = [];}
    else s.comments.push({user, created_at: at(12)});
    const {github, writes} = client(s); const errors = [];
    await run({github, context: {repo: {}}, core: {info() {}, setFailed: e => errors.push(e)}, now: Date.parse(at(25))});
    assert.deepEqual(errors, []);
    assert.deepEqual(writes.map(w => w[0]), action === 'mark'
      ? ['createComment', 'addLabels'] : ['removeLabel', 'removeLabel', 'addLabels']);
  });
}

test('old-dated commits pushed after triage still lift stale via head change', () => {
  const s = snapshot(); s.head = 'def';
  s.comments.push({user: bot, created_at: at(10), body: '<!-- assessment-head:abc -->'});
  assert.equal(decide(s, Date.parse(at(30))), 'unstale');
  s.head = 'abc';
  assert.equal(decide(s, Date.parse(at(30))), 'close');
});

test('legacy stale PR with a newly pushed old-dated commit cannot close', () => {
  const s = snapshot(); s.comments = []; s.head = 'def';
  s.timeline.push({event: 'committed', author: user, committer: {date: at(5)}});
  assert.equal(decide(s, Date.parse(at(30))), 'baseline');
});
test('legacy baseline grants full 14 days, survives retries, and detects later pushes', () => {
  const s = snapshot(); s.comments = [{user: {login: 'github-actions[bot]'}, created_at: at(30),
    body: `<!-- stale-baseline:${Date.parse(at(10))} head:abc -->`}];
  assert.equal(decide(s, Date.parse(at(30))), 'skip');
  assert.equal(decide(s, Date.parse(at(43))), 'skip');
  assert.equal(decide(s, Date.parse(at(44))), 'close');
  s.head = 'def';
  assert.equal(decide(s, Date.parse(at(31))), 'unstale');
});
test('baseline from another stale cycle or an untrusted user is not accepted', () => {
  const s = snapshot(); s.comments = [{user: {login: 'github-actions[bot]'}, created_at: at(30),
    body: `<!-- stale-baseline:${Date.parse(at(1))} head:abc -->`}];
  assert.equal(decide(s, Date.parse(at(45))), 'baseline');
  s.comments[0] = {user, created_at: at(5), body: '<!-- assessment-head:abc -->'};
  assert.equal(decide(s, Date.parse(at(45))), 'baseline');
});
test('human response takes priority over legacy baseline creation', () => {
  const s = snapshot(); s.comments = [{user, created_at: at(12)}];
  assert.equal(decide(s, Date.parse(at(30))), 'unstale');
});
test('legacy migration only posts a baseline and never closes or relabels', async () => {
  const s = snapshot(); s.comments = [];
  const {github, writes} = client(s); const errors = [];
  await run({github, context: {repo: {}}, core: {info() {}, setFailed: e => errors.push(e)}, now: Date.parse(at(30))});
  assert.deepEqual(errors, []);
  assert.deepEqual(writes.map(w => w[0]), ['createComment']);
  assert.ok(writes[0][1].body.includes(`<!-- stale-baseline:${Date.parse(at(10))} head:abc -->`));
});
test('failed baseline write never falls through to closing', async () => {
  const s = snapshot(); s.comments = [];
  const {github, writes} = client(s); const errors = [];
  github.rest.issues.createComment = async () => {throw Error('write failed');};
  await run({github, context: {repo: {}}, core: {info() {}, setFailed: e => errors.push(e)}, now: Date.parse(at(30))});
  assert.equal(errors.length, 1); assert.deepEqual(writes, []);
});

for (const collection of ['comments', 'reviews', 'reviewComments']) {
  test(`${collection}: another human's response lifts only stale`, () => {
    const s = snapshot();
    s[collection].push({user: maintainer, created_at: at(12), submitted_at: at(12)});
    assert.equal(decide(s, Date.parse(at(30))), 'lift');
  });
}
test('publication applies lift: stale removed, waiting for feedback kept', async () => {
  const s = snapshot(); s.comments.push({user: maintainer, created_at: at(12)});
  const {github, writes} = client(s); const errors = [];
  await run({github, context: {repo: {}}, core: {info() {}, setFailed: e => errors.push(e)}, now: Date.parse(at(25))});
  assert.deepEqual(errors, []);
  assert.deepEqual(writes, [['removeLabel', {issue_number: 14488, name: 'stale'}]]);
});
for (const event of ['cross-referenced', 'review_requested', 'assigned', 'milestoned', 'renamed']) {
  test(`${event} by a human is bookkeeping, not a response`, () => {
    const s = snapshot();
    s.timeline.push({event, actor: maintainer, created_at: at(12), source: {issue: {number: 1}}});
    assert.equal(decide(s, Date.parse(at(13))), 'skip');
    assert.equal(decide(s, Date.parse(at(24))), 'close');
    s.issue.labels = [{name: 'waiting for feedback'}];
    assert.equal(decide(s, Date.parse(at(24))), 'mark');
  });
}
test('force push, reopen and ready-for-review count by actor', () => {
  for (const event of ['head_ref_force_pushed', 'reopened', 'ready_for_review']) {
    const s = snapshot();
    s.timeline.push({event, actor: user, created_at: at(12)});
    assert.equal(decide(s, Date.parse(at(30))), 'unstale', event);
    s.timeline.at(-1).actor = maintainer;
    assert.equal(decide(s, Date.parse(at(30))), 'lift', event);
    s.timeline.at(-1).actor = {login: 'github-actions[bot]', type: 'Bot'};
    assert.equal(decide(s, Date.parse(at(30))), 'close', event);
  }
});
test('bot-authored commits do not count; unknown git identities do', () => {
  assert.ok(isBotCommitter({name: 'github-actions[bot]', email: '41898282+github-actions[bot]@users.noreply.github.com'}));
  assert.ok(isBotCommitter({name: 'svc-trtllm-gh-bot', email: 'x@nvidia.com'}));
  assert.equal(isBotCommitter({name: 'Jane Doe', email: 'jane@example.com'}), false);
  const s = snapshot();
  const botGit = {name: 'github-actions[bot]', email: 'github-actions[bot]@users.noreply.github.com', date: at(12)};
  s.timeline.push({event: 'committed', author: botGit, committer: botGit});
  assert.equal(decide(s, Date.parse(at(24))), 'close');
  s.timeline.push({event: 'committed', author: {name: 'Unknown', email: 'u@example.com'}, committer: botGit});
  assert.equal(decide(s, Date.parse(at(24))), 'unstale');
});
test('second stale cycle uses the marker from the latest cycle', () => {
  const s = snapshot();
  s.timeline.push({event: 'unlabeled', label: {name: 'stale'}, actor: user, created_at: at(11)},
    {event: 'labeled', label: {name: 'stale'}, actor: bot, created_at: at(20)});
  s.comments.push({user: {login: 'github-actions[bot]'}, created_at: at(20), body: '<!-- stale-head:def -->'});
  s.head = 'def';
  assert.equal(decide(s, Date.parse(at(33))), 'skip');
  assert.equal(decide(s, Date.parse(at(34))), 'close');
  s.head = 'abc';
  assert.equal(decide(s, Date.parse(at(34))), 'unstale');
});
test('mark with a failed label write is retried and the retry marker wins', async () => {
  const s = snapshot(); s.issue.labels = [{name: 'waiting for feedback'}]; s.timeline = [];
  s.comments = [{user: {login: 'github-actions[bot]'}, created_at: at(19), body: '<!-- stale-head:old -->'}];
  const {github, writes} = client(s); const errors = [];
  await run({github, context: {repo: {}}, core: {info() {}, setFailed: e => errors.push(e)}, now: Date.parse(at(20))});
  assert.deepEqual(errors, []);
  assert.deepEqual(writes.map(w => w[0]), ['createComment', 'addLabels']);
  s.comments.push({user: {login: 'github-actions[bot]'}, created_at: at(20), body: writes[0][1].body});
  s.issue.labels.push({name: 'stale'});
  s.timeline.push({event: 'labeled', label: {name: 'stale'}, actor: bot, created_at: at(20)});
  assert.equal(decide(s, Date.parse(at(33))), 'skip');
  assert.equal(decide(s, Date.parse(at(34))), 'close');
});
test('dry run logs decisions and writes nothing', async () => {
  const {github, writes} = client(snapshot()); const errors = [], logs = [];
  await run({github, context: {repo: {}}, core: {info: m => logs.push(m), setFailed: e => errors.push(e)},
    now: Date.parse(at(24)), dryRun: true});
  assert.deepEqual(errors, []); assert.deepEqual(writes, []);
  assert.ok(logs.some(m => m.includes('close') && m.includes('dry run')));
});
