// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const test = require('node:test');
const assert = require('node:assert/strict');
const {run, decide, parseCommand, splitArgs, checkOrgMember} = require('./bot_commit_check');

const HEAD = '0123456789abcdef0123456789abcdef01234567';
const OTHER = 'fedcba9876543210fedcba9876543210fedcba98';
const MEMBERS = new Set(['dev', 'tensorrt-cicd', 'trtllm-agent']);

function pullRequest({author = 'dev', headOwner = author, sha = HEAD} = {}) {
  return {
    user: {login: author},
    head: {sha, repo: headOwner === null ? null : {owner: {login: headOwner}}},
  };
}

function membership(answers = {}) {
  const asked = [];
  const isOrgMember = async login => {
    asked.push(login);
    return login in answers ? answers[login] : MEMBERS.has(login.toLowerCase());
  };
  return {asked, isOrgMember};
}

async function decideFor(body, pr, answers) {
  const {asked, isOrgMember} = membership(answers);
  return {...await decide({body, pr, isOrgMember}), asked};
}

test('splitArgs follows shlex rules', () => {
  assert.deepEqual(splitArgs('run --stage-list "A10-PyTorch-1, xxx"  --debug'),
    ['run', '--stage-list', 'A10-PyTorch-1, xxx', '--debug']);
  assert.deepEqual(splitArgs("skip --comment 'doc only'\n--commit x"),
    ['skip', '--comment', 'doc only', '--commit', 'x']);
  assert.deepEqual(splitArgs('a "b \\" c" \'\' d\\ e'), ['a', 'b " c', '', 'd e']);
  assert.deepEqual(splitArgs('--commit abc'), ['--commit abc']);
  assert.equal(splitArgs('run --stage-list "A10'), null);
  assert.equal(splitArgs('run \\'), null);
});

test('parseCommand reads the action and every --commit', () => {
  assert.deepEqual(parseCommand('/bot run'), {action: 'run', commits: []});
  assert.deepEqual(parseCommand(`/bot run --commit ${HEAD}`), {action: 'run', commits: [HEAD]});
  assert.deepEqual(parseCommand(`  /bot run --commit=${HEAD}\n`), {action: 'run', commits: [HEAD]});
  assert.deepEqual(parseCommand(`/bot run --commit a\nsome text --commit ${HEAD}`),
    {action: 'run', commits: ['a', HEAD]});
  assert.deepEqual(parseCommand('/bot run --commit'), {action: 'run', commits: ['']});
  assert.deepEqual(parseCommand(`/bot run --stage-list " --commit ${HEAD} "`), {action: 'run', commits: []});
  assert.deepEqual(parseCommand('/bot reuse-pipeline'), {action: 'reuse-pipeline', commits: []});
  assert.deepEqual(parseCommand('/bot run --stage-list "A10'), {action: null, commits: []});
});

test('kill is always allowed and needs no membership check', async () => {
  const result = await decideFor('/bot kill', pullRequest({author: 'outsider'}));
  assert.equal(result.allow, true);
  assert.deepEqual(result.asked, []);
});

test('NVIDIA member pull requests keep working without --commit', async () => {
  for (const body of ['/bot run', '/bot run --disable-fail-fast', '/bot skip --comment "x"', '/bot reuse-pipeline']) {
    assert.equal((await decideFor(body, pullRequest())).allow, true, body);
  }
  const fromOrgRepo = await decideFor('/bot run', pullRequest({headOwner: 'NVIDIA'}));
  assert.equal(fromOrgRepo.allow, true);
  assert.deepEqual(fromOrgRepo.asked, ['dev']);
  const fromOtherMember = await decideFor('/bot run', pullRequest({author: 'trtllm-agent', headOwner: 'tensorrt-cicd'}));
  assert.equal(fromOtherMember.allow, true);
  assert.deepEqual(fromOtherMember.asked, ['trtllm-agent', 'tensorrt-cicd']);
});

test('external pull requests need --commit', async () => {
  for (const body of ['/bot run', '/bot skip --comment "x"', '/bot reuse-pipeline', '/bot run --stage-list "A10']) {
    const result = await decideFor(body, pullRequest({author: 'outsider'}));
    assert.equal(result.allow, false, body);
    assert.equal(result.reason, 'external pull request without --commit');
    assert.match(result.message, /--commit <SHA>/);
    assert.ok(!result.message.includes(HEAD), 'the message must not give the head SHA');
  }
});

test('the source repository owner must be an NVIDIA member too', async () => {
  assert.equal((await decideFor('/bot run', pullRequest({headOwner: 'outsider'}))).allow, false);
  assert.equal((await decideFor('/bot run', pullRequest({headOwner: null}))).allow, false);
});

test('--commit must be the full SHA of the pull request head', async () => {
  for (const author of ['dev', 'outsider']) {
    const pr = pullRequest({author});
    const good = await decideFor(`/bot run --commit ${HEAD}`, pr);
    assert.equal(good.allow, true);
    assert.deepEqual(good.asked, []);
    assert.equal((await decideFor(`/bot run --commit=${HEAD.toUpperCase()}`, pr)).allow, true);
    assert.equal((await decideFor(`/bot skip --comment "x" --commit ${HEAD}`, pr)).allow, true);

    const moved = await decideFor(`/bot run --commit ${OTHER}`, pr);
    assert.equal(moved.allow, false);
    assert.equal(moved.reason, '--commit is not the pull request head');
    assert.ok(!moved.message.includes(HEAD.slice(0, 7)), 'the message must not give the head SHA');

    for (const body of [
      `/bot run --commit ${HEAD.slice(0, 7)}`,
      '/bot run --commit',
      '/bot run --commit ""',
      `/bot run --commit ${HEAD} --commit ${HEAD}`,
      `/bot run --commit ${OTHER} --commit ${HEAD}`,
      `/bot run --commit ${HEAD}0`,
    ]) {
      const result = await decideFor(body, pr);
      assert.equal(result.allow, false, body);
      assert.equal(result.reason, 'bad --commit value', body);
    }
  }
});

test('unknown membership is rejected with its own message', async () => {
  for (const answers of [{dev: null}, {'tensorrt-cicd': null}]) {
    const result = await decideFor('/bot run', pullRequest({headOwner: 'tensorrt-cicd'}), answers);
    assert.equal(result.allow, false);
    assert.equal(result.reason, 'could not check NVIDIA membership');
  }
  const pinned = await decideFor(`/bot run --commit ${HEAD}`, pullRequest(), {dev: null});
  assert.equal(pinned.allow, true);
});

test('checkOrgMember maps the API status', async () => {
  const warnings = [];
  const core = {warning: message => warnings.push(message)};
  const calls = [];
  const fetchWith = status => async (url, options) => {
    calls.push({url, options});
    return {status};
  };
  assert.equal(await checkOrgMember({fetchImpl: fetchWith(204), token: 't', login: 'dev', core}), true);
  assert.equal(await checkOrgMember({fetchImpl: fetchWith(404), token: 't', login: 'dev', core}), false);
  assert.equal(calls[0].url, 'https://api.github.com/orgs/NVIDIA/members/dev');
  assert.equal(calls[0].options.redirect, 'manual');
  assert.equal(calls[0].options.headers.Authorization, 'Bearer t');
  for (const status of [302, 401, 403, 500]) {
    assert.equal(await checkOrgMember({fetchImpl: fetchWith(status), token: 't', login: 'dev', core}), null);
  }
  const failing = async () => { throw new Error('network'); };
  assert.equal(await checkOrgMember({fetchImpl: failing, token: 't', login: 'dev', core}), null);
  assert.equal(await checkOrgMember({fetchImpl: fetchWith(204), token: '', login: 'dev', core}), null);
  assert.equal(warnings.length, 6);
});

function harness({body, pr, isPullRequest = true, getError = null}) {
  const comments = [];
  const failures = [];
  const warnings = [];
  const github = {rest: {
    pulls: {get: async () => {
      if (getError) throw getError;
      return {data: pr};
    }},
    issues: {createComment: async args => comments.push(args)},
  }};
  const context = {
    repo: {owner: 'NVIDIA', repo: 'TensorRT-LLM'},
    payload: {
      issue: {number: 7, pull_request: isPullRequest ? {} : undefined},
      comment: {body, html_url: 'https://example.test/comment'},
    },
  };
  const core = {
    setFailed: message => failures.push(message),
    warning: message => warnings.push(message),
    error: () => {},
  };
  const fetchImpl = async url => ({status: MEMBERS.has(url.split('/').pop()) ? 204 : 404});
  return {github, context, core, fetchImpl, comments, failures, warnings};
}

test('run comments and fails the job when it rejects', async () => {
  const h = harness({body: '/bot run', pr: pullRequest({author: 'outsider'})});
  await run({...h, orgToken: 't'});
  assert.equal(h.failures.length, 1);
  assert.equal(h.comments.length, 1);
  assert.equal(h.comments[0].issue_number, 7);
  assert.ok(!h.comments[0].body.startsWith('/bot'));
  assert.match(h.comments[0].body, /\[Link to invocation\]\(https:\/\/example\.test\/comment\)$/);
});

test('run stays quiet when the command is allowed', async () => {
  for (const [body, author] of [['/bot run', 'dev'], [`/bot run --commit ${HEAD}`, 'outsider'], ['/bot kill', 'outsider']]) {
    const h = harness({body, pr: pullRequest({author})});
    await run({...h, orgToken: 't'});
    assert.deepEqual([h.failures, h.comments], [[], []], body);
  }
});

test('run fails closed when the check itself breaks', async () => {
  const broken = harness({body: '/bot run', pr: null, getError: new Error('boom')});
  await run({...broken, orgToken: 't'});
  assert.equal(broken.failures.length, 1);
  assert.equal(broken.comments.length, 1);

  const noToken = harness({body: '/bot run', pr: pullRequest()});
  await run({...noToken, orgToken: ''});
  assert.equal(noToken.failures.length, 1);
  assert.match(noToken.comments[0].body, /could not check/);
});

test('run skips comments on issues', async () => {
  const h = harness({body: '/bot run', pr: null, isPullRequest: false, getError: new Error('must not be called')});
  await run({...h, orgToken: 't'});
  assert.deepEqual([h.failures, h.comments], [[], []]);
});
