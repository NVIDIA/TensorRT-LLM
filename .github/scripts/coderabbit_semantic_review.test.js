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
const scripts = [...workflow.matchAll(/^ {10}script: \|\n((?: {12}[^\n]*(?:\n|$)|\n)*)/gm)]
  .map(match => match[1].split('\n').map(line => line.replace(/^ {12}/, '')).join('\n'));
assert.equal(scripts.length, 2);
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const execute = new AsyncFunction('github', 'context', 'core', 'process', scripts[0]);
const publish = new AsyncFunction('github', 'context', 'core', scripts[1]);

const HEAD = 'a'.repeat(40);
const BASE = 'b'.repeat(40);
const NEW_BASE = 'c'.repeat(40);
const MERGE_BASE = 'e'.repeat(40);

/** Run the actual workflow script against an in-memory GitHub API. */
function harness(overrides = {}) {
  const pr = {number: 12, state: 'open', draft: false,
    head: {sha: HEAD}, base: {ref: 'main', sha: 'outdated-event-sha'},
    labels: [{name: 'ai: semantic-conflict'}], ...overrides};
  const prs = [pr];
  const comments = [];
  const posted = [];
  const checks = [];
  let target = BASE;
  const calls = [];
  const github = {
    rest: {
      checks: {create: async args => {checks.push(args);}},
      pulls: {
        list: 'list-pulls',
        get: async args => {
          calls.push(['get-pr', args]);
          return {data: prs.find(p => p.number === args.pull_number)};
        },
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
          return {data: {html_url: `https://github.com/example/repo/pull/${args.issue_number}#${posted.length}`}};
        },
      },
    },
    paginate: async (method, args) => {
      calls.push([method, args]);
      if (method === 'list-pulls') return prs;
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
  return {pr, prs, comments, posted, checks, calls, summaries,
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
  assert.equal(h.checks[0].conclusion, 'neutral');
  assert.equal(h.checks[0].head_sha, HEAD);
  assert.match(h.checks[0].output.summary, /No AI verdict yet/);
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
    await assert.rejects(h.run('workflow_dispatch', '12'), /retry not posted/);
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

test('a nonexistent manual retry cannot affect another PR', async () => {
  const h = harness();
  await assert.rejects(h.run('workflow_dispatch', '99'), /PR #99: retry not posted/);
  assert.deepEqual(h.posted, []);
  assert.deepEqual(h.checks, []);
});

test('a coalesced event sweeps all opted-in PRs; manual retry is limited to its PR', async () => {
  const h = harness();
  h.prs.push({...h.pr, number: 13, head: {sha: 'd'.repeat(40)}});
  await h.run();
  assert.deepEqual(h.posted.map(c => c.issue_number), [12, 13]);
  h.setTarget(NEW_BASE);
  await h.run();
  assert.deepEqual(h.posted.map(c => c.issue_number), [12, 13, 12, 13]);
  await h.run('workflow_dispatch', '13');
  assert.deepEqual(h.posted.map(c => c.issue_number), [12, 13, 12, 13, 13]);
});

test('a manual retry cannot request an unrelated PR without an existing marker', async () => {
  const h = harness();
  h.prs.push({...h.pr, number: 13});
  await h.run('workflow_dispatch', '13');
  assert.deepEqual(h.posted.map(comment => comment.issue_number), [13]);
});

/** Build the observed CodeRabbit result format with a verifiable revision record. */
function resultBody(verdict = 'FAIL', head = HEAD, target = BASE) {
  const status = {PASS: '✅ Passed', FAIL: '⚠️ Warning', INCONCLUSIVE: '❓ Inconclusive'}[verdict];
  return '<!-- pre-merge-checks-results -->\n' +
    `| Semantic Conflict With Target Branch | ${status} | Explanation preview |\n` +
    '<details>\n<summary>Full details: Semantic Conflict With Target Branch</summary>\n' +
    `SEMANTIC_RESULT head=${head} target=${target} merge_base=${MERGE_BASE} verdict=${verdict}\n` +
    'Evidence and a minimal regression input.\n</details>';
}

/** Exercise the result publisher without checking out or executing PR code. */
function resultHarness(body = resultBody()) {
  const request = harness();
  const comment = {id: 123, body, user: {login: 'coderabbitai[bot]', type: 'Bot'},
    html_url: 'https://github.com/example/repo/pull/12#issuecomment-123'};
  const comments = [comment];
  const checks = [{id: 1, app: {slug: 'github-actions'},
    external_id: `semantic-conflict:12:${HEAD}:${BASE}`}];
  const updated = [];
  const warnings = [];
  const failures = [];
  const summaries = [];
  let mergeBase = MERGE_BASE;
  let target = BASE;
  let afterCompare = () => {};
  const github = {
    rest: {
      issues: {getComment: async () => ({data: comment}), listComments: 'list-comments'},
      pulls: {get: async () => ({data: request.pr})},
      git: {getRef: async () => ({data: {object: {sha: target}}})},
      checks: {listForRef: 'list-checks', update: async args => {updated.push(args);}},
    },
    paginate: async method => method === 'list-comments' ? comments : checks,
    request: async (route, args) => {
      assert.equal(route, 'GET /repos/{owner}/{repo}/compare/{basehead}');
      assert.equal(args.basehead, `${BASE}...${HEAD}`);
      afterCompare();
      return {data: {merge_base_commit: {sha: mergeBase}}};
    },
  };
  const core = {warning: text => warnings.push(text), setFailed: text => failures.push(text),
    summary: {addRaw(text) {summaries.push(text); return this;}, async write() {}}};
  return {comment, comments, checks, updated, warnings, failures, summaries, pr: request.pr,
    setMergeBase: sha => {mergeBase = sha;},
    setTarget: sha => {target = sha;},
    afterCompare: callback => {afterCompare = callback;},
    run: (eventName = 'issue_comment') => publish(github, {repo: {owner: 'example', repo: 'repo'},
      eventName, payload: {issue: {number: 12}, comment: {id: 123},
        pull_request: {number: 12, head: {sha: HEAD}}}}, core),
  };
}

test('verified conflicts publish failure with a false-positive and non-blocking notice', async () => {
  for (const body of [resultBody('FAIL'), resultBody('FAIL').toLowerCase()]) {
    const h = resultHarness(body);
    await h.run();
    assert.equal(h.updated[0].conclusion, 'failure');
    assert.match(h.updated[0].output.title, /Possible semantic conflict/);
    assert.match(h.updated[0].output.summary, /CodeRabbit can make mistakes, including false positives/);
    assert.match(h.updated[0].output.summary, /must remain non-required; its failure does not block merging/);
    assert.equal(h.updated[0].details_url, h.comment.html_url);
    assert.equal(h.failures.length, 1);
    assert.match(h.failures[0], /does not block merging/);
    assert.match(h.summaries[0], /false positives/);
  }
});

test('inconclusive results stay neutral and warn without failing the job', async () => {
  const h = resultHarness(resultBody('INCONCLUSIVE'));
  await h.run();
  assert.equal(h.updated[0].conclusion, 'neutral');
  assert.match(h.updated[0].output.title, /inconclusive/);
  assert.equal(h.warnings.length, 1);
  assert.equal(h.failures.length, 0);
});

test('only a verified pass publishes success, including a passed table without full details', async () => {
  const body = resultBody('PASS');
  const record = body.match(/SEMANTIC_RESULT[^\n]+/)[0];
  for (const text of [body, '<!-- pre_merge_checks_walkthrough_start -->\n' +
    `| Semantic conflict with target branch | ✅ Passed | ${record} |`]) {
    const h = resultHarness(text);
    await h.run();
    assert.equal(h.updated[0].conclusion, 'success');
    assert.equal(h.warnings.length, 0);
    assert.equal(h.failures.length, 0);
  }
});

test('stale head or target results never overwrite the current check', async () => {
  for (const body of [resultBody('PASS', NEW_BASE), resultBody('PASS', HEAD, NEW_BASE)]) {
    const h = resultHarness(body);
    await h.run();
    assert.deepEqual(h.updated, []);
  }
});

test('missing, contradictory, or invalid evidence cannot publish a pass', async () => {
  for (const body of [
    resultBody('PASS').replace(/SEMANTIC_RESULT[^\n]+/, 'Clone failed'),
    resultBody('PASS').replace('✅ Passed', '❓ Inconclusive'),
    resultBody('PASS').replace('</details>', resultBody('FAIL') + '</details>'),
  ]) {
    const h = resultHarness(body);
    await h.run();
    assert.equal(h.updated[0].conclusion, 'neutral');
  }
  const h = resultHarness(resultBody('PASS'));
  h.setMergeBase(NEW_BASE);
  await h.run();
  assert.equal(h.updated[0].conclusion, 'neutral');
});

test('untrusted authors, unrelated reviews, and unrequested fixtures cannot publish results', async () => {
  for (const change of [
    h => {h.comment.user.login = 'someone-else';},
    h => {h.comment.user.type = 'User';},
    h => {h.comment.body = h.comment.body.replace('<!-- pre-merge-checks-results -->', '');},
    h => {h.comment.body = h.comment.body.replaceAll('Semantic Conflict With Target Branch', 'Controlled Experiment');},
    h => {h.checks.length = 0;},
    h => {h.checks[0].app.slug = 'another-app';},
    h => {h.pr.draft = true;},
    h => {h.pr.labels = [];},
  ]) {
    const h = resultHarness();
    change(h);
    await h.run();
    assert.deepEqual(h.updated, []);
  }
});

test('head or main updates during analysis cannot publish a stale pass', async () => {
  for (const change of [h => {h.pr.head.sha = NEW_BASE;}, h => h.setTarget(NEW_BASE)]) {
    const h = resultHarness(resultBody('PASS'));
    h.afterCompare(() => change(h));
    await h.run();
    assert.deepEqual(h.updated, []);
  }
});

test('a retry updates only the newest matching check', async () => {
  const h = resultHarness();
  h.checks.push({...h.checks[0], id: 2});
  await h.run();
  assert.equal(h.updated[0].check_run_id, 2);
});

test('PR preview uses the same verdict mapping without writing a Check, even for drafts', async () => {
  for (const verdict of ['PASS', 'FAIL', 'INCONCLUSIVE']) {
    const h = resultHarness(resultBody(verdict).toLowerCase());
    h.pr.draft = true;
    h.pr.labels = [];
    h.checks.length = 0;
    await h.run('pull_request');
    assert.deepEqual(h.updated, []);
    assert.equal(h.failures.length, verdict === 'FAIL' ? 1 : 0);
    assert.equal(h.warnings.length, verdict === 'INCONCLUSIVE' ? 1 : 0);
    assert.match(h.summaries[0], /Read-only PR preview/);
    assert.match(h.summaries[0], /non-required/);
  }
});

test('PR preview cannot reuse another SHA pair, an untrusted reply, or an older PR run', async () => {
  for (const change of [
    h => {h.comment.body = resultBody('FAIL', NEW_BASE);},
    h => {h.comment.body = resultBody('FAIL', HEAD, NEW_BASE);},
    h => {h.comment.user.login = 'someone-else';},
    h => {h.comment.user.type = 'User';},
    h => {h.comments.length = 0;},
    h => {h.pr.head.sha = NEW_BASE;},
  ]) {
    const h = resultHarness();
    change(h);
    await h.run('pull_request');
    assert.deepEqual(h.updated, []);
    assert.deepEqual(h.failures, []);
    assert.equal(h.warnings.length, 1);
  }
});

test('PR preview chooses the latest matching reply and rechecks live refs', async () => {
  const h = resultHarness(resultBody('FAIL'));
  h.comments.push({...h.comment, id: 124, body: resultBody('PASS')});
  await h.run('pull_request');
  assert.deepEqual(h.failures, []);
  assert.match(h.summaries[0], /No semantic conflict found/);
  for (const change of [h => {h.pr.head.sha = NEW_BASE;}, h => h.setTarget(NEW_BASE)]) {
    const stale = resultHarness();
    stale.afterCompare(() => change(stale));
    await stale.run('pull_request');
    assert.deepEqual(stale.updated, []);
    assert.deepEqual(stale.failures, []);
    assert.equal(stale.warnings.length, 1);
  }
});
