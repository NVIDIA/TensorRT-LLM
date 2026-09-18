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
const request = require('./coderabbit_semantic_review_request.js');
const publish = require('./coderabbit_semantic_review_result.js');
const {AUDIT, requests} = publish;
const APPROVED = 'ci: full pre-merge approved';
const HEAD = 'a'.repeat(40), BASE = 'b'.repeat(40), NEW_BASE = 'c'.repeat(40);
const MERGED = 'd'.repeat(40), MERGE_BASE = 'e'.repeat(40), TREE = 'f'.repeat(40);
const HOUR = 3600000, DAY = 24 * HOUR;
const NOW = Date.parse('2026-09-17T08:00:00Z');
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;

function resultBody(verdict, pair) {
  const status = {PASS: '✅ Passed', FAIL: '⚠️ Warning', INCONCLUSIVE: '❓ Inconclusive'}[verdict];
  return '<!-- pre-merge-checks-results -->\n' +
    `| Semantic Conflict With Target Branch | ${status} | Explanation preview |\n` +
    '<details>\n<summary>Full details: Semantic Conflict With Target Branch</summary>\n' +
    `SEMANTIC_RESULT head=${pair.head} target=${pair.target} merge_base=${pair.mergeBase} verdict=${verdict}\n` +
    (pair.merged ? `SEMANTIC_MERGED sha=${pair.merged}\n` : '') +
    `Caller: https://github.com/example/repo/blob/${pair.head}/caller.py#L12\n` +
    `Implementation: https://github.com/example/repo/blob/${pair.target}/callee.py#L25\n` +
    'Code evidence and a minimal regression input.\n</details>';
}

// Execute the production request and publication modules against an in-memory API.
function harness(overrides = {}) {
  const pr = {number: 12, state: 'open', merged: false, draft: false,
    head: {sha: HEAD}, base: {ref: 'main'}, labels: [], merge_commit_sha: MERGED,
    auto_merge: null, ...overrides};
  const prs = [pr], comments = [], checks = [], posted = [], writes = [], calls = [];
  const outputs = {}, warnings = [], failures = [], summaries = [];
  let target = BASE, now = NOW, count = 30, age = 0, progressCount = 1;
  let mergeTree = TREE, expectedMergeBase = MERGE_BASE, afterCompare = () => {};
  let rules = [{type: 'pull_request', parameters: {allowed_merge_methods: ['squash']}}];
  let mergeParents;
  const github = {
    rest: {
      pulls: {list: 'pulls', get: async args => {
        calls.push(['pull', args]);
        const found = prs.find(p => p.number === args.pull_number);
        if (!found) throw Object.assign(new Error('Not Found'), {status: 404});
        return {data: structuredClone(found)};
      }},
      git: {
        getRef: async args => { calls.push(['ref', args]); return {data: {object: {sha: target}}}; },
        getCommit: async args => { calls.push(['commit', args]); return {data: {
          sha: args.commit_sha, tree: {sha: mergeTree}, parents: mergeParents ||
            (pr.merged ? [{sha: BASE}] : [{sha: target}, {sha: pr.head.sha}]),
        }}; },
      },
      issues: {
        listComments: 'comments', getComment: async args => ({data: comments.find(c => c.id === args.comment_id)}),
        createComment: async args => {
          posted.push(args); writes.push(['comment', args]);
          const comment = {id: comments.length + 1, issue_number: args.issue_number, body: args.body,
            user: {login: 'github-actions[bot]', type: 'Bot'}, created_at: new Date(now).toISOString(),
            html_url: `https://github.com/example/repo/pull/${args.issue_number}#${comments.length + 1}`};
          comments.push(comment); return {data: comment};
        },
      },
      checks: {
        listForRef: 'checks', create: async args => {
          const check = {...args, id: checks.length + 1, app: {slug: 'github-actions'}};
          checks.push(check); writes.push(['create', args]); return {data: check};
        }, update: async args => {
          Object.assign(checks.find(c => c.id === args.check_run_id), args);
          writes.push(['update', args]); return {data: {}};
        },
      },
    },
    paginate: async (method, args) => {
      calls.push([method, args]);
      if (method === 'pulls') return prs.filter(p => p.state === args.state);
      if (method === 'comments') return comments.filter(c => (c.issue_number || 12) === args.issue_number);
      assert.equal(method, 'checks');
      return checks.filter(c => c.head_sha === args.ref && c.name === args.check_name);
    },
    request: async (route, args) => {
      calls.push([route, args]);
      if (route.endsWith('/rules/branches/{branch}')) return {data: rules};
      assert.equal(route, 'GET /repos/{owner}/{repo}/compare/{basehead}');
      const data = args.basehead.endsWith(`...${pr.head.sha}`) ?
        {behind_by: count, merge_base_commit: {sha: expectedMergeBase,
          commit: {committer: {date: new Date(NOW - age).toISOString()}}}} :
        {status: progressCount ? 'ahead' : 'identical', ahead_by: progressCount};
      afterCompare(); return {data};
    },
  };
  github.paginate.iterator = async function* () {
    yield {data: prs.filter(p => p.state === 'closed')};
  };
  const core = {setOutput: (k, v) => {outputs[k] = v;}, info() {},
    warning: v => warnings.push(v), error: v => warnings.push(v), setFailed: v => failures.push(v),
    summary: {addRaw(v) {summaries.push(v); return this;}, async write() {}}};
  const context = {repo: {owner: 'example', repo: 'repo'}, eventName: 'pull_request_target',
    payload: {action: 'opened', pull_request: {number: 12, head: {sha: HEAD}}}};
  return {pr, prs, comments, checks, posted, writes, calls, outputs, warnings, failures, summaries, github, core,
    setTarget: v => {target = v;}, setNow: v => {now = v;}, setLag: (n, a) => {count = n; age = a;},
    setProgress: v => {progressCount = v;}, setTree: v => {mergeTree = v;},
    setRules: v => {rules = v;}, setParents: v => {mergeParents = v;},
    setMergeBase: v => {expectedMergeBase = v;}, afterCompare: fn => {afterCompare = fn;},
    async run(eventName = 'pull_request_target', action = 'opened', manual = '12', authorized = false) {
      process.env.DISPATCH_PULL_NUMBER = manual;
      process.env.SEMANTIC_APPROVAL_VALIDATED = String(authorized);
      try {
        await request({github, core, now, context: {...context, eventName,
          payload: {...context.payload, action, label: {name: APPROVED}}}});
      } finally {
        delete process.env.DISPATCH_PULL_NUMBER;
        delete process.env.SEMANTIC_APPROVAL_VALIDATED;
      }
    },
    reply(verdict = 'PASS', pair = requests(comments)[0] ||
      {head: pr.head.sha, target, mergeBase: MERGE_BASE}) {
      const comment = {id: comments.length + 1, body: resultBody(verdict, pair),
        created_at: new Date(now).toISOString(), user: {login: 'coderabbitai[bot]', type: 'Bot'},
        html_url: `https://github.com/example/repo/pull/12#${comments.length + 1}`};
      comments.push(comment); return comment;
    },
    publish: (comment, preview = false) => publish({github, core, context: {
      ...context, eventName: preview ? 'pull_request' : 'issue_comment',
      payload: {...context.payload, issue: {number: 12}, comment: {id: comment?.id}},
    }}),
  };
}

for (const [count, age, expected] of [[0, 3 * DAY, 0], [29, DAY - 1, 0],
  [30, 0, 1], [1, DAY, 1], [29, DAY, 1]]) {
  test(`first analysis: ${count} target commits, age ${age} => ${expected} requests`, async () => {
    const h = harness(); h.setLag(count, age); await h.run();
    assert.equal(h.posted.length, expected);
    assert.equal(h.checks[0].conclusion, 'neutral');
    if (expected) {
      assert.match(h.posted[0].body, /^@coderabbitai evaluate custom pre-merge check/);
      assert.match(h.posted[0].body, /--mode warning/);
      assert.match(h.posted[0].body, /No AI verdict is asserted/);
      assert.ok(h.posted[0].body.includes(HEAD) && h.posted[0].body.includes(BASE));
    }
  });
}

test('creation and ready events use the threshold; drafts and unrelated targets are excluded', async () => {
  for (const action of ['opened', 'ready_for_review']) {
    const h = harness(); h.setLag(1, 0); await h.run('pull_request_target', action);
    assert.equal(h.posted.length, 0);
  }
  for (const overrides of [{draft: true}, {state: 'closed'}, {base: {ref: 'feature/a'}}]) {
    const h = harness(overrides); await h.run();
    assert.equal(h.writes.length, 0);
    await assert.rejects(h.run('workflow_dispatch'), /requires a non-draft open or merged/);
  }
});

test('release PRs use their actual target and need no opt-in label', async () => {
  const h = harness({base: {ref: 'release/1.2'}}); await h.run();
  assert.equal(h.posted.length, 1);
  assert.ok(h.calls.filter(([kind]) => kind === 'ref').every(([, args]) => args.ref === 'heads/release/1.2'));
  const reply = h.reply(); await h.publish(reply);
  assert.equal(h.checks[0].conclusion, 'success');
});

test('an hourly scan visits eligible PRs; a PR event or manual retry visits only its PR', async () => {
  const h = harness(); h.prs.push({...h.pr, number: 13}); await h.run();
  assert.deepEqual(h.posted.map(c => c.issue_number), [12]);
  await h.run('schedule'); assert.deepEqual(h.posted.map(c => c.issue_number), [12, 13]);
  await h.run('workflow_dispatch', '', '13');
  assert.deepEqual(h.posted.map(c => c.issue_number), [12, 13, 13]);
});

test('dispatch rejects malformed or missing PRs without touching other PRs', async () => {
  for (const raw of ['', '0', '-1', '12x', '9007199254740992']) {
    const h = harness(); await assert.rejects(h.run('workflow_dispatch', '', raw), /positive integer/);
    assert.equal(h.writes.length, 0);
  }
  const h = harness(); await assert.rejects(h.run('workflow_dispatch', '', '99'), /Not Found/);
  assert.equal(h.writes.length, 0);
  await assert.rejects(h.run('push'), /Unsupported event/);
});

test('subsequent thresholds use the last completed analysis, not the original old base', async () => {
  const h = harness(); h.setLag(80, 3 * DAY); await h.run(); h.reply();
  h.setTarget(NEW_BASE); h.setNow(NOW + 2 * HOUR); h.setProgress(1); await h.run('schedule');
  assert.equal(h.posted.length, 1);
  assert.equal(h.checks[0].conclusion, 'neutral');
  assert.match(h.checks[0].output.title, /stale/);
  h.setProgress(30); await h.run('schedule'); assert.equal(h.posted.length, 2);
});

test('24 hours with new target commits triggers, while an unchanged target never does', async () => {
  for (const [progress, expected] of [[0, 1], [1, 2]]) {
    const h = harness(); await h.run(); h.reply(); h.setNow(NOW + DAY);
    h.setTarget(NEW_BASE); h.setProgress(progress); await h.run('schedule');
    assert.equal(h.posted.length, expected);
  }
});

test('a PR head update invalidates the verdict without bypassing the target threshold', async () => {
  const h = harness(); await h.run(); await h.publish(h.reply());
  h.pr.head.sha = MERGED; h.setNow(NOW + 2 * HOUR); h.setProgress(0); await h.run();
  assert.equal(h.posted.length, 1);
  assert.equal(h.checks.at(-1).head_sha, MERGED);
  assert.equal(h.checks.at(-1).conclusion, 'neutral');
});

test('approval and auto-merge bypass thresholds but share a one-hour cooldown across SHAs', async () => {
  const h = harness({labels: [{name: APPROVED}], auto_merge: {enabled_by: 'maintainer'}});
  h.setLag(0, 0); await h.run('pull_request_target', 'labeled', '12', true);
  assert.equal(h.posted.length, 1);
  h.setTarget(NEW_BASE); h.setNow(NOW + HOUR - 1);
  await h.run('pull_request_target', 'auto_merge_enabled');
  assert.equal(h.posted.length, 1); assert.equal(h.checks.at(-1).conclusion, 'neutral');
  h.setNow(NOW + HOUR); await h.run('pull_request_target', 'auto_merge_enabled');
  assert.equal(h.posted.length, 2);
});

test('an unvalidated approval label cannot bypass the threshold', async () => {
  const h = harness({labels: [{name: APPROVED}]}); h.setLag(0, 0);
  await h.run('pull_request_target', 'labeled'); assert.equal(h.posted.length, 0);
});

test('pending exact pairs deduplicate, manual retry bypasses cooldown and clears a pass', async () => {
  const h = harness(); await h.run(); await h.run(); assert.equal(h.posted.length, 1);
  await h.publish(h.reply()); assert.equal(h.checks[0].conclusion, 'success');
  await h.run('workflow_dispatch'); assert.equal(h.posted.length, 2);
  assert.equal(h.checks[0].conclusion, 'neutral');
});

test('missing analysis cannot repeatedly spend AI calls as the target changes each hour', async () => {
  const h = harness(); await h.run(); h.setNow(NOW + 2 * HOUR); h.setTarget(NEW_BASE);
  await h.run('schedule'); assert.equal(h.posted.length, 1);
});

test('untrusted or malformed request markers cannot suppress analysis', async () => {
  const h = harness(); await h.run(); h.comments[0].user.login = 'someone-else';
  h.comments.push({id: 100, user: {login: 'github-actions[bot]', type: 'Bot'},
    body: '<!-- semantic-request-v2:{malformed} -->'});
  await h.run(); assert.equal(h.posted.length, 2);
});

for (const verdict of ['PASS', 'FAIL', 'INCONCLUSIVE']) {
  test(`verified ${verdict} maps to the advisory conclusion and notices`, async () => {
    const h = harness(); await h.run(); const reply = h.reply(verdict);
    reply.body = reply.body.toLowerCase(); await h.publish(reply);
    assert.equal(h.checks[0].conclusion, {PASS: 'success', FAIL: 'failure', INCONCLUSIVE: 'neutral'}[verdict]);
    assert.equal(h.failures.length, verdict === 'FAIL' ? 1 : 0);
    assert.match(h.checks[0].output.summary, /false positives/);
    assert.match(h.checks[0].output.summary, /non-required; its failure does not block merging/);
  });
}

test('malformed, contradictory, stale and untrusted results never publish a pass', async () => {
  for (const corrupt of [
    c => {c.user.login = 'attacker';}, c => {c.user.type = 'User';},
    c => {c.body = c.body.replace('<!-- pre-merge-checks-results -->', '');},
    c => {c.body = c.body.replace('✅ Passed', '❓ Inconclusive');},
    c => {c.body = c.body.replaceAll(HEAD, NEW_BASE);},
    c => {c.body = c.body.replaceAll(MERGE_BASE, NEW_BASE);},
    c => {c.body = c.body.replace('</details>', c.body.replace('PASS', 'FAIL') + '</details>');},
  ]) {
    const h = harness(); await h.run(); const reply = h.reply(); corrupt(reply); await h.publish(reply);
    assert.equal(h.checks[0].conclusion, 'neutral');
  }
  const h = harness(); await h.run(); const reply = h.reply(); h.comments.shift();
  await h.publish(reply); assert.equal(h.checks[0].conclusion, 'neutral');
});

test('unsupported PASS/FAIL becomes inconclusive instead of reusing an older verdict', async () => {
  for (const verdict of ['PASS', 'FAIL']) {
    for (const removeEvidence of [
      body => body.replace(/https:\/\/github\.com\/example\/repo\/blob\/\S+/g, ''),
      body => body.replace(`/blob/${BASE}/`, `/blob/${NEW_BASE}/`),
      body => body.replaceAll('/example/repo/blob/', '/unrelated/repo/blob/'),
      body => body.replace(/#L\d+/g, ''),
    ]) {
      const h = harness(); await h.run(); await h.publish(h.reply('PASS'));
      const reply = h.reply(verdict); reply.body = removeEvidence(reply.body);
      await h.publish(reply);
      assert.equal(h.checks[0].conclusion, 'neutral');
      assert.match(h.checks[0].output.summary, /source links.*both revisions/);
      assert.equal(h.failures.length, 0);
      await h.publish(null, true);
      assert.equal(h.outputs.verdict, 'INCONCLUSIVE');
    }
  }
});

test('a result after its explanation is accepted with immutable citations', async () => {
  const h = harness(); await h.run(); const reply = h.reply('FAIL');
  const record = reply.body.match(/SEMANTIC_RESULT[^\n]+\n/)[0];
  reply.body = reply.body.replace(record, '').replace('</details>', `${record}</details>`);
  await h.publish(reply);
  assert.equal(h.checks[0].conclusion, 'failure');
});

test('live ref changes during verification cannot publish a current pass', async () => {
  for (const change of [h => {h.pr.head.sha = NEW_BASE;}, h => h.setTarget(NEW_BASE)]) {
    const h = harness(); await h.run(); const reply = h.reply();
    h.afterCompare(() => change(h)); await h.publish(reply);
    assert.equal(h.checks[0].conclusion, 'neutral');
  }
});

test('post-merge audit ignores thresholds and cooldown and pins the historical target', async () => {
  const h = harness({merged: true, state: 'closed', base: {ref: 'release/1.2'}});
  h.setTarget(NEW_BASE); h.setLag(0, 0); await h.run('pull_request_target', 'closed');
  const pair = requests(h.comments)[0];
  assert.equal(pair.target, BASE); assert.equal(pair.merged, MERGED);
  assert.equal(h.checks[0].name, AUDIT); assert.equal(h.checks[0].head_sha, MERGED);
  assert.ok(!h.calls.some(([kind]) => kind === 'ref'));
  const reply = h.reply('FAIL'); await h.publish(reply);
  assert.equal(h.checks[0].conclusion, 'failure');
  assert.match(h.posted.at(-1).body, /Post-merge semantic audit: FAIL/);
  await h.publish(reply); assert.equal(h.posted.length, 2); // No duplicate audit receipt.
});

test('a matching pre-merge result and actual tree can serve the audit without another AI call', async () => {
  const h = harness(); await h.run(); const reply = h.reply(); await h.publish(reply);
  h.pr.merged = true; h.pr.state = 'closed'; h.setTarget(NEW_BASE);
  await h.run('pull_request_target', 'closed');
  assert.equal(h.posted.filter(c => c.body.startsWith('@coderabbitai')).length, 1);
  assert.equal(h.checks.at(-1).name, AUDIT); assert.equal(h.checks.at(-1).conclusion, 'success');
});

test('a matching pre-merge request in flight can complete the audit after merge', async () => {
  const h = harness(); await h.run(); const original = requests(h.comments)[0];
  h.pr.merged = true; h.pr.state = 'closed'; await h.run('pull_request_target', 'closed');
  assert.equal(h.posted.length, 1); assert.equal(h.checks.at(-1).conclusion, 'neutral');
  await h.publish(h.reply('PASS', original)); assert.equal(h.checks.at(-1).conclusion, 'success');
});

test('a changed final tree, changed target or inconclusive pre-merge result requires fresh audit', async () => {
  for (const scenario of ['tree', 'target', 'inconclusive']) {
    const h = harness(); await h.run(); h.reply(scenario === 'inconclusive' ? 'INCONCLUSIVE' : 'PASS');
    h.pr.merged = true; h.pr.state = 'closed';
    if (scenario === 'tree') h.setTree(NEW_BASE);
    if (scenario === 'target') h.setParents([{sha: NEW_BASE}]);
    await h.run('pull_request_target', 'closed');
    assert.equal(h.posted.length, 2);
    assert.equal(requests(h.comments)[0].merged, MERGED);
  }
});

test('audit refuses an unverified merge method and an audit reply for another merged commit', async () => {
  const h = harness({merged: true, state: 'closed'}); h.setRules([]);
  await assert.rejects(h.run('pull_request_target', 'closed'), /squash-only/);
  assert.equal(h.posted.length, 0);
  h.setParents([{sha: BASE}, {sha: HEAD}]); await h.run('pull_request_target', 'closed');
  const reply = h.reply(); reply.body = reply.body.replace(`sha=${MERGED}`, `sha=${NEW_BASE}`);
  await h.publish(reply); assert.equal(h.checks[0].conclusion, 'neutral');
});

test('read-only draft preview accepts only a verified current pair and never writes', async () => {
  for (const verdict of ['PASS', 'FAIL', 'INCONCLUSIVE']) {
    const h = harness({draft: true}); const reply = h.reply(verdict); await h.publish(reply, true);
    assert.equal(h.outputs.verdict, verdict); assert.equal(h.writes.length, 0);
    assert.equal(h.failures.length, 0); assert.match(h.outputs.summary, /Verified head/);
  }
  const h = harness({draft: true}); h.reply('PASS'); h.setTarget(NEW_BASE); await h.publish(null, true);
  assert.equal(h.outputs.verdict, 'INCONCLUSIVE'); assert.equal(h.writes.length, 0);
});

test('the real preview display script fails for conflicts and keeps the advisory notice', async () => {
  const text = fs.readFileSync(path.join(__dirname, '../workflows/coderabbit-semantic-review-tests.yml'), 'utf8');
  const script = [...text.matchAll(/^ {10}script: \|\n((?: {12}[^\n]*(?:\n|$)|\n)*)/gm)].at(-1)[1]
    .replace(/^ {12}/gm, '');
  const display = new AsyncFunction('core', 'process', script);
  for (const verdict of ['PASS', 'FAIL']) {
    const h = harness({draft: true}); await h.publish(h.reply(verdict), true);
    await display(h.core, {env: {SEMANTIC_VERDICT: h.outputs.verdict,
      SEMANTIC_MESSAGE: h.outputs.message, SEMANTIC_SUMMARY: h.outputs.summary}});
    assert.equal(h.failures.length, verdict === 'FAIL' ? 1 : 0);
    assert.match(h.summaries.at(-1), /false positives/);
  }
});

test('a manual retry cannot be satisfied by an older comment event', async () => {
  const h = harness(); await h.run(); const oldReply = h.reply(); await h.publish(oldReply);
  h.setNow(NOW + HOUR); await h.run('workflow_dispatch'); await h.publish(oldReply);
  assert.equal(h.checks[0].conclusion, 'neutral');
  await h.publish(h.reply('FAIL')); assert.equal(h.checks[0].conclusion, 'failure');
});

test('a new audit request cannot be overwritten by an older pre-merge result', async () => {
  const h = harness(); await h.run(); const oldReply = h.reply('INCONCLUSIVE');
  h.pr.merged = true; h.pr.state = 'closed'; await h.run('pull_request_target', 'closed');
  await h.publish(h.reply('FAIL')); oldReply.body = oldReply.body.replaceAll('INCONCLUSIVE', 'PASS')
    .replace('❓ Inconclusive', '✅ Passed');
  await h.publish(oldReply); assert.equal(h.checks.at(-1).conclusion, 'failure');
});

test('the hourly scan recovers recent merged PRs without auditing old closed history', async () => {
  const h = harness({merged: true, state: 'closed', merged_at: new Date(NOW - HOUR).toISOString(),
    updated_at: new Date(NOW).toISOString()});
  h.prs.push({...h.pr, number: 13, merged_at: new Date(NOW - 2 * DAY).toISOString()});
  await h.run('schedule'); assert.deepEqual(h.posted.map(c => c.issue_number), [12]);
});

test('a manual retry reports when refs change before the request is posted', async () => {
  const h = harness(); h.afterCompare(() => h.setTarget(NEW_BASE));
  await assert.rejects(h.run('workflow_dispatch'), /Manual retry not posted/);
  assert.equal(h.posted.length, 0);
});

test('crossing the time threshold updates a previously waiting Check to awaiting analysis', async () => {
  const h = harness(); h.setLag(1, 0); await h.run();
  h.setNow(NOW + DAY); await h.run();
  assert.equal(h.checks.length, 1); assert.equal(h.posted.length, 1);
  assert.equal(h.checks[0].output.title, 'Awaiting CodeRabbit analysis');
});

test('approval validation checks the current label, actor and active membership', async () => {
  const workflow = fs.readFileSync(path.join(__dirname, '../workflows/coderabbit-semantic-review.yml'), 'utf8');
  const script = [...workflow.matchAll(/^ {10}script: \|\n((?: {12}[^\n]*(?:\n|$)|\n)*)/gm)][0][1]
    .replace(/^ {12}/gm, '');
  const validate = new AsyncFunction('github', 'context', 'core', script);
  for (const [label, actor, membership, expected] of [
    [true, 'maintainer', 'active', true], [false, 'maintainer', 'active', false],
    [true, 'different-user', 'active', false], [true, 'maintainer', 'pending', false],
    [true, 'maintainer', 404, false],
  ]) {
    const github = {rest: {
      pulls: {get: async () => ({data: {number: 12, labels: label ? [{name: APPROVED}] : []}})},
      issues: {listEventsForTimeline: 'timeline'},
      teams: {getMembershipForUser: async args => {
        assert.equal(args.username, 'maintainer'); assert.equal(args.team_slug, 'trt-llm-ci-approvers');
        if (membership === 404) throw Object.assign(new Error('Not Found'), {status: 404});
        return {data: {state: membership}};
      }},
    }, paginate: async () => [{event: 'labeled', label: {name: APPROVED}, actor: {login: actor}}]};
    const actual = await validate(github, {repo: {owner: 'example', repo: 'repo'},
      payload: {pull_request: {number: 12}, sender: {login: 'maintainer'}}}, {warning() {}});
    assert.equal(actual, expected);
  }
});
