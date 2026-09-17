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

// Shared by the trusted publisher and read-only PR preview.
module.exports = async ({github, context, core}) => {
  const name = 'Semantic conflict with target branch';
  const repo = context.repo;
  const preview = context.eventName === 'pull_request';
  if (preview) core.setOutput('verdict', 'INCONCLUSIVE');
  const number = preview ? context.payload.pull_request.number : context.payload.issue.number;
  const advisory = 'CodeRabbit can make mistakes, including false positives. Review the evidence. ' +
    'This check is advisory and must remain non-required; its failure does not block merging ' +
    'under that configuration. Other merge requirements still apply.';
  const {data: pr} = await github.rest.pulls.get({...repo, pull_number: number});
  const eligible = pr => pr.state === 'open' && pr.base.ref === 'main' &&
    (preview || (!pr.draft && pr.labels.some(label => label.name === 'ai: semantic-conflict')));
  if (!eligible(pr)) return;
  if (preview && pr.head.sha !== context.payload.pull_request.head.sha) {
    core.warning('This preview run is stale; use a run for the current PR head.');
    return;
  }
  const {data: ref} = await github.rest.git.getRef({...repo, ref: 'heads/main'});
  const head = pr.head.sha;
  const base = ref.object.sha;
  let comment;
  if (preview) {
    const comments = await github.paginate(github.rest.issues.listComments, {
      ...repo, issue_number: number, per_page: 100,
    });
    comment = comments.filter(c => c.user?.login === 'coderabbitai[bot]' &&
      c.user?.type === 'Bot' && c.body?.toLowerCase().includes(name.toLowerCase()) &&
      c.body.toLowerCase().includes(`semantic_result head=${head} target=${base} `))
      .sort((a, b) => b.id - a.id)[0];
    if (!comment) {
      const message = `No CodeRabbit result for head ${head} + main ${base}. ` +
        'This preview has no AI verdict. Request a custom pre-merge evaluation for these SHAs, ' +
        'then rerun the tests and result lookup job after the reply arrives.';
      core.warning(message);
      await core.summary.addRaw(`${message}\n\n${advisory}\n`).write();
      return;
    }
  } else {
    ({data: comment} = await github.rest.issues.getComment({
      ...repo, comment_id: context.payload.comment.id,
    }));
  }
  const body = comment.body || '';
  if (comment.user?.login !== 'coderabbitai[bot]' || comment.user?.type !== 'Bot') return;
  if (!body.includes('<!-- pre-merge-checks-results -->') &&
      !body.includes('<!-- pre_merge_checks_walkthrough_start -->')) return;
  const row = body.split('\n').map(line => line.split('|').map(cell => cell.trim()))
    .find(cells => cells[1]?.toLowerCase() === name.toLowerCase());
  if (!row) return;
  const details = body.match(/<summary>Full details: Semantic conflict with target branch<\/summary>([\s\S]*?)<\/details>/i);
  const result = details ? details[1] : row.join('|');

  let check;
  if (!preview) {
    const checks = await github.paginate(github.rest.checks.listForRef, {
      ...repo, ref: head, check_name: name, filter: 'all', per_page: 100,
    });
    check = checks.filter(check => check.app?.slug === 'github-actions' &&
      check.external_id === `semantic-conflict:${number}:${head}:${base}`)
      .sort((a, b) => b.id - a.id)[0];
    if (!check) return; // Includes unrelated manual fixture evaluations.
  }

  const pattern = /semantic_result head=([a-f0-9]{40}) target=([a-f0-9]{40}) merge_base=([a-f0-9]{40}) verdict=(pass|fail|inconclusive)\b/g;
  const records = [...new Set([...result.toLowerCase().matchAll(pattern)].map(match => match[0]))];
  let verdict = 'INCONCLUSIVE';
  let reason = 'The result has no unique, verifiable revision record.';
  if (records.length === 1) {
    const [, reportedHead, reportedBase, mergeBase, rawVerdict] =
      [...records[0].matchAll(pattern)][0];
    const reportedVerdict = rawVerdict.toUpperCase();
    // A stale comment must never overwrite the current revision's result.
    if (reportedHead !== head || reportedBase !== base) return;
    const {data: comparison} = await github.request(
      'GET /repos/{owner}/{repo}/compare/{basehead}',
      {...repo, basehead: `${base}...${head}`});
    const expectedStatus = {PASS: /Passed/i, FAIL: /Warning|Error/i, INCONCLUSIVE: /Inconclusive/i};
    if (comparison.merge_base_commit?.sha === mergeBase && expectedStatus[reportedVerdict].test(row[2])) {
      verdict = reportedVerdict;
      reason = `Verified head ${head}, target ${base}, merge base ${mergeBase}.`;
    } else {
      reason = 'The reported verdict or merge base could not be verified.';
    }
  }
  // Recheck refs after fetching the result and history.
  const {data: current} = await github.rest.pulls.get({...repo, pull_number: number});
  const {data: currentRef} = await github.rest.git.getRef({...repo, ref: 'heads/main'});
  if (current.head.sha !== head || currentRef.object.sha !== base ||
      !eligible(current)) {
    if (preview) core.warning('The PR or main changed during this preview; rerun for fresh evidence.');
    return;
  }
  const titles = {
    PASS: 'No semantic conflict found (best effort)',
    FAIL: 'Possible semantic conflict — CodeRabbit may be wrong',
    INCONCLUSIVE: '⚠️ Semantic analysis inconclusive',
  };
  const summary = `${reason}\n\n[CodeRabbit analysis](${comment.html_url})\n\n${advisory}`;
  if (!preview) await github.rest.checks.update({
    ...repo, check_run_id: check.id, status: 'completed',
    conclusion: {PASS: 'success', FAIL: 'failure', INCONCLUSIVE: 'neutral'}[verdict],
    details_url: comment.html_url,
    output: {
      title: titles[verdict],
      summary,
    },
  });
  await core.summary.addRaw(`${preview ? 'Read-only PR preview\n\n' : ''}` +
    `${titles[verdict]}\n\n${summary}\n`).write();
  core.info(`Verified semantic verdict: ${verdict}; head=${head}; target=${base}`);
  const message = `${titles[verdict]}. ${advisory} ${comment.html_url}`;
  if (preview) {
    core.setOutput('verdict', verdict);
    core.setOutput('summary', `${titles[verdict]}\n\n${summary}`);
    core.setOutput('message', message);
  } else if (verdict === 'FAIL') {
    core.setFailed(message);
  }
  if (verdict === 'INCONCLUSIVE') core.warning(`${titles[verdict]}: ${comment.html_url}`);
};
