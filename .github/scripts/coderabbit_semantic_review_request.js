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

const fs = require('node:fs');
const publish = require('./coderabbit_semantic_review_result.js');
const {NAME, AUDIT, NOTICE, supported, compare, requests, parseResult, matches, identity, evidence, candidateTree, isCommandUser} = publish;
const HOUR = 60 * 60 * 1000;
const DAY = 24 * HOUR;
const APPROVED = 'ci: full pre-merge approved';

function command(pair) {
  const config = fs.readFileSync('.coderabbit.yaml', 'utf8');
  const match = config.match(/name: "Semantic conflict with target branch"[\s\S]*?instructions: \|\n((?: {10}[^\n]*(?:\n|$)|\n)*)/);
  if (!match) throw new Error('Missing semantic check instructions');
  const instructions = match[1].replace(/^ {10}/gm, '').trim() + '\n' +
    `Inspect these fixed revisions: head=${pair.head}, target=${pair.target}, merge_base=${pair.mergeBase}. ` +
    (pair.merged ? `This is a post-merge audit. Also inspect the actual merged code at ${pair.merged}. ` +
      `Include SEMANTIC_MERGED sha=${pair.merged} immediately after the result line. ` +
      'Later changes to the target branch do not invalidate this historical audit.' :
      `This is a pre-merge analysis against ${pair.branch}. The result applies only to this pair.`);
  return `@coderabbitai\nPlease perform this advisory semantic analysis and reply in a normal PR chat comment. ` +
    `Do not invoke the custom pre-merge check command or submit a review/request changes.\n\n${instructions}`;
}

async function requestOne({github, commandGithub, context, core, number, manual, now}) {
  const repo = context.repo;
  const {data: pr} = await github.rest.pulls.get({...repo, pull_number: number});
  if (!supported(pr.base.ref) || (!pr.merged && (pr.state !== 'open' || pr.draft))) {
    if (manual) throw new Error(`PR #${number}: requires a non-draft open or merged main/release PR.`);
    return;
  }
  const pair = await evidence(github, repo, pr);
  const comments = await github.paginate(github.rest.issues.listComments, {
    ...repo, issue_number: number, per_page: 100,
  });
  const history = requests(comments).filter(r => r.branch === pair.branch);
  const results = comments.toSorted((a, b) => b.id - a.id).map(parseResult).filter(Boolean);
  const resultFor = r => results.find(v => matches(v, r) && v.comment.id > r.id &&
    Date.parse(v.comment.created_at) >= Date.parse(r.created_at));
  const reusable = history.find(r => r.head === pair.head && r.target === pair.target &&
    r.mergeBase === pair.mergeBase && (!pair.merged ? !r.merged :
      (r.merged === pair.merged || (!r.merged && r.tree && r.tree === pair.tree))));
  const checks = await github.paginate(github.rest.checks.listForRef, {
    ...repo, ref: pair.merged || pair.head, check_name: pair.merged ? AUDIT : NAME,
    filter: 'all', per_page: 100,
  });
  let check = checks.filter(c => c.app?.slug === 'github-actions' &&
    c.external_id?.startsWith(`semantic-v2:${number}:`)).sort((a, b) => b.id - a.id)[0];
  const currentId = identity(pr, pair);
  // Clear a stale green/red result even if this revision is below the AI threshold.
  if (check && check.external_id !== currentId && !pair.merged) {
    await github.rest.checks.update({...repo, check_run_id: check.id, status: 'completed',
      conclusion: 'neutral', output: {title: 'Previous semantic result is stale',
        summary: `No current verdict for head ${pair.head} + ${pair.branch} ${pair.target}. ${NOTICE}`}});
  }
  async function ensureCheck(reason) {
    if (check?.external_id === currentId) return;
    const output = {title: pair.merged ? 'Post-merge audit awaiting analysis' :
      check ? 'Previous semantic result is stale' : 'No current AI verdict',
    summary: `${reason} Head ${pair.head} + ${pair.branch} ${pair.target}. ${NOTICE}`};
    if (check) {
      await github.rest.checks.update({...repo, check_run_id: check.id, external_id: currentId,
        status: 'completed', conclusion: 'neutral', output});
      check.external_id = currentId;
    } else {
      ({data: check} = await github.rest.checks.create({...repo,
        name: pair.merged ? AUDIT : NAME, head_sha: pair.merged || pair.head,
        external_id: currentId, status: 'completed', conclusion: 'neutral', output}));
    }
  }
  if (reusable && !manual) {
    const result = resultFor(reusable);
    // Reuse an in-flight exact pair too. The publisher can complete the audit
    // when its pre-merge reply arrives, provided the final tree also matches.
    if (!pair.merged || reusable.merged || !result || result.verdict !== 'INCONCLUSIVE') {
      await ensureCheck('This version already has an analysis request.');
      if (result) await publish({github, core, context: {...context, eventName: 'issue_comment',
        payload: {issue: {number}}}});
      core.info(`PR #${number}: reusing the exact revision pair.`);
      return;
    }
  }
  const intent = context.eventName === 'pull_request_target' &&
    (context.payload.action === 'auto_merge_enabled' && pr.auto_merge ||
     context.payload.action === 'labeled' && context.payload.label?.name === APPROVED &&
     pr.labels.some(l => l.name === APPROVED) && process.env.SEMANTIC_APPROVAL_VALIDATED === 'true');
  let reason = manual ? 'Manual retry' : pair.merged ? 'Post-merge audit' : intent ? 'Merge intent' : '';
  const last = history.find(r => !r.merged);
  if (!reason) {
    const previous = last && resultFor(last);
    if (last && !['PASS', 'FAIL'].includes(previous?.verdict)) {
      await ensureCheck('Previous analysis has no verified verdict; use a manual retry. Routine requests are paused.');
      return;
    }
    let count = pair.comparison.behind_by;
    let since = Date.parse(pair.comparison.merge_base_commit.commit.committer.date);
    if (previous) {
      const progress = await compare(github, repo, last.target, pair.target);
      if (['ahead', 'identical'].includes(progress.status)) {
        count = progress.ahead_by;
        since = Date.parse(previous.comment.created_at);
      }
    }
    if (!count || (count < 30 && now - since < DAY)) {
      await ensureCheck('Below the 24-hour / 30-commit threshold; waiting for target changes.');
      return;
    }
    reason = '24-hour / 30-commit threshold';
  }
  // Approval and auto-merge share this budget even when the branch SHAs change.
  // Count any recent pre-merge request, so a routine scan cannot double the cost.
  if (!manual && !pair.merged && last && now - Date.parse(last.created_at) < HOUR) {
    await ensureCheck('Pre-merge analysis is in its one-hour cooldown; old results are stale.');
    return;
  }
  const {data: current} = await github.rest.pulls.get({...repo, pull_number: number});
  if (current.head.sha !== pair.head || current.base.ref !== pair.branch ||
      current.state !== pr.state || current.merged !== pr.merged || current.draft !== pr.draft ||
      (pair.merged && current.merge_commit_sha !== pair.merged)) {
    if (manual) throw new Error('Manual retry not posted: PR changed during validation.');
    return;
  }
  if (!pair.merged) {
    const {data: ref} = await github.rest.git.getRef({...repo, ref: `heads/${pair.branch}`});
    if (ref.object.sha !== pair.target) {
      if (manual) throw new Error('Manual retry not posted: target changed during validation.');
      return;
    }
  }
  await ensureCheck(reason);
  await github.rest.checks.update({...repo, check_run_id: check.id,
    status: 'completed', conclusion: 'neutral',
    output: {title: 'Awaiting CodeRabbit analysis',
      summary: `${reason}: head ${pair.head}, target ${pair.target}. ${NOTICE}`}});
  if (!pair.merged) pair.tree = await candidateTree(github, repo, current, pair.target);
  const {comparison, ...record} = pair;
  const {data: comment} = await commandGithub.rest.issues.createComment({...repo, issue_number: number,
    body: `${command(pair)}\n\n<!-- semantic-request-v2:${JSON.stringify({...record, reason})} -->\n\n` +
      `${reason} for PR #${number}. ${NOTICE} No AI verdict is asserted by posting this request.`});
  core.info(`PR #${number}: ${reason}; ${comment.html_url}`);
  await core.summary.addRaw(`PR #${number}: [${reason}](${comment.html_url}). No AI verdict is asserted.\n\n`).write();
  return true;
}

class ScanStopped extends Error {}

function rateLimited(error) {
  const headers = error.response?.headers || {};
  return error.status === 429 || error.status === 403 &&
    (headers['x-ratelimit-remaining'] === '0' || headers['retry-after'] !== undefined ||
     /rate limit|abuse detection/i.test(error.message));
}

async function run({github, commandGithub, context, core, now, progress}) {
  let numbers;
  const manual = context.eventName === 'workflow_dispatch';
  if (manual) {
    const raw = (process.env.DISPATCH_PULL_NUMBER || '').trim();
    const number = Number(raw);
    if (!/^[1-9][0-9]*$/.test(raw) || !Number.isSafeInteger(number)) {
      throw new Error('pull_number must be a positive integer');
    }
    numbers = [number];
  } else if (context.eventName === 'pull_request_target') {
    numbers = [context.payload.pull_request.number];
  } else if (context.eventName === 'schedule') {
    const pulls = await github.paginate(github.rest.pulls.list, {
      ...context.repo, state: 'open', per_page: 100,
    });
    const open = pulls.filter(p => !p.draft && supported(p.base.ref)).map(p => p.number).sort((a, b) => a - b);
    // Rotate the starting PR each scan; no persistent cursor or exact resume is implied.
    const offset = Math.floor(now / (6 * HOUR)) % (open.length || 1);
    numbers = [];
    // Recover recent merges if an event was dropped or a release branch still
    // lacks the workflow. Do not backfill the repository's entire merge history.
    for await (const response of github.paginate.iterator(github.rest.pulls.list, {
      ...context.repo, state: 'closed', sort: 'updated', direction: 'desc', per_page: 100,
    })) {
      numbers.push(...response.data.filter(p => supported(p.base.ref) &&
        Date.parse(p.merged_at) >= now - DAY).map(p => p.number));
      if (!response.data.length || Date.parse(response.data.at(-1).updated_at) < now - DAY) break;
    }
    numbers = [...new Set([...numbers, ...open.slice(offset), ...open.slice(0, offset)])];
  } else throw new Error(`Unsupported event: ${context.eventName}`);
  for (const number of numbers) {
    if (context.eventName === 'schedule' && progress.requested >= 20) {
      throw new ScanStopped('Reached the limit of 20 new AI requests for this scan.');
    }
    try {
      if (await requestOne({github, commandGithub, context, core, number, manual, now})) progress.requested++;
      progress.processed++;
    } catch (error) {
      if (context.eventName !== 'schedule' || error instanceof ScanStopped || rateLimited(error)) throw error;
      core.error(`PR #${number}: ${error.message}`);
      core.setFailed('Some PRs could not be scanned; see per-PR errors.');
    }
  }
}

module.exports = async ({github, commandGithub, context, core, now = Date.now()}) => {
  const scheduled = context.eventName === 'schedule';
  const progress = {processed: 0, requested: 0};
  const guards = [];
  let stopReason = 'Complete';
  try {
    if (scheduled) {
      for (const client of [github, commandGithub]) {
        let remaining = Infinity;
        const guard = async (request, options) => {
          if (remaining < 100) throw new ScanStopped('REST quota is below the 100-request reserve.');
          const response = await request(options);
          const value = response.headers?.['x-ratelimit-remaining'];
          if (value !== undefined) remaining = Number(value);
          return response;
        };
        client.hook.wrap('request', guard);
        guards.push([client, guard]);
        const {data} = await client.rest.rateLimit.get();
        remaining = data.resources.core.remaining;
      }
    }
    const {data: user} = await commandGithub.rest.users.getAuthenticated();
    if (!isCommandUser(user)) throw new Error('Semantic commands require the trtllm-agent User account.');
    await run({github, commandGithub, context, core, now, progress});
  } catch (error) {
    if (!scheduled || !(error instanceof ScanStopped || rateLimited(error))) throw error;
    stopReason = error.message;
    core.warning(`Scheduled scan stopped: ${stopReason} Unvisited PRs wait for a later scan or manual dispatch.`);
  } finally {
    for (const [client, guard] of guards) client.hook.remove('request', guard);
    if (scheduled) await core.summary.addRaw(`Scheduled scan: ${progress.processed} PRs processed; ` +
      `${progress.requested} new AI requests. ${stopReason}. Scans restart from live state, not a saved cursor.\n`).write();
  }
};
Object.assign(module.exports, {command});
