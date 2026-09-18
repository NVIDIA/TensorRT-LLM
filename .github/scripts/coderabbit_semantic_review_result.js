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

const NAME = 'Semantic conflict with target branch';
const AUDIT = 'Semantic conflict audit (post-merge)';
const NOTICE = 'CodeRabbit can make mistakes, including false positives. Review the evidence. ' +
  'This check is advisory and must remain non-required; its failure does not block merging ' +
  'under that configuration. Other merge requirements still apply.';
const supported = ref => ref === 'main' || /^release\/.+/.test(ref);
const isBot = (comment, login) => comment.user?.login === login && comment.user?.type === 'Bot';
const compare = async (github, repo, base, head) => (await github.request(
  'GET /repos/{owner}/{repo}/compare/{basehead}', {...repo, basehead: `${base}...${head}`}
)).data;

// Request metadata is written only by the trusted workflow, never accepted from PR authors.
function requests(comments) {
  return comments.filter(c => isBot(c, 'github-actions[bot]')).flatMap(c => {
    const text = c.body?.match(/<!-- semantic-request-v2:(.+) -->/);
    if (!text) return [];
    try {
      const request = JSON.parse(text[1]);
      if (![request.head, request.target, request.mergeBase].every(s => /^[a-f0-9]{40}$/.test(s)) ||
          !supported(request.branch)) return [];
      return [{...request, created_at: c.created_at, id: c.id}];
    } catch { return []; }
  }).sort((a, b) => b.id - a.id);
}

function parseResult(comment) {
  const body = comment.body || '';
  if (!isBot(comment, 'coderabbitai[bot]') ||
      (!body.includes('<!-- pre-merge-checks-results -->') &&
       !body.includes('<!-- pre_merge_checks_walkthrough_start -->'))) return;
  const row = body.split('\n').map(line => line.split('|').map(cell => cell.trim()))
    .find(cells => cells[1]?.toLowerCase() === NAME.toLowerCase());
  if (!row) return;
  const details = body.match(/<summary>Full details: Semantic conflict with target branch<\/summary>([\s\S]*?)<\/details>/i);
  const result = details ? details[1] : row.join('|');
  const pattern = /semantic_result head=([a-f0-9]{40}) target=([a-f0-9]{40}) merge_base=([a-f0-9]{40}) verdict=(pass|fail|inconclusive)\b/g;
  const matches = [...new Map([...result.toLowerCase().matchAll(pattern)].map(m => [m[0], m])).values()];
  if (matches.length !== 1) return;
  const [, head, target, mergeBase, rawVerdict] = matches[0];
  let verdict = rawVerdict.toUpperCase();
  const status = {PASS: /Passed/i, FAIL: /Warning|Error/i, INCONCLUSIVE: /Inconclusive/i};
  if (!status[verdict].test(row[2])) return;
  const merged = [...new Set([...result.toLowerCase().matchAll(/semantic_merged sha=([a-f0-9]{40})\b/g)].map(m => m[1]))];
  if (merged.length > 1) return;
  // This checks citation presence, not the correctness of the AI's reasoning.
  const repository = comment.html_url?.match(/^https:\/\/github\.com\/([^/]+\/[^/]+)\/pull\//i)?.[1].toLowerCase();
  const citations = [...result.matchAll(/https:\/\/github\.com\/([^/\s]+\/[^/\s]+)\/blob\/([a-f0-9]{40})\/[^\s<>)|]+#L[1-9]\d*/gi)]
    .filter(m => m[1].toLowerCase() === repository).map(m => m[2].toLowerCase());
  const missingEvidence = verdict !== 'INCONCLUSIVE' && ![head, target].every(sha => citations.includes(sha));
  if (missingEvidence) verdict = 'INCONCLUSIVE';
  return {head, target, mergeBase, verdict, missingEvidence, merged: merged[0], comment};
}

const matches = (result, request) => result && result.head === request.head &&
  result.target === request.target && result.mergeBase === request.mergeBase &&
  result.merged === request.merged;
const identity = (pr, pair) => `semantic-v2:${pr.number}:${pair.head}:${pair.target}:${pair.merged || 'open'}`;

async function evidence(github, repo, pr) {
  let target;
  let tree;
  let merged;
  if (pr.merged) {
    merged = pr.merge_commit_sha;
    const {data: commit} = await github.rest.git.getCommit({...repo, commit_sha: merged});
    // The repository requires squash merges. A normal two-parent merge is also
    // unambiguous; rebases must not silently be treated as a squash merge.
    if (commit.parents.length === 1) {
      const {data: rules} = await github.request('GET /repos/{owner}/{repo}/rules/branches/{branch}',
        {...repo, branch: pr.base.ref});
      if (!rules.some(r => r.type === 'pull_request' &&
          r.parameters?.allowed_merge_methods?.length === 1 &&
          r.parameters.allowed_merge_methods[0] === 'squash')) {
        throw new Error('Cannot verify a squash-only merge policy; historical target is inconclusive.');
      }
    } else if (commit.parents.length !== 2 || commit.parents[1].sha !== pr.head.sha) {
      throw new Error('Cannot verify the historical merge parents.');
    }
    target = commit.parents[0].sha;
    tree = commit.tree.sha;
  } else {
    const {data: ref} = await github.rest.git.getRef({...repo, ref: `heads/${pr.base.ref}`});
    target = ref.object.sha;

  }
  const comparison = await compare(github, repo, target, pr.head.sha);
  return {head: pr.head.sha, target, mergeBase: comparison.merge_base_commit.sha,
    branch: pr.base.ref, merged, tree, comparison};
}

// This extra API read is needed only when actually requesting AI, not on every scan.
async function candidateTree(github, repo, pr, target) {
  if (!pr.merge_commit_sha) return;
  try {
    const {data: candidate} = await github.rest.git.getCommit({...repo, commit_sha: pr.merge_commit_sha});
    if (candidate.parents.length === 2 && candidate.parents[0].sha === target &&
        candidate.parents[1].sha === pr.head.sha) return candidate.tree.sha;
  } catch (error) {
    if (error.status !== 404 && error.status !== 409) throw error;
  }
}

// Both the privileged publisher and the read-only PR preview use this verifier.
async function publish({github, context, core}) {
  const preview = context.eventName === 'pull_request';
  if (preview) core.setOutput('verdict', 'INCONCLUSIVE');
  const repo = context.repo;
  const number = preview ? context.payload.pull_request.number : context.payload.issue.number;
  const {data: pr} = await github.rest.pulls.get({...repo, pull_number: number});
  if (!supported(pr.base.ref) || (!pr.merged && (pr.state !== 'open' || (!preview && pr.draft)))) return;
  if (preview && (pr.merged || pr.head.sha !== context.payload.pull_request.head.sha)) {
    core.warning('This preview is stale; use a run for the current PR head.');
    return;
  }
  const pair = await evidence(github, repo, pr);
  const comments = await github.paginate(github.rest.issues.listComments, {
    ...repo, issue_number: number, per_page: 100,
  });
  const allRequests = requests(comments);
  const latestRequest = allRequests.find(r => r.branch === pair.branch &&
    r.head === pair.head && r.target === pair.target && r.mergeBase === pair.mergeBase &&
    (pair.merged ? (r.merged === pair.merged || (!r.merged && r.tree && r.tree === pair.tree)) : !r.merged));
  let result;
  // Reconcile the latest API result, even when an older comment event is replayed.
  for (const comment of comments.toSorted((a, b) => b.id - a.id)) {
    const parsed = parseResult(comment);
    if (!parsed || parsed.head !== pair.head || parsed.target !== pair.target ||
        parsed.mergeBase !== pair.mergeBase) continue;
    if (preview && !latestRequest && !parsed.merged) { result = parsed; break; }
    if (latestRequest && matches(parsed, latestRequest) &&
        comment.id > latestRequest.id &&
        Date.parse(comment.created_at) >= Date.parse(latestRequest.created_at)) {
      result = parsed; break;
    }
  }
  if (!result) {
    if (preview) {
      const message = `No verified CodeRabbit verdict for head ${pair.head} + ${pair.branch} ${pair.target}. ` +
        'Request a custom evaluation and rerun the tests and result lookup job after the reply arrives.';
      core.warning(message);
      await core.summary.addRaw(`${message}\n\n${NOTICE}`).write();
    }
    return;
  }
  const {data: current} = await github.rest.pulls.get({...repo, pull_number: number});
  if (current.head.sha !== pair.head || current.base.ref !== pair.branch ||
      current.merged !== pr.merged || current.state !== pr.state || (!preview && current.draft)) return;
  if (pair.merged) {
    if (current.merge_commit_sha !== pair.merged) return;
  } else {
    const {data: ref} = await github.rest.git.getRef({...repo, ref: `heads/${pair.branch}`});
    if (ref.object.sha !== pair.target) {
      if (preview) core.warning('The target changed during this preview; the result is stale.');
      return;
    }
  }
  const title = {PASS: 'No semantic conflict found (best effort)',
    FAIL: 'Possible semantic conflict — CodeRabbit may be wrong',
    INCONCLUSIVE: 'Semantic analysis inconclusive'}[result.verdict];
  const summary = `${pair.merged ? `Post-merge audit of ${pair.merged}. ` : ''}` +
    `Verified head ${pair.head}, target ${pair.target}, merge base ${pair.mergeBase}.\n\n` +
    (result.missingEvidence ? 'CodeRabbit did not provide source links with full SHAs and line numbers for both revisions; ' +
      'its reported verdict is treated as inconclusive.\n\n' : '') +
    `[CodeRabbit analysis](${result.comment.html_url})\n\n${NOTICE}`;
  if (!preview) {
    const checks = await github.paginate(github.rest.checks.listForRef, {
      ...repo, ref: pair.merged || pair.head, check_name: pair.merged ? AUDIT : NAME,
      filter: 'all', per_page: 100,
    });
    const check = checks.filter(c => c.app?.slug === 'github-actions' && c.external_id === identity(pr, pair))
      .sort((a, b) => b.id - a.id)[0];
    if (!check) return;
    await github.rest.checks.update({...repo, check_run_id: check.id, status: 'completed',
      conclusion: {PASS: 'success', FAIL: 'failure', INCONCLUSIVE: 'neutral'}[result.verdict],
      details_url: result.comment.html_url, output: {title, summary}});
    if (pair.merged) {
      const marker = `<!-- semantic-audit:${pair.merged}:${result.comment.id}:${result.verdict} -->`;
      if (!comments.some(c => isBot(c, 'github-actions[bot]') && c.body?.includes(marker))) {
        await github.rest.issues.createComment({...repo, issue_number: number,
          body: `${marker}\n**Post-merge semantic audit: ${result.verdict}**\n\n${summary}`});
      }
    }
  }
  await core.summary.addRaw(`${preview ? 'Read-only PR preview\n\n' : ''}${title}\n\n${summary}\n`).write();
  core.info(`Verified semantic verdict: ${result.verdict}; head=${pair.head}; target=${pair.target}`);
  const message = `${title}. ${NOTICE} ${result.comment.html_url}`;
  if (preview) {
    core.setOutput('verdict', result.verdict);
    core.setOutput('summary', `${title}\n\n${summary}`);
    core.setOutput('message', message);
  } else if (result.verdict === 'FAIL') core.setFailed(message);
  if (result.verdict === 'INCONCLUSIVE') core.warning(message);
}

module.exports = publish;
Object.assign(module.exports, {NAME, AUDIT, NOTICE, supported, compare, requests, parseResult, matches, identity, evidence, candidateTree});
