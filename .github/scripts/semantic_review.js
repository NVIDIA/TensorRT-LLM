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

const {readFileSync} = require('node:fs');
const {join} = require('node:path');

const NAME = 'Semantic conflict with target branch';
const notice = 'Best-effort AI judgment for the recorded revisions. ' +
  'PASS, FAIL and INCONCLUSIVE may be incomplete or incorrect. ' +
  'PR authors and reviewers should independently verify the evidence and relevant behavior. ' +
  'This semantic review and its status/workflow are advisory, not required merge checks ' +
  'under current repository rules; other merge requirements still apply. ' +
  'Advisory status does not make a confirmed defect safe to ignore.';
const supported = ref => ref === 'main' || /^release\/[^\s]+$/.test(ref);
const eligible = pr => pr.state === 'open' && !pr.draft && supported(pr.base.ref) &&
  (pr.auto_merge || pr.labels.some(label => label.name === 'ci: full pre-merge approved'));
const isCommandUser = user => user?.login === 'trtllm-agent' &&
  user.id === 296075020 && user.type === 'User';
const isReviewer = user => user?.login === 'coderabbitai[bot]' &&
  user.id === 136622811 && user.type === 'Bot';
const uuid = /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/;
const sha = /^[a-f0-9]{40}$/;
const identity = (number, request) => `semantic-review:${number}:${request.id}`;
const statusContext = number => `${NAME} / PR #${number}`;
const isPublisher = user => user?.login === 'github-actions[bot]' &&
  user.id === 41898282 && user.type === 'Bot';
const titles = {PASS: 'No semantic conflict found (best effort)',
  FAIL: 'Possible semantic conflict', INCONCLUSIVE: 'Review completed: inconclusive'};
const STICKY_MARKER = '<!-- semantic-review-sticky -->';

function requests(comments) {
  return comments.filter(comment => isCommandUser(comment.user)).flatMap(comment => {
    const marker = comment.body?.match(/<!-- semantic-review-request:(.+) -->/);
    if (!marker) return [];
    let request;
    try { request = JSON.parse(marker[1]); } catch (error) {
      if (error instanceof SyntaxError) return [];
      throw error;
    }
    if (!request || !uuid.test(request.id) || !supported(request.branch) ||
        ![request.head, request.target, request.mergeBase].every(value => sha.test(value)) ||
        (request.automaticRetryOf !== undefined &&
          (typeof request.automaticRetryOf !== 'string' || !uuid.test(request.automaticRetryOf))) ||
        (request.checkId !== undefined &&
          (!Number.isSafeInteger(request.checkId) || request.checkId <= 0))) return [];
    return [{...request, commentId: comment.id, created_at: comment.created_at}];
  }).sort((a, b) => b.commentId - a.commentId);
}

function command(request) {
  const prompt = readFileSync(join(__dirname, '../semantic-review-prompt.md'), 'utf8')
    .replace(/<!--[^]*?-->/g, '').trim();
  return `@coderabbitai\n\n${prompt}\n\n` +
    'Analyze only these fixed revisions in NVIDIA/TensorRT-LLM, not the hosting PR:\n' +
    `request_id=${request.id}\nhead=${request.head}\ntarget=${request.target}\n` +
    `merge_base=${request.mergeBase}\nbranch=${request.branch}`;
}

function awaiting(request) {
  return {title: 'Waiting for CodeRabbit response',
    summary: `Request ${request.id}. Head ${request.head}, target ${request.target}, ` +
      `merge base ${request.mergeBase}.\n\n${notice}`};
}

function parseResult(comment, request, repo) {
  if (!isReviewer(comment.user)) return;
  const parts = (comment.body || '').split(/^(?:#{1,6}[ \t]+)?SEMANTIC_REVIEW[ \t]*\r?$/m);
  if (parts.length !== 2) return;
  const lines = parts[1].match(/^SEMANTIC_RESULT[^\r\n]*$/gmi) || [];
  if (lines.length !== 1) return;
  const record = lines[0].match(/^SEMANTIC_RESULT request_id=([^\s]+) head=([^\s]+) target=([^\s]+) merge_base=([^\s]+) verdict=(PASS|FAIL|INCONCLUSIVE)[ \t]*$/i);
  if (!record) return;
  const [, id, head, target, mergeBase, rawVerdict] = record;
  if (id !== request.id || head !== request.head || target !== request.target ||
      mergeBase !== request.mergeBase) return;
  let verdict = rawVerdict.toUpperCase();
  const citations = [...parts[1].matchAll(/https:\/\/github\.com\/([^/\s]+\/[^/\s]+)\/blob\/([a-f0-9]{40})\/[^\s<>)]+#L[1-9]\d*/g)]
    .filter(match => match[1].toLowerCase() === `${repo.owner}/${repo.repo}`.toLowerCase())
    .map(match => match[2]);
  const missingEvidence = verdict !== 'INCONCLUSIVE' &&
    ![head, target].every(revision => citations.includes(revision));
  if (missingEvidence) verdict = 'INCONCLUSIVE';
  return {verdict, missingEvidence, comment};
}

function refersToRequest(comment, request) {
  const body = comment.body || '';
  const bindings = [...body.matchAll(/^SEMANTIC_RESULT request_id=([^\s]+) head=([^\s]+) target=([^\s]+) merge_base=([^\s]+)/gmi)];
  return bindings.length ? bindings.some(([, id, head, target, mergeBase]) =>
    id === request.id && head === request.head && target === request.target &&
      mergeBase === request.mergeBase) : body.includes(request.id);
}

// The newest reply a publication has observed for a request, recorded in the
// commit status URL (or the legacy check summary). It survives reply deletion
// and edits, so older replies cannot resurrect a revoked verdict.
function recordedSource({statuses, checks, link, request, number}) {
  for (const item of statuses.filter(status => isPublisher(status.creator))) {
    let url;
    try { url = new URL(item.target_url); } catch { continue; }
    if (`${url.origin}${url.pathname}` !== link ||
        url.searchParams.get('semantic_review_request') !== request.id) continue;
    return Number(url.searchParams.get('semantic_review_source'));
  }
  const legacy = checks.filter(check => check.app?.slug === 'github-actions' &&
    check.head_sha === request.head &&
    check.external_id === identity(number, request)).sort((a, b) => b.id - a.id)[0];
  return Number(legacy?.output?.summary?.match(/<!-- semantic-review-source:(\d+) -->/)?.[1]);
}

// The newest reply at or after the recorded source decides the verdict; a
// non-parsing reply at that position revokes it instead of letting an older
// reply win.
function latestResult(comments, request, repo, sourceId) {
  const replies = comments.filter(comment => isReviewer(comment.user) &&
    comment.id > request.commentId && (!sourceId || comment.id >= sourceId) &&
    Date.parse(comment.created_at) >= Date.parse(request.created_at))
    .sort((a, b) => b.id - a.id);
  let result;
  let invalidSource;
  for (const comment of replies) {
    const parsed = parseResult(comment, request, repo);
    if (parsed) { result = parsed; break; }
    if (comment.id === sourceId || refersToRequest(comment, request)) {
      invalidSource = comment;
      break;
    }
  }
  return {result, invalidSource};
}

async function reviewState({github, repo, number, comments, head}) {
  comments ??= await github.paginate(github.rest.issues.listComments,
    {...repo, issue_number: number, per_page: 100});
  const request = requests(comments)[0];
  const refs = new Set([request?.head, head].filter(Boolean));
  if (!refs.size) return;
  const checks = [];
  for (const ref of refs) {
    checks.push(...await github.paginate(github.rest.checks.listForRef,
      {...repo, ref, check_name: NAME, filter: 'all', per_page: 100}));
  }
  const owned = checks.filter(check => check.app?.slug === 'github-actions' &&
    check.name === NAME && refs.has(check.head_sha) &&
    check.external_id?.startsWith(`semantic-review:${number}:`));
  const cleanup = owned.filter(item => item.status !== 'completed');
  if (!request) return {cleanup};
  const statuses = (await github.paginate(github.rest.repos.listCommitStatusesForRef,
    {...repo, ref: request.head, per_page: 100}))
    .filter(status => status.context === statusContext(number)).sort((a, b) => b.id - a.id);
  const status = statuses[0];
  const link = `https://github.com/${repo.owner}/${repo.repo}/pull/${number}`;
  const publishedId = recordedSource({statuses, checks: owned, link, request, number});
  const sourceId = Number.isSafeInteger(publishedId) && publishedId > request.commentId ? publishedId : 0;
  const {result, invalidSource} = latestResult(comments, request, repo, sourceId);
  const source = result?.comment.id || invalidSource?.id || sourceId;
  const url = source ? `${link}#issuecomment-${source}` : undefined;
  const output = result ? {
    title: titles[result.verdict],
    summary: `Request ${request.id}. Head ${request.head}, target ${request.target}, ` +
      `merge base ${request.mergeBase}.\n\n` +
      `Result received ${result.comment.created_at}. [CodeRabbit analysis](${url}).\n\n` +
      (result.missingEvidence ? 'Missing fixed-revision source citations; no verified verdict.\n\n' : '') + notice,
  } : awaiting(request);
  if (source && !result) output.summary += `\n\n[Reply without a valid result](${url}).`;
  const desired = {
    state: result ? {PASS: 'success', FAIL: 'failure', INCONCLUSIVE: 'pending'}[result.verdict] : 'pending',
    description: output.title,
    target_url: `${link}?semantic_review_request=${request.id}` +
      (source ? `&semantic_review_source=${source}` : '') +
      `#issuecomment-${result ? result.comment.id : request.commentId}`,
  };
  const changed = !isPublisher(status?.creator) ||
    Object.entries(desired).some(([key, value]) => status?.[key] !== value);
  return {request, result, status, cleanup, output,
    update: changed ? {...repo, sha: request.head, context: statusContext(number), ...desired} : undefined};
}

async function publish({github, context, core, number, comments, head}) {
  if (number === undefined) {
    if (!context.payload.issue?.pull_request || !isReviewer(context.payload.comment?.user)) return;
    number = context.payload.issue.number;
  }
  const state = await reviewState({github, repo: context.repo, number, comments, head});
  if (state?.update) {
    await github.rest.repos.createCommitStatus(state.update);
    const {title, summary} = state.output;
    await core.summary.addRaw(`${title}\n\n${summary}\n`).write();
  }
  for (const check of state?.cleanup || []) {
    await github.rest.checks.update({...context.repo, check_run_id: check.id,
      status: 'completed', conclusion: 'cancelled', output: {
        title: 'Semantic review is published as a commit status',
        summary: 'Semantic review results use the per-PR commit status. ' +
          'Cancellation clears the inactive check and does not assign an AI verdict.',
      }});
  }
  return state;
}

function stickyBody({repo, number, entries}) {
  const link = `https://github.com/${repo.owner}/${repo.repo}/pull/${number}`;
  const url = entry => `${link}#issuecomment-${entry.result ? entry.result.comment.id : entry.request.commentId}`;
  const [latest] = entries;
  const rows = entries.map(entry => {
    const verdict = entry.result ? entry.result.verdict : entry.active ? 'WAITING' : 'NO RESULT';
    return `| ${entry.request.created_at} | \`${entry.request.head.slice(0, 12)}\` | ` +
      `\`${entry.request.target.slice(0, 12)}\` | ${verdict} | ` +
      `[${entry.result ? 'reply' : 'request'}](${url(entry)}) |`;
  });
  return `${STICKY_MARKER}\n## Semantic conflict review\n\n` +
    `The verdict of record is the \`${statusContext(number)}\` commit status on the requested ` +
    'head commit. This summary updates on reply events and may lag between a new request ' +
    'and its reply.\n\n' +
    `**Latest recorded state:** ${latest.result ? titles[latest.result.verdict] : 'Waiting for CodeRabbit response'} ` +
    `for head \`${latest.request.head}\`, target \`${latest.request.target}\`, ` +
    `merge base \`${latest.request.mergeBase}\` (request \`${latest.request.id}\`).` +
    `${latest.result ? ` [CodeRabbit analysis](${url(latest)}).` : ''}\n\n${notice}\n\n` +
    '| Requested (UTC) | Head | Target | Verdict | Comment |\n' +
    `| --- | --- | --- | --- | --- |\n${rows.join('\n')}\n\n` +
    'Processed request and reply comments are minimized to reduce timeline noise; ' +
    'they remain expandable for audit.';
}

async function tidy({github, context, core, number, comments}) {
  const repo = context.repo;
  if (number === undefined) {
    if (!context.payload.issue?.pull_request || !isReviewer(context.payload.comment?.user)) return;
    number = context.payload.issue.number;
  }
  comments ??= await github.paginate(github.rest.issues.listComments,
    {...repo, issue_number: number, per_page: 100});
  const history = requests(comments);
  if (!history.length) return;
  // Every row must agree with its published status, including reply
  // edit/deletion revocations, so all rows use the same recorded-source and
  // newest-reply rules as publication: the latest row from reviewState, the
  // historical rows from the sources recorded for their own heads.
  const state = await reviewState({github, repo, number, comments});
  const link = `https://github.com/${repo.owner}/${repo.repo}/pull/${number}`;
  const heads = new Map();
  const recorded = async request => {
    if (!heads.has(request.head)) {
      const statuses = (await github.paginate(github.rest.repos.listCommitStatusesForRef,
        {...repo, ref: request.head, per_page: 100}))
        .filter(status => status.context === statusContext(number)).sort((a, b) => b.id - a.id);
      const checks = await github.paginate(github.rest.checks.listForRef,
        {...repo, ref: request.head, check_name: NAME, filter: 'all', per_page: 100});
      heads.set(request.head, {statuses, checks});
    }
    const publishedId = recordedSource({...heads.get(request.head), link, request, number});
    return Number.isSafeInteger(publishedId) && publishedId > request.commentId ? publishedId : 0;
  };
  const entries = [];
  for (const [index, request] of history.entries()) {
    if (index === 0) {
      entries.push({request, result: state?.result, active: true});
      continue;
    }
    const {result} = latestResult(comments, request, repo, await recorded(request));
    entries.push({request, result, active: false});
  }
  const body = stickyBody({repo, number, entries});
  const sticky = comments.find(comment => isPublisher(comment.user) &&
    comment.body?.includes(STICKY_MARKER));
  if (!sticky) {
    await github.rest.issues.createComment({...repo, issue_number: number, body});
  } else if (sticky.body !== body) {
    await github.rest.issues.updateComment({...repo, comment_id: sticky.id, body});
  }
  // A request/reply pair is audit trail once its verdict is recorded or a newer
  // request supersedes it. Minimization is best-effort display cleanup, never
  // a result override, and must not fail the sticky summary above. An active
  // request whose verdict was revoked (reply edited or deleted) was minimized
  // while it had a result, so it is restored to keep the waiting request
  // visible.
  const nodes = new Map(comments.map(comment => [comment.id, comment.node_id]));
  const restore = new Set(entries.filter(entry => entry.active && !entry.result)
    .map(entry => nodes.get(entry.request.commentId)).filter(Boolean));
  // Minimize every reply bound to a processed request, not only the selected
  // one, so an obsolete reply never stays visible while its correction is
  // minimized. The recorded source is deliberately not applied here:
  // pre-source replies are obsolete and belong in the audit trail.
  const bound = request => comments.filter(comment => isReviewer(comment.user) &&
    comment.id > request.commentId &&
    Date.parse(comment.created_at) >= Date.parse(request.created_at) &&
    (parseResult(comment, request, repo) || refersToRequest(comment, request)));
  const targets = [...new Set(entries.filter(entry => entry.result || !entry.active)
    .flatMap(entry => [nodes.get(entry.request.commentId),
      ...bound(entry.request).map(comment => nodes.get(comment.id))]).filter(Boolean))];
  if (!targets.length && !restore.size) return {entries, minimized: [], restored: []};
  const minimized = [];
  const restored = [];
  try {
    const visible = await github.graphql(
      'query($ids: [ID!]!) { nodes(ids: $ids) { id ... on Minimizable { isMinimized } } }',
      {ids: [...targets, ...restore]});
    for (const node of visible.nodes || []) {
      if (!node) continue;
      if (restore.has(node.id)) {
        if (node.isMinimized !== true) continue;
        await github.graphql(
          'mutation($id: ID!) { unminimizeComment(input: {subjectId: $id}) ' +
          '{ unminimizedComment { isMinimized } } }', {id: node.id});
        restored.push(node.id);
        continue;
      }
      if (node.isMinimized !== false) continue;
      await github.graphql(
        'mutation($id: ID!) { minimizeComment(input: {subjectId: $id, classifier: OUTDATED}) ' +
        '{ minimizedComment { isMinimized } } }', {id: node.id});
      minimized.push(node.id);
    }
  } catch (error) {
    core.warning(`PR #${number}: comment minimization failed (${error.message}).`);
  }
  return {entries, minimized, restored};
}

module.exports = {NAME, notice, supported, eligible, isCommandUser, isReviewer,
  identity, statusContext, requests, command, awaiting, parseResult, reviewState, publish,
  STICKY_MARKER, stickyBody, tidy};
