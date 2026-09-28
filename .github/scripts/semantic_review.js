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
const notice = 'Advisory, non-required AI analysis of the recorded revisions. ' +
  'CodeRabbit can miss problems or report false positives. Review the evidence.';
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
        !Number.isSafeInteger(request.checkId) || request.checkId <= 0) return [];
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
  return {title: 'Awaiting CodeRabbit analysis (no verdict)',
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
  const check = request ? owned.filter(check => check.head_sha === request.head &&
    check.external_id === identity(number, request)).sort((a, b) => b.id - a.id)[0] : undefined;
  const cleanup = owned.filter(item => item.status !== 'completed' && item.id !== check?.id);
  if (!check) return {request, check, result: undefined, update: undefined, cleanup};
  const publishedId = Number(check.output?.summary?.match(/<!-- semantic-review-source:(\d+) -->/)?.[1]);
  const sourceId = Number.isSafeInteger(publishedId) && publishedId > request.commentId ? publishedId : 0;
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
  const source = result?.comment.id || invalidSource?.id || sourceId;
  const url = source ? `https://github.com/${repo.owner}/${repo.repo}/pull/${number}#issuecomment-${source}` : undefined;
  const output = result ? {
    title: {PASS: 'No semantic conflict found (best effort)',
      FAIL: 'Possible semantic conflict', INCONCLUSIVE: 'Semantic analysis inconclusive'}[result.verdict],
    summary: `Request ${request.id}. Head ${request.head}, target ${request.target}, ` +
      `merge base ${request.mergeBase}.\n\n` +
      `Result received ${result.comment.created_at}. [CodeRabbit analysis](${url}).\n\n` +
      (result.missingEvidence ? 'Missing fixed-revision source citations; no verified verdict.\n\n' : '') + notice,
  } : awaiting(request);
  if (source) {
    if (!result) output.summary += `\n\n[Reply without a valid result](${url}).`;
    output.summary += `\n\n<!-- semantic-review-source:${source} -->`;
  }
  const desired = {status: result ? 'completed' : 'in_progress',
    ...(result ? {conclusion: {PASS: 'success', FAIL: 'failure', INCONCLUSIVE: 'neutral'}[result.verdict]} : {}),
    output};
  const changed = check.status !== desired.status ||
    (result ? check.conclusion !== desired.conclusion : check.conclusion != null) ||
    check.output?.title !== output.title || check.output?.summary !== output.summary;
  const create = !result && check.status === 'completed';
  return {request, check, result, create, cleanup,
    update: changed ? {...repo,
      ...(create ? {head_sha: request.head, name: NAME, external_id: identity(number, request)} :
        {check_run_id: check.id}), ...desired} : undefined};
}

async function publish({github, context, core, number, comments, head}) {
  if (number === undefined) {
    if (!context.payload.issue?.pull_request || !isReviewer(context.payload.comment?.user)) return;
    number = context.payload.issue.number;
  }
  const state = await reviewState({github, repo: context.repo, number, comments, head});
  if (state?.update) {
    if (state.create) await github.rest.checks.create(state.update);
    else await github.rest.checks.update(state.update);
    const {title, summary} = state.update.output;
    await core.summary.addRaw(`${title}\n\n${summary}\n`).write();
  }
  for (const check of state?.cleanup || []) {
    await github.rest.checks.update({...context.repo, check_run_id: check.id,
      status: 'completed', conclusion: 'cancelled', output: {
        title: 'Superseded or unrecorded semantic request',
        summary: 'This pending check was superseded or has no current request record. ' +
          'Cancellation clears the inactive check and does not assign an AI verdict.',
      }});
  }
  return state;
}

module.exports = {NAME, notice, supported, eligible, isCommandUser, isReviewer,
  identity, requests, command, awaiting, parseResult, reviewState, publish};
