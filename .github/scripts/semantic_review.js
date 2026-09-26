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
  const records = [...parts[1].matchAll(/^SEMANTIC_RESULT request_id=([^\s]+) head=([^\s]+) target=([^\s]+) merge_base=([^\s]+) verdict=(PASS|FAIL|INCONCLUSIVE)[ \t]*\r?$/gmi)];
  if (records.length !== 1) return;
  const [, id, head, target, mergeBase, rawVerdict] = records[0];
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

async function publish({github, context, core}) {
  if (!context.payload.issue?.pull_request || !isReviewer(context.payload.comment?.user)) return;
  const repo = context.repo;
  const number = context.payload.issue.number;
  const comments = await github.paginate(github.rest.issues.listComments,
    {...repo, issue_number: number, per_page: 100});
  const request = requests(comments)[0];
  if (!request) return;
  // The workflow serializes this read/update with switches to a newer request.
  const {data: check} = await github.rest.checks.get({...repo, check_run_id: request.checkId});
  if (check.app?.slug !== 'github-actions' || check.name !== NAME ||
      check.head_sha !== request.head || check.external_id !== identity(number, request)) return;
  const publishedId = Number(check.details_url?.match(/#issuecomment-(\d+)$/)?.[1]);
  const sourceId = publishedId > request.commentId ? publishedId : 0;
  const replies = comments.filter(comment => isReviewer(comment.user) &&
    comment.id > request.commentId && (!sourceId || comment.id >= sourceId) &&
    Date.parse(comment.created_at) >= Date.parse(request.created_at))
    .sort((a, b) => b.id - a.id);
  let result;
  let invalidSource;
  for (const comment of replies) {
    const parsed = parseResult(comment, request, repo);
    if (parsed) { result = parsed; break; }
    // An invalid current-request reply must not resurrect an earlier PASS.
    if (comment.id === sourceId || comment.body?.includes(request.id)) {
      invalidSource = comment;
      break;
    }
  }
  if (!result && !sourceId && !invalidSource) return;
  const verdict = result?.verdict || 'INCONCLUSIVE';
  const title = {PASS: 'No semantic conflict found (best effort)',
    FAIL: 'Possible semantic conflict', INCONCLUSIVE: 'Semantic analysis inconclusive'}[verdict];
  const url = result?.comment.html_url || invalidSource?.html_url || check.details_url;
  const summary = `Request ${request.id}. Head ${request.head}, target ${request.target}, ` +
    `merge base ${request.mergeBase}.\n\n` +
    (result ? `Result received ${result.comment.created_at}. [CodeRabbit analysis](${url}).\n\n` :
      'The published reply no longer provides a valid result for this request.\n\n') +
    (result?.missingEvidence ? 'Missing fixed-revision source citations; no verified verdict.\n\n' : '') + notice;
  await github.rest.checks.update({...repo, check_run_id: check.id, status: 'completed',
    conclusion: {PASS: 'success', FAIL: 'failure', INCONCLUSIVE: 'neutral'}[verdict],
    ...(url ? {details_url: url} : {}), output: {title, summary}});
  await core.summary.addRaw(`${title}\n\n${summary}\n`).write();
}

module.exports = {NAME, notice, supported, eligible, isCommandUser, isReviewer,
  identity, requests, command, awaiting, parseResult, publish};
