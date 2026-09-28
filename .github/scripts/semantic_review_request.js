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

const { randomUUID } = require('node:crypto');
const {
  NAME, supported, eligible, isCommandUser, requests, identity, command, awaiting, reviewState, publish,
} = require('./semantic_review');

const REQUEST_LIMIT = 30;
const REST_RESERVE = 1000;
const RETRY_AFTER_MS = 2 * 60 * 60 * 1000;
const MATRIX_LIMIT = 256;

function isRateLimitError(error) {
  const headers = error.response?.headers || {};
  return error.code === 'SEMANTIC_REVIEW_QUOTA' || error.status === 429 ||
    (error.status === 403 && (headers['x-ratelimit-remaining'] === '0' ||
      headers['retry-after'] || /rate limit|abuse detection/i.test(error.message)));
}

function quotaError() {
  const error = new Error('Stopped to preserve the REST API reserve.');
  error.code = 'SEMANTIC_REVIEW_QUOTA';
  return error;
}

async function withReserve(github, operation) {
  const { data: rate } = await github.rest.rateLimit.get();
  let remaining = rate.resources.core.remaining;
  const before = () => {
    if (remaining <= REST_RESERVE) throw quotaError();
  };
  const after = (response) => {
    if (response.headers?.['x-ratelimit-remaining'] !== undefined) {
      remaining = Number(response.headers['x-ratelimit-remaining']);
    }
  };
  github.hook.before('request', before);
  github.hook.after('request', after);
  try {
    before();
    return await operation();
  } finally {
    github.hook.remove('request', before);
    github.hook.remove('request', after);
  }
}

function candidate(pr, manual) {
  return manual ? pr.state === 'open' && supported(pr.base.ref) : eligible(pr);
}

async function pending({ github, context, number, manual, now = Date.now(), onVisit }) {
  const repo = context.repo;
  let pr;
  try {
    ({ data: pr } = await github.rest.pulls.get({ ...repo, pull_number: number }));
  } catch (error) {
    if (error.code !== 'SEMANTIC_REVIEW_QUOTA') onVisit?.();
    throw error;
  }
  onVisit?.();
  const comments = await github.paginate(github.rest.issues.listComments, {
    ...repo, issue_number: number, per_page: 100,
  });
  const review = await reviewState({ github, repo, number, comments, head: pr.head.sha });
  const snapshot = { comments, review, head: pr.head.sha };
  if (!candidate(pr, manual)) return { ...snapshot, status: 'ineligible' };
  const branch = pr.base.ref;
  const { data: ref } = await github.rest.git.getRef({ ...repo, ref: `heads/${branch}` });
  const head = pr.head.sha;
  const target = ref.object.sha;
  Object.assign(snapshot, { head, target, branch });
  const history = requests(comments);
  const sameVersion = (r) => r.head === head && r.target === target && r.branch === branch;
  const previous = history.find(sameVersion);
  if (manual || !previous) return { ...snapshot, status: 'ready' };
  if (previous.id === history[0].id && review?.check && review.request.id === previous.id && !review.result &&
      now - Date.parse(previous.created_at) >= RETRY_AFTER_MS &&
      !history.some((r) => sameVersion(r) && r.automaticRetryOf)) {
    return { ...snapshot, status: 'ready', automaticRetryOf: previous.id };
  }
  return { ...snapshot, status: 'unchanged' };
}

async function requestOne({ github, commandGithub, context, core, number, manual = false,
  allowRequest = true, now = Date.now() }) {
  if (!allowRequest) {
    const { data: pr } = await github.rest.pulls.get({ ...context.repo, pull_number: number });
    await publish({ github, context, core, number, head: pr.head.sha });
    return { status: 'reconciled' };
  }
  const snapshot = await pending({ github, context, number, manual, now });
  if (snapshot.review?.update || snapshot.review?.cleanup.length) {
    await publish({ github, context, core, number, comments: snapshot.comments, head: snapshot.head });
  }
  if (snapshot.status !== 'ready') return { status: snapshot.status };
  if (!commandGithub) throw new Error('Missing semantic command token.');
  return withReserve(commandGithub, async () => {
    const { head, target, branch } = snapshot;
    const repo = context.repo;
    const { data: comparison } = await github.rest.repos.compareCommits({
      ...repo, base: target, head, per_page: 1,
    });
    const mergeBase = comparison.merge_base_commit?.sha;
    if (!/^[a-f0-9]{40}$/.test(mergeBase || '')) {
      throw new Error('The comparison did not return a full merge-base SHA.');
    }

    const { data: user } = await commandGithub.rest.users.getAuthenticated();
    if (!isCommandUser(user)) {
      const error = new Error('The command token must belong to the configured service account.');
      error.code = 'SEMANTIC_REVIEW_COMMAND_USER';
      throw error;
    }

    const current = await pending({ github, context, number, manual, now });
    if (current.review?.update || current.review?.cleanup.length) {
      await publish({ github, context, core, number, comments: current.comments, head: current.head });
    }
    if (current.status !== 'ready') return { status: current.status };
    if (current.head !== head || current.target !== target || current.branch !== branch ||
        current.automaticRetryOf !== snapshot.automaticRetryOf) return { status: 'moved' };

    const request = { id: randomUUID(), head, target, mergeBase, branch,
      ...(current.automaticRetryOf ? { automaticRetryOf: current.automaticRetryOf } : {}) };
    const check = {
      ...repo,
      name: NAME,
      external_id: identity(number, request),
      status: 'in_progress',
      started_at: new Date(now).toISOString(),
      details_url: `https://github.com/${repo.owner}/${repo.repo}/pull/${number}`,
      output: awaiting(request),
    };
    const { data: created } = await github.rest.checks.create({ ...check, head_sha: head });
    request.checkId = created.id;
    try {
      await commandGithub.rest.issues.createComment({
        ...repo,
        issue_number: number,
        body: `${command(request)}\n\n<!-- semantic-review-request:${JSON.stringify(request)} -->`,
        request: { retries: 0 },
      });
    } catch (error) {
      let accepted;
      try {
        const comments = await github.paginate(github.rest.issues.listComments, {
          ...repo, issue_number: number, per_page: 100,
        });
        accepted = requests(comments).some((r) => r.id === request.id);
      } catch (readError) {
        core.warning(`PR #${number}: request delivery remains unknown (HTTP ${readError.status || 'unknown'}).`);
      }
      await github.rest.checks.update({
        ...repo,
        check_run_id: request.checkId,
        ...(accepted === false ? { status: 'completed', conclusion: 'cancelled' } :
          { status: 'in_progress' }),
        output: {
          title: 'Request delivery could not be confirmed',
          summary: 'No AI verdict is available. A later scan can recover from the request comment or try again.',
        },
      });
      throw error;
    }
    await publish({ github, context, core, number, head: current.review?.request?.head || head });
    return { status: 'requested', request };
  });
}

async function discover({ github, context, core, now = Date.now(), cursor }) {
  const input = process.env.INPUT_PULL_NUMBER || '';
  const manual = context.eventName === 'workflow_dispatch';
  if ((manual && !/^[1-9]\d*$/.test(input)) || (!manual && input) ||
    (input && !Number.isSafeInteger(Number(input)))) {
    throw new Error('Manual review requires a positive pull request number.');
  }
  if (!manual && cursor !== undefined && (!Number.isSafeInteger(cursor) || cursor <= 0)) {
    throw new Error('The scan cursor must be a positive pull request number.');
  }
  const result = { jobs: [], requested: 0, skipped: 0, failed: 0, limited: false };
  try {
    await withReserve(github, async () => {
      let candidates = manual ? [{ number: Number(input) }] :
        (await github.paginate(github.rest.pulls.list, {
          ...context.repo, state: 'open', sort: 'created', direction: 'desc', per_page: 100,
        })).sort((a, b) => b.number - a.number);
      if (!manual && cursor !== undefined) {
        candidates = candidates.filter(pr => pr.number < cursor)
          .concat(candidates.filter(pr => pr.number >= cursor));
      }
      for (const { number } of candidates) {
        if (result.requested + result.failed >= REQUEST_LIMIT || result.jobs.length >= MATRIX_LIMIT) break;
        try {
          const snapshot = await pending({ github, context, number, manual, now,
            onVisit: manual ? undefined : () => { result.cursor = number; } });
          if (snapshot.status === 'ready') {
            result.jobs.push({ number, allowRequest: true });
            result.requested += 1;
          } else if (!manual && (snapshot.review?.update || snapshot.review?.cleanup.length)) {
            result.jobs.push({ number, allowRequest: false });
          } else result.skipped += 1;
        } catch (error) {
          if (isRateLimitError(error)) throw error;
          result.failed += 1;
          if (result.failed <= 5) {
            core.warning(`PR #${number}: discovery failed (HTTP ${error.status || 'unknown'}).`);
          }
        }
      }
    });
  } catch (error) {
    if (!isRateLimitError(error)) throw error;
    result.limited = true;
  }
  core.info(`Semantic review: ${result.requested} request slots, ${result.jobs.length - result.requested} repairs, ${result.skipped} skipped, ` +
    `${result.failed} failed${result.limited ? '; stopped for API quota' : ''}.`);
  if (result.failed) core.setFailed('Some semantic review candidates could not be read.');
  return result;
}

async function run({ github, commandGithub, context, core, number, allowRequest = true, now = Date.now() }) {
  if (!Number.isSafeInteger(number) || number <= 0) {
    throw new Error('A positive pull request number is required.');
  }
  let result;
  try {
    const operation = () => requestOne({ github, commandGithub, context, core, number,
      allowRequest, now, manual: context.eventName === 'workflow_dispatch' });
    result = allowRequest ? await withReserve(github, operation) : await operation();
  } catch (error) {
    if (isRateLimitError(error)) {
      result = { status: 'limited' };
    } else {
      core.setFailed(`PR #${number}: semantic review request failed (HTTP ${error.status || 'unknown'}).`);
      result = { status: 'failed' };
    }
  }
  const summary = `PR #${number}: semantic review ${result.status}.`;
  core.info(summary);
  if (core.summary) await core.summary.addRaw(summary).write();
  return result;
}

module.exports = { discover, run, requestOne, isRateLimitError, withReserve };
