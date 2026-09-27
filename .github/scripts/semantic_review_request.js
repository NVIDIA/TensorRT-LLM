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
  NAME, supported, eligible, isCommandUser, requests, identity, command, awaiting,
} = require('./semantic_review');

const REQUEST_LIMIT = 30;
const REST_RESERVE = 1000;

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

async function pending({ github, context, number, manual }) {
  const repo = context.repo;
  const { data: pr } = await github.rest.pulls.get({ ...repo, pull_number: number });
  if (!candidate(pr, manual)) return { status: 'ineligible' };
  const branch = pr.base.ref;
  const { data: ref } = await github.rest.git.getRef({ ...repo, ref: `heads/${branch}` });
  const head = pr.head.sha;
  const target = ref.object.sha;
  const comments = await github.paginate(github.rest.issues.listComments, {
    ...repo, issue_number: number, per_page: 100,
  });
  if (!manual && requests(comments).some((r) =>
    r.head === head && r.target === target && r.branch === branch)) {
    return { status: 'unchanged' };
  }
  return { status: 'ready', head, target, branch };
}

async function requestOne({ github, commandGithub, context, number, manual = false }) {
  const snapshot = await pending({ github, context, number, manual });
  if (snapshot.status !== 'ready') return snapshot;
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

  const checks = await github.paginate(github.rest.checks.listForRef, {
    ...repo, ref: head, check_name: NAME, filter: 'all', per_page: 100,
  });
  const existing = checks.filter((check) => check.app?.slug === 'github-actions' &&
    check.external_id?.startsWith(`semantic-review:${number}:`))
    .sort((a, b) => b.id - a.id)[0];

  const { data: current } = await github.rest.pulls.get({ ...repo, pull_number: number });
  if (!candidate(current, manual) || current.head.sha !== head || current.base.ref !== branch) {
    return { status: 'moved' };
  }
  const { data: currentRef } = await github.rest.git.getRef({
    ...repo, ref: `heads/${branch}`,
  });
  if (currentRef.object.sha !== target) return { status: 'moved' };

  const request = { id: randomUUID(), head, target, mergeBase, branch };
  const check = {
    ...repo,
    name: NAME,
    external_id: identity(number, request),
    status: 'completed',
    conclusion: 'neutral',
    output: awaiting(request),
  };
  if (existing) {
    request.checkId = existing.id;
    await github.rest.checks.update({ ...check, check_run_id: existing.id });
  } else {
    const { data: created } = await github.rest.checks.create({ ...check, head_sha: head });
    request.checkId = created.id;
  }

  try {
    await commandGithub.rest.issues.createComment({
      ...repo,
      issue_number: number,
      body: `${command(request)}\n\n<!-- semantic-review-request:${JSON.stringify(request)} -->`,
      request: { retries: 0 },
    });
  } catch (error) {
    await github.rest.checks.update({
      ...repo,
      check_run_id: request.checkId,
      status: 'completed',
      conclusion: 'neutral',
      output: {
        title: 'Request delivery could not be confirmed',
        summary: 'No AI verdict is available. A later scan can recover from the request comment or try again.',
      },
    });
    throw error;
  }
  return { status: 'requested', request };
}

async function discover({ github, context, core }) {
  const input = process.env.INPUT_PULL_NUMBER || '';
  const manual = context.eventName === 'workflow_dispatch';
  if ((manual && !/^[1-9]\d*$/.test(input)) || (!manual && input) ||
    (input && !Number.isSafeInteger(Number(input)))) {
    throw new Error('Manual review requires a positive pull request number.');
  }
  const result = { numbers: [], skipped: 0, failed: 0, limited: false };
  try {
    await withReserve(github, async () => {
      const candidates = manual ? [{ number: Number(input) }] :
        (await github.paginate(github.rest.pulls.list, {
          ...context.repo, state: 'open', sort: 'created', direction: 'desc', per_page: 100,
        })).filter(eligible).sort((a, b) => b.number - a.number);
      for (const { number } of candidates) {
        if (result.numbers.length + result.failed >= REQUEST_LIMIT) break;
        try {
          const snapshot = await pending({ github, context, number, manual });
          if (snapshot.status === 'ready') result.numbers.push(number);
          else result.skipped += 1;
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
  core.info(`Semantic review: ${result.numbers.length} selected, ${result.skipped} skipped, ` +
    `${result.failed} failed${result.limited ? '; stopped for API quota' : ''}.`);
  if (result.failed) core.setFailed('Some semantic review candidates could not be read.');
  return result;
}

async function run({ github, commandGithub, context, core, number }) {
  if (!Number.isSafeInteger(number) || number <= 0) {
    throw new Error('A positive pull request number is required.');
  }
  let result;
  try {
    result = await withReserve(github, () => withReserve(commandGithub, () => requestOne({
      github, commandGithub, context, number, manual: context.eventName === 'workflow_dispatch',
    })));
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

module.exports = { discover, run, requestOne, isRateLimitError };
