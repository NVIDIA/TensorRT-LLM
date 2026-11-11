// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const MARKER = '<!-- pyrefly-diff-report -->';
const BOT = 'github-actions[bot]';
const escapeHtml = text => String(text).replace(/[&<>]/g,
  character => ({'&': '&amp;', '<': '&lt;', '>': '&gt;'}[character]));

function validate(report) {
  if (!Number.isSafeInteger(report.pr_number) || report.pr_number <= 0 ||
      !/^[a-f0-9]{40}$/.test(report.head_sha) ||
      !/^(?:[a-f0-9]{40})?$/.test(report.base_sha) ||
      typeof report.base_ref !== 'string' || !report.base_ref ||
      !(report.error === null || typeof report.error === 'string') ||
      !Array.isArray(report.errors) || report.errors.some(error =>
        !error || typeof error.path !== 'string' || typeof error.name !== 'string' ||
        !Number.isSafeInteger(error.line) || error.line < 1 ||
        typeof (error.description ?? error.concise_description) !== 'string')) {
    throw new Error('Invalid Pyrefly report.');
  }
  if (!report.error && !report.base_sha) throw new Error('Missing comparison base SHA.');
}

function bodyFor(report, run, serverUrl) {
  const title = report.error ? '⚠️ Pyrefly check incomplete' :
    report.errors.length ? `⚠️ Pyrefly: ${report.errors.length} new diagnostics` :
      '✅ Pyrefly: no new diagnostics';
  const lines = [MARKER, `<!-- pyrefly-run:${run.id}:${run.run_attempt} -->`, `### ${title}`, '',
    `Target: <code>${escapeHtml(report.base_ref)}</code> ` +
      `(${report.base_sha.slice(0, 12) || 'unavailable'}). PR head: ${report.head_sha.slice(0, 12)}.`, '',
    'This check is advisory. Please fix or acknowledge the findings, then resolve this conversation.',
    'Type errors do not fail the check.', ''];
  if (report.error) {
    lines.push(`<pre>${escapeHtml(report.error.slice(0, 2000))}</pre>`);
  } else {
    let shown = 0;
    for (const error of report.errors.slice(0, 50)) {
      const entry = `<code>${escapeHtml(error.path.slice(0, 500))}:${error.line}</code> ` +
        `<code>[${escapeHtml(error.name.slice(0, 100))}]</code>\n` +
        `<pre>${escapeHtml((error.description || error.concise_description).slice(0, 1200))}</pre>`;
      if (lines.join('\n').length + entry.length > 48000) break;
      lines.push(entry, '');
      shown++;
    }
    if (shown < report.errors.length) lines.push(`Showing ${shown} of ${report.errors.length} diagnostics.`);
    if (shown) lines.push('Locations refer to the temporary merged source tree.');
  }
  lines.push('', `[Full reports and workflow logs](${serverUrl}/${run.repository.full_name}/actions/runs/${run.id})`);
  return lines.join('\n');
}

async function findThread(github, repo, number) {
  let cursor = null;
  do {
    const result = await github.graphql(`query($owner: String!, $repo: String!, $number: Int!, $cursor: String) {
      repository(owner: $owner, name: $repo) {
        pullRequest(number: $number) {
          reviewThreads(first: 100, after: $cursor) {
            nodes { id isResolved comments(first: 1) { nodes { fullDatabaseId body author { login } } } }
            pageInfo { hasNextPage endCursor }
          }
        }
      }
    }`, {...repo, number, cursor});
    const threads = result.repository.pullRequest.reviewThreads;
    for (const thread of threads.nodes) {
      const comment = thread.comments.nodes[0];
      if (comment?.author?.login === BOT && comment.body.includes(MARKER)) {
        return {...thread, comment};
      }
    }
    cursor = threads.pageInfo.hasNextPage ? threads.pageInfo.endCursor : null;
  } while (cursor);
  return null;
}

async function publish({github, context, core, report}) {
  validate(report);
  const run = context.payload.workflow_run;
  const repo = context.repo;
  if (run.event !== 'pull_request' || run.path !== '.github/workflows/pyrefly-diff.yml' ||
      run.repository.full_name !== `${repo.owner}/${repo.repo}` || run.head_sha !== report.head_sha) {
    throw new Error('Report does not match its source workflow run.');
  }
  const {data: pr} = await github.rest.pulls.get({...repo, pull_number: report.pr_number});
  if (pr.state !== 'open' || pr.head.sha !== report.head_sha || pr.base.ref !== report.base_ref ||
      pr.head.repo?.full_name !== run.head_repository.full_name) {
    core.info('Ignoring a report for a closed or changed PR.');
    return;
  }
  const thread = await findThread(github, repo, report.pr_number);
  const previous = thread?.comment.body.match(/<!-- pyrefly-run:(\d+):(\d+) -->/);
  if (previous && (Number(previous[1]) > run.id ||
      Number(previous[1]) === run.id && Number(previous[2]) > run.run_attempt)) {
    core.info('This run was superseded.');
    return;
  }
  if (!thread && !report.error && !report.errors.length) return;
  const body = bodyFor(report, run, context.serverUrl || 'https://github.com');
  if (!thread) {
    const {data: files} = await github.rest.pulls.listFiles({
      ...repo, pull_number: report.pr_number, per_page: 100,
    });
    const file = files.find(item => item.status !== 'removed') || files[0];
    if (!file) {
      core.warning('PR has no changed file on which to create a review thread.');
      return;
    }
    await github.rest.pulls.createReviewComment({...repo, pull_number: report.pr_number,
      commit_id: report.head_sha, path: file.filename, subject_type: 'file',
      side: file.status === 'removed' ? 'LEFT' : 'RIGHT', body});
    return;
  }
  await github.rest.pulls.updateReviewComment({...repo, comment_id: thread.comment.fullDatabaseId, body});
  const shouldResolve = report.errors.length === 0;
  if (!report.error && thread.isResolved !== shouldResolve) {
    const mutation = shouldResolve ? 'resolveReviewThread' : 'unresolveReviewThread';
    await github.graphql(`mutation($id: ID!) {
      ${mutation}(input: {threadId: $id}) { thread { id } }
    }`, {id: thread.id});
  }
}

module.exports = {MARKER, validate, bodyFor, publish};
