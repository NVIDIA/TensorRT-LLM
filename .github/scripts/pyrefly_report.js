// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const MARKER = '<!-- pyrefly-diff-report -->';
const WORKFLOW = '.github/workflows/pyrefly-diff.yml';
const escapeHtml = text => String(text).replace(/[&<>]/g,
  character => ({'&': '&amp;', '<': '&lt;', '>': '&gt;'}[character]));

function validate(report) {
  const isText = value => typeof value === 'string';
  const isPositive = value => Number.isSafeInteger(value) && value > 0;
  const isSha = value => /^[a-f0-9]{40}$/.test(value);
  // An incomplete check may have failed before it could resolve the target branch.
  const incomplete = isText(report.error) && report.error !== '';
  const valid = isPositive(report.pr_number) && isSha(report.head_sha) &&
    isText(report.base_ref) && report.base_ref !== '' &&
    (isSha(report.base_sha) || incomplete && report.base_sha === '') &&
    (incomplete || report.error === null) &&
    Array.isArray(report.diagnostics) && report.diagnostics.every(item =>
      item && isText(item.path) && isPositive(item.line) && isText(item.name) && isText(item.message));
  if (!valid) throw new Error('Invalid Pyrefly report.');
}

function bodyFor(report, run, serverUrl) {
  const count = report.diagnostics.length;
  const title = report.error ? '⚠️ Pyrefly check incomplete' :
    count ? `⚠️ Pyrefly: ${count} new diagnostics` : '✅ Pyrefly: no new diagnostics';
  const lines = [MARKER, `### ${title}`, '',
    `Target: <code>${escapeHtml(report.base_ref)}</code> ` +
      `(${report.base_sha.slice(0, 12) || 'unavailable'}). PR head: ${report.head_sha.slice(0, 12)}.`, ''];
  if (report.error) {
    lines.push('The comparison did not finish, so this is not a clean result.',
      `<pre>${escapeHtml(report.error.slice(0, 2000))}</pre>`);
  } else if (count) {
    lines.push('This check is advisory and does not fail CI. ' +
      'Please fix these diagnostics or accept them, then resolve this conversation.', '');
    let shown = 0;
    for (const item of report.diagnostics.slice(0, 50)) {
      const entry = `<code>${escapeHtml(item.path.slice(0, 500))}:${item.line}</code> ` +
        `<code>[${escapeHtml(item.name.slice(0, 100))}]</code>\n` +
        `<pre>${escapeHtml(item.message.slice(0, 1200))}</pre>`;
      if (lines.join('\n').length + entry.length > 48000) break;
      lines.push(entry, '');
      shown++;
    }
    if (shown < count) lines.push(`Showing ${shown} of ${count} diagnostics.`);
    lines.push('Locations refer to the temporary merged source tree.');
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
            nodes { id isResolved comments(first: 1) { nodes { fullDatabaseId body author { __typename login } } } }
            pageInfo { hasNextPage endCursor }
          }
        }
      }
    }`, {...repo, number, cursor});
    const threads = result.repository.pullRequest.reviewThreads;
    for (const thread of threads.nodes) {
      const comment = thread.comments.nodes[0];
      // GraphQL reports the Actions bot as a Bot named `github-actions`; only REST appends `[bot]`.
      if (comment?.author?.__typename === 'Bot' && comment.author.login === 'github-actions' &&
          comment.body.includes(MARKER)) {
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
  if (run.event !== 'pull_request' || run.path !== WORKFLOW ||
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
  const clean = !report.error && !report.diagnostics.length;
  // Only findings open a thread. An incomplete check shows as a failed job instead, so an
  // infrastructure problem leaves no conversation to resolve.
  if (!thread && (clean || report.error)) return;
  const body = bodyFor(report, run, context.serverUrl || 'https://github.com');
  if (!thread) {
    // Only a review comment gives a conversation that can be resolved; anchor it to a changed file.
    const {data: files} = await github.rest.pulls.listFiles({
      ...repo, pull_number: report.pr_number, per_page: 100,
    });
    const file = files.find(item => item.status !== 'removed') || files[0];
    if (!file) {
      core.warning('PR has no changed file on which to create a review thread.');
      return;
    }
    await github.rest.pulls.createReviewComment({...repo, pull_number: report.pr_number,
      commit_id: report.head_sha, path: file.filename, subject_type: 'file', body});
    return;
  }
  await github.rest.pulls.updateReviewComment({...repo, comment_id: thread.comment.fullDatabaseId, body});
  // An incomplete check says nothing about the findings, so it leaves the thread as it is.
  if (!report.error && thread.isResolved !== clean) {
    const mutation = clean ? 'resolveReviewThread' : 'unresolveReviewThread';
    await github.graphql(`mutation($id: ID!) {
      ${mutation}(input: {threadId: $id}) { thread { id } }
    }`, {id: thread.id});
  }
}

module.exports = {MARKER, validate, bodyFor, publish};
