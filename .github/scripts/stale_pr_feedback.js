// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const DAY = 24 * 60 * 60 * 1000;
const WAITING = 'waiting for feedback';
const STALE = 'stale';
const BOT_LOGINS = new Set([
  'coderabbitai', 'coderabbitai[bot]', 'trtllm-agent',
  'svc-trtllm-gh-bot', 'tensorrt-cicd', 'github-actions[bot]',
]);

function isBot(user) {
  const login = (user?.login || '').toLowerCase();
  return user?.type === 'Bot' || login.endsWith('[bot]') || BOT_LOGINS.has(login);
}

// Timeline `committed` events carry git author/committer objects (name, email,
// date) without a GitHub login, so bots are recognised by name or email.
function isBotCommitter(person) {
  const name = (person?.name || '').toLowerCase();
  const email = (person?.email || '').toLowerCase();
  return name.endsWith('[bot]') || BOT_LOGINS.has(name) || email.endsWith('[bot]@users.noreply.github.com');
}

function timestamp(value) {
  const time = Date.parse(value);
  if (!Number.isFinite(time)) throw new Error(`Missing or invalid activity timestamp: ${value}`);
  return time;
}

// Only these timeline events are responses to a feedback request. Everything
// else (labels, assignments, review requests, cross-references, milestones,
// renames) is maintainer bookkeeping or unrelated traffic, regardless of actor.
const ACTIVITY_EVENTS = new Set(['head_ref_force_pushed', 'reopened', 'ready_for_review']);

// Returns one of: skip, mark, answered (author responded before the PR went
// stale: drop waiting for feedback), unstale (author responded: drop both labels),
// lift (another human responded: drop only stale), baseline, close.
function decide({issue, timeline, comments, reviews, reviewComments, head}, now) {
  const labels = issue.labels.map(label => label.name.toLowerCase());
  if (issue.state !== 'open' || !issue.pull_request || !labels.includes(WAITING)) return 'skip';
  const author = (issue.user?.login || '').toLowerCase();
  const isAuthor = user => author !== '' && (user?.login || '').toLowerCase() === author;
  let lastActivity = timestamp(issue.created_at);
  let authorActivity = lastActivity;
  let waitingAt = lastActivity;
  let waitingKnown = false;
  let staleAt;
  const record = (time, byAuthor) => {
    lastActivity = Math.max(lastActivity, time);
    if (byAuthor) authorActivity = Math.max(authorActivity, time);
  };
  for (const event of timeline) {
    const label = event.label?.name?.toLowerCase();
    if (label === WAITING && event.event === 'labeled') { waitingAt = timestamp(event.created_at); waitingKnown = true; }
    if (label === STALE && event.event === 'labeled') staleAt = timestamp(event.created_at);
    if (label === STALE && event.event === 'unlabeled') staleAt = undefined;
    if (!ACTIVITY_EVENTS.has(event.event) || isBot(event.actor)) continue;
    record(timestamp(event.created_at), isAuthor(event.actor));
  }
  // Commits are counted as author activity: they come from the PR branch and
  // the git identity cannot be mapped to a login. Unknown identities must not
  // cause a PR with new code to be closed, so only known bots are excluded.
  for (const event of timeline.filter(event => event.event === 'committed')) {
    if (isBotCommitter(event.author) && isBotCommitter(event.committer)) continue;
    record(timestamp(event.committer?.date || event.created_at), true);
  }
  for (const item of [...comments, ...reviews, ...reviewComments]) {
    if (isBot(item.user) || item.state === 'PENDING') continue;
    record(timestamp(item.updated_at || item.submitted_at || item.created_at), isAuthor(item.user));
  }
  if (!labels.includes(STALE)) {
    // The label workflow clears the request on comments only; pushes and formal
    // reviews by the author are answers too. Legacy PRs without a label event
    // have no request time to compare against and keep the label.
    if (waitingKnown && authorActivity > waitingAt) return 'answered';
    return now - Math.max(lastActivity, waitingAt) >= 14 * DAY ? 'mark' : 'skip';
  }
  // Never infer a closure deadline from updated_at: bot writes also update it.
  if (staleAt === undefined) throw new Error('Stale label has no corresponding timeline event');
  // A pushed commit can retain an old commit date. The recorded head catches
  // these updates as well as rebases, even when the committer is an automation.
  // `assessment-head` is written by the external PR-triage publisher
  // (svc-trtllm-gh-bot) when it labels a PR; `stale-head` and `stale-baseline`
  // by this script. All record the full 40-hex head SHA.
  const baselines = comments.filter(comment =>
    ['svc-trtllm-gh-bot', 'github-actions[bot]'].includes(comment.user?.login)
  ).flatMap(comment => {
    const created = timestamp(comment.created_at);
    const body = comment.body || '';
    const initial = body.match(/<!-- (?:assessment|stale)-head:([0-9a-f]+) -->/);
    const recovery = body.match(/<!-- stale-baseline:(\d+) head:([0-9a-f]+) -->/);
    if (recovery && Number(recovery[1]) === staleAt && created >= staleAt) {
      return [{head: recovery[2], created}];
    }
    if (initial && created <= staleAt) return [{head: initial[1], created}];
    return [];
  }).sort((a, b) => b.created - a.created);
  const baseline = baselines[0];
  const recordedHead = baseline?.head;
  if (head && recordedHead && head !== recordedHead) return 'unstale';
  if (authorActivity > staleAt) return 'unstale';
  if (lastActivity > staleAt) return 'lift';
  // Legacy/manual stale labels may predate head tracking. Start a new grace
  // period rather than closing based on commit dates that do not record pushes.
  if (!recordedHead) return 'baseline';
  return now - Math.max(staleAt, baseline.created) >= 14 * DAY ? 'close' : 'skip';
}

async function readSnapshot(github, repo, number) {
  const args = {...repo, issue_number: number, per_page: 100};
  const {data: issue} = await github.rest.issues.get(args);
  const timeline = await github.paginate(github.rest.issues.listEventsForTimeline, args);
  const comments = await github.paginate(github.rest.issues.listComments, args);
  const pullArgs = {...repo, pull_number: number, per_page: 100};
  const reviews = await github.paginate(github.rest.pulls.listReviews, pullArgs);
  const reviewComments = await github.paginate(github.rest.pulls.listReviewComments, pullArgs);
  const {data: pull} = await github.rest.pulls.get(pullArgs);
  return {issue, timeline, comments, reviews, reviewComments, head: pull.head.sha};
}

async function run({github, context, core, now = Date.now(), dryRun = false}) {
  const repo = context.repo;
  const issues = await github.paginate(github.rest.issues.listForRepo, {
    ...repo, state: 'open', labels: WAITING, per_page: 100,
  });
  for (const candidate of issues.filter(issue => issue.pull_request)) {
    const args = {...repo, issue_number: candidate.number};
    try {
      const before = await readSnapshot(github, repo, candidate.number);
      const action = decide(before, now);
      core.info(`#${candidate.number}: ${action}${dryRun ? ' (dry run, not applied)' : ''}`);
      if (action === 'skip' || dryRun) continue;
      // Re-read all activity before writing; skip changing PRs until the next run.
      const fresh = await readSnapshot(github, repo, candidate.number);
      if (JSON.stringify(before) !== JSON.stringify(fresh)) {
        core.info(`#${candidate.number}: changed while checking; skipped`);
        continue;
      }
      if (action === 'unstale') {
        await github.rest.issues.removeLabel({...args, name: STALE});
        await github.rest.issues.removeLabel({...args, name: WAITING});
        await github.rest.issues.addLabels({...args, labels: ['Investigating']});
      } else if (action === 'answered') {
        await github.rest.issues.removeLabel({...args, name: WAITING});
        await github.rest.issues.addLabels({...args, labels: ['Investigating']});
      } else if (action === 'lift') {
        // Another human responded; the team is still waiting on the author.
        await github.rest.issues.removeLabel({...args, name: STALE});
      } else if (action === 'mark') {
        await github.rest.issues.createComment({...args,
          body: `PR has not received a human update in over 14 days. Adding stale label.\n\n<!-- stale-head:${fresh.head} -->`});
        await github.rest.issues.addLabels({...args, labels: [STALE]});
      } else if (action === 'baseline') {
        const staleEvent = fresh.timeline.filter(event =>
          event.event === 'labeled' && event.label?.name?.toLowerCase() === STALE
        ).at(-1);
        await github.rest.issues.createComment({...args,
          body: `We are restarting the feedback window to allow a full 14 days for an update before automatic closure.\n\n<!-- stale-baseline:${timestamp(staleEvent.created_at)} head:${fresh.head} -->`});
      } else if (action === 'close') {
        // Close before posting so a failed close cannot leave a misleading notice.
        await github.rest.pulls.update({...repo, pull_number: candidate.number, state: 'closed'});
        await github.rest.issues.createComment({...args,
          body: 'This PR was closed because it has been 14 days without human activity since it was marked as stale.'});
      }
    } catch (error) {
      core.setFailed(`#${candidate.number}: ${error.message}`);
    }
  }
}

module.exports = {run, decide, isBot, isBotCommitter};
