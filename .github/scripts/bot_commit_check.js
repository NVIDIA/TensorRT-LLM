// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Gate for `/bot run`, `/bot skip` and `/bot reuse-pipeline` comments, used by blossom-ci.yml.
//
// The CI job reads the pull request head only when it starts, so code pushed after the review
// (or after the comment) would run on internal CI machines. To prevent this:
//   - a pull request from outside the NVIDIA organization must name the reviewed commit
//     with `--commit <full SHA>`;
//   - when `--commit` is given, it must be the current head of the pull request.
// The CI job (PR_Github) checks the head again when it starts and then runs on that SHA only.

const ORG = 'NVIDIA';
const FULL_SHA = /^[0-9a-f]{40}$/i;
const HEADER = ':no_entry: **Bot command not started**';
const HOW_TO =
  '1. Review the changes of this pull request.\n' +
  '2. Copy the full SHA (40 characters) of the latest commit you reviewed from the **Commits** tab.\n' +
  '3. Post the command again with `--commit <SHA>`, for example `/bot run --commit <SHA>`.\n\n' +
  'CI starts only if `<SHA>` is still the latest commit of this pull request, ' +
  'so CI never runs code that was pushed after your review.';

// Split like Python shlex.split(), which the bot parser (bin/parse_args.py) uses.
// Returns null when the quoting is not closed.
function splitArgs(text) {
  const tokens = [];
  let current = '';
  let inToken = false;
  let quote = null;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quote === "'") {
      if (ch === "'") quote = null;
      else current += ch;
    } else if (quote === '"') {
      if (ch === '"') quote = null;
      else if (ch === '\\' && (text[i + 1] === '"' || text[i + 1] === '\\')) current += text[++i];
      else current += ch;
    } else if (' \t\r\n'.includes(ch)) {
      if (inToken) tokens.push(current);
      current = '';
      inToken = false;
    } else {
      inToken = true;
      if (ch === "'" || ch === '"') quote = ch;
      else if (ch !== '\\') current += ch;
      else if (i + 1 < text.length) current += text[++i];
      else return null;
    }
  }
  if (quote) return null;
  if (inToken) tokens.push(current);
  return tokens;
}

// Read the action and every `--commit` value the same way PR_Github does:
// trim, remove every "/bot", trim, then split.
function parseCommand(body) {
  const tokens = splitArgs(String(body || '').trim().split('/bot').join('').trim());
  if (!tokens) return {action: null, commits: []};
  const commits = [];
  for (let i = 1; i < tokens.length; i++) {
    if (tokens[i] === '--commit') commits.push(tokens[i + 1] ?? '');
    else if (tokens[i].startsWith('--commit=')) commits.push(tokens[i].slice('--commit='.length));
  }
  return {action: tokens[0] ?? null, commits};
}

const sameLogin = (a, b) => Boolean(a) && Boolean(b) && a.toLowerCase() === b.toLowerCase();

// Decide if the command may start CI. `isOrgMember(login)` returns true, false,
// or null when the membership could not be checked.
async function decide({body, pr, isOrgMember}) {
  const {action, commits} = parseCommand(body);
  if (action === 'kill') {
    return {allow: true, reason: 'kill does not run pull request code'};
  }

  if (commits.length > 0) {
    if (commits.length !== 1 || !FULL_SHA.test(commits[0])) {
      return {allow: false, reason: 'bad --commit value',
        message: `${HEADER}: \`--commit\` must be given once, with the full 40-character commit SHA.`};
    }
    if (commits[0].toLowerCase() !== String(pr.head.sha).toLowerCase()) {
      return {allow: false, reason: '--commit is not the pull request head',
        message: `${HEADER}: the commit given by \`--commit\` (\`${commits[0].slice(0, 7).toLowerCase()}\`) ` +
          'is not the latest commit of this pull request. New commits may have been pushed after your review. ' +
          'Please review the latest changes, then post the command again with the SHA of the latest commit.'};
    }
    return {allow: true, reason: '--commit is the pull request head'};
  }

  // No --commit: only allowed when both the author and the owner of the source repository are
  // NVIDIA members. The author opened the pull request, but whoever owns the source repository
  // can push to its branch.
  const author = pr.user?.login;
  const headOwner = pr.head.repo?.owner?.login;
  let internal = false;
  if (author && headOwner) {
    const authorIsMember = await isOrgMember(author);
    const ownerIsMember = sameLogin(headOwner, ORG) ? true
      : sameLogin(headOwner, author) ? authorIsMember : await isOrgMember(headOwner);
    if (authorIsMember === null || ownerIsMember === null) {
      return {allow: false, reason: 'could not check NVIDIA membership',
        message: `${HEADER}: could not check if this pull request is from a member of the NVIDIA GitHub ` +
          'organization, so the command must name the commit to run.\n\n' + HOW_TO +
          '\n\nIf you see this message on pull requests from NVIDIA members, please contact the infra team.'};
    }
    internal = authorIsMember && ownerIsMember;
  }
  if (internal) {
    return {allow: true, reason: 'pull request is from NVIDIA members'};
  }
  return {allow: false, reason: 'external pull request without --commit',
    message: `${HEADER}: the author of this pull request, or the owner of its source repository, is not a ` +
      'member of the NVIDIA GitHub organization, so the command must name the commit to run.\n\n' + HOW_TO};
}

// 204: member. 404: not a member. Anything else (for example 302 when the token
// cannot see private members): unknown.
async function checkOrgMember({fetchImpl, token, login, core}) {
  if (!token) {
    core.warning('No token to check NVIDIA membership.');
    return null;
  }
  try {
    const response = await fetchImpl(
      `https://api.github.com/orgs/${ORG}/members/${encodeURIComponent(login)}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Accept': 'application/vnd.github+json',
          'X-GitHub-Api-Version': '2022-11-28',
          'User-Agent': 'bot-commit-check',
        },
        redirect: 'manual',
      });
    if (response.status === 204) return true;
    if (response.status === 404) return false;
    core.warning(`Membership check for ${login} returned HTTP ${response.status}.`);
  } catch (error) {
    core.warning(`Membership check for ${login} failed: ${error.message}`);
  }
  return null;
}

async function run({github, context, core, orgToken, fetchImpl = fetch}) {
  const {owner, repo} = context.repo;
  const issue = context.payload.issue;
  const comment = context.payload.comment;
  const reject = async (reason, message) => {
    await github.rest.issues.createComment({owner, repo, issue_number: issue.number,
      body: `${message}\n\n[Link to invocation](${comment.html_url})`});
    core.setFailed(`Bot command rejected: ${reason}`);
  };

  try {
    if (!issue.pull_request) {
      console.log('Not a pull request, nothing to check.');
      return;
    }
    const {data: pr} = await github.rest.pulls.get({owner, repo, pull_number: issue.number});
    const known = new Map();
    const isOrgMember = async login => {
      const key = login.toLowerCase();
      if (!known.has(key)) known.set(key, await checkOrgMember({fetchImpl, token: orgToken, login, core}));
      return known.get(key);
    };
    const decision = await decide({body: comment.body, pr, isOrgMember});
    console.log(`PR #${issue.number} by ${pr.user?.login}, source repository owner ` +
      `${pr.head.repo?.owner?.login}, head ${pr.head.sha}: ${decision.reason}`);
    if (!decision.allow) await reject(decision.reason, decision.message);
  } catch (error) {
    // Fail closed: do not start CI when the check itself is broken.
    core.error(`Commit check failed: ${error.stack || error.message}`);
    await reject('the commit check failed with an error',
      `${HEADER}: the commit check failed with an error. Please post the command again. ` +
      'If it fails again, please contact the infra team.');
  }
}

module.exports = {run, decide, parseCommand, splitArgs, checkOrgMember};
