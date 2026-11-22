// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Removes the transient spec roots (speckit design documents) from a pull
// request's branch once the pull request is ready for review, and comments on
// it. Decisions are pure functions; `run()` is the thin layer that calls the
// GitHub API. Branch content is never fetched onto the runner or executed:
// only tree listings and, for restorations, blobs of the target branch are read.

const TARGET_REPOSITORY = process.env.GITHUB_REPOSITORY || 'NVIDIA/TensorRT-LLM';
const SPEC_ROOTS = ['specs', 'openspec/changes'];
const NOTICE_AUTHOR = 'github-actions[bot]';
const TRIGGER_ACTIONS = new Set(['opened', 'reopened', 'ready_for_review', 'synchronize']);
const REMOVAL_MARKER = /<!-- specs-removal:([0-9a-f]{40}) -->/;
const FAILURE_MARKER = /<!-- specs-removal-failed:([a-z_]+) -->/;
const MAX_LISTED_DIRS = 10;
// GitHub lists at most this many changed files in a comparison.
const COMPARE_FILE_LIMIT = 300;
// Same ticket and type rules as the PR title check in pr-check.yml.
const TITLE_PREFIX = /^\[(None|[A-Z0-9]+-[0-9]+|#[0-9]+|https:\/\/nvbugs\/[0-9]+)\] *\[([a-z0-9]+)\]/;
const SIGNOFF = /^([^<>\n]+?) <([^<>\s]+@[^<>\s]+)>$/;

const REASON_SENTENCES = {
  maintainer_edits_disabled:
    'this pull request does not allow edits by maintainers, so CI cannot push to your branch.',
  fork_owned_by_org: 'the fork belongs to an organization, and GitHub does not let CI push to those branches.',
  protected_branch: 'the branch is protected against the push CI would make.',
  push_rule_rejected: "the project's push rules rejected CI's commit.",
  permission_denied: 'CI does not have permission to push to the branch.',
  unknown: 'the push failed unexpectedly. The workflow run has the details.',
};

const VIEWER_QUERY = 'query { viewer { login name databaseId } }';
const CREATE_COMMIT = `mutation($input: CreateCommitOnBranchInput!) {
  createCommitOnBranch(input: $input) { commit { oid url author { name email } } }
}`;

// Only Spec Kit drafts and OpenSpec changes are transient roots. Compare
// their components case-sensitively on paths as git records them. Paths with
// a `.github` component are refused even inside the folder, so a widened match
// can never let this automation touch workflow files.
function isSpecPath(path) {
  if (typeof path !== 'string') return false;
  const parts = path.split('/');
  return SPEC_ROOTS.some(root => path.startsWith(root + '/')) &&
    parts.every(part => part !== '' && part !== '.' && part !== '..') &&
    !parts.some(part => part.toLowerCase() === '.github');
}

function specChangePaths(paths) {
  return paths.filter(isSpecPath);
}

function decide(payload) {
  const pr = payload?.pull_request;
  const skip = reason => ({act: false, reason});
  if (payload?.repository?.full_name !== TARGET_REPOSITORY) return skip(`not ${TARGET_REPOSITORY}`);
  if (!pr) return skip('the event carries no pull request');
  if (!TRIGGER_ACTIONS.has(payload.action)) return skip(`"${payload.action}" is not a trigger`);
  if (pr.state !== 'open') return skip(`the pull request is ${pr.state}`);
  // Draft state is taken from the event and not re-read before writing.
  if (pr.draft !== false) return skip('the pull request is a draft');
  return {act: true, reason: `"${payload.action}" on a ready pull request`};
}

function featureDir(path) {
  const root = SPEC_ROOTS.find(root => path.startsWith(root + '/'));
  const parts = path.slice(root.length + 1).split('/');
  return parts.length > 1 ? `${root}/${parts[0]}/` : `${root}/`;
}

// `head` and `base` map spec paths to {sha, mode}. `base` is the merge base, so
// the result is exactly what the pull request changes in the spec folder: files
// it added are deleted; files it modified or deleted are restored.
function specChangeSet({head, base = new Map(), changedFiles}) {
  const deletions = [];
  const restorations = [];
  const refused = [];
  const accept = path => {
    if (isSpecPath(path)) return true;
    refused.push(path);
    return false;
  };
  for (const [path, entry] of head) {
    const target = base.get(path);
    if ((target?.sha === entry.sha && target?.mode === entry.mode) || !accept(path)) continue;
    if (target && (target.mode !== entry.mode || target.mode !== '100644')) {
      throw new Error(`Cannot restore spec file mode faithfully through GraphQL: ${path}`);
    }
    if (target) restorations.push({path, sha: target.sha, inHead: true});
    else deletions.push(path);
  }
  for (const [path, target] of base) {
    if (!head.has(path) && accept(path)) {
      if (target.mode !== '100644') throw new Error(`Cannot restore spec file mode faithfully through GraphQL: ${path}`);
      restorations.push({path, sha: target.sha, inHead: false});
    }
  }
  deletions.sort();
  restorations.sort((a, b) => a.path.localeCompare(b.path));
  const paths = [...deletions, ...restorations.map(r => r.path)].sort();
  return {
    deletions, restorations, refused, paths,
    // The changed files that exist in the head, before the removal.
    presentInHead: [...deletions, ...restorations.filter(r => r.inHead).map(r => r.path)].sort(),
    featureDirs: [...new Set(paths.map(featureDir))].sort(),
    fileCount: paths.length,
    isEmpty: paths.length === 0,
    isWholePr: paths.length > 0 && Number.isInteger(changedFiles) && paths.length >= changedFiles,
  };
}

function writeCapability(pr) {
  const headRepo = pr.head?.repo;
  if (!headRepo) return {canWrite: false, reason: 'unknown'};
  if (headRepo.full_name === pr.base?.repo?.full_name) return {canWrite: true, reason: 'ok'};
  if (headRepo.owner?.type !== 'User') return {canWrite: false, reason: 'fork_owned_by_org'};
  if (pr.maintainer_can_modify === false) return {canWrite: false, reason: 'maintainer_edits_disabled'};
  return {canWrite: true, reason: 'ok'};
}

function failureReasonFrom({types = [], status, message}) {
  if (types.includes('STALE_DATA') || /expected branch to point to/i.test(message)) return 'stale';
  if (/protected branch|branch protection|required status check|changes must be made through/i.test(message)) {
    return 'protected_branch';
  }
  if (/repository rule|rule violation|push rule/i.test(message)) return 'push_rule_rejected';
  if (types.includes('FORBIDDEN') || status === 401 || status === 403 ||
      /permission|not accessible|must have push access/i.test(message)) {
    return 'permission_denied';
  }
  return 'unknown';
}

// The message of an API error from the write path: GraphQL errors carry one
// per entry in `errors`, HTTP errors one in `message`.
function apiErrorMessage(error) {
  return Array.isArray(error?.errors)
    ? error.errors.map(e => e?.message ?? '').join('\n')
    : String(error?.message ?? '');
}

// Maps an API error from the write path to a WriteCapability reason, or to
// `stale` when the branch moved under the compare-and-swap. Returns null for
// anything that is not an API error, which the caller must rethrow.
function writeFailureReason(error) {
  if (Array.isArray(error?.errors)) {
    return failureReasonFrom({types: error.errors.map(e => e?.type), message: apiErrorMessage(error)});
  }
  if (Number.isInteger(error?.status)) {
    return failureReasonFrom({status: error.status, message: apiErrorMessage(error)});
  }
  return null;
}

function codeSpan(text) {
  const clean = String(text).replace(/[\r\n]+/g, ' ');
  const longest = Math.max(0, ...(clean.match(/`+/g) ?? []).map(run => run.length));
  const fence = '`'.repeat(longest + 1);
  const pad = longest > 0 ? ' ' : '';
  return `${fence}${pad}${clean}${pad}${fence}`;
}

function listOf(items) {
  return items.length === 1 ? items[0] : `${items.slice(0, -1).join(', ')} and ${items.at(-1)}`;
}

function describeDirs(dirs) {
  const shown = dirs.slice(0, MAX_LISTED_DIRS).map(codeSpan);
  if (dirs.length > shown.length) shown.push(`${dirs.length - shown.length} more`);
  return listOf(shown);
}

const fileCount = n => (n === 1 ? '1 file' : `${n} files`);
const shortSha = sha => `\`${sha.slice(0, 7)}\``;
const commitLink = ({sha, url}) => `[${shortSha(sha)}](${url})`;

function titlePrefix(title) {
  const match = TITLE_PREFIX.exec(title ?? '');
  return match ? `[${match[1]}][${match[2]}]` : '[None][chore]';
}

function fullRemovalNotice({commit, parentSha, changeSet, baseRef}) {
  const whole = changeSet.isWholePr
    ? ' The folder was everything this pull request changed, so it now has no changes.' : '';
  const removal = shortSha(commit.sha);
  return [
    '### Design documents removed from this branch',
    '',
    `This pull request is ready for review, so CI removed the \`${SPEC_ROOTS.map(root => root + '/').join(' and ')}\` folder from its branch in ` +
      `${commitLink(commit)}. Those documents are for review while a pull request is a draft; they are kept ` +
      `out of the ready diff and off ${codeSpan(baseRef)}.${whole}`,
    '',
    `Removed ${fileCount(changeSet.fileCount)} from ${describeDirs(changeSet.featureDirs)}.`,
    '',
    '**Your local copy is safe, but take these three steps before you pull.** They keep every spec file, ' +
      'including edits you have not committed:',
    '',
    '    git rm -r --cached --ignore-unmatch specs openspec/changes',
    '    echo "/specs/" >> "$(git rev-parse --git-path info/exclude)"',
    '    echo "/openspec/changes/" >> "$(git rev-parse --git-path info/exclude)"',
    '    git pull --ff-only',
    '',
    'The first command stops git tracking the folder without touching your files; the second keeps it out ' +
      'of later commits; the third brings in the removal. Pulling without them deletes the spec files that ' +
      'you have not edited and stops on the ones you have. If that already happened, get them back with:',
    '',
    `    git restore --source=${parentSha} -- specs/ openspec/changes/`,
    '',
    `**Pushing a branch that predates ${removal} — for example by amending and ` +
      'force-pushing — brings the folder back**, and CI removes it again.',
    '',
    'Review comments made on those files while this was a draft are still here, in the Conversation tab, ' +
      'marked as outdated. They no longer appear in Files changed, because the lines they point at are gone.',
    '',
    `**${removal} needs its own \`blossom-ci\` result before this pull request can merge.** If a full CI run ` +
      `passed on ${shortSha(parentSha)}, the commit just before it, comment \`/bot reuse-pipeline\` to reuse ` +
      `that run, since ${removal} changes only design documents. Otherwise, comment \`/bot run\`.`,
    '',
    '**To get updated documents reviewed**, turn this back into a draft and push them again.',
    '',
    `<!-- specs-removal:${commit.sha} -->`,
  ].join('\n');
}

function shortRemovalNotice({commit, changeSet, firstNoticeUrl}) {
  return [
    '### Design documents removed again',
    '',
    `The \`${SPEC_ROOTS.map(root => root + '/').join(' and ')}\` folder came back on this branch, so CI removed it again in ${commitLink(commit)} ` +
      `(${fileCount(changeSet.fileCount)}). See the steps in [the first notice](${firstNoticeUrl}) to keep ` +
      'your local copy.',
    '',
    `<!-- specs-removal:${commit.sha} -->`,
  ].join('\n');
}

function failureNotice({reason, title}) {
  return [
    '### Could not remove the design documents',
    '',
    `This pull request is ready for review and its branch still contains a \`${SPEC_ROOTS.map(root => root + '/').join(' and ')}\` folder, but CI ` +
      `could not remove it: ${REASON_SENTENCES[reason]}`,
    '',
    'Please remove it yourself:',
    '',
    '    git rm -r --cached --ignore-unmatch specs openspec/changes',
    '    echo "/specs/" >> "$(git rev-parse --git-path info/exclude)"',
    '    echo "/openspec/changes/" >> "$(git rev-parse --git-path info/exclude)"',
    `    git commit -m "${titlePrefix(title)} remove design documents from the pull request"`,
    '    git push',
    '',
    `<!-- specs-removal-failed:${reason} -->`,
  ].join('\n');
}

// This automation's earlier notices, oldest first. Only comments posted by the
// workflow's own identity count, so a marker pasted by anyone else is ignored.
function parseNotices(comments) {
  return comments
    .filter(comment => comment.user?.login === NOTICE_AUTHOR)
    .flatMap(comment => {
      const body = comment.body ?? '';
      const created = Date.parse(comment.created_at);
      const removal = REMOVAL_MARKER.exec(body);
      if (removal) return [{kind: 'removal', sha: removal[1], url: comment.html_url, created}];
      const failure = FAILURE_MARKER.exec(body);
      if (failure && Object.hasOwn(REASON_SENTENCES, failure[1])) {
        return [{kind: 'failure', reason: failure[1], url: comment.html_url, created}];
      }
      return [];
    })
    .sort((a, b) => a.created - b.created);
}

// One notice per removal, the short form once a removal notice exists, and no
// repeat of a failure that the newest notice already reports.
function selectNotice(prior, outcome) {
  if (outcome.kind === 'removal') {
    const first = prior.find(notice => notice.kind === 'removal');
    return first ? {form: 'short', firstNoticeUrl: first.url} : {form: 'full'};
  }
  const newest = prior.at(-1);
  if (newest?.kind === 'failure' && newest.reason === outcome.reason) return {form: 'none'};
  return {form: 'failure'};
}

function noticeBody(selection, outcome, pr) {
  if (selection.form === 'failure') return failureNotice({reason: outcome.reason, title: pr.title});
  const details = {...outcome, baseRef: pr.base.ref, firstNoticeUrl: selection.firstNoticeUrl};
  return selection.form === 'short' ? shortRemovalNotice(details) : fullRemovalNotice(details);
}

function signoffIdentity(viewer, override) {
  if (override) {
    const match = SIGNOFF.exec(override.trim());
    if (!match) throw new Error(`SPECS_REMOVAL_SIGNOFF must look like "Name <email>", got ${JSON.stringify(override)}`);
    return {name: match[1], email: match[2]};
  }
  return {name: viewer.name || viewer.login, email: `${viewer.databaseId}+${viewer.login}@users.noreply.github.com`};
}

function commitMessage(identity) {
  return {
    headline: 'Remove design documents from the pull request',
    body: `The ${SPEC_ROOTS.map(root => root + '/').join(' and ')} folder holds design documents for review while a pull request is a draft.\n\n` +
      `Signed-off-by: ${identity.name} <${identity.email}>`,
  };
}

// Lists the blobs under the root spec folder of `treeish`. There is no
// folder-level delete in the GitHub API, so every path is listed individually.
async function enumerateSpecBlobs(github, {owner, repo, treeish}) {
  const {data: rootTree} = await github.rest.git.getTree({owner, repo, tree_sha: treeish});
  if (rootTree.truncated) throw new Error('Root tree was truncated');
  const blobs = new Map();
  for (const root of SPEC_ROOTS) {
    let tree = rootTree;
    let folder;
    const parts = root.split('/');
    for (let index = 0; index < parts.length; index++) {
      folder = tree.tree.find(entry => entry.path === parts[index]);
      if (!folder) break;
      if (folder.type !== 'tree') throw new Error('Spec root is not a directory');
      if (index < parts.length - 1) {
        ({data: tree} = await github.rest.git.getTree({owner, repo, tree_sha: folder.sha}));
        if (tree.truncated) throw new Error('Spec parent tree was truncated');
      }
    }
    if (!folder) continue;
    const {data: subtree} = await github.rest.git.getTree({owner, repo, tree_sha: folder.sha, recursive: '1'});
    if (subtree.truncated) throw new Error(`${owner}/${repo}@${treeish}: ${root}/ tree was truncated`);
    for (const entry of subtree.tree.filter(entry => entry.type !== 'tree')) {
      if (entry.type !== 'blob' || !['100644', '100755'].includes(entry.mode)) {
        throw new Error('Refusing symlink or submodule in a spec tree');
      }
      blobs.set(`${root}/${entry.path}`, {sha: entry.sha, mode: entry.mode});
    }
  }
  return blobs;
}

// The number of files a comparison changes, or undefined when its file list is
// missing or cut off at GitHub's limit.
function comparedFileCount(comparison) {
  const files = comparison.files;
  return Array.isArray(files) && files.length < COMPARE_FILE_LIMIT ? files.length : undefined;
}

// A head without a spec folder has nothing to remove, so the merge base is only
// looked up when the head has one. When the caller has no changed-file count
// for this head, as after a retry, the comparison supplies it.
async function planRemoval(github, {baseRepo, headRepo, baseSha, headSha, changedFiles}) {
  const head = await enumerateSpecBlobs(github, {...headRepo, treeish: headSha});
  if (head.size === 0) return specChangeSet({head, changedFiles});
  const {data: comparison} = await github.rest.repos.compareCommitsWithBasehead({
    ...baseRepo, basehead: `${baseSha}...${headSha}`, per_page: 1,
  });
  const base = await enumerateSpecBlobs(github, {...baseRepo, treeish: comparison.merge_base_commit.sha});
  return specChangeSet({head, base, changedFiles: changedFiles ?? comparedFileCount(comparison)});
}

async function restorationContents(github, baseRepo, restorations) {
  return Promise.all(restorations.map(async ({path, sha}) => {
    const {data} = await github.rest.git.getBlob({...baseRepo, file_sha: sha});
    if (data.encoding !== 'base64') throw new Error(`Unexpected ${data.encoding} encoding for blob ${sha}`);
    return {path, contents: data.content.replace(/\n/g, '')};
  }));
}

// Appends one commit to the branch. `expectedHeadOid` makes the write a
// compare-and-swap: if the author pushed in the meantime, nothing is written.
async function removeSpecs(pusher, {repositoryNameWithOwner, branchName, expectedHeadOid, deletions, additions,
  identity}) {
  const paths = [...deletions, ...additions.map(addition => addition.path)];
  if (paths.length === 0) throw new Error('No spec changes to remove');
  const outside = paths.filter(path => !isSpecPath(path));
  if (outside.length) throw new Error(`Refusing to change paths outside ${SPEC_ROOTS.map(root => root + '/').join(' and ')}: ${outside.join(', ')}`);
  const {createCommitOnBranch} = await pusher.graphql(CREATE_COMMIT, {input: {
    branch: {repositoryNameWithOwner, branchName},
    expectedHeadOid,
    message: commitMessage(identity),
    fileChanges: {deletions: deletions.map(path => ({path})), additions},
  }});
  return createCommitOnBranch.commit;
}

async function postNotice(github, {repo, pr, outcome}) {
  const comments = await github.paginate(github.rest.issues.listComments, {
    ...repo, issue_number: pr.number, per_page: 100,
  });
  const selection = selectNotice(parseNotices(comments), outcome);
  if (selection.form === 'none') return null;
  const {data} = await github.rest.issues.createComment({
    ...repo, issue_number: pr.number, body: noticeBody(selection, outcome, pr),
  });
  return data;
}

async function run({github, pusher, context, core, signoff, archive}) {
  const decision = decide(context.payload);
  core.info(`${decision.act ? 'Acting' : 'Skipped'}: ${decision.reason}`);
  if (!decision.act) return {outcome: 'skipped'};

  const pr = context.payload.pull_request;
  const baseRepo = {owner: context.repo.owner, repo: context.repo.repo};
  const headRepo = pr.head.repo ? {owner: pr.head.repo.owner.login, repo: pr.head.repo.name} : baseRepo;
  // The failure notice says only the reason; the API error's message goes in
  // the run log, which the `unknown` notice points the author to.
  const fail = async (reason, error) => {
    const message = error ? apiErrorMessage(error) : '';
    core.warning(`Could not remove ${SPEC_ROOTS.map(root => root + '/').join(' and ')} from #${pr.number}: ${reason}${message ? ` (${message})` : ''}`);
    await postNotice(github, {repo: baseRepo, pr, outcome: {kind: 'failure', reason}});
    return {outcome: 'failed', reason};
  };

  let headSha = pr.head.sha;
  let changeSet = await planRemoval(github, {baseRepo, headRepo, baseSha: pr.base.sha, headSha,
    changedFiles: pr.changed_files});
  if (changeSet.refused.length) core.warning(`Left in place: ${changeSet.refused.join(', ')}`);
  if (changeSet.isEmpty) {
    core.info(`#${pr.number} has no spec changes`);
    return {outcome: 'none'};
  }

  const capability = writeCapability(pr);
  if (!capability.canWrite) return fail(capability.reason);
  if (!pusher) {
    core.warning('SPECS_REMOVAL_TOKEN is not configured.');
    return fail('permission_denied');
  }

  let identity;
  try {
    identity = signoffIdentity((await pusher.graphql(VIEWER_QUERY)).viewer, signoff);
  } catch (error) {
    const reason = writeFailureReason(error);
    if (reason === null) throw error;
    return fail(reason === 'stale' ? 'unknown' : reason, error);
  }

  let commit;
  for (let attempt = 1; !commit; attempt++) {
    const additions = await restorationContents(github, baseRepo, changeSet.restorations);
    // Archive this exact candidate head before every removal attempt. A stale
    // CAS retry reaches this barrier again and archives its new candidate.
    // A rejected upload or changed/closed PR throws before any branch write.
    if (typeof archive !== 'function') throw new Error('Archive callback is required before removal');
    const archived = await archive({number: pr.number, headSha});
    if (!archived || archived.head_sha !== headSha || !/^[0-9a-f]{64}$/.test(archived.sha256 ?? '')) {
      throw new Error('Refusing removal without a finalized archive for the exact head');
    }
    try {
      commit = await removeSpecs(pusher, {
        repositoryNameWithOwner: pr.head.repo.full_name, branchName: pr.head.ref,
        expectedHeadOid: headSha, deletions: changeSet.deletions, additions, identity,
      });
    } catch (error) {
      const reason = writeFailureReason(error);
      if (reason === null) throw error;
      if (reason !== 'stale' || attempt > 1) return fail(reason === 'stale' ? 'unknown' : reason, error);
      core.info('The branch moved while CI was acting; retrying once on its newest head.');
      const {data: ref} = await github.rest.git.getRef({...headRepo, ref: `heads/${pr.head.ref}`});
      headSha = ref.object.sha;
      changeSet = await planRemoval(github, {baseRepo, headRepo, baseSha: pr.base.sha, headSha});
      if (changeSet.isEmpty) {
        core.info(`#${pr.number} has no spec changes at ${headSha}`);
        return {outcome: 'none'};
      }
    }
  }

  const author = commit.author;
  if (author && (author.name !== identity.name || author.email !== identity.email)) {
    core.warning(`The removal commit is authored as "${author.name} <${author.email}>" but signed off as ` +
      `"${identity.name} <${identity.email}>". Set SPECS_REMOVAL_SIGNOFF to the author identity.`);
  }
  const sha = commit.oid;
  const url = `https://github.com/${context.repo.owner}/${context.repo.repo}/pull/${pr.number}/commits/${sha}`;
  core.info(`Removed ${changeSet.fileCount} spec file(s) from #${pr.number} in ${sha}`);
  await postNotice(github, {repo: baseRepo, pr, outcome: {
    kind: 'removal', commit: {sha, url}, parentSha: headSha, changeSet,
  }});
  return {
    outcome: 'removed', number: pr.number, headRepository: pr.head.repo.full_name,
    parentSha: headSha, removalSha: sha, paths: changeSet.presentInHead,
  };
}

module.exports = {
  REASON_SENTENCES, isSpecPath, specChangePaths, decide, specChangeSet, writeCapability, writeFailureReason,
  titlePrefix, fullRemovalNotice, shortRemovalNotice, failureNotice, parseNotices, selectNotice,
  signoffIdentity, commitMessage, enumerateSpecBlobs, planRemoval, removeSpecs, postNotice, run,
};
