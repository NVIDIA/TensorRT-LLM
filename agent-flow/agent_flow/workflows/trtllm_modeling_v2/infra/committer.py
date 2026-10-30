# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One safe way for concurrent agents to commit into one worktree.

Smiths run concurrently inside a dependency batch, in the same checkout, and
git's index is a single shared file. The prompts tell agents to stage by
explicit path and never `git add -A`; that rule is right but not sufficient.
Two failures survive it:

- **`.git/index.lock` contention.** Two agents committing in the same second
  and one gets ``Unable to create '.git/index.lock': File exists``. An agent
  that reads that as "the repo is broken" then does something creative.
- **Cross-attribution.** Another agent's in-progress file or mid-edit
  contract lands in an unrelated commit, splitting one entry's attribution
  across two commits and leaving dangling cross-references.

The fix is a critical section plus a verification, and neither is judgement,
so neither belongs in a prompt. This module generates a small shell script
into the workspace; the agent prompts hand out its path and nothing else.
What the script guarantees:

1. **Serialized.** An ``flock`` around the whole stage-and-commit, with a
   bounded wait, so concurrency becomes a short queue rather than an error.
2. **Scoped.** Only explicit, repo-relative file paths are accepted: no
   directory, no glob, no absolute path, no ``..``. ``git commit -- <paths>``
   then commits the working-tree content of exactly those paths on top of
   ``HEAD``, whatever else sits in the index.
3. **Filtered.** A path is refused before anything is staged when it is

   - a symlink git does not track, or lies beneath one. A worktree links the
     main checkout's native build (``tensorrt_llm/bindings`` and friends);
     ``.gitignore`` names the directory form only, so the link form shows up
     as untracked and ``git add -A`` commits it;
   - a sandbox placeholder: a ``.mcp.json`` or anything under a ``.claude/``
     or ``.cc-writes/`` directory, which the operator's sandbox leaves in
     every directory its shell enters;
   - outside the policy's allowed path prefixes, or inside a denied one (the
     run's own workspace, when it sits inside the repository).

4. **Signed as the repository's user.** The author, the committer and the
   ``Signed-off-by`` trailer are one identity: the repository's configured
   ``user.name`` / ``user.email``, resolved when the script is installed and
   baked into it. A commit whose sign-off email differs from its author
   email fails the DCO check.
5. **Verified.** After committing it checks the new commit against its
   parent and **undoes it** unless the file list is exactly what was asked
   for, no symlink was added, every ``Signed-off-by`` carries the author's
   email, and no ``Co-authored-by`` trailer is present (unless the policy
   allows one). Misattribution stops being something a reviewer notices later
   and becomes something the tool refuses to do.

A private ``GIT_INDEX_FILE`` per dispatch is deliberately not used. It
would stop one agent's ``add`` from seeing another's staged files, but an
index seeded from ``HEAD`` before a concurrent commit lands produces a tree
that silently *reverts* that commit — trading a visible attribution bug for
an invisible data-loss one.
"""

from __future__ import annotations

import os
import shlex
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

#: How long an agent waits for the commit lock before giving up. Generous
#: because the section is milliseconds of git and the alternative — an agent
#: that decides the lock is stale — is much worse than a slow queue.
LOCK_TIMEOUT_S = 300

SCRIPT_NAME = "commit.sh"

#: The task.yaml block that configures the policy. Optional; every key has a
#: default.
COMMIT_POLICY_FIELD = "commit_policy"
COMMIT_POLICY_KEYS: tuple[str, ...] = ("allowed_paths", "allow_co_authors", "base_ref")

#: Where an onboarding writes, as repo-relative prefixes. The target, the
#: catalog and its tests live under ``tensorrt_llm/`` and ``tests/``; a new
#: kernel may need ``cpp/``; a deployment example ``examples/``. Everything
#: else in the repository is somebody else's.
DEFAULT_ALLOWED_PATHS: tuple[str, ...] = ("tensorrt_llm/", "tests/", "cpp/", "examples/")

#: File names the operator's sandbox drops into every directory its shell
#: enters, and directory names it creates there.
SANDBOX_PLACEHOLDER_NAMES: tuple[str, ...] = (".mcp.json",)
SANDBOX_PLACEHOLDER_DIRS: tuple[str, ...] = (".claude", ".cc-writes")

#: Exit codes. 2 is a refusal before anything was staged, 1 a commit that was
#: made and undone (or failed), 75 a lock that was never granted.
EXIT_REFUSED = 2
EXIT_UNDONE = 1
EXIT_LOCK_TIMEOUT = 75

ENV_NAME = "MV2FLOW_GIT_NAME"
ENV_EMAIL = "MV2FLOW_GIT_EMAIL"


class CommitPolicyError(ValueError):
    """The commit policy cannot be established, so nothing may commit."""


@dataclass(frozen=True)
class CommitIdentity:
    """Who authors, commits and signs off every commit the run makes."""

    name: str
    email: str
    #: Where the identity came from, for the preflight report.
    source: str = ""

    def render(self) -> str:
        return f"{self.name} <{self.email}>"


@dataclass(frozen=True)
class CommitPolicy:
    """What the commit helper accepts, and as whom it commits.

    ``identity`` is ``None`` only where none could be resolved when the
    script was installed. The script then resolves it from the repository's
    git configuration at commit time and **refuses** if there is none — it
    never invents one from ``$USER``.

    An empty ``allowed_paths`` means "anywhere in the repository"; the
    placeholder and symlink refusals apply either way. A run's policy comes
    from :meth:`from_task`, which defaults to :data:`DEFAULT_ALLOWED_PATHS`.
    """

    identity: CommitIdentity | None = None
    allowed_paths: tuple[str, ...] = ()
    denied_paths: tuple[str, ...] = ()
    allow_co_authors: bool = False
    base_ref: str = ""

    @classmethod
    def from_task(
        cls,
        task_data: Mapping[str, Any],
        *,
        workspace: str | Path | None = None,
        identity: CommitIdentity | None = None,
    ) -> "CommitPolicy":
        """The policy a run's ``task.yaml`` asks for.

        The identity is resolved from the task's repository unless one is
        given. A workspace inside the repository is denied: the run's own
        records are never part of a delivery.
        """
        repo = str(task_data.get("trtllm_repo_path") or "")
        block = task_data.get(COMMIT_POLICY_FIELD) or {}
        if identity is None and repo:
            try:
                identity = resolve_commit_identity(repo)
            except CommitPolicyError:
                identity = None
        allowed = block.get("allowed_paths")
        denied: list[str] = []
        if workspace is not None and repo:
            rel = _relative_inside(Path(workspace), Path(repo))
            if rel:
                denied.append(rel.rstrip("/") + "/")
        return cls(
            identity=identity,
            allowed_paths=(
                DEFAULT_ALLOWED_PATHS
                if allowed is None
                else tuple(_as_prefix(str(p)) for p in allowed)
            ),
            denied_paths=tuple(denied),
            allow_co_authors=bool(block.get("allow_co_authors", False)),
            base_ref=str(block.get("base_ref") or ""),
        )


def validate_commit_policy_block(block: Any, errors: list[str]) -> None:
    """Validate ``task.yaml``'s optional ``commit_policy`` block.

    Problems are appended to ``errors`` so the schema can report every one
    of them at once.
    """
    if block is None:
        return
    if not isinstance(block, dict):
        errors.append(f"'{COMMIT_POLICY_FIELD}' must be a mapping, got {type(block).__name__}")
        return
    unknown = sorted(set(block) - set(COMMIT_POLICY_KEYS))
    if unknown:
        errors.append(
            f"'{COMMIT_POLICY_FIELD}' has unknown key(s) {unknown}; known keys are "
            f"{list(COMMIT_POLICY_KEYS)}"
        )
    allowed = block.get("allowed_paths")
    if allowed is not None:
        if not isinstance(allowed, list) or not all(isinstance(p, str) for p in allowed):
            errors.append(f"'{COMMIT_POLICY_FIELD}.allowed_paths' must be a list of strings")
        else:
            for entry in allowed:
                problem = _path_problem(entry)
                if problem:
                    errors.append(
                        f"'{COMMIT_POLICY_FIELD}.allowed_paths' entry {entry!r}: {problem}"
                    )
    if "allow_co_authors" in block and not isinstance(block["allow_co_authors"], bool):
        errors.append(f"'{COMMIT_POLICY_FIELD}.allow_co_authors' must be true or false")
    if "base_ref" in block and not isinstance(block["base_ref"], str):
        errors.append(f"'{COMMIT_POLICY_FIELD}.base_ref' must be a string (a git revision)")


def _path_problem(entry: str) -> str:
    if not entry.strip():
        return "is empty"
    if entry.startswith("/"):
        return "must be repo-relative"
    if any(part in ("..", ".") for part in entry.strip("/").split("/")):
        return "must not contain '.' or '..' components"
    return ""


def _as_prefix(entry: str) -> str:
    """A repo-relative prefix: a bare file name stays one; a directory gets its slash."""
    entry = entry.strip()
    return entry if entry.endswith("/") or "." in entry.rsplit("/", 1)[-1] else entry + "/"


def _relative_inside(path: Path, root: Path) -> str:
    """``path`` relative to ``root`` when it lies inside it, else ``""``."""
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return ""
    text = rel.as_posix()
    return "" if text in ("", ".") else text


def _git_config(repo: str | Path, key: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "config", "--get", key],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def resolve_commit_identity(
    repo: str | Path, *, environ: Mapping[str, str] | None = None
) -> CommitIdentity:
    """The repository's configured user, or the operator's explicit override.

    ``MV2FLOW_GIT_NAME`` / ``MV2FLOW_GIT_EMAIL`` win when **both** are
    set: an operator who sets one means to set the identity, and half an
    override splits author from sign-off. Otherwise the
    repository's own ``git config`` answers (local, then global — git's
    order). There is deliberately no ``$USER`` fallback: a login name with a
    guessed domain is an identity no DCO check will match.
    """
    env = os.environ if environ is None else environ
    name, email = env.get(ENV_NAME, "").strip(), env.get(ENV_EMAIL, "").strip()
    if name and email:
        return CommitIdentity(name=name, email=email, source=f"${ENV_NAME} / ${ENV_EMAIL}")
    name, email = _git_config(repo, "user.name"), _git_config(repo, "user.email")
    if name and email:
        return CommitIdentity(name=name, email=email, source=f"git config of {repo}")
    raise CommitPolicyError(
        f"no commit identity for {repo}: git config user.name={name!r}, user.email={email!r}. "
        f"Every commit the run makes is authored and signed off as the repository's user, "
        f"so set `git -C {repo} config user.name/user.email` (or export both {ENV_NAME} and "
        f"{ENV_EMAIL}) before starting the run."
    )


_SCRIPT = """#!/usr/bin/env bash
# Generated by the trtllm-modeling-v2 orchestrator. Do not edit.
#
# usage: {name} <message> <path> [path...]
#
# Commits exactly the named files, under a lock, as the repository's user
# with a DCO sign-off, and verifies afterwards that the commit contains
# exactly those files and carries exactly the trailers the repository wants.
set -uo pipefail

REPO={repo}
LOCK={lock}
TIMEOUT={timeout}
# The identity is baked when the script is installed. Empty means it could
# not be resolved then; it is resolved from the repository's git config at
# commit time instead, and the commit is refused if there is none.
COMMIT_NAME={commit_name}
COMMIT_EMAIL={commit_email}
ALLOW_CO_AUTHORS={allow_co_authors}
# Repo-relative prefixes a commit may touch (empty: anywhere in the repo),
# and prefixes it may never touch.
ALLOWED_PATHS=({allowed_paths})
DENIED_PATHS=({denied_paths})

refuse() {{
  echo "commit.sh: REFUSED: $*" >&2
  echo "  Nothing was staged or committed. Fix the call and run it again." >&2
  exit {exit_refused}
}}

# A task's own worktree commits through the same rules: only a worktree of
# this very repository, never another checkout.
if [ -n "${{MV2FLOW_COMMIT_TREE:-}}" ]; then
  COMMON=$(git -C "$REPO" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)
  TREE_COMMON=$(git -C "$MV2FLOW_COMMIT_TREE" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)
  if [ -z "$TREE_COMMON" ] || [ "$TREE_COMMON" != "$COMMON" ]; then
    refuse "MV2FLOW_COMMIT_TREE=$MV2FLOW_COMMIT_TREE is not a worktree of $REPO"
  fi
  REPO=$(git -C "$MV2FLOW_COMMIT_TREE" rev-parse --show-toplevel)
fi

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <message> <path> [path...]" >&2
  exit {exit_refused}
fi
MESSAGE="$1"; shift

if [ "$ALLOW_CO_AUTHORS" != 1 ] && printf '%s\\n' "$MESSAGE" | grep -qiE '^[[:space:]]*co-authored-by:'; then
  refuse "the message carries a Co-authored-by trailer; commits in this repository carry none"
fi
if printf '%s\\n' "$MESSAGE" | grep -qiE '^[[:space:]]*signed-off-by:'; then
  refuse "do not write a Signed-off-by line; this script adds it as the repository's user"
fi

tracked() {{
  git -C "$REPO" ls-files --error-unmatch -- "$1" >/dev/null 2>&1
}}

tracked_symlink() {{
  local line
  line=$(git -C "$REPO" ls-files -s -- "$1" | head -n 1)
  [ "${{line%% *}}" = 120000 ] && [ "${{line#*$'\\t'}}" = "$1" ]
}}

check_path() {{
  local p="$1" acc="" part base
  local -a parts
  case "$p" in
    "") refuse "an empty path" ;;
    /*) refuse "$p is absolute; name files relative to the repo root ($REPO)" ;;
    *'*'*|*'?'*|*'['*) refuse "$p is a glob; name every file explicitly" ;;
  esac
  case "/$p/" in
    */../*|*/./*|*//*) refuse "$p is not a plain repo-relative path (no '.', '..' or '//')" ;;
  esac
  base="${{p##*/}}"
  case "$base" in
    {placeholder_names}) refuse "$p is a sandbox placeholder file, never part of a commit" ;;
  esac
  case "/$p/" in
    {placeholder_dirs}) refuse "$p lies in a sandbox placeholder directory, never part of a commit" ;;
  esac
  IFS='/' read -r -a parts <<< "$p"
  for part in "${{parts[@]}}"; do
    acc="${{acc:+$acc/}}$part"
    if [ -L "$REPO/$acc" ] && ! tracked_symlink "$acc"; then
      refuse "$acc is a symlink git does not track (a worktree's link to a build output," \\
        "typically); it is never committed. Name real files only, and do not stage the link"
    fi
  done
  if [ -d "$REPO/$p" ] && [ ! -L "$REPO/$p" ]; then
    refuse "$p is a directory; name each file in it explicitly"
  fi
  if [ ! -e "$REPO/$p" ] && [ ! -L "$REPO/$p" ] && ! tracked "$p"; then
    echo "commit.sh: $p does not exist; commit by path relative to the repo root" >&2
    exit {exit_refused}
  fi
  local d
  for d in "${{DENIED_PATHS[@]}}"; do
    case "$p" in
      "$d"*) refuse "$p is under $d, which this run never commits (its own workspace)" ;;
    esac
  done
  if [ "${{#ALLOWED_PATHS[@]}}" -gt 0 ]; then
    local a ok=0
    for a in "${{ALLOWED_PATHS[@]}}"; do
      case "$p" in
        "$a"*) ok=1 ;;
      esac
    done
    if [ "$ok" != 1 ]; then
      refuse "$p is outside the paths this run may commit: ${{ALLOWED_PATHS[*]}}"
    fi
  fi
}}

for path in "$@"; do
  check_path "$path"
done

if [ -z "$COMMIT_NAME" ]; then COMMIT_NAME=$(git -C "$REPO" config --get user.name 2>/dev/null); fi
if [ -z "$COMMIT_EMAIL" ]; then COMMIT_EMAIL=$(git -C "$REPO" config --get user.email 2>/dev/null); fi
if [ -z "$COMMIT_NAME" ] || [ -z "$COMMIT_EMAIL" ]; then
  refuse "no commit identity: the repository has no user.name/user.email and none was baked in"
fi

exec {fd}>"$LOCK"
if ! flock -w "$TIMEOUT" {fd}; then
  echo "commit.sh: waited ${{TIMEOUT}}s for the commit lock and did not get it." >&2
  echo "Another agent is committing. This is a symptom to report, not to work around." >&2
  exit {exit_lock}
fi

cd "$REPO" || exit 1

# One identity for author, committer and sign-off. Exported rather than set
# with `-c user.*`, because an inherited GIT_AUTHOR_* / GIT_COMMITTER_* beats
# `-c`, and a sign-off is the committer's identity: an operator environment
# that exports either would split them. Never written to config: a redirected
# $HOME takes the real ~/.gitconfig off git's search path, and a stale
# `.git/config.lock` on Lustre can refuse `git config --local` outright.
export GIT_AUTHOR_NAME="$COMMIT_NAME" GIT_AUTHOR_EMAIL="$COMMIT_EMAIL"
export GIT_COMMITTER_NAME="$COMMIT_NAME" GIT_COMMITTER_EMAIL="$COMMIT_EMAIL"

# `git add` so a new file is known to the index; `git commit -- <paths>` so
# the commit carries the working-tree content of exactly these paths on top
# of HEAD, whatever else another agent has staged.
if ! git add -- "$@"; then
  echo "commit.sh: git add failed" >&2
  exit 1
fi
if ! git -c user.name="$COMMIT_NAME" -c user.email="$COMMIT_EMAIL" \\
     commit -s -q -m "$MESSAGE" -- "$@"; then
  echo "commit.sh: git commit failed (nothing to commit is the usual cause)" >&2
  exit 1
fi

SHA=$(git rev-parse HEAD)
ACTUAL=$(git diff-tree --root --no-commit-id --name-only -r "$SHA" | LC_ALL=C sort)
WANTED=$(printf '%s\\n' "$@" | LC_ALL=C sort -u)
PROBLEMS=()
if [ "$ACTUAL" != "$WANTED" ]; then
  PROBLEMS+=("the commit contains [$(echo $ACTUAL)], not exactly the paths asked for [$(echo $WANTED)]")
fi
NEW_LINKS=$(git diff-tree --root --no-commit-id -r --raw "$SHA" \\
  | awk '$2 == "120000" && $1 != ":120000" {{ sub(/^[^\\t]*\\t/, ""); print }}')
if [ -n "$NEW_LINKS" ]; then
  PROBLEMS+=("the commit adds a symlink: $(echo $NEW_LINKS)")
fi
AUTHOR_EMAIL=$(git log -1 --format=%ae "$SHA")
COMMITTER_EMAIL=$(git log -1 --format=%ce "$SHA")
SIGNOFFS=$(git log -1 --format='%(trailers:key=Signed-off-by,valueonly)' "$SHA" | sed '/^$/d')
if [ -z "$SIGNOFFS" ]; then
  PROBLEMS+=("the commit has no Signed-off-by trailer")
fi
while IFS= read -r line; do
  [ -z "$line" ] && continue
  case "$line" in
    *"<$AUTHOR_EMAIL>") ;;
    *) PROBLEMS+=("sign-off '$line' does not carry the author email <$AUTHOR_EMAIL>") ;;
  esac
done <<< "$SIGNOFFS"
if [ "$AUTHOR_EMAIL" != "$COMMIT_EMAIL" ] || [ "$COMMITTER_EMAIL" != "$COMMIT_EMAIL" ]; then
  PROBLEMS+=("author <$AUTHOR_EMAIL> / committer <$COMMITTER_EMAIL> is not the repository user <$COMMIT_EMAIL>")
fi
if [ "$ALLOW_CO_AUTHORS" != 1 ]; then
  CO=$(git log -1 --format='%(trailers:key=Co-authored-by,valueonly)' "$SHA" | sed '/^$/d')
  if [ -n "$CO" ]; then
    PROBLEMS+=("the commit carries Co-authored-by: $(echo $CO)")
  fi
fi

if [ "${{#PROBLEMS[@]}}" -gt 0 ]; then
  echo "commit.sh: REFUSING TO LEAVE THIS COMMIT AS IS ($SHA)." >&2
  printf '  - %s\\n' "${{PROBLEMS[@]}}" >&2
  echo "  asked for:" >&2
  printf '    %s\\n' $WANTED >&2
  echo "  commit $SHA actually contains:" >&2
  printf '    %s\\n' $ACTUAL >&2
  # Single quotes, not double: a backtick inside a double-quoted echo is a
  # command substitution, so writing the reset as an example inside "..."
  # runs it — and then the explicit reset below runs it a second time,
  # rewinding a commit that had nothing to do with this one.
  echo '  Undoing it with: git reset --soft HEAD^' >&2
  echo "  Your work stays in the tree. Report this: a path you named matched more" >&2
  echo "  than you meant, another agent's file was already staged, or a hook or" >&2
  echo "  the environment changed what git wrote." >&2
  if git rev-parse -q --verify "$SHA^" >/dev/null; then
    git reset -q --soft HEAD^
  else
    # A root commit has no parent to reset to; the branch goes back to unborn.
    git update-ref -d HEAD
  fi
  exit {exit_undone}
fi

echo "commit.sh: $SHA"
printf '  %s\\n' $ACTUAL
"""


def _bash_words(values: Iterable[str]) -> str:
    return " ".join(shlex.quote(v) for v in values)


def _case_alternatives(patterns: Iterable[str]) -> str:
    return "|".join(patterns)


def render_script(
    repo: str | Path,
    lock: str | Path,
    *,
    timeout: int = LOCK_TIMEOUT_S,
    policy: CommitPolicy | None = None,
) -> str:
    """The script body, with this run's repo, lock and policy baked in.

    Baked rather than passed so an agent has no argument it can get wrong,
    and so the same call works from any working directory.
    """
    policy = policy or CommitPolicy()
    identity = policy.identity
    return _SCRIPT.format(
        name=SCRIPT_NAME,
        repo=shlex.quote(str(repo)),
        lock=shlex.quote(str(lock)),
        timeout=int(timeout),
        commit_name=shlex.quote(identity.name if identity else ""),
        commit_email=shlex.quote(identity.email if identity else ""),
        allow_co_authors=1 if policy.allow_co_authors else 0,
        allowed_paths=_bash_words(policy.allowed_paths),
        denied_paths=_bash_words(policy.denied_paths),
        placeholder_names=_case_alternatives(SANDBOX_PLACEHOLDER_NAMES),
        placeholder_dirs=_case_alternatives(f"*/{d}/*" for d in SANDBOX_PLACEHOLDER_DIRS),
        exit_refused=EXIT_REFUSED,
        exit_undone=EXIT_UNDONE,
        exit_lock=EXIT_LOCK_TIMEOUT,
        # An explicit high fd, so the redirect cannot collide with a
        # descriptor the surrounding tool already holds open.
        fd=9,
    )


def install_committer(
    workspace: str | Path,
    repo: str | Path,
    *,
    timeout: int = LOCK_TIMEOUT_S,
    policy: CommitPolicy | None = None,
) -> Path:
    """Write the commit helper into the workspace and return its path."""
    workspace = Path(workspace)
    bin_dir = workspace / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    lock = workspace / ".commit.lock"
    path = bin_dir / SCRIPT_NAME
    path.write_text(render_script(repo, lock, timeout=timeout, policy=policy), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    lock.touch(exist_ok=True)
    return path


def trial_commit(policy: CommitPolicy, *, scratch: str | Path | None = None) -> list[str]:
    """Commit once through the helper in a throwaway repository; return the problems.

    This checks not that "the template passes ``-s``" but that "a commit
    made through the helper, in this environment, carries the trailers the
    target repository requires". The environment is the part a template
    cannot vouch for — a global ``core.hooksPath`` whose ``commit-msg`` hook
    appends a trailer, an exported ``GIT_COMMITTER_EMAIL``, a
    ``format.signOff`` setting — and every one of those changes what lands.

    Returns an empty list when the commit is signed off by its author, the
    author is the policy's identity, and no ``Co-authored-by`` slipped in.
    """
    if policy.identity is None:
        return ["no commit identity could be resolved; the run cannot sign its commits"]
    with tempfile.TemporaryDirectory(prefix="mv2flow-commit-trial-", dir=scratch) as tmp:
        root = Path(tmp)
        repo = root / "repo"
        repo.mkdir()
        init = subprocess.run(
            ["git", "-C", str(repo), "init", "-q"], capture_output=True, text=True
        )
        if init.returncode != 0:
            return [f"git init failed: {init.stderr.strip()}"]
        (repo / "trial.txt").write_text("trial\n", encoding="utf-8")
        trial_policy = CommitPolicy(
            identity=policy.identity,
            allowed_paths=(),
            allow_co_authors=policy.allow_co_authors,
        )
        script = install_committer(root / "ws", repo, policy=trial_policy)
        result = subprocess.run(
            [str(script), "[None][chore] trtllm_modeling_v2 commit-helper trial", "trial.txt"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            return [f"the trial commit failed (rc {result.returncode}): {result.stderr.strip()}"]
        findings = audit_commits(
            repo, "HEAD", allow_co_authors=policy.allow_co_authors, include_root=True
        )
        problems = [f.problem for f in findings]
        author = subprocess.run(
            ["git", "-C", str(repo), "log", "-1", "--format=%an <%ae>"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if author != policy.identity.render():
            problems.append(
                f"the trial commit was authored {author!r}, not {policy.identity.render()!r}"
            )
        return problems


@dataclass(frozen=True)
class CommitFinding:
    """One way one commit falls short of the repository's commit rules."""

    sha: str
    subject: str
    problem: str

    def render(self) -> str:
        return f"{self.sha[:10]} {self.subject[:70]}: {self.problem}"


def _email_of(trailer_value: str) -> str:
    value = trailer_value.strip()
    if "<" in value and value.endswith(">"):
        return value.rsplit("<", 1)[1][:-1].strip()
    return ""


def audit_commits(
    repo: str | Path,
    rev_range: str,
    *,
    allow_co_authors: bool = False,
    include_root: bool = False,
    paths: Sequence[str] = (),
) -> list[CommitFinding]:
    """Check every non-merge commit in ``rev_range`` against the commit rules.

    The same rules ``commit.sh`` enforces one commit at a time, applied to a
    whole branch, because not every commit goes through ``commit.sh``: the
    operator's own commits, a cherry-picked fix, a rebase that dropped a
    trailer. Per commit:

    - a ``Signed-off-by`` carrying the author's email exists (what the DCO
      check matches; an additional sign-off is not a violation here, though
      ``commit.sh`` itself only ever writes one);
    - no ``Co-authored-by``, unless allowed;
    - no symlink is added, and no sandbox placeholder file is added.

    ``rev_range`` is anything ``git log`` accepts (``base..HEAD``, or a single
    revision with ``include_root=True`` for "this commit and its ancestors").
    ``paths`` limits the audit to commits touching those repo-relative
    prefixes — a run's own commits, on a branch that also carries other work.
    """
    fmt = (
        "%x1e%H%x1f%ae%x1f%s"
        "%x1f%(trailers:key=Signed-off-by,valueonly,separator=%x1d)"
        "%x1f%(trailers:key=Co-authored-by,valueonly,separator=%x1d)%x1f"
    )
    args = [
        "git",
        "-C",
        str(repo),
        "log",
        "--no-merges",
        "--raw",
        "--no-renames",
        f"--format={fmt}",
    ]
    if not include_root and ".." not in rev_range:
        args.append("-1")
    args.append(rev_range)
    if paths:
        args += ["--", *(p.rstrip("/") for p in paths)]
    out = subprocess.run(args, capture_output=True, text=True)
    if out.returncode != 0:
        raise CommitPolicyError(f"git log {rev_range} failed: {out.stderr.strip()}")
    findings: list[CommitFinding] = []
    for record in out.stdout.split("\x1e"):
        if not record.strip():
            continue
        fields = record.split("\x1f")
        if len(fields) < 6:
            continue
        sha, author_email, subject, signoffs, coauthors, raw = fields[:6]
        signed = [s for s in signoffs.split("\x1d") if s.strip()]
        if not signed:
            findings.append(CommitFinding(sha, subject, "no Signed-off-by"))
        elif not any(_email_of(v).lower() == author_email.strip().lower() for v in signed):
            found = ", ".join(repr(v.strip()) for v in signed)
            findings.append(
                CommitFinding(
                    sha,
                    subject,
                    f"no sign-off carries the author email <{author_email}> (found {found})",
                )
            )
        if not allow_co_authors and any(c.strip() for c in coauthors.split("\x1d")):
            findings.append(CommitFinding(sha, subject, "carries Co-authored-by"))
        for line in raw.splitlines():
            if not line.startswith(":") or "\t" not in line:
                continue
            meta, path = line.split("\t", 1)
            modes = meta.split()
            if len(modes) >= 2 and modes[1] == "120000" and modes[0] != ":120000":
                findings.append(CommitFinding(sha, subject, f"adds a symlink: {path}"))
            if is_sandbox_placeholder(path) and not (len(modes) >= 2 and modes[1] == "000000"):
                findings.append(CommitFinding(sha, subject, f"adds a sandbox placeholder: {path}"))
    return findings


def is_sandbox_placeholder(path: str) -> bool:
    """Whether ``path`` (repo-relative) is a file the operator's sandbox leaves behind."""
    parts = path.strip("/").split("/")
    return parts[-1] in SANDBOX_PLACEHOLDER_NAMES or any(
        part in SANDBOX_PLACEHOLDER_DIRS for part in parts[:-1]
    )


def resolve_base(repo: str | Path, candidates: Sequence[str]) -> str:
    """The merge base of ``HEAD`` with the first candidate ref that resolves, or ``""``."""
    for ref in candidates:
        if not ref:
            continue
        out = subprocess.run(
            ["git", "-C", str(repo), "merge-base", "HEAD", ref], capture_output=True, text=True
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    return ""


#: Where a branch's commits are counted from when the policy names no base.
DEFAULT_BASE_REFS: tuple[str, ...] = ("origin/main", "upstream/main", "main")


def commit_instruction(script: str | Path) -> str:
    """The block every committing role's prompt carries."""
    return (
        "## Committing\n"
        "\n"
        "Other agents work in this same worktree at the same time, so commit through "
        "this script and through nothing else:\n"
        "\n"
        "```bash\n"
        f'{script} "<your commit message>" <path> [path...]\n'
        "```\n"
        "\n"
        "Paths are **relative to the repo root**, explicit, and complete — every "
        "file of your unit, no directories, no globs that could match a file you "
        "did not write.\n"
        "\n"
        "It refuses, before staging anything: a directory, a glob, an absolute path; "
        "a symlink git does not track (a worktree's link to the native build); a "
        "sandbox placeholder (`.mcp.json`, anything under `.claude/`); and any path "
        "outside the ones this run may commit. It commits and signs off as the "
        "repository's configured user — do not write a `Signed-off-by` line, and "
        "write no `Co-authored-by` line.\n"
        "\n"
        "It takes a lock, so you may wait behind another agent; that is normal and "
        "is not an error. It then checks that the commit contains exactly the paths "
        "you named and carries the sign-off, and **undoes the commit** if it does "
        "not, leaving your work in the tree. If that happens, report it rather than "
        "retrying: it means a path you named swept in more than you meant.\n"
        "\n"
        "One commit per unit of work, so attribution and revert stay per-unit. No "
        "other git command that writes — never `push`, never a branch operation, "
        "never a rebase, never `git add -A`. No AI attribution lines in the message.\n"
    )


__all__ = [
    "COMMIT_POLICY_FIELD",
    "COMMIT_POLICY_KEYS",
    "CommitFinding",
    "CommitIdentity",
    "CommitPolicy",
    "CommitPolicyError",
    "DEFAULT_ALLOWED_PATHS",
    "DEFAULT_BASE_REFS",
    "EXIT_LOCK_TIMEOUT",
    "EXIT_REFUSED",
    "EXIT_UNDONE",
    "LOCK_TIMEOUT_S",
    "SANDBOX_PLACEHOLDER_DIRS",
    "SANDBOX_PLACEHOLDER_NAMES",
    "SCRIPT_NAME",
    "audit_commits",
    "commit_instruction",
    "install_committer",
    "is_sandbox_placeholder",
    "render_script",
    "resolve_base",
    "resolve_commit_identity",
    "trial_commit",
    "validate_commit_policy_block",
]
