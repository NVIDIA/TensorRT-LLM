# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Git for the run: the questions it asks a checkout, and the rules every commit is held to.

QUESTIONS are asked on this host with an argv, never a shell, and with ``--no-optional-locks``:
a ``git status`` otherwise refreshes the index under ``index.lock``, and the orchestrator asks one
(the target's marker) while a task's branch is fast-forwarded into the same checkout, which then
fails on the lock.

COMMIT RULES. Every tree-changing task commits with plain ``git commit -s`` on a branch in its own
worktree (``run/workspace/integration.py``), as the run's one identity (:meth:`CommitIdentity.env`: the
repository's configured user). Nothing checks a commit as it is made; before a finished task is
integrated, :func:`audit_commits` reads every commit on its branch, and one that breaks a rule
goes back to its worker:

- a ``Signed-off-by`` carries the author's email, and the author is the run's identity; no
  ``Co-authored-by`` unless the policy allows one;
- no symlink is added (a worktree links the main checkout's native build, and ``.gitignore``
  names only the directory form), and no sandbox placeholder: a ``.mcp.json``, or anything under
  ``.claude/`` or ``.cc-writes/``;
- every path lies under the policy's allowed prefixes, and outside the denied ones (the run's own
  workspace, when it sits inside the repository).

The delivery audits its whole branch by the same rules, less the author and the paths.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

# Seconds. A git query that takes longer than this is a sick host, and the
# gate's answer is `unknown` rather than a stall in front of a multi-hour run.
_GIT_TIMEOUT_S = 15


def _git(
    repo: str | Path,
    *args: str,
    timeout: float | None = _GIT_TIMEOUT_S,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess | None:
    """Run ``git -C repo args``; ``None`` when git cannot be started or does not answer in time."""
    try:
        return subprocess.run(
            ["git", "--no-optional-locks", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def git_output(repo_root: str | Path, *args: str) -> str | None:
    """Ask git one question about ``repo_root``. ``None`` if it cannot answer.

    Every failure collapses to ``None``: a gate that dies because a VCS query died reports a
    tooling failure as a modeling failure.
    """
    done = _git(repo_root, *args)
    return done.stdout if done is not None and done.returncode == 0 else None


def git_checked(repo: str | Path, *args: str, check: bool = True) -> str:
    """Run git in ``repo`` and return its stripped stdout; raise on failure when ``check``."""
    done = _git(repo, *args, timeout=None)
    if done is None:
        raise RuntimeError(f"git {' '.join(args)}: git could not be run")
    if check and done.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {done.stderr.strip()}")
    return done.stdout.strip()


def resolve_commit(repo: str | Path, rev: str = "HEAD") -> str | None:
    """The full commit ``rev`` names in ``repo``; ``None`` when it names none or git cannot say."""
    answer = git_output(repo, "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}") or ""
    return answer.strip() or None


#: The optional task.yaml block that configures the policy; every key has a default.
COMMIT_POLICY_FIELD = "commit_policy"
COMMIT_POLICY_KEYS: tuple[str, ...] = ("allowed_paths", "allow_co_authors", "base_ref")

#: Where an onboarding writes, as repo-relative prefixes.
DEFAULT_ALLOWED_PATHS: tuple[str, ...] = ("tensorrt_llm/", "tests/", "cpp/", "examples/")

#: File names the operator's sandbox drops into every directory its shell
#: enters, and directory names it creates there.
SANDBOX_PLACEHOLDER_NAMES: tuple[str, ...] = (".mcp.json",)
SANDBOX_PLACEHOLDER_DIRS: tuple[str, ...] = (".claude", ".cc-writes")

ENV_NAME = "MV2FLOW_GIT_NAME"
ENV_EMAIL = "MV2FLOW_GIT_EMAIL"


class CommitPolicyError(ValueError):
    """The commit policy cannot be established, so nothing may commit."""


@dataclass(frozen=True)
class CommitIdentity:
    """Who authors, commits and signs off every commit the run makes."""

    name: str
    email: str

    def render(self) -> str:
        return f"{self.name} <{self.email}>"

    def env(self) -> dict[str, str]:
        """The variables that make git author, commit and sign off as this identity.

        Exported rather than written to config: an inherited ``GIT_AUTHOR_*`` / ``GIT_COMMITTER_*``
        beats any ``user.*`` setting, and a sign-off is the committer's identity. A redirected
        ``$HOME`` hides the real ``~/.gitconfig``, and a stale ``.git/config.lock`` on Lustre can
        refuse ``git config --local`` outright.
        """
        return {
            "GIT_AUTHOR_NAME": self.name,
            "GIT_AUTHOR_EMAIL": self.email,
            "GIT_COMMITTER_NAME": self.name,
            "GIT_COMMITTER_EMAIL": self.email,
        }


@dataclass(frozen=True)
class CommitPolicy:
    """What the run's commits may touch, and as whom they are made.

    With no ``identity`` (none could be resolved) the audit does not ask who authored a commit;
    an empty ``allowed_paths`` allows the whole repository.
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
        """The policy a run's ``task.yaml`` asks for, as the repository's user unless ``identity``.

        A workspace inside the repository is denied: the run's records are never delivered.
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

    def path_problem(self, path: str) -> str:
        """Why ``path`` (repo-relative) may not be committed under this policy, or ``""``."""
        for prefix in self.denied_paths:
            if path.startswith(prefix):
                return f"{path} is under {prefix}, which this run never commits (its own workspace)"
        if self.allowed_paths and not any(path.startswith(p) for p in self.allowed_paths):
            return (
                f"{path} is outside the paths this run may commit: {', '.join(self.allowed_paths)}"
            )
        return ""


def validate_commit_policy_block(block: Any, errors: list[str]) -> None:
    """Validate ``task.yaml``'s optional ``commit_policy`` block; problems go to ``errors``."""
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


def resolve_commit_identity(
    repo: str | Path, *, environ: Mapping[str, str] | None = None
) -> CommitIdentity:
    """The repository's configured user, or the operator's explicit override.

    ``MV2FLOW_GIT_NAME`` / ``MV2FLOW_GIT_EMAIL`` win only when **both** are set: half an override
    splits author from sign-off. No ``$USER`` fallback: a login name with a guessed domain is an
    identity no DCO check will match.
    """
    env = os.environ if environ is None else environ
    name, email = env.get(ENV_NAME, "").strip(), env.get(ENV_EMAIL, "").strip()
    if name and email:
        return CommitIdentity(name=name, email=email)
    name = (git_output(repo, "config", "--get", "user.name") or "").strip()
    email = (git_output(repo, "config", "--get", "user.email") or "").strip()
    if name and email:
        return CommitIdentity(name=name, email=email)
    raise CommitPolicyError(
        f"no commit identity for {repo}: git config user.name={name!r}, user.email={email!r}. "
        f"Every commit the run makes is authored and signed off as the repository's user, "
        f"so set `git -C {repo} config user.name/user.email` (or export both {ENV_NAME} and "
        f"{ENV_EMAIL}) before starting the run."
    )


def trial_commit(identity: CommitIdentity, *, allow_co_authors: bool = False) -> list[str]:
    """Commit once with ``git commit -s`` in a throwaway repository; return the problems.

    This tests what the rules cannot vouch for, the host's git setup: a global ``core.hooksPath``
    whose ``commit-msg`` hook appends a trailer, ``format.signOff``, a signing key git cannot
    reach. Each changes every agent's commit, and no agent can undo it.
    """
    env = {**os.environ, **identity.env()}
    with tempfile.TemporaryDirectory(prefix="mv2flow-commit-trial-") as tmp:
        (Path(tmp) / "trial.txt").write_text("trial\n", encoding="utf-8")
        for args in (
            ("init", "-q"),
            ("add", "trial.txt"),
            ("commit", "-q", "-s", "-m", "[None][chore] trtllm_modeling_v2 commit trial"),
        ):
            done = _git(tmp, *args, timeout=120, env=env)
            if done is None:
                return [f"git {args[0]} could not be run"]
            if done.returncode != 0:
                return [f"git {args[0]} failed (rc {done.returncode}): {done.stderr.strip()}"]
        policy = CommitPolicy(identity=identity, allow_co_authors=allow_co_authors)
        return [f.problem for f in audit_commits(tmp, "HEAD", policy, own=True)]


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
    policy: CommitPolicy | None = None,
    *,
    paths: Sequence[str] = (),
    own: bool = False,
) -> list[CommitFinding]:
    """Check every non-merge commit in ``rev_range`` (``git log`` syntax) against the commit rules.

    Only ``own`` commits (a task's branch) are held to the author and path rules. ``paths``
    limits the audit to commits touching those repo-relative prefixes.
    """
    policy = policy or CommitPolicy()
    fmt = (
        "%x1e%H%x1f%an <%ae>%x1f%ae%x1f%s"
        "%x1f%(trailers:key=Signed-off-by,valueonly,separator=%x1d)"
        "%x1f%(trailers:key=Co-authored-by,valueonly,separator=%x1d)%x1f"
    )
    args = ["log", "--no-merges", "--raw", "--no-renames", f"--format={fmt}", rev_range]
    if paths:
        args += ["--", *(p.rstrip("/") for p in paths)]
    out = _git(repo, *args, timeout=None)
    if out is None or out.returncode != 0:
        why = "git could not be run" if out is None else out.stderr.strip()
        raise CommitPolicyError(f"git log {rev_range} failed: {why}")
    findings: list[CommitFinding] = []
    for record in out.stdout.split("\x1e"):
        if not record.strip():
            continue
        fields = record.split("\x1f")
        if len(fields) < 7:
            continue
        sha, author, author_email, subject, signoffs, coauthors, raw = fields[:7]

        def found(problem: str) -> None:
            findings.append(CommitFinding(sha, subject, problem))

        signed = [s for s in signoffs.split("\x1d") if s.strip()]
        if not signed:
            found("no Signed-off-by")
        elif not any(_email_of(v).lower() == author_email.strip().lower() for v in signed):
            shown = ", ".join(repr(v.strip()) for v in signed)
            found(f"no sign-off carries the author email <{author_email}> (found {shown})")
        if not policy.allow_co_authors and any(c.strip() for c in coauthors.split("\x1d")):
            found("carries Co-authored-by")
        if own and policy.identity is not None and author != policy.identity.render():
            found(f"authored as {author!r}, not the run's {policy.identity.render()!r}")
        for line in raw.splitlines():
            if not line.startswith(":") or "\t" not in line:
                continue
            meta, path = line.split("\t", 1)
            modes = meta.split()
            if len(modes) >= 2 and modes[1] == "120000" and modes[0] != ":120000":
                found(f"adds a symlink: {path}")
            if is_sandbox_placeholder(path) and not (len(modes) >= 2 and modes[1] == "000000"):
                found(f"adds a sandbox placeholder: {path}")
            if own and policy.path_problem(path):
                found(policy.path_problem(path))
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
        out = _git(repo, "merge-base", "HEAD", ref, timeout=None)
        if out is not None and out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    return ""


#: Where a branch's commits are counted from when the policy names no base.
DEFAULT_BASE_REFS: tuple[str, ...] = ("origin/main", "upstream/main", "main")


def commit_instruction(policy: CommitPolicy) -> str:
    """The block every committing role's prompt carries."""
    where = ", ".join(f"`{p}`" for p in policy.allowed_paths) or "anywhere in the repository"
    return (
        "## Committing\n"
        "\n"
        "Commit in your own worktree with plain git, one commit per unit of work so that "
        "attribution and revert stay per unit:\n"
        "\n"
        "```bash\n"
        "git -C <your worktree> add <path> [path...]\n"
        'git -C <your worktree> commit -s -m "<your commit message>"\n'
        "```\n"
        "\n"
        "Name every file explicitly: no directories, no globs that could match a file you did "
        f"not write, never `git add -A`. Commit only under {where}; never a symlink git does "
        "not track (a worktree's link to the native build), never a sandbox placeholder "
        "(`.mcp.json`, anything under `.claude/`). The run sets the author and the committer: "
        "leave them, sign off with `-s`, and write no `Co-authored-by` line and no AI "
        "attribution.\n"
        "\n"
        "When you finish, every commit on your branch is checked against these rules before "
        "it is integrated, and one that breaks them comes back to you to fix. No other git "
        "command that writes: never `push`, never a branch operation, never a rebase.\n"
    )


NATIVE_ROOT = "tensorrt_llm"


def native_artifacts(repo: str | Path) -> list[str]:
    """Git-ignored entries directly under ``tensorrt_llm/`` in ``repo``: the native build."""
    out = git_checked(
        Path(repo),
        *("ls-files", "--others", "--ignored", "--exclude-standard", "--directory"),
        *("--", f"{NATIVE_ROOT}/"),
        check=False,
    )
    found = []
    for line in out.splitlines():
        parts = line.strip().rstrip("/").split("/")
        if len(parts) == 2 and parts[0] == NATIVE_ROOT and parts[1] != "__pycache__":
            if not parts[1].endswith(".egg-info"):
                found.append("/".join(parts))
    return found


@contextlib.contextmanager
def repo_lock(repo: Path):
    """Serialize worktree creation on one repository.

    Concurrent `git worktree add` on one repository (a campaign stacking a candidate
    while a tuner request opens another) fails with `Could not reset index file to
    revision 'HEAD'`.
    """
    path = Path(git_checked(repo, "rev-parse", "--git-common-dir"))
    if not path.is_absolute():
        path = repo / path
    with open(path / "mv2flow-worktree.lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def add_worktree(repo: Path, tree: Path, *args: str) -> None:
    """`git worktree add` under the repo lock, retried once after cleaning up a failed attempt.

    Only the registration holds the lock (``--no-checkout``); the checkout, the slow part, runs
    outside it against the worktree's own index, so several candidates check out at once.
    """
    with repo_lock(repo):
        for attempt in (1, 2):
            out = subprocess.run(
                ["git", "-C", str(repo), "worktree", "add", "--no-checkout", *args],
                capture_output=True,
                text=True,
            )
            if out.returncode == 0:
                break
            git_checked(repo, "worktree", "remove", "--force", str(tree), check=False)
            git_checked(repo, "worktree", "prune", check=False)
            if attempt == 2:
                raise RuntimeError(f"git worktree add {' '.join(args)}: {out.stderr.strip()}")
            time.sleep(5)
    git_checked(tree, "reset", "--hard", "-q")


def link_native(repo: Path, tree: Path) -> None:
    for rel in native_artifacts(repo):
        target = tree / rel
        if target.exists() or target.is_symlink():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(repo / rel)


__all__ = [
    "repo_lock",
    "native_artifacts",
    "link_native",
    "add_worktree",
    "COMMIT_POLICY_FIELD",
    "COMMIT_POLICY_KEYS",
    "CommitFinding",
    "CommitIdentity",
    "CommitPolicy",
    "CommitPolicyError",
    "DEFAULT_ALLOWED_PATHS",
    "DEFAULT_BASE_REFS",
    "ENV_EMAIL",
    "ENV_NAME",
    "SANDBOX_PLACEHOLDER_DIRS",
    "SANDBOX_PLACEHOLDER_NAMES",
    "audit_commits",
    "commit_instruction",
    "git_checked",
    "git_output",
    "is_sandbox_placeholder",
    "resolve_base",
    "resolve_commit",
    "resolve_commit_identity",
    "trial_commit",
    "validate_commit_policy_block",
]
