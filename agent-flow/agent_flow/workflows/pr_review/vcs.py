"""Local ``git`` helpers for the pr-review workflow's diff context.

The PR/MR itself is sourced by the *sourcing agent* (see :mod:`.sourcing`),
which runs ``gh`` / ``glab`` itself to check the branch out and read its
metadata — this module never shells out to those binaries. Given the
agent-reported metadata, the helpers here derive everything the reviewer/coder
agents need with local ``git``: the "diff under review" is the diff from the
branch's fork point (merge-base with the base branch) to the **working tree**,
so it includes the coder's uncommitted edits across rounds.

All ``git`` calls go through :func:`_run`, so tests patch a single seam. The
binary resolves from an env override (``GIT_BIN``) then ``PATH``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

GIT_BIN_ENV = "GIT_BIN"

_RUN_TIMEOUT_S = 120


class VcsError(RuntimeError):
    """Raised when a ``git`` invocation (or base-branch resolution) cannot be completed."""


@dataclass(frozen=True)
class PrContext:
    """Everything the orchestrator needs after sourcing the PR/MR.

    The PR/MR may be a GitHub PR or a GitLab MR — the orchestrator does not
    track which (the sourcing agent figures that out). ``diff_base_ref`` is the
    ref the diff is taken against (e.g. ``origin/main``); ``diff_command`` is the
    exact shell command handed to the agents to view the change (a merge-base
    diff that includes the coder's uncommitted working-tree edits).
    """

    identifier: str
    repo: Path
    base_branch: str
    head_branch: str
    diff_base_ref: str
    diff_command: str
    title: str
    author: str
    url: str
    body: str


def resolve_bin(tool: str = "git") -> str:
    """Resolve the local ``git`` binary (env override ``GIT_BIN`` then ``PATH``).

    Only ``"git"`` is supported — sourcing the PR/MR (``gh`` / ``glab``) is the
    sourcing agent's job, not this module's. Raises :class:`VcsError` with an
    actionable message when ``git`` is missing.
    """
    if tool != "git":
        raise VcsError(f"unknown tool: {tool!r} (this module only runs `git`)")
    override = os.environ.get(GIT_BIN_ENV)
    candidate = override or "git"
    resolved = shutil.which(candidate) or (candidate if override else None)
    if resolved is None:
        raise VcsError(f"`{candidate}` not found on PATH. Install `git`.")
    return resolved


def _run(cmd: list[str], *, cwd: Path | None = None) -> str:
    """Run ``cmd``, returning stdout. Raise :class:`VcsError` on failure."""
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=_RUN_TIMEOUT_S,
        )
    except FileNotFoundError as exc:
        raise VcsError(f"command not found: {cmd[0]}") from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise VcsError(f"failed to run {' '.join(cmd)}: {exc}") from exc
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise VcsError(f"`{' '.join(cmd)}` exited {proc.returncode}: {stderr}")
    return proc.stdout


def _rev_parse_ok(repo: Path, ref: str) -> bool:
    """Return True iff ``ref`` resolves in ``repo`` (``git rev-parse --verify``)."""
    git = resolve_bin("git")
    proc = subprocess.run(
        [git, "-C", str(repo), "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def resolve_diff_base_ref(repo: Path, base_branch: str) -> str:
    """Pick the local ref to diff against for ``base_branch``.

    Prefers the remote-tracking ``origin/<base_branch>`` (stable across the
    run) and falls back to the bare branch name. Raises :class:`VcsError` if
    neither resolves so the user can correct ``--base``.
    """
    for candidate in (f"origin/{base_branch}", base_branch):
        if _rev_parse_ok(repo, candidate):
            return candidate
    raise VcsError(
        f"base ref {base_branch!r} not found in {repo} (tried 'origin/{base_branch}' "
        f"and '{base_branch}'). Fetch it or pass --base with a ref that exists locally."
    )


def merge_base(repo: Path, ref: str) -> str:
    """Return the merge-base SHA of ``ref`` and ``HEAD`` in ``repo``."""
    out = _run([resolve_bin("git"), "-C", str(repo), "merge-base", ref, "HEAD"], cwd=repo)
    sha = out.strip()
    if not sha:
        raise VcsError(f"could not compute merge-base of {ref} and HEAD in {repo}")
    return sha


def diff_command(repo: Path, diff_base_ref: str) -> str:
    """Return the shell command the agents run to view the diff under review.

    A merge-base (two-dot) diff to the working tree, so it includes the
    coder's *uncommitted* edits across rounds. Quoted so paths with spaces
    survive.
    """
    repo_q = repo
    return f'git -C "{repo_q}" diff "$(git -C "{repo_q}" merge-base {diff_base_ref} HEAD)"'


def diff_stat(repo: Path, diff_base_ref: str) -> str:
    """Return ``git diff --stat`` from the merge-base to the working tree."""
    base_sha = merge_base(repo, diff_base_ref)
    return _run([resolve_bin("git"), "-C", str(repo), "diff", "--stat", base_sha], cwd=repo)


def build_pr_context(
    repo: Path,
    identifier: str,
    metadata: dict[str, str],
    base_override: str | None = None,
) -> PrContext:
    """Resolve the diff context for an already-checked-out PR/MR.

    The sourcing agent has already checked the branch out and reported its
    ``metadata`` (keys ``base`` / ``head`` / ``title`` / ``author`` / ``url`` /
    ``body``). This adds only the local ``git`` part: pick the base branch
    (``--base`` overrides the reported base) → resolve the local diff-base ref →
    build the merge-base diff command. No ``gh``/``glab``, no checkout, no
    posting, committing, or pushing.
    """
    identifier = identifier.strip()
    base_branch = (base_override or metadata.get("base") or "").strip()
    if not base_branch:
        raise VcsError(
            "Could not determine the base branch from the PR/MR metadata; pass --base explicitly."
        )
    diff_base_ref = resolve_diff_base_ref(repo, base_branch)
    return PrContext(
        identifier=identifier,
        repo=repo,
        base_branch=base_branch,
        head_branch=(metadata.get("head") or "").strip(),
        diff_base_ref=diff_base_ref,
        diff_command=diff_command(repo, diff_base_ref),
        title=metadata.get("title") or "",
        author=metadata.get("author") or "",
        url=metadata.get("url") or "",
        body=metadata.get("body") or "",
    )


def format_pr_context_md(ctx: PrContext) -> str:
    """Render a ``pr_context.md`` body the agents read as their "task" file."""
    lines = [
        "# PR/MR under review",
        "",
        f"- **Identifier:** {ctx.identifier}",
        f"- **Title:** {ctx.title or '(none)'}",
        f"- **Author:** {ctx.author or '(unknown)'}",
        f"- **URL:** {ctx.url or '(none)'}",
        f"- **Source branch (head):** {ctx.head_branch or '(unknown)'}",
        f"- **Base branch:** {ctx.base_branch}",
        f"- **Local repo:** {ctx.repo}",
        "",
        "## Diff under review",
        "",
        "Run this to see the full change (it includes the coder's uncommitted",
        "working-tree edits, which accumulate across rounds):",
        "",
        "```bash",
        ctx.diff_command,
        "```",
        "",
        "## Description",
        "",
        (ctx.body.strip() or "_(no description provided)_"),
        "",
    ]
    return "\n".join(lines)
