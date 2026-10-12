# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Split a branch into reviewable PR stacks by path, commit by commit.

A stack is a name, a branch and a set of repo-relative path prefixes; one
stack may claim ``*`` — everything no other stack claims. Every commit in
``base..src`` (first-parent order) is replayed onto the stacks its paths
belong to. A commit touching several stacks is split into one commit per
stack, each sharing its message, author and committer, so attribution and
sign-offs survive unchanged.

Nothing touches a worktree or a checked-out branch: each commit is rebuilt
with ``git apply --cached`` into a temporary index and ``git commit-tree``.
The branches are created (or, with ``force``, reset) to point at the
results. With ``stacked``, each stack is built on the previous stack's tip,
in the order given; otherwise every stack starts at ``base``.

``drop_paths`` removes a path from every commit (default
``tensorrt_llm/bindings``, a symlink one worktree committed by accident), so
it appears in no PR; a commit left with no change is dropped.

Checks, every one of which must pass (``SplitResult.ok``):

- each stack equals ``src`` inside its paths (dropped paths excepted);
- each stack equals its parent (``base``, or the previous stack) outside them;
- no stack commit touches a dropped path;
- every ``forbid`` pattern (``stack -> regex``) finds nothing in that stack's
  paths at its tip — e.g. "the model PR does not import agent_flow";
- every stack commit passes the commit audit (sign-off by author, no
  co-author, no symlink, no placeholder).
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from ..committer import audit_commits
from ._git import git

REST = "*"
DEFAULT_DROP: tuple[str, ...] = ("tensorrt_llm/bindings",)


class SplitError(ValueError):
    """The stack specification is not usable."""


@dataclass(frozen=True)
class Stack:
    name: str
    branch: str
    paths: tuple[str, ...]

    @property
    def is_rest(self) -> bool:
        return self.paths == (REST,)


def parse_stack(spec: str) -> Stack:
    """``NAME=BRANCH:prefix[,prefix...]``; the prefix list ``*`` means "the rest"."""
    try:
        name, rest = spec.split("=", 1)
        branch, paths = rest.split(":", 1)
    except ValueError as exc:
        raise SplitError(f"stack {spec!r} is not NAME=BRANCH:prefix[,prefix...]") from exc
    prefixes = tuple(p.strip() for p in paths.split(",") if p.strip())
    if not name.strip() or not branch.strip() or not prefixes:
        raise SplitError(f"stack {spec!r} is missing a name, a branch or a path")
    if REST in prefixes and len(prefixes) > 1:
        raise SplitError(f"stack {spec!r}: '*' cannot be combined with other prefixes")
    return Stack(name=name.strip(), branch=branch.strip(), paths=prefixes)


def _under(path: str, prefix: str) -> bool:
    prefix = prefix.rstrip("/")
    return path == prefix or path.startswith(prefix + "/")


@dataclass
class SplitResult:
    tips: dict[str, str] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    split_commits: int = 0
    dropped_commits: int = 0
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(passed for _, passed, _ in self.checks)

    def render(self) -> str:
        lines = [
            "split: "
            + ", ".join(
                f"{name} {self.counts.get(name, 0)} -> {tip[:10]}"
                for name, tip in self.tips.items()
            )
            + f"; {self.split_commits} split, {self.dropped_commits} dropped"
        ]
        for label, passed, detail in self.checks:
            lines.append(
                f"  {'ok  ' if passed else 'FAIL'} {label}" + ("" if passed else f"\n{detail}")
            )
        return "\n".join(lines)


def _replay(
    repo: str, tip: str, commit: str, paths: Sequence[str], env_base: Mapping[str, str]
) -> str:
    """Commit ``commit``'s changes to ``paths`` on top of ``tip``; return the new commit."""
    fd, index = tempfile.mkstemp(prefix="premerge_split_index_")
    os.close(fd)
    os.unlink(index)
    env = {**env_base, "GIT_INDEX_FILE": index}
    try:
        git(repo, "read-tree", tip, env=env)
        patch = git(
            repo,
            "diff",
            "--no-renames",
            "--binary",
            "--full-index",
            f"{commit}^",
            commit,
            "--",
            *paths,
        )
        if patch:
            git(repo, "apply", "--cached", "--whitespace=nowarn", env=env, input_=patch)
        tree = git(repo, "write-tree", env=env).strip()
    finally:
        if os.path.exists(index):
            os.unlink(index)
    meta = git(repo, "log", "-1", "--format=%an%x00%ae%x00%aI%x00%cn%x00%ce%x00%cI", commit).split(
        "\x00"
    )
    message = git(repo, "log", "-1", "--format=%B", commit)
    commit_env = {
        **env_base,
        "GIT_AUTHOR_NAME": meta[0],
        "GIT_AUTHOR_EMAIL": meta[1],
        "GIT_AUTHOR_DATE": meta[2],
        "GIT_COMMITTER_NAME": meta[3],
        "GIT_COMMITTER_EMAIL": meta[4],
        "GIT_COMMITTER_DATE": meta[5].strip(),
    }
    return git(repo, "commit-tree", tree, "-p", tip, env=commit_env, input_=message).strip()


def split_branch(
    repo: str | Path,
    *,
    base: str,
    src: str,
    stacks: Sequence[Stack],
    drop_paths: Sequence[str] = DEFAULT_DROP,
    stacked: bool = False,
    force: bool = False,
    forbid: Mapping[str, str] | None = None,
    allow_co_authors: bool = False,
) -> SplitResult:
    """Replay ``base..src`` onto ``stacks`` and verify the result."""
    repo = str(repo)
    names = [s.name for s in stacks]
    if len(set(names)) != len(names):
        raise SplitError(f"stack names repeat: {names}")
    rests = [s for s in stacks if s.is_rest]
    if len(rests) > 1:
        raise SplitError("at most one stack may claim '*'")
    claimed = [p for s in stacks if not s.is_rest for p in s.paths]

    def owner(path: str) -> str | None:
        for stack in stacks:
            if not stack.is_rest and any(_under(path, p) for p in stack.paths):
                return stack.name
        return rests[0].name if rests else None

    env_base = dict(os.environ)
    base_sha = git(repo, "rev-parse", base).strip()
    src_sha = git(repo, "rev-parse", src).strip()
    commits = git(repo, "rev-list", "--reverse", "--first-parent", f"{base_sha}..{src_sha}").split()
    result = SplitResult(
        tips={s.name: base_sha for s in stacks}, counts={s.name: 0 for s in stacks}
    )
    unowned: list[str] = []
    for commit in commits:
        files = [
            f
            for f in git(
                repo, "show", "--no-renames", "--name-only", "--format=", commit
            ).splitlines()
            if f and not any(_under(f, d) for d in drop_paths)
        ]
        if not files:
            result.dropped_commits += 1
            continue
        parts: dict[str, list[str]] = {}
        for f in files:
            who = owner(f)
            if who is None:
                unowned.append(f"{commit[:10]} {f}")
                continue
            parts.setdefault(who, []).append(f)
        if len(parts) > 1:
            result.split_commits += 1
        for stack in stacks:
            if stack.name in parts:
                result.tips[stack.name] = _replay(
                    repo, result.tips[stack.name], commit, parts[stack.name], env_base
                )
                result.counts[stack.name] += 1
    if unowned:
        raise SplitError(
            "paths no stack claims (add a '*' stack, or claim them):\n  "
            + "\n  ".join(unowned[:20])
        )

    parents = {s.name: base_sha for s in stacks}
    if stacked:
        tip = base_sha
        for stack in stacks:
            stack_commits = git(
                repo, "rev-list", "--reverse", f"{base_sha}..{result.tips[stack.name]}"
            ).split()
            parents[stack.name] = tip
            for commit in stack_commits:
                tip = _replay(repo, tip, commit, _pathspec(stack, claimed), env_base)
            result.tips[stack.name] = tip

    for stack in stacks:
        git(repo, "branch", *(["-f"] if force else []), stack.branch, result.tips[stack.name])

    drop_excludes = [f":(exclude){d}" for d in drop_paths]
    for stack in stacks:
        inside = _pathspec(stack, claimed)
        outside = _complement(stack, claimed)
        tip = result.tips[stack.name]
        same_inside = git(repo, "diff", "--stat", src_sha, tip, "--", *inside, *drop_excludes)
        result.checks.append(
            (f"{stack.name} == src inside its paths", not same_inside.strip(), same_inside)
        )
        same_outside = git(
            repo, "diff", "--stat", parents[stack.name], tip, "--", *outside, *drop_excludes
        )
        result.checks.append(
            (
                f"{stack.name} == its parent outside its paths",
                not same_outside.strip(),
                same_outside,
            )
        )
        leaked = [
            d
            for d in drop_paths
            if git(repo, "log", "--format=%h", f"{parents[stack.name]}..{tip}", "--", d).strip()
        ]
        result.checks.append(
            (
                f"{stack.name} never touches {', '.join(drop_paths) or '(nothing dropped)'}",
                not leaked,
                str(leaked),
            )
        )
        pattern = (forbid or {}).get(stack.name)
        if pattern:
            hits = git(repo, "grep", "-n", "-E", pattern, tip, "--", *inside, check=False)
            result.checks.append(
                (f"{stack.name} has no match for /{pattern}/", not hits.strip(), hits)
            )
        findings = audit_commits(
            repo, f"{parents[stack.name]}..{tip}", allow_co_authors=allow_co_authors
        )
        result.checks.append(
            (
                f"{stack.name} commits pass the commit audit",
                not findings,
                "\n".join(f.render() for f in findings[:20]),
            )
        )
    return result


def _pathspec(stack: Stack, claimed: Sequence[str]) -> list[str]:
    if stack.is_rest:
        return [".", *(f":(exclude){p.rstrip('/')}" for p in claimed)]
    return [p.rstrip("/") for p in stack.paths]


def _complement(stack: Stack, claimed: Sequence[str]) -> list[str]:
    if stack.is_rest:
        return [p.rstrip("/") for p in claimed] or [":(exclude)."]
    return [".", *(f":(exclude){p.rstrip('/')}" for p in stack.paths)]


__all__ = [
    "DEFAULT_DROP",
    "REST",
    "SplitError",
    "SplitResult",
    "Stack",
    "parse_stack",
    "split_branch",
]
