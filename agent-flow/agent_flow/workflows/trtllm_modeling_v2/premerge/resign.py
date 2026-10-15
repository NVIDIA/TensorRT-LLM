# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-sign a branch's commits for the DCO check, on a NEW branch.

Non-destructive: the source branch is untouched, no worktree is used and no
hook runs. Every commit in ``base..src`` (linear history only) is rebuilt
with :program:`git commit-tree` on its own tree, so content cannot change;
only the message gains a trailer, and only where one is missing:

    Signed-off-by: <the commit's AUTHOR name> <the commit's AUTHOR email>

and the rebuilt commit's committer is that same author. **Signing as the
operator is the mistake this exists to avoid**: the DCO check matches the
sign-off email to the author email, and one re-sign under the operator's
configured identity left 45 commits with a sign-off that matched nobody.

A commit is rewritten when it has no ``Signed-off-by`` carrying its author's
email; every descendant of a rewritten commit gets a new hash as well. The
result maps every old hash to its new one, so records that cite the old
hashes can be translated.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from ..committer import audit_commits
from ._git import git


class ResignError(RuntimeError):
    """The re-sign cannot be done safely."""


@dataclass
class ResignResult:
    base: str
    new_branch: str
    mapping: list[tuple[str, str, bool, str]] = field(default_factory=list)
    unsigned_left: int = 0
    same_tree: bool = False

    @property
    def rewritten(self) -> int:
        return sum(1 for old, new, _, _ in self.mapping if old != new)

    @property
    def ok(self) -> bool:
        return self.same_tree and self.unsigned_left == 0

    def render_markdown(self, src: str) -> str:
        lines = [
            f"# DCO re-sign: `{src}` -> `{self.new_branch}`",
            "",
            f"- Base: `{self.base[:10]}`. Rewritten commits: {self.rewritten} of {len(self.mapping)}.",
            f"- Commits still failing the sign-off audit on `{self.new_branch}`: {self.unsigned_left}. "
            f"Final tree identical to `{src}`: {'yes' if self.same_tree else 'NO'}.",
            f"- `{src}` is unchanged. Use this map to translate hashes cited against it.",
            "",
            "| # | old | new | re-signed | subject |",
            "|---|---|---|---|---|",
        ]
        for n, (old, new, signed, subject) in enumerate(self.mapping, 1):
            lines.append(
                f"| {n} | `{old[:10]}` | `{new[:10]}` | {'yes' if signed else 'no'} | "
                f"{subject.replace('|', chr(92) + '|')} |"
            )
        return "\n".join(lines) + "\n"


def _needs_signoff(repo: str, commit: str) -> bool:
    return any(
        "Signed-off-by" in f.problem or "sign-off" in f.problem for f in audit_commits(repo, commit)
    )


def resign_branch(repo: str | Path, *, src: str, new_branch: str, base: str) -> ResignResult:
    """Create ``new_branch`` from ``src`` with every commit signed off by its author."""
    repo = str(repo)
    if git(repo, "rev-parse", "--verify", "-q", f"refs/heads/{new_branch}", check=False).strip():
        raise ResignError(f"branch {new_branch} exists; refusing to overwrite it")
    base_sha = git(repo, "merge-base", src, base).strip()
    src_sha = git(repo, "rev-parse", src).strip()
    merges = git(repo, "rev-list", "--merges", f"{base_sha}..{src_sha}").split()
    if merges:
        raise ResignError(
            f"{len(merges)} merge commit(s) in {base_sha[:10]}..{src}; only linear history is re-signed"
        )
    commits = git(repo, "rev-list", "--reverse", f"{base_sha}..{src_sha}").split()
    result = ResignResult(base=base_sha, new_branch=new_branch)
    parent = base_sha
    changed = False
    for commit in commits:
        fields = git(
            repo, "log", "-1", "--format=%an%x00%ae%x00%aI%x00%cn%x00%ce%x00%cI%x00%T%x00%s", commit
        ).split("\x00")
        an, ae, ad, cn, ce, cd, tree, subject = (f.strip() for f in fields)
        sign = _needs_signoff(repo, commit)
        if not sign and not changed:
            result.mapping.append((commit, commit, False, subject))
            parent = commit
            continue
        message = git(repo, "log", "-1", "--format=%B", commit)
        if sign:
            message = git(
                repo,
                "interpret-trailers",
                "--if-exists",
                "addIfDifferent",
                "--trailer",
                f"Signed-off-by: {an} <{ae}>",
                input_=message,
            )
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": an,
            "GIT_AUTHOR_EMAIL": ae,
            "GIT_AUTHOR_DATE": ad,
            # Signed as the author, so the committer is the author too.
            "GIT_COMMITTER_NAME": an if sign else cn,
            "GIT_COMMITTER_EMAIL": ae if sign else ce,
            "GIT_COMMITTER_DATE": cd,
        }
        new = git(repo, "commit-tree", tree, "-p", parent, env=env, input_=message).strip()
        result.mapping.append((commit, new, sign, subject))
        parent = new
        changed = True
    git(repo, "branch", new_branch, parent)
    result.unsigned_left = len(
        [f for f in audit_commits(repo, f"{base_sha}..{new_branch}") if "sign" in f.problem.lower()]
    )
    result.same_tree = (
        git(repo, "rev-parse", f"{src_sha}^{{tree}}").strip()
        == git(repo, "rev-parse", f"{new_branch}^{{tree}}").strip()
    )
    return result


__all__ = ["ResignError", "ResignResult", "resign_branch"]
