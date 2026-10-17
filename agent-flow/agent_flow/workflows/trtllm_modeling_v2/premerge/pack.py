# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The pre-merge report written at the end of delivery.

The steps that need nothing but git run here, every delivery: the commit
audit over the branch, the copyright check, and the list of test-db lines
that name the delivered model, each with the exact command that runs it the
way CI does. The steps that need a GPU, the network or a decision — the lint
pass, the CI cases themselves, the split, opening the PR — are written into
the report as commands, not run: they belong to the operator, after the
last unit lands.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

from ..committer import DEFAULT_BASE_REFS, CommitFinding, audit_commits, resolve_base
from ._git import GitError, git
from .ci_case import DbEntry, find_test_db_entries
from .copyright import CopyrightReport, check_copyright

PACK_FILENAME = "premerge.md"


@dataclass
class PackReport:
    repo: str
    base: str = ""
    head: str = ""
    hygiene: list[CommitFinding] = field(default_factory=list)
    copyright: CopyrightReport | None = None
    ci_entries: list[DbEntry] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def hygiene_ok(self) -> bool:
        return not self.hygiene

    def render_markdown(self) -> str:
        out = [
            "# Pre-merge pack",
            "",
            f"- repository: `{self.repo}`",
            f"- HEAD: `{self.head[:10] or 'unknown'}`; base: `{self.base[:10] or 'not established'}`",
            f"- written: {datetime.now().isoformat(timespec='seconds')}",
            "",
            f"## Commit audit ({'clean' if self.hygiene_ok else f'{len(self.hygiene)} finding(s)'})",
            "",
        ]
        out += [f"- {f.render()}" for f in self.hygiene] or [
            "Every commit is signed off by its author; no co-author, symlink or placeholder."
        ]
        out += ["", "## Copyright headers", ""]
        if self.copyright is None:
            out.append("Not checked (no base).")
        else:
            out += ["```", self.copyright.render(), "```"]
        out += ["", f"## CI lines naming this model ({len(self.ci_entries)})", ""]
        out += [f"- `{e.yml}`: `{e.line}`" for e in self.ci_entries] or [
            "None found — CI would never run this model's accuracy test."
        ]
        out += ["", "## Remaining steps (run by the operator, in order)", ""]
        out += [f"{n}. `{c}`" for n, c in enumerate(self.commands, 1)]
        if self.notes:
            out += ["", "## Notes", ""] + [f"- {n}" for n in self.notes]
        return "\n".join(out) + "\n"


def build_pack(
    repo: str | Path,
    *,
    test_pattern: str = "",
    base_ref: str = "",
    year: int | None = None,
    allow_co_authors: bool = False,
    paths: Sequence[str] = (),
) -> PackReport:
    """Run the cheap checks and assemble the operator's remaining commands.

    ``paths`` scopes the commit audit to commits touching them (a run's own
    commits); empty audits every commit since the base.
    """
    repo = str(repo)
    report = PackReport(repo=repo)
    try:
        report.head = git(repo, "rev-parse", "HEAD").strip()
    except GitError as exc:
        report.notes.append(f"not a git checkout: {exc}")
        return report
    candidates = [base_ref] if base_ref else list(DEFAULT_BASE_REFS)
    report.base = resolve_base(repo, candidates)
    if report.base:
        report.hygiene = audit_commits(
            repo, f"{report.base}..HEAD", allow_co_authors=allow_co_authors, paths=paths
        )
        report.copyright = check_copyright(
            repo, report.base, "HEAD", year=year or datetime.now().year
        )
    else:
        report.notes.append(
            f"no base: none of {candidates} resolves here, so the commit audit and the "
            f"copyright check did not run. Set `commit_policy.base_ref`."
        )
    if test_pattern:
        report.ci_entries = find_test_db_entries(repo, test_pattern)
    q = shlex.quote
    base = report.base or "<base>"
    report.commands = [
        f"trtllm-modeling-v2 premerge lint --repo {q(repo)} --base {base} --dry-run",
        f"trtllm-modeling-v2 premerge lint --repo {q(repo)} --base {base}",
        *(
            f"trtllm-modeling-v2 premerge ci --repo {q(repo)} --case {q(e.test_id)} "
            f"--launcher '<rank launcher, if the case is multi-GPU>' --out <log dir>"
            for e in report.ci_entries
        ),
        f"trtllm-modeling-v2 premerge hygiene --repo {q(repo)} --base {base}",
        f"trtllm-modeling-v2 premerge split --repo {q(repo)} --base {base} --src HEAD "
        f"--stack 'model=<branch>:*' [--stack 'name=<branch>:<prefix>,...'] [--stacked]",
        "trtllm-modeling-v2 premerge pr-open --upstream NVIDIA/TensorRT-LLM --owner <fork> "
        "--branch <pushed branch> --title '[JIRA][type] ...' --body-file <md>  "
        "(dry run; add --yes to open)",
    ]
    if report.hygiene:
        report.commands.insert(
            0,
            f"trtllm-modeling-v2 premerge resign --repo {q(repo)} --src HEAD --new <new branch> "
            f"--base {base}",
        )
    return report


def write_pack(report: PackReport, directory: str | Path) -> Path:
    path = Path(directory) / PACK_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.render_markdown(), encoding="utf-8")
    return path


def summarize(report: PackReport) -> Sequence[str]:
    """One line per finding class, for a console."""
    lines = [
        f"commit audit: {'clean' if report.hygiene_ok else f'{len(report.hygiene)} finding(s)'}"
    ]
    if report.copyright is not None:
        lines.append(
            f"copyright: {len(report.copyright.missing)} missing, {len(report.copyright.stale)} stale"
        )
    lines.append(f"CI lines naming the model: {len(report.ci_entries)}")
    return lines


__all__ = ["PACK_FILENAME", "PackReport", "build_pack", "summarize", "write_pack"]
