# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA copyright headers on the files a branch changes.

New source files need an NVIDIA copyright header; modified ones need the
current year in it (the repository's AGENTS.md and CODING_GUIDELINES.md).
"Source" is a compiled or interpreted file; documents and data are not
checked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ._git import git

SOURCE_SUFFIXES: tuple[str, ...] = (
    ".py",
    ".pyi",
    ".cpp",
    ".cc",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".sh",
    ".groovy",
)

#: How far into a file a header may start (a shebang, an encoding line).
HEADER_LINES = 25

_HEADER = re.compile(r"Copyright[^\n]*?(\d{4})(?:\s*-\s*(\d{4}))?[^\n]*NVIDIA", re.I)


@dataclass
class CopyrightReport:
    """Files without an NVIDIA header, and files whose header is out of date."""

    year: int
    missing: list[tuple[str, str]] = field(default_factory=list)
    stale: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.missing and not self.stale

    def render(self) -> str:
        lines = [f"== missing NVIDIA header ({len(self.missing)})"]
        lines += [f"  {status} {path}" for status, path in self.missing]
        lines.append(f"== header year older than {self.year} ({len(self.stale)})")
        lines += [f"  {status} {path}: {why}" for status, path, why in self.stale]
        return "\n".join(lines)


def check_copyright(
    repo: str | Path,
    base: str,
    head: str = "HEAD",
    *,
    year: int,
    suffixes: tuple[str, ...] = SOURCE_SUFFIXES,
) -> CopyrightReport:
    """Check every source file ``head`` adds or modifies relative to ``base``."""
    report = CopyrightReport(year=year)
    out = git(repo, "diff", "--name-status", "--no-renames", base, head)
    for line in out.splitlines():
        if "\t" not in line:
            continue
        status, path = line.split("\t", 1)
        if status.startswith("D") or not path.endswith(suffixes):
            continue
        text = git(repo, "show", f"{head}:{path}", check=False)
        top = "\n".join(text.splitlines()[:HEADER_LINES])
        match = _HEADER.search(top)
        if not match:
            if status.startswith("A") or not re.search(r"Copyright", top, re.I):
                report.missing.append((status, path))
            else:
                report.stale.append((status, path, "non-NVIDIA or unparsed header"))
            continue
        years = [int(y) for y in match.groups() if y]
        if max(years) < year:
            report.stale.append((status, path, match.group(0)[:90]))
    return report


__all__ = ["CopyrightReport", "SOURCE_SUFFIXES", "check_copyright"]
