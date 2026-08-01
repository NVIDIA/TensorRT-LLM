#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Enforce the Claude Code skill (and agent) naming convention.

The skill naming convention is defined in ``.claude/README.md``. Agents follow
the same convention, so this check covers both.

For every skill (``.claude/skills/*/SKILL.md``) and agent
(``.claude/agents/*.md``):

1. The on-disk name (file stem or directory name) must start with one of the
   prefixes in ``ALLOWED_PREFIXES`` below, followed by ``-`` and at least one
   descriptive character.
2. The frontmatter ``name:`` field must match the on-disk name.

``ALLOWED_PREFIXES`` is the authoritative list. When adding or renaming a
prefix, update both this list and the prefix table in
``.claude/README.md``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

ALLOWED_PREFIXES = (
    "ad-",
    "exec-",
    "kernel-",
    "perf-",
    "trtllm-",
)

REPO_ROOT = Path(__file__).resolve().parent.parent
NAMING_DOC = REPO_ROOT / ".claude" / "README.md"
AGENTS_DIR = REPO_ROOT / ".claude" / "agents"
SKILLS_DIR = REPO_ROOT / ".claude" / "skills"

FRONTMATTER_RE = re.compile(r"\A(?:<!--.*?-->\s*)*---\n(.*?)\n---\n?", re.DOTALL)


def load_frontmatter_name(path: Path) -> str | None:
    m = FRONTMATTER_RE.match(path.read_text())
    if not m:
        return None
    data = yaml.safe_load(m.group(1)) or {}
    name = data.get("name")
    return name if isinstance(name, str) else None


def check_prefix(name: str) -> str | None:
    for prefix in ALLOWED_PREFIXES:
        if name.startswith(prefix) and len(name) > len(prefix):
            return None
    options = ", ".join(ALLOWED_PREFIXES)
    return f"does not start with an allowed prefix ({options})"


def collect_items() -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []
    for path in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        items.append((path, path.parent.name))
    for path in sorted(AGENTS_DIR.glob("*.md")):
        items.append((path, path.stem))
    return items


def main() -> int:
    violations: list[str] = []

    for path, expected_name in collect_items():
        rel = path.relative_to(REPO_ROOT)

        msg = check_prefix(expected_name)
        if msg:
            violations.append(f"{rel}: {expected_name!r} {msg}")

        fm_name = load_frontmatter_name(path)
        if fm_name is None:
            violations.append(f"{rel}: missing `name` field in frontmatter")
        elif fm_name != expected_name:
            violations.append(
                f"{rel}: frontmatter name {fm_name!r} != on-disk name {expected_name!r}"
            )

    if violations:
        doc_rel = NAMING_DOC.relative_to(REPO_ROOT)
        print(f"Skill naming convention violations (see {doc_rel}):", file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
