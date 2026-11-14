# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The audit that keeps pinned versions and dates out of the modeling_v2 tree.

Out of tree, modeling_v2 was a separate repo against a pinned `tensorrt_llm`, so
writing that version into a contract was meaningful: the pin was the thing a
receipt was valid against, and a bump really did void what had been measured.

In tree there is no external pin to drift against. The catalog and the ops it
wraps move with the trunk together, and the catalog tests run in pre-merge, so
the trunk proves itself on every commit. A version written into a contract is
then a number that is wrong the next day with nothing to notice -- worse than
absent, because every release bump makes a receipt CI is actively keeping
green look stale and reopens a question already answered.

Removing them once was the easy half; this is what stops the next one. A
scan rather than a review checklist, for the reason the tree prefers
everywhere: a rule nothing enforces is a rule that decays.
"""

from __future__ import annotations

import re
from pathlib import Path

import tensorrt_llm._torch.modeling_v2 as _modeling_v2

# The package, not this file: the tree under audit lives under tensorrt_llm/
# while this test lives under tests/.
_ROOT = Path(_modeling_v2.__file__).resolve().parent

# Each pattern, and what to say instead of it.
_STALE_CLAIM_PATTERNS = (
    (
        re.compile(r"\b\d+\.\d+\.\d+rc\d+\b"),
        "a pinned tensorrt_llm release. The catalog moves with the trunk, so "
        "state the behaviour, not the build it was seen on",
    ),
    (
        re.compile(r"(?i)\bflashinfer[-\w]*\s+v?\d+\.\d+\.\d+"),
        "a pinned flashinfer release. Name the kernel or the code path, not "
        "the version that happened to be installed",
    ),
    (
        re.compile(r"(?i)\btensorrt[-_]llm\s*[=<>!]=\s*\d"),
        "a pinned tensorrt_llm requirement. A target moves with the trunk; "
        "the SM assert is the part of its identity that does not move",
    ),
    (
        re.compile(r"(?i)\bmeasured\s+(?:on\s+)?20\d\d-\d\d-\d\d"),
        "a measurement date, which says nothing about whether the claim still "
        "holds. Say that it was measured rather than when",
    ),
)

#: Scanned for the above. The catalog contracts are the point -- they are where
#: a version pin last accumulated -- but a target's modeling code carries the
#: same kind of prose, so the whole package is in.
_PROSE_SUFFIXES = {".py", ".md", ".yaml", ".yml"}


def _package_files():
    for path in sorted(_ROOT.rglob("*")):
        if path.suffix in _PROSE_SUFFIXES and "__pycache__" not in path.parts:
            yield path


def test_no_file_pins_a_version_or_a_date():
    """Report every stale claim at once, with the line and what to write instead."""
    offences = []
    for path in _package_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), 1):
            for pattern, why in _STALE_CLAIM_PATTERNS:
                if (hit := pattern.search(line)) is not None:
                    rel = path.relative_to(_ROOT)
                    offences.append(f"  {rel}:{lineno}: {hit.group(0)!r} is {why}")
    assert not offences, "modeling_v2 files must not pin a version or a date:\n" + "\n".join(
        offences
    )
