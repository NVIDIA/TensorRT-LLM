# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3 `tp16_moetp16ep1` target is a copy of `tp16_moetp4ep4`'s files, changed only in marked blocks.

Targets share no files (test_modeling_v2_claims.py), so `tp16_moetp16ep1` (route B) ships its own copy of every
module of `tp16_moetp4ep4` (route A). Each place the copy differs is a block:

    # >>> route B: <why it differs>
    <route B's lines, none where it drops route A's>
    # <<< route B

This reads both targets' files and imports neither. It fails on any difference outside the blocks, so a change to
route A's modules reaches route B's copy, or becomes one of its blocks, in the same commit.
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import List, Tuple

import pytest

import tensorrt_llm._torch._experimental.modeling_v2 as _modeling_v2

# The package, not this file: the targets live under tensorrt_llm/ while this test lives under tests/.
_FAMILY = Path(_modeling_v2.__file__).resolve().parent / "models" / "kimi_k3_vl"
_ROUTE_A = _FAMILY / "kimi_k3_mxfp4__sm_100__tp16_moetp4ep4"
_ROUTE_B = _FAMILY / "kimi_k3_mxfp4__sm_100__tp16_moetp16ep1"

_BEGIN = "# >>> route B"
_END = "# <<< route B"

# Each target's package marker holds only its own one-line description.
_NOT_COPIED = {"__init__.py"}


def _modules(target: Path) -> List[str]:
    return sorted(p.name for p in target.glob("*.py") if p.name not in _NOT_COPIED)


def _blocks(lines: List[str]) -> List[Tuple[int, int]]:
    """The route B blocks as (begin, end) indices of their marker lines; asserts the markers are well formed."""
    blocks, begin = [], None
    for i, line in enumerate(lines):
        text = line.strip()
        if text.startswith(_BEGIN):
            assert begin is None, (
                f"line {i + 1}: a block opens inside the one opened on line {begin + 1}"
            )
            assert text[len(_BEGIN) :].startswith(":") and text[len(_BEGIN) + 1 :].strip(), (
                f"line {i + 1}: a block opens with '{_BEGIN}: <why it differs>'"
            )
            begin = i
        elif text.startswith(_END):
            assert text == _END, f"line {i + 1}: a block closes with '{_END}' alone"
            assert begin is not None, f"line {i + 1}: a block closes without one open"
            blocks.append((begin, i))
            begin = None
    assert begin is None, f"line {begin + 1}: a block is never closed"
    return blocks


def _spans(lines: List[str], blocks: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Each block with the blank lines around it, which the formatters add and drop next to a comment."""
    spans = []
    for begin, end in blocks:
        while begin > 0 and not lines[begin - 1].strip():
            begin -= 1
        while end + 1 < len(lines) and not lines[end + 1].strip():
            end += 1
        spans.append((begin, end))
    return spans


def _inside(spans: List[Tuple[int, int]], j1: int, j2: int) -> bool:
    """Whether route B's lines [j1, j2), or the point between lines j1 - 1 and j1 when j1 == j2, are in one block."""
    if j1 == j2:
        return any(begin < j1 <= end for begin, end in spans)
    return any(begin <= j1 and j2 <= end + 1 for begin, end in spans)


def _drift(a: List[str], b: List[str]) -> List[str]:
    """Each difference between route A's lines ``a`` and route B's ``b`` outside route B's blocks, as text."""
    assert not _blocks(a) and not any(_BEGIN in line or _END in line for line in a), (
        "route A's files have no route B markers"
    )
    blocks = _spans(b, _blocks(b))
    drift = []
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal" or _inside(blocks, j1, j2):
            continue
        shown = [f"  route A {i + 1}: {a[i]}" for i in range(i1, min(i2, i1 + 5))]
        shown += [f"  route B {j + 1}: {b[j]}" for j in range(j1, min(j2, j1 + 5))]
        drift.append(
            f"route A lines {i1 + 1}-{i2}, route B lines {j1 + 1}-{j2}:\n" + "\n".join(shown)
        )
    return drift


def test_route_b_copies_every_module():
    assert _modules(_ROUTE_A), f"no modules in {_ROUTE_A}"
    assert _modules(_ROUTE_B) == _modules(_ROUTE_A)


@pytest.mark.parametrize("name", _modules(_ROUTE_A))
def test_route_b_matches_route_a_outside_its_blocks(name):
    a = (_ROUTE_A / name).read_text(encoding="utf-8").splitlines()
    b = (_ROUTE_B / name).read_text(encoding="utf-8").splitlines()
    drift = _drift(a, b)
    assert not drift, (
        f"{name}: route B's copy differs from route A's outside its blocks. Carry route A's "
        "change into the copy, or mark route B's lines as a block that says why they differ:\n"
        + "\n".join(drift)
    )


def test_the_check_sees_drift_only_outside_the_blocks():
    a = ["x = 1", "y = 2", "z = 3"]
    changed = ["x = 1", "# >>> route B: why", "y = 4", "# <<< route B", "z = 3"]
    dropped = ["x = 1", "", "# >>> route B: why", "# <<< route B", "", "z = 3"]
    unmarked = ["x = 1", "# >>> route B: why", "y = 2", "# <<< route B", "z = 4"]
    added = ["x = 1", "y = 2", "w = 0", "z = 3"]
    spaced = ["x = 1", "", "y = 2", "z = 3"]
    assert not _drift(a, changed)
    assert not _drift(a, dropped)
    assert _drift(a, unmarked)
    assert _drift(a, added)
    assert _drift(a, spaced)
    assert _drift(a[:2], a)
    with pytest.raises(AssertionError, match="never closed"):
        _drift(a, ["x = 1", "# >>> route B: why", "y = 2", "z = 3"])
    with pytest.raises(AssertionError, match="why it differs"):
        _drift(a, ["x = 1", "# >>> route B", "y = 2", "# <<< route B", "z = 3"])
