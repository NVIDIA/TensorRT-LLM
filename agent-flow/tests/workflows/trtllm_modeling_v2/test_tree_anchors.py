# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Anchors: what the workflow reads off the engine's tree, found where it moved or reported missing."""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import pytest

from agent_flow.workflows.trtllm_modeling_v2.operators import tree_anchors

ANCHORS = tree_anchors.ANCHORS


def test_every_anchor_is_found_by_what_identifies_it_in_this_checkout():
    """Inside a TensorRT-LLM checkout, a search that ignores each hint finds what the hint points at.

    The search is what a run relies on once the tree moves, so it is what this
    holds. A hint this tree lacks (an older or newer layout) only warns.
    """
    root = Path(__file__).resolve().parents[4]
    if not (root / "tensorrt_llm").is_dir():
        pytest.skip("not inside a TensorRT-LLM checkout")
    unfindable = []
    for key, anchor in ANCHORS.items():
        here = tree_anchors.resolve(root, anchor)
        if here.via != "hint":
            where = f"found at {list(here.paths[:3])}" if here.found else "missing"
            warnings.warn(f"anchor {key}: no {anchor.hint} in this tree, {where}", stacklevel=1)
            continue
        moved = dataclasses.replace(anchor, hint=f"{anchor.hint}.moved")
        found = tree_anchors.resolve(root, moved).paths
        hint = anchor.hint
        if not any(
            p == hint or p.startswith(hint + "/") or hint.startswith(p + "/") for p in found
        ):
            unfindable.append(f"{key}: a search found {list(found[:3])}")
    assert not unfindable, f"anchors a search cannot find by what identifies them: {unfindable}"
