# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every kernel that runs on `dep_lamport_round_trip`'s workspace re-arms the same slot.

The round trip, the fused reduce-scatter (`dep_lamport_rs_inject`) and the packed gather
(`dep_lamport_pack_gather`) share one workspace and its slot counts, and a decode step interleaves
them. A launch re-arms one slot for the vectors another launch recorded there, so every kernel must
re-arm the same one, `(slot + _CLEAR_AHEAD) mod 3`: a kernel on another offset leaves a slot
holding stale vectors, which the next launch on that slot reads as published. A source check; no
kernel runs."""

import ast
import importlib.util
import os

import pytest

pytestmark = pytest.mark.cpu_only

_MOE = os.path.join(
    os.path.dirname(importlib.util.find_spec("tensorrt_llm").origin),
    "_torch",
    "modeling_v2",
    "models",
    "qwen4_exp",
    "ops",
    "moe",
)
_KERNELS = ("dep_lamport_round_trip", "dep_lamport_rs_inject", "dep_lamport_pack_gather")


def _clear_slot_offsets(module: str) -> list:
    """The added term of every `clear_slot = (... + <term>) % ...` assignment, as source text."""
    tree = ast.parse(open(os.path.join(_MOE, module + ".py")).read())
    out = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "clear_slot"
        ):
            mod = node.value
            assert isinstance(mod, ast.BinOp) and isinstance(mod.op, ast.Mod), ast.unparse(node)
            add = mod.left
            assert isinstance(add, ast.BinOp) and isinstance(add.op, ast.Add), ast.unparse(node)
            out.append(ast.unparse(add.right))
    return out


@pytest.mark.parametrize("module", _KERNELS)
def test_every_workspace_kernel_re_arms_the_round_trips_slot(module):
    offsets = _clear_slot_offsets(module)
    assert offsets, f"{module}: no re-arm slot found"
    for term in offsets:
        assert "_CLEAR_AHEAD" in term, (
            f"{module}: re-arms (slot + {term}), not `_CLEAR_AHEAD`'s slot"
        )
