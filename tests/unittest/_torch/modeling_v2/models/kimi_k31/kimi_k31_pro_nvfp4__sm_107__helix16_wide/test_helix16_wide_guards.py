# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Construction guards of ``kimi_k31_pro_nvfp4__sm_107__helix16_wide`` (CPU only, on the source).

The class refuses a GPU other than Rubin first (``require_rubin``); the chain gathers every rank's
target, routed-expert layout, MoE front and row counts before its load and first symmetric-memory
rendezvous (``check_chain_agreement``) and raises, each rank's values named, where they differ.
The two pushed all-reduces finish on ``rubin_ull_k3_lamport_ar_pdlfix``'s finish kernel; the
target registers the SiTU op its dense fallback calls.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from types import SimpleNamespace

import pytest
import torch
from _helix16_wide_cpu import function_node, modeling_path, source_objects

pytestmark = pytest.mark.cpu_only

CLASS = "ModelingV2KimiK31ProNvfp4Sm107Helix16Wide"
NAME = "kimi_k31_pro_nvfp4__sm_107__helix16_wide"


@pytest.fixture(scope="module")
def guards():
    names = ("_TARGET_NAME", "_EXPERTS_LAYOUT", "_FRONT_LAYOUT", "RUBIN_CAPABILITY")
    names += ("require_rubin", "check_chain_agreement")
    return source_objects(modeling_path(), names, {"Sequence": Sequence})


def _calls(node: ast.AST) -> set:
    return {
        getattr(call.func, "id", None) or getattr(call.func, "attr", None)
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
    }


@pytest.mark.parametrize("capability, ok", [((10, 7), True), ((10, 3), False), ((10, 0), False)])
def test_rubin_only_and_first(guards, capability, ok):
    guards["torch"] = SimpleNamespace(
        cuda=SimpleNamespace(get_device_capability=lambda *device: capability)
    )
    # right after ``nn.Module.__init__``: before the limits, the config checks and the load
    body = function_node(modeling_path(), f"{CLASS}.__init__").body
    assert "__init__" in _calls(body[0]) and _calls(body[1]) == {"require_rubin"}
    if ok:
        guards["require_rubin"]()
        return
    with pytest.raises(ValueError) as error:
        guards["require_rubin"]()
    text = str(error.value)
    assert f"sm_{capability[0]}{capability[1]}" in text and "TLLM_K31_HELIX16_WIDE" in text
    assert NAME in text


@pytest.fixture
def gather(monkeypatch):
    """``torch.distributed``'s world size and object all-gather stood in: ``values[rank]`` (None:
    this rank's own payload) is what each rank contributes."""
    if not torch.distributed.is_available():
        pytest.skip("this torch build has no torch.distributed")
    state = SimpleNamespace(values=[None] * 16, groups=[])

    def all_gather_object(out, obj, group=None):
        state.groups.append(group)
        out[:] = [obj if v is None else v for v in state.values]

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: len(state.values))
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    return state


ROWS = tuple(sorted({*range(1, 17), *range(8, 129, 8)}))


def test_ranks_that_differ_are_named(guards, gather):
    group = object()
    guards["check_chain_agreement"](group, rows=list(ROWS))  # all agree: passes
    assert gather.groups == [group]
    mine = (guards["_TARGET_NAME"], guards["_EXPERTS_LAYOUT"], guards["_FRONT_LAYOUT"], ROWS)
    other_rows = (*mine[:3], tuple(r for r in ROWS if r <= 64))
    other_layout = (mine[0], "another_layout", mine[2], ROWS)
    calls = (mine[0], mine[1], ROWS)  # the same name and experts, no front
    gather.values = [None] * 7 + [other_rows] * 7 + [other_layout, calls]
    with pytest.raises(RuntimeError) as error:
        guards["check_chain_agreement"](None, rows=ROWS)
    text = str(error.value)
    assert f"ranks {list(range(7))}: {mine!r}" in text
    assert f"ranks {list(range(7, 14))}: {other_rows!r}" in text
    assert f"ranks [14]: {other_layout!r}" in text and f"ranks [15]: {calls!r}" in text
    # Whatever a rank of another target contributed instead is named as it came.
    gather.values = [None] * 15 + [{"handle": 3}]
    with pytest.raises(RuntimeError, match=r"ranks \[15\]: \{'handle': 3\}"):
        guards["check_chain_agreement"](None, rows=ROWS)


def test_the_chain_checks_before_its_load_and_first_rendezvous():
    """In ``FlatChainRuntime.__init__`` the gather precedes every load and every box."""
    body = function_node(modeling_path(), "FlatChainRuntime.__init__").body
    check = next(i for i, s in enumerate(body) if "check_chain_agreement" in _calls(s))
    first = {"kda", "moe", "module", "NvScratch", "place_experts"}
    assert check < next(i for i, s in enumerate(body) if _calls(s) & first)
    call = next(c for c in ast.walk(body[check]) if isinstance(c, ast.Call))
    assert ast.unparse(call.args[0]) == "group"


def test_the_lamport_finishes_are_p14s():
    """L0's dense all-reduce and the drafter's (``draft_reduce``) use no shared Lamport finish."""
    modules = set()
    for node in ast.walk(ast.parse(modeling_path().read_text())):
        if isinstance(node, ast.ImportFrom):
            base = node.module or ""
            modules.update({base, *(f"{base}.{alias.name}" for alias in node.names)})
    shared = ("rubin_ull_k3_lamport_ar", "rubin_lamport_ar")
    shared += ("rubin_lamport_push.rubin_lamport_finish",)
    assert not [m for m in modules if m.endswith(shared)]
    assert "kernels.comm.rubin_ull_k3_lamport_ar_pdlfix.run_lamport_finish" in modules
    draft = ast.unparse(function_node(modeling_path(), "draft_reduce"))
    assert "lamport_finish_p14(push.view, out, one_shot, push.pdl, False, None, False)" in draft


def test_the_target_registers_the_situ_op_its_dense_fallback_calls():
    """``dense_partial`` calls the family's ``situ_and_mul`` entry (``torch.ops.trtllm``), whose
    op ``tensorrt_llm._torch.modules.situ`` registers: the target imports that module."""
    tree = ast.parse(modeling_path().read_text())
    assert "tensorrt_llm._torch.modules.situ" in {
        a.name for n in tree.body if isinstance(n, ast.Import) for a in n.names
    }
    assert "situ_and_mul(pair, beta, linear_beta)" in ast.unparse(
        function_node(modeling_path(), "dense_partial")
    )
    situ = modeling_path().parents[5] / "modules/situ.py"
    assert '@torch.library.custom_op("trtllm::situ_and_mul"' in situ.read_text()
