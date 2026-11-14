# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sm_107 TP4 target keeps its measured draft route (S4 through _refine_block_logits) on a tree whose in-tree
DSpark worker has the ancestral draft: the TP4 worker opts out of DSparkWorker._samples_block_ancestrally unless
TRTLLM_M31_ENABLE_TP4_ANCESTRAL is set or the TP4 Gumbel block sampler is installed (opt-in, TP4_GUMBEL_DRAFT). The
sampler / drafter items (SOFTMAX_MC, SM107_DRAFT_FMHA_SM100F) are default on with their kill switches. Source-level
(stdlib ast): no CUDA."""

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[7]
_FAMILY = _ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3"
_TP4 = _FAMILY / "minimax_m3_1_flash_nvfp4__sm_107__tp4"

pytestmark = pytest.mark.cpu_only


def _calls(node: ast.AST, name: str) -> list:
    return [
        c
        for c in ast.walk(node)
        if isinstance(c, ast.Call) and getattr(c.func, "id", None) == name and c.args
    ]


def _assign(tree: ast.Module, name: str) -> ast.Assign:
    (assign,) = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == name for t in n.targets)
    ]
    return assign


def _method(tree: ast.Module, cls_name: str, name: str) -> ast.FunctionDef:
    (cls,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name]
    (fn,) = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name]
    return fn


def test_tp4_ancestral_is_opt_in() -> None:
    tree = ast.parse((_TP4 / "dspark.py").read_text())
    assign = _assign(tree, "TP4_ANCESTRAL")
    assert [c.args[0].value for c in _calls(assign.value, "m31_enabled")] == ["TP4_ANCESTRAL"]
    assert [c.args[0].value for c in _calls(assign.value, "m31_disabled")] == ["TP4_ANCESTRAL"]


def test_tp4_gumbel_is_opt_in_until_measured() -> None:
    tree = ast.parse((_TP4 / "dspark.py").read_text())
    default = _assign(tree, "TP4_GUMBEL_DEFAULT_ON")
    assert isinstance(default.value, ast.Constant) and default.value.value is False
    assign = _assign(tree, "TP4_GUMBEL_DRAFT")
    assert [c.args[0].value for c in _calls(assign.value, "m31_enabled")] == ["TP4_GUMBEL_DRAFT"]
    assert [c.args[0].value for c in _calls(assign.value, "m31_disabled")] == ["TP4_GUMBEL_DRAFT"]


def test_the_tp4_worker_gates_the_ancestral_route() -> None:
    tree = ast.parse((_TP4 / "dspark.py").read_text())
    fn = _method(tree, "M3DSparkWorker", "_samples_block_ancestrally")
    (ret,) = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    assert isinstance(ret.value, ast.BoolOp) and isinstance(ret.value.op, ast.And)
    assert getattr(ret.value.values[0], "id", None) == "ancestral"
    route = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "tp4_ancestral_route"
    ]
    assert len(route) == 1
    (r,) = [n for n in ast.walk(route[0]) if isinstance(n, ast.Return)]
    names = {getattr(n, "id", None) for n in ast.walk(r.value)}
    assert {"TP4_ANCESTRAL", "_TP4_GUMBEL_INSTALLED"} <= names


def test_the_family_gumbel_module_installs_the_block_override() -> None:
    tree = ast.parse((_FAMILY / "ops" / "gumbel_draft.py").read_text())
    (fn,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "install"]
    # A runtime without the ancestral block registry keeps the stock loop (looked up with getattr).
    assert "set_ancestral_block_override" in ast.unparse(fn)


def test_tp4_sampler_and_drafter_ports_default_on_with_kill_switches() -> None:
    tree = ast.parse((_TP4 / "modeling.py").read_text())
    for name, switch in (
        ("SOFTMAX_MC", "SOFTMAX_MC"),
        ("VR_DRAFT_FMHA_SM100F", "SM107_DRAFT_FMHA_SM100F"),
    ):
        assign = _assign(tree, name)
        assert isinstance(assign.value, ast.UnaryOp) and isinstance(assign.value.op, ast.Not)
        assert [c.args[0].value for c in _calls(assign.value, "m31_disabled")] == [switch]
    source = (_TP4 / "modeling.py").read_text()
    assert "_SET_SOFTMAX_OVERRIDE(_softmax_mc.softmax_mc)" in source
    assert "_drafter_sliding_fi.install_sliding_fi(draft)" in source
    assert "_dspark.install_tp4_gumbel()" in source
