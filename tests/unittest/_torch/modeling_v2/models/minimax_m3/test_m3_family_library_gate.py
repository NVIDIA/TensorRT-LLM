# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MiniMax-M3 targets refuse, rather than fail later, on an installation without the family library.

The targets' fused all-reduces, low-M MXFP8 GEMM and long-context sparse decode run the family library's ops
(``torch.ops.mv2_minimax_m3``). Each target constructor opens that library before it builds anything, so an
installation without it raises ``NoTarget`` there and ``modeling_v2='auto'`` keeps the built-in model.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tensorrt_llm._torch._experimental.modeling_v2 import _native
from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import native, routing

pytestmark = pytest.mark.cpu_only

_FAMILY_DIR = Path(routing.__file__).resolve().parent
# The MiniMax-M3 targets and their classes; every one calls the family library.
_M3_TARGETS = {
    "minimax_m3_nvfp4__sm_103__tp4": "ModelingV2MiniMaxM3Nvfp4Sm103Tp4",
    "minimax_m3_nvfp4__sm_103__tp8": "ModelingV2MiniMaxM3Nvfp4Sm103Tp8",
    "minimax_m3_nvfp4__sm_107__tp2": "ModelingV2MiniMaxM3Nvfp4Sm107Tp2",
    "minimax_m3_nvfp4__sm_107__tp4": "ModelingV2MiniMaxM3Nvfp4Sm107Tp4",
    "minimax_m3_nvfp4__sm_107__tp8": "ModelingV2MiniMaxM3Nvfp4Sm107Tp8",
}


def test_family_ops_raises_no_target_without_the_library(monkeypatch, tmp_path):
    monkeypatch.setattr(_native, "_LIBS_DIR", tmp_path)
    monkeypatch.setattr(_native, "_built_targets", {})
    monkeypatch.setattr(_native, "_family_ops", {})
    with pytest.raises(NoTarget, match="has no libth_modeling_v2_minimax_m3.so"):
        native.family_ops()


def _constructor(directory: str, name: str) -> ast.FunctionDef:
    tree = ast.parse((_FAMILY_DIR / directory / "modeling.py").read_text())
    (cls,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name]
    (init,) = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"]
    return init


def _is_call(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == name
    )


@pytest.mark.parametrize("directory", sorted(_M3_TARGETS))
def test_the_constructor_opens_the_family_library_before_building(directory):
    """``family_ops()`` is a bare statement of the constructor, after the SM check and before anything else."""
    body = _constructor(directory, _M3_TARGETS[directory]).body
    calls = [i for i, node in enumerate(body) if _is_call(node, "family_ops")]
    assert calls, f"{directory}: the constructor does not call family_ops()"
    first = calls[0]
    assert all(isinstance(node, ast.Assert) for node in body[:first]), (
        f"{directory}: the constructor does work before it opens the family library"
    )


def test_the_routed_m3_targets_are_the_ones_checked():
    """Every MiniMax-M3 target ``route`` returns is in ``_M3_TARGETS``."""
    routed = {
        node.module.split(".")[0]
        for node in ast.walk(ast.parse(Path(routing.__file__).read_text()))
        if isinstance(node, ast.ImportFrom)
        and node.level == 1
        and node.module
        and node.module.startswith("minimax_m3_nvfp4__")
    }
    assert routed == set(_M3_TARGETS)
