# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every dense attention call of the qwen4_exp target names the KV storage it runs over.

The FMHA libraries read the pool through the layer's quant config and the call's two KV
scales, and the three have to agree with the pool's dtype. A call that leaves one at its
default reads an fp8 pool as bf16 (or the reverse) and still returns finite values, so no run
notices. Every KV cache call goes through the core's `attention`, which takes all three from
the core. The check reads `modeling.py` and `vision.py` as text: it needs no device and no
import of the target.

The vision tower is the one exception: its attention has no KV cache, so its layer has no
quant config and its calls pass no KV scales.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import tensorrt_llm._torch.modeling_v2 as _modeling_v2

pytestmark = pytest.mark.cpu_only

_TARGET = (
    Path(_modeling_v2.__file__).resolve().parent
    / "models/qwen4_exp/qwen3_8_flash_next_nvfp4__sm_103__default"
)
_KV_SCALES = ("kv_scale_orig_quant", "kv_scale_quant_orig")
_TREE = ast.parse((_TARGET / "modeling.py").read_text())
_VISION_TREE = ast.parse((_TARGET / "vision.py").read_text())


def _calls(tree: ast.AST, name: str) -> list[ast.Call]:
    """Calls of `name` or of an attribute `.name` in `tree`."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == name)
        )
    ]


def _method(tree: ast.Module, name: str) -> ast.FunctionDef:
    (method,) = [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    return method


def _keywords(call: ast.Call) -> dict[str, ast.expr]:
    return {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}


def test_every_kv_cache_call_goes_through_the_core() -> None:
    assert not _calls(_TREE, "thop_attention")
    # Against going vacuous: the trunk and the draft layer each call it from the prefill
    # and the decode target.
    calls = [
        call
        for call in _calls(_TREE, "attention")
        if isinstance(call.func, ast.Attribute) and ast.unparse(call.func.value) == "core"
    ]
    assert len(calls) >= 4


def test_the_core_attention_passes_the_pool_storage() -> None:
    """The layer's quant config follows the core's KV cache algorithm and the call's KV scales
    are the core's, whose `FP8` / `NVFP4` / `None` resolution both follow."""
    (layer,) = _calls(_method(_TREE, "_attention_layer"), "AttentionLayer")
    (quant_config,) = _calls(layer, "QuantConfig")
    assert ast.unparse(_keywords(quant_config)["kv_cache_quant_algo"]) == (
        "self.kv_cache_quant_algo"
    )
    (fwd,) = _calls(_method(_TREE, "attention"), "AttentionForwardArgs")
    keywords = _keywords(fwd)
    for name in _KV_SCALES:
        assert ast.unparse(keywords[name]) == f"self.{name}", name


def test_the_vision_tower_runs_without_kv_storage() -> None:
    """The cache-less attention states a bf16, unscaled configuration."""
    assert not _calls(_VISION_TREE, "thop_attention")
    (layer,) = _calls(_VISION_TREE, "AttentionLayer")
    assert "quant_config" not in _keywords(layer)
    (fwd,) = _calls(_VISION_TREE, "AttentionForwardArgs")
    assert not set(_keywords(fwd)) & set(_KV_SCALES)
