# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every `thop_attention` call of the qwen4_exp target names the KV storage it runs over.

The attention op reads the pool through `quant_mode` and the two KV scales, and the three
have to agree with the pool's dtype. A call that leaves one at its default reads an fp8 pool
as bf16 (or the reverse) and still returns finite values, so no run notices. The check reads
`modeling.py` as text: it needs no device and no import of the target.

The vision tower's call is the one exception: its attention is the encoder-only mode of the
op, which has no KV cache, so it passes `kv_scale_*` None and `quant_mode` 0 through
`_VISION_CALL_CONSTANTS`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import tensorrt_llm._torch._experimental.modeling_v2 as _modeling_v2

pytestmark = pytest.mark.cpu_only

_MODELING = (
    Path(_modeling_v2.__file__).resolve().parent
    / "models/qwen4_exp/qwen3_8_flash_next_nvfp4__sm_103__tp1/modeling.py"
)
_KV_STORAGE_ARGS = ("quant_mode", "kv_scale_orig_quant", "kv_scale_quant_orig")
_TREE = ast.parse(_MODELING.read_text())


def _thop_attention_calls() -> list[ast.Call]:
    return [
        node
        for node in ast.walk(_TREE)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "thop_attention"
    ]


def _module_dict(name: str) -> ast.Call:
    """The `dict(...)` a module-level `name = dict(...)` assigns."""
    for node in _TREE.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            assert isinstance(node.value, ast.Call), name
            return node.value
    raise AssertionError(f"modeling.py has no module-level {name}")


def _keywords(call: ast.Call) -> dict[str, ast.expr]:
    return {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}


def _unpacked(call: ast.Call) -> set[str]:
    """Names of the `**name` arguments of `call`."""
    return {
        kw.value.id for kw in call.keywords if kw.arg is None and isinstance(kw.value, ast.Name)
    }


def _is_vision(call: ast.Call) -> bool:
    return "_VISION_CALL_CONSTANTS" in _unpacked(call)


_KV_CACHE_CALLS = [call for call in _thop_attention_calls() if not _is_vision(call)]


def test_every_kv_cache_call_passes_the_pool_storage() -> None:
    # Against going vacuous: the trunk and the draft layer each call it from the prefill
    # and the decode target.
    assert len(_KV_CACHE_CALLS) >= 4
    for call in _KV_CACHE_CALLS:
        keywords = _keywords(call)
        for name in _KV_STORAGE_ARGS:
            assert name in keywords, f"thop_attention call at line {call.lineno} omits {name}"
            value = keywords[name]
            # Read off the core, whose `FP8` / `None` resolution all three follow.
            assert isinstance(value, ast.Attribute) and value.attr == name, (
                f"line {call.lineno}: {name}={ast.unparse(value)}, not the core's `{name}`"
            )


def test_the_unpacked_arguments_carry_no_pool_storage() -> None:
    """`**_CALL_CONSTANTS` and `**step` (each target's `step_args`) are unpacked into every
    KV cache call; a storage argument in either would raise as a duplicate keyword or, if the
    call dropped its own, silently pin the pool to one storage."""
    constants = {kw.arg for kw in _module_dict("_CALL_CONSTANTS").keywords}
    assert constants
    assert not constants & set(_KV_STORAGE_ARGS)
    for call in _KV_CACHE_CALLS:
        assert "_CALL_CONSTANTS" in _unpacked(call)
    steps = [
        node
        for node in ast.walk(_TREE)
        if isinstance(node, ast.FunctionDef) and node.name == "step_args"
    ]
    assert len(steps) == 2, "one step_args per target (prefill, decode)"
    for step in steps:
        returned = [n for n in ast.walk(step) if isinstance(n, ast.Return)]
        assert returned
        for node in returned:
            assert isinstance(node.value, ast.Call), "step_args returns dict(...)"
            assert not {kw.arg for kw in node.value.keywords} & set(_KV_STORAGE_ARGS)


def test_the_vision_tower_runs_without_kv_storage() -> None:
    """The encoder-only call, made once, states a bf16, unscaled, cache-less configuration."""
    constants = {kw.arg: kw.value for kw in _module_dict("_VISION_CALL_CONSTANTS").keywords}
    assert ast.literal_eval(constants["quant_mode"]) == 0
    assert ast.literal_eval(constants["kv_scale_orig_quant"]) is None
    assert ast.literal_eval(constants["kv_scale_quant_orig"]) is None
    vision = [call for call in _thop_attention_calls() if _is_vision(call)]
    assert len(vision) == 1
    for call in vision:
        assert not set(_keywords(call)) & set(_KV_STORAGE_ARGS)
