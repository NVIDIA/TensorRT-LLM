# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Layer-local DSA quantization and RoPE contracts of the specialized SM107 targets."""

import ast
from pathlib import Path

import pytest

from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo, QuantMode

pytestmark = pytest.mark.cpu_only

REPO = Path(__file__).resolve().parents[6]
MODELING_V2 = REPO / "tensorrt_llm/_torch/_experimental/modeling_v2"
FAMILY = MODELING_V2 / "models/kimi_k31"
ROPE_ENTRY = MODELING_V2 / "catalog/attention/mla_rope_generation.py"
LAYOUTS = ("dep16", "helix8", "tep8")
# The layer constants of the attention op's calls (dep16's padded-head calls have their own copy)
# and of the RoPE kernel's call (dep16's padded-head generation call has its own copy).
_ATTN_TABLES = ("_MLA_ATTN", "_MLA_ATTN_PAD128")
_ROPE_TABLE = "_MLA_ROPE_GEN"
_ROPE_TABLES = (_ROPE_TABLE, "_MLA_ROPE_GEN_PAD128")
_WEIGHT_QUANT = (
    "has_fp8_qdq",
    "has_fp8_rowwise",
    "has_fp8_block_scales",
    "has_nvfp4",
    "has_w4a8_nvfp4_fp8",
    "has_mxfp8",
    "has_mxfp4",
)


def _fp8_kv_mode() -> int:
    """The quant mode ``update_quant_config`` gives a layer of the FP8 KV recipe."""
    return int(QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8).layer_quant_mode)


def _tree(layout: str) -> ast.Module:
    path = FAMILY / f"kimi_k31_pro_nvfp4__sm_107__{layout}" / "modeling.py"
    return ast.parse(path.read_text(), str(path))


def _assigned(tree: ast.Module) -> dict:
    """The module's top-level ``NAME = value`` assignments."""
    return {
        node.targets[0].id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }


def _table(tree: ast.Module, name: str) -> dict:
    """A ``dict(...)`` constant of the module: literal values, other expressions as source text."""
    values = _assigned(tree)
    node = values[name]
    assert isinstance(node, ast.Call) and getattr(node.func, "id", None) == "dict", name
    table = {}
    for arg in node.args:
        assert isinstance(arg, ast.Name), ast.unparse(arg)
        table.update(_table(tree, arg.id))
    for kw in node.keywords:
        assert kw.arg is not None, f"{name}: ** in a layer constant"
        try:
            table[kw.arg] = ast.literal_eval(kw.value)
        except ValueError:
            table[kw.arg] = ast.unparse(kw.value)
    return table


def _calls(tree: ast.Module, name: str) -> list:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    ]


def _last_spread(call: ast.Call) -> str:
    """The name spread last into a call: ``**NAME`` or ``**{..., **NAME}``."""
    spreads = [kw.value for kw in call.keywords if kw.arg is None]
    assert spreads, ast.unparse(call)
    last = spreads[-1]
    if isinstance(last, ast.Dict):
        assert last.keys[-1] is None, ast.unparse(last)
        last = last.values[-1]
    assert isinstance(last, ast.Name), ast.unparse(last)
    return last.id


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_layer_constants_are_the_fp8_kv_recipe(layout):
    """The constants hold the quant mode ``update_quant_config`` gives a layer of the FP8 KV
    recipe."""
    tree = _tree(layout)
    assert ast.literal_eval(_assigned(tree)["_INDEXER_K_DTYPE"]) == "fp8"
    _check_fp8_tables(tree, layout)


def _check_fp8_tables(tree: ast.Module, layout: str) -> None:
    values = _assigned(tree)
    tables = [name for name in (*_ATTN_TABLES, *_ROPE_TABLES) if name in values]
    assert "_MLA_ATTN" in tables and _ROPE_TABLE in tables, tables
    for name in tables:
        table = _table(tree, name)
        assert table["quant_mode"] == _fp8_kv_mode(), (layout, name, table["quant_mode"])
        mode = QuantMode(table["quant_mode"])
        assert mode.has_fp8_kv_cache() and not mode.has_fp4_kv_cache(), (layout, name)
        assert not mode.has_int8_kv_cache(), (layout, name)
        assert not any(getattr(mode, flag)() for flag in _WEIGHT_QUANT), (layout, name)
        # Unset KV scales read as 1.0: the unit scales of the FP8 cache.
        for scale in ("kv_scale_orig_quant", "kv_scale_quant_orig"):
            assert table[scale] is None, (layout, name, scale)
    assert _table(tree, _ROPE_TABLE)["kv_cache_scale_orig_quant"] is None, layout
    assert _table(tree, "_MLA_ATTN")["is_mla_enable"] is True, layout


@pytest.mark.parametrize("layout", LAYOUTS)
def test_every_mla_call_passes_the_layer_constants_last(layout):
    """Every attention and RoPE call of a DSA layer passes the constants last, so no metadata
    argument can override the quant mode; nothing else names one or changes the constants."""
    _check_mla_calls(_tree(layout), layout)


def _check_mla_calls(tree: ast.Module, layout: str) -> None:
    attn, rope = _calls(tree, "thop_attention"), _calls(tree, "mla_rope_generation")
    assert attn and rope, layout
    for call in attn:
        assert _last_spread(call) in _ATTN_TABLES, ast.unparse(call)
    for call in rope:
        assert _last_spread(call) in _ROPE_TABLES, ast.unparse(call)
        # Over an FP8 cache the RoPE kernel writes the quantized query into this buffer.
        (buffer,) = [kw.value for kw in call.keywords if kw.arg == "quant_q_buffer"]
        assert not (isinstance(buffer, ast.Constant) and buffer.value is None), ast.unparse(call)
    tables = {*_ATTN_TABLES, *_ROPE_TABLES}
    owner = {}
    for top in tree.body:
        target = top.targets[0] if isinstance(top, ast.Assign) else top
        name = getattr(target, "id", None) or getattr(target, "name", None)
        for node in ast.walk(top):
            owner[node] = name
    named = [
        node.lineno
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.keyword) and node.arg == "quant_mode" and owner[node] not in tables
        )
        or (isinstance(node, ast.Constant) and node.value == "quant_mode")
    ]
    assert not named, f"{layout}: modeling.py lines {named} name a quant mode"
    changed = [
        node.lineno
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Store)
            and getattr(node.value, "id", None) in tables
        )
        or (
            isinstance(node, ast.Attribute)
            and getattr(node.value, "id", None) in tables
            and node.attr in ("update", "pop", "setdefault", "clear", "popitem")
        )
    ]
    assert not changed, f"{layout}: modeling.py lines {changed} change a layer constant"


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_core_refuses_another_kv_cache_before_it_builds(layout):
    tree = _tree(layout)
    (check,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_check_build"]
    asserts = {ast.unparse(n.test) for n in ast.walk(check) if isinstance(n, ast.Assert)}
    assert "quant.kv_cache_quant_algo == QuantAlgo.FP8" in asserts, layout
    inits = [
        node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef)
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    ]
    callers = [
        init
        for init in inits
        if any(
            getattr(c.func, "id", None) == "_check_build"
            for c in ast.walk(init)
            if isinstance(c, ast.Call)
        )
    ]
    assert len(callers) == 1, layout
    # The shell checks first: before the core, its weights and views, and the base constructor.
    body = [
        s
        for s in callers[0].body
        if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
    ]
    assert ast.unparse(body[0]) == "_check_build(model_config)", (layout, ast.unparse(body[0]))


def test_the_generation_rope_reaches_the_op_with_the_quant_mode():
    """The catalog entry the targets call hands its ``quant_mode`` to the op unchanged."""
    tree = ast.parse(ROPE_ENTRY.read_text(), str(ROPE_ENTRY))
    (entry,) = [
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MlaRopeGeneration"
    ]
    (call,) = [n for n in entry.body if isinstance(n, ast.FunctionDef) and n.name == "__call__"]
    params = [a.arg for a in call.args.args]
    (op,) = [
        n
        for n in ast.walk(call)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "torch.ops.trtllm.mla_rope_generation"
    ]
    args = [ast.unparse(a) for a in op.args]
    # Every parameter but self goes to the op positionally, in order.
    assert args == params[1:], args
    assert "quant_mode" in args
