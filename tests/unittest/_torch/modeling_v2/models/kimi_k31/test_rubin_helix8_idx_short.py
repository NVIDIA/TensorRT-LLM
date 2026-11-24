# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""helix8's indexer projection widens ``x`` and scales its logits weights in fewer kernels with the
same bits.

``x.float()`` becomes the full-width widening ``rubin_upcast_fp32`` and ``w * qs * scale`` becomes
one kernel (``rubin_idx_weights``). Each entry's own test holds it to the torch call it replaces.
The FP32 key keeps its own cast to ``x``'s dtype before ``rubin_layer_norm``: a norm kernel that
rounds the key itself may sum a row in another order, so the projection uses none. The projection
runs in three parts (dsa_idx_fork): ``_idx_keys`` widens ``x`` and normalizes the key,
``_idx_queries`` quantizes the queries, and ``mla_generate`` scales the weights once both are in.

CPU (``cpu_only``), from helix8's modeling.py source:

* the three parts call the two entries once each and none of the calls they replace, and
  normalize the key as ``rubin_layer_norm`` of ``ik.to(x.dtype)``; the same check flags today's
  parts and a key rounded inside its norm (the planted faults);
* modeling.py imports ``rubin_layer_norm`` and no rounding norm; the norm's module defines none;
* run over the entries' references on an eager step, it returns what today's function returns on
  the CPU, bit for bit: the FP8 queries and keys, the key scales and the logits weights;
* the first call logs the INFO line the README names, once; the line names the two parts.
"""

from __future__ import annotations
import __future__

import ast
import copy
import functools
import importlib.util
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.fused_cat_fp8 import (  # noqa: E501
    fused_cat_fp8,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_idx_weights import (  # noqa: E501
    rubin_idx_weights,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_upcast_fp32 import (  # noqa: E501
    rubin_upcast_fp32,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm.rubin_layer_norm import (  # noqa: E501
    _layer_norm,
)

pytestmark = pytest.mark.cpu_only

_TARGET = "kimi_k31_pro_nvfp4__sm_107__helix8"
_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_FAMILY = _PKG_ROOT / "_torch/_experimental/modeling_v2/models/kimi_k31"
_MODELING = _FAMILY / _TARGET / "modeling.py"

_HIDDEN, _Q_LORA, _IDX_HEADS, _IDX_DIM = 7168, 1536, 32, 128
_ROWS = 12

# The INFO line: the two parts the projection runs.
_LINE = (
    "Kimi K3.1 Rubin DSA indexer: x widened with full-width accesses, "
    "(w * qs) * scale in one kernel"
)

# The projection's parts: the key and weights on x alone, the queries, and the step that scales
# the weights with the query scales.
_PARTS = ("_idx_keys", "_idx_queries", "mla_generate")

_QUERIES = """
def _idx_queries(L, qr):
    iq = cublas_mm(qr, L.idx_wq_b_t).view(-1, _IDX_HEADS, _IDX_DIM)
    iq_a, iq_b = iq.split([_IDX_HALF, _IDX_DIM - _IDX_HALF], -1)
    q8, qs = fused_cat_fp8(iq_a, iq_b, True)
    return q8.view(-1, _IDX_HEADS, _IDX_DIM), qs.view(-1, _IDX_HEADS)
"""

# Today's parts (the base of this change), for the planted fault and the bitwise comparison.
_TODAY = (
    """
def _idx_keys(L, x):
    ik, w = rubin_linear(x.float(), L.idx_wk_wp, True).split([_IDX_DIM, _IDX_HEADS], -1)
    ik = ik.to(x.dtype)
    ik = rubin_layer_norm(ik, L.idx_k_norm_w, L.idx_k_norm_b, _IDX_K_NORM_EPS)
    ik_a, ik_b = ik.split([_IDX_HALF, _IDX_DIM - _IDX_HALF], -1)
    k8, ks = fused_cat_fp8(ik_a, ik_b, True)
    return k8, ks, w
"""
    + _QUERIES
    + """
def mla_generate(L, i, x, st, rt):
    w = w * qs * _IDX_WEIGHT_SCALE
"""
)

# The key rounded inside its LayerNorm's kernel: a planted fault (not the two kernels' bits).
_KEY_ROUNDED_IN_NORM = (
    """
def _idx_keys(L, x):
    ik, w = rubin_linear(rubin_upcast_fp32(x), L.idx_wk_wp, True).split([_IDX_DIM, _IDX_HEADS], -1)
    ik = rubin_layer_norm_rounded(ik, L.idx_k_norm_w, L.idx_k_norm_b, _IDX_K_NORM_EPS, x.dtype)
    ik_a, ik_b = ik.split([_IDX_HALF, _IDX_DIM - _IDX_HALF], -1)
    k8, ks = fused_cat_fp8(ik_a, ik_b, True)
    return k8, ks, w
"""
    + _QUERIES
    + """
def mla_generate(L, i, x, st, rt):
    w = rubin_idx_weights(w, qs, _IDX_WEIGHT_SCALE)
"""
)

# An eager step's projection as mla_generate runs it: the two parts, then the step's statements
# that scale the weights (and log the line).
_EAGER = """
def _projection(L, x, qr):
    k8, ks, w = _idx_keys(L, x)
    q8, qs = _idx_queries(L, qr)
    return q8, k8, ks, w
"""


@functools.lru_cache(maxsize=None)
def _tree() -> ast.Module:
    return ast.parse(_MODELING.read_text(), filename=str(_MODELING))


@functools.lru_cache(maxsize=None)
def _constants() -> MappingProxyType:
    """modeling.py's module-level constants that evaluate on their own, in source order."""
    ns: dict = {"MappingProxyType": MappingProxyType, "torch": torch}
    for node in _tree().body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        if not isinstance(node.targets[0], ast.Name):
            continue
        try:
            value = eval(compile(ast.Expression(node.value), str(_MODELING), "eval"), ns)
        except Exception:
            continue
        ns[node.targets[0].id] = value
    return MappingProxyType(ns)


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    (fn,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    return fn


def _projection(tree: ast.Module) -> ast.Module:
    """The projection's three parts of ``tree``."""
    return ast.Module(body=[_function(tree, name) for name in _PARTS], type_ignores=[])


def _weights(step: ast.FunctionDef) -> list:
    """``mla_generate``'s statements that scale the weights with the query scales, and its line."""
    return [
        s
        for s in step.body
        if (isinstance(s, ast.Assign) and [ast.unparse(t) for t in s.targets] == ["w"])
        or "_IDX_SHORT_LINE" in ast.unparse(s)
    ]


def _run(parts: ast.Module, namespace: dict):
    """An eager step's projection over ``parts``: ``(L, x, qr) -> (q8, k8, ks, w)``."""
    keys, queries, step = copy.deepcopy(parts.body)
    (projection,) = ast.parse(_EAGER).body
    projection.body[2:2] = _weights(step)
    ns = {**_constants(), **namespace}
    module = ast.fix_missing_locations(
        ast.Module(body=[keys, queries, projection], type_ignores=[])
    )
    flags = __future__.annotations.compiler_flag
    exec(compile(module, str(_MODELING), "exec", flags=flags, dont_inherit=True), ns)
    return ns["_projection"]


def _cast_first(fn: ast.AST, norm: ast.Call) -> bool:
    """The norm takes ``ik.to(x.dtype)``, or ``ik`` right after ``ik = ik.to(x.dtype)``."""
    arg = ast.unparse(norm.args[0]) if norm.args else ""
    if arg == "ik.to(x.dtype)":
        return True
    for f in ast.walk(fn):
        if isinstance(f, ast.FunctionDef):
            for prev, s in zip(f.body, f.body[1:]):
                if any(n is norm for n in ast.walk(s)):
                    return arg == "ik" and ast.unparse(prev) == "ik = ik.to(x.dtype)"
    return False


def _problems(fn: ast.AST) -> list:
    """What keeps ``fn`` (the parts) from the shorter prologue: an entry not called once, a call
    it replaces, or a key that does not reach ``rubin_layer_norm`` through its own cast to ``x``'s
    dtype."""
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
    names = [ast.unparse(c.func) for c in calls]
    problems = []
    for entry in ("rubin_upcast_fp32", "rubin_idx_weights", "rubin_layer_norm"):
        if names.count(entry) != 1:
            problems.append(f"{entry} called {names.count(entry)} times, not once")
    for name in names:
        if "rounded" in name:
            problems.append(f"the key rounded inside its norm: {name}")
    for c in calls:
        if isinstance(c.func, ast.Attribute) and c.func.attr in ("float", "to"):
            if ast.unparse(c) != "ik.to(x.dtype)":
                problems.append(f"a cast: {ast.unparse(c)}")
    for b in ast.walk(fn):
        if isinstance(b, ast.BinOp) and "_IDX_WEIGHT_SCALE" in ast.unparse(b):
            problems.append(f"the weights scaled in torch: {ast.unparse(b)}")
    widen = [c for c in calls if ast.unparse(c.func) == "rubin_upcast_fp32"]
    if widen and [ast.unparse(a) for a in widen[0].args] != ["x"]:
        problems.append("the widening does not take x")
    norm = [c for c in calls if ast.unparse(c.func) == "rubin_layer_norm"]
    if norm and not _cast_first(fn, norm[0]):
        problems.append("the key norm does not take ik.to(x.dtype)")
    return problems


def test_the_projection_widens_x_and_scales_the_weights_in_one_kernel_each():
    assert _problems(_projection(_tree())) == []


def test_todays_projection_is_flagged():
    problems = _problems(_projection(ast.parse(_TODAY)))
    assert any("rubin_upcast_fp32 called 0 times" in p for p in problems), problems
    assert any("rubin_idx_weights called 0 times" in p for p in problems), problems
    assert any("a cast: x.float()" in p for p in problems), problems
    assert any("the weights scaled in torch" in p for p in problems), problems


def test_a_key_rounded_inside_its_norm_is_flagged():
    problems = _problems(_projection(ast.parse(_KEY_ROUNDED_IN_NORM)))
    assert sorted(problems) == [
        "rubin_layer_norm called 0 times, not once",
        "the key rounded inside its norm: rubin_layer_norm_rounded",
    ], problems


def test_no_rounding_norm_is_imported_or_defined():
    imported = {
        alias.asname or alias.name
        for node in _tree().body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "rubin_layer_norm" in imported
    assert not [name for name in imported if "rounded" in name]
    assert "rubin_layer_norm_rounded" not in _MODELING.read_text()
    norm_module = importlib.import_module(_layer_norm.__module__)
    assert not [name for name in vars(norm_module) if "rounded" in name]


def _layer(seed: int) -> SimpleNamespace:
    gen = torch.Generator().manual_seed(seed)
    return SimpleNamespace(
        idx_wk_wp=torch.randn(_IDX_DIM + _IDX_HEADS, _HIDDEN, generator=gen) * 0.02,
        idx_wq_b_t=(torch.randn(_IDX_HEADS * _IDX_DIM, _Q_LORA, generator=gen) * 0.02)
        .to(torch.bfloat16)
        .t(),
        idx_k_norm_w=torch.rand(_IDX_DIM, generator=gen) + 0.5,
        idx_k_norm_b=torch.randn(_IDX_DIM, generator=gen) * 0.1,
    )


class _Log:
    def __init__(self):
        self.lines = []

    def info_once(self, msg, key=None):
        self.lines.append((msg, key))


def _entries(log: _Log) -> dict:
    """The entries' CPU references (the GEMMs in FP32, no TF32 on the CPU)."""
    return {
        "rubin_linear": lambda x, w, tf32: F.linear(x, w),
        "cublas_mm": lambda a, b: torch.mm(a, b),
        "fused_cat_fp8": fused_cat_fp8.reference,
        "rubin_layer_norm": lambda x, w, b, eps: _layer_norm(x, w, b, eps),
        "rubin_upcast_fp32": rubin_upcast_fp32.reference,
        "rubin_idx_weights": rubin_idx_weights.reference,
        "logger": log,
        # The GEMM pins run today's call here (test_rubin_helix8_gemm_pins.py checks them).
        "_gemm_pinned": lambda site, a, b, today, out=None, out_dtype=None: today(),
    }


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.element_size() == 1 else t.view(torch.int32)


@pytest.mark.parametrize("seed", [0, 1])
def test_an_eager_step_returns_todays_bits(seed):
    log = _Log()
    entries = _entries(log)
    tip = _run(_projection(_tree()), entries)
    today = _run(_projection(ast.parse(_TODAY)), entries)
    gen = torch.Generator().manual_seed(100 + seed)
    x = (torch.randn(_ROWS, _HIDDEN, generator=gen) * 2).to(torch.bfloat16)
    qr = (torch.randn(_ROWS, _Q_LORA, generator=gen)).to(torch.bfloat16)
    layer = _layer(seed)
    got = tip(layer, x, qr)
    want = today(layer, x, qr)
    for name, g, w in zip(("q8", "k8", "ks", "w"), got, want):
        assert g.shape == w.shape and g.dtype == w.dtype, name
        assert torch.equal(_bits(g.contiguous()), _bits(w.contiguous())), name


def test_the_first_call_logs_the_readme_line_once():
    log = _Log()
    tip = _run(_projection(_tree()), _entries(log))
    x = torch.zeros(2, _HIDDEN, dtype=torch.bfloat16)
    qr = torch.zeros(2, _Q_LORA, dtype=torch.bfloat16)
    tip(_layer(0), x, qr)
    line = _constants()["_IDX_SHORT_LINE"]
    assert line == _LINE
    assert log.lines == [(line, line)]
    assert f"\n{line}\n" in (_FAMILY / "README.md").read_text()
