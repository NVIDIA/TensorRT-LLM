# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_MERGE_MXFP8 on CPU tensors and stub layers.

LC2 gets the hand-off buffers when they are reserved on the output's device, and the decode publishes them keyed by its
output; every decode call first clears the slot. The o_proj of a layer LC2 serves runs its GEMM on the published pair
when its input is that output, and the stock apply on any other input. Where DEP_PROJ_PIN engages the table, that GEMM
is the pinned CuTe DSL tactic (DEP_PROJ_CUTE), else the tuned GEMM half. The bitwise checks are the GPU tests
(test_modeling_v2_m31_dep_proj_cute_gpu.py for DEP_PROJ_CUTE). CPU only.
"""

import ast
import hashlib
import inspect
import sys
import textwrap
import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch.modules import linear as _linear
from tensorrt_llm._torch.modules.linear import MXFP8LinearMethod, UnquantizedLinearMethod

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
        lc2_decode,
        m31_switches,
        merge_mxfp8,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

pytestmark = pytest.mark.cpu_only

_STOCK_MODULE = (
    "tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.triton_sparse_decode"
)
CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(merge_mxfp8, "_BUFFERS", {})
    monkeypatch.setattr(merge_mxfp8, "DEP_MERGE_MXFP8", True)
    monkeypatch.setattr(merge_mxfp8, "DEP_PROJ_CUTE", True)
    merge_mxfp8.SLOT.clear()
    yield
    merge_mxfp8.SLOT.clear()


def _decode_call(reqs=11, dql=8):
    total_q = reqs * dql
    cache = torch.zeros(6, 2, 4, 128, 64, dtype=torch.uint8)
    scales = torch.zeros(6, 2, 4, 128, 8, dtype=torch.uint8)
    globals_ = torch.ones(2, 4, dtype=torch.float32)
    args = (
        torch.zeros(total_q, 64, 128, dtype=torch.float8_e4m3fn),
        cache[:, 0],
        cache[:, 1],
        torch.full((total_q, 4, 16), -1, dtype=torch.int32).permute(1, 0, 2),
        torch.zeros(reqs, 6, dtype=torch.int32),
        torch.zeros(reqs, dtype=torch.int32),
    )
    kwargs = dict(
        sm_scale=128**-0.5,
        output=torch.empty(total_q, 64, 128, dtype=torch.bfloat16),
        decode_query_len=dql,
        k_block_scale=scales[:, 0],
        v_block_scale=scales[:, 1],
        k_global_scale=globals_[0, :1],
        v_global_scale=globals_[1, :1],
    )
    return args, kwargs


def test_the_switch_is_the_dep_merge_mxfp8_item(monkeypatch):
    monkeypatch.undo()
    assert merge_mxfp8.DEP_MERGE_MXFP8 == (not m31_switches.m31_disabled("DEP_MERGE_MXFP8"))


def test_the_switch_is_the_dep_proj_cute_item(monkeypatch):
    monkeypatch.undo()
    assert merge_mxfp8.DEP_PROJ_CUTE == (not m31_switches.m31_disabled("DEP_PROJ_CUTE"))


def test_the_slot_returns_the_pair_only_for_the_published_output_and_clears():
    x = torch.empty(16, merge_mxfp8.WIDTH, dtype=torch.bfloat16)
    q8, sf = (
        torch.empty(16, merge_mxfp8.WIDTH, dtype=torch.float8_e4m3fn),
        torch.empty(32768, dtype=torch.uint8),
    )
    merge_mxfp8.SLOT.publish(x, q8, sf)
    assert merge_mxfp8.SLOT.take(x) == (q8, sf)
    assert merge_mxfp8.SLOT.pending is None and merge_mxfp8.SLOT.take(x) is None
    for other in (x.clone(), x[:8], x.view(32, merge_mxfp8.WIDTH // 2), x.view(-1)):
        merge_mxfp8.SLOT.publish(x, q8, sf)
        assert merge_mxfp8.SLOT.take(other) is None
        assert merge_mxfp8.SLOT.pending is None


def test_the_buffers_are_views_of_one_reserved_pair():
    assert merge_mxfp8.buffers(88, CPU) is None
    reserved = merge_mxfp8.reserve(CPU)
    assert reserved == 448 * 8192 + 512 * 256
    q8_full, sf_full = merge_mxfp8._BUFFERS[CPU]
    assert merge_mxfp8.reserve(CPU) == reserved and merge_mxfp8._BUFFERS[CPU][0] is q8_full
    assert bool((sf_full == 0).all())
    for rows, sf_bytes in ((8, 128 * 256), (88, 128 * 256), (129, 256 * 256), (448, 512 * 256)):
        q8, sf = merge_mxfp8.buffers(rows, CPU)
        assert q8.shape == (rows, 8192) and q8.dtype == torch.float8_e4m3fn and q8.is_contiguous()
        assert q8.data_ptr() == q8_full.data_ptr() and sf.data_ptr() == sf_full.data_ptr()
        assert sf.numel() == sf_bytes == merge_mxfp8.scale_bytes(rows)
    assert merge_mxfp8.buffers(449, CPU) is None


def test_off_reserves_nothing(monkeypatch):
    monkeypatch.setattr(merge_mxfp8, "DEP_MERGE_MXFP8", False)
    assert merge_mxfp8.reserve(CPU) == 0
    assert merge_mxfp8.buffers(88, CPU) is None


@pytest.mark.parametrize("reqs", [1, 11, 56])
def test_the_decode_hands_lc2_the_buffers_and_publishes_its_output(monkeypatch, reqs):
    calls = []
    monkeypatch.setattr(lc2_decode, "lc2_op", lambda: lambda *a: calls.append(a))
    merge_mxfp8.reserve(CPU)
    args, kwargs = _decode_call(reqs=reqs)
    lc2_decode.dep_lc2_decode(*args, **kwargs)
    (call,) = calls
    rows = reqs * 8
    out, q8, sf = call[10], call[11], call[12]
    assert out.shape == (rows, 8192) and out.data_ptr() == kwargs["output"].data_ptr()
    assert q8.shape == (rows, 8192) and sf.numel() == merge_mxfp8.scale_bytes(rows)
    assert call[13] == 64
    assert merge_mxfp8.SLOT.take(kwargs["output"].view(rows, 8192)) == (q8, sf)


def test_without_buffers_the_decode_hands_none_and_publishes_nothing(monkeypatch):
    calls = []
    monkeypatch.setattr(lc2_decode, "lc2_op", lambda: lambda *a: calls.append(a))
    args, kwargs = _decode_call()
    lc2_decode.dep_lc2_decode(*args, **kwargs)
    (call,) = calls
    assert call[11:13] == (None, None)
    assert merge_mxfp8.SLOT.pending is None


def test_a_declined_call_clears_the_slot_and_publishes_nothing(monkeypatch):
    fake = types.ModuleType(_STOCK_MODULE)
    fake.minimax_m3_sparse_attn_decode = lambda *a, **kw: pytest.fail(
        "the DEP hook has no Triton fallback"
    )
    monkeypatch.setitem(sys.modules, _STOCK_MODULE, fake)
    monkeypatch.setattr(
        lc2_decode, "lc2_op", lambda: lambda *a: pytest.fail("a declined call must not run LC2")
    )
    merge_mxfp8.reserve(CPU)
    x = torch.empty(8, merge_mxfp8.WIDTH, dtype=torch.bfloat16)
    merge_mxfp8.SLOT.publish(x, *merge_mxfp8.buffers(8, CPU))
    args, kwargs = _decode_call(dql=4)
    with pytest.raises(RuntimeError, match="decode_query_len 4"):
        lc2_decode.dep_lc2_decode(*args, **kwargs)
    assert merge_mxfp8.SLOT.pending is None


def _attn(served, method):
    hook = lc2_decode.dep_lc2_decode if served else None
    manager = types.SimpleNamespace(
        fmha_libs=[object(), types.SimpleNamespace(sparse_decode_hook=hook)]
    )
    return types.SimpleNamespace(
        attn=types.SimpleNamespace(_fmha_manager=manager),
        o_proj=types.SimpleNamespace(quant_method=method),
    )


def test_install_swaps_the_mxfp8_o_proj_of_each_layer_lc2_serves():
    taken, unserved, bf16 = MXFP8LinearMethod(), MXFP8LinearMethod(), UnquantizedLinearMethod()
    layers = [
        types.SimpleNamespace(self_attn=_attn(True, taken)),
        types.SimpleNamespace(self_attn=_attn(False, unserved)),
        types.SimpleNamespace(self_attn=_attn(True, bf16)),
        types.SimpleNamespace(),
    ]
    assert merge_mxfp8.install(layers, lc2_decode.serves) == 1
    assert type(taken) is merge_mxfp8.DepMergeMxfp8LinearMethod
    assert type(unserved) is MXFP8LinearMethod and type(bf16) is UnquantizedLinearMethod
    assert merge_mxfp8.install(layers, lc2_decode.serves) == 1


def test_install_swaps_nothing_when_off(monkeypatch):
    monkeypatch.setattr(merge_mxfp8, "DEP_MERGE_MXFP8", False)
    method = MXFP8LinearMethod()
    assert (
        merge_mxfp8.install(
            [types.SimpleNamespace(self_attn=_attn(True, method))], lc2_decode.serves
        )
        == 0
    )
    assert type(method) is MXFP8LinearMethod


def _method(**attrs):
    method = MXFP8LinearMethod()
    method.__class__ = merge_mxfp8.DepMergeMxfp8LinearMethod
    method.use_cutlass = True
    method.backend = "trtllm"
    method.use_native_autotuner = False
    method._flashinfer_autotuned = False
    method.tune_decode_graph_backends = False
    for name, value in attrs.items():
        setattr(method, name, value)
    return method


_MODULE = types.SimpleNamespace(
    weight=torch.empty(4, 8192, dtype=torch.float8_e4m3fn),
    weight_scale=torch.empty(512, dtype=torch.uint8),
    dtype=torch.bfloat16,
    _mxfp8_alpha=torch.ones(1),
)


def _published(rows=16):
    merge_mxfp8.reserve(CPU)
    x = torch.empty(rows, merge_mxfp8.WIDTH, dtype=torch.bfloat16)
    pair = merge_mxfp8.buffers(rows, CPU)
    merge_mxfp8.SLOT.publish(x, *pair)
    return x, pair


def test_apply_in_decode_graphs_runs_the_tuned_gemm_half_on_the_hand_off(monkeypatch):
    seen = []
    monkeypatch.setattr(
        merge_mxfp8, "mxfp8_gemm_autotuned", lambda *a, **kw: seen.append((a, kw)) or torch.zeros(1)
    )
    x, (q8, sf) = _published()
    token = _linear._FLASHINFER_MXFP8_DECODE_GRAPH_CAPTURE_ACTIVE.set(True)
    try:
        out = _method(tune_decode_graph_backends=True).apply(_MODULE, x, None)
    finally:
        _linear._FLASHINFER_MXFP8_DECODE_GRAPH_CAPTURE_ACTIVE.reset(token)
    ((a, kw),) = seen
    assert a[0] is q8 and a[1] is sf and a[2] is _MODULE.weight and a[3] is _MODULE.weight_scale
    assert a[4] is torch.bfloat16 and kw == dict(tune=False) and out.shape == (1,)
    assert merge_mxfp8.SLOT.pending is None


def _decode_graph_apply(method, module, x, bias=None):
    token = _linear._FLASHINFER_MXFP8_DECODE_GRAPH_CAPTURE_ACTIVE.set(True)
    try:
        return method.apply(module, x, bias)
    finally:
        _linear._FLASHINFER_MXFP8_DECODE_GRAPH_CAPTURE_ACTIVE.reset(token)


def test_where_the_pin_engages_the_hand_off_runs_the_pinned_cute_tactic(monkeypatch):
    pinned, tuned, lines = [], [], []
    monkeypatch.setattr(merge_mxfp8, "_LOGGED", set())
    monkeypatch.setattr(merge_mxfp8.logger, "info", lines.append)
    monkeypatch.setattr(merge_mxfp8._pin, "run", lambda *a: pinned.append(a) or torch.ones(2))
    monkeypatch.setattr(merge_mxfp8, "mxfp8_gemm_autotuned", lambda *a, **kw: tuned.append(a))
    x, (q8, sf) = _published()
    bias = torch.full((2,), 3.0)
    out = _decode_graph_apply(_method(tune_decode_graph_backends=True), _MODULE, x, bias)
    ((a),) = pinned
    assert a == (q8, sf, _MODULE.weight, _MODULE.weight_scale, torch.bfloat16)
    assert tuned == [] and torch.equal(out, torch.full((2,), 4.0))
    assert len(lines) == 1 and "taken" in lines[0] and "DEP_PROJ_CUTE" in lines[0]
    assert merge_mxfp8.SLOT.pending is None


@pytest.mark.parametrize("killed", [False, True], ids=["unengaged", "killed"])
def test_unengaged_or_killed_the_hand_off_keeps_the_tuned_gemm(monkeypatch, killed):
    pinned, tuned = [], []
    monkeypatch.setattr(merge_mxfp8, "DEP_PROJ_CUTE", not killed)
    monkeypatch.setattr(merge_mxfp8._pin, "run", lambda *a: pinned.append(a))
    monkeypatch.setattr(
        merge_mxfp8, "mxfp8_gemm_autotuned", lambda *a, **kw: tuned.append(a) or torch.zeros(1)
    )
    x, (q8, sf) = _published()
    _decode_graph_apply(_method(tune_decode_graph_backends=True), _MODULE, x)
    assert len(pinned) == (0 if killed else 1) and len(tuned) == 1 and tuned[0][0] is q8


def test_outside_an_engagement_the_pin_declines(monkeypatch):
    """The real table: no engaged caller, so it declines before reading the device."""
    x, (q8, sf) = _published()
    assert (
        merge_mxfp8._pin.run(q8, sf, _MODULE.weight, _MODULE.weight_scale, torch.bfloat16) is None
    )


def test_apply_outside_decode_graphs_runs_the_native_gemm_on_the_hand_off(monkeypatch):
    seen = []
    monkeypatch.setattr(
        torch.ops.trtllm,
        "mxfp8_mxfp8_gemm",
        lambda *a: seen.append(a) or torch.zeros(1),
        raising=False,
    )
    x, (q8, sf) = _published()
    _method().apply(_MODULE, x, None)
    ((a),) = seen
    assert a[0] is q8 and a[1] is sf and a[4] is _MODULE._mxfp8_alpha and a[5] is torch.bfloat16


def test_apply_runs_the_stock_apply_on_any_other_input(monkeypatch):
    stock = []
    monkeypatch.setattr(MXFP8LinearMethod, "apply", lambda self, m, i, b: stock.append(i) or i)
    x, pair = _published()
    other = x.clone()
    assert _method().apply(_MODULE, other, None) is other
    assert stock == [other] and merge_mxfp8.SLOT.pending is None
    assert _method().apply(_MODULE, x, None) is x
    merge_mxfp8.SLOT.publish(x, *pair)
    assert _method(use_cutlass=False).apply(_MODULE, x, None) is x
    assert stock == [other, x, x] and merge_mxfp8.SLOT.pending is None


def test_a_take_and_a_miss_are_logged_once_per_width(monkeypatch):
    lines = []
    monkeypatch.setattr(merge_mxfp8, "_LOGGED", set())
    monkeypatch.setattr(merge_mxfp8.logger, "info", lines.append)
    monkeypatch.setattr(
        torch.ops.trtllm, "mxfp8_mxfp8_gemm", lambda *a: torch.zeros(1), raising=False
    )
    monkeypatch.setattr(MXFP8LinearMethod, "apply", lambda self, m, i, b: i)
    x, pair = _published()
    _method().apply(_MODULE, x, None)
    merge_mxfp8.SLOT.publish(x, *pair)
    _method().apply(_MODULE, x, None)
    merge_mxfp8.SLOT.publish(x, *pair)
    _method().apply(_MODULE, x.clone(), None)
    _method().apply(_MODULE, x, None)
    assert len(lines) == 2
    assert "taken" in lines[0] and "16 rows" in lines[0]
    assert "missed" in lines[1] and "16-row" in lines[1]


def _source_md5(cls, name):
    source = inspect.getsource(_linear)
    lines = source.splitlines(keepends=True)
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == name:
                    text = textwrap.dedent("".join(lines[fn.lineno - 1 : fn.end_lineno]))
                    return hashlib.md5(text.encode()).hexdigest()
    raise AssertionError(f"{cls.__name__}.{name} not found")


def test_the_stock_apply_the_hand_off_mirrors_is_unchanged():
    """DepMergeMxfp8LinearMethod.apply copies this apply's GEMM half; a change here must be made there too."""
    assert _source_md5(MXFP8LinearMethod, "apply") == "075a5b9401835cbb6e0f5b52cf0535dd"
