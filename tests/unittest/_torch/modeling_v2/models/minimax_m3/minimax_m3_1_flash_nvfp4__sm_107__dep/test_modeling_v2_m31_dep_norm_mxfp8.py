# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_NORM_MXFP8 on CPU tensors and stub layers.

The install wraps each boundary norm that feeds an MXFP8 qkv_proj and swaps that qkv_proj. A wrapped norm runs the
add + RMSNorm kernel with the hand-off buffers when it serves the call, and publishes its output; DEP_TAIL's form (b)
gets the same buffers at a site. The qkv_proj takes the norm's hand-off, not the decode's. The kernels are replaced by
recorders; their bitwise checks are the GPU tests. CPU only.
"""

import ast
import hashlib
import inspect
import textwrap
import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch.modules import rms_norm
from tensorrt_llm._torch.modules.linear import MXFP8LinearMethod, UnquantizedLinearMethod
from tensorrt_llm._torch.modules.rms_norm import RMSNorm

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
        m31_switches,
        merge_mxfp8,
        norm_mxfp8,
        tail,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

pytestmark = pytest.mark.cpu_only

CPU = torch.device("cpu")
HIDDEN = 128


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(norm_mxfp8, "_BUFFERS", {})
    monkeypatch.setattr(norm_mxfp8, "DEP_NORM_MXFP8", True)
    norm_mxfp8.SLOT.clear()
    merge_mxfp8.SLOT.clear()
    yield
    norm_mxfp8.SLOT.clear()
    merge_mxfp8.SLOT.clear()


def test_the_switch_is_the_dep_norm_mxfp8_item(monkeypatch):
    monkeypatch.undo()
    assert norm_mxfp8.DEP_NORM_MXFP8 == (not m31_switches.m31_disabled("DEP_NORM_MXFP8"))


def _rows(rows, hidden=HIDDEN):
    return torch.zeros(rows, hidden, dtype=torch.bfloat16)


def _metadata():
    return types.SimpleNamespace(all_rank_num_tokens=[3, 7])


def test_the_buffers_are_views_of_one_reserved_pair_per_width():
    assert norm_mxfp8.buffers(_rows(8)) is None
    reserved = norm_mxfp8.reserve(CPU, HIDDEN)
    # 448 E4M3 rows of 128 and 4 tiles of 128 rows x 4 scale columns.
    assert reserved == 448 * HIDDEN + 512 * 4
    q8_full, sf_full = norm_mxfp8._BUFFERS[(CPU, HIDDEN)]
    assert norm_mxfp8.reserve(CPU, HIDDEN) == reserved
    assert norm_mxfp8._BUFFERS[(CPU, HIDDEN)][0] is q8_full and bool((sf_full == 0).all())
    for rows, sf_bytes in ((1, 512), (88, 512), (129, 1024), (448, 2048)):
        q8, sf = norm_mxfp8.buffers(_rows(rows))
        assert q8.shape == (rows, HIDDEN) and q8.dtype == torch.float8_e4m3fn
        assert q8.data_ptr() == q8_full.data_ptr() and sf.data_ptr() == sf_full.data_ptr()
        assert sf.numel() == sf_bytes == merge_mxfp8.scale_bytes(rows, HIDDEN)
    assert norm_mxfp8.buffers(_rows(0)) is None and norm_mxfp8.buffers(_rows(449)) is None
    assert norm_mxfp8.buffers(_rows(8, 2 * HIDDEN)) is None


def test_the_scales_of_the_decode_width_are_unchanged():
    assert (
        merge_mxfp8.scale_bytes(448) == merge_mxfp8.scale_bytes(448, merge_mxfp8.WIDTH) == 512 * 256
    )
    assert merge_mxfp8.scale_bytes(88, 6144) == 128 * 192


def test_off_reserves_nothing(monkeypatch):
    monkeypatch.setattr(norm_mxfp8, "DEP_NORM_MXFP8", False)
    assert norm_mxfp8.reserve(CPU, HIDDEN) == 0
    assert norm_mxfp8.buffers(_rows(8)) is None


# --- install ---------------------------------------------------------------------------------------------------------


def _norm():
    return RMSNorm(hidden_size=HIDDEN, eps=1e-6, dtype=torch.bfloat16)


def _model(*methods):
    """Decoder layers whose layer i + 1 has qkv_proj quant method methods[i], each boundary norm wired as setup_aliases
    wires it."""
    layers = [
        types.SimpleNamespace(
            input_layernorm=_norm(),
            self_attn=types.SimpleNamespace(qkv_proj=types.SimpleNamespace(quant_method=method)),
            post_feed_forward_fusion=False,
        )
        for method in (MXFP8LinearMethod(),) + methods
    ]
    for layer, next_layer in zip(layers, layers[1:]):
        layer.next_layer_layernorm = next_layer.input_layernorm
    layers[-1].next_layer_layernorm = _norm()
    return layers


@pytest.fixture
def flashinfer_norms(monkeypatch):
    monkeypatch.setattr(tail, "boundary_norm_decline_reason", lambda layer: None)


def test_install_wraps_each_boundary_norm_before_an_mxfp8_qkv_proj(flashinfer_norms):
    layers = _model(MXFP8LinearMethod(), UnquantizedLinearMethod(), MXFP8LinearMethod())
    assert norm_mxfp8.install(layers) == 2
    sites = [norm_mxfp8.is_site(layer.input_layernorm) for layer in layers]
    assert sites == [False, True, False, True]
    methods = [type(layer.self_attn.qkv_proj.quant_method) for layer in layers]
    assert methods == [
        MXFP8LinearMethod,
        norm_mxfp8.DepNormMxfp8LinearMethod,
        UnquantizedLinearMethod,
        norm_mxfp8.DepNormMxfp8LinearMethod,
    ]
    assert not norm_mxfp8.is_site(layers[-1].next_layer_layernorm)
    assert norm_mxfp8.install(layers) == 2


def test_install_declines_an_unwired_or_declined_boundary(monkeypatch):
    layers = _model(MXFP8LinearMethod(), MXFP8LinearMethod(), MXFP8LinearMethod())
    layers[0].next_layer_layernorm = None
    layers[2].input_layernorm.forward = lambda *a, **kw: None
    monkeypatch.setattr(
        tail,
        "boundary_norm_decline_reason",
        lambda layer: "FlashInfer is not available" if layer is layers[1] else None,
    )
    assert norm_mxfp8.install(layers) == 1
    assert [norm_mxfp8.is_site(layer.input_layernorm) for layer in layers] == [
        False,
        False,
        False,
        True,
    ]
    assert norm_mxfp8.site_decline_reason(layers[0], layers[1]) == (
        "the boundary norm is not the next layer's input norm"
    )
    assert norm_mxfp8.site_decline_reason(layers[1], layers[2]) == "FlashInfer is not available"
    assert norm_mxfp8.site_decline_reason(layers[2], layers[3]) is None
    monkeypatch.setattr(tail, "boundary_norm_decline_reason", lambda layer: None)
    assert norm_mxfp8.site_decline_reason(layers[1], layers[2]) == (
        "the norm's forward is already replaced"
    )


def test_install_wraps_nothing_when_off(monkeypatch, flashinfer_norms):
    monkeypatch.setattr(norm_mxfp8, "DEP_NORM_MXFP8", False)
    layers = _model(MXFP8LinearMethod())
    assert norm_mxfp8.install(layers) == 0
    assert not norm_mxfp8.is_site(layers[1].input_layernorm)
    assert type(layers[1].self_attn.qkv_proj.quant_method) is MXFP8LinearMethod


# --- the wrapped norm ------------------------------------------------------------------------------------------------


@pytest.fixture
def site(monkeypatch, flashinfer_norms):
    """A wrapped boundary norm whose kernel and stock forward record their calls."""
    layers = _model(MXFP8LinearMethod())
    assert norm_mxfp8.install(layers) == 1
    norm = layers[1].input_layernorm
    norm.kernel_calls, norm.stock_calls = [], []
    monkeypatch.setattr(
        norm_mxfp8,
        "kernel",
        lambda: lambda *args, **kwargs: norm.kernel_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        RMSNorm,
        "forward",
        lambda self, h, r=..., *, return_norm_out=False: self.stock_calls.append((h, r)) or (h, r),
    )
    norm_mxfp8.reserve(CPU, HIDDEN)
    return norm


def _same(pair, expected):
    return len(pair) == len(expected) and all(a is b for a, b in zip(pair, expected))


def test_a_served_call_runs_the_kernel_and_publishes_its_output(site):
    x, residual = _rows(88), _rows(88)
    assert _same(site(x, residual), (x, residual))
    ((args, kwargs),) = site.kernel_calls
    q8, sf = norm_mxfp8.buffers(x)
    assert _same(args[:3], (x, residual, site.weight)) and args[3] == site.variance_epsilon
    assert args[4].data_ptr() == q8.data_ptr() and args[4].shape == q8.shape
    assert args[5].data_ptr() == sf.data_ptr() and args[5].numel() == sf.numel()
    assert (
        set(kwargs) == {"enable_pdl", "prefetch"}
        and kwargs["prefetch"] is None
        and site.stock_calls == []
    )
    assert _same(norm_mxfp8.SLOT.take(x), args[4:6])
    assert merge_mxfp8.SLOT.pending is None


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda norm: norm(_rows(8)), id="no-residual"),
        pytest.param(lambda norm: norm(_rows(8), None), id="none-residual"),
        pytest.param(lambda norm: norm(_rows(8), _rows(8), return_norm_out=True), id="norm-out"),
        pytest.param(lambda norm: norm(_rows(449), _rows(449)), id="rows"),
        pytest.param(lambda norm: norm(_rows(8).float(), _rows(8).float()), id="fp32"),
        pytest.param(lambda norm: norm(_rows(8), _rows(9)), id="shape"),
        pytest.param(
            lambda norm: norm(torch.zeros(HIDDEN, 8, dtype=torch.bfloat16).t(), _rows(8)),
            id="strided",
        ),
    ],
)
def test_a_call_the_kernel_does_not_serve_is_the_stock_norm(site, call):
    call(site)
    assert site.kernel_calls == [] and len(site.stock_calls) == 1
    assert norm_mxfp8.SLOT.pending is None


def test_a_norm_that_gained_an_nvfp4_scale_is_the_stock_norm(site):
    site.nvfp4_scale = torch.ones(1)
    site(_rows(8), _rows(8))
    assert site.kernel_calls == [] and len(site.stock_calls) == 1


def test_the_stock_norm_the_kernel_replaces_is_unchanged():
    """The wrapper serves only the calls this forward sends to flashinfer_fused_add_rmsnorm; a change here must be
    checked there."""
    source = inspect.getsource(rms_norm)
    lines = source.splitlines(keepends=True)
    (cls,) = [
        n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "RMSNorm"
    ]
    (fn,) = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"]
    text = textwrap.dedent("".join(lines[fn.lineno - 1 : fn.end_lineno]))
    assert hashlib.md5(text.encode()).hexdigest() == "c853f060681b7936af9d559adb87359c"


# --- the qkv_proj ----------------------------------------------------------------------------------------------------


def _method():
    method = MXFP8LinearMethod()
    method.__class__ = norm_mxfp8.DepNormMxfp8LinearMethod
    method.use_cutlass = True
    method.backend = "trtllm"
    method.use_native_autotuner = False
    method._flashinfer_autotuned = False
    method.tune_decode_graph_backends = False
    return method


_MODULE = types.SimpleNamespace(
    weight=torch.empty(4, HIDDEN, dtype=torch.float8_e4m3fn),
    weight_scale=torch.empty(512, dtype=torch.uint8),
    dtype=torch.bfloat16,
    _mxfp8_alpha=torch.ones(1),
)


def test_the_qkv_proj_takes_the_norms_hand_off_not_the_decodes(monkeypatch):
    seen, stock, lines = [], [], []
    monkeypatch.setattr(
        torch.ops.trtllm,
        "mxfp8_mxfp8_gemm",
        lambda *a: seen.append(a) or torch.zeros(1),
        raising=False,
    )
    monkeypatch.setattr(MXFP8LinearMethod, "apply", lambda self, m, i, b: stock.append(i) or i)
    monkeypatch.setattr(norm_mxfp8, "_LOGGED", set())
    monkeypatch.setattr(norm_mxfp8.logger, "info", lines.append)
    norm_mxfp8.reserve(CPU, HIDDEN)
    x = _rows(16)
    pair = norm_mxfp8.buffers(x)
    merge_mxfp8.SLOT.publish(x, *pair)
    assert _method().apply(_MODULE, x, None) is x
    assert stock == [x] and seen == [] and merge_mxfp8.SLOT.pending is not None
    norm_mxfp8.SLOT.publish(x, *pair)
    _method().apply(_MODULE, x, None)
    ((a),) = seen
    assert a[0] is pair[0] and a[1] is pair[1] and norm_mxfp8.SLOT.pending is None
    norm_mxfp8.SLOT.publish(x, *pair)
    _method().apply(_MODULE, x.clone(), None)
    assert len(lines) == 2
    assert "DEP_NORM_MXFP8 taken" in lines[0] and "qkv_proj GEMM on the boundary norm's" in lines[0]
    assert "DEP_NORM_MXFP8 missed" in lines[1] and "16-row boundary norm output" in lines[1]


# --- NORM_L2PF -------------------------------------------------------------------------------------------------------


def test_the_l2pf_switch_is_the_norm_l2pf_item():
    assert norm_mxfp8.NORM_L2PF == (not m31_switches.m31_disabled("NORM_L2PF"))


def _l2pf_site(monkeypatch, armed):
    """A wrapped boundary norm (NORM_L2PF ``armed`` or not) whose next qkv_proj carries an MXFP8 weight + scales, with
    a recording kernel and the op's prefetch arguments present."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops import (
        add_rmsnorm as native,
    )

    monkeypatch.setattr(tail, "boundary_norm_decline_reason", lambda layer: None)
    monkeypatch.setattr(norm_mxfp8, "NORM_L2PF", armed)
    monkeypatch.setattr(native, "takes_prefetch", lambda: True)
    lines = []
    monkeypatch.setattr(norm_mxfp8, "_LOGGED", set())
    monkeypatch.setattr(norm_mxfp8.logger, "info", lines.append)
    layers = _model(MXFP8LinearMethod())
    qkv_proj = layers[1].self_attn.qkv_proj
    qkv_proj.weight = torch.zeros(320, HIDDEN, dtype=torch.float8_e4m3fn)
    qkv_proj.weight_scale = torch.zeros(1600, dtype=torch.uint8)
    assert norm_mxfp8.install(layers) == 1
    norm = layers[1].input_layernorm
    norm.kernel_calls = []
    monkeypatch.setattr(
        norm_mxfp8,
        "kernel",
        lambda: lambda *args, **kwargs: norm.kernel_calls.append((args, kwargs)),
    )
    norm_mxfp8.reserve(CPU, HIDDEN)
    return norm, qkv_proj, lines


def test_an_armed_site_prefetches_a_quarter_of_the_next_qkv_weight_in_the_band(monkeypatch):
    norm, qkv_proj, lines = _l2pf_site(monkeypatch, armed=True)
    assert any(line.startswith("[M3.1 DEP] NORM_L2PF: on: 1 qkv boundary norms") for line in lines)
    for rows, want in ((64, None), (65, True), (88, True), (128, True), (129, None)):
        norm(_rows(rows), _rows(rows))
        prefetch = norm.kernel_calls[-1][1]["prefetch"]
        if want is None:
            assert prefetch is None, rows
        else:
            (w, w_bytes), (s, s_bytes) = prefetch
            assert w is qkv_proj.weight and s is qkv_proj.weight_scale
            assert (w_bytes, s_bytes) == (320 * HIDDEN // 4, 1600 // 4), rows
    assert "[M3.1 DEP] NORM_L2PF: taken at 88 rows" in lines


def test_off_arms_no_site_and_says_so(monkeypatch):
    norm, _, lines = _l2pf_site(monkeypatch, armed=False)
    assert "_m31_l2pf_qkv" not in norm.__dict__
    norm(_rows(88), _rows(88))
    assert norm.kernel_calls[-1][1]["prefetch"] is None
    assert "[M3.1 DEP] NORM_L2PF: off (TRTLLM_M31_DISABLE_NORM_L2PF=1)" in lines


def test_the_armed_target_is_not_a_submodule(monkeypatch):
    norm, _, _ = _l2pf_site(monkeypatch, armed=True)
    assert all(module is norm for module in norm.modules())
    assert not any(name.startswith("_m31") for name, _ in norm.named_parameters())
