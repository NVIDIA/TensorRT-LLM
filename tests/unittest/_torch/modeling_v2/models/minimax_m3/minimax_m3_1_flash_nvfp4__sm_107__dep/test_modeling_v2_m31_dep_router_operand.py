# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_ROUTER_OPERAND on CPU tensors and stub layers.

The install wraps the post-attention norm and the gate of each MoE layer whose gate runs the split BF16 router GEMM.
A wrapped norm runs the add + RMSNorm kernel with the operand buffer when it serves the call, and publishes its
output; the gate runs the stock GEMM on the published copies when its input is that output, and is the stock gate
otherwise. The kernel and the GEMM are replaced by recorders; their bitwise checks are the GPU tests. CPU only.
"""

import ast
import hashlib
import inspect
import textwrap
import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch.models import modeling_minimaxm3
from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3Gate
from tensorrt_llm._torch.modules import rms_norm
from tensorrt_llm._torch.modules.rms_norm import RMSNorm

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
        m31_switches,
        router_operand,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

pytestmark = pytest.mark.cpu_only

CPU = torch.device("cpu")
HIDDEN = 128
EXPERTS = 4


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(router_operand, "_BUFFERS", {})
    monkeypatch.setattr(router_operand, "DEP_ROUTER_OPERAND", True)
    router_operand.SLOT.clear()
    yield
    router_operand.SLOT.clear()


def test_the_switch_is_the_dep_router_operand_item(monkeypatch):
    monkeypatch.undo()
    assert router_operand.DEP_ROUTER_OPERAND == (
        not m31_switches.m31_disabled("DEP_ROUTER_OPERAND")
    )


def test_the_operand_holds_the_split_terms():
    assert router_operand.TERMS == modeling_minimaxm3.ROUTER_BF16_SPLIT_TERMS == 3


def _rows(rows, hidden=HIDDEN):
    return torch.zeros(rows, hidden, dtype=torch.bfloat16)


def test_the_operands_are_views_of_one_reserved_buffer_per_width():
    assert router_operand.operand(_rows(8)) is None
    reserved = router_operand.reserve(CPU, HIDDEN)
    assert reserved == 448 * 3 * HIDDEN * 2
    full = router_operand._BUFFERS[(CPU, HIDDEN)]
    assert router_operand.reserve(CPU, HIDDEN) == reserved
    assert router_operand._BUFFERS[(CPU, HIDDEN)] is full
    for rows in (1, 88, 129, 448):
        view = router_operand.operand(_rows(rows))
        assert view.shape == (rows, 3 * HIDDEN) and view.dtype == torch.bfloat16
        assert view.data_ptr() == full.data_ptr() and view.is_contiguous()
    assert router_operand.operand(_rows(0)) is None
    assert router_operand.operand(_rows(449)) is None
    assert router_operand.operand(_rows(8, 2 * HIDDEN)) is None


def test_off_reserves_nothing(monkeypatch):
    monkeypatch.setattr(router_operand, "DEP_ROUTER_OPERAND", False)
    assert router_operand.reserve(CPU, HIDDEN) == 0
    assert router_operand.operand(_rows(8)) is None


def test_the_slot_returns_the_operand_only_for_the_published_output_and_clears():
    x = _rows(8)
    operand = torch.zeros(8, 3 * HIDDEN, dtype=torch.bfloat16)
    router_operand.SLOT.publish(x, operand)
    assert router_operand.SLOT.take(x) is operand and router_operand.SLOT.pending is None
    assert router_operand.SLOT.take(x) is None
    for other in (x.clone(), x[:4], torch.zeros(8, 2 * HIDDEN, dtype=torch.bfloat16)[:, :HIDDEN]):
        router_operand.SLOT.publish(x, operand)
        assert router_operand.SLOT.take(other) is None and router_operand.SLOT.pending is None
    router_operand.SLOT.publish(x, operand[:, : 2 * HIDDEN])
    assert router_operand.SLOT.take(x) is None


# --- install ---------------------------------------------------------------------------------------------------------


def _gate(split=True):
    gate = MiniMaxM3Gate(HIDDEN, EXPERTS, 2, 1.0)
    if split:
        gate.load_weights([{"weight": torch.randn(EXPERTS, HIDDEN)}])
    return gate


def _layer(gate=None, fusion=False):
    """A decoder layer stub: a post-attention norm and, given a gate, a MoE block with it."""
    layer = types.SimpleNamespace(
        post_attention_layernorm=RMSNorm(hidden_size=HIDDEN, eps=1e-6, dtype=torch.bfloat16),
        pre_feed_forward_fusion=fusion,
    )
    if gate is not None:
        layer.block_sparse_moe = types.SimpleNamespace(gate=gate)
    return layer


@pytest.fixture
def flashinfer_norms(monkeypatch):
    monkeypatch.setattr(router_operand, "flashinfer_norm_decline_reason", lambda norm: None)


def test_install_wraps_the_norm_and_gate_of_each_split_gate_moe_layer(flashinfer_norms):
    layers = [_layer(), _layer(_gate()), _layer(_gate()), _layer(_gate(split=False))]
    assert router_operand.install(layers) == 2
    assert [router_operand.is_site(layer) for layer in layers] == [False, True, True, False]
    assert router_operand.install(layers) == 2
    norm, gate = layers[1].post_attention_layernorm, layers[1].block_sparse_moe.gate
    assert norm.forward.__func__ is router_operand._norm_forward
    assert gate.forward.__func__ is router_operand._gate_forward


@pytest.mark.parametrize(
    "build, reason",
    [
        pytest.param(lambda: _layer(), "no MiniMax-M3 MoE gate", id="dense"),
        pytest.param(
            lambda: _layer(types.SimpleNamespace()), "no MiniMax-M3 MoE gate", id="foreign-gate"
        ),
        pytest.param(
            lambda: _layer(_gate(split=False)),
            "the gate's weight is not split into 3 BF16 terms",
            id="fp32-gate",
        ),
        pytest.param(
            lambda: _layer(_gate(), fusion=True),
            "the post-attention norm is fused with an AllReduce",
            id="allreduce",
        ),
    ],
)
def test_install_declines(flashinfer_norms, build, reason):
    layer = build()
    assert router_operand.site_decline_reason(layer) == reason
    assert router_operand.install([layer]) == 0 and not router_operand.is_site(layer)


def test_install_declines_a_gate_split_into_other_terms(flashinfer_norms):
    gate = _gate()
    gate.split_weight = gate.split_weight[:, : 2 * HIDDEN].contiguous()
    assert router_operand.site_decline_reason(_layer(gate)) == (
        "the gate's weight is not split into 3 BF16 terms"
    )


def test_install_declines_a_norm_the_tail_declines(monkeypatch):
    monkeypatch.setattr(
        router_operand, "flashinfer_norm_decline_reason", lambda norm: "FlashInfer is not available"
    )
    assert router_operand.site_decline_reason(_layer(_gate())) == "FlashInfer is not available"


def test_the_norm_check_is_the_tails():
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
        tail,
    )

    assert router_operand.flashinfer_norm_decline_reason is tail.flashinfer_norm_decline_reason
    layer = types.SimpleNamespace(next_layer_layernorm=None, post_feed_forward_fusion=False)
    assert tail.boundary_norm_decline_reason(layer) == "no next-layer norm"
    assert tail.flashinfer_norm_decline_reason(None) == "norm NoneType"


def test_install_declines_mismatched_widths(flashinfer_norms):
    layer = _layer(_gate())
    layer.post_attention_layernorm = RMSNorm(hidden_size=2 * HIDDEN, eps=1e-6, dtype=torch.bfloat16)
    assert router_operand.site_decline_reason(layer) == (
        "the post-attention norm and the gate disagree on the hidden size"
    )


def test_install_leaves_a_forward_something_else_replaced(flashinfer_norms):
    norm_taken, gate_taken = _layer(_gate()), _layer(_gate())
    norm_taken.post_attention_layernorm.forward = lambda *a, **kw: None
    gate_taken.block_sparse_moe.gate.forward = lambda *a, **kw: None
    assert (
        router_operand.site_decline_reason(norm_taken) == "the norm's forward is already replaced"
    )
    assert (
        router_operand.site_decline_reason(gate_taken) == "the gate's forward is already replaced"
    )
    assert router_operand.install([norm_taken, gate_taken]) == 0


def test_install_wraps_nothing_when_off(monkeypatch, flashinfer_norms):
    monkeypatch.setattr(router_operand, "DEP_ROUTER_OPERAND", False)
    layer = _layer(_gate())
    assert router_operand.install([layer]) == 0
    assert "forward" not in layer.post_attention_layernorm.__dict__
    assert "forward" not in layer.block_sparse_moe.gate.__dict__


# --- the wrapped norm ------------------------------------------------------------------------------------------------


@pytest.fixture
def site(monkeypatch, flashinfer_norms):
    """A wrapped post-attention norm whose kernel and stock forward record their calls."""
    layer = _layer(_gate())
    assert router_operand.install([layer]) == 1
    norm = layer.post_attention_layernorm
    norm.kernel_calls, norm.stock_calls = [], []
    monkeypatch.setattr(
        router_operand,
        "kernel",
        lambda: lambda *args, **kwargs: norm.kernel_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        RMSNorm,
        "forward",
        lambda self, h, r=..., *, return_norm_out=False: self.stock_calls.append((h, r)) or (h, r),
    )
    router_operand.reserve(CPU, HIDDEN)
    return norm


def test_a_served_call_runs_the_kernel_and_publishes_its_output(site):
    x, residual = _rows(88), _rows(88)
    out, res = site(x, residual)
    assert out is x and res is residual
    ((args, kwargs),) = site.kernel_calls
    assert args[0] is x and args[1] is residual and args[2] is site.weight
    assert args[3] == site.variance_epsilon
    expected = router_operand.operand(x)
    assert args[4].data_ptr() == expected.data_ptr() and args[4].shape == (88, 3 * HIDDEN)
    assert set(kwargs) == {"enable_pdl"} and site.stock_calls == []
    assert router_operand.SLOT.take(x) is args[4]


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
    assert router_operand.SLOT.pending is None


def test_a_norm_that_gained_an_nvfp4_scale_is_the_stock_norm(site):
    site.nvfp4_scale = torch.ones(1)
    site(_rows(8), _rows(8))
    assert site.kernel_calls == [] and len(site.stock_calls) == 1


def test_under_torch_compile_the_norm_is_stock(site, monkeypatch):
    monkeypatch.setattr(router_operand, "is_torch_compiling", lambda: True)
    site(_rows(8), _rows(8))
    assert site.kernel_calls == [] and len(site.stock_calls) == 1


# --- the wrapped gate ------------------------------------------------------------------------------------------------


@pytest.fixture
def gate(monkeypatch, flashinfer_norms):
    """A wrapped gate whose GEMM and stock forward record their calls."""
    layer = _layer(_gate())
    assert router_operand.install([layer]) == 1
    gate = layer.block_sparse_moe.gate
    gate.gemm_calls, gate.stock_calls = [], []
    monkeypatch.setattr(
        router_operand,
        "_router_gemm",
        lambda operand, weight: gate.gemm_calls.append((operand, weight)) or "ours",
    )
    monkeypatch.setattr(
        MiniMaxM3Gate, "forward", lambda self, h: self.stock_calls.append(h) or "stock"
    )
    router_operand.reserve(CPU, HIDDEN)
    return gate


def _publish(rows):
    x = _rows(rows)
    operand = router_operand.operand(x)
    router_operand.SLOT.publish(x, operand)
    return x, operand


def test_the_gate_runs_the_gemm_on_the_published_copies(gate):
    x, operand = _publish(88)
    assert gate(x) == "ours"
    ((taken, weight),) = gate.gemm_calls
    assert taken is operand and weight is gate.split_weight
    assert gate.stock_calls == [] and router_operand.SLOT.pending is None


def test_the_gate_on_another_input_is_stock_and_clears_the_slot(gate):
    x, _ = _publish(88)
    other = x.clone()
    assert gate(other) == "stock"
    assert gate.stock_calls == [other] and gate.gemm_calls == []
    assert router_operand.SLOT.pending is None
    assert gate(x) == "stock" and gate.gemm_calls == []


def test_the_gate_without_a_published_operand_is_stock(gate):
    x = _rows(8)
    assert gate(x) == "stock" and gate.stock_calls == [x]


def test_the_gate_is_stock_where_the_stock_gate_runs_no_cat_gemm(gate):
    x, _ = _publish(8)
    gate.split_weight = None
    assert gate(x) == "stock" and gate.gemm_calls == []
    assert router_operand.SLOT.pending is None


def test_a_take_and_a_miss_are_logged_once_per_width(gate, monkeypatch):
    lines = []
    monkeypatch.setattr(router_operand, "_LOGGED", set())
    monkeypatch.setattr(router_operand.logger, "info", lines.append)
    for _ in range(2):
        x, _ = _publish(16)
        gate(x)
        # The published rows stay alive, so the input below cannot reuse their address and pass for them.
        published, _ = _publish(16)
        gate(x.clone())
    published, _ = _publish(32)
    gate(_rows(32))
    assert len(lines) == 3
    assert "DEP_ROUTER_OPERAND taken" in lines[0] and "norm's copies, 16 rows" in lines[0]
    assert (
        "DEP_ROUTER_OPERAND missed" in lines[1] and "16-row post-attention norm output" in lines[1]
    )
    assert "32-row post-attention norm output" in lines[2]


# --- the stock forwards the wrappers mirror --------------------------------------------------------------------------


def _method_md5(module, class_name, method):
    source = inspect.getsource(module)
    lines = source.splitlines(keepends=True)
    (cls,) = [
        n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == class_name
    ]
    (fn,) = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method]
    text = textwrap.dedent("".join(lines[fn.lineno - 1 : fn.end_lineno]))
    return hashlib.md5(text.encode()).hexdigest()


def test_the_stock_norm_the_kernel_replaces_is_unchanged():
    """The norm wrapper serves only the calls this forward sends to flashinfer_fused_add_rmsnorm."""
    assert _method_md5(rms_norm, "RMSNorm", "forward") == "c853f060681b7936af9d559adb87359c"


def test_the_stock_gate_the_gemm_mirrors_is_unchanged():
    """``_runs_the_cat_gemm`` and ``_router_gemm`` are this forward's ``cat`` branch; a change here must be made there
    too."""
    assert (
        _method_md5(modeling_minimaxm3, "MiniMaxM3Gate", "forward")
        == "cebc8a894d05c8f85205638c69383fff"
    )
