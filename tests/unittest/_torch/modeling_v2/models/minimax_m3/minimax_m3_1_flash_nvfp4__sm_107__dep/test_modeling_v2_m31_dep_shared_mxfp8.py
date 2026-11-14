# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_SHARED_MXFP8 on CPU tensors and stub layers.

The install gives each MoE layer whose shared expert is the MXFP8 SwiGLU GatedMLP its two hand-offs: the gate_up_proj
one where the post-attention norm is a DEP_ROUTER_OPERAND site, whose add + RMSNorm then also writes the normalized
rows' MXFP8 form, and the down_proj one, whose SwiGLU then also writes its output's. The kernels and GEMMs are
replaced by recorders; their bitwise checks are the GPU tests. CPU only.
"""

import ast
import hashlib
import inspect
import textwrap
import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch.modules import gated_mlp
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.linear import MXFP8LinearMethod
from tensorrt_llm._torch.modules.rms_norm import RMSNorm

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
        m31_switches,
        merge_mxfp8,
        norm_mxfp8,
        router_operand,
        shared_mxfp8,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

pytestmark = pytest.mark.cpu_only

CPU = torch.device("cpu")
HIDDEN = 256
WIDTH = 128


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(shared_mxfp8, "_BUFFERS", {})
    monkeypatch.setattr(shared_mxfp8, "_WIDTHS", set())
    monkeypatch.setattr(shared_mxfp8, "DEP_SHARED_MXFP8", True)
    monkeypatch.setattr(shared_mxfp8, "DEP_SHARED_GU_MXFP8", True)
    monkeypatch.setattr(shared_mxfp8, "DEP_SHARED_DOWN_MXFP8", True)
    monkeypatch.setattr(router_operand, "_BUFFERS", {})
    monkeypatch.setattr(router_operand, "DEP_ROUTER_OPERAND", True)
    for slot in (shared_mxfp8.GATE_UP_SLOT, shared_mxfp8.DOWN_SLOT, router_operand.SLOT):
        slot.clear()
    yield
    for slot in (shared_mxfp8.GATE_UP_SLOT, shared_mxfp8.DOWN_SLOT, router_operand.SLOT):
        slot.clear()


def test_the_switches_are_the_dep_shared_mxfp8_items(monkeypatch):
    monkeypatch.undo()
    on = not m31_switches.m31_disabled("DEP_SHARED_MXFP8")
    assert shared_mxfp8.DEP_SHARED_MXFP8 == on
    assert shared_mxfp8.DEP_SHARED_GU_MXFP8 == (
        on and not m31_switches.m31_disabled("DEP_SHARED_GU_MXFP8")
    )
    assert shared_mxfp8.DEP_SHARED_DOWN_MXFP8 == (
        on and not m31_switches.m31_disabled("DEP_SHARED_DOWN_MXFP8")
    )


def test_each_hand_off_has_its_own_slot():
    slots = [
        shared_mxfp8.GATE_UP_SLOT,
        shared_mxfp8.DOWN_SLOT,
        merge_mxfp8.SLOT,
        norm_mxfp8.SLOT,
    ]
    assert len({id(slot) for slot in slots}) == len(slots)
    assert shared_mxfp8.DepSharedGateUpMxfp8LinearMethod.slot is shared_mxfp8.GATE_UP_SLOT
    assert shared_mxfp8.DepSharedDownMxfp8LinearMethod.slot is shared_mxfp8.DOWN_SLOT
    for method in (
        shared_mxfp8.DepSharedGateUpMxfp8LinearMethod,
        shared_mxfp8.DepSharedDownMxfp8LinearMethod,
    ):
        assert issubclass(method, merge_mxfp8.DepMergeMxfp8LinearMethod)


def test_reserve_allocates_each_installed_width_once():
    assert shared_mxfp8.reserve(CPU) == 0
    shared_mxfp8._WIDTHS.update((HIDDEN, WIDTH))
    reserved = shared_mxfp8.reserve(CPU)
    expected = sum(448 * w + merge_mxfp8.scale_bytes(448, w) for w in (HIDDEN, WIDTH))
    assert reserved == expected
    full = shared_mxfp8._BUFFERS[(CPU, HIDDEN)]
    assert shared_mxfp8.reserve(CPU) == reserved and shared_mxfp8._BUFFERS[(CPU, HIDDEN)] is full
    assert bool((full[1] == 0).all())
    for rows in (1, 44, 129, 448):
        for width in (HIDDEN, WIDTH):
            q8, sf = shared_mxfp8.buffers(rows, width, CPU)
            assert (
                q8.shape == (rows, width) and q8.dtype == torch.float8_e4m3fn and q8.is_contiguous()
            )
            assert sf.numel() == merge_mxfp8.scale_bytes(rows, width)
    assert shared_mxfp8.buffers(0, HIDDEN, CPU) is None
    assert shared_mxfp8.buffers(449, HIDDEN, CPU) is None
    assert shared_mxfp8.buffers(8, 2 * HIDDEN, CPU) is None


def test_the_swiglu_constants_are_the_ones_silu_and_mul_applies():
    mlp = types.SimpleNamespace(swiglu_alpha=1.702, swiglu_beta=1.0, swiglu_limit=7.0)
    assert shared_mxfp8.swiglu_constants(mlp) == (1.702, 1.0, 7.0)
    mlp = types.SimpleNamespace(swiglu_alpha=None, swiglu_beta=None, swiglu_limit=None)
    assert shared_mxfp8.swiglu_constants(mlp) == (1.0, 0.0, None)
    mlp = types.SimpleNamespace(swiglu_alpha=None, swiglu_beta=None, swiglu_limit=0.0)
    assert shared_mxfp8.swiglu_constants(mlp) == (1.0, 0.0, None)


# --- install ---------------------------------------------------------------------------------------------------------


def _shared_expert():
    """The M3.1 shared expert as MiniMaxM3MoE builds it (SwiGLU-OAI, MXFP8), at test widths."""
    from transformers.configuration_utils import PretrainedConfig

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_minimaxm3 import _build_swiglu_oai_dense_mlp
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

    config = PretrainedConfig()
    config.hidden_size, config.intermediate_size = HIDDEN, WIDTH
    config.swiglu_alpha, config.swiglu_limit = 1.702, 7.0
    config.torch_dtype = torch.bfloat16
    model_config = ModelConfig(
        pretrained_config=config,
        mapping=Mapping(world_size=1, tp_size=1),
        quant_config=QuantConfig(quant_algo=QuantAlgo.MXFP8),
    )
    return _build_swiglu_oai_dense_mlp(
        model_config=model_config, intermediate_size=WIDTH, layer_idx=0, is_shared_expert=True
    )


@pytest.fixture(scope="module")
def built():
    mlp = _shared_expert()
    # QuantAlgo.MXFP8 always selects MXFP8LinearMethod (linear.get_quant_method): another method is a builder bug.
    assert type(mlp.gate_up_proj.quant_method) is MXFP8LinearMethod, type(
        mlp.gate_up_proj.quant_method
    )
    return mlp


def _layer(mlp):
    """A MoE decoder layer stub: a post-attention norm and a MoE block with the shared expert ``mlp``."""
    return types.SimpleNamespace(
        post_attention_layernorm=RMSNorm(hidden_size=HIDDEN, eps=1e-6, dtype=torch.bfloat16),
        block_sparse_moe=types.SimpleNamespace(shared_experts=mlp),
        pre_feed_forward_fusion=False,
    )


def _fresh_expert(built):
    return _shared_expert()


def test_the_m3_shared_expert_takes_both_hand_offs(built):
    assert shared_mxfp8.shared_decline_reason(built) is None


def test_install_gives_the_gate_up_hand_off_only_at_router_sites(built):
    site, other = _layer(_fresh_expert(built)), _layer(_fresh_expert(built))
    dense = types.SimpleNamespace()
    assert shared_mxfp8.install([site, other, dense], lambda layer: layer is site) == (1, 2)
    for layer in (site, other):
        mlp = layer.block_sparse_moe.shared_experts
        assert mlp.forward.__func__ is shared_mxfp8._shared_forward
        assert type(mlp.down_proj.quant_method) is shared_mxfp8.DepSharedDownMxfp8LinearMethod
    site_mlp, other_mlp = (layer.block_sparse_moe.shared_experts for layer in (site, other))
    assert type(site_mlp.gate_up_proj.quant_method) is shared_mxfp8.DepSharedGateUpMxfp8LinearMethod
    assert type(other_mlp.gate_up_proj.quant_method) is MXFP8LinearMethod
    assert site.post_attention_layernorm.__dict__.get(shared_mxfp8._GATE_UP_NORM) is True
    assert shared_mxfp8._GATE_UP_NORM not in other.post_attention_layernorm.__dict__
    assert shared_mxfp8._WIDTHS == {HIDDEN, WIDTH}
    assert shared_mxfp8.install([site, other], lambda layer: layer is site) == (1, 2)


@pytest.mark.parametrize(
    "switch, counts",
    [
        ("DEP_SHARED_MXFP8", (0, 0)),
        ("DEP_SHARED_GU_MXFP8", (0, 1)),
        ("DEP_SHARED_DOWN_MXFP8", (1, 0)),
    ],
)
def test_install_keeps_switched_off_hand_offs_stock(monkeypatch, built, switch, counts):
    monkeypatch.setattr(shared_mxfp8, switch, False)
    layer = _layer(_fresh_expert(built))
    mlp = layer.block_sparse_moe.shared_experts
    assert shared_mxfp8.install([layer], lambda layer: True) == counts
    assert ("forward" in mlp.__dict__) == bool(counts[1])
    assert (type(mlp.gate_up_proj.quant_method) is MXFP8LinearMethod) == (not counts[0])


@pytest.mark.parametrize(
    "change, reason",
    [
        pytest.param(
            lambda mlp: types.SimpleNamespace(), "shared expert SimpleNamespace", id="foreign"
        ),
        pytest.param(
            lambda mlp: setattr(mlp, "split_gate_up", True) or mlp,
            "the shared expert is not one fused gate_up_proj with a SwiGLU",
            id="split",
        ),
        pytest.param(
            lambda mlp: setattr(mlp, "activation", torch.nn.functional.gelu) or mlp,
            "the shared expert is not one fused gate_up_proj with a SwiGLU",
            id="gelu",
        ),
        pytest.param(
            lambda mlp: setattr(mlp, "_can_fuse_swiglu_fp8_quant", lambda: True) or mlp,
            "the shared expert fuses its SwiGLU elsewhere",
            id="fused",
        ),
        pytest.param(
            lambda mlp: setattr(mlp.down_proj, "quant_method", object()) or mlp,
            "the shared expert's projections are not MXFP8",
            id="bf16",
        ),
    ],
)
def test_install_declines(built, change, reason):
    mlp = change(_fresh_expert(built))
    assert shared_mxfp8.shared_decline_reason(mlp) == reason
    assert shared_mxfp8.install([_layer(mlp)], lambda layer: True) == (0, 0)


def test_install_leaves_a_forward_something_else_replaced(built):
    layer = _layer(_fresh_expert(built))
    mlp = layer.block_sparse_moe.shared_experts
    mlp.forward = lambda *a, **kw: None
    assert shared_mxfp8.install([layer], lambda layer: True) == (1, 0)
    assert type(mlp.down_proj.quant_method) is MXFP8LinearMethod


# --- the post-attention norm -----------------------------------------------------------------------------------------


@pytest.fixture
def norm_site(monkeypatch, built):
    """A post-attention norm at both sites whose two kernel forms record their calls."""
    monkeypatch.setattr(router_operand, "flashinfer_norm_decline_reason", lambda norm: None)
    gate = types.SimpleNamespace()
    layer = _layer(_fresh_expert(built))
    norm = layer.post_attention_layernorm
    norm.forward = types.MethodType(router_operand._norm_forward, norm)
    layer.block_sparse_moe.gate = gate
    assert shared_mxfp8.install([layer], lambda layer: True) == (1, 1)
    norm.calls = []
    monkeypatch.setattr(
        router_operand, "kernel", lambda: lambda *a, **kw: norm.calls.append(("replicated", a, kw))
    )
    monkeypatch.setattr(
        shared_mxfp8, "norm_kernel", lambda: lambda *a, **kw: norm.calls.append(("mxfp8", a, kw))
    )
    router_operand.reserve(CPU, HIDDEN)
    shared_mxfp8.reserve(CPU)
    return norm


def _rows(rows, width=HIDDEN):
    return torch.zeros(rows, width, dtype=torch.bfloat16)


def test_the_norm_writes_and_publishes_the_gate_up_hand_off(norm_site):
    x, residual = _rows(44), _rows(44)
    assert norm_site(x, residual) == (x, residual)
    ((form, args, kwargs),) = norm_site.calls
    assert form == "mxfp8" and set(kwargs) == {"enable_pdl"}
    assert args[0] is x and args[1] is residual and args[2] is norm_site.weight
    q8, sf = shared_mxfp8.buffers(44, HIDDEN, CPU)
    assert args[5].data_ptr() == q8.data_ptr() and args[5].shape == (44, HIDDEN)
    assert args[6].data_ptr() == sf.data_ptr() and args[6].numel() == sf.numel()
    assert router_operand.SLOT.take(x) is args[4]
    assert shared_mxfp8.GATE_UP_SLOT.take(x) == (args[5], args[6])


def test_a_norm_without_the_hand_off_runs_the_replicated_form(norm_site):
    del norm_site.__dict__[shared_mxfp8._GATE_UP_NORM]
    x, residual = _rows(44), _rows(44)
    norm_site(x, residual)
    ((form, args, _),) = norm_site.calls
    assert form == "replicated" and len(args) == 5
    assert shared_mxfp8.GATE_UP_SLOT.pending is None and router_operand.SLOT.take(x) is args[4]


def test_rows_past_the_buffers_publish_no_hand_off(norm_site, monkeypatch):
    monkeypatch.setattr(shared_mxfp8, "buffers", lambda rows, width, device: None)
    x, residual = _rows(44), _rows(44)
    norm_site(x, residual)
    ((form, _, _),) = norm_site.calls
    assert form == "replicated" and shared_mxfp8.GATE_UP_SLOT.pending is None


# --- the shared expert's forward -------------------------------------------------------------------------------------


@pytest.fixture
def expert(monkeypatch, built):
    """A wrapped shared expert whose projections, SwiGLU kernel and stock activation record their calls."""
    layer = _layer(_fresh_expert(built))
    assert shared_mxfp8.install([layer], lambda layer: False) == (0, 1)
    mlp = layer.block_sparse_moe.shared_experts
    shared_mxfp8.reserve(CPU)
    mlp.calls = []
    mlp.gate_up_proj.forward = lambda x: mlp.calls.append(("gate_up", x)) or torch.zeros(
        x.shape[0], 2 * WIDTH, dtype=torch.bfloat16
    )
    mlp.down_proj.forward = lambda h, **kw: mlp.calls.append(("down", h, kw)) or "out"
    monkeypatch.setattr(
        shared_mxfp8, "swiglu_kernel", lambda: lambda *a, **kw: mlp.calls.append(("swiglu", a, kw))
    )
    monkeypatch.setattr(
        GatedMLP,
        "_apply_activation",
        lambda self, h, **kw: mlp.calls.append(("stock activation", h)) or h[:, :WIDTH],
    )
    return mlp


def test_the_swiglu_writes_and_publishes_the_down_hand_off(expert):
    x = _rows(44)
    assert expert(x) == "out"
    (gate_up, swiglu, down) = expert.calls
    assert gate_up == ("gate_up", x)
    h1, h2, q8, sf = swiglu[1]
    assert h1.shape == (44, 2 * WIDTH) and h2.shape == (44, WIDTH) and h2.dtype == torch.bfloat16
    assert swiglu[2] == {
        "alpha": 1.702,
        "beta": 1.0,
        "limit": 7.0,
        "enable_pdl": swiglu[2]["enable_pdl"],
    }
    assert down[1] is h2 and down[2] == {"all_reduce_params": None, "layer_idx": expert.layer_idx}
    assert shared_mxfp8.DOWN_SLOT.take(h2) == (q8, sf)
    assert q8.shape == (44, WIDTH) and sf.numel() == merge_mxfp8.scale_bytes(44, WIDTH)


def test_rows_past_the_buffers_run_the_stock_swiglu(expert):
    x = _rows(449)
    assert expert(x) == "out"
    assert [call[0] for call in expert.calls] == ["gate_up", "stock activation", "down"]
    assert shared_mxfp8.DOWN_SLOT.pending is None


def test_a_lora_call_is_the_stock_forward(expert, monkeypatch):
    seen = []
    monkeypatch.setattr(GatedMLP, "forward", lambda self, *a, **kw: seen.append((a, kw)) or "stock")
    assert expert(_rows(8), lora_params={"x": 1}) == "stock"
    assert len(seen) == 1 and expert.calls == []


def test_under_torch_compile_the_swiglu_is_stock(expert, monkeypatch):
    monkeypatch.setattr(shared_mxfp8, "is_torch_compiling", lambda: True)
    expert(_rows(8))
    assert [call[0] for call in expert.calls] == ["gate_up", "stock activation", "down"]


# --- the stock forward the wrapper mirrors ---------------------------------------------------------------------------


def _method_md5(module, class_name, method):
    source = inspect.getsource(module)
    lines = source.splitlines(keepends=True)
    (cls,) = [
        n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == class_name
    ]
    (fn,) = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method]
    text = textwrap.dedent("".join(lines[fn.lineno - 1 : fn.end_lineno]))
    return hashlib.md5(text.encode()).hexdigest()


def test_the_stock_forward_the_wrapper_mirrors_is_unchanged():
    """``_shared_forward`` is this forward's unfused branch (gate_up_proj, ``_apply_activation``, down_proj); a change
    here must be made there too."""
    assert _method_md5(gated_mlp, "GatedMLP", "forward") == "a9eb05defbe85dda73308c70fc55caa2"
