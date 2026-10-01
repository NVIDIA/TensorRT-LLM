# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the BF16 gate/up GEMM with SwiGLU in the epilogue (QuACK, SM100 family)."""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules import gate_up_swiglu_quack as fused
from tensorrt_llm._torch.modules import gated_mlp as gated_mlp_module
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.swiglu import swiglu

requires_kernel = pytest.mark.skipif(
    not (torch.cuda.is_available() and fused.gate_up_swiglu_quack_available()),
    reason="needs an SM100-family GPU with QuACK installed",
)


def _reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The unfused BF16 path: gate/up GEMM into BF16, then the SwiGLU kernel."""
    return swiglu(F.linear(x, weight))


def _oracle(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    gate, up = F.linear(x.double(), weight.double()).chunk(2, dim=-1)
    return F.silu(gate) * up


@requires_kernel
@pytest.mark.parametrize(
    "tokens,hidden,intermediate", [(1, 256, 512), (257, 512, 1024), (4096, 5376, 14336)]
)
def test_fused_epilogue_matches_unfused_path(tokens, hidden, intermediate):
    torch.manual_seed(0)
    x = torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16)
    weight = (
        torch.randn((2 * intermediate, hidden), device="cuda", dtype=torch.bfloat16) * hidden**-0.5
    )
    actual = fused.gate_up_swiglu_quack_bf16(x, weight)
    expected = _reference(x, weight)
    oracle = _oracle(x, weight)
    assert actual.shape == expected.shape == (tokens, intermediate)
    assert actual.dtype == torch.bfloat16 and actual.is_contiguous()
    # One rounding instead of two: the fused result must be at least as close to the FP64 oracle
    # as the unfused path, and close to the unfused path at the BF16 scale of the output.
    err_actual = (actual.double() - oracle).abs()
    err_expected = (expected.double() - oracle).abs()
    assert err_actual.norm() <= err_expected.norm() * 1.05
    assert (actual.float() - expected.float()).abs().max() <= expected.float().abs().max() * 2**-6


@requires_kernel
def test_fused_epilogue_rejects_invalid_inputs():
    x = torch.randn((4, 64), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((128, 64), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        fused.gate_up_swiglu_quack_bf16(x.float(), weight)
    with pytest.raises(ValueError):
        fused.gate_up_swiglu_quack_bf16(x, weight[:127])
    with pytest.raises(ValueError):
        fused.gate_up_swiglu_quack_bf16(x[None], weight)


@requires_kernel
def test_gated_mlp_dispatches_bf16_epilogue(monkeypatch):
    torch.manual_seed(1)
    mlp = GatedMLP(
        hidden_size=256,
        intermediate_size=512,
        bias=False,
        dtype=torch.bfloat16,
        use_quack_swiglu_epilogue=True,
    ).cuda()
    for p in mlp.parameters():
        p.data.normal_(std=0.05)
    # GatedMLP consumes 2-D [tokens, hidden] activations (the SwiGLU kernel requires it).
    x = torch.randn((133, 256), device="cuda", dtype=torch.bfloat16)
    assert mlp._can_fuse_gate_up_swiglu_bf16()
    calls = []
    op = gated_mlp_module.gate_up_swiglu_quack_bf16

    def tracked(*args, **kwargs):
        calls.append(True)
        return op(*args, **kwargs)

    monkeypatch.setattr(gated_mlp_module, "gate_up_swiglu_quack_bf16", tracked)
    with torch.inference_mode():
        fused_out = mlp(x)
        monkeypatch.setattr(gated_mlp_module, "gate_up_swiglu_quack_available", lambda: False)
        assert not mlp._can_fuse_gate_up_swiglu_bf16()
        unfused_out = mlp(x)
    assert calls == [True]
    assert fused_out.shape == unfused_out.shape == x.shape
    assert (
        fused_out.float() - unfused_out.float()
    ).abs().max() <= unfused_out.float().abs().max() * 2**-6


def test_gated_mlp_bf16_epilogue_is_opt_in_and_excludes_bias_quant_and_tp(monkeypatch):
    monkeypatch.setattr(gated_mlp_module, "gate_up_swiglu_quack_available", lambda: True)
    mlp = GatedMLP(hidden_size=16, intermediate_size=128, bias=False, dtype=torch.bfloat16)
    assert (
        not mlp._can_fuse_gate_up_swiglu_bf16()
    )  # default off: every other model keeps cuBLAS + SwiGLU
    mlp = GatedMLP(
        hidden_size=16,
        intermediate_size=128,
        bias=True,
        dtype=torch.bfloat16,
        use_quack_swiglu_epilogue=True,
    )
    assert not mlp._can_fuse_gate_up_swiglu_bf16()  # bias
    mlp = GatedMLP(
        hidden_size=16,
        intermediate_size=128,
        bias=False,
        dtype=torch.bfloat16,
        use_quack_swiglu_epilogue=True,
    )
    assert mlp._can_fuse_gate_up_swiglu_bf16()
    monkeypatch.setattr(mlp.gate_up_proj, "tp_size", 2)
    assert not mlp._can_fuse_gate_up_swiglu_bf16()  # tensor parallel
    monkeypatch.setattr(mlp.gate_up_proj, "tp_size", 1)
    mlp.swiglu_limit = 7.0
    assert not mlp._can_fuse_gate_up_swiglu_bf16()  # clamped SwiGLU stays on the Triton kernel
    mlp.swiglu_limit = None
    mlp = GatedMLP(
        hidden_size=16,
        intermediate_size=96,
        bias=False,
        dtype=torch.bfloat16,
        use_quack_swiglu_epilogue=True,
    )
    assert not mlp._can_fuse_gate_up_swiglu_bf16()  # intermediate size not a multiple of 128
