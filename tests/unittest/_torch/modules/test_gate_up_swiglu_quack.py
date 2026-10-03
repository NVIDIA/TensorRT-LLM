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
from tensorrt_llm._utils import get_sm_version


def _kernel_available_or_fail() -> bool:
    """Skip on unsupported hardware; fail on SM100/SM103 if QuACK (a pinned dependency) is missing."""
    if not torch.cuda.is_available() or get_sm_version() not in (100, 103):
        return False
    if not fused.gate_up_swiglu_quack_available():
        pytest.fail(
            "SM100-family GPU but QuACK gemm_act is unavailable; the pinned quack-kernels dependency is broken"
        )
    return True


requires_kernel = pytest.mark.skipif(
    not (torch.cuda.is_available() and get_sm_version() in (100, 103)),
    reason="needs an SM100/SM103 GPU",
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
    _kernel_available_or_fail()
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
    _kernel_available_or_fail()
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
    _kernel_available_or_fail()
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
        monkeypatch.setattr(mlp, "use_quack_swiglu_epilogue", False)
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
    monkeypatch.setattr(type(mlp.gate_up_proj), "has_any_quant", True)
    assert not mlp._can_fuse_gate_up_swiglu_bf16()  # quantized projection
    monkeypatch.undo()
    monkeypatch.setattr(gated_mlp_module, "gate_up_swiglu_quack_available", lambda: True)
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
    mlp = GatedMLP(
        hidden_size=16,
        intermediate_size=128,
        bias=False,
        dtype=torch.bfloat16,
        use_quack_swiglu_epilogue=True,
    )
    monkeypatch.setattr(mlp.gate_up_proj, "use_cute_dsl_bf16_gemm", True, raising=False)
    assert not mlp._can_fuse_gate_up_swiglu_bf16()  # another GEMM provider was selected


@requires_kernel
def test_gated_mlp_bf16_epilogue_replays_under_cuda_graph(monkeypatch):
    """The fused op must capture and replay in a CUDA graph (shared-module consumers may use graphs)."""
    _kernel_available_or_fail()
    torch.manual_seed(2)
    mlp = GatedMLP(
        hidden_size=256,
        intermediate_size=512,
        bias=False,
        dtype=torch.bfloat16,
        use_quack_swiglu_epilogue=True,
    ).cuda()
    for p in mlp.parameters():
        p.data.normal_(std=0.05)
    calls = []
    op = gated_mlp_module.gate_up_swiglu_quack_bf16

    def tracked(*args, **kwargs):
        calls.append(True)
        return op(*args, **kwargs)

    monkeypatch.setattr(gated_mlp_module, "gate_up_swiglu_quack_bf16", tracked)
    static_x = torch.randn((64, 256), device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.inference_mode(), torch.cuda.stream(stream):
        for _ in range(3):  # warmup: JIT compile and allocator state
            mlp(static_x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        static_out = mlp(static_x)
    assert calls, "fused op was not used"
    inputs = [torch.randn((64, 256), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    with torch.inference_mode():
        for x in inputs:
            static_x.copy_(x)
            graph.replay()
            torch.cuda.synchronize()
            eager = mlp(x)
            assert torch.equal(static_out, eager)


def test_availability_falls_back_when_the_probe_fails(monkeypatch):
    """A QuACK/CUTLASS DSL mismatch shows up at kernel compile time; availability must report False."""
    import types

    fused._quack_gemm_act.cache_clear()
    broken = types.ModuleType("quack.gemm_interface")

    def gemm_act(*args, **kwargs):
        raise ValueError("too many values to unpack (expected 3)")

    broken.gemm_act = gemm_act
    monkeypatch.setitem(__import__("sys").modules, "quack.gemm_interface", broken)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(fused, "get_sm_version", lambda: 100)
    try:
        assert fused._quack_gemm_act() is None
        assert not fused.gate_up_swiglu_quack_available()
    finally:
        fused._quack_gemm_act.cache_clear()
