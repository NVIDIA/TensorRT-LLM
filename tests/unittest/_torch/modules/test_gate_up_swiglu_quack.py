# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the BF16 gate/up GEMM with SwiGLU in the epilogue (QuACK, SM100 family)."""

import sys

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules import gate_up_swiglu_quack as fused
from tensorrt_llm._torch.modules import gated_mlp as gated_mlp_module
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.swiglu import swiglu
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo


def _kernel_available_or_fail() -> bool:
    """Skip on unsupported hardware; fail on SM100/SM103 if QuACK (a pinned dependency) is missing.

    Skips, with the reason, when QuACK imports but its kernel does not build against the installed
    CUTLASS DSL: the op then computes the unfused result, which these tests are not about.
    """
    if not torch.cuda.is_available() or get_sm_version() not in (100, 103):
        return False
    if fused._quack_gemm_act() is None:
        pytest.fail(
            "SM100-family GPU but QuACK gemm_act is unavailable; the pinned quack-kernels dependency is broken"
        )
    if fused._kernel_state["ok"]:
        x = torch.randn((16, 64), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn((128, 64), device="cuda", dtype=torch.bfloat16)
        fused.gate_up_swiglu_quack_bf16(x, weight)
    if not fused._kernel_state["ok"]:
        pytest.skip("QuACK gemm_act does not run against the installed CUTLASS DSL (see warning)")
    return True


requires_kernel = pytest.mark.skipif(
    not (torch.cuda.is_available() and get_sm_version() in (100, 103)),
    reason="needs an SM100/SM103 GPU",
)
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The unfused BF16 path: gate/up GEMM into BF16, then the SwiGLU kernel."""
    return swiglu(F.linear(x, weight))


def _oracle(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    gate, up = F.linear(x.double(), weight.double()).chunk(2, dim=-1)
    return F.silu(gate) * up


def _make_mlp(hidden: int = 256, intermediate: int = 512, seed: int = 0, **kwargs) -> GatedMLP:
    torch.manual_seed(seed)
    mlp = GatedMLP(
        hidden_size=hidden,
        intermediate_size=intermediate,
        bias=False,
        dtype=torch.bfloat16,
        fuse_bf16_gate_up_swiglu=True,
        **kwargs,
    ).cuda()
    for p in mlp.parameters():
        p.data.normal_(std=0.05)
    return mlp


@requires_kernel
@pytest.mark.parametrize(
    "tokens,hidden,intermediate",
    [
        (1, 256, 512),
        (257, 512, 1024),
        (133, 256, 200),  # 2I is not a multiple of the 256-wide default tile
        (257, 264, 1032),  # K and I multiples of 8 only
        (4096, 5376, 14336),  # MiniMax-H3
    ],
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
@pytest.mark.parametrize(
    "case",
    ["fp32 x", "odd 2I", "3-D x", "K mismatch", "K not a multiple of 8", "I not a multiple of 8"],
)
def test_fused_epilogue_rejects_invalid_inputs(case):
    """The op enforces what the dispatch predicate enforces: a direct caller gets an error, not a kernel fault."""
    _kernel_available_or_fail()
    x = torch.randn((4, 64), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((128, 64), device="cuda", dtype=torch.bfloat16)
    bad_x, bad_w = {
        "fp32 x": (x.float(), weight),
        "odd 2I": (x, weight[:127]),
        "3-D x": (x[None], weight),
        "K mismatch": (x[:, :56], weight),
        "K not a multiple of 8": (x[:, :60].contiguous(), weight[:, :60].contiguous()),
        "I not a multiple of 8": (x, weight[:100]),
    }[case]
    assert fused.gate_up_swiglu_quack_bf16(x, weight).shape == (4, 64)
    with pytest.raises(ValueError, match="gate_up_swiglu_quack_bf16 needs BF16 x"):
        fused.gate_up_swiglu_quack_bf16(bad_x, bad_w)


@requires_kernel
def test_gated_mlp_dispatches_bf16_epilogue(monkeypatch):
    _kernel_available_or_fail()
    mlp = _make_mlp(seed=1)
    x = torch.randn((133, 256), device="cuda", dtype=torch.bfloat16)
    calls = []
    op = gated_mlp_module.gate_up_swiglu_quack_bf16

    def tracked(*args, **kwargs):
        calls.append(True)
        return op(*args, **kwargs)

    monkeypatch.setattr(gated_mlp_module, "gate_up_swiglu_quack_bf16", tracked)
    with torch.inference_mode():
        assert mlp._can_fuse_gate_up_swiglu_bf16(x)
        fused_out = mlp(x)
        monkeypatch.setattr(mlp, "fuse_bf16_gate_up_swiglu", False)
        assert not mlp._can_fuse_gate_up_swiglu_bf16(x)
        unfused_out = mlp(x)
    assert calls == [True]
    assert fused_out.shape == unfused_out.shape == x.shape
    assert (
        fused_out.float() - unfused_out.float()
    ).abs().max() <= unfused_out.float().abs().max() * 2**-6


@requires_kernel
def test_gated_mlp_bf16_epilogue_flattens_higher_rank_inputs(monkeypatch):
    """A [batch, seq, hidden] activation runs the fused GEMM on its 2-D view, as the NVFP4 path does."""
    _kernel_available_or_fail()
    mlp = _make_mlp(seed=3)
    x = torch.randn((3, 37, 256), device="cuda", dtype=torch.bfloat16)
    calls = []
    op = gated_mlp_module.gate_up_swiglu_quack_bf16

    fused_outputs = []

    def tracked(*args, **kwargs):
        calls.append(tuple(args[0].shape))
        fused_outputs.append(op(*args, **kwargs))
        return fused_outputs[-1]

    monkeypatch.setattr(gated_mlp_module, "gate_up_swiglu_quack_bf16", tracked)
    with torch.inference_mode():
        assert mlp._can_fuse_gate_up_swiglu_bf16(x)
        out = mlp(x)
        flat = mlp(x.reshape(-1, 256))
    assert calls == [(111, 256), (111, 256)]
    assert torch.equal(*fused_outputs)  # the op saw the same 2-D view both times
    assert out.shape == x.shape
    torch.testing.assert_close(out.reshape(-1, 256), flat, rtol=2**-7, atol=2**-7)


@requires_cuda
def test_gated_mlp_bf16_epilogue_predicate(monkeypatch):
    """The predicate is the one place that decides; every exclusion flips it, nothing else does.

    A pure predicate test: kernel availability is patched in, the modules stay on the CPU and
    only ``x`` is on the GPU, so it runs on any CUDA device.
    """
    monkeypatch.setattr(gated_mlp_module, "gate_up_swiglu_quack_available", lambda: True)
    x = torch.zeros((4, 16), device="cuda", dtype=torch.bfloat16)

    def mlp(**kwargs):
        args = dict(hidden_size=16, intermediate_size=128, bias=False, dtype=torch.bfloat16)
        args.update(kwargs)
        return GatedMLP(**args)

    with torch.no_grad():
        # Default off: every other model keeps cuBLAS + the SwiGLU kernel.
        assert not mlp()._can_fuse_gate_up_swiglu_bf16(x)
        on = mlp(fuse_bf16_gate_up_swiglu=True)
        assert on._can_fuse_gate_up_swiglu_bf16(x)
        # The activation: CUDA BF16 of rank >= 2.
        assert on._can_fuse_gate_up_swiglu_bf16(x[None])
        assert not on._can_fuse_gate_up_swiglu_bf16(x.float())
        assert not on._can_fuse_gate_up_swiglu_bf16(x[0])
        assert not on._can_fuse_gate_up_swiglu_bf16(x.cpu())
        # The projection.
        assert not mlp(fuse_bf16_gate_up_swiglu=True, bias=True)._can_fuse_gate_up_swiglu_bf16(x)
        assert not mlp(
            fuse_bf16_gate_up_swiglu=True, split_gate_up=True
        )._can_fuse_gate_up_swiglu_bf16(x)
        # has_any_quant reads the projection's quant_config.
        monkeypatch.setattr(on.gate_up_proj, "quant_config", QuantConfig(quant_algo=QuantAlgo.FP8))
        assert not on._can_fuse_gate_up_swiglu_bf16(x)  # quantized projection
        monkeypatch.setattr(on.gate_up_proj, "quant_config", None)
        assert on._can_fuse_gate_up_swiglu_bf16(x)
        monkeypatch.setattr(on.gate_up_proj, "tp_size", 2)
        assert not on._can_fuse_gate_up_swiglu_bf16(x)  # tensor parallel
        monkeypatch.setattr(on.gate_up_proj, "tp_size", 1)
        monkeypatch.setattr(on.gate_up_proj, "use_cute_dsl_bf16_gemm", True)
        assert not on._can_fuse_gate_up_swiglu_bf16(x)  # another GEMM provider was selected
        monkeypatch.setattr(on.gate_up_proj, "use_cute_dsl_bf16_gemm", False)
        monkeypatch.setattr(on.gate_up_proj, "use_custom_cublas_mm", True)
        assert not on._can_fuse_gate_up_swiglu_bf16(x)
        monkeypatch.setattr(on.gate_up_proj, "use_custom_cublas_mm", False)
        assert on._can_fuse_gate_up_swiglu_bf16(x)
        # The activation function.
        on.swiglu_limit = 7.0
        assert not on._can_fuse_gate_up_swiglu_bf16(x)  # clamped SwiGLU stays on the Triton kernel
        on.swiglu_limit = None
        assert not mlp(
            fuse_bf16_gate_up_swiglu=True, swiglu_alpha=1.702, swiglu_beta=1.0
        )._can_fuse_gate_up_swiglu_bf16(x)
        # Shapes: K and I multiples of 8 (16-byte rows), nothing coarser.
        assert mlp(
            fuse_bf16_gate_up_swiglu=True, intermediate_size=96
        )._can_fuse_gate_up_swiglu_bf16(x)
        assert not mlp(
            fuse_bf16_gate_up_swiglu=True, intermediate_size=100
        )._can_fuse_gate_up_swiglu_bf16(x)
        assert not mlp(fuse_bf16_gate_up_swiglu=True, hidden_size=12)._can_fuse_gate_up_swiglu_bf16(
            x[:, :12]
        )
    # Grad mode: the op has no backward, so the unfused path must take the call.
    with torch.enable_grad():
        assert not on._can_fuse_gate_up_swiglu_bf16(x)


@requires_kernel
def test_gated_mlp_bf16_epilogue_compiles_fullgraph():
    """torch.compile captures the custom op as one node and matches eager."""
    _kernel_available_or_fail()
    mlp = _make_mlp(seed=4)
    x = torch.randn((133, 256), device="cuda", dtype=torch.bfloat16)
    fused_nodes = []
    unfused_nodes = []

    def backend(gm, example_inputs):
        del example_inputs
        targets = [str(node.target) for node in gm.graph.nodes if node.op == "call_function"]
        fused_nodes.append(targets.count("trtllm.gate_up_swiglu_quack_bf16.default"))
        unfused_nodes.extend(t for t in targets if "swiglu" in t and "gate_up" not in t)
        return gm.forward

    try:
        with torch.inference_mode():
            assert mlp._can_fuse_gate_up_swiglu_bf16(x)
            expected = mlp(x)
            compiled = torch.compile(mlp, backend=backend, fullgraph=True)
            actual = compiled(x)
    finally:
        torch._dynamo.reset()
    assert fused_nodes == [1]
    assert unfused_nodes == []  # no separate SwiGLU kernel left in the graph
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@requires_kernel
def test_gated_mlp_bf16_epilogue_replays_under_cuda_graph(monkeypatch):
    """The fused op must capture and replay in a CUDA graph (shared-module consumers may use graphs)."""
    _kernel_available_or_fail()
    mlp = _make_mlp(seed=2)
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


def test_availability_falls_back_when_quack_does_not_import(monkeypatch):
    """Without an importable QuACK the module reports unavailable and GatedMLP stays unfused."""
    fused._quack_gemm_act.cache_clear()
    monkeypatch.setattr(
        "tensorrt_llm._torch.cute_dsl_utils.install_cutlass_dsl_compatibility", lambda: None
    )
    monkeypatch.setitem(sys.modules, "quack.gemm_interface", None)  # import raises ImportError
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(fused, "get_sm_version", lambda: 100)
    try:
        assert fused._quack_gemm_act() is None
        assert not fused.gate_up_swiglu_quack_available()
        mlp = GatedMLP(
            hidden_size=16,
            intermediate_size=128,
            bias=False,
            dtype=torch.bfloat16,
            fuse_bf16_gate_up_swiglu=True,
        )
        assert not mlp.fuse_bf16_gate_up_swiglu
    finally:
        fused._quack_gemm_act.cache_clear()


@requires_kernel
def test_op_falls_back_when_the_kernel_raises(monkeypatch):
    """A kernel that raises when first built leaves correct results and disables itself."""
    _kernel_available_or_fail()

    def broken_gemm_act(*args, **kwargs):
        raise ValueError(
            "too many values to unpack (expected 3)"
        )  # the CUTLASS DSL drift seen in CI

    monkeypatch.setattr(fused, "_quack_gemm_act", lambda: broken_gemm_act)
    monkeypatch.setitem(fused._kernel_state, "ok", True)
    mlp = _make_mlp(hidden=64, intermediate=128)
    x = torch.randn((9, 64), device="cuda", dtype=torch.bfloat16)
    weight = mlp.gate_up_proj.weight
    with torch.inference_mode():
        assert mlp._can_fuse_gate_up_swiglu_bf16(x)
        out = fused.gate_up_swiglu_quack_bf16(x, weight)
        torch.testing.assert_close(out, fused._unfused_gate_up_swiglu(x, weight), rtol=0, atol=0)
        assert not fused._kernel_state["ok"]
        assert not fused.gate_up_swiglu_quack_available()
        assert not mlp._can_fuse_gate_up_swiglu_bf16(x)  # later calls take the native unfused path
        # The op stays callable (compiled graphs keep its node) and keeps computing the unfused result.
        assert torch.equal(fused.gate_up_swiglu_quack_bf16(x, weight), out)
        # GatedMLP now runs its native unfused path, the same as with the fusion switched off.
        disabled_out = mlp(x)
        monkeypatch.setattr(mlp, "fuse_bf16_gate_up_swiglu", False)
        assert torch.equal(disabled_out, mlp(x))
