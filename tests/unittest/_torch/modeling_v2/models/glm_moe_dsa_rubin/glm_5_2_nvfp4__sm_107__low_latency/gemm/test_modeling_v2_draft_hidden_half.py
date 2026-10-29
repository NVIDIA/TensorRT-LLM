# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the glm_moe_dsa tp16 draft_hidden_half entry."""

import os
import zlib
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency._small_batch_draft import (  # noqa: E501
    SmallBatchDraftKernels,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.gemm.draft_hidden_half import (  # noqa: E501
    draft_hidden_half as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 107, reason="the glm_moe_dsa tp16 entries are certified on sm_107"
)


def _build(spec: dict, seed: int) -> dict:
    """The op's arguments for one cell.

    The hidden state has the scale of a residual stream (standard deviation 8); the norm weight
    sits around 1; the weight is scaled to unit-variance outputs. `out` starts as NaN, so an
    output the kernel never stores fails the comparison.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows, k, n = spec["rows"], spec["k"], spec["n"]
    x = (8.0 * torch.randn(rows, k, generator=g, device="cuda")).to(torch.bfloat16)
    norm_weight = (1.0 + 0.1 * torch.randn(k, generator=g, device="cuda")).to(torch.bfloat16)
    weight = (torch.randn(n, k, generator=g, device="cuda") * k**-0.5).to(torch.bfloat16)
    in_kernel_norm = spec["in_kernel_norm"]
    if not in_kernel_norm:
        # The fallback's input: the row(s) already normalized and rounded to BF16.
        xf = x.float()
        x = (
            xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + spec["eps"]) * norm_weight.float()
        ).to(torch.bfloat16)
    return dict(
        x=x,
        weight=weight,
        out=torch.full((rows, n), float("nan"), dtype=torch.float32, device="cuda"),
        norm_weight=norm_weight if in_kernel_norm else None,
        eps=spec["eps"] if in_kernel_norm else None,
    )


def _compare(out: torch.Tensor, ref: torch.Tensor, in_kernel_norm: bool) -> None:
    """FP32 outputs; the band depends on what can differ.

    Plain GEMM: the same BF16 operands, products exact in FP32, sums in another order (split-K
    8 over a 6144-long K against torch): about 1e-6 of a row's scale. Allowed: 2**-13 of the row
    maximum element-wise and 2**-15 relative RMS, two orders above that and orders below a lost
    or doubled K slice (about a third of the values).

    In-kernel norm: the kernel normalizes with FlashInfer's summation order and approximate
    rsqrt, the reference with torch's, so a normalized element sitting next to a BF16 rounding
    midpoint can round the other way (one BF16 ulp; a few FP32 ulps of rstd make that about one
    element in 3 * 10**4, so a row sees none or one). One such flip moves every output by at
    most 2**-7 of that element's product with its weight, about 2**-10.5 of the row maximum
    with these operands. Allowed: 2**-8 of the row maximum element-wise and 2**-10 relative
    RMS. A wrong eps, a missing norm weight or an unnormalized row is off by whole values.
    """
    if in_kernel_norm:
        assert_within_ulp(out, ref, element_ulp=1.0, rms_ulp=0.25, ulp=2.0**-8)
    else:
        assert_within_ulp(out, ref, element_ulp=1.0, rms_ulp=0.25, ulp=2.0**-13)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    x_before = args["x"].clone()
    assert op(**args) is args["out"], "the op writes in place and returns what it wrote"
    assert torch.equal(args["x"], x_before)
    _compare(args["out"], op.reference(**args), cell.spec["in_kernel_norm"])


def _flashinfer_norm_is_cute_dsl() -> bool:
    """Whether trtllm::flashinfer_rmsnorm runs FlashInfer's CuTe DSL kernel, the one the in-kernel
    norm reproduces. FlashInfer picks its CUDA kernel under FLASHINFER_USE_CUDA_NORM=1 and where
    its own device check says so; without that check the default, CuTe DSL, is assumed."""
    if os.environ.get("FLASHINFER_USE_CUDA_NORM", "0") == "1":
        return False
    try:
        from flashinfer.norm import _use_cuda_norm
    except ImportError:
        return True
    return not _use_cuda_norm(torch.device("cuda", torch.cuda.current_device()))


@pytest.mark.parametrize("n", [384, 768])
def test_in_kernel_norm_is_flashinfer_rmsnorm_bit_for_bit(n) -> None:
    """The point of the in-kernel norm: the result of FlashInfer's RMSNorm followed by this GEMM,
    bit for bit, from one kernel."""
    if not _flashinfer_norm_is_cute_dsl():
        pytest.skip("FlashInfer runs its CUDA RMSNorm kernel here, not the one reproduced")
    args = _build(dict(op.CELLS[0].spec, n=n), seed=71)
    with validating(op):
        fused = op(**args)
    normed = torch.ops.trtllm.flashinfer_rmsnorm(args["x"], args["norm_weight"], args["eps"])
    plain = op(normed, args["weight"], torch.empty_like(args["out"]), None, None)
    assert torch.equal(fused, plain)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_cuda_graph_capture_and_replay(cell) -> None:
    """The draft loop captures the op on the auxiliary stream: it must replay on new contents of
    the same buffers. Warm-up compiles the kernel outside the capture."""
    spec = cell.spec
    args = _build(spec, seed=72)

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        op(**args)
    torch.cuda.current_stream().wait_stream(warmup)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(**args)

    args["x"].copy_(_build(spec, seed=73)["x"])
    args["out"].fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    _compare(args["out"], op.reference(**args), in_kernel_norm=spec["in_kernel_norm"])


@pytest.mark.parametrize("rows", [1, 2])
def test_tp8_head_auxiliary_half_join_replays_new_hidden_states(rows):
    """The production head keeps weighted hnorm and its FP32 half alive until the main-stream join."""
    args = _build(dict(op.CELLS[0].spec, n=768, rows=rows), seed=91)
    helper = object.__new__(SmallBatchDraftKernels)
    helper.use_table = True
    helper.hidden_rows = args["weight"]
    helper.hnorm_weight = args["norm_weight"]
    helper.half_start, helper.half_done = torch.cuda.Event(), torch.cuda.Event()

    class Norm:
        variance_epsilon = args["eps"]

        def __call__(self, hidden):
            return torch.ops.trtllm.flashinfer_rmsnorm(
                hidden, helper.hnorm_weight, self.variance_epsilon
            )

    helper.layer = SimpleNamespace(
        mlp=SimpleNamespace(aux_stream=torch.cuda.Stream()), hnorm=Norm()
    )
    helper.lm_head = object()
    helper.decode = SimpleNamespace(_lm_head_shard=lambda lm, hidden: hidden.square())

    def run():
        logits, half = helper.head(args["x"], next_step=True)
        torch.cuda.current_stream().wait_event(helper.half_done)
        return logits, half.clone()

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logits, result = run()
    for seed in (92, 93):
        args["x"].copy_(_build(dict(op.CELLS[0].spec, n=768, rows=rows), seed=seed)["x"])
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(logits, args["x"].square(), rtol=0, atol=0)
        if rows == 1:
            reference = op.reference(**args)
        else:
            normed = helper.layer.hnorm(args["x"])
            reference = op.reference(normed, args["weight"], result, None, None)
        _compare(result, reference, in_kernel_norm=rows == 1)


def test_guard_refuses_eps_without_the_norm_weight() -> None:
    """eps alone would be ignored: the raw row would go into the GEMM."""
    args = _build(op.CELLS[0].spec, seed=74)
    args["norm_weight"] = None
    with pytest.raises(AssertionError, match="pass norm_weight and eps together"):
        with validating(op):
            op(**args)


def test_guard_refuses_an_out_over_the_weight() -> None:
    """The owner CTAs store while other clusters still stream the weight: silently wrong."""
    args = _build(op.CELLS[0].spec, seed=75)
    rows, n = args["out"].shape
    args["out"] = args["weight"].view(torch.float32)[:rows, :n]
    with pytest.raises(AssertionError, match="out must not share memory with weight"):
        with validating(op):
            op(**args)
