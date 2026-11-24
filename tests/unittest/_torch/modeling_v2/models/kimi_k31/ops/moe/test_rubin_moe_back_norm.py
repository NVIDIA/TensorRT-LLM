# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The rubin_moe_back_norm entry: every cell writes, bit for bit, what the engine's TopkReduce (BF16
combine, FP32 scores) followed by FlashInfer's RMSNorm writes, on SM107, also from a CUDA graph; the
geometry and operand guards on the CPU."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.moe import (
    rubin_moe_back_norm as kernel,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.rubin_moe_back_norm import (  # noqa: E501
    rubin_moe_back_norm,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 -- path above; fixture by name

_SM107 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
pytestmark = pytest.mark.skipif(
    torch.cuda.is_available() and not _SM107, reason="certified on SM107 only"
)
gpu = pytest.mark.skipif(not _SM107, reason="needs an SM107 GPU")

HIDDEN, TOP_K, EPS = 3584, 16, 1e-5


def _inputs(t, seed, mode="normal", device="cuda"):
    """A combine plane of expert-output-like BF16 rows and softmax-like FP32 routing weights."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    plane = torch.randn(t, TOP_K, HIDDEN, generator=g) * 0.05
    if mode == "heavy":
        plane = plane * torch.exp(torch.randn(t, TOP_K, HIDDEN, generator=g) * 1.5)
    s = torch.rand(t, TOP_K, generator=g) + 0.05
    scores = s / s.sum(-1, keepdim=True) * 2.5
    if mode == "zero_scores":
        scores[::3] = 0.0
    weight = (torch.randn(HIDDEN, generator=g) * 0.2 + 1.0).to(torch.bfloat16)
    return (
        plane.to(torch.bfloat16).to(device),
        scores.float().to(device),
        weight.to(device),
    )


def _today(plane, scores, weight, rows):
    """What a generation step runs today: the engine's TopkReduce over the plane's rows (BF16
    combine, FP32 scores), then FlashInfer's RMSNorm."""
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
        flashinfer_rmsnorm,
    )
    from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.kernel_src.blackwell.inference.mega.topk_reduce import (  # noqa: E501
        TopkReduce,
    )
    from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.quant_def import CombineFormat

    reduced = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=plane.device)
    args = (
        from_dlpack(plane[:rows], assumed_align=16),
        None,
        from_dlpack(reduced, assumed_align=16),
        from_dlpack(scores[:rows], assumed_align=16),
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    reduce = cute.compile(TopkReduce(HIDDEN, TOP_K, CombineFormat.parse("bf16")), *args, stream)
    reduce(*args, stream)
    return flashinfer_rmsnorm(reduced, weight, EPS)


@gpu
@pytest.mark.parametrize(
    "cell", rubin_moe_back_norm.CELLS, ids=[c.why[:48] for c in rubin_moe_back_norm.CELLS]
)
def test_certified_cells(cell, certifying):  # noqa: F811 -- fixture resolved by name
    certifying(rubin_moe_back_norm)
    spec = cell.spec
    rows, t = spec["rows"], spec["bucket"]
    plane, scores, weight = _inputs(t, zlib.crc32(cell.why.encode()), spec.get("mode", "normal"))
    out = torch.full((rows, HIDDEN), float("nan"), dtype=torch.bfloat16, device="cuda")
    rubin_moe_back_norm(plane, scores, weight, EPS, out, pdl=spec.get("pdl", True))
    want = _today(plane, scores, weight, rows)
    torch.cuda.synchronize()
    assert torch.equal(out, want), (
        f"{int((out != want).sum())} values differ from today's two kernels"
    )
    ref = rubin_moe_back_norm.reference(plane, scores, weight, EPS, out)
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)


@gpu
def test_cuda_graph_replay_follows_the_plane():
    """Captured at 96 rows of bucket 128; the replay reads the plane and weights as they are then."""
    plane, scores, weight = _inputs(128, 11)
    out = torch.empty(96, HIDDEN, dtype=torch.bfloat16, device="cuda")
    rubin_moe_back_norm(plane, scores, weight, EPS, out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rubin_moe_back_norm(plane, scores, weight, EPS, out)
    new_plane, new_scores, _ = _inputs(128, 12)
    plane.copy_(new_plane)
    scores.copy_(new_scores)
    graph.replay()
    want = _today(plane, scores, weight, 96)
    torch.cuda.synchronize()
    assert torch.equal(out, want)


def test_flashinfer_geometry_of_the_k31_latent():
    """3584: FlashInfer sums a row with 64 threads of 7 16-byte vectors, two rows per block."""
    assert kernel.flashinfer_geometry(3584) == (64, 8, 7)
    assert kernel.flashinfer_threads_per_row(3584) == 64


@pytest.mark.parametrize("hidden", [3072, 3584 + 8, 7168])
def test_widths_flashinfer_tiles_otherwise_are_refused(hidden):
    with pytest.raises(ValueError):
        kernel.flashinfer_geometry(hidden)


def test_rows_per_cta():
    assert kernel.rows_per_cta_for(96, 212) == 1
    assert kernel.rows_per_cta_for(424, 212) == 1
    assert kernel.rows_per_cta_for(425, 212) == 2


def test_guard_refuses_what_the_kernel_would_read_wrong():
    plane, scores, weight = _inputs(8, 3, device="cpu")
    out = torch.empty(8, HIDDEN, dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="CUDA"):
        rubin_moe_back_norm.is_valid(plane, scores, weight, EPS, out)
    with pytest.raises(AssertionError, match="rows"):
        rubin_moe_back_norm.is_valid(
            plane, scores, weight, EPS, torch.empty(9, HIDDEN, dtype=torch.bfloat16)
        )
    with pytest.raises(AssertionError, match="dense"):
        rubin_moe_back_norm.is_valid(plane[:, :, :HIDDEN], scores, weight.float(), EPS, out)
