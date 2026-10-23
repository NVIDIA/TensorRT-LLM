# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: `nvfp4_gemv` against an fp32 reference over the dequantized NVFP4 weight, per
cell; the quantizer against its definition; CUDA-graph replay; one compiled kernel per row
octet; the f16-range guard."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.gemm import (  # noqa: E501
    nvfp4_gemv as nv,
)

assert torch.cuda.is_available(), "nvfp4_gemv requires a CUDA device"

DEV = "cuda"
nvfp4_gemv_op = nv.nvfp4_gemv_op


def _weight(n, k, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = torch.randn(n, k, generator=g, device=DEV) * k**-0.5
    w[:, 7] *= 30.0  # outlier column: its blocks' scales run up to e4m3's top
    w[: min(n, 64), 128:256] *= 1e-4  # near-zero blocks: small (subnormal) e4m3 scales
    w = w.to(torch.bfloat16)
    codes, scales, gscale = nv.quantize_nvfp4(w)
    return w, codes, scales, gscale, nv.pack_weight(codes, scales, gscale)


def _x(m, k, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    # A column slice of a wider row, as the target's split activations hand over.
    return torch.randn(m, k + 256, generator=g, device=DEV).to(torch.bfloat16)[:, :k]


def _close(y, ref, x, deq) -> None:
    """Within one bf16 rounding of the fp32 reference, plus fp32 summation noise."""
    with true_fp32_matmul():
        absref = x.float().abs() @ deq.abs().t()
    ulp = torch.exp2(torch.floor(torch.log2(ref.abs().clamp_min(1e-30))) - 7)
    excess = ((y.float() - ref).abs() - 0.5 * ulp).clamp_min(0) / absref.clamp_min(1e-30)
    assert excess.max().item() < 1e-5, f"error beyond one rounding: {excess.max().item():.2e}"


@pytest.mark.parametrize("cell", nvfp4_gemv_op.CELLS, ids=[c.why[:40] for c in nvfp4_gemv_op.CELLS])
def test_gemv_cells(cell) -> None:
    n, k, m = cell.spec["n"], cell.spec["k"], cell.spec["m"]
    _, codes, scales, gscale, packed = _weight(n, k, n + m)
    x = _x(m, k, m)
    nvfp4_gemv_op.is_valid(x, packed)
    y = nvfp4_gemv_op(x, packed)
    assert y.shape == (m, n) and y.dtype == torch.bfloat16
    ref = nvfp4_gemv_op.reference(x, codes, scales, gscale)
    _close(y, ref, x, nv.dequantize_nvfp4(codes, scales, gscale))


def test_quantizer_matches_its_definition() -> None:
    """Codes are the nearest e2m1 value of w / (block scale x global scale); the largest block
    scale is e4m3's 448; the relative error is NVFP4's (well under 2^-2 a row)."""
    w, codes, scales, gscale, _ = _weight(512, 1024, 3)
    assert codes.max().item() <= 15 and torch.isfinite(scales.float()).all()
    assert scales.float().max().item() == 448.0
    step = scales.float().repeat_interleave(nv.SF_VEC, 1) * gscale
    v = torch.where(step > 0, w.float() / step.clamp_min(1e-38), 0.0)
    grid = torch.tensor(nv.E2M1_VALUES, device=DEV)
    best = (v.abs()[..., None] - grid).abs().min(-1).values
    got = nv.dequantize_nvfp4(codes, scales, gscale) / step.clamp_min(1e-38)
    err = (v - torch.where(step > 0, got, 0.0)).abs()
    assert (err <= best + 1e-6).all(), "a code is not the nearest e2m1 value"
    deq = nv.dequantize_nvfp4(codes, scales, gscale)
    rel = (deq - w.float()).norm(dim=1) / w.float().norm(dim=1)
    assert rel.max().item() < 0.2, rel.max().item()


def test_chunked_pack_matches_whole() -> None:
    """`pack_from_bf16` (row chunks, a partial last tile) is `pack_weight` of the whole."""
    w = torch.randn(1000, 256, device=DEV).to(torch.bfloat16)
    a = nv.pack_from_bf16(w, rows_per_chunk=192)
    b = nv.pack_weight(*nv.quantize_nvfp4(w))
    assert a.global_scale == b.global_scale and torch.equal(a.data, b.data)


def test_one_kernel_per_row_octet() -> None:
    """The row count is a launch argument: rows 1..16 build two kernels per shape."""
    _, codes, scales, gscale, packed = _weight(2048, 512, 5)
    before = nv._compiled.cache_info().currsize
    for m in range(1, nv.MAX_ROWS + 1):
        x = _x(m, 512, m)
        _close(
            nvfp4_gemv_op(x, packed),
            nvfp4_gemv_op.reference(x, codes, scales, gscale),
            x,
            nv.dequantize_nvfp4(codes, scales, gscale),
        )
    assert nv._compiled.cache_info().currsize - before <= 2


def test_graph_replay_matches_eager() -> None:
    _, _, _, _, packed = _weight(4096, 2560, 9)
    x = torch.randn(4, 2560, device=DEV).to(torch.bfloat16)
    nvfp4_gemv_op(x, packed)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = nvfp4_gemv_op(x, packed)
    x.copy_(torch.randn(4, 2560, device=DEV).to(torch.bfloat16))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, nvfp4_gemv_op(x, packed))


def test_is_valid_rejects_activation_beyond_f16() -> None:
    _, _, _, _, packed = _weight(256, 256, 1)
    x = torch.randn(2, 256, device=DEV).to(torch.bfloat16)
    nvfp4_gemv_op.is_valid(x, packed)
    x[1, 3] = 1e5
    with pytest.raises(AssertionError):
        nvfp4_gemv_op.is_valid(x, packed)
