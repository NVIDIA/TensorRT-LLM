# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: `mid_rows_mm` (FP8 with the per-row activation quantization, and bf16) against an
fp32 reference over the operands it multiplies, per cell; `fp8_row_quant` bitwise against
torch's e4m3 cast; strided activation rows; CUDA-graph replay with new activations."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.gemm.mid_rows_mm import (  # noqa: E501
    AMAX_FLOOR,
    E4M3_MAX,
    MidWeight,
    fp8_row_quant,
    fp8_row_quant_op,
    mid_rows_mm_op,
    precompile,
)

assert torch.cuda.is_available(), "mid_rows_mm requires a CUDA device"

DEV = "cuda"


def _weight(n, k, mode, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = torch.randn(n, k, generator=g, device=DEV) * k**-0.5
    w[:, 7] *= 20.0  # outlier column
    w = w.to(torch.bfloat16)
    if mode == "bf16":
        return w, MidWeight(w.contiguous())
    if mode == "row":
        scale = (w.float().abs().amax(1) / E4M3_MAX).clamp_min(1e-12)
        return w, MidWeight((w.float() / scale[:, None]).to(torch.float8_e4m3fn), scale)
    scale = (w.float().abs().amax() / E4M3_MAX).clamp_min(1e-12).reshape(1)
    return w, MidWeight((w.float() / scale).to(torch.float8_e4m3fn), scale)


def _activation(m, k, seed, pad=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    wide = torch.randn(m, k + pad, generator=g, device=DEV) * 0.7
    wide[:, 3] *= 12.0  # outlier column
    if m > 2:
        wide[1] = 0.0  # an all-zero row
    return wide.to(torch.bfloat16)[:, :k]


def _dequantized(x, mw):
    """The fp32 operands the call multiplies."""
    if not mw.fp8:
        return x.float(), mw.w.float()
    q, s = fp8_row_quant(x)
    wd = mw.w.float() * (mw.scale[:, None] if mw.scale.numel() > 1 else mw.scale)
    return q.float() * s[:, None], wd


def _close(y, x, mw) -> None:
    """Within one bf16 rounding of the fp32 product of the operands (plus fp32 summation
    noise relative to sum |x w|)."""
    xd, wd = _dequantized(x, mw)
    with true_fp32_matmul():
        ref = xd @ wd.t()
        absref = xd.abs() @ wd.abs().t()
    ulp = torch.exp2(torch.floor(torch.log2(ref.abs().clamp_min(1e-30))) - 7)
    excess = ((y.float() - ref).abs() - 0.5 * ulp).clamp_min(0) / absref.clamp_min(1e-30)
    assert torch.isfinite(y.float()).all()
    assert excess.max().item() < 1e-5, f"error beyond one rounding: {excess.max().item():.2e}"


@pytest.mark.parametrize(
    "cell", mid_rows_mm_op.CELLS, ids=[c.why[:48] for c in mid_rows_mm_op.CELLS]
)
def test_cells(cell) -> None:
    n, k, m, mode = cell.spec["n"], cell.spec["k"], cell.spec["m"], cell.spec["mode"]
    w, mw = _weight(n, k, mode, n + m)
    x = _activation(m, k, m, pad=64 if mode != "bf16" else 0)
    mid_rows_mm_op.is_valid(x, mw)
    y = mid_rows_mm_op(x, mw)
    assert y.shape == (m, n) and y.dtype == torch.bfloat16 and y.is_contiguous()
    _close(y, x, mw)
    if mw.fp8:
        # The FP8 product against the bf16 projection: e4m3 keeps 3 mantissa bits on both
        # operands; per-row activation scales are at least as accurate as one per tensor.
        full = x.float() @ w.float().t()
        assert ((y.float() - full).norm() / full.norm()).item() < 0.06


@pytest.mark.parametrize(
    "cell", fp8_row_quant_op.CELLS, ids=[c.why[:48] for c in fp8_row_quant_op.CELLS]
)
def test_row_quant_is_torchs_cast(cell) -> None:
    m, k, pad = cell.spec["m"], cell.spec["k"], cell.spec.get("pad", 0)
    x = _activation(m, k, 5 + m, pad=pad) * 3
    if m > 4:
        x[4, 9] = 900.0  # a row whose maximum is far above the rest
    q, s = fp8_row_quant_op(x)
    # The kernel divides correctly rounded (torch divides by a scalar as a product with
    # its reciprocal): the quotients in float64, rounded once.
    amax = x.float().abs().amax(1).clamp_min(AMAX_FLOOR).double()
    assert torch.equal(s, (amax / E4M3_MAX).float())
    want = (x.float() * (E4M3_MAX / amax).float()[:, None]).to(torch.float8_e4m3fn)
    assert torch.equal(q.view(torch.uint8), want.view(torch.uint8))


def test_graph_replay_matches_eager() -> None:
    for mode, n, k, m in (("tensor", 16480, 2560, 64), ("bf16", 512, 2560, 200)):
        _, mw = _weight(n, k, mode, 3)
        precompile(mw)
        x = _activation(m, k, 1).contiguous()
        mid_rows_mm_op(x, mw)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = mid_rows_mm_op(x, mw)
        x.copy_(_activation(m, k, 2) * 4)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, mid_rows_mm_op(x, mw))
