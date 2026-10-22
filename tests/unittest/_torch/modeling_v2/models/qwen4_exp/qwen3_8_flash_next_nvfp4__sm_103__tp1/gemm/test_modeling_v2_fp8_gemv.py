# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: `fp8_gemv` / `fp8_gemv_conv1d` against an fp32 reference over the dequantized
weight (128x128-block e4m3), per cell; strided activations; CUDA-graph replay."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.gemm.fp8_gemv import (  # noqa: E501
    dequantize_fp8_block,
    fp8_gemv_conv1d_op,
    fp8_gemv_op,
    pack_weight,
    quantize_fp8_block,
)

assert torch.cuda.is_available(), "fp8_gemv requires a CUDA device"

DEV = "cuda"


def _weight(n, k, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = torch.randn(n, k, generator=g, device=DEV) * k**-0.5
    w[:, 7] *= 30.0  # outlier column
    w[: min(n, 128), 128:256] *= 1e-4  # a tiny block: subnormal codes
    codes, scale = quantize_fp8_block(w.to(torch.bfloat16))
    return codes, scale, pack_weight(codes, scale)


def _close(y, ref, x, deq) -> None:
    """Within one bf16 rounding of the fp32 reference (plus fp32 summation noise)."""
    with true_fp32_matmul():
        absref = x.float().abs() @ deq.abs().t()
    ulp = torch.exp2(torch.floor(torch.log2(ref.abs().clamp_min(1e-30))) - 7)
    excess = ((y.float() - ref).abs() - 0.5 * ulp).clamp_min(0) / absref.clamp_min(1e-30)
    assert excess.max().item() < 1e-5, f"error beyond one rounding: {excess.max().item():.2e}"


@pytest.mark.parametrize("cell", fp8_gemv_op.CELLS, ids=[c.why[:40] for c in fp8_gemv_op.CELLS])
def test_gemv_cells(cell) -> None:
    n, k, m = cell.spec["n"], cell.spec["k"], cell.spec["m"]
    codes, scale, packed = _weight(n, k, n + m)
    g = torch.Generator(device=DEV).manual_seed(m)
    # A column slice of a wider row, as the target's split projections hand over.
    wide = torch.randn(m, k + 256, generator=g, device=DEV).to(torch.bfloat16)
    x = wide[:, :k]
    y = fp8_gemv_op(x, packed)
    assert y.shape == (m, n) and y.dtype == torch.bfloat16
    _close(y, fp8_gemv_op.reference(x, codes, scale), x, dequantize_fp8_block(codes, scale))


@pytest.mark.parametrize(
    "cell", fp8_gemv_conv1d_op.CELLS, ids=[c.why[:40] for c in fp8_gemv_conv1d_op.CELLS]
)
def test_conv1d_cells(cell) -> None:
    n, k, conv_dim, pool, pad = 16480, 2560, 10240, 64, -1
    codes, scale, packed = _weight(n, k, 3)
    deq = dequantize_fp8_block(codes, scale)
    slots = list(cell.spec["slots"])
    idx = torch.tensor(slots, dtype=torch.int32, device=DEV)
    g = torch.Generator(device=DEV).manual_seed(11)
    backing = torch.randn(pool, conv_dim, 4, generator=g, device=DEV).to(torch.bfloat16)
    conv_state = backing[:, :, 1:]  # a strided pool view
    conv_weight = torch.randn(conv_dim, 4, generator=g, device=DEV).to(torch.bfloat16)
    x = torch.randn(len(slots), k, generator=g, device=DEV).to(torch.bfloat16)
    before = conv_state.clone()
    with true_fp32_matmul():
        proj = (x.float() @ deq.t()).to(torch.bfloat16)
    want, want_state = proj.clone(), conv_state.clone()
    for r, s in enumerate(slots):
        if s == pad:
            continue
        window = torch.cat([conv_state[s].float(), proj[r, :conv_dim].float()[:, None]], 1)
        c = (window * conv_weight.float()).sum(1)
        want[r, :conv_dim] = (c * torch.sigmoid(c)).to(torch.bfloat16)
        want_state[s] = window[:, 1:].to(torch.bfloat16)
    out = fp8_gemv_conv1d_op(x, packed, conv_state, conv_weight, idx, conv_dim, pad)
    # The projection may round one ulp differently than the reference's fp32 matmul; a flip
    # carries through the conv. Nearly every output is bitwise the reference's.
    assert (out == want).float().mean().item() > 0.999
    assert ((out.float() - want.float()).abs() / want.float().abs().clamp_min(1e-2)).max() < 0.02
    assert ((conv_state.float() - want_state.float()).abs()).max().item() < 0.02
    others = [s for s in range(pool) if s not in slots]
    assert torch.equal(conv_state[others], before[others]), "other slots untouched"


def test_graph_replay_matches_eager() -> None:
    _, _, packed = _weight(2560, 6144, 5)
    x = torch.randn(4, 6144, device=DEV).to(torch.bfloat16)
    fp8_gemv_op(x, packed)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fp8_gemv_op(x, packed)
    x.copy_(torch.randn(4, 6144, device=DEV).to(torch.bfloat16))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, fp8_gemv_op(x, packed))
