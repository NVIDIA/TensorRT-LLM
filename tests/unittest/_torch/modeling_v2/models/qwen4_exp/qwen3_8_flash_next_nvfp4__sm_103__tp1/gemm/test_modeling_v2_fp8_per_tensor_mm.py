# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: `fp8_per_tensor_mm` against its fp32 reference (the activation quantized as the
op does, both operands dequantized) per cell, the activation quantization bitwise against
torch's e4m3 cast, and CUDA-graph replay with a new activation."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.gemm.fp8_per_tensor_mm import (  # noqa: E501
    AMAX_PARTS,
    E4M3_MAX,
    amax_partials,
    fp8_per_tensor_mm,
    fp8_per_tensor_weight,
    fp8_quantize_per_tensor,
)

assert torch.cuda.is_available(), "fp8_per_tensor_mm requires a CUDA device"

DEV = "cuda"


def _operands(m, k, n, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = (torch.randn(m, k, generator=g, device=DEV) * 0.7).to(torch.bfloat16)
    w = (torch.randn(n, k, generator=g, device=DEV) * k**-0.5).to(torch.bfloat16)
    codes, scale = fp8_per_tensor_weight(w)
    return x, w, codes.t(), scale


@pytest.mark.parametrize(
    "cell", fp8_per_tensor_mm.CELLS, ids=[c.why[:40] for c in fp8_per_tensor_mm.CELLS]
)
def test_cells(cell) -> None:
    m, k, n = cell.spec["m"], cell.spec["k"], cell.spec["n"]
    x, w, codes_t, scale = _operands(m, k, n, 1000 + m)
    got = fp8_per_tensor_mm(x, codes_t, scale)
    want = fp8_per_tensor_mm.reference(x, codes_t, scale)
    assert got.dtype == torch.bfloat16 and got.shape == (m, n)
    rel = (got.float() - want).norm() / want.norm()
    # The bf16 output rounding and cuBLASLt's accumulation separate them: its small-M FP8
    # kernels accumulate below fp32 precision (measured 8.0e-3 at one row, under 4e-3 from
    # 128 rows) -- a fifth of the e4m3 quantization error the op is gated on below.
    assert rel < 1.2e-2, f"relative error {rel:.2e} against the dequantized reference"
    full = x.float() @ w.float().t()
    lossy = (got.float() - full).norm() / full.norm()
    assert lossy < 0.06, f"FP8 error against the bf16 projection {lossy:.3f}"


def test_quantization_is_torchs_cast() -> None:
    g = torch.Generator(device=DEV).manual_seed(5)
    x = (torch.randn(37, 2560, generator=g, device=DEV) * 3).to(torch.bfloat16)
    q, scale = fp8_quantize_per_tensor(x)
    want_scale = x.abs().amax().float() / E4M3_MAX
    assert torch.equal(scale, want_scale.reshape(1))
    want = (x.float() * (E4M3_MAX / x.abs().amax().float())).to(torch.float8_e4m3fn)
    assert (q.view(torch.uint8) != want.view(torch.uint8)).float().mean() < 1e-4


@pytest.mark.parametrize("m,k", [(16384, 6144), (3, 2560)])
def test_producer_partials_give_the_same_bits(m, k) -> None:
    """Partial maxima a producer folded (here: per row block, modulo the buffer) replace the
    maximum pass: same scale, same codes, same output."""
    x, _, codes_t, scale = _operands(m, k, 2560, 11)
    parts = amax_partials(x.device)
    blocks = torch.split(x.abs().float().amax(1), 16)
    for i, b in enumerate(blocks):
        parts[i % AMAX_PARTS] = torch.maximum(parts[i % AMAX_PARTS], b.max())
    q0, s0 = fp8_quantize_per_tensor(x)
    q1, s1 = fp8_quantize_per_tensor(x, parts)
    assert torch.equal(s0, s1) and torch.equal(q0.view(torch.uint8), q1.view(torch.uint8))
    assert torch.equal(
        fp8_per_tensor_mm(x, codes_t, scale, parts), fp8_per_tensor_mm(x, codes_t, scale)
    )


def test_zero_activation_and_strided_rows() -> None:
    x, _, codes_t, scale = _operands(8, 2560, 2560, 3)
    zeros = torch.zeros_like(x)
    assert torch.equal(
        fp8_per_tensor_mm(zeros, codes_t, scale),
        torch.zeros(8, 2560, dtype=torch.bfloat16, device=DEV),
    )
    wide = torch.zeros(8, 4096, dtype=torch.bfloat16, device=DEV)
    wide[:, :2560] = x
    assert torch.equal(
        fp8_per_tensor_mm(wide[:, :2560], codes_t, scale), fp8_per_tensor_mm(x, codes_t, scale)
    )


def test_graph_replay_matches_eager() -> None:
    x, _, codes_t, scale = _operands(64, 2560, 16480, 7)
    fp8_per_tensor_mm(x, codes_t, scale)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fp8_per_tensor_mm(x, codes_t, scale)
    x.copy_(_operands(64, 2560, 16480, 8)[0] * 4)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, fp8_per_tensor_mm(x, codes_t, scale))
