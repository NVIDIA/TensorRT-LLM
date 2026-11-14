# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test the pinned MXFP8 tactic, its reproducibility, and both stock candidates against the reference."""

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.mxfp8_quantize
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp8.modeling import (  # noqa: E501
    _pinned_mxfp8_gemm,
    _pinned_mxfp8_quantize,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.gemm.m3_mxfp8_linear import (
    mxfp8_reference,
)
from tensorrt_llm._torch.custom_ops.torch_custom_ops import FlashInferMXFP8GemmRunner

__extra_import_path__ = ["../../../..", ".", "../.."]
from _validating import assert_within_ulp  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("MX-TAC requires a CUDA device", allow_module_level=True)

# The TP8 per-rank projections (ops/gemm/m3_mx_gemv) at decode rows the GEMVs do not serve (9..16) and a few beyond.
CASES = [
    ((1536, 6144), 16),
    ((1280, 6144), 12),
    ((6144, 1024), 9),
    ((768, 6144), 16),
    ((6144, 384), 12),
    ((3072, 6144), 24),
    ((6144, 1536), 32),
    ((1536, 6144), 48),
]


@pytest.mark.parametrize("nk,m", CASES, ids=[f"{n}x{k}-m{m}" for (n, k), m in CASES])
@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_pinned_decision_matches_every_stock_candidate(nk, m, scale) -> None:
    n, k = nk
    g = torch.Generator(device="cuda").manual_seed(n + k + m)
    w = (torch.randn(n, k, generator=g, device="cuda") / k**0.5).to(torch.bfloat16)
    x = (torch.randn(m, k, generator=g, device="cuda") * scale).to(torch.bfloat16)
    x[0, ::32] *= 1e3  # one outlier per 32-element block on the first row
    wq, wsf = torch.ops.trtllm.mxfp8_quantize(w, True)
    wq = wq.view(n, k)
    xq, xsf = _pinned_mxfp8_quantize(x)
    tq, tsf = torch.ops.trtllm.mxfp8_quantize(x, True)
    assert torch.equal(xq.view(torch.uint8).view(m, k), tq.view(torch.uint8).view(m, k))
    assert torch.equal(xsf.view(torch.uint8).flatten(), tsf.view(torch.uint8).flatten())
    xq = xq.view(m, k)
    out = _pinned_mxfp8_gemm(xq, xsf, wq, wsf, torch.bfloat16)
    runner = FlashInferMXFP8GemmRunner(torch.bfloat16)
    # The pinned path must use the same tactic and produce identical bits on every call.
    cute = runner.forward([xq, xsf, wq, wsf], tactic=FlashInferMXFP8GemmRunner.CUTE_DSL)
    assert torch.equal(out.view(torch.int16), cute.view(torch.int16))
    cutlass = runner.forward([xq, xsf, wq, wsf], tactic=FlashInferMXFP8GemmRunner.CUTLASS)
    ref = mxfp8_reference(xq, xsf, wq, wsf).to(torch.bfloat16)
    # Different GEMM implementations may accumulate in a different order. Check both against
    # the independent dequantized reference using the certified MXFP8 GEMM accuracy bounds.
    assert_within_ulp(out, ref, element_ulp=2.0, rms_ulp=0.25)
    assert_within_ulp(cutlass, ref, element_ulp=2.0, rms_ulp=0.25)
    again = _pinned_mxfp8_gemm(xq, xsf, wq, wsf, torch.bfloat16)
    assert torch.equal(out.view(torch.int16), again.view(torch.int16))
