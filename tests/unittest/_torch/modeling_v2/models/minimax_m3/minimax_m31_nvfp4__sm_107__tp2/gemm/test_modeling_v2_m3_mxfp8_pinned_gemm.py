# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_mxfp8_pinned_gemm entry (MX-TAC): the pinned decode-graph MXFP8 GEMM gives the same bits as
every candidate stock's AutoTuner can pick (CuTe DSL, CUTLASS), the same on every call, within the MXFP8 band of the
dequantized product."""

import zlib

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.mxfp8_quantize
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (  # noqa: E501
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.ops.gemm.m3_mxfp8_pinned_gemm import (  # noqa: E501
    m3_mxfp8_pinned_gemm as op,
)
from tensorrt_llm._torch.custom_ops.torch_custom_ops import FlashInferMXFP8GemmRunner

__extra_import_path__ = ["../../../..", "."]
from _m3_mxfp8_operands import mxfp8  # noqa: E402
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "m3_mxfp8_pinned_gemm requires a CUDA device"

TACTIC = modeling.MXFP8_DECODE_GEMM_TACTIC

# The TP2 projections at rows the GEMVs do not serve (SHARED_DOWN 17..32 -- SWQ's rows --, over 48) and the bf16
# activations over the GEMVs' bf16 bounds.
CASES = [
    ((6144, 1536), 17),
    ((6144, 1536), 32),
    ((4992, 6144), 24),
    ((4608, 6144), 64),
    ((6144, 4096), 49),
    ((3072, 6144), 20),
    ((6144, 6144), 12),
    ((12288, 6144), 56),
]


def test_the_pinned_tactic_is_cute_dsl() -> None:
    assert TACTIC == FlashInferMXFP8GemmRunner.CUTE_DSL


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    act, act_sf = mxfp8(spec["m"], spec["k"], g)
    w, w_sf = mxfp8(spec["n"], spec["k"], g)
    with validating(op):
        out = op(act, act_sf, w, w_sf, torch.bfloat16, TACTIC)
    assert out.shape == (spec["m"], spec["n"]) and out.dtype == torch.bfloat16
    assert_within_ulp(
        out,
        op.reference(act, act_sf, w, w_sf, torch.bfloat16, TACTIC),
        element_ulp=2.0,
        rms_ulp=0.25,
    )


@pytest.mark.parametrize("nk,m", CASES, ids=[f"{n}x{k}-m{m}" for (n, k), m in CASES])
@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3, "stress"])
def test_pinned_decision_matches_every_stock_candidate(nk, m, scale) -> None:
    """``stress``: every 32-element block of the weight and the activation scaled by its own 2^-12..2^+12."""
    n, k = nk
    g = torch.Generator(device="cuda").manual_seed(n + k + m)
    w = torch.randn(n, k, generator=g, device="cuda") / k**0.5
    x = torch.randn(m, k, generator=g, device="cuda")
    if scale == "stress":
        for t in (w, x):
            block = torch.randint(-12, 13, (t.shape[0], k // 32), generator=g, device="cuda")
            t.mul_(torch.exp2(block.float()).repeat_interleave(32, dim=1))
    else:
        x = x * scale
    w, x = w.to(torch.bfloat16), x.to(torch.bfloat16)
    x[0, ::32] *= 1e3  # one outlier per 32-element block on the first row
    wq, wsf = torch.ops.trtllm.mxfp8_quantize(w, True)
    wq = wq.view(n, k)
    xq, xsf = torch.ops.trtllm.mxfp8_quantize(x, True)
    xq = xq.view(m, k)
    out = op(xq, xsf, wq, wsf, torch.bfloat16, TACTIC)
    runner = FlashInferMXFP8GemmRunner(torch.bfloat16)
    for tactic in (FlashInferMXFP8GemmRunner.CUTE_DSL, FlashInferMXFP8GemmRunner.CUTLASS):
        ref = runner.forward([xq, xsf, wq, wsf], tactic=tactic)
        assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), tactic
    again = op(xq, xsf, wq, wsf, torch.bfloat16, TACTIC)
    assert torch.equal(out.view(torch.int16), again.view(torch.int16))
