# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gemm/m3_mx_gemv_bf16 entry: the decode MXFP8 GEMV quantizing a bf16 activation in its prologue.

The weight as the MXFP8 linear method stores it (mxfp8_quantize, swizzled scales). Besides the reference band, the
prologue's quantization is held to mxfp8_quantize itself: the GEMV on a bf16 x must equal, bit for bit, the
prequantized GEMV on mxfp8_quantize(x) -- the claim the target relies on when it mixes the two paths.
"""

import zlib

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.mxfp8_quantize
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.gemm.m3_mx_gemv import (  # noqa: E501
    O_PROJ,
    m3_mx_gemv,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.gemm.m3_mx_gemv_bf16 import (  # noqa: E501
    m3_mx_gemv_bf16 as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "m3_mx_gemv_bf16 requires a CUDA device"

# The element band of test_modeling_v2_m3_mx_gemv.compare (derived there): the same fp32 sums against fp64.
ELEMENT_ULP = 1.5


def build(n: int, k: int, m: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(n, k, generator=g, device="cuda") / k**0.5).to(torch.bfloat16)
    x = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
    wq, wsf = torch.ops.trtllm.mxfp8_quantize(w, True)
    return x, wq.view(n, k), wsf


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    (n, k), m = cell.spec["nk"], cell.spec["m"]
    x, w, wsf = build(n, k, m, seed=zlib.crc32(cell.why.encode()) % 2**31)
    with validating(op):
        out = op(x, w, wsf, use_pdl=True)  # the target launches with PDL
    assert out is not None, "a certified shape must be served"
    assert out.shape == (m, n) and out.dtype is torch.bfloat16
    # The band of test_modeling_v2_m3_mx_gemv.compare: the quantized operands are the reference's exactly (checked
    # below), the sums differ only in order.
    assert_within_ulp(out, op.reference(x, w, wsf), element_ulp=ELEMENT_ULP, rms_ulp=0.1)
    xq, xsf = torch.ops.trtllm.mxfp8_quantize(x, True)
    pre = m3_mx_gemv(xq.view(m, k), xsf, w, wsf, use_pdl=True)
    assert pre is not None and torch.equal(out.view(torch.int16), pre.view(torch.int16)), (
        "the prologue's MXFP8 quantization differs from mxfp8_quantize"
    )


def test_a_strided_activation_is_declined() -> None:
    """The prologue reads x densely; the op declines a strided x (None) rather than computing on it (a served shape:
    the same x, contiguous, is computed)."""
    n, k = O_PROJ
    x, w, wsf = build(n, k, 2, seed=1)
    wide = torch.zeros(2, 2 * k, dtype=torch.bfloat16, device="cuda")
    wide[:, :k] = x
    assert op(wide[:, :k], w, wsf) is None
    assert op(wide[:, :k].contiguous(), w, wsf) is not None
