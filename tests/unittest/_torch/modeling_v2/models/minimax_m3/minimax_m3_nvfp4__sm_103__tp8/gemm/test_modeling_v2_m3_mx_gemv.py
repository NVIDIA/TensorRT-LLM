# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gemm/m3_mx_gemv entry: the decode MXFP8 GEMV on a producer-quantized activation.

Operands as the target holds them: the weight through mxfp8_quantize with the swizzled scale layout (the MXFP8 linear
method's storage), the activation through mxfp8_quantize with the swizzled layout (what the boundary all-reduce and
the attention merge publish).
"""

import zlib

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.mxfp8_quantize
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.gemm.m3_mx_gemv import (  # noqa: E501
    O_PROJ,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.gemm.m3_mx_gemv import (  # noqa: E501
    m3_mx_gemv as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "m3_mx_gemv requires a CUDA device"


# Element band, in ULP units of the row scale: see compare.
ELEMENT_ULP = 1.5


def compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Element ELEMENT_ULP of the row scale, relative RMS 0.1 ulp.

    Kernel and reference multiply bit-identical dequantized operands; the kernel sums in fp32 (tensor-core blocks,
    then the split-K partials), the reference in fp64. Over K <= 6144 that order difference is ~1e-6 relative, so the
    two bf16 roundings disagree only where the sum sits within that of a rounding boundary: one bf16 ulp of that
    element, which for an element in its row maximum's binade is up to two of ``assert_within_ulp``'s units of the row
    scale, and a few in ten thousand elements in aggregate. Any fp32 sum shows it: torch's IEEE fp32 matmul of the
    same dequantized operands measured at most 1.22 units against the fp64 reference over every row count 1..48 on
    the tp4 target's o_proj and QKV (6 seeds each, GB300; K 2048 and 6144, the tp8 projections' K is 1024..6144), the
    kernel at most 1.25, both at most 0.009 ulp RMS. The band leaves the margin between that and a full extra ulp.
    """
    assert_within_ulp(out, ref, element_ulp=ELEMENT_ULP, rms_ulp=0.1)


def build(n: int, k: int, m: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(n, k, generator=g, device="cuda") / k**0.5).to(torch.bfloat16)
    x = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
    wq, wsf = torch.ops.trtllm.mxfp8_quantize(w, True)
    xq, xsf = torch.ops.trtllm.mxfp8_quantize(x, True)
    return x, xq.view(m, k), xsf, wq.view(n, k), wsf


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    (n, k), m = cell.spec["nk"], cell.spec["m"]
    _, xq, xsf, w, wsf = build(n, k, m, seed=zlib.crc32(cell.why.encode()) % 2**31)
    with validating(op):
        out = op(xq, xsf, w, wsf, use_pdl=True)  # the target launches with PDL
    assert out is not None, "a certified shape must be served"
    assert out.shape == (m, n) and out.dtype is torch.bfloat16
    compare(out, op.reference(xq, xsf, w, wsf))


def test_an_unserved_shape_returns_none() -> None:
    """Shapes outside the tuned table fall back at the call site; the op says so rather than computing."""
    _, xq, xsf, w, wsf = build(256, 512, 1, seed=1)
    assert op(xq, xsf, w, wsf) is None


def test_guard_refuses_linear_activation_scales() -> None:
    """A linear [M, K / 32] scale buffer is read as the swizzled one."""
    n, k = O_PROJ
    x, _, _, w, wsf = build(n, k, 4, seed=2)
    xq, xsf = torch.ops.trtllm.mxfp8_quantize(x, False)
    with pytest.raises(AssertionError, match="swizzled"):
        with validating(op):
            op(xq.view(4, k), xsf, w, wsf)


def test_guard_refuses_a_strided_weight() -> None:
    n, k = O_PROJ
    _, xq, xsf, w, wsf = build(n, k, 1, seed=3)
    wide = torch.zeros(n, 2 * k, dtype=w.dtype, device="cuda")
    wide[:, :k] = w
    with pytest.raises(AssertionError, match="contiguous e4m3"):
        with validating(op):
            op(xq, xsf, wide[:, :k], wsf)
