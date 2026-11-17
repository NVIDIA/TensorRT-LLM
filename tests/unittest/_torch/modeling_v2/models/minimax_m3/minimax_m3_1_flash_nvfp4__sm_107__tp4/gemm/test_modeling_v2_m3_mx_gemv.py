# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gemm/m3_mx_gemv entry: the decode MXFP8 GEMV on a producer-quantized activation, and for the
gemm/m3_mx_gemv_bf16 entry that quantizes a bf16 activation in its prologue.

Operands as the target holds them: the weight through mxfp8_quantize with the swizzled scale layout (the MXFP8 linear
method's storage), the activation through mxfp8_quantize with the swizzled layout (what the boundary all-reduce and
the attention merge publish).
"""

import zlib

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.mxfp8_quantize
from tensorrt_llm._torch.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4.ops.gemm.m3_mx_gemv import (  # noqa: E501
        m3_mx_gemv as op,
    )
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4.ops.gemm.m3_mx_gemv_bf16 import (  # noqa: E501
        m3_mx_gemv_bf16 as op_bf16,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP4 target runs on SM107 only",
)

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("the sm_107 TP2 target runs on SM107 only", allow_module_level=True)


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
    the o_proj and the QKV (6 seeds each, GB300), the kernel at most 1.25, both at most 0.009 ulp RMS. The band
    leaves the margin between that and a full extra ulp.
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


def test_guard_refuses_a_strided_weight() -> None:
    _, xq, xsf, w, wsf = build(6144, 2048, 1, seed=3)
    wide = torch.zeros(6144, 4096, dtype=w.dtype, device="cuda")
    wide[:, :2048] = w
    with pytest.raises(AssertionError, match="contiguous e4m3"):
        with validating(op):
            op(xq, xsf, wide[:, :2048], wsf)


@pytest.mark.parametrize("nk", [(2560, 6144), (2304, 6144)], ids=["qkv_sparse", "qkv_dense"])
@pytest.mark.parametrize("m", [1, 4, 8])
def test_qkv_split8_config(nk, m) -> None:
    """The TP4 dispatch's (split, ring) override for the QKV at decode rows: within the band of the fp64 reference,
    and the graph replay bit-identical to the eager call (one split-K order per config)."""
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4 import (
        modeling as dispatch,
    )

    cfg = dispatch.mx_gemv_cfg(nk[0], nk[1], m)
    assert cfg == (8, 8)
    _, xq, xsf, wq, wsf = build(nk[0], nk[1], m, zlib.crc32(f"qkv8-{nk}-{m}".encode()))
    eager = op(xq, xsf, wq, wsf, cfg=cfg)
    assert eager is not None, "the GEMV declined the split-8 config"
    compare(eager, op.reference(xq, xsf, wq, wsf))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(xq, xsf, wq, wsf, cfg=cfg)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured.view(torch.int16), eager.view(torch.int16))


@pytest.mark.parametrize("m", [1, 4, 8])
def test_oproj_partial_tmem_a_is_bitwise(m) -> None:
    """OPROJ_TMEM256: the o_proj with only its first 6 k-tiles' A in TMEM (the rest read from the smem ring) gives
    the same bits as all 8 in TMEM, and as A all in smem; graph replay equal to eager."""
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4.kernels.mx_gemv import (  # noqa: E501
        op as mx_gemv,
    )

    n, k = 6144, 2048
    _, xq, xsf, wq, wsf = build(n, k, m, zlib.crc32(f"oproj-tmem256-{m}".encode()))
    resolved = mx_gemv._resolve(m, n, k, False, True, (2, 8, 0, 1, 1, 1, 6), True, None)
    assert resolved is not None and resolved[0][18] == 6, (
        "the partial-TMEM config must resolve as such"
    )
    outs = [
        op(xq, xsf, wq, wsf, cfg=cfg) for cfg in ((2, 8, 0), (2, 8, 0, 1, 1, 1, 6), (2, 8, 0, 1, 0))
    ]
    assert all(o is not None for o in outs)
    for o in outs[1:]:
        assert torch.equal(o.view(torch.int16), outs[0].view(torch.int16))
    compare(outs[1], op.reference(xq, xsf, wq, wsf))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(xq, xsf, wq, wsf, cfg=(2, 8, 0, 1, 1, 1, 6))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured.view(torch.int16), outs[0].view(torch.int16))


# ---- gemm/m3_mx_gemv_bf16: the prologue's quantization is mxfp8_quantize itself -------------------------------------
# Besides the reference band, the GEMV on a bf16 x must equal, bit for bit, the prequantized GEMV on
# mxfp8_quantize(x) -- the claim the target relies on when it mixes the two paths.


def _build_bf16(n: int, k: int, m: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(n, k, generator=g, device="cuda") / k**0.5).to(torch.bfloat16)
    x = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
    wq, wsf = torch.ops.trtllm.mxfp8_quantize(w, True)
    return x, wq.view(n, k), wsf


@pytest.mark.parametrize("cell", op_bf16.CELLS, ids=[c.why[:44] for c in op_bf16.CELLS])
def test_bf16_certified_cells(cell) -> None:
    (n, k), m = cell.spec["nk"], cell.spec["m"]
    x, w, wsf = _build_bf16(n, k, m, seed=zlib.crc32(cell.why.encode()) % 2**31)
    with validating(op_bf16):
        out = op_bf16(x, w, wsf, use_pdl=True)  # the target launches with PDL
    assert out is not None, "a certified shape must be served"
    assert out.shape == (m, n) and out.dtype is torch.bfloat16
    # The band of compare: the quantized operands are the reference's exactly (checked below), the sums differ only
    # in order.
    compare(out, op_bf16.reference(x, w, wsf))
    xq, xsf = torch.ops.trtllm.mxfp8_quantize(x, True)
    pre = op(xq.view(m, k), xsf, w, wsf, use_pdl=True)
    assert pre is not None and torch.equal(out.view(torch.int16), pre.view(torch.int16)), (
        "the prologue's MXFP8 quantization differs from mxfp8_quantize"
    )
