# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of rubin_mla_ctx_wuk_fp8, W_UK with the E4M3 epilogue of the padded sparse MLA prefill query.

CPU: the reference on hand-checked values (the BF16 rounding of the product, then the attention op's conversion:
unit scale, nearest even, saturated to +-448), its padding heads, the guards. GPU (sm_107): every cell bit for bit
against the two-pass form the dep16 target ran before (``bmm_out`` into a BF16 buffer, then
``rubin_mla_ctx_q_fp8``), the RoPE columns of the full FP8 query buffer untouched; and against the reference within
one E4M3 code (the reference sums in another order).
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_mla_ctx_wuk_fp8 import (  # noqa: E501
    rubin_mla_ctx_wuk_fp8 as op,
)

__extra_import_path__ = ["../../../..", "."]
from _validating import certifying, validating  # noqa: E402,F401

on_sm107 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the Rubin family entries are certified on sm_107",
)

FP8 = torch.float8_e4m3fn

# --- CPU -------------------------------------------------------------------------------------


def test_reference_rounds_to_bf16_then_converts():
    # One NoPE column of 1 and a weight row of values: the products are the weights themselves.
    w = torch.tensor(
        [1.0, -1.5, 448.0, 480.0, 1.0e4, 2.0**-9, 2.0**-11, 0.328125], dtype=torch.float32
    )
    codes = torch.tensor([0x38, 0xBC, 0x7E, 0x7E, 0x7E, 0x01, 0x00, 0x2A], dtype=torch.uint8)
    nope, lora = 16, w.numel()
    q = torch.zeros(1, 1, nope, dtype=torch.bfloat16)
    q[0, 0, 0] = 1.0
    w_uk = torch.zeros(1, nope, lora, dtype=torch.bfloat16)
    w_uk[0, 0] = w.to(torch.bfloat16)
    dst = torch.full((1, 3, lora), 1.0).to(FP8)
    out = op.reference(dst, q, w_uk)
    assert torch.equal(out[0, 0].view(torch.uint8), codes)
    assert (out[:, 1:].view(torch.uint8) == 0).all(), "padding heads are zeros"
    assert (dst.view(torch.uint8) == 0x38).all(), "the reference leaves dst untouched"


def test_reference_rounds_the_sum_to_bf16_first():
    # 1 + 2^-9 is not a BF16 value: the BMM's output is 1.0, whose E4M3 code is 0x38 (the FP32 sum
    # would also give 0x38, so take a sum the BF16 rounding moves across an E4M3 tie: 1.0625 + 2^-9).
    nope = 16
    q = torch.zeros(1, 1, nope, dtype=torch.bfloat16)
    q[0, 0, :2] = 1.0
    w_uk = torch.zeros(1, nope, 1, dtype=torch.bfloat16)
    w_uk[0, 0, 0] = 1.0625
    w_uk[0, 1, 0] = 2.0**-9
    out = op.reference(torch.zeros(1, 1, 1, dtype=FP8), q, w_uk)
    # BF16(1.0625 + 2^-9) = 1.0625, a tie between E4M3 1.0 and 1.125: to the even code 1.0 (0x38).
    # Without the BF16 step the FP32 sum is above the tie and converts to 1.125 (0x39).
    assert int(out.view(torch.uint8)) == 0x38


def test_guards():
    q = torch.zeros(2, 4, 64, dtype=torch.bfloat16)
    w = torch.zeros(4, 64, 256, dtype=torch.bfloat16)
    dst = torch.zeros(2, 6, 256, dtype=FP8)
    op.is_valid(dst, q, w)
    with pytest.raises(AssertionError):
        op.is_valid(dst.view(torch.uint8), q, w)
    with pytest.raises(AssertionError):
        op.is_valid(dst, q.float(), w)
    with pytest.raises(AssertionError):
        op.is_valid(torch.zeros(2, 3, 256, dtype=FP8), q, w)
    with pytest.raises(AssertionError, match="64-wide"):
        op.is_valid(
            dst,
            torch.zeros(2, 4, 24, dtype=torch.bfloat16),
            torch.zeros(4, 24, 256, dtype=torch.bfloat16),
        )
    with pytest.raises(AssertionError, match="contiguous"):
        op.is_valid(dst, torch.zeros(2, 4, 128, dtype=torch.bfloat16)[..., ::2], w)


# --- GPU -------------------------------------------------------------------------------------


def _inputs(spec, gen, scale):
    """q ``[n, heads * (nope + rope)]`` as q_b writes it (q_nope a strided view), W_UK, and an FP8 query buffer
    ``[n, pad, lora + rope]`` holding a sentinel everywhere."""
    n, pad, heads = spec["n"], spec["pad"], spec["heads"]
    nope, rope, lora = spec["nope"], spec["rope"], spec["lora"]
    q = (torch.randn(n, heads * (nope + rope), generator=gen, device="cuda") * scale).to(
        torch.bfloat16
    )
    w_uk = (torch.randn(heads, nope, lora, generator=gen, device="cuda") * 0.05).to(torch.bfloat16)
    q_nope = q.view(n, heads, nope + rope)[..., :nope]
    quant_q = torch.full((n, pad, lora + rope), 0x5A, dtype=torch.uint8, device="cuda").view(FP8)
    return q_nope, w_uk, quant_q


def _two_pass(q_nope, w_uk, pad):
    """The dep16 target's FP8 context query before this entry: W_UK into a BF16 buffer, then its conversion."""
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.bmm_out import bmm_out
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_mla_ctx_q_fp8 import (  # noqa: E501
        rubin_mla_ctx_q_fp8,
    )

    n, heads, _ = q_nope.shape
    lora = w_uk.shape[2]
    fused_q = q_nope.new_empty(n, pad, lora + 64)
    bmm_out(q_nope.transpose(0, 1), w_uk, fused_q[:, :heads, :lora].transpose(0, 1))
    quant_q = torch.empty(n, pad, lora + 64, dtype=FP8, device=q_nope.device)
    rubin_mla_ctx_q_fp8(quant_q[..., :lora], fused_q[..., :lora], heads)
    return quant_q[..., :lora]


@on_sm107
@pytest.mark.parametrize("scale", (2.0, 40.0), ids=("typical", "saturating"))
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, scale, certifying):  # noqa: F811
    certifying(op)
    s = cell.spec
    q_nope, w_uk, quant_q = _inputs(s, torch.Generator(device="cuda").manual_seed(s["n"]), scale)
    lora = s["lora"]
    ref = op.reference(quant_q[..., :lora], q_nope, w_uk)
    old = _two_pass(q_nope, w_uk, s["pad"])
    out = op(quant_q[..., :lora], q_nope, w_uk)
    assert out.data_ptr() == quant_q.data_ptr()
    assert torch.equal(quant_q[..., :lora].view(torch.uint8), old.view(torch.uint8)), (
        "not the two-pass bytes"
    )
    assert (quant_q[..., lora:].view(torch.uint8) == 0x5A).all(), "the RoPE columns were written"
    # The reference sums in FP32 in another order: its BF16 rounding, so its code, may move by one.
    a, b = quant_q[..., :lora].float(), ref.float()
    assert torch.equal(a.isnan(), b.isnan())
    step = torch.maximum(a.abs(), b.abs()).mul(2.0**-3).clamp_min(2.0**-9)
    assert ((a - b).abs() <= step).all()
    assert (a != b).float().mean() < 1e-3


@on_sm107
def test_strided_query_and_guard():
    gen = torch.Generator(device="cuda").manual_seed(7)
    q_nope, w_uk, quant_q = _inputs(op.CELLS[2].spec, gen, 2.0)
    with validating(op):
        op(quant_q[..., :512], q_nope, w_uk)
    assert torch.equal(
        quant_q[..., :512].view(torch.uint8), _two_pass(q_nope, w_uk, 128).view(torch.uint8)
    )
