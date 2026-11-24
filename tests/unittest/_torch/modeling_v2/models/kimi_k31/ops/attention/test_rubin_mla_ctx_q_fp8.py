# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of rubin_mla_ctx_q_fp8, the FP8 latent columns of the padded sparse MLA prefill query.

CPU: the reference on hand-checked values (the attention op's conversion of a BF16 query: unit scale, nearest
even, saturated to +-448), its padding heads, the guards. GPU (sm_107): every cell bit for bit against the
reference, written into the latent columns of a full FP8 query buffer whose RoPE columns must stay as they were.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_mla_ctx_q_fp8 import (  # noqa: E501
    rubin_mla_ctx_q_fp8 as op,
)

__extra_import_path__ = ["../../../..", "."]
from _validating import certifying, validating  # noqa: E402,F401

on_sm107 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the Rubin family entries are certified on sm_107",
)

FP8 = torch.float8_e4m3fn

# BF16 inputs and the E4M3 bytes the attention op makes of them at unit scale.
_SPECIAL = (
    (0.0, 0x00),
    (-0.0, 0x80),
    (1.0, 0x38),
    (-1.5, 0xBC),
    (448.0, 0x7E),
    (-448.0, 0xFE),
    (480.0, 0x7E),  # saturated, not NaN
    (1.0e4, 0x7E),
    (-3.0e38, 0xFE),
    (float("inf"), 0x7E),
    (float("-inf"), 0xFE),
    (2.0**-9, 0x01),  # the smallest subnormal
    (2.0**-11, 0x00),  # below half of it: zero
    (0.3125, 0x2A),
    (0.328125, 0x2A),  # halfway between 0.3125 and 0.34375: to the even code
)


def _special_rows():
    vals = torch.tensor([v for v, _ in _SPECIAL], dtype=torch.float32).to(torch.bfloat16)
    codes = torch.tensor([c for _, c in _SPECIAL], dtype=torch.uint8)
    return vals, codes


# --- CPU -------------------------------------------------------------------------------------


def test_reference_converts_like_the_attention_op():
    vals, codes = _special_rows()
    assert torch.equal(
        vals.float()[:2].view(torch.int32), torch.tensor([0.0, -0.0]).view(torch.int32)
    )
    src = vals.view(1, 1, -1).expand(2, 3, -1).contiguous()
    dst = torch.full((2, 4, vals.numel()), 1.0).to(FP8)
    out = op.reference(dst, src, 3)
    assert torch.equal(out[:, :3].view(torch.uint8), codes.view(1, 1, -1).expand(2, 3, -1))
    assert (out[:, 3:].view(torch.uint8) == 0).all(), "padding heads are zeros"
    assert (dst.view(torch.uint8) == 0x38).all(), "the reference leaves dst untouched"


def test_reference_keeps_nan():
    src = torch.full((1, 1, 4), float("nan"), dtype=torch.bfloat16)
    out = op.reference(torch.zeros(1, 2, 4, dtype=FP8), src, 1)
    assert torch.isnan(out[:, :1].float()).all()


def test_reference_reads_only_the_real_heads():
    gen = torch.Generator().manual_seed(0)
    src = torch.randn(5, 8, 16, generator=gen).to(torch.bfloat16)
    dst = torch.zeros(5, 8, 16, dtype=FP8)
    a = op.reference(dst, src, 6)
    src[:, 6:] = float("nan")
    b = op.reference(dst, src, 6)
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def test_guards():
    src = torch.zeros(2, 4, 16, dtype=torch.bfloat16)
    dst = torch.zeros(2, 4, 16, dtype=FP8)
    op.is_valid(dst, src, 4)
    with pytest.raises(AssertionError):
        op.is_valid(dst.view(torch.uint8), src, 4)
    with pytest.raises(AssertionError):
        op.is_valid(dst, src.float(), 4)
    with pytest.raises(AssertionError):
        op.is_valid(dst, src, 5)
    with pytest.raises(AssertionError, match="power of two"):
        op.is_valid(
            torch.zeros(2, 4, 12, dtype=FP8), torch.zeros(2, 4, 12, dtype=torch.bfloat16), 4
        )
    with pytest.raises(AssertionError, match="contiguous"):
        op.is_valid(dst, torch.zeros(2, 4, 32, dtype=torch.bfloat16)[..., ::2], 4)


# --- GPU -------------------------------------------------------------------------------------


def _inputs(spec, gen):
    """The BMM's BF16 buffer [n, pad, width] (NaN beyond the real heads' latent columns) with specials spread
    over the real heads, and an FP8 query buffer [n, pad, width] holding a sentinel everywhere."""
    n, pad, heads, d, width = spec["n"], spec["pad"], spec["heads"], spec["d"], spec["width"]
    src = torch.randn(n, pad, width, generator=gen, device="cuda").mul_(96.0).to(torch.bfloat16)
    vals, _ = _special_rows()
    real = src[:, :heads, :d].reshape(-1)
    pos = torch.randint(0, real.numel(), (min(4096, real.numel()),), generator=gen, device="cuda")
    real[pos] = vals.cuda()[torch.arange(pos.numel(), device="cuda") % vals.numel()]
    src[:, :heads, :d] = real.view(n, heads, d)
    src[:, heads:] = float("nan")
    src[:, :heads, d:] = float("nan")
    return src, torch.full((n, pad, width), 0x5A, dtype=torch.uint8, device="cuda").view(FP8)


@on_sm107
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    s = cell.spec
    src, quant_q = _inputs(s, torch.Generator(device="cuda").manual_seed(s["n"]))
    d = s["d"]
    ref = op.reference(quant_q[..., :d], src[..., :d], s["heads"])
    out = op(quant_q[..., :d], src[..., :d], s["heads"])
    assert out.data_ptr() == quant_q.data_ptr()
    assert torch.equal(quant_q[..., :d].view(torch.uint8), ref.view(torch.uint8))
    assert (quant_q[..., d:].view(torch.uint8) == 0x5A).all(), "the RoPE columns were written"


@on_sm107
def test_nan_stays_nan():
    src = torch.full((3, 4, 64), float("nan"), dtype=torch.bfloat16, device="cuda")
    dst = torch.zeros(3, 4, 64, dtype=FP8, device="cuda")
    with validating(op):
        op(dst, src, 2)
    assert torch.isnan(dst[:, :2].float()).all() and (dst[:, 2:].view(torch.uint8) == 0).all()
