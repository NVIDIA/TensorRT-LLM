# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MXFP4 rounding of the FP4 index queries and keys: torch.ops.trtllm.fused_cat_fp4 (the only rounding the DSA
indexer uses) against an exact MXFP4 reference with ties to even. The two agree everywhere except at the exact
midpoints that ties-to-even rounds up and in the scale byte of all-zero blocks."""

import pytest
import torch

from tensorrt_llm._utils import get_sm_version

skip_no_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() < 100,
    reason="needs an SM100+ GPU (fused_cat_fp4)",
)

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _ref(x: torch.Tensor):
    """Exact reference on [M, 128] BF16: (codes uint8 [M, 128], ue8m0 uint8 [M, 4])."""
    xf = x.float().view(-1, 4, 32)
    amax = xf.abs().amax(-1)
    ratio = torch.clamp(amax, min=6.0 * 2.0**-126) / 6.0
    mant, ex = torch.frexp(ratio)  # ratio = mant * 2^ex, mant in [0.5, 1)
    log2_scale = torch.where(mant == 0.5, ex - 1, ex).clamp(-127, 127)
    scaled = xf * torch.exp2(-log2_scale.float())[..., None]
    a = scaled.abs()
    idx = sum(
        (a >= t if even_above else a > t).int()
        for t, even_above in (
            (0.25, False),
            (0.75, True),
            (1.25, False),
            (1.75, True),
            (2.5, False),
            (3.5, True),
            (5.0, False),
        )
    )
    sign = torch.signbit(scaled).int()
    codes = (idx | (sign << 3)).to(torch.uint8).view(x.shape[0], 128)
    return codes, (log2_scale + 127).to(torch.uint8)


def _unpack(packed: torch.Tensor, scale: torch.Tensor):
    """(codes uint8 [M, 128], ue8m0 uint8 [M, 4]) of a packed [M, 64] / [M, 1] int32 pair."""
    p = packed.view(torch.uint8)
    codes = torch.stack([p & 0xF, p >> 4], dim=-1).view(p.shape[0], 128)
    return codes, scale.view(torch.uint8).view(-1, 4)


def _decode(codes: torch.Tensor, ue8m0: torch.Tensor) -> torch.Tensor:
    mag = _E2M1.to(codes.device)[(codes & 7).long()]
    val = torch.where((codes & 8) != 0, -mag, mag).view(-1, 4, 32)
    return (val * torch.exp2(ue8m0.float() - 127.0)[..., None]).view(codes.shape)


def _inputs(m: int, gen: torch.Generator) -> torch.Tensor:
    x = torch.randn(m, 128, generator=gen, device="cuda") * torch.exp2(
        torch.randint(-20, 8, (m, 1), generator=gen, device="cuda").float()
    )
    x = x.bfloat16()
    # Exact midpoints of every block scale (amax 6 * 2^k pins the scale to 2^k) and all-zero / tiny blocks.
    k = torch.randint(-10, 10, (m, 1), generator=gen, device="cuda").float()
    mids = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.75, -1.75, -3.5, 6.0], device="cuda"
    )
    rows = torch.arange(0, m, 3, device="cuda")
    tie = mids[
        torch.randint(0, mids.numel(), (rows.numel(), 31), generator=gen, device="cuda")
    ] * torch.exp2(k[rows])
    x[rows, 1:32] = tie.bfloat16()
    x[rows, 0] = (6.0 * torch.exp2(k[rows, 0])).bfloat16()
    x[1::7, 32:64] = 0
    x[2::11, 64:96] = 1e-30
    return x


@skip_no_sm100
def test_fused_cat_fp4_rounds_ties_toward_zero():
    """Only exact midpoints rounding up under ties-to-even (0.75, 1.75, 3.5 times the scale) and the scale byte of
    all-zero blocks differ from fused_cat_fp4; every decoded value differs by at most one E2M1 step."""
    import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm)

    gen = torch.Generator(device="cuda").manual_seed(3)
    x = _inputs(2048, gen)
    pe, nope = x[:, :64], x[:, 64:]
    codes, ue8m0 = _ref(x)
    tz_codes, tz_ue8m0 = _unpack(*torch.ops.trtllm.fused_cat_fp4(pe, nope))
    block_nonzero = x.float().view(-1, 4, 32).abs().amax(-1) > 1e-12
    assert torch.equal(ue8m0[block_nonzero], tz_ue8m0[block_nonzero])
    vals = _decode(codes, ue8m0).view(-1, 4, 32)[block_nonzero]
    tz_vals = _decode(tz_codes, tz_ue8m0).view(-1, 4, 32)[block_nonzero]
    diff = vals != tz_vals
    scaled = (x.float().view(-1, 4, 32) * torch.exp2(127.0 - ue8m0.float())[..., None])[
        block_nonzero
    ]
    up_ties = torch.isin(scaled.abs(), torch.tensor([0.75, 1.75, 3.5], device="cuda"))
    assert torch.equal(diff, up_ties)
    assert up_ties.any()
    assert (vals.abs() >= tz_vals.abs()).all()
