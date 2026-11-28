# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the csa2_rope entry: the tail rotation against the reference's complex-exponential form.

Band: the op rotates with two fp32 products and a subtract per pair, the reference multiplies
complex64 numbers; they differ by at most an fp32 ulp of the larger term before the cast. For fp32
inputs that is the gate (4 fp32 ulp of the row's largest value); for bf16 the cast turns it into an
occasional one-bf16-ulp flip. bf16 keeps 7 explicit mantissa bits, so one flip of an element in the
binade of the row's largest value is up to 2^-7 of that binade, i.e. up to 2 units of 2^-8 against
the row's largest: the gate is 2 such units. The untouched head of each row must be bit-identical.
"""

import math

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_rope import (  # noqa: E501
    csa2_rope_table,
    csa2_rope_tail,
)

SEQ, RD = 4096, 64


def _table(original):
    base = 160000.0 if original else 10000.0
    return csa2_rope_table(RD, SEQ, original, base, 16, 32, 1, DEV)


@pytest.mark.parametrize(
    "cell", csa2_rope_tail.CELLS, ids=[c.why[:44] for c in csa2_rope_tail.CELLS]
)
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(csa2_rope_tail)
    s = cell.spec
    gen = gen_for(cell)
    shape = (s["n"], s["dim"]) if s["heads"] is None else (s["n"], s["heads"], s["dim"])
    x = torch.randn(shape, generator=gen, device=DEV).to(s["dtype"])
    pos = torch.randint(0, SEQ, (s["n"],), generator=gen, device=DEV, dtype=torch.int32)
    table = _table(s["original"])
    got = csa2_rope_tail(x, table, pos, RD, s["inverse"])
    ref = csa2_rope_tail.reference(x, table, pos, RD, s["inverse"])
    assert got.dtype == x.dtype and got.shape == x.shape
    assert torch.equal(got[..., :-RD], x[..., :-RD])
    if x.dtype == torch.float32:
        assert_within_ulp(got, ref, element_ulp=4.0, rms_ulp=1.0)
    else:
        assert_within_ulp(got, ref, element_ulp=2.0, rms_ulp=0.1)


def test_inverse_undoes_the_rotation():
    with validating(csa2_rope_tail):
        table = _table(65536)
        x = torch.randn(7, 8, 512, device=DEV)
        pos = torch.tensor([0, 5, 17, 64, 128, 255, 299], device=DEV, dtype=torch.int32)
        back = csa2_rope_tail(csa2_rope_tail(x, table, pos, RD), table, pos, RD, inverse=True)
    assert torch.allclose(back, x, atol=1e-4)


def _ref_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:

        def cd(r):
            return dim * math.log(original_seq_len / (r * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(cd(beta_fast)), 0)
        high = min(math.ceil(cd(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(
            0, 1
        )
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return torch.polar(torch.ones(seqlen, dim // 2), torch.outer(torch.arange(seqlen), freqs))


@pytest.mark.parametrize("original,base", [(0, 10000.0), (65536, 160000.0)])
def test_table_matches_reference_polar_form(original, base):
    tab = csa2_rope_table(RD, 300, original, base, 16, 32, 1, DEV)
    fc = _ref_freqs_cis(RD, 300, original, base, 16, 32, 1)
    assert torch.allclose(torch.view_as_complex(tab.cpu().contiguous()), fc, atol=2e-6)


@pytest.mark.parametrize(
    "x,pos,table",
    [
        (
            torch.zeros(2, 2, 2, 512),
            torch.zeros(2, dtype=torch.int32),
            torch.zeros(8, 32, 2),
        ),  # 4-D
        (
            torch.zeros(4, 512),
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(8, 32, 2),
        ),  # broadcast pos
        (torch.zeros(4, 512), torch.zeros(4), torch.zeros(8, 32, 2)),  # float positions
    ],
)
def test_rejections(x, pos, table):
    with pytest.raises(AssertionError):
        csa2_rope_tail.is_valid(x, table, pos, RD)


@pytest.mark.parametrize("compressed", [False, True])
def test_table_is_the_engines(compressed):
    """The table equals the engine's CSA2 `RotaryEmbedding.rotary_cos_sin` bit for bit, with the
    rope parameters `CSA2Attention.from_hf_config` sets (DeepSeek-V4.1: rd 64; window-only layers the plain
    theta-10000 table, compressed layers YaRN over theta 160000, factor 16, 65536 original positions, beta 32 / 1,
    mscale 1), at every position of its 1M-position table."""
    from tensorrt_llm._torch.attention.backends.interface import RopeParams
    from tensorrt_llm._torch.attention.rotary_embedding import RotaryEmbedding
    from tensorrt_llm.functional import RotaryScalingType

    seq = 1 << 20  # every position of the DeepSeek-V4.1 table (max_position_embeddings)
    rope = RopeParams()
    rope.dim, rope.max_positions, rope.duplicate_data = RD, seq, True
    if compressed:
        rope.theta, rope.scale_type, rope.scale = 160000.0, RotaryScalingType.yarn, 16.0
        rope.original_max_positions, rope.beta_fast, rope.beta_slow = 65536, 32, 1
        rope.mscale = rope.mscale_all_dim = 0.0
    else:
        rope.theta, rope.scale_type, rope.scale = 10000.0, RotaryScalingType.none, 1.0
    engine = RotaryEmbedding(rope, head_dim=RD, is_neox=False).rotary_cos_sin.reshape(
        seq, 2, RD // 2
    )
    ours = csa2_rope_table(
        RD, seq, 65536 if compressed else 0, 160000.0 if compressed else 10000.0, 16, 32, 1, DEV
    )
    assert torch.equal(ours.permute(0, 2, 1), engine.to(ours.device))


def test_zero_width_is_rejected():
    """`x[..., -0:]` is the whole head: a zero RoPE width is rejected before any slicing."""
    x = torch.zeros(2, 512, device=DEV)
    pos = torch.zeros(2, dtype=torch.int32, device=DEV)
    with pytest.raises(AssertionError, match="positive"):
        csa2_rope_tail.is_valid(x, torch.zeros(1, 0, 2, device=DEV), pos, 0)
