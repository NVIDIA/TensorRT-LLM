# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the step5 vision_rope_2d_interleaved entry.

`op.reference` shares no code with the entry: angles from Python-float inverse
frequencies in fp64, the rotation as a complex128 multiply of each adjacent
pair. The test builds positions with nested loops over each tile, as the patch
embedding flattens it (row-major).
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.step5.step5_preview_nvfp4__sm_103__unified.ops.attention.vision_rope_2d_interleaved import (  # noqa: E501
    CROP,
    HEAD_DIM,
    NUM_HEADS,
    OVERVIEW,
    THETA,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.step5.step5_preview_nvfp4__sm_103__unified.ops.attention.vision_rope_2d_interleaved import (  # noqa: E501
    vision_rope_2d_interleaved as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "vision_rope_2d_interleaved requires a CUDA device"

#: The band. The entry and the reference round once each to bf16, from fp32
#: and from fp64. The fp32 value carries the rounding of cos/sin to fp32 and of
#: two products and a sum: at most about 3 * 2^-24 * (|x_2j| + |x_2j+1|), which
#: is below 2^-20 of the head's largest output magnitude (a rotation keeps each
#: pair's norm, so that magnitude is at least the pair's norm / sqrt 2). That
#: error moves the bf16 rounding by one ulp only where the exact value sits
#: within it of a rounding midpoint, and by more only where the pair cancels to
#: far below the head's scale. So each element must lie within one bf16 ulp of
#: its own reference value plus 2^-20 of its head's scale, and few may differ
#: at all: the miss rate of about 2^-22 / 2^-8 per element is gated at 1e-3.
SCALE_TERM = 2.0**-20
MAX_OFF_FRACTION = 1e-3


def _seed(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


def _positions(tiles, device="cuda", dtype=torch.int64) -> torch.Tensor:
    """`[T, 2]` (row, col) of every patch, tile after tile, row-major inside one."""
    coords = [(r, c) for rows, cols in tiles for r in range(rows) for c in range(cols)]
    return torch.tensor(coords, dtype=dtype, device=device)


def _operands(tokens: int, layout: str, dtype: torch.dtype, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    width = NUM_HEADS * HEAD_DIM
    if layout == "packed_qkv":
        qkv = torch.randn(tokens, 3 * width, generator=g, device="cuda").to(dtype)
        q, k, _ = qkv.chunk(3, dim=-1)
        q = q.view(tokens, NUM_HEADS, HEAD_DIM)
        k = k.view(tokens, NUM_HEADS, HEAD_DIM)
        assert not q.is_contiguous() and q.stride(-1) == 1
        return q, k
    assert layout == "separate", layout
    q = torch.randn(tokens, NUM_HEADS, HEAD_DIM, generator=g, device="cuda").to(dtype)
    k = torch.randn(tokens, NUM_HEADS, HEAD_DIM, generator=g, device="cuda").to(dtype)
    return q, k


def _build(spec, seed):
    positions = _positions(spec["tiles"])
    q, k = _operands(positions.shape[0], spec["layout"], spec["dtype"], seed)
    return q, k, positions, THETA


def _ordered(t: torch.Tensor) -> torch.Tensor:
    """bf16 bit patterns as integers ordered like the values they encode."""
    bits = t.contiguous().view(torch.int16).int()
    return torch.where(bits < 0, -(bits & 0x7FFF), bits)


def _bf16_ulp(t: torch.Tensor) -> torch.Tensor:
    """One bf16 ulp at each element's magnitude (normal range)."""
    mag = t.float().abs().clamp_min(2.0**-126)
    return torch.exp2(torch.floor(torch.log2(mag)) - 7.0)


def _measure(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float]:
    """(worst excess over the band, off fraction, max ulp distance)."""
    assert out.shape == ref.shape and out.dtype == ref.dtype == torch.bfloat16
    diff = (out.float() - ref.float()).abs()
    scale = ref.float().abs().amax(dim=-1, keepdim=True)
    allowed = _bf16_ulp(ref) + SCALE_TERM * scale
    excess = (diff - allowed).max().item()
    dist = (_ordered(out) - _ordered(ref)).abs()
    return excess, (dist > 0).float().mean().item(), dist.max().item()


def _compare(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """Each element within one bf16 ulp of its reference plus `SCALE_TERM` of
    its head's largest magnitude; at most `MAX_OFF_FRACTION` of them off at all
    (derivation at `SCALE_TERM`)."""
    assert not torch.isnan(out).any()
    excess, off, worst = _measure(out, ref)
    assert excess <= 0.0, f"an element exceeds the band by {excess:.3e}"
    assert off <= MAX_OFF_FRACTION, f"{off:.3e} of elements off the reference"
    return off, worst


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    q, k, positions, theta = _build(cell.spec, _seed(cell.why))
    q_before, k_before = q.clone(), k.clone()
    q_out, k_out = op(q, k, positions, theta)
    for got, given in ((q_out, q), (k_out, k)):
        assert got.shape == given.shape and got.dtype == given.dtype
        assert got.is_contiguous() and got.data_ptr() != given.data_ptr()
    assert torch.equal(q, q_before) and torch.equal(k, k_before), "inputs must not be mutated"
    q_ref, k_ref = op.reference(q, k, positions, theta)
    q_off, q_worst = _compare(q_out, q_ref)
    k_off, k_worst = _compare(k_out, k_ref)
    print(
        f"\n  {cell.why}: q off {q_off:.3e} (max {q_worst} ulp), "
        f"k off {k_off:.3e} (max {k_worst} ulp)"
    )


def test_int32_positions_give_the_same_result():
    """The cells pass int64 positions; int32 gives bit-identical outputs."""
    tiles = (CROP, OVERVIEW)
    q, k = _operands(sum(r * c for r, c in tiles), "packed_qkv", torch.bfloat16, 7)
    wide = op(q, k, _positions(tiles), THETA)
    narrow = op(q, k, _positions(tiles, dtype=torch.int32), THETA)
    assert all(torch.equal(a, b) for a, b in zip(wide, narrow))


def test_tiles_are_independent_of_their_neighbours():
    """A crop packed after the overview turns exactly as the crop alone."""
    tiles = (OVERVIEW, CROP)
    q, k = _operands(sum(r * c for r, c in tiles), "separate", torch.bfloat16, 11)
    packed_q, packed_k = op(q, k, _positions(tiles), THETA)
    rows = OVERVIEW[0] * OVERVIEW[1]
    alone_q, alone_k = op(q[rows:], k[rows:], _positions((CROP,)), THETA)
    assert torch.equal(packed_q[rows:], alone_q) and torch.equal(packed_k[rows:], alone_k)


def _neox(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    c, s = angles.cos()[:, None, :], angles.sin()[:, None, :]
    lo, hi = x[..., :half], x[..., half:]
    return torch.cat((lo * c - hi * s, hi * c + lo * s), dim=-1)


def _adjacent(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    c, s = angles.cos()[:, None, :], angles.sin()[:, None, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * c - odd * s, odd * c + even * s), dim=-1).flatten(-2)


def _variants64(x: torch.Tensor, positions: torch.Tensor) -> dict:
    """Wrong operators, in fp64: each lands far outside the fp32 deviation."""
    half = x.shape[-1] // 2
    n = half // 2
    j = torch.arange(n, dtype=torch.float64, device=x.device)
    inv_half = THETA ** (-2.0 * j / half)
    inv_head = THETA ** (-2.0 * j / x.shape[-1])
    pos = positions.double()
    row, col = pos[:, :1], pos[:, 1:]
    x64 = x.double()
    col_row = torch.cat((col * inv_half, row * inv_half), -1)
    return {
        "NeoX pairing (i, i + 48)": _neox(x64, col_row),
        "axes swapped (row angles first)": _adjacent(
            x64, torch.cat((row * inv_half, col * inv_half), -1)
        ),
        "inv_freq over head_dim (2i / 96)": _adjacent(
            x64, torch.cat((col * inv_head, row * inv_head), -1)
        ),
        "column angles on both halves (no row)": _adjacent(
            x64, torch.cat((col * inv_half, col * inv_half), -1)
        ),
    }


def test_identity_is_the_interleaved_column_first_rotation():
    """In fp32, where every variant is visible: the entry agrees with the fp64
    rotation to fp32 precision and each wrong operator lands far away.
    """
    positions = _positions((OVERVIEW,))
    q, _ = _operands(positions.shape[0], "separate", torch.float32, 13)
    out, _ = op(q, q, positions, THETA)
    ref, _ = op.reference(q.double(), q.double(), positions, THETA)
    correct = (out.double() - ref).abs().max().item()
    print(f"\n  fp32: correct-path max deviation {correct:.3e}")
    assert correct < 1e-5, correct
    for name, wrong in _variants64(q, positions).items():
        dev = (out.double() - wrong).abs().max().item()
        print(f"  {name:40s} max dev {dev:.4e} ({dev / correct:.3g}x correct)")
        assert dev > 1e4 * correct, (name, dev, correct)


def _stepfun_bf16(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """The rotation as StepFun's inference code runs it on bf16 activations:
    fp32 angles cast to bf16, cos/sin and the rotation in bf16."""
    half = x.shape[-1] // 2
    inv = 1.0 / (THETA ** (torch.arange(0, half, 2, device=x.device).float() / half))
    pos = positions.float()
    freqs_w = (pos[:, 1:] * inv).repeat_interleave(2, dim=-1)
    freqs_h = (pos[:, :1] * inv).repeat_interleave(2, dim=-1)
    freqs = torch.cat((freqs_w, freqs_h), -1).to(x.dtype)[:, None, :]
    pairs = x.unflatten(-1, (half, 2))
    turned = torch.stack((-pairs[..., 1], pairs[..., 0]), -1).flatten(-2)
    return x * freqs.cos() + turned * freqs.sin()


def test_bf16_angle_table_path_is_outside_the_band():
    """What the bf16-rounded angle table costs, measured against the fp64
    rotation; the entry's band refuses it, so the entry is not that path.
    """
    positions = _positions((OVERVIEW,))
    q, k = _operands(positions.shape[0], "packed_qkv", torch.bfloat16, 17)
    ref, _ = op.reference(q, k, positions, THETA)
    stepfun = _stepfun_bf16(q, positions)
    entry, _ = op(q, k, positions, THETA)
    excess, off, worst = _measure(stepfun, ref)
    dev = (stepfun.float() - ref.float()).abs()
    rel = (dev.pow(2).mean().sqrt() / ref.float().pow(2).mean().sqrt()).item()
    entry_dev = (entry.float() - ref.float()).abs().max().item()
    print(
        f"\n  bf16 angle table: max dev {dev.max().item():.4e}, relative RMS {rel:.3e}, "
        f"off fraction {off:.3e}, max {worst} ulp; entry max dev {entry_dev:.4e}"
    )
    with pytest.raises(AssertionError):
        _compare(stepfun, ref)


def test_guard_refuses_a_one_row_k():
    """The hazard, unguarded: a [1, H, D] k broadcasts to [T, H, D]."""
    positions = _positions(((4, 4),))
    q, k = _operands(16, "separate", torch.bfloat16, 5)
    _, k_out = op(q, k[:1], positions, THETA)
    assert k_out.shape == (16, NUM_HEADS, HEAD_DIM)
    with pytest.raises(AssertionError, match="k carries 1 tokens"):
        with validating(op):
            op(q, k[:1], positions, THETA)


def test_guard_refuses_one_row_of_positions():
    """The hazard, unguarded: one (row, col) broadcasts to every token."""
    positions = _positions(((4, 4),))
    q, k = _operands(16, "separate", torch.bfloat16, 6)
    q_out, _ = op(q, k, positions[-1:], THETA)
    assert q_out.shape == (16, NUM_HEADS, HEAD_DIM)
    with pytest.raises(AssertionError, match="positions must be"):
        with validating(op):
            op(q, k, positions[-1:], THETA)
