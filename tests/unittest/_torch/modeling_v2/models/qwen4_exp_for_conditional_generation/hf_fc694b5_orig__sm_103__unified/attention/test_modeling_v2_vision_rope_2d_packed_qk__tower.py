# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the vision_rope_2d_packed_qk__tower variant: the vision tower's 2-D rotary,
one Triton launch in place on the q|k prefix of the packed qkv projection.

Each cell builds a packed `[T, 3 * 16, 72]` bf16 qkv projection and the caller's angle
table from the cell's grid (spatial-merge positions, fp64 angles rounded once to fp32,
fp32 cos / sin), rotates `qkv[:, :32]` in place, and gates:
  1. bitwise against the entry's `reference` on the same table;
  2. bitwise against the whole-op reference of `vision_rope_2d__r52` (copied below:
     positions by nested loops, fp64 angles, `polar`, complex multiply), q and k
     separately, with that entry's distribution gates beside it;
  3. the v heads of every row untouched.
Each case `is_valid` refuses is first driven unguarded to show the silent answer.
"""

import zlib
from typing import List, Sequence, Tuple

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.attention.vision_rope_2d_packed_qk__tower import (  # noqa: E501
    HEAD_DIM,
    NUM_HEADS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.attention.vision_rope_2d_packed_qk__tower import (  # noqa: E501
    vision_rope_2d_packed_qk as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "vision_rope_2d_packed_qk requires a CUDA device"

MERGE = 2
THETA = 10000.0
HALF = HEAD_DIM // 2


def _num_tokens(grid) -> int:
    return sum(t * h * w for t, h, w in grid)


def _spatial_merge_positions(grid: Sequence[Sequence[int]], merge: int) -> List[Tuple[int, int]]:
    """Per-token `(row, col)` by four nested loops, in spatial-merge order."""
    positions: List[Tuple[int, int]] = []
    for frames, height, width in grid:
        frame = [
            (block_row * merge + r, block_col * merge + c)
            for block_row in range(height // merge)
            for block_col in range(width // merge)
            for r in range(merge)
            for c in range(merge)
        ]
        positions.extend(frame * frames)
    return positions


def _angle_table(grid_thw: torch.Tensor, merge: int, theta: float, head_dim: int):
    """The caller's glue: per-token fp32 `cos` / `sin` `[T, head_dim // 2]`, height angles
    then width, from fp64 angles rounded once to fp32."""
    rope_dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float64) / rope_dim))
    pos = torch.tensor(_spatial_merge_positions(grid_thw.tolist(), merge), dtype=torch.float64)
    angles = (pos[:, :, None] * inv_freq).flatten(1).to(device="cuda", dtype=torch.float32)
    return angles.cos().contiguous(), angles.sin().contiguous()


def _r52_reference(q, k, grid_thw, spatial_merge_size, theta=THETA):
    """`vision_rope_2d__r52.reference` (k52's commit 8b3806d4da), verbatim: the whole op from
    the grid, as a fresh `(q_out, k_out)`."""
    rope_dim = q.shape[-1] // 2
    axis_dim = rope_dim // 2
    positions = _spatial_merge_positions(grid_thw.tolist(), spatial_merge_size)
    inverse = torch.tensor(
        [theta ** (-(2.0 * j) / rope_dim) for j in range(axis_dim)], dtype=torch.float64
    )
    pos = torch.tensor(positions, dtype=torch.float64).reshape(-1, 2)
    angles = (pos[:, :, None] * inverse).flatten(1).to(device=q.device, dtype=torch.float32)
    rotor = torch.polar(torch.ones_like(angles), angles).unsqueeze(-2)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        fp32 = x.float()
        rotated = torch.complex(fp32[..., :rope_dim], fp32[..., rope_dim:]) * rotor
        return torch.cat((rotated.real, rotated.imag), dim=-1).to(x.dtype)

    return rotate(q), rotate(k)


def _packed_qkv(tokens: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(tokens, 3 * NUM_HEADS, HEAD_DIM, generator=g, device="cuda").to(dtype)


def _qk(qkv: torch.Tensor) -> torch.Tensor:
    qk = qkv[:, : 2 * NUM_HEADS]
    assert qk.stride() == (3 * NUM_HEADS * HEAD_DIM, HEAD_DIM, 1) and not qk.is_contiguous()
    return qk


def _distribution(got: torch.Tensor, want: torch.Tensor) -> Tuple[float, float, float]:
    """(cosine, relative L2, p99.9 absolute error) of `got` against `want`, flattened."""
    g, w = got.float().reshape(-1), want.float().reshape(-1)
    cos = torch.nn.functional.cosine_similarity(g, w, dim=0).item()
    rel_l2 = ((g - w).norm() / w.norm().clamp_min(1e-30)).item()
    err = (g - w).abs()
    # torch.quantile caps its input size; a strided sample of 2^24 is plenty for p99.9.
    if err.numel() > 2**24:
        err = err[:: -(-err.numel() // 2**24)]
    return cos, rel_l2, torch.quantile(err, 0.999).item()


#: `vision_rope_2d__r52`'s distribution gates: both sides round the same fp32 rotation to bf16
#: once, so they differ by at most one bf16 ulp per element, at most 2^-7 of its value: the
#: relative L2 is bounded by that, and 1 - cosine by half its square.
_REL_L2_BOUND = 2.0**-7
_COS_BOUND = 1.0 - 2.0**-15


def _mismatch(got: torch.Tensor, want: torch.Tensor) -> Tuple[int, float]:
    """(elements differing in their bits, largest absolute difference)."""
    differing = (got.reshape(-1).view(torch.int16) != want.reshape(-1).view(torch.int16)).sum()
    return int(differing.item()), (got.float() - want.float()).abs().max().item()


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    grid_thw = torch.tensor([list(row) for row in spec["grid"]], dtype=spec["grid_dtype"])
    tokens = _num_tokens(spec["grid"])
    qkv = _packed_qkv(tokens, spec["dtype"], seed=zlib.crc32(cell.why.encode()) % 2**31)
    before = qkv.clone()
    cos, sin = _angle_table(grid_thw, MERGE, THETA, HEAD_DIM)
    qk = _qk(qkv)

    want = op.reference(qk, cos, sin)
    assert torch.equal(qkv, before), "reference must not mutate its input"
    out = op(qk, cos, sin)
    torch.cuda.synchronize()

    assert out is qk, "the op returns the view it rotated"
    assert torch.equal(qkv[:, 2 * NUM_HEADS :], before[:, 2 * NUM_HEADS :]), "v was written"
    assert not torch.equal(qk, before[:, : 2 * NUM_HEADS]), "q|k was not rotated"

    # 1. The entry's own reference on the same table: bitwise.
    differing, max_dev = _mismatch(qk, want)
    print(f"[{cell.why}] tokens={tokens} vs reference: {differing} differing, max {max_dev:.3e}")
    assert differing == 0, f"{differing} elements differ from reference (max {max_dev:.3e})"

    # 2. vision_rope_2d__r52's whole-op reference from the grid: bitwise, plus its
    # distribution gates.
    q_r52, k_r52 = _r52_reference(
        before[:, :NUM_HEADS], before[:, NUM_HEADS : 2 * NUM_HEADS], grid_thw, MERGE, THETA
    )
    for name, got, ref in (
        ("q", qkv[:, :NUM_HEADS], q_r52),
        ("k", qkv[:, NUM_HEADS : 2 * NUM_HEADS], k_r52),
    ):
        differing, max_dev = _mismatch(got, ref)
        cos_sim, rel_l2, p999 = _distribution(got, ref)
        print(
            f"[{cell.why}] {name} vs r52: {differing} differing, max {max_dev:.3e}, "
            f"cos={cos_sim:.9f} rel_l2={rel_l2:.3e} p99.9={p999:.3e}"
        )
        assert cos_sim >= _COS_BOUND and rel_l2 <= _REL_L2_BOUND, (name, cos_sim, rel_l2)
        assert differing == 0, f"{name}: {differing} elements differ from r52 (max {max_dev:.3e})"


def _small():
    grid_thw = torch.tensor([(1, 8, 12)], dtype=torch.int64)
    qkv = _packed_qkv(96, torch.bfloat16, seed=17)
    cos, sin = _angle_table(grid_thw, MERGE, THETA, HEAD_DIM)
    return qkv, cos, sin


def test_guard_refuses_a_bf16_table() -> None:
    """Unguarded, a bf16 table is widened in the kernel and rotates by rounded angles."""
    qkv, cos, sin = _small()
    want = op.reference(_qk(qkv), cos, sin)
    out = op(_qk(qkv.clone()), cos.to(torch.bfloat16), sin.to(torch.bfloat16))
    differing, max_dev = _mismatch(out, want)
    print(f"bf16 table, unguarded: {differing} differing, max {max_dev:.3e}")
    assert differing > 0, "the hazard this guard covers is gone"

    with pytest.raises(AssertionError, match="must be fp32"):
        with validating(op):
            op(_qk(qkv), cos.to(torch.bfloat16), sin.to(torch.bfloat16))


def test_guard_refuses_a_table_short_of_the_tokens() -> None:
    """Unguarded, rows past the table load cos 1 / sin 0: the tail comes back unrotated."""
    qkv, cos, sin = _small()
    before = qkv.clone()
    out = op(_qk(qkv), cos[:64], sin[:64])
    torch.cuda.synchronize()
    assert torch.equal(out[64:], before[64:, : 2 * NUM_HEADS]), "the tail was rotated"
    assert not torch.equal(out[:64], before[:64, : 2 * NUM_HEADS])

    qkv = before.clone()
    with pytest.raises(AssertionError, match=r"\[T, head_dim // 2\]"):
        with validating(op):
            op(_qk(qkv), cos[:64], sin[:64])


def test_guard_refuses_a_narrow_table() -> None:
    """Unguarded, a [T, 32] table rotates the first 64 dims of each head as pairs (i, i + 32)
    and leaves the last 8 untouched: partial rotary, not this tower's full-head pairing."""
    qkv, cos, sin = _small()
    before = qkv.clone()
    out = op(_qk(qkv), cos[:, :32].contiguous(), sin[:, :32].contiguous())
    torch.cuda.synchronize()
    assert torch.equal(out[..., 64:], before[:, : 2 * NUM_HEADS, 64:]), "dims past 64 were written"
    assert not torch.equal(out[..., :64], before[:, : 2 * NUM_HEADS, :64])

    with pytest.raises(AssertionError, match=r"\[T, head_dim // 2\]"):
        with validating(op):
            op(_qk(before.clone()), cos[:, :32].contiguous(), sin[:, :32].contiguous())


def test_guard_refuses_a_strided_table() -> None:
    """Unguarded, the first half of a duplicated [T, 72] table is indexed as t * 36 + i and
    reads the wrong rows' angles."""
    qkv, cos, sin = _small()
    cos_wide, sin_wide = torch.cat((cos, cos), dim=-1), torch.cat((sin, sin), dim=-1)
    cos_view, sin_view = cos_wide[:, :HALF], sin_wide[:, :HALF]
    assert torch.equal(cos_view, cos) and not cos_view.is_contiguous()
    want = op.reference(_qk(qkv), cos, sin)
    out = op(_qk(qkv.clone()), cos_view, sin_view)
    differing, max_dev = _mismatch(out, want)
    print(f"strided table, unguarded: {differing} differing, max {max_dev:.3e}")
    assert differing > 0, "the hazard this guard covers is gone"

    with pytest.raises(AssertionError, match="must be contiguous"):
        with validating(op):
            op(_qk(qkv), cos_view, sin_view)
