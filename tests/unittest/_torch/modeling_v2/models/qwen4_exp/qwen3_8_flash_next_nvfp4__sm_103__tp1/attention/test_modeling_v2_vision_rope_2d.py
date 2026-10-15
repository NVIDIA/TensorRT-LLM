# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp vision_rope_2d entry.

`op.reference` shares no code with the entry: positions from nested loops,
angles in fp64 from `theta ** (-2j / rope_dim)`, rotation as a complex
multiply. Four ways of getting a 2-D vision rope subtly wrong are run too, so
the gate reads against the distance to a wrong answer.
"""

import zlib
from typing import List, Tuple

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.vision_rope_2d import (  # noqa: E501
    vision_rope_2d as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "vision_rope_2d requires a CUDA device"


def _compare(
    out: Tuple[torch.Tensor, torch.Tensor],
    ref: Tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Default band on each of `(q_out, k_out)`; nothing is loosened."""
    for got, want in zip(out, ref, strict=True):
        torch.testing.assert_close(got, want)


HEADS = 16
HEAD_DIM = 72
MERGE = 2
THETA = 10000.0


def _num_tokens(grid) -> int:
    return sum(t * h * w for t, h, w in grid)


def _operands(tokens, heads, heads_k, head_dim, dtype, seed, layout="separate"):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if layout == "packed_qkv":
        assert heads == heads_k
        qkv = torch.randn(tokens, 3 * heads, head_dim, generator=g, device="cuda").to(dtype)
        q, k, _ = torch.split(qkv, [heads] * 3, 1)
        assert not q.is_contiguous() and q.stride(-1) == 1
        return q, k
    q = torch.randn(tokens, heads, head_dim, generator=g, device="cuda").to(dtype)
    k = torch.randn(tokens, heads_k, head_dim, generator=g, device="cuda").to(dtype)
    return q, k


def _build(spec, seed):
    grid = spec["grid"]
    heads = spec.get("heads", HEADS)
    q, k = _operands(
        _num_tokens(grid),
        heads,
        spec.get("heads_k", heads),
        spec.get("head_dim", HEAD_DIM),
        spec["dtype"],
        seed,
        spec.get("layout", "separate"),
    )
    grid_thw = torch.tensor(
        [list(row) for row in grid],
        dtype=spec.get("grid_dtype", torch.int32),
        device=spec.get("grid_device", "cpu"),
    )
    return q, k, grid_thw, MERGE, spec.get("theta", THETA)


def _max_deviation(lhs, rhs) -> float:
    return (lhs.float() - rhs.float()).abs().max().item()


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    q, k, grid_thw, merge, theta = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    q_before, k_before = q.clone(), k.clone()
    q_out, k_out = op(q, k, grid_thw, merge, theta)
    assert q_out.shape == q.shape and q_out.dtype == q.dtype
    assert k_out.shape == k.shape and k_out.dtype == k.dtype
    assert torch.equal(q, q_before) and torch.equal(k, k_before), "inputs must not be mutated"
    assert q_out.data_ptr() != q.data_ptr() and k_out.data_ptr() != k.data_ptr()
    q_ref, k_ref = op.reference(q, k, grid_thw, merge, theta)
    print(
        f"[{cell.why}] q_dev={_max_deviation(q_out, q_ref):.3e} "
        f"k_dev={_max_deviation(k_out, k_ref):.3e}"
    )
    _compare((q_out, k_out), (q_ref, k_ref))


# --- test-local positions and angles, for the structural checks and the wrong
# operators below ------------------------------------------------------------


def _positions(grid, merge) -> List[Tuple[int, int]]:
    positions: List[Tuple[int, int]] = []
    for frames, height, width in grid:
        frame = []
        for block_row in range(height // merge):
            for block_col in range(width // merge):
                for intra_row in range(merge):
                    for intra_col in range(merge):
                        frame.append((block_row * merge + intra_row, block_col * merge + intra_col))
        positions.extend(frame * frames)
    return positions


def _angles(positions, rope_dim, theta, inv_scale_dim=None) -> torch.Tensor:
    """`[tokens, rope_dim]` fp32 on cuda; height half first, then width."""
    axis_dim = rope_dim // 2
    denom = rope_dim if inv_scale_dim is None else inv_scale_dim
    inverse = torch.tensor(
        [theta ** (-(2.0 * j) / denom) for j in range(axis_dim)], dtype=torch.float64
    )
    pos = torch.tensor(positions, dtype=torch.float64).reshape(-1, 2)
    return (pos[:, :, None] * inverse).flatten(1).to(device="cuda", dtype=torch.float32)


def _rotate_by_pairs(x, angles) -> torch.Tensor:
    rope_dim = angles.shape[-1]
    fp32 = x.float()
    pair = torch.complex(fp32[..., :rope_dim], fp32[..., rope_dim:])
    rotated = pair * torch.polar(torch.ones_like(angles), angles).unsqueeze(-2)
    return torch.cat((rotated.real, rotated.imag), dim=-1).to(x.dtype)


def _wrong_interleaved(q, grid):
    """GPT-J pairing: dims (2j, 2j+1) instead of (j, j + rope_dim)."""
    rope_dim = q.shape[-1] // 2
    angles = _angles(_positions(grid, MERGE), rope_dim, THETA)
    pair = torch.view_as_complex(q.float().unflatten(-1, (rope_dim, 2)).contiguous())
    rotor = torch.polar(torch.ones_like(angles), angles).unsqueeze(-2)
    return torch.view_as_real(pair * rotor).flatten(-2).to(q.dtype)


def _wrong_axes_swapped(q, grid):
    rope_dim = q.shape[-1] // 2
    axis_dim = rope_dim // 2
    angles = _angles(_positions(grid, MERGE), rope_dim, THETA)
    return _rotate_by_pairs(q, torch.cat((angles[:, axis_dim:], angles[:, :axis_dim]), dim=-1))


def _wrong_row_major(q, grid):
    positions = []
    for frames, height, width in grid:
        positions.extend([(r, c) for r in range(height) for c in range(width)] * frames)
    return _rotate_by_pairs(q, _angles(positions, q.shape[-1] // 2, THETA))


def _wrong_full_head_inv_freq(q, grid):
    head_dim = q.shape[-1]
    angles = _angles(_positions(grid, MERGE), head_dim // 2, THETA, inv_scale_dim=head_dim)
    return _rotate_by_pairs(q, angles)


WRONG_VARIANTS = {
    "interleaved_pairing": _wrong_interleaved,
    "axes_swapped": _wrong_axes_swapped,
    "row_major_positions": _wrong_row_major,
    "full_head_inv_freq": _wrong_full_head_inv_freq,
}


@pytest.mark.parametrize("variant", sorted(WRONG_VARIANTS))
def test_discrimination_against_wrong_operators(variant: str) -> None:
    """At 1024 bf16 tokens each wrong operator lands > 0.5 away (measured
    8.5-9.5) and > 100x the correct deviation; the entry's gate refuses it."""
    grid = [(1, 32, 32)]
    q, k = _operands(_num_tokens(grid), HEADS, HEADS, HEAD_DIM, torch.bfloat16, seed=23)
    grid_thw = torch.tensor(grid, dtype=torch.int32)

    q_out, k_out = op(q, k, grid_thw, MERGE, THETA)
    q_ref, k_ref = op.reference(q, k, grid_thw, MERGE, THETA)
    correct = _max_deviation(q_out, q_ref)
    wrong_q = WRONG_VARIANTS[variant](q, grid)
    wrong = _max_deviation(wrong_q, q_ref)
    print(f"[discrimination/{variant}] correct={correct:.3e} wrong={wrong:.3e}")
    assert wrong > 0.5, f"{variant} is only {wrong} away -- not a discriminating case"
    assert wrong > 100.0 * max(correct, 1e-6)
    with pytest.raises(AssertionError):
        _compare((wrong_q, k_out), (q_ref, k_ref))


def test_origin_token_is_the_identity() -> None:
    """Patch (0, 0) has every angle zero: bitwise unchanged. No other is."""
    grid = [(1, 4, 4)]
    q, k = _operands(16, HEADS, HEADS, HEAD_DIM, torch.bfloat16, seed=3)
    q_out, k_out = op(q, k, torch.tensor(grid, dtype=torch.int32), MERGE, THETA)
    assert torch.equal(q_out[0], q[0]) and torch.equal(k_out[0], k[0])
    assert not torch.equal(q_out[1:], q[1:])


def test_rotation_preserves_the_pair_norms() -> None:
    grid = [(1, 16, 24)]
    q, k = _operands(_num_tokens(grid), HEADS, HEADS, HEAD_DIM, torch.float32, seed=11)
    rope_dim = HEAD_DIM // 2
    q_out, k_out = op(q, k, torch.tensor(grid, dtype=torch.int32), MERGE, THETA)
    for before, after in ((q, q_out), (k, k_out)):
        pair_before = before[..., :rope_dim] ** 2 + before[..., rope_dim:] ** 2
        pair_after = after[..., :rope_dim] ** 2 + after[..., rope_dim:] ** 2
        torch.testing.assert_close(pair_after, pair_before, rtol=1e-5, atol=1e-5)


def test_height_and_width_occupy_separate_halves_of_the_angle_table() -> None:
    """A column-0 token moves only in the height lanes of both rotation halves,
    a row-0 token only in the width lanes."""
    grid = [(1, 4, 4)]
    q, k = _operands(16, HEADS, HEADS, HEAD_DIM, torch.float32, seed=17)
    rope_dim = HEAD_DIM // 2
    axis_dim = rope_dim // 2
    q_out, _ = op(q, k, torch.tensor(grid, dtype=torch.int32), MERGE, THETA)
    positions = _positions(grid, MERGE)
    row_only = [i for i, (r, c) in enumerate(positions) if r != 0 and c == 0]
    col_only = [i for i, (r, c) in enumerate(positions) if r == 0 and c != 0]
    assert row_only and col_only
    for t in row_only:
        torch.testing.assert_close(q_out[t, :, axis_dim:rope_dim], q[t, :, axis_dim:rope_dim])
        torch.testing.assert_close(
            q_out[t, :, rope_dim + axis_dim :], q[t, :, rope_dim + axis_dim :]
        )
        assert not torch.equal(q_out[t, :, :axis_dim], q[t, :, :axis_dim])
    for t in col_only:
        torch.testing.assert_close(q_out[t, :, :axis_dim], q[t, :, :axis_dim])
        torch.testing.assert_close(
            q_out[t, :, rope_dim : rope_dim + axis_dim], q[t, :, rope_dim : rope_dim + axis_dim]
        )
        assert not torch.equal(q_out[t, :, axis_dim:rope_dim], q[t, :, axis_dim:rope_dim])


def test_grid_placement_and_integer_dtype_do_not_change_the_result() -> None:
    grid = [(1, 16, 24), (1, 8, 12)]
    q, k = _operands(_num_tokens(grid), HEADS, HEADS, HEAD_DIM, torch.bfloat16, seed=29)
    host = torch.tensor(grid, dtype=torch.int32)
    q_host, k_host = op(q, k, host, MERGE, THETA)
    for other in (host.to("cuda"), host.to(torch.int64), host.to(torch.int64).to("cuda")):
        q_other, k_other = op(q, k, other, MERGE, THETA)
        assert torch.equal(q_host, q_other) and torch.equal(k_host, k_other)


def test_theta_changes_the_result() -> None:
    grid = [(1, 4, 4)]
    q, k = _operands(16, HEADS, HEADS, HEAD_DIM, torch.float32, seed=31)
    grid_thw = torch.tensor(grid, dtype=torch.int32)
    default, _ = op(q, k, grid_thw, MERGE, THETA)
    other, _ = op(q, k, grid_thw, MERGE, 1.0e6)
    assert not torch.equal(default, other)


def test_cuda_graph_replay_matches_eager() -> None:
    """Capturable; the grid is baked in at capture."""
    grid_thw = torch.tensor([(1, 16, 16)], dtype=torch.int32)
    q, k = _operands(256, HEADS, HEADS, HEAD_DIM, torch.bfloat16, seed=37)
    eager_q, eager_k = op(q, k, grid_thw, MERGE, THETA)

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            op(q, k, grid_thw, MERGE, THETA)
    torch.cuda.current_stream().wait_stream(warmup)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_q, captured_k = op(q, k, grid_thw, MERGE, THETA)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured_q, eager_q) and torch.equal(captured_k, eager_k)


# --- guards ------------------------------------------------------------------
#
# The indivisible-side and short-grid hazards are not driven unguarded here:
# the unwritten position rows are uninitialized memory, and an out-of-range
# value there trips a device-side assert in the angle gather that poisons the
# CUDA context for every later test in the process.


def test_guard_refuses_a_grid_side_not_divisible_by_the_merge() -> None:
    grid = [(1, 4, 5)]
    q, k = _operands(20, HEADS, HEADS, HEAD_DIM, torch.bfloat16, seed=2)
    with pytest.raises(AssertionError, match="divisible by spatial_merge_size"):
        with validating(op):
            op(q, k, torch.tensor(grid, dtype=torch.int32), MERGE, THETA)


def test_guard_refuses_a_token_count_that_does_not_match_the_grid() -> None:
    q, k = _operands(17, HEADS, HEADS, HEAD_DIM, torch.bfloat16, seed=2)
    with pytest.raises(AssertionError, match="describes 16 tokens but q/k carry 17"):
        with validating(op):
            op(q, k, torch.tensor([(1, 4, 4)], dtype=torch.int32), MERGE, THETA)
    q, k = _operands(64, HEADS, HEADS, HEAD_DIM, torch.bfloat16, seed=2)
    with pytest.raises(AssertionError, match="describes 1 tokens but q/k carry 64"):
        with validating(op):
            op(q, k, torch.tensor([(1, 1, 1)], dtype=torch.int32), 1, THETA)


def test_guard_refuses_a_one_token_k() -> None:
    """The hazard, unguarded: a [1, H, D] k broadcasts to [T, H, D]."""
    grid_thw = torch.tensor([(1, 4, 4)], dtype=torch.int32)
    q, k = _operands(16, HEADS, HEADS, HEAD_DIM, torch.bfloat16, seed=5)
    _, k_out = op(q, k[:1], grid_thw, MERGE, THETA)
    assert k_out.shape == (16, HEADS, HEAD_DIM)

    with pytest.raises(AssertionError, match="must agree on tokens and head_dim"):
        with validating(op):
            op(q, k[:1], grid_thw, MERGE, THETA)


def test_head_dim_not_divisible_by_4_is_loud() -> None:
    """Not guarded: the angle table comes out wider than the head and the
    broadcast raises."""
    grid_thw = torch.tensor([(1, 4, 4)], dtype=torch.int32)
    for head_dim in (70, 71):
        q, k = _operands(16, HEADS, HEADS, head_dim, torch.bfloat16, seed=7)
        with pytest.raises(RuntimeError):
            op(q, k, grid_thw, MERGE, THETA)
