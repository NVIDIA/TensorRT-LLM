# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp vision_rope_2d entry.

`op.reference` shares no code with the entry: positions from nested loops,
angles in fp64 from `theta ** (-2j / rope_dim)`, rotation as a complex
multiply.
"""

import zlib
from typing import Tuple

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


def test_grid_placement_and_integer_dtype_do_not_change_the_result() -> None:
    """The target hands over an int64 grid on the device; the cells pass host grids."""
    grid = [(1, 16, 24), (1, 8, 12)]
    q, k = _operands(_num_tokens(grid), HEADS, HEADS, HEAD_DIM, torch.bfloat16, seed=29)
    host = torch.tensor(grid, dtype=torch.int32)
    q_host, k_host = op(q, k, host, MERGE, THETA)
    for other in (host.to("cuda"), host.to(torch.int64), host.to(torch.int64).to("cuda")):
        q_other, k_other = op(q, k, other, MERGE, THETA)
        assert torch.equal(q_host, q_other) and torch.equal(k_host, k_other)


def test_guard_refuses_a_one_token_k() -> None:
    """The hazard, unguarded: a [1, H, D] k broadcasts to [T, H, D]."""
    grid_thw = torch.tensor([(1, 4, 4)], dtype=torch.int32)
    q, k = _operands(16, HEADS, HEADS, HEAD_DIM, torch.bfloat16, seed=5)
    _, k_out = op(q, k[:1], grid_thw, MERGE, THETA)
    assert k_out.shape == (16, HEADS, HEAD_DIM)

    with pytest.raises(AssertionError, match="must agree on tokens and head_dim"):
        with validating(op):
            op(q, k[:1], grid_thw, MERGE, THETA)
