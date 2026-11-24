# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 indexer_topk_prefill entry (CUDA radix prefill top-k).

Each row's window holds distinct logits and everything outside it holds +inf, so a kernel that
reads outside its window selects a wrong column. Rows are compared as sets (the kernel's slot
order is atomic).
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.indexer_topk_prefill import (
    indexer_topk_prefill as op,
)

__extra_import_path__ = ["../../../..", "."]
import _k31_topk_inputs as inputs  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = inputs.on_sm103

#: Rows at or past this index of one call take the radix-sort kernel; the rows before it take
#: the insertion-sort one (the launch splits the rows, not the columns).
RADIX_ROW_START = 12288


def _build(spec, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    ks, ke, num_keys = inputs.chunk_windows(spec["requests"], spec["q_rows"])
    logits = inputs.distinct_logits(ks.shape[0], num_keys, gen, spec["logits_pad"])
    inputs.poison_outside(logits, ks, ke)
    indices = torch.full((ks.shape[0], spec["index_topk"]), -7, dtype=torch.int32, device="cuda")
    return dict(
        logits=logits, row_starts=ks, row_ends=ke, indices=indices, index_topk=spec["index_topk"]
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, inputs.seed_of(cell))
    op(**args)
    ref = torch.full_like(args["indices"], -9)
    op.reference(**dict(args, indices=ref))
    torch.cuda.synchronize()
    out = args["indices"]
    assert not bool((out == -7).any()), "the op left output slots unwritten"
    inputs.assert_same_sets(out, ref)
    # Short windows are written as 0 .. len - 1 in order, then -1.
    lengths = (args["row_ends"] - args["row_starts"]).long()
    short = lengths <= args["index_topk"]
    assert torch.equal(out[short], ref[short])


def test_rows_past_the_insertion_launch_take_radix_and_agree():
    """The 12800-row cell straddles the launch split: both kernels' rows match the reference."""
    cell = next(c for c in op.CELLS if c.spec["requests"] == ((0, 12800),))
    args = _build(cell.spec, inputs.seed_of(cell))
    op(**args)
    ref = torch.empty_like(args["indices"])
    op.reference(**dict(args, indices=ref))
    torch.cuda.synchronize()
    inputs.assert_same_sets(args["indices"][RADIX_ROW_START:], ref[RADIX_ROW_START:])
    inputs.assert_same_sets(args["indices"][:RADIX_ROW_START], ref[:RADIX_ROW_START])


def test_guard_refuses_a_wider_indices_buffer():
    args = _build(op.CELLS[2].spec, seed=1)
    wide = torch.empty(args["indices"].shape[0], 2056, dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="contiguous"):
        with validating(op):
            op(**dict(args, indices=wide))


def test_a_wider_indices_buffer_is_answered_wrongly():
    """What the guard stops: rows are written at r * index_topk, not at the buffer's stride."""
    args = _build(op.CELLS[2].spec, seed=2)
    rows = args["indices"].shape[0]
    wide = torch.full((rows, 2056), -7, dtype=torch.int32, device="cuda")
    op(**dict(args, indices=wide))
    ref = torch.empty_like(args["indices"])
    op.reference(**dict(args, indices=ref))
    torch.cuda.synchronize()
    assert not torch.equal(wide[:, :2048].sort(dim=1).values, ref.sort(dim=1).values)


def test_guard_refuses_a_window_past_the_row():
    args = _build(op.CELLS[2].spec, seed=3)
    ends = args["row_ends"].clone()
    ends[-1] = args["logits"].shape[1] + 1
    with pytest.raises(AssertionError, match="past the logits row"):
        with validating(op):
            op(**dict(args, row_ends=ends))


def test_guard_refuses_an_inverted_window():
    args = _build(op.CELLS[2].spec, seed=4)
    starts = args["row_starts"].clone()
    starts[0] = args["row_ends"][0] + 1
    with pytest.raises(AssertionError, match="starts after it ends"):
        with validating(op):
            op(**dict(args, row_starts=starts))
