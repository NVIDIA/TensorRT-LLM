# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 indexer_topk_decode entry (CUDA radix decode / verify top-k).

Each row's causal keys hold distinct logits and every column past the row's causal end holds
+inf, so a kernel that reads past it selects a wrong column. The split-work scratch is a row
slice of a larger [rows, 10, 2048] buffer, as the DSA metadata hands it over. Rows are compared
as sets (the kernel's slot order is atomic).
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.indexer_topk_decode import (
    indexer_topk_decode as op,
)

__extra_import_path__ = ["../../../..", "."]
import _k31_topk_inputs as inputs  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = inputs.on_sm103

#: Split-work blocks per row the caller's radix workspace holds (the DSA metadata allocates
#: [max rows, 10, index_topk]); the launch never uses more.
RADIX_BLOCKS_PER_ROW = 10


def _causal_lengths(seq_lens: torch.Tensor, next_n: int) -> torch.Tensor:
    rows = seq_lens.shape[0] * next_n
    r = torch.arange(rows, device=seq_lens.device)
    return seq_lens.long()[r // next_n] - next_n + r % next_n + 1


def _build(spec, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    batch, next_n, topk = spec["batch"], spec["next_n"], spec["index_topk"]
    lo, hi = spec["seq_lens"]
    seq_lens = torch.randint(lo, hi + 1, (batch,), generator=gen, dtype=torch.int32).cuda()
    rows = batch * next_n
    logits = inputs.distinct_logits(rows, spec["width"], gen, spec["row_pad"])
    lengths = _causal_lengths(seq_lens, next_n)
    inputs.poison_outside(logits, torch.zeros_like(lengths), lengths)
    shape = (rows + 3, RADIX_BLOCKS_PER_ROW, topk)
    aux_i = torch.full(shape, -5, dtype=torch.int32, device="cuda")
    aux_l = torch.full(shape, float("nan"), dtype=torch.float32, device="cuda")
    return dict(
        logits=logits,
        seq_lens=seq_lens,
        indices=torch.full((rows, topk), -7, dtype=torch.int32, device="cuda"),
        next_n=next_n,
        index_topk=topk,
        compress_ratio=spec["compress_ratio"],
        radix_aux_indices=aux_i[:rows],
        radix_aux_logits=aux_l[:rows],
        row_kv_lens=None,
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


def test_two_calls_select_the_same_sets():
    """The slot order may change between calls; the selected sets do not."""
    cell = next(c for c in op.CELLS if c.spec["batch"] == 64)
    args = _build(cell.spec, inputs.seed_of(cell))
    op(**args)
    first = args["indices"].clone()
    op(**args)
    torch.cuda.synchronize()
    inputs.assert_same_sets(args["indices"], first)


def test_guard_refuses_a_wider_indices_buffer():
    args = _build(op.CELLS[3].spec, seed=1)
    wide = torch.empty(args["indices"].shape[0], 2056, dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match=r"indices must be \[rows, 2048\]"):
        with validating(op):
            op(**dict(args, indices=wide))


def test_guard_refuses_keys_past_the_logits_row():
    args = _build(op.CELLS[3].spec, seed=2)
    seq_lens = args["seq_lens"].clone()
    seq_lens[0] = args["logits"].shape[1] + 1
    with pytest.raises(AssertionError, match="past the logits row"):
        with validating(op):
            op(**dict(args, seq_lens=seq_lens))
