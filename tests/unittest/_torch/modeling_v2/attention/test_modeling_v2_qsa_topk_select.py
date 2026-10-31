# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qsa_topk_select catalog entry.

The op selects; it computes nothing. `_compare` therefore checks the output's
structure exactly and the *multiset of selected values* bit for bit, never the
permutation, which the op leaves unspecified.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_topk_select import (  # noqa: E501
    TopkExpectation,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_topk_select import (  # noqa: E501
    qsa_topk_select as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "qsa_topk_select requires a CUDA device"


def _compare(out: torch.Tensor, ref: TopkExpectation) -> None:
    """Structure exactly, then the selected values bit for bit.

    A selection does no arithmetic, so nothing is tolerated. Values rather
    than offsets are compared so that a tied k-th score -- common here: the
    relu cells are half zeros -- is a pass rather than a coin flip.
    """
    assert out.dtype == torch.int32, out.dtype
    assert out.shape == ref.values.shape, (out.shape, ref.values.shape)
    index_topk = out.shape[1]
    valid = out >= 0
    assert not bool((valid[:, 1:] & ~valid[:, :-1]).any()), "a valid offset follows a -1 pad"
    assert bool((out >= -1).all()), "padding must be exactly -1"
    assert torch.equal(valid.sum(1), ref.window.clamp(max=index_topk)), (
        "each row must select exactly min(index_topk, window) offsets"
    )
    assert bool((out < ref.window[:, None]).all()), "an offset reaches past row_end"
    descending = torch.sort(out, dim=1, descending=True).values
    assert not bool(((descending[:, :-1] == descending[:, 1:]) & (descending[:, 1:] >= 0)).any()), (
        "duplicate offset within a row"
    )
    absolute = (out.long() + ref.row_starts[:, None]).clamp(0, ref.logits.shape[1] - 1)
    picked = torch.gather(ref.logits, 1, absolute).masked_fill(~valid, float("-inf"))
    got = torch.sort(picked, dim=1, descending=True).values
    assert torch.equal(got, ref.values), "selected values are not the window's top-k"


DEV = "cuda"
COMPRESS_RATIO = 4
BLOCK_TOPK = 512
#: Pre-fill for every output buffer, so a slot the op failed to write shows up
#: as an offset past `row_end` in `_compare`.
SENTINEL = 12345


def _logits(kind: str, rows: int, cols: int, gen: torch.Generator) -> torch.Tensor:
    x = torch.randn(rows, cols, generator=gen, device=DEV)
    # QSA's scores are relu(q.k) summed over heads: non-negative with ties at 0.
    return torch.relu(x) * 3.0 if kind == "relu" else x


def _build(spec: dict, seed: int):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    rows, cols, index_topk = spec["rows"], spec["cols"], spec["index_topk"]
    logits = _logits(spec["logits"], rows, cols, gen)

    kind, start = spec["windows"]
    if kind == "positions":
        positions = torch.tensor(start, device=DEV)
    else:
        assert kind == "chunk", kind
        positions = start + torch.arange(rows, device=DEV)
    assert positions.numel() == rows
    # QSA's per-row causal bound: complete compressed blocks at the position.
    row_ends = ((positions + 1) // COMPRESS_RATIO).clamp_max(cols).to(torch.int32)
    row_starts = torch.zeros(rows, dtype=torch.int32, device=DEV)

    if spec["tail"] == "nan":
        columns = torch.arange(cols, device=DEV)[None, :]
        outside = (columns < row_starts[:, None]) | (columns >= row_ends[:, None])
        logits.masked_fill_(outside, float("nan"))
    indices = torch.full((rows, index_topk), SENTINEL, dtype=torch.int32, device=DEV)
    return logits, row_starts, row_ends, indices, index_topk


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    args = _build(cell.spec, seed=sum(map(ord, cell.why)))
    out = op(*args)
    assert out is args[3], "the op writes `indices` and returns it"
    _compare(out, op.reference(*args))


#: (rows, score columns, replay window): tp1's 256-row verify step, then the unified
#: target's decode batches 1 and 128 and its 1024-row MTP7 verify step at the
#: 262144-token envelope's 65536 columns. Every capture runs full windows (the
#: histogram path); the replays move the bound, 128 rows down to the budget's
#: copy-through shortcut.
GRAPH_CASES = ((256, 2048, 1500), (1, 65536, 65535), (128, 65536, 511), (1024, 65536, 30000))


@pytest.mark.parametrize("rows, cols, replay_end", GRAPH_CASES)
def test_cuda_graph_capture_and_replay(rows: int, cols: int, replay_end: int) -> None:
    """Captured, then replayed against new scores and a new causal bound through the
    captured addresses."""
    gen = torch.Generator(device=DEV).manual_seed(8 + rows)
    logits = _logits("relu", rows, cols, gen)
    row_starts = torch.zeros(rows, dtype=torch.int32, device=DEV)
    row_ends = torch.full((rows,), cols, dtype=torch.int32, device=DEV)
    indices = torch.empty(rows, BLOCK_TOPK, dtype=torch.int32, device=DEV)

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        op(logits, row_starts, row_ends, indices, BLOCK_TOPK)
    torch.cuda.current_stream().wait_stream(warmup)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(logits, row_starts, row_ends, indices, BLOCK_TOPK)

    logits.copy_(_logits("relu", rows, cols, gen))
    row_ends.fill_(replay_end)
    indices.fill_(SENTINEL)
    graph.replay()
    torch.cuda.synchronize()
    _compare(indices, op.reference(logits, row_starts, row_ends, indices, BLOCK_TOPK))


def _layout_case():
    torch.manual_seed(10)
    rows, cols, top_k = 4, 128, 8
    logits = torch.randn(rows, cols, device=DEV)
    row_starts = torch.zeros(rows, dtype=torch.int32, device=DEV)
    row_ends = torch.full((rows,), cols, dtype=torch.int32, device=DEV)
    return logits, row_starts, row_ends, top_k


def test_guard_refuses_the_token_budget_default() -> None:
    """The default `index_topk` is 2048, the token budget; the target's buffer
    is 512 wide. That is the wider-buffer fault the other way round, and the
    guard catches it at the call that forgot to pass the block budget."""
    logits, row_starts, row_ends, _ = _layout_case()
    indices = torch.empty(logits.shape[0], BLOCK_TOPK, dtype=torch.int32, device=DEV)
    with pytest.raises(AssertionError, match="index_topk=2048"):
        with validating(op):
            op(logits, row_starts, row_ends, indices)
