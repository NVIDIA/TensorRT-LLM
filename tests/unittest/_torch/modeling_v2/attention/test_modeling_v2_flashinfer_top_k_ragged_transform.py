# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the unified target's flashinfer_top_k_ragged_transform variant.

The entry returns the canonical selection, which fixes every slot of every row, so
`_compare` is index equality against the composite-key top-k `reference`. Every cell's
columns outside the windows are poisoned with NaN, +inf and 3e38.
"""

import zlib

import pytest
import torch
from flashinfer.topk import TopKTieBreak

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.flashinfer_top_k_ragged_transform import (  # noqa: E501
    flashinfer_top_k_ragged_transform as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "flashinfer_top_k_ragged_transform requires a CUDA device"

DEV = "cuda"
COMPRESS_RATIO = 4
#: Rows per pass when the test itself scans a `[rows, cols]` matrix.
_SCAN_ROWS = 512


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Index for index. The canonical row fixes every slot, so nothing is tolerated: a
    selection does no arithmetic."""
    assert out.dtype == torch.int32, out.dtype
    assert out.shape == ref.shape, (out.shape, ref.shape)
    wrong = (out != ref).any(1)
    if bool(wrong.any()):
        row = int(wrong.nonzero()[0, 0])
        raise AssertionError(
            f"{int(wrong.sum())} of {out.shape[0]} rows differ from the canonical "
            f"selection; first is row {row}: {out[row, :8].tolist()} vs {ref[row, :8].tolist()}"
        )


def _scores(kind: str, rows: int, cols: int, gen: torch.Generator) -> torch.Tensor:
    x = torch.relu(torch.randn(rows, cols, generator=gen, device=DEV)) * 3.0
    if kind == "relu":
        return x
    keep = {"relu_sparse": 0.2, "relu_rare": 0.01}[kind]
    return x * (torch.rand(rows, cols, generator=gen, device=DEV) < keep)


def _windows(spec: dict, rows: int, cols: int, gen: torch.Generator):
    """int32 (row_starts, lengths) for the spec's window placement."""
    kind, k = spec["windows"][0], spec["k"]
    starts = torch.zeros(rows, dtype=torch.int64, device=DEV)
    if kind == "ragged":
        starts = torch.randint(0, cols // 2 + 1, (rows,), generator=gen, device=DEV)
        room = cols - starts
        lengths = (torch.rand(rows, generator=gen, device=DEV) * (room + 1)).long()
        # Empty, one block, the budget, one past it, the rest of the table.
        edges = torch.tensor([0, 1, k, k + 1], device=DEV)
        lengths[: edges.numel()] = edges
        lengths[edges.numel()] = room[edges.numel()]
        return starts.int(), torch.minimum(lengths, room).int()
    last = COMPRESS_RATIO * cols - 1
    if kind == "edges":
        # Empty, one block, k - 1, k and k + 1 blocks, the full table; then random.
        edges = torch.tensor(
            [2, 3, COMPRESS_RATIO * k - 2, COMPRESS_RATIO * k - 1, COMPRESS_RATIO * k + 3, last],
            device=DEV,
        )
        rest = torch.randint(0, last + 1, (rows - edges.numel(),), generator=gen, device=DEV)
        positions = torch.cat([edges, rest])
    elif kind == "random":
        lo, hi = spec["windows"][1:]
        positions = torch.randint(lo, hi + 1, (rows,), generator=gen, device=DEV)
    elif kind == "verify":
        requests, width, lo, hi = spec["windows"][1:]
        assert rows == requests * width, (rows, requests, width)
        first = torch.randint(lo, hi + 1, (requests, 1), generator=gen, device=DEV)
        positions = (first + torch.arange(width, device=DEV)).reshape(-1)
    else:
        assert kind == "chunk", kind
        positions = spec["windows"][1] + torch.arange(rows, device=DEV)
    lengths = ((positions + 1) // COMPRESS_RATIO).clamp_max(cols)
    return starts.int(), lengths.int()


def _poison(scores: torch.Tensor, starts: torch.Tensor, lengths: torch.Tensor) -> None:
    """Values that would win any comparison they were allowed into, outside each window."""
    columns = torch.arange(scores.shape[1], device=DEV)[None, :]
    lo = starts.long()[:, None]
    outside = (columns < lo) | (columns >= lo + lengths.long()[:, None])
    pattern = torch.tensor([float("nan"), float("inf"), 3e38, float("nan")], device=DEV)
    scores.copy_(torch.where(outside, pattern[columns % 4].expand_as(scores), scores))


def _build(spec: dict, rows: int, seed: int):
    """`(args, kwargs)` of one call at `spec`, `rows` rows."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    cols = spec["cols"]
    scores = _scores(spec["scores"], rows, cols, gen).to(spec["dtype"])
    starts, lengths = _windows(spec, rows, cols, gen)
    assert spec["tail"] == "poison", spec["tail"]
    _poison(scores, starts, lengths)
    assert spec["offsets"] == "zero", spec["offsets"]
    offsets = torch.zeros(rows, dtype=torch.int32, device=DEV)
    kwargs = dict(
        deterministic=spec["deterministic"],
        tie_break=spec["tie_break"],
        dsa_graph_safe=spec["dsa_graph_safe"],
        row_starts=starts,
    )
    return (scores, offsets, lengths, spec["k"]), kwargs


def _straddles(scores, starts, lengths, k) -> torch.Tensor:
    """Per row: whether the tie class at the window's k-th value straddles the budget --
    fewer than k columns above it, more than k at or above it -- so the tie-break, not the
    scores, decides which columns are kept."""
    rows, cols = scores.shape
    columns = torch.arange(cols, device=DEV)[None, :]
    out = torch.empty(rows, dtype=torch.bool, device=DEV)
    for lo in range(0, rows, _SCAN_ROWS):
        hi = min(lo + _SCAN_ROWS, rows)
        first = starts[lo:hi].long()[:, None]
        inside = (columns >= first) & (columns < first + lengths[lo:hi].long()[:, None])
        windowed = torch.where(inside, scores[lo:hi].float(), float("-inf"))
        kth = torch.topk(windowed, min(k, cols), dim=1).values[:, -1:]
        above = (windowed > kth).sum(1)
        at_least = (windowed >= kth).sum(1)
        out[lo:hi] = (above < k) & (at_least > k)
    return out


def _spec(why_prefix: str) -> dict:
    return next(c.spec for c in op.CELLS if c.why.startswith(why_prefix))


#: The 128-row cell over the tune's table whose zero tie class straddles the budget.
_STRADDLE = "decode batch 128 at 65536 columns"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    spec = cell.spec
    args, kwargs = _build(spec, spec["rows"], zlib.crc32(cell.why.encode()) % 2**31)
    expected = op.reference(*args, **kwargs)
    if spec["scores"] == "relu_rare":
        # What these cells are for: the tie-break decides every row.
        assert bool(_straddles(args[0], kwargs["row_starts"], args[2], spec["k"]).all()), (
            "a relu_rare row's tie class no longer straddles the budget"
        )
    out = op(*args, **kwargs)
    _compare(out, expected)


def test_cuda_graph_replay_at_decode_batches() -> None:
    """The decode call site past 4096 columns runs inside the step's CUDA graph: one graph per
    batch (1 and 128 rows at 65536 columns) sharing a pool, replayed after every input is
    refilled, writes the canonical rows of the new inputs into the captured output."""
    spec = _spec(_STRADDLE)
    pool, graphs = None, {}
    for rows in (1, 128):
        args, kwargs = _build(spec, rows, 100 + rows)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                op(*args, **kwargs)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph) if pool is None else torch.cuda.graph(graph, pool=pool):
            out = op(*args, **kwargs)
        pool = pool if pool is not None else graph.pool()
        graphs[rows] = (graph, args, kwargs, out)
    for round_ in range(2):
        for rows in (128, 1):
            graph, args, kwargs, out = graphs[rows]
            fresh_args, fresh_kwargs = _build(spec, rows, 900 + 17 * round_ + rows)
            for buffer, value in zip(args[:3], fresh_args[:3]):
                buffer.copy_(value)
            kwargs["row_starts"].copy_(fresh_kwargs["row_starts"])
            graph.replay()
            torch.cuda.synchronize()
            _compare(out, op.reference(*fresh_args, **fresh_kwargs))


@pytest.mark.parametrize("tie_break", [TopKTieBreak.NONE, TopKTieBreak.SMALL], ids=str)
def test_guard_refuses_a_tie_break_other_than_large(tie_break) -> None:
    """Accepted, and a plausible selection of the right size -- but another part of the tie
    class at the k-th value than the canonical row's. Shown unguarded first, then refused."""
    spec = _spec(_STRADDLE)
    args, kwargs = _build(spec, spec["rows"], 31)
    expected = op.reference(*args, **kwargs)
    kwargs = dict(kwargs, tie_break=tie_break)
    out = op(*args, **kwargs)
    assert torch.equal((out >= 0).sum(1), (expected >= 0).sum(1)), "the row sizes changed"
    assert bool((out != expected).any()), f"tie_break {tie_break} now returns the canonical row"
    with pytest.raises(AssertionError, match="tie_break must be TopKTieBreak.LARGE"):
        with validating(op):
            op(*args, **kwargs)


def _ragged_call(rows: int = 64, seed: int = 41):
    spec = _spec("non-zero row starts")
    args, kwargs = _build(spec, rows, seed)
    scores, _, lengths, k = args
    # Non-zero offsets, so that a misread offset shows in the output.
    offsets = torch.arange(rows, dtype=torch.int32, device=DEV) * 5
    return (scores, offsets, lengths, k), kwargs


@pytest.mark.parametrize("name", ["offsets", "lengths", "row_starts"])
def test_guard_refuses_a_non_int32_index_tensor(name) -> None:
    """The binding reads `offsets`, `lengths` and `row_starts` as int32 data whatever their
    dtype: an int64 one is misread -- its high words become every other row's value --
    without an error. Shown against the same call at int32, then refused."""
    (scores, offsets, lengths, k), kwargs = _ragged_call()
    good = op(scores, offsets, lengths, k, **kwargs)
    tensors = dict(offsets=offsets, lengths=lengths, row_starts=kwargs["row_starts"])
    tensors[name] = tensors[name].long()
    bad_kwargs = dict(kwargs, row_starts=tensors["row_starts"])
    bad = op(scores, tensors["offsets"], tensors["lengths"], k, **bad_kwargs)
    assert bad.shape == good.shape and not torch.equal(bad, good), (
        f"an int64 {name} stopped being misread"
    )
    with pytest.raises(AssertionError, match=f"{name} must be int32"):
        with validating(op):
            op(scores, tensors["offsets"], tensors["lengths"], k, **bad_kwargs)


@pytest.mark.parametrize("name", ["offsets", "lengths"])
def test_guard_refuses_a_short_index_tensor(name) -> None:
    """A one-element `offsets` or `lengths` meant to broadcast is accepted (the binding checks
    only that it is 1-D), and every row past the first reads past its end. Here the memory
    past it holds other values, which those rows take. Shown against the broadcast the
    caller meant, then refused."""
    (scores, offsets, lengths, k), kwargs = _ragged_call()
    rows = scores.shape[0]
    meant = dict(offsets=offsets, lengths=lengths)
    meant[name] = torch.full_like(meant[name], int(meant[name][1]))
    good = op(scores, meant["offsets"], meant["lengths"], k, **kwargs)
    # A one-element view whose storage continues with values other than its own.
    backing = torch.full((2 * rows,), 3, dtype=torch.int32, device=DEV)
    backing[0] = meant[name][0]
    short = dict(meant, **{name: backing[:1]})
    assert short[name].is_contiguous() and short[name].shape == (1,)
    bad = op(scores, short["offsets"], short["lengths"], k, **kwargs)
    assert bad.shape == good.shape and not torch.equal(bad, good), (
        f"a one-element {name} stopped being read past its end"
    )
    with pytest.raises(AssertionError, match=f"{name} needs one element per input row"):
        with validating(op):
            op(scores, short["offsets"], short["lengths"], k, **kwargs)


def test_negative_zero_ranks_below_positive_zero() -> None:
    """Outside the contract, pinned: the kernel orders scores by their bits, so -0.0 ranks
    below +0.0, where the canonical row holds them as one tie class. A tie class at the
    k-th value that mixes the two gets the kernel's order: the row is `reference`'s for the
    same scores with every -0.0 moved just below zero."""
    gen = torch.Generator(device=DEV).manual_seed(5)
    rows, cols, k = 4, 2080, 512
    scores = torch.zeros(rows, cols, device=DEV)
    scores[:, ::2] = -0.0
    positive = torch.rand(rows, cols, generator=gen, device=DEV) < 0.05
    scores = torch.where(positive, torch.rand(rows, cols, generator=gen, device=DEV) + 1.0, scores)
    starts = torch.zeros(rows, dtype=torch.int32, device=DEV)
    lengths = torch.full((rows,), cols, dtype=torch.int32, device=DEV)
    offsets = torch.zeros(rows, dtype=torch.int32, device=DEV)
    kwargs = dict(tie_break=TopKTieBreak.LARGE, row_starts=starts)
    out = op(scores, offsets, lengths, k, **kwargs)
    negative_zero = (scores == 0) & torch.signbit(scores)
    assert bool(negative_zero.any()) and bool(_straddles(scores, starts, lengths, k).all())
    below = torch.where(negative_zero, -torch.finfo(torch.float32).tiny * 2**-23, scores)
    _compare(out, op.reference(below, offsets, lengths, k, **kwargs))
    assert not torch.equal(out, op.reference(scores, offsets, lengths, k, **kwargs)), (
        "the kernel now folds -0.0 into +0.0: the note's precondition is gone"
    )
