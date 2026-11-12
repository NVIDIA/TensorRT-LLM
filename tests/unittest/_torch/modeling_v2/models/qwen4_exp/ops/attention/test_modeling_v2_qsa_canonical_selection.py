# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qsa_canonical_selection entry.

The rewrite fixes every slot of every row, so `_compare` is index equality against a
composite-key top-k, and against `_torch_rewrite`, the same rule as composed torch ops,
which also pins the rows outside the contract (NaN in a window, a window cut by the
table's width). The input rows come from an adversary that keeps a random part of the
tie class at the k-th value in a random order, or from the radix engine the target
calls. Beyond the cells: graph replay at 256 and 1024 rows, and the guard against a
one-element `row_starts`.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_canonical_selection import (  # noqa: E501
    _SPLIT_ROWS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_canonical_selection import (  # noqa: E501
    qsa_canonical_selection as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_topk_select import (  # noqa: E501
    qsa_topk_select,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "qsa_canonical_selection requires a CUDA device"

DEV = "cuda"
COMPRESS_RATIO = 4
BLOCK_TOPK = 512
COLS = 2080


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Index for index. The rule fixes every slot of every row, so nothing is tolerated: a
    selection does no arithmetic."""
    assert out.dtype == torch.int32, out.dtype
    assert out.shape == ref.shape, (out.shape, ref.shape)
    wrong = (out != ref).any(1)
    if bool(wrong.any()):
        row = int(wrong.nonzero()[0, 0])
        raise AssertionError(
            f"{int(wrong.sum())} of {out.shape[0]} rows differ from the canonical "
            f"selection; first is row {row}"
        )


def _torch_rewrite(logits, row_starts, row_ends, indices) -> torch.Tensor:
    """The rewrite as composed torch ops, in row tiles of 2^24 elements: a second oracle,
    and the one that fixes the rows outside the contract, where it and `reference` may
    disagree."""
    num_rows, num_cols = logits.shape
    index_topk = indices.shape[1]
    tile = max(1, (1 << 24) // max(num_cols, 1))
    for lo in range(0, num_rows, tile):
        hi = min(lo + tile, num_rows)
        scores, out = logits[lo:hi], indices[lo:hi]
        starts, ends = row_starts[lo:hi, None], row_ends[lo:hi, None]
        columns = torch.arange(num_cols, dtype=torch.int32, device=logits.device)[None, :]
        windowed = torch.where((columns >= starts) & (columns < ends), scores, float("nan"))
        picked = torch.gather(scores, 1, (out + starts).clamp_(0, num_cols - 1).long())
        kth = picked.masked_fill_(out < 0, float("inf")).amin(1, keepdim=True)
        at_least = windowed >= kth
        tied = windowed == kth
        surplus = at_least.sum(1, keepdim=True, dtype=torch.int32) - index_topk
        dropped = tied & (torch.cumsum(tied, 1, dtype=torch.int32) <= surplus)
        kept_seen = torch.cumsum(at_least ^ dropped, 1, dtype=torch.int32)
        slots = torch.arange(1, index_topk + 1, dtype=torch.int32, device=logits.device)
        slots = slots.repeat(hi - lo, 1)
        torch.searchsorted(kept_seen, slots, out_int32=True, out=out)
        out.sub_(starts)
        out.masked_fill_(slots > kept_seen[:, -1:], -1)
    return indices


def _logits(kind: str, rows: int, cols: int, gen: torch.Generator) -> torch.Tensor:
    if kind == "relu":
        return torch.relu(torch.randn(rows, cols, generator=gen, device=DEV)) * 3.0
    if kind == "relu_sparse":
        x = torch.relu(torch.randn(rows, cols, generator=gen, device=DEV)) * 3.0
        return x * (torch.rand(rows, cols, generator=gen, device=DEV) < 0.2)
    if kind == "relu_rare":
        # About one positive score in 750: a long window still has fewer than k, so
        # the k-th value is 0 and the tie class is most of the window, half of it -0.0.
        x = torch.relu(torch.randn(rows, cols, generator=gen, device=DEV) - 3.0)
        flip = torch.rand(rows, cols, generator=gen, device=DEV) < 0.5
        return torch.where(flip & (x == 0), -0.0, x)
    if kind == "quantized":
        # A handful of levels, so every k-th value is tied; rounding a small negative
        # score gives -0.0.
        return torch.round(torch.randn(rows, cols, generator=gen, device=DEV) * 2.0) / 2.0
    raise ValueError(kind)


def _windows(spec: dict, rows: int, cols: int, gen: torch.Generator):
    kind = spec["windows"][0]
    starts = torch.zeros(rows, dtype=torch.int32, device=DEV)
    if kind == "chunk":
        positions = spec["windows"][1] + torch.arange(rows, device=DEV)
    else:
        assert kind == "positions_random", kind
        lo, hi = spec["windows"][1:]
        positions = torch.randint(lo, hi + 1, (rows,), generator=gen, device=DEV)
    ends = ((positions + 1) // COMPRESS_RATIO).clamp_max(cols).to(torch.int32)
    return starts, ends


def _poison(logits, starts, ends, kind) -> None:
    columns = torch.arange(logits.shape[1], device=DEV)[None, :]
    outside = (columns < starts[:, None]) | (columns >= ends[:, None])
    if kind == "nan":
        logits.masked_fill_(outside, float("nan"))
        return
    # Values that would win any comparison they were allowed into.
    pattern = torch.tensor([float("nan"), float("inf"), 3e38, float("nan")], device=DEV)
    logits.copy_(torch.where(outside, pattern[columns % 4].expand_as(logits), logits))


def _order_key(logits: torch.Tensor) -> torch.Tensor:
    """int64 that orders as the fp32 values do, -0.0 folded into +0.0."""
    bits = (logits.float() + 0.0).view(torch.int32).long()
    return torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits)


def _adversary(logits, starts, ends, index_topk, gen) -> torch.Tensor:
    """An exact top-k of each window that keeps a random part of the tie class at the
    k-th value, in a random order, `-1` last."""
    rows, cols = logits.shape
    columns = torch.arange(cols, device=DEV)[None, :]
    in_window = (columns >= starts.long()[:, None]) & (columns < ends.long()[:, None])
    # A random 32-bit low word: equal scores order at random (a collision only repeats
    # an order), without sorting a `[rows, cols]` matrix.
    tiebreak = torch.randint(0, 1 << 32, (rows, cols), generator=gen, device=DEV)
    key = torch.where(
        in_window, (_order_key(logits) << 32) | tiebreak, torch.iinfo(torch.int64).min
    )
    width = min(index_topk, cols)
    top = torch.topk(key, width, dim=1).indices
    local = torch.where(torch.gather(in_window, 1, top), top - starts.long()[:, None], -1)
    shuffle = torch.where(local >= 0, torch.rand(local.shape, generator=gen, device=DEV), 2.0)
    local = torch.gather(local, 1, shuffle.argsort(1))
    out = torch.full((rows, index_topk), -1, dtype=torch.int32, device=DEV)
    out[:, :width] = local.to(torch.int32)
    return out


def _radix(logits, starts, ends) -> torch.Tensor:
    out = torch.empty(logits.shape[0], BLOCK_TOPK, dtype=torch.int32, device=DEV)
    return qsa_topk_select(logits, starts, ends, out, index_topk=BLOCK_TOPK)


def _build(spec: dict, rows: int, seed: int):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    logits = _logits(spec["logits"], rows, spec["cols"], gen)
    starts, ends = _windows(spec, rows, spec["cols"], gen)
    _poison(logits, starts, ends, spec["tail"])
    if spec["selection"] == "random":
        indices = _adversary(logits, starts, ends, spec["index_topk"], gen)
    else:
        indices = _radix(logits, starts, ends)
    return logits, starts, ends, indices


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    spec = cell.spec
    seed = sum(map(ord, cell.why))
    row_counts = spec["rows"] if isinstance(spec["rows"], tuple) else (spec["rows"],)
    for rows in row_counts:
        args = _build(spec, rows, seed + rows)
        expected = op.reference(*args)
        composed = _torch_rewrite(args[0], args[1], args[2], args[3].clone())
        out = op(*args)
        assert out is args[3], "the rewrite writes `indices` and returns it"
        _compare(out, expected)
        _compare(out, composed)


def _random_windows(rows: int, cols: int, gen: torch.Generator):
    """Windows anywhere in the table: a tenth empty, a fifth at most k wide, the rest of
    any width the table leaves room for."""
    starts = torch.randint(0, cols + 1, (rows,), generator=gen, device=DEV)
    u = torch.rand(rows, generator=gen, device=DEV)
    short = torch.randint(1, BLOCK_TOPK + 1, (rows,), generator=gen, device=DEV)
    wide = torch.randint(0, cols + 1, (rows,), generator=gen, device=DEV)
    length = torch.where(u < 0.1, 0, torch.where(u < 0.3, short, wide))
    ends = (starts + length).clamp_max(cols)
    return starts.to(torch.int32), ends.to(torch.int32)


def _chunk_windows(rows: int, cols: int, last_position: int):
    """A prefill chunk's causal windows, the last row at `last_position`."""
    positions = last_position - rows + 1 + torch.arange(rows, device=DEV)
    ends = ((positions + 1) // COMPRESS_RATIO).clamp(0, cols).to(torch.int32)
    return torch.zeros(rows, dtype=torch.int32, device=DEV), ends


def _long_windows(kind: str, rows: int, cols: int, gen: torch.Generator):
    if kind == "random":
        return _random_windows(rows, cols, gen)
    if kind == "chunk_start":
        return _chunk_windows(rows, cols, rows - 1)
    assert kind == "chunk_end", kind
    return _chunk_windows(rows, cols, cols * COMPRESS_RATIO - 1)


#: (rows, cols, windows, logits) at long context: decode-sized to 16384-row prefill
#: chunks, 2080- to 65536-wide tables (65536 is the 256k context), on both sides of the
#: rows at which the op stops splitting a row into segments.
_LONG_CASES = (
    (1, 65536, "chunk_end", "relu_rare"),
    (16, 32768, "random", "quantized"),
    (256, 65536, "random", "relu_rare"),
    (_SPLIT_ROWS - 1, 25000, "random", "quantized"),
    (_SPLIT_ROWS, 25000, "random", "relu_rare"),
    (4096, 8192, "random", "quantized"),
    (4096, 32768, "chunk_end", "relu_sparse"),
    (16384, 2080, "chunk_start", "relu_rare"),
    (16384, 65536, "chunk_end", "quantized"),
)


@pytest.mark.parametrize(
    "rows,cols,windows,kind", _LONG_CASES, ids=["-".join(map(str, c)) for c in _LONG_CASES]
)
def test_long_context_rows_match_reference_and_torch_rewrite(rows, cols, windows, kind) -> None:
    """Tie classes of most of a window (relu zeros, half of them -0.0) or of a quantized
    level, empty windows and windows of at most k, garbage past the windows."""
    gen = torch.Generator(device=DEV).manual_seed(rows * 131 + cols)
    logits = _logits(kind, rows, cols, gen)
    starts, ends = _long_windows(windows, rows, cols, gen)
    _poison(logits, starts, ends, "poison")
    raw = _adversary(logits, starts, ends, BLOCK_TOPK, gen)
    expected = op.reference(logits, starts, ends, raw)
    _compare(_torch_rewrite(logits, starts, ends, raw.clone()), expected)
    _compare(op(logits, starts, ends, raw), expected)


@pytest.mark.parametrize("rows", (_SPLIT_ROWS - 1, 2048))
def test_rows_outside_the_contract_match_the_torch_rewrite(rows) -> None:
    """NaN and +-inf inside the windows, NaN picks, rows with no pick, picks past the
    window, and windows cut short by a narrowed table: no canonical row to compare
    against, so the rows are the torch rewrite's, bit for bit."""
    gen = torch.Generator(device=DEV).manual_seed(5)
    cols, live = 9000, 6000
    clean = _logits("quantized", rows, cols, gen)
    starts, ends = _random_windows(rows, cols, gen)
    _poison(clean, starts, ends, "poison")
    specials = torch.tensor([float("nan"), float("inf"), float("-inf")], device=DEV)
    which = torch.randint(0, 3, (rows, cols), generator=gen, device=DEV)
    sprinkle = torch.rand(rows, cols, generator=gen, device=DEV) < 0.002
    dirty = torch.where(sprinkle, specials[which], clean)
    # Even rows were selected over the dirty scores (NaN sorts first: NaN picks), odd
    # rows over the clean ones.
    seen = torch.where((torch.arange(rows, device=DEV) % 2 == 0)[:, None], dirty, clean)
    raw = _adversary(seen, starts, ends, BLOCK_TOPK, gen)
    raw[::7] = -1
    raw[3::11, 0] = cols + 100
    raw[5::13, 1] = -5
    for scores in (dirty, dirty[:, :live]):
        expected = _torch_rewrite(scores, starts, ends, raw.clone())
        _compare(op(scores, starts, ends, raw.clone()), expected)


@pytest.mark.parametrize("rows", (256, _SPLIT_ROWS))
def test_replays_in_a_cuda_graph(rows) -> None:
    """Captured once, replayed over new scores and new picks in the same buffers."""
    gen = torch.Generator(device=DEV).manual_seed(77)
    cols = 32768
    logits = torch.empty(rows, cols, device=DEV)
    starts = torch.empty(rows, dtype=torch.int32, device=DEV)
    ends = torch.empty(rows, dtype=torch.int32, device=DEV)
    indices = torch.empty(rows, BLOCK_TOPK, dtype=torch.int32, device=DEV)

    def stage(kind):
        scores = _logits(kind, rows, cols, gen)
        lo, hi = _random_windows(rows, cols, gen)
        _poison(scores, lo, hi, "poison")
        logits.copy_(scores)
        starts.copy_(lo)
        ends.copy_(hi)
        indices.copy_(_adversary(scores, lo, hi, BLOCK_TOPK, gen))
        return op.reference(logits, starts, ends, indices)

    stage("relu")
    op(logits, starts, ends, indices)  # compile outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(logits, starts, ends, indices)
    for kind in ("relu_rare", "quantized"):
        expected = stage(kind)
        graph.replay()
        _compare(indices, expected)


def test_narrowing_the_table_to_a_bound_on_the_windows_changes_nothing() -> None:
    """A caller that knows every window ends by column 1500 passes `logits[:, :1500]` of a
    wide table: the rows are the full table's."""
    gen = torch.Generator(device=DEV).manual_seed(21)
    rows, cols, bound = 4096, 8192, 1500
    logits = _logits("relu_sparse", rows, cols, gen)
    starts = torch.zeros(rows, dtype=torch.int32, device=DEV)
    ends = torch.randint(0, bound + 1, (rows,), generator=gen, device=DEV).to(torch.int32)
    _poison(logits, starts, ends, "poison")
    raw = _radix(logits, starts, ends)
    wide = op(logits, starts, ends, raw.clone())
    narrow = op(logits[:, :bound], starts, ends, raw.clone())
    _compare(wide, op.reference(logits, starts, ends, raw))
    assert torch.equal(wide, narrow)


def test_guard_refuses_a_broadcast_row_start() -> None:
    """A one-element `row_starts` broadcasts over every row without an error and frames
    every row by row 0's start. Shown unguarded first, then refused."""
    gen = torch.Generator(device=DEV).manual_seed(9)
    rows = 4
    logits = torch.relu(torch.randn(rows, COLS, generator=gen, device=DEV))
    starts = torch.tensor([0, 10, 20, 30], dtype=torch.int32, device=DEV)
    ends = torch.full((rows,), 2000, dtype=torch.int32, device=DEV)
    raw = _adversary(logits, starts, ends, BLOCK_TOPK, gen)
    expected = op.reference(logits, starts, ends, raw)
    out = op(logits, starts[:1], ends, raw.clone())
    assert out.shape == expected.shape and not torch.equal(out, expected), (
        "the broadcast hazard stopped reproducing"
    )
    with pytest.raises(AssertionError, match="one entry per logits row"):
        with validating(op):
            op(logits, starts[:1], ends, raw.clone())
