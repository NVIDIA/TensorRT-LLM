# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the indexer_topk_decode catalog entry.

The `_ML4_1M_DEP4` and `_ML4_1M_DSPARK` cells use scorer widths 1048576 and
1048584, respectively, and run through the scratch-footprint drivers
(`_scratch_dep4`, `_drive_dep4`, `_drive_graph_dep4`). There
the radix scratch is the size each cell names, the leading view of a buffer
with a canary tail. At every call such a cell gates (eager, a graph's warm-up
and each replay) the test checks that the op's first launch wrote a leading
prefix of rows x blocks x index_topk elements and nothing past it, and that the
split is 10 blocks per row at every row count, the whole 16-request arena at
16 single-token or 64 verification rows. Every other cell uses `_scratch`,
`_drive` and `_drive_graph`, whose arena has a guard slot that no call may write.
"""

import math
import re
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.indexer_topk_decode import (  # noqa: E501
    MAX_INDEX_TOPK_SINGLE_BLOCK,
    MAX_INDEX_TOPK_SPLIT_WORK,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.indexer_topk_decode import (  # noqa: E501
    indexer_topk_decode as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — needs the path declared above

assert torch.cuda.is_available(), "indexer_topk_decode requires a CUDA device"

#: Written past each row's causal end. A kernel that read past the end would
#: select these before any real score.
POISON = 1.0e9
#: What `indices` holds before a call; a row the op should not write keeps it.
UNWRITTEN = -7
#: Blocks per row the DSA metadata's radix scratch reserves for each sequence
#: slot (`_MAX_RADIX_BLOCKS_PER_ROW`), the most the op ever splits a row into.
AUX_BLOCKS = 10
#: The CUDA-graph padding dummy's KV length (one token, no speculation).
PAD_SEQ_LEN = 1
#: What `_scratch`'s radix arena holds before a call, and the slot past it keeps.
AUX_SENTINEL = -24301
#: Columns the `edges` scores plant at each end of every split block: one
#: 16-byte vector of fp32, so a block that starts off that grid hands over its
#: unaligned head and tail columns among them.
EDGE_COLUMNS = 4
#: Elements past the end of each `_scratch_dep4` radix scratch buffer, which
#: the op is never given and must never write.
CANARY = 4096
#: What the dep4 drivers' radix index scratch holds before a call (`_arm`): no
#: column, and not the -1 the first launch writes for a block's padding. The
#: logits scratch holds NaN.
AUX_UNWRITTEN = -(2**31)


def _ends(seq_lens, next_n: int, rows: int) -> list[int]:
    return [max(0, seq_lens[r // next_n] - next_n + r % next_n + 1) for r in range(rows)]


def _edge_plan(end: int, k: int) -> tuple[list[int], list[int], list[int]]:
    """The `edges` scores' planted columns of a row ending at `end`, as `(top, inside, outside)`.

    The row is cut as the op cuts it at `AUX_BLOCKS` blocks: floor(end / blocks)
    columns each, the last block the rest. Each block's first and last
    `EDGE_COLUMNS` columns are planted, alternately at the top of the row (`top`)
    and around its k-th score: `inside` just above it, so among the winners, and
    `outside` just below it, so not.
    """
    span = end // AUX_BLOCKS
    edges = set()
    for b in range(AUX_BLOCKS):
        lo = b * span
        hi = end if b == AUX_BLOCKS - 1 else lo + span
        edges.update(range(lo, min(lo + EDGE_COLUMNS, hi)))
        edges.update(range(max(hi - EDGE_COLUMNS, lo), hi))
    edges = sorted(edges)
    top, around = edges[0::2], edges[1::2]
    inside, outside = around[0::2], around[1::2]
    assert end > k + len(outside), (end, k)
    return top, inside, outside


def _compare(out: torch.Tensor, ref: torch.Tensor, logits, seq_lens, next_n, by_score) -> None:
    """Exact per row: the winners first, then -1 padding; the op does not order its winners.

    A top-k picks the same keys or it does not; there is no rounding to absorb.
    With distinct scores the winning set is unique, so each row is sorted and
    compared to the reference's (`by_score=False`). With ties at the k-th
    score the set is not unique: then the selected columns must be distinct
    and causal, and the multiset of their fp32 scores must be the reference's
    exactly, which every correct choice among the tied columns satisfies and
    a wrong winner does not.
    """
    assert out.dtype == ref.dtype == torch.int32, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    k = out.shape[1]
    ends = _ends(seq_lens, next_n, out.shape[0])
    for r, end in enumerate(ends):
        n = min(k, end)
        row = out[r]
        assert bool((row[n:] == -1).all()) and bool((row[:n] >= 0).all()), (
            f"row {r} (end {end}): the {n} winners must come first and -1 after; "
            f"got {int((row >= 0).sum())} columns and {int((row == -1).sum())} -1s"
        )
        if not by_score:
            continue
        won = row[:n].long()
        assert won.unique().numel() == n and bool((won < end).all()), (
            f"row {r}: selected columns repeat or pass the causal end {end}"
        )
        got = logits[r, won].sort().values
        want = logits[r, ref[r, :n].long()].sort().values
        assert torch.equal(got, want), f"row {r}: the selected scores are not the top {n}"
    if not by_score:
        got, want = out.sort(dim=1).values, ref.sort(dim=1).values
        bad = (got != want).any(dim=1)
        assert not bool(bad.any()), (
            f"{int(bad.sum())} of {out.shape[0]} rows select a different key set; "
            f"first at row {int(bad.nonzero()[0])}"
        )


def _scores(kind: str, row: int, end: int, g: torch.Generator, k: int = 2048) -> torch.Tensor:
    if kind == "edges":
        # A random permutation of distinct integers with every split block's
        # first and last columns moved to the ranks `_edge_plan` gives them:
        # the top ranks, and the ranks either side of the k-th score.
        scores = torch.randperm(end, generator=g)
        where = torch.empty(end, dtype=torch.int64)
        where[scores] = torch.arange(end)
        top, inside, outside = _edge_plan(end, k)
        ranks = list(range(len(top)))
        ranks += [k - 1 - j for j in range(len(inside))] + [k + j for j in range(len(outside))]
        for col, rank in zip(top + inside + outside, ranks):
            value, held = end - 1 - rank, int(scores[col])
            other = int(where[value])
            scores[col], scores[other] = value, held
            where[value], where[held] = col, other
        return scores.float()
    if kind == "perm":
        # A random permutation of distinct integers: the top-k set is unique.
        return torch.randperm(end, generator=g).float()
    if kind == "ties":
        # Ties at the k-th value: 16 levels, all +0.0 (every head's relu at its
        # floor), and +0.0 mixed with -0.0, which compare equal.
        if row % 3 == 0:
            return torch.randint(0, 16, (end,), generator=g).float()
        zeros = torch.zeros(end)
        if row % 3 == 2:
            zeros[torch.rand(end, generator=g) < 0.5] = -0.0
        return zeros
    if kind == "wide":
        # Distinct magnitudes from 2^-100 to 2^100, both signs: most are outside
        # fp16's range or below its smallest subnormal.
        exponent = torch.randperm(end, generator=g).double() / end * 200.0 - 100.0
        sign = torch.where(torch.rand(end, generator=g) < 0.5, -1.0, 1.0).double()
        return (sign * torch.exp2(exponent)).float()
    raise ValueError(kind)


def _logits(spec, seq_lens, next_n: int, g: torch.Generator) -> torch.Tensor:
    width, pad = spec["width"], spec.get("row_pad", 0)
    rows = len(seq_lens) * next_n
    parent = torch.empty(rows, width + pad, dtype=torch.float32)
    for r, end in enumerate(_ends(seq_lens, next_n, rows)):
        parent[r] = POISON + torch.arange(width + pad, dtype=torch.float32)
        parent[r, :end] = _scores(spec.get("scores", "perm"), r, end, g, spec["index_topk"])
    logits = parent.cuda()[:, :width]
    assert pad == 0 or not logits.is_contiguous()
    return logits


def _scratch(spec, rows: int):
    """The radix scratch, as `((indices, logits), (guard_indices, guard_logits))`.

    The engine's per-sequence arena when the cell names its size (`aux_rows`),
    else sized for the batch. The arena is the leading `slots` of a buffer one
    sequence slot longer, all of it `AUX_SENTINEL`; the extra slot is the
    guard, which no call may write.
    """
    slots = spec.get("aux_rows", rows)
    assert slots >= rows
    shape = (slots + 1, AUX_BLOCKS, spec["index_topk"])
    idx = torch.full(shape, AUX_SENTINEL, dtype=torch.int32, device="cuda")
    val = torch.full(shape, AUX_SENTINEL, dtype=torch.float32, device="cuda")
    return (idx[:slots], val[:slots]), (idx[slots:], val[slots:])


def _assert_guard_untouched(guard) -> None:
    assert bool((guard[0] == AUX_SENTINEL).all()) and bool((guard[1] == AUX_SENTINEL).all()), (
        "the call wrote past the radix arena"
    )


def _scratch_dep4(spec, rows: int):
    """The dep4 target's radix scratch (its file's `_scratch`), for its drivers
    and its own tests: the engine's per-sequence arena when the cell names its
    size (`aux_rows`), else sized for the batch.

    Each buffer is exactly that size, `[slots, 10, index_topk]` as the DSA
    metadata allocates it, and the leading view of an allocation whose
    CANARY-element tail the op is never given, so a write past the arena shows.
    """
    slots = spec.get("aux_rows", rows)
    assert slots >= rows
    shape = (slots, AUX_BLOCKS, spec["index_topk"])
    n = math.prod(shape)
    return tuple(
        torch.empty(n + CANARY, dtype=dtype, device="cuda")[:n].view(shape)
        for dtype in (torch.int32, torch.float32)
    )


def _arm(aux) -> None:
    """Fill both scratch buffers and their canaries before a call."""
    aux[0]._base.fill_(AUX_UNWRITTEN)
    aux[1]._base.fill_(float("nan"))


def _check_scratch(aux, rows: int, k: int, blocks=None) -> int:
    """What the call's first launch wrote: rows x blocks x index_topk elements of
    each buffer, a leading prefix of the arena, and nothing after it or in the
    canary. Returns the blocks per row that footprint implies (0: one block a
    row, no scratch)."""
    idx, val = aux[0]._base, aux[1]._base
    w_i, w_l = idx != AUX_UNWRITTEN, ~torch.isnan(val)
    gaps = [(w[1:] & ~w[:-1]).sum() for w in (w_i, w_l)]
    n_i, n_l, gap_i, gap_l = torch.stack([w_i.sum(), w_l.sum(), *gaps]).tolist()
    assert not gap_i and not gap_l, "the scratch written is not a leading prefix"
    assert n_i == n_l and n_i <= aux[0].numel(), (
        f"the scratch footprint is {n_i} / {n_l} elements against an arena of {aux[0].numel()}"
    )
    assert n_i % (rows * k) == 0, (n_i, rows, k)
    got = n_i // (rows * k)
    assert blocks is None or got == blocks, f"{rows} rows split {got} blocks per row, not {blocks}"
    return got


def _call(logits, seq_lens, indices, next_n, k, aux_idx, aux_val):
    # As the targets call it: keyword scratch, compress_ratio left at 1.
    return op(
        logits,
        seq_lens,
        indices,
        next_n,
        k,
        radix_aux_indices=aux_idx,
        radix_aux_logits=aux_val,
    )


def _drive(spec, seq_lens, logits, aux, guard, by_score):
    """One eager call over `seq_lens`, gated against the entry's reference."""
    next_n, k = spec["next_n"], spec["index_topk"]
    rows = logits.shape[0]
    lens = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    tail_of = spec.get("tail_of")
    if tail_of is None:
        buf = torch.full((rows, k), UNWRITTEN, dtype=torch.int32, device="cuda")
        indices = buf
    else:
        # A mixed step: the context rows' selection sits ahead of these rows in
        # one buffer, and the call writes only its tail.
        buf = torch.full((tail_of, k), UNWRITTEN, dtype=torch.int32, device="cuda")
        indices = buf[tail_of - rows :]
    out = _call(logits, lens, indices, next_n, k, *aux)
    assert out is indices, "the op writes the caller's buffer and returns it"
    _assert_guard_untouched(guard)
    _compare(
        out, op.reference(logits, lens, indices, next_n, k), logits, seq_lens, next_n, by_score
    )
    if tail_of is not None:
        assert bool((buf[: tail_of - rows] == UNWRITTEN).all()), (
            "a row ahead of the slice was written"
        )
    if spec.get("scores") == "edges":
        # The planted block-edge columns are where the gate looks: the winners
        # hold every one scored above the k-th score and none below it.
        for r, end in enumerate(_ends(seq_lens, next_n, rows)):
            top, inside, outside = _edge_plan(end, k)
            won = set(out[r].tolist())
            assert set(top + inside) <= won and not set(outside) & won, (
                f"row {r} (end {end}): a block-edge column is missing or wrongly selected"
            )


def _drive_dep4(spec, seq_lens, logits, aux, by_score):
    """One eager call over `seq_lens`, gated against the entry's reference.

    The dep4 target's `_drive`: its scratch armed before the call and its
    footprint checked after it, in place of `_drive`'s guard slot.
    """
    next_n, k = spec["next_n"], spec["index_topk"]
    rows = logits.shape[0]
    lens = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    tail_of = spec.get("tail_of")
    if tail_of is None:
        buf = torch.full((rows, k), UNWRITTEN, dtype=torch.int32, device="cuda")
        indices = buf
    else:
        # A mixed step: the context rows' selection sits ahead of these rows in
        # one buffer, and the call writes only its tail.
        buf = torch.full((tail_of, k), UNWRITTEN, dtype=torch.int32, device="cuda")
        indices = buf[tail_of - rows :]
    _arm(aux)
    out = _call(logits, lens, indices, next_n, k, *aux)
    assert out is indices, "the op writes the caller's buffer and returns it"
    _check_scratch(aux, rows, k, spec.get("blocks_per_row"))
    _compare(
        out, op.reference(logits, lens, indices, next_n, k), logits, seq_lens, next_n, by_score
    )
    if tail_of is not None:
        assert bool((buf[: tail_of - rows] == UNWRITTEN).all()), (
            "a row ahead of the slice was written"
        )


def _drive_graph(spec, g, aux, guard, by_score):
    """Capture once per batch, as the engine does, and replay on new logits and
    seq_lens. The op orders its winners differently on every call, so a replay
    is gated as a set, against an eager call on the same data and against the
    entry's reference."""
    next_n, k, width = spec["next_n"], spec["index_topk"], spec["width"]
    for batch in spec["graph_batches"]:
        capture_lens = spec["seq_lens"][:batch]
        logits = _logits(spec, capture_lens, next_n, g)
        lens = torch.tensor(capture_lens, dtype=torch.int32, device="cuda")
        indices = torch.full((batch * next_n, k), UNWRITTEN, dtype=torch.int32, device="cuda")
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            _call(logits, lens, indices, next_n, k, *aux)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _call(logits, lens, indices, next_n, k, *aux)
        for i in range(spec["replays"]):
            if "replay_lens" in spec:
                new_lens = list(spec["replay_lens"][i][:batch])
            else:
                new_lens = torch.randint(next_n, width + 1, (batch,), generator=g).tolist()
            if batch > 2:
                new_lens[-1] = PAD_SEQ_LEN + next_n - 1
            new = _logits(spec, new_lens, next_n, g)
            logits.copy_(new)
            lens.copy_(torch.tensor(new_lens, dtype=torch.int32))
            indices.fill_(UNWRITTEN)
            graph.replay()
            torch.cuda.synchronize()
            _assert_guard_untouched(guard)
            eager = _call(
                new,
                torch.tensor(new_lens, dtype=torch.int32, device="cuda"),
                torch.full_like(indices, UNWRITTEN),
                next_n,
                k,
                *aux,
            )
            _compare(indices, eager, new, new_lens, next_n, by_score)
            ref = op.reference(new, lens, indices, next_n, k)
            _compare(indices, ref, new, new_lens, next_n, by_score)


def _drive_graph_dep4(spec, g, aux, by_score):
    """Capture once per batch, as the engine does, and replay on new logits and
    seq_lens. The op orders its winners differently on every call, so a replay
    is gated as a set, against an eager call on the same data and against the
    entry's reference.

    The dep4 target's `_drive_graph`: its scratch armed before the warm-up and
    each replay and its footprint checked after them, in place of
    `_drive_graph`'s guard slot. The engine's warm-up call ahead of each capture runs eagerly on the
    capture's own rows (at boot, the long-family dummy) and is gated too. With
    `live`, each replay is attention DP's padded batch: `live` rows of this
    rank's own, the first at the full width, one less, then a drawn length,
    and the rest of the largest rank's batch padding dummies at `next_n` keys.
    """
    next_n, k, width = spec["next_n"], spec["index_topk"], spec["width"]
    blocks = spec.get("blocks_per_row")
    for batch in spec["graph_batches"]:
        capture_lens = spec["seq_lens"][:batch]
        logits = _logits(spec, capture_lens, next_n, g)
        lens = torch.tensor(capture_lens, dtype=torch.int32, device="cuda")
        indices = torch.full((batch * next_n, k), UNWRITTEN, dtype=torch.int32, device="cuda")
        _arm(aux)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            _call(logits, lens, indices, next_n, k, *aux)
        torch.cuda.current_stream().wait_stream(side)
        _check_scratch(aux, batch * next_n, k, blocks)
        warm = op.reference(logits, lens, indices, next_n, k)
        _compare(indices, warm, logits, capture_lens, next_n, by_score)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _call(logits, lens, indices, next_n, k, *aux)
        for replay in range(spec["replays"]):
            if "replay_lens" in spec:
                new_lens = list(spec["replay_lens"][replay][:batch])
            else:
                new_lens = torch.randint(next_n, width + 1, (batch,), generator=g).tolist()
            if "live" in spec:
                live = min(spec["live"], batch)
                if replay < 2:
                    new_lens[0] = width - replay
                new_lens[live:] = [PAD_SEQ_LEN + next_n - 1] * (batch - live)
            elif batch > 2:
                new_lens[-1] = PAD_SEQ_LEN + next_n - 1
            new = _logits(spec, new_lens, next_n, g)
            logits.copy_(new)
            lens.copy_(torch.tensor(new_lens, dtype=torch.int32))
            indices.fill_(UNWRITTEN)
            _arm(aux)
            graph.replay()
            torch.cuda.synchronize()
            _check_scratch(aux, batch * next_n, k, blocks)
            eager = _call(
                new,
                torch.tensor(new_lens, dtype=torch.int32, device="cuda"),
                torch.full_like(indices, UNWRITTEN),
                next_n,
                k,
                *aux,
            )
            _compare(indices, eager, new, new_lens, next_n, by_score)
            ref = op.reference(new, lens, indices, next_n, k)
            _compare(indices, ref, new, new_lens, next_n, by_score)


def _is_dep4(spec) -> bool:
    """Cells with an explicit split count use the scratch-footprint drivers."""
    return "blocks_per_row" in spec


def _cell_id(cell) -> str:
    """`why[:44]`, as each target's file named its cells. Five of the dep4
    target's whys share their first 44 characters with the tp4 target's 1M
    cells, so the dep4 target's cells carry a ` (dep4)` suffix."""
    return cell.why[:44] + (" (dep4)" if _is_dep4(cell.spec) and cell.spec["next_n"] == 1 else "")


@pytest.mark.parametrize("cell", op.CELLS, ids=[_cell_id(c) for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    spec = cell.spec
    g = torch.Generator().manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    by_score = spec.get("scores") == "ties"
    rows = len(spec["seq_lens"]) * spec["next_n"]
    # The dep4 target's cells take its scratch and drivers (armed scratch,
    # footprint and split checked at every call); every other cell the guard
    # slot's.
    dep4 = _is_dep4(spec)
    aux, guard = (_scratch_dep4(spec, rows), None) if dep4 else _scratch(spec, rows)
    if "graph_batches" in spec:
        if dep4:
            _drive_graph_dep4(spec, g, aux, by_score)
        else:
            _drive_graph(spec, g, aux, guard, by_score)
        return
    logits = _logits(spec, spec["seq_lens"], spec["next_n"], g)
    for batch in spec.get("batches", (len(spec["seq_lens"]),)):
        # The leading `batch` sequences: rows are independent, so a smaller
        # batch is the same operands' leading rows.
        lens = spec["seq_lens"][:batch]
        if dep4:
            _drive_dep4(spec, lens, logits[: batch * spec["next_n"]], aux, by_score)
        else:
            _drive(spec, lens, logits[: batch * spec["next_n"]], aux, guard, by_score)


@pytest.mark.parametrize("next_n,width,row_pad", [(1, 1048576, 0), (4, 1048584, 248)])
def test_sw1m_sixteen_requests_need_the_whole_arena(next_n, width, row_pad) -> None:
    """Every row at 1M splits 10 ways: one missing scratch element must fail
    before any write, both for single-token decode and four-token verification."""
    k, rows = 2048, 16 * next_n
    spec = dict(next_n=next_n, width=width, row_pad=row_pad, index_topk=k, aux_rows=rows)
    lens = [width] * 16
    logits = _logits(spec, lens, next_n, torch.Generator().manual_seed(14))
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    indices = torch.full((rows, k), UNWRITTEN, dtype=torch.int32, device="cuda")
    aux = _scratch_dep4(spec, rows)
    short = tuple(t.view(-1)[:-1] for t in aux)
    _arm(aux)
    need = rows * AUX_BLOCKS * k
    with pytest.raises(RuntimeError, match=rf"got {need - 1} / {need - 1}, need {need}\)"):
        _call(logits, seq_lens, indices, next_n, k, *short)
    assert bool((indices == UNWRITTEN).all())
    assert _check_scratch(aux, rows, k) == 0
    _call(logits, seq_lens, indices, next_n, k, *aux)
    assert _check_scratch(aux, rows, k) == 10
    _compare(
        indices, op.reference(logits, seq_lens, indices, next_n, k), logits, lens, next_n, False
    )


def test_short_rows_list_their_columns_in_order() -> None:
    """On the single-block path, a row no wider than top-k is the identity, in order, then -1."""
    spec = dict(next_n=2, width=64, index_topk=16)
    g = torch.Generator().manual_seed(11)
    logits = _logits(spec, [5, 40], 2, g)
    indices = torch.full((4, 16), UNWRITTEN, dtype=torch.int32, device="cuda")
    out = _call(
        logits, torch.tensor([5, 40], dtype=torch.int32, device="cuda"), indices, 2, 16, None, None
    )
    assert out[0].tolist() == [0, 1, 2, 3] + [-1] * 12
    assert out[1].tolist() == [0, 1, 2, 3, 4] + [-1] * 11


def _small():
    spec = dict(next_n=1, width=128, index_topk=16)
    logits = _logits(spec, [100, 80], 1, torch.Generator().manual_seed(7))
    lens = torch.tensor([100, 80], dtype=torch.int32, device="cuda")
    return logits, lens


def test_guard_refuses_a_wider_indices_buffer() -> None:
    """The kernel writes at a row stride of index_topk, not of indices.size(1)."""
    logits, lens = _small()
    ref = op.reference(logits, lens, None, 1, 16)
    wide = torch.full((2, 32), UNWRITTEN, dtype=torch.int32, device="cuda")
    _call(logits, lens, wide, 1, 16, None, None)
    # Unguarded: row 1's winners land in row 0's second half, row 1 is never written.
    assert set(wide[0, 16:].tolist()) == set(ref[1].tolist())
    assert bool((wide[1] == UNWRITTEN).all())
    with pytest.raises(AssertionError, match="exactly 16 columns wide"):
        with validating(op):
            _call(logits, lens, wide, 1, 16, None, None)


def test_the_op_refuses_int64_seq_lens_itself() -> None:
    """Loud, so `is_valid` does not restate it."""
    logits, lens = _small()
    indices = torch.full((2, 16), UNWRITTEN, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="expected scalar type Int but found Long"):
        _call(logits, lens.long(), indices, 1, 16, None, None)
    assert bool((indices == UNWRITTEN).all())


#: The Mistral Large 4 targets' scorer widths: the FP8 target's 9216, the NVFP4
#: target's 32768 and its 1M envelope's 1048576, the dep4 target's at the
#: checkpoint's own context (10 blocks per row at the two rows the guards below
#: pass, at both NVFP4 widths).
ML4_WIDTHS = [9216, 32768, 1048576]


@pytest.mark.parametrize("width", [32776, 1048584])
@pytest.mark.parametrize("seq_len", [1, 2])
def test_guard_refuses_a_negative_uniform_causal_end(seq_len, width) -> None:
    spec = dict(next_n=4, width=width, row_pad=248, index_topk=2048, aux_rows=4)
    logits = _logits(spec, [4], 4, torch.Generator().manual_seed(9))
    lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    indices = torch.full((4, 2048), UNWRITTEN, dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="negative causal end"):
        with validating(op):
            _call(logits, lens, indices, 4, 2048, *_scratch(spec, 4))
    assert bool((indices == UNWRITTEN).all())


@pytest.mark.parametrize("width", ML4_WIDTHS)
def test_guard_refuses_a_causal_end_past_the_row(width) -> None:
    """At Mistral Large 4's widths: an end past the row reads the next row's columns."""
    spec = dict(next_n=1, width=width, index_topk=2048, aux_rows=16)
    logits = _logits(spec, [width, width], 1, torch.Generator().manual_seed(9))
    logits[1, :64] = 5.0e8 + torch.arange(64, device="cuda", dtype=torch.float32)
    long_lens = torch.tensor([width + 64, width], dtype=torch.int32, device="cuda")
    indices = torch.full((2, 2048), UNWRITTEN, dtype=torch.int32, device="cuda")
    aux, _ = _scratch(spec, 2)
    _call(logits, long_lens, indices, 1, 2048, *aux)
    # Unguarded: row 0 selects all 64 planted columns of row 1, as indices past its end.
    assert int((indices[0] >= width).sum()) == 64
    with pytest.raises(AssertionError, match=f"exceeds the {width}-column logits"):
        with validating(op):
            _call(logits, long_lens, indices, 1, 2048, *aux)


@pytest.mark.parametrize("width", ML4_WIDTHS)
def test_guard_refuses_a_column_strided_view(width) -> None:
    """At Mistral Large 4's widths the split path selects a wrong set, past the row's end."""
    spec = dict(next_n=1, width=width, index_topk=2048, aux_rows=16)
    lens = [width, 5000]
    logits = _logits(spec, lens, 1, torch.Generator().manual_seed(8))
    holder = torch.full((width, 2), -5.0e8, dtype=torch.float32, device="cuda")
    strided = holder.t()
    strided.copy_(logits)
    assert strided.stride() == (1, 2)
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    indices = torch.full((2, 2048), UNWRITTEN, dtype=torch.int32, device="cuda")
    aux, _ = _scratch(spec, 2)
    _call(strided, seq_lens, indices, 1, 2048, *aux)
    ref = op.reference(logits, seq_lens, indices, 1, 2048)
    # Unguarded: both rows select other columns than the reference, row 0 past its end.
    for r in range(2):
        assert set(indices[r].tolist()) != set(ref[r].tolist())
    assert int((indices[0] >= width).sum()) > 0
    with pytest.raises(AssertionError, match="unit-stride"):
        with validating(op):
            _call(strided, seq_lens, indices, 1, 2048, *aux)


@pytest.mark.parametrize(
    "width,k",
    [(128, MAX_INDEX_TOPK_SINGLE_BLOCK + 1)]
    + [(w, MAX_INDEX_TOPK_SPLIT_WORK + 1) for w in ML4_WIDTHS],
)
def test_guard_refuses_top_k_past_the_launch_ceiling(width, k) -> None:
    spec = dict(next_n=1, width=width, index_topk=k)
    lens = [min(width, 100)] if width < 4096 else [width]
    logits = _logits(spec, lens, 1, torch.Generator().manual_seed(12))
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    indices = torch.full((1, k), UNWRITTEN, dtype=torch.int32, device="cuda")
    aux, _ = _scratch(spec, 1)
    # Unguarded: the call returns quietly, the launch never ran, and its error
    # surfaces at the next kernel launch, here a torch one of its own.
    _call(logits, seq_lens, indices, 1, k, *aux)
    with pytest.raises(RuntimeError, match="invalid argument"):
        torch.ones(1, device="cuda").add_(1)
    assert bool((indices == UNWRITTEN).all())
    # One past the ceiling is refused; the ceiling itself launches and answers.
    with pytest.raises(AssertionError, match="index_topk must be in"):
        with validating(op):
            _call(logits, seq_lens, indices, 1, k, *aux)
    at = indices[:, : k - 1].contiguous()
    with validating(op):
        _call(logits, seq_lens, at, 1, k - 1, *aux)
    ref = op.reference(logits, seq_lens, at, 1, k - 1)
    _compare(at, ref, logits, lens, 1, by_score=False)


#: Columns from which the host splits every row across 10 blocks whatever the
#: row count (its forced-split threshold), and the 1M envelope's scorer width.
#: Below the threshold the split also depends on the device's SM count.
FORCED_SPLIT_WIDTHS = [200000, 1048576]


def _blocks_per_row(rows: int, width: int, k: int = 2048) -> int:
    """The host's split, from the scratch the op asks a one-element arena for.

    The op refuses that arena before any launch.
    """
    logits = torch.zeros(rows, width, device="cuda")
    lens = torch.full((rows,), width, dtype=torch.int32, device="cuda")
    indices = torch.full((rows, k), UNWRITTEN, dtype=torch.int32, device="cuda")
    one = (
        torch.empty(1, dtype=torch.int32, device="cuda"),
        torch.empty(1, dtype=torch.float32, device="cuda"),
    )
    with pytest.raises(RuntimeError, match=r"need \d+\)") as err:
        _call(logits, lens, indices, 1, k, *one)
    need = int(re.search(r"need (\d+)\)", str(err.value)).group(1))
    assert need % (rows * k) == 0, need
    return need // (rows * k)


@pytest.mark.parametrize("width", FORCED_SPLIT_WIDTHS)
def test_ten_blocks_per_row_at_every_row_count_from_200000_columns(width) -> None:
    """At 1048576 columns every decode batch 1..16 splits its rows 10 ways."""
    assert [_blocks_per_row(rows, width) for rows in range(1, 17)] == [AUX_BLOCKS] * 16


def test_sixteen_rows_at_1m_fill_the_radix_arena_exactly() -> None:
    """16 rows x 10 blocks x 2048 is the DSA arena [16, 10, 2048] to the element.

    Every element is written and nothing past it; one element less is refused by the op.
    """
    spec = dict(next_n=1, width=1 << 20, index_topk=2048, aux_rows=16)
    lens = [1 << 20, (1 << 20) - 1, 2049, 20481, 20490, 1, 524288, 1000000] * 2
    logits = _logits(spec, lens, 1, torch.Generator().manual_seed(13))
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    indices = torch.full((16, 2048), UNWRITTEN, dtype=torch.int32, device="cuda")
    aux, guard = _scratch(spec, 16)
    _call(logits, seq_lens, indices, 1, 2048, *aux)
    ref = op.reference(logits, seq_lens, indices, 1, 2048)
    _compare(indices, ref, logits, lens, 1, by_score=False)
    assert not bool((aux[0] == AUX_SENTINEL).any()), "an arena element was left unwritten"
    _assert_guard_untouched(guard)
    short = (aux[0].reshape(-1)[:-1], aux[1].reshape(-1)[:-1])
    with pytest.raises(RuntimeError, match=r"got 327679 / 327679, need 327680"):
        _call(logits, seq_lens, indices, 1, 2048, *short)
