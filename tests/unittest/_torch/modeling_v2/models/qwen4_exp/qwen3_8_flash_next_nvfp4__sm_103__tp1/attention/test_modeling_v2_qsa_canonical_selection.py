# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qsa_canonical_selection entry.

The rewrite fixes every slot of every row, so `_compare` is index equality against a
composite-key top-k. The input rows come from an adversary that keeps a random part of the
tie class at the k-th value in a random order, or from the radix engine the target calls.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.qsa_canonical_selection import (  # noqa: E501
    qsa_canonical_selection as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.qsa_topk_select import (  # noqa: E501
    qsa_topk_select,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "qsa_canonical_selection requires a CUDA device"

DEV = "cuda"
COMPRESS_RATIO = 4
BLOCK_TOPK = 512
COLS = 2080
#: What a padding slot sorts under when a raw selection is put in ascending order: past
#: every column, so padding stays last.
PAD_KEY = torch.iinfo(torch.int32).max


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


def _logits(kind: str, rows: int, cols: int, gen: torch.Generator) -> torch.Tensor:
    if kind == "relu":
        return torch.relu(torch.randn(rows, cols, generator=gen, device=DEV)) * 3.0
    if kind == "relu_sparse":
        x = torch.relu(torch.randn(rows, cols, generator=gen, device=DEV)) * 3.0
        return x * (torch.rand(rows, cols, generator=gen, device=DEV) < 0.2)
    if kind == "signed_zeros":
        x = _logits("relu_sparse", rows, cols, gen)
        negative = torch.rand(rows, cols, generator=gen, device=DEV) < 0.5
        return torch.where((x == 0) & negative, torch.full_like(x, -0.0), x)
    if kind == "pair_at_k":
        x = torch.rand(rows, cols, generator=gen, device=DEV).argsort(1).float() + 1.0
        order = torch.argsort(x[:, :2048], dim=1, descending=True)
        kth, nxt = order[:, BLOCK_TOPK - 1 : BLOCK_TOPK], order[:, BLOCK_TOPK : BLOCK_TOPK + 1]
        x.scatter_(1, nxt, torch.gather(x, 1, kth))
        return x
    raise ValueError(kind)


def _windows(spec: dict, rows: int, cols: int, gen: torch.Generator):
    kind = spec["windows"][0]
    starts = torch.zeros(rows, dtype=torch.int32, device=DEV)
    if kind in ("chunk", "positions_random"):
        if kind == "chunk":
            positions = spec["windows"][1] + torch.arange(rows, device=DEV)
        else:
            lo, hi = spec["windows"][1:]
            positions = torch.randint(lo, hi + 1, (rows,), generator=gen, device=DEV)
        ends = ((positions + 1) // COMPRESS_RATIO).clamp_max(cols).to(torch.int32)
    else:
        assert kind == "ends", kind
        ends = torch.tensor(spec["windows"][1], dtype=torch.int32, device=DEV)
    assert starts.numel() == rows and ends.numel() == rows
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
    tiebreak = torch.rand(rows, cols, generator=gen, device=DEV).argsort(1)
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


def _sorted(raw: torch.Tensor) -> torch.Tensor:
    """A raw selection in ascending order, padding last."""
    keyed = torch.where(raw < 0, PAD_KEY, raw)
    ordered = torch.sort(keyed, dim=1).values
    return torch.where(ordered == PAD_KEY, -1, ordered)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    spec = cell.spec
    seed = sum(map(ord, cell.why))
    row_counts = spec["rows"] if isinstance(spec["rows"], tuple) else (spec["rows"],)
    for rows in row_counts:
        args = _build(spec, rows, seed + rows)
        expected = op.reference(*args)
        out = op(*args)
        assert out is args[3], "the rewrite writes `indices` and returns it"
        _compare(out, expected)
    if not spec.get("graph"):
        return

    # Captured once, replayed against new scores, new causal bounds and a new engine
    # selection through the same buffers.
    logits, starts, ends, indices = _build(spec, rows, seed)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(logits, starts, ends, indices)
    for replay in range(2):
        fresh = _build(spec, rows, seed + 1000 + replay)
        for static, new in zip((logits, starts, ends, indices), fresh):
            static.copy_(new)
        expected = op.reference(logits, starts, ends, indices)
        graph.replay()
        torch.cuda.synchronize()
        _compare(indices, expected)


def test_tie_free_rows_equal_the_radix_engines_sorted_selection() -> None:
    """Where the k-th value has no tie, the rewrite is the engine's own set in ascending
    order, bit for bit."""
    gen = torch.Generator(device=DEV).manual_seed(4)
    rows = 256
    logits = torch.randn(rows, COLS, generator=gen, device=DEV)
    positions = torch.randint(0, 8192, (rows,), generator=gen, device=DEV)
    ends = ((positions + 1) // COMPRESS_RATIO).clamp_max(COLS).to(torch.int32)
    starts = torch.zeros(rows, dtype=torch.int32, device=DEV)
    _poison(logits, starts, ends, "nan")
    window = torch.where(
        torch.arange(COLS, device=DEV)[None, :] < ends[:, None], logits, float("-inf")
    )
    top = torch.topk(window, BLOCK_TOPK + 1, dim=1).values
    long_rows = ends > BLOCK_TOPK
    assert bool((top[long_rows, -2] != top[long_rows, -1]).all()), "a tie at k: reseed"
    assert int(long_rows.sum()) > 100 and int((~long_rows).sum()) > 10
    raw = _radix(logits, starts, ends)
    expected = _sorted(raw)
    assert torch.equal(op(logits, starts, ends, raw), expected)


def test_narrowing_the_table_to_a_bound_on_the_windows_changes_nothing() -> None:
    """A caller that knows every window ends by column 1500 passes `logits[:, :1500]` of a
    wide table: the rows are the full table's, and a 2^24-element table is tiled."""
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


@pytest.mark.parametrize("kind", ["relu_sparse", "pair_at_k"])
def test_radix_launches_differ_and_canonical_rows_do_not(kind) -> None:
    """Repeated radix launches over one tie-heavy score matrix. The raw rows differ from
    launch to launch -- in order, and where the tie class at the k-th score straddles the
    budget, in which tied columns are kept -- and the canonical row is one and the same
    row, the reference's, for every launch of every row. Zero ties as the target's relu
    scores have them, and a positive pair tied exactly at the k-th value."""
    gen = torch.Generator(device=DEV).manual_seed(17)
    rows, launches = 64, 20
    logits = _logits(kind, rows, COLS, gen)
    starts = torch.zeros(rows, dtype=torch.int32, device=DEV)
    ends = torch.full((rows,), 2048, dtype=torch.int32, device=DEV)
    _poison(logits, starts, ends, "nan")
    expected = op.reference(logits, starts, ends, torch.empty(rows, BLOCK_TOPK, device=DEV))
    orders = [set() for _ in range(rows)]
    sets = [set() for _ in range(rows)]
    canonical = [set() for _ in range(rows)]
    for _ in range(launches):
        raw = _radix(logits, starts, ends)
        for r, (row, ordered) in enumerate(zip(raw.cpu().tolist(), _sorted(raw).cpu().tolist())):
            orders[r].add(tuple(row))
            sets[r].add(tuple(ordered))
        out = op(logits, starts, ends, raw)
        for r, row in enumerate(out.cpu().tolist()):
            canonical[r].add(tuple(row))
        _compare(out, expected)
    assert any(len(o) > 1 for o in orders), "radix returned one order on every launch"
    assert all(len(c) == 1 for c in canonical)


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
