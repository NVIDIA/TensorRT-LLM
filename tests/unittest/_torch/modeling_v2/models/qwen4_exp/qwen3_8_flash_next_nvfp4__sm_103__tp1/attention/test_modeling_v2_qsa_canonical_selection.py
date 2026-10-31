# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qsa_canonical_selection entry.

The rewrite fixes every slot of every row, so `_compare` is index equality against a
composite-key top-k. The input rows come from an adversary that keeps a random part of the
tie class at the k-th value in a random order, or from the radix engine the target calls.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_topk_select import (  # noqa: E501
    qsa_topk_select,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.qsa_canonical_selection import (  # noqa: E501
    qsa_canonical_selection as op,
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


def _logits(kind: str, rows: int, cols: int, gen: torch.Generator) -> torch.Tensor:
    if kind == "relu":
        return torch.relu(torch.randn(rows, cols, generator=gen, device=DEV)) * 3.0
    if kind == "relu_sparse":
        x = torch.relu(torch.randn(rows, cols, generator=gen, device=DEV)) * 3.0
        return x * (torch.rand(rows, cols, generator=gen, device=DEV) < 0.2)
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
