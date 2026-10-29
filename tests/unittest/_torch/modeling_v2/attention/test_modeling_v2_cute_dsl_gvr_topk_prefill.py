# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the cute_dsl_gvr_topk_prefill catalog entry.

Every cell is gated exactly: a short row (`len <= top_k`) must be `0..len-1` then -1, order
included, and a long row must be the deterministic engine's set, the scores above the row's
k-th score plus the lowest columns among those equal to it, in the same order on a second
call. Each cell is also compared with `cute_dsl_indexer_topk_prefill`, the entry it replaces
in the dep4 target: short rows identical, long rows the same multiset of selected scores (the
same set where the k-th score is not tied).
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.cute_dsl_gvr_topk_prefill import (  # noqa: E501
    cute_dsl_gvr_topk_prefill as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.cute_dsl_gvr_topk_prefill import (  # noqa: E501
    gvr_warmup,
    gvr_workspace,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.cute_dsl_indexer_topk_prefill import (  # noqa: E501
    cute_dsl_indexer_topk_prefill as radix,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.fp8_mqa_logits import (
    fp8_mqa_logits,
)
from tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k import (
    gvr_topk_decode_self_sampling as gvr_kernels,
)
from tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k import (
    gvr_topk_decode_self_sampling_host as gvr_host,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — needs the path declared above

assert torch.cuda.is_available(), "the entry's cells drive a CuTe DSL kernel on the GPU"

#: Written everywhere outside a row's window, its row padding included. A
#: kernel that read outside would rank these above every real score.
POISON = 1.0e30
HEADS, HEAD_DIM = 32, 128
#: Rows argsorted per call while drawing the permutations.
_SORT_ROWS = 1024


def _windows(requests):
    """`(row_starts, row_ends, width, rows)` for prompts packed back to back in one K block.

    Prompt `(cached, new, taken)` owns `cached + new` keys starting where the previous one
    ended; its `j`-th new token sees `[offset, offset + cached + j + 1)`. `taken` keeps only
    the prompt's last rows, like one row tile.
    """
    starts, ends, offset = [], [], 0
    for cached, new, taken in requests:
        first = 0 if taken is None else new - taken
        for j in range(first, new):
            starts.append(offset)
            ends.append(offset + cached + j + 1)
        offset += cached + new
    as_t = lambda v: torch.tensor(v, dtype=torch.int32, device="cuda")  # noqa: E731
    return as_t(starts), as_t(ends), offset


def _scores(rows: int, width: int, values: str, g: torch.Generator) -> torch.Tensor:
    """Per-row scores; "distinct", "narrow", "spread" and "wide" are distinct within a row."""
    if values == "ties":
        return torch.randint(0, 4, (rows, width), generator=g, device="cuda").float()
    perm = torch.rand(rows, width, generator=g, device="cuda")
    for lo in range(0, rows, _SORT_ROWS):
        perm[lo : lo + _SORT_ROWS] = perm[lo : lo + _SORT_ROWS].argsort(dim=1).float()
    if values == "distinct":
        return perm
    if values == "narrow":
        # Distinct in fp32, one value in fp16: every key in one coarse bin.
        return perm.mul_(2.0**-20).add_(1.0)
    assert values == "spread", values
    # 2^15 at the top, falling by a factor 2^(1/8192) per rank.
    return perm.sub_(width - 1).mul_(2.0**-13).add_(15.0).exp2_()


def _build(requests, values: str, top_k: int, seed: int):
    """`(input_values, row_starts, row_ends, output_indices)` for one K block.

    `input_values` is a `[rows, W]` view of rows padded as fp8_mqa_logits pads them,
    align(W + 256, 256) wide. "logits" scores the block with fp8_mqa_logits itself over
    random e4m3 queries and keys (-inf outside the windows, as the target hands it over);
    every other pattern writes `POISON` outside the windows. `output_indices` starts at -7
    so an unwritten slot shows.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    row_starts, row_ends, width = _windows(requests)
    rows = row_starts.shape[0]
    if values == "logits":
        q = torch.randn(rows, HEADS, HEAD_DIM, generator=g, device="cuda").to(torch.float8_e4m3fn)
        k = torch.randn(width, HEAD_DIM, generator=g, device="cuda").to(torch.float8_e4m3fn)
        k_scale = torch.rand(width, generator=g, device="cuda") + 0.5
        w = torch.randn(rows, HEADS, generator=g, device="cuda") * (HEAD_DIM * HEADS) ** -0.5
        input_values = fp8_mqa_logits(q, (k, k_scale), w, row_starts, row_ends)
    else:
        cols = torch.arange(width, device="cuda")
        outside = (cols < row_starts[:, None]) | (cols >= row_ends[:, None])
        scores = _scores(rows, width, values, g).masked_fill_(outside, POISON)
        del outside
        stride = (width + 255) // 256 * 256 + 256
        base = torch.full((rows, stride), POISON, device="cuda")
        base[:, :width] = scores
        del scores
        input_values = base[:, :width]
    out = torch.full((rows, top_k), -7, dtype=torch.int32, device="cuda")
    return input_values, row_starts, row_ends, out


def _deterministic(input_values, row_starts, row_ends, top_k: int) -> torch.Tensor:
    """Each row's deterministic set, sorted, window-local (`[rows, top_k]`, int64): the
    scores above the k-th plus the lowest columns among those equal to it; for a short row
    `0..len-1` then -1."""
    rows = input_values.shape[0]
    out = torch.full((rows, top_k), -1, dtype=torch.int64, device=input_values.device)
    for r in range(rows):
        s, e = int(row_starts[r]), int(row_ends[r])
        n = e - s
        if n <= top_k:
            out[r, : max(n, 0)] = torch.arange(max(n, 0), device=out.device)
            continue
        win = input_values[r, s:e]
        kth = torch.topk(win, top_k).values[-1]
        above = (win > kth).nonzero().flatten()
        tied = (win == kth).nonzero().flatten()[: top_k - above.numel()]
        out[r] = torch.cat([above, tied]).sort().values
    return out


def _compare_to_radix(out, other, input_values, row_starts, row_ends) -> None:
    """Short rows identical, order included; long rows the same multiset of selected scores."""
    top_k = out.shape[1]
    lengths = (row_ends - row_starts).long()
    short = lengths <= top_k
    assert torch.equal(out[short], other[short]), "a short row differs from the radix entry"
    long_rows = (~short).nonzero().flatten()
    if not long_rows.numel():
        return
    start = row_starts.long()[long_rows, None]
    mine = input_values[long_rows[:, None], out[long_rows].long() + start].sort(dim=1).values
    theirs = input_values[long_rows[:, None], other[long_rows].long() + start].sort(dim=1).values
    wrong = (mine.view(torch.int32) != theirs.view(torch.int32)).any(dim=1)
    assert not bool(wrong.any()), (
        f"{int(wrong.sum())} of {long_rows.numel()} long rows select other scores than the "
        f"radix entry"
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    top_k = cell.spec["top_k"]
    for b, requests in enumerate(cell.spec["blocks"]):
        seed = zlib.crc32(f"{cell.why}/{b}".encode()) % 2**31
        values, starts, ends, out = _build(requests, cell.spec["values"], top_k, seed)
        ws = gvr_workspace(values.device)
        got = op(values, starts, ends, out, top_k, ws)
        assert got is out
        ref = op.reference(values, starts, ends, out, top_k)
        want = _deterministic(values, starts, ends, top_k)
        lengths = (ends - starts).long()
        short = lengths <= top_k
        assert torch.equal(got[short], ref[short]), "a short row is not 0..len-1 then -1"
        assert not bool((got[~short] < 0).any()), "a long row holds -1 or an unwritten slot"
        assert torch.equal(got.long().sort(dim=1).values[~short], want[~short]), (
            "a long row is not the deterministic set"
        )
        again = op(values, starts, ends, torch.full_like(out, -7), top_k, ws)
        assert torch.equal(again, got), "a second call gives another order"
        other = radix(values, starts, ends, torch.full_like(out, -7), top_k)
        _compare_to_radix(got, other, values, starts, ends)
        print(
            f"GVR {cell.why[:60]}: block {b}, {values.shape[0]} rows x {values.shape[1]}, "
            f"{int(short.sum())} short rows, radix set-equal rows "
            f"{int((other.long().sort(dim=1).values == got.long().sort(dim=1).values).all(dim=1).sum())}"
        )


def _ok():
    return _build([(4096, 64, None)], "distinct", 2048, seed=3)


def test_engine_workspace_default_selects_the_same() -> None:
    """Without a slab the engine takes its per-device one: the same selection."""
    values, starts, ends, out = _ok()
    mine = op(values, starts, ends, out, 2048, gvr_workspace(values.device))
    theirs = op(values, starts, ends, torch.full_like(out, -7), 2048)
    assert torch.equal(mine, theirs)


def test_guard_refuses_another_width() -> None:
    """k is the output's width, not `top_k`."""
    values, starts, ends, _ = _ok()
    wide = torch.full((values.shape[0], 4096), -7, dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="exactly top_k"):
        with validating(op):
            op(values, starts, ends, wide, 2048)


def test_guard_refuses_a_window_past_the_block() -> None:
    """The engine reads past W into the row padding and ranks it."""
    values, starts, ends, out = _ok()
    with pytest.raises(AssertionError, match="past the"):
        with validating(op):
            op(values, starts, ends + 300, out, 2048)
    with pytest.raises(AssertionError, match="starts before its row"):
        with validating(op):
            op(values, starts - 1, ends, out, 2048)


def _engine_caches() -> tuple:
    """The engine's launcher keys and compiled variants: what a call that compiles adds to."""
    return set(gvr_host._PREFILL_CACHE), set(gvr_kernels._COMPILE_CACHE)


@pytest.mark.parametrize("width", [1, 7, 2047, 2048])
def test_a_block_within_top_k_is_answered_without_the_engine(width) -> None:
    """Every row of a block of at most top_k columns is short: the identity rows, as the radix
    entry writes them, with no engine launch (nothing compiles for the narrow widths)."""
    values, starts, ends, out = _build([(0, width, None), (0, 3, None)], "distinct", 2048, 5)
    before = _engine_caches()
    got = op(values, starts, ends, out, 2048, gvr_workspace(values.device))
    assert got is out and _engine_caches() == before
    other = radix(values, starts, ends, torch.full_like(out, -7), 2048)
    assert torch.equal(got, other)


#: The targets' engines: (max_seq_len, logits budget): dep4 at the 1M config, tep4 at the 1M
#: config, both at the 32K config.
_ENGINES = {"dep4-1m": (1048576, 2**26), "tep4-1m": (1048576, 2**28), "32k": (32768, 2**24)}


@pytest.mark.parametrize("engine", list(_ENGINES))
def test_warmup_leaves_nothing_to_compile(engine, monkeypatch) -> None:
    """After `gvr_warmup`, the target's tile shapes compile nothing: every tier, each tier-0
    width bucket and both sides of each tier's stride bound (65536 / 131072 / 262144 columns),
    full tiles of floor(budget / W) rows and short remainder tiles, up to max_seq_len keys.
    Each engine starts from empty engine caches, as a server process does: what another
    engine's warmup or an earlier test compiled must not hide a launch this one misses."""
    monkeypatch.setattr(gvr_host, "_PREFILL_CACHE", {})
    monkeypatch.setattr(gvr_kernels, "_COMPILE_CACHE", {})
    max_cols, budget = _ENGINES[engine]
    dev = torch.device("cuda")
    ws = gvr_workspace(dev)
    calls = gvr_warmup(2048, max_cols, budget, dev, ws)
    before = _engine_caches()
    widths = [2049, 3000, 4096, 4097, 8192, 8448, 8449, 16384, 16385, 32768, 65280, 65281,
              65536, 98304, 131072, 131073, 225000, 262144, 262145, 450000, 524288, 557056,
              1015808, 1048559, 1048576]  # fmt: skip
    shapes = []
    for width in (w for w in widths if w <= max_cols):
        full = max(1, budget // width)
        for rows in {full, 1, 75, 148, 149, 296, 297}:
            if rows <= full:
                shapes.append((rows, width))
    for rows, width in shapes:
        stride = (width + 255) // 256 * 256 + 256
        scores = torch.rand(rows, stride, device=dev)
        starts = torch.zeros(rows, dtype=torch.int32, device=dev)
        ends = (width - rows + 1 + torch.arange(rows, device=dev)).clamp(min=1).to(torch.int32)
        out = torch.empty(rows, 2048, dtype=torch.int32, device=dev)
        op(scores[:, :width], starts, ends, out, 2048, ws)
        del scores
    torch.cuda.synchronize()
    after = _engine_caches()
    print(f"GVR warmup {engine}: {calls} warm calls, then {len(shapes)} tile shapes; launcher keys "
          f"{sorted(after[0])}, compiled variants {len(after[1])}")  # fmt: skip
    assert after == before, (
        f"a tile shape compiled after the warmup: new launchers {after[0] - before[0]}, "
        f"new variants {len(after[1] - before[1])}"
    )
