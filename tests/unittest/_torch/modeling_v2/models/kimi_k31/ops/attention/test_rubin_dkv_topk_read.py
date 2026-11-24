# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of rubin_dkv_topk_read, the DKV sparse MLA read: the history rows a rank's Top-K selects,
copied once each behind the step's new tokens, and the Top-K as rows of that buffer; and of
rubin_dkv_topk_read_fused, the same read with the remap, the history entries, the -1 padding (to
row 0) and the zeroing of the buffer's page 0 in one CuTe DSL launch.

CPU: the reference on a hand-checked selection (dedup, order, -1, new tokens, a request without a
history), the guards; the fused reference on a hand-checked step (history, new tokens, padding,
table holes, a page past the table, page 0); the fused kernel's per-entry rule, in int32 torch
arithmetic, against the fused reference (the plain steps) on every fused cell at CPU sizes.
GPU (sm_107): every cell, the kernel's buffer rows and indices bit for bit against the reference;
rows past the selection keep their bytes; nothing waits for the host. Every fused cell bit for bit
against its reference and against the plain GPU path the cep target ran (page 0's zero_, the
Triton remap from 256 rows and the C++ one below, rubin_dkv_topk_read, clamp), over a buffer of
E4M3 NaN bytes: page 0 ends zero, the new tokens' rows and the rows past the selection keep their
bytes; the index kernel's three thread layouts with and without history tokens and PDL, and replayed
from a CUDA graph.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.attention.rubin_dkv_topk_read import (  # noqa: E501
    topk_rows_into,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.convert_req_index_to_global import (  # noqa: E501
    convert_req_index_to_global,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_cp_dkv import (  # noqa: E501
    new_token_table,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_dkv_topk_read import (  # noqa: E501
    rubin_dkv_topk_read as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_dkv_topk_read import (  # noqa: E501
    rubin_dkv_topk_read_fused as fused,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_dsa_topk_remap import (  # noqa: E501
    rubin_dsa_topk_remap,
)

__extra_import_path__ = ["../../../..", "."]
from _validating import certifying, validating  # noqa: E402,F401

on_sm107 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the Rubin family entries are certified on sm_107",
)

SENTINEL = 0xA5
#: An E4M3 NaN byte: the fused cells' buffer starts as all NaN, so page 0's zeros are the op's.
POISON = 0x7F
#: The cep target's row count from which its remap is the Triton kernel (C++ below).
TRITON_ROWS = 256


def _inputs(spec, device):
    """The read's inputs for a cell: a source of random rows (every rank's pool through the
    mapping), each history token's distinct source row, each row's Top-K over its causal window
    (all of it up to the Top-K, else a random subset; ascending, then -1), new-token rows past
    the history, and a buffer of sentinel bytes."""
    gen = torch.Generator().manual_seed(spec["seed"])
    row_bytes, k = spec["row_bytes"], spec["topk"]
    cached, new = spec["cached"], spec["new"]
    holders = dict(spec["rows"])
    lens = [c if r in holders else 0 for r, c in enumerate(cached)]
    starts = [sum(lens[:r]) for r in range(len(lens))]
    tokens = sum(lens)
    src = torch.randint(0, 256, (2 * tokens + 64, row_bytes), dtype=torch.uint8, generator=gen)
    hist_src = torch.randperm(2 * tokens + 32, generator=gen)[:tokens].to(torch.int64)
    topk, req = [], []
    new_rows = 64 * sum(-(-(c + n) // 64) for c, n in zip(cached, new))
    for r, count in spec["rows"]:
        c, n = cached[r], new[r]
        for q in range(c + n - count, c + n):
            visible = q + 1
            pick = (
                torch.arange(visible)
                if visible <= k
                else torch.randperm(visible, generator=gen)[:k].sort().values
            )
            row = torch.full((k,), -1, dtype=torch.int32)
            row[: pick.numel()] = pick.to(torch.int32)
            topk.append(row)
            req.append(r)
    topk = torch.stack(topk)
    req = torch.tensor(req, dtype=torch.int32)
    # New tokens' rows: distinct per (request, position), inside the buffer's head.
    base = torch.tensor([sum(cached[:r]) + sum(new[:r]) for r in range(len(cached))])
    new_idx = torch.where(topk >= 0, (base[req.long()].unsqueeze(1) + topk) % new_rows, -1)
    bound = min(tokens, topk.numel())
    kv = torch.full((new_rows + bound, row_bytes), SENTINEL, dtype=torch.uint8)
    args = (
        kv,
        new_rows,
        topk,
        req,
        new_idx.to(torch.int32),
        torch.tensor(starts, dtype=torch.int32),
        torch.tensor(lens, dtype=torch.int32),
        hist_src,
        src,
        int(spec.get("row_offset", 16)),
    )
    return tuple(a.to(device) if isinstance(a, torch.Tensor) else a for a in args)


# --- CPU -------------------------------------------------------------------------------------


def test_reference_copies_each_selected_row_once_in_order():
    row = 16
    src = torch.arange(40 * row, dtype=torch.int32).to(torch.uint8).view(40, row)
    # Request 0: history 6 (rows 10..15 of src at offset 2 -> 12..17); request 1: none here.
    hist_src = torch.arange(10, 16, dtype=torch.int64)
    topk = torch.tensor([[5, 1, 5, -1], [7, 1, 0, 6]], dtype=torch.int32)  # 6, 7: new tokens
    req = torch.zeros(2, dtype=torch.int32)
    # The new tokens' rows (the remap of their pages; -1 where the Top-K is negative).
    new_idx = torch.full((2, 4), 100, dtype=torch.int32)
    new_idx[0, 3], new_idx[1, 0], new_idx[1, 3] = -1, 3, 2
    kv = torch.zeros(4 + 6, row, dtype=torch.uint8)
    out, idx = op.reference(
        kv,
        4,
        topk,
        req,
        new_idx,
        torch.tensor([0, 6], dtype=torch.int32),
        torch.tensor([6, 0], dtype=torch.int32),
        hist_src,
        src,
        2,
    )
    # Selected history tokens 0, 1, 5 -> buffer rows 4, 5, 6, from src rows 12, 13, 17.
    assert idx.tolist() == [[6, 5, 6, -1], [3, 5, 4, 2]]
    assert torch.equal(out[4:7], src[[12, 13, 17]])
    assert (out[:4] == 0).all() and (out[7:] == 0).all()


def test_reference_without_history_hands_back_the_new_rows():
    topk = torch.tensor([[0, 1, -1]], dtype=torch.int32)
    new_idx = torch.tensor([[7, 8, -1]], dtype=torch.int32)
    kv = torch.zeros(9, 16, dtype=torch.uint8)
    _, idx = op.reference(
        kv,
        9,
        topk,
        torch.zeros(1, dtype=torch.int32),
        new_idx,
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.empty(0, dtype=torch.int64),
        torch.zeros(4, 16, dtype=torch.uint8),
        0,
    )
    assert torch.equal(idx, new_idx)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_cells_are_valid_inputs(cell):
    spec = dict(cell.spec, cached=[min(c, 3000) for c in cell.spec["cached"]])
    args = _inputs(spec, "cpu")
    op.is_valid(*args)


def test_guards():
    args = list(_inputs(op.CELLS[1].spec, "cpu"))
    op.is_valid(*args)
    short = list(args)
    short[0] = args[0][: args[1] + 1]
    with pytest.raises(AssertionError, match="every row"):
        op.is_valid(*short)
    odd = list(args)
    odd[0] = torch.zeros(args[0].shape[0], 24, dtype=torch.uint8)
    odd[8] = torch.zeros(4, 24, dtype=torch.uint8)
    with pytest.raises(AssertionError, match="16-byte"):
        op.is_valid(*odd)
    wide = list(args)
    wide[2] = args[2].long()
    with pytest.raises(AssertionError):
        op.is_valid(*wide)


# --- GPU -------------------------------------------------------------------------------------


@on_sm107
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _inputs(cell.spec, "cuda")
    ref_kv, ref_idx = op.reference(*args)
    kv = args[0]
    torch.cuda.set_sync_debug_mode("error")
    try:
        idx = op(*args)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    torch.cuda.synchronize()
    assert torch.equal(idx, ref_idx)
    assert torch.equal(kv, ref_kv), "buffer rows differ (copied rows, or rows past the selection)"


@on_sm107
def test_a_selection_far_below_the_bound_copies_only_the_selection():
    """One row selecting 3 of 50000 history tokens into a buffer sized for all of them."""
    spec = dict(op.CELLS[0].spec, cached=[50000], new=[1], rows=[(0, 1)], topk=2048)
    args = list(_inputs(spec, "cuda"))
    args[2] = torch.full_like(args[2], -1)
    args[2][0, :3] = torch.tensor([7, 49999, 123], dtype=torch.int32, device="cuda")
    args[4] = torch.full_like(args[4], -1)
    kv = torch.full((args[1] + 50000, args[0].shape[1]), SENTINEL, dtype=torch.uint8, device="cuda")
    args[0] = kv
    ref_kv, ref_idx = op.reference(*args)
    with validating(op):
        idx = op(*args)
    torch.cuda.synchronize()
    assert torch.equal(idx, ref_idx) and torch.equal(kv, ref_kv)
    assert (kv[args[1] + 3 :] == SENTINEL).all()


# --- rubin_dkv_topk_read_fused -----------------------------------------------------------------


def _topk_rows(spec, gen):
    """Each listed request's last rows: row ``q`` holds its Top-K over positions ``[0, q]``, all of
    them (ascending, then -1) up to the Top-K width, else random positions in ascending order;
    ``pad_rows`` first rows all -1."""
    k = spec["topk"]
    cols = torch.arange(k)
    topk, req = [], []
    for r, count in spec["rows"]:
        c, n = spec["cached"][r], spec["new"][r]
        visible = torch.arange(c + n - count, c + n) + 1
        picks = (torch.rand(count, k, generator=gen, dtype=torch.float64) * visible[:, None]).long()
        picks = torch.minimum(picks, visible[:, None] - 1).sort(dim=1).values
        full = torch.where(cols < visible[:, None], cols, -1)
        topk.append(torch.where((visible > k)[:, None], picks, full))
        req.append(torch.full((count,), r))
    topk = torch.cat(topk).to(torch.int32)
    topk[: spec.get("pad_rows", 0)] = -1
    return topk.contiguous(), torch.cat(req).to(torch.int32)


def _fused_inputs(spec, device):
    """The fused read's inputs for a cell: as ``_inputs``, with the plan's new-token table (pages of
    ``block_size`` tokens, page 0 for the history) in place of the remapped Top-K, and a buffer of
    E4M3 NaN bytes whose new tokens' rows (pages 1 and up) hold random bytes, as the append or the
    owners' copy leaves them. ``holes``: some table entries -1 and the table cut short of the
    longest request's last two pages."""
    gen = torch.Generator().manual_seed(spec["seed"])
    row_bytes, bs = spec["row_bytes"], spec["block_size"]
    cached, new = spec["cached"], spec["new"]
    holders = dict(spec["rows"])
    lens = [c if r in holders else 0 for r, c in enumerate(cached)]
    starts = [sum(lens[:r]) for r in range(len(lens))]
    tokens = sum(lens)
    src = torch.randint(0, 256, (2 * tokens + 64, row_bytes), dtype=torch.uint8, generator=gen)
    hist_src = torch.randperm(2 * tokens + 32, generator=gen)[:tokens].to(torch.int64)
    topk, req = _topk_rows(spec, gen)
    table, pages = new_token_table(cached, new, bs, spec["width"])
    if spec.get("holes"):
        table = table[:, : max(1, table.shape[1] - 2)].clone()
        table[torch.rand(table.shape, generator=gen) < 0.2] = -1
    new_rows = pages * bs
    bound = min(tokens, topk.numel())
    kv = torch.full((new_rows + bound, row_bytes), POISON, dtype=torch.uint8)
    kv[bs:new_rows] = torch.randint(
        0, 256, (new_rows - bs, row_bytes), dtype=torch.uint8, generator=gen
    )
    args = (
        kv,
        new_rows,
        topk,
        req,
        table,
        bs,
        bs,
        0,
        torch.tensor(starts, dtype=torch.int32),
        torch.tensor(lens, dtype=torch.int32),
        hist_src,
        src,
        int(spec.get("row_offset", 16)),
    )
    return tuple(a.to(device) if isinstance(a, torch.Tensor) else a for a in args)


def _cpu_spec(spec):
    """A cell at CPU sizes: at most 3000 cached and 600 new tokens a request."""
    new = [min(n, 600) for n in spec["new"]]
    rows = [(r, min(c, new[r])) for r, c in spec["rows"]]
    return dict(spec, cached=[min(c, 3000) for c in spec["cached"]], new=new, rows=rows)


def _plain(kv, new_rows, topk, req, table, bs, stride, layer, starts, lens, hist_src, src, off):
    """The cep target's plain steps on the GPU: page 0's zero_, the remap (Triton from 256 rows,
    C++ below), rubin_dkv_topk_read, clamp."""
    kv[:bs].zero_()
    remap = rubin_dsa_topk_remap if topk.shape[0] >= TRITON_ROWS else convert_req_index_to_global
    new_idx = remap(req, table, topk, bs, topk.shape[1], stride, layer)
    idx = op(kv, new_rows, topk, req, new_idx, starts, lens, hist_src, src, off)
    return idx.clamp(min=0)


def _kernel_rule(new_rows, topk, req, table, bs, stride, layer, starts, lens, rank):
    """DkvTopkRowsKernel's per-entry rule in int32 torch arithmetic: a history entry's rank plus
    ``new_rows - 1``, else its page's table entry as a row, else 0."""
    q = req.long().unsqueeze(1)
    width = table.shape[1]
    page = torch.where(topk >= 0, topk >> (bs.bit_length() - 1), 0)
    entry = table[q, page.clamp(max=width - 1).long()]
    out = torch.where(
        (topk >= 0) & (page < width) & (entry >= 0),
        entry * stride + layer * bs + (topk & (bs - 1)),
        0,
    )
    if rank is not None:
        hist = (topk >= 0) & (topk < lens[q])
        token = torch.where(hist, starts[q] + topk, 0).long()
        out = torch.where(hist, rank[token] + (new_rows - 1), out)
    return out.to(torch.int32)


def _rank(topk, req, starts, lens, tokens):
    """The history tokens' running count of selected ones (rubin_dkv_topk_read's selection)."""
    q = req.long().unsqueeze(1)
    hist = (topk >= 0) & (topk < lens[q])
    picked = torch.zeros(tokens + 1, dtype=torch.int32)
    picked[torch.where(hist, topk + starts[q], tokens).view(-1).long()] = 1
    return torch.cumsum(picked, 0, dtype=torch.int32)


def test_fused_reference_on_a_hand_checked_step():
    # Pages of 4 tokens. Request 0: 6 cached (its history here), 5 new; request 1: 3 new, none
    # cached. The new-token table: request 0's pages 1, 2 -> buffer pages 1, 2 (page 0 for the
    # history), request 1's page 0 -> buffer page 3; then a hole in request 0's page 2.
    table, pages = new_token_table([6, 0], [5, 3], 4)
    assert table.tolist() == [[0, 1, 2], [3, 0, 0]] and pages == 4
    table[0, 2] = -1
    row = 16
    src = torch.arange(40 * row, dtype=torch.int32).to(torch.uint8).view(40, row)
    hist_src = torch.arange(10, 16, dtype=torch.int64)  # history token p: src row 10 + p (+ 2)
    topk = torch.tensor(
        [
            [5, 1, 5, -1, 7, 6, 10, 12],  # history 5, 1; new 7, 6; 10 on the hole; 12 past
            [0, 2, -1, -1, -1, -1, -1, -1],  # request 1: new tokens only
        ],
        dtype=torch.int32,
    )
    req = torch.tensor([0, 1], dtype=torch.int32)
    kv = torch.full((16 + 6, row), POISON, dtype=torch.uint8)
    out, idx = fused.reference(
        kv,
        16,
        topk,
        req,
        table,
        4,
        4,
        0,
        torch.tensor([0, 6], dtype=torch.int32),
        torch.tensor([6, 0], dtype=torch.int32),
        hist_src,
        src,
        2,
    )
    # History tokens 1, 5 -> rows 16, 17; position 7 -> page 1 row 3 (7), 6 -> row 6; 10 (the
    # hole), 12 (past the table) and every -1 -> row 0; request 1's 0 and 2 -> page 3: 12, 14.
    assert idx.tolist() == [[17, 16, 17, 0, 7, 6, 0, 0], [12, 14, 0, 0, 0, 0, 0, 0]]
    assert (out[:4] == 0).all(), "page 0 zeroed"
    assert (out[4:16] == POISON).all(), "the new tokens' rows untouched"
    assert torch.equal(out[16:18], src[[13, 17]]) and (out[18:] == POISON).all()


@pytest.mark.parametrize("cell", fused.CELLS, ids=[c.why[:48] for c in fused.CELLS])
def test_fused_kernel_rule_is_the_plain_steps(cell):
    """The kernel's per-entry rule equals the fused reference (zero_, remap, read, clamp)."""
    args = _fused_inputs(_cpu_spec(cell.spec), "cpu")
    fused.is_valid(*args)
    kv, new_rows, topk, req, table, bs, stride, layer, starts, lens, hist_src, src, off = args
    ref_kv, ref_idx = fused.reference(*args)
    tokens = hist_src.numel()
    rank = _rank(topk, req, starts, lens, tokens) if tokens else None
    got = _kernel_rule(new_rows, topk, req, table, bs, stride, layer, starts, lens, rank)
    assert torch.equal(got, ref_idx)
    # The clamp changes only the remap's -1s: every history row is at least new_rows > 0.
    assert (ref_idx >= 0).all() and (ref_idx < kv.shape[0]).all()


def test_fused_guards():
    args = list(_fused_inputs(_cpu_spec(fused.CELLS[2].spec), "cpu"))
    fused.is_valid(*args)
    narrow = list(args)
    narrow[2] = args[2][:, :1000].contiguous()
    with pytest.raises(AssertionError, match="multiple of 512"):
        fused.is_valid(*narrow)
    odd = list(args)
    odd[5] = 48
    with pytest.raises(AssertionError, match="2\\^n"):
        fused.is_valid(*odd)
    far = list(args)
    far[3] = args[3].clone()
    far[3][0] = args[4].shape[0]
    with pytest.raises(AssertionError, match="past the block table"):
        fused.is_valid(*far)
    short = list(args)
    short[0] = args[0][: args[1] + 1]
    with pytest.raises(AssertionError, match="every row"):
        fused.is_valid(*short)
    head = list(args)
    head[1] = args[5] - 1
    with pytest.raises(AssertionError, match="page 0"):
        fused.is_valid(*head)


@on_sm107
@pytest.mark.parametrize("cell", fused.CELLS, ids=[c.why[:48] for c in fused.CELLS])
def test_fused_certified_cells(cell, certifying):  # noqa: F811
    """Indices and every buffer byte: the op = its reference = the plain GPU path."""
    certifying(fused)
    args = _fused_inputs(cell.spec, "cuda")
    kv, new_rows, bs = args[0], args[1], args[5]
    before = kv.clone()
    ref_kv, ref_idx = fused.reference(*args)
    plain_kv = kv.clone()
    plain_idx = _plain(plain_kv, *args[1:])
    idx = fused(*args)
    torch.cuda.synchronize()
    assert torch.equal(idx, ref_idx), "indices differ from the reference"
    assert torch.equal(kv, ref_kv), "buffer bytes differ from the reference"
    assert torch.equal(idx, plain_idx), "indices differ from the plain path"
    assert torch.equal(kv, plain_kv), "buffer bytes differ from the plain path"
    assert (kv[:bs] == 0).all() and (before[:bs] == POISON).all()
    assert torch.equal(kv[bs:new_rows], before[bs:new_rows]), "the new tokens' rows changed"


@on_sm107
@pytest.mark.parametrize("cell", (0, 1, 6), ids=["history", "fresh", "one row"])
def test_fused_waits_for_nothing_on_the_host(cell):
    spec = fused.CELLS[cell].spec
    fused(*_fused_inputs(spec, "cuda"))  # compile outside the check
    args = _fused_inputs(spec, "cuda")
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        idx = fused(*args)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    _, ref_idx = fused.reference(*args)
    assert torch.equal(idx, ref_idx)


def _rows_args(spec):
    """``topk_rows_into``'s inputs for a cell with history, its rank from the selection."""
    args = _fused_inputs(spec, "cpu")
    kv, new_rows, topk, req, table, bs, stride, layer, starts, lens, hist_src, _, _ = args
    rank = _rank(topk, req, starts, lens, hist_src.numel())
    cuda = [a.cuda() for a in (kv, topk, req, table, starts, lens, rank)]
    return dict(new_rows=new_rows, bs=bs, stride=stride, layer=layer), cuda


def _rows_want(meta, topk, req, table, starts, lens, rank):
    return _kernel_rule(
        meta["new_rows"],
        topk.cpu(),
        req.cpu(),
        table.cpu(),
        meta["bs"],
        meta["stride"],
        meta["layer"],
        starts.cpu(),
        lens.cpu(),
        None if rank is None else rank.cpu(),
    )


@on_sm107
@pytest.mark.parametrize("history", (True, False), ids=["history", "no_history"])
@pytest.mark.parametrize("row_vecs", (1, 2, 4))
def test_rows_kernel_with_and_without_pdl(row_vecs, history):
    """Every vectors-per-thread variant (the row count picks one), with and without history
    tokens (without: the -1 padding and every entry from the table), in both launch modes."""
    meta, (kv, topk, req, table, starts, lens, rank) = _rows_args(fused.CELLS[2].spec)
    rank = rank if history else None
    assert (topk < 0).any(), "the cell's Top-K has -1 padding"
    want = _rows_want(meta, topk, req, table, starts, lens, rank)
    bs = meta["bs"]
    for use_pdl in (False, True):
        buf = kv.clone()
        out = torch.full_like(topk, -7)
        topk_rows_into(
            out,
            buf,
            bs,
            topk,
            req,
            table,
            bs,
            meta["stride"],
            meta["layer"],
            starts,
            lens,
            rank,
            meta["new_rows"] - 1,
            use_pdl=use_pdl,
            row_vecs=row_vecs,
        )
        torch.cuda.synchronize()
        assert torch.equal(out.cpu(), want), f"use_pdl={use_pdl}"
        assert (buf[:bs] == 0).all() and torch.equal(buf[bs:], kv[bs:])


@on_sm107
def test_rows_kernel_replays_from_a_cuda_graph():
    """The launch reads no host value: a captured launch replays over new inputs in place."""
    meta, (kv, topk, req, table, starts, lens, rank) = _rows_args(fused.CELLS[2].spec)
    bs = meta["bs"]
    out = torch.empty_like(topk)
    call = (out, kv, bs, topk, req, table, bs, meta["stride"], meta["layer"], starts, lens, rank)
    topk_rows_into(*call, meta["new_rows"] - 1)  # compile outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        topk_rows_into(*call, meta["new_rows"] - 1)
    # Other Top-K rows (the same rows rolled by one), their selection, page 0 poisoned again.
    topk.copy_(torch.roll(topk, 1, dims=0))
    req.copy_(torch.roll(req, 1, dims=0))
    rank.copy_(_rank(topk.cpu(), req.cpu(), starts.cpu(), lens.cpu(), rank.numel() - 1))
    kv[:bs] = POISON
    graph.replay()
    torch.cuda.synchronize()
    want = _rows_want(meta, topk, req, table, starts, lens, rank)
    assert torch.equal(out.cpu(), want) and (kv[:bs] == 0).all()
