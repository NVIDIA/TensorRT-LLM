# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flash_mla_sparse_decode catalog entry.

The kernel computes in bf16 (query, dequantized keys, P) with fp32 accumulation; the
reference is fp32 throughout. Band: relative L2 per (row, head) under 1e-2 and the whole
call under 4e-3 -- bf16's 2^-8 relative step, softened by the 512-wide sums. The 1M cells
keep that band, each driven by the harness of the run that certified it, named by its why
("1M tp4"; "1M tep4", "1M tep8" and "1M dep4" share one, dep4's extension of the tep4 and
tep8 one). Their pools are 0xFF (e4m3, fp32 and bf16 NaN) wherever the requests hold no
key, so every output row of theirs is first checked finite: a key read from anywhere else
turns its row NaN. A call of more than 1024 rows is gated against the reference on 1024 of
them, its last 128 and 896 drawn (the reference loops over rows); under the tep4 / tep8 /
dep4 harness a context call's rows FlashMLA's scheduler split over their keys
(`num_splits` > 1, the parts merged in fp32) are gated too. A "1M tp4" ragged-tail cell
gates every row of every row count it drives, against one reference run over the largest.
A "1M tep4" or "1M tep8" `ragged_tail` cell runs one sub-batch at many row counts (the
tep8 cells' largest first): the reference runs once over its rows and every row of every
call is gated, and a row the decode scheduler kept in one part is also bitwise the whole
sub-batch's (a split row is merged in fp32 and rounds differently, inside the band). A
"1M dep4" ragged-tail cell runs one sub-batch at many row counts too: the reference and a
whole-sub-batch call run once over its rows, each call's split rows and every row not
bitwise the whole call's are gated, every other row must be bitwise the whole call's, and
each call's rel L2 counts every row.
"""

import collections
import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.flash_mla_sparse_decode import (
    flash_mla_sparse_decode as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.bmm_out import bmm_out
from tensorrt_llm.flash_mla import get_mla_metadata

__extra_import_path__ = [".."]
from _ds_mla_pool import ds_mla_rows  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401 — needs the path declared above

assert torch.cuda.is_available(), "flash_mla_sparse_decode requires a CUDA device"

SCALE = 128**-0.5


def _build(spec, seed):
    g = torch.Generator().manual_seed(seed)
    rows, page, owners, pages = spec["rows"], spec["page_size"], spec["owners"], spec["pages"]
    blocks = math.ceil(pages / owners) + 1
    kv = ds_mla_rows(blocks * owners * page, g, nope=spec.get("nope", False))
    kv = kv.view(blocks * owners, page, 1, 656)
    row = 1
    kv_len = spec["kv_len"]
    per_seq = math.ceil(kv_len / page)
    idx = torch.full((rows, spec["topk"]), -1, dtype=torch.int32)
    for r in range(rows):
        blk = torch.randint(0, blocks, (per_seq,), generator=g)
        pos = torch.randperm(kv_len, generator=g)[: min(kv_len, spec["topk"])]
        slots = (blk[pos // page] * owners + row) * page + pos % page
        if spec.get("holes"):
            slots[torch.rand(slots.shape, generator=g) < 0.2] = -1
        idx[r, : slots.numel()] = slots.int()
    q = torch.zeros(rows, spec["h_q"], 576)
    q[:, : spec["h_out"]] = torch.randn(rows, spec["h_out"], 576, generator=g)
    if spec.get("nope"):
        q[..., 512:] = 0
    out = torch.full((rows, spec["h_out"], 512), float("nan"), dtype=torch.bfloat16)
    return q.to(torch.bfloat16).cuda(), kv.cuda(), idx.cuda(), out.cuda()


def _gate(out, ref):
    o, r = out.float(), ref.float()
    per = (o - r).norm(dim=-1) / r.norm(dim=-1).clamp_min(1e-30)
    whole = float((o - r).norm() / r.norm())
    assert float(per.max()) < 1e-2 and whole < 4e-3, (float(per.max()), whole)


# --- the "1M tp4" cells: pools built on the device, only the requests' pages written

#: Every pool byte the cell's requests do not hold: e4m3, fp32 and bf16 NaN, so a key read
#: from anywhere else turns its row's output NaN.
_POOL_FILL = 0xFF
#: Rows of a step's scratch outside the call's tail view: bf16 NaN bits the call must leave.
_OUT_FILL = -1
_GATED_ROWS = 1024
#: The step scratch a `q_offset` cell's views tail: max_num_tokens rows.
_STEP_ROWS = 32768
_ROWS_PER_DRAW = 1 << 20


def _dev_rows(n: int, g: torch.Generator, nope: bool = False) -> torch.Tensor:
    """`ds_mla_rows` drawn on g's device."""
    dev = g.device
    codes = ((torch.rand(n, 512, generator=g, device=dev) * 2 - 1) * 448).to(torch.float8_e4m3fn)
    scale = torch.exp2(torch.randint(-13, 1, (n, 4), generator=g, device=dev).float())
    if nope:
        rope = torch.zeros(n, 64, device=dev)
    else:
        rope = torch.randn(n, 64, generator=g, device=dev) * 3
    return torch.cat(
        [
            codes.view(torch.uint8),
            scale.view(torch.uint8),
            rope.to(torch.bfloat16).view(torch.uint8),
        ],
        1,
    )


def _write_pages(pool, pages, row, g, nope) -> None:
    """Draw the owner `row` of every page in `pages` (block ids), a bounded slab at a time."""
    tpb = pool.shape[2]
    step = _ROWS_PER_DRAW // tpb
    for lo in range(0, pages.numel(), step):
        part = pages[lo : lo + step]
        pool[part, row] = _dev_rows(part.numel() * tpb, g, nope).view(-1, tpb, 656)


def _kv_lens(kv_len: int, n: int) -> list[int]:
    """`n` sequence lengths, the first `kv_len` and the rest spread below it: a row one page
    short of it, eighth, sixteenth and 32nd fractions, the rows around topk 2048 (2048 valid
    of 2049, all 2048, a -1 tail), and short rows down to 1. At 1048576 the sixteen hold
    2348481 keys, 73398 pages: the delivered 1M boot's 74183-block pool holds them apart."""
    spread = (kv_len, kv_len - 31, kv_len // 8 + 1, kv_len // 16, kv_len // 32 + 1, 9217, 4097,
              2049, 2048, 2047, 1500, 700, 257, 64, 2, 1)  # fmt: skip
    return [max(1, min(kv_len, n_kv)) for n_kv in spread[:n]]


def _requests_tp4(spec, n: int) -> list[tuple[int, int]]:
    """(cached, new) per request: a context cell's chunks, else `n` decode rows, each one
    query at its sequence's last position."""
    if spec.get("context"):
        if "contexts" in spec:
            return [tuple(c) for c in spec["contexts"]]
        return [(spec["cached"], spec["chunk_rows"])]
    return [(n_kv - 1, 1) for n_kv in _kv_lens(spec["kv_len"], n)]


def _pool_1m(spec, reqs, g):
    """The pool, its owner row, each request's pages (block ids, requests in order) and page
    counts, and the slot its first request's last position reads.

    `far`: the pool is `pages` / `owners` blocks exactly, every byte `_POOL_FILL` but the
    owner row of the requests' pages, which are distinct and the pool's top ones, the first
    request's last page on the top block. Otherwise the pages are drawn over the whole pool
    with repeats (its every block's row written), as the shared cells draw them.
    """
    dev = g.device
    owners, tpb = spec["owners"], spec["page_size"]
    blocks = -(-spec["pages"] // owners)
    row = 35 if spec.get("nope") else (owners - 1 if spec.get("far") else 1)
    counts = [-(-(c + n) // tpb) for c, n in reqs]
    need = sum(counts)
    pool = torch.empty((blocks, owners, tpb, 656), dtype=torch.uint8, device=dev)
    pool.fill_(_POOL_FILL)
    if spec.get("far"):
        assert need <= blocks, (need, blocks)
        pages = blocks - need + torch.randperm(need, generator=g, device=dev)
        top = int((pages == blocks - 1).nonzero()[0])
        pages[[top, counts[0] - 1]] = pages[[counts[0] - 1, top]]
        _write_pages(pool, pages, row, g, spec.get("nope", False))
    else:
        pages = torch.randint(0, blocks, (need,), generator=g, device=dev)
        _write_pages(pool, torch.arange(blocks, device=dev), row, g, spec.get("nope", False))
    last = sum(reqs[0]) - 1
    first_last = (int(pages[counts[0] - 1]) * owners + row) * tpb + last % tpb
    return pool, row, pages, counts, first_last


def _selections_tp4(spec, reqs, pages, counts, row, g) -> torch.Tensor:
    """int32 `[rows, topk]` slots, the rows the requests' new tokens in order. A row at
    position p sees keys 0..p: below topk all of them in order then -1, else topk draws over
    [0, p] (repeats counted by both sides) with p itself first. Each key's slot through its
    request's pages."""
    dev = g.device
    topk, tpb, owners = spec["topk"], spec["page_size"], spec["owners"]
    pos = torch.cat([torch.arange(c, c + n, device=dev) for c, n in reqs])
    req = torch.cat([torch.full((n,), r, device=dev) for r, (_, n) in enumerate(reqs)])
    first = torch.tensor([0] + counts[:-1], device=dev).cumsum(0)
    col = torch.arange(topk, device=dev)[None]
    p = pos[:, None]
    draw = torch.minimum(
        (torch.rand(pos.numel(), topk, generator=g, device=dev) * (p + 1)).long(), p
    )
    draw[:, 0] = pos
    sel = torch.where(p < topk, torch.where(col <= p, col, -1), draw)
    del draw
    key = sel.clamp_min(0)
    slot = (pages[first[req][:, None] + key // tpb] * owners + row) * tpb + key % tpb
    slot = torch.where(sel >= 0, slot, -1)
    assert int(slot.max()) < 2**31
    return slot.to(torch.int32)


def _queries_tp4(n: int, spec, g, scratch_rows: int = 0) -> torch.Tensor:
    """bf16 `[n, h_q, 576]`, the first h_out heads drawn and the rest zero (rope zero on a
    NoPE layer); with `scratch_rows`, the tail view of a zeroed scratch that many rows long."""
    dev = g.device
    q = torch.zeros(max(n, scratch_rows), spec["h_q"], 576, dtype=torch.bfloat16, device=dev)
    q = q[q.shape[0] - n :]
    q[:, : spec["h_out"]] = torch.randn(n, spec["h_out"], 576, generator=g, device=dev)
    if spec.get("nope"):
        q[..., 512:] = 0
    return q


def _calls_tp4(spec, q, kv, idx, out, metas) -> torch.Tensor:
    """Every layer's call, in sub-batches of `sub_batch` rows (one call without it), one
    sched meta per sub-batch size shared by every layer."""
    n = q.shape[0]
    sub = spec.get("sub_batch", n)
    for _ in range(spec["layers"]):
        for lo in range(0, n, sub):
            hi = min(lo + sub, n)
            meta = metas.setdefault(hi - lo, get_mla_metadata()[0])
            op(q[lo:hi], kv, idx[lo:hi], meta, SCALE, out[lo:hi])
    return out


def _check_tp4(q, kv, idx, out, g) -> None:
    """Every row finite; the rows past `_GATED_ROWS` on a sample against the reference."""
    n = q.shape[0]
    nan = out.isnan().flatten(1).any(1)
    assert not bool(nan.any()), f"{int(nan.sum())} of {n} rows read an unheld pool byte"
    rows = torch.arange(n, device=q.device)
    if n > _GATED_ROWS:
        picked = torch.randperm(n - 128, generator=g, device=q.device)[: _GATED_ROWS - 128]
        rows = torch.cat([rows[-128:], picked])
    _gate(out[rows], op.reference(q[rows], kv, idx[rows], None, SCALE, out[rows]))


def _replay_bitwise(spec, q, kv, idx, out, reqs, pages, counts, row, g) -> None:
    """Capture the layers' calls as the step does (a fresh sched meta its first call fills),
    replay on a new query and new slots over the same pages, gate it, and match an eager
    call on the same data bitwise."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        op(q, kv, idx, get_mla_metadata()[0], SCALE, out)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    meta = get_mla_metadata()[0]
    with torch.cuda.graph(graph):
        for _ in range(spec["layers"]):
            op(q, kv, idx, meta, SCALE, out)
    q.copy_(_queries_tp4(q.shape[0], spec, g))
    idx.copy_(_selections_tp4(spec, reqs, pages, counts, row, g))
    graph.replay()
    torch.cuda.synchronize()
    _check_tp4(q, kv, idx, out, g)
    eager = op(q, kv, idx, get_mla_metadata()[0], SCALE, torch.empty_like(out))
    assert torch.equal(eager.view(torch.int16), out.view(torch.int16)), "replay != eager"
    del graph


def _run_1m_tp4(spec, seed) -> None:
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    batches = spec.get("graph_batches", (spec["rows"],))
    reqs = _requests_tp4(spec, max(batches))
    pool, row, pages, counts, first_last = _pool_1m(spec, reqs, g)
    kv = pool.view(-1, spec["page_size"], 1, 656)
    if spec.get("far"):
        top = (pool.shape[0] * spec["owners"] - 1) * spec["page_size"] + spec["page_size"] - 1
        print(f"pool {tuple(pool.shape)}: first request's last slot {first_last} "
              f"(byte {first_last * 656}), pool's last {top}")  # fmt: skip
        if not spec.get("nope"):
            assert first_last == top - (
                spec["page_size"] - 1 - (sum(reqs[0]) - 1) % spec["page_size"]
            )
    for b in batches:
        sub = reqs[: len(reqs) if spec.get("context") else b]
        idx = _selections_tp4(spec, sub, pages, counts, row, g)
        if spec.get("far"):
            assert int(idx.max()) == first_last, (int(idx.max()), first_last)
        n = idx.shape[0]
        q = _queries_tp4(n, spec, g, _STEP_ROWS if "q_offset" in spec else 0)
        out_buf = torch.empty(max(n, _STEP_ROWS if "q_offset" in spec else 0), spec["h_out"], 512,
                              dtype=torch.bfloat16, device=dev)  # fmt: skip
        out_buf.view(torch.int16).fill_(_OUT_FILL)
        out = out_buf[out_buf.shape[0] - n :]
        if "q_offset" in spec:
            assert out_buf.shape[0] - n == spec["q_offset"]
        metas = {}
        _calls_tp4(spec, q, kv, idx, out, metas)
        if spec.get("context"):
            assert max(metas) == spec["rows"], sorted(metas)
        _check_tp4(q, kv, idx, out, g)
        if "q_offset" in spec:
            head = out_buf[: spec["q_offset"]]
            assert bool((head.view(torch.int16) == _OUT_FILL).all()), "wrote outside the view"
        if spec.get("captured"):
            _replay_bitwise(spec, q, kv, idx, out, sub, pages, counts, row, g)


def _ragged_tail_tp4(spec, seed) -> None:
    """A chunk's last sub-batch at every row count r of `tail_rows` (else 1..`sub_batch`): rows
    `lead_rows` .. `lead_rows` + r - 1 of the chunk behind `cached`, q and out views from that row
    of the step's scratch, each r one call on a sched meta of its own. Row i is the same sequence
    at every r, so the reference runs once over the `sub_batch` rows and each r's output is gated
    against its first r rows in `_gate`'s band (a NaN row fails it); the scratch rows before the
    view stay unwritten. Largest r first: rising sizes leave the caching allocator a new block per
    call (a rising sweep reserved 254.75 GiB)."""
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    sub, lead = spec["sub_batch"], spec["lead_rows"]
    reqs = [(spec["cached"] + lead, sub)]
    pool, row, pages, counts, _ = _pool_1m(spec, reqs, g)
    kv = pool.view(-1, spec["page_size"], 1, 656)
    idx = _selections_tp4(spec, reqs, pages, counts, row, g)
    q = _queries_tp4(sub, spec, g, lead + sub)
    out_buf = torch.empty(lead + sub, spec["h_out"], 512, dtype=torch.bfloat16, device=dev)
    out_buf.view(torch.int16).fill_(_OUT_FILL)
    out = out_buf[lead:]
    ref = op.reference(q, kv, idx, None, SCALE, out).float()
    ref_norm = ref.norm(dim=-1)
    ref_sq = ref_norm.square().sum(1).cumsum(0)
    tails = spec.get("tail_rows", range(1, sub + 1))
    per = torch.empty(len(tails), device=dev)
    whole = torch.empty(len(tails), device=dev)
    for i in sorted(range(len(tails)), key=lambda i: -tails[i]):
        r = tails[i]
        o = op(q[:r], kv, idx[:r], get_mla_metadata()[0], SCALE, out[:r])
        err = (o.float() - ref[:r]).square().sum(-1)
        per[i] = (err.sqrt() / ref_norm[:r].clamp_min(1e-30)).max()
        whole[i] = (err.sum() / ref_sq[r - 1]).sqrt()
    bad = ((per < 1e-2) & (whole < 4e-3)).logical_not().nonzero()[:8, 0].tolist()
    assert not bad, [(tails[i], float(per[i]), float(whole[i])) for i in bad]
    assert bool((out_buf[:lead].view(torch.int16) == _OUT_FILL).all()), "wrote outside the view"
    i, j = int(per.argmax()), int(whole.argmax())
    print(f"ragged tail behind {reqs[0][0]}: {len(tails)} row counts {tails[0]}..{tails[-1]}, "
          f"per-(row, head) rel L2 max {float(per[i]):.3e} at r {tails[i]}, whole call max "
          f"{float(whole[j]):.3e} at r {tails[j]}")  # fmt: skip


# --- the "1M tep4", "1M dep4" and "1M tep8" cells: pools on the device, one owner row of the
# requests' pages written

#: Pool rows drawn per batch while writing the requests' pages.
_DRAW_ROWS = 1 << 18


def _draw_rows(n: int, g: torch.Generator, nope: bool) -> torch.Tensor:
    """`ds_mla_rows` drawn on g's device."""
    dev = g.device
    codes = ((torch.rand(n, 512, generator=g, device=dev) * 2 - 1) * 448).to(torch.float8_e4m3fn)
    scale = torch.exp2(torch.randint(-13, 1, (n, 4), generator=g, device=dev).float())
    if nope:
        rope = torch.zeros(n, 64, device=dev)
    else:
        rope = torch.randn(n, 64, generator=g, device=dev) * 3
    rope = rope.to(torch.bfloat16)
    return torch.cat([codes.view(torch.uint8), scale.view(torch.uint8), rope.view(torch.uint8)], 1)


def _decode_lens(kv_len: int, n: int) -> list[int]:
    """`n` decode sequence lengths: `kv_len`, a row 31 keys short of it, its 1/8, 1/16 and
    1/32 plus one, rows across the sub-chunk sizes, around topk 2048 (2049: 2048 of 2049
    valid; 2048: all; 2047: a -1 tail) and short rows down to one key. At 1048576 the sixteen
    hold 2364359 keys on 73898 pages: the 74183- and the 147456-block pools hold them on
    distinct pages."""
    spread = (kv_len, kv_len - 31, kv_len // 8 + 1, kv_len // 16 + 1, kv_len // 32 + 1, 16385,
              9217, 4097, 2049, 2048, 2047, 1025, 700, 257, 33, 1)  # fmt: skip
    return [max(1, min(kv_len, n_kv)) for n_kv in spread[:n]]


def _requests(spec, n: int) -> list[tuple[int, int]]:
    """(cached, new) per request. A context cell: its chunks. A `boot_capture` cell: the live
    row (kv_len keys), the warm-up's longest dummy (kv_len - 1 keys), n - 1 one-key fillers,
    the padding dummy and an idle rank's dummy (one key each). Otherwise `n` decode rows. A
    decode row is one query at its sequence's last position."""
    if spec.get("context"):
        if "contexts" in spec:
            return [tuple(c) for c in spec["contexts"]]
        return [(spec["cached"], spec["chunk_rows"])]
    if spec.get("boot_capture"):
        kv_len = spec["kv_len"]
        return [(kv_len - 1, 1), (kv_len - 2, 1), *[(0, 1)] * (n - 1), (0, 1), (0, 1)]
    return [(n_kv - 1, 1) for n_kv in _decode_lens(spec["kv_len"], n)]


def _pool(spec, reqs, g):
    """The pool [blocks, owners, page, 656], its written owner row, the requests' pages (block
    ids, request after request), each request's page count, and the slot of the first
    request's last key.

    `far`: the pool is `pages` / `owners` blocks, every byte `_POOL_FILL` but the owner row of
    the requests' pages, which are distinct and the pool's top ones, the first request's last
    page on the top block. `top_blocks`: the pages are drawn with repeats from the pool's top
    `top_blocks` blocks, whose owner row is written. Otherwise they are drawn over the whole
    pool and every block's owner row is written.
    """
    dev = g.device
    owners, page = spec["owners"], spec["page_size"]
    row = 35 if spec.get("nope") else owners - 1
    counts = [-(-(c + n) // page) for c, n in reqs]
    need = sum(counts)
    blocks = -(-spec["pages"] // owners)
    pool = torch.full((blocks, owners, page, 656), _POOL_FILL, dtype=torch.uint8, device=dev)
    if spec.get("far"):
        assert spec["pages"] % owners == 0 and need <= blocks, (need, blocks)
        pages = blocks - need + torch.randperm(need, generator=g, device=dev)
        top, last = int((pages == blocks - 1).nonzero()[0, 0]), counts[0] - 1
        pages[[top, last]] = pages[[last, top]]
        written = pages
    elif "top_blocks" in spec:
        low = blocks - spec["top_blocks"]
        pages = torch.randint(low, blocks, (need,), generator=g, device=dev)
        written = torch.arange(low, blocks, device=dev)
    else:
        pages = torch.randint(0, blocks, (need,), generator=g, device=dev)
        written = torch.arange(blocks, device=dev)
    step = _DRAW_ROWS // page
    for lo in range(0, written.numel(), step):
        part = written[lo : lo + step]
        rows = _draw_rows(part.numel() * page, g, spec.get("nope", False))
        pool[part, row] = rows.view(-1, page, 656)
    first_last = (int(pages[counts[0] - 1]) * owners + row) * page + (sum(reqs[0]) - 1) % page
    return pool, row, pages, counts, first_last


def _selections(spec, reqs, pages, counts, row, g, which=None) -> torch.Tensor:
    """int32 `[rows, topk]` slots, the rows the new tokens of the requests `which` names (all,
    in order, by default). A row at position p sees keys 0..p: below topk keys all of them,
    then -1; else topk distinct keys, one drawn in each of topk equal strata of [0, p], p
    itself the last. Each row's valid keys in a random order, -1 last (as the indexer's top-k
    hands them over), each key's slot through its request's pages."""
    dev = g.device
    which = range(len(reqs)) if which is None else which
    topk, page, owners = spec["topk"], spec["page_size"], spec["owners"]
    pos = torch.cat([torch.arange(reqs[r][0], sum(reqs[r]), device=dev) for r in which])
    req = torch.cat([torch.full((reqs[r][1],), r, device=dev) for r in which])
    first = torch.tensor([0, *counts[:-1]], device=dev).cumsum(0)
    n_keys = pos[:, None] + 1
    col = torch.arange(topk + 1, device=dev)[None]
    edge = col * n_keys // topk
    width = edge[:, 1:] - edge[:, :-1]
    draw = edge[:, :-1] + (torch.rand(pos.numel(), topk, generator=g, device=dev) * width).long()
    draw[:, -1] = pos
    del edge, width
    short = torch.where(col[:, :-1] < n_keys, col[:, :-1], -1)
    sel = torch.where(n_keys > topk, draw, short)
    del draw, short
    order = (torch.rand(sel.shape, generator=g, device=dev) + (sel < 0)).argsort(1)
    sel = sel.gather(1, order)
    del order
    key = sel.clamp_min(0)
    slot = (pages[first[req][:, None] + key // page] * owners + row) * page + key % page
    slot = torch.where(sel >= 0, slot, -1)
    assert int(slot.max()) < 2**31
    return slot.to(torch.int32)


def _queries(n: int, spec, g, scratch: int = 0) -> torch.Tensor:
    """bf16 `[n, h_q, 576]`, the first h_out heads drawn and the rest zero (rope zero on a
    NoPE layer); with `scratch`, the tail view of a scratch that many rows long whose rows
    ahead of the view carry draws of their own."""
    dev = g.device
    total = max(n, scratch)
    q = torch.zeros(total, spec["h_q"], 576, dtype=torch.bfloat16, device=dev)
    q[:, : spec["h_out"]] = torch.randn(total, spec["h_out"], 576, generator=g, device=dev)
    if spec.get("nope"):
        q[..., 512:] = 0
    return q[total - n :]


def _calls(spec, q, kv, idx, out, metas) -> None:
    """Every layer's call, in sub-batches of `sub_batch` rows (one call without it), one
    sched meta per sub-batch size shared by every layer."""
    n = q.shape[0]
    sub = spec.get("sub_batch", n)
    for _ in range(spec["layers"]):
        for lo in range(0, n, sub):
            hi = min(lo + sub, n)
            meta = metas.setdefault(hi - lo, get_mla_metadata()[0])
            op(q[lo:hi], kv, idx[lo:hi], meta, SCALE, out[lo:hi])


def _check(q, kv, idx, out, g) -> None:
    """Every row finite; up to `_GATED_ROWS` rows against the entry's reference."""
    n = q.shape[0]
    bad = ~torch.isfinite(out).flatten(1).all(1)
    assert not bool(bad.any()), f"{int(bad.sum())} of {n} rows read an unheld pool byte"
    rows = torch.arange(n, device=q.device)
    if n > _GATED_ROWS:
        drawn = torch.randperm(n - 128, generator=g, device=q.device)[: _GATED_ROWS - 128]
        rows = torch.cat([rows[-128:], drawn])
    _gate(out[rows], op.reference(q[rows], kv, idx[rows], None, SCALE, out[rows]))


def _split_rows(meta) -> torch.Tensor:
    """The rows FlashMLA's decode scheduler split over their keys on the call that filled
    `meta`: more than one part in its `num_splits` prefix sums, the parts merged in fp32 by
    the combine kernel."""
    starts = meta.num_splits
    return (starts[1:] - starts[:-1]) > 1


def _check_split(q, kv, idx, out, metas, sub) -> None:
    """Gate the rows each sub-batch's scheduler split against the reference; print how many
    per sub-batch size."""
    n = q.shape[0]
    rows = torch.cat([lo + _split_rows(metas[min(lo + sub, n) - lo]).nonzero()[:, 0]
                      for lo in range(0, n, sub)])  # fmt: skip
    if rows.numel():
        _gate(out[rows], op.reference(q[rows], kv, idx[rows], None, SCALE, out[rows]))
    split = {size: int(_split_rows(m).sum()) for size, m in sorted(metas.items())}
    print(f"context sub-batches: split rows per sub-batch size {split}")


def _run_ragged_tail(spec, seed, largest_first: bool = False) -> None:
    """The last sub-batch of a `_STEP_ROWS`-row chunk behind `cached` keys at each r in `tails`
    (every r in 1..`sub_batch` without it; `largest_first`, as the "1M tep8" cells' run walked
    it, the largest r first): q, slots and out the views [lo, lo + r) of step buffers, lo =
    `_STEP_ROWS` - `sub_batch`, as `_attn_ctx` slices them; each r one call on a sched meta of
    its own into rows reset to NaN. Every row finite; a row neither this call's scheduler nor
    the whole sub-batch's split is bitwise the whole sub-batch's; every row in the band
    against the entry's reference, run once over the sub-batch's rows; rows ahead of the
    views untouched. Prints how many rows the scheduler split per r."""
    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    g = torch.Generator(device=dev).manual_seed(seed)
    sub, cached = spec["sub_batch"], spec["cached"]
    lo = _STEP_ROWS - sub
    pool, row, pages, counts, _ = _pool(spec, [(cached, _STEP_ROWS)], g)
    kv = pool.view(-1, spec["page_size"], 1, 656)
    idx_buf = torch.full((_STEP_ROWS, spec["topk"]), -1, dtype=torch.int32, device=dev)
    idx_buf[lo:] = _selections(spec, [(cached + lo, sub)], pages, counts, row, g)
    idx = idx_buf[lo:]
    lowest = (pool.shape[0] - spec["top_blocks"]) * spec["owners"] * spec["page_size"]
    assert lowest * 656 >= 2**31 and int(idx.min()) >= lowest, lowest
    q = _queries(sub, spec, g, _STEP_ROWS)
    out_buf = torch.empty(_STEP_ROWS, spec["h_out"], 512, dtype=torch.bfloat16, device=dev)
    out_buf.view(torch.int16).fill_(_OUT_FILL)
    out = out_buf[lo:]
    assert q.storage_offset() == lo * q.stride(0) and out.storage_offset() == lo * out.stride(0)
    ref = op.reference(q, kv, idx, None, SCALE, out)
    meta = get_mla_metadata()[0]
    whole = op(q, kv, idx, meta, SCALE, torch.empty_like(out))
    whole_split = _split_rows(meta)
    assert bool(torch.isfinite(whole).all())
    _gate(whole, ref)
    split_rows, differ_rows, bitwise_rows = {}, 0, 0
    tails = spec.get("tails", range(1, sub + 1))
    # Largest first: each call's FlashMLA buffers then fit in blocks the caching allocator
    # already holds.
    for r in sorted(tails, reverse=True) if largest_first else tails:
        o = out[:r]
        o.view(torch.int16).fill_(_OUT_FILL)
        meta = get_mla_metadata()[0]
        op(q[:r], kv, idx[:r], meta, SCALE, o)
        split = _split_rows(meta)
        bad = ~torch.isfinite(o).flatten(1).all(1)
        assert not bool(bad.any()), f"r {r}: {int(bad.sum())} rows read an unheld pool byte"
        differ = (o.view(torch.int16) != whole[:r].view(torch.int16)).flatten(1).any(1)
        unsplit = ~split & ~whole_split[:r]
        stray = differ & unsplit
        assert not bool(stray.any()), f"r {r}: {int(stray.sum())} unsplit rows differ"
        try:
            _gate(o, ref[:r])
        except AssertionError as exc:
            raise AssertionError(f"r {r}: {exc}") from exc
        split_rows[r] = int(split.sum())
        differ_rows += int(differ.sum())
        bitwise_rows += int(unsplit.sum())
    ahead = out_buf[:lo].view(torch.int16)
    assert bool((ahead == _OUT_FILL).all()), "a call wrote ahead of its view"
    # Both per-row checks saw rows: some calls split rows, some kept rows in one part, and
    # some rows neither scheduler split were held bitwise.
    assert 0 < sum(split_rows.values()) < sum(split_rows), "no split row, or no unsplit one"
    assert bitwise_rows > 0, "no row neither scheduler split"
    hist = dict(sorted(collections.Counter(split_rows.values()).items()))
    named = {r: split_rows[r] for r in (1, 16, 96, 127, 128, 8175, 8192) if r in split_rows}
    print(f"ragged tail behind {cached}: {len(split_rows)} calls, r in {min(split_rows)}.."
          f"{max(split_rows)}; the whole sub-batch split {int(whole_split.sum())} rows; "
          f"{sum(split_rows.values())} split rows over the calls, {differ_rows} not bitwise the "
          f"whole sub-batch's, {bitwise_rows} neither split held bitwise; split rows per call "
          f"(count: calls) {hist}; at r {named}; "
          f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GiB reserved at most")  # fmt: skip


def _ragged_tail(spec, seed) -> None:
    """A chunk's last sub-batch at each r of `tail_rows` (every r in 1..`sub_batch` without
    it), as `_attn_ctx` calls it: rows `lead_rows` + [0, r) of the chunk behind `cached` keys,
    q, slots and out views from that row of the step's buffers, each r one call on a sched
    meta of its own into rows reset to NaN, largest r first (rising sizes leave the caching
    allocator a new block per call). Row i is the same sequence at every r, so the reference
    and one whole-sub-batch call run once. Per call: the rows the scheduler split, and any
    row not bitwise the whole call's, gated against the reference; a row neither call split
    bitwise the whole call's (itself gated); the call's rel L2 over every row. The step's
    rows ahead of the views stay unwritten. Prints the rows split per call."""
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    sub, lead, cached = spec["sub_batch"], spec["lead_rows"], spec["cached"]
    pool, row, pages, counts, _ = _pool(spec, [(cached, lead + sub)], g)
    kv = pool.view(-1, spec["page_size"], 1, 656)
    idx_buf = torch.full((lead + sub, spec["topk"]), -1, dtype=torch.int32, device=dev)
    idx_buf[lead:] = _selections(spec, [(cached + lead, sub)], pages, counts, row, g)
    idx = idx_buf[lead:]
    lowest = (pool.shape[0] - spec["top_blocks"]) * spec["owners"] * spec["page_size"]
    assert lowest * 656 >= 2**31 and int(idx.min()) >= lowest, lowest
    q = _queries(sub, spec, g, lead + sub)
    out_buf = torch.empty(lead + sub, spec["h_out"], 512, dtype=torch.bfloat16, device=dev)
    out_buf.view(torch.int16).fill_(_OUT_FILL)
    out = out_buf[lead:]
    assert q.storage_offset() == lead * q.stride(0) and out.storage_offset() == lead * out.stride(0)
    ref = op.reference(q, kv, idx, None, SCALE, out).float()
    meta = get_mla_metadata()[0]
    whole = op(q, kv, idx, meta, SCALE, torch.empty_like(out))
    whole_split = _split_rows(meta)
    _gate(whole, ref)
    ref_norm = ref.norm(dim=-1)
    ref_sq = ref_norm.square().sum(1).cumsum(0)
    whole_err = (whole.float() - ref).square().sum(-1)
    whole_sq = whole_err.sum(1).cumsum(0)
    tails = spec.get("tail_rows", range(1, sub + 1))
    per, rel, stray, n_split, n_differ = (torch.zeros(len(tails), device=dev) for _ in range(5))
    for i in sorted(range(len(tails)), key=lambda i: -tails[i]):
        r = tails[i]
        o = out[:r]
        o.view(torch.int16).fill_(_OUT_FILL)
        meta = get_mla_metadata()[0]
        op(q[:r], kv, idx[:r], meta, SCALE, o)
        split = _split_rows(meta)
        differ = torch.ne(o.view(torch.int64), whole[:r].view(torch.int64)).flatten(1).any(1)
        gated = (split | differ).nonzero()[:, 0]
        err = (o[gated].float() - ref[gated]).square().sum(-1)
        per[i] = torch.cat([(err.sqrt() / ref_norm[gated]).flatten(), per[i : i + 1]]).max()
        err_sq = whole_sq[r - 1] - whole_err[gated].sum() + err.sum()
        rel[i] = (err_sq.clamp_min(0) / ref_sq[r - 1]).sqrt()
        stray[i] = (differ & ~split & ~whole_split[:r]).sum()
        n_split[i], n_differ[i] = split.sum(), differ.sum()
    bad = ((per < 1e-2) & (rel < 4e-3) & (stray == 0)).logical_not().nonzero()[:8, 0].tolist()
    assert not bad, [(tails[i], float(per[i]), float(rel[i]), int(stray[i])) for i in bad]
    assert bool((out_buf[:lead].view(torch.int16) == _OUT_FILL).all()), "wrote ahead of the view"
    splits = n_split.long().tolist()
    # Both kinds of row were seen: some calls split rows, some kept rows in one part.
    assert 0 < sum(splits) < sum(tails), "no split row, or no unsplit one"
    hist = dict(sorted(collections.Counter(splits).items()))
    shown = (1, 127, 128, 6445, 8096, 8175, 8192)
    named = {r: n for r, n in zip(tails, splits) if r in shown}
    print(f"ragged tail behind {cached + lead}: {len(tails)} calls, r {min(tails)}..{max(tails)}; "
          f"the whole sub-batch split {int(whole_split.sum())} rows; {sum(splits)} split rows over "
          f"the calls, {int(n_differ.sum())} not bitwise the whole call's; per-(row, head) rel L2 "
          f"max {float(per.max()):.3e}, call max {float(rel.max()):.3e}; split rows per call "
          f"(count: calls) {hist}; at r {named}")  # fmt: skip


def _capture(spec, q, kv, idx, out) -> torch.cuda.CUDAGraph:
    """The decode step's graph over the static q / idx / out: a warm-up call on a side stream,
    then every layer's call on one fresh sched meta, which the graph's first call fills."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        op(q, kv, idx, get_mla_metadata()[0], SCALE, out)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    meta = get_mla_metadata()[0]
    with torch.cuda.graph(graph):
        for _ in range(spec["layers"]):
            op(q, kv, idx, meta, SCALE, out)
    return graph


def _replay(graph, q, kv, idx, out, q_new, idx_new, g) -> None:
    """Replay on new queries and slots into a NaN-filled `out`, check the replay, and match an
    eager call on the same data bitwise."""
    q.copy_(q_new)
    idx.copy_(idx_new)
    out.view(torch.int16).fill_(_OUT_FILL)
    graph.replay()
    torch.cuda.synchronize()
    _check(q, kv, idx, out, g)
    eager = op(q, kv, idx, get_mla_metadata()[0], SCALE, torch.empty_like(out))
    assert torch.equal(eager.view(torch.int16), out.view(torch.int16)), "replay != eager"


def _run_boot_capture(spec, g, kv, reqs, pages, counts, row, first_last, lowest) -> None:
    """Each batch b of `graph_batches` as the dep4 engine runs it: warm-up calls and the capture
    on the boot rows (the longest dummy, b - 1 fillers), then two replays as attention DP
    pads them: the live row behind b - 1 copies of the padding dummy's row (one slot, 2047
    x -1, one query), and an idle rank's dummy behind the same padding."""
    pad, idle = len(reqs) - 2, len(reqs) - 1

    def rows(first: int, b: int):
        idx = _selections(spec, reqs, pages, counts, row, g, which=[first])
        q = _queries(1, spec, g)
        if b > 1:
            padding = _selections(spec, reqs, pages, counts, row, g, which=[pad])
            idx = torch.cat([idx, padding.expand(b - 1, -1)])
            q = torch.cat([q, _queries(1, spec, g).expand(b - 1, -1, -1)])
        return q, idx

    for b in spec["graph_batches"]:
        boot = [1, *range(2, b + 1)]
        idx = _selections(spec, reqs, pages, counts, row, g, which=boot)
        assert idx.shape[0] == b and int(idx[idx >= 0].min()) >= lowest
        assert int((idx[0] >= 0).sum()) == spec["topk"] and bool((idx[1:, 1:] == -1).all())
        q = _queries(b, spec, g)
        out = torch.empty(b, spec["h_out"], 512, dtype=torch.bfloat16, device=q.device)
        out.view(torch.int16).fill_(_OUT_FILL)
        _calls(spec, q, kv, idx, out, {})
        _check(q, kv, idx, out, g)
        graph = _capture(spec, q, kv, idx, out)
        q_live, idx_live = rows(0, b)
        if spec.get("far"):
            assert int(idx_live.max()) == first_last, (int(idx_live.max()), first_last)
        assert int((idx_live[1:] >= 0).sum()) == b - 1
        assert bool((idx_live[1:] == idx_live[-1]).all() and (q_live[1:] == q_live[-1]).all())
        _replay(graph, q, kv, idx, out, q_live, idx_live, g)
        _replay(graph, q, kv, idx, out, *rows(idle, b), g)
        del graph


def _run_1m(spec, seed) -> None:
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    batches = spec.get("graph_batches", spec.get("row_counts", (spec["rows"],)))
    reqs = _requests(spec, max(batches))
    pool, row, pages, counts, first_last = _pool(spec, reqs, g)
    page = spec["page_size"]
    kv = pool.view(-1, page, 1, 656)
    top = pool.shape[0] * spec["owners"] * page - 1
    if spec.get("far"):
        print(f"pool {tuple(pool.shape)}: first request's last slot {first_last} "
              f"(byte {first_last * 656}), pool's last slot {top}")  # fmt: skip
        if not spec.get("nope"):
            assert first_last == top - (page - 1 - (sum(reqs[0]) - 1) % page)
    lowest = 0
    if "top_blocks" in spec:
        lowest = (pool.shape[0] - spec["top_blocks"]) * spec["owners"] * page
        assert lowest * 656 >= 2**31, lowest
    if spec.get("boot_capture"):
        _run_boot_capture(spec, g, kv, reqs, pages, counts, row, first_last, lowest)
        return
    for b in batches:
        sub = reqs if spec.get("context") else reqs[:b]
        idx = _selections(spec, sub, pages, counts, row, g)
        assert int(idx.min()) >= -1 and int(idx[idx >= 0].min()) >= lowest
        if spec.get("far"):
            assert int(idx.max()) == first_last, (int(idx.max()), first_last)
        n = idx.shape[0]
        scratch = _STEP_ROWS if "q_offset" in spec else 0
        q = _queries(n, spec, g, scratch)
        out_buf = torch.empty(max(n, scratch), spec["h_out"], 512, dtype=torch.bfloat16, device=dev)
        out_buf.view(torch.int16).fill_(_OUT_FILL)
        out = out_buf[out_buf.shape[0] - n :]
        if "q_offset" in spec:
            assert out_buf.shape[0] - n == spec["q_offset"]
            assert q.storage_offset() == spec["q_offset"] * q.stride(0)
        metas = {}
        _calls(spec, q, kv, idx, out, metas)
        if spec.get("context"):
            assert max(metas) == spec["rows"] and n == sum(c[1] for c in reqs), sorted(metas)
        else:
            assert n == b
        _check(q, kv, idx, out, g)
        if spec.get("context"):
            _check_split(q, kv, idx, out, metas, spec["sub_batch"])
        if "q_offset" in spec:
            ahead = out_buf[: spec["q_offset"]].view(torch.int16)
            assert bool((ahead == _OUT_FILL).all()), "the call wrote ahead of its view"
        if spec.get("captured"):
            graph = _capture(spec, q, kv, idx, out)
            q_new, idx_new = _queries(n, spec, g), _selections(spec, sub, pages, counts, row, g)
            _replay(graph, q, kv, idx, out, q_new, idx_new, g)
            del graph


def _view_cell(spec, seed) -> None:
    """`out` None: the returned `[rows, h_q, 512]` kernel output is gated against the
    reference over every head, and its first `h_out` heads are bitwise what a call with `out`
    copies out on the same inputs (the kernel is deterministic; measured)."""
    q, kv, idx, out = _build(spec, seed)
    meta = get_mla_metadata()[0]
    for _ in range(spec["layers"]):
        res = op(q, kv, idx, meta, SCALE, None)
    assert res.shape == (spec["rows"], spec["h_q"], 512), res.shape
    _gate(res, op.reference(q, kv, idx, meta, SCALE, None))
    op(q, kv, idx, meta, SCALE, out)
    assert torch.equal(res[:, : spec["h_out"]].view(torch.int16), out.view(torch.int16))
    if spec.get("captured"):
        meta = get_mla_metadata()[0]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            op(q, kv, idx, get_mla_metadata()[0], SCALE, None)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(spec["layers"]):
                res = op(q, kv, idx, meta, SCALE, None)
        q2, kv2, idx2, _ = _build(spec, seed + 1)
        q.copy_(q2), kv.copy_(kv2), idx.copy_(idx2)
        graph.replay()
        torch.cuda.synchronize()
        _gate(res, op.reference(q, kv, idx, None, SCALE, None))


def _run_1m_view(spec, seed) -> None:
    """A "1M view" cell: decode rows over the "1M tep4" / "1M dep4" harness's pool, every
    layer's call with `out` None on one sched meta (captured and replayed on new queries and
    slots when `captured`). The returned kernel output is `[rows, h_q, 512]`; its first `h_out`
    heads are every row finite and gated to the reference (`_check`), and bitwise what a call
    with `out` copies out on the same inputs."""
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    batches = spec.get("graph_batches", (spec["rows"],))
    reqs = _requests(spec, max(batches))
    pool, row, pages, counts, first_last = _pool(spec, reqs, g)
    kv = pool.view(-1, spec["page_size"], 1, 656)
    h_out = spec["h_out"]

    def copied(q, idx) -> torch.Tensor:
        out = torch.empty(q.shape[0], h_out, 512, dtype=torch.bfloat16, device=dev)
        return op(q, kv, idx, get_mla_metadata()[0], SCALE, out).view(torch.int16)

    for b in batches:
        idx = _selections(spec, reqs[:b], pages, counts, row, g)
        if spec.get("far"):
            assert int(idx.max()) == first_last, (int(idx.max()), first_last)
        q = _queries(b, spec, g)
        meta = get_mla_metadata()[0]
        for _ in range(spec["layers"]):
            res = op(q, kv, idx, meta, SCALE, None)
        assert res.shape == (b, spec["h_q"], 512), res.shape
        _check(q, kv, idx, res[:, :h_out], g)
        assert torch.equal(res[:, :h_out].contiguous().view(torch.int16), copied(q, idx))
        if not spec.get("captured"):
            continue
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            op(q, kv, idx, get_mla_metadata()[0], SCALE, None)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        meta = get_mla_metadata()[0]
        with torch.cuda.graph(graph):
            for _ in range(spec["layers"]):
                res = op(q, kv, idx, meta, SCALE, None)
        q.copy_(_queries(b, spec, g))
        idx.copy_(_selections(spec, reqs[:b], pages, counts, row, g))
        res.view(torch.int16).fill_(_OUT_FILL)
        graph.replay()
        torch.cuda.synchronize()
        _check(q, kv, idx, res[:, :h_out], g)
        live = res[:, :h_out].contiguous().view(torch.int16)
        assert torch.equal(live, copied(q, idx)), "the replay differs from an eager out= call"
        del graph


def _is_1m(spec) -> bool:
    keys = ("graph_batches", "row_counts", "far", "top_blocks", "context", "q_offset")
    return any(k in spec for k in keys)


_CELLS = [
    pytest.param(c, id=c.why[:60], marks=[pytest.mark.large] if c.spec.get("large") else [])
    for c in op.CELLS
]


@pytest.mark.parametrize("cell", _CELLS)
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode())
    if spec.get("view"):
        try:
            (_run_1m_view if _is_1m(spec) else _view_cell)(spec, seed)
        finally:
            torch.cuda.empty_cache()
        return
    if _is_1m(spec):
        try:
            if cell.why.startswith("1M tp4 "):
                ragged = spec.get("ragged_tail") or "tail_rows" in spec
                (_ragged_tail_tp4 if ragged else _run_1m_tp4)(spec, seed)
            elif cell.why.startswith(("1M tep4 ", "1M tep8 ")) and spec.get("ragged_tail"):
                _run_ragged_tail(spec, seed, largest_first=cell.why.startswith("1M tep8 "))
            elif spec.get("ragged_tail") or "tail_rows" in spec:
                _ragged_tail(spec, seed)
            else:
                _run_1m(spec, seed)
        finally:
            torch.cuda.empty_cache()
        return
    q, kv, idx, out = _build(spec, seed)
    meta = get_mla_metadata()[0]
    for _ in range(spec["layers"]):
        op(q, kv, idx, meta, SCALE, out)
    _gate(out, op.reference(q, kv, idx, meta, SCALE, out))
    if spec.get("captured"):
        meta = get_mla_metadata()[0]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            op(q, kv, idx, get_mla_metadata()[0], SCALE, out)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(spec["layers"]):
                op(q, kv, idx, meta, SCALE, out)
        q2, kv2, idx2, _ = _build(spec, seed + 1)
        q.copy_(q2), kv.copy_(kv2), idx.copy_(idx2)
        graph.replay()
        torch.cuda.synchronize()
        _gate(out, op.reference(q, kv, idx, None, SCALE, out))


def test_is_valid_refuses_sixteen_kernel_heads() -> None:
    q, kv, idx, out = _build(dict(rows=2, h_q=64, h_out=16, topk=2048, kv_len=3000, page_size=32,
                                  owners=4, pages=400), 1)  # fmt: skip
    with pytest.raises(ValueError, match="h_q"):
        op.is_valid(q[:, :16], kv, idx, None, SCALE, out)


def test_is_valid_refuses_more_than_8192_rows() -> None:
    """Past its shared-memory row limit FlashMLA's scheduler exits the process ("invalid
    argument", the note's Hard limits), so nothing here launches: shapes only, on expanded
    views of the tep8 call (8 heads out), the tp4 and tep4 call (16 heads out) and the dep4
    call (64 heads in and out). 8192 rows pass is_valid; 8193, 11623, 16384 and 32768 (a
    whole chunk in one call) are refused before the call reaches the kernel."""
    dev = torch.device("cuda")
    kv = torch.full((64, 32, 1, 656), _POOL_FILL, dtype=torch.uint8, device=dev)
    cases = ((8192, False), (8193, True), (11623, True), (16384, True), (32768, True))
    for h_out in (8, 16, 64):
        for n, refused in cases:
            q = torch.zeros(1, 64, 576, dtype=torch.bfloat16, device=dev).expand(n, 64, 576)
            idx = torch.full((1, 2048), -1, dtype=torch.int32, device=dev).expand(n, 2048)
            out = torch.empty(1, h_out, 512, dtype=torch.bfloat16, device=dev)
            out = out.expand(n, h_out, 512)
            if not refused:
                op.is_valid(q, kv, idx, None, SCALE, out)
                continue
            with validating(op), pytest.raises(ValueError, match=f"{n} rows in one call"):
                op(q, kv, idx, get_mla_metadata()[0], SCALE, out)


def test_1m_tp4_harness_sees_a_misread() -> None:
    """The "1M tp4" cells' harness against two wrong slot maps on a 1100-block pool: every key one
    page on (the next owner row, unwritten: NaN rows) and every key at the neighbouring
    offset of its page (written rows, other keys: outside the band). Both fail, the right
    map passes."""
    tail_view = next(c for c in op.CELLS if c.why.startswith("1M tp4 mixed step's tail view"))
    spec = {k: v for k, v in tail_view.spec.items() if k != "q_offset"}
    assert spec["pages"] == 1100 * 63 and not spec.get("far")
    g = torch.Generator(device="cuda").manual_seed(7)
    reqs = _requests_tp4(spec, 16)
    pool, row, pages, counts, _ = _pool_1m(spec, reqs, g)
    kv = pool.view(-1, spec["page_size"], 1, 656)
    idx = _selections_tp4(spec, reqs, pages, counts, row, g)
    q = _queries_tp4(idx.shape[0], spec, g)
    out = torch.empty(idx.shape[0], spec["h_out"], 512, dtype=torch.bfloat16, device="cuda")
    next_page = torch.where(idx >= 0, idx + spec["page_size"], idx)
    neighbour = torch.where(idx >= 0, idx ^ 1, idx)
    for wrong, seen in ((next_page, "unheld pool byte"), (neighbour, r"^\(")):
        op(q, kv, wrong, get_mla_metadata()[0], SCALE, out)
        with pytest.raises(AssertionError, match=seen):
            _check_tp4(q, kv, idx, out, g)
    op(q, kv, idx, get_mla_metadata()[0], SCALE, out)
    _check_tp4(q, kv, idx, out, g)


def test_1m_harness_sees_a_misread_on_a_tep4_pool() -> None:
    """The 1M harness against three wrong slot maps over the pre-merge context cell's pool,
    whose requests' keys all sit past 2**31 bytes: every key one owner row down (row 61 of
    its block, unwritten: NaN rows), every key's byte offset wrapped at 2**31 (what an
    offset kept in 31 bits reads: the unwritten low blocks), and every key at the neighbouring
    offset of its page (written rows, other keys: outside the band). Each is refused; the
    right map passes."""
    cell = next(c for c in op.CELLS if c.why.startswith("1M tep4 pre-merge context"))
    spec = dict(cell.spec, chunk_rows=1024, cached=1047552)
    g = torch.Generator(device="cuda").manual_seed(7)
    reqs = _requests(spec, 1)
    pool, row, pages, counts, _ = _pool(spec, reqs, g)
    kv = pool.view(-1, spec["page_size"], 1, 656)
    idx = _selections(spec, reqs, pages, counts, row, g)
    q = _queries(idx.shape[0], spec, g)
    out = torch.empty(idx.shape[0], spec["h_out"], 512, dtype=torch.bfloat16, device="cuda")
    valid = idx >= 0
    row_down = torch.where(valid, idx - spec["page_size"], idx)
    wrapped = torch.where(valid, (idx.long() * 656 % 2**31 // 656).int(), idx)
    neighbour = torch.where(valid, idx ^ 1, idx)
    for wrong, seen in ((row_down, "unheld pool byte"), (wrapped, "unheld pool byte"),
                        (neighbour, r"^\(")):  # fmt: skip
        op(q, kv, wrong, get_mla_metadata()[0], SCALE, out)
        with pytest.raises(AssertionError, match=seen):
            _check(q, kv, idx, out, g)
    op(q, kv, idx, get_mla_metadata()[0], SCALE, out)
    _check(q, kv, idx, out, g)


def test_1m_harness_sees_a_misread() -> None:
    """The 1M harness against three wrong slot maps over the pre-merge context cell's pool past
    2**31 bytes, its requests' keys all there: every key one owner row down (row 61 of its
    block, unwritten: NaN rows), every key's byte offset wrapped at 2**31 (what an offset kept
    in 31 bits reads: the unwritten low blocks), and every key at the neighbouring offset of
    its page (written rows, other keys: outside the band). Each is refused; the right map
    passes."""
    cell = next(c for c in op.CELLS if c.why.startswith("1M dep4 pre-merge context sub-batches"))
    spec = dict(cell.spec, chunk_rows=1024, cached=1047552)
    g = torch.Generator(device="cuda").manual_seed(7)
    reqs = _requests(spec, 1)
    pool, row, pages, counts, _ = _pool(spec, reqs, g)
    kv = pool.view(-1, spec["page_size"], 1, 656)
    idx = _selections(spec, reqs, pages, counts, row, g)
    q = _queries(idx.shape[0], spec, g)
    out = torch.empty(idx.shape[0], spec["h_out"], 512, dtype=torch.bfloat16, device="cuda")
    valid = idx >= 0
    row_down = torch.where(valid, idx - spec["page_size"], idx)
    wrapped = torch.where(valid, (idx.long() * 656 % 2**31 // 656).int(), idx)
    neighbour = torch.where(valid, idx ^ 1, idx)
    for wrong, seen in ((row_down, "unheld pool byte"), (wrapped, "unheld pool byte"),
                        (neighbour, r"^\(")):  # fmt: skip
        op(q, kv, wrong, get_mla_metadata()[0], SCALE, out)
        with pytest.raises(AssertionError, match=seen):
            _check(q, kv, idx, out, g)
    op(q, kv, idx, get_mla_metadata()[0], SCALE, out)
    _check(q, kv, idx, out, g)


def test_1m_harness_sees_a_misread_on_a_tep8_pool() -> None:
    """The 1M harness against three wrong slot maps over the pre-merge context cell's pool,
    whose requests' keys all sit past 2**31 bytes: every key one owner row down (row 61 of
    its block, unwritten: NaN rows), every key's byte offset wrapped at 2**31 (what an
    offset kept in 31 bits reads: the unwritten low blocks), and every key at the neighbouring
    offset of its page (written rows, other keys: outside the band). Each is refused; the
    right map passes."""
    cell = next(c for c in op.CELLS if c.why.startswith("1M tep8 pre-merge context"))
    spec = dict(cell.spec, chunk_rows=1024, cached=1047552)
    g = torch.Generator(device="cuda").manual_seed(7)
    reqs = _requests(spec, 1)
    pool, row, pages, counts, _ = _pool(spec, reqs, g)
    kv = pool.view(-1, spec["page_size"], 1, 656)
    idx = _selections(spec, reqs, pages, counts, row, g)
    q = _queries(idx.shape[0], spec, g)
    out = torch.empty(idx.shape[0], spec["h_out"], 512, dtype=torch.bfloat16, device="cuda")
    valid = idx >= 0
    row_down = torch.where(valid, idx - spec["page_size"], idx)
    wrapped = torch.where(valid, (idx.long() * 656 % 2**31 // 656).int(), idx)
    neighbour = torch.where(valid, idx ^ 1, idx)
    for wrong, seen in ((row_down, "unheld pool byte"), (wrapped, "unheld pool byte"),
                        (neighbour, r"^\(")):  # fmt: skip
        op(q, kv, wrong, get_mla_metadata()[0], SCALE, out)
        with pytest.raises(AssertionError, match=seen):
            _check(q, kv, idx, out, g)
    op(q, kv, idx, get_mla_metadata()[0], SCALE, out)
    _check(q, kv, idx, out, g)


@pytest.mark.parametrize("rows", [1, 16, 64, 256])
@pytest.mark.parametrize("h_out", [8, 16, 64])
def test_v_b_over_the_view_bit_equal_to_over_the_copy(h_out, rows) -> None:
    """Old versus new on identical inputs, `torch.equal` on the bits: the decode targets' V
    up-projection BMM over the live heads of the returned 64-head output, read in place,
    equals the same BMM over those heads copied out (`out`), with `b` the transposed view of
    the V rows as the targets pass it."""
    spec = dict(rows=rows, h_q=64, h_out=h_out, topk=2048, kv_len=4096, page_size=32,
                owners=63, pages=512, layers=1)  # fmt: skip
    q, kv, idx, out = _build(spec, 1000 * h_out + rows)
    meta = get_mla_metadata()[0]
    view = op(q, kv, idx, meta, SCALE, None)[:, :h_out]
    op(q, kv, idx, meta, SCALE, out)
    assert torch.equal(view.view(torch.int16), out.view(torch.int16))
    g = torch.Generator(device="cuda").manual_seed(rows)
    v_b_t = torch.transpose(
        torch.randn(h_out, 128, 512, generator=g, device="cuda").to(torch.bfloat16), 1, 2
    )
    new = torch.empty(rows, h_out * 128, dtype=torch.bfloat16, device="cuda")
    old = torch.empty_like(new)
    bmm_out(torch.transpose(view, 0, 1), v_b_t, torch.transpose(new.view(rows, h_out, 128), 0, 1))
    bmm_out(torch.transpose(out, 0, 1), v_b_t, torch.transpose(old.view(rows, h_out, 128), 0, 1))
    assert torch.equal(new.view(torch.int16), old.view(torch.int16))
