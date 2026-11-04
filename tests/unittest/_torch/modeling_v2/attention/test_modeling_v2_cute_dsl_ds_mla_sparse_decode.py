# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the cute_dsl_ds_mla_sparse_decode catalog entry.

Against `reference`: the kernel computes in bf16 (query, dequantized keys, P) with fp32
accumulation; the reference is fp32 throughout. Band: relative L2 per (row, head) under 1e-2
and the whole call under 4e-3 -- bf16's 2^-8 relative step, softened by the 512-wide sums;
flash_mla_sparse_decode's band, derived the same way, since the arithmetic is FlashMLA's.

Against flash_mla_sparse_decode, the kernel the targets ran before, on identical inputs (its
query padded to 64 heads): bit-identical wherever the two split a row's selections alike --
FlashMLA's own scheduler output is read back to show the partition is the same -- and, at the
split this entry dispatches, within the same band of each other.

The "1M" cells and `test_bitwise_equal_to_flash_mla_at_1m_where_it_splits_alike` run the 1M
config's pools on the device (`_pool_1m`): every byte 0xFF (NaN as e4m3, as a tile's fp32 scale
and as bf16 rope) but one owner row of the pages 16 decode sequences hold, keys up to 1048576,
those pages at the pool's top so the slots read pass 2**31 bytes; every output row is first
checked finite (a key read from anywhere else turns its row NaN), then gated as above.
"""

import math
import zlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention import (
    _ds_mla_sparse_decode_kernel as kernel,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.cute_dsl_ds_mla_sparse_decode import (  # noqa: E501
    SPLITS,
    build,
    splits_for,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.cute_dsl_ds_mla_sparse_decode import (  # noqa: E501
    cute_dsl_ds_mla_sparse_decode as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.flash_mla_sparse_decode import (  # noqa: E501
    flash_mla_sparse_decode as flash_op,
)
from tensorrt_llm.flash_mla import get_mla_metadata

__extra_import_path__ = [".."]
from _ds_mla_pool import ds_mla_rows  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401 — needs the path declared above

assert torch.cuda.is_available(), "cute_dsl_ds_mla_sparse_decode requires a CUDA device"

SCALE = 128**-0.5
BLOCKS = 2048 // 64  # FlashMLA's 64-token blocks per row
_COMPILED = {}


def _compiled(heads):
    dev = torch.cuda.current_device()
    if (dev, heads) not in _COMPILED:
        _COMPILED[dev, heads] = build(dev, num_heads=heads)
    return _COMPILED[dev, heads]


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
    heads = spec["heads"]
    q = torch.zeros(rows, spec.get("h_q", heads), 576)
    q[:, :heads] = torch.randn(rows, heads, 576, generator=g)
    if spec.get("nope"):
        q[..., 512:] = 0
    out = torch.full((rows, heads, 512), float("nan"), dtype=torch.bfloat16)
    return q.to(torch.bfloat16).cuda(), kv.cuda(), idx.cuda(), out.cuda()


def _gate(out, ref):
    o, r = out.float(), ref.float()
    per = (o - r).norm(dim=-1) / r.norm(dim=-1).clamp_min(1e-30)
    whole = float((o - r).norm() / r.norm())
    assert float(per.max()) < 1e-2 and whole < 4e-3, (float(per.max()), whole)


# ---- the 1M config's pools, on the device

#: Pool rows drawn per batch while writing the sequences' pages.
_DRAW_ROWS = 1 << 18
#: The step scratch a `q_offset` cell's views come from.
_STEP_ROWS = 32768


def _lens_1m(kv_len: int) -> list[int]:
    """The 16 decode sequences' key counts: `kv_len`, one 31 keys short, its 1/8, 1/16 and 1/32
    plus one, rows across the sub-chunk sizes, around top-k 2048 and down to one key. At 1048576
    they hold 2364359 keys on 73898 pages, distinct on every 1M pool."""
    spread = (kv_len, kv_len - 31, kv_len // 8 + 1, kv_len // 16 + 1, kv_len // 32 + 1, 16385,
              9217, 4097, 2049, 2048, 2047, 1025, 700, 257, 33, 1)  # fmt: skip
    return [max(1, min(kv_len, n)) for n in spread]


def _draw_rows(n: int, g: torch.Generator, nope: bool) -> torch.Tensor:
    """`ds_mla_rows` drawn on g's device: e4m3 codes, power-of-two tile scales, bf16 rope."""
    dev = g.device
    codes = ((torch.rand(n, 512, generator=g, device=dev) * 2 - 1) * 448).to(torch.float8_e4m3fn)
    scale = torch.exp2(torch.randint(-13, 1, (n, 4), generator=g, device=dev).float())
    rope = torch.zeros(n, 64, device=dev) if nope else torch.randn(n, 64, generator=g, device=dev)
    rope = (rope * 3).to(torch.bfloat16)
    return torch.cat([codes.view(torch.uint8), scale.view(torch.uint8), rope.view(torch.uint8)], 1)


def _pool_1m(spec, g) -> SimpleNamespace:
    """The pool `[blocks * owners, page, 1, 656]`, every byte 0xFF but the owner row (63's last,
    or 35 on a NoPE layer) of the 16 sequences' pages: with `far` the pool's top pages,
    distinct, the first sequence's last page on the top block; else drawn over the pool."""
    dev = g.device
    blocks, owners, page = spec["blocks"], spec["owners"], spec["page_size"]
    lens = _lens_1m(spec["kv_len"])
    counts = [-(-n // page) for n in lens]
    need = sum(counts)
    row = 35 if spec.get("nope") else owners - 1
    pool = torch.full((blocks, owners, page, 656), 0xFF, dtype=torch.uint8, device=dev)
    if spec.get("far"):
        assert need <= blocks, (need, blocks)
        pages = blocks - need + torch.randperm(need, generator=g, device=dev)
        top, last = int((pages == blocks - 1).nonzero()[0, 0]), counts[0] - 1
        pages[[top, last]] = pages[[last, top]]
    else:
        pages = torch.randint(0, blocks, (need,), generator=g, device=dev)
    step = _DRAW_ROWS // page
    for lo in range(0, need, step):
        part = pages[lo : lo + step]
        pool[part, row] = _draw_rows(part.numel() * page, g, spec.get("nope", False)).view(
            -1, page, 656
        )
    first = torch.tensor([0, *counts[:-1]], device=dev).cumsum(0)
    last_slot = (int(pages[counts[0] - 1]) * owners + row) * page + (lens[0] - 1) % page
    return SimpleNamespace(
        kv=pool.view(-1, page, 1, 656), pages=pages, first=first, lens=lens, row=row,
        last_slot=last_slot,
    )  # fmt: skip


def _selections_1m(spec, pool: SimpleNamespace, rows: int, g) -> torch.Tensor:
    """int32 `[rows, topk]` slots of decode rows, row r one of sequence r % 16 at its last
    position p: below topk keys every key 0..p then -1; else topk distinct keys, one in each of
    topk equal strata of [0, p], p itself the last. Each row's valid keys in a random order, -1
    last, each key's slot through its sequence's pages."""
    dev = g.device
    topk, page, owners = spec["topk"], spec["page_size"], spec["owners"]
    req = torch.arange(rows, device=dev) % 16
    pos = torch.tensor(pool.lens, device=dev)[req] - 1
    n_keys = pos[:, None] + 1
    col = torch.arange(topk + 1, device=dev)[None]
    edge = col * n_keys // topk
    draw = (
        edge[:, :-1]
        + (torch.rand(rows, topk, generator=g, device=dev) * (edge[:, 1:] - edge[:, :-1])).long()
    )
    draw[:, -1] = pos
    short = torch.where(col[:, :-1] < n_keys, col[:, :-1], -1)
    sel = torch.where(n_keys > topk, draw, short)
    sel = sel.gather(1, (torch.rand(sel.shape, generator=g, device=dev) + (sel < 0)).argsort(1))
    key = sel.clamp_min(0)
    slot = (pool.pages[pool.first[req][:, None] + key // page] * owners + pool.row) * page
    slot = torch.where(sel >= 0, slot + key % page, -1)
    assert int(slot.max()) < 2**31
    return slot.to(torch.int32)


def _queries_1m(rows: int, spec, g, scratch: int = 0) -> torch.Tensor:
    """bf16 `[rows, h_q, 576]`, the first `heads` heads drawn and the rest zero (rope zero on a
    NoPE layer); with `scratch`, the tail view of a scratch that many rows long."""
    total = max(rows, scratch)
    q = torch.zeros(total, spec["h_q"], 576, dtype=torch.bfloat16, device=g.device)
    q[:, : spec["heads"]] = torch.randn(total, spec["heads"], 576, generator=g, device=g.device)
    if spec.get("nope"):
        q[..., 512:] = 0
    return q[total - rows :]


def _check_1m(q, kv, idx, out) -> None:
    bad = ~torch.isfinite(out.float()).flatten(1).all(1)
    assert not bool(bad.any()), f"{int(bad.sum())} of {out.shape[0]} rows read an unheld byte"
    _gate(out, op.reference(q, kv, idx, SCALE, out))


def _run_1m(spec, seed) -> None:
    """A "1M" cell: per batch, the decode rows' slots over the device pool, `layers` calls into
    a NaN-filled `out` (tail views with `q_offset`, the rows ahead left alone), every row finite
    and in the band; with `captured`, the batch captured, replayed on new queries and slots, and
    the replay bitwise an eager call."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    pool = _pool_1m(spec, g)
    print(
        f"\n  pool of {spec['blocks']} blocks ({pool.kv.numel() / 1e9:.1f} GB), the first "
        f"sequence's last key at slot {pool.last_slot} (byte {pool.last_slot * 656:.4g})"
    )
    compiled = _compiled(spec["heads"])
    scratch = _STEP_ROWS if "q_offset" in spec else 0
    for b in spec.get("graph_batches", (spec["rows"],)):
        idx = _selections_1m(spec, pool, b, g)
        if spec.get("far"):
            assert int(idx.max()) == pool.last_slot, (int(idx.max()), pool.last_slot)
            assert int(idx.max()) * 656 >= 2**31
        q = _queries_1m(b, spec, g, scratch)
        out_buf = torch.full((max(b, scratch), spec["heads"], 512), -1, dtype=torch.int16)
        out_buf = out_buf.cuda().view(torch.bfloat16)
        out = out_buf[out_buf.shape[0] - b :]
        if scratch:
            assert out_buf.shape[0] - b == spec["q_offset"]
        for _ in range(spec["layers"]):
            op(q, pool.kv, idx, SCALE, out, compiled=compiled)
        _check_1m(q, pool.kv, idx, out)
        if scratch:
            ahead = out_buf[: spec["q_offset"]].view(torch.int16)
            assert bool((ahead == -1).all()), "the call wrote ahead of its view"
        if not spec.get("captured"):
            continue
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            op(q, pool.kv, idx, SCALE, out, compiled=compiled)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(spec["layers"]):
                op(q, pool.kv, idx, SCALE, out, compiled=compiled)
        q.copy_(_queries_1m(b, spec, g))
        idx.copy_(_selections_1m(spec, pool, b, g))
        out.view(torch.int16).fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        _check_1m(q, pool.kv, idx, out)
        eager = op(q, pool.kv, idx, SCALE, torch.empty_like(out), compiled=compiled)
        assert torch.equal(eager.view(torch.int16), out.view(torch.int16)), "replay != eager"
        del graph


@pytest.mark.parametrize(
    "cell",
    [pytest.param(c, marks=[pytest.mark.large] if c.spec.get("large") else []) for c in op.CELLS],
    ids=lambda c: c.why[:60],
)
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode())
    if "blocks" in spec:
        try:
            _run_1m(spec, seed)
        finally:
            torch.cuda.empty_cache()
        return
    q, kv, idx, out = _build(spec, seed)
    compiled = _compiled(spec["heads"])
    for _ in range(spec["layers"]):
        op(q, kv, idx, SCALE, out, compiled=compiled)
    _gate(out, op.reference(q, kv, idx, SCALE, out))
    if spec.get("captured"):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            op(q, kv, idx, SCALE, out, compiled=compiled)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(spec["layers"]):
                op(q, kv, idx, SCALE, out, compiled=compiled)
        q2, kv2, idx2, _ = _build(spec, seed + 1)
        q.copy_(q2), kv.copy_(kv2), idx.copy_(idx2)
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _gate(out, op.reference(q, kv, idx, SCALE, out))


def _flash_mla(q, kv, idx, heads):
    """flash_mla_sparse_decode on the same inputs, the query padded to 64 heads; its output
    and the per-row partition its scheduler chose, as (first block, end block) ranges."""
    rows = q.shape[0]
    q64 = torch.zeros(rows, 64, 576, dtype=q.dtype, device=q.device)
    q64[:, :heads] = q[:, :heads]
    out = torch.empty(rows, heads, 512, dtype=torch.bfloat16, device=q.device)
    meta = get_mla_metadata()[0]
    flash_op(q64, kv, idx, meta, SCALE, out)
    parts = {r: [] for r in range(rows)}
    for b_req, e_req, b_blk, e_blk, *_ in meta.tile_scheduler_metadata.cpu().tolist():
        for r in range(b_req, min(e_req, rows - 1) + 1):
            lo = b_blk if r == b_req else 0
            hi = e_blk if r == e_req else BLOCKS
            if hi > lo:
                parts[r].append((lo, hi))
    return out, {r: sorted(v) for r, v in parts.items()}


def _uniform(ns):
    return [(k * BLOCKS // ns, (k + 1) * BLOCKS // ns) for k in range(ns)]


_ALIGNED = dict(topk=2048, kv_len=9000, page_size=32, owners=63, pages=512, layers=1)


@pytest.mark.parametrize(
    "rows,heads,ns,extra",
    [
        (16, 16, 8, {}),
        (32, 16, 4, {}),
        (64, 16, 2, {}),
        (128, 16, 1, {}),
        (16, 8, 8, {}),
        (32, 8, 4, {}),
        (64, 8, 2, {}),
        (128, 8, 1, {}),
        (64, 16, 2, dict(holes=True)),
        (32, 8, 4, dict(holes=True, kv_len=1500)),
        (16, 16, 8, dict(nope=True)),
        (128, 16, 1, dict(h_q=64)),
    ],
)
def test_bitwise_equal_to_flash_mla_where_it_splits_alike(rows, heads, ns, extra) -> None:
    """FlashMLA's scheduler gives a 2048-selection row 32 blocks + 5 of overhead and every
    partition ceil(37 rows / 152) + 5: at 16, 32, 64 and 128 rows that cuts each row into 8,
    4, 2 and 1 equal parts. Run at that split, this kernel must reproduce FlashMLA's output
    bit for bit: same conversions, same accumulators, same summation orders."""
    if torch.cuda.get_device_properties(0).multi_processor_count != 152:
        pytest.skip("the partitions coincide at these batches on 152 SMs")
    spec = dict(_ALIGNED, rows=rows, heads=heads, **extra)
    q, kv, idx, out = _build(spec, rows * 31 + heads)
    ref_out, parts = _flash_mla(q, kv, idx, heads)
    assert all(p == _uniform(ns) for p in parts.values()), parts[0]
    kernel.launch(_compiled(heads).fns[ns], q, kv, idx, out, SCALE)
    torch.cuda.synchronize()
    diff = int((out.view(torch.int16) != ref_out.view(torch.int16)).sum())
    assert diff == 0, f"{diff} of {out.numel()} outputs differ from FlashMLA's"


@pytest.mark.parametrize(
    "rows,heads,ns,h_q",
    [
        (16, 16, 8, 16),
        (32, 16, 4, 16),
        (64, 16, 2, 16),
        (128, 16, 1, 16),
        (16, 8, 8, 8),
        (32, 8, 4, 8),
        (64, 8, 2, 8),
        (128, 8, 1, 8),
        (16, 16, 8, 64),
    ],
)
def test_bitwise_equal_to_flash_mla_at_1m_where_it_splits_alike(rows, heads, ns, h_q) -> None:
    """At the 1M config's shapes -- the boot pool (74183 blocks, slots past 2**31 bytes, the
    pool's top slot read), keys up to 1048576 down to one, -1 tails -- FlashMLA's scheduler
    still cuts every row of a 16 / 32 / 64 / 128-row batch into 8 / 4 / 2 / 1 equal parts (read
    back from its metadata), and this kernel run at that split reproduces its output bit for
    bit, at 16 and 8 heads and reading 16 of a 64-head query."""
    if torch.cuda.get_device_properties(0).multi_processor_count != 152:
        pytest.skip("the partitions coincide at these batches on 152 SMs")
    spec = dict(op.CELLS[0].spec, **dict(
        topk=2048, kv_len=1048576, page_size=32, owners=63, blocks=74183, far=True,
        heads=heads, h_q=h_q,
    ))  # fmt: skip
    g = torch.Generator(device="cuda").manual_seed(rows * 131 + heads + h_q)
    try:
        pool = _pool_1m(spec, g)
        idx = _selections_1m(spec, pool, rows, g)
        assert int(idx.max()) == pool.last_slot and pool.last_slot * 656 >= 2**31
        q = _queries_1m(rows, spec, g)
        ref_out, parts = _flash_mla(q, pool.kv, idx, heads)
        assert all(p == _uniform(ns) for p in parts.values()), parts[0]
        out = torch.full((rows, heads, 512), float("nan"), dtype=torch.bfloat16, device="cuda")
        kernel.launch(_compiled(heads).fns[ns], q, pool.kv, idx, out, SCALE)
        torch.cuda.synchronize()
        _check_1m(q, pool.kv, idx, out)
        diff = int((out.view(torch.int16) != ref_out.view(torch.int16)).sum())
        assert diff == 0, f"{diff} of {out.numel()} outputs differ from FlashMLA's"
    finally:
        torch.cuda.empty_cache()


@pytest.mark.parametrize("heads", [16, 8])
@pytest.mark.parametrize("rows", [1, 4, 16, 24, 64, 200])
def test_matches_flash_mla_at_the_dispatched_split(rows, heads) -> None:
    """At the split this entry dispatches the partition can differ from FlashMLA's (32
    one-block splits at 1-4 rows, mixed ones at 200): the same arithmetic over a different
    split of the selections. Both stay in the reference band, and in it of each other."""
    spec = dict(_ALIGNED, rows=rows, heads=heads)
    q, kv, idx, out = _build(spec, rows * 7 + heads)
    compiled = _compiled(heads)
    op(q, kv, idx, SCALE, out, compiled=compiled)
    ref_out, _ = _flash_mla(q, kv, idx, heads)
    assert splits_for(rows, compiled) in SPLITS
    ref = op.reference(q, kv, idx, SCALE, out)
    _gate(out, ref)
    _gate(ref_out, ref)
    _gate(out, ref_out)


def test_is_valid_refuses_padded_heads_and_strided_queries() -> None:
    spec = dict(_ALIGNED, rows=2, heads=16)
    q, kv, idx, out = _build(spec, 1)
    compiled = _compiled(16)
    op.is_valid(q, kv, idx, SCALE, out, compiled=compiled)
    q64 = torch.zeros(2, 64, 576, dtype=q.dtype, device=q.device)
    out64 = torch.empty(2, 64, 512, dtype=out.dtype, device=out.device)
    with pytest.raises(ValueError, match="heads"):
        op.is_valid(q64, kv, idx, SCALE, out64, compiled=compiled)
    with pytest.raises(ValueError, match="contiguous"):
        op.is_valid(q64[:, :16], kv, idx, SCALE, out, compiled=compiled)
