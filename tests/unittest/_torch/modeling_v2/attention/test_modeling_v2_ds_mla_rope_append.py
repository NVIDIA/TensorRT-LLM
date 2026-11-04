# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ds_mla_rope_append catalog entry.

Every cell gates the pool bitwise against the entry's reference on the codes and the tile
scales and on every byte the call must not touch; the rope halves (key in the pool, query in
place) within one bf16 unit in the last place, the room an FMA in the rotation leaves.

The catalog's cells draw their pool on the host and compare all of it. The `_ML4_1M`,
`_ML4_1M_DEP4` and `_ML4_1M_TEP8` cells build theirs on the device, because their pools reach
195.0 GB (142.7 GB at tep4, tp4 and dep4): every byte a sentinel but the blocks the call
writes (one per request page, random bytes). The reference runs on a copy of just those
blocks under the block table renumbered to them; after the call they must equal it, and once
they are set back to the sentinel the whole pool must read as the sentinel, so a write
anywhere else shows. Their rope table is built on the device as the target builds its
`_rope.indexer`, 1048576 rows, with a seeded LeRoPE alpha in place of the checkpoint's.
"""

import math
import zlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.ds_mla_rope_append import (
    ROPE_OFFSET,
    TOKEN_BYTES,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.ds_mla_rope_append import (
    ds_mla_rope_append as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — needs the path declared above

assert torch.cuda.is_available(), "ds_mla_rope_append requires a CUDA device"

_Q_HEADS = 64  # the absorbed query's kernel width; the cell's heads are its first ones
_POSITIONS = 32768 + 64


def _every_bf16_amax(tokens: int) -> torch.Tensor:
    """`tokens` x 4 tiles whose amaxes are every finite non-negative bf16 value once (the rest
    of each tile below it): zero, the subnormals, the 1e-4 floor, powers of two, the largest."""
    bits = torch.arange(0, 0x7F80, dtype=torch.int32)  # +0 .. largest finite
    amax = bits.to(torch.int16).view(torch.bfloat16).float()
    n = tokens * 4
    amax = torch.cat([amax, amax[: max(0, n - amax.numel())]])[:n]
    g = torch.Generator().manual_seed(7)
    tiles = torch.rand(n, 128, generator=g) * 2 - 1
    tiles = tiles * amax[:, None]
    tiles[torch.arange(n), torch.randint(0, 128, (n,), generator=g)] = amax
    return tiles.view(tokens, 512)


def _build(spec, seed: int):
    g = torch.Generator().manual_seed(seed)
    t, tpb, blocks, owners = (
        spec["tokens"],
        spec["tokens_per_block"],
        spec["blocks"],
        spec["owners"],
    )
    if spec["phase"] == "decode":
        # One request per row; its token's page is its own, every other entry a stale page.
        width = math.ceil(_POSITIONS / tpb)
        pos = torch.randint(0, width * tpb, (t,), generator=g)
        req = torch.arange(t)
        table = torch.randint(0, blocks, (t, width), generator=g)
        table[req, pos // tpb] = torch.randperm(blocks, generator=g)[:t]
    else:
        pos = spec["cached"] + torch.arange(t)
        req = torch.zeros(t, dtype=torch.long)
        width = math.ceil((spec["cached"] + t) / tpb)
        table = torch.randperm(blocks, generator=g)[:width].view(1, width)
    if spec.get("amax") == "every_bf16":
        nope = _every_bf16_amax(t)
    else:
        mag = torch.exp2(torch.rand(t, 4, 1, generator=g) * 24 - 16)
        nope = (torch.randn(t, 4, 128, generator=g) * mag).view(t, 512)
    lat = torch.cat([nope, torch.randn(t, 64, generator=g) * 4], 1).to(torch.bfloat16)
    ang = torch.arange(_POSITIONS).float()[:, None] * torch.rand(32, generator=g)[None]
    table_cs = torch.cat([ang.cos(), ang.sin()], 1).to(torch.bfloat16).float()
    pool = torch.randint(0, 256, (blocks, owners, tpb, 656), dtype=torch.uint8, generator=g)
    q = None
    if spec["heads"]:
        q = torch.randn(t, _Q_HEADS, 576, generator=g).to(torch.bfloat16)
    cuda = lambda x: None if x is None else x.cuda()  # noqa: E731
    return dict(
        latent=cuda(lat),
        q_full=cuda(q),
        pos=cuda(pos.to(torch.int32)),
        table=cuda(table_cs),
        req_idx=cuda(req.to(torch.int32)),
        block_table=cuda(table.to(torch.int32)),
        pool=cuda(pool),
        row=spec["row"],
        rope=spec["rope"],
        heads=spec["heads"],
    )


def _args(b):
    q = None if b["q_full"] is None else b["q_full"][:, : b["heads"]]
    return (b["latent"], q, b["pos"], b["table"], b["req_idx"], b["block_table"], b["pool"],
            b["row"], b["rope"])  # fmt: skip


def _ulp_close(a: torch.Tensor, b: torch.Tensor) -> None:
    a, b = a.float(), b.float()
    ulp = torch.exp2(torch.floor(torch.log2(b.abs().clamp_min(2.0**-126))) - 7)
    assert bool(((a - b).abs() <= ulp).all()), float(((a - b).abs() / ulp).max())


def _check(b, pool_before, q_before) -> None:
    args = list(_args(b))
    args[6] = pool_before.clone()
    if args[1] is not None:
        args[1] = q_before[:, : b["heads"]].clone()
    ref_pool, ref_q = op.reference(*args)
    pool = b["pool"]
    # The written rows: their codes and scales bitwise, their rope halves within an ulp.
    # Every other byte of the pool bitwise (it starts as random bytes, so a stray write shows).
    tpb = pool.shape[2]
    pos = b["pos"].long()
    block = b["block_table"][b["req_idx"].long(), pos // tpb].long()
    written = (block, torch.full_like(block, b["row"]), pos % tpb)
    assert torch.equal(pool[..., :ROPE_OFFSET], ref_pool[..., :ROPE_OFFSET]), (
        f"{int((pool[..., :ROPE_OFFSET] != ref_pool[..., :ROPE_OFFSET]).sum())} code/scale "
        "bytes differ"
    )
    _ulp_close(
        pool[written][:, ROPE_OFFSET:].contiguous().view(torch.bfloat16),
        ref_pool[written][:, ROPE_OFFSET:].contiguous().view(torch.bfloat16),
    )
    untouched, ref_untouched = pool.clone(), ref_pool.clone()
    untouched[written] = 0
    ref_untouched[written] = 0
    assert torch.equal(untouched, ref_untouched), "bytes outside the written rows changed"
    if b["q_full"] is not None:
        h = b["heads"]
        _ulp_close(b["q_full"][:, :h], ref_q)
        assert torch.equal(b["q_full"][:, h:], q_before[:, h:]), (
            "heads past the call's were written"
        )


# -------------------------------------------- the _ML4_1M, _ML4_1M_DEP4 and _ML4_1M_TEP8 cells

#: The checkpoint's rope_parameters: YaRN, rope_theta 1e4, factor 2048, original
#: max_position_embeddings 512, beta_fast 32, beta_slow 1.
_YARN = dict(theta=1.0e4, factor=2048.0, original=512.0, beta_fast=32.0, beta_slow=1.0)
#: Under `positions` "tail" a decode row's position is one of the table's last this many.
_TAIL = 64
_SENTINEL = 0x5A
_SENTINEL_WORD = int.from_bytes(bytes([_SENTINEL]) * 8, "little", signed=True)
_TABLES: dict = {}


def _indexer_table(rows: int) -> torch.Tensor:
    """The target's `_rope.indexer` at `rows` positions, built as `_rope_tables` builds it:
    YaRN's blend of the extrapolated and the interpolated inverse frequency, both scaled by
    LeRoPE's exp(alpha), over a ramp taken without LeRoPE (fp32); the fp32 angle grid; cos
    and sin rounded through bf16; `[cos(32) | sin(32)]` fp32 on the device. alpha is a
    checkpoint tensor, so a seeded draw stands in for it. Built once per process (256 MiB at
    1048576 rows)."""
    if rows not in _TABLES:
        y, dim = _YARN, 64
        pos_freqs = y["theta"] ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        alpha = 0.1 * torch.randn(dim // 2, generator=torch.Generator().manual_seed(11))
        lerope = torch.exp(alpha)
        extrapolated = 1.0 / pos_freqs * lerope
        interpolated = 1.0 / (y["factor"] * pos_freqs) * lerope

        def correction_dim(rotations: float) -> float:
            turns = y["original"] / (rotations * 2 * math.pi)
            return dim * math.log(turns) / (2 * math.log(y["theta"]))

        low = max(math.floor(correction_dim(y["beta_fast"])), 0)
        high = min(math.ceil(correction_dim(y["beta_slow"])), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
        extrapolation = 1 - ramp
        inv = interpolated * (1 - extrapolation) + extrapolated * extrapolation
        angle = torch.einsum("i,j -> ij", torch.arange(rows, dtype=torch.float32), inv)
        cos = angle.cos().to(torch.bfloat16).float()
        sin = angle.sin().to(torch.bfloat16).float()
        _TABLES[rows] = torch.cat([cos, sin], dim=1).cuda().contiguous()
    return _TABLES[rows]


def _requests(spec, gen: torch.Generator, chunk: int = 0, batch: int = 0) -> list:
    """`(cached, new)` per request of the call, in its row order; request i is block-table
    row i."""
    phase = spec["phase"]
    if phase == "context":
        if "sweep_positions" in spec:
            return [(spec["sweep_positions"][0] + chunk * spec["tokens"], spec["tokens"])]
        return [(spec["cached"], spec["tokens"])]
    if phase == "packed":
        return [tuple(c) for c in spec["contexts"]]
    if phase == "mixed":
        return [tuple(c) for c in spec["contexts"]] + [(p, 1) for p in spec["decode_positions"]]
    assert phase == "decode" and spec["positions"] == "tail", spec
    rows = batch or spec["tokens"]
    top = spec["max_pos"] - 1
    pos = (top - torch.randint(0, _TAIL, (rows,), generator=gen)).tolist()
    pos[int(torch.randint(0, rows, (1,), generator=gen))] = top
    return [(p, 1) for p in pos]


def _step(spec, reqs: list, gen: torch.Generator, gpu: torch.Generator, pool, table, pad=0):
    """One call's inputs. Each request page the call writes gets a block of its own (under
    `far` the pool's top blocks, the step's last position on its last block); every other
    block-table entry points at a block the call does not own. The last `pad` requests are
    attention DP's padding dummy repeated: one row each, each its own block-table row holding
    the first one's pages, and one latent and query row repeated, so they all write one slot.
    The pool is reset to the sentinel and the owned blocks drawn at random; `compact` is the
    block table renumbered to the owned blocks in `own` order, for the reference."""
    tpb, width, blocks = spec["tokens_per_block"], spec["table_width"], spec["blocks"]
    pos = torch.cat([torch.arange(c, c + n) for c, n in reqs])
    req = torch.repeat_interleave(torch.arange(len(reqs)), torch.tensor([n for _, n in reqs]))
    top = int(pos.max())
    assert len(reqs) <= 16 and top < spec["max_pos"] and top // tpb < width, (len(reqs), top)
    first = len(reqs) - pad
    assert all(n == 1 for _, n in reqs[first:]), reqs
    key = req.clamp(max=first) * width + pos // tpb
    pages = torch.unique(key)
    n = pages.numel()
    assert n < blocks, f"{n} pages leave no stale block in {blocks}"
    if spec.get("far"):
        own = blocks - n + torch.randperm(n, generator=gen)
        i = int(torch.searchsorted(pages, key[int(pos.argmax())]))
        j = int((own == blocks - 1).nonzero())
        own[[i, j]] = own[[j, i]]
    else:
        own = torch.randperm(blocks, generator=gen)[:n]
    free = torch.ones(blocks, dtype=torch.bool)
    free[own] = False
    stale = free.nonzero().flatten()
    block_table = stale[torch.randint(0, stale.numel(), (16 * width,), generator=gen)]
    block_table[pages] = own
    compact = torch.zeros(16 * width, dtype=torch.int32)
    compact[pages] = torch.arange(n, dtype=torch.int32)
    block_table, compact = block_table.view(16, width), compact.view(16, width)
    if pad > 1:
        block_table[first + 1 : len(reqs)] = block_table[first]
        compact[first + 1 : len(reqs)] = compact[first]
    t = pos.numel()
    mag = torch.exp2(torch.rand(t, 4, 1, generator=gpu, device="cuda") * 24 - 16)
    nope = (torch.randn(t, 4, 128, generator=gpu, device="cuda") * mag).view(t, 512)
    k_pe = torch.randn(t, 64, generator=gpu, device="cuda") * 4
    latent = torch.cat([nope, k_pe], 1).to(torch.bfloat16)
    q_full = torch.randn(t, _Q_HEADS, 576, generator=gpu, device="cuda", dtype=torch.bfloat16)
    if pad:
        latent[t - pad :] = latent[t - pad]
        q_full[t - pad :] = q_full[t - pad]
    own = own.cuda()
    pool.fill_(_SENTINEL)
    pool[own] = torch.randint(
        0, 256, (n, *pool.shape[1:]), dtype=torch.uint8, device="cuda", generator=gpu
    )
    return SimpleNamespace(
        latent=latent,
        q_full=q_full,
        pos=pos.to(torch.int32).cuda(),
        req=req.to(torch.int32).cuda(),
        block_table=block_table.to(torch.int32).cuda(),
        compact=compact.cuda(),
        own=own,
        table=table,
        pool=pool,
        row=spec["row"],
        rope=spec["rope"],
        heads=spec["heads"],
    )


def _args_1m(s, block_table=None, table=None, heads=None):
    q = s.q_full[:, : s.heads if heads is None else heads]
    return (s.latent, q, s.pos, s.table if table is None else table, s.req,
            s.block_table if block_table is None else block_table, s.pool, s.row,
            s.rope)  # fmt: skip


def _assert_sentinel(pool: torch.Tensor) -> None:
    """Every byte of `pool` reads as the sentinel: int64 words, 2 GiB at a time."""
    words = pool.view(-1).view(torch.int64)
    step = 1 << 28
    bad = torch.zeros((), dtype=torch.int64, device=pool.device)
    for i in range(0, words.numel(), step):
        bad += torch.count_nonzero(words[i : i + step] != _SENTINEL_WORD)
    assert int(bad) == 0, f"{int(bad)} 8-byte words outside the written blocks changed"


def _check_1m(s, snap: torch.Tensor, q_before: torch.Tensor) -> int:
    """Gate the call against `reference` on `snap` (the owned blocks before it) under the
    renumbered table: the owned blocks' codes and scales and every byte outside the written
    rows' rope halves bitwise, those within a bf16 ulp; the rest of the pool the sentinel;
    the call's heads' rope tails within an ulp, the rest of `q_full` bitwise. Returns the
    largest pool byte offset a row was written at."""
    h = s.heads
    ref_pool, ref_q = op.reference(
        s.latent, q_before[:, :h], s.pos, s.table, s.req, s.compact, snap, s.row, s.rope
    )
    after = s.pool[s.own]
    tpb, owners = s.pool.shape[2], s.pool.shape[1]
    pos = s.pos.long()
    slot = s.compact[s.req.long(), pos // tpb].long()
    off = pos % tpb
    assert torch.equal(after[..., :ROPE_OFFSET], ref_pool[..., :ROPE_OFFSET]), (
        f"{int((after[..., :ROPE_OFFSET] != ref_pool[..., :ROPE_OFFSET]).sum())} code/scale "
        "bytes of the written blocks differ"
    )
    _ulp_close(
        after[slot, s.row, off, ROPE_OFFSET:].contiguous().view(torch.bfloat16),
        ref_pool[slot, s.row, off, ROPE_OFFSET:].contiguous().view(torch.bfloat16),
    )
    after[slot, s.row, off, ROPE_OFFSET:] = 0
    ref_pool[slot, s.row, off, ROPE_OFFSET:] = 0
    assert torch.equal(after, ref_pool), "bytes of the written blocks outside its rows changed"
    s.pool[s.own] = _SENTINEL
    _assert_sentinel(s.pool)
    assert torch.equal(s.q_full[:, h:], q_before[:, h:]), "heads past the call's were written"
    assert torch.equal(s.q_full[:, :h, :512], q_before[:, :h, :512]), "q's latent part changed"
    if s.rope:
        _ulp_close(s.q_full[:, :h, 512:], ref_q[..., 512:])
    else:
        assert torch.equal(s.q_full, q_before), "q changed on a NoPE owner"
    block = s.block_table[s.req.long(), pos // tpb].long()
    return int((((block * owners + s.row) * tpb + off) * TOKEN_BYTES).max())


def _eager_1m(s) -> int:
    snap, q_before = s.pool[s.own], s.q_full.clone()
    op(*_args_1m(s))
    return _check_1m(s, snap, q_before)


def _capture(s) -> torch.cuda.CUDAGraph:
    """The engine's capture: one eager call on a side stream, then the captured call."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(*_args_1m(s))
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(*_args_1m(s))
    return graph


def _replay(s, new, graph, rows: int) -> tuple:
    """Write `new`'s rows, positions and pages into the captured inputs, replay and gate it;
    then run the same inputs eagerly from the same pool and require the replay's bytes.
    Returns the top byte offset written, and the pool snapshot, the replayed blocks and the
    query before the replay."""
    for name in ("latent", "q_full", "pos", "req", "block_table"):
        getattr(s, name).copy_(getattr(new, name))
    s.compact, s.own = new.compact, new.own
    snap, q_before = s.pool[s.own], s.q_full.clone()
    graph.replay()
    torch.cuda.synchronize()
    replayed, replayed_q = s.pool[s.own], s.q_full.clone()
    written = _check_1m(s, snap, q_before)
    s.pool[s.own] = snap
    s.q_full.copy_(q_before)
    op(*_args_1m(s))
    assert torch.equal(s.pool[s.own], replayed), f"batch {rows}: the replay differs from eager"
    assert torch.equal(s.q_full, replayed_q), f"batch {rows}: replayed q differs from eager"
    return written, snap, replayed, q_before


def _graphs_1m(spec, seed: int, pool, table) -> int:
    """At each of the cell's batches: capture the call, replay it on new rows, positions and
    pages, gate it and require the eager call's bytes."""
    top = 0
    for rows in spec.get("graph_batches", (spec["tokens"],)):
        gen = torch.Generator().manual_seed(seed * 100 + rows)
        gpu = torch.Generator(device="cuda").manual_seed(seed * 100 + rows)
        s = _step(spec, _requests(spec, gen, batch=rows), gen, gpu, pool, table)
        graph = _capture(s)
        new = _step(spec, _requests(spec, gen, batch=rows), gen, gpu, pool, table)
        written = _replay(s, new, graph, rows)[0]
        print(f"graph batch {rows}: replay gated, bitwise eager, top byte offset {written}")
        top = max(top, written)
        del graph
    return top


def _boot_1m(spec, seed: int, pool, table) -> int:
    """The dep4 engine's decode graphs under attention DP, at each of the cell's batches b:
    captured on the CUDA-graph warm-up batch (the longest dummy's new key at max_pos - 2 in
    row 0 ahead of b - 1 one-key fillers at 0, each a request on a page of its own; that
    eager call gated), then replayed with `live` rows ahead of b - live rows of the padding
    dummy at position 0 on one slot -- the live row at the table's last position, at 0 (an
    idle rank's attention-DP dummy) and at a drawn one. Each replay is gated and bitwise the
    eager call; with two or more padding rows, also bitwise the eager call with the padding
    cut to its first row, so the repeated rows leave the slot as one dummy row's write
    does."""
    top, live, tpb = 0, spec["live"], spec["tokens_per_block"]
    last = spec["max_pos"] - 1
    for rows in spec["graph_batches"]:
        gen = torch.Generator().manual_seed(seed * 100 + rows)
        gpu = torch.Generator(device="cuda").manual_seed(seed * 100 + rows)
        s = _step(spec, [(last - 1, 1)] + [(0, 1)] * (rows - 1), gen, gpu, pool, table)
        snap, q_before = pool[s.own], s.q_full.clone()
        graph = _capture(s)
        top = max(top, _check_1m(s, snap, q_before))
        pad = rows - live
        drawn = int(torch.randint(0, spec["max_pos"], (1,), generator=gen))
        for at in (last, 0, drawn):
            reqs = [(at, 1)] * live + [(0, 1)] * pad
            new = _step(spec, reqs, gen, gpu, pool, table, pad=pad)
            slots = new.block_table[new.req.long(), new.pos.long() // tpb]
            assert slots[live:].unique().numel() == min(pad, 1), "padding rows on one slot"
            assert slots[:live].unique().numel() == live and not bool(
                torch.isin(slots[:live], slots[live:]).any()
            ), "live rows on slots of their own"
            written, snap, replayed, q_before = _replay(s, new, graph, rows)
            top = max(top, written)
            if pad > 1:
                k = live + 1
                pool[s.own] = snap
                q_cut = q_before[:k].clone()
                op(s.latent[:k], q_cut, s.pos[:k], s.table, s.req[:k], s.block_table, pool,
                   s.row, s.rope)  # fmt: skip
                assert torch.equal(pool[s.own], replayed), (
                    f"batch {rows}: {pad} padding rows on one slot differ from one row's write"
                )
                assert torch.equal(q_cut, s.q_full[:k]), f"batch {rows}: the cut call's q"
            cut = ", bitwise the call cut to one padding row" if pad > 1 else ""
            print(f"boot graph batch {rows}: live row at {at} ahead of {pad} padding rows on "
                  f"one slot, gated, bitwise eager{cut}")  # fmt: skip
        del graph
    return top


def _run_1m(spec, seed: int) -> None:
    torch.cuda.reset_peak_memory_stats()
    table = _indexer_table(spec["max_pos"])
    pool = torch.empty(
        (spec["blocks"], spec["owners"], spec["tokens_per_block"], TOKEN_BYTES),
        dtype=torch.uint8,
        device="cuda",
    )
    if spec["phase"] == "decode":
        drive = _boot_1m if spec["positions"] == "boot" else _graphs_1m
        top = drive(spec, seed, pool, table)
    else:
        chunks = 1
        if "sweep_positions" in spec:
            lo, hi = spec["sweep_positions"]
            chunks = (hi - lo) // spec["tokens"]
            assert lo + chunks * spec["tokens"] == hi, spec["sweep_positions"]
        top = 0
        for chunk in range(chunks):
            gen = torch.Generator().manual_seed(seed * 100 + chunk)
            gpu = torch.Generator(device="cuda").manual_seed(seed * 100 + chunk)
            s = _step(spec, _requests(spec, gen, chunk=chunk), gen, gpu, pool, table)
            top = max(top, _eager_1m(s))
    if pool.numel() > 2**31:
        assert top >= 2**31, f"no row was written past 2^31 bytes (top byte offset {top})"
    if spec.get("far"):
        # The step's last position is the table's last, on the pool's last block.
        tpb = spec["tokens_per_block"]
        last = (((spec["blocks"] - 1) * spec["owners"] + spec["row"]) * tpb
                + (spec["max_pos"] - 1) % tpb) * TOKEN_BYTES  # fmt: skip
        assert top == last, (top, last)
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"pool {pool.numel()} bytes, top written row at byte offset {top}, peak {peak:.1f} GiB")
    del pool
    torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "cell",
    [pytest.param(c, marks=pytest.mark.large) if c.spec.get("large") else c for c in op.CELLS],
    ids=lambda c: c.why[:60],
)
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    spec = cell.spec
    if "max_pos" in spec:
        _run_1m(spec, seed=zlib.crc32(cell.why.encode()) % 1000)
        return
    b = _build(spec, seed=zlib.crc32(cell.why.encode()) % 1000)
    pool0 = b["pool"].clone()
    q0 = None if b["q_full"] is None else b["q_full"].clone()
    if spec.get("captured"):
        # Capture on the cell's data, replay on new rows, positions and slots written into
        # the captured inputs.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            op(*_args(b))
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op(*_args(b))
        nb = _build(spec, seed=zlib.crc32(cell.why.encode()) % 1000 + 1)
        for k in ("latent", "pos", "table", "req_idx", "block_table", "pool", "q_full"):
            if b[k] is not None:
                b[k].copy_(nb[k])
        pool0 = b["pool"].clone()
        q0 = None if b["q_full"] is None else b["q_full"].clone()
        graph.replay()
        torch.cuda.synchronize()
    else:
        op(*_args(b))
    _check(b, pool0, q0)


def test_1m_harness_refuses_misplaced_writes() -> None:
    """Controls for the 1M harness at the `_ML4_1M` cells' 16 heads: a call whose block table
    sends one page to a stale block, a call that reads its rope one table row early, a call
    that ropes a 17th query head, and a call followed by one stray byte at the pool's far end
    each fail `_check_1m`, by the check meant for it."""
    spec = dict(phase="context", tokens=64, cached=1015808, heads=16, rope=True, blocks=256,
                owners=63, tokens_per_block=32, row=62, max_pos=1048576,
                table_width=32772)  # fmt: skip
    table = _indexer_table(spec["max_pos"])
    pool = torch.empty((256, 63, 32, TOKEN_BYTES), dtype=torch.uint8, device="cuda")
    gen, gpu = torch.Generator().manual_seed(1), torch.Generator(device="cuda").manual_seed(1)

    def run(**kw):
        s = _step(spec, _requests(spec, gen), gen, gpu, pool, table)
        snap, q_before = pool[s.own], s.q_full.clone()
        if kw.pop("misplace", False):
            kw["block_table"] = s.block_table.clone()
            kw["block_table"][0, 1015808 // 32] = s.block_table[1, 0]
        op(*_args_1m(s, **kw))
        return s, snap, q_before

    with pytest.raises(AssertionError, match="code/scale bytes"):
        _check_1m(*run(misplace=True))
    with pytest.raises(AssertionError):
        _check_1m(*run(table=torch.roll(table, 1, dims=0)))
    with pytest.raises(AssertionError, match="heads past the call's"):
        _check_1m(*run(heads=17))
    s, snap, q_before = run()
    stray = max(set(range(256)) - set(s.own.tolist()))
    pool[stray, -1, -1, -1] = _SENTINEL ^ 1
    with pytest.raises(AssertionError, match="8-byte words outside"):
        _check_1m(s, snap, q_before)


def test_1m_dep4_harness_refuses_misplaced_writes() -> None:
    """Controls for the 1M harness at the `_ML4_1M_DEP4` cells' 64 heads: a call whose block
    table sends one page to a stale block, a call that reads its rope one table row early, a
    call that ropes 63 of the 64 query heads, a call followed by one stray byte at the pool's
    far end, and a padding replay whose shared slot is then changed in one byte each fail
    `_check_1m` or the one-row comparison, by the check meant for it."""
    spec = dict(phase="context", tokens=64, cached=1015808, heads=64, rope=True, blocks=256,
                owners=63, tokens_per_block=32, row=62, max_pos=1048576,
                table_width=32772)  # fmt: skip
    table = _indexer_table(spec["max_pos"])
    pool = torch.empty((256, 63, 32, TOKEN_BYTES), dtype=torch.uint8, device="cuda")
    gen, gpu = torch.Generator().manual_seed(1), torch.Generator(device="cuda").manual_seed(1)

    def run(**kw):
        s = _step(spec, _requests(spec, gen), gen, gpu, pool, table)
        snap, q_before = pool[s.own], s.q_full.clone()
        if kw.pop("misplace", False):
            kw["block_table"] = s.block_table.clone()
            kw["block_table"][0, 1015808 // 32] = s.block_table[1, 0]
        op(*_args_1m(s, **kw))
        return s, snap, q_before

    with pytest.raises(AssertionError, match="code/scale bytes"):
        _check_1m(*run(misplace=True))
    with pytest.raises(AssertionError):
        _check_1m(*run(table=torch.roll(table, 1, dims=0)))
    with pytest.raises(AssertionError):
        _check_1m(*run(heads=63))
    s, snap, q_before = run()
    stray = max(set(range(256)) - set(s.own.tolist()))
    pool[stray, -1, -1, -1] = _SENTINEL ^ 1
    with pytest.raises(AssertionError, match="8-byte words outside"):
        _check_1m(s, snap, q_before)

    # One live row ahead of 7 padding rows: the shared slot, one byte off one row's write.
    boot = dict(spec, phase="decode", tokens=8, row=1)
    s = _step(boot, [(1048575, 1)] + [(0, 1)] * 7, gen, gpu, pool, table, pad=7)
    snap, q_before = pool[s.own], s.q_full.clone()
    op(*_args_1m(s))
    slot = s.block_table[1, 0].long()
    pool[slot, 1, 0, 100] ^= 1
    replayed = pool[s.own]
    pool[s.own] = snap
    op(s.latent[:2], q_before[:2].clone(), s.pos[:2], s.table, s.req[:2], s.block_table, pool,
       s.row, s.rope)  # fmt: skip
    assert not torch.equal(pool[s.own], replayed), "the one-row comparison is blind"
    pool[s.own] = replayed
    with pytest.raises(AssertionError, match="code/scale bytes"):
        _check_1m(s, snap, q_before)


def test_1m_tep8_harness_refuses_misplaced_writes() -> None:
    """Controls for the 1M harness at the `_ML4_1M_TEP8` cells' 8 heads: a call whose block
    table sends one page to a stale block, a call that reads its rope one table row early, a
    call that ropes a 9th query head (at tep8, the first head the kernel's 16-head block
    masks), and a call followed by one stray byte at the pool's far end each fail
    `_check_1m`, by the check meant for it."""
    spec = dict(phase="context", tokens=64, cached=1015808, heads=8, rope=True, blocks=256,
                owners=63, tokens_per_block=32, row=62, max_pos=1048576,
                table_width=32772)  # fmt: skip
    table = _indexer_table(spec["max_pos"])
    pool = torch.empty((256, 63, 32, TOKEN_BYTES), dtype=torch.uint8, device="cuda")
    gen, gpu = torch.Generator().manual_seed(1), torch.Generator(device="cuda").manual_seed(1)

    def run(**kw):
        s = _step(spec, _requests(spec, gen), gen, gpu, pool, table)
        snap, q_before = pool[s.own], s.q_full.clone()
        if kw.pop("misplace", False):
            kw["block_table"] = s.block_table.clone()
            kw["block_table"][0, 1015808 // 32] = s.block_table[1, 0]
        op(*_args_1m(s, **kw))
        return s, snap, q_before

    with pytest.raises(AssertionError, match="code/scale bytes"):
        _check_1m(*run(misplace=True))
    with pytest.raises(AssertionError):
        _check_1m(*run(table=torch.roll(table, 1, dims=0)))
    with pytest.raises(AssertionError, match="heads past the call's"):
        _check_1m(*run(heads=9))
    s, snap, q_before = run()
    stray = max(set(range(256)) - set(s.own.tolist()))
    pool[stray, -1, -1, -1] = _SENTINEL ^ 1
    with pytest.raises(AssertionError, match="8-byte words outside"):
        _check_1m(s, snap, q_before)


def test_scales_are_powers_of_two_with_the_floor() -> None:
    """A zero tile stores the 1e-4 floor rounded up (2^-13) and zero codes; an exact power of
    two times 448 keeps its own scale."""
    spec = dict(phase="context", tokens=2, cached=0, heads=0, rope=False, blocks=4, owners=1,
                tokens_per_block=32, row=0)  # fmt: skip
    b = _build(spec, seed=3)
    b["latent"][0, :512] = 0
    b["latent"][1, :512] = 0
    b["latent"][1, 0] = 448.0 * 2.0**-5
    op(*_args(b))
    blk0 = int(b["block_table"][0, 0])
    rows = b["pool"][blk0, 0, :2]
    scales = rows[:, 512:528].contiguous().view(torch.float32)
    assert scales[0].tolist() == [2.0**-13] * 4
    assert scales[1, 0].item() == 2.0**-5
    assert int(rows[0, :512].sum()) == 0


def test_is_valid_refuses_a_strided_pool() -> None:
    spec = dict(phase="decode", tokens=4, heads=16, rope=True, blocks=64, owners=4,
                tokens_per_block=32, row=1)  # fmt: skip
    b = _build(spec, seed=4)
    args = list(_args(b))
    args[6] = b["pool"][:, :2]
    with pytest.raises(ValueError, match="contiguous"):
        op.is_valid(*args)
