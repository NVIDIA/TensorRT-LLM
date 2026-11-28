# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: the QSA cold-page codec (`qsa_cold_index_codec`) on pools laid out as a context
server's: 26 K/V buffers of 16,384 B a page and the packed side planes (13 index planes of
32 x 256 B, then the 32 x 12 B position page). A first turn writes requests of every `L % 4` and
page edge in calls that pack several requests and cut groups between calls, over pages whose
junk carries stale markers in every anchor slot; each call marks as the forward does
(`qsa_cold_index_mark`). Every owned page is encoded, the hot pages are overwritten with junk,
and the cold pages are decoded into other page ids. Every byte a kernel reads afterwards -- K/V,
the anchors of pooled groups, the raw keys of the groups a later turn can still pool (the
partial one and the one before it), the first-token coordinates -- must come back bitwise; a
decode without the escaped groups must lose exactly the escaped raw keys (negative control).
Requests of 2 and 3 tokens leave group 0 of their first page unanchored over a stale marker: its
raw keys must still be escaped. The plain fallback (a layout the codec declines) copies every byte
of buffers whose sizes and addresses are not multiples of 8."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_cold_index_codec import (
    MARKER,
    ColdIndexTables,
    cold_layout,
    qsa_cold_index_mark,
)

_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="the codec runs on a CUDA device")

DEV = "cuda"
TPB, CR, NPL = 32, 4, 13
KV_BUF, NVB = 16384, 26
SIDE = NPL * 8192 + 384
#: Every L % 4, page edges, a page ending on a group end before its last slot (the next group's
#: stale marker is cleared by the anchor row), a request spanning several pages, and requests of
#: 2 and 3 tokens (group 0 of the first page partial, its stale marker cleared by row 1).
LENS = [37, 40, 3, 64, 65, 66, 2, 67, 95, 96, 100, 129, 1002]
CHUNK = 23
WIDTH = 40


def _planes(side_pool):
    pages = side_pool.shape[0]
    return [
        side_pool[:, p * 8192 : (p + 1) * 8192].view(torch.bfloat16).view(pages, TPB, 1, 128)
        for p in range(NPL)
    ]


def _plant_markers(side_pool):
    """A stale marker in every anchor slot of every page (the junk of a previous owner)."""
    words = torch.tensor(MARKER, dtype=torch.int32, device=DEV).view(torch.uint8)
    for pl in range(NPL):
        for g in range(8):
            start = pl * 8192 + (4 * g + 3) * 256 + 128
            side_pool[:, start : start + 16] = words


def _turn_one(side_pool, block_table, gen):
    """Raw keys and coordinates for every row, an anchor at every group end, then the marks, in
    calls of CHUNK rows over the requests side by side."""
    rows = [(r, p) for r, n in enumerate(LENS) for p in range(n)]
    planes = _planes(side_pool)
    table = block_table.cpu()
    for start in range(0, len(rows), CHUNK):
        call = rows[start : start + CHUNK]
        for r, p in call:
            page, slot = int(table[r, p // TPB]), p % TPB
            for pl in range(NPL):
                at = pl * 8192 + slot * 256
                side_pool[page, at : at + 256] = torch.randint(
                    0, 256, (256,), generator=gen, device=DEV, dtype=torch.uint8
                )
            at = NPL * 8192 + slot * 12
            side_pool[page, at : at + 12] = torch.randint(
                0, 256, (12,), generator=gen, device=DEV, dtype=torch.uint8
            )
        req = torch.tensor([r for r, _ in call], dtype=torch.int32, device=DEV)
        logical = torch.tensor([p for _, p in call], dtype=torch.int64, device=DEV)
        for plane in planes:
            qsa_cold_index_mark(plane, block_table, req, logical, TPB, CR)


def _live(side_pool, kv_pool, block_table):
    """Per owned page (in table order): every byte a kernel reads, by kind -- `kv` (the K/V
    bytes), `anchors` (each pooled group's), `escaped` (the raw keys of the groups a later turn
    can still pool) and `coords` (the first-token coordinates)."""
    out = []
    table = block_table.cpu()
    for r, n in enumerate(LENS):
        for column in range((n + TPB - 1) // TPB):
            page = int(table[r, column])
            filled = min(n - column * TPB, TPB)
            complete = filled // CR  # groups with an anchor on this page
            parts = dict(kv=[kv_pool[page].clone()], anchors=[], escaped=[], coords=[])
            for pl in range(NPL):
                base = pl * 8192
                for g in range(complete):
                    at = base + (4 * g + 3) * 256
                    parts["anchors"].append(side_pool[page, at : at + 128].clone())
                for e in sorted({complete - 1, complete} & set(range(8))):
                    at = base + 4 * e * 256
                    parts["escaped"].append(side_pool[page, at : at + 768].clone())
            for g in range(8):
                at = NPL * 8192 + 4 * g * 12
                parts["coords"].append(side_pool[page, at : at + 12].clone())
            out.append(parts)
    return out


def _same(got, want, kind) -> list:
    """Per (page, slice) of `kind`: whether it came back bitwise."""
    return [
        torch.equal(a, b) for g, w in zip(got, want) for a, b in zip(g[kind], w[kind], strict=True)
    ]


@pytest.mark.cpu_only
def test_cold_layout_of_the_context_server_page():
    layout = cold_layout(NVB * KV_BUF, NPL)
    assert layout["COLD_BYTES"] == 462848
    assert cold_layout(NVB * KV_BUF + SIDE, 0)["COLD_BYTES"] == NVB * KV_BUF + SIDE


@_GPU
def test_round_trip_restores_every_live_byte():
    gen = torch.Generator(device=DEV).manual_seed(26)
    owned = sum((n + TPB - 1) // TPB for n in LENS)
    pages = 2 * owned + 8
    kv_pool = torch.randint(0, 256, (pages, NVB * KV_BUF), generator=gen, device=DEV).to(
        torch.uint8
    )
    side_pool = torch.randint(0, 256, (pages, SIDE), generator=gen, device=DEV).to(torch.uint8)
    _plant_markers(side_pool)
    ids = torch.randperm(pages, generator=gen, device=DEV).to(torch.int32)
    own = torch.zeros(len(LENS), WIDTH, dtype=torch.int32, device=DEV)
    target = torch.zeros_like(own)
    cursor = 0
    for r, n in enumerate(LENS):
        count = (n + TPB - 1) // TPB
        own[r, :count] = ids[cursor : cursor + count]
        target[r, :count] = ids[owned + cursor : owned + cursor + count]
        cursor += count
    _turn_one(side_pool, own, gen)
    want = _live(side_pool, kv_pool, own)

    tables = ColdIndexTables(
        [(kv_pool.data_ptr() + i * KV_BUF, kv_pool.stride(0), KV_BUF) for i in range(NVB)],
        (side_pool.data_ptr(), side_pool.stride(0)),
        NPL,
        torch.device(DEV),
    )
    cold = torch.zeros(owned, tables.cold_page_bytes, dtype=torch.uint8, device=DEV)
    src = torch.cat([own[r, : (n + TPB - 1) // TPB] for r, n in enumerate(LENS)])
    dst = torch.cat([target[r, : (n + TPB - 1) // TPB] for r, n in enumerate(LENS)])
    slots = torch.arange(owned, dtype=torch.int32, device=DEV)
    tables.encode(torch.stack([slots, src], 1).contiguous(), owned, cold.data_ptr())
    # Every partial page escaped one group; every plane agreed with plane 0 on it.
    torch.cuda.synchronize()
    assert int(tables.counters[1]) == 0

    # The 2- and 3-token requests escape group 0 of their page.
    assert all(len(p["escaped"]) == NPL for n, p in zip(LENS, want) if n < CR)

    for no_escape in (True, False):
        junk = torch.randint(0, 256, side_pool.shape, generator=gen, device=DEV).to(torch.uint8)
        side_pool.copy_(junk)
        _plant_markers(side_pool)
        kv_pool.random_(0, 256, generator=gen)
        tables.decode(
            torch.stack([dst, slots], 1).contiguous(), owned, cold.data_ptr(), no_escape=no_escape
        )
        got = _live(side_pool, kv_pool, target)
        for kind in ("kv", "anchors", "coords"):
            assert all(_same(got, want, kind)), f"{kind} not restored (no_escape={no_escape})"
        escaped = _same(got, want, "escaped")
        if no_escape:
            # Exactly the escaped raw keys are lost without the escape (junk in their place).
            assert escaped and not any(escaped), "the negative control restored escaped raw keys"
        else:
            assert all(escaped), "escaped raw keys not restored"


@_GPU
@pytest.mark.parametrize("offsets, sizes", [((3, 29), (20, 37)), ((0, 24), (24, 40))])
def test_plain_fallback_copies_every_byte(offsets, sizes):
    """The plain concatenation of a declined layout: buffers at any byte offset and of any size
    (the first case is not 8-byte aligned and takes the byte copy) come back whole, and the
    bytes around them in the destination page are left alone."""
    gen = torch.Generator(device=DEV).manual_seed(7)
    pages, page_bytes = 6, 96
    pool = torch.randint(0, 256, (pages, page_bytes), generator=gen, device=DEV).to(torch.uint8)
    base = pool.data_ptr()
    verbatim = [(base + o, page_bytes, n) for o, n in zip(offsets, sizes)]
    tables = ColdIndexTables(verbatim, None, 0, torch.device(DEV))
    assert tables.word == (
        8 if all(o % 8 == 0 and n % 8 == 0 for o, n in zip(offsets, sizes)) else 1
    )
    cold = torch.zeros(2, tables.cold_page_bytes, dtype=torch.uint8, device=DEV)
    src = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32, device=DEV)  # (cold slot, page)
    tables.encode(src, 2, cold.data_ptr())
    before = pool.clone()
    dst = torch.tensor([[4, 0], [5, 1]], dtype=torch.int32, device=DEV)  # (page, cold slot)
    tables.decode(dst, 2, cold.data_ptr())
    torch.cuda.synchronize()
    want = before.clone()
    for page, origin in ((4, 1), (5, 2)):
        for o, n in zip(offsets, sizes):
            want[page, o : o + n] = before[origin, o : o + n]
    assert torch.equal(pool, want)
