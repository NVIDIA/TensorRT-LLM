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
decode without the escaped groups must not (negative control)."""

import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_cold_index_codec import (
    MARKER,
    ColdIndexTables,
    cold_layout,
    qsa_cold_index_mark,
)

assert torch.cuda.is_available(), "the cold index codec requires a CUDA device"

DEV = "cuda"
TPB, CR, NPL = 32, 4, 13
KV_BUF, NVB = 16384, 26
SIDE = NPL * 8192 + 384
#: Every L % 4, page edges, a page ending on a group end before its last slot (the next group's
#: stale marker is cleared by the anchor row), and a request spanning several pages.
LENS = [37, 40, 64, 65, 66, 67, 95, 96, 100, 129, 1002]
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
    """Per owned page (in table order): its K/V bytes and every side-plane byte a kernel reads."""
    out = []
    table = block_table.cpu()
    for r, n in enumerate(LENS):
        for column in range((n + TPB - 1) // TPB):
            page = int(table[r, column])
            filled = min(n - column * TPB, TPB)
            complete = filled // CR  # groups with an anchor on this page
            parts = [kv_pool[page].clone()]
            for pl in range(NPL):
                base = pl * 8192
                for g in range(complete):
                    at = base + (4 * g + 3) * 256
                    parts.append(side_pool[page, at : at + 128].clone())
                for e in {complete - 1, complete} & set(range(8)):
                    at = base + 4 * e * 256
                    parts.append(side_pool[page, at : at + 768].clone())
            for g in range(8):
                at = NPL * 8192 + 4 * g * 12
                parts.append(side_pool[page, at : at + 12].clone())
            out.append(torch.cat(parts))
    return out


def test_cold_layout_of_the_context_server_page():
    layout = cold_layout(NVB * KV_BUF, NPL)
    assert layout["COLD_BYTES"] == 462848
    assert cold_layout(NVB * KV_BUF + SIDE, 0)["COLD_BYTES"] == NVB * KV_BUF + SIDE


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

    for no_escape, expect_equal in ((True, False), (False, True)):
        junk = torch.randint(0, 256, side_pool.shape, generator=gen, device=DEV).to(torch.uint8)
        side_pool.copy_(junk)
        _plant_markers(side_pool)
        kv_pool.random_(0, 256, generator=gen)
        tables.decode(
            torch.stack([dst, slots], 1).contiguous(), owned, cold.data_ptr(), no_escape=no_escape
        )
        got = _live(side_pool, kv_pool, target)
        same = all(torch.equal(a, b) for a, b in zip(got, want))
        assert same is expect_equal, (
            "every live byte restored"
            if expect_equal
            else "the negative control restored the escaped raw keys without the escape"
        )
