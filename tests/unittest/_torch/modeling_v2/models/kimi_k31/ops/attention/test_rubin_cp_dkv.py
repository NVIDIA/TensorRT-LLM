# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""rubin_cp_dkv on the CPU: N ranks write the step's tokens they own at rank-local positions into
their own pools, and one rank's history rows (``hist_src``) and new-token pages of the requests
rebuild the replicated cache; the Top-K read (``rubin_dkv_topk_read``'s reference) then hands the
sparse MLA each selected row.

A pool here is ``[slots * scale, tokens_per_block, row]`` (``scale`` layers per slot), the rows of a
layer's page at ``slot * scale + layer offset``; every rank's slot ids differ (as allocation does),
and the plan takes them from the exchanged page tables.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_cp_dkv import (
    dkv_plan,
    index_slots,
    owner_write,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_dkv_topk_read import (  # noqa: E501
    rubin_dkv_topk_read,
)
from tensorrt_llm._torch.utils import helix_local_len

pytestmark = pytest.mark.cpu_only

TPB, ROW, SCALE, SLOTS = 8, 4, 2, 40


def _dkv(size, rank):
    return SimpleNamespace(
        tokens_per_block=TPB,
        size=size,
        rank=rank,
        local_len=lambda n, r=None: helix_local_len(n, TPB, size, rank if r is None else r),
    )


@pytest.mark.parametrize("size", [2, 4])
def test_owned_writes_and_the_topk_read_rebuild_the_replicated_cache(size):
    torch.manual_seed(0)
    cached, new = [21, 0, 40], [30, 19, 3]  # three context requests; chunk ends inside pages
    total = [c + n for c, n in zip(cached, new)]
    # The replicated latent of every token of every request, and the step's new rows in order.
    full = [torch.randint(1, 255, (t, ROW), dtype=torch.uint8) for t in total]
    step_rows = torch.cat([f[c:] for f, c in zip(full, cached)])
    blocks = [-(-t // (TPB * size)) for t in total]
    # Each rank's slot of each DKV block, all different.
    tables = [
        [[(7 * r + 3 * q + 11 * rid) % SLOTS for q in range(b)] for r in range(size)]
        for rid, b in enumerate(blocks)
    ]
    layer_offset = 1
    pools = torch.zeros(size, SLOTS * SCALE, TPB, ROW, dtype=torch.uint8)
    for rank in range(size):
        led = _dkv(size, rank)
        # History: what earlier steps wrote, the rank's own pages of the cached tokens.
        for rid, (c, f) in enumerate(zip(cached, full)):
            for pos in range(c):
                if (pos // TPB) % size == rank:
                    j = led.local_len(pos)
                    slot = tables[rid][rank][j // TPB]
                    pools[rank, slot * SCALE + layer_offset, j % TPB] = f[pos]
        plan = dkv_plan(led, tables, cached, new, SLOTS * SCALE, SCALE, "cpu")
        # This step's write: the owned rows at rank-local positions (the paged append's view).
        mine = step_rows.index_select(0, plan.owned_rows)
        k = 0
        for rid in range(len(cached)):
            lo = int(plan.cu_cached[rid + 1] - plan.cu_cached[rid])
            n = int(plan.cu_kv[rid + 1] - plan.cu_kv[rid]) - lo
            for j in range(lo, lo + n):
                slot = tables[rid][rank][j // TPB]
                pools[rank, slot * SCALE + layer_offset, j % TPB] = mine[k]
                k += 1
        assert k == mine.shape[0]
    # Every token stored exactly once across the group.
    assert sum(int(_dkv(size, r).local_len(t)) for r in range(size) for t in total) == sum(total)

    rank = size - 1
    plan = dkv_plan(_dkv(size, rank), tables, cached, new, SLOTS * SCALE, SCALE, "cpu")
    tokens = pools.view(-1, ROW)  # the peer mapping as token rows
    # Each history token's row in the mapping, at the layer's offset.
    assert plan.hist_len.tolist() == cached
    assert plan.hist_start.tolist() == [0, cached[0], cached[0] + cached[1]]
    for rid, c in enumerate(cached):
        for pos in range(c):
            row = int(plan.hist_src[plan.hist_start[rid] + pos]) + layer_offset * TPB
            assert torch.equal(tokens[row], full[rid][pos]), (rid, pos)
    # The new tokens get pages from page 1 (page 0 stands for history blocks), from the page of
    # their first token on.
    pages = [-(-t // TPB) - c // TPB for c, t in zip(cached, total)]
    assert plan.pages == 1 + sum(pages)
    assert plan.offsets.shape == (1, 3, 2, plan.table.shape[1])
    kv = torch.zeros(plan.pages * TPB + sum(cached), ROW, dtype=torch.uint8)
    paged = kv[: plan.pages * TPB].view(-1, TPB, ROW)
    for rid, (c, f) in enumerate(zip(cached, full)):
        assert plan.table[rid, : c // TPB].tolist() == [0] * (c // TPB)
        for pos in range(c, total[rid]):
            paged[plan.table[rid, pos // TPB], pos % TPB] = f[pos]  # the append
    # Every position of every request read through the Top-K: one query row per request.
    width = max(total)
    topk = torch.full((3, width), -1, dtype=torch.int32)
    for rid, t in enumerate(total):
        topk[rid, :t] = torch.randperm(t, dtype=torch.int32)
    req = torch.arange(3, dtype=torch.int32)
    new_idx = torch.where(
        topk >= 0,
        plan.table.gather(1, (topk.clamp_min(0) // TPB).long()) * TPB + topk % TPB,
        -1,
    ).to(torch.int32)
    out, idx = rubin_dkv_topk_read.reference(
        kv,
        plan.pages * TPB,
        topk,
        req,
        new_idx,
        plan.hist_start,
        plan.hist_len,
        plan.hist_src,
        tokens,
        layer_offset * TPB,
    )
    for rid, t in enumerate(total):
        for j in range(t):
            pos = int(topk[rid, j])
            assert torch.equal(out[int(idx[rid, j])], full[rid][pos]), (rid, pos)
        assert (idx[rid, t:] == -1).all()


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("needed", [None, (True, False, True)])
def test_owner_writes_reach_the_owners_and_the_readers_copy_what_the_append_wrote(size, needed):
    """Owner writes: every rank's rows of the step (the contiguous cep split) pushed per the plan
    land on each owner as its owned new tokens in step row order (what it appends from the
    all-gather); after the owners' appends, a reader's copies from their pools fill its read
    buffer's head exactly as the append of every new row does, for the requests it reads."""
    torch.manual_seed(1)
    cached, new = [21, 0, 40], [30, 19, 3]
    total = [c + n for c, n in zip(cached, new)]
    full = [torch.randint(1, 255, (t, ROW), dtype=torch.uint8) for t in total]
    step_rows = torch.cat([f[c:] for f, c in zip(full, cached)])
    num = step_rows.shape[0]
    rows = -(-num // size)
    blocks = [-(-t // (TPB * size)) for t in total]
    tables = [
        [[(7 * r + 3 * q + 11 * rid) % SLOTS for q in range(b)] for r in range(size)]
        for rid, b in enumerate(blocks)
    ]
    reads = [True] * 3 if needed is None else list(needed)
    plans = [
        owner_write(_dkv(size, r), tables, cached, new, rows, reads, SLOTS * SCALE, SCALE, "cpu", 4)
        for r in range(size)
    ]
    # The push: rank s's ordered live rows, owner d's block of them at row d0 of d's buffer.
    recv = [torch.zeros(plans[d].recv[d], ROW, dtype=torch.uint8) for d in range(size)]
    for s_, plan in enumerate(plans):
        assert plan.recv == plans[0].recv
        lo, hi = min(s_ * rows, num), min((s_ + 1) * rows, num)
        sent = step_rows[lo:hi].index_select(0, plan.send_order)
        for d, (s0, d0, n) in enumerate(plan.blocks):
            recv[d][d0 : d0 + n] = sent[s0 : s0 + n]
    layer_offset = 1
    pools = torch.zeros(size, SLOTS * SCALE, TPB, ROW, dtype=torch.uint8)
    for d in range(size):
        led = _dkv(size, d)
        for rid, (c, f) in enumerate(zip(cached, full)):  # the history earlier steps wrote
            for pos in range(c):
                if (pos // TPB) % size == d:
                    j = led.local_len(pos)
                    pools[d, tables[rid][d][j // TPB] * SCALE + layer_offset, j % TPB] = f[pos]
        plan = dkv_plan(led, tables, cached, new, SLOTS * SCALE, SCALE, "cpu")
        assert torch.equal(recv[d], step_rows.index_select(0, plan.owned_rows)), d
        k = 0
        for rid in range(len(cached)):
            lo = int(plan.cu_cached[rid + 1] - plan.cu_cached[rid])
            n = int(plan.cu_kv[rid + 1] - plan.cu_kv[rid]) - lo
            for j in range(lo, lo + n):
                pools[d, tables[rid][d][j // TPB] * SCALE + layer_offset, j % TPB] = recv[d][k]
                k += 1
        assert k == recv[d].shape[0]
    tokens = pools.view(-1, ROW)
    for rank, own in enumerate(plans):
        plan = dkv_plan(_dkv(size, rank), tables, cached, new, SLOTS * SCALE, SCALE, "cpu", width=4)
        head = torch.zeros(plan.pages * TPB, ROW, dtype=torch.uint8)
        paged = head.view(-1, TPB, ROW)
        for rid, (c, f) in enumerate(zip(cached, full)):
            for pos in range(c, total[rid]):
                paged[plan.table[rid, pos // TPB], pos % TPB] = f[pos]  # the append
        assert own.new_dst.numel() == sum(n for n, r in zip(new, reads) if r), rank
        assert own.new_pos.tolist() == list(range(own.new_src.numel()))
        assert own.new_count.tolist() == [own.new_src.numel()]
        copied = tokens[own.new_src + layer_offset * TPB]  # rubin_dkv_rows_read's reference
        assert torch.equal(copied, head[own.new_dst]), rank


def test_a_rank_reads_only_the_history_of_requests_it_has_rows_of():
    """Request 1 has no rows on this rank: no history (length 0), page 0 for its history blocks
    and pages only for the blocks of its new tokens."""
    size = 2
    cached, new = [40, 30], [5, 20]
    tables = [[[3, 4, 5], [6, 7, 8]], [[9, 10, 11], [12, 13, 14]]]
    plan = dkv_plan(_dkv(size, 0), tables, cached, new, 100, 1, "cpu", needed=[True, False])
    assert plan.hist_len.tolist() == [40, 0] and plan.hist_start.tolist() == [0, 40]
    assert plan.hist_src.numel() == 40
    # Request 0: tokens 40..44 in page 5 (page 1 of the buffer); request 1: blocks 3..6.
    assert plan.table[0, :6].tolist() == [0, 0, 0, 0, 0, 1]
    assert plan.table[1, :3].tolist() == [0, 0, 0] and plan.table[1, 3:7].tolist() == [2, 3, 4, 5]
    assert plan.pages == 6
    # Token p: owner (p // 8) % 2, that owner's slot of DKV block p // 16, row p % 8.
    want = [
        ((p // TPB) % size * 100 + tables[0][(p // TPB) % size][p // (TPB * size)]) * TPB + p % TPB
        for p in range(40)
    ]
    assert plan.hist_src.tolist() == want
    wide = dkv_plan(
        _dkv(size, 0), tables, cached, new, 100, 1, "cpu", needed=[True, False], width=16
    )
    assert wide.table.shape == (2, 16) and torch.equal(wide.table[:, :7], plan.table)
    nothing = dkv_plan(_dkv(size, 0), tables, [0, 0], new, 100, 1, "cpu")
    assert nothing.hist_src.numel() == 0 and nothing.hist_len.tolist() == [0, 0]


def test_sharded_index_slots_insert_owned_keys_and_read_every_key_from_its_owner():
    """A key at position p: owner (p // tpb) % N, the owner's slot of DKV block p // (tpb N),
    row p % tpb; inserts in this rank's view, full-KV slots in the view of every rank's copy."""
    size, rank, head, slot_bytes, rank_bytes = 2, 1, 16, 1000, 10**6
    cached, new = [10, 0], [9, 5]
    tables = [[[3, 4], [6, 7]], [[9], [12]]]
    slots = index_slots(_dkv(size, rank), tables, cached, new, slot_bytes, rank_bytes, head, "cpu")

    def key(r, p, mirror):
        owner = (p // TPB) % size
        base = tables[r][owner][p // (TPB * size)] * slot_bytes + (
            owner * rank_bytes if mirror else 0
        )
        return base + (p % TPB) * head, base + TPB * head + (p % TPB) * 4

    owned = [
        (r, p)
        for r, (c, n) in enumerate(zip(cached, new))
        for p in range(c, c + n)
        if (p // TPB) % size == rank
    ]
    assert slots.insert_fp8.tolist() == [key(r, p, False)[0] for r, p in owned]
    assert slots.insert_scale.tolist() == [key(r, p, False)[1] for r, p in owned]
    every = [(r, p) for r, (c, n) in enumerate(zip(cached, new)) for p in range(c + n)]
    assert slots.full_fp8.tolist() == [key(r, p, True)[0] for r, p in every]
    assert slots.full_scale.tolist() == [key(r, p, True)[1] for r, p in every]
