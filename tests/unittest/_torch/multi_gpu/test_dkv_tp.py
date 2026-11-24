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
"""A DKV across real ranks: every rank's KV cache manager V2 pool mapped into every rank,
the page tables exchanged, each rank writing only the pages it owns, and the rows one rank's sparse
MLA reads (the history rows its Top-K selects, copied from their owners: ``rubin_dkv_topk_read``),
and the pages it sends as a disaggregated context rank (to every rank of a TP generation server, or
to its partner in a Helix CP generation server), equal to the replicated cache. The sparse MLA's
output over the DKV read equals its output over a replicated cache, also with the query handed in
FP8 (dep16's form), and the indexer's Top-K over sharded index keys equals its Top-K over
replicated ones.

Each rank allocates its slots after a rank-dependent number of others, so the ranks' slot ids for
the same block differ and the read must translate them through the exchanged tables. A page an
owner restores from the host tier in the step that reads it is read after the group's GPU fence
(``Dkv.take_fence``), not before the restore lands. Needs one node with ``WORLD_SIZE`` GPUs and
fabric memory. ``python3 test_dkv_tp.py`` under ``mpirun -n 4`` runs the same body without the pool
executor.
"""

import itertools
import pickle
import sys
import traceback

import cloudpickle
import numpy as np
import pytest
import torch
from mpi4py import MPI

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop_attention import (
    thop_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.fp8_mqa_logits import (
    fp8_mqa_logits,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.gather_k_cache_paged import (
    k31_gather_k_cache_paged,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.indexer_k_cache_scatter_op import (
    indexer_k_cache_scatter_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.indexer_topk_prefill import (
    indexer_topk_prefill,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_cp_dkv import (
    dkv_plan,
    index_mirror_views,
    index_slots,
    peer_tokens,
    read_pointers,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_dkv_topk_read import (
    rubin_dkv_topk_read,
)
from tensorrt_llm._torch.disaggregation.resource.dkv import (
    DkvView,
    dkv_helix_page_ids,
    dkv_page_ids,
    own_page_slots,
)
from tensorrt_llm._torch.distributed import allgather
from tensorrt_llm._torch.pyexecutor.kv_cache.dkv import (
    Dkv,
    gpu_tier_chunk_bytes,
    map_peer_regions,
    resume_copies,
    unmap_peer_regions,
    verify_peer_regions,
)
from tensorrt_llm._torch.utils import helix_local_len
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
    DataRole,
    GpuCacheTierConfig,
    HostCacheTierConfig,
    KVCacheManager,
    KVCacheManagerConfig,
    LayerId,
    TokenId,
)

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)
pytestmark = pytest.mark.threadleak(enabled=False)

TPB = 64
ROW = 576  # one FP8 MLA latent row
PAGE = TPB * ROW
QUOTA = 256 << 20


def _token(rid: int, pos: int) -> torch.Tensor:
    """The replicated latent row of a token: distinct per (request, position)."""
    base = (rid * 131 + pos * 7) % 251
    return (torch.arange(ROW, device="cuda") + base) % 251 + 1


def _tokens(rid: int, positions: torch.Tensor, row: int = ROW) -> torch.Tensor:
    """``_token`` of each position (``row`` bytes each), as uint8 ``[positions, row]``."""
    base = (rid * 131 + positions.long().cuda() * 7) % 251
    return ((torch.arange(row, device="cuda") + base.unsqueeze(1)) % 251 + 1).to(torch.uint8)


def _read(dkv, peers, tables, lengths, positions, row=ROW):
    """The rows at ``positions`` (one list per request, request-local, all before ``lengths``)
    as the cep sparse MLA reads them: one Top-K row per request over history only, read from the
    owners through the mapping (``rubin_dkv_topk_read``). Returns uint8 ``[requests, most
    positions, row]`` (rows past a request's positions undefined)."""
    tpb = dkv.tokens_per_block
    plan = dkv_plan(dkv, tables, lengths, [0] * len(lengths), peers.rows_per_rank, 1, "cuda")
    k = max(len(p) for p in positions)
    topk = torch.full((len(positions), k), -1, dtype=torch.int32)
    for r, p in enumerate(positions):
        topk[r, : len(p)] = torch.as_tensor(list(p), dtype=torch.int32)
    topk = topk.cuda()
    new_rows = plan.pages * tpb
    kv = torch.empty(
        new_rows + min(plan.hist_src.numel(), topk.numel()), row, dtype=torch.uint8, device="cuda"
    )
    idx = rubin_dkv_topk_read(
        kv,
        new_rows,
        topk,
        torch.arange(len(positions), dtype=torch.int32, device="cuda"),
        torch.full_like(topk, -1),
        plan.hist_start,
        plan.hist_len,
        plan.hist_src,
        peer_tokens(peers, row),
        0,
    )
    return kv[idx.clamp_min(0).long()]


def _dkv_body(world: int, rank: int, comm) -> None:
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dkv = Dkv(world, rank, TPB, comm.allgather)
    config = KVCacheManagerConfig(
        tokens_per_block=TPB,
        cache_tiers=[GpuCacheTierConfig(quota=QUOTA)],
        layers=[
            AttentionLayerConfig(
                layer_id=LayerId(0),
                buffers=[BufferConfig(DataRole("key"), PAGE)],
                sliding_window_size=None,
            )
        ],
    )
    manager = KVCacheManager(config)
    stream = torch.cuda.current_stream().cuda_stream
    caches, peers = [], None
    try:
        (group,) = list(manager.pool_group_descs)
        base = int(manager.get_mem_pool_base_address(LayerId(0), DataRole("key")))
        chunk = gpu_tier_chunk_bytes(QUOTA)
        size = -(-int(group.num_slots) * PAGE // chunk) * chunk
        (peers,) = map_peer_regions(dkv, [(base, size)], chunk, PAGE)
        verify_peer_regions(dkv, [(base, size)], [peers], chunk)

        # A rank-dependent number of blocks first, so the ranks' slot ids differ.
        pad = manager.create_kv_cache()
        assert pad.resume(stream) and pad.resize(max(1, 1 + 3 * rank) * TPB)
        caches.append(pad)
        cached, new = [700, 2000, 129], [50, 0, 511]
        tables = []
        for rid, (c, n) in enumerate(zip(cached, new)):
            kv = manager.create_kv_cache()
            caches.append(kv)
            assert kv.resume(stream)
            blocks = -(-(c + n) // dkv.dkv_tokens_per_block)
            assert kv.resize(blocks * TPB)
            slots = [int(s) for s in kv.get_base_page_indices(0)[:blocks]]
            dkv.exchange_pages([((rid, 0), 0, slots)])
            tables.append(dkv.peer_pages((rid, 0)))
        # Every rank's tables agree, and the slot ids differ between ranks.
        assert comm.allgather(repr(tables)) == [repr(tables)] * world
        assert any(len({t[r][0] for r in range(world)}) > 1 for t in tables)

        # Each rank writes the cached tokens it owns at rank-local positions in its own pages.
        own = convert_to_torch_tensor(
            TensorWrapper(base, torch.uint8, [int(group.num_slots) * PAGE])
        ).view(-1, TPB, ROW)
        for rid, c in enumerate(cached):
            for pos in range(c):
                if dkv.owner(pos) == rank:
                    j = dkv.local_len(pos)
                    own[tables[rid][rank][j // TPB], j % TPB] = _token(rid, pos).to(torch.uint8)
        torch.cuda.synchronize()
        comm.Barrier()

        # This rank's sparse MLA read: query rows over each request's history and its new
        # tokens. The new tokens sit in the buffer's head (the append's pages); the history rows
        # the Top-K selects are copied from their owners through the mapping.
        plan = dkv_plan(dkv, tables, cached, new, peers.rows_per_rank, 1, "cuda")
        new_rows = plan.pages * TPB
        gen = torch.Generator().manual_seed(rank)
        topk, req = [], []
        for rid, (c, n) in enumerate(zip(cached, new)):
            for q in range(c + n - min(n, 4), c + n):
                visible = q + 1
                pick = torch.randperm(visible, generator=gen)[:2048].sort().values
                row = torch.full((2048,), -1, dtype=torch.int32)
                row[: pick.numel()] = pick.to(torch.int32)
                topk.append(row)
                req.append(rid)
        topk = torch.stack(topk).cuda()
        req = torch.tensor(req, dtype=torch.int32, device="cuda")
        kv = torch.empty(
            new_rows + min(plan.hist_src.numel(), topk.numel()),
            ROW,
            dtype=torch.uint8,
            device="cuda",
        )
        paged = kv[:new_rows].view(-1, TPB, ROW)
        for rid, (c, n) in enumerate(zip(cached, new)):
            pos = torch.arange(c, c + n, device="cuda")
            paged[plan.table[rid].long()[pos // TPB], pos % TPB] = _tokens(rid, pos)
        new_idx = torch.where(
            topk >= 0,
            plan.table[req.long()].gather(1, (topk.clamp_min(0) // TPB).long()) * TPB + topk % TPB,
            -1,
        ).to(torch.int32)
        idx = rubin_dkv_topk_read(
            kv,
            new_rows,
            topk,
            req,
            new_idx,
            plan.hist_start,
            plan.hist_len,
            plan.hist_src,
            peer_tokens(peers, ROW),
            0,
        )
        torch.cuda.synchronize()
        for j in range(topk.shape[0]):
            ok = topk[j] >= 0
            want = _tokens(int(req[j]), topk[j][ok])
            assert torch.equal(kv[idx[j][ok].long()], want), (rank, j, "Top-K read")
            assert (idx[j][~ok] == -1).all()
        comm.Barrier()

        # What this rank sends as a DKV context rank: the pages it owns, from its own slots;
        # over the group every page goes once (resource/dkv.py, fan-out).
        for rid, c in enumerate(cached):
            pages = -(-c // TPB)
            mine = own_page_slots(tables[rid][rank], world, pages)
            src, dst = dkv_page_ids(DkvView(), world, rank, mine, np.arange(pages))
            sent = own[torch.as_tensor(src, device="cuda")].reshape(-1, TPB, ROW)
            for k, page in enumerate(dst.tolist()):
                rows = range(page * TPB, min((page + 1) * TPB, c))
                want = torch.stack([_token(rid, p) for p in rows]).to(torch.uint8)
                assert torch.equal(sent[k][: len(rows)], want), (rank, rid, page, "sent")
            everyone = comm.allgather(dst.tolist())
            assert sorted(p for d in everyone for p in d) == list(range(pages)), (rank, rid)
        comm.Barrier()

        # What this rank sends a Helix CP generation server of the same size: its partner, Helix
        # rank ``rank``, stores global page ``rank + j * world`` as local page ``j`` -- what a
        # replicated context server's Helix scatter (its table ``[rank::world]``) puts there.
        for rid, c in enumerate(cached):
            pages = -(-c // TPB)
            mine = own_page_slots(tables[rid][rank], world, pages)
            held = list(range(rank, pages, world))
            gen_table = np.arange(len(held), dtype=np.int64)[::-1] + 2  # the partner's own slots
            src, dst = dkv_helix_page_ids(DkvView(), world, rank, mine, gen_table)
            gen = own.new_zeros((len(held) + 2,) + tuple(own.shape[1:]))
            if src.size:
                gen[torch.as_tensor(dst, device="cuda")] = own[torch.as_tensor(src, device="cuda")]
            gen = gen.reshape(len(held) + 2, TPB, ROW)
            for j, page in enumerate(held):
                rows = range(page * TPB, min((page + 1) * TPB, c))
                want = torch.stack([_token(rid, p) for p in rows]).to(torch.uint8)
                assert torch.equal(gen[gen_table[j]][: len(rows)], want), (rank, rid, page, "helix")
            tokens = sum(min(TPB, c - page * TPB) for page in held)
            assert tokens == helix_local_len(c, TPB, world, rank), (rank, rid, "helix_local_len")
        comm.Barrier()

        # Sharded index keys (the cep target's option): each rank inserts the keys it owns in the
        # paged index-key layout, then every rank gathers every key from its owner through the
        # mapping with the indexer's key-gather kernel. The pool slots stand in for index pages.
        head = 128
        fp8_gen = torch.Generator(device="cuda").manual_seed(7)  # the same keys on every rank
        keys = [
            torch.randn(c + n, head, generator=fp8_gen, device="cuda")
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
            for rid, (c, n) in enumerate(zip(cached, new))
        ]
        scales = [
            torch.arange(c + n, dtype=torch.float32, device="cuda") + rid
            for rid, (c, n) in enumerate(zip(cached, new))
        ]
        per_token = head + 4
        flat = own.view(-1)
        pages = flat.numel() // (TPB * per_token)
        index_view = flat[: pages * TPB * per_token].view(pages, TPB, 1, per_token)
        slots = index_slots(
            dkv,
            tables,
            [0] * len(cached),
            [c + n for c, n in zip(cached, new)],
            PAGE,
            peers.stride,
            head,
            "cuda",
        )
        owned = [
            (rid, p)
            for rid, (c, n) in enumerate(zip(cached, new))
            for p in range(c + n)
            if dkv.owner(p) == rank
        ]
        if owned:
            k8 = torch.stack([keys[rid][p] for rid, p in owned])
            ks = torch.stack([scales[rid][p] for rid, p in owned]).view(-1, 1)
            indexer_k_cache_scatter_op(
                k8.view(torch.float8_e4m3fn),
                ks,
                index_view,
                slots.insert_fp8,
                slots.insert_scale,
                len(owned),
            )
        torch.cuda.synchronize()
        comm.Barrier()
        mirror = index_mirror_views(peers, base, {0: index_view})[0]
        total = sum(c + n for c, n in zip(cached, new))
        got8, gots = k31_gather_k_cache_paged(
            mirror, slots.full_fp8, slots.full_scale, 0, total, head
        )
        torch.cuda.synchronize()
        assert torch.equal(got8.view(torch.uint8), torch.cat(keys)), (rank, "index keys")
        assert torch.equal(gots.view(-1), torch.cat(scales)), (rank, "index scales")
        # The indexer's Top-K of query rows of request 2 over its keys: from the sharded keys,
        # as from the replicated ones.
        lo = sum(c + n for c, n in zip(cached[:2], new[:2]))
        hi = lo + cached[2] + new[2]
        rows = 64
        q = torch.randn(rows, 32, head, generator=fp8_gen, device="cuda").to(torch.float8_e4m3fn)
        w = torch.randn(rows, 32, generator=fp8_gen, device="cuda")
        ks = torch.full((rows,), lo, dtype=torch.int32, device="cuda")
        ke = torch.arange(hi - rows + 1, hi + 1, dtype=torch.int32, device="cuda")
        picks = []
        for k8, k_scale in (
            (got8, gots.view(-1)),
            (torch.cat(keys).view(torch.float8_e4m3fn), torch.cat(scales)),
        ):
            logits = fp8_mqa_logits(q, k8, k_scale, w, ks, ke, False)
            out = torch.empty(rows, 2048, dtype=torch.int32, device="cuda")
            indexer_topk_prefill(logits, ks, ke, out, 2048)
            picks.append(out)
        torch.cuda.synchronize()
        assert torch.equal(picks[0], picks[1]), (rank, "indexer Top-K")
        comm.Barrier()
    finally:
        if peers is not None:
            unmap_peer_regions([peers])
        for kv in caches:
            kv.close()
        manager.shutdown()


def _host_tier_body(world: int, rank: int, comm) -> None:
    """A DKV over a GPU + host cache: a request's committed pages leave the GPU independently on
    each rank, and a later request that reuses them gets them back in new slots; the page tables
    the group exchanges then still lead every rank to every page."""
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dkv = Dkv(world, rank, TPB, comm.allgather)
    config = KVCacheManagerConfig(
        tokens_per_block=TPB,
        cache_tiers=[GpuCacheTierConfig(quota=2 << 20), HostCacheTierConfig(quota=64 << 20)],
        layers=[
            AttentionLayerConfig(
                layer_id=LayerId(0),
                buffers=[BufferConfig(DataRole("key"), PAGE)],
                sliding_window_size=None,
            )
        ],
    )
    manager = KVCacheManager(config)
    stream = torch.cuda.current_stream().cuda_stream
    peers, kvs = None, []
    token_ids = itertools.count(1)
    try:
        (group,) = list(manager.pool_group_descs)
        base = int(manager.get_mem_pool_base_address(LayerId(0), DataRole("key")))
        chunk = gpu_tier_chunk_bytes(2 << 20)
        size = -(-int(group.num_slots) * PAGE // chunk) * chunk
        (peers,) = map_peer_regions(dkv, [(base, size)], chunk, PAGE)
        own = convert_to_torch_tensor(
            TensorWrapper(base, torch.uint8, [int(group.num_slots) * PAGE])
        ).view(-1, TPB, ROW)

        # Request A computes 6 DKV blocks (its owned pages written) and commits them.
        blocks = 6
        kv_a = manager.create_kv_cache()
        assert kv_a.resume(stream)
        prompt = []
        for block in range(blocks):
            kv_a.capacity = TPB * (block + 1)
            slot = int(kv_a.get_base_page_indices(0)[block])
            page = block * world + rank
            for row in range(TPB):
                own[slot, row] = _token(0, page * TPB + row).to(torch.uint8)
            torch.cuda.synchronize()
            ids = [TokenId(next(token_ids)) for _ in range(TPB)]
            kv_a.commit(ids)
            prompt += ids
        first_slots = [int(x) for x in kv_a.get_base_page_indices(0)[:blocks]]
        kv_a.close()

        # Churn: other requests fill the GPU tier, so A's committed pages move to the host.
        for _ in range(40):
            kv = manager.create_kv_cache()
            if kv.resume(stream):
                for block in range(4):
                    kv.capacity = TPB * (block + 1)
                    kv.commit([TokenId(10**6 + next(token_ids)) for _ in range(TPB)])
            kv.close()
        torch.cuda.synchronize()

        # Request B reuses A's prompt: every rank agrees on the depth, onboards its pages, and the
        # group exchanges B's (new) slots.
        kv_b = manager.create_kv_cache(input_tokens=list(prompt))
        kvs.append(kv_b)
        matched = dkv.agree(dkv.round_down(int(kv_b.num_committed_tokens) * world))
        assert kv_b.resume(stream)
        torch.cuda.synchronize()
        held = matched // dkv.dkv_tokens_per_block
        slots = [int(x) for x in kv_b.get_base_page_indices(0)[:held]]
        moved = comm.allgather(slots != first_slots[:held])
        dkv.exchange_pages([((1, 0), 0, slots)])
        tables = [dkv.peer_pages((1, 0))]
        got = _read(dkv, peers, tables, [matched], [range(matched)])[0]
        torch.cuda.synchronize()
        want = _tokens(0, torch.arange(matched))
        assert matched == blocks * dkv.dkv_tokens_per_block, (rank, matched)
        assert torch.equal(got, want), (rank, "host tier")
        if rank == 0:
            print(
                f"DKV host tier: {held} blocks reused, slots moved on ranks {moved}",
                flush=True,
            )
        comm.Barrier()
    finally:
        if peers is not None:
            unmap_peer_regions([peers])
        for kv in kvs:
            kv.close()
        manager.shutdown()


SLOW = 1  # the owner whose restore lands late in the restore-race body
SLOW_CYCLES = 2_000_000_000  # ~1 s of GPU sleep on SLOW's stream


def _restore_race_body(world: int, rank: int, comm, fence: bool) -> int:
    """A peer reads pages that their owner restores from the host tier in the same step.

    Request A commits 6 DKV blocks and moves to the host tier on every rank; every GPU slot is then
    overwritten, so only the host holds A's pages. Owner ``SLOW`` sleeps on its stream before more
    requests fill its GPU tier, so the slots that request B's restore evicts become free only after
    the sleep, and B's restore of A's pages on ``SLOW`` lands about a second late. Each rank then
    exchanges B's page tables and reads every row as the sparse MLA read does (an SM gather from
    the owners on its own stream). Returns how many pages this rank read wrong: without the fence
    the other ranks read ``SLOW``'s pages before they land; with it (one GPU all-gather of the
    group when a rank copied pages, ``Dkv.take_fence``), none.
    """
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dkv = Dkv(world, rank, TPB, comm.allgather)
    mapping = Mapping(world_size=world, rank=rank, tp_size=world)
    config = KVCacheManagerConfig(
        tokens_per_block=TPB,
        cache_tiers=[GpuCacheTierConfig(quota=2 << 20), HostCacheTierConfig(quota=64 << 20)],
        layers=[
            AttentionLayerConfig(
                layer_id=LayerId(0),
                buffers=[BufferConfig(DataRole("key"), PAGE)],
                sliding_window_size=None,
            )
        ],
    )
    manager = KVCacheManager(config)
    stream = torch.cuda.current_stream().cuda_stream
    peers, kvs = None, []
    token_ids = itertools.count(1)
    try:
        (group,) = list(manager.pool_group_descs)
        num_slots = int(group.num_slots)
        base = int(manager.get_mem_pool_base_address(LayerId(0), DataRole("key")))
        chunk = gpu_tier_chunk_bytes(2 << 20)
        size = -(-num_slots * PAGE // chunk) * chunk
        (peers,) = map_peer_regions(dkv, [(base, size)], chunk, PAGE)
        own = convert_to_torch_tensor(TensorWrapper(base, torch.uint8, [num_slots * PAGE])).view(
            -1, TPB, ROW
        )

        def churn(requests):
            for _ in range(requests):
                kv = manager.create_kv_cache()
                if kv.resume(stream):
                    for block in range(4):
                        kv.capacity = TPB * (block + 1)
                        kv.commit([TokenId(10**6 + next(token_ids)) for _ in range(TPB)])
                kv.close()

        blocks = 6
        kv_a = manager.create_kv_cache()
        assert kv_a.resume(stream)
        prompt = []
        for block in range(blocks):
            kv_a.capacity = TPB * (block + 1)
            slot = int(kv_a.get_base_page_indices(0)[block])
            page = block * world + rank
            for row in range(TPB):
                own[slot, row] = _token(0, page * TPB + row).to(torch.uint8)
            torch.cuda.synchronize()
            ids = [TokenId(next(token_ids)) for _ in range(TPB)]
            kv_a.commit(ids)
            prompt += ids
        kv_a.close()
        churn(40)  # A's pages move to the host
        torch.cuda.synchronize()
        own.fill_(0xEE)  # no GPU slot holds A's rows any more
        torch.cuda.synchronize()
        comm.Barrier()

        if rank == SLOW:
            torch.cuda._sleep(SLOW_CYCLES)
        churn(-(-num_slots // 4) + 2)  # the GPU tier again, on SLOW after its sleep

        kv_b = manager.create_kv_cache(input_tokens=list(prompt))
        kvs.append(kv_b)
        matched = dkv.agree(dkv.round_down(int(kv_b.num_committed_tokens) * world))
        assert matched == blocks * dkv.dkv_tokens_per_block, (rank, matched)
        assert kv_b.resume(stream)
        copied = resume_copies(kv_b, fresh=True)
        assert copied, (rank, list(kv_b.cached_tokens_by_level))
        slots = [int(x) for x in kv_b.get_base_page_indices(0)[:blocks]]
        dkv.exchange_pages([((1, 0), 0, slots)], copied=copied)
        if dkv.take_fence() and fence:
            allgather(torch.zeros(1, dtype=torch.int32, device="cuda"), mapping)
        got = _read(dkv, peers, [dkv.peer_pages((1, 0))], [matched], [range(matched)])[0]
        torch.cuda.synchronize()
        want = _tokens(0, torch.arange(matched))
        wrong = (got != want).view(-1, TPB, ROW).flatten(1).any(1)
        comm.Barrier()
        return int(wrong.sum())
    finally:
        torch.cuda.synchronize()
        if peers is not None:
            unmap_peer_regions([peers])
        for kv in kvs:
            kv.close()
        manager.shutdown()


LATENT = 576  # an MLA latent row: 512 compressed KV + 64 RoPE
KV_LORA = 512
ROPE = 64
HEADS, PAD_HEADS = 96, 128
TOPK = 2048


def _mla_call(q, q_pe, quant_q, latent_cache, pool, offsets, indices, kv_len, rows, cos_sin):
    """The cep target's sparse MLA context call (``_mla_context``) of one request's last ``rows``
    query rows over ``pool`` (pages ``offsets``), its arguments as the target passes them."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__cep import (  # noqa: E501
        modeling as cep,
    )

    pointers, mapping = read_pointers(pool, 1)
    dev = "cuda"
    return thop_attention(
        q=q,
        output=torch.empty(rows, PAD_HEADS * KV_LORA, dtype=torch.bfloat16, device=dev),
        workspace_=torch.empty(0, dtype=torch.int8, device=dev),
        sequence_length=torch.tensor([kv_len], dtype=torch.int32, device=dev),
        host_past_key_value_lengths=torch.tensor([kv_len], dtype=torch.int32),
        host_total_kv_lens=torch.tensor([kv_len, 0], dtype=torch.int32),
        context_lengths=torch.tensor([rows], dtype=torch.int32, device=dev),
        host_context_lengths=torch.tensor([rows], dtype=torch.int32),
        host_request_types=torch.tensor([0], dtype=torch.int32),
        max_context_q_len_override=None,
        kv_cache_block_offsets=offsets,
        host_kv_cache_pool_pointers=pointers,
        host_kv_cache_pool_mapping=mapping,
        cache_indirection=None,
        latent_cache=latent_cache,
        q_pe=q_pe,
        block_ids_per_seq=None,
        local_layer_idx=0,
        tokens_per_block=TPB,
        max_num_requests=8,
        max_num_sequences=8,
        max_context_length=cos_sin.numel() // (2 * ROPE),
        max_seq_len=cos_sin.numel() // (2 * ROPE),
        attention_window_size=cos_sin.numel() // (2 * ROPE),
        beam_width=1,
        rotary_cos_sin=cos_sin,
        rope_max_positions=cos_sin.numel() // (2 * ROPE),
        rope_original_max_positions=cos_sin.numel() // (2 * ROPE),
        attention_input_type=1,
        helix_position_offsets=None,
        helix_is_inactive_rank=None,
        softmax_stats_tensor=None,
        is_spec_decoding_enabled=False,
        use_spec_decoding=False,
        is_spec_dec_tree=False,
        spec_decoding_generation_lengths=None,
        spec_decoding_position_offsets_for_cpp=None,
        spec_decoding_packed_mask=None,
        spec_decoding_bl_tree_mask_offset=None,
        spec_decoding_bl_tree_mask=None,
        spec_bl_tree_first_sparse_mask_offset_kv=None,
        sparse_attn_indices=indices,
        num_sparse_topk=TOPK,
        cu_q_seqlens=torch.tensor([0, rows], dtype=torch.int32, device=dev),
        cu_kv_seqlens=torch.tensor([0, kv_len], dtype=torch.int32, device=dev),
        fmha_scheduler_counter=None,
        mla_bmm1_scale=None,
        mla_bmm2_scale=None,
        quant_q_buffer=quant_q,
        quant_scale_qkv=None if quant_q is None else torch.ones(1, device=dev),
        flash_mla_tile_scheduler_metadata=None,
        flash_mla_num_splits=None,
        num_contexts=1,
        num_ctx_tokens=rows,
        trtllm_gen_jit_warmup=False,
        is_cross=False,
        **cep._MLA_ATTN_PAD128,
    )


def _attention_body(world: int, rank: int, comm) -> None:
    """The cep sparse MLA of a rank's rows over the DKV read equals the same call over a
    replicated cache, bit for bit: (a) BF16 query, replicated pool, the pool rows of the Top-K;
    (b) BF16 query, the DKV read buffer (new tokens appended in its head, the selected history
    rows copied from the owners); (c) the query in FP8 (dep16's ``mla_ctx_fused_q_fp8`` form: its
    RoPE columns and the FMHA's scales from the op's RoPE kernel, which appends the rank's own rows
    again) over the DKV read buffer."""
    torch.cuda.set_device(rank % torch.cuda.device_count())
    page = TPB * LATENT
    dkv = Dkv(world, rank, TPB, comm.allgather)
    config = KVCacheManagerConfig(
        tokens_per_block=TPB,
        cache_tiers=[GpuCacheTierConfig(quota=QUOTA)],
        layers=[
            AttentionLayerConfig(
                layer_id=LayerId(0),
                buffers=[BufferConfig(DataRole("key"), page)],
                sliding_window_size=None,
            )
        ],
    )
    manager = KVCacheManager(config)
    stream = torch.cuda.current_stream().cuda_stream
    caches, peers = [], None
    try:
        (group,) = list(manager.pool_group_descs)
        base = int(manager.get_mem_pool_base_address(LayerId(0), DataRole("key")))
        chunk = gpu_tier_chunk_bytes(QUOTA)
        size = -(-int(group.num_slots) * page // chunk) * chunk
        (peers,) = map_peer_regions(dkv, [(base, size)], chunk, page)
        pad = manager.create_kv_cache()
        assert pad.resume(stream) and pad.resize(max(1, 1 + 2 * rank) * TPB)
        caches.append(pad)
        cached, new = [3000, 700], [300, 64]
        total = [c + n for c, n in zip(cached, new)]
        tables = []
        for rid, t in enumerate(total):
            kv = manager.create_kv_cache()
            caches.append(kv)
            assert kv.resume(stream)
            blocks = -(-t // dkv.dkv_tokens_per_block)
            assert kv.resize(blocks * TPB)
            dkv.exchange_pages(
                [((rid, 0), 0, [int(x) for x in kv.get_base_page_indices(0)[:blocks]])]
            )
            tables.append(dkv.peer_pages((rid, 0)))
        # Every token's latent (BF16, the same on every rank) and its FP8 cache bytes.
        gen = torch.Generator(device="cuda").manual_seed(11)
        latent = [
            torch.randn(t, LATENT, generator=gen, device="cuda").to(torch.bfloat16) for t in total
        ]
        fp8 = [x.float().to(torch.float8_e4m3fn).view(torch.uint8) for x in latent]
        own = convert_to_torch_tensor(
            TensorWrapper(base, torch.uint8, [int(group.num_slots) * page])
        ).view(-1, TPB, LATENT)
        for rid, c in enumerate(cached):  # the owned history pages
            for g in range(rank, -(-c // TPB), world):
                n = min(TPB, c - g * TPB)
                own[tables[rid][rank][g // world], :n] = fp8[rid][g * TPB : g * TPB + n]
        torch.cuda.synchronize()
        comm.Barrier()

        # The replicated cache: every token of every request, pages in order.
        first = [0] + [sum(-(-t // TPB) for t in total[:r]) for r in range(1, len(total))]
        rep = torch.zeros(sum(-(-t // TPB) for t in total) + 1, TPB, LATENT, dtype=torch.uint8)
        rep = rep.cuda()
        for rid, t in enumerate(total):
            pages = -(-t // TPB)
            rows_ = torch.zeros(pages * TPB, LATENT, dtype=torch.uint8, device="cuda")
            rows_[:t] = fp8[rid]
            rep[first[rid] : first[rid] + pages] = rows_.view(pages, TPB, LATENT)
        # This rank's rows: the last ``rows`` new tokens of one request.
        rid = rank % 2
        c, n = cached[rid], new[rid]
        rows = min(n, 40 + 8 * rank)
        positions = torch.arange(c + n - rows, c + n)
        pick = torch.Generator().manual_seed(100 + rank)
        topk = torch.full((rows, TOPK), -1, dtype=torch.int32)
        for j, q in enumerate(positions.tolist()):
            sel = torch.randperm(q + 1, generator=pick)[:TOPK].sort().values
            topk[j, : sel.numel()] = sel.to(torch.int32)
        topk = topk.cuda()
        req = torch.full((rows,), rid, dtype=torch.int32, device="cuda")
        rep_table = torch.full(
            (len(total), max(-(-t // TPB) for t in total)), -1, dtype=torch.int32
        )
        for r, t in enumerate(total):
            rep_table[r, : -(-t // TPB)] = torch.arange(first[r], first[r] + -(-t // TPB))
        rep_table = rep_table.cuda()

        def remap(table, top):
            pages_ = table[req.long()].gather(1, (top.clamp_min(0) // TPB).long())
            return torch.where(top >= 0, pages_ * TPB + top % TPB, -1).to(torch.int32)

        idx_rep = remap(rep_table, topk)
        rep_offsets = rep_table[rid : rid + 1].view(1, 1, 1, -1).expand(1, 1, 2, -1).contiguous()

        # The DKV read, as the target's _mla_read builds it.
        plan = dkv_plan(
            dkv,
            tables,
            cached,
            new,
            peers.rows_per_rank,
            1,
            "cuda",
            needed=[r == rid for r in (0, 1)],
        )
        new_rows = plan.pages * TPB
        buf = torch.empty(
            new_rows + min(plan.hist_src.numel(), topk.numel()),
            LATENT,
            dtype=torch.uint8,
            device="cuda",
        )
        paged = buf[:new_rows].view(-1, TPB, LATENT)
        for r, (cr, nr) in enumerate(zip(cached, new)):
            pos = torch.arange(cr, cr + nr, device="cuda")
            paged[plan.table[r].long()[pos // TPB], pos % TPB] = fp8[r][cr : cr + nr]
        idx_dkv = rubin_dkv_topk_read(
            buf,
            new_rows,
            topk,
            req,
            remap(plan.table, topk),
            plan.hist_start,
            plan.hist_len,
            plan.hist_src,
            peer_tokens(peers, LATENT),
            0,
        )
        torch.cuda.synchronize()
        ok = topk >= 0
        assert torch.equal(buf[idx_dkv[ok].long()], rep.view(-1, LATENT)[idx_rep[ok].long()]), (
            rank,
            "read rows",
        )
        assert torch.equal(idx_dkv[~ok], idx_rep[~ok])
        dkv_offsets = plan.offsets.index_select(1, torch.tensor([rid], device="cuda"))

        # The query: absorbed latent columns and RoPE columns of 96 heads, zero padding heads.
        fused = torch.zeros(rows, PAD_HEADS, LATENT, dtype=torch.bfloat16, device="cuda")
        fused[:, :HEADS] = torch.randn(rows, HEADS, LATENT, generator=gen, device="cuda").to(
            torch.bfloat16
        )
        q_pe = fused[..., KV_LORA:].contiguous()
        cos_sin = torch.zeros(1, (c + n + 64) * ROPE * 2, dtype=torch.float32, device="cuda")
        cos_sin.view(-1, 2)[:, 0] = 1.0  # the target's identity RoPE table
        out_a = _mla_call(
            fused.view(rows, -1), q_pe, None, None, rep, rep_offsets, idx_rep, c + n, rows, cos_sin
        )
        out_b = _mla_call(
            fused.view(rows, -1), q_pe, None, None, buf, dkv_offsets, idx_dkv, c + n, rows, cos_sin
        )
        quant_q = torch.full((rows, PAD_HEADS, LATENT), 0x5A, dtype=torch.uint8, device="cuda")
        quant_q = quant_q.view(torch.float8_e4m3fn)
        quant_q[..., :KV_LORA] = fused[..., :KV_LORA].float().clamp(-448, 448).to(quant_q.dtype)
        stand_in = fused.new_empty(1, PAD_HEADS * LATENT).expand(rows, -1)
        own_rows = latent[rid][c + n - rows : c + n].contiguous()
        out_c = _mla_call(
            stand_in, q_pe, quant_q, own_rows, buf, dkv_offsets, idx_dkv, c + n, rows, cos_sin
        )
        torch.cuda.synchronize()
        assert torch.equal(out_a, out_b), (rank, "attention over the DKV read")
        assert torch.equal(out_a, out_c), (rank, "attention with the FP8 query")
        assert torch.equal(buf[idx_dkv[ok].long()], rep.view(-1, LATENT)[idx_rep[ok].long()]), (
            rank,
            "the op's own-row append",
        )
        if rank == 0:
            print(
                f"DKV sparse MLA: {rows} rows equal over the read and with the FP8 query",
                flush=True,
            )
        comm.Barrier()
    finally:
        if peers is not None:
            unmap_peer_regions([peers])
        for kv in caches:
            kv.close()
        manager.shutdown()


PT = 16  # tokens of a physical page in the partial-reuse body
PROW = 64  # bytes of a token's row there


def _prow(rid: int, pos: int) -> torch.Tensor:
    base = (rid * 37 + pos * 11) % 251
    return ((torch.arange(PROW, device="cuda") + base) % 251 + 1).to(torch.uint8)


def _partial_body(world: int, rank: int, comm) -> None:
    """Partial reuse under a DKV: a match may end inside a DKV block, inside a page, on a page
    boundary or on a DKV-block boundary. Every rank's resume copies its own page of the last
    DKV block into a private slot (the native partial-block copy); the owner of the page the
    match ends in then writes the recomputed rows there, and the shared page keeps the old ones."""
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dkv = Dkv(world, rank, PT, comm.allgather)
    lpb = dkv.dkv_tokens_per_block
    page = PT * PROW
    config = KVCacheManagerConfig(
        tokens_per_block=lpb,
        cache_tiers=[GpuCacheTierConfig(quota=QUOTA)],
        layers=[
            AttentionLayerConfig(
                layer_id=LayerId(0),
                buffers=[BufferConfig(DataRole("key"), page)],
                sliding_window_size=None,
            )
        ],
        enable_partial_reuse=True,
    )
    manager = KVCacheManager(config)
    stream = torch.cuda.current_stream().cuda_stream
    peers, kvs = None, []
    try:
        (group,) = list(manager.pool_group_descs)
        base = int(manager.get_mem_pool_base_address(LayerId(0), DataRole("key")))
        chunk = gpu_tier_chunk_bytes(QUOTA)
        size = -(-int(group.num_slots) * page // chunk) * chunk
        (peers,) = map_peer_regions(dkv, [(base, size)], chunk, page)
        own = convert_to_torch_tensor(
            TensorWrapper(base, torch.uint8, [int(group.num_slots) * page])
        ).view(-1, PT, PROW)

        def write(kv, rid, lo, hi):
            """This rank's rows of positions [lo, hi) of request ``rid`` in its pages of ``kv``."""
            slots = [int(x) for x in kv.get_base_page_indices(0)[: -(-hi // lpb)]]
            for pos in range(lo, hi):
                if dkv.owner(pos) == rank:
                    own[slots[pos // lpb], pos % PT] = _prow(rid, pos)
            torch.cuda.synchronize()

        def read(kv, key, length):
            """Every row of [0, length) through the exchanged tables."""
            slots = [int(x) for x in kv.get_base_page_indices(0)[: -(-length // lpb)]]
            dkv.exchange_pages([(key, 0, slots)])
            got = _read(dkv, peers, [dkv.peer_pages(key)], [length], [range(length)], PROW)[0]
            torch.cuda.synchronize()
            return got

        # Request A: 300 tokens (4 whole DKV blocks of 64 and 44 tokens of the fifth), committed.
        length_a = 300
        tokens_a = [TokenId(1000 + i) for i in range(length_a)]
        kv_a = manager.create_kv_cache()
        assert kv_a.resume(stream)
        kv_a.capacity = length_a
        write(kv_a, 0, 0, length_a)
        kv_a.commit(tokens_a)
        kv_a.close()
        want_a = torch.stack([_prow(0, p) for p in range(length_a)])

        # B diverges from A at a match end inside a page, on a page boundary inside a DKV block,
        # on a DKV-block boundary, and at A's partial end; it recomputes 40 tokens from there.
        for rid, match in enumerate((149, 160, 192, 300), start=1):
            tail = 40
            prompt = tokens_a[:match] + [TokenId(10**6 * rid + i) for i in range(tail)]
            kv_b = manager.create_kv_cache(input_tokens=prompt)
            agreed = dkv.agree(int(kv_b.num_committed_tokens))
            assert agreed == match, (rank, rid, int(kv_b.num_committed_tokens), match)
            assert kv_b.resume(stream)
            kv_b.capacity = match + tail
            write(kv_b, rid, match, match + tail)
            comm.Barrier()
            got = read(kv_b, (rid, 0), match + tail)
            want = torch.cat(
                [want_a[:match], torch.stack([_prow(rid, p) for p in range(match, match + tail)])]
            )
            assert torch.equal(got, want), (rank, rid, match, "partial reuse")
            comm.Barrier()
            kv_b.close()

        # A's committed pages kept their rows: the recomputed tails went to private copies.
        kv_c = manager.create_kv_cache(input_tokens=tokens_a + [TokenId(5)])
        kvs.append(kv_c)
        assert dkv.agree(int(kv_c.num_committed_tokens)) == length_a
        assert kv_c.resume(stream)
        assert torch.equal(read(kv_c, (9, 0), length_a), want_a), (rank, "shared pages")
        if rank == 0:
            print("DKV partial reuse: matches 149 / 160 / 192 / 300 resumed exactly", flush=True)
        comm.Barrier()
    finally:
        if peers is not None:
            unmap_peer_regions([peers])
        for kv in kvs:
            kv.close()
        manager.shutdown()


def _run(world: int) -> bool:
    comm, rank = MPI.COMM_WORLD, MPI.COMM_WORLD.Get_rank()
    try:
        _dkv_body(world, rank, comm)
        _attention_body(world, rank, comm)
        _host_tier_body(world, rank, comm)
        _partial_body(world, rank, comm)
        # Without the fence the race is timing-dependent, so it is reported, not asserted.
        unfenced = comm.allgather(_restore_race_body(world, rank, comm, fence=False))
        fenced = comm.allgather(_restore_race_body(world, rank, comm, fence=True))
        if rank == 0:
            print(
                f"DKV restore race: pages read wrong per rank, unfenced {unfenced}, "
                f"fenced {fenced}",
                flush=True,
            )
        assert fenced == [0] * world, fenced
    except Exception:
        traceback.print_exc()
        raise
    return True


@pytest.mark.parametrize("world_size", [4], ids=lambda n: f"tp{n}")
def test_every_rank_reads_every_rank_s_rows(world_size: int) -> None:
    from mpi4py.futures import MPIPoolExecutor

    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs, has {torch.cuda.device_count()}")
    with MPIPoolExecutor(max_workers=world_size) as executor:
        assert all(executor.map(_run, [world_size] * world_size))


if __name__ == "__main__":
    _run(MPI.COMM_WORLD.Get_size())
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("DKV TP test passed", flush=True)
