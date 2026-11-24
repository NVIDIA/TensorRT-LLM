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
"""A DKV context rank's transfer layout, the pages it sends, and its pairings.

The layout and pages live in resource/dkv.py, the pairings in PeerRegistrar.dkv_fan_out
(a plain TP generation server) and PeerRegistrar.dkv_helix (a Helix CP generation server whose CP
size is a multiple or a divisor of the DKV group's).
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.base.region import MemRegionGroup, SpecRegion
from tensorrt_llm._torch.disaggregation.native.mixers.attention.spec import AttentionInfo
from tensorrt_llm._torch.disaggregation.native.mixers.ssm.peer import MambaPolicy
from tensorrt_llm._torch.disaggregation.native.peer import PeerRegistrar
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.resource.dkv import (
    DkvLayout,
    DkvView,
    build_dkv_layout,
    dkv_helix_page_ids,
    dkv_helix_shares_pages,
    dkv_page_ids,
    merge_runs,
    own_page_slots,
    split_expanded_dkv_views,
)
from tensorrt_llm._torch.disaggregation.resource.page import (
    BUFFER_ENTRY_DTYPE,
    AttentionLayerGroup,
    KVCachePageTable,
    LocalLayer,
    MapperKind,
    PhysicalPool,
    PhysicalPoolGroup,
    PoolView,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.dkv import Dkv, PeerRegion
from tensorrt_llm._torch.utils import helix_local_len

pytestmark = pytest.mark.cpu_only

N = 4
RANK = 1


def _own_blocks():
    # This rank's slot of DKV block b is 10 * RANK + b; it holds 3 blocks.
    return [10 * RANK + b for b in range(3)]


def test_own_page_slots_name_this_ranks_slot_of_each_pages_block():
    own = own_page_slots(_own_blocks(), N, 10)
    np.testing.assert_array_equal(own, 10 * RANK + np.arange(10) // N)
    own = own_page_slots(_own_blocks()[:2], N, 10)  # the third block is not allocated yet
    assert (own[8:] == -1).all() and (own[:8] >= 0).all()


def test_a_rank_sends_the_pages_it_owns_from_its_own_slots():
    own = own_page_slots(_own_blocks(), N, 10)
    dst = np.arange(100, 110)
    dst[5] = -1  # the receiver holds no page 5
    src, out = dkv_page_ids(DkvView(), N, RANK, own, dst, dst_start_page=2)
    # Pages 1, 5, 9 are this rank's (g % 4 == 1); page 1 is before the start, page 5 not held.
    np.testing.assert_array_equal(src, own[[9]])
    np.testing.assert_array_equal(out, dst[[9]])


def test_an_expanded_view_sends_this_ranks_share_of_its_sub_pages():
    own = own_page_slots(_own_blocks(), N, 10)
    src, out = dkv_page_ids(DkvView(pages_per_slot=8), N, RANK, own, np.arange(100, 110))
    pages = np.array([1, 5, 9])
    np.testing.assert_array_equal(src, (10 * RANK + pages // N) * 8 + pages % N)
    np.testing.assert_array_equal(out, 100 + pages)


def test_every_page_goes_once_from_the_group():
    sent = []
    for rank in range(N):
        own = own_page_slots([10 * rank + b for b in range(3)], N, 10)
        _, out = dkv_page_ids(DkvView(), N, rank, own, np.arange(10))
        sent += out.tolist()
    assert sorted(sent) == list(range(10))


def test_dkv_page_ids_refuse_tables_of_different_prompts():
    own = own_page_slots(_own_blocks(), N, 10)
    with pytest.raises(ValueError, match="differ in length"):
        dkv_page_ids(DkvView(pages_per_slot=8), N, RANK, own, np.arange(9))


def test_helix_stores_page_g_on_cp_rank_g_mod_n_like_the_dkv():
    """Helix CP rank r's tokens (helix_local_len) are the global pages g % N == r.

    The partial last page included: the pages DKV rank r owns.
    """
    for prompt in (1, 63, 64, 65, 255, 256, 257, 700, 1023):
        pages = -(-prompt // 64)
        for rank in range(N):
            owned = [g for g in range(pages) if g % N == rank]
            tokens = sum(min(64, prompt - 64 * g) for g in owned)
            assert helix_local_len(prompt, 64, N, rank) == tokens, (prompt, rank)


def test_a_rank_sends_its_helix_partner_its_pages_in_local_order():
    own = own_page_slots(_own_blocks(), N, 10)
    # Helix rank 1 holds global pages 1, 5, 9 as its local pages 0, 1, 2.
    dst = np.array([200, 201, 202])
    src, out = dkv_helix_page_ids(DkvView(), N, RANK, own, dst)
    np.testing.assert_array_equal(src, own[[1, 5, 9]])
    np.testing.assert_array_equal(out, dst)
    # An expanded view: this rank's sub-page of each of its slots.
    src, out = dkv_helix_page_ids(DkvView(pages_per_slot=8), N, RANK, own, dst)
    np.testing.assert_array_equal(src, (10 * RANK + np.arange(3)) * 8 + RANK)
    # A page the receiver does not hold, and pages before the start (global), move nowhere.
    dst[1] = -1
    src, out = dkv_helix_page_ids(DkvView(), N, RANK, own, dst, dst_start_page=2)
    np.testing.assert_array_equal(src, own[[9]])
    np.testing.assert_array_equal(out, [202])


def test_the_helix_pages_of_the_group_are_what_a_replicated_sender_scatters():
    """The DKV ranks fill the Helix local slots a replicated sender's Helix scatter fills.

    A replicated context rank sends Helix rank r its table [r::N] in order (transfer.py).
    """
    pages = 11
    for rank in range(N):
        own = own_page_slots([10 * rank + b for b in range(3)], N, pages)
        dst = 300 + np.arange(len(range(rank, pages, N)))
        src, out = dkv_helix_page_ids(DkvView(), N, rank, own, dst)
        replicated = np.arange(pages)[rank::N]  # the global pages, in the Helix table's order
        np.testing.assert_array_equal(out, dst)
        np.testing.assert_array_equal(src, own[replicated])
    with pytest.raises(ValueError, match="same prompt"):
        dkv_helix_page_ids(DkvView(), N, 0, own_page_slots([0, 1, 2], N, 10), np.arange(2))


def _dkv_rank_slots(size, rank, pages):
    # DKV rank r's slot of DKV block b is 100 * r + b.
    return own_page_slots([100 * rank + b for b in range(-(-pages // size))], size, pages)


def test_a_helix_group_of_the_dkv_size_takes_each_ranks_pages_in_order():
    """With a Helix CP of N, DKV rank r sends Helix rank r its pages g = r + j * N, j in order.

    Holes on either side and pages before the start move nowhere. The Helix size and rank default
    to the DKV's own.
    """
    rng = np.random.default_rng(0)
    for pages in (1, 10, 11, 64, 333):
        for rank in range(N):
            own = _dkv_rank_slots(N, rank, pages)
            own[rng.random(pages) < 0.2] = -1
            held = np.arange(rank, pages, N)
            dst = 900 + np.arange(held.size)
            dst[rng.random(held.size) < 0.2] = -1
            for start in (0, 3, pages // 2):
                keep = (own[held] >= 0) & (dst >= 0) & (held >= start)
                for view in (DkvView(), DkvView(pages_per_slot=8)):
                    want = own[held[keep]] if view.sharded else own[held[keep]] * 8 + rank
                    for got in (
                        dkv_helix_page_ids(view, N, rank, own, dst, start),
                        dkv_helix_page_ids(
                            view, N, rank, own, dst, start, helix_size=N, helix_rank=rank
                        ),
                    ):
                        np.testing.assert_array_equal(got[0], want)
                        np.testing.assert_array_equal(got[1], dst[keep])


def test_a_dkv_rank_of_eight_feeds_two_helix_ranks_of_sixteen():
    """cep8 -> Helix CP16: DKV rank 3 owns the pages g % 8 == 3, which Helix ranks 3 and 11 store.

    Helix rank 3 stores global pages 3, 19, 35 as its local pages 0, 1, 2, and Helix rank 11
    stores 11, 27 as 0, 1: all of them rank 3's. Helix rank 4 stores none of them.
    """
    own = _dkv_rank_slots(8, 3, 40)
    src, out = dkv_helix_page_ids(
        DkvView(), 8, 3, own, np.array([200, 201, 202]), helix_size=16, helix_rank=3
    )
    np.testing.assert_array_equal(src, own[[3, 19, 35]])
    np.testing.assert_array_equal(out, [200, 201, 202])
    src, out = dkv_helix_page_ids(
        DkvView(), 8, 3, own, np.array([300, 301]), helix_size=16, helix_rank=11
    )
    np.testing.assert_array_equal(src, own[[11, 27]])
    np.testing.assert_array_equal(out, [300, 301])
    # An expanded view: sub-page g % 8 == 3 of each slot.
    src, _ = dkv_helix_page_ids(
        DkvView(pages_per_slot=16), 8, 3, own, np.array([300, 301]), helix_size=16, helix_rank=11
    )
    np.testing.assert_array_equal(src, own[[11, 27]] * 16 + 3)
    src, out = dkv_helix_page_ids(
        DkvView(), 8, 3, own, np.array([400, 401, 402]), helix_size=16, helix_rank=4
    )
    assert src.size == out.size == 0
    # A page the receiver does not hold (local 1 = page 19) and pages before the start move nowhere.
    src, out = dkv_helix_page_ids(
        DkvView(), 8, 3, own, np.array([200, -1, 202]), 4, helix_size=16, helix_rank=3
    )
    np.testing.assert_array_equal(src, own[[35]])
    np.testing.assert_array_equal(out, [202])


def test_a_helix_rank_of_eight_hears_from_two_dkv_ranks_of_sixteen():
    """cep16 -> Helix CP8: Helix rank 3 stores global pages 3, 11, 19, 27, 35 (local 0-4).

    DKV rank 3 owns 3, 19, 35 (local 0, 2, 4) and DKV rank 11 owns 11, 27 (local 1, 3).
    """
    dst = 500 + np.arange(5)
    own3, own11 = _dkv_rank_slots(16, 3, 40), _dkv_rank_slots(16, 11, 40)
    src, out = dkv_helix_page_ids(DkvView(), 16, 3, own3, dst, helix_size=8, helix_rank=3)
    np.testing.assert_array_equal(src, own3[[3, 19, 35]])
    np.testing.assert_array_equal(out, dst[[0, 2, 4]])
    src, out = dkv_helix_page_ids(DkvView(), 16, 11, own11, dst, helix_size=8, helix_rank=3)
    np.testing.assert_array_equal(src, own11[[11, 27]])
    np.testing.assert_array_equal(out, dst[[1, 3]])
    src, _ = dkv_helix_page_ids(
        DkvView(pages_per_slot=32), 16, 11, own11, dst, helix_size=8, helix_rank=3
    )
    np.testing.assert_array_equal(src, own11[[11, 27]] * 32 + 11)
    src, out = dkv_helix_page_ids(
        DkvView(), 16, 4, _dkv_rank_slots(16, 4, 40), dst, helix_size=8, helix_rank=3
    )
    assert src.size == out.size == 0
    held = dst.copy()
    held[2] = -1  # page 19
    src, out = dkv_helix_page_ids(DkvView(), 16, 3, own3, held, 4, helix_size=8, helix_rank=3)
    np.testing.assert_array_equal(src, own3[[35]])
    np.testing.assert_array_equal(out, [504])
    with pytest.raises(ValueError, match="same prompt"):
        dkv_helix_page_ids(DkvView(), 16, 3, own3, dst[:4], helix_size=8, helix_rank=3)


@pytest.mark.parametrize("size, helix_size", [(4, 4), (8, 8), (8, 16), (16, 8), (4, 16), (16, 4)])
def test_the_dkv_group_fills_every_helix_page_once_from_its_owner(size, helix_size):
    """Each page a Helix rank stores comes once, from the DKV rank that owns it.

    Helix rank h stores global page g = h + j * M as local page j; DKV rank g % N owns it, in its
    slot of DKV block g // N. The ranks that write Helix rank h are the owners of its pages, and a
    pair writes pages only when the two ranks share some.
    """
    for pages in (1, 7, 16, 37, 69):
        for helix_rank in range(helix_size):
            stored = np.arange(helix_rank, pages, helix_size)
            dst = 1000 * (helix_rank + 1) + np.arange(stored.size)
            received, writers = {}, set()
            for rank in range(size):
                own = _dkv_rank_slots(size, rank, pages)
                src, out = dkv_helix_page_ids(
                    DkvView(), size, rank, own, dst, helix_size=helix_size, helix_rank=helix_rank
                )
                shares = dkv_helix_shares_pages(size, rank, helix_size, helix_rank)
                assert shares == any(g % size == rank for g in range(helix_rank, 64, helix_size))
                assert shares or src.size == 0
                if src.size:
                    writers.add(rank)
                for src_id, dst_id in zip(src.tolist(), out.tolist()):
                    assert dst_id not in received
                    received[dst_id] = (rank, src_id)
            assert writers == {int(g) % size for g in stored}, (pages, helix_rank)
            for j, g in enumerate(stored.tolist()):
                assert received[int(dst[j])] == (g % size, 100 * (g % size) + g // size)
            assert len(received) == stored.size


def test_contiguous_copies_merge_into_runs():
    src = np.array([300, 100, 200, 400, 500], dtype=np.int64)
    dst = np.array([1300, 1100, 1200, 1500, 1600], dtype=np.int64)
    sizes = np.full(5, 100, dtype=np.int64)
    s, d, n = merge_runs(src, dst, sizes)
    # 100-300 continue each other on both sides; 400 does not (its dst jumps); 500 continues 400.
    np.testing.assert_array_equal(s, [100, 400])
    np.testing.assert_array_equal(d, [1100, 1500])
    np.testing.assert_array_equal(n, [300, 200])


def _rank_info(tp_rank, dkv=1, mla=True, adp=False, cp=1):
    attention = AttentionInfo(
        kv_heads_per_rank=1,
        tokens_per_block=64,
        dims_per_head=576,
        element_bytes=1,
        enable_attention_dp=adp,
        is_mla=mla,
    )
    return RankInfo(
        instance_name="ctx" if dkv > 1 else "gen",
        instance_rank=tp_rank,
        tp_size=8,
        tp_rank=tp_rank,
        pp_size=1,
        pp_rank=0,
        layer_num_per_pp=[4],
        sender_endpoints=[],
        self_endpoint="",
        transfer_engine_info=b"",
        cp_size=cp,
        attention=attention,
        dkv_size=dkv,
    )


def _registrar(ri):
    return PeerRegistrar(ri, SimpleNamespace(page_table=None))


def test_a_dkv_group_fans_out_to_every_generation_rank_and_back():
    ctx = _registrar(_rank_info(3, dkv=8))
    gen = _registrar(_rank_info(5))
    assert ctx.dkv_fan_out(_rank_info(5)) and gen.dkv_fan_out(_rank_info(3, dkv=8))
    assert ctx.get_peer_overlap(_rank_info(5), 0).ranks == list(range(8))
    assert gen.get_peer_overlap(_rank_info(3, dkv=8), 0).ranks == list(range(8))
    # The aux goes once, from the TP-paired rank.
    assert not ctx.should_send_aux(_rank_info(5))
    assert _registrar(_rank_info(5, dkv=8)).should_send_aux(_rank_info(5))


def _helix_info(cp_rank, cp=8, tpb=64):
    ri = _rank_info(0, cp=cp)
    ri.tp_size, ri.tp_rank, ri.cp_rank, ri.instance_name = 1, 0, cp_rank, "helix"
    ri.attention.tokens_per_block = tpb
    return ri


def test_a_dkv_rank_pairs_with_the_helix_rank_that_stores_its_pages():
    ctx = _registrar(_rank_info(3, dkv=8))
    gen = _registrar(_helix_info(3))
    assert ctx.dkv_helix(_helix_info(3)) and gen.dkv_helix(_rank_info(3, dkv=8))
    assert not ctx.dkv_fan_out(_helix_info(3))
    # The replicated sender's overlap (every context rank writes every Helix rank); the paged KV
    # moves only between DKV rank 3 and Helix CP rank 3.
    assert ctx.get_peer_overlap(_helix_info(3), 0).ranks == list(range(8))
    assert gen.get_peer_overlap(_rank_info(3, dkv=8), 0).ranks == list(range(8))
    # Not a Helix pair: a CP size that is neither a multiple nor a divisor of 8, a page size of
    # its own, or TP inside the Helix group.
    assert not ctx.dkv_helix(_helix_info(3, cp=6))
    assert not ctx.dkv_helix(_helix_info(3, cp=12))
    assert not ctx.dkv_helix(_helix_info(3, tpb=32))
    tp2 = _helix_info(3)
    tp2.tp_size = 2
    assert not ctx.dkv_helix(tp2)
    # A replicated context server keeps the plain Helix overlap: every rank, one elected owner.
    plain = _registrar(_rank_info(3))
    assert not plain.dkv_helix(_helix_info(3))
    assert plain.get_peer_overlap(_helix_info(3), 0).ranks == list(range(8))


def _dkv_info(rank, size):
    ri = _rank_info(rank, dkv=size)
    ri.tp_size = size
    return ri


def _head_routed_page_table():
    """The DKV manager's page table with the latent as the MLA cache describes it.

    The latent is head-routed (the replicated sender elects one rank per duplicated head), the
    index keys replicated (fan-in election).
    """
    _, page_table, _ = _dkv_manager()
    lg = page_table.layer_groups[0]
    latent = lg.pool_views[0]
    lg.pool_views[0] = PoolView(
        pool_idx=latent.pool_idx,
        buffer_entries=latent.buffer_entries,
        pool_role=latent.pool_role,
        mapper_kind=MapperKind.INDEXED,
        bytes_per_layer=latent.bytes_per_layer,
    )
    return page_table


def test_every_dkv_rank_sends_its_paged_views_to_its_helix_partner_only():
    """DKV rank r alone sends Helix rank r every paged view.

    Not the replicated sender's head routing, which elects rank 0 for the latent.
    """
    page_table = _head_routed_page_table()
    for rank in range(8):
        ri = _rank_info(rank, dkv=8)
        registrar = PeerRegistrar(ri, SimpleNamespace(page_table=page_table))
        for cp_rank in range(8):
            helix = _helix_info(cp_rank)
            overlap = registrar.get_peer_overlap(helix, 0)
            for view in range(2):
                sends = registrar.should_send_pool(overlap, helix, 0, view)
                assert sends == (rank == cp_rank), (rank, cp_rank, view)


@pytest.mark.parametrize("size", [8, 16])
def test_a_dkv_group_pairs_with_a_helix_group_of_a_multiple_or_a_divisor_of_its_size(size):
    ri = _dkv_info(3, size)
    ctx = _registrar(ri)
    for cp in (2, 4, 8, 16, 32):
        assert ctx.dkv_helix(_helix_info(1, cp=cp)), (size, cp)
        assert _registrar(_helix_info(1, cp=cp)).dkv_helix(ri), (size, cp)
        assert not ctx.dkv_fan_out(_helix_info(1, cp=cp))
    for cp in (3, 6, 12, 20):
        assert not ctx.dkv_helix(_helix_info(1, cp=cp)), (size, cp)
    # CP 1 is a plain TP server: fan-out, not Helix.
    assert not ctx.dkv_helix(_helix_info(0, cp=1)) and ctx.dkv_fan_out(_helix_info(0, cp=1))


@pytest.mark.parametrize("size, helix_size", [(8, 8), (8, 16), (16, 8)])
def test_every_dkv_rank_writes_every_helix_rank_and_each_side_counts_the_other(size, helix_size):
    """The replicated sender's overlap: M targets per DKV rank, N writers per Helix rank.

    The receiver waits for a write from every DKV rank; the aux comes from DKV rank 0 alone.
    """
    for rank in range(size):
        ctx = _registrar(_dkv_info(rank, size))
        for cp_rank in range(helix_size):
            helix = _helix_info(cp_rank, cp=helix_size)
            assert ctx.get_peer_overlap(helix, 0).ranks == list(range(helix_size))
            gen = _registrar(helix)
            assert gen.get_peer_overlap(_dkv_info(rank, size), 0).ranks == list(range(size))
            assert ctx.should_send_aux(helix) == (rank == 0)


@pytest.mark.parametrize("size, helix_size", [(8, 16), (16, 8), (4, 16), (16, 4)])
def test_every_dkv_rank_sends_its_paged_views_to_the_helix_ranks_that_store_its_pages(
    size, helix_size
):
    """DKV rank r sends Helix rank h every paged view when r and h agree modulo min(N, M).

    That is when they share pages. Helix rank h hears from DKV ranks h % N, h % N + M, ...: one
    rank when M >= N, and N / M ranks when M < N.
    """
    page_table = _head_routed_page_table()
    senders = {cp_rank: [] for cp_rank in range(helix_size)}
    for rank in range(size):
        registrar = PeerRegistrar(_dkv_info(rank, size), SimpleNamespace(page_table=page_table))
        for cp_rank in range(helix_size):
            helix = _helix_info(cp_rank, cp=helix_size)
            overlap = registrar.get_peer_overlap(helix, 0)
            sends = {registrar.should_send_pool(overlap, helix, 0, view) for view in range(2)}
            assert sends == {rank % min(size, helix_size) == cp_rank % min(size, helix_size)}
            if sends == {True}:
                senders[cp_rank].append(rank)
    for cp_rank, ranks in senders.items():
        assert ranks == list(range(cp_rank % size, size, helix_size)), (cp_rank, ranks)


def test_no_fan_out_without_exactly_one_dkv_side_or_with_dp_cp_or_non_mla():
    plain = _registrar(_rank_info(3))
    assert not plain.dkv_fan_out(_rank_info(3))
    assert plain.get_peer_overlap(_rank_info(3), 0).ranks == [3]
    dkv = _registrar(_rank_info(3, dkv=8))
    assert not dkv.dkv_fan_out(_rank_info(3, dkv=8))
    assert not dkv.dkv_fan_out(_rank_info(3, adp=True))
    assert not dkv.dkv_fan_out(_rank_info(3, mla=False))
    assert not dkv.dkv_fan_out(_rank_info(3, cp=2))


PAGE_LATENT = 64 * 576
PAGE_INDEX = 64 * 132
LATENT_BASE = 1 << 40
INDEX_BASE = 2 << 40
NUM_SLOTS = 16


def _dkv_manager():
    """A DKV manager with two DSA layers, and the page table the builder made for it.

    One life cycle holds a latent pool (one page per layer per slot, sharded) and an index-key
    pool (N pages per layer per slot, expanded).
    """
    latent = SimpleNamespace(
        single_buffer_size=PAGE_LATENT,
        buffer_ids=[SimpleNamespace(layer_id=i, role="key") for i in range(2)],
    )
    index = SimpleNamespace(
        single_buffer_size=PAGE_INDEX * N,
        buffer_ids=[SimpleNamespace(layer_id=i, role="index_key") for i in range(2)],
    )
    pg_desc = SimpleNamespace(
        slot_desc=SimpleNamespace(
            variants=[SimpleNamespace(layer_group_id=0, coalesced_buffers=[latent, index])]
        )
    )
    expansion = {"key": 1, "index_key": N}
    impl = SimpleNamespace(
        pool_group_descs=[pg_desc],
        get_page_index_converter=lambda layer, role: SimpleNamespace(expansion=expansion[role]),
    )
    dkv = Dkv(size=N, rank=RANK, tokens_per_block=64, all_gather=lambda x: [x] * N)
    mirror = PeerRegion(base=7 << 40, stride=1 << 32, size=NUM_SLOTS * 2 * PAGE_LATENT, row_bytes=1)
    manager = SimpleNamespace(
        impl=impl,
        dkv=dkv,
        dkv_life_cycles=(0,),
        tokens_per_block=64,
        dkv_peer_pools=lambda: [(LATENT_BASE, NUM_SLOTS * 2 * PAGE_LATENT, mirror)],
    )

    def entries(size, buffer_bytes):
        return np.array([(i, i * buffer_bytes, size) for i in range(2)], dtype=BUFFER_ENTRY_DTYPE)

    page_table = KVCachePageTable(
        tokens_per_block=64 * N,
        layer_groups=[
            AttentionLayerGroup(
                pool_group_idx=0,
                kv_head_num_per_rank=1,
                sliding_window_size=None,
                local_layers=[LocalLayer(local_layer_id=i, global_layer_id=i) for i in range(2)],
                pool_views=[
                    PoolView(
                        pool_idx=0,
                        buffer_entries=entries(PAGE_LATENT, PAGE_LATENT),
                        pool_role=frozenset({"key"}),
                        mapper_kind=MapperKind.REPLICATED,
                        bytes_per_layer=PAGE_LATENT,
                    ),
                    PoolView(
                        pool_idx=1,
                        buffer_entries=entries(PAGE_INDEX * N, PAGE_INDEX * N),
                        pool_role=frozenset({"index_key"}),
                        mapper_kind=MapperKind.REPLICATED,
                        bytes_per_layer=PAGE_INDEX * N,
                    ),
                ],
            )
        ],
        pool_groups=[
            PhysicalPoolGroup(
                pools=[
                    PhysicalPool(LATENT_BASE, 2 * PAGE_LATENT, NUM_SLOTS),
                    PhysicalPool(INDEX_BASE, 2 * PAGE_INDEX * N, NUM_SLOTS),
                ]
            )
        ],
    )
    return manager, page_table, mirror


def test_the_page_table_describes_expanded_pools_page_by_page():
    manager, page_table, _ = _dkv_manager()
    split_expanded_dkv_views(manager, page_table)
    latent, index = page_table.layer_groups[0].pool_views
    assert latent.pool_idx == 0 and latent.bytes_per_layer == PAGE_LATENT
    pool = page_table.pool_groups[0].pools[index.pool_idx]
    assert index.pool_idx == 2 and index.bytes_per_layer == PAGE_INDEX
    assert (pool.base_address, pool.slot_bytes, pool.slot_stride_bytes) == (
        INDEX_BASE,
        PAGE_INDEX,
        PAGE_INDEX,
    )
    assert pool.num_slots == NUM_SLOTS * 2 * N
    assert list(index.buffer_entries["size"]) == [PAGE_INDEX] * 2
    assert list(index.buffer_entries["offset"]) == [0, PAGE_INDEX * N]
    assert index.pool_role == frozenset({"index_key"})


def test_the_dkv_layout_names_sharded_and_expanded_views():
    manager, page_table, _ = _dkv_manager()
    split_expanded_dkv_views(manager, page_table)
    layout = build_dkv_layout(manager, page_table)
    assert (layout.size, layout.rank, layout.tokens_per_block) == (N, RANK, 64)
    assert layout.life_cycles == frozenset({0})
    assert layout.views[(0, 0)] == DkvView() and layout.views[(0, 0)].sharded
    assert layout.views[(0, 1)] == DkvView(pages_per_slot=2 * N)


def test_an_expanded_page_id_addresses_the_sub_page_the_writer_used():
    manager, page_table, _ = _dkv_manager()
    split_expanded_dkv_views(manager, page_table)
    layout = build_dkv_layout(manager, page_table)
    own = own_page_slots(_own_blocks(), N, 10)
    src, _ = dkv_page_ids(layout.views[(0, 1)], N, RANK, own, np.arange(10))
    page = np.array([1, 5, 9])
    # Page g of layer k: slot start + k * (N pages) + (g % N) pages.
    expected = INDEX_BASE + own[page] * (2 * PAGE_INDEX * N) + (page % N) * PAGE_INDEX
    np.testing.assert_array_equal(INDEX_BASE + src * PAGE_INDEX, expected)


def test_without_a_dkv_there_is_no_layout():
    manager, page_table, _ = _dkv_manager()
    manager.dkv = None
    assert build_dkv_layout(manager, page_table) is None


def test_every_dkv_rank_of_a_larger_group_sends_its_paged_views_by_fan_out():
    """cep16 -> TEP8: every DKV rank writes its own pages to every generation rank.

    The replicated fan-in election (16 / 8 = 2 senders per generation rank) would skip the odd
    ranks, whose pages would then never reach the generation server.
    """
    _, page_table, _ = _dkv_manager()
    for rank in range(16):
        ri = _rank_info(rank, dkv=16)
        ri.tp_size = 16
        registrar = PeerRegistrar(ri, SimpleNamespace(page_table=page_table))
        for gen_rank in range(8):
            gen = _rank_info(gen_rank)
            assert registrar.dkv_fan_out(gen)
            overlap = registrar.get_peer_overlap(gen, 0)
            for view in range(2):
                assert registrar.should_send_pool(overlap, gen, 0, view), (rank, gen_rank, view)


DRAFT_LAYERS = range(93, 98)


def _draft_page_table(shards):
    """One unified draft layer group (five FP8 MLA latent layers) with ``shards`` draft shards."""
    entries = np.array(
        [(i, i * PAGE_LATENT, PAGE_LATENT) for i in range(len(DRAFT_LAYERS))],
        dtype=BUFFER_ENTRY_DTYPE,
    )
    group = AttentionLayerGroup(
        pool_group_idx=0,
        kv_head_num_per_rank=1,
        sliding_window_size=None,
        local_layers=[
            LocalLayer(local_layer_id=i, global_layer_id=g) for i, g in enumerate(DRAFT_LAYERS)
        ],
        pool_views=[
            PoolView(
                pool_idx=0,
                buffer_entries=entries,
                pool_role=frozenset({"key"}),
                mapper_kind=MapperKind.INDEXED,
                bytes_per_layer=PAGE_LATENT,
            )
        ],
        cp_as_tp=True,
        total_kv_head_num=1,
        cp_as_tp_shards=shards,
    )
    pool = PhysicalPool(LATENT_BASE, len(DRAFT_LAYERS) * PAGE_LATENT, NUM_SLOTS)
    return KVCachePageTable(
        tokens_per_block=64, layer_groups=[group], pool_groups=[PhysicalPoolGroup(pools=[pool])]
    )


def test_a_dkv_rank_lists_its_draft_pages_by_global_page():
    """A split draft's table is positional over the prompt's global pages.

    Split over the DKV group (``draft_page_split``): this rank's slot of DKV block g // N at its
    pages g % N == rank, -1 at the others; a replicated draft lists every page.
    """
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    def describe(split, rank, slots):
        transceiver = object.__new__(KvCacheTransceiverV2)
        transceiver._reuse_adapter = SimpleNamespace(
            tokens_per_block=64, get_block_ordinals=lambda req, idx, lg: np.asarray(slots)
        )
        transceiver._kv_cache_manager = SimpleNamespace(
            draft_cp_rank=rank,
            draft_page_split=(N, rank) if split else None,
            draft_window_size=None,
        )
        req = SimpleNamespace(
            py_request_id=7,
            prompt_len=700,  # 11 pages
            is_generation_only_request=False,
            context_remaining_length=0,
        )
        lg = SimpleNamespace(cp_as_tp_shards=N if split else 1)
        return transceiver._describe_draft_group(req, 0, lg, 0)

    table = describe(True, RANK, [10, 11, 12])
    np.testing.assert_array_equal(table, [-1, 10, -1, -1, -1, 11, -1, -1, -1, 12, -1])
    owners = np.full(11, -1)
    for rank in range(N):
        held = np.flatnonzero(describe(True, rank, [20 * rank + b for b in range(3)]) >= 0)
        assert (owners[held] == -1).all()
        owners[held] = rank
    np.testing.assert_array_equal(owners, np.arange(11) % N)
    with pytest.raises(ValueError, match="missing allocated prompt pages"):
        describe(True, RANK, [10, 11])
    np.testing.assert_array_equal(describe(False, 0, np.arange(11)), np.arange(11))


def test_every_dkv_rank_sends_its_draft_pages_to_every_receiver_that_keeps_them():
    """A split draft has no replicas to elect.

    cep16 -> TEP8: every DKV rank writes its own pages to every generation rank; cep8 -> Helix8
    (draft shards 8): rank r to Helix rank r. Each receiver gets every page it keeps once, from
    its owner. A replicated draft keeps the election: one sender per generation rank.
    """
    from tensorrt_llm._torch.disaggregation.native.transfer import Sender

    pages = 37
    for size, helix in ((16, False), (8, False), (8, True)):
        received = {}
        for rank in range(size):
            ri = _rank_info(rank, dkv=size)
            ri.tp_size = size
            ri.page_table = _draft_page_table(size)
            registrar = PeerRegistrar(ri, SimpleNamespace(page_table=ri.page_table))
            # This rank's slot of DKV block g // size at its own pages.
            src = np.where(np.arange(pages) % size == rank, 1000 * rank + np.arange(pages), -1)
            for peer in range(8):
                gen = _helix_info(peer) if helix else _rank_info(peer)
                gen.page_table = _draft_page_table(8 if helix else 1)
                overlap = registrar.get_peer_overlap(gen, 0)
                sends = registrar.should_send_pool(overlap, gen, 0, 0)
                assert sends == (rank % 8 == peer if helix else True), (size, helix, rank, peer)
                if not sends:
                    continue
                kept = np.arange(peer, pages, 8) if helix else np.arange(pages)
                dst = 5000 + np.arange(kept.size)
                src_ids, dst_ids = Sender._pair_ordinals(
                    Sender._draft_source_table(src, gen, 0), dst
                )
                assert (src_ids // 1000 == rank).all()
                for s, d in zip(src_ids, dst_ids):
                    assert (peer, d) not in received
                    received[(peer, d)] = s
        for peer in range(8):
            kept = np.arange(peer, pages, 8) if helix else np.arange(pages)
            got = [received[(peer, 5000 + j)] for j in range(kept.size)]
            np.testing.assert_array_equal(got, 1000 * (kept % size) + kept)
    # A replicated draft (every page on every rank): the elected rank of each pair sends.
    for rank in range(16):
        ri = _rank_info(rank, dkv=16)
        ri.tp_size = 16
        ri.page_table = _draft_page_table(1)
        registrar = PeerRegistrar(ri, SimpleNamespace(page_table=ri.page_table))
        gen = _rank_info(rank // 2)
        gen.page_table = _draft_page_table(1)
        assert registrar.should_send_pool(registrar.get_peer_overlap(gen, 0), gen, 0, 0) == (
            rank % 2 == 0
        )


class _TaggedExtractor:
    """Stands in for a region extractor: names the ids it was asked for."""

    def __init__(self, side):
        self.side = side

    def extract(self, ids, layer_group, pool):
        return (self.side, layer_group, pool, np.asarray(ids).tolist())


def _dkv_sender(rank, size):
    from tensorrt_llm._torch.disaggregation.native.transfer import Sender

    registrar = _registrar(_dkv_info(rank, size))
    sender = object.__new__(Sender)
    sender._dkv = DkvLayout(
        size=size,
        rank=rank,
        tokens_per_block=64,
        life_cycles=frozenset({0}),
        views={(0, 0): DkvView()},
    )
    sender._registrar = SimpleNamespace(
        dkv_fan_out=registrar.dkv_fan_out,
        dkv_helix=registrar.dkv_helix,
        self_extractor=_TaggedExtractor("src"),
        peer_extractor=lambda name, rank: _TaggedExtractor("dst"),
    )
    return sender


@pytest.mark.parametrize("size, helix_size", [(8, 8), (8, 16), (16, 8)])
def test_the_regions_of_a_dkv_helix_pair_are_the_pages_the_two_share(size, helix_size):
    """Sender._dkv_regions: DKV rank r writes Helix rank h the pages it owns there.

    Those are the pages g % N == r, g % M == h, from its slot of DKV block g // N to the Helix
    rank's local page g // M; nothing goes to a rank that stores none of them. The receiver's
    start token cuts pages on the global grid.
    """
    pages = 45
    for rank in range(size):
        sender = _dkv_sender(rank, size)
        own = _dkv_rank_slots(size, rank, pages)
        task = SimpleNamespace(_chunk=SimpleNamespace(block_ids_per_layer_groups=[own]))
        for cp_rank in range(helix_size):
            helix = _helix_info(cp_rank, cp=helix_size)
            dst = 1000 * cp_rank + np.arange(len(range(cp_rank, pages, helix_size)))
            for start_page in (0, 20):
                req = SimpleNamespace(
                    dst_start_token=64 * start_page if start_page else None,
                    block_ids_per_layer_groups=[dst],
                )
                src_region, dst_region = sender._dkv_regions(task, req, helix, (0, 0), (0, 0))
                shared = [
                    g
                    for g in range(start_page, pages)
                    if g % size == rank and g % helix_size == cp_rank
                ]
                if not shared:
                    assert src_region is None and dst_region is None
                    continue
                assert src_region == ("src", 0, 0, own[shared].tolist())
                assert dst_region == ("dst", 0, 0, [int(dst[g // helix_size]) for g in shared])
    # A Helix size that is neither a multiple nor a divisor of the group's is refused.
    with pytest.raises(ValueError, match="multiple or a divisor of 8"):
        _dkv_sender(3, 8)._dkv_regions(task, req, _helix_info(1, cp=6), (0, 0), (0, 0))


@pytest.mark.parametrize(
    "size, helix_size, shards", [(8, 8, 8), (16, 8, 8), (8, 16, 16), (16, 8, 1), (8, 16, 1)]
)
def test_a_split_draft_reaches_every_helix_rank_from_the_ranks_that_own_its_pages(
    size, helix_size, shards
):
    """Each Helix rank gets every draft page it keeps once, from the DKV rank that owns it.

    The Helix drafter keeps global page g on its grid rank g % S (S its draft shards, the Helix CP
    size), or every page with S = 1; DKV rank r owns the pages g % N == r. A pair writes when its
    ranks agree modulo gcd(N, S).
    """
    from tensorrt_llm._torch.disaggregation.native.transfer import Sender

    pages = 37
    common = math.gcd(size, shards)
    received = {}
    for rank in range(size):
        ri = _dkv_info(rank, size)
        ri.page_table = _draft_page_table(size)
        registrar = PeerRegistrar(ri, SimpleNamespace(page_table=ri.page_table))
        src = np.where(np.arange(pages) % size == rank, 1000 * rank + np.arange(pages), -1)
        for peer in range(helix_size):
            gen = _helix_info(peer, cp=helix_size)
            gen.page_table = _draft_page_table(shards)
            sends = registrar.should_send_pool(registrar.get_peer_overlap(gen, 0), gen, 0, 0)
            assert sends == (shards == 1 or rank % common == peer % common), (rank, peer)
            if not sends:
                continue
            kept = np.arange(peer % shards, pages, shards)
            dst = 5000 + np.arange(kept.size)
            src_ids, dst_ids = Sender._pair_ordinals(Sender._draft_source_table(src, gen, 0), dst)
            assert (src_ids // 1000 == rank).all()
            for s_id, d_id in zip(src_ids.tolist(), dst_ids.tolist()):
                assert (peer, d_id) not in received
                received[(peer, d_id)] = s_id
    for peer in range(helix_size):
        kept = np.arange(peer % shards, pages, shards)
        got = [received[(peer, 5000 + j)] for j in range(kept.size)]
        np.testing.assert_array_equal(got, 1000 * (kept % size) + kept)


@pytest.mark.parametrize("size, helix_size", [(8, 8), (16, 8), (8, 16)])
@pytest.mark.parametrize("kind", [MapperKind.INDEXED, MapperKind.SECTIONED])
def test_the_kda_state_reaches_every_helix_rank_head_by_head(size, helix_size, kind):
    """The KDA state goes by MambaPolicy, not by the DKV pages.

    Both sides shard the recurrent state's heads (and each of the conv state's three sections)
    over their TP x CP grid, DKV rank r holding heads [r * H / N, (r + 1) * H / N). Every head of
    every Helix rank comes once, from the paired DKV rank that holds it.
    """
    heads, head_bytes = 96, 16
    sections = 1 if kind == MapperKind.INDEXED else 3

    def view(ranks):
        section = heads // ranks * head_bytes
        return PoolView(
            pool_idx=0,
            buffer_entries=np.zeros(0, dtype=BUFFER_ENTRY_DTYPE),
            pool_role=frozenset({"ssm" if kind == MapperKind.INDEXED else "conv"}),
            mapper_kind=kind,
            bytes_per_layer=sections * section,
            section_bytes=[section] * sections if kind == MapperKind.SECTIONED else None,
            bytes_per_head=head_bytes if kind == MapperKind.INDEXED else None,
        )

    def region():
        return SpecRegion(
            memory=MemRegionGroup(ptrs=np.zeros(1, dtype=np.int64), bytes_per_region=0)
        )

    src_view, dst_view = view(size), view(helix_size)
    src_section, dst_section = heads // size * head_bytes, heads // helix_size * head_bytes
    got = {}
    for rank in range(size):
        ri = _dkv_info(rank, size)
        for cp_rank in range(helix_size):
            helix = _helix_info(cp_rank, cp=helix_size)
            if not MambaPolicy.is_paired(ri, helix):
                continue
            mapper = MambaPolicy(ri).build_mapper(
                peer_ri=helix,
                mapper_kind=kind,
                self_layer_offsets=[0],
                peer_layer_offsets=[0],
                self_bytes_per_layer=src_view.bytes_per_layer,
                peer_bytes_per_layer=dst_view.bytes_per_layer,
                self_pv=src_view,
                peer_pv=dst_view,
            )
            pairs = mapper.map(region(), region())
            for pair in pairs if isinstance(pairs, list) else [pairs]:
                src, dst = int(pair.src.memory.ptrs[0]), int(pair.dst.memory.ptrs[0])
                for offset in range(0, pair.src.memory.bytes_per_region, head_bytes):
                    src_sec, src_off = divmod(src + offset, src_section)
                    dst_sec, dst_off = divmod(dst + offset, dst_section)
                    assert src_sec == dst_sec
                    head = rank * (heads // size) + src_off // head_bytes
                    assert head == cp_rank * (heads // helix_size) + dst_off // head_bytes
                    assert (cp_rank, dst_sec, head) not in got
                    got[(cp_rank, dst_sec, head)] = rank
    assert set(got) == {
        (cp_rank, section, cp_rank * (heads // helix_size) + i)
        for cp_rank in range(helix_size)
        for section in range(sections)
        for i in range(heads // helix_size)
    }
