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
"""Loopback of the standalone drafter's context window (F0a), at unit level.

A context rank with one TP rank per attention-DP group holds all four drafter KV heads; generation
TP4 holds one head per rank (head r) and TP8 one duplicated head per rank (head r // 2). The real
rank-info serialization, peer registrars, extractors, head-mismatch mappers and
``Sender._build_kv_write_meta`` build the write; the write is then applied with host memcpy to
host-memory pools laid out like the V2 HND pools, so no NIXL agent or GPU is needed.
"""

import ctypes
import importlib.util
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest
import torch

from tensorrt_llm import DisaggregatedParams
from tensorrt_llm._torch.disaggregation.base import CacheExtent, CacheKind, Chunk, TokenRange
from tensorrt_llm._torch.disaggregation.draft_kv import (
    DUMP_DIR_ENV,
    DraftKvCounters,
    draft_dump_dir,
    draft_receive_disabled_reason,
    encode_draft_ctx_state,
)
from tensorrt_llm._torch.disaggregation.native.mixers.attention.spec import AttentionInfo
from tensorrt_llm._torch.disaggregation.native.peer import PeerOverlap, PeerRegistrar
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.native.transfer import (
    KVSendTask,
    Receiver,
    RecvReqInfo,
    Sender,
    aux_meta_without_draft_ctx,
    check_draft_kv_peer,
    draft_peer_mismatch,
    draft_peer_not_engaged,
)
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
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
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm.disaggregated_params import DisaggScheduleStyle

pytestmark = pytest.mark.cpu_only

# Target and draft pools deliberately share role labels and global layer ids: one pool mapping
# would cross-match them, so the draft set must go through its own registrar.
_ROLES = frozenset({"key", "value"})
_INDEX_ROLES = frozenset({"index_key"})
_LAYER_IDS = [0, 1]
_RID = 4242
_SENTINEL = -1

_TARGET_TPB = 64
_TARGET_DIM = 8
_TARGET_HEADS = 8  # all heads on the context rank (attention DP)
_DRAFT_TPB = 32
_DRAFT_DIM = 16
_DRAFT_HEADS = 4  # the drafter's KV heads, all on the context rank (attention DP)
_DRAFT_WINDOW = 2048
# The context side's last prefill chunk starts this far before the prompt end: its windowed draft
# pool still holds the window behind that chunk start, a superset of what generation keeps.
_CTX_LAST_CHUNK = 4096

PROMPT_LENS = [31, 32, 33, 4095, 4097, 9000]


def _blocks(tokens: int, tpb: int) -> int:
    return (tokens + tpb - 1) // tpb


def _window_start_block(history: int, tpb: int) -> int:
    """First ordinal a windowed pool still holds at ``history`` (released below it)."""
    return max(0, (history + 1 - _DRAFT_WINDOW) // tpb)


def _value(layer, kv, head, pos, dim):
    """The K/V the context side stored: f(layer, K/V, global head, position, dim) as int16."""
    return ((layer * 2 + kv) * 1009 + head * 131 + pos * 7 + dim * 3) % 30011


def _index_value(layer, pos, dim):
    """The MSA index key: one vector per token, the same on every TP rank (replicated weights)."""
    return ((layer + 3) * 977 + pos * 5 + dim * 11) % 30011


class _Pool:
    """A host-memory paged pool laid out like a V2 HND attention pool.

    Shape ``[slot, layer, K/V, head, token, dim]``: one slot holds every layer's K then V buffer,
    each buffer head-major. ``head_offset`` is the global index of this rank's first head.
    ``index_dim`` > 0 adds MiniMax M3's MSA index-K cache as a second physical pool, one
    ``[token, index_dim]`` vector per layer and block, declared REPLICATED.
    """

    def __init__(
        self,
        num_slots: int,
        heads: int,
        tpb: int,
        head_dim: int,
        window=None,
        head_offset: int = 0,
        index_dim: int = 0,
        dtype=np.int16,
    ):
        self.window = window
        self.head_offset = head_offset
        self.array = np.full(
            (num_slots, len(_LAYER_IDS), 2, heads, tpb, head_dim), _SENTINEL, dtype=dtype
        )
        self.index = (
            np.full((num_slots, len(_LAYER_IDS), tpb, index_dim), _SENTINEL, dtype=np.int16)
            if index_dim
            else None
        )

    @property
    def heads(self) -> int:
        return self.array.shape[3]

    @property
    def tpb(self) -> int:
        return self.array.shape[4]

    def page_table(self) -> KVCachePageTable:
        num_slots, layers, _, heads, tpb, dim = self.array.shape
        buf = heads * tpb * dim * self.array.itemsize
        entries = [(i, (2 * i + kv) * buf, buf) for i in range(layers) for kv in range(2)]
        view = PoolView(
            pool_idx=0,
            buffer_entries=np.array(entries, dtype=BUFFER_ENTRY_DTYPE),
            pool_role=_ROLES,
            mapper_kind=MapperKind.INDEXED,
            bytes_per_layer=2 * buf,
        )
        views = [view]
        pools = [
            PhysicalPool(
                base_address=self.array.ctypes.data,
                slot_bytes=self.array[0].nbytes,
                num_slots=num_slots,
            )
        ]
        if self.index is not None:
            index_buf = self.index[0, 0].nbytes
            views.append(
                PoolView(
                    pool_idx=1,
                    buffer_entries=np.array(
                        [(i, i * index_buf, index_buf) for i in range(layers)],
                        dtype=BUFFER_ENTRY_DTYPE,
                    ),
                    pool_role=_INDEX_ROLES,
                    mapper_kind=MapperKind.REPLICATED,
                    bytes_per_layer=index_buf,
                )
            )
            pools.append(
                PhysicalPool(
                    base_address=self.index.ctypes.data,
                    slot_bytes=self.index[0].nbytes,
                    num_slots=num_slots,
                )
            )
        group = AttentionLayerGroup(
            pool_group_idx=0,
            kv_head_num_per_rank=heads,
            sliding_window_size=self.window,
            local_layers=[LocalLayer(i, g) for i, g in enumerate(_LAYER_IDS)],
            pool_views=views,
        )
        return KVCachePageTable(
            tokens_per_block=tpb,
            layer_groups=[group],
            pool_groups=[PhysicalPoolGroup(pools=pools)],
        )

    def fill(self, table: np.ndarray) -> None:
        """Store f(...) at every block the positional table holds."""
        _, layers, _, heads, tpb, dim = self.array.shape
        layer, kv, h, t, d = np.indices((layers, 2, heads, tpb, dim))
        for ordinal, slot in enumerate(table):
            if slot >= 0:
                self.array[slot] = _value(layer, kv, h + self.head_offset, ordinal * tpb + t, d)
                if self.index is not None:
                    il, it, idim = np.indices(self.index.shape[1:])
                    self.index[slot] = _index_value(il, ordinal * tpb + it, idim)


def _attention(pool: "_Pool", adp: bool) -> AttentionInfo:
    return AttentionInfo(
        kv_heads_per_rank=pool.heads,
        tokens_per_block=pool.tpb,
        dims_per_head=pool.array.shape[-1],
        element_bytes=pool.array.itemsize,
        enable_attention_dp=adp,
        is_mla=False,
    )


def _rank_info(name, rank, *, tp_size, tp_rank, dp_size, adp, target, draft) -> RankInfo:
    ri = RankInfo(
        instance_name=name,
        instance_rank=rank,
        tp_size=tp_size,
        tp_rank=tp_rank,
        pp_size=1,
        pp_rank=0,
        layer_num_per_pp=[len(_LAYER_IDS)],
        sender_endpoints=[],
        self_endpoint="",
        transfer_engine_info=b"",
        dp_size=dp_size,
        dp_rank=tp_rank if adp else 0,
        attention=_attention(target, adp),
        page_table=target.page_table(),
    )
    if draft is not None:
        ri.draft_attention = _attention(draft, adp)
        ri.draft_page_table = draft.page_table()
    return ri


def _positional(num_blocks: int, first_held: int, slot_of) -> np.ndarray:
    table = np.full(num_blocks, -1, dtype=np.int64)
    for ordinal in range(first_held, num_blocks):
        table[ordinal] = slot_of(ordinal)
    return table


class _Context:
    """One context rank of CTX TP2.

    ``ctx_tp`` 1 is the attention-DP layout: one TP rank per DP group holds every head (rank 0
    modelled). ``ctx_tp`` > 1 is the TP-attention layout: rank ``ctx_rank`` holds its slice of
    every request's heads, ``heads // ctx_tp`` of the target's and of the drafter's.
    """

    def __init__(
        self,
        prompt_len: int,
        with_draft: bool = True,
        reuse_prefix: int = 0,
        ctx_tp: int = 1,
        ctx_rank: int = 0,
        index_dim: int = 0,
    ):
        """Model one context rank's pools after its prefill.

        ``reuse_prefix`` > 0 is a prefix-reuse hit (turn 2 of a conversation): the target prefill
        starts there, so the unpaired draft pool's history is that position and the drafter
        stored only the tail ``[reuse_prefix, prompt_len)``.
        """
        self.prompt_len = prompt_len
        self.ctx_tp = ctx_tp
        self.ctx_rank = ctx_rank
        n_target = _blocks(prompt_len, _TARGET_TPB)
        n_draft = _blocks(prompt_len, _DRAFT_TPB)
        target_heads = _TARGET_HEADS // ctx_tp
        draft_heads = _DRAFT_HEADS // ctx_tp
        self.target = _Pool(
            n_target + 5,
            target_heads,
            _TARGET_TPB,
            _TARGET_DIM,
            head_offset=ctx_rank * target_heads,
            index_dim=index_dim,
        )
        self.draft = _Pool(
            n_draft + 7,
            draft_heads,
            _DRAFT_TPB,
            _DRAFT_DIM,
            _DRAFT_WINDOW,
            head_offset=ctx_rank * draft_heads,
        )
        self.target_table = _positional(n_target, 0, lambda o: o + 5)
        last_chunk_start = max(reuse_prefix, prompt_len - _CTX_LAST_CHUNK)
        self.draft_table = _positional(
            n_draft, _window_start_block(last_chunk_start, _DRAFT_TPB), lambda o: o + 7
        )
        self.target.fill(self.target_table)
        self.draft.fill(self.draft_table)
        if reuse_prefix:
            # Positions below the reused prefix were never stored by the drafter.
            for ordinal, slot in enumerate(self.draft_table):
                low = ordinal * _DRAFT_TPB
                if slot >= 0 and low < reuse_prefix:
                    self.draft.array[slot][:, :, :, : min(reuse_prefix - low, _DRAFT_TPB)] = (
                        _SENTINEL
                    )
        adp = ctx_tp == 1
        self.rank_info = _rank_info(
            "ctx",
            ctx_rank,
            tp_size=2,
            tp_rank=ctx_rank,
            dp_size=2 if adp else 1,
            adp=adp,
            target=self.target,
            draft=self.draft if with_draft else None,
        )
        self.sender = Sender.__new__(Sender)
        self.sender._registrar = PeerRegistrar(
            self.rank_info, KVRegionExtractorV1(self.rank_info.page_table)
        )
        if with_draft:
            draft_ri = self.rank_info.draft_view()
            self.sender._draft_registrar = PeerRegistrar(
                draft_ri, KVRegionExtractorV1(draft_ri.page_table)
            )

    def register(self, gen_ri: RankInfo) -> None:
        """What the sender's REGISTER_RANK_INFO handler does, minus the transfer agent."""
        gen_ri = RankInfo.from_bytes(gen_ri.to_bytes())  # as the wire delivers it
        peer_draft_ri = check_draft_kv_peer(self.sender._draft_registrar, gen_ri)
        self.sender._registrar.register(gen_ri.instance_name, gen_ri.instance_rank, gen_ri)
        if peer_draft_ri is not None:
            self.sender._register_draft_peer(gen_ri, peer_draft_ri)
        else:
            self.sender._note_draft_peer_not_engaged(gen_ri)

    def chunk(self, *, with_draft: bool = True, is_last: bool = True) -> Chunk:
        return Chunk(
            block_ids_per_layer_groups=[self.target_table],
            kind_per_layer_group=[CacheKind.PAGED],
            token_range=TokenRange(start=0, end=self.prompt_len),
            is_last=is_last,
            draft_block_ids_per_layer_groups=[self.draft_table] if with_draft else None,
        )

    def task(self, *, with_draft: bool = True, is_last: bool = True, chunk=None) -> KVSendTask:
        return KVSendTask(
            chunk if chunk is not None else self.chunk(with_draft=with_draft, is_last=is_last),
            DisaggregatedParams(disagg_request_id=_RID),
            slice_id=0,
            prompt_len=self.prompt_len,
        )


class _Generation:
    """One generation TP rank (no attention DP): 1 drafter head, target heads split by TP."""

    def __init__(
        self,
        prompt_len: int,
        tp_size: int,
        tp_rank: int,
        with_draft: bool = True,
        index_dim: int = 0,
        draft_dim: int = _DRAFT_DIM,
        draft_dtype=np.int16,
    ):
        n_target = _blocks(prompt_len, _TARGET_TPB)
        n_draft = _blocks(prompt_len, _DRAFT_TPB)
        self.target = _Pool(
            n_target + 2,
            max(1, _TARGET_HEADS // tp_size),
            _TARGET_TPB,
            _TARGET_DIM,
            index_dim=index_dim,
        )
        self.draft = _Pool(
            n_draft + 3,
            max(1, _DRAFT_HEADS // tp_size),
            _DRAFT_TPB,
            draft_dim,
            _DRAFT_WINDOW,
            dtype=draft_dtype,
        )
        # Reversed slot order, so a pointer-arithmetic slip cannot pass by accident.
        self.target_table = _positional(n_target, 0, lambda o: n_target + 1 - o)
        self.draft_table = _positional(
            n_draft, _window_start_block(prompt_len, _DRAFT_TPB), lambda o: n_draft + 2 - o
        )
        self.rank_info = _rank_info(
            "gen",
            tp_rank,
            tp_size=tp_size,
            tp_rank=tp_rank,
            dp_size=1,
            adp=False,
            target=self.target,
            draft=self.draft if with_draft else None,
        )

    def chunk(self, prompt_len: int, *, with_draft: bool = True) -> Chunk:
        return Chunk(
            block_ids_per_layer_groups=[self.target_table],
            kind_per_layer_group=[CacheKind.PAGED],
            token_range=TokenRange(start=0, end=prompt_len),
            is_last=True,
            draft_block_ids_per_layer_groups=[self.draft_table] if with_draft else None,
        )

    def req_info(self, *, with_draft: bool = True, bounce_dst_base=None) -> RecvReqInfo:
        info = RecvReqInfo(
            sender_req_id=_RID,
            instance_name=self.rank_info.instance_name,
            instance_rank=self.rank_info.instance_rank,
            block_ids_per_layer_groups=[self.target_table],
            unique_rid=_RID,
            bounce_dst_base=bounce_dst_base,
            draft_block_ids_per_layer_groups=[self.draft_table] if with_draft else None,
        )
        return RecvReqInfo.from_bytes(info.to_bytes())  # as the wire delivers it


def _apply(write_meta) -> None:
    for src, dst, size in zip(write_meta.src_ptrs, write_meta.dst_ptrs, write_meta.sizes):
        ctypes.memmove(int(dst), int(src), int(size))


def _chunk_num_bytes(gen: _Generation, chunk: Chunk) -> int:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._page_table = gen.rank_info.page_table
    transceiver._draft_page_table = gen.rank_info.draft_page_table
    return transceiver._chunk_num_bytes(chunk)


@pytest.mark.parametrize("prompt_len", PROMPT_LENS)
@pytest.mark.parametrize("gen_tp", [4, 8])
def test_draft_window_lands_head_sliced_on_every_generation_rank(gen_tp, prompt_len):
    ctx = _Context(prompt_len)
    for rank in range(gen_tp):
        gen = _Generation(prompt_len, gen_tp, rank)
        ctx.register(gen.rank_info)

        write_meta = ctx.sender._build_kv_write_meta(ctx.task(), gen.req_info())
        _apply(write_meta)

        # The fan-out reaches every generation rank of the DP group, target and draft alike.
        assert write_meta.expected_transfers == gen_tp
        head = rank if gen_tp == 4 else rank // 2  # duplicate_kv_weight at TP8
        common = (ctx.draft_table >= 0) & (gen.draft_table >= 0)
        assert common.any()
        for ordinal in np.flatnonzero(common):
            received = gen.draft.array[gen.draft_table[ordinal]]
            stored = ctx.draft.array[ctx.draft_table[ordinal]][:, :, head : head + 1]
            assert np.array_equal(received, stored), f"ordinal {ordinal}, rank {rank}"
            layer, kv, _, t, d = np.indices(received.shape)
            assert np.array_equal(received, _value(layer, kv, head, ordinal * _DRAFT_TPB + t, d))
        # Released or never-held pages, and every slot no ordinal names, are untouched.
        written = {int(gen.draft_table[o]) for o in np.flatnonzero(common)}
        for slot in range(gen.draft.array.shape[0]):
            if slot not in written:
                assert (gen.draft.array[slot] == _SENTINEL).all(), f"slot {slot} was written"

        # Exact byte accounting: the receiver's expected-write total is what landed.
        assert _chunk_num_bytes(gen, gen.chunk(prompt_len)) == int(write_meta.sizes.sum())


@pytest.mark.parametrize("prompt_len", PROMPT_LENS)
@pytest.mark.parametrize("gen_tp", [4, 8])
def test_target_fragments_are_unchanged_by_the_draft_set(gen_tp, prompt_len):
    rank = gen_tp - 1
    on, off = _Context(prompt_len), _Context(prompt_len, with_draft=False)
    gen_on = _Generation(prompt_len, gen_tp, rank)
    gen_off = _Generation(prompt_len, gen_tp, rank, with_draft=False)
    on.register(gen_on.rank_info)
    off.register(gen_off.rank_info)

    meta_on = on.sender._build_kv_write_meta(on.task(), gen_on.req_info())
    meta_off = off.sender._build_kv_write_meta(
        off.task(with_draft=False), gen_off.req_info(with_draft=False)
    )

    # The target fragments come first and are the feature-off fragments, rebased onto the other
    # pools' (distinct host) addresses; the draft fragments follow, all inside the draft pools.
    n = meta_off.src_ptrs.size
    assert meta_on.src_ptrs.size > n
    rebase_src = on.rank_info.page_table.pool_groups[0].pools[0].base_address - (
        off.rank_info.page_table.pool_groups[0].pools[0].base_address
    )
    rebase_dst = gen_on.target.array.ctypes.data - gen_off.target.array.ctypes.data
    assert np.array_equal(meta_on.src_ptrs[:n], meta_off.src_ptrs + rebase_src)
    assert np.array_equal(meta_on.dst_ptrs[:n], meta_off.dst_ptrs + rebase_dst)
    assert np.array_equal(meta_on.sizes[:n], meta_off.sizes)
    assert meta_on.expected_transfers == meta_off.expected_transfers
    draft_lo = gen_on.draft.array.ctypes.data
    draft_hi = draft_lo + gen_on.draft.array.nbytes
    assert ((meta_on.dst_ptrs[n:] >= draft_lo) & (meta_on.dst_ptrs[n:] < draft_hi)).all()

    _apply(meta_on)
    _apply(meta_off)
    assert np.array_equal(gen_on.target.array, gen_off.target.array)
    assert _chunk_num_bytes(gen_off, gen_off.chunk(prompt_len, with_draft=False)) == int(
        meta_off.sizes.sum()
    )


def test_draft_set_ships_only_with_the_last_piece():
    ctx = _Context(4097)
    gen = _Generation(4097, 4, 1)
    ctx.register(gen.rank_info)
    last = ctx.sender._build_kv_write_meta(ctx.task(), gen.req_info())
    middle = ctx.sender._build_kv_write_meta(ctx.task(is_last=False), gen.req_info())

    lo = gen.draft.array.ctypes.data
    hi = lo + gen.draft.array.nbytes

    def into_draft_pool(write_meta):
        return (write_meta.dst_ptrs >= lo) & (write_meta.dst_ptrs < hi)

    assert into_draft_pool(last).any()
    assert not into_draft_pool(middle).any()
    assert middle.src_ptrs.size == int((~into_draft_pool(last)).sum())


def _target_only_meta(prompt_len, gen_tp, rank, **req_kwargs):
    off = _Context(prompt_len, with_draft=False)
    gen_off = _Generation(prompt_len, gen_tp, rank, with_draft=False)
    off.register(gen_off.rank_info)
    return (
        off,
        gen_off,
        off.sender._build_kv_write_meta(
            off.task(with_draft=False), gen_off.req_info(with_draft=False, **req_kwargs)
        ),
    )


def _same_target_write(meta, ref, ctx, ref_ctx, gen, ref_gen):
    rebase_src = ctx.target.array.ctypes.data - ref_ctx.target.array.ctypes.data
    rebase_dst = gen.target.array.ctypes.data - ref_gen.target.array.ctypes.data
    return (
        np.array_equal(meta.src_ptrs, ref.src_ptrs + rebase_src)
        and np.array_equal(meta.dst_ptrs, ref.dst_ptrs + rebase_dst)
        and np.array_equal(meta.sizes, ref.sizes)
        and meta.expected_transfers == ref.expected_transfers
    )


def test_a_bounce_reservation_leaves_the_window_out_not_the_request():
    ctx = _Context(4097)
    gen = _Generation(4097, 4, 0)
    ctx.register(gen.rank_info)
    meta = ctx.sender._build_kv_write_meta(ctx.task(), gen.req_info(bounce_dst_base=0x1000))
    ref_ctx, ref_gen, ref = _target_only_meta(4097, 4, 0, bounce_dst_base=0x1000)
    assert _same_target_write(meta, ref, ctx, ref_ctx, gen, ref_gen)
    assert meta.bounce_dst_base == 0x1000


def test_an_overlap_mismatch_leaves_the_window_out_not_the_request(monkeypatch):
    ctx = _Context(4097)
    gen = _Generation(4097, 4, 0)
    ctx.register(gen.rank_info)
    monkeypatch.setattr(
        ctx.sender._draft_registrar,
        "get_peer_overlap",
        lambda *_args: PeerOverlap(ranks=[0, 1]),
    )
    meta = ctx.sender._build_kv_write_meta(ctx.task(), gen.req_info())
    ref_ctx, ref_gen, ref = _target_only_meta(4097, 4, 0)
    assert _same_target_write(meta, ref, ctx, ref_ctx, gen, ref_gen)


@pytest.mark.parametrize("ctx_has_draft", [True, False])
def test_one_sided_draft_set_registers_the_target_and_disables_the_window(ctx_has_draft):
    ctx = _Context(64, with_draft=ctx_has_draft)
    gen = _Generation(64, 4, 0, with_draft=not ctx_has_draft)
    ctx.register(gen.rank_info)  # never raises
    assert ctx.sender._registrar.get_peer_rank_info("gen", 0) is not None
    assert ctx.sender._draft_disabled_peers["gen0"][0] == "peer_not_engaged"
    assert check_draft_kv_peer(getattr(ctx.sender, "_draft_registrar", None), gen.rank_info) is None
    assert draft_peer_not_engaged(None, gen.rank_info if not ctx_has_draft else ctx.rank_info)


def test_receiver_draft_set_without_a_sender_draft_registrar_sends_the_target_only():
    """A receiver that still asks for the window gets the target set only, not a failure.

    (It never registered the sender's verdict.)
    """
    prompt_len = 4097
    ctx = _Context(prompt_len, with_draft=False)
    gen = _Generation(prompt_len, 4, 0, with_draft=False)
    ctx.register(gen.rank_info)
    counters = DraftKvCounters("context")
    ctx.sender._draft_kv_counters = counters
    meta = ctx.sender._build_kv_write_meta(ctx.task(with_draft=False), gen.req_info())
    ref_ctx, ref_gen, ref = _target_only_meta(prompt_len, 4, 0)
    assert _same_target_write(meta, ref, ctx, ref_ctx, gen, ref_gen)
    assert counters.fallback == {"peer_not_engaged": 1}


def test_target_only_rank_info_has_no_draft_view():
    gen = _Generation(64, 4, 0, with_draft=False)
    assert gen.rank_info.draft_view() is None
    with_draft = _Generation(64, 4, 0).rank_info
    view = with_draft.draft_view()
    assert view.attention == with_draft.draft_attention
    assert view.page_table is with_draft.draft_page_table
    assert view.draft_view() is None
    assert replace(view, draft_attention=None) == view
    assert (view.tp_size, view.dp_size, view.tp_rank) == (
        with_draft.tp_size,
        with_draft.dp_size,
        with_draft.tp_rank,
    )


@pytest.mark.parametrize("prompt_len", [33, 4097, 100_013])
@pytest.mark.parametrize("gen_tp", [4, 8])
def test_hook_off_write_meta_is_the_target_only_write(gen_tp, prompt_len):
    """With no draft set on either side the write is exactly the target pool pair's fragments.

    That is also the kill-switch case (TRTLLM_M31_DISABLE_DISAGG_DRAFT_KV): neither rank carries a
    draft set, including on a 100K prompt whose draft mirror the generation side window-sizes.
    """
    rank = gen_tp - 1
    ctx = _Context(prompt_len, with_draft=False)
    gen = _Generation(prompt_len, gen_tp, rank, with_draft=False)
    ctx.register(gen.rank_info)
    info = gen.req_info(with_draft=False)
    assert info.draft_block_ids_per_layer_groups is None

    write_meta = ctx.sender._build_kv_write_meta(ctx.task(with_draft=False), info)

    registrar = ctx.sender._registrar
    peer = registrar.get_peer_rank_info("gen", rank)
    targets = registrar.get_peer_overlap(peer, peer.dp_rank)
    ((self_key, peer_key),) = registrar.get_pool_mapping(peer).items()
    assert registrar.should_send_pool(targets, peer, *self_key)
    src, dst = Sender._pair_ordinals(ctx.target_table, gen.target_table)
    pair = registrar.get_kv_map(peer, self_key, peer_key).map(
        registrar.self_extractor.extract(src, *self_key),
        registrar.peer_extractor("gen", rank).extract(dst, *peer_key),
    )
    assert np.array_equal(write_meta.src_ptrs, pair.src.memory.ptrs)
    assert np.array_equal(write_meta.dst_ptrs, pair.dst.memory.ptrs)
    assert (write_meta.sizes == pair.src.memory.bytes_per_region).all()
    assert write_meta.expected_transfers == len(targets.ranks) == gen_tp


# ---------------------------------------------------------------------------
# The context side's per-request decision, end to end through the write
# ---------------------------------------------------------------------------


class _StoreEvent:
    def synchronize(self):
        pass


class _Worker:
    """The drafter state the context side's forward left for the request, and the seeds."""

    def __init__(self, ctx_len=None, ctx_base=0):
        self.ready = {} if ctx_len is None else {_RID: (_StoreEvent(), ctx_len, ctx_base)}
        self.seeds = {}

    def take_disagg_draft_ctx_ready(self, request_id):
        return self.ready.pop(request_id, None)

    def seed_disagg_draft_ctx(self, request_id, ctx_len, ctx_base, prompt_len, on_seed=None):
        self.seeds[request_id] = (ctx_len, ctx_base, prompt_len)
        self.hooks = getattr(self, "hooks", {})
        if on_seed is not None:
            self.hooks[request_id] = on_seed

    def first_step(self, request_id):
        """What DFlashSpecMetadata.prepare does once the request holds its slot."""
        hook = getattr(self, "hooks", {}).pop(request_id, None)
        if hook is not None:
            hook()


def _transceiver(rank_info, worker) -> KvCacheTransceiverV2:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._draft_ctx_owner = worker
    transceiver._send_before_draft_ctx_event = False
    transceiver._draft_page_table = rank_info.draft_page_table
    transceiver._draft_kv_ctx_counters = DraftKvCounters("context")
    transceiver._draft_kv_gen_counters = DraftKvCounters("generation")
    return transceiver


def _request(prompt_len, style, **kwargs):
    return SimpleNamespace(
        py_request_id=_RID,
        prompt_len=prompt_len,
        is_generation_only_request=False,
        py_disaggregated_params=DisaggregatedParams(disagg_request_id=_RID, schedule_style=style),
        **kwargs,
    )


def _context_decision(ctx, worker, style):
    transceiver = _transceiver(ctx.rank_info, worker)
    request = _request(ctx.prompt_len, style)
    extent = transceiver._await_draft_ctx(request, CacheExtent(name=_RID, local=ctx.chunk()))
    return transceiver, request, extent.local


def _generation_request(prompt_len, style, ctx_request):
    """The generation request as the context side's response (or aux slot) delivers it."""
    if style == DisaggScheduleStyle.CONTEXT_FIRST:
        params = DisaggregatedParams(
            disagg_request_id=_RID,
            schedule_style=style,
            draft_ctx_state=encode_draft_ctx_state(*ctx_request.py_draft_ctx_aux),
        )
        return SimpleNamespace(
            py_request_id=_RID, prompt_len=prompt_len, py_disaggregated_params=params
        )
    params = DisaggregatedParams(disagg_request_id=_RID, schedule_style=style)
    return SimpleNamespace(
        py_request_id=_RID,
        prompt_len=prompt_len,
        py_disaggregated_params=params,
        py_draft_ctx_aux=ctx_request.py_draft_ctx_aux,
    )


@pytest.mark.parametrize("style", list(DisaggScheduleStyle))
@pytest.mark.parametrize("prompt_len, reuse_prefix", [(33, 16), (4097, 1000), (9000, 8200)])
@pytest.mark.parametrize("gen_tp", [4, 8])
def test_reuse_hit_window_lands_bitwise_with_its_base(gen_tp, prompt_len, reuse_prefix, style):
    """Context-side reuse hit: the draft window and its base travel, rows < ctx_len match.

    Turn 2 of a conversation: the target reuses a cached prefix, so the unpaired draft pool
    stored only the tail from the reused position on and the drafter reads past
    ``ctx_base = ceil(reuse_prefix / page)`` pages. The generation side receives the window's
    pages bitwise (head-sliced) and seeds the same ``(ctx_len, ctx_base)``, so every row its
    block decode reads, ``[max(base * page, prompt_len - window + 1), prompt_len)``, is the
    context side's.
    """
    ctx = _Context(prompt_len, reuse_prefix=reuse_prefix)
    base = _blocks(reuse_prefix, _DRAFT_TPB)
    transceiver, ctx_request, chunk = _context_decision(ctx, _Worker(prompt_len, base), style)
    assert transceiver._draft_kv_ctx_counters.sent == 1
    assert not transceiver._draft_kv_ctx_counters.fallback
    assert ctx_request.py_draft_ctx_aux == (prompt_len, base, True)
    assert chunk.draft_block_ids_per_layer_groups is not None

    gen_request = _generation_request(prompt_len, style, ctx_request)
    assert KvCacheTransceiverV2._received_draft_ctx_state(gen_request) == (prompt_len, base)
    for rank in range(gen_tp):
        gen = _Generation(prompt_len, gen_tp, rank)
        ctx.register(gen.rank_info)
        meta = ctx.sender._build_kv_write_meta(ctx.task(chunk=chunk), gen.req_info())
        _apply(meta)
        head = rank if gen_tp == 4 else rank // 2
        common = (ctx.draft_table >= 0) & (gen.draft_table >= 0)
        assert (gen.draft_table >= 0).sum() == common.sum()  # the context side covers it all
        for ordinal in np.flatnonzero(common):
            received = gen.draft.array[gen.draft_table[ordinal]]
            stored = ctx.draft.array[ctx.draft_table[ordinal]][:, :, head : head + 1]
            assert np.array_equal(received, stored), f"ordinal {ordinal}, rank {rank}"
            # Rows the drafter stored (position >= reuse_prefix) are f(...) exactly.
            layer, kv, _, t, d = np.indices(received.shape)
            pos = ordinal * _DRAFT_TPB + t
            expected = _value(layer, kv, head, pos, d)
            mask = pos >= reuse_prefix
            assert np.array_equal(received[mask], expected[mask])
        assert _chunk_num_bytes(gen, gen.chunk(prompt_len)) == int(meta.sizes.sum())

    worker = _Worker()
    gen_transceiver = _transceiver(gen.rank_info, worker)
    gen_transceiver._seed_draft_ctx(gen_request)
    assert worker.seeds[_RID] == (prompt_len, base, prompt_len)
    assert gen_transceiver._draft_kv_gen_counters.sent == 1


@pytest.mark.parametrize("prompt_len", [33, 4097, 9000])
@pytest.mark.parametrize("gen_tp", [4, 8])
def test_unservable_state_completes_with_no_draft_set_in_context_first(gen_tp, prompt_len):
    """An unservable context state completes with an empty window and the target-only write.

    Here the base lies past the prompt's pages, which the generation side cannot reproduce.
    """
    ctx = _Context(prompt_len, reuse_prefix=prompt_len // 2)
    base = _blocks(prompt_len, _DRAFT_TPB) + 1
    transceiver, ctx_request, chunk = _context_decision(
        ctx, _Worker(prompt_len, base), DisaggScheduleStyle.CONTEXT_FIRST
    )
    assert transceiver._draft_kv_ctx_counters.fallback == {"reuse_base": 1}
    assert transceiver._draft_kv_ctx_counters.sent == 0
    assert chunk.draft_block_ids_per_layer_groups is None
    assert np.array_equal(chunk.block_ids_per_layer_groups[0], ctx.target_table)

    gen_request = _generation_request(prompt_len, DisaggScheduleStyle.CONTEXT_FIRST, ctx_request)
    assert KvCacheTransceiverV2._received_draft_ctx_state(gen_request) is None
    for rank in range(gen_tp):
        gen = _Generation(prompt_len, gen_tp, rank)
        ctx.register(gen.rank_info)
        meta = ctx.sender._build_kv_write_meta(
            ctx.task(chunk=chunk), gen.req_info(with_draft=False)
        )
        ref_ctx, ref_gen, ref = _target_only_meta(prompt_len, gen_tp, rank)
        assert _same_target_write(meta, ref, ctx, ref_ctx, gen, ref_gen)
        _apply(meta)
        assert (gen.draft.array == _SENTINEL).all()
        assert _chunk_num_bytes(gen, gen.chunk(prompt_len, with_draft=False)) == int(
            meta.sizes.sum()
        )
    worker = _Worker()
    _transceiver(gen.rank_info, worker)._seed_draft_ctx(gen_request)
    assert worker.seeds[_RID] == (prompt_len, _blocks(prompt_len, _DRAFT_TPB), prompt_len)


@pytest.mark.parametrize("prompt_len", [33, 4097, 9000])
def test_generation_first_fallback_still_moves_the_asked_bytes_but_seeds_empty(prompt_len):
    """Generation-first fallback: the asked pages travel, marked unusable.

    The generation side asks for the window before the context side decides, so the pages still
    move (the expected-byte total stays exact) and the aux slot marks them unusable.
    """
    ctx = _Context(prompt_len)
    transceiver, ctx_request, chunk = _context_decision(
        ctx, _Worker(prompt_len - 1, 0), DisaggScheduleStyle.GENERATION_FIRST
    )
    assert transceiver._draft_kv_ctx_counters.fallback == {"ctx_len_mismatch": 1}
    assert ctx_request.py_draft_ctx_aux == (0, 0, False)
    assert chunk.draft_block_ids_per_layer_groups is not None

    gen = _Generation(prompt_len, 4, 2)
    ctx.register(gen.rank_info)
    meta = ctx.sender._build_kv_write_meta(ctx.task(chunk=chunk), gen.req_info())
    assert _chunk_num_bytes(gen, gen.chunk(prompt_len)) == int(meta.sizes.sum())

    gen_request = _generation_request(prompt_len, DisaggScheduleStyle.GENERATION_FIRST, ctx_request)
    worker = _Worker()
    gen_transceiver = _transceiver(gen.rank_info, worker)
    gen_transceiver._seed_draft_ctx(gen_request)
    assert worker.seeds[_RID] == (prompt_len, _blocks(prompt_len, _DRAFT_TPB), prompt_len)
    assert gen_transceiver._draft_kv_gen_counters.fallback == {"no_ctx_state": 1}


@pytest.mark.parametrize("style", list(DisaggScheduleStyle))
def test_a_whole_prompt_store_ships_and_seeds_the_context_state(style):
    prompt_len = 4097
    ctx = _Context(prompt_len)
    transceiver, ctx_request, chunk = _context_decision(ctx, _Worker(prompt_len, 0), style)
    assert transceiver._draft_kv_ctx_counters.sent == 1
    assert not transceiver._draft_kv_ctx_counters.fallback
    assert chunk.draft_block_ids_per_layer_groups is not None
    gen_request = _generation_request(prompt_len, style, ctx_request)
    assert KvCacheTransceiverV2._received_draft_ctx_state(gen_request) == (prompt_len, 0)
    worker = _Worker()
    _transceiver(ctx.rank_info, worker)._seed_draft_ctx(gen_request)
    assert worker.seeds[_RID] == (prompt_len, 0, prompt_len)


# ---------------------------------------------------------------------------
# Test only: the draft-window dump knob (TRTLLM_M31_F0A_DUMP_DIR) and its compare script
# ---------------------------------------------------------------------------


def _compare_module():
    spec = importlib.util.spec_from_file_location(
        "f0a_g1d_compare", Path(__file__).with_name("f0a_g1d_compare.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dumping(transceiver, directory, rank, heads):
    transceiver._draft_dump_dir = str(directory)
    transceiver._draft_dump_rank = rank
    transceiver._draft_dump_heads = heads
    return transceiver


def _serving_context(contexts, gen):
    """The one context rank whose overlap reaches this generation rank (what its receiver asks)."""
    serving = [
        ctx
        for ctx in contexts
        if gen.rank_info.instance_rank
        in ctx.sender._registrar.get_peer_overlap(gen.rank_info, gen.rank_info.dp_rank).ranks
    ]
    assert len(serving) == 1, [ctx.ctx_rank for ctx in serving]
    receiver = PeerRegistrar(gen.rank_info, KVRegionExtractorV1(gen.rank_info.page_table))
    assert receiver.get_peer_overlap(contexts[0].rank_info, 0).ranks == [serving[0].ctx_rank]
    return serving[0]


def _dump_loopback(directory, gen_tp, prompt_len=4097, reuse_prefix=0, ctx_base=0, ctx_tp=1):
    """Ship one request from every CTX rank to every GEN rank with dumps on."""
    contexts = [
        _Context(prompt_len, reuse_prefix=reuse_prefix, ctx_tp=ctx_tp, ctx_rank=c)
        for c in range(ctx_tp)
    ]
    chunks, ctx_requests = [], []
    for ctx in contexts:
        ctx_transceiver = _dumping(
            _transceiver(ctx.rank_info, _Worker(prompt_len, ctx_base)),
            directory,
            5 if ctx_tp == 1 else ctx.ctx_rank,  # attention DP: whichever rank holds it
            ctx.draft.heads,
        )
        ctx_request = _request(prompt_len, DisaggScheduleStyle.CONTEXT_FIRST)
        chunks.append(
            ctx_transceiver._await_draft_ctx(
                ctx_request, CacheExtent(name=_RID, local=ctx.chunk())
            ).local
        )
        ctx_requests.append(ctx_request)
    # Context-first: the response the generation request carries is TP rank 0's.
    response = ctx_requests[0]
    shipped = response.py_draft_ctx_aux[2]
    for rank in range(gen_tp):
        gen = _Generation(prompt_len, gen_tp, rank)
        ctx = _serving_context(contexts, gen)
        ctx.register(gen.rank_info)
        _apply(
            ctx.sender._build_kv_write_meta(
                ctx.task(chunk=chunks[ctx.ctx_rank]), gen.req_info(with_draft=shipped)
            )
        )
        worker = _Worker()
        gen_transceiver = _dumping(_transceiver(gen.rank_info, worker), directory, rank, 1)
        gen_transceiver._draft_reuse_adapter = SimpleNamespace(
            tokens_per_block=_DRAFT_TPB,
            get_block_ordinals=lambda req, idx, lg, table=gen.draft_table: table,
        )
        gen_request = _generation_request(prompt_len, DisaggScheduleStyle.CONTEXT_FIRST, response)
        gen_request.is_generation_only_request = True
        gen_request.py_disaggregated_params.ctx_request_id = _RID
        gen_request.py_disaggregated_params.disagg_request_id = None
        gen_request.py_request_id = 9000 + rank  # local ids differ; the match id does not
        gen_request.request_id = gen_request.py_request_id
        gen_transceiver._seed_draft_ctx(gen_request)
        worker.first_step(gen_request.py_request_id)


def test_dump_knob_unset_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv(DUMP_DIR_ENV, raising=False)
    assert draft_dump_dir() is None
    assert draft_dump_dir({DUMP_DIR_ENV: ""}) is None
    ctx = _Context(4097)
    worker = _Worker(4097, 0)
    transceiver = _transceiver(ctx.rank_info, worker)
    request = _request(4097, DisaggScheduleStyle.CONTEXT_FIRST)
    transceiver._await_draft_ctx(request, CacheExtent(name=_RID, local=ctx.chunk()))
    gen_worker = _Worker()
    _transceiver(ctx.rank_info, gen_worker)._seed_draft_ctx(
        _generation_request(4097, DisaggScheduleStyle.CONTEXT_FIRST, request)
    )
    assert getattr(gen_worker, "hooks", {}) == {}
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("gen_tp", [4, 8])
def test_dumps_pair_and_the_compare_passes_then_catches_one_flip(tmp_path, gen_tp):
    _dump_loopback(tmp_path, gen_tp)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == sorted([f"ctx_5_{_RID}.pt"] + [f"gen_{r}_{_RID}.pt" for r in range(gen_tp)])

    ctx_dump = torch.load(tmp_path / f"ctx_5_{_RID}.pt", weights_only=False)
    assert ctx_dump["format"] == "f0a_g1d_v1" and ctx_dump["shipped"]
    assert (ctx_dump["ctx_len"], ctx_dump["ctx_base"], ctx_dump["prompt_len"]) == (4097, 0, 4097)
    assert ctx_dump["heads_per_rank"] == _DRAFT_HEADS
    assert len(ctx_dump["layers"]) == len(_LAYER_IDS)
    first = ctx_dump["layers"][0]
    assert first["k"].shape[1:] == (_DRAFT_HEADS, _DRAFT_TPB, _DRAFT_DIM * 2)

    compare = _compare_module()
    report = compare.compare(str(tmp_path), 1, gen_tp)
    assert report["verdict"] == "G1D_PASS", report
    entry = report["requests"][str(_RID)]
    assert entry["status"] == "compared" and entry["bitwise"]
    assert all(r["rows_compared"] > 0 and r["mismatches"] == 0 for r in entry["gen_ranks"].values())

    flipped = tmp_path / f"gen_{gen_tp - 1}_{_RID}.pt"
    dump = torch.load(flipped, weights_only=False)
    dump["layers"][1]["v"][-1, 0, 0, 0] ^= 1
    torch.save(dump, flipped)
    report = compare.compare(str(tmp_path), 1, gen_tp)
    assert report["verdict"] == "G1D_FAIL"
    assert report["requests"][str(_RID)]["gen_ranks"][str(gen_tp - 1)]["mismatches"] == 1

    out = tmp_path / "report.json"
    assert compare.main(["f0a_g1d_compare.py", str(tmp_path), "1", str(gen_tp), str(out)]) == 1
    assert json.loads(out.read_text())["verdict"] == "G1D_FAIL"


def test_a_fallback_is_listed_and_alone_does_not_pass(tmp_path):
    # A base past the prompt's pages cannot be served: the context side falls back.
    _dump_loopback(tmp_path, 4, prompt_len=4097, ctx_base=_blocks(4097, _DRAFT_TPB) + 1)
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"ctx_5_{_RID}.pt"]
    report = _compare_module().compare(str(tmp_path), 1, 4)
    assert report["requests"][str(_RID)]["status"] == "fallback"
    assert report["verdict"] == "G1D_FAIL"  # nothing compared


@pytest.mark.parametrize("gen_tp", [4, 8])
def test_reuse_hit_dumps_compare_bitwise_with_equal_base(tmp_path, gen_tp):
    reuse_prefix = 8200
    base = _blocks(reuse_prefix, _DRAFT_TPB)
    _dump_loopback(tmp_path, gen_tp, prompt_len=9000, reuse_prefix=reuse_prefix, ctx_base=base)
    report = _compare_module().compare(str(tmp_path), 1, gen_tp)
    assert report["verdict"] == "G1D_PASS", report
    entry = report["requests"][str(_RID)]
    assert (entry["ctx_len"], entry["ctx_base"]) == (9000, base)
    assert all(r["ctx_base_equal"] for r in entry["gen_ranks"].values())


# ---------------------------------------------------------------------------
# CTX TP-attention layout (tp2, no attention DP): each rank holds a slice of every request
# ---------------------------------------------------------------------------

_INDEX_DIM = 8
_TP_CASES = [(p, 0) for p in PROMPT_LENS] + [(33, 16), (4097, 1000), (9000, 8200)]


def _tp_contexts(prompt_len, ctx_tp=2, **kwargs):
    return [_Context(prompt_len, ctx_tp=ctx_tp, ctx_rank=c, **kwargs) for c in range(ctx_tp)]


def _global_head(gen_rank, local_head, gen_heads, gen_tp, total_heads):
    """The head generation rank ``gen_rank`` holds in ``local_head`` (the HND mapper's rule)."""
    return (gen_rank * gen_heads + local_head) * total_heads // (gen_tp * gen_heads) % total_heads


@pytest.mark.parametrize("style", list(DisaggScheduleStyle))
def test_a_valid_tp_attention_request_ships_on_every_ctx_rank(style):
    """Both CTX TP ranks decide from the same host state, so both ship: zero fallbacks."""
    for ctx in _tp_contexts(4097):
        transceiver, request, chunk = _context_decision(ctx, _Worker(4097, 0), style)
        assert transceiver._draft_kv_ctx_counters.sent == 1
        assert not transceiver._draft_kv_ctx_counters.fallback
        assert request.py_draft_ctx_aux == (4097, 0, True)
        assert chunk.draft_block_ids_per_layer_groups is not None


@pytest.mark.parametrize("prompt_len, reuse_prefix", _TP_CASES)
@pytest.mark.parametrize("gen_tp", [4, 8])
def test_tp_attention_ctx_window_lands_bitwise_on_every_generation_rank(
    gen_tp, prompt_len, reuse_prefix
):
    contexts = _tp_contexts(prompt_len, reuse_prefix=reuse_prefix)
    base = _blocks(reuse_prefix, _DRAFT_TPB)
    chunks = []
    for ctx in contexts:
        transceiver, request, chunk = _context_decision(
            ctx, _Worker(prompt_len, base), DisaggScheduleStyle.CONTEXT_FIRST
        )
        assert transceiver._draft_kv_ctx_counters.sent == 1
        assert not transceiver._draft_kv_ctx_counters.fallback
        assert request.py_draft_ctx_aux == (prompt_len, base, True)
        chunks.append(chunk)

    ctx_heads = contexts[0].draft.heads
    total_heads = ctx_heads * len(contexts)
    for rank in range(gen_tp):
        gen = _Generation(prompt_len, gen_tp, rank)
        ctx = _serving_context(contexts, gen)
        ctx.register(gen.rank_info)
        peer = ctx.sender._registrar.get_peer_rank_info("gen", rank)
        peer_draft = ctx.sender._draft_registrar.get_peer_rank_info("gen", rank)
        # The draft view keeps the rank topology, so the draft set reaches exactly the target's
        # peers: the overlap check never drops a valid window.
        target_ranks = ctx.sender._registrar.get_peer_overlap(peer, peer.dp_rank).ranks
        draft_ranks = ctx.sender._draft_registrar.get_peer_overlap(peer_draft, 0).ranks
        assert draft_ranks == target_ranks and rank in target_ranks
        assert len(target_ranks) == gen_tp // len(contexts)

        meta = ctx.sender._build_kv_write_meta(ctx.task(chunk=chunks[ctx.ctx_rank]), gen.req_info())
        assert meta.expected_transfers == len(target_ranks)
        _apply(meta)

        head = _global_head(rank, 0, gen.draft.heads, gen_tp, total_heads)
        local = head - ctx.ctx_rank * ctx_heads
        assert 0 <= local < ctx_heads, (rank, head, ctx.ctx_rank)
        common = (ctx.draft_table >= 0) & (gen.draft_table >= 0)
        assert (gen.draft_table >= 0).sum() == common.sum()
        for ordinal in np.flatnonzero(common):
            received = gen.draft.array[gen.draft_table[ordinal]]
            stored = ctx.draft.array[ctx.draft_table[ordinal]][:, :, local : local + 1]
            assert np.array_equal(received, stored), f"ordinal {ordinal}, rank {rank}"
            layer, kv, _, t, d = np.indices(received.shape)
            pos = ordinal * _DRAFT_TPB + t
            mask = pos >= reuse_prefix
            assert np.array_equal(received[mask], _value(layer, kv, head, pos, d)[mask])
        assert _chunk_num_bytes(gen, gen.chunk(prompt_len)) == int(meta.sizes.sum())


@pytest.mark.parametrize("prompt_len", [33, 4097])
@pytest.mark.parametrize("gen_tp", [4, 8])
def test_tp_attention_ctx_target_kv_and_replicated_index_k(gen_tp, prompt_len):
    """Target K/V head mapping and MSA index-K (REPLICATED) from a TP2 CTX without ADP.

    Every CTX rank holds the same index-K (one replicated vector per token); each GEN rank is
    written by exactly one CTX rank, its overlap source, and receives the full index-K bitwise.
    """
    contexts = _tp_contexts(prompt_len, with_draft=False, index_dim=_INDEX_DIM)
    assert np.array_equal(contexts[0].target.index, contexts[1].target.index)
    ctx_heads = contexts[0].target.heads
    total_heads = ctx_heads * len(contexts)
    for rank in range(gen_tp):
        gen = _Generation(prompt_len, gen_tp, rank, with_draft=False, index_dim=_INDEX_DIM)
        writers = []
        for ctx in contexts:
            ctx.register(gen.rank_info)
            peer = ctx.sender._registrar.get_peer_rank_info("gen", rank)
            if rank not in ctx.sender._registrar.get_peer_overlap(peer, peer.dp_rank).ranks:
                continue
            writers.append(ctx)
            meta = ctx.sender._build_kv_write_meta(
                ctx.task(with_draft=False), gen.req_info(with_draft=False)
            )
            assert meta.expected_transfers == gen_tp // len(contexts)
            _apply(meta)
            assert _chunk_num_bytes(gen, gen.chunk(prompt_len, with_draft=False)) == int(
                meta.sizes.sum()
            )
        # One writer per generation rank: no duplicate index-K writes.
        assert [w.ctx_rank for w in writers] == [_serving_context(contexts, gen).ctx_rank]
        ctx = writers[0]
        for ordinal, gen_slot in enumerate(gen.target_table):
            ctx_slot = ctx.target_table[ordinal]
            # Full index-K, bitwise, on every generation rank.
            assert np.array_equal(gen.target.index[gen_slot], ctx.target.index[ctx_slot])
            il, it, idim = np.indices(gen.target.index[gen_slot].shape)
            assert np.array_equal(
                gen.target.index[gen_slot], _index_value(il, ordinal * _TARGET_TPB + it, idim)
            )
            # Target K/V: this rank's heads, from the CTX rank that holds them.
            for local in range(gen.target.heads):
                head = _global_head(rank, local, gen.target.heads, gen_tp, total_heads)
                assert head // ctx_heads == ctx.ctx_rank
                received = gen.target.array[gen_slot][:, :, local]
                layer, kv, t, d = np.indices(received.shape)
                assert np.array_equal(
                    received, _value(layer, kv, head, ordinal * _TARGET_TPB + t, d)
                )


@pytest.mark.parametrize("gen_tp", [4, 8])
def test_tp_attention_dumps_compare_per_ctx_rank_and_catch_one_flip(tmp_path, gen_tp):
    _dump_loopback(tmp_path, gen_tp, prompt_len=9000, reuse_prefix=8200, ctx_base=257, ctx_tp=2)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == sorted(
        [f"ctx_{c}_{_RID}.pt" for c in range(2)] + [f"gen_{r}_{_RID}.pt" for r in range(gen_tp)]
    )
    compare = _compare_module()
    report = compare.compare(str(tmp_path), 2, gen_tp)
    assert report["verdict"] == "G1D_PASS", report
    entry = report["requests"][str(_RID)]
    assert sorted(entry["ctx_ranks"]) == [0, 1]
    assert all(r["rows_compared"] > 0 for r in entry["gen_ranks"].values())

    flipped = tmp_path / f"gen_0_{_RID}.pt"
    dump = torch.load(flipped, weights_only=False)
    dump["layers"][0]["k"][-1, 0, 3, 1] ^= 1
    torch.save(dump, flipped)
    report = compare.compare(str(tmp_path), 2, gen_tp)
    assert report["verdict"] == "G1D_FAIL"
    assert report["requests"][str(_RID)]["gen_ranks"]["0"]["mismatches"] == 1


# ---------------------------------------------------------------------------
# CTX TP-rank divergence: one CTX rank ships its slice, another does not
# ---------------------------------------------------------------------------


def _gen_receive_outcome(gen, transceiver, request, meta, aux_state):
    """What this generation rank's completed receive reports: its local drafter-window veto."""
    written = int(meta.sizes.sum())
    expected = _chunk_num_bytes(gen, gen.chunk(request.prompt_len, with_draft=bool(request.asks)))
    session = SimpleNamespace(
        kv_write_byte_counts=lambda: (written, expected), peek_draft_ctx=lambda: aux_state
    )
    return transceiver._local_draft_veto(request, session)


@pytest.mark.parametrize("style", list(DisaggScheduleStyle))
@pytest.mark.parametrize("shipping_rank", [0, 1])
@pytest.mark.parametrize("gen_tp", [4, 8])
def test_ctx_rank_divergence_falls_back_on_every_generation_rank(gen_tp, shipping_rank, style):
    """CTX TP ranks diverge: rank ``shipping_rank`` ships its slice, the other falls back.

    The other rank had no drafter-store event. No hang (the write completes with what was sent),
    no write outside the pages the generation rank named, the target set intact, every generation
    rank starts from an empty window, and the divergence is counted.
    """
    prompt_len = 4097
    contexts = _tp_contexts(prompt_len)
    decisions = []
    for ctx in contexts:
        worker = _Worker(prompt_len, 0) if ctx.ctx_rank == shipping_rank else _Worker()
        decisions.append(_context_decision(ctx, worker, style))
    assert decisions[shipping_rank][1].py_draft_ctx_aux[2]
    assert not decisions[1 - shipping_rank][1].py_draft_ctx_aux[2]
    # Context-first: the generation request carries TP rank 0's response.
    response = decisions[0][1]
    context_first = style == DisaggScheduleStyle.CONTEXT_FIRST

    outcomes, gens = {}, {}
    for rank in range(gen_tp):
        gen = _Generation(prompt_len, gen_tp, rank)
        ctx = _serving_context(contexts, gen)
        ctx_transceiver, ctx_request, chunk = decisions[ctx.ctx_rank]
        ctx.register(gen.rank_info)
        ctx.sender._draft_kv_counters = ctx_transceiver._draft_kv_ctx_counters
        gen_request = _generation_request(
            prompt_len, style, response if context_first else ctx_request
        )
        gen_request.is_generation_only_request = True
        gen_request.asks = (
            KvCacheTransceiverV2._received_draft_ctx_state(gen_request) is not None
            if context_first
            else True
        )
        gen_transceiver = _transceiver(gen.rank_info, _Worker())
        gen_transceiver._note_expected_draft_bytes(
            gen_request, gen.chunk(prompt_len, with_draft=gen_request.asks)
        )

        meta = ctx.sender._build_kv_write_meta(
            ctx.task(chunk=chunk), gen.req_info(with_draft=gen_request.asks)
        )
        # Never more than the pages this generation rank named: its own target and draft pools.
        pools = [gen.target.array, gen.draft.array]
        for dst, size in zip(meta.dst_ptrs, meta.sizes):
            assert any(
                pool.ctypes.data <= dst and dst + size <= pool.ctypes.data + pool.nbytes
                for pool in pools
            )
        _apply(meta)
        ref_ctx, ref_gen, ref = _target_only_meta(prompt_len, gen_tp, rank)
        n = ref.src_ptrs.size
        assert np.array_equal(meta.sizes[:n], ref.sizes)  # the target set, unaffected
        _apply(ref)
        assert np.array_equal(gen.target.array, ref_gen.target.array)

        aux = ctx_request.py_draft_ctx_aux
        outcomes[rank] = _gen_receive_outcome(gen, gen_transceiver, gen_request, meta, aux)
        gens[rank] = (gen, gen_transceiver, gen_request)

    # The vetoes ride the receive consensus: flagged on any generation rank, flagged on all.
    def allgather(_local):
        return [[[], [], [_RID], [_RID] if outcomes[r] else []] for r in range(gen_tp)]

    stand_in = object.__new__(KvCacheTransceiverV2)
    *_, flagged = KvCacheTransceiverV2._consensus_outcome(
        stand_in, [_RID], [], [], [_RID], allgather, True, locally_flagged=[]
    )
    # Context-first with the response's rank (0) falling back: every generation rank already asked
    # for nothing and starts empty, so nothing is vetoed; the shipping rank 1 counts it.
    receivers_follow_fallback = context_first and shipping_rank == 1
    assert flagged == (set() if receivers_follow_fallback else {_RID})
    for rank, (gen, gen_transceiver, gen_request) in gens.items():
        veto = outcomes[rank] or ("ctx_rank_divergence" if flagged else None)
        worker = gen_transceiver._draft_ctx_owner
        gen_transceiver._seed_draft_ctx(gen_request, veto)
        # A clean fallback: the drafter starts from an empty window on every rank.
        assert worker.seeds[_RID] == (prompt_len, _blocks(prompt_len, _DRAFT_TPB), prompt_len)
        assert gen_transceiver._draft_kv_gen_counters.sent == 0
    reasons = Counter()
    for _, gen_transceiver, _ in gens.values():
        reasons.update(gen_transceiver._draft_kv_gen_counters.fallback)
    if receivers_follow_fallback:
        assert reasons == {"no_ctx_state": gen_tp}
    else:
        assert reasons["ctx_rank_divergence"] > 0
    assert sum(reasons.values()) == gen_tp
    # The context rank whose decision the receivers did not follow counts it too: as a divergence
    # when it ships nothing the receivers asked for, as receiver_not_asked when it ships and the
    # receivers (following rank 0's fallback) asked for none. A rank without a drafter-store event
    # also counted its own fallback once, when it decided.
    if context_first:
        divergent = contexts[1]
        counters = decisions[1][0]._draft_kv_ctx_counters
        if shipping_rank == 0:
            expected = {"no_store_event": 1, "ctx_rank_divergence": gen_tp // len(contexts)}
        else:
            expected = {"receiver_not_asked": gen_tp // len(contexts)}
        assert counters.fallback == expected
        assert divergent.sender._draft_kv_counters is counters


# ---------------------------------------------------------------------------
# Registration-time draft validation (from the upstream per-view transfer, #20709): a mismatch
# disables the draft set for that peer pair only, both sides agreeing; the target is untouched
# ---------------------------------------------------------------------------

_CTX_ENDPOINT = "tcp://ctx:1"


def _receiver(gen, ctx, counters=None):
    """The generation rank's receiver after it fetched the context peer's rank info."""
    receiver = Receiver.__new__(Receiver)
    receiver._registrar = PeerRegistrar(
        gen.rank_info, KVRegionExtractorV1(gen.rank_info.page_table)
    )
    draft_ri = gen.rank_info.draft_view()
    if draft_ri is not None:
        receiver._draft_registrar = PeerRegistrar(
            draft_ri, KVRegionExtractorV1(draft_ri.page_table)
        )
    receiver._draft_kv_counters = counters
    receiver._check_draft_peer(_CTX_ENDPOINT, RankInfo.from_bytes(ctx.rank_info.to_bytes()))
    return receiver


def _receive(receiver, gen, prompt_len):
    """The receive's task and request info after the receiver's per-peer draft decision."""
    engaged = gen.rank_info.has_draft_kv
    task = SimpleNamespace(
        expected_write_bytes=_chunk_num_bytes(gen, gen.chunk(prompt_len, with_draft=engaged)),
        draft_disabled_reason=None,
    )
    info = gen.req_info(with_draft=engaged)
    params = DisaggregatedParams(disagg_request_id=_RID, ctx_info_endpoint=_CTX_ENDPOINT)
    receiver._drop_disabled_draft_window(task, info, params)
    return task, info


@pytest.mark.parametrize("draft", [False, True], ids=["target_only", "draft_set"])
@pytest.mark.parametrize("disabled_peers", [None, {_CTX_ENDPOINT: ("peer_not_engaged", "why")}])
def test_receive_without_a_ctx_endpoint_never_needs_it(draft, disabled_peers):
    """Gen-first and ADP dispatches carry no ctx_info_endpoint; the window decision never needs one.

    Looking it up on every receive raised "ctx_info_endpoint is required" and broke the stock
    receive path (test_transfer_ownership_regressions).
    """
    receiver = Receiver.__new__(Receiver)
    receiver._registrar = SimpleNamespace(
        self_rank_info=SimpleNamespace(draft_receive_disabled=None)
    )
    receiver._draft_registrar = None
    receiver._draft_kv_counters = None
    receiver._draft_disabled_peers = disabled_peers
    draft_ids = [[1, 2]] if draft else None
    info = SimpleNamespace(draft_block_ids_per_layer_groups=draft_ids, unique_rid=_RID)
    task = SimpleNamespace(expected_write_bytes=None, draft_disabled_reason=None)
    params = DisaggregatedParams(disagg_request_id=_RID)
    assert receiver._drop_disabled_draft_window(task, info, params) == draft_ids
    assert task.draft_disabled_reason is None


def _disabled_window_round_trip(ctx, gen, prompt_len, gen_tp, rank, reason):
    """Register both sides, move one request, and check the window-only fallback end to end."""
    counters = DraftKvCounters("context")
    ctx.sender._draft_kv_counters = counters
    ctx.register(gen.rank_info)  # never raises: the target registers, the window is disabled
    key = gen.rank_info.instance_name + str(gen.rank_info.instance_rank)
    assert ctx.sender._draft_disabled_peers[key][0] == reason
    assert ctx.sender._registrar.get_peer_rank_info("gen", rank) is not None

    receiver = _receiver(gen, ctx)
    task, info = _receive(receiver, gen, prompt_len)
    assert info.draft_block_ids_per_layer_groups is None
    assert task.draft_disabled_reason == reason
    assert task.expected_write_bytes == _chunk_num_bytes(
        gen, gen.chunk(prompt_len, with_draft=False)
    )

    meta = ctx.sender._build_kv_write_meta(ctx.task(), info)
    assert counters.fallback == {reason: 1}
    ref_ctx, ref_gen, ref = _target_only_meta(prompt_len, gen_tp, rank)
    assert _same_target_write(meta, ref, ctx, ref_ctx, gen, ref_gen)
    _apply(meta)
    _apply(ref)
    assert np.array_equal(gen.target.array, ref_gen.target.array)
    assert (gen.draft.array == _SENTINEL).all()
    assert int(meta.sizes.sum()) == task.expected_write_bytes  # exact: the write verifies

    # The generation rank drafts from an empty window, counted under the same reason.
    worker = _Worker()
    gen_transceiver = _transceiver(gen.rank_info, worker)
    session = SimpleNamespace(
        draft_disabled_reason=lambda: task.draft_disabled_reason,
        kv_write_byte_counts=lambda: (int(meta.sizes.sum()), task.expected_write_bytes),
        peek_draft_ctx=lambda: None,
    )
    request = _request(prompt_len, DisaggScheduleStyle.GENERATION_FIRST)
    request.py_draft_ctx_aux = (prompt_len, 0, True)
    veto = gen_transceiver._local_draft_veto(request, session)
    assert veto == reason
    gen_transceiver._seed_draft_ctx(request, veto)
    assert worker.seeds[_RID] == (prompt_len, _blocks(prompt_len, _DRAFT_TPB), prompt_len)
    assert gen_transceiver._draft_kv_gen_counters.fallback == {reason: 1}


@pytest.mark.parametrize("drift", ["head_dim", "dtype"])
@pytest.mark.parametrize("gen_tp", [4, 8])
def test_draft_geometry_drift_disables_only_the_window(gen_tp, drift):
    prompt_len = 4097
    rank = gen_tp - 1
    ctx = _Context(prompt_len)
    kwargs = {"draft_dim": 2 * _DRAFT_DIM} if drift == "head_dim" else {"draft_dtype": np.int32}
    gen = _Generation(prompt_len, gen_tp, rank, **kwargs)
    reason = draft_peer_mismatch(ctx.rank_info.draft_view(), gen.rank_info.draft_view())
    assert reason is not None and (
        "head_dim" in reason if drift == "head_dim" else "dtype" in reason
    )
    _disabled_window_round_trip(ctx, gen, prompt_len, gen_tp, rank, "peer_draft_geometry")


def _rename_roles(page_table, roles):
    page_table.layer_groups[0].pool_views[0].pool_role = frozenset(roles)


def _duplicate_view(page_table):
    group = page_table.layer_groups[0]
    group.pool_views.append(replace(group.pool_views[0]))


def _shift_layers(page_table):
    group = page_table.layer_groups[0]
    group.local_layers = [
        LocalLayer(ll.local_layer_id, ll.global_layer_id + 100) for ll in group.local_layers
    ]


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda pt: _rename_roles(pt, {"key", "value", "scale"}), "matches 0 peer views"),
        (_duplicate_view, "matches 2 peer views"),
        (_shift_layers, "draft layers"),
    ],
)
def test_each_draft_view_check_disables_the_window_on_both_sides(mutate, expected):
    prompt_len = 4097
    ctx = _Context(prompt_len)
    gen = _Generation(prompt_len, 4, 1)
    mutate(gen.rank_info.draft_page_table)
    why = draft_peer_mismatch(ctx.rank_info.draft_view(), gen.rank_info.draft_view())
    assert why is not None and expected in why
    # The receiver's view of the same pair reaches a verdict too.
    assert draft_peer_mismatch(gen.rank_info.draft_view(), ctx.rank_info.draft_view()) is not None
    _disabled_window_round_trip(ctx, gen, prompt_len, 4, 1, "peer_draft_geometry")


def test_draft_pipeline_parallelism_and_overlap_mismatch_are_refused():
    ctx = _Context(64)
    gen = _Generation(64, 4, 0)
    pp2 = replace(gen.rank_info.draft_view(), pp_size=2)
    assert "pipeline parallelism" in draft_peer_mismatch(ctx.rank_info.draft_view(), pp2)
    why = draft_peer_mismatch(
        ctx.rank_info.draft_view(), gen.rank_info.draft_view(), ([0, 1, 2, 3], [0, 1])
    )
    assert "overlap" in why
    assert draft_peer_mismatch(ctx.rank_info.draft_view(), gen.rank_info.draft_view()) is None


def test_a_draft_fragment_error_drops_only_the_window(monkeypatch):
    prompt_len = 4097
    ctx = _Context(prompt_len)
    gen = _Generation(prompt_len, 4, 2)
    ctx.register(gen.rank_info)
    counters = DraftKvCounters("context")
    ctx.sender._draft_kv_counters = counters

    def broken(*_args, **_kwargs):
        raise ValueError("HND bytes per head mismatch")

    monkeypatch.setattr(ctx.sender._draft_registrar, "get_kv_map", broken)
    meta = ctx.sender._build_kv_write_meta(ctx.task(), gen.req_info())  # completes
    assert counters.fallback == {"draft_fragment_error": 1}
    ref_ctx, ref_gen, ref = _target_only_meta(prompt_len, 4, 2)
    assert _same_target_write(meta, ref, ctx, ref_ctx, gen, ref_gen)
    # The receiver asked for the window, so its attestation is short by exactly that: it falls back.
    expected = _chunk_num_bytes(gen, gen.chunk(prompt_len))
    session = SimpleNamespace(
        kv_write_byte_counts=lambda: (int(meta.sizes.sum()), expected),
        peek_draft_ctx=lambda: None,
    )
    gen_transceiver = _transceiver(gen.rank_info, _Worker())
    request = _request(prompt_len, DisaggScheduleStyle.CONTEXT_FIRST)
    gen_transceiver._note_expected_draft_bytes(request, gen.chunk(prompt_len))
    assert gen_transceiver._local_draft_veto(request, session) == "ctx_rank_divergence"


# ---------------------------------------------------------------------------
# Synchronous generation receive with TP ranks: no window at all (sync_path_tp)
# ---------------------------------------------------------------------------


def test_sync_receive_with_tp_ranks_takes_no_window():
    tp4 = SimpleNamespace(tp_size=4, enable_attention_dp=False)
    assert draft_receive_disabled_reason(tp4, synchronous_receive=True) == "sync_path_tp"
    assert draft_receive_disabled_reason(tp4, synchronous_receive=False) is None
    adp = SimpleNamespace(tp_size=4, enable_attention_dp=True)
    assert draft_receive_disabled_reason(adp, synchronous_receive=True) is None
    tp1 = SimpleNamespace(tp_size=1, enable_attention_dp=False)
    assert draft_receive_disabled_reason(tp1, synchronous_receive=True) is None


def test_receive_disabled_rank_info_round_trips_and_is_absent_by_default():
    gen = _Generation(64, 4, 0)
    assert "draft_receive_disabled" not in msgpack.unpackb(
        gen.rank_info.to_bytes(), strict_map_key=False
    )
    gen.rank_info.draft_receive_disabled = "sync_path_tp"
    restored = RankInfo.from_bytes(gen.rank_info.to_bytes())
    assert restored.draft_receive_disabled == "sync_path_tp"
    assert restored.draft_view().draft_receive_disabled is None


@pytest.mark.parametrize("gen_tp", [4, 8])
def test_sync_receive_tp_ranks_all_seed_the_same_empty_window_with_a_divergent_ctx(gen_tp):
    """GEN TP on the synchronous receive path seeds one empty window on every rank.

    CTX TP2 with rank 1 diverging: every generation rank takes the target set only and seeds the
    same empty window; both sides count it.
    """
    prompt_len = 4097
    contexts = _tp_contexts(prompt_len)
    decisions = [
        _context_decision(
            ctx,
            _Worker(prompt_len, 0) if ctx.ctx_rank == 0 else _Worker(),
            DisaggScheduleStyle.CONTEXT_FIRST,
        )
        for ctx in contexts
    ]
    seeds = set()
    for rank in range(gen_tp):
        gen = _Generation(prompt_len, gen_tp, rank)
        gen.rank_info.draft_receive_disabled = "sync_path_tp"
        ctx = _serving_context(contexts, gen)
        ctx_counters = decisions[ctx.ctx_rank][0]._draft_kv_ctx_counters
        ctx.sender._draft_kv_counters = ctx_counters
        ctx.register(gen.rank_info)
        key = "gen" + str(rank)
        assert ctx.sender._draft_disabled_peers[key][0] == "sync_path_tp"
        receiver = _receiver(gen, ctx)
        task, info = _receive(receiver, gen, prompt_len)
        assert info.draft_block_ids_per_layer_groups is None
        assert task.draft_disabled_reason == "sync_path_tp"
        chunk = decisions[ctx.ctx_rank][2]
        meta = ctx.sender._build_kv_write_meta(ctx.task(chunk=chunk), info)  # no hang
        ref_ctx, ref_gen, ref = _target_only_meta(prompt_len, gen_tp, rank)
        assert np.array_equal(meta.sizes, ref.sizes)
        assert int(meta.sizes.sum()) == task.expected_write_bytes
        worker = _Worker()
        gen_transceiver = _transceiver(gen.rank_info, worker)
        session = SimpleNamespace(
            draft_disabled_reason=lambda t=task: t.draft_disabled_reason,
            kv_write_byte_counts=lambda m=meta, t=task: (
                int(m.sizes.sum()),
                t.expected_write_bytes,
            ),
            peek_draft_ctx=lambda: None,
        )
        request = _generation_request(
            prompt_len, DisaggScheduleStyle.CONTEXT_FIRST, decisions[0][1]
        )
        gen_transceiver._seed_draft_ctx(
            request, gen_transceiver._local_draft_veto(request, session)
        )
        seeds.add(worker.seeds[_RID])
        assert gen_transceiver._draft_kv_gen_counters.fallback == {"sync_path_tp": 1}
    assert seeds == {(prompt_len, _blocks(prompt_len, _DRAFT_TPB), prompt_len)}
    # Rank 0 shipped, so its sender counts the disabled peers; rank 1 fell back on its own.
    assert decisions[0][0]._draft_kv_ctx_counters.fallback["sync_path_tp"] == gen_tp // 2


# ---------------------------------------------------------------------------
# Mixed peers: only one side engages the drafter window
# ---------------------------------------------------------------------------
#
# E.g. a legacy (hook-off) MiniMax CTX TP2 serving a modeling_v2 (hook-on) GEN TP4 / TP8, or the
# reverse. The peer pair degrades to target-only transfers: no request fails, the target write
# is bitwise the target-only write, both sides count draft_kv_fallback{peer_not_engaged}, and an
# engaged generation side seeds the empty window at the prompt end.


def _target_only_reference(prompt_len, gen_tp, gen_rank, ctx_tp, ctx_rank):
    ref_ctx = _Context(prompt_len, with_draft=False, ctx_tp=ctx_tp, ctx_rank=ctx_rank)
    ref_gen = _Generation(prompt_len, gen_tp, gen_rank, with_draft=False)
    ref_ctx.register(ref_gen.rank_info)
    ref = ref_ctx.sender._build_kv_write_meta(
        ref_ctx.task(with_draft=False), ref_gen.req_info(with_draft=False)
    )
    return ref_ctx, ref_gen, ref


@pytest.mark.parametrize("prompt_len", [4097, 100_013])
@pytest.mark.parametrize("style", list(DisaggScheduleStyle))
@pytest.mark.parametrize("gen_tp", [4, 8])
@pytest.mark.parametrize("ctx_layout", ["adp", "tp_attention"])
@pytest.mark.parametrize("engaged", ["gen_only", "ctx_only"])
def test_mixed_peers_degrade_to_target_only(engaged, ctx_layout, gen_tp, style, prompt_len):
    ctx_engaged = engaged == "ctx_only"
    ctx_tp = 1 if ctx_layout == "adp" else 2
    contexts = [
        _Context(prompt_len, with_draft=ctx_engaged, ctx_tp=ctx_tp, ctx_rank=c)
        for c in range(ctx_tp)
    ]
    ctx_counters, chunks, ctx_requests = {}, {}, {}
    for ctx in contexts:
        if ctx_engaged:
            # The engaged context side ships the whole-prompt window as usual.
            transceiver, request, chunk = _context_decision(ctx, _Worker(prompt_len, 0), style)
            assert chunk.draft_block_ids_per_layer_groups is not None
            ctx_counters[ctx.ctx_rank] = transceiver._draft_kv_ctx_counters
            ctx_requests[ctx.ctx_rank] = request
        else:
            chunk = ctx.chunk(with_draft=False)
            ctx_counters[ctx.ctx_rank] = DraftKvCounters("context")
        ctx.sender._draft_kv_counters = ctx_counters[ctx.ctx_rank]
        chunks[ctx.ctx_rank] = chunk

    empty_window = (prompt_len, _blocks(prompt_len, _DRAFT_TPB), prompt_len)
    for rank in range(gen_tp):
        gen = _Generation(prompt_len, gen_tp, rank, with_draft=not ctx_engaged)
        ctx = _serving_context(contexts, gen) if ctx_tp > 1 else contexts[0]
        ctx.register(gen.rank_info)  # never raises
        assert ctx.sender._draft_disabled_peers[f"gen{rank}"][0] == "peer_not_engaged"

        gen_counters = DraftKvCounters("generation")
        receiver = _receiver(gen, ctx, counters=None if not ctx_engaged else gen_counters)
        assert receiver._draft_disabled_peers[_CTX_ENDPOINT][0] == "peer_not_engaged"
        task, info = _receive(receiver, gen, prompt_len)
        assert info.draft_block_ids_per_layer_groups is None
        target_bytes = _chunk_num_bytes(gen, gen.chunk(prompt_len, with_draft=False))
        assert task.expected_write_bytes == target_bytes

        meta = ctx.sender._build_kv_write_meta(ctx.task(chunk=chunks[ctx.ctx_rank]), info)
        ref_ctx, ref_gen, ref = _target_only_reference(
            prompt_len, gen_tp, rank, ctx_tp, ctx.ctx_rank
        )
        assert _same_target_write(meta, ref, ctx, ref_ctx, gen, ref_gen)
        _apply(meta)
        _apply(ref)
        assert np.array_equal(gen.target.array, ref_gen.target.array)  # bitwise target-only
        assert (gen.draft.array == _SENTINEL).all()  # no draft byte moved
        assert int(meta.sizes.sum()) == task.expected_write_bytes  # exact: the write verifies

        if ctx_engaged:
            # The generation rank does not engage: its receiver counts the request.
            assert gen_counters.fallback == {"peer_not_engaged": 1}
            continue
        # The engaged generation rank vetoes the window and seeds the empty one at the prompt end,
        # even over a stale drafter-context aux slot (the legacy peer never writes it).
        worker = _Worker()
        gen_transceiver = _transceiver(gen.rank_info, worker)
        session = SimpleNamespace(
            draft_disabled_reason=lambda t=task: t.draft_disabled_reason,
            kv_write_byte_counts=lambda m=meta, t=task: (
                int(m.sizes.sum()),
                t.expected_write_bytes,
            ),
            peek_draft_ctx=lambda: (prompt_len, 0, True),
        )
        request = _request(prompt_len, style)
        request.py_draft_ctx_aux = (prompt_len, 0, True)
        veto = gen_transceiver._local_draft_veto(request, session)
        assert veto == "peer_not_engaged"
        gen_transceiver._seed_draft_ctx(request, veto)
        assert worker.seeds[_RID] == empty_window
        assert gen_transceiver._draft_kv_gen_counters.fallback == {"peer_not_engaged": 1}

    # The context side counts every request toward a peer that does not engage the window
    # (an engaged one also counted it as decided-to-ship before the per-peer veto).
    assert sum(c.fallback["peer_not_engaged"] for c in ctx_counters.values()) == gen_tp
    for counters in ctx_counters.values():
        assert set(counters.fallback) <= {"peer_not_engaged"}


@pytest.mark.parametrize("sender_engaged", [True, False])
def test_mixed_peer_aux_write_moves_the_common_buffers_only(sender_engaged):
    """Generation-first: only the common auxiliary buffers travel between mixed peers.

    The engaged side's layout carries one extra drafter-context buffer; toward a peer without it
    the common buffers travel instead of failing on the layout mismatch.
    """
    from tensorrt_llm._torch.disaggregation.native.auxiliary import (
        AuxBufferMeta,
        build_aux_transfer_layout,
    )

    def aux_meta(base, engaged):
        sizes = [8, 32, 8, 8] + ([12] if engaged else [])
        ptrs = [base + 0x10000 * i for i in range(len(sizes))]
        return AuxBufferMeta(
            ptrs=np.array(ptrs, dtype=np.int64),
            size=np.array([s * 16 for s in sizes], dtype=np.int64),
            item_sizes=np.array(sizes, dtype=np.int64),
        )

    src = aux_meta(0x100000, sender_engaged)
    dst = aux_meta(0x900000, not sender_engaged)
    with pytest.raises(ValueError, match="do not match"):
        build_aux_transfer_layout(src, dst)
    self_ri = SimpleNamespace(has_draft_kv=sender_engaged, aux_meta=src)
    peer_ri = SimpleNamespace(
        has_draft_kv=not sender_engaged,
        aux_meta=dst,
        instance_name="gen",
        instance_rank=0,
        dp_rank=0,
        self_endpoint="",
    )
    cached = {}
    registrar = SimpleNamespace(
        self_rank_info=self_ri,
        get_peer_rank_info=lambda name, rank: peer_ri,
        get_peer_overlap=lambda ri, dp_rank: PeerOverlap(ranks=[0]),
        should_send_aux=lambda ri: True,
        get_aux_transfer_layout=lambda name, rank: cached.get((name, rank)),
        cache_aux_transfer_layout=lambda name, rank, layout: cached.__setitem__(
            (name, rank), layout
        ),
    )
    sender = Sender.__new__(Sender)
    sender._registrar = registrar
    task = SimpleNamespace(_perf_timer=None, _slot=2, _unique_rid=_RID)
    meta = sender._build_aux_write_meta(
        task, SimpleNamespace(instance_name="gen", instance_rank=0, aux_slot=3, unique_rid=_RID)
    )
    common = aux_meta(0x100000, False)
    assert np.array_equal(meta.sizes, common.item_sizes)
    assert np.array_equal(meta.src_ptrs, common.ptrs + common.item_sizes * 2)
    assert np.array_equal(meta.dst_ptrs, aux_meta(0x900000, False).ptrs + common.item_sizes * 3)
    trimmed = aux_meta_without_draft_ctx(aux_meta(0x100000, True))
    assert np.array_equal(trimmed.ptrs, common.ptrs)
    assert np.array_equal(trimmed.item_sizes, common.item_sizes)


def test_matched_peers_keep_the_aux_layout_unchanged():
    """Both engaged or both not: the auxiliary layout is passed through as before."""
    from tensorrt_llm._torch.disaggregation.native.auxiliary import AuxBufferMeta

    for engaged in (True, False):
        n = 5 if engaged else 4
        meta = AuxBufferMeta(
            ptrs=np.arange(1, n + 1, dtype=np.int64) * 0x10000,
            size=np.full(n, 64, dtype=np.int64),
            item_sizes=np.full(n, 4, dtype=np.int64),
        )
        ri = SimpleNamespace(
            has_draft_kv=engaged,
            aux_meta=meta,
            instance_name="gen",
            instance_rank=0,
            dp_rank=0,
            self_endpoint="",
        )
        registrar = SimpleNamespace(
            self_rank_info=ri,
            get_peer_rank_info=lambda name, rank, ri=ri: ri,
            get_peer_overlap=lambda r, d: PeerOverlap(ranks=[0]),
            should_send_aux=lambda r: True,
            get_aux_transfer_layout=lambda name, rank: None,
            cache_aux_transfer_layout=lambda *a: None,
        )
        sender = Sender.__new__(Sender)
        sender._registrar = registrar
        meta_out = sender._build_aux_write_meta(
            SimpleNamespace(_perf_timer=None, _slot=0, _unique_rid=_RID),
            SimpleNamespace(instance_name="gen", instance_rank=0, aux_slot=0, unique_rid=_RID),
        )
        assert meta_out.sizes.size == n
