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
"""The disagg drafter context window: switch, wire formats, send/receive decisions and loopback.

The loopback tests build the write with the real rank-info serialization, peer registrars,
extractors, head-mismatch mappers and ``Sender._build_kv_write_meta``, then apply it with host
memcpy to host-memory pools laid out like the V2 HND pools, so no NIXL agent or GPU is needed.
A context rank with attention DP holds all four drafter KV heads; generation TP4 holds head r on
rank r and TP8 one duplicated head (r // 2) per rank.
"""

import ctypes
from collections import Counter
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import msgpack
import numpy as np
import pytest
import torch

from tensorrt_llm import DisaggregatedParams
from tensorrt_llm._torch.disaggregation import transceiver as transceiver_module
from tensorrt_llm._torch.disaggregation.base import CacheKind, TokenRange
from tensorrt_llm._torch.disaggregation.draft_kv import (
    DISABLE_DISAGG_DRAFT_KV_ENV,
    DraftKvCounters,
    decode_draft_ctx_state,
    disagg_draft_kv_disabled,
    draft_receive_disabled_reason,
    encode_draft_ctx_state,
    select_disagg_draft_kv_cache_manager,
)
from tensorrt_llm._torch.disaggregation.native import rank_info as rank_info_module
from tensorrt_llm._torch.disaggregation.native.auxiliary import (
    AuxBuffer,
    AuxBufferMeta,
    build_aux_transfer_layout,
)
from tensorrt_llm._torch.disaggregation.native.extent import CacheExtent, Chunk
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
)
from tensorrt_llm._torch.disaggregation.orchestration.coordinator import DisaggTransferCoordinator
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
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import _response_draft_ctx_state
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks
from tensorrt_llm.disaggregated_params import DisaggScheduleStyle

pytestmark = pytest.mark.cpu_only

_CONTEXT_FIRST = DisaggScheduleStyle.CONTEXT_FIRST
_GENERATION_FIRST = DisaggScheduleStyle.GENERATION_FIRST

# ---------------------------------------------------------------------------
# The switch: the model's opt-in hook AND a separate draft pool AND the kill switch unset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [(None, False), ("1", True), ("On", True), ("", False), ("off", False), ("2", ValueError)],
)
def test_strict_bool_env_knobs(value, expected):
    env = {} if value is None else {DISABLE_DISAGG_DRAFT_KV_ENV: value}
    if expected is ValueError:
        with pytest.raises(ValueError, match=DISABLE_DISAGG_DRAFT_KV_ENV):
            disagg_draft_kv_disabled(env)
    else:
        assert disagg_draft_kv_disabled(env) is expected


class _SelectWorker:
    def engage_disagg_draft_kv(self):
        raise AssertionError("selection must not engage the worker")


def _model(hook: bool):
    model = SimpleNamespace(spec_worker=_SelectWorker())
    if hook:
        model.runtime_hooks = RuntimeHooks(disagg_draft_kv=True)
    return model


def _selection_manager(joint_reuse=False):
    return SimpleNamespace(enable_joint_kv_cache_reuse=joint_reuse)


def test_selected_only_with_hook_and_separate_pool(monkeypatch):
    monkeypatch.delenv(DISABLE_DISAGG_DRAFT_KV_ENV, raising=False)
    manager = _selection_manager()
    assert select_disagg_draft_kv_cache_manager(_model(hook=True), manager) is manager
    assert select_disagg_draft_kv_cache_manager(_model(hook=False), manager) is None
    assert select_disagg_draft_kv_cache_manager(SimpleNamespace(), manager) is None
    # No separate draft pool (one-model Eagle3 sharing the target's).
    assert select_disagg_draft_kv_cache_manager(_model(hook=True), None) is None
    # A worker with no paged drafter context to ship.
    no_ctx = SimpleNamespace(runtime_hooks=RuntimeHooks(disagg_draft_kv=True), spec_worker=object())
    assert select_disagg_draft_kv_cache_manager(no_ctx, manager) is None
    with pytest.raises(ValueError, match="enable_joint_kv_cache_reuse"):
        select_disagg_draft_kv_cache_manager(_model(hook=True), _selection_manager(True))


@pytest.mark.parametrize("value, raises", [("1", False), ("please", True)])
def test_kill_switch_wins_and_an_invalid_value_fails_selection(monkeypatch, value, raises):
    monkeypatch.setenv(DISABLE_DISAGG_DRAFT_KV_ENV, value)
    if raises:
        with pytest.raises(ValueError, match=DISABLE_DISAGG_DRAFT_KV_ENV):
            select_disagg_draft_kv_cache_manager(_model(hook=True), _selection_manager())
    else:
        assert select_disagg_draft_kv_cache_manager(_model(hook=True), _selection_manager()) is None


# ---------------------------------------------------------------------------
# Host-memory pools and ranks for the loopback
# ---------------------------------------------------------------------------

# Target and draft pools deliberately share role labels and global layer ids: one pool mapping
# would cross-match them, so the draft set must go through its own registrar.
_ROLES = frozenset({"key", "value"})
_LAYER_IDS = [0, 1]
_RID = 4242
_SENTINEL = -1

_TARGET_TPB = 64
_TARGET_DIM = 8
_TARGET_HEADS = 8
_DRAFT_TPB = 32
_DRAFT_DIM = 16
_DRAFT_HEADS = 4
_DRAFT_WINDOW = 2048
# The context side's last prefill chunk starts this far before the prompt end: its windowed draft
# pool still holds the window behind that chunk start, a superset of what generation keeps.
_CTX_LAST_CHUNK = 4096

PROMPT_LENS = [32, 33, 4097, 9000]


def _blocks(tokens: int, tpb: int) -> int:
    return (tokens + tpb - 1) // tpb


def _window_start_block(history: int, tpb: int) -> int:
    """First ordinal a windowed pool still holds at ``history`` (released below it)."""
    return max(0, (history + 1 - _DRAFT_WINDOW) // tpb)


def _value(layer, kv, head, pos, dim):
    """The K/V the context side stored: f(layer, K/V, global head, position, dim) as int16."""
    return ((layer * 2 + kv) * 1009 + head * 131 + pos * 7 + dim * 3) % 30011


class _Pool:
    """A host-memory paged pool shaped ``[slot, layer, K/V, head, token, dim]`` (V2 HND)."""

    def __init__(self, num_slots, heads, tpb, head_dim, window=None, head_offset=0, dtype=np.int16):
        self.window = window
        self.head_offset = head_offset
        self.array = np.full(
            (num_slots, len(_LAYER_IDS), 2, heads, tpb, head_dim), _SENTINEL, dtype=dtype
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
        views = [
            PoolView(
                pool_idx=0,
                buffer_entries=np.array(entries, dtype=BUFFER_ENTRY_DTYPE),
                pool_role=_ROLES,
                mapper_kind=MapperKind.INDEXED,
                bytes_per_layer=2 * buf,
                num_kv_heads=heads,
            )
        ]
        pools = [
            PhysicalPool(
                base_address=self.array.ctypes.data,
                slot_bytes=self.array[0].nbytes,
                num_slots=num_slots,
            )
        ]
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
    """One context rank of CTX TP2 after its prefill.

    ``ctx_tp`` 1 is the attention-DP layout (rank 0 holds every head); ``ctx_tp`` > 1 is the
    TP-attention layout (rank ``ctx_rank`` holds ``heads // ctx_tp`` of target and drafter).
    ``reuse_prefix`` > 0 is a prefix-reuse hit: the drafter stored only ``[reuse_prefix, prompt)``.
    """

    def __init__(self, prompt_len, with_draft=True, reuse_prefix=0, ctx_tp=1, ctx_rank=0):
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

    def chunk(self, *, with_draft=True, is_last=True) -> Chunk:
        return Chunk(
            block_ids_per_layer_groups=[self.target_table],
            kind_per_layer_group=[CacheKind.PAGED],
            token_range=TokenRange(start=0, end=self.prompt_len),
            is_last=is_last,
            draft_block_ids_per_layer_groups=[self.draft_table] if with_draft else None,
        )

    def task(self, *, with_draft=True, is_last=True, chunk=None) -> KVSendTask:
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
        prompt_len,
        tp_size,
        tp_rank,
        with_draft=True,
        draft_dim=_DRAFT_DIM,
        draft_dtype=np.int16,
    ):
        n_target = _blocks(prompt_len, _TARGET_TPB)
        n_draft = _blocks(prompt_len, _DRAFT_TPB)
        self.target = _Pool(
            n_target + 2,
            max(1, _TARGET_HEADS // tp_size),
            _TARGET_TPB,
            _TARGET_DIM,
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

    def chunk(self, prompt_len, *, with_draft=True) -> Chunk:
        return Chunk(
            block_ids_per_layer_groups=[self.target_table],
            kind_per_layer_group=[CacheKind.PAGED],
            token_range=TokenRange(start=0, end=prompt_len),
            is_last=True,
            draft_block_ids_per_layer_groups=[self.draft_table] if with_draft else None,
        )

    def req_info(self, *, with_draft=True) -> RecvReqInfo:
        info = RecvReqInfo(
            sender_req_id=_RID,
            instance_name=self.rank_info.instance_name,
            instance_rank=self.rank_info.instance_rank,
            block_ids_per_layer_groups=[self.target_table],
            unique_rid=_RID,
            draft_block_ids_per_layer_groups=[self.draft_table] if with_draft else None,
        )
        return RecvReqInfo.from_bytes(info.to_bytes())  # as the wire delivers it


def _apply(write_meta) -> None:
    for src, dst, size in zip(write_meta.src_ptrs, write_meta.dst_ptrs, write_meta.sizes):
        ctypes.memmove(int(dst), int(src), int(size))


def _chunk_num_bytes(gen: _Generation, chunk: Chunk) -> int:
    """The receive's expected write bytes: the target's KV plus the drafter context window."""
    paged, state = transceiver_module._chunk_bytes_by_kind(gen.rank_info.page_table, chunk)
    draft = KvCacheTransceiverV2._draft_chunk_num_bytes(gen.rank_info.draft_page_table, chunk)
    return paged + state + draft


def _target_only_meta(prompt_len, gen_tp, gen_rank, ctx_tp=1, ctx_rank=0):
    ref_ctx = _Context(prompt_len, with_draft=False, ctx_tp=ctx_tp, ctx_rank=ctx_rank)
    ref_gen = _Generation(prompt_len, gen_tp, gen_rank, with_draft=False)
    ref_ctx.register(ref_gen.rank_info)
    ref = ref_ctx.sender._build_kv_write_meta(
        ref_ctx.task(with_draft=False), ref_gen.req_info(with_draft=False)
    )
    return ref_ctx, ref_gen, ref


def _same_target_write(meta, ref, ctx, ref_ctx, gen, ref_gen):
    rebase_src = ctx.target.array.ctypes.data - ref_ctx.target.array.ctypes.data
    rebase_dst = gen.target.array.ctypes.data - ref_gen.target.array.ctypes.data
    return (
        np.array_equal(meta.src_ptrs, ref.src_ptrs + rebase_src)
        and np.array_equal(meta.dst_ptrs, ref.dst_ptrs + rebase_dst)
        and np.array_equal(meta.sizes, ref.sizes)
        and meta.expected_transfers == ref.expected_transfers
    )


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


# ---------------------------------------------------------------------------
# RankInfo's optional draft-pool geometry and the request-info wire format
# ---------------------------------------------------------------------------


def _legacy_bytes(ri: RankInfo) -> bytes:
    """What a rank without the draft fields serializes: no draft keys at all."""
    data = asdict(ri)
    for key in (
        "draft_attention",
        "draft_page_table",
        "draft_receive_disabled",
        "fabric_export",
        "draft_layers_reshard_across_tp",
    ):
        data.pop(key)
    if not ri.context_first_ownership:
        data.pop("context_first_ownership")
    if ri.draft_kv_dtype is None:
        data.pop("draft_kv_dtype")
    data["attention"] = ri.attention.to_dict()
    data["aux_meta"] = None
    data["page_table"] = ri.page_table.to_dict()
    return msgpack.packb(data)


def test_target_only_rank_info_keeps_the_legacy_wire_format():
    ri = _Context(64, with_draft=False).rank_info
    data = ri.to_bytes()
    assert data == _legacy_bytes(ri)
    restored = RankInfo.from_bytes(data)
    assert restored.draft_attention is None and restored.draft_page_table is None
    assert not restored.has_draft_kv
    assert restored.draft_view() is None
    assert not object.__new__(RankInfo).has_draft_kv


def test_draft_fields_round_trip_and_the_view_swaps_only_the_geometry():
    ri = _Context(64).rank_info
    restored = RankInfo.from_bytes(ri.to_bytes())
    assert restored.has_draft_kv
    assert restored.draft_attention == ri.draft_attention
    assert restored.draft_page_table.to_dict() == ri.draft_page_table.to_dict()
    assert restored.attention == ri.attention
    assert restored.page_table.to_dict() == ri.page_table.to_dict()
    view = restored.draft_view()
    assert view.attention == ri.draft_attention
    assert view.page_table.to_dict() == ri.draft_page_table.to_dict()
    assert not view.has_draft_kv and view.draft_view() is None
    for name in ("instance_name", "instance_rank", "tp_size", "tp_rank", "dp_size", "dp_rank"):
        assert getattr(view, name) == getattr(ri, name), name
    assert view.tp_size_per_dp_group == ri.tp_size_per_dp_group

    # The synchronous-receive verdict rides the wire only when set, never on the draft view.
    assert "draft_receive_disabled" not in msgpack.unpackb(ri.to_bytes(), strict_map_key=False)
    ri.draft_receive_disabled = "sync_path_tp"
    restored = RankInfo.from_bytes(ri.to_bytes())
    assert restored.draft_receive_disabled == "sync_path_tp"
    assert restored.draft_view().draft_receive_disabled is None


def _manager(heads, tpb, head_dim, layers):
    mapping = SimpleNamespace(
        rank=0,
        tp_size=2,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        dp_size=1,
        cp_size=1,
        cp_rank=0,
        enable_attention_dp=True,
    )
    return SimpleNamespace(
        mapping=mapping,
        num_kv_heads_per_layer=[heads] * layers,
        pp_layers=list(range(layers)),
        tokens_per_block=tpb,
        head_dim=head_dim,
        dtype=None,
        kv_factor=2,
    )


@pytest.mark.parametrize("with_draft", [False, True])
def test_from_kv_cache_manager_reads_the_draft_pool_geometry(monkeypatch, with_draft):
    tables = {}

    def build(manager):
        tables[id(manager)] = object()
        return tables[id(manager)]

    monkeypatch.setattr(rank_info_module, "build_page_table_from_manager", build)
    monkeypatch.setattr(rank_info_module, "get_size_in_bytes", lambda n, _dtype: 2 * n)
    target = _manager(heads=8, tpb=64, head_dim=128, layers=62)
    draft = _manager(heads=4, tpb=32, head_dim=128, layers=1) if with_draft else None

    ri = RankInfo.from_kv_cache_manager("ctx", target, 0, draft_kv_cache_manager=draft)

    assert ri.attention == AttentionInfo(8, 64, 128, 2, True, False)
    assert ri.page_table is tables[id(target)]
    if with_draft:
        assert ri.draft_attention == AttentionInfo(4, 32, 128, 2, True, False)
        assert ri.draft_page_table is tables[id(draft)]
    else:
        assert ri.draft_attention is None and ri.draft_page_table is None


def test_recv_req_info_without_a_draft_set_is_the_legacy_message():
    info = RecvReqInfo(
        sender_req_id=7,
        instance_name="gen",
        instance_rank=3,
        block_ids_per_layer_groups=[np.array([1, -1, 2], dtype=np.int64)],
        unique_rid=_RID,
        dst_start_token=None,
        aux_slot=5,
        slice_id=0,
    )
    legacy = msgpack.packb(
        {
            "sender_req_id": 7,
            "instance_name": "gen",
            "instance_rank": 3,
            "block_ids_per_layer_groups": [np.array([1, -1, 2], dtype=np.int64).tobytes()],
            "unique_rid": _RID,
            "dst_start_token": None,
            "aux_slot": 5,
            "slice_id": 0,
            "context_first": False,
        }
    )
    assert info.to_bytes() == legacy
    assert RecvReqInfo.from_bytes(legacy).draft_block_ids_per_layer_groups is None


# ---------------------------------------------------------------------------
# Loopback: the window lands head-sliced and bitwise, the target write is unchanged
# ---------------------------------------------------------------------------


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
    # pools' addresses; the draft fragments follow, all inside the draft pools.
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


def test_receiver_draft_set_without_a_sender_draft_registrar_sends_the_target_only():
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


# ---------------------------------------------------------------------------
# Aux slot and the context-state codec
# ---------------------------------------------------------------------------


def _unit_request(**kwargs):
    defaults = dict(
        py_request_id=_RID,
        py_disaggregated_params=DisaggregatedParams(
            disagg_request_id=_RID, schedule_style=_GENERATION_FIRST
        ),
        prompt_len=64,
        cached_tokens=0,
        py_draft_tokens=[],
        get_last_tokens=lambda: [17],
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_aux_draft_slot_carries_the_context_state_only_when_engaged():
    plain = AuxBuffer(max_slot_num=4, beam_width=1, max_draft_len=3)
    assert plain.meta.ptrs.size == 4
    slot = plain.alloc_slot().id
    plain.fill_slot(slot, _unit_request(py_draft_ctx_aux=(64, 1, True)))
    assert plain.get_slot_draft_ctx(slot) is None

    aux = AuxBuffer(max_slot_num=4, beam_width=1, max_draft_len=3, draft_ctx=True)
    assert aux.meta.ptrs.size == 5
    assert np.array_equal(aux.meta.item_sizes[:4], plain.meta.item_sizes)
    assert aux.meta.item_sizes[4] == 3 * 4
    slot = aux.alloc_slot().id
    aux.fill_slot(slot, _unit_request(py_draft_ctx_aux=(9000, 3, True)))
    assert aux.get_slot_draft_ctx(slot) == (9000, 3, True)
    aux.fill_slot(slot, _unit_request(py_draft_ctx_aux=(0, 0, False)))
    assert aux.get_slot_draft_ctx(slot) == (0, 0, False)
    aux.fill_slot(slot, _unit_request(py_draft_ctx_aux=(9000, 3, True)))
    assert aux.get_slot_draft_ctx(slot) == (9000, 3, True)
    aux.fill_slot(slot, _unit_request())  # no decision at all reads as "not shipped"
    assert aux.get_slot_draft_ctx(slot) == (0, 0, False)


def test_context_state_codec_round_trips_and_ignores_foreign_values():
    state = encode_draft_ctx_state(9000, 3, True)
    assert state == [9000, 3, 1]
    assert decode_draft_ctx_state(state) == (9000, 3, True)
    assert decode_draft_ctx_state(encode_draft_ctx_state(64, 0, False)) == (64, 0, False)
    for foreign in (None, b"", "opaque", [1, 2], [1, 2, 3, 4], [-1, 0, 1], [1, 0, 2], ["a", 0, 1]):
        assert decode_draft_ctx_state(foreign) is None


def _aux_meta(base, engaged):
    sizes = [8, 32, 8, 8] + ([12] if engaged else [])
    return AuxBufferMeta(
        ptrs=np.array([base + 0x10000 * i for i in range(len(sizes))], dtype=np.int64),
        size=np.array([s * 16 for s in sizes], dtype=np.int64),
        item_sizes=np.array(sizes, dtype=np.int64),
    )


@pytest.mark.parametrize("sender_engaged", [True, False])
@pytest.mark.parametrize("peer_engaged", [True, False])
def test_aux_write_moves_the_common_buffers_between_mixed_peers(sender_engaged, peer_engaged):
    """Mixed peers move only the common aux buffers; matched peers keep the layout as is."""
    src = _aux_meta(0x100000, sender_engaged)
    dst = _aux_meta(0x900000, peer_engaged)
    mixed = sender_engaged != peer_engaged
    if mixed:
        with pytest.raises(ValueError, match="do not match"):
            build_aux_transfer_layout(src, dst)
    peer_ri = SimpleNamespace(
        has_draft_kv=peer_engaged,
        aux_meta=dst,
        instance_name="gen",
        instance_rank=0,
        dp_rank=0,
        self_endpoint="",
    )
    cached = {}
    sender = Sender.__new__(Sender)
    sender._registrar = SimpleNamespace(
        self_rank_info=SimpleNamespace(has_draft_kv=sender_engaged, aux_meta=src),
        get_peer_rank_info=lambda name, rank: peer_ri,
        get_peer_overlap=lambda ri, dp_rank: PeerOverlap(ranks=[0]),
        should_send_aux=lambda ri: True,
        get_aux_transfer_layout=lambda name, rank: cached.get((name, rank)),
        cache_aux_transfer_layout=lambda name, rank, layout: cached.__setitem__(
            (name, rank), layout
        ),
    )
    task = SimpleNamespace(_perf_timer=None, _slot=2, _unique_rid=_RID)
    meta = sender._build_aux_write_meta(
        task, SimpleNamespace(instance_name="gen", instance_rank=0, aux_slot=3, unique_rid=_RID)
    )
    if not mixed:
        assert meta.sizes.size == src.item_sizes.size
        return
    common = _aux_meta(0x100000, False)
    assert np.array_equal(meta.sizes, common.item_sizes)
    assert np.array_equal(meta.src_ptrs, common.ptrs + common.item_sizes * 2)
    assert np.array_equal(meta.dst_ptrs, _aux_meta(0x900000, False).ptrs + common.item_sizes * 3)
    trimmed = aux_meta_without_draft_ctx(_aux_meta(0x100000, True))
    assert np.array_equal(trimmed.ptrs, common.ptrs)
    assert np.array_equal(trimmed.item_sizes, common.item_sizes)


# ---------------------------------------------------------------------------
# The context side's send: wait on the drafter-store event, decide, then publish
# ---------------------------------------------------------------------------


class _StoreEvent:
    def __init__(self, order):
        self._order = order

    def synchronize(self):
        self._order.append("wait")


class _Worker:
    """The drafter state the context side's forward left for the request, and the seeds."""

    def __init__(self, ctx_len=None, ctx_base=0):
        self.order = []
        self.ready = {} if ctx_len is None else {_RID: (_StoreEvent(self.order), ctx_len, ctx_base)}
        self.seeds = {}

    def take_disagg_draft_ctx_ready(self, request_id):
        return self.ready.pop(request_id, None)

    def seed_disagg_draft_ctx(self, request_id, ctx_len, ctx_base, prompt_len):
        self.seeds[request_id] = (ctx_len, ctx_base, prompt_len)


def _small_chunk(draft=None):
    return Chunk(
        block_ids_per_layer_groups=[np.array([3, 4], dtype=np.int64)],
        kind_per_layer_group=[CacheKind.PAGED],
        token_range=TokenRange(start=0, end=64),
        is_last=True,
        draft_block_ids_per_layer_groups=draft,
    )


def _engage(transceiver, owner):
    transceiver._draft_ctx_owner = owner
    transceiver._draft_page_table = SimpleNamespace(
        tokens_per_block=32, layer_groups=[SimpleNamespace(sliding_window_size=None)]
    )
    transceiver._draft_kv_ctx_counters = DraftKvCounters("context")
    transceiver._draft_kv_gen_counters = DraftKvCounters("generation")


def _send(monkeypatch, *, owner=None, request=None, chunk=None):
    order = [] if owner is None else owner.order
    published = []
    chunk = chunk if chunk is not None else _small_chunk(draft=[np.array([1, 2])])

    class _Publish:
        def __init__(self, session, req):
            pass

        def publish(self, extent):
            order.append("publish")
            published.append(extent)

    monkeypatch.setattr(transceiver_module, "PeerPublish", _Publish)
    transceiver = object.__new__(KvCacheTransceiverV2)
    session = MagicMock()
    session._enforce_physical_ownership = False
    transceiver._enable_pipelined_transfer = False
    transceiver._send_reqs = {}
    transceiver._get_or_create_send_session = MagicMock(return_value=session)
    transceiver._create_cache_extent = MagicMock(return_value=CacheExtent(name=_RID, local=chunk))
    transceiver._finalize_send = MagicMock()
    if owner is not None:
        _engage(transceiver, owner)
    request = request if request is not None else _unit_request()
    request.set_kv_cache_transfer_start = lambda _ts: None
    KvCacheTransceiverV2.respond_and_send_async(transceiver, request)
    session.set_exception.assert_not_called()
    transceiver._finalize_send.assert_called_once()
    return order, transceiver, request, published[0]


def _params(**kwargs):
    return DisaggregatedParams(disagg_request_id=_RID, **kwargs)


def test_send_without_the_draft_set_publishes_untouched(monkeypatch):
    order, _, request, extent = _send(monkeypatch)
    assert order == ["publish"]
    assert not hasattr(request, "py_draft_ctx_aux")
    assert extent.local.draft_block_ids_per_layer_groups is not None  # nothing strips it


@pytest.mark.parametrize(
    "ctx_base, style",
    [
        (0, _GENERATION_FIRST),
        (1, _CONTEXT_FIRST),  # a reuse hit ships its base
    ],
)
def test_send_waits_on_the_drafter_store_before_publishing(monkeypatch, ctx_base, style):
    owner = _Worker(64, ctx_base)
    request = _unit_request(py_disaggregated_params=_params(schedule_style=style))
    order, transceiver, request, extent = _send(monkeypatch, owner=owner, request=request)
    assert order == ["wait", "publish"]
    assert request.py_draft_ctx_aux == (64, ctx_base, True)
    assert extent.local.draft_block_ids_per_layer_groups is not None
    assert owner.ready == {}
    assert transceiver._draft_kv_ctx_counters.sent == 1
    assert not transceiver._draft_kv_ctx_counters.fallback


@pytest.mark.parametrize(
    "ctx_len, ctx_base, draft, reason",
    [
        (None, None, [np.array([1, 2])], "no_store_event"),
        (0, 0, [np.array([1, 2])], "ctx_len_mismatch"),  # the drafter stored nothing
        (64, 3, [np.array([1, 2])], "reuse_base"),  # a base past the prompt's two pages
        (64, 0, [np.array([-1, 2])], "page_mismatch"),  # a window page is missing
    ],
)
@pytest.mark.parametrize("context_first", [True, False])
def test_a_window_that_cannot_be_served_falls_back(
    monkeypatch, context_first, ctx_len, ctx_base, draft, reason
):
    owner = _Worker(ctx_len, ctx_base)
    request = (
        _unit_request(py_disaggregated_params=_params(schedule_style=_CONTEXT_FIRST))
        if context_first
        else _unit_request()
    )
    order, transceiver, request, extent = _send(
        monkeypatch, owner=owner, request=request, chunk=_small_chunk(draft=draft)
    )

    # The request is published, never failed; the target set is untouched.
    assert order[-1] == "publish"
    assert np.array_equal(extent.local.block_ids_per_layer_groups[0], [3, 4])
    assert request.py_draft_ctx_aux == (0, 0, False)
    counters = transceiver._draft_kv_ctx_counters
    assert counters.sent == 0 and counters.fallback == {reason: 1}
    # Either style leaves the window's pages out; in generation-first the receiver vetoes the
    # short draft write while the target set still travels.
    assert extent.local.draft_block_ids_per_layer_groups is None


def test_context_first_response_carries_the_decision(monkeypatch):
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._send_reqs = {}
    transceiver._dp_rank = 0
    transceiver._context_info_endpoint = "tcp://ctx:1"
    seen = {}
    monkeypatch.setattr(
        transceiver_module, "ContextPhaseParams", lambda **kwargs: seen.update(kwargs) or kwargs
    )
    request = _unit_request(py_disaggregated_params=_params(schedule_style=_CONTEXT_FIRST))
    request.py_draft_ctx_aux = (0, 0, False)
    KvCacheTransceiverV2._finalize_send(transceiver, request, MagicMock())
    assert seen["opaque_state"] is None

    _engage(transceiver, _Worker(64))
    for state in ((0, 0, False), (64, 0, True)):
        request.py_draft_ctx_aux = state
        KvCacheTransceiverV2._finalize_send(transceiver, request, MagicMock())
        # Never opaque_state: C++ ContextPhaseParams parses it as a DataTransceiverState, so the
        # drafter state rides DisaggregatedParams.draft_ctx_state.
        assert seen["opaque_state"] is None
        assert request.py_draft_ctx_aux == state

    gen_first = _unit_request()
    gen_first.py_draft_ctx_aux = (64, 0, True)
    session = MagicMock()
    KvCacheTransceiverV2._finalize_send(transceiver, gen_first, session)
    assert seen["opaque_state"] is None  # generation-first: the aux slot carries it
    session.pack_aux.assert_called_once_with(gen_first)


# ---------------------------------------------------------------------------
# draft_ctx_state through the context response and the serving protocol
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "aux, context_only, expected",
    [
        ((9000, 3, True), True, [9000, 3, 1]),
        ((0, 0, False), True, [0, 0, 0]),
        ((64, 0, True), False, None),  # a generation request's own aux (generation-first)
        (None, True, None),  # no window decision
    ],
)
def test_the_context_response_carries_the_decision(aux, context_only, expected):
    request = SimpleNamespace(py_draft_ctx_aux=aux, is_context_only_request=context_only)
    assert _response_draft_ctx_state(request) == expected


def _gen_request(params, prompt_len=64, **kwargs):
    return _unit_request(
        py_disaggregated_params=params,
        prompt_len=prompt_len,
        is_generation_only_request=True,
        **kwargs,
    )


def test_context_first_state_survives_the_serving_params_round_trip():
    """CTX response -> serving DisaggregatedParams (draft_ctx_state) -> generation request."""
    from tensorrt_llm.serve.openai_protocol import (
        to_disaggregated_params,
        to_llm_disaggregated_params,
    )

    for state in ((9000, 256, True), (0, 0, False)):
        ctx_side = _params(
            schedule_style=_CONTEXT_FIRST,
            request_type="context_only",
            draft_ctx_state=encode_draft_ctx_state(*state),
        )
        wire = to_disaggregated_params(ctx_side)
        assert wire.draft_ctx_state == encode_draft_ctx_state(*state)
        assert wire.encoded_opaque_state is None  # never the C++ transceiver-state carrier
        wire = type(wire).model_validate_json(wire.model_dump_json())
        gen_side = to_llm_disaggregated_params(
            wire.model_copy(update={"request_type": "generation_only"})
        )
        assert decode_draft_ctx_state(gen_side.draft_ctx_state) == state
        assert gen_side.opaque_state is None
        gen_request = _gen_request(gen_side, prompt_len=state[0] or 64)
        expected = state[:2] if state[2] else None
        assert KvCacheTransceiverV2._received_draft_ctx_state(gen_request) == expected

    # The generation request builds its C++ ContextPhaseParams from opaque_state only.
    params = DisaggregatedParams(
        request_type="generation_only", ctx_request_id=1, draft_ctx_state=[9000, 3, 1]
    )
    assert params.get_context_phase_params().opaque_state is None


# ---------------------------------------------------------------------------
# The generation side: expected pages and the seed follow the context side's decision
# ---------------------------------------------------------------------------


def test_generation_seeds_the_received_state_or_an_empty_window():
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._seed_draft_ctx(_unit_request(py_draft_ctx_aux=(9000, 0, True)))  # off: nothing

    owner = _Worker(64)
    _engage(transceiver, owner)
    counters = transceiver._draft_kv_gen_counters

    # Generation-first: the aux slot.
    gen_first = _params(schedule_style=_GENERATION_FIRST)
    transceiver._seed_draft_ctx(_gen_request(gen_first, 9000, py_draft_ctx_aux=(9000, 0, True)))
    assert owner.seeds[_RID] == (9000, 0, 9000)
    transceiver._seed_draft_ctx(_gen_request(gen_first, 9000, py_draft_ctx_aux=(0, 0, False)))
    # Not shipped: based past every prompt page, the drafter reads nothing.
    assert owner.seeds[_RID] == (9000, 282, 9000)

    # Context-first: the response's draft_ctx_state.
    shipped = _params(
        schedule_style=_CONTEXT_FIRST, draft_ctx_state=encode_draft_ctx_state(33, 0, True)
    )
    transceiver._seed_draft_ctx(_gen_request(shipped, 33))
    assert owner.seeds[_RID] == (33, 0, 33)
    for params in (
        _params(schedule_style=_CONTEXT_FIRST, draft_ctx_state=encode_draft_ctx_state(0, 0, False)),
        _params(schedule_style=_CONTEXT_FIRST),  # no state arrived at all
    ):
        transceiver._seed_draft_ctx(_gen_request(params, 33))
        assert owner.seeds[_RID] == (33, 2, 33)

    assert counters.sent == 2
    assert counters.fallback == {"no_ctx_state": 3}


def test_context_first_generation_expects_no_draft_pages_after_a_fallback():
    transceiver = object.__new__(KvCacheTransceiverV2)
    _engage(transceiver, _Worker(64))
    transceiver._draft_reuse_adapter = SimpleNamespace(
        tokens_per_block=32,
        get_block_ordinals=lambda req, idx, lg: np.array([7, 8], dtype=np.int64),
    )
    shipped = _gen_request(
        _params(schedule_style=_CONTEXT_FIRST, draft_ctx_state=encode_draft_ctx_state(64, 0, True))
    )
    dropped = _gen_request(
        _params(schedule_style=_CONTEXT_FIRST, draft_ctx_state=encode_draft_ctx_state(0, 0, False))
    )
    assert np.array_equal(transceiver._describe_draft_local(shipped)[0], [7, 8])
    assert transceiver._describe_draft_local(dropped) is None
    # Generation-first asks before the context side decides: always the whole window.
    gen_first = _gen_request(_params(schedule_style=_GENERATION_FIRST))
    assert transceiver._describe_draft_local(gen_first) is not None


class _Fence:
    def __init__(self, done):
        self.done = done
        self.synchronized = 0

    def query(self):
        return self.done

    def synchronize(self):
        self.synchronized += 1


def _coordinator(manager):
    coordinator = object.__new__(DisaggTransferCoordinator)
    coordinator._draft_kv_cache_manager = manager
    return coordinator


def test_receive_fences_follow_the_draft_pool_readiness():
    # Not engaged: the target fences are untouched.
    req = SimpleNamespace(py_request_id=1)
    target = _Fence(True)
    fences = {1: target}
    object.__new__(DisaggTransferCoordinator)._merge_draft_receive_fences([req], fences, True)
    manager = SimpleNamespace(take_disagg_receive_ready=Mock())
    _coordinator(manager)._merge_draft_receive_fences([req], fences, True)
    manager.take_disagg_receive_ready.assert_not_called()
    assert fences == {1: target}

    # Deferred receive: the draft readiness joins the fence.
    draft_fences = {1: _Fence(False), 2: _Fence(True)}
    manager = SimpleNamespace(
        _disagg_draft_kv_receive=True,
        take_disagg_receive_ready=lambda r: draft_fences.pop(r.py_request_id, None),
    )
    fences = {1: _Fence(True), 3: None}
    reqs = [SimpleNamespace(py_request_id=i) for i in (1, 2, 3)]
    _coordinator(manager)._merge_draft_receive_fences(reqs, fences, True)
    assert fences[1].query() is False  # target done, draft not yet
    assert fences[2].query() is True  # the draft event alone
    assert fences[3] is None

    # Without deferral: waited on in place.
    draft = _Fence(False)
    manager = SimpleNamespace(
        _disagg_draft_kv_receive=True, take_disagg_receive_ready=lambda r: draft
    )
    fences = {}
    _coordinator(manager)._merge_draft_receive_fences([req], fences, False)
    assert draft.synchronized == 1
    assert fences == {}


class _Cache:
    """A fresh KV cache: no capacity until a resize grants it; a failed resize changes nothing."""

    def __init__(self, ok=True):
        self.ok = ok
        self.calls = []
        self.capacity = 0
        self.history_length = 0

    def resize(self, capacity, history_length=None):
        self.calls.append((capacity, history_length))
        if self.ok:
            self.capacity = capacity
            if history_length is not None:
                self.history_length = history_length
        return self.ok


def _draft_pool_manager(monkeypatch):
    class _Event:
        def record(self, stream=None):
            self.stream = stream

    monkeypatch.setattr(torch.cuda, "Event", _Event)
    manager = object.__new__(KVCacheManagerV2)
    manager.is_draft = True
    manager.num_extra_kv_tokens = 1
    manager._stream = "execution-stream"
    manager._disagg_receive_ready = {}
    manager._sharing = None
    manager.kv_cache_map = {}
    manager._gpu_max_tokens = 0
    manager.max_attention_window_vec = [4096]  # read by the resize-failure diagnostic
    manager.tokens_per_block = 32
    return manager


def test_gen_init_holds_the_window_and_fences_a_received_one(monkeypatch):
    manager = _draft_pool_manager(monkeypatch)
    seeded = []
    manager.engage_disagg_gen_init_window(lambda *args: seeded.append(args))
    manager.engage_disagg_draft_kv_receive()
    cache = _Cache()
    req = SimpleNamespace(py_request_id=9, prompt_len=9000, py_draft_tokens=[0, 0, 0])

    manager._prepare_disagg_gen_init_draft(req, cache)

    assert cache.calls == [(9000 + 3 + 1, 9000)]
    # The empty window at the prompt end, which a received window then replaces.
    assert seeded == [(9, 9000, manager.tokens_per_block)]
    assert manager._disagg_receive_ready[9].stream == "execution-stream"
    assert manager.take_disagg_receive_ready(req) is not None


def test_gen_init_without_the_transfer_records_no_fence_and_fails_loud(monkeypatch):
    manager = _draft_pool_manager(monkeypatch)
    manager.engage_disagg_gen_init_window(lambda *args: None)
    req = SimpleNamespace(py_request_id=9, prompt_len=64, py_draft_tokens=None)
    manager._prepare_disagg_gen_init_draft(req, _Cache())
    assert manager._disagg_receive_ready == {}
    with pytest.raises(RuntimeError, match="disagg gen-init resize failed"):
        manager._prepare_disagg_gen_init_draft(req, _Cache(ok=False))
    assert manager._disagg_receive_ready == {}


def test_the_transfer_engages_only_a_draft_pool_with_window_sizing(monkeypatch):
    manager = _draft_pool_manager(monkeypatch)
    with pytest.raises(AssertionError, match="gen-init window sizing"):
        manager.engage_disagg_draft_kv_receive()
    manager.is_draft = False
    with pytest.raises(AssertionError):
        manager.engage_disagg_draft_kv_receive()
    assert manager._disagg_draft_kv_receive is False


def test_draft_vetoes_ride_the_generation_consensus_and_are_absent_when_off():
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._recv_sessions = {1: object(), 2: object()}
    transceiver._gen_need_sync = True
    seen = []

    def allgather(local):
        seen.append(local)
        other = [[], [], [1, 2], [1, 2], [1, 2]] + ([[2]] if len(local) == 6 else [])
        return [local, other]

    transceiver._gen_allgather = allgather
    # Off: the same four results and the same allgather payload as without the feature.
    result = transceiver._gen_consensus_outcome([1, 2], [], [], [1, 2], [1, 2])
    assert len(result) == 4 and len(seen[-1]) == 5
    # On: one more list in the same allgather; a veto on any rank is a veto on all.
    _, _, completed, verified, vetoed = transceiver._gen_consensus_outcome(
        [1, 2], [], [], [1, 2], [1, 2], locally_draft_vetoed=[]
    )
    assert len(seen) == 2 and len(seen[-1]) == 6
    assert completed == [1, 2] and verified == {1, 2} and vetoed == {2}


@pytest.mark.parametrize("idle_skip", [False, True])
@pytest.mark.parametrize("remote_veto", [False, True])
def test_gen_consensus_returns_the_draft_vetoes_with_idle_poll_skip(idle_skip, remote_veto):
    """The draft veto set is the last consensus field, after the optional idle-agreement flag."""
    stand_in = object.__new__(KvCacheTransceiverV2)
    stand_in._recv_sessions = {_RID: SimpleNamespace()}
    stand_in._gen_idle_poll_skip = idle_skip
    stand_in._gen_idle_agreed = False
    stand_in._gen_deferred_receives = 0
    stand_in._gen_need_sync = True

    def allgather(local):
        peer = [list(field) for field in local]
        if idle_skip:
            peer[-2] = [0]
        peer[-1] = [_RID] if remote_veto else []
        return [local, peer]

    stand_in._gen_allgather = allgather
    cancelled, failed, completed, verified, globally_vetoed = (
        KvCacheTransceiverV2._gen_consensus_outcome(
            stand_in, [_RID], [], [], [_RID], [_RID], locally_draft_vetoed=[]
        )
    )
    assert (cancelled, failed, completed, verified) == ([], [], [_RID], {_RID})
    assert isinstance(globally_vetoed, set)
    assert globally_vetoed == ({_RID} if remote_veto else set())
    # The idle flag still lands on its own field: one session is open, so no rank is idle.
    assert stand_in._gen_idle_agreed is False


# ---------------------------------------------------------------------------
# The context side's per-request decision, end to end through the loopback write
# ---------------------------------------------------------------------------


def _transceiver(rank_info, worker) -> KvCacheTransceiverV2:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._draft_ctx_owner = worker
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
    if style == _CONTEXT_FIRST:
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


_WINDOW_CASES = [(p, 0) for p in PROMPT_LENS] + [(33, 16), (9000, 8200)]


@pytest.mark.parametrize("style", list(DisaggScheduleStyle))
@pytest.mark.parametrize("prompt_len, reuse_prefix", _WINDOW_CASES)
@pytest.mark.parametrize("gen_tp", [4, 8])
def test_draft_window_lands_head_sliced_and_bitwise(gen_tp, prompt_len, reuse_prefix, style):
    """The window and its base travel; every row the generation drafter reads is the context's.

    ``reuse_prefix`` > 0 is a reuse hit: the drafter stored only the tail from the reused position
    on and reads past ``ctx_base = ceil(reuse_prefix / page)`` pages; generation seeds the same.
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
        # The fan-out reaches every generation rank of the DP group, target and draft alike.
        assert meta.expected_transfers == gen_tp
        head = rank if gen_tp == 4 else rank // 2  # duplicate_kv_weight at TP8
        common = (ctx.draft_table >= 0) & (gen.draft_table >= 0)
        assert common.any()
        assert (gen.draft_table >= 0).sum() == common.sum()  # the context side covers it all
        for ordinal in np.flatnonzero(common):
            received = gen.draft.array[gen.draft_table[ordinal]]
            stored = ctx.draft.array[ctx.draft_table[ordinal]][:, :, head : head + 1]
            assert np.array_equal(received, stored), f"ordinal {ordinal}, rank {rank}"
            layer, kv, _, t, d = np.indices(received.shape)
            pos = ordinal * _DRAFT_TPB + t
            mask = pos >= reuse_prefix  # rows the drafter stored are f(...) exactly
            assert np.array_equal(received[mask], _value(layer, kv, head, pos, d)[mask])
        # Released or never-held pages, and every slot no ordinal names, are untouched.
        written = {int(gen.draft_table[o]) for o in np.flatnonzero(common)}
        for slot in range(gen.draft.array.shape[0]):
            if slot not in written:
                assert (gen.draft.array[slot] == _SENTINEL).all(), f"slot {slot} was written"
        # Exact byte accounting: the receiver's expected-write total is what landed.
        assert _chunk_num_bytes(gen, gen.chunk(prompt_len)) == int(meta.sizes.sum())

    worker = _Worker()
    gen_transceiver = _transceiver(gen.rank_info, worker)
    gen_transceiver._seed_draft_ctx(gen_request)
    assert worker.seeds[_RID] == (prompt_len, base, prompt_len)
    assert gen_transceiver._draft_kv_gen_counters.sent == 1


# ---------------------------------------------------------------------------
# CTX TP-attention layout (tp2, no attention DP): each rank holds a slice of every request
# ---------------------------------------------------------------------------


def _tp_contexts(prompt_len, ctx_tp=2, **kwargs):
    return [_Context(prompt_len, ctx_tp=ctx_tp, ctx_rank=c, **kwargs) for c in range(ctx_tp)]


def _global_head(gen_rank, local_head, gen_heads, gen_tp, total_heads):
    """The head generation rank ``gen_rank`` holds in ``local_head`` (the HND mapper's rule)."""
    return (gen_rank * gen_heads + local_head) * total_heads // (gen_tp * gen_heads) % total_heads


@pytest.mark.parametrize("prompt_len, reuse_prefix", _WINDOW_CASES)
@pytest.mark.parametrize("gen_tp", [4, 8])
def test_tp_attention_ctx_window_lands_bitwise_on_every_generation_rank(
    gen_tp, prompt_len, reuse_prefix
):
    contexts = _tp_contexts(prompt_len, reuse_prefix=reuse_prefix)
    base = _blocks(reuse_prefix, _DRAFT_TPB)
    chunks = []
    for ctx in contexts:
        transceiver, request, chunk = _context_decision(
            ctx, _Worker(prompt_len, base), _CONTEXT_FIRST
        )
        # Both CTX TP ranks decide from the same host state, so both ship.
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
    """CTX TP ranks diverge: rank ``shipping_rank`` ships its slice, the other has no store event.

    No hang, no write outside the pages the generation rank named, the target set intact, every
    generation rank starts from an empty window, and the divergence is counted.
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
    context_first = style == _CONTEXT_FIRST

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
        pools = [gen.target.array, gen.draft.array]
        for dst, size in zip(meta.dst_ptrs, meta.sizes):
            assert any(
                pool.ctypes.data <= dst and dst + size <= pool.ctypes.data + pool.nbytes
                for pool in pools
            )
        _apply(meta)
        ref_ctx, ref_gen, ref = _target_only_meta(prompt_len, gen_tp, rank)
        n = ref.src_ptrs.size
        assert np.array_equal(meta.sizes[:n], ref.sizes)
        _apply(ref)
        assert np.array_equal(gen.target.array, ref_gen.target.array)

        aux = ctx_request.py_draft_ctx_aux
        outcomes[rank] = _gen_receive_outcome(gen, gen_transceiver, gen_request, meta, aux)
        if not context_first and ctx.ctx_rank != shipping_rank:
            # A falling-back generation-first writer sends no draft bytes, only the target set,
            # and its receiver reports the missing state rather than a byte mismatch.
            assert meta.sizes.size == n
            assert outcomes[rank] == "no_ctx_state"
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
    # The context rank whose decision the receivers did not follow counts it too.
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
# Registration-time draft validation: a mismatch disables the draft set for that peer pair only,
# both sides agreeing; the target is untouched
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
    """Gen-first and ADP dispatches carry no ctx_info_endpoint; the window decision never needs one."""
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


def _rename_roles(gen):
    group = gen.rank_info.draft_page_table.layer_groups[0]
    group.pool_views[0].pool_role = frozenset({"key", "value", "scale"})


def _duplicate_view(gen):
    group = gen.rank_info.draft_page_table.layer_groups[0]
    group.pool_views.append(replace(group.pool_views[0]))


def _shift_layers(gen):
    group = gen.rank_info.draft_page_table.layer_groups[0]
    group.local_layers = [
        LocalLayer(ll.local_layer_id, ll.global_layer_id + 100) for ll in group.local_layers
    ]


def _sync_receive(gen):
    gen.rank_info.draft_receive_disabled = "sync_path_tp"


_GEOMETRY = "peer_draft_geometry"


@pytest.mark.parametrize(
    "gen_kwargs, mutate, why, reason",
    [
        ({"draft_dim": 2 * _DRAFT_DIM}, None, "head_dim", _GEOMETRY),
        ({"draft_dtype": np.int32}, None, "dtype", _GEOMETRY),
        ({}, _rename_roles, "matches 0 peer views", _GEOMETRY),
        ({}, _duplicate_view, "matches 2 peer views", _GEOMETRY),
        ({}, _shift_layers, "draft layers", _GEOMETRY),
        # GEN TP on the synchronous receive path takes no window.
        ({}, _sync_receive, None, "sync_path_tp"),
    ],
)
@pytest.mark.parametrize("gen_tp", [4, 8])
def test_a_disabled_peer_pair_drops_only_the_window(gen_tp, gen_kwargs, mutate, why, reason):
    prompt_len = 4097
    rank = gen_tp - 1
    ctx = _Context(prompt_len)
    gen = _Generation(prompt_len, gen_tp, rank, **gen_kwargs)
    if mutate is not None:
        mutate(gen)
    if why is not None:
        found = draft_peer_mismatch(ctx.rank_info.draft_view(), gen.rank_info.draft_view())
        assert found is not None and why in found
        # The receiver's view of the same pair reaches a verdict too.
        assert (
            draft_peer_mismatch(gen.rank_info.draft_view(), ctx.rank_info.draft_view()) is not None
        )

    counters = DraftKvCounters("context")
    ctx.sender._draft_kv_counters = counters
    ctx.register(gen.rank_info)  # never raises: the target registers, the window is disabled
    assert ctx.sender._draft_disabled_peers[f"gen{rank}"][0] == reason
    assert ctx.sender._registrar.get_peer_rank_info("gen", rank) is not None

    task, info = _receive(_receiver(gen, ctx), gen, prompt_len)
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
    request = _request(prompt_len, _GENERATION_FIRST)
    request.py_draft_ctx_aux = (prompt_len, 0, True)
    veto = gen_transceiver._local_draft_veto(request, session)
    assert veto == reason
    gen_transceiver._seed_draft_ctx(request, veto)
    assert worker.seeds[_RID] == (prompt_len, _blocks(prompt_len, _DRAFT_TPB), prompt_len)
    assert gen_transceiver._draft_kv_gen_counters.fallback == {reason: 1}


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


def _add_replicated_view(rank_info, layer_bytes, ring_slot_bytes=None):
    """Append a REPLICATED draft view holding ``layer_bytes`` per layer."""
    group = rank_info.draft_page_table.layer_groups[0]
    entries = [(i, i * layer_bytes, layer_bytes) for i in range(len(group.local_layers))]
    group.pool_views.append(
        PoolView(
            pool_idx=0,
            buffer_entries=np.array(entries, dtype=BUFFER_ENTRY_DTYPE),
            pool_role=frozenset({"draft_index"}),
            mapper_kind=MapperKind.REPLICATED,
            bytes_per_layer=layer_bytes,
            ring_slot_bytes=ring_slot_bytes,
        )
    )


@pytest.mark.parametrize("ring", [False, True])
def test_replicated_draft_views_must_match_in_bytes_per_layer(ring):
    """A replicated draft view moves byte for byte, so its size must match; a ring may differ."""
    ctx = _Context(64)
    gen = _Generation(64, 4, 0)
    _add_replicated_view(ctx.rank_info, 256, ring_slot_bytes=64 if ring else None)
    _add_replicated_view(gen.rank_info, 512, ring_slot_bytes=64 if ring else None)
    for self_ri, peer_ri in ((ctx.rank_info, gen.rank_info), (gen.rank_info, ctx.rank_info)):
        why = draft_peer_mismatch(self_ri.draft_view(), peer_ri.draft_view())
        if ring:
            assert why is None
        else:
            assert why is not None and "bytes per layer" in why


def test_a_draft_fragment_error_drops_only_the_window(monkeypatch):
    prompt_len = 4097
    ctx = _Context(prompt_len)
    gen = _Generation(prompt_len, 4, 2)
    ctx.register(gen.rank_info)
    counters = DraftKvCounters("context")
    ctx.sender._draft_kv_counters = counters

    real = Sender._collect_pool_fragments
    appended = []

    def broken(registrar, *args):
        # The draft set fails only after it appended fragments, which must then be dropped.
        size_specs = args[-1]
        before = sum(len(specs) for specs in size_specs.values())
        real(registrar, *args)
        if registrar is ctx.sender._draft_registrar:
            appended.append(sum(len(specs) for specs in size_specs.values()) - before)
            raise ValueError("HND bytes per head mismatch")

    monkeypatch.setattr(Sender, "_collect_pool_fragments", staticmethod(broken))
    meta = ctx.sender._build_kv_write_meta(ctx.task(), gen.req_info())  # completes
    assert len(appended) == 1 and appended[0] > 0
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
    request = _request(prompt_len, _CONTEXT_FIRST)
    gen_transceiver._note_expected_draft_bytes(request, gen.chunk(prompt_len))
    assert gen_transceiver._local_draft_veto(request, session) == "ctx_rank_divergence"


@pytest.mark.parametrize(
    "tp_size, adp, synchronous, expected",
    [
        (4, False, True, "sync_path_tp"),
        (4, False, False, None),
        (4, True, True, None),
        (1, False, True, None),
    ],
)
def test_sync_receive_with_tp_ranks_takes_no_window(tp_size, adp, synchronous, expected):
    mapping = SimpleNamespace(tp_size=tp_size, enable_attention_dp=adp)
    assert draft_receive_disabled_reason(mapping, synchronous_receive=synchronous) == expected


# ---------------------------------------------------------------------------
# Mixed peers: only one side engages the drafter window. The pair degrades to target-only
# transfers: no request fails, the target write is bitwise the target-only write, both sides count
# draft_kv_fallback{peer_not_engaged}, and an engaged generation side seeds the empty window.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prompt_len", [4097])
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
    ctx_counters, chunks = {}, {}
    for ctx in contexts:
        if ctx_engaged:
            transceiver, _, chunk = _context_decision(ctx, _Worker(prompt_len, 0), style)
            assert chunk.draft_block_ids_per_layer_groups is not None
            ctx_counters[ctx.ctx_rank] = transceiver._draft_kv_ctx_counters
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
        receiver = _receiver(gen, ctx, counters=gen_counters if ctx_engaged else None)
        assert receiver._draft_disabled_peers[_CTX_ENDPOINT][0] == "peer_not_engaged"
        task, info = _receive(receiver, gen, prompt_len)
        assert info.draft_block_ids_per_layer_groups is None
        target_bytes = _chunk_num_bytes(gen, gen.chunk(prompt_len, with_draft=False))
        assert task.expected_write_bytes == target_bytes

        meta = ctx.sender._build_kv_write_meta(ctx.task(chunk=chunks[ctx.ctx_rank]), info)
        ref_ctx, ref_gen, ref = _target_only_meta(prompt_len, gen_tp, rank, ctx_tp, ctx.ctx_rank)
        assert _same_target_write(meta, ref, ctx, ref_ctx, gen, ref_gen)
        _apply(meta)
        _apply(ref)
        assert np.array_equal(gen.target.array, ref_gen.target.array)  # bitwise target-only
        assert (gen.draft.array == _SENTINEL).all()  # no draft byte moved
        assert int(meta.sizes.sum()) == task.expected_write_bytes

        if ctx_engaged:
            assert gen_counters.fallback == {"peer_not_engaged": 1}
            continue
        # The engaged generation rank vetoes the window and seeds the empty one at the prompt end,
        # even over a stale drafter-context aux slot (the non-engaged peer never writes it).
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

    # The context side counts every request toward a peer that does not engage the window.
    assert sum(c.fallback["peer_not_engaged"] for c in ctx_counters.values()) == gen_tp
    for counters in ctx_counters.values():
        assert set(counters.fallback) <= {"peer_not_engaged"}
