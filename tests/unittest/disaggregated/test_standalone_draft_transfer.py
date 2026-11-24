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
"""Disaggregated transfer of a unified standalone draft cache (CPU only).

A KV cache manager that owns unified standalone draft layers transfers them as
``cp_as_tp`` attention layer groups and sends the drafter's history through a
fifth aux buffer. Every other manager must keep the existing aux layout and
page-table serialization, which these tests pin down as well.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.native.auxiliary import (
    AuxBuffer,
    _decode_draft_history,
    _encode_draft_history,
    build_aux_transfer_layout,
)
from tensorrt_llm._torch.disaggregation.native.bounce.impl import block_bytes_per_group
from tensorrt_llm._torch.disaggregation.native.mixers.attention.peer import AttentionPolicy
from tensorrt_llm._torch.disaggregation.native.mixers.attention.spec import AttentionInfo
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.native.transfer import (
    _carries_draft_history,
    _make_aux_buffer,
)
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import _compute_global_layer_ids
from tensorrt_llm._torch.disaggregation.resource.page import (
    BUFFER_ENTRY_DTYPE,
    AttentionLayerGroup,
    KVCachePageTable,
    LocalLayer,
    MambaLayerGroup,
    PhysicalPool,
    PhysicalPoolGroup,
    PoolView,
)
from tensorrt_llm._torch.disaggregation.resource.utils import is_standalone_draft_group
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm.disaggregated_params import DisaggScheduleStyle

pytestmark = pytest.mark.cpu_only


def _history(window_size=None, valid_length=96, position=96) -> dict:
    return {
        "valid_length": valid_length,
        "position": position,
        "layout": {
            "num_layers": 1,
            "num_kv_heads": 8,
            "head_dim": 128,
            "dtype": "torch.bfloat16",
            "attention_backend": "TRTLLM",
            "kv_factor": 2,
            "window_size": window_size,
        },
    }


def _request(history=None) -> SimpleNamespace:
    request = SimpleNamespace(
        get_last_tokens=lambda: [11, 12],
        py_draft_tokens=[21, 22, 23],
        py_disaggregated_params=None,
        prompt_len=96,
        cached_tokens=32,
    )
    if history is not None:
        request.py_draft_transfer_history = history
    return request


# ---------------------------------------------------------------------------
# Aux buffer
# ---------------------------------------------------------------------------


def test_aux_buffer_without_draft_history_keeps_four_buffer_layout() -> None:
    buffer = AuxBuffer(max_slot_num=4, beam_width=2, max_draft_len=3)

    assert not buffer.has_draft_history
    int32 = 4
    assert buffer.meta.item_sizes.tolist() == [2 * int32, 3 * int32, 2 * int32, 2 * int32]
    assert buffer.meta.size.tolist() == [4 * size for size in buffer.meta.item_sizes.tolist()]
    assert buffer.meta.ptrs.shape == (4,)

    slot = buffer.alloc_slot().id
    buffer.fill_slot(slot, _request())
    assert buffer.get_slot_data(slot) == ([11, 12], [21, 22, 23], (96, 32))
    with pytest.raises(ValueError, match="not enabled"):
        buffer.get_slot_draft_history(slot)


def test_aux_buffer_draft_history_appends_fifth_buffer() -> None:
    plain = AuxBuffer(max_slot_num=4, beam_width=2, max_draft_len=3)
    draft = AuxBuffer(max_slot_num=4, beam_width=2, max_draft_len=3, draft_history=True)

    assert draft.has_draft_history
    assert draft.meta.ptrs.shape == (5,)
    assert draft.meta.item_sizes.tolist()[:4] == plain.meta.item_sizes.tolist()
    assert draft.meta.size.tolist()[:4] == plain.meta.size.tolist()
    int64 = 8
    assert draft.meta.item_sizes[4] == 10 * int64
    assert draft.meta.size[4] == 4 * 10 * int64


@pytest.mark.parametrize("window_size", [None, 4096])
def test_aux_buffer_draft_history_round_trip(window_size) -> None:
    buffer = AuxBuffer(max_slot_num=2, beam_width=2, max_draft_len=3, draft_history=True)
    slot = buffer.alloc_slot().id
    history = _history(window_size=window_size)

    buffer.fill_slot(slot, _request(history))

    assert buffer.get_slot_draft_history(slot) == history
    assert buffer.get_slot_data(slot) == ([11, 12], [21, 22, 23], (96, 32))


def test_aux_buffer_draft_history_requires_request_history() -> None:
    buffer = AuxBuffer(max_slot_num=1, beam_width=2, max_draft_len=3, draft_history=True)
    slot = buffer.alloc_slot().id

    with pytest.raises(ValueError, match="requires draft history"):
        buffer.fill_slot(slot, _request())


def test_aux_buffer_recycled_slot_has_no_draft_history() -> None:
    buffer = AuxBuffer(max_slot_num=1, beam_width=2, max_draft_len=3, draft_history=True)
    slot = buffer.alloc_slot().id
    buffer.fill_slot(slot, _request(_history()))
    buffer.free_slot(slot)

    assert buffer.alloc_slot().id == slot
    with pytest.raises(ValueError, match="unsupported standalone draft history"):
        buffer.get_slot_draft_history(slot)


def test_draft_history_codec_rejects_other_versions() -> None:
    values = _encode_draft_history(_history())
    assert _decode_draft_history(values) == _history()

    with pytest.raises(ValueError, match="unsupported"):
        _decode_draft_history([1] + values[1:])
    with pytest.raises(ValueError, match="unsupported"):
        _decode_draft_history(values[:-1])


def test_aux_layouts_with_and_without_draft_history_do_not_pair() -> None:
    plain = AuxBuffer(max_slot_num=2, beam_width=2, max_draft_len=3)
    draft = AuxBuffer(max_slot_num=2, beam_width=2, max_draft_len=3, draft_history=True)

    build_aux_transfer_layout(draft.meta, draft.meta)
    build_aux_transfer_layout(plain.meta, plain.meta)
    with pytest.raises(ValueError, match="do not match"):
        build_aux_transfer_layout(draft.meta, plain.meta)
    with pytest.raises(ValueError, match="do not match"):
        build_aux_transfer_layout(plain.meta, draft.meta)


def test_make_aux_buffer_adds_draft_history_only_for_unified_draft_cache() -> None:
    unified = SimpleNamespace(has_standalone_draft_cache=True, max_beam_width=1, max_draft_len=2)
    plain = SimpleNamespace(max_beam_width=1, max_draft_len=2)

    unified_buffer = _make_aux_buffer(unified, max_slots=2)
    plain_buffer = _make_aux_buffer(plain, max_slots=2)
    mocked_buffer = _make_aux_buffer(MagicMock(max_beam_width=1, max_draft_len=2), max_slots=2)

    assert _carries_draft_history(unified_buffer)
    assert not _carries_draft_history(plain_buffer)
    assert not _carries_draft_history(mocked_buffer)
    assert not _carries_draft_history(None)
    assert not _carries_draft_history(MagicMock())
    assert plain_buffer.meta.ptrs.shape == (4,)

    assert _make_aux_buffer(plain, max_slots=0) is None
    with pytest.raises(ValueError, match="requires auxiliary buffer slots"):
        _make_aux_buffer(unified, max_slots=0)


# ---------------------------------------------------------------------------
# Page table
# ---------------------------------------------------------------------------


def _views(num_layers: int, bytes_per_layer: int) -> list[PoolView]:
    views = []
    for role_index, role in enumerate(("key", "value")):
        entries = np.array(
            [
                (layer, (role_index * num_layers + layer) * bytes_per_layer, bytes_per_layer)
                for layer in range(num_layers)
            ],
            dtype=BUFFER_ENTRY_DTYPE,
        )
        views.append(
            PoolView(
                pool_idx=0,
                buffer_entries=entries,
                pool_role=frozenset({role}),
                bytes_per_layer=bytes_per_layer,
            )
        )
    return views


def _group(
    *, cp_as_tp: bool, global_layer_id: int = 61, total_heads=8, bytes_per_layer: int = 4096
) -> AttentionLayerGroup:
    group = AttentionLayerGroup(
        pool_group_idx=0,
        local_layers=[LocalLayer(local_layer_id=0, global_layer_id=global_layer_id)],
        pool_views=_views(1, bytes_per_layer),
        kv_head_num_per_rank=1,
    )
    if cp_as_tp:
        group.cp_as_tp = True
        group.total_kv_head_num = total_heads
        group.cp_as_tp_shards = 8
    return group


def _page_table(group: AttentionLayerGroup, tokens_per_block: int = 32) -> KVCachePageTable:
    return KVCachePageTable(
        tokens_per_block=tokens_per_block,
        layer_groups=[group],
        pool_groups=[
            PhysicalPoolGroup(
                pools=[PhysicalPool(base_address=0x100000, slot_bytes=4096, num_slots=16)]
            )
        ],
    )


def test_plain_attention_group_serializes_without_draft_keys() -> None:
    group = _group(cp_as_tp=False)
    data = group.to_dict()

    assert not {"cp_as_tp", "total_kv_head_num", "cp_as_tp_shards"} & data.keys()
    restored = AttentionLayerGroup.from_dict(data)
    assert not restored.cp_as_tp
    assert restored.total_kv_head_num is None
    assert restored.cp_as_tp_shards == 1


def test_standalone_draft_group_round_trips() -> None:
    restored = AttentionLayerGroup.from_dict(_group(cp_as_tp=True).to_dict())

    assert restored.cp_as_tp
    assert restored.total_kv_head_num == 8
    assert restored.cp_as_tp_shards == 8
    assert is_standalone_draft_group(restored)


def test_is_standalone_draft_group() -> None:
    assert is_standalone_draft_group(_group(cp_as_tp=True))
    assert not is_standalone_draft_group(_group(cp_as_tp=False))
    assert not is_standalone_draft_group(MambaLayerGroup(pool_group_idx=0))
    assert not is_standalone_draft_group(MagicMock())


def test_block_bytes_of_draft_group_cover_every_layer_and_role() -> None:
    group = _group(cp_as_tp=True)
    group.local_layers = [LocalLayer(local_layer_id=i, global_layer_id=61 + i) for i in range(3)]
    group.pool_views = _views(3, 1024)
    page_table = _page_table(group)

    # Three layers of K and V, one 1024-byte physical-token page each.
    assert block_bytes_per_group(page_table) == [3 * 2 * 1024]
    assert block_bytes_per_group(_page_table(_group(cp_as_tp=False))) == [4096]


def _manager(**kwargs) -> SimpleNamespace:
    # Two target layers followed by draft layer 61 at local index 2.
    return SimpleNamespace(
        impl=SimpleNamespace(layer_grouping=[[0, 1], [2]]),
        pp_layers=[0, 1, 61],
        layer_offsets={0: 0, 1: 1, 61: 2},
        **kwargs,
    )


def test_global_layer_ids_name_unified_draft_layers_by_model_id() -> None:
    manager = _manager(has_standalone_draft_cache=True, draft_layer_ids=[61])

    assert _compute_global_layer_ids(manager, 0) == [0, 1]
    assert _compute_global_layer_ids(manager, 1) == [61]


def test_global_layer_ids_ignore_draft_ids_without_unified_cache() -> None:
    # One-model MTP/Eagle managers expose draft_layer_ids without a unified cache.
    manager = _manager(draft_layer_ids=[7])

    assert _compute_global_layer_ids(manager, 0) == [0, 1]
    assert _compute_global_layer_ids(manager, 1) == [61]


# ---------------------------------------------------------------------------
# Attention peer policy
# ---------------------------------------------------------------------------


def _rank_info(*, tp=1, tp_rank=0, cp=1, cp_rank=0, attention_dp=False, page_table=None):
    return RankInfo(
        instance_name="instance",
        instance_rank=tp_rank * cp + cp_rank,
        tp_size=tp,
        tp_rank=tp_rank,
        pp_size=1,
        pp_rank=0,
        layer_num_per_pp=[62],
        sender_endpoints=[],
        self_endpoint="",
        transfer_engine_info=b"",
        dp_size=tp if attention_dp else 1,
        dp_rank=tp_rank if attention_dp else 0,
        cp_size=cp,
        cp_rank=cp_rank,
        attention=AttentionInfo(
            kv_heads_per_rank=1,
            tokens_per_block=32,
            dims_per_head=576,
            element_bytes=1,
            enable_attention_dp=attention_dp,
            is_mla=True,
        ),
        page_table=page_table,
    )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, (1, 0)),
        ({"tp": 4, "tp_rank": 3}, (4, 3)),
        ({"tp": 4, "tp_rank": 3, "attention_dp": True}, (1, 0)),
        ({"cp": 8, "cp_rank": 5}, (8, 5)),
        ({"tp": 2, "tp_rank": 1, "cp": 4, "cp_rank": 2}, (8, 6)),
        ({"tp": 2, "tp_rank": 1, "cp": 4, "cp_rank": 2, "attention_dp": True}, (8, 6)),
    ],
)
def test_draft_rank_grid(kwargs, expected) -> None:
    assert AttentionPolicy.draft_rank_grid(_rank_info(**kwargs)) == expected


def test_draft_group_rank_info_flattens_helix() -> None:
    rank_info = _rank_info(tp=2, tp_rank=1, cp=4, cp_rank=2, attention_dp=True)
    group = _group(cp_as_tp=True)
    group.kv_head_num_per_rank = 2

    draft = AttentionPolicy.draft_group_rank_info(rank_info, group)

    assert (draft.tp_size, draft.tp_rank) == (8, 6)
    assert (draft.cp_size, draft.cp_rank, draft.dp_size) == (1, 0, 1)
    assert draft.attention.kv_heads_per_rank == 2
    assert not draft.attention.is_mla
    assert not draft.attention.enable_attention_dp
    # The target's rank info is left untouched.
    assert (rank_info.cp_size, rank_info.attention.is_mla) == (4, True)


def test_validate_peer_compatible_accepts_matching_draft_groups() -> None:
    local = _page_table(_group(cp_as_tp=True))
    peer = _page_table(_group(cp_as_tp=True))

    AttentionPolicy.validate_peer_compatible(local, peer, 32, 32)
    # Same bytes per token with a different block size is compatible.
    peer_64 = _page_table(_group(cp_as_tp=True, bytes_per_layer=8192), tokens_per_block=64)
    AttentionPolicy.validate_peer_compatible(local, peer_64, 32, 64)
    # Plain groups are not inspected.
    AttentionPolicy.validate_peer_compatible(
        _page_table(_group(cp_as_tp=False)), _page_table(_group(cp_as_tp=False, bytes_per_layer=1))
    )


@pytest.mark.parametrize(
    ("peer_group", "match"),
    [
        (_group(cp_as_tp=False), "disagree on standalone draft storage"),
        (_group(cp_as_tp=True, total_heads=16), "different global KV head counts"),
        (_group(cp_as_tp=True, bytes_per_layer=2048), "context_kv_dtype"),
    ],
)
def test_validate_peer_compatible_rejects_mismatched_draft_groups(peer_group, match) -> None:
    local = _page_table(_group(cp_as_tp=True))

    with pytest.raises(ValueError, match=match):
        AttentionPolicy.validate_peer_compatible(local, _page_table(peer_group), 32, 32)


def _windowed(window, *, cp_as_tp: bool = True) -> AttentionLayerGroup:
    """A draft group whose manager declares a drafter history window (None: the whole history)."""
    group = _group(cp_as_tp=cp_as_tp)
    group.draft_window_size = window
    group.draft_window_declared = True
    return group


@pytest.mark.parametrize("window", [1024, None])
def test_declared_draft_window_round_trips(window) -> None:
    data = _windowed(window).to_dict()

    assert data["draft_window_size"] == window
    restored = AttentionLayerGroup.from_dict(data)
    assert restored.draft_window_declared
    assert restored.draft_window_size == window


def test_undeclared_draft_window_serializes_as_before() -> None:
    # A draft group without a declared window, and every plain group, carry no window key.
    draft = _group(cp_as_tp=True).to_dict()
    assert "draft_window_size" not in draft
    assert not AttentionLayerGroup.from_dict(draft).draft_window_declared
    assert "draft_window_size" not in _windowed(1024, cp_as_tp=False).to_dict()


@pytest.mark.parametrize(("ctx_window", "gen_window"), [(1024, None), (None, 1024), (1024, 2048)])
def test_validate_peer_compatible_rejects_another_draft_window(ctx_window, gen_window) -> None:
    """Peers whose draft caches keep different history windows are refused when they pair.

    A windowed context drafter in front of a whole-history generation drafter, the reverse, or
    two windows: refused from the page table the receiver decodes, before any KV moves.
    """
    ctx = _rank_info(cp=8, page_table=_page_table(_windowed(ctx_window)))
    gen = _page_table(_windowed(gen_window))

    received = RankInfo.from_bytes(ctx.to_bytes()).page_table
    with pytest.raises(ValueError, match="different drafter history windows on layer 61"):
        AttentionPolicy.validate_peer_compatible(gen, received)
    with pytest.raises(
        ValueError, match="Keep the same draft window on the context and generation servers"
    ):
        AttentionPolicy.validate_peer_compatible(received, gen, 32, 32)


@pytest.mark.parametrize(
    ("local", "peer"),
    [
        (_windowed(1024), _windowed(1024)),
        (_windowed(None), _windowed(None)),
        # A peer that does not declare its window is not compared.
        (_windowed(1024), _group(cp_as_tp=True)),
        (_group(cp_as_tp=True), _windowed(None)),
    ],
)
def test_validate_peer_compatible_accepts_the_same_or_an_undeclared_draft_window(
    local, peer
) -> None:
    AttentionPolicy.validate_peer_compatible(_page_table(local), _page_table(peer), 32, 32)


# ---------------------------------------------------------------------------
# Transceiver
# ---------------------------------------------------------------------------


def _transceiver(manager) -> KvCacheTransceiverV2:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._kv_cache_manager = manager
    return transceiver


def _draft_manager(**kwargs) -> SimpleNamespace:
    fields = dict(
        has_standalone_draft_cache=True,
        _has_cp_helix=False,
        draft_max_position_embeddings=None,
        draft_window_size=None,
        draft_cp_rank=0,
        restore_draft_history=MagicMock(),
    )
    fields.update(kwargs)
    return SimpleNamespace(**fields)


def _transfer_request(style=DisaggScheduleStyle.CONTEXT_FIRST, **kwargs) -> SimpleNamespace:
    fields = dict(
        py_request_id=5,
        prompt_len=96,
        py_disaggregated_params=SimpleNamespace(schedule_style=style),
    )
    fields.update(kwargs)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize("style", list(DisaggScheduleStyle))
def test_need_aux_transfer(style) -> None:
    request = _transfer_request(style)
    generation_first = style == DisaggScheduleStyle.GENERATION_FIRST

    assert _transceiver(_draft_manager())._need_aux_transfer(request)
    assert _transceiver(SimpleNamespace())._need_aux_transfer(request) == generation_first
    assert _transceiver(MagicMock())._need_aux_transfer(request) == generation_first


@pytest.mark.parametrize(
    ("manager_kwargs", "history_kwargs"),
    [
        ({}, {}),
        ({}, {"window_size": 64, "valid_length": 64}),
        ({"draft_max_position_embeddings": 64}, {"valid_length": 64}),
        ({"draft_max_position_embeddings": 64}, {"valid_length": 0}),
    ],
)
def test_validate_draft_history_range_accepts(manager_kwargs, history_kwargs) -> None:
    transceiver = _transceiver(_draft_manager(**manager_kwargs))

    transceiver._validate_draft_history_range(_transfer_request(), _history(**history_kwargs))


@pytest.mark.parametrize(
    ("manager_kwargs", "history_kwargs"),
    [
        ({}, {"valid_length": 95}),
        ({}, {"position": 128}),
        ({}, {"window_size": 64}),
        ({"draft_max_position_embeddings": 64}, {"valid_length": 96}),
    ],
)
def test_validate_draft_history_range_rejects(manager_kwargs, history_kwargs) -> None:
    transceiver = _transceiver(_draft_manager(**manager_kwargs))

    with pytest.raises(ValueError, match="covering the complete prompt"):
        transceiver._validate_draft_history_range(_transfer_request(), _history(**history_kwargs))


def test_helix_draft_history_covers_the_global_prompt() -> None:
    transceiver = _transceiver(_draft_manager(_has_cp_helix=True))
    request = _transfer_request(prompt_len=12, total_input_len_cp=96)

    transceiver._validate_draft_history_range(request, _history())


def test_restore_draft_history() -> None:
    manager = _draft_manager()
    history = _history()

    _transceiver(manager)._restore_draft_history(
        _transfer_request(py_draft_transfer_history=history)
    )
    manager.restore_draft_history.assert_called_once_with(5, history)

    with pytest.raises(ValueError, match="metadata is missing"):
        _transceiver(manager)._restore_draft_history(_transfer_request())
    # Without a unified draft cache the request is left alone.
    _transceiver(SimpleNamespace())._restore_draft_history(MagicMock())


def _describe(manager, request, ordinals, cached_blocks=0, shards=2) -> np.ndarray:
    transceiver = _transceiver(manager)
    transceiver._reuse_adapter = SimpleNamespace(
        tokens_per_block=32, get_block_ordinals=lambda *_: np.asarray(ordinals, dtype=np.int64)
    )
    group = _group(cp_as_tp=True)
    group.cp_as_tp_shards = shards
    return transceiver._describe_draft_group(request, 0, group, cached_blocks)


def test_describe_draft_group_keeps_the_rank_shard_of_prompt_pages() -> None:
    # Five global prompt pages; draft rank 3 % 2 == 1 keeps pages 1 and 3.
    request = _transfer_request(
        prompt_len=160, is_generation_only_request=False, context_remaining_length=0
    )
    manager = _draft_manager(draft_cp_rank=3)

    assert _describe(manager, request, [7, 9, 11]).tolist() == [7, 9]
    assert _describe(manager, request, [7, 9], cached_blocks=1).tolist() == [-1, 9]
    assert _describe(manager, request, [4, 5, 6, 7, 8], shards=1).tolist() == [4, 5, 6, 7, 8]


def test_describe_draft_group_requires_full_history_pages() -> None:
    complete = _transfer_request(
        prompt_len=160, is_generation_only_request=True, context_remaining_length=0
    )
    in_flight = _transfer_request(
        prompt_len=160, is_generation_only_request=False, context_remaining_length=64
    )

    with pytest.raises(ValueError, match="missing allocated prompt pages"):
        _describe(_draft_manager(draft_cp_rank=1), complete, [7, -1])
    with pytest.raises(ValueError, match="missing allocated prompt pages"):
        _describe(_draft_manager(draft_cp_rank=1), complete, [7])
    # Earlier context chunks and windowed drafters may hold fewer pages.
    assert _describe(_draft_manager(draft_cp_rank=1), in_flight, [7]).tolist() == [7, -1]
    windowed = _draft_manager(draft_cp_rank=1, draft_window_size=64)
    assert _describe(windowed, complete, [-1, 9]).tolist() == [-1, 9]
