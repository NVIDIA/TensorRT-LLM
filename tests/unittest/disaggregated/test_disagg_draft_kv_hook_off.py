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
"""The disagg drafter context window (F0a) off and on, around the write itself.

Off (the hook False, the default): the transceiver's chunks, wire messages, byte accounting, aux
layout, send path and receive fences are exactly the target-only ones. On: the context side waits
on the drafter-store event before publishing (unless the test-only knob skips it), the aux slot
carries the drafter context lengths, the generation side hands them to the worker, and the draft
pool's readiness event joins the receive fence.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import msgpack
import numpy as np
import pytest
import torch

from tensorrt_llm import DisaggregatedParams
from tensorrt_llm._torch.disaggregation import transceiver as transceiver_module
from tensorrt_llm._torch.disaggregation.base import CacheExtent, CacheKind, Chunk, TokenRange
from tensorrt_llm._torch.disaggregation.draft_kv import (
    DraftKvCounters,
    decode_draft_ctx_state,
    encode_draft_ctx_state,
)
from tensorrt_llm._torch.disaggregation.native.auxiliary import AuxBuffer
from tensorrt_llm._torch.disaggregation.native.transfer import RecvReqInfo
from tensorrt_llm._torch.disaggregation.orchestration.coordinator import DisaggTransferCoordinator
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks, runtime_hooks_of
from tensorrt_llm.disaggregated_params import DisaggScheduleStyle

pytestmark = pytest.mark.cpu_only

_RID = 42


def _chunk(draft=None, is_last=True):
    return Chunk(
        block_ids_per_layer_groups=[np.array([3, 4], dtype=np.int64)],
        kind_per_layer_group=[CacheKind.PAGED],
        token_range=TokenRange(start=0, end=64),
        is_last=is_last,
        draft_block_ids_per_layer_groups=draft,
    )


# ---------------------------------------------------------------------------
# Off: identical to the target-only path
# ---------------------------------------------------------------------------


def test_hook_is_off_for_every_model_that_does_not_set_it():
    assert RuntimeHooks().disagg_draft_kv is False
    assert runtime_hooks_of(object()).disagg_draft_kv is False
    assert KVCacheManagerV2._disagg_draft_kv_receive is False


def test_chunk_and_transceiver_carry_no_draft_set_by_default():
    assert _chunk().draft_block_ids_per_layer_groups is None
    transceiver = object.__new__(KvCacheTransceiverV2)
    assert transceiver._draft_ctx_owner is None
    assert transceiver._describe_draft_local(SimpleNamespace(prompt_len=64)) is None


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
        bounce_dst_base=None,
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
            "bounce_dst_base": None,
        }
    )
    assert info.to_bytes() == legacy
    assert RecvReqInfo.from_bytes(legacy).draft_block_ids_per_layer_groups is None


def test_recv_req_info_round_trips_the_draft_set():
    draft = [np.array([-1, -1, 9, 8], dtype=np.int64)]
    info = RecvReqInfo(
        sender_req_id=7,
        instance_name="gen",
        instance_rank=0,
        block_ids_per_layer_groups=[np.array([1], dtype=np.int64)],
        unique_rid=_RID,
        draft_block_ids_per_layer_groups=draft,
    )
    restored = RecvReqInfo.from_bytes(info.to_bytes())
    assert len(restored.draft_block_ids_per_layer_groups) == 1
    assert np.array_equal(restored.draft_block_ids_per_layer_groups[0], draft[0])


def test_chunk_bytes_count_no_draft_pages_when_off():
    stand_in = SimpleNamespace(_page_table=None)
    assert KvCacheTransceiverV2._chunk_num_bytes(stand_in, _chunk()) == 0
    transceiver = object.__new__(KvCacheTransceiverV2)
    assert KvCacheTransceiverV2._draft_chunk_num_bytes(None, _chunk(draft=[np.array([1])])) == 0
    assert transceiver._draft_page_table is None


def _request(**kwargs):
    defaults = dict(
        py_request_id=_RID,
        py_disaggregated_params=DisaggregatedParams(
            disagg_request_id=_RID, schedule_style=DisaggScheduleStyle.GENERATION_FIRST
        ),
        prompt_len=64,
        cached_tokens=0,
        py_draft_tokens=[],
        get_last_tokens=lambda: [17],
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_aux_layout_is_unchanged_without_the_draft_slot():
    aux = AuxBuffer(max_slot_num=4, beam_width=1, max_draft_len=3)
    assert aux.meta.ptrs.size == 4
    slot = aux.alloc_slot().id
    aux.fill_slot(slot, _request(py_draft_ctx_aux=(64, 1, True)))
    assert aux.get_slot_draft_ctx(slot) is None


def test_aux_draft_slot_carries_the_context_state():
    aux = AuxBuffer(max_slot_num=4, beam_width=1, max_draft_len=3, draft_ctx=True)
    plain = AuxBuffer(max_slot_num=4, beam_width=1, max_draft_len=3)
    assert aux.meta.ptrs.size == 5
    assert np.array_equal(aux.meta.item_sizes[:4], plain.meta.item_sizes)
    assert aux.meta.item_sizes[4] == 3 * 4
    slot = aux.alloc_slot().id
    aux.fill_slot(slot, _request(py_draft_ctx_aux=(9000, 3, True)))
    assert aux.get_slot_draft_ctx(slot) == (9000, 3, True)
    aux.fill_slot(slot, _request(py_draft_ctx_aux=(0, 0, False)))  # the window did not travel
    assert aux.get_slot_draft_ctx(slot) == (0, 0, False)
    aux.fill_slot(slot, _request())  # no decision at all reads as "not shipped"
    assert aux.get_slot_draft_ctx(slot) == (0, 0, False)


def test_context_state_codec_round_trips_and_ignores_foreign_values():
    state = encode_draft_ctx_state(9000, 3, True)
    assert state == [9000, 3, 1]
    assert decode_draft_ctx_state(state) == (9000, 3, True)
    assert decode_draft_ctx_state(encode_draft_ctx_state(64, 0, False)) == (64, 0, False)
    for foreign in (
        None,
        b"",
        b"F0A1" + b"\x00" * 17,
        "opaque",
        [1, 2],
        [1, 2, 3, 4],
        [-1, 0, 1],
        [1, 0, 2],
        ["a", 0, 1],
    ):
        assert decode_draft_ctx_state(foreign) is None


# ---------------------------------------------------------------------------
# The context side's send: wait on the drafter-store event, decide, then publish
# ---------------------------------------------------------------------------

_CONTEXT_FIRST = dict(schedule_style=DisaggScheduleStyle.CONTEXT_FIRST)
_GENERATION_FIRST = dict(schedule_style=DisaggScheduleStyle.GENERATION_FIRST)


class _ReadyEvent:
    def __init__(self, order):
        self._order = order

    def synchronize(self):
        self._order.append("wait")


class _Owner:
    def __init__(self, order, ctx_len, ctx_base):
        self.ready = {} if ctx_len is None else {_RID: (_ReadyEvent(order), ctx_len, ctx_base)}
        self.seeds = {}

    def take_disagg_draft_ctx_ready(self, request_id):
        return self.ready.pop(request_id, None)

    def seed_disagg_draft_ctx(self, request_id, ctx_len, ctx_base, prompt_len):
        self.seeds[request_id] = (ctx_len, ctx_base, prompt_len)


def _engage(transceiver, owner, knob=False):
    transceiver._draft_ctx_owner = owner
    transceiver._send_before_draft_ctx_event = knob
    transceiver._draft_page_table = SimpleNamespace(
        tokens_per_block=32, layer_groups=[SimpleNamespace(sliding_window_size=None)]
    )
    transceiver._draft_kv_ctx_counters = DraftKvCounters("context")
    transceiver._draft_kv_gen_counters = DraftKvCounters("generation")


def _send(monkeypatch, *, owner=None, request=None, knob=False, chunk=None):
    order = [] if owner is None else owner.order
    published = []
    chunk = chunk if chunk is not None else _chunk(draft=[np.array([1, 2])])

    class _Publish:
        def __init__(self, session, req):
            pass

        def publish(self, extent):
            order.append("publish")
            published.append(extent)

    monkeypatch.setattr(transceiver_module, "PeerPublish", _Publish)
    transceiver = object.__new__(KvCacheTransceiverV2)
    session = MagicMock()
    transceiver._enable_pipelined_transfer = False
    transceiver._send_reqs = {}
    transceiver._get_or_create_send_session = MagicMock(return_value=session)
    transceiver._create_cache_extent = MagicMock(return_value=CacheExtent(name=_RID, local=chunk))
    transceiver._finalize_send = MagicMock()
    if owner is not None:
        _engage(transceiver, owner, knob)
    request = request if request is not None else _request()
    request.set_kv_cache_transfer_start = lambda _ts: None
    KvCacheTransceiverV2.respond_and_send_async(transceiver, request)
    session.set_exception.assert_not_called()
    transceiver._finalize_send.assert_called_once()
    return order, transceiver, request, published[0]


def _owner(ctx_len=64, ctx_base=0):
    order = []
    owner = _Owner(order, ctx_len, ctx_base)
    owner.order = order
    return owner


def _params(**kwargs):
    return DisaggregatedParams(disagg_request_id=_RID, **kwargs)


def test_send_without_the_draft_set_publishes_untouched(monkeypatch):
    order, _, request, extent = _send(monkeypatch)
    assert order == ["publish"]
    assert not hasattr(request, "py_draft_ctx_aux")
    assert extent.local.draft_block_ids_per_layer_groups is not None  # nothing strips it


def test_send_waits_on_the_drafter_store_before_publishing(monkeypatch):
    owner = _owner(ctx_len=64, ctx_base=0)
    order, transceiver, request, extent = _send(monkeypatch, owner=owner)
    assert order == ["wait", "publish"]
    assert request.py_draft_ctx_aux == (64, 0, True)
    assert extent.local.draft_block_ids_per_layer_groups is not None
    assert owner.ready == {}
    assert transceiver._draft_kv_ctx_counters.sent == 1
    assert not transceiver._draft_kv_ctx_counters.fallback


def test_a_reuse_hit_ships_its_base(monkeypatch):
    owner = _owner(ctx_len=64, ctx_base=1)  # the drafter stored the tail from position 20
    order, transceiver, request, extent = _send(
        monkeypatch,
        owner=owner,
        request=_request(py_disaggregated_params=_params(**_CONTEXT_FIRST)),
    )
    assert order == ["wait", "publish"]
    assert request.py_draft_ctx_aux == (64, 1, True)
    assert extent.local.draft_block_ids_per_layer_groups is not None
    assert transceiver._draft_kv_ctx_counters.sent == 1


def test_context_first_state_survives_the_serving_params_round_trip():
    """CTX response -> serving DisaggregatedParams (draft_ctx_state) -> generation request."""
    from tensorrt_llm.serve.openai_protocol import (
        to_disaggregated_params,
        to_llm_disaggregated_params,
    )

    for state in ((9000, 256, True), (33, 0, True), (0, 0, False)):
        ctx_side = _params(
            **_CONTEXT_FIRST,
            request_type="context_only",
            draft_ctx_state=encode_draft_ctx_state(*state),
        )
        wire = to_disaggregated_params(ctx_side)
        assert wire.draft_ctx_state == encode_draft_ctx_state(*state)
        assert wire.encoded_opaque_state is None  # never the C++ transceiver-state carrier
        # The serving JSON hop and back.
        wire = type(wire).model_validate_json(wire.model_dump_json())
        gen_side = to_llm_disaggregated_params(
            wire.model_copy(update={"request_type": "generation_only"})
        )
        assert decode_draft_ctx_state(gen_side.draft_ctx_state) == state
        assert gen_side.opaque_state is None
        gen_request = _gen_request(gen_side, prompt_len=state[0] or 64)
        expected = state[:2] if state[2] else None
        assert KvCacheTransceiverV2._received_draft_ctx_state(gen_request) == expected


def test_test_knob_publishes_before_the_drafter_store(monkeypatch):
    owner = _owner()
    order, _, request, _ = _send(monkeypatch, owner=owner, knob=True)
    assert order == ["publish"]
    assert request.py_draft_ctx_aux == (64, 0, True)


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
    owner = _owner(ctx_len=ctx_len, ctx_base=ctx_base)
    params = _params(**_CONTEXT_FIRST) if context_first else None
    request = _request(py_disaggregated_params=params) if params else _request()
    order, transceiver, request, extent = _send(
        monkeypatch, owner=owner, request=request, chunk=_chunk(draft=draft)
    )

    # The request is published, never failed; the target set is untouched.
    assert order[-1] == "publish"
    assert np.array_equal(extent.local.block_ids_per_layer_groups[0], [3, 4])
    assert request.py_draft_ctx_aux == (0, 0, False)
    counters = transceiver._draft_kv_ctx_counters
    assert counters.sent == 0 and counters.fallback == {reason: 1}
    if context_first:
        # The receiver learns it from the response first, so the window's pages are left out.
        assert extent.local.draft_block_ids_per_layer_groups is None
    else:
        # The receiver already asked for the pages: they still travel, marked unusable.
        assert extent.local.draft_block_ids_per_layer_groups is draft


def test_context_first_response_carries_the_decision(monkeypatch):
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._send_reqs = {}
    transceiver._dp_rank = 0
    transceiver._context_info_endpoint = "tcp://ctx:1"
    seen = {}
    monkeypatch.setattr(
        transceiver_module, "ContextPhaseParams", lambda **kwargs: seen.update(kwargs) or kwargs
    )
    request = _request(py_disaggregated_params=_params(**_CONTEXT_FIRST))
    request.py_draft_ctx_aux = (0, 0, False)
    KvCacheTransceiverV2._finalize_send(transceiver, request, MagicMock())
    assert seen["opaque_state"] is None  # not engaged: today's response

    _engage(transceiver, _owner())
    for state in ((0, 0, False), (64, 0, True)):
        request.py_draft_ctx_aux = state
        KvCacheTransceiverV2._finalize_send(transceiver, request, MagicMock())
        # Never opaque_state: C++ ContextPhaseParams parses it as a DataTransceiverState and
        # re-serializes it, so the drafter state rides DisaggregatedParams.draft_ctx_state.
        assert seen["opaque_state"] is None
        assert request.py_draft_ctx_aux == state

    gen_first = _request()
    gen_first.py_draft_ctx_aux = (64, 0, True)
    session = MagicMock()
    KvCacheTransceiverV2._finalize_send(transceiver, gen_first, session)
    assert seen["opaque_state"] is None  # generation-first: the aux slot carries it
    session.pack_aux.assert_called_once_with(gen_first)


# ---------------------------------------------------------------------------
# The generation side: expected pages and the seed follow the context side's decision
# ---------------------------------------------------------------------------


def _gen_request(params, prompt_len=64, **kwargs):
    return _request(
        py_disaggregated_params=params,
        prompt_len=prompt_len,
        is_generation_only_request=True,
        **kwargs,
    )


def test_generation_seeds_the_received_state_or_an_empty_window():
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._seed_draft_ctx(_request(py_draft_ctx_aux=(9000, 0, True)))  # off: nothing

    owner = _owner()
    _engage(transceiver, owner)
    counters = transceiver._draft_kv_gen_counters

    # Generation-first: the aux slot.
    transceiver._seed_draft_ctx(
        _gen_request(_params(**_GENERATION_FIRST), 9000, py_draft_ctx_aux=(9000, 0, True))
    )
    assert owner.seeds[_RID] == (9000, 0, 9000)
    transceiver._seed_draft_ctx(
        _gen_request(_params(**_GENERATION_FIRST), 9000, py_draft_ctx_aux=(0, 0, False))
    )
    # Not shipped: based past every prompt page, the drafter reads nothing.
    assert owner.seeds[_RID] == (9000, 282, 9000)

    # Context-first: the response's draft_ctx_state.
    shipped = _params(**_CONTEXT_FIRST, draft_ctx_state=encode_draft_ctx_state(33, 0, True))
    transceiver._seed_draft_ctx(_gen_request(shipped, 33))
    assert owner.seeds[_RID] == (33, 0, 33)
    for params in (
        _params(**_CONTEXT_FIRST, draft_ctx_state=encode_draft_ctx_state(0, 0, False)),
        _params(**_CONTEXT_FIRST),  # no state arrived at all
    ):
        transceiver._seed_draft_ctx(_gen_request(params, 33))
        assert owner.seeds[_RID] == (33, 2, 33)

    assert counters.sent == 2
    assert counters.fallback == {"no_ctx_state": 3}


def test_context_first_generation_expects_no_draft_pages_after_a_fallback():
    transceiver = object.__new__(KvCacheTransceiverV2)
    _engage(transceiver, _owner())
    transceiver._draft_reuse_adapter = SimpleNamespace(
        tokens_per_block=32,
        get_block_ordinals=lambda req, idx, lg: np.array([7, 8], dtype=np.int64),
    )
    shipped = _gen_request(
        _params(**_CONTEXT_FIRST, draft_ctx_state=encode_draft_ctx_state(64, 0, True))
    )
    dropped = _gen_request(
        _params(**_CONTEXT_FIRST, draft_ctx_state=encode_draft_ctx_state(0, 0, False))
    )
    assert np.array_equal(transceiver._describe_draft_local(shipped)[0], [7, 8])
    assert transceiver._describe_draft_local(dropped) is None
    # Generation-first asks before the context side decides: always the whole window.
    assert transceiver._describe_draft_local(_gen_request(_params(**_GENERATION_FIRST))) is not None


# ---------------------------------------------------------------------------
# Receive fences
# ---------------------------------------------------------------------------


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


def test_fences_are_untouched_without_an_engaged_draft_pool():
    req = SimpleNamespace(py_request_id=1)
    target = _Fence(True)
    fences = {1: target}
    object.__new__(DisaggTransferCoordinator)._merge_draft_receive_fences([req], fences, True)
    manager = SimpleNamespace(take_disagg_receive_ready=Mock())  # not engaged
    _coordinator(manager)._merge_draft_receive_fences([req], fences, True)
    manager.take_disagg_receive_ready.assert_not_called()
    assert fences == {1: target}


def test_draft_readiness_joins_the_deferred_receive_fence():
    draft_fences = {1: _Fence(False), 2: _Fence(True)}
    manager = SimpleNamespace(
        _disagg_draft_kv_receive=True,
        take_disagg_receive_ready=lambda req: draft_fences.pop(req.py_request_id, None),
    )
    target = _Fence(True)
    fences = {1: target, 3: None}
    reqs = [SimpleNamespace(py_request_id=i) for i in (1, 2, 3)]
    _coordinator(manager)._merge_draft_receive_fences(reqs, fences, True)
    assert fences[1].query() is False  # target done, draft not yet
    assert fences[2].query() is True  # the draft event alone
    assert fences[3] is None


def test_draft_readiness_is_waited_on_without_deferral():
    draft = _Fence(False)
    manager = SimpleNamespace(
        _disagg_draft_kv_receive=True, take_disagg_receive_ready=lambda req: draft
    )
    fences = {}
    _coordinator(manager)._merge_draft_receive_fences(
        [SimpleNamespace(py_request_id=1)], fences, False
    )
    assert draft.synchronized == 1
    assert fences == {}


# ---------------------------------------------------------------------------
# The draft pool on a generation worker
# ---------------------------------------------------------------------------


class _Cache:
    def __init__(self, capacity=0, ok=True):
        self.capacity = capacity
        self.ok = ok
        self.calls = []

    def resize(self, capacity, history_length=None):
        self.calls.append((capacity, history_length))
        return self.ok


def _draft_manager(monkeypatch):
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
    manager = _draft_manager(monkeypatch)
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


def test_gen_init_without_the_transfer_records_no_fence(monkeypatch):
    manager = _draft_manager(monkeypatch)
    manager.engage_disagg_gen_init_window(lambda *args: None)
    manager._prepare_disagg_gen_init_draft(
        SimpleNamespace(py_request_id=9, prompt_len=64, py_draft_tokens=None), _Cache()
    )
    assert manager._disagg_receive_ready == {}


def test_the_transfer_needs_the_window_sizing_engaged(monkeypatch):
    manager = _draft_manager(monkeypatch)
    with pytest.raises(AssertionError, match="gen-init window sizing"):
        manager.engage_disagg_draft_kv_receive()


def test_gen_init_resize_failure_is_loud(monkeypatch):
    manager = _draft_manager(monkeypatch)
    manager.engage_disagg_gen_init_window(lambda *args: None)
    req = SimpleNamespace(py_request_id=9, prompt_len=64, py_draft_tokens=None)
    with pytest.raises(RuntimeError, match="disagg gen-init resize failed"):
        manager._prepare_disagg_gen_init_draft(req, _Cache(ok=False))
    assert manager._disagg_receive_ready == {}


def test_only_a_draft_pool_can_be_engaged(monkeypatch):
    manager = _draft_manager(monkeypatch)
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
    # Off: the same four results and the same allgather payload as before.
    result = transceiver._gen_consensus_outcome([1, 2], [], [], [1, 2], [1, 2])
    assert len(result) == 4 and len(seen[-1]) == 5
    # On: one more list in the same allgather; a veto on any rank is a veto on all.
    result = transceiver._gen_consensus_outcome(
        [1, 2], [], [], [1, 2], [1, 2], locally_draft_vetoed=[]
    )
    assert len(seen) == 2 and len(seen[-1]) == 6
    cancelled, failed, completed, verified, vetoed = result
    assert completed == [1, 2] and verified == {1, 2} and vetoed == {2}
