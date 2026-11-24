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
"""KVCacheManagerV2 parked conversation caches (block_reuse_config.max_parked_conversation_caches)
on the CPU: which sent context caches park, how many stay, and when they close."""

from collections import OrderedDict
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.dkv import Dkv
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState

pytestmark = pytest.mark.cpu_only


class _Cache:
    """A native cache: open until closed, its page index rows set or cleared."""

    def __init__(self, active=True, beam_width=1):
        self.is_active, self.beam_width = active, beam_width
        self.closed, self.cleared = False, []

    def set_base_page_index_buf(self, beam, pool, buf):
        assert buf is None
        self.cleared.append((int(beam), pool))

    def close(self):
        assert not self.closed
        self.closed = True

    def discard_pending_stats(self):
        pass


class _Rows:
    def __init__(self):
        self.removed = []

    def record_read(self, stream):
        pass

    def remove(self, cache):
        self.removed.append(cache)


def _manager(max_parked=2, dkv=None, **attributes):
    manager = object.__new__(KVCacheManagerV2)
    fields = dict(
        _max_parked_kv_caches=max_parked,
        _parked_kv_caches=OrderedDict(),
        _parked_counts=dict.fromkeys(
            (
                "parked",
                "taken_over",
                "closed_early",
                "newer_turn",
                "over_limit",
                "no_pages",
                "reset",
            ),
            0,
        ),
        _early_freed_index_requests=set(),
        sparse_metadata_batch=_Rows(),
        _stream=SimpleNamespace(cuda_stream=0),
        num_pools=2,
        _dkv=dkv,
    )
    fields.update(attributes)
    for name, value in fields.items():
        setattr(manager, name, value)
    return manager


def _sent(rid, conversation="c1", **overrides):
    fields = dict(
        py_request_id=rid,
        is_dummy_request=False,
        py_conversation_params=None
        if conversation is None
        else SimpleNamespace(conversation_id=conversation),
        is_context_only_request=True,
        context_remaining_length=0,
        state=LlmRequestState.DISAGG_CONTEXT_COMPLETE,
        py_kv_transfer_timed_out=False,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_a_sent_context_parks_and_gives_up_its_index_rows():
    manager = _manager()
    cache = _Cache()
    assert manager._park_kv_cache(_sent(1), cache)
    assert not cache.closed
    assert list(manager._parked_kv_caches.items()) == [("c1", cache)]
    # The index slot goes to another request: the parked cache keeps none of its rows.
    assert manager.sparse_metadata_batch.removed == [cache]
    assert cache.cleared == [(0, 0), (0, 1)]
    # A slot already released early has no rows left to clear.
    early = _Cache()
    manager._early_freed_index_requests.add(2)
    assert manager._park_kv_cache(_sent(2, conversation="c2"), early)
    assert early.cleared == [] and manager.sparse_metadata_batch.removed == [cache]


@pytest.mark.parametrize(
    "request_fields, cache_fields, max_parked",
    [
        (dict(conversation=None), {}, 2),
        (dict(is_context_only_request=False), {}, 2),
        (dict(context_remaining_length=64), {}, 2),
        (dict(state=LlmRequestState.DISAGG_TRANS_ERROR), {}, 2),
        (dict(py_kv_transfer_timed_out=True), {}, 2),
        (dict(is_dummy_request=True), {}, 2),
        ({}, dict(active=False), 2),
        ({}, {}, 0),
    ],
    ids=[
        "no_conversation",
        "not_context_only",
        "prompt_incomplete",
        "transfer_error",
        "timed_out",
        "dummy",
        "suspended",
        "disabled",
    ],
)
def test_other_caches_close_as_usual(request_fields, cache_fields, max_parked):
    manager = _manager(max_parked=max_parked)
    assert not manager._park_kv_cache(_sent(1, **request_fields), _Cache(**cache_fields))
    assert not manager._parked_kv_caches


def test_a_dkv_rank_ignores_its_own_transfer_timeout():
    """The ranks of a DKV group free a request at the same step; a timeout one rank observed alone
    must not make it close a cache its peers park."""
    ranks = [_manager(dkv=Dkv(2, rank, 64, lambda obj: [obj, obj])) for rank in range(2)]
    parked = [
        rank._park_kv_cache(_sent(1, py_kv_transfer_timed_out=(i == 0)), _Cache())
        for i, rank in enumerate(ranks)
    ]
    assert parked == [True, True]


def test_the_newest_turn_and_the_newest_conversations_stay():
    manager = _manager(max_parked=2)
    first, second = _Cache(), _Cache()
    manager._park_kv_cache(_sent(1, conversation="a"), first)
    manager._park_kv_cache(_sent(2, conversation="a"), second)
    assert first.closed and not second.closed  # a newer turn of the same conversation
    b, c = _Cache(), _Cache()
    manager._park_kv_cache(_sent(3, conversation="b"), b)
    manager._park_kv_cache(_sent(4, conversation="c"), c)
    assert second.closed and list(manager._parked_kv_caches) == ["b", "c"]  # oldest first
    assert manager._parked_counts == {
        "parked": 4,
        "taken_over": 0,
        "closed_early": 2,
        "newer_turn": 1,
        "over_limit": 1,
        "no_pages": 0,
        "reset": 0,
    }


def test_the_next_turn_closes_the_parked_cache_once_it_holds_its_own():
    manager = _manager()
    parked = _Cache()
    manager._park_kv_cache(_sent(1, conversation="a"), parked)
    manager._close_parked_kv_cache(_sent(9, conversation="other"))
    assert not parked.closed
    manager._close_parked_kv_cache(_sent(2, conversation="a"))
    assert parked.closed and not manager._parked_kv_caches
    assert manager._parked_counts["taken_over"] == 1


def _resizable(manager, rid, needs_room):
    """A context cache whose growth fails while any parked cache is open."""
    cache = _Cache()
    cache.capacity = 0

    def resize(capacity, history=None):
        if needs_room and manager._parked_kv_caches:
            return False
        cache.capacity = capacity
        return True

    cache.resize = resize
    manager.kv_cache_map = {rid: cache}
    return cache


@pytest.mark.parametrize("needs_room", [False, True])
def test_a_failed_context_growth_closes_every_parked_cache_and_retries(needs_room):
    manager = _manager(
        _has_cp_helix=False,
        is_disagg=True,
        num_extra_kv_tokens=0,
        _standalone_draft_reserve=0,
        _connector_reservations_enabled=lambda: False,
        _fill_fresh_kv_pages=lambda rid: None,
        _log_window_crossing=lambda *args: None,
    )
    parked = [_Cache(), _Cache()]
    manager._park_kv_cache(_sent(1, conversation="a"), parked[0])
    manager._park_kv_cache(_sent(2, conversation="b"), parked[1])
    cache = _resizable(manager, 7, needs_room)
    req = SimpleNamespace(
        py_request_id=7,
        is_disagg_generation_init_state=False,
        is_dummy_request=False,
        context_current_position=0,
        is_first_context_chunk=True,
    )
    assert manager.resize_context(req, 128)
    assert cache.capacity == 128
    assert [p.closed for p in parked] == [needs_room, needs_room]
    assert manager._parked_counts["no_pages"] == (2 if needs_room else 0)


def test_a_reset_or_shutdown_closes_the_parked_caches_first():
    calls = []
    manager = _manager(conversation_manager=None, _sharing=None)
    parked = _Cache()
    manager._park_kv_cache(_sent(1), parked)

    def clear_reusable_blocks():
        # The native manager refuses while any cache is open.
        assert not manager._parked_kv_caches
        calls.append("clear")

    manager.impl = SimpleNamespace(clear_reusable_blocks=clear_reusable_blocks)
    manager.reset_reuse_state()
    assert parked.closed and calls == ["clear"]
    assert manager._parked_counts["reset"] == 1


def test_free_resources_parks_instead_of_closing():
    removed = []
    cache = _Cache()
    manager = _manager(
        _inflight_prefix=None,
        _pending_preemption={},
        kv_connector_manager=None,
        conversation_manager=None,
        _allocated_draft_lens={},
        _inflight_draft_lens={},
        _standalone_draft_layer_ids=(),
        _request_stats_enabled_ids=set(),
        _fresh_pages_filled={},
        _disagg_receive_ready={},
        _on_allocation_stream=set(),
        _allocation_ready={},
        kv_cache_map={1: cache},
        impl=SimpleNamespace(clear_stats_excluded=lambda rid: None),
        _sharing=None,
        index_mapper=SimpleNamespace(remove_sequence=removed.append),
    )
    manager.free_resources(_sent(1))
    assert not cache.closed and manager._parked_kv_caches == {"c1": cache}
    assert manager.kv_cache_map == {} and removed == [1]
