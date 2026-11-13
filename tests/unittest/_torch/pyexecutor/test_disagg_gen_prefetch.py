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
"""Disaggregated generation KV prefetch (TRTLLM_DISAGG_GEN_PREFETCH_REQUESTS): the gate and leases.

Prefetch is opt-in: unset or 0 leaves every pool and the admission cap unchanged. The
executor's admission (the widened cap and the generation-only prefetch slots) is covered here.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.disagg_gen_prefetch import (
    DISAGG_GEN_PREFETCH_REQUESTS_ENV,
    PRIORITIZE_FIRST_TOKEN_ENV,
    PrefetchEngagement,
    disagg_gen_prefetch_leases,
    get_disagg_gen_prefetch_requests,
    resolve_disagg_gen_prefetch_requests,
)
from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    _compute_index_mapper_capacity,
)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler.waiting_queue import FCFSWaitingQueue
from tensorrt_llm.bindings.executor import RequestType
from tensorrt_llm.executor.request import DEFAULT_REQUEST_PRIORITY
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

GEN = RequestType.REQUEST_TYPE_GENERATION_ONLY
CTX = RequestType.REQUEST_TYPE_CONTEXT_ONLY

_ELIGIBLE = dict(is_disagg=True, enable_attention_dp=False, has_pp=False)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (PRIORITIZE_FIRST_TOKEN_ENV, DISAGG_GEN_PREFETCH_REQUESTS_ENV):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def prefetch_env(monkeypatch):
    def _set(value):
        if value is None:
            monkeypatch.delenv(DISAGG_GEN_PREFETCH_REQUESTS_ENV, raising=False)
        else:
            monkeypatch.setenv(DISAGG_GEN_PREFETCH_REQUESTS_ENV, value)

    return _set


# --- env and gate ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw, expected", [(None, 0), ("", 0), ("0", 0), ("1", 1), (" 3 ", 3)])
def test_env_parsing_defaults_to_off(prefetch_env, raw, expected):
    prefetch_env(raw)
    assert get_disagg_gen_prefetch_requests() == expected


@pytest.mark.parametrize("raw", ["-1", "one", "1.5"])
def test_env_rejects_invalid_values(prefetch_env, raw):
    prefetch_env(raw)
    with pytest.raises(ValueError):
        get_disagg_gen_prefetch_requests()


@pytest.mark.parametrize(
    "override",
    [
        {"is_disagg": False},
        {"enable_attention_dp": True},
        {"has_pp": True},
        {"is_hybrid": True},
        {"has_mrope_delta_cache": True},
        {"kv_cache_manager_is_v2": False},
    ],
)
def test_resolve_ignores_unsupported_executors(prefetch_env, override):
    prefetch_env("2")
    assert resolve_disagg_gen_prefetch_requests(**{**_ELIGIBLE, **override}) == 0


def test_resolve_is_off_unless_requested(prefetch_env):
    prefetch_env(None)
    assert resolve_disagg_gen_prefetch_requests(**_ELIGIBLE) == 0
    prefetch_env("2")
    assert resolve_disagg_gen_prefetch_requests(**_ELIGIBLE) == 2


@pytest.mark.parametrize("value, prefetch", [("1", 0), ("0", 2), ("true", 2)])
def test_resolve_follows_the_prioritize_first_token_scheduler_option(
    prefetch_env, monkeypatch, value, prefetch
):
    # The scheduler turns the option on only for exactly "1"; prefetch is
    # turned off in exactly the same cases.
    prefetch_env("2")
    monkeypatch.setenv(PRIORITIZE_FIRST_TOKEN_ENV, value)
    assert resolve_disagg_gen_prefetch_requests(**_ELIGIBLE) == prefetch


def test_leases_follow_the_executor_gate(prefetch_env):
    tp2 = Mapping(world_size=2, tp_size=2, rank=0)
    adp = Mapping(world_size=2, tp_size=2, rank=0, enable_attention_dp=True)
    assert disagg_gen_prefetch_leases(is_disagg=True, mapping=tp2) == 0
    prefetch_env("1")
    assert disagg_gen_prefetch_leases(is_disagg=True, mapping=tp2) == 1
    assert disagg_gen_prefetch_leases(is_disagg=False, mapping=tp2) == 0
    assert disagg_gen_prefetch_leases(is_disagg=True, mapping=adp) == 0


@pytest.mark.parametrize("mbs", [1, 2])
def test_lease_pool_covers_the_seat_pool(prefetch_env, mbs):
    """IndexMapper leases (2 * mbs + N in disagg) cover the widened seat pool (mbs + N)."""
    prefetch_env("1")
    tp2 = Mapping(world_size=2, tp_size=2, rank=0)
    prefetch = disagg_gen_prefetch_leases(is_disagg=True, mapping=tp2)
    seats = mbs + prefetch
    leases, capacity = _compute_index_mapper_capacity(
        mbs,
        tp2,
        is_disagg=True,
        disable_overlap_scheduler=False,
        num_reserved_index_slots=0,
        has_guard_page=False,
        prefetch_headroom=prefetch,
    )
    assert leases == capacity == 2 * mbs + 1
    assert leases >= seats + mbs  # the seated requests plus as many still receiving


# --- admission ----------------------------------------------------------------------------


def _executor(max_num_active_requests=1, prefetch=0, attention_dp=False, active=()):
    ex = object.__new__(PyExecutor)
    ex.enable_attention_dp = attention_dp
    ex.max_num_active_requests = max_num_active_requests
    ex.disagg_gen_prefetch_requests = prefetch
    ex.is_benchmark_disagg = False
    ex._benchmark_fill_phase_active = False
    ex.active_requests = list(active)
    ex._kvpf_engagement = PrefetchEngagement(
        enabled=prefetch > 0, prefetch=prefetch, max_batch_size=max_num_active_requests
    )
    ex.dist = Mock()
    ex.dist.tp_size = 4
    return ex


@pytest.mark.parametrize("mbs", [1, 2, 3])
def test_admission_capacity_gate(mbs):
    assert _executor(mbs, prefetch=0)._get_request_admission_capacity() == mbs
    assert _executor(mbs, prefetch=1)._get_request_admission_capacity() == mbs + 1
    assert _executor(mbs, prefetch=2)._get_request_admission_capacity() == mbs + 2


def test_admission_capacity_unchanged_without_the_attribute():
    ex = _executor(2)
    del ex.disagg_gen_prefetch_requests
    assert ex._get_request_admission_capacity() == 2


def test_admission_capacity_ignores_prefetch_under_attention_dp():
    ex = _executor(2, prefetch=1, attention_dp=True)
    assert ex._get_request_admission_capacity() == 4 * 2


def _item(rid, request_type, child_req_ids=None):
    return RequestQueueItem(
        rid,
        SimpleNamespace(request_type=request_type, priority=DEFAULT_REQUEST_PRIORITY),
        child_req_ids=child_req_ids,
    )


def _pop(ex, items, num_active):
    queue = FCFSWaitingQueue()
    queue.add_requests(items)
    popped, placement = ex._pop_from_waiting_queue(queue, num_active)
    assert placement == {}
    return [i.id for i in popped], [i.id for i in queue]


def test_gate_off_pops_up_to_max_batch_size():
    ex = _executor(2, prefetch=0)
    assert _pop(ex, [_item(1, GEN), _item(2, GEN)], 2) == ([], [1, 2])
    assert _pop(ex, [_item(1, GEN), _item(2, GEN), _item(3, GEN)], 0) == ([1, 2], [3])


@pytest.mark.parametrize("mbs", [1, 2])
def test_prefetch_admits_one_more_generation_request(mbs):
    ex = _executor(mbs, prefetch=1)
    # A full batch decoding: the next request is admitted to receive its KV.
    assert _pop(ex, [_item(1, GEN), _item(2, GEN)], mbs) == ([1], [2])
    # Idle: mbs regular slots and one prefetch slot.
    items = [_item(i, GEN) for i in range(1, mbs + 3)]
    assert _pop(ex, items, 0) == (list(range(1, mbs + 2)), [mbs + 2])
    # Full including the prefetch slot.
    assert _pop(ex, [_item(1, GEN)], mbs + 1) == ([], [1])


def test_prefetch_slots_take_only_generation_only_requests():
    ex = _executor(1, prefetch=2)
    # A context request cannot use a prefetch slot; it and everything after it
    # go back to the front of the queue in order.
    assert _pop(ex, [_item(1, CTX), _item(2, GEN)], 1) == ([], [1, 2])
    assert _pop(ex, [_item(1, CTX), _item(2, CTX), _item(3, GEN)], 0) == ([1], [2, 3])
    assert _pop(ex, [_item(1, GEN), _item(2, CTX), _item(3, GEN)], 0) == ([1], [2, 3])
    assert _pop(ex, [_item(1, GEN), _item(2, GEN), _item(3, CTX)], 0) == ([1, 2], [3])
    # Requests with children never use a prefetch slot.
    assert _pop(ex, [_item(1, GEN, child_req_ids=[9])], 1) == ([], [1])


def test_prefetch_admission_is_logged_once_per_extra_request():
    ex = _executor(2, prefetch=1)
    _pop(ex, [_item(1, GEN), _item(2, GEN), _item(3, GEN)], 0)
    assert ex._kvpf_engagement.admitted == 1
    assert list(ex._kvpf_engagement._pending) == [3]


# --- engagement -----------------------------------------------------------------------------


def _req(rid, *, init=False, in_progress=False, complete=False):
    return SimpleNamespace(
        py_request_id=rid,
        is_disagg_generation_init_state=init,
        is_disagg_generation_transmission_in_progress=in_progress,
        is_disagg_generation_transmission_complete=complete,
    )


def test_engagement_counts_a_landing_during_decode():
    eng = PrefetchEngagement(enabled=True, prefetch=1, max_batch_size=2)
    eng.note_admitted(7, num_active=2)
    eng.observe([_req(1), _req(2), _req(7, in_progress=True)])
    assert (eng.landed_during_decode, eng.landed_idle) == (0, 0)
    eng.observe([_req(1), _req(2), _req(7, complete=True)])
    assert (eng.landed_during_decode, eng.landed_idle) == (1, 0)
    assert not eng._pending


def test_engagement_counts_an_idle_landing_and_a_gone_request():
    eng = PrefetchEngagement(enabled=True, prefetch=1, max_batch_size=1)
    eng.note_admitted(7, num_active=1)
    eng.note_admitted(8, num_active=1)
    eng.observe([_req(7, complete=True)])
    assert (eng.landed_during_decode, eng.landed_idle, eng.gone) == (0, 1, 1)
    assert "admitted=2" in eng.summary()


def test_engagement_disabled_is_inert():
    eng = PrefetchEngagement(enabled=False, prefetch=1, max_batch_size=1)
    eng.note_admitted(7, num_active=1)
    eng.observe([_req(7, complete=True)])
    assert eng.admitted == 0 and not eng._pending
