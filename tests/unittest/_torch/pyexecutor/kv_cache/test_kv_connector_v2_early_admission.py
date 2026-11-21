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
"""Early connector admission: an asynchronously loaded first chunk is parked at admission.

The scheduler asks the KV cache manager to run the connector query as soon as the
chunk's pages are allocated. A request that loads asynchronously keeps its pages,
leaves the context state and takes no batch slot or token budget; its load starts
at once and it re-enters when the load lands.
TRTLLM_DISABLE_KV_CONNECTOR_EARLY_ADMISSION=1 restores the final-batch query.
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from test_kv_cache_v2_scheduler import ids, make_ctx_request, make_kv_cache_manager, make_scheduler
from test_kv_connector_v2_prefix import FakeKvCache, FakeRequest, make_manager

from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import KvCacheConnectorManager
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState

pytestmark = pytest.mark.cpu_only

CONNECTOR_MODULE = "tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector"
KILL_SWITCH = "TRTLLM_DISABLE_KV_CONNECTOR_EARLY_ADMISSION"
CONTEXT_INIT = LlmRequestState.CONTEXT_INIT.value
LOADING = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS.value
COMPLETE = LlmRequestState.GENERATION_COMPLETE.value
STEP = 1024
UNIT = 64


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(KILL_SWITCH, raising=False)


# ---------------------------------------------------------------------------
# Scheduler: what a park buys and what it must never do
# ---------------------------------------------------------------------------


def _parking_manager(parks=(), blockers=None):
    """A scheduler-test manager whose connector parks the requests in ``parks`` once."""
    mgr = make_kv_cache_manager(tokens_per_block=UNIT)
    mgr.early_connector_admission_blockers.return_value = [] if blockers is None else blockers
    to_park = set(parks)
    reported = set()

    def park(req):
        if req.py_request_id in reported:
            return False  # the allocation is reported once; a re-entry is not queried again
        reported.add(req.py_request_id)
        if req.py_request_id not in to_park:
            return False
        req.state_value = LOADING
        return True

    mgr.park_for_async_connector_load.side_effect = park
    return mgr


def _sched(mgr, max_batch_size=2):
    return make_scheduler(
        mgr, max_batch_size=max_batch_size, max_num_tokens=STEP, ctx_chunk_config=(None, UNIT)
    )


def test_parked_request_takes_no_slot_and_no_tokens():
    mgr = _parking_manager(parks={0})
    sched = _sched(mgr)
    warm, a, b = make_ctx_request(0, 900), make_ctx_request(1, 600), make_ctx_request(2, 400)

    out = sched.schedule_request([warm, a, b], set())

    # Two slots, 1024 tokens: both cold prompts fit whole because the parked one is free.
    assert ids(out.context_requests) == [1, 2]
    assert (a.context_chunk_size, b.context_chunk_size) == (600, 400)
    assert warm.state_value == LOADING
    assert sched._early_counts == [1, 0, 0, 900]
    assert sched._early_admission is True
    # The parked request keeps its pages.
    assert all(c.args[0] is not warm for c in mgr.free_resources.call_args_list)


def test_kill_switch_restores_the_final_batch_path(monkeypatch):
    monkeypatch.setenv(KILL_SWITCH, "1")
    mgr = _parking_manager(parks={0})
    sched = _sched(mgr)
    warm, a, b = make_ctx_request(0, 900), make_ctx_request(1, 600), make_ctx_request(2, 400)

    out = sched.schedule_request([warm, a, b], set())

    # The outcome without parking: the warm prompt holds a slot and 900 tokens.
    assert ids(out.context_requests) == [0, 1]
    assert a.context_chunk_size == UNIT
    mgr.park_for_async_connector_load.assert_not_called()
    assert sched._early_admission is False


@pytest.mark.parametrize(
    "blockers", [["sync_loads"], ["leader_broadcast_query"], ["no_early_dispatch"]]
)
def test_an_unsupported_connector_never_parks(blockers):
    mgr = _parking_manager(parks={0}, blockers=blockers)
    sched = _sched(mgr)
    out = sched.schedule_request([make_ctx_request(0, 900), make_ctx_request(1, 600)], set())
    assert ids(out.context_requests) == [0, 1]
    mgr.park_for_async_connector_load.assert_not_called()


def test_a_plain_mock_manager_leaves_it_off():
    """Anything but an explicit empty blocker list disables it."""
    mgr = make_kv_cache_manager(tokens_per_block=UNIT)
    sched = _sched(mgr)
    sched.schedule_request([make_ctx_request(0, 100)], set())
    assert sched._early_admission is False
    mgr.park_for_async_connector_load.assert_not_called()


def test_parked_request_is_not_scheduled_until_its_load_lands():
    mgr = _parking_manager(parks={0})
    sched = _sched(mgr)
    warm, a = make_ctx_request(0, 900), make_ctx_request(1, 600)

    assert ids(sched.schedule_request([warm, a], set()).context_requests) == [1]
    # Still loading: outside the schedulable range, and not queried again.
    assert ids(sched.schedule_request([warm, a], set()).context_requests) == [1]
    queried = [c.args[0].py_request_id for c in mgr.park_for_async_connector_load.call_args_list]
    assert queried.count(0) == 1
    assert sched._early_counts[:3] == [1, 0, 0]

    warm.state_value = CONTEXT_INIT  # the connector moved it back when the load landed
    out = sched.schedule_request([warm, a], set())
    assert ids(out.context_requests) == [0, 1]
    assert sched._early_counts[:3] == [1, 1, 0]
    assert sched._early_parked == {}


def test_a_request_finished_while_parked_is_dropped():
    mgr = _parking_manager(parks={0})
    sched = _sched(mgr)
    warm = make_ctx_request(0, 900)
    sched.schedule_request([warm], set())
    warm.state_value = COMPLETE  # cancelled; the executor releases it after the load
    sched.schedule_request([], set())
    assert sched._early_counts[:3] == [1, 0, 1]
    assert sched._early_parked == {}


def test_only_parked_work_is_progress_not_a_stall():
    mgr = _parking_manager(parks={0, 1})
    sched = _sched(mgr)
    out = sched.schedule_request([make_ctx_request(0, 900), make_ctx_request(1, 100)], set())
    assert out.context_requests == []
    assert sched._stalled_schedules == 0


def test_a_continuation_chunk_is_never_queried():
    mgr = _parking_manager(parks={0})
    sched = _sched(mgr)
    req = make_ctx_request(0, 900, is_first_context_chunk=False)
    out = sched.schedule_request([req], set())
    assert ids(out.context_requests) == [0]
    mgr.park_for_async_connector_load.assert_not_called()


# ---------------------------------------------------------------------------
# KV cache manager: the admission-time query against a real connector manager
# ---------------------------------------------------------------------------

PROMPT = 256  # test_kv_connector_v2_prefix.PROMPT_LEN, 8 blocks of 32
OFFERED = 192
CHUNK = 32


class _AdpMock(MagicMock):
    supports_attention_dp = True


def _connector(offered=OFFERED, load_async=True, rank=0):
    worker = _AdpMock()
    worker.get_finished.return_value = ([], [])
    worker.take_failed_load_requests.return_value = set()
    worker.take_failed_async_load_requests.return_value = set()
    scheduler = _AdpMock()
    scheduler.get_num_new_matched_tokens.return_value = (offered, load_async and offered > 0)
    scheduler.loads_async = load_async
    with patch(f"{CONNECTOR_MODULE}.mpi_rank", return_value=rank):
        return KvCacheConnectorManager(worker, scheduler, enable_attention_dp=True)


def _fake_req(request_id=0):
    req = FakeRequest(request_id=request_id, prompt_len=PROMPT)
    req.is_dummy_request = False
    req.state = LlmRequestState.CONTEXT_INIT
    req.context_chunk_size = CHUNK
    return req


def _admitted(connector, request_id=0):
    """A first chunk the scheduler has just given one chunk of pages."""
    manager = make_manager(connector)
    req = _fake_req(request_id)
    manager.kv_cache_map[request_id] = FakeKvCache(committed=0, capacity=CHUNK)
    return manager, req


def test_connector_manager_reads_the_scheduler_async_declaration():
    assert _connector(load_async=True).loads_prefix_async() is True
    assert _connector(load_async=False).loads_prefix_async() is False
    undeclared = _connector()
    undeclared.scheduler = SimpleNamespace(loads_async="yes")
    assert undeclared.loads_prefix_async() is None
    undeclared.scheduler = None
    assert undeclared.loads_prefix_async() is None


def test_blockers_for_an_async_owner_local_connector_are_empty():
    manager, _ = _admitted(_connector())
    assert manager.early_connector_admission_blockers() == []


def test_blockers_name_what_rules_it_out():
    connector = _connector(load_async=False)
    connector.enable_attention_dp = False
    manager = make_manager(connector)
    with patch(f"{CONNECTOR_MODULE}.mpi_world_size", return_value=2):
        assert manager.early_connector_admission_blockers() == [
            "sync_loads",
            "leader_broadcast_query",
            "no_early_dispatch",
        ]
    assert make_manager(None).early_connector_admission_blockers() == ["no_connector"]


def test_a_scheduler_without_early_dispatch_rules_it_out():
    connector = _connector()
    connector.scheduler = SimpleNamespace(loads_async=True)
    manager = make_manager(connector)
    assert manager.early_connector_admission_blockers() == ["no_early_dispatch"]
    assert connector.dispatch_async_loads(lambda request_ids: None) is False


def test_park_reserves_the_served_prefix_and_starts_the_load():
    connector = _connector()
    manager, req = _admitted(connector)

    assert manager.park_for_async_connector_load(req)

    kv_cache = manager.kv_cache_map[0]
    # Pages for the served prefix plus the chunk stay allocated to the request.
    assert kv_cache.capacity >= OFFERED + CHUNK
    assert req.context_current_position == OFFERED
    assert req.state == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    assert req.py_connector_allocation_reported
    assert connector.has_pending_load(req)
    connector.scheduler.update_state_after_alloc_by_layer_group.assert_called_once()
    # Handed to this rank's loaders now, behind this manager's allocation fence.
    connector.scheduler.dispatch_async_loads.assert_called_once_with(manager.wait_allocation_fences)


def test_the_final_batch_hook_does_not_query_a_parked_request_again():
    connector = _connector()
    manager, req = _admitted(connector)
    assert manager.park_for_async_connector_load(req)
    batch = SimpleNamespace(context_requests=[req], reset_context_requests=lambda: None)
    manager._run_kv_connector_hooks(batch)
    assert connector.scheduler.get_num_new_matched_tokens.call_count == 1
    assert not manager.park_for_async_connector_load(req)
    assert connector.scheduler.get_num_new_matched_tokens.call_count == 1


def test_no_hit_stays_in_the_batch_with_its_allocation_reported_once():
    connector = _connector(offered=0)
    manager, req = _admitted(connector)

    assert not manager.park_for_async_connector_load(req)

    assert req.state == LlmRequestState.CONTEXT_INIT
    assert req.py_connector_allocation_reported
    assert not connector.has_pending_load(req)
    connector.scheduler.dispatch_async_loads.assert_not_called()
    manager._run_kv_connector_hooks(
        SimpleNamespace(context_requests=[req], reset_context_requests=lambda: None)
    )
    assert connector.scheduler.get_num_new_matched_tokens.call_count == 1


def test_a_failed_early_dispatch_is_not_raised():
    connector = _connector()
    connector.scheduler.dispatch_async_loads.side_effect = RuntimeError("store down")
    manager, req = _admitted(connector)
    # The request is parked either way; get_finished reports the load it never started.
    assert manager.park_for_async_connector_load(req)
    assert connector.has_pending_load(req)


def test_pages_of_a_parked_request_are_held_until_its_load_lands():
    connector = _connector()
    manager, req = _admitted(connector)
    assert manager.park_for_async_connector_load(req)
    kv_cache = manager.kv_cache_map[0]

    # The load may still be writing: the pages must not be released under it.
    with pytest.raises(RuntimeError, match="is loading"):
        manager.free_resources(req)
    assert not kv_cache.closed

    # The load lands: the request is back in the context state and owns nothing pending.
    connector.worker.get_finished.return_value = ([], [0])
    connector.get_finished()
    assert req.state == LlmRequestState.CONTEXT_INIT
    assert not connector.has_pending_load(req)


def test_park_moves_a_side_stream_cache_onto_the_execution_stream(monkeypatch):
    class FakeEvent:
        def record(self, stream):
            self.stream = stream

    from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2

    monkeypatch.setattr(kv_cache_manager_v2.torch.cuda, "Event", FakeEvent)
    connector = _connector()
    manager, req = _admitted(connector)
    manager._allocation_stream = SimpleNamespace(cuda_stream=7)
    manager._on_allocation_stream.add(0)

    assert manager.park_for_async_connector_load(req)

    assert 0 not in manager._on_allocation_stream
    assert manager._allocation_ready[0].stream is manager._allocation_stream
    assert manager.kv_cache_map[0].cuda_stream == manager._stream.cuda_stream


# ---------------------------------------------------------------------------
# Attention DP: one rank parks, the other computes, the collectives still pair up
# ---------------------------------------------------------------------------


class _SchedRequest(FakeRequest):
    """A ``FakeRequest`` that also carries what the scheduler reads."""

    def __init__(self, request_id):
        super().__init__(request_id=request_id, prompt_len=PROMPT)
        self.state_value = CONTEXT_INIT
        self.is_dummy_request = False
        self.py_csa2_remote_tail_mode = None
        self.py_ced_replay = None
        self.lora_task_id = None
        self.expect_snapshot_points = []
        self.is_context_init_state = True
        self.is_generation_in_progress_state = False
        self.encoder_output_len = None
        self.py_encoder_output_ready_event = None
        self.py_skip_cross_kv_projection = False
        self.num_draft_tokens = 0
        self.has_draft_tokens = False

    def __getattr__(self, name):
        # Optional per-request scheduler fields read here are unset, as on a fresh request.
        if name.startswith("py_"):
            return None
        raise AttributeError(name)

    @property
    def state(self):
        return LlmRequestState(self.state_value)

    @state.setter
    def state(self, value):
        self.state_value = value.value


def _forbid_collectives(*_args, **_kwargs):
    raise AssertionError("an attention-DP scheduling pass entered a collective")


def _rank(rank, offered, vote, iterations, land_after, log):
    connector = _connector(offered=offered, rank=rank)
    kvcm = make_manager(connector)
    req = _SchedRequest(request_id=100 + rank)
    kvcm.kv_cache_map[req.py_request_id] = FakeKvCache(committed=0, capacity=PROMPT)

    mgr = make_kv_cache_manager(tokens_per_block=32)
    mgr.kv_connector_manager = connector
    mgr.early_connector_admission_blockers.side_effect = kvcm.early_connector_admission_blockers
    mgr.park_for_async_connector_load.side_effect = kvcm.park_for_async_connector_load
    sched = make_scheduler(
        mgr, max_batch_size=4, max_num_tokens=STEP, ctx_chunk_config=(None, CHUNK)
    )
    for iteration in range(iterations):
        if iteration == land_after and connector.has_pending_load(req):
            connector.worker.get_finished.return_value = ([], [req.request_id])
        out = sched.schedule_request([req], set())
        connector.get_finished()  # every owner, every iteration, as the executor does
        connector.worker.get_finished.return_value = ([], [])
        log.append((rank, iteration, ids(out.context_requests), req.state_value))
        vote(len(out.context_requests))  # the per-iteration status vote
    return sched


def test_adp_ranks_keep_issuing_identical_collectives_while_one_parks():
    iterations = 5
    barrier = threading.Barrier(2, timeout=10)
    votes = {0: 0, 1: 0}
    log = []
    results = {}
    errors = []

    def vote_for(rank):
        def vote(_batch_size):
            barrier.wait()  # a missing peer times out here instead of hanging the test
            votes[rank] += 1

        return vote

    def run(rank, offered):
        try:
            results[rank] = _rank(rank, offered, vote_for(rank), iterations, land_after=2, log=log)
        except BaseException as exc:  # surfaced below
            errors.append(exc)
            barrier.abort()

    with (
        patch(f"{CONNECTOR_MODULE}.mpi_allgather", side_effect=_forbid_collectives),
        patch(f"{CONNECTOR_MODULE}.mpi_broadcast", side_effect=_forbid_collectives),
    ):
        threads = [
            threading.Thread(target=run, args=(0, OFFERED)),  # a store hit: parks
            threading.Thread(target=run, args=(1, 0)),  # cold: computes at once
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

    assert not errors, errors
    assert votes == {0: iterations, 1: iterations}
    rank0 = [entry for entry in log if entry[0] == 0]
    rank1 = [entry for entry in log if entry[0] == 1]
    # Rank 0 parks on the first pass and computes nothing while loading. The load lands
    # in the completion poll of the third iteration, so the fourth schedules it.
    assert [entry[2] for entry in rank0] == [[], [], [], [100], [100]]
    assert rank0[0][3] == LOADING
    assert results[0]._early_counts[:2] == [1, 1]
    # Rank 1 never parks and computes on the first pass.
    assert rank1[0][2] == [101]
    assert results[1]._early_counts[0] == 0
