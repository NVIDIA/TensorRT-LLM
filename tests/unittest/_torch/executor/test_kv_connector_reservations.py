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

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.connectors import kv_cache_connector as connector
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import LlmRequestState

pytestmark = pytest.mark.cpu_only


class ReservationScheduler(connector.KvCacheConnectorScheduler):
    def __init__(self):
        super().__init__(llm_args=None)
        self.protected = {}
        self.releases = []
        self.queries = []
        self.is_async = True

    def reserve_prefix(self, req, num_computed_tokens, reservation_id):
        self.queries.append((req.request_id, num_computed_tokens, reservation_id))
        self.protected[reservation_id] = set(range(num_computed_tokens, 96))
        return 96 - num_computed_tokens, self.is_async

    def release_prefix_reservation(self, req, reservation_id, start, end):
        self.releases.append((reservation_id, start, end))
        for position in range(start, end):
            self.protected[reservation_id].remove(position)
        if not self.protected[reservation_id]:
            del self.protected[reservation_id]

    def replace_source(self):
        if self.protected:
            raise RuntimeError("Source is reserved")

    def build_connector_meta(self, scheduler_output):
        return scheduler_output

    def get_num_new_matched_tokens(self, req, num_computed_tokens):
        return 0, False

    def update_state_after_alloc(self, req, block_ids):
        pass

    def request_finished(self, req, cache_block_ids):
        return False


class ReservationWorker(connector.KvCacheConnectorWorker):
    def __init__(self):
        super().__init__(llm_args=None)
        self.started = []
        self.finished = []
        self.legacy_finished = ([], [])

    def register_kv_caches(self, kv_cache_tensor):
        pass

    def start_load_kv(self, stream):
        self.started.extend(self.get_connector_meta().prefix_loads)

    def wait_for_layer_load(self, layer_idx, stream):
        pass

    def save_kv_layer(self, layer_idx, stream):
        pass

    def wait_for_save(self, stream):
        pass

    def get_finished(self, finished_gen_req_ids, started_loading_req_ids):
        result = self.legacy_finished
        self.legacy_finished = ([], [])
        return result

    def get_finished_prefix_loads(self):
        result = self.finished
        self.finished = []
        return result


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setattr(connector, "mpi_rank", lambda: 0)
    monkeypatch.setattr(connector, "mpi_broadcast", lambda result, root: result)
    monkeypatch.setattr(connector, "mpi_allgather", lambda result: [result])
    monkeypatch.setattr(connector, "mpi_world_size", lambda: 1)
    result = connector.KvCacheConnectorManager(ReservationWorker(), ReservationScheduler())
    result.configure_prefix_reservations(True)
    return result


@pytest.fixture
def req():
    return SimpleNamespace(
        request_id=7,
        state=LlmRequestState.CONTEXT_INIT,
        cache_salt="tenant",
        get_tokens=lambda beam: list(range(128)),
    )


def accept(manager, req):
    reservation = manager.reserve_prefix(req, 32)
    manager.accept_prefix_load(req, 32, 96, [[10, 11, 12], [20, -1, 22]])
    batch = ScheduledRequests()
    batch.context_requests_last_chunk = [req]
    manager.build_scheduler_output(batch, None)
    manager.take_scheduled_requests_pending_load(batch)
    assert batch.context_requests == []
    return reservation


def dispatch(manager):
    manager.handle_metadata()
    manager.mark_prefix_loads_dispatched()
    manager.worker.start_load_kv(None)


def finish(manager):
    manager.get_finished()
    manager.finish_prefix_loads(manager.take_finished_prefix_loads())


def test_reservation_protects_source_without_transmission(manager, req):
    reservation = manager.reserve_prefix(req, 32)
    assert manager.reserve_prefix(req, 32) is reservation
    assert len(manager.scheduler.queries) == 1
    assert manager.worker.started == []
    assert not manager.has_pending_load(req)
    with pytest.raises(RuntimeError, match="reserved"):
        manager.scheduler.replace_source()

    manager.release_prefix_reservation(req)
    manager.release_prefix_reservation(req)
    assert manager.scheduler.releases == [(reservation.reservation_id, 32, 96)]
    assert manager.pending_prefix_requests() == []
    manager.scheduler.replace_source()


def test_trim_releases_each_unused_range_once(manager, req):
    original = manager.reserve_prefix(req, 16)
    trimmed = manager.trim_prefix_reservation(req, 32, 64)
    assert (trimmed.start, trimmed.end) == (32, 64)
    manager.release_prefix_reservation(req)
    assert manager.scheduler.releases == [
        (original.reservation_id, 16, 32),
        (original.reservation_id, 64, 96),
        (original.reservation_id, 32, 64),
    ]
    assert manager.scheduler.protected == {}


def test_overlapping_reservations_keep_independent_source_protection(manager, req):
    other = SimpleNamespace(**vars(req))
    other.request_id = 8
    manager.reserve_prefix(req, 32)
    manager.reserve_prefix(other, 32)
    manager.release_prefix_reservation(req)
    with pytest.raises(RuntimeError, match="reserved"):
        manager.scheduler.replace_source()
    manager.release_prefix_reservation(other)
    manager.scheduler.replace_source()


def test_confirmed_async_load_survives_a_second_output_build(manager, req):
    reservation = accept(manager, req)
    manager.build_scheduler_output(ScheduledRequests(), None)
    assert manager.worker.started == []
    assert manager.new_async_requests.loading == {}
    assert manager.has_pending_loads()
    dispatch(manager)
    load = manager.worker.started[0]
    assert load.reservation_id == reservation.reservation_id
    assert (load.start, load.end, load.cache_salt) == (32, 96, "tenant")
    assert load.block_ids_by_layer_group == [[10, 11, 12], [20, -1, 22]]
    assert load.tokens == list(range(128))
    assert manager.worker.get_connector_meta().new_requests == []


@pytest.mark.parametrize("already_bound", [False, True])
def test_unstarted_load_can_be_released_without_dispatch(manager, req, already_bound):
    reservation = accept(manager, req)
    if already_bound:
        manager.handle_metadata()
    manager.release_unstarted_prefix_loads(req)
    manager.handle_metadata()
    manager.mark_prefix_loads_dispatched()
    manager.worker.start_load_kv(None)
    assert manager.worker.started == []
    assert manager.scheduler.releases == [(reservation.reservation_id, 32, 96)]
    assert not manager.has_pending_loads()


def test_cancelled_load_waits_for_every_rank(manager, req):
    tracker = manager._prefix_completion_tracker
    tracker._size = 2
    reservation = accept(manager, req)
    dispatch(manager)
    assert manager.defer_load_termination(req)
    manager.worker.finished = [reservation.reservation_id]
    finish(manager)
    manager.release_unstarted_prefix_loads(req)
    assert manager.has_pending_load(req)
    assert manager.scheduler.releases == []
    assert manager.take_finished_load_terminations() == []
    with pytest.raises(RuntimeError, match="owns the allocation"):
        manager.reset_request_state(req)

    tracker._record(1, {reservation.reservation_id})
    completed = manager.take_finished_prefix_loads()
    assert completed == [reservation.reservation_id]
    assert manager.has_pending_load(req)
    assert manager.scheduler.releases == []
    manager.finish_prefix_loads(completed)
    assert manager.has_pending_loads()
    assert req.state == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    assert manager.take_finished_load_terminations() == [req]
    assert not manager.has_pending_loads()
    assert manager.take_finished_load_terminations() == []
    assert manager.scheduler.releases == [(reservation.reservation_id, 32, 96)]
    assert req.request_id not in manager.finished_async_loading_requests


def test_stale_completion_cannot_finish_a_replayed_allocation(manager, req):
    first = accept(manager, req)
    dispatch(manager)
    manager.worker.finished = [first.reservation_id]
    finish(manager)
    assert req.state == LlmRequestState.CONTEXT_INIT
    assert not manager.should_add_sequence(req)
    manager.reset_request_state(req)
    assert manager.should_add_sequence(req)

    second = accept(manager, req)
    assert second.reservation_id > first.reservation_id
    dispatch(manager)
    manager.worker.finished = [first.reservation_id, first.reservation_id, 10000]
    finish(manager)
    assert manager.has_pending_load(req)
    assert req.state == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    manager.worker.finished = [second.reservation_id, second.reservation_id]
    finish(manager)
    assert not manager.has_pending_load(req)
    assert req.state == LlmRequestState.CONTEXT_INIT
    assert manager.scheduler.protected == {}


def test_sync_load_retains_ownership_until_identity_completion(manager, req):
    manager.scheduler.is_async = False
    reservation = manager.reserve_prefix(req, 32)
    manager.accept_prefix_load(req, 32, 96, [[1, 2, 3]])
    assert manager.scheduler_output_manager.external_loads == {req.request_id: 64}
    manager.build_scheduler_output(ScheduledRequests(), None)
    dispatch(manager)
    assert manager.has_pending_load(req)
    assert manager.defer_load_termination(req)
    manager.worker.finished = [reservation.reservation_id]
    finish(manager)
    assert manager.take_finished_load_terminations() == [req]
    assert not manager.has_pending_load(req)


def test_legacy_load_cancellation_retains_locally_finished_ownership(manager, req, monkeypatch):
    manager.configure_prefix_reservations(False)
    manager.commit_new_matched_tokens(req, 64, True)
    req.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    assert manager.defer_load_termination(req)
    manager.worker.legacy_finished = ([], [req.request_id])
    remote_finished = ([], [])
    monkeypatch.setattr(connector, "mpi_allgather", lambda value: [value, remote_finished])
    manager.get_finished()
    assert req.request_id in manager.local_finished_async_requests.loading
    assert manager.has_pending_load(req)
    assert manager.take_finished_load_terminations() == []
    remote_finished = ([], [req.request_id])
    manager.get_finished()
    assert manager.take_finished_load_terminations() == [req]
    assert req.state == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    assert not manager.has_pending_loads()


@pytest.mark.parametrize("missing", ["reserve_prefix", "release_prefix_reservation", "worker"])
def test_partial_reservation_capability_is_rejected(manager, monkeypatch, missing):
    manager._prefix_capability = None
    if missing == "worker":
        monkeypatch.setattr(
            ReservationWorker,
            "get_finished_prefix_loads",
            connector.KvCacheConnectorWorker.get_finished_prefix_loads,
        )
    else:
        monkeypatch.setattr(
            ReservationScheduler, missing, getattr(connector.KvCacheConnectorScheduler, missing)
        )
    with pytest.raises(ValueError, match="prefix reservations require"):
        manager.configure_prefix_reservations(True)


def test_legacy_connectors_do_not_enable_reservations(manager, monkeypatch):
    manager._prefix_capability = None
    for name in ("reserve_prefix", "release_prefix_reservation"):
        monkeypatch.setattr(
            ReservationScheduler, name, getattr(connector.KvCacheConnectorScheduler, name)
        )
    monkeypatch.setattr(
        ReservationWorker,
        "get_finished_prefix_loads",
        connector.KvCacheConnectorWorker.get_finished_prefix_loads,
    )
    manager.configure_prefix_reservations(True)
    assert not manager.prefix_reservations_enabled


def test_reserving_and_releasing_do_not_gather_worker_state(manager, req, monkeypatch):
    def unexpected_collective(*args, **kwargs):
        pytest.fail("Reservation validation or release entered a collective")

    monkeypatch.setattr(connector, "mpi_allgather", unexpected_collective)
    reservation = manager.reserve_prefix(req, 32)
    monkeypatch.setattr(connector, "mpi_broadcast", unexpected_collective)
    manager.trim_prefix_reservation(req, 32, 64)
    manager.release_prefix_reservation(req)
    assert manager.scheduler.releases == [
        (reservation.reservation_id, 64, 96),
        (reservation.reservation_id, 32, 64),
    ]


@pytest.mark.parametrize("with_load", [False, True])
def test_prefix_polling_adds_no_collective(manager, req, monkeypatch, with_load):
    if with_load:
        reservation = accept(manager, req)
        dispatch(manager)
        manager.worker.finished = [reservation.reservation_id]
    calls = []

    def gather(value):
        calls.append(value)
        return [value]

    monkeypatch.setattr(connector, "mpi_allgather", gather)
    manager.get_finished()
    assert calls == [([], [])]
    if with_load:
        assert req.state == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        manager.finish_prefix_loads(manager.take_finished_prefix_loads())
        assert req.state == LlmRequestState.CONTEXT_INIT


def test_retirement_requires_local_transfer_completion(manager, req):
    reservation = accept(manager, req)
    dispatch(manager)
    with pytest.raises(RuntimeError, match="has not finished locally"):
        manager.finish_prefix_loads([reservation.reservation_id])
    assert manager.has_pending_load(req)
    assert manager.scheduler.releases == []


@pytest.mark.parametrize("answer", [(-1, False), (True, False), (0, True), (32, 1)])
def test_invalid_reservation_answer_is_rejected(manager, req, monkeypatch, answer):
    monkeypatch.setattr(manager.scheduler, "reserve_prefix", lambda *args: answer)
    with pytest.raises(ValueError):
        manager.reserve_prefix(req, 32)
    assert manager.get_prefix_reservation(req) is None
