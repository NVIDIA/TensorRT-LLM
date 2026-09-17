# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from threading import Barrier
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import (
    AsyncRequests,
    KvCacheConnectorManager,
    KvCacheConnectorScheduler,
    KvCacheConnectorSchedulerOutputManager,
    KvCacheConnectorWorker,
    SchedulerOutput,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import LlmRequestState

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

pytestmark = pytest.mark.cpu_only
CONNECTOR_MODULE = "tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector"


class ADPAdapterMock(MagicMock):
    supports_attention_dp = True


def _request(request_id: int, *, dummy: bool = False) -> MagicMock:
    req = MagicMock(is_dummy_request=dummy, is_generation_only_request=False)
    req.request_id = request_id
    req.py_request_id = request_id
    req.is_child = False
    req.state = LlmRequestState.CONTEXT_INIT
    req.context_current_position = 0
    req.context_remaining_length = 8
    req.context_chunk_size = 8
    req.get_tokens.return_value = list(range(8))
    req.py_draft_tokens = []
    req.kv_cache_retention_config = None
    req.cache_salt = None
    return req


def _manager(rank: int = 1) -> KvCacheConnectorManager:
    worker = ADPAdapterMock()
    worker.get_finished.return_value = ([], [])
    scheduler = ADPAdapterMock()
    scheduler.get_num_new_matched_tokens.return_value = (4, True)
    scheduler.request_finished.return_value = True
    with patch(f"{CONNECTOR_MODULE}.mpi_rank", return_value=rank):
        return KvCacheConnectorManager(worker, scheduler, enable_attention_dp=True)


def _configure_recovery_controls(executor: "PyExecutor") -> None:
    from tensorrt_llm._torch.pyexecutor.executor_request_queue import ExecutorRequestQueue

    executor.dist = MagicMock(tp_size=1, cp_size=1)
    executor.is_shutdown = False
    executor.canceled_req_ids = []
    executor.executor_request_queue = ExecutorRequestQueue(
        dist=executor.dist,
        max_batch_size=2,
        enable_iter_perf_stats=False,
        batch_wait_timeout_ms=0,
    )


@pytest.fixture
def forbid_connector_collectives() -> Iterator[None]:
    with (
        patch(f"{CONNECTOR_MODULE}.mpi_allgather", side_effect=AssertionError("ADP allgather")),
        patch(f"{CONNECTOR_MODULE}.mpi_broadcast", side_effect=AssertionError("ADP broadcast")),
    ):
        yield


@pytest.mark.parametrize("unsupported", [KvCacheConnectorWorker, KvCacheConnectorScheduler])
def test_adp_requires_explicit_backend_support(unsupported: type) -> None:
    with pytest.raises(NotImplementedError, match="supports_attention_dp=True"):
        KvCacheConnectorManager.validate_attention_dp(ADPAdapterMock, unsupported)


@pytest.mark.parametrize("rank", [0, 1])
def test_adp_requires_scheduler_on_every_owner(rank: int) -> None:
    with patch(f"{CONNECTOR_MODULE}.mpi_rank", return_value=rank):
        with pytest.raises(AssertionError, match="every attention-DP owner"):
            KvCacheConnectorManager(ADPAdapterMock(), None, enable_attention_dp=True)


@pytest.mark.parametrize("request_counts", [(2, 2), (3, 0), (0, 4)])
def test_disjoint_owner_callbacks_and_metadata(
    forbid_connector_collectives: None, request_counts: tuple[int, int]
) -> None:
    for rank, count in enumerate(request_counts):
        manager = _manager(rank)
        cache = MagicMock(spec=KVCacheManagerV2)
        cache.get_page_indices_by_layer_group.return_value = [[17 + rank]]
        batch = ScheduledRequests()
        manager.scheduler.get_num_new_matched_tokens.return_value = (4, False)
        for index in range(count):
            request = _request(rank * 100 + index)
            assert manager.get_num_new_matched_tokens(request, 0) == 4
            manager.update_state_after_alloc(request, [17 + rank])
            request.context_current_position = 4
            batch.context_requests_last_chunk.append(request)

        with patch(f"{CONNECTOR_MODULE}.mpi_rank", return_value=rank):
            manager.build_scheduler_output(batch, cache)
        manager.handle_metadata()
        output = manager.scheduler.build_connector_meta.call_args.args[0]
        assert output.attention_dp_rank == rank
        assert [req.request_id for req in output.new_requests] == [
            rank * 100 + index for index in range(count)
        ]
        assert all(req.new_block_ids == [17 + rank] for req in output.new_requests)
        assert all(req.computed_position == 0 for req in output.new_requests)
        manager.worker.bind_connector_meta.assert_called_once_with(
            manager.scheduler.build_connector_meta.return_value
        )
        assert manager.get_finished() == []


def test_owner_completion_does_not_wait_for_idle_peer(forbid_connector_collectives: None) -> None:
    manager = _manager()
    loading, saving = _request(10), _request(20)
    assert manager.get_num_new_matched_tokens(loading, 0) == 4
    manager.update_state_after_alloc(loading, [7, 8])
    batch = ScheduledRequests()
    batch.context_requests_last_chunk = [loading]
    manager.take_scheduled_requests_pending_load(batch)
    assert batch.batch_size == 0
    assert loading.state == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    assert manager.request_finished(saving, [9])
    assert manager.has_pending_transfers(10)
    assert manager.has_pending_transfers(20)

    assert manager.get_finished() == []
    manager.worker.get_finished.assert_called_once_with([20], [10])

    # The worker may report load/save completion on different iterations.
    manager.worker.get_finished.return_value = ([], [10])
    assert manager.get_finished() == []
    assert loading.state == LlmRequestState.CONTEXT_INIT
    assert not manager.should_add_sequence(loading)
    assert not manager.has_pending_transfers(10)
    assert manager.has_pending_transfers(20)
    manager.worker.get_finished.assert_called_with([], [])

    manager.worker.get_finished.return_value = ([20], [])
    assert manager.get_finished() == [saving]
    assert not manager.has_pending_transfers(20)
    assert not manager.local_finished_async_requests.saving
    assert not manager.pending_async_requests.loading


def test_dummy_requests_never_reach_storage(forbid_connector_collectives: None) -> None:
    manager = _manager()
    dummy = _request(999, dummy=True)
    assert manager.get_num_new_matched_tokens(dummy, 0) == 0
    manager.update_state_after_alloc(dummy, [5])
    assert not manager.request_finished(dummy, [5])
    manager.scheduler.get_num_new_matched_tokens.assert_not_called()
    manager.scheduler.update_state_after_alloc.assert_not_called()
    manager.scheduler.request_finished.assert_not_called()
    assert not manager.has_pending_transfers(999)

    batch = ScheduledRequests()
    batch.context_requests_last_chunk = [dummy]
    batch.generation_requests = [dummy]
    cache = MagicMock(spec=KVCacheManagerV2)
    output_manager = KvCacheConnectorSchedulerOutputManager(enable_attention_dp=True)
    output = output_manager.build_scheduler_output(batch, AsyncRequests({}, {}), cache)
    assert not output.new_requests and not output.cached_requests
    assert not output_manager.requests
    cache.get_page_indices_by_layer_group.assert_not_called()


def test_finished_request_releases_metadata_state(forbid_connector_collectives: None) -> None:
    manager = _manager()
    req = _request(42)
    manager.scheduler_output_manager.requests[42]
    manager.scheduler_output_manager.external_loads[42] = 4
    manager.finished_async_loading_requests[42] = req
    manager.scheduler.request_finished.return_value = False
    assert not manager.request_finished(req, [3])
    assert 42 not in manager.scheduler_output_manager.requests
    assert 42 not in manager.scheduler_output_manager.external_loads
    assert manager.should_add_sequence(req)


def test_tp_still_waits_for_all_shards() -> None:
    worker, scheduler = MagicMock(), MagicMock()
    scheduler.request_finished.return_value = True
    with (
        patch(f"{CONNECTOR_MODULE}.mpi_rank", return_value=0),
        patch(f"{CONNECTOR_MODULE}.mpi_broadcast", side_effect=lambda result, root: result),
        patch(f"{CONNECTOR_MODULE}.mpi_allgather") as gather,
    ):
        manager = KvCacheConnectorManager(worker, scheduler)
        req = _request(42)
        manager.request_finished(req, [5])
        worker.get_finished.return_value = ([42], [])
        gather.return_value = [([42], []), ([], [])]
        assert manager.get_finished() == []
        assert manager.has_pending_transfers(42)
        worker.get_finished.return_value = ([], [])
        gather.return_value = [([42], []), ([42], [])]
        assert manager.get_finished() == [req]


def test_async_padding_preserves_connector_plan() -> None:
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
    from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager, ResourceManagerType

    manager = _manager()
    req = _request(7)
    manager.get_num_new_matched_tokens(req, 0)
    batch = ScheduledRequests()
    batch.context_requests_last_chunk = [req]
    plan = SchedulerOutput()
    manager.set_scheduler_output(plan)
    executor = object.__new__(PyExecutor)
    _configure_recovery_controls(executor)
    executor.kv_connector_manager = manager
    executor.enable_attention_dp = True
    cache = MagicMock(spec=KVCacheManagerV2)
    cache.report_batch_to_connector.side_effect = lambda batch: (
        manager.set_scheduler_output(SchedulerOutput())
    )
    executor.resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: cache})
    dummy = _request(999, dummy=True)

    def pad_batch(scheduled_batch: ScheduledRequests) -> None:
        assert scheduled_batch.batch_size == 0
        scheduled_batch.generation_requests.append(dummy)

    executor._pad_empty_attention_dp_batch = pad_batch
    with patch("torch.cuda.current_stream"):
        executor._kv_connector_start_batch(batch)
    assert batch.generation_requests == [dummy]
    cache.prepare_resources.assert_called_once_with(batch)
    cache.report_batch_to_connector.assert_not_called()
    manager.scheduler.build_connector_meta.assert_called_once_with(plan)
    manager.worker.start_load_kv.assert_called_once()


def test_cancel_waits_for_connector_dma(forbid_connector_collectives: None) -> None:
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    manager = _manager()
    req = _request(8)
    executor = object.__new__(PyExecutor)
    executor.kv_connector_manager = manager
    executor.kv_cache_transceiver = None
    executor.enable_attention_dp = True
    manager.get_num_new_matched_tokens(req, 0)
    assert not executor._try_cancel_request(req)
    manager.worker.get_finished.return_value = ([], [8])
    manager.get_finished()
    assert executor._try_cancel_request(req)


@pytest.mark.parametrize("padding_failure", ["slot_cap", "kv_capacity"])
def test_async_without_dummy_keeps_prepared_batch(
    forbid_connector_collectives: None, padding_failure: str
) -> None:
    """Resume the first completed load without reallocation or waiting for all loads."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    manager = _manager()
    ready, pending = _request(10), _request(11)
    batch = ScheduledRequests()
    batch.context_requests_last_chunk = [ready, pending]
    cache = MagicMock(spec=KVCacheManagerV2)
    cache.get_page_indices_by_layer_group.return_value = [[5, 6]]
    for req in batch.context_requests:
        manager.get_num_new_matched_tokens(req, 0)
        req.context_current_position = 4
        req.context_remaining_length = 4
        req.context_chunk_size = 4
        manager.update_state_after_alloc(req, [5, 6])

    # Mirror the executor's already-prepared batch and initial load plan.
    manager.build_scheduler_output(batch, cache)
    manager.handle_metadata()
    executor = object.__new__(PyExecutor)
    _configure_recovery_controls(executor)
    executor.kv_connector_manager = manager
    executor.kv_cache_manager = cache
    executor.kv_cache_transceiver = None
    executor.enable_attention_dp = True
    executor.active_requests = [ready, pending]
    executor.expected_num_active_requests = 2
    executor.max_num_active_requests = 2 if padding_failure == "slot_cap" else 3
    executor._count_schedulable_active_requests = MagicMock(return_value=0)
    executor._should_skip_dummy_for_benchmark_disagg = MagicMock(return_value=False)
    executor._has_adp_dummy_kv_capacity = MagicMock(return_value=False)
    executor.resource_manager = MagicMock()
    executor.resource_manager.prepare_resources.side_effect = AssertionError("double allocation")
    manager.worker.get_finished.side_effect = [([], []), ([], [10])]

    with (
        patch("torch.cuda.current_stream"),
        patch("tensorrt_llm._torch.pyexecutor.py_executor.time.sleep") as sleep,
    ):
        executor._kv_connector_start_batch(batch)

    assert batch.context_requests == [ready]
    assert not batch.generation_requests
    assert ready.state == LlmRequestState.CONTEXT_INIT
    assert pending.state == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    assert manager.has_pending_transfers(11)
    assert not manager.should_add_sequence(ready)
    executor.resource_manager.prepare_resources.assert_not_called()
    cache.add_dummy_requests.assert_not_called()
    sleep.assert_called_once_with(0.001)
    compute_plan = manager.scheduler.build_connector_meta.call_args.args[0]
    assert [req.request_id for req in compute_plan.new_requests] == [10]
    assert compute_plan.new_requests[0].computed_position == 4
    assert compute_plan.new_requests[0].new_block_ids == [5, 6]
    assert manager.worker.start_load_kv.call_count == 2


ConnectorVote = tuple[int, int] | tuple[bool, set[int]] | str | None


class _OwnerCollective:
    """Run two owners in lockstep; fail promptly on divergent collective order."""

    def __init__(self) -> None:
        self.barrier = Barrier(2, timeout=5)
        self.values: list[ConnectorVote] = [None, None]

    def gather(self, rank: int, value: ConnectorVote) -> list[ConnectorVote]:
        self.values[rank] = value
        self.barrier.wait()
        result = self.values.copy()
        self.barrier.wait()
        return result

    def gather_int64(self, rank: int, values: list[int]) -> np.ndarray:
        return np.array(self.gather(rank, (values[0], values[1])), dtype=np.int64)


@pytest.mark.parametrize(
    "failure",
    [
        "timeout",
        "poll_error",
        "start_error",
        "shutdown",
        "cancel",
        "cancel_child",
        "complete",
        "cancel_completes",
    ],
)
def test_adp_recovery_failure_reaches_ready_peer_without_reallocation(
    forbid_connector_collectives: None, failure: str
) -> None:
    """A ready leader and a blocked owner advance or fail together, preserving allocations."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    collective = _OwnerCollective()
    executors = []
    batches = []
    for rank in range(2):
        manager = _manager(rank)
        req = _request(10 + rank)
        manager.scheduler.get_num_new_matched_tokens.return_value = (
            (4, True) if rank else (0, False)
        )
        manager.get_num_new_matched_tokens(req, 0)
        manager.update_state_after_alloc(req, [5, 6])
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [req]
        cache = MagicMock(spec=KVCacheManagerV2)
        cache.get_page_indices_by_layer_group.return_value = [[5, 6]]
        manager.build_scheduler_output(batch, cache)

        executor = object.__new__(PyExecutor)
        _configure_recovery_controls(executor)
        executor.dist.tp_size = 2
        executor.dist.tp_allgather.side_effect = partial(collective.gather, rank)
        executor.dist.tp_allgather_int64.side_effect = partial(collective.gather_int64, rank)
        executor.kv_connector_manager = manager
        executor.kv_cache_manager = cache
        executor.enable_attention_dp = True
        executor._pad_empty_attention_dp_batch = MagicMock()
        executor.resource_manager = MagicMock()
        executor.resource_manager.prepare_resources.side_effect = AssertionError(
            "double allocation"
        )
        executor._release_transfer = MagicMock()
        executors.append(executor)
        batches.append(batch)

    blocked_worker = executors[1].kv_connector_manager.worker

    polls = 0

    def poll(finished_ids: list[int], loading_ids: list[int]) -> tuple[list[int], list[int]]:
        nonlocal polls
        polls += 1
        if failure == "poll_error":
            raise OSError("backend unavailable")
        if failure == "shutdown":
            executors[0].executor_request_queue.enqueue_shutdown_request()
        elif failure in ("cancel", "cancel_child"):
            executors[0].executor_request_queue.enqueue_cancel_request(11)
        elif failure in ("complete", "cancel_completes"):
            # Leave cancellation sentinels queued for normal processing when
            # the transfer drains, whether they target this request or another.
            executors[0].executor_request_queue.enqueue_cancel_request(
                11 if failure == "cancel_completes" else 99
            )
            if polls == 2:
                return [], [11]
        return [], []

    blocked_worker.get_finished.side_effect = poll
    if failure == "cancel_child":
        blocked_request = batches[1].context_requests[0]
        blocked_request.is_child = True
        blocked_request.parent_request_id = 11
        blocked_request.py_request_id = 1011
    if failure == "start_error":
        blocked_worker.start_load_kv.side_effect = OSError("backend unavailable")

    def run(rank: int) -> str:
        if failure in ("complete", "cancel_completes"):
            executors[rank]._kv_connector_start_batch(batches[rank])
            return "complete"
        with pytest.raises(RuntimeError, match="KV connector failed") as exc:
            executors[rank]._kv_connector_start_batch(batches[rank])
        return str(exc.value)

    ticks = itertools.count(step=0.01)
    clock = SimpleNamespace(monotonic=lambda: next(ticks), sleep=MagicMock())
    with (
        patch("torch.cuda.current_stream"),
        patch("tensorrt_llm._torch.pyexecutor.py_executor.time", clock),
        patch("tensorrt_llm._torch.pyexecutor.py_executor._KV_CONNECTOR_LOAD_TIMEOUT_SEC", 1.0),
        patch("tensorrt_llm._torch.pyexecutor.py_executor._KV_CONNECTOR_CONTROL_GRACE_SEC", 0.03),
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        futures = [pool.submit(run, rank) for rank in range(2)]
        errors = [future.result(timeout=10) for future in futures]

    if failure in ("complete", "cancel_completes"):
        assert errors == ["complete", "complete"]
        assert not executors[1].kv_connector_manager.has_pending_transfers(11)
        assert [req.py_request_id for req in batches[1].context_requests] == [11]
        assert executors[0].executor_request_queue.pending_cancellation_ids() == {
            11 if failure == "cancel_completes" else 99
        }
        for executor in executors:
            executor.resource_manager.prepare_resources.assert_not_called()
            executor.kv_connector_manager.scheduler.update_state_after_alloc.assert_called_once()
        return

    expected = {
        "timeout": "Timed out",
        "poll_error": "backend unavailable",
        "start_error": "backend unavailable",
        "shutdown": "Shutdown requested",
        "cancel": "Cancellation requested",
        "cancel_child": "Cancellation requested",
    }[failure]
    assert errors[0] == errors[1]
    assert expected in errors[0]
    if failure in ("shutdown", "cancel", "cancel_child"):
        assert polls < 10, "Control requests must shorten the load deadline"
    elif failure == "timeout":
        assert 10 < polls < 150, "A permanently incomplete load must reach its deadline"
    assert executors[1].kv_connector_manager.has_pending_transfers(11)
    for executor in executors:
        assert executor._kv_connector_failed
        executor.resource_manager.prepare_resources.assert_not_called()
        executor.resource_manager.free_resources.assert_not_called()
        executor._release_transfer.assert_not_called()
        executor.kv_connector_manager.scheduler.update_state_after_alloc.assert_called_once()
        executor.kv_connector_manager.worker.start_load_kv.assert_called_once()


def test_connector_failure_shutdown_retains_registered_pools() -> None:
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = object.__new__(PyExecutor)
    _configure_recovery_controls(executor)
    executor._kv_connector_failed = True
    executor.shutdown_event = MagicMock()
    executor.resource_manager = MagicMock()
    executor.model_engine = MagicMock()
    executor.shutdown()
    executor.shutdown_event.wait.assert_called_once()
    executor.resource_manager.shutdown.assert_not_called()
    executor.model_engine._release_cuda_graphs.assert_not_called()


def test_adp_v2_query_commit_and_grouped_callbacks(forbid_connector_collectives: None) -> None:
    manager = _manager(rank=3)
    request = _request(17)
    manager.scheduler.get_num_new_matched_tokens.return_value = (8, False)
    assert manager.query_num_new_matched_tokens(request, 0) == (8, False)
    assert not manager.scheduler_output_manager.external_loads
    manager.commit_new_matched_tokens(request, 4, False)
    assert manager.scheduler_output_manager.external_loads[17] == 4
    groups = [[2, 4], [9, 3]]
    manager.update_state_after_alloc(request, [], groups)
    manager.scheduler.update_state_after_alloc_by_layer_group.assert_called_once_with(
        request, groups
    )
    manager.scheduler.update_state_after_alloc.assert_not_called()
    manager.scheduler.request_finished_by_layer_group.return_value = True
    assert manager.request_finished(request, [], groups)
    manager.scheduler.request_finished_by_layer_group.assert_called_once_with(request, groups)
    manager.scheduler.request_finished.assert_not_called()
    manager.worker.get_finished.return_value = ([17], [])
    assert manager.get_finished() == [request]


def test_dummy_v2_callbacks_do_not_publish_transfer_state(
    forbid_connector_collectives: None,
) -> None:
    manager = _manager()
    dummy = _request(999, dummy=True)
    assert manager.query_num_new_matched_tokens(dummy, 0) == (0, False)
    manager.commit_new_matched_tokens(dummy, 4, True)
    manager.update_state_after_alloc(dummy, [], [[3], [5]])
    assert not manager.request_finished(dummy, [], [[3], [5]])
    manager.scheduler.get_num_new_matched_tokens.assert_not_called()
    manager.scheduler.update_state_after_alloc_by_layer_group.assert_not_called()
    manager.scheduler.request_finished_by_layer_group.assert_not_called()
    assert not manager.scheduler_output_manager.external_loads
    assert not manager.has_pending_transfers(dummy.request_id)


def test_adp_rejects_a_resolved_v1_manager() -> None:
    """Automatic manager selection must not silently enable ADP on V1."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = object.__new__(PyExecutor)
    executor.kv_connector_manager = _manager()
    executor.kv_cache_transceiver = None
    executor.dist = SimpleNamespace(pp_size=1, cp_size=1)
    executor.max_beam_width = 1
    executor.enable_attention_dp = True
    executor.kv_cache_manager = object()

    with pytest.raises(NotImplementedError, match="requires KVCacheManagerV2"):
        executor._maybe_init_kv_connector_manager()


def test_non_adp_cancellation_keeps_existing_behavior() -> None:
    """The new DMA cancellation gate applies only to the V2 ADP path."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = object.__new__(PyExecutor)
    executor.enable_attention_dp = False
    executor.kv_connector_manager = MagicMock()
    executor.kv_cache_transceiver = None

    assert executor._try_cancel_request(_request(7))
    executor.kv_connector_manager.has_pending_transfers.assert_not_called()
