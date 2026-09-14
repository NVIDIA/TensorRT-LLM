# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import (
    AsyncRequests,
    KvCacheConnectorManager,
    KvCacheConnectorScheduler,
    KvCacheConnectorSchedulerOutputManager,
    KvCacheConnectorWorker,
    SchedulerOutput,
)
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import LlmRequestState

pytestmark = pytest.mark.cpu_only
CONNECTOR_MODULE = "tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector"


class ADPAdapterMock(MagicMock):
    supports_attention_dp = True


def _request(request_id: int, *, dummy: bool = False) -> MagicMock:
    req = MagicMock(is_dummy_request=dummy, is_generation_only_request=False)
    req.request_id = request_id
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
        cache = MagicMock()
        cache.get_cache_indices.return_value = [17 + rank]
        cache.commit_and_get_block_hashes.return_value = [123]
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
    cache = MagicMock()
    output_manager = KvCacheConnectorSchedulerOutputManager()
    output = output_manager.build_scheduler_output(batch, AsyncRequests({}, {}), cache)
    assert not output.new_requests and not output.cached_requests
    assert not output_manager.requests
    cache.get_cache_indices.assert_not_called()
    cache.commit_and_get_block_hashes.assert_not_called()


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
    executor.kv_connector_manager = manager
    executor.enable_attention_dp = True
    cache = MagicMock()
    cache.publish_connector_scheduler_output.side_effect = lambda batch: (
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
    cache.publish_connector_scheduler_output.assert_not_called()
    manager.scheduler.build_connector_meta.assert_called_once_with(plan)
    manager.worker.start_load_kv.assert_called_once()


def test_cancel_waits_for_connector_dma(forbid_connector_collectives: None) -> None:
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    manager = _manager()
    req = _request(8)
    executor = object.__new__(PyExecutor)
    executor.kv_connector_manager = manager
    executor.kv_cache_transceiver = None
    manager.get_num_new_matched_tokens(req, 0)
    assert not executor._try_cancel_request(req)
    manager.worker.get_finished.return_value = ([], [8])
    manager.get_finished()
    assert executor._try_cancel_request(req)
