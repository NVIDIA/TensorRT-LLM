# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pickle
import sys
from unittest.mock import MagicMock

import cloudpickle
import mpi4py
import pytest

from tensorrt_llm import mpi_rank
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import (
    AsyncRequests, KvCacheConnectorManager, KvCacheConnectorScheduler,
    KvCacheConnectorSchedulerOutputManager, KvCacheConnectorWorker)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

cloudpickle.register_pickle_by_value(sys.modules[__name__])
mpi4py.MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)


def run_across_mpi(executor, fun, num_ranks):
    return list(executor.starmap(fun, [() for i in range(num_ranks)]))


@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
# TODO(jthomson04): I don't have the slightest idea why this test is leaking threads.
@pytest.mark.threadleak(enabled=False)
def test_connector_manager_get_finished_allgather(mpi_pool_executor):

    def test():
        worker = MagicMock()

        if mpi_rank() == 0:
            scheduler = MagicMock()

            scheduler.request_finished.return_value = True
        else:
            scheduler = None

        manager = KvCacheConnectorManager(worker, scheduler=scheduler)

        req = MagicMock(is_dummy_request=False)

        req.request_id = 42

        manager.request_finished(req, [])

        # To start, make both workers return nothing.
        worker.get_finished.return_value = ([], [])

        assert manager.get_finished() == []

        assert worker.get_finished.call_count == 1
        assert worker.get_finished.call_args[0] == ([42], [])

        worker.get_finished.reset_mock()

        # Now, only return the request id on one worker.
        if mpi_rank() == 0:
            worker.get_finished.return_value = ([42], [])
        else:
            worker.get_finished.return_value = ([], [])

        # It should still return nothing, since rank 1 is still saving.
        assert manager.get_finished() == []

        assert worker.get_finished.call_count == 1
        assert worker.get_finished.call_args[0] == ([], [])

        # Now, also return it on worker 1.
        if mpi_rank() == 0:
            worker.get_finished.return_value = ([], [])
        else:
            worker.get_finished.return_value = ([42], [])

        assert manager.get_finished() == [req]

    run_across_mpi(mpi_pool_executor, test, 2)


@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_connector_manager_num_matched_tokens(mpi_pool_executor):

    def test():
        worker = MagicMock()

        if mpi_rank() == 0:
            scheduler = MagicMock()
            scheduler.get_num_new_matched_tokens.return_value = (16, True)
        else:
            scheduler = None

        manager = KvCacheConnectorManager(worker, scheduler=scheduler)

        req = MagicMock(is_dummy_request=False)

        req.request_id = 42
        req.is_generation_only_request = False

        assert manager.get_num_new_matched_tokens(req, 32) == 16

        if mpi_rank() == 0:
            assert scheduler.get_num_new_matched_tokens.call_count == 1
            assert scheduler.get_num_new_matched_tokens.call_args[0] == (req,
                                                                         32)

    run_across_mpi(mpi_pool_executor, test, 2)


@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_connector_manager_take_scheduled_requests(mpi_pool_executor):

    def test():
        worker = MagicMock()

        if mpi_rank() == 0:
            scheduler = MagicMock()
        else:
            scheduler = None

        manager = KvCacheConnectorManager(worker, scheduler=scheduler)

        scheduled_requests = ScheduledRequests()

        req0 = MagicMock(is_dummy_request=False)
        req0.request_id = 0
        req0.is_generation_only_request = False

        req1 = MagicMock(is_dummy_request=False)
        req1.request_id = 1
        req1.is_generation_only_request = False

        if mpi_rank() == 0:
            scheduler.get_num_new_matched_tokens.return_value = (16, True)

        assert manager.get_num_new_matched_tokens(req0, 0) == 16
        if mpi_rank() == 0:
            assert scheduler.get_num_new_matched_tokens.call_count == 1
            assert scheduler.get_num_new_matched_tokens.call_args[0] == (req0,
                                                                         0)

            scheduler.get_num_new_matched_tokens.reset_mock()
            scheduler.get_num_new_matched_tokens.return_value = (32, False)

        assert manager.get_num_new_matched_tokens(req1, 0) == 32
        if mpi_rank() == 0:
            assert scheduler.get_num_new_matched_tokens.call_count == 1
            assert scheduler.get_num_new_matched_tokens.call_args[0] == (req1,
                                                                         0)

        scheduled_requests.context_requests_last_chunk = [req0, req1]

        manager.take_scheduled_requests_pending_load(scheduled_requests)

        assert scheduled_requests.context_requests_last_chunk == [req1]

    run_across_mpi(mpi_pool_executor, test, 2)


@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_connector_manager_query_is_side_effect_free(mpi_pool_executor):
    """The query and the commit are separable, and the query records nothing.

    KVCacheManagerV2 asks during a speculative scheduling pass and resolves the
    answer in whichever iteration the request actually runs. That only works if
    asking is inert: `external_loads` is cleared by every
    `build_scheduler_output`, and a request registered as loading is dropped
    from the batch. Recording at query time would attribute the load to
    whichever iteration happened to ask.
    """

    def test():
        worker = MagicMock()

        if mpi_rank() == 0:
            scheduler = MagicMock()
            scheduler.get_num_new_matched_tokens.return_value = (16, True)
        else:
            scheduler = None

        manager = KvCacheConnectorManager(worker, scheduler=scheduler)

        req = MagicMock(is_dummy_request=False)
        req.request_id = 42
        req.is_generation_only_request = False
        req.py_num_connector_matched_tokens = 0

        assert manager.query_num_new_matched_tokens(req, 32) == (16, True)

        assert manager.new_async_requests.loading_ids == set()
        assert manager.scheduler_output_manager.external_loads == {}
        assert req.py_num_connector_matched_tokens == 0

        manager.commit_new_matched_tokens(req, 16, True)

        assert manager.new_async_requests.loading_ids == {42}
        assert manager.scheduler_output_manager.external_loads == {42: 16}
        assert req.py_num_connector_matched_tokens == 16

        if mpi_rank() == 0:
            assert scheduler.get_num_new_matched_tokens.call_count == 1

    run_across_mpi(mpi_pool_executor, test, 2)


@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_connector_manager_cancel_load_reaches_the_leader(mpi_pool_executor):

    def test():
        worker = MagicMock()
        scheduler = MagicMock() if mpi_rank() == 0 else None

        manager = KvCacheConnectorManager(worker, scheduler=scheduler)

        req = MagicMock(is_dummy_request=False)
        req.request_id = 42

        manager.cancel_load(req, 0, 32)

        if mpi_rank() == 0:
            assert scheduler.cancel_load.call_args[0] == (req, 0, 32)

    run_across_mpi(mpi_pool_executor, test, 2)


def test_cancel_load_is_additive():
    """Existing connectors predate `cancel_load` and must keep working.

    It is only ever raised by KVCacheManagerV2, so it has to stay optional with
    a no-op default rather than becoming another abstract method.
    """
    assert "cancel_load" not in KvCacheConnectorScheduler.__abstractmethods__


class MinimalWorker(KvCacheConnectorWorker):
    """A worker with nothing filled in beyond the abstract methods."""

    def register_kv_caches(self, kv_cache_tensor):
        pass

    def start_load_kv(self, stream):
        pass

    def wait_for_layer_load(self, layer_idx, stream):
        pass

    def save_kv_layer(self, layer_idx, stream):
        pass

    def wait_for_save(self, stream):
        pass

    def get_finished(self, finished_gen_req_ids, started_loading_req_ids):
        return [], []


def test_a_connector_is_assumed_to_move_kv():
    """Several executor restrictions exist only because a connector normally
    registers page addresses and transfers against them: the capacity
    scheduler is pinned to GUARANTEED_NO_EVICT, cache tiers below GPU are
    refused, and every decoder layer gets a pre/post hook. So the default has
    to be the restrictive one, and a connector written before `capacity_only`
    existed keeps every guard that was written for it.
    """
    assert not MinimalWorker(MagicMock()).capacity_only


def test_the_manager_reports_whether_its_worker_moves_kv():
    """The executor asks the manager, since that is the only handle it holds."""

    class CapacityWorker(MinimalWorker):
        capacity_only = True

    assert not KvCacheConnectorManager(MinimalWorker(MagicMock()),
                                       scheduler=MagicMock()).capacity_only
    assert KvCacheConnectorManager(CapacityWorker(MagicMock()),
                                   scheduler=MagicMock()).capacity_only


def test_a_capacity_only_manager_builds_no_scheduler_output():
    """A capacity-only manager skips the per-iteration scheduler output, and a
    transferring one still builds it."""

    class CapacityWorker(MinimalWorker):
        capacity_only = True

    scheduled_batch = ScheduledRequests()
    scheduled_batch.generation_requests = [
        MagicMock(is_dummy_request=False,
                  request_id=7,
                  state=LlmRequestState.GENERATION_IN_PROGRESS)
    ]

    capacity = KvCacheConnectorManager(CapacityWorker(MagicMock()),
                                       scheduler=MagicMock())
    capacity.build_scheduler_output(scheduled_batch, MagicMock())
    # With no output built, handle_metadata binds nothing to the worker.
    capacity.handle_metadata()
    capacity.scheduler.build_connector_meta.assert_not_called()
    assert capacity.worker.get_connector_meta() is None

    # The transferring role needs the output and must be unaffected.
    transferring = KvCacheConnectorManager(MinimalWorker(MagicMock()),
                                           scheduler=MagicMock())
    transferring.build_scheduler_output(scheduled_batch, MagicMock())
    transferring.handle_metadata()
    transferring.scheduler.build_connector_meta.assert_called_once()


def test_releasing_an_allocation_for_replay_tells_the_connector():
    """The connector's per-request state is keyed to the pages being freed."""
    scheduler = MagicMock()
    manager = KvCacheConnectorManager(MinimalWorker(MagicMock()),
                                      scheduler=scheduler)

    req = MagicMock()
    req.request_id = 11

    manager.reset_request_state(req)

    scheduler.request_reset.assert_called_once_with(req)


def test_scheduler_output_num_scheduled_tokens_with_mtp():
    """Test that num_scheduled_tokens is correctly set for MTP (multi-token prediction)."""
    NUM_DRAFT_TOKENS = 3

    kv_cache_manager = MagicMock()
    kv_cache_manager.get_cache_indices.return_value = [0, 1, 2]
    kv_cache_manager.commit_and_get_block_hashes.return_value = []

    # Create a mock request in generation state with draft tokens
    req = MagicMock(is_dummy_request=False)
    req.request_id = 42
    req.state = LlmRequestState.GENERATION_IN_PROGRESS
    req.get_tokens.return_value = [1, 2, 3, 4, 5]  # 5 tokens already generated
    req.py_draft_tokens = [100, 101, 102]  # 3 MTP draft tokens

    scheduled_batch = ScheduledRequests()
    scheduled_batch.generation_requests = [req]

    manager = KvCacheConnectorSchedulerOutputManager()
    scheduler_output = manager.build_scheduler_output(scheduled_batch,
                                                      AsyncRequests({}, {}),
                                                      kv_cache_manager)

    assert len(scheduler_output.cached_requests) == 1
    request_data = scheduler_output.cached_requests[0]

    # For generation requests: num_scheduled_tokens = 1 + draft_token_length
    expected_num_scheduled_tokens = 1 + NUM_DRAFT_TOKENS
    assert request_data.num_scheduled_tokens == expected_num_scheduled_tokens, \
        f"Expected {expected_num_scheduled_tokens}, got {request_data.num_scheduled_tokens}"


def test_scheduler_output_block_hashes_read_through():
    """``RequestData.block_hashes`` reflects the chain returned by the KV cache manager.

    The connector path does not recompute hashes Python-side; each scheduler step
    is a pure pass-through of whatever ``commit_and_get_block_hashes`` returns.
    A subsequent step that observes a longer chain simply forwards the longer
    chain. The block-completion semantics (when the next hash actually appears)
    are owned by the C++ KV cache manager and exercised by the C++ unit tests
    for ``commitAndGetBlockHashesForRequest``.
    """
    kv_cache_manager = MagicMock()
    kv_cache_manager.get_cache_indices.return_value = [0]
    # Two consecutive scheduler steps: first sees no full block yet, second sees
    # one full block whose hash has just been committed by the manager.
    kv_cache_manager.commit_and_get_block_hashes.side_effect = [[], [12345]]

    req = MagicMock(is_dummy_request=False)
    req.request_id = 42
    req.state = LlmRequestState.GENERATION_IN_PROGRESS
    req.py_draft_tokens = []
    req.get_tokens.return_value = [1, 2, 3]

    scheduled_batch = ScheduledRequests()
    scheduled_batch.generation_requests = [req]

    manager = KvCacheConnectorSchedulerOutputManager()

    output = manager.build_scheduler_output(scheduled_batch,
                                            AsyncRequests({}, {}),
                                            kv_cache_manager)
    assert output.cached_requests[0].block_hashes == []

    req.get_tokens.return_value = [1, 2, 3, 4]
    output = manager.build_scheduler_output(scheduled_batch,
                                            AsyncRequests({}, {}),
                                            kv_cache_manager)
    assert output.cached_requests[0].block_hashes == [12345]

    # Each scheduler step asks the manager exactly once per request; no Python
    # caching layer reshapes the request between calls.
    assert kv_cache_manager.commit_and_get_block_hashes.call_count == 2
    for call in kv_cache_manager.commit_and_get_block_hashes.call_args_list:
        assert call.args == (req, )


@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.threadleak(enabled=False)
def test_connector_adp_disjoint_requests(mpi_pool_executor) -> None:
    """A busy owner must finish without request callbacks on its idle peer."""

    def test() -> None:

        class AdapterMock(MagicMock):
            supports_attention_dp = True

        worker = AdapterMock()
        scheduler = AdapterMock()
        scheduler.get_num_new_matched_tokens.return_value = (4, False)
        scheduler.request_finished.return_value = True
        manager = KvCacheConnectorManager(worker,
                                          scheduler,
                                          enable_attention_dp=True)
        requests = []
        # Different callback counts would deadlock the old per-request MPI
        # broadcasts; intersecting all owners' completions would never release.
        if mpi_rank() == 1:
            for request_id in (10, 20, 30):
                req = MagicMock(is_dummy_request=False,
                                is_generation_only_request=False)
                req.request_id = request_id
                assert manager.get_num_new_matched_tokens(req, 0) == 4
                manager.update_state_after_alloc(req, [request_id])
                assert manager.request_finished(req, [request_id])
                requests.append(req)
        worker.get_finished.return_value = ([
            req.request_id for req in requests
        ], [])
        assert {req.request_id
                for req in manager.get_finished()
                } == {req.request_id
                      for req in requests}
        assert not manager.pending_async_requests.saving

    run_across_mpi(mpi_pool_executor, test, 2)
