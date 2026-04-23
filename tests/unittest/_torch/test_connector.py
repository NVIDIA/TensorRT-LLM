# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    AsyncRequests, KvCacheConnectorManager,
    KvCacheConnectorSchedulerOutputManager)
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

        req = MagicMock()

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

        req = MagicMock()

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

        req0 = MagicMock()
        req0.request_id = 0
        req0.is_generation_only_request = False

        req1 = MagicMock()
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


def test_scheduler_output_num_scheduled_tokens_with_mtp():
    """Test that num_scheduled_tokens is correctly set for MTP (multi-token prediction)."""
    NUM_DRAFT_TOKENS = 3

    kv_cache_manager = MagicMock()
    kv_cache_manager.get_cache_indices.return_value = [0, 1, 2]

    # Create a mock request in generation state with draft tokens
    req = MagicMock()
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


def test_handle_load_errors_no_failures():
    """handle_load_errors returns empty list when no blocks failed."""
    worker = MagicMock()
    worker.get_block_ids_with_load_errors.return_value = []

    manager = KvCacheConnectorManager(worker, scheduler=MagicMock())

    kv_cache_manager = MagicMock()
    affected = manager.handle_load_errors([], kv_cache_manager)

    assert affected == []
    worker.get_block_ids_with_load_errors.assert_called_once()


def test_handle_load_errors_identifies_affected_requests():
    """handle_load_errors matches failed block IDs to active requests."""
    worker = MagicMock()
    # Blocks 5 and 10 failed to load
    worker.get_block_ids_with_load_errors.return_value = [5, 10]

    manager = KvCacheConnectorManager(worker, scheduler=MagicMock())

    # Set up active requests with known block allocations
    req_a = MagicMock()
    req_a.py_request_id = 1
    req_b = MagicMock()
    req_b.py_request_id = 2
    req_c = MagicMock()
    req_c.py_request_id = 3

    kv_cache_manager = MagicMock()
    # req_a uses blocks [1, 2, 5] — overlaps with failed block 5
    # req_b uses blocks [3, 4, 6] — no overlap
    # req_c uses blocks [7, 8, 10] — overlaps with failed block 10
    kv_cache_manager.get_cache_indices.side_effect = lambda req: {
        1: [1, 2, 5],
        2: [3, 4, 6],
        3: [7, 8, 10],
    }[req.py_request_id]

    affected = manager.handle_load_errors([req_a, req_b, req_c],
                                          kv_cache_manager)

    assert len(affected) == 2
    assert req_a in affected
    assert req_c in affected
    assert req_b not in affected


def test_handle_load_errors_tolerates_cache_index_errors():
    """handle_load_errors skips requests that fail get_cache_indices."""
    worker = MagicMock()
    worker.get_block_ids_with_load_errors.return_value = [5]

    manager = KvCacheConnectorManager(worker, scheduler=MagicMock())

    req_good = MagicMock()
    req_good.py_request_id = 1
    req_bad = MagicMock()
    req_bad.py_request_id = 2

    kv_cache_manager = MagicMock()

    def mock_get_cache_indices(req):
        if req.py_request_id == 2:
            raise RuntimeError("request already freed")
        return [1, 5]

    kv_cache_manager.get_cache_indices.side_effect = mock_get_cache_indices

    affected = manager.handle_load_errors([req_good, req_bad],
                                          kv_cache_manager)

    # req_good should be affected (has block 5), req_bad skipped due to error
    assert len(affected) == 1
    assert req_good in affected


def test_get_block_ids_with_load_errors_default_returns_empty():
    """Base KvCacheConnectorWorker.get_block_ids_with_load_errors returns []."""
    from tensorrt_llm._torch.pyexecutor.kv_cache_connector import \
        KvCacheConnectorWorker

    # The ABC default returns empty list
    assert KvCacheConnectorWorker.get_block_ids_with_load_errors(
        MagicMock()) == []
