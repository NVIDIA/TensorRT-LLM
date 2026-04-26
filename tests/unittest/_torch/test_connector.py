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
from unittest.mock import MagicMock, patch

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


def test_scheduler_output_block_hashes_front_run_completed_block():
    kv_cache_manager = MagicMock()
    kv_cache_manager.get_cache_indices.return_value = [0]
    kv_cache_manager.tokens_per_block = 4

    req = MagicMock()
    req.request_id = 42
    req.state = LlmRequestState.GENERATION_IN_PROGRESS
    req.py_draft_tokens = []

    scheduled_batch = ScheduledRequests()
    scheduled_batch.generation_requests = [req]

    manager = KvCacheConnectorSchedulerOutputManager()

    with patch(
            "tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector.get_stored_block_hashes"
    ) as get_stored_block_hashes_mock:
        get_stored_block_hashes_mock.side_effect = [[], [12345]]

        req.get_tokens.return_value = [1, 2, 3]
        scheduler_output = manager.build_scheduler_output(
            scheduled_batch, AsyncRequests({}, {}), kv_cache_manager)

        request_data = scheduler_output.cached_requests[0]
        assert request_data.new_tokens == [1, 2, 3]
        assert request_data.block_hashes == []

        req.get_tokens.return_value = [1, 2, 3, 4]
        scheduler_output = manager.build_scheduler_output(
            scheduled_batch, AsyncRequests({}, {}), kv_cache_manager)

        request_data = scheduler_output.cached_requests[0]
        assert request_data.new_tokens == [4]
        # The connector-facing hash chain is updated as soon as the block is
        # complete, even if KV cache event emission may lag slightly.
        assert request_data.block_hashes == [12345]

        # Only the newly completed blocks are hashed each call: both calls
        # request hashes starting at block index 0 with no parent.
        assert get_stored_block_hashes_mock.call_count == 2
        for call in get_stored_block_hashes_mock.call_args_list:
            assert call.args == (req, 4, 0, 0)


def test_scheduler_output_block_hashes_incremental_matches_from_scratch():
    """Caching the block-hash chain across iterations must produce the same
    cumulative chain as a fresh manager would build from the final tokens,
    and each step must only re-hash newly completed blocks."""

    PATCH_TARGET = (
        "tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector"
        ".get_stored_block_hashes")

    def fake_get_stored_block_hashes(req, tokens_per_block, start_block_idx,
                                     parent_hash):
        # Mimic the real C++ semantics: hash only full blocks at indices
        # [start_block_idx, num_full_blocks), chained from parent_hash.
        num_full_blocks = len(req.get_tokens(0)) // tokens_per_block
        result = []
        h = parent_hash
        for b in range(start_block_idx, num_full_blocks):
            h = (b * 1_000_003 + h * 31 + 7) & 0xFFFFFFFFFFFFFFFF
            result.append(h)
        return result

    kv_cache_manager = MagicMock()
    kv_cache_manager.get_cache_indices.return_value = [0]
    kv_cache_manager.tokens_per_block = 4

    req = MagicMock()
    req.request_id = 99
    req.state = LlmRequestState.GENERATION_IN_PROGRESS
    req.py_draft_tokens = []

    scheduled_batch = ScheduledRequests()
    scheduled_batch.generation_requests = [req]

    # Token growths chosen to cover: no new block, one new block, several
    # tokens without crossing a boundary, a new block after a partial, and
    # multiple new blocks completing in a single step.
    token_growths = [
        [1, 2, 3],
        [1, 2, 3, 4],
        [1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        list(range(1, 17)),
    ]

    incremental_manager = KvCacheConnectorSchedulerOutputManager()
    per_step_chains = []
    with patch(PATCH_TARGET,
               side_effect=fake_get_stored_block_hashes) as mock_inc:
        for tokens in token_growths:
            req.get_tokens.return_value = tokens
            output = incremental_manager.build_scheduler_output(
                scheduled_batch, AsyncRequests({}, {}), kv_cache_manager)
            per_step_chains.append(list(output.cached_requests[0].block_hashes))

    # A fresh manager seeing the same tokens for the first time must build the
    # same cumulative chain as the incremental one cached up to that step.
    for tokens, observed in zip(token_growths, per_step_chains, strict=True):
        fresh_manager = KvCacheConnectorSchedulerOutputManager()
        fresh_req = MagicMock()
        fresh_req.request_id = 100
        fresh_req.state = LlmRequestState.GENERATION_IN_PROGRESS
        fresh_req.py_draft_tokens = []
        fresh_req.get_tokens.return_value = tokens

        fresh_batch = ScheduledRequests()
        fresh_batch.generation_requests = [fresh_req]

        with patch(PATCH_TARGET, side_effect=fake_get_stored_block_hashes):
            fresh_output = fresh_manager.build_scheduler_output(
                fresh_batch, AsyncRequests({}, {}), kv_cache_manager)
        from_scratch = list(fresh_output.cached_requests[0].block_hashes)
        assert observed == from_scratch, (
            f"Incremental chain {observed} != from-scratch "
            f"{from_scratch} for tokens={tokens}")

    # The cache must actually save work: each call's start_block_idx equals
    # the prefix length already cached and parent_hash equals the last cached
    # hash.
    expected_starts_and_parents = []
    running_chain: list = []
    for tokens in token_growths:
        req.get_tokens.return_value = tokens
        start = len(running_chain)
        parent = running_chain[-1] if running_chain else 0
        expected_starts_and_parents.append((start, parent))
        running_chain.extend(fake_get_stored_block_hashes(
            req, 4, start, parent))

    actual_starts_and_parents = [(c.args[2], c.args[3])
                                 for c in mock_inc.call_args_list]
    assert actual_starts_and_parents == expected_starts_and_parents


@patch(
    "tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector.get_stored_block_hashes",
    return_value=[])
def test_scheduler_output_num_scheduled_tokens_with_mtp(_):
    """Test that num_scheduled_tokens is correctly set for MTP (multi-token prediction)."""
    NUM_DRAFT_TOKENS = 3

    kv_cache_manager = MagicMock()
    kv_cache_manager.get_cache_indices.return_value = [0, 1, 2]
    kv_cache_manager.tokens_per_block = 4

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
