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

pytestmark = pytest.mark.cpu_only

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


@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_connector_manager_query_is_side_effect_free(mpi_pool_executor):
    """The query and the commit are separable, and the query records nothing.

    KVCacheManagerV2 allocates per context chunk, so an offer can be larger
    than the pages the scheduler reserved. It therefore commits the amount it
    honours rather than the amount it was offered, which is only possible if
    asking is inert: `external_loads` is what tells the connector where its
    load begins, and a request registered as loading is dropped from the batch.
    """

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
def test_connector_manager_commits_only_what_is_honoured(mpi_pool_executor):
    """Committing less than the offer is legal and is what gets reported.

    The unconsumed tail is recomputed locally; the connector releases its
    ownership of the whole request at `request_finished`. V1 relies on the same
    release -- with block reuse disabled it applies none of an offer it asked
    for, because `setPrepopulatedPromptLen` is gated on `mEnableBlockReuse`.
    """

    def test():
        worker = MagicMock()

        if mpi_rank() == 0:
            scheduler = MagicMock()
            scheduler.get_num_new_matched_tokens.return_value = (128, False)
        else:
            scheduler = None

        manager = KvCacheConnectorManager(worker, scheduler=scheduler)

        req = MagicMock()
        req.request_id = 7
        req.is_generation_only_request = False
        req.py_num_connector_matched_tokens = 0

        num_tokens, load_async = manager.query_num_new_matched_tokens(req, 0)
        assert num_tokens == 128
        manager.commit_new_matched_tokens(req, 32, load_async)

        assert manager.scheduler_output_manager.external_loads == {7: 32}
        assert req.py_num_connector_matched_tokens == 32

    run_across_mpi(mpi_pool_executor, test, 2)


def test_scheduler_output_resets_a_destroyed_allocation():
    """A replayed request must be reported as new, with its whole block list.

    `block_ids` and `tokens` are cumulative deltas, which is right while an
    allocation lives and wrong the moment one is destroyed. On KVCacheManagerV2
    a destructive recompute pause does exactly that and the request replays --
    reachable there and only there, because a connector on V1 must run under
    GUARANTEED_NO_EVICT, where V1 never pauses. Left stale, the replay lands in
    `cached_requests` with a delta against pages that no longer exist, and a
    connector that walks only `new_requests` issues no load.
    """
    manager = KvCacheConnectorSchedulerOutputManager()
    kv_cache_manager = MagicMock()
    kv_cache_manager.get_cache_indices.return_value = [0, 1, 2, 3]
    kv_cache_manager.commit_and_get_block_hashes.return_value = []
    kv_cache_manager.get_priority_by_block_id.return_value = 0

    req = MagicMock()
    req.request_id = 7
    req.state = LlmRequestState.CONTEXT_INIT
    req.get_tokens.return_value = list(range(64))
    req.context_current_position = 0
    req.context_remaining_length = 64
    req.context_chunk_size = 64
    req.kv_cache_retention_config = None
    req.cache_salt = None

    batch = ScheduledRequests()
    batch.context_requests_last_chunk = [req]

    first = manager.build_scheduler_output(batch, AsyncRequests(dict(), dict()),
                                           kv_cache_manager)
    assert len(first.new_requests) == 1
    assert first.new_requests[0].new_block_ids == [0, 1, 2, 3]

    # Without the drop this is a `cached_request` carrying an empty delta.
    manager.reset_request(req.request_id)

    replay = manager.build_scheduler_output(batch,
                                            AsyncRequests(dict(), dict()),
                                            kv_cache_manager)
    assert len(replay.cached_requests) == 0
    assert len(replay.new_requests) == 1
    assert replay.new_requests[0].new_block_ids == [0, 1, 2, 3]


def test_scheduler_output_keeps_deltas_while_the_allocation_lives():
    """The drop is scoped to a destroyed allocation, not to every re-report."""
    manager = KvCacheConnectorSchedulerOutputManager()
    kv_cache_manager = MagicMock()
    kv_cache_manager.get_cache_indices.return_value = [0, 1, 2, 3]
    kv_cache_manager.commit_and_get_block_hashes.return_value = []
    kv_cache_manager.get_priority_by_block_id.return_value = 0

    req = MagicMock()
    req.request_id = 8
    req.state = LlmRequestState.CONTEXT_INIT
    req.get_tokens.return_value = list(range(64))
    req.context_current_position = 0
    req.context_remaining_length = 64
    req.context_chunk_size = 64
    req.kv_cache_retention_config = None
    req.cache_salt = None

    batch = ScheduledRequests()
    batch.context_requests_last_chunk = [req]

    manager.build_scheduler_output(batch, AsyncRequests(dict(), dict()),
                                   kv_cache_manager)
    again = manager.build_scheduler_output(batch, AsyncRequests(dict(), dict()),
                                           kv_cache_manager)

    assert len(again.new_requests) == 0
    assert len(again.cached_requests) == 1
    assert again.cached_requests[0].new_block_ids == []


def test_scheduler_output_num_scheduled_tokens_with_mtp():
    """Test that num_scheduled_tokens is correctly set for MTP (multi-token prediction)."""
    NUM_DRAFT_TOKENS = 3

    kv_cache_manager = MagicMock()
    kv_cache_manager.get_cache_indices.return_value = [0, 1, 2]
    kv_cache_manager.commit_and_get_block_hashes.return_value = []

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

    req = MagicMock()
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


class _FlatOnlyScheduler(KvCacheConnectorScheduler):
    """A connector written against the flat API, as every V1 connector is."""

    def __init__(self):
        super().__init__(llm_args=None)
        self.finished = []
        self.allocs = []

    def build_connector_meta(self, scheduler_output):
        return None

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return 0, False

    def request_finished(self, request, cache_block_ids):
        self.finished.append(list(cache_block_ids))
        return False

    def update_state_after_alloc(self, request, block_ids):
        self.allocs.append(list(block_ids))


def test_an_empty_group_list_reaches_the_flat_callbacks():
    """`[]` means the request has no KV cache, not that it has zero layer groups.

    `KVCacheManagerV2.get_page_indices_by_layer_group` returns `[]` for a request
    its `kv_cache_map` holds no entry for. Dispatched to the per-layer-group
    form, that reaches the ABC default's refusal, so a connector implementing
    only the flat API dies with "0 layer groups" instead of being told there is
    nothing to save.
    """
    scheduler = _FlatOnlyScheduler()
    manager = KvCacheConnectorManager(MagicMock(), scheduler)

    req = MagicMock()
    req.request_id = 3

    manager.update_state_after_alloc(req, [], [])
    assert scheduler.allocs == [[]]

    assert manager.request_finished(req, [], []) is False
    assert scheduler.finished == [[]]


def test_a_per_layer_group_connector_needs_no_flat_stubs():
    """Overriding the grouped form is enough to instantiate.

    The flat methods are the abstract ones and abstractness is tracked per
    method name, so overriding `request_finished_by_layer_group` does not clear
    the flag on `request_finished`. Without the stand-in, a VSWA-only connector
    would carry three dead methods purely to construct -- which is a trap,
    because the failure is a `TypeError` at bring-up naming methods the
    connector has no reason to implement.
    """

    class VswaWorker(KvCacheConnectorWorker):

        def register_kv_cache_layout(self, layout):
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

    class VswaScheduler(KvCacheConnectorScheduler):

        def build_connector_meta(self, scheduler_output):
            return None

        def get_num_new_matched_tokens(self, request, num_computed_tokens):
            return 0, False

        def request_finished_by_layer_group(self, request,
                                            cache_block_ids_by_layer_group):
            return False

        def update_state_after_alloc_by_layer_group(self, request,
                                                    block_ids_by_layer_group):
            pass

    worker = VswaWorker(llm_args=None)
    scheduler = VswaScheduler(llm_args=None)

    # The flat forms are still reachable -- the V1 path calls them -- and say
    # which form this connector implements rather than reporting nothing.
    with pytest.raises(NotImplementedError, match="register_kv_cache_layout"):
        worker.register_kv_caches(None)
    with pytest.raises(NotImplementedError,
                       match="request_finished_by_layer_group"):
        scheduler.request_finished(None, [])
    with pytest.raises(NotImplementedError,
                       match="update_state_after_alloc_by_layer_group"):
        scheduler.update_state_after_alloc(None, [])


def test_a_connector_implementing_neither_form_still_fails_at_construction():
    """The stand-in only fires for a connector that implements the grouped form.

    A connector implementing neither must keep failing where it always has, at
    construction, naming the method to implement.
    """

    class NeitherScheduler(KvCacheConnectorScheduler):

        def build_connector_meta(self, scheduler_output):
            return None

        def get_num_new_matched_tokens(self, request, num_computed_tokens):
            return 0, False

    with pytest.raises(TypeError, match="request_finished"):
        NeitherScheduler(llm_args=None)
