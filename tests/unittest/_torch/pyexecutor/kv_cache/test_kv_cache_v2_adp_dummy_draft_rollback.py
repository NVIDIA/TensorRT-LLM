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
"""A skipped iteration must not strand the attention-DP dummy's draft KV mirror.

``KVCacheV2Scheduler`` admits an unpaired draft pool's mirror while it schedules
a generation request (``_admit_unpaired_draft`` -> ``admit_mirror``). For the
attention-DP pad dummy that is a fresh, empty draft cache under the reused
``ATTENTION_DP_DUMMY_REQUEST_ID``. When a peer rank schedules nothing, the
iteration is not queued and ``PyExecutor._finalize_adp_dummy_allocation`` rolls
the dummy back before ``prepare_resources``. If the draft mirror survives that
rollback, the deferred previous batch -- which still holds the earlier,
finished dummy with the same ID -- is processed on the next queued iteration,
and the draft pool's ``update_resources`` resizes the stranded empty cache to
that dummy's history: ``ValueError: History length cannot exceed capacity``.
That is the crash of a disaggregated DSpark generation server (separate draft
KV cache, overlap scheduler, attention DP) one iteration after a skipped one.
"""

import gc
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import tensorrt_llm.bindings
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import (
    ATTENTION_DP_DUMMY_REQUEST_ID,
    LlmRequest,
    LlmRequestState,
)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager, ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler
from tensorrt_llm.llmapi.llm_args import (
    CapacitySchedulerPolicy,
    DFlashDecodingConfig,
    KvCacheConfig,
)
from tensorrt_llm.mapping import Mapping

DataType = tensorrt_llm.bindings.DataType
CacheType = tensorrt_llm.bindings.internal.batch_manager.CacheType

DRAFT_LEN = 5


def _executor(**kwargs: object) -> SimpleNamespace:
    """The attributes ``_finalize_adp_dummy_allocation`` and ``_revert_gen_alloc`` read."""
    defaults = dict(
        _pending_adp_dummy_request=None,
        _is_kv_manager_v2=True,
        enable_joint_kv_cache_reuse=False,
        active_requests=[],
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


@pytest.mark.cpu_only
def test_rollback_releases_the_dummy_draft_mirror() -> None:
    dummy = SimpleNamespace(
        py_request_id=ATTENTION_DP_DUMMY_REQUEST_ID,
        py_seq_slot=None,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
    )
    resource_manager = Mock()
    resource_manager.get_resource_manager.return_value = None
    executor = _executor(
        _pending_adp_dummy_request=dummy,
        kv_cache_manager=Mock(),
        draft_kv_cache_manager=Mock(),
        resource_manager=resource_manager,
        active_requests=[dummy],
    )

    PyExecutor._finalize_adp_dummy_allocation(executor, False)

    assert dummy.state == LlmRequestState.GENERATION_COMPLETE
    assert executor.active_requests == []
    executor.kv_cache_manager.free_resources.assert_called_once_with(dummy)
    executor.draft_kv_cache_manager.free_resources.assert_called_once_with(dummy)


@pytest.mark.cpu_only
def test_queued_dummy_keeps_its_draft_mirror() -> None:
    dummy = SimpleNamespace(py_request_id=ATTENTION_DP_DUMMY_REQUEST_ID, py_seq_slot=None)
    executor = _executor(
        _pending_adp_dummy_request=dummy,
        kv_cache_manager=Mock(),
        draft_kv_cache_manager=Mock(),
        resource_manager=Mock(),
        active_requests=[dummy],
    )

    PyExecutor._finalize_adp_dummy_allocation(executor, True)

    assert executor.active_requests == [dummy]
    executor.kv_cache_manager.free_resources.assert_not_called()
    executor.draft_kv_cache_manager.free_resources.assert_not_called()


def _manager(*, is_draft: bool) -> KVCacheManagerV2:
    return KVCacheManagerV2(
        KvCacheConfig(enable_block_reuse=False, max_gpu_total_bytes=4 << 20),
        CacheType.SELF,
        num_layers=1,
        num_kv_heads=2,
        head_dim=64,
        tokens_per_block=32,
        max_seq_len=4096,
        max_batch_size=4,
        max_num_tokens=128,
        mapping=Mapping(),
        dtype=DataType.HALF,
        spec_config=DFlashDecodingConfig(max_draft_len=DRAFT_LEN, speculative_model="draft"),
        is_draft=is_draft,
        is_disagg=True,
    )


@pytest.fixture
def pools() -> Iterator[tuple[KVCacheManagerV2, KVCacheManagerV2]]:
    torch.cuda.init()
    target = _manager(is_draft=False)
    draft = _manager(is_draft=True)
    yield target, draft
    draft.shutdown()
    target.shutdown()
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="allocates real KV cache pools")
def test_skipped_iteration_does_not_strand_the_dummy_draft_mirror(
    pools: tuple[KVCacheManagerV2, KVCacheManagerV2],
) -> None:
    """Replay the overlap loop on one attention-DP rank with real KV pools.

    J: the rank is idle, pads a dummy, every rank queues, the dummy runs and is
       terminated; batch J becomes the previous batch.
    K: the rank pads a new dummy, but a peer scheduled nothing, so can_queue is
       False: the dummy is rolled back, and batch J stays deferred because this
       rank's own batch was not empty.
    K+1: the rank has real work and pads no dummy, so the deferred batch J is
       processed (_process_previous_batch skips the V2 target manager there).
    """
    target, draft = pools
    resources = ResourceManager(
        {
            ResourceManagerType.KV_CACHE_MANAGER: target,
            ResourceManagerType.DRAFT_KV_CACHE_MANAGER: draft,
        }
    )
    scheduler = KVCacheV2Scheduler(
        max_batch_size=4,
        max_num_tokens=128,
        kv_cache_manager=target,
        scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
        draft_kv_cache_manager=draft,
    )
    # Without block reuse the draft pool is unpaired, as on the DSpark
    # generation server; the scheduler admits its mirror on its own.
    assert scheduler._joint_draft_manager is None
    executor = _executor(
        kv_cache_manager=target,
        draft_kv_cache_manager=draft,
        resource_manager=resources,
    )

    def pad_and_schedule() -> tuple[LlmRequest, ScheduledRequests]:
        # _pad_attention_dp_dummy_request, then _schedule().
        dummy = target.add_dummy_requests(
            [ATTENTION_DP_DUMMY_REQUEST_ID], is_gen=True, max_num_draft_tokens=DRAFT_LEN
        )[0]
        dummy.is_attention_dp_dummy = True
        executor.active_requests.append(dummy)
        executor._pending_adp_dummy_request = dummy
        output = scheduler.schedule_request(executor.active_requests, set())
        assert output.generation_requests == [dummy]
        batch = ScheduledRequests()
        batch.generation_requests = list(output.generation_requests)
        return dummy, batch

    # Iteration J.
    dummy_j, batch_j = pad_and_schedule()
    resources.prepare_resources(batch_j)
    PyExecutor._finalize_adp_dummy_allocation(executor, True)
    # _update_request_states -> _terminate_attention_dp_dummy_requests.
    dummy_j.state = LlmRequestState.GENERATION_COMPLETE
    resources.free_resources(dummy_j)
    executor.active_requests.remove(dummy_j)
    assert ATTENTION_DP_DUMMY_REQUEST_ID not in draft.kv_cache_map

    # Iteration K: not queued.
    _, batch_k = pad_and_schedule()
    assert ATTENTION_DP_DUMMY_REQUEST_ID in draft.kv_cache_map
    PyExecutor._revert_gen_alloc(executor, batch_k)
    PyExecutor._finalize_adp_dummy_allocation(executor, False)
    assert ATTENTION_DP_DUMMY_REQUEST_ID not in target.kv_cache_map

    # Iteration K+1: _process_previous_batch on the deferred batch J.
    resources.update_resources(batch_j, skip_kv_cache_manager=True)

    assert ATTENTION_DP_DUMMY_REQUEST_ID not in draft.kv_cache_map
    assert executor.active_requests == []
