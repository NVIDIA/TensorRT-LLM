# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A non-first pipeline rank runs the context chunks the first rank scheduled
(the leader-batch adoption, on with the PP stage agreement; without it the
schedule carries no chunks and the follower keeps its own pass).

The first PP rank sends the schedule as request ids; every other rank re-runs
its own scheduler pass, which sizes context chunks (and, with KV cache manager
V2, allocates their pages) against that rank's KV pool. When the pools differ
the pass can refuse a first chunk the first rank admitted, and a refused first
chunk is rewound to the whole prompt, so the rank forwarded a prompt longer than
``max_num_tokens``.
"""

import pickle
import types
from unittest.mock import MagicMock, Mock

import pytest

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler import (
    RequestScheduler,
    ScheduledRequests,
    SerializableSchedulerOutput,
)
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler

pytestmark = pytest.mark.cpu_only

PROMPT_LEN = 3100
CHUNK = 2048


def _request(request_id: int, prompt_len: int = PROMPT_LEN) -> LlmRequest:
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=4,
        input_tokens=list(range(prompt_len)),
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )


def _leader_schedule(context: list, generation: list = ()) -> SerializableSchedulerOutput:
    scheduled = ScheduledRequests()
    scheduled.reset_context_requests(list(context))
    scheduled.generation_requests = list(generation)
    return pickle.loads(
        pickle.dumps(
            SerializableSchedulerOutput.from_scheduler_result(
                scheduled, [], 0, record_context_chunks=True
            )
        )
    )


def _follower(is_kv_manager_v2: bool) -> PyExecutor:
    executor = object.__new__(PyExecutor)
    executor.dist = Mock(pp_rank=1, rank=1)
    executor._is_kv_manager_v2 = is_kv_manager_v2
    executor.enable_joint_kv_cache_reuse = False
    executor.scheduler = Mock()
    # Base-class behaviour: set the chunk, nothing to allocate.
    executor.scheduler.adopt_context_chunk.side_effect = (
        lambda req, size: RequestScheduler.adopt_context_chunk(None, req, size)
    )
    executor.scheduler.adopt_generation_step.return_value = True
    executor.kv_cache_manager = Mock()
    executor._revert_ctx_alloc = Mock()
    return executor


def _local_pass(context=(), generation=()):
    return types.SimpleNamespace(
        context_requests=list(context), generation_requests=list(generation)
    )


def test_schedule_round_trip_carries_the_context_chunks():
    leader_request = _request(2)
    leader_request.context_chunk_size = CHUNK
    follower_request = _request(2)

    batch, _, _ = _leader_schedule([leader_request]).to_scheduler_result([follower_request])

    assert batch.leader_context_chunks == {2: (0, CHUNK)}
    assert batch.context_requests_chunking == [follower_request]


@pytest.mark.parametrize("is_kv_manager_v2", [False, True], ids=["kv_v1", "kv_v2"])
def test_follower_runs_the_leader_chunk_its_own_pass_refused(is_kv_manager_v2):
    """The failure this covers: the local pass refused the first chunk,
    which left the request at its whole prompt."""
    leader_request = _request(2)
    leader_request.context_chunk_size = CHUNK
    follower_request = _request(2)
    assert follower_request.context_chunk_size == PROMPT_LEN
    batch, _, _ = _leader_schedule([leader_request]).to_scheduler_result([follower_request])
    executor = _follower(is_kv_manager_v2)

    PyExecutor._pp_adopt_leader_schedule(executor, batch, _local_pass())

    executor.scheduler.adopt_context_chunk.assert_called_once_with(follower_request, CHUNK)
    assert follower_request.context_chunk_size == CHUNK
    assert not follower_request.is_last_context_chunk


def test_follower_whose_pass_agrees_changes_nothing():
    leader_context, leader_generation = _request(2), _request(1, prompt_len=16)
    leader_context.context_chunk_size = CHUNK
    follower_context, follower_generation = _request(2), _request(1, prompt_len=16)
    follower_context.context_chunk_size = CHUNK
    batch, _, _ = _leader_schedule([leader_context], [leader_generation]).to_scheduler_result(
        [follower_generation, follower_context]
    )
    executor = _follower(is_kv_manager_v2=True)

    PyExecutor._pp_adopt_leader_schedule(
        executor, batch, _local_pass([follower_context], [follower_generation])
    )

    executor.scheduler.adopt_context_chunk.assert_not_called()
    executor.scheduler.adopt_generation_step.assert_not_called()
    executor.scheduler.release_context_admission.assert_not_called()
    executor.kv_cache_manager.revert_allocate_generation.assert_not_called()
    executor._revert_ctx_alloc.assert_not_called()


def test_v2_follower_allocates_the_leader_batch_and_undoes_local_only_work():
    leader_context, leader_generation = _request(2), _request(1, prompt_len=16)
    leader_context.context_chunk_size = CHUNK
    follower = {i: _request(i, prompt_len=16 if i in (1, 4) else PROMPT_LEN) for i in (1, 2, 3, 4)}
    batch, _, _ = _leader_schedule([leader_context], [leader_generation]).to_scheduler_result(
        list(follower.values())
    )
    executor = _follower(is_kv_manager_v2=True)
    # The local pass admitted request 3's first chunk and grew request 4's
    # generation step instead of what the first rank chose.
    local = _local_pass([follower[3]], [follower[4]])

    PyExecutor._pp_adopt_leader_schedule(executor, batch, local)

    executor.scheduler.adopt_context_chunk.assert_called_once_with(follower[2], CHUNK)
    executor.scheduler.adopt_generation_step.assert_called_once_with(follower[1])
    executor.scheduler.release_context_admission.assert_called_once_with(follower[3])
    executor.kv_cache_manager.revert_allocate_generation.assert_called_once_with(follower[4])
    executor._revert_ctx_alloc.assert_not_called()
    # The local-only admission gives its pages back before the batch takes its own.
    calls = [name for name, _, _ in executor.scheduler.mock_calls]
    assert calls.index("release_context_admission") < calls.index("adopt_context_chunk")


def test_follower_fails_loudly_when_the_leader_chunk_does_not_fit():
    leader_request = _request(2)
    leader_request.context_chunk_size = CHUNK
    follower_request = _request(2)
    batch, _, _ = _leader_schedule([leader_request]).to_scheduler_result([follower_request])
    executor = _follower(is_kv_manager_v2=True)
    executor.scheduler.adopt_context_chunk.side_effect = None
    executor.scheduler.adopt_context_chunk.return_value = False

    with pytest.raises(RuntimeError, match="cannot allocate KV cache"):
        PyExecutor._pp_adopt_leader_schedule(executor, batch, _local_pass())


def test_follower_without_leader_chunks_is_untouched():
    """A rank that scheduled the batch itself carries no leader chunks."""
    executor = _follower(is_kv_manager_v2=True)
    batch = ScheduledRequests()
    batch.reset_context_requests([_request(2)])

    PyExecutor._pp_adopt_leader_schedule(executor, batch, _local_pass())

    executor.scheduler.adopt_context_chunk.assert_not_called()


def test_v2_scheduler_allocates_the_adopted_chunk():
    scheduler = object.__new__(KVCacheV2Scheduler)
    scheduler._prepare_context_pair = Mock(return_value=True)
    scheduler._try_allocate_context = Mock(return_value=True)
    request = _request(2)

    assert scheduler.adopt_context_chunk(request, CHUNK)

    scheduler._prepare_context_pair.assert_called_once_with(request)
    scheduler._try_allocate_context.assert_called_once_with(request, CHUNK)
    assert request.context_chunk_size == CHUNK


def test_v2_scheduler_refuses_an_adopted_chunk_without_a_cache():
    scheduler = object.__new__(KVCacheV2Scheduler)
    scheduler._prepare_context_pair = Mock(return_value=False)
    scheduler._try_allocate_context = Mock()

    assert not scheduler.adopt_context_chunk(_request(2), CHUNK)
    scheduler._try_allocate_context.assert_not_called()


def test_nonzero_pp_rank_adopts_the_leader_schedule_after_its_local_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StopAfterAdoption(RuntimeError):
        pass

    executor = object.__new__(PyExecutor)
    executor.dist = Mock(pp_rank=1, rank=1)
    executor.device_id = 0
    profiler = MagicMock()
    profiler.__enter__.return_value = Mock()
    executor._profiler = Mock(return_value=profiler)
    executor.hang_detector = MagicMock()
    executor.enable_iter_perf_stats = False
    executor._is_kv_manager_v2 = True
    executor._pp_rebalance_drain_iters = None
    executor._can_pause_for_rebalance = Mock(return_value=False)
    executor._fetch_and_activate_new_requests = Mock(return_value=[])
    executor.is_shutdown = False
    executor._handle_control_request = Mock()
    executor.kv_cache_transceiver = Mock()
    executor._pad_attention_dp_dummy_request = Mock()
    executor._pp_retry_until_can_schedule = Mock()
    executor.inflight_req_ids = set()
    # No store tier attached: KVCacheManagerV2.store_level keeps its default None.
    executor.kv_cache_manager = Mock(store_level=None)
    executor.scheduler = Mock()
    executor.active_requests = []
    scheduled_batch = ScheduledRequests()
    executor._pp_schedule_and_propagate = Mock(return_value=(scheduled_batch, [], 0, False))
    local_pass = types.SimpleNamespace(
        context_requests=[], generation_requests=[], fitting_disagg_gen_init_requests=[]
    )
    executor.scheduler.schedule_request.return_value = local_pass
    executor.disagg.poll_gen_transfers = Mock()
    executor._pp_adopt_leader_schedule = Mock(side_effect=StopAfterAdoption)

    monkeypatch.setattr("tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.set_device", Mock())
    monkeypatch.setattr("tensorrt_llm._torch.pyexecutor.py_executor.cudart.cudaSetDevice", Mock())
    monkeypatch.setattr("tensorrt_llm._torch.pyexecutor.py_executor.CUASSERT", Mock())

    with pytest.raises(StopAfterAdoption):
        PyExecutor._executor_loop_pp(executor)

    executor.scheduler.schedule_request.assert_called_once()
    executor._pp_adopt_leader_schedule.assert_called_once_with(scheduled_batch, local_pass)


@pytest.mark.parametrize(
    "attr, value",
    [
        ("_pp_adopt_leader_chunks", True),
        ("_pp_adopt_in_admission_order", True),
        ("_subagent_idle_drop_on", True),
        ("_longchain_protect_on", True),
    ],
)
def test_schedule_fields_follow_the_features(attr, value):
    """The first rank attaches the pipeline-parallel fields to its schedule
    message only when one of their features is on (a stage agreement included)."""
    executor = object.__new__(PyExecutor)
    assert executor._pp_schedule_fields_on(None) is False
    assert executor._pp_schedule_fields_on(object()) is True
    setattr(executor, attr, value)
    assert executor._pp_schedule_fields_on(None) is True
