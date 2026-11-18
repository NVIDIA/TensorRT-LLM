# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("parallelism", ["tp", "cp", "pp"])
def test_asymmetric_admission_failure_uses_request_error_path(parallelism):
    barrier = threading.Barrier(2, timeout=5)
    votes = [None, None]

    def gather(rank, value):
        votes[rank] = value
        barrier.wait()
        result = list(votes)
        barrier.wait()
        return result

    failed = [
        SimpleNamespace(py_request_id=7, is_last_context_chunk=True),
        SimpleNamespace(py_request_id=9),
    ]
    healthy = SimpleNamespace(py_request_id=8, is_last_context_chunk=True)
    executors, batches = [], []
    for rank in range(2):
        executor = object.__new__(PyExecutor)
        executor._is_kv_manager_v2 = True
        executor.enable_attention_dp = False
        executor._scheduler_failure_sync_tick = 63
        executor.dist = SimpleNamespace(
            cp_size=2 if parallelism == "cp" else 1,
            tp_size=2 if parallelism == "tp" else 1,
            pp_size=2 if parallelism == "pp" else 1,
        )
        collective = {"tp": "tp_allgather", "cp": "tp_cp_allgather", "pp": "allgather"}[parallelism]
        setattr(executor.dist, collective, lambda ids, rank=rank: gather(rank, ids))
        executor.active_requests = (
            [healthy, *failed] if parallelism == "pp" else [healthy] if rank == 0 else failed.copy()
        )
        executor._handle_errors = Mock()
        batch = ScheduledRequests()
        batch.failed_requests = failed.copy() if rank == 1 else []
        batch.generation_requests = (
            failed[1:] if parallelism == "pp" and rank == 0 else failed.copy()
        )
        batch.reset_context_requests(
            [failed[0], healthy]
            if parallelism == "pp" and rank == 0
            else [healthy]
            if rank == 0
            else []
        )
        executors.append(executor)
        batches.append(batch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(e._fail_scheduler_requests, b) for e, b in zip(executors, batches)]
        for future in futures:
            future.result(timeout=10)
    for rank, (executor, batch) in enumerate(zip(executors, batches)):
        assert batch.generation_requests == []
        executor._handle_errors.assert_called_once_with(
            "KV cache admission made no progress within the scheduler limit",
            requests=failed if rank == 1 or parallelism == "pp" else [],
            charge_budget=False,
        )
    assert batches[0].context_requests == [healthy]


def test_admission_failure_delivers_one_error_and_cleans_up_once():
    executor = object.__new__(PyExecutor)
    executor._is_kv_manager_v2 = True
    executor.dist = SimpleNamespace(
        rank=0,
        world_size=1,
        tp_size=1,
        cp_size=1,
        pp_size=1,
        mapping=SimpleNamespace(tp_group=[0]),
    )
    blocked = SimpleNamespace(
        py_request_id=7,
        request_id=7,
        py_client_id=17,
        context_current_position=0,
        context_remaining_length=100,
    )
    healthy = SimpleNamespace(py_request_id=8, is_last_context_chunk=True)
    executor.active_requests = [blocked, healthy]
    executor.kv_connector_manager = executor.kv_cache_transceiver = None
    executor.async_transfer_manager = executor._disagg_pp_termination_handler = None
    executor.enable_attention_dp = executor.gather_all_responses = False
    executor._fatal_error, executor.is_shutdown = None, False
    executor._error_budget = Mock()
    executor._wait_for_model_engine_input_copy = Mock()
    executor._pending_transfer_responses, executor._pending_response_terminations = [], []
    executor._prefetched_request_ids = {7, 8}
    executor._release_multimodal_resources = Mock()
    executor.sampler, executor.resource_manager = Mock(), Mock()
    executor.model_engine = executor.draft_model_engine = None
    result_queue = Mock()
    executor.result_wait_queues = {7: result_queue}
    executor.response_cv, executor.responses = threading.Condition(), {}
    executor._merged_transfer_flush, executor._stage_responses = None, False
    executor._maybe_record_hang_diagnostic_phase = Mock()

    def check_delivery_before_free(request):
        result_queue.put_response.remote.assert_called_once()
        assert executor.result_wait_queues[7] is result_queue

    executor.resource_manager.free_resources.side_effect = check_delivery_before_free
    batch = ScheduledRequests()
    batch.failed_requests = [blocked]
    batch.reset_context_requests([healthy])

    executor._fail_scheduler_requests(batch)
    executor._fail_scheduler_requests(batch)

    result_queue.put_response.remote.assert_called_once()
    client_id, response = result_queue.put_response.remote.call_args.args
    assert client_id == 17
    assert response.request_id == blocked.py_request_id
    assert executor.responses[7] == [response]
    assert 7 not in executor.result_wait_queues
    assert response.error_msg == "KV cache admission made no progress within the scheduler limit"
    executor.sampler.release_request.assert_called_once_with(blocked)
    executor.resource_manager.free_resources.assert_called_once_with(blocked)
    executor._error_budget.consume.assert_not_called()
    assert executor.active_requests == [healthy]
    assert batch.context_requests == [healthy]
    assert blocked.py_kv_reuse_poisoned
    assert not executor.is_shutdown


def test_model_parallel_failure_votes_are_batched_and_adp_is_rank_local():
    executor = object.__new__(PyExecutor)
    executor._is_kv_manager_v2, executor.enable_attention_dp = True, False
    failed = SimpleNamespace(py_request_id=7)
    executor.active_requests = [failed]
    executor.dist = SimpleNamespace(
        tp_size=1, cp_size=1, pp_size=3, allgather=Mock(return_value=[[7], [], []])
    )
    executor._handle_errors = Mock()
    batch = ScheduledRequests()
    batch.failed_requests = [failed]
    for _ in range(63):
        executor._fail_scheduler_requests(batch)
    executor.dist.allgather.assert_not_called()
    executor._handle_errors.assert_not_called()
    executor._fail_scheduler_requests(batch)
    executor.dist.allgather.assert_called_once_with([7])
    executor._handle_errors.assert_called_once()

    executor.enable_attention_dp = True
    executor.dist.pp_size, executor.dist.tp_size = 1, 3
    executor.dist.allgather.reset_mock()
    batch.failed_requests = [failed]
    executor._fail_scheduler_requests(batch)
    executor.dist.allgather.assert_not_called()
    assert executor._handle_errors.call_count == 2


def test_held_transfer_remains_visible_to_the_scheduler_hold_probe():
    executor = object.__new__(PyExecutor)
    executor._disagg_coordinator = SimpleNamespace(has_local_retiring_gen_receives=False)
    executor._transfer_held_failures = {7: (object(), "parked receive")}
    assert executor._has_scheduler_transfer_holds()
    executor._transfer_held_failures.clear()
    assert not executor._has_scheduler_transfer_holds()
