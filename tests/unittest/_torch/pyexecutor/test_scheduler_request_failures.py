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

    failed = [SimpleNamespace(py_request_id=7), SimpleNamespace(py_request_id=9)]
    healthy = SimpleNamespace(py_request_id=8, is_last_context_chunk=True)
    executors, batches = [], []
    for rank in range(2):
        executor = object.__new__(PyExecutor)
        executor._is_kv_manager_v2 = True
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
        batch.generation_requests = failed.copy()
        batch.reset_context_requests([healthy] if rank == 0 else [])
        executors.append(executor)
        batches.append(batch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(e._fail_scheduler_requests, b) for e, b in zip(executors, batches)]
        for future in futures:
            future.result(timeout=10)
    for executor, batch in zip(executors, batches):
        assert batch.generation_requests == []
        assert executor._handle_errors.call_args.kwargs["charge_budget"] is False
    assert executors[1]._handle_errors.call_args.kwargs["requests"] == failed
    assert executors[0]._handle_errors.call_args.kwargs["requests"] == (
        failed if parallelism == "pp" else []
    )
    assert batches[0].context_requests == [healthy]


def test_admission_failure_delivers_one_error_and_cleans_up_once():
    executor = object.__new__(PyExecutor)
    executor._is_kv_manager_v2 = True
    executor.dist = SimpleNamespace(rank=0, world_size=1, tp_size=1, cp_size=1, pp_size=1)
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
    executor.disagg = Mock()
    executor.model_engine = executor.draft_model_engine = None
    executor.result_wait_queues = {}
    executor._enqueue_responses = Mock()
    batch = ScheduledRequests()
    batch.failed_requests = [blocked]
    batch.reset_context_requests([healthy])

    executor._fail_scheduler_requests(batch)
    executor._fail_scheduler_requests(batch)

    executor._enqueue_responses.assert_called_once()
    [(rid, response)] = executor._enqueue_responses.call_args.args[0]
    assert rid == blocked.py_request_id
    assert response.error_msg == "KV cache admission made no progress within the scheduler limit"
    executor.sampler.release_request.assert_called_once_with(blocked)
    executor.resource_manager.free_resources.assert_called_once_with(blocked)
    executor._error_budget.consume.assert_not_called()
    assert executor.active_requests == [healthy]
    assert batch.context_requests == [healthy]
    assert blocked.py_kv_reuse_poisoned
    assert not executor.is_shutdown
