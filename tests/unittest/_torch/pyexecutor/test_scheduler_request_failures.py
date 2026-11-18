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
        executor.active_requests = [healthy] if rank == 0 else failed.copy()
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
    assert executors[0]._handle_errors.call_args.kwargs["requests"] == []
    assert batches[0].context_requests == [healthy]
