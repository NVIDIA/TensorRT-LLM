# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.request_delivery import ResponseChannel
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
        executor._reset_scheduler_failure_state()
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
        executor.active_requests = [healthy, *failed]
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
            requests=failed,
            charge_budget=False,
        )
    assert batches[0].context_requests == [healthy]


def test_admission_failure_delivers_one_error_and_cleans_up_once():
    executor = object.__new__(PyExecutor)
    executor._reset_scheduler_failure_state()
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
    executor._response_channel = ResponseChannel()
    executor._response_channel.register([7], result_queue)
    executor._merged_transfer_flush, executor._stage_responses = None, False
    executor._maybe_record_hang_diagnostic_phase = Mock()

    def check_delivery_before_free(request):
        result_queue.put_response.remote.assert_called_once()
        assert executor._response_channel.result_wait_queues[7] is result_queue

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
    assert executor._response_channel.responses[7] == [response]
    assert 7 not in executor._response_channel.result_wait_queues
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
    executor._reset_scheduler_failure_state()
    executor._is_kv_manager_v2, executor.enable_attention_dp = True, False
    failed = SimpleNamespace(py_request_id=7)
    executor.active_requests = [failed]
    executor.dist = SimpleNamespace(
        tp_size=1,
        cp_size=1,
        pp_size=3,
        allgather=Mock(return_value=[([7], False), ([], False), ([], False)]),
    )
    executor._handle_errors = Mock()
    batch = ScheduledRequests()
    batch.failed_requests = [failed]
    for _ in range(63):
        executor._fail_scheduler_requests(batch)
    executor.dist.allgather.assert_not_called()
    executor._handle_errors.assert_not_called()
    executor._fail_scheduler_requests(batch)
    executor.dist.allgather.assert_called_once_with(([7], False))
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
    executor._reset_scheduler_failure_state()
    executor._disagg_coordinator = SimpleNamespace(has_local_retiring_gen_receives=False)
    executor._transfer_held_failures = {7: (object(), "parked receive")}
    assert executor._has_scheduler_transfer_holds()
    executor._transfer_held_failures.clear()
    assert not executor._has_scheduler_transfer_holds()


@pytest.mark.parametrize("progress", [False, True])
def test_persistent_admission_failures_escalate_only_without_progress(progress):
    executor = object.__new__(PyExecutor)
    executor._reset_scheduler_failure_state()
    executor._is_kv_manager_v2, executor.enable_attention_dp = True, False
    executor.dist = SimpleNamespace(
        tp_size=2, cp_size=1, pp_size=1, tp_allgather=lambda vote: [vote, ([], False)]
    )
    failed = SimpleNamespace(py_request_id=7)
    healthy = SimpleNamespace(py_request_id=8)
    executor.active_requests = [failed, healthy]
    executor._handle_errors = Mock()
    executor._scheduler_stall_rounds = 2 if progress else 0
    for round_id in range(3):
        batch = ScheduledRequests()
        batch.failed_requests = [failed]
        batch.generation_requests = [healthy] if progress else []
        # The forward-success hook latches work executed since the last vote.
        executor._scheduler_progress_since_vote = progress
        executor._scheduler_failure_sync_tick = 63
        if round_id == 2 and not progress:
            with pytest.raises(RuntimeError, match="three failure rounds"):
                executor._fail_scheduler_requests(batch)
        else:
            executor._fail_scheduler_requests(batch)
    assert executor._handle_errors.call_count == (3 if progress else 2)
    assert executor._scheduler_stall_rounds == (0 if progress else 3)


def test_adp_failure_consensus_covers_both_cp_and_pp_without_tp_vote():
    executor = object.__new__(PyExecutor)
    executor._reset_scheduler_failure_state()
    executor._scheduler_failure_sync_tick = 63
    executor._is_kv_manager_v2, executor.enable_attention_dp = True, True
    executor.dist = SimpleNamespace(
        tp_size=2,
        cp_size=2,
        pp_size=2,
        cp_allgather=Mock(return_value=[([], False), ([7], False)]),
        pp_allgather=Mock(return_value=[([7], False), ([9], False)]),
    )
    failed = [SimpleNamespace(py_request_id=rid) for rid in (7, 9)]
    executor.active_requests = failed
    executor._handle_errors = Mock()
    batch = ScheduledRequests()
    executor._fail_scheduler_requests(batch)
    executor.dist.cp_allgather.assert_called_once_with(([], False))
    executor.dist.pp_allgather.assert_called_once_with(([7], False))
    assert executor._handle_errors.call_args.kwargs["requests"] == failed


def test_mpi_scheduler_failure_consensus_spans_cp_and_pp():
    from mpi4py import MPI

    world = MPI.COMM_WORLD
    if world.Get_size() != 4:
        pytest.skip("launch this test with mpirun -np 4")
    rank = world.Get_rank()
    cp, pp = world.Split(rank // 2, rank), world.Split(rank % 2, rank)
    try:
        executor = object.__new__(PyExecutor)
        executor._reset_scheduler_failure_state()
        executor._is_kv_manager_v2 = executor.enable_attention_dp = True
        executor.dist = SimpleNamespace(
            tp_size=1, cp_size=2, pp_size=2, cp_allgather=cp.allgather, pp_allgather=pp.allgather
        )
        failed = SimpleNamespace(py_request_id=7)
        executor.active_requests = [failed]
        executor._handle_errors = Mock()
        for round_id in range(3):
            executor._scheduler_failure_sync_tick = 63
            batch = ScheduledRequests()
            batch.failed_requests = [failed] if rank == 3 else []
            batch.generation_requests = [] if rank == 3 else [failed]
            if round_id < 2:
                executor._fail_scheduler_requests(batch)
                assert not batch.all_requests()
                assert executor._scheduler_stall_rounds == round_id + 1
                assert executor._handle_errors.call_args.kwargs["requests"] == [failed]
            else:
                with pytest.raises(RuntimeError, match="three failure rounds"):
                    executor._fail_scheduler_requests(batch)
        assert executor._handle_errors.call_count == 2
        # Reuse these communicators for transport failure/retirement agreement.
        from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

        node = object.__new__(KvCacheTransceiverV2)
        for drained in (False, True):
            result = (
                [],
                [],
                [],
                [73] if drained or rank != 3 else [],
                {73} if rank == 3 else set(),
                drained or rank != 3,
            )
            for gather in (cp.allgather, pp.allgather):
                c, f, d, q, intents, ready = result
                result = node._consensus_outcome(
                    [73], c, f, d, gather, True, q, failure_intents=list(intents), agent_ready=ready
                )
            assert result[1] == [73] and result[3] == ([73] if drained else [])
            assert result[-1] is drained
    finally:
        cp.Free()
        pp.Free()


@pytest.mark.parametrize("parallelism", ["tp", "cp", "pp"])
def test_rejected_rows_never_reset_stall_rounds_without_execution(parallelism):
    barrier = threading.Barrier(2, timeout=5)
    votes = [None, None]

    def gather(rank, vote):
        votes[rank] = vote
        barrier.wait()
        result = list(votes)
        barrier.wait()
        return result

    executors = []
    for rank in range(2):
        executor = object.__new__(PyExecutor)
        executor._reset_scheduler_failure_state()
        executor._is_kv_manager_v2, executor.enable_attention_dp = True, False
        executor.dist = SimpleNamespace(
            tp_size=2 if parallelism == "tp" else 1,
            cp_size=2 if parallelism == "cp" else 1,
            pp_size=2 if parallelism == "pp" else 1,
        )
        collective = {"tp": "tp_allgather", "cp": "tp_cp_allgather", "pp": "allgather"}[parallelism]
        setattr(executor.dist, collective, lambda vote, rank=rank: gather(rank, vote))
        executor._handle_errors = Mock()
        executors.append(executor)
    with ThreadPoolExecutor(max_workers=2) as pool:
        # Each round is a new arrival, accepted by rank 0 and rejected by rank
        # 1. Agreement leaves no executable rows on either rank.
        for round_id in range(3):
            request = SimpleNamespace(py_request_id=round_id)
            batches = []
            for rank, executor in enumerate(executors):
                executor.active_requests = [request]
                executor._scheduler_failure_sync_tick = 63
                batch = ScheduledRequests()
                batch.generation_requests = [request] if rank == 0 else []
                batch.failed_requests = [request] if rank == 1 else []
                batches.append(batch)
            futures = [
                pool.submit(ex._fail_scheduler_requests, batch)
                for ex, batch in zip(executors, batches)
            ]
            for executor, batch, future in zip(executors, batches, futures):
                if round_id == 2:
                    with pytest.raises(RuntimeError, match="three failure rounds"):
                        future.result(timeout=10)
                else:
                    future.result(timeout=10)
                    assert not batch.all_requests()
                assert executor._scheduler_stall_rounds == round_id + 1
                assert not executor._scheduler_progress_since_vote
    assert all(ex._handle_errors.call_count == 2 for ex in executors)
