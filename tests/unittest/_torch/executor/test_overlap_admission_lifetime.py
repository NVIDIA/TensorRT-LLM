# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import MethodType, SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor._util import (
    compute_max_num_sequences,
    should_enable_overlap_headroom,
)
from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.kv_cache_load import KvLoadTracker
from tensorrt_llm._torch.pyexecutor.llm_request import ExecutorRequest, LlmRequest, LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler import FCFSWaitingQueue, ScheduledRequests
from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import DefaultADPRouter
from tensorrt_llm._torch.pyexecutor.seq_slot_manager import SeqSlotManager
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


def _admission_executor(
    rank: int, rank_count: int, batch_size: int, headroom: bool
) -> SimpleNamespace:
    dist = SimpleNamespace(
        rank=rank,
        tp_rank=rank,
        tp_size=rank_count,
        cp_rank=0,
        cp_size=1,
        cp_config={},
        has_cp_helix=False,
    )
    executor = SimpleNamespace(
        dist=dist,
        adp_router=DefaultADPRouter(dist=dist, has_seq_slot_headroom=headroom),
        enable_attention_dp=True,
        max_num_active_requests=batch_size,
        enable_iter_perf_stats=False,
        num_fetch_requests=0,
        num_fetch_requests_cur_rank=0,
        is_benchmark_disagg=False,
        _enable_non_overlap_adp_forward_intent=True,
        _kv_load_tracker=KvLoadTracker(None, rank=rank, enabled=False),
        _kv_cache_capacity={},
        _kv_connector_moves_kv=False,
        _kv_connector_deadlines={},
        kv_cache_transceiver=None,
        _should_exclude_last_generation_logits=lambda: False,
        observed_live_counts=[],
    )
    executor._fetch_and_enqueue_requests = lambda queue, live: executor.observed_live_counts.append(
        live
    )
    for name in ("_get_request_admission_capacity", "_pop_from_waiting_queue"):
        setattr(executor, name, MethodType(getattr(PyExecutor, name), executor))
    return executor


def _fetch_all_ranks(
    executors: list[SimpleNamespace],
    queues: list[FCFSWaitingQueue],
    residents: list[list[LlmRequest]],
) -> list[list[LlmRequest]]:
    # Produce peer payloads through the real router before simulating transport.
    payloads = []
    for executor, active in zip(executors, residents, strict=True):
        executor.dist.tp_allgather = lambda payload: [payload]
        state = executor.adp_router.gather_all_rank_states(active)[0]
        payloads.append(state.serialize())

    admitted = []
    for rank, (executor, queue, active) in enumerate(
        zip(executors, queues, residents, strict=True)
    ):
        executor.dist.tp_allgather = lambda payload, rank=rank: [
            payload if peer == rank else peer_payload for peer, peer_payload in enumerate(payloads)
        ]
        admitted.append(PyExecutor._fetch_new_requests(executor, queue, active))
    return admitted


def _seat(manager: SeqSlotManager, requests: list[LlmRequest]) -> None:
    batch = ScheduledRequests()
    batch.generation_requests = requests
    manager.prepare_resources(batch)


def _enqueue_cohort(queues: list[FCFSWaitingQueue], first_id: int, count: int) -> None:
    for queue in queues:
        queue.add_requests(
            RequestQueueItem(
                request_id,
                ExecutorRequest(input_token_ids=[1, 2], max_tokens=1),
            )
            for request_id in range(first_id, first_id + count)
        )


@pytest.mark.parametrize("disable_overlap", [False, True], ids=["overlap", "no-overlap"])
def test_adp_replacements_acquire_slots_before_retiring_requests_release_them(
    disable_overlap: bool,
) -> None:
    rank_count, batch_size = 2, 2
    mapping = Mapping(world_size=rank_count, tp_size=rank_count, enable_attention_dp=True)
    headroom = should_enable_overlap_headroom(mapping, disable_overlap, kv_cache_manager_is_v2=True)
    capacity = compute_max_num_sequences(
        mapping, batch_size, disable_overlap, enable_overlap_headroom=headroom
    )
    executors = [
        _admission_executor(rank, rank_count, batch_size, headroom) for rank in range(rank_count)
    ]
    queues = [FCFSWaitingQueue() for _ in executors]
    managers = [SeqSlotManager(capacity) for _ in executors]
    cohort_size = rank_count * batch_size
    _enqueue_cohort(queues, 1, cohort_size)
    resident = _fetch_all_ranks(executors, queues, [[] for _ in executors])
    for manager, requests in zip(managers, resident, strict=True):
        assert len(requests) == batch_size
        _seat(manager, requests)

    for iteration in range(3):
        first_id = 1 + (iteration + 1) * cohort_size
        _enqueue_cohort(queues, first_id, cohort_size)
        for requests in resident:
            for request in requests:
                request.state = LlmRequestState.GENERATION_TO_COMPLETE

        replacements = _fetch_all_ranks(executors, queues, resident)
        for executor, queue, manager, retiring, incoming in zip(
            executors, queues, managers, resident, replacements, strict=True
        ):
            assert executor.observed_live_counts[-1] == cohort_size
            assert len(incoming) == (0 if disable_overlap else batch_size)
            assert len(queue) == (cohort_size if disable_overlap else 0)
            assert len(manager.slot_manager.free_slots) >= len(incoming), (
                "admitted replacements must fit while retiring requests still own their slots"
            )
            _seat(manager, incoming)
            slots = [request.py_seq_slot for request in retiring + incoming]
            assert len(set(slots)) == len(slots)
            assert all(0 <= slot < capacity for slot in slots)
            for request in retiring:
                manager.free_resources(request)

        if disable_overlap:
            replacements = _fetch_all_ranks(executors, queues, [[] for _ in executors])
            for manager, incoming in zip(managers, replacements, strict=True):
                _seat(manager, incoming)
        admitted_ids = [request.request_id for requests in replacements for request in requests]
        assert sorted(admitted_ids) == list(range(first_id, first_id + cohort_size))
        assert all(len(requests) == batch_size for requests in replacements)
        assert all(len(queue) == 0 for queue in queues)
        resident = replacements

    for manager, requests in zip(managers, resident, strict=True):
        for request in requests:
            manager.free_resources(request)
        assert len(manager.slot_manager.free_slots) == capacity
