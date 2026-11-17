# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np
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
from tensorrt_llm.bindings.executor import Request, RequestType
from tensorrt_llm.executor.request import DEFAULT_REQUEST_PRIORITY
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
        is_warmup=False,
        _enable_non_overlap_adp_forward_intent=True,
        _fetch_hint_enabled=False,
        _kv_load_tracker=KvLoadTracker(None, rank=rank, enabled=False),
        _kv_cache_capacity={},
        _kv_connector_moves_kv=False,
        _kv_connector_deadlines={},
        kv_cache_transceiver=None,
        canceled_req_ids=[],
        request_broadcaster=SimpleNamespace(deliver_prompts=lambda requests: None),
        track_unique_tokens=False,
        _should_exclude_last_generation_logits=lambda: False,
        observed_live_counts=[],
        _num_embeddings=123,
    )
    executor._fetch_and_enqueue_requests = (
        lambda queue, live, **_: executor.observed_live_counts.append(live)
    )
    for name in (
        "_get_request_admission_capacity",
        "_pop_from_waiting_queue",
        "_cancel_queued_context_requests",
    ):
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
        executor.dist.tp_allgather_int64 = lambda payload: np.asarray([payload], dtype=np.int64)
        state = executor.adp_router.gather_all_rank_states(active)[0]
        payloads.append(state.serialize())

    admitted = []
    for rank, (executor, queue, active) in enumerate(
        zip(executors, queues, residents, strict=True)
    ):
        executor.dist.tp_allgather_int64 = lambda payload, rank=rank: np.asarray(
            [
                payload if peer == rank else peer_payload
                for peer, peer_payload in enumerate(payloads)
            ],
            dtype=np.int64,
        )
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
                ExecutorRequest.adapt(Request(input_token_ids=[1, 2], max_tokens=1)),
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


def _generation_only_item(request_id: int, child_req_ids=None) -> RequestQueueItem:
    request = SimpleNamespace(
        request_type=RequestType.REQUEST_TYPE_GENERATION_ONLY,
        num_input_tokens=2,
        py_scheduling_params=None,
        priority=DEFAULT_REQUEST_PRIORITY,
    )
    return RequestQueueItem(request_id, request, child_req_ids=child_req_ids)


@pytest.mark.parametrize("blocker", ["context", "child-bearing"])
def test_adp_receive_ahead_admits_only_the_leading_generation_only_run(blocker: str) -> None:
    # Under attention DP the executor gives batch 3 two slots, leaving the padding dummy a seat.
    rank_count, batch_size, slots = 2, 3, 2
    mapping = Mapping(world_size=rank_count, tp_size=rank_count, enable_attention_dp=True)
    headroom = should_enable_overlap_headroom(mapping, False, kv_cache_manager_is_v2=True)
    capacity = compute_max_num_sequences(
        mapping, batch_size, False, enable_overlap_headroom=headroom
    )
    executors = [
        _admission_executor(rank, rank_count, batch_size, headroom) for rank in range(rank_count)
    ]
    queues = [FCFSWaitingQueue() for _ in executors]
    _enqueue_cohort(queues, 1, rank_count * batch_size)
    resident = _fetch_all_ranks(executors, queues, [[] for _ in executors])
    for requests in resident:
        for request in requests:
            request.state = LlmRequestState.GENERATION_IN_PROGRESS
    # Rank 0 decodes a full batch; rank 1 decodes one request beside a retiring one.
    resident[1][0].state = LlmRequestState.GENERATION_TO_COMPLETE

    for queue in queues:
        queue.add_requests(
            [
                _generation_only_item(101),
                _generation_only_item(102),
                RequestQueueItem(104, Request(input_token_ids=[1, 2], max_tokens=1))
                if blocker == "context"
                else _generation_only_item(104, child_req_ids=[1104]),
                _generation_only_item(105),
            ]
        )
    for executor in executors:
        executor._receive_ahead_slots = slots
    with patch(
        "tensorrt_llm._torch.pyexecutor.py_executor.merge_requests",
        side_effect=lambda requests, **_: requests,
    ):
        incoming = _fetch_all_ranks(executors, queues, resident)

    # The batches have room for one request: the rest of the leading run goes
    # beyond them, and the blocker and the request behind it wait, though the
    # raised room would fit the blocker too.
    assert sorted(item.id for items in incoming for item in items) == [101, 102]
    assert all([item.id for item in queue] == [104, 105] for queue in queues)
    for active, items in zip(resident, incoming, strict=True):
        assert len(active) + len(items) <= min(batch_size + slots, capacity)
