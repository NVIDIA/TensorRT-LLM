# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attention DP: a request that fails in guided decoding gets its error response.

Two attention-DP ranks run in threads over an in-process TP group. Rank 1
owns a guided request whose grammar matcher reports a state with no valid
token, so the real ``GuidedDecoder._build`` fails it and the real
``_handle_guided_decoder_errors`` / ``_handle_errors`` path runs. Under
attention DP that path removes the request from ``active_requests`` and
buffers its error response for ``_flush_pending_transfer_responses``, the one
response gather every rank enters together.

The overlap executor loop handles guided-decoding errors after that pass's
flush, so the response is published by the next pass. That pass starts with
the routing fetch. When the failed request was the last live request in the
fleet, the fetch must not take the idle path, where rank 0 blocks on the
request queue until new traffic arrives: the flush would never run and the
client would wait forever.

The fetch, router, error, flush and response-gather code is real; queue,
broadcaster and termination are stubs.
"""

import datetime
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from threading import Barrier
from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.guided_decoder import (
    GuidedDecoder,
    GuidedRequest,
    GuidedRequests,
)
from tensorrt_llm._torch.pyexecutor.kv_cache_load import KvLoadTracker
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import DefaultADPRouter, RankState

pytestmark = pytest.mark.cpu_only

_VOCAB_SIZE = 64
_FAILING_REQUEST_ID = 7
_HEALTHY_REQUEST_ID = 8
_OWNER_RANK = 1
_NO_WAIT = datetime.timedelta(0)

# Grammar handles; the injecting matcher factory keys its behavior on them.
_DEAD_END_GRAMMAR = object()
_HEALTHY_GRAMMAR = object()


class _TpGroup:
    """In-process TP group: every collective is a barrier-paired exchange.

    Collectives pair by call order, so a rank that enters one its peer does
    not enter breaks the barrier after its timeout and fails the test.
    """

    def __init__(self, size: int) -> None:
        self.size = size
        self.barrier = Barrier(size, timeout=10)
        self.values: List[object] = [None] * size

    def allgather(self, rank: int, value: object) -> List[object]:
        self.values[rank] = value
        self.barrier.wait()
        result = list(self.values)
        self.barrier.wait()
        return result

    def gather(self, rank: int, value: object) -> Optional[List[object]]:
        result = self.allgather(rank, value)
        return result if rank == 0 else None


class _Matcher:
    """Grammar matcher whose next-token rows are all valid or all dead."""

    def __init__(self, dead_end: bool) -> None:
        self.dead_end = dead_end

    def accept_token(self, token_id: int) -> bool:
        return True

    def fill_next_token_bitmask(self, bitmask: torch.Tensor, index: int) -> None:
        bitmask[index].fill_(0 if self.dead_end else -1)

    def is_terminated(self) -> bool:
        return False


class _InjectingMatcherFactory:
    """Creates a dead-end matcher for the chosen grammar, a healthy one otherwise."""

    def create(self, guided_decoding_params: object, stop_token_ids: object) -> _Matcher:
        return _Matcher(dead_end=guided_decoding_params is _DEAD_END_GRAMMAR)


def _guided_decoder(max_num_sequences: int = 2) -> GuidedDecoder:
    """GuidedDecoder with host buffers only; ``_build`` runs unchanged."""
    decoder = object.__new__(GuidedDecoder)
    decoder.vocab_size_padded = _VOCAB_SIZE
    decoder.max_num_sequences = max_num_sequences
    decoder.max_num_draft_tokens = 0
    decoder.grammar_matcher_factory = _InjectingMatcherFactory()
    decoder.grammar_matchers = [None] * max_num_sequences
    decoder.bitmask_host = torch.zeros(
        max_num_sequences, math.ceil(_VOCAB_SIZE / 32), dtype=torch.int32
    )
    decoder.token_mask_host = torch.zeros(max_num_sequences, dtype=torch.int32)
    decoder.num_advanced_tokens = [0] * max_num_sequences
    decoder.num_guided_tokens = [0] * max_num_sequences
    decoder.num_advanced_draft_tokens = [0] * max_num_sequences
    decoder.is_draft_terminated = [False] * max_num_sequences
    return decoder


def _last_context_chunk(request_id: int, grammar: object, seq_slot: int) -> GuidedRequest:
    return GuidedRequest(
        guided_decoding_params=grammar,
        request_id=request_id,
        seq_slot=seq_slot,
        is_context_init_state=True,
        is_last_context_chunk=True,
        draft_tokens=[],
    )


def _llm_request(request_id: int) -> SimpleNamespace:
    """The fields the executor error path reads from an LlmRequest."""
    return SimpleNamespace(
        py_request_id=request_id,
        request_id=request_id,
        py_client_id=100 + request_id,
        py_orig_prompt_len=8,
        context_remaining_length=0,
        context_current_position=8,
        state=None,
        is_attention_dp_dummy=False,
    )


def _executor(rank: int, group: _TpGroup) -> PyExecutor:
    """Attention-DP executor shell for the fetch, error and response paths."""
    executor = object.__new__(PyExecutor)
    dist = MagicMock()
    dist.rank = dist.tp_rank = rank
    dist.tp_size = dist.world_size = group.size
    dist.cp_size, dist.cp_rank, dist.has_cp_helix, dist.has_pp = 1, 0, False, False
    dist.mapping.tp_group = list(range(group.size))
    dist.tp_allgather.side_effect = partial(group.allgather, rank)
    dist.tp_gather.side_effect = partial(group.gather, rank)
    executor.dist = dist
    executor.enable_attention_dp = True
    executor.gather_all_responses = False
    executor.enable_iter_perf_stats = False
    executor.is_shutdown = False

    # Response publication (_enqueue_responses).
    executor.response_cv = threading.Condition()
    executor.responses = {}
    executor.result_wait_queues = {}
    executor._maybe_record_hang_diagnostic_phase = Mock()

    # Request-scoped error path (_handle_errors).
    executor._error_budget = Mock()
    executor._fatal_error = None
    executor.kv_connector_manager = None
    executor.kv_cache_transceiver = None
    executor._transfer_held_failures = {}
    executor._pending_transfer_responses = []
    executor._pending_response_terminations = []
    executor._wait_for_model_engine_input_copy = Mock()
    executor._terminate_request = Mock()
    executor.active_requests = []

    # Routing fetch (_fetch_new_requests / _fetch_and_enqueue_requests).
    executor.adp_router = DefaultADPRouter(dist)
    executor._kv_load_tracker = KvLoadTracker(None, rank=rank, enabled=False)
    executor._kv_cache_capacity = {}
    executor._kv_connector_has_pending = False
    executor._kv_connector_deadlines = {}
    executor._disable_mpi = False
    executor.control_requests = []
    executor.request_accumulated = []
    executor.hang_detector = MagicMock()
    executor.executor_request_queue = MagicMock()
    executor.executor_request_queue.get_from_request_queue.return_value = []
    executor.request_broadcaster = MagicMock()
    executor.request_broadcaster.broadcast.return_value = ([], {})
    executor.waiting_queue = MagicMock()
    executor.waiting_queue.__len__.return_value = 0
    executor._pop_from_waiting_queue = MagicMock(return_value=([], {}))
    executor._enable_non_overlap_adp_forward_intent = True
    executor.should_exclude_last_generation_logits = False
    executor.num_fetch_requests = executor.num_fetch_requests_cur_rank = 0
    executor.max_num_active_requests = 2
    return executor


def _next_pass(executor: PyExecutor) -> None:
    """The collective-bearing head and response flush of an executor pass."""
    executor._fetch_new_requests(executor.waiting_queue, executor.active_requests)
    executor._flush_pending_transfer_responses()


def _run_pass(executors: List[PyExecutor]) -> None:
    with (
        patch(
            "tensorrt_llm._torch.pyexecutor.py_executor.merge_requests",
            side_effect=lambda requests, **kwargs: list(requests),
        ),
        ThreadPoolExecutor(max_workers=len(executors)) as pool,
    ):
        futures = [pool.submit(_next_pass, executor) for executor in executors]
        for future in futures:
            future.result(timeout=30)


def _queue_waits(executor: PyExecutor) -> List[Optional[datetime.timedelta]]:
    calls = executor.executor_request_queue.get_from_request_queue.call_args_list
    return [call.args[0] for call in calls]


def _fail_in_guided_decoding(
    executor: PyExecutor, scheduled: List[SimpleNamespace], guided: List[GuidedRequest]
) -> None:
    """One sampling step of the owner rank: guided build, then error handling."""
    decoder = _guided_decoder()
    failed = decoder._build(
        GuidedRequests(guided, num_contexts=len(guided), num_generations=0, max_num_draft_tokens=0)
    )
    assert [request_id for request_id, _ in failed] == [_FAILING_REQUEST_ID]
    batch = ScheduledRequests()
    batch.context_requests_last_chunk = list(scheduled)
    # Rank-local: the owner reaches this without its peers, so it must not
    # enter a collective here (the in-process group would time out).
    executor._handle_guided_decoder_errors(batch, failed)


@pytest.mark.parametrize("healthy_request", [False, True], ids=["alone", "with_healthy_request"])
def test_guided_decoding_error_reaches_client_under_attention_dp(
    healthy_request: bool,
) -> None:
    group = _TpGroup(size=2)
    executors = [_executor(rank, group) for rank in range(group.size)]
    owner = executors[_OWNER_RANK]

    failing = _llm_request(_FAILING_REQUEST_ID)
    owner.active_requests = [failing]
    scheduled = [failing]
    guided = [_last_context_chunk(_FAILING_REQUEST_ID, _DEAD_END_GRAMMAR, seq_slot=0)]
    if healthy_request:
        healthy = _llm_request(_HEALTHY_REQUEST_ID)
        owner.active_requests.append(healthy)
        scheduled.append(healthy)
        guided.append(_last_context_chunk(_HEALTHY_REQUEST_ID, _HEALTHY_GRAMMAR, seq_slot=1))

    _fail_in_guided_decoding(owner, scheduled, guided)

    # Under attention DP the response waits for the synchronized flush.
    assert failing not in owner.active_requests
    assert [request_id for request_id, _ in owner._pending_transfer_responses] == [
        _FAILING_REQUEST_ID
    ]
    owner._terminate_request.assert_not_called()

    # Next pass: the fleet must stay off the blocking idle fetch until the
    # flush has published the buffered response.
    _run_pass(executors)

    leader = executors[0]
    assert _queue_waits(leader) == [_NO_WAIT], (
        "rank 0 took the blocking idle wait for new requests while rank "
        f"{_OWNER_RANK} still held an undelivered error response"
    )
    for executor in executors:
        assert executor.request_broadcaster.broadcast.call_args.kwargs == {"prefer_cpu": False}

    [response] = leader.responses.pop(_FAILING_REQUEST_ID)
    assert response.client_id == 100 + _FAILING_REQUEST_ID
    assert "Guided decoding error" in response.error_msg
    assert "no valid token" in response.error_msg
    assert leader.responses == {}
    owner._terminate_request.assert_called_once_with(failing)
    assert owner._pending_transfer_responses == []
    assert owner._pending_response_terminations == []
    if healthy_request:
        # The other request is untouched: still active, not answered.
        assert owner.active_requests == [healthy]
        assert healthy.state is None

    # Once delivered, an empty fleet goes idle again instead of spinning.
    if not healthy_request:
        _run_pass(executors)
        assert _queue_waits(leader) == [_NO_WAIT, None]


def test_buffered_responses_count_as_retiring_requests() -> None:
    """The routing exchange carries buffered responses as retiring requests."""
    group = _TpGroup(size=1)
    dist = MagicMock()
    dist.tp_rank, dist.has_cp_helix = 0, False
    dist.tp_allgather.side_effect = partial(group.allgather, 0)
    router = DefaultADPRouter(dist)

    [state] = router.gather_all_rank_states([], num_buffered_responses=2)

    assert state == RankState(rank=0, num_retiring_requests=2)
