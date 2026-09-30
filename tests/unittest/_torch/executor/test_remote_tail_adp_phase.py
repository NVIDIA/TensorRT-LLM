# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host contracts for uniform conditional-disaggregation execution across ADP ranks."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from _torch.executor.kv_cache.test_kv_cache_v2_scheduler import (
    make_ctx_request,
    make_disagg_request,
    make_filtered_request,
    make_gen_request,
    make_kv_cache_manager,
    make_scheduler,
)

from tensorrt_llm._torch.pyexecutor import py_executor as executor_module
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState, SamplingConfig
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import RankState
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import RemoteTailPhase, ScheduledRequests

pytestmark = pytest.mark.cpu_only


def _make_executor() -> PyExecutor:
    executor = PyExecutor.__new__(PyExecutor)
    executor.inflight_req_ids = set()
    executor._remote_tail_phase = RemoteTailPhase.NONE
    executor._last_remote_tail_phase = RemoteTailPhase.DECODE
    executor.model_engine = SimpleNamespace(
        cuda_graph_runner=SimpleNamespace(padding_dummy_requests={0: object()})
    )
    executor.scheduler = Mock()
    executor.scheduler.is_request_in_schedulable_state.side_effect = (
        lambda request: request.state
        in (
            LlmRequestState.ENCODER_INIT,
            LlmRequestState.CONTEXT_INIT,
            LlmRequestState.GENERATION_IN_PROGRESS,
        )
    )
    return executor


def _phase_request(
    request_id: int, state: LlmRequestState, *, dummy: bool = False
) -> SimpleNamespace:
    return SimpleNamespace(request_id=request_id, state=state, is_dummy=dummy)


def _make_generation_request(request_id: int) -> LlmRequest:
    # Real state predicates are essential to exercising the deadlock guard.
    request = LlmRequest(
        request_id=request_id,
        max_new_tokens=16,
        input_tokens=[1, 2],
        sampling_config=SamplingConfig(1),
        is_streaming=False,
    )
    request.state = LlmRequestState.GENERATION_IN_PROGRESS
    assert request.is_generation_in_progress_state
    assert not request.is_generation_to_complete_state
    return request


@pytest.mark.parametrize("phase", list(RemoteTailPhase))
def test_deferred_generation_waits_for_next_phase_vote(phase: RemoteTailPhase) -> None:
    manager = make_kv_cache_manager()
    scheduler = make_scheduler(manager)
    generation = _make_generation_request(1)
    generation.py_batch_idx = 7
    # CONTEXT/NONE defer existing decode work; DECODE defers a late arrival.
    frozen_generation_ids = frozenset() if phase is RemoteTailPhase.DECODE else frozenset({1})
    scheduler.set_remote_tail_phase(phase, frozenset(), frozen_generation_ids)

    deferred = scheduler.schedule_request([generation], set())

    assert not deferred.context_requests
    assert not deferred.generation_requests
    assert generation.py_batch_idx is None
    manager.try_allocate_generation.assert_not_called()
    manager.suspend_request.assert_not_called()

    scheduler.set_remote_tail_phase(RemoteTailPhase.DECODE, frozenset(), frozenset({1}))
    resumed = scheduler.schedule_request([generation], set())

    assert resumed.generation_requests == [generation]
    manager.try_allocate_generation.assert_called_once_with(generation)


@pytest.mark.parametrize("phase", [None, RemoteTailPhase.DECODE])
@pytest.mark.parametrize("progress", ["inflight_request", "inflight_peer", "complete"])
def test_deadlock_guard_preserves_inflight_and_complete_exclusions(
    phase: RemoteTailPhase | None, progress: str
) -> None:
    manager = make_kv_cache_manager(try_allocate_generation_fn=lambda request: False)
    scheduler = make_scheduler(manager, enable_recompute_pause=False)
    generation = _make_generation_request(1)
    manager.kv_cache_map[generation.py_request_id].is_active = False
    if phase is not None:
        scheduler.set_remote_tail_phase(phase, frozenset(), frozenset({1}))
    inflight_ids = set()
    if progress == "complete":
        generation.state = LlmRequestState.GENERATION_TO_COMPLETE
    else:
        inflight_ids.add(1 if progress == "inflight_request" else 2)

    result = scheduler.schedule_request([generation], inflight_ids)

    assert not result.context_requests
    assert not result.generation_requests
    if progress == "inflight_peer":
        manager.try_allocate_generation.assert_called_once_with(generation)
    else:
        manager.try_allocate_generation.assert_not_called()
    manager.suspend_request.assert_not_called()


@pytest.mark.parametrize("phase", [None, RemoteTailPhase.DECODE])
def test_eligible_generation_kv_exhaustion_still_raises(phase: RemoteTailPhase | None) -> None:
    manager = make_kv_cache_manager(try_allocate_generation_fn=lambda request: False)
    scheduler = make_scheduler(manager, enable_recompute_pause=False)
    generation = _make_generation_request(1)
    # Already suspended: allocation cannot succeed, and self-eviction cannot
    # release any more pages. No in-flight work can free capacity either.
    manager.kv_cache_map[generation.py_request_id].is_active = False
    active = [generation]
    if phase is not None:
        scheduler.set_remote_tail_phase(phase, frozenset(), frozenset({1}))
        # The late arrival must neither mask the deadlock nor inflate its count.
        active.append(_make_generation_request(2))

    with pytest.raises(RuntimeError, match=r"V2 scheduler deadlock: 1 generation request\(s\)"):
        scheduler.schedule_request(active, set())

    manager.try_allocate_generation.assert_called_once_with(generation)
    manager.suspend_request.assert_not_called()


@pytest.mark.parametrize("phase", list(RemoteTailPhase))
def test_frozen_phase_filters_before_kv_work_and_keeps_transfer_admission(
    phase: RemoteTailPhase,
) -> None:
    manager = make_kv_cache_manager()
    scheduler = make_scheduler(manager, max_num_tokens=128)
    context = make_ctx_request(1, context_remaining_length=128)
    context.py_csa2_remote_tail_mode = "destination"
    generation = _make_generation_request(2)
    generation.py_batch_idx = 7
    late_context = make_ctx_request(3, context_remaining_length=128)
    late_context.py_csa2_remote_tail_mode = "destination"
    late_generation = _make_generation_request(4)
    late_generation.py_batch_idx = 8
    transfer = make_disagg_request(5)
    terminal = make_filtered_request(6, LlmRequestState.GENERATION_TO_COMPLETE.value)
    scheduler.set_remote_tail_phase(phase, frozenset({1}), frozenset({2}))

    result = scheduler.schedule_request(
        [late_context, late_generation, context, generation, transfer, terminal], set()
    )

    expected_contexts = [context] if phase is RemoteTailPhase.CONTEXT else []
    expected_generations = [generation] if phase is RemoteTailPhase.DECODE else []
    assert result.context_requests == expected_contexts
    assert result.generation_requests == expected_generations
    assert result.fitting_disagg_gen_init_requests == [transfer]
    manager.prepare_disagg_gen_init.assert_called_once_with(transfer)
    deferred = [late_context, late_generation]
    if not expected_contexts:
        deferred.append(context)
    if not expected_generations:
        deferred.append(generation)
        assert generation.py_batch_idx is None
    assert late_generation.py_batch_idx is None
    for cache_call in manager.mock_calls:
        assert not any(argument is request for argument in cache_call.args for request in deferred)


def test_category_changes_and_late_transfer_activation_wait_for_next_vote() -> None:
    manager = make_kv_cache_manager()
    scheduler = make_scheduler(manager)
    executor = _make_executor()
    ready_context = _phase_request(1, LlmRequestState.CONTEXT_INIT)
    ready_generation = _phase_request(2, LlmRequestState.GENERATION_IN_PROGRESS)
    transferring = _phase_request(3, LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS)
    context_ids, generation_ids = executor._snapshot_remote_tail_phase_work(
        [ready_context, ready_generation, transferring]
    )
    assert context_ids == frozenset({1})
    assert generation_ids == frozenset({2})
    # A request may change category or finish its transfer after the collective.
    active = [
        _make_generation_request(1),
        make_ctx_request(2, context_remaining_length=128),
        make_ctx_request(3, context_remaining_length=128),
    ]
    for phase in (RemoteTailPhase.CONTEXT, RemoteTailPhase.DECODE):
        scheduler.set_remote_tail_phase(phase, context_ids, generation_ids)
        result = scheduler.schedule_request(active, set())
        assert not result.context_requests
        assert not result.generation_requests
    assert not manager.mock_calls


def test_disabled_phase_preserves_existing_request_selection() -> None:
    scheduler = make_scheduler(make_kv_cache_manager())
    requests = [make_ctx_request(1, 128), make_gen_request(2)]
    assert scheduler._filter_remote_tail_phase_requests(requests) is requests


def test_phase_snapshot_excludes_inflight_dummy_and_unready_requests() -> None:
    executor = _make_executor()
    executor.inflight_req_ids = {4}
    requests = [
        _phase_request(1, LlmRequestState.CONTEXT_INIT),
        _phase_request(2, LlmRequestState.ENCODER_INIT),
        _phase_request(3, LlmRequestState.GENERATION_IN_PROGRESS),
        _phase_request(4, LlmRequestState.CONTEXT_INIT),
        _phase_request(5, LlmRequestState.GENERATION_IN_PROGRESS, dummy=True),
        _phase_request(6, LlmRequestState.DISAGG_GENERATION_INIT),
        _phase_request(7, LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS),
        _phase_request(8, LlmRequestState.GENERATION_TO_COMPLETE),
    ]
    assert executor._snapshot_remote_tail_phase_work(requests) == (
        frozenset({1, 2}),
        frozenset({3}),
    )


def test_phase_consensus_alternates_busy_categories_including_idle_ranks() -> None:
    executors = [_make_executor() for _ in range(3)]
    states = [
        RankState(rank=0, num_context_requests=2),
        RankState(rank=1, num_generation_requests=3),
        RankState(rank=2),
    ]
    local_ids = [
        (frozenset({1, 2}), frozenset()),
        (frozenset(), frozenset({3, 4, 5})),
        (frozenset(), frozenset()),
    ]
    for expected in (
        RemoteTailPhase.CONTEXT,
        RemoteTailPhase.DECODE,
        RemoteTailPhase.CONTEXT,
        RemoteTailPhase.DECODE,
    ):
        for executor, (context_ids, generation_ids) in zip(executors, local_ids):
            executor._select_remote_tail_phase(states, context_ids, generation_ids)
            assert executor._remote_tail_phase is expected
            executor.scheduler.set_remote_tail_phase.assert_called_with(
                expected, context_ids, generation_ids
            )
    for executor in executors:
        executor._select_remote_tail_phase([RankState(rank=0)], frozenset(), frozenset())
        assert executor._remote_tail_phase is RemoteTailPhase.NONE
        assert executor._last_remote_tail_phase is RemoteTailPhase.DECODE


@pytest.mark.parametrize(
    "contexts,generations,expected",
    [(1, 0, RemoteTailPhase.CONTEXT), (0, 1, RemoteTailPhase.DECODE), (0, 0, RemoteTailPhase.NONE)],
)
def test_phase_consensus_uses_ready_categories_not_total_active_requests(
    contexts: int,
    generations: int,
    expected: RemoteTailPhase,
) -> None:
    executor = _make_executor()
    executor._select_remote_tail_phase(
        [
            RankState(
                rank=0,
                num_active_requests=100,
                num_context_requests=contexts,
                num_generation_requests=generations,
            )
        ],
        frozenset(),
        frozenset(),
    )
    assert executor._remote_tail_phase is expected


@pytest.mark.parametrize(
    "sizes,idle,expected",
    [
        ([[2, 2], [1, 0]], True, (True, False)),
        ([[2, 2], [1, 0]], False, (True, True)),
        ([[1, 0], [1, 0]], True, (False, False)),
        ([[2, 2], [0, 0]], False, (False, True)),
    ],
)
def test_queue_requires_execution_rows_and_some_uncancelled_semantic_work(
    sizes: list[list[int]],
    idle: bool,
    expected: tuple[bool, bool],
) -> None:
    executor = _make_executor()
    executor._enable_remote_tail_adp = True
    executor.dist = Mock()
    executor.dist.tp_allgather_int64.return_value = torch.tensor(sizes)
    batch = ScheduledRequests()
    batch.attention_dp_phase = RemoteTailPhase.DECODE
    batch.is_attention_dp_phase_idle = idle
    if not idle:
        batch.generation_requests = [object(), object()]

    assert executor._can_queue(batch) == expected
    executor.dist.tp_allgather_int64.assert_called_once_with([1, 0] if idle else [2, 2])


@pytest.mark.parametrize("local_reservation_lost", [False, True])
def test_queue_reports_lost_execution_reservation_collectively(
    local_reservation_lost: bool,
) -> None:
    executor = _make_executor()
    executor._enable_remote_tail_adp = True
    executor.dist = Mock()
    if local_reservation_lost:
        executor.model_engine.cuda_graph_runner.padding_dummy_requests.clear()
    executor.dist.tp_allgather_int64.return_value = torch.tensor([[1, 0], [-1, 2]])
    batch = ScheduledRequests()
    batch.is_attention_dp_phase_idle = True

    with pytest.raises(RuntimeError, match="at least one rank"):
        executor._can_queue(batch)
    executor.dist.tp_allgather_int64.assert_called_once_with(
        [-1, 0] if local_reservation_lost else [1, 0]
    )


@pytest.mark.parametrize("reservation_survives", [False, True])
def test_rebalance_reserves_execution_row_before_resuming_real_requests(
    reservation_survives: bool,
) -> None:
    executor = _make_executor()
    executor._enable_remote_tail_adp = True
    executor.resource_manager = object()
    real = SimpleNamespace(py_request_id=1)
    dummy = SimpleNamespace(py_request_id=2)
    executor.active_requests = [real]
    runner = Mock()
    runner.padding_dummy_requests = {0: dummy}
    executor.model_engine.cuda_graph_runner = runner
    manager = Mock()
    executor.kv_cache_manager = manager
    active_ids = {1, 2}
    resume_order = []
    manager.is_request_active.side_effect = lambda request_id: request_id in active_ids
    manager.suspend_request.side_effect = lambda request: active_ids.remove(request.py_request_id)

    def resume(request: SimpleNamespace) -> bool:
        resume_order.append(request.py_request_id)
        if request is dummy and not reservation_survives:
            return False
        active_ids.add(request.py_request_id)
        return True

    def adjust() -> None:
        assert not active_ids

    manager.resume_request.side_effect = resume
    manager.impl.adjust.side_effect = adjust
    executor.dist = Mock()
    executor.dist.tp_allgather.return_value = [reservation_survives, True]
    if reservation_survives:
        executor._rebalance_kv_pools_now()
        assert resume_order == [2, 1]
        assert active_ids == {1, 2}
    else:
        with pytest.raises(RuntimeError, match="execution row on every rank"):
            executor._rebalance_kv_pools_now()
        assert resume_order == [2]
        assert not active_ids
    manager.impl.adjust.assert_called_once()
    executor.dist.tp_allgather.assert_called_once_with(reservation_survives)
    runner.release_padding_dummy.assert_not_called()
    assert runner.padding_dummy_requests[0] is dummy


@pytest.mark.parametrize("phase", [RemoteTailPhase.CONTEXT, RemoteTailPhase.DECODE])
def test_idle_execution_reuses_retained_row_without_changing_semantic_batch(
    phase: RemoteTailPhase,
) -> None:
    dummy = object()
    runner = SimpleNamespace(padding_dummy_requests={0: dummy})
    engine = SimpleNamespace(cuda_graph_runner=runner)
    semantic_batch = ScheduledRequests()
    semantic_batch.attention_dp_phase = phase
    semantic_batch.is_attention_dp_phase_idle = True

    for _ in range(2):
        execution_batch = PyTorchModelEngine._get_attention_dp_execution_batch(
            engine, semantic_batch
        )
        assert execution_batch is not semantic_batch
        assert execution_batch.generation_requests == [dummy]
        assert execution_batch.attention_dp_phase is phase
        assert execution_batch.is_attention_dp_phase_idle
        assert execution_batch.can_run_cuda_graph is (phase is RemoteTailPhase.DECODE)
        assert semantic_batch.batch_size == 0
        assert not semantic_batch.added_inflight_req_ids
    assert runner.padding_dummy_requests == {0: dummy}


def test_idle_execution_refuses_missing_row_or_nonempty_semantic_batch() -> None:
    engine = SimpleNamespace(cuda_graph_runner=SimpleNamespace(padding_dummy_requests={}))
    batch = ScheduledRequests()
    batch.is_attention_dp_phase_idle = True
    with pytest.raises(RuntimeError, match="execution row is missing"):
        PyTorchModelEngine._get_attention_dp_execution_batch(engine, batch)
    for bucket in (batch.generation_requests, batch.encoder_requests):
        bucket.append(object())
        with pytest.raises(ValueError, match="semantically empty"):
            PyTorchModelEngine._get_attention_dp_execution_batch(engine, batch)
        bucket.clear()


@pytest.mark.parametrize("idle", [False, True])
def test_semantic_outputs_hide_execution_row_and_preserve_retained_graph_outputs(
    idle: bool,
) -> None:
    batch = ScheduledRequests()
    batch.is_attention_dp_phase_idle = idle
    logits = torch.arange(7, dtype=torch.float32).reshape(1, 7)
    outputs = {"logits": logits, "hidden_states": object()}
    restored = PyTorchModelEngine._restore_attention_dp_semantic_outputs(batch, outputs)
    assert restored["logits"].shape == ((0, 7) if idle else (1, 7))
    assert outputs["logits"] is logits
    assert outputs["logits"].shape == (1, 7)
    assert restored["hidden_states"] is outputs["hidden_states"]
    assert (restored is outputs) is not idle
    if not idle:
        assert PyTorchModelEngine._get_attention_dp_execution_batch(object(), batch) is batch


def test_context_phase_disables_graph_for_real_work_and_idle_rows() -> None:
    batch = ScheduledRequests()
    assert batch.attention_dp_phase is None
    assert not batch.is_attention_dp_phase_idle
    assert batch.can_run_cuda_graph
    batch.context_requests_last_chunk = [object()]
    assert not batch.can_run_cuda_graph
    batch.context_requests_last_chunk.clear()
    batch.generation_requests = [object()]
    batch.attention_dp_phase = RemoteTailPhase.CONTEXT
    assert not batch.can_run_cuda_graph
    batch.attention_dp_phase = RemoteTailPhase.DECODE
    assert batch.can_run_cuda_graph


@pytest.mark.parametrize("graph_flags", [[1, 1], [0, 1], [1, 0]])
def test_decode_graph_requirement_uses_token_count_collective(
    graph_flags: list[int],
) -> None:
    dist = Mock()
    dist.tp_allgather_int64.return_value = torch.tensor(
        [[16, graph_flags[0]], [16, graph_flags[1]]]
    )
    engine = SimpleNamespace(dist=dist)
    metadata = SimpleNamespace(num_tokens=16)
    if all(graph_flags):
        assert PyTorchModelEngine._get_remote_tail_decode_rank_tokens(
            engine, metadata, bool(graph_flags[0])
        ) == [16, 16]
    else:
        with pytest.raises(RuntimeError, match="at least one rank missed its graph"):
            PyTorchModelEngine._get_remote_tail_decode_rank_tokens(
                engine, metadata, bool(graph_flags[0])
            )
    dist.tp_allgather_int64.assert_called_once_with([16, graph_flags[0]])


def _make_startup_executor() -> PyExecutor:
    executor = _make_executor()
    executor._enable_remote_tail_adp = False
    executor.enable_attention_dp = True
    executor.kv_cache_transceiver = object()
    executor.kv_cache_manager = SimpleNamespace(is_estimating_kv_cache=False)
    executor.resource_manager = object()
    executor.max_beam_width = 1
    executor.dist = SimpleNamespace(pp_size=1, cp_size=1)
    executor.scheduler = make_scheduler(make_kv_cache_manager())
    executor.model_engine.model = SimpleNamespace(
        model=SimpleNamespace(disagg_remote_tail_replay=True, disagg_context_only=False)
    )
    executor.model_engine.spec_config = None
    dummy = object()
    executor.model_engine.cuda_graph_runner = SimpleNamespace(
        enabled=True,
        padding_enabled=True,
        padding_dummy_requests={0: dummy},
        _get_or_create_padding_dummy=Mock(return_value=dummy),
    )
    return executor


@pytest.mark.parametrize("has_transceiver", [False, True])
@pytest.mark.parametrize("context_only", [False, True])
def test_constructor_initializes_remote_tail_adp_before_starting_worker(
    monkeypatch: pytest.MonkeyPatch, has_transceiver: bool, context_only: bool
) -> None:
    """Exercise constructor ordering without pre-populating executor attributes."""
    events = []
    in_execution_stream = False
    stream = SimpleNamespace(wait_stream=Mock())

    @contextmanager
    def stream_scope(_stream):
        nonlocal in_execution_stream
        in_execution_stream = True
        try:
            yield
        finally:
            in_execution_stream = False

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "Stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "Event", Mock())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "stream", stream_scope)
    monkeypatch.setattr(executor_module, "_distributed_warmup_guard", lambda *_: nullcontext())
    monkeypatch.setattr(executor_module, "ExecutorRequestQueue", Mock())
    monkeypatch.setattr(executor_module, "mpi_disabled", lambda: True)
    monkeypatch.setattr(PyExecutor, "_set_global_steady_clock_offset", lambda _self: None)
    monkeypatch.setattr(PyExecutor, "_emit_initial_stats", lambda _self: None)
    monkeypatch.setenv("TRTLLM_GPU_KEEPALIVE", "0")

    manager = SimpleNamespace(
        is_estimating_kv_cache=False,
        event_buffer_max_size=0,
        enable_block_reuse=False,
        tokens_per_block=128,
    )
    resource_manager = SimpleNamespace(
        resource_managers={ResourceManagerType.KV_CACHE_MANAGER: manager}
    )
    scheduler = make_scheduler(make_kv_cache_manager())
    transceiver = SimpleNamespace(get_status_dump=Mock()) if has_transceiver else None
    dist = SimpleNamespace(
        rank=0, tp_size=4, pp_size=1, cp_size=1, mapping=object(), barrier=Mock()
    )
    dummy = object()
    runner = SimpleNamespace(enabled=False, padding_enabled=True, padding_dummy_requests={})

    def warmup(resources):
        assert resources is resource_manager and in_execution_stream
        events.append("warmup")
        runner.enabled = True
        runner.padding_dummy_requests[0] = dummy

    def reserve(resources, draft_len):
        assert resources is resource_manager and draft_len == 0
        assert events == ["warmup"] and in_execution_stream
        events.append("reserve")
        return runner.padding_dummy_requests[0]

    runner._get_or_create_padding_dummy = Mock(side_effect=reserve)
    engine = SimpleNamespace(
        enable_attention_dp=True,
        spec_config=None,
        cuda_graph_runner=runner,
        model=SimpleNamespace(
            model=SimpleNamespace(
                disagg_remote_tail_replay=True,
                disagg_context_only=context_only,
            )
        ),
        llm_args=SimpleNamespace(
            max_stats_len=1,
            max_num_tokens=32,
            print_iter_log=False,
            enable_iter_perf_stats=False,
            enable_iter_req_stats=False,
            stream_interval=1,
            attention_dp_config=None,
            batch_wait_timeout_ms=0,
            batch_wait_timeout_iters=0,
            batch_wait_max_tokens_ratio=0,
        ),
        get_max_num_sequences=lambda: 8,
        warmup=warmup,
    )
    should_enable = has_transceiver and not context_only

    def start_worker(executor):
        assert executor.kv_cache_transceiver is transceiver
        assert executor.scheduler is scheduler and executor.adp_router.dist is dist
        assert executor.kv_cache_manager is manager
        assert executor._enable_remote_tail_adp is should_enable
        assert not executor.is_warmup and not in_execution_stream
        assert executor.active_requests == []
        assert runner.padding_dummy_requests[0] is dummy
        events.append("start_worker")

    monkeypatch.setattr(PyExecutor, "start_worker", start_worker)
    executor = PyExecutor(
        resource_manager,
        scheduler,
        engine,
        SimpleNamespace(),
        dist,
        max_num_sequences=8,
        kv_cache_transceiver=transceiver,
    )
    assert executor.kv_cache_transceiver is transceiver
    assert events == (
        ["warmup", "reserve", "start_worker"] if should_enable else ["warmup", "start_worker"]
    )
    assert runner._get_or_create_padding_dummy.call_count == int(should_enable)


def test_final_startup_reserves_idle_row_without_admitting_a_request() -> None:
    executor = _make_startup_executor()
    executor.active_requests = []
    executor._initialize_remote_tail_adp()
    assert executor._enable_remote_tail_adp
    assert executor.active_requests == []
    executor.model_engine.cuda_graph_runner._get_or_create_padding_dummy.assert_called_once_with(
        executor.resource_manager, 0
    )


def test_estimation_does_not_reserve_idle_execution_resources() -> None:
    executor = _make_startup_executor()
    executor.kv_cache_manager.is_estimating_kv_cache = True
    executor._initialize_remote_tail_adp()
    assert not executor._enable_remote_tail_adp
    executor.model_engine.cuda_graph_runner._get_or_create_padding_dummy.assert_not_called()


@pytest.mark.parametrize("unsupported", ["graphs", "padding", "beam", "speculation", "scheduler"])
def test_startup_rejects_unsupported_execution_contracts(unsupported: str) -> None:
    executor = _make_startup_executor()
    runner = executor.model_engine.cuda_graph_runner
    if unsupported == "graphs":
        runner.enabled = False
    elif unsupported == "padding":
        runner.padding_enabled = False
    elif unsupported == "beam":
        executor.max_beam_width = 2
    elif unsupported == "speculation":
        executor.model_engine.spec_config = object()
    else:
        executor.scheduler = object()
    with pytest.raises(ValueError, match="padded decode CUDA graphs"):
        executor._initialize_remote_tail_adp()
    assert not executor._enable_remote_tail_adp
    runner._get_or_create_padding_dummy.assert_not_called()


def test_startup_reports_missing_idle_reservation() -> None:
    executor = _make_startup_executor()
    executor.model_engine.cuda_graph_runner._get_or_create_padding_dummy.return_value = None
    with pytest.raises(RuntimeError, match="padding request at startup"):
        executor._initialize_remote_tail_adp()
    assert not executor._enable_remote_tail_adp


def test_rank_state_appends_forward_counts_without_changing_legacy_stats() -> None:
    state = RankState(
        rank=3, num_active_requests=8, num_context_requests=2, num_generation_requests=4
    )
    state.iter_stats.has_iter_stats = 1
    state.iter_stats.iter_stats_iter = 9
    state.iter_stats.num_context_requests = 7
    payload = state.serialize()
    assert RankState.deserialize(payload) == state
    legacy = RankState.deserialize(payload[:-2])
    assert legacy.iter_stats == state.iter_stats
    assert legacy.num_context_requests == legacy.num_generation_requests == 0
    short_legacy = RankState.deserialize([3, 8])
    assert short_legacy.num_context_requests == short_legacy.num_generation_requests == 0
