# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Forward-failure survival and multimodal admission validation.

``_forward_step`` returns None after a non-fatal forward exception:
``_handle_errors`` has already failed the affected requests and enqueued
their error responses. These tests pin the two layers that make such an
iteration survivable:

- the executor loops handle a None forward result — the overlap and
  non-overlap loops skip the iteration's sampling and bookkeeping and keep
  running, while the PP loop raises an explicit fatal on the last PP rank
  (inter-PP peers cannot skip a microbatch in lockstep);
- ``PyExecutor._validate_request`` calls the model-provided
  ``validate_multimodal_request_data`` hook so a model can reject a bad
  multimodal payload at admission, failing only that request instead of
  the whole scheduled batch at forward time.

All CUDA/MPI machinery is bypassed via object.__new__ + attribute
injection; the loop bodies and the ``_validate_request`` call site are
real code.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from tensorrt_llm._torch.models.modeling_multimodal_mixin import MultimodalModelMixin
from tensorrt_llm._torch.pyexecutor.py_executor import BatchState, PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import ScheduledRequests

pytestmark = pytest.mark.cpu_only


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _AcceptingModel(MultimodalModelMixin):
    """Model that relies on the mixin's default (accept-everything) hook."""


class _RejectingModel(MultimodalModelMixin):
    def validate_multimodal_request_data(self, mm_data: dict) -> None:
        raise ValueError("multimodal payload exceeds the encoder budget")


def _request(mm_data=None):
    return SimpleNamespace(
        py_disaggregated_params=None,
        sampling_config=None,
        py_multimodal_data=mm_data,
    )


def _validating_executor(model):
    """Executor shell exposing only what _validate_request touches."""
    executor = object.__new__(PyExecutor)
    executor.kv_cache_transceiver = None
    executor.model_engine = SimpleNamespace(model=model)
    executor.sampler = Mock()
    executor._validate_token_id_range = Mock()
    executor._validate_request_budget = Mock()
    return executor


def _scheduled_batch():
    batch = ScheduledRequests()
    batch.generation_requests = [SimpleNamespace(py_batch_idx=0, is_attention_dp_dummy=False)]
    return batch


def _loop_executor(monkeypatch):
    """Executor shell whose one scheduled batch fails its forward.

    The first iteration schedules a batch whose forward returns None; the
    second observes shutdown. Executor-owned helpers are stubbed; the loop
    body under test is real.
    """
    for target in ("torch.cuda.set_device", "cudart.cudaSetDevice", "CUASSERT"):
        monkeypatch.setattr(f"tensorrt_llm._torch.pyexecutor.py_executor.{target}", Mock())

    executor = object.__new__(PyExecutor)
    executor.device_id = 0
    profiler = MagicMock()
    profiler.__enter__.return_value = Mock()
    executor._profiler = Mock(return_value=profiler)
    executor.hang_detector = MagicMock()
    executor._step_scope = MagicMock()
    executor.perf_manager = MagicMock()
    executor.perf_manager.enabled = False
    executor.perf_manager.create_timing_events.return_value = (None, None, None)

    executor.enable_iter_perf_stats = False
    executor.enable_attention_dp = False
    executor.is_benchmark_disagg = False
    executor.iter_counter = 0
    executor._resource_governor_enabled = False
    executor._is_kv_manager_v2 = False
    executor._mm_encoder_item_scheduling_enabled = False
    executor.enable_early_first_token_response = False
    executor.kv_cache_transceiver = None
    executor.kv_connector_manager = None
    executor.guided_decoder = None
    executor.drafter = None
    executor.dwdp_manager = None
    executor.speculation_gate = None
    executor.model_engine = None
    executor.previous_batch = None
    executor.has_previous_draft_tokens = False
    executor.enable_kv_cache_events = False
    executor.active_requests = []
    executor.waiting_queue = []
    executor.inflight_req_ids = set()
    executor.is_shutdown = False
    executor.dist = Mock(tp_size=1, world_size=1)

    executor._disagg_coordinator = MagicMock()
    executor.resource_manager = MagicMock()

    batches = iter([(_scheduled_batch(), None), (None, None)])
    executor._prepare_and_schedule_batch = lambda: next(batches)
    executor._can_queue = Mock(return_value=(True, True))
    executor._check_benchmark_disagg_gate = Mock(return_value=(True, False))
    executor._forward_step = Mock(return_value=None)
    for name in (
        "_terminate_requests",
        "_pause_requests",
        "_prepare_disagg_gen_transmission_complete",
        "_handle_dynamic_draft_len",
        "_kv_connector_start_batch",
        "_kv_connector_terminate_requests",
        "_finalize_adp_dummy_allocation",
        "_commit_kv_cache_stats",
        "_flush_pending_transfer_responses",
        "_flush_iter_stats_synced",
        "_wait_for_model_engine_input_copy",
        "_enqueue_responses",
        "_update_generation_requests_that_will_complete_next_iteration",
        "_sample_async",
        "_update_request_states",
        "_update_requests",
        "_send_kv_async",
        "_handle_canceled_requests",
        "_handle_responses",
        "_handle_guided_decoder_errors",
        "_maybe_prefetch_next_iter_mm_encoders",
        "_submit_encoder_step",
        "_run_encoder_step",
        "_revert_gen_alloc",
    ):
        setattr(executor, name, Mock())
    return executor


# ---------------------------------------------------------------------------
# The mixin hook contract
# ---------------------------------------------------------------------------


def test_mixin_hook_accepts_everything_by_default():
    assert _AcceptingModel().validate_multimodal_request_data({"image": object()}) is None


def test_mixin_hook_override_can_reject():
    with pytest.raises(ValueError, match="encoder budget"):
        _RejectingModel().validate_multimodal_request_data({"image": object()})


# ---------------------------------------------------------------------------
# The _validate_request call site
# ---------------------------------------------------------------------------


def test_validate_request_rejects_via_model_hook():
    executor = _validating_executor(_RejectingModel())
    with pytest.raises(ValueError, match="encoder budget"):
        PyExecutor._validate_request(executor, _request(mm_data={"image": object()}))


def test_validate_request_accepts_when_model_hook_accepts():
    executor = _validating_executor(_AcceptingModel())
    PyExecutor._validate_request(executor, _request(mm_data={"image": object()}))


def test_validate_request_skips_hook_for_text_only_requests():
    model = Mock(spec=["validate_multimodal_request_data"])
    executor = _validating_executor(model)
    PyExecutor._validate_request(executor, _request(mm_data=None))
    model.validate_multimodal_request_data.assert_not_called()


def test_validate_request_accepts_models_without_the_hook():
    executor = _validating_executor(object())
    PyExecutor._validate_request(executor, _request(mm_data={"image": object()}))


# ---------------------------------------------------------------------------
# Loop-level None handling
# ---------------------------------------------------------------------------


def test_executor_loop_survives_a_failed_forward(monkeypatch):
    executor = _loop_executor(monkeypatch)

    PyExecutor._executor_loop(executor)

    executor._forward_step.assert_called_once()
    executor._sample_async.assert_not_called()
    executor._update_request_states.assert_not_called()
    # The loop reached the scheduling-observed shutdown, i.e. it survived the
    # failed-forward iteration instead of crashing on batch_outputs['logits'].
    assert executor._event_loop_completed


def test_executor_loop_overlap_survives_a_failed_forward(monkeypatch):
    executor = _loop_executor(monkeypatch)
    # A previous batch holding timing events borrowed from the perf pool.
    # Dropping it on the failed iteration skips _process_iter_stats, which
    # normally returns them, so the loop must release them itself.
    prev_start, prev_end = object(), object()
    executor.previous_batch = BatchState(
        scheduled_requests=_scheduled_batch(),
        sample_state=None,
        gpu_forward_start_event=prev_start,
        gpu_forward_end_event=prev_end,
        gpu_forward_events_from_perf_pool=True,
    )

    PyExecutor._executor_loop_overlap(executor)

    executor._forward_step.assert_called_once()
    executor._sample_async.assert_not_called()
    # The failed iteration must not become a previous batch for the next
    # iteration's bookkeeping.
    assert executor.previous_batch is None
    # The dropped previous batch's borrowed events went back to the pool
    # (exactly once: the failed iteration itself borrowed none).
    executor.perf_manager.release_forward_timing_events.assert_called_once_with(
        prev_start, prev_end
    )
    assert executor._event_loop_completed


def test_executor_loop_pp_raises_explicit_fatal_on_last_rank(monkeypatch):
    executor = _loop_executor(monkeypatch)
    executor.dist = Mock(
        rank=0,
        pp_rank=1,
        pp_size=2,
        tp_size=1,
        cp_size=1,
        world_size=2,
        is_first_pp_rank=False,
        is_last_pp_rank=True,
    )
    executor.num_micro_batches = 1
    executor.micro_batches = [None]
    executor._pp_rebalance_drain_iters = None
    executor.pp_async_broadcast_sample_state = True
    executor._fetch_and_activate_new_requests = Mock(return_value=[])
    executor._handle_control_request = Mock()
    executor._pad_attention_dp_dummy_request = Mock()
    executor._pp_schedule_and_propagate = Mock(return_value=(_scheduled_batch(), [], None, None))
    executor._add_inflight_ids = Mock()
    executor.num_scheduled_requests = 0

    with pytest.raises(RuntimeError, match="last PP rank"):
        PyExecutor._executor_loop_pp(executor)

    executor._forward_step.assert_called_once()
    executor._sample_async.assert_not_called()


def _response_gather_entries(executor):
    """Response-gather entries a pass makes besides the synced flush.

    Under attention DP both `_handle_responses` (via `_enqueue_responses`)
    and a direct `_enqueue_responses` call perform exactly one tp_gather,
    and collectives pair by call order across ranks.
    """
    return executor._handle_responses.call_count + executor._enqueue_responses.call_count


def test_executor_loop_failed_forward_keeps_response_gather_parity(monkeypatch):
    """A failed-forward pass must enter as many response gathers as a healthy one.

    Under attention DP a rank-local forward failure leaves batch_outputs None
    on one rank only; if that rank skips the `_handle_responses` gather while
    healthy ranks enter it, the TP group desynchronizes and later pairs an
    int payload against a response list (the fail-fast in _enqueue_responses).
    """
    healthy = _loop_executor(monkeypatch)
    healthy._forward_step = Mock(return_value={"logits": Mock()})
    PyExecutor._executor_loop(healthy)
    healthy._forward_step.assert_called_once()

    failed = _loop_executor(monkeypatch)  # its _forward_step returns None
    PyExecutor._executor_loop(failed)
    failed._forward_step.assert_called_once()
    failed._sample_async.assert_not_called()

    healthy_gathers = _response_gather_entries(healthy)
    failed_gathers = _response_gather_entries(failed)
    assert healthy_gathers == failed_gathers
    assert failed_gathers > 0
    # The synced flush is entered once per completed pass on both sides.
    assert (
        healthy._flush_pending_transfer_responses.call_count
        == failed._flush_pending_transfer_responses.call_count
    )


# ---------------------------------------------------------------------------
# First-chunk prefill failure must not publish unwritten V1 reuse blocks
# ---------------------------------------------------------------------------


def _error_executor():
    """Executor shell exposing only what the non-ADP _handle_errors path touches."""
    executor = object.__new__(PyExecutor)
    executor._error_budget = Mock()
    executor._fatal_error = None
    executor.enable_attention_dp = False
    executor.dist = Mock(world_size=1)
    executor._pending_transfer_responses = []
    executor._pending_response_terminations = []
    executor._enqueue_responses = Mock()
    executor._terminate_request = Mock()
    return executor


def _failed_request(request_id, *, remaining, position):
    return SimpleNamespace(
        py_request_id=request_id,
        py_client_id=100 + request_id,
        context_remaining_length=remaining,
        context_current_position=position,
        state=None,
    )


def test_handle_errors_tags_first_chunk_prefill_failures_as_reuse_poisoned():
    executor = _error_executor()
    first_chunk = _failed_request(1, remaining=8, position=0)
    mid_context = _failed_request(2, remaining=4, position=4)
    generation = _failed_request(3, remaining=0, position=8)
    # The disagg coordinator's error path feeds TransferRequest objects
    # (no context_* attributes) through _handle_errors; tagging must not
    # raise on them and must leave them untagged.
    transfer_like = SimpleNamespace(py_request_id=4, py_client_id=104, state=None)
    requests = [first_chunk, mid_context, generation, transfer_like]
    executor.active_requests = list(requests)

    PyExecutor._handle_errors(
        executor,
        "forward failed",
        requests=requests,
        charge_budget=False,
    )

    # Only the request that never completed a context chunk wrote no KV.
    assert getattr(first_chunk, "py_kv_reuse_poisoned", False)
    assert not getattr(mid_context, "py_kv_reuse_poisoned", False)
    assert not getattr(generation, "py_kv_reuse_poisoned", False)
    assert not getattr(transfer_like, "py_kv_reuse_poisoned", False)


def _kv_cache_manager_shell():
    manager = object.__new__(KVCacheManager)
    manager.impl = Mock()
    manager._preprepared_dummy_request_ids = set()
    return manager


def test_free_resources_releases_poisoned_request_without_reuse_store():
    """A poisoned release must advance the context cursor off position 0.

    The remove_sequence binding only accepts (request_id, LlmRequest, bool),
    so the no-store release is achieved by moving the cursor off 0 before
    the call: usable-token accounting then yields zero tokens and the legacy
    position-0 fallback -- which would publish unwritten blocks under the
    prompt's keys -- no longer applies.
    """
    manager = _kv_cache_manager_shell()
    positions_at_call = []
    manager.impl.remove_sequence.side_effect = lambda _req_id, req, _pin: positions_at_call.append(
        req.context_current_position
    )
    request = SimpleNamespace(
        py_request_id=7, py_kv_reuse_poisoned=True, context_current_position=0
    )

    KVCacheManager.free_resources(manager, request)

    # The binding's exact (request_id, LlmRequest, bool) signature is kept.
    manager.impl.remove_sequence.assert_called_once_with(7, request, False)
    assert positions_at_call == [1]


def test_free_resources_keeps_reuse_store_for_untagged_requests():
    manager = _kv_cache_manager_shell()
    request = SimpleNamespace(py_request_id=7, context_current_position=0)

    KVCacheManager.free_resources(manager, request)

    manager.impl.remove_sequence.assert_called_once_with(7, request, False)
    # An untagged request keeps its cursor: a position-0 release with a real
    # completed prefill is the legacy fallback's intended territory.
    assert request.context_current_position == 0
