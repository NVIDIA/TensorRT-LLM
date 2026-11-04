# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""How the executor consumes the failures of the capturable guided decoder.

`CapturableGuidedDecoder` (one-model speculative decoding) builds its bitmasks
in CUDA host functions and reports failures through a queue, tagged with the
batch they belong to. The executor must consume a batch's failures after it
has synchronized on that batch's sampler event and before the batch's
responses are built. The loop tests run the real executor loops on a shell
with stubbed collaborators and record the order of the calls that matter.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import ScheduledRequests

_ERROR = "Guided decoding error: matcher compile error"


class _FakeCapturableDecoder:
    """Numbers the batches a forward hands it and returns scripted failures.

    `failures` maps a batch number to the `(request_id, error_msg)` list its
    build reports. Like the real decoder, `take_failed_requests` returns only
    the failures of the batches up to the completed one.
    """

    def __init__(self, failures=None):
        self.last_added_batch_id = 0
        self._failures = dict(failures or {})
        self.completed_batch_ids = []

    def add_batch(self):
        self.last_added_batch_id += 1

    def take_failed_requests(self, completed_batch_id):
        self.completed_batch_ids.append(completed_batch_id)
        taken = []
        for batch_id in sorted(self._failures):
            if batch_id <= completed_batch_id:
                taken.extend(self._failures.pop(batch_id))
        return taken


def _request(request_id: int):
    return SimpleNamespace(
        py_request_id=request_id,
        py_batch_idx=None,
        is_attention_dp_dummy=False,
        is_cuda_graph_dummy=False,
    )


def _batch(*requests, batch_no: int = 0) -> ScheduledRequests:
    batch = ScheduledRequests()
    batch.generation_requests = list(requests)
    batch.batch_no = batch_no
    return batch


def _shell(decoder) -> PyExecutor:
    executor = object.__new__(PyExecutor)
    executor.capturable_guided_decoder = decoder
    executor._handle_errors = MagicMock()
    return executor


# --- the handler --------------------------------------------------------------


def test_completed_batch_failures_fail_their_requests():
    decoder = _FakeCapturableDecoder({1: [(7, _ERROR)]})
    executor = _shell(decoder)
    request_7, request_8 = _request(7), _request(8)

    executor._handle_capturable_guided_decoder_errors(_batch(request_7, request_8), 1)

    assert decoder.completed_batch_ids == [1]
    executor._handle_errors.assert_called_once_with(
        _ERROR, requests=[request_7], charge_budget=False
    )


def test_nothing_is_consumed_without_a_decoder_or_a_batch_number():
    executor = _shell(None)
    executor._handle_capturable_guided_decoder_errors(_batch(_request(7)), 1)
    executor._handle_errors.assert_not_called()

    decoder = _FakeCapturableDecoder({1: [(7, _ERROR)]})
    executor = _shell(decoder)
    executor._handle_capturable_guided_decoder_errors(_batch(_request(7)), None)
    assert decoder.completed_batch_ids == []
    executor._handle_errors.assert_not_called()


def test_failure_of_a_request_missing_from_its_batch_fails_nothing():
    decoder = _FakeCapturableDecoder({1: [(99, _ERROR)]})
    executor = _shell(decoder)
    executor._handle_capturable_guided_decoder_errors(_batch(_request(7)), 1)
    executor._handle_errors.assert_not_called()


# --- the executor loops ---------------------------------------------------------


def _loop_executor(monkeypatch, decoder, batches, log):
    """Executor shell that runs a real loop over `batches`.

    Executor-owned helpers are stubbed; the loop body under test is real, as
    are the two guided-decoding error handlers. The calls that matter append
    to `log`: the forward and sampling of a batch,
    the synchronization on its sampler event (`_update_requests`), the
    failure of a request (`_handle_errors`) and the building of a batch's
    responses (`_handle_responses`).
    """
    for target in ("torch.cuda.set_device", "cudart.cudaSetDevice", "CUASSERT"):
        monkeypatch.setattr(f"tensorrt_llm._torch.pyexecutor.py_executor.{target}", Mock())

    executor = object.__new__(PyExecutor)
    executor.device_id = 0
    profiler = MagicMock()
    profiler.__enter__.return_value = Mock()
    executor._profiler = Mock(return_value=profiler)
    executor._profile_enabled = False
    executor._overlap_loop_stream_scope = MagicMock()
    executor.hang_detector = MagicMock()
    executor._step_scope = MagicMock()
    executor.perf_manager = MagicMock()
    executor.perf_manager.enabled = False
    executor.perf_manager.create_timing_events.return_value = (None, None, None)

    executor.enable_iter_perf_stats = False
    executor.enable_attention_dp = False
    executor.gather_all_responses = False
    executor.is_benchmark_disagg = False
    executor.iter_counter = 0
    executor._resource_governor_enabled = False
    executor._is_kv_manager_v2 = False
    executor._mm_encoder_item_scheduling_enabled = False
    executor.enable_early_first_token_response = False
    executor.enable_kv_cache_events = False
    executor.kv_cache_transceiver = None
    executor.kv_connector_manager = None
    executor.guided_decoder = None
    executor.capturable_guided_decoder = decoder
    executor.drafter = None
    executor.draft_model_engine = None
    executor.use_spec_decode = False
    executor.dwdp_manager = None
    executor.speculation_gate = None
    executor.sampler = Mock()
    executor.execution_stream = None
    executor.model_engine = SimpleNamespace(
        attn_metadata=None, kv_cache_dtype_byte_size=None, spec_metadata=None
    )
    # The setter mirrors the flag into the model engine, so it needs one.
    executor.is_warmup = False
    executor.previous_batch = None
    executor._disagg_pp_termination_handler = None
    executor.has_previous_draft_tokens = False
    executor.active_requests = []
    executor.waiting_queue = []
    executor.inflight_req_ids = set()
    executor._transfer_held_failures = {}
    executor.is_shutdown = False
    executor.dist = Mock(tp_size=1, world_size=1)
    executor._disagg_coordinator = MagicMock()  # read through the `disagg` property
    executor.resource_manager = MagicMock()
    executor.resource_manager.resource_managers.get.return_value = None

    schedule = iter([*((batch, None) for batch in batches), (None, None)])
    executor._prepare_and_schedule_batch = lambda: next(schedule)
    executor._can_queue = Mock(return_value=(True, True))
    executor._kv_connector_start_batch = Mock(return_value=(True, True))
    executor._check_benchmark_disagg_gate = Mock(return_value=(True, False))
    current = {"batch_no": None}

    def forward_step(scheduled_batch, *args):
        # The model engine hands every forwarded batch to the decoder.
        decoder.add_batch()
        current["batch_no"] = scheduled_batch.batch_no
        log.append(("forward", scheduled_batch.batch_no))
        return {"logits": object()}

    def sample_async(scheduled_batch, batch_outputs):
        log.append(("sample", scheduled_batch.batch_no))
        return SimpleNamespace(batch_no=scheduled_batch.batch_no, device=None)

    def update_requests(sample_state, *args):
        log.append(("synchronize", sample_state.batch_no))

    def handle_errors(error_msg, *, requests, charge_budget):
        log.append(("fail", tuple(request.py_request_id for request in requests), charge_budget))

    def handle_responses(**kwargs):
        previous = executor.previous_batch
        batch_no = (
            previous.scheduled_requests.batch_no if previous is not None else current["batch_no"]
        )
        log.append(("responses", batch_no))
        return []

    executor._forward_step = forward_step
    executor._sample_async = sample_async
    executor._update_requests = update_requests
    executor._handle_errors = handle_errors
    executor._handle_responses = handle_responses
    for name in (
        "_terminate_requests",
        "_pause_requests",
        "_prepare_disagg_gen_transmission_complete",
        "_handle_dynamic_draft_len",
        "_kv_connector_terminate_requests",
        "_remove_inflight_ids",
        "_finalize_adp_dummy_allocation",
        "_commit_kv_cache_stats",
        "_flush_pending_transfer_responses",
        "_flush_iter_stats_synced",
        "_wait_for_model_engine_input_copy",
        "_enqueue_responses",
        "_update_generation_requests_that_will_complete_next_iteration",
        "_update_request_states",
        "_update_v2_context_resources",
        "_update_v2_generation_resources",
        "_update_batch_acceptance_rate",
        "_send_kv_async",
        "_handle_canceled_requests",
        "_handle_first_token_response",
        "_emit_first_token_responses",
        "_maybe_prefetch_next_iter_mm_encoders",
        "_forward_multimodal_encoder_step",
        "_submit_encoder_step",
        "_run_encoder_step",
        "_revert_gen_alloc",
        "_release_unused_connector_reservations",
        "_terminate_recompute_paused_requests",
        "_pause_recompute_paused_requests",
        "_maybe_record_hang_diagnostic_phase",
        "_maybe_rebalance_kv_pools",
        "_can_pause_for_rebalance",
        "_collect_scheduled_batch_stats",
        "_take_pending_new_active_requests",
        "_add_kv_cache_events",
        "_sync_and_process_resource_governor_queue",
        "_build_step_scope_label",
    ):
        setattr(executor, name, Mock())
    return executor


def _three_batches():
    """The same two generation requests scheduled three iterations in a row."""
    return [_batch(_request(7), _request(8), batch_no=n) for n in (1, 2, 3)]


def _between(log, event, after, before):
    """`event` happens exactly once in `log`, after `after` and before `before`."""
    assert log.count(event) == 1, log
    assert after in log and before in log, log
    assert log.index(after) < log.index(event) < log.index(before), log


@pytest.mark.parametrize("failing_batch", [1, 2])
def test_overlap_loop_fails_the_request_once_its_batch_is_synchronized(monkeypatch, failing_batch):
    """With overlap scheduling, a batch's failure is handled right after that
    batch's sampler event is synchronized, before its responses, and never
    while the batch may still be running."""
    log = []
    decoder = _FakeCapturableDecoder({failing_batch: [(7, _ERROR)]})
    executor = _loop_executor(monkeypatch, decoder, _three_batches(), log)

    PyExecutor._executor_loop_overlap(executor)

    _between(
        log,
        ("fail", (7,), False),
        after=("synchronize", failing_batch),
        before=("responses", failing_batch),
    )
    # Every completed batch is consumed exactly once, in order.
    assert decoder.completed_batch_ids == [1, 2]


def test_executor_loop_fails_the_request_once_its_batch_is_synchronized(monkeypatch):
    """Without overlap scheduling, the batch's failure is handled after its
    synchronization and before its responses, in the same iteration."""
    log = []
    decoder = _FakeCapturableDecoder({1: [(7, _ERROR)]})
    executor = _loop_executor(monkeypatch, decoder, _three_batches()[:2], log)

    PyExecutor._executor_loop(executor)

    _between(log, ("fail", (7,), False), after=("synchronize", 1), before=("responses", 1))
    assert log.index(("responses", 1)) < log.index(("forward", 2))
    assert decoder.completed_batch_ids == [1, 2]


def test_pp_executed_batch_fails_the_request_once_its_batch_is_synchronized(monkeypatch):
    """With pipeline parallelism, the last rank handles an executed batch's
    failure after synchronizing on that batch and before its responses."""
    log = []
    decoder = _FakeCapturableDecoder({1: [(7, _ERROR)]})
    batch = _batch(_request(7), _request(8), batch_no=1)
    executor = _loop_executor(monkeypatch, decoder, [batch], log)
    executor._handle_responses = lambda **kwargs: log.append(("responses", 1)) or []
    executed_batch = SimpleNamespace(
        scheduled_requests=batch,
        sample_state=SimpleNamespace(batch_no=1, device=None),
        guided_batch_id=1,
        microbatch_id=0,
    )

    PyExecutor._handle_executed_batch(executor, executed_batch)

    _between(log, ("fail", (7,), False), after=("synchronize", 1), before=("responses", 1))
    assert decoder.completed_batch_ids == [1]


def test_overlap_loop_withholds_the_first_token_of_a_failed_request(monkeypatch):
    """With early first-token responses, the failed request of the previous
    batch gets no first-token response; its error follows, before the batch's
    responses, once the batch is synchronized."""
    log = []
    decoder = _FakeCapturableDecoder({1: [(7, _ERROR)]})
    executor = _loop_executor(monkeypatch, decoder, _three_batches()[:2], log)
    executor.enable_early_first_token_response = True

    def emit_first_token_responses(batch, skip_request_ids=frozenset()):
        log.append(("first_tokens", batch.batch_no, frozenset(skip_request_ids)))

    executor._emit_first_token_responses = emit_first_token_responses

    PyExecutor._executor_loop_overlap(executor)

    first_tokens = ("first_tokens", 1, frozenset({7}))
    assert log.count(first_tokens) == 1, log
    assert log.index(("synchronize", 1)) < log.index(first_tokens), log
    _between(log, ("fail", (7,), False), after=first_tokens, before=("responses", 1))
