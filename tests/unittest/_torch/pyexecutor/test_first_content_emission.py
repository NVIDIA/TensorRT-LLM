# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the streaming emission schedule in ``PyExecutor``.

``_should_emit_response`` decides on which decoding iterations a streaming
request gets a response. With ``stream_interval = S`` it emits on iteration 1,
on finish and every S iterations; it also emits on iterations
``2..first_content_emit_iters`` (the model's ``RuntimeHooks.first_content_emit_iters``; 1, the
default, keeps the stock schedule; the MiniMax-M3 sm_107 targets set 2),
so text held back by a reasoning parser after the first token reaches the
client one step later instead of one stream interval later. These tests call
the real method unbound against a minimal fake ``self`` and check that:

* with ``first_content_emit_iters = 1`` (the default) the schedule is the stock
  one: iteration 1, every S iterations, and finish;
* ``first_content_emit_iters = 2`` adds iteration 2 only;
* larger values add iterations ``2..N``;
* a finished request always emits, and ``emit_first_iter=False`` (early
  first-token emission) still suppresses only a non-terminal iteration 1;
* ``stream_interval = 1`` emits every iteration whatever the setting;
* a request with a lead-in of ``L`` tokens (``py_stream_lead_in_tokens``) emits
  every iteration until a response carried more than ``L`` tokens, also for a
  disaggregated generation-only request.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, Mock

import pytest

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks, runtime_hooks_of

pytestmark = pytest.mark.cpu_only


def _fake_self(stream_interval: int, first_content_emit_iters: int):
    return types.SimpleNamespace(
        stream_interval=stream_interval, first_content_emit_iters=first_content_emit_iters
    )


def _request(decoding_iter: int, finished: bool = False):
    return types.SimpleNamespace(
        py_decoding_iter=decoding_iter,
        is_finished=finished,
        py_stream_lead_in_tokens=0,
        py_stream_emitted_tokens=0,
    )


def _emitted_iters(stream_interval, first_content_emit_iters, last=60, emit_first_iter=True):
    fake = _fake_self(stream_interval, first_content_emit_iters)
    return [
        it
        for it in range(1, last + 1)
        if PyExecutor._should_emit_response(fake, _request(it), emit_first_iter)
    ]


def _stock_iters(stream_interval, last=60):
    return [it for it in range(1, last + 1) if it == 1 or it % stream_interval == 0]


@pytest.mark.parametrize("stream_interval", [2, 3, 20])
def test_kill_matches_stock_schedule(stream_interval):
    assert _emitted_iters(stream_interval, 1) == _stock_iters(stream_interval)


@pytest.mark.parametrize("stream_interval", [3, 20])
def test_default_adds_iteration_two_only(stream_interval):
    expected = sorted(set(_stock_iters(stream_interval)) | {2})
    assert _emitted_iters(stream_interval, 2) == expected


def test_larger_window_adds_iterations_up_to_n():
    expected = sorted(set(_stock_iters(20)) | {2, 3, 4, 5})
    assert _emitted_iters(20, 5) == expected


@pytest.mark.parametrize("first_content_emit_iters", [1, 2, 5])
@pytest.mark.parametrize("decoding_iter", [1, 2, 7, 19])
def test_finished_request_always_emits(first_content_emit_iters, decoding_iter):
    fake = _fake_self(20, first_content_emit_iters)
    for emit_first_iter in (True, False):
        assert PyExecutor._should_emit_response(
            fake, _request(decoding_iter, finished=True), emit_first_iter
        )


@pytest.mark.parametrize("first_content_emit_iters", [1, 2, 5])
def test_early_first_token_suppresses_only_iteration_one(first_content_emit_iters):
    with_first = _emitted_iters(20, first_content_emit_iters)
    without_first = _emitted_iters(20, first_content_emit_iters, emit_first_iter=False)
    assert without_first == [it for it in with_first if it != 1]


@pytest.mark.parametrize("first_content_emit_iters", [1, 2, 5])
def test_stream_interval_one_emits_every_iteration(first_content_emit_iters):
    assert _emitted_iters(1, first_content_emit_iters) == list(range(1, 61))


class _PlainModel:
    """A model that declares no runtime hooks (every model but the MiniMax-M3 sm_107 targets)."""


class _M3Model:
    """A model whose runtime hooks opt into the iteration-2 emission, as the MiniMax-M3 sm_107 targets do."""

    runtime_hooks = RuntimeHooks(first_content_emit_iters=2)


def test_runtime_hooks_choose_the_schedule():
    """PyExecutor reads the schedule from the model's runtime hooks: a model without hooks keeps the stock schedule
    (1, 20, 40, ...), one that opts in also emits on iteration 2."""
    plain = runtime_hooks_of(_PlainModel()).first_content_emit_iters
    m3 = runtime_hooks_of(_M3Model()).first_content_emit_iters

    assert _emitted_iters(20, plain) == [1, 20, 40, 60]
    assert _emitted_iters(20, m3) == [1, 2, 20, 40, 60]


@pytest.mark.parametrize("decoding_iter", [1, 2, 19])
@pytest.mark.parametrize("emit_first_iter", [False, True])
@pytest.mark.parametrize("first_gen_emit", [False, True])
def test_first_generation_content_preserves_early_emit_suppression(
    decoding_iter, emit_first_iter, first_gen_emit
):
    executor = PyExecutor.__new__(PyExecutor)
    executor.stream_interval = 20
    executor.first_content_emit_iters = 1
    request = Mock(
        spec_set=[
            "py_decoding_iter",
            "is_finished",
            "py_stream_lead_in_tokens",
            "py_stream_emitted_tokens",
        ],
        py_decoding_iter=decoding_iter,
        is_finished=False,
        py_stream_lead_in_tokens=0,
        py_stream_emitted_tokens=0,
    )
    expected = emit_first_iter if decoding_iter == 1 else first_gen_emit
    assert executor._should_emit_response(request, emit_first_iter, first_gen_emit) is expected


PROMPT_LEN = 100


def _stream(stream_interval, lead_in, tokens_per_iter, last=60, emit_first_iter=True):
    """Mimic ``_handle_responses``; return (iteration, generated tokens) per emission."""
    fake = _fake_self(stream_interval, 1)
    request = types.SimpleNamespace(
        py_decoding_iter=0,
        is_finished=False,
        py_stream_lead_in_tokens=lead_in,
        py_stream_emitted_tokens=0,
        orig_prompt_len=PROMPT_LEN,
        max_beam_num_tokens=PROMPT_LEN,
    )
    emitted = []
    for it in range(1, last + 1):
        request.py_decoding_iter = it
        request.max_beam_num_tokens += tokens_per_iter
        if PyExecutor._should_emit_response(fake, request, emit_first_iter):
            PyExecutor._record_stream_emission(request)
            emitted.append((it, request.max_beam_num_tokens - PROMPT_LEN))
    return emitted


@pytest.mark.parametrize("tokens_per_iter", [1, 2, 3, 4])
@pytest.mark.parametrize("lead_in", [1, 6, 11])
def test_lead_in_emits_the_step_of_the_first_visible_token(lead_in, tokens_per_iter):
    """The first visible token (``lead_in + 1``) is emitted in the iteration that generated it."""
    emitted = _stream(20, lead_in, tokens_per_iter)
    first_visible_iter = -(-(lead_in + 1) // tokens_per_iter)
    assert first_visible_iter in [it for it, _ in emitted]
    assert [it for it, _ in emitted][:first_visible_iter] == list(range(1, first_visible_iter + 1))


@pytest.mark.parametrize("tokens_per_iter", [1, 3])
@pytest.mark.parametrize("lead_in", [0, 6])
def test_first_token_is_emitted_at_once(lead_in, tokens_per_iter):
    assert _stream(20, lead_in, tokens_per_iter)[0] == (1, tokens_per_iter)


@pytest.mark.parametrize("tokens_per_iter", [1, 3])
@pytest.mark.parametrize("stream_interval", [3, 20])
def test_lead_in_window_then_stock_batching(stream_interval, tokens_per_iter):
    """After the lead-in window the stock schedule resumes."""
    lead_in = 6
    emitted = [it for it, _ in _stream(stream_interval, lead_in, tokens_per_iter)]
    window_end = -(-(lead_in + 1) // tokens_per_iter)
    assert emitted[:window_end] == list(range(1, window_end + 1))
    assert emitted[window_end:] == [it for it in _stock_iters(stream_interval) if it > window_end]


@pytest.mark.parametrize("tokens_per_iter", [1, 3])
@pytest.mark.parametrize("stream_interval", [2, 3, 20])
def test_no_lead_in_keeps_stock_schedule(stream_interval, tokens_per_iter):
    emitted = [it for it, _ in _stream(stream_interval, 0, tokens_per_iter)]
    assert emitted == _stock_iters(stream_interval)


@pytest.mark.parametrize("lead_in", [0, 6])
def test_lead_in_with_stream_interval_one_emits_every_iteration(lead_in):
    assert [it for it, _ in _stream(1, lead_in, 3)] == list(range(1, 61))


def test_lead_in_keeps_early_first_token_suppression():
    """With ``emit_first_iter=False`` iteration 1 stays suppressed."""
    emitted = [it for it, _ in _stream(20, 6, 1, emit_first_iter=False)]
    assert emitted[:7] == [2, 3, 4, 5, 6, 7, 20]


def _gen_only_stream(stream_interval, lead_in, tokens_per_iter, overlap, no_response=(), last=60):
    """``_stream`` for a generation-only request: the first token came from the context server, overlap skips
    iteration 1, and iterations in ``no_response`` get no response."""
    fake = _fake_self(stream_interval, 1)
    request = types.SimpleNamespace(
        py_decoding_iter=0,
        is_finished=False,
        py_stream_lead_in_tokens=lead_in,
        py_stream_emitted_tokens=0,
        py_gen_first_emitted=False,
        orig_prompt_len=PROMPT_LEN,
        max_beam_num_tokens=PROMPT_LEN + 1,
    )
    emitted = []
    for it in range(1, last + 1):
        request.py_decoding_iter = it
        request.max_beam_num_tokens += tokens_per_iter
        if overlap and it <= 1:
            continue
        first_gen_emit = not request.py_gen_first_emitted
        if not PyExecutor._should_emit_response(fake, request, True, first_gen_emit):
            continue
        if it in no_response:
            continue
        if first_gen_emit:
            request.py_gen_first_emitted = True
        PyExecutor._record_stream_emission(request)
        emitted.append((it, request.max_beam_num_tokens - PROMPT_LEN))
    return emitted


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("tokens_per_iter", [1, 2, 3, 4])
@pytest.mark.parametrize("lead_in", [1, 6, 11])
def test_lead_in_on_a_generation_only_request(lead_in, tokens_per_iter, overlap):
    """The context server's token counts toward the lead-in."""
    emitted = _gen_only_stream(20, lead_in, tokens_per_iter, overlap)
    iters = [it for it, _ in emitted]
    first_allowed = 2 if overlap else 1
    first_visible_iter = max(first_allowed, -(-lead_in // tokens_per_iter))
    window = list(range(first_allowed, first_visible_iter + 1))
    assert iters[: len(window)] == window
    assert [n > lead_in for _, n in emitted[: len(window)]] == [False] * (len(window) - 1) + [True]
    assert iters[len(window) :] == [it for it in _stock_iters(20) if it > first_visible_iter]


@pytest.mark.parametrize("overlap", [False, True])
def test_lead_in_on_a_generation_only_request_without_a_response(overlap):
    """An iteration without a response records nothing."""
    first_allowed = 2 if overlap else 1
    emitted = _gen_only_stream(20, 6, 1, overlap, no_response={first_allowed, first_allowed + 1})
    # one token per iteration after the context server's: token 7, the first visible one, comes at iteration 6
    assert [it for it, _ in emitted] == list(range(first_allowed + 2, 7)) + [20, 40, 60]
    assert emitted[0][1] == first_allowed + 3


def test_record_stream_emission_without_lead_in_is_a_no_op():
    request = types.SimpleNamespace(
        py_stream_lead_in_tokens=0,
        py_stream_emitted_tokens=0,
        orig_prompt_len=PROMPT_LEN,
        max_beam_num_tokens=PROMPT_LEN + 50,
    )
    PyExecutor._record_stream_emission(request)
    assert request.py_stream_emitted_tokens == 0


@pytest.mark.parametrize("streaming", [False, True])
def test_lead_in_reaches_the_llm_request(streaming):
    """The lead-in reaches the LlmRequest for streamed requests only."""
    from tensorrt_llm._torch.pyexecutor.llm_request import executor_request_to_llm_request
    from tensorrt_llm.executor.base_worker import BaseWorker
    from tensorrt_llm.executor.request import GenerationRequest
    from tensorrt_llm.sampling_params import SamplingParams

    sampling_params = SamplingParams(max_tokens=4)
    sampling_params._stream_lead_in_tokens = 6
    request = GenerationRequest([1, 2, 3], sampling_params, streaming=streaming)
    request.set_id(42)
    worker = MagicMock()
    worker._is_pytorch_backend = True
    worker.max_seq_len = None
    worker.engine.enqueue_request.return_value = 42
    BaseWorker._enqueue_request(worker, request)
    native_request = worker.engine.enqueue_request.call_args.args[0]

    llm_request = executor_request_to_llm_request(
        42, native_request, child_req_ids=[], exclude_last_generation_logits=False
    )
    assert llm_request.py_stream_lead_in_tokens == (6 if streaming else 0)
    assert llm_request.py_stream_emitted_tokens == 0


def test_child_requests_inherit_the_lead_in():
    from tensorrt_llm._torch.pyexecutor.llm_request import executor_request_to_llm_request
    from tensorrt_llm.bindings import executor as bindings

    request = bindings.Request(
        input_token_ids=[1, 2, 3],
        max_tokens=4,
        streaming=True,
        sampling_config=bindings.SamplingConfig(num_return_sequences=2),
    )
    request.py_stream_lead_in_tokens = 6
    llm_request = executor_request_to_llm_request(
        42, request, child_req_ids=[43], exclude_last_generation_logits=False
    )
    assert llm_request.py_stream_lead_in_tokens == 6
    assert llm_request.child_requests[0].py_stream_lead_in_tokens == 6
