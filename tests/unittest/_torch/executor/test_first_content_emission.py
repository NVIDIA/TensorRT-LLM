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
* ``stream_interval = 1`` emits every iteration whatever the setting.
"""

from __future__ import annotations

import types

import pytest

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks, runtime_hooks_of

pytestmark = pytest.mark.cpu_only


def _fake_self(stream_interval: int, first_content_emit_iters: int):
    return types.SimpleNamespace(
        stream_interval=stream_interval, first_content_emit_iters=first_content_emit_iters
    )


def _request(decoding_iter: int, finished: bool = False):
    return types.SimpleNamespace(py_decoding_iter=decoding_iter, is_finished=finished)


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
