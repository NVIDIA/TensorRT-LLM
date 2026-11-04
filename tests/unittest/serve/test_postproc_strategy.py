# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The post-processing strategy a server chooses once at construction.

Every handler used to decide per response whether the post-processed value
comes from a postproc worker (stored on ``outputs[0]``) or must be computed
in the server process. These tests pin the strategy objects that replaced
those decisions and the public ``CompletionOutput.postprocess_result``
accessor the worker strategy reads.
"""

import asyncio
from types import SimpleNamespace

import pytest

from tensorrt_llm.executor.postproc_worker import PostprocParams
from tensorrt_llm.executor.result import CompletionOutput, GenerationResultBase
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.openai_server import OpenAIServer
from tensorrt_llm.serve.postprocess_handlers import (
    InlinePostprocStrategy,
    PostprocStrategy,
    WorkerPostprocStrategy,
    select_postproc_strategy,
)

pytestmark = pytest.mark.cpu_only


def _result_with_stored_value(value):
    """A real result object as the frontend leaves it in worker mode."""
    result = GenerationResultBase(id=1, sampling_params=SamplingParams())
    result._outputs[0]._postprocess_result = value
    return result


def test_select_postproc_strategy_maps_the_flag():
    worker = select_postproc_strategy(True)
    inline = select_postproc_strategy(False)
    assert isinstance(worker, WorkerPostprocStrategy)
    assert isinstance(inline, InlinePostprocStrategy)
    assert isinstance(worker, PostprocStrategy)
    assert isinstance(inline, PostprocStrategy)


def test_worker_strategy_reads_the_stored_result_and_accepts_no_params():
    # openai_completion passes postproc_params=None in worker mode.
    result = _result_with_stored_value("worker-result")
    assert WorkerPostprocStrategy().apply(result, None) == "worker-result"


def test_inline_strategy_runs_the_post_processor_exactly_once():
    result = GenerationResultBase(id=2, sampling_params=SamplingParams())
    calls = []

    def post_processor(res, args):
        calls.append((res, args))
        return "inline-result"

    params = PostprocParams(post_processor=post_processor, postproc_args="args-sentinel")
    assert InlinePostprocStrategy().apply(result, params) == "inline-result"
    assert calls == [(result, "args-sentinel")]


def test_worker_apply_async_returns_the_field_without_calling_inline():
    result = _result_with_stored_value("stored")

    def inline():
        raise AssertionError("inline must not run in worker mode")

    value = asyncio.run(WorkerPostprocStrategy().apply_async(result, inline))
    assert value == "stored"


def test_inline_apply_async_awaits_the_inline_coroutine_once():
    result = GenerationResultBase(id=3, sampling_params=SamplingParams())
    calls = []

    async def inline():
        calls.append("called")
        return "async-result"

    value = asyncio.run(InlinePostprocStrategy().apply_async(result, inline))
    assert value == "async-result"
    assert calls == ["called"]


class _ServerStandIn:
    """Just what choosing the strategy reads off a server.

    Reuses the real ``postproc_worker_enabled`` property so these tests pin
    the production decision, not a copy of it.
    """

    postproc_worker_enabled = OpenAIServer.postproc_worker_enabled

    def __init__(self, num_postprocess_workers):
        self.generator = SimpleNamespace(
            args=SimpleNamespace(num_postprocess_workers=num_postprocess_workers)
        )
        self._is_visual_gen = False


def test_server_strategy_is_chosen_from_num_postprocess_workers():
    assert isinstance(OpenAIServer._postproc.func(_ServerStandIn(0)), InlinePostprocStrategy)
    assert isinstance(OpenAIServer._postproc.func(_ServerStandIn(2)), WorkerPostprocStrategy)


def test_server_strategy_is_the_same_object_on_every_access():
    stand_in = _ServerStandIn(2)
    first = OpenAIServer._postproc.__get__(stand_in)
    assert OpenAIServer._postproc.__get__(stand_in) is first
    assert isinstance(first, WorkerPostprocStrategy)


def test_completion_output_postprocess_result_accessor():
    output = CompletionOutput(0)
    assert output.postprocess_result is None
    output._postprocess_result = "worker-result"
    assert output.postprocess_result == "worker-result"
