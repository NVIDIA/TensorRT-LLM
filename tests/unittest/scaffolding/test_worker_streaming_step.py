# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scaffolding streaming helpers use only the public result surface.

That surface is the ``finished`` property and async iteration (``anext``).
The fake handle below deliberately implements nothing else, so a regression
back to private members (``_done``, ``_aresult_step``) fails here.
"""

import asyncio

import pytest

from tensorrt_llm.scaffolding import TaskStatus
from tensorrt_llm.scaffolding.contrib.AsyncGeneration.stream_generation import (
    StreamGenerationTask as ContribStreamGenerationTask,
)
from tensorrt_llm.scaffolding.contrib.AsyncGeneration.stream_generation import (
    stream_generation_handler,
)
from tensorrt_llm.scaffolding.task import StreamGenerationTask
from tensorrt_llm.scaffolding.worker import TRTLLMWorker

pytestmark = pytest.mark.cpu_only


class _FakeOutput:
    def __init__(self):
        self.text = ""
        self.token_ids = []
        self.cumulative_logprob = None
        self.logprobs = []


class _FakeStreamingHandle:
    """A streaming generation handle exposing only the public surface.

    One generated token per async-iteration step; ``finished`` once
    ``total_steps`` tokens are out. No ``_done``, no ``_aresult_step``.
    """

    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.steps_taken = 0
        self.outputs = [_FakeOutput()]
        self.context_logits = None

    @property
    def finished(self) -> bool:
        return self.steps_taken >= self.total_steps

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.finished:
            raise StopAsyncIteration
        await asyncio.sleep(0)
        self.steps_taken += 1
        self.outputs[0].text += "t"
        self.outputs[0].token_ids = self.outputs[0].token_ids + [self.steps_taken]
        return self


def test_streaming_generate_helper_steps_until_finished():
    worker = TRTLLMWorker(llm=None, tokenizer=None)
    handle = _FakeStreamingHandle(total_steps=3)
    collected = []

    asyncio.run(worker.streaming_generate_helper(handle, None, collected))

    assert handle.finished
    assert handle.steps_taken == 3
    # One ScaffoldingOutput per non-final step; the last token is not queued.
    assert [out.text for out in collected] == ["t", "tt"]
    assert [out.token_ids for out in collected] == [[1], [1, 2]]


def test_streaming_generate_helper_stops_early_at_step_at_least():
    worker = TRTLLMWorker(llm=None, tokenizer=None)
    handle = _FakeStreamingHandle(total_steps=10)
    collected = []

    asyncio.run(worker.streaming_generate_helper(handle, 2, collected))

    assert not handle.finished
    assert handle.steps_taken == 2
    assert [out.text for out in collected] == ["t", "tt"]


def test_worker_stream_generation_handler_sets_end_flag_when_finished():
    worker = TRTLLMWorker(llm=None, tokenizer=None)
    task = StreamGenerationTask()
    task.streaming_step = 5
    task.request_handle = _FakeStreamingHandle(total_steps=2)

    status = asyncio.run(worker.stream_generation_handler(task))

    assert status == TaskStatus.SUCCESS
    assert task.end_flag
    assert task.output_str == "tt"
    assert task.output_tokens == [1, 2]


class _FakeContribWorker:
    """Only what the AsyncGeneration handler calls on a worker."""

    def convert_task_params(self, task):
        return None


def test_contrib_stream_generation_handler_sets_end_flag_when_finished():
    task = ContribStreamGenerationTask()
    task.streaming_step = 5
    task.request_handle = _FakeStreamingHandle(total_steps=3)

    status = asyncio.run(stream_generation_handler(_FakeContribWorker(), task))

    assert status == TaskStatus.SUCCESS
    assert task.end_flag
    assert task.output_str == "ttt"
    assert task.output_tokens == [1, 2, 3]


def test_contrib_stream_generation_handler_waits_for_more_steps():
    task = ContribStreamGenerationTask()
    task.streaming_step = 2
    task.request_handle = _FakeStreamingHandle(total_steps=5)

    status = asyncio.run(stream_generation_handler(_FakeContribWorker(), task))

    assert status == TaskStatus.SUCCESS
    assert not task.end_flag
    assert task.request_handle.steps_taken == 2
    assert task.output_str == "tt"
    assert task.output_tokens == [1, 2]
