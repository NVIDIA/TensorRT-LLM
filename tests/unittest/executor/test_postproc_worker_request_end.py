# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A request the post-processing worker drops without a final response tells its hook.

The worker forwards an executor ``ErrorResponse`` without detokenizing, and turns a
postprocessing exception into one. Neither path gives the post-processing hook a final
chunk, so a hook that keeps per-request state would hold it for as long as the worker
lives. These tests drive the worker's own loop over both paths.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm.executor.postproc_worker import PostprocWorker
from tensorrt_llm.executor.postprocessor_hook import apply_post_processor_hook
from tensorrt_llm.executor.result import (
    CompletionOutput,
    DetokenizedGenerationResultBase,
    GenerationResult,
)
from tensorrt_llm.executor.utils import ErrorResponse

pytestmark = pytest.mark.cpu_only


class _Hook:
    def __init__(self):
        self.ended = []

    def __call__(self, chunk):  # pragma: no cover - never reached on these paths
        raise AssertionError("no chunk is expected")

    def on_request_end(self, request_id, reason):
        self.ended.append((request_id, reason))


class _Record:
    """Stands in for a request the worker has already post-processed chunks for."""

    def __init__(self, record_id, hook):
        self.id = record_id
        self._post_processor_hook = hook


class _Pipe:
    def __init__(self, items):
        self._items = list(items)

    async def get_async(self):
        return self._items.pop(0)


class _BrokenResponse:
    """A response whose result cannot be read: postprocessing raises on it."""

    def __init__(self, client_id):
        self.client_id = client_id
        self.request_id = client_id

    @property
    def result(self):
        raise RuntimeError("unreadable result")


class _Response:
    """A readable response for a request the worker has no record for yet."""

    def __init__(self, client_id):
        self.client_id = client_id
        self.request_id = client_id
        self.result = SimpleNamespace(context_logits=None, generation_logits=None)


class _NewRecord:
    """The record the worker creates for a new request; each response runs its hook."""

    def __init__(self, inp, tokenizer):
        self.id = inp.rsp.client_id
        self.outputs = [CompletionOutput(index=0, text="hello")]
        self._done = False
        self._aborted = False
        self._streaming = True

    def _handle_response(self, response):
        DetokenizedGenerationResultBase._apply_post_processor_hook(self)


class _FailingHook(_Hook):
    """Keeps state for each request it sees, then raises on the chunk."""

    def __init__(self):
        super().__init__()
        self.state = {}

    def __call__(self, chunk):
        self.state[chunk.request_id] = chunk.text
        raise RuntimeError("hook bug")

    def on_request_end(self, request_id, reason):
        super().on_request_end(request_id, reason)
        self.state.pop(request_id, None)


class _HookWithRaisingEndLookup(_Hook):
    """A hook whose ``on_request_end`` raises as soon as it is looked up."""

    @property
    def on_request_end(self):
        raise RuntimeError("broken property")


def _worker(hook, inputs):
    worker = object.__new__(PostprocWorker)
    worker._records = {}
    worker._post_processor_hook = hook
    worker._to_stop = asyncio.Event()
    # The trailing None is the shutdown sentinel that ends the loop.
    worker._pull_pipe = _Pipe(inputs + [None])
    return worker


async def _drain(worker):
    batches = []
    async for batch in worker._mainloop():
        if batch is not None:
            batches.append(batch)
    return batches


def test_an_executor_error_releases_the_request_in_the_hook():
    hook = _Hook()
    worker = _worker(hook, [[PostprocWorker.Input(rsp=ErrorResponse(7, "boom", 7))]])
    worker._records[7] = _Record(7, hook)

    batches = asyncio.run(_drain(worker))

    assert worker._records == {}
    assert hook.ended == [(7, "error")], (
        "a request that ends in an executor error must be released in the hook"
    )
    assert isinstance(batches[0][0], ErrorResponse)
    assert batches[0][0].client_id == 7


def test_a_postprocessing_exception_releases_the_request_in_the_hook():
    hook = _Hook()
    worker = _worker(hook, [[PostprocWorker.Input(rsp=_BrokenResponse(8))]])
    worker._records[8] = _Record(8, hook)

    batches = asyncio.run(_drain(worker))

    assert worker._records == {}
    assert hook.ended == [(8, "postprocessing_error")], (
        "a request whose postprocessing raised must be released in the hook"
    )
    assert isinstance(batches[0][0], ErrorResponse)
    assert batches[0][0].client_id == 8


def test_a_hook_exception_releases_the_request_in_the_hook():
    hook = _FailingHook()
    worker = _worker(hook, [[PostprocWorker.Input(rsp=_Response(8))]])
    worker._record_creator = _NewRecord
    worker._tokenizer = None

    batches = asyncio.run(_drain(worker))

    assert isinstance(batches[0][0], ErrorResponse)
    assert batches[0][0].client_id == 8
    assert "hook bug" in batches[0][0].error_msg, "the hook itself must be what raised"
    assert worker._records == {}
    # The hook failure and worker teardown both notify the idempotent callback.
    assert hook.ended == [(8, "postprocessing_error")] * 2
    assert hook.state == {}, "the state the hook kept for the request must be released"


@pytest.mark.parametrize("streaming", [False, True])
def test_a_hook_exception_releases_the_request_without_a_worker(streaming):
    hook = _FailingHook()
    result = SimpleNamespace(
        id=8, outputs=[CompletionOutput(index=0, text="hello")], _done=False, _aborted=False
    )

    with pytest.raises(RuntimeError, match="hook bug"):
        apply_post_processor_hook(hook, result, streaming)

    assert hook.ended == [(8, "postprocessing_error")]
    assert hook.state == {}, "in-proxy hook failures must release per-request state too"


@pytest.mark.parametrize(
    "response, expected_msg",
    [(ErrorResponse(7, "boom", 7), None), (_BrokenResponse(7), "unreadable result")],
    ids=["executor_error", "postprocessing_error"],
)
def test_a_raising_on_request_end_lookup_still_forwards_the_error(response, expected_msg):
    hook = _HookWithRaisingEndLookup()
    worker = _worker(hook, [[PostprocWorker.Input(rsp=response)]])
    worker._records[7] = _Record(7, hook)

    batches = asyncio.run(_drain(worker))

    assert worker._records == {}
    assert isinstance(batches[0][0], ErrorResponse)
    assert batches[0][0].client_id == 7
    if expected_msg is None:
        assert batches[0][0] is response, "the executor's own error must be forwarded"
    else:
        assert expected_msg in batches[0][0].error_msg


def test_an_error_for_a_request_the_worker_never_saw_is_harmless():
    hook = _Hook()
    worker = _worker(hook, [[PostprocWorker.Input(rsp=ErrorResponse(9, "boom", 9))]])

    batches = asyncio.run(_drain(worker))

    assert hook.ended == []
    assert isinstance(batches[0][0], ErrorResponse)


@pytest.mark.parametrize("read_method", ["result", "__next__", "aresult", "__anext__"])
@pytest.mark.parametrize("abort_raises", [False, True])
def test_hook_failure_is_terminal_for_repeated_reads(read_method, abort_raises):
    hook = _FailingHook()
    result = SimpleNamespace(
        id=8,
        outputs=[CompletionOutput(index=0, text="hello")],
        _done=False,
        _aborted=False,
        _terminal_error=None,
        abort=Mock(side_effect=RuntimeError("abort failed") if abort_raises else None),
    )

    def step(*args):
        apply_post_processor_hook(hook, result, streaming=True)

    async def astep():
        step()

    result._result_step = step
    result._aresult_step = astep

    def read():
        value = getattr(GenerationResult, read_method)(result)
        if read_method in ("aresult", "__anext__"):
            asyncio.run(value)

    with pytest.raises(RuntimeError, match="hook bug") as first:
        read()
    with pytest.raises(RuntimeError, match="hook bug") as second:
        read()

    assert second.value is first.value
    assert result._done
    assert result._aborted
    result.abort.assert_called_once_with()
    assert hook.ended == [(8, "postprocessing_error")]
    assert hook.state == {}
