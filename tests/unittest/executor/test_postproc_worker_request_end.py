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

import pytest

from tensorrt_llm.executor.postproc_worker import PostprocWorker
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


def test_an_error_for_a_request_the_worker_never_saw_is_harmless():
    hook = _Hook()
    worker = _worker(hook, [[PostprocWorker.Input(rsp=ErrorResponse(9, "boom", 9))]])

    batches = asyncio.run(_drain(worker))

    assert hook.ended == []
    assert isinstance(batches[0][0], ErrorResponse)
