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
"""The chat SSE generator (``tensorrt_llm/serve/_base/sse.py``), driven with fakes.

The server and the engine promise are local stand-ins: the generator reads
three things from the server (``postproc_worker_enabled``, the adjusted clock
and ``_extract_metrics``) and iterates the promise, and nothing else.
"""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tensorrt_llm.executor.postproc_worker import PostprocParams
from tensorrt_llm.logger import logger
from tensorrt_llm.serve._base.sse import chat_stream_generator

pytestmark = pytest.mark.cpu_only


class _Promise:
    """An async iterator of engine results, as a streaming ``RequestOutput`` is."""

    def __init__(self, results, *, then=None):
        self._results = list(results)
        self._then = then

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._results:
            return self._results.pop(0)
        if self._then is not None:
            raise self._then
        raise StopAsyncIteration


def _result(frames):
    """One engine result whose post-processing worker output is ``frames``."""
    return SimpleNamespace(outputs=[SimpleNamespace(_postprocess_result=frames)])


def _server(*, postproc_worker_enabled):
    return SimpleNamespace(
        postproc_worker_enabled=postproc_worker_enabled,
        _adjusted_steady_clock=SimpleNamespace(now=lambda: 7.0),
        _extract_metrics=AsyncMock(),
    )


def _raw_request():
    return SimpleNamespace(state=SimpleNamespace())


async def _collect(stream):
    return [frame async for frame in stream]


@pytest.fixture
def error_records():
    """Messages of the ERROR records the TRT-LLM logger emits during the test.

    The library logger does not propagate, so ``caplog`` would see nothing; a
    handler is attached to it instead.
    """
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = Capture(level=logging.ERROR)
    logger._logger.addHandler(handler)
    try:
        yield records
    finally:
        logger._logger.removeHandler(handler)


def test_frames_then_done_then_metrics(error_records):
    results = [_result(["data: a\n\n"]), _result(["data: b\n\n"])]
    server = _server(postproc_worker_enabled=True)
    raw_request = _raw_request()
    # With the post-processing worker on, the generator never reads the params.
    params = PostprocParams(post_processor=None, postproc_args=None)

    frames = asyncio.run(
        _collect(
            chat_stream_generator(_Promise(results), params, server=server, raw_request=raw_request)
        )
    )

    assert frames == ["data: a\n\n", "data: b\n\n", "data: [DONE]\n\n"]
    assert raw_request.state.server_first_token_time == 7.0
    server._extract_metrics.assert_awaited_once_with(results[-1], raw_request)
    assert error_records == []


def test_in_process_post_processor_is_used():
    results = [_result(None), _result(None)]
    server = _server(postproc_worker_enabled=False)
    raw_request = _raw_request()
    seen = []

    def post_processor(result, args):
        seen.append((result, args))
        return [f"data: {len(seen)}\n\n"]

    params = PostprocParams(post_processor=post_processor, postproc_args="the args")

    frames = asyncio.run(
        _collect(
            chat_stream_generator(_Promise(results), params, server=server, raw_request=raw_request)
        )
    )

    assert frames == ["data: 1\n\n", "data: 2\n\n", "data: [DONE]\n\n"]
    assert seen == [(results[0], "the args"), (results[1], "the args")]
    assert raw_request.state.server_first_token_time == 7.0
    server._extract_metrics.assert_awaited_once_with(results[-1], raw_request)


def test_a_failure_is_logged_and_raised(error_records):
    server = _server(postproc_worker_enabled=True)
    promise = _Promise([_result(["data: a\n\n"])], then=RuntimeError("boom"))
    params = PostprocParams(post_processor=None, postproc_args=None)

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(
            _collect(
                chat_stream_generator(promise, params, server=server, raw_request=_raw_request())
            )
        )

    assert len(error_records) == 1, error_records
    assert "RuntimeError: boom" in error_records[0]
    server._extract_metrics.assert_not_awaited()
