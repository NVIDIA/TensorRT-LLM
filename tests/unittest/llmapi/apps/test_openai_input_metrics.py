# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, call, patch

import pytest

from tensorrt_llm.metrics.perf_utils import _input_processing_observer, _time_input_processing
from tensorrt_llm.serve.openai_server import OpenAIServer

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def observer():
    observer = MagicMock()
    token = _input_processing_observer.set(observer)
    yield observer
    _input_processing_observer.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_processor_metrics_separate_queue_and_execution(fails, observer):
    server = object.__new__(OpenAIServer)

    def process():
        if fails:
            raise ValueError("bad input")
        return "processed"

    with ThreadPoolExecutor(max_workers=1) as pool:
        server._input_proc_executor = pool
        with patch(
            "tensorrt_llm.metrics.perf_utils.time.perf_counter", side_effect=[1.0, 1.5, 1.5, 1.75]
        ):
            if fails:
                with pytest.raises(ValueError, match="bad input"):
                    await server._run_input_processor(process)
            else:
                assert await server._run_input_processor(process) == "processed"

    assert observer.call_args_list == [
        call("processor_queue", 0.5, None),
        call("processor", 0.25, None),
    ]


@pytest.mark.asyncio
async def test_input_stage_preserves_failure_and_records_elapsed_time(observer):
    with patch("tensorrt_llm.metrics.perf_utils.time.perf_counter", side_effect=[1.0, 2.0]):
        with pytest.raises(ValueError, match="bad media"):
            with _time_input_processing("media_load"):
                raise ValueError("bad media")

    observer.assert_called_once_with("media_load", 1.0, None)


@pytest.mark.asyncio
async def test_disabled_input_metrics_do_not_read_clock():
    server = object.__new__(OpenAIServer)

    async def load():
        return "media"

    with ThreadPoolExecutor(max_workers=1) as pool:
        server._input_proc_executor = pool
        with patch("tensorrt_llm.metrics.perf_utils.time.perf_counter") as clock:
            assert await server._run_input_processor(lambda: "processed") == "processed"
            with _time_input_processing("media_load"):
                assert await load() == "media"
        clock.assert_not_called()
