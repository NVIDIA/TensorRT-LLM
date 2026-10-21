# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest
from test_common.serve.app_factory import AggConfig, isolated_serve_env

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def serve_env(tmp_path):
    with isolated_serve_env(tmp_path) as env:
        yield env


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_processor_metrics_separate_queue_and_execution(fails, serve_env):
    server = serve_env.build_openai_server(AggConfig())
    server.metrics_collector = MagicMock()

    def process():
        if fails:
            raise ValueError("bad input")
        return "processed"

    with ThreadPoolExecutor(max_workers=1) as pool:
        server._input_proc_executor = pool
        with patch(
            "tensorrt_llm.serve.openai_server.time.perf_counter", side_effect=[1.0, 1.5, 1.75]
        ):
            if fails:
                with pytest.raises(ValueError, match="bad input"):
                    await server._run_input_processor(process)
            else:
                assert await server._run_input_processor(process) == "processed"

    server.metrics_collector.log_input_processing.assert_called_once_with("processor", 0.25, 0.5)


@pytest.mark.asyncio
async def test_async_input_stage_preserves_failure_and_records_elapsed_time(serve_env):
    server = serve_env.build_openai_server(AggConfig())
    server.metrics_collector = MagicMock()

    async def load():
        raise ValueError("bad media")

    with patch("tensorrt_llm.serve.openai_server.time.perf_counter", side_effect=[1.0, 2.0]):
        with pytest.raises(ValueError, match="bad media"):
            await server._time_input_stage("media_load", load())

    server.metrics_collector.log_input_processing.assert_called_once_with("media_load", 1.0)


@pytest.mark.asyncio
async def test_disabled_input_metrics_do_not_read_clock(serve_env):
    server = serve_env.build_openai_server(AggConfig())
    server.metrics_collector = None

    async def load():
        return "media"

    with ThreadPoolExecutor(max_workers=1) as pool:
        server._input_proc_executor = pool
        with patch("tensorrt_llm.serve.openai_server.time.perf_counter") as clock:
            assert await server._run_input_processor(lambda: "processed") == "processed"
            assert await server._time_input_stage("media_load", load()) == "media"
        clock.assert_not_called()
