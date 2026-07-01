# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
from unittest.mock import MagicMock

import pytest

from tensorrt_llm import SamplingParams
from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm.bench.benchmark.utils.asynchronous import LlmManager
from tensorrt_llm.bench.dataclasses.general import InferenceRequest
from tensorrt_llm.executor.postproc_worker import PostprocParams


@pytest.mark.asyncio
async def test_llm_manager_duration():
    # Mock LLM
    mock_llm = MagicMock(spec=LLM)
    mock_llm.args = MagicMock()
    mock_llm.args.parallel_config = MagicMock()
    mock_llm.args.parallel_config.world_size = 1

    # Mock generate_async to return a mock output
    mock_output = MagicMock()
    mock_output.prompt_token_ids = [1, 2, 3]
    mock_output.outputs = [MagicMock(token_ids=[4, 5])]
    mock_output.finished = True
    mock_output.id = 1
    mock_output.decoding_iter = 1

    # We need to mock aresult() which is an async method
    async def mock_aresult():
        await asyncio.sleep(0.6)  # Make it take time
        return mock_output

    mock_output.aresult = mock_aresult
    mock_llm.generate_async.return_value = mock_output

    outbox = asyncio.Queue()

    manager = LlmManager(
        llm=mock_llm,
        outbox=outbox,
        streaming=False,
        concurrency=1,
        duration=1,  # 1 second
    )

    req = InferenceRequest(input_ids=[1, 2, 3], output_tokens=10)
    sampling_params = SamplingParams()
    post_proc_params = PostprocParams()

    # Enqueue 3 requests. Each takes 0.6s.
    # Total time if all processed: 1.8s.
    # With duration=1, it should stop after processing 2 requests.
    await manager.enqueue(req, sampling_params, post_proc_params)
    await manager.enqueue(req, sampling_params, post_proc_params)
    await manager.enqueue(req, sampling_params, post_proc_params)

    manager.run()

    # Wait for more than 1 second (e.g., 1.5s)
    await asyncio.sleep(1.5)

    # The worker should have stopped and cleared the inbox.
    assert manager._inbox.empty()

    # 2 requests should have been processed and put into outbox.
    assert outbox.qsize() == 2

    await manager.stop()


@pytest.mark.asyncio
async def test_llm_manager_duration_not_exceeded():
    # Mock LLM
    mock_llm = MagicMock(spec=LLM)
    mock_llm.args = MagicMock()
    mock_llm.args.parallel_config = MagicMock()
    mock_llm.args.parallel_config.world_size = 1

    # Mock generate_async to return a mock output
    mock_output = MagicMock()
    mock_output.prompt_token_ids = [1, 2, 3]
    mock_output.outputs = [MagicMock(token_ids=[4, 5])]
    mock_output.finished = True
    mock_output.id = 1
    mock_output.decoding_iter = 1

    async def mock_aresult():
        await asyncio.sleep(0.6)
        return mock_output

    mock_output.aresult = mock_aresult
    mock_llm.generate_async.return_value = mock_output

    outbox = asyncio.Queue()

    manager = LlmManager(
        llm=mock_llm,
        outbox=outbox,
        streaming=False,
        concurrency=1,
        duration=5,  # 5 seconds, plenty of time
    )

    req = InferenceRequest(input_ids=[1, 2, 3], output_tokens=10)
    sampling_params = SamplingParams()
    post_proc_params = PostprocParams()

    # Enqueue 2 requests. Each takes 0.6s.
    # Total time: 1.2s.
    # With duration=5, all requests should be processed.
    await manager.enqueue(req, sampling_params, post_proc_params)
    await manager.enqueue(req, sampling_params, post_proc_params)

    manager.run()

    # Wait for them to complete
    await asyncio.sleep(1.5)

    # All 2 requests should have been processed and put into outbox.
    assert outbox.qsize() == 2
    assert manager._inbox.empty()

    await manager.stop()


@pytest.mark.asyncio
async def test_async_benchmark_duration():
    from unittest.mock import patch
    from tensorrt_llm.bench.benchmark.utils.asynchronous import async_benchmark

    # Mock LLM
    mock_llm = MagicMock(spec=LLM)
    mock_llm.args = MagicMock()
    mock_llm.args.parallel_config = MagicMock()
    mock_llm.args.parallel_config.world_size = 1

    # Mock generate_async to return a mock output
    mock_output = MagicMock()
    mock_output.prompt_token_ids = [1, 2, 3]
    mock_output.outputs = [MagicMock(token_ids=[4, 5])]
    mock_output.finished = True
    mock_output.id = 1
    mock_output.decoding_iter = 1

    async def mock_aresult():
        await asyncio.sleep(0.6)  # Make it take time
        return mock_output

    mock_output.aresult = mock_aresult
    mock_llm.generate_async.return_value = mock_output

    req = InferenceRequest(input_ids=[1, 2, 3], output_tokens=10)
    requests = [req, req, req]

    # Patch EnergyMonitor and tqdm so we don't depend on actual NVML / environment
    with patch('tensorrt_llm.bench.benchmark.utils.asynchronous.EnergyMonitor') as mock_energy, \
         patch('tensorrt_llm.bench.benchmark.utils.asynchronous.tqdm.tqdm') as mock_tqdm:

        # Mock the context manager of EnergyMonitor
        mock_energy.return_value.__enter__.return_value.total_energy = 100.0

        stats = await async_benchmark(
            llm=mock_llm,
            sampling_params=SamplingParams(),
            post_proc_params=PostprocParams(),
            requests=requests,
            streaming=False,
            duration=1,  # 1 second limit
        )

    # Out of 3 requests taking 0.6s each, only 2 should complete within 1s limit
    # Since we fixed the assertion, this should now finish gracefully without AssertionError
    assert len(stats.requests) == 2
