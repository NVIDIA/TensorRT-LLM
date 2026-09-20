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
import itertools
from unittest.mock import MagicMock

import pytest

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.bench.benchmark.utils.asynchronous import LlmManager
from tensorrt_llm.bench.dataclasses.general import InferenceRequest
from tensorrt_llm.executor.postproc_worker import PostprocParams

pytestmark = pytest.mark.cpu_only


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

    req = InferenceRequest(task_id=0, input_ids=[1, 2, 3], output_tokens=10)
    sampling_params = SamplingParams()
    post_proc_params = PostprocParams()

    # Enqueue 3 requests. Each takes 0.6s.
    # Total time if all processed: 1.8s.
    # With duration=1, it should stop after processing 2 requests.
    await manager.enqueue(req, sampling_params, post_proc_params)
    await manager.enqueue(req, sampling_params, post_proc_params)
    await manager.enqueue(req, sampling_params, post_proc_params)

    manager.run()

    # Wait for the worker to fully drain: it exits its loop when the duration
    # elapses and then awaits all in-flight tasks. Asserting after full drain
    # (rather than sampling mid-flight) ensures requests dispatched past the
    # deadline were actually skipped, not merely still running.
    await asyncio.wait_for(manager._backend_task, timeout=10)

    # The worker should have stopped and cleared the inbox.
    assert manager._inbox.empty()

    # Request 3 acquires its concurrency slot at t=1.2s, past the 1s deadline,
    # so it is always skipped. Request 2 acquires at t=0.6s and normally makes
    # the deadline, but a scheduling stall on a loaded machine can push it past;
    # the point of the test is that the limit drops requests, not exactly how
    # many land on the boundary.
    assert 1 <= outbox.qsize() < 3

    await manager.stop()


@pytest.mark.asyncio
async def test_llm_manager_duration_bounds_runtime_with_eager_dispatch():
    """Regression test for duration enforcement under eager task dispatch.

    The worker dispatches the entire inbox into tasks before any request
    runs, so the duration must be enforced at execution time; otherwise
    every dispatched request runs to completion and duration has no
    effect on runtime.
    """
    mock_llm = MagicMock(spec=LLM)
    mock_llm.args = MagicMock()
    mock_llm.args.parallel_config = MagicMock()
    mock_llm.args.parallel_config.world_size = 1

    mock_output = MagicMock()
    mock_output.prompt_token_ids = [1, 2, 3]
    mock_output.outputs = [MagicMock(token_ids=[4, 5])]
    mock_output.finished = True
    mock_output.id = 1
    mock_output.decoding_iter = 1

    request_latency = 0.2

    async def mock_aresult():
        await asyncio.sleep(request_latency)
        return mock_output

    mock_output.aresult = mock_aresult
    mock_llm.generate_async.return_value = mock_output

    outbox = asyncio.Queue()
    # Sized so that unenforced execution (num_requests / concurrency *
    # request_latency = 5s) is far above the enforced runtime (~1s), leaving a
    # wide margin for the wall-clock assertion below on a loaded machine.
    num_requests = 50
    concurrency = 2
    duration = 1

    manager = LlmManager(
        llm=mock_llm,
        outbox=outbox,
        streaming=False,
        concurrency=concurrency,
        duration=duration,
    )

    req = InferenceRequest(task_id=0, input_ids=[1, 2, 3], output_tokens=10)
    sampling_params = SamplingParams()
    post_proc_params = PostprocParams()

    for _ in range(num_requests):
        await manager.enqueue(req, sampling_params, post_proc_params)

    start = asyncio.get_running_loop().time()
    manager.run()
    await asyncio.wait_for(manager._backend_task, timeout=30)
    elapsed = asyncio.get_running_loop().time() - start

    # With enforcement, roughly duration / request_latency * concurrency = 10
    # requests complete; without it, all 50 would.
    assert outbox.qsize() < num_requests, (
        "Duration limit had no effect: all requests were processed."
    )
    # Wall time is duration plus at most one in-flight drain. The slack keeps
    # this robust on a loaded machine while staying far below the ~5s an
    # unenforced run would take.
    assert elapsed < duration + request_latency + 1.5

    await manager.stop()


@pytest.mark.asyncio
async def test_llm_manager_cancels_in_flight_requests_on_failure():
    """A failed request aborts the run instead of draining the in-flight ones.

    Only a duration-triggered exit waits for in-flight requests, so that their
    statistics are recorded. On failure the benchmark is aborting anyway, and
    waiting would make an erroring run slower than a successful one.
    """
    mock_llm = MagicMock(spec=LLM)
    mock_llm.args = MagicMock()
    mock_llm.args.parallel_config = MagicMock()
    mock_llm.args.parallel_config.world_size = 1

    slow_latency = 30  # Far longer than the test should ever wait.
    call_count = itertools.count()

    def generate_async(*args, **kwargs):
        output = MagicMock()
        output.prompt_token_ids = [1, 2, 3]
        output.outputs = [MagicMock(token_ids=[4, 5])]
        output.finished = True
        output.id = next(call_count)
        output.decoding_iter = 1

        # The first request fails; the rest hang until cancelled.
        if output.id == 0:

            async def mock_aresult():
                raise ValueError("simulated request failure")
        else:

            async def mock_aresult():
                await asyncio.sleep(slow_latency)
                return output

        output.aresult = mock_aresult
        return output

    mock_llm.generate_async.side_effect = generate_async

    outbox = asyncio.Queue()
    manager = LlmManager(
        llm=mock_llm,
        outbox=outbox,
        streaming=False,
        concurrency=4,
    )

    req = InferenceRequest(task_id=0, input_ids=[1, 2, 3], output_tokens=10)
    for _ in range(4):
        await manager.enqueue(req, SamplingParams(), PostprocParams())

    start = asyncio.get_running_loop().time()
    manager.run()
    with pytest.raises(ValueError, match="simulated request failure"):
        await asyncio.wait_for(manager._backend_task, timeout=10)
    elapsed = asyncio.get_running_loop().time() - start

    # Without cancellation the worker would block on the 30s requests.
    assert elapsed < slow_latency, (
        "Worker waited for in-flight requests instead of cancelling them."
    )


@pytest.mark.asyncio
async def test_llm_manager_stops_draining_when_a_request_fails():
    """A failure during the duration drain is not held back by a hung request.

    Draining preserves the statistics of requests still running at the
    deadline, but once one of them fails the run ends regardless, so the
    remaining ones must not delay the error.
    """
    mock_llm = MagicMock(spec=LLM)
    mock_llm.args = MagicMock()
    mock_llm.args.parallel_config = MagicMock()
    mock_llm.args.parallel_config.world_size = 1

    duration = 1
    fail_latency = 1.5  # Fails after the deadline, while draining.
    hang_latency = 30  # Must never be waited on in full.
    call_count = itertools.count()

    def generate_async(*args, **kwargs):
        output = MagicMock()
        output.prompt_token_ids = [1, 2, 3]
        output.outputs = [MagicMock(token_ids=[4, 5])]
        output.finished = True
        output.id = next(call_count)
        output.decoding_iter = 1

        if output.id == 0:

            async def mock_aresult():
                await asyncio.sleep(fail_latency)
                raise ValueError("simulated request failure")
        else:

            async def mock_aresult():
                await asyncio.sleep(hang_latency)
                return output

        output.aresult = mock_aresult
        return output

    mock_llm.generate_async.side_effect = generate_async

    outbox = asyncio.Queue()
    manager = LlmManager(
        llm=mock_llm,
        outbox=outbox,
        streaming=False,
        concurrency=2,
        duration=duration,
    )

    req = InferenceRequest(task_id=0, input_ids=[1, 2, 3], output_tokens=10)
    for _ in range(2):
        await manager.enqueue(req, SamplingParams(), PostprocParams())

    start = asyncio.get_running_loop().time()
    manager.run()
    with pytest.raises(ValueError, match="simulated request failure"):
        await asyncio.wait_for(manager._backend_task, timeout=15)
    elapsed = asyncio.get_running_loop().time() - start

    # The failure surfaces once it happens; draining for the hung request
    # would delay it until hang_latency.
    assert elapsed < hang_latency, (
        "Drain waited for the hung request instead of surfacing the failure."
    )


@pytest.mark.asyncio
async def test_llm_manager_drops_truncated_multi_turn_request():
    """A conversation cut short by the deadline is not recorded.

    Its turns are incomplete, so emitting it would mix partial and full
    conversations in the same statistics.
    """
    mock_llm = MagicMock(spec=LLM)
    mock_llm.args = MagicMock()
    mock_llm.args.parallel_config = MagicMock()
    mock_llm.args.parallel_config.world_size = 1

    turn_latency = 0.6

    def generate_async(*args, **kwargs):
        output = MagicMock()
        output.prompt_token_ids = [1, 2, 3]
        output.outputs = [MagicMock(token_ids=[4, 5])]
        output.finished = True
        output.id = 1
        output.decoding_iter = 1

        async def mock_aresult():
            await asyncio.sleep(turn_latency)
            return output

        output.aresult = mock_aresult
        return output

    mock_llm.generate_async.side_effect = generate_async

    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = [1, 2, 3]
    tokenizer.decode.return_value = "answer"

    outbox = asyncio.Queue()
    manager = LlmManager(
        llm=mock_llm,
        outbox=outbox,
        streaming=False,
        concurrency=1,
        duration=1,
        tokenizer=tokenizer,
    )

    # Four turns at 0.6s each cannot finish inside the 1s deadline, so the
    # conversation is cut short after turn 2.
    req = InferenceRequest(
        task_id=0, input_ids=[1, 2, 3], output_tokens=10, turns=["q1", "q2", "q3", "q4"]
    )
    await manager.enqueue(req, SamplingParams(), PostprocParams())

    manager.run()
    await asyncio.wait_for(manager._backend_task, timeout=15)

    assert outbox.empty(), "A conversation truncated by the deadline was recorded as complete."

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

    req = InferenceRequest(task_id=0, input_ids=[1, 2, 3], output_tokens=10)
    sampling_params = SamplingParams()
    post_proc_params = PostprocParams()

    # Enqueue 2 requests. Each takes 0.6s.
    # Total time: 1.2s.
    # With duration=5, all requests should be processed.
    await manager.enqueue(req, sampling_params, post_proc_params)
    await manager.enqueue(req, sampling_params, post_proc_params)

    manager.run()

    # Await the perf items themselves rather than sleeping a fixed interval.
    # The worker loops until the 5s duration elapses, so there is no task
    # completion to await here, and a fixed sleep would race the requests on a
    # loaded machine.
    for _ in range(2):
        await asyncio.wait_for(outbox.get(), timeout=10)

    # Both requests were processed and none are left pending.
    assert outbox.empty()
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

    # StatsKeeper records requests in a dict keyed by request id, so each
    # response needs its own id -- a shared one would merge the requests into a
    # single record and skew their timings.
    def make_output(request_id):
        output = MagicMock()
        output.prompt_token_ids = [1, 2, 3]
        output.outputs = [MagicMock(token_ids=[4, 5])]
        output.finished = True
        output.id = request_id
        output.decoding_iter = 1

        async def mock_aresult():
            await asyncio.sleep(0.6)  # Make it take time
            return output

        output.aresult = mock_aresult
        return output

    response_ids = itertools.count()
    mock_llm.generate_async.side_effect = lambda *args, **kwargs: make_output(next(response_ids))

    requests = [
        InferenceRequest(task_id=i, input_ids=[1, 2, 3], output_tokens=10) for i in range(3)
    ]

    # Patch EnergyMonitor and tqdm so we don't depend on actual NVML / environment
    with (
        patch("tensorrt_llm.bench.benchmark.utils.asynchronous.EnergyMonitor") as mock_energy,
        patch("tensorrt_llm.bench.benchmark.utils.asynchronous.tqdm.tqdm"),
    ):
        # Mock the context manager of EnergyMonitor
        mock_energy.return_value.__enter__.return_value.total_energy = 100.0

        stats = await async_benchmark(
            llm=mock_llm,
            sampling_params=SamplingParams(),
            post_proc_params=PostprocParams(),
            requests=requests,
            streaming=False,
            concurrency=1,
            duration=1,  # 1 second limit
        )

    # With concurrency=1 requests run back-to-back (0.6s each), so request 3
    # acquires its slot past the 1s deadline and is always skipped. Request 2
    # sits on the boundary, so assert the limit took effect rather than the
    # exact count. Without a concurrency limit all three would start at once
    # and finish before the deadline, which is why the CLI rejects that
    # combination.
    assert 1 <= len(stats.requests) < 3
