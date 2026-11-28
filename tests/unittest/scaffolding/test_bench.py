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

import asyncio
import math
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Iterator, List

import numpy as np
import pytest

from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.executor.result import GenerationResult
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.scaffolding import (GenerationTask,
                                      NativeGenerationController,
                                      ScaffoldingBenchRequest, ScaffoldingLlm,
                                      Task, TaskCollection, TaskStatus, Worker,
                                      async_scaffolding_benchmark, benchmark,
                                      load_generation_strategy)
from tensorrt_llm.scaffolding.load_generation_strategy import (
    ConcurrentStrategy, ConstantRateStrategy, LoadGenerationStrategy,
    PoissonRateStrategy, SynchronousStrategy, ThroughputStrategy)

pytestmark = pytest.mark.cpu_only

OUTPUT_STR = "Yes."


@pytest.fixture(autouse=True)
def _restore_native_generation_controller() -> Iterator[None]:
    original_init = NativeGenerationController.__init__
    original_process = NativeGenerationController.process

    yield

    NativeGenerationController.__init__ = original_init
    NativeGenerationController.process = original_process


@contextmanager
def _mock_benchmark_clock() -> Iterator[None]:
    current_time = 1000.0
    sleepers: dict[asyncio.Future[None], float] = {}

    def now() -> float:
        return current_time

    def advance() -> None:
        nonlocal current_time
        for future in list(sleepers):
            if future.cancelled():
                del sleepers[future]
        if not sleepers:
            return
        current_time = min(sleepers.values())
        ready = [
            future for future, deadline in sleepers.items()
            if deadline == current_time
        ]
        for future in ready:
            del sleepers[future]
            future.set_result(None)
        if sleepers:
            asyncio.get_running_loop().call_soon(advance)

    async def sleep(delay: float) -> None:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        if not sleepers:
            loop.call_soon(advance)
        sleepers[future] = current_time + delay
        await future

    clock = SimpleNamespace(time=now)
    benchmark_asyncio = SimpleNamespace(**vars(asyncio))
    benchmark_asyncio.sleep = sleep
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(benchmark, "time", clock)
        monkeypatch.setattr(load_generation_strategy, "time", clock)
        monkeypatch.setattr(benchmark, "asyncio", benchmark_asyncio)
        yield


class DummyWorker(Worker):

    async def dummy_generation_handler(self, task: GenerationTask):
        task.result = GenerationResult(
            GenerationRequest(prompt_token_ids=[0],
                              sampling_params=SamplingParams()))
        task.result._done = True
        task.output_str = OUTPUT_STR
        return TaskStatus.SUCCESS

    task_handlers = {GenerationTask: dummy_generation_handler}


class DummyTaskCollection(TaskCollection):

    def __init__(self):
        super().__init__()
        self.output_len = 0

    def before_yield(self, tasks: List[Task]):
        pass

    def after_yield(self, tasks: List[Task]):
        self.output_len = len(tasks[0].output_str)


def test_scaffolding_benchmark():
    task_collection_types = {"bench_dummy_collection": DummyTaskCollection}

    prototype_controller = NativeGenerationController()
    dummy_worker = DummyWorker()
    workers = {NativeGenerationController.WorkerTag.GENERATION: dummy_worker}
    scaffolding_llm = ScaffoldingLlm(prototype_controller, workers)

    try:
        requests_num = 100
        requests = [
            ScaffoldingBenchRequest(prompt="Is today a nice day?")
            for _ in range(requests_num)
        ]

        concurrency = 10

        results, requests_start_time, requests_execution_time, total_time = asyncio.run(
            async_scaffolding_benchmark(scaffolding_llm, task_collection_types,
                                        requests, concurrency))

        assert len(results) == requests_num
        assert len(requests_start_time) == requests_num
        assert len(requests_execution_time) == requests_num
    finally:
        scaffolding_llm.shutdown()


def setup_scaffolding():
    """Setup scaffolding LLM for testing."""
    prototype_controller = NativeGenerationController()
    dummy_worker = DummyWorker()
    workers = {NativeGenerationController.WorkerTag.GENERATION: dummy_worker}
    scaffolding_llm = ScaffoldingLlm(prototype_controller, workers)
    return scaffolding_llm


def create_requests(num_requests: int = 20):
    """Create test requests."""
    return [
        ScaffoldingBenchRequest(prompt=f"Request {i}: Is today a nice day?")
        for i in range(num_requests)
    ]


def calculate_intervals(times: List[float], strategy):
    rate = strategy.rate
    initial_burst = strategy.initial_burst
    burst_size = math.floor(rate) if initial_burst else 0

    intervals = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    test_intervals = intervals[burst_size:]

    return test_intervals, 1.0 / rate


def verify_strategy_timing(strategy: LoadGenerationStrategy,
                           requests_start_time: List[float],
                           rtol: float = 0.3):
    times = sorted(requests_start_time)
    if isinstance(strategy, ConstantRateStrategy):
        test_intervals, expected_interval = calculate_intervals(times, strategy)

        if test_intervals:
            avg_interval = np.mean(test_intervals)
            max_interval = np.max(test_intervals)
            min_interval = np.min(test_intervals)

            assert np.allclose(avg_interval, expected_interval, atol=0.0, rtol=rtol), \
                f"Avg Interval: {avg_interval:.3f}s vs Expected Interval: {expected_interval:.3f}s"
            assert np.allclose(max_interval, expected_interval, atol=0.0, rtol=rtol), \
                f"Max Interval: {max_interval:.3f}s vs Expected Interval: {expected_interval:.3f}s"
            assert np.allclose(min_interval, expected_interval, atol=0.0, rtol=rtol), \
                f"Min Interval: {min_interval:.3f}s vs Expected Interval: {expected_interval:.3f}s"

    elif isinstance(strategy, PoissonRateStrategy):
        test_intervals, expected_mean_interval = calculate_intervals(
            times, strategy)

        if test_intervals:
            avg_interval = np.mean(test_intervals)
            std_dev = np.std(test_intervals)
            cv = std_dev / avg_interval if avg_interval > 0 else 0

            assert np.allclose(avg_interval, expected_mean_interval, atol=0.0, rtol=rtol), \
                f"Avg Interval: {avg_interval:.3f}s vs Expected Mean Interval: {expected_mean_interval:.3f}s"
            assert np.allclose(std_dev, expected_mean_interval, atol=0.0, rtol=rtol), \
                f"Std Dev: {std_dev:.3f}s vs Expected Mean Interval: {expected_mean_interval:.3f}s"
            assert np.allclose(cv, 1.0, atol=0.0, rtol=rtol), \
                f"CV (std/mean): {cv:.2f} vs Expected: 1.0"


def test_scaffolding_benchmark_strategies():
    task_collection_types = {"bench_collection": DummyTaskCollection}
    scaffolding_llm = setup_scaffolding()

    try:
        with _mock_benchmark_clock():
            strategies = [
                ("Synchronous", SynchronousStrategy(), 10),
                ("Concurrent@3", ConcurrentStrategy(concurrency=3), 10),
                ("Throughput", ThroughputStrategy(), 10),
                ("ConstantRate@2", ConstantRateStrategy(rate=2.0), 100),
                ("PoissonRate@2", PoissonRateStrategy(rate=2.0), 100),
            ]
            for name, strategy, num_requests in strategies:
                print(
                    f"==============  Testing {name} strategy =================="
                )
                requests = create_requests(num_requests=num_requests)

                results, requests_start_time, requests_execution_time, total_time = asyncio.run(
                    async_scaffolding_benchmark(scaffolding_llm,
                                                task_collection_types,
                                                requests,
                                                strategy=strategy))

                assert len(results) == num_requests
                assert len(requests_start_time) == num_requests
                assert len(requests_execution_time) == num_requests
                assert results[0].outputs[
                    0].text == OUTPUT_STR, f"Expected {OUTPUT_STR} but got {results[0].outputs[0].text}"
                assert results[0].task_collections[
                    "bench_collection"].output_len == len(OUTPUT_STR)

                # Verify strategy timing characteristics
                verify_strategy_timing(strategy, requests_start_time, rtol=0.3)
    finally:
        scaffolding_llm.shutdown()


def test_benchmark_dispatch_concurrency() -> None:

    async def check_concurrency() -> None:
        requests = create_requests(num_requests=3)
        expected_prompts = [request.prompt for request in requests]
        all_dispatched = asyncio.Event()
        dispatched_prompts: list[str] = []
        completed_prompts: list[str] = []

        async def finish_request(prompt: str) -> None:
            await all_dispatched.wait()
            completed_prompts.append(prompt)

        def generate_async(prompt: str) -> SimpleNamespace:
            dispatched_prompts.append(prompt)
            if len(dispatched_prompts) == len(requests):
                all_dispatched.set()
            return SimpleNamespace(prompt=prompt,
                                   aresult=lambda: finish_request(prompt))

        input_queue = asyncio.Queue()
        output_queue = asyncio.Queue()
        await benchmark.enqueue_requests(input_queue, requests)
        await benchmark.run_scaffolding_llm(
            SimpleNamespace(generate_async=generate_async), input_queue,
            output_queue, ConstantRateStrategy(rate=2.0))

        result_prompts = []
        while (entry := await output_queue.get()) is not None:
            result_prompts.append(entry[0].prompt)

        assert dispatched_prompts == expected_prompts
        assert sorted(completed_prompts) == sorted(expected_prompts)
        assert sorted(result_prompts) == sorted(expected_prompts)

    with _mock_benchmark_clock():
        assert benchmark.asyncio.sleep is not asyncio.sleep
        asyncio.run(asyncio.wait_for(check_concurrency(), timeout=10))


@pytest.mark.parametrize("cancel_parent", [False, True])
def test_benchmark_clock_cancellation(cancel_parent: bool) -> None:
    """Self-test the clock with a live peer or with every sleeper cancelled."""

    async def check_cancellation() -> None:
        loop = asyncio.get_running_loop()
        callback_errors: list[dict[str, object]] = []
        callback_error = asyncio.Event()

        def record_error(_event_loop: asyncio.AbstractEventLoop,
                         context: dict[str, object]) -> None:
            callback_errors.append(context)
            callback_error.set()

        async def wait_for_sleepers() -> list[BaseException | None]:
            sleepers = [
                asyncio.create_task(benchmark.asyncio.sleep(0.01)),
                asyncio.create_task(benchmark.asyncio.sleep(0.02)),
            ]
            if cancel_parent:
                current_task = asyncio.current_task()
                assert current_task is not None
                loop.call_soon(current_task.cancel)
            else:
                loop.call_soon(sleepers[0].cancel)
            return await asyncio.gather(*sleepers, return_exceptions=True)

        original_handler = loop.get_exception_handler()
        loop.set_exception_handler(record_error)
        parent = asyncio.create_task(wait_for_sleepers())
        error_waiter = asyncio.create_task(callback_error.wait())
        try:
            completed, _pending = await asyncio.wait(
                (parent, error_waiter),
                timeout=10,
                return_when=asyncio.FIRST_COMPLETED)
            assert not callback_errors
            assert parent in completed
            if cancel_parent:
                assert parent.cancelled()
            else:
                results = parent.result()
                assert isinstance(results[0], asyncio.CancelledError)
                assert results[1] is None
        finally:
            parent.cancel()
            error_waiter.cancel()
            await asyncio.gather(parent, error_waiter, return_exceptions=True)
            loop.set_exception_handler(original_handler)

    with _mock_benchmark_clock():
        assert benchmark.asyncio.sleep is not asyncio.sleep
        start_time = benchmark.time.time()
        asyncio.run(check_cancellation())
        if cancel_parent:
            assert benchmark.time.time() == start_time


if __name__ == "__main__":
    test_scaffolding_benchmark()
    test_scaffolding_benchmark_strategies()
