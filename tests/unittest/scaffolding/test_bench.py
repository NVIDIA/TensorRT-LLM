import asyncio
import math
from typing import List

import numpy as np

from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.executor.result import GenerationResult
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.scaffolding import (GenerationTask,
                                      NativeGenerationController,
                                      ScaffoldingBenchRequest, ScaffoldingLlm,
                                      Task, TaskCollection, TaskStatus, Worker,
                                      async_scaffolding_benchmark)
from tensorrt_llm.scaffolding.load_generation_strategy import (
    ConcurrentStrategy, ConstantRateStrategy, LoadGenerationStrategy,
    PoissonRateStrategy, SynchronousStrategy, ThroughputStrategy)

OUTPUT_STR = "Yes."


class DummyWorker(Worker):

    async def dummy_generation_handler(self, task: GenerationTask):
        task.result = GenerationResult(
            GenerationRequest(prompt_token_ids=[0],
                              sampling_params=SamplingParams()))
        task.result._done = True
        task.result.output_str = OUTPUT_STR
        return TaskStatus.SUCCESS

    task_handlers = {GenerationTask: dummy_generation_handler}


class DummyTaskCollection(TaskCollection):

    def __init__(self):
        super().__init__()
        self.output_len = 0

    def before_yield(self, tasks: List[Task]):
        pass

    def after_yield(self, tasks: List[Task]):
        self.output_len = len(tasks[0].result.output_str)


def test_scaffolding_benchmark():
    task_collection_types = {"bench_dummy_collection": DummyTaskCollection}

    prototype_controller = NativeGenerationController()
    dummy_worker = DummyWorker()
    workers = {NativeGenerationController.WorkerTag.GENERATION: dummy_worker}
    scaffolding_llm = ScaffoldingLlm(prototype_controller, workers)

    requests_num = 100
    requests = [
        ScaffoldingBenchRequest(prompt="Is today a nice day?")
        for _ in range(requests_num)
    ]

    concurrency = 10

    results, requests_start_time, requests_execution_time, total_time = asyncio.run(
        async_scaffolding_benchmark(scaffolding_llm, task_collection_types,
                                    requests, concurrency))

    scaffolding_llm.shutdown()

    assert len(results) == requests_num
    assert len(requests_execution_time) == requests_num
    assert results[0].cur_output.output_str == OUTPUT_STR
    assert results[0].task_collections[
        "bench_dummy_collection"].output_len == len(OUTPUT_STR)


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

    strategies = [
        ("Synchronous", SynchronousStrategy(), 10),
        ("Concurrent@3", ConcurrentStrategy(concurrency=3), 10),
        ("Throughput", ThroughputStrategy(), 10),
        ("ConstantRate@10", ConstantRateStrategy(rate=2.0), 100),
        ("PoissonRate@10", PoissonRateStrategy(rate=2.0), 100),
    ]
    for name, strategy, num_requests in strategies:
        print(f"==============  Testing {name} strategy ==================")
        requests = create_requests(num_requests=num_requests)

        results, requests_start_time, requests_execution_time, total_time = asyncio.run(
            async_scaffolding_benchmark(scaffolding_llm,
                                        task_collection_types,
                                        requests,
                                        strategy=strategy))

        # Verify strategy timing characteristics
        verify_strategy_timing(strategy, requests_start_time, rtol=0.3)

    print(f"before shutdown")
    scaffolding_llm.shutdown()


if __name__ == "__main__":
    test_scaffolding_benchmark()
    test_scaffolding_benchmark_strategies()
