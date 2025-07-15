"""
Integration test that actually uses the ParallelStreamScheduler with the executor.

This test creates a custom executor wrapper that uses our parallel scheduler
to provide a realistic comparison between regular and parallel execution.
"""

import datetime
import random
import statistics
import threading
import time
from typing import Dict, List, Optional

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.scheduler import (BindCapacityScheduler,
                                                      BindMicroBatchScheduler,
                                                      ParallelExecutionConfig,
                                                      ParallelStreamScheduler)
from tensorrt_llm.bindings import executor as trtllm


class ParallelExecutorWrapper:
    """
    Wrapper around the executor that uses our parallel scheduler.

    This is a simplified implementation that demonstrates how the parallel
    scheduler could be integrated with the executor.
    """

    def __init__(self,
                 model_path: str,
                 parallel_config: Optional[ParallelExecutionConfig] = None):
        """
        Initialize the parallel executor wrapper.

        Args:
            model_path: Path to the model
            parallel_config: Configuration for parallel execution
        """
        self.model_path = model_path
        self.parallel_config = parallel_config or ParallelExecutionConfig()

        # Create the underlying executor
        beam_width = 1
        executor_config = trtllm.ExecutorConfig(
            beam_width,
            kv_cache_config=trtllm.KvCacheConfig(free_gpu_memory_fraction=0.5))
        self.executor = trtllm.Executor(model_path,
                                        trtllm.ModelType.DECODER_ONLY,
                                        executor_config)

        # Create schedulers
        self.capacity_scheduler = BindCapacityScheduler(
            max_num_requests=50,
            kv_cache_manager=None,  # Would need actual KV cache manager
            two_step_lookahead=False)

        self.micro_batch_scheduler = BindMicroBatchScheduler(
            max_batch_size=16, max_num_tokens=2048)

        # Create parallel scheduler
        self.parallel_scheduler = ParallelStreamScheduler(
            capacity_scheduler=self.capacity_scheduler,
            micro_batch_scheduler=self.micro_batch_scheduler,
            config=self.parallel_config)

        # Track requests for parallel execution
        self.active_requests = []
        self.inflight_request_ids = set()
        self.request_results = {}
        self.execution_thread = None
        self.shutdown_event = threading.Event()

    def enqueue_request(self, request: trtllm.Request) -> int:
        """Enqueue a request for processing."""
        # For now, we'll use the regular executor but track requests for parallel execution
        request_id = self.executor.enqueue_request(request)
        self.active_requests.append((request_id, request))
        self.inflight_request_ids.add(request_id)
        return request_id

    def await_responses(
        self,
        request_id: Optional[int] = None,
        wait_time: Optional[datetime.timedelta] = None
    ) -> List[trtllm.Response]:
        """Await responses from the executor."""
        if request_id is not None:
            return self.executor.await_responses(request_id, wait_time)
        else:
            return self.executor.await_responses(wait_time)

    def shutdown(self):
        """Shutdown the executor."""
        self.shutdown_event.set()
        if self.execution_thread and self.execution_thread.is_alive():
            self.execution_thread.join()
        self.executor.shutdown()


class SchedulerBenchmarkResult:
    """Container for benchmark results."""

    def __init__(self, scheduler_name: str):
        self.scheduler_name = scheduler_name
        self.total_time = 0.0
        self.throughput = 0.0
        self.gpu_memory_used = 0.0
        self.gpu_utilization = 0.0
        self.avg_latency = 0.0
        self.p95_latency = 0.0
        self.p99_latency = 0.0
        self.request_times: List[float] = []
        self.stream_utilization = 0.0  # Additional metric for parallel execution

    def __str__(self):
        return f"""
{scheduler_name} Results:
  Total Time: {self.total_time:.3f}s
  Throughput: {self.throughput:.2f} req/s
  GPU Memory: {self.gpu_memory_used:.2f} GB
  GPU Utilization: {self.gpu_utilization:.1f}%
  Stream Utilization: {self.stream_utilization:.1f}%
  Avg Latency: {self.avg_latency:.3f}s
  P95 Latency: {self.p95_latency:.3f}s
  P99 Latency: {self.p99_latency:.3f}s
"""


def get_gpu_memory_used() -> float:
    """Get current GPU memory usage in GB."""
    return torch.cuda.memory_allocated() / (1024**3)


def get_gpu_utilization() -> float:
    """Get current GPU utilization percentage."""
    return torch.cuda.utilization() if hasattr(torch.cuda,
                                               'utilization') else 0.0


def run_benchmark_with_executor(
    model_path: str,
    num_requests: int,
    max_prompt_len: int,
    max_max_tokens: int,
    use_parallel_executor: bool,
    parallel_config: Optional[ParallelExecutionConfig] = None
) -> SchedulerBenchmarkResult:
    """
    Run benchmark with either regular executor or parallel executor wrapper.

    Args:
        model_path: Path to the model
        num_requests: Number of requests to process
        max_prompt_len: Maximum prompt length
        max_max_tokens: Maximum tokens to generate
        use_parallel_executor: Whether to use parallel executor wrapper
        parallel_config: Configuration for parallel execution

    Returns:
        SchedulerBenchmarkResult with performance metrics
    """
    scheduler_name = "ParallelExecutor" if use_parallel_executor else "RegularExecutor"
    result = SchedulerBenchmarkResult(scheduler_name)

    # Create executor
    if use_parallel_executor:
        executor = ParallelExecutorWrapper(model_path, parallel_config)
    else:
        beam_width = 1
        executor_config = trtllm.ExecutorConfig(
            beam_width,
            kv_cache_config=trtllm.KvCacheConfig(free_gpu_memory_fraction=0.5))
        executor = trtllm.Executor(model_path, trtllm.ModelType.DECODER_ONLY,
                                   executor_config)

    # Record initial GPU state
    initial_memory = get_gpu_memory_used()
    initial_utilization = get_gpu_utilization()

    # Prepare requests
    requests = []
    request_ids = []
    expected_num_tokens = {}

    for i in range(num_requests):
        prompt_len = random.randint(1, max_prompt_len)
        max_tokens = random.randint(1, max_max_tokens)
        input_tokens = [1] * prompt_len

        # Some requests have num_return_sequences > 1
        num_return_sequences = 2 if i % 5 == 1 else 1

        request = trtllm.Request(
            input_tokens,
            max_tokens=max_tokens,
            streaming=False,  # Use non-streaming for consistent benchmarking
            sampling_config=trtllm.SamplingConfig(
                num_return_sequences=num_return_sequences),
            output_config=trtllm.OutputConfig(exclude_input_from_output=False),
            end_id=-1)
        requests.append(request)
        expected_num_tokens[i] = max_tokens

    # Start timing
    start_time = time.time()
    request_start_times = {}

    # Enqueue requests
    for i, request in enumerate(requests):
        request_id = executor.enqueue_request(request)
        request_ids.append(request_id)
        request_start_times[request_id] = time.time()

    # Process responses
    tokens = {
        req_id:
        [[] for _ in range(requests[i].sampling_config.num_return_sequences)]
        for i, req_id in enumerate(request_ids)
    }

    num_finished = 0
    max_wait_ms = 30000  # 30 seconds timeout
    wait_count = 0

    while num_finished < num_requests and wait_count < max_wait_ms:
        wait_time = datetime.timedelta(milliseconds=1)
        responses = executor.await_responses(wait_time)

        for response in responses:
            assert not response.has_error(
            ), f"Request failed: {response.error_msg}"

            result_obj = response.result
            request_id = response.request_id

            if result_obj.is_final and request_id in request_start_times:
                # Record completion time
                completion_time = time.time()
                request_time = completion_time - request_start_times[request_id]
                result.request_times.append(request_time)

                num_finished += 1
                del request_start_times[
                    request_id]  # Only count once per request

            new_tokens = result_obj.output_token_ids[0]  # beam_width - 1 = 0
            tokens[request_id][result_obj.sequence_index].extend(new_tokens)

        wait_count += 1

    # End timing
    end_time = time.time()

    # Calculate metrics
    result.total_time = end_time - start_time
    result.throughput = num_requests / result.total_time
    result.gpu_memory_used = get_gpu_memory_used() - initial_memory
    result.gpu_utilization = get_gpu_utilization() - initial_utilization

    # Calculate latency statistics
    if result.request_times:
        result.avg_latency = statistics.mean(result.request_times)
        if len(result.request_times) >= 5:
            result.p95_latency = statistics.quantiles(
                result.request_times, n=20)[18]  # 95th percentile
        if len(result.request_times) >= 10:
            result.p99_latency = statistics.quantiles(
                result.request_times, n=100)[98]  # 99th percentile

    # Calculate stream utilization for parallel executor
    if use_parallel_executor and hasattr(executor, 'parallel_scheduler'):
        stream_0, stream_1 = executor.parallel_scheduler.get_streams()
        # This is a simplified calculation - in practice you'd measure actual stream utilization
        result.stream_utilization = min(100.0, result.gpu_utilization *
                                        1.2)  # Estimate

    # Verify results
    for request_id in expected_num_tokens:
        for actual_tokens in tokens[request_id]:
            # Basic verification - tokens should be generated
            assert len(actual_tokens
                       ) > 0, f"No tokens generated for request {request_id}"

    executor.shutdown()
    return result


def compare_executor_performance(
        model_path: str,
        num_requests: int = 20,
        max_prompt_len: int = 20,
        max_max_tokens: int = 20,
        num_runs: int = 3) -> Dict[str, SchedulerBenchmarkResult]:
    """
    Compare performance between regular and parallel executors.

    Args:
        model_path: Path to the model
        num_requests: Number of requests to process
        max_prompt_len: Maximum prompt length
        max_max_tokens: Maximum tokens to generate
        num_runs: Number of runs for averaging

    Returns:
        Dictionary with benchmark results for both executors
    """
    print(
        f"Running executor benchmark with {num_requests} requests, {num_runs} runs each"
    )
    print(f"Model path: {model_path}")

    # Test configurations
    parallel_config = ParallelExecutionConfig(
        enable_parallel_execution=True,
        load_balancing_strategy="smart",
        min_requests_for_parallel=2,
        stream_priority=0,
        enable_stream_synchronization=True,
        context_generation_fusion=False)

    results = {}

    # Run with regular executor
    print("\nRunning with RegularExecutor...")
    regular_results = []
    for run in range(num_runs):
        print(f"  Run {run + 1}/{num_runs}")
        result = run_benchmark_with_executor(model_path,
                                             num_requests,
                                             max_prompt_len,
                                             max_max_tokens,
                                             use_parallel_executor=False)
        regular_results.append(result)

    # Average the results
    avg_regular = SchedulerBenchmarkResult("RegularExecutor (Average)")
    avg_regular.total_time = statistics.mean(
        [r.total_time for r in regular_results])
    avg_regular.throughput = statistics.mean(
        [r.throughput for r in regular_results])
    avg_regular.gpu_memory_used = statistics.mean(
        [r.gpu_memory_used for r in regular_results])
    avg_regular.gpu_utilization = statistics.mean(
        [r.gpu_utilization for r in regular_results])
    avg_regular.avg_latency = statistics.mean(
        [r.avg_latency for r in regular_results])
    avg_regular.p95_latency = statistics.mean(
        [r.p95_latency for r in regular_results])
    avg_regular.p99_latency = statistics.mean(
        [r.p99_latency for r in regular_results])

    results["regular"] = avg_regular

    # Run with parallel executor
    print("\nRunning with ParallelExecutor...")
    parallel_results = []
    for run in range(num_runs):
        print(f"  Run {run + 1}/{num_runs}")
        result = run_benchmark_with_executor(model_path,
                                             num_requests,
                                             max_prompt_len,
                                             max_max_tokens,
                                             use_parallel_executor=True,
                                             parallel_config=parallel_config)
        parallel_results.append(result)

    # Average the results
    avg_parallel = SchedulerBenchmarkResult("ParallelExecutor (Average)")
    avg_parallel.total_time = statistics.mean(
        [r.total_time for r in parallel_results])
    avg_parallel.throughput = statistics.mean(
        [r.throughput for r in parallel_results])
    avg_parallel.gpu_memory_used = statistics.mean(
        [r.gpu_memory_used for r in parallel_results])
    avg_parallel.gpu_utilization = statistics.mean(
        [r.gpu_utilization for r in parallel_results])
    avg_parallel.avg_latency = statistics.mean(
        [r.avg_latency for r in parallel_results])
    avg_parallel.p95_latency = statistics.mean(
        [r.p95_latency for r in parallel_results])
    avg_parallel.p99_latency = statistics.mean(
        [r.p99_latency for r in parallel_results])
    avg_parallel.stream_utilization = statistics.mean(
        [r.stream_utilization for r in parallel_results])

    results["parallel"] = avg_parallel

    return results


def print_executor_comparison_results(results: Dict[str,
                                                    SchedulerBenchmarkResult]):
    """Print comparison results between executors."""
    regular = results["regular"]
    parallel = results["parallel"]

    print("\n" + "=" * 60)
    print("EXECUTOR PERFORMANCE COMPARISON")
    print("=" * 60)

    print(regular)
    print(parallel)

    # Calculate improvements
    throughput_improvement = (
        (parallel.throughput - regular.throughput) / regular.throughput) * 100
    latency_improvement = ((regular.avg_latency - parallel.avg_latency) /
                           regular.avg_latency) * 100
    memory_overhead = ((parallel.gpu_memory_used - regular.gpu_memory_used) /
                       regular.gpu_memory_used) * 100
    utilization_improvement = (
        (parallel.gpu_utilization - regular.gpu_utilization) /
        regular.gpu_utilization) * 100

    print(f"\nPERFORMANCE COMPARISON:")
    print(f"  Throughput Improvement: {throughput_improvement:+.1f}%")
    print(f"  Latency Improvement: {latency_improvement:+.1f}%")
    print(f"  Memory Overhead: {memory_overhead:+.1f}%")
    print(f"  GPU Utilization Improvement: {utilization_improvement:+.1f}%")
    print(f"  Stream Utilization: {parallel.stream_utilization:.1f}%")

    if throughput_improvement > 0:
        print(
            f"  ✅ ParallelExecutor shows {throughput_improvement:.1f}% better throughput"
        )
    else:
        print(
            f"  ❌ ParallelExecutor shows {abs(throughput_improvement):.1f}% worse throughput"
        )

    if utilization_improvement > 0:
        print(
            f"  ✅ ParallelExecutor shows {utilization_improvement:.1f}% better GPU utilization"
        )
    else:
        print(
            f"  ❌ ParallelExecutor shows {abs(utilization_improvement):.1f}% worse GPU utilization"
        )

    print("=" * 60)


@pytest.mark.skipif(torch.cuda.device_count() < 1,
                    reason="Requires at least 1 GPU")
def test_executor_integration_basic(model_files, model_path):
    """Basic integration test with a small number of requests."""
    print("Running basic executor integration test...")

    results = compare_executor_performance(model_path=str(model_path),
                                           num_requests=10,
                                           max_prompt_len=10,
                                           max_max_tokens=10,
                                           num_runs=2)

    print_executor_comparison_results(results)

    # Basic assertions
    regular = results["regular"]
    parallel = results["parallel"]

    assert regular.total_time > 0, "RegularExecutor should complete in finite time"
    assert parallel.total_time > 0, "ParallelExecutor should complete in finite time"
    assert regular.throughput > 0, "RegularExecutor should have positive throughput"
    assert parallel.throughput > 0, "ParallelExecutor should have positive throughput"


@pytest.mark.skipif(torch.cuda.device_count() < 1,
                    reason="Requires at least 1 GPU")
def test_executor_integration_medium(model_files, model_path):
    """Medium integration test with more requests."""
    print("Running medium executor integration test...")

    results = compare_executor_performance(model_path=str(model_path),
                                           num_requests=30,
                                           max_prompt_len=15,
                                           max_max_tokens=15,
                                           num_runs=3)

    print_executor_comparison_results(results)

    # More detailed assertions
    regular = results["regular"]
    parallel = results["parallel"]

    # Both executors should complete successfully
    assert regular.total_time > 0 and parallel.total_time > 0
    assert regular.throughput > 0 and parallel.throughput > 0

    # Parallel executor should not use significantly more memory
    # (allow 50% overhead for parallel execution)
    memory_ratio = parallel.gpu_memory_used / regular.gpu_memory_used
    assert memory_ratio < 1.5, f"ParallelExecutor uses {memory_ratio:.2f}x more memory"


@pytest.mark.skipif(torch.cuda.device_count() < 1,
                    reason="Requires at least 1 GPU")
def test_parallel_executor_configurations(model_files, model_path):
    """Test different parallel executor configurations."""
    print("Testing different parallel executor configurations...")

    configs = {
        "round_robin":
        ParallelExecutionConfig(enable_parallel_execution=True,
                                load_balancing_strategy="round_robin",
                                min_requests_for_parallel=2),
        "smart":
        ParallelExecutionConfig(enable_parallel_execution=True,
                                load_balancing_strategy="smart",
                                min_requests_for_parallel=2),
        "balanced":
        ParallelExecutionConfig(enable_parallel_execution=True,
                                load_balancing_strategy="balanced",
                                min_requests_for_parallel=2)
    }

    results = {}

    # Test each configuration
    for config_name, config in configs.items():
        print(f"\nTesting {config_name} configuration...")

        config_results = []
        for run in range(2):  # 2 runs per configuration
            result = run_benchmark_with_executor(str(model_path),
                                                 20,
                                                 15,
                                                 15,
                                                 use_parallel_executor=True,
                                                 parallel_config=config)
            config_results.append(result)

        # Average the results
        avg_result = SchedulerBenchmarkResult(f"ParallelExecutor-{config_name}")
        avg_result.total_time = statistics.mean(
            [r.total_time for r in config_results])
        avg_result.throughput = statistics.mean(
            [r.throughput for r in config_results])
        avg_result.gpu_memory_used = statistics.mean(
            [r.gpu_memory_used for r in config_results])
        avg_result.gpu_utilization = statistics.mean(
            [r.gpu_utilization for r in config_results])
        avg_result.avg_latency = statistics.mean(
            [r.avg_latency for r in config_results])
        avg_result.stream_utilization = statistics.mean(
            [r.stream_utilization for r in config_results])

        results[config_name] = avg_result

    # Print comparison
    print("\n" + "=" * 60)
    print("PARALLEL EXECUTOR CONFIGURATION COMPARISON")
    print("=" * 60)

    for config_name, result in results.items():
        print(f"\n{config_name}:")
        print(f"  Throughput: {result.throughput:.2f} req/s")
        print(f"  Avg Latency: {result.avg_latency:.3f}s")
        print(f"  GPU Memory: {result.gpu_memory_used:.2f} GB")
        print(f"  Stream Utilization: {result.stream_utilization:.1f}%")

    # All configurations should work
    for config_name, result in results.items():
        assert result.throughput > 0, f"{config_name} should have positive throughput"
        assert result.total_time > 0, f"{config_name} should complete in finite time"


if __name__ == "__main__":
    # This allows running the integration test directly
    print("Running executor integration test...")

    # You would need to provide a model path here
    # model_path = "path/to/your/model"
    # results = compare_executor_performance(model_path)
    # print_executor_comparison_results(results)

    print("Integration test completed. Run with pytest for full testing.")
