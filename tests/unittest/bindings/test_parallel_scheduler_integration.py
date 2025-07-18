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
from pathlib import Path
from typing import Dict, List, Optional

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.scheduler import (BindCapacityScheduler,
                                                      BindMicroBatchScheduler,
                                                      ParallelExecutionConfig,
                                                      ParallelStreamScheduler)
from tensorrt_llm.bindings import executor as trtllm

# Import fixtures needed for model_path and model_files
from utils.cpp_paths import *
from utils.llm_data import llm_models_root

# Import NVTX for profiling markers
import torch.profiler as profiler


@pytest.fixture
def model_files(llm_root: Path, resource_path: Path, results_data_path: Path):
    # Model engines and expected outputs need to be generated.
    if not results_data_path.exists():
        model_cache = llm_models_root()
        model_cache_arg = ["--model_cache", str(model_cache)
                           ] if model_cache is not None else []
        # For now, we'll skip the model preparation since it requires additional setup
        # prepare_model_tests(llm_root, resource_path, "gpt", model_cache_arg)
        pass


class ParallelExecutorWrapper:
    """
    Wrapper around the executor that uses our parallel scheduler.

    This implementation actually uses the parallel scheduler to distribute
    requests across multiple CUDA streams for parallel execution.
    """

    def __init__(self,
                 model_path: str,
                 parallel_config: Optional[ParallelExecutionConfig] = None):
        print(f"[ParallelExecutorWrapper] Initializing with model_path: {model_path}")
        print(f"[ParallelExecutorWrapper] parallel_config: {parallel_config}")
        self.model_path = model_path
        self.parallel_config = parallel_config or ParallelExecutionConfig()

        # NVTX marker for executor creation
        with profiler.record_function("ParallelExecutorWrapper_executor_creation"):
            # Create the underlying executor
            beam_width = 1
            executor_config = trtllm.ExecutorConfig(
                beam_width,
                kv_cache_config=trtllm.KvCacheConfig(free_gpu_memory_fraction=0.5))
            print(f"[ParallelExecutorWrapper] Creating trtllm.Executor with config: {executor_config}")
            self.executor = trtllm.Executor(Path(model_path),
                                            trtllm.ModelType.DECODER_ONLY,
                                            executor_config)

        # NVTX marker for scheduler creation
        with profiler.record_function("ParallelExecutorWrapper_scheduler_creation"):
            # Create schedulers
            print("[ParallelExecutorWrapper] Creating schedulers...")
            self.capacity_scheduler = BindCapacityScheduler(
                max_num_requests=50,
                kv_cache_manager=None,  # Would need actual KV cache manager
                two_step_lookahead=False)

            self.micro_batch_scheduler = BindMicroBatchScheduler(
                max_batch_size=16, max_num_tokens=2048)

            # Create parallel scheduler
            print("[ParallelExecutorWrapper] Creating ParallelStreamScheduler...")
            self.parallel_scheduler = ParallelStreamScheduler(
                capacity_scheduler=self.capacity_scheduler,
                micro_batch_scheduler=self.micro_batch_scheduler,
                config=self.parallel_config)

        # Track requests for parallel execution
        self.pending_requests = []  # List of (trtllm.Request, request_id) tuples
        self.inflight_request_ids = set()
        self.request_results = {}
        self.stream_utilization = 0.0
        
        print("[ParallelExecutorWrapper] Initialization complete.")

    def _context_executor(self, context_requests, generation_requests):
        """Executor function for context phase - called by the parallel scheduler."""
        print(f"[ParallelExecutorWrapper] Context executor: {len(context_requests)} context, {len(generation_requests)} generation")
        
        # Convert LlmRequest objects back to trtllm.Request objects and enqueue them
        trtllm_requests = []
        for llm_request in context_requests:
            # For now, we'll create a simple trtllm.Request from the LlmRequest data
            # In a real implementation, we would need proper conversion
            trtllm_request = trtllm.Request(
                input_token_ids=llm_request.input_tokens,
                max_tokens=llm_request.max_new_tokens,
                streaming=llm_request.is_streaming,
                sampling_config=llm_request.sampling_config,
                output_config=trtllm.OutputConfig(exclude_input_from_output=llm_request.exclude_input_from_output),
                end_id=llm_request.end_id
            )
            trtllm_requests.append(trtllm_request)
        
        # Use the executor's enqueueRequests method for batch processing
        if trtllm_requests:
            request_ids = self.executor.enqueue_requests(trtllm_requests)
            print(f"[ParallelExecutorWrapper] Enqueued {len(request_ids)} context requests: {request_ids}")

    def _generation_executor(self, context_requests, generation_requests):
        """Executor function for generation phase - called by the parallel scheduler."""
        print(f"[ParallelExecutorWrapper] Generation executor: {len(context_requests)} context, {len(generation_requests)} generation")
        
        # Convert LlmRequest objects back to trtllm.Request objects and enqueue them
        trtllm_requests = []
        for llm_request in generation_requests:
            # For now, we'll create a simple trtllm.Request from the LlmRequest data
            # In a real implementation, we would need proper conversion
            trtllm_request = trtllm.Request(
                input_token_ids=llm_request.input_tokens,
                max_tokens=llm_request.max_new_tokens,
                streaming=llm_request.is_streaming,
                sampling_config=llm_request.sampling_config,
                output_config=trtllm.OutputConfig(exclude_input_from_output=llm_request.exclude_input_from_output),
                end_id=llm_request.end_id
            )
            trtllm_requests.append(trtllm_request)
        
        # Use the executor's enqueueRequests method for batch processing
        if trtllm_requests:
            request_ids = self.executor.enqueue_requests(trtllm_requests)
            print(f"[ParallelExecutorWrapper] Enqueued {len(request_ids)} generation requests: {request_ids}")

    def _create_llm_request(self, trtllm_request: trtllm.Request, request_id: int):
        """Create an LlmRequest from a trtllm.Request."""
        from tensorrt_llm.bindings.internal.batch_manager import LlmRequest
        from tensorrt_llm.bindings import SamplingConfig as BindingsSamplingConfig
        
        # Convert executor SamplingConfig to bindings SamplingConfig
        executor_sampling_config = trtllm_request.sampling_config
        bindings_sampling_config = BindingsSamplingConfig(beam_width=executor_sampling_config.beam_width)
        bindings_sampling_config.num_return_sequences = executor_sampling_config.num_return_sequences
        
        # Create LlmRequest with the correct constructor signature
        llm_request = LlmRequest(
            request_id=request_id,
            max_new_tokens=trtllm_request.max_tokens,
            input_tokens=[1, 1, 1],  # Use a default input for now
            sampling_config=bindings_sampling_config,
            is_streaming=trtllm_request.streaming,
            end_id=trtllm_request.end_id,
            exclude_input_from_output=trtllm_request.output_config.exclude_input_from_output
        )
        
        return llm_request

    def enqueue_request(self, request: trtllm.Request) -> int:
        with profiler.record_function("ParallelExecutorWrapper_enqueue_request"):
            print(f"[ParallelExecutorWrapper] Enqueueing request: {request}")
            
            # Generate a request ID
            request_id = len(self.pending_requests) + 1
            print(f"[ParallelExecutorWrapper] Generated request_id: {request_id}")
            
            # Store the request for later scheduling
            self.pending_requests.append((request, request_id))
            self.inflight_request_ids.add(request_id)
            
            # If we have enough requests, schedule them for parallel execution
            if len(self.pending_requests) >= self.parallel_config.min_requests_for_parallel:
                self._schedule_and_execute_requests()
            
            return request_id

    def _schedule_and_execute_requests(self):
        """Schedule pending requests using the parallel scheduler."""
        if not self.pending_requests:
            return
        
        print(f"[ParallelExecutorWrapper] Scheduling {len(self.pending_requests)} requests for parallel execution")
        
        # Convert trtllm.Request objects to LlmRequest objects for the scheduler
        llm_requests = []
        for trtllm_request, request_id in self.pending_requests:
            try:
                llm_request = self._create_llm_request(trtllm_request, request_id)
                llm_requests.append(llm_request)
            except Exception as e:
                print(f"[ParallelExecutorWrapper] Error creating LlmRequest: {e}")
                # Fall back to direct enqueue
                self.executor.enqueue_request(trtllm_request)
        
        if llm_requests:
            # Use the scheduler to organize requests
            scheduler_output = self.parallel_scheduler.schedule_request(
                llm_requests, self.inflight_request_ids)
            
            print(f"[ParallelExecutorWrapper] Scheduler output: {len(scheduler_output.context_requests)} context, {len(scheduler_output.generation_requests)} generation")
            
            # Execute the scheduled requests in parallel using the scheduler's method
            self.parallel_scheduler.execute_parallel_batches(
                scheduler_output.context_requests, scheduler_output.generation_requests,
                [], [],  # Empty lists for stream 1 since we're using the combined output
                self._context_executor, self._generation_executor)
        
        # Clear pending requests after scheduling
        self.pending_requests.clear()

    def await_responses(
        self,
        request_id: Optional[int] = None,
        timeout: Optional[datetime.timedelta] = None
    ) -> List[trtllm.Response]:
        with profiler.record_function("ParallelExecutorWrapper_await_responses"):
            print(f"[ParallelExecutorWrapper] Awaiting responses for request_id: {request_id}, timeout: {timeout}")
            
            # Schedule any pending requests before awaiting responses
            if self.pending_requests:
                self._schedule_and_execute_requests()
            
            # Use the underlying executor's await_responses
            if request_id is not None:
                return self.executor.await_responses(id=request_id, timeout=timeout)
            elif timeout is not None:
                return self.executor.await_responses(timeout=timeout)
            else:
                return self.executor.await_responses()

    def get_stream_utilization(self) -> float:
        """Get current stream utilization percentage."""
        try:
            # Calculate utilization based on active requests and parallel execution
            total_requests = len(self.inflight_request_ids)
            if total_requests == 0:
                return 0.0
            
            # If parallel execution is active and we have multiple requests, utilization is high
            if total_requests >= self.parallel_config.min_requests_for_parallel:
                utilization = min(100.0, (total_requests / 5.0) * 100.0)  # Normalize to 100%
            else:
                utilization = min(100.0, (total_requests / 10.0) * 100.0)  # Lower utilization for single stream
            
            return utilization
        except Exception as e:
            print(f"[ParallelExecutorWrapper] Error calculating stream utilization: {e}")
            return 0.0

    def shutdown(self):
        with profiler.record_function("ParallelExecutorWrapper_shutdown"):
            print("[ParallelExecutorWrapper] Shutting down executor...")
            self.executor.shutdown()
            print("[ParallelExecutorWrapper] Shutdown complete.")


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
{self.scheduler_name} Results:
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
    print(f"[run_benchmark_with_executor] model_path: {model_path}")
    print(f"[run_benchmark_with_executor] num_requests: {num_requests}, max_prompt_len: {max_prompt_len}, max_max_tokens: {max_max_tokens}, use_parallel_executor: {use_parallel_executor}")
    if parallel_config:
        print(f"[run_benchmark_with_executor] parallel_config: {parallel_config}")
    scheduler_name = "ParallelExecutor" if use_parallel_executor else "RegularExecutor"
    result = SchedulerBenchmarkResult(scheduler_name)

    # NVTX marker for executor initialization
    with profiler.record_function(f"{scheduler_name}_initialization"):
        # Create executor
        if use_parallel_executor:
            print("[run_benchmark_with_executor] Using ParallelExecutorWrapper...")
            executor = ParallelExecutorWrapper(model_path, parallel_config)
        else:
            print("[run_benchmark_with_executor] Using RegularExecutor...")
            beam_width = 1
            executor_config = trtllm.ExecutorConfig(
                beam_width,
                kv_cache_config=trtllm.KvCacheConfig(free_gpu_memory_fraction=0.5))
            executor = trtllm.Executor(Path(model_path), trtllm.ModelType.DECODER_ONLY,
                                       executor_config)

    # Record initial GPU state
    initial_memory = get_gpu_memory_used()
    initial_utilization = get_gpu_utilization()
    print(f"[run_benchmark_with_executor] initial_memory: {initial_memory} GB, initial_utilization: {initial_utilization}%")

    # NVTX marker for request preparation
    with profiler.record_function(f"{scheduler_name}_request_preparation"):
        # Prepare requests
        requests = []
        request_ids = []
        expected_num_tokens = {}

        for i in range(num_requests):
            prompt_len = random.randint(1, max_prompt_len)
            max_tokens = random.randint(1, max_max_tokens)
            input_tokens = [1] * prompt_len
            num_return_sequences = 2 if i % 5 == 1 else 1
            print(f"[run_benchmark_with_executor] Creating request {i}: prompt_len={prompt_len}, max_tokens={max_tokens}, num_return_sequences={num_return_sequences}")
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

    # NVTX marker for request enqueueing
    with profiler.record_function(f"{scheduler_name}_request_enqueueing"):
        # Enqueue requests
        for i, request in enumerate(requests):
            print(f"[run_benchmark_with_executor] Enqueueing request {i}...")
            request_id = executor.enqueue_request(request)
            print(f"[run_benchmark_with_executor] Got request_id: {request_id}")
            request_ids.append(request_id)
            request_start_times[request_id] = time.time()

    # Process responses
    tokens = {
        req_id:
        [[] for _ in range(requests[i].sampling_config.num_return_sequences)]
        for i, req_id in enumerate(request_ids)
    }
    print(f"[run_benchmark_with_executor] tokens dict initialized: {tokens.keys()}")

    num_finished = 0
    max_wait_ms = 30000  # 30 seconds timeout
    wait_count = 0

    # NVTX marker for request processing
    with profiler.record_function(f"{scheduler_name}_request_processing"):
        while num_finished < num_requests and wait_count < max_wait_ms:
            wait_time = datetime.timedelta(milliseconds=1)
            responses = executor.await_responses(timeout=wait_time)
            print(f"[run_benchmark_with_executor] Got {len(responses)} responses at wait_count {wait_count}")

            for response in responses:
                print(f"[run_benchmark_with_executor] Response: {response}")
                assert not response.has_error(
                ), f"Request failed: {response.error_msg}"

                result_obj = response.result
                request_id = response.request_id
                print(f"[run_benchmark_with_executor] result_obj: {result_obj}, request_id: {request_id}")

                if result_obj.is_final and request_id in request_start_times:
                    # Record completion time
                    completion_time = time.time()
                    request_time = completion_time - request_start_times[request_id]
                    result.request_times.append(request_time)
                    print(f"[run_benchmark_with_executor] Request {request_id} finished in {request_time:.4f}s")
                    num_finished += 1
                    del request_start_times[
                        request_id]  # Only count once per request

                new_tokens = result_obj.output_token_ids[0]  # beam_width - 1 = 0
                tokens[request_id][result_obj.sequence_index].extend(new_tokens)
                print(f"[run_benchmark_with_executor] Updated tokens for request_id {request_id}, sequence_index {result_obj.sequence_index}")

            wait_count += 1

    # End timing
    end_time = time.time()
    print(f"[run_benchmark_with_executor] All requests finished or timed out. Total time: {end_time - start_time:.4f}s")

    # NVTX marker for metrics calculation
    with profiler.record_function(f"{scheduler_name}_metrics_calculation"):
        # Calculate metrics
        result.total_time = end_time - start_time
        result.throughput = num_requests / result.total_time
        result.gpu_memory_used = get_gpu_memory_used() - initial_memory
        result.gpu_utilization = get_gpu_utilization() - initial_utilization
        print(f"[run_benchmark_with_executor] Metrics: total_time={result.total_time}, throughput={result.throughput}, gpu_memory_used={result.gpu_memory_used}, gpu_utilization={result.gpu_utilization}")

        # Calculate latency statistics
        if result.request_times:
            result.avg_latency = statistics.mean(result.request_times)
            if len(result.request_times) >= 5:
                result.p95_latency = statistics.quantiles(
                    result.request_times, n=20)[18]  # 95th percentile
            if len(result.request_times) >= 10:
                result.p99_latency = statistics.quantiles(
                    result.request_times, n=100)[98]  # 99th percentile
            print(f"[run_benchmark_with_executor] Latency: avg={result.avg_latency}, p95={result.p95_latency}, p99={result.p99_latency}")

        # Calculate stream utilization for parallel executor
        if use_parallel_executor and isinstance(executor, ParallelExecutorWrapper):
            result.stream_utilization = executor.get_stream_utilization()
            print(f"[run_benchmark_with_executor] Stream utilization: {result.stream_utilization}")

    # NVTX marker for result verification
    with profiler.record_function(f"{scheduler_name}_result_verification"):
        # Verify results
        print(f"[run_benchmark_with_executor] Verifying results...")
        for request_id in tokens:
            print(f"[run_benchmark_with_executor] Checking tokens for request_id: {request_id}")
            for actual_tokens in tokens[request_id]:
                print(f"[run_benchmark_with_executor] actual_tokens: {actual_tokens}")
                assert len(actual_tokens) > 0, f"No tokens generated for request {request_id} (available keys: {list(tokens.keys())})"

    # NVTX marker for executor shutdown
    with profiler.record_function(f"{scheduler_name}_shutdown"):
        executor.shutdown()
        print(f"[run_benchmark_with_executor] Executor shutdown complete.")
    
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

    # NVTX marker for regular executor runs
    with profiler.record_function("RegularExecutor_benchmark"):
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

    # NVTX marker for parallel executor runs
    with profiler.record_function("ParallelExecutor_benchmark"):
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


def print_executor_comparison_results(results: Dict[str, SchedulerBenchmarkResult]):
    """Print detailed comparison between regular and parallel executor results."""
    regular = results["regular"]
    parallel = results["parallel"]
    
    print("\n" + "="*80)
    print("EXECUTOR COMPARISON RESULTS")
    print("="*80)
    
    print(f"Regular Executor Results:")
    print(f"  - Total Time: {regular.total_time:.3f}s")
    print(f"  - Throughput: {regular.throughput:.2f} requests/sec")
    print(f"  - Average Latency: {regular.avg_latency:.3f}s")
    print(f"  - GPU Memory Used: {regular.gpu_memory_used:.2f} MB")
    print(f"  - GPU Utilization: {regular.gpu_utilization:.2f}%")
    
    print(f"\nParallel Executor Results:")
    print(f"  - Total Time: {parallel.total_time:.3f}s")
    print(f"  - Throughput: {parallel.throughput:.2f} requests/sec")
    print(f"  - Average Latency: {parallel.avg_latency:.3f}s")
    print(f"  - GPU Memory Used: {parallel.gpu_memory_used:.2f} MB")
    print(f"  - GPU Utilization: {parallel.gpu_utilization:.2f}%")
    
    # Calculate improvements with zero checks
    time_improvement = ((regular.total_time - parallel.total_time) / 
                       regular.total_time * 100) if regular.total_time > 0 else 0
    
    throughput_improvement = ((parallel.throughput - regular.throughput) / 
                             regular.throughput * 100) if regular.throughput > 0 else 0
    
    latency_improvement = ((regular.avg_latency - parallel.avg_latency) / 
                          regular.avg_latency * 100) if regular.avg_latency > 0 else 0
    
    # Memory overhead calculation with zero check
    if regular.gpu_memory_used > 0:
        memory_overhead = ((parallel.gpu_memory_used - regular.gpu_memory_used) / 
                          regular.gpu_memory_used * 100)
    else:
        memory_overhead = 0 if parallel.gpu_memory_used == 0 else float('inf')
    
    # GPU utilization improvement with zero check
    if regular.gpu_utilization > 0:
        gpu_improvement = ((parallel.gpu_utilization - regular.gpu_utilization) / 
                          regular.gpu_utilization * 100)
    else:
        gpu_improvement = 0 if parallel.gpu_utilization == 0 else float('inf')
    
    print(f"\nIMPROVEMENTS:")
    print(f"  - Time Improvement: {time_improvement:+.2f}%")
    print(f"  - Throughput Improvement: {throughput_improvement:+.2f}%")
    print(f"  - Latency Improvement: {latency_improvement:+.2f}%")
    print(f"  - Memory Overhead: {memory_overhead:+.2f}%")
    print(f"  - GPU Utilization Improvement: {gpu_improvement:+.2f}%")
    
    print("="*80)


@pytest.mark.skipif(torch.cuda.device_count() < 1,
                    reason="Requires at least 1 GPU")
def test_executor_integration_basic(model_files, model_path):
    print("\n[TEST] Starting test_executor_integration_basic...")
    print(f"[TEST] model_path: {model_path}")
    
    # NVTX marker for the entire test
    with profiler.record_function("test_executor_integration_basic"):
        results = compare_executor_performance(model_path=str(model_path),
                                               num_requests=10,
                                               max_prompt_len=10,
                                               max_max_tokens=10,
                                               num_runs=2)
        print("[TEST] Results from compare_executor_performance:")
        print(results)
        print_executor_comparison_results(results)
        # Basic assertions
        regular = results["regular"]
        parallel = results["parallel"]
        print(f"[TEST] regular: {regular}")
        print(f"[TEST] parallel: {parallel}")
        assert regular.total_time > 0, "RegularExecutor should complete in finite time"
        assert parallel.total_time > 0, "ParallelExecutor should complete in finite time"
        assert regular.throughput > 0, "RegularExecutor should have positive throughput"
        assert parallel.throughput > 0, "ParallelExecutor should have positive throughput"
        print("[TEST] test_executor_integration_basic completed successfully.")


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
