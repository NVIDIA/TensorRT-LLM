"""
Example demonstrating how to use the ParallelStreamScheduler for running two batches in parallel.

This example shows how to:
1. Configure parallel execution
2. Create a parallel stream scheduler
3. Schedule requests for parallel execution
4. Execute batches on different CUDA streams
"""

from typing import List

import torch

from .llm_request import LlmRequest
from .scheduler import (BindCapacityScheduler, BindMicroBatchScheduler,
                        ParallelExecutionConfig, ParallelStreamScheduler)


def create_parallel_scheduler_example():
    """
    Example of creating and using a ParallelStreamScheduler.
    """

    # 1. Create configuration for parallel execution
    config = ParallelExecutionConfig(
        enable_parallel_execution=True,
        load_balancing_strategy="smart",  # Use smart load balancing
        min_requests_for_parallel=2,  # Enable parallel execution with 2+ requests
        stream_priority=0,  # Normal priority for streams
        enable_stream_synchronization=True,
        context_generation_fusion=False  # Keep context and generation separate
    )

    # 2. Create underlying schedulers
    # Note: In a real implementation, you would need actual KV cache manager and other components
    max_num_requests = 10
    max_batch_size = 8
    max_num_tokens = 2048

    # Create capacity scheduler (you'll need to provide actual KV cache manager)
    capacity_scheduler = BindCapacityScheduler(
        max_num_requests=max_num_requests,
        kv_cache_manager=None,  # You'll need to provide this
        two_step_lookahead=False)

    # Create micro-batch scheduler
    micro_batch_scheduler = BindMicroBatchScheduler(
        max_batch_size=max_batch_size, max_num_tokens=max_num_tokens)

    # 3. Create parallel stream scheduler
    parallel_scheduler = ParallelStreamScheduler(
        capacity_scheduler=capacity_scheduler,
        micro_batch_scheduler=micro_batch_scheduler,
        config=config)

    return parallel_scheduler


def execute_parallel_batches_example(
        parallel_scheduler: ParallelStreamScheduler):
    """
    Example of executing batches in parallel.
    """

    # Mock context and generation executor functions
    def context_executor(context_requests: List[LlmRequest],
                         generation_requests: List[LlmRequest]):
        """Mock context phase executor."""
        print(
            f"Executing context phase with {len(context_requests)} context requests and {len(generation_requests)} generation requests"
        )
        # In real implementation, this would run the actual context phase
        torch.cuda.synchronize()  # Simulate some work

    def generation_executor(context_requests: List[LlmRequest],
                            generation_requests: List[LlmRequest]):
        """Mock generation phase executor."""
        print(
            f"Executing generation phase with {len(context_requests)} context requests and {len(generation_requests)} generation requests"
        )
        # In real implementation, this would run the actual generation phase
        torch.cuda.synchronize()  # Simulate some work

    # Example request lists (in real implementation, these would come from scheduling)
    stream_0_context = []  # Mock context requests for stream 0
    stream_0_generation = []  # Mock generation requests for stream 0
    stream_1_context = []  # Mock context requests for stream 1
    stream_1_generation = []  # Mock generation requests for stream 1

    # Execute batches in parallel
    parallel_scheduler.execute_parallel_batches(
        stream_0_context=stream_0_context,
        stream_0_generation=stream_0_generation,
        stream_1_context=stream_1_context,
        stream_1_generation=stream_1_generation,
        context_executor=context_executor,
        generation_executor=generation_executor)


def execute_parallel_batches_async_example(
        parallel_scheduler: ParallelStreamScheduler):
    """
    Example of executing batches asynchronously.
    """

    # Mock context and generation executor functions
    def context_executor(context_requests: List[LlmRequest],
                         generation_requests: List[LlmRequest]):
        """Mock context phase executor."""
        print(
            f"Executing context phase with {len(context_requests)} context requests and {len(generation_requests)} generation requests"
        )
        # In real implementation, this would run the actual context phase

    def generation_executor(context_requests: List[LlmRequest],
                            generation_requests: List[LlmRequest]):
        """Mock generation phase executor."""
        print(
            f"Executing generation phase with {len(context_requests)} context requests and {len(generation_requests)} generation requests"
        )
        # In real implementation, this would run the actual generation phase

    # Example request lists
    stream_0_context = []  # Mock context requests for stream 0
    stream_0_generation = []  # Mock generation requests for stream 0
    stream_1_context = []  # Mock context requests for stream 1
    stream_1_generation = []  # Mock generation requests for stream 1

    # Execute batches asynchronously
    stream_0_event, stream_1_event = parallel_scheduler.execute_parallel_batches_async(
        stream_0_context=stream_0_context,
        stream_0_generation=stream_0_generation,
        stream_1_context=stream_1_context,
        stream_1_generation=stream_1_generation,
        context_executor=context_executor,
        generation_executor=generation_executor)

    # Wait for completion when needed
    stream_0_event.wait()
    stream_1_event.wait()
    print("Both streams completed")


def load_balancing_strategies_example():
    """
    Example showing different load balancing strategies.
    """

    # Round-robin strategy
    config_round_robin = ParallelExecutionConfig(
        load_balancing_strategy="round_robin", enable_parallel_execution=True)

    # Smart strategy (default)
    config_smart = ParallelExecutionConfig(load_balancing_strategy="smart",
                                           enable_parallel_execution=True)

    # Balanced strategy
    config_balanced = ParallelExecutionConfig(
        load_balancing_strategy="balanced", enable_parallel_execution=True)

    print("Available load balancing strategies:")
    print("- round_robin: Simple alternating assignment")
    print("- smart: Separate context and generation requests, then distribute")
    print("- balanced: Use workload estimation for optimal distribution")


def main():
    """
    Main example function.
    """
    print("ParallelStreamScheduler Example")
    print("=" * 40)

    # Create parallel scheduler
    parallel_scheduler = create_parallel_scheduler_example()

    # Get streams and events
    stream_0, stream_1 = parallel_scheduler.get_streams()
    event_0, event_1 = parallel_scheduler.get_events()

    print(f"Created parallel scheduler with streams: {stream_0}, {stream_1}")
    print(f"Events for synchronization: {event_0}, {event_1}")

    # Show load balancing strategies
    load_balancing_strategies_example()

    # Execute examples (these are mocked since we don't have real requests)
    print(
        "\nNote: The following examples are mocked since we don't have real LLM requests"
    )
    print("In a real implementation, you would:")
    print("1. Create actual LlmRequest objects")
    print("2. Provide a real KV cache manager")
    print("3. Implement actual context and generation executors")
    print("4. Use the scheduler in your inference pipeline")


if __name__ == "__main__":
    main()
