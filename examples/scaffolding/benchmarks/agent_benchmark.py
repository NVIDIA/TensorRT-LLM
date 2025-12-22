"""Agent benchmark implementations for scaffolding benchmarks."""

import asyncio
import sys

from openai import AsyncOpenAI

from tensorrt_llm.scaffolding import (
    MCPWorker,
    QueryCollector,
    TaskMetricsCollector,
    TRTOpenaiWorker,
)
from tensorrt_llm.scaffolding.benchmark import ScaffoldingBenchRequest, async_scaffolding_benchmark
from tensorrt_llm.scaffolding.contrib.DeepResearch import create_open_deep_research_scaffolding_llm
from tensorrt_llm.scaffolding.load_generation_strategy import ConcurrentStrategy

from .benchmark_utils import load_prompts_from_json, print_benchmark_results, print_lock


async def run_agent_benchmark_core(llm, prompts, concurrency, benchmark_name, args, times=1):
    """Core agent benchmark logic that can be reused.

    Args:
        llm: The ScaffoldingLlm instance to use
        prompts: List of prompts to benchmark
        concurrency: Number of concurrent requests
        benchmark_name: Name for the benchmark (used in output)
        args: Command line arguments
        times: Number of times to run the benchmark
    """
    task_collection_types = {}
    requests = [ScaffoldingBenchRequest(prompt=prompt) for prompt in prompts]
    strategy = ConcurrentStrategy(concurrency=concurrency)

    for i in range(times):
        (
            results,
            requests_start_time,
            requests_execution_time,
            total_time,
        ) = await async_scaffolding_benchmark(
            llm, task_collection_types, requests, strategy=strategy
        )

        print_benchmark_results(
            benchmark_name, results, requests_start_time, requests_execution_time, total_time
        )

        with print_lock:
            if args.enable_statistics:
                print(f"Task metrics summary: {benchmark_name}")
                TaskMetricsCollector.print_summary()

        if args.enable_query_collector:
            QueryCollector.get_global_info()
            print(f"Query info dumped to query_result.json! ({benchmark_name})")

        if args.export_task_metrics_path is not None:
            TaskMetricsCollector.export_to_json(args.export_task_metrics_path)

    return results


async def create_agent_resources(args):
    """Create isolated resources for an agent benchmark.

    Returns:
        Tuple of (llm, mcp_worker, generation_worker) for cleanup
    """
    client = AsyncOpenAI(api_key=args.openai_api_key, base_url=args.base_url)
    generation_worker = TRTOpenaiWorker(client, args.model, args.kv_cache_hint_enabled)

    mcp_worker = MCPWorker.init_with_urls(["http://0.0.0.0:8082/sse"])
    await mcp_worker.init_in_asyncio_event_loop()

    llm = create_open_deep_research_scaffolding_llm(
        generation_worker,
        mcp_worker,
        max_tokens=args.max_tokens,
        max_parallel_requests=args.max_parallel_requests,
        enable_statistics=args.enable_statistics,
        enable_query_collector=args.enable_query_collector,
    )

    return llm, mcp_worker, generation_worker


async def cleanup_agent_resources(llm, mcp_worker, generation_worker):
    """Cleanup agent resources."""
    # Shutdown MCP worker and wait for background tasks to complete
    await mcp_worker.shutdown()

    llm.shutdown()
    generation_worker.shutdown()

    if not llm.own_loop:
        await llm.main_loop_stop_event.wait()


async def async_agent_benchmark(args):
    """Normal agent benchmark."""
    llm, mcp_worker, generation_worker = await create_agent_resources(args)
    prompts = load_prompts_from_json(args.agent_prompt_num)
    try:
        await run_agent_benchmark_core(
            llm, prompts, args.normal_agent_concurrency, "Agent-Normal", args, times=args.times
        )
    finally:
        await cleanup_agent_resources(llm, mcp_worker, generation_worker)


async def async_burst_agent_benchmark(args):
    """Burst agent benchmark that simulates sudden traffic spike."""
    with print_lock:
        print(f"\n[Burst] Waiting {args.burst_delay}s before starting burst traffic...")
        sys.stdout.flush()

    await asyncio.sleep(args.burst_delay)

    llm, mcp_worker, generation_worker = await create_agent_resources(args)

    with print_lock:
        print(
            f"\n[Burst] Starting burst traffic with "
            f"{args.burst_prompt_num} prompts, "
            f"concurrency={args.burst_agent_concurrency}"
        )
        sys.stdout.flush()

    try:
        prompts = load_prompts_from_json(args.burst_prompt_num)
        await run_agent_benchmark_core(
            llm, prompts, args.burst_agent_concurrency, "Agent-Burst", args, times=1
        )
    finally:
        await cleanup_agent_resources(llm, mcp_worker, generation_worker)
