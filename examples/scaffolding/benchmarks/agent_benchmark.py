"""Agent benchmark implementations for scaffolding benchmarks."""

import asyncio
import sys

from openai import AsyncOpenAI

from tensorrt_llm.scaffolding import MCPWorker, QueryCollector, TRTOpenaiWorker
from tensorrt_llm.scaffolding.benchmark import ScaffoldingBenchRequest, async_scaffolding_benchmark
from tensorrt_llm.scaffolding.contrib.open_deep_research import (
    create_open_deep_research_scaffolding_llm,
)
from tensorrt_llm.scaffolding.load_generation_strategy import (
    ConcurrentStrategy,
    PoissonWarmupStrategy,
)

from .benchmark_utils import (
    load_prompts_from_json,
    print_benchmark_results,
    print_lock,
    shutdown_llm,
)


async def run_agent_benchmark_core(
    llm, prompts, concurrency, benchmark_name, args, use_poisson_arrival: bool = True
):
    """Core agent benchmark logic that can be reused.

    Args:
        llm: The ScaffoldingLlm instance to use
        prompts: List of prompts to benchmark
        concurrency: Number of concurrent requests
        benchmark_name: Name for the benchmark (used in output)
        args: Command line arguments

    Returns:
        Tuple of (results, requests_start_time, requests_execution_time, total_time)
    """
    task_collection_types = {}
    requests = [ScaffoldingBenchRequest(prompt=prompt) for prompt in prompts]

    # Select strategy based on Poisson arrival flag
    if use_poisson_arrival and getattr(args, "enable_poisson_arrival", False):
        strategy = PoissonWarmupStrategy(
            num_requests=len(requests),
            warmup_window=getattr(args, "poisson_warmup_window", 60.0),
            max_concurrency=concurrency,
            random_seed=getattr(args, "poisson_arrival_seed", 42),
        )
        print(f"  Using Poisson warmup arrival: {strategy}")
    else:
        strategy = ConcurrentStrategy(concurrency=concurrency)

    (
        results,
        requests_start_time,
        requests_execution_time,
        total_time,
    ) = await async_scaffolding_benchmark(llm, task_collection_types, requests, strategy=strategy)

    print_benchmark_results(
        benchmark_name, results, requests_start_time, requests_execution_time, total_time
    )

    if args.enable_query_collector:
        QueryCollector.get_global_info()
        with print_lock:
            print(f"Query info dumped to query_result.json! ({benchmark_name})")

    return results, requests_start_time, requests_execution_time, total_time


async def create_agent_resources(args):
    """Create isolated resources for an agent benchmark.

    Returns:
        Tuple of (llm, mcp_worker, generation_worker) for cleanup
    """
    client = AsyncOpenAI(api_key=args.openai_api_key, base_url=args.base_url)
    generation_worker = TRTOpenaiWorker(
        client, args.model, getattr(args, "kv_cache_hint_agent", False)
    )

    mcp_worker = MCPWorker.init_with_urls(["http://0.0.0.0:8082/sse"])
    await mcp_worker.init_in_asyncio_event_loop()

    llm = create_open_deep_research_scaffolding_llm(
        generation_worker,
        mcp_worker,
        max_tokens=args.max_tokens_agent,
        max_parallel_requests=args.max_parallel_requests,
        enable_statistics=args.enable_statistics,
        enable_query_collector=args.enable_query_collector,
    )

    return llm, mcp_worker, generation_worker


async def cleanup_agent_resources(llm, mcp_worker):
    """Cleanup agent resources."""
    # Shutdown MCP worker first and wait for background tasks to complete
    await mcp_worker.async_shutdown()

    # Shutdown LLM and all workers
    await shutdown_llm(llm)


async def async_agent_benchmark(args):
    """Normal agent benchmark.

    Returns:
        Tuple of (results, requests_start_time, requests_execution_time, total_time)
    """
    llm, mcp_worker, _ = await create_agent_resources(args)
    prompts = load_prompts_from_json(args.agent_prompt_num)
    try:
        return await run_agent_benchmark_core(
            llm, prompts, args.normal_agent_concurrency, "Agent-Normal", args
        )
    finally:
        await cleanup_agent_resources(llm, mcp_worker)


async def async_burst_agent_benchmark(args):
    """Burst agent benchmark that simulates sudden traffic spike.

    Returns:
        Tuple of (results, requests_start_time, requests_execution_time, total_time)
    """
    with print_lock:
        print(f"\n[Burst] Waiting {args.burst_delay}s before starting burst traffic...")
        sys.stdout.flush()

    await asyncio.sleep(args.burst_delay)

    llm, mcp_worker, _ = await create_agent_resources(args)

    with print_lock:
        print(
            f"\n[Burst] Starting burst traffic with "
            f"{args.burst_prompt_num} prompts, "
            f"concurrency={args.burst_agent_concurrency}"
        )
        sys.stdout.flush()

    try:
        prompts = load_prompts_from_json(args.burst_prompt_num)
        return await run_agent_benchmark_core(
            llm,
            prompts,
            args.burst_agent_concurrency,
            "Agent-Burst",
            args,
            use_poisson_arrival=False,
        )
    finally:
        await cleanup_agent_resources(llm, mcp_worker)
