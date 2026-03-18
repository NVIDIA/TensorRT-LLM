"""Coder agent benchmark for scaffolding benchmarks.

Runs the Coder agent against Apiary sandboxes with configurable concurrency.
Each concurrent request gets its own isolated sandbox via ApiaryMCPWorker.
"""

import sys

from openai import AsyncOpenAI

from tensorrt_llm.scaffolding import ApiaryMCPWorker, QueryCollector, TRTOpenaiWorker
from tensorrt_llm.scaffolding.benchmark import ScaffoldingBenchRequest, async_scaffolding_benchmark
from tensorrt_llm.scaffolding.contrib.Coder import create_coder_scaffolding_llm
from tensorrt_llm.scaffolding.load_generation_strategy import (
    ConcurrentStrategy,
    PoissonRateStrategy,
    UniformWarmupStrategy,
)

from .benchmark_utils import print_benchmark_results, print_lock, shutdown_llm

DEFAULT_CODER_PROMPTS = [
    "Add comprehensive error handling to all public functions in the project.",
    "Implement a thread-safe LRU cache with configurable capacity.",
    "Write a CLI tool that converts CSV files to JSON with streaming support.",
    "Create a retry decorator with exponential backoff and jitter.",
    "Implement a simple key-value store with TTL-based expiration.",
    "Add input validation and type checking to the API endpoint handlers.",
    "Write a log aggregation utility that merges and deduplicates log entries.",
    "Create a configuration loader that supports YAML, JSON, and environment variables.",
]


def load_coder_prompts(num_prompts: int) -> list[str]:
    """Load prompts for the Coder benchmark from built-in coding tasks."""
    prompts = DEFAULT_CODER_PROMPTS.copy()
    if len(prompts) < num_prompts:
        original = prompts.copy()
        repeat = (num_prompts + len(original) - 1) // len(original)
        prompts = []
        for i in range(repeat):
            for p in original:
                tag = f"[{i}]." if i > 0 else ""
                prompts.append(f"{tag}{p}")
    return prompts[:num_prompts]


async def create_coder_resources(args):
    """Create isolated resources for a Coder benchmark run.

    Returns:
        Tuple of (llm, mcp_worker, generation_worker) for cleanup.
    """
    client = AsyncOpenAI(api_key=args.openai_api_key, base_url=args.base_url)
    generation_worker = TRTOpenaiWorker(
        client, args.model, getattr(args, "kv_cache_hint_agent", False)
    )

    mcp_url = getattr(args, "mcp_url", "http://0.0.0.0:8083/sse")
    max_conns = getattr(args, "coder_max_connections", 200)
    mcp_worker = ApiaryMCPWorker(mcp_url, max_connections=max_conns)

    llm = create_coder_scaffolding_llm(
        generation_worker,
        mcp_worker,
        max_tokens=getattr(args, "max_tokens_agent", 65536),
        max_iterations=getattr(args, "coder_max_iterations", 50),
        max_parallel_requests=getattr(args, "max_parallel_requests", 1024),
        enable_statistics=getattr(args, "enable_statistics", False),
    )

    return llm, mcp_worker, generation_worker


async def cleanup_coder_resources(llm, mcp_worker):
    """Cleanup Coder benchmark resources."""
    await mcp_worker.async_shutdown()
    await shutdown_llm(llm)


async def run_coder_benchmark_core(
    llm, prompts, concurrency, benchmark_name, args, use_poisson_arrival=True
):
    """Core Coder benchmark logic.

    Args:
        llm: The ScaffoldingLlm instance.
        prompts: List of prompts to benchmark.
        concurrency: Number of concurrent requests.
        benchmark_name: Name for the benchmark (used in output).
        args: Command line arguments.

    Returns:
        Tuple of (results, requests_start_time, requests_execution_time, total_time).
    """
    task_collection_types = {}
    requests = [ScaffoldingBenchRequest(prompt=prompt) for prompt in prompts]

    if use_poisson_arrival and getattr(args, "load_mode", "concurrent") == "rate":
        strategy = PoissonRateStrategy(
            rate=getattr(args, "coder_rate", 1.0),
            random_seed=getattr(args, "rate_seed", 42),
        )
    elif getattr(args, "warmup_window", None) is not None:
        strategy = UniformWarmupStrategy(
            num_requests=len(requests),
            warmup_window=args.warmup_window,
            max_concurrency=concurrency,
        )
    else:
        strategy = ConcurrentStrategy(concurrency=concurrency)
    print(f"  Strategy: {strategy}")

    (
        results,
        requests_start_time,
        requests_execution_time,
        total_time,
    ) = await async_scaffolding_benchmark(llm, task_collection_types, requests, strategy=strategy)

    print_benchmark_results(
        benchmark_name,
        results,
        requests_start_time,
        requests_execution_time,
        total_time,
    )

    if getattr(args, "enable_query_collector", False):
        QueryCollector.get_global_info()
        with print_lock:
            print(f"Query info dumped to query_result.json! ({benchmark_name})")

    return results, requests_start_time, requests_execution_time, total_time


async def async_coder_benchmark(args):
    """Run the Coder agent benchmark.

    Returns:
        Tuple of (results, requests_start_time, requests_execution_time, total_time).
    """
    concurrency = getattr(args, "coder_concurrency", 32)
    num_prompts = getattr(args, "coder_prompt_num", 8)

    llm, mcp_worker, _ = await create_coder_resources(args)
    prompts = load_coder_prompts(num_prompts)

    with print_lock:
        print(f"\n[Coder] Starting benchmark with {num_prompts} prompts, concurrency={concurrency}")
        sys.stdout.flush()

    try:
        return await run_coder_benchmark_core(llm, prompts, concurrency, "Coder", args)
    finally:
        await cleanup_coder_resources(llm, mcp_worker)
