import json
import os
from pathlib import Path

import click
from utils.benchmarkers import gptSessionBenchmarker
from utils.dataclasses import BenchmarkConfig, BenchmarkResults


@click.command("static")
@click.option(
    "--batch",
    required=True,
    type=int,
    help="Batch size to build and run the static benchmark with.",
)
@click.option("--isl",
              type=int,
              required=True,
              help="Input sequence length (in tokens).")
@click.option("--osl",
              type=int,
              required=True,
              help="Output sequence length (in tokens).")
@click.option(
    "--gpt-session-path",
    "-b",
    type=click.Path(),
    default=Path(os.path.dirname(os.path.realpath(__file__)), "../../..",
                 "cpp/build/benchmarks/gptSessionBenchmark").absolute(),
    help="Path to TRT-LLM gptSession benchmark binary.")
@click.option("--max-tokens-in-kv-cache",
              type=int,
              default=None,
              help="Maximum number of tokens to store in KV cache")
@click.option(
    "--kv-cache-mem-percent",
    type=float,
    default=0.9,
    help="The percentage of free memory that the KV Cache is allowed to occupy.",
)
@click.option("--warm-up-runs",
              type=int,
              default=2,
              help="Number of warm up runs before benchmarking")
@click.option("--num-runs",
              type=int,
              default=10,
              help="Number of times to run benchmark")
@click.option("--duration",
              type=int,
              default=60,
              help="Minimum duration of iteration to measure, in seconds")
@click.pass_obj
def static_benchmark(benchmark_cfg: BenchmarkConfig, batch: int, isl: int,
                     osl: int, gpt_session_path: Path, warm_up_runs: int,
                     num_runs: int, duration: int, max_tokens_in_kv_cache: int,
                     kv_cache_mem_percent: float):
    """Run a static benchmark with a fixed batch size, ISL, and OSL."""
    if max_tokens_in_kv_cache is None:
        max_tokens_in_kv_cache = batch * isl

    benchmarker = gptSessionBenchmarker(
        benchmark_cfg,
        gpt_session_path,
        batch,
        isl,
        osl,
        warm_up_runs,
        num_runs,
        duration,
        max_tokens_in_kv_cache,
        kv_cache_mem_percent,
    )

    print(f"Building TRT-LLM engine for '{benchmark_cfg.model}'...")
    benchmarker.build()

    print("Build complete. Running benchmark...")
    result: BenchmarkResults = benchmarker.benchmark()

    print(f"JSON: {json.dumps(result.model_dump())}")
    print(result.get_summary(benchmarker.config))
