import os
from pathlib import Path

import click
from benchmarkers.static import gptSessionBenchmarker
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
                     num_runs: int, duration: int):
    """Run a static benchmark with a fixed batch size, ISL, and OSL."""

    benchmark_cfg.max_batch_size = batch
    benchmarker = gptSessionBenchmarker(
        benchmark_cfg,
        gpt_session_path,
        benchmark_cfg.max_batch_size,
        isl,
        osl,
        warm_up_runs,
        num_runs,
        duration,
        benchmark_cfg.kv_cache_mem_percentage,
    )

    print(f"Building TRT-LLM engine for '{benchmark_cfg.model}'...")
    benchmarker.build()

    print("Build complete. Running benchmark...")
    result: BenchmarkResults = benchmarker.benchmark()

    print(f"JSON: {result.model_dump_json()}")
    print(result.get_summary(benchmarker.config))
