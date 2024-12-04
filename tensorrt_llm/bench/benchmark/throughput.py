from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
from pathlib import Path

import click
from click_option_group import optgroup

from tensorrt_llm.bench.benchmark.utils.asynchronous import async_benchmark

# isort: off
from tensorrt_llm.bench.benchmark.utils.general import (get_executor_requests,
                                                        get_settings_from_engine
                                                        )
# isort: on
from tensorrt_llm.bench.benchmark.utils.multiproc import ThroughputBenchmark
from tensorrt_llm.bench.dataclasses.configuration import RuntimeConfig
from tensorrt_llm.bench.dataclasses.enums import IFBSchedulingPolicy
from tensorrt_llm.bench.dataclasses.general import BenchmarkEnvironment
from tensorrt_llm.bench.utils.data import (create_dataset_from_stream,
                                           initialize_tokenizer)
from tensorrt_llm.logger import logger


@click.command(name="throughput")
@optgroup.group("Engine run configuration.",
                help="Runtime settings for executing a TensorRT-LLM engine.")
@optgroup.option(
    "--engine_dir",
    type=click.Path(exists=True,
                    readable=True,
                    path_type=Path,
                    resolve_path=True),
    required=True,
    help="Path to a serialized TRT-LLM engine.",
)
@optgroup.option(
    "--max_batch_size",
    type=int,
    help="Maximum runtime batch size to run the engine with.",
)
@optgroup.option(
    "--max_num_tokens",
    type=int,
    help="Maximum runtime tokens that an engine can accept.",
)
@optgroup.option(
    "--beam_width",
    type=int,
    default=1,
    help="Number of search beams.",
)
@optgroup.option(
    "--kv_cache_free_gpu_mem_fraction",
    type=float,
    default=.90,
    help="The percentage of memory to use for KV Cache after model load.",
)
@optgroup.group(
    "Engine Input Configuration",
    help="Input configuration for driving the engine.",
)
@optgroup.option(
    "--dataset",
    type=click.Path(exists=True,
                    readable=True,
                    path_type=Path,
                    resolve_path=True),
    default=None,
    help="Pass in a dataset file for parsing instead of stdin.",
)
@optgroup.option(
    "--request_rate",
    type=int,
    default=-1,
    help="Desired input request rate (number of messages per second).",
    hidden=True,
)
@optgroup.option(
    "--num_requests",
    type=int,
    default=0,
    help="Number of requests to cap benchmark run at. Minimum between value and"
    "length of dataset.",
)
@click.option(
    "--streaming",
    is_flag=True,
    default=False,
    help="Enable streaming mode for requests.",
)
@click.option(
    "--iteration_log",
    type=click.Path(dir_okay=False,
                    writable=True,
                    readable=False,
                    path_type=Path,
                    resolve_path=True),
    required=False,
    help="Path where iteration stats should be written to.",
)
@click.pass_obj
def throughput_command(
    bench_env: BenchmarkEnvironment,
    **params,
) -> None:
    """Run a throughput test on a TRT-LLM engine."""

    TRTLLM_BENCH_EXPERIMENTAL = os.environ.get("TRTLLM_BENCH_EXPERIMENT", False)

    logger.set_level("info")
    logger.info("Preparing to run throughput benchmark...")
    # Parameters from CLI
    # Model, experiment, and engine params
    dataset_path: Path = params.pop("dataset")
    request_rate: int = params.pop("request_rate")
    num_requests: int = params.pop("num_requests")
    model: str = bench_env.model
    engine_dir: Path = params.pop("engine_dir")
    iteration_log: Path = params.pop("iteration_log")

    # Engine configuration parsing
    exec_settings, build_cfg = get_settings_from_engine(engine_dir)
    exec_settings["model"] = model
    engine_bs = exec_settings["settings_config"]["max_batch_size"]
    engine_tokens = exec_settings["settings_config"]["max_num_tokens"]
    engine_max_seq_len = build_cfg["max_seq_len"]

    # Check that we are not using a low latency engine
    # Right now, this is based on max batch size.
    if engine_bs == 1:
        raise ValueError(
            "An engine with a batch size greater than 1 should be used for "
            "throughput benchmarking. Exiting.")

    # Runtime Options
    runtime_max_bs = params.pop("max_batch_size")
    runtime_max_bs = runtime_max_bs if runtime_max_bs else engine_bs
    runtime_max_tokens = params.pop("max_num_tokens")
    runtime_max_tokens = runtime_max_bs if runtime_max_tokens else engine_tokens
    kv_cache_percent = params.pop("kv_cache_free_gpu_mem_fraction")
    beam_width = params.pop("beam_width")
    streaming: bool = params.pop("streaming")

    # Update configuration with runtime options
    exec_settings["settings_config"]["kv_cache_percent"] = kv_cache_percent
    exec_settings["settings_config"]["max_batch_size"] = runtime_max_bs
    exec_settings["settings_config"]["max_num_tokens"] = runtime_max_tokens
    exec_settings["settings_config"]["beam_width"] = beam_width
    exec_settings["settings_config"][
        "scheduler_policy"] = IFBSchedulingPolicy.NO_EVICT

    # Dynamic runtime features.
    exec_settings["settings_config"]["dynamic_max_batch_size"] = True

    # Construct the runtime configuration dataclass.
    runtime_config = RuntimeConfig(**exec_settings)

    # Initialize the HF tokenizer for the specified model.
    tokenizer = initialize_tokenizer(bench_env.model)

    # Dataset Loading and Preparation
    with open(dataset_path, "r") as dataset:
        metadata, requests = create_dataset_from_stream(
            tokenizer, dataset, num_requests=num_requests)
    # TODO: Verify that the engine can handle the max/min ISL/OSL.
    if metadata.max_sequence_length > engine_max_seq_len:
        raise RuntimeError(
            f"Engine supports a max sequence of {engine_max_seq_len}. Provided "
            "dataset contains a maximum sequence of "
            f"{metadata.max_sequence_length}. Please rebuild a new engine to"
            "support this dataset.")

    if TRTLLM_BENCH_EXPERIMENTAL:
        asyncio.run(async_benchmark(runtime_config, requests, streaming))
    else:
        # Dataset Loading and Preparation
        executor_requests = get_executor_requests(
            requests,
            streaming,
            eos_id=-1,
            pad_id=-1,
        )
        del requests

    logger.info("Setting up benchmarker and infrastructure.")
    new_request_queue = mp.Queue()
    response_queue = mp.Queue()
    benchmark = ThroughputBenchmark(
        dataset=executor_requests,
        request_rate=request_rate,
        runtime_cfg=runtime_config,
        request_queue=new_request_queue,
        response_queue=response_queue,
        streaming=streaming,
        iteration_log=iteration_log,
    )

    try:
        logger.info("Ready to start benchmark.")
        benchmark.start_benchmark()
        benchmark.wait()
        benchmark.stop_benchmark()
        benchmark.dump_extra_stats()
        benchmark.report_statistics()
    except KeyboardInterrupt:
        benchmark.stop_benchmark()
    finally:
        benchmark.shutdown()
