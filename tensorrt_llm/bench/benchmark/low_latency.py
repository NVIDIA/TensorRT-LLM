from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import click
import yaml
from click_option_group import (MutuallyExclusiveOptionGroup, OptionGroup,
                                optgroup)

from tensorrt_llm.bench.benchmark.utils.asynchronous import async_benchmark
from tensorrt_llm.bench.benchmark.utils.general import generate_warmup_dataset
from tensorrt_llm.bench.benchmark.utils.processes import IterationWriter
from tensorrt_llm.bench.dataclasses.configuration import RuntimeConfig
from tensorrt_llm.bench.dataclasses.general import BenchmarkEnvironment
from tensorrt_llm.bench.dataclasses.reporting import ReportUtility
from tensorrt_llm.llmapi import LLM, CapacitySchedulerPolicy
from tensorrt_llm.models.modeling_utils import SpeculativeDecodingMode

# isort: off
from tensorrt_llm.bench.benchmark.utils.general import get_settings_from_engine
# isort: on
from tensorrt_llm.bench.utils.data import (create_dataset_from_stream,
                                           initialize_tokenizer)
from tensorrt_llm.logger import logger
from tensorrt_llm.sampling_params import SamplingParams


@click.command(name="latency")
@optgroup.group("Engine run configuration",
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
    "--num_requests",
    type=int,
    default=0,
    help="Number of requests to cap benchmark run at. Minimum between value and"
    "length of dataset.",
)
@optgroup.option(
    "--warmup",
    type=int,
    default=2,
    help="Number of requests warm up benchmark.",
)
@optgroup.group("Request Load Control Options",
                cls=MutuallyExclusiveOptionGroup,
                help="Limits how requests are loaded.")
@optgroup.option(
    "--beam_width",
    type=int,
    default=1,
    help="Number of search beams.",
)
@optgroup.option(
    "--concurrency",
    type=int,
    default=1,
    help=
    "Desired concurrency rate (number of requests processing at the same time), <=0 for no concurrency limit.",
)
@optgroup.group("Speculative Decode Options",
                help="Runtime settings for executing a TensorRT-LLM engine.")
@optgroup.option(
    "--medusa_choices",
    type=click.Path(exists=True,
                    readable=True,
                    path_type=Path,
                    resolve_path=True),
    default=None,
    required=False,
    help="Path to a YAML file that defines the Medusa tree.",
)
@optgroup.group("Reporting Options",
                help="Options for reporting benchmark results.",
                cls=OptionGroup)
@optgroup.option(
    "--report_json",
    type=click.Path(dir_okay=False,
                    writable=True,
                    readable=False,
                    path_type=Path,
                    resolve_path=True),
    required=False,
    help="Path where report should be written to.",
)
@optgroup.option(
    "--iteration_log",
    type=click.Path(dir_okay=False,
                    writable=True,
                    readable=False,
                    path_type=Path,
                    resolve_path=True),
    required=False,
    help="Path where iteration logging is written to.",
)
@click.pass_obj
def latency_command(
    bench_env: BenchmarkEnvironment,
    **params,
) -> None:
    """Run a latency test on a TRT-LLM engine."""

    logger.info("Preparing to run latency benchmark...")
    # Parameters from CLI
    # Model, experiment, and engine params
    dataset_path: Path = params.pop("dataset")
    num_requests: int = params.pop("num_requests")
    model: str = bench_env.model
    checkpoint_path: Path = bench_env.checkpoint_path or bench_env.model
    engine_dir: Path = params.pop("engine_dir")
    concurrency: int = params.pop("concurrency")
    beam_width: int = params.pop("beam_width")
    warmup: int = params.get("warmup")
    # Engine configuration parsing
    exec_settings, build_cfg = get_settings_from_engine(engine_dir)
    exec_settings["model"] = model
    engine_tokens = exec_settings["settings_config"]["max_num_tokens"]
    engine_max_seq_len = build_cfg["max_seq_len"]

    # Runtime Options
    kv_cache_percent = params.pop("kv_cache_free_gpu_mem_fraction")
    medusa_choices = params.pop("medusa_choices")

    # Reporting Options
    report_json: Path = params.pop("report_json")
    iteration_log: Path = params.pop("iteration_log")
    iteration_writer = IterationWriter(iteration_log)

    # Update configuration with runtime options
    exec_settings["settings_config"]["kv_cache_percent"] = kv_cache_percent
    exec_settings["settings_config"]["max_batch_size"] = 1
    exec_settings["settings_config"]["max_num_tokens"] = engine_tokens
    exec_settings["settings_config"]["beam_width"] = beam_width
    exec_settings["settings_config"]["chunking"] = False
    exec_settings["settings_config"][
        "scheduler_policy"] = CapacitySchedulerPolicy.GUARANTEED_NO_EVICT

    # Set environment variables for setting runtime options.
    # TODO: Once passing of variables is fixed, these should work
    # when using MPI in C++ runtime.
    os.environ["TRTLLM_ENABLE_MMHA_MULTI_BLOCK_DEBUG"] = "1"
    os.environ["TRTLLM_MMHA_KERNEL_BLOCK_SIZE"] = "256"
    os.environ["FORCE_MULTI_BLOCK_MODE"] = "1"
    os.environ["TRTLLM_ENABLE_PDL"] = "1"

    # Performance options
    exec_settings["performance_options"]["cuda_graphs"] = True
    exec_settings["performance_options"]["multi_block_mode"] = True

    # Decoding Options
    if medusa_choices is not None:
        with open(medusa_choices, "r") as medusa_yml:
            exec_settings["decoding_config"]["medusa_choices"] = \
                yaml.load(medusa_yml, Loader=yaml.SafeLoader)

    # Construct the runtime configuration dataclass.
    runtime_config = RuntimeConfig(**exec_settings)

    # Initialize the HF tokenizer for the specified model.
    ignore_eos = True if runtime_config.decoding_config.decoding_mode == SpeculativeDecodingMode.NONE else False
    tokenizer = initialize_tokenizer(checkpoint_path)
    eos_id = tokenizer.eos_token_id if not ignore_eos else -1
    pad_id = tokenizer.pad_token_id if not ignore_eos else -1

    # Dataset Loading and Preparation
    with open(dataset_path, "r") as dataset:
        metadata, requests = create_dataset_from_stream(
            tokenizer, dataset, num_requests=num_requests)
        metadata.dataset_path = dataset_path

    if metadata.max_sequence_length > engine_max_seq_len:
        raise RuntimeError(
            f"Engine supports a max sequence of {engine_max_seq_len}. Provided "
            "dataset contains a maximum sequence of "
            f"{metadata.max_sequence_length}. Please rebuild a new engine to"
            "support this dataset.")

    logger.info(metadata.get_summary_for_print())
    logger.info("Running experimental latency benchmark.")

    llm = None
    kwargs = runtime_config.get_llm_args()

    try:
        sampling_params = SamplingParams(
            end_id=eos_id,
            pad_id=pad_id,
            n=beam_width,
            use_beam_search=beam_width > 1,
        )
        post_proc_params = None  # No detokenization
        llm = LLM(**kwargs)

        # Perform warmup if requested.
        if warmup > 0:
            logger.info("Setting up for warmup...")
            warmup_dataset = generate_warmup_dataset(requests, warmup)
            logger.info("Running warmup.")
            asyncio.run(
                async_benchmark(llm, sampling_params, post_proc_params,
                                warmup_dataset, False, concurrency))
            # WAR: IterationResult is a singleton tied to the executor.
            # Since the benchmark calls asyncio.run() multiple times (e.g., during warmup),
            # we must reset it to ensure it attaches to the correct event loop.
            llm._executor._iter_stats_result = None
            logger.info("Warmup done.")

        with iteration_writer.capture():
            statistics = asyncio.run(
                async_benchmark(llm, sampling_params, post_proc_params,
                                requests, True, concurrency,
                                iteration_writer.full_address))

        logger.info(f"Benchmark done. Reporting results...")
        report_utility = ReportUtility(statistics, metadata, runtime_config,
                                       logger, kwargs, True)
        if report_json:
            logger.info(f"Writing report to '{report_json}'.")
            with open(report_json, "w") as f:
                f.write(
                    json.dumps(report_utility.get_statistics_dict(), indent=4))
        report_utility.report_statistics()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt, exiting benchmark...")
    finally:
        if llm is not None:
            llm.shutdown()
