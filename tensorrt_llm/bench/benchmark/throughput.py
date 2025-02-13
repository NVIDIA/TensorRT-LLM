from __future__ import annotations

import asyncio
from pathlib import Path

import click
from click_option_group import MutuallyExclusiveOptionGroup, optgroup

from tensorrt_llm.bench.benchmark.utils.asynchronous import async_benchmark

# isort: off
from tensorrt_llm.bench.benchmark.utils.general import (
    get_settings_from_engine, get_settings)
# isort: on
from tensorrt_llm._torch.llm import LLM as PyTorchLLM
from tensorrt_llm.bench.benchmark.utils.general import generate_warmup_dataset
from tensorrt_llm.bench.dataclasses.configuration import RuntimeConfig
from tensorrt_llm.bench.dataclasses.enums import IFBSchedulingPolicy
from tensorrt_llm.bench.dataclasses.general import BenchmarkEnvironment
from tensorrt_llm.bench.dataclasses.reporting import report_statistics
from tensorrt_llm.bench.utils.data import (create_dataset_from_stream,
                                           initialize_tokenizer)
from tensorrt_llm.builder import BuildConfig
from tensorrt_llm.llmapi.llm import LLM
from tensorrt_llm.logger import logger
from tensorrt_llm.sampling_params import SamplingParams


@click.command(name="throughput")
@optgroup.group("Engine run configuration.",
                help="Runtime settings for executing a TensorRT-LLM engine.")
@optgroup.option(
    "--engine_dir",
    type=click.Path(exists=True,
                    readable=True,
                    path_type=Path,
                    resolve_path=True),
    default=None,
    help="Path to a serialized TRT-LLM engine.",
)
@optgroup.option("--backend",
                 type=click.Choice(["pytorch"]),
                 default=None,
                 help="Set to 'pytorch' for pytorch path. Default is cpp path.")
@optgroup.option(
    "--extra_llm_api_options",
    type=str,
    default=None,
    help=
    "Path to a YAML file that overwrites the parameters specified by trtllm-bench."
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
    "--max_seq_len",
    type=int,
    default=None,
    help="Maximum sequence length.",
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
    required=False,
    help="Pass in a dataset file for parsing instead of stdin.",
)
@optgroup.option(
    "--num_requests",
    type=int,
    default=0,
    help=
    "Number of requests to cap benchmark run at. If not specified or set to 0, it will be the"
    "length of dataset.",
)
@optgroup.option(
    "--warmup",
    type=int,
    default=2,
    help="Number of requests warm up benchmark.",
)
@optgroup.option(
    "--tp",
    type=int,
    default=1,
    help="tensor parallelism size",
)
@optgroup.option(
    "--pp",
    type=int,
    default=1,
    help="pipeline parallelism size",
)
@optgroup.option(
    "--ep",
    type=int,
    default=None,
    help="expert parallelism size",
)
@optgroup.option(
    "--target_input_len",
    default=None,
    type=click.IntRange(min=1),
    help="Target (average) input length for tuning heuristics.",
)
@optgroup.option(
    "--target_output_len",
    default=None,
    type=click.IntRange(min=1),
    help="Target (average) sequence length for tuning heuristics.",
)
@optgroup.group("Request Load Control Options",
                cls=MutuallyExclusiveOptionGroup,
                help="Limits how requests are loaded.")
@optgroup.option(
    "--request_rate",
    type=int,
    default=-1,
    help="Desired input request rate (number of messages per second).",
    hidden=True,
)
@optgroup.option(
    "--concurrency",
    type=int,
    default=-1,
    help=
    "Desired concurrency rate (number of requests processing at the same time), <=0 for no concurrency limit.",
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
    hidden=True,
    help="Path where iteration stats should be written to.",
)
@click.pass_obj
def throughput_command(
    bench_env: BenchmarkEnvironment,
    **params,
) -> None:
    """Run a throughput test on a TRT-LLM engine."""
    logger.set_level("info")
    logger.info("Preparing to run throughput benchmark...")
    # Parameters from CLI
    # Model, experiment, and engine params
    dataset_path: Path = params.pop("dataset")
    request_rate: int = params.pop("request_rate")
    warmup: int = params.get("warmup")
    num_requests: int = params.pop("num_requests")
    max_seq_len: int = params.pop("max_seq_len")
    model: str = bench_env.model
    checkpoint_path: Path = bench_env.checkpoint_path or bench_env.model
    engine_dir: Path = params.pop("engine_dir")
    # TODO: Re-add iteration log. Disabled due to instability in LLM API.
    #iteration_log: Path = params.pop("iteration_log")
    concurrency: int = params.pop("concurrency")
    backend: str = params.get("backend")

    # Runtime kwargs and option tracking.
    kwargs = {}

    # Initialize the HF tokenizer for the specified model.
    tokenizer = initialize_tokenizer(checkpoint_path)

    # Dataset Loading and Preparation
    with open(dataset_path, "r") as dataset:
        metadata, requests = create_dataset_from_stream(
            tokenizer, dataset, num_requests=num_requests)
        params["target_input_len"] = params.get(
            "target_input_len") or metadata.avg_isl
        params["target_output_len"] = params.get(
            "target_output_len") or metadata.avg_osl

    # Engine configuration parsing
    if backend and backend.lower() == "pytorch":
        exec_settings = get_settings(params, metadata, bench_env.model,
                                     bench_env.checkpoint_path)
        kwargs_max_sql = max_seq_len or metadata.max_sequence_length
        logger.info(f"Setting PyTorch max sequence length to {kwargs_max_sql}")
        kwargs["build_config"] = BuildConfig(max_seq_len=kwargs_max_sql, )
    else:
        assert max_seq_len is None, (
            "max_seq_len is not a runtime parameter for C++ backend")
        exec_settings, build_cfg = get_settings_from_engine(engine_dir)
        engine_max_seq_len = build_cfg["max_seq_len"]

        # TODO: Verify that the engine can handle the max/min ISL/OSL.
        if metadata.max_sequence_length > engine_max_seq_len:
            raise RuntimeError(
                f"Engine supports a max sequence of {engine_max_seq_len}. "
                "Provided dataset contains a maximum sequence of "
                f"{metadata.max_sequence_length}. Please rebuild a new engine "
                "to support this dataset.")

    exec_settings["model"] = model
    engine_bs = exec_settings["settings_config"]["max_batch_size"]
    engine_tokens = exec_settings["settings_config"]["max_num_tokens"]

    # Check that we are not using a low latency engine
    # Right now, this is based on max batch size.
    if engine_bs == 1:
        raise ValueError(
            "An engine with a batch size greater than 1 should be used for "
            "throughput benchmarking. Exiting.")

    # Runtime Options
    runtime_max_bs = params.pop("max_batch_size")
    runtime_max_tokens = params.pop("max_num_tokens")
    runtime_max_bs = runtime_max_bs or engine_bs
    runtime_max_tokens = runtime_max_tokens or engine_tokens
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

    # LlmArgs
    exec_settings["extra_llm_api_options"] = params.pop("extra_llm_api_options")

    # Construct the runtime configuration dataclass.
    runtime_config = RuntimeConfig(**exec_settings)
    llm = None
    try:
        logger.info("Setting up throughput benchmark.")
        kwargs = kwargs | runtime_config.get_llm_args()
        if runtime_config.backend == 'pytorch':
            llm = PyTorchLLM(**kwargs)
        else:
            llm = LLM(**kwargs)

        sampling_params = SamplingParams(end_id=-1,
                                         pad_id=-1,
                                         beam_width=beam_width)

        # Perform warmup if requested.
        if warmup > 0:
            logger.info("Setting up for warmup...")
            warmup_dataset = generate_warmup_dataset(requests, warmup)
            logger.info("Running warmup.")
            asyncio.run(
                async_benchmark(llm, sampling_params, warmup_dataset,
                                concurrency))
            logger.info("Warmup done.")

        statistics = asyncio.run(
            async_benchmark(llm, sampling_params, requests, streaming,
                            concurrency))
        report_statistics(statistics, runtime_config, logger, kwargs, streaming)
    finally:
        if llm is not None:
            llm.__exit__(None, None, None)
