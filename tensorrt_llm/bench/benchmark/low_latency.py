from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import click
import yaml
from click_option_group import (MutuallyExclusiveOptionGroup, OptionGroup,
                                optgroup)
from huggingface_hub import snapshot_download

from tensorrt_llm import LLM as PyTorchLLM
from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm._torch.auto_deploy import LLM as AutoDeployLLM
from tensorrt_llm.bench.benchmark.utils.asynchronous import async_benchmark
from tensorrt_llm.bench.benchmark.utils.general import generate_warmup_dataset
from tensorrt_llm.bench.benchmark.utils.processes import IterationWriter
from tensorrt_llm.bench.build.build import get_model_config
from tensorrt_llm.bench.dataclasses.configuration import RuntimeConfig
from tensorrt_llm.bench.dataclasses.general import BenchmarkEnvironment
from tensorrt_llm.bench.dataclasses.reporting import ReportUtility
from tensorrt_llm.llmapi import CapacitySchedulerPolicy
from tensorrt_llm.models.modeling_utils import SpeculativeDecodingMode

# isort: off
from tensorrt_llm.bench.benchmark.utils.general import get_settings_from_engine, get_settings, update_sampler_args_with_extra_options, ALL_SUPPORTED_BACKENDS
# isort: on
from tensorrt_llm.bench.utils.data import (create_dataset_from_stream,
                                           initialize_tokenizer,
                                           update_metadata_for_multimodal)
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
    default=None,
    help="Path to a serialized TRT-LLM engine.",
)
@optgroup.option("--backend",
                 type=click.Choice(ALL_SUPPORTED_BACKENDS),
                 default="pytorch",
                 help="The backend to use when running benchmarking.")
@optgroup.option(
    "--kv_cache_free_gpu_mem_fraction",
    type=float,
    default=.90,
    help="The percentage of memory to use for KV Cache after model load.",
)
@optgroup.option(
    "--max_seq_len",
    type=int,
    default=None,
    help="Maximum sequence length.",
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
    "--modality",
    type=click.Choice(["image", "video"]),
    default=None,
    help="Modality of the multimodal requests.",
)
@optgroup.option(
    "--max_input_len",
    type=int,
    default=4096,
    help=
    "Maximum input sequence length to use for multimodal models. This is used only when --modality "
    "is specified since the actual number of vision tokens is unknown before the model is run.",
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
@optgroup.group("Request Load Control Options",
                cls=MutuallyExclusiveOptionGroup,
                help="Limits how requests are loaded.")
@optgroup.option(
    "--beam_width",
    type=int,
    default=1,
    help="Number of search beams.",
)
@optgroup.option("--sampler_options",
                 type=click.Path(exists=True,
                                 readable=True,
                                 path_type=Path,
                                 resolve_path=True),
                 default=None,
                 help="Path to a YAML file that sets sampler options.")
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
    dataset_path: Path = params.get("dataset")
    num_requests: int = params.get("num_requests")
    model: str = bench_env.model
    checkpoint_path: Path = bench_env.checkpoint_path or bench_env.model
    engine_dir: Path = params.get("engine_dir")
    concurrency: int = params.get("concurrency")
    beam_width: int = params.get("beam_width")
    warmup: int = params.get("warmup")
    modality: str = params.get("modality")
    max_input_len: int = params.get("max_input_len")
    max_seq_len: int = params.get("max_seq_len")
    backend: str = params.get("backend")
    model_type = get_model_config(model, checkpoint_path).model_type

    # Runtime Options
    kv_cache_percent = params.get("kv_cache_free_gpu_mem_fraction")
    medusa_choices = params.get("medusa_choices")

    # Reporting Options
    report_json: Path = params.pop("report_json")
    iteration_log: Path = params.pop("iteration_log")
    iteration_writer = IterationWriter(iteration_log)

    # Initialize the HF tokenizer for the specified model.
    tokenizer = initialize_tokenizer(checkpoint_path)

    # Dataset Loading and Preparation
    with open(dataset_path, "r") as dataset:
        metadata, requests = create_dataset_from_stream(
            tokenizer,
            dataset,
            num_requests=num_requests,
            model_dir=checkpoint_path,
            model_type=model_type,
            modality=modality,
            max_input_seq_len_for_multimodal=max_input_len)

        metadata.dataset_path = dataset_path

    if modality is None:
        # Log dataset info
        # NOTE: This table is only accurate for non-multimodal models.
        #       The accurate table for multimodal models will be logged after the benchmark is done.
        logger.info(metadata.get_summary_for_print())

    # Engine configuration parsing for PyTorch backend
    kwargs = {}
    if backend and backend.lower() in ALL_SUPPORTED_BACKENDS and backend.lower(
    ) != "tensorrt":
        if bench_env.checkpoint_path is None:
            snapshot_download(model)

        exec_settings = get_settings(params, metadata, bench_env.model,
                                     bench_env.checkpoint_path)
        kwargs_max_sql = max_seq_len or metadata.max_sequence_length
        logger.info(f"Setting PyTorch max sequence length to {kwargs_max_sql}")
        kwargs["max_seq_len"] = kwargs_max_sql
    elif backend.lower() == "tensorrt":
        assert max_seq_len is None, (
            "max_seq_len is not a runtime parameter for C++ backend")
        exec_settings, build_cfg = get_settings_from_engine(engine_dir)
        engine_max_seq_len = build_cfg["max_seq_len"]

        if metadata.max_sequence_length > engine_max_seq_len:
            raise RuntimeError(
                f"Engine supports a max sequence of {engine_max_seq_len}. Provided "
                "dataset contains a maximum sequence of "
                f"{metadata.max_sequence_length}. Please rebuild a new engine to"
                "support this dataset.")
    else:
        raise RuntimeError(
            f"Invalid backend: {backend}, please use one of the following: "
            f"{ALL_SUPPORTED_BACKENDS}")

    exec_settings["model"] = model
    engine_tokens = exec_settings["settings_config"]["max_num_tokens"]

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

    llm = None
    kwargs = kwargs | runtime_config.get_llm_args()
    kwargs['backend'] = backend

    try:
        logger.info("Setting up latency benchmark.")

        if "pytorch_backend_config" in kwargs and iteration_log is not None:
            kwargs["pytorch_backend_config"].enable_iter_perf_stats = True

        if runtime_config.backend == 'pytorch':
            if kwargs.pop("extended_runtime_perf_knob_config", None):
                logger.warning(
                    "Ignore extended_runtime_perf_knob_config for pytorch backend."
                )
            llm = PyTorchLLM(**kwargs)
        elif runtime_config.backend == "_autodeploy":
            if kwargs.pop("extended_runtime_perf_knob_config", None):
                logger.warning(
                    "Ignore extended_runtime_perf_knob_config for _autodeploy backend."
                )
            kwargs["world_size"] = kwargs.pop("tensor_parallel_size", None)
            kwargs.pop("pipeline_parallel_size", None)

            llm = AutoDeployLLM(**kwargs)
        else:
            llm = LLM(**kwargs)

        ignore_eos = True if runtime_config.decoding_config.decoding_mode == SpeculativeDecodingMode.NONE else False
        eos_id = tokenizer.eos_token_id if not ignore_eos else -1
        pad_id = tokenizer.pad_token_id if not ignore_eos else -1

        sampler_args = {
            "end_id": eos_id,
            "pad_id": pad_id,
            "n": beam_width,
            "use_beam_search": beam_width > 1
        }
        sampler_args = update_sampler_args_with_extra_options(
            sampler_args, params.pop("sampler_options"))
        sampling_params = SamplingParams(**sampler_args)

        post_proc_params = None  # No detokenization

        # Perform warmup if requested.
        if warmup > 0:
            logger.info("Setting up for warmup...")
            warmup_dataset = generate_warmup_dataset(requests, warmup)
            logger.info("Running warmup.")
            asyncio.run(
                async_benchmark(llm,
                                sampling_params,
                                post_proc_params,
                                warmup_dataset,
                                False,
                                concurrency,
                                modality=modality))
            # WAR: IterationResult is a singleton tied to the executor.
            # Since the benchmark calls asyncio.run() multiple times (e.g., during warmup),
            # we must reset it to ensure it attaches to the correct event loop.
            llm._executor._iter_stats_result = None
            logger.info("Warmup done.")

        with iteration_writer.capture():
            statistics = asyncio.run(
                async_benchmark(llm,
                                sampling_params,
                                post_proc_params,
                                requests,
                                True,
                                concurrency,
                                iteration_writer.full_address,
                                modality=modality))

        logger.info(f"Benchmark done. Reporting results...")

        if modality is not None:
            # For multimodal models, we need to update the metadata with the correct input lengths
            metadata = update_metadata_for_multimodal(metadata, statistics)

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
