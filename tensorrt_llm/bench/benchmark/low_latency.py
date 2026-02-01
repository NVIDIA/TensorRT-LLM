from __future__ import annotations

import asyncio
from functools import partial
from pathlib import Path

import click
import yaml
from click_option_group import (MutuallyExclusiveOptionGroup, OptionGroup,
                                optgroup)

from tensorrt_llm.bench.benchmark import (generate_json_report,
                                          get_general_cli_options)
from tensorrt_llm.bench.benchmark.utils.asynchronous import async_benchmark
from tensorrt_llm.bench.benchmark.utils.general import (
    ALL_SUPPORTED_BACKENDS, generate_warmup_dataset,
    get_exec_settings_for_backend, update_sampler_args_with_extra_options)
from tensorrt_llm.bench.dataclasses.general import BenchmarkEnvironment
from tensorrt_llm.bench.dataclasses.reporting import ReportUtility
from tensorrt_llm.bench.utils.data import (create_dataset_from_stream,
                                           initialize_tokenizer,
                                           update_metadata_for_multimodal)
from tensorrt_llm.llmapi import CapacitySchedulerPolicy
from tensorrt_llm.llmapi.llm_args import CudaGraphConfig
from tensorrt_llm.llmapi.llm_create import create_llm_from_llm_args
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import SpeculativeDecodingMode
from tensorrt_llm.sampling_params import SamplingParams


@click.command(name="latency")
@optgroup.group("Engine run configuration",
                help="Runtime settings for executing a TensorRT LLM engine.")
@optgroup.option(
    "--engine_dir",
    type=click.Path(exists=True,
                    readable=True,
                    path_type=Path,
                    resolve_path=True),
    default=None,
    help="Path to a serialized TRT-LLM engine.",
)
@optgroup.option(
    "--config",
    "--extra_llm_api_options",
    "extra_llm_api_options",
    type=str,
    default=None,
    help=
    "Path to a YAML file that overwrites the parameters specified by trtllm-bench. "
    "Can be specified as either --config or --extra_llm_api_options.")
@optgroup.option(
    "--backend",
    type=click.Choice(ALL_SUPPORTED_BACKENDS),
    default="pytorch",
    help="The backend to use for benchmark. Default is pytorch backend.")
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
                help="Runtime settings for executing a TensorRT LLM engine.")
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
    options = get_general_cli_options(params, bench_env)

    # Speculative Decode Options
    medusa_choices = params.get("medusa_choices")
    # Initialize the HF tokenizer for the specified model.
    tokenizer = initialize_tokenizer(options.checkpoint_path)

    # Dataset Loading and Preparation
    with open(options.dataset_path, "r") as dataset:
        metadata, requests = create_dataset_from_stream(
            tokenizer,
            dataset,
            num_requests=options.num_requests,
            model_dir=options.checkpoint_path,
            model_type=options.model_type,
            modality=options.modality,
            max_input_seq_len_for_multimodal=options.max_input_len)

        metadata.dataset_path = options.dataset_path

    if options.modality is None:
        # Log dataset info
        # NOTE: This table is only accurate for non-multimodal models.
        #       The accurate table for multimodal models will be logged after the benchmark is done.
        logger.info(metadata.get_summary_for_print())

    # Engine configuration parsing
    exec_settings = get_exec_settings_for_backend(params, metadata, options,
                                                  bench_env)

    exec_settings.revision = bench_env.revision

    # Update llm_args with runtime options
    llm_args = exec_settings.llm_args
    llm_args.max_batch_size = 1
    llm_args.cuda_graph_config = CudaGraphConfig(max_batch_size=1)
    llm_args.enable_chunked_prefill = False
    llm_args.scheduler_config.capacity_scheduler_policy = CapacitySchedulerPolicy.GUARANTEED_NO_EVICT

    # Decoding Options
    if medusa_choices is not None:
        with open(medusa_choices, "r") as medusa_yml:
            medusa_choices_data = yaml.load(medusa_yml, Loader=yaml.SafeLoader)
            if llm_args.decoding_config is not None:
                llm_args.decoding_config.medusa_choices = medusa_choices_data

    # Set environment variables for setting runtime options.
    default_env_overrides = {
        "TRTLLM_ENABLE_MMHA_MULTI_BLOCK_DEBUG": "1",
        "TRTLLM_MMHA_KERNEL_BLOCK_SIZE": "256",
        "FORCE_MULTI_BLOCK_MODE": "1",
        "TRTLLM_ENABLE_PDL": "1",
    }
    # Update defaults with existing overrides (user preference takes priority)
    llm_args.env_overrides = default_env_overrides | llm_args.env_overrides

    llm = None
    try:
        logger.info("Setting up latency benchmark.")
        llm = create_llm_from_llm_args(llm_args)

        ignore_eos = True if llm_args.decoding_config.decoding_mode == SpeculativeDecodingMode.NONE else False
        eos_id = tokenizer.eos_token_id if not ignore_eos else -1
        pad_id = tokenizer.pad_token_id if not ignore_eos else -1

        sampler_args = {
            "end_id": eos_id,
            "pad_id": pad_id,
            "n": llm_args.max_beam_width,
            "use_beam_search": llm_args.max_beam_width > 1
        }

        sampler_args = update_sampler_args_with_extra_options(
            sampler_args, params.pop("sampler_options"))
        sampling_params = SamplingParams(**sampler_args)

        post_proc_params = None  # No detokenization

        # Perform warmup if requested.
        if options.warmup > 0:
            logger.info("Setting up for warmup...")
            warmup_dataset = generate_warmup_dataset(requests, options.warmup)
            logger.info("Running warmup.")
            asyncio.run(
                async_benchmark(llm,
                                sampling_params,
                                post_proc_params,
                                warmup_dataset,
                                False,
                                options.concurrency,
                                modality=options.modality))
            # WAR: IterationResult is a singleton tied to the executor.
            # Since the benchmark calls asyncio.run() multiple times (e.g., during warmup),
            # we must reset it to ensure it attaches to the correct event loop.
            llm._executor._iter_stats_result = None
            logger.info("Warmup done.")

        iteration_writer = options.iteration_writer
        with iteration_writer.capture():
            statistics = asyncio.run(
                async_benchmark(llm,
                                sampling_params,
                                post_proc_params,
                                requests,
                                True,
                                options.concurrency,
                                iteration_writer.full_address,
                                modality=options.modality))

        logger.info("Benchmark done. Reporting results...")

        if options.modality is not None:
            # For multimodal models, we need to update the metadata with the correct input lengths
            metadata = update_metadata_for_multimodal(metadata, statistics)

        report_utility = ReportUtility(statistics, metadata, exec_settings,
                                       logger, llm_args, True)
        # Generate reports for statistics, output tokens, and request info.
        generate_json_report(options.report_json,
                             report_utility.get_statistics_dict)
        generate_json_report(
            options.output_json,
            partial(report_utility.get_output_tokens, tokenizer))
        generate_json_report(
            options.request_json,
            partial(report_utility.get_request_info, tokenizer))
        report_utility.report_statistics()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt, exiting benchmark...")
    finally:
        if llm is not None:
            llm.shutdown()
