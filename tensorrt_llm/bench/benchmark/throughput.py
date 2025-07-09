from __future__ import annotations

import asyncio
import json
from pathlib import Path

import click
from click_option_group import (MutuallyExclusiveOptionGroup, OptionGroup,
                                optgroup)

from tensorrt_llm.bench.benchmark.utils.asynchronous import async_benchmark
from tensorrt_llm.bench.benchmark.utils.processes import IterationWriter

# isort: off
from tensorrt_llm.bench.benchmark.utils.general import (
<<<<<<< HEAD
    get_settings_from_engine, get_settings, ALL_SUPPORTED_BACKENDS)
# isort: on
=======
    get_settings_from_engine, get_settings)
>>>>>>> c1880441d (Move snapshot download.)
from tensorrt_llm import LLM as PyTorchLLM
from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm._torch.auto_deploy import LLM as AutoDeployLLM
from tensorrt_llm.bench.benchmark.utils.general import generate_warmup_dataset
from tensorrt_llm.bench.dataclasses.configuration import RuntimeConfig
from tensorrt_llm.bench.dataclasses.reporting import ReportUtility
from tensorrt_llm.bench.dataclasses.scenario import (
    BatchingConfiguration, BenchmarkEnvironment, LlmRuntimeSpecification,
    ReportingConfiguration, ScenarioSpecification, TuningConstraints,
    WorldConfig)
from tensorrt_llm.bench.tuning.factory import HeuristicFactory
# isort: on
from tensorrt_llm.bench.utils.data import (create_dataset_from_stream,
                                           initialize_tokenizer,
                                           update_metadata_for_multimodal)
from tensorrt_llm.llmapi import CapacitySchedulerPolicy
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
                 type=click.Choice(ALL_SUPPORTED_BACKENDS),
                 default="pytorch",
                 help="The backend to use when running benchmarking.")
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
    "--eos_id",
    type=int,
    default=-1,
    required=False,
    help=
    "Set the end-of-sequence token for the benchmark. Set to -1 to disable EOS.",
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
    help=
    "Number of requests to cap benchmark run at. If not specified or set to 0, it will be the "
    "length of dataset.",
)
@optgroup.option(
    "--warmup",
    type=int,
    default=2,
    help="Number of requests warm up benchmark.",
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
@optgroup.group(
    "World Configuration",
    help="Options for configuring the backend multi-GPU world.",
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
    "--cluster_size",
    type=int,
    default=None,
    help="expert cluster parallelism size",
)
@optgroup.group("Request Load Control Options",
                cls=MutuallyExclusiveOptionGroup,
                help="Limits how requests are loaded.")
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
    help="Path where report is written to.",
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
@optgroup.option(
    "--output_json",
    type=click.Path(dir_okay=False,
                    writable=True,
                    readable=False,
                    path_type=Path,
                    resolve_path=True),
    required=False,
    help="Path where output should be written to.",
)
@optgroup.option(
    "--request_json",
    type=click.Path(dir_okay=False,
                    writable=True,
                    readable=False,
                    path_type=Path,
                    resolve_path=True),
    required=False,
    help="Path where per request information is written to.",
)
@optgroup.option(
    "--enable_chunked_context",
    is_flag=True,
    default=False,
    help="Enable chunking in prefill stage for enhanced throughput benchmark.",
)
@optgroup.option(
    "--scheduler_policy",
    type=click.Choice(["guaranteed_no_evict", "max_utilization"]),
    default="guaranteed_no_evict",
    help=
    "KV cache scheduler policy: guaranteed_no_evict prevents request eviction, max_utilization optimizes for throughput.",
)
@click.pass_obj
def throughput_command(
    bench_env: BenchmarkEnvironment,
    **params,
) -> None:
    """Run a throughput test on a TRT-LLM engine."""

    logger.info("Preparing to run throughput benchmark...")
    # Local variables:
    llm = None

    LLM_CLS_MAP = {
        "pytorch": PyTorchLLM,
        "tensorrt": LLM,
    }

    # Populate the benchmark specification with the parameters from the CLI.
    benchmark_specification = ScenarioSpecification(
        **params,
        environment=bench_env,
        batching_config=BatchingConfiguration(**params),
        llm_config=LlmRuntimeSpecification(**params),
        world=WorldConfig(**params),
        reporting_config=ReportingConfiguration(**params),
    )

    # Initialize the HF tokenizer for the specified model.
    tokenizer = initialize_tokenizer(benchmark_specification.checkpoint)

    # Dataset Loading and Preparation
    with open(benchmark_specification.dataset_path, "r") as dataset:
        metadata, requests = create_dataset_from_stream(
            tokenizer,
            dataset,
            model_dir=benchmark_specification.checkpoint,
            num_requests=benchmark_specification.num_requests,
            model_type=benchmark_specification.environment.model_type,
            modality=benchmark_specification.modality,
            max_input_seq_len_for_multimodal=benchmark_specification.
            batching_config.max_seq_len)

    benchmark_specification.constraints = TuningConstraints.from_dataset_metadata(
        metadata)
    benchmark_specification.dataset_metadata = metadata

    if benchmark_specification.modality == "text":
        # Log dataset info
        # NOTE: This table is only accurate for non-multimodal models.
        #       The accurate table for multimodal models will be logged after the benchmark is done.
        logger.info(benchmark_specification.get_dataset_summary())

    print(benchmark_specification)

    heuristic_cls = HeuristicFactory.get_heuristic(benchmark_specification.llm_config.backend, "throughput")
    kwargs = heuristic_cls.get_settings(benchmark_specification)
    kwargs['backend'] = benchmark_specification.llm_config.backend
    backend = kwargs['backend']
    iteration_log = benchmark_specification.reporting_config.iteration_log

    if backend == "pytorch" and iteration_log is not None:
        kwargs["enable_iter_perf_stats"] = True

    # NOTE: We need to separate what is considered a benchmark setting and
    # what is considered a tuning setting (for potential future re-use).
    # Update to disable block reuse and partial reuse
    kwargs["kv_cache_config"] |= {
        "enable_block_reuse": False,
        "enable_partial_reuse": False,
    }
    # Sample Parameters
    sampling_params = \
        SamplingParams(
            end_id=benchmark_specification.llm_config.eos_id,
            pad_id=benchmark_specification.llm_config.eos_id,
            n=benchmark_specification.llm_config.beam_width,
            use_beam_search=benchmark_specification.llm_config.beam_width > 1
        )


    try:
        logger.info("Setting up throughput benchmark.")
        llm_cls = LLM_CLS_MAP[backend]
        llm = llm_cls(**kwargs)
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

        iteration_writer = IterationWriter(
            benchmark_specification.reporting_config.iteration_log)
        with iteration_writer.capture():
            statistics = asyncio.run(
                async_benchmark(llm,
                                sampling_params,
                                post_proc_params,
                                requests,
                                streaming,
                                concurrency,
                                iteration_writer.full_address,
                                modality=modality))

        logger.info(f"Benchmark done. Reporting results...")
        if modality is not None:
            # For multimodal models, we need to update the metadata with the correct input lengths
            metadata = update_metadata_for_multimodal(metadata, statistics)

        report_utility = ReportUtility(statistics, metadata, runtime_config,
                                       logger, kwargs, streaming)
        if report_json:
            logger.info(f"Writing report to '{report_json}'.")
            with open(report_json, "w") as f:
                f.write(
                    json.dumps(report_utility.get_statistics_dict(), indent=4))
        if output_json:
            logger.info(f"Writing output to {output_json}.")
            with open(output_json, "w") as f:
                output_token_info = report_utility.get_output_tokens(tokenizer)
                f.write(json.dumps(output_token_info, indent=4))
        if request_json:
            logger.info(f"Writing request information to {request_json}.")
            with open(request_json, "w") as f:
                f.write(json.dumps(report_utility.get_request_info(tokenizer)))
        report_utility.report_statistics()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt, exiting benchmark...")
    finally:
        if llm is not None:
            llm.shutdown()
