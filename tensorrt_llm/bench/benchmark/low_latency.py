from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from random import choices, shuffle
from time import monotonic_ns, sleep
from typing import List

import click
import yaml
from click_option_group import optgroup

import tensorrt_llm.bindings.executor as trtllm
from tensorrt_llm.bench.dataclasses.configuration import RuntimeConfig
from tensorrt_llm.bench.dataclasses.enums import IFBSchedulingPolicy
from tensorrt_llm.bench.dataclasses.general import BenchmarkEnvironment
from tensorrt_llm.bench.dataclasses.reporting import StatsKeeper
from tensorrt_llm.bench.dataclasses.statistics import BenchmarkStatistics
from tensorrt_llm.models.modeling_utils import SpeculativeDecodingMode

# isort: off
from tensorrt_llm.bench.benchmark.utils.general import (get_executor_requests,
                                                        get_settings_from_engine
                                                        )
# isort: on
from tensorrt_llm.bench.utils.data import (create_dataset_from_stream,
                                           initialize_tokenizer)
from tensorrt_llm.logger import logger


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
    default=0,
    help="Number of requests warm up benchmark.",
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
@click.pass_obj
def latency_command(
    bench_env: BenchmarkEnvironment,
    **params,
) -> None:
    """Run a latency test on a TRT-LLM engine."""

    logger.set_level("info")
    logger.info("Preparing to run latency benchmark...")
    # Parameters from CLI
    # Model, experiment, and engine params
    dataset_path: Path = params.pop("dataset")
    num_requests: int = params.pop("num_requests")
    model: str = bench_env.model
    engine_dir: Path = params.pop("engine_dir")
    # Engine configuration parsing
    exec_settings, build_cfg = get_settings_from_engine(engine_dir)
    exec_settings["model"] = model
    engine_bs = exec_settings["settings_config"]["max_batch_size"]
    engine_tokens = exec_settings["settings_config"]["max_num_tokens"]
    engine_max_seq_len = build_cfg["max_seq_len"]

    # Runtime Options
    kv_cache_percent = params.pop("kv_cache_free_gpu_mem_fraction")
    medusa_choices = params.pop("medusa_choices")

    # Update configuration with runtime options
    exec_settings["settings_config"]["kv_cache_percent"] = kv_cache_percent
    exec_settings["settings_config"]["max_batch_size"] = engine_bs
    exec_settings["settings_config"]["max_num_tokens"] = engine_tokens
    exec_settings["settings_config"]["beam_width"] = 1
    exec_settings["settings_config"]["chunking"] = False
    exec_settings["settings_config"][
        "scheduler_policy"] = IFBSchedulingPolicy.NO_EVICT

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
    warmup_steps = params.get("warmup")

    # Initialize the HF tokenizer for the specified model.
    ignore_eos = True if runtime_config.decoding_config.decoding_mode == SpeculativeDecodingMode.NONE else False
    tokenizer = initialize_tokenizer(bench_env.model)
    eos_id = tokenizer.eos_token_id if not ignore_eos else -1
    pad_id = tokenizer.pad_token_id if not ignore_eos else -1

    # Dataset Loading and Preparation
    with open(dataset_path, "r") as dataset:
        metadata, requests = create_dataset_from_stream(
            tokenizer, dataset, num_requests=num_requests)

    if metadata.max_sequence_length > engine_max_seq_len:
        raise RuntimeError(
            f"Engine supports a max sequence of {engine_max_seq_len}. Provided "
            "dataset contains a maximum sequence of "
            f"{metadata.max_sequence_length}. Please rebuild a new engine to"
            "support this dataset.")

    # Dataset Loading and Preparation
    executor_requests = get_executor_requests(
        requests,
        True,
        eos_id=eos_id,
        pad_id=pad_id,
    )
    del requests

    # Instantiate the low latency benchmark.
    benchmark = LatencyBenchmark(
        executor_requests,
        runtime_config,
    )

    try:
        logger.info("Ready to start benchmark.")
        benchmark.setup_warmup(warmup_steps)
        benchmark.start_benchmark()
        benchmark.report_statistics()
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted! Shutting down...")
    finally:
        benchmark.stop_benchmark()


class LatencyBenchmark:
    """Latency benchmark utility class."""

    def __init__(
        self,
        dataset: List[trtllm.Request],
        runtime_cfg: RuntimeConfig,
    ) -> None:
        """Initialize the throughput benchmark.

        Args:
            dataset (List[trtllm.Request]): A dataset of TRT-LLM requests to
            benchmark against.
            runtime_cfg (RuntimeConfig): Runtime configuration.
        """
        # Dataset and input properties.
        self.requests = dataset
        self.warm_up_dataset = None
        self.runtime_config = deepcopy(runtime_cfg)
        self.streaming = True

        # Benchmark stats and time tracking.
        self.start_time = None
        self.end_time = None
        self.submitted_requests = 0
        self.statistics = StatsKeeper()

        logger.info("Starting Executor backend...")
        self.executor = None
        logger.info("Executor started.")

    def _setup_environment(self) -> None:
        # TODO: Once passing of variables is fixed, these should work
        # when using MPI in C++ runtime.
        os.environ["TRTLLM_ENABLE_MMHA_MULTI_BLOCK_DEBUG"] = "1"
        os.environ["TRTLLM_MMHA_KERNEL_BLOCK_SIZE"] = "256"
        os.environ["TRTLLM_MMHA_KERNEL_BLOCK_SIZE"] = "32"
        os.environ["FORCE_MULTI_BLOCK_MODE"] = "1"
        os.environ["TRTLLM_ENABLE_PDL"] = "1"

    def setup_warmup(self, steps) -> None:
        """Warm up the benchmarker."""
        if steps > 0:
            self.warm_up_dataset = choices(self.requests, k=steps)
            shuffle(self.warm_up_dataset)

    def start_benchmark(self) -> None:
        """Start the benchmark."""
        logger.info("Initializing backend...")
        self._setup_environment()
        self.executor = trtllm.Executor(
            self.runtime_config.engine_dir,
            trtllm.ModelType.DECODER_ONLY,
            executor_config=self.runtime_config.get_config())

        logger.info("WAITING ON EXECUTOR...")
        while not self.executor.can_enqueue_requests():
            logger.info("Waiting for executor to stand up...")
            sleep(1)

        if self.warm_up_dataset and len(self.warm_up_dataset) > 0:
            logger.info(f"WARMING UP...")
            for i, request in enumerate(self.warm_up_dataset, start=1):
                logger.info(f"Running warm up step {i}...")
                req_id = self.executor.enqueue_request(request)
                final = False
                while not final:
                    responses = self.executor.await_responses(req_id)
                    final = any([resp.result.is_final for resp in responses])

            logger.info("WARMUP COMPLETE.")

        logger.info("Low latency benchmark started.")
        self.start_time = monotonic_ns()
        while len(self.requests) > 0:
            final = False
            request = self.requests.pop(0)

            req_id = self.executor.enqueue_request(request)
            self.statistics.register_request(req_id, monotonic_ns(),
                                             len(request.input_token_ids))

            while not final:
                responses = self.executor.await_responses(req_id)
                now = monotonic_ns()
                for resp in responses:
                    self.statistics.register_response(
                        req_id, now, resp.result.is_final, resp.has_error(),
                        resp.result.decoding_iter,
                        resp.result.output_token_ids[0])
                    final = resp.result.is_final

        self.end_time = monotonic_ns()
        logger.info("Low latency benchmark finished.")

    def stop_benchmark(self) -> None:
        """Stop the benchmark and clean up backend and threads."""
        logger.info("Benchmark Shutdown called!")
        if self.executor is not None:
            self.executor.shutdown()
        logger.info("Executor shutdown.")

    def report_statistics(self) -> BenchmarkStatistics:
        """Report internal statistics about benchmark."""

        config_path = self.runtime_config.engine_dir / "config.json"
        with open(config_path, "r") as config:
            engine_config = json.load(config)

        stats = self.statistics.generate_statistics_summary()
        rt_cfg = self.runtime_config
        build_cfg = engine_config["build_config"]
        pretrain_cfg = engine_config["pretrained_config"]

        logging_info = (
            "\n\n===========================================================\n"
            "= ENGINE DETAILS\n"
            "===========================================================\n"
            f"Model:\t\t\t{rt_cfg.model}\n"
            f"Engine Directory:\t{rt_cfg.engine_dir}\n"
            f"TensorRT-LLM Version:\t{rt_cfg.sw_version}\n"
            f"Dtype:\t\t\t{pretrain_cfg['dtype']}\n"
            f"KV Cache Dtype:\t\t{pretrain_cfg['quantization']['kv_cache_quant_algo']}\n"
            f"Quantization:\t\t{pretrain_cfg['quantization']['quant_algo']}\n"
            f"Max Input Length:\t{build_cfg['max_input_len']}\n"
            f"Max Sequence Length:\t{build_cfg['max_seq_len']}\n"
            f"\n"
            "===========================================================\n"
            "= WORLD + RUNTIME INFORMATION \n"
            "===========================================================\n"
            f"TP Size:\t\t{rt_cfg.world_config.tp_size}\n"
            f"PP Size:\t\t{rt_cfg.world_config.pp_size}\n"
            f"Max Runtime Batch Size:\t{rt_cfg.settings_config.max_batch_size}\n"
            f"Max Runtime Tokens:\t{rt_cfg.settings_config.max_num_tokens}\n"
            f"Scheduling Policy:\t{rt_cfg.settings_config.scheduler_policy.values[1]}\n"
            f"KV Memory Percentage:\t{rt_cfg.settings_config.kv_cache_percent * 100.0:.2f}%\n"
            f"\n"
            "===========================================================\n"
            "= GENERAL OVERVIEW \n"
            "===========================================================\n"
            f"Number of requests:\t\t{stats.num_requests}\n"
            f"Average Input Length (tokens):\t{stats.average_input_length:.4f}\n"
            f"Average Output Length (tokens):\t{stats.average_output_length:.4f}\n"
            f"Average request latency (ms):\t{stats.request_latency_percentiles.average * 1.0e-6:.4f}\n"
            f"\n"
            "===========================================================\n"
            "= THROUGHPUT OVERVIEW \n"
            "===========================================================\n"
            f"Request Throughput (req/sec):\t\t  {stats.request_throughput_ns * 1.0e9:.4f}\n"
            f"Total Token Throughput (tokens/sec):\t  {stats.token_throughput_ns * 1.0e9:.4f}\n"
            f"Generation Token Throughput (tokens/sec): {stats.generation_tp_percentiles.average * 1.0e9:.4f}\n"
            f"\n"
            "===========================================================\n"
            "= LATENCY OVERVIEW \n"
            "===========================================================\n"
            f"Total Latency (ms):\t\t  {stats.total_latency_ns * 1.0e-6:.4f}\n"
            f"Average time-to-first-token (ms): {stats.ttft_percentiles.average * 1.0e-6:.4f}\n"
            f"Average inter-token latency (ms): {stats.itl_percentiles.average * 1.0e-6:.4f}\n"
            f"Acceptance Rate (Speculative):\t  {stats.acceptance_rate:.2f}\n"
            f"\n"
            "===========================================================\n"
            "= GENERATION LATENCY BREAKDOWN \n"
            "===========================================================\n"
            f"MIN (ms): {stats.generation_latency_percentiles.minimum * 1.0e-6:.4f}\n"
            f"MAX (ms): {stats.generation_latency_percentiles.maximum * 1.0e-6:.4f}\n"
            f"AVG (ms): {stats.generation_latency_percentiles.average * 1.0e-6:.4f}\n"
            f"P90 (ms): {stats.generation_latency_percentiles.p50 * 1.0e-6:.4f}\n"
            f"P95 (ms): {stats.generation_latency_percentiles.p95 * 1.0e-6:.4f}\n"
            f"P99 (ms): {stats.generation_latency_percentiles.p99 * 1.0e-6:.4f}\n"
            f"\n"
            "===========================================================\n"
            "= ACCEPTANCE BREAKDOWN \n"
            "===========================================================\n"
            f"MIN: {stats.acceptance_percentiles.minimum:.2f}\n"
            f"MAX: {stats.acceptance_percentiles.maximum:.2f}\n"
            f"AVG: {stats.acceptance_percentiles.average:.2f}\n"
            f"P90: {stats.acceptance_percentiles.p50:.2f}\n"
            f"P95: {stats.acceptance_percentiles.p95:.2f}\n"
            f"P99: {stats.acceptance_percentiles.p99:.2f}\n"
            f"\n"
            "===========================================================\n")

        logger.info(logging_info)
        return stats
