from __future__ import annotations

import json
import multiprocessing as mp
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from threading import Event, Thread
from time import monotonic_ns, sleep
from typing import Generator, List, Tuple

import click
from click_option_group import optgroup

import tensorrt_llm.bindings.executor as trtllm
from tensorrt_llm.bench.dataclasses import BenchmarkEnvironment
from tensorrt_llm.bench.enums import IFBSchedulingPolicy
from tensorrt_llm.bench.run.dataclasses import ResponseRecord, RuntimeConfig
from tensorrt_llm.bench.run.utils import (StatsKeeper, get_executor_request,
                                          get_settings_from_engine)
from tensorrt_llm.bench.utils.data import generate_dataset_from_stream
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
@click.pass_obj
def run_command(
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
    runtime_max_bs = params.pop("max_batch_size")
    runtime_max_bs = runtime_max_bs if runtime_max_bs else engine_bs
    runtime_max_tokens = params.pop("max_num_tokens")
    runtime_max_tokens = runtime_max_bs if runtime_max_tokens else engine_tokens
    kv_cache_percent = params.pop("kv_cache_free_gpu_mem_fraction")
    beam_width = params.pop("beam_width")

    # Update configuration with runtime options
    exec_settings["settings_config"]["kv_cache_percent"] = kv_cache_percent
    exec_settings["settings_config"]["max_batch_size"] = runtime_max_bs
    exec_settings["settings_config"]["max_num_tokens"] = runtime_max_tokens
    exec_settings["settings_config"]["beam_width"] = beam_width
    exec_settings["settings_config"][
        "scheduler_policy"] = IFBSchedulingPolicy.NO_EVICT
    # Construct the runtime configuration dataclass.
    runtime_config = RuntimeConfig(**exec_settings)

    # Dataset Loading and Preparation
    metadata, requests = generate_dataset_from_stream(dataset_path, model,
                                                      num_requests)
    # TODO: Verify that the engine can handle the max/min ISL/OSL.
    if metadata.max_sequence_length > engine_max_seq_len:
        raise RuntimeError(
            f"Engine supports a max sequence of {engine_max_seq_len}. Provided "
            "dataset contains a maximum sequence of "
            f"{metadata.max_sequence_length}. Please rebuild a new engine to"
            "support this dataset.")
    executor_requests = []
    while requests:
        request = requests.pop()
        executor_requests.append(
            get_executor_request(request, pad_id=-1, eos_id=-1))
        del request

    logger.info("Setting up benchmarker and infrastructure.")
    new_request_queue = mp.Queue()
    response_queue = mp.Queue()
    logger.set_level("error")
    benchmark = ThroughputBenchmark(
        dataset=executor_requests,
        request_rate=request_rate,
        runtime_cfg=runtime_config,
        request_queue=new_request_queue,
        response_queue=response_queue,
    )
    logger.set_level("info")
    try:
        logger.info("Ready to start benchmark.")
        benchmark.start_benchmark()
        benchmark.wait()
        benchmark.stop_benchmark()
        benchmark.report_statistics()
    except KeyboardInterrupt:
        logger.set_level("error")
        benchmark.stop_benchmark()
    finally:
        logger.set_level("error")
        benchmark.shutdown()


class ExecutorManager:
    """Utility class for managing a TRT-LLM Executor instance."""

    def __init__(self, runtime_cfg: RuntimeConfig,
                 response_queue: mp.Queue) -> None:
        """Initialize the ExecutorManager.

        Args:
            runtime_cfg (RuntimeConfig): Execution runtime configuration.
            response_queue (mp.Queue): Process-safe queue for passing request
            responses to main process.
        """
        logger.info("Initializing Executor.")
        # Runtime related properties.
        self.runtime_config: RuntimeConfig = runtime_cfg
        self.executor = trtllm.Executor(
            self.runtime_config.engine_dir,
            trtllm.ModelType.DECODER_ONLY,
            executor_config=self.runtime_config.get_config())

        # Runtime tracking and multiprocessing.
        self.responses = response_queue
        self._shutdown = Event()
        self._resp_daemon_finished = Event()

        self.response_thread = Thread(target=self.response_daemon)
        self.response_thread.start()

    def enqueue(self, *requests: trtllm.Request) -> Generator[int]:
        """Generate the next request identifier.

        Yields:
            Generator[int]: The request identifier of the last queued request.
        """
        for request in requests:
            req_id = self.executor.enqueue_request(request)
            yield req_id, len(request.input_token_ids)

    def stop(self) -> None:
        """Stop a running manager."""

        logger.info("Stopping response parsing.")
        self._shutdown.set()
        self.response_thread.join()
        logger.info("Parsing stopped.")

    def shutdown(self) -> None:
        """Shutdown daemon components."""

        if self.executor is not None:
            logger.info("Shutting down ExecutorServer.")
            self.executor.shutdown()

    def response_daemon(self) -> None:
        """Daemon method for retrieving messages from the Executor."""

        logger.info("Starting response daemon...")

        def _process_response() -> None:
            responses = self.executor.await_responses(timeout=timedelta(
                milliseconds=1))
            now = monotonic_ns()
            for response in responses:
                # logger.info("Pushing response to queue")
                self.responses.put(
                    ResponseRecord(
                        timestamp=now,
                        request_id=response.request_id,
                        has_error=response.has_error(),
                        is_final=response.result.is_final,
                        output_tokens=response.result.output_token_ids[0]))

        while not self._shutdown.is_set():
            _process_response()

        logger.info("Collecting last responses before shutdown.")
        # Reap the last messages before shutting down
        _process_response()
        self._resp_daemon_finished.set()
        logger.info("Completed request parsing.")


class ThroughputBenchmark:
    """Throughput benchmark utility class."""

    def __init__(
        self,
        dataset: List[trtllm.Request],
        request_rate: int,
        runtime_cfg: RuntimeConfig,
        request_queue: mp.Queue,
        response_queue: mp.Queue,
    ) -> None:
        """Initialize the throughput benchmark.

        Args:
            dataset (List[trtllm.Request]): A dataset of TRT-LLM requests to
            benchmark against.
            request_rate (int): Rate to deliver input requests to the backend.
            runtime_cfg (RuntimeConfig): Runtime configuration.
            request_queue (mp.Queue): Process-safe queue of request identifiers
            response_queue (mp.Queue): Process-safe queue for passing request
            responses to main process.
        """
        logger.info(f"Initializing Throughput Benchmark. [rate=%d req/s]")
        # Dataset and input properties.
        self.requests = dataset
        self.delay_func = lambda x: sleep(
            x) if request_rate > 0 else lambda x: None
        self.request_delay = 1.0 / request_rate

        # Runtime configuration for Executor
        self.runtime_config = deepcopy(runtime_cfg)
        self.executor = None

        # Request and response reporting structures
        self.new_request_queue = request_queue
        self.response_queue = response_queue

        # Benchmark stats and time tracking.
        self.start_time = None
        self.end_time = None
        self.submitted_requests = 0
        self.statistics = StatsKeeper()

        # Multiprocessing for handling request load generation
        # and response parsing.
        self.stop = mp.Event()
        self.parsing_complete = mp.Event()
        self.request_thread: Thread = Thread(target=self.enqueue_process)
        self.stats_process: Thread = Thread(target=self.collect_statistics)

    def enqueue_process(self) -> None:
        """Method for starting enqueueing requests."""
        logger.info("Request serving started.")

        request_generator = self.executor.enqueue(*self.requests)
        # Iterate the generator until we run out of requests.
        # Note the walrus operator.
        while ((request := next(request_generator, False))
               and not self.stop.is_set()):
            self.submitted_requests += 1
            timestamp = monotonic_ns()
            self.new_request_queue.put((timestamp, request[0], request[1]))
            self.delay_func(self.request_delay)
        logger.info("Request serving stopped.")

    def start_benchmark(self) -> None:
        """Start the benchmark."""
        # Start the ExecutorManager for running the backend.
        self.executor = ExecutorManager(self.runtime_config,
                                        self.response_queue)
        logger.info("Executor started.")
        # Note the time we started the thread.
        self.start_time = monotonic_ns()
        self.request_thread.start()
        # Start the statistics thread.
        self.stats_process.start()
        logger.info("Benchmark started.")

    def stop_benchmark(self) -> None:
        """Stop the benchmark and clean up backend and threads."""
        logger.info("Stop received.")
        self.stop.set()
        self.executor.stop()
        self.request_thread.join()
        logger.info("Request generator successfully joined.")
        self.stats_process.join()
        logger.info("Statistics process successfully joined.")

    def shutdown(self) -> None:
        """Shutdown the backend."""
        logger.info("Benchmark Shutdown called!")
        if self.executor is not None:
            self.executor.shutdown()
        logger.info("Executor shutdown.")

    def wait(self) -> bool:
        """Wait (blocking) on the benchmark.

        Returns:
            bool: Return whether the event is set.
        """
        return not self.parsing_complete.wait()

    def collect_statistics(self) -> None:
        """Collect statistics (daemon method)."""
        logger.info("Starting statistics collection.")

        def _process_requests() -> None:
            while not self.new_request_queue.empty():
                new_request: Tuple[float,
                                   int] = self.new_request_queue.get_nowait()
                self.statistics.register_request(new_request[1], new_request[0],
                                                 new_request[2])

            while not self.response_queue.empty():
                response: ResponseRecord = self.response_queue.get_nowait()
                self.statistics.register_response(response)

        logger.info("Collecting live stats...")
        # TODO: Revisit this conditional, if the request rate is slow enough this
        # will probably prematurely trip. We will likely need a conditional that
        # captures a new event for submission being complete, with the stop event
        # overriding it if detected.
        while not self.stop.is_set(
        ) and self.statistics.num_complete < self.submitted_requests:
            _process_requests()

        logger.info("Collecting last stats...")
        _process_requests()
        self.end_time = monotonic_ns()
        self.parsing_complete.set()
        logger.info("Ending statistics collection.")

    def report_statistics(self) -> None:
        """Report internal statistics about benchmark."""

        config_path = self.runtime_config.engine_dir / "config.json"
        with open(config_path, "r") as config:
            engine_config = json.load(config)

        stats = self.statistics.generate_statistics_summary()
        rt_cfg = self.runtime_config
        build_cfg = engine_config["build_config"]
        pretrain_cfg = engine_config["pretrained_config"]
        total_latency_s = stats.total_latency_ns / 1.0e9

        logger.info(
            "\n===========================================================\n"
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
            f"KV Memory Percentage:\t{rt_cfg.settings_config.kv_cache_percent * 100.0}%\n"
            f"Issue Rate (req/sec):\t{stats.issue_rate_ns * 1e9}"
            f"\n"
            "===========================================================\n"
            "= STATISTICS\n"
            "===========================================================\n"
            f"Number of requests:\t\t{stats.num_requests}\n"
            f"Average Input Length (tokens):\t{stats.average_input_length}\n"
            f"Average Output Length (tokens):\t{stats.average_output_length}\n"
            f"Token Throughput (tokens/sec):\t{stats.total_output_tokens / total_latency_s}\n"
            f"Request Throughput (req/sec):\t{stats.num_requests / total_latency_s}\n"
            f"Total Latency (seconds):\t{total_latency_s}\n"
            "===========================================================\n")
