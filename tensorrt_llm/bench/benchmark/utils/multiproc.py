from __future__ import annotations

import json
import multiprocessing as mp
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from threading import Event, Thread
from time import monotonic_ns, sleep
from typing import Generator, List, Tuple

import tensorrt_llm.bindings.executor as trtllm
from tensorrt_llm.bench.dataclasses.configuration import RuntimeConfig
from tensorrt_llm.bench.dataclasses.reporting import ResponseTuple, StatsKeeper
from tensorrt_llm.bench.dataclasses.statistics import BenchmarkStatistics
from tensorrt_llm.logger import logger


class ExecutorManager:
    """Utility class for managing a TRT-LLM Executor instance."""

    def __init__(self,
                 runtime_cfg: RuntimeConfig,
                 response_queue: mp.Queue,
                 iteration_log: Path = None) -> None:
        """Initialize the ExecutorManager.

        Args:
            runtime_cfg (RuntimeConfig): Execution runtime configuration.
            response_queue (mp.Queue): Process-safe queue for passing request
            responses to main process.
            iteration_log (Path): Path to iteration log stored at end of run.
        """
        logger.info("Initializing Executor.")
        # Runtime related properties.
        self.runtime_config: RuntimeConfig = runtime_cfg
        # Runtime tracking and multiprocessing.
        self.responses = response_queue
        self._shutdown = Event()
        self.backend_ready = Event()
        self._resp_daemon_finished = Event()

        config = self.runtime_config.get_config()
        self.iteration_log = iteration_log
        config.iter_stats_max_iterations = 100000000 if self.iteration_log else 0
        self.executor = trtllm.Executor(self.runtime_config.engine_dir,
                                        trtllm.ModelType.DECODER_ONLY,
                                        executor_config=config)

        logger.info("WAITING ON EXECUTOR...")
        while not self.executor.can_enqueue_requests():
            logger.info("Waiting for executor to stand up...")
            sleep(1)

        self.backend_ready.set()

        self.response_thread = Thread(target=self.response_daemon)
        self.response_thread.start()

    def enqueue(self, *requests: trtllm.Request) -> Generator[Tuple[int, int]]:
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

    def dump_extra_stats(self) -> None:
        if self.iteration_log is not None:
            with open(self.iteration_log, "w") as iter_log:
                for iteration in self.executor.get_latest_iteration_stats():
                    iter_log.write(f"{iteration.to_json_str()}\n")

    def response_daemon(self) -> None:
        """Daemon method for retrieving messages from the Executor."""

        logger.info("Starting response daemon...")

        def _process_response() -> None:
            responses = self.executor.await_responses(timeout=timedelta(
                microseconds=0.00000000000001))
            now = monotonic_ns()
            if len(responses) > 0:
                self.responses.put([
                    ResponseTuple(now, r.request_id, r.result.is_final,
                                  r.has_error(), r.result.output_token_ids[0],
                                  r.result.decoding_iter, None)
                    for r in responses
                ])

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
        streaming: bool,
        iteration_log: Path = None,
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
            streaming (bool): Enable/disable streaming mode.
            iteration_log (Path): Path to iteration log stored at end of run.

        """
        logger.info(
            f"Initializing Throughput Benchmark. [rate={request_rate} req/s]")
        # Dataset and input properties.
        self.requests = dataset
        self.delay_func = lambda x: sleep(
            x) if request_rate > 0 else lambda x: None
        self.request_delay = 1.0 / request_rate

        # Runtime configuration for Executor
        self.runtime_config = deepcopy(runtime_cfg)
        self.streaming = streaming
        self.iteration_log = iteration_log
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
        logger.info("WAITING ON BACKEND TO BE READY...")
        self.executor.backend_ready.wait()
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
        self.executor = ExecutorManager(
            self.runtime_config,
            self.response_queue,
            iteration_log=self.iteration_log,
        )

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

    def dump_extra_stats(self) -> None:
        """Write extended stats to a file."""
        self.executor.dump_extra_stats()

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
                responses: Tuple[
                    int,
                    List[ResponseTuple]] = self.response_queue.get_nowait()
                for response in responses:
                    self.statistics.register_response(
                        response.request_id,
                        response.timestamp,
                        response.final,
                        response.error,
                        response.decoding_iteration,
                        response.tokens,
                        None,  # time_on_first_token
                    )

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

    def report_statistics(self) -> BenchmarkStatistics:
        """Report internal statistics about benchmark."""

        config_path = self.runtime_config.engine_dir / "config.json"
        with open(config_path, "r") as config:
            engine_config = json.load(config)

        stats = self.statistics.generate_statistics_summary()
        rt_cfg = self.runtime_config
        build_cfg = engine_config["build_config"]
        pretrain_cfg = engine_config["pretrained_config"]
        total_latency_s = stats.total_latency_ns / 1.0e9

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
            f"Issue Rate (req/sec):\t{stats.issue_rate_ns * 1e9:.4E}\n"
            f"\n"
            "===========================================================\n"
            "= PERFORMANCE OVERVIEW \n"
            "===========================================================\n"
            f"Number of requests:\t\t{stats.num_requests}\n"
            f"Average Input Length (tokens):\t{stats.average_input_length:.4f}\n"
            f"Average Output Length (tokens):\t{stats.average_output_length:.4f}\n"
            f"Token Throughput (tokens/sec):\t{stats.total_output_tokens / total_latency_s:.4f}\n"
            f"Request Throughput (req/sec):\t{stats.num_requests / total_latency_s:.4f}\n"
            f"Total Latency (ms):\t\t{stats.total_latency_ns * 1.0e-6:.4f}\n")

        if self.streaming:
            logging_info = (
                f"{logging_info}"
                "\n"
                "===========================================================\n"
                "= STREAMING STATISTICS \n"
                "===========================================================\n"
                f"Average request latency (ms):\t\t{stats.request_latency_percentiles.average * 1.0e-6:.4f}\n"
                f"Average time-to-first-token (ms):\t{stats.ttft_percentiles.average * 1.0e-6:.4f}\n"
                f"Average inter-token latency (ms):\t{stats.itl_percentiles.average * 1.0e-6:.4f}\n"
            )

        logging_info = (
            f"{logging_info}"
            "\n===========================================================\n")

        logger.info(logging_info)
        return stats
