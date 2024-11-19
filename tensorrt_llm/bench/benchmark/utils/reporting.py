from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List, NamedTuple

from tensorrt_llm.bench.benchmark.dataclasses import (BenchmarkStatistics,
                                                      PercentileStats,
                                                      RequestRecord,
                                                      RuntimeConfig)
from tensorrt_llm.logger import Logger


class ResponseTuple(NamedTuple):
    """A tuple for recording responses to requests."""
    timestamp: int
    request_id: int
    final: bool
    error: bool
    tokens: List[int]
    decoding_iteration: int
    time_on_first_token: int


class NewRequestTuple(NamedTuple):
    """A tuple for recording new requests."""
    timestamp: int
    request_id: int
    input_length: int


class StatsKeeper:
    """A statistics keeper for benchmarking."""

    def __init__(self) -> None:
        self.requests: Dict[RequestRecord] = defaultdict(RequestRecord)
        self.num_complete: int = 0

    def register_request(
        self,
        request_id: int,
        timestamp: int,
        num_tokens: int,
    ) -> None:
        """Register a new request.

        Args:
            request_id (int): Identifier of the request.
            timestamp (int): Timestamp of when the request was submitted.
            num_tokens (int): Number of input tokens in the request.
        """
        record = self.requests[request_id]
        record.num_input_tokens = num_tokens
        record.start_timestamp = timestamp

    def register_response(self, request_id: int, timestamp: int, final: bool,
                          error: bool, decode_iter: int, tokens: List[int],
                          time_on_first_token: int) -> None:
        record = self.requests[request_id]
        record.register_event(error, final, timestamp, decode_iter, tokens,
                              time_on_first_token)
        if final:
            self.num_complete = self.num_complete + 1

    def register_request_perf_items(self, request_id: int, start_timestamp: int,
                                    num_input_tokens: int, end_timestamp: int,
                                    response_is_final: bool, error: bool,
                                    decode_iter: int, tokens: List[int],
                                    time_on_first_token: int):
        """
        Register request perf items, used exclusively with LLM API.
        """
        record = self.requests[request_id]
        record.num_input_tokens = num_input_tokens
        record.start_timestamp = start_timestamp
        record.register_event(error, response_is_final, end_timestamp,
                              decode_iter, tokens, time_on_first_token)
        if response_is_final:
            self.num_complete = self.num_complete + 1

    def generate_statistics_summary(self) -> None:
        """Generate summary statistics from internally stored statistics.

        Returns:
            BenchmarkStatistic: Benchmark run statistics.
        """
        total_output_tokens: int = 0
        total_input_tokens: int = 0
        num_requests = len(self.requests)
        start_time = float("inf")
        end_time = -1

        request_latencies = []
        generation_latencies = []
        generation_throughputs = []
        intertoken_avg_latencies = []
        request_acceptance = []
        total_decoding_iterations = 0
        ttft_times = []
        last_queue_time = 0.0
        queue_time_total = 0.0

        for entry in self.requests.values():
            start_time = min(entry.start_timestamp, start_time)
            end_time = max(entry.end_timestamp, end_time)
            last_queue_time = max(entry.start_timestamp, last_queue_time)
            request_ar = entry.num_generated_tokens / entry.decode_iteration

            request_latencies.append(entry.end_to_end_latency)
            generation_latencies.append(entry.generation_time)
            generation_throughputs.append(entry.output_token_throughput)
            ttft_times.append(entry.time_to_first_token)
            intertoken_avg_latencies.append(entry.intertoken_latency)
            request_acceptance.append(request_ar)
            total_decoding_iterations += entry.decode_iteration

            total_output_tokens += entry.num_output_tokens
            total_input_tokens += entry.num_input_tokens

        global_acceptance_rate = total_output_tokens / total_decoding_iterations
        queue_time_total = last_queue_time - start_time
        percentile_request_accept = PercentileStats.from_iterable(
            request_acceptance) if request_acceptance else None

        stats = BenchmarkStatistics(
            num_requests=num_requests,
            total_latency_ns=end_time - start_time,
            total_output_tokens=total_output_tokens,
            total_input_tokens=total_input_tokens,
            acceptance_rate=global_acceptance_rate,
            request_latency_percentiles=PercentileStats.from_iterable(
                request_latencies),
            itl_percentiles=PercentileStats.from_iterable(
                intertoken_avg_latencies),
            ttft_percentiles=PercentileStats.from_iterable(ttft_times),
            generation_tp_percentiles=PercentileStats.from_iterable(
                generation_throughputs),
            generation_latency_percentiles=PercentileStats.from_iterable(
                generation_latencies),
            issue_rate_ns=queue_time_total / num_requests,
            acceptance_percentiles=percentile_request_accept,
        )

        return stats


def report_statistics(statistics: StatsKeeper,
                      rt_cfg: RuntimeConfig,
                      logger: Logger,
                      streaming: bool = False) -> BenchmarkStatistics:
    """Report internal statistics about benchmark.

    Args:
        statistics (StatsKeeper): A statistics container.
        rt_cfg (RuntimeConfig): Configuration for the run.
        logger (Logger): A logger for logging.
        streaming (bool, optional): Streaming benchmark used. Defaults to False.

    Returns:
        BenchmarkStatistics: Benchmark statistics for the provided keeper.
    """

    config_path = rt_cfg.engine_dir / "config.json"
    with open(config_path, "r") as config:
        engine_config = json.load(config)

    stats = statistics.generate_statistics_summary()
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

    if streaming:
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
