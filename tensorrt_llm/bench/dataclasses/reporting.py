from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List, NamedTuple

from tensorrt_llm._torch.pyexecutor.pytorch_model_engine import \
    validate_and_set_kv_cache_quant
from tensorrt_llm.bench.dataclasses.configuration import RuntimeConfig
from tensorrt_llm.bench.dataclasses.statistics import (BenchmarkStatistics,
                                                       PercentileStats,
                                                       RequestRecord)
from tensorrt_llm.logger import Logger


class ResponseTuple(NamedTuple):
    """
    A tuple for recording responses to requests.

    DEPRECATED after switching to LLM API.
    """
    timestamp: int
    request_id: int
    final: bool
    error: bool
    tokens: List[int]
    decoding_iteration: int
    time_on_first_token: int


class NewRequestTuple(NamedTuple):
    """
    A tuple for recording new requests.

    DEPRECATED after switching to LLM API.
    """
    timestamp: int
    request_id: int
    input_length: int


class NewRequestPerfItemTuple(NamedTuple):
    """A tuple for recording new requests and their responses."""
    start_timestamp: int
    end_timestamp: int
    request_id: int
    num_input_tokens: int
    response_is_final: bool
    error: bool
    tokens: List[int]
    decoding_iteration: int
    time_on_first_token: int


class StatsKeeper:
    """A statistics keeper for benchmarking."""

    def __init__(self) -> None:
        self.requests: Dict[int, RequestRecord] = defaultdict(RequestRecord)
        self.num_complete: int = 0

    def register_request(
        self,
        request_id: int,
        timestamp: int,
        num_tokens: int,
    ) -> None:
        """Register a new request.

        DEPRECATED after switching to LLM API.

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
        """
        Register a response from a new request.

        DEPRECATED after switching to LLM API.
        """
        record = self.requests[request_id]
        record.register_event(error, final, timestamp, decode_iter, tokens,
                              time_on_first_token)
        if final:
            self.num_complete = self.num_complete + 1

    def register_request_perf_item(self,
                                   request_perf_item: NewRequestPerfItemTuple):
        """
        Register request perf items, used exclusively with LLM API.
        """
        record = self.requests[request_perf_item.request_id]
        record.num_input_tokens = request_perf_item.num_input_tokens
        record.start_timestamp = request_perf_item.start_timestamp
        record.register_event(request_perf_item.error,
                              request_perf_item.response_is_final,
                              request_perf_item.end_timestamp,
                              request_perf_item.decoding_iteration,
                              request_perf_item.tokens,
                              request_perf_item.time_on_first_token)
        if request_perf_item.response_is_final:
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
            request_ar = entry.num_generated_tokens / (entry.decode_iteration +
                                                       1)

            request_latencies.append(entry.end_to_end_latency)
            generation_latencies.append(entry.generation_time)
            generation_throughputs.append(entry.output_token_throughput)
            ttft_times.append(entry.time_to_first_token)
            intertoken_avg_latencies.append(entry.intertoken_latency)
            request_acceptance.append(request_ar)
            total_decoding_iterations += entry.decode_iteration + 1

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
                      kwargs: Dict[str, Any],
                      streaming: bool = False) -> BenchmarkStatistics:
    """Report internal statistics about benchmark.

    Args:
        statistics (StatsKeeper): A statistics container.
        rt_cfg (RuntimeConfig): Configuration for the run.
        logger (Logger): A logger for logging.
        kwargs (Dict[str, Any]): Dictionary of LLM API kwargs.
        streaming (bool, optional): Streaming benchmark used. Defaults to False.

    Returns:
        BenchmarkStatistics: Benchmark statistics for the provided keeper.
    """
    if rt_cfg.backend != 'pytorch':
        config_path = rt_cfg.engine_dir / "config.json"
        with open(config_path, "r") as config:
            engine_config = json.load(config)
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
            f"\n")
    else:
        from tensorrt_llm._torch.model_config import ModelConfig
        from tensorrt_llm._utils import torch_dtype_to_str

        model = rt_cfg.model_path or rt_cfg.model
        model_config = ModelConfig.from_pretrained(model,
                                                   trust_remote_code=True)
        validate_and_set_kv_cache_quant(
            model_config, kwargs["pytorch_backend_config"].kv_cache_dtype)

        dtype = torch_dtype_to_str(model_config.pretrained_config.torch_dtype)
        quant_algo = model_config.quant_config.quant_algo
        kv_cache_quant_algo = model_config.quant_config.kv_cache_quant_algo

        logging_info = (
            "\n\n===========================================================\n"
            "= PyTorch backend\n"
            "===========================================================\n"
            f"Model:\t\t\t{rt_cfg.model}\n"
            f"Model Path:\t\t{rt_cfg.model_path}\n"
            f"TensorRT-LLM Version:\t{rt_cfg.sw_version}\n"
            f"Dtype:\t\t\t{dtype}\n"
            f"KV Cache Dtype:\t\t{kv_cache_quant_algo}\n"
            f"Quantization:\t\t{quant_algo}\n"
            # TODO
            # f"Max Input Length:\t{build_cfg['max_input_len']}\n"
            # f"Max Sequence Length:\t{build_cfg['max_seq_len']}\n"
            f"\n")

    stats = statistics.generate_statistics_summary()

    total_latency_s = stats.total_latency_ns / 1.0e9

    logging_info += (
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


def report_latency_statistics(stats: StatsKeeper, rt_cfg: RuntimeConfig,
                              logger: Logger) -> BenchmarkStatistics:
    """Report internal statistics about low-latency.

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

    stats = stats.generate_statistics_summary()
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
