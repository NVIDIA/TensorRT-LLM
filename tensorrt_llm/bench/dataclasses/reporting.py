from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List, NamedTuple

from tensorrt_llm._torch.pyexecutor.model_loader import \
    validate_and_set_kv_cache_quant
from tensorrt_llm.bench.dataclasses.configuration import RuntimeConfig
from tensorrt_llm.bench.dataclasses.general import DatasetMetadata
from tensorrt_llm.bench.dataclasses.statistics import (BenchmarkStatistics,
                                                       PercentileStats,
                                                       RequestRecord)
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.logger import Logger
from tensorrt_llm.models.modeling_utils import SpeculativeDecodingMode


class PerfItemTuple(NamedTuple):
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

    def register_request_perf_item(self, request_perf_item: PerfItemTuple):
        """
        Register request perf items, used exclusively with LLM API.
        """
        record = self.requests[request_perf_item.request_id]
        record.id = request_perf_item.request_id
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

    def generate_statistics_summary(self, max_draft_tokens: int) -> None:
        """Generate summary statistics from internally stored statistics.

        Returns:
            BenchmarkStatistic: Benchmark run statistics.
        """
        total_input_tokens: int = 0
        num_requests = len(self.requests)
        start_time = float("inf")
        end_time = -1

        request_latencies = []
        generation_latencies = []
        generation_throughputs = []
        output_throughput_per_user = []

        intertoken_avg_latencies = []
        output_tokens = []
        total_decoding_iterations = 0
        ttft_times = []
        last_queue_time = 0.0
        queue_time_total = 0.0

        num_draft_tokens = []
        num_accepted_draft_tokens = []
        draft_acceptance_rate = []
        acceptance_length = []

        for entry in self.requests.values():
            start_time = min(entry.start_timestamp, start_time)
            end_time = max(entry.end_timestamp, end_time)
            last_queue_time = max(entry.start_timestamp, last_queue_time)

            request_latencies.append(entry.end_to_end_latency)
            generation_latencies.append(entry.generation_time)
            generation_throughputs.append(entry.generation_token_throughput)
            ttft_times.append(entry.time_to_first_token)
            intertoken_avg_latencies.append(entry.intertoken_latency)
            output_throughput_per_user.append(entry.output_token_throughput)
            total_decoding_iterations += entry.decode_iteration + 1

            output_tokens.append(entry.num_total_output_tokens)
            total_input_tokens += entry.num_input_tokens

            # For speculative decoding, we need to track the number of draft tokens per request and the number of accepted draft tokens per request
            if max_draft_tokens > 0:
                num_draft_tokens.append(max_draft_tokens *
                                        (entry.decode_iteration + 1))
                num_accepted_draft_tokens.append(entry.num_total_output_tokens -
                                                 entry.decode_iteration - 1)
                draft_acceptance_rate.append(
                    float(num_accepted_draft_tokens[-1]) /
                    float(num_draft_tokens[-1]))
                acceptance_length.append(entry.num_total_output_tokens /
                                         (entry.decode_iteration + 1))

        global_acceptance_length = sum(
            output_tokens) / total_decoding_iterations
        queue_time_total = last_queue_time - start_time

        num_draft_tokens_percentiles = PercentileStats.from_iterable(
            num_draft_tokens) if num_draft_tokens else None
        num_accepted_draft_tokens_percentiles = PercentileStats.from_iterable(
            num_accepted_draft_tokens) if num_accepted_draft_tokens else None
        draft_acceptance_rate_percentiles = PercentileStats.from_iterable(
            draft_acceptance_rate) if draft_acceptance_rate else None
        acceptance_length_percentiles = PercentileStats.from_iterable(
            acceptance_length) if acceptance_length else None

        stats = BenchmarkStatistics(
            num_requests=num_requests,
            total_latency_ns=end_time - start_time,
            total_output_tokens=sum(output_tokens),
            total_input_tokens=total_input_tokens,
            request_latency_percentiles=PercentileStats.from_iterable(
                request_latencies),
            tpot_percentiles=PercentileStats.from_iterable(
                intertoken_avg_latencies),
            output_throughput_percentiles=PercentileStats.from_iterable(
                output_throughput_per_user),
            ttft_percentiles=PercentileStats.from_iterable(ttft_times),
            generation_tp_percentiles=PercentileStats.from_iterable(
                generation_throughputs),
            generation_latency_percentiles=PercentileStats.from_iterable(
                generation_latencies),
            token_percentiles=PercentileStats.from_iterable(output_tokens),
            issue_rate_ns=queue_time_total / num_requests,
            acceptance_length=global_acceptance_length,
            num_draft_tokens_percentiles=num_draft_tokens_percentiles,
            num_accepted_draft_tokens_percentiles=
            num_accepted_draft_tokens_percentiles,
            draft_acceptance_rate_percentiles=draft_acceptance_rate_percentiles,
            acceptance_length_percentiles=acceptance_length_percentiles,
        )

        return stats


class ReportUtility:
    """A utility for reporting statistics."""

    def __init__(self,
                 statistics: StatsKeeper,
                 dataset_metadata: DatasetMetadata,
                 rt_cfg: RuntimeConfig,
                 logger: Logger,
                 kwargs: Dict[str, Any],
                 streaming: bool = False) -> None:
        """Initialize the ReportingController.

        Args:
            statistics (StatsKeeper): A statistics container.
            dataset_metadata (DatasetMetadata): Metadata about the dataset.
            rt_cfg (RuntimeConfig): Configuration for the run.
            logger (Logger): A logger for logging.
            streaming (bool, optional): Streaming benchmark used. Defaults to False.
        """
        self.dataset_metadata = dataset_metadata
        self.rt_cfg = rt_cfg
        self.logger = logger
        self.kwargs = kwargs
        self.raw_statistics = statistics
        self.statistics = statistics.generate_statistics_summary(
            self.get_max_draft_len())
        self.streaming = streaming

    @staticmethod
    def convert_to_ms(ns: float) -> float:
        """Convert nanoseconds to milliseconds."""
        return ns * 1.0e-6

    @staticmethod
    def convert_to_s(ns: float) -> float:
        """Convert nanoseconds to seconds."""
        return ns * 1.0e-9

    @staticmethod
    def convert_rate_to_s(rate: float) -> float:
        """Convert rate to seconds."""
        return rate * 1.0e9

    @property
    def request_throughput_req_s(self) -> float:
        """Request throughput in requests per second."""
        return self.convert_rate_to_s(self.statistics.request_throughput_ns)

    @property
    def output_throughput_tok_s(self) -> float:
        """Output throughput in tokens per second."""
        return self.convert_rate_to_s(self.statistics.output_throughput_tok_ns)

    @property
    def total_token_throughput_tok_s(self) -> float:
        """Total token throughput in tokens per second."""
        return self.convert_rate_to_s(
            self.statistics.total_token_throughput_tok_ns)

    @property
    def per_user_generation_token_throughput_s(self) -> float:
        """Output throughput per user in tokens per second."""
        return self.convert_rate_to_s(
            self.statistics.per_user_generation_token_throughput_ns)

    @property
    def per_user_output_throughput_tok_s(self) -> float:
        """Output throughput per user in tokens per second."""
        return self.convert_rate_to_s(
            self.statistics.output_throughput_tok_ns_per_user)

    def get_output_tokens(self, tokenizer) -> Dict[int, List[str]]:
        retval = {}
        for req_id, request in self.raw_statistics.requests.items():
            output_str = tokenizer.decode(request.tokens)
            retval[req_id] = output_str
        return dict(sorted(retval.items()))

    def get_request_info(self, tokenizer) -> Dict[int, List[str]]:
        requests = []
        for request in self.raw_statistics.requests.values():
            entry = request.model_dump()
            entry["output"] = tokenizer.decode(entry["tokens"])
            entry["output_tokens"] = len(entry["tokens"])
            entry.pop("tokens")
            requests.append(entry)
        return requests

    def get_statistics_dict(self) -> Dict[str, Any]:
        """Get statistics as a dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing benchmark statistics.
        """
        stats_dict = {
            "engine": {
                "model": self.rt_cfg.model,
                "model_path": str(self.rt_cfg.model_path),
                "engine_dir": str(self.rt_cfg.engine_dir),
                "version": self.rt_cfg.sw_version,
            },
        }

        # Retrieve KV cache information.
        kv_cache_config = self.kwargs.get("kv_cache_config", KvCacheConfig())
        if isinstance(kv_cache_config, KvCacheConfig):
            kv_cache_dtype = kv_cache_config.dtype
            kv_cache_mem_percent = kv_cache_config.free_gpu_memory_fraction
        elif isinstance(kv_cache_config, dict):
            kv_cache_dtype = kv_cache_config.get("dtype", "auto")
            kv_cache_mem_percent = kv_cache_config.get(
                "free_gpu_memory_fraction")
        else:
            raise ValueError(
                f"Invalid kv_cache_config type: {type(kv_cache_config)}.")

        kv_cache_mem_percent = kv_cache_mem_percent \
            if kv_cache_mem_percent is not None else None

        # Engine/Backend details
        if self.rt_cfg.backend not in ('pytorch', '_autodeploy'):
            config_path = self.rt_cfg.engine_dir / "config.json"
            with open(config_path, "r") as config:
                engine_config = json.load(config)
            build_cfg = engine_config["build_config"]
            pretrain_cfg = engine_config["pretrained_config"]

            stats_dict["engine"] |= {
                "backend":
                "TRT",
                "dtype":
                pretrain_cfg["dtype"],
                "kv_cache_dtype":
                pretrain_cfg["quantization"]["kv_cache_quant_algo"],
                "quantization":
                pretrain_cfg["quantization"]["quant_algo"],
                "max_input_length":
                build_cfg["max_input_len"],
                "max_sequence_length":
                build_cfg["max_seq_len"]
            }
        else:
            from tensorrt_llm._torch.model_config import ModelConfig
            from tensorrt_llm._utils import torch_dtype_to_str

            model = self.rt_cfg.model_path or self.rt_cfg.model
            model_config = ModelConfig.from_pretrained(model,
                                                       trust_remote_code=True)

            validate_and_set_kv_cache_quant(model_config, kv_cache_dtype)

            stats_dict["engine"] |= {
                "backend":
                "Pytorch",
                "dtype":
                torch_dtype_to_str(model_config.torch_dtype
                                   or model_config.pretrained_config.
                                   get_text_config().torch_dtype),
                "kv_cache_dtype":
                model_config.quant_config.kv_cache_quant_algo,
                "quantization":
                model_config.quant_config.quant_algo
            }

        # World and runtime info
        stats_dict["world_info"] = {
            "tp_size": self.rt_cfg.mapping["tp_size"],
            "pp_size": self.rt_cfg.mapping["pp_size"],
            "ep_size": self.rt_cfg.mapping["moe_ep_size"],
            "world_size": self.rt_cfg.mapping["world_size"],
            "max_batch_size": self.rt_cfg.settings_config.max_batch_size,
            "max_num_tokens": self.rt_cfg.settings_config.max_num_tokens,
            "scheduling_policy": self.rt_cfg.settings_config.scheduler_policy,
            "kv_cache_percentage": kv_cache_mem_percent,
            "issue_rate": self.convert_rate_to_s(self.statistics.issue_rate_ns)
        }

        # Request details
        stats_dict["request_info"] = {
            "num_requests": self.statistics.num_requests,
            "avg_num_concurrent_requests":
            self.statistics.avg_concurrent_requests,
            "avg_input_length": self.statistics.average_input_length,
            "avg_output_length": self.statistics.average_output_length
        }

        # Performance stats
        stats_dict["performance"] = {
            # End-to-End Latency (last request end - 1st request start)
            "total_latency_ms":
            self.convert_to_ms(self.statistics.total_latency_ns),
            # Average per request latency (sum request latencies / num requests)
            "avg_request_latency_ms":
            self.convert_to_ms(
                self.statistics.request_latency_percentiles.average),
            # Request throughput (num requests / end-to-end latency)
            "request_throughput_req_s":
            self.request_throughput_req_s,
            # NOTE: All mention of "output" below is in reference to OSL tokens
            # including the first token.
            # Output throughput (total output (OSL) tokens / end-to-end latency)
            "system_output_throughput_tok_s":
            self.output_throughput_tok_s,
            # Output throughput per user (average per request output throughput)
            "system_total_throughput_tok_s":
            self.total_token_throughput_tok_s,
            "output_throughput_per_user_tok_s":
            self.per_user_output_throughput_tok_s,
            # Output throughput per GPU (total throughput / world size)
            "output_throughput_per_gpu_tok_s":
            self.output_throughput_tok_s / self.rt_cfg.mapping["world_size"],
            # Request latency percentiles
            "request_latency_percentiles_ms":
            self.statistics.request_latency_percentiles.model_dump(
                exclude_none=True, by_alias=True, mode='json') | {
                    k: self.convert_to_ms(v)
                    for k, v in self.statistics.request_latency_percentiles.
                    model_dump().items()
                },
        }

        if self.streaming:
            avg_tpot = self.convert_to_ms(
                self.statistics.per_user_time_per_output_token_ns)

            stats_dict["streaming_metrics"] = {
                # NOTE: Excludes TTFT by nature as this is a genphase calculation.
                "token_output_speed_tok_s":
                self.per_user_generation_token_throughput_s,
                # Average per request time-to-first-token (TTFT)
                "avg_ttft_ms":
                self.convert_to_ms(
                    self.statistics.per_user_time_to_first_token_ns),
                # Average per request token time-per-output-token (TPOT)
                "avg_tpot_ms":
                avg_tpot,
                # Average per request Time-per-output-token percentiles (TPOT)
                "tpot_percentiles":
                self.statistics.tpot_percentiles.model_dump(
                    exclude_none=True, by_alias=True, mode='json') | {
                        k: self.convert_to_ms(v)
                        for k, v in
                        self.statistics.tpot_percentiles.model_dump().items()
                    },
                # Per request Time-to-first-token percentiles (TTFT)
                "ttft_percentiles":
                self.statistics.ttft_percentiles.model_dump(
                    exclude_none=True, by_alias=True, mode='json') | {
                        k: self.convert_to_ms(v)
                        for k, v in
                        self.statistics.ttft_percentiles.model_dump().items()
                    },
                "gen_tps_percentiles":
                self.statistics.generation_tp_percentiles.model_dump(
                    exclude_none=True, by_alias=True, mode='json') | {
                        k: self.convert_rate_to_s(v)
                        for k, v in self.statistics.generation_tp_percentiles.
                        model_dump().items()
                    },
            }

        spec_decoding, decoding_mode = False, None
        if (self.rt_cfg.decoding_config
                and self.rt_cfg.decoding_config.decoding_mode
                != SpeculativeDecodingMode.NONE):
            # cpp decoding
            spec_decoding = True
            decoding_mode = self.rt_cfg.decoding_config.decoding_mode.values[1]
        elif ("speculative_config" in self.kwargs
              and self.kwargs["speculative_config"] is not None):
            # pytorch speculative decoding
            spec_decoding = True
            decoding_mode = self.kwargs["speculative_config"].decoding_type
        if (spec_decoding):
            stats_dict["decoding_stats"] = {
                "mode":
                decoding_mode,
                "num_draft_tokens_percentiles":
                self.statistics.num_draft_tokens_percentiles.model_dump(
                    exclude_none=True, by_alias=True, mode='json')
                if self.statistics.num_draft_tokens_percentiles else None,
                "num_accepted_draft_tokens_percentiles":
                self.statistics.num_accepted_draft_tokens_percentiles.
                model_dump(exclude_none=True, by_alias=True, mode='json') if
                self.statistics.num_accepted_draft_tokens_percentiles else None,
                "draft_acceptance_rate_percentiles":
                self.statistics.draft_acceptance_rate_percentiles.model_dump(
                    exclude_none=True, by_alias=True, mode='json')
                if self.statistics.draft_acceptance_rate_percentiles else None,
                "acceptance_length_percentiles":
                self.statistics.acceptance_length_percentiles.model_dump(
                    exclude_none=True, by_alias=True, mode='json')
                if self.statistics.acceptance_length_percentiles else None
            }
        # Dataset metadata
        stats_dict["dataset"] = self.dataset_metadata.model_dump(by_alias=True,
                                                                 mode='json')

        return stats_dict

    def report_statistics(self) -> None:
        """Report internal statistics about benchmark.

        Returns:
            BenchmarkStatistics: Benchmark statistics for the provided keeper.
        """
        stats_dict = self.get_statistics_dict()
        engine = stats_dict["engine"]
        world_info = stats_dict["world_info"]
        requests = stats_dict["request_info"]
        perf = stats_dict["performance"]
        streaming = stats_dict.get("streaming_metrics")
        decoding = stats_dict.get("decoding_stats", None)

        backend_info = ""
        if self.rt_cfg.backend not in ('pytorch', '_autodeploy'):
            config_path = self.rt_cfg.engine_dir / "config.json"
            with open(config_path, "r") as config:
                engine_config = json.load(config)
            build_cfg = engine_config["build_config"]
            pretrain_cfg = engine_config["pretrained_config"]

            backend_info = (
                "\n\n===========================================================\n"
                "= ENGINE DETAILS\n"
                "===========================================================\n"
                f"Model:\t\t\t{engine['model']}\n"
                f"Model Path:\t\t{engine['model_path']}\n"
                f"Engine Directory:\t{engine['engine_dir']}\n"
                f"TensorRT LLM Version:\t{engine['version']}\n"
                f"Dtype:\t\t\t{pretrain_cfg['dtype']}\n"
                f"KV Cache Dtype:\t\t{pretrain_cfg['quantization']['kv_cache_quant_algo']}\n"
                f"Quantization:\t\t{pretrain_cfg['quantization']['quant_algo']}\n"
                f"Max Input Length:\t{build_cfg['max_input_len']}\n"
                f"Max Sequence Length:\t{build_cfg['max_seq_len']}\n"
                f"\n")
        else:
            backend_info = (
                "\n\n===========================================================\n"
                "= PYTORCH BACKEND\n"
                "===========================================================\n"
                f"Model:\t\t\t{engine['model']}\n"
                f"Model Path:\t\t{engine['model_path']}\n"
                f"TensorRT LLM Version:\t{engine['version']}\n"
                f"Dtype:\t\t\t{engine['dtype']}\n"
                f"KV Cache Dtype:\t\t{engine['kv_cache_dtype']}\n"
                f"Quantization:\t\t{engine['quantization']}\n"
                # TODO
                # f"Max Input Length:\t{build_cfg['max_input_len']}\n"
                # f"Max Sequence Length:\t{build_cfg['max_seq_len']}\n"
                f"\n")

        kv_cache_percentage = world_info.get("kv_cache_percentage", None)
        if kv_cache_percentage is not None:
            kv_cache_percentage = f"{kv_cache_percentage * 100.0:.2f}%"

        world_info = (
            "===========================================================\n"
            "= WORLD + RUNTIME INFORMATION \n"
            "===========================================================\n"
            f"TP Size:                {world_info['tp_size']}\n"
            f"PP Size:                {world_info['pp_size']}\n"
            f"EP Size:                {world_info['ep_size']}\n"
            f"Max Runtime Batch Size: {world_info['max_batch_size']}\n"
            f"Max Runtime Tokens:     {world_info['max_num_tokens']}\n"
            f"Scheduling Policy:      {world_info['scheduling_policy']}\n"
            f"KV Memory Percentage:   {kv_cache_percentage}\n"
            f"Issue Rate (req/sec):   {world_info['issue_rate']:.4E}\n"
            f"\n")

        req_lat_info = "\n".join(
            f"[Latency] {key.upper():<7}: {perf['request_latency_percentiles_ms'][key]:.4f}"
            for key in perf['request_latency_percentiles_ms'].keys())

        request_info = (
            "===========================================================\n"
            "= REQUEST DETAILS \n"
            "===========================================================\n"
            f"Number of requests:             {requests['num_requests']}\n"
            f"Number of concurrent requests:  {requests['avg_num_concurrent_requests']:.4f}\n"
            f"Average Input Length (tokens):  {requests['avg_input_length']:.4f}\n"
            f"Average Output Length (tokens): {requests['avg_output_length']:.4f}\n"
        )

        perf_header = (
            "===========================================================\n"
            "= PERFORMANCE OVERVIEW \n"
            "===========================================================\n")

        perf_stats = (
            f"Request Throughput (req/sec):                     {perf['request_throughput_req_s']:.4f}\n"
            f"Total Output Throughput (tokens/sec):             {perf['system_output_throughput_tok_s']:.4f}\n"
            f"Total Token Throughput (tokens/sec):              {perf['system_total_throughput_tok_s']:.4f}\n"
            f"Total Latency (ms):                               {perf['total_latency_ms']:.4f}\n"
            f"Average request latency (ms):                     {perf['avg_request_latency_ms']:.4f}\n"
            # Output Throughput includes context/first token.
            f"Per User Output Throughput [w/ ctx] (tps/user):   {perf['output_throughput_per_user_tok_s']:.4f}\n"
            f"Per GPU Output Throughput (tps/gpu):              {perf['output_throughput_per_gpu_tok_s']:.4f}\n"
        )

        if streaming:
            streaming = stats_dict["streaming_metrics"]
            itl = streaming["tpot_percentiles"]
            ttft = streaming["ttft_percentiles"]

            tpot_stats = "\n".join(
                f"[TPOT] {key.upper():<7}: {itl[key]:.4f}" for key in
                ["minimum", "maximum", "average", "p50", "p90", "p95", "p99"])

            ttft_stats = "\n".join(
                f"[TTFT] {key.upper():<7}: {ttft[key]:.4f}" for key in
                ["minimum", "maximum", "average", "p50", "p90", "p95", "p99"])

            gen_tps_stats = "\n".join(
                f"[GTPS] {key.upper():<7}: {streaming['gen_tps_percentiles'][key]:.4f}"
                for key in
                ["minimum", "maximum", "average", "p50", "p90", "p95", "p99"])

            perf_stats += (
                f"Average time-to-first-token [TTFT] (ms):          {streaming['avg_ttft_ms']:.4f}\n"
                f"Average time-per-output-token [TPOT] (ms):        {streaming['avg_tpot_ms']:.4f}\n"
                f"Per User Output Speed (tps/user):                 {streaming['token_output_speed_tok_s']:.4f}\n"
                "\n-- Per-Request Time-per-Output-Token [TPOT] Breakdown (ms)\n\n"
                f"{tpot_stats}\n"
                "\n-- Per-Request Time-to-First-Token [TTFT] Breakdown (ms) \n\n"
                f"{ttft_stats}\n"
                "\n-- Per-Request Generation Throughput [GTPS] Breakdown (tps/user)\n\n"
                f"{gen_tps_stats}\n")

        perf_stats += (
            "\n-- Request Latency Breakdown (ms) -----------------------\n\n"
            f"{req_lat_info}\n")

        decoding_stats = ""
        if decoding is not None:
            decoding = stats_dict["decoding_stats"]
            if self.get_max_draft_len() > 0:
                num_draft_tokens = decoding["num_draft_tokens_percentiles"]
                num_draft_tokens_stats = "\n".join(
                    f"[DT] {key.upper():<7}: {num_draft_tokens[key]:.2f}"
                    for key in [
                        "minimum", "maximum", "average", "p50", "p90", "p95",
                        "p99"
                    ])

                num_accepted_draft_tokens = decoding[
                    "num_accepted_draft_tokens_percentiles"]
                num_accepted_draft_tokens_stats = "\n".join(
                    f"[ADT] {key.upper():<7}: {num_accepted_draft_tokens[key]:.2f}"
                    for key in [
                        "minimum", "maximum", "average", "p50", "p90", "p95",
                        "p99"
                    ])

                draft_acceptance_rate = decoding[
                    "draft_acceptance_rate_percentiles"]
                draft_acceptance_rate_stats = "\n".join(
                    f"[DAR] {key.upper():<7}: {draft_acceptance_rate[key]:.2f}"
                    for key in [
                        "minimum", "maximum", "average", "p50", "p90", "p95",
                        "p99"
                    ])

                acceptance_length = decoding["acceptance_length_percentiles"]
                acceptance_length_stats = "\n".join(
                    f"[AL] {key.upper():<7}: {acceptance_length[key]:.2f}"
                    for key in [
                        "minimum", "maximum", "average", "p50", "p90", "p95",
                        "p99"
                    ])

                decoding_stats = (
                    "===========================================================\n"
                    f"= DECODING STATISTICS ({decoding['mode']})\n"
                    "===========================================================\n"
                    "\n"
                    "-- Number of Draft Tokens Details --------------------------------\n\n"
                    "\n"
                    f"{num_draft_tokens_stats}"
                    f"\n"
                    "-- Number of Accepted Draft Tokens Details --------------------------------\n\n"
                    f"{num_accepted_draft_tokens_stats}"
                    f"\n"
                    "-- Draft Acceptance Rate Details --------------------------------\n\n"
                    f"{draft_acceptance_rate_stats}"
                    f"\n"
                    "-- Acceptance Length Details --------------------------------\n\n"
                    f"{acceptance_length_stats}"
                    f"\n"
                    "===========================================================\n"
                )

        logging_info = (f"{backend_info}"
                        f"{request_info}"
                        f"{world_info}"
                        f"{perf_header}"
                        f"{perf_stats}"
                        f"{decoding_stats}"
                        f"{self.dataset_metadata.get_summary_for_print()}")
        self.logger.info(logging_info)
        return self.statistics

    def get_max_draft_len(self) -> int:
        """Get max_draft_len from speculative_config."""
        # Try to get from speculative_config
        if ("speculative_config" in self.kwargs
                and self.kwargs["speculative_config"] is not None):
            return self.kwargs["speculative_config"].max_draft_len or 0

        return 0
