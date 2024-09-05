from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, Union

import tensorrt_llm.bindings.executor as trtllm
from tensorrt_llm.bench.run.dataclasses import (BenchmarkStatistics,
                                                PercentileStats, RequestStats,
                                                ResponseRecord)
from tensorrt_llm.bindings import InferenceRequest


def get_executor_request(request: InferenceRequest,
                         pad_id: int,
                         eos_id: int,
                         streaming: bool = False) -> trtllm.Request:
    return trtllm.Request(
        input_token_ids=request.logits,
        max_tokens=request.output_tokens,
        stop_words=[],
        bad_words=[],
        streaming=streaming,
        output_config=trtllm.OutputConfig(exclude_input_from_output=True),
        pad_id=pad_id,
        end_id=eos_id,
    )


def get_settings_from_engine(
    engine_path: Path
) -> Tuple[Dict[str, Union[str, int]], Dict[str, Union[str, int]]]:
    config_path = engine_path / "config.json"
    runtime_config = {}

    with open(config_path, "r") as config_json:
        config = json.load(config_json)

    engine_world_map = config["pretrained_config"]["mapping"]
    engine_build_cfg = config["build_config"]
    engine_parallel_map = engine_build_cfg["auto_parallel_config"]

    world_config = {
        "pp_size": engine_world_map["pp_size"],
        "tp_size": engine_world_map["tp_size"],
        "world_size": engine_world_map["world_size"],
        "gpus_per_node": engine_parallel_map["gpus_per_node"],
    }

    executor_settings = {
        "max_batch_size": engine_build_cfg["max_batch_size"],
        "max_num_tokens": engine_build_cfg["max_num_tokens"],
    }

    runtime_config.update({
        "sw_version": config["version"],
        "engine_dir": str(engine_path.absolute()),
        "settings_config": executor_settings,
        "world_config": world_config,
    })

    return runtime_config, engine_build_cfg


class StatsKeeper:

    def __init__(self) -> None:
        self.requests: RequestStats = {}
        self.num_complete: int = 0

        self._unseen_cache = defaultdict(list)

    def register_request(
        self,
        request_id: int,
        timestamp: float,
        num_tokens: int,
    ) -> None:
        request = RequestStats(request_id=request_id, input_tokens=num_tokens)
        request.register_event(False, False, timestamp, 0)
        self.requests[request_id] = request

    def register_response(self, response: ResponseRecord) -> None:
        request_id = response.request_id

        if request_id not in self.requests:
            self._unseen_cache[request_id].append(response)
        else:
            self.requests[request_id].register_event(
                is_error=response.has_error,
                is_response=True,
                timestamp=response.timestamp,
                num_tokens=len(response.output_tokens))

            if response.is_final:
                self.num_complete += 1

    def generate_statistics_summary(self) -> None:
        total_output_tokens: int = 0
        total_input_tokens: int = 0
        num_requests = len(self.requests)
        total_request_latency: float = 0.0
        start_time = float("inf")
        end_time = -1

        request_latencies = []
        last_queue_time = 0.0
        queue_time_total = 0.0

        for entry in self.requests.values():
            entry.time_log.sort()

            queue_time_total += entry.time_log[0] - last_queue_time
            last_queue_time = entry.time_log[0]

            request_latencies.append(entry.request_latency)
            total_output_tokens += entry.num_tokens
            total_input_tokens += entry.input_tokens
            total_request_latency += entry.request_latency
            start_time = min(start_time, entry.time_log[0])
            end_time = max(end_time, entry.time_log[-1])

        stats = BenchmarkStatistics(
            num_requests=num_requests,
            total_latency_ns=end_time - start_time,
            total_output_tokens=total_output_tokens,
            total_input_tokens=total_input_tokens,
            request_percentiles=PercentileStats.from_iterable(
                request_latencies),
            issue_rate_ns=queue_time_total / num_requests)

        return stats
