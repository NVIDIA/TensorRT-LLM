# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from tensorrt_llm.llmapi.disagg_utils import ServerRole

COUNTER_METRICS = [
    ("total_requests", "Total number of requests"),
    ("error_requests", "Total number of error requests"),
    ("retry_requests", "Total number of retry requests"),
    ("completed_requests", "Total number of completed requests"),
]
# fmt: off
LONG_TIME_BUCKETS = [
    0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0,
     60.0, 120.0, 240.0, 480.0, 960.0, 1920.0,
]
SHORT_TIME_BUCKETS = [
    0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0,
    7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0,
]
# fmt: on
HISTOGRAM_METRICS = [
    (
        "first_token_latency_seconds",
        "Histogram of latency from first token to completion in seconds",
        SHORT_TIME_BUCKETS,
    ),
    (
        "complete_latency_seconds",
        "Histogram of latency from request arrival to last token in seconds",
        LONG_TIME_BUCKETS,
    ),
    (
        "per_token_latency_seconds",
        "Histogram of latency from request arrival to completion in seconds",
        SHORT_TIME_BUCKETS,
    ),
]

MetricsTypeLiteral = Literal["counter", "histogram"]


@dataclass
class MetricsDefinition:
    name: str
    description: str
    type: MetricsTypeLiteral
    buckets: Optional[List[float]] = None


CLIENT_METRICS_DEFINITIONS = [
    MetricsDefinition("total_requests", "Total number of requests", "counter"),
    MetricsDefinition("error_requests", "Total number of error requests", "counter"),
    MetricsDefinition("retry_requests", "Total number of retry requests", "counter"),
    MetricsDefinition("completed_requests", "Total number of completed requests", "counter"),
    MetricsDefinition(
        "first_token_latency_seconds",
        "Histogram of latency from first token to completion in seconds",
        "histogram",
        SHORT_TIME_BUCKETS,
    ),
    MetricsDefinition(
        "complete_latency_seconds",
        "Histogram of latency from request arrival to last token in seconds",
        "histogram",
        LONG_TIME_BUCKETS,
    ),
    MetricsDefinition(
        "per_token_latency_seconds",
        "Histogram of latency from request arrival to completion in seconds",
        "histogram",
        SHORT_TIME_BUCKETS,
    ),
]

ROLE_TO_CLIENT_TYPE = {
    ServerRole.CONTEXT: "ctx",
    ServerRole.GENERATION: "gen",
    ServerRole.MM_ENCODER: "mme",
}


def instance_metric(definition: MetricsDefinition, role: Optional[ServerRole] = None):
    # import lazily to avoid breaking `set_prometheus_multiproc_dir`
    from prometheus_client import Counter, Histogram

    name = (
        f"{ROLE_TO_CLIENT_TYPE[role]}_{definition.name}"
        if role in ROLE_TO_CLIENT_TYPE
        else definition.name
    )
    if definition.type == "counter":
        return Counter(name, definition.description)
    elif definition.type == "histogram":
        return Histogram(name, definition.description, buckets=definition.buckets)
    else:
        raise ValueError(f"Invalid metric type: {definition.type}")


class ClientMetricsCollector:
    def __init__(self, role: ServerRole):
        self._role = role
        self._metrics = {
            definition.name: instance_metric(definition, role)
            for definition in CLIENT_METRICS_DEFINITIONS
        }

    def __getattr__(
        self, key: str
    ):  # no return type hint to not import prometheus_client at module level
        return self._metrics[key]


SERVER_METRICS_DEFINITIONS = [
    MetricsDefinition("total_requests", "Total number of requests", "counter"),
    MetricsDefinition("stream_requests", "Total number of stream requests", "counter"),
    MetricsDefinition("nonstream_requests", "Total number of non-stream requests", "counter"),
    MetricsDefinition("validation_exceptions", "Total number of validation exceptions", "counter"),
    MetricsDefinition("http_exceptions", "Total number of HTTP exceptions", "counter"),
    MetricsDefinition("internal_errors", "Total number of internal errors", "counter"),
    MetricsDefinition("total_responses", "Total number of responses", "counter"),
    MetricsDefinition(
        "queue_latency_seconds",
        "Histogram of latency from request arrival to being processed in seconds",
        "histogram",
        SHORT_TIME_BUCKETS,
    ),
]


class DisaggPerfMetricsCollector:
    def __init__(self, max_requests: int):
        self._max_requests = max_requests
        self._request_meteics = deque(maxlen=max_requests)
        self._server_metrics = defaultdict(dict)
        self._lock = asyncio.Lock()
        self._clients = []
        self._metrics = {
            definition.name: instance_metric(definition)
            for definition in SERVER_METRICS_DEFINITIONS
        }

    def add_client(self, client):
        self._clients.append(client)

    def __getattr__(self, key: str):
        return self._metrics[key]

    async def add_per_request_metrics(
        self,
        ctx_server: str,
        gen_server: str,
        ctx_request_id: int,
        server_arrival_time: float,
        server_first_token_time: float,
    ):
        async with self._lock:
            self._request_meteics.append(
                (
                    ctx_server,
                    gen_server,
                    ctx_request_id,
                    server_arrival_time,
                    server_first_token_time,
                )
            )

    async def get_perf_metrics(self) -> List[Dict[str, Any]]:
        perf_metrics = {}
        for client in self._clients:
            metrics_dict = await client.collect_metrics()
            perf_metrics.update(metrics_dict)

        return_metrics = []
        async with self._lock:
            for server, metrics_data in perf_metrics.items():
                server_metrics = self._server_metrics[server]
                # avoid metrics map inflation by limiting the number of requests to add
                available_req_num = min(
                    max(0, self._max_requests - len(server_metrics)), len(metrics_data)
                )
                req_metrics_map = {
                    req_metrics["ctx_request_id"]: req_metrics
                    for req_metrics in metrics_data[:available_req_num]
                    if "ctx_request_id" in req_metrics
                }
                server_metrics.update(req_metrics_map)

            remain_keys = []
            for (
                ctx_server,
                gen_server,
                ctx_request_id,
                server_arrival_time,
                server_first_token_time,
            ) in self._request_meteics:
                gen_perf_metrics = self._server_metrics[gen_server].pop(ctx_request_id, None)
                if gen_perf_metrics is None:
                    # generation not finished
                    remain_keys.append(
                        (
                            ctx_server,
                            gen_server,
                            ctx_request_id,
                            server_arrival_time,
                            server_first_token_time,
                        )
                    )
                    continue
                ctx_perf_metrics = self._server_metrics[ctx_server].pop(ctx_request_id, None)
                # TODO: strip the keys for less repeating and use table style response
                return_metrics.append(
                    {
                        "ctx_server": ctx_server,
                        "gen_server": gen_server,
                        "disagg_server_arrival_time": server_arrival_time,
                        "disagg_server_first_token_time": server_first_token_time,
                        "ctx_perf_metrics": ctx_perf_metrics,
                        "gen_perf_metrics": gen_perf_metrics,
                    }
                )
            self._request_meteics = deque(remain_keys, maxlen=self._max_requests)
        return return_metrics
