# Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

"""Prometheus and per-request serving metrics.

Non-streaming request opt-in and response example::

    POST /v1/completions HTTP/1.1
    X-TRTLLM-return-metrics: 1

    HTTP/1.1 200 OK
    Content-Type: application/json
    Server-Timing: server_queue;dur=1.250000, server_ttft;dur=8.500000, server_e2e;dur=24.000000
    X-TRTLLM-Start-End-Time: server-start;ts=12345.123456, server-end;ts=12345.147456
    X-TRTLLM-Step-Metrics: server-step-0-forward;dur=2.100000, server-step-0-sample;dur=0.400000
    X-TRTLLM-Ctx-Chunk-Metrics: server-ctx-chunk-0-forward;dur=4.200000

Streaming responses carry the same fields in a named SSE event after ``[DONE]``::

    data: [DONE]

    event: trtllm.perf_metrics
    data: {
        "Server-Timing": "server_queue;dur=1.250000, server_ttft;dur=8.500000",
        "X-TRTLLM-Start-End-Time": "server-start;ts=12345.123456, server-end;ts=12345.147456",
        "X-TRTLLM-Step-Metrics": "server-step-0-forward;dur=2.100000",
        "X-TRTLLM-Ctx-Chunk-Metrics": "server-ctx-chunk-0-forward;dur=4.200000",
    }
"""

import asyncio
import json
import math
import os
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.logger import logger
from tensorrt_llm.serve._perf_metrics_schema import (
    DisaggPerfMetricsRecord,
    PerfMetrics,
    PerfMetricsRecord,
    WorkerPerfMetrics,
    WorkerPerfMetricsRecord,
)

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
    MetricsDefinition(name, description, "counter") for name, description in COUNTER_METRICS
] + [
    MetricsDefinition(name, description, "histogram", buckets)
    for name, description, buckets in HISTOGRAM_METRICS
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


SERVER_COUNTER_METRICS = (
    ("total_requests", "Total number of requests"),
    ("stream_requests", "Total number of stream requests"),
    ("nonstream_requests", "Total number of non-stream requests"),
    ("validation_exceptions", "Total number of validation exceptions"),
    ("http_exceptions", "Total number of HTTP exceptions"),
    ("internal_errors", "Total number of internal errors"),
    ("total_responses", "Total number of responses"),
)
SERVER_METRICS_DEFINITIONS = [
    MetricsDefinition(name, description, "counter") for name, description in SERVER_COUNTER_METRICS
] + [
    MetricsDefinition(
        "queue_latency_seconds",
        "Histogram of latency from request arrival to being processed in seconds",
        "histogram",
        SHORT_TIME_BUCKETS,
    )
]


class DisaggPerfMetricsCollector:
    """Prometheus metrics owned by one disaggregated HTTP server process."""

    def __init__(self, max_requests: int = 0):
        # Kept for compatibility; per-request retention now belongs to JSONL.
        del max_requests
        self._metrics = {
            definition.name: instance_metric(definition)
            for definition in SERVER_METRICS_DEFINITIONS
        }

    def __getattr__(self, key: str):
        return self._metrics[key]


SERVER_TIMING_HEADER = "Server-Timing"
START_END_TIME_HEADER = "X-TRTLLM-Start-End-Time"
STEP_METRICS_HEADER = "X-TRTLLM-Step-Metrics"
CTX_CHUNK_METRICS_HEADER = "X-TRTLLM-Ctx-Chunk-Metrics"
SSE_METRICS_EVENT = "trtllm.perf_metrics"
RETURN_METRICS_HEADER = "X-TRTLLM-return-metrics"
_RETURN_METRICS_HEADER_BYTES = RETURN_METRICS_HEADER.lower().encode()

_SCHEMA_VERSION = 1
_METRICS_PAYLOAD_BUDGET_BYTES = 80 * 1024
_WRITER_QUEUE_SIZE = 1024
_WRITER_BATCH_SIZE = 64
_WRITER_SHUTDOWN_TIMEOUT_SECONDS = 5


_TIMING_FIELDS = (
    "arrival_time",
    "first_scheduled_time",
    "first_token_time",
    "last_token_time",
    "kv_cache_transfer_start",
    "kv_cache_transfer_end",
)
_KV_FIELDS = (
    "num_total_allocated_blocks",
    "num_new_allocated_blocks",
    "num_reused_blocks",
    "num_missed_blocks",
    "kv_cache_hit_rate",
)
_SPEC_FIELDS = ("acceptance_rate", "total_accepted_draft_tokens", "total_draft_tokens")


def _as_seconds(value: Any, offset: float = 0) -> Optional[float]:
    try:
        seconds = float(value.total_seconds())
    except (AttributeError, TypeError, ValueError):
        return None
    return seconds + offset if seconds > 0 else None


def build_request_metrics_record(
    result: Any,
    raw_request: Any = None,
    phase: str = "server",
    steady_clock_offset: float = 0,
) -> Optional[Dict[str, Any]]:
    """Convert a completed RequestOutput metrics snapshot to JSON-safe data."""
    if not result or not getattr(result, "outputs", None):
        return None
    output = result.outputs[0]
    metrics = getattr(output, "request_perf_metrics", None)
    if metrics is None:
        return None

    timing = metrics.timing_metrics
    timing_metrics = {
        name: _as_seconds(getattr(timing, name), steady_clock_offset) for name in _TIMING_FIELDS
    }
    timing_metrics["kv_cache_size"] = timing.kv_cache_size
    if raw_request is not None:
        for name in ("server_arrival_time", "server_first_token_time"):
            value = getattr(raw_request.state, name, None)
            timing_metrics[name] = value + steady_clock_offset if value is not None else None

    phase_record: Dict[str, Any] = {
        "first_iter": metrics.first_iter,
        "last_iter": metrics.last_iter,
        "timing_metrics": timing_metrics,
        "kv_cache_metrics": {name: getattr(metrics.kv_cache_metrics, name) for name in _KV_FIELDS},
    }
    speculative = metrics.speculative_decoding
    if speculative.total_draft_tokens > 0:
        phase_record["speculative_decoding"] = {
            name: getattr(speculative, name) for name in _SPEC_FIELDS
        }
    if getattr(result, "time_breakdown_metrics", None) is not None:
        phase_record["time_breakdown_metrics"] = result.time_breakdown_metrics

    record: Dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "request_id": str(getattr(result, "request_id", "")),
        "status": "complete",
        "phases": {phase: phase_record},
    }
    disagg = getattr(output, "disaggregated_params", None)
    if disagg:
        for name in ("ctx_request_id", "disagg_request_id"):
            value = getattr(disagg, name, None)
            if value is not None:
                record[name] = value
    return record


def _elapsed_ms(values: Dict[str, Any], start: str, end: Optional[str] = None) -> Optional[float]:
    try:
        value = float(values[start])
        if end is not None:
            value = (float(values[end]) - value) * 1000
    except (KeyError, TypeError, ValueError):
        return None
    return value if value >= 0 and math.isfinite(value) else None


def build_metrics_headers(record: Dict[str, Any]) -> Dict[str, str]:
    """Format one completed record as Server-Timing-style metric lists."""
    values = {
        SERVER_TIMING_HEADER: [],
        START_END_TIME_HEADER: [],
        STEP_METRICS_HEADER: [],
        CTX_CHUNK_METRICS_HEADER: [],
    }
    for phase, phase_record in record.get("phases", {}).items():
        timing = phase_record.get("timing_metrics", {})
        for name, field in (
            ("start", "arrival_time"),
            ("end", "last_token_time"),
        ):
            timestamp = timing.get(field)
            if timestamp is not None:
                values[START_END_TIME_HEADER].append(f"{phase}-{name};ts={float(timestamp):.9f}")
        timing_ranges = (
            (
                ("queue", "server_arrival_time", "ctx_dispatch_time"),
                ("ttft", "server_arrival_time", "server_first_token_time"),
            )
            if phase == "disagg"
            else (
                ("queue", "arrival_time", "first_scheduled_time"),
                ("ttft", "arrival_time", "first_token_time"),
                ("e2e", "arrival_time", "last_token_time"),
                ("kv_transfer", "kv_cache_transfer_start", "kv_cache_transfer_end"),
            )
        )
        for name, start, end in timing_ranges:
            duration = _elapsed_ms(timing, start, end)
            if duration is not None:
                values[SERVER_TIMING_HEADER].append(f"{phase}_{name};dur={duration:.6f}")

        breakdown = phase_record.get("time_breakdown_metrics") or {}
        for header, key, label in (
            (STEP_METRICS_HEADER, "step_metrics", "step"),
            (CTX_CHUNK_METRICS_HEADER, "ctx_chunk_metrics", "ctx-chunk"),
        ):
            for index, metrics in enumerate(breakdown.get(key, [])):
                item = metrics.get("iter", index) if key == "step_metrics" else index
                durations = (
                    ("forward", _elapsed_ms(metrics, "forward_start_time", "forward_end_time")),
                    ("sample", _elapsed_ms(metrics, "sample_start_time", "sample_end_time")),
                    ("gpu-forward", _elapsed_ms(metrics, "gpu_forward_time")),
                    ("gpu-sample", _elapsed_ms(metrics, "gpu_sample_time")),
                )
                values[header].extend(
                    f"{phase}-{label}-{item}-{name};dur={duration:.6f}"
                    for name, duration in durations
                    if duration is not None
                )

        inherited = phase_record.get("metrics_headers") or {}
        for header in values:
            if inherited.get(header):
                values[header].append(inherited[header])

    return {header: ", ".join(items) for header, items in values.items() if items}


def build_metrics_record_from_headers(
    headers: Any,
    phase: str,
    request_id: str = "",
) -> Optional[Dict[str, Any]]:
    """Build a request-local phase from standard metrics fields."""
    metrics_headers = {}
    for header_name in (
        SERVER_TIMING_HEADER,
        START_END_TIME_HEADER,
        STEP_METRICS_HEADER,
        CTX_CHUNK_METRICS_HEADER,
    ):
        value = headers.get(header_name)
        if value:
            metrics_headers[header_name] = value.replace("server_", f"{phase}_").replace(
                "server-", f"{phase}-"
            )
    if not metrics_headers:
        return None
    timing_metrics = {}
    fields = {
        f"{phase}-start": "arrival_time",
        f"{phase}-end": "last_token_time",
    }
    for item in metrics_headers.get(START_END_TIME_HEADER, "").split(","):
        name, separator, timestamp = item.strip().partition(";ts=")
        if separator and name in fields:
            try:
                timing_metrics[fields[name]] = float(timestamp)
            except ValueError:
                logger.warning("Ignoring invalid %s timestamp: %s", name, timestamp)

    durations = {}
    prefix = f"{phase}_"
    for item in metrics_headers.get(SERVER_TIMING_HEADER, "").split(","):
        name, separator, duration = item.strip().partition(";dur=")
        if separator and name.startswith(prefix):
            try:
                durations[name[len(prefix) :]] = float(duration) / 1000
            except ValueError:
                logger.warning("Ignoring invalid %s duration: %s", name, duration)

    arrival_time = timing_metrics.get("arrival_time")
    if arrival_time is not None:
        for name, field in (
            ("queue", "first_scheduled_time"),
            ("ttft", "first_token_time"),
            ("e2e", "last_token_time"),
        ):
            if name in durations:
                timing_metrics.setdefault(field, arrival_time + durations[name])

    phase_record = {"metrics_headers": metrics_headers}
    if timing_metrics:
        phase_record["timing_metrics"] = timing_metrics
    return {
        "schema_version": _SCHEMA_VERSION,
        "request_id": request_id,
        "status": "complete",
        "metrics_headers": metrics_headers,
        "phases": {phase: phase_record},
    }


def _limit_metrics_headers(headers: Dict[str, str]) -> Dict[str, str]:
    def size(values: Dict[str, str]) -> int:
        return sum(len(name.encode()) + len(value.encode()) + 4 for name, value in values.items())

    if size(headers) <= _METRICS_PAYLOAD_BUDGET_BYTES:
        return headers
    logger.warning(
        "Performance metrics payload exceeds %d bytes; omitting step and context-chunk metrics",
        _METRICS_PAYLOAD_BUDGET_BYTES,
    )
    return {
        name: value
        for name, value in headers.items()
        if name in (SERVER_TIMING_HEADER, START_END_TIME_HEADER)
    }


def combine_disagg_metrics(
    request_id: str,
    disagg_phase: Dict[str, Any],
    ctx_record: Optional[Dict[str, Any]],
    gen_record: Optional[Dict[str, Any]],
    disagg_request_id: Optional[int] = None,
) -> Dict[str, Any]:
    phases: Dict[str, Any] = {"disagg": disagg_phase}

    def add_worker_phase(name: str, record: Dict[str, Any]) -> None:
        phase_record = dict(next(iter(record.get("phases", {}).values()), {}))
        phase_record["request_id"] = record.get("request_id")
        if record.get("ctx_request_id") is not None:
            phase_record["ctx_request_id"] = record["ctx_request_id"]
        if record.get("metrics_headers"):
            phase_record["metrics_headers"] = record["metrics_headers"]
        phases[name] = phase_record

    if ctx_record:
        add_worker_phase("ctx", ctx_record)
    if gen_record:
        add_worker_phase("gen", gen_record)
    combined = {
        "schema_version": _SCHEMA_VERSION,
        "request_id": request_id,
        "status": "complete",
        "phases": phases,
    }
    if disagg_request_id is not None:
        combined["disagg_request_id"] = disagg_request_id
    return combined


def _jsonl_perf_metrics(phase_record: Dict[str, Any]) -> PerfMetrics:
    perf_metrics: PerfMetrics = {
        "timing_metrics": dict(phase_record.get("timing_metrics", {})),
    }
    if "first_iter" in phase_record:
        perf_metrics["first_iter"] = phase_record["first_iter"]
    if "last_iter" in phase_record:
        perf_metrics["last_iter"] = phase_record["last_iter"]
    if "kv_cache_metrics" in phase_record:
        perf_metrics["kv_cache_metrics"] = phase_record["kv_cache_metrics"]
    if "speculative_decoding" in phase_record:
        perf_metrics["speculative_decoding"] = phase_record["speculative_decoding"]

    timing_metrics = dict(perf_metrics.get("timing_metrics", {}))
    if not timing_metrics.get("kv_cache_size"):
        for name in ("kv_cache_size", "kv_cache_transfer_start", "kv_cache_transfer_end"):
            timing_metrics.pop(name, None)
    perf_metrics["timing_metrics"] = timing_metrics

    kv_cache_metrics = dict(perf_metrics.get("kv_cache_metrics", {}))
    kv_cache_metrics.pop("kv_cache_hit_rate", None)
    if kv_cache_metrics:
        perf_metrics["kv_cache_metrics"] = kv_cache_metrics
    return perf_metrics


def _jsonl_worker_metrics(
    record: Dict[str, Any], phase_record: Dict[str, Any]
) -> WorkerPerfMetrics:
    request_id = record["request_id"]
    try:
        request_id = int(request_id)
    except (TypeError, ValueError):
        pass
    worker_metrics: WorkerPerfMetrics = {
        "request_id": request_id,
        "perf_metrics": _jsonl_perf_metrics(phase_record),
    }
    if record.get("ctx_request_id") is not None:
        worker_metrics["ctx_request_id"] = record["ctx_request_id"]
    if phase_record.get("time_breakdown_metrics") is not None:
        worker_metrics["time_breakdown_metrics"] = phase_record["time_breakdown_metrics"]
    return worker_metrics


def _jsonl_record(record: Dict[str, Any]) -> PerfMetricsRecord:
    phases = record.get("phases", {})
    if "disagg" not in phases:
        worker_metrics = _jsonl_worker_metrics(record, phases["server"])
        jsonl_record: WorkerPerfMetricsRecord = {
            **worker_metrics,
            "status": record.get("status", "complete"),
        }
        if record.get("disagg_request_id") is not None:
            jsonl_record["disagg_request_id"] = record["disagg_request_id"]
        return jsonl_record

    disagg_phase = phases["disagg"]
    disagg_timing = disagg_phase["timing_metrics"]
    disagg_record: DisaggPerfMetricsRecord = {
        "ctx_server": disagg_phase["ctx_server"],
        "gen_server": disagg_phase["gen_server"],
        "disagg_server_arrival_time": disagg_timing["server_arrival_time"],
        "disagg_ctx_dispatch_time": disagg_timing["ctx_dispatch_time"],
        "disagg_server_first_token_time": disagg_timing["server_first_token_time"],
        "status": record.get("status", "complete"),
    }
    for phase, field in (
        ("ctx", "ctx_perf_metrics"),
        ("gen", "gen_perf_metrics"),
    ):
        phase_record = phases.get(phase)
        if phase_record:
            worker_record = {
                "request_id": phase_record.get("request_id", record["request_id"]),
                "ctx_request_id": phase_record.get("ctx_request_id"),
            }
            worker_metrics = _jsonl_worker_metrics(worker_record, phase_record)
            if field == "ctx_perf_metrics":
                disagg_record["ctx_perf_metrics"] = worker_metrics
            else:
                disagg_record["gen_perf_metrics"] = worker_metrics
    if record.get("disagg_request_id") is not None:
        disagg_record["disagg_request_id"] = record["disagg_request_id"]
    return disagg_record


class PerfMetricsJsonlWriter:
    """Best-effort bounded JSONL writer shared by both serving apps."""

    def __init__(self, output_dir: Optional[str], server_kind: str):
        self._output_dir = Path(output_dir) if output_dir else None
        self._server_kind = server_kind
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=_WRITER_QUEUE_SIZE)
        self._task: Optional[asyncio.Task] = None
        self._path: Optional[Path] = None
        self.dropped_records = 0
        self._write_error_count = 0

    async def start(self) -> None:
        if self._output_dir is None or self._task is not None:
            return
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            filename = (
                f"perf_metrics-{self._server_kind}-{socket.gethostname()}-"
                f"{os.getpid()}-{timestamp}.jsonl"
            )
            self._path = self._output_dir / filename
            self._task = asyncio.create_task(self._run())
        except OSError as error:
            logger.error("Disabling performance metrics JSONL output: %s", error)
            self._output_dir = None

    def submit(self, record: Dict[str, Any]) -> None:
        if self._task is None:
            return
        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull:
            self.dropped_records += 1
            if self.dropped_records == 1 or self.dropped_records % 1000 == 0:
                logger.warning("Dropped %d performance metrics records", self.dropped_records)

    async def close(self) -> None:
        if self._task is None:
            return
        task = self._task
        try:
            await asyncio.wait_for(
                self._queue.put(None),
                timeout=_WRITER_SHUTDOWN_TIMEOUT_SECONDS,
            )
            await asyncio.wait_for(task, timeout=_WRITER_SHUTDOWN_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            logger.warning(
                "Timed out flushing performance metrics JSONL; dropping remaining records"
            )
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        finally:
            self._task = None

    async def _run(self) -> None:
        stop = False
        while not stop:
            item = await self._queue.get()
            if item is None:
                return
            records = [item]
            while len(records) < _WRITER_BATCH_SIZE:
                try:
                    item = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is None:
                    stop = True
                    break
                records.append(item)
            serialized = []
            for record in records:
                try:
                    item = _jsonl_record(record)
                    serialized.append(
                        json.dumps(item, separators=(",", ":"), allow_nan=False) + "\n"
                    )
                except (KeyError, TypeError, ValueError) as error:
                    self.dropped_records += 1
                    if self.dropped_records == 1 or self.dropped_records % 1000 == 0:
                        logger.warning("Dropped malformed performance metrics record: %s", error)
            if not serialized:
                continue
            try:
                data = "".join(serialized)
                await asyncio.to_thread(self._write, data)
            except OSError as error:
                self.dropped_records += len(serialized)
                self._write_error_count += 1
                if self._write_error_count == 1:
                    logger.warning("Failed to write performance metrics JSONL: %s", error)

    def _write(self, data: str) -> None:
        if self._path is not None:
            with self._path.open("a", encoding="utf-8") as output:
                output.write(data)


def build_metrics_sse_event(headers: Dict[str, str]) -> bytes:
    headers = _limit_metrics_headers(headers)
    if not headers:
        return b""
    payload = json.dumps(headers, separators=(",", ":"))
    return (f"event: {SSE_METRICS_EVENT}\ndata: {payload}\n\n").encode()


class PerfMetricsMiddleware:
    """Expose request metrics and optionally persist completed records."""

    def __init__(
        self, app: Any, expose_headers: bool, writer: Optional[PerfMetricsJsonlWriter] = None
    ):
        self._app = app
        self._expose_headers = expose_headers
        self._writer = writer

    async def __call__(self, scope: Dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        is_stream = False
        metrics_headers = None
        return_metrics = self._expose_headers and any(
            name.lower() == _RETURN_METRICS_HEADER_BYTES and value.strip() == b"1"
            for name, value in scope.get("headers", [])
        )

        async def send_metrics(message: Dict[str, Any]) -> None:
            nonlocal is_stream, metrics_headers
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                is_stream = any(
                    key.lower() == b"content-type" and b"text/event-stream" in value.lower()
                    for key, value in headers
                )
                record = scope.get("state", {}).get("perf_metrics_record")
                if record and return_metrics and not is_stream:
                    metrics_headers = build_metrics_headers(record)
                    public_headers = _limit_metrics_headers(metrics_headers)
                    headers.extend(
                        (name.encode(), value.encode()) for name, value in public_headers.items()
                    )
                message["headers"] = headers

            elif message["type"] == "http.response.body" and not message.get("more_body", False):
                record = scope.get("state", {}).get("perf_metrics_record")
                if record:
                    if metrics_headers is None:
                        metrics_headers = build_metrics_headers(record)
                    if self._writer is not None:
                        details = {
                            name: value
                            for name, value in metrics_headers.items()
                            if name
                            in (
                                START_END_TIME_HEADER,
                                STEP_METRICS_HEADER,
                                CTX_CHUNK_METRICS_HEADER,
                            )
                        }
                        self._writer.submit(
                            {**record, "streaming": is_stream, "metrics_headers": details}
                        )
                    if return_metrics and is_stream:
                        message["body"] = message.get("body", b"") + build_metrics_sse_event(
                            metrics_headers
                        )
                        try:
                            await send(message)
                        except OSError:
                            pass
                        return
            await send(message)

        await self._app(scope, receive, send_metrics)
