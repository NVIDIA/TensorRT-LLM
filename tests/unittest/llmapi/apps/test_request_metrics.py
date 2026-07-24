# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import json

import pytest

from tensorrt_llm.serve.perf_metrics import (
    CTX_CHUNK_METRICS_HEADER,
    RETURN_METRICS_HEADER,
    SERVER_TIMING_HEADER,
    SSE_METRICS_EVENT,
    START_END_TIME_HEADER,
    STEP_METRICS_HEADER,
    PerfMetricsJsonlWriter,
    PerfMetricsMiddleware,
    _jsonl_record,
    build_metrics_headers,
    build_metrics_record_from_headers,
    combine_disagg_metrics,
)
from tensorrt_llm.serve.scripts.time_breakdown import RequestDataParser


def _record(status="complete"):
    return {
        "schema_version": 1,
        "request_id": "42",
        "status": status,
        "phases": {
            "server": {
                "timing_metrics": {
                    "arrival_time": 1.0,
                    "first_scheduled_time": 1.01,
                    "first_token_time": 1.02,
                    "last_token_time": 1.05,
                    "kv_cache_transfer_start": None,
                    "kv_cache_transfer_end": None,
                },
                "time_breakdown_metrics": {
                    "step_metrics": [
                        {
                            "iter": 3,
                            "forward_start_time": 2.0,
                            "forward_end_time": 2.002,
                            "sample_start_time": 2.002,
                            "sample_end_time": 2.003,
                            "gpu_forward_time": 1.5,
                            "gpu_sample_time": 0.5,
                        }
                    ],
                    "ctx_chunk_metrics": [
                        {
                            "forward_start_time": 1.0,
                            "forward_end_time": 1.004,
                            "sample_start_time": 1.004,
                            "sample_end_time": 1.005,
                            "gpu_forward_time": 3.0,
                            "gpu_sample_time": 0.25,
                        }
                    ],
                },
            }
        },
    }


def test_metrics_headers_use_metric_list_syntax():
    headers = build_metrics_headers([_record()])

    assert "server_queue;dur=10.000000" in headers[SERVER_TIMING_HEADER]
    assert "server_ttft;dur=20.000000" in headers[SERVER_TIMING_HEADER]
    assert "server_e2e;dur=50.000000" in headers[SERVER_TIMING_HEADER]
    assert "server-start;ts=1.000000000" in headers[START_END_TIME_HEADER]
    assert "server-end;ts=1.050000000" in headers[START_END_TIME_HEADER]
    assert "server-step-3-forward;dur=2.000000" in headers[STEP_METRICS_HEADER]
    assert "server-step-3-gpu-sample;dur=0.500000" in headers[STEP_METRICS_HEADER]
    assert "server-ctx-chunk-0-forward;dur=4.000000" in headers[CTX_CHUNK_METRICS_HEADER]


def test_combine_disagg_metrics_is_request_local():
    ctx = {
        "request_id": "ctx-7",
        "ctx_request_id": 7,
        "metrics_headers": {CTX_CHUNK_METRICS_HEADER: "ctx-ctx-chunk-0-forward;dur=4.000000"},
        "phases": {"server": {"timing_metrics": {"arrival_time": 1.0}}},
    }
    gen = {
        "request_id": "gen-7",
        "ctx_request_id": 7,
        "metrics_headers": {STEP_METRICS_HEADER: "gen-step-3-forward;dur=2.000000"},
        "phases": {"server": {"timing_metrics": {"arrival_time": 2.0}}},
    }

    record = combine_disagg_metrics(
        "7",
        {"timing_metrics": {}},
        ctx,
        gen,
        disagg_request_id=7,
    )

    assert set(record["phases"]) == {"disagg", "ctx", "gen"}
    assert record["disagg_request_id"] == 7
    assert record["phases"]["ctx"]["request_id"] == "ctx-7"
    assert record["phases"]["gen"]["ctx_request_id"] == 7
    headers = build_metrics_headers([record])
    assert "ctx-ctx-chunk-0-forward;dur=4.000000" in headers[CTX_CHUNK_METRICS_HEADER]
    assert "gen-step-3-forward;dur=2.000000" in headers[STEP_METRICS_HEADER]


def test_time_breakdown_parser_accepts_header_derived_disagg_record():
    headers = build_metrics_headers([_record()])
    ctx = build_metrics_record_from_headers(headers, "ctx", request_id="42")
    gen = build_metrics_record_from_headers(headers, "gen", request_id="42")
    record = combine_disagg_metrics(
        "42",
        {
            "ctx_server": "ctx:8000",
            "gen_server": "gen:8000",
            "timing_metrics": {
                "server_arrival_time": 0.99,
                "ctx_dispatch_time": 1.0,
                "server_first_token_time": 1.03,
            },
        },
        ctx,
        gen,
        disagg_request_id=42,
    )

    parsed = RequestDataParser().parse_request(_jsonl_record(record), 0)
    combined_headers = build_metrics_headers([record])

    assert parsed["ctx_arrival_time"] == pytest.approx(1.0)
    assert parsed["ctx_first_scheduled_time"] == pytest.approx(1.01)
    assert parsed["ctx_first_token_time"] == pytest.approx(1.02)
    assert parsed["ctx_server_arrival_time"] == pytest.approx(1.0)
    assert parsed["ctx_server_first_token_time"] == pytest.approx(1.05)
    assert parsed["gen_arrival_time"] == pytest.approx(1.0)
    assert parsed["gen_first_scheduled_time"] == pytest.approx(1.01)
    assert parsed["gen_first_token_time"] == pytest.approx(1.02)
    assert parsed["gen_server_arrival_time"] == pytest.approx(1.0)
    assert parsed["gen_server_first_token_time"] == pytest.approx(1.05)
    assert parsed["disagg_server_arrival_time"] == pytest.approx(0.99)
    assert combined_headers[START_END_TIME_HEADER].count("ctx-start;") == 1
    assert combined_headers[SERVER_TIMING_HEADER].count("ctx_queue;") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("expose_headers", "request_metrics", "expected"),
    [
        (False, True, False),
        (True, False, False),
        (True, True, True),
    ],
)
async def test_middleware_controls_public_headers(expose_headers, request_metrics, expected):
    sent = []

    async def app(scope, receive, send):
        scope["state"]["perf_metrics_records"].extend([_record(), _record()])
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b"{}",
                "more_body": False,
            }
        )

    middleware = PerfMetricsMiddleware(app, expose_headers=expose_headers)
    headers = [(RETURN_METRICS_HEADER.encode(), b"1")] if request_metrics else []
    scope = {"type": "http", "headers": headers, "state": {}}

    async def capture(message):
        sent.append(message)

    await middleware(scope, None, capture)

    header_names = {key.lower() for key, _ in sent[0]["headers"]}
    assert (SERVER_TIMING_HEADER.lower().encode() in header_names) is expected
    assert (STEP_METRICS_HEADER.lower().encode() in header_names) is expected
    assert (CTX_CHUNK_METRICS_HEADER.lower().encode() in header_names) is expected
    if expected:
        headers = dict(sent[0]["headers"])
        assert headers[SERVER_TIMING_HEADER.encode()].count(b"server_queue;") == 2


@pytest.mark.asyncio
async def test_middleware_limits_non_streaming_metrics_headers():
    sent = []

    async def app(scope, receive, send):
        record = _record()
        breakdown = record["phases"]["server"]["time_breakdown_metrics"]
        breakdown["step_metrics"] *= 2000
        breakdown["ctx_chunk_metrics"] *= 2000
        scope["state"]["perf_metrics_records"].append(record)
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b"{}",
                "more_body": False,
            }
        )

    async def capture(message):
        sent.append(message)

    middleware = PerfMetricsMiddleware(app, expose_headers=True)
    await middleware(
        {
            "type": "http",
            "headers": [(RETURN_METRICS_HEADER.encode(), b"1")],
            "state": {},
        },
        None,
        capture,
    )

    header_names = {key.lower() for key, _ in sent[0]["headers"]}
    assert SERVER_TIMING_HEADER.lower().encode() in header_names
    assert START_END_TIME_HEADER.lower().encode() in header_names
    assert STEP_METRICS_HEADER.lower().encode() not in header_names
    assert CTX_CHUNK_METRICS_HEADER.lower().encode() not in header_names


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("expose_headers", "request_metrics", "expected"),
    [
        (False, True, False),
        (True, False, False),
        (True, True, True),
    ],
)
async def test_stream_metrics_follow_done(expose_headers, request_metrics, expected):
    sent = []

    async def app(scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b"data: [DONE]\n\n",
                "more_body": True,
            }
        )
        scope["state"]["perf_metrics_records"].append(_record())
        await send(
            {
                "type": "http.response.body",
                "body": b"",
                "more_body": False,
            }
        )

    async def capture(message):
        sent.append(message)

    middleware = PerfMetricsMiddleware(app, expose_headers=expose_headers)
    headers = [(RETURN_METRICS_HEADER.encode(), b"1")] if request_metrics else []
    await middleware({"type": "http", "headers": headers, "state": {}}, None, capture)

    assert sent[-2]["body"] == b"data: [DONE]\n\n"
    has_metrics_event = f"event: {SSE_METRICS_EVENT}".encode() in sent[-1]["body"]
    assert has_metrics_event is expected


@pytest.mark.asyncio
async def test_disconnect_after_done_is_ignored():
    async def app(scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b"data: [DONE]\n\n",
                "more_body": True,
            }
        )
        scope["state"]["perf_metrics_records"].append(_record())
        await send(
            {
                "type": "http.response.body",
                "body": b"",
                "more_body": False,
            }
        )

    async def disconnect(message):
        if message["type"] == "http.response.body" and not message.get("more_body", False):
            raise OSError("client disconnected")

    middleware = PerfMetricsMiddleware(app, expose_headers=True)
    await middleware(
        {
            "type": "http",
            "headers": [(RETURN_METRICS_HEADER.encode(), b"1")],
            "state": {},
        },
        None,
        disconnect,
    )


@pytest.mark.asyncio
async def test_file_middleware_intercepts_detail_headers(tmp_path):
    writer = PerfMetricsJsonlWriter(str(tmp_path), "test")
    await writer.start()

    async def app(scope, receive, send):
        records = [_record(), _record()]
        records[0]["disagg_request_id"] = 17
        records[1]["disagg_request_id"] = 18
        scope["state"]["perf_metrics_records"].extend(records)
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b"{}",
                "more_body": False,
            }
        )

    sent = []

    async def capture(message):
        sent.append(message)

    middleware = PerfMetricsMiddleware(app, expose_headers=False, writer=writer)
    await middleware({"type": "http", "headers": [], "state": {}}, None, capture)
    await writer.close()

    header_names = {key.lower() for key, _ in sent[0]["headers"]}
    assert STEP_METRICS_HEADER.lower().encode() not in header_names
    assert CTX_CHUNK_METRICS_HEADER.lower().encode() not in header_names

    output_file = next(tmp_path.glob("perf_metrics-test-*.jsonl"))
    saved = [json.loads(line) for line in output_file.read_text().splitlines()]
    assert [record["disagg_request_id"] for record in saved] == [17, 18]
    assert saved[0]["time_breakdown_metrics"]["step_metrics"]
    assert saved[0]["time_breakdown_metrics"]["ctx_chunk_metrics"]


@pytest.mark.asyncio
async def test_jsonl_writer_drops_only_malformed_record(tmp_path):
    writer = PerfMetricsJsonlWriter(str(tmp_path), "test")
    await writer.start()
    writer.submit({"phases": {}})
    writer.submit(_record())
    await writer.close()

    output_file = next(tmp_path.glob("perf_metrics-test-*.jsonl"))
    records = [json.loads(line) for line in output_file.read_text().splitlines()]
    assert writer.dropped_records == 1
    assert records[0]["request_id"] == 42
