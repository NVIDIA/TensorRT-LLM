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
from typing import Optional

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
    _jsonl_perf_metrics,
    _jsonl_record,
    build_metrics_headers,
    build_metrics_record_from_headers,
    combine_disagg_metrics,
)
from tensorrt_llm.serve.scripts.time_breakdown import RequestDataParser, RequestTimeBreakdown


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
    """Backward compatibility: a peer that omits the srv-/kv- timestamps.

    ``_record()`` carries neither ``server_arrival_time`` nor
    ``server_first_token_time``, and its KV timestamps are ``None`` -- i.e. what
    a worker built before those tokens were added to the header transport, or a
    non-disaggregated request, sends. The parser must still produce a record,
    falling back to ``arrival_time`` / ``last_token_time``. The asserted values
    below are therefore *fallbacks*, not measurements; the full-fidelity
    contract is asserted by
    ``test_header_transport_preserves_every_lifecycle_timestamp`` and
    ``test_header_derived_record_yields_twelve_non_zero_spans``.
    """
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


# One realistic disaggregated request whose 12 lifecycle spans are all distinct
# and non-zero: span k lasts exactly k ms. Distinct widths matter -- with equal
# widths a span attributed to the wrong pair of timestamps still reads correct.
_T0 = 1000.0
_LIFECYCLE = {
    "disagg_arrival": _T0 + 0.000,  # span 1 start
    "ctx_server_arrival": _T0 + 0.001,  # span 1 end  / span 2 start   1 ms
    "ctx_arrival": _T0 + 0.003,  # span 2 end  / span 3 start   2 ms
    "ctx_scheduled": _T0 + 0.006,  # span 3 end  / span 4 start   3 ms
    "ctx_first_token": _T0 + 0.010,  # span 4 end  / span 5 start   4 ms
    "ctx_server_first_token": _T0 + 0.015,  # span 5 end  / span 6 start   5 ms
    "gen_server_arrival": _T0 + 0.021,  # span 6 end  / span 7 start   6 ms
    "gen_arrival": _T0 + 0.028,  # span 7 end  / span 8 start   7 ms
    "kv_start": _T0 + 0.036,  # span 8 end  / span 9 start   8 ms
    "kv_end": _T0 + 0.045,  # span 9 end  / span 10 start  9 ms
    "gen_scheduled": _T0 + 0.055,  # span 10 end / span 11 start 10 ms
    "gen_server_first_token": _T0 + 0.066,  # span 11 end / span 12 start 11 ms
    "disagg_first_token": _T0 + 0.078,  # span 12 end                 12 ms
}

_EXPECTED_SPAN_MS = {
    "disagg_preprocessing": 1.0,
    "ctx_preprocessing": 2.0,
    "ctx_queue": 3.0,
    "ctx_processing": 4.0,
    "ctx_postprocessing": 5.0,
    "disagg_relay": 6.0,
    "gen_preprocessing": 7.0,
    "gen_queue_wait": 8.0,
    "gen_kv_transfer": 9.0,
    "gen_post_transfer": 10.0,
    "gen_postprocessing": 11.0,
    "disagg_postprocessing": 12.0,
}


def _ctx_worker_record():
    return {
        "request_id": "ctx-1",
        "phases": {
            "server": {
                "timing_metrics": {
                    "server_arrival_time": _LIFECYCLE["ctx_server_arrival"],
                    "arrival_time": _LIFECYCLE["ctx_arrival"],
                    "first_scheduled_time": _LIFECYCLE["ctx_scheduled"],
                    "first_token_time": _LIFECYCLE["ctx_first_token"],
                    "last_token_time": _LIFECYCLE["ctx_first_token"],
                    "server_first_token_time": _LIFECYCLE["ctx_server_first_token"],
                    "kv_cache_size": 2048,
                }
            }
        },
    }


def _gen_worker_record():
    return {
        "request_id": "gen-1",
        "phases": {
            "server": {
                "timing_metrics": {
                    "server_arrival_time": _LIFECYCLE["gen_server_arrival"],
                    "arrival_time": _LIFECYCLE["gen_arrival"],
                    "kv_cache_transfer_start": _LIFECYCLE["kv_start"],
                    "kv_cache_transfer_end": _LIFECYCLE["kv_end"],
                    "first_scheduled_time": _LIFECYCLE["gen_scheduled"],
                    "first_token_time": _LIFECYCLE["gen_server_first_token"],
                    "last_token_time": _T0 + 0.200,
                    "server_first_token_time": _LIFECYCLE["gen_server_first_token"],
                    "kv_cache_size": 2048,
                }
            }
        },
    }


def _combined_disagg_record():
    """Reproduce the worker -> header -> disagg-server -> JSONL chain."""
    ctx = build_metrics_record_from_headers(
        build_metrics_headers([_ctx_worker_record()]), "ctx", request_id="ctx-1"
    )
    gen = build_metrics_record_from_headers(
        build_metrics_headers([_gen_worker_record()]), "gen", request_id="gen-1"
    )
    return combine_disagg_metrics(
        "req-1",
        {
            "ctx_server": "http://ctx0:8001",
            "gen_server": "http://gen0:8002",
            "timing_metrics": {
                "server_arrival_time": _LIFECYCLE["disagg_arrival"],
                "ctx_dispatch_time": _T0 + 0.0005,
                "server_first_token_time": _LIFECYCLE["disagg_first_token"],
            },
        },
        ctx,
        gen,
        disagg_request_id=1,
    )


@pytest.mark.parametrize("phase", ["ctx", "gen"])
def test_header_transport_preserves_every_lifecycle_timestamp(phase):
    """All six absolute timestamps must survive the Server-Timing round trip.

    Only ``arrival_time`` and ``last_token_time`` used to be forwarded, so a
    disagg server reconstructed ``server_arrival_time``,
    ``server_first_token_time`` and the two KV-transfer timestamps from
    fallbacks. That is not detectable downstream: the affected spans come back as
    0 or as a plausible-looking wrong number, never as an error.
    """
    worker = _ctx_worker_record() if phase == "ctx" else _gen_worker_record()
    source = worker["phases"]["server"]["timing_metrics"]

    headers = build_metrics_headers([worker])
    derived = build_metrics_record_from_headers(headers, phase, request_id="x")
    timing = derived["phases"][phase]["timing_metrics"]

    for field in (
        "arrival_time",
        "last_token_time",
        "server_arrival_time",
        "server_first_token_time",
        "kv_cache_transfer_start",
        "kv_cache_transfer_end",
    ):
        if source.get(field) is None:
            continue
        assert timing[field] == pytest.approx(source[field]), field

    # The phase rewrite is an unqualified str.replace() of "server-"/"server_",
    # so a token name containing a second occurrence would be substituted twice.
    assert f"{phase}-{phase}-" not in headers[START_END_TIME_HEADER]
    assert "server-" not in derived["metrics_headers"][START_END_TIME_HEADER]


def test_jsonl_record_keeps_header_derived_kv_transfer_timestamps():
    """kv_cache_size is worker-local; it must not gate the KV timestamps.

    ``kv_cache_size`` is set only when a worker builds its own record, so it
    never reaches a header-derived one. Stripping the KV timestamps whenever it
    is absent zeroed the KV-transfer span for every disaggregated request.
    """
    gen = build_metrics_record_from_headers(
        build_metrics_headers([_gen_worker_record()]), "gen", request_id="gen-1"
    )
    timing = _jsonl_perf_metrics(gen["phases"]["gen"])["timing_metrics"]

    assert "kv_cache_size" not in timing
    assert timing["kv_cache_transfer_start"] == pytest.approx(_LIFECYCLE["kv_start"])
    assert timing["kv_cache_transfer_end"] == pytest.approx(_LIFECYCLE["kv_end"])


@pytest.mark.parametrize("absent", [None, 0, 0.0])
def test_jsonl_record_still_strips_absent_kv_transfer_timestamps(absent: Optional[float]) -> None:
    """A request that never transferred KV must not gain zero-width KV fields.

    ``0`` is not hypothetical: the aggregated path reads these off a default-
    initialised C++ duration (``timing_metrics.kv_cache_transfer_start
    .total_seconds()``), so a non-transferring request arrives as ``0.0`` rather
    than ``None``. Both encodings must be stripped, or a consumer testing for
    presence rather than truthiness reads a zero-width transfer as a measurement.
    """
    record = _record()
    phase = record["phases"]["server"]
    phase["timing_metrics"]["kv_cache_transfer_start"] = absent
    phase["timing_metrics"]["kv_cache_transfer_end"] = absent
    timing = _jsonl_perf_metrics(phase)["timing_metrics"]

    assert "kv_cache_transfer_start" not in timing
    assert "kv_cache_transfer_end" not in timing


def test_header_derived_record_yields_twelve_non_zero_spans(tmp_path):
    """The acceptance test for the transport: 12/12 spans, each exactly right.

    This is what the ``time_breakdown`` perf-sanity modifier uploads. A span that
    computes to 0 is reported by the tool as "0 ms", not as missing data, so
    without this test a silently collapsed span would land in OpenSearch as a
    real-looking measurement.
    """
    record = _jsonl_record(_combined_disagg_record())
    jsonl = tmp_path / "perf_metrics-disagg.jsonl"
    jsonl.write_text(json.dumps(record) + "\n")

    parsed = RequestTimeBreakdown().parse_json_file(str(jsonl))
    assert len(parsed) == 1

    for span, expected_ms in _EXPECTED_SPAN_MS.items():
        assert parsed[0][f"{span}_time"] * 1000 == pytest.approx(expected_ms, abs=1e-3), span


def test_span_statistics_are_reported_in_milliseconds(tmp_path):
    """compute_statistics feeds the perf-sanity metrics; check units and shape."""
    record = _jsonl_record(_combined_disagg_record())
    jsonl = tmp_path / "perf_metrics-disagg.jsonl"
    jsonl.write_text("".join(json.dumps(record) + "\n" for _ in range(3)))

    analyzer = RequestTimeBreakdown()
    stats = analyzer.compute_statistics(analyzer.parse_json_file(str(jsonl)))

    assert set(stats) == set(_EXPECTED_SPAN_MS)
    for span, expected_ms in _EXPECTED_SPAN_MS.items():
        assert stats[span]["count"] == 3
        for statistic in ("mean", "median", "p75", "p99"):
            assert stats[span][statistic] == pytest.approx(expected_ms, abs=1e-3)


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
