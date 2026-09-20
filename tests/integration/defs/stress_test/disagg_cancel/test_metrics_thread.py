# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for ``DisaggCancellationStressHarness._metrics_thread_body``.

The metrics thread is testable without a real disagg cluster: stand
up a tiny HTTP server in the test process that responds at
``/prometheus/metrics``, point a ``WorkerLaunchSpec`` at it, run the
thread for a few scrape intervals, and assert on the collected
samples.

Parse-only behaviour (handling of ``# HELP`` / ``# TYPE`` lines,
labels, scientific notation, missing metric) is verified directly
against the standalone parser; transport behaviour (timeouts,
connection refused, restart-mid-scrape) is verified against the
in-process HTTP server.
"""

from __future__ import annotations

import socket
import textwrap
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from ._testing import DUMMY_YAML, make_spec
from .harness import (
    DisaggCancellationStressHarness,
    WorkerLaunchSpec,
    _fetch_kv_cache_utilization,
    _parse_kv_cache_utilization,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _pick_port() -> int:
    """Bind-and-release to get an OS-allocated free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _MetricsHandler(BaseHTTPRequestHandler):
    """Serves a single configurable body at ``/prometheus/metrics``.

    The body is read off the server instance (``server.metrics_body``)
    on every request so tests can mutate it mid-run to simulate
    utilization drift.
    """

    def do_GET(self) -> None:  # noqa: N802 — fixed by BaseHTTPRequestHandler
        if self.path != "/prometheus/metrics":
            self.send_response(404)
            self.end_headers()
            return
        body: str = getattr(self.server, "metrics_body", "")
        if body is None:
            # Simulate a worker that's mid-restart: serve a 503 so the
            # scraper records a miss rather than a parse failure.
            self.send_response(503)
            self.end_headers()
            return
        encoded = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_args, **_kwargs) -> None:  # silence test noise
        pass


@pytest.fixture
def metrics_server():
    """A single in-process HTTP server serving ``/prometheus/metrics``.

    Yields ``(server, port)``. Tests mutate ``server.metrics_body`` to
    change what the next scrape sees.
    """
    port = _pick_port()
    server = HTTPServer(("127.0.0.1", port), _MetricsHandler)
    # Default empty body; tests set this to drive scrape outcomes.
    server.metrics_body = ""  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def _make_harness(tmp_path: Path, specs: list[WorkerLaunchSpec]) -> DisaggCancellationStressHarness:
    """Construct a harness with a minimal YAML and the given worker specs."""
    yaml_path = tmp_path / "marathon.yaml"
    yaml_path.write_text(DUMMY_YAML)
    h = DisaggCancellationStressHarness(
        yaml_path,
        metrics_scrape_interval_s=0.05,
        metrics_scrape_timeout_s=0.5,
    )
    h._worker_specs = list(specs)
    return h


def _run_metrics_thread_briefly(
    harness: DisaggCancellationStressHarness, duration_s: float
) -> None:
    """Drive ``_metrics_thread_body`` for ``duration_s`` seconds then stop."""
    thread = threading.Thread(target=harness._metrics_thread_body, daemon=True)
    thread.start()
    time.sleep(duration_s)
    harness.stop_event.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive(), "metrics thread failed to exit after stop_event"


def _wait_until(predicate, *, timeout_s: float, poll_s: float = 0.01) -> None:
    """Poll ``predicate`` until true or ``timeout_s`` elapses; assert on timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(poll_s)
    raise AssertionError(f"predicate did not become true within {timeout_s}s")


# ---------------------------------------------------------------------------
# Parser-only tests
# ---------------------------------------------------------------------------


def test_parse_simple_gauge_returns_float() -> None:
    body = textwrap.dedent(
        """\
        # HELP trtllm_kv_cache_utilization KV cache utilization fraction.
        # TYPE trtllm_kv_cache_utilization gauge
        trtllm_kv_cache_utilization 0.42
        """
    )
    assert _parse_kv_cache_utilization(body) == pytest.approx(0.42)


def test_parse_gauge_with_labels_returns_float() -> None:
    body = textwrap.dedent(
        """\
        # TYPE trtllm_kv_cache_utilization gauge
        trtllm_kv_cache_utilization{instance="ctx_0",pool="kv"} 0.875
        """
    )
    assert _parse_kv_cache_utilization(body) == pytest.approx(0.875)


def test_parse_gauge_with_scientific_notation() -> None:
    body = "trtllm_kv_cache_utilization 7.5e-2\n"
    assert _parse_kv_cache_utilization(body) == pytest.approx(0.075)


def test_parse_returns_none_when_metric_absent() -> None:
    body = textwrap.dedent(
        """\
        # HELP trtllm_request_success_total Counter.
        trtllm_request_success_total 17
        """
    )
    assert _parse_kv_cache_utilization(body) is None


def test_parse_skips_help_and_type_lines_with_same_prefix() -> None:
    # The HELP/TYPE lines mention ``trtllm_kv_cache_utilization`` but
    # are not data samples — must not be parsed as values.
    body = textwrap.dedent(
        """\
        # HELP trtllm_kv_cache_utilization KV cache utilization 0.99 (this is a comment).
        # TYPE trtllm_kv_cache_utilization gauge
        """
    )
    assert _parse_kv_cache_utilization(body) is None


def test_parse_returns_first_sample_when_multiple_present() -> None:
    body = textwrap.dedent(
        """\
        trtllm_kv_cache_utilization{instance="ctx_0"} 0.10
        trtllm_kv_cache_utilization{instance="gen_0"} 0.90
        """
    )
    assert _parse_kv_cache_utilization(body) == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# Fetch-only tests (in-process HTTP server)
# ---------------------------------------------------------------------------


def test_fetch_success_returns_value_and_no_error(metrics_server) -> None:
    server, port = metrics_server
    server.metrics_body = "trtllm_kv_cache_utilization 0.31\n"
    util, err = _fetch_kv_cache_utilization("127.0.0.1", port, timeout_s=1.0)
    assert util == pytest.approx(0.31)
    assert err is None


def test_fetch_missing_metric_returns_none_with_metric_absent_error(metrics_server) -> None:
    server, port = metrics_server
    server.metrics_body = "trtllm_other_metric 99\n"
    util, err = _fetch_kv_cache_utilization("127.0.0.1", port, timeout_s=1.0)
    assert util is None
    assert err == "metric_absent"


def test_fetch_connection_refused_returns_url_error() -> None:
    # Pick a port and don't bind it.
    port = _pick_port()
    util, err = _fetch_kv_cache_utilization("127.0.0.1", port, timeout_s=0.5)
    assert util is None
    assert err is not None and err.startswith("url_error")


def test_fetch_http_503_returns_url_error(metrics_server) -> None:
    server, port = metrics_server
    server.metrics_body = None  # handler responds 503
    util, err = _fetch_kv_cache_utilization("127.0.0.1", port, timeout_s=1.0)
    assert util is None
    assert err is not None and err.startswith("url_error")


# ---------------------------------------------------------------------------
# Thread-body integration tests
# ---------------------------------------------------------------------------


def test_thread_records_sample_per_scrape(tmp_path, metrics_server) -> None:
    server, port = metrics_server
    server.metrics_body = "trtllm_kv_cache_utilization 0.25\n"
    spec = make_spec("ctx", 0, host="127.0.0.1", port=port)
    harness = _make_harness(tmp_path, [spec])

    _run_metrics_thread_briefly(harness, duration_s=0.2)

    assert len(harness._kv_utilization_samples) >= 2
    for sample in harness._kv_utilization_samples:
        assert sample["role"] == "ctx"
        assert sample["index"] == 0
        assert sample["host"] == "127.0.0.1"
        assert sample["port"] == port
        assert sample["utilization"] == pytest.approx(0.25)
        assert sample["error"] is None
        assert sample["timestamp"] > 0


def test_thread_exits_immediately_when_no_specs(tmp_path) -> None:
    harness = _make_harness(tmp_path, [])
    thread = threading.Thread(target=harness._metrics_thread_body, daemon=True)
    thread.start()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert harness._kv_utilization_samples == []


def test_thread_records_miss_when_scrape_fails(tmp_path) -> None:
    # Worker port nobody is listening on → connection refused on every scrape.
    port = _pick_port()
    spec = make_spec("gen", 0, host="127.0.0.1", port=port)
    harness = _make_harness(tmp_path, [spec])

    _run_metrics_thread_briefly(harness, duration_s=0.15)

    assert len(harness._kv_utilization_samples) >= 1
    for sample in harness._kv_utilization_samples:
        assert sample["utilization"] is None
        assert sample["error"] is not None
        assert sample["role"] == "gen"


def test_thread_recovers_when_worker_starts_serving_mid_run(tmp_path, metrics_server) -> None:
    server, port = metrics_server
    server.metrics_body = None  # initial: handler 503s every request
    spec = make_spec("ctx", 0, host="127.0.0.1", port=port)
    harness = _make_harness(tmp_path, [spec])

    thread = threading.Thread(target=harness._metrics_thread_body, daemon=True)
    thread.start()
    try:
        # Drive the loop deterministically: wait until at least one miss has
        # been recorded before flipping the handler to a success response.
        _wait_until(
            lambda: any(s["utilization"] is None for s in harness._kv_utilization_samples),
            timeout_s=2.0,
        )
        server.metrics_body = "trtllm_kv_cache_utilization 0.50\n"
        _wait_until(
            lambda: any(s["utilization"] is not None for s in harness._kv_utilization_samples),
            timeout_s=2.0,
        )
    finally:
        harness.stop_event.set()
        thread.join(timeout=2.0)

    samples = harness._kv_utilization_samples
    assert any(s["utilization"] is None for s in samples), "expected a miss before serving"
    hits = [s for s in samples if s["utilization"] is not None]
    assert hits and all(s["utilization"] == pytest.approx(0.50) for s in hits)


def test_thread_scrapes_multiple_workers_per_cycle(tmp_path, metrics_server) -> None:
    server, port = metrics_server
    server.metrics_body = "trtllm_kv_cache_utilization 0.6\n"
    specs = [
        make_spec("ctx", 0, host="127.0.0.1", port=port),
        make_spec("ctx", 1, host="127.0.0.1", port=port),
        make_spec("gen", 0, host="127.0.0.1", port=port),
    ]
    harness = _make_harness(tmp_path, specs)

    _run_metrics_thread_briefly(harness, duration_s=0.15)

    # Each cycle should produce one sample per worker. We tolerate
    # extra samples (timing-dependent) but require coverage of all
    # three roles+indices.
    seen = {(s["role"], s["index"]) for s in harness._kv_utilization_samples}
    assert ("ctx", 0) in seen
    assert ("ctx", 1) in seen
    assert ("gen", 0) in seen


def test_thread_exits_promptly_on_failed_event(tmp_path, metrics_server) -> None:
    server, port = metrics_server
    server.metrics_body = "trtllm_kv_cache_utilization 0.0\n"
    spec = make_spec("ctx", 0, host="127.0.0.1", port=port)
    harness = _make_harness(tmp_path, [spec])

    thread = threading.Thread(target=harness._metrics_thread_body, daemon=True)
    thread.start()
    time.sleep(0.05)
    harness.failed_event.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive(), "metrics thread must exit on failed_event too"
