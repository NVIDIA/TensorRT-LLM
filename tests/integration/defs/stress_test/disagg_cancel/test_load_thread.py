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
"""Unit tests for ``DisaggCancellationStressHarness._load_thread_body``.

The load thread is tested with a monkeypatched
``_run_cancel_stress_iteration`` so these tests do not import the
heavy disaggregated integration module, start a server, or require
GPU/model resources.
"""

from __future__ import annotations

import textwrap
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from .harness import (
    DisaggCancellationStressHarness,
    StressConfig,
    _load_iteration_shape,
    _parse_cancel_after_range,
    _parse_token_range,
)


def _write_load_yaml(
    tmp_path: Path,
    *,
    extra_stress_config: str = "",
) -> Path:
    """Write a minimal valid marathon YAML for load-thread tests."""
    yaml_path = tmp_path / "stress.yaml"
    content = textwrap.dedent(
        """\
        hostname: localhost
        model: dummy
        backend: pytorch
        context_servers: {}
        generation_servers: {}
        stress_config:
          duration_min: 1
          kv_cache_manager: v1
          transceiver: cpp
          base_concurrency: 4
          input_length:
            distribution: uniform
            min_tokens: 11
            max_tokens: 22
        """
    )
    if extra_stress_config:
        content += textwrap.indent(textwrap.dedent(extra_stress_config).strip(), "  ") + "\n"
    yaml_path.write_text(content)
    return yaml_path


def _make_harness(
    tmp_path: Path,
    *,
    extra_stress_config: str = "",
    load_duration_s: float = 0.05,
) -> DisaggCancellationStressHarness:
    """Construct a load-thread harness with a short test duration."""
    h = DisaggCancellationStressHarness(
        _write_load_yaml(tmp_path, extra_stress_config=extra_stress_config),
        load_duration_s=load_duration_s,
        load_iteration_pause_s=0.005,
    )
    h.bind_server_endpoint("http://127.0.0.1:8000", "test-model")
    h._marathon_start_monotonic = time.monotonic()
    return h


def _run_load_thread(h: DisaggCancellationStressHarness, timeout_s: float = 2.0) -> None:
    """Run the load thread to self-exit and assert it joined."""
    thread = threading.Thread(target=h._load_thread_body, name="test-load", daemon=True)
    thread.start()
    thread.join(timeout=timeout_s)
    assert not thread.is_alive(), "load thread did not exit within timeout"


def test_parse_token_range_defaults_and_validates() -> None:
    assert _parse_token_range(None, (1, 2), "input") == (1, 2)
    assert _parse_token_range({"min_tokens": "3", "max_tokens": 5}, (1, 2), "input") == (
        3,
        5,
    )
    with pytest.raises(ValueError, match="min_tokens <= max_tokens"):
        _parse_token_range({"min_tokens": 8, "max_tokens": 7}, (1, 2), "input")


def test_parse_cancel_after_range_defaults_and_validates() -> None:
    assert _parse_cancel_after_range(None) == pytest.approx((0.01, 0.1))
    assert _parse_cancel_after_range({"min_s": 0.2, "max_s": 0.4}) == pytest.approx((0.2, 0.4))
    with pytest.raises(ValueError, match="0 <= min <= max"):
        _parse_cancel_after_range({"min_s": 0.5, "max_s": 0.4})


def test_load_iteration_shape_switches_from_steady_to_burst(tmp_path: Path) -> None:
    yaml_path = _write_load_yaml(
        tmp_path,
        extra_stress_config=textwrap.dedent(
            """\
            bursts:
              interval_min: 1
              concurrency: 9
              duration_s: 10
              input_length:
                min_tokens: 33
                max_tokens: 44
            """
        ),
    )
    cfg = StressConfig.from_yaml_path(yaml_path)

    steady = _load_iteration_shape(cfg, elapsed_s=30)
    assert steady["mode"] == "steady"
    assert steady["requests_per_burst"] == 4
    assert steady["prompt_len_range"] == (11, 22)

    burst = _load_iteration_shape(cfg, elapsed_s=61)
    assert burst["mode"] == "burst"
    assert burst["requests_per_burst"] == 9
    assert burst["prompt_len_range"] == (33, 44)


@pytest.mark.parametrize(
    ("burst_config", "match"),
    [
        (
            "interval_min: 0\nconcurrency: 9\nduration_s: 10\n",
            "bursts.interval_min must be positive",
        ),
        (
            "interval_min: 1\nconcurrency: 9\nduration_s: 0\n",
            "bursts.duration_s must be positive",
        ),
    ],
)
def test_load_iteration_shape_rejects_invalid_burst_timing(
    tmp_path: Path, burst_config: str, match: str
) -> None:
    yaml_path = _write_load_yaml(
        tmp_path,
        extra_stress_config="bursts:\n" + textwrap.indent(burst_config, "  "),
    )
    cfg = StressConfig.from_yaml_path(yaml_path)

    with pytest.raises(ValueError, match=match):
        _load_iteration_shape(cfg, elapsed_s=61)


def test_load_thread_without_server_endpoint_exits_and_signals_stop(tmp_path: Path) -> None:
    h = DisaggCancellationStressHarness(
        _write_load_yaml(tmp_path),
        load_duration_s=0.05,
        load_iteration_pause_s=0.005,
    )

    _run_load_thread(h)

    assert h.stop_event.is_set()
    assert h._load_records == []


def test_load_thread_runs_steady_iterations_and_records_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = _make_harness(tmp_path)
    calls: list[dict[str, Any]] = []

    def fake_runner(**kwargs: Any) -> None:
        calls.append(kwargs)
        time.sleep(0.002)

    monkeypatch.setattr(h, "_run_cancel_stress_iteration", fake_runner)

    _run_load_thread(h)

    assert h.stop_event.is_set()
    assert not h.failed_event.is_set()
    assert len(calls) >= 1
    assert len(h._load_records) == len(calls)
    assert all(record["mode"] == "steady" for record in h._load_records)
    assert calls[0]["server_url"] == "http://127.0.0.1:8000"
    assert calls[0]["num_bursts"] == 1
    assert calls[0]["requests_per_burst"] == 4
    assert calls[0]["prompt_len_range"] == (11, 22)


def test_load_thread_uses_burst_shape_inside_burst_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = _make_harness(
        tmp_path,
        extra_stress_config=textwrap.dedent(
            """\
            bursts:
              interval_min: 0.001
              concurrency: 9
              duration_s: 0.04
              input_length:
                min_tokens: 33
                max_tokens: 44
            """
        ),
        load_duration_s=0.12,
    )

    def fake_runner(**_kwargs: Any) -> None:
        time.sleep(0.004)

    monkeypatch.setattr(h, "_run_cancel_stress_iteration", fake_runner)

    _run_load_thread(h)

    modes = {record["mode"] for record in h._load_records}
    assert modes == {"steady", "burst"}
    burst_records = [record for record in h._load_records if record["mode"] == "burst"]
    assert burst_records
    assert all(record["requests_per_burst"] == 9 for record in burst_records)
    assert all(record["prompt_len_range"] == (33, 44) for record in burst_records)


def test_load_thread_observes_stop_event_after_runner_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = _make_harness(tmp_path, load_duration_s=10.0)
    calls: list[dict[str, Any]] = []

    def fake_runner(**kwargs: Any) -> None:
        calls.append(kwargs)
        h.stop_event.set()

    monkeypatch.setattr(h, "_run_cancel_stress_iteration", fake_runner)

    _run_load_thread(h)

    assert len(calls) == 1
    assert len(h._load_records) == 1
    assert h._load_records[0]["success"] is True


def test_load_thread_runner_exception_trips_fail_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = _make_harness(tmp_path, load_duration_s=10.0)

    def fake_runner(**_kwargs: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(h, "_run_cancel_stress_iteration", fake_runner)

    _run_load_thread(h)

    assert h.failed_event.is_set()
    assert h.failure_reason == "load_thread runner failed: RuntimeError: boom"
    assert len(h._load_records) == 1
    assert h._load_records[0]["success"] is False
    assert h._load_records[0]["error"] == "RuntimeError: boom"
