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
"""Unit tests for ``DisaggCancellationStressHarness._injector_thread_body``.

The injector thread has no dependency on a running disagg cluster:
feed it ``_TrackedWorker`` entries whose ``wrapper`` points at a
test-owned subprocess, drive a short ``injections:`` schedule, and
assert on ``_injection_events`` / ``failure_reason``.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from .harness import (
    DisaggCancellationStressHarness,
    WorkerLaunchSpec,
    _execute_sigkill,
    _execute_sigstop_pause,
    _parse_injection_schedule,
    _resolve_injection_target,
    _TrackedWorker,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_spec(role: str, index: int) -> WorkerLaunchSpec:
    return WorkerLaunchSpec(
        role=role,
        index=index,
        model_name="dummy",
        worker_config={},
        work_dir="/tmp",
        port=19000 + index,
        device="0",
        env={},
        log_path=None,
    )


def _make_wrapper(proc: subprocess.Popen[Any]) -> SimpleNamespace:
    return SimpleNamespace(process=proc, port=0, log_file=None, log_path=None)


@pytest.fixture
def sleeping_subprocess() -> subprocess.Popen[Any]:
    """Long-lived child for SIGSTOP/SIGKILL exercises."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(3600)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    yield proc
    if proc.poll() is None:
        proc.kill()
        proc.wait(timeout=5.0)


def _dummy_yaml_with_injections(injections_yaml: str) -> str:
    """Build a minimal marathon YAML with a custom ``injections:`` block."""
    # Append injections after dedenting the static prefix only — embedding
    # the dynamic block inside dedent() would strip its indentation.
    prefix = textwrap.dedent(
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
          injections:
        """
    )
    injections_block = textwrap.indent(injections_yaml.strip(), "    ")
    return prefix + injections_block + "\n"


def _run_injector_until_done(
    h: DisaggCancellationStressHarness,
    timeout_s: float = 5.0,
) -> None:
    h._marathon_start_monotonic = time.monotonic()
    t = threading.Thread(target=h._injector_thread_body, name="test-injector", daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    assert not t.is_alive(), "injector thread did not exit within timeout"


# ---------------------------------------------------------------------------
# Parser / resolver unit tests
# ---------------------------------------------------------------------------


def test_parse_injection_schedule_sorts_and_validates() -> None:
    raw = [
        {"at_min": 30, "type": "sigstop", "target": "gen_worker_0", "duration_s": 10},
        {"at_min": 15, "type": "sigstop", "target": "ctx_worker_0", "duration_s": 5},
        "not-a-dict",
        {"at_min": 60, "type": "nope", "target": "gen_worker_0"},
    ]
    specs = _parse_injection_schedule(raw)
    assert [s.at_min for s in specs] == [15.0, 30.0]
    assert specs[0].duration_s == 5.0


def test_resolve_injection_target_fixed_index() -> None:
    tracked = [
        _TrackedWorker(spec=_make_spec("gen", 0), wrapper=SimpleNamespace()),
        _TrackedWorker(spec=_make_spec("gen", 1), wrapper=SimpleNamespace()),
    ]
    picked = _resolve_injection_target("gen_worker_0", tracked)
    assert picked.spec.index == 0


def test_resolve_injection_target_random_draws_from_role_pool() -> None:
    tracked = [
        _TrackedWorker(spec=_make_spec("gen", 0), wrapper=SimpleNamespace()),
        _TrackedWorker(spec=_make_spec("gen", 1), wrapper=SimpleNamespace()),
        _TrackedWorker(spec=_make_spec("ctx", 0), wrapper=SimpleNamespace()),
    ]
    picked = _resolve_injection_target("gen_worker_random", tracked)
    assert picked.spec.role == "gen"


def test_resolve_injection_target_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        _resolve_injection_target("bogus_target", [])


# ---------------------------------------------------------------------------
# Signal helpers (real subprocess)
# ---------------------------------------------------------------------------


def test_sigstop_pause_stops_then_resumes_process(
    sleeping_subprocess: subprocess.Popen[Any],
) -> None:
    wrapper = _make_wrapper(sleeping_subprocess)
    assert sleeping_subprocess.poll() is None

    outcome = _execute_sigstop_pause(wrapper, duration_s=0.3)

    assert outcome["sigstop_sent"] is True
    assert outcome["sigcont_sent"] is True
    assert sleeping_subprocess.poll() is None


def test_sigkill_terminates_process(sleeping_subprocess: subprocess.Popen[Any]) -> None:
    wrapper = _make_wrapper(sleeping_subprocess)
    outcome = _execute_sigkill(wrapper)
    assert outcome["sigkill_sent"] is True
    assert sleeping_subprocess.wait(timeout=5.0) is not None


def test_sigstop_on_dead_process_is_skipped_gracefully() -> None:
    proc = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    proc.wait(timeout=5.0)
    wrapper = _make_wrapper(proc)
    outcome = _execute_sigstop_pause(wrapper, duration_s=0.1)
    assert outcome["sigstop_sent"] is False


# ---------------------------------------------------------------------------
# Injector thread integration (harness wiring)
# ---------------------------------------------------------------------------


def test_injector_fires_immediate_sigstop(
    tmp_path: Path, sleeping_subprocess: subprocess.Popen[Any]
) -> None:
    yaml_path = tmp_path / "stress.yaml"
    yaml_path.write_text(
        _dummy_yaml_with_injections(
            textwrap.dedent(
                """\
                - at_min: 0
                  type: sigstop
                  target: gen_worker_0
                  duration_s: 0.2
                """
            )
        )
    )
    h = DisaggCancellationStressHarness(yaml_path, injector_poll_interval_s=0.02)
    h._tracked_workers = [
        _TrackedWorker(spec=_make_spec("gen", 0), wrapper=_make_wrapper(sleeping_subprocess))
    ]

    _run_injector_until_done(h)

    assert len(h._injection_events) == 1
    ev = h._injection_events[0]
    assert ev["type"] == "sigstop"
    assert ev["sigstop_sent"] is True
    assert sleeping_subprocess.poll() is None


def test_injector_runs_two_scheduled_events(
    tmp_path: Path, sleeping_subprocess: subprocess.Popen[Any]
) -> None:
    yaml_path = tmp_path / "stress.yaml"
    yaml_path.write_text(
        _dummy_yaml_with_injections(
            textwrap.dedent(
                """\
                - at_min: 0
                  type: sigstop
                  target: gen_worker_0
                  duration_s: 0.1
                - at_min: 0.05
                  type: sigstop
                  target: gen_worker_0
                  duration_s: 0.1
                """
            )
        )
    )
    h = DisaggCancellationStressHarness(yaml_path, injector_poll_interval_s=0.02)
    h._tracked_workers = [
        _TrackedWorker(spec=_make_spec("gen", 0), wrapper=_make_wrapper(sleeping_subprocess))
    ]

    _run_injector_until_done(h, timeout_s=8.0)

    assert len(h._injection_events) == 2


def test_injector_sigkill_with_respawn_timeout_trips_fail_fast(
    tmp_path: Path, sleeping_subprocess: subprocess.Popen[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    yaml_path = tmp_path / "stress.yaml"
    yaml_path.write_text(
        _dummy_yaml_with_injections(
            textwrap.dedent(
                """\
                - at_min: 0
                  type: sigkill
                  target: gen_worker_0
                  respawn_within_s: 1
                """
            )
        )
    )
    h = DisaggCancellationStressHarness(yaml_path, injector_poll_interval_s=0.02)
    h._tracked_workers = [
        _TrackedWorker(spec=_make_spec("gen", 0), wrapper=_make_wrapper(sleeping_subprocess))
    ]

    monkeypatch.setattr(h, "_respawn_tracked_worker", lambda *a, **k: False)

    _run_injector_until_done(h)

    assert h.failed_event.is_set()
    assert h.failure_reason is not None
    assert "healthy within" in h.failure_reason
    assert h._injection_events[0]["respawned"] is False


def test_injector_sigkill_respawn_success_records_event(
    tmp_path: Path, sleeping_subprocess: subprocess.Popen[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    yaml_path = tmp_path / "stress.yaml"
    yaml_path.write_text(
        _dummy_yaml_with_injections(
            textwrap.dedent(
                """\
                - at_min: 0
                  type: sigkill
                  target: gen_worker_0
                  respawn_within_s: 30
                """
            )
        )
    )
    h = DisaggCancellationStressHarness(yaml_path, injector_poll_interval_s=0.02)
    tracked = _TrackedWorker(spec=_make_spec("gen", 0), wrapper=_make_wrapper(sleeping_subprocess))
    h._tracked_workers = [tracked]
    respawned_proc: subprocess.Popen[Any] | None = None

    def _fake_respawn(t: _TrackedWorker, *, timeout_s: float) -> bool:
        nonlocal respawned_proc
        respawned_proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(3600)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        tracked.wrapper = _make_wrapper(respawned_proc)
        return True

    monkeypatch.setattr(h, "_respawn_tracked_worker", _fake_respawn)

    try:
        _run_injector_until_done(h)

        assert not h.failed_event.is_set()
        assert h._injection_events[0]["respawned"] is True
    finally:
        if respawned_proc is not None and respawned_proc.poll() is None:
            respawned_proc.kill()
            respawned_proc.wait(timeout=5.0)


def test_respawn_uses_allocated_port_and_bounded_health_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    yaml_path = tmp_path / "stress.yaml"
    yaml_path.write_text(
        textwrap.dedent(
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
              injections: []
            """
        )
    )
    h = DisaggCancellationStressHarness(yaml_path)
    spec = _make_spec("gen", 0)
    tracked = _TrackedWorker(
        spec=spec,
        wrapper=SimpleNamespace(process=None, port=19000, log_file=None, log_path=None),
    )

    def _fake_run_worker(
        model_name: str,
        worker_config: dict[str, Any],
        role: str,
        port: int,
        work_dir: str,
        device: str,
        save_log: bool,
        env: dict[str, str],
    ) -> SimpleNamespace:
        assert model_name == spec.model_name
        assert worker_config == spec.worker_config
        assert role == "gen"
        assert port == 23456
        assert work_dir == spec.work_dir
        assert device == spec.device
        assert save_log is False
        assert env == spec.env
        return SimpleNamespace(process=None, port=port, log_file=None, log_path=None)

    health_calls: list[tuple[int, float]] = []

    def _fake_health(port: int, *, timeout_s: float) -> bool:
        health_calls.append((port, timeout_s))
        return True

    fake_disagg_test_utils = ModuleType("disagg_test_utils")
    fake_disagg_test_utils.get_free_port = lambda: 23456
    fake_disagg_test_utils._run_worker = _fake_run_worker
    monkeypatch.setitem(sys.modules, "disagg_test_utils", fake_disagg_test_utils)
    monkeypatch.setattr(h, "_wait_for_worker_health", _fake_health)

    assert h._respawn_tracked_worker(tracked, timeout_s=7.0) is True
    assert tracked.wrapper.port == 23456
    assert spec.port == 23456
    assert health_calls == [(23456, 7.0)]


def test_injector_exits_on_stop_event_before_distant_injection(tmp_path: Path) -> None:
    yaml_path = tmp_path / "stress.yaml"
    yaml_path.write_text(
        _dummy_yaml_with_injections(
            textwrap.dedent(
                """\
                - at_min: 999
                  type: sigstop
                  target: gen_worker_0
                  duration_s: 1
                """
            )
        )
    )
    h = DisaggCancellationStressHarness(yaml_path, injector_poll_interval_s=0.05)
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(3600)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        h._tracked_workers = [
            _TrackedWorker(spec=_make_spec("gen", 0), wrapper=_make_wrapper(proc))
        ]
        h._marathon_start_monotonic = time.monotonic()
        t = threading.Thread(target=h._injector_thread_body, daemon=True)
        t.start()
        time.sleep(0.15)
        h.stop_event.set()
        t.join(timeout=2.0)
        assert not t.is_alive()
        assert h._injection_events == []
        # Distant injection never fired — child should still be running.
        time.sleep(0.1)
        assert proc.poll() is None
    finally:
        proc.kill()
        proc.wait(timeout=5.0)


def test_injector_no_tracked_workers_exits_without_events(tmp_path: Path) -> None:
    yaml_path = tmp_path / "stress.yaml"
    yaml_path.write_text(
        _dummy_yaml_with_injections(
            textwrap.dedent(
                """\
                - at_min: 0
                  type: sigstop
                  target: gen_worker_0
                  duration_s: 1
                """
            )
        )
    )
    h = DisaggCancellationStressHarness(yaml_path)
    _run_injector_until_done(h)
    assert h._injection_events == []


def test_injector_empty_schedule_exits_immediately(tmp_path: Path) -> None:
    yaml_path = tmp_path / "stress.yaml"
    yaml_path.write_text(
        textwrap.dedent(
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
              injections: []
            """
        )
    )
    h = DisaggCancellationStressHarness(yaml_path)
    _run_injector_until_done(h)
    assert h._injection_events == []


def test_injector_skips_unknown_target_without_fail_fast(
    tmp_path: Path, sleeping_subprocess: subprocess.Popen[Any]
) -> None:
    yaml_path = tmp_path / "stress.yaml"
    yaml_path.write_text(
        _dummy_yaml_with_injections(
            textwrap.dedent(
                """\
                - at_min: 0
                  type: sigstop
                  target: gen_worker_99
                  duration_s: 0.1
                """
            )
        )
    )
    h = DisaggCancellationStressHarness(yaml_path, injector_poll_interval_s=0.02)
    h._tracked_workers = [
        _TrackedWorker(spec=_make_spec("gen", 0), wrapper=_make_wrapper(sleeping_subprocess))
    ]

    _run_injector_until_done(h)

    assert not h.failed_event.is_set()
    assert len(h._injection_events) == 1
    assert "skipped" in h._injection_events[0]
