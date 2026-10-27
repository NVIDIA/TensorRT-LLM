# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostics armed by environment only: the phase-armed Nsight window (TRTLLM_M31_PHASE_NSYS_ARM,
phase_capture.py) and the per-rank host-path probe (TRTLLM_M31_HOST_PATH_PROBE, host_path_probe.py).
Off, the executor's per-iteration cost is one None check (profiling.profile_step). CPU only."""

import array
import ast
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor import host_path_probe as hpp
from tensorrt_llm._torch.pyexecutor import phase_capture as pc
from tensorrt_llm._torch.pyexecutor import profiling

pytestmark = pytest.mark.cpu_only


# ---- off path -------------------------------------------------------------------------------------------------


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(pc.PHASE_NSYS_ARM_ENV_VAR_NAME, raising=False)
    monkeypatch.delenv(hpp.HOST_PATH_PROBE_ENV_VAR_NAME, raising=False)
    executor = SimpleNamespace(dist=SimpleNamespace(rank=0), global_rank=0)
    fwd, upd = object(), object()
    executor._forward_step, executor._update_requests = fwd, upd
    assert pc.make_phase_capture(None, executor, conflicting_env=False) is None
    assert hpp.make_host_path_probe(executor) is None
    assert executor._forward_step is fwd and executor._update_requests is upd


def test_per_iteration_cost_is_one_none_check():
    """profile_step_fn touches the diagnostics only through `if per_step_diag is not None: per_step_diag()`."""
    tree = ast.parse(Path(profiling.__file__).read_text())
    (fn,) = [
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "profile_step_fn"
    ]
    names = [n.id for n in ast.walk(fn) if isinstance(n, ast.Name)]
    assert "phase_capture" not in names and "host_path_probe" not in names
    guards = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Compare)
        and isinstance(n.test.left, ast.Name)
        and n.test.left.id == "per_step_diag"
        and isinstance(n.test.ops[0], ast.IsNot)
    ]
    assert len(guards) == 1 and names.count("per_step_diag") == 2


# ---- phase capture --------------------------------------------------------------------------------------------


class _Manager:
    def __init__(self):
        self.starts, self.stop_configs = [], 0

    def start_profile(self, **kwargs):
        self.starts.append(kwargs)

    def apply_stop_config(self):
        self.stop_configs += 1


def _executor(rank=0):
    queue = SimpleNamespace(stops=0)
    queue.enqueue_profile_stop_request = lambda: setattr(queue, "stops", queue.stops + 1)
    return SimpleNamespace(
        dist=SimpleNamespace(rank=rank),
        iter_counter=0,
        is_warmup=False,
        _profile_enabled=False,
        executor_request_queue=queue,
    )


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t


def _capture(tmp_path, monkeypatch, marker=None, rank=0):
    path = tmp_path / "arm.json"
    if marker is not None:
        path.write_text(json.dumps(marker))
    monkeypatch.setenv(pc.PHASE_NSYS_ARM_ENV_VAR_NAME, str(path))
    clock = _Clock()
    monkeypatch.setattr(pc.time, "monotonic", clock.monotonic)
    manager, executor = _Manager(), _executor(rank)
    return (
        pc.make_phase_capture(manager, executor, conflicting_env=False),
        manager,
        executor,
        clock,
        path,
    )


def test_rank0_only_and_conflicts(tmp_path, monkeypatch):
    cap, *_ = _capture(tmp_path, monkeypatch, rank=1)
    assert cap is None
    monkeypatch.setenv(pc.PHASE_NSYS_ARM_ENV_VAR_NAME, str(tmp_path / "arm.json"))
    assert pc.make_phase_capture(_Manager(), _executor(), conflicting_env=True) is None


def test_polls_at_most_once_per_second_then_starts(tmp_path, monkeypatch):
    cap, manager, executor, clock, path = _capture(tmp_path, monkeypatch)
    cap.tick()  # no marker yet: polled once
    path.write_text(
        json.dumps({"schema_version": 1, "capture_at_unix_s": time.time() - 1, "num_steps": 50})
    )
    clock.t += 0.5
    cap.tick()  # within the second: not re-read
    assert manager.starts == []
    clock.t += 0.6
    cap.tick()  # re-read and armed; capture time already passed: started
    assert manager.starts == [{"num_steps": 50, "activities": ["CUDA_PROFILER"]}]


def test_waits_for_capture_time_and_skips_warmup(tmp_path, monkeypatch):
    cap, manager, executor, clock, _ = _capture(
        tmp_path, monkeypatch, {"schema_version": 1, "capture_at_unix_s": time.time() + 3600}
    )
    executor.is_warmup = True
    cap.tick()
    assert cap._capture_at is None  # nothing read during warmup
    executor.is_warmup = False
    cap.tick()
    assert cap._capture_at is not None and manager.starts == []


def test_stops_on_num_steps(tmp_path, monkeypatch):
    cap, manager, executor, clock, _ = _capture(
        tmp_path, monkeypatch, {"schema_version": 1, "capture_at_unix_s": 1.0}
    )
    cap.tick()
    executor._profile_enabled = True
    cap.tick()
    executor._profile_enabled = False  # the broadcast stop fired after num_steps
    cap.tick()
    assert cap.done and manager.stop_configs == 0


def test_stops_on_max_seconds_without_blocking(tmp_path, monkeypatch):
    cap, manager, executor, clock, _ = _capture(
        tmp_path, monkeypatch, {"schema_version": 1, "capture_at_unix_s": 1.0, "max_seconds": 2}
    )
    cap.tick()
    executor._profile_enabled = True
    clock.t += 2.5
    cap.tick()
    assert cap.done and manager.stop_configs == 1 and executor.executor_request_queue.stops == 1


@pytest.mark.parametrize(
    "marker",
    [
        {"schema_version": 2, "capture_at_unix_s": 1.0},
        {"schema_version": 1},
        {"schema_version": 1, "capture_at_unix_s": 1.0, "num_steps": 0},
        {"schema_version": 1, "capture_at_unix_s": 1.0, "max_seconds": 500},
    ],
)
def test_bad_marker_disarms(tmp_path, monkeypatch, marker):
    cap, manager, *_ = _capture(tmp_path, monkeypatch, marker)
    cap.tick()
    assert cap.done and manager.starts == []


# ---- host path probe ------------------------------------------------------------------------------------------


class _Event:
    def __init__(self):
        self.syncs = 0

    def synchronize(self):
        self.syncs += 1


def test_probe_rows_and_passthrough(tmp_path, monkeypatch):
    monkeypatch.setenv(hpp.HOST_PATH_PROBE_ENV_VAR_NAME, str(tmp_path))
    calls = []
    executor = SimpleNamespace(dist=SimpleNamespace(rank=0), global_rank=3)
    executor._forward_step = lambda req, *a, **k: calls.append(("fwd", req, a, k)) or "out"
    executor._update_requests = lambda state, *a, **k: calls.append(("upd", state, a, k)) or None
    probe = hpp.make_host_path_probe(executor)
    req = SimpleNamespace(num_context_requests=2, generation_requests=[1, 2, 3])
    event = _Event()
    state = SimpleNamespace(sampler_event=event)
    for _ in range(3):
        probe.top()
        assert executor._forward_step(req, "t", x=1) == "out"
        executor._update_requests(state, "rm")
    probe.top()
    probe.flush()
    (f,) = list(tmp_path.glob("hostpath-rank3-pid*.bin"))
    rows = array.array("q")
    rows.frombytes(f.read_bytes())
    assert len(rows) == 3 * 7
    for i in range(3):
        t_top, t_launch, t_fwd, t_w0, t_w1, n_ctx, n_gen = rows[7 * i : 7 * i + 7]
        assert t_top <= t_launch <= t_fwd <= t_w0 <= t_w1
        assert (n_ctx, n_gen) == (2, 3)
    assert event.syncs == 3
    assert calls[0] == ("fwd", req, ("t",), {"x": 1}) and calls[1] == ("upd", state, ("rm",), {})
