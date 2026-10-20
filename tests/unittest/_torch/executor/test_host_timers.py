# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TRTLLM_HOST_TIMERS (measurement only): per-iteration host time, SYNC / COMM split, batch keys."""

import contextlib
import sys
import time
import types

import pytest

from tensorrt_llm._torch.pyexecutor import host_timers

pytestmark = pytest.mark.cpu_only

_MODULE = "host_timers_fake_targets"


class _Batch:
    def __init__(self, num_ctx, num_gen):
        self.num_context_requests = num_ctx
        self.num_generation_requests = num_gen


def _make_fake_module():
    module = types.ModuleType(_MODULE)

    class PyExecutor:  # the qualname host_timers keys batches on
        def _forward_step(self, scheduled_requests):
            time.sleep(0.002)

    class FakeDist:
        def allgather(self, obj):
            time.sleep(0.001)
            return [obj]

    class Waiter:
        def synchronize(self):
            time.sleep(0.003)

    PyExecutor.__module__ = FakeDist.__module__ = Waiter.__module__ = _MODULE
    module.PyExecutor, module.FakeDist, module.Waiter = PyExecutor, FakeDist, Waiter
    return module


class _FakeExecutor:
    """The two hooks install_host_timers uses: _profiler and event_loop."""

    def __init__(self, module, batches):
        self.module, self.batches = module, batches

    @contextlib.contextmanager
    def _profiler(self):
        yield lambda: None

    def event_loop(self):
        executor = self.module.PyExecutor()
        dist, waiter = self.module.FakeDist(), self.module.Waiter()
        with self._profiler() as step:
            for batch in self.batches:
                step()
                executor._forward_step(batch)
                dist.allgather(1)
                waiter.synchronize()
            step()


@pytest.fixture
def fake_module(monkeypatch):
    module = _make_fake_module()
    monkeypatch.setitem(sys.modules, _MODULE, module)
    # A fresh registry so the targets of this test are the only ones wrapped.
    monkeypatch.setattr(host_timers, "_REGISTRY", host_timers._Registry())
    monkeypatch.setattr(host_timers, "COMM_TARGETS", [f"{_MODULE}:FakeDist.allgather"])
    # Leave torch's sync points alone; the fake waiter is the SYNC leaf here.
    monkeypatch.setattr(
        host_timers._Registry,
        "install_sync_points",
        lambda self: self.install_target(
            f"{_MODULE}:Waiter.synchronize", prefix=host_timers._SYNC_PREFIX
        ),
    )
    return module


def _run(monkeypatch, tmp_path, module, batches, by_batch):
    monkeypatch.setenv("TRTLLM_HOST_TIMERS", "1-1000")
    monkeypatch.setenv("TRTLLM_HOST_TIMERS_ONLY", "1")
    monkeypatch.setenv("TRTLLM_HOST_TIMERS_OUT", str(tmp_path))
    monkeypatch.setenv("TRTLLM_HOST_TIMERS_FUNCS", f"{_MODULE}:PyExecutor._forward_step")
    if by_batch:
        monkeypatch.setenv("TRTLLM_HOST_TIMERS_BY_BATCH", "1")
    else:
        monkeypatch.delenv("TRTLLM_HOST_TIMERS_BY_BATCH", raising=False)
    executor = _FakeExecutor(module, batches)
    host_timers.install_host_timers(executor)
    executor.event_loop()
    files = list(tmp_path.glob("hosttimers_r*_*.txt"))
    assert len(files) == 1
    return files[0].read_text()


def _row(text, label, column=0):
    """A summary row's median (column 0) or mean (column 2: '<med> med <mean> mean ...')."""
    for line in text.splitlines():
        if f"ms/iter  {label}" in line:
            return float(line.split()[column])
    raise AssertionError(f"no row {label!r} in\n{text}")


def test_written_at_loop_exit_with_comm_row(monkeypatch, tmp_path, fake_module):
    batches = [_Batch(0, 1)] * 4
    text = _run(monkeypatch, tmp_path, fake_module, batches, by_batch=False)
    # Every pass: forward 2 ms + allgather 1 ms (COMM) + wait 3 ms (SYNC).
    # (sleep may overshoot, so the bounds are one-sided where it matters)
    period, sync, busy, comm = (_row(text, f"{k} (") for k in ("PERIOD", "SYNC", "BUSY", "COMM"))
    assert period >= 6.0
    assert 3.0 <= sync < period
    assert 1.0 <= comm < sync
    # BUSY = PERIOD - SYNC per iteration, so the means (not the medians) subtract exactly
    assert _row(text, "BUSY (", 2) == pytest.approx(
        _row(text, "PERIOD (", 2) - _row(text, "SYNC (", 2), abs=0.002
    )
    assert busy >= 3.0  # forward + allgather
    assert "PyExecutor._forward_step" in text
    assert "comm:FakeDist.allgather" in text
    assert "[gen1]" not in text


def test_batch_keys(monkeypatch, tmp_path, fake_module):
    batches = [_Batch(1, 0)] * 2 + [_Batch(0, 1)] * 3 + [_Batch(0, 2)] * 3
    text = _run(monkeypatch, tmp_path, fake_module, batches, by_batch=True)
    assert "# 3 iterations [gen1]" in text
    assert "# 3 iterations [gen2]" in text
    assert "# 2 iterations [ctx1+gen0]" in text
    assert "# 8 iterations (forward passes 1..999, all batches)" in text


def test_spike_attribution_and_gc_rows(monkeypatch, tmp_path, fake_module):
    """Every 5th forward is 3 ms slower and collects garbage: the spike summary names the forward, and the
    collections show up as gc:gen2 rows."""
    import gc

    count = {"n": 0}
    original = fake_module.PyExecutor._forward_step

    def forward(self, scheduled_requests):
        count["n"] += 1
        if count["n"] % 5 == 0:
            time.sleep(0.003)
            gc.collect()
        return original(self, scheduled_requests)

    fake_module.PyExecutor._forward_step = forward
    batches = [_Batch(0, 1)] * 40
    text = _run(monkeypatch, tmp_path, fake_module, batches, by_batch=False)
    spike_lines = [line for line in text.splitlines() if line.startswith("#   ")][:2]
    assert "spikes (forward passes" in text
    # the two causes of the spikes, in either order (gc.collect() of the test process takes a few ms too)
    assert any("PyExecutor._forward_step" in line for line in spike_lines), spike_lines
    assert any("gc:gen2" in line for line in spike_lines), spike_lines


def test_disabled_is_noop(monkeypatch, tmp_path, fake_module):
    monkeypatch.delenv("TRTLLM_HOST_TIMERS", raising=False)
    executor = _FakeExecutor(fake_module, [_Batch(0, 1)])
    original = executor._profiler
    host_timers.install_host_timers(executor)
    assert executor._profiler == original
    assert not host_timers._REGISTRY.installed


def test_light_mode_keeps_the_forward_hook(monkeypatch, tmp_path):
    """TRTLLM_HOST_TIMERS_ONLY=1 without TRTLLM_HOST_TIMERS_FUNCS still times the forward passes (the forward-pass
    hook is always installed); before the fix it timed nothing and wrote no summary."""
    module = _make_fake_module()
    monkeypatch.setitem(sys.modules, _MODULE, module)
    monkeypatch.setattr(host_timers, "_REGISTRY", host_timers._Registry())
    monkeypatch.setattr(host_timers, "COMM_TARGETS", [])
    monkeypatch.setattr(host_timers._Registry, "install_sync_points", lambda self: None)
    monkeypatch.setattr(host_timers, "_PX", f"{_MODULE}:PyExecutor.")
    monkeypatch.setenv("TRTLLM_HOST_TIMERS", "1-1000")
    monkeypatch.setenv("TRTLLM_HOST_TIMERS_ONLY", "1")
    monkeypatch.setenv("TRTLLM_HOST_TIMERS_BY_BATCH", "1")
    monkeypatch.setenv("TRTLLM_HOST_TIMERS_OUT", str(tmp_path))
    monkeypatch.delenv("TRTLLM_HOST_TIMERS_FUNCS", raising=False)
    executor = _FakeExecutor(module, [_Batch(0, 1)] * 5)
    host_timers.install_host_timers(executor)
    executor.event_loop()
    files = list(tmp_path.glob("hosttimers_r*_*.txt"))
    assert len(files) == 1
    text = files[0].read_text()
    assert _row(text, "PERIOD") > 0 and "[gen1]" in text
