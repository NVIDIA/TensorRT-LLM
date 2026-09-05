# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""CPU-only tests for the opt-in fill-gate GPU keepalive.

The keepalive's CUDA surface is replaced by a fake. The gate wiring is tested
against the production method bound onto a bare stub; the no-keepalive path
is covered by the existing gate tests in test_benchmark_disagg.py.
"""

import ast
import collections
import inspect
import threading
from unittest.mock import Mock, patch

import pytest

from tensorrt_llm._torch.pyexecutor import gpu_keepalive as ka

pytestmark = pytest.mark.cpu_only


class _FakeEvent:
    """Event on one in-order fake stream: synchronize() completes this chunk
    and every earlier one, as CUDA stream ordering guarantees."""

    def __init__(self, done_ref, all_chunks):
        self._done_ref = done_ref
        self._all_chunks = all_chunks

    def query(self):
        return self._done_ref["done"]

    def synchronize(self):
        for ref in self._all_chunks:
            ref["done"] = True
            if ref is self._done_ref:
                break


class FakeKeepalive(ka.GpuKeepalive):
    """GpuKeepalive with every CUDA touch replaced by bookkeeping."""

    def __init__(self, queue_depth=ka._QUEUE_DEPTH, mode="spin"):
        # Bypass GpuKeepalive.__init__ (device probe); pretend the first tick already ran.
        self.device = "fake"
        self.period_s = ka._PERIOD_SEC
        self.queue_depth = queue_depth
        self.mode = mode
        self._initialized = True
        self._disabled = False
        self._stream = Mock()
        self._selftest = None
        self._inflight = collections.deque()
        self._launches = 0
        self._last_launch = 0.0
        self._active = False
        self._grid = 304
        self._recalibrate_when_hot = False
        self._sink = self._a = self._b = self._c = Mock()  # the buffers _release must drop
        self.chunks = []
        self.raise_on_launch = None

    def _launch_chunk(self):
        if self.raise_on_launch is not None:
            raise self.raise_on_launch
        ref = {"done": False}
        self.chunks.append(ref)
        self._inflight.append(_FakeEvent(ref, self.chunks))
        self._launches += 1


@pytest.fixture
def clock():
    now = {"t": 1000.0}
    with patch.object(ka.time, "monotonic", side_effect=lambda: now["t"]):
        yield now


@pytest.fixture(autouse=True)
def quiet_logger():
    log = Mock()
    with patch.object(ka, "_log", return_value=log):
        yield log


def _released(k):
    return (
        k._sink is None
        and k._a is None
        and k._b is None
        and k._c is None
        and k._stream is None
        and not k._inflight
        and not k._initialized
    )


# ---------------------------------------------------------------- keepalive


def test_tick_drain_release_and_errors(clock):
    k = FakeKeepalive(queue_depth=2)
    assert k.tick() is True
    assert k.tick() is True  # spin: two ~period chunks queued back to back
    assert k.tick() is False  # two resident, none done
    k.chunks[0]["done"] = True  # oldest finishes -> one slot opens
    assert k.tick() is True
    assert k.launches == 3
    k.drain()
    assert all(ref["done"] for ref in k.chunks)
    assert _released(k) and not k._active  # the gate is one-shot: memory goes back

    mm = FakeKeepalive(queue_depth=2, mode="mm")
    assert mm.tick() is True
    assert mm.tick() is False  # one GEMM burst per period, however fast the loop retries
    clock["t"] += ka._PERIOD_SEC
    assert mm.tick() is True

    # Any error disables the keepalive without raising; submitted work is waited for.
    class _CompileError(Exception):  # Triton raises its own classes, not RuntimeError
        pass

    k = FakeKeepalive()
    stream = k._stream
    k.raise_on_launch = _CompileError("ptxas failed")
    assert k.tick() is False
    stream.synchronize.assert_called_once()
    assert k._disabled and _released(k)
    assert k.tick() is False  # inert afterwards

    k = FakeKeepalive()
    assert k.tick() is True
    k._inflight[-1].synchronize = Mock(side_effect=RuntimeError("event sync failed"))
    k.drain()  # drain error: disabled, released, no exception
    assert k._disabled and _released(k)

    k = FakeKeepalive()
    k._initialized = False
    k._initialize = Mock(side_effect=RuntimeError("no CUDA device"))
    assert k.tick() is False  # first-tick device setup fails: disabled, nothing launched
    assert k._disabled and k.launches == 0


def test_selftest_runs_asynchronously(monkeypatch, clock, quiet_logger):
    """The compile check is a polled child: ticks launch nothing until it reports,
    the verdict is cached per device, and a hung child is killed into the fallback."""
    children = []

    class FakeChild:
        def __init__(self, device_index):
            self.rc, self.out, self.killed, self.started = None, b"", False, clock["t"]
            children.append(self)

        def poll(self):
            return self.rc

        def kill(self):
            self.killed = True

        def output(self):
            return self.out, "tail"

    monkeypatch.setattr(ka, "_SelftestChild", FakeChild)
    monkeypatch.setattr(ka, "_HAVE_TRITON", True)
    monkeypatch.setattr(ka, "_SELFTEST_VERDICT", {})
    monkeypatch.setattr(ka.torch.cuda, "Stream", Mock())
    monkeypatch.setattr(
        ka.torch.cuda, "get_device_properties", Mock(return_value=Mock(multi_processor_count=152))
    )

    def fake_init_spin(self):
        self.mode, self._ns_per_iter, self._recalibrate_when_hot = "spin", 3.4, False

    monkeypatch.setattr(ka.GpuKeepalive, "_init_spin", fake_init_spin)
    monkeypatch.setattr(ka.GpuKeepalive, "_init_mm", lambda self: setattr(self, "mode", "mm"))
    monkeypatch.setattr(
        ka.GpuKeepalive,
        "_launch_chunk",
        lambda self: setattr(self, "_launches", self._launches + 1),
    )

    k = ka.GpuKeepalive(0)
    assert (
        k.tick() is False and len(children) == 1 and k.launches == 0
    )  # child started, nothing launched
    assert k.tick() is False and len(children) == 1  # still running: keep polling, no new child
    children[0].rc, children[0].out = 0, ka._SELFTEST_OK_MARKER.encode()
    assert k.tick() is True and k.mode == "spin" and k._grid == 304
    assert ka.GpuKeepalive(0).tick() is True and len(children) == 1  # verdict cached per device

    monkeypatch.setattr(ka, "_SELFTEST_VERDICT", {})
    k = ka.GpuKeepalive(1)
    k.tick()
    children[-1].rc = ka._SELFTEST_RC_NO_CONTEXT
    assert k.tick() is True and k.mode == "mm"
    assert "CUDA context" in quiet_logger.warning.call_args.args[0]

    monkeypatch.setattr(ka, "_SELFTEST_VERDICT", {})
    k = ka.GpuKeepalive(2)
    k.tick()
    clock["t"] += ka._SELFTEST_TIMEOUT_SEC + 1
    assert k.tick() is True and k.mode == "mm" and children[-1].killed  # hung compiler: fallback
    assert (
        ka.GpuKeepalive(2).tick() is True and len(children) == 3
    )  # a failed verdict is cached too

    monkeypatch.setattr(ka, "_SELFTEST_VERDICT", {})
    k = ka.GpuKeepalive(3)
    k.tick()
    children[-1].rc = 0  # exited 0 but never printed the marker: not trusted
    assert k.tick() is True and k.mode == "mm"

    monkeypatch.setattr(ka, "_SELFTEST_VERDICT", {})
    k = ka.GpuKeepalive(4)
    k.tick()  # child pending ...
    k.close()  # ... and the executor loop exits: child stopped, nothing leaks, no verdict recorded
    assert (
        children[-1].killed
        and k._selftest is None
        and k._stream is None
        and 4 not in ka._SELFTEST_VERDICT
    )


def test_create_from_env_gating(monkeypatch):
    with patch.object(ka.GpuKeepalive, "__init__", return_value=None) as init:
        for raw in (None, "0", "true"):
            if raw is None:
                monkeypatch.delenv(ka.KEEPALIVE_ENV_VAR_NAME, raising=False)
            else:
                monkeypatch.setenv(ka.KEEPALIVE_ENV_VAR_NAME, raw)
            assert ka.GpuKeepalive.create_from_env(0) is None
        init.assert_not_called()  # disabled: the constructor is never reached
        monkeypatch.setenv(ka.KEEPALIVE_ENV_VAR_NAME, "1")
        assert ka.GpuKeepalive.create_from_env(3) is not None
        assert init.call_count == 1 and init.call_args.args[-1] == 3
    with patch.object(ka.GpuKeepalive, "__init__", side_effect=RuntimeError("bad device")):
        assert ka.GpuKeepalive.create_from_env(0) is None  # never fails executor init
    # Construction touches no CUDA state: the device side waits for the first tick.
    with patch.object(ka.torch.cuda, "Stream", Mock(side_effect=AssertionError("CUDA touched"))):
        k = ka.GpuKeepalive.create_from_env(0)
    assert k is not None and not k._initialized and k.mode is None


# ------------------------------------------------------------ gate wiring


def test_fill_gate_wiring():
    """A closed gate ticks the keepalive and still sleeps as before; an opening gate
    drains it; an already-open gate and warmup touch neither. The constructor must
    build it from the environment and the loop cleanup must close it."""
    from tensorrt_llm._torch.pyexecutor import py_executor as pe

    def gate(*, can_forward=False, complete=False, progress=False, warmup=False):
        class Stub:
            _check_benchmark_disagg_gate = pe.PyExecutor._check_benchmark_disagg_gate

        stub = Stub()
        stub.is_warmup = warmup
        stub._gpu_keepalive = Mock()
        stub._disagg_gen_transfer_made_progress = False
        stub._benchmark_transfer_progress_global = progress
        stub._is_benchmark_disagg_fill_complete = lambda batch, made_progress: complete
        stub._fail_if_fill_gate_stalled = Mock()
        stub._benchmark_fill_phase_active = True
        stub._fill_admit_cap = 4
        stub._benchmark_fill_stall_since = 1.0
        stub._benchmark_completed_gen_transfer_ids = {7}
        order = []
        stub._gpu_keepalive.tick.side_effect = lambda: order.append("tick")
        with patch.object(pe.time, "sleep", side_effect=lambda s: order.append("sleep")):
            result = stub._check_benchmark_disagg_gate(Mock(), can_forward)
        return stub._gpu_keepalive, result, order

    k, result, order = gate()  # closed, no transfer progress: tick first, then the usual sleep
    assert result == (False, True) and order == ["tick", "sleep"]
    k.drain.assert_not_called()
    k, result, order = gate(progress=True)  # closed, progress: tick, no sleep (as before)
    assert result == (False, True) and order == ["tick"]
    k, result, order = gate(complete=True)  # opens: drain before the first forward
    assert result == (True, False) and order == []
    k.drain.assert_called_once()
    for kw in ({"can_forward": True}, {"warmup": True}):  # gate bypassed: nothing at all
        k, result, order = gate(**kw)
        assert result[1] is False and order == []
        k.drain.assert_not_called()

    # Exactly one `self._gpu_keepalive = GpuKeepalive.create_from_env(self.device_id)` in __init__.
    tree = ast.parse(inspect.getsource(pe))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PyExecutor")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    constructions = [
        n
        for n in ast.walk(init)
        if isinstance(n, ast.Assign)
        and ast.unparse(n.targets[0]) == "self._gpu_keepalive"
        and ast.unparse(n.value) == "GpuKeepalive.create_from_env(self.device_id)"
    ]
    assert len(constructions) == 1

    # Loop cleanup (the loop may exit while the gate is still closed): waiters are
    # notified first, then the keepalive is closed, then the PP handles are awaited.
    class CleanupStub:
        _executor_loop_cleanup = pe.PyExecutor._executor_loop_cleanup

        def __init__(self, keepalive):
            self.events = []
            self.response_cv = threading.Condition(threading.Lock())
            self.response_cv.notify_all = lambda: self.events.append("notify_all")
            self.is_shutdown = False
            self.shutdown_event = threading.Event()
            self.num_micro_batches = 1
            self.send_handles = self.send_schedule_handles = (
                self.send_expected_batch_num_handles
            ) = {}
            self._gpu_keepalive = keepalive
            if keepalive is not None:
                keepalive.close.side_effect = lambda: self.events.append("close")

        def wait_on_pp_send_handles(self, handles, idx):
            self.events.append("wait_pp")

    stub = CleanupStub(Mock())
    stub._executor_loop_cleanup()
    assert stub.is_shutdown and stub.shutdown_event.is_set()
    assert stub.events == ["notify_all", "close", "wait_pp", "wait_pp", "wait_pp"]
    plain = CleanupStub(None)
    plain._executor_loop_cleanup()  # without a keepalive: unchanged
    assert plain.events == ["notify_all", "wait_pp", "wait_pp", "wait_pp"]
