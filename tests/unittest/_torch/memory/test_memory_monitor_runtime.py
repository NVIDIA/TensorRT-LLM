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
"""Reporting while serving: periodic, on request, and from other threads,
plus startup attribution of memory that worker threads allocate."""

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.memory.monitor import tagging, triggers
from tensorrt_llm.logger import logger

MB = 1 << 20
needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(autouse=True)
def _armed_cleanly(monkeypatch):
    monkeypatch.setenv("TLLM_MEM_MONITOR", "1")
    monkeypatch.setenv("TLLM_MEM_MONITOR_OOM_INTERVAL", "0")
    triggers.uninstall_oom_reporter()
    yield
    triggers.uninstall_oom_reporter()


@pytest.fixture
def reports(monkeypatch):
    """Record OOM reports instead of building them."""
    seen = []

    def fake_report_now(tag, rank=None, **kw):
        seen.append((tag, kw.get("headline", "")))
        # A real report logs; that must not loop back through the filter.
        logger.error("memory monitor report mentioning out of memory")
        return "report"

    monkeypatch.setattr(triggers, "report_now", fake_report_now)
    monkeypatch.setattr(triggers, "_attached", True)  # leave torch's observer alone
    return seen


class TestPeriodic:
    @needs_gpu
    def test_reports_on_its_interval_and_stops(self, monkeypatch, tmp_path):
        monkeypatch.setenv(triggers.PERIODIC_INTERVAL_VAR, "0.05")
        monkeypatch.setattr(triggers, "MIN_PERIODIC_INTERVAL_S", 0.01)
        monkeypatch.setattr(triggers, "_attached", True)
        monkeypatch.setenv(triggers.DIR_VAR, str(tmp_path))
        assert triggers.install_oom_reporter()
        deadline = time.monotonic() + 30
        while not list(tmp_path.glob("*_periodic.txt")) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert list(tmp_path.glob("*_periodic.txt"))
        triggers.uninstall_oom_reporter()
        assert not any(t.name == "memory-monitor-periodic" for t in threading.enumerate())


@pytest.mark.cpu_only
def test_errors_from_other_threads(reports, monkeypatch):
    def periodic_running():
        return any(t.name == "memory-monitor-periodic" for t in threading.enumerate())

    logger.error("CUDA out of memory")
    assert reports == [], "nothing is reported before arming"

    previous = threading.excepthook
    monkeypatch.delenv(triggers.PERIODIC_INTERVAL_VAR, raising=False)
    triggers.install_oom_reporter()
    assert not periodic_running(), "periodic reports are off unless an interval is set"

    # How the KV transfer workers report a failed delivery.
    logger.error("_process_task_queue[0]: unhandled exception: CUDA error: out of memory")
    assert len(reports) == 1 and reports[0][0] == "out_of_memory", "a logged OOM"

    level = logger.level
    logger.set_level("info")  # let the warning reach the filter
    try:
        logger.error("request 7 failed: invalid token id")
        # Runtime autotuning skips tactics this way; not a failure.
        logger.warning("autotuner: tactic 3 ran out of memory, skipping")
    finally:
        logger.set_level(level)
    assert len(reports) == 1, "other errors and warnings are ignored"

    def run():
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    worker = threading.Thread(target=run)
    worker.start()
    worker.join()
    assert len(reports) == 2, "a thread dying of an OOM"

    triggers.uninstall_oom_reporter()
    assert threading.excepthook is previous, "the hooks are removed with the reporter"
    logger.error("CUDA out of memory")
    assert len(reports) == 2


class TestOnRequest:
    @needs_gpu
    def test_worker_reports_even_with_the_monitor_off(self, monkeypatch, tmp_path):
        from tensorrt_llm.executor.base_worker import BaseWorker

        monkeypatch.delenv("TLLM_MEM_MONITOR", raising=False)
        monkeypatch.setenv(triggers.DIR_VAR, str(tmp_path))
        worker = SimpleNamespace(engine=SimpleNamespace(device_id=torch.cuda.current_device()))
        (text,) = BaseWorker.get_memory_report(worker)
        assert "This process uses" in text
        assert not list(tmp_path.iterdir()), "on-demand reports are not written to files"


@pytest.mark.cpu_only
def test_every_rank_has_an_entry(monkeypatch):
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
    from tensorrt_llm.executor.base_worker import BaseWorker

    # Under MPI without full mode only rank 0 answers; the rest are named.
    engine = SimpleNamespace(
        dist=SimpleNamespace(rank=0, world_size=3), _disable_mpi=False, _memory_report_ranks=None
    )
    engine.gather_memory_reports = lambda: PyExecutor.gather_memory_reports(engine)
    monkeypatch.setattr(triggers, "report_now", lambda **kw: None)
    reports = BaseWorker.get_memory_report(SimpleNamespace(engine=engine))
    assert reports[0] == "rank 0: the report could not be built; see this rank's log"
    assert [r.split(":")[0] for r in reports[1:]] == ["rank 1", "rank 2"]
    assert "TLLM_MEM_MONITOR=1" in reports[1]

    # Ray workers each answer for themselves; nothing to add.
    engine._disable_mpi = True
    assert PyExecutor.gather_memory_reports(engine) == []


@needs_gpu
def test_real_out_of_memory_is_reported_once_it_escapes(monkeypatch, tmp_path):
    """A real PyTorch OOM through the armed observer, in the default mode."""
    monkeypatch.delenv("TLLM_MEM_MONITOR", raising=False)
    monkeypatch.setenv(triggers.DIR_VAR, str(tmp_path))
    assert triggers.install_oom_reporter(rank=0)
    try:
        torch.empty(1 << 50, dtype=torch.uint8, device="cuda")
    except torch.OutOfMemoryError as exc:
        error = f"{type(exc).__name__}: {exc}"
    # Caught by its caller: nothing is reported.
    assert not list(tmp_path.iterdir()), "a caught OOM"
    # What the executor's error handler passes on once the error escapes.
    triggers.notify_error(error)
    (path,) = tmp_path.glob("*_out_of_memory.txt")
    text = path.read_text()
    assert "PyTorch could not allocate 1048576.00 GiB" in text
    assert "This process uses" in text


@needs_gpu
def test_worker_thread_allocation_is_named_after_the_running_step(monkeypatch):
    if not tagging.start_tagging_window():
        pytest.skip("tagging unsupported here")
    held = {}
    try:
        with tagging.tag("model"):
            worker = threading.Thread(
                target=lambda: held.setdefault(
                    "x", torch.empty(64 * MB, dtype=torch.uint8, device="cuda")
                )
            )
            worker.start()
            worker.join()
        after = torch.empty(16 * MB, dtype=torch.uint8, device="cuda")
        frozen = tagging.freeze_tag_map()
    finally:
        tagging.stop_tagging_window()
    by_tag = tagging.live_bytes_by_tag(frozen)
    # The thread carried no tag of its own; the step's window names it.
    assert by_tag.get("model", 0) >= 64 * MB
    # Outside every window stays untagged.
    assert by_tag.get(tagging.UNTAGGED, 0) >= 16 * MB
    held.clear()
    del after


@pytest.mark.cpu_only
def test_llm_returns_each_workers_report():
    from tensorrt_llm import LLM
    from tensorrt_llm.executor.proxy import _check_collective_rpc_guard

    llm = object.__new__(LLM)
    # Rank 0's worker answers for every rank under MPI; Ray workers each
    # answer for themselves. Either way one flat list comes back.
    llm._collective_rpc = MagicMock(return_value=[["rank 0", "rank 1"], None, ["rank 2"]])
    assert llm.get_memory_report() == ["rank 0", "rank 1", "rank 2"]
    llm._collective_rpc.assert_called_once_with("get_memory_report")
    # Allowed on a multi-rank MPI executor.
    _check_collective_rpc_guard(2, None, None, method="get_memory_report")


@pytest.mark.cpu_only
def test_nested_windows_resolve_to_the_innermost():
    lookup = tagging._window_lookup([(0, 100, "outer"), (40, 60, "inner")])
    assert [lookup(t) for t in (10, 50, 70, 150)] == ["outer", "inner", "outer", None]


class _FakeComm:
    """Point-to-point messages between in-process ranks, mpi4py-shaped."""

    def __init__(self, rank, mailbox, lock):
        self.rank, self._box, self._lock = rank, mailbox, lock

    def send(self, obj, dest, tag):
        with self._lock:
            self._box.setdefault((self.rank, dest, tag), []).append(obj)

    def Iprobe(self, source, tag):
        with self._lock:
            return bool(self._box.get((source, self.rank, tag)))

    def recv(self, source, tag):
        with self._lock:
            return self._box[(source, self.rank, tag)].pop(0)

    def isend(self, obj, dest, tag):
        self.send(obj, dest, tag)
        return SimpleNamespace(Test=lambda: True)


def _ranks(monkeypatch, world, listening):
    from tensorrt_llm._torch.memory.monitor import ranks

    monkeypatch.setattr(ranks, "_POLL_S", 0.005)
    monkeypatch.setattr(ranks, "_SEND_POLL_S", 0.005)
    monkeypatch.setattr(
        triggers, "report_now", lambda tag, rank=None, **_: f"report of rank {rank}"
    )
    box, lock = {}, threading.Lock()
    reporters = [
        ranks.RankReporter(_FakeComm(r, box, lock), r, world, None)
        for r in range(world)
        if r == 0 or r in listening
    ]
    return reporters, box, lock


@pytest.mark.cpu_only
def test_rank_zero_gathers_every_rank(monkeypatch):
    from tensorrt_llm._torch.memory.monitor import ranks

    reporters, _, _ = _ranks(monkeypatch, 3, {1, 2})
    try:
        got = reporters[0].gather(timeout=10)
        assert got == ["report of rank 1", "report of rank 2"], "in rank order"
    finally:
        for r in reporters:
            r.stop()

    reporters, _, _ = _ranks(monkeypatch, 3, {1})
    try:
        got = reporters[0].gather(timeout=0.3)
        assert got == ["report of rank 1", "rank 2: no report within 0.3 s"], "a silent rank"
    finally:
        for r in reporters:
            r.stop()

    reporters, box, lock = _ranks(monkeypatch, 2, set())
    with lock:  # rank 1's answer to request 0, which timed out long ago
        box.setdefault((1, 0, ranks._REPLY_TAG), []).append((0, "stale"))
    got = reporters[0].gather(timeout=0.2)
    assert got == ["rank 1: no report within 0.2 s"], "a late answer is dropped"


@pytest.mark.cpu_only
def test_a_reply_nobody_collects_does_not_hold_up_shutdown(monkeypatch):
    from tensorrt_llm._torch.memory.monitor import ranks

    warnings = []
    monkeypatch.setattr(ranks.logger, "warning", warnings.append)
    reporters, box, lock = _ranks(monkeypatch, 2, {1})
    listener = reporters[1]
    # A report above MPI's eager limit completes only when rank 0 receives it.
    sent = threading.Event()
    listener._comm.isend = lambda obj, dest, tag: sent.set() or SimpleNamespace(Test=lambda: False)
    with lock:
        box.setdefault((0, 1, ranks._REQUEST_TAG), []).append(1)
    assert sent.wait(5), "the listener answered"
    started = time.monotonic()
    listener.stop()
    assert time.monotonic() - started < 1.0, "stop() returned while the reply was pending"
    assert not listener._thread.is_alive()
    assert any("before rank 0 collected" in w for w in warnings)

    # Rank 0 drains replies that arrive after their gather gave up.
    with lock:
        box.setdefault((1, 0, ranks._REPLY_TAG), []).append((7, "late"))
    reporters[0].stop()
    assert not box[(1, 0, ranks._REPLY_TAG)], "drained when rank 0 stops"


@pytest.mark.cpu_only
def test_bounded_receive_is_shared_with_the_sleep_wakeup_acks(monkeypatch):
    from tensorrt_llm._torch.pyexecutor import mpi_polling
    from tensorrt_llm._torch.pyexecutor import py_executor as pe

    box, lock = {}, threading.Lock()
    rank0, rank1 = _FakeComm(0, box, lock), _FakeComm(1, box, lock)

    # Rejected messages are received and dropped; the first kept one returns.
    for message in ("stale", "late", "current"):
        rank1.send(message, dest=0, tag=7)
    assert mpi_polling.try_recv(rank0, 1, 7, accept=lambda m: m == "current") == (
        True,
        "current",
    ), "stale messages before the current one are dropped"
    assert mpi_polling.try_recv(rank0, 1, 7) == (False, None), "nothing left"

    # A deadline already passed raises, naming what was awaited.
    with pytest.raises(TimeoutError, match="timed out waiting for report from rank 1"):
        mpi_polling.recv_until(rank0, 1, 7, time.monotonic(), poll_interval_s=0.001, what="report")

    # The lower-case object API is used when present.
    lower = SimpleNamespace(iprobe=lambda source, tag: True)
    assert mpi_polling.message_ready(lower, 1, 7), "iprobe is consulted"
    with pytest.raises(RuntimeError):
        mpi_polling.message_ready(object(), 1, 7)

    # Sleep/wakeup ACKs for another operation or phase are dropped; the
    # matching one is returned and the timeout text is unchanged.
    warnings = []
    monkeypatch.setattr(pe.logger, "warning", warnings.append)
    ack = pe._SleepWakeupTag.ACK
    rank1.send({"op_id": "old", "phase": "sleep"}, dest=0, tag=ack)
    rank1.send({"op_id": "new", "phase": "prepare"}, dest=0, tag=ack)
    rank1.send({"op_id": "new", "phase": "sleep", "status": "ok"}, dest=0, tag=ack)
    got = pe._recv_sleep_wakeup_ack_until(
        rank0, 1, time.monotonic() + 5, expected_op_id="new", expected_phase="sleep"
    )
    assert got["status"] == "ok", "the matching ACK is returned"
    assert len(warnings) == 2 and all("Ignoring stale sleep/wakeup ACK" in w for w in warnings)
    with pytest.raises(TimeoutError, match="timed out waiting for sleep/wakeup ACK from rank 1"):
        pe._recv_sleep_wakeup_ack(rank0, 1, 0.0)
