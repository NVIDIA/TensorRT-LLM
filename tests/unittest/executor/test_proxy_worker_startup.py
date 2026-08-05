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
"""Startup handshake between the executor proxy and its MPI workers.

Every proxy test here drives the real
``GenerationExecutorProxy._start_executor_workers``; only the MPI session and
the init status queue are faked, so no GPU (and no MPI spawn) is needed.  The
worker-side tests drive the real
``tensorrt_llm.executor.worker._worker_init_stall_watchdog`` /
``_arm_worker_init_stall_watchdog``.
"""

import ast
import pathlib
import queue
import sys
import threading
import time
import types
from concurrent.futures import Future

import pytest

from tensorrt_llm._utils import print_all_stacks
from tensorrt_llm.executor import worker as worker_module
from tensorrt_llm.executor.proxy import GenerationExecutorProxy
from tensorrt_llm.executor.utils import WORKER_INIT_STALL_WARN_ENV, worker_init_stall_warn_sec


class RecordingLogger:
    """Minimal stand-in for ``tensorrt_llm.logger`` that keeps the messages."""

    def __init__(self):
        self.warnings = []
        self.errors = []

    def warning(self, message, *args, **kwargs):
        self.warnings.append(str(message))

    def error(self, message, *args, **kwargs):
        self.errors.append(str(message))

    def info(self, message, *args, **kwargs):
        pass

    def debug(self, message, *args, **kwargs):
        pass


class FakeMpiSession:
    def __init__(self, futures):
        self.futures = futures
        self.shutdown_abort_reasons = []
        self.submitted_kwargs = None

    def submit(self, *args, **kwargs):
        self.submitted_kwargs = kwargs
        return self.futures

    def shutdown_abort(self, *, reason=None, grace=60):
        del grace
        self.shutdown_abort_reasons.append(reason)


class FakeWorkerInitStatusQueue:
    def __init__(self, messages=None, on_poll=None):
        self.messages = queue.Queue()
        for message in messages or []:
            self.messages.put(message)
        self.on_poll = on_poll
        self.acks = []
        self.poll_count = 0

    def poll(self, timeout):
        del timeout
        self.poll_count += 1
        if self.on_poll is not None:
            self.on_poll(self.poll_count)
        return not self.messages.empty()

    def get(self):
        return self.messages.get_nowait()

    def put(self, message):
        self.acks.append(message)


def _make_proxy(monkeypatch, *, futures, init_messages=None, on_poll=None, owns_mpi_session=True):
    proxy = object.__new__(GenerationExecutorProxy)
    proxy._error_queue = queue.Queue()
    proxy._fatal_error = None
    proxy.doing_shutdown = False
    proxy.worker_cls = object
    proxy.workers_started = False
    proxy._owns_mpi_session = owns_mpi_session
    proxy.mpi_session = FakeMpiSession(futures)
    proxy.worker_init_status_queue = FakeWorkerInitStatusQueue(init_messages, on_poll)
    fake_modeling_auto = types.SimpleNamespace(MODEL_CLASS_MAPPING={})
    monkeypatch.setitem(sys.modules, "tensorrt_llm._torch.models.modeling_auto", fake_modeling_auto)
    monkeypatch.setattr("tensorrt_llm.executor.proxy.torch.cuda.Stream", lambda: None)
    monkeypatch.setattr("tensorrt_llm.executor.proxy.enable_llm_tracer", lambda: False)
    # The startup knobs are environment driven; never inherit them from the
    # environment the test suite happens to run in.
    monkeypatch.delenv(WORKER_INIT_STALL_WARN_ENV, raising=False)
    return proxy


def test_worker_ready_signal_exits_startup_loop(monkeypatch):
    future = Future()
    proxy = _make_proxy(
        monkeypatch,
        futures=[future],
        init_messages=[(GenerationExecutorProxy.READY_SIGNAL, None)],
    )

    proxy._start_executor_workers({"tokenizer": object(), "keep": "value"})

    assert proxy.workers_started is True
    assert proxy.worker_init_status_queue.acks == ["ACK"]
    assert proxy.mpi_session.shutdown_abort_reasons == []
    assert "tokenizer" not in proxy.mpi_session.submitted_kwargs
    assert proxy.mpi_session.submitted_kwargs["keep"] == "value"


def test_worker_init_error_aborts_mpi_session(monkeypatch):
    future = Future()
    init_error = RuntimeError("rank 1 failed during init")
    proxy = _make_proxy(
        monkeypatch,
        futures=[future],
        init_messages=[(init_error, "rank 1 traceback")],
    )

    with pytest.raises(RuntimeError, match="Executor worker returned error"):
        proxy._start_executor_workers({})

    assert proxy.worker_init_status_queue.acks == ["ACK"]
    assert proxy.mpi_session.shutdown_abort_reasons == [init_error]


def test_worker_future_done_before_ready_fails_fast(monkeypatch):
    future = Future()
    future.set_exception(RuntimeError("rank 1 exited"))
    proxy = _make_proxy(monkeypatch, futures=[future])

    with pytest.raises(RuntimeError, match="Executor worker died during initialization"):
        proxy._start_executor_workers({})


def test_alive_worker_without_ready_signal_keeps_waiting_by_default(monkeypatch):
    """No bound is imposed unless one is asked for.

    Initialization that is slow but healthy (a very large checkpoint loading
    from a cold mount) must not be killed, so the default behavior stays
    "wait"; only the reporting is new.
    """
    future = Future()
    polled_repeatedly = threading.Event()

    def on_poll(poll_count):
        if poll_count >= 5:
            polled_repeatedly.set()

    proxy = _make_proxy(monkeypatch, futures=[future], on_poll=on_poll)
    result = {}

    def start_workers():
        try:
            proxy._start_executor_workers({})
        except BaseException as exc:
            result["exception"] = exc

    startup_thread = threading.Thread(target=start_workers, daemon=True)
    startup_thread.start()

    try:
        assert polled_repeatedly.wait(timeout=2)
        assert startup_thread.is_alive()
        assert proxy.mpi_session.shutdown_abort_reasons == []
    finally:
        if not future.done():
            future.set_exception(RuntimeError("rank 1 eventually exited"))

    startup_thread.join(timeout=2)

    assert not startup_thread.is_alive()
    assert isinstance(result["exception"], RuntimeError)
    assert str(result["exception"]) == ("Executor worker died during initialization")


def test_stalled_startup_is_reported_while_waiting(monkeypatch):
    """A wedged-but-alive startup is reported repeatedly, not silently awaited.

    Nothing terminates the wait, so the test releases the loop the same way
    reality would -- by letting a rank die -- once it has seen enough reports.
    """
    records = RecordingLogger()
    monkeypatch.setattr("tensorrt_llm.executor.proxy.logger", records)
    future = Future()
    proxy = _make_proxy(monkeypatch, futures=[future])
    monkeypatch.setenv(WORKER_INIT_STALL_WARN_ENV, "0.02")

    def start_workers():
        try:
            proxy._start_executor_workers({})
        except BaseException:  # noqa: BLE001 - the release path, not the subject
            pass

    startup_thread = threading.Thread(target=start_workers, daemon=True)
    startup_thread.start()
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if len([m for m in records.warnings if "has not completed" in m]) >= 2:
                break
            time.sleep(0.01)
    finally:
        future.set_exception(RuntimeError("rank 1 eventually exited"))
    startup_thread.join(timeout=5)
    assert not startup_thread.is_alive()

    stall_reports = [
        message
        for message in records.warnings
        if "Executor worker initialization has not completed" in message
    ]
    assert len(stall_reports) >= 2, records.warnings
    # The report must attribute the stall: no rank died, and it must point at
    # the per-rank stacks that carry the "which rank, and where" detail.
    assert "1/1 worker task(s) are still running" in stall_reports[0]
    assert "has not finished initialization" in stall_reports[0]
    # A stall must never be dressed up as the (separately handled) crash.
    assert "died during initialization" not in stall_reports[0]


def test_ready_signal_is_not_delayed_by_stall_reporting(monkeypatch):
    """Reporting must never get between the proxy and a queued ready signal."""
    proxy = _make_proxy(
        monkeypatch,
        futures=[Future()],
        init_messages=[(GenerationExecutorProxy.READY_SIGNAL, None)],
    )
    monkeypatch.setenv(WORKER_INIT_STALL_WARN_ENV, "0.01")

    proxy._start_executor_workers({})

    assert proxy.workers_started is True
    assert proxy.mpi_session.shutdown_abort_reasons == []


def test_stall_report_without_worker_futures_claims_no_liveness(monkeypatch):
    """``RemoteMpiCommSessionClient.submit()`` returns ``[]``.

    An empty ``mpi_futures`` is absence of evidence, not evidence of absence:
    counting an empty list yields "0/0 ... so no rank has exited", a confident
    statement made with zero visibility.  ``pre_shutdown()`` documents the same
    empty-list trap.
    """
    proxy = _make_proxy(monkeypatch, futures=[])
    proxy.mpi_futures = proxy.mpi_session.submit()

    report = proxy._worker_init_stall_report(900.0)

    assert "0/0" not in report, report
    assert "no rank has exited" not in report, report
    assert "worker task(s) are still running" not in report, report
    # It must say the liveness question is unanswerable here, and name the
    # channel that can answer it.
    assert "cannot be told from here" in report, report
    assert "check_worker_error()" in report, report
    # The attribution the report exists for is unaffected.
    assert "has not finished initialization" in report, report


def test_stall_report_with_worker_futures_still_states_liveness(monkeypatch):
    """The visible case must keep saying what it can see."""
    alive, dead = Future(), Future()
    dead.set_exception(RuntimeError("rank 1 exited"))
    proxy = _make_proxy(monkeypatch, futures=[alive, dead])
    proxy.mpi_futures = proxy.mpi_session.submit()

    report = proxy._worker_init_stall_report(900.0)

    assert "1/2 worker task(s) are still running" in report, report
    assert "cannot be told from here" not in report, report


def test_stalled_startup_without_worker_futures_is_reported(monkeypatch):
    """The same, driven through the real startup loop rather than the helper."""
    records = RecordingLogger()
    monkeypatch.setattr("tensorrt_llm.executor.proxy.logger", records)

    def on_poll(poll_count):
        # Nothing can die here (there are no futures), so release the loop the
        # only other way: let the ready signal finally arrive.
        if any("has not completed" in m for m in records.warnings):
            proxy.worker_init_status_queue.messages.put(
                (GenerationExecutorProxy.READY_SIGNAL, None)
            )

    proxy = _make_proxy(monkeypatch, futures=[], on_poll=on_poll)
    monkeypatch.setenv(WORKER_INIT_STALL_WARN_ENV, "0.02")

    proxy._start_executor_workers({})

    stall_reports = [
        m for m in records.warnings if "Executor worker initialization has not completed" in m
    ]
    assert stall_reports, records.warnings
    assert "0/0" not in stall_reports[0], stall_reports[0]
    assert "no rank has exited" not in stall_reports[0], stall_reports[0]
    assert "cannot be told from here" in stall_reports[0], stall_reports[0]


def test_stall_warn_knob_defaults_and_blank_values(monkeypatch):
    monkeypatch.delenv(WORKER_INIT_STALL_WARN_ENV, raising=False)
    assert worker_init_stall_warn_sec() > 0.0  # reporting on by default

    # Unset and blank are the only inputs that fall back to the default.
    monkeypatch.setenv(WORKER_INIT_STALL_WARN_ENV, "")
    assert worker_init_stall_warn_sec() > 0.0
    monkeypatch.setenv(WORKER_INIT_STALL_WARN_ENV, "   ")
    assert worker_init_stall_warn_sec() > 0.0

    monkeypatch.setenv(WORKER_INIT_STALL_WARN_ENV, "30")
    assert worker_init_stall_warn_sec() == 30.0


@pytest.mark.parametrize("raw", ["nan", "NaN", "-nan", "inf", "-inf", "1e400"])
def test_stall_warn_knob_rejects_non_finite_values(monkeypatch, raw):
    """``float()`` accepts these; the watchdog cannot survive them.

    ``nan`` defeats the ``period <= 0`` disable check (every comparison with
    it is False) and then makes ``Event.wait(nan)`` return immediately rather
    than sleep, so the watchdog becomes a hot loop dumping every thread's
    stack on every rank.  ``Event.wait(inf)`` raises ``OverflowError`` inside
    the watchdog thread instead, silently killing the reporting.
    """
    monkeypatch.setenv(WORKER_INIT_STALL_WARN_ENV, raw)

    with pytest.raises(ValueError, match="finite"):
        worker_init_stall_warn_sec()


def test_stall_warn_knob_rejects_unparsable_values(monkeypatch):
    """A misconfiguration must not be swallowed at the one place it matters.

    This PR's premise is that startup is where information gets lost; falling
    back to the default here would hide the typo behind exactly the silence
    the reporting exists to remove.
    """
    monkeypatch.setenv(WORKER_INIT_STALL_WARN_ENV, "ten minutes")

    with pytest.raises(ValueError, match=WORKER_INIT_STALL_WARN_ENV):
        worker_init_stall_warn_sec()


def test_non_finite_knob_never_arms_a_watchdog(monkeypatch):
    """The consequence, pinned on the arming path the worker ranks take."""
    dumps = []
    monkeypatch.setattr(
        "tensorrt_llm.executor.worker.print_all_stacks",
        lambda **kwargs: dumps.append(kwargs.get("log")),
    )
    monkeypatch.setattr("tensorrt_llm.executor.worker.logger", RecordingLogger())
    monkeypatch.setenv(WORKER_INIT_STALL_WARN_ENV, "nan")
    init_done = threading.Event()
    armed = []

    try:
        with pytest.raises(ValueError):
            armed.append(worker_module._arm_worker_init_stall_watchdog(init_done))
    finally:
        # If the guard ever regresses, a watchdog is now hot-looping on
        # ``Event.wait(nan)`` (which returns immediately rather than sleeping);
        # retire it here so the failure is a failure and not a flooded run.
        init_done.set()
        for watchdog in armed:
            if watchdog is not None:
                watchdog.join(timeout=5)

    assert not dumps, "a nan period armed a watchdog that dumped stacks"


def test_non_finite_knob_never_reaches_the_startup_loop(monkeypatch):
    """...and on the proxy path, before any rank is spawned.

    The future is pre-failed so that a regression here fails instead of
    hanging: without the guard the loop would run with ``next_warn_time``
    permanently ``None`` (``nan > 0`` is False) and never report or exit.
    """
    dead = Future()
    dead.set_exception(RuntimeError("rank 1 exited"))
    proxy = _make_proxy(monkeypatch, futures=[dead])
    monkeypatch.setenv(WORKER_INIT_STALL_WARN_ENV, "nan")

    with pytest.raises(ValueError, match="finite"):
        proxy._start_executor_workers({})

    # The knob is read before anything is spawned, so nothing was left behind.
    assert proxy.workers_started is False
    assert proxy.mpi_session.submitted_kwargs is None


def test_worker_init_watchdog_dumps_this_ranks_stacks(monkeypatch):
    """Drives the real ``worker._worker_init_stall_watchdog``."""
    dumps = []
    records = RecordingLogger()
    monkeypatch.setattr(
        "tensorrt_llm.executor.worker.print_all_stacks",
        lambda **kwargs: dumps.append(kwargs.get("log")),
    )
    monkeypatch.setattr("tensorrt_llm.executor.worker.logger", records)

    init_done = threading.Event()
    watchdog = threading.Thread(
        target=worker_module._worker_init_stall_watchdog, args=(init_done, 0.02), daemon=True
    )
    watchdog.start()
    try:
        deadline = time.monotonic() + 10
        while len(dumps) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        init_done.set()
    watchdog.join(timeout=5)

    assert not watchdog.is_alive()
    assert len(dumps) >= 2
    # A slow-but-healthy startup is not a fault: the report and the stack dump
    # it triggers must both be WARNING, never ERROR.
    assert records.errors == []
    assert any("has not finished initialization" in message for message in records.warnings), (
        records.warnings
    )
    assert dumps == [records.warning] * len(dumps)


def test_worker_init_watchdog_is_silent_once_init_completes(monkeypatch):
    dumps = []
    monkeypatch.setattr(
        "tensorrt_llm.executor.worker.print_all_stacks",
        lambda **kwargs: dumps.append(kwargs.get("log")),
    )
    monkeypatch.setattr("tensorrt_llm.executor.worker.logger", RecordingLogger())

    init_done = threading.Event()
    init_done.set()
    watchdog = threading.Thread(
        target=worker_module._worker_init_stall_watchdog, args=(init_done, 0.02), daemon=True
    )
    watchdog.start()
    watchdog.join(timeout=5)

    assert not watchdog.is_alive()
    assert dumps == []


def test_worker_init_watchdog_is_armed_by_default(monkeypatch):
    monkeypatch.delenv(WORKER_INIT_STALL_WARN_ENV, raising=False)
    init_done = threading.Event()

    watchdog = worker_module._arm_worker_init_stall_watchdog(init_done)
    try:
        assert watchdog is not None
        assert watchdog.daemon
        assert watchdog.is_alive()
    finally:
        init_done.set()
    # Completing initialization retires the watchdog immediately, even though
    # its default period is far longer than this test.
    watchdog.join(timeout=5)
    assert not watchdog.is_alive()


def test_worker_init_watchdog_can_be_disabled(monkeypatch):
    monkeypatch.setenv(WORKER_INIT_STALL_WARN_ENV, "0")

    assert worker_module._arm_worker_init_stall_watchdog(threading.Event()) is None


def test_print_all_stacks_honours_the_log_callable():
    """Drives the real ``tensorrt_llm._utils.print_all_stacks``.

    The init watchdog relies on being able to emit the dump at WARNING; the
    default must stay ERROR for the existing callers.
    """
    emitted = []

    print_all_stacks(log=emitted.append)

    assert emitted
    assert all("stack trace:" in message for message in emitted)


# --- worker_main disarm placement ------------------------------------------
#
# ``worker_main`` needs a live MPI world, so the *placement* of the disarm is
# pinned structurally rather than behaviourally.  This is not decoration: a
# leader that disarms when its constructor returns -- instead of when it has
# delivered the ready signal -- leaves the proxy waiting on a rank that has
# stopped reporting, and every behavioural test still passes.  That regression
# was found on hardware, not here; these tests are what would catch it next
# time.


def _worker_main_ast():
    source = pathlib.Path(worker_module.__file__).read_text()
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == "worker_main":
            return node
    raise AssertionError("worker_main not found in tensorrt_llm.executor.worker")


def _disarm_linenos(scope):
    return sorted(
        node.lineno
        for node in ast.walk(scope)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "set"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "worker_init_done"
    )


def _ready_send_lineno(worker_main):
    for node in ast.walk(worker_main):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "notify_with_retry"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "ready_msg"
        ):
            return node.lineno
    raise AssertionError("ready-signal send not found in worker_main")


def _subordinate_only_disarm_linenos(worker_main):
    """Disarms guarded by ``if not is_leader:``."""
    linenos = []
    for node in ast.walk(worker_main):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.UnaryOp)
            and isinstance(test.op, ast.Not)
            and isinstance(test.operand, ast.Name)
            and test.operand.id == "is_leader"
        ):
            for statement in node.body:
                linenos.extend(_disarm_linenos(statement))
    return sorted(linenos)


def test_leader_stays_armed_until_the_ready_signal_is_sent():
    worker_main = _worker_main_ast()
    ready_send = _ready_send_lineno(worker_main)

    disarms = _disarm_linenos(worker_main)
    assert disarms, "worker_main never disarms the init watchdog"

    after_ready = [lineno for lineno in disarms if lineno > ready_send]
    assert after_ready, (
        "no worker_init_done.set() after the ready signal is sent: the leader "
        "would stop reporting while the proxy is still waiting for it"
    )


def test_no_unguarded_disarm_before_the_ready_signal():
    """The regression guard: a bare disarm after construction re-breaks this."""
    worker_main = _worker_main_ast()
    ready_send = _ready_send_lineno(worker_main)

    subordinate_only = set(_subordinate_only_disarm_linenos(worker_main))
    in_failure_path = {
        lineno
        for handler in (
            node for node in ast.walk(worker_main) if isinstance(node, ast.ExceptHandler)
        )
        for lineno in _disarm_linenos(handler)
    }

    unguarded = [
        lineno
        for lineno in _disarm_linenos(worker_main)
        if lineno < ready_send and lineno not in subordinate_only and lineno not in in_failure_path
    ]
    assert not unguarded, (
        f"worker_init_done.set() at line(s) {unguarded} runs on the leader "
        "before it has delivered the ready signal; the leader must stay armed "
        "until then (subordinates disarm under 'if not is_leader:', and the "
        "construction-failure path disarms inside its except handler)"
    )


def test_subordinate_disarms_after_construction():
    worker_main = _worker_main_ast()
    ready_send = _ready_send_lineno(worker_main)

    subordinate_only = _subordinate_only_disarm_linenos(worker_main)
    assert subordinate_only, (
        "no 'if not is_leader:' disarm: a subordinate blocks in "
        "block_subordinates() forever and would report for the life of the job"
    )
    assert all(lineno < ready_send for lineno in subordinate_only)


# ---------------------------------------------------------------------------
# The ready signal can fail to arrive. That is the one case where the worker
# must NOT disarm its stall watchdog: the proxy is still blocked, looking for
# a signal that will never come, and the per-rank stall reports are the only
# remaining evidence of it.
# ---------------------------------------------------------------------------
def _ready_delivery_block(src: str) -> ast.If:
    """The `if ready_delivered:` block inside worker_main, from source.

    Read from the AST rather than executed: reaching this line for real needs
    a constructed worker, MPI ranks and an engine. The property under test is
    a control-flow one -- which branch sets the event -- and that is exactly
    what the AST shows.
    """
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "ready_delivered"
        ):
            return node
    raise AssertionError("no `if ready_delivered:` branch found in worker_main")


def _sets_init_done(body) -> bool:
    for node in body:
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "set"
                and isinstance(sub.func.value, ast.Name)
                and sub.func.value.id == "worker_init_done"
            ):
                return True
    return False


def test_watchdog_is_disarmed_only_when_the_ready_signal_was_delivered():
    src = pathlib.Path(worker_module.__file__).read_text()
    branch = _ready_delivery_block(src)

    assert _sets_init_done(branch.body), (
        "the success path must disarm the watchdog -- otherwise every healthy "
        "startup keeps dumping stacks for the life of the process"
    )
    assert not _sets_init_done(branch.orelse), (
        "the failure path must NOT disarm the watchdog: the proxy is still "
        "waiting for a ready signal that never arrived, and disarming removes "
        "the last thing reporting that"
    )


def test_failed_ready_delivery_is_logged_at_error():
    """A warning is not enough: this leaves the proxy hung."""
    src = pathlib.Path(worker_module.__file__).read_text()
    branch = _ready_delivery_block(src)
    levels = {
        sub.func.attr
        for node in branch.orelse
        for sub in ast.walk(node)
        if isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Attribute)
        and isinstance(sub.func.value, ast.Name)
        and sub.func.value.id == "logger"
    }
    assert "error" in levels, f"expected logger.error on this path, saw {levels}"
