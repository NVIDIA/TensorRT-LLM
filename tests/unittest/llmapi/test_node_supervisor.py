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
"""Tests for the Wide-EP fault-tolerance node supervisor with fake ranks and a fake authority.

The fake authority is a real rendezvous store plus a listener that sends the supervisor the views
a test gives it. The supervisor runs in a thread, or in a process when a test signals or kills it.
"""

import contextlib
import importlib.util
import json
import os
import secrets
import select
import selectors
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from torch.distributed import TCPStore

from tensorrt_llm._torch.distributed import membership
from tensorrt_llm._torch.distributed.membership import (
    LIVENESS_KEY,
    RENDEZVOUS_ENV,
    SHARED_SECRET_ENV,
    MembershipRecord,
    MembershipState,
    NodeReport,
    NodeView,
    SignedStore,
    records,
)
from tensorrt_llm._torch.distributed.membership.liveness import Lines, hello_token
from tensorrt_llm.llmapi import _llmapi_process_guard as process_guard
from tensorrt_llm.logger import logger

# The tests run supervisors, ranks and launchers as processes. A hang fails its test, not the job.
pytestmark = [pytest.mark.cpu_only, pytest.mark.timeout(120)]

_LLMAPI = Path(__file__).parents[3] / "tensorrt_llm" / "llmapi"
_SUPERVISOR = _LLMAPI / "_llmapi_node_supervisor.py"
_spec = importlib.util.spec_from_file_location("_llmapi_node_supervisor", _SUPERVISOR)
supervisor_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(supervisor_module)
NODE, NODES, GPUS = 1, 2, 2  # node 1 runs ranks 2 and 3 of 4
RANKS = (2, 3)
_JOIN_TIMEOUT_S = 10.0
# How long a test waits for a supervisor process, which imports TensorRT-LLM (about 10 s) before it
# starts its ranks or checks its parent and launch variables.
_SUPERVISOR_START_TIMEOUT_S = 90.0
_TIMINGS = dict(_CONNECT_TIMEOUT_S=0.5, _STOP_GRACE_S=2.0, _BACKOFF_S=0.05)


@pytest.fixture(autouse=True)
def _timings(monkeypatch):
    """Shorten the supervisor's timings for the tests."""
    for name, value in _TIMINGS.items():
        monkeypatch.setattr(supervisor_module, name, value)


# A fake rank. It records itself in SUPERVISOR_TEST_ROOT as the rank's n-th process, 0 for the
# original, and ends with status 0 once the fake authority sent SHUTDOWN and wrote "shutdown" there.
# SUPERVISOR_TEST_MODES maps "<rank>:<n>" to a mode, "serves" by default. "records-signals" records
# the first SIGINT or SIGTERM and exits. "starts-helper" records the PID of a helper it starts.
# "drains" ends once SUPERVISOR_TEST_ROOT/drained exists. "crashes", "fails" and "cannot-start" exit
# at once.
_RANK = r"""
import json, os, signal, subprocess, sys, time
from pathlib import Path

rank, world_size, _, replacement = os.environ["TLLM_FAULT_TOLERANCE_RANK"].split(",")
root = Path(os.environ["SUPERVISOR_TEST_ROOT"])
started = [int(path.stem.rsplit("-", 1)[1]) for path in root.glob(f"rank-{rank}-*.json")]
incarnation = max(started, default=0) + 1 if replacement == "1" else 0
modes = dict(item.split("=") for item in os.environ.get("SUPERVISOR_TEST_MODES", "").split())
mode = modes.get(f"{rank}:{incarnation}", "serves")


def publish(name, text):
    # Write a hidden file, then rename it, so the test never reads a partly written record.
    written = root / f".{name}"
    written.write_text(text)
    written.replace(root / name)


if mode == "records-signals":
    def record(signum, frame):
        publish(f"signals-{rank}-{incarnation}", signal.Signals(signum).name)
        sys.exit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, record)
if mode == "ignores-sigterm":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
helper = None
if mode == "starts-helper":
    helper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"]).pid
publish(
    f"rank-{rank}-{incarnation}.json",
    json.dumps(
        {"pid": os.getpid(), "helper": helper, "argv": sys.argv, "environ": dict(os.environ)}
    ),
)
statuses = {"crashes": 7, "fails": 5, "cannot-start": 78}
if mode in statuses:
    sys.exit(statuses[mode])
end = root / ("drained" if mode == "drains" else "shutdown")
deadline = time.monotonic() + 600
while not end.exists() and time.monotonic() < deadline:
    time.sleep(0.05)
"""

# Run the supervisor as its script does, with the test's timings. The "started" file marks the start
# of main(), before its imports of TensorRT-LLM.
_SUPERVISOR_SCRIPT = r"""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("node_supervisor", %(supervisor)r)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for name, value in %(timings)r.items():
    setattr(module, name, value)
Path(%(started)r).touch()
module.main()
"""


def _record(
    state=MembershipState.COMMITTED, active_mask=0b1111, incarnations=(0, 0, 0, 0), generation=1
):
    return MembershipRecord(
        generation=generation, state=state, active_mask=active_mask, incarnations=incarnations
    )


class FakeAuthority:
    """Fake membership authority that node supervisors connect to.

    :meth:`start` opens a rendezvous store that publishes the listener's address. The listener
    accepts node supervisors and sends them the views given to :meth:`push`. Once it sent SHUTDOWN,
    the fake ranks under ``root`` end. One thread, started with the authority before a test runs,
    serves every connection until :meth:`close`, so that a test leaves no thread of it behind.
    """

    def __init__(self, root):
        self._root = root
        self._hello_token = hello_token()
        # Bound without SO_REUSEADDR, so that no other process can bind the port, and listening only
        # once start() hands it to the store, so that connections are refused until then.
        self._reserved = socket.socket()
        self._reserved.bind(("127.0.0.1", 0))
        self.address = f"127.0.0.1:{self._reserved.getsockname()[1]}"
        self.store = None
        self.hellos = []
        # The next hellos to refuse, as the authority refuses a node whose previous connection it
        # still holds.
        self.refusals = 0
        self.closed = threading.Event()  # a supervisor's connection closed
        # The ranks that the last report of a supervisor gave up, or None before any report.
        self.given_up = None
        # The paused mask of each report, in order.
        self.paused_masks = []
        # Cleared to hold the view that answers a supervisor's hello.
        self.answering = threading.Event()
        self.answering.set()
        self._connections = []
        self._lock = threading.Lock()
        self._view = NodeView(record=_record())
        self._listener = socket.create_server(("127.0.0.1", 0))
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def start(self, record=None):
        if record is not None:
            self._view = NodeView(record=record)
        port = self._reserved.getsockname()[1]
        self._reserved.listen()
        self.store = SignedStore(
            TCPStore(
                "127.0.0.1",
                port,
                is_master=True,
                wait_for_workers=False,
                master_listen_fd=self._reserved.detach(),
            )
        )
        self.store.set(LIVENESS_KEY, f"127.0.0.1:{self._listener.getsockname()[1]}".encode())
        return self

    def _serve(self):
        with selectors.DefaultSelector() as selector:
            selector.register(self._listener, selectors.EVENT_READ)
            while True:
                for key, _ in selector.select():
                    if key.fileobj is not self._listener:
                        self._read_reports(selector, key.fileobj, key.data)
                        continue
                    try:
                        sock, _ = self._listener.accept()
                    except OSError:
                        return
                    lines = self._answer_hello(sock)
                    if lines is not None:
                        selector.register(sock, selectors.EVENT_READ, lines)

    def _answer_hello(self, sock):
        """Send the view to a supervisor that sent a hello, and return the connection's framing.

        Return None for a connection that is refused.
        """
        # Nothing follows the hello before the view, so the buffered read takes nothing more.
        token, *announcement = sock.makefile("rb").readline().decode().split() or [None]
        if token != self._hello_token:
            sock.close()
            return None
        if self.refusals:
            self.refusals -= 1
            sock.close()
            return None
        lines = Lines()
        self.hellos.append(announcement)
        self.answering.wait()
        with self._lock:
            self._connections.append((sock, lines))
            sock.sendall(lines.encode(self._view.model_dump_json().encode()))
        return lines

    def _read_reports(self, selector, sock, lines):
        with contextlib.suppress(OSError):
            if data := sock.recv(256):
                for line in lines.decode(data):
                    report = NodeReport.model_validate_json(line)
                    self.given_up = report.given_up
                    self.paused_masks.append(report.paused_mask)
                return
        selector.unregister(sock)
        self.closed.set()

    def push(self, record, stalled_replacements=(), connected_mask=0):
        with self._lock:
            self._view = NodeView(
                record=record,
                stalled_replacements=stalled_replacements,
                connected_mask=connected_mask,
            )
            for sock, lines in self._connections:
                with contextlib.suppress(OSError):
                    sock.sendall(lines.encode(self._view.model_dump_json().encode()))
        if record.state == MembershipState.SHUTDOWN:
            (self._root / "shutdown").touch()

    def close(self):
        """Join the thread and close every socket.

        Only a shutdown, not a close, wakes the thread blocked on the listener. The thread ends
        first, so that a connection it accepted during the close is shut down too.
        """
        self.answering.set()
        self._reserved.close()
        with contextlib.suppress(OSError):
            self._listener.shutdown(socket.SHUT_RDWR)
        self._thread.join(_JOIN_TIMEOUT_S)
        with self._lock:
            sockets = [*(sock for sock, _ in self._connections), self._listener]
        for sock in sockets:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
            sock.close()


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    """Set up a shared secret, an authority that is not started yet, and the fake rank script.

    Yields the authority and the environment the node's launcher would give the supervisor. After
    the test, stops every supervisor still running and joins the threads that connected to the
    authority, so that a test that failed part-way leaves none of them to the next test.
    """
    secret = secrets.token_hex(32)
    monkeypatch.setenv(SHARED_SECRET_ENV, secret)
    authority = FakeAuthority(tmp_path)
    monkeypatch.setenv(RENDEZVOUS_ENV, authority.address)
    connectors = []
    open_store = membership.open_store

    def open_recorded_store(*args):
        connectors.append(threading.current_thread())
        return open_store(*args)

    monkeypatch.setattr(membership, "open_store", open_recorded_store)
    (tmp_path / "rank.py").write_text(_RANK)
    environ = {
        **os.environ,
        "SUPERVISOR_TEST_ROOT": str(tmp_path),
        "OMPI_COMM_WORLD_RANK": "1",
        "PMIX_RANK": "1",
    }
    yield authority, environ
    try:
        while Running.started:
            Running.started.pop().stop()
        for thread in connectors:
            thread.join(_JOIN_TIMEOUT_S)
            assert not thread.is_alive(), f"{thread.name} still connects"
    finally:
        authority.close()


@pytest.fixture
def held_connect(monkeypatch):
    """Hold each connect in its store client, once ``hold`` is set, until ``release`` is set.

    ``finish()`` releases the connects and joins their threads. A test that holds a connect calls
    it before it ends, since the thread leak check runs before the fixtures' teardown.
    """
    connect = SimpleNamespace(
        hold=threading.Event(), held=threading.Event(), release=threading.Event(), threads=[]
    )
    open_store = membership.open_store

    def open_held_store(*args):
        if connect.hold.is_set():
            connect.threads.append(threading.current_thread())
            connect.held.set()
            connect.release.wait(60)
        return open_store(*args)

    def finish():
        connect.release.set()
        for thread in connect.threads:
            thread.join(30)
            assert not thread.is_alive(), f"{thread.name} still connects"

    connect.finish = finish

    monkeypatch.setattr(membership, "open_store", open_held_store)
    yield connect
    finish()


@pytest.fixture
def pause_words():
    """Open the pause words of ranks as the ranks do, and unlink them after the test."""
    paths = []

    def open_words(ranks):
        paths.extend(Path(membership.pause_word_path(rank)) for rank in ranks)
        return [membership.open_pause_word(str(path)) for path in paths[-len(ranks) :]]

    yield open_words
    for path in paths:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()


def _read(tmp_path, rank, incarnation):
    path = tmp_path / f"rank-{rank}-{incarnation}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _until(predicate, timeout_s=30.0):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


def _alive(pid):
    try:
        with open(f"/proc/{pid}/stat") as stat:
            return stat.read().split(")")[-1].split()[0] != "Z"
    # The process may be reaped between the open and the read.
    except (FileNotFoundError, ProcessLookupError):
        return False


class Running:
    """Run a node supervisor in a thread and keep its exit status in :attr:`status`.

    An error that ends the supervisor is kept in :attr:`error`. :attr:`started` holds the
    supervisors that the ``deployment`` fixture stops after the test.
    """

    started = []

    def __init__(self, tmp_path, environ, node=NODE, **options):
        self.supervisor = supervisor_module.NodeSupervisor(
            node,
            NODES,
            GPUS,
            [sys.executable, str(tmp_path / "rank.py")],
            environ,
            **options,
        )
        self.status = None
        self.error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        Running.started.append(self)
        self._thread.start()

    def _run(self):
        try:
            self.status = self.supervisor.run()
        except Exception as error:
            self.error = error

    def wait(self, timeout_s=30.0):
        self._thread.join(timeout_s)
        assert not self._thread.is_alive(), "the supervisor did not end"
        return self.status

    def stop(self):
        """Stop the supervisor and its ranks as a signal does, unless it ended, and wait for it.

        Its ranks get the signal's grace period, and then the stop's.
        """
        if self._thread.is_alive():
            self.supervisor.on_signal(signal.SIGTERM)
        self.wait(2 * supervisor_module._STOP_GRACE_S + _JOIN_TIMEOUT_S)


def _serving(tmp_path, authority, environ, node=NODE, **options):
    """Start a supervisor whose ranks start before the authority, as the original ranks do."""
    running = Running(tmp_path, environ, node=node, **options)
    _until(lambda: all(_read(tmp_path, rank, 0) for rank in range(node * GPUS, (node + 1) * GPUS)))
    authority.start()
    _until(lambda: authority.hellos and running.supervisor._view is not None)
    return running


def _push_and_wait_for_step(running, authority, record, **view):
    """Push a new view and wait until the supervisor has acted on it and on every earlier event.

    The supervisor acts on each view before it waits for the next event. Once it has taken a later
    view, it has acted on this one. ``view`` holds the view's other fields.
    """
    authority.push(record, **view)
    _until(lambda: running.supervisor._view == NodeView(record=record, **view))
    following = record.model_copy(update={"generation": record.generation + 1})
    authority.push(following, **view)
    _until(lambda: running.supervisor._view.record == following)


def _shut_down(running, authority, *args, **fields):
    """Push SHUTDOWN and check that the supervisor ends with status 0."""
    authority.push(_record(MembershipState.SHUTDOWN, *args, **fields))
    assert running.wait() == 0


def test_ranks_start_outside_launcher_job(tmp_path, deployment):
    """Check that ranks start in their own session, without the launcher's variables or btl sm.

    A rank keeps the opt-in to running as root. The shared secret reaches a rank only through its
    environment.
    """
    authority, environ = deployment
    environ.update(OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_MCA_btl="self,sm,tcp")
    running = _serving(tmp_path, authority, environ)
    kept = ("OMPI_MCA_btl", *records._RUN_AS_ROOT_VARS)
    for rank in RANKS:
        started = _read(tmp_path, rank, 0)
        assert started["environ"]["TLLM_FAULT_TOLERANCE_RANK"] == f"{rank},4,2,0"  # on GPU rank % 2
        assert not any(
            key.startswith(("OMPI_", "PMIX_")) for key in started["environ"] if key not in kept
        )
        # The user's transports but sm, which leaves a /dev/shm segment behind.
        assert started["environ"]["OMPI_MCA_btl"] == "self,tcp"
        assert started["environ"]["OMPI_ALLOW_RUN_AS_ROOT"] == "1"
        assert started["environ"][SHARED_SECRET_ENV] == os.environ[SHARED_SECRET_ENV]
        assert os.environ[SHARED_SECRET_ENV] not in " ".join(started["argv"])
        # A terminal signal reaches only the supervisor.
        assert os.getsid(started["pid"]) == started["pid"]
    assert authority.hellos == [["node", "1", "2,3", socket.gethostname()]]
    _shut_down(running, authority)


@pytest.mark.parametrize(
    "btl, rank_btl",
    [
        (None, "^sm"),
        ("self,sm,tcp", "self,tcp"),
        ("self,vader,tcp", "self,tcp"),
        ("sm", "^sm"),
        ("^openib", "^openib,sm"),
        ("^sm,openib", "^sm,openib"),
    ],
)
def test_rank_btl_excludes_shared_memory(btl, rank_btl):
    """Check that a rank keeps the user's btl selection without the shared-memory transport.

    It also keeps the opt-in to running as root, and drops the launcher's other variables.
    """
    run_as_root = {
        name: "1"
        for name in (
            "OMPI_ALLOW_RUN_AS_ROOT",
            "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM",
            "PRTE_ALLOW_RUN_AS_ROOT",
            "PRTE_ALLOW_RUN_AS_ROOT_CONFIRM",
        )
    }
    environ = {**run_as_root, "OMPI_COMM_WORLD_RANK": "3", "PMIX_RANK": "3", "PATH": "/bin"}
    if btl is not None:
        environ["OMPI_MCA_btl"] = btl
    assert membership.rank_env(environ) == {**run_as_root, "OMPI_MCA_btl": rank_btl, "PATH": "/bin"}


@pytest.mark.parametrize("failure", ["crash", "hang"])
def test_failed_rank_is_replaced(tmp_path, deployment, failure):
    """Replace a rank the record leaves out, and kill it with its helpers if it still runs."""
    authority, environ = deployment
    environ["SUPERVISOR_TEST_MODES"] = "3:0=starts-helper"
    running = _serving(tmp_path, authority, environ)
    failed = _read(tmp_path, 3, 0)["pid"]
    if failure == "crash":
        os.kill(failed, signal.SIGKILL)
        _until(lambda: running.supervisor._processes[3].status is not None)
        # The authority has not removed the rank from the record yet.
        _push_and_wait_for_step(running, authority, _record(generation=2))
        assert not running.supervisor._processes[3].replacement
    authority.push(_record(MembershipState.PROPOSE, active_mask=0b0111, generation=4))
    _until(lambda: _read(tmp_path, 3, 1))
    replacement = _read(tmp_path, 3, 1)
    assert replacement["environ"]["TLLM_FAULT_TOLERANCE_RANK"] == "3,4,2,1"
    assert not _alive(failed)
    _until(lambda: not _alive(_read(tmp_path, 3, 0)["helper"]), 10)
    assert _alive(_read(tmp_path, 2, 0)["pid"])
    _shut_down(running, authority, incarnations=(0, 0, 0, 1), generation=5)
    assert not _alive(replacement["pid"])


@pytest.mark.parametrize("failure", ["cannot-start", "cannot-spawn"])
def test_failed_replacement_restarts_until_cannot_start(tmp_path, deployment, monkeypatch, failure):
    """Restart a replacement after a backoff until one exits with REPLACEMENT_CANNOT_START.

    A replacement whose process cannot be started counts as a failed one.
    """
    monkeypatch.setattr(supervisor_module, "_BACKOFF_S", 1.0)
    authority, environ = deployment
    environ["SUPERVISOR_TEST_MODES"] = "3:1=fails 3:2=cannot-start"
    running = _serving(tmp_path, authority, environ)
    if failure == "cannot-spawn":
        running.supervisor._exec_rank = [str(tmp_path / "missing")]
    os.kill(_read(tmp_path, 3, 0)["pid"], signal.SIGKILL)
    authority.push(_record(MembershipState.COMMITTED, active_mask=0b0111))
    _until(lambda: running.supervisor._given_up == {3})
    if failure == "cannot-spawn":
        failed = supervisor_module._MAX_FAILED_REPLACEMENTS
        assert running.supervisor._failed_replacements[3] == failed
    else:
        assert _read(tmp_path, 3, 2)
    _push_and_wait_for_step(
        running, authority, _record(MembershipState.COMMITTED, active_mask=0b0111, generation=2)
    )
    # No new replacement starts because it would fail the same way.
    assert running.supervisor._processes[3].status is not None
    assert running.supervisor._given_up == {3}
    _until(lambda: authority.given_up == (3,))
    _shut_down(running, authority, active_mask=0b0111)


def test_failed_replacements_back_off(monkeypatch):
    """Delay each replacement by the backoff, doubled per failed one in a row, then give up the rank."""
    monkeypatch.setattr(supervisor_module, "_BACKOFF_S", 1.0)
    monkeypatch.setenv(SHARED_SECRET_ENV, secrets.token_hex(32))
    supervisor = supervisor_module.NodeSupervisor(NODE, NODES, GPUS, [], {})
    started_s = []

    def cannot_start(rank, replacement):
        started_s.append(now_s)
        raise OSError("cannot start")

    monkeypatch.setattr(supervisor, "_start", cannot_start)
    record = _record(active_mask=0b0111)
    for now_s in (0.0, 0.9, 1.0, 2.9, 3.0, 100.0):
        supervisor._replace_when_due(3, record, now_s)
    assert started_s == [0.0, 1.0, 3.0]
    assert supervisor._given_up == {3}


_INSTALL = "a replacement could not install the change that adds it"
# The membership gives no reason when it leaves out a hung replacement.
_HUNG = ""


@pytest.mark.parametrize(
    "reasons", [(_INSTALL,), (_INSTALL, _HUNG)], ids=["same-reason", "other-reasons"]
)
def test_left_out_replacements_give_up_rank(tmp_path, deployment, reasons):
    """Check that no replacement starts once the membership left out three in a row, for any reason.

    Each change that adds a replacement of rank 3 leaves it out. The node serves without rank 3.
    """
    authority, environ = deployment
    running = _serving(tmp_path, authority, environ)
    os.kill(_read(tmp_path, 3, 0)["pid"], signal.SIGKILL)
    authority.push(_record(MembershipState.COMMITTED, active_mask=0b0111))
    generation = 1
    for incarnation in range(1, supervisor_module._MAX_FAILED_REPLACEMENTS + 1):
        _until(lambda incarnation=incarnation: _read(tmp_path, 3, incarnation))
        _until(lambda: authority.given_up == ())
        joined = (0, 0, 0, incarnation)
        adding = _record(MembershipState.INSTALL, 0b1111, joined, generation + 1)
        authority.push(adding.model_copy(update=dict(joining_mask=0b1000)))
        _until(lambda: running.supervisor._processes[3].listed)
        left_out = _record(MembershipState.PROPOSE, 0b0111, joined, generation + 2)
        reason = reasons[incarnation % len(reasons)]
        authority.push(left_out.model_copy(update=dict(reason=reason)))
        generation += 2
    _until(lambda: running.supervisor._given_up == {3})
    _push_and_wait_for_step(
        running, authority, _record(MembershipState.COMMITTED, 0b0111, joined, generation + 1)
    )
    assert _read(tmp_path, 3, supervisor_module._MAX_FAILED_REPLACEMENTS + 1) is None
    _until(lambda: authority.given_up == (3,))
    _shut_down(running, authority, 0b0111, joined, generation + 2)


@pytest.mark.parametrize(
    "admitted, served_s, given_up",
    [(False, 600.0, True), (True, 600.0, True), (True, 0.0, False)],
    ids=["before-admission", "after-admission", "after-serving"],
)
def test_short_lived_replacements_give_up_rank(
    tmp_path, deployment, monkeypatch, admitted, served_s, given_up
):
    """Check that no replacement starts once three in a row ended within ``_HEALTHY_SERVING_S``.

    A replacement that ends after serving longer is a new failure, so its replacement starts.
    """
    monkeypatch.setattr(supervisor_module, "_HEALTHY_SERVING_S", served_s)
    authority, environ = deployment
    running = _serving(tmp_path, authority, environ)
    os.kill(_read(tmp_path, 3, 0)["pid"], signal.SIGKILL)
    generation = 1
    authority.push(_record(MembershipState.COMMITTED, active_mask=0b0111, generation=generation))
    for incarnation in range(1, supervisor_module._MAX_FAILED_REPLACEMENTS + 1):
        _until(lambda incarnation=incarnation: _read(tmp_path, 3, incarnation))
        joined = (0, 0, 0, incarnation)
        if admitted:
            serving = _record(MembershipState.COMMITTED, 0b1111, joined, generation + 1)
            _push_and_wait_for_step(running, authority, serving)
            generation += 2
        os.kill(_read(tmp_path, 3, incarnation)["pid"], signal.SIGKILL)
        generation += 1
        authority.push(_record(MembershipState.COMMITTED, 0b0111, joined, generation))
    if given_up:
        _until(lambda: running.supervisor._given_up == {3})
    else:
        _until(lambda: _read(tmp_path, 3, supervisor_module._MAX_FAILED_REPLACEMENTS + 1))
    _push_and_wait_for_step(
        running, authority, _record(MembershipState.COMMITTED, 0b0111, joined, generation + 1)
    )
    assert (_read(tmp_path, 3, supervisor_module._MAX_FAILED_REPLACEMENTS + 1) is None) == given_up
    _until(lambda: authority.given_up == ((3,) if given_up else ()))
    _shut_down(running, authority, 0b0111, joined, generation + 3)


def test_replacement_killed_once_reported_stalled(tmp_path, deployment):
    """Check that a slow replacement runs on until the authority reports it stalled."""
    authority, environ = deployment
    running = _serving(tmp_path, authority, environ)
    os.kill(_read(tmp_path, 3, 0)["pid"], signal.SIGKILL)
    without_3 = _record(MembershipState.COMMITTED, active_mask=0b0111)
    authority.push(without_3)
    _until(lambda: _read(tmp_path, 3, 1))
    slow = _read(tmp_path, 3, 1)["pid"]
    _push_and_wait_for_step(running, authority, without_3.model_copy(update=dict(generation=2)))
    assert _alive(slow)
    authority.push(without_3.model_copy(update=dict(generation=4)), stalled_replacements=(3,))
    _until(lambda: _read(tmp_path, 3, 2))
    assert not _alive(slow)
    # The view that reported the stalled startup came before the next replacement started.
    assert _alive(_read(tmp_path, 3, 2)["pid"])
    _shut_down(running, authority, active_mask=0b0111, generation=5)


def test_replacement_killed_if_never_connected(tmp_path, deployment, monkeypatch):
    """Kill a replacement that does not connect to the authority within the rendezvous timeout."""
    authority, environ = deployment
    clock_s = [0.0]
    monkeypatch.setattr(
        supervisor_module, "time", SimpleNamespace(monotonic=lambda: time.monotonic() + clock_s[0])
    )
    running = _serving(tmp_path, authority, environ)
    os.kill(_read(tmp_path, 3, 0)["pid"], signal.SIGKILL)
    without_3 = _record(MembershipState.COMMITTED, active_mask=0b0111)
    authority.push(without_3, connected_mask=0b1111)
    _until(lambda: _read(tmp_path, 3, 1))
    clock_s[0] += membership.RENDEZVOUS_TIMEOUT_S
    _push_and_wait_for_step(
        running, authority, without_3.model_copy(update=dict(generation=2)), connected_mask=0b1111
    )
    assert _alive(_read(tmp_path, 3, 1)["pid"])
    _push_and_wait_for_step(
        running, authority, without_3.model_copy(update=dict(generation=4)), connected_mask=0b0111
    )
    _until(lambda: _read(tmp_path, 3, 2))
    assert not _alive(_read(tmp_path, 3, 1)["pid"])
    _shut_down(running, authority, active_mask=0b0111, generation=6)


def test_rank_0_end_stops_node(tmp_path, deployment):
    """Check that the end of rank 0 stops the node's ranks and starts no replacement of it."""
    authority, environ = deployment
    running = _serving(tmp_path, authority, environ, node=0)
    os.kill(_read(tmp_path, 0, 0)["pid"], signal.SIGKILL)
    assert running.wait() == 128 + signal.SIGKILL
    assert _read(tmp_path, 0, 1) is None
    assert not _alive(_read(tmp_path, 1, 0)["pid"])


@pytest.fixture
def held_kills(monkeypatch):
    """Record the signals that a supervisor sends to the process groups ``pgids``, and send none.

    A killed rank is given up after a rendezvous timeout of 1 s. The held ranks are killed after the
    test.
    """
    held = SimpleNamespace(pgids=set(), signals=[])
    signal_group = process_guard._signal_group

    def hold(pgid, signum):
        if pgid in held.pgids:
            held.signals.append((pgid, signum))
        else:
            signal_group(pgid, signum)

    monkeypatch.setattr(process_guard, "_signal_group", hold)
    monkeypatch.setattr(membership, "RENDEZVOUS_TIMEOUT_S", 1.0)
    yield held
    for pgid in held.pgids:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pgid, signal.SIGKILL)


def test_unkillable_rank_is_given_up(tmp_path, deployment, held_kills):
    """Give up a rank whose killed process did not end within the rendezvous timeout.

    Its GPU memory may not be free, so no replacement of it starts.
    """
    authority, environ = deployment
    running = _serving(tmp_path, authority, environ)
    pid = _read(tmp_path, 3, 0)["pid"]
    held_kills.pgids.add(pid)
    authority.push(_record(MembershipState.COMMITTED, active_mask=0b0111))
    _until(lambda: authority.given_up == (3,))
    assert held_kills.signals == [(pid, signal.SIGKILL)]
    assert _alive(pid)
    os.kill(pid, signal.SIGKILL)
    _push_and_wait_for_step(
        running, authority, _record(MembershipState.COMMITTED, active_mask=0b0111, generation=2)
    )
    assert _read(tmp_path, 3, 1) is None
    _shut_down(running, authority, active_mask=0b0111, generation=4)


def test_unreachable_rendezvous_warns(tmp_path, deployment, monkeypatch):
    """Warn once per interval about a rendezvous that does not answer, also before serving."""
    monkeypatch.setattr(supervisor_module, "_CONNECT_WARNING_INTERVAL_S", 1.0)
    warnings = []
    monkeypatch.setattr(
        logger, "warning", lambda message, *args, **kwargs: warnings.append(message)
    )
    authority, environ = deployment
    running = Running(tmp_path, environ)
    _until(lambda: any("Cannot connect to the membership authority" in w for w in warnings))
    authority.start()
    _until(lambda: authority.hellos)
    _shut_down(running, authority)


def test_wrong_shared_secret_ends_supervisor(tmp_path, deployment, monkeypatch):
    """End with status 1 and start no rank when the store's values carry another shared secret."""
    authority, environ = deployment
    authority.start()
    monkeypatch.setenv(SHARED_SECRET_ENV, secrets.token_hex(32))
    running = Running(tmp_path, environ)
    assert running.wait() == 1
    assert running.supervisor._processes == {}
    assert authority.hellos == []


def test_late_store_keeps_first_connection(tmp_path, deployment, monkeypatch):
    """Keep the first connection when the store answers only at the end of the connect timeout.

    The authority answers the hello later still, as it may while the deployment starts.
    """
    authority, environ = deployment
    authority.start(_record(MembershipState.STARTING))
    authority.answering.clear()
    clock_s = [0.0]
    open_store = membership.open_store

    def open_late_store(*args):
        store = open_store(*args)
        clock_s[0] += supervisor_module._CONNECT_TIMEOUT_S
        return store

    monkeypatch.setattr(membership, "open_store", open_late_store)
    monkeypatch.setattr(
        supervisor_module, "time", SimpleNamespace(monotonic=lambda: time.monotonic() + clock_s[0])
    )
    running = Running(tmp_path, environ)
    _until(lambda: authority.hellos)
    authority.answering.set()
    _until(lambda: running.supervisor._view is not None)
    assert len(authority.hellos) == 1
    assert not authority.closed.is_set()
    _shut_down(running, authority)


def test_first_connect_waits_without_polling(tmp_path, deployment, held_connect, monkeypatch):
    """Wait for a first connect that outlasts the connect timeout on the connect itself."""

    class RecordingSelector(selectors.DefaultSelector):
        def __init__(self):
            super().__init__()
            self.timeouts = []

        def select(self, timeout=None):
            self.timeouts.append(timeout)
            return super().select(timeout)

    monkeypatch.setattr(supervisor_module.selectors, "DefaultSelector", RecordingSelector)
    authority, environ = deployment
    authority.start(_record(MembershipState.STARTING))
    held_connect.hold.set()
    running = Running(tmp_path, environ)
    _until(held_connect.held.is_set)
    timeouts = running.supervisor._selector.timeouts
    _until(lambda: timeouts)
    # Only the connect's end wakes the loop, so it waits once and with no timeout.
    assert timeouts == [None]
    held_connect.finish()
    _until(lambda: running.supervisor._view is not None)
    _shut_down(running, authority)


def test_refused_reconnect_retried_after_start(tmp_path, deployment):
    """Connect again, and keep the ranks, when the authority refuses a reconnect after the start.

    The authority refuses a node's supervisor while it still holds the previous connection. It then
    holds its answer to the next hello, which may take up to the hello timeout, and a rank that ends
    meanwhile is reaped.
    """
    authority, environ = deployment
    running = _serving(tmp_path, authority, environ)
    authority.refusals = 1
    authority.answering.clear()
    for sock, _ in authority._connections:
        sock.shutdown(socket.SHUT_RDWR)
    _until(lambda: len(authority.hellos) == 2)
    os.kill(_read(tmp_path, 3, 0)["pid"], signal.SIGKILL)
    _until(lambda: running.supervisor._processes[3].status is not None, 5)
    authority.answering.set()
    _until(lambda: running.supervisor._connection is not None)
    assert running.status is None
    assert _alive(_read(tmp_path, 2, 0)["pid"])
    _shut_down(running, authority, active_mask=0b0111)


def test_ranks_replaced_after_refused_reconnect(tmp_path, deployment):
    """Replace the node's ranks that ended while the authority refused the supervisor's reconnects.

    The authority fails a node's ranks once it drops the node's previous connection, so they may end
    before it admits the reconnect. The node stops only with no authority to reach.
    """
    authority, environ = deployment
    running = _serving(tmp_path, authority, environ)
    authority.refusals = 1000
    for sock, _ in authority._connections:
        sock.shutdown(socket.SHUT_RDWR)
    _until(lambda: authority.refusals <= 998)
    for rank in RANKS:
        os.kill(_read(tmp_path, rank, 0)["pid"], signal.SIGKILL)
    _until(lambda: all(running.supervisor._processes[rank].status is not None for rank in RANKS))
    refusals = authority.refusals
    _until(lambda: authority.refusals <= refusals - 2)
    assert running.status is None
    authority.push(_record(MembershipState.COMMITTED, active_mask=0b0011, generation=2))
    authority.refusals = 0
    _until(lambda: all(_read(tmp_path, rank, 1) for rank in RANKS))
    _shut_down(running, authority, incarnations=(0, 0, 1, 1), generation=3)


def test_rank_runs_only_with_parent_death_signal(tmp_path):
    """Run no rank's command without the parent-death signal that ends it with its supervisor."""
    marker = tmp_path / "ran"
    code = (
        "import runpy, sys\n"
        f"sys.path.insert(0, {str(_LLMAPI)!r})\n"
        "import _llmapi_process_guard\n"
        "_llmapi_process_guard._arm_parent_death_signal = lambda: False\n"
        f"runpy.run_path({str(_SUPERVISOR)!r})['_exec_rank'](int(sys.argv[1]), sys.argv[2:])\n"
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", code, str(os.getpid()), "touch", str(marker)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 1
    assert "would outlive its node supervisor" in result.stderr
    assert not marker.exists()


def test_supervisor_without_parent_runs(tmp_path, deployment):
    """Run a supervisor without a parent, as a container's init process, which watches no parent."""
    authority, environ = deployment
    running = _serving(tmp_path, authority, environ, parent=0)
    assert not running.supervisor._parent_ended
    _shut_down(running, authority)


@pytest.fixture
def fake_mpirun(tmp_path):
    """Return a starter of a process named prterun, which is killed after the test.

    The starter takes the process's arguments and returns it and its PMIx server URI.
    """
    started = []

    def start(*arguments):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "prterun").symlink_to(shutil.which("bash"))
        mpirun = subprocess.Popen([str(bin_dir / "prterun"), "-c", "sleep 60; :", *arguments])
        started.append(mpirun)
        return mpirun, f"prterun-{socket.gethostname()}-{mpirun.pid}@0.0;tcp4://127.0.0.1:1"

    yield start
    for mpirun in started:
        mpirun.kill()
        mpirun.wait()


def _supervisor_command(tmp_path, node=NODE):
    """Return the command that runs the supervisor of ``node`` as its script does, with fake ranks."""
    script = tmp_path / "supervisor.py"
    script.write_text(
        _SUPERVISOR_SCRIPT
        % {
            "supervisor": str(_SUPERVISOR),
            "timings": _TIMINGS,
            "started": str(tmp_path / "supervisor.started"),
        }
    )
    options = ["--node-rank", str(node), "--nodes", str(NODES), "--gpus-per-node", str(GPUS)]
    return [sys.executable, str(script), *options, "--", sys.executable, str(tmp_path / "rank.py")]


@pytest.fixture
def supervisor_process(tmp_path, deployment):
    """Return a starter of node supervisor processes, which are killed after the test.

    With ``serving``, the starter waits until the node's ranks started, then starts the authority
    and waits for the supervisor's hello.
    """
    authority, _ = deployment
    started = []

    def start(environ, node=NODE, serving=False, **options):
        supervisor = subprocess.Popen(_supervisor_command(tmp_path, node), env=environ, **options)
        started.append(supervisor)
        if serving:
            ranks = range(node * GPUS, (node + 1) * GPUS)
            _until(
                lambda: all(_read(tmp_path, rank, 0) for rank in ranks), _SUPERVISOR_START_TIMEOUT_S
            )
            authority.start()
            _until(lambda: authority.hellos)
        return supervisor

    yield start
    for supervisor in started:
        supervisor.kill()
        supervisor.wait()


@pytest.mark.parametrize("supervisor_alive", [True, False])
def test_rank_runs_only_under_supervisor(tmp_path, supervisor_alive):
    """Check that a rank's command runs only if its supervisor still runs once its signal is armed.

    The supervisor is the test, or a process the rank's process no longer has as its parent.
    """
    marker = tmp_path / "ran"
    supervisor = os.getpid() if supervisor_alive else 1
    exec_rank = [sys.executable, "-S", str(_SUPERVISOR), "--exec-rank", str(supervisor)]
    status = subprocess.call([*exec_rank, "touch", str(marker)])
    assert status == (0 if supervisor_alive else 1)
    assert marker.exists() == supervisor_alive


def test_stop_reaps_session_supervisor():
    """Reap the session's node supervisor when it stops without a grace period, as on an abort."""
    supervisor = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    supervisor_module.stop_session_node_supervisor(supervisor, None)
    assert supervisor.returncode == -signal.SIGTERM


def test_session_supervisor_runs_in_own_session(monkeypatch):
    """Start the session's node supervisor outside the session's process group.

    A terminal's Ctrl-C then reaches only rank 0, which shuts the deployment down.
    """
    monkeypatch.setattr(supervisor_module.shutil, "which", lambda name: f"/bin/{name}")
    popen = Mock()
    monkeypatch.setattr(supervisor_module.subprocess, "Popen", popen)
    supervisor_module.start_session_node_supervisor(2, {})
    assert popen.call_args.kwargs["start_new_session"]


def _session_spawns(monkeypatch):
    """Fake the pool and the node supervisor of rank 0's session, and return the module and spawns.

    Each spawn is ``(kind, workers or supervisor, env or grace)``.
    """
    from tensorrt_llm.llmapi import _node_supervised_mpi_session as session_module

    spawns = []

    def spawn_pool(max_workers, env, **kwargs):
        spawns.append(("pool", max_workers, env))
        return SimpleNamespace(shutdown=lambda wait=True: None, submit=lambda *args: "future")

    def start_supervisor(n_workers, env):
        spawns.append(("supervisor", n_workers, env))
        return "supervisor"

    def stop_supervisor(supervisor, grace_s):
        spawns.append(("stop", supervisor, grace_s))

    monkeypatch.setattr(session_module, "MPIPoolExecutor", spawn_pool)
    monkeypatch.setattr(session_module, "start_session_node_supervisor", start_supervisor)
    monkeypatch.setattr(session_module, "stop_session_node_supervisor", stop_supervisor)
    return session_module, spawns


@pytest.mark.parametrize("wait", [True, False])
def test_session_pools_rank_0_only(monkeypatch, wait):
    """Check that rank 0's session pools rank 0 alone, and its node supervisor the others.

    A shutdown that waits gives the supervisor the shutdown grace period.
    """
    monkeypatch.setenv("TLLM_MGMN_SHUTDOWN_GRACE_SECONDS", "7")
    session_module, spawns = _session_spawns(monkeypatch)
    session = session_module.NodeSupervisedMpiPoolSession(n_workers=4)
    assert session.submit(print) == ["future"]
    session.shutdown(wait=wait)
    assert [spawn[:2] for spawn in spawns[:2]] == [("supervisor", 4), ("pool", 1)]
    assert spawns[1][2]["TLLM_FAULT_TOLERANCE_RANK"] == "0,4,4,0"
    assert spawns[2:] == [("stop", "supervisor", 7.0 if wait else None)]


def test_failed_pool_start_stops_supervisor(monkeypatch):
    """Check that a session whose rank-0 pool fails to start stops its node supervisor at once."""
    session_module, spawns = _session_spawns(monkeypatch)
    spawn_fails = Mock(side_effect=RuntimeError("spawn failed"))
    monkeypatch.setattr(session_module, "MPIPoolExecutor", spawn_fails)
    with pytest.raises(RuntimeError, match="spawn failed"):
        session_module.NodeSupervisedMpiPoolSession(n_workers=2)
    # Stopped at once, not by the shutdown of the session's finalizer.
    assert spawns[1:] == [("stop", "supervisor", None)]


def test_sessions_keep_own_rendezvous(monkeypatch):
    """Check that two sessions in one process start their ranks with their own rendezvous."""
    from tensorrt_llm._torch.distributed.membership import startup

    names = (SHARED_SECRET_ENV, RENDEZVOUS_ENV)
    for name in (*names, "TLLM_SPAWN_PROXY_PROCESS"):
        monkeypatch.delenv(name, raising=False)
    session_module, spawns = _session_spawns(monkeypatch)
    rendezvous = [startup.ensure_rendezvous(None) for _ in range(2)]
    for env, reserved in rendezvous:
        session_module.NodeSupervisedMpiPoolSession(n_workers=4, env_overrides=env).shutdown()
        reserved.close()
    first, second = (env for env, _ in rendezvous)
    assert all(first[name] != second[name] for name in names)
    started = [env for kind, _, env in spawns if kind != "stop"]
    assert [{name: env[name] for name in names} for env in started] == [first] * 2 + [second] * 2
    assert not any(name in os.environ for name in names)


def test_paused_rank_reported_while_not_stopped(tmp_path, deployment, pause_words):
    """Report a rank paused while its pause word names a process that is not stopped.

    The word names a process that the supervisor did not start, as rank 0's process in its session.
    SIGSTOP leaves the rank out of the paused mask until SIGCONT. A replacement that opens the word
    clears it.
    """
    authority, environ = deployment
    running = _serving(tmp_path, authority, environ)
    word, _ = pause_words(RANKS)
    helper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    try:
        word[0] = helper.pid
        _until(lambda: authority.paused_masks[-1:] == [0b0100])
        helper.send_signal(signal.SIGSTOP)
        _until(lambda: authority.paused_masks[-1:] == [0])
        helper.send_signal(signal.SIGCONT)
        _until(lambda: authority.paused_masks[-1:] == [0b0100])
        pause_words([2])
        assert word[0] == 0
        _until(lambda: authority.paused_masks[-1:] == [0])
    finally:
        helper.kill()
        helper.wait()
    _shut_down(running, authority)


def test_ended_process_not_paused(tmp_path, deployment, pause_words):
    """Report a rank not paused once the process that its pause word names has ended, silently.

    A rank that ends inside a paused call leaves its PID in the word until a replacement opens it.
    """
    authority, environ = deployment
    running = _serving(tmp_path, authority, environ)
    warnings = []
    running.supervisor._logger = Mock(warning=Mock(side_effect=warnings.append))
    helper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    word, _ = pause_words(RANKS)
    word[0] = helper.pid
    _until(lambda: authority.paused_masks[-1:] == [0b0100])
    helper.kill()
    helper.wait()
    _until(lambda: authority.paused_masks[-1:] == [0])
    count = len(authority.paused_masks)
    _until(lambda: len(authority.paused_masks) >= count + 2)
    assert authority.paused_masks[-1] == 0
    assert not [message for message in warnings if "Cannot read the state" in message]
    _shut_down(running, authority)


@pytest.mark.parametrize("end", ["SIGTERM", "ABORTED", "SHUTDOWN"])
def test_pause_words_unlinked_at_end(tmp_path, deployment, supervisor_process, pause_words, end):
    """Unlink the pause words of the node's ranks however the deployment ends.

    Keep the pause words of node 0, whose supervisor may run on the same host.
    """
    authority, environ = deployment
    pause_words(range(NODES * GPUS))
    words = [Path(membership.pause_word_path(rank)) for rank in RANKS]
    node_0_words = [Path(membership.pause_word_path(rank)) for rank in range(GPUS)]
    supervisor = supervisor_process(environ, serving=True)
    if end.startswith("SIG"):
        supervisor.send_signal(getattr(signal, end))
    else:
        authority.push(_record(MembershipState[end]))
    assert supervisor.wait(30) == {"SIGTERM": 143, "ABORTED": 1, "SHUTDOWN": 0}[end]
    assert not any(word.exists() for word in words)
    assert all(word.exists() for word in node_0_words)


def test_exit_skips_connect_in_flight(deployment, monkeypatch, supervisor_process):
    """Exit on a signal without waiting for a connect that blocks in the store client."""
    _, environ = deployment
    monkeypatch.setitem(_TIMINGS, "_CONNECT_TIMEOUT_S", 60.0)
    with socket.create_server(("127.0.0.1", 0)) as silent:
        environ[RENDEZVOUS_ENV] = f"127.0.0.1:{silent.getsockname()[1]}"
        supervisor = supervisor_process(environ)
        # The store client connected, and waits for an answer that never comes.
        assert select.select([silent], [], [], _SUPERVISOR_START_TIMEOUT_S)[0]
        supervisor.send_signal(signal.SIGTERM)
        assert supervisor.wait(10) == 128 + signal.SIGTERM


def test_supervisor_removes_cpu_affinity_constraints(deployment, supervisor_process):
    """Run the supervisor on every CPU, though its launcher binds it to one."""
    _, environ = deployment
    environ.pop("TLLM_NUMA_AWARE_WORKER_AFFINITY", None)
    allowed = os.sched_getaffinity(0)
    if len(allowed) < 2:
        pytest.skip("needs two CPUs")
    supervisor = supervisor_process(
        environ, serving=True, preexec_fn=lambda: os.sched_setaffinity(0, {min(allowed)})
    )
    assert os.sched_getaffinity(supervisor.pid) >= allowed


def test_signal_forwarded_to_rank_0_only(tmp_path, deployment, supervisor_process, fake_mpirun):
    """Forward the signal to rank 0, which runs the frontend, and later SIGTERM the other ranks."""
    _, environ = deployment
    environ["SUPERVISOR_TEST_MODES"] = "0:0=records-signals 1:0=records-signals"
    _, environ["PMIX_SERVER_URI41"] = fake_mpirun("--enable-recovery")
    supervisor = supervisor_process(environ, node=0, serving=True)
    supervisor.send_signal(signal.SIGINT)
    _until(lambda: (tmp_path / "signals-0-0").exists())
    assert (tmp_path / "signals-0-0").read_text() == "SIGINT"
    assert _alive(_read(tmp_path, 1, 0)["pid"])
    assert not (tmp_path / "signals-1-0").exists()
    assert supervisor.wait(30) == 130
    assert (tmp_path / "signals-1-0").read_text() == "SIGTERM"


@pytest.mark.parametrize("authority_ends", [False, True])
def test_ranks_end_after_shutdown(tmp_path, deployment, monkeypatch, authority_ends):
    """Wait for the active ranks to end after SHUTDOWN while the authority runs, however long.

    Each rank leaves the membership and ends on its own, and the authority runs until every active
    rank left. Rank 0 is waited for past the grace period that follows the authority's end too:
    its end is the deployment's end, which its launcher bounds. The supervisor's clock runs ahead
    of the grace period instead of the test waiting for it.
    """
    authority, environ = deployment
    clock_s = [0.0]
    monkeypatch.setattr(
        supervisor_module, "time", SimpleNamespace(monotonic=lambda: time.monotonic() + clock_s[0])
    )
    draining = (0,) if authority_ends else (0, 1)
    environ["SUPERVISOR_TEST_MODES"] = " ".join(f"{rank}:0=drains" for rank in draining)
    running = _serving(tmp_path, authority, environ, node=0)
    authority.push(_record(MembershipState.SHUTDOWN))
    _until(lambda: running.supervisor._view.record.state == MembershipState.SHUTDOWN)
    if authority_ends:
        _until(lambda: not _alive(_read(tmp_path, 1, 0)["pid"]))
        authority.close()
        _until(lambda: running.supervisor._connection is None)
    steps_s = []
    step = running.supervisor._step

    def recorded_step(now_s):
        status = step(now_s)
        steps_s.append(now_s)
        return status

    running.supervisor._step = recorded_step
    # Two steps, each two grace periods later, so a grace period that the first one starts ends.
    for _ in range(2):
        clock_s[0] += 2 * supervisor_module._STOP_GRACE_S
        later_s = supervisor_module.time.monotonic()
        running.supervisor._wake()
        _until(lambda later_s=later_s: any(s >= later_s for s in steps_s))
    assert all(_alive(_read(tmp_path, rank, 0)["pid"]) for rank in draining)
    (tmp_path / "drained").touch()
    assert running.wait() == 0
    assert [process.status for process in running.supervisor._processes.values()] == [0, 0]


def test_shutdown_skips_given_up_rank(tmp_path, deployment, held_kills):
    """End after SHUTDOWN while the killed process of a given-up rank still runs.

    The membership no longer lists the rank, so the supervisor does not wait for its process.
    """
    authority, environ = deployment
    environ["SUPERVISOR_TEST_MODES"] = "3:0=drains"
    running = _serving(tmp_path, authority, environ)
    pid = _read(tmp_path, 3, 0)["pid"]
    held_kills.pgids.add(pid)
    authority.push(_record(MembershipState.COMMITTED, active_mask=0b0111))
    _until(lambda: authority.given_up == (3,))
    _shut_down(running, authority, active_mask=0b0111, generation=2)
    assert _alive(pid)


@pytest.mark.parametrize("end, status", [("authority", 1), ("rank", 128 + signal.SIGKILL)])
def test_shutdown_stops_rank_left_after_authority(tmp_path, deployment, end, status):
    """Stop an active rank still running a grace period after the authority ended after SHUTDOWN.

    Exit with the status of an active rank that failed after SHUTDOWN.
    """
    authority, environ = deployment
    environ["SUPERVISOR_TEST_MODES"] = "3:0=drains"
    running = _serving(tmp_path, authority, environ)
    authority.push(_record(MembershipState.SHUTDOWN))
    pid = _read(tmp_path, 3, 0)["pid"]
    if end == "authority":
        _until(lambda: not _alive(_read(tmp_path, 2, 0)["pid"]))
        assert _alive(pid)
        authority.close()
    else:
        os.kill(pid, signal.SIGKILL)
    assert running.wait() == status
    assert not _alive(pid)
