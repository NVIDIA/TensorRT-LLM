# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU coverage of managed evaluator cleanup; no model server or GPU is needed."""

import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples" / "llm-api"))
from gsm8k_eval import server  # noqa: E402

pytestmark = pytest.mark.cpu_only


def test_snapshot_checks_leader_identity_and_includes_detached_children(monkeypatch):
    table = {
        100: (1, 100, b"10"),
        101: (100, 101, b"11"),  # worker starts its own session
        102: (101, 101, b"12"),
        103: (1, 100, b"13"),  # reparented session member
        104: (1, 104, b"14"),  # unrelated process
    }
    monkeypatch.setattr(server.os, "listdir", lambda _: [str(pid) for pid in table])
    monkeypatch.setattr(server, "_process_stat", table.get)
    assert server.owned_processes(100, b"10") == {pid: table[pid][2] for pid in range(100, 104)}
    table[100] = (1, 100, b"99")
    assert server.owned_processes(100, b"10") == {}


@pytest.mark.parametrize("current", [b"10", b"99", None])
def test_signal_checks_identity_after_opening_pidfd(monkeypatch, current):
    events = []
    monkeypatch.setattr(
        server.os, "pidfd_open", lambda pid: events.append(("open", pid)) or 77, raising=False
    )
    monkeypatch.setattr(server, "_process_stat", lambda pid: (1, 100, current) if current else None)
    monkeypatch.setattr(
        server.signal,
        "pidfd_send_signal",
        lambda fd, sig: events.append(("signal", fd, sig)),
        raising=False,
    )
    monkeypatch.setattr(server.os, "close", lambda fd: events.append(("close", fd)))
    server._signal_process(100, b"10", signal.SIGKILL)
    expected = [("open", 100)]
    if current == b"10":
        expected.append(("signal", 77, signal.SIGKILL))
    assert events == expected + [("close", 77)]


@pytest.mark.parametrize("interrupted", [False, True])
def test_stop_finishes_sweep_and_restores_handlers(monkeypatch, interrupted):
    process = SimpleNamespace(pid=100, returncode=None, wait=Mock())
    snapshots = iter([{100: b"10", 101: b"11"}, {100: b"10", 102: b"12"}])
    monkeypatch.setattr(server, "_process_stat", lambda pid: (1, 100, b"10"))
    monkeypatch.setattr(server, "owned_processes", lambda *args: next(snapshots))
    sent = Mock()
    monkeypatch.setattr(server, "_signal_process", sent)
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def waiting(pid, timeout):
        assert process.wait.call_count == 0, "leader must remain unreaped through the scan"
        for sig in handlers:
            assert signal.getsignal(sig) == signal.SIG_IGN
            os.kill(os.getpid(), sig)  # repeated shutdown signals are harmless
        if interrupted:
            raise KeyboardInterrupt

    monkeypatch.setattr(server, "_wait_unreaped", waiting)
    if interrupted:
        with pytest.raises(KeyboardInterrupt):
            server.stop(process, grace_s=0)
    else:
        server.stop(process, grace_s=0)
    assert sent.call_args_list[0].args == (100, b"10", signal.SIGTERM)
    assert {call.args for call in sent.call_args_list[1:]} == {
        (100, b"10", signal.SIGKILL),
        (101, b"11", signal.SIGKILL),
        (102, b"12", signal.SIGKILL),
    }
    process.wait.assert_called_once_with(timeout=60)
    assert {sig: signal.getsignal(sig) for sig in handlers} == handlers


def test_reaped_leader_is_not_rediscovered(monkeypatch):
    process = SimpleNamespace(pid=100, returncode=0, poll=lambda: 0)
    lookup = Mock(side_effect=AssertionError("must not inspect a released leader PID"))
    monkeypatch.setattr(server, "_process_stat", lookup)
    assert server.stop(process) == 0


@pytest.mark.parametrize("exited, code", [(True, 3), (False, -9)])
def test_managed_poll_preserves_leader_for_cleanup(monkeypatch, exited, code):
    # Call the method on a minimal process stand-in; no real process is reaped.
    process = SimpleNamespace(pid=100, returncode=None)
    monkeypatch.setattr(server.os, "CLD_EXITED", 1, raising=False)
    wait = Mock(return_value=SimpleNamespace(si_code=1 if exited else 2, si_status=abs(code)))
    monkeypatch.setattr(server.os, "waitid", wait, raising=False)
    monkeypatch.setattr(server.os, "WNOWAIT", 0x1000000, raising=False)
    assert server._ManagedServer.poll(process) == code
    assert process.returncode is None
    assert wait.call_args.args[2] & server.os.WNOWAIT


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc, pidfd and waitid lifecycle")
@pytest.mark.parametrize("exit_on_term", [False, True])
def test_linux_cleanup_kills_workers_and_preserves_unrelated_process(tmp_path, exit_on_term):
    # Exercise real sessions, a detached child, and a worker spawned during SIGTERM.
    # Run in a bounded driver so a cleanup regression cannot hang the test runner.
    driver = r'''
import json, os, select, signal, subprocess, sys, time
from pathlib import Path
from gsm8k_eval import server
worker_code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
leader_code = r"""
import json, os, signal, subprocess, sys, time
from gsm8k_eval import server
worker_code, exit_on_term, late = sys.argv[1:]
workers = [subprocess.Popen([sys.executable, '-c', worker_code], start_new_session=detach) for detach in (False, True)]
def stop(sig, frame):
    worker = subprocess.Popen([sys.executable, '-c', worker_code])
    with open(late, 'w') as f:
        json.dump([worker.pid, server._process_stat(worker.pid)[2].decode()], f)
    if exit_on_term == 'True':
        sys.exit(0)
signal.signal(signal.SIGTERM, stop)
print(json.dumps([[p.pid, server._process_stat(p.pid)[2].decode()] for p in workers]), flush=True)
while True:
    time.sleep(1)
"""
late = Path(sys.argv[2])
leader = server._ManagedServer(
    [sys.executable, '-c', leader_code, worker_code, sys.argv[1], str(late)],
    start_new_session=True, stdout=subprocess.PIPE, text=True,
)
unrelated = subprocess.Popen([sys.executable, '-c', worker_code], start_new_session=True)
workers = []
try:
    assert select.select([leader.stdout], [], [], 5)[0], 'leader did not become ready'
    workers = json.loads(leader.stdout.readline())
    server.stop(leader, grace_s=0.3)
    assert leader.returncode is not None
    assert late.exists(), 'SIGTERM handler did not run'
    workers.append(json.loads(late.read_text()))
    deadline = time.monotonic() + 3
    while True:
        living = []
        for pid, identity in workers:
            try:
                state = Path(f'/proc/{pid}/stat').read_bytes().rsplit(b')', 1)[1].split()[0]
                if state != b'Z':
                    living.append(pid)
            except FileNotFoundError:
                pass
        if not living or time.monotonic() >= deadline:
            break
        time.sleep(0.01)
    assert not living, living
    assert unrelated.poll() is None
finally:
    if late.exists():
        workers.append(json.loads(late.read_text()))
    for pid, identity in workers:
        server._signal_process(pid, identity.encode(), signal.SIGKILL)
    leader.kill()
    leader.wait(timeout=5)
    unrelated.kill()
    unrelated.wait(timeout=5)
'''
    env = dict(os.environ, PYTHONPATH=str(Path(server.__file__).resolve().parents[1]))
    result = subprocess.run(
        [sys.executable, "-c", driver, str(exit_on_term), str(tmp_path / "late.pid")],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("timeouts", [0, 1, 2])
def test_memory_poller_shutdown_is_bounded(tmp_path, monkeypatch, caplog, timeouts):
    process = Mock(pid=123)
    process.wait.side_effect = [subprocess.TimeoutExpired("nvidia-smi", 10)] * timeouts + [0]
    monkeypatch.setattr(server.subprocess, "Popen", Mock(return_value=process))
    with server.gpu_memory_poller(tmp_path / "memory.csv", ["0"]):
        pass
    process.terminate.assert_called_once_with()
    assert process.kill.call_count == bool(timeouts)
    assert len(process.wait.call_args_list) == min(timeouts + 1, 2)
    assert all(call.kwargs == {"timeout": 10} for call in process.wait.call_args_list)
    assert ("did not exit after SIGKILL" in caplog.text) == (timeouts == 2)
