# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the GLM-5.2 target's Lamport collective entries.

One job certifies all four -- `cute_dsl_allreduce`,
`cute_dsl_allreduce_add_rmsnorm`, `cute_dsl_allreduce_add_rmsnorm_quant_nvfp4`
and `cute_dsl_allgather_widen` -- because they share one workspace and its
device slot counter. The matrix is
`_cute_dsl_allreduce_op_matrix.py` beside this file, its own 4-rank launcher;
`_rank_job` (in the shared comm test directory) starts it in a fresh
interpreter and turns its exit code into an assertion.
"""

import atexit
import ctypes
import errno
import os
import select
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal

import psutil
import pytest
import torch
from test_common import magic_import as magic_import

__extra_import_path__ = ["../../../../comm"]
if TYPE_CHECKING:
    from _torch.modeling_v2.comm import _rank_job
else:
    import _rank_job  # noqa: E402 — needs the path declared above


def _sm103_devices() -> int:
    """How many of the devices the job would claim are sm_103."""
    if not torch.cuda.is_available():
        return 0
    claimed = range(min(torch.cuda.device_count(), _rank_job.WORLD_SIZE))
    return sum(torch.cuda.get_device_capability(i) == (10, 3) for i in claimed)


# Spawns 4 rank processes over the visible devices. Under xdist several
# workers would fight for the same GPUs, so this must run alone, like the
# shared collective entries.
@pytest.mark.no_xdist
def test_cute_dsl_allreduce_op_matrix() -> None:
    if _sm103_devices() < _rank_job.WORLD_SIZE:
        pytest.skip(f"the Lamport all-reduce entries run on {_rank_job.WORLD_SIZE} sm_103 GPUs")
    _rank_job.run("cute_dsl_allreduce", directory=Path(__file__).resolve().parent)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("outcome", "use_libuv", "ignore_sigchld"),
    [
        pytest.param("success", None, False, id="success"),
        pytest.param("success", "0", False, id="success-without-libuv"),
        pytest.param("success", None, True, id="success-with-ignored-sigchld"),
        pytest.param("worker-failure", None, False, id="worker-failure"),
        pytest.param("worker-signal", None, False, id="worker-signal"),
        pytest.param("spawn-failure", None, False, id="spawn-failure"),
        pytest.param("deadline", None, False, id="deadline"),
        pytest.param(
            "probe-teardown-failure",
            None,
            True,
            id="probe-teardown-failure-with-ignored-sigchld",
        ),
    ],
)
def test_rendezvous_port_ownership(
    outcome: Literal[
        "success",
        "worker-failure",
        "worker-signal",
        "spawn-failure",
        "deadline",
        "probe-teardown-failure",
    ],
    use_libuv: str | None,
    ignore_sigchld: bool,
    tmp_path: Path,
) -> None:
    supervisor = r'''
import ctypes
import os
import select
import signal
import socket
import subprocess
import sys
import time

control = socket.socket(fileno=int(sys.argv[1]))
control.settimeout(120)
control_events = select.poll()
control_events.register(control, select.POLLIN)
inherited_sigchld = signal.getsignal(signal.SIGCHLD)
signal.signal(signal.SIGCHLD, signal.SIG_DFL)
supervisor_pid = os.getpid()
prctl = ctypes.CDLL(None, use_errno=True).prctl
prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
prctl.restype = ctypes.c_int

def prepare_probe():
    if prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "could not bind probe lifetime")
    if os.getppid() != supervisor_pid:
        signal.raise_signal(signal.SIGKILL)
    signal.signal(signal.SIGCHLD, inherited_sigchld)

def receive():
    try:
        return control.recv(1)
    except ConnectionResetError:
        return b""

def send(message):
    try:
        control.sendall(message)
    except (BrokenPipeError, ConnectionResetError):
        return False
    return True

probe_code = """
import os
import runpy
import socket
import sys
from pathlib import Path

gate = socket.socket(fileno=int(sys.argv[6]))
if gate.recv(1) != b"G":
    raise AssertionError("probe startup was not authorized")
entry = runpy.run_path(sys.argv[1])
entry["_run_rendezvous_case"](
    sys.argv[2], sys.argv[3] or None, sys.argv[4] == "1", Path(sys.argv[5])
)
gate.sendall(b"C")
"""
probe = None
gate, child_gate = socket.socketpair()
try:
    if receive() == b"G":
        probe = subprocess.Popen(
            [sys.executable, "-c", probe_code, *sys.argv[2:], str(child_gate.fileno())],
            pass_fds=(child_gate.fileno(),),
            preexec_fn=prepare_probe,
        )
        child_gate.close()
        if send(b"S") and receive() == b"G":
            gate.sendall(b"G")
            deadline = time.monotonic() + 120
            while probe.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("probe did not exit in 120 seconds")
                if control_events.poll(min(100, max(1, int(remaining * 1000)))):
                    if not receive():
                        break
            else:
                try:
                    completed = gate.recv(1) == b"C"
                except ConnectionResetError:
                    completed = False
                prefix = b"completed:" if completed else b"incomplete:"
                send(prefix + str(probe.returncode).encode() + b"\n")
finally:
    if probe is not None:
        if probe.poll() is None:
            probe.kill()
        probe.wait(timeout=10)
    gate.close()
    child_gate.close()
'''
    control, child_control = socket.socketpair()
    with control, child_control:
        command = [
            sys.executable,
            "-S",
            "-c",
            supervisor,
            str(child_control.fileno()),
            str(Path(__file__).resolve()),
            outcome,
            use_libuv or "",
            str(int(ignore_sigchld)),
            str(tmp_path),
        ]
        deadline = time.monotonic() + 120
        process = subprocess.Popen(
            command,
            pass_fds=(child_control.fileno(),),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        process_fd = None
        try:
            process_fd = _open_supervisor_pidfd(process, command, child_control, deadline)
            child_control.close()
            control.settimeout(max(0, deadline - time.monotonic()))
            control.sendall(b"G")
            if control.recv(1) != b"S":
                raise AssertionError("supervisor did not acknowledge startup")
            control.sendall(b"G")
            output, _ = process.communicate(timeout=max(0, deadline - time.monotonic()))
            control.settimeout(max(0, deadline - time.monotonic()))
            with control.makefile("rb") as status_stream:
                status = status_stream.readline(32)
            expected = (
                b"completed:" + str(-signal.SIGTERM).encode()
                if outcome == "probe-teardown-failure" and not ignore_sigchld
                else b"completed:0"
            ) + b"\n"
            if status != expected:
                raise AssertionError(
                    f"probe exit status {status!r}\n{output.decode(errors='replace')}"
                )
        finally:
            child_control.close()
            control.close()
            if process_fd is not None:
                try:
                    process_events = select.poll()
                    process_events.register(process_fd, select.POLLIN)
                    if not process_events.poll(0):
                        syscall = ctypes.CDLL(None, use_errno=True).syscall
                        pidfd_send_signal = 424
                        if syscall(pidfd_send_signal, process_fd, signal.SIGKILL, None, 0) < 0:
                            error = ctypes.get_errno()
                            if error != errno.ESRCH:
                                raise OSError(error, "could not stop supervisor")
                finally:
                    os.close(process_fd)
            process.communicate(timeout=10)


def _open_supervisor_pidfd(
    process: subprocess.Popen, command: list[str], child_control: socket.socket, deadline: float
) -> int:
    """Pin the gated child only after checking its command and socket identity."""
    syscall = ctypes.CDLL(None, use_errno=True).syscall
    pidfd_open = 434
    descriptor = syscall(pidfd_open, process.pid, 0)
    if descriptor < 0:
        raise OSError(ctypes.get_errno(), "could not open supervisor descriptor")
    try:
        p_pidfd = 3
        os.waitid(p_pidfd, descriptor, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        expected_socket = f"socket:[{os.fstat(child_control.fileno()).st_ino}]"
        if os.readlink(f"/proc/{process.pid}/fd/{child_control.fileno()}") != expected_socket:
            raise AssertionError("supervisor control socket does not match")
        process_events = select.poll()
        process_events.register(descriptor, select.POLLIN)
        while not (
            actual_command := Path(f"/proc/{process.pid}/cmdline").read_bytes().split(b"\0")[:-1]
        ):
            if process_events.poll(0):
                raise ChildProcessError("supervisor exited before publishing its command")
            if time.monotonic() >= deadline:
                raise TimeoutError("supervisor command was not published")
        if actual_command != [os.fsencode(argument) for argument in command]:
            raise AssertionError("supervisor command does not match")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _run_rendezvous_case(
    outcome: Literal[
        "success",
        "worker-failure",
        "worker-signal",
        "spawn-failure",
        "deadline",
        "probe-teardown-failure",
    ],
    use_libuv: str | None,
    ignore_sigchld: bool,
    tmp_path: Path,
) -> None:
    if ignore_sigchld:
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        test_rendezvous_port_ownership(outcome, use_libuv, False, tmp_path)
        return
    if outcome == "probe-teardown-failure":
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM})
        atexit.register(signal.raise_signal, signal.SIGTERM)
        return
    if outcome in ("worker-failure", "worker-signal"):
        pr_set_child_subreaper = 36
        prctl = ctypes.CDLL(None).prctl
        prctl.argtypes = [
            ctypes.c_int,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
        ]
        prctl.restype = ctypes.c_int
        if prctl(pr_set_child_subreaper, 1, 0, 0, 0) != 0:
            raise AssertionError("could not enable isolated child adoption")
    controller_read, controller_write = os.pipe()
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            _check_rendezvous_port_ownership(
                monkeypatch,
                outcome,
                use_libuv,
                tmp_path,
                controller_read,
            )
    finally:
        os.close(controller_write)
        os.close(controller_read)


def _check_rendezvous_port_ownership(
    monkeypatch: pytest.MonkeyPatch,
    outcome: Literal["success", "worker-failure", "worker-signal", "spawn-failure", "deadline"],
    use_libuv: str | None,
    tmp_path: Path,
    controller_read: int,
) -> None:
    launcher = _rank_job._load_launcher_constants(
        Path(__file__).with_name("_cute_dsl_allreduce_op_matrix.py")
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    if use_libuv is None:
        monkeypatch.delenv("USE_LIBUV", raising=False)
    else:
        monkeypatch.setenv("USE_LIBUV", use_libuv)
    monkeypatch.setattr(launcher, "DEADLINE_S", 60)
    if outcome == "deadline":
        clock_time = 0.0

        def advance_clock(seconds: float) -> None:
            nonlocal clock_time
            clock_time += 61.0
            time.sleep(seconds)

        monkeypatch.setattr(
            launcher,
            "time",
            SimpleNamespace(monotonic=lambda: clock_time, sleep=advance_clock),
        )
    helper_path = tmp_path / "helper.pid"
    watch_controller = """
import os
import signal
import threading

def exit_with_controller():
    os.read(int(os.environ["TEST_CONTROLLER_FD"]), 1)
    signal.raise_signal(signal.SIGKILL)

threading.Thread(target=exit_with_controller, daemon=True).start()
"""
    helper = watch_controller + "print('ready', flush=True); threading.Event().wait()"
    worker = (
        watch_controller
        + """
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

if os.environ["TEST_RANK_OUTCOME"] == "deadline":
    os.write(sys.stdout.fileno(), b"R")
    threading.Event().wait()

import torch
import torch.distributed as dist

if os.environ["TEST_RANK_OUTCOME"] == "spawn-failure":
    threading.Event().wait()
dist.init_process_group("gloo", timeout=timedelta(seconds=30))
rank = dist.get_rank()
value = torch.tensor([rank])
dist.all_reduce(value)
if value.item() != sum(range(dist.get_world_size())):
    raise AssertionError("incorrect all-reduce result")
if os.environ["TEST_RANK_OUTCOME"] in ("worker-failure", "worker-signal"):
    if rank == 0:
        helper = subprocess.Popen(
            [sys.executable, "-c", os.environ["TEST_HELPER_CODE"]],
            pass_fds=(int(os.environ["TEST_CONTROLLER_FD"]),),
            stdout=subprocess.PIPE,
        )
        Path(os.environ["TEST_HELPER_PID_PATH"]).write_text(f"{helper.pid} {os.getpgid(helper.pid)}")
        if helper.stdout.readline().strip() != b"ready":
            raise AssertionError("worker helper did not become ready")
        if os.environ["TEST_RANK_OUTCOME"] == "worker-signal":
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM})
            os.kill(os.getpid(), signal.SIGTERM)
        sys.exit(7)
    threading.Event().wait()
dist.destroy_process_group()
"""
    )
    spawn = subprocess.Popen
    tcp_store = launcher.dist.TCPStore
    kill_group = os.killpg
    children = []
    port = None
    owned_listeners = set()
    store_backends = []
    launcher_running = True

    def listeners(port_number: int) -> set[Path]:
        return {
            Path(f"/proc/self/fd/{connection.fd}").readlink()
            for connection in psutil.Process().net_connections(kind="tcp")
            if connection.status == psutil.CONN_LISTEN and connection.laddr.port == port_number
        }

    def start_store(*args: Any, **kwargs: Any) -> torch.distributed.TCPStore:
        store = tcp_store(*args, **kwargs)
        store_backends.append(store.libuvBackend)
        return store

    def require_owned_listener() -> None:
        if port is None or not owned_listeners or not owned_listeners.issubset(listeners(port)):
            raise AssertionError("launcher listener closed before worker cleanup")

    def kill_owned_group(process_group: int, signum: int) -> None:
        require_owned_listener()
        status = os.waitid(os.P_PID, process_group, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        if outcome == "deadline" and status is not None:
            raise AssertionError("deadline worker exited before cleanup")
        if outcome == "success":
            raise AssertionError("successful workers must not be signaled")
        kill_group(process_group, signum)

    with socket.socket() as competitor:

        def start_worker(
            _command: list[str], env: dict[str, str], start_new_session: bool
        ) -> subprocess.Popen:
            nonlocal port, owned_listeners
            if outcome == "spawn-failure" and children:
                raise OSError(errno.EAGAIN, "worker start failed")
            if port is None:
                port = int(env["MASTER_PORT"])
                owned_listeners = listeners(port)
                try:
                    competitor.bind((env["MASTER_ADDR"], port))
                except OSError as error:
                    if error.errno != errno.EADDRINUSE:
                        raise
                else:
                    competitor.listen()
            child = spawn(
                [sys.executable, "-c", worker],
                env=dict(
                    env,
                    TEST_RANK_OUTCOME=outcome,
                    TEST_HELPER_PID_PATH=str(helper_path),
                    TEST_CONTROLLER_FD=str(controller_read),
                    TEST_HELPER_CODE=helper,
                ),
                pass_fds=(controller_read,),
                start_new_session=start_new_session,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            children.append(child)
            wait = child.wait

            def wait_owned_worker(timeout: float | None = None) -> int:
                if launcher_running:
                    require_owned_listener()
                return wait(timeout=timeout)

            monkeypatch.setattr(child, "wait", wait_owned_worker)
            if outcome == "deadline":
                if child.stdout is None:
                    raise AssertionError("deadline worker pipe is missing")
                ready = select.poll()
                ready.register(child.stdout, select.POLLIN)
                if not ready.poll(10_000) or os.read(child.stdout.fileno(), 1) != b"R":
                    raise AssertionError("deadline worker did not become ready")
            return child

        monkeypatch.setattr(launcher.subprocess, "Popen", start_worker)
        monkeypatch.setattr(launcher.dist, "TCPStore", start_store)
        monkeypatch.setattr(launcher.os, "killpg", kill_owned_group)
        try:
            if outcome == "spawn-failure":
                with pytest.raises(OSError, match="worker start failed"):
                    launcher._spawn_ranks()
            elif outcome in ("worker-failure", "worker-signal"):
                expected_code = 7 if outcome == "worker-failure" else -signal.SIGTERM
                with pytest.raises(AssertionError, match=f"rank 0 exited {expected_code}"):
                    launcher._spawn_ranks()
            elif outcome == "deadline":
                with pytest.raises(AssertionError, match=r"did not finish in 60s \(wedged\)"):
                    launcher._spawn_ranks()
            else:
                launcher._spawn_ranks()
            launcher_running = False
            if not children:
                raise AssertionError("no workers were started")
            if outcome == "deadline" and len(children) != launcher.WORLD_SIZE:
                raise AssertionError("deadline did not exercise every worker")
            for child in children:
                with pytest.raises(ChildProcessError):
                    os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            if port is None or not owned_listeners:
                raise AssertionError("launcher did not own a listening socket")
            if not owned_listeners.isdisjoint(listeners(port)):
                raise AssertionError("launcher listener remained open")
            if not store_backends or not all(
                backend == (use_libuv != "0") for backend in store_backends
            ):
                raise AssertionError("unexpected rendezvous store backend")
            if outcome in ("worker-failure", "worker-signal"):
                helper_pid, helper_group = map(int, helper_path.read_text().split())
                if helper_group != children[0].pid:
                    raise AssertionError("helper left its worker's process group")
                deadline = time.monotonic() + 10
                while os.waitid(os.P_PID, helper_pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
                    if time.monotonic() >= deadline:
                        raise AssertionError("worker helper did not exit")
                    time.sleep(0.05)
            for child in children:
                child.communicate(timeout=10)
        finally:
            launcher_running = False
            signal.signal(signal.SIGCHLD, signal.SIG_DFL)
            for child in children:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=10)
            if helper_path.exists():
                helper_pid = int(helper_path.read_text().split()[0])
                if os.waitid(os.P_PID, helper_pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
                    os.kill(helper_pid, signal.SIGKILL)
                os.waitpid(helper_pid, 0)
            for child in children:
                child.communicate(timeout=10)
