# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise launcher supervision without importing TensorRT-LLM or initializing MPI."""

import importlib.util
import json
import os
import pty
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

pytestmark = pytest.mark.cpu_only

_LAUNCHER = Path(__file__).parents[3] / "tensorrt_llm" / "llmapi" / "trtllm-llmapi-launch"
_PROCESS_GUARD = _LAUNCHER.with_name("_llmapi_process_guard.py")
_MPI_PREFIXES = (
    "OMPI_",
    "PMIX_",
    "PMI_",
    "SLURM_",
    "MPI_",
    "UCX_",
    "I_MPI_",
    "HYDRA_",
    "KMP_",
    "MPICH_",
    "MV2_",
    "CRAY_",
)

# FIFOs establish readiness explicitly: the engine cannot finish before the
# server starts, and an early-exiting server waits for the engine's child.
_STUB = r"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

root = Path(os.environ["LAUNCHER_TEST_ROOT"])
mode = os.environ["LAUNCHER_TEST_MODE"]
role = "task" if sys.argv[1] == "task" else "server"
if "--action" in sys.argv:
    role = "stop"
if sys.argv[1] == "-S":
    if "--parent-pid" in sys.argv and "--role" in sys.argv:
        guard_role = sys.argv[sys.argv.index("--role") + 1]
        guard_directory = root / "guards"
        guard_directory.mkdir(exist_ok=True)
        guard_record = guard_directory / f".{guard_role}.tmp"
        ready_file = Path(sys.argv[sys.argv.index("--ready-file") + 1])
        guard_record.write_text(json.dumps({
            "pid": os.getpid(), "pgid": os.getpgrp(), "ready_file": str(ready_file),
        }))
        guard_record.replace(guard_directory / f"{guard_role}.json")
        if mode == "stop_guard_fails" and guard_role == "stop":
            (root / "guard-start-failed").touch()
            sys.exit(23)
        if guard_role == "task" and mode in ("guard_go_fails", "guard_reap_fails"):
            gate = "go" if mode == "guard_go_fails" else "reap"
            Path(f"{ready_file}.{gate}").mkdir()
            (root / "guard-write-fault").touch()
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])

try:
    os.fstat(200)
    lock_fd_open = True
except OSError:
    lock_fd_open = False
(root / f"{role}.json").write_text(json.dumps({
    "pid": os.getpid(),
    "pgid": os.getpgrp(),
    "pmi_rank": os.environ.get("PMI_RANK"),
    "lock_fd_open": lock_fd_open,
    "workspace": os.environ.get("FLASHINFER_WORKSPACE_BASE"),
}))

def send(name):
    with (root / name).open("w") as stream:
        stream.write("ready\n")

def receive(name):
    with (root / name).open() as stream:
        assert stream.readline() == "ready\n"

def hang():
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        signal.pause()

def wait_for_phase_release():
    (root / "phase-active").touch()
    deadline = time.monotonic() + 8
    while not (root / "phase-release").exists():
        assert time.monotonic() < deadline, "launcher did not poll the active phase"
        time.sleep(0.01)
    (root / "phase-active").unlink()
    (root / "phase-completed").touch()

def start_child(name="task_child", ignore_term=False):
    child_code = "import signal; "
    if ignore_term:
        child_code += "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
    child_code += "print('ready', flush=True); signal.pause()"
    child = subprocess.Popen([
        sys.executable, "-c", child_code,
    ], stdout=subprocess.PIPE, text=True)
    assert child.stdout.readline() == "ready\n"
    (root / f"{name}.json").write_text(json.dumps({
        "pid": child.pid, "pgid": os.getpgid(child.pid),
    }))
    return child

if role == "server":
    if mode == "follower":
        (root / "worker-ready").touch()
        while True:
            signal.pause()
    if mode.startswith("owner_dies"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    send("server_ready")
    receive("task_ready")
    if mode.startswith("owner_dies"):
        (root / "server-owner-ready").touch()
        hang()
    if mode.startswith("server_exits"):
        sys.exit(int(os.environ.get("LAUNCHER_TEST_SERVER_STATUS", "17")))
    receive("stop_requested")
    if mode == "server_hangs":
        hang()
elif role == "stop":
    if mode == "owner_dies_during_stop":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        start_child("stop_child", ignore_term=True)
        (root / "stop-owner-ready").touch()
        hang()
    if mode == "stop_hangs":
        hang()
    if mode == "stop_fails":
        sys.exit(23)
    if os.environ.get("LAUNCHER_TEST_PHASE") == "shutdown":
        wait_for_phase_release()
    send("stop_requested")
else:
    receive("server_ready")
    if mode == "read_stdin":
        assert sys.stdin.read() == ""
    if mode in ("owner_dies", "task_leaves_child"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        start_child(ignore_term=True)
        send("task_ready")
        if mode == "task_leaves_child":
            sys.exit(int(os.environ.get("LAUNCHER_TEST_TASK_STATUS", "0")))
        (root / "task-owner-ready").touch()
        hang()
    if mode == "server_exits_naturally":
        def terminate(signum, frame):
            (root / "task-term").touch()
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, terminate)
        send("task_ready")
        deadline = time.monotonic() + 8
        while not (root / "server-observed").exists():
            assert time.monotonic() < deadline, "launcher did not observe server exit"
            time.sleep(0.01)
        if os.environ.get("LAUNCHER_TEST_PHASE") == "task_wait":
            wait_for_phase_release()
        else:
            time.sleep(0.3)
        (root / "task-completed-naturally").touch()
        sys.exit(int(os.environ["LAUNCHER_TEST_TASK_STATUS"]))
    elif mode.startswith("server_exits"):
        child = start_child(ignore_term=mode == "server_exits_ignores_term")
        def terminate(signum, frame):
            (root / "task-term").touch()
            if mode == "server_exits_ignores_term":
                return
            if mode == "server_exits_slow_cleanup":
                if os.environ.get("LAUNCHER_TEST_PHASE") == "cleanup":
                    wait_for_phase_release()
                else:
                    time.sleep(float(os.environ.get("LAUNCHER_TEST_CLEANUP_DELAY", "3")))
                (root / "task-cleanup-completed").touch()
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, terminate)
        try:
            send("task_ready")
            while True:
                signal.pause()
        finally:
            child.terminate()
            child.wait(timeout=5)
    else:
        send("task_ready")
        sys.exit(int(os.environ.get("LAUNCHER_TEST_TASK_STATUS", "0")))
"""

# Advance the whole-second clock at a known child handshake, retaining real
# sleeps. A child that needs a second poll must still be allowed to finish.
_PHASE_SLEEP = r"""
launcher_test_phase_polls=0
sleep() {
    if [ "$#" -eq 1 ] && [ "$1" = "0.1" ] && \
        [ -f "$LAUNCHER_TEST_ROOT/phase-active" ]; then
        launcher_test_phase_polls=$((launcher_test_phase_polls + 1))
        printf '%s\n' "$launcher_test_phase_polls" > "$LAUNCHER_TEST_ROOT/phase-polls"
        if [ "$launcher_test_phase_polls" -eq 1 ]; then
            SECONDS=$((SECONDS + 1))
        elif [ "$launcher_test_phase_polls" -eq 2 ]; then
            : > "$LAUNCHER_TEST_ROOT/phase-release"
        fi
    fi
    command sleep "$@"
}
"""


def _launcher_env(tmp_path: Path, mode: str) -> dict[str, str]:
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    python_stub = stub_bin / "python3"
    python_stub.write_text(f"#!{sys.executable}\n{_STUB}")
    python_stub.chmod(0o755)
    for name in ("server_ready", "task_ready", "stop_requested"):
        os.mkfifo(tmp_path / name)
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(_MPI_PREFIXES)
        and not name.startswith("TLLM_LLMAPI_LAUNCH_")
        and not name.startswith(("FLASHINFER_", "TRTLLM_FLASHINFER_", "TLLM_SPAWN_PROXY_PROCESS"))
    }
    env.update(
        {
            "HOME": str(tmp_path / "home"),
            "PATH": f"{stub_bin}{os.pathsep}{env['PATH']}",
            "PMI_RANK": "0",
            "PMI_SIZE": "1",
            "TLLM_LLMAPI_LAUNCH_STOP_TIMEOUT": "3",
            "TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR": f"ipc://{tmp_path / 'ipc'}",
            "LAUNCHER_TEST_ROOT": str(tmp_path),
            "LAUNCHER_TEST_MODE": mode,
        }
    )
    return env


def _process_records(tmp_path: Path) -> dict[str, dict]:
    return {path.stem: json.loads(path.read_text()) for path in tmp_path.glob("*.json")}


def _owned_process_records(tmp_path: Path) -> dict[str, dict]:
    records = _process_records(tmp_path)
    for path in (tmp_path / "guards").glob("*.json"):
        record = json.loads(path.read_text())
        records[f"guard_{path.stem}"] = record
        if "payload_pid" in record:
            records[f"payload_{path.stem}"] = {
                "pid": record["payload_pid"],
                "pgid": record["payload_pid"],
            }
    return records


def _cleanup_owned_processes(process: subprocess.Popen, tmp_path: Path) -> None:
    records = _owned_process_records(tmp_path)
    process_groups = {process.pid} if process.poll() is None else set()

    def add_existing_group(pid: int, pgid: int) -> None:
        try:
            if os.getpgid(pid) == pgid:
                process_groups.add(pgid)
        except ProcessLookupError:
            pass

    # Completed children no longer reserve their numeric process-group IDs.
    for record in records.values():
        add_existing_group(record["pid"], record["pgid"])
    # A failed go acknowledgment can leave a gated payload that never executes
    # the stand-in and therefore has no role record of its own.
    for record in records.values():
        if "ready_file" in record:
            try:
                payload_pid = int(Path(record["ready_file"]).read_text())
            except FileNotFoundError:
                pass
            else:
                add_existing_group(payload_pid, payload_pid)
    for group in process_groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=5)


def _run_launcher(
    tmp_path: Path, env: dict[str, str], *, terminal_fd: int | None = None
) -> subprocess.CompletedProcess[str]:
    command = ["bash", str(_LAUNCHER), str(tmp_path / "bin" / "python3"), "task"]
    if terminal_fd is not None:
        command = [
            sys.executable,
            "-c",
            "import fcntl, os, sys, termios; "
            "fcntl.ioctl(0, termios.TIOCSCTTY, 0); "
            "os.execvp(sys.argv[1], sys.argv[1:])",
            *command,
        ]
    with subprocess.Popen(  # nosec B603
        command,
        env=env,
        stdin=terminal_fd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            if env.get("LAUNCHER_TEST_OBSERVE_SERVER_EXIT") == "1":
                stdout_lines = []
                stderr_lines = []
                server_observed = threading.Event()

                def drain_stdout() -> None:
                    for line in process.stdout:
                        stdout_lines.append(line)

                def observe_server_exit() -> None:
                    for line in process.stderr:
                        stderr_lines.append(line)
                        if "MPI Comm server exited before the task" in line:
                            (tmp_path / "server-observed").touch()
                            server_observed.set()

                stdout_reader = threading.Thread(target=drain_stdout, daemon=True)
                observer = threading.Thread(target=observe_server_exit, daemon=True)
                stdout_reader.start()
                observer.start()
                process.wait(timeout=15)
                stdout_reader.join(timeout=1)
                observer.join(timeout=1)
                assert not stdout_reader.is_alive(), "launcher descendants kept stdout open"
                assert not observer.is_alive(), "launcher descendants kept stderr open"
                stdout = "".join(stdout_lines)
                stderr = "".join(stderr_lines)
                assert server_observed.is_set(), stderr
            else:
                stdout, stderr = process.communicate(timeout=15)
            # Registration precedes go release, so these PIDs also identify
            # gated payloads that never run the stand-in to write a role record.
            for role, guard_pid, payload_pid in re.findall(
                r"(server|task|stop) guard PID: (\d+); workload PGID: (\d+)", stderr
            ):
                record_path = tmp_path / "guards" / f"{role}.json"
                record = json.loads(record_path.read_text())
                assert record["pid"] == int(guard_pid)
                record["payload_pid"] = int(payload_pid)
                record_path.write_text(json.dumps(record))
            _assert_exited(tmp_path)
            return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        finally:
            # Clean up this test's groups even if a broken launcher times out.
            _cleanup_owned_processes(process, tmp_path)


def _assert_exited(tmp_path: Path, *, timeout: float = 1) -> None:
    deadline = time.monotonic() + timeout
    for role, record in _owned_process_records(tmp_path).items():
        while True:
            try:
                os.kill(record["pid"], 0)
            except ProcessLookupError:
                break
            stat_path = Path(f"/proc/{record['pid']}/stat")
            try:
                state = stat_path.read_text().rsplit(")", 1)[1].split()[0]
            except FileNotFoundError:
                pass
            else:
                if state == "Z":
                    # A group-wide KILL can leave an exited child awaiting
                    # init's reap; it is no longer a live descendant.
                    break
            if time.monotonic() >= deadline:
                pytest.fail(f"Launcher left {role} process {record['pid']} alive")
            time.sleep(0.01)


def _wait_for_readiness(process: subprocess.Popen, markers: list[Path]) -> None:
    deadline = time.monotonic() + 5
    while not all(marker.exists() for marker in markers):
        assert process.poll() is None, "Owner exited before child readiness"
        assert time.monotonic() < deadline, "Children did not report readiness"
        time.sleep(0.01)


@pytest.mark.parametrize("task_status", [0, 7])
def test_launcher_preserves_task_status_and_child_environment(
    tmp_path: Path, task_status: int
) -> None:
    env = _launcher_env(tmp_path, "clean")
    env["LAUNCHER_TEST_TASK_STATUS"] = str(task_status)
    result = _run_launcher(tmp_path, env)
    assert result.returncode == task_status, result.stderr
    records = _process_records(tmp_path)
    assert set(records) == {"task", "server", "stop"}
    assert records["server"]["pmi_rank"] == "0"
    for role in ("task", "stop"):
        assert records[role]["pmi_rank"] is None
    for record in records.values():
        assert not record["lock_fd_open"]
        assert record["workspace"].endswith("/rank-0")
        assert record["pid"] == record["pgid"]
    _assert_exited(tmp_path)


@pytest.mark.parametrize("mode", ["server_hangs", "stop_hangs"])
@pytest.mark.parametrize("task_status", [0, 7])
def test_launcher_deadline_covers_stop_helper_and_server(
    tmp_path: Path,
    mode: str,
    task_status: int,
) -> None:
    env = _launcher_env(tmp_path, mode)
    env["LAUNCHER_TEST_TASK_STATUS"] = str(task_status)
    result = _run_launcher(tmp_path, env)
    assert result.returncode == (task_status or 124), result.stderr
    assert "MPI Comm shutdown exceeded 3s" in result.stderr
    assert "stop" in _process_records(tmp_path)
    _assert_exited(tmp_path)


@pytest.mark.parametrize("task_status", [0, 7])
def test_launcher_exits_promptly_when_stop_helper_fails(tmp_path: Path, task_status: int) -> None:
    env = _launcher_env(tmp_path, "stop_fails")
    env["LAUNCHER_TEST_TASK_STATUS"] = str(task_status)
    # The outer timeout catches a launcher waiting for the server after its
    # helper has already failed without sending a stop request.
    env["TLLM_LLMAPI_LAUNCH_STOP_TIMEOUT"] = "120"
    result = _run_launcher(tmp_path, env)
    assert result.returncode == (task_status or 23), result.stderr
    assert "stop helper exit code: 23" in result.stderr
    assert "MPI Comm shutdown exceeded" not in result.stderr
    assert set(_process_records(tmp_path)) == {"task", "server", "stop"}
    _assert_exited(tmp_path)


@pytest.mark.parametrize("task_status", [0, 7])
def test_stop_guard_start_failure_preserves_task_status(tmp_path: Path, task_status: int) -> None:
    env = _launcher_env(tmp_path, "stop_guard_fails")
    env["LAUNCHER_TEST_TASK_STATUS"] = str(task_status)
    kill_trace = tmp_path / "kill-invocations"
    kill_trace.touch()
    bash_env = tmp_path / "trace-kill.bash"
    bash_env.write_text(r"""
kill() {
    case "$1" in
        -TERM|-KILL) printf '%s\n' "$*" >> "$LAUNCHER_TEST_ROOT/kill-invocations" ;;
    esac
    builtin kill "$@"
}
""")
    env["BASH_ENV"] = str(bash_env)
    result = _run_launcher(tmp_path, env)
    assert (tmp_path / "guard-start-failed").exists(), result.stderr
    assert result.returncode == (task_status or 23), result.stderr
    assert "stop" not in _process_records(tmp_path)
    finished_guard = _owned_process_records(tmp_path)["guard_stop"]["pid"]
    invocations = kill_trace.read_text().splitlines()
    assert invocations, "Expected launcher cleanup signals to be traced"
    for invocation in invocations:
        assert f"-{finished_guard}" not in invocation.split(), invocation


@pytest.mark.parametrize("gate", ["go", "reap"])
def test_failed_guard_acknowledgment_cleans_owned_processes(tmp_path: Path, gate: str) -> None:
    env = _launcher_env(tmp_path, f"guard_{gate}_fails")
    env["TLLM_LLMAPI_LAUNCH_TERM_GRACE_SECONDS"] = "1"
    result = _run_launcher(tmp_path, env)
    assert (tmp_path / "guard-write-fault").exists(), result.stderr
    assert result.returncode != 0, result.stderr
    assert "Is a directory" in result.stderr
    assert "payload_task" in _owned_process_records(tmp_path)
    if gate == "go":
        assert "task" not in _process_records(tmp_path)


def test_launcher_terminates_engine_and_child_when_server_exits(tmp_path: Path) -> None:
    env = _launcher_env(tmp_path, "server_exits")
    result = _run_launcher(tmp_path, env)
    assert result.returncode == 17, result.stderr
    assert "MPI Comm server exited before the task" in result.stderr
    assert "task_child" in _process_records(tmp_path)
    assert "stop" not in _process_records(tmp_path)
    _assert_exited(tmp_path)


@pytest.mark.parametrize("server_status", [0, 17])
@pytest.mark.parametrize("task_status", [0, 7, 143])
def test_server_first_exit_preserves_natural_task_completion(
    tmp_path: Path, server_status: int, task_status: int
) -> None:
    env = _launcher_env(tmp_path, "server_exits_naturally")
    env.update(
        {
            "LAUNCHER_TEST_SERVER_STATUS": str(server_status),
            "LAUNCHER_TEST_TASK_STATUS": str(task_status),
            "LAUNCHER_TEST_OBSERVE_SERVER_EXIT": "1",
            "TLLM_LLMAPI_LAUNCH_STOP_TIMEOUT": "3",
        }
    )
    result = _run_launcher(tmp_path, env)
    assert result.returncode == (task_status or server_status or 1), result.stderr
    assert (tmp_path / "task-completed-naturally").exists(), result.stderr
    assert not (tmp_path / "task-term").exists(), result.stderr
    assert "stop" not in _process_records(tmp_path)


@pytest.mark.parametrize("term_grace", [None, "8"], ids=["default", "configured"])
def test_server_first_timeout_allows_slow_term_cleanup(
    tmp_path: Path, term_grace: str | None
) -> None:
    env = _launcher_env(tmp_path, "server_exits_slow_cleanup")
    if term_grace is not None:
        env["TLLM_LLMAPI_LAUNCH_TERM_GRACE_SECONDS"] = term_grace
        env["LAUNCHER_TEST_CLEANUP_DELAY"] = "6"
    result = _run_launcher(tmp_path, env)
    assert result.returncode == 17, result.stderr
    assert "Task exit after MPI Comm server failure exceeded 3s" in result.stderr
    assert (tmp_path / "task-term").exists(), result.stderr
    assert (tmp_path / "task-cleanup-completed").exists(), result.stderr
    assert set(_process_records(tmp_path)) == {"server", "task", "task_child"}


@pytest.mark.parametrize("server_status", [0, 17])
def test_server_first_timeout_kills_term_ignoring_descendants(
    tmp_path: Path, server_status: int
) -> None:
    env = _launcher_env(tmp_path, "server_exits_ignores_term")
    env["LAUNCHER_TEST_SERVER_STATUS"] = str(server_status)
    env["TLLM_LLMAPI_LAUNCH_TERM_GRACE_SECONDS"] = "1"
    result = _run_launcher(tmp_path, env)
    assert result.returncode == (server_status or 1), result.stderr
    assert "Task exit after MPI Comm server failure exceeded 3s" in result.stderr
    assert (tmp_path / "task-term").exists(), result.stderr
    assert set(_process_records(tmp_path)) == {"server", "task", "task_child"}
    _assert_exited(tmp_path)


@pytest.mark.parametrize("during_stop", [False, True], ids=["active-task", "active-stop"])
def test_launcher_owner_death_reclaims_guards_and_term_resistant_children(
    tmp_path: Path, during_stop: bool
) -> None:
    mode = "owner_dies_during_stop" if during_stop else "owner_dies"
    env = _launcher_env(tmp_path, mode)
    env["TLLM_LLMAPI_LAUNCH_TERM_GRACE_SECONDS"] = "1"
    env["TLLM_LLMAPI_LAUNCH_STOP_TIMEOUT"] = "10"
    readiness = [tmp_path / "server-owner-ready"]
    readiness.append(tmp_path / ("stop-owner-ready" if during_stop else "task-owner-ready"))
    with (
        (tmp_path / "launcher.log").open("w") as log,
        subprocess.Popen(  # nosec B603
            ["bash", str(_LAUNCHER), str(tmp_path / "bin" / "python3"), "task"],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        ) as process,
    ):
        try:
            _wait_for_readiness(process, readiness)
            process.kill()
            assert process.wait(timeout=2) == -signal.SIGKILL
            # Kill only the shell owner. Its separate workload groups and
            # guards must leave without help from the test's final cleanup.
            _assert_exited(tmp_path, timeout=5)
            records = _owned_process_records(tmp_path)
            assert {"guard_server", "guard_task"} <= records.keys()
            if during_stop:
                assert {"guard_stop", "stop", "stop_child"} <= records.keys()
            else:
                assert "task_child" in records
                assert "stop" not in records
        finally:
            _cleanup_owned_processes(process, tmp_path)


@pytest.mark.parametrize("task_status", [0, 7])
def test_normal_task_exit_reclaims_descendant_and_preserves_status(
    tmp_path: Path, task_status: int
) -> None:
    env = _launcher_env(tmp_path, "task_leaves_child")
    env["LAUNCHER_TEST_TASK_STATUS"] = str(task_status)
    env["TLLM_LLMAPI_LAUNCH_TERM_GRACE_SECONDS"] = "1"
    result = _run_launcher(tmp_path, env)
    assert result.returncode == task_status, result.stderr
    assert "task_child" in _process_records(tmp_path)
    assert {"guard_server", "guard_task", "guard_stop"} <= _owned_process_records(tmp_path).keys()


@pytest.mark.parametrize("group_state", ["missing", "unconfirmed", "ready"])
def test_guard_cleanup_grace_requires_confirmed_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, group_state: str
) -> None:
    spec = importlib.util.spec_from_file_location("process_guard", _PROCESS_GUARD)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    parent_pid, child_pid = 123, 456
    startup_error = OSError("injected parent setpgid failure")
    process_ops = Mock(spec_set=os)
    process_ops.getppid.return_value = parent_pid
    process_ops.pipe.return_value = (3, 4)
    process_ops.fork.return_value = child_pid
    process_ops.waitpid.return_value = (child_pid, 0)
    process_ops.waitstatus_to_exitcode.return_value = 0
    if group_state != "ready":
        process_ops.setpgid.side_effect = startup_error
    if group_state == "missing":
        process_ops.killpg.side_effect = ProcessLookupError
    # Never let simulated process identities reach the host's process APIs.
    monkeypatch.setattr(guard, "os", process_ops)
    monkeypatch.setattr(
        guard,
        "signal",
        Mock(spec_set=signal, SIGTERM=signal.SIGTERM, SIGINT=signal.SIGINT, SIGKILL=signal.SIGKILL),
    )
    monkeypatch.setattr(guard, "_arm_parent_death_signal", lambda: None)
    monkeypatch.setattr(guard, "_child_finished", lambda _: True)
    # Unknown membership must retain grace only after successful group setup.
    monkeypatch.setattr(guard, "_group_has_live_members", lambda _: True)
    elapsed = 0.0

    def advance_clock(seconds: float) -> None:
        nonlocal elapsed
        elapsed += seconds

    monkeypatch.setattr(
        guard, "time", SimpleNamespace(monotonic=lambda: elapsed, sleep=advance_clock)
    )
    ready_file = tmp_path / "guard-ready"
    if group_state == "ready":
        assert guard._run(parent_pid, 5, None, None, "task", ["unused"]) == 0
        assert 5 <= elapsed < 5.1
        process_ops.kill.assert_not_called()
    else:
        with pytest.raises(OSError) as error:
            guard._run(parent_pid, 5, ready_file, tmp_path / "guard-go", "task", ["unused"])
        assert error.value is startup_error
        assert elapsed == 0, "Failed startup consumed the cleanup grace"
        process_ops.kill.assert_called_once_with(child_pid, signal.SIGKILL)
    assert not ready_file.exists()
    process_ops.write.assert_not_called()
    process_ops.close.assert_has_calls([call(3), call(4)])
    process_ops.killpg.assert_any_call(child_pid, signal.SIGKILL)
    process_ops.waitpid.assert_called_once_with(child_pid, 0)
    reap_index = process_ops.mock_calls.index(call.waitpid(child_pid, 0))
    assert all(
        index < reap_index
        for index, operation in enumerate(process_ops.mock_calls)
        if operation[0] in ("kill", "killpg")
    ), "Reaping must not release the child's identity before signaling finishes"


def test_owner_death_before_guard_registration_release_does_not_start_workload(
    tmp_path: Path,
) -> None:
    ready = tmp_path / "guard-ready"
    go = tmp_path / "guard-go"
    command_started = tmp_path / "command-started"
    owner_code = """
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
root = Path(sys.argv[1])
command = sys.argv[2:]
command[command.index("--parent-pid") + 1] = str(os.getpid())
guard = subprocess.Popen(command, start_new_session=True)
(root / "guard.json").write_text(json.dumps({"pid": guard.pid, "pgid": guard.pid}))
(root / "owner-ready").touch()
while True:
    signal.pause()
"""
    command = [
        sys.executable,
        "-c",
        owner_code,
        str(tmp_path),
        sys.executable,
        "-S",
        str(_PROCESS_GUARD),
        "--parent-pid",
        "0",
        "--term-grace",
        "1",
        "--ready-file",
        str(ready),
        "--go-file",
        str(go),
        "--role",
        "task",
        "--",
        sys.executable,
        "-c",
        "import pathlib, sys; pathlib.Path(sys.argv[1]).touch()",
        str(command_started),
    ]
    with (
        (tmp_path / "guard.log").open("w") as log,
        subprocess.Popen(  # nosec B603
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        ) as process,
    ):
        try:
            _wait_for_readiness(process, [tmp_path / "owner-ready", ready])
            payload_pid = int(ready.read_text())
            (tmp_path / "payload.json").write_text(
                json.dumps({"pid": payload_pid, "pgid": payload_pid})
            )
            assert not command_started.exists()
            process.kill()
            assert process.wait(timeout=2) == -signal.SIGKILL
            _assert_exited(tmp_path, timeout=5)
            assert not command_started.exists(), "Workload ran before launcher registration"
        finally:
            _cleanup_owned_processes(process, tmp_path)


@pytest.mark.parametrize("phase", ["cleanup", "task_wait", "shutdown"])
def test_phase_completion_survives_a_whole_second_boundary(tmp_path: Path, phase: str) -> None:
    mode = {
        "cleanup": "server_exits_slow_cleanup",
        "task_wait": "server_exits_naturally",
        "shutdown": "clean",
    }[phase]
    env = _launcher_env(tmp_path, mode)
    env["LAUNCHER_TEST_PHASE"] = phase
    env["TLLM_LLMAPI_LAUNCH_STOP_TIMEOUT"] = "1"
    env["TLLM_LLMAPI_LAUNCH_TERM_GRACE_SECONDS"] = "1"
    bash_env = tmp_path / "phase-sleep.bash"
    bash_env.write_text(_PHASE_SLEEP)
    env["BASH_ENV"] = str(bash_env)
    if phase == "task_wait":
        env["LAUNCHER_TEST_TASK_STATUS"] = "7"
        env["LAUNCHER_TEST_OBSERVE_SERVER_EXIT"] = "1"

    result = _run_launcher(tmp_path, env)

    assert (tmp_path / "phase-completed").exists(), result.stderr
    if phase == "cleanup":
        assert result.returncode == 17, result.stderr
        assert (tmp_path / "task-cleanup-completed").exists(), result.stderr
        assert "Task exit after MPI Comm server failure exceeded 1s" in result.stderr
    elif phase == "task_wait":
        assert result.returncode == 7, result.stderr
        assert (tmp_path / "task-completed-naturally").exists(), result.stderr
        assert not (tmp_path / "task-term").exists(), result.stderr
        assert "Task exit after MPI Comm server failure exceeded" not in result.stderr
        assert "stop" not in _process_records(tmp_path)
    else:
        assert result.returncode == 0, result.stderr
        assert "MPI Comm shutdown exceeded" not in result.stderr
        assert set(_process_records(tmp_path)) == {"task", "server", "stop"}


def test_launcher_task_sees_eof_with_terminal_stdin(tmp_path: Path) -> None:
    env = _launcher_env(tmp_path, "read_stdin")
    master_fd, slave_fd = pty.openpty()
    with os.fdopen(master_fd, "rb"), os.fdopen(slave_fd, "rb") as terminal:
        result = _run_launcher(tmp_path, env, terminal_fd=terminal.fileno())
    assert result.returncode == 0, result.stderr
    _assert_exited(tmp_path)


def test_follower_launcher_responds_to_sigterm(tmp_path: Path) -> None:
    env = _launcher_env(tmp_path, "follower")
    env["PMI_RANK"] = "1"
    with subprocess.Popen(  # nosec B603
        ["bash", str(_LAUNCHER), "/usr/bin/true"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    ) as process:
        try:
            deadline = time.monotonic() + 5
            while not (tmp_path / "worker-ready").exists():
                assert process.poll() is None, "Follower launcher exited before worker startup"
                assert time.monotonic() < deadline, "Follower worker did not start"
                time.sleep(0.01)
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=2) == -signal.SIGTERM
        finally:
            # The foreground worker belongs to this test's isolated session.
            _cleanup_owned_processes(process, tmp_path)


@pytest.mark.parametrize(
    "setting", ["TLLM_LLMAPI_LAUNCH_STOP_TIMEOUT", "TLLM_LLMAPI_LAUNCH_TERM_GRACE_SECONDS"]
)
@pytest.mark.parametrize("value", ["0", "-1", "invalid", "1.5", "9999999"])
def test_launcher_rejects_invalid_shutdown_deadline(
    tmp_path: Path, setting: str, value: str
) -> None:
    env = _launcher_env(tmp_path, "clean")
    env[setting] = value
    result = _run_launcher(tmp_path, env)
    assert result.returncode == 2, result.stderr
    assert f"{setting} must be a positive integer" in result.stderr
    assert not _process_records(tmp_path)
