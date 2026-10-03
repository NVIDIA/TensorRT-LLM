# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise launcher supervision without importing TensorRT-LLM or initializing MPI."""

import json
import os
import pty
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_LAUNCHER = Path(__file__).parents[3] / "tensorrt_llm" / "llmapi" / "trtllm-llmapi-launch"
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
from pathlib import Path

root = Path(os.environ["LAUNCHER_TEST_ROOT"])
mode = os.environ["LAUNCHER_TEST_MODE"]
role = "task" if sys.argv[1] == "task" else "server"
if "--action" in sys.argv:
    role = "stop"
if sys.argv[1] == "-S":
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

if role == "server":
    if mode == "follower":
        (root / "worker-ready").touch()
        while True:
            signal.pause()
    send("server_ready")
    receive("task_ready")
    if mode == "server_exits":
        sys.exit(17)
    receive("stop_requested")
    if mode == "server_hangs":
        hang()
elif role == "stop":
    if mode == "stop_hangs":
        hang()
    if mode == "stop_fails":
        sys.exit(23)
    send("stop_requested")
else:
    receive("server_ready")
    if mode == "read_stdin":
        assert sys.stdin.read() == ""
    if mode == "server_exits":
        child = subprocess.Popen([
            sys.executable, "-c",
            "import signal; print('ready', flush=True); signal.pause()",
        ], stdout=subprocess.PIPE, text=True)
        assert child.stdout.readline() == "ready\n"
        (root / "task_child.json").write_text(json.dumps({
            "pid": child.pid,
            "pgid": os.getpgid(child.pid),
        }))
        def terminate(signum, frame):
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
        and not name.startswith(("FLASHINFER_", "TRTLLM_FLASHINFER_", "TLLM_SPAWN_PROXY_PROCESS"))
    }
    env.update(
        {
            "HOME": str(tmp_path / "home"),
            "PATH": f"{stub_bin}{os.pathsep}{env['PATH']}",
            "PMI_RANK": "0",
            "PMI_SIZE": "1",
            "TLLM_LLMAPI_LAUNCH_STOP_TIMEOUT": "1",
            "TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR": f"ipc://{tmp_path / 'ipc'}",
            "LAUNCHER_TEST_ROOT": str(tmp_path),
            "LAUNCHER_TEST_MODE": mode,
        }
    )
    return env


def _process_records(tmp_path: Path) -> dict[str, dict]:
    return {path.stem: json.loads(path.read_text()) for path in tmp_path.glob("*.json")}


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
            stdout, stderr = process.communicate(timeout=12)
            _assert_exited(tmp_path)
            return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        finally:
            # Clean up this test's groups even if a broken launcher times out.
            process_groups = {process.pid}
            process_groups.update(record["pgid"] for record in _process_records(tmp_path).values())
            for group in process_groups:
                try:
                    os.killpg(group, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait(timeout=5)


def _assert_exited(tmp_path: Path) -> None:
    for role, record in _process_records(tmp_path).items():
        try:
            os.kill(record["pid"], 0)
        except ProcessLookupError:
            continue
        pytest.fail(f"Launcher left {role} process {record['pid']} alive")


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
    assert "MPI Comm shutdown exceeded 1s" in result.stderr
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


def test_launcher_terminates_engine_and_child_when_server_exits(tmp_path: Path) -> None:
    env = _launcher_env(tmp_path, "server_exits")
    result = _run_launcher(tmp_path, env)
    assert result.returncode == 17, result.stderr
    assert "MPI Comm server exited before the task" in result.stderr
    assert "task_child" in _process_records(tmp_path)
    assert "stop" not in _process_records(tmp_path)
    _assert_exited(tmp_path)


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
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)


@pytest.mark.parametrize("value", ["0", "-1", "invalid", "1.5", "9999999"])
def test_launcher_rejects_invalid_shutdown_deadline(tmp_path: Path, value: str) -> None:
    env = _launcher_env(tmp_path, "clean")
    env["TLLM_LLMAPI_LAUNCH_STOP_TIMEOUT"] = value
    result = _run_launcher(tmp_path, env)
    assert result.returncode == 2, result.stderr
    assert "TLLM_LLMAPI_LAUNCH_STOP_TIMEOUT must be a positive integer" in result.stderr
    assert not _process_records(tmp_path)
