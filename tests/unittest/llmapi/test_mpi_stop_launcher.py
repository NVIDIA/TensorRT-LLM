# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test stop-helper routing and launcher status propagation without MPI imports."""

import fcntl
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
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

_FAKE_PYTHON = r'''
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

args = sys.argv[1:]
root = Path(os.environ["LAUNCHER_TEST_ROOT"])
if args[0] in ("-S", "-c"):
    if args[0] == "-S" and "--role" in args:
        role = args[args.index("--role") + 1]
        (root / f"guard-{role}.json").write_text(json.dumps({
            "pid": os.getpid(),
            "path": args[1],
        }))
    if args[-1] == os.environ["LAUNCHER_TEST_LAUNCHER"]:
        assert args[:2] == ["-S", "-c"]
        (root / "resolver.json").write_text(json.dumps(args))
        if os.environ.get("LAUNCHER_TEST_RESOLVER_FAIL"):
            sys.exit(17)
    os.execv(sys.executable, [sys.executable, *args])

if args == ["task"]:
    role = "task"
elif args == ["-m", "tensorrt_llm.llmapi.mgmn_leader_node"]:
    role = "server"
elif args == ["-m", "tensorrt_llm.llmapi.mgmn_leader_node", "--action", "stop"]:
    role = "stop"
elif len(args) == 1 and Path(args[0]).resolve() == Path(os.environ["LAUNCHER_TEST_HELPER"]):
    role = "stop"
else:
    raise AssertionError(f"Unexpected Python invocation: {args}")

try:
    os.fstat(200)
    lock_fd_open = True
except OSError:
    lock_fd_open = False

workspace = Path(os.environ["FLASHINFER_WORKSPACE_BASE"])
with (workspace / ".lock").open("a") as stream:
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        workspace_locked = False
    except BlockingIOError:
        workspace_locked = True

(root / f"{role}.json").write_text(json.dumps({
    "args": args,
    "pid": os.getpid(),
    "mpi_env": {k: v for k, v in os.environ.items() if k.startswith(PREFIXES)},
    "key": os.environ["TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY"],
    "workspace": str(workspace),
    "cwd": os.getcwd(),
    "lock_fd_open": lock_fd_open,
    "workspace_locked": workspace_locked,
}))

def wait_for(name: str) -> None:
    """Wait for a peer event with a bounded test-only deadline."""
    deadline = time.monotonic() + 8
    while not (root / name).exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Missing launcher event: {name}")
        time.sleep(0.01)

def server_is_alive() -> bool:
    """Exclude the exited payload zombie retained by its real process guard."""
    server_pid = json.loads((root / "server.json").read_text())["pid"]
    result = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(server_pid)],
        capture_output=True, text=True, check=False, timeout=1,
    )
    assert result.returncode in (0, 1), result.stderr
    state = result.stdout.strip()
    return bool(state) and not state.startswith(("Z", "X"))

if role == "server":
    (root / "server-ready").touch()
    if not os.environ.get("LAUNCHER_TEST_SERVER_EARLY_EXIT"):
        wait_for("server-release" if os.environ.get("LAUNCHER_TEST_KEEP_SERVER_ALIVE")
                 else "stop-requested")
elif role == "task":
    wait_for("server-ready")
    if os.environ.get("LAUNCHER_TEST_SERVER_EARLY_EXIT"):
        deadline = time.monotonic() + 8
        while server_is_alive():
            if time.monotonic() >= deadline:
                raise TimeoutError("Server did not exit before task completion")
            time.sleep(0.01)
        (root / "server-exit-observed").touch()
    (root / "task-completed").touch()
else:
    assert (root / "task-completed").exists(), "stop ran before task completion"
    if os.environ.get("LAUNCHER_TEST_SERVER_EARLY_EXIT"):
        assert (root / "server-exit-observed").exists()
        assert not server_is_alive(), "stop helper ran before server process exit"
    with (root / "stop-invocations.jsonl").open("a") as stream:
        stream.write(json.dumps(args) + "\n")
    # Release the fake server even when testing a nonzero stop exit status.
    (root / "stop-requested").touch()
    if len(args) == 1 and os.environ.get("LAUNCHER_TEST_REAL_HELPER"):
        if os.environ.get("LAUNCHER_TEST_SHORT_HELPER_RETRY"):
            command = """
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("standalone_stop", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.exit(module.main(ack_timeout=0.1, max_attempts=2))
"""
            os.execv(sys.executable, [sys.executable, "-c", command, *args])
        os.execv(sys.executable, [sys.executable, *args])

sys.exit(int(os.environ[f"LAUNCHER_TEST_{role.upper()}_STATUS"]))
'''


def _launcher_environment(tmp_path: Path, helper: Path) -> dict[str, str]:
    """Create a fake Python router and a deterministic, validated key generator.

    Returns:
        An isolated environment that exercises real workspace locking.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_python = tmp_path / "fake_python.py"
    fake_python.write_text(f"PREFIXES = {_MPI_PREFIXES!r}\n" + _FAKE_PYTHON, encoding="utf-8")
    python_command = bin_dir / "python3"
    python_command.write_text(
        f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(fake_python))} "$@"\n',
        encoding="utf-8",
    )
    python_command.chmod(0o755)
    openssl_command = bin_dir / "openssl"
    openssl_command.write_text(
        "#!/bin/sh\n"
        '[ "$*" = "rand -hex 32" ] || exit 96\n'
        '[ -z "${TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY+x}" ] || exit 97\n'
        'printf "%s\\n" "$*" > "$LAUNCHER_TEST_ROOT/key-generated"\n'
        f"printf '%s\\n' {'01' * 32}\n",
        encoding="utf-8",
    )
    openssl_command.chmod(0o755)
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', os.defpath)}",
        "HOME": str(tmp_path / "home"),
        "TMPDIR": str(tmp_path),
        "PMI_RANK": "0",
        "TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR": f"ipc://{tmp_path / 'server'}",
        "TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY": "stale-inherited-key",
        "LAUNCHER_TEST_ROOT": str(tmp_path),
        "LAUNCHER_TEST_HELPER": str(helper.resolve()),
        "TLLM_LLMAPI_LAUNCH_STOP_TIMEOUT": "3",
        "TLLM_LLMAPI_LAUNCH_TERM_GRACE_SECONDS": "1",
    }
    if "LD_LIBRARY_PATH" in os.environ:
        env["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
    env.update({f"{prefix}TEST": "inherited" for prefix in _MPI_PREFIXES})
    import_traps = tmp_path / "import traps"
    import_traps.mkdir()
    for name in ("tensorrt_llm", "torch", "mpi4py"):
        (import_traps / f"{name}.py").write_text(
            f"raise AssertionError('launcher resolver imported {name}')\n", encoding="utf-8"
        )
    env["PYTHONPATH"] = str(import_traps)
    return env


def _installed_launcher(tmp_path: Path) -> Path:
    """Copy the launcher and both sibling helpers into a path containing spaces."""
    install_dir = tmp_path / "installed scripts"
    install_dir.mkdir()
    launcher = install_dir / _LAUNCHER.name
    shutil.copy2(_LAUNCHER, launcher)
    shutil.copy2(_LAUNCHER.with_name("_llmapi_mpi_stop.py"), install_dir)
    shutil.copy2(_LAUNCHER.with_name("_llmapi_process_guard.py"), install_dir)
    return launcher


def _run_launcher(
    tmp_path: Path,
    launcher: Path,
    env: dict[str, str],
    task_command: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the launcher with bounded child cleanup and the real stdlib resolver."""
    env["LAUNCHER_TEST_LAUNCHER"] = str(launcher)
    for role in ("TASK", "STOP", "SERVER"):
        env.setdefault(f"LAUNCHER_TEST_{role}_STATUS", "0")
    with subprocess.Popen(
        ["/bin/bash", str(launcher), *(task_command or ["python3", "task"])],
        env=env,
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=15)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.communicate(timeout=5)
    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)


def _assert_launcher_children(
    tmp_path: Path, env: dict[str, str], stop_args: list[str] | None
) -> None:
    """Check child credentials, environment, lock ownership and exact stop routing."""
    assert (tmp_path / "key-generated").read_text().strip() == "rand -hex 32"
    roles = ("task", "stop", "server") if stop_args is not None else ("task", "server")
    if stop_args is not None:
        assert (tmp_path / "stop-requested").exists()
        invocations = (tmp_path / "stop-invocations.jsonl").read_text().splitlines()
        assert [json.loads(line) for line in invocations] == [stop_args]
    else:
        assert not (tmp_path / "stop-invocations.jsonl").exists()
        assert not (tmp_path / "stop.json").exists()
        assert not (tmp_path / "guard-stop.json").exists()
    workspace = tmp_path / "home" / ".cache" / "tensorrt_llm" / "flashinfer" / "rank-0"
    for role in roles:
        record = json.loads((tmp_path / f"{role}.json").read_text())
        assert record["key"] == "01" * 32
        assert record["workspace"] == str(workspace)
        assert record["workspace_locked"]
        assert not record["lock_fd_open"]
        expected_env = {k: v for k, v in env.items() if k.startswith(_MPI_PREFIXES)}
        assert record["mpi_env"] == (expected_env if role == "server" else {})
        if role == "stop":
            assert record["args"] == stop_args
        guard = json.loads((tmp_path / f"guard-{role}.json").read_text())
        assert Path(guard["path"]) == Path(env["LAUNCHER_TEST_HELPER"]).with_name(
            "_llmapi_process_guard.py"
        )
        for pid in (record["pid"], guard["pid"]):
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
    assert not list(tmp_path.glob("trtllm-launch-guards.*"))
    with (workspace / ".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize("installed", [False, True], ids=["source", "installed"])
@pytest.mark.parametrize(
    ("task_status", "stop_status", "server_status", "expected_status"),
    [
        pytest.param(0, 0, 0, 0, id="success"),
        pytest.param(7, 0, 0, 7, id="task-failure"),
        pytest.param(0, 11, 0, 11, id="stop-failure"),
        pytest.param(0, 0, 13, 13, id="server-failure"),
        pytest.param(7, 11, 13, 7, id="task-precedence"),
        pytest.param(0, 11, 13, 11, id="stop-precedence"),
        pytest.param(7, 0, 13, 7, id="task-over-server"),
    ],
)
def test_mpi_stop_launcher_routing_and_status(
    tmp_path: Path,
    installed: bool,
    task_status: int,
    stop_status: int,
    server_status: int,
    expected_status: int,
) -> None:
    """Preserve task/stop/server precedence and route stop to the sibling script."""
    launcher = _installed_launcher(tmp_path) if installed else _LAUNCHER
    helper = launcher.with_name("_llmapi_mpi_stop.py")
    assert helper.is_file()
    env = _launcher_environment(tmp_path, helper)
    env.update(
        {
            "LAUNCHER_TEST_TASK_STATUS": str(task_status),
            "LAUNCHER_TEST_STOP_STATUS": str(stop_status),
            "LAUNCHER_TEST_SERVER_STATUS": str(server_status),
        }
    )

    result = _run_launcher(tmp_path, launcher, env)
    assert result.returncode == expected_status, result.stdout + result.stderr
    assert "falling back" not in result.stderr
    _assert_launcher_children(tmp_path, env, [str(helper.resolve())])


@pytest.mark.parametrize("installed", [False, True], ids=["source", "installed"])
@pytest.mark.parametrize("link_kind", ["relative", "absolute", "chained"])
def test_mpi_stop_launcher_resolves_symlinks(
    tmp_path: Path, installed: bool, link_kind: str
) -> None:
    """Find the real launcher's sibling through links with spaces in their paths."""
    launcher = _installed_launcher(tmp_path) if installed else _LAUNCHER
    helper = launcher.with_name("_llmapi_mpi_stop.py")
    link_dir = tmp_path / "launcher links"
    link_dir.mkdir()
    link = link_dir / "launch alias"
    if link_kind == "absolute":
        link.symlink_to(launcher)
    else:
        target = launcher
        if link_kind == "chained":
            intermediate = tmp_path / "intermediate link"
            intermediate.symlink_to(launcher)
            target = intermediate
        link.symlink_to(os.path.relpath(target, link_dir))
    # An unrelated sibling at the alias location must not win over the real one.
    (link_dir / helper.name).write_text("raise AssertionError('wrong helper')\n")
    env = _launcher_environment(tmp_path, helper)
    result = _run_launcher(tmp_path, link.relative_to(tmp_path), env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "falling back" not in result.stderr
    assert (tmp_path / "resolver.json").exists()
    _assert_launcher_children(tmp_path, env, [str(helper.resolve())])


def test_mpi_stop_launcher_resolves_with_task_in_another_directory(tmp_path: Path) -> None:
    """A guarded shell task keeps its cwd separate from sibling helper lookup."""
    launcher = _installed_launcher(tmp_path)
    helper = launcher.with_name("_llmapi_mpi_stop.py")
    task_dir = tmp_path / "task directory"
    task_dir.mkdir()
    env = _launcher_environment(tmp_path, helper)
    result = _run_launcher(
        tmp_path,
        launcher.relative_to(tmp_path),
        env,
        ["/bin/sh", "-c", f"cd {shlex.quote(str(task_dir))} && exec python3 task"],
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "falling back" not in result.stderr
    assert json.loads((tmp_path / "task.json").read_text())["cwd"] == str(task_dir)
    assert json.loads((tmp_path / "stop.json").read_text())["cwd"] == str(tmp_path)
    _assert_launcher_children(tmp_path, env, [str(helper.resolve())])


@pytest.mark.parametrize("failure", ["missing-helper", "resolver-failure"])
@pytest.mark.parametrize(
    ("task_status", "stop_status", "server_status", "expected_status"),
    [
        pytest.param(0, 0, 0, 0, id="success"),
        pytest.param(7, 11, 13, 7, id="task-precedence"),
        pytest.param(0, 11, 13, 11, id="stop-precedence"),
        pytest.param(0, 0, 13, 13, id="server-failure"),
    ],
)
def test_mpi_stop_launcher_falls_back_when_helper_unavailable(
    tmp_path: Path,
    failure: str,
    task_status: int,
    stop_status: int,
    server_status: int,
    expected_status: int,
) -> None:
    """A lookup failure uses the canonical stop command with unchanged precedence."""
    launcher = _installed_launcher(tmp_path)
    helper = launcher.with_name("_llmapi_mpi_stop.py")
    env = _launcher_environment(tmp_path, helper)
    if failure == "missing-helper":
        helper.unlink()
    else:
        env["LAUNCHER_TEST_RESOLVER_FAIL"] = "1"
    env.update(
        {
            "LAUNCHER_TEST_TASK_STATUS": str(task_status),
            "LAUNCHER_TEST_STOP_STATUS": str(stop_status),
            "LAUNCHER_TEST_SERVER_STATUS": str(server_status),
        }
    )
    result = _run_launcher(tmp_path, launcher, env)

    assert result.returncode == expected_status, result.stdout + result.stderr
    assert "falling back to the TensorRT-LLM module stop command" in result.stderr
    _assert_launcher_children(
        tmp_path, env, ["-m", "tensorrt_llm.llmapi.mgmn_leader_node", "--action", "stop"]
    )


@pytest.mark.parametrize("failure", ["transport", "import", "no-peer"])
def test_mpi_stop_launcher_does_not_fall_back_on_helper_failure(
    tmp_path: Path, failure: str
) -> None:
    """A real helper or dependency failure must propagate instead of retrying via imports."""
    launcher = _installed_launcher(tmp_path)
    helper = launcher.with_name("_llmapi_mpi_stop.py")
    env = _launcher_environment(tmp_path, helper)
    env["LAUNCHER_TEST_REAL_HELPER"] = "1"
    if failure == "transport":
        env["TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR"] = "invalid://test-endpoint"
        expected_error = "Failed to send MPI Comm server shutdown"
    elif failure == "import":
        (Path(env["PYTHONPATH"]) / "zmq.py").write_text(
            "raise ImportError('test dependency unavailable')\n", encoding="utf-8"
        )
        expected_error = "ImportError: test dependency unavailable"
    else:
        env["LAUNCHER_TEST_SHORT_HELPER_RETRY"] = "1"
        env["LAUNCHER_TEST_KEEP_SERVER_ALIVE"] = "1"
        expected_error = "no acknowledgment after 2 attempts"
    with socket.socket() as unavailable_peer:
        if failure == "no-peer":
            unavailable_peer.bind(("127.0.0.1", 0))
            env["TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR"] = (
                f"tcp://127.0.0.1:{unavailable_peer.getsockname()[1]}"
            )
        result = _run_launcher(tmp_path, launcher, env)

    assert result.returncode == 1, result.stdout + result.stderr
    assert expected_error in result.stderr
    assert "falling back" not in result.stderr
    _assert_launcher_children(tmp_path, env, [str(helper.resolve())])


@pytest.mark.parametrize(
    ("task_status", "expected_status"),
    [pytest.param(0, 13, id="server-failure"), pytest.param(7, 7, id="task-precedence")],
)
def test_mpi_stop_launcher_preserves_status_after_server_exits(
    tmp_path: Path, task_status: int, expected_status: int
) -> None:
    """An already-exited server needs no stop helper; task failure keeps priority."""
    launcher = _installed_launcher(tmp_path)
    helper = launcher.with_name("_llmapi_mpi_stop.py")
    env = _launcher_environment(tmp_path, helper)
    env.update(
        {
            "LAUNCHER_TEST_TASK_STATUS": str(task_status),
            "LAUNCHER_TEST_SERVER_STATUS": "13",
            "LAUNCHER_TEST_SERVER_EARLY_EXIT": "1",
        }
    )
    result = _run_launcher(tmp_path, launcher, env)

    assert result.returncode == expected_status, result.stdout + result.stderr
    assert (tmp_path / "server-exit-observed").exists()
    assert "no acknowledgment" not in result.stderr
    assert f"Rank0 Task exit code: {task_status}" in result.stderr
    assert "MPI Comm server exited before the task (status 13)" in result.stderr or (
        "Rank0 MPI Comm server exit code: 13" in result.stderr
    )
    assert "falling back" not in result.stderr
    _assert_launcher_children(tmp_path, env, None)
