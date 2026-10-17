# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test stop-helper routing and launcher status propagation without MPI imports."""

import fcntl
import json
import os
import shlex
import shutil
import signal
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
import sys
import time
from pathlib import Path

args = sys.argv[1:]
if args[0] in ("-S", "-c"):
    os.execv(sys.executable, [sys.executable, *args])

root = Path(os.environ["LAUNCHER_TEST_ROOT"])
if args == ["task"]:
    role = "task"
elif args == ["-m", "tensorrt_llm.llmapi.mgmn_leader_node"]:
    role = "server"
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
    "mpi_env": {k: v for k, v in os.environ.items() if k.startswith(PREFIXES)},
    "key": os.environ["TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY"],
    "workspace": str(workspace),
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

if role == "server":
    (root / "server-ready").touch()
    wait_for("stop-requested")
elif role == "task":
    wait_for("server-ready")
    (root / "task-completed").touch()
else:
    assert (root / "task-completed").exists(), "stop ran before task completion"
    # Release the fake server even when testing a nonzero stop exit status.
    (root / "stop-requested").touch()

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
    }
    if "LD_LIBRARY_PATH" in os.environ:
        env["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
    env.update({f"{prefix}TEST": "inherited" for prefix in _MPI_PREFIXES})
    return env


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
    launcher = _LAUNCHER
    if installed:
        install_dir = tmp_path / "installed scripts"
        install_dir.mkdir()
        launcher = install_dir / _LAUNCHER.name
        shutil.copy2(_LAUNCHER, launcher)
        shutil.copy2(_LAUNCHER.with_name("_llmapi_mpi_stop.py"), install_dir)
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

    with subprocess.Popen(
        ["/bin/bash", str(launcher), "python3", "task"],
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
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate(timeout=5)

    assert process.returncode == expected_status, stdout + stderr
    assert (tmp_path / "key-generated").read_text().strip() == "rand -hex 32"
    assert (tmp_path / "stop-requested").exists()
    workspace = tmp_path / "home" / ".cache" / "tensorrt_llm" / "flashinfer" / "rank-0"
    for role in ("task", "stop", "server"):
        record = json.loads((tmp_path / f"{role}.json").read_text())
        assert record["key"] == "01" * 32
        assert record["workspace"] == str(workspace)
        assert record["workspace_locked"]
        assert not record["lock_fd_open"]
        expected_env = {k: v for k, v in env.items() if k.startswith(_MPI_PREFIXES)}
        assert record["mpi_env"] == (expected_env if role == "server" else {})
        if role == "stop":
            assert record["args"] == [str(helper.resolve())]
    with (workspace / ".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
