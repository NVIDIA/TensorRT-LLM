# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the disaggregated leader's proxy ownership with model-free processes."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest

pytestmark = pytest.mark.cpu_only

_THIS_FILE = Path(__file__).resolve()
_PROCESS_GUARD = _THIS_FILE.parents[3] / "tensorrt_llm" / "llmapi" / "_llmapi_process_guard.py"


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def _record_process(directory: Path, role: str) -> None:
    _write_json(directory / f"{role}.pid.json", {"pid": os.getpid(), "pgid": os.getpgrp()})


def _wait_for_file(
    path: Path, timeout: float = 90, process: subprocess.Popen | None = None
) -> None:
    deadline = time.monotonic() + timeout
    while not path.is_file():
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Owner exited with {process.returncode} before {path.name}")
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Process did not publish {path.name}")
        time.sleep(0.02)


def _live(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        # Darwin has no /proc; ps also distinguishes an exited zombie from a survivor.
        result = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, timeout=2
        )
        state = result.stdout.strip()
    return bool(state) and not state.startswith(("Z", "X"))


def _records(directory: Path) -> dict[str, dict]:
    return {path.name: json.loads(path.read_text()) for path in directory.glob("*.pid.json")}


def _assert_exited(directory: Path) -> None:
    deadline = time.monotonic() + 15
    while True:
        survivors = {role: row for role, row in _records(directory).items() if _live(row["pid"])}
        if not survivors or time.monotonic() >= deadline:
            _write_json(directory / "survivors.json", survivors)
            assert not survivors, f"Disaggregated leader left live processes: {survivors}"
            return
        time.sleep(0.05)


def _cleanup(leader: subprocess.Popen, directory: Path) -> None:
    # This fallback runs only after the ownership assertions have observed the result.
    groups = {leader.pid} if leader.poll() is None else set()
    groups.update(row["pgid"] for row in _records(directory).values() if _live(row["pid"]))
    for group in groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            # On Darwin, a group that just became zombie-only can report EPERM.
            assert not any(
                row["pgid"] == group and _live(row["pid"]) for row in _records(directory).values()
            )
    leader.wait(timeout=5)


def _run_proxy(directory: Path, scenario: str) -> int:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    subprocess.Popen([sys.executable, str(_THIS_FILE), "descendant", str(directory), scenario])
    _wait_for_file(directory / "descendant.pid.json")
    _record_process(directory, "proxy")
    _write_json(
        directory / "proxy-env.json",
        {
            name: os.environ.get(name)
            for name in (
                "OMPI_H7_TEST_MARKER",
                "TLLM_DISAGG_INSTANCE_IDX",
                "TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT",
                "TRTLLM_NO_USAGE_STATS",
            )
        },
    )
    (directory / "proxy-ready").touch()
    if scenario.startswith("proxy_exit_"):
        return int(scenario.rsplit("_", 1)[1])
    time.sleep(180)
    return 99


def _run_leader(serve: ModuleType, directory: Path, scenario: str) -> int:
    from tensorrt_llm.llmapi import mgmn_leader_node

    original_popen = subprocess.Popen
    os.environ["OMPI_H7_TEST_MARKER"] = "remove-before-proxy-exec"
    os.environ["TRTLLM_NO_USAGE_STATS"] = "1"
    serve._child_p_global = None

    def launch_proxy(command: list[str], **kwargs: object) -> subprocess.Popen:
        # Keep the real guard command, environment, session and stream wiring.
        # Only replace the model-serving payload with a CPU-only stand-in.
        index = command.index("disaggregated_mpi_worker")
        proxy_command = command[: index - 2] + [
            sys.executable,
            str(_THIS_FILE),
            "proxy",
            str(directory),
            scenario,
        ]
        child = original_popen(proxy_command, **kwargs)
        _write_json(directory / "owner-child.pid.json", {"pid": child.pid, "pgid": child.pid})
        _write_json(
            directory / "launch.json",
            {
                "command": command,
                "start_new_session": kwargs["start_new_session"],
                "stdout_inherited": kwargs["stdout"] is sys.stdout,
                "stderr_inherited": kwargs["stderr"] is sys.stderr,
            },
        )
        return child

    def run_server(_comm: object) -> None:
        _wait_for_file(directory / "proxy-ready")
        (directory / "server-ready").touch()
        if scenario == "abort":
            os._exit(23)
        if scenario == "exception":
            raise RuntimeError("original disaggregated server failure")
        if scenario in ("sigterm", "sigint"):
            os.kill(os.getpid(), signal.SIGTERM if scenario == "sigterm" else signal.SIGINT)
        if scenario == "owner_sigkill":
            time.sleep(180)
        if scenario.startswith("proxy_exit_"):
            serve._child_p_global.wait(timeout=20)

    serve.subprocess.Popen = launch_proxy
    mgmn_leader_node.launch_server_main = run_server
    code = 0
    error = None
    try:
        serve._launch_disaggregated_leader(
            SimpleNamespace(Get_rank=lambda: 0), 2, "fake.yaml", "info"
        )
    except serve._command_telemetry.SignalExit as caught:
        code = caught.code
        error = str(caught)
    except RuntimeError as caught:
        code = 42
        error = str(caught)
    _write_json(
        directory / "leader-result.json",
        {"code": code, "error": error, "proxy_status": serve._child_p_global.poll()},
    )
    return code


@pytest.mark.parametrize(
    ("scenario", "expected_status"),
    [
        ("abort", 23),
        ("owner_sigkill", -signal.SIGKILL),
        ("exception", 42),
        ("sigterm", 143),
        ("sigint", 130),
        ("return", 0),
        ("proxy_exit_0", 0),
        ("proxy_exit_7", 0),
    ],
)
def test_disaggregated_leader_reclaims_proxy_and_descendant(
    tmp_path: Path, scenario: str, expected_status: int
) -> None:
    """Both returning cleanup and nonreturning owner death reclaim the proxy group."""
    with (tmp_path / "leader.log").open("w") as log:
        leader = subprocess.Popen(
            [sys.executable, str(_THIS_FILE), "leader", str(tmp_path), scenario],
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        try:
            _wait_for_file(tmp_path / "server-ready", process=leader)
            if scenario == "owner_sigkill":
                leader.kill()
            status = leader.wait(timeout=45)
            assert status == expected_status, (tmp_path / "leader.log").read_text()
            _assert_exited(tmp_path)
            records = _records(tmp_path)
            assert {
                "proxy.pid.json",
                "descendant.pid.json",
                "owner-child.pid.json",
            } <= records.keys()
            assert records["proxy.pid.json"]["pgid"] == records["descendant.pid.json"]["pgid"]
            assert records["proxy.pid.json"]["pid"] != records["owner-child.pid.json"]["pid"]
            launch = json.loads((tmp_path / "launch.json").read_text())
            assert (
                launch["start_new_session"]
                and launch["stdout_inherited"]
                and launch["stderr_inherited"]
            )
            assert "--autonomous" in launch["command"]
            assert json.loads((tmp_path / "proxy-env.json").read_text()) == {
                "OMPI_H7_TEST_MARKER": None,
                "TLLM_DISAGG_INSTANCE_IDX": "2",
                "TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT": "1",
                "TRTLLM_NO_USAGE_STATS": "1",
            }
            if scenario == "exception":
                assert json.loads((tmp_path / "leader-result.json").read_text())["error"] == (
                    "original disaggregated server failure"
                )
            if scenario.startswith("proxy_exit_"):
                assert json.loads((tmp_path / "leader-result.json").read_text())[
                    "proxy_status"
                ] == int(scenario.rsplit("_", 1)[1])
        finally:
            _cleanup(leader, tmp_path)


def test_autonomous_guard_does_not_start_payload_for_missing_owner(tmp_path: Path) -> None:
    marker = tmp_path / "payload-started"
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(_PROCESS_GUARD),
            "--parent-pid",
            str(os.getpid() + 1),
            "--term-grace",
            "1",
            "--autonomous",
            "--role",
            "proxy",
            "--",
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; Path(sys.argv[1]).touch()",
            str(marker),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 143, result.stderr
    assert not marker.exists()


@pytest.mark.parametrize("outcome", ["return", "exception", "signal"])
def test_disaggregated_leader_cleanup_timeout_retains_guard(
    monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    """A delayed guard remains alive to finish cleanup without masking the server error."""
    from tensorrt_llm.commands import serve
    from tensorrt_llm.llmapi import mgmn_leader_node

    child = mock.MagicMock(pid=12345)
    child.poll.return_value = None
    child.wait.side_effect = subprocess.TimeoutExpired("process-guard", 30)
    original_error = (
        serve._command_telemetry.SignalExit(signal.SIGTERM)
        if outcome == "signal"
        else RuntimeError("original server failure")
    )

    def run_server(_comm: object) -> None:
        if outcome != "return":
            raise original_error

    monkeypatch.setattr(serve, "_child_p_global", None)
    monkeypatch.setattr(serve, "find_free_ipc_addr", lambda: "ipc://fake-proxy")
    monkeypatch.setattr(serve.subprocess, "Popen", mock.Mock(return_value=child))
    monkeypatch.setattr(mgmn_leader_node, "launch_server_main", run_server)
    if outcome == "return":
        with pytest.raises(RuntimeError, match="cleanup is still pending after 30s"):
            serve._launch_disaggregated_leader(
                SimpleNamespace(Get_rank=lambda: 0), 2, "fake", "info"
            )
    else:
        with pytest.raises(type(original_error)) as caught:
            serve._launch_disaggregated_leader(
                SimpleNamespace(Get_rank=lambda: 0), 2, "fake", "info"
            )
        assert caught.value is original_error
    child.terminate.assert_called_once_with()
    child.wait.assert_called_once_with(timeout=30)
    child.kill.assert_not_called()


if __name__ == "__main__":
    mode, directory, scenario = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    if mode == "descendant":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        _record_process(directory, "descendant")
        time.sleep(180)
    elif mode == "proxy":
        sys.exit(_run_proxy(directory, scenario))
    elif mode == "leader":
        from tensorrt_llm.commands import serve

        sys.exit(_run_leader(serve, directory, scenario))
