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
"""Subprocess coverage for terminal-signal telemetry boundaries."""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TRTLLM_PACKAGE = _REPO_ROOT / "tensorrt_llm"

pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="The test verifies POSIX SIGTERM process semantics.",
)


def _wait_for_path(path: Path, process: subprocess.Popen, timeout: float = 10.0) -> None:
    """Wait for a child-process marker while detecting early termination."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(
                f"Signal-test child exited early with {process.returncode}.\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}"
            )
        time.sleep(0.01)
    pytest.fail(f"Timed out waiting for child marker: {path}")


@contextmanager
def _signal_test_child(mode: str, *paths: Path) -> Iterator[subprocess.Popen]:
    """Launch and reliably clean up an enabled telemetry test child."""
    env = os.environ.copy()
    for name in ("TRTLLM_NO_USAGE_STATS", "TELEMETRY_DISABLED", "DO_NOT_TRACK"):
        env.pop(name, None)
    env["TRTLLM_USAGE_FORCE_ENABLED"] = "1"
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), mode, *(str(path) for path in paths)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        yield process
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def _read_payloads(payload_path: Path) -> list[dict]:
    return [json.loads(line) for line in payload_path.read_text(encoding="utf-8").splitlines()]


def _run_uvicorn_case(
    tmp_path: Path,
    *,
    mode: str = "uvicorn",
    signal_to_send: int = 0,
    signal_count: int = 1,
) -> tuple[int, dict, str, str]:
    ready_path = tmp_path / "ready"
    shutdown_path = tmp_path / "shutdown"
    release_path = tmp_path / "release"
    payload_path = tmp_path / "payloads.jsonl"
    with _signal_test_child(mode, ready_path, shutdown_path, release_path, payload_path) as process:
        _wait_for_path(ready_path, process)
        if signal_to_send:
            process.send_signal(signal_to_send)
        _wait_for_path(shutdown_path, process)
        for _ in range(signal_count - 1):
            process.send_signal(signal_to_send)
        release_path.touch()
        stdout, stderr = process.communicate(timeout=10)

    payloads = _read_payloads(payload_path)
    assert len(payloads) == 1
    return process.returncode, payloads[0]["events"][0], stdout, stderr


def _install_lightweight_tensorrt_llm_package() -> None:
    """Avoid importing GPU dependencies in the signal-test child process."""
    if "tensorrt_llm" in sys.modules:
        return
    package = types.ModuleType("tensorrt_llm")
    package.__path__ = [str(_TRTLLM_PACKAGE)]
    package.__file__ = str(_TRTLLM_PACKAGE / "__init__.py")
    package.__package__ = "tensorrt_llm"
    package.__version__ = "0.0.0-test"
    sys.modules["tensorrt_llm"] = package

    serve_package = types.ModuleType("tensorrt_llm.serve")
    serve_package.__path__ = [str(_TRTLLM_PACKAGE / "serve")]
    serve_package.__file__ = str(_TRTLLM_PACKAGE / "serve" / "__init__.py")
    serve_package.__package__ = "tensorrt_llm.serve"
    sys.modules["tensorrt_llm.serve"] = serve_package


def _run_signal_test_child(
    ready_path: Path,
    shutdown_path: Path,
    release_path: Path,
    payload_path: Path,
    *,
    observe_worker_failure: bool = False,
) -> None:
    """Run a minimal instrumented Uvicorn server in a fresh interpreter."""
    _install_lightweight_tensorrt_llm_package()

    from tensorrt_llm.serve._telemetry import TelemetryUvicornServer
    from tensorrt_llm.usage import usage_lib

    usage_lib._OPT_OUT_FILE = ready_path.parent / "no-opt-out-file"

    def is_reporting_rank() -> bool:
        return True

    def capture_payload(payload: dict) -> None:
        with payload_path.open("a", encoding="utf-8") as payload_file:
            payload_file.write(json.dumps(payload) + "\n")

    usage_lib._is_reporting_rank = is_reporting_rank
    usage_lib._send_to_gxt = capture_payload
    assert usage_lib.start_usage_session(
        default_usage_context="cli_serve",
        component="server",
        lifecycle_phase="serving",
    )

    async def app(scope, receive, send) -> None:
        assert scope["type"] == "lifespan"
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                if observe_worker_failure:
                    usage_lib.record_termination_observation(
                        usage_lib.TerminalOutcome(
                            termination_kind="worker_failure",
                            component="disagg_worker",
                            reporting_source="supervisor",
                            exit_code_known=False,
                        )
                    )
                ready_path.touch()
                if observe_worker_failure:
                    signal.raise_signal(signal.SIGINT)
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                shutdown_path.touch()
                while not release_path.exists():
                    await asyncio.sleep(0.01)
                await send({"type": "lifespan.shutdown.complete"})
                return

    import uvicorn

    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=0,
        lifespan="on",
        log_level="warning",
    )
    # An explicit empty socket list exercises Uvicorn's real lifecycle and
    # signal machinery without binding a network port in the unit-test sandbox.
    asyncio.run(TelemetryUvicornServer(config).serve(sockets=[]))


def _run_locked_signal_test_child(ready_path: Path, payload_path: Path) -> None:
    """Deliver SIGTERM while the main thread owns the telemetry lock."""
    _install_lightweight_tensorrt_llm_package()

    import click

    from tensorrt_llm.commands import _telemetry
    from tensorrt_llm.usage import usage_lib
    from tensorrt_llm.usage.config import UsageContext

    usage_lib._OPT_OUT_FILE = ready_path.parent / "no-opt-out-file"
    usage_lib._is_reporting_rank = lambda: True

    def capture_payload(payload: dict) -> None:
        with payload_path.open("a", encoding="utf-8") as payload_file:
            payload_file.write(json.dumps(payload) + "\n")

    usage_lib._send_to_gxt = capture_payload

    @click.group(
        cls=_telemetry.TelemetryGroup,
        telemetry_usage_context=UsageContext.CLI_SERVE,
        telemetry_component="server",
    )
    def cli() -> None:
        pass

    @cli.command()
    def run() -> None:
        signal.signal(signal.SIGTERM, _telemetry.raise_signal_exit)
        session = usage_lib._get_session()
        assert session is not None
        with session.lock:
            ready_path.touch()
            os.kill(os.getpid(), signal.SIGTERM)

    cli.main(args=["run"], prog_name="locked-signal-test")


def _run_incompatible_uvicorn_child() -> None:
    """Verify a changed private Uvicorn lifecycle fails explicitly."""
    _install_lightweight_tensorrt_llm_package()
    import uvicorn

    from tensorrt_llm.serve._telemetry import TelemetryUvicornServer

    uvicorn.Server._serve = None
    config = uvicorn.Config(lambda scope, receive, send: None)
    try:
        TelemetryUvicornServer(config)
    except RuntimeError as exc:
        assert "Unsupported Uvicorn signal API" in str(exc)
        return
    raise AssertionError("Incompatible Uvicorn signal API was accepted")


@pytest.mark.parametrize("signal_count", [1, 2])
def test_sigterm_reports_before_uvicorn_reraises(
    tmp_path: Path,
    signal_count: int,
) -> None:
    """SIGTERM reports once and retains Uvicorn's signal exit semantics."""
    returncode, event, stdout, stderr = _run_uvicorn_case(
        tmp_path,
        signal_to_send=signal.SIGTERM,
        signal_count=signal_count,
    )
    assert returncode == -signal.SIGTERM, (stdout, stderr)
    assert event["name"] == "trtllm_exit_report"
    assert event["parameters"]["terminationKind"] == "signal"
    assert event["parameters"]["signalNumber"] == signal.SIGTERM
    assert event["parameters"]["exitCodeKnown"] is True
    assert event["parameters"]["exitCode"] == 128 + signal.SIGTERM


def test_uvicorn_preserves_worker_failure_observation(tmp_path: Path) -> None:
    """A causal worker failure wins over Uvicorn's subsequent SIGINT."""
    returncode, event, stdout, stderr = _run_uvicorn_case(tmp_path, mode="uvicorn-worker-failure")
    assert returncode == -signal.SIGINT, (stdout, stderr)
    parameters = event["parameters"]
    assert parameters["terminationKind"] == "worker_failure"
    assert parameters["component"] == "disagg_worker"
    assert parameters["reportingSource"] == "supervisor"
    assert parameters["exitCodeKnown"] is False


def test_signal_handler_unwinds_before_recording(tmp_path: Path) -> None:
    """SIGTERM cannot self-deadlock on an interrupted telemetry lock."""
    ready_path = tmp_path / "ready"
    payload_path = tmp_path / "payloads.jsonl"
    with _signal_test_child("locked", ready_path, payload_path) as process:
        _wait_for_path(ready_path, process)
        stdout, stderr = process.communicate(timeout=10)

    assert process.returncode == 128 + signal.SIGTERM, (stdout, stderr)
    payloads = _read_payloads(payload_path)
    assert len(payloads) == 1
    parameters = payloads[0]["events"][0]["parameters"]
    assert parameters["terminationKind"] == "signal"
    assert parameters["signalNumber"] == signal.SIGTERM
    assert parameters["exitCode"] == 128 + signal.SIGTERM


def test_incompatible_uvicorn_signal_api_fails_predictably() -> None:
    """A changed private Uvicorn lifecycle cannot silently drop telemetry."""
    with _signal_test_child("incompatible") as process:
        stdout, stderr = process.communicate(timeout=10)
    assert process.returncode == 0, (stdout, stderr)


if __name__ == "__main__":
    mode = sys.argv[1]
    paths = (Path(argument) for argument in sys.argv[2:])
    if mode in ("uvicorn", "uvicorn-worker-failure"):
        _run_signal_test_child(
            *paths,
            observe_worker_failure=mode == "uvicorn-worker-failure",
        )
    elif mode == "locked":
        _run_locked_signal_test_child(*paths)
    elif mode == "incompatible":
        _run_incompatible_uvicorn_child()
    else:
        raise ValueError(f"Unknown signal-test mode: {mode}")
