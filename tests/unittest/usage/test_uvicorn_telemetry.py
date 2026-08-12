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
from pathlib import Path

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
                ready_path.touch()
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


@pytest.mark.parametrize("signal_count", [1, 2])
def test_sigterm_reports_before_uvicorn_reraises(
    tmp_path: Path,
    signal_count: int,
) -> None:
    """SIGTERM reports once and retains Uvicorn's signal exit semantics."""
    ready_path = tmp_path / "ready"
    shutdown_path = tmp_path / "shutdown"
    release_path = tmp_path / "release"
    payload_path = tmp_path / "payloads.jsonl"

    env = os.environ.copy()
    for name in ("TRTLLM_NO_USAGE_STATS", "TELEMETRY_DISABLED", "DO_NOT_TRACK"):
        env.pop(name, None)
    env["TRTLLM_USAGE_FORCE_ENABLED"] = "1"

    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "uvicorn",
            str(ready_path),
            str(shutdown_path),
            str(release_path),
            str(payload_path),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_path(ready_path, process)
        process.send_signal(signal.SIGTERM)
        _wait_for_path(shutdown_path, process)
        if signal_count == 2:
            process.send_signal(signal.SIGTERM)
        release_path.touch()
        stdout, stderr = process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    assert process.returncode == -signal.SIGTERM, (stdout, stderr)
    payloads = [json.loads(line) for line in payload_path.read_text(encoding="utf-8").splitlines()]
    assert len(payloads) == 1
    event = payloads[0]["events"][0]
    assert event["name"] == "trtllm_exit_report"
    assert event["parameters"]["terminationKind"] == "signal"
    assert event["parameters"]["signalNumber"] == signal.SIGTERM
    assert event["parameters"]["exitCodeKnown"] is True
    assert event["parameters"]["exitCode"] == 128 + signal.SIGTERM


def test_signal_handler_unwinds_before_recording(tmp_path: Path) -> None:
    """SIGTERM cannot self-deadlock on an interrupted telemetry lock."""
    ready_path = tmp_path / "ready"
    payload_path = tmp_path / "payloads.jsonl"
    env = os.environ.copy()
    for name in ("TRTLLM_NO_USAGE_STATS", "TELEMETRY_DISABLED", "DO_NOT_TRACK"):
        env.pop(name, None)
    env["TRTLLM_USAGE_FORCE_ENABLED"] = "1"

    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "locked",
            str(ready_path),
            str(payload_path),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_path(ready_path, process)
        stdout, stderr = process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    assert process.returncode == 128 + signal.SIGTERM, (stdout, stderr)
    payloads = [json.loads(line) for line in payload_path.read_text(encoding="utf-8").splitlines()]
    assert len(payloads) == 1
    parameters = payloads[0]["events"][0]["parameters"]
    assert parameters["terminationKind"] == "signal"
    assert parameters["signalNumber"] == signal.SIGTERM
    assert parameters["exitCode"] == 128 + signal.SIGTERM


if __name__ == "__main__":
    mode = sys.argv[1]
    paths = (Path(argument) for argument in sys.argv[2:])
    if mode == "uvicorn":
        _run_signal_test_child(*paths)
    elif mode == "locked":
        _run_locked_signal_test_child(*paths)
    else:
        raise ValueError(f"Unknown signal-test mode: {mode}")
