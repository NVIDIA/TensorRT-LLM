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
"""The graceful-shutdown bound that ``OpenAIServer`` takes from its environment.

``TRTLLM_SERVE_GRACEFUL_SHUTDOWN_TIMEOUT_S`` bounds how long uvicorn waits for running request
handlers after a shutdown signal. The ``uvicorn.Config`` call is checked in this process. The
shutdown itself runs the real ``OpenAIServer.__call__`` under uvloop, as ``trtllm-serve`` does, in
a fresh interpreter that receives SIGINT. Its ASGI app has a request handler that never finishes,
like a handler waiting for the final response of a request that the executor dropped.
"""

import asyncio
import inspect
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator, NoReturn

import pytest
import uvicorn

from tensorrt_llm.serve.openai_server import OpenAIServer

# The CPU stage collects with `-m cpu_only`; unittest/conftest.py also skips collecting any file
# that does not contain the cpu_only marker.
pytestmark = pytest.mark.cpu_only

_TIMEOUT_ENV = "TRTLLM_SERVE_GRACEFUL_SHUTDOWN_TIMEOUT_S"
# The parameters of the installed uvicorn.Config, read before any test replaces it.
_CONFIG_SIGNATURE = inspect.signature(uvicorn.Config)
# Graceful-shutdown bound of the bounded case, in seconds.
_BOUND_S = 1
# Covers the full tensorrt_llm import in the child before it serves.
_STARTUP_TIMEOUT_S = 300.0
# Upper limit for every other child event, so a regression fails instead of hanging.
_EVENT_TIMEOUT_S = 60.0
# How long the unbounded case must stay blocked: several times the bound.
_BLOCKED_FOR_S = 4.0 * _BOUND_S
_REQUEST = b"GET /never-finishes HTTP/1.1\r\nHost: localhost\r\n\r\n"
# Runs this file as __main__ in the child. Unlike `python <this file>`, it keeps this directory,
# whose helper modules (utils.py, openai_server.py) would shadow installed ones, off sys.path.
_RUN_AS_MAIN = (
    f"import runpy; runpy.run_path({str(Path(__file__).resolve())!r}, run_name='__main__')"
)
_needs_posix = pytest.mark.skipif(
    os.name != "posix", reason="The test verifies POSIX SIGINT semantics."
)


# ---- the uvicorn.Config call ----


class _ConfigCalled(Exception):
    """Stops ``OpenAIServer.__call__`` at its ``uvicorn.Config`` call."""


async def _no_op_app(scope, receive, send) -> None:
    pass


def _config_keywords(monkeypatch: pytest.MonkeyPatch, value: str | None) -> dict:
    """Return the keywords that ``OpenAIServer.__call__`` passes to ``uvicorn.Config``.

    Args:
        monkeypatch: Sets the variable and intercepts ``uvicorn.Config``.
        value: Value of the graceful-shutdown variable, or None to unset it.

    Returns:
        The keyword arguments of the ``uvicorn.Config`` call.
    """
    if value is None:
        monkeypatch.delenv(_TIMEOUT_ENV, raising=False)
    else:
        monkeypatch.setenv(_TIMEOUT_ENV, value)

    def config(app, **kwargs) -> NoReturn:
        # Binding raises TypeError for a keyword that the installed uvicorn does not accept.
        _CONFIG_SIGNATURE.bind(app, **kwargs)
        raise _ConfigCalled(kwargs)

    monkeypatch.setattr(uvicorn, "Config", config)
    server = object.__new__(OpenAIServer)
    server.app = _no_op_app
    try:
        asyncio.run(server("localhost", 0))
    except _ConfigCalled as called:
        return called.args[0]
    pytest.fail("OpenAIServer.__call__ returned without calling uvicorn.Config")


@pytest.mark.parametrize("value", [None, ""], ids=["unset", "empty"])
def test_unset_timeout_leaves_uvicorn_config_unchanged(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    """Unset or empty, the variable adds no keyword to the ``uvicorn.Config`` call."""
    assert "timeout_graceful_shutdown" not in _config_keywords(monkeypatch, value)


@pytest.mark.parametrize(("value", "seconds"), [("1", 1), ("45", 45)])
def test_positive_timeout_is_passed_to_uvicorn(
    monkeypatch: pytest.MonkeyPatch, value: str, seconds: int
) -> None:
    """A positive integer adds exactly ``timeout_graceful_shutdown`` to the Config call."""
    unset = _config_keywords(monkeypatch, None)

    assert _config_keywords(monkeypatch, value) == {**unset, "timeout_graceful_shutdown": seconds}


@pytest.mark.parametrize(
    "value", ["abc", "1.5", "0", "-3", " "], ids=["word", "fraction", "zero", "negative", "blank"]
)
def test_malformed_timeout_fails_before_uvicorn_config(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """A malformed value raises a ValueError that names the variable and the value."""
    with pytest.raises(ValueError) as error:
        _config_keywords(monkeypatch, value)

    assert _TIMEOUT_ENV in str(error.value) and repr(value) in str(error.value)


# ---- shutdown after SIGINT, in a fresh interpreter ----


@dataclass
class _Child:
    """A fresh interpreter serving the never-finishing request handler."""

    process: subprocess.Popen
    workdir: Path

    def log(self) -> str:
        """Return the combined stdout and stderr of the child so far."""
        return (self.workdir / "child.log").read_text(errors="replace")


@contextmanager
def _serving_child(timeout_value: str | None) -> Iterator[_Child]:
    """Serve the never-finishing request handler in a fresh interpreter; kill it on exit.

    Args:
        timeout_value: Value of the graceful-shutdown variable, or None to unset it.

    Yields:
        The running child. Its socket, markers and log live in ``workdir``.
    """
    env = os.environ.copy()
    env.pop(_TIMEOUT_ENV, None)
    if timeout_value is not None:
        env[_TIMEOUT_ENV] = timeout_value
    env.pop("TRTLLM_USAGE_FORCE_ENABLED", None)
    # TRT-LLM info logs, unbuffered so the log is complete while the child still runs, and no
    # telemetry from the child.
    env.update(TLLM_LOG_LEVEL="INFO", PYTHONUNBUFFERED="1", TRTLLM_NO_USAGE_STATS="1")
    # A short directory keeps the socket path under the AF_UNIX pathname limit.
    with TemporaryDirectory(prefix="trtllm-gs-") as directory:
        workdir = Path(directory)
        with (workdir / "child.log").open("wb") as log_file:
            process = subprocess.Popen(
                [sys.executable, "-c", _RUN_AS_MAIN, directory],
                cwd=directory,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        try:
            yield _Child(process, workdir)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


def _wait_for_path(child: _Child, path: Path, timeout: float) -> None:
    """Wait for a marker that the child creates, failing early if the child exits first."""
    deadline = time.monotonic() + timeout
    while not path.exists():
        if child.process.poll() is not None and not path.exists():
            pytest.fail(
                f"Child exited with {child.process.returncode} before {path.name}:\n{child.log()}"
            )
        if time.monotonic() > deadline:
            pytest.fail(f"Timed out waiting for {path.name}:\n{child.log()}")
        time.sleep(0.01)


def _wait_for_exit(child: _Child, timeout: float) -> int:
    """Return the exit code of the child, failing if it does not exit in time."""
    try:
        return child.process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pytest.fail(f"Child still running after {timeout}s:\n{child.log()}")


def _start_never_finishing_request(child: _Child) -> None:
    """Send one request, wait until its handler runs, then drop the client.

    The client leaves like the disaggregated router does at shutdown, so uvicorn then waits on
    the running handler rather than on an open connection.
    """
    _wait_for_path(child, child.workdir / "ready", _STARTUP_TIMEOUT_S)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(child.workdir / "http.sock"))
        client.sendall(_REQUEST)
        _wait_for_path(child, child.workdir / "handler_started", _EVENT_TIMEOUT_S)


def _serve_never_finishing_handler(directory: str) -> None:
    """Serve a request handler that never finishes through ``OpenAIServer.__call__``."""
    import uvloop

    workdir = Path(directory)

    async def app(scope, receive, send) -> None:
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    (workdir / "ready").touch()
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    (workdir / "lifespan_shutdown").touch()
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        (workdir / "handler_started").touch()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            (workdir / "handler_cancelled").touch()
            raise

    server = object.__new__(OpenAIServer)
    server.app = app
    server.disagg_cluster_worker = None
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(workdir / "http.sock"))
    listener.listen()
    # The same event loop as trtllm-serve.
    uvloop.run(server("localhost", 0, sockets=[listener]))


@_needs_posix
def test_bounded_shutdown_cancels_handler_then_runs_lifespan_shutdown() -> None:
    """With the bound, a stuck handler is cancelled and the lifespan shutdown still runs."""
    with _serving_child(str(_BOUND_S)) as child:
        _start_never_finishing_request(child)
        signalled = time.monotonic()
        child.process.send_signal(signal.SIGINT)
        _wait_for_path(child, child.workdir / "lifespan_shutdown", _EVENT_TIMEOUT_S)
        lifespan_shutdown_after = time.monotonic() - signalled
        # The exit code comes from interpreter teardown after uvicorn re-raises SIGINT, so only
        # the exit itself is checked.
        _wait_for_exit(child, _EVENT_TIMEOUT_S)
        log = child.log()
        assert lifespan_shutdown_after >= _BOUND_S, log
        assert (child.workdir / "handler_cancelled").exists(), log
        assert f"{_TIMEOUT_ENV}={_BOUND_S}:" in log
        # uvicorn waits on the handler, cancels it at the bound, then runs the lifespan shutdown.
        stages = [
            "Waiting for background tasks to complete",
            "Cancel 1 running task(s), timeout graceful shutdown exceeded",
            "Waiting for application shutdown",
            "Finished server process",
        ]
        positions = [log.find(stage) for stage in stages]
        assert positions[0] >= 0 and positions == sorted(positions), log


@_needs_posix
def test_unbounded_shutdown_waits_for_handler() -> None:
    """Without the variable, a stuck handler keeps shutdown from reaching the lifespan."""
    with _serving_child(None) as child:
        _start_never_finishing_request(child)
        child.process.send_signal(signal.SIGINT)
        time.sleep(_BLOCKED_FOR_S)
        assert child.process.poll() is None, child.log()
        log = child.log()
        assert "Waiting for background tasks to complete" in log
        assert not (child.workdir / "lifespan_shutdown").exists(), log
        assert not (child.workdir / "handler_cancelled").exists(), log
        assert _TIMEOUT_ENV not in log


if __name__ == "__main__":
    _serve_never_finishing_handler(sys.argv[1])
