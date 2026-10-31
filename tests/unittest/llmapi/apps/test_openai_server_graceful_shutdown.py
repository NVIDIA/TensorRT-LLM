# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The graceful-shutdown bound that ``OpenAIServer`` passes to uvicorn.

``serve_graceful_shutdown_timeout_s`` bounds how long uvicorn waits for running request handlers
after a shutdown signal; the LLM, multimodal-encoder and embedding launchers of trtllm-serve hand
the LLM's value to ``OpenAIServer``. The ``uvicorn.Config`` call is checked in this process. The
shutdown itself runs the real ``OpenAIServer.__call__`` under uvloop, as ``trtllm-serve`` does, in a
fresh interpreter that receives SIGINT. Its ASGI app has a request handler that never finishes,
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
from collections.abc import Awaitable, Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest
import uvicorn

from tensorrt_llm.commands import serve
from tensorrt_llm.serve.openai_server import OpenAIServer

# The CPU stage collects with `-m cpu_only`; unittest/conftest.py also skips collecting any file
# that does not contain the cpu_only marker.
pytestmark = pytest.mark.cpu_only

# The parameters of the installed uvicorn.Config, read before any test replaces it.
_CONFIG_SIGNATURE = inspect.signature(uvicorn.Config)
# Graceful-shutdown bound of the child server, in seconds.
_BOUND_S = 1
# Allowance over the bound for the lifespan shutdown to start after SIGINT, in seconds: generous
# for a loaded CI host, yet it fails a shutdown that cancels the handler well past the bound.
_SLACK_S = 10


class _ConfigCalled(Exception):
    """Stops ``OpenAIServer.__call__`` at its ``uvicorn.Config`` call."""


class _ServerBuilt(Exception):
    """Stops a launcher once it has built its ``OpenAIServer``."""


def _config_keywords(monkeypatch: pytest.MonkeyPatch, timeout: int | None) -> dict:
    """Return the keywords that ``OpenAIServer.__call__`` passes to ``uvicorn.Config``."""

    def config(app: object, **kwargs: object) -> None:
        # Binding raises TypeError for a keyword that the installed uvicorn does not accept.
        _CONFIG_SIGNATURE.bind(app, **kwargs)
        raise _ConfigCalled(kwargs)

    monkeypatch.setattr(uvicorn, "Config", config)
    server = object.__new__(OpenAIServer)
    server.app = None
    server.graceful_shutdown_timeout = timeout
    with pytest.raises(_ConfigCalled) as called:
        asyncio.run(server("localhost", 0))
    return called.value.args[0]


@pytest.mark.parametrize("timeout", [None, 45], ids=["unset", "45"])
def test_timeout_is_passed_to_uvicorn_only_when_set(
    monkeypatch: pytest.MonkeyPatch, timeout: int | None
) -> None:
    """None adds no keyword; a timeout adds ``timeout_graceful_shutdown`` only."""
    unset = _config_keywords(monkeypatch, None)
    assert "timeout_graceful_shutdown" not in unset

    expected = unset if timeout is None else {**unset, "timeout_graceful_shutdown": timeout}
    assert _config_keywords(monkeypatch, timeout) == expected


@pytest.mark.parametrize("timeout", [0, -1])
def test_a_timeout_that_is_not_positive_is_rejected(timeout: int) -> None:
    """OpenAIServer refuses a bound uvicorn would read as no wait at all."""
    with pytest.raises(ValueError, match="graceful_shutdown_timeout must be a positive"):
        OpenAIServer(
            generator=None,
            model="dummy",
            tool_parser=None,
            server_role=None,
            metadata_server_cfg=None,
            graceful_shutdown_timeout=timeout,
        )


def _launch(
    launcher: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, generator: object
) -> None:
    """Run ``launcher`` of ``serve`` with ``generator`` as its LLM, up to its ``OpenAIServer``."""
    if launcher == "launch_mm_encoder_server":
        monkeypatch.setattr(serve, "MultimodalEncoder", lambda **kwargs: generator)
        serve.launch_mm_encoder_server("127.0.0.1", 0, {"model": "dummy"})
        return
    monkeypatch.setattr(serve, "PyTorchLLM", lambda **kwargs: generator)
    if launcher == "launch_embedding_server":
        # The launcher reads the model's architecture from its config, which "dummy" lacks.
        monkeypatch.setattr(serve, "_resolve_embedding_architecture_override", lambda *args: None)
        serve.launch_embedding_server(
            "127.0.0.1", 0, {"model": "dummy"}, max_queue_delay=0.0, max_queue_size=1
        )
        return
    llm_args = {"backend": "pytorch", "model": "dummy", "num_serve_frontends": 1}
    serve.launch_server("127.0.0.1", 0, llm_args, report_addr=str(tmp_path / "addr"))


@pytest.mark.parametrize("timeout", [None, 7], ids=["unset", "7"])
@pytest.mark.parametrize(
    "launcher", ["launch_server", "launch_mm_encoder_server", "launch_embedding_server"]
)
def test_each_launcher_passes_the_llm_timeout_to_its_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, launcher: str, timeout: int | None
) -> None:
    """The validated ``serve_graceful_shutdown_timeout_s`` of the LLM reaches ``OpenAIServer``.

    The LLM, multimodal-encoder and embedding servers each pass it on.
    """
    llm = SimpleNamespace(args=SimpleNamespace(serve_graceful_shutdown_timeout_s=timeout))
    built = {}

    def build_server(**kwargs: object) -> None:
        built.update(kwargs)
        raise _ServerBuilt

    monkeypatch.setattr(serve, "OpenAIServer", build_server)
    with pytest.raises(_ServerBuilt):
        _launch(launcher, monkeypatch, tmp_path, llm)

    assert built["generator"] is llm
    assert built["graceful_shutdown_timeout"] == timeout


def _wait_for(path: Path, child: subprocess.Popen, timeout: float) -> None:
    """Wait for a marker that the child creates, failing early if the child exits first."""
    deadline = time.monotonic() + timeout
    while not path.exists():
        if (child.poll() is not None or time.monotonic() > deadline) and not path.exists():
            pytest.fail(f"No {path.name}; child exit code {child.poll()}")
        time.sleep(0.01)


def _serve_never_finishing_handler(directory: str) -> None:
    """Serve a request handler that never finishes through ``OpenAIServer.__call__``."""
    import uvloop

    workdir = Path(directory)

    async def app(
        scope: dict,
        receive: Callable[[], Awaitable[dict]],
        send: Callable[[dict], Awaitable[None]],
    ) -> None:
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
    server.graceful_shutdown_timeout = _BOUND_S
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(workdir / "http.sock"))
    listener.listen()
    # The same event loop as trtllm-serve.
    uvloop.run(server("localhost", 0, sockets=[listener]))


@pytest.mark.skipif(os.name != "posix", reason="The test verifies POSIX SIGINT semantics.")
def test_bounded_shutdown_cancels_handler_then_runs_lifespan_shutdown() -> None:
    """With the bound, a stuck handler is cancelled and the lifespan shutdown still runs."""
    env = {k: v for k, v in os.environ.items() if k != "TRTLLM_USAGE_FORCE_ENABLED"}
    # Info logs, unbuffered so the log is complete while the child runs, and no telemetry.
    env.update(TLLM_LOG_LEVEL="INFO", PYTHONUNBUFFERED="1", TRTLLM_NO_USAGE_STATS="1")
    # Runs this file as __main__. Unlike `python <this file>`, it keeps this directory, whose
    # helper modules (utils.py, openai_server.py) would shadow installed ones, off sys.path.
    run_as_main = (
        f"import runpy; runpy.run_path({str(Path(__file__).resolve())!r}, run_name='__main__')"
    )
    # A short directory keeps the socket path under the AF_UNIX pathname limit.
    with TemporaryDirectory(prefix="trtllm-gs-") as directory:
        workdir = Path(directory)
        with (workdir / "child.log").open("wb") as log_file:
            child = subprocess.Popen(
                [sys.executable, "-c", run_as_main, directory],
                cwd=directory,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        try:
            # The first wait covers the full tensorrt_llm import in the child.
            _wait_for(workdir / "ready", child, 300)
            # The client leaves once its handler runs, as the disaggregated router does at
            # shutdown, so uvicorn then waits on the running handler, not on a connection.
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(str(workdir / "http.sock"))
                client.sendall(b"GET /never-finishes HTTP/1.1\r\nHost: localhost\r\n\r\n")
                _wait_for(workdir / "handler_started", child, 60)
            signalled = time.monotonic()
            child.send_signal(signal.SIGINT)
            _wait_for(workdir / "lifespan_shutdown", child, 60)
            lifespan_shutdown_after = time.monotonic() - signalled
            # The exit code comes from interpreter teardown after uvicorn re-raises SIGINT, so
            # only the exit itself is checked.
            child.wait(timeout=60)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
            log = (workdir / "child.log").read_text(errors="replace")
            print(log)  # Reported with any failure.
        assert _BOUND_S <= lifespan_shutdown_after < _BOUND_S + _SLACK_S
        assert (workdir / "handler_cancelled").exists()
        assert f"graceful_shutdown_timeout={_BOUND_S}:" in log
    # uvicorn waits on the handler, cancels it at the bound, then runs the lifespan shutdown.
    stages = [
        "Waiting for background tasks to complete",
        "Cancel 1 running task(s), timeout graceful shutdown exceeded",
        "Waiting for application shutdown",
        "Finished server process",
    ]
    positions = [log.find(stage) for stage in stages]
    assert positions[0] >= 0 and positions == sorted(positions)


if __name__ == "__main__":
    _serve_never_finishing_handler(sys.argv[1])
