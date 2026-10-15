# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the standalone MPI stop sender with real sockets and subprocesses."""

import ctypes
import hashlib
import hmac
import importlib.util
import os
import pickle
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import pytest
import zmq
from zmq.utils.monitor import recv_monitor_message

pytestmark = pytest.mark.cpu_only

_STOP_SCRIPT = Path(__file__).parents[3] / "tensorrt_llm" / "llmapi" / "_llmapi_mpi_stop.py"
_ADDRESS_ENV = "TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR"
_KEY_ENV = "TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY"
_KEY = bytes(range(32))
_CALL_MAIN = """
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("standalone_stop", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.exit(module.main(ack_timeout=float(sys.argv[2]), max_attempts=int(sys.argv[3])))
"""


@pytest.fixture
def stop_environment(tmp_path: Path) -> dict[str, str]:
    """Make any heavy-package import fail in the standalone subprocess.

    Returns:
        A subprocess environment with valid credentials and import traps.
    """
    for name in ("tensorrt_llm", "torch", "mpi4py"):
        package = tmp_path / name
        package.mkdir()
        (package / "__init__.py").write_text(
            "from pathlib import Path\n"
            f"Path({str(tmp_path / f'imported-{name}')!r}).touch()\n"
            f"raise AssertionError('stop sender imported {name}')\n",
            encoding="utf-8",
        )
    env = {"PATH": os.environ.get("PATH", os.defpath), "PYTHONPATH": str(tmp_path)}
    if "LD_LIBRARY_PATH" in os.environ:
        env["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
    env[_KEY_ENV] = _KEY.hex()
    env[_ADDRESS_ENV] = "tcp://127.0.0.1:1"
    return env


def _stop_command(ack_timeout: float | None, max_attempts: int) -> list[str]:
    """Select the real CLI or a file-loaded entry point with a short retry budget."""
    if ack_timeout is None:
        return [sys.executable, str(_STOP_SCRIPT)]
    return [
        sys.executable,
        "-c",
        _CALL_MAIN,
        str(_STOP_SCRIPT),
        str(ack_timeout),
        str(max_attempts),
    ]


def _run_stop(
    env: dict[str, str], *, ack_timeout: float | None = None, max_attempts: int = 3
) -> subprocess.CompletedProcess[str]:
    """Run the standalone sender with a 15-second outer deadline.

    Args:
        env: Environment variables for the sender subprocess.
        ack_timeout: Optional shortened acknowledgement deadline for each attempt.
        max_attempts: Attempt limit used with the shortened deadline.

    Returns:
        The sender's exit status and captured text output.
    """
    return subprocess.run(
        _stop_command(ack_timeout, max_attempts),
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )


@contextmanager
def _start_stop(
    env: dict[str, str], *, ack_timeout: float | None = None, max_attempts: int = 3
) -> Iterator[subprocess.Popen[str]]:
    """Run the sender while the calling thread owns and services the peer socket."""
    with subprocess.Popen(
        _stop_command(ack_timeout, max_attempts),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as process:
        try:
            yield process
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=5)


def _signed(payload: bytes) -> bytes:
    """Build a peer response in the production pickle-plus-HMAC wire format."""
    return payload + hmac.digest(_KEY, payload, "sha256")


def _receive_stop(peer: zmq.Socket) -> None:
    """Check the real sender's authenticated request before acknowledging it."""
    assert peer.poll(5000), "stop sender did not deliver a request"
    frame = peer.recv()
    signature_size = hashlib.sha256().digest_size
    assert len(frame) > signature_size
    payload, signature = frame[:-signature_size], frame[-signature_size:]
    assert hmac.compare_digest(signature, hmac.digest(_KEY, payload, "sha256"))
    assert pickle.loads(payload) is None  # nosec B301 - authenticated local test payload


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_mpi_stop_sends_authenticated_none_without_heavy_imports(
    stop_environment: dict[str, str],
    protocol: int,
) -> None:
    """The real CLI must await an authenticated ACK in every canonical protocol."""
    with zmq.Context() as context, context.socket(zmq.PAIR) as peer:
        peer.setsockopt(zmq.LINGER, 0)
        port = peer.bind_to_random_port("tcp://127.0.0.1")
        stop_environment[_ADDRESS_ENV] = f"tcp://127.0.0.1:{port}"

        with _start_stop(stop_environment) as process:
            _receive_stop(peer)
            peer.send(_signed(pickle.dumps("shutdown-ack", protocol=protocol)))
            _, stderr = process.communicate(timeout=15)

        assert process.returncode == 0, stderr
        assert not stderr


def test_mpi_stop_fails_within_deadline_without_zmq_peer(
    stop_environment: dict[str, str],
) -> None:
    """Verify a peer that never completes its handshake causes bounded failure."""
    # Keep the port owned, but never complete a ZeroMQ handshake. This avoids
    # the free-port race and exercises both connection readiness and send bounds.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        stop_environment[_ADDRESS_ENV] = f"tcp://127.0.0.1:{listener.getsockname()[1]}"

        result = _run_stop(stop_environment, ack_timeout=0.1, max_attempts=3)

    assert result.returncode == 1, result.stderr
    assert "Failed to send MPI Comm server shutdown" in result.stderr


def test_mpi_stop_requires_ack_after_sending(stop_environment: dict[str, str]) -> None:
    """A successful send without a server acknowledgement must still fail."""
    with zmq.Context() as context, context.socket(zmq.PAIR) as peer:
        peer.setsockopt(zmq.LINGER, 0)
        port = peer.bind_to_random_port("tcp://127.0.0.1")
        stop_environment[_ADDRESS_ENV] = f"tcp://127.0.0.1:{port}"

        with _start_stop(stop_environment, ack_timeout=0.2, max_attempts=2) as process:
            _receive_stop(peer)
            _, stderr = process.communicate(timeout=3)

        assert process.returncode == 1, stderr
        assert "Failed to send MPI Comm server shutdown" in stderr


def test_mpi_stop_reconnects_after_unacknowledged_request(
    stop_environment: dict[str, str],
) -> None:
    """A dropped ACK must cause a stop request on a new PAIR connection."""
    with zmq.Context() as context, context.socket(zmq.PAIR) as peer:
        peer.setsockopt(zmq.LINGER, 0)
        port = peer.bind_to_random_port("tcp://127.0.0.1")
        stop_environment[_ADDRESS_ENV] = f"tcp://127.0.0.1:{port}"

        with peer.get_monitor_socket(events=zmq.EVENT_ACCEPTED) as monitor:
            monitor.setsockopt(zmq.LINGER, 0)
            with _start_stop(stop_environment, ack_timeout=0.5, max_attempts=6) as process:
                _receive_stop(peer)
                # Leave the server alive, but deliberately do not acknowledge
                # this request. A resend on the same socket is insufficient.
                _receive_stop(peer)
                peer.send(_signed(pickle.dumps("shutdown-ack")))
                _, stderr = process.communicate(timeout=5)

                for _ in range(2):
                    assert monitor.poll(2000), "retry reused the original PAIR connection"
                    assert recv_monitor_message(monitor)["event"] == zmq.EVENT_ACCEPTED

            peer.disable_monitor()
        assert process.returncode == 0, stderr
        assert not stderr


@pytest.mark.parametrize(
    "frame",
    [
        pytest.param(b"", id="empty"),
        pytest.param(_signed(pickle.dumps("shutdown-ack"))[:-1], id="truncated-signature"),
        pytest.param(pickle.dumps("shutdown-ack"), id="missing-signature"),
        pytest.param(pickle.dumps("shutdown-ack") + b"\0" * 32, id="wrong-signature"),
        pytest.param(_signed(pickle.dumps(None)), id="wrong-value"),
    ],
)
def test_mpi_stop_does_not_accept_invalid_ack(
    stop_environment: dict[str, str],
    frame: bytes,
) -> None:
    """A malformed or unrelated reply must not produce a successful stop."""
    with zmq.Context() as context, context.socket(zmq.PAIR) as peer:
        peer.setsockopt(zmq.LINGER, 0)
        port = peer.bind_to_random_port("tcp://127.0.0.1")
        stop_environment[_ADDRESS_ENV] = f"tcp://127.0.0.1:{port}"

        with _start_stop(stop_environment, ack_timeout=0.2, max_attempts=1) as process:
            _receive_stop(peer)
            peer.send(frame)
            _, stderr = process.communicate(timeout=3)

        assert process.returncode == 1, stderr
        assert "Failed to send MPI Comm server shutdown" in stderr


def test_mpi_stop_ignores_unrelated_and_invalid_ack_frames(
    stop_environment: dict[str, str],
    tmp_path: Path,
) -> None:
    """Ignore stale replies without importing globals from authenticated pickles."""
    ack = _signed(pickle.dumps("shutdown-ack"))
    frames = [
        b"",
        ack[:-1],
        ack[:-32],
        ack[:-32] + b"\0" * 32,
        _signed(pickle.dumps(None)),
        _signed(b"ctorch\nTensor\n."),
        _signed(b"cmpi4py\nMPI\n."),
        _signed(b"ctensorrt_llm\nLLM\n."),
    ]
    with zmq.Context() as context, context.socket(zmq.PAIR) as peer:
        peer.setsockopt(zmq.LINGER, 0)
        port = peer.bind_to_random_port("tcp://127.0.0.1")
        stop_environment[_ADDRESS_ENV] = f"tcp://127.0.0.1:{port}"

        with _start_stop(stop_environment) as process:
            _receive_stop(peer)
            for frame in frames:
                peer.send(frame)
            peer.send(ack)
            _, stderr = process.communicate(timeout=15)

        assert process.returncode == 0, stderr
        assert not stderr
    assert not list(tmp_path.glob("imported-*")), "a peer reply imported a heavy package"


def test_mpi_stop_unrelated_frames_do_not_extend_ack_deadline(
    stop_environment: dict[str, str],
) -> None:
    """Continuous unrelated replies must not restart the attempt's ACK deadline."""
    with zmq.Context() as context, context.socket(zmq.PAIR) as peer:
        peer.setsockopt(zmq.LINGER, 0)
        port = peer.bind_to_random_port("tcp://127.0.0.1")
        stop_environment[_ADDRESS_ENV] = f"tcp://127.0.0.1:{port}"

        with _start_stop(stop_environment, ack_timeout=0.2, max_attempts=1) as process:
            _receive_stop(peer)
            deadline = time.monotonic() + 3
            while process.poll() is None:
                assert time.monotonic() < deadline, "unrelated replies extended the ACK deadline"
                try:
                    peer.send(_signed(pickle.dumps("unrelated")), flags=zmq.DONTWAIT)
                except zmq.Again:
                    # The helper can close its socket before process.poll sees exit.
                    pass
                # A bounded socket wait paces replies without a timing-sensitive sleep.
                assert not peer.poll(10), "single attempt unexpectedly sent another request"
            _, stderr = process.communicate(timeout=3)

        assert process.returncode == 1, stderr
        assert "Failed to send MPI Comm server shutdown" in stderr


@pytest.mark.parametrize(
    ("name", "value"),
    [
        pytest.param(_ADDRESS_ENV, None, id="missing-address"),
        pytest.param(_ADDRESS_ENV, "invalid://endpoint", id="invalid-endpoint"),
        pytest.param(_KEY_ENV, None, id="missing-key"),
        pytest.param(_KEY_ENV, "invalid-key", id="malformed-key"),
        pytest.param(_KEY_ENV, "01", id="short-key"),
        pytest.param(_KEY_ENV, "01 " * 32, id="whitespace-key"),
    ],
)
def test_mpi_stop_reports_invalid_configuration(
    stop_environment: dict[str, str], name: str, value: str | None
) -> None:
    """Verify invalid configuration fails without exposing key material."""
    if value is None:
        stop_environment.pop(name)
    else:
        stop_environment[name] = value

    result = _run_stop(stop_environment)

    assert result.returncode == 1, result.stderr
    assert "Failed to send MPI Comm server shutdown" in result.stderr
    assert _KEY.hex() not in result.stderr
    if name == _KEY_ENV and value is not None:
        assert value not in result.stderr


@pytest.fixture
def stop_module() -> ModuleType:
    """Load the helper by path without importing TensorRT-LLM.

    Returns:
        A fresh helper module for credential handling checks.
    """
    spec = importlib.util.spec_from_file_location("llmapi_mpi_stop_test", _STOP_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "key",
    [_KEY.hex(), "invalid-key", "01", "01 " * 32],
    ids=["valid", "nonhex", "short", "whitespace"],
)
def test_mpi_stop_consumes_key_environment(
    stop_module: ModuleType, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    """Verify consuming either a valid or invalid key removes its environment entry."""
    monkeypatch.setenv(_KEY_ENV, key)

    if key == _KEY.hex():
        assert stop_module._consume_hmac_key() == _KEY
    else:
        with pytest.raises(ValueError, match="64-character hex string"):
            stop_module._consume_hmac_key()

    assert _KEY_ENV not in os.environ


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux libc environment scrubbing")
def test_mpi_stop_zeroes_libc_key_bytes(
    stop_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify key consumption also clears the original Linux libc environment bytes."""
    key = _KEY.hex().encode()
    monkeypatch.setenv(_KEY_ENV, key.decode())
    libc = ctypes.CDLL(None)
    libc.getenv.restype = ctypes.c_void_p
    value_ptr = libc.getenv(os.fsencode(_KEY_ENV))
    assert value_ptr
    assert ctypes.string_at(value_ptr, len(key)) == key

    assert stop_module._consume_hmac_key() == _KEY

    assert _KEY_ENV not in os.environ
    assert ctypes.string_at(value_ptr, len(key)) == b"\0" * len(key)
