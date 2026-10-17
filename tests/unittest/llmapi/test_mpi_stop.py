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
from pathlib import Path
from types import ModuleType

import pytest
import zmq

pytestmark = pytest.mark.cpu_only

_STOP_SCRIPT = Path(__file__).parents[3] / "tensorrt_llm" / "llmapi" / "_llmapi_mpi_stop.py"
_ADDRESS_ENV = "TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR"
_KEY_ENV = "TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY"
_KEY = bytes(range(32))


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
            f"raise AssertionError('stop sender imported {name}')\n", encoding="utf-8"
        )
    env = {"PATH": os.environ.get("PATH", os.defpath), "PYTHONPATH": str(tmp_path)}
    if "LD_LIBRARY_PATH" in os.environ:
        env["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
    env[_KEY_ENV] = _KEY.hex()
    env[_ADDRESS_ENV] = "tcp://127.0.0.1:1"
    return env


def _run_stop(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run the standalone sender with a 15-second outer deadline.

    Args:
        env: Environment variables for the sender subprocess.

    Returns:
        The sender's exit status and captured text output.
    """
    return subprocess.run(
        [sys.executable, str(_STOP_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )


def test_mpi_stop_sends_authenticated_none_without_heavy_imports(
    stop_environment: dict[str, str],
) -> None:
    """Verify the standalone sender delivers an authenticated shutdown sentinel."""
    with zmq.Context() as context, context.socket(zmq.PAIR) as peer:
        peer.setsockopt(zmq.LINGER, 0)
        port = peer.bind_to_random_port("tcp://127.0.0.1")
        stop_environment[_ADDRESS_ENV] = f"tcp://127.0.0.1:{port}"

        result = _run_stop(stop_environment)

        assert result.returncode == 0, result.stderr
        assert not result.stderr
        assert peer.poll(1000), "stop sender exited without delivering a message"
        frame = peer.recv()
        signature_size = hashlib.sha256().digest_size
        assert len(frame) > signature_size
        payload, signature = frame[:-signature_size], frame[-signature_size:]
        assert hmac.compare_digest(signature, hmac.digest(_KEY, payload, "sha256"))
        assert pickle.loads(payload) is None  # nosec B301 - authenticated local test payload


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

        result = _run_stop(stop_environment)

    assert result.returncode == 1, result.stderr
    assert "Failed to send MPI Comm server shutdown" in result.stderr


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
