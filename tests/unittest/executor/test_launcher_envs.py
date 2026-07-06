import os
import threading

import pytest

from tensorrt_llm.executor import utils as executor_utils
from tensorrt_llm.executor.utils import LlmLauncherEnvs


def _reset_ipc_hmac_key_env(monkeypatch):
    monkeypatch.setattr(executor_utils, "_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY", None)
    monkeypatch.delenv(
        LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY_FD.value, raising=False
    )


def _write_key_fd(key_hex: str) -> int:
    read_fd, write_fd = os.pipe()
    os.write(write_fd, key_hex.encode("ascii"))
    os.close(write_fd)
    return read_fd


def test_get_spawn_proxy_process_ipc_hmac_key_from_fd(monkeypatch):
    _reset_ipc_hmac_key_env(monkeypatch)
    key_hex = "01" * 32
    read_fd = _write_key_fd(key_hex)
    monkeypatch.setenv(LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY_FD.value, str(read_fd))

    key = executor_utils.get_spawn_proxy_process_ipc_hmac_key_env()

    assert key == bytes.fromhex(key_hex)
    assert LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY_FD not in os.environ
    with pytest.raises(OSError):
        os.fstat(read_fd)


def test_get_spawn_proxy_process_ipc_hmac_key_caches_fd_key(monkeypatch):
    _reset_ipc_hmac_key_env(monkeypatch)
    key_hex = "02" * 32
    read_fd = _write_key_fd(key_hex)
    monkeypatch.setenv(LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY_FD.value, str(read_fd))

    key = executor_utils.get_spawn_proxy_process_ipc_hmac_key_env()

    assert key == bytes.fromhex(key_hex)
    assert LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY_FD not in os.environ
    assert executor_utils.get_spawn_proxy_process_ipc_hmac_key_env() == key


def test_get_spawn_proxy_process_ipc_hmac_key_from_nonblocking_fd(monkeypatch):
    _reset_ipc_hmac_key_env(monkeypatch)
    key_hex = "03" * 32
    read_fd, write_fd = os.pipe()
    os.set_blocking(read_fd, False)

    def write_key():
        try:
            os.write(write_fd, key_hex.encode("ascii"))
        finally:
            os.close(write_fd)

    writer = threading.Timer(0.05, write_key)
    writer.start()
    monkeypatch.setenv(LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY_FD.value, str(read_fd))

    try:
        key = executor_utils.get_spawn_proxy_process_ipc_hmac_key_env()
    finally:
        writer.join()

    assert key == bytes.fromhex(key_hex)


def test_get_spawn_proxy_process_ipc_hmac_key_missing(monkeypatch):
    _reset_ipc_hmac_key_env(monkeypatch)

    with pytest.raises(RuntimeError, match="HMAC encryption is required"):
        executor_utils.get_spawn_proxy_process_ipc_hmac_key_env()
