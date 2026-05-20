import os

import pytest

from tensorrt_llm.executor import utils as executor_utils
from tensorrt_llm.executor.utils import LlmLauncherEnvs

_DEPRECATED_RAW_HMAC_KEY_ENV = "TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY"


def _reset_ipc_hmac_key_env(monkeypatch):
    monkeypatch.setattr(executor_utils,
                        "_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY", None)
    monkeypatch.delenv(_DEPRECATED_RAW_HMAC_KEY_ENV, raising=False)
    monkeypatch.delenv(
        LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY_FD.value,
        raising=False)


def _write_key_fd(key_hex: str) -> int:
    read_fd, write_fd = os.pipe()
    os.write(write_fd, key_hex.encode("ascii"))
    os.close(write_fd)
    return read_fd


def test_get_spawn_proxy_process_ipc_hmac_key_ignores_raw_env(monkeypatch):
    _reset_ipc_hmac_key_env(monkeypatch)
    key_hex = "01" * 32
    monkeypatch.setenv(_DEPRECATED_RAW_HMAC_KEY_ENV, key_hex)

    with pytest.raises(AssertionError, match="HMAC encryption is required"):
        executor_utils.get_spawn_proxy_process_ipc_hmac_key_env()

    assert _DEPRECATED_RAW_HMAC_KEY_ENV not in os.environ


def test_get_spawn_proxy_process_ipc_hmac_key_from_fd(monkeypatch):
    _reset_ipc_hmac_key_env(monkeypatch)
    key_hex = "02" * 32
    read_fd = _write_key_fd(key_hex)
    monkeypatch.setenv(
        LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY_FD.value,
        str(read_fd))

    key = executor_utils.get_spawn_proxy_process_ipc_hmac_key_env()

    assert key == bytes.fromhex(key_hex)
    assert (LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY_FD
            not in os.environ)
    with pytest.raises(OSError):
        os.fstat(read_fd)


def test_get_spawn_proxy_process_ipc_hmac_key_uses_fd_and_clears_raw_env(
        monkeypatch):
    _reset_ipc_hmac_key_env(monkeypatch)
    key_hex = "03" * 32
    raw_key_hex = "04" * 32
    read_fd = _write_key_fd(key_hex)
    monkeypatch.setenv(
        LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY_FD.value,
        str(read_fd))
    monkeypatch.setenv(_DEPRECATED_RAW_HMAC_KEY_ENV, raw_key_hex)

    key = executor_utils.get_spawn_proxy_process_ipc_hmac_key_env()

    assert key == bytes.fromhex(key_hex)
    assert (LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY_FD
            not in os.environ)
    assert _DEPRECATED_RAW_HMAC_KEY_ENV not in os.environ


def test_get_spawn_proxy_process_ipc_hmac_key_missing(monkeypatch):
    _reset_ipc_hmac_key_env(monkeypatch)

    with pytest.raises(AssertionError, match="HMAC encryption is required"):
        executor_utils.get_spawn_proxy_process_ipc_hmac_key_env()
