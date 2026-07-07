import os

import pytest

import tensorrt_llm.executor.utils as executor_utils
from tensorrt_llm.executor.utils import LlmLauncherEnvs


@pytest.fixture(autouse=True)
def reset_ipc_hmac_key_env(monkeypatch):
    monkeypatch.setattr(executor_utils, "_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY", None)
    monkeypatch.delenv(LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY.value, raising=False)


def _mock_non_dumpable(monkeypatch):
    calls = []
    monkeypatch.setattr(executor_utils, "_set_process_non_dumpable", lambda: calls.append(True))
    return calls


def test_get_spawn_proxy_process_ipc_hmac_key_from_env(monkeypatch):
    calls = _mock_non_dumpable(monkeypatch)
    key_hex = "01" * 32
    monkeypatch.setenv(LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY.value, key_hex)

    key = executor_utils.get_spawn_proxy_process_ipc_hmac_key_env()

    assert key == bytes.fromhex(key_hex)
    assert calls == [True]
    assert LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY.value not in os.environ


def test_get_spawn_proxy_process_ipc_hmac_key_caches_env_key(monkeypatch):
    calls = _mock_non_dumpable(monkeypatch)
    key_hex = "02" * 32
    monkeypatch.setenv(LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY.value, key_hex)

    key = executor_utils.get_spawn_proxy_process_ipc_hmac_key_env()

    assert LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY.value not in os.environ
    assert executor_utils.get_spawn_proxy_process_ipc_hmac_key_env() == key
    assert calls == [True]


def test_get_spawn_proxy_process_ipc_hmac_key_missing(monkeypatch):
    calls = _mock_non_dumpable(monkeypatch)

    with pytest.raises(RuntimeError, match="HMAC encryption is required"):
        executor_utils.get_spawn_proxy_process_ipc_hmac_key_env()

    assert calls == [True]


def test_get_spawn_proxy_process_ipc_hmac_key_rejects_invalid_hex(monkeypatch):
    _mock_non_dumpable(monkeypatch)
    monkeypatch.setenv(LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY.value, "not-a-hex-key")

    with pytest.raises(ValueError, match="IPC HMAC key must be a 64-character hex string"):
        executor_utils.get_spawn_proxy_process_ipc_hmac_key_env()

    assert LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY.value not in os.environ


def test_get_spawn_proxy_process_ipc_hmac_key_rejects_wrong_length(monkeypatch):
    _mock_non_dumpable(monkeypatch)
    monkeypatch.setenv(LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY.value, "01")

    with pytest.raises(ValueError, match="IPC HMAC key must be 32 bytes"):
        executor_utils.get_spawn_proxy_process_ipc_hmac_key_env()

    assert LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY.value not in os.environ
