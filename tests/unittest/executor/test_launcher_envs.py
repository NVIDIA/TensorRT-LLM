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

import ctypes
import os
import sys

import pytest

import tensorrt_llm.executor.utils as executor_utils
from tensorrt_llm.executor.utils import LlmLauncherEnvs


@pytest.fixture(autouse=True)
def reset_ipc_hmac_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(executor_utils, "_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY", None)
    monkeypatch.delenv(LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY.value, raising=False)


def _mock_env_scrub(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        executor_utils,
        "_scrub_process_env_value",
        lambda key_name, value: calls.append((key_name, value)),
    )
    return calls


def test_get_spawn_proxy_process_ipc_hmac_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _mock_env_scrub(monkeypatch)
    key_hex = "01" * 32
    monkeypatch.setenv(LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY.value, key_hex)

    key = executor_utils.get_spawn_proxy_process_ipc_hmac_key_env()

    assert key == bytes.fromhex(key_hex)
    assert calls == [(LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY.value, key_hex)]
    assert LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY.value not in os.environ


def test_get_spawn_proxy_process_ipc_hmac_key_caches_env_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _mock_env_scrub(monkeypatch)
    key_hex = "02" * 32
    monkeypatch.setenv(LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY.value, key_hex)

    key = executor_utils.get_spawn_proxy_process_ipc_hmac_key_env()

    assert LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY.value not in os.environ
    assert executor_utils.get_spawn_proxy_process_ipc_hmac_key_env() == key
    assert calls == [(LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY.value, key_hex)]


def test_get_spawn_proxy_process_ipc_hmac_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _mock_env_scrub(monkeypatch)

    with pytest.raises(RuntimeError, match="HMAC encryption is required"):
        executor_utils.get_spawn_proxy_process_ipc_hmac_key_env()

    assert calls == []


def test_get_spawn_proxy_process_ipc_hmac_key_rejects_invalid_hex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_env_scrub(monkeypatch)
    monkeypatch.setenv(LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY.value, "not-a-hex-key")

    with pytest.raises(ValueError, match="IPC HMAC key must be a 64-character hex string"):
        executor_utils.get_spawn_proxy_process_ipc_hmac_key_env()

    assert LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY.value not in os.environ


def test_get_spawn_proxy_process_ipc_hmac_key_rejects_wrong_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_env_scrub(monkeypatch)
    monkeypatch.setenv(LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY.value, "01")

    with pytest.raises(ValueError, match="IPC HMAC key must be a 64-character hex string"):
        executor_utils.get_spawn_proxy_process_ipc_hmac_key_env()

    assert LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY.value not in os.environ


def test_get_spawn_proxy_process_ipc_hmac_key_rejects_hex_with_whitespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_env_scrub(monkeypatch)
    monkeypatch.setenv(LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY.value, "01 " * 32)

    with pytest.raises(ValueError, match="IPC HMAC key must be a 64-character hex string"):
        executor_utils.get_spawn_proxy_process_ipc_hmac_key_env()

    assert LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY.value not in os.environ


@pytest.mark.skipif(sys.platform != "linux", reason="requires libc getenv")
def test_scrub_process_env_value_zeroes_real_libc_env_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_name = "TRTLLM_TEST_SCRUB_PROCESS_ENV_VALUE"
    env_value = "bad-\u00e9-value"
    env_value_bytes = os.fsencode(env_value)
    monkeypatch.setenv(env_name, env_value)

    libc = ctypes.CDLL(None)
    libc.getenv.restype = ctypes.c_void_p
    value_ptr = libc.getenv(os.fsencode(env_name))
    assert value_ptr
    assert ctypes.string_at(value_ptr, len(env_value_bytes)) == env_value_bytes

    executor_utils._scrub_process_env_value(env_name, env_value)

    assert ctypes.string_at(value_ptr, len(env_value_bytes)) == b"\0" * len(env_value_bytes)
