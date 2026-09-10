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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import ExecutorMemoryType, RuntimeMemoryStatus, SleepConfig

pytestmark = pytest.mark.cpu_only


def _make_llm() -> LLM:
    llm = object.__new__(LLM)
    llm.args = SimpleNamespace(sleep_config=SleepConfig())
    llm._collective_rpc = MagicMock()
    return llm


def test_release_defaults_to_all_usable_tags():
    llm = _make_llm()

    llm.release()

    expected = [
        tag.value
        for tag in ExecutorMemoryType
        if tag
        not in (
            ExecutorMemoryType.INIT_KV_CACHE,
            ExecutorMemoryType.INIT_EXTRA_RESOURCES,
        )
    ]
    llm._collective_rpc.assert_called_once_with("sleep", (expected,))


def test_release_normalizes_enum_strings_and_duplicates():
    llm = _make_llm()

    llm.release(
        [
            "model",
            ExecutorMemoryType.KV_CACHE,
            "model",
        ]
    )

    llm._collective_rpc.assert_called_once_with("sleep", (["model", "kv_cache"],))


@pytest.mark.parametrize(
    "tags, message",
    [
        ([], "must not be empty"),
        (["unknown"], "Invalid runtime memory tag"),
        ([ExecutorMemoryType.INIT_KV_CACHE], "initialization-only"),
        ([None], "Invalid runtime memory tag"),
    ],
)
def test_release_rejects_invalid_explicit_tags(tags, message):
    llm = _make_llm()

    with pytest.raises(ValueError, match=message):
        llm.release(tags)

    llm._collective_rpc.assert_not_called()


def test_resume_defaults_to_currently_parked_tags():
    llm = _make_llm()
    llm.get_memory_status = MagicMock(
        return_value=RuntimeMemoryStatus(
            state="parked",
            parked_tags=[
                ExecutorMemoryType.MODEL_ENGINE_MAIN,
                ExecutorMemoryType.KV_CACHE,
            ],
        )
    )

    llm.resume()

    llm._collective_rpc.assert_called_once_with("wakeup", (["model", "kv_cache"],))


def test_resume_is_noop_when_nothing_is_parked():
    llm = _make_llm()
    llm.get_memory_status = MagicMock(
        return_value=RuntimeMemoryStatus(
            state="running",
            parked_tags=[],
        )
    )

    llm.resume()

    llm._collective_rpc.assert_not_called()


def test_get_memory_status_reconciles_worker_replies():
    llm = _make_llm()
    reply = {"state": "parked", "parked_tags": ["model"]}
    llm._collective_rpc.return_value = [reply, reply]

    status = llm.get_memory_status()

    assert status == RuntimeMemoryStatus(
        state="parked",
        parked_tags=[ExecutorMemoryType.MODEL_ENGINE_MAIN],
    )


def test_get_memory_status_fails_on_worker_divergence():
    llm = _make_llm()
    llm._collective_rpc.return_value = [
        {"state": "parked", "parked_tags": ["model"]},
        {"state": "running", "parked_tags": []},
    ]

    with pytest.raises(RuntimeError, match="diverged across workers"):
        llm.get_memory_status()


def test_runtime_memory_api_requires_sleep_config():
    llm = _make_llm()
    llm.args.sleep_config = None

    with pytest.raises(ValueError, match="not enabled"):
        llm.release()
    with pytest.raises(ValueError, match="not enabled"):
        llm.resume()
    with pytest.raises(ValueError, match="not enabled"):
        llm.get_memory_status()
