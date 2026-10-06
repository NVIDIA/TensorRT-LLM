# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from tensorrt_llm.scaffolding.task import ChatTask, DropKVCacheTask, RoleMessage, TaskStatus
from tensorrt_llm.scaffolding.worker import OpenaiWorker


@pytest.mark.parametrize("suffix", ["", "/", "/v1", "/v1/"])
@pytest.mark.parametrize("api_key", [None, "test-key"])
def test_kv_cache_hint_control_endpoint(suffix: str, api_key: str | None) -> None:
    client = SimpleNamespace(base_url=f"http://localhost:8000{suffix}", api_key=api_key)
    OpenaiWorker(async_client=client, model="test-model", kv_cache_hint_enabled=True)
    messages = [
        RoleMessage(role="system", content="shared instructions"),
        RoleMessage(role="user", content="first question"),
        RoleMessage(role="assistant", content="tool call"),
        RoleMessage(role="tool", content="tool result"),
        RoleMessage(role="user", content="follow-up question"),
    ]
    task = DropKVCacheTask(ChatTask.create_from_messages(messages), worker_tag="drop")
    params = {
        "model": "test-model",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    response = httpx.Response(200)
    transport = AsyncMock()
    transport.post.return_value = response

    with patch.object(httpx, "AsyncClient") as http_client:
        http_client.return_value.__aenter__.return_value = transport
        result = asyncio.run(client.create_kv_cache_hint(task, params))

    assert result is response
    transport.post.assert_awaited_once_with(
        "http://localhost:8000/_control/kv_cache/truncate",
        json={
            "action": "truncate",
            "messages": [message.to_dict() for message in messages],
            "messages_to_retain": [{"role": "system", "content": "shared instructions"}],
            "model": "test-model",
            "chat_template_kwargs": {"enable_thinking": False},
        },
        headers={"Authorization": f"Bearer {api_key}"} if api_key is not None else {},
    )


@pytest.mark.parametrize(
    ("roles", "keep"),
    [
        ([], 0),
        (["user", "assistant", "user"], 0),
        (["system"], 1),
        (["system", "system", "user"], 2),
        (["system", "user", "assistant", "user", "assistant"], 1),
        (["system", "user", "assistant", "tool", "assistant", "user"], 1),
        (["system", "user", "system", "assistant"], 1),
        (["user", "system"], 0),
    ],
)
def test_drop_kv_cache_retains_only_leading_system_messages(roles: list[str], keep: int) -> None:
    messages = [RoleMessage(role=role, content=f"message-{i}") for i, role in enumerate(roles)]
    chat_task = ChatTask.create_from_messages(messages)
    original = list(messages)

    task = DropKVCacheTask(chat_task, worker_tag="drop")

    assert task.messages_to_retain == original[:keep]
    assert task.chat_task is chat_task
    assert task.worker_tag == "drop"
    assert chat_task.messages == original


def test_drop_kv_cache_handler_sends_system_prefix() -> None:
    client = SimpleNamespace(base_url="http://localhost:8000/v1/", api_key=None)
    worker = OpenaiWorker(client, model="test-model", kv_cache_hint_enabled=True)
    client.create_kv_cache_hint = AsyncMock(return_value=httpx.Response(200))
    messages = [
        RoleMessage(role="system", content="shared instructions"),
        RoleMessage(role="user", content="first question"),
        RoleMessage(role="assistant", content="answer"),
        RoleMessage(role="user", content="follow-up question"),
    ]
    task = DropKVCacheTask(ChatTask.create_from_messages(messages), worker_tag="drop")

    assert asyncio.run(worker.run_task(task)) == TaskStatus.SUCCESS
    client.create_kv_cache_hint.assert_awaited_once()
    sent_task, params = client.create_kv_cache_hint.call_args.args
    assert sent_task.messages_to_retain == messages[:1]
    assert params["messages"] == [message.to_dict() for message in messages]


@pytest.mark.parametrize("phase", ["INITIAL", "INSTRUCTION", "LAST_INSTRUCTION"])
def test_iter_research_keeps_task_context_out_of_system_prompt(phase: str) -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "tensorrt_llm/scaffolding/contrib/iter_research/prompts.py"
    )
    # Load prompt constants without importing the research agent's optional tool dependencies.
    spec = importlib.util.spec_from_file_location("iter_research_prompts_under_test", path)
    assert spec is not None and spec.loader is not None
    prompts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompts)
    system_template = getattr(prompts, f"{phase}_SYSTEM_PROMPT")
    user_template = getattr(prompts, f"{phase}_INPUT_PROMPT")
    first = {
        "question": "unique-question-one",
        "date_to_use": "2026-01-01",
        "tools": "unique-tool-definitions",
        "report": "unique-report",
        "action": "unique-action",
        "observation": "unique-observation",
    }
    second = {key: f"different-{value}" for key, value in first.items()}

    system = system_template.format(**first)
    assert system == system_template.format(**second)
    assert system.system_prompt_id == system_template.system_prompt_id
    assert first["question"] in user_template.format(**first)
    assert first["date_to_use"] in user_template.format(**first)
    if phase != "INITIAL":
        for key in ("report", "action", "observation"):
            assert first[key] in user_template.format(**first)
    if phase != "LAST_INSTRUCTION":
        assert first["tools"] in user_template.format(**first)
