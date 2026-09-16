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
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tensorrt_llm.scaffolding.worker import OpenaiWorker


@pytest.mark.parametrize("suffix", ["", "/", "/v1", "/v1/"])
@pytest.mark.parametrize("api_key", [None, "test-key"])
def test_kv_cache_hint_control_endpoint(suffix: str, api_key: str | None) -> None:
    client = SimpleNamespace(base_url=f"http://localhost:8000{suffix}", api_key=api_key)
    OpenaiWorker(async_client=client, model="test-model", kv_cache_hint_enabled=True)
    message = MagicMock()
    message.to_dict.return_value = {"role": "user", "content": "hello"}
    task = SimpleNamespace(
        chat_task=SimpleNamespace(messages=[message]), messages_to_retain=[message]
    )
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
            "messages": [{"role": "user", "content": "hello"}],
            "messages_to_retain": [{"role": "user", "content": "hello"}],
            "model": "test-model",
            "chat_template_kwargs": {"enable_thinking": False},
        },
        headers={"Authorization": f"Bearer {api_key}"} if api_key is not None else {},
    )
