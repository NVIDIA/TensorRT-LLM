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
"""Offline unit tests for the A2A scaffolding contrib.

These tests inject fake remote-agent connections into ``A2AWorker`` so they
require neither the ``a2a-sdk`` package nor any network access, mirroring the
``DummyWorker`` approach in ``test_mcp_worker.py``.
"""

import json

import pytest

from tensorrt_llm.scaffolding import ScaffoldingLlm, TaskStatus, Worker
from tensorrt_llm.scaffolding.contrib.a2a import (
    A2AController,
    A2AListTask,
    A2ASendTask,
    A2AWorker,
    AgentInfo,
)
from tensorrt_llm.scaffolding.contrib.mcp.chat_task import ChatTask

# ============================================================
# Fakes
# ============================================================


class FakeA2AConnection:
    """Stand-in for A2AAgentConnection that echoes messages without network."""

    def __init__(self, name: str, description: str = "", skills=None):
        self._info = AgentInfo(name=name, description=description, skills=skills or [])
        self.cleaned_up = False

    def get_agent_info(self) -> AgentInfo:
        return self._info

    async def send_message(self, message: str) -> str:
        return f"[{self._info.name}] handled: {message}"

    async def cleanup(self):
        self.cleaned_up = True


class FunctionCall:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class ToolCall:
    def __init__(self, name: str, arguments: str):
        self.function = FunctionCall(name, arguments)


class DummyGenerationWorker(Worker):
    """Deterministic generation worker: first delegates, then summarizes."""

    async def dummy_handler(self, task: ChatTask):
        # The controller seeds [system, user]; after delegation it appends an
        # assistant message and a user message carrying the agent reply.
        if len(task.messages) == 2:
            task.output_str = "delegating to weather_agent"
            task.tool_calls = [ToolCall("weather_agent", json.dumps({"message": "weather in LA?"}))]
            task.finish_reason = "tool_calls"
        else:
            task.output_str = "It is sunny in LA."
            task.tool_calls = None
            task.finish_reason = "stop"
        return TaskStatus.SUCCESS

    task_handlers = {ChatTask: dummy_handler}


# ============================================================
# Worker-level tests
# ============================================================


@pytest.mark.asyncio
async def test_a2a_worker_list_agents():
    worker = A2AWorker(
        [
            FakeA2AConnection("weather_agent", "Provides weather", ["forecast"]),
            FakeA2AConnection("math_agent", "Does arithmetic"),
        ]
    )
    task = A2AListTask.create_a2a_task()
    status = await worker.run_task(task)

    assert status == TaskStatus.SUCCESS
    names = [agent.name for agent in task.result_agents]
    assert names == ["weather_agent", "math_agent"]
    assert task.result_agents[0].skills == ["forecast"]


@pytest.mark.asyncio
async def test_a2a_worker_send_routes_to_named_agent():
    worker = A2AWorker(
        [
            FakeA2AConnection("weather_agent"),
            FakeA2AConnection("math_agent"),
        ]
    )
    task = A2ASendTask.create_a2a_task("math_agent", "1 + 1")
    status = await worker.run_task(task)

    assert status == TaskStatus.SUCCESS
    assert task.output_str == "[math_agent] handled: 1 + 1"


@pytest.mark.asyncio
async def test_a2a_worker_send_unknown_agent():
    worker = A2AWorker([FakeA2AConnection("weather_agent")])
    task = A2ASendTask.create_a2a_task("missing_agent", "hello")
    status = await worker.run_task(task)

    assert status == TaskStatus.WORKER_NOT_SUPPORTED
    assert "missing_agent" in task.output_str


@pytest.mark.asyncio
async def test_a2a_worker_async_shutdown():
    connections = [FakeA2AConnection("a"), FakeA2AConnection("b")]
    worker = A2AWorker(connections)
    await worker.async_shutdown()
    assert all(connection.cleaned_up for connection in connections)


# ============================================================
# End-to-end controller test through ScaffoldingLlm
# ============================================================


@pytest.mark.asyncio
async def test_scaffolding_with_a2a_controller():
    a2a_worker = A2AWorker(
        [
            FakeA2AConnection("weather_agent", "Provides weather", ["forecast"]),
        ]
    )
    generation_worker = DummyGenerationWorker()

    controller = A2AController()
    scaffolding_llm = ScaffoldingLlm(
        controller,
        {
            A2AController.WorkerTag.GENERATION: generation_worker,
            A2AController.WorkerTag.A2A: a2a_worker,
        },
    )

    try:
        future = scaffolding_llm.generate_async("What's the weather in LA?")
        result = await future.aresult()
        assert isinstance(result.outputs[0].text, str)
        assert result.outputs[0].text == "It is sunny in LA."
    finally:
        await a2a_worker.async_shutdown()
        scaffolding_llm.shutdown(shutdown_workers=False)
