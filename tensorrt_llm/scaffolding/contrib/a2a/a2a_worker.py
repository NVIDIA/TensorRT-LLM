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
import asyncio
from typing import List

from tensorrt_llm.scaffolding import TaskStatus, Worker

from .a2a_task import A2AListTask, A2ASendTask
from .a2a_utils import A2AAgentConnection


class A2AWorker(Worker):
    """A Scaffolding worker that delegates work to remote A2A-protocol agents.

    The worker holds one connection per remote agent. It mirrors ``MCPWorker``:
    the ``A2AListTask`` discovers the reachable agents (so a controller can let
    the LLM decide where to route), and the ``A2ASendTask`` dispatches a message
    to a named agent and collects its reply.

    ``connections`` is any object exposing ``get_agent_info()`` and an async
    ``send_message(text)`` (plus an optional async ``cleanup()``). Production
    code uses :class:`A2AAgentConnection`; tests can inject fakes so they need
    neither ``a2a-sdk`` nor network access.
    """

    def __init__(self, connections: List):
        self.connections = connections

    @classmethod
    async def init_with_urls(cls, urls: List[str]) -> "A2AWorker":
        connections = []
        for url in urls:
            connection = A2AAgentConnection()
            await connection.connect(url)
            connections.append(connection)
        return cls(connections)

    async def list_handler(self, task: A2AListTask) -> TaskStatus:
        task.result_agents = [connection.get_agent_info() for connection in self.connections]
        return TaskStatus.SUCCESS

    async def send_handler(self, task: A2ASendTask) -> TaskStatus:
        for connection in self.connections:
            if connection.get_agent_info().name != task.agent_name:
                continue
            task.output_str = await connection.send_message(task.message)
            return TaskStatus.SUCCESS

        # No remote agent matched the requested name. Surface a clear message
        # so the controller can still produce a final answer.
        task.output_str = f"No remote A2A agent named '{task.agent_name}' is available."
        return TaskStatus.WORKER_NOT_SUPPORTED

    async def async_shutdown(self):
        """Close all remote-agent connections."""
        for connection in self.connections:
            cleanup = getattr(connection, "cleanup", None)
            if cleanup is not None:
                await cleanup()

    def shutdown(self):
        """Best-effort synchronous shutdown.

        Prefer :meth:`async_shutdown` from inside an event loop; this fallback
        is provided to satisfy the :class:`Worker` interface.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            loop.create_task(self.async_shutdown())
        else:
            asyncio.run(self.async_shutdown())

    task_handlers = {A2AListTask: list_handler, A2ASendTask: send_handler}
