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
import copy
import json
from enum import Enum
from typing import List

from tensorrt_llm.scaffolding import Controller, Task

# Reuse the MCP contrib's portable chat task/handler (see contrib/README.md:
# "a project can import Controller/Task/Worker from other projects"). We import
# from the submodules directly to avoid pulling in the `mcp` package via the
# mcp package __init__.
from tensorrt_llm.scaffolding.contrib.mcp.chat_task import ChatTask

from .a2a_task import A2AListTask, A2ASendTask


def _agent_to_tool(agent) -> dict:
    """Represent a remote agent as an OpenAI-style function the LLM can call.

    The LLM delegates to an agent by emitting a tool call whose name is the
    agent name and whose single argument is the ``message`` to forward.
    """
    description = agent.description or f"Remote agent '{agent.name}'."
    if agent.skills:
        description = f"{description} Skills: {', '.join(agent.skills)}."
    return {
        "type": "function",
        "function": {
            "name": agent.name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The natural-language request to send to this agent.",
                    }
                },
                "required": ["message"],
            },
        },
    }


class A2AController(Controller):
    """Route a user request to remote A2A agents, then summarize the replies.

    Flow (mirrors ``MCPController``):
      1. Discover reachable agents via an ``A2AListTask``.
      2. Ask the generation LLM which agent(s) to delegate to, exposing each
         agent as a callable tool.
      3. Dispatch the chosen ``A2ASendTask`` requests to the remote agents.
      4. Feed the agents' replies back to the LLM for a final answer.
    """

    class WorkerTag(Enum):
        GENERATION = "generation"
        A2A = "a2a"

    def __init__(self, custom_sampling_params: dict = None):
        super().__init__()
        self.custom_sampling_params = (
            copy.deepcopy(custom_sampling_params) if custom_sampling_params else None
        )

    def _apply_sampling_params(self, task):
        if not self.custom_sampling_params:
            return
        for key, value in self.custom_sampling_params.items():
            if hasattr(task, key) and getattr(task, key) is None:
                setattr(task, key, value)

    def process(self, tasks: List[Task], **kwargs):
        assert len(tasks) == 1, "A2AController handles a single task at a time."
        result_task = tasks[0]

        # 1. Discover the remote agents reachable through the A2A worker.
        list_task = A2AListTask.create_a2a_task(self.WorkerTag.A2A)
        yield [list_task]
        agents = list_task.result_agents or []
        available_tools = [_agent_to_tool(agent) for agent in agents]

        # 2. Let the LLM decide which agent (if any) should handle the request.
        system_message = (
            "You are an orchestrator agent with access to the remote agents "
            "exposed as tools below. When a remote agent is better suited to "
            "answer, delegate by calling it with a concise `message`. "
            "After receiving an agent's response, turn the raw result into a "
            "clear, concise answer for the user. Only call agents that are "
            "explicitly defined above."
        )
        messages = [{"role": "system", "content": system_message}]
        chat_task = ChatTask.create_from_prompt(messages, result_task.input_str, available_tools)
        chat_task.worker_tag = self.WorkerTag.GENERATION
        self._apply_sampling_params(chat_task)
        yield [chat_task]

        # 3. If the LLM did not delegate, return its direct answer.
        if chat_task.finish_reason != "tool_calls":
            result_task.output_str = chat_task.output_str
            return

        # 4. Dispatch each requested delegation to the remote agents.
        send_tasks = []
        for tool_call in chat_task.tool_calls:
            # Models occasionally emit malformed JSON (or a non-object payload)
            # in the tool-call arguments. Degrade gracefully instead of aborting
            # the whole controller flow with an unhandled exception.
            raw_args = tool_call.function.arguments
            try:
                parsed_args = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                parsed_args = None

            if isinstance(parsed_args, dict):
                message = parsed_args.get("message", "")
            else:
                # Forward the raw text so the remote agent still gets something
                # actionable rather than an empty request.
                message = raw_args if isinstance(raw_args, str) else ""

            send_tasks.append(
                A2ASendTask.create_a2a_task(tool_call.function.name, message, self.WorkerTag.A2A)
            )
        yield send_tasks

        agent_results = "\n".join(task.output_str for task in send_tasks if task.output_str)

        # 5. Summarize the agents' replies into a final answer.
        messages.append({"role": "assistant", "content": chat_task.output_str or ""})
        final_chat_task = ChatTask.create_from_prompt(messages, agent_results, available_tools)
        final_chat_task.worker_tag = self.WorkerTag.GENERATION
        self._apply_sampling_params(final_chat_task)
        yield [final_chat_task]
        result_task.output_str = final_chat_task.output_str
        return
