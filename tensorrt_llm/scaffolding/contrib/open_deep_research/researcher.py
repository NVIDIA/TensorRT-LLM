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

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from tensorrt_llm.logger import logger
from tensorrt_llm.scaffolding import (
    AssistantMessage,
    ChatTask,
    Controller,
    MCPCallTask,
    SystemMessage,
    Task,
    UserMessage,
)
from tensorrt_llm.scaffolding.contrib.mcp.fetch_webpage import VisitController, VisitTask
from tensorrt_llm.scaffolding.controller import ChatWithMCPController
from tensorrt_llm.scaffolding.task import ToolMessage
from tensorrt_llm.scaffolding.task_collection import sub_request_node

from .prompts import RESEARCHER_SYSTEM_PROMPT
from .tools import (
    fetch_webpage_tool,
    google_scholar_tool,
    python_interpreter_tool,
    reflection_tool,
    tavily_search_tool,
)
from .utils import get_today_str


@dataclass
class ResearchTask(Task):
    research_topic: str = field(default=None)
    research_result: str = field(default=None)
    tool_call_id: str = field(default=None)

    @staticmethod
    def from_topic(topic: str, tool_call_id: str) -> "ResearchTask":
        return ResearchTask(research_topic=topic, research_result="", tool_call_id=tool_call_id)


class Compressor(Controller):
    def __init__(
        self,
        generation_controller: Controller,
        system_prompts: list[SystemMessage],
        max_iterations: int = 3,
    ):
        super().__init__()
        self.generation_controller = generation_controller
        self.system_prompts = system_prompts
        self.max_iterations = max_iterations

    def clone(self):
        return Compressor(
            self.generation_controller.clone(), self.system_prompts, self.max_iterations
        )

    def process(self, tasks: List[Task], **kwargs):
        assert len(tasks) == 1 and isinstance(tasks[0], ChatTask), (
            "Compressor only supports one ChatTask"
        )
        compress_task = ChatTask.create_from_prompt(None, self.system_prompts)
        compress_task.add_message(UserMessage(str([str(message) for message in tasks[0].messages])))

        for i in range(self.max_iterations):
            yield from self.generation_controller.process([compress_task])
            if compress_task.finish_reason == "stop":
                break
            if i < self.max_iterations - 1:
                compress_task.messages.pop()

        last_message = compress_task.messages[-1]
        assert isinstance(last_message, AssistantMessage), (
            f"last_message is not AssistantMessage, {type(last_message)=}"
        )
        tasks[0].output_str = last_message.content
        return


def _extraction_goal_from_chat(chat_task: ChatTask) -> str:
    """Goal string for webpage extraction (matches IterResearch ``Visit`` goal)."""
    for message in chat_task.messages:
        if message.role == "user" and getattr(message, "content", None):
            return str(message.content)
    return "Extract relevant information for the research task."


def _parse_tool_arguments(arguments: Any) -> dict:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        return json.loads(arguments)
    raise TypeError(f"Unsupported tool arguments type: {type(arguments).__name__}")


class ResearchChatWithMCPController(ChatWithMCPController):
    """Dispatches ``fetch_webpage`` through :class:`VisitController` (fetch + LLM summary).

    Other MCP tools keep the same batching behavior as :class:`ChatWithMCPController`.
    """

    def __init__(
        self,
        generation_controller: Controller,
        visit_controller: VisitController,
        system_prompts: Any = None,
        max_iterations: int = 3,
        tools: Any = None,
    ):
        super().__init__(
            generation_controller=generation_controller,
            system_prompts=system_prompts,
            max_iterations=max_iterations,
            tools=tools,
        )
        self.visit_controller = visit_controller

    def clone(self) -> ResearchChatWithMCPController:
        return ResearchChatWithMCPController(
            generation_controller=self.generation_controller.clone(),
            visit_controller=self.visit_controller.clone(),
            system_prompts=self.system_prompts,
            max_iterations=self.max_iterations,
            tools=self.tools,
        )

    def process(self, tasks: List[Task], **kwargs):
        assert len(tasks) == 1, "ResearchChatWithMCPController only supports one task"
        chat_task = tasks[0]
        assert isinstance(chat_task, ChatTask), (
            "ResearchChatWithMCPController only supports ChatTask"
        )
        goal = _extraction_goal_from_chat(chat_task)

        for _ in range(self.max_iterations):
            yield from self.generation_controller.process([chat_task])
            response_message = chat_task.messages[-1]
            if not isinstance(response_message, AssistantMessage):
                logger.warning(
                    "Stopping ChatWithMCP tool loop: expected AssistantMessage "
                    "after generation, got %s",
                    type(response_message).__name__,
                )
                break
            if response_message.tool_calls:
                tool_calls = response_message.tool_calls

                index_to_payload: Dict[int, Tuple[str, Optional[str], Optional[str]]] = {}

                mcp_tasks: List[MCPCallTask] = []
                mcp_tool_indices: List[int] = []
                for i, tool_call in enumerate(tool_calls):
                    if tool_call.function.name != "fetch_webpage":
                        mcp_tasks.append(
                            MCPCallTask.create_mcptask(
                                tool_call.id,
                                tool_call.function.name,
                                tool_call.function.arguments,
                                self.WorkerTag.TOOLCALL,
                            )
                        )
                        mcp_tool_indices.append(i)

                if mcp_tasks:
                    yield mcp_tasks

                for batch_j, call_i in enumerate(mcp_tool_indices):
                    mt = mcp_tasks[batch_j]
                    if mt.result_str is not None:
                        index_to_payload[call_i] = (
                            mt.result_str,
                            mt.result_stdout,
                            mt.result_stderr,
                        )

                for i, tool_call in enumerate(tool_calls):
                    if tool_call.function.name != "fetch_webpage":
                        continue
                    args = _parse_tool_arguments(tool_call.function.arguments)
                    urls = args.get("url", [])
                    if isinstance(urls, str):
                        urls = [urls]
                    visit_task = VisitTask(
                        urls=urls or [],
                        goal=goal,
                        parse_type=args.get("parse_type", "html"),
                    )
                    yield from self.visit_controller.process([visit_task])
                    if visit_task.result_str is not None:
                        index_to_payload[i] = (
                            visit_task.result_str,
                            None,
                            None,
                        )

                for i, tool_call in enumerate(tool_calls):
                    payload = index_to_payload.get(i)
                    if payload is None:
                        continue
                    body, out, err = payload
                    chat_task.add_message(
                        ToolMessage(body, tool_call.id, trace_stdout=out, trace_stderr=err)
                    )
                if any(tc.function.name == "complete_task" for tc in tool_calls):
                    break
            else:
                break


@sub_request_node("Researcher")
# @drop_kv_cache_scope()
class Researcher(Controller):
    tools = [
        tavily_search_tool,
        google_scholar_tool,
        fetch_webpage_tool,
        python_interpreter_tool,
        reflection_tool,
    ]

    def __init__(self, chat_with_tools_controller: Controller, compress_controller: Controller):
        super().__init__()
        self.chat_with_tools_controller = chat_with_tools_controller
        self.compress_controller = compress_controller

    def clone(self):
        return Researcher(
            chat_with_tools_controller=self.chat_with_tools_controller.clone(),
            compress_controller=self.compress_controller.clone(),
        )

    def process(self, research_tasks: List[ResearchTask], **kwargs):
        assert len(research_tasks) == 1, "Researcher only supports one ResearchTask"
        assert research_tasks[0].research_topic is not None, (
            "ResearchTask must have a research topic"
        )
        assert research_tasks[0].tool_call_id is not None, "ResearchTask must have a tool call id"

        chat_task = ChatTask.create_from_prompt(
            research_tasks[0].research_topic,
            [
                SystemMessage(
                    RESEARCHER_SYSTEM_PROMPT.format(date=get_today_str()),
                )
            ],
            tools=self.tools,
        )

        yield from self.chat_with_tools_controller.process([chat_task])

        yield from self.compress_controller.process([chat_task])

        research_tasks[0].research_findings = chat_task.output_str
        return
