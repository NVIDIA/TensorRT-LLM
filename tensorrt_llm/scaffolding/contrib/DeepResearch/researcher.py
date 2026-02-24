import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List

from tensorrt_llm.scaffolding import Controller, Task
from tensorrt_llm.scaffolding.contrib.mcp import ChatTask, MCPCallTask, MCPController

from .prompts import (
    compress_research_simple_human_message,
    compress_system_prompt,
    research_system_prompt,
)
from .utils import AssistantMessage, SystemMessage, UserMessage, get_today_str

LOGGER = logging.getLogger()


@dataclass
class ResearchTask(Task):
    research_topic: str = field(default=None)
    research_result: str = field(default=None)

    @staticmethod
    def from_topic(topic: str) -> "ResearchTask":
        task = ResearchTask()
        task.research_topic = topic
        task.research_result = ""
        return task


class Researcher(Controller):
    class WorkerTag(Enum):
        GENERATION = "generation"

    def __init__(self, max_tools_iter: int = 3, max_compress_iter: int = 3):
        # TODO: Add more tools (e.g., MCP tools) beyond search.
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "tavily_search",
                    "description": "For conducting web searches to gather information",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "reflection",
                    "description": "For reflection and strategic planning during research",
                    "parameters": {
                        "type": "object",
                        "properties": {"reflection": {"type": "string"}},
                    },
                },
            },
        ]

        self.max_tools_iter = max_tools_iter
        self.max_compress_iter = max_compress_iter

    def process(self, research_tasks: List[ResearchTask], **kwargs):
        for research_task in research_tasks:
            research_prompt_messages = [
                SystemMessage(research_system_prompt.format(date=get_today_str())).to_dict(),
                UserMessage(research_task.research_topic).to_dict(),
            ]

            research_tools_messages = []
            chat_with_tools_task = ChatTask.from_messages(
                research_prompt_messages + research_tools_messages, self.tools
            )
            chat_with_tools_task.worker_tag = Researcher.WorkerTag.GENERATION

            for _ in range(self.max_tools_iter):
                yield [chat_with_tools_task]

                if chat_with_tools_task.finish_reason != "tool_calls":
                    break

                if chat_with_tools_task.output_str is not None:
                    research_tools_messages.append(
                        AssistantMessage(chat_with_tools_task.output_str).to_dict()
                    )

                mcp_call_tasks = [
                    MCPCallTask.create_mcptask(
                        tool_call.function.name, tool_call.function.arguments
                    )
                    for tool_call in chat_with_tools_task.tool_calls
                ]

                for mcp_call_task in mcp_call_tasks:
                    mcp_call_task.worker_tag = MCPController.WorkerTag.MCP

                yield mcp_call_tasks

                for mcp_call_task in mcp_call_tasks:
                    research_tools_messages.append(UserMessage(mcp_call_task.output_str).to_dict())

                chat_with_tools_task = ChatTask.from_messages(
                    research_prompt_messages + research_tools_messages, self.tools
                )
                chat_with_tools_task.worker_tag = Researcher.WorkerTag.GENERATION

            compress_prompt_messages = [
                SystemMessage(compress_system_prompt.format(date=get_today_str())).to_dict()
            ]

            compress_messages = research_tools_messages + [
                UserMessage(compress_research_simple_human_message).to_dict()
            ]
            compress_task = ChatTask.from_messages(compress_prompt_messages + compress_messages)
            compress_task.worker_tag = Researcher.WorkerTag.GENERATION

            for _ in range(self.max_compress_iter):
                yield [compress_task]
                research_task.research_result = compress_task.output_str
                if compress_task.finish_reason == "finish":
                    break
        return
