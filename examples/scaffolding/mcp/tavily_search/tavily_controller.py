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

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from tensorrt_llm.scaffolding import system_prompt
from tensorrt_llm.scaffolding.controller import Controller
from tensorrt_llm.scaffolding.task import ChatTask, MCPCallTask, SystemMessage, Task, UserMessage

_TAVILY_COMPRESS_SYSTEM_PROMPT = system_prompt(
    """You compress Tavily search output for a downstream research agent.

## Requirements
- Keep the original answer structure and result ordering.
- Preserve result numbers, titles, URLs, and useful date/source metadata.
- Keep core facts, numbers, names, claims, and evidence relevant to the user goal.
- Remove secondary, repeated, boilerplate, or low-value details.
- Do not add facts that are not in the Tavily output.
""",
    name="tavily_search.tavily_compress_system_prompt",
)

_TAVILY_COMPRESS_USER_PROMPT = """## User Goal
{goal}

## Tavily Search Output
{search_output}
"""

_DEFAULT_COMPRESS_THRESHOLD_CHARS = 6000
_DEFAULT_MAX_INPUT_CHARS = 24000
_DEFAULT_MAX_COMPRESSED_CHARS = 6000
_DEFAULT_MAX_OUTPUT_TOKENS = 2048

LOGGER = logging.getLogger(__name__)


def _truncate_with_notice(content: str, max_chars: int, source: str) -> str:
    if len(content) <= max_chars:
        return content
    truncated = content[:max_chars].rstrip()
    return (
        f"{truncated}\n\n"
        f"[truncated {source}: kept first {max_chars} characters to control prompt size]"
    )


def _last_assistant_content(chat_task: ChatTask) -> Optional[str]:
    for message in reversed(chat_task.messages):
        if getattr(message, "role", None) == "assistant":
            return message.content or ""
    return None


@dataclass
class TavilyTask(Task):
    """Queries and goal for Tavily search plus optional LLM compression."""

    query: Optional[List[str]] = field(default=None)
    goal: Optional[str] = field(default=None)
    result_str: Optional[str] = field(default=None)
    result_stdout: Optional[str] = field(default=None)
    result_stderr: Optional[str] = field(default=None)


class TavilyController(Controller):
    """MCP ``tavily_search``, then compress oversized search output."""

    class WorkerTag(Enum):
        TOOL_CALL = "tavily_tool_call"

    def __init__(
        self,
        generation_controller: Controller,
        compress_threshold_chars: int = _DEFAULT_COMPRESS_THRESHOLD_CHARS,
        max_input_chars: int = _DEFAULT_MAX_INPUT_CHARS,
        max_compressed_chars: int = _DEFAULT_MAX_COMPRESSED_CHARS,
        max_output_tokens: int = _DEFAULT_MAX_OUTPUT_TOKENS,
    ):
        super().__init__()
        self.generation_controller = generation_controller
        self.compress_threshold_chars = compress_threshold_chars
        self.max_input_chars = max_input_chars
        self.max_compressed_chars = max_compressed_chars
        self.max_output_tokens = max_output_tokens

    def clone(self):
        return TavilyController(
            generation_controller=self.generation_controller.clone(),
            compress_threshold_chars=self.compress_threshold_chars,
            max_input_chars=self.max_input_chars,
            max_compressed_chars=self.max_compressed_chars,
            max_output_tokens=self.max_output_tokens,
        )

    def process(self, tasks: List[Task], **kwargs):
        assert len(tasks) == 1 and isinstance(tasks[0], TavilyTask), (
            "TavilyController only supports a single TavilyTask"
        )
        tavily_task = tasks[0]
        LOGGER.warning(
            "[tavily] search start: query_count=%d compress_threshold_chars=%d",
            len(tavily_task.query or []),
            self.compress_threshold_chars,
        )

        mcp_task = MCPCallTask.create_mcptask(
            tool_call_id="tavily_search",
            tool_name="tavily_search",
            args=json.dumps({"query": tavily_task.query or []}),
            worker_tag=self.WorkerTag.TOOL_CALL,
        )
        yield [mcp_task]

        raw_result = mcp_task.result_str or ""
        tavily_task.result_stdout = mcp_task.result_stdout
        tavily_task.result_stderr = mcp_task.result_stderr
        if len(raw_result) <= self.compress_threshold_chars:
            tavily_task.result_str = raw_result
            return

        LOGGER.warning("[tavily] compressing search output: raw_chars=%d", len(raw_result))
        bounded_result = _truncate_with_notice(raw_result, self.max_input_chars, "Tavily output")
        user_prompt = _TAVILY_COMPRESS_USER_PROMPT.format(
            goal=tavily_task.goal or "Extract relevant information for the research task.",
            search_output=bounded_result,
        )
        chat_task = ChatTask.create_from_messages(
            [
                SystemMessage(content=_TAVILY_COMPRESS_SYSTEM_PROMPT),
                UserMessage(user_prompt),
            ]
        )
        chat_task.max_tokens = self.max_output_tokens
        chat_task.temperature = 0.0

        yield from self.generation_controller.process([chat_task])

        compressed = _last_assistant_content(chat_task)
        if compressed is None:
            tavily_task.result_str = raw_result
            return

        tavily_task.result_str = _truncate_with_notice(
            compressed.strip(),
            self.max_compressed_chars,
            "compressed Tavily output",
        )
