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
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from tensorrt_llm.scaffolding.controller import Controller
from tensorrt_llm.scaffolding.task import ChatTask, MCPCallTask, Task, UserMessage

from ...iter_research.prompts import VISIT_EXTRACTOR_PROMPT

_VISIT_MAX_PROMPT_CHARS = 32000
_VISIT_PDF_MAX_PROMPT_CHARS = 12000
_VISIT_PDF_MAX_LINES = 240
_VISIT_MAX_RESULT_CHARS = 12000


def _looks_like_raw_pdf(content: str) -> bool:
    text = content.lstrip()
    if text.startswith("%PDF-"):
        return True
    pdf_markers = (" obj", "endobj", "stream", "endstream", "/Type/Page")
    hit_count = sum(1 for marker in pdf_markers if marker in content)
    return hit_count >= 3


def _truncate_with_notice(content: str, max_chars: int, source: str) -> str:
    if len(content) <= max_chars:
        return content
    truncated = content[:max_chars]
    return (
        f"{truncated}\n\n"
        f"[truncated {source}: kept first {max_chars} characters to control prompt size]"
    )


def _sanitize_text(content: str) -> str:
    sanitized_chars: List[str] = []
    for ch in content:
        if ch in ("\n", "\t"):
            sanitized_chars.append(ch)
            continue
        if ch.isprintable():
            sanitized_chars.append(ch)
    return "".join(sanitized_chars)


def _extract_pdf_text_lines(raw_content: str) -> List[str]:
    syntax_line = re.compile(
        r"^\s*(\d+\s+\d+\s+obj|endobj|stream|endstream|xref|trailer|%%EOF)\s*$"
    )
    mostly_punct = re.compile(r"^[^A-Za-z0-9\u4e00-\u9fff\u3040-\u30ff]+$")
    extracted: List[str] = []
    for raw_line in raw_content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if syntax_line.match(line):
            continue
        if mostly_punct.match(line):
            continue
        line = re.sub(r"\s+", " ", line)
        if len(line) < 8:
            continue
        if len(line) > 300:
            line = line[:300]
        extracted.append(line)
        if len(extracted) >= _VISIT_PDF_MAX_LINES:
            break
    return extracted


def _prepare_webpage_content(raw_content: str, parse_type: str, max_webpage_tokens: int) -> str:
    # Keep prompt size conservative because binary/garbled text can explode tokenization.
    max_chars = min(max_webpage_tokens * 2, _VISIT_MAX_PROMPT_CHARS)
    is_pdf = parse_type == "pdf" or _looks_like_raw_pdf(raw_content)
    if is_pdf:
        pdf_max_chars = min(max_webpage_tokens, _VISIT_PDF_MAX_PROMPT_CHARS)
        extracted_lines = _extract_pdf_text_lines(raw_content)
        if extracted_lines:
            structured_text = _sanitize_text("\n".join(extracted_lines))
            return _truncate_with_notice(
                structured_text,
                pdf_max_chars,
                "PDF structured extraction",
            )
        return _truncate_with_notice(_sanitize_text(raw_content), pdf_max_chars, "raw PDF content")
    return _truncate_with_notice(_sanitize_text(raw_content), max_chars, "webpage content")


def _get_last_assistant_message_content(chat_task: ChatTask) -> Optional[str]:
    for message in reversed(chat_task.messages):
        if getattr(message, "role", None) == "assistant":
            return message.content or ""
    return None


def _parse_json_output(raw: str) -> Optional[dict]:
    if not raw:
        return None
    triple_match = re.search(r"```json\s*\n(.*?)\n```", raw, re.DOTALL)
    if triple_match:
        json_str = triple_match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


@dataclass
class VisitTask(Task):
    """URLs, goal, and parse_type for Visit (fetch + summarize)."""

    urls: Optional[list] = field(default=None)
    goal: Optional[str] = field(default=None)
    parse_type: str = field(default="html")
    result_str: Optional[str] = field(default=None)


class VisitController(Controller):
    """MCP ``fetch_webpage``, truncate, then LLM summarization."""

    class WorkerTag(Enum):
        TOOL_CALL = "visit_tool_call"

    def __init__(self, generation_controller: Controller, max_webpage_tokens: int = 48000):
        super().__init__()
        self.generation_controller = generation_controller
        self.max_webpage_tokens = max_webpage_tokens

    def clone(self):
        return VisitController(
            generation_controller=self.generation_controller.clone(),
            max_webpage_tokens=self.max_webpage_tokens,
        )

    def process(self, tasks: List[Task], **kwargs):
        assert len(tasks) == 1 and isinstance(tasks[0], VisitTask), (
            "VisitController only supports a single VisitTask"
        )
        visit_task = tasks[0]

        mcp_task = MCPCallTask.create_mcptask(
            tool_call_id="visit_fetch",
            tool_name="fetch_webpage",
            args=json.dumps(
                {
                    "url": visit_task.urls,
                    "parse_type": visit_task.parse_type,
                }
            ),
            worker_tag=self.WorkerTag.TOOL_CALL,
        )
        yield [mcp_task]

        raw_content = mcp_task.result_str or ""
        if not raw_content:
            visit_task.result_str = "[visit] Failed to fetch webpage content."
            return

        prepared_content = _prepare_webpage_content(
            raw_content,
            visit_task.parse_type,
            self.max_webpage_tokens,
        )
        if not prepared_content.strip():
            visit_task.result_str = "[visit] Failed to prepare webpage content for summarization."
            return

        extractor_prompt = VISIT_EXTRACTOR_PROMPT.format(
            webpage_content=prepared_content,
            goal=visit_task.goal,
        )
        chat_task = ChatTask.create_from_messages(
            [
                UserMessage(extractor_prompt),
            ]
        )
        yield from self.generation_controller.process([chat_task])

        urls_str = ", ".join(visit_task.urls) if visit_task.urls else "unknown"
        llm_response = _get_last_assistant_message_content(chat_task)
        if llm_response is None:
            visit_task.result_str = (
                f"[visit] Failed to summarize webpage content from '{urls_str}': "
                "no assistant response was produced."
            )
            return

        parsed = _parse_json_output(llm_response)
        useful_info = (
            f"The useful information in '{urls_str}' for user goal '{visit_task.goal}':\n\n"
        )

        if parsed and "evidence" in parsed and "summary" in parsed:
            useful_info += f"Evidence in page:\n{parsed['evidence']}\n\n"
            useful_info += f"Summary:\n{parsed['summary']}\n"
        else:
            useful_info += llm_response

        visit_task.result_str = _truncate_with_notice(
            useful_info,
            _VISIT_MAX_RESULT_CHARS,
            "visit summary",
        )
