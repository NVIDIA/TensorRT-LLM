import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from tensorrt_llm.scaffolding import system_prompt
from tensorrt_llm.scaffolding.controller import Controller
from tensorrt_llm.scaffolding.task import ChatTask, MCPCallTask, SystemMessage, Task, UserMessage

_VISIT_MAX_PROMPT_CHARS = 15000
_VISIT_PDF_MAX_PROMPT_CHARS = 10000
_VISIT_PDF_MAX_LINES = 200
_VISIT_MAX_RESULT_CHARS = 2000
_VISIT_EXTRACTOR_SYSTEM_PROMPT = system_prompt(
    """You process webpage content and a user goal to extract relevant information.

## **Task Guidelines**
1. **Content Scanning for Rational**:
    Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**:
    Identify and extract the **most relevant information** from the content, you never miss any important information,
    output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**:
    Organize into a concise paragraph with logical flow,
    prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" fields**
""",
    name="fetch_webpage.visit_extractor_system_prompt",
)

_VISIT_EXTRACTOR_USER_PROMPT = """## **Webpage Content**
{webpage_content}

## **User Goal**
{goal}
"""

LOGGER = logging.getLogger(__name__)


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
        LOGGER.warning(
            "[visit] fetch_webpage start: url_count=%d parse_type=%s max_webpage_tokens=%d",
            len(visit_task.urls or []),
            visit_task.parse_type,
            self.max_webpage_tokens,
        )

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
        LOGGER.warning(
            "[visit] fetch_webpage raw result: chars=%d parse_type=%s",
            len(raw_content),
            visit_task.parse_type,
        )
        if not raw_content:
            visit_task.result_str = "[visit] Failed to fetch webpage content."
            return

        prepared_content = _prepare_webpage_content(
            raw_content,
            visit_task.parse_type,
            self.max_webpage_tokens,
        )
        LOGGER.warning(
            "[visit] prepared webpage content: raw_chars=%d prepared_chars=%d parse_type=%s",
            len(raw_content),
            len(prepared_content),
            visit_task.parse_type,
        )
        if not prepared_content.strip():
            visit_task.result_str = "[visit] Failed to prepare webpage content for summarization."
            return

        user_prompt = _VISIT_EXTRACTOR_USER_PROMPT.format(
            webpage_content=prepared_content,
            goal=visit_task.goal,
        )
        LOGGER.warning(
            "[visit] extractor prompt: user_chars=%d goal_chars=%d prepared_chars=%d",
            len(user_prompt),
            len(visit_task.goal or ""),
            len(prepared_content),
        )
        chat_task = ChatTask.create_from_messages(
            [
                SystemMessage(content=_VISIT_EXTRACTOR_SYSTEM_PROMPT),
                UserMessage(user_prompt),
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
