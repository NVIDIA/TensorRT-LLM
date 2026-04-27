import json
import random
import re
from datetime import datetime, timedelta

from tensorrt_llm.scaffolding.task import AssistantMessage, OpenAIToolDescription

TOOLS = [
    OpenAIToolDescription(
        name="tavily_search",
        description=(
            "Perform web searches via Tavily then returns a string of the top "
            "search results. Accepts multiple queries."
        ),
        parameters={
            "query": {
                "type": "array",
                "items": {"type": "string", "description": "The search query."},
                "minItems": 1,
                "description": "The list of search queries.",
            }
        },
    ),
    OpenAIToolDescription(
        name="Visit",
        description="Visit webpage(s) or paper(s) and return the summary of the content.",
        parameters={
            "url": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "description": "The URL(s) of the webpage(s) or paper(s) to visit.",
            },
            "goal": {
                "type": "string",
                "description": "The goal of the visit for webpage(s) or paper(s).",
            },
            "parse_type": {
                "type": "string",
                "enum": ["html", "pdf"],
                "default": "html",
                "description": "Specify whether to visit a HTML webpage or a PDF paper.",
            },
        },
    ),
    OpenAIToolDescription(
        name="PythonInterpreter",
        description=(
            "Executes arbitrary Python code in a secure, sandboxed environment. This "
            "tool is designed for performing complex calculations, data manipulations, "
            "string processing, logical operations, and general programming tasks. Use "
            "print() for any output you want to see."
        ),
        parameters={
            "code": {
                "type": "string",
                "description": (
                    "The Python code to execute. All output should be explicitly "
                    "printed using print() functions."
                ),
            }
        },
    ),
]


def extract_tags(text: str, tag: str) -> str:
    pattern = r"<{TAG}>(.*?)</{TAG}>".format(TAG=tag)
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def check_report_action(
    message: AssistantMessage,
    require_tool_call: bool = False,
    allow_tool_call: bool = True,
    allow_tool_call_without_report: bool = False,
) -> tuple:
    content = message.content or ""
    report = extract_tags(content, "report")
    text_action = extract_tags(content, "tool_call")
    answer = extract_tags(content, "answer")
    tool_calls = message.tool_calls or []

    if text_action:
        return False, "Tool call must use the native tool_calls field, not text tags!"
    if answer and tool_calls:
        return False, "Both answer and tool call found!"
    if len(tool_calls) > 1:
        return False, "Expected at most one tool call!"
    if require_tool_call and len(tool_calls) != 1:
        return False, "Expected exactly one tool call!"
    if tool_calls and not allow_tool_call:
        return False, "Tool call is not allowed in the final response!"
    if not report:
        if not (allow_tool_call_without_report and tool_calls and not answer):
            return False, "Report not found!"
    if not tool_calls and not answer:
        return False, "Neither answer nor tool call found!"

    return True, "success"


def random_date(start=None, end=None) -> str:
    if start is None:
        start = datetime(2023, 1, 1)
    if end is None:
        end = datetime(2025, 12, 31)
    delta = end - start
    random_days = random.randint(0, delta.days)
    final_date = start + timedelta(days=random_days)
    return final_date.strftime("%Y-%m-%d")


def get_tool_definitions() -> str:
    return json.dumps([tool.to_dict()["function"] for tool in TOOLS], indent=2)


def truncate_text(text: str, max_length: int) -> str:
    if len(text) > max_length:
        return text[:max_length]
    return text
