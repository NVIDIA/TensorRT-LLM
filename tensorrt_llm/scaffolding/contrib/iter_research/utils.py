import json
import random
import re
from datetime import datetime, timedelta

TOOLS = [
    # {
    #     "name": "google_search",
    #     "description":
    #     "Perform Google web searches then returns a string of the top search results. Accepts multiple queries.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "query": {
    #                 "type": "array",
    #                 "items": {
    #                     "type": "string",
    #                     "description": "The search query."
    #                 },
    #                 "minItems": 1,
    #                 "description": "The list of search queries."
    #             }
    #         },
    #         "required": ["query"]
    #     }
    # },
    {
        "name": "tavily_search",
        "description": {
            "Perform web searches via Tavily then returns a string of the top search results."
            "Accepts multiple queries."
        },
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "array",
                    "items": {"type": "string", "description": "The search query."},
                    "minItems": 1,
                    "description": "The list of search queries.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "google_scholar",
        "description": (
            "Leverage Google Scholar to retrieve relevant information from academic "
            "publications. Accepts multiple queries. This tool will also return "
            "results from google search"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "array",
                    "items": {"type": "string", "description": "The search query."},
                    "minItems": 1,
                    "description": "The list of search queries for Google Scholar.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "Visit",
        "description": "Visit webpage(s) or paper(s) and return the summary of the content.",
        "parameters": {
            "type": "object",
            "properties": {
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
            "required": ["url", "goal"],
        },
    },
    {
        "name": "PythonInterpreter",
        "description": (
            "Executes arbitrary Python code in a secure, sandboxed environment. This "
            "tool is designed for performing complex calculations, data manipulations, "
            "string processing, logical operations, and general programming tasks. Use "
            "print() for any output you want to see."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": {
                        "The Python code to execute. "
                        "All output should be explicitly printed using print() functions."
                    },
                }
            },
            "required": ["code"],
        },
    },
]


def extract_tags(text: str, tag: str) -> str:
    pattern = r"<{TAG}>(.*?)</{TAG}>".format(TAG=tag)
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def check_report_action(content: str) -> tuple:
    report = extract_tags(content, "report")
    action = extract_tags(content, "tool_call")
    answer = extract_tags(content, "answer")

    if action:
        try:
            tool_call = json.loads(action)
            assert isinstance(tool_call, dict)
            assert "arguments" in tool_call
        except (json.JSONDecodeError, AssertionError):
            return False, "Tool parse error!"

    if not report:
        return False, "Report not found!"

    if not action and not answer:
        return False, "Neither answer nor action found!"

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
    return json.dumps(TOOLS, indent=2)


def truncate_text(text: str, max_length: int) -> str:
    if len(text) > max_length:
        return text[:max_length]
    return text
