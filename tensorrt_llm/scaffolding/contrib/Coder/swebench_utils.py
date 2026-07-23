"""SWE-bench helpers for extracting and normalizing the final patch.

These utilities turn the agent's ``complete_task`` tool call into the
``preds.json`` fields SWE-bench expects (a normalized ``model_patch`` plus a
``summary``). They are used by :class:`SWEBenchCoder` in ``coder.py`` and kept
separate so the controller/factory code there stays focused on orchestration.
"""

# ruff: noqa: E501

import json
import re
from typing import Any, List, Optional

from tensorrt_llm.scaffolding.task import AssistantMessage, ChatTask, ToolMessage

# ---------------------------------------------------------------------------
# preds.json extraction (complete_task tool result ``answer_patch``)
# ---------------------------------------------------------------------------

_SWEBENCH_PREDS_UNFINISHED = "unfinished"

_GIT_INDEX_LINE_RE = re.compile(
    r"^index [0-9a-f]+\.\.[0-9a-f]+(?:\s+\S+)?\s*$",
    re.IGNORECASE,
)


def _strip_markdown_fences(text: str) -> str:
    s = text.strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    while lines and lines[-1].strip() in ("```", "```diff"):
        lines.pop()
    return "\n".join(lines).strip()


def _strip_git_index_lines_after_diff_git(lines: List[str]) -> List[str]:
    out: List[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        out.append(line)
        i += 1
        if line.startswith("diff --git "):
            while i < n and _GIT_INDEX_LINE_RE.match(lines[i]):
                i += 1
    return out


def _repo_path_from_minus_header(line: str) -> Optional[str]:
    head = line.split("\t", 1)[0].strip()
    if head == "--- /dev/null":
        return None
    if head.startswith("--- a/"):
        return head[len("--- a/") :]
    if head.startswith("--- "):
        rest = head[4:].strip()
        if rest.startswith("a/"):
            return rest[2:]
    return None


def _repo_path_from_plus_header(line: str) -> Optional[str]:
    head = line.split("\t", 1)[0].strip()
    if head == "+++ /dev/null":
        return None
    if head.startswith("+++ b/"):
        return head[len("+++ b/") :]
    if head.startswith("+++ "):
        rest = head[4:].strip()
        if rest.startswith("b/"):
            return rest[2:]
    return None


def _prepend_diff_git_headers(lines: List[str]) -> List[str]:
    out: List[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        is_file_minus = line.startswith("--- a/") or line.startswith("--- /dev/null")
        if is_file_minus and i + 1 < n and lines[i + 1].startswith("+++ "):
            minus = line
            plus = lines[i + 1]
            p_old = _repo_path_from_minus_header(minus)
            p_new = _repo_path_from_plus_header(plus)
            if p_old is None and p_new is None:
                out.append(minus)
                out.append(plus)
                i += 2
                while i < n:
                    nxt = lines[i]
                    if nxt.startswith("--- a/") or nxt.startswith("--- /dev/null"):
                        break
                    out.append(nxt)
                    i += 1
                continue
            if p_old is None:
                path_a = p_new
                path_b = p_new
            elif p_new is None:
                path_a = p_old
                path_b = p_old
            else:
                path_a, path_b = p_old, p_new
            out.append(f"diff --git a/{path_a} b/{path_b}")
            out.append(minus)
            out.append(plus)
            i += 2
            while i < n:
                nxt = lines[i]
                if nxt.startswith("--- a/") or nxt.startswith("--- /dev/null"):
                    break
                out.append(nxt)
                i += 1
            continue
        out.append(line)
        i += 1
    return out


def normalize_swebench_pred_patch(patch: str) -> str:
    """Normalize **only** the ``complete_task`` ``answer_patch`` for SWE-bench ``preds.json``.

    Ensures each file section starts with ``diff --git a/<path> b/<path>`` (gold style) and
    strips ``index`` lines after ``diff --git``. Does **not** apply to ``apply_patch`` tool input.
    """
    if not isinstance(patch, str):
        return _SWEBENCH_PREDS_UNFINISHED
    stripped = patch.strip()
    if not stripped or stripped == _SWEBENCH_PREDS_UNFINISHED:
        return _SWEBENCH_PREDS_UNFINISHED

    body = _strip_markdown_fences(stripped)
    if not body:
        return _SWEBENCH_PREDS_UNFINISHED

    lines = body.splitlines()
    if any(ln.startswith("diff --git ") for ln in lines):
        normalized_lines = _strip_git_index_lines_after_diff_git(lines)
    else:
        normalized_lines = _prepend_diff_git_headers(lines)
        normalized_lines = _strip_git_index_lines_after_diff_git(normalized_lines)

    result = "\n".join(normalized_lines)
    if result and not result.endswith("\n"):
        result += "\n"
    return result


def _parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str) or not arguments.strip():
        return {}
    try:
        payload = json.loads(arguments)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _last_complete_task_tool_call(chat_task: ChatTask) -> tuple[Optional[str], dict[str, Any]]:
    """Return the id and arguments for the last ``complete_task`` in chat order."""
    last_id: Optional[str] = None
    last_arguments: dict[str, Any] = {}
    for message in chat_task.messages:
        if not isinstance(message, AssistantMessage):
            continue
        if message.tool_calls:
            for tc in message.tool_calls:
                name = tc.function.name
                if isinstance(name, str) and name.strip() == "complete_task":
                    last_id = tc.id
                    last_arguments = _parse_tool_arguments(tc.function.arguments)
    return last_id, last_arguments


def extract_swebench_complete_task_for_preds(chat_task: ChatTask) -> dict[str, str]:
    """Return ``complete_task`` summary and normalized patch for ``preds.json``."""
    tcid, arguments = _last_complete_task_tool_call(chat_task)
    summary = arguments.get("summary")
    patch = arguments.get("answer_patch")
    if tcid and isinstance(patch, str) and patch.strip():
        return {
            "summary": summary if isinstance(summary, str) else "",
            "model_patch": normalize_swebench_pred_patch(patch),
        }

    return {
        "summary": summary if isinstance(summary, str) else "",
        "model_patch": _SWEBENCH_PREDS_UNFINISHED,
    }


def extract_swebench_model_patch_for_preds(chat_task: ChatTask) -> str:
    """Patch string for SWE-bench ``preds.json`` when the run finished correctly.

    Reads the last ``complete_task`` arguments and returns the ``answer_patch``
    field (raw diff only). If there is no such call, no matching tool message,
    or ``answer_patch`` is missing or empty, returns
    :data:`_SWEBENCH_PREDS_UNFINISHED`.
    """
    tcid, arguments = _last_complete_task_tool_call(chat_task)
    if not tcid:
        return _SWEBENCH_PREDS_UNFINISHED
    patch = arguments.get("answer_patch")
    if isinstance(patch, str) and patch.strip():
        return normalize_swebench_pred_patch(patch)
    for message in chat_task.messages:
        if not isinstance(message, ToolMessage):
            continue
        if message.tool_call_id != tcid:
            continue
        raw = message.content
        if not isinstance(raw, str) or not raw.strip():
            return _SWEBENCH_PREDS_UNFINISHED
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return _SWEBENCH_PREDS_UNFINISHED
        if not isinstance(payload, dict):
            return _SWEBENCH_PREDS_UNFINISHED
        patch = payload.get("answer_patch")
        if not isinstance(patch, str) or not patch.strip():
            return _SWEBENCH_PREDS_UNFINISHED
        return normalize_swebench_pred_patch(patch)
    return _SWEBENCH_PREDS_UNFINISHED
