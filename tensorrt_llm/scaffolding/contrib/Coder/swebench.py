# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""SWE-bench variant of the Coder agent.

Provides a system prompt, controller, and factory function tailored for
SWE-bench evaluation tasks where the agent must fix a bug described in a
GitHub pull-request / issue within a pre-configured ``/testbed`` sandbox.
"""

# ruff: noqa: E501

import json
import re
from typing import Any, List, Optional

from tensorrt_llm.scaffolding.controller import (
    ChatWithMCPController,
    Controller,
    NativeGenerationController,
)
from tensorrt_llm.scaffolding.scaffolding_llm import ScaffoldingLlm
from tensorrt_llm.scaffolding.task import (
    AssistantMessage,
    ChatTask,
    MCPCallTask,
    SystemMessage,
    Task,
    ToolMessage,
)
from tensorrt_llm.scaffolding.task_collection import (
    DropKVCacheWorkerTag,
    TaskMetricsCollector,
    TokenizeWorkerTag,
    sub_request_node,
    tokenize_trace_scope,
    with_execution_tracing,
    with_task_collection,
)
from tensorrt_llm.scaffolding.worker import Worker

from .prompts import SWEBENCH_SYSTEM_PROMPT
from .tools import ALL_CODER_TOOLS

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


# ---------------------------------------------------------------------------
# SWE-bench controller
# ---------------------------------------------------------------------------


@sub_request_node("agent_swebench_coder", is_top_level=True)
# @drop_kv_cache_scope()
class SWEBenchCoder(Controller):
    """SWE-bench variant of the Coder controller.

    Uses :data:`SWEBENCH_SYSTEM_PROMPT` instead of the generic Coder prompt.
    """

    tools = ALL_CODER_TOOLS

    def __init__(self, chat_with_tools_controller: Controller):
        super().__init__()
        self.chat_with_tools_controller = chat_with_tools_controller

    def clone(self):
        cloned_ctrl = self.chat_with_tools_controller.clone()
        return SWEBenchCoder(chat_with_tools_controller=cloned_ctrl)

    def process(self, tasks: List[Task], **kwargs):
        task = tasks[0]
        user_prompt = task.input_str

        chat_task = ChatTask.create_from_prompt(
            user_prompt,
            [SystemMessage(content=SWEBENCH_SYSTEM_PROMPT)],
            tools=self.tools,
        )

        yield from self.chat_with_tools_controller.process([chat_task])

        task.output_str = chat_task.last_assistant_content()
        complete_task_result = extract_swebench_complete_task_for_preds(chat_task)
        task.customized_result_fields["swebench_model_patch"] = complete_task_result["model_patch"]
        task.customized_result_fields["swebench_summary"] = complete_task_result["summary"]
        return


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_swebench_coder_scaffolding_llm(
    generation_worker: Worker,
    mcp_worker: Worker,
    max_tokens: int = 131072,
    max_iterations: int = 100,
    max_parallel_requests: int = 16,
    enable_statistics: bool = False,
    enable_tracing: bool = False,
) -> ScaffoldingLlm:
    """Create a :class:`ScaffoldingLlm` configured for SWE-bench evaluation.

    Mirrors :func:`create_coder_scaffolding_llm` but uses
    :class:`SWEBenchCoder` with the SWE-bench-specific system prompt.
    """
    sampling_params = {
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }

    generation_controller = NativeGenerationController(sampling_params=sampling_params)

    chat_with_mcp_controller_type = ChatWithMCPController
    coder_type = SWEBenchCoder

    if enable_statistics:

        def wrap_with_detailed_profiler(controller_type, controller_name):
            return with_task_collection(
                f"{controller_name}TaskCollection",
                TaskMetricsCollector,
                controller_name=controller_name,
                task_types=[ChatTask, MCPCallTask],
                enable_print=True,
                capture_messages=True,
            )(controller_type)

        chat_with_mcp_controller_type = wrap_with_detailed_profiler(
            ChatWithMCPController, "ChatWithMCP"
        )
        coder_type = wrap_with_detailed_profiler(SWEBenchCoder, "SWEBenchCoder")

    if enable_tracing:
        coder_type = with_execution_tracing("SWEBenchCoder")(coder_type)
        coder_type = tokenize_trace_scope()(coder_type)

    chat_with_tools_controller = chat_with_mcp_controller_type(
        generation_controller, max_iterations=max_iterations
    )

    coder_controller = coder_type(
        chat_with_tools_controller=chat_with_tools_controller,
    )

    workers = {
        NativeGenerationController.WorkerTag.GENERATION: generation_worker,
        ChatWithMCPController.WorkerTag.TOOLCALL: mcp_worker,
        DropKVCacheWorkerTag.DROP_KV_CACHE: generation_worker,
    }
    if enable_tracing:
        workers[TokenizeWorkerTag.TOKENIZE] = generation_worker

    scaffolding_llm = ScaffoldingLlm(
        coder_controller,
        workers,
        max_parallel_requests=max_parallel_requests,
    )

    if enable_tracing:
        scaffolding_llm.enable_output_task_collection()

    return scaffolding_llm
