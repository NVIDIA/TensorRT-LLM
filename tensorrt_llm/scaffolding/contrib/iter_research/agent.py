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

from tensorrt_llm.logger import logger
from tensorrt_llm.scaffolding.controller import (
    Controller,
    NativeGenerationController,
)
from tensorrt_llm.scaffolding.scaffolding_llm import ScaffoldingLlm
from tensorrt_llm.scaffolding.task import (
    ChatTask,
    MCPCallTask,
    SystemMessage,
    Task,
    UserMessage,
)
from tensorrt_llm.scaffolding.task_collection import (
    DropKVCacheWorkerTag,
    drop_kv_cache_scope,
    sub_request_node,
    TaskMetricsCollector,
    with_execution_tracing,
    with_task_collection,
)
from tensorrt_llm.scaffolding.worker import Worker

from .prompts import (
    INITIAL_INPUT_PROMPT,
    INITIAL_SYSTEM_PROMPT,
    INSTRUCTION_PROMPT,
    LAST_INSTRUCTION_PROMPT,
    OBSERVATION_PROMPT,
    VISIT_EXTRACTOR_PROMPT,
)
from .utils import (
    check_report_action,
    extract_tags,
    get_tool_definitions,
    random_date,
    truncate_text,
)

# Mapping from prompt-facing tool names (what the LLM outputs) to MCP server
# function names.  Only entries where the names differ need to be listed;
# tools whose names are identical (google_search, google_scholar) fall
# through to the identity mapping via dict.get(name, name).
_TOOL_NAME_TO_MCP = {
    "PythonInterpreter": "python_interpreter",
}


def _parse_json_output(raw: str) -> Optional[dict]:
    """Parse JSON output from LLM response, handling markdown code blocks."""
    if not raw:
        return None
    triple_match = re.search(r'```json\s*\n(.*?)\n```', raw, re.DOTALL)
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
    """Custom task for the Visit subController.

    Carries the URLs and goal for webpage fetching + LLM summarization.
    """
    urls: Optional[list] = field(default=None)
    goal: Optional[str] = field(default=None)
    parse_type: str = field(default="html")
    result_str: Optional[str] = field(default=None)


class VisitController(Controller):
    """Fetch webpage content via MCP, then summarize with LLM.

    This is a subController that:
    1. Dispatches a ``fetch_webpage`` MCP call to retrieve raw page content.
    2. Truncates the content if it exceeds *max_webpage_tokens*.
    3. Sends a summarization prompt to the LLM generation worker.
    4. Parses the structured JSON response (rational / evidence / summary).
    """

    class WorkerTag(Enum):
        TOOL_CALL = "visit_tool_call"

    def __init__(self, generation_controller: Controller,
                 max_webpage_tokens: int = 48000):
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
            "VisitController only supports a single VisitTask")
        visit_task = tasks[0]

        # Step 1: Fetch webpage content via MCP
        mcp_task = MCPCallTask.create_mcptask(
            tool_call_id="visit_fetch",
            tool_name="fetch_webpage",
            args=json.dumps({
                "url": visit_task.urls,
                "parse_type": visit_task.parse_type,
            }),
            worker_tag=self.WorkerTag.TOOL_CALL,
        )
        yield [mcp_task]

        raw_content = mcp_task.result_str or ""
        if not raw_content:
            visit_task.result_str = (
                "[visit] Failed to fetch webpage content.")
            return

        # Step 2: Truncate content if too long (rough char-based fallback)
        if len(raw_content) > self.max_webpage_tokens * 4:
            raw_content = raw_content[:self.max_webpage_tokens * 4]

        # Step 3: Summarize with LLM
        extractor_prompt = VISIT_EXTRACTOR_PROMPT.format(
            webpage_content=raw_content,
            goal=visit_task.goal,
        )
        chat_task = ChatTask.create_from_messages([
            UserMessage(extractor_prompt),
        ])
        yield from self.generation_controller.process([chat_task])

        llm_response = chat_task.messages[-1].content if chat_task.messages else ""

        # Step 4: Parse JSON response
        parsed = _parse_json_output(llm_response)
        urls_str = ", ".join(visit_task.urls) if visit_task.urls else "unknown"
        useful_info = (
            f"The useful information in '{urls_str}' "
            f"for user goal '{visit_task.goal}':\n\n")

        if parsed and 'evidence' in parsed and 'summary' in parsed:
            useful_info += f"Evidence in page:\n{parsed['evidence']}\n\n"
            useful_info += f"Summary:\n{parsed['summary']}\n"
        else:
            useful_info += llm_response

        visit_task.result_str = useful_info


@sub_request_node("iter_research", is_top_level=True)
@drop_kv_cache_scope()
class IterResearchController(Controller):
    """Core iterative research controller implementing Markovian state
    reconstruction.

    Each turn reconstructs a fresh prompt from only:
    - The last report
    - The last tool call
    - The last observation

    This keeps context size constant regardless of how many turns have elapsed,
    enabling 2048+ tool calls within a ~40k context window.
    """

    class WorkerTag(Enum):
        TOOL_CALL = "iter_research_tool_call"

    def __init__(
        self,
        generation_controller: Controller,
        visit_controller: VisitController,
        max_turn: int = 25,
        max_format_retries: int = 5,
        max_observation_length: int = 128000,
    ):
        super().__init__()
        self.generation_controller = generation_controller
        self.visit_controller = visit_controller
        self.max_turn = max_turn
        self.max_format_retries = max_format_retries
        self.max_observation_length = max_observation_length

    def clone(self):
        return IterResearchController(
            generation_controller=self.generation_controller.clone(),
            visit_controller=self.visit_controller.clone(),
            max_turn=self.max_turn,
            max_format_retries=self.max_format_retries,
            max_observation_length=self.max_observation_length,
        )

    def _generate_llm(self, prompt_text: str):
        """Helper: create a ChatTask from a single user message and yield it
        through the generation controller.

        Returns the ChatTask so the caller can inspect the response.
        """
        chat_task = ChatTask.create_from_messages([
            UserMessage(prompt_text),
        ])
        yield from self.generation_controller.process([chat_task])
        return chat_task

    def process(self, tasks: List[Task], **kwargs):  # noqa: C901
        assert len(tasks) >= 1, "IterResearchController requires at least one task"
        question = tasks[0].input_str
        date = random_date()
        tool_str = get_tool_definitions()

        # ---- Turn 1: Initial prompt (system + user) ----
        initial_input = INITIAL_INPUT_PROMPT.format(
            question=question,
            tools=tool_str,
            date_to_use=date,
        )

        chat_task = ChatTask.create_from_messages([
            SystemMessage(INITIAL_SYSTEM_PROMPT),
            UserMessage(initial_input),
        ])

        # Retry loop for format validation on the first turn
        content = ""
        for _retry in range(self.max_format_retries):
            yield from self.generation_controller.process([chat_task])
            content = (chat_task.messages[-1].content
                       if chat_task.messages else "")
            is_valid, reason = check_report_action(content)
            if is_valid:
                break
            logger.warning(
                f"IterResearch: initial format check failed ({reason}), "
                f"retrying ({_retry + 1}/{self.max_format_retries})")
            # Remove the invalid assistant message and retry
            if chat_task.messages and chat_task.messages[-1].role == "assistant":
                chat_task.messages.pop()

        # ---- Iterative turns ----
        for turn in range(self.max_turn):
            # Parse XML tags from LLM response
            report = extract_tags(content, 'report')
            tool_call_str = extract_tags(content, 'tool_call')
            answer = extract_tags(content, 'answer')

            # If answer found, we're done
            if answer:
                tasks[0].output_str = answer
                tasks[0].output_tokens = tasks[0].output_tokens or []
                return

            # Try to parse the tool call
            tool_call = None
            if tool_call_str:
                try:
                    tool_call = json.loads(tool_call_str)
                except json.JSONDecodeError:
                    logger.warning(
                        f"IterResearch turn {turn}: failed to parse tool_call "
                        f"JSON: {tool_call_str[:200]}")

            # If no tool call, agent has finished (perhaps with just a report)
            if not tool_call:
                tasks[0].output_str = report or content
                tasks[0].output_tokens = tasks[0].output_tokens or []
                return

            # Execute tool
            tool_name = tool_call.get('name', 'unknown')
            tool_args = tool_call.get('arguments', {})
            logger.info(f"IterResearch turn {turn + 1}: calling tool "
                        f"'{tool_name}'")

            if tool_name == 'Visit':
                # Use VisitController subController (fetch + LLM summarize)
                urls = tool_args.get('url', [])
                if isinstance(urls, str):
                    urls = [urls]
                visit_task = VisitTask(
                    urls=urls,
                    goal=tool_args.get('goal', question),
                    parse_type=tool_args.get('parse_type', 'html'),
                )
                yield from self.visit_controller.process([visit_task])
                observation = visit_task.result_str or ""
            else:
                # All other tools -> MCP.
                # Map prompt-facing names to MCP server function names
                # where they differ (e.g. PythonInterpreter ->
                # python_interpreter).
                mcp_name = _TOOL_NAME_TO_MCP.get(tool_name, tool_name)
                mcp_task = MCPCallTask.create_mcptask(
                    tool_call_id=str(turn),
                    tool_name=mcp_name,
                    args=json.dumps(tool_args),
                    worker_tag=self.WorkerTag.TOOL_CALL,
                )
                yield [mcp_task]
                observation = mcp_task.result_str or ""

            # Truncate observation if too long
            observation = truncate_text(
                observation, self.max_observation_length)

            # ---- Markovian context reconstruction ----
            is_last = (turn == self.max_turn - 2)
            prompt_template = (LAST_INSTRUCTION_PROMPT
                               if is_last else INSTRUCTION_PROMPT)
            formatted_observation = OBSERVATION_PROMPT.format(
                tool_response=observation)

            new_prompt = prompt_template.format(
                question=question,
                report=report,
                action=tool_call_str,
                observation=formatted_observation,
                tools=tool_str,
                date_to_use=date,
            )

            # Create fresh chat task (Markovian: no history carried over)
            chat_task = ChatTask.create_from_messages([
                UserMessage(new_prompt),
            ])

            # Retry loop for format validation
            content = ""
            for _retry in range(self.max_format_retries):
                yield from self.generation_controller.process([chat_task])
                content = (chat_task.messages[-1].content
                           if chat_task.messages else "")
                is_valid, reason = check_report_action(content)
                if is_valid:
                    break
                logger.warning(
                    f"IterResearch turn {turn + 1}: format check failed "
                    f"({reason}), retrying "
                    f"({_retry + 1}/{self.max_format_retries})")
                if (chat_task.messages
                        and chat_task.messages[-1].role == "assistant"):
                    chat_task.messages.pop()

        # Loop exhausted — extract whatever answer/report we have
        final_answer = extract_tags(content, 'answer')
        if not final_answer:
            final_answer = extract_tags(content, 'report')
        if not final_answer:
            final_answer = content

        tasks[0].output_str = final_answer
        tasks[0].output_tokens = tasks[0].output_tokens or []


def create_iter_research_controller(
    max_tokens: int = 16384,
    max_turn: int = 25,
    max_format_retries: int = 5,
    max_observation_length: int = 128000,
    max_webpage_tokens: int = 48000,
    enable_statistics: bool = False,
    enable_tracing: bool = False,
) -> Controller:
    """Create the controller graph for IterResearch.

    Args:
        max_tokens: Maximum generation tokens per LLM call.
        max_turn: Maximum number of iterative research turns.
        max_format_retries: Maximum retries when LLM output fails format
            validation.
        max_observation_length: Maximum character length for tool observations
            before truncation.
        max_webpage_tokens: Maximum tokens for webpage content before
            truncation in the Visit subController.
        enable_statistics: Enable task metrics collection.
        enable_tracing: Enable execution tracing.

    Returns:
        The top-level IterResearchController.
    """
    sampling_params = {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": max_tokens,
    }

    generation_controller = NativeGenerationController(
        sampling_params=sampling_params)
    visit_controller = VisitController(
        generation_controller=generation_controller,
        max_webpage_tokens=max_webpage_tokens,
    )

    controller_type = IterResearchController

    if enable_statistics:

        def wrap_with_profiler(cls, name):
            return with_task_collection(
                f"{name}TaskCollection",
                TaskMetricsCollector,
                controller_name=name,
                task_types=[ChatTask, MCPCallTask],
                enable_print=True,
                capture_messages=True,
            )(cls)

        controller_type = wrap_with_profiler(controller_type,
                                             "IterResearch")

    if enable_tracing:
        controller_type = with_execution_tracing("IterResearch")(
            controller_type)

    return controller_type(
        generation_controller=generation_controller,
        visit_controller=visit_controller,
        max_turn=max_turn,
        max_format_retries=max_format_retries,
        max_observation_length=max_observation_length,
    )


def create_iter_research_scaffolding_llm(
    generation_worker: Worker,
    mcp_worker: Worker,
    max_tokens: int = 16384,
    max_turn: int = 25,
    max_format_retries: int = 5,
    max_observation_length: int = 128000,
    max_webpage_tokens: int = 48000,
    max_parallel_requests: int = 1024,
    enable_statistics: bool = False,
    enable_tracing: bool = False,
) -> ScaffoldingLlm:
    """Create a ScaffoldingLlm for IterResearch.

    Args:
        generation_worker: Worker for LLM generation (e.g. TRTOpenaiWorker).
        mcp_worker: Worker for MCP tool calls (e.g. MCPWorker).
        max_tokens: Maximum generation tokens per LLM call.
        max_turn: Maximum number of iterative research turns.
        max_format_retries: Maximum retries for format validation.
        max_observation_length: Maximum character length for tool observations.
        max_webpage_tokens: Maximum tokens for webpage content in Visit.
        max_parallel_requests: Maximum parallel requests for ScaffoldingLlm.
        enable_statistics: Enable task metrics collection.
        enable_tracing: Enable execution tracing.

    Returns:
        Configured ScaffoldingLlm instance.
    """
    controller = create_iter_research_controller(
        max_tokens=max_tokens,
        max_turn=max_turn,
        max_format_retries=max_format_retries,
        max_observation_length=max_observation_length,
        max_webpage_tokens=max_webpage_tokens,
        enable_statistics=enable_statistics,
        enable_tracing=enable_tracing,
    )

    scaffolding_llm = ScaffoldingLlm(
        controller,
        {
            NativeGenerationController.WorkerTag.GENERATION:
            generation_worker,
            IterResearchController.WorkerTag.TOOL_CALL:
            mcp_worker,
            VisitController.WorkerTag.TOOL_CALL:
            mcp_worker,
            DropKVCacheWorkerTag.DROP_KV_CACHE:
            generation_worker,
        },
        max_parallel_requests=max_parallel_requests,
    )

    if enable_tracing:
        scaffolding_llm.enable_output_task_collection()

    return scaffolding_llm
