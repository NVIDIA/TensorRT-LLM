from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from examples.scaffolding.mcp.fetch_webpage import VisitController, VisitTask
from examples.scaffolding.mcp.tavily_search import TavilyController, TavilyTask
from tensorrt_llm.logger import logger
from tensorrt_llm.scaffolding.controller import (
    Controller,
    NativeGenerationController,
    ParallelProcess,
)
from tensorrt_llm.scaffolding.scaffolding_llm import ScaffoldingLlm
from tensorrt_llm.scaffolding.task import (
    AssistantMessage,
    ChatTask,
    MCPCallTask,
    OpenAIToolDescription,
    SystemMessage,
    Task,
    UserMessage,
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

from .prompts import (
    EVALUATION_INPUT_PROMPT,
    EVALUATION_SYSTEM_PROMPT,
    EXPANSION_INPUT_PROMPT,
    EXPANSION_SYSTEM_PROMPT,
    FINAL_INPUT_PROMPT,
    FINAL_SYSTEM_PROMPT,
)
from .tools import TOT_RESEARCH_TOOLS


@dataclass
class _TOTBranch:
    thought: str
    state: str
    parent: "_TOTBranch | None" = None
    branch_path: tuple[int, ...] = field(default_factory=tuple)
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    tool_call_id: str = ""
    observation: str = ""
    score: float = 0.0
    evaluation: str = ""
    is_complete: bool = False
    final_answer: str = ""

    def path(self) -> list["_TOTBranch"]:
        current: _TOTBranch | None = self
        nodes: list[_TOTBranch] = []
        while current is not None:
            nodes.append(current)
            current = current.parent
        return list(reversed(nodes))


def _parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str) or not arguments.strip():
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        logger.warning("TOTResearch: failed to parse tool arguments: %s", arguments[:200])
        return {}
    if not isinstance(parsed, dict):
        logger.warning("TOTResearch: tool arguments must be a JSON object")
        return {}
    return parsed


def _json_dumps(value: Any) -> str:
    return json.dumps(value or {}, ensure_ascii=False)


def _truncate_text(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length]


@sub_request_node("tot_research", is_top_level=True)
class TOTResearchController(Controller):
    """Tree-of-thought research controller with native tool calls."""

    class WorkerTag(Enum):
        TOOL_CALL = "tot_research_tool_call"

    def __init__(
        self,
        generation_controller: Controller,
        visit_controller: VisitController,
        tavily_controller: TavilyController,
        tools: list[OpenAIToolDescription] | None = None,
        max_depth: int = 5,
        num_thoughts_per_step: int = 3,
        branch_factor: int = 2,
        max_observation_length: int = 128000,
        complete_score_threshold: float = 8.0,
    ):
        super().__init__()
        self.generation_controller = generation_controller
        self.visit_controller = visit_controller
        self.tavily_controller = tavily_controller
        self.tools = tools or TOT_RESEARCH_TOOLS
        self.max_depth = max_depth
        self.num_thoughts_per_step = num_thoughts_per_step
        self.branch_factor = branch_factor
        self.max_observation_length = max_observation_length
        self.complete_score_threshold = complete_score_threshold

    def clone(self) -> "TOTResearchController":
        return TOTResearchController(
            generation_controller=self.generation_controller.clone(),
            visit_controller=self.visit_controller.clone(),
            tavily_controller=self.tavily_controller.clone(),
            tools=self.tools,
            max_depth=self.max_depth,
            num_thoughts_per_step=self.num_thoughts_per_step,
            branch_factor=self.branch_factor,
            max_observation_length=self.max_observation_length,
            complete_score_threshold=self.complete_score_threshold,
        )

    def process(self, tasks: list[Task], **kwargs):
        assert len(tasks) == 1, "TOTResearchController only supports one task"
        question = getattr(tasks[0], "input_str", None) or str(tasks[0])
        root = _TOTBranch(
            thought="Start research.",
            state=f"Question: {question}",
        )
        frontier = [root]
        completed_branches: list[_TOTBranch] = []

        for depth in range(self.max_depth):
            expansion_tasks = self._make_expansion_tasks(question, frontier, depth)
            if not expansion_tasks:
                break

            yield ParallelProcess(
                [self.generation_controller.clone() for _ in expansion_tasks],
                [[task] for task, _, _ in expansion_tasks],
                [{} for _ in expansion_tasks],
                branch_paths=[branch_path for _, _, branch_path in expansion_tasks],
            )

            candidates: list[_TOTBranch] = []
            for chat_task, parent, branch_path in expansion_tasks:
                candidate = self._branch_from_chat_task(chat_task, parent, branch_path)
                if candidate is None:
                    continue
                yield ParallelProcess.from_generators(
                    [self._run_tool(candidate, question)],
                    branch_paths=[candidate.branch_path],
                )
                candidate.observation = _truncate_text(
                    candidate.observation,
                    self.max_observation_length,
                )
                candidate.state = self._render_branch_state(parent.state, candidate)
                candidates.append(candidate)

            if not candidates:
                break

            eval_tasks = [
                ChatTask.create_from_messages(self._evaluation_messages(question, branch))
                for branch in candidates
            ]
            yield ParallelProcess(
                [self.generation_controller.clone() for _ in eval_tasks],
                [[task] for task in eval_tasks],
                [{} for _ in eval_tasks],
                branch_paths=[branch.branch_path for branch in candidates],
            )
            for branch, eval_task in zip(candidates, eval_tasks):
                branch.evaluation = eval_task.last_assistant_content()
                branch.score = self._parse_score(branch.evaluation)

            ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
            completed_branches.extend(branch for branch in ranked if branch.is_complete)
            completed_frontier = [branch for branch in ranked if branch.is_complete]
            if completed_frontier and completed_frontier[0].score >= self.complete_score_threshold:
                frontier = [completed_frontier[0]]
                break

            frontier = [branch for branch in ranked if not branch.is_complete][
                : max(1, self.branch_factor)
            ]
            if not frontier:
                frontier = [ranked[0]]
                break

        final_candidates = frontier + completed_branches
        best_branch = max(final_candidates, key=lambda item: (item.score, len(item.path())))
        final_task = ChatTask.create_from_messages(self._final_messages(question, best_branch))
        yield ParallelProcess.from_generators(
            [self.generation_controller.process([final_task])],
            branch_paths=[best_branch.branch_path],
        )

        tasks[0].output_str = final_task.last_assistant_content()
        tasks[0].output_tokens = final_task.output_tokens or []

    def _make_expansion_tasks(
        self,
        question: str,
        frontier: list[_TOTBranch],
        depth: int,
    ) -> list[tuple[ChatTask, _TOTBranch, tuple[int, ...]]]:
        expansion_tasks: list[tuple[ChatTask, _TOTBranch, tuple[int, ...]]] = []
        for branch in frontier:
            for branch_index in range(max(1, self.num_thoughts_per_step)):
                task = ChatTask.create_from_messages(
                    self._expansion_messages(question, branch, depth),
                    tools=self.tools,
                )
                expansion_tasks.append((task, branch, branch.branch_path + (branch_index,)))
        return expansion_tasks

    def _branch_from_chat_task(
        self,
        chat_task: ChatTask,
        parent: _TOTBranch,
        branch_path: tuple[int, ...],
    ) -> _TOTBranch | None:
        if not chat_task.messages:
            return None
        message = chat_task.messages[-1]
        if not isinstance(message, AssistantMessage):
            logger.warning("TOTResearch: expected AssistantMessage, got %s", type(message).__name__)
            return None

        content = (message.content or "").strip()
        tool_calls = message.tool_calls or []
        if not tool_calls:
            logger.warning("TOTResearch: model returned no tool call; using reflection fallback.")
            if not content:
                return None
            return _TOTBranch(
                thought=content,
                state="",
                parent=parent,
                branch_path=branch_path,
                tool_name="reflection",
                tool_args={"reflection": content},
                tool_call_id=self._branch_tool_call_id(branch_path),
            )

        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = _parse_tool_arguments(tool_call.function.arguments)
        return _TOTBranch(
            thought=content or self._thought_from_tool(tool_name, tool_args),
            state="",
            parent=parent,
            branch_path=branch_path,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_call_id=tool_call.id or self._branch_tool_call_id(branch_path),
        )

    def _run_tool(self, branch: _TOTBranch, question: str):
        tool_name = branch.tool_name
        tool_args = branch.tool_args
        if tool_name == "reflection":
            branch.observation = str(
                tool_args.get("reflection") or tool_args.get("think") or branch.thought
            )
            return

        if tool_name == "complete_task":
            branch.is_complete = True
            branch.final_answer = str(tool_args.get("answer", "")).strip()
            justification = str(tool_args.get("justification", "")).strip()
            branch.observation = "\n".join(
                [
                    f"Completed with answer: {branch.final_answer}",
                    f"Justification: {justification}",
                ]
            )
            return

        if tool_name == "fetch_webpage":
            urls = tool_args.get("url", [])
            if isinstance(urls, str):
                urls = [urls]
            visit_task = VisitTask(
                urls=urls,
                goal=tool_args.get("goal", question),
                parse_type=tool_args.get("parse_type", "html"),
            )
            yield from self.visit_controller.process([visit_task])
            branch.observation = visit_task.result_str or ""
            return

        if tool_name == "tavily_search":
            queries = tool_args.get("query", [])
            if isinstance(queries, str):
                queries = [queries]
            tavily_task = TavilyTask(query=queries or [], goal=question)
            yield from self.tavily_controller.process([tavily_task])
            branch.observation = tavily_task.result_str or ""
            if tavily_task.result_stdout:
                branch.observation += f"\nstdout:\n{tavily_task.result_stdout}"
            if tavily_task.result_stderr:
                branch.observation += f"\nstderr:\n{tavily_task.result_stderr}"
            return

        mcp_task = MCPCallTask.create_mcptask(
            tool_call_id=branch.tool_call_id or self._branch_tool_call_id(branch.branch_path),
            tool_name=tool_name or "",
            args=_json_dumps(tool_args),
            worker_tag=self.WorkerTag.TOOL_CALL,
        )
        yield [mcp_task]
        branch.observation = mcp_task.result_str or ""
        if mcp_task.result_stdout:
            branch.observation += f"\nstdout:\n{mcp_task.result_stdout}"
        if mcp_task.result_stderr:
            branch.observation += f"\nstderr:\n{mcp_task.result_stderr}"

    def _expansion_messages(self, question: str, branch: _TOTBranch, depth: int):
        messages = [
            SystemMessage(EXPANSION_SYSTEM_PROMPT),
            UserMessage(
                EXPANSION_INPUT_PROMPT.format(
                    question=question,
                    trajectory=self._render_path(branch),
                    depth=depth + 1,
                    max_depth=self.max_depth,
                )
            ),
        ]
        return messages

    def _evaluation_messages(self, question: str, branch: _TOTBranch):
        return [
            SystemMessage(EVALUATION_SYSTEM_PROMPT),
            UserMessage(
                EVALUATION_INPUT_PROMPT.format(
                    question=question,
                    thought=branch.thought,
                    tool_name=branch.tool_name,
                    tool_args=_json_dumps(branch.tool_args),
                    observation=branch.observation,
                )
            ),
        ]

    def _final_messages(self, question: str, branch: _TOTBranch):
        messages = [
            SystemMessage(FINAL_SYSTEM_PROMPT),
            UserMessage(
                FINAL_INPUT_PROMPT.format(
                    question=question,
                    trajectory=self._render_path(branch),
                )
            ),
        ]
        return messages

    def _render_path(self, branch: _TOTBranch) -> str:
        rows: list[str] = []
        for i, node in enumerate(branch.path()):
            if i == 0:
                rows.append(node.state)
                continue
            node_rows = [
                f"Step {i}: {node.thought}",
                f"Tool: {node.tool_name}({_json_dumps(node.tool_args)})",
            ]
            node_rows.append(f"Observation: {node.observation}")
            node_rows.extend(
                [
                    f"Score: {node.score}",
                    f"Complete: {node.is_complete}",
                ]
            )
            rows.append("\n".join(node_rows))
        return "\n\n".join(rows)

    def _branch_tool_call_id(self, branch_path: tuple[int, ...]) -> str:
        path = "_".join(str(item) for item in branch_path) or "root"
        return f"tot_{path}"

    def _render_branch_state(self, parent_state: str, branch: _TOTBranch) -> str:
        return "\n\n".join(
            [
                parent_state,
                f"Next thought: {branch.thought}",
                f"Tool call: {branch.tool_name}({_json_dumps(branch.tool_args)})",
                f"Observation: {branch.observation}",
                f"Complete: {branch.is_complete}",
            ]
        )

    def _thought_from_tool(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        if tool_name == "tavily_search":
            query = tool_args.get("query", "")
            if isinstance(query, list):
                query = "; ".join(str(item) for item in query)
            return f"Search for evidence about: {query}"
        if tool_name == "fetch_webpage":
            urls = tool_args.get("url", "")
            if isinstance(urls, list):
                urls = ", ".join(str(url) for url in urls)
            return f"Read source content from: {urls}"
        if tool_name == "python_interpreter":
            return "Use Python to compute or verify this branch."
        if tool_name == "complete_task":
            return "Complete the task with the final answer."
        return "Reflect on the current research path."

    def _parse_score(self, text: str) -> float:
        match = re.search(r"score\s*:\s*(\d+(?:\.\d+)?)", text or "", re.IGNORECASE)
        if not match:
            match = re.search(r"(\d+(?:\.\d+)?)", text or "")
        if not match:
            return 0.0
        score = float(match.group(1))
        return max(0.0, min(10.0, score))


def create_tot_research_controller(
    max_tokens: int = 16384,
    max_depth: int = 3,
    num_thoughts_per_step: int = 3,
    branch_factor: int = 2,
    max_observation_length: int = 128000,
    complete_score_threshold: float = 8.0,
    max_webpage_tokens: int = 48000,
    max_tavily_search_chars: int = 6000,
    enable_statistics: bool = False,
    enable_tracing: bool = False,
) -> Controller:
    sampling_params = {
        # "temperature": 0.7,
        # "top_p": 0.95,
        "max_tokens": max_tokens,
    }
    generation_controller = NativeGenerationController(sampling_params=sampling_params)
    visit_controller = VisitController(
        generation_controller=generation_controller,
        max_webpage_tokens=max_webpage_tokens,
    )
    tavily_controller = TavilyController(
        generation_controller=generation_controller,
        compress_threshold_chars=max_tavily_search_chars,
    )

    controller_type = TOTResearchController
    if enable_statistics:

        def wrap_with_profiler(controller_cls, name):
            return with_task_collection(
                f"{name}TaskCollection",
                TaskMetricsCollector,
                controller_name=name,
                task_types=[ChatTask, MCPCallTask],
                enable_print=True,
                capture_messages=True,
            )(controller_cls)

        controller_type = wrap_with_profiler(controller_type, "TOTResearch")

    if enable_tracing:
        controller_type = with_execution_tracing("TOTResearch")(controller_type)
        controller_type = tokenize_trace_scope()(controller_type)

    return controller_type(
        generation_controller=generation_controller,
        visit_controller=visit_controller,
        tavily_controller=tavily_controller,
        max_depth=max_depth,
        num_thoughts_per_step=num_thoughts_per_step,
        branch_factor=branch_factor,
        max_observation_length=max_observation_length,
        complete_score_threshold=complete_score_threshold,
    )


def create_tot_research_scaffolding_llm(
    generation_worker: Worker,
    mcp_worker: Worker,
    max_tokens: int = 16384,
    max_depth: int = 3,
    num_thoughts_per_step: int = 3,
    branch_factor: int = 2,
    max_observation_length: int = 128000,
    complete_score_threshold: float = 8.0,
    max_webpage_tokens: int = 48000,
    max_tavily_search_chars: int = 6000,
    max_parallel_requests: int = 1024,
    enable_statistics: bool = False,
    enable_tracing: bool = False,
) -> ScaffoldingLlm:
    controller = create_tot_research_controller(
        max_tokens=max_tokens,
        max_depth=max_depth,
        num_thoughts_per_step=num_thoughts_per_step,
        branch_factor=branch_factor,
        max_observation_length=max_observation_length,
        complete_score_threshold=complete_score_threshold,
        max_webpage_tokens=max_webpage_tokens,
        max_tavily_search_chars=max_tavily_search_chars,
        enable_statistics=enable_statistics,
        enable_tracing=enable_tracing,
    )
    workers = {
        NativeGenerationController.WorkerTag.GENERATION: generation_worker,
        TOTResearchController.WorkerTag.TOOL_CALL: mcp_worker,
        VisitController.WorkerTag.TOOL_CALL: mcp_worker,
        TavilyController.WorkerTag.TOOL_CALL: mcp_worker,
        DropKVCacheWorkerTag.DROP_KV_CACHE: generation_worker,
    }
    if enable_tracing:
        workers[TokenizeWorkerTag.TOKENIZE] = generation_worker

    scaffolding_llm = ScaffoldingLlm(
        controller,
        workers,
        max_parallel_requests=max_parallel_requests,
    )
    if enable_tracing:
        scaffolding_llm.enable_output_task_collection()
    return scaffolding_llm
