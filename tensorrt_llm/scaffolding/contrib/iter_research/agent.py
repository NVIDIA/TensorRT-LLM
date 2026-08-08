import json
from enum import Enum
from typing import Any, Dict, List

from examples.scaffolding.mcp.fetch_webpage import VisitController, VisitTask
from examples.scaffolding.mcp.tavily_search import TavilyController, TavilyTask
from tensorrt_llm.logger import logger
from tensorrt_llm.scaffolding.controller import Controller, NativeGenerationController
from tensorrt_llm.scaffolding.scaffolding_llm import ScaffoldingLlm
from tensorrt_llm.scaffolding.task import (
    AssistantMessage,
    ChatTask,
    MCPCallTask,
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
    INITIAL_INPUT_PROMPT,
    INITIAL_SYSTEM_PROMPT,
    INSTRUCTION_INPUT_PROMPT,
    INSTRUCTION_SYSTEM_PROMPT,
    LAST_INSTRUCTION_INPUT_PROMPT,
    LAST_INSTRUCTION_SYSTEM_PROMPT,
    OBSERVATION_PROMPT,
)
from .utils import (
    TOOLS,
    check_report_action,
    extract_tags,
    get_tool_definitions,
    random_date,
    truncate_text,
)

_TOOL_NAME_TO_MCP = {
    "PythonInterpreter": "python_interpreter",
}


def _parse_tool_arguments(arguments: Any) -> Dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str) or not arguments.strip():
        return {}

    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        logger.warning("IterResearch: failed to parse native tool arguments: %s", arguments[:200])
        return {}

    if not isinstance(parsed, dict):
        logger.warning("IterResearch: native tool arguments must be a JSON object")
        return {}
    return parsed


def _render_tool_call(tool_name: str, tool_args: Dict[str, Any]) -> str:
    return json.dumps(
        {
            "name": tool_name,
            "arguments": tool_args,
        },
        ensure_ascii=False,
    )


@sub_request_node("iter_research", is_top_level=True)
# @drop_kv_cache_scope()
class IterResearchController(Controller):
    """Iterative research with Markovian context (last report, tool call, observation)."""

    class WorkerTag(Enum):
        TOOL_CALL = "iter_research_tool_call"

    def __init__(
        self,
        generation_controller: Controller,
        visit_controller: VisitController,
        tavily_controller: TavilyController,
        max_turn: int = 25,
        max_format_retries: int = 5,
        max_observation_length: int = 128000,
    ):
        super().__init__()
        self.generation_controller = generation_controller
        self.visit_controller = visit_controller
        self.tavily_controller = tavily_controller
        self.max_turn = max_turn
        self.max_format_retries = max_format_retries
        self.max_observation_length = max_observation_length

    def clone(self):
        return IterResearchController(
            generation_controller=self.generation_controller.clone(),
            visit_controller=self.visit_controller.clone(),
            tavily_controller=self.tavily_controller.clone(),
            max_turn=self.max_turn,
            max_format_retries=self.max_format_retries,
            max_observation_length=self.max_observation_length,
        )

    def process(self, tasks: List[Task], **kwargs):  # noqa: C901
        assert len(tasks) >= 1, "IterResearchController requires at least one task"
        question = tasks[0].input_str
        date = random_date()
        tool_str = get_tool_definitions()

        initial_input = INITIAL_INPUT_PROMPT.format(
            question=question,
            tools=tool_str,
            date_to_use=date,
        )

        chat_task = ChatTask.create_from_messages(
            [
                SystemMessage(INITIAL_SYSTEM_PROMPT),
                UserMessage(initial_input),
            ],
            tools=TOOLS,
        )

        content = ""
        last_report = ""
        response_message = None
        for _retry in range(self.max_format_retries):
            yield from self.generation_controller.process([chat_task])
            response_message = chat_task.messages[-1] if chat_task.messages else None
            if not isinstance(response_message, AssistantMessage):
                reason = "Assistant message not found!"
                is_valid = False
            else:
                content = response_message.content or ""
                is_valid, reason = check_report_action(
                    response_message,
                    require_tool_call=True,
                    allow_tool_call_without_report=True,
                )
            if is_valid:
                break
            logger.warning(
                f"IterResearch: initial format check failed ({reason}), "
                f"retrying ({_retry + 1}/{self.max_format_retries})"
            )
            if chat_task.messages and chat_task.messages[-1].role == "assistant":
                chat_task.messages.pop()

        for turn in range(self.max_turn):
            if not isinstance(response_message, AssistantMessage):
                tasks[0].output_str = content
                tasks[0].output_tokens = tasks[0].output_tokens or []
                return

            current_report = extract_tags(content, "report")
            if current_report:
                last_report = current_report
            report = current_report or last_report
            answer = extract_tags(content, "answer")
            tool_calls = response_message.tool_calls or []

            if answer:
                logger.info(f"IterResearch turn {turn + 1}: found answer")
                tasks[0].output_str = answer
                tasks[0].output_tokens = tasks[0].output_tokens or []
                return

            if not tool_calls:
                tasks[0].output_str = report or content
                tasks[0].output_tokens = tasks[0].output_tokens or []
                return

            tool_call = tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = _parse_tool_arguments(tool_call.function.arguments)
            tool_call_id = tool_call.id or str(turn)
            tool_call_str = _render_tool_call(tool_name, tool_args)
            logger.info(f"IterResearch turn {turn + 1}: calling tool '{tool_name}'")

            if tool_name == "Visit":
                urls = tool_args.get("url", [])
                if isinstance(urls, str):
                    urls = [urls]
                visit_task = VisitTask(
                    urls=urls,
                    goal=tool_args.get("goal", question),
                    parse_type=tool_args.get("parse_type", "html"),
                )
                yield from self.visit_controller.process([visit_task])
                observation = visit_task.result_str or ""
            elif tool_name == "tavily_search":
                queries = tool_args.get("query", [])
                if isinstance(queries, str):
                    queries = [queries]
                tavily_task = TavilyTask(query=queries or [], goal=question)
                yield from self.tavily_controller.process([tavily_task])
                observation = tavily_task.result_str or ""
                if tavily_task.result_stdout:
                    observation += f"\nstdout:\n{tavily_task.result_stdout}"
                if tavily_task.result_stderr:
                    observation += f"\nstderr:\n{tavily_task.result_stderr}"
            else:
                mcp_name = _TOOL_NAME_TO_MCP.get(tool_name, tool_name)
                mcp_task = MCPCallTask.create_mcptask(
                    tool_call_id=tool_call_id,
                    tool_name=mcp_name,
                    args=json.dumps(tool_args),
                    worker_tag=self.WorkerTag.TOOL_CALL,
                )
                yield [mcp_task]
                observation = mcp_task.result_str or ""

            observation = truncate_text(observation, self.max_observation_length)

            is_last = turn == self.max_turn - 2
            if is_last:
                system_prompt_template = LAST_INSTRUCTION_SYSTEM_PROMPT
                user_prompt_template = LAST_INSTRUCTION_INPUT_PROMPT
            else:
                system_prompt_template = INSTRUCTION_SYSTEM_PROMPT
                user_prompt_template = INSTRUCTION_INPUT_PROMPT

            prompt_kwargs = {
                "question": question,
                "report": report,
                "action": tool_call_str,
                "observation": OBSERVATION_PROMPT.format(tool_response=observation),
                "tools": tool_str,
                "date_to_use": date,
            }

            system_prompt = system_prompt_template.format(**prompt_kwargs)
            user_prompt = user_prompt_template.format(**prompt_kwargs)

            new_messages = [
                SystemMessage(system_prompt),
                UserMessage(user_prompt),
            ]

            chat_task = ChatTask.create_from_messages(
                new_messages,
                tools=None if is_last else TOOLS,
            )

            content = ""
            response_message = None
            for _retry in range(self.max_format_retries):
                yield from self.generation_controller.process([chat_task])
                response_message = chat_task.messages[-1] if chat_task.messages else None
                if not isinstance(response_message, AssistantMessage):
                    reason = "Assistant message not found!"
                    is_valid = False
                else:
                    content = response_message.content or ""
                    is_valid, reason = check_report_action(
                        response_message,
                        allow_tool_call=not is_last,
                        allow_tool_call_without_report=not is_last,
                    )
                if is_valid:
                    break
                logger.warning(
                    f"IterResearch turn {turn + 1}: format check failed "
                    f"({reason}), retrying "
                    f"({_retry + 1}/{self.max_format_retries})"
                )
                if chat_task.messages and chat_task.messages[-1].role == "assistant":
                    chat_task.messages.pop()

        final_answer = extract_tags(content, "answer")
        if not final_answer:
            final_answer = extract_tags(content, "report")
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
    max_tavily_search_chars: int = 6000,
    enable_statistics: bool = False,
    enable_tracing: bool = False,
) -> Controller:
    sampling_params = {
        # "temperature": 0.6,
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

        controller_type = wrap_with_profiler(controller_type, "IterResearch")

    if enable_tracing:
        controller_type = with_execution_tracing("IterResearch")(controller_type)
        controller_type = tokenize_trace_scope()(controller_type)

    return controller_type(
        generation_controller=generation_controller,
        visit_controller=visit_controller,
        tavily_controller=tavily_controller,
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
    max_tavily_search_chars: int = 6000,
    max_parallel_requests: int = 1024,
    enable_statistics: bool = False,
    enable_tracing: bool = False,
) -> ScaffoldingLlm:
    controller = create_iter_research_controller(
        max_tokens=max_tokens,
        max_turn=max_turn,
        max_format_retries=max_format_retries,
        max_observation_length=max_observation_length,
        max_webpage_tokens=max_webpage_tokens,
        max_tavily_search_chars=max_tavily_search_chars,
        enable_statistics=enable_statistics,
        enable_tracing=enable_tracing,
    )

    workers = {
        NativeGenerationController.WorkerTag.GENERATION: generation_worker,
        IterResearchController.WorkerTag.TOOL_CALL: mcp_worker,
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
