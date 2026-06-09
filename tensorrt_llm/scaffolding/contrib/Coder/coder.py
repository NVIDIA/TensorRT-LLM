"""Coder agent supervisor and factory functions for the Scaffolding framework.

This module implements a coding agent that can read, write, and modify code
using MCP-based tool execution for file system operations and shell commands.
It also hosts the SWE-bench variant (:class:`SWEBenchCoder`), which shares all
orchestration with :class:`Coder` and only differs in its system prompt and how
the final result is post-processed.
"""

from dataclasses import dataclass, field
from typing import List

from tensorrt_llm.scaffolding.controller import (
    ChatWithMCPController,
    Controller,
    NativeGenerationController,
)
from tensorrt_llm.scaffolding.scaffolding_llm import ScaffoldingLlm
from tensorrt_llm.scaffolding.task import ChatTask, MCPCallTask, SystemMessage, Task
from tensorrt_llm.scaffolding.task_collection import (
    DropKVCacheWorkerTag,
    TaskMetricsCollector,
    TokenizeWorkerTag,
    drop_kv_cache_scope,
    sub_request_node,
    tokenize_trace_scope,
    with_execution_tracing,
    with_task_collection,
)
from tensorrt_llm.scaffolding.worker import Worker

from .prompts import CODER_SYSTEM_PROMPT, SWEBENCH_SYSTEM_PROMPT
from .swebench_utils import extract_swebench_complete_task_for_preds
from .tools import ALL_CODER_TOOLS


@dataclass
class CoderTask(Task):
    """Task for the Coder agent."""

    user_prompt: str = field(default=None)
    final_response: str = field(default=None)

    @staticmethod
    def create_from_prompt(prompt: str) -> "CoderTask":
        """Create a CoderTask from a user prompt."""
        task = CoderTask()
        task.user_prompt = prompt
        task.input_str = prompt
        return task


class _BaseCoderController(Controller):
    """Shared logic for the Coder controllers.

    Subclasses set :attr:`SYSTEM_PROMPT` and may override :meth:`_user_prompt`
    (how the user text is read off the task) and :meth:`_finalize` (how the
    final result is written back). The tool-calling loop itself is identical:
    build a :class:`ChatTask` with the system prompt and tools, delegate to the
    ``chat_with_tools_controller``, then record the last assistant message.
    """

    tools = ALL_CODER_TOOLS
    SYSTEM_PROMPT: str = None

    def __init__(self, chat_with_tools_controller: Controller):
        """Initialize the controller.

        Args:
            chat_with_tools_controller: Controller for chat with tool calling
                                        (typically ChatWithMCPController).
        """
        super().__init__()
        self.chat_with_tools_controller = chat_with_tools_controller

    def clone(self):
        """Create a copy of this controller for parallel execution.

        Each clone runs in its own :class:`ExecutionScope` (assigned
        automatically by :class:`ScaffoldingLlm` when entering a
        ``ParallelProcess`` branch), so ``ApiaryMCPWorker`` routes its
        tool calls to a dedicated SSE connection without manual wiring.
        """
        cloned_ctrl = self.chat_with_tools_controller.clone()
        return type(self)(chat_with_tools_controller=cloned_ctrl)

    def process(self, tasks: List[Task], **kwargs):
        """Run the tool-calling loop for a single task."""
        task = tasks[0]

        chat_task = ChatTask.create_from_prompt(
            self._user_prompt(task),
            [SystemMessage(content=self.SYSTEM_PROMPT)],
            tools=self.tools,
        )

        # Delegate to chat_with_tools_controller for the tool-calling loop.
        yield from self.chat_with_tools_controller.process([chat_task])

        task.output_str = chat_task.last_assistant_content()
        self._finalize(task, chat_task)
        return

    def _user_prompt(self, task: Task) -> str:
        """Read the user prompt from the task (override as needed)."""
        return task.input_str

    def _finalize(self, task: Task, chat_task: ChatTask) -> None:
        """Hook to post-process the finished chat (override as needed)."""


@sub_request_node("agent_coder", is_top_level=True)
@drop_kv_cache_scope()
class Coder(_BaseCoderController):
    """Coding agent controller that handles tool calling and code modifications.

    Receives a user prompt, runs the tool-calling loop via
    :class:`ChatWithMCPController`, and returns the final assistant response.
    """

    SYSTEM_PROMPT = CODER_SYSTEM_PROMPT

    def _user_prompt(self, task: Task) -> str:
        if isinstance(task, CoderTask):
            return task.user_prompt
        return task.input_str

    def _finalize(self, task: Task, chat_task: ChatTask) -> None:
        if isinstance(task, CoderTask):
            task.final_response = task.output_str


@sub_request_node("agent_swebench_coder", is_top_level=True)
# @drop_kv_cache_scope()
class SWEBenchCoder(_BaseCoderController):
    """SWE-bench variant of the Coder controller.

    Uses :data:`SWEBENCH_SYSTEM_PROMPT` instead of the generic Coder prompt and
    extracts the ``preds.json`` patch/summary from the final ``complete_task``
    tool call.
    """

    SYSTEM_PROMPT = SWEBENCH_SYSTEM_PROMPT

    def _finalize(self, task: Task, chat_task: ChatTask) -> None:
        complete_task_result = extract_swebench_complete_task_for_preds(chat_task)
        task.customized_result_fields["swebench_model_patch"] = complete_task_result["model_patch"]
        task.customized_result_fields["swebench_summary"] = complete_task_result["summary"]


def _wrap_with_detailed_profiler(controller_type, controller_name):
    """Attach a :class:`TaskMetricsCollector` to a controller class."""
    return with_task_collection(
        f"{controller_name}TaskCollection",
        TaskMetricsCollector,
        controller_name=controller_name,
        task_types=[ChatTask, MCPCallTask],
        enable_print=True,
        capture_messages=True,
    )(controller_type)


def _build_coder_scaffolding_llm(
    *,
    coder_type: type,
    coder_name: str,
    generation_worker: Worker,
    mcp_worker: Worker,
    max_tokens: int,
    max_iterations: int,
    max_parallel_requests: int,
    enable_statistics: bool,
    enable_tracing: bool,
) -> ScaffoldingLlm:
    """Assemble a :class:`ScaffoldingLlm` for a Coder-family controller.

    Shared by :func:`create_coder_scaffolding_llm` and
    :func:`create_swebench_coder_scaffolding_llm`; ``coder_type`` and
    ``coder_name`` are the only things that differ between them.
    """
    sampling_params = {
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }

    generation_controller = NativeGenerationController(sampling_params=sampling_params)

    chat_with_mcp_controller_type = ChatWithMCPController
    if enable_statistics:
        chat_with_mcp_controller_type = _wrap_with_detailed_profiler(
            ChatWithMCPController, "ChatWithMCP"
        )
        coder_type = _wrap_with_detailed_profiler(coder_type, coder_name)

    if enable_tracing:
        coder_type = with_execution_tracing(coder_name)(coder_type)
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


def create_coder_scaffolding_llm(
    generation_worker: Worker,
    mcp_worker: Worker,
    max_tokens: int = 131072,
    max_iterations: int = 50,
    max_parallel_requests: int = 16,
    enable_statistics: bool = False,
    enable_tracing: bool = False,
) -> ScaffoldingLlm:
    """Create a ScaffoldingLlm configured for the Coder agent.

    Args:
        generation_worker: Worker for LLM generation (e.g., TRTOpenaiWorker).
        mcp_worker: Worker for MCP tool calls (e.g., MCPWorker connected to CoderMCP server).
        max_tokens: Maximum tokens for generation.
        max_iterations: Maximum tool-calling iterations.
        max_parallel_requests: Maximum parallel requests to process.
        enable_statistics: Enable detailed profiling/metrics.

    Returns:
        Configured ScaffoldingLlm instance.

    Example:
        ```python
        from tensorrt_llm.scaffolding.worker import TRTOpenaiWorker, ApiaryMCPWorker
        from tensorrt_llm.scaffolding.contrib.Coder import create_coder_scaffolding_llm

        # Create workers
        generation_worker = TRTOpenaiWorker(
            base_url="http://localhost:8000/v1",
            api_key="your-api-key",
            model="your-model",
        )

        # Start CoderMCP server first:
        # python examples/scaffolding/mcp/coder/coder_mcp.py --port 8083

        mcp_worker = ApiaryMCPWorker("http://localhost:8083/sse")

        # Create the Coder agent
        coder = create_coder_scaffolding_llm(
            generation_worker=generation_worker,
            mcp_worker=mcp_worker,
        )

        # Run a coding task
        result = coder.generate("Add a hello world function to main.py")
        print(result.text)
        ```
    """
    return _build_coder_scaffolding_llm(
        coder_type=Coder,
        coder_name="Coder",
        generation_worker=generation_worker,
        mcp_worker=mcp_worker,
        max_tokens=max_tokens,
        max_iterations=max_iterations,
        max_parallel_requests=max_parallel_requests,
        enable_statistics=enable_statistics,
        enable_tracing=enable_tracing,
    )


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
    return _build_coder_scaffolding_llm(
        coder_type=SWEBenchCoder,
        coder_name="SWEBenchCoder",
        generation_worker=generation_worker,
        mcp_worker=mcp_worker,
        max_tokens=max_tokens,
        max_iterations=max_iterations,
        max_parallel_requests=max_parallel_requests,
        enable_statistics=enable_statistics,
        enable_tracing=enable_tracing,
    )
