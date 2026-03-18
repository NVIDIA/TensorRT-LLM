"""Coder agent supervisor and factory function for the Scaffolding framework.

This module implements a coding agent that can read, write, and modify code
using MCP-based tool execution for file system operations and shell commands.
"""

from dataclasses import dataclass, field
from typing import List

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
)
from tensorrt_llm.scaffolding.task_collection import (
    DropKVCacheWorkerTag,
    TaskMetricsCollector,
    drop_kv_cache_scope,
    sub_request_node,
    with_execution_tracing,
    with_task_collection,
)
from tensorrt_llm.scaffolding.worker import Worker

from .prompts import CODER_SYSTEM_PROMPT
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


@sub_request_node("agent_coder", is_top_level=True)
@drop_kv_cache_scope()
class Coder(Controller):
    """Coding agent controller that handles tool calling and code modifications.

    The Coder agent uses ChatWithMCPController to execute tools via MCP:
    1. Receives a user prompt
    2. Creates a ChatTask with system prompt and tools
    3. Delegates to ChatWithMCPController for the tool-calling loop
    4. Returns the final response
    """

    tools = ALL_CODER_TOOLS

    def __init__(
        self,
        chat_with_tools_controller: Controller,
    ):
        """Initialize the Coder controller.

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
        return Coder(chat_with_tools_controller=cloned_ctrl)

    def process(self, tasks: List[Task], **kwargs):
        """Process a list of tasks through the coding agent workflow.

        Args:
            tasks: List of tasks to process (expects CoderTask or Task with input_str).
            **kwargs: Additional arguments passed to sub-controllers.

        Yields:
            Tasks for the chat_with_tools_controller.
        """
        task = tasks[0]

        # Get user prompt from task
        if isinstance(task, CoderTask):
            user_prompt = task.user_prompt
        else:
            user_prompt = task.input_str

        # Create the chat task with system prompt and tools
        chat_task = ChatTask.create_from_prompt(
            user_prompt,
            [SystemMessage(content=CODER_SYSTEM_PROMPT)],
            tools=self.tools,
        )

        # Delegate to chat_with_tools_controller for the tool-calling loop
        yield from self.chat_with_tools_controller.process([chat_task])

        # Set the output
        final_message = chat_task.messages[-1]
        if isinstance(final_message, AssistantMessage):
            task.output_str = final_message.content
        else:
            task.output_str = str(final_message.content) if final_message.content else ""

        if isinstance(task, CoderTask):
            task.final_response = task.output_str

        return


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
    # Sampling parameters
    sampling_params = {
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }

    # Create generation controller
    generation_controller = NativeGenerationController(sampling_params=sampling_params)

    # Create ChatWithMCPController for tool calling
    chat_with_mcp_controller_type = ChatWithMCPController
    coder_type = Coder

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
        coder_type = wrap_with_detailed_profiler(Coder, "Coder")

    if enable_tracing:
        coder_type = with_execution_tracing("Coder")(coder_type)

    # Create the ChatWithMCPController
    chat_with_tools_controller = chat_with_mcp_controller_type(
        generation_controller, max_iterations=max_iterations
    )

    # Create the Coder controller
    coder_controller = coder_type(
        chat_with_tools_controller=chat_with_tools_controller,
    )

    # Create and return the ScaffoldingLlm
    scaffolding_llm = ScaffoldingLlm(
        coder_controller,
        {
            NativeGenerationController.WorkerTag.GENERATION: generation_worker,
            ChatWithMCPController.WorkerTag.TOOLCALL: mcp_worker,
            DropKVCacheWorkerTag.DROP_KV_CACHE: generation_worker,
        },
        max_parallel_requests=max_parallel_requests,
    )

    if enable_tracing:
        scaffolding_llm.enable_output_task_collection()

    return scaffolding_llm
