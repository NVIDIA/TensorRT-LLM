import asyncio
import hashlib
import json
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Any

from .controller import Controller
from .execution_trace import ExecutionTrace
from .scaffolding_llm import ScaffoldingLlm
from .task import MCPCallTask, TaskStatus
from .worker import Worker


class ReplayDriver(ABC):
    """Abstract base for providing recorded MCP responses during replay."""

    @abstractmethod
    async def get_mcp_response(self, tool_name: str, args: Any) -> str:
        """Return the recorded MCP tool call result."""

    @abstractmethod
    def get_simulated_delay(self, tool_name: str, args: Any) -> float:
        """Return delay in seconds to simulate MCP execution time."""


def _args_hash(tool_name: str, args: Any) -> str:
    """Produce a stable hash key from tool_name and arguments."""
    raw = json.dumps({"tool_name": tool_name, "args": args}, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


class TraceReplayDriver(ReplayDriver):
    """Serves recorded MCP responses from an ExecutionTrace.

    Looks up responses by (tool_name, args) content hash. Falls back to a
    sequential queue when the content-based lookup misses (e.g. if the same
    tool is called twice with identical args).
    """

    def __init__(
        self, trace: ExecutionTrace, simulate_latency: bool = True, latency_scale: float = 1.0
    ):
        self.simulate_latency = simulate_latency
        self.latency_scale = latency_scale

        self._content_index: dict[str, deque] = defaultdict(deque)
        self._sequential_queue: deque = deque()

        for tool_name, tool_args, result_str, duration_ms in trace.get_mcp_responses():
            entry = {
                "tool_name": tool_name,
                "tool_args": tool_args,
                "result_str": result_str,
                "duration_ms": duration_ms,
            }
            key = _args_hash(tool_name, tool_args)
            self._content_index[key].append(entry)
            self._sequential_queue.append(entry)

    async def get_mcp_response(self, tool_name: str, args: Any) -> str:
        parsed_args = args
        if isinstance(args, str):
            try:
                parsed_args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                parsed_args = args

        key = _args_hash(tool_name, parsed_args)
        if key in self._content_index and self._content_index[key]:
            entry = self._content_index[key].popleft()
            return entry["result_str"] or ""

        if self._sequential_queue:
            entry = self._sequential_queue.popleft()
            return entry["result_str"] or ""

        return ""

    def get_simulated_delay(self, tool_name: str, args: Any) -> float:
        if not self.simulate_latency:
            return 0.0

        parsed_args = args
        if isinstance(args, str):
            try:
                parsed_args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                parsed_args = args

        key = _args_hash(tool_name, parsed_args)
        if key in self._content_index and self._content_index[key]:
            entry = self._content_index[key][0]
            return (entry["duration_ms"] / 1000.0) * self.latency_scale

        if self._sequential_queue:
            entry = self._sequential_queue[0]
            return (entry["duration_ms"] / 1000.0) * self.latency_scale

        return 0.0


class ReplayMCPWorker(Worker):
    """A Worker that handles MCPCallTask using a ReplayDriver instead of a live MCP server."""

    def __init__(self, driver: ReplayDriver):
        self.driver = driver

    async def _handle_mcp_call(self, task: MCPCallTask) -> TaskStatus:
        delay = self.driver.get_simulated_delay(task.tool_name, task.args)
        if delay > 0:
            await asyncio.sleep(delay)
        task.result_str = await self.driver.get_mcp_response(task.tool_name, task.args)
        return TaskStatus.SUCCESS

    task_handlers = {MCPCallTask: _handle_mcp_call}


def create_replay_scaffolding_llm(
    trace: ExecutionTrace,
    generation_worker: Worker,
    prototype_controller: Controller,
    simulate_mcp_latency: bool = True,
    latency_scale: float = 1.0,
    max_parallel_requests: int = 64,
) -> ScaffoldingLlm:
    """Create a ScaffoldingLlm configured for replay benchmarking.

    ChatTasks are served by the real ``generation_worker``.
    MCPTasks are mocked from the loaded ``trace``.

    Args:
        trace: The execution trace to replay.
        generation_worker: Real LLM worker for ChatTask / GenerationTask.
        prototype_controller: The same controller that produced the trace
            (or an equivalent one).
        simulate_mcp_latency: Whether to sleep for recorded MCP durations.
        latency_scale: Multiplier for simulated MCP latency (1.0 = real-time).
        max_parallel_requests: Max parallel requests for the ScaffoldingLlm.

    Returns:
        A ScaffoldingLlm ready for ``generate_async(trace.prompt)``.
    """
    from .controller import ChatWithMCPController
    from .task_collection import DropKVCacheWorkerTag

    driver = TraceReplayDriver(
        trace, simulate_latency=simulate_mcp_latency, latency_scale=latency_scale
    )
    replay_mcp_worker = ReplayMCPWorker(driver)

    workers: dict[str, Worker] = {
        "generation": generation_worker,
        ChatWithMCPController.WorkerTag.TOOLCALL: replay_mcp_worker,
        DropKVCacheWorkerTag.DROP_KV_CACHE: generation_worker,
    }

    # Also map NativeGenerationController.WorkerTag if present
    from .controller import NativeGenerationController

    workers[NativeGenerationController.WorkerTag.GENERATION] = generation_worker

    return ScaffoldingLlm(
        prototype_controller=prototype_controller,
        workers=workers,
        max_parallel_requests=max_parallel_requests,
    )
