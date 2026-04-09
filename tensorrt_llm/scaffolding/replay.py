import asyncio
import random
import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .execution_trace import ExecutionTrace, TraceEvent
from .task import GenerationTask, TaskStatus
from .worker import Worker


class ReplayGenerationStats:
    """Per-assistant-generation token metrics collected during trace replay.

    For each assistant :class:`TraceEvent`, records the trace-file
    ``completion_tokens`` budget and the lengths actually produced by the
    worker during replay (full decode sequence and content after stripping
    leading reasoning tokens).
    """

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: List[Dict[str, int]] = []

    def record_assistant(
        self,
        *,
        trace_completion_tokens: int,
        replay_output_token_len: int,
        replay_content_token_len: int,
        reasoning_tokens: int,
    ) -> None:
        self._entries.append(
            {
                "trace_completion_tokens": trace_completion_tokens,
                "replay_output_token_len": replay_output_token_len,
                "replay_content_token_len": replay_content_token_len,
                "reasoning_tokens": reasoning_tokens,
            }
        )

    @property
    def entries(self) -> List[Dict[str, int]]:
        return list(self._entries)

    def sum_trace_completion_tokens(self) -> int:
        return sum(e["trace_completion_tokens"] for e in self._entries)

    def sum_replay_output_tokens(self) -> int:
        return sum(e["replay_output_token_len"] for e in self._entries)


def _generate_random_token_ids(length: int) -> List[int]:
    """Return *length* random token ids in the range [100, 30000]."""
    if length <= 0:
        return []
    return [random.randint(100, 30000) for _ in range(length)]


class QueueExecutor:
    """Consumes events from a single queue.

    One ``QueueExecutor`` is created per queue by ``QueueManager``.  It runs
    as an ``asyncio.Task`` that dequeues events until a sentinel (``None``)
    is received, at which point it sets ``done_event`` and exits.

    Event processing is a placeholder for future work — dequeued events are
    currently discarded.
    """

    def __init__(
        self,
        queue: asyncio.Queue,
        trace_id: str,
        worker: Worker,
        system_token_cache: Dict[int, List[int]],
        generation_stats: Optional[ReplayGenerationStats] = None,
    ):
        self.queue = queue
        self.trace_id = trace_id
        self.worker = worker
        self._system_token_cache = system_token_cache
        self._generation_stats = generation_stats
        self._conversation_token_ids: Dict[int, List[List[int]]] = defaultdict(list)
        self.done_event = asyncio.Event()
        self._task = asyncio.create_task(self._run())

    async def _run(self):
        # Always set done_event so :meth:`QueueManager.wait_all_done` cannot
        # deadlock if ``_handle_message`` / ``worker.run_task`` raises (otherwise
        # the sentinel path that sets the event is never reached).
        try:
            while True:
                event = await self.queue.get()
                if event is None:  # sentinel
                    return
                if event.event_type == "tool_call":
                    await self._handle_tool_call(event)
                elif event.event_type == "message":
                    await self._handle_message(event)
                # drop_kv_cache and others: no-op
        finally:
            self.done_event.set()

    async def _handle_tool_call(self, event: TraceEvent):
        duration = event.duration_ms or 0.0
        if duration > 0:
            await asyncio.sleep(duration / 1000)

    def _store_segment(self, conv_id: int, message_index, token_ids):
        """Store a token segment, overwriting if message_index is within bounds."""
        if message_index is not None and message_index < len(self._conversation_token_ids[conv_id]):
            self._conversation_token_ids[conv_id][message_index] = token_ids
        else:
            self._conversation_token_ids[conv_id].append(token_ids)

    async def _handle_message(self, event: TraceEvent):
        conv_id = event.conversation_id
        role = event.role
        message_index = event.message_index

        if role == "system":
            if conv_id in self._system_token_cache:
                token_ids = self._system_token_cache[conv_id]
            else:
                token_ids = _generate_random_token_ids(event.tokens or 0)
                self._system_token_cache[conv_id] = token_ids
            self._store_segment(conv_id, message_index, token_ids)

        elif role in ("user", "tool"):
            token_ids = _generate_random_token_ids(event.tokens or 0)
            self._store_segment(conv_id, message_index, token_ids)

        elif role == "assistant":
            # Build input from all accumulated segments
            input_tokens = []
            for segment in self._conversation_token_ids[conv_id]:
                input_tokens.extend(segment)

            completion_tokens = event.completion_tokens or 0
            reasoning_tokens = event.reasoning_tokens or 0
            if completion_tokens <= 0:
                raise ValueError(
                    "assistant message needs completion_tokens > 0 "
                    f"(got {event.completion_tokens!r}); cannot run generation"
                )

            gen_task = GenerationTask(
                input_tokens=input_tokens,
                max_tokens=completion_tokens,
                ignore_eos=True,
            )
            status = await self.worker.run_task(gen_task)
            if status != TaskStatus.SUCCESS:
                raise RuntimeError(f"GenerationTask failed with status {status}")

            # Strip leading reasoning tokens, keep only the content portion.
            # If the worker did not return token_ids (e.g. OpenAI completions
            # API), fall back to synthetic tokens of the expected length.
            output_tokens = gen_task.output_tokens
            if output_tokens is None:
                output_tokens = _generate_random_token_ids(completion_tokens)
            content_tokens = output_tokens[reasoning_tokens:]
            if self._generation_stats is not None:
                self._generation_stats.record_assistant(
                    trace_completion_tokens=int(completion_tokens),
                    replay_output_token_len=len(output_tokens),
                    replay_content_token_len=len(content_tokens),
                    reasoning_tokens=int(reasoning_tokens),
                )
            self._store_segment(conv_id, message_index, content_tokens)


class QueueManager:
    """Manages a pool of ``(asyncio.Queue, QueueExecutor)`` pairs.

    Tracks which ``trace_id`` each queue belongs to and maintains a mapping
    from ``branch_path`` (as a tuple) to ``queue_id`` for event routing.
    """

    def __init__(
        self,
        worker: Worker,
        system_token_cache: Dict[int, List[int]],
        generation_stats: Optional[ReplayGenerationStats] = None,
    ):
        self._worker = worker
        self._system_token_cache = system_token_cache
        self._generation_stats = generation_stats
        self._queues: Dict[str, asyncio.Queue] = {}
        self._executors: Dict[str, QueueExecutor] = {}
        self._trace_ids: Dict[str, str] = {}
        self._branch_to_queue: Dict[Tuple[int, ...], str] = {}

    def allocate_queue(self, trace_id: str) -> str:
        """Create a new queue + executor pair and return its queue_id."""
        queue_id = str(uuid.uuid4())
        queue: asyncio.Queue = asyncio.Queue()
        executor = QueueExecutor(
            queue,
            trace_id,
            self._worker,
            self._system_token_cache,
            self._generation_stats,
        )
        self._queues[queue_id] = queue
        self._executors[queue_id] = executor
        self._trace_ids[queue_id] = trace_id
        return queue_id

    def register_branch(self, queue_id: str, branch_path: Tuple[int, ...]):
        """Map a branch_path to an existing queue."""
        self._branch_to_queue[branch_path] = queue_id

    def get_queue(self, branch_path: Tuple[int, ...]) -> asyncio.Queue:
        """Look up a queue by branch_path."""
        queue_id = self._branch_to_queue[branch_path]
        return self._queues[queue_id]

    def close_queue(self, queue_id: str):
        """Send sentinel ``None`` to the queue to signal completion."""
        self._queues[queue_id].put_nowait(None)

    async def wait_all_done(self, queue_ids: List[str]):
        """Await ``done_event`` for each executor in *queue_ids*."""
        await asyncio.gather(*(self._executors[qid].done_event.wait() for qid in queue_ids))
        for qid in queue_ids:
            task = self._executors[qid]._task
            if not task.done():
                continue
            if task.cancelled():
                continue
            exc = task.exception()
            if exc is not None:
                raise exc

    def unregister_queue(self, queue_id: str):
        """Clean up all mappings for *queue_id*."""
        self._queues.pop(queue_id, None)
        self._executors.pop(queue_id, None)
        self._trace_ids.pop(queue_id, None)
        # Remove branch_path → queue_id entries
        to_remove = [bp for bp, qid in self._branch_to_queue.items() if qid == queue_id]
        for bp in to_remove:
            del self._branch_to_queue[bp]


class ReplayEngine:
    """Replays an ``ExecutionTrace`` by routing events to per-branch queues.

    The engine owns a ``QueueManager`` and iterates over trace events,
    creating child queues for ``parallel_start`` events and routing all
    other events to the queue registered for their ``branch_path``.
    """

    def __init__(
        self,
        worker: Worker,
        generation_stats: Optional[ReplayGenerationStats] = None,
    ):
        self._system_token_cache: Dict[int, List[int]] = {}
        self.queue_manager = QueueManager(
            worker=worker,
            system_token_cache=self._system_token_cache,
            generation_stats=generation_stats,
        )

    async def launch_trace(self, trace: ExecutionTrace):
        """Iterate *trace.events* and dispatch each event to its queue.

        Flow:
        1. Allocate a root queue for ``branch_path=()``.
        2. Walk ``trace.events`` in order:
           - ``parallel_start``: create child queues, push onto stack.
           - ``parallel_end``: close children, await completion, clean up.
           - All others: route to queue matching ``event.branch_path``.
        3. Close and await the root queue.
        """
        trace_id = trace.trace_id

        # Root queue for branch_path = ()
        root_queue_id = self.queue_manager.allocate_queue(trace_id)
        self.queue_manager.register_branch(root_queue_id, ())

        parallel_stack: List[List[str]] = []

        for event in trace.events:
            if event.event_type == "parallel_start":
                num_branches = event.num_branches or 0
                parent_path = tuple(event.branch_path or ())
                child_queue_ids = []
                for i in range(num_branches):
                    child_qid = self.queue_manager.allocate_queue(trace_id)
                    child_path = parent_path + (i,)
                    self.queue_manager.register_branch(child_qid, child_path)
                    child_queue_ids.append(child_qid)
                parallel_stack.append(child_queue_ids)

            elif event.event_type == "parallel_end":
                child_queue_ids = parallel_stack.pop()
                for qid in child_queue_ids:
                    self.queue_manager.close_queue(qid)
                await self.queue_manager.wait_all_done(child_queue_ids)
                for qid in child_queue_ids:
                    self.queue_manager.unregister_queue(qid)

            else:
                branch_path = tuple(event.branch_path or ())
                queue = self.queue_manager.get_queue(branch_path)
                await queue.put(event)

        # Close and await root queue
        self.queue_manager.close_queue(root_queue_id)
        await self.queue_manager.wait_all_done([root_queue_id])
        self.queue_manager.unregister_queue(root_queue_id)
