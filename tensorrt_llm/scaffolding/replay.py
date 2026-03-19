# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import asyncio
import copy
import uuid
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer

from .controller import Controller, NativeGenerationController
from .execution_scope import current_scope
from .execution_trace import ExecutionTrace, TraceEvent
from .scaffolding_llm import ScaffoldingLlm
from .task import (
    AssistantMessage,
    ChatTask,
    DropKVCacheTask,
    GenerationTask,
    MCPCallTask,
    RoleMessage,
    Task,
    TaskStatus,
)
from .worker import Worker


class ReplayMismatchError(Exception):
    """The replay controller diverged from the recorded trace.

    Raised when the controller yields a task list that doesn't match the
    next expected event for its branch path.  Common causes:
      - Corrupted or hand-edited trace file
      - Controller code changed between recording and replay
      - Trace recorded with a different controller topology
    """


class TraceReplayEngine:
    """Drives generation replay from a recorded ExecutionTrace.

    For each recorded event the engine constructs a synthetic request whose
    input and output token counts match the trace, sends it to the real
    ``generation_worker`` with ``ignore_eos=True``, and then restores the
    minimum control-plane state (finish_reason, tool_calls) needed to keep
    the controller on the recorded trajectory.

    Event types and how they are consumed:

    * ``message`` with ``role="assistant"`` — dispatched to the real
      generation_worker with a synthetic payload sized to match recorded
      token counts.  The only consumable message events.
    * ``message`` with other roles — context messages (system, user, tool)
      recorded between yields.  **Not** consumed.
    * ``tool_call`` — result injected directly from the trace.
    * ``drop_kv_cache`` — no-op.
    * ``parallel_start`` with ``children`` (pseudo-fork from a multi-task
      yield) — each child event is matched to the corresponding task.
    * ``parallel_start`` without ``children`` (ParallelProcess) —
      structural marker, **not** indexed or consumed.

    Events are pre-indexed by branch path and consumed FIFO within each
    branch.  Concurrent branches have disjoint paths by construction, so
    no synchronization is needed.
    """

    def __init__(
        self,
        trace: ExecutionTrace,
        latency_scale: float = 1.0,
    ):
        self.latency_scale = latency_scale
        self._tokenizer = None
        self._filler_token_id = None

        # Index replayable events into per-branch FIFO queues keyed by
        # branch_path.  One event is consumed per controller yield, so only
        # events that correspond to yield points belong here.  Non-assistant
        # messages (system/user/tool) are recorded in the trace for
        # observability but don't map to yields — the controller populates
        # them on ChatTask.messages between yields, and the replay engine
        # sees them as part of task.messages when building the synthetic
        # prompt.  Childless parallel_start events are structural markers
        # for ParallelProcess branching, not consumable yield points.
        self._event_queues: Dict[tuple, deque] = defaultdict(deque)
        for event in trace.events:
            if event.event_type == "message" and event.role != "assistant":
                continue
            if event.event_type == "parallel_start" and event.children is None:
                continue
            key = tuple(event.branch_path)
            self._event_queues[key].append(event)

    async def replay_task_list(
        self,
        tasks: List[Task],
        generation_worker: Worker,
    ) -> None:
        """Replay one yielded task list against the next trace event.

        For a single-task yield the next event is a flat ``message``
        (role=assistant), ``tool_call``, or ``drop_kv_cache`` event.
        For a multi-task yield it is a ``parallel_start`` event whose
        ``children`` list contains one child per concurrently-dispatched
        task.
        """
        if not tasks:
            return

        scope = current_scope.get()
        path = scope.branch_path_list if scope is not None else []
        event = self._consume_next_event(path)

        if event.event_type == "parallel_start" and event.children is not None:
            child_events = event.children
        else:
            child_events = [event]

        self._validate_event_match(tasks, child_events, event)

        coros: list = []
        for task, child_event in zip(tasks, child_events):
            if isinstance(task, ChatTask):
                coros.append(self._replay_chat_task(task, child_event, generation_worker))
            elif isinstance(task, GenerationTask):
                coros.append(self._replay_generation_task(task, child_event, generation_worker))
            elif isinstance(task, MCPCallTask):
                coros.append(self._replay_mcp_task(task, child_event))
            elif isinstance(task, DropKVCacheTask):
                pass
            else:
                raise ReplayMismatchError(
                    f"Unsupported task type during replay: {type(task).__name__}"
                )

        if coros:
            await asyncio.gather(*coros)

    def _consume_next_event(self, branch_path: List[int]) -> TraceEvent:
        """Pop and return the next pending event for *branch_path*.

        Events within a branch are consumed strictly in trace-recorded order
        (FIFO).  Concurrent branches always have disjoint path tuples, so
        no locking is needed.
        """
        key = tuple(branch_path)
        queue = self._event_queues.get(key)
        if not queue:
            active = sorted(list(k) for k in self._event_queues if self._event_queues[k])
            raise ReplayMismatchError(
                f"No remaining trace events for branch {list(key)}. "
                f"Branches with remaining events: {active}"
            )
        return queue.popleft()

    def _validate_event_match(
        self,
        tasks: List[Task],
        child_events: List[TraceEvent],
        parent_event: TraceEvent,
    ) -> None:
        """Assert that the yielded task list structurally matches the event."""
        if len(tasks) != len(child_events):
            raise ReplayMismatchError(
                f"Task count mismatch on branch "
                f"{parent_event.branch_path}: "
                f"controller yielded {len(tasks)} tasks, "
                f"trace expected {len(child_events)}"
            )

        for i, (task, child_event) in enumerate(zip(tasks, child_events)):
            expected = _task_to_event_type(task)
            if child_event.event_type != expected:
                raise ReplayMismatchError(
                    f"Event type mismatch at index {i} on branch "
                    f"{parent_event.branch_path}: "
                    f"controller yielded {type(task).__name__} "
                    f"(expects '{expected}'), "
                    f"trace has '{child_event.event_type}'"
                )

    async def _replay_chat_task(
        self,
        task: ChatTask,
        event: TraceEvent,
        generation_worker: Worker,
    ) -> None:
        tokenizer = self._get_tokenizer(generation_worker)
        target_completion = max(0, event.completion_tokens or 0)

        synthetic = copy.deepcopy(task)
        synthetic.messages = self._build_synthetic_chat_messages(
            task.messages, event.prompt_tokens or 0, tokenizer
        )
        synthetic.streaming_output_flag = False
        synthetic.streaming_output_list = []
        synthetic.max_tokens = target_completion
        synthetic.ignore_eos = True

        status = await generation_worker.run_task(synthetic)
        if status != TaskStatus.SUCCESS:
            raise RuntimeError(f"Generation worker failed during ChatTask replay: {status}")

        task.prompt_tokens_num = event.prompt_tokens or 0
        task.completion_tokens_num = event.completion_tokens or 0
        task.reasoning_tokens_num = event.reasoning_tokens or 0
        task.finish_reason = event.finish_reason
        replay_message = self._build_replay_assistant_message(event, tokenizer)
        task.messages.append(replay_message)
        task.output_str = replay_message.content
        task.output_tokens = _placeholder_token_ids(event.completion_tokens)

    async def _replay_generation_task(
        self,
        task: GenerationTask,
        event: TraceEvent,
        generation_worker: Worker,
    ) -> None:
        tokenizer = self._get_tokenizer(generation_worker)
        target_output = max(0, event.completion_tokens or 0)
        target_input = max(0, event.prompt_tokens or 0)

        synthetic = copy.deepcopy(task)
        synthetic.input_str = self._build_text_with_token_count(target_input, tokenizer)
        synthetic.output_str = None
        synthetic.output_tokens = None
        synthetic.finish_reason = None
        synthetic.streaming_output_flag = False
        synthetic.streaming_output_list = []
        synthetic.max_tokens = target_output
        synthetic.ignore_eos = True

        status = await generation_worker.run_task(synthetic)
        if status != TaskStatus.SUCCESS:
            raise RuntimeError(f"Generation worker failed during GenerationTask replay: {status}")

        task.output_str = self._build_text_with_token_count(target_output, tokenizer)
        task.output_tokens = _placeholder_token_ids(event.completion_tokens)

    async def _replay_mcp_task(
        self,
        task: MCPCallTask,
        event: TraceEvent,
    ) -> None:
        """Simulate the original MCP call latency.

        The trace no longer stores MCP results, so replay can only
        reproduce timing.  The task receives an empty result string.
        """
        target_ms = (event.duration_ms or 0.0) * self.latency_scale
        if target_ms > 0:
            await asyncio.sleep(target_ms / 1000)
        task.result_str = ""

    def _build_replay_assistant_message(
        self,
        event: TraceEvent,
        tokenizer,
    ) -> AssistantMessage:
        """Reconstruct the AssistantMessage from the trace event.

        If the recorded step was a tool-call, restores the exact tool_calls
        structure (required for the controller to branch correctly).
        Otherwise fills content with length-matched filler text.
        """
        tool_calls = _extract_tool_calls(event)
        content = ""
        if not tool_calls:
            content = self._build_text_with_token_count(event.completion_tokens or 0, tokenizer)
        return AssistantMessage(content=content, tool_calls=tool_calls)

    def _build_synthetic_chat_messages(
        self,
        messages: List[RoleMessage],
        target_prompt_tokens: int,
        tokenizer,
    ) -> List[RoleMessage]:
        """Build a message list whose prompt token count matches the trace.

        Computes each message's content token count, then distributes filler
        text proportionally so that relative sizes (e.g. system vs. user) are
        preserved.  This lets the replay produce a realistic KV-cache layout
        where the system-prompt length matches the original.
        """
        synthetic = copy.deepcopy(messages)
        if not synthetic:
            return synthetic

        msg_token_counts = []
        for msg in synthetic:
            content = getattr(msg, "content", None) or ""
            tokens = len(tokenizer.encode(content, add_special_tokens=False))
            msg_token_counts.append(tokens)

        for msg in synthetic:
            msg.content = ""

        overhead = self._count_chat_prompt_tokens(synthetic, tokenizer)
        filler_total = max(0, target_prompt_tokens - overhead)

        if filler_total <= 0:
            return synthetic

        total_content_tokens = sum(msg_token_counts)
        if total_content_tokens > 0:
            allocated = 0
            for i, msg in enumerate(synthetic):
                if i == len(synthetic) - 1:
                    msg_filler = filler_total - allocated
                else:
                    msg_filler = round(msg_token_counts[i] / total_content_tokens * filler_total)
                    allocated += msg_filler
                if msg_filler > 0:
                    msg.content = self._build_text_with_token_count(msg_filler, tokenizer)
        else:
            synthetic[-1].content = self._build_text_with_token_count(filler_total, tokenizer)

        return synthetic

    def _count_chat_prompt_tokens(
        self,
        messages: List[RoleMessage],
        tokenizer,
    ) -> int:
        """Count prompt tokens by applying the tokenizer's chat template."""
        payload = []
        for msg in messages:
            entry: Dict[str, Any] = {
                "role": getattr(msg, "role", None),
                "content": getattr(msg, "content", None),
            }
            if hasattr(msg, "tool_call_id"):
                entry["tool_call_id"] = msg.tool_call_id
            payload.append(entry)

        token_ids = tokenizer.apply_chat_template(
            payload, tokenize=True, add_generation_prompt=False
        )
        return len(token_ids)

    def _get_tokenizer(self, generation_worker: Worker):
        """Lazily obtain a tokenizer from the worker or its model name."""
        if self._tokenizer is not None:
            return self._tokenizer

        tokenizer = getattr(generation_worker, "tokenizer", None)
        if tokenizer is not None:
            self._tokenizer = tokenizer
            return tokenizer

        model_name = getattr(generation_worker, "model", None)
        if model_name is None:
            raise ValueError(
                "Replay requires a worker with a tokenizer attribute or "
                "a model name for AutoTokenizer.from_pretrained."
            )

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            legacy=False,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=False,
            use_fast=True,
        )
        return self._tokenizer

    def _build_text_with_token_count(
        self,
        token_count: int,
        tokenizer,
    ) -> str:
        """Generate a string that tokenizes to approximately *token_count*.

        Uses a stable single-token filler piece repeated N times.  Falls back
        to incremental concatenation when the fast path doesn't round-trip to
        the exact count.
        """
        if token_count <= 0:
            return ""

        filler_id = self._get_filler_token_id(tokenizer)

        text = tokenizer.decode(
            [filler_id] * token_count, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if len(tokenizer.encode(text, add_special_tokens=False)) == token_count:
            return text

        piece = tokenizer.decode(
            [filler_id], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        text = ""
        while len(tokenizer.encode(text, add_special_tokens=False)) < token_count:
            text += piece
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) > token_count:
            return tokenizer.decode(
                encoded[:token_count], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        return text

    def _get_filler_token_id(self, tokenizer) -> int:
        """Find a single token ID that round-trips through encode/decode."""
        if self._filler_token_id is not None:
            return self._filler_token_id

        for candidate in [" a", "a", " x", "x", ".", " the", " test"]:
            ids = tokenizer.encode(candidate, add_special_tokens=False)
            if len(ids) != 1:
                continue
            decoded = tokenizer.decode(
                ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            if tokenizer.encode(decoded, add_special_tokens=False) == ids:
                self._filler_token_id = ids[0]
                return ids[0]

        for token_id in range(10, 1000):
            decoded = tokenizer.decode(
                [token_id], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            if not decoded:
                continue
            if tokenizer.encode(decoded, add_special_tokens=False) == [token_id]:
                self._filler_token_id = token_id
                return token_id

        raise ValueError("Unable to find a stable filler token for replay.")


class _ReplayToolFunction:
    """Stand-in for openai ChatCompletionMessageToolCall.Function."""

    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _ReplayToolCall:
    """Stand-in for openai ChatCompletionMessageToolCall."""

    def __init__(self, tool_call_id: str, name: str, arguments: str):
        self.id = tool_call_id
        self.type = "function"
        self.function = _ReplayToolFunction(name, arguments)


def _extract_tool_calls(event: TraceEvent) -> Optional[List[_ReplayToolCall]]:
    """Reconstruct tool-call objects from ``event.tool_calls``.

    Returns ``None`` (never ``[]``) when there are no tool calls.  This
    is critical: controllers branch on ``tool_calls is not None``, so an
    empty list would incorrectly enter the tool-call code path and yield
    an empty task list with no matching trace event.

    The trace stores tool_calls as a list of tool name strings.
    A dummy id and empty arguments are synthesised for the replay objects.
    """
    if not event.tool_calls:
        return None
    return [
        _ReplayToolCall(
            tool_call_id=str(uuid.uuid4()),
            name=tc,
            arguments="{}",
        )
        for tc in event.tool_calls
    ]


def _task_to_event_type(task: Task) -> str:
    """Map a Task subclass to the expected trace event_type."""
    if isinstance(task, (ChatTask, GenerationTask)):
        return "message"
    if isinstance(task, MCPCallTask):
        return "tool_call"
    if isinstance(task, DropKVCacheTask):
        return "drop_kv_cache"
    raise ReplayMismatchError(f"Unsupported task type: {type(task).__name__}")


def _placeholder_token_ids(count: Optional[int]) -> List[int]:
    """Return a list of *count* zeroes as a dummy token-ID sequence."""
    if count is None or count <= 0:
        return []
    return [0] * count


class TraceReplayScaffoldingLlm(ScaffoldingLlm):
    def __init__(
        self,
        engine: TraceReplayEngine,
        generation_worker: Worker,
        prototype_controller: Controller,
        max_parallel_requests: int = 64,
    ):
        workers: Dict[str, Worker] = {
            NativeGenerationController.WorkerTag.GENERATION: generation_worker,
        }
        super().__init__(
            prototype_controller=prototype_controller,
            workers=workers,
            max_parallel_requests=max_parallel_requests,
        )
        self._engine = engine
        self._generation_worker = generation_worker

    async def _handle_task_list(self, tasks: List[Task], request=None):
        await self._engine.replay_task_list(tasks, self._generation_worker)
        for task in tasks:
            if task.streaming_output_flag:
                for output in task.streaming_output_list:
                    request.result.set_output_streaming(output)
                task.streaming_output_list = []


def create_replay_scaffolding_llm(
    trace: ExecutionTrace,
    generation_worker: Worker,
    prototype_controller: Controller,
    latency_scale: float = 1.0,
    max_parallel_requests: int = 64,
) -> ScaffoldingLlm:
    """Create a ScaffoldingLlm configured for trace-driven replay.

    During replay each yielded task list is matched against the next
    recorded event for the current branch path.  ``message`` events
    (role=assistant) run on the real *generation_worker* with synthetic
    inputs whose token lengths come from the trace.  ``tool_call``
    events are replayed from the trace with simulated latency matching
    the recorded duration (scaled by *latency_scale*).
    ``drop_kv_cache`` events are replayed as no-ops.  Multi-task yields
    are matched against ``parallel_start`` events with inline
    ``children``.
    """
    engine = TraceReplayEngine(
        trace,
        latency_scale=latency_scale,
    )
    return TraceReplayScaffoldingLlm(
        engine=engine,
        generation_worker=generation_worker,
        prototype_controller=prototype_controller,
        max_parallel_requests=max_parallel_requests,
    )
