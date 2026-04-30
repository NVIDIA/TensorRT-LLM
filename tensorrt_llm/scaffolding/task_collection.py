import json
import os
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type

from .controller import Controller, ParallelProcess
from .execution_scope import current_scope
from .execution_trace import ExecutionTrace, TraceEvent
from .task import (ChatTask, DropKVCacheTask, GenerationTask, MCPCallTask, Task,
                   TokenizeTask)


class TaskCollection:

    def __init__(self):
        # reserved for future use
        pass

    def before_yield(self, tasks: List[Task]):
        pass

    def after_yield(self, tasks: List[Task]):
        pass

    def on_parallel_start(self, num_branches: int):
        pass

    def on_parallel_end(self, num_branches: int):
        pass

    @staticmethod
    def get_global_info() -> Any:
        pass


def with_task_collection(name: str, task_collection_cls: Type[TaskCollection],
                         **task_collection_kwargs):

    def decorator(controller_cls: Type[Controller]):
        original_init = controller_cls.__init__
        original_process = controller_cls.process

        # add task collection to controller
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.task_collections[name] = task_collection_cls(
                **task_collection_kwargs)

        def new_process(self, tasks: List[Task], **kwargs):

            class TaskCollectionWrapper:

                def __init__(self, task_collection, gen):
                    self.task_collection = task_collection
                    self.gen = gen

                def __call__(self):
                    for obj in self.gen:
                        if isinstance(obj, ParallelProcess):
                            num_branches = len(obj.sub_gens)
                            obj.sub_gens = [
                                TaskCollectionWrapper(self.task_collection,
                                                      sub_gen)
                                for sub_gen in obj.sub_gens
                            ]
                            self.task_collection.on_parallel_start(num_branches)
                            yield obj
                            self.task_collection.on_parallel_end(num_branches)
                        else:
                            self.task_collection.before_yield(obj)
                            yield obj
                            self.task_collection.after_yield(obj)

                def __iter__(self):
                    return self.__call__()

            original_gen = original_process(self, tasks, **kwargs)
            new_gen = TaskCollectionWrapper(self.task_collections[name],
                                            original_gen)
            return new_gen()

        controller_cls.__init__ = new_init
        controller_cls.process = new_process

        return controller_cls

    return decorator


class GenerationTokenCounter(TaskCollection):

    def __init__(self):
        super().__init__()
        self.generation_token_count = 0
        self.pre_worker_token_sum = 0

    def before_yield(self, tasks: List[Task]):
        self.pre_worker_token_sum = 0
        for task in tasks:
            if isinstance(task, GenerationTask) or issubclass(
                    type(task), GenerationTask):
                if task.output_tokens:
                    self.pre_worker_token_sum += len(task.output_tokens)

    def after_yield(self, tasks: List[Task]):
        post_worker_token_sum = 0
        for task in tasks:
            # only support GenerationTask for now
            if isinstance(task, GenerationTask) or issubclass(
                    type(task), GenerationTask):
                if task.output_tokens:
                    post_worker_token_sum += len(task.output_tokens)
        self.generation_token_count += post_worker_token_sum - self.pre_worker_token_sum


class ChatTokenCounter(TaskCollection):

    # prompt tokens, completion tokens
    statistics: Dict[str, List[Tuple[int, int]]] = {}

    def __init__(self, statistics_name: str):
        super().__init__()
        self.statistics_name = statistics_name
        if statistics_name not in ChatTokenCounter.statistics:
            ChatTokenCounter.statistics[statistics_name] = []

    def before_yield(self, tasks: List[Task]):
        for task in tasks:
            if not isinstance(task, ChatTask):
                continue
            task.enable_token_counting = True

    def after_yield(self, tasks: List[Task]):
        for task in tasks:
            if not isinstance(task, ChatTask):
                continue
            ChatTokenCounter.statistics[self.statistics_name].append(
                (task.prompt_tokens_num, task.completion_tokens_num))

    def get_global_info() -> Any:
        return ChatTokenCounter.statistics


class TaskTimer(TaskCollection):

    statistics: Dict[str, Dict[type, List[float]]] = {}

    def __init__(self, statistics_name: str, task_types: List[Type[Task]]):
        super().__init__()
        self.statistics_name = statistics_name
        self.task_types = task_types
        self.start_time_map = {}
        if statistics_name not in TaskTimer.statistics:
            TaskTimer.statistics[statistics_name] = {}
        for task_type in task_types:
            if task_type not in TaskTimer.statistics[statistics_name]:
                TaskTimer.statistics[statistics_name][task_type] = []

    def before_yield(self, tasks: List[Task]):
        for task in tasks:
            if type(task) not in self.task_types:
                continue

            self.start_time_map[id(task)] = time.time()

    def after_yield(self, tasks: List[Task]):
        for task in tasks:
            if type(task) not in self.task_types:
                continue

            end_time = time.time()
            TaskTimer.statistics[self.statistics_name][type(task)].append(
                end_time - self.start_time_map[id(task)])
            del self.start_time_map[id(task)]

    def get_global_info() -> Any:
        return TaskTimer.statistics


class QueryCollector(TaskCollection):
    file_name = "query_result.json"
    query_dict = {}

    def __init__(self):
        super().__init__()

    def after_yield(self, tasks: List[Task]):
        for task in tasks:
            if not isinstance(task, MCPCallTask):
                continue
            args = json.loads(task.args)
            if 'query' in args:
                QueryCollector.query_dict[args['query']] = task.result_str

    def get_global_info() -> Any:
        with open(QueryCollector.file_name, 'w', encoding='utf-8') as f:
            json.dump(QueryCollector.query_dict,
                      f,
                      indent=4,
                      ensure_ascii=False)
        return None


class TaskMetricsCollector(TaskCollection):
    """Task profiler that captures tasks at yield points, avoiding duplicate counting.

    Records token usage and execution time for each task.

    Supports filtering by task types and captures additional fields for ChatTask
    including finish_reason, unique_id, and optionally message content.

    For :class:`MCPCallTask`, records ``tool_call_id``, ``tool_name``, ``mcp_args``
    (arguments as dict when JSON-decodable, else the original value), and
    ``result_str`` after the worker completes.

    When capture_messages is enabled, also captures comprehensive trace information
    including messages, new_messages added during the yield, and sub_request_markers.
    """

    # Global statistics: controller_name -> List[task_info_dict]
    statistics: Dict[str, List[Dict[str, Any]]] = {}

    def __init__(self,
                 controller_name: str,
                 task_types: List[Type[Task]] = None,
                 enable_print: bool = True,
                 capture_messages: bool = False):
        super().__init__()
        self.controller_name = controller_name
        self.task_types = task_types
        self.enable_print = enable_print
        self.capture_messages = capture_messages
        self.start_time_map: Dict[int, float] = {}
        self.pre_message_count_map: Dict[int, int] = {}

        if controller_name not in TaskMetricsCollector.statistics:
            TaskMetricsCollector.statistics[controller_name] = []

    def _should_process_task(self, task: Task) -> bool:
        if self.task_types is not None and type(task) not in self.task_types:
            return False
        return True

    def _is_task_already_profiled(self, task: Task) -> bool:
        return getattr(task, '_profiling_in_progress', False)

    def _mark_task_profiling_start(self, task: Task):
        task._profiling_in_progress = True

    def _mark_task_profiling_end(self, task: Task):
        task._profiling_in_progress = False

    def before_yield(self, tasks: List[Task]):
        for task in tasks:
            if not self._should_process_task(task):
                continue
            if self._is_task_already_profiled(task):
                continue

            self._mark_task_profiling_start(task)
            task_id = id(task)
            self.start_time_map[task_id] = time.time()

            if isinstance(task, ChatTask):
                task.enable_token_counting = True
                if self.capture_messages:
                    self.pre_message_count_map[task_id] = len(task.messages)
            if self.enable_print:
                self._print_task_start(task)

    def after_yield(self, tasks: List[Task]):
        for task in tasks:
            task_id = id(task)
            if task_id not in self.start_time_map:
                continue

            end_time = time.time()
            duration = end_time - self.start_time_map[task_id]
            del self.start_time_map[task_id]
            self._mark_task_profiling_end(task)

            task_info = {
                'controller': self.controller_name,
                'task_type': type(task).__name__,
                'duration_ms': duration * 1000,
                'timestamp': end_time,
            }

            if isinstance(task, ChatTask):
                task_info['prompt_tokens'] = getattr(task, 'prompt_tokens_num',
                                                     0)
                task_info['completion_tokens'] = getattr(
                    task, 'completion_tokens_num', 0)
                task_info['reasoning_tokens'] = getattr(task,
                                                        'reasoning_tokens_num',
                                                        0)
                task_info['total_tokens'] = task_info[
                    'prompt_tokens'] + task_info['completion_tokens']
                task_info['finish_reason'] = getattr(task, 'finish_reason',
                                                     None)
                task_info['unique_id'] = getattr(task, 'unique_id', None)
                task_info['sub_request_markers'] = getattr(
                    task, 'sub_request_markers', [])

                # Capture messages if enabled
                if self.capture_messages:
                    pre_message_count = self.pre_message_count_map.get(
                        task_id, 0)
                    if task_id in self.pre_message_count_map:
                        del self.pre_message_count_map[task_id]

                    task_info['message_count_before'] = pre_message_count
                    task_info['message_count_after'] = len(task.messages)
                    task_info['messages'] = [
                        self._serialize_message(msg) for msg in task.messages
                    ]
                    # Capture only the new messages added during this yield
                    if len(task.messages) > pre_message_count:
                        task_info['new_messages'] = [
                            self._serialize_message(msg)
                            for msg in task.messages[pre_message_count:]
                        ]
                    else:
                        task_info['new_messages'] = []

            elif isinstance(task, MCPCallTask):
                task_info['tool_call_id'] = task.tool_call_id
                task_info['tool_name'] = task.tool_name
                task_info[
                    'mcp_args'] = TaskMetricsCollector._serialize_mcp_args(
                        task.args)
                task_info['result_str'] = task.result_str

            TaskMetricsCollector.statistics[self.controller_name].append(
                task_info)

            if self.enable_print:
                self._print_task_info(task_info)

    @staticmethod
    def _serialize_mcp_args(args: Any) -> Any:
        """Normalize MCP task arguments for metrics export."""
        if args is None:
            return None
        if isinstance(args, dict):
            return args
        if isinstance(args, str):
            try:
                return json.loads(args)
            except (json.JSONDecodeError, TypeError):
                return args
        return args

    def _serialize_message(self, message) -> Dict[str, Any]:
        """Serialize a RoleMessage to a dictionary."""
        result = {
            "role": getattr(message, "role", None),
            "content": getattr(message, "content", None),
        }
        # Capture additional fields for AssistantMessage
        if hasattr(message, "reasoning") and message.reasoning is not None:
            result["reasoning"] = message.reasoning
        if hasattr(
                message,
                "reasoning_content") and message.reasoning_content is not None:
            result["reasoning_content"] = message.reasoning_content
        if hasattr(message, "tool_calls") and message.tool_calls is not None:
            result["tool_calls"] = [str(tc) for tc in message.tool_calls]
        if getattr(message, "role", None) == "assistant":
            fr = getattr(message, "finish_reason", None)
            if fr is not None:
                result["finish_reason"] = fr
        return result

    def _print_task_start(self, task: Task):
        """Print a compact preview before the task is sent to a worker."""
        log_parts = [
            f"[{self.controller_name}]",
            f"{type(task).__name__}",
            "START",
        ]

        if isinstance(task, ChatTask):
            content_chars = sum(
                len(str(getattr(message, "content", "") or ""))
                for message in task.messages)
            tool_count = len(task.tools) if task.tools is not None else 0
            log_parts.append(
                f"messages={len(task.messages)} content_chars={content_chars} tools={tool_count}"
            )
        elif isinstance(task, MCPCallTask):
            log_parts.append(
                f"tool={task.tool_name!r} id={task.tool_call_id!r} "
                f"args={TaskMetricsCollector._serialize_mcp_args(task.args)!r}")

        print(" | ".join(log_parts))

    def _print_task_info(self, task_info: Dict[str, Any]):
        log_parts = [
            f"[{task_info['controller']}]", f"{task_info['task_type']}",
            f"⏱️ {task_info['duration_ms']:.2f}ms"
        ]

        if 'prompt_tokens' in task_info:
            log_parts.append(f"🎯 prompt={task_info['prompt_tokens']} "
                             f"completion={task_info['completion_tokens']} "
                             f"reasoning={task_info['reasoning_tokens']} "
                             f"total={task_info['total_tokens']}")

        print(" | ".join(log_parts))

        if task_info.get('task_type') == 'MCPCallTask':
            print(f"    MCP tool={task_info.get('tool_name')!r} "
                  f"id={task_info.get('tool_call_id')!r} "
                  f"args={task_info.get('mcp_args')!r}")
            rs = task_info.get('result_str')
            if rs is not None:
                preview = rs if len(rs) <= 200 else rs[:200] + "..."
                print(f"    result_str: {preview}")

        # Print message details if capture_messages is enabled
        if 'new_messages' in task_info and task_info['new_messages']:
            print(
                f"    Messages: {task_info['message_count_before']} -> {task_info['message_count_after']}"
            )
            print("    New Messages:")
            for msg in task_info['new_messages']:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                # Truncate long content for display
                if content and len(content) > 200:
                    content = content[:200] + "..."
                print(f"      [{role}]: {content}")

    @staticmethod
    def _compute_stats(values: List[float]) -> Dict[str, float]:
        """Compute avg, median, min, max, sum for a list of values."""
        if not values:
            return {'avg': 0, 'median': 0, 'min': 0, 'max': 0, 'sum': 0}
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        median = sorted_vals[n //
                             2] if n % 2 == 1 else (sorted_vals[n // 2 - 1] +
                                                    sorted_vals[n // 2]) / 2
        return {
            'avg': sum(values) / n,
            'median': median,
            'min': min(values),
            'max': max(values),
            'sum': sum(values),
        }

    @staticmethod
    def print_summary():
        """Print summary statistics for all controllers."""
        print("\n" + "=" * 80)
        print("TASK METRICS SUMMARY")
        print("=" * 80)

        for controller_name, task_list in TaskMetricsCollector.statistics.items(
        ):
            if not task_list:
                continue

            print(f"\n📊 {controller_name} ({len(task_list)} records)")
            print("-" * 70)

            # Group by task type
            task_type_data: Dict[str, Dict[str, List[float]]] = {}

            for task_info in task_list:
                task_type = task_info['task_type']
                if task_type not in task_type_data:
                    task_type_data[task_type] = {
                        'duration_ms': [],
                        'prompt_tokens': [],
                        'completion_tokens': [],
                        'reasoning_tokens': [],
                        'total_tokens': [],
                    }

                data = task_type_data[task_type]
                data['duration_ms'].append(task_info['duration_ms'])
                data['prompt_tokens'].append(task_info.get('prompt_tokens', 0))
                data['completion_tokens'].append(
                    task_info.get('completion_tokens', 0))
                data['reasoning_tokens'].append(
                    task_info.get('reasoning_tokens', 0))
                data['total_tokens'].append(task_info.get('total_tokens', 0))

            # Print statistics for each task type
            for task_type, data in task_type_data.items():
                count = len(data['duration_ms'])
                print(f"\n  {task_type} (count: {count})")

                # Duration stats
                duration_stats = TaskMetricsCollector._compute_stats(
                    data['duration_ms'])
                print(
                    f"    Duration (ms):     sum={duration_stats['sum']:.2f}, "
                    f"avg={duration_stats['avg']:.2f}, "
                    f"median={duration_stats['median']:.2f}, "
                    f"min={duration_stats['min']:.2f}, max={duration_stats['max']:.2f}"
                )

                # Token stats (only if there are tokens)
                if sum(data['total_tokens']) > 0:
                    prompt_stats = TaskMetricsCollector._compute_stats(
                        data['prompt_tokens'])
                    completion_stats = TaskMetricsCollector._compute_stats(
                        data['completion_tokens'])
                    reasoning_stats = TaskMetricsCollector._compute_stats(
                        data['reasoning_tokens'])
                    total_stats = TaskMetricsCollector._compute_stats(
                        data['total_tokens'])

                    print(
                        f"    Prompt tokens:     sum={prompt_stats['sum']:.0f}, "
                        f"avg={prompt_stats['avg']:.1f}, "
                        f"median={prompt_stats['median']:.1f}, "
                        f"min={prompt_stats['min']:.0f}, max={prompt_stats['max']:.0f}"
                    )
                    print(
                        f"    Completion tokens: sum={completion_stats['sum']:.0f}, "
                        f"avg={completion_stats['avg']:.1f}, "
                        f"median={completion_stats['median']:.1f}, "
                        f"min={completion_stats['min']:.0f}, max={completion_stats['max']:.0f}"
                    )
                    print(
                        f"    Reasoning tokens:  sum={reasoning_stats['sum']:.0f}, "
                        f"avg={reasoning_stats['avg']:.1f}, "
                        f"median={reasoning_stats['median']:.1f}, "
                        f"min={reasoning_stats['min']:.0f}, max={reasoning_stats['max']:.0f}"
                    )
                    print(
                        f"    Total tokens:      sum={total_stats['sum']:.0f}, "
                        f"avg={total_stats['avg']:.1f}, "
                        f"median={total_stats['median']:.1f}, "
                        f"min={total_stats['min']:.0f}, max={total_stats['max']:.0f}"
                    )

        print("\n" + "=" * 80 + "\n")

    @staticmethod
    def get_statistics(
            controller_name: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """Get statistics for a specific controller or all controllers."""
        if controller_name is not None:
            return {
                controller_name:
                TaskMetricsCollector.statistics.get(controller_name, [])
            }
        return TaskMetricsCollector.statistics

    @staticmethod
    def get_all_records() -> List[Dict[str, Any]]:
        """Get all records across all controllers as a flat list."""
        all_records = []
        for records in TaskMetricsCollector.statistics.values():
            all_records.extend(records)
        # Sort by timestamp
        all_records.sort(key=lambda x: x.get('timestamp', 0))
        return all_records

    @staticmethod
    def export_to_json(file_path: str, controller_name: str = None):
        """Export metrics to a JSON file."""
        if controller_name is not None:
            data = TaskMetricsCollector.statistics.get(controller_name, [])
        else:
            data = TaskMetricsCollector.statistics
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    @staticmethod
    def reset(controller_name: str = None):
        """Reset statistics for a specific controller or all controllers."""
        if controller_name is not None:
            if controller_name in TaskMetricsCollector.statistics:
                TaskMetricsCollector.statistics[controller_name] = []
        else:
            TaskMetricsCollector.statistics.clear()

    @staticmethod
    def get_global_info() -> Any:
        return TaskMetricsCollector.statistics


class ExecutionTracer(TaskCollection):
    """Captures the full execution flow of a controller as a chat transcript.

    Every message in a ChatTask conversation becomes its own ``message``
    event with a ``conversation_id``.  Non-assistant messages (system, user,
    tool) are emitted in ``before_yield`` as context; assistant messages are
    emitted in ``after_yield`` with generation metadata (token counts,
    tool_calls, duration).

    GenerationTask yields produce two ``message`` events: a ``user`` event
    for the input and an ``assistant`` event for the output.

    MCPCallTask and DropKVCacheTask are recorded as ``tool_call`` and
    ``drop_kv_cache`` events respectively.  Full traces (see
    ``ExecutionTrace.save(..., full=True)``) also store MCP arguments,
    tool results, per-turn LLM latency, request message snapshots, and
    assistant message bodies.  Each ``message`` event with role ``tool``
    repeats ``tool_name`` and ``tool_arguments`` from the matching
    ``tool_call``.  ``llm_request_messages`` preserves the messages sent to
    the chat completion API; request tool schemas are stored once on the
    conversation's system message in ``llm_request_tools``.

    Multi-task yields and ParallelProcess forks both produce a
    ``parallel_start`` / ``parallel_end`` pair with child events recorded
    as separate top-level events on sub-branch paths.

    Attach via ``@with_execution_tracing`` decorator.
    After execution, call ``export_trace()`` to obtain an
    ``ExecutionTrace`` that can be saved to JSON and later replayed.
    """

    def __init__(self, controller_name: str = "root"):
        super().__init__()
        self.controller_name = controller_name
        self.events: List[TraceEvent] = []
        self._start_times: Dict[int, float] = {}
        self._pre_message_counts: Dict[int, int] = {}
        self._conversation_ids: Dict[int, int] = {}
        self._last_recorded_counts: Dict[int, int] = {}
        self._conv_counter = 0

    def _get_conversation_id(self, task_id: int) -> int:
        """Return a stable conversation_id for a task, assigning one on first encounter."""
        if task_id not in self._conversation_ids:
            self._conversation_ids[task_id] = self._conv_counter
            self._conv_counter += 1
        return self._conversation_ids[task_id]

    def _is_task_already_traced(self, task: Task) -> bool:
        return getattr(task, '_tracing_in_progress', False)

    def _mark_task_tracing_start(self, task: Task):
        task._tracing_in_progress = True

    def _mark_task_tracing_end(self, task: Task):
        task._tracing_in_progress = False

    def before_yield(self, tasks: List[Task]):
        for task in tasks:
            if self._is_task_already_traced(task):
                continue
            self._mark_task_tracing_start(task)
            task_id = id(task)
            self._start_times[task_id] = time.time()

            if isinstance(task, ChatTask):
                task.enable_token_counting = True
                pre_count = len(task.messages)
                self._pre_message_counts[task_id] = pre_count

                conv_id = self._get_conversation_id(task_id)
                last_recorded = self._last_recorded_counts.get(task_id, 0)
                branch_path = self._get_branch_path()
                llm_request_tools = None
                if task.tools is not None:
                    llm_request_tools = [tool.to_dict() for tool in task.tools]
                for i, msg in enumerate(task.messages[last_recorded:pre_count]):
                    message_index = last_recorded + i
                    self.events.append(
                        self._message_event_from_role_message(
                            msg,
                            conv_id,
                            branch_path,
                            llm_request_tools=llm_request_tools,
                            message_index=message_index))
                self._last_recorded_counts[task_id] = pre_count

    def after_yield(self, tasks: List[Task]):
        assistant_events: List[TraceEvent] = []
        now = time.time()

        for task in tasks:
            task_id = id(task)
            if task_id not in self._start_times:
                continue

            duration_ms = (now - self._start_times[task_id]) * 1000
            del self._start_times[task_id]
            self._mark_task_tracing_end(task)

            event = self._build_yield_event(task, task_id, duration_ms)
            if event is not None:
                assistant_events.append(event)

        if not assistant_events:
            return

        if len(assistant_events) == 1:
            self.events.append(assistant_events[0])
        else:
            parent_path = self._get_branch_path()
            self.events.append(
                TraceEvent(
                    event_type="parallel_start",
                    branch_path=parent_path,
                    num_branches=len(assistant_events),
                ))
            for i, ev in enumerate(assistant_events):
                ev.branch_path = parent_path + [i]
                self.events.append(ev)
            self.events.append(
                TraceEvent(
                    event_type="parallel_end",
                    branch_path=parent_path,
                ))

    def on_parallel_start(self, num_branches: int):
        self.events.append(
            TraceEvent(
                event_type="parallel_start",
                num_branches=num_branches,
                branch_path=self._get_branch_path(),
            ))

    def on_parallel_end(self, num_branches: int):
        self.events.append(
            TraceEvent(
                event_type="parallel_end",
                branch_path=self._get_branch_path(),
            ))

    @staticmethod
    def _get_branch_path() -> List[int]:
        scope = current_scope.get()
        return scope.branch_path_list if scope is not None else []

    @staticmethod
    def _serialize_mcp_args_for_trace(args: Any) -> Any:
        if args is None:
            return None
        if isinstance(args, dict):
            return args
        if isinstance(args, str):
            try:
                return json.loads(args)
            except (json.JSONDecodeError, TypeError):
                return args
        return args

    @staticmethod
    def _tool_calls_detail_from_message(msg) -> Optional[List[Dict[str, Any]]]:
        tc_list = getattr(msg, "tool_calls", None)
        if not tc_list:
            return None
        detail: List[Dict[str, Any]] = []
        for tc in tc_list:
            fn = getattr(tc, "function", None)
            detail.append({
                "id": getattr(tc, "id", None),
                "type": getattr(tc, "type", "function"),
                "function": {
                    "name": getattr(fn, "name", None),
                    "arguments": getattr(fn, "arguments", None),
                } if fn is not None else None,
            })
        return detail

    @classmethod
    def _tool_call_meta_by_id_from_messages(
            cls, messages) -> Dict[str, Dict[str, Any]]:
        """Map ``tool_call_id`` to name/arguments from prior assistant ``tool_calls``."""
        out: Dict[str, Dict[str, Any]] = {}
        for msg in messages:
            if getattr(msg, "role", None) != "assistant":
                continue
            tcd = cls._tool_calls_detail_from_message(msg)
            if not tcd:
                continue
            for item in tcd:
                tid = item.get("id")
                fn = item.get("function") if item else None
                if not tid or not fn:
                    continue
                name = fn.get("name")
                args = cls._serialize_mcp_args_for_trace(fn.get("arguments"))
                out[tid] = {"tool_name": name, "tool_arguments": args}
        return out

    @staticmethod
    def _serialize_role_message_for_trace(
        message,
        tool_meta_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """JSON-serializable chat message for full traces (LLM request / context)."""
        result: Dict[str, Any] = {
            "role": getattr(message, "role", None),
            "content": getattr(message, "content", None),
        }
        if hasattr(message, "tool_call_id") and getattr(message, "tool_call_id",
                                                        None):
            result["tool_call_id"] = message.tool_call_id
        if getattr(message, "role", None) == "tool":
            ts = getattr(message, "trace_stdout", None)
            te = getattr(message, "trace_stderr", None)
            if ts is not None:
                result["stdout"] = ts
            if te is not None:
                result["stderr"] = te
            if tool_meta_by_id:
                tcid = getattr(message, "tool_call_id", None)
                meta = tool_meta_by_id.get(tcid) if tcid else None
                if meta is not None:
                    result["tool_name"] = meta.get("tool_name")
                    result["tool_arguments"] = meta.get("tool_arguments")
        if hasattr(message, "reasoning") and message.reasoning is not None:
            result["reasoning"] = message.reasoning
        if hasattr(message, "reasoning_content") and getattr(
                message, "reasoning_content", None) is not None:
            result["reasoning_content"] = message.reasoning_content
        tcd = ExecutionTracer._tool_calls_detail_from_message(message)
        if tcd:
            result["tool_calls"] = tcd
        if getattr(message, "role", None) == "assistant":
            fr = getattr(message, "finish_reason", None)
            if fr is not None:
                result["finish_reason"] = fr
        return result

    def _build_yield_event(self, task: Task, task_id: int,
                           duration_ms: float) -> Optional[TraceEvent]:
        """Build the consumable event emitted at a yield point."""
        branch_path = self._get_branch_path()

        if isinstance(task, ChatTask):
            pre_count = self._pre_message_counts.pop(task_id, 0)
            conv_id = self._get_conversation_id(task_id)
            new_messages = task.messages[pre_count:]
            self._last_recorded_counts[task_id] = len(task.messages)

            llm_request_messages = [
                message.to_dict() for message in task.messages[:pre_count]
                if getattr(message, "content", None) is not None
            ]
            first_assistant = None
            for msg in new_messages:
                if getattr(msg, "role", None) == "assistant":
                    first_assistant = msg
                    break
            # If a ChatTask yield failed and produced no assistant message,
            # avoid emitting a synthetic assistant event with zero tokens.
            if first_assistant is None:
                return None

            content_val = None
            reasoning_val = None
            reasoning_content_val = None
            tool_calls_detail = None
            finish_reason_val = getattr(task, 'finish_reason', None)
            if first_assistant is not None:
                content_val = getattr(first_assistant, "content", None)
                reasoning_val = getattr(first_assistant, "reasoning", None)
                reasoning_content_val = getattr(first_assistant,
                                                "reasoning_content", None)
                tool_calls_detail = self._tool_calls_detail_from_message(
                    first_assistant)
                msg_fr = getattr(first_assistant, "finish_reason", None)
                if msg_fr is not None:
                    finish_reason_val = msg_fr

            return TraceEvent(
                event_type="message",
                branch_path=branch_path,
                conversation_id=conv_id,
                role="assistant",
                message_index=pre_count,
                tool_calls=self._extract_tool_calls(new_messages),
                prompt_tokens=getattr(task, 'prompt_tokens_num', 0),
                completion_tokens=getattr(task, 'completion_tokens_num', 0),
                reasoning_tokens=getattr(task, 'reasoning_tokens_num', 0),
                finish_reason=finish_reason_val,
                content=content_val,
                llm_duration_ms=duration_ms,
                llm_request_params=getattr(task, "llm_request_params", None),
                llm_request_messages=llm_request_messages,
                tool_calls_detail=tool_calls_detail,
                reasoning=reasoning_val,
                reasoning_content=reasoning_content_val,
            )
        elif isinstance(task, MCPCallTask):
            return TraceEvent(
                event_type="tool_call",
                branch_path=branch_path,
                duration_ms=duration_ms,
                tool_call_id=task.tool_call_id,
                tool_name=task.tool_name,
                tool_arguments=self._serialize_mcp_args_for_trace(task.args),
                tool_result=task.result_str,
                tool_stdout=task.result_stdout,
                tool_stderr=task.result_stderr,
            )
        elif isinstance(task, GenerationTask):
            conv_id = self._conv_counter
            self._conv_counter += 1

            user_content = getattr(task, "input_str", None)
            self.events.append(
                TraceEvent(
                    event_type="message",
                    branch_path=branch_path,
                    conversation_id=conv_id,
                    role="user",
                    content=user_content,
                ))

            out_str = getattr(task, "output_str", None)
            llm_req = None
            if user_content is not None:
                llm_req = [{"role": "user", "content": user_content}]

            return TraceEvent(
                event_type="message",
                branch_path=branch_path,
                conversation_id=conv_id,
                role="assistant",
                content=out_str,
                prompt_tokens=(len(task.input_tokens)
                               if task.input_tokens else 0),
                completion_tokens=(len(task.output_tokens)
                                   if task.output_tokens else 0),
                reasoning_tokens=getattr(task, 'reasoning_tokens_num', 0),
                finish_reason=getattr(task, 'finish_reason', None),
                llm_duration_ms=duration_ms,
                llm_request_params=getattr(task, "llm_request_params", None),
                llm_request_messages=llm_req,
            )
        elif isinstance(task, DropKVCacheTask):
            return TraceEvent(
                event_type="drop_kv_cache",
                branch_path=branch_path,
            )
        else:
            return TraceEvent(
                event_type="message",
                branch_path=branch_path,
                role="assistant",
            )

    def _tool_call_event_for_tool_message(
            self, branch_path: List[int],
            tool_call_id: Optional[str]) -> Optional[TraceEvent]:
        """Most recent ``tool_call`` on ``branch_path`` matching ``tool_call_id``."""
        for ev in reversed(self.events):
            if ev.event_type != "tool_call":
                continue
            if ev.branch_path != branch_path:
                continue
            if tool_call_id is not None:
                if ev.tool_call_id == tool_call_id:
                    return ev
            else:
                return ev
        return None

    def _message_event_from_role_message(
            self,
            message,
            conversation_id: int,
            branch_path: List[int],
            llm_request_tools: Optional[List[Dict[str, Any]]] = None,
            message_index: Optional[int] = None) -> TraceEvent:
        """Convert a RoleMessage into a non-assistant message event."""
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)
        ev = TraceEvent(
            event_type="message",
            branch_path=branch_path,
            conversation_id=conversation_id,
            role=role,
            content=content,
            message_index=message_index,
        )
        if role == "assistant":
            fr = getattr(message, "finish_reason", None)
            if fr is not None:
                ev.finish_reason = fr
        if role == "system" and message_index == 0 and llm_request_tools:
            ev.llm_request_tools = llm_request_tools
        if role == "tool":
            ts = getattr(message, "trace_stdout", None)
            te = getattr(message, "trace_stderr", None)
            if ts is not None:
                ev.tool_stdout = ts
            if te is not None:
                ev.tool_stderr = te
            tcid = getattr(message, "tool_call_id", None)
            match = self._tool_call_event_for_tool_message(branch_path, tcid)
            if match is not None:
                ev.tool_name = match.tool_name
                ev.tool_arguments = match.tool_arguments
        return ev

    @staticmethod
    def _extract_tool_calls(output_messages) -> Optional[List[str]]:
        """Extract tool call names from the first assistant message, or None."""
        for msg in output_messages:
            if getattr(msg, "role", None) != "assistant":
                continue
            tc_list = getattr(msg, "tool_calls", None)
            if not tc_list:
                return None
            return [tc.function.name for tc in tc_list]
        return None

    def export_trace(self) -> ExecutionTrace:
        """Build and return the complete ExecutionTrace for this request."""
        return ExecutionTrace(events=list(self.events), )


def with_execution_tracing(controller_name: str):
    """Convenience decorator that attaches an ExecutionTracer to a controller."""
    return with_task_collection("execution_tracer",
                                ExecutionTracer,
                                controller_name=controller_name)


class SubRequestMarker(TaskCollection):

    class UniqueIdGenerator:

        def __init__(self):
            self.unique_id = -1

        def generate(self):
            self.unique_id += 1
            return self.unique_id

    unique_id_generator = UniqueIdGenerator()
    top_node_count = 0
    enable_sub_request_marker = os.getenv("ENABLE_SUB_REQUEST_MARKER",
                                          "0") == "1"
    set_unique_id_zero = os.getenv("SET_UNIQUE_ID_ZERO", "0") == "1"

    def __init__(self, node_name: str, is_top_level: bool = False):
        super().__init__()
        self.node_name = node_name
        if not SubRequestMarker.set_unique_id_zero:
            if is_top_level:
                self.unique_id = SubRequestMarker.top_node_count
                SubRequestMarker.top_node_count += 1
            else:
                self.unique_id = SubRequestMarker.unique_id_generator.generate()
        else:
            self.unique_id = 0
        self.sub_node_id_tansfer_map = {}
        self.sub_node_counter = {}

    def before_yield(self, tasks: List[Task]):
        if not SubRequestMarker.enable_sub_request_marker:
            return
        for task in tasks:
            if not isinstance(task, ChatTask):
                continue

            if task.unique_id is None:
                # new request from this chat task
                task.unique_id = SubRequestMarker.unique_id_generator.generate()
                task.sub_request_markers = [("LeafNode", task.unique_id)]

            # transfer the global unique id to the unique id from this node
            sub_node_name, raw_sub_node_id = task.sub_request_markers[-1]
            if sub_node_name not in self.sub_node_id_tansfer_map:
                sub_node_id = 0
                self.sub_node_id_tansfer_map[sub_node_name] = {
                    raw_sub_node_id: sub_node_id
                }
                self.sub_node_counter[sub_node_name] = sub_node_id + 1
            else:
                sub_node_id_map = self.sub_node_id_tansfer_map[sub_node_name]
                if raw_sub_node_id in sub_node_id_map:
                    sub_node_id = sub_node_id_map[raw_sub_node_id]
                else:
                    sub_node_id = self.sub_node_counter[sub_node_name]
                    sub_node_id_map[raw_sub_node_id] = sub_node_id
                    self.sub_node_counter[sub_node_name] += 1
            task.sub_request_markers[-1] = (sub_node_name, sub_node_id)

            task.sub_request_markers.append((self.node_name, self.unique_id))

    def after_yield(self, tasks: List[Task]):
        if not SubRequestMarker.enable_sub_request_marker:
            return
        for task in tasks:
            if not isinstance(task, ChatTask):
                continue
            task.unique_id = None


def sub_request_node(node_name: str, is_top_level: bool = False):

    def decorator(controller_cls: Type[Controller]):
        controller_cls_with_sub_request_marker = with_task_collection(
            "sub_request_marker",
            SubRequestMarker,
            node_name=node_name,
            is_top_level=is_top_level)(controller_cls)
        return controller_cls_with_sub_request_marker

    return decorator


class ChatCollection(TaskCollection):

    global_chat_task_set = set()

    def __init__(self):
        super().__init__()
        self.chat_tasks = []

    def before_yield(self, tasks: List[Task]):
        for task in tasks:
            if not isinstance(task, ChatTask):
                continue
            if id(task) not in ChatCollection.global_chat_task_set:
                ChatCollection.global_chat_task_set.add(id(task))
                self.chat_tasks.append(task)

    def __del__(self):
        for task in self.chat_tasks:
            ChatCollection.global_chat_task_set.remove(id(task))


class DropKVCacheWorkerTag(Enum):
    DROP_KV_CACHE = "drop_kv_cache"


def drop_kv_cache_scope():

    def decorator(controller_cls: Type[Controller]):
        controller_cls_with_chat_collection = with_task_collection(
            "ChatCollection", ChatCollection)(controller_cls)
        original_process = controller_cls_with_chat_collection.process

        def new_process(self, tasks: List[Task], **kwargs):

            def wrapper():
                yield from original_process(self, tasks, **kwargs)

                drop_kv_cache_tasks = []
                for task in self.task_collections["ChatCollection"].chat_tasks:
                    drop_kv_cache_tasks.append(
                        DropKVCacheTask(
                            chat_task=task,
                            worker_tag=DropKVCacheWorkerTag.DROP_KV_CACHE))
                yield drop_kv_cache_tasks

            return wrapper()

        controller_cls_with_chat_collection.process = new_process
        return controller_cls_with_chat_collection

    return decorator


class TokenizeWorkerTag(Enum):
    TOKENIZE = "tokenize"


def _collect_tokenizable_events(events: List[TraceEvent]) -> List[TraceEvent]:
    """Collect all user/system/tool message events."""
    return [
        ev for ev in events
        if ev.event_type == "message" and ev.role in ("system", "user", "tool")
    ]


def _balance_system_tokens(events: List[TraceEvent]) -> None:
    """Force each conversation's system tokens to match the first assistant prompt."""
    conversations: Dict[int, List[TraceEvent]] = {}
    for event in events:
        if event.event_type != "message" or event.conversation_id is None:
            continue
        conversations.setdefault(event.conversation_id, []).append(event)

    for messages in conversations.values():
        system_message = None
        first_assistant_index = None
        first_assistant_prompt_tokens = None
        for index, message in enumerate(messages):
            if message.role == "system" and system_message is None:
                system_message = message
                continue
            if message.role == "assistant" and message.prompt_tokens is not None:
                first_assistant_index = index
                first_assistant_prompt_tokens = message.prompt_tokens
                break

        if system_message is None or first_assistant_index is None or first_assistant_prompt_tokens is None:
            continue

        prompt_user_tokens = 0
        can_balance = True
        for message in messages[:first_assistant_index]:
            if message.role != "user":
                continue
            if message.tokens is None:
                can_balance = False
                break
            prompt_user_tokens += message.tokens
        if can_balance:
            system_message.tokens = first_assistant_prompt_tokens - prompt_user_tokens


def tokenize_trace_scope():

    def decorator(controller_cls: Type["Controller"]):
        original_process = controller_cls.process

        def new_process(self, tasks: List[Task], **kwargs):

            def wrapper():
                yield from original_process(self, tasks, **kwargs)

                tracer = self.task_collections.get("execution_tracer")
                if tracer is None:
                    return

                tokenizable_events = _collect_tokenizable_events(tracer.events)
                if not tokenizable_events:
                    return

                tokenize_tasks = []
                for event in tokenizable_events:
                    tokenize_tasks.append(
                        TokenizeTask(
                            content=event.content,
                            event=event,
                            worker_tag=TokenizeWorkerTag.TOKENIZE,
                        ))
                yield tokenize_tasks

                for task in tokenize_tasks:
                    if task.token_count is not None and task.event is not None:
                        task.event.tokens = task.token_count
                        task.event.tokens_source = "tokenize_endpoint"
                        task.event.tokenize_error = None
                    elif task.event is not None and task.tokenize_error:
                        task.event.tokenize_error = task.tokenize_error

                _balance_system_tokens(tracer.events)

            return wrapper()

        controller_cls.process = new_process
        return controller_cls

    return decorator
