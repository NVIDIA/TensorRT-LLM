import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TaskRecord:
    """Captures a single task's input/output at a yield point."""

    task_type: str = ""
    worker_tag: str = ""

    # ChatTask fields
    input_messages: Optional[List[Dict[str, Any]]] = None
    output_messages: Optional[List[Dict[str, Any]]] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None

    # MCPCallTask fields
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Any] = None
    result_str: Optional[str] = None

    # GenerationTask fields
    input_str: Optional[str] = None
    output_str: Optional[str] = None
    output_token_count: Optional[int] = None


@dataclass
class TraceEvent:
    """A single event in the execution trace."""

    event_type: str = ""
    timestamp: float = 0.0
    duration_ms: float = 0.0
    tasks: Optional[List[TaskRecord]] = None
    num_branches: Optional[int] = None


@dataclass
class ExecutionTrace:
    """The complete per-request execution trace, serializable to JSON."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str = ""
    timestamp: float = field(default_factory=time.time)
    events: List[TraceEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: str):
        """Serialize and write the trace to a JSON file."""
        data = asdict(self)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    @classmethod
    def load(cls, path: str) -> "ExecutionTrace":
        """Deserialize an ExecutionTrace from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        events = []
        for ev_data in data.get("events", []):
            task_records = None
            if ev_data.get("tasks") is not None:
                task_records = [TaskRecord(**tr) for tr in ev_data["tasks"]]
            events.append(
                TraceEvent(
                    event_type=ev_data.get("event_type", ""),
                    timestamp=ev_data.get("timestamp", 0.0),
                    duration_ms=ev_data.get("duration_ms", 0.0),
                    tasks=task_records,
                    num_branches=ev_data.get("num_branches"),
                )
            )

        return cls(
            trace_id=data.get("trace_id", str(uuid.uuid4())),
            prompt=data.get("prompt", ""),
            timestamp=data.get("timestamp", 0.0),
            events=events,
            metadata=data.get("metadata", {}),
        )

    def get_mcp_responses(self) -> List[Tuple[str, Any, str, float]]:
        """Extract (tool_name, tool_args, result_str, duration_ms) from all MCPCallTask records."""
        results = []
        for event in self.events:
            if event.event_type != "task_yield" or event.tasks is None:
                continue
            for task_record in event.tasks:
                if task_record.task_type == "MCPCallTask":
                    results.append(
                        (
                            task_record.tool_name,
                            task_record.tool_args,
                            task_record.result_str,
                            event.duration_ms,
                        )
                    )
        return results
