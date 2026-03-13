import json
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple


def _strip_none(obj):
    """Recursively remove keys with None values from dicts."""
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(item) for item in obj]
    return obj


@dataclass
class TraceEvent:
    """A single event in the execution trace.

    event_type semantics:
      - "message": a single message in a conversation.  ``conversation_id``
        groups messages belonging to the same ChatTask or GenerationTask.
        ``role`` is "system", "user", "assistant", or "tool".
        Assistant messages carry generation metadata (token counts,
        tool_calls, finish_reason, duration_ms).  Non-assistant messages
        are informational context recorded between yields.
      - "tool_call": a single MCP tool invocation (MCPCallTask).
      - "parallel_start": a parallel branching point.  Two sub-cases:
          * ParallelProcess — ``children`` is ``None``; child events
            are recorded as separate top-level events on sub-branch paths.
          * Multi-task yield (pseudo-fork) — ``children`` contains one
            child TraceEvent per concurrently-dispatched task.
      - "drop_kv_cache": a KV-cache eviction marker (DropKVCacheTask).
    """

    event_type: str = ""
    branch_path: List[int] = field(default_factory=list)
    timestamp: float = 0.0
    duration_ms: float = 0.0
    worker_tag: str = ""

    # -- message fields (event_type == "message") --
    conversation_id: Optional[int] = None
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    output_token_count: Optional[int] = None
    input_tokens: Optional[int] = None

    # -- tool_call fields (event_type == "tool_call") --
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Any] = None
    result_str: Optional[str] = None

    # -- parallel_start fields (event_type == "parallel_start") --
    num_branches: Optional[int] = None
    children: Optional[List["TraceEvent"]] = None


@dataclass
class ExecutionTrace:
    """The complete per-request execution trace, serializable to JSON."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str = ""
    timestamp: float = field(default_factory=time.time)
    events: List[TraceEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: str):
        """Serialize and write the trace to a JSON file.

        None-valued fields are stripped to keep the JSON compact — each
        event only contains the keys relevant to its ``event_type``.
        """
        data = _strip_none(asdict(self))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    @classmethod
    def load(cls, path: str) -> "ExecutionTrace":
        """Deserialize an ExecutionTrace from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        events = [_parse_event(ev) for ev in data.get("events", [])]

        return cls(
            trace_id=data.get("trace_id", str(uuid.uuid4())),
            prompt=data.get("prompt", ""),
            timestamp=data.get("timestamp", 0.0),
            events=events,
            metadata=data.get("metadata", {}),
        )

    def annotate_input_tokens(self, tokenizer) -> None:
        """Add ``input_tokens`` to system/user/tool message events.

        Uses the provided tokenizer to compute the token count for each
        non-assistant message's content.  This makes per-message token
        lengths available in the saved JSON, so that tools like the replay
        engine can determine e.g. the system-prompt length without
        re-tokenizing.
        """
        self._annotate_events(self.events, tokenizer)

    @staticmethod
    def _annotate_events(events: List[TraceEvent], tokenizer) -> None:
        for event in events:
            if event.event_type == "message" and event.role in ("system", "user", "tool"):
                content = event.content or ""
                event.input_tokens = len(tokenizer.encode(content, add_special_tokens=False))
            if event.children:
                ExecutionTrace._annotate_events(event.children, tokenizer)

    def get_mcp_responses(self) -> List[Tuple[str, Any, str, float]]:
        """Extract (tool_name, tool_args, result_str, duration_ms) tuples."""
        results: List[Tuple[str, Any, str, float]] = []
        for event in self.events:
            if event.event_type == "tool_call":
                results.append(
                    (
                        event.tool_name,
                        event.tool_args,
                        event.result_str,
                        event.duration_ms,
                    )
                )
            elif event.event_type == "parallel_start" and event.children:
                for child in event.children:
                    if child.event_type == "tool_call":
                        results.append(
                            (
                                child.tool_name,
                                child.tool_args,
                                child.result_str,
                                child.duration_ms,
                            )
                        )
        return results


_TRACE_EVENT_FIELDS = {f.name for f in fields(TraceEvent)}


def _parse_event(ev_data: dict) -> TraceEvent:
    """Deserialize a single TraceEvent from a JSON dict."""
    d = dict(ev_data)
    if d.get("children") is not None:
        d["children"] = [_parse_event(c) for c in d["children"]]
    d = {k: v for k, v in d.items() if k in _TRACE_EVENT_FIELDS}
    return TraceEvent(**d)
