import json
import uuid
from dataclasses import asdict, dataclass, field, fields
from typing import List, Optional


def _strip_none(obj):
    """Recursively remove keys with None values from dicts."""
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(item) for item in obj]
    return obj


def _strip_keys(obj, keys):
    """Recursively remove specified keys from dicts."""
    if isinstance(obj, dict):
        return {k: _strip_keys(v, keys) for k, v in obj.items() if k not in keys}
    if isinstance(obj, list):
        return [_strip_keys(item, keys) for item in obj]
    return obj


@dataclass
class TraceEvent:
    """A single event in the execution trace.

    event_type semantics:
      - "message": a single message in a conversation.  ``conversation_id``
        groups messages belonging to the same ChatTask or GenerationTask.
        ``role`` is "system", "user", "assistant", or "tool".
        Assistant messages carry generation metadata (token counts,
        tool_calls, finish_reason).  Non-assistant messages are
        informational context recorded between yields.
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

    # -- message fields (event_type == "message") --
    conversation_id: Optional[int] = None
    role: Optional[str] = None
    tool_calls: Optional[List[str]] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    finish_reason: Optional[str] = None

    # -- tool_call fields (event_type == "tool_call") --
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    duration_ms: Optional[float] = None

    # -- parallel_start fields (event_type == "parallel_start") --
    num_branches: Optional[int] = None
    children: Optional[List["TraceEvent"]] = None

    # -- message content (for system/user messages, used by tokenize_trace_scope) --
    content: Optional[str] = None

    # -- tokenization annotation (filled by tokenize_trace_scope) --
    tokens: Optional[int] = None


@dataclass
class ExecutionTrace:
    """The complete per-request execution trace, serializable to JSON."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    events: List[TraceEvent] = field(default_factory=list)

    def save(self, path: str):
        """Serialize and write the trace to a JSON file.

        None-valued fields are stripped to keep the JSON compact — each
        event only contains the keys relevant to its ``event_type``.
        """
        data = _strip_keys(_strip_none(asdict(self)), {"content"})
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
            events=events,
        )


_TRACE_EVENT_FIELDS = {f.name for f in fields(TraceEvent)}


def _parse_event(ev_data: dict) -> TraceEvent:
    """Deserialize a single TraceEvent from a JSON dict."""
    d = dict(ev_data)
    if d.get("children") is not None:
        d["children"] = [_parse_event(c) for c in d["children"]]
    d = {k: v for k, v in d.items() if k in _TRACE_EVENT_FIELDS}
    return TraceEvent(**d)
