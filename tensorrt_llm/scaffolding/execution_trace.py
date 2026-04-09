import json
import uuid
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

# Keys omitted from the compact trace (``save(..., full=False)``) so the
# on-disk format matches historical ``*.trace.json`` files.
_FULL_TRACE_ONLY_KEYS = frozenset(
    {
        "llm_duration_ms",
        "llm_request_messages",
        "tool_calls_detail",
        "reasoning",
        "reasoning_content",
        "tool_arguments",
        "tool_result",
        "tool_stdout",
        "tool_stderr",
    }
)


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

    Full-trace-only fields (see ``ExecutionTrace.save(..., full=True)``):
      - ``llm_duration_ms``: wall time for the ChatTask / GenerationTask yield
        (LLM round-trip) in milliseconds.
      - ``llm_request_messages``: messages sent to the model for that
        completion (chat), or a minimal request description.
      - ``tool_calls_detail``: structured native tool_calls on assistant turns.
      - ``reasoning`` / ``reasoning_content``: assistant reasoning payloads.
      - ``tool_arguments`` / ``tool_result``: MCP call input and raw result.
      - ``tool_stdout`` / ``tool_stderr``: sandbox streams for tool responses
        (``message`` with role ``tool`` and ``tool_call`` events when present).
      - ``tool_name`` / ``tool_arguments`` on ``message`` with role ``tool``:
        duplicate of the preceding matching ``tool_call`` event (full trace).
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

    # -- full trace: LLM turn (event_type == "message", assistant) --
    llm_duration_ms: Optional[float] = None
    llm_request_messages: Optional[List[Dict[str, Any]]] = None
    tool_calls_detail: Optional[List[Dict[str, Any]]] = None
    reasoning: Optional[str] = None
    reasoning_content: Optional[str] = None

    # -- tool_call fields (event_type == "tool_call") --
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    duration_ms: Optional[float] = None
    tool_arguments: Optional[Any] = None
    tool_result: Optional[str] = None
    tool_stdout: Optional[str] = None
    tool_stderr: Optional[str] = None

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

    def save(self, path: str, *, full: bool = False):
        """Serialize and write the trace to a JSON file.

        With ``full=False`` (default), omit message ``content`` and other
        verbose fields so the file matches the historical compact trace
        format (token counts and structure only).

        With ``full=True``, write message bodies, LLM request snapshots,
        structured tool calls, MCP arguments/results, and per-turn timings.
        """
        if full:
            data = {
                "trace_id": self.trace_id,
                "events": [_full_trace_event_dict(ev) for ev in self.events],
            }
        else:
            data = {
                "trace_id": self.trace_id,
                "events": [_compact_trace_event_dict(ev) for ev in self.events],
            }
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


def _compact_trace_event_dict(ev: TraceEvent) -> dict:
    """Dict for compact JSON export (backward-compatible with older traces)."""
    d = _strip_none(asdict(ev))
    d.pop("content", None)
    for k in _FULL_TRACE_ONLY_KEYS:
        d.pop(k, None)
    # Tool name/args on role=tool messages are full-trace-only (mirror tool_call).
    if ev.event_type == "message" and ev.role == "tool":
        d.pop("tool_name", None)
    return d


def _full_trace_event_dict(ev: TraceEvent) -> dict:
    """Dict for full JSON export; keep tool stdio keys as JSON null when unset."""
    d = _strip_none(asdict(ev))
    if ev.event_type == "tool_call":
        d["tool_stdout"] = ev.tool_stdout
        d["tool_stderr"] = ev.tool_stderr
    elif ev.event_type == "message" and ev.role == "tool":
        d["tool_stdout"] = ev.tool_stdout
        d["tool_stderr"] = ev.tool_stderr
    return d


def _parse_event(ev_data: dict) -> TraceEvent:
    """Deserialize a single TraceEvent from a JSON dict."""
    d = dict(ev_data)
    if d.get("children") is not None:
        d["children"] = [_parse_event(c) for c in d["children"]]
    d = {k: v for k, v in d.items() if k in _TRACE_EVENT_FIELDS}
    return TraceEvent(**d)
