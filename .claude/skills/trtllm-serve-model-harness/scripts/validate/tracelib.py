# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Readers shared by the conformance checks.

The checks read the request-trace layout that trtllm-serve writes when
``TRTLLM_REQUEST_TRACE_DIR`` is set (``tensorrt_llm/serve/request_trace.py``):

    <hour>/requests-<pid>.jsonl    {"event": "request", "trace_id", "route", "body", "headers", ...}
    <hour>/responses-<pid>.jsonl   {"event": "response", "trace_id", "status",
                                    "response": {"kind": "sse_text" | "json" | "text", "body": ...}, ...}

Shards may be plain, ``.zst`` or ``.gz``. The replay tool (``../replay_requests.py``)
writes the same two shapes, so every check
runs unchanged on recorded traffic and on replayed traffic.

Nothing here prints record content. Locators are (file, 1-based line, trace_id).
"""

import gzip
import hashlib
import io
import json
import os
import re
from typing import Iterator, List, Optional, Tuple

TERMINAL_EVENTS = ("response.completed", "response.incomplete", "response.failed")
OPENING_EVENTS = ("response.created", "response.in_progress")
CALL_TYPES = ("function_call", "custom_tool_call")
CALL_OUTPUT_TYPES = ("function_call_output", "custom_tool_call_output")

_SHARD = re.compile(r"(^|/)(requests|responses|raw)-[^/]*\.jsonl(\.zst|\.gz)?$")


def iter_lines(path: str) -> Iterator[Tuple[int, bytes]]:
    """Yield ``(line_no, raw_bytes)`` of one shard, without the newline."""
    if path.endswith(".zst"):
        import zstandard  # imported lazily: only compressed shards need it

        with open(path, "rb") as fh:
            reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(fh)
            for no, raw in enumerate(io.BufferedReader(reader, buffer_size=1 << 22), 1):
                yield no, raw.rstrip(b"\n")
    elif path.endswith(".gz"):
        with gzip.open(path, "rb") as fh:
            for no, raw in enumerate(fh, 1):
                yield no, raw.rstrip(b"\n")
    else:
        with open(path, "rb") as fh:
            for no, raw in enumerate(fh, 1):
                yield no, raw.rstrip(b"\n")


def expand_shards(paths: List[str], kind: Optional[str] = None) -> List[str]:
    """Files and directories -> sorted shard files. ``kind``: "requests", "responses" or "raw"."""
    found = []
    for path in paths:
        if os.path.isdir(path):
            for dirpath, _, names in os.walk(path, followlinks=True):
                for name in names:
                    full = os.path.join(dirpath, name)
                    m = _SHARD.search(full)
                    if m and (kind is None or m.group(2) == kind):
                        found.append(full)
        else:
            m = _SHARD.search(path)
            if kind is None or m is None or m.group(2) == kind:
                found.append(path)
    return sorted(set(found))


def sse_events(body: str) -> Iterator[Tuple[Optional[str], Optional[dict]]]:
    """Yield ``(event_name, data)`` per SSE frame; ``data`` is None when it is not JSON.

    Frames are split first and only then read, so generated text that happens
    to contain ``event:`` or ``data:`` lines cannot be mistaken for a frame.
    A frame without an ``event:`` line takes its name from ``data["type"]``.
    ``data: [DONE]`` frames are skipped.
    """
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        name, data = None, []
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                value = line[5:]
                data.append(value[1:] if value.startswith(" ") else value)
        if not data or data == ["[DONE]"]:
            continue
        try:
            obj = json.loads("\n".join(data))
        except json.JSONDecodeError:
            yield name, None
            continue
        if name is None and isinstance(obj, dict):
            name = obj.get("type")
        yield name, obj if isinstance(obj, dict) else None


class SseFrame:
    """One SSE frame, in arrival order.

    ``kind`` is ``json`` (data parsed to an object), ``done`` (``data: [DONE]``),
    ``comment`` (only ``:`` lines, a keepalive), ``unparsable`` (data that is not
    JSON) or ``empty`` (no data lines). ``name`` is the ``event:`` value, if any.
    """

    __slots__ = ("index", "name", "kind", "obj", "has_event_line")

    def __init__(self, index, name, kind, obj, has_event_line):
        self.index = index
        self.name = name
        self.kind = kind
        self.obj = obj
        self.has_event_line = has_event_line


def sse_frames(body: str) -> List[SseFrame]:
    """Every frame of an SSE body, including ``[DONE]`` and comment-only frames.

    Unlike ``sse_events`` nothing is skipped, so wire rules can check framing and
    order. Frames are split first and only then read (see ``sse_events``).
    """
    frames = []
    text = body.replace("\r\n", "\n")
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        name, data, comment, has_event = None, [], False, False
        for line in block.split("\n"):
            if line.startswith("event:"):
                name, has_event = line[6:].strip(), True
            elif line.startswith("data:"):
                value = line[5:]
                data.append(value[1:] if value.startswith(" ") else value)
            elif line.startswith(":"):
                comment = True
        index = len(frames)
        if not data:
            frames.append(SseFrame(index, name, "comment" if comment else "empty", None, has_event))
            continue
        joined = "\n".join(data)
        if joined.strip() == "[DONE]":
            frames.append(SseFrame(index, name, "done", None, has_event))
            continue
        try:
            obj = json.loads(joined)
        except json.JSONDecodeError:
            frames.append(SseFrame(index, name, "unparsable", None, has_event))
            continue
        if not isinstance(obj, dict):
            frames.append(SseFrame(index, name, "unparsable", None, has_event))
            continue
        frames.append(SseFrame(index, name, "json", obj, has_event))
    return frames


def header_map(record: dict) -> dict:
    """Lower-cased header name -> value, from a trace or capture record.

    Captures store headers as ``[[name, value], ...]``; other writers use a dict.
    """
    headers = record.get("headers") if isinstance(record, dict) else None
    out = {}
    if isinstance(headers, dict):
        items = headers.items()
    elif isinstance(headers, list):
        items = [h for h in headers if isinstance(h, (list, tuple)) and len(h) == 2]
    else:
        items = []
    for name, value in items:
        if isinstance(name, str):
            out[name.lower()] = value
    return out


def is_chat_stream(events: List[Tuple[Optional[str], Optional[dict]]]) -> bool:
    """A chat-completions stream carries no ``response.*`` events."""
    return not any((name or "").startswith("response.") for name, _ in events)


def terminal_object(record: dict) -> Tuple[str, Optional[str], Optional[dict]]:
    """``(kind, terminal_event, response_object)`` of one response record.

    kind is "responses", "chat" or "other". For a streamed Responses reply the
    object is the last terminal event's ``response``; for a JSON reply it is
    the body itself when the body is a Responses object.
    """
    resp = record.get("response") if isinstance(record, dict) else None
    if not isinstance(resp, dict):
        return "other", None, None
    body = resp.get("body")
    if resp.get("kind") == "sse_text" and isinstance(body, str):
        events = list(sse_events(body))
        if is_chat_stream(events):
            return "chat", None, None
        terminal = [(n, d) for n, d in events if n in TERMINAL_EVENTS and isinstance(d, dict)]
        if not terminal:
            return "responses", None, None
        name, data = terminal[-1]
        final = data.get("response")
        return "responses", name, final if isinstance(final, dict) else None
    if isinstance(body, dict) and body.get("object") == "response":
        return "responses", "json:" + str(body.get("status")), body
    if isinstance(body, dict) and body.get("object") == "chat.completion":
        return "chat", None, None
    return "other", None, None


def item_texts(item: dict) -> List[str]:
    """Text parts of a message or reasoning item (content, then summary)."""
    texts = []
    content = item.get("content")
    if isinstance(content, str):
        texts.append(content)
    elif isinstance(content, list):
        texts += [
            p["text"] for p in content if isinstance(p, dict) and isinstance(p.get("text"), str)
        ]
    summary = item.get("summary")
    if isinstance(summary, list):
        texts += [
            p["text"] for p in summary if isinstance(p, dict) and isinstance(p.get("text"), str)
        ]
    return texts


def group_of(path: str, depth: int = 0) -> str:
    """A report group for a shard path.

    depth=0 picks, in order: ``<cluster>/<pool>`` around a ``traces`` directory
    (the frozen-snapshot layout ``.../<cluster>/traces/<pool>/<run>/...``), the
    trace root directory name (``trace_*``), else the run directory three levels
    above the shard. depth=N>0: the N path components above the hour directory.
    """
    parts = path.split("/")
    if depth > 0:
        return "/".join(parts[max(0, len(parts) - 2 - depth) : len(parts) - 2]) or "?"
    if "traces" in parts:
        i = len(parts) - 1 - parts[::-1].index("traces")
        if 0 < i < len(parts) - 1:
            return f"{parts[i - 1]}/{parts[i + 1]}"
    for part in parts:
        if part.startswith("trace_"):
            return part
    return parts[-4] if len(parts) >= 4 else "?"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def first_difference(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def load_json_line(raw: bytes) -> Optional[dict]:
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return obj if isinstance(obj, dict) else None
