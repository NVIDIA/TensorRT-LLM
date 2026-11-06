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
"""Turn records -> per-check counts, for every protocol (references/validate.md section 4).

Three passes over request-trace shaped input (trace windows, replay output,
recorder captures), after an optional raw pass that reads the model's own text
(``raw-*.jsonl``) in its own format (``native.py``, check C-13):

1. requests: one small summary per request (identity, order, tool
   declarations by key, call policy), plus the FIRST appearance per thread of
   every item the harness echoes back and of every tool result;
2. responses: each reply joined to its request and parsed by its protocol
   adapter, then read a second time by the protocol's official SDK
   (``sdk_diff.py``, check C-12); per-reply checks run here, with content in
   memory only;
3. threads: each turn against the next request of its thread (echo, call
   outcomes, resends, over-length handling).

Memory grows with requests and calls, not with history length: a history
replays the same items every turn, and only their first appearance is kept.
"""

import bisect
import json
import multiprocessing
import re
import zlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import native as native_mod
import profiles as prof
import proto_anthropic
import proto_chat
import proto_responses
import sdk_diff
import usage as usage_mod
from tracelib import expand_shards, header_map, iter_lines, load_json_line
from turnlib import (
    DELIVERED_ONLY_REWRITES,
    ECHO_REWRITES,
    SERVING_SHAPE,
    RequestView,
    Violation,
    classify_call,
    marker_hits,
    payload_digest,
    rewritten_digest,
    safe_name,
    sha256_text,
    text_blocks,
)

EMPTY = sha256_text("")


def context_management_clears(value) -> bool:
    """Whether a request's ``context_management`` lets the server remove part of its history.

    Clearing thinking with ``keep: all`` removes nothing (Claude Code sends it on every thinking
    request), so it does not excuse a prompt that shrank. A shape this does not know is taken to
    clear, which only makes USG-08 skip the pair.
    """
    if not value:
        return False
    edits = value.get("edits") if isinstance(value, dict) else None
    if not isinstance(edits, list):
        return True
    for edit in edits:
        if not isinstance(edit, dict):
            return True
        keep = edit.get("keep")
        keeps_all = keep == "all" or (isinstance(keep, dict) and keep.get("type") == "all")
        if not (str(edit.get("type") or "").startswith("clear_thinking") and keeps_all):
            return True
    return False


# A template's tools section holds, besides the declarations, a heading, instructions and the call
# format, which go with the last tool: 96 tokens in GLM-5.3's chat template and 211 in DeepSeek
# V4.1's encoder (each measured with its own tokenizer). 512 is more than twice the longer.
TOOLS_PREAMBLE_TOKENS = 512


def removed_tools_bound(turn: tuple, following: tuple) -> Optional[int]:
    """The most a prompt can shrink, in tokens, because it declares fewer of a turn's tools.

    ``turn`` and ``following`` are two requests' ``turnlib.tool_declarations``. None unless the
    following request declares strictly fewer tools: its tool names a strict subset of the turn's
    (none at all included); other tools are not fewer ones. Otherwise the UTF-8 length of every
    declaration of the turn that the following request does not carry as it was (one it changed
    may have shrunk too), plus TOOLS_PREAMBLE_TOKENS when no tool is left. A template renders a
    declaration from its own fields, and every token of a rendering spans at least one byte.
    """
    names, left = {d[0] for d in turn}, {d[0] for d in following}
    if not left < names:
        return None
    kept = {d[1] for d in following}
    removed = sum(d[2] for d in turn if d[1] not in kept)
    return removed + (0 if left else TOOLS_PREAMBLE_TOKENS)


def side_request(body: dict) -> bool:
    """A Messages request for structured output without tools: a side request of the session.

    Claude Code asks for titles and classifications this way (``output_config.format``) and
    keeps the reply out of the conversation, so no later request echoes it (C-03).
    """
    config = body.get("output_config")
    return isinstance(config, dict) and bool(config.get("format")) and not body.get("tools")


def _echoable(blocks) -> tuple:
    """Text and reasoning blocks a harness can echo: empty text carries nothing to echo."""
    return tuple(b for b in blocks or () if not (b[1] in (EMPTY, None) and not b[2]))


def wording_matches(wording: str, message: str) -> bool:
    """A declared error wording matches a message: its parts, split at "...", appear in order."""
    parts = [p.strip().lower() for p in wording.split("...") if p.strip()]
    text, at = message.lower(), 0
    for part in parts:
        at = text.find(part, at)
        if at < 0:
            return False
        at += len(part)
    return bool(parts)


def overlength_patterns(spec) -> dict:
    """The compiled ``matches`` and ``unless`` of an ``errors.overlength`` with ``shape: any_error``.

    Case-insensitive, as a harness that matches error text by regex (pi) compiles them. A pattern
    that does not compile is a usage error, never a silent mismatch.
    """
    if not isinstance(spec, dict) or spec.get("shape") != "any_error":
        return {}
    compiled = {}
    for key in ("matches", "unless"):
        patterns = spec.get(key) or []
        if not isinstance(patterns, list) or not all(isinstance(p, str) for p in patterns):
            raise ValueError(f"harness profile errors.overlength.{key}: a list of regexes")
        try:
            compiled[key] = [re.compile(p, re.I) for p in patterns]
        except re.error as error:
            raise ValueError(f"harness profile errors.overlength.{key}: {error}") from error
    return compiled


def overlength_read_by_text(patterns: dict, http_status, terminal: dict) -> bool:
    """C-10 under ``shape: any_error``: the harness reads this over-length reply as one.

    Such a harness (pi) reads only the error's text, whatever its status and wherever it came: an
    HTTP error, or an error or failed terminal event in the stream. The text is the status, the
    error's code and type, and its message; the harness recovers when one of its ``matches``
    patterns is found in it and none of its ``unless`` patterns is (compiled, case-insensitive).
    """
    error = terminal.get("error") or {}
    failed = ((isinstance(http_status, int) and http_status >= 400)
              or terminal.get("kind") in ("error", "failed"))  # fmt: skip
    text = " ".join(str(x) for x in (http_status, error.get("code"), error.get("type"),
                                      terminal.get("message")) if x)  # fmt: skip
    return (failed and any(p.search(text) for p in patterns.get("matches", ()))
            and not any(p.search(text) for p in patterns.get("unless", ())))  # fmt: skip


def resend_cause(http_status, overlength: bool, failed: bool) -> str:
    """Why a harness sent a turn's request again, as H-02 counts it.

    After an over-length answer or another client error, a resend means our error mapping did not
    lead the harness to its recovery path (e.g. Codex resends after an HTTP 400 instead of
    compacting): H-02.
    A capacity answer (429) asks the client to come back, so its resend is a retry after a
    failure, as is one after a server error or a failed stream.
    """
    client_error = (isinstance(http_status, int) and 400 <= http_status < 500
                    and http_status not in CAPACITY_STATUSES)  # fmt: skip
    if overlength or client_error:
        return "H-02"
    return "H-02:retry_after_failure" if failed else "H-02:other"


def _digest(value) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str))


def _reask_key(body: dict) -> Optional[tuple]:
    """A messages request's frame digest, and the block digests of its last user message.

    Two requests with the same frame differ in the last user message only. A later one whose
    blocks extend the earlier one's asked the same thing again with more said: Claude Code does
    that when it drops an empty answer. The earlier answer is then superseded (C-03).
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return None
    users = [i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "user"]
    if not users:
        return None
    last = users[-1]
    content = messages[last].get("content")
    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
    if not isinstance(blocks, list):
        return None
    frame = dict(body, messages=messages[:last] + [None] + messages[last + 1 :])
    return _digest(frame), tuple(_digest(block) for block in blocks)


ADAPTERS = {
    "/v1/responses": proto_responses,
    "/v1/messages": proto_anthropic,
    "/v1/chat/completions": proto_chat,
}
PROTOCOL_OF = {route: mod.PROTOCOL for route, mod in ADAPTERS.items()}
CLIENT_SIDE = ("client_disconnected", "client_first_event_timeout", "client_timeout")
# The recorder could not reach the server, or lost it before the reply ended
# (record_requests.py writes `upstream_error`): the body is the recorder's own error, or the
# server's bytes up to the loss. Neither is a reply to judge; R-06 counts them.
UPSTREAM_SIDE = ("upstream_error",)
# The recorder stopped before the reply ended: the request is pending at the capture's end, the
# way a request at a window's cutoff is. Nothing is judged.
CAPTURE_SIDE = ("recorder_shutdown",)
# HTTP statuses a server may answer a well-formed request with (C-16 does not judge them).
CAPACITY_STATUSES = (429,)
# Operator probes (a self-test through the recorder): recorded, never the cell's traffic.
PROBE_HEADERS = ("x-mh-probe", "x-mh-selftest")
# Protocols whose requests carry the conversation as `messages` (re-asks, C-03).
REASK_PROTOCOLS = ("anthropic-messages", "chat-completions")
OVERLENGTH_WORDS = ("context_length_exceeded", "prompt is too long", "context length",
                    "maximum context", "too many tokens", "prompt_too_long", "too long")  # fmt: skip
COUNT_TOKENS = "/v1/messages/count_tokens"
# The engine request id inside a reply id: Chat's chatcmpl-<id>, Completions' cmpl-<id>, an
# Anthropic message named msg_<id>_<hex>, and a Responses reply named resp_<id>_<hex> (a
# disaggregated server names it after its disagg_request_id; the usual resp_<hex> names none).
ENGINE_ID = re.compile(
    r'"id"\s*:\s*"(?:(?:chatcmpl|cmpl)-(\d+)|msg_(\d+)_[0-9a-f]+|resp_(\d+)_[0-9a-f]+)"'
)
# An engine id counts the requests of one server process, so with several generation workers one
# id names a raw record on each. A reply joins the one that ran inside its window, from its
# request's arrival to its response's completion, widened on each side by this many seconds for
# the clock skew between the recorder's host and the generation worker's (C-13, join_raw).
RAW_CLOCK_SKEW_S = 2.0
REPETITIVE_MIN_CHARS = 2000
REPETITIVE_MAX_ZLIB_RATIO = 0.10
EXAMPLES = 5


def _example_key(example: dict) -> tuple:
    """Orders a check's examples by where they come from, whatever order the shards were read in."""
    line = example.get("line")
    return (str(example.get("file") or ""), line if isinstance(line, int) else -1,
            str(example.get("trace_id") or ""), json.dumps(example, sort_keys=True, default=str))  # fmt: skip


def _first_examples(examples: list) -> list:
    """The EXAMPLES examples that come first by _example_key: a report lists the same ones each run."""
    return sorted(examples, key=_example_key)[:EXAMPLES]


def instant(value) -> Optional[Tuple[float, float]]:
    """An ISO-8601 stamp as the earliest and latest instant it names, in seconds since the epoch.

    The recorder writes whole seconds (``2026-10-28T03:04:05Z``: some instant in that second); the
    raw-output hook and trtllm-serve's request trace write microseconds
    (``2026-10-28T03:04:05.123456+00:00``). A stamp without a zone is UTC, as every writer here
    writes UTC. Anything else is unknown (None).
    """
    if not isinstance(value, str) or "T" not in value:
        return None
    try:
        moment = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    clock = value.partition("T")[2]
    at = moment.timestamp()
    return at, at + (0.0 if "." in clock or "," in clock else 1.0)


def reply_window(arrival, response: dict) -> Optional[Tuple[float, float]]:
    """When a reply's generation can have run, as the earliest and latest instants.

    From its request's arrival (the request record's ``recorded_at``) to its response's completion
    (the response record's ``finished_at``; without one, the arrival plus ``timing.total_s``).
    """
    start = instant(arrival)
    if start is None:
        return None
    end = instant(response.get("finished_at"))
    if end is not None:
        return start[0], end[1]
    timing = response.get("timing") if isinstance(response.get("timing"), dict) else {}
    total = timing.get("total_s")
    if isinstance(total, (int, float)) and not isinstance(total, bool):
        return start[0], start[1] + float(total)
    return None


def fitting_raw(readings: list, window, reported: Optional[int]) -> Optional[dict]:
    """The one raw reading, of several under one engine id, that can be this reply's generation.

    A reading is ruled out only on evidence: a start or finish known to fall outside the reply's
    window widened by RAW_CLOCK_SKEW_S, or an output token count known to differ from the one the
    reply reports. The reading left is returned when it is the only one and its own start and
    finish are known; with no window, nothing is (fail closed).
    """
    if window is None:
        return None
    low, high = window[0] - RAW_CLOCK_SKEW_S, window[1] + RAW_CLOCK_SKEW_S

    def outside(stamp) -> bool:
        return stamp is not None and (stamp[1] < low or stamp[0] > high)

    left = [r for r in readings if not outside(r.get("started_at"))
            and not outside(r.get("finished_at"))
            and (reported is None or r.get("output_tokens") in (None, reported))]  # fmt: skip
    if len(left) != 1 or left[0].get("started_at") is None or left[0].get("finished_at") is None:
        return None
    return left[0]


def raw_stopped(record: dict) -> Optional[str]:
    """How a raw record says its generation was stopped: ``partial``, ``aborted``, or None.

    The raw-output hook writes a record per output when its request ends. A request that ended
    without a final chunk (an error) gets ``partial: true`` and the request's ``end_reason``; a
    request aborted in the worker gets ``aborted: true``. Either record holds what the model
    wrote until the server stopped it, not a finished generation.
    """
    if record.get("partial") is True or record.get("end_reason") not in (None, ""):
        return "partial"
    if record.get("aborted") is True:
        return "aborted"
    return None


def delivered_as_answer(reply) -> bool:
    """Whether a reply ended as an answer the client keeps: completed, or cut at its budget.

    A reply that ended in an error (an error event or body), a failed status or without a
    terminal event is not one: the client drops what it received.
    """
    return reply.terminal.get("kind") in ("completed", "incomplete")


def join_raw(index: dict, keys: list, window, reported: Optional[int]) -> tuple:
    """The raw reading a reply joins: ``(reading, "unique" | "resolved")``.

    ``index`` maps an id key to every raw reading under it: ``d:`` the orchestrator's
    ``disagg_request_id``, ``r:`` an engine request id. The first of the reply's keys that names
    exactly one reading joins it, whatever its times. Failing that, an engine id that names
    several joins the one that fits the reply (``fitting_raw``). A ``disagg_request_id`` is unique
    across the instance, so two readings under one join nothing. Without a join the result is
    ``(None, "ambiguous")`` when the keys name readings, ``(None, None)`` when they name none.
    """
    named = [(key, index[key]) for key in keys if key in index]
    for _, readings in named:
        if len(readings) == 1:
            return readings[0], "unique"
    for key, readings in named:
        fit = fitting_raw(readings, window, reported) if key.startswith("r:") else None
        if fit is not None:
            return fit, "resolved"
    return None, ("ambiguous" if named else None)


class Pipeline:
    def __init__(self, harness: prof.Profile, model: prof.Profile, build_id: Optional[str] = None):
        self.harness = harness
        self.model = model
        self.build_id = build_id
        self.outcomes = prof.outcome_rules(harness)
        # What the harness changes in our calls when it sends them back (C-03); `exact` or a
        # list of turnlib.ECHO_REWRITES. An unknown name fails here, not as a silent pass.
        rewrites = harness.get_path("echo.tool_input", "exact") if harness else "exact"
        rewrites = (
            [] if rewrites == "exact" else [rewrites] if isinstance(rewrites, str) else rewrites
        )
        unknown = [r for r in rewrites if r not in ECHO_REWRITES]
        if not isinstance(rewrites, list) or unknown:
            raise ValueError(
                f"harness profile echo.tool_input: unknown rewrite {unknown or rewrites}"
            )
        self.echo_rewrites = tuple(rewrites)
        # Rejections the harness recovers from by dropping a capability (C-16).
        retries = harness.get_path("errors.capability_retries") if harness else None
        self.capability_retries = tuple(
            (str(e.get("field") or "?"), e["wording"]) for e in retries or ()
            if isinstance(e, dict) and isinstance(e.get("wording"), str) and not prof.is_todo(e["wording"])
        )  # fmt: skip
        # The patterns a harness reads an over-length error by (C-10, `shape: any_error`).
        self.overlength_patterns = overlength_patterns(
            harness.get_path("errors.overlength") if harness else None
        )
        self.markers = prof.marker_patterns(model)
        self.degenerate = prof.degenerate_patterns(model)
        self.counts: Counter = Counter()
        self.examples: Dict[str, list] = defaultdict(list)
        self.by_rule: Dict[str, Counter] = defaultdict(Counter)
        self.call_classes: Counter = Counter()
        self.outcome_classes: Counter = Counter()
        self.session_events: Dict[str, set] = defaultdict(set)  # check key -> sessions with it
        self.sessions: set = set()
        self.requests: Dict[str, dict] = {}
        self.auxiliary: set = set()
        self.tools: Dict[str, tuple] = {}
        self.declarations: Dict[str, tuple] = {}  # tools_key -> turnlib.tool_declarations (USG-08)
        self.echo_first: Dict[tuple, tuple] = {}
        self.result_first: Dict[tuple, tuple] = {}
        self.bodies: Dict[tuple, list] = defaultdict(list)
        self.turns: List[dict] = []
        self.unknown_fields: Counter = Counter()
        self.protocols: Counter = Counter()
        self.native = prof.native_card(model)
        # Prompt tokens the model's usage leaves out, per protocol (USG-07).
        self.prompt_offsets = {p: prof.prompt_offset(model, p) for p in PROTOCOL_OF.values()}
        self.count_requests: Dict[str, str] = {}  # count_tokens trace id -> counted-body digest
        self.count_results: Dict[str, list] = defaultdict(list)  # digest -> counts answered
        self.raw: Dict[str, list] = {}  # id key -> every raw reading under it (join_raw)
        self.raw_given = False
        self.boundary_requests: set = set()  # --requests: join-only context, never lost (I-01)

    # ------------------------------------------------------------------ helpers

    def note(self, key: str, locator: dict, session: Optional[str] = None, **detail) -> None:
        self.counts[key] += 1
        if session is not None:
            self.session_events[key].add(session)
        examples = self.examples[key]
        example = {**locator, **detail}
        if len(examples) < EXAMPLES or _example_key(example) < _example_key(examples[-1]):
            self.examples[key] = _first_examples(examples + [example])

    # ------------------------------------------------------------------ raw pass

    def add_raw(self, paths: List[str], workers: int = 1) -> None:
        """Read every raw record in the model's own format, keyed by its engine ids."""
        shards = expand_shards(paths, "raw")
        self.raw_given = bool(paths)
        self.counts["den:raw_shards"] += len(shards)
        if not shards or self.native is None:
            return
        patterns = [rx.pattern for rx in self.degenerate]
        id_runs = prof.degenerate_id_runs(self.model) if self.model else []
        tasks = [(path, self.native, patterns, id_runs) for path in shards]
        if workers > 1 and len(shards) > 1:
            with multiprocessing.get_context("fork").Pool(workers) as pool:
                parts = list(pool.imap_unordered(_raw_task, tasks))
        else:
            parts = [_raw_task(task) for task in tasks]
        # An engine id names one generation only within its server process, so with several
        # generation workers it names a record on each. Every record under an id is kept; each
        # reply decides which one, if any, is its own (join_raw). Ids that name several are counted.
        for part, counts in parts:
            self.counts.update(counts)
            for key, readings in part.items():
                self.raw.setdefault(key, []).extend(readings)
        for key, readings in self.raw.items():
            if len(readings) > 1:
                self.counts["I-01:raw_ambiguous_id:" + key.split(":", 1)[0]] += 1

    # ------------------------------------------------------------------ pass 1

    def add_requests(self, paths: List[str], workers: int = 1) -> None:
        shards = expand_shards(paths, "requests")
        if workers > 1 and len(shards) > 1:
            tasks = [(i, s, self.harness, self.model, self.build_id) for i, s in enumerate(shards)]
            with multiprocessing.get_context("fork").Pool(workers) as pool:
                for part in pool.imap_unordered(_requests_task, tasks):
                    self._merge_requests(part)
        else:
            for index, path in enumerate(shards):
                self._scan_request_shard(index, path)

    def _scan_request_shard(self, index: int, path: str) -> None:
        for line_no, raw in iter_lines(path):
            rec = load_json_line(raw)
            if rec is None:
                self.counts["inventory:unparsable_request_lines"] += 1
                continue
            if rec.get("event", "request") != "request":
                continue
            self._add_request(rec, {"file": path, "line": line_no, "trace_id": rec.get("trace_id")},
                              (index, line_no))  # fmt: skip

    def _request_state(self) -> dict:
        return {"counts": self.counts, "requests": self.requests, "auxiliary": self.auxiliary,
                "tools": self.tools, "declarations": self.declarations,
                "echo_first": self.echo_first, "result_first": self.result_first,
                "bodies": dict(self.bodies), "unknown_fields": self.unknown_fields,
                "protocols": self.protocols, "count_requests": self.count_requests}  # fmt: skip

    def _merge_requests(self, part: dict) -> None:
        self.counts.update(part["counts"])
        for trace_id, summary in part["requests"].items():
            if trace_id in self.requests:
                self.counts["inventory:duplicate_trace_id"] += 1
            self.requests[trace_id] = summary
        self.auxiliary |= part["auxiliary"]
        for key, value in part["tools"].items():
            self.tools.setdefault(key, value)
        for key, value in part["declarations"].items():
            self.declarations.setdefault(key, value)
        for mine, theirs in ((self.echo_first, part["echo_first"]),
                             (self.result_first, part["result_first"])):  # fmt: skip
            for key, value in theirs.items():
                if key not in mine or value[0] < mine[key][0]:
                    mine[key] = value
        for key, value in part["bodies"].items():
            self.bodies[key].extend(value)
        self.unknown_fields.update(part["unknown_fields"])
        self.protocols.update(part["protocols"])
        self.count_requests.update(part["count_requests"])

    def _add_request(self, rec: dict, locator: dict, position: tuple = (0, 0)) -> None:
        route = str(rec.get("route") or "").split("?", 1)[0]
        body = rec.get("body")
        self.counts["inventory:requests"] += 1
        if route == COUNT_TOKENS and isinstance(body, dict):
            # Not a turn: its answer is held against the input a turn with the same body billed.
            self.count_requests[rec.get("trace_id")] = usage_mod.counted_digest(body)
            self.counts["inventory:count_tokens_requests"] += 1
            return
        if any(name in header_map(rec) for name in PROBE_HEADERS):
            self.counts["inventory:operator_probes"] += 1
            self.auxiliary.add(rec.get("trace_id"))
            return
        adapter = ADAPTERS.get(route)
        if adapter is None:
            self.counts["inventory:auxiliary_route:" + route] += 1
            self.auxiliary.add(rec.get("trace_id"))
            return
        if not isinstance(body, dict):
            self.counts["inventory:request_without_body"] += 1
            return
        cell_protocol = prof.known(self.harness.get("protocol")) if self.harness else None
        if cell_protocol and adapter.PROTOCOL != cell_protocol:
            # Traffic of another protocol in the window (probes, other clients) is not this
            # cell's: it is counted and excluded, never judged against this harness.
            self.counts["inventory:foreign_protocol:" + adapter.PROTOCOL] += 1
            self.auxiliary.add(rec.get("trace_id"))
            return
        headers = header_map(rec)
        identity = prof.extract_identity(self.harness, headers, body, rec)
        view = adapter.parse_request(body, headers)
        # Arrival time first; the shard index and line make the order total and independent of
        # which worker read the shard.
        order = (str(rec.get("recorded_at") or ""),) + tuple(position)
        thread = identity.get("thread")
        if view.tools_key not in self.tools:
            self.tools[view.tools_key] = (view.tools, view.nested)
            self.declarations[view.tools_key] = view.declarations
        for key in view.unknown_fields:
            self.unknown_fields[key] += 1
        self.protocols[view.protocol] += 1
        summary = {
            "locator": locator,
            "protocol": view.protocol,
            "route": route,
            "identity": identity,
            "thread": thread,
            "order": order,
            "stream": view.stream,
            "tools_key": view.tools_key,
            "tool_choice": view.tool_choice,
            "parallel": view.parallel,
            "include_usage": view.include_usage,
            "choices": view.choices,
            "counted": usage_mod.counted_digest(body)
            if adapter.PROTOCOL == "anthropic-messages"
            else None,
            # A prompt the request ends itself carries no model prompt offset (USG-07).
            "own_prompt": usage_mod.renders_own_prompt(adapter.PROTOCOL, body),
            # Server-side clearing the client asked for: the rendered prompt may shrink (USG-08).
            "context_management": context_management_clears(body.get("context_management")),
            "reasoning": dict(view.reasoning or {}),
            "build_id": rec.get("build_id"),
            "history": view.history_messages,
            "last_digest": view.last_message_digest,
            # The last assistant message's position and the message before it: the request
            # whose history ends there is the one this request continues (C-03, compaction).
            "continues": view.assistant_turns[-1][:2] if view.assistant_turns else None,
            "side_request": adapter.PROTOCOL == "anthropic-messages" and side_request(body),
            "body_sha": rec.get("body_sha256") or _digest(body),
            "reask": _reask_key(body) if adapter.PROTOCOL in REASK_PROTOCOLS else None,
        }
        trace_id = rec.get("trace_id")
        if trace_id in self.requests:
            self.counts["inventory:duplicate_trace_id"] += 1
        self.requests[trace_id] = summary
        if thread:
            self.bodies[(thread, summary["body_sha"])].append((order, trace_id))
            for item_id, echo in view.echo.items():
                key = (thread, item_id)
                known = self.echo_first.get(key)
                if known is not None and known[0] <= order:
                    continue  # histories replay every item each turn: digest the first only
                if echo["kind"] == "call":
                    # Its value, its value under the declared rewrites (echo.tool_input), its bytes.
                    self.echo_first[key] = (order, "call", echo.get("name")) + self._call_digests(
                        echo.get("payload"), echo.get("call_kind"),
                        view.tools.get(echo.get("name") or ""), False)  # fmt: skip
                    continue
                # A reasoning carrier is compared by its value where it has one.
                digest = (sha256_text(echo.get("text") or "") if echo["kind"] == "text" else
                          echo.get("carrier_digest") or echo.get("carrier"))  # fmt: skip
                self.echo_first[key] = (order, echo["kind"], echo.get("name"), digest, None)
            for position, previous, blocks, call_ids in view.assistant_turns:
                # Keyed by the message it answered as well: a history that was rewritten does not
                # match, and only the first appearance is kept, as for every echoed item.
                key = (thread, "pos", position, previous)
                known = self.echo_first.get(key)
                if known is None or order < known[0]:
                    self.echo_first[key] = (order, "texts", None, _echoable(blocks))
                if call_ids:
                    # A turn with calls is found again by its call ids: parallel sub-agents of
                    # one session can share a position and a previous message, and a compacted
                    # history moves the turn, but the ids name it.
                    key = (thread, "calls", tuple(sorted(call_ids)))
                    known = self.echo_first.get(key)
                    if known is None or order < known[0]:
                        self.echo_first[key] = (order, "texts", rec.get("trace_id"), _echoable(blocks),
                                                view.history_messages)  # fmt: skip
            for call_id, text, is_error in view.tool_results:
                key = (thread, call_id)
                if key in self.result_first and self.result_first[key][0] <= order:
                    continue
                cls = self.outcomes.classify(text, is_error) if self.outcomes else None
                self.result_first[key] = (order, cls)

    # ------------------------------------------------------------------ pass 2

    def add_responses(self, paths: List[str], workers: int = 1) -> None:
        global _SHARED
        shards = expand_shards(paths, "responses")
        if workers > 1 and len(shards) > 1:
            _SHARED = self  # read by forked workers; copy-on-write, never sent through a pipe
            try:
                with multiprocessing.get_context("fork").Pool(workers) as pool:
                    for part in pool.imap_unordered(_responses_task, shards):
                        self._merge_replies(part)
            finally:
                _SHARED = None
        else:
            for path in shards:
                self._scan_response_shard(path)

    def add_boundary_responses(self, paths: List[str]) -> None:
        """Responses from the next hours that answer requests of this population (I-01)."""
        answered = {t["trace_id"] for t in self.turns}
        wanted = (set(self.requests) | set(self.count_requests)) - self.boundary_requests - answered
        for path in expand_shards(paths, "responses"):
            for line_no, raw in iter_lines(path):
                rec = load_json_line(raw)
                if not isinstance(rec, dict) or rec.get("event", "response") != "response":
                    continue
                if rec.get("trace_id") not in wanted:
                    self.counts["inventory:boundary_responses_ignored"] += 1
                    continue
                wanted.discard(rec.get("trace_id"))
                self.counts["inventory:boundary_responses_joined"] += 1
                self._add_response(
                    rec, {"file": path, "line": line_no, "trace_id": rec.get("trace_id")}
                )

    def _scan_response_shard(self, path: str) -> None:
        for line_no, raw in iter_lines(path):
            rec = load_json_line(raw)
            if rec is None:
                self.counts["inventory:unparsable_response_lines"] += 1
                continue
            if rec.get("event", "response") != "response":
                continue
            self._add_response(
                rec, {"file": path, "line": line_no, "trace_id": rec.get("trace_id")}
            )

    def _reset_replies(self) -> None:
        self.counts = Counter()
        self.examples = defaultdict(list)
        self.by_rule = defaultdict(Counter)
        self.call_classes = Counter()
        self.session_events = defaultdict(set)
        self.sessions = set()
        self.turns = []
        self.count_results = defaultdict(list)

    def _reply_state(self) -> dict:
        return {"counts": self.counts, "examples": dict(self.examples),
                "by_rule": {k: dict(v) for k, v in self.by_rule.items()},
                "call_classes": self.call_classes, "session_events": dict(self.session_events),
                "sessions": self.sessions, "turns": self.turns,
                "count_results": dict(self.count_results)}  # fmt: skip

    def _merge_replies(self, part: dict) -> None:
        self.counts.update(part["counts"])
        for key, examples in part["examples"].items():
            self.examples[key] = _first_examples(self.examples[key] + list(examples))
        for check, rules in part["by_rule"].items():
            self.by_rule[check].update(rules)
        self.call_classes.update(part["call_classes"])
        for key, sessions in part["session_events"].items():
            self.session_events[key] |= sessions
        self.sessions |= part["sessions"]
        self.turns.extend(part["turns"])
        for digest, values in part["count_results"].items():
            self.count_results[digest].extend(values)

    def _request_view(self, summary: dict) -> RequestView:
        tools, nested = self.tools.get(summary["tools_key"], ({}, {}))
        view = RequestView(protocol=summary["protocol"], route=summary["route"])
        view.stream = summary["stream"]
        view.tools, view.nested = tools, nested
        view.tool_choice = summary["tool_choice"]
        view.parallel = summary["parallel"]
        view.include_usage = summary["include_usage"]
        view.choices = summary.get("choices", 1)
        view.reasoning = summary.get("reasoning") or {}
        return view

    def _add_response(self, rec: dict, loc: dict) -> None:
        self.counts["inventory:responses"] += 1
        summary = self.requests.get(rec.get("trace_id"))
        if summary is None and rec.get("trace_id") in self.count_requests:
            body = (rec.get("response") or {}).get("body")
            value = body.get("input_tokens") if isinstance(body, dict) else None
            if rec.get("http_status") in (None, 200) and isinstance(value, int):
                self.count_results[self.count_requests[rec["trace_id"]]].append(value)
            else:
                self.counts["C-14:count_tokens_unanswered"] += 1
                error = body.get("error") if isinstance(body, dict) else None
                error = error if isinstance(error, dict) else {}
                message = error.get("message") or (
                    body.get("message") if isinstance(body, dict) else ""
                )
                self._rejection(rec.get("http_status"), str(message or ""), str(error.get("type") or ""),
                                False, loc, None)  # fmt: skip
            return
        if summary is None:
            if rec.get("trace_id") in self.auxiliary:
                self.counts["inventory:auxiliary_responses"] += 1
            else:
                self.note("I-01:response_without_request", loc)
            return
        adapter = ADAPTERS[summary["route"]]
        request = self._request_view(summary)
        reply = adapter.parse_reply(rec, request)
        session = summary["identity"].get("session") or summary["thread"]
        if session:
            self.sessions.add(session)
        self.counts["den:replies"] += 1
        self.counts["den:replies:" + summary["protocol"]] += 1
        build_id = rec.get("build_id") or summary.get("build_id")
        if not build_id:
            self.counts["I-05:missing_build_id"] += 1
        else:
            # Which builds served, whether or not a revision was declared to check them against.
            self.counts["I-05:with_build_id"] += 1
            self.counts["I-05:build_id:" + str(build_id)[:120]] += 1
            if self.build_id and build_id != self.build_id:
                self.note("I-05:other_build_id", loc)
        self._timing(rec, reply)
        # A client that stopped reading before the terminal event left a stream that is
        # incomplete by construction: its wire, call and content rules are not judged, and H-03
        # attributes the cut. A client that closed after the terminal event cut nothing.
        client_cut = reply.status in CLIENT_SIDE and reply.terminal.get("kind") == "none"
        delivered = reply.delivered()
        if reply.status in CLIENT_SIDE and not client_cut:
            self.counts["H-03:closed_after_terminal"] += 1
        if client_cut:
            self.note("H-03:" + str(reply.status), loc, session)
            self.note("H-03", loc, session, status=str(reply.status))
            self.counts["den:client_cut_replies"] += 1
            self._cut_before_reply(rec, loc, session)
        elif reply.status in UPSTREAM_SIDE:
            # Not the server's reply, or not all of it: R-06 counts it and nothing else judges it.
            self.note("R-06", loc, session, http_status=reply.http_status or 0)
            self.counts["den:upstream_cut_replies"] += 1
        elif reply.status in CAPTURE_SIDE:
            self.counts["inventory:pending_at_capture_end"] += 1
        else:
            self._wire(adapter, reply, loc, session, request)
            self._sdk(summary["protocol"], rec, reply, loc, session)
            raw_degenerate = self._raw_view(rec, reply, delivered, request, loc, session,
                                            summary["order"][0])  # fmt: skip
            self._usage(summary, rec, reply, delivered, request, loc, session)
            self._calls(delivered, request, reply, loc, session)
            self._content(delivered, reply, loc, session, raw_degenerate)
            self._overlength(reply, loc, session)
            self._rejected(reply, loc, session)
        self._runtime(reply, loc, session)
        if summary["thread"] is None:
            self.counts["I-09:turn_without_thread"] += 1
        self.turns.append({
            "trace_id": rec.get("trace_id"),
            "loc": loc,
            "thread": summary["thread"],
            "session": session,
            "agent": summary["identity"].get("agent"),
            "order": summary["order"],
            "terminal": reply.terminal.get("kind"),
            "http_status": reply.http_status,
            "overlength": self._is_overlength(reply),
            "failed": reply.terminal.get("kind") in ("failed", "error", "none")
            or (isinstance(reply.http_status, int) and reply.http_status >= 400),
            "calls": [(i.call_id, i.name) + self._call_digests(i.payload, i.call_kind,
                                                                request.tools.get(i.name or ""), True)
                      for i in delivered if i.kind == "call" and i.call_id],
            "protocol": summary["protocol"],
            "texts": [(i.id, sha256_text(i.text or "")) for i in delivered
                      if i.kind == "text" and i.id and summary["protocol"] == "responses"],
            "reasonings": [(i.id, i.carrier_digest or i.carrier) for i in delivered
                           if i.kind == "reasoning" and i.id and summary["protocol"] == "responses"],
            "text_blocks": (_echoable(text_blocks(delivered))
                            if summary["protocol"] != "responses" else ()),
            "side_request": summary.get("side_request"),
            "body_sha": summary["body_sha"],
            "reask": summary.get("reask"),
            "history": summary["history"],
            "last_digest": summary.get("last_digest"),
            "counted": summary.get("counted"),
            "billed_input": usage_mod.input_tokens(summary["protocol"], reply.usage),
        })  # fmt: skip

    def _timing(self, rec: dict, reply) -> None:
        timing = rec.get("timing") if isinstance(rec.get("timing"), dict) else {}
        first = timing.get("first_event_s", timing.get("first_byte_s"))
        if isinstance(first, (int, float)):
            reply.first_event_s = float(first)
        chunks = timing.get("chunks")
        if isinstance(chunks, list) and chunks:
            times = [
                c[0] for c in chunks if isinstance(c, list) and c and isinstance(c[0], (int, float))
            ]
            start = float(timing.get("headers_s") or 0.0)
            gaps = [b - a for a, b in zip([start] + times, times)]
            reply.max_gap_s = max(gaps) if gaps else None
            reply.timing_known = True

    def _body_rules(self, request, reply) -> List[tuple]:
        """Protocol-independent body rules (protocols.md section 4: WIRE-01 .. WIRE-04)."""
        ok_status = not isinstance(reply.http_status, int) or reply.http_status < 400
        if reply.kind == "text":
            return [("C-01", "WIRE-01")]  # neither SSE nor JSON
        if reply.kind == "none":
            return [("C-04", "WIRE-02")]  # no body at all
        if ok_status and request.stream and reply.kind == "json":
            return [("C-01", "WIRE-03")]  # stream requested, JSON answered
        if not request.stream and reply.kind == "sse":
            return [("C-01", "WIRE-04")]  # JSON requested, stream answered
        return []

    def _wire(self, adapter, reply, loc, session, request=None) -> None:
        mapping = getattr(adapter, "RULE_CHECK", {})
        seen = set()
        if request is not None:
            for check, rule in self._body_rules(request, reply):
                self.by_rule[check][rule] += 1
                self.note(check, loc, session, rule=rule)
        for violation in reply.grammar:
            check = mapping.get(violation.rule, "C-01")
            self.by_rule[check][violation.rule] += 1
            if (check, violation.rule) in seen:
                continue
            seen.add((check, violation.rule))
            self.note(
                check,
                loc,
                session,
                rule=violation.rule,
                **{k: v for k, v in violation.detail.items() if isinstance(v, (int, str, bool))},
            )
        unchecked = sorted(reply.extras.get("unchecked") or ())
        if unchecked:
            self.by_rule["C-11"]["WIRE-05"] += 1
            self.note("C-11", loc, session, elements=",".join(unchecked)[:200])
            for element in unchecked:
                self.counts["C-11:element:" + element] += 1
        if reply.kind == "sse":
            self.counts["den:streams"] += 1
            if reply.final_items is None:
                self.counts["den:no_streamed_snapshot"] += 1

    def _sdk(self, protocol, rec, reply, loc, session) -> None:
        """C-12: the official SDK's reading of the same bytes agrees with the adapter's."""
        if sdk_diff.unavailable(protocol):
            self.counts["C-12:sdk_unavailable:" + protocol] += 1
            return
        found = sdk_diff.compare(protocol, rec, reply)
        if found is None:
            self.counts["C-12:not_read"] += 1
            return
        self.counts["den:sdk_read"] += 1
        for violation in found:
            self.by_rule["C-12"][violation.rule] += 1
            what = violation.detail.get("fields") or violation.detail.get("error")
            self.counts[f"C-12:{violation.rule}:{what}"] += 1
        if found:
            first = found[0]
            self.note("C-12", loc, session, rules=",".join(v.rule for v in found),
                      **{k: v for k, v in first.detail.items() if isinstance(v, (int, str, bool))})  # fmt: skip

    def _raw_view(self, rec, reply, delivered, request, loc, session,
                  arrival=None) -> Optional[bool]:  # fmt: skip
        """C-13: what was delivered is what the model wrote (its raw record, read natively).

        ``arrival`` is the request record's ``recorded_at``: with the response's completion, it
        bounds when the reply's generation ran, which tells apart the raw records one engine id
        names on several generation workers (join_raw).

        Returns M-02's reading of the raw text: whether it holds a runaway run, or None when the
        raw text cannot show the declared tokens (no record, or the server dropped special tokens).
        """
        if not self.raw_given:
            return None
        if isinstance(reply.http_status, int) and reply.http_status >= 400:
            self.counts["C-13:http_error"] += 1  # no generation delivered to hold against one
            return None
        if self.native is None:
            self.counts["C-13:no_native_reader"] += 1
            return None
        # The keys a reply joins by, in this order (join_raw): the disagg_request_id the trace
        # recorded, the one a Responses name carries, then engine ids, which repeat across
        # generation workers where a disagg_request_id does not.
        keys = []
        if rec.get("disagg_request_id") not in (None, ""):
            keys.append("d:" + str(rec["disagg_request_id"]))
        body = (rec.get("response") or {}).get("body")
        match = ENGINE_ID.search(body if isinstance(body, str) else json.dumps(body or {})[:400])
        if match and match.group(3):
            # A Responses reply named resp_<id>_<hex> was named by a disaggregated server after its
            # disagg_request_id; no server names one after an engine id, so the number is read
            # only so: as an engine id it could join another request's record by coincidence.
            keys.append("d:" + match.group(3))
        # The engine id the reply's header named (record_requests.py ENGINE_ID_HEADER).
        if rec.get("engine_request_id") not in (None, ""):
            keys.append("r:" + str(rec["engine_request_id"]))
        if match and not match.group(3):
            # The id in a Chat or Messages reply's name is the generation worker's engine id. A
            # server that names replies after the orchestrator's disagg_request_id instead is
            # read last, so that reading changes no join an engine id makes.
            number = match.group(1) or match.group(2)
            keys += ["r:" + number, "d:" + number]
        completed = reply.terminal.get("kind") == "completed"
        reported = usage_mod.output_tokens(request.protocol, reply.usage) if completed else None
        # A request for several choices has one raw record per choice under one engine id, and
        # neither the window nor the reply's total output count tells them apart.
        window = reply_window(arrival, rec) if request.choices == 1 else None
        raw, how = join_raw(self.raw, keys, window, reported)
        self.counts["den:raw_candidates"] += 1
        if raw is None:
            self.counts["C-13:unjoined"] += 1
            if how == "ambiguous":
                self.counts["C-13:unjoined_ambiguous_id"] += 1
            self.counts["M-02:raw_unobservable"] += 1
            return None
        if how == "resolved":
            self.counts["C-13:resolved_ambiguous_id"] += 1
        self.counts["den:raw_joined"] += 1
        raw_degenerate = None
        if raw.get("specials_visible"):
            raw_degenerate = bool(raw.get("degenerate"))
        else:
            self.counts["M-02:raw_unobservable"] += 1
        # A generation the server stopped (raw_stopped) holds what the model wrote until then:
        # what was delivered can only be a prefix of it, whatever the reply's terminal says.
        stopped = raw.get("stopped")
        if stopped:
            self.counts["C-13:stopped_raw:" + stopped] += 1
        if raw.get("without_thinking") is not None and request.protocol == "anthropic-messages" \
                and (request.reasoning or {}).get("thinking") in (None, "disabled"):  # fmt: skip
            # The request turned thinking off, so the prompt opened in the reply's content.
            raw = dict(raw["without_thinking"], output_tokens=raw.get("output_tokens"))
            self.counts["C-13:read_without_thinking"] += 1
        found, notes = native_mod.compare(self.native, raw, delivered, completed and not stopped,
                                          request.parallel == "single")  # fmt: skip
        # USG-06 on a completed reply. A record joined by resolving a shared engine id already
        # matched the reply's count where both were known, so it is not compared again.
        if how == "unique" and raw.get("output_tokens") is not None and reported is not None:
            self.counts["den:usage_vs_engine_output"] += 1
            if reported != raw["output_tokens"]:
                self._usage_found(
                    [Violation("USG-06", {"why": "output_vs_engine_count"})], loc, session
                )
        for item in notes:
            self.counts["C-13:" + item] += 1
        anomalies = [n.split(":", 1)[1] for n in notes if n.startswith("anomaly:")]
        if anomalies:
            self.note("M-07", loc, session, anomalies=",".join(anomalies)[:200])
            # The model broke its own format: the reply was joined but not compared.
            self.counts["C-13:not_judged"] += 1
        else:
            self.counts["den:raw_compared"] += 1
        for violation in found:
            self.by_rule["C-13"][violation.rule] += 1
            self.counts[f"C-13:{violation.rule}:{violation.detail.get('why')}"] += 1
        if found:
            self.note("C-13", loc, session, rules=",".join(v.rule for v in found),
                      why=str(found[0].detail.get("why")))  # fmt: skip
        return raw_degenerate

    def _usage(self, summary, rec, reply, delivered, request, loc, session) -> None:
        """C-14: usage present, typed and consistent (USG-01 .. 04), and true to the trace (USG-07)."""
        protocol = summary["protocol"]
        found = usage_mod.rules(protocol, reply, delivered, request.include_usage)
        prompt = rec.get("prompt_tokens") if isinstance(rec.get("prompt_tokens"), dict) else {}
        billed = usage_mod.input_tokens(protocol, reply.usage)
        if isinstance(prompt.get("count"), int) and billed is not None:
            self.counts["den:usage_vs_trace_prompt"] += 1
            offset = 0 if summary.get("own_prompt") else self.prompt_offsets.get(protocol, 0)
            if billed != prompt["count"] - offset:
                found.append(Violation("USG-07", {"why": "input_vs_server_prompt_count"}))
        self._usage_found(found, loc, session)

    def _usage_found(self, found, loc, session) -> None:
        for violation in found:
            self.by_rule["C-14"][violation.rule] += 1
            self.counts[f"C-14:{violation.rule}:{violation.detail.get('why')}"] += 1
        if found:
            self.note("C-14", loc, session, rules=",".join(v.rule for v in found),
                      why=str(found[0].detail.get("why")))  # fmt: skip

    def _calls(self, delivered, request, reply, loc, session) -> None:
        calls = [i for i in delivered if i.kind == "call"]
        self.counts["den:calls"] += len(calls)
        ids = [i.call_id for i in calls]
        if any(not i for i in ids) or len(ids) != len(set(ids)):
            self.note("C-05", loc, session)
        if request.parallel == "single":
            self.counts["den:single_call_replies"] += 1
            if len(calls) > 1:
                self.note("C-07", loc, session, calls=len(calls))
        mode = request.tool_choice.get("mode")
        if mode in ("none", "required", "named"):
            self.counts["den:tool_choice_replies"] += 1
        completed = reply.terminal.get("kind") == "completed"
        if mode == "none" and calls:
            self.note("C-08", loc, session, mode=mode)
        elif mode == "required" and completed and not calls:
            self.note("C-08", loc, session, mode=mode)
        elif (
            mode == "named"
            and completed
            and (not calls or any(c.name != request.tool_choice.get("name") for c in calls))
        ):
            self.note("C-08", loc, session, mode=mode)
        tools_policy = self.harness.get_path("tools") if self.harness else None
        tolerate = (tools_policy or {}).get("tolerated_payload") or ()
        tolerate = tuple(tolerate) if isinstance(tolerate, (list, tuple)) else ()
        for call in calls:
            result = classify_call(call, request, tolerate)
            cls = result["class"]
            self.call_classes[cls] += 1
            if cls in SERVING_SHAPE:
                self.note("C-06", loc, session, cls=cls, **safe_name(call.name))
            elif cls in ("undeclared", "nested_tool_called_directly"):
                self.note("M-01:undeclared", loc, session, cls=cls, **safe_name(call.name))
            elif cls == "payload_invalid":
                self.note("M-01:payload_invalid", loc, session, **safe_name(call.name),
                          **(result.get("detail") or {}))  # fmt: skip
            elif cls == "payload_tolerated":
                # Invalid only in a way the harness tolerates (its profile's
                # tools.tolerated_payload): the call runs, so it is counted, not rated.
                self.counts["M-01:payload_tolerated"] += 1
            elif cls == "server_tool_called":
                self.note("M-01:server_tool_called", loc, session, **safe_name(call.name))

    def _content(self, delivered, reply, loc, session, raw_degenerate=None) -> None:
        texts = [i.text or "" for i in delivered if i.kind == "text"]
        reasoning = [i.text or "" for i in delivered if i.kind == "reasoning"]
        payloads = [i.payload or "" for i in delivered if i.kind == "call"]
        joined = "\n".join(texts + reasoning)
        shown = any(rx.search(joined) for rx in self.degenerate)
        found = shown or bool(raw_degenerate)
        # With raw output, M-02 counts only replies that could show the declared tokens: a raw
        # record that kept special tokens, or a run that was found anyway.
        if self.raw_given and (raw_degenerate is not None or found):
            self.counts["den:m02_observable"] += 1
        if found:
            # The model wrote the run (M-02), whether or not the client saw it.
            self.note("M-02", loc, session)
            # R-04: the run reached the client, in what was delivered or in a generation the
            # reply delivered as an answer (decoding drops the run's special tokens). A run only
            # the raw record shows, in a reply that ended without an answer, was stopped.
            if shown or delivered_as_answer(reply):
                self.note("R-04", loc, session)
            else:
                self.note("R-04:stopped", loc, session)
        bare = [
            m for m, quoted in marker_hits("\n".join(texts + payloads), self.markers) if not quoted
        ]
        if bare:
            self.note("M-03", loc, session, markers=len(bare))
        think = [m for m, quoted in marker_hits("\n".join(texts), self.markers)
                 if not quoted and "think" in m]  # fmt: skip
        if think:
            self.note("C-09", loc, session)
        text = "\n".join(reasoning)
        if len(text) >= REPETITIVE_MIN_CHARS:
            raw = text.encode("utf-8", "surrogatepass")
            if len(zlib.compress(raw, 6)) / len(raw) < REPETITIVE_MAX_ZLIB_RATIO:
                self.note("M-05", loc, session)
        if (reply.terminal.get("kind") == "completed" and not any(t.strip() for t in texts)
                and not payloads):  # fmt: skip
            self.note("M-06", loc, session)

    def _is_overlength(self, reply) -> bool:
        error = reply.terminal.get("error") or {}
        words = " ".join(str(x) for x in (error.get("code"), error.get("type"),
                                          reply.terminal.get("message")) if x).lower()  # fmt: skip
        return any(w in words for w in OVERLENGTH_WORDS)

    def _rejected(self, reply, loc, session) -> None:
        """C-16: the server accepted what the harness sent.

        A 4xx other than over-length (C-10) or capacity fails, unless the harness profile
        declares it a capability the harness drops and retries without (`errors.capability_retries`,
        matched by wording); a declared one is flagged, so the round trip it costs stays visible.
        """
        error = reply.terminal.get("error") or {}
        self._rejection(reply.http_status, str(reply.terminal.get("message") or ""),
                        str(error.get("type") or ""), self._is_overlength(reply), loc, session)  # fmt: skip

    def _rejection(self, status, message, error_type, overlength, loc, session) -> None:
        if not isinstance(status, int) or not 400 <= status < 500 or status in CAPACITY_STATUSES:
            return
        if overlength:
            return
        for field, wording in self.capability_retries:
            if wording_matches(wording, message):
                self.note("C-16:declared", loc, session, field=field[:80], http_status=status)
                self.counts["C-16:declared_field:" + field[:80]] += 1
                return
        self.note("C-16", loc, session, http_status=status, error_type=error_type)

    def _cut_before_reply(self, rec, loc, session) -> None:
        """R-01: count a client that gave up before the first byte after the cutoff.

        Having waited at least the harness's first-byte cutoff, it saw that cutoff pass: the
        first event came later, if at all.
        """
        timing = rec.get("timing") if isinstance(rec.get("timing"), dict) else {}
        if not timing.get("before_reply") and timing.get("first_byte_s") is not None:
            return
        waited = timing.get("total_s")
        if not isinstance(waited, (int, float)):
            return
        self.counts["den:cuts_before_reply"] += 1
        cut, exact = prof.client_cut(self.harness, "timeouts.first_event_s")
        if cut is not None and waited >= cut:
            if exact:
                self.note("R-01", loc, session, waited_s=round(waited, 3), before_reply=True)
            else:
                # The client may have timed out, but its cut is only known to be no shorter.
                self.counts["R-01:beyond_known_bound"] += 1

    def _overlength(self, reply, loc, session) -> None:
        if not self._is_overlength(reply):
            return
        self.counts["den:overlength"] += 1
        expected = self.harness.get_path("errors.overlength")
        if not isinstance(expected, dict) or any(prof.is_todo(v) for v in expected.values()):
            self.note("C-10:profile_unknown", loc, session)
            return
        shape = expected.get("shape")
        error = reply.terminal.get("error") or {}
        if shape == "in_stream":
            ok = (reply.kind == "sse" and reply.terminal.get("kind") == "failed"
                  and error.get("code") == expected.get("code"))  # fmt: skip
        elif shape == "http_400":
            message = (reply.terminal.get("message") or "").lower()
            ok = (reply.http_status == 400
                  and (not expected.get("error_type") or error.get("type") == expected["error_type"])
                  and (not expected.get("wording") or expected["wording"].lower() in message))  # fmt: skip
        elif shape == "any_error":
            ok = overlength_read_by_text(
                self.overlength_patterns, reply.http_status, reply.terminal
            )
        else:
            ok = False
        if not ok:
            self.note("C-10", loc, session, http_status=reply.http_status or 0,
                      terminal=reply.terminal.get("kind"))  # fmt: skip

    def _runtime(self, reply, loc, session) -> None:
        first_cut, first_exact = prof.client_cut(self.harness, "timeouts.first_event_s")
        idle_cut, idle_exact = prof.client_cut(self.harness, "timeouts.idle_s")
        if reply.kind != "sse":
            return
        if reply.first_event_s is None:
            self.counts["R-01:no_timing"] += 1
        elif first_cut is not None and first_exact and reply.first_event_s > first_cut:
            self.note("R-01", loc, session, first_event_s=round(reply.first_event_s, 3))
        elif first_cut is not None and not first_exact and reply.first_event_s >= first_cut:
            # Waited as long as silences are known to survive: whether the client cuts it is unknown.
            self.counts["R-01:beyond_known_bound"] += 1
        if not reply.timing_known:
            self.counts["R-02:no_timing"] += 1
        elif idle_cut is not None and reply.max_gap_s is not None:
            if idle_exact and reply.max_gap_s > idle_cut:
                self.note("R-02", loc, session, max_gap_s=round(reply.max_gap_s, 3))
            elif not idle_exact and reply.max_gap_s >= idle_cut:
                self.counts["R-02:beyond_known_bound"] += 1

    # ------------------------------------------------------------------ pass 3

    def finish(self) -> None:
        # I-01: a request with no response record at all. The capture lost it, or its answer lies
        # outside the window; either way the evidence is partial (inventory is BLOCKED).
        answered = {t["trace_id"] for t in self.turns}
        for trace_id, summary in self.requests.items():
            if trace_id not in answered and trace_id not in self.boundary_requests:
                session = (summary.get("identity") or {}).get("session") or summary["thread"]
                self.note("I-01:request_without_response", summary["locator"], session)
        for turn in self.turns:
            # USG-05: count_tokens over the same counted body answers the input this turn billed.
            counted = self.count_results.get(turn.get("counted")) if turn.get("counted") else None
            if counted and turn.get("billed_input") is not None:
                self.counts["den:count_tokens_pairs"] += 1
                if any(value != turn["billed_input"] for value in counted):
                    self._usage_found([Violation("USG-05", {"why": "count_tokens_vs_billed_input"})],
                                      turn["loc"], turn["session"])  # fmt: skip
        by_thread = defaultdict(list)
        for summary in self.requests.values():
            if summary["thread"]:
                by_thread[summary["thread"]].append(summary)
        for items in by_thread.values():
            items.sort(key=lambda s: s["order"])
        orders = {thread: [s["order"] for s in items] for thread, items in by_thread.items()}
        # Where each request's history ends: a request whose last assistant message follows one
        # of these ends continues that request's conversation.
        ends = {(s["thread"], s["history"], s.get("last_digest"))
                for s in self.requests.values() if s["thread"]}  # fmt: skip
        replies = {t["trace_id"]: t for t in self.turns}
        # A request answered more than once (the harness re-sent it after a cut, or a proxy
        # retried it after the network failed), or asked again with more appended to its last
        # user message: only the last completed answer can be the one the client went on with.
        # The earlier answers are superseded, not echoed.
        completed = defaultdict(list)
        reasks = defaultdict(list)
        for t in self.turns:
            if t["thread"] and t["terminal"] == "completed":
                completed[(t["thread"], t["body_sha"])].append(t["order"])
                if t.get("reask"):
                    reasks[(t["thread"], t["reask"][0])].append((t["order"], tuple(t["reask"][1])))
        for turn in self.turns:
            thread = turn["thread"]
            if not thread:
                continue
            again = any(o > turn["order"] for o in completed.get((thread, turn["body_sha"]), ()))
            reasked = not again and self._reasked(turn, reasks)
            if again or reasked:
                self.counts["C-03:superseded_answers"] += 1
                self.counts["C-03:echo_superseded"] += len(turn["calls"])
                if reasked:
                    self.note("C-03:reasked_answers", turn["loc"], turn["session"])
                continue
            nxt = successor = None
            start = bisect.bisect_right(orders[thread], turn["order"])
            for candidate in by_thread[thread][start:]:
                if not turn["agent"] or candidate["identity"].get("agent") in (None, turn["agent"]):
                    if nxt is None:
                        nxt = candidate
                    # The next request of the conversation itself: side requests stay out of it.
                    if not candidate.get("side_request"):
                        successor = candidate
                        break
            loc = turn["loc"]
            session = turn["session"]
            echoed = bool(turn["calls"]) and nxt is not None and all(
                (e := self.echo_first.get((thread, call[0]))) is not None and e[0] > turn["order"]
                and self._call_echo(e, call, turn["protocol"]) is not None
                for call in turn["calls"])  # fmt: skip
            # The request that sends this turn back is the one carrying its call ids, not the
            # next request in time: a side request of the same session (no tools) can come first.
            echo_turn = self._echo_turn(turn, thread)
            pair_id = (echo_turn[2] if echo_turn is not None
                       else (nxt or {}).get("locator", {}).get("trace_id"))  # fmt: skip
            following = replies.get(pair_id) if echoed else None
            if following and following.get("billed_input") is not None \
                    and turn.get("billed_input") is not None:  # fmt: skip
                # USG-08: the next prompt holds this turn's calls, so it can only have grown.
                self.counts["den:next_input_pairs"] += 1
                if following["billed_input"] < turn["billed_input"]:
                    paired = self.requests.get(pair_id) or {}
                    own = self.requests.get(turn["trace_id"]) or {}
                    # The next request may declare fewer of this turn's tools (Codex asks for its
                    # compaction summary with the whole history and no tools): its prompt may then
                    # be smaller by at most what those declarations took.
                    bound = removed_tools_bound(
                        self.declarations.get(own.get("tools_key"), ()),
                        self.declarations.get(paired.get("tools_key"), ()),
                    )
                    shrank = turn["billed_input"] - following["billed_input"]
                    if paired.get("history", turn["history"]) < turn["history"]:
                        # The harness replaced its history with a summary (compaction).
                        self.counts["C-14:next_input_shrank_by_compaction"] += 1
                    elif paired.get("context_management"):
                        self.counts["C-14:next_input_shrank_by_context_management"] += 1
                    elif bound is not None and shrank <= bound:
                        self.counts["C-14:next_input_shrank_by_tools_change"] += 1
                    else:
                        why = ("next_input_shrank" if bound is None
                               else "next_input_shrank_beyond_removed_tools")  # fmt: skip
                        self._usage_found([Violation("USG-08", {"why": why})], loc, session)
            for call in turn["calls"]:
                call_id, name, digest = call[:3]
                echo = self.echo_first.get((thread, call_id))
                if nxt is None:
                    self.counts["C-03:no_next_turn"] += 1
                elif echo is None:
                    self.counts["C-03:echo_missing"] += 1
                elif echo[0] < turn["order"]:
                    self.counts["C-03:call_id_seen_before"] += 1
                elif (how := self._call_echo(echo, call, turn["protocol"])) is None:
                    self.note("C-03", loc, session, **safe_name(name),
                              name_differs=echo[2] != name, payload_differs=echo[3] != digest)  # fmt: skip
                else:
                    self.counts["C-03:" + how] += 1
                    if how.startswith("echo_rewritten:"):
                        self.counts["C-03:echo_rewritten"] += 1
                result = self.result_first.get((thread, call_id))
                if result is None or result[0] < turn["order"]:
                    self.outcome_classes["pending" if nxt is None else "unanswered"] += 1
                    if nxt is not None:
                        self.counts["I-08:unanswered"] += 1
                elif result[1] is None:
                    self.outcome_classes["unknown_profile"] += 1
                else:
                    self.outcome_classes[result[1]] += 1
                    if result[1] != "result":
                        self.note("H-01:" + result[1], loc, session, **safe_name(name))
            if nxt is not None:
                self._text_echo(turn, thread, loc, session, successor, ends)
            else:
                # Delivered, with no later request to echo it: counted, as calls are, so the
                # C-03 record adds up to what was delivered. Responses items count one each;
                # Messages and Chat texts and reasoning once per reply each.
                kinds = {block[0] for block in turn["text_blocks"]}
                self.counts["C-03:text_no_next_turn"] += len(turn["texts"]) + ("text" in kinds)
                self.counts["C-03:reasoning_no_next_turn"] += (len(turn.get("reasonings", ()))
                                                               + ("reasoning" in kinds))  # fmt: skip
            if turn["overlength"] and nxt is not None:
                self.counts["den:overlength_next"] += 1
                if nxt["body_sha"] == turn["body_sha"]:
                    self.note("H-04", loc, session, why="identical_resend")
                elif nxt["identity"].get("compaction") or nxt["history"] < turn["history"]:
                    self.counts["H-04:compacted"] += 1
                else:
                    self.counts["H-04:unknown"] += 1
        for (thread, _), occurrences in self.bodies.items():
            if len(occurrences) < 2:
                continue
            occurrences.sort()
            for (_, first_id), (_, again_id) in zip(occurrences, occurrences[1:]):
                summary = self.requests.get(again_id) or {}
                loc = summary.get("locator", {})
                session = (summary.get("identity") or {}).get("session") or thread
                first = replies.get(first_id) or {}
                cause = resend_cause(first.get("http_status"), bool(first.get("overlength")),
                                     bool(first.get("failed")))  # fmt: skip
                if cause == "H-02":
                    self.note("H-02", loc, session, after="our_client_error")
                else:
                    self.note(cause, loc, session)

    @staticmethod
    def _reasked(turn: dict, reasks: dict) -> bool:
        """A later answered request of the thread asked this one again with more appended."""
        if not turn.get("reask"):
            return False
        frame, blocks = turn["reask"][0], tuple(turn["reask"][1])
        return any(order > turn["order"] and len(later) > len(blocks)
                   and later[:len(blocks)] == blocks
                   for order, later in reasks.get((turn["thread"], frame), ()))  # fmt: skip

    def _loose_digest(self, payload, call_kind, tool, delivered: bool,
                      only: Optional[tuple] = None) -> Optional[str]:  # fmt: skip
        """A function call's digest under the harness's declared echo rewrites, else None.

        Default-valued arguments are left out on both sides. A cut ``cd <dir> &&`` is cut
        from what we delivered only, so an echo that changed the directory still differs, and
        pi's ``edit`` arguments are taken as pi keeps them on that side only too. ``only``: these
        of the declared rewrites, not all of them.
        """
        if not self.echo_rewrites or call_kind != "function":
            return None
        rewrites = [r for r in (only or self.echo_rewrites)
                    if delivered or r not in DELIVERED_ONLY_REWRITES]  # fmt: skip
        return rewritten_digest(payload, tool.schema if tool is not None else None, rewrites,
                                tool.name if tool is not None else None)  # fmt: skip

    def _call_digests(self, payload, call_kind, tool, delivered: bool) -> tuple:
        """What C-03 holds a call's payload to, on either side of the echo.

        ``(value, under every declared rewrite, bytes, ((rewrite, under it alone), ...))``: the
        typed JSON value (``payload_digest``), the value under the profile's declared rewrites, the
        payload's bytes, and with two rewrites or more the value under each alone, which names the
        rewrite an echo needed.
        """
        value = payload_digest(payload, call_kind)
        loose = self._loose_digest(payload, call_kind, tool, delivered)
        each = ()
        if len(self.echo_rewrites) > 1 and call_kind == "function":
            each = tuple((r, self._loose_digest(payload, call_kind, tool, delivered, (r,)))
                         for r in self.echo_rewrites)  # fmt: skip
        return value, loose, sha256_text(payload) if isinstance(payload, str) else None, each

    def _call_echo(self, echo: tuple, call: tuple, protocol: str) -> Optional[str]:
        """How the harness echoed one of our calls, or None when it differs (C-03).

        ``echo_equal``: the same bytes; a Messages echo is an object, so there its JSON value.
        ``echo_reserialized``: the same JSON value in other bytes (spacing, key order, escapes),
        from a harness whose profile declares no re-serialization; ``echo_rewritten:<rewrite>``:
        equal under a rewrite the profile declares in ``echo.tool_input`` (pi re-serializes every
        call: ``pi_json_reserialize``), or under all of them together.
        """
        name, value, loose = call[1], call[2], call[3] if len(call) > 3 else None
        if echo[2] != name:
            return None
        same_bytes = len(echo) > 5 and len(call) > 4 and echo[5] is not None and echo[5] == call[4]
        if same_bytes or (echo[3] == value and protocol == "anthropic-messages"):
            return "echo_equal"
        if echo[3] == value:
            return ("echo_rewritten:pi_json_reserialize" if "pi_json_reserialize" in self.echo_rewrites
                    else "echo_reserialized")  # fmt: skip
        delivered, echoed = (
            dict(call[5] if len(call) > 5 else ()),
            dict(echo[6] if len(echo) > 6 else ()),
        )
        for rewrite in self.echo_rewrites:
            if delivered.get(rewrite) is not None and delivered[rewrite] == echoed.get(rewrite):
                return "echo_rewritten:" + rewrite
        if loose is not None and len(echo) > 4 and echo[4] == loose:
            return "echo_rewritten:" + "+".join(self.echo_rewrites)
        return None

    def _echo_turn(self, turn, thread) -> Optional[tuple]:
        """The first later request whose history holds this turn's calls, by their ids.

        ``(order, "texts", trace_id, echoed text blocks, history length)``, or None for a turn
        without calls or one never sent back.
        """
        if not turn["calls"]:
            return None
        found = self.echo_first.get((thread, "calls", tuple(sorted(c[0] for c in turn["calls"]))))
        return found if found is not None and found[0] > turn["order"] else None

    def _calls_sent_back(self, turn, thread) -> bool:
        """Whether a later request of the thread sends back any of this turn's calls."""
        return any((e := self.echo_first.get((thread, call[0]))) is not None
                   and e[0] > turn["order"] for call in turn["calls"])  # fmt: skip

    @staticmethod
    def _not_echoed_by_design(turn, successor, ends, calls_sent_back=False) -> Optional[str]:
        """Why no later request echoes this reply when the harness never appends it, else None.

        ``side_request``: the reply answers a side request (structured output without tools,
        ``side_request()``), which the harness keeps out of the conversation. ``compacted``: a
        reply none of whose calls is sent back, after which the conversation went on from a
        summary. The next request of the conversation (side requests aside) has a shorter history
        than this turn's, as in USG-08's compaction, and continues no earlier request; a
        sub-agent's last reply is followed by its parent's next request, which can be shorter but
        continues the parent's conversation. The reply to a compaction request is that summary,
        with any call in it, which the harness drops. A Responses request names no position, so
        on Responses the shorter history decides (a Codex sub-agent has a thread of its own).
        """
        if turn.get("side_request"):
            return "side_request"
        if calls_sent_back or successor is None or successor["history"] >= turn["history"]:
            return None
        continues = successor.get("continues")
        if continues is not None and (successor["thread"], *continues) in ends:
            return None
        return "compacted"

    def _text_echo(self, turn, thread, loc, session, successor=None, ends=frozenset()) -> None:
        """C-03 for texts: what the harness sent back is what we delivered (harness ``echo``).

        Messages and Chat replies count once per reply and kind: their text blocks under
        ``text_*``, their thinking or reasoning blocks under ``reasoning_*``.
        """
        policy = self.harness.get_path("echo") if self.harness else None
        policy = policy if isinstance(policy, dict) else {}
        kinds = [k for k, key in (("text", "text"), ("reasoning", "reasoning"))
                 if policy.get(key) == "exact"]  # fmt: skip
        unknown = [k for k in ("text", "reasoning") if prof.is_todo(policy.get(k))]
        # An item or block no later request sends back is missing, unless the harness never
        # appends it (a side request's reply, a compaction summary).
        why = self._not_echoed_by_design(turn, successor, ends, self._calls_sent_back(turn, thread))
        missing = f"_not_echoed_{why}" if why else "_echo_missing"
        if turn["texts"] or turn.get("reasonings"):  # Responses: the echo carries our item ids
            for item_id, digest in turn["texts"]:
                echo = self.echo_first.get((thread, item_id))
                if echo is None or echo[0] <= turn["order"]:
                    self.counts["C-03:text" + missing] += 1
                elif "text" not in kinds:
                    self.counts["C-03:text_echo_unchecked"] += "text" in unknown
                elif echo[3] != digest:
                    self.note("C-03", loc, session, field="text")
                else:
                    self.counts["C-03:text_echo_equal"] += 1
            # Reasoning items come back by id too, with their carrier (encrypted_content), which
            # is compared by value.
            for item_id, carrier in turn.get("reasonings", ()):
                echo = self.echo_first.get((thread, item_id))
                if echo is None or echo[0] <= turn["order"]:
                    self.counts["C-03:reasoning" + missing] += 1
                elif "reasoning" not in kinds:
                    self.counts["C-03:reasoning_echo_unchecked"] += "reasoning" in unknown
                elif echo[3] != carrier:
                    self.note("C-03", loc, session, field="reasoning")
                else:
                    self.counts["C-03:reasoning_echo_equal"] += 1
            return
        delivered = turn["text_blocks"]
        if not delivered:
            return
        present = [k for k in ("text", "reasoning") if any(b[0] == k for b in delivered)]
        echo = self._echo_turn(turn, thread) or self.echo_first.get(
            (thread, "pos", turn["history"], turn.get("last_digest"))
        )
        if echo is None or echo[0] <= turn["order"]:
            for kind in present:
                self.counts[f"C-03:{kind}{missing}"] += 1
            return
        echoed = echo[3]
        for kind in present:
            if kind in unknown:
                self.counts[f"C-03:{kind}_echo_unchecked"] += 1
        differs = [k for k in kinds
                   if [b for b in delivered if b[0] == k] != [b for b in echoed if b[0] == k]]  # fmt: skip
        if differs:
            self.note("C-03", loc, session, field=",".join(differs))
        for kind in present:
            if kind in kinds and kind not in differs:
                self.counts[f"C-03:{kind}_echo_equal"] += 1

    # ------------------------------------------------------------------ output

    def report(self) -> dict:
        return {
            "counts": dict(self.counts),
            "by_rule": {k: dict(v) for k, v in self.by_rule.items()},
            "call_classes": dict(self.call_classes),
            "outcome_classes": dict(self.outcome_classes),
            "sessions": len(self.sessions),
            "sessions_with": {k: len(v) for k, v in self.session_events.items()},
            "protocols": dict(self.protocols),
            "unknown_request_fields": dict(
                sorted(self.unknown_fields.items(), key=lambda kv: (-kv[1], kv[0]))[:50]
            ),
            "examples": dict(self.examples),
            # M-02 can see only the runaway tokens the model profile names (or the default [PAD]
            # run, profiles.DEFAULT_DEGENERATE).
            "model_degenerate_declared": bool(prof.known(self.model.get("degenerate")))
            if self.model
            else False,
            "model_degenerate_text_declared": prof.degenerate_text_declared(self.model)
            if self.model
            else False,
        }


_SHARED: Optional["Pipeline"] = None


def _requests_task(args) -> dict:
    index, path, harness, model, build_id = args
    pipeline = Pipeline(harness, model, build_id)
    pipeline._scan_request_shard(index, path)
    return pipeline._request_state()


def _raw_task(args):
    path, card, patterns, id_runs = args
    part, counts = {}, Counter()
    degenerate = [re.compile(p) for p in patterns]
    stops = [s for s in (card or {}).get("stop_markers") or [] if s]
    for _, raw in iter_lines(path):
        rec = load_json_line(raw)
        if rec is None:
            counts["inventory:unparsable_raw_lines"] += 1
            continue
        counts["den:raw_records"] += 1
        keys = []
        if rec.get("disagg_request_id") not in (None, ""):
            keys.append("d:" + str(rec["disagg_request_id"]))
        if rec.get("request_id") not in (None, ""):
            keys.append("r:" + str(rec["request_id"]))
        if not keys:
            counts["I-01:raw_without_id"] += 1
            continue
        reading = native_mod.summary(card, rec)
        text = rec.get("text") or ""
        # M-02 on what the model wrote: the declared runaway tokens are special tokens, which the
        # raw text shows only when the server kept them (its stop marker is there too).
        reading["specials_visible"] = any(stop in text for stop in stops)
        reading["degenerate"] = any(rx.search(text) for rx in degenerate)
        token_ids = rec.get("token_ids")
        if isinstance(token_ids, list) and id_runs:
            # The ids show a run whether or not decoding kept the special tokens.
            reading["specials_visible"] = True
            reading["degenerate"] = reading["degenerate"] or prof.has_id_run(token_ids, id_runs)
        count = rec.get("num_output_tokens")
        reading["output_tokens"] = (
            count
            if isinstance(count, int)
            else (len(rec["token_ids"]) if isinstance(rec.get("token_ids"), list) else None)
        )
        # When the generation ran: it tells apart the records one engine id names (join_raw).
        reading["started_at"] = instant(rec.get("started_at"))
        reading["finished_at"] = instant(rec.get("finished_at"))
        # Whether the server stopped it: such a record is not read as a finished generation.
        reading["stopped"] = raw_stopped(rec)
        if reading["stopped"]:
            counts["inventory:raw_stopped:" + reading["stopped"]] += 1
            if rec.get("end_reason") not in (None, ""):
                counts["inventory:raw_end_reason:" + str(rec["end_reason"])[:40]] += 1
        for key in keys:
            # Every record under an id is kept, within a shard as across shards.
            part.setdefault(key, []).append(reading)
    return part, counts


def _responses_task(path: str) -> dict:
    pipeline = _SHARED
    pipeline._reset_replies()
    pipeline._scan_response_shard(path)
    return pipeline._reply_state()


def _trace_ids(paths: List[str], kind: str) -> set:
    ids = set()
    for path in expand_shards(paths, kind):
        for _, raw in iter_lines(path):
            rec = load_json_line(raw)
            if isinstance(rec, dict) and rec.get("trace_id"):
                ids.add(rec["trace_id"])
    return ids


def run(paths: List[str], harness: prof.Profile, model: prof.Profile,
        request_paths: Optional[List[str]] = None, build_id: Optional[str] = None,
        workers: int = 1, raw_paths: Optional[List[str]] = None,
        response_paths: Optional[List[str]] = None) -> dict:  # fmt: skip
    """``request_paths`` and ``response_paths`` are boundary hours: join-only context.

    A request from ``request_paths`` is never counted as lost; a response from
    ``response_paths`` is read only when it answers a request of the population.
    """
    pipeline = Pipeline(harness, model, build_id)
    pipeline.add_raw(list(raw_paths or []), workers)
    pipeline.add_requests(list(paths) + list(request_paths or []), workers)
    if request_paths:
        pipeline.boundary_requests = _trace_ids(list(request_paths), "requests") - _trace_ids(
            list(paths), "requests"
        )
    pipeline.add_responses(paths, workers)
    if response_paths:
        pipeline.add_boundary_responses(list(response_paths))
    pipeline.finish()
    return pipeline.report()
