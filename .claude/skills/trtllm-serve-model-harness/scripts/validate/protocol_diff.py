#!/usr/bin/env python3
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
"""A protocol differential: one agent and one task, run over several wires (check X-P).

The same harness (e.g. pi, which speaks all three) runs the same task against the
same served model once per API: Chat Completions, Responses, Anthropic Messages.
Each run's turns are put in one protocol-free form and compared turn by turn:

  request   what the harness sent: system text, tools, and the conversation
            (user text, assistant text and calls, tool results). Call ids are
            compared by position: every protocol mints its own.
  billed    the input tokens the server billed: with equal requests, a different
            count means the server rendered the same conversation differently.
  reply     what the server delivered: reasoning, text, calls. Under greedy
            decoding (recorder --override temperature=0), equal prompts must give
            equal replies; a difference is a protocol-specific output conversion.

Each difference is attributed: `harness` (the requests differ), `after_divergence`
(the requests differ because an earlier turn of this pair already did),
`harness_literal`
(the requests mean the same but are written differently, e.g. `"required": []`
in one API's tool schema only: the server's part cannot be told apart),
`server_render` (equal requests, replayed reasoning included, other billed input),
`server_output` (equal
requests and billed input, another reply), `server_rejected` (one path refused or
broke off a request another path served). Reasoning is compared apart: a harness
may legitimately replay it differently per API.

    protocol_diff.py --out report.json LABEL=TRACE_DIR [LABEL=TRACE_DIR ...]

`divergent_generation` (equal prompts, generations that differ mid-text) is not
attributed: serve the differential deterministically (no KV-cache block reuse,
one request at a time) to settle it.

`not_judged` (a turn whose reply on either path says nothing of the server, as the
verdict reads it: no response record, a stream its client cut before the terminal
event, a reply the recorder lost (R-06 counts it), or one pending when the recorder
stopped) is not attributed either; the row's `not_judged` names the reason per path.

Exit code: 0 no server-side difference; 1 a server-side difference; 2 not
conclusive (no turns to compare, only divergent generations or turns not judged,
or a shard that cannot be read to its end: the message names it, and no report is
written); 3 usage error.
Nothing in the report carries content: lengths and SHA-256 prefixes only.
"""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import proto_anthropic  # noqa: E402
import proto_chat  # noqa: E402
import proto_responses  # noqa: E402
import usage as usage_mod  # noqa: E402
from pipeline import CAPTURE_SIDE, CLIENT_SIDE, UPSTREAM_SIDE  # noqa: E402
from tracelib import ShardError, expand_shards, iter_lines, load_json_line  # noqa: E402
from turnlib import _canonical, content_text, sha256_text  # noqa: E402

ADAPTERS = {"/v1/chat/completions": proto_chat, "/v1/responses": proto_responses,
            "/v1/messages": proto_anthropic}  # fmt: skip


def _digest(value) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str))[:16]


def _args(value):
    if isinstance(value, str):
        try:
            value = json.loads(value) if value.strip() else {}
        except json.JSONDecodeError:
            return ["unparsable", sha256_text(value)[:16]]
    return _canonical(value)


def _assistant(text="", reasoning="", calls=None) -> dict:
    return {"role": "assistant", "text": text, "reasoning": reasoning, "calls": list(calls or [])}


def _merge(turns: list, piece: dict) -> None:
    """Consecutive assistant pieces (reasoning, message, call items) are one assistant turn."""
    if turns and turns[-1]["role"] == "assistant":
        last = turns[-1]
        last["text"] += piece["text"]
        last["reasoning"] += piece["reasoning"]
        last["calls"] += piece["calls"]
    else:
        turns.append(piece)


def conversation(route: str, body: dict) -> dict:
    """System text, tools and turns of one request, in a protocol-free form."""
    system, turns, tools = [], [], []
    if route == "/v1/chat/completions":
        for spec in body.get("tools") or []:
            fn = spec.get("function") or {}
            tools.append((fn.get("name"), _canonical(fn.get("parameters"))))
        for m in body.get("messages") or []:
            role = m.get("role")
            if role in ("system", "developer"):
                system.append(content_text(m.get("content")))
            elif role == "user":
                turns.append({"role": "user", "text": content_text(m.get("content"))})
            elif role == "assistant":
                calls = [((tc.get("function") or {}).get("name"), _args((tc.get("function") or {}).get("arguments")))
                         for tc in m.get("tool_calls") or []]  # fmt: skip
                reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
                _merge(turns, _assistant(content_text(m.get("content")),
                                         reasoning if isinstance(reasoning, str) else "", calls))  # fmt: skip
            elif role == "tool":
                turns.append({"role": "tool", "text": content_text(m.get("content"))})
    elif route == "/v1/responses":
        for spec in body.get("tools") or []:
            tools.append((spec.get("name"), _canonical(spec.get("parameters"))))
        if isinstance(body.get("instructions"), str):
            system.append(body["instructions"])
        items = body.get("input")
        for item in items if isinstance(items, list) else [{"role": "user", "content": items}]:
            kind = item.get("type", "message")
            role = item.get("role")
            if kind == "message" and role in ("system", "developer"):
                system.append(content_text(item.get("content")))
            elif kind == "message" and role == "user":
                turns.append({"role": "user", "text": content_text(item.get("content"))})
            elif kind == "message" and role == "assistant":
                _merge(turns, _assistant(text=content_text(item.get("content"))))
            elif kind == "reasoning":
                parts = item.get("content") or item.get("summary") or []
                _merge(turns, _assistant(reasoning=content_text(parts)))
            elif kind in ("function_call", "custom_tool_call"):
                payload = (
                    item.get("arguments")
                    if kind == "function_call"
                    else {"input": item.get("input")}
                )
                _merge(turns, _assistant(calls=[(item.get("name"), _args(payload))]))
            elif kind in ("function_call_output", "custom_tool_call_output"):
                turns.append({"role": "tool", "text": content_text(item.get("output"))})
    elif route == "/v1/messages":
        for spec in body.get("tools") or []:
            tools.append((spec.get("name"), _canonical(spec.get("input_schema"))))
        system.append(content_text(body.get("system")))
        for m in body.get("messages") or []:
            role, content = m.get("role"), m.get("content")
            blocks = (
                content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
            )
            if role == "system":
                system.append(content_text(blocks))
            elif role == "user":
                text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
                if text:
                    turns.append({"role": "user", "text": text})
                for b in blocks:
                    if b.get("type") == "tool_result":
                        turns.append({"role": "tool", "text": content_text(b.get("content"))})
            elif role == "assistant":
                _merge(
                    turns,
                    _assistant(
                        text="".join(b.get("text", "") for b in blocks if b.get("type") == "text"),
                        reasoning="".join(
                            b.get("thinking", "") for b in blocks if b.get("type") == "thinking"
                        ),
                        calls=[
                            (b.get("name"), _args(b.get("input")))
                            for b in blocks
                            if b.get("type") == "tool_use"
                        ],
                    ),
                )
    # Request parameters that change what the server renders. They are written differently per
    # API, so only their absence everywhere compares equal: run the differential with the
    # harness's thinking off.
    params = {key: body.get(key) for key in ("reasoning_effort", "reasoning", "thinking",
                                             "chat_template_kwargs", "output_config")
              if body.get(key) not in (None, {}, [])}  # fmt: skip
    literal = sorted((str(name), json.dumps(_literal_schema(route, body, name), ensure_ascii=False))
                     for name, _ in tools)  # fmt: skip
    return {"system": "\n".join(s for s in system if s), "tools": sorted(tools, key=lambda t: str(t[0])),
            "tools_literal": literal, "turns": turns, "params": _canonical(params)}  # fmt: skip


def _literal_schema(route: str, body: dict, name):
    """A tool's schema as the harness wrote it: key order is what the template renders."""
    for spec in body.get("tools") or []:
        if route == "/v1/chat/completions" and (spec.get("function") or {}).get("name") == name:
            return spec["function"].get("parameters")
        if route == "/v1/responses" and spec.get("name") == name:
            return spec.get("parameters")
        if route == "/v1/messages" and spec.get("name") == name:
            return spec.get("input_schema")
    return None


def load_run(path: str) -> list:
    """The run's turns in arrival order: request conversation, billed input, delivered reply."""
    requests, replies = {}, {}
    for shard in expand_shards([path], "requests"):
        for _, raw in iter_lines(shard):
            rec = load_json_line(raw)
            route = str((rec or {}).get("route") or "").split("?", 1)[0]
            if rec and route in ADAPTERS and isinstance(rec.get("body"), dict):
                requests[rec["trace_id"]] = (rec.get("recorded_at") or "", route, rec)
    for shard in expand_shards([path], "responses"):
        for _, raw in iter_lines(shard):
            rec = load_json_line(raw)
            if rec and rec.get("trace_id") in requests:
                replies[rec["trace_id"]] = rec
    turns = []
    for trace_id, (at, route, rec) in sorted(requests.items(), key=lambda kv: kv[1][0]):
        adapter = ADAPTERS[route]
        view = adapter.parse_request(rec["body"], {})
        reply_rec = replies.get(trace_id)
        reply = adapter.parse_reply(reply_rec, view) if reply_rec else None
        delivered = reply.delivered() if reply else []
        turns.append({
            "trace_id": trace_id,
            "protocol": adapter.PROTOCOL,
            "conversation": conversation(route, rec["body"]),
            "billed_input": usage_mod.input_tokens(adapter.PROTOCOL, reply.usage) if reply else None,
            "reply": {
                "text": "".join(i.text or "" for i in delivered if i.kind == "text"),
                "reasoning": "".join(i.text or "" for i in delivered if i.kind == "reasoning"),
                "calls": [(i.name, _args(i.payload)) for i in delivered if i.kind == "call"],
                "terminal": reply.terminal.get("kind") if reply else None,
                "http_status": reply_rec.get("http_status") if reply_rec else None,
                "not_judged": _unjudged(reply),
            },
        })  # fmt: skip
    return turns


def _unjudged(reply):
    """Why a reply says nothing of the server, as the verdict reads it; None when it does."""
    if reply is None:
        return "no_reply"  # no response record: a gap in the capture
    if reply.status in CLIENT_SIDE and reply.terminal.get("kind") == "none":
        return "client_cut"  # cut before its terminal event: incomplete by construction
    if reply.status in UPSTREAM_SIDE:
        return "recorder_lost"  # the recorder's own error, or the server's bytes up to the loss
    if reply.status in CAPTURE_SIDE:
        return "pending"  # the recorder stopped before the reply ended
    return None


def _semantic_schema(value):
    """A canonical schema without the literal-only differences: an empty `required` list."""
    if isinstance(value, list) and value and value[0] == "o":
        return [
            "o",
            [
                [k, _semantic_schema(v)]
                for k, v in value[1]
                if not (k == "required" and v == ["l", []])
            ],
        ]
    if isinstance(value, list) and value and value[0] == "l":
        return ["l", [_semantic_schema(v) for v in value[1]]]
    return value


def _request_key(conv: dict, with_reasoning: bool, semantic: bool = False):
    """Strict: tools exactly as written (key order included); semantic: what they mean."""
    turns = [
        {k: v for k, v in t.items() if with_reasoning or k != "reasoning"} for t in conv["turns"]
    ]
    if semantic:
        tools = [(name, _semantic_schema(schema)) for name, schema in conv["tools"]]
    else:
        tools = conv.get("tools_literal", conv["tools"])
    return _digest(
        {"system": conv["system"], "tools": tools, "turns": turns, "params": conv["params"]}
    )


def _reply_difference(a: dict, b: dict) -> str:
    """`server_output` when one generation was converted two ways; else `divergent_generation`.

    Conversion leaves the generated stream itself alone: the same reasoning and text
    split differently between channels, edge whitespace, or the same text with other
    calls. Free text that diverges in the middle is two different generations from
    the same prompt: runtime nondeterminism (KV-cache block reuse, batching), not a
    protocol path, until a deterministic re-run shows otherwise.
    """
    joined_a, joined_b = a["reasoning"] + a["text"], b["reasoning"] + b["text"]
    if joined_a == joined_b or joined_a.strip() == joined_b.strip():
        return "server_output"
    if a["text"].strip() == b["text"].strip() and a["reasoning"].strip() == b["reasoning"].strip():
        return "server_output"  # the same text, other calls
    return "divergent_generation"


def _failed(reply: dict) -> bool:
    status = reply.get("http_status")
    return (isinstance(status, int) and status >= 400) or reply.get("terminal") == "error"


def _lengths(reply: dict) -> list:
    """Text, reasoning and call counts of a reply: what a report may say about its content."""
    return [len(reply["text"]), len(reply["reasoning"]), len(reply["calls"])]


def compare(runs: dict) -> dict:
    labels = list(runs)
    base = labels[0]
    findings, rows = Counter(), []
    # Each pair is compared as deep as both of its runs go.
    depth = max((min(len(runs[base]), len(runs[label])) for label in labels[1:]), default=0)
    diverged = set()
    for index in range(depth):
        row = {"turn": index + 1}
        ref = runs[base][index]
        for label in labels[1:]:
            if index >= len(runs[label]):
                continue
            other = runs[label][index]
            same_request = _request_key(ref["conversation"], False) == _request_key(
                other["conversation"], False
            )
            parts = {}
            for part in ("system", "tools", "params"):
                parts[part] = ref["conversation"][part] == other["conversation"][part]
            blank = {"system": "", "tools": [], "tools_literal": [], "params": None}
            parts["turns"] = _request_key({**ref["conversation"], **blank}, False) == \
                _request_key({**other["conversation"], **blank}, False)  # fmt: skip
            parts["reasoning_replay"] = _request_key(ref["conversation"], True) == \
                _request_key(other["conversation"], True)  # fmt: skip
            billed_same = ref["billed_input"] == other["billed_input"]
            reply_same = {
                k: ref["reply"][k] == other["reply"][k] for k in ("text", "reasoning", "calls")
            }
            same_meaning = _request_key(ref["conversation"], False, True) == \
                _request_key(other["conversation"], False, True)  # fmt: skip
            failed = [_failed(ref["reply"]), _failed(other["reply"])]
            unjudged = [ref["reply"].get("not_judged"), other["reply"].get("not_judged")]
            if any(unjudged):
                # A reply missing or cut on its way says nothing of either protocol path.
                verdict = "not_judged"
            elif same_meaning and failed[0] != failed[1]:
                # One protocol path refused or broke off what another served.
                verdict = "server_rejected"
            elif not same_request and label in diverged:
                # The history carries an earlier turn that already differed: not a new difference.
                verdict = "after_divergence"
            elif not same_request:
                verdict = "harness_literal" if same_meaning else "harness"
            elif not billed_same and not parts["reasoning_replay"]:
                # Equal but for the reasoning they replay: the bill follows that reasoning, so
                # it says nothing of the server. After an earlier divergence the replayed
                # reasoning is that divergence; otherwise the harness replayed it differently.
                verdict = "after_divergence" if label in diverged else "harness"
            elif not billed_same:
                verdict = "server_render"
            elif not all(reply_same.values()):
                verdict = _reply_difference(ref["reply"], other["reply"])
            else:
                verdict = "equal"
            if verdict != "equal":
                diverged.add(label)
            findings[verdict] += 1
            row[label] = {
                "request_equal": parts,
                "billed_input": [ref["billed_input"], other["billed_input"]],
                "reply_equal": reply_same,
                "reply_lengths": [_lengths(ref["reply"]), _lengths(other["reply"])],
                "not_judged": unjudged,
                "attribution": verdict,
            }
        rows.append(row)
    return {"base": base, "turns_compared": depth, "turn_counts": {k: len(v) for k, v in runs.items()},
            "attributions": dict(findings), "rows": rows}  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    ap.add_argument("--out", required=True)
    ap.add_argument("runs", nargs="+", help="LABEL=TRACE_DIR, the first is the reference")
    args = ap.parse_args(argv)
    runs = {}
    for spec in args.runs:
        label, _, path = spec.partition("=")
        if not path:
            print(f"protocol_diff.py: expected LABEL=TRACE_DIR, got {spec!r}", file=sys.stderr)
            return 3
        try:
            runs[label] = load_run(path)
        except ShardError as error:  # turns from part of a shard would compare as all of them
            print(f"protocol_diff.py: {error}", file=sys.stderr)
            return 2
    report = compare(runs)
    report["protocols"] = {
        label: sorted({t["protocol"] for t in turns}) for label, turns in runs.items()
    }
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps({"turns_compared": report["turns_compared"], "turn_counts": report["turn_counts"],
                      "attributions": report["attributions"]}))  # fmt: skip
    if report["turns_compared"] == 0:
        return 2
    server = ("server_render", "server_output", "server_rejected")
    if any(report["attributions"].get(k) for k in server):
        return 1
    inconclusive = ("divergent_generation", "not_judged")
    return 2 if any(report["attributions"].get(k) for k in inconclusive) else 0


if __name__ == "__main__":
    sys.exit(main())
