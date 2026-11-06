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
"""Tool-call conformance of Responses replies against their OWN request (read-only).

Joins every response record to its request record by ``trace_id`` and checks the
call items (function_call, custom_tool_call) of the terminal Responses object
against the declarations of that same request -- top-level ``tools``, tools
inside ``{"type": "namespace"}`` groups, and ``additional_tools`` input items
(e.g. where Codex declares its tools). Never against a global tool list: a call is
valid or not only relative to what its own client offered.

Per call, one class:

  valid
  type_mismatch               a custom tool delivered as function_call or the reverse
  namespace_mismatch          the call's namespace is not the declaring namespace
  prefixed_name_of_declared   ``functions.exec`` delivered for a tool declared as ``exec``
  nested_tool_called_directly a tool that a custom tool's description declares inside
                              itself (``### `name``` headers, e.g. Codex's ``exec``) called
                              top-level
  undeclared                  no declaration answers to the name
  payload_invalid             function arguments not a strict-JSON object or not valid
                              against the declared schema; custom input empty or not
                              accepted by the declared Lark grammar
  unsupported_declared_type   the name resolves to a declared type this check does not
                              evaluate (web_search, local_shell, ...); counted, not judged

Per response: ``ptc_violation`` when the request set ``parallel_tool_calls`` to
JSON false and the terminal object holds two or more call items (true, null and
an absent field constrain nothing).

Screens (suspicions, never verdicts): ``degenerate_run`` (configurable regexes
over message and reasoning text, default a run of 20 ``[PAD]``) and
``markup_in_message`` / ``markup_in_call`` (format markers left in delivered text;
``quoted`` when every hit sits between quotes or backticks).

Attribution hint, used for the verdicts:
  serving_shape: type_mismatch, namespace_mismatch, prefixed_name_of_declared
                 -> the server should have normalized these; FAIL when > 0
  contract:      ptc_violation -> the server must enforce it; FAIL when > 0
  model:         undeclared, nested_tool_called_directly, payload_invalid
                 -> usually what the model wrote; reported as rates. Prove the
                 attribution with the raw output before blaming either side
                 (--strict turns these into FAIL as well).

usage: check_tool_calls.py --out report.json [--workers N] [--requests PATH ...] PATH [PATH ...]
PATH: shards or directories. Request shards are read from --requests when
given, else from the same paths. Output carries tool names, classes, schema
keywords and JSON paths -- never argument values or text.

Exit code: 0 no check FAILs; 1 a check FAILs; 2 nothing could be judged, as a
shard cannot be read to its end (cut short or corrupt: the message names it, and
no report is written) or the arguments are wrong.
"""

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import re
import sys
import zlib
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profiles import DEFAULT_DEGENERATE, DEFAULT_MARKERS  # noqa: E402
from tracelib import (  # noqa: E402
    CALL_TYPES,
    ShardError,
    expand_shards,
    group_of,
    item_texts,
    iter_lines,
    load_json_line,
    terminal_object,
)

try:
    import jsonschema
except ImportError:  # schema validation is skipped and reported as such
    jsonschema = None
try:
    from referencing.exceptions import Unresolvable  # jsonschema 4.18+: a $ref it cannot resolve
except ImportError:
    Unresolvable = None
try:
    import lark
except ImportError:
    lark = None

# A declared schema no payload can be checked against: invalid, or a $ref that does not resolve.
SCHEMA_UNUSABLE = tuple(e for e in (getattr(jsonschema, "SchemaError", None), Unresolvable) if e)

NESTED_HEADER = re.compile(r"^### `([A-Za-z0-9_.-]+)`", re.M)
# A dotted identifier: what a qualified spelling of a declared tool looks like. A
# name with whitespace or markup in it is the model's mangled text, which the
# server must not "recover" (resolve_tool_name refuses prose), so it is undeclared.
QUALIFIED_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z_][A-Za-z0-9_-]*)+$")
QUOTES = "`'\""
SERVING_SHAPE = ("type_mismatch", "namespace_mismatch", "prefixed_name_of_declared")
MODEL_CLASSES = ("undeclared", "nested_tool_called_directly", "payload_invalid")
REPETITIVE_MIN_CHARS = 2000
REPETITIVE_MAX_ZLIB_RATIO = 0.10


def strict_json(text):
    """json.loads that rejects duplicate keys and non-finite numbers."""

    def pairs(items):
        keys = [k for k, _ in items]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate key")
        return dict(items)

    def reject_constant(name):
        raise ValueError("non-finite number " + name)

    def finite_float(literal):
        value = float(literal)
        if not math.isfinite(value):
            raise ValueError("number overflows to non-finite")
        return value

    return json.loads(
        text, object_pairs_hook=pairs, parse_constant=reject_constant, parse_float=finite_float
    )


def declarations(body):
    """{name: spec} from tools, namespace groups and additional_tools; plus '__nested__' and a key."""
    decl, sources = {}, []

    def walk(tools, namespace, source):
        for tool in tools or []:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") == "namespace":
                walk(tool.get("tools"), tool.get("name"), source)
                continue
            name = tool.get("name")
            if name is None and isinstance(tool.get("function"), dict):  # chat-completions shape
                name = tool["function"].get("name")
                tool = dict(tool["function"], type="function")
            decl[name] = {
                "type": tool.get("type"),
                "namespace": namespace,
                "source": source,
                "parameters": tool.get("parameters"),
                "format": tool.get("format"),
                "description": tool.get("description") or "",
            }

    if body.get("tools"):
        sources.append(body["tools"])
        walk(body["tools"], None, "tools")
    for item in body.get("input") or []:
        if isinstance(item, dict) and item.get("type") == "additional_tools":
            sources.append(item.get("tools"))
            walk(item.get("tools"), None, "additional_tools")
    nested = {}
    for name, spec in list(decl.items()):
        if spec["type"] == "custom":
            for m in NESTED_HEADER.finditer(spec["description"]):
                nested.setdefault(m.group(1), name)
    decl["__nested__"] = nested
    key = hashlib.sha256(
        json.dumps(sources, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return decl, key


_GRAMMARS = {}


def payload_check(payload, spec):
    """(ok, detail) for one call payload against its declaration. detail never holds a value."""
    if spec["type"] == "function":
        if not isinstance(payload, str):
            return False, {"reason": "arguments_not_string"}
        try:
            args = strict_json(payload) if payload.strip() else {}  # no arguments: as verdict.py
        except ValueError as error:
            return False, {"reason": "arguments_not_strict_json", "error": type(error).__name__}
        if not isinstance(args, dict):
            return False, {"reason": "arguments_not_object"}
        schema = spec.get("parameters")
        if isinstance(schema, dict):
            if jsonschema is None:
                return True, {"reason": "schema_unchecked_no_jsonschema"}
            try:
                jsonschema.validate(args, schema)
            except jsonschema.ValidationError as error:
                return False, {
                    "reason": "schema",
                    "validator": error.validator,
                    "path": "/".join(str(p) for p in error.absolute_path),
                }
            except SCHEMA_UNUSABLE:
                return True, {"reason": "declared_schema_invalid"}
        return True, None
    if spec["type"] == "custom":
        if not isinstance(payload, str):
            return False, {"reason": "input_not_string"}
        fmt = spec.get("format") or {}
        if fmt.get("type") == "grammar" and fmt.get("syntax") == "lark":
            if lark is None:
                return bool(payload), {"reason": "grammar_unchecked_no_lark"}
            grammar = fmt.get("definition") or ""
            if grammar not in _GRAMMARS:
                try:
                    _GRAMMARS[grammar] = lark.Lark(grammar, parser="earley")
                except lark.exceptions.LarkError:
                    _GRAMMARS[grammar] = None
            parser = _GRAMMARS[grammar]
            if parser is None:
                return True, {"reason": "declared_grammar_invalid"}
            try:
                parser.parse(payload)  # Earley: LALR's contextual lexer rejects valid inputs
            except lark.exceptions.LarkError as error:
                return False, {"reason": "grammar", "error": type(error).__name__}
            return True, None
        return (bool(payload), None if payload else {"reason": "empty_input"})
    return None, {"reason": "unsupported_declared_type", "declared_type": spec["type"]}


def check_call(call, decl):
    """One class for one call against its own request's declarations."""
    name = call.get("name")
    if name is not None and not isinstance(name, str):
        # A name that is not a string answers to no declaration.
        return {"name": name, "type": call["type"], "namespace": call.get("namespace"),
                "class": "undeclared", "detail": {"reason": "malformed_name"}}  # fmt: skip
    name = name or ""
    nested = decl.get("__nested__") or {}
    spec = decl.get(name) if name != "__nested__" else None
    result = {"name": name, "type": call["type"], "namespace": call.get("namespace")}
    if spec is None:
        qualified = bool(QUALIFIED_NAME.match(name))
        tail = name.rsplit(".", 1)[1] if qualified else None
        if name in nested or tail in nested:
            result["class"] = "nested_tool_called_directly"
            result["nested_in"] = nested.get(name) or nested.get(tail)
        elif tail and tail in decl and tail != "__nested__":
            result["class"] = "prefixed_name_of_declared"
        else:
            result["class"] = "undeclared"
            if name and not qualified and not re.match(r"^[A-Za-z0-9_-]+$", name):
                result["detail"] = {"reason": "malformed_name"}
        return result
    expected = {"function": "function_call", "custom": "custom_tool_call"}.get(spec["type"])
    if expected is None:
        result["class"] = "unsupported_declared_type"
        result["declared_type"] = spec["type"]
        return result
    if expected != call["type"]:
        result["class"] = "type_mismatch"
        result["declared_type"] = spec["type"]
        return result
    if call.get("namespace") is not None and call.get("namespace") != spec["namespace"]:
        result["class"] = "namespace_mismatch"
        result["declared_namespace"] = spec["namespace"]
        return result
    ok, detail = payload_check(call.get("payload"), spec)
    if detail:
        result["detail"] = detail
    result["class"] = "valid" if ok else "payload_invalid"
    return result


def marker_hits(text, markers):
    hits = []
    for rx in markers:
        for m in rx.finditer(text):
            before = text[m.start() - 1] if m.start() > 0 else ""
            after = text[m.end()] if m.end() < len(text) else ""
            hits.append(
                (m.group(0), bool(before) and bool(after) and before in QUOTES and after in QUOTES)
            )
    return hits


def ptc_class(body):
    if "parallel_tool_calls" not in body:
        return "absent"
    value = body["parallel_tool_calls"]
    if value is False:
        return "false"
    if value is True:
        return "true"
    return "null" if value is None else "other"


def safe_name(name):
    """A tool name fit for a report: verbatim when it is an identifier, else only its shape."""
    if isinstance(name, str) and re.match(r"^[A-Za-z0-9_.-]{1,64}$", name):
        return {"name": name}
    text = name if isinstance(name, str) else ""
    return {
        "name_sha256_12": hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:12],
        "name_chars": len(text),
    }


# Set in the parent before the pool forks; read-only in workers.
REQUESTS, DECLS, CFG = {}, {}, {}


def scan_requests(path):
    """Pass 1 over one request shard: trace_id -> (declaration key, ptc class, line)."""
    requests, decls, counts = {}, {}, Counter()
    for line_no, raw in iter_lines(path):
        rec = load_json_line(raw)
        if rec is None or rec.get("event", "request") != "request":
            continue
        body = rec.get("body")
        if not isinstance(body, dict):
            continue
        decl, key = declarations(body)
        decls.setdefault(key, decl)
        requests[rec.get("trace_id")] = (key, ptc_class(body), line_no)
        counts["requests_read"] += 1
        counts["ptc_request:" + ptc_class(body)] += 1
    return requests, decls, counts


def scan_responses(path):
    """Pass 2 over one response shard; returns (counts, counts by group, examples)."""
    group = group_of(path, CFG["group_depth"])
    counts, by_group, examples = Counter(), defaultdict(Counter), defaultdict(list)
    for line_no, raw in iter_lines(path):
        rec = load_json_line(raw)
        if rec is None:
            counts["unparsable_lines"] += 1
            continue
        kind, _, final = terminal_object(rec)
        if kind != "responses" or final is None:
            continue
        counts["responses_with_terminal_object"] += 1
        by_group[group]["responses_with_terminal_object"] += 1
        where = {"file": path, "line": line_no, "trace_id": rec.get("trace_id")}

        def note(check, **detail):
            counts[check] += 1
            by_group[group][check] += 1
            if len(examples[check]) < CFG["examples"]:
                examples[check].append(dict(where, **detail))

        calls, message, reasoning = [], [], []
        for item in final.get("output") or []:
            if not isinstance(item, dict):
                continue
            kind_ = item.get("type")
            if kind_ in CALL_TYPES:
                calls.append(
                    {
                        "type": kind_,
                        "name": item.get("name"),
                        "namespace": item.get("namespace"),
                        "call_id": item.get("call_id"),
                        "payload": item.get("arguments")
                        if kind_ == "function_call"
                        else item.get("input"),
                    }
                )
            elif kind_ == "message":
                message += item_texts(item)
            elif kind_ == "reasoning":
                reasoning += item_texts(item)
        mtext, rtext = "\n".join(message), "\n".join(reasoning)
        for rx in CFG["degenerate"]:
            if rx.search(mtext) or rx.search(rtext):
                note("screen:degenerate_run", pattern=rx.pattern)
                break
        if len(rtext) >= REPETITIVE_MIN_CHARS:
            raw_r = rtext.encode("utf-8", "surrogatepass")
            if len(zlib.compress(raw_r, 6)) / len(raw_r) < REPETITIVE_MAX_ZLIB_RATIO:
                note("screen:repetitive_reasoning", reasoning_chars=len(rtext))
        hits = marker_hits(mtext, CFG["markers"])
        if hits:
            note(
                "screen:markup_in_message:" + ("quoted" if all(q for _, q in hits) else "bare"),
                markers=sorted({h for h, _ in hits}),
            )
        payload_hits = marker_hits(
            "\n".join(c["payload"] for c in calls if isinstance(c["payload"], str)), CFG["markers"]
        )
        if payload_hits:
            note(
                "screen:markup_in_call:"
                + ("quoted" if all(q for _, q in payload_hits) else "bare"),
                markers=sorted({h for h, _ in payload_hits}),
            )
        if not calls:
            continue
        counts["responses_with_calls"] += 1
        by_group[group]["responses_with_calls"] += 1
        request = REQUESTS.get(rec.get("trace_id"))
        if request is None:
            note("request_not_found", calls=len(calls))
            continue
        key, ptc, _ = request
        if ptc == "false":
            counts["responses_with_calls_under_ptc_false"] += 1
            if len(calls) > 1:
                note("ptc_violation", calls=len(calls))
        for call in calls:
            counts["calls"] += 1
            by_group[group]["calls"] += 1
            result = check_call(call, DECLS[key])
            cls = result.pop("class")
            counts["call:" + cls] += 1
            by_group[group]["call:" + cls] += 1
            if cls != "valid" and len(examples["call:" + cls]) < CFG["examples"]:
                result.update(safe_name(result.pop("name")))
                examples["call:" + cls].append(dict(where, call_id=call.get("call_id"), **result))
    return counts, by_group, examples


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--requests", nargs="*", default=None, help="request shards/dirs (default: the paths)"
    )
    ap.add_argument(
        "--marker",
        action="append",
        default=None,
        help="regex of a format marker that must not reach delivered text (repeatable)",
    )
    ap.add_argument(
        "--degenerate",
        action="append",
        default=None,
        help="regex of a degenerate generation (repeatable); default a run of 20 [PAD]",
    )
    ap.add_argument("--strict", action="store_true", help="model-class findings also FAIL")
    ap.add_argument("--group-depth", type=int, default=0)
    ap.add_argument("--examples", type=int, default=20)
    ap.add_argument("--workers", type=int, default=1, help="processes (fork); one shard per task")
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()
    CFG.update(
        markers=[re.compile(m) for m in (args.marker or DEFAULT_MARKERS)],
        degenerate=[re.compile(d) for d in (args.degenerate or DEFAULT_DEGENERATE)],
        group_depth=args.group_depth,
        examples=args.examples,
    )

    request_shards = expand_shards(args.requests if args.requests else args.paths, "requests")
    response_shards = expand_shards(args.paths, "responses")
    counts, by_group = Counter(), defaultdict(Counter)
    examples = defaultdict(list)

    def run(fn, shards):
        if args.workers > 1 and len(shards) > 1:
            ctx = multiprocessing.get_context("fork")
            with ctx.Pool(min(args.workers, len(shards))) as pool:
                yield from pool.imap(fn, shards)
        else:
            yield from map(fn, shards)

    # Pass 1: requests -> compact summary; declarations deduplicated by content.
    for path, (requests, decls, part) in zip(request_shards, run(scan_requests, request_shards)):
        for trace_id, (key, ptc, line_no) in requests.items():
            REQUESTS[trace_id] = (key, ptc, (path, line_no))
        for key, decl in decls.items():
            DECLS.setdefault(key, decl)
        counts.update(part)
    # Pass 2: responses (workers inherit REQUESTS/DECLS through fork).
    for part, part_groups, part_examples in run(scan_responses, response_shards):
        counts.update(part)
        for group, c in part_groups.items():
            by_group[group].update(c)
        for check, rows in part_examples.items():
            room = args.examples - len(examples[check])
            if room > 0:
                examples[check].extend(rows[:room])

    denominator = counts["calls"]
    verdicts = {}
    for cls in SERVING_SHAPE + MODEL_CLASSES:
        n = counts["call:" + cls]
        if denominator == 0:
            verdicts[cls] = "BLOCKED"
        elif cls in SERVING_SHAPE or args.strict:
            verdicts[cls] = "PASS" if n == 0 else "FAIL"
        else:
            verdicts[cls] = "OBSERVED" if n else "PASS"
    if counts["responses_with_calls_under_ptc_false"] == 0:
        verdicts["ptc_violation"] = "NOT_APPLICABLE"
    else:
        verdicts["ptc_violation"] = "PASS" if counts["ptc_violation"] == 0 else "FAIL"
    verdicts["request_join"] = "PASS" if counts["request_not_found"] == 0 else "BLOCKED_PARTIAL"
    report = {
        "tool": "check_tool_calls",
        "inputs": {
            "request_shards": len(request_shards),
            "response_shards": len(response_shards),
            "distinct_declaration_sets": len(DECLS),
            "jsonschema": jsonschema is not None,
            "lark": lark is not None,
        },
        "counts": dict(counts),
        "rates": {
            k[5:]: round(v / denominator, 6)
            for k, v in counts.items()
            if k.startswith("call:") and denominator
        },
        "counts_by_group": {g: dict(c) for g, c in sorted(by_group.items())},
        "verdicts": verdicts,
        "examples": examples,
    }
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps({"calls": denominator, "verdicts": verdicts}))
    sys.exit(1 if any(v == "FAIL" for v in verdicts.values()) else 0)


if __name__ == "__main__":
    try:
        main()
    except ShardError as error:  # nothing is judged on a shard that cannot be read to its end
        print(error, file=sys.stderr)
        sys.exit(2)
