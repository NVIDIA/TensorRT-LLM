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
"""Summarize what an agent harness actually sends and receives -- the skeleton of its contract.

Reads request-trace records (record_requests.py output, or the trace of a server-side trace
writer, ``TRTLLM_REQUEST_TRACE_DIR``, that is not on main) and reports the wire surface the
harness uses:

  routes            method/path counts
  body keys         per key: how often present and the JSON types seen
  categorical       values of small enumerated fields (stream, parallel_tool_calls,
                    store, tool_choice, service_tier, reasoning.*, text.*, include, ...)
  tools             declared tool types; per tool name: type, namespace, where
                    declared (tools / additional_tools), schema or grammar present
  input items       item types, message roles, content part types per role,
                    tool-output shapes (string vs content parts)
  metadata          client_metadata keys (and keys of JSON-encoded values inside it)
  headers           header names; values only for an allowlist (user-agent, originator,
                    openai-beta, accept, content-type, version, x-stainless-*, and the
                    identity headers session_id and x-client-request-id, whose values show
                    whether they name a session or a request), never auth
  responses         HTTP statuses, reply kinds, terminal events, error codes and the
                    most common SSE event-type sequences (repeated deltas collapsed)

No free text, argument value or credential is printed: tool names, enumerations,
counts and types only. Output: JSON report plus a markdown skeleton to complete
by hand (see contracts/TEMPLATE.md).

usage: summarize_capture.py --out summary.json --markdown contract-skeleton.md PATH [PATH ...]
A shard that cannot be read to its end stops the run with a message naming it.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "validate"))
from tracelib import ShardError, iter_lines  # noqa: E402

SHARD = re.compile(r"(^|/)(requests|responses)-[^/]*\.jsonl(\.zst|\.gz)?$")
CATEGORICAL = (
    "stream",
    "parallel_tool_calls",
    "store",
    "tool_choice",
    "service_tier",
    "truncation",
    "background",
    "reasoning.effort",
    "reasoning.summary",
    "reasoning.generate_summary",
    "text.verbosity",
    "text.format.type",
    "include",
    "stream_options.include_usage",
    "user",
)
SAFE_HEADERS = re.compile(
    r"^(user-agent|originator|openai-beta|accept|content-type|x-stainless-[a-z-]+|"
    r"version|session_id|x-client-request-id)$",
    re.I,
)
TERMINAL = ("response.completed", "response.incomplete", "response.failed")
TOP_SEQUENCES = 10


def expand(paths):
    found = []
    for path in paths:
        if os.path.isdir(path):
            for dirpath, _, names in os.walk(path, followlinks=True):
                found += [
                    os.path.join(dirpath, n)
                    for n in names
                    if SHARD.search(os.path.join(dirpath, n))
                ]
        else:
            found.append(path)
    return sorted(set(found))


def jtype(value):
    if value is None:
        return "null"
    return {bool: "bool", int: "int", float: "float", str: "str", list: "list", dict: "object"}.get(
        type(value), "other"
    )


def dig(body, dotted):
    value = body
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None, False
        value = value[part]
    return value, True


def category(value):
    """A categorical value fit for a report: scalars short enough to be an enum, lists of them."""
    if isinstance(value, (bool, int, float)) or value is None:
        return json.dumps(value)
    if isinstance(value, str):
        return (
            value
            if len(value) <= 40 and re.match(r"^[\w.:/-]*$", value)
            else f"<str len={len(value)}>"
        )
    if isinstance(value, list):
        return json.dumps(sorted(category(v) for v in value))
    if isinstance(value, dict):
        return "<object keys=" + ",".join(sorted(value)) + ">"
    return "<other>"


def walk_tools(tools, namespace, source, out):
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "namespace":
            walk_tools(tool.get("tools"), tool.get("name"), source, out)
            continue
        fmt = tool.get("format") if isinstance(tool.get("format"), dict) else {}
        name = (
            tool.get("name") or (tool.get("function") or {}).get("name") or f"<{tool.get('type')}>"
        )
        out[
            (
                name if re.match(r"^[\w.-]{1,64}$", name) else "<unprintable>",
                tool.get("type"),
                namespace,
                source,
                bool(tool.get("parameters") or (tool.get("function") or {}).get("parameters")),
                fmt.get("type"),
                fmt.get("syntax"),
            )
        ] += 1


def sse_sequence(body):
    names = []
    for block in body.split("\n\n"):
        name = None
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:") and name is None:
                text = line[5:].strip()
                if text == "[DONE]":
                    name = "[DONE]"
                else:
                    try:
                        name = (json.loads(text) or {}).get("type") or (
                            "chat.chunk" if '"choices"' in text else "?"
                        )
                    except json.JSONDecodeError:
                        name = "?"
        if name:
            if names and names[-1] in (name, name + "*") and name.endswith(".delta"):
                names[-1] = name + "*"
            elif not (names and names[-1] == name == "chat.chunk"):
                names.append(name)
    return names


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--markdown")
    ap.add_argument(
        "--max-records", type=int, default=0, help="stop after N request records (0 = all)"
    )
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()

    stats = {
        "routes": Counter(),
        "body_keys": Counter(),
        "body_key_types": defaultdict(Counter),
        "categorical": defaultdict(Counter),
        "tool_types": Counter(),
        "tools": Counter(),
        "input_item_types": Counter(),
        "message_roles": Counter(),
        "content_parts": Counter(),
        "tool_output_shape": Counter(),
        "client_metadata_keys": Counter(),
        "metadata_inner_keys": Counter(),
        "header_names": Counter(),
        "header_values": defaultdict(Counter),
        "http_status": Counter(),
        "reply_kind": Counter(),
        "terminal": Counter(),
        "error_codes": Counter(),
        "sequences": Counter(),
        "status": Counter(),
    }
    requests = responses = 0
    for path in expand(args.paths):
        is_request = "/requests-" in path or os.path.basename(path).startswith("requests-")
        for _, raw in iter_lines(path):
            try:
                rec = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(rec, dict):
                continue
            if is_request:
                if args.max_records and requests >= args.max_records:
                    continue
                requests += 1
                stats["routes"][f"{rec.get('method', 'POST')} {rec.get('route')}"] += 1
                stats["status"]["request:" + str(rec.get("status"))] += 1
                headers = rec.get("headers") or []
                pairs = (
                    headers.items()
                    if isinstance(headers, dict)
                    else [h for h in headers if isinstance(h, (list, tuple)) and len(h) == 2]
                )
                for name, value in pairs:
                    stats["header_names"][str(name).lower()] += 1
                    if SAFE_HEADERS.match(str(name)) and isinstance(value, str):
                        stats["header_values"][str(name).lower()][value[:80]] += 1
                body = rec.get("body")
                if not isinstance(body, dict):
                    continue
                for key, value in body.items():
                    stats["body_keys"][key] += 1
                    stats["body_key_types"][key][jtype(value)] += 1
                for dotted in CATEGORICAL:
                    value, present = dig(body, dotted)
                    stats["categorical"][dotted][category(value) if present else "<absent>"] += 1
                for tool in body.get("tools") or []:
                    if isinstance(tool, dict):
                        stats["tool_types"][str(tool.get("type"))] += 1
                walk_tools(body.get("tools"), None, "tools", stats["tools"])
                meta = body.get("client_metadata")
                if isinstance(meta, dict):
                    for key, value in meta.items():
                        stats["client_metadata_keys"][key] += 1
                        if isinstance(value, str) and value.startswith("{"):
                            try:
                                inner = json.loads(value)
                            except json.JSONDecodeError:
                                inner = None
                            if isinstance(inner, dict):
                                for k in inner:
                                    stats["metadata_inner_keys"][f"{key}.{k}"] += 1
                items = body.get("input")
                if isinstance(items, str):
                    stats["input_item_types"]["<string input>"] += 1
                for item in items if isinstance(items, list) else []:
                    if not isinstance(item, dict):
                        continue
                    kind = item.get("type") or ("message" if "role" in item else "?")
                    stats["input_item_types"][kind] += 1
                    if kind == "additional_tools":
                        walk_tools(item.get("tools"), None, "additional_tools", stats["tools"])
                    if kind == "message":
                        role = item.get("role")
                        stats["message_roles"][str(role)] += 1
                        content = item.get("content")
                        if isinstance(content, str):
                            stats["content_parts"][f"{role}:<string>"] += 1
                        for part in content if isinstance(content, list) else []:
                            if isinstance(part, dict):
                                stats["content_parts"][f"{role}:{part.get('type')}"] += 1
                    if kind in ("function_call_output", "custom_tool_call_output"):
                        output = item.get("output")
                        shape = (
                            "string"
                            if isinstance(output, str)
                            else (
                                "parts:"
                                + ",".join(
                                    sorted(
                                        {str(p.get("type")) for p in output if isinstance(p, dict)}
                                    )
                                )
                                if isinstance(output, list)
                                else jtype(output)
                            )
                        )
                        stats["tool_output_shape"][f"{kind}:{shape}"] += 1
            else:
                responses += 1
                stats["status"]["response:" + str(rec.get("status"))] += 1
                if rec.get("http_status") is not None:
                    stats["http_status"][str(rec.get("http_status"))] += 1
                resp = rec.get("response") if isinstance(rec.get("response"), dict) else {}
                stats["reply_kind"][str(resp.get("kind"))] += 1
                body = resp.get("body")
                if resp.get("kind") == "sse_text" and isinstance(body, str):
                    seq = sse_sequence(body)
                    stats["sequences"][" > ".join(seq)] += 1
                    for name in seq:
                        if name.rstrip("*") in TERMINAL:
                            stats["terminal"][name.rstrip("*")] += 1
                    for match in re.finditer(r'"code"\s*:\s*"([a-z_]{3,60})"', body[-4000:]):
                        stats["error_codes"][match.group(1)] += 1
                elif isinstance(body, dict):
                    error = body.get("error")
                    if isinstance(error, dict) and isinstance(error.get("code"), str):
                        stats["error_codes"][error["code"][:60]] += 1

    report = {"requests": requests, "responses": responses}
    for key, value in stats.items():
        if key == "tools":
            report["tools"] = [
                dict(
                    zip(
                        (
                            "name",
                            "type",
                            "namespace",
                            "declared_in",
                            "has_parameters",
                            "format_type",
                            "format_syntax",
                        ),
                        k,
                    ),
                    records=n,
                )
                for k, n in value.most_common()
            ]
        elif key == "sequences":
            report["top_sse_sequences"] = [
                {"count": n, "events": s} for s, n in value.most_common(TOP_SEQUENCES)
            ]
        elif isinstance(value, defaultdict):
            report[key] = {k: dict(v.most_common(20)) for k, v in sorted(value.items())}
        else:
            report[key] = dict(value.most_common())
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    if args.markdown:
        with open(args.markdown, "w") as md:
            md.write(
                f"# Harness contract skeleton\n\nFrom {requests} requests and {responses} responses.\n\n"
            )
            md.write(
                "## Wire\n\n"
                + "\n".join(f"- `{k}`: {v}" for k, v in report["routes"].items())
                + "\n\n"
            )
            md.write("## Request body keys (present in N requests; JSON types)\n\n")
            for key, n in report["body_keys"].items():
                md.write(f"- `{key}`: {n} ({', '.join(report['body_key_types'].get(key, {}))})\n")
            md.write("\n## Enumerated fields\n\n")
            for key, values in report["categorical"].items():
                md.write(f"- `{key}`: " + ", ".join(f"{v} x{n}" for v, n in values.items()) + "\n")
            md.write(
                "\n## Tools as declared\n\n| name | type | namespace | declared in | params | format |\n"
                "|---|---|---|---|---|---|\n"
            )
            for t in report["tools"][:60]:
                md.write(
                    f"| {t['name']} | {t['type']} | {t['namespace']} | {t['declared_in']} | "
                    f"{t['has_parameters']} | {t['format_type'] or ''} {t['format_syntax'] or ''} |\n"
                )
            md.write(
                "\n## Input items\n\n"
                + json.dumps(
                    {
                        k: report[k]
                        for k in (
                            "input_item_types",
                            "message_roles",
                            "content_parts",
                            "tool_output_shape",
                        )
                    },
                    indent=1,
                )
                + "\n"
            )
            md.write(
                "\n## Metadata and headers\n\n"
                + json.dumps(
                    {
                        k: report[k]
                        for k in (
                            "client_metadata_keys",
                            "metadata_inner_keys",
                            "header_names",
                            "header_values",
                        )
                    },
                    indent=1,
                )
                + "\n"
            )
            md.write(
                "\n## Replies\n\n"
                + json.dumps(
                    {
                        k: report[k]
                        for k in (
                            "http_status",
                            "reply_kind",
                            "terminal",
                            "error_codes",
                            "top_sse_sequences",
                        )
                    },
                    indent=1,
                )
                + "\n"
            )
            md.write(
                "\n## To complete by hand\n\nSee contracts/TEMPLATE.md: client-side semantics "
                "(timeouts, retries, compaction trigger, how tool results are matched, what is replayed).\n"
            )
    print(json.dumps({"requests": requests, "responses": responses, "tools": len(report["tools"])}))


if __name__ == "__main__":
    try:
        main()
    except ShardError as error:  # a summary of part of a shard would read as all of it
        sys.exit(str(error))
