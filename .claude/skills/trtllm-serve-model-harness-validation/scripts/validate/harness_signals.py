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
"""Harness-side failure signals, read from the server's request trace of an agent run (read-only).

A harness reports what it did with each tool call in its NEXT request: the
``function_call_output`` / ``custom_tool_call_output`` items carry the result,
and for Codex a call it could not use comes back as

    aborted                       cancelled -- e.g. a custom tool delivered as function_call
    unsupported call: <name>      a call the harness does not know
    ... unavailable in <mode> mode
    failed to parse function arguments: ...   arguments that do not match the tool

A serving bug can show up only here: when the server delivers a custom tool as a
function call, Codex aborts the tool result. Server-side checks see well-formed
calls; the harness sees the wrong kind.

Per thread (``thread.thread_id`` / Codex turn metadata / session), each call_id's
output is counted once, the first time it appears (history replays it every
turn). Also counted: identical request bodies sent again in one thread (resends
after an error the harness did not understand), the response transport status
(client disconnects), HTTP errors and terminal failure codes.

usage: harness_signals.py --out signals.json PATH [PATH ...]
PATH: request/response shards or directories (request-trace layout).
A shard that cannot be read to its end stops the run with a message naming it.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracelib import ShardError, iter_lines  # noqa: E402

SHARD = re.compile(r"(^|/)(requests|responses)-[^/]*\.jsonl(\.zst|\.gz)?$")
OUTPUT_TYPES = ("function_call_output", "custom_tool_call_output")
UNAVAILABLE = re.compile(r"is unavailable in (\w+) mode", re.I)


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
        elif SHARD.search(path):
            found.append(path)
        else:
            # The name says which side a shard holds: a file named otherwise cannot be read.
            raise SystemExit(f"{path}: not a requests-* or responses-* shard")
    return sorted(set(found))


def thread_of(rec, body):
    thread = rec.get("thread")
    if isinstance(thread, dict) and thread.get("thread_id"):
        return str(thread["thread_id"])
    meta = body.get("client_metadata") if isinstance(body.get("client_metadata"), dict) else {}
    if meta.get("thread_id"):
        return str(meta["thread_id"])
    return str(rec.get("session") or body.get("prompt_cache_key") or rec.get("trace_id"))


def output_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            p.get("text", "")
            for p in value
            if isinstance(p, dict) and isinstance(p.get("text"), str)
        )
    return ""


def classify(text):
    stripped = text.strip()
    if stripped == "aborted":
        return "aborted"
    if stripped.startswith("unsupported call:"):
        return "unsupported_call"
    if stripped.startswith("failed to parse function arguments"):
        return "arguments_rejected"
    if len(stripped) < 300 and UNAVAILABLE.search(stripped):
        return "unavailable_in_mode"
    return "result"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()

    seen_calls = defaultdict(set)
    bodies = defaultdict(Counter)
    per_thread = defaultdict(Counter)
    totals = Counter()
    for path in expand(args.paths):
        kind = SHARD.search(path).group(2)
        for _, raw in iter_lines(path):
            try:
                rec = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                totals["unparsable_lines"] += 1
                continue
            if not isinstance(rec, dict):
                continue
            if kind == "responses":
                totals["responses"] += 1
                totals["response_status:" + str(rec.get("status"))] += 1
                if isinstance(rec.get("http_status"), int) and rec["http_status"] >= 400:
                    totals[f"http_{rec['http_status']}"] += 1
                body = (rec.get("response") or {}).get("body")
                if isinstance(body, str) and "response.failed" in body[-6000:]:
                    for code in re.findall(r'"code"\s*:\s*"([a-z_]{3,60})"', body[-6000:]):
                        totals["failed_code:" + code] += 1
                continue
            body = rec.get("body")
            if not isinstance(body, dict):
                continue
            totals["requests"] += 1
            thread = thread_of(rec, body)
            per_thread[thread]["requests"] += 1
            digest = (
                rec.get("body_sha256")
                or hashlib.sha256(
                    json.dumps(body, sort_keys=True, ensure_ascii=False).encode()
                ).hexdigest()
            )
            bodies[thread][digest] += 1
            if bodies[thread][digest] == 2:
                per_thread[thread]["resent_bodies"] += 1
                totals["resent_bodies"] += 1
            for item in body.get("input") or []:
                if not isinstance(item, dict) or item.get("type") not in OUTPUT_TYPES:
                    continue
                call_id = item.get("call_id")
                if call_id in seen_calls[thread]:
                    continue
                seen_calls[thread].add(call_id)
                cls = classify(output_text(item.get("output")))
                per_thread[thread]["call_outputs"] += 1
                per_thread[thread][cls] += 1
                totals["call_outputs"] += 1
                totals["call_output:" + cls] += 1

    outputs = totals["call_outputs"]
    rates = {
        k.split(":", 1)[1]: round(v / outputs, 6)
        for k, v in totals.items()
        if k.startswith("call_output:") and outputs
    }
    worst = sorted(
        ((t, c) for t, c in per_thread.items() if c["aborted"] or c["unsupported_call"]),
        key=lambda tc: -(tc[1]["aborted"] + tc[1]["unsupported_call"]),
    )[:20]
    report = {
        "tool": "harness_signals",
        "threads": len(per_thread),
        "totals": dict(totals),
        "rates": rates,
        "threads_with_rejected_calls": len(
            [1 for c in per_thread.values() if c["aborted"] or c["unsupported_call"]]
        ),
        "worst_threads": [
            {"thread_sha256_12": hashlib.sha256(t.encode()).hexdigest()[:12], **dict(c)}
            for t, c in worst
        ],
    }
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(
        json.dumps(
            {
                "threads": report["threads"],
                "call_outputs": outputs,
                "rates": rates,
                "resent_bodies": totals["resent_bodies"],
            }
        )
    )


if __name__ == "__main__":
    try:
        main()
    except ShardError as error:  # signals from part of a shard would read as all of them
        sys.exit(str(error))
