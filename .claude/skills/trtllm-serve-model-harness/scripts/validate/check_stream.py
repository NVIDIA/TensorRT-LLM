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
"""Stream/snapshot conformance of streamed Responses replies (read-only).

For every response record whose body was streamed (``response.kind == "sse_text"``)
and speaks the Responses protocol, check that the stream is self-consistent and
agrees with its own terminal snapshot:

  no_terminal_event         the stream never reached completed/incomplete/failed
  unparsable_event          a frame whose data is not JSON (an SDK client fails on it)
  multiple_terminal_events  more than one terminal event
  events_after_terminal     frames after the terminal event
  sequence_not_increasing   ``sequence_number`` not strictly increasing
  response_id_changes       opening events and terminal event name different ids
  model_changes             ... or different models
  unpaired_output_item      an ``output_item.added`` without its ``done`` (or the reverse)
  unpaired_content_part     a ``content_part.added`` without its ``done`` (or the reverse)
  delta_done_mismatch       concatenated ``*.delta`` != the matching ``*.done`` payload
                            (output text, reasoning text, function arguments, custom input)
  done_item_mismatch        a ``*.done`` payload != the text/arguments/input of ``output_item.done``
  item_list_differs         (type, id) list of ``output_item.done`` != the snapshot's ``output``
  item_content_differs      per item id: text, name, call_id, arguments or input differ
  item_status_missing       a snapshot message without ``status``, or a function_call /
                            reasoning item whose ``status`` is an explicit null

``failed_snapshot_without_streamed_items`` is counted, not failed: a
``response.failed`` snapshot carries no generated output by design. Chat
completion streams are counted as ``non_responses_streams`` and skipped.

Output: a JSON report with denominators, counts per check and per group, up to
--examples locators per check, and a per-check verdict (PASS when the count is
0, FAIL otherwise, BLOCKED when nothing was checked). Locators never carry
content: differing texts are described by lengths, first differing offset and
SHA-256 of each side. ``--excerpt N`` opts in to N characters around the first
difference, for synthetic or already-cleared data only.

usage: check_stream.py --out report.json [--workers N] [--group-depth N] [--examples K] PATH [PATH ...]
PATH: response shards or directories holding them (requests-* files are ignored).

Exit code: 0 no check FAILs; 1 a check FAILs; 2 nothing could be judged, as a
shard cannot be read to its end (cut short or corrupt: the message names it, and
no report is written) or the arguments are wrong.
"""

import argparse
import json
import multiprocessing
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracelib import (  # noqa: E402
    OPENING_EVENTS,
    TERMINAL_EVENTS,
    ShardError,
    expand_shards,
    first_difference,
    group_of,
    is_chat_stream,
    iter_lines,
    load_json_line,
    sha256_text,
    sse_events,
)

CHECKS = (
    "no_terminal_event",
    "unparsable_event",
    "multiple_terminal_events",
    "events_after_terminal",
    "sequence_not_increasing",
    "response_id_changes",
    "model_changes",
    "unpaired_output_item",
    "unpaired_content_part",
    "delta_done_mismatch",
    "done_item_mismatch",
    "item_list_differs",
    "item_content_differs",
    "item_status_missing",
)
# (delta event, done event, field holding the full payload on the done event,
#  whether the payload is per content part)
DELTA_KINDS = (
    ("response.output_text.delta", "response.output_text.done", "text", True),
    ("response.reasoning_text.delta", "response.reasoning_text.done", "text", True),
    (
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "arguments",
        False,
    ),
    (
        "response.custom_tool_call_input.delta",
        "response.custom_tool_call_input.done",
        "input",
        False,
    ),
)
# Item types whose snapshot status a client reads. In the OpenAI types a message
# requires ``status``; function_call and reasoning items may omit it but must not
# carry an explicit null; custom_tool_call has no status field.
STATUS_REQUIRED = ("message",)
STATUS_NOT_NULL = ("function_call", "reasoning")


def item_view(item):
    kind = item.get("type")
    if kind in ("message", "reasoning"):
        parts = item.get("content") or []
        if isinstance(parts, str):
            return {"text": parts}
        return {"text": "".join((p or {}).get("text") or "" for p in parts if isinstance(p, dict))}
    if kind == "function_call":
        return {k: item.get(k) for k in ("name", "call_id", "arguments", "namespace")}
    if kind == "custom_tool_call":
        return {k: item.get(k) for k in ("name", "call_id", "input", "namespace")}
    return {}


def describe(a, b, excerpt):
    """Locator detail for two differing strings: lengths, offset, hashes; content only on opt-in."""
    if not (isinstance(a, str) and isinstance(b, str)):
        return {"left_type": type(a).__name__, "right_type": type(b).__name__}
    at = first_difference(a, b)
    detail = {
        "left_chars": len(a),
        "right_chars": len(b),
        "first_difference_at": at,
        "left_sha256": sha256_text(a),
        "right_sha256": sha256_text(b),
    }
    if excerpt:
        lo = max(0, at - excerpt // 2)
        detail["left_excerpt"] = a[lo : at + excerpt // 2]
        detail["right_excerpt"] = b[lo : at + excerpt // 2]
    return detail


def check_record(events, note):
    """Run every check on one Responses stream. ``note(check, **detail)`` records a finding."""
    terminal_positions = [
        i for i, (n, d) in enumerate(events) if n in TERMINAL_EVENTS and isinstance(d, dict)
    ]
    if not terminal_positions:
        note("no_terminal_event")
        return None
    if len(terminal_positions) > 1:
        note("multiple_terminal_events", count=len(terminal_positions))
    last = terminal_positions[-1]
    if last != len(events) - 1:
        note("events_after_terminal", frames_after=len(events) - 1 - last)
    name, data = events[last]
    final = data.get("response") if isinstance(data.get("response"), dict) else {}

    seq = [
        d.get("sequence_number")
        for _, d in events
        if isinstance(d, dict) and "sequence_number" in d
    ]
    if any(b is None or a is None or b <= a for a, b in zip(seq, seq[1:])):
        note("sequence_not_increasing")

    opening = [
        d.get("response") or {} for n, d in events if n in OPENING_EVENTS and isinstance(d, dict)
    ]
    ids = {o.get("id") for o in opening} | {final.get("id")}
    if len(ids - {None}) > 1:
        note("response_id_changes", ids=sorted(i for i in ids if i))
    models = {o.get("model") for o in opening} | {final.get("model")}
    if len(models - {None}) > 1:
        note("model_changes", models=sorted(m for m in models if m))

    added_items, done_items = Counter(), Counter()
    added_parts, done_parts = Counter(), Counter()
    deltas = defaultdict(list)
    dones = {}
    done_list = []
    done_item_by_id = {}
    for n, d in events:
        if not isinstance(d, dict):
            continue
        if n == "response.output_item.added":
            added_items[(d.get("item") or {}).get("id")] += 1
        elif n == "response.output_item.done":
            item = d.get("item") or {}
            done_items[item.get("id")] += 1
            done_list.append(item)
            done_item_by_id[item.get("id")] = item
        elif n == "response.content_part.added":
            added_parts[(d.get("item_id"), d.get("content_index"))] += 1
        elif n == "response.content_part.done":
            done_parts[(d.get("item_id"), d.get("content_index"))] += 1
        else:
            for delta_name, done_name, field, per_part in DELTA_KINDS:
                key = (done_name, d.get("item_id"), d.get("content_index") if per_part else None)
                if n == delta_name:
                    deltas[key].append(d.get("delta") or "")
                elif n == done_name:
                    dones[key] = d.get(field)
    if added_items != done_items:
        note("unpaired_output_item", added=sum(added_items.values()), done=sum(done_items.values()))
    if added_parts != done_parts:
        note(
            "unpaired_content_part", added=sum(added_parts.values()), done=sum(done_parts.values())
        )

    for key, payload in dones.items():
        done_name, item_id, content_index = key
        if key in deltas:
            joined = "".join(deltas[key])
            if joined != payload:
                note(
                    "delta_done_mismatch",
                    event=done_name,
                    item_id=item_id,
                    **describe(joined, payload, 0),
                )
        item = done_item_by_id.get(item_id)
        if item is None:
            continue
        if done_name.endswith("arguments.done"):
            other = item.get("arguments")
        elif done_name.endswith("input.done"):
            other = item.get("input")
        else:
            parts = item.get("content") or []
            part = (
                parts[content_index]
                if isinstance(parts, list)
                and isinstance(content_index, int)
                and content_index < len(parts)
                else None
            )
            other = (part or {}).get("text") if isinstance(part, dict) else None
        if other is not None and other != payload:
            note(
                "done_item_mismatch",
                event=done_name,
                item_id=item_id,
                **describe(payload, other, 0),
            )
    return name, final, done_list, done_item_by_id


CFG = {}


def scan_shard(path):
    """All checks over one response shard; returns (counts, counts by group, examples)."""
    group = group_of(path, CFG["group_depth"])
    counts, by_group, examples = Counter(), defaultdict(Counter), defaultdict(list)
    for line_no, raw in iter_lines(path):
        rec = load_json_line(raw)
        if rec is None:
            counts["unparsable_lines"] += 1
            continue
        resp = rec.get("response")
        if not isinstance(resp, dict) or resp.get("kind") != "sse_text":
            continue
        counts["streamed_records"] += 1
        by_group[group]["streamed_records"] += 1
        events = list(sse_events(resp.get("body") or ""))
        if is_chat_stream(events):
            counts["non_responses_streams"] += 1
            by_group[group]["non_responses_streams"] += 1
            continue
        counts["responses_streams"] += 1
        by_group[group]["responses_streams"] += 1
        where = {"file": path, "line": line_no, "trace_id": rec.get("trace_id")}

        def note(check, **detail):
            counts[check] += 1
            by_group[group][check] += 1
            if len(examples[check]) < CFG["examples"]:
                examples[check].append(dict(where, **detail))

        if any(d is None for _, d in events):
            note("unparsable_event")
        result = check_record(events, note)
        if result is None:
            note_status = rec.get("status")
            if note_status is not None:
                counts["no_terminal_event_status:" + str(note_status)] += 1
            continue
        name, final, done_list, done_item_by_id = result
        counts["terminal:" + name] += 1
        by_group[group]["terminal:" + name] += 1
        final_items = [i for i in final.get("output") or [] if isinstance(i, dict)]
        done_keys = [(i.get("type"), i.get("id")) for i in done_list]
        final_keys = [(i.get("type"), i.get("id")) for i in final_items]
        if done_keys != final_keys:
            if name == "response.failed" and not final_keys:
                counts["failed_snapshot_without_streamed_items"] += 1
                by_group[group]["failed_snapshot_without_streamed_items"] += 1
            else:
                note(
                    "item_list_differs",
                    terminal=name,
                    streamed=len(done_keys),
                    final=len(final_keys),
                )
        for item in final_items:
            kind = item.get("type")
            missing = (kind in STATUS_REQUIRED and item.get("status") is None) or (
                kind in STATUS_NOT_NULL and "status" in item and item["status"] is None
            )
            if missing:
                note("item_status_missing", item_id=item.get("id"), item_type=kind)
                break
        for item in final_items:
            streamed = done_item_by_id.get(item.get("id"))
            if streamed is None:
                continue
            a, b = item_view(streamed), item_view(item)
            for field in sorted(set(a) | set(b)):
                if a.get(field) != b.get(field):
                    note(
                        "item_content_differs",
                        terminal=name,
                        item_id=item.get("id"),
                        item_type=item.get("type"),
                        field=field,
                        **describe(a.get(field), b.get(field), CFG["excerpt"]),
                    )
    return counts, by_group, examples


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", required=True, help="JSON report path")
    ap.add_argument("--group-depth", type=int, default=0, help="see tracelib.group_of")
    ap.add_argument("--examples", type=int, default=20, help="locators kept per check")
    ap.add_argument(
        "--excerpt",
        type=int,
        default=0,
        help="characters of content around a difference (opt-in; synthetic data only)",
    )
    ap.add_argument("--workers", type=int, default=1, help="processes (fork); one shard per task")
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()
    CFG.update(group_depth=args.group_depth, examples=args.examples, excerpt=args.excerpt)

    shards = expand_shards(args.paths, "responses")
    counts, by_group = Counter(), defaultdict(Counter)
    examples = defaultdict(list)
    if args.workers > 1 and len(shards) > 1:
        pool = multiprocessing.get_context("fork").Pool(min(args.workers, len(shards)))
        parts = pool.imap(scan_shard, shards)
    else:
        pool, parts = None, map(scan_shard, shards)
    for part, part_groups, part_examples in parts:
        counts.update(part)
        for group, c in part_groups.items():
            by_group[group].update(c)
        for check, rows in part_examples.items():
            room = args.examples - len(examples[check])
            if room > 0:
                examples[check].extend(rows[:room])
    if pool is not None:
        pool.close()
        pool.join()

    checked = counts["responses_streams"]
    verdicts = {}
    for check in CHECKS:
        if checked == 0:
            verdicts[check] = "BLOCKED"
        else:
            verdicts[check] = "PASS" if counts[check] == 0 else "FAIL"
    report = {
        "tool": "check_stream",
        "inputs": {"shards": len(shards), "paths": args.paths},
        "denominators": {
            k: counts[k] for k in ("streamed_records", "responses_streams", "non_responses_streams")
        },
        "counts": dict(counts),
        "counts_by_group": {g: dict(c) for g, c in sorted(by_group.items())},
        "verdicts": verdicts,
        "examples": examples,
    }
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(
        json.dumps(
            {
                "denominators": report["denominators"],
                "failing": {c: counts[c] for c in CHECKS if counts[c]},
            }
        )
    )
    if any(v == "FAIL" for v in verdicts.values()):
        sys.exit(1)
    # Every check BLOCKED (e.g. no stream to read) judged nothing, which is not a pass.
    sys.exit(2 if verdicts and all(v == "BLOCKED" for v in verdicts.values()) else 0)


if __name__ == "__main__":
    try:
        main()
    except ShardError as error:  # nothing is judged on a shard that cannot be read to its end
        print(error, file=sys.stderr)
        sys.exit(2)
