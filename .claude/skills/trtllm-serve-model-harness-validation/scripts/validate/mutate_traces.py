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
"""Seeded faults on REAL Responses traffic: negative controls next to a normal control.

The fake-server matrix (controls.py) proves each check fires on synthetic
streams. This proves it on production-shaped ones: a sample of recorded replies
whose calls are valid is copied, one fault per copy, in the shape of a known
serving bug, and ``verdict.py`` must find each fault with its own check while
the unmutated sample stays clean.

  custom_as_function   a custom tool delivered as function_call (Codex aborts it)  -> C-06
  prefixed_name        functions.exec instead of exec                              -> C-06
  second_call          a second call under parallel_tool_calls=false               -> C-07
  snapshot_differs     the terminal snapshot's text differs from the stream        -> C-02
  no_terminal          the terminal event is lost                                  -> C-04

The copies hold trace content: they are written under a 0700 directory, never
printed, and removed at exit unless --keep. Only locators and counts are reported.

usage: mutate_traces.py --harness H.yaml --model M.yaml --sample N --out-dir DIR
                        --requests REQ_DIR [...] -- RESPONSE_DIR [...]
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import verdict as verdict_mod  # noqa: E402
from tracelib import expand_shards, iter_lines, load_json_line, sse_frames  # noqa: E402

EXPECT = {
    "none": set(),
    "custom_as_function": {"C-06"},
    "prefixed_name": {"C-06"},
    "second_call": {"C-07"},
    "snapshot_differs": {"C-02", "RSP-11"},
    "no_terminal": {"C-04", "RSP-02"},
}
CHECKS_OF_INTEREST = {"C-01", "C-02", "C-04", "C-05", "C-06", "C-07", "C-08"}


def _events(body):
    return [
        (f.name or (f.obj or {}).get("type"), f.obj) for f in sse_frames(body) if f.kind == "json"
    ]


def _serialize(events):
    return "".join(f"event: {n}\ndata: {json.dumps(o, ensure_ascii=False)}\n\n" for n, o in events)


def _is_candidate(record):
    """A completed Responses stream with exactly one custom_tool_call: every fault applies."""
    body = (record.get("response") or {}).get("body")
    if record.get("status") != "completed" or not isinstance(body, str):
        return False
    events = _events(body)
    if not events or events[-1][0] != "response.completed":
        return False
    output = (events[-1][1].get("response") or {}).get("output") or []
    calls = [i for i in output if isinstance(i, dict) and i.get("type") == "custom_tool_call"]
    texts = [i for i in output if isinstance(i, dict) and i.get("type") == "message"]
    return (
        len(calls) == 1
        and len(texts) >= 1
        and len(output)
        == len(calls)
        + len(texts)
        + len([i for i in output if isinstance(i, dict) and i.get("type") == "reasoning"])
    )


def mutate(record, fault):
    record = json.loads(json.dumps(record))
    events = _events(record["response"]["body"])
    final = events[-1][1]["response"]

    def each_call(fn):
        for name, obj in events:
            item = obj.get("item") if isinstance(obj, dict) else None
            if isinstance(item, dict) and item.get("type") == "custom_tool_call":
                fn(item)
        for item in final.get("output") or []:
            if item.get("type") == "custom_tool_call":
                fn(item)

    if fault == "custom_as_function":
        # The whole input goes into every copy of the item, so the stream stays internally
        # consistent (one fault, C-06) for the adapter and for an SDK client alike.
        whole = next(i.get("input", "") for i in final.get("output") or []
                     if i.get("type") == "custom_tool_call")  # fmt: skip

        def to_function(item):
            item["type"] = "function_call"
            item.pop("input", None)
            item["arguments"] = json.dumps({"input": whole})

        each_call(to_function)
        events = [(n, o) for n, o in events if not n.startswith("response.custom_tool_call_input")]
    elif fault == "prefixed_name":

        def prefix(item):
            item["name"] = "functions." + str(item.get("name"))
            item.pop("namespace", None)

        each_call(prefix)
    elif fault == "second_call":
        calls = [i for i in final["output"] if i.get("type") == "custom_tool_call"]
        twin = dict(calls[0], id=str(calls[0].get("id")) + "_twin",
                    call_id=str(calls[0].get("call_id")) + "_twin")  # fmt: skip
        last = len(events) - 1
        position = len(final["output"])  # a well-formed second item: the next output_index
        events[last:last] = [("response.output_item.added", {"type": "response.output_item.added",
                                                             "output_index": position,
                                                             "item": dict(twin)}),
                             ("response.output_item.done", {"type": "response.output_item.done",
                                                            "output_index": position,
                                                            "item": dict(twin)})]  # fmt: skip
        final["output"].append(dict(twin))
    elif fault == "snapshot_differs":
        for item in final["output"]:
            if item.get("type") == "message" and item.get("content"):
                part = item["content"][0]
                part["text"] = str(part.get("text", "")) + " "
                break
    elif fault == "no_terminal":
        events = events[:-1]
    if fault != "no_terminal":
        events[-1] = (events[-1][0], dict(events[-1][1], response=final))
    seq = 0
    for _, obj in events:
        if isinstance(obj, dict) and "sequence_number" in obj:
            obj["sequence_number"] = seq
            seq += 1
    record["response"]["body"] = _serialize(events)
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    ap.add_argument("--harness", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--requests", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("responses", nargs="+")
    args = ap.parse_args(argv)

    picked = {}
    for path in expand_shards(args.responses, "responses"):
        for _, raw in iter_lines(path):
            rec = load_json_line(raw)
            if rec and rec.get("trace_id") and _is_candidate(rec):
                picked[rec["trace_id"]] = rec
                if len(picked) >= args.sample:
                    break
        if len(picked) >= args.sample:
            break
    requests = {}
    for path in expand_shards(args.requests, "requests"):
        for _, raw in iter_lines(path):
            rec = load_json_line(raw)
            if rec and rec.get("trace_id") in picked:
                requests[rec["trace_id"]] = rec
    keep = [t for t in picked if t in requests]

    os.makedirs(args.out_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix="mh-mutate-")
    os.chmod(work, 0o700)
    results = []
    try:
        for fault, expected in EXPECT.items():
            directory = os.path.join(work, fault)
            os.makedirs(directory)
            with open(os.path.join(directory, "requests-m.jsonl"), "w") as req, \
                    open(os.path.join(directory, "responses-m.jsonl"), "w") as resp:  # fmt: skip
                for trace_id in keep:
                    req.write(json.dumps(requests[trace_id], ensure_ascii=False) + "\n")
                    record = (
                        picked[trace_id] if fault == "none" else mutate(picked[trace_id], fault)
                    )
                    resp.write(json.dumps(record, ensure_ascii=False) + "\n")
            out = os.path.join(work, fault + "-verdict")
            verdict_mod.main(["--harness", args.harness, "--model", args.model, "--population", "window",
                              "--claims", "conformance", "--workers", "1", "--out-dir", out, directory])  # fmt: skip
            with open(os.path.join(out, "report.json")) as fh:
                report = json.load(fh)
            counts = report["counts"]
            tripped = {k for k in CHECKS_OF_INTEREST if counts.get(k)} | {
                rule for rules in report["by_rule"].values() for rule in rules
            }
            per_record = {k: counts.get(k, 0) for k in expected if k in CHECKS_OF_INTEREST}
            ok = tripped == expected and all(v == len(keep) for v in per_record.values())
            results.append({"fault": fault, "records": len(keep), "expected": sorted(expected),
                            "tripped": sorted(tripped), "hits": per_record, "ok": ok})  # fmt: skip
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
    summary = {"schema": "trtllm-serve-model-harness-validation/mutations@1",
               "validator_tree_sha256": verdict_mod.tree_sha256(), "results": results,
               "passed": bool(results) and all(r["ok"] for r in results) and len(keep) > 0}  # fmt: skip
    with open(os.path.join(args.out_dir, "mutations.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    for r in results:
        print(
            f"{'OK ' if r['ok'] else 'BAD'} {r['fault']:20s} records={r['records']} "
            f"expected={r['expected']} tripped={r['tripped']} hits={r['hits']}"
        )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
