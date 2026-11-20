#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize the pi measurement runs (runs/<case>/) into RESULTS.md and results.json.

Per case: requests the recorder saw (and whether resends were byte-identical), how each answer
ended and when, pi's retries (delay, error kind), compactions, the tool-result strings pi sent back
(pi's own fixed strings; the fake servers' answers carry no user content), and how the run ended.
"""

import datetime
import hashlib
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
# run.sh's output root (PIKIT_MEASURE_DIR, default ./pikit-measure); cases.tsv stays beside this file.
WORK = os.path.abspath(os.environ.get("PIKIT_MEASURE_DIR", "pikit-measure"))


def pi_version() -> str:
    """The measured pi's version, from the install PIKIT_PI_DIR names."""
    package = os.path.join(os.environ.get("PIKIT_PI_DIR", ""), "node_modules", "@earendil-works",
                           "pi-coding-agent", "package.json")  # fmt: skip
    try:
        with open(package) as fh:
            return json.load(fh)["version"]
    except (OSError, ValueError, KeyError):
        return "(version unknown: set PIKIT_PI_DIR)"


def load(path):
    try:
        with open(path) as fh:
            return [json.loads(line) for line in fh if line.strip()]
    except FileNotFoundError:
        return []


def error_kind(message: str) -> str:
    """A short, fixed label for pi's error text."""
    for pattern, label in ((r"^(\d{3})[: ]", r"HTTP \1"), (r"API error \((\d{3})\)", r"HTTP \1"),
                           (r"Request timed out", "Request timed out."), (r"^terminated", "terminated"),
                           (r"ended without finish_reason", "Stream ended without finish_reason"),
                           (r"ended before message_stop", "ended before message_stop"),
                           (r"before a terminal response event",
                            "ended before a terminal response event")):  # fmt: skip
        found = re.search(pattern, message or "")
        if found:
            return found.expand(label)
    return "other"


def tool_results(body: dict):
    for m in body.get("messages") or []:
        if m.get("role") == "tool":
            yield None, str(m.get("content"))
        if isinstance(m.get("content"), list):
            for block in m["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    c = block.get("content")
                    text = (
                        c
                        if isinstance(c, str)
                        else " ".join(x.get("text", "") for x in c or [] if isinstance(x, dict))
                    )
                    yield block.get("is_error"), text
    for item in body.get("input") or []:
        if isinstance(item, dict) and item.get("type") == "function_call_output":
            yield None, str(item.get("output"))


def summarize(case_dir: str, spec: dict) -> dict:
    requests = load(os.path.join(case_dir, "trace", "requests-capture.jsonl"))
    responses = {
        r["trace_id"]: r for r in load(os.path.join(case_dir, "trace", "responses-capture.jsonl"))
    }
    events = load(os.path.join(case_dir, "events.jsonl"))
    answers, digests = [], []
    for q in requests:
        digests.append(
            hashlib.sha256(json.dumps(q["body"], sort_keys=True).encode()).hexdigest()[:12]
        )
        r = responses.get(q["trace_id"]) or {}
        timing = r.get("timing") or {}
        chunks = timing.get("chunks") or []
        gap = max((b[0] - a[0] for a, b in zip(chunks, chunks[1:])), default=None)
        answers.append({"status": r.get("status"), "http": r.get("http_status"),
                        "total_s": timing.get("total_s"), "max_gap_s": round(gap, 2) if gap else gap,
                        "before_reply": bool(timing.get("before_reply"))})  # fmt: skip
    resends = sum(1 for a, b in zip(digests, digests[1:]) if a == b)
    retries = [{"delay_ms": e.get("delayMs"), "error": error_kind(e.get("errorMessage"))}
               for e in events if e.get("type") == "auto_retry_start"]  # fmt: skip
    compactions = [{"reason": e.get("reason"), "ok": bool(e.get("result")),
                    "error": (e.get("errorMessage") or "")[:80]}
                   for e in events if e.get("type") == "compaction_end"]  # fmt: skip
    finals = [e["message"] for e in events if e.get("type") == "message_end"
              and (e.get("message") or {}).get("role") == "assistant"]  # fmt: skip
    results = []
    for q in requests:
        for is_error, text in tool_results(q["body"]):
            entry = {"is_error": is_error, "text": text[:160]}
            if entry not in results:
                results.append(entry)
    rc = open(os.path.join(WORK, "runs", os.path.basename(case_dir) + ".out")).read().strip() \
        if os.path.exists(os.path.join(WORK, "runs", os.path.basename(case_dir) + ".out")) else ""  # fmt: skip
    last_error = finals[-1].get("errorMessage") if finals else None
    return dict(spec, requests=len(requests), identical_resends=resends, answers=answers,
                retries=retries, compactions=compactions,
                final_stop_reason=finals[-1].get("stopReason") if finals else None,
                final_error=error_kind(last_error) if last_error else None,
                tool_results=results, pi=rc)  # fmt: skip


def main():
    specs = {}
    with open(os.path.join(HERE, "cases.tsv")) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            name, api, server, mode, envs, args = (line.rstrip("\n").split("\t") + [""] * 6)[:6]
            specs[name] = {"case": name, "api": api, "mode": mode, "env": envs, "server_args": args}
    out = [summarize(os.path.join(WORK, "runs", n), s) for n, s in specs.items()
           if os.path.isdir(os.path.join(WORK, "runs", n))]  # fmt: skip
    with open(os.path.join(WORK, "results.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    lines = [f"# pi {pi_version()} measurements against the fake servers ({datetime.date.today()})", "",
             "Generated by `summarize.py` from `runs/` (`run.sh`, `cases.tsv`). Default pi settings; "
             "`record_requests.py` in front unless `NOREC=1`. Times are seconds after the request "
             "arrived at the recorder.", "",
             "| case | api | mode | requests | identical resends "
             "| answers (status http total_s max_gap_s) | retries (delay ms: error) | compactions | final |",
             "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in out:
        answers = "; ".join(
            f"{a['status']} {a['http']} {a['total_s']} {a['max_gap_s']}" for a in r["answers"]
        )
        retries = "; ".join(f"{x['delay_ms']}: {x['error']}" for x in r["retries"]) or "-"
        comps = (
            "; ".join(f"{c['reason']} {'ok' if c['ok'] else 'failed'}" for c in r["compactions"])
            or "-"
        )
        # A case whose pi wrote no assistant message_end (it died or timed out first) has none.
        final = (r["final_stop_reason"] or "-") + (
            f" ({r['final_error']})" if r["final_error"] else ""
        )
        lines.append(f"| {r['case']} | {r['api']} | {r['mode']} | {r['requests']} | {r['identical_resends']} "
                     f"| {answers} | {retries} | {comps} | {final} |")  # fmt: skip
    lines += ["", "## Tool results pi sent back (outcome cases)", "", "| case | is_error | text (first 160 chars) |",
              "|---|---|---|"]  # fmt: skip
    for r in out:
        if r["case"].startswith("u-"):
            for t in r["tool_results"]:
                lines.append(
                    f"| {r['case']} | {t['is_error']} | `{t['text'].replace(chr(10), ' / ')}` |"
                )
    with open(os.path.join(WORK, "RESULTS.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"{len(out)} cases -> RESULTS.md, results.json")


if __name__ == "__main__":
    main()
