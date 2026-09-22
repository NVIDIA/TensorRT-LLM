#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-request timeline for disaggregated KV transfers.

Reads every ``lifecycle_*.jsonl`` written under ``TRTLLM_KVCACHE_TIME_OUTPUT_PATH``
(see ``tensorrt_llm/_torch/disaggregation/native/perf_logger.py``), groups the
events by disaggregated request id, orders them on the rank-aligned steady
clock and prints the gap between consecutive events. A second table gives the
p50/p90/max of every observed ``A -> B`` transition so slow phases stand out.

Usage::

    python scripts/disagg_lifecycle_timeline.py /path/to/kvcache_time_output
    python scripts/disagg_lifecycle_timeline.py DIR --rid 42        # one request
    python scripts/disagg_lifecycle_timeline.py DIR --json out.json # machine readable

Steady-clock values are only comparable across processes when the C++ clock
alignment ran (same MPI world). Across ctx and gen servers the gaps are still
printed but flagged with ``~``; use ``t_wall`` for a rough cross-host check.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from collections import defaultdict
from typing import Iterable


def load_events(paths: Iterable[str]) -> tuple[list[dict], int]:
    """Return parsed events (excluding ``lifecycle_start``) and the malformed line count."""
    events: list[dict] = []
    malformed = 0
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if rec.get("event") == "lifecycle_start":
                    continue
                rec["_file"] = os.path.basename(path)
                events.append(rec)
    return events, malformed


def _label(rec: dict) -> str:
    name = rec["event"]
    side = rec.get("side")
    if side and not name.startswith(("ctx_", "gen_")):
        name = f"{side}:{name}"
    for key in ("admitted", "outcome"):
        if key in rec:
            name += f"[{rec[key]}]"
    return name


def build_timelines(events: list[dict]) -> dict:
    """Group by rid; each timeline is sorted by steady clock and annotated with gaps."""
    by_rid: dict = defaultdict(list)
    for rec in events:
        if rec.get("rid") is None:
            continue
        by_rid[rec["rid"]].append(rec)

    timelines = {}
    for rid, recs in by_rid.items():
        recs.sort(key=lambda r: (r.get("t_steady", 0.0), r.get("t_wall", 0.0)))
        rows = []
        prev = None
        for rec in recs:
            row = {
                "event": _label(rec),
                "rank": rec.get("rank"),
                "pid": rec.get("pid"),
                "t_steady": rec.get("t_steady"),
                "t_wall": rec.get("t_wall"),
                "gap_ms": None,
                "cross_process": False,
            }
            if prev is not None and rec.get("t_steady") is not None:
                row["gap_ms"] = (rec["t_steady"] - prev["t_steady"]) * 1000.0
                row["cross_process"] = rec.get("pid") != prev.get("pid")
            rows.append(row)
            prev = rec
        timelines[rid] = rows
    return timelines


def transition_stats(timelines: dict) -> list[dict]:
    """p50/p90/max of every observed consecutive-event pair, over all requests."""
    samples: dict = defaultdict(list)
    for rows in timelines.values():
        for a, b in zip(rows, rows[1:]):
            if b["gap_ms"] is None:
                continue
            key = (a["event"], b["event"], b["cross_process"])
            samples[key].append(b["gap_ms"])
    out = []
    for (src, dst, cross), values in samples.items():
        values.sort()
        out.append(
            {
                "from": src,
                "to": dst,
                "cross_process": cross,
                "count": len(values),
                "p50_ms": statistics.median(values),
                "p90_ms": values[min(len(values) - 1, int(0.9 * len(values)))],
                "max_ms": values[-1],
            }
        )
    out.sort(key=lambda s: -s["p90_ms"])
    return out


def _print_timeline(rid, rows, out) -> None:
    print(f"\n=== rid={rid} ({len(rows)} events)", file=out)
    for row in rows:
        gap = (
            ""
            if row["gap_ms"] is None
            else f"{'~' if row['cross_process'] else '+'}{row['gap_ms']:.2f} ms"
        )
        print(
            f"  {gap:>14}  rank={row['rank']!s:<4} pid={row['pid']!s:<7} {row['event']}", file=out
        )


def _print_stats(stats, out) -> None:
    if not stats:
        return
    print("\n=== transitions (sorted by p90; '~' = across processes)", file=out)
    print(f"  {'count':>5} {'p50 ms':>10} {'p90 ms':>10} {'max ms':>10}  transition", file=out)
    for s in stats:
        mark = "~" if s["cross_process"] else " "
        print(
            f"  {s['count']:>5} {s['p50_ms']:>10.2f} {s['p90_ms']:>10.2f} {s['max_ms']:>10.2f} "
            f"{mark} {s['from']} -> {s['to']}",
            file=out,
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("path", help="TRTLLM_KVCACHE_TIME_OUTPUT_PATH directory or one .jsonl file")
    parser.add_argument("--rid", type=int, action="append", help="only show these request ids")
    parser.add_argument("--json", help="write timelines and transition stats to this file")
    parser.add_argument(
        "--no-timelines", action="store_true", help="only print the transition table"
    )
    args = parser.parse_args(argv)

    if not os.path.exists(args.path):
        print(f"{args.path} does not exist", file=sys.stderr)
        return 1
    if os.path.isdir(args.path):
        paths = sorted(glob.glob(os.path.join(args.path, "lifecycle_*.jsonl")))
    else:
        paths = [args.path]
    if not paths:
        print(f"no lifecycle_*.jsonl under {args.path}", file=sys.stderr)
        return 1

    events, malformed = load_events(paths)
    timelines = build_timelines(events)
    if args.rid:
        timelines = {rid: rows for rid, rows in timelines.items() if rid in set(args.rid)}
    stats = transition_stats(timelines)

    print(
        f"{len(paths)} files, {len(events)} events, {len(timelines)} requests, {malformed} malformed lines"
    )
    if not args.no_timelines:
        for rid in sorted(timelines, key=lambda r: (str(type(r)), r)):
            _print_timeline(rid, timelines[rid], sys.stdout)
    _print_stats(stats, sys.stdout)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"requests": timelines, "transitions": stats}, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
