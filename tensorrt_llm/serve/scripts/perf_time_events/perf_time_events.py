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
"""Perf Time Events aggregator.

Stitches the per-rank ``time_events_rank{N}_pid{P}.jsonl`` files written live by
the executor (gated by ``TRTLLM_PERF_TIME_EVENTS_PATH``) together with the
disaggregation KV-transfer CSVs (``TRTLLM_KVCACHE_TIME_OUTPUT_PATH``) into a
single combined JSON. It also computes a few DERIVED signals (inter-step /
inter-chunk gaps, per-iteration starvation) and can optionally emit the same
interactive HTML timeline as the ``time_breakdown`` tool.

This module is intentionally torch-free and stdlib-only for the parse/merge/JSON
path; ``plotly``/``numpy`` are pulled in lazily (via the sibling
``time_breakdown`` package) only when ``--html`` is requested.
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# KV-transfer CSV headers (mirrors _torch/disaggregation/native/perf_logger.py)
# ---------------------------------------------------------------------------

# Python-native transceiver task rows: {instance}_{rank}.csv
_NATIVE_TASK_HEADER = [
    "timestamp",
    "task_type",
    "unique_rid",
    "peer_rank",
    "transfer_size_bytes",
    "avg_segment_size_bytes",
    "transfer_entry_count",
    "prepare_args_latency_ms",
    "queue_latency_ms",
    "transfer_latency_ms",
    "task_latency_ms",
    "throughput_mbs",
]
# Gen-side summary rows: {instance}_{rank}_gen_transfer_summary.csv
_GEN_SUMMARY_HEADER = ["timestamp", "RequestID", "gen_side_transfer_time(ms)", "kv_cache_size"]


def _iter_jsonl(path: str):
    """Yield parsed JSON objects from a .jsonl file, skipping blank/bad lines."""
    with open(path, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"WARNING: {path}:{line_no}: skipping malformed JSON line ({e})")


def parse_event_dir(event_dir: str) -> List[Dict[str, Any]]:
    """Parse per-rank ``time_events_*.jsonl`` files into a list of records.

    Each record is one finished request as written by
    ``PerfMetricsManager.maybe_write_request_events`` -- it already carries
    top-level ``request_id`` + ``time_breakdown_metrics`` (the shape
    ``time_breakdown``'s ``parse_request`` reads directly), plus ``rank`` and an
    optional disagg ``ctx_request_id``.

    Records are returned in a stable order (sorted by file then line) so repeated
    runs produce identical output.
    """
    records: List[Dict[str, Any]] = []
    files = sorted(glob.glob(os.path.join(event_dir, "time_events_*.jsonl")))
    for path in files:
        for obj in _iter_jsonl(path):
            if isinstance(obj, dict):
                records.append(obj)
    return records


def _to_float(value: str) -> Optional[float]:
    """Best-effort float parse; blank/None/garbage -> None."""
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_kv_csv_dir(kv_csv_dir: str) -> Dict[str, Any]:
    """Parse KV-transfer CSVs into join-ready structures.

    Handles two producers:

    * Python-native transceiver (``perf_logger.py``):
      - ``{instance}_{rank}.csv`` -> task rows keyed by ``unique_rid``
        (KVSendTask / AuxSendTask / KVRecvTask; recv rows leave the middle
        latency fields blank).
      - ``{instance}_{rank}_gen_transfer_summary.csv`` -> gen summary keyed by
        ``RequestID``.
    * C++ transceiver: ``*_send.csv`` / ``*_recv.csv`` keyed by ctx-phase
      ``RequestID`` (parsed for completeness so mixed dirs still aggregate).

    Returns a dict with ``task_events`` (rid -> list[row]), ``gen_summary``
    (rid -> row) and ``cpp_events`` (rid -> list[row]).
    """
    task_events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    gen_summary: Dict[str, Dict[str, Any]] = {}
    cpp_events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    csv_files = sorted(glob.glob(os.path.join(kv_csv_dir, "*.csv")))
    for path in csv_files:
        base = os.path.basename(path)
        try:
            with open(path, "r", newline="") as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                except StopIteration:
                    continue
                header = [h.strip() for h in header]
                rows = [row for row in reader if row]
        except OSError as e:
            print(f"WARNING: failed to read KV CSV {path}: {e}")
            continue

        if base.endswith("_gen_transfer_summary.csv") or header == _GEN_SUMMARY_HEADER:
            for row in rows:
                rec = dict(zip(header, row))
                rid = rec.get("RequestID")
                if rid is not None:
                    rec["_source_file"] = base
                    gen_summary[str(rid)] = rec
        elif header == _NATIVE_TASK_HEADER:
            for row in rows:
                rec = dict(zip(header, row))
                rid = rec.get("unique_rid")
                if rid is not None:
                    rec["_source_file"] = base
                    task_events[str(rid)].append(rec)
        else:
            # C++ (or unknown) layout: try to key on a RequestID-like column.
            rid_key = None
            for candidate in ("RequestID", "request_id", "unique_rid"):
                if candidate in header:
                    rid_key = candidate
                    break
            for row in rows:
                rec = dict(zip(header, row))
                rec["_source_file"] = base
                rid = rec.get(rid_key) if rid_key else None
                if rid is not None:
                    cpp_events[str(rid)].append(rec)

    return {
        "task_events": dict(task_events),
        "gen_summary": gen_summary,
        "cpp_events": dict(cpp_events),
    }


def _derive_gaps(metrics: List[Dict[str, Any]]) -> List[Optional[float]]:
    """Compute idle gaps between consecutive forward passes.

    Gap (seconds) between each entry's forward_start and the previous entry's
    forward_end. First entry -> None.
    """
    gaps: List[Optional[float]] = []
    prev_end = None
    for m in metrics:
        start = m.get("forward_start_time")
        if prev_end is not None and start is not None:
            gaps.append(start - prev_end)
        else:
            gaps.append(None)
        prev_end = m.get("forward_end_time", prev_end)
    return gaps


class PerfTimeEventsMerger:
    """Merge per-rank time-event records with KV-transfer CSVs.

    The per-rank event dir is the primary input; a ``/perf_metrics`` JSON dump
    (aggregated single-server runs) may be supplied additionally / instead.
    """

    def __init__(self):
        self.records: List[Dict[str, Any]] = []
        self.kv: Dict[str, Any] = {"task_events": {}, "gen_summary": {}, "cpp_events": {}}
        self.unjoined_kv_events: Dict[str, Any] = {}
        self.match_stats: Dict[str, int] = {}

    def merge(
        self,
        event_dir: Optional[str] = None,
        perf_json: Optional[str] = None,
        kv_csv_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        if event_dir:
            records.extend(parse_event_dir(event_dir))
        if perf_json:
            with open(perf_json, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                records.extend(data)
            elif isinstance(data, dict):
                records.append(data)

        if kv_csv_dir:
            self.kv = parse_kv_csv_dir(kv_csv_dir)

        task_events = self.kv.get("task_events", {})
        gen_summary = self.kv.get("gen_summary", {})
        cpp_events = self.kv.get("cpp_events", {})

        matched_rids = set()
        for rec in records:
            # Resolve every id this record could join on.
            rids = []
            for key in ("request_id", "ctx_request_id"):
                val = rec.get(key)
                if val is not None:
                    rids.append(str(val))

            joined_tasks: List[Dict[str, Any]] = []
            joined_summary = None
            joined_cpp: List[Dict[str, Any]] = []
            for rid in rids:
                if rid in task_events:
                    joined_tasks.extend(task_events[rid])
                    matched_rids.add(rid)
                if rid in gen_summary:
                    joined_summary = gen_summary[rid]
                    matched_rids.add(rid)
                if rid in cpp_events:
                    joined_cpp.extend(cpp_events[rid])
                    matched_rids.add(rid)

            if joined_tasks:
                rec["kv_transfer_events"] = joined_tasks
            if joined_summary is not None:
                rec["kv_gen_summary"] = joined_summary
            if joined_cpp:
                rec["kv_cpp_events"] = joined_cpp

            # DERIVED signals.
            tbm = rec.get("time_breakdown_metrics") or {}
            step_metrics = tbm.get("step_metrics") or []
            ctx_chunk_metrics = tbm.get("ctx_chunk_metrics") or []
            derived: Dict[str, Any] = {}
            if step_metrics:
                derived["inter_step_gaps"] = _derive_gaps(step_metrics)
            if ctx_chunk_metrics:
                derived["inter_chunk_gaps"] = _derive_gaps(ctx_chunk_metrics)
            # Per-iteration starvation from the extended batch-context fields.
            starved = []
            for m in step_metrics:
                if "num_capacity_fitting" in m and "num_scheduled" in m:
                    starved.append(m["num_capacity_fitting"] - m["num_scheduled"])
            if starved:
                derived["starved"] = starved
            if derived:
                rec["derived"] = derived

        # Report KV rows that never joined to a request record.
        unjoined_tasks = {rid: rows for rid, rows in task_events.items() if rid not in matched_rids}
        unjoined_summary = {rid: row for rid, row in gen_summary.items() if rid not in matched_rids}
        unjoined_cpp = {rid: rows for rid, rows in cpp_events.items() if rid not in matched_rids}
        self.unjoined_kv_events = {
            "task_events": unjoined_tasks,
            "gen_summary": unjoined_summary,
            "cpp_events": unjoined_cpp,
        }

        # Count UNIQUE rids across the three structures: the same rid routinely
        # appears in both task_events and gen_summary (send tasks + gen summary
        # for one request), so summing the dict lengths would double-count it
        # and report a spurious sub-100% match rate.
        all_kv_rids = set(task_events) | set(gen_summary) | set(cpp_events)
        total_kv = len(all_kv_rids)
        matched_kv = len(matched_rids)
        self.match_stats = {
            "num_records": len(records),
            "total_kv_rids": total_kv,
            "matched_kv_rids": matched_kv,
        }
        if total_kv:
            rate = 100.0 * matched_kv / total_kv
            if rate < 100.0:
                print(
                    f"WARNING: KV-transfer join match-rate {rate:.1f}% "
                    f"({matched_kv}/{total_kv} rids matched); unmatched rows "
                    f"reported under 'unjoined_kv_events'"
                )

        self.records = records
        return records

    def write(self, output_path: str) -> None:
        """Write the combined JSON (records + unjoined KV + match stats)."""
        payload = {
            "records": self.records,
            "unjoined_kv_events": self.unjoined_kv_events,
            "match_stats": self.match_stats,
        }
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"Wrote combined perf time events to {output_path} ({len(self.records)} records)")

    def write_html(self, html_path: str) -> None:
        """Render the interactive HTML timeline by reusing time_breakdown.

        Imported lazily so the merge/JSON path stays dependency-light.
        """
        from ..time_breakdown.time_breakdown import RequestTimeBreakdown

        analyzer = RequestTimeBreakdown()
        timing_data = []
        for i, rec in enumerate(self.records):
            timing_data.append(analyzer.parser.parse_request(rec, i))
            for metric in analyzer.config.metrics:
                duration = metric.calculate_duration(timing_data[-1])
                timing_data[-1][f"{metric.name}_time"] = duration
        analyzer.create_timing_diagram(timing_data, html_path)
        print(f"Wrote HTML timeline to {html_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-rank perf time-event files (+ KV-transfer CSVs) "
        "into a single combined JSON / HTML."
    )
    parser.add_argument(
        "--event-dir",
        default=os.getenv("TRTLLM_PERF_TIME_EVENTS_PATH") or None,
        help="Directory of per-rank time_events_*.jsonl files "
        "(default: $TRTLLM_PERF_TIME_EVENTS_PATH).",
    )
    parser.add_argument(
        "--kv-csv-dir",
        default=os.getenv("TRTLLM_KVCACHE_TIME_OUTPUT_PATH") or None,
        help="Directory of KV-transfer CSVs (default: $TRTLLM_KVCACHE_TIME_OUTPUT_PATH).",
    )
    parser.add_argument(
        "--perf-json",
        default=None,
        help="Optional /perf_metrics JSON dump to merge (aggregated runs).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="perf_time_events.combined.json",
        help="Output combined JSON path.",
    )
    parser.add_argument(
        "--html",
        default=None,
        const="perf_time_events.timeline.html",
        nargs="?",
        help="Also emit an HTML timeline (optional path).",
    )
    args = parser.parse_args()

    if not args.event_dir and not args.perf_json:
        parser.error("provide --event-dir (or $TRTLLM_PERF_TIME_EVENTS_PATH) or --perf-json")

    merger = PerfTimeEventsMerger()
    merger.merge(event_dir=args.event_dir, perf_json=args.perf_json, kv_csv_dir=args.kv_csv_dir)
    merger.write(args.output)
    if args.html:
        merger.write_html(args.html)


if __name__ == "__main__":
    main()
