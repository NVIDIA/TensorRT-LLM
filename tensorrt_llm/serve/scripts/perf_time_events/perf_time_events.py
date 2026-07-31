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


def parse_router_dir(router_dir: str) -> Dict[str, Any]:
    """Parse disagg-router ``disagg_router_*.jsonl`` files, keyed by ctx id.

    Each record is one finished request as written by
    ``RawRequestResponseHooks._maybe_write_router_event`` -- it carries the
    router dispatch timeline (arrival / ctx_dispatch / gen_dispatch / first_token
    / resp_done, all steady-clock seconds) plus both join ids
    (``ctx_request_id`` -- the cross-process key shared with the worker per-rank
    files -- and the router's own ``disagg_request_id``).

    Returns a dict keyed by ``str(ctx_request_id)`` (each mapping to a single
    router record ``dict``) so worker records join in O(1), plus one reserved
    key ``"_no_ctx"`` -> ``List[dict]`` holding every record that CANNOT be
    joined on ``ctx_request_id``:

    * records with no ``ctx_request_id`` (the gen-only / no-ctx path), and
    * records whose ``ctx_request_id`` is ambiguous because more than one
      distinct request reported it. This happens on the gen-only benchmark
      path, where the service hardcodes ``ctx_request_id=1`` for every request
      (``TRTLLM_DISAGG_BENCHMARK_GEN_ONLY``); keeping such a key joinable would
      false-attach one surviving router row to every worker record, so those
      records are made non-joinable and surfaced as leftovers instead.

    (Return type is ``Dict[str, Any]`` because the ``"_no_ctx"`` value is a list
    while every other value is a single record dict.)
    """
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    no_ctx: List[Dict[str, Any]] = []
    files = sorted(glob.glob(os.path.join(router_dir, "disagg_router_*.jsonl")))
    for path in files:
        for obj in _iter_jsonl(path):
            if not isinstance(obj, dict):
                continue
            ctx_id = obj.get("ctx_request_id")
            if ctx_id is None:
                no_ctx.append(obj)
            else:
                grouped[str(ctx_id)].append(obj)

    by_ctx: Dict[str, Any] = {}
    for ctx_id, rows in grouped.items():
        if len(rows) == 1:
            by_ctx[ctx_id] = rows[0]
        else:
            # Ambiguous ctx id -- cannot uniquely attribute a router record to a
            # worker record (e.g. the gen-only ctx_request_id=1 hardcode). Make
            # it non-joinable rather than last-write-wins false-joining.
            print(
                f"WARNING: router ctx_request_id={ctx_id!r} reported by "
                f"{len(rows)} distinct requests; treating as non-joinable "
                f"(gen-only benchmark hardcode?)"
            )
            no_ctx.extend(rows)
    if no_ctx:
        by_ctx["_no_ctx"] = no_ctx
    return by_ctx


def parse_client_dir(client_dir: str) -> List[Dict[str, Any]]:
    """Parse benchmark-client ``client_*.jsonl`` files into a flat list.

    Each record is one request's client send timeline (process-local monotonic
    ``send_time`` + wall-clock ``send_wall_time`` anchor) as written by
    ``benchmark_serving._maybe_write_client_time_events``. The client has no
    shared request id and a different clock epoch from the server, so these are
    surfaced as a STANDALONE timeline (not joined to worker/router records);
    ``send_wall_time`` is the only cross-process alignment handle and is
    best-effort. Sorted by ``send_wall_time`` (then ``client_index``) for stable,
    time-ordered output.
    """
    records: List[Dict[str, Any]] = []
    files = sorted(glob.glob(os.path.join(client_dir, "client_*.jsonl")))
    for path in files:
        for obj in _iter_jsonl(path):
            if isinstance(obj, dict):
                records.append(obj)
    records.sort(key=lambda r: (r.get("send_wall_time") or 0.0, r.get("client_index") or 0))
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
        # Router dispatch records keyed by ctx_request_id (+ "_no_ctx" list).
        self.router_events: Dict[str, Any] = {}
        self.unjoined_router_events: List[Dict[str, Any]] = []
        # Client send timeline -- standalone, not joined (no shared key/clock).
        self.client_events: List[Dict[str, Any]] = []

    def merge(
        self,
        event_dir: Optional[str] = None,
        perf_json: Optional[str] = None,
        kv_csv_dir: Optional[str] = None,
        router_dir: Optional[str] = None,
        client_dir: Optional[str] = None,
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
        if router_dir:
            self.router_events = parse_router_dir(router_dir)
        if client_dir:
            self.client_events = parse_client_dir(client_dir)

        task_events = self.kv.get("task_events", {})
        gen_summary = self.kv.get("gen_summary", {})
        cpp_events = self.kv.get("cpp_events", {})
        # Router records keyed by ctx id; "_no_ctx" holds the gen-only path list.
        router_by_ctx = {k: v for k, v in self.router_events.items() if k != "_no_ctx"}
        matched_router_ctx = set()

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

            # Router dispatch join -- STRICTLY on ctx_request_id, the only key
            # the router and the worker per-rank files genuinely share. An
            # earlier version also tried the worker-local request_id, which can
            # false-join to an unrelated router record whose ctx id numerically
            # equals this record's request_id. Ambiguous/duplicate ctx ids were
            # already dropped from router_by_ctx by parse_router_dir.
            router_rec = None
            ctx_rid = rec.get("ctx_request_id")
            if ctx_rid is not None:
                ctx_rid = str(ctx_rid)
                router_rec = router_by_ctx.get(ctx_rid)
                if router_rec is not None:
                    matched_router_ctx.add(ctx_rid)
            if router_rec is not None:
                rec["router_dispatch"] = router_rec

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

            # Cross-process lifecycle spans (steady-clock seconds), when the
            # request-level timing scalars are present (serve path forces
            # return_perf_metrics on under TRTLLM_PERF_TIME_EVENTS_PATH).
            rtm = rec.get("request_timing_metrics") or {}
            arrival = rtm.get("arrival_time")
            first_sched = rtm.get("first_scheduled_time")
            first_tok = rtm.get("first_token_time")
            last_tok = rtm.get("last_token_time")
            if arrival is not None and first_sched is not None:
                derived["arrival_to_first_schedule"] = first_sched - arrival
            if first_sched is not None and first_tok is not None:
                derived["schedule_to_first_token"] = first_tok - first_sched
            if first_tok is not None and last_tok is not None:
                derived["decode_duration"] = last_tok - first_tok
            # Router-side dispatch waits (steady-clock, same epoch as the worker).
            if router_rec is not None:
                r_arr = router_rec.get("arrival_time")
                r_ctx = router_rec.get("ctx_dispatch_time")
                r_gen = router_rec.get("gen_dispatch_time")
                if r_arr is not None and r_ctx is not None:
                    derived["router_arrival_to_ctx_dispatch"] = r_ctx - r_arr
                if r_ctx is not None and r_gen is not None:
                    derived["router_ctx_to_gen_dispatch"] = r_gen - r_ctx
                # Router arrival -> worker arrival: dispatch + network to the
                # worker (both steady-clock). Only meaningful when both present.
                if r_arr is not None and arrival is not None:
                    derived["router_to_worker_arrival"] = arrival - r_arr
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

        # Router records that never joined a request record: the ctx-keyed
        # leftovers plus every gen-only ("_no_ctx") record (which by construction
        # cannot join on ctx_request_id).
        unjoined_router = [
            row for ctx_id, row in router_by_ctx.items() if ctx_id not in matched_router_ctx
        ]
        unjoined_router.extend(self.router_events.get("_no_ctx", []))
        self.unjoined_router_events = unjoined_router

        # Count UNIQUE rids across the three structures: the same rid routinely
        # appears in both task_events and gen_summary (send tasks + gen summary
        # for one request), so summing the dict lengths would double-count it
        # and report a spurious sub-100% match rate.
        all_kv_rids = set(task_events) | set(gen_summary) | set(cpp_events)
        total_kv = len(all_kv_rids)
        matched_kv = len(matched_rids)
        total_router = len(router_by_ctx)
        matched_router = len(matched_router_ctx)
        self.match_stats = {
            "num_records": len(records),
            "total_kv_rids": total_kv,
            "matched_kv_rids": matched_kv,
            "total_router_ctx": total_router,
            "matched_router_ctx": matched_router,
            "num_client_events": len(self.client_events),
        }
        if total_kv:
            rate = 100.0 * matched_kv / total_kv
            if rate < 100.0:
                print(
                    f"WARNING: KV-transfer join match-rate {rate:.1f}% "
                    f"({matched_kv}/{total_kv} rids matched); unmatched rows "
                    f"reported under 'unjoined_kv_events'"
                )
        if total_router:
            rate = 100.0 * matched_router / total_router
            if rate < 100.0:
                print(
                    f"WARNING: router-dispatch join match-rate {rate:.1f}% "
                    f"({matched_router}/{total_router} ctx ids matched); unmatched "
                    f"rows reported under 'unjoined_router_events'"
                )

        self.records = records
        return records

    def write(self, output_path: str) -> None:
        """Write the combined JSON (records + unjoined KV/router + client + stats)."""
        payload = {
            "records": self.records,
            "unjoined_kv_events": self.unjoined_kv_events,
            "unjoined_router_events": self.unjoined_router_events,
            # Client send timeline: standalone (no shared key/clock), not joined.
            "client_events": self.client_events,
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
        "--router-dir",
        default=os.getenv("TRTLLM_PERF_TIME_EVENTS_ROUTER_PATH") or None,
        help="Directory of disagg-router disagg_router_*.jsonl files "
        "(default: $TRTLLM_PERF_TIME_EVENTS_ROUTER_PATH). Joined to request "
        "records by ctx_request_id.",
    )
    parser.add_argument(
        "--client-dir",
        default=os.getenv("TRTLLM_PERF_TIME_EVENTS_CLIENT_PATH") or None,
        help="Directory of benchmark-client client_*.jsonl files "
        "(default: $TRTLLM_PERF_TIME_EVENTS_CLIENT_PATH). Surfaced as a "
        "standalone timeline (no shared key/clock with the server records).",
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

    if not any((args.event_dir, args.perf_json, args.router_dir, args.client_dir)):
        parser.error(
            "provide at least one input: --event-dir (or "
            "$TRTLLM_PERF_TIME_EVENTS_PATH), --perf-json, --router-dir, or "
            "--client-dir"
        )

    merger = PerfTimeEventsMerger()
    merger.merge(
        event_dir=args.event_dir,
        perf_json=args.perf_json,
        kv_csv_dir=args.kv_csv_dir,
        router_dir=args.router_dir,
        client_dir=args.client_dir,
    )
    merger.write(args.output)
    if args.html:
        merger.write_html(args.html)


if __name__ == "__main__":
    main()
