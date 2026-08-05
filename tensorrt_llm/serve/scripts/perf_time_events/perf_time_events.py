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
"""Perf Time Events aggregator (per-event long -> wide compiler).

Every process that participates in an end-to-end perf timeline appends **one
JSONL line per lifecycle event** (flushed as the event happens) to its own
per-rank / per-process file:

* worker (ctx/gen ``PyExecutor`` ranks) -> ``time_events_rank{N}_pid{P}.jsonl``
  (``TRTLLM_PERF_TIME_EVENTS_PATH``); each line is a flat envelope
  ``{"role","event","request_id","ctx_request_id","rank","t","pid"}`` where
  ``role in {ctx, gen}``.
* disagg router -> ``disagg_router_pid{P}.jsonl``
  (``TRTLLM_PERF_TIME_EVENTS_ROUTER_PATH``); same envelope with ``role=router``,
  ``request_id`` a router-local sequence, plus ``disagg_request_id`` /
  ``ctx_server`` / ``gen_server`` provenance.
* benchmark client -> ``client_pid{P}.jsonl``
  (``TRTLLM_PERF_TIME_EVENTS_CLIENT_PATH``); this one stays a COMPOUND
  one-record-per-request dump (``source=client``) -- it is a post-hoc batch
  write, not subject to hangs, and keeps the vLLM ttft/e2e path intact.

This compiler reads all of the above (plus the KV-transfer CSVs) and pivots the
per-event worker + router lines **long -> wide**: it groups events by
``request_id``, joins ctx <-> gen <-> router on ``ctx_request_id``, dedups TP
ranks by ``(role, request_id)``, and emits **one combined record per request**
carrying every event timestamp plus the derived intra-domain spans. A request
that hung emits its *partial* record (events that never fired are simply
absent, not zero) -- which is the whole point of the per-event redesign:
the last stamp written localizes the stall.

Two output shapes:

* ``--events-jsonl combined_time_events.jsonl`` (new primary): one line per
  request, ``{ctx_request_id, gen_request_id, <event>: t, ..., spans: {...}}``.
* ``-o`` combined JSON + ``-a/--agg-jsonl`` latency aggregate: unchanged in
  spirit; the aggregate reuses the same span/stat machinery.

This module is intentionally torch-free and stdlib-only for the
parse/merge/JSON path; ``plotly``/``numpy`` are pulled in lazily (via the
sibling ``time_breakdown`` package) only when ``--html`` is requested over a
``--perf-json`` dump.
"""

import argparse
import csv
import glob
import json
import math
import os
import statistics
from collections import Counter, defaultdict
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

# ---------------------------------------------------------------------------
# Per-event schema constants
# ---------------------------------------------------------------------------

# Worker lifecycle events, per role. Every worker line is one of these; the
# offline pivot flattens them into one wide record per (role, request_id) and
# joins ctx <-> gen on ctx_request_id. GEN carries the gen-init lifecycle only
# (decode requests record no time events, by design).
_CTX_EVENTS = ("ctx_arrival", "ctx_first_scheduled", "ctx_first_token", "ctx_ready_sent")
_GEN_EVENTS = (
    "gen_arrival",
    "gen_init_scheduled",
    "gen_kv_transfer_start",
    "gen_kv_transfer_end",
    "gen_first_scheduled",
    "gen_first_token",
    "gen_last_token",
)

# Router event name -> pivoted field name. The pivoted router record exposes the
# ``*_time`` field names that _ROUTER_METRICS differences.
_ROUTER_EVENT_TO_FIELD = {
    "arrival": "arrival_time",
    "ctx_dispatch": "ctx_dispatch_time",
    "gen_dispatch": "gen_dispatch_time",
    "first_token": "first_token_time",
    "resp_done": "resp_done_time",
}
# Router provenance fields carried through the pivot (first non-null wins).
_ROUTER_PROVENANCE = ("disagg_request_id", "ctx_server", "gen_server")


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
    """Parse per-rank ``time_events_*.jsonl`` files into a flat list of EVENTS.

    Each line is now **one lifecycle event** (the flat envelope
    ``{role, event, request_id, ctx_request_id, rank, t, pid}``), not one
    finished request. The list is returned in a stable order (sorted by file
    then line) so the TP-rank dedup below is deterministic -- lexical filename
    sort makes ``time_events_rank0_*`` the canonical first-seen copy.
    """
    events: List[Dict[str, Any]] = []
    files = sorted(glob.glob(os.path.join(event_dir, "time_events_*.jsonl")))
    for path in files:
        for obj in _iter_jsonl(path):
            if isinstance(obj, dict):
                events.append(obj)
    return events


def _pivot_worker_events(
    events: List[Dict[str, Any]],
) -> "tuple[List[Dict[str, Any]], List[Dict[str, Any]]]":
    """Pivot flat worker events long -> wide into per-(role, request_id) records.

    A tensor-parallel worker writes the SAME ``request_id`` once per rank (tp4
    gen -> 4 near-identical copies with lockstep timings). Grouping by
    ``request_id`` collapses the ranks; first-seen ``t`` wins per event name
    (``parse_event_dir`` sorts by filename so rank0 is canonical), so ``n`` is
    the number of logical requests, never inflated by the TP degree.

    Returns ``(ctx_records, gen_records)`` where each record is
    ``{"request_id", "ctx_request_id", <event_name>: t, ...}`` -- only events
    that actually fired are present (a hung request keeps its partial set).
    """
    ctx_groups: Dict[Any, Dict[str, Any]] = {}
    gen_groups: Dict[Any, Dict[str, Any]] = {}
    for e in events:
        role = e.get("role")
        if role == "ctx":
            groups, allowed = ctx_groups, _CTX_EVENTS
        elif role == "gen":
            groups, allowed = gen_groups, _GEN_EVENTS
        else:
            continue
        event = e.get("event")
        if event not in allowed:
            continue
        rid = e.get("request_id")
        g = groups.get(rid)
        if g is None:
            g = {"request_id": rid, "ctx_request_id": None}
            groups[rid] = g
        cid = e.get("ctx_request_id")
        if g["ctx_request_id"] is None and cid is not None:
            g["ctx_request_id"] = cid
        # First-seen wins (rank0 canonical); ignore later ranks' copies.
        if event not in g:
            g[event] = e.get("t")
    return list(ctx_groups.values()), list(gen_groups.values())


def parse_router_dir(router_dir: str) -> Dict[str, Any]:
    """Parse disagg-router ``disagg_router_*.jsonl`` (per-event) -> keyed by ctx id.

    Each line is one router lifecycle event (``arrival`` / ``ctx_dispatch`` /
    ``gen_dispatch`` / ``first_token`` / ``resp_done``). A single request's
    lines are stitched by ``(pid, request_id)`` -- ``request_id`` is the
    router-local sequence, unique within a process -- then pivoted into one wide
    record exposing the ``*_time`` fields ``_ROUTER_METRICS`` differences. The
    cross-process join key ``ctx_request_id`` is resolved as the first non-null
    value seen across the request's lines (it is assigned during the ctx
    round-trip, so ``arrival`` / ``ctx_dispatch`` precede it and are null).

    Returns a dict keyed by ``str(ctx_request_id)`` (each -> a single pivoted
    router record ``dict``) plus one reserved key ``"_no_ctx"`` ->
    ``List[dict]`` holding every record that CANNOT be joined on
    ``ctx_request_id``:

    * records whose resolved ``ctx_request_id`` is null (the gen-only / no-ctx
      path), and
    * records whose ``ctx_request_id`` is ambiguous because more than one
      distinct request (``(pid, request_id)`` group) reported it. This happens
      on the gen-only benchmark path, where the service hardcodes
      ``ctx_request_id=1`` for every request; keeping such a key joinable would
      false-attach one surviving router row to every worker record, so those
      records are made non-joinable and surfaced as leftovers instead.

    (Return type is ``Dict[str, Any]`` because the ``"_no_ctx"`` value is a list
    while every other value is a single record dict.)
    """
    # Group per-event lines by (pid, router-local request_id).
    raw: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    files = sorted(glob.glob(os.path.join(router_dir, "disagg_router_*.jsonl")))
    for path in files:
        for obj in _iter_jsonl(path):
            if not isinstance(obj, dict):
                continue
            raw[(obj.get("pid"), obj.get("request_id"))].append(obj)

    # Pivot each request's lines into one wide record.
    pivoted: List[Dict[str, Any]] = []
    for _key, rows in raw.items():
        rec: Dict[str, Any] = {"ctx_request_id": None}
        for prov in _ROUTER_PROVENANCE:
            rec[prov] = None
        for row in rows:
            field = _ROUTER_EVENT_TO_FIELD.get(row.get("event"))
            if field is not None and field not in rec:
                rec[field] = row.get("t")
            cid = row.get("ctx_request_id")
            if rec["ctx_request_id"] is None and cid is not None:
                rec["ctx_request_id"] = cid
            for prov in _ROUTER_PROVENANCE:
                if rec[prov] is None and row.get(prov) is not None:
                    rec[prov] = row.get(prov)
        pivoted.append(rec)

    # Bucket by ctx id, detecting the ambiguous / no-ctx cases.
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    no_ctx: List[Dict[str, Any]] = []
    for rec in pivoted:
        cid = rec.get("ctx_request_id")
        if cid is None:
            no_ctx.append(rec)
        else:
            grouped[str(cid)].append(rec)

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

    Unlike the worker / router files, the client file stays a COMPOUND
    one-record-per-request dump (``source=client``) written post-hoc by
    ``benchmark_serving._maybe_write_client_time_events``: it is not subject to
    hangs and preserves the self-contained vLLM ttft/e2e view. The client has no
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


# ===========================================================================
# Latency aggregation (--agg-jsonl): mean / P50 / P99 of the lifecycle
# intervals between time events. Pure stdlib -- no numpy / no time_breakdown.
# ===========================================================================
#
# Clock domains (empirically confirmed on a GB200 disagg gpt-oss run):
#   Domain A = {disagg router, gen worker}   ~46790 s steady-clock epoch
#   Domain B = {ctx worker, benchmark client} ~217445 s steady-clock epoch
# Cross-domain subtraction is invalid. Every worker/router span below differences
# two timestamps from the SAME record (one worker rank, or one router request),
# so all are clock_safe. The 12 canonical cross-worker spans (Group 1) come only
# from an aggregated --perf-json dump that TRT-LLM has already offset-corrected;
# the two that straddle A<->B are marked clock_safe=False.
#
# The 12 canonical spans mirror time_breakdown.TimingMetricsConfig verbatim
# (name, start_field, end_field). They are hardcoded rather than imported so
# the aggregate path stays stdlib-only / importable outside the torch
# container; test_perf_time_events.py has a drift guard that cross-checks this
# list against TimingMetricsConfig whenever that module can be imported.
_CANONICAL_METRICS = [
    # (name, start_field, end_field, source, clock_safe)
    (
        "disagg_preprocessing",
        "disagg_server_arrival_time",
        "ctx_server_arrival_time",
        "disagg",
        False,
    ),
    ("ctx_preprocessing", "ctx_server_arrival_time", "ctx_arrival_time", "ctx_worker", True),
    ("ctx_queue", "ctx_arrival_time", "ctx_first_scheduled_time", "ctx_worker", True),
    ("ctx_processing", "ctx_first_scheduled_time", "ctx_first_token_time", "ctx_worker", True),
    (
        "ctx_postprocessing",
        "ctx_first_token_time",
        "ctx_server_first_token_time",
        "ctx_worker",
        True,
    ),
    ("disagg_relay", "ctx_server_first_token_time", "gen_server_arrival_time", "disagg", False),
    ("gen_preprocessing", "gen_server_arrival_time", "gen_arrival_time", "gen_worker", True),
    ("gen_queue_wait", "gen_arrival_time", "gen_kv_cache_transfer_start", "gen_worker", True),
    (
        "gen_kv_transfer",
        "gen_kv_cache_transfer_start",
        "gen_kv_cache_transfer_end",
        "gen_worker",
        True,
    ),
    (
        "gen_post_transfer",
        "gen_kv_cache_transfer_end",
        "gen_first_scheduled_time",
        "gen_worker",
        True,
    ),
    (
        "gen_postprocessing",
        "gen_first_scheduled_time",
        "gen_server_first_token_time",
        "gen_worker",
        True,
    ),
    (
        "disagg_postprocessing",
        "gen_server_first_token_time",
        "disagg_server_first_token_time",
        "disagg",
        True,
    ),
]

# Router dispatch chain -- all fields live on ONE pivoted router record
# (Domain A), so every span is clock_safe.
_ROUTER_METRICS = [
    # (name, start_field, end_field)
    ("router:arrival->ctx_dispatch", "arrival_time", "ctx_dispatch_time"),
    ("router:arrival->gen_dispatch", "arrival_time", "gen_dispatch_time"),
    ("router:ctx_dispatch->gen_dispatch", "ctx_dispatch_time", "gen_dispatch_time"),
    ("router:gen_dispatch->first_token", "gen_dispatch_time", "first_token_time"),
    ("router:first_token->resp_done", "first_token_time", "resp_done_time"),
]

# Custom worker spans, now sourced from the pivoted per-event worker records
# (long->wide) rather than request_timing_metrics / step_metrics. Each span
# differences two stamps on the SAME worker record, so all are clock_safe.
#   (name, role, start_event, end_event)
# ``ctx:forward_start->sampler_end`` (old) is intentionally DROPPED: the
# per-chunk device events were removed with the decode series, so there is no
# per-event source for it. ``gen:kv_transfer_start->end`` now populates from the
# #11/#12 Python stamps -- the durability win, and no longer 0.0-on-sync.
_WORKER_EVENT_METRICS = [
    ("ctx:arrival->first_scheduled", "ctx", "ctx_arrival", "ctx_first_scheduled"),
    ("gen:arrival->first_scheduled", "gen", "gen_arrival", "gen_first_scheduled"),
    ("gen:kv_transfer_start->end", "gen", "gen_kv_transfer_start", "gen_kv_transfer_end"),
    ("gen:first_token->last_token", "gen", "gen_first_token", "gen_last_token"),
]


def _percentile(values: List[float], p: float) -> float:
    """p-th percentile with linear interpolation.

    Mirrors tests/integration/defs/perf/perf_regression_utils.py. Empty -> 0.0.
    """
    if not values:
        return 0.0
    s = sorted(values)
    k = (p / 100.0) * (len(s) - 1)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + frac * (s[hi] - s[lo])


def _duration(start: Optional[float], end: Optional[float]) -> float:
    """Interval end-start with time_breakdown.TimingMetric's duration rules.

    Returns 0.0 for missing / NaN / inverted (start>end), matching
    time_breakdown.TimingMetric.calculate_duration. A wall-clock interval
    between two distinct steady-clock reads is never genuinely 0, so callers
    treat 0.0 as "not measured" and exclude it from the sample.
    """
    if start is None or end is None:
        return 0.0
    try:
        if math.isnan(start) or math.isnan(end):
            return 0.0
    except TypeError:
        return 0.0
    if start > end:
        return 0.0
    return end - start


def _nz(value: Optional[float]) -> Optional[float]:
    """Treat a 0.0 timestamp as "not recorded" (None).

    Retained for the ``--perf-json`` canonical path, whose KV-transfer setters on
    the synchronous disagg receive path are deferred (ticket #15871) and land as
    0.0. The per-event worker stamps never carry 0.0 (an event that did not fire
    is simply absent), so this is not needed on the event-dir path.
    """
    return value if value else None


def _span_or_none(start: Optional[float], end: Optional[float]) -> Optional[float]:
    """Positive interval in seconds, or None when either endpoint is absent.

    Unlike ``_duration`` (which returns 0.0 for the "not measured" case so the
    aggregator can filter it), the combined per-request record OMITS a span when
    it cannot be computed -- a hung request's later spans are absent, never a
    misleading 0.0.
    """
    d = _duration(start, end)
    return d if d else None


def _extract_canonical_fields(rec: Dict[str, Any]) -> Dict[str, float]:
    """Flatten the canonical lifecycle timestamps from a ``/perf_metrics`` record.

    Mirrors time_breakdown.RequestDataParser.parse_request over the aggregated
    ``--perf-json`` shape. This is the ONLY consumer of the nested
    ``ctx_perf_metrics`` / ``gen_perf_metrics`` containers; the per-event
    ``--event-dir`` path never produces them, so the 12 canonical rows read as
    not_recorded on a pure event-dir capture (the worker spans in
    ``_WORKER_EVENT_METRICS`` cover the event-dir case instead).
    """
    nan = float("nan")
    ctx_perf = rec.get("ctx_perf_metrics")
    gen_perf = rec.get("gen_perf_metrics")
    disagg = ctx_perf is not None and gen_perf is not None
    if disagg:
        ctxm = ((ctx_perf or {}).get("perf_metrics") or {}).get("timing_metrics") or {}
        genm = ((gen_perf or {}).get("perf_metrics") or {}).get("timing_metrics") or {}
        d_arr = rec.get("disagg_server_arrival_time", nan)
        d_ftt = rec.get("disagg_server_first_token_time", nan)
    else:
        ctxm = (rec.get("perf_metrics") or {}).get("timing_metrics") or {}
        genm = {}
        d_arr = nan
        d_ftt = nan
    return {
        "disagg_server_arrival_time": d_arr,
        "ctx_server_arrival_time": ctxm.get("server_arrival_time", nan),
        "ctx_arrival_time": ctxm.get("arrival_time", nan),
        "ctx_first_scheduled_time": ctxm.get("first_scheduled_time", nan),
        "ctx_first_token_time": ctxm.get("first_token_time", nan),
        "ctx_server_first_token_time": ctxm.get("server_first_token_time", nan),
        "gen_server_arrival_time": genm.get("server_arrival_time", nan),
        "gen_arrival_time": genm.get("arrival_time", nan),
        "gen_first_scheduled_time": genm.get("first_scheduled_time", nan),
        "gen_kv_cache_transfer_start": genm.get("kv_cache_transfer_start", nan),
        "gen_kv_cache_transfer_end": genm.get("kv_cache_transfer_end", nan),
        "gen_server_first_token_time": genm.get("server_first_token_time", nan),
        "disagg_server_first_token_time": d_ftt,
    }


def _stats_row(
    metric: str,
    source: str,
    clock_safe: bool,
    samples_s: List[float],
    unit: str = "ms",
    scale: float = 1000.0,
) -> Dict[str, Any]:
    """Build one JSONL row from a list of second-valued samples.

    ``samples_s`` are seconds; zeros are excluded (see _duration) and the rest
    scaled to ms and rounded to 3 dp. An empty sample yields a
    ``status: not_recorded`` row (never silently dropped).
    """
    vals = [v * scale for v in samples_s if v]
    if not vals:
        return {
            "metric": metric,
            "unit": unit,
            "source": source,
            "clock_safe": clock_safe,
            "status": "not_recorded",
        }
    return {
        "metric": metric,
        "unit": unit,
        "source": source,
        "clock_safe": clock_safe,
        "n": len(vals),
        "mean": round(statistics.mean(vals), 3),
        "p50": round(_percentile(vals, 50), 3),
        "p99": round(_percentile(vals, 99), 3),
        "min": round(min(vals), 3),
        "max": round(max(vals), 3),
    }


def _joinable_ctx_ids(
    ctx_records: List[Dict[str, Any]],
    gen_records: List[Dict[str, Any]],
) -> set:
    """ctx_request_ids that map 1:1 across roles (safe to merge ctx <-> gen).

    A ctx id shared by more than one ctx record OR more than one gen record is
    ambiguous -- the gen-only benchmark hardcodes ``ctx_request_id=1`` for every
    request, so merging on it would collapse every gen request into one. Such
    ids are excluded here; their records stay standalone (and, like the router's
    ``_no_ctx`` bucket, never cross-join).
    """
    ctx_counts = Counter(
        str(r["ctx_request_id"]) for r in ctx_records if r.get("ctx_request_id") is not None
    )
    gen_counts = Counter(
        str(r["ctx_request_id"]) for r in gen_records if r.get("ctx_request_id") is not None
    )
    ids = set(ctx_counts) | set(gen_counts)
    return {i for i in ids if ctx_counts.get(i, 0) <= 1 and gen_counts.get(i, 0) <= 1}


def _combined_record(
    ctx_request_id: Any,
    ctx_rec: Optional[Dict[str, Any]],
    gen_rec: Optional[Dict[str, Any]],
    router_rec: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble ONE wide per-request record from its (partial) event sources.

    Raw event stamps are copied through only when present -- a request that hung
    keeps exactly the stamps it reached, nothing back-filled. ``spans`` carries
    the derived intra-domain durations (seconds); only spans whose two endpoints
    both fired are included. Cross-host spans (ctx<->gen, router<->worker) are
    deliberately omitted: the two live in different steady-clock epochs, so their
    absolute difference is meaningless (see the clock-domain note above).
    """
    rec: Dict[str, Any] = {"ctx_request_id": ctx_request_id}
    if gen_rec is not None:
        rec["gen_request_id"] = gen_rec.get("request_id")
    if ctx_rec is not None:
        rec["ctx_worker_request_id"] = ctx_rec.get("request_id")

    # Raw stamps, flattened; only what fired.
    if router_rec is not None:
        for field in _ROUTER_EVENT_TO_FIELD.values():
            if router_rec.get(field) is not None:
                rec[f"router_{field}"] = router_rec[field]
        for prov in _ROUTER_PROVENANCE:
            if router_rec.get(prov) is not None:
                rec[prov] = router_rec[prov]
    if ctx_rec is not None:
        for ev in _CTX_EVENTS:
            if ctx_rec.get(ev) is not None:
                rec[ev] = ctx_rec[ev]
    if gen_rec is not None:
        for ev in _GEN_EVENTS:
            if gen_rec.get(ev) is not None:
                rec[ev] = gen_rec[ev]

    spans: Dict[str, float] = {}

    def add(name: str, start_field: str, end_field: str) -> None:
        v = _span_or_none(rec.get(start_field), rec.get(end_field))
        if v is not None:
            spans[name] = round(v, 6)

    # Router (all same record / host).
    if router_rec is not None:
        add("router:arrival->ctx_dispatch", "router_arrival_time", "router_ctx_dispatch_time")
        add("router:arrival->gen_dispatch", "router_arrival_time", "router_gen_dispatch_time")
        add("router:ctx_dispatch->gen_dispatch", "router_ctx_dispatch_time",
            "router_gen_dispatch_time")
        add("router:gen_dispatch->first_token", "router_gen_dispatch_time", "router_first_token_time")
        add("router:first_token->resp_done", "router_first_token_time", "router_resp_done_time")
    # ctx worker.
    add("ctx:arrival->first_scheduled", "ctx_arrival", "ctx_first_scheduled")
    add("ctx:first_scheduled->first_token", "ctx_first_scheduled", "ctx_first_token")
    add("ctx:first_token->ready_sent", "ctx_first_token", "ctx_ready_sent")
    # gen worker.
    add("gen:arrival->init_scheduled", "gen_arrival", "gen_init_scheduled")
    add("gen:init_scheduled->kv_transfer_start", "gen_init_scheduled", "gen_kv_transfer_start")
    add("gen:kv_transfer_start->end", "gen_kv_transfer_start", "gen_kv_transfer_end")
    add("gen:kv_transfer_end->first_scheduled", "gen_kv_transfer_end", "gen_first_scheduled")
    add("gen:first_scheduled->first_token", "gen_first_scheduled", "gen_first_token")
    add("gen:first_token->last_token", "gen_first_token", "gen_last_token")

    if spans:
        rec["spans"] = spans
    return rec


class PerfTimeEventsMerger:
    """Compile per-event worker + router logs (+ client + KV CSVs) into one
    combined per-request timeline.

    The per-event worker/router dirs are the primary input; a ``/perf_metrics``
    JSON dump (aggregated single-server runs) may be supplied additionally /
    instead and feeds only the canonical 12-span group.
    """

    def __init__(self):
        # Combined, pivoted per-request records (the primary long->wide output).
        self.records: List[Dict[str, Any]] = []
        # Pivoted per-role worker records (feed the custom worker spans).
        self.ctx_records: List[Dict[str, Any]] = []
        self.gen_records: List[Dict[str, Any]] = []
        # Aggregated /perf_metrics dumps (feed the canonical 12 + --html).
        self.perf_json_records: List[Dict[str, Any]] = []
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
        if event_dir:
            self.ctx_records, self.gen_records = _pivot_worker_events(parse_event_dir(event_dir))
        if perf_json:
            with open(perf_json, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                self.perf_json_records.extend(data)
            elif isinstance(data, dict):
                self.perf_json_records.append(data)

        if kv_csv_dir:
            self.kv = parse_kv_csv_dir(kv_csv_dir)
        if router_dir:
            self.router_events = parse_router_dir(router_dir)
        if client_dir:
            self.client_events = parse_client_dir(client_dir)

        router_by_ctx = {k: v for k, v in self.router_events.items() if k != "_no_ctx"}
        matched_router_ctx = set()

        # ---- Assemble combined records (long -> wide) --------------------
        joinable = _joinable_ctx_ids(self.ctx_records, self.gen_records)
        ctx_by_cid = {
            str(r["ctx_request_id"]): r
            for r in self.ctx_records
            if r.get("ctx_request_id") is not None and str(r["ctx_request_id"]) in joinable
        }
        gen_by_cid = {
            str(r["ctx_request_id"]): r
            for r in self.gen_records
            if r.get("ctx_request_id") is not None and str(r["ctx_request_id"]) in joinable
        }
        used_ctx_rids: set = set()
        used_gen_rids: set = set()
        combined: List[Dict[str, Any]] = []
        for cid in sorted(set(ctx_by_cid) | set(gen_by_cid)):
            ctx_rec = ctx_by_cid.get(cid)
            gen_rec = gen_by_cid.get(cid)
            router_rec = router_by_ctx.get(cid)
            if router_rec is not None:
                matched_router_ctx.add(cid)
            # Restore the raw (non-str) ctx id for output when available.
            raw_cid = (gen_rec or ctx_rec).get("ctx_request_id")
            combined.append(_combined_record(raw_cid, ctx_rec, gen_rec, router_rec))
            if ctx_rec is not None:
                used_ctx_rids.add(ctx_rec["request_id"])
            if gen_rec is not None:
                used_gen_rids.add(gen_rec["request_id"])
        # Standalone (non-joinable) records: gen-only path, or null/ambiguous ctx
        # id. These never attach a router record (consistent with "_no_ctx").
        for gen_rec in self.gen_records:
            if gen_rec["request_id"] in used_gen_rids:
                continue
            combined.append(
                _combined_record(gen_rec.get("ctx_request_id"), None, gen_rec, None)
            )
        for ctx_rec in self.ctx_records:
            if ctx_rec["request_id"] in used_ctx_rids:
                continue
            combined.append(
                _combined_record(ctx_rec.get("ctx_request_id"), ctx_rec, None, None)
            )

        # ---- KV-transfer join onto the combined records ------------------
        task_events = self.kv.get("task_events", {})
        gen_summary = self.kv.get("gen_summary", {})
        cpp_events = self.kv.get("cpp_events", {})
        matched_rids = set()
        for rec in combined:
            rids = []
            for key in ("gen_request_id", "ctx_worker_request_id", "ctx_request_id"):
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

        self.records = combined

        # ---- Unjoined reporting ------------------------------------------
        unjoined_tasks = {rid: rows for rid, rows in task_events.items() if rid not in matched_rids}
        unjoined_summary = {rid: row for rid, row in gen_summary.items() if rid not in matched_rids}
        unjoined_cpp = {rid: rows for rid, rows in cpp_events.items() if rid not in matched_rids}
        self.unjoined_kv_events = {
            "task_events": unjoined_tasks,
            "gen_summary": unjoined_summary,
            "cpp_events": unjoined_cpp,
        }

        unjoined_router = [
            row for ctx_id, row in router_by_ctx.items() if ctx_id not in matched_router_ctx
        ]
        unjoined_router.extend(self.router_events.get("_no_ctx", []))
        self.unjoined_router_events = unjoined_router

        all_kv_rids = set(task_events) | set(gen_summary) | set(cpp_events)
        total_kv = len(all_kv_rids)
        matched_kv = len(matched_rids)
        total_router = len(router_by_ctx)
        matched_router = len(matched_router_ctx)
        self.match_stats = {
            "num_records": len(self.records),
            "num_ctx_records": len(self.ctx_records),
            "num_gen_records": len(self.gen_records),
            "num_perf_json_records": len(self.perf_json_records),
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

        return self.records

    def write(self, output_path: str) -> None:
        """Write the combined JSON (records + unjoined KV/router + client + stats)."""
        payload = {
            "records": self.records,
            "perf_json_records": self.perf_json_records,
            "unjoined_kv_events": self.unjoined_kv_events,
            "unjoined_router_events": self.unjoined_router_events,
            # Client send timeline: standalone (no shared key/clock), not joined.
            "client_events": self.client_events,
            "match_stats": self.match_stats,
        }
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"Wrote combined perf time events to {output_path} ({len(self.records)} records)")

    def write_events_jsonl(self, path: str) -> None:
        """Write the combined per-request timeline as JSONL (the primary output).

        One JSON object per line -- one request -- carrying every event
        timestamp that fired plus the derived intra-domain ``spans``. A request
        that hung is still present with its partial set of stamps (later events
        absent, never zero), so the last stamp on the line localizes the stall.
        Ordered by ctx_request_id then gen_request_id for stable output.
        """
        def _sort_key(rec: Dict[str, Any]):
            cid = rec.get("ctx_request_id")
            gid = rec.get("gen_request_id")
            return (str(cid) if cid is not None else "", str(gid) if gid is not None else "")

        with open(path, "w") as f:
            for rec in sorted(self.records, key=_sort_key):
                f.write(json.dumps(rec, default=str) + "\n")
        print(f"Wrote combined time-events JSONL ({len(self.records)} requests) to {path}")

    def write_html(self, html_path: str) -> None:
        """Render the interactive HTML timeline by reusing time_breakdown.

        Operates on the aggregated ``--perf-json`` records (the only shape
        ``time_breakdown``'s parser understands); the per-event combined records
        are not fed here. Imported lazily so the merge/JSON path stays
        dependency-light.
        """
        from ..time_breakdown.time_breakdown import RequestTimeBreakdown

        if not self.perf_json_records:
            print("WARNING: --html needs --perf-json records; none loaded, skipping HTML.")
            return
        analyzer = RequestTimeBreakdown()
        timing_data = []
        for i, rec in enumerate(self.perf_json_records):
            timing_data.append(analyzer.parser.parse_request(rec, i))
            for metric in analyzer.config.metrics:
                duration = metric.calculate_duration(timing_data[-1])
                timing_data[-1][f"{metric.name}_time"] = duration
        analyzer.create_timing_diagram(timing_data, html_path)
        print(f"Wrote HTML timeline to {html_path}")

    def aggregate_metrics(self) -> List[Dict[str, Any]]:
        """Summarize every lifecycle interval as mean / P50 / P99 across requests.

        Also emits min / max / n. Returns one row dict per metric, in stable
        order: the 12 canonical time_breakdown spans (from --perf-json), the
        router dispatch chain, the custom worker spans (from the pivoted
        per-event records), then the vLLM-named client views. Pure stdlib.

        Rank duplication is already collapsed by ``_pivot_worker_events`` (one
        record per logical request), and router rows are unique per request
        (keyed by ctx id); the client rows dedup by response_id here.
        """
        rows: List[Dict[str, Any]] = []

        # ---- Group 1: canonical 12 (from aggregated /perf_metrics records) ----
        seen_ids = set()
        canon_recs = []
        for rec in self.perf_json_records:
            rid = rec.get("request_id")
            key = rid if rid is not None else id(rec)
            if key in seen_ids:
                continue
            seen_ids.add(key)
            canon_recs.append(_extract_canonical_fields(rec))
        for name, sf, ef, source, clock_safe in _CANONICAL_METRICS:
            samples = [_duration(f.get(sf), f.get(ef)) for f in canon_recs]
            rows.append(_stats_row(name, source, clock_safe, samples))

        # ---- Group 2a: router dispatch chain ----
        router_recs = [v for k, v in self.router_events.items() if k != "_no_ctx"]
        router_recs.extend(self.router_events.get("_no_ctx", []))
        for name, sf, ef in _ROUTER_METRICS:
            samples = [_duration(r.get(sf), r.get(ef)) for r in router_recs]
            rows.append(_stats_row(name, "router", True, samples))

        # ---- Group 2b: custom worker spans (from pivoted per-event records) ----
        by_role = {"ctx": self.ctx_records, "gen": self.gen_records}
        for name, role, start_ev, end_ev in _WORKER_EVENT_METRICS:
            recs = by_role[role]
            samples = [_duration(r.get(start_ev), r.get(end_ev)) for r in recs]
            rows.append(_stats_row(name, f"{role}_worker", True, samples))

        # ---- Group 3: vLLM-named client views ----
        # ttft / e2e / tpot: one value per successful client request, stats
        # across requests. Dedup by response_id (client_index fallback).
        # tpot is the client-observed per-output-token time (latency-ttft over
        # the tokens after the first) -- the engine-side per-step ITL series was
        # dropped with the decode series, so this client view replaces it.
        seen_resp = set()
        ttfts, e2es, tpots = [], [], []
        for c in self.client_events:
            if c.get("success") is False:
                continue
            rid = c.get("response_id", c.get("client_index"))
            key = rid if rid is not None else id(c)
            if key in seen_resp:
                continue
            seen_resp.add(key)
            ttft = c.get("ttft")
            latency = c.get("latency")
            out_toks = c.get("output_tokens")
            if ttft is not None:
                ttfts.append(ttft)
            if latency is not None:
                e2es.append(latency)
            if (latency is not None and ttft is not None and out_toks is not None
                    and out_toks > 1):
                tpots.append((latency - ttft) / (out_toks - 1))
        # Client fields are ALREADY in seconds -> scale to ms like the rest.
        rows.append(_stats_row("vllm:ttft", "client", True, ttfts))
        rows.append(_stats_row("vllm:e2e", "client", True, e2es))
        rows.append(_stats_row("vllm:tpot", "client", True, tpots))

        return rows

    def write_agg_jsonl(self, path: str) -> None:
        """Write the latency aggregate as JSONL.

        One JSON object per line, one line per metric (see aggregate_metrics
        for the row schema and order).
        """
        rows = self.aggregate_metrics()
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        n_ok = sum(1 for r in rows if r.get("status") != "not_recorded")
        print(f"Wrote latency aggregate ({n_ok}/{len(rows)} metrics populated) to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compile per-rank / per-process perf time-event logs "
        "(+ KV-transfer CSVs) into one combined per-request timeline."
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
        help="Optional /perf_metrics JSON dump to merge (aggregated runs); "
        "feeds the canonical 12-span group and --html.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="perf_time_events.combined.json",
        help="Output combined JSON path.",
    )
    parser.add_argument(
        "--events-jsonl",
        default=None,
        const="combined_time_events.jsonl",
        nargs="?",
        help="Emit the combined per-request timeline as JSONL -- one line per "
        "request with every event timestamp + derived spans (a hung request "
        "keeps its partial line). Optional path (default: "
        "combined_time_events.jsonl).",
    )
    parser.add_argument(
        "--html",
        default=None,
        const="perf_time_events.timeline.html",
        nargs="?",
        help="Also emit an HTML timeline from --perf-json records (optional path).",
    )
    parser.add_argument(
        "-a",
        "--agg-jsonl",
        default=os.getenv("TRTLLM_PERF_TIME_EVENTS_AGG_PATH") or None,
        const="perf_time_events.latency_agg.jsonl",
        nargs="?",
        help="Also emit a latency-aggregate JSONL (mean / P50 / P99 + min / "
        "max / n of every lifecycle interval; one JSON object per line). "
        "Optional path (default: $TRTLLM_PERF_TIME_EVENTS_AGG_PATH, else "
        "perf_time_events.latency_agg.jsonl). Pure stdlib -- no numpy.",
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
    if args.events_jsonl:
        merger.write_events_jsonl(args.events_jsonl)
    if args.html:
        merger.write_html(args.html)
    if args.agg_jsonl:
        merger.write_agg_jsonl(args.agg_jsonl)


if __name__ == "__main__":
    main()
