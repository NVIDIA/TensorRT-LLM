#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the perf_time_events offline compiler (per-event long -> wide).

GPU-free and torch-free: the parse/merge/JSON path is pure stdlib, mirroring
test_time_breakdown.py. Every process now appends ONE JSONL line per lifecycle
event (worker/router) -- the compiler groups those lines by request, joins
ctx <-> gen <-> router on ctx_request_id, dedups TP ranks, and pivots long ->
wide into one combined record per request (a hung request keeps its partial
stamps). The benchmark client stays a compound one-record-per-request dump.

Run with:
    python -m pytest tests/unittest/others/test_perf_time_events.py -v
"""

import ast
import json
import os
import tempfile
import unittest

from tensorrt_llm.serve.scripts.perf_time_events import (
    PerfTimeEventsMerger,
    parse_client_dir,
    parse_event_dir,
    parse_kv_csv_dir,
    parse_router_dir,
)

# Internal aggregate-path helpers (not re-exported by the package __init__).
from tensorrt_llm.serve.scripts.perf_time_events import perf_time_events as pte

# Header strings copied from _torch/disaggregation/native/perf_logger.py so a
# drift in the producer header trips these tests.
_NATIVE_TASK_HEADER = (
    "timestamp,task_type,unique_rid,peer_rank,"
    "transfer_size_bytes,avg_segment_size_bytes,"
    "transfer_entry_count,prepare_args_latency_ms,"
    "queue_latency_ms,transfer_latency_ms,task_latency_ms,"
    "throughput_mbs"
)
_GEN_SUMMARY_HEADER = "timestamp,RequestID,gen_side_transfer_time(ms),kv_cache_size"


def _write(path, text):
    with open(path, "w") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# Per-event line builders (mirror perf_time_events_writer.make_event_record):
# {"role","event","request_id","ctx_request_id","rank","t","pid",**extra}
# ---------------------------------------------------------------------------


def _ev(role, event, request_id, ctx_request_id=None, rank=0, t=None, pid=100, **extra):
    rec = {
        "role": role,
        "event": event,
        "request_id": request_id,
        "ctx_request_id": ctx_request_id,
        "rank": rank,
        "t": t,
        "pid": pid,
    }
    rec.update(extra)
    return rec


def _gen_lines(request_id, ctx_request_id=None, rank=0, pid=100, **events):
    """One gen-worker record's worth of event lines. ``events`` maps event
    name (e.g. gen_arrival=..) -> steady-clock t."""
    return [
        _ev("gen", name, request_id, ctx_request_id, rank=rank, t=t, pid=pid)
        for name, t in events.items()
    ]


def _ctx_lines(request_id, ctx_request_id=None, rank=0, pid=100, **events):
    return [
        _ev("ctx", name, request_id, ctx_request_id, rank=rank, t=t, pid=pid)
        for name, t in events.items()
    ]


def _router_lines(router_seq, ctx_request_id=None, pid=100, disagg_request_id=111,
                  ctx_server="ctx0:8000", gen_server="gen0:8001", **events):
    """One router request's event lines, keyed by (pid, router_seq).

    ``ctx_request_id`` is attached only from ``gen_dispatch`` onward (the ctx
    round-trip assigns it), so ``arrival`` / ``ctx_dispatch`` carry None -- the
    compiler resolves it as the first non-null across the group.
    """
    early = {"arrival", "ctx_dispatch"}
    lines = []
    for name, t in events.items():
        cid = None if name in early else ctx_request_id
        lines.append(_ev(
            "router", name, router_seq, cid, rank=0, t=t, pid=pid,
            disagg_request_id=disagg_request_id,
            ctx_server=ctx_server, gen_server=gen_server,
        ))
    return lines


class TestParseEventDir(unittest.TestCase):
    def test_globs_and_returns_flat_event_list(self):
        with tempfile.TemporaryDirectory() as d:
            _write(
                os.path.join(d, "time_events_rank0_pid100.jsonl"),
                json.dumps(_ev("gen", "gen_arrival", 1, "cA", t=1.0)) + "\n"
                + json.dumps(_ev("gen", "gen_first_token", 1, "cA", t=1.5)) + "\n",
            )
            _write(
                os.path.join(d, "time_events_rank1_pid100.jsonl"),
                json.dumps(_ev("ctx", "ctx_arrival", 3, "cB", t=2.0)) + "\n",
            )
            # A blank line + an unrelated file must be ignored.
            _write(os.path.join(d, "time_events_rank2_pid100.jsonl"), "\n")
            _write(os.path.join(d, "other.jsonl"), json.dumps(_ev("gen", "x", 99)) + "\n")

            events = parse_event_dir(d)
            # One entry per line across time_events_* files only.
            self.assertEqual(len(events), 3)
            self.assertEqual({e["event"] for e in events},
                             {"gen_arrival", "gen_first_token", "ctx_arrival"})

    def test_filename_sort_puts_rank0_first(self):
        # parse_event_dir sorts by filename so rank0 lines precede rank1 -- the
        # guarantee the TP-rank dedup (first-seen wins) relies on.
        with tempfile.TemporaryDirectory() as d:
            _write(os.path.join(d, "time_events_rank1_pid1.jsonl"),
                   json.dumps(_ev("gen", "gen_arrival", 5, "c", rank=1, t=9.0)) + "\n")
            _write(os.path.join(d, "time_events_rank0_pid1.jsonl"),
                   json.dumps(_ev("gen", "gen_arrival", 5, "c", rank=0, t=1.0)) + "\n")
            events = parse_event_dir(d)
            self.assertEqual([e["rank"] for e in events], [0, 1])

    def test_skips_malformed_lines(self):
        with tempfile.TemporaryDirectory() as d:
            _write(
                os.path.join(d, "time_events_rank0_pid1.jsonl"),
                "{not json}\n" + json.dumps(_ev("gen", "gen_arrival", 5, t=1.0)) + "\n",
            )
            events = parse_event_dir(d)
            self.assertEqual([e["event"] for e in events], ["gen_arrival"])


class TestPivotWorkerEvents(unittest.TestCase):
    def test_pivots_ctx_and_gen_into_separate_records(self):
        events = (
            _gen_lines(10, "cA", gen_arrival=100.0, gen_first_scheduled=100.5)
            + _ctx_lines(5, "cA", ctx_arrival=90.0, ctx_first_token=90.3)
        )
        ctx_r, gen_r = pte._pivot_worker_events(events)
        self.assertEqual(len(ctx_r), 1)
        self.assertEqual(len(gen_r), 1)
        self.assertEqual(gen_r[0]["request_id"], 10)
        self.assertEqual(gen_r[0]["ctx_request_id"], "cA")
        self.assertAlmostEqual(gen_r[0]["gen_arrival"], 100.0)
        self.assertAlmostEqual(ctx_r[0]["ctx_first_token"], 90.3)

    def test_tp_ranks_collapse_first_seen_wins(self):
        # request 777 written by 4 TP ranks (lockstep). parse_event_dir's
        # filename sort makes rank0 canonical; the pivot keeps its t.
        with tempfile.TemporaryDirectory() as d:
            for rank in range(4):
                _write(
                    os.path.join(d, f"time_events_rank{rank}_pid1.jsonl"),
                    json.dumps(_ev("gen", "gen_arrival", 777, "cZ", rank=rank,
                                   t=100.0 + rank)) + "\n"
                    + json.dumps(_ev("gen", "gen_first_scheduled", 777, "cZ",
                                     rank=rank, t=100.01 + rank)) + "\n",
                )
            ctx_r, gen_r = pte._pivot_worker_events(parse_event_dir(d))
            self.assertEqual(len(gen_r), 1)  # 4 ranks collapsed to 1 request
            self.assertAlmostEqual(gen_r[0]["gen_arrival"], 100.0)  # rank0 wins
            self.assertAlmostEqual(gen_r[0]["gen_first_scheduled"], 100.01)

    def test_partial_hung_request_keeps_only_fired_events(self):
        # A gen request that wedged after init_scheduled: no first/last token.
        events = _gen_lines(10, "cA", gen_arrival=100.0, gen_init_scheduled=100.1)
        _ctx, gen_r = pte._pivot_worker_events(events)
        self.assertEqual(len(gen_r), 1)
        self.assertIn("gen_init_scheduled", gen_r[0])
        self.assertNotIn("gen_first_token", gen_r[0])
        self.assertNotIn("gen_last_token", gen_r[0])

    def test_unknown_event_name_ignored(self):
        # A bogus event is skipped BEFORE the group is created, so on its own it
        # forms no record; a valid sibling still does, without the bogus key.
        events = [
            _ev("gen", "not_a_real_event", 1, "c", t=1.0),
            _ev("gen", "gen_arrival", 1, "c", t=2.0),
        ]
        _ctx, gen_r = pte._pivot_worker_events(events)
        self.assertEqual(len(gen_r), 1)
        self.assertNotIn("not_a_real_event", gen_r[0])
        self.assertAlmostEqual(gen_r[0]["gen_arrival"], 2.0)

    def test_unknown_event_alone_forms_no_group(self):
        events = [_ev("gen", "not_a_real_event", 1, "c", t=1.0)]
        _ctx, gen_r = pte._pivot_worker_events(events)
        self.assertEqual(gen_r, [])


class TestParseKvCsvDir(unittest.TestCase):
    def test_native_task_and_gen_summary(self):
        with tempfile.TemporaryDirectory() as d:
            _write(
                os.path.join(d, "gen_instance_0.csv"),
                _NATIVE_TASK_HEADER + "\n"
                "1.0,KVSendTask,abc,1,1024,512,2,0.1,0.2,0.3,0.6,100\n"
                "2.0,KVRecvTask,abc,0,1024,,,,,,,\n",
            )
            _write(
                os.path.join(d, "gen_instance_0_gen_transfer_summary.csv"),
                _GEN_SUMMARY_HEADER + "\n3.0,abc,0.6,1024\n",
            )

            kv = parse_kv_csv_dir(d)
            self.assertIn("abc", kv["task_events"])
            self.assertEqual(len(kv["task_events"]["abc"]), 2)
            self.assertEqual(kv["task_events"]["abc"][0]["task_type"], "KVSendTask")
            self.assertIn("abc", kv["gen_summary"])
            self.assertEqual(kv["gen_summary"]["abc"]["kv_cache_size"], "1024")

    def test_cpp_send_recv_keyed_on_request_id(self):
        with tempfile.TemporaryDirectory() as d:
            _write(os.path.join(d, "ctx_0_send.csv"), "RequestID,transfer_time_ms\n77,1.5\n")
            kv = parse_kv_csv_dir(d)
            self.assertIn("77", kv["cpp_events"])
            self.assertEqual(kv["cpp_events"]["77"][0]["transfer_time_ms"], "1.5")


class TestMerge(unittest.TestCase):
    def test_join_kv_by_ctx_id(self):
        with tempfile.TemporaryDirectory() as ev, tempfile.TemporaryDirectory() as kv:
            # Gen record request_id=10, disagg ctx id "ctxABC"; KV rows keyed on it.
            _write(
                os.path.join(ev, "time_events_rank0_pid1.jsonl"),
                "\n".join(json.dumps(e) for e in _gen_lines(
                    10, "ctxABC", gen_arrival=1.0, gen_first_token=2.0)) + "\n",
            )
            _write(
                os.path.join(kv, "gen_0.csv"),
                _NATIVE_TASK_HEADER + "\n1.0,KVRecvTask,ctxABC,0,1024,,,,,,,\n",
            )
            _write(
                os.path.join(kv, "gen_0_gen_transfer_summary.csv"),
                _GEN_SUMMARY_HEADER + "\n2.0,ctxABC,0.6,1024\n",
            )

            merger = PerfTimeEventsMerger()
            records = merger.merge(event_dir=ev, kv_csv_dir=kv)

            self.assertEqual(len(records), 1)
            rec = records[0]
            self.assertIn("kv_transfer_events", rec)
            self.assertEqual(rec["kv_transfer_events"][0]["task_type"], "KVRecvTask")
            self.assertIn("kv_gen_summary", rec)
            self.assertEqual(merger.unjoined_kv_events["task_events"], {})
            self.assertEqual(
                merger.match_stats["matched_kv_rids"], merger.match_stats["total_kv_rids"]
            )

    def test_unjoined_kv_events_reported(self):
        with tempfile.TemporaryDirectory() as ev, tempfile.TemporaryDirectory() as kv:
            _write(
                os.path.join(ev, "time_events_rank0_pid1.jsonl"),
                "\n".join(json.dumps(e) for e in _gen_lines(
                    10, "cX", gen_arrival=1.0)) + "\n",
            )
            # KV row for an rid that no record references.
            _write(
                os.path.join(kv, "gen_0.csv"),
                _NATIVE_TASK_HEADER + "\n1.0,KVSendTask,orphan,0,1024,512,1,0.1,0.2,0.3,0.6,100\n",
            )

            merger = PerfTimeEventsMerger()
            merger.merge(event_dir=ev, kv_csv_dir=kv)
            self.assertIn("orphan", merger.unjoined_kv_events["task_events"])
            self.assertEqual(merger.match_stats["matched_kv_rids"], 0)

    def test_write_combined_json(self):
        with tempfile.TemporaryDirectory() as ev:
            _write(
                os.path.join(ev, "time_events_rank0_pid1.jsonl"),
                "\n".join(json.dumps(e) for e in _gen_lines(
                    10, "cA", gen_arrival=1.0)) + "\n",
            )
            merger = PerfTimeEventsMerger()
            merger.merge(event_dir=ev)
            out = os.path.join(ev, "combined.json")
            merger.write(out)
            with open(out) as f:
                payload = json.load(f)
            self.assertEqual(len(payload["records"]), 1)
            self.assertIn("match_stats", payload)
            self.assertIn("unjoined_kv_events", payload)

    def test_merge_perf_json_feeds_perf_json_records(self):
        # --perf-json is the aggregated single-server path: it feeds the
        # canonical-12 group (and --html), not the per-event combined records.
        with tempfile.TemporaryDirectory() as d:
            perf_json = os.path.join(d, "perf.json")
            _write(perf_json, json.dumps([
                _disagg_perf_record(**_FULL_DISAGG_TIMES),
                _disagg_perf_record(**_FULL_DISAGG_TIMES),
            ]))
            merger = PerfTimeEventsMerger()
            merger.merge(perf_json=perf_json)
            self.assertEqual(len(merger.perf_json_records), 2)
            # No worker event lines -> no combined per-request records.
            self.assertEqual(len(merger.records), 0)


def _router_record(ctx_request_id, **kw):
    """Legacy compound router record shape (still accepted by aggregate_metrics,
    which reads the pivoted ``*_time`` fields directly)."""
    return {
        "source": "disagg_router",
        "ctx_request_id": ctx_request_id,
        "disagg_request_id": kw.get("disagg_request_id", 111),
        "ctx_server": kw.get("ctx_server", "ctx0:8000"),
        "gen_server": kw.get("gen_server", "gen0:8001"),
        "arrival_time": kw.get("arrival_time"),
        "ctx_dispatch_time": kw.get("ctx_dispatch_time"),
        "gen_dispatch_time": kw.get("gen_dispatch_time"),
        "first_token_time": kw.get("first_token_time"),
        "resp_done_time": kw.get("resp_done_time"),
    }


class TestParseRouterDir(unittest.TestCase):
    def test_pivots_per_event_lines_keyed_by_ctx_id(self):
        with tempfile.TemporaryDirectory() as d:
            lines = (
                _router_lines(0, "ctxA", arrival=1.0, ctx_dispatch=1.1,
                              gen_dispatch=1.5, first_token=2.0, resp_done=3.0)
                + _router_lines(1, "ctxB", arrival=4.0, gen_dispatch=4.5)
                # gen-only path: ctx id never resolves -> _no_ctx bucket.
                + _router_lines(2, None, disagg_request_id=222, arrival=5.0,
                                gen_dispatch=5.2)
            )
            _write(os.path.join(d, "disagg_router_pid100.jsonl"),
                   "\n".join(json.dumps(x) for x in lines) + "\n")
            by_ctx = parse_router_dir(d)
            self.assertIn("ctxA", by_ctx)
            self.assertIn("ctxB", by_ctx)
            # ctxA pivoted to the *_time field names _ROUTER_METRICS differences.
            self.assertAlmostEqual(by_ctx["ctxA"]["arrival_time"], 1.0)
            self.assertAlmostEqual(by_ctx["ctxA"]["resp_done_time"], 3.0)
            self.assertEqual(by_ctx["ctxA"]["ctx_server"], "ctx0:8000")
            self.assertIn("_no_ctx", by_ctx)
            self.assertEqual(len(by_ctx["_no_ctx"]), 1)
            self.assertEqual(by_ctx["_no_ctx"][0]["disagg_request_id"], 222)

    def test_ignores_unrelated_files(self):
        with tempfile.TemporaryDirectory() as d:
            _write(os.path.join(d, "other.jsonl"),
                   json.dumps(_ev("router", "arrival", 0, "x", t=1.0)) + "\n")
            self.assertEqual(parse_router_dir(d), {})

    def test_duplicate_ctx_id_is_non_joinable(self):
        # The gen-only benchmark path hardcodes ctx_request_id=1 for every
        # request. Such an ambiguous key must NOT stay joinable (that would
        # false-attach one surviving row to every worker record); the rows are
        # moved to _no_ctx instead.
        with tempfile.TemporaryDirectory() as d:
            lines = (
                _router_lines(0, "1", disagg_request_id=10, arrival=1.0, gen_dispatch=1.2)
                + _router_lines(1, "1", disagg_request_id=20, arrival=2.0, gen_dispatch=2.2)
                + _router_lines(2, "1", disagg_request_id=30, arrival=3.0, gen_dispatch=3.2)
            )
            _write(os.path.join(d, "disagg_router_pid100.jsonl"),
                   "\n".join(json.dumps(x) for x in lines) + "\n")
            by_ctx = parse_router_dir(d)
            self.assertNotIn("1", by_ctx)
            self.assertIn("_no_ctx", by_ctx)
            self.assertEqual(len(by_ctx["_no_ctx"]), 3)

    def test_lines_split_across_pids_group_independently(self):
        # Two router processes can reuse router_seq=0; (pid, request_id) keeps
        # them distinct.
        with tempfile.TemporaryDirectory() as d:
            _write(os.path.join(d, "disagg_router_pid100.jsonl"),
                   "\n".join(json.dumps(x) for x in _router_lines(
                       0, "cA", pid=100, arrival=1.0, gen_dispatch=1.2)) + "\n")
            _write(os.path.join(d, "disagg_router_pid200.jsonl"),
                   "\n".join(json.dumps(x) for x in _router_lines(
                       0, "cB", pid=200, arrival=2.0, gen_dispatch=2.2)) + "\n")
            by_ctx = parse_router_dir(d)
            self.assertIn("cA", by_ctx)
            self.assertIn("cB", by_ctx)


class TestParseClientDir(unittest.TestCase):
    def test_sorted_by_send_wall_time(self):
        with tempfile.TemporaryDirectory() as d:
            _write(
                os.path.join(d, "client_pid100.jsonl"),
                json.dumps({"source": "client", "client_index": 1, "send_wall_time": 2.0})
                + "\n"
                + json.dumps({"source": "client", "client_index": 0, "send_wall_time": 1.0})
                + "\n",
            )
            recs = parse_client_dir(d)
            self.assertEqual([r["client_index"] for r in recs], [0, 1])

    def test_missing_wall_time_sorts_first(self):
        with tempfile.TemporaryDirectory() as d:
            _write(
                os.path.join(d, "client_pid1.jsonl"),
                json.dumps({"source": "client", "client_index": 5, "send_wall_time": 3.0})
                + "\n"
                + json.dumps({"source": "client", "client_index": 9})
                + "\n",
            )
            recs = parse_client_dir(d)
            self.assertEqual([r["client_index"] for r in recs], [9, 5])


class TestCombinedRecords(unittest.TestCase):
    def test_join_ctx_gen_router_and_spans(self):
        # ctx worker (request_id=5) and gen worker (request_id=10) are distinct
        # processes joined only by ctx_request_id="ctxA"; the router adds its
        # dispatch chain by the same key.
        with tempfile.TemporaryDirectory() as ev, tempfile.TemporaryDirectory() as rt:
            worker = (
                _ctx_lines(5, "ctxA", ctx_arrival=100.0, ctx_first_scheduled=100.5,
                           ctx_first_token=101.0, ctx_ready_sent=101.2)
                + _gen_lines(10, "ctxA", gen_arrival=200.0, gen_init_scheduled=200.1,
                             gen_kv_transfer_start=200.3, gen_kv_transfer_end=200.9,
                             gen_first_scheduled=201.0, gen_first_token=201.5,
                             gen_last_token=205.0)
            )
            _write(os.path.join(ev, "time_events_rank0_pid1.jsonl"),
                   "\n".join(json.dumps(e) for e in worker) + "\n")
            _write(os.path.join(rt, "disagg_router_pid1.jsonl"),
                   "\n".join(json.dumps(x) for x in _router_lines(
                       0, "ctxA", arrival=99.0, ctx_dispatch=99.2, gen_dispatch=100.8,
                       first_token=101.0, resp_done=205.5)) + "\n")

            merger = PerfTimeEventsMerger()
            records = merger.merge(event_dir=ev, router_dir=rt)
            self.assertEqual(len(records), 1)
            rec = records[0]
            self.assertEqual(rec["ctx_request_id"], "ctxA")
            self.assertEqual(rec["gen_request_id"], 10)
            self.assertEqual(rec["ctx_worker_request_id"], 5)
            self.assertEqual(rec["ctx_server"], "ctx0:8000")

            spans = rec["spans"]
            # ctx worker spans.
            self.assertAlmostEqual(spans["ctx:arrival->first_scheduled"], 0.5)
            self.assertAlmostEqual(spans["ctx:first_token->ready_sent"], 0.2, places=5)
            # gen worker spans -- KV transfer now from #11/#12 stamps (not 0.0).
            self.assertAlmostEqual(spans["gen:kv_transfer_start->end"], 0.6, places=5)
            self.assertAlmostEqual(spans["gen:first_token->last_token"], 3.5, places=5)
            # router dispatch chain (same host/clock).
            self.assertAlmostEqual(spans["router:arrival->ctx_dispatch"], 0.2, places=5)
            self.assertAlmostEqual(spans["router:ctx_dispatch->gen_dispatch"], 1.6, places=5)

            self.assertEqual(merger.match_stats["matched_router_ctx"], 1)
            self.assertEqual(merger.match_stats["total_router_ctx"], 1)
            self.assertEqual(merger.unjoined_router_events, [])

    def test_hung_request_emits_partial_record(self):
        # A gen request that wedged during KV transfer: arrival + init_scheduled
        # + kv_transfer_start fired, nothing after. The record is still emitted
        # with exactly those stamps -- the last one localizes the stall.
        with tempfile.TemporaryDirectory() as ev:
            worker = _gen_lines(10, "ctxH", gen_arrival=100.0, gen_init_scheduled=100.1,
                                gen_kv_transfer_start=100.3)
            _write(os.path.join(ev, "time_events_rank0_pid1.jsonl"),
                   "\n".join(json.dumps(e) for e in worker) + "\n")
            merger = PerfTimeEventsMerger()
            records = merger.merge(event_dir=ev)
            self.assertEqual(len(records), 1)
            rec = records[0]
            self.assertIn("gen_kv_transfer_start", rec)
            self.assertNotIn("gen_kv_transfer_end", rec)
            self.assertNotIn("gen_first_token", rec)
            spans = rec["spans"]
            self.assertIn("gen:arrival->init_scheduled", spans)
            # No end stamp -> the transfer span is absent, never a bogus 0.0.
            self.assertNotIn("gen:kv_transfer_start->end", spans)
            self.assertNotIn("gen:first_token->last_token", spans)

    def test_gen_only_ctx_id_1_does_not_false_join(self):
        # Under the gen-only benchmark the service stamps ctx_request_id=1 on
        # every request; no worker record should pick up a router_dispatch by
        # that ambiguous key. Both gen requests stay standalone.
        with tempfile.TemporaryDirectory() as ev, tempfile.TemporaryDirectory() as rt:
            worker = (
                _gen_lines(10, "1", gen_arrival=100.0, gen_first_token=101.0)
                + _gen_lines(20, "1", gen_arrival=200.0, gen_first_token=201.0)
            )
            _write(os.path.join(ev, "time_events_rank0_pid1.jsonl"),
                   "\n".join(json.dumps(e) for e in worker) + "\n")
            _write(os.path.join(rt, "disagg_router_pid1.jsonl"),
                   "\n".join(json.dumps(x) for x in (
                       _router_lines(0, "1", disagg_request_id=10, arrival=99.0, gen_dispatch=99.5)
                       + _router_lines(1, "1", disagg_request_id=20, arrival=199.0, gen_dispatch=199.5)
                   )) + "\n")

            merger = PerfTimeEventsMerger()
            records = merger.merge(event_dir=ev, router_dir=rt)
            self.assertEqual(len(records), 2)
            for rec in records:
                self.assertNotIn("router_arrival_time", rec)
                self.assertEqual(rec["ctx_request_id"], "1")
            self.assertEqual(merger.match_stats["matched_router_ctx"], 0)
            self.assertEqual(len(merger.unjoined_router_events), 2)

    def test_unjoined_router_leftover_and_no_ctx(self):
        with tempfile.TemporaryDirectory() as ev, tempfile.TemporaryDirectory() as rt:
            _write(os.path.join(ev, "time_events_rank0_pid1.jsonl"),
                   "\n".join(json.dumps(e) for e in _gen_lines(
                       10, "ctxA", gen_arrival=100.0)) + "\n")
            _write(os.path.join(rt, "disagg_router_pid1.jsonl"),
                   "\n".join(json.dumps(x) for x in (
                       _router_lines(0, "ctxORPHAN", arrival=1.0, gen_dispatch=1.2)
                       + _router_lines(1, None, disagg_request_id=999, arrival=2.0, gen_dispatch=2.2)
                   )) + "\n")
            merger = PerfTimeEventsMerger()
            records = merger.merge(event_dir=ev, router_dir=rt)
            self.assertNotIn("router_arrival_time", records[0])
            ids = [
                (r.get("ctx_request_id"), r.get("disagg_request_id"))
                for r in merger.unjoined_router_events
            ]
            self.assertIn(("ctxORPHAN", 111), ids)
            self.assertIn((None, 999), ids)
            self.assertEqual(merger.match_stats["matched_router_ctx"], 0)

    def test_client_events_standalone_in_payload(self):
        with tempfile.TemporaryDirectory() as ev, tempfile.TemporaryDirectory() as cl:
            _write(os.path.join(ev, "time_events_rank0_pid1.jsonl"),
                   "\n".join(json.dumps(e) for e in _gen_lines(
                       10, "cA", gen_arrival=1.0)) + "\n")
            _write(
                os.path.join(cl, "client_pid1.jsonl"),
                json.dumps({
                    "source": "client", "client_index": 0, "send_wall_time": 1.0,
                    "ttft": 0.1, "latency": 2.0,
                }) + "\n",
            )
            merger = PerfTimeEventsMerger()
            merger.merge(event_dir=ev, client_dir=cl)
            self.assertEqual(len(merger.client_events), 1)
            self.assertEqual(merger.match_stats["num_client_events"], 1)
            # Client records are NOT joined onto request records.
            self.assertNotIn("client", merger.records[0])

            out = os.path.join(cl, "combined.json")
            merger.write(out)
            with open(out) as f:
                payload = json.load(f)
            self.assertEqual(len(payload["client_events"]), 1)
            self.assertIn("unjoined_router_events", payload)

    def test_write_events_jsonl_one_line_per_request(self):
        with tempfile.TemporaryDirectory() as ev:
            worker = (
                _gen_lines(10, "cA", gen_arrival=100.0, gen_first_token=101.0,
                           gen_last_token=105.0)
                + _gen_lines(20, "cB", gen_arrival=200.0, gen_first_token=201.0)
            )
            _write(os.path.join(ev, "time_events_rank0_pid1.jsonl"),
                   "\n".join(json.dumps(e) for e in worker) + "\n")
            merger = PerfTimeEventsMerger()
            merger.merge(event_dir=ev)
            out = os.path.join(ev, "combined_time_events.jsonl")
            merger.write_events_jsonl(out)
            with open(out) as f:
                lines = [json.loads(ln) for ln in f if ln.strip()]
            self.assertEqual(len(lines), 2)
            # Sorted by ctx id -> cA then cB.
            self.assertEqual([ln["ctx_request_id"] for ln in lines], ["cA", "cB"])
            self.assertIn("gen:first_token->last_token", lines[0]["spans"])
            # The second request only reached first_token -> no decode span.
            self.assertNotIn("gen:first_token->last_token", lines[1].get("spans", {}))


# ---------------------------------------------------------------------------
# Aggregate path: mean / P50 / P99 of lifecycle intervals (--agg-jsonl)
# ---------------------------------------------------------------------------


def _row(rows, name):
    """Fetch the single aggregate row for a metric name."""
    return next(r for r in rows if r["metric"] == name)


def _gen_wrec(request_id, ctx_request_id=None, **events):
    """A pivoted gen-worker record (post long->wide): request_id + event stamps."""
    r = {"request_id": request_id, "ctx_request_id": ctx_request_id}
    r.update(events)
    return r


def _ctx_wrec(request_id, ctx_request_id=None, **events):
    r = {"request_id": request_id, "ctx_request_id": ctx_request_id}
    r.update(events)
    return r


def _disagg_perf_record(**tm):
    """A disaggregated /perf_metrics-shaped record (the --perf-json input).

    Has nested ctx_perf_metrics / gen_perf_metrics, mirroring what
    time_breakdown.RequestDataParser.parse_request consumes.
    """
    return {
        "ctx_perf_metrics": {
            "request_id": tm.get("request_id", "r0"),
            "perf_metrics": {
                "timing_metrics": {
                    "server_arrival_time": tm["ctx_server_arrival_time"],
                    "arrival_time": tm["ctx_arrival_time"],
                    "first_scheduled_time": tm["ctx_first_scheduled_time"],
                    "first_token_time": tm["ctx_first_token_time"],
                    "server_first_token_time": tm["ctx_server_first_token_time"],
                }
            },
        },
        "gen_perf_metrics": {
            "perf_metrics": {
                "timing_metrics": {
                    "server_arrival_time": tm["gen_server_arrival_time"],
                    "arrival_time": tm["gen_arrival_time"],
                    "first_scheduled_time": tm["gen_first_scheduled_time"],
                    "kv_cache_transfer_start": tm["gen_kv_cache_transfer_start"],
                    "kv_cache_transfer_end": tm["gen_kv_cache_transfer_end"],
                    "server_first_token_time": tm["gen_server_first_token_time"],
                }
            }
        },
        "disagg_server_arrival_time": tm["disagg_server_arrival_time"],
        "disagg_server_first_token_time": tm["disagg_server_first_token_time"],
    }


_FULL_DISAGG_TIMES = dict(
    disagg_server_arrival_time=999.9,
    ctx_server_arrival_time=1000.0,
    ctx_arrival_time=1000.2,
    ctx_first_scheduled_time=1000.25,
    ctx_first_token_time=1000.5,
    ctx_server_first_token_time=1001.0,
    gen_server_arrival_time=1001.2,
    gen_arrival_time=1001.3,
    gen_first_scheduled_time=1001.35,
    gen_kv_cache_transfer_start=1001.31,
    gen_kv_cache_transfer_end=1001.33,
    gen_server_first_token_time=1001.5,
    disagg_server_first_token_time=1001.6,
)


# Every metric name the aggregate emits, in stable emission order (24 total):
# 12 canonical (--perf-json) + 5 router + 4 worker-event + 3 client.
_EXPECTED_METRICS = [
    "disagg_preprocessing",
    "ctx_preprocessing",
    "ctx_queue",
    "ctx_processing",
    "ctx_postprocessing",
    "disagg_relay",
    "gen_preprocessing",
    "gen_queue_wait",
    "gen_kv_transfer",
    "gen_post_transfer",
    "gen_postprocessing",
    "disagg_postprocessing",
    "router:arrival->ctx_dispatch",
    "router:arrival->gen_dispatch",
    "router:ctx_dispatch->gen_dispatch",
    "router:gen_dispatch->first_token",
    "router:first_token->resp_done",
    "ctx:arrival->first_scheduled",
    "gen:arrival->first_scheduled",
    "gen:kv_transfer_start->end",
    "gen:first_token->last_token",
    "vllm:ttft",
    "vllm:e2e",
    "vllm:tpot",
]


class TestAggregateHelpers(unittest.TestCase):
    def test_percentile_math(self):
        self.assertEqual(pte._percentile([], 50), 0.0)
        self.assertEqual(pte._percentile([42.0], 99), 42.0)
        self.assertEqual(pte._percentile([1, 2, 3, 4, 5], 50), 3.0)
        self.assertAlmostEqual(pte._percentile([10, 20, 30, 40], 50), 25.0)
        self.assertAlmostEqual(pte._percentile([10, 20, 30, 40], 99), 39.7)

    def test_duration_excludes_missing_and_inverted(self):
        self.assertEqual(pte._duration(1.0, 3.0), 2.0)
        self.assertEqual(pte._duration(None, 3.0), 0.0)
        self.assertEqual(pte._duration(3.0, 1.0), 0.0)  # inverted
        self.assertEqual(pte._duration(float("nan"), 3.0), 0.0)
        self.assertEqual(pte._duration(2.0, 2.0), 0.0)  # zero-width

    def test_nz_treats_zero_as_missing(self):
        self.assertIsNone(pte._nz(0.0))
        self.assertIsNone(pte._nz(None))
        self.assertEqual(pte._nz(1.5), 1.5)

    def test_span_or_none(self):
        self.assertAlmostEqual(pte._span_or_none(1.0, 3.0), 2.0)
        self.assertIsNone(pte._span_or_none(None, 3.0))
        self.assertIsNone(pte._span_or_none(3.0, 1.0))  # inverted -> None
        self.assertIsNone(pte._span_or_none(2.0, 2.0))  # zero-width -> None


class TestAggregateMetrics(unittest.TestCase):
    def _agg(self, ctx_records=None, gen_records=None, perf_json_records=None,
             router_events=None, client_events=None):
        m = PerfTimeEventsMerger()
        m.ctx_records = ctx_records or []
        m.gen_records = gen_records or []
        m.perf_json_records = perf_json_records or []
        m.router_events = router_events or {}
        m.client_events = client_events or []
        return m.aggregate_metrics()

    def test_all_rows_present_and_ordered(self):
        rows = self._agg()
        self.assertEqual([r["metric"] for r in rows], _EXPECTED_METRICS)

    def test_empty_input_all_not_recorded(self):
        rows = self._agg()
        self.assertEqual(len(rows), 24)
        for r in rows:
            self.assertEqual(r["status"], "not_recorded")
            self.assertIn("unit", r)
            self.assertIn("source", r)
            self.assertIn("clock_safe", r)
            self.assertNotIn("mean", r)

    def test_gen_arrival_to_first_scheduled_stats(self):
        recs = [
            _gen_wrec(i, gen_arrival=base, gen_first_scheduled=base + ms / 1000.0)
            for i, (base, ms) in enumerate([(100.0, 10), (200.0, 20), (300.0, 30), (400.0, 40)])
        ]
        row = _row(self._agg(gen_records=recs), "gen:arrival->first_scheduled")
        self.assertEqual(row["source"], "gen_worker")
        self.assertTrue(row["clock_safe"])
        self.assertEqual(row["unit"], "ms")
        self.assertEqual(row["n"], 4)
        self.assertAlmostEqual(row["mean"], 25.0, places=3)
        self.assertAlmostEqual(row["p50"], 25.0, places=3)
        self.assertAlmostEqual(row["p99"], 39.7, places=3)
        self.assertAlmostEqual(row["min"], 10.0, places=3)
        self.assertAlmostEqual(row["max"], 40.0, places=3)

    def test_zeros_excluded(self):
        recs = [
            _gen_wrec(1, gen_arrival=100.0, gen_first_scheduled=100.05),   # 50 ms
            _gen_wrec(2, gen_arrival=100.0, gen_first_scheduled=99.0),     # inverted -> 0
            _gen_wrec(3, gen_arrival=100.0, gen_first_scheduled=100.0),    # zero-width
        ]
        row = _row(self._agg(gen_records=recs), "gen:arrival->first_scheduled")
        self.assertEqual(row["n"], 1)
        self.assertAlmostEqual(row["mean"], 50.0, places=3)

    def test_kv_span_from_events(self):
        recs = [
            _gen_wrec(i, gen_kv_transfer_start=200.0, gen_kv_transfer_end=200.0 + ms / 1000.0)
            for i, ms in enumerate([2, 4, 6])
        ]
        row = _row(self._agg(gen_records=recs), "gen:kv_transfer_start->end")
        self.assertEqual(row["source"], "gen_worker")
        self.assertEqual(row["n"], 3)
        self.assertAlmostEqual(row["mean"], 4.0, places=3)

    def test_kv_span_not_recorded_when_events_absent(self):
        # Requests that never emitted the KV-transfer stamps (e.g. hung earlier).
        recs = [_gen_wrec(i, gen_arrival=100.0) for i in range(4)]
        row = _row(self._agg(gen_records=recs), "gen:kv_transfer_start->end")
        self.assertEqual(row["status"], "not_recorded")
        self.assertEqual(row["source"], "gen_worker")
        self.assertNotIn("mean", row)

    def test_decode_wall_span_from_events(self):
        recs = [
            _gen_wrec(1, gen_first_token=101.0, gen_last_token=105.0),   # 4 s
            _gen_wrec(2, gen_first_token=201.0, gen_last_token=203.0),   # 2 s
        ]
        row = _row(self._agg(gen_records=recs), "gen:first_token->last_token")
        self.assertEqual(row["source"], "gen_worker")
        self.assertEqual(row["n"], 2)
        self.assertAlmostEqual(row["mean"], 3000.0, places=3)  # 3 s -> ms

    def test_ctx_arrival_span_from_events(self):
        recs = [_ctx_wrec(1, ctx_arrival=90.0, ctx_first_scheduled=90.5)]  # 500 ms
        row = _row(self._agg(ctx_records=recs), "ctx:arrival->first_scheduled")
        self.assertEqual(row["source"], "ctx_worker")
        self.assertEqual(row["n"], 1)
        self.assertAlmostEqual(row["mean"], 500.0, places=3)

    def test_router_chain_stats(self):
        router_events = {
            "ctxA": {
                "arrival_time": 46790.0,
                "ctx_dispatch_time": 46790.001,  # 1 ms
                "gen_dispatch_time": 46790.301,
                "first_token_time": 46792.0,
                "resp_done_time": 46794.0,
            },
            "ctxB": {
                "arrival_time": 46795.0,
                "ctx_dispatch_time": 46795.003,  # 3 ms
                "gen_dispatch_time": 46795.303,
                "first_token_time": 46797.0,
                "resp_done_time": 46799.0,
            },
        }
        rows = self._agg(router_events=router_events)
        row = _row(rows, "router:arrival->ctx_dispatch")
        self.assertEqual(row["source"], "router")
        self.assertTrue(row["clock_safe"])
        self.assertEqual(row["n"], 2)
        self.assertAlmostEqual(row["mean"], 2.0, places=3)
        self.assertAlmostEqual(row["min"], 1.0, places=3)
        self.assertAlmostEqual(row["max"], 3.0, places=3)

    def test_router_no_ctx_bucket_included(self):
        router_events = {
            "_no_ctx": [
                {"arrival_time": 46790.0, "gen_dispatch_time": 46790.1},  # 100 ms
                {"arrival_time": 46791.0, "gen_dispatch_time": 46791.2},  # 200 ms
            ]
        }
        row = _row(self._agg(router_events=router_events), "router:arrival->gen_dispatch")
        self.assertEqual(row["n"], 2)
        self.assertAlmostEqual(row["mean"], 150.0, places=3)

    def test_tpot_from_client(self):
        # tpot = (latency - ttft) / (output_tokens - 1), per successful request.
        client_events = [
            {"source": "client", "response_id": 1, "success": True,
             "ttft": 1.0, "latency": 5.0, "output_tokens": 5},   # (4)/(4)=1.0s
            {"source": "client", "response_id": 2, "success": True,
             "ttft": 2.0, "latency": 8.0, "output_tokens": 4},   # (6)/(3)=2.0s
        ]
        row = _row(self._agg(client_events=client_events), "vllm:tpot")
        self.assertEqual(row["source"], "client")
        self.assertEqual(row["n"], 2)
        self.assertAlmostEqual(row["mean"], 1500.0, places=3)  # (1.0+2.0)/2 s -> ms

    def test_vllm_ttft_e2e_from_client_events(self):
        client_events = [
            {"source": "client", "response_id": 1, "success": True, "ttft": 2.72, "latency": 5.13},
            {"source": "client", "response_id": 2, "success": True, "ttft": 2.74, "latency": 5.15},
            {"source": "client", "response_id": 3, "success": False, "ttft": 9.9, "latency": 99.0},
            {"source": "client", "response_id": 1, "success": True, "ttft": 2.72, "latency": 5.13},
        ]
        rows = self._agg(client_events=client_events)
        ttft = _row(rows, "vllm:ttft")
        e2e = _row(rows, "vllm:e2e")
        self.assertEqual(ttft["source"], "client")
        self.assertEqual(ttft["n"], 2)
        self.assertAlmostEqual(ttft["mean"], 2730.0, places=3)
        self.assertEqual(e2e["n"], 2)
        self.assertAlmostEqual(e2e["mean"], 5140.0, places=3)

    def test_clock_safe_flags_and_canonical_from_perf_json(self):
        rec = _disagg_perf_record(**_FULL_DISAGG_TIMES)
        rows = self._agg(perf_json_records=[rec])
        # The two cross-domain disagg spans are flagged, not dropped.
        self.assertFalse(_row(rows, "disagg_preprocessing")["clock_safe"])
        self.assertFalse(_row(rows, "disagg_relay")["clock_safe"])
        ctx_queue = _row(rows, "ctx_queue")
        self.assertTrue(ctx_queue["clock_safe"])
        self.assertAlmostEqual(ctx_queue["mean"], 50.0, places=3)  # 1000.25 - 1000.2
        relay = _row(rows, "disagg_relay")
        self.assertAlmostEqual(relay["mean"], 200.0, places=3)  # 1001.2 - 1001.0

    def test_canonical_rows_not_recorded_on_per_event_capture(self):
        # A pure per-event worker capture carries no nested perf_metrics -> the
        # 12 canonical rows are not_recorded (they need --perf-json input).
        rows = self._agg(gen_records=[_gen_wrec(1, gen_arrival=1.0)])
        for name, _s, _e, _src, _cs in pte._CANONICAL_METRICS:
            self.assertEqual(_row(rows, name)["status"], "not_recorded")

    def test_write_agg_jsonl_roundtrip(self):
        recs = [
            _gen_wrec(i, gen_arrival=100.0 + i, gen_first_scheduled=100.05 + i)
            for i in range(3)
        ]
        m = PerfTimeEventsMerger()
        m.gen_records = recs
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "agg.jsonl")
            m.write_agg_jsonl(out)
            with open(out) as f:
                lines = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(lines), 24)
        self.assertEqual([r["metric"] for r in lines], _EXPECTED_METRICS)
        gen_row = _row(lines, "gen:arrival->first_scheduled")
        self.assertEqual(gen_row["n"], 3)


class TestAggregatePathIsDependencyLight(unittest.TestCase):
    def test_no_toplevel_numpy_or_time_breakdown_import(self):
        """The aggregate path must stay stdlib-only.

        No module-scope import of numpy or time_breakdown (both are pulled in
        lazily inside write_html only). Guards the module's
        importable-outside-the-container promise.
        """
        with open(pte.__file__) as f:
            tree = ast.parse(f.read())
        banned = ("numpy", "plotly")
        for node in tree.body:  # module scope only
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotIn(alias.name.split(".")[0], banned)
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                self.assertNotIn(root, banned)
                self.assertNotIn("time_breakdown", (node.module or ""))

    def test_canonical_matches_time_breakdown(self):
        """Drift guard for the hardcoded canonical-12 spec.

        The spec must stay in lockstep with time_breakdown.TimingMetricsConfig.
        Skips where that module (numpy / plotly) is unavailable.
        """
        try:
            from tensorrt_llm.serve.scripts.time_breakdown.time_breakdown import TimingMetricsConfig
        except ImportError as e:  # numpy/plotly absent outside the others container
            self.skipTest(f"time_breakdown unimportable: {e}")
        canonical = {n: (sf, ef) for n, sf, ef, _src, _cs in pte._CANONICAL_METRICS}
        official = {m.name: (m.start_field, m.end_field) for m in TimingMetricsConfig().metrics}
        self.assertEqual(canonical, official)


if __name__ == "__main__":
    unittest.main()
