#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the perf_time_events offline aggregator.

GPU-free and torch-free: the merge/JSON path is pure stdlib, mirroring
test_time_breakdown.py. Run with:
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


def _make_record(request_id, ctx_request_id=None, step_metrics=None):
    rec = {
        "request_id": request_id,
        "rank": 0,
        "ctx_request_id": ctx_request_id,
        "time_breakdown_metrics": {
            "step_metrics": step_metrics or [],
        },
    }
    return rec


class TestParseEventDir(unittest.TestCase):
    def test_globs_and_concatenates(self):
        with tempfile.TemporaryDirectory() as d:
            _write(
                os.path.join(d, "time_events_rank0_pid100.jsonl"),
                json.dumps(_make_record(1)) + "\n" + json.dumps(_make_record(2)) + "\n",
            )
            _write(
                os.path.join(d, "time_events_rank1_pid100.jsonl"),
                json.dumps(_make_record(3)) + "\n",
            )
            # A blank line + an unrelated file must be ignored.
            _write(os.path.join(d, "time_events_rank2_pid100.jsonl"), "\n")
            _write(os.path.join(d, "other.jsonl"), json.dumps(_make_record(99)) + "\n")

            records = parse_event_dir(d)
            ids = sorted(r["request_id"] for r in records)
            self.assertEqual(ids, [1, 2, 3])

    def test_skips_malformed_lines(self):
        with tempfile.TemporaryDirectory() as d:
            _write(
                os.path.join(d, "time_events_rank0_pid1.jsonl"),
                "{not json}\n" + json.dumps(_make_record(5)) + "\n",
            )
            records = parse_event_dir(d)
            self.assertEqual([r["request_id"] for r in records], [5])


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
    def test_join_by_request_and_ctx_id(self):
        with tempfile.TemporaryDirectory() as ev, tempfile.TemporaryDirectory() as kv:
            # Gen record keyed on request_id=10; its KV rows are keyed on the
            # disagg ctx id "ctxABC".
            _write(
                os.path.join(ev, "time_events_rank0_pid1.jsonl"),
                json.dumps(_make_record(10, ctx_request_id="ctxABC")) + "\n",
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
            # 100% match -> no unjoined rows.
            self.assertEqual(merger.unjoined_kv_events["task_events"], {})
            self.assertEqual(
                merger.match_stats["matched_kv_rids"], merger.match_stats["total_kv_rids"]
            )

    def test_unjoined_kv_events_reported(self):
        with tempfile.TemporaryDirectory() as ev, tempfile.TemporaryDirectory() as kv:
            _write(
                os.path.join(ev, "time_events_rank0_pid1.jsonl"),
                json.dumps(_make_record(10)) + "\n",
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

    def test_derived_inter_step_gaps_and_starved(self):
        with tempfile.TemporaryDirectory() as ev:
            step_metrics = [
                {
                    "forward_start_time": 1.0,
                    "forward_end_time": 1.2,
                    "num_capacity_fitting": 5,
                    "num_scheduled": 4,
                },
                {
                    "forward_start_time": 1.5,  # gap = 1.5 - 1.2 = 0.3
                    "forward_end_time": 1.7,
                    "num_capacity_fitting": 4,
                    "num_scheduled": 4,
                },
            ]
            _write(
                os.path.join(ev, "time_events_rank0_pid1.jsonl"),
                json.dumps(_make_record(10, step_metrics=step_metrics)) + "\n",
            )

            merger = PerfTimeEventsMerger()
            records = merger.merge(event_dir=ev)
            derived = records[0]["derived"]
            self.assertIsNone(derived["inter_step_gaps"][0])
            self.assertAlmostEqual(derived["inter_step_gaps"][1], 0.3)
            self.assertEqual(derived["starved"], [1, 0])

    def test_write_combined_json(self):
        with tempfile.TemporaryDirectory() as ev:
            _write(
                os.path.join(ev, "time_events_rank0_pid1.jsonl"),
                json.dumps(_make_record(10)) + "\n",
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

    def test_merge_perf_json_input(self):
        with tempfile.TemporaryDirectory() as d:
            perf_json = os.path.join(d, "perf.json")
            _write(perf_json, json.dumps([_make_record(1), _make_record(2)]))
            merger = PerfTimeEventsMerger()
            records = merger.merge(perf_json=perf_json)
            self.assertEqual(len(records), 2)


def _router_record(ctx_request_id, **kw):
    rec = {
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
    return rec


class TestParseRouterDir(unittest.TestCase):
    def test_keyed_by_ctx_id_and_no_ctx_collected(self):
        with tempfile.TemporaryDirectory() as d:
            _write(
                os.path.join(d, "disagg_router_pid100.jsonl"),
                json.dumps(_router_record("ctxA"))
                + "\n"
                + json.dumps(_router_record("ctxB"))
                + "\n"
                # gen-only path: no ctx_request_id -> _no_ctx bucket.
                + json.dumps(_router_record(None, disagg_request_id=222))
                + "\n",
            )
            by_ctx = parse_router_dir(d)
            self.assertIn("ctxA", by_ctx)
            self.assertIn("ctxB", by_ctx)
            self.assertIn("_no_ctx", by_ctx)
            self.assertEqual(len(by_ctx["_no_ctx"]), 1)
            self.assertEqual(by_ctx["_no_ctx"][0]["disagg_request_id"], 222)

    def test_ignores_unrelated_files(self):
        with tempfile.TemporaryDirectory() as d:
            _write(os.path.join(d, "other.jsonl"), json.dumps(_router_record("x")) + "\n")
            self.assertEqual(parse_router_dir(d), {})

    def test_duplicate_ctx_id_is_non_joinable(self):
        # The gen-only benchmark path hardcodes ctx_request_id=1 for every
        # request (TRTLLM_DISAGG_BENCHMARK_GEN_ONLY). Such an ambiguous key must
        # NOT stay joinable (that would false-attach one surviving row to every
        # worker record); it is moved to _no_ctx instead.
        with tempfile.TemporaryDirectory() as d:
            _write(
                os.path.join(d, "disagg_router_pid100.jsonl"),
                json.dumps(_router_record("1", disagg_request_id=10))
                + "\n"
                + json.dumps(_router_record("1", disagg_request_id=20))
                + "\n"
                + json.dumps(_router_record("1", disagg_request_id=30))
                + "\n",
            )
            by_ctx = parse_router_dir(d)
            # The ambiguous ctx key is not directly joinable...
            self.assertNotIn("1", by_ctx)
            # ...all three rows are surfaced as leftovers instead.
            self.assertIn("_no_ctx", by_ctx)
            self.assertEqual(len(by_ctx["_no_ctx"]), 3)


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
            # None send_wall_time -> treated as 0.0 -> sorts before 3.0.
            self.assertEqual([r["client_index"] for r in recs], [9, 5])


class TestRouterAndClientMerge(unittest.TestCase):
    def test_router_join_on_ctx_id_and_derived_spans(self):
        with tempfile.TemporaryDirectory() as ev, tempfile.TemporaryDirectory() as rt:
            rec = _make_record(10, ctx_request_id="ctxA")
            # Request-level lifecycle scalars (steady-clock secs) as the worker
            # would enrich them from get_metrics_dict.
            rec["request_timing_metrics"] = {
                "arrival_time": 100.0,
                "first_scheduled_time": 100.5,
                "first_token_time": 101.0,
                "last_token_time": 105.0,
            }
            _write(
                os.path.join(ev, "time_events_rank0_pid1.jsonl"),
                json.dumps(rec) + "\n",
            )
            _write(
                os.path.join(rt, "disagg_router_pid1.jsonl"),
                json.dumps(
                    _router_record(
                        "ctxA",
                        arrival_time=99.0,
                        ctx_dispatch_time=99.2,
                        gen_dispatch_time=100.8,
                        first_token_time=101.0,
                    )
                )
                + "\n",
            )

            merger = PerfTimeEventsMerger()
            records = merger.merge(event_dir=ev, router_dir=rt)
            self.assertEqual(len(records), 1)
            out = records[0]
            self.assertIn("router_dispatch", out)
            self.assertEqual(out["router_dispatch"]["ctx_server"], "ctx0:8000")

            derived = out["derived"]
            # Worker-side request lifecycle spans.
            self.assertAlmostEqual(derived["arrival_to_first_schedule"], 0.5)
            self.assertAlmostEqual(derived["schedule_to_first_token"], 0.5)
            self.assertAlmostEqual(derived["decode_duration"], 4.0)
            # Router-side dispatch spans (steady-clock, same epoch as worker).
            self.assertAlmostEqual(derived["router_arrival_to_ctx_dispatch"], 0.2)
            self.assertAlmostEqual(derived["router_ctx_to_gen_dispatch"], 1.6)
            # Router arrival (99.0) -> worker arrival (100.0).
            self.assertAlmostEqual(derived["router_to_worker_arrival"], 1.0)

            self.assertEqual(merger.match_stats["matched_router_ctx"], 1)
            self.assertEqual(merger.match_stats["total_router_ctx"], 1)
            self.assertEqual(merger.unjoined_router_events, [])

    def test_unjoined_router_events_include_no_ctx_and_leftovers(self):
        with tempfile.TemporaryDirectory() as ev, tempfile.TemporaryDirectory() as rt:
            # A worker record whose ctx id no router row references.
            _write(
                os.path.join(ev, "time_events_rank0_pid1.jsonl"),
                json.dumps(_make_record(10, ctx_request_id="ctxA")) + "\n",
            )
            _write(
                os.path.join(rt, "disagg_router_pid1.jsonl"),
                # ctx-keyed leftover (ctxORPHAN never joins ctxA)...
                json.dumps(_router_record("ctxORPHAN"))
                + "\n"
                # ...plus a gen-only (_no_ctx) record.
                + json.dumps(_router_record(None, disagg_request_id=999))
                + "\n",
            )

            merger = PerfTimeEventsMerger()
            records = merger.merge(event_dir=ev, router_dir=rt)
            self.assertNotIn("router_dispatch", records[0])
            # Both the ctx-keyed leftover and the _no_ctx record surface.
            # (No sort: a None ctx id vs a str ctx id is unorderable in py3.)
            ids = [
                (r.get("ctx_request_id"), r.get("disagg_request_id"))
                for r in merger.unjoined_router_events
            ]
            self.assertIn(("ctxORPHAN", 111), ids)
            self.assertIn((None, 999), ids)
            self.assertEqual(merger.match_stats["matched_router_ctx"], 0)

    def test_gen_only_ctx_id_1_does_not_false_join(self):
        # Regression: under TRTLLM_DISAGG_BENCHMARK_GEN_ONLY the service stamps
        # ctx_request_id=1 on every request, so multiple router rows share it.
        # No worker record should pick up a router_dispatch by that ambiguous
        # key; all router rows fall through to unjoined leftovers.
        with tempfile.TemporaryDirectory() as ev, tempfile.TemporaryDirectory() as rt:
            _write(
                os.path.join(ev, "time_events_rank0_pid1.jsonl"),
                json.dumps(_make_record(10, ctx_request_id="1"))
                + "\n"
                + json.dumps(_make_record(20, ctx_request_id="1"))
                + "\n",
            )
            _write(
                os.path.join(rt, "disagg_router_pid1.jsonl"),
                json.dumps(_router_record("1", disagg_request_id=10))
                + "\n"
                + json.dumps(_router_record("1", disagg_request_id=20))
                + "\n",
            )

            merger = PerfTimeEventsMerger()
            records = merger.merge(event_dir=ev, router_dir=rt)
            self.assertEqual(len(records), 2)
            for out in records:
                self.assertNotIn("router_dispatch", out)
            self.assertEqual(merger.match_stats["matched_router_ctx"], 0)
            # Both router rows surface as leftovers rather than false-joining.
            self.assertEqual(len(merger.unjoined_router_events), 2)

    def test_client_events_standalone_in_payload(self):
        with tempfile.TemporaryDirectory() as ev, tempfile.TemporaryDirectory() as cl:
            _write(
                os.path.join(ev, "time_events_rank0_pid1.jsonl"),
                json.dumps(_make_record(10)) + "\n",
            )
            _write(
                os.path.join(cl, "client_pid1.jsonl"),
                json.dumps(
                    {
                        "source": "client",
                        "client_index": 0,
                        "send_wall_time": 1.0,
                        "ttft": 0.1,
                        "latency": 2.0,
                    }
                )
                + "\n",
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


# ---------------------------------------------------------------------------
# Aggregate path: mean / P50 / P99 of lifecycle intervals (--agg-jsonl)
# ---------------------------------------------------------------------------


def _row(rows, name):
    """Fetch the single aggregate row for a metric name."""
    return next(r for r in rows if r["metric"] == name)


def _gen_record(request_id, rank=0, rtm=None, step_metrics=None, kv_cache_size=301989888):
    """A generation-worker record.

    Discriminated as 'gen' by step_metrics (if given) or a nonzero
    kv_cache_size (default).
    """
    tbm = {}
    if step_metrics is not None:
        tbm["step_metrics"] = step_metrics
    r = dict(rtm or {})
    r.setdefault("kv_cache_size", kv_cache_size)
    return {
        "request_id": request_id,
        "rank": rank,
        "time_breakdown_metrics": tbm,
        "request_timing_metrics": r,
    }


def _ctx_record(request_id, rank=0, rtm=None, ctx_chunk_metrics=None):
    """A context-worker record.

    Discriminated as 'ctx' by ctx_chunk_metrics and the absence of
    step_metrics / nonzero kv_cache_size.
    """
    tbm = {}
    if ctx_chunk_metrics is not None:
        tbm["ctx_chunk_metrics"] = ctx_chunk_metrics
    rec = {"request_id": request_id, "rank": rank, "time_breakdown_metrics": tbm}
    if rtm is not None:
        rec["request_timing_metrics"] = rtm
    return rec


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


# Every metric name the aggregate emits, in stable emission order.
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
    "ctx:forward_start->sampler_end",
    "gen:kv_transfer_start->end",
    "ctx:arrival->first_scheduled",
    "gen:arrival->first_scheduled",
    "vllm:ttft",
    "vllm:e2e",
    "vllm:tpot",
    "vllm:itl",
]


class TestAggregateHelpers(unittest.TestCase):
    def test_percentile_math(self):
        self.assertEqual(pte._percentile([], 50), 0.0)
        self.assertEqual(pte._percentile([42.0], 99), 42.0)
        self.assertEqual(pte._percentile([1, 2, 3, 4, 5], 50), 3.0)
        # Linear interpolation between order statistics.
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

    def test_infer_role(self):
        self.assertEqual(pte._infer_role(_gen_record(1, step_metrics=[{"token_time": 1.0}])), "gen")
        # No step_metrics, but nonzero kv_cache_size -> gen.
        self.assertEqual(pte._infer_role(_gen_record(1, step_metrics=None)), "gen")
        self.assertEqual(
            pte._infer_role(_ctx_record(1, ctx_chunk_metrics=[{"forward_start_time": 1.0}])),
            "ctx",
        )
        # kv_cache_transfer_start must NOT flip a ctx record to gen.
        ctx = _ctx_record(
            1,
            rtm={"kv_cache_transfer_start": 217445.4, "kv_cache_size": 0},
            ctx_chunk_metrics=[{"forward_start_time": 1.0}],
        )
        self.assertEqual(pte._infer_role(ctx), "ctx")


class TestAggregateMetrics(unittest.TestCase):
    def _agg(self, records=None, router_events=None, client_events=None):
        m = PerfTimeEventsMerger()
        m.records = records or []
        m.router_events = router_events or {}
        m.client_events = client_events or []
        return m.aggregate_metrics()

    def test_all_rows_present_and_ordered(self):
        rows = self._agg()
        self.assertEqual([r["metric"] for r in rows], _EXPECTED_METRICS)

    def test_empty_input_all_not_recorded(self):
        rows = self._agg()
        self.assertEqual(len(rows), 25)
        for r in rows:
            self.assertEqual(r["status"], "not_recorded")
            self.assertIn("unit", r)
            self.assertIn("source", r)
            self.assertIn("clock_safe", r)
            self.assertNotIn("mean", r)

    def test_gen_arrival_to_first_scheduled_stats(self):
        recs = [
            _gen_record(i, rtm={"arrival_time": base, "first_scheduled_time": base + ms / 1000.0})
            for i, (base, ms) in enumerate([(100.0, 10), (200.0, 20), (300.0, 30), (400.0, 40)])
        ]
        row = _row(self._agg(records=recs), "gen:arrival->first_scheduled")
        self.assertEqual(row["source"], "gen_worker")
        self.assertTrue(row["clock_safe"])
        self.assertEqual(row["unit"], "ms")
        self.assertEqual(row["n"], 4)
        self.assertAlmostEqual(row["mean"], 25.0, places=3)
        self.assertAlmostEqual(row["p50"], 25.0, places=3)
        self.assertAlmostEqual(row["p99"], 39.7, places=3)
        self.assertAlmostEqual(row["min"], 10.0, places=3)
        self.assertAlmostEqual(row["max"], 40.0, places=3)

    def test_rank_duplication_deduped(self):
        # request 777 written by 4 TP ranks (lockstep-identical) + one more
        # request. Ranks must collapse: n == 2 unique requests, not 8.
        recs = []
        for rank in range(4):
            recs.append(
                _gen_record(
                    777, rank=rank, rtm={"arrival_time": 100.0, "first_scheduled_time": 100.01}
                )
            )
        recs.append(_gen_record(888, rtm={"arrival_time": 200.0, "first_scheduled_time": 200.02}))
        row = _row(self._agg(records=recs), "gen:arrival->first_scheduled")
        self.assertEqual(row["n"], 2)
        self.assertAlmostEqual(row["mean"], 15.0, places=3)
        self.assertAlmostEqual(row["min"], 10.0, places=3)
        self.assertAlmostEqual(row["max"], 20.0, places=3)

    def test_zeros_excluded(self):
        recs = [
            _gen_record(1, rtm={"arrival_time": 100.0, "first_scheduled_time": 100.05}),  # 50 ms
            _gen_record(
                2, rtm={"arrival_time": 100.0, "first_scheduled_time": 99.0}
            ),  # inverted -> 0
            _gen_record(
                3, rtm={"arrival_time": 100.0, "first_scheduled_time": 100.0}
            ),  # zero-width
        ]
        row = _row(self._agg(records=recs), "gen:arrival->first_scheduled")
        self.assertEqual(row["n"], 1)
        self.assertAlmostEqual(row["mean"], 50.0, places=3)
        self.assertAlmostEqual(row["min"], 50.0, places=3)
        self.assertAlmostEqual(row["max"], 50.0, places=3)

    def test_kv_span_not_recorded_on_zero_setters(self):
        # Sync disagg receive path: kv_cache_transfer_start==end==0.0 (#15871).
        recs = [
            _gen_record(i, rtm={"kv_cache_transfer_start": 0.0, "kv_cache_transfer_end": 0.0})
            for i in range(4)
        ]
        row = _row(self._agg(records=recs), "gen:kv_transfer_start->end")
        self.assertEqual(row["status"], "not_recorded")
        self.assertEqual(row["source"], "gen_worker")
        self.assertNotIn("mean", row)

    def test_ctx_forward_span_skips_warmup(self):
        recs = [
            # Warmup: no request_timing_metrics -> skipped by the rtm gate.
            _ctx_record(
                1024, ctx_chunk_metrics=[{"forward_start_time": 1.0, "sample_end_time": 1.5}]
            ),
            _ctx_record(
                1,
                rtm={"arrival_time": 217445.3},
                ctx_chunk_metrics=[
                    {"forward_start_time": 217445.30, "sample_end_time": 217445.31},
                    {"forward_start_time": 217445.33, "sample_end_time": 217445.34},  # 40 ms span
                ],
            ),
        ]
        row = _row(self._agg(records=recs), "ctx:forward_start->sampler_end")
        self.assertEqual(row["source"], "ctx_worker")
        self.assertEqual(row["n"], 1)  # warmup excluded
        self.assertAlmostEqual(row["mean"], 40.0, places=3)

    def test_router_chain_stats(self):
        router_events = {
            "ctxA": {
                "source": "disagg_router",
                "arrival_time": 46790.0,
                "ctx_dispatch_time": 46790.001,  # 1 ms
                "gen_dispatch_time": 46790.301,
                "first_token_time": 46792.0,
                "resp_done_time": 46794.0,
            },
            "ctxB": {
                "source": "disagg_router",
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
        # gen-only path: router record with no ctx_request_id lands in _no_ctx.
        router_events = {
            "_no_ctx": [
                {"arrival_time": 46790.0, "gen_dispatch_time": 46790.1},  # 100 ms
                {"arrival_time": 46791.0, "gen_dispatch_time": 46791.2},  # 200 ms
            ]
        }
        row = _row(self._agg(router_events=router_events), "router:arrival->gen_dispatch")
        self.assertEqual(row["n"], 2)
        self.assertAlmostEqual(row["mean"], 150.0, places=3)

    def test_itl_and_tpot_from_step_metrics(self):
        recA = _gen_record(
            1,
            step_metrics=[
                {"token_time": 10.0},
                {"token_time": 10.002},  # 2 ms
                {"token_time": 10.005},  # 3 ms
                {"token_time": 10.009},  # 4 ms
            ],
        )
        recB = _gen_record(
            2,
            step_metrics=[
                {"token_time": 20.0},
                {"token_time": 20.010},  # 10 ms
                {"token_time": 20.030},  # 20 ms
            ],
        )
        rows = self._agg(records=[recA, recB])
        tpot = _row(rows, "vllm:tpot")
        itl = _row(rows, "vllm:itl")
        # tpot: per-request mean ITL -> [3ms, 15ms] across 2 requests.
        self.assertEqual(tpot["source"], "gen_worker")
        self.assertEqual(tpot["n"], 2)
        self.assertAlmostEqual(tpot["mean"], 9.0, places=3)
        self.assertAlmostEqual(tpot["min"], 3.0, places=3)
        self.assertAlmostEqual(tpot["max"], 15.0, places=3)
        # itl: pooled across every inter-token gap -> [2,3,4,10,20] ms.
        self.assertEqual(itl["n"], 5)
        self.assertAlmostEqual(itl["mean"], 7.8, places=3)
        self.assertAlmostEqual(itl["p50"], 4.0, places=3)
        self.assertAlmostEqual(itl["min"], 2.0, places=3)
        self.assertAlmostEqual(itl["max"], 20.0, places=3)

    def test_vllm_ttft_e2e_from_client_events(self):
        client_events = [
            {"source": "client", "response_id": 1, "success": True, "ttft": 2.72, "latency": 5.13},
            {"source": "client", "response_id": 2, "success": True, "ttft": 2.74, "latency": 5.15},
            # A failed request is excluded.
            {"source": "client", "response_id": 3, "success": False, "ttft": 9.9, "latency": 99.0},
            # A duplicate response_id is deduped.
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

    def test_clock_safe_flags(self):
        rec = _disagg_perf_record(
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
        rows = self._agg(records=[rec])
        # The two cross-domain disagg spans are flagged, not dropped.
        self.assertFalse(_row(rows, "disagg_preprocessing")["clock_safe"])
        self.assertFalse(_row(rows, "disagg_relay")["clock_safe"])
        # Intra-domain spans are clock_safe and populated.
        ctx_queue = _row(rows, "ctx_queue")
        self.assertTrue(ctx_queue["clock_safe"])
        self.assertAlmostEqual(ctx_queue["mean"], 50.0, places=3)  # 1000.25 - 1000.2
        relay = _row(rows, "disagg_relay")
        self.assertAlmostEqual(relay["mean"], 200.0, places=3)  # 1001.2 - 1001.0

    def test_canonical_rows_not_recorded_on_per_rank_capture(self):
        # A pure per-rank worker record carries no nested perf_metrics -> the
        # 12 canonical rows are not_recorded (they need --perf-json input).
        rows = self._agg(records=[_gen_record(1, rtm={"arrival_time": 1.0})])
        for name, _s, _e, _src, _cs in pte._CANONICAL_METRICS:
            self.assertEqual(_row(rows, name)["status"], "not_recorded")

    def test_write_agg_jsonl_roundtrip(self):
        recs = [
            _gen_record(i, rtm={"arrival_time": 100.0 + i, "first_scheduled_time": 100.05 + i})
            for i in range(3)
        ]
        m = PerfTimeEventsMerger()
        m.records = recs
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "agg.jsonl")
            m.write_agg_jsonl(out)
            with open(out) as f:
                lines = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(lines), 25)
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
