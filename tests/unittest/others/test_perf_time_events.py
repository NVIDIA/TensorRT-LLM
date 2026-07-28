#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the perf_time_events offline aggregator.

GPU-free and torch-free: the merge/JSON path is pure stdlib, mirroring
test_time_breakdown.py. Run with:
    python -m pytest tests/unittest/others/test_perf_time_events.py -v
"""

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


if __name__ == "__main__":
    unittest.main()
