# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU-free / torch-free tests for the stdlib JSONL event writer.

``tensorrt_llm.serve.perf_time_events_writer.JsonlEventWriter`` is the off-hot
-path writer the disagg router and the benchmark client use (both must stay
torch-free -- see ``test_import_gpu_free.py``). This pins its contract: inert
when no directory is configured, one JSON line per record drained on
``close()``, per-pid filenames, and non-blocking enqueue.

Run with:
    python -m pytest tests/unittest/others/test_perf_time_events_writer.py -v
"""

import glob
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from tensorrt_llm.serve import perf_time_events_writer as wr
from tensorrt_llm.serve.perf_time_events_writer import (
    CLIENT_EVENTS_PATH_ENV,
    ROUTER_EVENTS_PATH_ENV,
    WORKER_EVENTS_PATH_ENV,
    JsonlEventWriter,
    emit_event,
    make_env_writer,
    make_event_record,
    set_worker_rank,
)


class TestJsonlEventWriter(unittest.TestCase):
    def test_inert_when_no_dir(self):
        # Falsy events_dir -> inert: enabled False, write a no-op, close safe.
        writer = JsonlEventWriter("", "disagg_router")
        self.assertFalse(writer.enabled)
        writer.write({"source": "disagg_router", "x": 1})  # must not raise
        writer.close()  # safe even though no thread ever started

    def test_writes_one_line_per_record_and_drains_on_close(self):
        with tempfile.TemporaryDirectory() as d:
            writer = JsonlEventWriter(d, "disagg_router")
            self.assertTrue(writer.enabled)
            writer.write({"source": "disagg_router", "ctx_request_id": 1})
            writer.write({"source": "disagg_router", "ctx_request_id": 2})
            writer.close()  # drains + joins the daemon thread

            files = glob.glob(os.path.join(d, "disagg_router_pid*.jsonl"))
            self.assertEqual(len(files), 1)
            self.assertIn(f"pid{os.getpid()}", os.path.basename(files[0]))
            with open(files[0]) as f:
                lines = [json.loads(ln) for ln in f if ln.strip()]
            self.assertEqual([r["ctx_request_id"] for r in lines], [1, 2])

    def test_non_serializable_value_falls_back_to_str(self):
        # default=str keeps a non-JSON-native value from killing the writer
        # thread (and losing every subsequent record).
        with tempfile.TemporaryDirectory() as d:
            writer = JsonlEventWriter(d, "client")
            writer.write({"source": "client", "weird": {1, 2, 3}})
            writer.close()
            files = glob.glob(os.path.join(d, "client_pid*.jsonl"))
            self.assertEqual(len(files), 1)
            with open(files[0]) as f:
                lines = [json.loads(ln) for ln in f if ln.strip()]
            self.assertEqual(len(lines), 1)
            # The set was stringified rather than raising TypeError.
            self.assertIsInstance(lines[0]["weird"], str)

    def test_close_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            writer = JsonlEventWriter(d, "disagg_router")
            writer.write({"source": "disagg_router"})
            writer.close()
            writer.close()  # second call is a no-op, must not raise


class TestMakeEnvWriter(unittest.TestCase):
    def test_inert_when_env_unset(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ROUTER_EVENTS_PATH_ENV, None)
            writer = make_env_writer(ROUTER_EVENTS_PATH_ENV, "disagg_router")
            self.assertFalse(writer.enabled)

    def test_enabled_when_env_set(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.dict(os.environ, {CLIENT_EVENTS_PATH_ENV: d}):
                writer = make_env_writer(CLIENT_EVENTS_PATH_ENV, "client")
                self.assertTrue(writer.enabled)
                writer.write({"source": "client", "client_index": 0})
                writer.close()
            files = glob.glob(os.path.join(d, "client_pid*.jsonl"))
            self.assertEqual(len(files), 1)


class TestMakeEventRecord(unittest.TestCase):
    def test_flat_envelope_fields_present(self):
        rec = make_event_record("gen", "gen_arrival", request_id=10,
                                 ctx_request_id="cA", rank=3, t=1.5)
        self.assertEqual(rec["role"], "gen")
        self.assertEqual(rec["event"], "gen_arrival")
        self.assertEqual(rec["request_id"], 10)
        self.assertEqual(rec["ctx_request_id"], "cA")
        self.assertEqual(rec["rank"], 3)
        self.assertEqual(rec["t"], 1.5)
        # pid is always stamped from the current process.
        self.assertEqual(rec["pid"], os.getpid())

    def test_defaults_are_none_but_keys_present(self):
        # Even unset fields exist as keys so the compiler pivot never KeyErrors.
        rec = make_event_record("router", "arrival")
        for k in ("role", "event", "request_id", "ctx_request_id", "rank", "t", "pid"):
            self.assertIn(k, rec)
        self.assertIsNone(rec["request_id"])
        self.assertIsNone(rec["ctx_request_id"])
        self.assertIsNone(rec["rank"])
        self.assertIsNone(rec["t"])

    def test_extra_provenance_merged(self):
        rec = make_event_record("router", "gen_dispatch", request_id=0,
                                 disagg_request_id=222, ctx_server="c:8000",
                                 gen_server="g:8001")
        self.assertEqual(rec["disagg_request_id"], 222)
        self.assertEqual(rec["ctx_server"], "c:8000")
        self.assertEqual(rec["gen_server"], "g:8001")


class TestEmitEvent(unittest.TestCase):
    """emit_event is the worker-side helper. It lazily builds a process-global
    writer keyed on TRTLLM_PERF_TIME_EVENTS_PATH; reset that global between
    tests so each one observes its own env. The steady-clock default is patched
    so this test never imports torch (the real _lazy_steady_clock_now pulls in
    tensorrt_llm.bindings)."""

    def setUp(self):
        self._saved_writer = wr._WORKER_WRITER
        self._saved_rank = wr._WORKER_RANK
        wr._WORKER_WRITER = None
        wr._WORKER_RANK = 0

    def tearDown(self):
        w = wr._WORKER_WRITER
        if w is not None:
            w.close()
        wr._WORKER_WRITER = self._saved_writer
        wr._WORKER_RANK = self._saved_rank

    def test_inert_when_env_unset(self):
        # No TRTLLM_PERF_TIME_EVENTS_PATH -> emit_event is a no-op and never
        # touches the steady clock (would import torch). No file appears.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(WORKER_EVENTS_PATH_ENV, None)
            with patch.object(wr, "_lazy_steady_clock_now",
                              side_effect=AssertionError("clock touched")):
                emit_event("gen", "gen_arrival", request_id=1, ctx_request_id="c")
            self.assertFalse(wr._get_worker_writer().enabled)

    def test_emits_flat_line_with_worker_filename(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.dict(os.environ, {WORKER_EVENTS_PATH_ENV: d}):
                set_worker_rank(2)
                emit_event("gen", "gen_arrival", request_id=7,
                           ctx_request_id="cA", t=1.25)
                emit_event("gen", "gen_first_token", request_id=7,
                           ctx_request_id="cA", t=1.75)
                wr._get_worker_writer().close()
            files = glob.glob(os.path.join(d, "time_events_rank2_pid*.jsonl"))
            self.assertEqual(len(files), 1)
            self.assertIn(f"pid{os.getpid()}", os.path.basename(files[0]))
            with open(files[0]) as f:
                lines = [json.loads(ln) for ln in f if ln.strip()]
            self.assertEqual([r["event"] for r in lines],
                             ["gen_arrival", "gen_first_token"])
            self.assertEqual(lines[0]["role"], "gen")
            self.assertEqual(lines[0]["request_id"], 7)
            self.assertEqual(lines[0]["ctx_request_id"], "cA")
            self.assertEqual(lines[0]["rank"], 2)  # pinned rank stamped
            self.assertEqual(lines[0]["t"], 1.25)

    def test_defaults_rank_and_t_when_omitted(self):
        # rank defaults to the pinned worker rank; t defaults via the (patched)
        # steady clock, so the caller need not pass either.
        with tempfile.TemporaryDirectory() as d:
            with patch.dict(os.environ, {WORKER_EVENTS_PATH_ENV: d}):
                set_worker_rank(5)
                with patch.object(wr, "_lazy_steady_clock_now", return_value=42.0):
                    emit_event("ctx", "ctx_arrival", request_id=3,
                               ctx_request_id="cZ")
                wr._get_worker_writer().close()
            files = glob.glob(os.path.join(d, "time_events_rank5_pid*.jsonl"))
            self.assertEqual(len(files), 1)
            with open(files[0]) as f:
                rec = json.loads(f.readline())
            self.assertEqual(rec["rank"], 5)
            self.assertEqual(rec["t"], 42.0)


if __name__ == "__main__":
    unittest.main()
