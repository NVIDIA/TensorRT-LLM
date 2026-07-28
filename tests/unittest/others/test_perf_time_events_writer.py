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

from tensorrt_llm.serve.perf_time_events_writer import (
    CLIENT_EVENTS_PATH_ENV,
    ROUTER_EVENTS_PATH_ENV,
    JsonlEventWriter,
    make_env_writer,
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


if __name__ == "__main__":
    unittest.main()
