# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise production direct-submit ordering without importing CUDA libraries."""

from __future__ import annotations

import ast
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[3] / (
    "tensorrt_llm/_torch/moe/fused_moe/mega_moe/rebalance_slot_scheduler_v2.py"
)


class Routes:
    def __init__(self, rows, trace, *, topk=6, dtype="int32", device="cuda:0", contiguous=True):
        self.shape = (rows, topk)
        self.ndim = 2
        self.dtype = dtype
        self.device = device
        self.contiguous = contiguous
        self.trace = trace

    def is_contiguous(self):
        return self.contiguous

    def record_stream(self, stream):
        self.trace.append(("lifetime", stream.cuda_stream))

    def __getitem__(self, rows):
        return Routes(rows.stop, self.trace)


class DirectSubmitContract(unittest.TestCase):
    def setUp(self):
        self.trace = []
        self.current_device = 0
        self.thread_id = 7
        self.main = SimpleNamespace(cuda_stream=44, priority=0)
        copy = SimpleNamespace(cuda_stream=19, priority=-1)
        cuda = SimpleNamespace(
            current_device=lambda: self.current_device,
            current_stream=lambda _: self.main,
            nvtx=SimpleNamespace(range=lambda _: nullcontext()),
        )
        tree = ast.parse(SOURCE.read_text())
        selected = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name in {"RebalanceSlotSchedulerGroupV2", "_V2LiveBankLeaseProvider"}
        ]
        module = ast.Module(
            body=[
                ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                *selected,
            ],
            type_ignores=[],
        )
        namespace = {
            "torch": SimpleNamespace(cuda=cuda, Tensor=Routes, int32="int32"),
            "threading": SimpleNamespace(get_ident=lambda: self.thread_id),
        }
        exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
        cls = namespace["RebalanceSlotSchedulerGroupV2"]
        group = cls.__new__(cls)
        for name in (
            "_generation",
            "_finished_generation",
            "_scheduled_generation",
            "_route_wait_generation",
            "plan_calls",
        ):
            setattr(group, name, 0)
        group._plan_part = None
        group._pending_release = None
        group._owner_thread_id = None
        group._execution_stream_handle = None
        group.device = 0
        group.topk = 6
        group.max_tokens_per_rank = 8192
        group.copy_stream = group.scheduler_stream = copy
        group._stream_handle = 19
        group._input_ready_handle = 1
        group._plan_ready_handle = 2
        group._consumer_done_handle = 3
        group._consumer_done = SimpleNamespace(cuda_event=3)
        group._driver = SimpleNamespace(
            CUresult=SimpleNamespace(CUDA_SUCCESS=0),
            cuEventRecord=lambda event, stream: self.record("record", event, stream),
            cuStreamWaitEvent=lambda stream, event, flags: self.record("wait", stream, event),
        )
        self.pending = False
        self.outputs = SimpleNamespace(physical_slot_ids=Routes(8192, self.trace))
        group.scheduler = SimpleNamespace(device="cuda:0", submit=self.submit)
        group.broadcaster = SimpleNamespace(
            submit=self.copy_submit,
            release_generation_after=lambda generation, event: self.trace.append(
                ("consumer_release", generation, event.cuda_event)
            ),
            mark_collective_reuse_safe=lambda provider, generation: self.trace.append(
                ("lease", generation)
            ),
        )
        group.lease = namespace["_V2LiveBankLeaseProvider"](group)
        self.group = group

    def record(self, *entry):
        self.trace.append(entry)
        return (0,)

    def submit(self, routes, handle):
        self.assertFalse(self.pending)
        self.assertEqual(handle, 19)
        self.pending = True
        self.trace.append(("HALO", routes.shape[0], id(routes)))
        return self.outputs

    def copy_submit(self, outputs):
        self.assertIs(outputs, self.outputs)
        self.assertTrue(self.pending)
        self.pending = False
        generation = self.group._scheduled_generation
        self.trace.append(("TMA", generation))
        return SimpleNamespace(generation=generation)

    def complete(self, rows):
        routes = Routes(rows, self.trace)
        part = self.group.plan_schedule(routes)
        physical, generation = self.group.plan_finish(part, defer_wait=True)
        self.assertEqual(physical.shape, (rows, 6))
        self.assertFalse(any(entry == ("wait", 44, 2) for entry in self.trace[-5:]))
        self.group.wait_for_routes()
        self.group.wait_for_routes()
        self.group.finish()
        return routes, generation

    def test_original_pointer_and_late_wait_for_shrinking_and_empty_inputs(self):
        for generation, rows in enumerate((8192, 8096, 0, 16), 1):
            self.trace.clear()
            routes, actual_generation = self.complete(rows)
            expected = [
                ("lifetime", 19),
                ("record", 1, 44),
                ("wait", 19, 1),
                ("HALO", rows, id(routes)),
                ("record", 2, 19),
            ]
            if generation > 1:
                expected.append(("lease", generation - 1))
            expected.extend(
                [
                    ("TMA", generation),
                    ("wait", 44, 2),
                    ("record", 3, 44),
                    ("consumer_release", generation, 3),
                ]
            )
            self.assertEqual(actual_generation, generation)
            self.assertEqual(self.trace, expected)

    def test_rejects_invalid_metadata_before_any_device_work(self):
        inputs = [
            Routes(8193, self.trace),
            Routes(8, self.trace, dtype="int64"),
            Routes(8, self.trace, contiguous=False),
            Routes(8, self.trace, topk=5),
            Routes(8, self.trace, device="cuda:1"),
            object(),
        ]
        for routes in inputs:
            with self.assertRaises((ValueError, RuntimeError)):
                self.group.plan_schedule(routes)
            self.assertEqual(self.trace, [])

    def test_owner_and_generation_guards_remain(self):
        self.complete(16)
        self.trace.clear()
        self.thread_id += 1
        with self.assertRaisesRegex(RuntimeError, "one MAIN thread"):
            self.group.plan_schedule(Routes(8, self.trace))
        self.thread_id -= 1
        self.main.cuda_stream = 55
        with self.assertRaisesRegex(RuntimeError, "one MAIN thread"):
            self.group.plan_schedule(Routes(8, self.trace))
        self.main.cuda_stream = 44
        self.current_device = 1
        with self.assertRaisesRegex(RuntimeError, "bound CUDA device"):
            self.group.plan_schedule(Routes(8, self.trace))
        self.current_device = 0
        with self.assertRaisesRegex(RuntimeError, "pair exactly once"):
            self.group.finish()
        self.assertEqual(self.trace, [])
        part = self.group.plan_schedule(Routes(8, self.trace))
        with self.assertRaisesRegex(RuntimeError, "paired finish"):
            self.group.plan_schedule(Routes(8, self.trace))
        self.group.plan_finish(part, defer_wait=True)
        with self.assertRaisesRegex(RuntimeError, "pair exactly once"):
            self.group.finish()

    def test_driver_failure_does_not_launch_halo(self):
        self.group._driver.cuStreamWaitEvent = lambda *args: (17,)
        with self.assertRaisesRegex(RuntimeError, "cuStreamWaitEvent"):
            self.group.plan_schedule(Routes(8, self.trace))
        self.assertFalse(any(row[0] == "HALO" for row in self.trace))
        self.assertIsNone(self.group._plan_part)


if __name__ == "__main__":
    unittest.main(verbosity=2)
