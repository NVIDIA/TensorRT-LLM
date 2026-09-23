# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests executing production methods with observable CUDA stand-ins."""

from __future__ import annotations

import ast
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3] / "tensorrt_llm/_torch"


def methods(path, class_name, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name)
    selected = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *selected,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return {n.name: namespace[n.name] for n in selected}


class Tensor:
    def __init__(self, name, trace, shape=(8, 6)):
        self.name, self.trace, self.shape = name, trace, shape

    def __getitem__(self, index):
        return Tensor(self.name, self.trace, self.shape)

    def view(self, dtype):
        return self

    def copy_(self, other, **kwargs):
        self.trace.append("copy:" + self.name)

    def zero_(self):
        self.trace.append("zero:" + self.name)

    def fill_(self, value):
        self.trace.append("fill:" + self.name)


class RouteContract(unittest.TestCase):
    def setUp(self):
        self.trace = []
        stream = SimpleNamespace(wait_event=lambda e: self.trace.append("wait"))
        torch = SimpleNamespace(
            cuda=SimpleNamespace(
                current_stream=lambda _: stream, nvtx=SimpleNamespace(range=lambda _: nullcontext())
            )
        )
        ns = dict(torch=torch)
        funcs = methods(
            "moe/fused_moe/mega_moe/rebalance_slot_scheduler_v2.py",
            "RebalanceSlotSchedulerGroupV2",
            {"plan_finish", "wait_for_routes", "finish"},
            ns,
        )
        cls = type("Group", (), funcs)
        self.group = cls()
        self.group._check_owner = lambda: None
        self.group.device = 0
        self.group._plan_ready = object()
        self.group._plan_ready_handle = 2
        self.group._execution_stream_handle = 44
        self.group._wait_event = lambda *_: self.trace.append("wait")
        self.group._generation = 1
        self.group._route_wait_generation = 0
        self.group.plan_calls = 0
        self.group._finished_generation = 0
        self.group.broadcaster = SimpleNamespace(
            release_generation_after=lambda *a: self.trace.append("release")
        )
        self.group._plan_part = (SimpleNamespace(physical_slot_ids=Tensor("routes", self.trace)), 4)

    def test_deferred_handoff_does_no_device_work_and_requires_wait(self):
        view, gen = self.group.plan_finish(self.group._plan_part, defer_wait=True)
        self.assertEqual((view.name, gen, self.trace), ("routes", 1, []))
        with self.assertRaises(RuntimeError):
            self.group.finish(object())
        self.group.wait_for_routes()
        self.group.wait_for_routes()
        self.assertEqual(self.trace, ["wait"])
        self.group.finish(object())
        self.assertEqual(self.trace, ["wait", "release"])
        self.assertEqual(self.group._finished_generation, 1)

    def test_default_handoff_retains_device_dependency(self):
        self.group.plan_finish(self.group._plan_part)
        self.assertEqual(self.trace, ["wait"])
        with self.assertRaises(RuntimeError):
            self.group.plan_finish(None)

    def test_new_generation_needs_new_wait(self):
        self.group.plan_finish(self.group._plan_part)
        self.group.finish(object())
        self.group._generation = 2
        self.group._plan_part = (SimpleNamespace(physical_slot_ids=Tensor("routes", self.trace)), 4)
        self.group.plan_finish(self.group._plan_part, defer_wait=True)
        self.group.wait_for_routes()
        self.group.finish(object())
        self.assertEqual(self.trace, ["wait", "release", "wait", "release"])


class StagingContract(unittest.TestCase):
    def run_stage(self, stage_routes):
        trace = []
        funcs = methods(
            "moe/fused_moe/mega_moe/mega_moe_cute_dsl.py",
            "TrtllmCutedslMegaMoeNvfp4Impl",
            {"_nvtxwrap__stage_inputs"},
            dict(torch=SimpleNamespace(uint8="uint8")),
        )
        owner = SimpleNamespace(
            _last_staged_T={8: 8}, _wait_rebalance_routes=lambda: trace.append("wait")
        )
        bufs = SimpleNamespace(
            **{
                key: Tensor(key, trace)
                for key in ["topk_idx_local", "activation", "activation_sf", "topk_weights"]
            }
        )
        funcs["_nvtxwrap__stage_inputs"](
            owner,
            bufs=bufs,
            x=Tensor("x", trace),
            x_sf=Tensor("sf", trace),
            topk_idx=Tensor("routes", trace),
            topk_weights=Tensor("weights", trace),
            num_tokens=4,
            top_k=6,
            stage_activation=True,
            stage_routes=stage_routes,
        )
        return trace, owner

    def test_fallback_wait_is_after_independent_staging(self):
        trace, owner = self.run_stage(True)
        self.assertLess(trace.index("copy:activation"), trace.index("wait"))
        self.assertLess(trace.index("copy:topk_weights"), trace.index("wait"))
        self.assertLess(trace.index("wait"), trace.index("copy:topk_idx_local"))
        self.assertEqual(owner._last_staged_T[8], 4)

    def test_borrowed_routes_skip_snapshot_and_preserve_staging_watermark(self):
        trace, owner = self.run_stage(False)
        self.assertNotIn("wait", trace)
        self.assertFalse(any("topk_idx_local" in item for item in trace))
        self.assertEqual(owner._last_staged_T[8], 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
