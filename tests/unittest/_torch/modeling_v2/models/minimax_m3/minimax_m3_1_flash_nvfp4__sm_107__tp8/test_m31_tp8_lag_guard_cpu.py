# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase-lag guard of the TP8 MoE engine: stdlib tests; no torch, CUDA import, allocation, or compilation.

Host constructor and kernel constant-prefix checks execute production AST with
only device/dependency doubles. Phase-plan checks run the actual planner on a
sparse integer tensor double. Runtime-import constructor UTs are separate.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import pytest

pytestmark = pytest.mark.cpu_only

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "tensorrt_llm").is_dir())
FAMILY = ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3"
TARGET = FAMILY / "minimax_m3_1_flash_nvfp4__sm_107__tp8"
PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8"


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


POLICY = _module("_lag_guard_policy", FAMILY / "m31_moe_lag.py")
SWITCHES = _module(PKG + ".m31_switches", TARGET / "m31_switches.py")
HOST = TARGET / "kernels/m3_moe/op.py"
KERNEL = TARGET / "kernels/m3_moe/m3_moe_kernel.py"


def _exec(nodes, scope, path):
    module = ast.Module([ast.ImportFrom("__future__", [ast.alias("annotations")], 0), *nodes], [])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope


def _host():
    tree = ast.parse(HOST.read_text())
    config = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "moe_lag_config"
    )
    state = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "M3MoeState")
    constructor = next(
        n for n in state.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    cls = ast.ClassDef("M3MoeState", [], [], [constructor], [])
    probe = mock.Mock(side_effect=RuntimeError("GPU_FORBIDDEN"))
    scope = {
        "__package__": PKG + ".kernels.m3_moe",
        "os": os,
        "phase_lag_config": POLICY.phase_lag_config,
        "M3_MOE_LAG": 2,
        "INTER": 384,
        "torch": types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_current_stream_capturing=probe)
        ),
    }
    return _exec([config, cls], scope, HOST), probe


def _kernel(config):
    tree = ast.parse(KERNEL.read_text())
    stop = next(
        i
        for i, n in enumerate(tree.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "STAGES" for t in n.targets)
    )
    nodes = [n for n in tree.body[: stop + 1] if not isinstance(n, (ast.Import, ast.ImportFrom))]
    return _exec(
        nodes,
        {
            "M3_CONFIG": config,
            "completion_ring_slots": POLICY.completion_ring_slots,
            "validate_kernel_lag": POLICY.validate_kernel_lag,
        },
        KERNEL,
    )


class SparsePlan:
    def __init__(self, *shape):
        self.shape = shape
        self.data = {}

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, key):
        return self.data.get(key, 0)


def _planner():
    tree = ast.parse(HOST.read_text())
    names = {
        "HIDDEN",
        "INTER",
        "TOP_K",
        "U1",
        "_UNIT_BYTES",
        "_ROW2_BYTES",
        "_S1",
        "_ITEM_BYTES",
        "FINE_PLAN",
        "HALFPLAN",
    }
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)
        or isinstance(n, ast.FunctionDef)
        and n.name in ("make_phase_plan", "_fine_ranges")
    ]
    return _exec(
        nodes,
        {
            "torch": types.SimpleNamespace(zeros=lambda *s, **kw: SparsePlan(*s), int32="int32"),
            "m31_disabled": lambda _: False,
        },
        HOST,
    )


def _wait_observed(depth, arrivals, group):
    completions = sum(g % depth == group % depth for g in arrivals)
    return completions % 2 != (group // depth) % 2


class TestTP8LagGuard(unittest.TestCase):
    def setUp(self):
        logger = types.ModuleType("tensorrt_llm.logger")
        logger.logger = mock.Mock()
        self.modules = mock.patch.dict(
            sys.modules, {PKG + ".m31_switches": SWITCHES, "tensorrt_llm.logger": logger}
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_default_and_nondefault_lag_keys(self):
        ns, _ = _host()
        for tokens in range(1, 9):
            ring = {} if tokens <= 4 else {"fc1d": 8}
            self.assertEqual(ns["moe_lag_config"](2, tokens), ring)
            self.assertEqual(ns["moe_lag_config"](3, tokens), {**ring, "lag": 3})

    def test_actual_constructor_rejects_lag_5_6_before_device_work(self):
        for lag in (4, 5, 6, 8):
            for tokens in range(1, 9):
                with self.subTest(lag=lag, tokens=tokens):
                    ns, probe = _host()
                    ns["M3_MOE_LAG"] = lag
                    with self.assertRaisesRegex(ValueError, "fc1_done"):
                        ns["M3MoeState"](None, tokens)
                    probe.assert_not_called()

    def test_the_host_admits_no_unsafe_lag(self):
        """Only the policy's explicit allow_unsafe keyword admits a lag past the ring; the host never passes it."""
        ns, probe = _host()
        for lag in (4, 5, 6):
            with self.subTest(lag=lag), self.assertRaisesRegex(ValueError, "fc1_done"):
                ns["moe_lag_config"](lag, 8)
        probe.assert_not_called()
        self.assertEqual(
            POLICY.phase_lag_config(6, 8, allow_unsafe=True), {"lag": 6, "fc1d": 8, "lag_unsafe": 1}
        )

    def test_invalid_lag_types(self):
        for lag in (-1, True, 2.0, "6"):
            with self.subTest(lag=lag), self.assertRaises(ValueError):
                POLICY.phase_lag_config(lag, 8)

    def test_direct_kernel_config_cannot_restore_unsafe_ring(self):
        for lag in (2, 5, 6):
            with self.subTest(lag=lag), self.assertRaisesRegex(ValueError, "fc1_done"):
                _kernel({"m_max": 8, "i_tp": 384, "lag": lag, "fc1d": 4})
        for lag in (5, 6):
            with self.assertRaisesRegex(ValueError, "fc1_done"):
                _kernel({"m_max": 8, "i_tp": 384, "lag": lag})
            ns = _kernel({"m_max": 8, "i_tp": 384, "lag": lag, "lag_unsafe": 1})
            self.assertEqual(ns["LAG"], lag)
            self.assertEqual(ns["FC1_DONE_SLOTS"], 8)

    def test_slot_allocation_indices_and_parities_share_geometry(self):
        for tokens in range(1, 9):
            ns = _kernel({"m_max": tokens, "i_tp": 384})
            slots = 4 if tokens <= 4 else 8
            self.assertEqual(
                (ns["FC1_DONE_SLOTS"], ns["FC1_DONE_SHIFT"]), (slots, slots.bit_length() - 1)
            )
            self.assertEqual(len({g & (slots - 1) for g in range(tokens)}), tokens)
        source = KERNEL.read_text()
        self.assertIn("cutlass.Int64, FC1_DONE_SLOTS", source)
        self.assertIn("if tidx < FC1_DONE_SLOTS:", source)
        self.assertEqual(source.count("fc1_done.subview(p & cutlass.Int32(FC1_DONE_SLOTS - 1))"), 2)
        self.assertEqual(source.count("fc1_done.subview(g & cutlass.Int32(FC1_DONE_SLOTS - 1))"), 1)
        self.assertNotIn("fc1_done.subview(p & cutlass.Int32(3))", source)
        self.assertIn("(FC1_DONE_SLOTS - 4) * 8", source)

    def test_production_phase_plan_exposes_idle_group_witness(self):
        ns = _planner()
        # 188 is a valid routed-CTA configuration (212 SMs minus 24 shared CTAs).
        plan, _ = ns["make_phase_plan"](188, 8, 2, 1.0, True)
        witness = None
        for experts in range(ns["TOP_K"], ns["TOP_K"] * 8 + 1):
            groups = (experts + 3) // 4
            for cta in range(plan.shape[1]):
                idle = 0
                for group in range(groups):
                    idle = idle + 1 if plan[experts, cta, group] == 0 else 0
                    if idle >= 5:
                        witness = (experts, cta, group - idle + 1, group)
                        break
                if witness:
                    break
            if witness:
                break
        self.assertIsNotNone(witness)
        _, _, first, last = witness
        arrivals = list(range(first, last + 1))
        self.assertFalse(_wait_observed(4, arrivals, first))
        self.assertTrue(_wait_observed(POLICY.completion_ring_slots(8), arrivals, first))

    def test_lag_5_6_phase_loss_witness_has_no_alias_with_eight_slots(self):
        for lag in (5, 6):
            # Group 0 retired; the epilogue completes 1..lag before the wait on 1.
            arrivals = list(range(1, lag + 1))
            self.assertFalse(_wait_observed(4, arrivals, 1))
            self.assertTrue(_wait_observed(8, arrivals, 1))

    def test_routed_212_plan_has_twenty_zero_fc2_ctas(self):
        ns = _planner()
        for ctas, empty in ((188, 0), (192, 0), (212, 20)):
            plan, _ = ns["make_phase_plan"](ctas, 3, 2, 1.0, True)
            # The structured three-row native fixture selects ten distinct experts.
            blocks = [plan[10, cta, 3] >> 16 for cta in range(ctas)]
            self.assertEqual(sum(blocks), 192)
            self.assertEqual(blocks.count(0), empty)
        self.assertEqual(len({g & 3 for g in range(3)}), 3)

    def test_zero_fc2_epilogue_can_lap_two_slot_free_barrier(self):
        # A zero-block epilogue skips every acc2_full wait. If it retires all
        # three groups before the loader's first reuse, slot0 turns twice.
        def phase(released):
            return sum(group % 2 == 0 for group in released) % 2

        loader_expected = ((2 >> 1) - 1) & 1
        self.assertNotEqual(phase([0]), loader_expected)
        self.assertEqual(phase([0, 1, 2]), loader_expected)

    def test_actual_fc2_loader_and_mma_gates_exclude_zero_blocks(self):
        tree = ast.parse(KERNEL.read_text())
        gates = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.If) or "nb" not in ast.unparse(node.test):
                continue
            body = ast.unparse(ast.Module(node.body, []))
            if "b2_free.subview" in body or "b2_ready.subview" in body:
                gates.append(node.test)
        self.assertEqual(len(gates), 3)  # B2IMG loader, record loader, MMA FC2.
        cutlass = types.SimpleNamespace(Int32=int, Boolean=bool)
        for b2img in (False, True):
            for warp in range(12):
                for blocks in (0, 1, 2):
                    actual = [
                        bool(
                            eval(
                                compile(ast.Expression(test), str(KERNEL), "eval"),
                                {
                                    "cutlass": cutlass,
                                    "nb": blocks,
                                    "warp": warp,
                                    "X_WARP": 5,
                                    "B2IMG": b2img,
                                    "p": 2,
                                    "LAG": 2,
                                },
                            )
                        )
                        and (
                            b2img
                            if "X_WARP" in ast.unparse(test) and "B2IMG" not in ast.unparse(test)
                            else True
                        )
                        for test in gates
                    ]
                    if blocks == 0:
                        self.assertFalse(any(actual))
                    else:
                        self.assertEqual(
                            sum(actual), 1 + int(warp == 5 or (warp >= 8 and not b2img))
                        )

    def test_relative_kernel_import_resolves_to_family_policy(self):
        package = PKG + ".kernels.m3_moe"
        name = importlib.util.resolve_name("....m31_moe_lag", package)
        self.assertEqual(
            ROOT.joinpath(*name.split(".")).with_suffix(".py"), FAMILY / "m31_moe_lag.py"
        )


if __name__ == "__main__":
    unittest.main()
