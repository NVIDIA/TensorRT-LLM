# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU key contract for the drafter-only M7 ring choice, without importing CUDA."""

import ast
import inspect
import unittest
from pathlib import Path
from typing import Optional

import pytest

pytestmark = pytest.mark.cpu_only

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "tensorrt_llm").is_dir())
TARGET = (
    ROOT
    / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8"
)
# The A key of a real checkpoint (rank 0, layer 0).
QKV_A = (
    1280,
    6144,
    8,
    6,
    True,
    False,
    False,
    1,
    True,
    True,
    True,
    0,
    1,
    8,
    8,
    True,
    False,
    False,
    False,
)
GU_A = (
    3072,
    6144,
    4,
    11,
    True,
    False,
    False,
    1,
    True,
    True,
    True,
    0,
    1,
    8,
    8,
    True,
    False,
    False,
    True,
)


def selector():
    """Execute the actual pure selector body, avoiding the module's GPU imports."""
    tree = ast.parse((TARGET / "drafter_mxfp8.py").read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_draft_qkv_ring2_cfg"
    )
    namespace = {"Optional": Optional}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "<actual ring selector>", "exec"),
        namespace,
    )
    return namespace["_draft_qkv_ring2_cfg"]


class TestDraftQkvRing2Key(unittest.TestCase):
    def test_install_resolver_calls_bind_actual_api(self):
        tree = ast.parse((TARGET / "kernels/mx_gemv/op.py").read_text())
        resolver = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_resolve"
        )
        resolver.body = [ast.Pass()]
        resolver.returns = None
        for arg in resolver.args.args:
            arg.annotation = None
        module = ast.fix_missing_locations(ast.Module(body=[resolver], type_ignores=[]))
        namespace = {}
        exec(compile(module, "<actual resolver signature>", "exec"), namespace)
        signature = inspect.signature(namespace["_resolve"])
        tree = ast.parse((TARGET / "drafter_mxfp8.py").read_text())
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "_gemv"
            and n.func.attr == "_resolve"
        ]
        self.assertEqual(len(calls), 2)
        values = {"n": 1280, "k": 6144, "qkv_cfg": (8, 2, False, 1, True, True)}
        for call in calls:
            args = [
                eval(compile(ast.Expression(a), "<call argument>", "eval"), values)
                for a in call.args
            ]
            kwargs = {
                kw.arg: eval(compile(ast.Expression(kw.value), "<call keyword>", "eval"), values)
                for kw in call.keywords
            }
            bound = signature.bind(*args, **kwargs)
            self.assertIsNone(bound.arguments["ar"])
            self.assertIsNone(bound.arguments["use_pdl"])
            self.assertTrue(bound.arguments["trigger_early"])

    def test_default_on_retains_every_non_ring_field(self):
        cfg = selector()(7, True, QKV_A, False)
        self.assertEqual(cfg, (8, 2, False, 1, True, True))
        rebuilt = (
            QKV_A[0],
            QKV_A[1],
            cfg[0],
            cfg[1],
            QKV_A[4],
            cfg[2],
            QKV_A[6],
            cfg[3],
            cfg[4],
            cfg[5],
            *QKV_A[10:],
        )
        self.assertEqual([i for i, (a, b) in enumerate(zip(QKV_A, rebuilt)) if a != b], [3])

    def test_kill_and_non_drafter_restore_default(self):
        choose = selector()
        self.assertIsNone(choose(7, True, QKV_A, True))
        self.assertIsNone(choose(7, False, QKV_A, False))

    def test_gu_and_other_rows_keep_default(self):
        choose = selector()
        self.assertIsNone(choose(7, True, GU_A, False))
        for m in range(1, 49):
            if m != 7:
                self.assertIsNone(choose(m, True, QKV_A, False))

    def test_changed_precision_tactic_or_epilogue_declines(self):
        choose = selector()
        for i, value in enumerate(QKV_A):
            changed = list(QKV_A)
            changed[i] = not value if isinstance(value, bool) else value + 1
            self.assertIsNone(choose(7, True, tuple(changed), False), f"field {i}")

    def test_precompile_resolves_the_explicit_config(self):
        tree = ast.parse((TARGET / "kernels/mx_gemv/op.py").read_text())
        function = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "precompile"
        )
        call = next(
            n
            for n in ast.walk(function)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_resolve"
        )
        self.assertIsInstance(call.args[5], ast.Name)
        self.assertEqual(call.args[5].id, "cfg")
        index = [a.arg for a in function.args.args].index("cfg")
        default_offset = len(function.args.args) - len(function.args.defaults)
        self.assertIsNone(function.args.defaults[index - default_offset].value)


if __name__ == "__main__":
    unittest.main()
