# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stdlib checks of the TP8 target's default-on MoE and drafter items' gates on the actual sources (no torch):
MOE_XDEFER and MOE_B1SPLIT engage only at 8 tokens and each has its own kill; the kernel keeps both off unless the
engine config sets them; the drafter GEMV ring retune changes only the QKV / o ring depths of the his-layout drafter
configs and its kill restores the default rings. These establish source shape, not GPU numerics or timings."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "tensorrt_llm").is_dir())
TP8 = (
    ROOT
    / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8"
)
SERVED_RINGS = {(1280, 6144): (8, 6), (6144, 1024): (2, 12), (3072, 6144): (4, 12)}


def _src(rel: str) -> str:
    return (TP8 / rel).read_text()


def _gates(tree: ast.AST, key: str):
    """The If nodes whose body assigns self.config[key]."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        for st in node.body:
            if (
                isinstance(st, ast.Assign)
                and isinstance(st.targets[0], ast.Subscript)
                and isinstance(st.targets[0].slice, ast.Constant)
                and st.targets[0].slice.value == key
            ):
                out.append(node)
    return out


class Tp8ItemGates(unittest.TestCase):
    def test_op_gates_are_8_tokens_and_own_kill(self) -> None:
        tree = ast.parse(_src("kernels/m3_moe/op.py"))
        for key, item in (("xdefer", "MOE_XDEFER"), ("b1split", "MOE_B1SPLIT")):
            with self.subTest(key=key):
                gates = _gates(tree, key)
                self.assertEqual(len(gates), 1, key)
                test = ast.unparse(gates[0].test)
                self.assertEqual(test, f"num_tokens == 8 and (not m31_disabled('{item}'))")
                vals = {
                    st.targets[0].slice.value: ast.literal_eval(st.value)
                    for st in gates[0].body
                    if isinstance(st, ast.Assign) and isinstance(st.targets[0], ast.Subscript)
                }
                self.assertEqual(
                    vals, {"xdefer": 1} if key == "xdefer" else {"b1split": 1, "b1s_tok": 4}
                )

    def test_engagement_line_once_per_process(self) -> None:
        src = _src("kernels/m3_moe/op.py")
        self.assertIn("_B1FRONT_LOGGED: set = set()", src)
        self.assertIn('f"[M31_TP8_B1FRONT] XDEFER=', src)
        self.assertIn('f"[M31_TP8_GVC8] RING_RETUNE=', _src("drafter_mxfp8.py"))

    def test_kernel_defaults_off(self) -> None:
        src = _src("kernels/m3_moe/m3_moe_kernel.py")
        self.assertIn('B1SPLIT = bool(_cfg("b1split", 0)) and XSTAGE', src)
        self.assertIn('XDEFER = bool(_cfg("xdefer", 0)) and XSTAGE', src)
        self.assertIn(
            "assert not B1SPLIT or (M_MAX == 8 and THREADS == 384 and 0 < B1S_TOK < M_MAX)", src
        )

    def test_gvc8_rings(self) -> None:
        tree = ast.parse(_src("drafter_mxfp8.py"))
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "HIS_CFGS"
        )
        self.assertIsInstance(node.value, ast.IfExp)
        self.assertEqual(ast.unparse(node.value.test), "m31_disabled('DRAFT_GEMV_RING_RETUNE')")
        killed, ring_retune = ast.literal_eval(node.value.body), ast.literal_eval(node.value.orelse)
        self.assertEqual(killed, SERVED_RINGS)
        self.assertEqual(set(ring_retune), set(SERVED_RINGS))
        for shape, (bn, ring) in SERVED_RINGS.items():
            with self.subTest(shape=shape):
                self.assertEqual(ring_retune[shape][0], bn, "only the ring depth changes")
        self.assertEqual(
            {s: ring_retune[s][1] for s in ring_retune},
            {(1280, 6144): 2, (6144, 1024): 2, (3072, 6144): 12},
        )


if __name__ == "__main__":
    unittest.main()
