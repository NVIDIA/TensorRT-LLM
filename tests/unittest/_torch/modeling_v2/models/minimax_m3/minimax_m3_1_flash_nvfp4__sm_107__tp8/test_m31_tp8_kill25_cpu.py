# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSpark drafter glue the TP8 target calls (``ops/dspark``) with its native ops killed
(TRTLLM_M31_DISABLE_DSPARK_GLUE_CPP=1): stdlib checks of the actual module bodies.

Imports are replaced with dependency doubles. These tests establish dispatch
and no-write fallback behavior, not Torch imports, GPU numerics or timings.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

pytestmark = pytest.mark.cpu_only

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "tensorrt_llm").is_dir())
DSPARK = ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3/ops/dspark"


def load(name, killed):
    path = DSPARK / (name + ".py")
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            node.body = [n for n in node.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    tree.body = [ast.ImportFrom("__future__", [ast.alias("annotations")], 0)] + [
        n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))
    ]
    query = Mock(return_value=Mock(name="native_op"))

    class Ops:
        def __getattr__(self, key):
            return query(key)

    ops = Ops()
    scope = {
        "torch": SimpleNamespace(ops=SimpleNamespace(trtllm=ops)),
        "family_ops": lambda: ops,
        "_GLUE_CPP_OFF": killed,
        "_glue_log": Mock(return_value=None),
    }
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), scope)
    return scope, query


class TestGlueKill(unittest.TestCase):
    def test_input_ops_killed_before_lookup(self):
        ns, query = load("dspark_drafter_inputs", True)
        for key in ("_OP", "_OP_REFRESH"):
            self.assertIsNone(ns["_op"](ns[key]))
        query.assert_not_called()
        self.assertEqual(ns["_glue_log"].call_count, 2)

    def test_input_ops_default_and_missing_registration_preserved(self):
        ns, query = load("dspark_drafter_inputs", False)
        for key in ("_OP", "_OP_REFRESH"):
            self.assertIs(ns["_op"](ns[key]), query.return_value)
        query.return_value = None
        self.assertIsNone(ns["_op"](ns["_OP"]))
        ns["_glue_log"].assert_not_called()

    def test_both_input_entry_points_fall_back_without_output_writes(self):
        ns, query = load("dspark_drafter_inputs", True)
        ns["_inputs_supported"] = Mock(side_effect=AssertionError("must short circuit"))
        ns["_outputs"] = Mock(side_effect=AssertionError("no output allocation"))
        ns["refresh_supported"] = Mock(side_effect=AssertionError("must short circuit"))
        counts = SimpleNamespace(shape=(1,))
        args = (None, counts, None, None, 0, None, None, 8, 8)
        self.assertIsNone(ns["drafter_inputs"](*args))
        self.assertIsNone(ns["drafter_inputs_and_refresh"](*args, None, 0, 1, 128, None, None))
        query.assert_not_called()
        ns["_outputs"].assert_not_called()

    def test_block_refresh_killed_before_lookup_or_tensor_access(self):
        ns, query = load("dspark_block_tables", True)
        self.assertFalse(ns["refresh_block_tables"](None, 0, 1, 128, None, None))
        query.assert_not_called()
        ns["_glue_log"].assert_called_once()

    def test_block_refresh_default_native_call_preserved(self):
        ns, query = load("dspark_block_tables", False)
        ns["supported"] = Mock(return_value=True)
        offsets = MagicMock(is_cuda=True, shape=(1, 1, 1, 32))
        tables = SimpleNamespace(shape=(1, 32))
        counts = Mock(shape=(1,))
        counts.is_contiguous.return_value = True
        self.assertTrue(ns["refresh_block_tables"](offsets, 0, 1, 128, tables, counts))
        query.return_value.assert_called_once_with(
            offsets.__getitem__.return_value, 128, tables, counts
        )
        ns["_glue_log"].assert_not_called()

    def test_block_refresh_missing_native_op_keeps_fallback(self):
        ns, query = load("dspark_block_tables", False)
        query.return_value = None
        self.assertFalse(ns["refresh_block_tables"](None, 0, 1, 128, None, None))


if __name__ == "__main__":
    unittest.main()
