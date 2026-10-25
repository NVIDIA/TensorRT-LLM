# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the production fused-dispatch predicates without CUDA imports."""

import ast
import types
import unittest
from pathlib import Path

ROOT = (
    Path(__file__).resolve().parents[4]
    / "tensorrt_llm"
    / "_torch"
    / "_experimental"
    / "modeling_v2"
)


def dispatch(tp, rows, requests, capacity=64, ndim=1, embeddings=False):
    path = ROOT / "models" / "minimax_m3" / f"minimax_m31_nvfp4__sm_103__tp{tp}" / "modeling.py"
    tree = ast.parse(path.read_text())
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "is_decode_step"
    )
    limit = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "FUSED_DECODE_MAX_TOKENS"
            for target in node.targets
        )
    )
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DecodeTarget"
    )
    forward = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    predicate = next(
        node.value
        for node in ast.walk(forward)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "kernels" for target in node.targets)
    )
    env = {
        "DECODE_MAX_TOKENS": capacity,
        "input_ids": types.SimpleNamespace(ndim=ndim, numel=lambda: rows),
        "inputs_embeds": object() if embeddings else None,
        "attn_metadata": types.SimpleNamespace(num_seqs=requests),
    }
    exec(compile(ast.Module(body=[limit, helper], type_ignores=[]), str(path), "exec"), env)
    return eval(compile(ast.Expression(predicate), str(path), "eval"), env)


class TestM31DecodeScope(unittest.TestCase):
    def test_tp8_single_request_boundary(self):
        for capacity in (48, 64):
            for rows, expected in (
                (1, True),
                (7, True),
                (8, True),
                (9, False),
                (32, False),
                (64, False),
            ):
                with self.subTest(capacity=capacity, rows=rows):
                    self.assertEqual(dispatch(8, rows, 1, capacity), expected)

    def test_tp8_multiple_requests_decline_even_when_rows_fit(self):
        self.assertFalse(dispatch(8, 8, 2))  # K3/c2 must not enter the c1 fused path.
        self.assertFalse(dispatch(8, 8, 8))  # Non-speculative c8 also has eight rows.

    def test_tp4_boundary(self):
        for capacity in (48, 64):
            for rows, requests, expected in (
                (8, 1, True),
                (16, 2, True),
                (32, 4, True),
                (33, 5, False),
                (64, 8, False),
            ):
                with self.subTest(capacity=capacity, rows=rows):
                    self.assertEqual(dispatch(4, rows, requests, capacity), expected)

    def test_embeddings_and_nonflat_ids_use_generic_path(self):
        for tp in (4, 8):
            with self.subTest(tp=tp):
                self.assertFalse(dispatch(tp, 8, 1, ndim=2))
                self.assertFalse(dispatch(tp, 8, 1, embeddings=True))


if __name__ == "__main__":
    unittest.main()
