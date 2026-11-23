# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for MiniMax-H3 selective cuDNN -> trtllm-gen-fp16sm (schemeA_k8) routing."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

U4 = (
    Path(__file__).resolve().parents[6]
    / "tensorrt_llm/_torch/visual_gen/models/minimax_h3/minimax_h3_fp8__sm_103_107__u4"
)


class _Scalar:
    def __init__(self, v: float):
        self.v = float(v)

    def __mul__(self, other):
        return _Scalar(self.v * (other.v if isinstance(other, _Scalar) else other))

    def __rmul__(self, other):
        return _Scalar(other * self.v)

    def __truediv__(self, other):
        return _Scalar(self.v / (other.v if isinstance(other, _Scalar) else other))

    def __rtruediv__(self, other):
        return _Scalar(other / self.v)

    def clamp(self, min=None, max=None):
        x = self.v
        if min is not None:
            x = x if x >= min else min
        if max is not None:
            x = x if x <= max else max
        return _Scalar(x)

    def bfloat16(self):
        return self

    def float(self):
        return self


def _load_complementary_descales():
    tree = ast.parse((U4 / "qkv_exchange.py").read_text())
    keep = {"_LOG2E", "_FP16SM_K_SHIFT_SCALE", "complementary_fp16sm_descales"}
    body = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in keep)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in keep for t in node.targets)
        )
    ]
    ns = {
        "math": math,
        "torch": SimpleNamespace(Tensor=object, stack=lambda xs: [x.v for x in xs]),
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(U4 / "qkv_exchange.py"), "exec"), ns)
    return ns


class TestFp8HybridFmhaContracts(unittest.TestCase):
    def test_complementary_fp16sm_descales_satisfies_unit_log2e_fold(self):
        ns = _load_complementary_descales()
        fn = ns["complementary_fp16sm_descales"]
        sm_scale = 128.0**-0.5
        for s_k in (0.015, 0.032, 0.085, 0.25):
            dq, dk, dv = fn([_Scalar(0.035), _Scalar(s_k), _Scalar(0.042)])
            self.assertAlmostEqual(dk, 32.0 * s_k, places=7)
            self.assertAlmostEqual(dv, 0.042, places=7)
            self.assertAlmostEqual(dq * dk * sm_scale * math.log2(math.e), 1.0, places=6)

    def test_pipeline_defines_schemeA_k8_threshold_and_sm107_guard(self):
        src = (U4 / "pipeline.py").read_text()
        self.assertIn("_HYBRID_CUDNN_INITIAL_STEPS = 8", src)
        self.assertIn("torch.cuda.get_device_capability(device) != (10, 7)", src)
        self.assertIn('fp8_dense_fmha="trtllm-gen-fp16sm"', src)
        self.assertIn("fp8_skip_corr_threshold=0.0", src)


if __name__ == "__main__":
    unittest.main()
