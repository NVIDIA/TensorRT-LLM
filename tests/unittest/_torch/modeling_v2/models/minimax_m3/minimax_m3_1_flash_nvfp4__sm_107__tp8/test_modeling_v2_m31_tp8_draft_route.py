# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The TP8 DSpark worker keeps its draft route (the Markov chain and draft probabilities through
``_refine_block_logits``): it never takes an engine's ancestral block draft, whatever the engine offers.
Source-level (stdlib ast): no CUDA."""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[7]
_TP8 = (
    _ROOT
    / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8"
)


def test_the_tp8_worker_never_samples_the_block_ancestrally() -> None:
    tree = ast.parse((_TP8 / "dspark.py").read_text())
    (cls,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "M3DSparkWorker"]
    (fn,) = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_samples_block_ancestrally"
    ]
    (ret,) = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    assert isinstance(ret.value, ast.Constant) and ret.value.value is False
