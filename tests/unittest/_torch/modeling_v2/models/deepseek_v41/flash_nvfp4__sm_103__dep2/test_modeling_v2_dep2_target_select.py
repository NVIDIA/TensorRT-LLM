# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which target a dep2 step runs: uniform generation-only steps take the paged decode path (at any width when the step
is on the engine's CUDA-graph metadata), everything else the general (prefill) path."""

import importlib
from types import SimpleNamespace as NS

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="imports the target (CUDA build)"
)
mod = importlib.import_module(
    "tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__dep2.modeling"
)
CORE = NS(_decode="decode", _prefill="prefill")


def test_uniform_generation_steps_are_paged():
    step = NS(n=8, uniform=True)
    assert mod.ModelingV2Core._select_target(CORE, step, NS(num_contexts=0)) == "decode"
    assert mod.ModelingV2Core._select_target(CORE, step, NS(num_contexts=1)) == "prefill"


def test_wide_cuda_graph_steps_are_paged():
    """A generation step on the engine's CUDA-graph metadata (its warm-up runs and its capture) takes the paged path at
    any width: the general path asserts it is never captured."""
    wide = NS(n=mod._PAGED_MAX_ROWS + 4, uniform=True)
    assert (
        mod.ModelingV2Core._select_target(CORE, wide, NS(num_contexts=0, is_cuda_graph=True))
        == "decode"
    )
    assert mod.ModelingV2Core._select_target(CORE, wide, NS(num_contexts=0)) == "prefill"
