# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which target a step runs: every generation-only step takes the paged decode path, as the engine's backend serves
it, whatever its requests' row counts (an eager step of mixed lengths, e.g. a draft-less first step after admission,
included); steps with context rows take the general (prefill) path."""

import importlib
from types import SimpleNamespace as NS

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="imports the target (CUDA build)"
)
mod = importlib.import_module(
    "tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__tep8_moe_tp4_ep2.modeling"
)
CORE = NS(_decode="decode", _prefill="prefill")


@pytest.mark.parametrize("uniform", [True, False])
def test_generation_only_steps_are_paged(uniform):
    step = NS(n=7 if not uniform else 8, uniform=uniform)
    assert mod.ModelingV2Core._select_target(CORE, step, NS(num_contexts=0)) == "decode"


def test_steps_with_context_rows_are_general():
    assert (
        mod.ModelingV2Core._select_target(CORE, NS(n=8, uniform=True), NS(num_contexts=1))
        == "prefill"
    )


def test_wide_steps_are_general():
    assert (
        mod.ModelingV2Core._select_target(
            CORE, NS(n=mod._PAGED_MAX_ROWS + 1, uniform=True), NS(num_contexts=0)
        )
        == "prefill"
    )


def test_wide_cuda_graph_steps_are_paged():
    """A generation step on the engine's CUDA-graph metadata (its warm-up runs and its capture) takes the paged path at
    any width: the general path asserts it is never captured, so a graph batch of more than `_PAGED_MAX_ROWS` rows
    (e.g. `cuda_graph_config.max_batch_size` 1280 with 3 drafts: 5120 rows) failed engine start there."""
    wide = NS(n=5120, uniform=True)
    assert (
        mod.ModelingV2Core._select_target(CORE, wide, NS(num_contexts=0, is_cuda_graph=True))
        == "decode"
    )
    assert (
        mod.ModelingV2Core._select_target(CORE, wide, NS(num_contexts=0, is_cuda_graph=False))
        == "prefill"
    )
