# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dep2 captures prefill CUDA graphs only where the engine runs them: its breakable pass graphs with
`prefill_cuda_graph_backend: breakable` (on the context role with a bounded replay), nothing otherwise. Before, it
captured its own padded static-body graphs whenever the bounded replay was on, also with `cuda_graph_config: null` and
the default (disabled) prefill graph backend."""

import importlib

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="imports the target (CUDA build)"
)
mod = importlib.import_module(
    "tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__dep2.modeling"
)


def test_no_static_prefill_body():
    core = mod.ModelingV2Core
    for name in ("_forward_static", "_precapture_static", "_static_body", "_graph_rows"):
        assert not hasattr(core, name), name
