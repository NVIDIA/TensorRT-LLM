# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""No deepseek_v41 target lets the engine capture a context step as a whole-body prefill CUDA graph.

The engine's prefill graph capture (`prefill_cuda_graph_backend`) reads `supports_prefill_cuda_graph` of the model
body and captures context warm-up steps through its forward when it is true. Every target's context path
(`PrefillTarget`) runs eagerly and asserts it is not captured, so the flag must be false in every layout, also where
stock's rule (decline under attention DP with a decoder replay plan) would allow a capture: attention DP off (tep8), or
bounded replay off. Before, those layouts reported true and a capture would fail engine start.
"""

import importlib
import importlib.util

import pytest

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("target", ["dep2", "dep", "tep8_moe_tp4_ep2"])
def test_no_whole_body_prefill_graph(target):
    module = importlib.import_module(
        f"tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__{target}.modeling"
    )
    assert module.ModelingV2Core.supports_prefill_cuda_graph is False
    # set on the class, never re-derived per instance from the layout
    assert "self.supports_prefill_cuda_graph" not in importlib.util.find_spec(
        module.__name__
    ).loader.get_source(module.__name__)
