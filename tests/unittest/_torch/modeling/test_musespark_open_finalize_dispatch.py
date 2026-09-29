# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep finalize fusion confined to the supported model execution paths."""

from contextlib import nullcontext
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenMoE


@pytest.mark.parametrize(
    "tokens,workspace_available,compiling,expect_fused",
    [
        (1, True, False, True),
        (128, True, False, True),
        (129, True, False, False),
        (8, False, False, False),
        (8, True, True, False),
    ],
)
def test_finalize_dispatch_preserves_fallback(
    tokens: int, workspace_available: bool, compiling: bool, expect_fused: bool
) -> None:
    """Exercise the real MoE forward selection and prevent double reduction."""
    hidden = torch.ones(tokens, 4, dtype=torch.bfloat16)
    indices = torch.zeros(tokens, 8, dtype=torch.int32)
    weights = torch.full((tokens, 8), 1 / 8)
    routed = torch.full_like(hidden, 2)
    finalized = torch.full_like(hidden, 3)
    reduced = torch.full_like(hidden, 4)
    routing = SimpleNamespace(
        apply=Mock(return_value=(indices, weights)), clear_routing_memo=Mock()
    )
    experts = Mock(return_value=(routed, weights, indices))
    experts.routing_method = routing
    experts.comm = None
    workspace = SimpleNamespace(finalize=Mock(return_value=finalized))
    module = SimpleNamespace(
        experts=experts,
        prepare_moe_inputs=Mock(return_value=(weights, hidden)),
        nvtx=lambda _: nullcontext(),
        _finalize_ar_workspace=workspace if workspace_available else None,
        post_expert_norm_weight=torch.zeros(4),
        post_expert_norm_eps=1e-8,
        finalize=Mock(return_value=finalized),
        all_reduce=Mock(return_value=reduced),
        unpooler=Mock(side_effect=lambda value: value),
        _unpool_gather_workspace=None,
        mapping=SimpleNamespace(enable_attention_dp=False, has_tp=lambda: True, moe_tp_size=1),
    )
    module._replicated_projection = MethodType(MuseSparkOpenMoE._replicated_projection, module)
    with patch("torch.compiler.is_compiling", return_value=compiling):
        result = MuseSparkOpenMoE.forward(module, hidden, None)
    if expect_fused:
        workspace.finalize.assert_called_once()
        module.finalize.assert_not_called()
        module.all_reduce.assert_not_called()
        assert result is finalized
    else:
        workspace.finalize.assert_not_called()
        module.finalize.assert_called_once()
        module.all_reduce.assert_called_once()
        assert result is reduced
    routing.clear_routing_memo.assert_called_once()
