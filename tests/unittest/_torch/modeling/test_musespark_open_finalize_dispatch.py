# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which finalize the MoE forward takes, and that the TP reduce runs exactly once.

The experts are a stub that calls the registered finalize the way the MoE
scheduler does, so these drive the real `forward` and `finalize` on the host.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.models.modeling_musespark_open import (
    MuseSparkOpenMoE,
    post_expert_norm_combine,
)

pytestmark = pytest.mark.cpu_only

WIDTH = 16


def _moe(tokens, workspace, *, moe_max_num_tokens=8192):
    module = MuseSparkOpenMoE.__new__(MuseSparkOpenMoE)
    torch.nn.Module.__init__(module)
    module.top_k = 8
    module.expert_hidden_size = WIDTH
    module.post_expert_norm_weight = torch.zeros(WIDTH, dtype=torch.bfloat16)
    module.post_expert_norm_eps = 1e-6
    module.mapping = SimpleNamespace(enable_attention_dp=False, has_tp=lambda: True, moe_tp_size=1)
    module.nvtx = lambda name: nullcontext()
    module.finalize_allreduce_workspace = workspace
    module._reduce_in_finalize = False
    module.all_reduce = Mock(side_effect=lambda partial: partial + 100)
    module.unpooler = lambda combined: combined
    ready = object()
    module.prepare_moe_inputs = lambda hidden: (torch.zeros(tokens, 256), "expert input", ready)

    # What the scheduler hands the registered finalize: TRTLLM-Gen dispatches
    # BF16 routing weights.
    gemm2 = torch.randn(tokens * 8, WIDTH, dtype=torch.bfloat16)
    index = torch.arange(tokens * 8, dtype=torch.int32)
    routing = torch.full((tokens, 8), 1 / 8, dtype=torch.bfloat16)
    seen = {}

    def experts(expert_input, router_logits, **kwargs):
        seen.update(kwargs)
        seen["reduce_in_finalize"] = module._reduce_in_finalize
        return module.finalize(
            gemm2_output=gemm2,
            expanded_idx_to_permuted_idx=index,
            token_selected_slots=None,
            routing_weights=routing,
            num_tokens=tokens,
        )

    module.experts = Mock(side_effect=experts, comm=None, moe_max_num_tokens=moe_max_num_tokens)
    unfused = post_expert_norm_combine(
        gemm2,
        index,
        routing,
        module.post_expert_norm_weight,
        module.post_expert_norm_eps,
        num_tokens=tokens,
        top_k=8,
        hidden_size=WIDTH,
    )
    return module, seen, ready, unfused


def _workspace(tokens):
    return SimpleNamespace(
        finalize=Mock(return_value=torch.full((tokens, WIDTH), 3.0, dtype=torch.bfloat16))
    )


@pytest.mark.parametrize(
    "tokens,with_workspace,compiling,moe_max_num_tokens,expect_fused",
    [
        (1, True, False, 8192, True),
        (128, True, False, 8192, True),
        (129, True, False, 8192, False),
        (8, False, False, 8192, False),
        (8, True, True, 8192, False),
        # A batch the MoE would split into chunks.
        (8, True, False, 4, False),
    ],
)
def test_the_forward_picks_one_finalize_and_reduces_once(
    tokens, with_workspace, compiling, moe_max_num_tokens, expect_fused
):
    workspace = _workspace(tokens) if with_workspace else None
    module, seen, ready, unfused = _moe(tokens, workspace, moe_max_num_tokens=moe_max_num_tokens)
    hidden = torch.ones(tokens, 32, dtype=torch.bfloat16)

    with patch("torch.compiler.is_compiling", return_value=compiling):
        result = module.forward(hidden, None)

    assert seen["input_ready_event"] is ready
    assert seen["do_finalize"] is False
    assert seen["reduce_in_finalize"] is expect_fused
    assert module._reduce_in_finalize is False
    if expect_fused:
        workspace.finalize.assert_called_once()
        routing_weights = workspace.finalize.call_args.args[2]
        assert routing_weights.dtype == torch.float32
        module.all_reduce.assert_not_called()
        torch.testing.assert_close(result, workspace.finalize.return_value, rtol=0, atol=0)
    else:
        if workspace is not None:
            workspace.finalize.assert_not_called()
        module.all_reduce.assert_called_once()
        torch.testing.assert_close(result, unfused.to(torch.bfloat16) + 100, rtol=0, atol=0)


def test_a_failed_expert_call_does_not_leave_the_fused_choice_behind():
    module, _, _, _ = _moe(4, _workspace(4))
    module.experts.side_effect = RuntimeError("injected")
    with pytest.raises(RuntimeError, match="injected"):
        module.forward(torch.ones(4, 32, dtype=torch.bfloat16), None)
    assert module._reduce_in_finalize is False
