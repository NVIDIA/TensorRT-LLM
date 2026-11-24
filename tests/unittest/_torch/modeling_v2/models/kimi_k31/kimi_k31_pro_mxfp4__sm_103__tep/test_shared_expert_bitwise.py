# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tep target's shared expert partial is bitwise the unfused path's at the attention-TP decode rows.

On attention TP16 the tep target computes a MoE layer's shared expert with ``K31SituMLP.low_m_forward`` (gate_up on
the layout's low-M plan, the MLP's activation and down projection) and leaves its all-reduce to the fused tail. The
unfused path calls the same MLP whole. Before the reduction the two give the same bits at every row count the
MoE front serves (1..32), on a rank's shard of the 6144-wide shared expert.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.bf16_linear import (
    K31SituMLP,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.low_m_plans import (
    k31_low_m_plan,
)
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="the low-M plans serve GB300 (sm_103)",
)

SHARED_INTERMEDIATE = 6144


def _shared_expert(tp: int, plan=None) -> K31SituMLP:
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(tp)
        with torch.device("cuda"):
            mlp = K31SituMLP(
                7168,
                SHARED_INTERMEDIATE // tp,
                dtype=torch.bfloat16,
                mapping=Mapping(world_size=1, tp_size=1, rank=0),
                beta=1.0,
                linear_beta=2.0,
                tp_size=1,
                reduce_output=False,
                low_m_plan=plan,
            )
        with torch.no_grad():
            for p in mlp.parameters():
                p.normal_(0.0, 0.02)
    return mlp.eval()


@pytest.mark.parametrize("tp", [16])
def test_planned_shared_expert_is_bitwise_to_the_whole_mlp(tp) -> None:
    mlp = _shared_expert(tp, k31_low_m_plan(tp, False))
    assert mlp._k31_low_m_plan is not None
    gen = torch.Generator(device="cuda").manual_seed(7)
    with torch.inference_mode():
        for rows in range(1, 33):
            x = torch.randn(rows, 7168, device="cuda", generator=gen).to(torch.bfloat16)
            planned = mlp.low_m_forward(x, reduce=False)
            reference = mlp(x)
            differ = int((planned.view(torch.int16) != reference.view(torch.int16)).sum())
            assert differ == 0, f"TP{tp} rows {rows}: {differ} elements differ"
