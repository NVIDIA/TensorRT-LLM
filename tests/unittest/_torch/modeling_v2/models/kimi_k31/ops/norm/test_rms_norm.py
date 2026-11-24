# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``K31RMSNorm`` parameter holder gives the engine ``RMSNorm``'s bits.

Same weight init and epsilon, and the same FlashInfer kernel on a BF16 input without a residual, so the outputs are
compared bit for bit: on whole rows (hidden 7168, the routed latent 3584) and on the row-strided column blocks of the
fused ``[q_a | kv_a | k_pe]`` projection output that the MLA norms read (q_a at offset 0, kv_a at offset 1536 of 2112).
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm.rms_norm import (
    K31RMSNorm,
)
from tensorrt_llm._torch.modules.rms_norm import RMSNorm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 3), (10, 7)),
    reason="certified on sm_103 (sm_107 pending)",
)

BF16 = torch.bfloat16


@pytest.mark.parametrize("rows", [1, 8, 300, 2048])
@pytest.mark.parametrize(
    "offset, width, stride",
    [
        (0, 7168, 7168),
        (0, 3584, 3584),
        (0, 1536, 1536),
        (0, 512, 512),
        (0, 1536, 2112),
        (1536, 512, 2112),
    ],
    ids=["hidden", "latent", "q_a", "kv_a", "q_a-of-fused", "kv_a-of-fused"],
)
def test_holder_matches_the_engine_rmsnorm(offset, width, stride, rows) -> None:
    norm = RMSNorm(hidden_size=width, eps=1e-5, dtype=BF16).cuda()
    holder = K31RMSNorm(width, eps=1e-5, dtype=BF16).cuda()
    assert (
        torch.equal(holder.weight, norm.weight) and holder.variance_epsilon == norm.variance_epsilon
    )
    assert dict(holder.named_parameters()).keys() == {"weight"}
    gen = torch.Generator(device="cpu").manual_seed(rows + width + offset)
    norm.weight.data.copy_(1 + torch.randn(width, generator=gen).to(BF16) * 0.05)
    holder.weight.data.copy_(norm.weight.data)
    fused = torch.randn(rows, stride, generator=gen).to(device="cuda", dtype=BF16)
    x = fused[:, offset : offset + width]
    assert torch.equal(holder(x), norm(x))
