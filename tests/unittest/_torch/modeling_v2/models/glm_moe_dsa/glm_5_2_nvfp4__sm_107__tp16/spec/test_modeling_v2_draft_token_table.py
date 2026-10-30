# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the construction-time operands of the glm_draft_ehproj_table entry."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_107__tp16.ops.spec._draft_token_table import (  # noqa: E501
    build_draft_token_table,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    get_sm_version() != 107, reason="the GLM-5.2 tp16 target's ops are certified on sm_107"
)

HIDDEN = 6144
EPS = 1e-5


@pytest.mark.parametrize("tp_size,tp_rank", [(16, 0), (16, 11), (4, 3)])
def test_table_is_the_token_half_of_eh_proj(tp_size, tp_rank) -> None:
    """Each table row is eh_proj's embedding half of one token of the rank's shard.

    Against an FP32 RMSNorm and product, rounded where the builder rounds (the normalized row and the product to
    BF16), the rows differ only where the kernel's approximate reciprocal square root moves the normalized row's
    BF16 rounding (one ulp of an input, carried through the product) and where the order of a 6144-term FP32 sum
    moves the product's (one ulp of the element): 2 ulp per element, 0.25 ulp of relative RMS. The hidden-half
    rows are a slice of the weight, compared exactly.
    """
    g = torch.Generator(device="cuda").manual_seed(tp_size * 100 + tp_rank)
    vocab_shard = 96
    weight = (torch.randn(HIDDEN, 2 * HIDDEN, generator=g, device="cuda") * HIDDEN**-0.5).to(
        torch.bfloat16
    )
    embed = torch.randn(vocab_shard * tp_size, HIDDEN, generator=g, device="cuda").to(
        torch.bfloat16
    )
    enorm = (1.0 + 0.1 * torch.randn(HIDDEN, generator=g, device="cuda")).to(torch.bfloat16)

    table, hidden_rows = build_draft_token_table(
        weight, embed, enorm, EPS, tp_rank, tp_size, vocab_shard
    )

    rows = embed[tp_rank * vocab_shard : (tp_rank + 1) * vocab_shard].float()
    rstd = torch.rsqrt(rows.pow(2).mean(dim=-1, keepdim=True) + EPS)
    normed = ((rows * rstd) * enorm.float()).to(torch.bfloat16).float()
    with true_fp32_matmul():
        expected = (normed @ weight[:, :HIDDEN].float().t()).to(torch.bfloat16)
    assert table.shape == (vocab_shard, HIDDEN) and table.dtype == torch.bfloat16
    assert_within_ulp(table, expected, element_ulp=2.0, rms_ulp=0.25)
    share = HIDDEN // tp_size
    torch.testing.assert_close(
        hidden_rows, weight[tp_rank * share : (tp_rank + 1) * share, HIDDEN:], rtol=0, atol=0
    )
    assert hidden_rows.is_contiguous()
