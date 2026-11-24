# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.visual_gen.modeling_v2.catalog.attention.sol_attention import sol_attention


@pytest.mark.parametrize("cell", sol_attention.CELLS, ids=lambda cell: cell.why)
def test_sol_attention(cell):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("CuTe SOL attention requires SM100 or SM103")
    tokens = cell.spec["tokens"]
    shape = (1, tokens, 2, 128)
    rng = torch.Generator(device="cuda").manual_seed(31)
    block = torch.arange(tokens, device="cuda") // 64
    centers = torch.tensor([1, -1, 0.5, -0.5, 0.125, -0.125], device="cuda")[block]
    q = (1 + torch.randn(shape, device="cuda", generator=rng) / 8).bfloat16()
    k = (
        centers.view(1, -1, 1, 1) + torch.randn(shape, device="cuda", generator=rng) / 8
    ).bfloat16()
    v = torch.randn(shape, dtype=q.dtype, device="cuda", generator=rng)
    kwargs = dict(tau=1.25, thresh_type=cell.spec["thresh_type"], sm_scale=0.05)
    sol_attention.is_valid(q, k, v, **kwargs)
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell import sol_attn as provider

    with patch.object(provider, "sol_attn", wraps=provider.sol_attn) as shared:
        actual = sol_attention(q, k, v, **kwargs)
    shared.assert_called_once()
    torch.testing.assert_close(
        actual,
        sol_attention.reference(q, k, v, **kwargs),
        rtol=0.02,
        atol=0.02,
    )
