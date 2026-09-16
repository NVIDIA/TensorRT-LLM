# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""L1/T1: fused video projection versus Framework's contiguous addmm path."""

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.visual_gen.models.cosmos3.transformer_cosmos3 import _project_video_tokens


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA projection required")
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.no_grad()
def test_video_projection_fuses_bias_for_gapped_batches(dtype, batch: int, compiled: bool) -> None:
    generator = torch.Generator(device="cuda").manual_seed(5617)
    hidden = torch.randn(batch, 3093, 2048, device="cuda", dtype=dtype, generator=generator)
    projection = nn.Linear(2048, 192, device="cuda", dtype=dtype)
    projection.weight.copy_(
        torch.randn(projection.weight.shape, device="cuda", dtype=dtype, generator=generator)
        / 2048**0.5
    )
    projection.bias.copy_(
        torch.randn(projection.bias.shape, device="cuda", dtype=dtype, generator=generator)
    )
    tokens = hidden[:, :3060]
    assert tokens.is_contiguous() == (batch == 1)
    expected = torch.addmm(
        projection.bias, tokens.contiguous().view(-1, 2048), projection.weight.t()
    ).view(batch, 3060, 192)
    project = (
        torch.compile(_project_video_tokens, fullgraph=True) if compiled else _project_video_tokens
    )
    for _ in range(2):
        actual = project(tokens, projection)
        delta, reference = actual.double() - expected.double(), expected.double()
        rel_l2 = delta.norm() / reference.norm()
        cosine = torch.nn.functional.cosine_similarity(
            actual.double().flatten(), reference.flatten(), dim=0
        )
        print(
            f"video projection {dtype=} {batch=} {compiled=}: "
            f"linf_abs={delta.abs().max().item():.9g}, "
            f"linf_rel={(delta.abs().max() / reference.abs().max()).item():.9g}, "
            f"l2_abs={delta.norm().item():.9g}, l2_rel={rel_l2.item():.9g}"
        )
        torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
        assert rel_l2 <= 1e-2
        assert cosine >= 0.9999
    if dtype == torch.bfloat16 and batch == 2:
        # Negative control: the old non-contiguous linear loses the bias fusion.
        assert not torch.allclose(projection(tokens), expected, atol=1e-3, rtol=1e-3)
