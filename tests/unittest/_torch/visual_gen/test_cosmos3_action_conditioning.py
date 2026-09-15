# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L1/T1: compiled action timestep insertion vs Framework's BF16 store boundary."""

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.cosmos3.transformer_cosmos3 import (
    _add_action_timestep_embedding,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA compilation required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("condition_state", [False, True])
def test_compiled_action_timestep_rounding(dtype, batch: int, condition_state: bool) -> None:
    """Same-precision oracle rounds bias+modality before Framework's scatter-add."""
    generator = torch.Generator(device="cuda").manual_seed(42)
    projected = torch.randn(batch, 33, 2048, device="cuda", dtype=dtype, generator=generator)
    bias = torch.randn(batch, 1, 2048, device="cuda", dtype=dtype, generator=generator)
    modality = torch.randn(2048, device="cuda", dtype=dtype, generator=generator)
    time = torch.randn(batch, 2048, device="cuda", dtype=dtype, generator=generator)
    mask = None
    if condition_state:
        mask = torch.ones(batch, 33, 1, device="cuda", dtype=dtype)
        mask[:, 0] = 0

    def condition(projected, bias, modality, time, mask):
        return _add_action_timestep_embedding(projected + bias + modality, time, mask)

    before_time = (projected.float() + bias.float() + modality.float()).to(dtype)
    time_rows = time.float().unsqueeze(1)
    if mask is not None:
        time_rows = time_rows * mask.float()
    expected = (before_time.float() + time_rows).to(dtype)
    if dtype == torch.bfloat16:
        fused = (projected.float() + bias.float() + modality.float() + time_rows).to(dtype)
        assert not torch.allclose(fused, expected, atol=1e-3, rtol=1e-3)

    compiled = torch.compile(condition, fullgraph=True, dynamic=True)
    for _ in range(2):
        actual = compiled(projected, bias, modality, time, mask)
        error = actual.double() - expected.double()
        rel_l2 = error.norm() / expected.double().norm()
        cosine = torch.nn.functional.cosine_similarity(
            actual.double().flatten(), expected.double().flatten(), dim=0
        )
        print(
            f"action conditioning {dtype=} {batch=} {condition_state=}: "
            f"max_abs={error.abs().max().item():.9g}, "
            f"p99_abs={error.abs().quantile(0.99).item():.9g}, "
            f"rel_l2={rel_l2.item():.9g}, cosine={cosine.item():.9g}"
        )
        torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
        assert rel_l2 <= 1e-2
        assert cosine >= 0.9999
        if condition_state:
            torch.testing.assert_close(actual[:, 0], before_time[:, 0], atol=1e-3, rtol=1e-3)
