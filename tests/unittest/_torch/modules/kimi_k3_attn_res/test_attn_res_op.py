# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the fused Kimi K3 attention-residual op."""

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.models.modeling_kimi_linear import KimiK3RMSNorm, _apply_attn_res_fused
from tensorrt_llm._torch.modules.kimi_k3_attn_res import apply_attn_res_reference

HIDDEN_SIZE = 7168
RMS_EPS = 1e-6


def _has_supported_gpu() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) in {(10, 0), (10, 3)}


pytestmark = pytest.mark.skipif(
    not _has_supported_gpu(),
    reason="Kimi K3 is supported only on Blackwell (SM100/SM103)",
)


def _similarity(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    actual_float = actual.float()
    expected_float = expected.float()
    cosine = torch.nn.functional.cosine_similarity(
        actual_float.flatten(), expected_float.flatten(), dim=0
    ).item()
    relative_l2 = ((actual_float - expected_float).norm() / (expected_float.norm() + 1e-12)).item()
    return cosine, relative_l2


@pytest.mark.parametrize(
    ("num_tokens", "num_snapshots"),
    [
        (1, 0),
        (1, 1),
        (1, 3),
        (1, 7),
        (1, 11),
        (64, 0),
        (128, 3),
        (1024, 11),
        (300, 5),
        (16384, 11),
    ],
)
@torch.no_grad()
def test_fused_attn_res_matches_torch_reference(num_tokens: int, num_snapshots: int) -> None:
    torch.manual_seed(0)
    projection = nn.Linear(HIDDEN_SIZE, 1, bias=False, dtype=torch.bfloat16, device="cuda")
    norm = KimiK3RMSNorm(HIDDEN_SIZE, eps=RMS_EPS).to(device="cuda", dtype=torch.bfloat16)
    projection.weight.mul_(0.02)

    prefix_sum = torch.randn(num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.05
    # Kernel-native [num_snapshots, num_tokens, H] layout — the layout the
    # model's `_apply_attn_res_fused` wrapper takes (snapshots are stacked on
    # dim 0 at runtime). The reference implementation uses the HF layout with
    # the snapshot axis in the middle, hence the transpose below.
    block_residual = (
        torch.randn(
            num_snapshots,
            num_tokens,
            HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.05
    )

    expected = apply_attn_res_reference(
        prefix_sum,
        block_residual.transpose(0, 1),
        projection.weight,
        norm.weight,
        RMS_EPS,
    )
    actual = _apply_attn_res_fused(prefix_sum, block_residual, projection, norm)

    assert actual is not None
    cosine, relative_l2 = _similarity(actual, expected)
    assert cosine > 0.999
    assert relative_l2 < 3e-2
