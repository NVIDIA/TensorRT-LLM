# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the fused Kimi K3 attention-residual op."""

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.models import modeling_kimi_linear as kimi_modeling
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
    block_residual_hf = (
        torch.randn(
            num_tokens,
            num_snapshots,
            HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.05
    )

    expected = apply_attn_res_reference(
        prefix_sum,
        block_residual_hf,
        projection.weight,
        norm.weight,
        RMS_EPS,
    )
    block_residual = block_residual_hf.transpose(0, 1).contiguous()
    actual = _apply_attn_res_fused(prefix_sum, block_residual, projection, norm)

    assert actual is not None
    cosine, relative_l2 = _similarity(actual, expected)
    assert cosine > 0.999
    assert relative_l2 < 3e-2


class _AddConstant(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, hidden_states: torch.Tensor, *_args) -> torch.Tensor:
        return hidden_states + self.value


def _make_snapshot_test_layer(layer_idx: int) -> kimi_modeling.KimiLinearDecoderLayer:
    layer = kimi_modeling.KimiLinearDecoderLayer.__new__(kimi_modeling.KimiLinearDecoderLayer)
    nn.Module.__init__(layer)
    layer.layer_idx = layer_idx
    layer.attn_res_block_size = 12
    layer.self_attention_res_proj = nn.Linear(4, 1, bias=False)
    layer.self_attention_res_norm = KimiK3RMSNorm(4)
    layer.mlp_res_proj = nn.Linear(4, 1, bias=False)
    layer.mlp_res_norm = KimiK3RMSNorm(4)
    with torch.no_grad():
        layer.self_attention_res_proj.weight.fill_(0.01)
        layer.mlp_res_proj.weight.fill_(0.02)
    layer.input_layernorm = nn.Identity()
    layer.post_attention_layernorm = nn.Identity()
    layer.is_kda = True
    layer.self_attn = _AddConstant(1.0)
    layer.is_moe = False
    layer.mlp = _AddConstant(2.0)
    layer._mlp_allreduce = None
    return layer


def _legacy_decoder_layer_forward(
    layer: kimi_modeling.KimiLinearDecoderLayer,
    hidden_states: torch.Tensor,
    block_residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix_sum = hidden_states
    if block_residual.shape[0] > 0:
        hidden_states = kimi_modeling._apply_attn_res(
            prefix_sum,
            block_residual,
            layer.self_attention_res_proj,
            layer.self_attention_res_norm,
        )
    if layer.layer_idx % layer.attn_res_block_size == 0:
        block_residual = torch.cat((block_residual, prefix_sum.unsqueeze(0)), dim=0)
        prefix_sum = None

    hidden_states = layer.self_attn(hidden_states, None)
    prefix_sum = hidden_states if prefix_sum is None else prefix_sum + hidden_states
    hidden_states = kimi_modeling._apply_attn_res(
        prefix_sum, block_residual, layer.mlp_res_proj, layer.mlp_res_norm
    )
    hidden_states = layer.mlp(hidden_states)
    return prefix_sum + hidden_states, block_residual


def test_preallocated_snapshot_bank_matches_legacy_growth() -> None:
    torch.manual_seed(0)
    legacy_hidden_states = torch.randn(2, 4)
    preallocated_hidden_states = legacy_hidden_states.clone()
    legacy_block_residual = torch.empty(0, 2, 4)
    snapshot_bank = torch.empty(3, 2, 4)
    bank_ptr = snapshot_bank.data_ptr()
    num_snapshots = 0

    for layer_idx in range(25):
        layer = _make_snapshot_test_layer(layer_idx)
        legacy_hidden_states, legacy_block_residual = _legacy_decoder_layer_forward(
            layer, legacy_hidden_states, legacy_block_residual
        )
        preallocated_hidden_states, num_snapshots = layer(
            preallocated_hidden_states, snapshot_bank, num_snapshots, None, None
        )
        torch.testing.assert_close(preallocated_hidden_states, legacy_hidden_states, rtol=0, atol=0)
        assert snapshot_bank.data_ptr() == bank_ptr

    torch.testing.assert_close(snapshot_bank[:num_snapshots], legacy_block_residual, rtol=0, atol=0)
