# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the fused Kimi K3 attention-residual op."""

from unittest import mock

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm._torch.models import modeling_kimi_linear
from tensorrt_llm._torch.models.modeling_kimi_linear import (
    KimiK3RMSNorm,
    _apply_attn_res,
    _apply_attn_res_add_and_rmsnorm,
    _apply_attn_res_add_rmsnorm_fused,
    _apply_attn_res_and_rmsnorm,
    _apply_attn_res_fused,
    _apply_attn_res_rmsnorm_fused,
)
from tensorrt_llm._torch.modules.kimi_k3_attn_res import apply_attn_res_reference
from tensorrt_llm._torch.modules.rms_norm import RMSNorm

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


OUTPUT_RMS_EPS = 1e-6
# Production decode: N=4 single-CTA and N=8 split-K. Other legal N share those two topologies.
_DECODE_SNAPSHOTS = (3, 7)


def _production_rms_norm(
    hidden_states: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    if IS_FLASHINFER_AVAILABLE:
        from tensorrt_llm._torch.custom_ops import flashinfer_rmsnorm

        return flashinfer_rmsnorm(hidden_states.contiguous(), weight, eps)
    hidden_float = hidden_states.float()
    variance = hidden_float.square().mean(dim=-1, keepdim=True)
    return weight * (hidden_float * torch.rsqrt(variance + eps)).to(hidden_states.dtype)


def _make_decode_case(num_snapshots: int):
    torch.manual_seed(0)
    prefix_sum = torch.randn(1, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.05
    addend = torch.randn(1, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.05
    block_residual = (
        torch.randn(num_snapshots, 1, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.05
    )
    projection = nn.Linear(HIDDEN_SIZE, 1, bias=False, dtype=torch.bfloat16, device="cuda")
    score_norm = KimiK3RMSNorm(HIDDEN_SIZE, eps=RMS_EPS).to(device="cuda", dtype=torch.bfloat16)
    output_norm = RMSNorm(
        hidden_size=HIDDEN_SIZE,
        eps=OUTPUT_RMS_EPS,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    projection.weight.mul_(0.02)
    return prefix_sum, addend, block_residual, projection, score_norm, output_norm


@pytest.mark.parametrize("num_snapshots", _DECODE_SNAPSHOTS)
@torch.no_grad()
def test_decode_rmsnorm_fusion_matches_unfused(num_snapshots: int) -> None:
    prefix_sum, _addend, block_residual, projection, score_norm, output_norm = _make_decode_case(
        num_snapshots
    )
    expected = output_norm(_apply_attn_res(prefix_sum, block_residual, projection, score_norm))
    actual = _apply_attn_res_and_rmsnorm(
        prefix_sum, block_residual, projection, score_norm, output_norm
    )
    cosine, relative_l2 = _similarity(actual, expected)
    assert cosine > 0.9999
    assert relative_l2 < 5e-3


@pytest.mark.parametrize("num_snapshots", _DECODE_SNAPSHOTS)
@torch.no_grad()
def test_decode_add_rmsnorm_fusion_matches_separate_add(num_snapshots: int) -> None:
    prefix_sum, addend, block_residual, projection, score_norm, output_norm = _make_decode_case(
        num_snapshots
    )
    expected_updated = prefix_sum + addend
    expected_output = output_norm(
        _apply_attn_res(expected_updated, block_residual, projection, score_norm)
    )
    actual_updated, actual_output = _apply_attn_res_add_and_rmsnorm(
        prefix_sum, addend, block_residual, projection, score_norm, output_norm
    )
    assert torch.equal(actual_updated, expected_updated)
    cosine, relative_l2 = _similarity(actual_output, expected_output)
    assert cosine > 0.9999
    assert relative_l2 < 5e-3


@torch.no_grad()
def test_decode_fusion_gate_skips_prefill() -> None:
    prefix_sum, addend, block_residual, projection, score_norm, output_norm = _make_decode_case(3)
    prefix_sum = prefix_sum.expand(64, -1).contiguous()
    addend = addend.expand(64, -1).contiguous()
    block_residual = block_residual.expand(-1, 64, -1).contiguous()
    assert (
        _apply_attn_res_rmsnorm_fused(
            prefix_sum, block_residual, projection, score_norm, output_norm
        )
        is None
    )
    assert (
        _apply_attn_res_add_rmsnorm_fused(
            prefix_sum, addend, block_residual, projection, score_norm, output_norm
        )
        is None
    )


@torch.no_grad()
def test_decode_norm_flag_keeps_unfused_path(monkeypatch: pytest.MonkeyPatch) -> None:
    prefix_sum, addend, block_residual, projection, score_norm, output_norm = _make_decode_case(3)
    unexpected = mock.Mock(side_effect=AssertionError("norm flag off still reached fused op"))
    monkeypatch.setattr(modeling_kimi_linear, "_FUSED_ATTN_RES_NORM_ENABLED", False)
    monkeypatch.setattr(modeling_kimi_linear, "_apply_attn_res_rmsnorm_fused", unexpected)
    monkeypatch.setattr(modeling_kimi_linear, "_apply_attn_res_add_rmsnorm_fused", unexpected)
    expected_updated = prefix_sum + addend
    expected_output = output_norm(
        _apply_attn_res(expected_updated, block_residual, projection, score_norm)
    )
    actual_updated, actual_output = _apply_attn_res_add_and_rmsnorm(
        prefix_sum, addend, block_residual, projection, score_norm, output_norm
    )
    unexpected.assert_not_called()
    assert torch.equal(actual_updated, expected_updated)
    cosine, relative_l2 = _similarity(actual_output, expected_output)
    assert cosine > 0.9999
    assert relative_l2 < 5e-3


@torch.no_grad()
def test_decode_add_rmsnorm_cuda_graph_replay() -> None:
    prefix_sum, addend, block_residual, projection, score_norm, output_norm = _make_decode_case(3)
    layer = prefix_sum.reshape(1, 1, HIDDEN_SIZE).contiguous()
    addend_b = addend.reshape(1, 1, HIDDEN_SIZE).contiguous()
    block = block_residual.reshape(block_residual.shape[0], 1, 1, HIDDEN_SIZE).contiguous()
    expected_updated = layer + addend_b
    mixed, *_ = torch.ops.trtllm.attn_res_fwd(
        expected_updated, block, projection.weight.reshape(-1), score_norm.weight, RMS_EPS
    )
    expected_output = _production_rms_norm(mixed, output_norm.weight, OUTPUT_RMS_EPS)

    torch.ops.trtllm.attn_res_add_rmsnorm_fwd(
        layer,
        addend_b,
        block,
        projection.weight.reshape(-1),
        score_norm.weight,
        output_norm.weight,
        RMS_EPS,
        OUTPUT_RMS_EPS,
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual_updated, actual_output = torch.ops.trtllm.attn_res_add_rmsnorm_fwd(
            layer,
            addend_b,
            block,
            projection.weight.reshape(-1),
            score_norm.weight,
            output_norm.weight,
            RMS_EPS,
            OUTPUT_RMS_EPS,
        )
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(actual_updated, expected_updated)
    cosine, relative_l2 = _similarity(actual_output, expected_output)
    assert cosine > 0.9999
    assert relative_l2 < 5e-3
