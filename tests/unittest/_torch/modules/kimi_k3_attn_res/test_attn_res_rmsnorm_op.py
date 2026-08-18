# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the fused Kimi K3 attention-residual + RMSNorm op."""

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
    _apply_attn_res_rmsnorm_fused,
)
from tensorrt_llm._torch.modules.rms_norm import RMSNorm

HIDDEN_SIZE = 7168
ATTN_RES_RMS_EPS = 1e-6
OUTPUT_RMS_EPS = 1e-6


def _has_supported_gpu() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) in {
        (10, 0),
        (10, 3),
    }


pytestmark = pytest.mark.skipif(
    not _has_supported_gpu(),
    reason="Kimi K3 attention-residual kernels require SM100/SM103",
)


def _production_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply the unfused production RMSNorm that the new op replaces."""
    if IS_FLASHINFER_AVAILABLE:
        from tensorrt_llm._torch.custom_ops import flashinfer_rmsnorm

        return flashinfer_rmsnorm(hidden_states.contiguous(), weight, eps)

    hidden_float = hidden_states.float()
    variance = hidden_float.square().mean(dim=-1, keepdim=True)
    normalized = hidden_float * torch.rsqrt(variance + eps)
    return weight * normalized.to(hidden_states.dtype)


def _make_inputs(
    num_tokens: int,
    num_snapshots: int,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    layer_residual = (
        torch.randn(
            num_tokens,
            1,
            HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.05
    )
    block_residual = (
        torch.randn(
            num_snapshots,
            num_tokens,
            1,
            HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.05
    )
    res_weight = torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.02
    score_rms_weight = 1 + torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.02
    output_rms_weight = 1 + torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.02
    return (
        layer_residual,
        block_residual,
        res_weight.contiguous(),
        score_rms_weight.contiguous(),
        output_rms_weight.contiguous(),
    )


def _unfused_reference(
    layer_residual: torch.Tensor,
    block_residual: torch.Tensor,
    res_weight: torch.Tensor,
    score_rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor,
) -> torch.Tensor:
    mixed, _rsigma, _probs, _logits = torch.ops.trtllm.attn_res_fwd(
        layer_residual,
        block_residual,
        res_weight,
        score_rms_weight,
        ATTN_RES_RMS_EPS,
    )
    return _production_rms_norm(mixed, output_rms_weight, OUTPUT_RMS_EPS)


def _fused(
    layer_residual: torch.Tensor,
    block_residual: torch.Tensor,
    res_weight: torch.Tensor,
    score_rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.trtllm.attn_res_rmsnorm_fwd(
        layer_residual,
        block_residual,
        res_weight,
        score_rms_weight,
        output_rms_weight,
        ATTN_RES_RMS_EPS,
        OUTPUT_RMS_EPS,
    )


def _fused_add(
    layer_residual: torch.Tensor,
    layer_residual_add: torch.Tensor,
    block_residual: torch.Tensor,
    res_weight: torch.Tensor,
    score_rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.trtllm.attn_res_add_rmsnorm_fwd(
        layer_residual,
        layer_residual_add,
        block_residual,
        res_weight,
        score_rms_weight,
        output_rms_weight,
        ATTN_RES_RMS_EPS,
        OUTPUT_RMS_EPS,
    )


def _similarity(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[float, float]:
    actual_float = actual.float()
    expected_float = expected.float()
    cosine = torch.nn.functional.cosine_similarity(
        actual_float.flatten(),
        expected_float.flatten(),
        dim=0,
    ).item()
    relative_l2 = ((actual_float - expected_float).norm() / (expected_float.norm() + 1e-12)).item()
    return cosine, relative_l2


@pytest.mark.parametrize(
    ("num_tokens", "num_snapshots"),
    [
        (1, 0),  # N=1 single-CTA decode
        (1, 1),  # N=2 single-CTA decode
        (1, 2),  # N=3 single-CTA decode
        (1, 3),  # N=4 single-CTA decode
        (1, 4),  # N=5 CTA-cluster decode
        (1, 5),  # N=6 CTA-cluster decode
        (1, 6),  # N=7 CTA-cluster decode
        (1, 7),  # N=8 CTA-cluster decode
        (1, 8),  # N=9 CTA-cluster decode
        (1, 11),  # N=12 CTA-cluster decode
    ],
)
@torch.no_grad()
def test_attn_res_rmsnorm_matches_unfused(
    num_tokens: int,
    num_snapshots: int,
) -> None:
    inputs = _make_inputs(num_tokens, num_snapshots)
    expected = _unfused_reference(*inputs)
    actual = _fused(*inputs)

    assert actual.shape == expected.shape
    assert actual.dtype == torch.bfloat16
    cosine, relative_l2 = _similarity(actual, expected)
    assert cosine > 0.9999
    assert relative_l2 < 5e-3


@torch.no_grad()
def test_attn_res_rmsnorm_op_rejects_multi_token() -> None:
    inputs = _make_inputs(num_tokens=2, num_snapshots=3)
    with pytest.raises(RuntimeError, match="only production decode shape"):
        _fused(*inputs)


@torch.no_grad()
def test_attn_res_rmsnorm_cuda_graph_replay() -> None:
    inputs = _make_inputs(num_tokens=1, num_snapshots=3)
    expected = _unfused_reference(*inputs)

    _fused(*inputs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = _fused(*inputs)
    graph.replay()
    torch.cuda.synchronize()

    cosine, relative_l2 = _similarity(actual, expected)
    assert cosine > 0.9999
    assert relative_l2 < 5e-3


@pytest.mark.parametrize("num_snapshots", list(range(9)) + [11])
@torch.no_grad()
def test_attn_res_add_rmsnorm_matches_separate_add(
    num_snapshots: int,
) -> None:
    inputs = _make_inputs(num_tokens=1, num_snapshots=num_snapshots)
    layer_residual_add = (torch.randn_like(inputs[0]) * 0.05).contiguous()
    expected_updated = inputs[0] + layer_residual_add
    expected_output = _unfused_reference(
        expected_updated,
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
    )
    actual_updated, actual_output = _fused_add(
        inputs[0],
        layer_residual_add,
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
    )

    assert torch.equal(actual_updated, expected_updated)
    cosine, relative_l2 = _similarity(actual_output, expected_output)
    assert cosine > 0.9999
    assert relative_l2 < 5e-3


@pytest.mark.parametrize("num_snapshots", [3, 7, 8])
@torch.no_grad()
def test_attn_res_add_rmsnorm_cuda_graph_replay(
    num_snapshots: int,
) -> None:
    inputs = _make_inputs(num_tokens=1, num_snapshots=num_snapshots)
    layer_residual_add = (torch.randn_like(inputs[0]) * 0.05).contiguous()
    expected_updated = inputs[0] + layer_residual_add
    expected_output = _unfused_reference(
        expected_updated,
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
    )

    _fused_add(inputs[0], layer_residual_add, *inputs[1:])
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual_updated, actual_output = _fused_add(inputs[0], layer_residual_add, *inputs[1:])
    for _ in range(100):
        graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(actual_updated, expected_updated)
    cosine, relative_l2 = _similarity(actual_output, expected_output)
    assert cosine > 0.9999
    assert relative_l2 < 5e-3


@torch.no_grad()
def test_model_helper_dispatches_fused_attn_res_add_rmsnorm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        layer_residual,
        block_residual,
        res_weight,
        score_rms_weight,
        output_rms_weight,
    ) = _make_inputs(num_tokens=1, num_snapshots=3)
    layer_residual_add = (torch.randn_like(layer_residual) * 0.05).contiguous()
    projection = nn.Linear(
        HIDDEN_SIZE,
        1,
        bias=False,
        dtype=torch.bfloat16,
        device="cuda",
    )
    score_norm = KimiK3RMSNorm(
        HIDDEN_SIZE,
        eps=ATTN_RES_RMS_EPS,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    output_norm = RMSNorm(
        hidden_size=HIDDEN_SIZE,
        eps=OUTPUT_RMS_EPS,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    projection.weight.copy_(res_weight.reshape(1, -1))
    score_norm.weight.copy_(score_rms_weight)
    output_norm.weight.copy_(output_rms_weight)

    prefix_sum = layer_residual[:, 0, :]
    addend = layer_residual_add[:, 0, :]
    block_kernel_layout = block_residual[:, :, 0, :]
    expected_updated = prefix_sum + addend
    expected_output = output_norm(
        _apply_attn_res(
            expected_updated,
            block_kernel_layout,
            projection,
            score_norm,
        )
    )
    unexpected_fallback = mock.Mock(
        side_effect=AssertionError("model helper did not dispatch fused add")
    )
    monkeypatch.setattr(
        modeling_kimi_linear,
        "_apply_attn_res_and_rmsnorm",
        unexpected_fallback,
    )
    actual_updated, actual_output = _apply_attn_res_add_and_rmsnorm(
        prefix_sum,
        addend,
        block_kernel_layout,
        projection,
        score_norm,
        output_norm,
    )

    unexpected_fallback.assert_not_called()
    assert torch.equal(actual_updated, expected_updated)
    cosine, relative_l2 = _similarity(actual_output, expected_output)
    assert cosine > 0.9999
    assert relative_l2 < 5e-3


@torch.no_grad()
def test_model_helper_rejects_multi_token_fused_add() -> None:
    (
        layer_residual,
        block_residual,
        res_weight,
        score_rms_weight,
        output_rms_weight,
    ) = _make_inputs(num_tokens=64, num_snapshots=5)
    layer_residual_add = (torch.randn_like(layer_residual) * 0.05).contiguous()
    projection = nn.Linear(
        HIDDEN_SIZE,
        1,
        bias=False,
        dtype=torch.bfloat16,
        device="cuda",
    )
    score_norm = KimiK3RMSNorm(
        HIDDEN_SIZE,
        eps=ATTN_RES_RMS_EPS,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    output_norm = RMSNorm(
        hidden_size=HIDDEN_SIZE,
        eps=OUTPUT_RMS_EPS,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    projection.weight.copy_(res_weight.reshape(1, -1))
    score_norm.weight.copy_(score_rms_weight)
    output_norm.weight.copy_(output_rms_weight)

    assert (
        _apply_attn_res_add_rmsnorm_fused(
            layer_residual[:, 0, :],
            layer_residual_add[:, 0, :],
            block_residual[:, :, 0, :],
            projection,
            score_norm,
            output_norm,
        )
        is None
    )


@pytest.mark.parametrize(
    ("num_tokens", "num_snapshots"),
    [
        (1, 3),
    ],
)
@torch.no_grad()
def test_model_helper_dispatches_fused_attn_res_rmsnorm(
    num_tokens: int,
    num_snapshots: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        layer_residual,
        block_residual,
        res_weight,
        score_rms_weight,
        output_rms_weight,
    ) = _make_inputs(num_tokens, num_snapshots)
    projection = nn.Linear(
        HIDDEN_SIZE,
        1,
        bias=False,
        dtype=torch.bfloat16,
        device="cuda",
    )
    score_norm = KimiK3RMSNorm(
        HIDDEN_SIZE,
        eps=ATTN_RES_RMS_EPS,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    output_norm = RMSNorm(
        hidden_size=HIDDEN_SIZE,
        eps=OUTPUT_RMS_EPS,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    projection.weight.copy_(res_weight.reshape(1, -1))
    score_norm.weight.copy_(score_rms_weight)
    output_norm.weight.copy_(output_rms_weight)

    prefix_sum = layer_residual[:, 0, :]
    block_kernel_layout = block_residual[:, :, 0, :]
    expected = output_norm(
        _apply_attn_res(
            prefix_sum,
            block_kernel_layout,
            projection,
            score_norm,
        )
    )
    unexpected_fallback = mock.Mock(
        side_effect=AssertionError("model helper did not dispatch the fused op")
    )
    monkeypatch.setattr(
        modeling_kimi_linear,
        "_apply_attn_res",
        unexpected_fallback,
    )
    actual = _apply_attn_res_and_rmsnorm(
        prefix_sum,
        block_kernel_layout,
        projection,
        score_norm,
        output_norm,
    )

    cosine, relative_l2 = _similarity(actual, expected)
    unexpected_fallback.assert_not_called()
    assert cosine > 0.9999
    assert relative_l2 < 5e-3


@torch.no_grad()
def test_model_helper_keeps_multi_token_rmsnorm_split() -> None:
    (
        layer_residual,
        block_residual,
        res_weight,
        score_rms_weight,
        output_rms_weight,
    ) = _make_inputs(num_tokens=64, num_snapshots=5)
    projection = nn.Linear(
        HIDDEN_SIZE,
        1,
        bias=False,
        dtype=torch.bfloat16,
        device="cuda",
    )
    score_norm = KimiK3RMSNorm(
        HIDDEN_SIZE,
        eps=ATTN_RES_RMS_EPS,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    output_norm = RMSNorm(
        hidden_size=HIDDEN_SIZE,
        eps=OUTPUT_RMS_EPS,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    projection.weight.copy_(res_weight.reshape(1, -1))
    score_norm.weight.copy_(score_rms_weight)
    output_norm.weight.copy_(output_rms_weight)

    prefix_sum = layer_residual[:, 0, :]
    block_kernel_layout = block_residual[:, :, 0, :]
    assert (
        _apply_attn_res_rmsnorm_fused(
            prefix_sum,
            block_kernel_layout,
            projection,
            score_norm,
            output_norm,
        )
        is None
    )

    expected = output_norm(
        _apply_attn_res(
            prefix_sum,
            block_kernel_layout,
            projection,
            score_norm,
        )
    )
    actual = _apply_attn_res_and_rmsnorm(
        prefix_sum,
        block_kernel_layout,
        projection,
        score_norm,
        output_norm,
    )
    cosine, relative_l2 = _similarity(actual, expected)
    assert cosine > 0.9999
    assert relative_l2 < 5e-3


@torch.no_grad()
def test_model_helper_norm_flag_keeps_unfused_norm_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KIMI_K3_FUSED_ATTN_RES_NORM=0 must keep attn_res_fwd + production RMSNorm.

    KIMI_K3_FUSED_ATTN_RES=0 is the wrong A/B knob: it drops all the way to
    the fp32 reference and skips the pre-port path.
    """
    (
        layer_residual,
        block_residual,
        res_weight,
        score_rms_weight,
        output_rms_weight,
    ) = _make_inputs(num_tokens=1, num_snapshots=3)
    projection = nn.Linear(
        HIDDEN_SIZE,
        1,
        bias=False,
        dtype=torch.bfloat16,
        device="cuda",
    )
    score_norm = KimiK3RMSNorm(
        HIDDEN_SIZE,
        eps=ATTN_RES_RMS_EPS,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    output_norm = RMSNorm(
        hidden_size=HIDDEN_SIZE,
        eps=OUTPUT_RMS_EPS,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    projection.weight.copy_(res_weight.reshape(1, -1))
    score_norm.weight.copy_(score_rms_weight)
    output_norm.weight.copy_(output_rms_weight)

    prefix_sum = layer_residual[:, 0, :]
    block_kernel_layout = block_residual[:, :, 0, :]
    unexpected_fused = mock.Mock(
        side_effect=AssertionError("norm flag off still reached the fused norm op")
    )
    monkeypatch.setattr(modeling_kimi_linear, "_FUSED_ATTN_RES_NORM_ENABLED", False)
    monkeypatch.setattr(
        modeling_kimi_linear,
        "_apply_attn_res_rmsnorm_fused",
        unexpected_fused,
    )
    expected = output_norm(
        _apply_attn_res(
            prefix_sum,
            block_kernel_layout,
            projection,
            score_norm,
        )
    )
    actual = _apply_attn_res_and_rmsnorm(
        prefix_sum,
        block_kernel_layout,
        projection,
        score_norm,
        output_norm,
    )
    unexpected_fused.assert_not_called()
    cosine, relative_l2 = _similarity(actual, expected)
    assert cosine > 0.9999
    assert relative_l2 < 5e-3
