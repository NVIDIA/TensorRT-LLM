# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm._utils import is_sm_100f

if IS_FLASHINFER_AVAILABLE:
    import flashinfer.norm

    from tensorrt_llm._torch.custom_ops import (
        flashinfer_fused_add_add_rmsnorm,
        flashinfer_fused_add_rmsnorm,
    )


HIDDEN_SIZE = 7168
EPSILON = 1e-6
TOKEN_COUNTS = tuple(range(1, 33))


pytestmark = pytest.mark.skipif(
    not is_sm_100f()
    or not IS_FLASHINFER_AVAILABLE
    or not IS_CUTLASS_DSL_AVAILABLE
    or os.environ.get("FLASHINFER_USE_CUDA_NORM", "0") == "1"
    or (IS_FLASHINFER_AVAILABLE and getattr(flashinfer.norm, "_USE_CUDA_NORM", True)),
    reason="Requires SM100f, FlashInfer 0.6.15 CuTe RMSNorm, and CUTLASS DSL",
)


def _make_inputs(num_tokens: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(20260722 + num_tokens)
    shape = (num_tokens, HIDDEN_SIZE)
    shared = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    routed = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    residual = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    weight = torch.randn((HIDDEN_SIZE,), dtype=torch.bfloat16, device="cuda", generator=generator)
    return shared, routed, residual, weight


def _reference(
    shared: torch.Tensor, routed: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    shared = shared.clone()
    residual = residual.clone()
    shared.add_(routed)
    flashinfer_fused_add_rmsnorm(shared, residual, weight, EPSILON)
    return shared, residual


def _assert_exact(
    actual_shared: torch.Tensor,
    actual_residual: torch.Tensor,
    expected_shared: torch.Tensor,
    expected_residual: torch.Tensor,
) -> None:
    assert torch.equal(actual_residual, expected_residual)
    assert torch.equal(actual_shared, expected_shared)


@pytest.mark.parametrize("num_tokens", TOKEN_COUNTS)
def test_matches_current_sequence_eager(num_tokens: int) -> None:
    shared, routed, residual, weight = _make_inputs(num_tokens)
    expected_shared, expected_residual = _reference(shared, routed, residual, weight)
    routed_before = routed.clone()
    weight_before = weight.clone()

    actual_shared = shared.clone()
    actual_residual = residual.clone()
    shared_ptr = actual_shared.data_ptr()
    residual_ptr = actual_residual.data_ptr()
    flashinfer_fused_add_add_rmsnorm(actual_shared, routed, actual_residual, weight, EPSILON)

    _assert_exact(actual_shared, actual_residual, expected_shared, expected_residual)
    assert actual_shared.data_ptr() == shared_ptr
    assert actual_residual.data_ptr() == residual_ptr
    assert torch.equal(routed, routed_before)
    assert torch.equal(weight, weight_before)


@pytest.mark.parametrize("num_tokens", TOKEN_COUNTS)
def test_matches_current_sequence_cuda_graph(num_tokens: int) -> None:
    shared, routed, residual, weight = _make_inputs(num_tokens)
    expected_shared, expected_residual = _reference(shared, routed, residual, weight)
    actual_shared = shared.clone()
    actual_residual = residual.clone()

    # Compile before capture.  CuTe compilation itself is not graph-capturable.
    flashinfer_fused_add_add_rmsnorm(actual_shared, routed, actual_residual, weight, EPSILON)
    actual_shared.copy_(shared)
    actual_residual.copy_(residual)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flashinfer_fused_add_add_rmsnorm(actual_shared, routed, actual_residual, weight, EPSILON)

    for _ in range(3):
        actual_shared.copy_(shared)
        actual_residual.copy_(residual)
        graph.replay()
        torch.cuda.synchronize()
        _assert_exact(actual_shared, actual_residual, expected_shared, expected_residual)


@pytest.mark.parametrize(
    ("invalid_case", "message"),
    [
        ("rank", "must all be 2D"),
        ("shape", "must match input shape"),
        ("dtype", "must match input dtype"),
        ("weight_shape", "weight must have shape"),
        ("weight_device", "weight must match input dtype and device"),
        ("hidden_stride", "must be contiguous in the hidden dimension"),
    ],
)
def test_rejects_invalid_tensor_contract(invalid_case: str, message: str) -> None:
    shared, routed, residual, weight = _make_inputs(2)
    if invalid_case == "rank":
        shared = shared.unsqueeze(0)
    elif invalid_case == "shape":
        routed = routed[:1]
    elif invalid_case == "dtype":
        residual = residual.to(torch.float16)
    elif invalid_case == "weight_shape":
        weight = weight[:-1]
    elif invalid_case == "weight_device":
        weight = weight.cpu()
    elif invalid_case == "hidden_stride":
        shared = shared.as_strided(shared.shape, (1, shared.shape[0]))
    else:
        raise AssertionError(f"Unhandled invalid case: {invalid_case}")

    with pytest.raises(ValueError, match=message):
        flashinfer_fused_add_add_rmsnorm(shared, routed, residual, weight, EPSILON)
