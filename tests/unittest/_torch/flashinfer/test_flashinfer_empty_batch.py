# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.custom_ops import flashinfer_custom_ops  # noqa: F401
from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not IS_FLASHINFER_AVAILABLE,
    reason="CUDA and FlashInfer are required",
)


@pytest.mark.parametrize("op_name", ["flashinfer_silu_and_mul", "flashinfer_gelu_tanh_and_mul"])
def test_empty_gated_activation(op_name):
    x = torch.empty((0, 128), device="cuda", dtype=torch.float16)
    y = getattr(torch.ops.trtllm, op_name)(x)
    torch.cuda.synchronize()
    assert y.shape == (0, 64)
    assert y.dtype == x.dtype
    assert y.device == x.device


def test_empty_rmsnorm():
    x = torch.empty((0, 64), device="cuda", dtype=torch.float16)
    weight = torch.ones(64, device=x.device, dtype=x.dtype)
    y = torch.ops.trtllm.flashinfer_rmsnorm(x, weight, 1e-6)
    torch.cuda.synchronize()
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.device == x.device


def test_empty_fused_add_rmsnorm():
    x = torch.empty((0, 64), device="cuda", dtype=torch.float16)
    residual = torch.empty_like(x)
    weight = torch.ones(64, device=x.device, dtype=x.dtype)
    assert torch.ops.trtllm.flashinfer_fused_add_rmsnorm(x, residual, weight, 1e-6) is None
    torch.cuda.synchronize()


def test_empty_rope():
    positions = torch.empty((0,), device="cuda", dtype=torch.int32)
    query = torch.empty((0, 128), device="cuda", dtype=torch.float16)
    key = torch.empty_like(query)
    cache = torch.zeros((16, 64), device="cuda", dtype=torch.float32)
    assert (
        torch.ops.trtllm.flashinfer_apply_rope_with_cos_sin_cache_inplace(
            positions, query, key, 64, cache, True
        )
        is None
    )
    torch.cuda.synchronize()
