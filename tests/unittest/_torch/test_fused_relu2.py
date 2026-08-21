# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``relu2`` must stay bit-identical to ``torch.square(F.relu(x))``.

The fused kernel is only a memory-traffic optimisation, so every dispatch
decision it makes has to be invisible in the result. That means checking both
that the kernel matches eager exactly on the shapes it claims, and that the
ineligible cases actually fall back rather than producing something close.
"""

import os
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.utils import _fused_relu2_impl, relu2


def eager_relu2(x: torch.Tensor) -> torch.Tensor:
    return torch.square(F.relu(x))


@pytest.fixture(autouse=True)
def _clear_impl_cache():
    """``_fused_relu2_impl`` memoises the env lookup, so reset it per test."""
    _fused_relu2_impl.cache_clear()
    yield
    _fused_relu2_impl.cache_clear()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (4095,),  # not a multiple of the kernel's block
        (4096,),
        (4097,),
        (32768, 3712),  # Nemotron-3.5's shared expert at ISL 32k
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_matches_eager_bitwise(shape, dtype):
    x = torch.randn(*shape, device="cuda", dtype=torch.float32).to(dtype)
    # randn alone barely reaches the negative saturation region
    x[..., ::7] *= -20.0
    torch.testing.assert_close(relu2(x), eager_relu2(x), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_negative_and_zero_inputs():
    """relu2 is flat at zero on the whole negative half, including -0.0."""
    x = torch.tensor(
        [-1e30, -1.0, -1e-30, -0.0, 0.0, 1e-30, 1.0], device="cuda", dtype=torch.float32
    )
    got = relu2(x)
    torch.testing.assert_close(got, eager_relu2(x), rtol=0, atol=0)
    assert not got[:5].any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_non_contiguous_falls_back():
    base = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    x = base.t()
    assert not x.is_contiguous()
    torch.testing.assert_close(relu2(x), eager_relu2(x), rtol=0, atol=0)


def test_cpu_falls_back():
    x = torch.randn(1024, dtype=torch.float32)
    torch.testing.assert_close(relu2(x), eager_relu2(x), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_unsupported_dtype_falls_back():
    x = torch.randn(1024, device="cuda", dtype=torch.float64)
    torch.testing.assert_close(relu2(x), eager_relu2(x), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_empty_tensor():
    x = torch.empty(0, device="cuda", dtype=torch.bfloat16)
    assert relu2(x).shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_env_var_disables_fusion():
    x = torch.randn(8192, device="cuda", dtype=torch.bfloat16)
    expected = eager_relu2(x)
    with mock.patch.dict(os.environ, {"TRTLLM_FUSED_RELU2": "0"}):
        _fused_relu2_impl.cache_clear()
        assert _fused_relu2_impl() is None
        torch.testing.assert_close(relu2(x), expected, rtol=0, atol=0)
