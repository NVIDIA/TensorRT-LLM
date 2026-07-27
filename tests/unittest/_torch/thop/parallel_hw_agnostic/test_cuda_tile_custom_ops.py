# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import pytest
import torch
import torch.nn.functional as F

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.cuda_tile_utils import IS_CUDA_TILE_AVAILABLE

# Skip all tests if CUDA tile is not available
pytestmark = pytest.mark.skipif(not IS_CUDA_TILE_AVAILABLE, reason="CUDA tile is not available")


@pytest.fixture(autouse=True)
def prepare_testcase_environment(tmp_path):
    """Set random seed and enable deterministic mode before each test."""
    # Enable deterministic execution.
    prev_cublas_workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    prev_deterministic_mode = torch.are_deterministic_algorithms_enabled()
    torch.manual_seed(19260817)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    # Enable cuTile debug dump.
    os.environ["CUDA_TILE_DUMP_BYTECODE"] = str(tmp_path / "bytecode")
    os.environ["CUDA_TILE_DUMP_TILEIR"] = str(tmp_path / "tileir")

    yield

    # Rewind to previous states.
    if prev_cublas_workspace_config is not None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = prev_cublas_workspace_config
    else:
        del os.environ["CUBLAS_WORKSPACE_CONFIG"]
    torch.use_deterministic_algorithms(prev_deterministic_mode)
    del os.environ["CUDA_TILE_DUMP_BYTECODE"]
    del os.environ["CUDA_TILE_DUMP_TILEIR"]


def reference_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    use_gemma: bool,
    residual: torch.Tensor | None = None,
):
    """
    Reference RMSNorm implementation using PyTorch operations.

    Args:
        hidden_states: Input tensor
        weight: Weight tensor
        eps: Epsilon for numerical stability
        use_gemma: Whether to use Gemma-style weight bias (weight + 1)
        residual: Optional residual tensor to add before normalization

    Returns:
        Tuple of (normalized output, new residual) if residual is provided,
        otherwise just normalized output
    """
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    new_residual = None
    if residual is not None:
        hidden_states = hidden_states + residual.to(torch.float32)
        new_residual = hidden_states.to(input_dtype)

    # Prepare weight with Gemma-style bias if needed
    if use_gemma:
        weight_to_apply = weight + 1.0
    else:
        weight_to_apply = weight

    # Use torch.nn.functional.rms_norm for the normalization
    hidden_states = F.rms_norm(
        hidden_states, (hidden_states.shape[-1],), weight=weight_to_apply.to(torch.float32), eps=eps
    )
    hidden_states = hidden_states.to(input_dtype)

    if residual is not None:
        return hidden_states, new_residual
    else:
        return hidden_states


@pytest.mark.parametrize(
    "M,N",
    [
        (1, 128),
        (4, 256),
        (16, 512),
        (32, 1024),
        (64, 2048),
        (128, 4096),
        (8, 8192),
    ],
)
@pytest.mark.parametrize("use_gemma", [False, True])
@pytest.mark.parametrize("static_persistent", [False, True])
@pytest.mark.parametrize("gather", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_tile_rms_norm(M, N, use_gemma, static_persistent, gather, dtype):
    """Test cuda_tile_rms_norm operator against reference implementation."""
    eps = 1e-5

    # Create input tensors
    x = torch.randn(M, N, dtype=dtype, device="cuda")
    weight = torch.randn(N, dtype=dtype, device="cuda")

    # Clone for reference computation
    x_ref = x.clone()
    weight_ref = weight.clone()

    # Compute reference
    ref_output = reference_rms_norm(x_ref, weight_ref, eps, use_gemma)

    # Compute with cuda_tile kernel
    cuda_output = torch.ops.trtllm.cuda_tile_rms_norm(
        x=x,
        weight=weight,
        eps=eps,
        static_persistent=static_persistent,
        gather=gather,
        use_gemma=use_gemma,
    )

    # Compare results
    # Use relatively loose tolerance due to different computation orders
    rtol = 1e-2 if dtype == torch.float16 else 5e-2
    atol = 1e-3 if dtype == torch.float16 else 5e-3

    torch.testing.assert_close(
        cuda_output,
        ref_output,
        rtol=rtol,
        atol=atol,
        msg=f"cuda_tile_rms_norm output mismatch for M={M}, N={N}, "
        f"use_gemma={use_gemma}, static_persistent={static_persistent}, "
        f"gather={gather}, dtype={dtype}",
    )


@pytest.mark.parametrize(
    "M,N",
    [
        (1, 128),
        (4, 256),
        (16, 512),
        (32, 1024),
        (64, 2048),
        (128, 4096),
        (8, 8192),
    ],
)
@pytest.mark.parametrize("use_gemma", [False, True])
@pytest.mark.parametrize("static_persistent", [False, True])
@pytest.mark.parametrize("gather", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_tile_rms_norm_fuse_residual(M, N, use_gemma, static_persistent, gather, dtype):
    """Test cuda_tile_rms_norm_fuse_residual_ operator against reference implementation."""
    eps = 1e-5

    # Create input tensors
    x = torch.randn(M, N, dtype=dtype, device="cuda")
    residual = torch.randn(M, N, dtype=dtype, device="cuda")
    weight = torch.randn(N, dtype=dtype, device="cuda")

    # Clone for reference computation
    x_ref = x.clone()
    residual_ref = residual.clone()
    weight_ref = weight.clone()

    # Compute reference
    ref_output, ref_new_residual = reference_rms_norm(
        x_ref, weight_ref, eps, use_gemma, residual_ref
    )

    # Ensure tensors are contiguous for in-place operation
    x = x.contiguous()
    residual = residual.contiguous()

    # Compute with cuda_tile kernel (in-place operation)
    torch.ops.trtllm.cuda_tile_rms_norm_fuse_residual_(
        x=x,
        residual=residual,
        weight=weight,
        eps=eps,
        static_persistent=static_persistent,
        gather=gather,
        use_gemma=use_gemma,
    )

    # After in-place operation:
    # x contains the normalized output
    # residual contains the un-normalized sum (new residual)

    # Compare results
    # Use relatively loose tolerance due to different computation orders
    rtol = 1e-2 if dtype == torch.float16 else 5e-2
    atol = 1e-3 if dtype == torch.float16 else 5e-3

    torch.testing.assert_close(
        x,
        ref_output,
        rtol=rtol,
        atol=atol,
        msg=f"cuda_tile_rms_norm_fuse_residual_ output mismatch for M={M}, N={N}, "
        f"use_gemma={use_gemma}, static_persistent={static_persistent}, "
        f"gather={gather}, dtype={dtype}",
    )

    torch.testing.assert_close(
        residual,
        ref_new_residual,
        rtol=rtol,
        atol=atol,
        msg=f"cuda_tile_rms_norm_fuse_residual_ residual mismatch for M={M}, N={N}, "
        f"use_gemma={use_gemma}, static_persistent={static_persistent}, "
        f"gather={gather}, dtype={dtype}",
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_tile_rms_norm_fuse_residual_inplace(dtype):
    """Test that fuse_residual operator truly modifies tensors in-place."""
    eps = 1e-5
    M, N = 16, 256

    x = torch.randn(M, N, dtype=dtype, device="cuda").contiguous()
    residual = torch.randn(M, N, dtype=dtype, device="cuda").contiguous()
    weight = torch.randn(N, dtype=dtype, device="cuda")

    # Store original data pointers
    x_data_ptr = x.data_ptr()
    residual_data_ptr = residual.data_ptr()

    # Call in-place operator
    torch.ops.trtllm.cuda_tile_rms_norm_fuse_residual_(
        x=x,
        residual=residual,
        weight=weight,
        eps=eps,
        static_persistent=True,
        gather=True,
        use_gemma=False,
    )

    # Verify that tensors were modified in-place (same memory location)
    assert x.data_ptr() == x_data_ptr, "x tensor was not modified in-place"
    assert residual.data_ptr() == residual_data_ptr, "residual tensor was not modified in-place"


def test_cuda_tile_rms_norm_fuse_residual_requires_contiguous():
    """Test that fuse_residual operator requires contiguous tensors."""
    eps = 1e-5
    M, N = 16, 256
    dtype = torch.float16

    # Create non-contiguous tensors
    x = torch.randn(M, N * 2, dtype=dtype, device="cuda")[:, ::2]
    residual = torch.randn(M, N, dtype=dtype, device="cuda").contiguous()
    weight = torch.randn(N, dtype=dtype, device="cuda")

    assert not x.is_contiguous(), "x should be non-contiguous for this test"

    # Should raise assertion error for non-contiguous x
    with pytest.raises(AssertionError, match="x must be contiguous"):
        torch.ops.trtllm.cuda_tile_rms_norm_fuse_residual_(
            x=x,
            residual=residual,
            weight=weight,
            eps=eps,
            static_persistent=True,
            gather=True,
            use_gemma=False,
        )

    # Create non-contiguous residual
    x = torch.randn(M, N, dtype=dtype, device="cuda").contiguous()
    residual = torch.randn(M, N * 2, dtype=dtype, device="cuda")[:, ::2]

    assert not residual.is_contiguous(), "residual should be non-contiguous for this test"

    # Should raise assertion error for non-contiguous residual
    with pytest.raises(AssertionError, match="residual must be contiguous"):
        torch.ops.trtllm.cuda_tile_rms_norm_fuse_residual_(
            x=x,
            residual=residual,
            weight=weight,
            eps=eps,
            static_persistent=True,
            gather=True,
            use_gemma=False,
        )
