# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Functional tests for sharded_rmsnorm custom op.

This module tests that the sharded_rmsnorm op produces correct
numerical results when run across multiple GPUs with column-sharded inputs.

The op computes RMSNorm on column-sharded activations by using all_reduce to
compute the global mean of squared values across all shards, ensuring correct
normalization equivalent to non-sharded global RMSNorm.
"""

from functools import partial

import pytest
import torch
import torch.distributed as dist

from tensorrt_llm._torch.auto_deploy.distributed.common import spawn_multiprocess_job


def _reference_rmsnorm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference RMSNorm implementation for comparison."""
    input_fp32 = input.to(torch.float32)
    variance = input_fp32.pow(2).mean(-1, keepdim=True)
    input_normalized = input_fp32 * torch.rsqrt(variance + eps)
    return weight * input_normalized.to(input.dtype)


def _run_sharded_rmsnorm_test(
    rank: int,
    world_size: int,
    batch_size: int = 2,
    seq_len: int = 8,
    hidden_size: int = 64,
    eps: float = 1e-6,
    dtype_str: str = "float16",
) -> None:
    """Test that sharded_rmsnorm matches non-sharded global RMSNorm.

    Each rank:
    1. Creates identical full input and weight tensors (same seed)
    2. Computes reference result using non-sharded RMSNorm
    3. Column-shards input and weight for this rank
    4. Calls sharded_rmsnorm on local shard
    5. All-gathers results and compares with reference
    """
    import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[dtype_str]

    torch.manual_seed(42)  # Same seed for reproducibility across ranks

    # Full tensors (same on all ranks due to seed)
    full_input = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
    full_weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

    # Reference: non-sharded global RMSNorm
    reference_output = _reference_rmsnorm(full_input, full_weight, eps)

    # Column shard: split hidden_size across ranks
    local_hidden_size = hidden_size // world_size
    start_idx = rank * local_hidden_size
    end_idx = start_idx + local_hidden_size

    local_input = full_input[..., start_idx:end_idx].contiguous()
    local_weight = full_weight[start_idx:end_idx].contiguous()

    local_output = torch.ops.auto_deploy.sharded_rmsnorm(local_input, local_weight, eps, world_size)

    # All-gather to reconstruct full output
    gathered_outputs = [torch.zeros_like(local_output) for _ in range(world_size)]
    dist.all_gather(gathered_outputs, local_output)
    reconstructed_output = torch.cat(gathered_outputs, dim=-1)

    # Compare with reference
    torch.testing.assert_close(
        reconstructed_output,
        reference_output,
        atol=1e-2,
        rtol=1e-2,
        msg=f"sharded_rmsnorm result doesn't match reference (rank={rank})",
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs for this test")
def test_sharded_rmsnorm_functional() -> None:
    """Functional test: verify sharded_rmsnorm produces correct numerical results."""
    spawn_multiprocess_job(job=_run_sharded_rmsnorm_test, size=2)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs for this test")
@pytest.mark.parametrize("hidden_size", [64, 128, 256])
def test_sharded_rmsnorm_different_hidden_sizes(hidden_size: int) -> None:
    """Test sharded_rmsnorm with different hidden sizes."""
    spawn_multiprocess_job(job=partial(_run_sharded_rmsnorm_test, hidden_size=hidden_size), size=2)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs for this test")
@pytest.mark.parametrize("dtype_str", ["float16", "bfloat16"])
def test_sharded_rmsnorm_different_dtypes(dtype_str: str) -> None:
    """Test sharded_rmsnorm with different dtypes."""
    spawn_multiprocess_job(job=partial(_run_sharded_rmsnorm_test, dtype_str=dtype_str), size=2)
