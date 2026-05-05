# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Functional tests for lamport_sharded_qk_rmsnorm custom op.

This module tests that the lamport_sharded_qk_rmsnorm op produces correct
numerical results when run across multiple GPUs with column-sharded Q and K inputs.

The op computes RMSNorm on column-sharded Q and K activations using a single
NVLink barrier (or NCCL fallback) on packed [N_tokens, 2] variance scalars,
which is mathematically equivalent to running non-sharded global RMSNorm on
the full Q and K tensors.
"""

import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
import torch.distributed as dist
from mpi4py import MPI

from tensorrt_llm._torch.auto_deploy.distributed.common import initialize, is_initialized

# Register this module for cloudpickle serialization for MPI workers
cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

# Reuse the MPI executor pool across tests (first test running will leak a thread)
pytestmark = pytest.mark.threadleak(enabled=False)


def _reference_rmsnorm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference (non-sharded) RMSNorm implementation."""
    input_fp32 = input.to(torch.float32)
    variance = input_fp32.pow(2).mean(-1, keepdim=True)
    input_normalized = input_fp32 * torch.rsqrt(variance + eps)
    return weight * input_normalized.to(input.dtype)


def _run_lamport_sharded_qk_rmsnorm_test(
    tensor_parallel_size: int,
    batch_size: int = 2,
    seq_len: int = 8,
    q_hidden: int = 64,
    k_hidden: int = 64,
    eps: float = 1e-6,
    dtype_str: str = "float16",
):
    """Test that lamport_sharded_qk_rmsnorm matches non-sharded global RMSNorm.

    Each rank:
    1. Creates identical full Q, K, weight_q, weight_k tensors (same seed)
    2. Computes reference results using non-sharded RMSNorm on the full tensors
    3. Column-shards Q, K, weight_q, weight_k for this rank
    4. Calls lamport_sharded_qk_rmsnorm on local shards
    5. All-gathers outputs and compares with reference
    """
    # Import inside worker to avoid cloudpickle serialization issues with torch.ops
    import tensorrt_llm
    import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401

    rank = tensorrt_llm.mpi_rank()
    world_size = tensor_parallel_size
    torch.cuda.set_device(rank)

    if not is_initialized():
        initialize(rank, port=29500)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[dtype_str]

    try:
        torch.manual_seed(42)  # Same seed for reproducibility across ranks

        # Full tensors (same on all ranks due to fixed seed).
        # Weights are created in the activation dtype so that the reference
        # _reference_rmsnorm and the op produce results in the same dtype.
        full_q = torch.randn(batch_size, seq_len, q_hidden, device="cuda", dtype=dtype)
        full_k = torch.randn(batch_size, seq_len, k_hidden, device="cuda", dtype=dtype)
        full_weight_q = torch.randn(q_hidden, device="cuda", dtype=dtype)
        full_weight_k = torch.randn(k_hidden, device="cuda", dtype=dtype)

        # Reference: non-sharded global RMSNorm on full tensors
        ref_q_out = _reference_rmsnorm(full_q, full_weight_q, eps)
        ref_k_out = _reference_rmsnorm(full_k, full_weight_k, eps)

        # Column shard: split last dim across ranks
        local_q_dim = q_hidden // world_size
        local_k_dim = k_hidden // world_size
        q_start = rank * local_q_dim
        k_start = rank * local_k_dim

        local_q = full_q[..., q_start : q_start + local_q_dim].contiguous()
        local_k = full_k[..., k_start : k_start + local_k_dim].contiguous()
        # The op accepts float32 weights internally but the test passes the activation dtype;
        # the Python fallback and CUDA kernel both cast to float32 as needed.
        local_weight_q = full_weight_q[q_start : q_start + local_q_dim].contiguous()
        local_weight_k = full_weight_k[k_start : k_start + local_k_dim].contiguous()

        # Call lamport_sharded_qk_rmsnorm via dynamic getattr to avoid cloudpickle capturing torch.ops
        op = getattr(getattr(torch, "ops"), "auto_deploy").lamport_sharded_qk_rmsnorm
        local_q_out, local_k_out = op(
            local_q, local_k, local_weight_q, local_weight_k, eps, world_size, rank
        )

        # All-gather Q outputs across ranks and compare
        gathered_q = [torch.zeros_like(local_q_out) for _ in range(world_size)]
        dist.all_gather(gathered_q, local_q_out)
        reconstructed_q = torch.cat(gathered_q, dim=-1)

        torch.testing.assert_close(
            reconstructed_q,
            ref_q_out,
            atol=1e-2,
            rtol=1e-2,
            msg=f"lamport_sharded_qk_rmsnorm Q output doesn't match reference (rank={rank})",
        )

        # All-gather K outputs across ranks and compare
        gathered_k = [torch.zeros_like(local_k_out) for _ in range(world_size)]
        dist.all_gather(gathered_k, local_k_out)
        reconstructed_k = torch.cat(gathered_k, dim=-1)

        torch.testing.assert_close(
            reconstructed_k,
            ref_k_out,
            atol=1e-2,
            rtol=1e-2,
            msg=f"lamport_sharded_qk_rmsnorm K output doesn't match reference (rank={rank})",
        )

    except Exception:
        traceback.print_exc()
        raise

    return True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs for this test")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_lamport_sharded_qk_rmsnorm_functional(mpi_pool_executor):
    """Functional test: verify lamport_sharded_qk_rmsnorm produces correct numerical results."""
    torch.manual_seed(0)
    tensor_parallel_size = mpi_pool_executor.num_workers

    results = mpi_pool_executor.map(
        _run_lamport_sharded_qk_rmsnorm_test,
        *zip(*[(tensor_parallel_size,)] * tensor_parallel_size),
    )
    for r in results:
        assert r is True


def _run_lamport_sharded_qk_rmsnorm_dtype_test(tensor_parallel_size: int, dtype_str: str):
    """Worker function for dtype parametrized test."""
    return _run_lamport_sharded_qk_rmsnorm_test(tensor_parallel_size, dtype_str=dtype_str)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs for this test")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.parametrize("dtype_str", ["float16", "bfloat16"])
def test_lamport_sharded_qk_rmsnorm_dtypes(mpi_pool_executor, dtype_str):
    """Test lamport_sharded_qk_rmsnorm with float16 and bfloat16."""
    torch.manual_seed(0)
    tensor_parallel_size = mpi_pool_executor.num_workers

    results = mpi_pool_executor.map(
        _run_lamport_sharded_qk_rmsnorm_dtype_test,
        *zip(*[(tensor_parallel_size, dtype_str)] * tensor_parallel_size),
    )
    for r in results:
        assert r is True


def _run_lamport_sharded_qk_rmsnorm_dim_test(
    tensor_parallel_size: int, q_hidden: int, k_hidden: int
):
    """Worker function for asymmetric Q/K dim test."""
    return _run_lamport_sharded_qk_rmsnorm_test(
        tensor_parallel_size, q_hidden=q_hidden, k_hidden=k_hidden
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs for this test")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.parametrize("q_hidden,k_hidden", [(64, 64), (128, 64)])
def test_lamport_sharded_qk_rmsnorm_dims(mpi_pool_executor, q_hidden, k_hidden):
    """Test lamport_sharded_qk_rmsnorm with symmetric and asymmetric Q/K dimensions."""
    torch.manual_seed(0)
    tensor_parallel_size = mpi_pool_executor.num_workers

    results = mpi_pool_executor.map(
        _run_lamport_sharded_qk_rmsnorm_dim_test,
        *zip(*[(tensor_parallel_size, q_hidden, k_hidden)] * tensor_parallel_size),
    )
    for r in results:
        assert r is True
