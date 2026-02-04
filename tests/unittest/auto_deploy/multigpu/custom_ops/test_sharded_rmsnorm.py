"""Functional tests for sharded_rmsnorm custom op.

This module tests that the sharded_rmsnorm op produces correct
numerical results when run across multiple GPUs with column-sharded inputs.

The op computes RMSNorm on column-sharded activations by using all_reduce to
compute the global mean of squared values across all shards, ensuring correct
normalization equivalent to non-sharded global RMSNorm.
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

# needed since we reuse the mpi executor pool, first test running will leak a thread
pytestmark = pytest.mark.threadleak(enabled=False)


def _reference_rmsnorm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference RMSNorm implementation for comparison."""
    input_fp32 = input.to(torch.float32)
    variance = input_fp32.pow(2).mean(-1, keepdim=True)
    input_normalized = input_fp32 * torch.rsqrt(variance + eps)
    return weight * input_normalized.to(input.dtype)


def _run_sharded_rmsnorm_test(
    tensor_parallel_size: int,
    batch_size: int = 2,
    seq_len: int = 8,
    hidden_size: int = 64,
    eps: float = 1e-6,
    dtype_str: str = "float16",
):
    """Test that sharded_rmsnorm matches non-sharded global RMSNorm.

    Each rank:
    1. Creates identical full input and weight tensors (same seed)
    2. Computes reference result using non-sharded RMSNorm
    3. Column-shards input and weight for this rank
    4. Calls sharded_rmsnorm on local shard
    5. All-gathers results and compares with reference
    """
    # Import inside worker to avoid cloudpickle serialization issues with torch.ops
    import tensorrt_llm
    import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401

    rank = tensorrt_llm.mpi_rank()
    world_size = tensor_parallel_size
    torch.cuda.set_device(rank)

    if not is_initialized():
        initialize(rank, port=29500)

    # Map string to torch.dtype inside worker to avoid cloudpickle serialization issues
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[dtype_str]

    try:
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

        # Call sharded_rmsnorm using dynamic getattr to avoid cloudpickle capturing torch.ops
        sharded_rmsnorm_op = getattr(getattr(torch, "ops"), "auto_deploy").sharded_rmsnorm
        local_output = sharded_rmsnorm_op(local_input, local_weight, eps, world_size)

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
    except Exception:
        traceback.print_exc()
        raise

    return True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs for this test")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_sharded_rmsnorm_functional(mpi_pool_executor):
    """Functional test: verify sharded_rmsnorm produces correct numerical results."""
    torch.manual_seed(0)
    tensor_parallel_size = mpi_pool_executor.num_workers

    results = mpi_pool_executor.map(
        _run_sharded_rmsnorm_test,
        *zip(*[(tensor_parallel_size,)] * tensor_parallel_size),
    )
    for r in results:
        assert r is True


def _run_sharded_rmsnorm_hidden_size_test(tensor_parallel_size: int, hidden_size: int):
    """Worker function for hidden size parametrized test."""
    return _run_sharded_rmsnorm_test(tensor_parallel_size, hidden_size=hidden_size)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs for this test")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.parametrize("hidden_size", [64, 128, 256])
def test_sharded_rmsnorm_different_hidden_sizes(mpi_pool_executor, hidden_size):
    """Test sharded_rmsnorm with different hidden sizes."""
    torch.manual_seed(0)
    tensor_parallel_size = mpi_pool_executor.num_workers

    results = mpi_pool_executor.map(
        _run_sharded_rmsnorm_hidden_size_test,
        *zip(*[(tensor_parallel_size, hidden_size)] * tensor_parallel_size),
    )
    for r in results:
        assert r is True


def _run_sharded_rmsnorm_dtype_test(tensor_parallel_size: int, dtype_str: str):
    """Worker function for dtype parametrized test."""
    return _run_sharded_rmsnorm_test(tensor_parallel_size, dtype_str=dtype_str)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs for this test")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.parametrize("dtype_str", ["float16", "bfloat16"])
def test_sharded_rmsnorm_different_dtypes(mpi_pool_executor, dtype_str):
    """Test sharded_rmsnorm with different dtypes."""
    torch.manual_seed(0)
    tensor_parallel_size = mpi_pool_executor.num_workers

    results = mpi_pool_executor.map(
        _run_sharded_rmsnorm_dtype_test,
        *zip(*[(tensor_parallel_size, dtype_str)] * tensor_parallel_size),
    )
    for r in results:
        assert r is True
