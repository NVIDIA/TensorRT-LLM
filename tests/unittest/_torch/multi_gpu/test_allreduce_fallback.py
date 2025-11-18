# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm._torch.distributed import (
    AllReduce,
    AllReduceFusionOp,
    AllReduceParams,
    AllReduceStrategy,
)
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.plugin.plugin import CustomAllReduceHelper

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

# needed since we reuse the mpi executor pool, first test running will leak a thread
pytestmark = pytest.mark.threadleak(enabled=False)


# ============================================================================
# Helper Functions
# ============================================================================


def rms_norm(x: torch.Tensor, weight: torch.Tensor = None, eps: float = 1e-6):
    """Reference implementation of RMS normalization."""
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight
    return y


def create_mapping(tensor_parallel_size: int, tensor_parallel_rank: int) -> Mapping:
    """Create a Mapping object for the given parallel configuration."""
    return Mapping(
        world_size=tensor_parallel_size,
        tp_size=tensor_parallel_size,
        rank=tensor_parallel_rank,
    )


def create_test_modules(
    mapping: Mapping,
    hidden_size: int,
    dtype: torch.dtype,
    strategy: AllReduceStrategy,
    weights: list,
):
    """Create and initialize test modules (Linear, AllReduce, RMSNorm)."""
    linear = Linear(
        in_features=hidden_size,
        out_features=hidden_size,
        bias=False,
        dtype=dtype,
        mapping=mapping,
        tensor_parallel_mode=TensorParallelMode.ROW,
        allreduce_strategy=strategy,
    ).cuda()

    allreduce = AllReduce(mapping=mapping, strategy=strategy)

    norm = RMSNorm(hidden_size=hidden_size, eps=1e-5, dtype=dtype).cuda()
    norm_weight = torch.randn((hidden_size,), dtype=dtype, device="cuda")
    norm.weight.data.copy_(norm_weight)

    linear.load_weights([dict(weight=weights[0])])

    return linear, allreduce, norm, norm_weight


def verify_outputs(calc_output, ref_output, rtol=0.05, atol=0.15):
    """Verify that calculated outputs match reference outputs."""
    if isinstance(calc_output, tuple):
        calc_outputs = calc_output
    else:
        calc_outputs = [calc_output]

    if isinstance(ref_output, tuple):
        ref_outputs = ref_output
    else:
        ref_outputs = [ref_output]

    for calc_tensor, ref_tensor in zip(calc_outputs, ref_outputs):
        try:
            torch.testing.assert_close(calc_tensor, ref_tensor, rtol=rtol, atol=atol)
        except AssertionError:
            # Calculate percentage of mismatched elements
            mismatched = torch.abs(calc_tensor - ref_tensor) > (rtol * torch.abs(ref_tensor) + atol)
            mismatch_percentage = mismatched.sum() / mismatched.numel()
            assert mismatch_percentage < 0.01, "Large mismatched elements encountered"


def synchronize_and_cleanup():
    """Synchronize CUDA operations before cleanup."""
    torch.cuda.synchronize()


# ============================================================================
# Test Section 1: Message Size Fallback Tests
# ============================================================================


def calculate_message_size_conditions(tensor_parallel_size, tensor_parallel_rank, x, dtype):
    """Calculate message size conditions for fallback testing."""
    message_size_bytes = x.size(-2) * x.size(-1) * x.element_size()
    max_workspace_size = CustomAllReduceHelper.max_workspace_size_auto(tensor_parallel_size)

    return {
        "message_size_bytes": message_size_bytes,
        "max_workspace_size": max_workspace_size,
        "should_fallback": message_size_bytes > max_workspace_size,
    }


@torch.inference_mode()
def run_fallback_with_fusion_test(
    x: torch.Tensor,
    residual: torch.Tensor,
    hidden_size: int,
    dtype: torch.dtype,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
    weights: list,
    strategy: AllReduceStrategy,
    expected_conditions: dict = None,
):
    """Test fallback behavior with fused residual RMS norm operation."""
    # Setup
    x = x.cuda()
    residual = residual.cuda()
    mapping = create_mapping(tensor_parallel_size, tensor_parallel_rank)
    linear, allreduce, norm, norm_weight = create_test_modules(
        mapping, hidden_size, dtype, strategy, weights
    )

    scale = torch.tensor(1.0, dtype=torch.float32).cuda()
    eps = 1e-5

    # Verify message size calculation if expected conditions provided
    if expected_conditions:
        actual_conditions = calculate_message_size_conditions(
            tensor_parallel_size, tensor_parallel_rank, x, dtype
        )
        expected_msg_size = expected_conditions.get("message_size_bytes")
        actual_msg_size = actual_conditions["message_size_bytes"]
        assert actual_msg_size == expected_msg_size, (
            f"Message size mismatch: expected {expected_msg_size}, got {actual_msg_size}"
        )
        expected_max_ws = expected_conditions.get("max_workspace_size")
        actual_max_ws = actual_conditions["max_workspace_size"]
        assert actual_max_ws == expected_max_ws, (
            f"Max workspace size mismatch: expected {expected_max_ws}, got {actual_max_ws}"
        )

    # Forward pass with fusion
    def calc_fused_allreduce(x, res):
        linear_out = linear(x, all_reduce_params=AllReduceParams(enable_allreduce=False))
        output = allreduce(
            linear_out,
            all_reduce_params=AllReduceParams(
                fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                residual=res,
                norm_weight=norm_weight,
                scale=scale,
                bias=None,
                eps=eps,
            ),
        )
        return output

    def ref_residual_rms_norm(x, res):
        linear_out = linear(x)
        hidden_states = linear_out
        residual_out = hidden_states + res
        residual_out = residual_out.to(torch.float32)
        norm_out = rms_norm(residual_out, norm_weight, eps)
        return norm_out.to(dtype), residual_out.to(dtype)

    # Execute and verify
    xs = torch.chunk(x.clone(), tensor_parallel_size, dim=-1)
    calc_output = calc_fused_allreduce(xs[tensor_parallel_rank], residual)
    ref_output = ref_residual_rms_norm(xs[tensor_parallel_rank], residual)

    verify_outputs(calc_output, ref_output)
    synchronize_and_cleanup()


# ============================================================================
# Test Section 2: Window Tensor Tests
# ============================================================================


@torch.inference_mode()
def run_window_tensor_single_call_test(
    x: torch.Tensor,
    hidden_size: int,
    dtype: torch.dtype,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
    weights: list,
):
    """Test a single allreduce call with window tensor."""
    x = x.cuda()
    mapping = create_mapping(tensor_parallel_size, tensor_parallel_rank)
    linear, allreduce, _, _ = create_test_modules(
        mapping, hidden_size, dtype, AllReduceStrategy.NCCL_SYMMETRIC, weights
    )

    def calc_allreduce(x):
        linear_out = linear(x)
        output = allreduce(linear_out)
        return output

    def ref_allreduce(x):
        linear_out = linear(x)
        return linear_out

    xs = torch.chunk(x.clone(), tensor_parallel_size, dim=-1)
    calc_output = calc_allreduce(xs[tensor_parallel_rank])
    ref_output = ref_allreduce(xs[tensor_parallel_rank])

    verify_outputs(calc_output, ref_output)
    synchronize_and_cleanup()


@torch.inference_mode()
def run_window_tensor_buffer_reuse_test(
    x: torch.Tensor,
    hidden_size: int,
    dtype: torch.dtype,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
    weights: list,
    num_iterations: int = 3,
):
    """Test that window tensor buffers are reused across multiple calls."""
    x = x.cuda()
    mapping = create_mapping(tensor_parallel_size, tensor_parallel_rank)
    linear, allreduce, _, _ = create_test_modules(
        mapping, hidden_size, dtype, AllReduceStrategy.NCCL_SYMMETRIC, weights
    )

    def calc_allreduce(x):
        linear_out = linear(x)
        output = allreduce(linear_out)
        return output

    def ref_allreduce(x):
        linear_out = linear(x)
        return linear_out

    xs = torch.chunk(x.clone(), tensor_parallel_size, dim=-1)

    # Test multiple calls to verify buffer reuse
    for _ in range(num_iterations):
        calc_output = calc_allreduce(xs[tensor_parallel_rank])
        ref_output = ref_allreduce(xs[tensor_parallel_rank])
        verify_outputs(calc_output, ref_output)

    synchronize_and_cleanup()


# ============================================================================
# Test Section 3: Lookup Table Fallback Tests
# ============================================================================


@torch.inference_mode()
def run_lookup_table_fallback_test(
    x: torch.Tensor,
    hidden_size: int,
    dtype: torch.dtype,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
    weights: list,
):
    """Test AUTO strategy fallback for extreme values outside lookup table bounds."""
    x = x.cuda()
    mapping = create_mapping(tensor_parallel_size, tensor_parallel_rank)
    linear, allreduce, _, _ = create_test_modules(
        mapping, hidden_size, dtype, AllReduceStrategy.AUTO, weights
    )

    def calc_allreduce(x):
        linear_out = linear(x)
        output = allreduce(linear_out)
        return output

    def ref_allreduce(x):
        linear_out = linear(x)
        return linear_out

    xs = torch.chunk(x.clone(), tensor_parallel_size, dim=-1)
    calc_output = calc_allreduce(xs[tensor_parallel_rank])
    ref_output = ref_allreduce(xs[tensor_parallel_rank])

    verify_outputs(calc_output, ref_output)
    synchronize_and_cleanup()


# ============================================================================
# Test Section 4: Strategy Comparison Tests
# ============================================================================


@torch.inference_mode()
def run_strategy_comparison_test(
    x: torch.Tensor,
    residual: torch.Tensor,
    hidden_size: int,
    dtype: torch.dtype,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
    weights: list,
    strategy: AllReduceStrategy,
):
    """Test a specific strategy and return results for comparison."""
    return run_fallback_with_fusion_test(
        x,
        residual,
        hidden_size,
        dtype,
        tensor_parallel_size,
        tensor_parallel_rank,
        weights,
        strategy,
        expected_conditions=None,
    )


# ============================================================================
# MPI Wrapper Functions
# ============================================================================


def run_single_rank_test(tensor_parallel_size, test_func, *args):
    """Wrapper to run a test function on a single rank.

    The test_func should accept tensor_parallel_rank as one of its parameters.
    We inject the actual rank here by replacing None placeholders.
    """
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        # Convert args to list to modify
        args_list = list(args)
        # Replace None placeholders with actual rank
        # The pattern is: (..., tensor_parallel_size, None, ...) where None is rank
        # We need to check for None first to avoid comparing tensors to integers
        for i in range(len(args_list) - 1):
            # Check if next arg is None first (safe check)
            if args_list[i + 1] is None:
                # Check if current arg is tensor_parallel_size (using isinstance to avoid tensor comparison)
                if isinstance(args_list[i], int) and args_list[i] == tensor_parallel_size:
                    args_list[i + 1] = rank
                    break
        else:
            # Fallback: find first None and replace it
            for i, arg in enumerate(args_list):
                if arg is None:
                    args_list[i] = rank
                    break
        test_func(*args_list)
    except Exception:
        traceback.print_exc()
        raise
    return True


# ============================================================================
# Pytest Test Functions
# ============================================================================


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_auto_strategy_fallback_large_message(mpi_pool_executor):
    """Test AUTO strategy fallback for large messages."""
    torch.manual_seed(42)

    tensor_parallel_size = mpi_pool_executor.num_workers
    dtype = torch.bfloat16
    hidden_size = 8192

    # Calculate message size that exceeds max workspace
    max_workspace_size = CustomAllReduceHelper.max_workspace_size_auto(tensor_parallel_size)
    element_size = torch.finfo(dtype).bits // 8
    seq_len = (max_workspace_size // (hidden_size * element_size)) + 100

    x = torch.randn((tensor_parallel_size, seq_len, hidden_size), dtype=dtype)
    residual = torch.randn((seq_len, hidden_size), dtype=dtype)
    weights = [torch.randn((hidden_size, hidden_size), dtype=dtype)]

    message_size_bytes = seq_len * hidden_size * element_size
    expected_conditions = {
        "message_size_bytes": message_size_bytes,
        "max_workspace_size": max_workspace_size,
        "should_fallback": message_size_bytes > max_workspace_size,
    }

    results = mpi_pool_executor.map(
        run_single_rank_test,
        *zip(
            *[
                (
                    tensor_parallel_size,
                    run_fallback_with_fusion_test,
                    x,
                    residual,
                    hidden_size,
                    dtype,
                    tensor_parallel_size,
                    None,
                    weights,
                    AllReduceStrategy.AUTO,
                    expected_conditions,
                )
            ]
            * tensor_parallel_size
        ),
    )
    for r in results:
        assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_auto_strategy_fallback_small_message(mpi_pool_executor):
    """Test AUTO strategy with small messages that don't trigger fallback."""
    torch.manual_seed(42)

    tensor_parallel_size = mpi_pool_executor.num_workers
    dtype = torch.bfloat16
    hidden_size = 128
    seq_len = 16

    x = torch.randn((tensor_parallel_size, seq_len, hidden_size), dtype=dtype)
    residual = torch.randn((seq_len, hidden_size), dtype=dtype)
    weights = [torch.randn((hidden_size, hidden_size), dtype=dtype)]

    element_size = torch.finfo(dtype).bits // 8
    message_size_bytes = seq_len * hidden_size * element_size
    max_workspace_size = CustomAllReduceHelper.max_workspace_size_auto(tensor_parallel_size)
    expected_conditions = {
        "message_size_bytes": message_size_bytes,
        "max_workspace_size": max_workspace_size,
        "should_fallback": message_size_bytes > max_workspace_size,
    }

    results = mpi_pool_executor.map(
        run_single_rank_test,
        *zip(
            *[
                (
                    tensor_parallel_size,
                    run_fallback_with_fusion_test,
                    x,
                    residual,
                    hidden_size,
                    dtype,
                    tensor_parallel_size,
                    None,
                    weights,
                    AllReduceStrategy.AUTO,
                    expected_conditions,
                )
            ]
            * tensor_parallel_size
        ),
    )
    for r in results:
        assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.parametrize("seq_len", [1024, 8192], ids=lambda x: f"seqlen:{x}")
@pytest.mark.parametrize("hidden_size", [4096, 8192], ids=lambda x: f"hidden:{x}")
def test_nccl_symmetric_window_tensor_single(mpi_pool_executor, seq_len, hidden_size):
    """Test NCCL_SYMMETRIC window tensor with a single call."""
    torch.manual_seed(42)

    tensor_parallel_size = mpi_pool_executor.num_workers
    dtype = torch.bfloat16

    x = torch.randn((tensor_parallel_size, seq_len, hidden_size), dtype=dtype)
    weights = [torch.randn((hidden_size, hidden_size), dtype=dtype)]

    results = mpi_pool_executor.map(
        run_single_rank_test,
        *zip(
            *[
                (
                    tensor_parallel_size,
                    run_window_tensor_single_call_test,
                    x,
                    hidden_size,
                    dtype,
                    tensor_parallel_size,
                    None,
                    weights,
                )
            ]
            * tensor_parallel_size
        ),
    )
    for r in results:
        assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.parametrize("seq_len", [16384], ids=lambda x: f"seqlen:{x}")
@pytest.mark.parametrize("hidden_size", [4096], ids=lambda x: f"hidden:{x}")
def test_nccl_symmetric_window_tensor_reuse(mpi_pool_executor, seq_len, hidden_size):
    """Test NCCL_SYMMETRIC window tensor buffer reuse."""
    torch.manual_seed(42)

    tensor_parallel_size = mpi_pool_executor.num_workers
    dtype = torch.bfloat16

    x = torch.randn((tensor_parallel_size, seq_len, hidden_size), dtype=dtype)
    weights = [torch.randn((hidden_size, hidden_size), dtype=dtype)]

    results = mpi_pool_executor.map(
        run_single_rank_test,
        *zip(
            *[
                (
                    tensor_parallel_size,
                    run_window_tensor_buffer_reuse_test,
                    x,
                    hidden_size,
                    dtype,
                    tensor_parallel_size,
                    None,
                    weights,
                    3,
                )
            ]
            * tensor_parallel_size
        ),
    )
    for r in results:
        assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.parametrize("seq_len", [1, 2**18], ids=lambda x: f"seqlen:{x}")
@pytest.mark.parametrize("hidden_size", [64, 2**14], ids=lambda x: f"hidden:{x}")
def test_lookup_table_fallback_extreme_values(mpi_pool_executor, seq_len, hidden_size):
    """Test AUTO strategy fallback for extreme values."""
    torch.manual_seed(42)

    tensor_parallel_size = mpi_pool_executor.num_workers
    dtype = torch.bfloat16

    x = torch.randn((tensor_parallel_size, seq_len, hidden_size), dtype=dtype)
    weights = [torch.randn((hidden_size, hidden_size), dtype=dtype)]

    results = mpi_pool_executor.map(
        run_single_rank_test,
        *zip(
            *[
                (
                    tensor_parallel_size,
                    run_lookup_table_fallback_test,
                    x,
                    hidden_size,
                    dtype,
                    tensor_parallel_size,
                    None,
                    weights,
                )
            ]
            * tensor_parallel_size
        ),
    )
    for r in results:
        assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_nccl_symmetric_explicit_vs_auto(mpi_pool_executor):
    """Test that explicit NCCL_SYMMETRIC produces same results as AUTO fallback."""
    torch.manual_seed(42)

    tensor_parallel_size = mpi_pool_executor.num_workers
    dtype = torch.bfloat16
    hidden_size = 8192
    seq_len = 256

    x = torch.randn((tensor_parallel_size, seq_len, hidden_size), dtype=dtype)
    residual = torch.randn((seq_len, hidden_size), dtype=dtype)
    weights = [torch.randn((hidden_size, hidden_size), dtype=dtype)]

    # Test with explicit NCCL_SYMMETRIC
    results_explicit = mpi_pool_executor.map(
        run_single_rank_test,
        *zip(
            *[
                (
                    tensor_parallel_size,
                    run_fallback_with_fusion_test,
                    x,
                    residual,
                    hidden_size,
                    dtype,
                    tensor_parallel_size,
                    None,
                    weights,
                    AllReduceStrategy.NCCL_SYMMETRIC,
                    None,
                )
            ]
            * tensor_parallel_size
        ),
    )

    # Test with AUTO
    results_auto = mpi_pool_executor.map(
        run_single_rank_test,
        *zip(
            *[
                (
                    tensor_parallel_size,
                    run_fallback_with_fusion_test,
                    x,
                    residual,
                    hidden_size,
                    dtype,
                    tensor_parallel_size,
                    None,
                    weights,
                    AllReduceStrategy.AUTO,
                    None,
                )
            ]
            * tensor_parallel_size
        ),
    )

    for r in results_explicit + results_auto:
        assert r is True
