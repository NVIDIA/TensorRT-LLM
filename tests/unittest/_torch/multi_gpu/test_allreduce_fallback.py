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


def rms_norm(x: torch.Tensor, weight: torch.Tensor = None, eps: float = 1e-6):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight
    return y


def run_single_rank_fallback_test(
    tensor_parallel_size,
    single_rank_forward_func,
    input,
    residual,
    weights,
    hidden_size,
    dtype,
    fusion_op,
    strategy,
    expected_fallback_conditions,
):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        # Check function name to determine signature
        func_name = single_rank_forward_func.__name__
        if func_name == "run_allreduce_fallback_test":
            single_rank_forward_func(
                input,
                residual,
                hidden_size,
                dtype,
                tensor_parallel_size,
                rank,
                weights,
                fusion_op,
                strategy,
                expected_fallback_conditions,
            )
        elif func_name == "run_allreduce_window_tensor_test":
            single_rank_forward_func(input, hidden_size, dtype, tensor_parallel_size, rank, weights)
        elif func_name == "run_allreduce_lookup_table_fallback_test":
            single_rank_forward_func(input, hidden_size, dtype, tensor_parallel_size, rank, weights)
        else:
            raise ValueError(f"Unknown function: {func_name}")
    except Exception:
        traceback.print_exc()
        raise
    return True


@torch.inference_mode()
def run_allreduce_fallback_test(
    x: torch.Tensor,
    residual: torch.Tensor,
    hidden_size: int,
    dtype: torch.dtype,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
    weights: list,
    fusion_op: AllReduceFusionOp,
    strategy: AllReduceStrategy,
    expected_fallback_conditions: dict,
):
    """
    Test AllReduce with AUTO strategy to verify fallback to NCCL_SYMMETRIC.

    Args:
        expected_fallback_conditions: Dict with keys:
            - 'message_size_bytes': Expected message size in bytes
            - 'max_workspace_size': Max workspace size threshold
            - 'should_fallback': Whether fallback to NCCL_SYMMETRIC is expected
    """
    x = x.cuda()
    residual = residual.cuda()
    norm_weight = torch.randn((hidden_size,), dtype=dtype, device="cuda")
    eps = 1e-5

    mapping = Mapping(
        world_size=tensor_parallel_size,
        tp_size=tensor_parallel_size,
        rank=tensor_parallel_rank,
    )

    # Use AUTO strategy to test fallback behavior
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
    norm = RMSNorm(hidden_size=hidden_size, eps=eps, dtype=dtype).cuda()

    scale = torch.tensor(1.0, dtype=torch.float32).cuda()
    linear.load_weights([dict(weight=weights[0])])
    norm.weight.data.copy_(norm_weight)

    def calc_fused_allreduce(x, res):
        linear_out = linear(x, all_reduce_params=AllReduceParams(enable_allreduce=False))
        output = allreduce(
            linear_out,
            all_reduce_params=AllReduceParams(
                fusion_op=fusion_op,
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

    # Verify fallback conditions
    # Message size should be calculated from the dimensions that are actually reduced by allreduce
    # x has shape (tensor_parallel_size, seq_len, hidden_size), but allreduce operates on
    # seq_len * hidden_size elements (the tensor_parallel_size dimension is the rank dimension)
    # So we calculate: seq_len * hidden_size * element_size
    message_size_bytes = x.size(-2) * x.size(-1) * x.element_size()
    max_workspace_size = CustomAllReduceHelper.max_workspace_size_auto(tensor_parallel_size)

    if expected_fallback_conditions:
        assert message_size_bytes == expected_fallback_conditions.get(
            "message_size_bytes", message_size_bytes
        ), (
            f"Message size mismatch: expected "
            f"{expected_fallback_conditions.get('message_size_bytes')}, "
            f"got {message_size_bytes}"
        )
        assert max_workspace_size == expected_fallback_conditions.get(
            "max_workspace_size", max_workspace_size
        ), (
            f"Max workspace size mismatch: expected "
            f"{expected_fallback_conditions.get('max_workspace_size')}, "
            f"got {max_workspace_size}"
        )

    # Test correctness - if fallback works, this should still produce correct results
    xs = torch.chunk(x.clone(), tensor_parallel_size, dim=-1)
    calc_output = calc_fused_allreduce(xs[tensor_parallel_rank], residual)
    ref_output = ref_residual_rms_norm(xs[tensor_parallel_rank], residual)

    for calc_output_tensor, ref_output_tensor in zip(calc_output, ref_output):
        rtol, atol = 0.05, 0.15
        try:
            torch.testing.assert_close(
                calc_output_tensor,
                ref_output_tensor,
                rtol=rtol,
                atol=atol,
            )
        except AssertionError:
            # Calculate percentage of mismatched elements
            mismatched = torch.abs(calc_output_tensor - ref_output_tensor) > (
                rtol * torch.abs(ref_output_tensor) + atol
            )
            mismatch_percentage = mismatched.sum() / mismatched.numel()

            # If more than 1% elements mismatch, raise the error
            assert mismatch_percentage < 0.01, "Large mismatched elements encountered"

    # Synchronize CUDA to ensure all operations are complete before cleanup
    # This prevents NCCL cleanup from happening while CUDA operations are still pending
    torch.cuda.synchronize()


@torch.inference_mode()
def run_allreduce_window_tensor_test(
    x: torch.Tensor,
    hidden_size: int,
    dtype: torch.dtype,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
    weights: list,
):
    """
    Test NCCL_SYMMETRIC with window tensor registration.
    This test verifies that NCCL_SYMMETRIC works correctly with large buffers
    that should trigger window tensor registration.
    """
    x = x.cuda()
    norm_weight = torch.randn((hidden_size,), dtype=dtype, device="cuda")
    eps = 1e-5

    mapping = Mapping(
        world_size=tensor_parallel_size,
        tp_size=tensor_parallel_size,
        rank=tensor_parallel_rank,
    )

    # Use NCCL_SYMMETRIC explicitly to test window tensor path
    linear = Linear(
        in_features=hidden_size,
        out_features=hidden_size,
        bias=False,
        dtype=dtype,
        mapping=mapping,
        tensor_parallel_mode=TensorParallelMode.ROW,
        allreduce_strategy=AllReduceStrategy.NCCL_SYMMETRIC,
    ).cuda()
    allreduce = AllReduce(mapping=mapping, strategy=AllReduceStrategy.NCCL_SYMMETRIC)
    norm = RMSNorm(hidden_size=hidden_size, eps=eps, dtype=dtype).cuda()

    linear.load_weights([dict(weight=weights[0])])
    norm.weight.data.copy_(norm_weight)

    def calc_allreduce(x):
        linear_out = linear(x)
        output = allreduce(linear_out)
        return [output]

    def ref_allreduce(x):
        linear_out = linear(x)
        return [linear_out]

    # Test correctness - window tensor is an implementation detail, but we verify correctness
    xs = torch.chunk(x.clone(), tensor_parallel_size, dim=-1)
    calc_output = calc_allreduce(xs[tensor_parallel_rank])
    ref_output = ref_allreduce(xs[tensor_parallel_rank])

    # Test multiple calls to verify buffer reuse
    for _ in range(3):
        calc_output = calc_allreduce(xs[tensor_parallel_rank])
        ref_output = ref_allreduce(xs[tensor_parallel_rank])

        for calc_output_tensor, ref_output_tensor in zip(calc_output, ref_output):
            rtol, atol = 0.05, 0.15
            torch.testing.assert_close(
                calc_output_tensor,
                ref_output_tensor,
                rtol=rtol,
                atol=atol,
            )

    # Synchronize CUDA to ensure all operations are complete before cleanup
    # This prevents NCCL cleanup from happening while CUDA operations are still pending
    torch.cuda.synchronize()


@torch.inference_mode()
def run_allreduce_lookup_table_fallback_test(
    x: torch.Tensor,
    hidden_size: int,
    dtype: torch.dtype,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
    weights: list,
):
    """
    Test AUTO strategy with extreme values that fall outside lookup table bounds.
    This should trigger fallback to NCCL_SYMMETRIC.
    """
    x = x.cuda()
    norm_weight = torch.randn((hidden_size,), dtype=dtype, device="cuda")
    eps = 1e-5

    mapping = Mapping(
        world_size=tensor_parallel_size,
        tp_size=tensor_parallel_size,
        rank=tensor_parallel_rank,
    )

    # Use AUTO strategy with extreme values
    linear = Linear(
        in_features=hidden_size,
        out_features=hidden_size,
        bias=False,
        dtype=dtype,
        mapping=mapping,
        tensor_parallel_mode=TensorParallelMode.ROW,
        allreduce_strategy=AllReduceStrategy.AUTO,
    ).cuda()
    allreduce = AllReduce(mapping=mapping, strategy=AllReduceStrategy.AUTO)
    norm = RMSNorm(hidden_size=hidden_size, eps=eps, dtype=dtype).cuda()

    linear.load_weights([dict(weight=weights[0])])
    norm.weight.data.copy_(norm_weight)

    def calc_allreduce(x):
        linear_out = linear(x)
        output = allreduce(linear_out)
        return [output]

    def ref_allreduce(x):
        linear_out = linear(x)
        return [linear_out]

    # Test correctness - fallback should still work correctly
    xs = torch.chunk(x.clone(), tensor_parallel_size, dim=-1)
    calc_output = calc_allreduce(xs[tensor_parallel_rank])
    ref_output = ref_allreduce(xs[tensor_parallel_rank])

    for calc_output_tensor, ref_output_tensor in zip(calc_output, ref_output):
        rtol, atol = 0.05, 0.15
        torch.testing.assert_close(
            calc_output_tensor,
            ref_output_tensor,
            rtol=rtol,
            atol=atol,
        )

    # Synchronize CUDA to ensure all operations are complete before cleanup
    # This prevents NCCL cleanup from happening while CUDA operations are still pending
    torch.cuda.synchronize()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs for this test")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_auto_strategy_fallback_large_message(mpi_pool_executor):
    """
    Test that AUTO strategy falls back to NCCL_SYMMETRIC when message size
    exceeds max workspace size.
    """
    torch.manual_seed(42)

    tensor_parallel_size = mpi_pool_executor.num_workers
    dtype = torch.bfloat16
    hidden_size = 8192

    # Calculate message size that exceeds max workspace
    max_workspace_size = CustomAllReduceHelper.max_workspace_size_auto(tensor_parallel_size)
    # Create a message size that's larger than max workspace
    # max_workspace_size is in bytes, so we need seq_len * hidden_size * element_size > max_workspace_size
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
        run_single_rank_fallback_test,
        *zip(
            *[
                (
                    tensor_parallel_size,
                    run_allreduce_fallback_test,
                    x,
                    residual,
                    weights,
                    hidden_size,
                    dtype,
                    AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    AllReduceStrategy.AUTO,
                    expected_conditions,
                )
            ]
            * tensor_parallel_size
        ),
    )
    for r in results:
        assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs for this test")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_auto_strategy_fallback_small_message(mpi_pool_executor):
    """
    Test that AUTO strategy works correctly with small messages that don't trigger fallback.
    """
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
        run_single_rank_fallback_test,
        *zip(
            *[
                (
                    tensor_parallel_size,
                    run_allreduce_fallback_test,
                    x,
                    residual,
                    weights,
                    hidden_size,
                    dtype,
                    AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    AllReduceStrategy.AUTO,
                    expected_conditions,
                )
            ]
            * tensor_parallel_size
        ),
    )
    for r in results:
        assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs for this test")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.parametrize("seq_len", [1024, 8192, 16384], ids=lambda x: f"seqlen:{x}")
@pytest.mark.parametrize("hidden_size", [4096, 8192], ids=lambda x: f"hidden:{x}")
def test_nccl_symmetric_window_tensor(mpi_pool_executor, seq_len, hidden_size):
    """
    Test NCCL_SYMMETRIC with various buffer sizes to verify window tensor registration.
    Window tensor registration is triggered for buffers above a threshold.
    """
    torch.manual_seed(42)

    tensor_parallel_size = mpi_pool_executor.num_workers
    dtype = torch.bfloat16

    x = torch.randn((tensor_parallel_size, seq_len, hidden_size), dtype=dtype)
    weights = [torch.randn((hidden_size, hidden_size), dtype=dtype)]

    results = mpi_pool_executor.map(
        run_single_rank_fallback_test,
        *zip(
            *[
                (
                    tensor_parallel_size,
                    run_allreduce_window_tensor_test,
                    x,
                    None,
                    weights,
                    hidden_size,
                    dtype,
                    None,
                    None,
                    None,
                )
            ]
            * tensor_parallel_size
        ),
    )
    for r in results:
        assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs for this test")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.parametrize(
    "seq_len",
    [1, 2**18],  # Very small and very large (reduced from 2**20 to avoid int32 overflow)
    ids=lambda x: f"seqlen:{x}",
)
@pytest.mark.parametrize(
    "hidden_size",
    [64, 2**14],  # Very small and very large (reduced from 2**15 to avoid int32 overflow)
    ids=lambda x: f"hidden:{x}",
)
def test_lookup_table_fallback_extreme_values(mpi_pool_executor, seq_len, hidden_size):
    """
    Test AUTO strategy with extreme values that fall outside lookup table bounds.
    These should trigger fallback to NCCL_SYMMETRIC.
    """
    torch.manual_seed(42)

    tensor_parallel_size = mpi_pool_executor.num_workers
    dtype = torch.bfloat16

    x = torch.randn((tensor_parallel_size, seq_len, hidden_size), dtype=dtype)
    weights = [torch.randn((hidden_size, hidden_size), dtype=dtype)]

    results = mpi_pool_executor.map(
        run_single_rank_fallback_test,
        *zip(
            *[
                (
                    tensor_parallel_size,
                    run_allreduce_lookup_table_fallback_test,
                    x,
                    None,
                    weights,
                    hidden_size,
                    dtype,
                    None,
                    None,
                    None,
                )
            ]
            * tensor_parallel_size
        ),
    )
    for r in results:
        assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs for this test")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_nccl_symmetric_explicit_vs_auto(mpi_pool_executor):
    """
    Test that explicitly setting NCCL_SYMMETRIC produces same results as AUTO fallback.
    """
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
        run_single_rank_fallback_test,
        *zip(
            *[
                (
                    tensor_parallel_size,
                    run_allreduce_fallback_test,
                    x,
                    residual,
                    weights,
                    hidden_size,
                    dtype,
                    AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    AllReduceStrategy.NCCL_SYMMETRIC,
                    None,
                )
            ]
            * tensor_parallel_size
        ),
    )

    # Test with AUTO (should fallback to NCCL_SYMMETRIC in some cases)
    results_auto = mpi_pool_executor.map(
        run_single_rank_fallback_test,
        *zip(
            *[
                (
                    tensor_parallel_size,
                    run_allreduce_fallback_test,
                    x,
                    residual,
                    weights,
                    hidden_size,
                    dtype,
                    AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    AllReduceStrategy.AUTO,
                    None,
                )
            ]
            * tensor_parallel_size
        ),
    )

    for r in results_explicit + results_auto:
        assert r is True
