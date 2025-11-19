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
# Window Tensor Creation Tests
# ============================================================================


@torch.inference_mode()
def run_window_tensor_creation_test(
    shape: tuple,
    dtype: torch.dtype,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
):
    """Test creating NCCL window tensors."""
    # Create group list for window tensor creation
    group = list(range(tensor_parallel_size))

    # Create window tensor
    window_tensor = torch.ops.trtllm.create_nccl_window_tensor(group, shape, dtype)

    # Verify tensor properties
    assert window_tensor.shape == shape, f"Shape mismatch: {window_tensor.shape} vs {shape}"
    assert window_tensor.dtype == dtype, f"Dtype mismatch: {window_tensor.dtype} vs {dtype}"
    assert window_tensor.is_cuda, "Window tensor should be on CUDA"

    # Verify tensor is contiguous
    assert window_tensor.is_contiguous(), "Window tensor should be contiguous"

    # Test that we can write to and read from the tensor
    test_data = torch.randn(shape, dtype=dtype, device="cuda")
    window_tensor.copy_(test_data)
    assert torch.allclose(window_tensor, test_data, rtol=1e-5, atol=1e-5), (
        "Data written to window tensor doesn't match"
    )

    # Synchronize CUDA to ensure all operations are complete before cleanup
    torch.cuda.synchronize()


@torch.inference_mode()
def run_window_tensor_multiple_test(
    shape: tuple,
    dtype: torch.dtype,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
    num_tensors: int,
):
    """Test creating multiple NCCL window tensors."""
    group = list(range(tensor_parallel_size))

    # Create multiple window tensors
    tensors = []
    for i in range(num_tensors):
        tensor = torch.ops.trtllm.create_nccl_window_tensor(group, shape, dtype)
        assert tensor.shape == shape
        assert tensor.dtype == dtype
        assert tensor.is_cuda
        tensors.append(tensor)

    # Verify all tensors are independent (different memory addresses)
    for i in range(len(tensors)):
        for j in range(i + 1, len(tensors)):
            assert not torch.equal(tensors[i], tensors[j]) or torch.allclose(
                tensors[i], tensors[j]
            ), "Tensors should be independent"

    # Test writing different data to each tensor
    for i, tensor in enumerate(tensors):
        test_data = torch.randn(shape, dtype=dtype, device="cuda") * (i + 1)
        tensor.copy_(test_data)
        assert torch.allclose(tensor, test_data, rtol=1e-5, atol=1e-5)

    # Synchronize CUDA to ensure all operations are complete before cleanup
    torch.cuda.synchronize()


@torch.inference_mode()
def run_window_tensor_different_shapes_test(
    shapes: list,
    dtype: torch.dtype,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
):
    """Test creating NCCL window tensors with different shapes."""
    group = list(range(tensor_parallel_size))

    # Create window tensors with different shapes
    tensors = []
    for shape in shapes:
        tensor = torch.ops.trtllm.create_nccl_window_tensor(group, shape, dtype)
        assert tensor.shape == shape, f"Shape mismatch: {tensor.shape} vs {shape}"
        assert tensor.dtype == dtype
        assert tensor.is_cuda
        tensors.append(tensor)

        # Test writing data to tensor
        test_data = torch.randn(shape, dtype=dtype, device="cuda")
        tensor.copy_(test_data)
        assert torch.allclose(tensor, test_data, rtol=1e-5, atol=1e-5)

    # Synchronize CUDA to ensure all operations are complete before cleanup
    torch.cuda.synchronize()


# ============================================================================
# Pytest Test Functions
# ============================================================================


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.parametrize("shape", [(1024,), (1024, 2048), (16, 32, 64)], ids=lambda x: f"shape:{x}")
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16], ids=lambda x: str(x)
)
def test_create_window_tensor(mpi_pool_executor, shape, dtype):
    """Test creating a single NCCL window tensor."""
    torch.manual_seed(42)

    tensor_parallel_size = mpi_pool_executor.num_workers

    results = mpi_pool_executor.map(
        run_single_rank_test,
        *zip(
            *[
                (
                    tensor_parallel_size,
                    run_window_tensor_creation_test,
                    shape,
                    dtype,
                    tensor_parallel_size,
                    None,
                )
            ]
            * tensor_parallel_size
        ),
    )
    for r in results:
        assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.parametrize("shape", [(1024, 2048)], ids=lambda x: f"shape:{x}")
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=lambda x: str(x))
@pytest.mark.parametrize("num_tensors", [2, 4, 8], ids=lambda x: f"num:{x}")
def test_create_multiple_window_tensors(mpi_pool_executor, shape, dtype, num_tensors):
    """Test creating multiple NCCL window tensors."""
    torch.manual_seed(42)

    tensor_parallel_size = mpi_pool_executor.num_workers

    results = mpi_pool_executor.map(
        run_single_rank_test,
        *zip(
            *[
                (
                    tensor_parallel_size,
                    run_window_tensor_multiple_test,
                    shape,
                    dtype,
                    tensor_parallel_size,
                    None,
                    num_tensors,
                )
            ]
            * tensor_parallel_size
        ),
    )
    for r in results:
        assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=lambda x: str(x))
def test_create_window_tensors_different_shapes(mpi_pool_executor, dtype):
    """Test creating NCCL window tensors with different shapes."""
    torch.manual_seed(42)

    tensor_parallel_size = mpi_pool_executor.num_workers
    shapes = [(1024,), (2048, 4096), (16, 32, 64)]

    results = mpi_pool_executor.map(
        run_single_rank_test,
        *zip(
            *[
                (
                    tensor_parallel_size,
                    run_window_tensor_different_shapes_test,
                    shapes,
                    dtype,
                    tensor_parallel_size,
                    None,
                )
            ]
            * tensor_parallel_size
        ),
    )
    for r in results:
        assert r is True
