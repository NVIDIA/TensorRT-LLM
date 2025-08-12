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
from tensorrt_llm._torch.distributed import alltoall

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

# needed since we reuse the mpi executor pool, first test running will leak a thread
pytestmark = pytest.mark.threadleak(enabled=False)


def run_single_rank(world_size, single_rank_forward_func, input_tensors, group,
                    dims, new_dims):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        single_rank_forward_func(input_tensors, group, dims, new_dims)
    except Exception:
        traceback.print_exc()
        raise
    return True


@torch.inference_mode()
def run_alltoall_op(input_tensors, expected_recv_tensors, group, dims,
                    new_dims):
    """Run alltoall operation on a single rank."""
    rank = tensorrt_llm.mpi_rank()
    input_tensors = input_tensors[rank]
    expected_recv_tensors = expected_recv_tensors[rank]
    torch.cuda.set_device(rank)

    # Move input tensors to GPU
    input_tensors = [tensor.cuda() for tensor in rank_input]

    # Call alltoall
    output_tensors = alltoall(input_tensors, group, dims, new_dims)

    # Verify output
    assert len(output_tensors) == len(input_tensors)
    assert len(output_tensors) == len(expected_recv_tensors)

    for i, output_tensor in enumerate(output_tensors):
        # Each output tensor should have the same shape as input tensors
        assert output_tensor.shape == input_tensors[i].shape
        assert output_tensor.dtype == input_tensors[i].dtype
        assert output_tensor.device == input_tensors[i].device

        assert torch.allclose(output_tensor, expected_recv_tensors[i])

    return True


@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="Requires at least 4 GPUs for this test")
@pytest.mark.parametrize("seq_len", [16, 256, 1024],
                         ids=lambda x: f"seqlen:{x}")
@pytest.mark.parametrize("hidden_size", [128, 2048, 7168],
                         ids=lambda x: f"hidden:{x}")
@pytest.mark.parametrize(
    "dims", [[0, None], [1, 0], [[0, 1], [None, 0]], [[1, 1], [0, 0]]],
    ids=lambda x: f"dims:{x}")
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_alltoall_patterns(seq_len, hidden_size, dims, mpi_pool_executor):
    torch.manual_seed(0)
    dtypes = [torch.bfloat16, torch.float]
    world_size = mpi_pool_executor.num_workers
    dims, new_dims = dims
    num_lists = len(dims) if isinstance(dims, list) else 1

    # Create input tensors for each rank
    input_tensors = []
    expected_send_tensors = []
    for rank in range(world_size):
        input_tensors.append([])
        expected_send_tensors.append([])
        for list_idx in range(num_lists):
            # Each rank creates a tensor with unique data
            tensor = torch.randn(seq_len, hidden_size, dtype=dtypes[list_idx])
            input_tensors[-1].append(tensor)
            d = dims[list_idx] if isinstance(dims, list) else dims
            send_tensors = []
            for r in range(world_size):
                if d == 0:
                    split = seq_len // world_size
                    send_tensors.append(tensor[r * split:(r + 1) * split])
                else:
                    split = hidden_size // world_size
                    send_tensors.append(tensor[:, r * split:(r + 1) * split])
            expected_send_tensors[-1].append(send_tensors)
    expected_recv_tensors = []
    for rank in range(world_size):
        expected_recv_tensors.append([])
        for list_idx in range(num_lists):
            recv_tensors = [
                expected_send_tensors[r][list_idx][rank]
                for r in range(world_size)
            ]
            new_dim = new_dims[list_idx] if isinstance(new_dims,
                                                       list) else new_dims
            if new_dim is None:
                new_dim = dims[list_idx] if isinstance(dims, list) else dims
            expected_recv_tensors[-1].append(
                torch.cat(recv_tensors, dim=new_dim))

    # Create group list
    group = list(range(world_size))

    results = mpi_pool_executor.map(
        run_single_rank,
        *zip(*[(world_size, run_alltoall_op, input_tensors,
                expected_recv_tensors, group, dims, new_dims)] * world_size),
    )
    for r in results:
        assert r is True
