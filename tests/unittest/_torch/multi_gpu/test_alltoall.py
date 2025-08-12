# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def run_single_rank(single_rank_forward_func, *args, **kwargs):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        single_rank_forward_func(*args, **kwargs)
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
    input_tensors = input_tensors.cuda() if isinstance(
        input_tensors, torch.Tensor) else [t.cuda() for t in input_tensors]

    # Call alltoall
    output_tensors = alltoall(input_tensors, group, dims, new_dims)

    # Verify output
    if isinstance(input_tensors, torch.Tensor):
        input_tensors = [input_tensors]
        expected_recv_tensors = [expected_recv_tensors.cuda()]
        output_tensors = [output_tensors]
    else:
        expected_recv_tensors = [t.cuda() for t in expected_recv_tensors]

    assert len(output_tensors) == len(input_tensors)
    assert len(output_tensors) == len(expected_recv_tensors)

    for i, output_tensor in enumerate(output_tensors):
        assert output_tensor.dtype == input_tensors[i].dtype
        assert output_tensor.device == input_tensors[i].device
        assert output_tensor.shape == expected_recv_tensors[i].shape

        assert torch.allclose(output_tensor, expected_recv_tensors[i])

    return True


def run_alltoall_test(mpi_pool_executor, all_dims, dtypes, shape):
    torch.manual_seed(0)
    world_size = mpi_pool_executor.num_workers
    dims, new_dims = all_dims
    assert not isinstance(dims, list) or len(dims) > 1
    num_lists = len(dims) if isinstance(dims, list) else 1

    # Create input tensors for each rank
    input_tensors = []
    expected_send_tensors = []
    for rank in range(world_size):
        input_tensors.append([])
        expected_send_tensors.append([])
        for list_idx in range(num_lists):
            # Each rank creates a tensor with unique data
            tensor = torch.randn(*shape, dtype=dtypes[list_idx])
            input_tensors[-1].append(tensor)
            d = dims[list_idx] if isinstance(dims, list) else dims
            send_tensors = []
            for r in range(world_size):
                idx = [slice(None)] * len(shape)
                split = shape[d] // world_size
                idx[d] = slice(r * split, (r + 1) * split)
                send_tensors.append(tensor[idx])
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
            else:
                expected_recv_tensors[-1].append(
                    torch.stack(recv_tensors, dim=new_dim))
    # if we have single tensors, replace the list with a single tensor
    if num_lists == 1:
        input_tensors = [t[0] for t in input_tensors]
        expected_recv_tensors = [t[0] for t in expected_recv_tensors]

    # Create group list
    group = list(range(world_size))

    results = mpi_pool_executor.map(
        run_single_rank,
        *zip(*[(run_alltoall_op, input_tensors, expected_recv_tensors, group,
                dims, new_dims)] * world_size),
    )
    for r in results:
        assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Requires at least 2 GPUs for this test")
@pytest.mark.parametrize("seq_len", [16, 256, 1024],
                         ids=lambda x: f"seqlen:{x}")
@pytest.mark.parametrize("hidden_size", [128, 2048, 7168],
                         ids=lambda x: f"hidden:{x}")
@pytest.mark.parametrize(
    "all_dims", [[0, None], [1, 0], [[0, 1], [None, 0]], [[1, 1], [0, 0]]],
    ids=lambda x: f"all_dims:{x}")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_alltoall_2gpu(seq_len, hidden_size, all_dims, mpi_pool_executor):
    dtypes = [torch.bfloat16, torch.float]
    shape = (seq_len, hidden_size)
    run_alltoall_test(mpi_pool_executor, all_dims, dtypes, shape)


@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="Requires at least 4 GPUs for this test")
@pytest.mark.parametrize("seq_len", [28, 1004], ids=lambda x: f"seqlen:{x}")
@pytest.mark.parametrize("hidden_size", [36, 6284], ids=lambda x: f"hidden:{x}")
@pytest.mark.parametrize(
    "all_dims", [[0, None], [1, 0], [[0, 1], [None, 0]], [[1, 1], [1, None]]],
    ids=lambda x: f"all_dims:{x}")
@pytest.mark.parametrize("mpi_pool_executor", [4], indirect=True)
def test_alltoall_4gpu(seq_len, hidden_size, all_dims, mpi_pool_executor):
    dtypes = [torch.bfloat16, torch.float]
    shape = (seq_len, hidden_size)
    run_alltoall_test(mpi_pool_executor, all_dims, dtypes, shape)
