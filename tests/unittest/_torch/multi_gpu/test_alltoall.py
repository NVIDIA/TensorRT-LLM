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
from tensorrt_llm._torch.distributed import alltoall_helix

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
def run_alltoall_op(input_tensors, expected_recv_tensors, group):
    """Run alltoall_helix operation on a single rank."""
    rank = tensorrt_llm.mpi_rank()
    input_tensors = input_tensors[rank]
    expected_recv_tensors = expected_recv_tensors[rank]
    torch.cuda.set_device(rank)

    # Move input tensors to GPU
    input_tensors = [t.cuda() for t in input_tensors]

    # Call alltoall_helix
    output_tensors = alltoall_helix(input_tensors, group)

    # Verify output
    expected_recv_tensors = [t.cuda() for t in expected_recv_tensors]

    assert len(output_tensors) * len(group) == len(input_tensors)
    assert len(output_tensors) == len(expected_recv_tensors)

    for i, output_tensor in enumerate(output_tensors):
        assert output_tensor.dtype == expected_recv_tensors[i].dtype
        assert output_tensor.device == expected_recv_tensors[i].device
        assert output_tensor.shape == expected_recv_tensors[i].shape

        assert torch.allclose(output_tensor, expected_recv_tensors[i])

    return True


def run_alltoall_test(mpi_pool_executor, dtypes, shapes):
    torch.manual_seed(0)
    world_size = mpi_pool_executor.num_workers
    num_lists = len(shapes)

    # Create input tensors for each rank
    send_tensors = []
    for rank in range(world_size):
        send_tensors.append([])
        for list_idx in range(num_lists):
            send_tensors[-1].append([])
            for r in range(world_size):
                # Each rank creates a tensor with unique data to send to rank `r`
                tensor = torch.randn(*shapes[list_idx], dtype=dtypes[list_idx])
                send_tensors[-1][-1].append(tensor)
    expected_recv_tensors = []
    # Given the expected tensors sent by rank `rank` to all other ranks `r`,
    # we can now determine the expected tensors received by each rank `rank`
    for rank in range(world_size):
        expected_recv_tensors.append([])
        # For each original tensor, determine the received tensors
        for list_idx in range(num_lists):
            # The received tensors are a transpose of the sent tensors
            recv_tensors = [
                send_tensors[r][list_idx][rank] for r in range(world_size)
            ]
            expected_recv_tensors[-1].append(torch.stack(recv_tensors))

    input_tensors = [[x for y in tensors for x in y]
                     for tensors in send_tensors]

    # Create group list
    group = list(range(world_size))

    results = mpi_pool_executor.map(
        run_single_rank,
        *zip(*[(run_alltoall_op, input_tensors, expected_recv_tensors, group)] *
             world_size),
    )
    for r in results:
        assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Requires at least 2 GPUs for this test")
@pytest.mark.parametrize("seq_len", [16, 256, 1024],
                         ids=lambda x: f"seqlen:{x}")
@pytest.mark.parametrize("hidden_size", [128, 2048, 7168],
                         ids=lambda x: f"hidden:{x}")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_alltoall_2gpu(seq_len, hidden_size, mpi_pool_executor):
    dtypes = [torch.bfloat16, torch.float]
    shapes1 = [(seq_len, hidden_size)]
    run_alltoall_test(mpi_pool_executor, dtypes, shapes1)
    shapes2 = [(seq_len, hidden_size), (seq_len + 1, hidden_size + 1)]
    run_alltoall_test(mpi_pool_executor, dtypes, shapes2)


@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="Requires at least 4 GPUs for this test")
@pytest.mark.parametrize("seq_len", [28, 1004], ids=lambda x: f"seqlen:{x}")
@pytest.mark.parametrize("hidden_size", [36, 6284], ids=lambda x: f"hidden:{x}")
@pytest.mark.parametrize("mpi_pool_executor", [4], indirect=True)
def test_alltoall_4gpu(seq_len, hidden_size, mpi_pool_executor):
    dtypes = [torch.bfloat16, torch.float]
    shapes1 = [(seq_len, hidden_size)]
    run_alltoall_test(mpi_pool_executor, dtypes, shapes1)
    shapes2 = [(seq_len, hidden_size), (seq_len + 1, hidden_size + 1)]
    run_alltoall_test(mpi_pool_executor, dtypes, shapes2)
