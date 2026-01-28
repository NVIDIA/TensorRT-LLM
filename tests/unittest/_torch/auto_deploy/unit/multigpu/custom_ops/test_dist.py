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
"""Unit tests for custom dist ops."""

import os
import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm._torch.auto_deploy.distributed.common import initialize_or_skip

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

# needed since we reuse the mpi executor pool, first test running will leak a thread
pytestmark = pytest.mark.threadleak(enabled=False)


def run_single_rank(tensor_parallel_size, single_rank_forward_func):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    # Initialize torch distributed process group for auto_deploy ops
    initialize_or_skip(rank=rank, world_size=tensor_parallel_size, port=29500)
    try:
        single_rank_forward_func(rank, tensor_parallel_size)
    except Exception:
        traceback.print_exc()
        raise
    return True


@torch.inference_mode()
def run_all_reduce_test(rank, world_size):
    x = torch.ones(10, 10).to("cuda")
    from tensorrt_llm._torch.auto_deploy import custom_ops  # noqa

    # wrap the custom all_reduce op in a function to war pickle issue
    def custom_all_reduce(x, strategy):
        return torch.ops.auto_deploy.torch_dist_all_reduce(x, strategy)

    y = custom_all_reduce(x, "AUTO")

    assert torch.equal(x * world_size, y)


@torch.inference_mode()
def run_all_gather_test(rank, world_size):
    x = torch.ones(10, 10).to("cuda")
    from tensorrt_llm._torch.auto_deploy import custom_ops  # noqa

    # wrap the custom all_gather op in a function to war pickle issue
    def custom_all_gather(x):
        return torch.ops.auto_deploy.torch_dist_all_gather(x)

    y = custom_all_gather(x)

    assert torch.sum(y) == world_size * torch.sum(x)
    assert y.shape == (world_size * x.shape[0], *x.shape[1:])


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs for this test")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_all_reduce(mpi_pool_executor):
    tensor_parallel_size = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(
        run_single_rank,
        *zip(*[(tensor_parallel_size, run_all_reduce_test)] * tensor_parallel_size),
    )
    for r in results:
        assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs for this test")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_all_gather(mpi_pool_executor):
    tensor_parallel_size = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(
        run_single_rank,
        *zip(*[(tensor_parallel_size, run_all_gather_test)] * tensor_parallel_size),
    )
    for r in results:
        assert r is True
