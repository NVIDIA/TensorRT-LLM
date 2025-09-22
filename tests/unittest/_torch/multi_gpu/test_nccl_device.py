# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
import tensorrt_llm.bindings.internal.userbuffers as ub
from tensorrt_llm._torch.distributed import (AllReduce, AllReduceFusionOp,
                                             AllReduceParams, AllReduceStrategy)
from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

# needed since we reuse the mpi executor pool, first test running will leak a thread
pytestmark = pytest.mark.threadleak(enabled=False)


def init_userbuffers_allocator(tp_size, rank, max_ub_size):
    ub.initialize_userbuffers_manager(tp_size, 1, 1, rank,
                                      torch.cuda.device_count(), max_ub_size,
                                      True)


def create_userbuffers_tensor(shape, dtype):
    # WAR pickle error
    def func(shape, dtype):
        return torch.ops.trtllm.create_userbuffers_tensor(shape, dtype)

    return func(shape, dtype)


# This rms_norm aligns with ub impl that calculate gamma * hidden in high
# precision
def rms_norm(input, gamma, eps):
    variance = input.pow(2).mean(-1, keepdim=True)
    hidden_states = input * torch.rsqrt(variance + eps)
    return gamma.to(torch.float32) * hidden_states


def run_single_rank_ar_rms_norm(tensor_parallel_size, a, b, c, gamma):
    rank = tensorrt_llm.mpi_rank()

    # Set CUDA device BEFORE any CUDA operations or NCCL initialization
    torch.cuda.set_device(rank)

    # Ensure CUDA context is properly initialized for this device
    torch.cuda.synchronize()

    try:
        support = ub.ub_supported()
        if not support:
            return True
        eps = 1e-6

        # Split tensors for tensor parallelism - ensure equal sizes
        k_chunk_size = a.size(1) // tensor_parallel_size
        b.size(0) // tensor_parallel_size

        # Ensure we get exactly tensor_parallel_size chunks
        a_partial = torch.split(a, k_chunk_size, dim=1)
        b_partial = torch.split(b, k_chunk_size, dim=0)

        a_local = a_partial[rank].cuda()
        b_local = b_partial[rank].cuda()
        c = c.cuda()
        gamma = gamma.cuda()

        ub_size = c.nelement() * c.element_size()
        init_userbuffers_allocator(tensor_parallel_size, rank, ub_size)

        ub0_tensor = create_userbuffers_tensor(c.size(), a.dtype)
        hidden = torch.matmul(a_local, b_local, out=ub0_tensor)

        # Add barrier to ensure all MPI processes are ready before NCCL initialization
        if ENABLE_MULTI_DEVICE:
            tensorrt_llm._utils.mpi_barrier()

        # Ensure all ranks have set their CUDA devices before creating AllReduce
        if ENABLE_MULTI_DEVICE:
            tensorrt_llm._utils.mpi_barrier()

        mapping = Mapping(
            world_size=tensor_parallel_size,
            tp_size=tensor_parallel_size,
            rank=rank,
        )
        ar = AllReduce(mapping=mapping, strategy=AllReduceStrategy.NCCL_DEVICE)
        ar_params = AllReduceParams(
            strategy=AllReduceStrategy.NCCL_DEVICE,
            fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
            residual=c,
            norm_weight=gamma,
            eps=eps,
        )

        res_ub, residual = ar.forward(hidden, all_reduce_params=ar_params)
        res = res_ub.clone()

        torch.cuda.synchronize()
        # Fully simulate matmul + allreduce behavior
        ax = [a_partial[i].cuda() for i in range(0, tensor_parallel_size)]
        bx = [b_partial[i].cuda() for i in range(0, tensor_parallel_size)]
        h1 = [
            torch.matmul(ax[i], bx[i]) for i in range(0, tensor_parallel_size)
        ]
        sum = h1[0]
        for i in range(1, tensor_parallel_size):
            sum = sum + h1[i]
        ref_residual = sum + c
        ref = rms_norm(ref_residual.to(torch.float32), gamma, eps).to(res.dtype)
        torch.testing.assert_close(ref, res, atol=5e-1, rtol=1e-2)

        chunked_residual_comparison = False  # Current production version performs full scatter also for the residual, so we can compare unchunked

        # Since we do not always perform an AllGather of the residual, let's compare on every rank the right portions of the residual
        residual_chunk_size = ref_residual.size(0) // tensor_parallel_size
        if ref_residual.size(0) % tensor_parallel_size != 0:
            residual_chunk_size += 1
        chunk_start = rank * residual_chunk_size
        chunk_end = min((rank + 1) * residual_chunk_size, ref_residual.size(0))

        # If we do perform the AllGather implicitly we can compare the entire tensor.
        if not chunked_residual_comparison:
            chunk_start = 0
            chunk_end = ref_residual.size(0)
        ref_residual = ref_residual[chunk_start:chunk_end]
        residual = residual[chunk_start:chunk_end]

        torch.testing.assert_close(ref_residual, residual, atol=5e-1, rtol=1e-2)

    except Exception:
        traceback.print_exc()
        raise
    return True


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="needs 2 GPUs to run this test")
@pytest.mark.parametrize("mnk", [(128, 8192, 64), (79, 512, 32)],
                         ids=lambda x: f"m{x[0]}_n{x[1]}_k{x[2]}")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_user_buffers_ar_rms_norm(mnk, mpi_pool_executor):
    # Ensure all MPI processes are synchronized before any test execution
    if ENABLE_MULTI_DEVICE:
        tensorrt_llm._utils.mpi_barrier()

    torch.manual_seed(42)
    tensor_parallel_size = 2
    dtype = torch.float16
    m = mnk[0]
    n = mnk[1]
    k = mnk[2]

    # Ensure tensor dimensions are compatible with 2-way tensor parallelism
    assert (
        k % tensor_parallel_size == 0
    ), f"k dimension {k} must be divisible by tensor_parallel_size {tensor_parallel_size}"
    assert (
        n % tensor_parallel_size == 0
    ), f"n dimension {n} must be divisible by tensor_parallel_size {tensor_parallel_size}"

    a = torch.randn((m, k), dtype=dtype)
    b = torch.randn((k, n), dtype=dtype)
    c = torch.randn((m, n), dtype=dtype)
    gamma = torch.randn((n), dtype=dtype)

    results = mpi_pool_executor.map(
        run_single_rank_ar_rms_norm,
        *zip(*[(tensor_parallel_size, a, b, c, gamma)] * tensor_parallel_size),
    )
    for r in results:
        assert r is True
