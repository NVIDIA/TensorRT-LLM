# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import tensorrt_llm.bindings.internal.userbuffers as ub
from tensorrt_llm._torch.distributed import (AllReduce, AllReduceFusionOp,
                                             AllReduceParams)
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

# needed since we reuse the mpi executor pool, first test running will leak a thread
pytestmark = pytest.mark.threadleak(enabled=False)

SEQ_LEN_CASES = (
    (1, ),
    (4, ),
    (15, ),
    (32, ),
    (128, ),
    (31, 11, 27, 4),  # Switching one/two shot in MNNVL
    (12, 2048),  # reallocate workspace in MNNVL
)
NCCL_SYMMETRIC_SEQ_LEN_CASES = tuple(seq_len for seq_len in SEQ_LEN_CASES
                                     if 2048 not in seq_len)
HIDDEN_SIZES = (8, 2880, 7168, 7176, 8192, 16384)
DTYPES = (torch.bfloat16, )
FUSION_CASES = (True, False)


def _seq_len_id(seq_len: tuple[int, ...]):
    return f"seqlen:{list(seq_len)}"


def _dtype_id(dtype: torch.dtype):
    return f"dtype:{torch.finfo(dtype).dtype}"


def _max_nccl_symmetric_buffer_size(dtype: torch.dtype):
    max_seq_len = max(max(seq_len) for seq_len in NCCL_SYMMETRIC_SEQ_LEN_CASES)
    return max_seq_len * max(HIDDEN_SIZES) * torch.empty(
        (), dtype=dtype).element_size()


def rms_norm(x: torch.Tensor, weight: torch.Tensor = None, eps: float = 1e-6):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight
    return y


def run_single_rank(
    tensor_parallel_size,
    single_rank_forward_func,
    input_list,
    residual_list,
    norm_weight,
    eps,
    hidden_size,
    dtype,
    fused_add_norm,
    reference_output_list,
    strategy,
    max_userbuffers_size,
):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        single_rank_forward_func(
            input_list,
            residual_list,
            norm_weight,
            eps,
            hidden_size,
            dtype,
            tensor_parallel_size,
            rank,
            fused_add_norm,
            reference_output_list,
            strategy,
            max_userbuffers_size,
        )
    except Exception:
        traceback.print_exc()
        raise
    return True


@torch.inference_mode()
def row_linear_residual_norm_fusion_forward(
    x_list: list[torch.Tensor],
    residual_list: list[torch.Tensor],
    norm_weight: torch.Tensor,
    eps: float,
    hidden_size: int,
    dtype: torch.dtype,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
    fusion: bool,
    reference_output_list: list[tuple[torch.Tensor, ...]],
    strategy: AllReduceStrategy,
    max_userbuffers_size: int | None,
):

    # Move all tensors to GPU
    x_list = [x.cuda() for x in x_list]
    residual_list = [residual.cuda() for residual in residual_list]
    norm_weight = norm_weight.cuda()
    reference_output_list = [
        tuple(t.cuda() for t in ref_output)
        for ref_output in reference_output_list
    ]

    if strategy == AllReduceStrategy.NCCL_SYMMETRIC:
        os.environ.pop("TLLM_TEST_MNNVL", None)
        assert max_userbuffers_size is not None
        ub.initialize_userbuffers_manager(tensor_parallel_size, 1, 1,
                                          tensor_parallel_rank,
                                          torch.cuda.device_count(),
                                          max_userbuffers_size)
    elif strategy == AllReduceStrategy.MNNVL:
        os.environ["TLLM_TEST_MNNVL"] = "1"

    MPI.COMM_WORLD.barrier()

    # Create a single AllReduce instance to be reused for all sequence lengths
    allreduce = AllReduce(
        mapping=Mapping(
            world_size=tensor_parallel_size,
            tp_size=tensor_parallel_size,
            rank=tensor_parallel_rank,
        ),
        strategy=strategy,
        dtype=dtype,
    )

    # Since all the modules here are provided by TRT-LLM,
    # so it has to be fullgraph compatible
    def func(input, residual, norm_weight, eps, enable_fusion):
        if enable_fusion:
            output, residual = allreduce(
                input,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=residual,
                    norm_weight=norm_weight,
                    eps=eps,
                ),
            )
            return (output, residual)
        else:
            output = allreduce(input)
            return (output, )

    try:
        # Process each sequence length using the same AllReduce instance
        for i, (x, residual, reference_output) in enumerate(
                zip(x_list, residual_list, reference_output_list)):
            output = func(x.clone(), residual.clone(), norm_weight, eps, fusion)

            torch.testing.assert_close(
                output[0],
                reference_output[0],
                rtol=0.05,
                atol=0.15,
            )

            if fusion:
                torch.testing.assert_close(
                    output[1],
                    reference_output[1],
                    rtol=0.05,
                    atol=0.15,
                )
    finally:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _run_row_linear_residual_norm_fusion(seq_len,
                                         hidden_size,
                                         dtype,
                                         strategy,
                                         fusion,
                                         mpi_pool_executor,
                                         max_userbuffers_size=None):
    torch.manual_seed(42)
    tensor_parallel_size = mpi_pool_executor.num_workers

    # Create norm_weight once (same for all sequence lengths)
    norm_weight = torch.randn((hidden_size, ), dtype=dtype)
    eps = 1e-5

    # Create lists of tensors for each sequence length
    x_list = []
    residual_list = []
    reference_output_list = []

    for seq_len_val in seq_len:
        x = torch.randn((tensor_parallel_size, seq_len_val, hidden_size),
                        dtype=dtype)
        residual = torch.randn((seq_len_val, hidden_size), dtype=dtype)
        reference_output = (torch.sum(x, dim=0), )
        if fusion:
            residual_out = reference_output[0] + residual
            reference_output = (rms_norm(residual_out.to(torch.float32),
                                         norm_weight,
                                         eps).to(dtype), residual_out)

        x_list.append(x)
        residual_list.append(residual)
        reference_output_list.append(reference_output)

    results = mpi_pool_executor.map(
        run_single_rank,
        *zip(*[
            (
                tensor_parallel_size,
                row_linear_residual_norm_fusion_forward,
                [x[i, :, :] for x in x_list
                 ],  # Extract the i-th rank's data from each sequence length
                residual_list,
                norm_weight,
                eps,
                hidden_size,
                dtype,
                fusion,
                reference_output_list,
                strategy,
                max_userbuffers_size,
            ) for i in range(tensor_parallel_size)
        ]),
    )
    for r in results:
        assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="needs 2 GPUs to run this test")
@pytest.mark.parametrize("seq_len", SEQ_LEN_CASES, ids=_seq_len_id)
@pytest.mark.parametrize("hidden_size",
                         HIDDEN_SIZES,
                         ids=lambda x: f"hidden:{x}")
@pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("fusion", FUSION_CASES, ids=["fusion", "no_fusion"])
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_mnnvl_row_linear_residual_norm_fusion(seq_len, hidden_size, dtype,
                                               fusion, mpi_pool_executor):
    _run_row_linear_residual_norm_fusion(seq_len, hidden_size, dtype,
                                         AllReduceStrategy.MNNVL, fusion,
                                         mpi_pool_executor)


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="needs 2 GPUs to run this test")
@pytest.mark.parametrize("seq_len",
                         NCCL_SYMMETRIC_SEQ_LEN_CASES,
                         ids=_seq_len_id)
@pytest.mark.parametrize("hidden_size",
                         HIDDEN_SIZES,
                         ids=lambda x: f"hidden:{x}")
@pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("fusion", FUSION_CASES, ids=["fusion", "no_fusion"])
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_nccl_symmetric_row_linear_residual_norm_fusion(seq_len, hidden_size,
                                                        dtype, fusion,
                                                        mpi_pool_executor):
    _run_row_linear_residual_norm_fusion(
        seq_len,
        hidden_size,
        dtype,
        AllReduceStrategy.NCCL_SYMMETRIC,
        fusion,
        mpi_pool_executor,
        max_userbuffers_size=_max_nccl_symmetric_buffer_size(dtype),
    )
