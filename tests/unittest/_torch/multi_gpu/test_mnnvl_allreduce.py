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
from mpi4py.futures import MPIPoolExecutor
from utils.util import skip_pre_blackwell

import tensorrt_llm
from tensorrt_llm._torch.distributed import (AllReduce, AllReduceFusionOp,
                                             AllReduceParams)
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)


def rms_norm(x: torch.Tensor, weight: torch.Tensor = None, eps: float = 1e-6):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight
    return y


def run_single_rank(
    tensor_parallel_size,
    single_rank_forward_func,
    input,
    residual,
    norm_weight,
    eps,
    hidden_size,
    dtype,
    fused_add_norm,
    reference_output,
):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        single_rank_forward_func(
            input,
            residual,
            norm_weight,
            eps,
            hidden_size,
            dtype,
            tensor_parallel_size,
            rank,
            fused_add_norm,
            reference_output,
        )
    except Exception:
        traceback.print_exc()
        raise
    return True


@torch.inference_mode()
def row_linear_residual_norm_fusion_forward(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hidden_size: int,
    dtype: torch.dtype,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
    fusion: bool,
    reference_output: tuple[torch.Tensor, ...],
):

    x = x.cuda()
    residual = residual.cuda()
    norm_weight = norm_weight.cuda()
    reference_output = tuple(t.cuda() for t in reference_output)

    MPI.COMM_WORLD.barrier()
    os.environ["TRTLLM_MNNVL_AR_ENABLED"] = "1"

    allreduce = AllReduce(
        mapping=Mapping(
            world_size=tensor_parallel_size,
            tp_size=tensor_parallel_size,
            rank=tensor_parallel_rank,
        ),
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
                ))
            return (output, residual)
        else:
            output = allreduce(input)
            return (output, )

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


@skip_pre_blackwell
@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="needs 2 GPUs to run this test")
@pytest.mark.parametrize("seq_len", [1, 4, 32, 128],
                         ids=lambda x: f"seqlen:{x}")
@pytest.mark.parametrize("hidden_size", [7168], ids=lambda x: f"hidden:{x}")
@pytest.mark.parametrize(
    "fusion",
    [True, False],
    ids=["fusion", "no_fusion"],
)
def test_row_linear_residual_norm_fusion(seq_len, hidden_size, fusion):

    torch.manual_seed(42)
    dtype = torch.bfloat16
    tensor_parallel_size = 2

    x = torch.randn((tensor_parallel_size, seq_len, hidden_size), dtype=dtype)
    residual = torch.randn((seq_len, hidden_size), dtype=dtype)
    norm_weight = torch.randn((hidden_size, ), dtype=dtype)
    eps = 1e-5
    reference_output = (torch.sum(x, dim=0), )
    if fusion:
        residual_out = reference_output[0] + residual
        reference_output = (rms_norm(residual_out.to(torch.float32),
                                     norm_weight, eps).to(dtype), residual_out)

    with MPIPoolExecutor(max_workers=tensor_parallel_size) as executor:
        results = executor.map(
            run_single_rank,
            *zip(*[(
                tensor_parallel_size,
                row_linear_residual_norm_fusion_forward,
                x[i, :, :],
                residual,
                norm_weight,
                eps,
                hidden_size,
                dtype,
                fusion,
                reference_output,
            ) for i in range(tensor_parallel_size)]),
        )
        for r in results:
            assert r is True
