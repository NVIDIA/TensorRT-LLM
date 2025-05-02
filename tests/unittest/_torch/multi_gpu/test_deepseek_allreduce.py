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
                                             DeepseekAllReduce)
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
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


def run_single_rank(tensor_parallel_size, single_rank_forward_func, input,
                    residual, hidden_size, dtype, fused_add_norm):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        single_rank_forward_func(input, residual, hidden_size, dtype,
                                 tensor_parallel_size, rank, fused_add_norm)
    except Exception:
        traceback.print_exc()
        raise
    return True


def run_moe_single_rank(tensor_parallel_size, single_rank_forward_func,
                        token_input, residual, active_experts_token_input,
                        scale, l0_weight):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        single_rank_forward_func(token_input, residual,
                                 active_experts_token_input, scale,
                                 tensor_parallel_size, rank, l0_weight)
    except Exception:
        traceback.print_exc()
        raise
    return True


@torch.inference_mode()
def row_linear_residual_norm_fusion_forward(
        x: torch.Tensor, residual: torch.Tensor, hidden_size: int,
        dtype: torch.dtype, tensor_parallel_size: int,
        tensor_parallel_rank: int, fusion_op: AllReduceFusionOp):

    x = x.cuda()
    residual = residual.cuda()
    norm_weight = torch.randn((hidden_size, ), dtype=dtype, device="cuda")
    eps = 1e-5

    norm = RMSNorm(hidden_size=hidden_size, eps=eps, dtype=dtype).cuda()

    allreduce = AllReduce(mapping=Mapping(
        world_size=tensor_parallel_size,
        tp_size=tensor_parallel_size,
        rank=tensor_parallel_rank,
    ), ).cuda()

    deepseek_allreduce = DeepseekAllReduce(mapping=Mapping(
        world_size=tensor_parallel_size,
        tp_size=tensor_parallel_size,
        rank=tensor_parallel_rank,
    ), ).cuda()

    scale = torch.tensor(1.0, dtype=torch.float32).cuda()

    # Since all the modules here are provided by TRT-LLM,
    # so it has to be fullgraph compatible
    def func(input, residual, enable_fusion):
        xs = torch.chunk(input, 2, dim=-1)
        if enable_fusion:
            inter_x = input / tensor_parallel_size
            if fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM:
                hidden_states, residual = deepseek_allreduce(
                    inter_x,
                    [residual, norm_weight],
                    eps,
                    AllReduceFusionOp.RESIDUAL_RMS_NORM,
                )
                output = (hidden_states, residual)
            elif fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4:
                act_fp4, act_sf, residual = deepseek_allreduce(
                    inter_x,
                    [residual, norm_weight, scale],
                    eps,
                    AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4,
                )
                output = (act_fp4, act_sf, residual)
            elif fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4:
                hidden_states, act_fp4, act_sf, residual = deepseek_allreduce(
                    inter_x,
                    [residual, norm_weight, scale],
                    eps,
                    AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4,
                )
                output = (hidden_states, act_fp4, act_sf, residual)
        else:
            hidden_states = x
            inter_output = hidden_states + residual
            hidden_states = norm(inter_output)
            output = (hidden_states, residual)
            if fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4:
                act_fp4, act_sf = torch.ops.trtllm.fp4_quantize(
                    hidden_states, scale, 16, False)

                output = (act_fp4, act_sf, residual)
            elif fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4:
                act_fp4, act_sf = torch.ops.trtllm.fp4_quantize(
                    hidden_states, scale, 16, False)

                output = (hidden_states, act_fp4, act_sf, residual)

        return output

    norm.weight.data.copy_(norm_weight)

    output = func(x.clone(), residual.clone(), enable_fusion=False)
    fuse_output = func(x.clone(), residual.clone(), enable_fusion=True)

    if fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM:
        h, r = output
        fh, fr = fuse_output

        torch.testing.assert_close(
            h,
            fh,
            rtol=0.05,
            atol=0.15,
        )
    elif fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4:
        act_fp4, act_sf, residual = output
        f_act_fp4, f_act_sf, f_residual = fuse_output
        torch.testing.assert_close(
            act_fp4,
            f_act_fp4,
            rtol=0.05,
            atol=0.15,
        )
    elif fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4:
        hidden_states, act_fp4, act_sf, residual = output
        f_hidden_states, f_act_fp4, f_act_sf, f_residual = fuse_output
        torch.testing.assert_close(
            hidden_states,
            f_hidden_states,
        )

    # torch run
    torch_output = x
    torch_inter_output = torch_output + residual
    torch_inter_output = torch_inter_output.to(torch.float32)
    torch_final_output = rms_norm(torch_inter_output, norm_weight,
                                  eps).to(dtype)

    if fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM or fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4:
        torch.testing.assert_close(
            output[0],
            torch_final_output,
            rtol=0.05,
            atol=0.15,
        )


@skip_pre_blackwell
@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="needs 2 GPUs to run this test")
@pytest.mark.parametrize("seq_len", [2, 32], ids=lambda x: f"seqlen:{x}")
@pytest.mark.parametrize("hidden_size", [7168], ids=lambda x: f"hidden:{x}")
@pytest.mark.parametrize("fusion_op", [
    AllReduceFusionOp.RESIDUAL_RMS_NORM,
    AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4,
    AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4,
],
                         ids=[
                             "residual_rms_norm",
                             "residual_rms_norm_quant_nvfp4",
                             "residual_rms_norm_out_quant_nvfp4"
                         ])
def test_row_linear_residual_norm_fusion(seq_len, hidden_size, fusion_op):
    torch.manual_seed(42)
    dtype = torch.bfloat16
    tensor_parallel_size = 2
    x = torch.randn((seq_len, hidden_size), dtype=dtype)
    residual = torch.randn_like(x)
    with MPIPoolExecutor(max_workers=tensor_parallel_size) as executor:
        results = executor.map(
            run_single_rank,
            *zip(*[(tensor_parallel_size,
                    row_linear_residual_norm_fusion_forward, x, residual,
                    hidden_size, dtype, fusion_op)] * 2),
        )
        for r in results:
            assert r is True
