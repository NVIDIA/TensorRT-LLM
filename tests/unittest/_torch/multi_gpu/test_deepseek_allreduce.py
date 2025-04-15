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
import torch.nn as nn
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from utils.util import skip_pre_blackwell

import tensorrt_llm
from tensorrt_llm._torch.distributed import (AllReduce, AllReduceFusionOp,
                                             AllReduceParams, DeepseekAllReduce)
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
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
            elif fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_AND_QUANT_NVFP4:
                hidden_states, act_fp4, act_sf, residual = deepseek_allreduce(
                    inter_x,
                    [residual, norm_weight, scale],
                    eps,
                    AllReduceFusionOp.RESIDUAL_RMS_NORM_AND_QUANT_NVFP4,
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
            elif fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_AND_QUANT_NVFP4:
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
    elif fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_AND_QUANT_NVFP4:
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

    if fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM or fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_AND_QUANT_NVFP4:
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
    AllReduceFusionOp.RESIDUAL_RMS_NORM_AND_QUANT_NVFP4,
],
                         ids=[
                             "residual_rms_norm",
                             "residual_rms_norm_quant_nvfp4",
                             "residual_rms_norm_and_quant_nvfp4"
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


@torch.inference_mode()
def moe_residual_norm_fusion_forward(
        token_input: torch.Tensor, residual: torch.Tensor,
        active_experts_token_input: torch.Tensor, scale: torch.Tensor,
        tensor_parallel_size: int, tensor_parallel_rank: int,
        l0_weight: torch.Tensor):
    torch.manual_seed(42)

    # * token_input:
    #   [num_token, 7168]
    #   different val for different device
    # * active_experts_token_input
    #   [num_global_exp, num_token, 7168]
    #   need to slice to [num_device_exp, num_token, 7168] before use
    # * scale
    #   [num_global_exp, num_token]
    #   per expert per token scale
    #   need to slice to [num_device_exp, num_token, 7168] before use
    #   different value for each device

    token_input = token_input.cuda()
    residual = residual.cuda()
    active_experts_token_input = active_experts_token_input.cuda()
    scale = scale.cuda()

    dtype = token_input.dtype
    num_global_experts = scale.size(0)
    num_device_experts = num_global_experts // tensor_parallel_size
    tensor_num_device_experts = torch.tensor(num_device_experts,
                                             dtype=torch.int32,
                                             device="cuda")
    # num_token = token_input.shape[0]
    hidden_size = token_input.shape[1]

    # Setup parameters
    eps = 1e-5
    norm_weight = torch.randn((hidden_size, ), dtype=dtype, device="cuda")

    # Initialize DeepseekAllReduce and AllReduce
    deepseek_allreduce = DeepseekAllReduce(mapping=Mapping(
        world_size=tensor_parallel_size,
        tp_size=tensor_parallel_size,
        rank=tensor_parallel_rank,
    )).cuda()

    # Initialize RMSNorm
    norm = RMSNorm(hidden_size=hidden_size, eps=eps, dtype=dtype).cuda()
    norm.weight.data.copy_(norm_weight)

    l0 = Linear(
        in_features=hidden_size,
        out_features=hidden_size,
        bias=False,
        dtype=dtype,
        mapping=Mapping(
            world_size=tensor_parallel_size,
            tp_size=tensor_parallel_size,
            rank=tensor_parallel_rank,
        ),
        tensor_parallel_mode=TensorParallelMode.ROW,
    ).cuda()
    l0.load_weights([dict(weight=l0_weight)])
    token_input_chunked = torch.chunk(token_input.clone(),
                                      tensor_parallel_size,
                                      dim=-1)
    fc2_output = l0(
        token_input_chunked[tensor_parallel_rank],
        all_reduce_params=AllReduceParams(
            fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
            residual=residual,
            norm_weight=norm_weight,
            eps=eps,
            enable_allreduce=False,
        ),
    )

    # Define fusion operation
    # slice [num_global_exp, num_token, 7168] -> [num_device_exp, num_token, 7168]
    active_experts_token_input_parallel = torch.chunk(
        active_experts_token_input.clone(), tensor_parallel_size, dim=0)
    active_experts_token_equalized = active_experts_token_input_parallel[
        tensor_parallel_rank]

    # slice [num_global_exp, num_token] -> [num_device_exp, num_token]
    scale_parallel = torch.chunk(scale.clone(), tensor_parallel_size, dim=0)
    scale_equalized = scale_parallel[tensor_parallel_rank]

    fusion_op = AllReduceFusionOp.MOE_ALLREDUCE_RESIDUAL_RMS_NORM

    # Run with fusion
    final_hidden_states, updated_residual = deepseek_allreduce(
        token_input.clone(), [
            residual.clone(),
            norm_weight.clone(),
            tensor_num_device_experts,
            scale_equalized.clone(),
            active_experts_token_equalized,
            fc2_output,
        ], eps, fusion_op)

    torch_l0 = nn.Linear(in_features=hidden_size,
                         out_features=hidden_size,
                         bias=False,
                         dtype=dtype)
    torch_l0.weight.data.copy_(l0_weight)
    torch_l0.cuda()

    torch_linear_output = torch_l0(token_input)
    # Verify with torch reference implementation
    expert_reduction = torch.sum(active_experts_token_input *
                                 scale.unsqueeze(-1),
                                 dim=0)
    torch_before_residual = (expert_reduction + torch_linear_output)
    torch_residual = torch_before_residual + residual
    torch_residual = torch_residual.to(torch.float32)
    torch_final_hidden_states = rms_norm(torch_residual, norm_weight,
                                         eps).to(dtype)

    # Verify results are close to reference
    torch.testing.assert_close(
        final_hidden_states,
        torch_final_hidden_states,
        rtol=0.2,
        atol=0.2,
    )

    return True


@torch.inference_mode()
def test_moe_residual_norm_fusion():
    torch.manual_seed(42)

    seq_len = 16
    hidden_size = 7168
    dtype = torch.bfloat16
    tensor_parallel_size = 2
    num_global_experts = 4

    # [num_token, 7168]
    token_input = torch.randn((seq_len, hidden_size), dtype=dtype)
    # [num_global_exp, num_token, 7168]
    active_experts_token_input = torch.randn(
        (num_global_experts, seq_len, hidden_size), dtype=dtype, device="cuda")
    # [num_global_exp, num_token]
    scale = torch.randn((num_global_experts, seq_len),
                        dtype=torch.float32,
                        device="cuda")
    # [num_token, 7168]
    residual = torch.randn_like(token_input)

    l0_weight = torch.randn((hidden_size, hidden_size), dtype=dtype)
    with MPIPoolExecutor(max_workers=tensor_parallel_size) as executor:
        results = executor.map(
            run_moe_single_rank,
            *zip(*[(tensor_parallel_size, moe_residual_norm_fusion_forward,
                    token_input, residual, active_experts_token_input, scale,
                    l0_weight)] * tensor_parallel_size),
        )
        for r in results:
            assert r is True
