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
from mpi4py.futures import MPIPoolExecutor
from utils.util import skip_pre_blackwell

import tensorrt_llm
from tensorrt_llm._torch.distributed import (AllReduce, AllReduceFusionOp,
                                             AllReduceParams, ParallelConfig,
                                             TensorParallelMode)
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.rms_norm import RMSNorm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)


def fp8_quant(input, scale):
    finfo = torch.finfo(torch.float8_e4m3fn)
    inv_scale = scale.reciprocal()
    qinput = (input.float() * inv_scale).clamp(min=finfo.min, max=finfo.max)
    return qinput.to(torch.float8_e4m3fn)


def dequant(input, scale, dtype):
    dqinput = input.to(torch.float32) * scale
    return dqinput.to(dtype)


def rms_norm(x: torch.Tensor, weight: torch.Tensor = None, eps: float = 1e-6):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight
    return y


def run_single_rank(tensor_parallel_size, single_rank_forward_func, input,
                    residual, weights, hidden_size, dtype, fusion_op):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        single_rank_forward_func(input, residual, hidden_size, dtype,
                                 tensor_parallel_size, rank, weights, fusion_op)
    except Exception:
        traceback.print_exc()
        raise
    return True


@torch.inference_mode()
def run_allreduce_op(x: torch.Tensor, residual: torch.Tensor, hidden_size: int,
                     dtype: torch.dtype, tensor_parallel_size: int,
                     tensor_parallel_rank: int, weights: torch.Tensor,
                     fusion_op: AllReduceFusionOp):

    x = x.cuda()
    residual = residual.cuda()
    norm_weight = torch.randn((hidden_size, ), dtype=dtype, device="cuda")
    eps = 1e-5

    parallel_config = ParallelConfig(
        tensor_parallel_size=tensor_parallel_size,
        tensor_parallel_rank=tensor_parallel_rank,
        tensor_parallel_mode=TensorParallelMode.ROW,
    )
    linear = Linear(
        in_features=hidden_size,
        out_features=hidden_size,
        bias=False,
        dtype=dtype,
        parallel_config=parallel_config,
    ).cuda()
    norm = RMSNorm(hidden_size=hidden_size, eps=eps, dtype=dtype).cuda()

    allreduce = AllReduce(parallel_config=parallel_config).cuda()

    scale = torch.tensor(1.0, dtype=torch.float32).cuda()
    linear.load_weights([dict(weight=weights[0])])
    norm.weight.data.copy_(norm_weight)

    xs = torch.chunk(x.clone(), tensor_parallel_size, dim=-1)
    linear_out = linear(
        xs[tensor_parallel_rank],
        all_reduce_params=AllReduceParams(enable_allreduce=False),
    )

    output = allreduce(
        linear_out,
        all_reduce_params=AllReduceParams(
            fusion_op=fusion_op,
            residual=residual,
            norm_weight=norm_weight,
            scale=scale,
            bias=None,
            eps=eps,
        ),
    )

    if fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8:
        output[0] = dequant(output[0], scale, dtype)
    if fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_FP8:
        output[1] = dequant(output[1], scale, dtype)

    def ref_residual_rms_norm(x, res):
        hidden_states = linear(x)
        residual_out = hidden_states + res
        residual_out = residual_out.to(torch.float32)
        norm_out = rms_norm(residual_out, norm_weight, eps)
        norm_out = norm_out.to(dtype)
        return norm_out, residual_out.to(dtype)

    def ref_residual_rms_norm_quant_fp8(x, res):
        norm_out, residual_out = ref_residual_rms_norm(x, res)
        return dequant(fp8_quant(norm_out.to(torch.float32), scale), scale,
                       dtype), residual_out

    def ref_residual_rms_norm_out_quant_fp8(x, res):
        norm_out, residual_out = ref_residual_rms_norm(x, res)
        return norm_out, dequant(fp8_quant(norm_out.to(torch.float32), scale),
                                 scale, dtype), residual_out

    def ref_residual_rms_norm_quant_nvfp4(x, res):
        norm_out, residual_out = ref_residual_rms_norm(x, res)
        act_fp4, act_sf = torch.ops.trtllm.fp4_quantize(norm_out.clone(), scale,
                                                        16, False)
        return act_fp4, act_sf, residual_out

    def ref_residual_rms_norm_out_quant_nvfp4(x, res):
        norm_out, residual_out = ref_residual_rms_norm(x, res)
        act_fp4, act_sf = torch.ops.trtllm.fp4_quantize(norm_out.clone(), scale,
                                                        16, False)
        return norm_out, act_fp4, act_sf, residual_out

    fusion_op_to_func = {
        AllReduceFusionOp.RESIDUAL_RMS_NORM:
        ref_residual_rms_norm,
        AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8:
        ref_residual_rms_norm_quant_fp8,
        AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_FP8:
        ref_residual_rms_norm_out_quant_fp8,
        AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4:
        ref_residual_rms_norm_quant_nvfp4,
        AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4:
        ref_residual_rms_norm_out_quant_nvfp4,
    }
    ref_func = fusion_op_to_func[fusion_op]

    # common allreduce path
    xs = torch.chunk(x.clone(), tensor_parallel_size, dim=-1)
    ref_output = ref_func(xs[tensor_parallel_rank], residual)

    for output_tensor, ref_output_tensor in zip(output, ref_output):
        torch.testing.assert_close(
            output_tensor,
            ref_output_tensor,
            rtol=0.05,
            atol=0.15,
        )


@skip_pre_blackwell
@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Requires at least 2 GPUs for this test")
@pytest.mark.parametrize("seq_len", [16, 256], ids=lambda x: f"seqlen:{x}")
@pytest.mark.parametrize("hidden_size", [7168], ids=lambda x: f"hidden:{x}")
@pytest.mark.parametrize("fusion_op", [
    AllReduceFusionOp.RESIDUAL_RMS_NORM,
    AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8,
    AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_FP8,
    AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4,
    AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4,
],
                         ids=[
                             "residual_rms_norm",
                             "residual_rms_norm_quant_fp8",
                             "residual_rms_norm_out_quant_fp8",
                             "residual_rms_norm_quant_nvfp4",
                             "residual_rms_norm_out_quant_nvfp4",
                         ])
def test_allreduce_fusion_patterns(seq_len, hidden_size, fusion_op):
    torch.manual_seed(42)
    dtype = torch.bfloat16
    tensor_parallel_size = 2
    x = torch.randn((seq_len, hidden_size), dtype=dtype)
    residual = torch.randn_like(x)
    linear_weight = torch.randn((hidden_size, hidden_size), dtype=dtype)
    with MPIPoolExecutor(max_workers=tensor_parallel_size) as executor:
        results = executor.map(
            run_single_rank,
            *zip(*[(tensor_parallel_size, run_allreduce_op, x, residual,
                    [linear_weight], hidden_size, dtype, fusion_op)] *
                 tensor_parallel_size),
        )
        for r in results:
            assert r is True
