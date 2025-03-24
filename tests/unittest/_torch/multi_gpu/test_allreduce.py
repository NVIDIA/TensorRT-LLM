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
                                             AllReduceParams)
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm.mapping import Mapping

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

    def e2m1_and_ufp8sf_scale_to_float_v2(e2m1_tensor,
                                          ufp8_scale_tensor,
                                          global_scale_tensor,
                                          sf_vec_size,
                                          ufp8_type,
                                          is_sf_swizzled_layout=True):
        return torch.ops.tensorrt_llm.e2m1_and_ufp8sf_scale_to_float_v2(
            e2m1_tensor, ufp8_scale_tensor, global_scale_tensor, sf_vec_size,
            ufp8_type, is_sf_swizzled_layout)

    x = x.cuda()
    residual = residual.cuda()
    norm_weight = torch.randn((hidden_size, ), dtype=dtype, device="cuda")
    eps = 1e-5

    mapping = Mapping(
        world_size=tensor_parallel_size,
        tp_size=tensor_parallel_size,
        rank=tensor_parallel_rank,
    )
    linear = Linear(
        in_features=hidden_size,
        out_features=hidden_size,
        bias=False,
        dtype=dtype,
        mapping=mapping,
        tensor_parallel_mode=TensorParallelMode.ROW,
    ).cuda()
    norm = RMSNorm(hidden_size=hidden_size, eps=eps, dtype=dtype).cuda()

    allreduce = AllReduce(mapping=mapping).cuda()

    scale = torch.tensor(1.0, dtype=torch.float32).cuda()
    linear.load_weights([dict(weight=weights[0])])
    norm.weight.data.copy_(norm_weight)

    def calc_allreduce(x, res):
        linear_out = linear(x)
        return [linear_out]

    def calc_fused_allreduce(x, res):
        linear_out = linear(
            x, all_reduce_params=AllReduceParams(enable_allreduce=False))
        output = allreduce(
            linear_out,
            all_reduce_params=AllReduceParams(
                fusion_op=fusion_op,
                residual=res,
                norm_weight=norm_weight,
                scale=scale,
                bias=None,
                eps=eps,
            ),
        )
        return output

    def calc_residual_rms_norm_quant_fp8(x, res):
        quant_fp8, residual_out = calc_fused_allreduce(x, res)
        return dequant(quant_fp8, scale, dtype), residual_out

    def calc_residual_rms_norm_out_quant_fp8(x, res):
        norm_out, quant_fp8, residual_out = calc_fused_allreduce(x, res)
        return norm_out, dequant(quant_fp8, scale, dtype), residual_out

    def calc_residual_rms_norm_quant_nvfp4(x, res):
        quant_fp4, scale_factor, residual_out = calc_fused_allreduce(x, res)
        dequant_fp4 = e2m1_and_ufp8sf_scale_to_float_v2(quant_fp4.cpu(),
                                                        scale_factor.cpu(),
                                                        1 / scale.cpu(), 16, 1)
        return dequant_fp4, residual_out

    def calc_residual_rms_norm_out_quant_nvfp4(x, res):
        norm_out, quant_fp4, scale_factor, residual_out = calc_fused_allreduce(
            x, res)
        dequant_fp4 = e2m1_and_ufp8sf_scale_to_float_v2(quant_fp4.cpu(),
                                                        scale_factor.cpu(),
                                                        1 / scale.cpu(), 16, 1)
        return norm_out, dequant_fp4, residual_out

    def ref_allreduce(x, res):
        linear_out = linear(x)
        return [linear_out]

    def ref_residual_rms_norm(x, res):
        hidden_states = ref_allreduce(x, res)[0]
        residual_out = hidden_states + res
        residual_out = residual_out.to(torch.float32)
        norm_out = rms_norm(residual_out, norm_weight, eps)
        return norm_out.to(dtype), residual_out.to(dtype)

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
        quant_fp4, scale_factor = torch.ops.trtllm.fp4_quantize(
            norm_out.clone(), scale, 16, False)
        dequant_fp4 = e2m1_and_ufp8sf_scale_to_float_v2(quant_fp4.cpu(),
                                                        scale_factor.cpu(),
                                                        1 / scale.cpu(), 16, 1)
        return dequant_fp4, residual_out

    def ref_residual_rms_norm_out_quant_nvfp4(x, res):
        norm_out, residual_out = ref_residual_rms_norm(x, res)
        quant_fp4, scale_factor = torch.ops.trtllm.fp4_quantize(
            norm_out.clone(), scale, 16, False)
        dequant_fp4 = e2m1_and_ufp8sf_scale_to_float_v2(quant_fp4.cpu(),
                                                        scale_factor.cpu(),
                                                        1 / scale.cpu(), 16, 1)
        return norm_out, dequant_fp4, residual_out

    fusion_op_to_func = {
        AllReduceFusionOp.NONE: (calc_allreduce, ref_allreduce),
        AllReduceFusionOp.RESIDUAL_RMS_NORM: (calc_fused_allreduce,
                                              ref_residual_rms_norm),
        AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8:
        (calc_residual_rms_norm_quant_fp8, ref_residual_rms_norm_quant_fp8),
        AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_FP8:
        (calc_residual_rms_norm_out_quant_fp8,
         ref_residual_rms_norm_out_quant_fp8),
        AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4:
        (calc_residual_rms_norm_quant_nvfp4, ref_residual_rms_norm_quant_nvfp4),
        AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4:
        (calc_residual_rms_norm_out_quant_nvfp4,
         ref_residual_rms_norm_out_quant_nvfp4),
    }

    calc_func, ref_func = fusion_op_to_func[fusion_op]

    # common allreduce path
    xs = torch.chunk(x.clone(), tensor_parallel_size, dim=-1)
    calc_output = calc_func(xs[tensor_parallel_rank], residual)
    ref_output = ref_func(xs[tensor_parallel_rank], residual)

    for calc_output_tensor, ref_output_tensor in zip(calc_output, ref_output):
        rtol, atol = 0.05, 0.15
        try:
            torch.testing.assert_close(
                calc_output_tensor,
                ref_output_tensor,
                rtol=rtol,
                atol=atol,
            )
        except AssertionError:
            # Calculate percentage of mismatched elements
            mismatched = torch.abs(calc_output_tensor - ref_output_tensor) > (
                rtol * torch.abs(ref_output_tensor) + atol)
            mismatch_percentage = (mismatched.sum() / mismatched.numel())

            # If more than 1% elements mismatch, raise the error
            assert mismatch_percentage < 0.01, f"Large mismatched elements encountered"


@skip_pre_blackwell
@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Requires at least 2 GPUs for this test")
@pytest.mark.parametrize("seq_len", [16, 256], ids=lambda x: f"seqlen:{x}")
@pytest.mark.parametrize("hidden_size", [128, 7168],
                         ids=lambda x: f"hidden:{x}")
@pytest.mark.parametrize("fusion_op", [
    AllReduceFusionOp.NONE,
    AllReduceFusionOp.RESIDUAL_RMS_NORM,
    AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8,
    AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_FP8,
    AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4,
    AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4,
],
                         ids=[
                             "none",
                             "residual_rms_norm",
                             "residual_rms_norm_quant_fp8",
                             "residual_rms_norm_out_quant_fp8",
                             "residual_rms_norm_quant_nvfp4",
                             "residual_rms_norm_out_quant_nvfp4",
                         ])
def test_allreduce_fusion_patterns(seq_len, hidden_size, fusion_op):
    torch.manual_seed(0)
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
