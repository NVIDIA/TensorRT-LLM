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
from utils.util import check_accuracy, skip_pre_blackwell

import tensorrt_llm
from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.distributed import (AllReduce, AllReduceFusionOp,
                                             AllReduceParams, AllReduceStrategy,
                                             MiniMaxAllReduceRMS, MoEAllReduce,
                                             MoEAllReduceParams)
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

# needed since we reuse the mpi executor pool, first test running will leak a thread
pytestmark = pytest.mark.threadleak(enabled=False)


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


def run_single_rank(
    tensor_parallel_size,
    single_rank_forward_func,
    input,
    residual,
    weights,
    hidden_size,
    dtype,
    fusion_op,
):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        single_rank_forward_func(input, residual, hidden_size, dtype,
                                 tensor_parallel_size, rank, weights, fusion_op)
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
def run_allreduce_op(
    x: torch.Tensor,
    residual: torch.Tensor,
    hidden_size: int,
    dtype: torch.dtype,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
    weights: torch.Tensor,
    fusion_op: AllReduceFusionOp,
):

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

    AutoTuner.get().setup_distributed_state(mapping)
    linear = Linear(
        in_features=hidden_size,
        out_features=hidden_size,
        bias=False,
        dtype=dtype,
        mapping=mapping,
        tensor_parallel_mode=TensorParallelMode.ROW,
        allreduce_strategy=AllReduceStrategy.NCCL_SYMMETRIC,
    ).cuda()
    allreduce = AllReduce(mapping=mapping)
    norm = RMSNorm(hidden_size=hidden_size, eps=eps, dtype=dtype).cuda()

    allreduce = AllReduce(mapping=mapping).cuda()

    scale = torch.tensor(1.0, dtype=torch.float32).cuda()
    linear.load_weights([dict(weight=weights[0])])
    norm.weight.data.copy_(norm_weight)

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
        return [output] if fusion_op in (AllReduceFusionOp.NONE,
                                         AllReduceFusionOp.RMS_NORM) else output

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

    def ref_rms_norm(x, res):
        hidden_states = ref_allreduce(x, res)[0]
        hidden_states = hidden_states.to(torch.float32)
        norm_out = rms_norm(hidden_states, norm_weight, eps)
        return [norm_out.to(dtype)]

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
        AllReduceFusionOp.NONE: (calc_fused_allreduce, ref_allreduce),
        AllReduceFusionOp.RMS_NORM: (calc_fused_allreduce, ref_rms_norm),
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

    # trigger autotune
    with autotune():
        calc_output = calc_func(xs[tensor_parallel_rank], residual)

    ref_output = ref_func(xs[tensor_parallel_rank], residual)

    for calc_output_tensor, ref_output_tensor in zip(calc_output, ref_output):
        check_accuracy(calc_output_tensor,
                       ref_output_tensor,
                       atol=0.05,
                       rtol=0.15,
                       percent=0.99)


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Requires at least 2 GPUs for this test")
@pytest.mark.parametrize("seq_len", [16, 256, 8192],
                         ids=lambda x: f"seqlen:{x}")
@pytest.mark.parametrize("hidden_size", [128, 7168],
                         ids=lambda x: f"hidden:{x}")
@pytest.mark.parametrize(
    "fusion_op",
    [
        pytest.param(AllReduceFusionOp.NONE, id="none"),
        pytest.param(AllReduceFusionOp.RMS_NORM, id="rms_norm"),
        pytest.param(AllReduceFusionOp.RESIDUAL_RMS_NORM,
                     id="residual_rms_norm"),
        pytest.param(AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8,
                     id="residual_rms_norm_quant_fp8"),
        pytest.param(AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_FP8,
                     id="residual_rms_norm_out_quant_fp8"),
        pytest.param(AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4,
                     id="residual_rms_norm_quant_nvfp4",
                     marks=skip_pre_blackwell),
        pytest.param(AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4,
                     id="residual_rms_norm_out_quant_nvfp4",
                     marks=skip_pre_blackwell),
    ],
)
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_allreduce_fusion_patterns(seq_len, hidden_size, fusion_op,
                                   mpi_pool_executor):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    tensor_parallel_size = mpi_pool_executor.num_workers
    x = torch.randn((seq_len, hidden_size), dtype=dtype)
    residual = torch.randn_like(x)
    linear_weight = torch.randn((hidden_size, hidden_size), dtype=dtype)
    results = mpi_pool_executor.map(
        run_single_rank,
        *zip(*[(tensor_parallel_size, run_allreduce_op, x, residual,
                [linear_weight], hidden_size, dtype, fusion_op)] *
             tensor_parallel_size),
    )
    for r in results:
        assert r is True


@torch.inference_mode()
def run_moe_allreduce_op(token_input: torch.Tensor, residual: torch.Tensor,
                         active_experts_token_input: torch.Tensor,
                         scale: torch.Tensor, tensor_parallel_size: int,
                         tensor_parallel_rank: int, l0_weight: torch.Tensor):
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

    # Initialize MoEAllreduce
    moe_allreduce = MoEAllReduce(mapping=Mapping(
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

    moe_all_reduce_params = MoEAllReduceParams(
        residual=residual,
        norm_weight=norm_weight,
        device_num_experts=tensor_num_device_experts,
        expert_scale_factor=scale_equalized,
        shared_expert_output=fc2_output,
        eps=eps,
        is_cutlass_min_latency=True,
    )

    # Run with fusion
    output_hidden_states, output_residual = moe_allreduce(
        active_experts_token_equalized, all_reduce_params=moe_all_reduce_params)

    torch_l0 = torch.nn.Linear(in_features=hidden_size,
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
    torch_before_residual = expert_reduction + torch_linear_output
    torch_residual = torch_before_residual + residual
    torch_residual = torch_residual.to(torch.float32)
    torch_output_hidden_states = rms_norm(torch_residual, norm_weight,
                                          eps).to(dtype)

    # Verify results are close to reference
    torch.testing.assert_close(
        output_hidden_states,
        torch_output_hidden_states,
        rtol=0.2,
        atol=0.2,
    )

    return True


@torch.inference_mode()
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_moe_allreduce_patterns(mpi_pool_executor):
    torch.manual_seed(42)

    seq_len = 16
    hidden_size = 7168
    dtype = torch.bfloat16
    tensor_parallel_size = mpi_pool_executor.num_workers
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
    results = mpi_pool_executor.map(
        run_moe_single_rank,
        *zip(*[(tensor_parallel_size, run_moe_allreduce_op, token_input,
                residual, active_experts_token_input, scale, l0_weight)] *
             tensor_parallel_size),
    )
    for r in results:
        assert r is True


def run_moe_finalize_single_rank(tensor_parallel_size, single_rank_forward_func,
                                 fc2_output, residual, shared_expert_output,
                                 expanded_idx_to_permuted_idx, scale):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        single_rank_forward_func(fc2_output, residual, shared_expert_output,
                                 expanded_idx_to_permuted_idx, scale, rank,
                                 tensor_parallel_size)
    except Exception:
        traceback.print_exc()
        raise
    return True


@torch.inference_mode()
def run_moe_finalize_allreduce_op(
        fc2_output: torch.Tensor, residual: torch.Tensor,
        shared_expert_output: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor, scale: torch.Tensor,
        tensor_parallel_rank: int, tensor_parallel_size: int):
    torch.manual_seed(42)

    fc2_output = fc2_output.cuda()
    residual = residual.cuda()
    shared_expert_output = shared_expert_output.cuda()
    expanded_idx_to_permuted_idx = expanded_idx_to_permuted_idx.cuda()
    scale = scale.cuda()

    dtype = fc2_output.dtype
    hidden_size = residual.shape[1]

    # Setup parameters
    eps = 1e-5
    norm_weight = torch.randn((hidden_size, ), dtype=dtype, device="cuda")

    # Initialize MoEAllreduce
    moe_allreduce = MoEAllReduce(mapping=Mapping(
        world_size=tensor_parallel_size,
        tp_size=tensor_parallel_size,
        rank=tensor_parallel_rank,
    ))

    # Initialize RMSNorm
    norm = RMSNorm(hidden_size=hidden_size, eps=eps, dtype=dtype).cuda()
    norm.weight.data.copy_(norm_weight)

    moe_all_reduce_params = MoEAllReduceParams(
        expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
        expert_scale_factor=scale,
        shared_expert_output=shared_expert_output,
        residual=residual,
        norm_weight=norm_weight,
        eps=eps,
        is_cutlass_min_latency=False,
    )

    # Run with fusion
    output_hidden_states, output_residual = moe_allreduce(
        fc2_output, all_reduce_params=moe_all_reduce_params)

    # Verify with torch reference implementation
    expert_reduction = torch.sum(fc2_output[expanded_idx_to_permuted_idx] *
                                 scale.unsqueeze(-1),
                                 dim=1)

    torch_before_residual = (expert_reduction +
                             shared_expert_output) * tensor_parallel_size
    torch_residual = torch_before_residual + residual
    torch_residual = torch_residual.to(torch.float32)
    torch_output_hidden_states = rms_norm(torch_residual, norm_weight,
                                          eps).to(dtype)

    # Verify results are close to reference
    torch.testing.assert_close(
        output_hidden_states,
        torch_output_hidden_states,
        rtol=0.2,
        atol=0.2,
    )

    return True


@torch.inference_mode()
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_moe_finalize_allreduce_patterns(mpi_pool_executor):
    torch.manual_seed(42)

    seq_len = 16
    hidden_size = 7168
    dtype = torch.bfloat16
    tensor_parallel_size = mpi_pool_executor.num_workers
    top_k = 8

    shared_expert_output = torch.randn((seq_len, hidden_size), dtype=dtype)
    fc2_output = torch.randn((seq_len * top_k, hidden_size), dtype=dtype)
    scale = torch.randn((seq_len, top_k), dtype=dtype)
    expanded_idx_to_permuted_idx = torch.randint(0,
                                                 seq_len * top_k,
                                                 (seq_len, top_k),
                                                 dtype=torch.int32)
    residual = torch.randn_like(shared_expert_output)

    results = mpi_pool_executor.map(
        run_moe_finalize_single_rank,
        *zip(*[(tensor_parallel_size, run_moe_finalize_allreduce_op, fc2_output,
                residual, shared_expert_output, expanded_idx_to_permuted_idx,
                scale)] * tensor_parallel_size),
    )
    for r in results:
        assert r is True


def run_moe_finalize_no_residual_single_rank(tensor_parallel_size,
                                             single_rank_forward_func,
                                             fc2_output, shared_expert_output,
                                             expanded_idx_to_permuted_idx,
                                             scale):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        single_rank_forward_func(fc2_output, shared_expert_output,
                                 expanded_idx_to_permuted_idx, scale, rank,
                                 tensor_parallel_size)
    except Exception:
        traceback.print_exc()
        raise
    return True


@torch.inference_mode()
def run_moe_finalize_allreduce_no_residual_op(
        fc2_output: torch.Tensor, shared_expert_output: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor, scale: torch.Tensor,
        tensor_parallel_rank: int, tensor_parallel_size: int):
    torch.manual_seed(42)

    fc2_output = fc2_output.cuda()
    shared_expert_output = shared_expert_output.cuda()
    expanded_idx_to_permuted_idx = expanded_idx_to_permuted_idx.cuda()
    scale = scale.cuda()

    dtype = fc2_output.dtype
    hidden_size = shared_expert_output.shape[1]

    # Setup parameters
    eps = 1e-5
    norm_weight = torch.randn((hidden_size, ), dtype=dtype, device="cuda")

    # Initialize MoEAllreduce
    moe_allreduce = MoEAllReduce(mapping=Mapping(
        world_size=tensor_parallel_size,
        tp_size=tensor_parallel_size,
        rank=tensor_parallel_rank,
    ))

    moe_all_reduce_params = MoEAllReduceParams(
        expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
        expert_scale_factor=scale,
        shared_expert_output=shared_expert_output,
        residual=None,
        norm_weight=norm_weight,
        eps=eps,
        is_cutlass_min_latency=False,
    )

    # Run with fusion (no residual)
    (output_hidden_states, ) = moe_allreduce(
        fc2_output, all_reduce_params=moe_all_reduce_params)

    # Verify with torch reference implementation (no residual addition)
    expert_reduction = torch.sum(fc2_output[expanded_idx_to_permuted_idx] *
                                 scale.unsqueeze(-1),
                                 dim=1)

    torch_sum = (expert_reduction + shared_expert_output) * tensor_parallel_size
    torch_sum = torch_sum.to(torch.float32)
    torch_output_hidden_states = rms_norm(torch_sum, norm_weight, eps).to(dtype)

    # Verify results are close to reference
    torch.testing.assert_close(
        output_hidden_states,
        torch_output_hidden_states,
        rtol=0.2,
        atol=0.2,
    )

    return True


@torch.inference_mode()
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_moe_finalize_allreduce_no_residual(mpi_pool_executor):
    torch.manual_seed(42)

    seq_len = 16
    hidden_size = 7168
    dtype = torch.bfloat16
    tensor_parallel_size = mpi_pool_executor.num_workers
    top_k = 8

    shared_expert_output = torch.randn((seq_len, hidden_size), dtype=dtype)
    fc2_output = torch.randn((seq_len * top_k, hidden_size), dtype=dtype)
    scale = torch.randn((seq_len, top_k), dtype=dtype)
    expanded_idx_to_permuted_idx = torch.randint(0,
                                                 seq_len * top_k,
                                                 (seq_len, top_k),
                                                 dtype=torch.int32)

    results = mpi_pool_executor.map(
        run_moe_finalize_no_residual_single_rank,
        *zip(*[(tensor_parallel_size, run_moe_finalize_allreduce_no_residual_op,
                fc2_output, shared_expert_output, expanded_idx_to_permuted_idx,
                scale)] * tensor_parallel_size),
    )
    for r in results:
        assert r is True


@torch.inference_mode()
def run_minimax_allreduce_rms_op(input: torch.Tensor, tensor_parallel_size: int,
                                 tensor_parallel_rank: int,
                                 rms_weights: torch.Tensor, eps: float):
    torch.manual_seed(42)

    total_tokens = input.shape[0]
    origin_dtype = input.dtype

    input = input.cuda()
    rms_weights = rms_weights.cuda()

    input = input.reshape(total_tokens, tensor_parallel_size,
                          -1).to(torch.float32)
    rms_weights = rms_weights.reshape(tensor_parallel_size,
                                      -1).to(torch.float32)
    rank_input = input[:, tensor_parallel_rank, :].contiguous()
    rank_rms_weights = rms_weights[tensor_parallel_rank, :].contiguous()
    # firstly, calculate the reference output for each rank
    ref_output = rms_norm(input, rms_weights, eps)
    ref_output = ref_output.reshape(total_tokens, tensor_parallel_size,
                                    -1).to(origin_dtype)
    ref_output = ref_output[:, tensor_parallel_rank, :]

    # then, calculate the minimax allreduce output
    minimax_allreduce_rms = MiniMaxAllReduceRMS(mapping=Mapping(
        world_size=tensor_parallel_size,
        tp_size=tensor_parallel_size,
        rank=tensor_parallel_rank,
    ))
    minimax_output = minimax_allreduce_rms(
        input=rank_input.to(torch.bfloat16),
        rms_weights=rank_rms_weights.to(torch.bfloat16),
        eps=eps,
    )
    # finally, verify the results
    torch.testing.assert_close(minimax_output, ref_output, rtol=0.2, atol=0.2)

    return rank_input


@torch.inference_mode()
def run_minimax_allreduce_rms_single_rank(tensor_parallel_size,
                                          single_rank_forward_func, input,
                                          rms_weights, eps):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        single_rank_forward_func(input, tensor_parallel_size, rank, rms_weights,
                                 eps)
    except Exception:
        traceback.print_exc()
        raise
    return True


@torch.inference_mode()
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_minimax_allreduce_rms(mpi_pool_executor):
    torch.manual_seed(42)

    seq_len = 16
    hidden_size = 6144
    dtype = torch.bfloat16
    tensor_parallel_size = mpi_pool_executor.num_workers

    token_input = torch.randn((seq_len, hidden_size), dtype=dtype)
    rms_weights = torch.randn((hidden_size, ), dtype=dtype, device="cuda")
    eps = 1e-5

    results = mpi_pool_executor.map(
        run_minimax_allreduce_rms_single_rank,
        *zip(*[(tensor_parallel_size, run_minimax_allreduce_rms_op, token_input,
                rms_weights, eps)] * tensor_parallel_size),
    )
    for r in results:
        assert r is True


@torch.inference_mode()
def run_minimax_allreduce_rms_qk_op(q_input: torch.Tensor,
                                    k_input: torch.Tensor,
                                    tensor_parallel_size: int,
                                    tensor_parallel_rank: int,
                                    rms_weights_q: torch.Tensor,
                                    rms_weights_k: torch.Tensor, eps: float):
    torch.manual_seed(42)

    num_tokens = q_input.shape[0]
    origin_dtype = q_input.dtype

    q_input = q_input.cuda()
    k_input = k_input.cuda()
    rms_weights_q = rms_weights_q.cuda()
    rms_weights_k = rms_weights_k.cuda()

    # firstly, calculate the reference output for each rank
    # Reference: each rank computes RMS norm on its local Q/K independently,
    # then all-reduce is applied to the variance before normalization
    # The all-reduce sum happens across TP ranks, so we need to simulate that
    q_input_fp32 = q_input.to(torch.float32)
    k_input_fp32 = k_input.to(torch.float32)

    # Compute reference: RMS norm with all-reduced variance
    q_variance_sum = q_input_fp32.pow(2).mean(-1, keepdim=True)
    k_variance_sum = k_input_fp32.pow(2).mean(-1, keepdim=True)

    ref_q_output = q_input_fp32 * torch.rsqrt(q_variance_sum + eps)
    ref_k_output = k_input_fp32 * torch.rsqrt(k_variance_sum + eps)

    # Apply weights
    ref_q_output = ref_q_output * rms_weights_q.to(torch.float32)
    ref_k_output = ref_k_output * rms_weights_k.to(torch.float32)

    ref_q_output = ref_q_output.to(origin_dtype)
    ref_k_output = ref_k_output.to(origin_dtype)

    # we only need to compare the reference output of the current rank
    ref_q_output = ref_q_output.reshape(num_tokens, tensor_parallel_size, -1)
    ref_k_output = ref_k_output.reshape(num_tokens, tensor_parallel_size, -1)
    ref_q_output = ref_q_output[:, tensor_parallel_rank, :].contiguous()
    ref_k_output = ref_k_output[:, tensor_parallel_rank, :].contiguous()

    # minimax input should be sliced by rank
    q_input = q_input.reshape(num_tokens, tensor_parallel_size, -1)
    k_input = k_input.reshape(num_tokens, tensor_parallel_size, -1)
    rank_q_input = q_input[:, tensor_parallel_rank, :].contiguous()
    rank_k_input = k_input[:, tensor_parallel_rank, :].contiguous()

    # rms weights should be sliced by rank
    rms_weights_q = rms_weights_q.reshape(tensor_parallel_size, -1)
    rms_weights_k = rms_weights_k.reshape(tensor_parallel_size, -1)
    rank_rms_weights_q = rms_weights_q[tensor_parallel_rank, :].contiguous()
    rank_rms_weights_k = rms_weights_k[tensor_parallel_rank, :].contiguous()

    # then, calculate the minimax allreduce output
    minimax_allreduce_rms = MiniMaxAllReduceRMS(mapping=Mapping(
        world_size=tensor_parallel_size,
        tp_size=tensor_parallel_size,
        rank=tensor_parallel_rank,
    ))
    minimax_q_output, minimax_k_output = minimax_allreduce_rms.forward_qk(
        q=rank_q_input,
        k=rank_k_input,
        rms_weights_q=rank_rms_weights_q,
        rms_weights_k=rank_rms_weights_k,
        eps=eps,
    )

    # finally, verify the results
    torch.testing.assert_close(minimax_q_output,
                               ref_q_output,
                               rtol=0.2,
                               atol=0.2)
    torch.testing.assert_close(minimax_k_output,
                               ref_k_output,
                               rtol=0.2,
                               atol=0.2)

    return q_input


@torch.inference_mode()
def run_minimax_allreduce_rms_qk_single_rank(tensor_parallel_size,
                                             single_rank_forward_func, q_input,
                                             k_input, rms_weights_q,
                                             rms_weights_k, eps):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        single_rank_forward_func(q_input, k_input, tensor_parallel_size, rank,
                                 rms_weights_q, rms_weights_k, eps)
    except Exception:
        traceback.print_exc()
        raise
    return True


@pytest.mark.parametrize("mpi_pool_executor", [4], indirect=True)
def test_minimax_allreduce_rms_qk(mpi_pool_executor):
    torch.manual_seed(42)

    seq_len = 1024
    q_size = 6144
    k_size = 1024
    dtype = torch.bfloat16
    tensor_parallel_size = mpi_pool_executor.num_workers

    q_input = torch.randn((seq_len, q_size), dtype=dtype)
    k_input = torch.randn((seq_len, k_size), dtype=dtype)
    rms_weights_q = torch.randn((q_size, ), dtype=dtype, device="cuda")
    rms_weights_k = torch.randn((k_size, ), dtype=dtype, device="cuda")
    eps = 1e-5

    results = mpi_pool_executor.map(
        run_minimax_allreduce_rms_qk_single_rank,
        *zip(*[(tensor_parallel_size, run_minimax_allreduce_rms_qk_op, q_input,
                k_input, rms_weights_q, rms_weights_k, eps)] *
             tensor_parallel_size),
    )
    for r in results:
        assert r is True
