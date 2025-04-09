# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import torch

import tensorrt_llm


def woq_torch_dtype(dtype):
    if dtype == "float16":
        torch_dtype = torch.half
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        assert False, f"{dtype} does not support WoQ"
    return torch_dtype


def woq_all_ones(n, k, dtype):
    torch_dtype = woq_torch_dtype(dtype)
    # Init operands for multiplication in int32
    weight = torch.ones((n, k), dtype=torch_dtype, device="cuda")
    return weight


def woq_all_zeros(n, k, dtype):
    torch_dtype = woq_torch_dtype(dtype)
    # Init operands for multiplication in int32
    weight = torch.zeros((n, k), dtype=torch_dtype, device="cuda")
    return weight


def woq_gen_weights(n, k, dtype):
    torch_dtype = woq_torch_dtype(dtype)
    # Init operands for multiplication in int32
    weight = torch.rand((n, k), dtype=torch_dtype, device="cuda") * 2 - 1.0
    return weight


def woq_conversion(weight, wTypeId):
    # only support int8 weight only
    if wTypeId == 1:
        torch_wTypeId = torch.int8
    elif wTypeId == 2:
        torch_wTypeId = torch.quint4x2
    else:
        assert False, f"wTypeId={wTypeId} is not supported by WoQ"
    return torch.ops.trtllm._symmetric_quantize_last_axis_of_batched_matrix(
        weight.cpu(), torch_wTypeId)


def woq_groupwise_gt_matmul(mat1, ref_torch_weights, bias=None):
    ref = torch.matmul(mat1, ref_torch_weights)
    if bias is not None:
        ref += bias
    return ref


def woq_gt_matmul(m,
                  mat1,
                  ref_torch_weights,
                  torch_weight_scales,
                  dtype,
                  bias=None):
    mat1 = mat1.to(dtype=torch.float)
    ref_torch_weights = ref_torch_weights.to(dtype=torch.float)
    # Do matmul
    ref = torch.matmul(mat1, ref_torch_weights)

    # Prepare per element scaling
    scaling = torch_weight_scales.expand((m, -1))
    # Scale output and cast to right type
    ref = ref * scaling

    # Round to the nearest int to match cuda rounding
    if dtype == "int32":
        ref = torch.round(ref)

    # Cast ref to the required output typy
    ref = ref.to(dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))

    if bias is not None:
        ref += bias

    return ref


def woq_assert_near_eq(ref, act, wTypeId):
    # match the scale in cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.cpp
    if wTypeId == 1:
        bits_in_type = 8
    else:
        bits_in_type = 4
    quant_range_scale = 1.0 / float(1 << (bits_in_type - 1))

    max_val = torch.max(abs(ref)).item()
    atol = (max_val * quant_range_scale) * 1.5  # allow for rounding
    torch.testing.assert_close(ref, act, atol=atol, rtol=1e-7)


def gt_matmul_smooth_quant(mat1, mat2, scale_a_, scale_b_, dtype, bias=None):
    # Convert to int32 for PyTorch GT Matmul with accumulation in int32.
    device = mat1.device
    mat1 = mat1.to(dtype=torch.int32).cpu()
    # Transpose the second matrix to support the native PyTorch format
    mat2 = mat2.transpose(0, 1).to(dtype=torch.int32).cpu()
    # Do matmul, int32 matmul must be in CPU. GPU does not support
    ref = torch.matmul(mat1, mat2)
    ref = ref.to(device)

    m = 1
    for ii in range(len(mat1.shape) - 1):
        m *= mat1.shape[ii]
    n = mat2.shape[1]

    # Prepare per element scaling
    scale_a = scale_a_.expand((m, 1)).float()
    scale_b = scale_b_.expand((1, n)).float()
    scaling = torch.matmul(scale_a, scale_b).reshape(ref.shape)
    # Scale output and cast to right type
    ref = ref * scaling

    # Round to the nearest int to match cuda rounding
    if dtype == "int32":
        ref = torch.round(ref)

    # Cast ref to the required output type
    ref = ref.to(dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))

    if bias is not None:
        ref += bias

    return ref


# Note: The qweight here should be torch Tensors before being processed by `qserve_convert_linear`
# i.e., without reordering and packing. There should be no packing and reordering for scales and zeros as well.
# The zeros here are scaled zeros
def gt_qserve_gemm_per_group(qact: torch.IntTensor,
                             act_scales: torch.HalfTensor,
                             qweight: torch.IntTensor,
                             s1_scales: torch.HalfTensor,
                             s2_scales: torch.IntTensor,
                             s2_szeros: torch.IntTensor,
                             group_size=128) -> torch.HalfTensor:
    out_features = qweight.shape[0]
    in_features = qweight.shape[1]

    # Step 1: Dequantize weight from int4 to int8
    s2_szeros = s2_szeros.reshape(out_features, in_features // group_size,
                                  1).to(qweight.device)
    s2_scales = s2_scales.reshape(out_features, in_features // group_size,
                                  1).to(qweight.device)

    assert qweight.dtype == torch.int8
    # The kernel relies on two's complement arithmetic of int8.
    # If qweight is converted to int32 the result will not match the kernel.
    dequantized_weight = qweight.reshape(
        out_features, in_features // group_size,
        group_size).mul(s2_scales).sub(s2_szeros)
    dequantized_weight = dequantized_weight.reshape(out_features, in_features)

    # Step 2: Perform matrix multiplication in int32
    result = torch.matmul(qact.to(torch.int32),
                          dequantized_weight.T.to(torch.int32))

    # Step 3: Dequantize the result to float
    # Convert int GEMM result, ascales and wscales all to float, which is aligned with the QServe GEMM kernel.
    result = result.float()
    s1_scales = s1_scales.reshape(1, out_features).to(result.device).float()
    act_scales = act_scales.reshape(act_scales.shape[0],
                                    1).to(result.device).float()
    # To match the result exactly to QServe, the multiplication order must be preserved due to float rounding errors.
    result = result.mul(s1_scales.mul(act_scales))
    return result.half()


def gt_qserve_gemm_per_channel(qact: torch.IntTensor,
                               act_scales: torch.HalfTensor,
                               act_sums: torch.HalfTensor,
                               qweight: torch.CharTensor,
                               s1_scales: torch.HalfTensor,
                               s1_szeros: torch.HalfTensor) -> torch.HalfTensor:
    out_features = qweight.shape[0]
    qweight.shape[1]
    num_activations = qact.shape[0]

    # Step 1: Perform matrix multiplication in int32
    result = torch.matmul(qact.to(torch.int32), qweight.T.to(torch.int32))

    # Step 2: Dequantize the result to float
    # Convert int GEMM result, ascales and wscales all to float, which is aligned with the QServe GEMM kernel.
    result = result.float()
    s1_scales = s1_scales.reshape(1, out_features).to(result.device).float()
    act_scales = act_scales.reshape(act_scales.shape[0],
                                    1).to(result.device).float()
    # To match the result exactly to QServe, the multiplication order must be preserved due to float rounding errors.
    result = result.mul(s1_scales.mul(act_scales))
    # Step 3: Add the outer product between act_sums and s1_szeros
    # Note: no unary minus before zeros like in per-channel version.
    act_sums = act_sums.reshape(num_activations, 1).to(result.device).float()
    s1_szeros = s1_szeros.reshape(1, out_features).to(result.device).float()
    result = result - act_sums * s1_szeros

    return result.half()


def gt_matmul_fp8_rowwise(mat1, mat2, scale_a_, scale_b_, dtype, bias=None):
    # Convert to float32 for PyTorch GT Matmul with accumulation in float32.
    device = mat1.device
    mat1 = mat1.to(dtype=torch.float32)
    # Transpose the second matrix to support the native PyTorch format
    mat2 = mat2.transpose(0, 1).to(dtype=torch.float32)
    # Do matmul, float32 matmul must be in CPU. GPU does not support
    ref = torch.matmul(mat1, mat2)
    ref = ref.to(device)

    m = 1
    for ii in range(len(mat1.shape) - 1):
        m *= mat1.shape[ii]
    n = mat2.shape[1]

    # Prepare per element scaling
    scale_a = scale_a_.expand((m, 1)).float()
    scale_b = scale_b_.expand((1, n)).float()
    scaling = torch.matmul(scale_a, scale_b).reshape(ref.shape)
    # Scale output and cast to right type
    ref = ref * scaling

    # Cast ref to the required output type
    ref = ref.to(dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))

    if bias is not None:
        ref += bias

    return ref


def gt_quantize_per_token(x):
    xmax, _ = x.abs().max(dim=-1, keepdim=True)
    x = (x * 127.0 / xmax).round().clip(-128, 127).to(dtype=torch.int8)
    scale_act = (xmax / 127.0).reshape(-1, 1)
    return x, scale_act
