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
"""Marlin weight repacking and scale processing utilities.

Ported from vLLM's marlin_utils.py and marlin_utils_fp4.py for use with
the vLLM-style Marlin NVFP4 kernels in TensorRT-LLM.
"""

import torch

GPTQ_MARLIN_TILE = 16
MARLIN_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]
FP4_MARLIN_SUPPORTED_GROUP_SIZES = [16]
USE_FP32_REDUCE_DEFAULT = True


def get_scale_perms():
    """Get the permutation indices for Marlin scale layout."""
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def marlin_permute_scales(
    s: torch.Tensor, size_k: int, size_n: int, group_size: int, is_a_8bit: bool = False
) -> torch.Tensor:
    """Permute scale tensor from [num_groups, N] to Marlin interleaved layout."""
    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1 and not is_a_8bit:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()
    return s


def marlin_permute_bias(s: torch.Tensor) -> torch.Tensor:
    """Permute bias to match Marlin kernel layout."""
    origin_shape = s.shape
    _, scale_perm_single = get_scale_perms()
    s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return s.reshape(*origin_shape).contiguous()


def marlin_make_workspace(device: torch.device, max_blocks_per_sm: int = 1) -> torch.Tensor:
    """Allocate int32 workspace tensor sized for grid parallelism."""
    props = torch.cuda.get_device_properties(device)
    sms = props.multi_processor_count
    return torch.zeros(sms * max_blocks_per_sm, dtype=torch.int, device=device, requires_grad=False)


def nvfp4_marlin_process_scales(marlin_scales: torch.Tensor) -> torch.Tensor:
    """Convert FP8-S1E4M3 scales to special FP8-S0E5M3 format for fast dequant.

    This assumes scales are non-negative. The conversion multiplies by 2^7 and
    left-shifts by 1 to create a format where the top bit is always 1 when
    scale > 0, allowing the kernel to use an exponent bias closer to zero.
    """
    # Convert to half for manipulation
    marlin_scales = marlin_scales.to(torch.half)

    # Fit the layout of fp8 dequantization
    marlin_scales = marlin_scales.view(-1, 4)[:, [0, 2, 1, 3]].view(marlin_scales.size(0), -1)

    # Convert to S0E5M3 format
    marlin_scales = (marlin_scales * (2**7)).view(torch.int16) << 1
    marlin_scales = marlin_scales.view(torch.float8_e4m3fn)
    marlin_scales = marlin_scales[:, 1::2].contiguous()

    return marlin_scales


def nvfp4_marlin_process_global_scale(global_scale: torch.Tensor) -> torch.Tensor:
    """Adjust global scale with exponent bias for BF16/FP16 dequantization."""
    assert global_scale.dtype in [torch.half, torch.bfloat16]
    fp4_exponent = 2
    if global_scale.dtype == torch.half:
        target_exponent = 5
    elif global_scale.dtype == torch.bfloat16:
        target_exponent = 8
    exponent_bias = 2 ** (target_exponent - 1) - 2 ** (fp4_exponent - 1)
    return global_scale * (2.0 ** (exponent_bias - 7))


def prepare_nvfp4_moe_weights_for_marlin(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    hidden_size: int,
    intermediate_size_per_partition: int,
    num_experts: int,
    is_act_and_mul: bool,
    param_dtype: torch.dtype,
):
    """Repack NVFP4 MoE weights and scales to Marlin tiled format.

    Returns: (w13, w13_scale, w13_global_scale, w2, w2_scale, w2_global_scale)
    """
    GROUP_SIZE = 16
    K = hidden_size
    N = intermediate_size_per_partition
    device = w13.device

    perm = torch.empty(0, dtype=torch.int, device=device)

    def repack_weight(weight: torch.Tensor, name: str) -> torch.Tensor:
        tensor_list = []
        num_shards = 2 if is_act_and_mul else 1
        if "w13" in name:
            size_n, size_k = N * num_shards, K
        else:
            size_n, size_k = K, N

        assert weight.shape == (num_experts, size_n, size_k // 2), (
            f"Expected {(num_experts, size_n, size_k // 2)}, got {weight.shape}"
        )

        for i in range(num_experts):
            qweight = weight[i].view(torch.int32).T.contiguous()
            marlin_qweight = torch.ops.trtllm.gptq_marlin_repack(
                b_q_weight=qweight,
                perm=perm,
                size_k=size_k,
                size_n=size_n,
                num_bits=4,
                is_a_8bit=False,
            )
            tensor_list.append(marlin_qweight)
        return torch.cat([x.unsqueeze(0) for x in tensor_list], 0)

    def permute_scales(scales: torch.Tensor, g_scales: torch.Tensor, name: str):
        scales = scales.to(param_dtype)
        g_scales = g_scales.to(param_dtype)

        tensor_list = []
        num_shards = 2 if is_act_and_mul else 1
        if "w13" in name:
            size_n, size_k = N * num_shards, K
        else:
            size_n, size_k = K, N

        for i in range(num_experts):
            scale = scales[i].T
            marlin_scales = marlin_permute_scales(
                s=scale,
                size_k=size_k,
                size_n=size_n,
                group_size=GROUP_SIZE,
                is_a_8bit=False,
            )
            marlin_scales = nvfp4_marlin_process_scales(marlin_scales)
            tensor_list.append(marlin_scales)

        scales = torch.cat([x.unsqueeze(0) for x in tensor_list], 0)
        g_scales = nvfp4_marlin_process_global_scale(g_scales)
        return scales, g_scales

    w13 = repack_weight(w13, "w13")
    w2 = repack_weight(w2, "w2")

    w13_scale, w13_global_scale = permute_scales(w13_scale, w13_global_scale, "w13")
    w2_scale, w2_global_scale = permute_scales(w2_scale, w2_global_scale, "w2")

    return w13, w13_scale, w13_global_scale, w2, w2_scale, w2_global_scale
