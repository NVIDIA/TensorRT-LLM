# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from .._utils import get_sm_version


def preprocess_weights_for_mixed_gemm(
    tensor: torch.Tensor,
    quant_mode: torch.dtype,
    act_dtype: torch.dtype,
    sm_: int = -1,
    do_weight_interleave: bool = True,
) -> torch.Tensor:
    sm_ = sm_ if sm_ > 0 else get_sm_version()
    # 3-D inputs (MoE) on Hopper+ and any input on SM120/SM121 reuse the SM80
    # interleaved layout. Check the original rank before unsqueeze.
    if (len(tensor.shape) == 3 and sm_ >= 90) or sm_ >= 120:
        sm_ = 80
    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0)
    if sm_ == 100 or sm_ == 103:
        do_weight_interleave = False

    permutation_map = {
        "16_8": [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15],
        "16_4": [
            0,
            1,
            8,
            9,
            16,
            17,
            24,
            25,
            2,
            3,
            10,
            11,
            18,
            19,
            26,
            27,
            4,
            5,
            12,
            13,
            20,
            21,
            28,
            29,
            6,
            7,
            14,
            15,
            22,
            23,
            30,
            31,
        ],
        "8_4": [
            0,
            1,
            2,
            3,
            16,
            17,
            18,
            19,
            4,
            5,
            6,
            7,
            20,
            21,
            22,
            23,
            8,
            9,
            10,
            11,
            24,
            25,
            26,
            27,
            12,
            13,
            14,
            15,
            28,
            29,
            30,
            31,
        ],
    }

    # permute_B_rows_for_mixed_gemm
    BITS_PER_ELT_A = 8 if act_dtype == torch.float8_e4m3fn else 16
    BITS_PER_ELT_B = 4 if quant_mode == torch.quint4x2 else 8
    MMA_SHAPE_N = 8
    B_ROWS_PER_MMA = 8 * 16 // BITS_PER_ELT_B

    num_experts = tensor.shape[0]
    num_rows = tensor.shape[1]
    num_cols = tensor.shape[2]

    assert sm_ >= 75
    assert num_rows % B_ROWS_PER_MMA == 0
    assert num_cols % MMA_SHAPE_N == 0

    if do_weight_interleave and sm_ < 100:
        row_idx_list = [(row_idx // B_ROWS_PER_MMA) * B_ROWS_PER_MMA +
                        permutation_map[f"{BITS_PER_ELT_A}_{BITS_PER_ELT_B}"][
                            row_idx % B_ROWS_PER_MMA]
                        for row_idx in range(num_rows)]
        tensor = tensor[:, row_idx_list, :]

    # subbyte_transpose
    original_shape = tensor.shape
    if BITS_PER_ELT_B == 4:
        tensor = tensor.view(torch.uint8)
        high_tensor = (tensor >> 4).permute(0, 2, 1).unsqueeze(2)
        low_tensor = ((tensor << 4) >> 4).permute(0, 2, 1).unsqueeze(2)
        new_tensor = torch.cat([low_tensor, high_tensor],
                               dim=2).reshape(tensor.shape[0], -1,
                                              tensor.shape[1])
        new_tensor = new_tensor[:, :, 0::2] + new_tensor[:, :, 1::2] * 16
        tensor = new_tensor.view(torch.int8).reshape(original_shape)
    else:
        tensor = tensor.permute(0, 2, 1).reshape(original_shape)

    if do_weight_interleave:
        # interleave_column_major_tensor
        interleave = BITS_PER_ELT_A // BITS_PER_ELT_B
        if interleave > 1 and sm_ < 90:
            rows_per_tile = 128 * 8 // BITS_PER_ELT_A
            elts_in_int32 = 32 // BITS_PER_ELT_B

            assert num_rows % elts_in_int32 == 0
            assert num_rows % rows_per_tile == 0

            tensor = tensor.reshape(
                num_experts,
                -1,
                interleave,
                num_rows // rows_per_tile,
                rows_per_tile * 4 // elts_in_int32,
            )
            tensor = tensor.permute(0, 1, 3, 2, 4).reshape(original_shape)

        # add_bias_and_interleave_quantized_tensor_inplace
        if BITS_PER_ELT_B == 8:
            tensor += -256 * (tensor > 127).byte() + 128
            tensor = tensor.reshape(-1, 4)[:,
                                           [0, 2, 1, 3]].reshape(tensor.shape)
        elif BITS_PER_ELT_B == 4:
            tensor = tensor.view(torch.uint8)
            high_tensor = (tensor >> 4).unsqueeze(-1)
            low_tensor = ((tensor << 4) >> 4).unsqueeze(-1)
            new_tensor = torch.cat([low_tensor, high_tensor],
                                   dim=-1).reshape(tensor.shape[0],
                                                   tensor.shape[1], -1)
            new_tensor = new_tensor.reshape(
                -1, 8)[:, [0, 2, 4, 6, 1, 3, 5, 7]].reshape(new_tensor.shape)
            new_tensor += -16 * (new_tensor > 7).byte() + 8
            new_tensor = new_tensor[:, :, 0::2] + new_tensor[:, :, 1::2] * 16
            tensor = new_tensor.view(torch.int8)
        else:
            raise NotImplementedError

    return tensor.squeeze(0).contiguous()
