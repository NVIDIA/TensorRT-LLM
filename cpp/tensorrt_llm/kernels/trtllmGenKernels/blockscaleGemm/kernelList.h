/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "cubin/GemmKernel_Bfloat16_E4m3_Fp32_tile128x128x128_epilogueTile128x64_mma128x64x32_cluster1x1x1_dsFp8_sm100a_cubin.h"
#include "cubin/GemmKernel_E4m3_E4m3_Fp32_tile128x128x128_epilogueTile128x64_mma128x64x32_cluster1x1_dsFp8_sm100a_cubin.h"
#include "cubin/GemmKernel_Fp32_E4m3_Fp32_tile128x128x128_epilogueTile128x64_mma128x64x32_cluster1x1_dsFp8_sm100a_cubin.h"
#include "kernelParams.h"
#include <vector>

namespace tensorrt_llm
{
namespace kernels
{
struct TrtllmGenBlockScaleGemmInfo
{
    unsigned char const* data{nullptr};
    unsigned const size{0};
    unsigned const sharedMemSize{0};
    char const* functionName{nullptr};
    unsigned threadsPerCTA{0};
    int32_t tileM;
    int32_t tileN;
    int32_t tileK;
    int32_t numSlicesForSplitK;
    int32_t epilogueTileM;
    int32_t epilogueTileN;
    int32_t mmaM;
    int32_t mmaN;
    bool useTmaStore;
    int32_t numStages;
    Data_type dtypeElt;
    Data_type dtypeC;
    Data_type dtypeAcc;
    bool transposeMmaOutput;
    bool sliceK;
};

// clang-format off
const std::vector<TrtllmGenBlockScaleGemmInfo> trtllmGenBlockScaleGemmInfo {
    {
        GemmKernel_E4m3_E4m3_Fp32_tile128x128x128_epilogueTile128x64_mma128x64x32_cluster1x1_dsFp8_sm100a_cubin_data,
        GemmKernel_E4m3_E4m3_Fp32_tile128x128x128_epilogueTile128x64_mma128x64x32_cluster1x1_dsFp8_sm100a_cubin_len,
        155648,
        "gemmKernel_E4m3_E4m3_Fp32_tile128x128x128_epilogueTile128x64_mma128x64x32_cluster1x1x1_dsFp8_sm100a",
        352,
        128,
        128,
        128,
        1,
        128,
        64,
        128,
        64,
        true,
        4,
        Data_type::DATA_TYPE_E4M3,
        Data_type::DATA_TYPE_E4M3,
        Data_type::DATA_TYPE_FP32,
        false,
        false
    },
    {
        GemmKernel_Fp32_E4m3_Fp32_tile128x128x128_epilogueTile128x64_mma128x64x32_cluster1x1_dsFp8_sm100a_cubin_data,
        GemmKernel_Fp32_E4m3_Fp32_tile128x128x128_epilogueTile128x64_mma128x64x32_cluster1x1_dsFp8_sm100a_cubin_len,
        204800,
        "gemmKernel_Fp32_E4m3_Fp32_tile128x128x128_epilogueTile128x64_mma128x64x32_cluster1x1x1_dsFp8_sm100a",
        352,
        128,
        128,
        128,
        1,
        128,
        64,
        128,
        64,
        true,
        4,
        Data_type::DATA_TYPE_E4M3,
        Data_type::DATA_TYPE_FP32,
        Data_type::DATA_TYPE_FP32,
        false,
        false
    },
    {
        GemmKernel_Bfloat16_E4m3_Fp32_tile128x128x128_epilogueTile128x64_mma128x64x32_cluster1x1x1_dsFp8_sm100a_cubin_data,
        GemmKernel_Bfloat16_E4m3_Fp32_tile128x128x128_epilogueTile128x64_mma128x64x32_cluster1x1x1_dsFp8_sm100a_cubin_len,
        172032,
        "gemmKernel_Bfloat16_E4m3_Fp32_tile128x128x128_epilogueTile128x64_mma128x64x32_cluster1x1x1_dsFp8_sm100a",
        352,
        128,
        128,
        128,
        1,
        128,
        64,
        128,
        64,
        true,
        4,
        Data_type::DATA_TYPE_E4M3,
        Data_type::DATA_TYPE_BF16,
        Data_type::DATA_TYPE_FP32,
        false,
        false
    }
};
// clang-format on
} // namespace kernels
} // namespace tensorrt_llm
