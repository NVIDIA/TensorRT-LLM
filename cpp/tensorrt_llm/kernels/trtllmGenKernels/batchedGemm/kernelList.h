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

#include "cubin/BatchedGemmKernel_BatchM_E4m3Fp32_Bfloat16_Tile128x128x128_EpiTile128x128_Mma128x128x32_Cluster1x1x1_sm100a_cubin.h"
#include "cubin/BatchedGemmKernel_BatchM_E4m3Fp32_Bfloat16_Tile128x128x128_EpiTile128x64_Mma128x64x32_Cluster1x1x1_DsFp8_sm100a_cubin.h"
#include "cubin/BatchedGemmKernel_BatchM_E4m3Fp32_E4m3_Tile128x128x128_EpiTile128x128_Mma128x128x32_Cluster1x1x1_sm100a_cubin.h"
#include "cubin/BatchedGemmKernel_BatchM_E4m3Fp32_E4m3_Tile128x128x128_EpiTile128x64_Mma128x64x32_Cluster1x1x1_DsFp8_sm100a_cubin.h"
#include "cubin/BatchedGemmKernel_BatchN_E4m3Fp32_Bfloat16_Tile128x128x128_EpiTile128x128_Mma128x128x32_Cluster1x1x1_transposeMmaOutput_sm100a_cubin.h"
#include "cubin/BatchedGemmKernel_BatchN_E4m3Fp32_Bfloat16_Tile128x128x128_EpiTile64x128_Mma64x128x32_Cluster1x1x1_transposeMmaOutput_DsFp8_sm100a_cubin.h"
#include "cubin/BatchedGemmKernel_BatchN_E4m3Fp32_E4m3_Tile128x128x128_EpiTile128x128_Mma128x128x32_Cluster1x1x1_transposeMmaOutput_sm100a_cubin.h"
#include "cubin/BatchedGemmKernel_BatchN_E4m3Fp32_E4m3_Tile128x128x128_EpiTile64x128_Mma64x128x32_Cluster1x1x1_transposeMmaOutput_DsFp8_sm100a_cubin.h"
#include "tensorrt_llm/kernels/multiHeadAttentionCommon.h"

#include <vector>

namespace tensorrt_llm::kernels
{
enum class BatchMode
{
    BatchM,
    BatchN
};

struct TrtllmGenBatchedStridedGemmInfo
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
    int32_t mmaK;
    bool useTmaStore;
    int32_t numStages;
    Data_type dtypeElt;
    Data_type dtypeC;
    Data_type dtypeAcc;
    bool transposeMmaOutput;
    bool useDeepSeekFp8;
    bool useShuffledMatrixA;
    int32_t forceDqSfsA;
    int32_t forceDqSfsB;
    BatchMode batchMode;
    int32_t numBatches;
    int32_t numExperts;
    int32_t numTokens;
    int32_t expertCapacity;
    int32_t topK;
    std::vector<double> expertProbabilities;
};

// clang-format off
const std::vector<TrtllmGenBatchedStridedGemmInfo> trtllmGenBatchedStridedGemmInfo = {
    {
        BatchedGemmKernel_BatchN_E4m3Fp32_E4m3_Tile128x128x128_EpiTile128x128_Mma128x128x32_Cluster1x1x1_transposeMmaOutput_sm100a_cubin_data,
        BatchedGemmKernel_BatchN_E4m3Fp32_E4m3_Tile128x128x128_EpiTile128x128_Mma128x128x32_Cluster1x1x1_transposeMmaOutput_sm100a_cubin_len,
        151552,
        "BatchedGemmKernel_BatchN_E4m3Fp32_E4m3_Tile128x128x128_EpiTile128x128_Mma128x128x32_Cluster1x1x1_transposeMmaOutput_sm100a",
        192,
        128,
        128,
        128,
        1,
        128,
        128,
        128,
        128,
        32,
        true,
        4,
        Data_type::DATA_TYPE_E4M3,
        Data_type::DATA_TYPE_E4M3,
        Data_type::DATA_TYPE_FP32,
        true,
        false,
        true,
        0,
        0,
        BatchMode::BatchN,
        0,
        0,
        0,
        0,
        0,
        std::vector<double>{}
    },
    {
        BatchedGemmKernel_BatchN_E4m3Fp32_E4m3_Tile128x128x128_EpiTile64x128_Mma64x128x32_Cluster1x1x1_transposeMmaOutput_DsFp8_sm100a_cubin_data,
        BatchedGemmKernel_BatchN_E4m3Fp32_E4m3_Tile128x128x128_EpiTile64x128_Mma64x128x32_Cluster1x1x1_transposeMmaOutput_DsFp8_sm100a_cubin_len,
        155648,
        "BatchedGemmKernel_BatchN_E4m3Fp32_E4m3_Tile128x128x128_EpiTile64x128_Mma64x128x32_Cluster1x1x1_transposeMmaOutput_DsFp8_sm100a",
        352,
        128,
        128,
        128,
        1,
        64,
        128,
        64,
        128,
        32,
        true,
        4,
        Data_type::DATA_TYPE_E4M3,
        Data_type::DATA_TYPE_E4M3,
        Data_type::DATA_TYPE_FP32,
        true,
        true,
        true,
        0,
        0,
        BatchMode::BatchN,
        0,
        0,
        0,
        0,
        0,
        std::vector<double>{}
    },
    {
        BatchedGemmKernel_BatchM_E4m3Fp32_E4m3_Tile128x128x128_EpiTile128x128_Mma128x128x32_Cluster1x1x1_sm100a_cubin_data,
        BatchedGemmKernel_BatchM_E4m3Fp32_E4m3_Tile128x128x128_EpiTile128x128_Mma128x128x32_Cluster1x1x1_sm100a_cubin_len,
        151552,
        "BatchedGemmKernel_BatchM_E4m3Fp32_E4m3_Tile128x128x128_EpiTile128x128_Mma128x128x32_Cluster1x1x1_sm100a",
        192,
        128,
        128,
        128,
        1,
        128,
        128,
        128,
        128,
        32,
        true,
        4,
        Data_type::DATA_TYPE_E4M3,
        Data_type::DATA_TYPE_E4M3,
        Data_type::DATA_TYPE_FP32,
        false,
        false,
        false,
        0,
        0,
        BatchMode::BatchM,
        0,
        0,
        0,
        0,
        0,
        std::vector<double>{}
    },
    {
        BatchedGemmKernel_BatchM_E4m3Fp32_E4m3_Tile128x128x128_EpiTile128x64_Mma128x64x32_Cluster1x1x1_DsFp8_sm100a_cubin_data,
        BatchedGemmKernel_BatchM_E4m3Fp32_E4m3_Tile128x128x128_EpiTile128x64_Mma128x64x32_Cluster1x1x1_DsFp8_sm100a_cubin_len,
        155648,
        "BatchedGemmKernel_BatchM_E4m3Fp32_E4m3_Tile128x128x128_EpiTile128x64_Mma128x64x32_Cluster1x1x1_DsFp8_sm100a",
        352,
        128,
        128,
        128,
        1,
        128,
        64,
        128,
        64,
        32,
        true,
        4,
        Data_type::DATA_TYPE_E4M3,
        Data_type::DATA_TYPE_E4M3,
        Data_type::DATA_TYPE_FP32,
        false,
        true,
        false,
        0,
        0,
        BatchMode::BatchM,
        0,
        0,
        0,
        0,
        0,
        std::vector<double>{}
    },
    {
        BatchedGemmKernel_BatchN_E4m3Fp32_Bfloat16_Tile128x128x128_EpiTile128x128_Mma128x128x32_Cluster1x1x1_transposeMmaOutput_sm100a_cubin_data,
        BatchedGemmKernel_BatchN_E4m3Fp32_Bfloat16_Tile128x128x128_EpiTile128x128_Mma128x128x32_Cluster1x1x1_transposeMmaOutput_sm100a_cubin_len,
        167936,
        "BatchedGemmKernel_BatchN_E4m3Fp32_Bfloat16_Tile128x128x128_EpiTile128x128_Mma128x128x32_Cluster1x1x1_transposeMmaOutput_sm100a",
        192,
        128,
        128,
        128,
        1,
        128,
        128,
        128,
        128,
        32,
        true,
        4,
        Data_type::DATA_TYPE_E4M3,
        Data_type::DATA_TYPE_BF16,
        Data_type::DATA_TYPE_FP32,
        true,
        false,
        true,
        0,
        0,
        BatchMode::BatchN,
        0,
        0,
        0,
        0,
        0,
        std::vector<double>{}
    },
    {
        BatchedGemmKernel_BatchN_E4m3Fp32_Bfloat16_Tile128x128x128_EpiTile64x128_Mma64x128x32_Cluster1x1x1_transposeMmaOutput_DsFp8_sm100a_cubin_data,
        BatchedGemmKernel_BatchN_E4m3Fp32_Bfloat16_Tile128x128x128_EpiTile64x128_Mma64x128x32_Cluster1x1x1_transposeMmaOutput_DsFp8_sm100a_cubin_len,
        172032,
        "BatchedGemmKernel_BatchN_E4m3Fp32_Bfloat16_Tile128x128x128_EpiTile64x128_Mma64x128x32_Cluster1x1x1_transposeMmaOutput_DsFp8_sm100a",
        352,
        128,
        128,
        128,
        1,
        64,
        128,
        64,
        128,
        32,
        true,
        4,
        Data_type::DATA_TYPE_E4M3,
        Data_type::DATA_TYPE_BF16,
        Data_type::DATA_TYPE_FP32,
        true,
        true,
        true,
        0,
        0,
        BatchMode::BatchN,
        0,
        0,
        0,
        0,
        0,
        std::vector<double>{}
    },
    {
        BatchedGemmKernel_BatchM_E4m3Fp32_Bfloat16_Tile128x128x128_EpiTile128x128_Mma128x128x32_Cluster1x1x1_sm100a_cubin_data,
        BatchedGemmKernel_BatchM_E4m3Fp32_Bfloat16_Tile128x128x128_EpiTile128x128_Mma128x128x32_Cluster1x1x1_sm100a_cubin_len,
        167936,
        "BatchedGemmKernel_BatchM_E4m3Fp32_Bfloat16_Tile128x128x128_EpiTile128x128_Mma128x128x32_Cluster1x1x1_sm100a",
        192,
        128,
        128,
        128,
        1,
        128,
        128,
        128,
        128,
        32,
        true,
        4,
        Data_type::DATA_TYPE_E4M3,
        Data_type::DATA_TYPE_BF16,
        Data_type::DATA_TYPE_FP32,
        false,
        false,
        false,
        0,
        0,
        BatchMode::BatchM,
        0,
        0,
        0,
        0,
        0,
        std::vector<double>{}
    },
    {
        BatchedGemmKernel_BatchM_E4m3Fp32_Bfloat16_Tile128x128x128_EpiTile128x64_Mma128x64x32_Cluster1x1x1_DsFp8_sm100a_cubin_data,
        BatchedGemmKernel_BatchM_E4m3Fp32_Bfloat16_Tile128x128x128_EpiTile128x64_Mma128x64x32_Cluster1x1x1_DsFp8_sm100a_cubin_len,
        172032,
        "BatchedGemmKernel_BatchM_E4m3Fp32_Bfloat16_Tile128x128x128_EpiTile128x64_Mma128x64x32_Cluster1x1x1_DsFp8_sm100a",
        352,
        128,
        128,
        128,
        1,
        128,
        64,
        128,
        64,
        32,
        true,
        4,
        Data_type::DATA_TYPE_E4M3,
        Data_type::DATA_TYPE_BF16,
        Data_type::DATA_TYPE_FP32,
        false,
        true,
        false,
        0,
        0,
        BatchMode::BatchM,
        0,
        0,
        0,
        0,
        0,
        std::vector<double>{}
    },
};
// clang-format on
} // namespace tensorrt_llm::kernels
