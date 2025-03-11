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

#include "gemmCubins/MoE_ProjDown__BatchN_E4m3Fp32_Bfloat16_Tile128x128x128_EpiTile64x128_Mma64x128x32_Cluster1x1x1_transposeMmaOutput_DsFp8_sm100a_cubin.h"
#include "gemmCubins/MoE_ProjDown__BatchN_E4m3Fp32_E4m3_Tile128x128x128_EpiTile64x128_Mma64x128x32_Cluster1x1x1_transposeMmaOutput_DsFp8_sm100a_cubin.h"
#include "gemmCubins/MoE_ProjUp__BatchN_E4m3Fp32_E4m3_Tile128x128x128_EpiTile64x128_Mma64x128x32_Cluster1x1x1_transposeMmaOutput_DsFp8_InplaceRoute_sm100a_cubin.h"
#include <vector>

namespace tensorrt_llm
{
namespace kernels
{
namespace trtllmGenFp8BlockScaleMoe
{
struct GemmInfo
{
    unsigned char const* data{nullptr};
    unsigned const size{0};
    unsigned const sharedMemSize{0};
    char const* functionName{nullptr};
    bool blockScale{false};
    bool permuteFusion{false};
    bool shuffledMatrixA{false};
    unsigned tileM{0};
    unsigned tileN{0};
    unsigned tileK{0};
    unsigned epilogueTileM{0};
    unsigned epilogueTileN{0};
    unsigned mmaM{0};
    unsigned mmaN{0};
    unsigned mmaK{0};
    unsigned numSlicesForSplitK{0};
    tg::Dtype dtypeElt{tg::Dtype::Void};
    tg::Dtype dtypeC{tg::Dtype::Void};
    tg::Dtype dtypeAcc{tg::Dtype::Void};
    bool useTmaStore{false};
    unsigned numStages{0};
    unsigned paramsStructSize{0};
    bool useFusedAct{false};
    unsigned threadsPerCTA{0};
    bool gateUseClusterSplitK;
    int projGateNumSplitKSlices;
};

namespace PermuteGemm1
{
// clang-format off
const std::vector<GemmInfo> gemmList {
    {
        MoE_ProjUp__BatchN_E4m3Fp32_E4m3_Tile128x128x128_EpiTile64x128_Mma64x128x32_Cluster1x1x1_transposeMmaOutput_DsFp8_InplaceRoute_sm100a_cubin_data,
        MoE_ProjUp__BatchN_E4m3Fp32_E4m3_Tile128x128x128_EpiTile64x128_Mma64x128x32_Cluster1x1x1_transposeMmaOutput_DsFp8_InplaceRoute_sm100a_cubin_len,
        122880,
        "MoE_ProjUp__BatchN_E4m3Fp32_E4m3_Tile128x128x128_EpiTile64x128_Mma64x128x32_Cluster1x1x1_transposeMmaOutput_DsFp8_InplaceRoute_sm100a",
        true,
        true,
        false,
        128,
        128,
        128,
        64,
        128,
        64,
        128,
        32,
        1,
        tg::Dtype::E4m3,
        tg::Dtype::E4m3,
        tg::Dtype::Fp32,
        true,
        3,
        17280,
        false,
        480,
        false,
        1
    }
};
// clang-format on
} // namespace PermuteGemm1

namespace Gemm2
{
// clang-format off
const std::vector<GemmInfo> gemmList {
    {
        MoE_ProjDown__BatchN_E4m3Fp32_E4m3_Tile128x128x128_EpiTile64x128_Mma64x128x32_Cluster1x1x1_transposeMmaOutput_DsFp8_sm100a_cubin_data,
        MoE_ProjDown__BatchN_E4m3Fp32_E4m3_Tile128x128x128_EpiTile64x128_Mma64x128x32_Cluster1x1x1_transposeMmaOutput_DsFp8_sm100a_cubin_len,
        121856,
        "MoE_ProjDown__BatchN_E4m3Fp32_E4m3_Tile128x128x128_EpiTile64x128_Mma64x128x32_Cluster1x1x1_transposeMmaOutput_DsFp8_sm100a",
        true,
        false,
        false,
        128,
        128,
        128,
        64,
        128,
        64,
        128,
        32,
        1,
        tg::Dtype::E4m3,
        tg::Dtype::E4m3,
        tg::Dtype::Fp32,
        true,
        3,
        17280,
        false,
        384,
        false,
        1
    },
    {
        MoE_ProjDown__BatchN_E4m3Fp32_Bfloat16_Tile128x128x128_EpiTile64x128_Mma64x128x32_Cluster1x1x1_transposeMmaOutput_DsFp8_sm100a_cubin_data,
        MoE_ProjDown__BatchN_E4m3Fp32_Bfloat16_Tile128x128x128_EpiTile64x128_Mma64x128x32_Cluster1x1x1_transposeMmaOutput_DsFp8_sm100a_cubin_len,
        139264,
        "MoE_ProjDown__BatchN_E4m3Fp32_Bfloat16_Tile128x128x128_EpiTile64x128_Mma64x128x32_Cluster1x1x1_transposeMmaOutput_DsFp8_sm100a",
        true,
        false,
        false,
        128,
        128,
        128,
        64,
        128,
        64,
        128,
        32,
        1,
        tg::Dtype::E4m3,
        tg::Dtype::Bfloat16,
        tg::Dtype::Fp32,
        true,
        3,
        17280,
        false,
        384,
        false,
        1
    }
};
// clang-format on
} // namespace Gemm2

} // namespace trtllmGenFp8BlockScaleMoe
} // namespace kernels
} // namespace tensorrt_llm
