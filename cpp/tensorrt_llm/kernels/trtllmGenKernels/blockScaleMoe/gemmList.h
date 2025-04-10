/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include "gemmCubins/MoE_ProjDown_BatchN_E2m1Fp32_Bfloat16_Tile128x8x512_EpiTile128x8_Mma128x8x64_Cluster1x1x1_transposeMmaOutput_sm100a_cubin.h"
#include "gemmCubins/MoE_ProjDown_BatchN_E4m3Fp32_Bfloat16_Tile128x8x128_EpiTile64x8_Mma64x8x32_Cluster1x1x1_transposeMmaOutput_DsFp8_sm100a_cubin.h"
#include "gemmCubins/MoE_ProjUp_BatchN_E2m1Fp32_E2m1_Tile128x8x512_EpiTile128x8_Mma128x8x64_Cluster1x1x1_transposeMmaOutput_InplaceRoute_GatedAct_sm100a_cubin.h"
#include "gemmCubins/MoE_ProjUp_BatchN_E4m3Fp32_E4m3_Tile128x8x128_EpiTile64x8_Mma64x8x32_Cluster1x1x1_transposeMmaOutput_DsFp8_InplaceRoute_sm100a_cubin.h"
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
    bool useTwoTmaLoadWarps{false};
    unsigned numStages{0};
    unsigned numStagesMma{0};
    unsigned paramsStructSize{0};
    bool useFusedAct{false};
    unsigned threadsPerCTA{0};
    bool gateUseClusterSplitK{false};
    int projGateNumSplitKSlices{0};
    bool sliceK{false};
    int mNumSlicesForSliceK{0};
    tg::SfLayout mSfLayoutB{tg::SfLayout::Linear};
    tg::SfLayout mSfLayoutC{tg::SfLayout::Linear};
};

namespace PermuteGemm1
{
// clang-format off
const std::vector<GemmInfo> gemmList {
    {
        /* data */ MoE_ProjUp_BatchN_E4m3Fp32_E4m3_Tile128x8x128_EpiTile64x8_Mma64x8x32_Cluster1x1x1_transposeMmaOutput_DsFp8_InplaceRoute_sm100a_cubin_data,
        /* size */ MoE_ProjUp_BatchN_E4m3Fp32_E4m3_Tile128x8x128_EpiTile64x8_Mma64x8x32_Cluster1x1x1_transposeMmaOutput_DsFp8_InplaceRoute_sm100a_cubin_len,
        /* sharedMemSize */ 44032,
        /* functionName */ "MoE_ProjUp_BatchN_E4m3Fp32_E4m3_Tile128x8x128_EpiTile64x8_Mma64x8x32_Cluster1x1x1_transposeMmaOutput_DsFp8_InplaceRoute_sm100a",
        /* blockScale */ true,
        /* permuteFusion */ true,
        /* shuffledMatrixA */ false,
        /* tileM */ 128,
        /* tileN */ 8,
        /* tileK */ 128,
        /* epilogueTileM */ 64,
        /* epilogueTileN */ 8,
        /* mmaM */ 64,
        /* mmaN */ 8,
        /* mmaK */ 32,
        /* numSlicesForSplitK */ 1,
        /* Dtype */ tg::Dtype::E4m3,
        /* Dtype */ tg::Dtype::E4m3,
        /* Dtype */ tg::Dtype::Fp32,
        /* useTmaStore */ true,
        /* useTwoTmaLoadWarps */ false,
        /* numStages */ 2,
        /* numStagesMma */ 1,
        /* paramsStructSize */ 17280,
        /* useFusedAct */ false,
        /* threadsPerCTA */ 416,
        /* gateUseClusterSplitK */ false,
        /* projGateNumSplitKSlices */ 1,
        /* sliceK */ false,
        /* mNumSlicesForSliceK */ 1,
        /* mSfLayoutB */ tg::SfLayout::Linear,
        /* mSfLayoutC */ tg::SfLayout::Linear,
    },
    {
        /* data */ MoE_ProjUp_BatchN_E2m1Fp32_E2m1_Tile128x8x512_EpiTile128x8_Mma128x8x64_Cluster1x1x1_transposeMmaOutput_InplaceRoute_GatedAct_sm100a_cubin_data,
        /* size */ MoE_ProjUp_BatchN_E2m1Fp32_E2m1_Tile128x8x512_EpiTile128x8_Mma128x8x64_Cluster1x1x1_transposeMmaOutput_InplaceRoute_GatedAct_sm100a_cubin_len,
        /* sharedMemSize */ 164864,
        /* functionName */ "MoE_ProjUp_BatchN_E2m1Fp32_E2m1_Tile128x8x512_EpiTile128x8_Mma128x8x64_Cluster1x1x1_transposeMmaOutput_InplaceRoute_GatedAct_sm100a",
        /* blockScale */ true,
        /* permuteFusion */ true,
        /* shuffledMatrixA */ true,
        /* tileM */ 128,
        /* tileN */ 8,
        /* tileK */ 512,
        /* epilogueTileM */ 128,
        /* epilogueTileN */ 8,
        /* mmaM */ 128,
        /* mmaN */ 8,
        /* mmaK */ 64,
        /* numSlicesForSplitK */ 1,
        /* Dtype */ tg::Dtype::E2m1,
        /* Dtype */ tg::Dtype::E2m1,
        /* Dtype */ tg::Dtype::Fp32,
        /* useTmaStore */ true,
        /* useTwoTmaLoadWarps */ false,
        /* numStages */ 3,
        /* numStagesMma */ 1,
        /* paramsStructSize */ 17280,
        /* useFusedAct */ true,
        /* threadsPerCTA */ 480,
        /* gateUseClusterSplitK */ false,
        /* projGateNumSplitKSlices */ 1,
        /* sliceK */ false,
        /* mNumSlicesForSliceK */ 1,
        // FIXME: The actual data layout needs to be Linear, but for the other checks and TMA descriptor creation,
        // we use the 8x4 SF layout. TMA descriptor won't be used for permute fusion version of the FC1.
        /* mSfLayoutB */ tg::SfLayout::R8c4,
        /* mSfLayoutC */ tg::SfLayout::R8c4,
    }
};
// clang-format on
} // namespace PermuteGemm1

namespace Gemm2
{
// clang-format off
const std::vector<GemmInfo> gemmList {
    {
        /* data */ MoE_ProjDown_BatchN_E4m3Fp32_Bfloat16_Tile128x8x128_EpiTile64x8_Mma64x8x32_Cluster1x1x1_transposeMmaOutput_DsFp8_sm100a_cubin_data,
        /* size */ MoE_ProjDown_BatchN_E4m3Fp32_Bfloat16_Tile128x8x128_EpiTile64x8_Mma64x8x32_Cluster1x1x1_transposeMmaOutput_DsFp8_sm100a_cubin_len,
        /* sharedMemSize */ 44032,
        /* functionName */ "MoE_ProjDown_BatchN_E4m3Fp32_Bfloat16_Tile128x8x128_EpiTile64x8_Mma64x8x32_Cluster1x1x1_transposeMmaOutput_DsFp8_sm100a",
        /* blockScale */ true,
        /* permuteFusion */ false,
        /* shuffledMatrixA */ false,
        /* tileM */ 128,
        /* tileN */ 8,
        /* tileK */ 128,
        /* epilogueTileM */ 64,
        /* epilogueTileN */ 8,
        /* mmaM */ 64,
        /* mmaN */ 8,
        /* mmaK */ 32,
        /* numSlicesForSplitK */ 1,
        /* Dtype */ tg::Dtype::E4m3,
        /* Dtype */ tg::Dtype::Bfloat16,
        /* Dtype */ tg::Dtype::Fp32,
        /* useTmaStore */ true,
        /* useTwoTmaLoadWarps */ false,
        /* numStages */ 2,
        /* numStagesMma */ 1,
        /* paramsStructSize */ 17280,
        /* useFusedAct */ false,
        /* threadsPerCTA */ 384,
        /* gateUseClusterSplitK */ false,
        /* projGateNumSplitKSlices */ 1,
        /* sliceK */ false,
        /* mNumSlicesForSliceK */ 1,
        /* mSfLayoutB */ tg::SfLayout::Linear,
        /* mSfLayoutC */ tg::SfLayout::Linear,
    },
    {
        /* data */ MoE_ProjDown_BatchN_E2m1Fp32_Bfloat16_Tile128x8x512_EpiTile128x8_Mma128x8x64_Cluster1x1x1_transposeMmaOutput_sm100a_cubin_data,
        /* size */ MoE_ProjDown_BatchN_E2m1Fp32_Bfloat16_Tile128x8x512_EpiTile128x8_Mma128x8x64_Cluster1x1x1_transposeMmaOutput_sm100a_cubin_len,
        /* sharedMemSize */ 165888,
        /* functionName */ "MoE_ProjDown_BatchN_E2m1Fp32_Bfloat16_Tile128x8x512_EpiTile128x8_Mma128x8x64_Cluster1x1x1_transposeMmaOutput_sm100a",
        /* blockScale */ true,
        /* permuteFusion */ true,
        /* shuffledMatrixA */ true,
        /* tileM */ 128,
        /* tileN */ 8,
        /* tileK */ 512,
        /* epilogueTileM */ 128,
        /* epilogueTileN */ 8,
        /* mmaM */ 128,
        /* mmaN */ 8,
        /* mmaK */ 64,
        /* numSlicesForSplitK */ 1,
        /* Dtype */ tg::Dtype::E2m1,
        /* Dtype */ tg::Dtype::Bfloat16,
        /* Dtype */ tg::Dtype::Fp32,
        /* useTmaStore */ true,
        /* useTwoTmaLoadWarps */ false,
        /* numStages */ 3,
        /* numStagesMma */ 1,
        /* paramsStructSize */ 17280,
        /* useFusedAct */ false,
        /* threadsPerCTA */ 448,
        /* gateUseClusterSplitK */ false,
        /* projGateNumSplitKSlices */ 1,
        /* sliceK */ false,
        /* mNumSlicesForSliceK */ 1,
        /* mSfLayoutB */ tg::SfLayout::R8c4,
        /* mSfLayoutC */ tg::SfLayout::R8c4,
    }
};
// clang-format on
} // namespace Gemm2

} // namespace trtllmGenFp8BlockScaleMoe
} // namespace kernels
} // namespace tensorrt_llm
