/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION &
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

#include "GemmGatedActOptions.h"

namespace gemmGatedAct
{

namespace tensorrt_llm
{
namespace kernels
{
// clang-format off

#define TLLM_GEN_COMMIT "051000ea"
#define TLLM_GEN_EXPORT_VERSION "7.0.3.0"

static constexpr size_t tllmGenGemmGatedActListLen = 13;

#ifndef EXCLUDE_SM_100
extern unsigned char GemmGatedActKernel_Bfloat16_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100a_cubin[];
extern unsigned char GemmGatedActKernel_Bfloat16_E4m3E4m3_Fp32_t128x128x256u2_s2_et128x128_m128x128x32_cga1x1x1_16dp256b_TN_schedS_swiGlu_sm100a_cubin[];
extern unsigned char GemmGatedActKernel_Bfloat16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x1_16dp256b_TN_transOut_schedS_swiGlu_sm100a_cubin[];
extern unsigned char GemmGatedActKernel_Bfloat16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100a_cubin[];
extern unsigned char GemmGatedActKernel_E2m1_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100a_cubin[];
extern unsigned char GemmGatedActKernel_E4m3_E4m3E4m3_Fp32_t128x128x256u2_s2_et128x128_m128x128x32_cga1x1x1_16dp256b_TN_schedS_swiGlu_sm100a_cubin[];
extern unsigned char GemmGatedActKernel_E4m3_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x1_16dp256b_TN_transOut_schedS_swiGlu_sm100a_cubin[];
extern unsigned char GemmGatedActKernel_E4m3_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100a_cubin[];
extern unsigned char GemmGatedActKernel_Fp16_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100a_cubin[];
extern unsigned char GemmGatedActKernel_Fp16_E4m3E4m3_Fp32_t128x128x256u2_s2_et128x128_m128x128x32_cga1x1x1_16dp256b_TN_schedS_swiGlu_sm100a_cubin[];
extern unsigned char GemmGatedActKernel_Fp16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x1_16dp256b_TN_transOut_schedS_swiGlu_sm100a_cubin[];
extern unsigned char GemmGatedActKernel_Fp16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100a_cubin[];
extern unsigned char GemmGatedActKernel_Fp32_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100a_cubin[];
#endif // EXCLUDE_SM_100

#ifndef EXCLUDE_SM_100
extern unsigned int GemmGatedActKernel_Bfloat16_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100a_cubin_len;
extern unsigned int GemmGatedActKernel_Bfloat16_E4m3E4m3_Fp32_t128x128x256u2_s2_et128x128_m128x128x32_cga1x1x1_16dp256b_TN_schedS_swiGlu_sm100a_cubin_len;
extern unsigned int GemmGatedActKernel_Bfloat16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x1_16dp256b_TN_transOut_schedS_swiGlu_sm100a_cubin_len;
extern unsigned int GemmGatedActKernel_Bfloat16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100a_cubin_len;
extern unsigned int GemmGatedActKernel_E2m1_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100a_cubin_len;
extern unsigned int GemmGatedActKernel_E4m3_E4m3E4m3_Fp32_t128x128x256u2_s2_et128x128_m128x128x32_cga1x1x1_16dp256b_TN_schedS_swiGlu_sm100a_cubin_len;
extern unsigned int GemmGatedActKernel_E4m3_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x1_16dp256b_TN_transOut_schedS_swiGlu_sm100a_cubin_len;
extern unsigned int GemmGatedActKernel_E4m3_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100a_cubin_len;
extern unsigned int GemmGatedActKernel_Fp16_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100a_cubin_len;
extern unsigned int GemmGatedActKernel_Fp16_E4m3E4m3_Fp32_t128x128x256u2_s2_et128x128_m128x128x32_cga1x1x1_16dp256b_TN_schedS_swiGlu_sm100a_cubin_len;
extern unsigned int GemmGatedActKernel_Fp16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x1_16dp256b_TN_transOut_schedS_swiGlu_sm100a_cubin_len;
extern unsigned int GemmGatedActKernel_Fp16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100a_cubin_len;
extern unsigned int GemmGatedActKernel_Fp32_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100a_cubin_len;
#endif // EXCLUDE_SM_100


static const gemmGatedAct::GemmGatedActConfig tllmGenGemmGatedActList[] = {
#ifndef EXCLUDE_SM_100
{GemmGatedActKernel_Bfloat16_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100a_cubin, GemmGatedActKernel_Bfloat16_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100a_cubin_len, 86016, "gemmGatedActKernel_Bfloat16_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100a", 448, "96e542e09e16b9927422afec4c16854cc29a142e7c1fee7dc9bbffd33f7eb87f", { { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
, /* mBiasType */ gemm::BiasType(0)
, /* mBlockK */ -1
, /* mClusterDimX */ 1
, /* mClusterDimY */ 1
, /* mClusterDimZ */ 4
, /* mDtypeAcc */ trtllm::gen::Dtype(1056776)
, /* mDtypeA */ trtllm::gen::Dtype(17826818)
, /* mDtypeB */ trtllm::gen::Dtype(17826818)
, /* mDtypeC */ trtllm::gen::Dtype(1052672)
, /* mDtypeMmaA */ trtllm::gen::Dtype(17826818)
, /* mDtypeMmaB */ trtllm::gen::Dtype(17826818)
, /* mEnablesEarlyExit */ 0
, /* mEnablesDelayedEarlyExit */ 0
, /* mEnablesGlobalPtxKnobs */ 1
, /* mEpilogueLdtmDps */ 16
, /* mEpilogueLdtmBits */ 256
, /* mEpilogueTileM */ 128
, /* mEpilogueTileN */ 8
, /* mGridTriggerSecondaryA */ 0
, /* mGridTriggerSecondaryB */ 0
, /* mGridWaitForPrimaryEarlyExit */ 1
, /* mGridWaitForPrimaryA */ 1
, /* mGridWaitForPrimaryB */ 1
, /* mHoistLoadTaskInit */ 1
, /* mHoistMmaTaskTryWaits */ 0
, /* mK */ 2048
, /* mKernelTraits */ {}
, /* mLayoutA */ gemm::MatrixLayout(0)
, /* mLayoutB */ gemm::MatrixLayout(0)
, /* mM */ 256
, /* mMmaK */ 64
, /* mMmaKind */ trtllm::gen::MmaKind(4)
, /* mMmaM */ 128
, /* mMmaN */ 8
, /* mMockAllReduce */ 0
, /* mN */ 256
, /* mNumSlicesForSplitK */ 4
, /* mNumSlicesForSliceK */ 1
, /* mNumStages */ 4
, /* mNumStagesMma */ 1
, /* mNumStagesMmaWithinWorkTile */ 1
, /* mNumStagesMmaAcrossWorkTile */ 1
, /* mNumStagesWorkId */ 3
, /* mOutputDebugTensors */ 0
, /* mPatchF2fp */ 0
, /* mUseShuffledMatrixA */ 1
, /* mSliceK */ 0
, /* mSplitK */ gemm::SplitK(2)
, /* mTransposeMmaOutput */ 1
, /* mTileM */ 128
, /* mTileN */ 8
, /* mTileK */ 256
, /* mUseUnrollLoop2xForMma */ 1
, /* mUseCustomMmaSchedule */ 1
, /* mUseHoistTryWaitForCustomMmaSchedule */ 0
, /* mUseDeepSeekFp8 */ 0
, /* mUsePerTokenSfA */ 0
, /* mUsePerTokenSfB */ 0
, /* mUseTmaStore */ 1
, /* mUseTwoTmaLoadWarps */ 1
, /* mUseTwoMmaWarps */ 0
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(1)
, /* mSfLayoutC */ trtllm::gen::SfLayout(1)
, /* mSfReshapeFactor */ 1
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, /* mActType */ gemmGatedAct::ActType(0)
, /* mClampBeforeAct */ 0
 }, gemm::SmVersion::Sm100a},
{GemmGatedActKernel_Bfloat16_E4m3E4m3_Fp32_t128x128x256u2_s2_et128x128_m128x128x32_cga1x1x1_16dp256b_TN_schedS_swiGlu_sm100a_cubin, GemmGatedActKernel_Bfloat16_E4m3E4m3_Fp32_t128x128x256u2_s2_et128x128_m128x128x32_cga1x1x1_16dp256b_TN_schedS_swiGlu_sm100a_cubin_len, 168960, "gemmGatedActKernel_Bfloat16_E4m3E4m3_Fp32_t128x128x256u2_s2_et128x128_m128x128x32_cga1x1x1_16dp256b_TN_schedS_swiGlu_sm100a", 224, "168780ebf86e7d79779a5b52b605e72d29de0fb8abc4bc1f38f040817b986607", { { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
, /* mBiasType */ gemm::BiasType(0)
, /* mBlockK */ -1
, /* mClusterDimX */ 1
, /* mClusterDimY */ 1
, /* mClusterDimZ */ 1
, /* mDtypeAcc */ trtllm::gen::Dtype(1056776)
, /* mDtypeA */ trtllm::gen::Dtype(1050629)
, /* mDtypeB */ trtllm::gen::Dtype(1050629)
, /* mDtypeC */ trtllm::gen::Dtype(1052672)
, /* mDtypeMmaA */ trtllm::gen::Dtype(1050629)
, /* mDtypeMmaB */ trtllm::gen::Dtype(1050629)
, /* mEnablesEarlyExit */ 0
, /* mEnablesDelayedEarlyExit */ 0
, /* mEnablesGlobalPtxKnobs */ 1
, /* mEpilogueLdtmDps */ 16
, /* mEpilogueLdtmBits */ 256
, /* mEpilogueTileM */ 128
, /* mEpilogueTileN */ 128
, /* mGridTriggerSecondaryA */ 0
, /* mGridTriggerSecondaryB */ 0
, /* mGridWaitForPrimaryEarlyExit */ 1
, /* mGridWaitForPrimaryA */ 1
, /* mGridWaitForPrimaryB */ 1
, /* mHoistLoadTaskInit */ 1
, /* mHoistMmaTaskTryWaits */ 0
, /* mK */ 2048
, /* mKernelTraits */ {}
, /* mLayoutA */ gemm::MatrixLayout(0)
, /* mLayoutB */ gemm::MatrixLayout(0)
, /* mM */ 256
, /* mMmaK */ 32
, /* mMmaKind */ trtllm::gen::MmaKind(2)
, /* mMmaM */ 128
, /* mMmaN */ 128
, /* mMockAllReduce */ 0
, /* mN */ 256
, /* mNumSlicesForSplitK */ 1
, /* mNumSlicesForSliceK */ 1
, /* mNumStages */ 2
, /* mNumStagesMma */ 1
, /* mNumStagesMmaWithinWorkTile */ 1
, /* mNumStagesMmaAcrossWorkTile */ 1
, /* mNumStagesWorkId */ 3
, /* mOutputDebugTensors */ 0
, /* mPatchF2fp */ 0
, /* mUseShuffledMatrixA */ 0
, /* mSliceK */ 0
, /* mSplitK */ gemm::SplitK(0)
, /* mTransposeMmaOutput */ 0
, /* mTileM */ 128
, /* mTileN */ 128
, /* mTileK */ 256
, /* mUseUnrollLoop2xForMma */ 1
, /* mUseCustomMmaSchedule */ 1
, /* mUseHoistTryWaitForCustomMmaSchedule */ 0
, /* mUseDeepSeekFp8 */ 0
, /* mUsePerTokenSfA */ 0
, /* mUsePerTokenSfB */ 0
, /* mUseTmaStore */ 1
, /* mUseTwoTmaLoadWarps */ 1
, /* mUseTwoMmaWarps */ 0
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(3)
, /* mSfLayoutC */ trtllm::gen::SfLayout(3)
, /* mSfReshapeFactor */ 1
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, /* mActType */ gemmGatedAct::ActType(0)
, /* mClampBeforeAct */ 0
 }, gemm::SmVersion::Sm100a},
{GemmGatedActKernel_Bfloat16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x1_16dp256b_TN_transOut_schedS_swiGlu_sm100a_cubin, GemmGatedActKernel_Bfloat16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x1_16dp256b_TN_transOut_schedS_swiGlu_sm100a_cubin_len, 112640, "gemmGatedActKernel_Bfloat16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x1_16dp256b_TN_transOut_schedS_swiGlu_sm100a", 224, "1a459a19a7d6e925c8f27bd9d4946ccbbf9758f4dedd799b658680d076bbfd21", { { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
, /* mBiasType */ gemm::BiasType(0)
, /* mBlockK */ -1
, /* mClusterDimX */ 1
, /* mClusterDimY */ 1
, /* mClusterDimZ */ 1
, /* mDtypeAcc */ trtllm::gen::Dtype(1056776)
, /* mDtypeA */ trtllm::gen::Dtype(1050629)
, /* mDtypeB */ trtllm::gen::Dtype(1050629)
, /* mDtypeC */ trtllm::gen::Dtype(1052672)
, /* mDtypeMmaA */ trtllm::gen::Dtype(1050629)
, /* mDtypeMmaB */ trtllm::gen::Dtype(1050629)
, /* mEnablesEarlyExit */ 0
, /* mEnablesDelayedEarlyExit */ 0
, /* mEnablesGlobalPtxKnobs */ 1
, /* mEpilogueLdtmDps */ 16
, /* mEpilogueLdtmBits */ 256
, /* mEpilogueTileM */ 128
, /* mEpilogueTileN */ 8
, /* mGridTriggerSecondaryA */ 0
, /* mGridTriggerSecondaryB */ 0
, /* mGridWaitForPrimaryEarlyExit */ 1
, /* mGridWaitForPrimaryA */ 1
, /* mGridWaitForPrimaryB */ 1
, /* mHoistLoadTaskInit */ 1
, /* mHoistMmaTaskTryWaits */ 0
, /* mK */ 2048
, /* mKernelTraits */ {}
, /* mLayoutA */ gemm::MatrixLayout(0)
, /* mLayoutB */ gemm::MatrixLayout(0)
, /* mM */ 256
, /* mMmaK */ 32
, /* mMmaKind */ trtllm::gen::MmaKind(2)
, /* mMmaM */ 128
, /* mMmaN */ 8
, /* mMockAllReduce */ 0
, /* mN */ 256
, /* mNumSlicesForSplitK */ 1
, /* mNumSlicesForSliceK */ 1
, /* mNumStages */ 3
, /* mNumStagesMma */ 1
, /* mNumStagesMmaWithinWorkTile */ 1
, /* mNumStagesMmaAcrossWorkTile */ 1
, /* mNumStagesWorkId */ 3
, /* mOutputDebugTensors */ 0
, /* mPatchF2fp */ 0
, /* mUseShuffledMatrixA */ 1
, /* mSliceK */ 0
, /* mSplitK */ gemm::SplitK(0)
, /* mTransposeMmaOutput */ 1
, /* mTileM */ 128
, /* mTileN */ 8
, /* mTileK */ 256
, /* mUseUnrollLoop2xForMma */ 1
, /* mUseCustomMmaSchedule */ 1
, /* mUseHoistTryWaitForCustomMmaSchedule */ 0
, /* mUseDeepSeekFp8 */ 0
, /* mUsePerTokenSfA */ 0
, /* mUsePerTokenSfB */ 0
, /* mUseTmaStore */ 1
, /* mUseTwoTmaLoadWarps */ 1
, /* mUseTwoMmaWarps */ 0
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(3)
, /* mSfLayoutC */ trtllm::gen::SfLayout(3)
, /* mSfReshapeFactor */ 1
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, /* mActType */ gemmGatedAct::ActType(0)
, /* mClampBeforeAct */ 0
 }, gemm::SmVersion::Sm100a},
{GemmGatedActKernel_Bfloat16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100a_cubin, GemmGatedActKernel_Bfloat16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100a_cubin_len, 110592, "gemmGatedActKernel_Bfloat16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100a", 224, "e7914f7ecee3c15cdc90053c78218e6549e0107cbfd15c38426ed2ca72398e5e", { { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
, /* mBiasType */ gemm::BiasType(0)
, /* mBlockK */ -1
, /* mClusterDimX */ 1
, /* mClusterDimY */ 1
, /* mClusterDimZ */ 4
, /* mDtypeAcc */ trtllm::gen::Dtype(1056776)
, /* mDtypeA */ trtllm::gen::Dtype(1050629)
, /* mDtypeB */ trtllm::gen::Dtype(1050629)
, /* mDtypeC */ trtllm::gen::Dtype(1052672)
, /* mDtypeMmaA */ trtllm::gen::Dtype(1050629)
, /* mDtypeMmaB */ trtllm::gen::Dtype(1050629)
, /* mEnablesEarlyExit */ 0
, /* mEnablesDelayedEarlyExit */ 0
, /* mEnablesGlobalPtxKnobs */ 1
, /* mEpilogueLdtmDps */ 16
, /* mEpilogueLdtmBits */ 256
, /* mEpilogueTileM */ 128
, /* mEpilogueTileN */ 8
, /* mGridTriggerSecondaryA */ 0
, /* mGridTriggerSecondaryB */ 0
, /* mGridWaitForPrimaryEarlyExit */ 1
, /* mGridWaitForPrimaryA */ 1
, /* mGridWaitForPrimaryB */ 1
, /* mHoistLoadTaskInit */ 1
, /* mHoistMmaTaskTryWaits */ 0
, /* mK */ 2048
, /* mKernelTraits */ {}
, /* mLayoutA */ gemm::MatrixLayout(0)
, /* mLayoutB */ gemm::MatrixLayout(0)
, /* mM */ 256
, /* mMmaK */ 32
, /* mMmaKind */ trtllm::gen::MmaKind(2)
, /* mMmaM */ 128
, /* mMmaN */ 8
, /* mMockAllReduce */ 0
, /* mN */ 256
, /* mNumSlicesForSplitK */ 4
, /* mNumSlicesForSliceK */ 1
, /* mNumStages */ 3
, /* mNumStagesMma */ 1
, /* mNumStagesMmaWithinWorkTile */ 1
, /* mNumStagesMmaAcrossWorkTile */ 1
, /* mNumStagesWorkId */ 3
, /* mOutputDebugTensors */ 0
, /* mPatchF2fp */ 0
, /* mUseShuffledMatrixA */ 1
, /* mSliceK */ 0
, /* mSplitK */ gemm::SplitK(2)
, /* mTransposeMmaOutput */ 1
, /* mTileM */ 128
, /* mTileN */ 8
, /* mTileK */ 256
, /* mUseUnrollLoop2xForMma */ 1
, /* mUseCustomMmaSchedule */ 1
, /* mUseHoistTryWaitForCustomMmaSchedule */ 0
, /* mUseDeepSeekFp8 */ 0
, /* mUsePerTokenSfA */ 0
, /* mUsePerTokenSfB */ 0
, /* mUseTmaStore */ 1
, /* mUseTwoTmaLoadWarps */ 1
, /* mUseTwoMmaWarps */ 0
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(3)
, /* mSfLayoutC */ trtllm::gen::SfLayout(3)
, /* mSfReshapeFactor */ 1
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, /* mActType */ gemmGatedAct::ActType(0)
, /* mClampBeforeAct */ 0
 }, gemm::SmVersion::Sm100a},
{GemmGatedActKernel_E2m1_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100a_cubin, GemmGatedActKernel_E2m1_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100a_cubin_len, 86016, "gemmGatedActKernel_E2m1_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100a", 448, "bc417f58a40b4ba68730e0dc1ff3a5343f883ffbf18c1056c9c1a7ebd4951595", { { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
, /* mBiasType */ gemm::BiasType(0)
, /* mBlockK */ -1
, /* mClusterDimX */ 1
, /* mClusterDimY */ 1
, /* mClusterDimZ */ 4
, /* mDtypeAcc */ trtllm::gen::Dtype(1056776)
, /* mDtypeA */ trtllm::gen::Dtype(17826818)
, /* mDtypeB */ trtllm::gen::Dtype(17826818)
, /* mDtypeC */ trtllm::gen::Dtype(17826818)
, /* mDtypeMmaA */ trtllm::gen::Dtype(17826818)
, /* mDtypeMmaB */ trtllm::gen::Dtype(17826818)
, /* mEnablesEarlyExit */ 0
, /* mEnablesDelayedEarlyExit */ 0
, /* mEnablesGlobalPtxKnobs */ 1
, /* mEpilogueLdtmDps */ 16
, /* mEpilogueLdtmBits */ 256
, /* mEpilogueTileM */ 128
, /* mEpilogueTileN */ 8
, /* mGridTriggerSecondaryA */ 0
, /* mGridTriggerSecondaryB */ 0
, /* mGridWaitForPrimaryEarlyExit */ 1
, /* mGridWaitForPrimaryA */ 1
, /* mGridWaitForPrimaryB */ 1
, /* mHoistLoadTaskInit */ 1
, /* mHoistMmaTaskTryWaits */ 0
, /* mK */ 2048
, /* mKernelTraits */ {}
, /* mLayoutA */ gemm::MatrixLayout(0)
, /* mLayoutB */ gemm::MatrixLayout(0)
, /* mM */ 256
, /* mMmaK */ 64
, /* mMmaKind */ trtllm::gen::MmaKind(4)
, /* mMmaM */ 128
, /* mMmaN */ 8
, /* mMockAllReduce */ 0
, /* mN */ 256
, /* mNumSlicesForSplitK */ 4
, /* mNumSlicesForSliceK */ 1
, /* mNumStages */ 4
, /* mNumStagesMma */ 1
, /* mNumStagesMmaWithinWorkTile */ 1
, /* mNumStagesMmaAcrossWorkTile */ 1
, /* mNumStagesWorkId */ 3
, /* mOutputDebugTensors */ 0
, /* mPatchF2fp */ 0
, /* mUseShuffledMatrixA */ 1
, /* mSliceK */ 0
, /* mSplitK */ gemm::SplitK(2)
, /* mTransposeMmaOutput */ 1
, /* mTileM */ 128
, /* mTileN */ 8
, /* mTileK */ 256
, /* mUseUnrollLoop2xForMma */ 1
, /* mUseCustomMmaSchedule */ 1
, /* mUseHoistTryWaitForCustomMmaSchedule */ 0
, /* mUseDeepSeekFp8 */ 0
, /* mUsePerTokenSfA */ 0
, /* mUsePerTokenSfB */ 0
, /* mUseTmaStore */ 1
, /* mUseTwoTmaLoadWarps */ 1
, /* mUseTwoMmaWarps */ 0
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(1)
, /* mSfLayoutC */ trtllm::gen::SfLayout(1)
, /* mSfReshapeFactor */ 1
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, /* mActType */ gemmGatedAct::ActType(0)
, /* mClampBeforeAct */ 0
 }, gemm::SmVersion::Sm100a},
{GemmGatedActKernel_E4m3_E4m3E4m3_Fp32_t128x128x256u2_s2_et128x128_m128x128x32_cga1x1x1_16dp256b_TN_schedS_swiGlu_sm100a_cubin, GemmGatedActKernel_E4m3_E4m3E4m3_Fp32_t128x128x256u2_s2_et128x128_m128x128x32_cga1x1x1_16dp256b_TN_schedS_swiGlu_sm100a_cubin_len, 152576, "gemmGatedActKernel_E4m3_E4m3E4m3_Fp32_t128x128x256u2_s2_et128x128_m128x128x32_cga1x1x1_16dp256b_TN_schedS_swiGlu_sm100a", 224, "ca3f4ea86ec01d7d6dbbebb3f5eb1248685403ea253e44931d0d01c8d24c81bf", { { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
, /* mBiasType */ gemm::BiasType(0)
, /* mBlockK */ -1
, /* mClusterDimX */ 1
, /* mClusterDimY */ 1
, /* mClusterDimZ */ 1
, /* mDtypeAcc */ trtllm::gen::Dtype(1056776)
, /* mDtypeA */ trtllm::gen::Dtype(1050629)
, /* mDtypeB */ trtllm::gen::Dtype(1050629)
, /* mDtypeC */ trtllm::gen::Dtype(1050629)
, /* mDtypeMmaA */ trtllm::gen::Dtype(1050629)
, /* mDtypeMmaB */ trtllm::gen::Dtype(1050629)
, /* mEnablesEarlyExit */ 0
, /* mEnablesDelayedEarlyExit */ 0
, /* mEnablesGlobalPtxKnobs */ 1
, /* mEpilogueLdtmDps */ 16
, /* mEpilogueLdtmBits */ 256
, /* mEpilogueTileM */ 128
, /* mEpilogueTileN */ 128
, /* mGridTriggerSecondaryA */ 0
, /* mGridTriggerSecondaryB */ 0
, /* mGridWaitForPrimaryEarlyExit */ 1
, /* mGridWaitForPrimaryA */ 1
, /* mGridWaitForPrimaryB */ 1
, /* mHoistLoadTaskInit */ 1
, /* mHoistMmaTaskTryWaits */ 0
, /* mK */ 2048
, /* mKernelTraits */ {}
, /* mLayoutA */ gemm::MatrixLayout(0)
, /* mLayoutB */ gemm::MatrixLayout(0)
, /* mM */ 256
, /* mMmaK */ 32
, /* mMmaKind */ trtllm::gen::MmaKind(2)
, /* mMmaM */ 128
, /* mMmaN */ 128
, /* mMockAllReduce */ 0
, /* mN */ 256
, /* mNumSlicesForSplitK */ 1
, /* mNumSlicesForSliceK */ 1
, /* mNumStages */ 2
, /* mNumStagesMma */ 1
, /* mNumStagesMmaWithinWorkTile */ 1
, /* mNumStagesMmaAcrossWorkTile */ 1
, /* mNumStagesWorkId */ 3
, /* mOutputDebugTensors */ 0
, /* mPatchF2fp */ 0
, /* mUseShuffledMatrixA */ 0
, /* mSliceK */ 0
, /* mSplitK */ gemm::SplitK(0)
, /* mTransposeMmaOutput */ 0
, /* mTileM */ 128
, /* mTileN */ 128
, /* mTileK */ 256
, /* mUseUnrollLoop2xForMma */ 1
, /* mUseCustomMmaSchedule */ 1
, /* mUseHoistTryWaitForCustomMmaSchedule */ 0
, /* mUseDeepSeekFp8 */ 0
, /* mUsePerTokenSfA */ 0
, /* mUsePerTokenSfB */ 0
, /* mUseTmaStore */ 1
, /* mUseTwoTmaLoadWarps */ 1
, /* mUseTwoMmaWarps */ 0
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(3)
, /* mSfLayoutC */ trtllm::gen::SfLayout(3)
, /* mSfReshapeFactor */ 1
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, /* mActType */ gemmGatedAct::ActType(0)
, /* mClampBeforeAct */ 0
 }, gemm::SmVersion::Sm100a},
{GemmGatedActKernel_E4m3_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x1_16dp256b_TN_transOut_schedS_swiGlu_sm100a_cubin, GemmGatedActKernel_E4m3_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x1_16dp256b_TN_transOut_schedS_swiGlu_sm100a_cubin_len, 111616, "gemmGatedActKernel_E4m3_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x1_16dp256b_TN_transOut_schedS_swiGlu_sm100a", 224, "f5f90e279514f6d8322964082bb1b019acfc8d3a596b5ed382fc6bb9e8a23608", { { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
, /* mBiasType */ gemm::BiasType(0)
, /* mBlockK */ -1
, /* mClusterDimX */ 1
, /* mClusterDimY */ 1
, /* mClusterDimZ */ 1
, /* mDtypeAcc */ trtllm::gen::Dtype(1056776)
, /* mDtypeA */ trtllm::gen::Dtype(1050629)
, /* mDtypeB */ trtllm::gen::Dtype(1050629)
, /* mDtypeC */ trtllm::gen::Dtype(1050629)
, /* mDtypeMmaA */ trtllm::gen::Dtype(1050629)
, /* mDtypeMmaB */ trtllm::gen::Dtype(1050629)
, /* mEnablesEarlyExit */ 0
, /* mEnablesDelayedEarlyExit */ 0
, /* mEnablesGlobalPtxKnobs */ 1
, /* mEpilogueLdtmDps */ 16
, /* mEpilogueLdtmBits */ 256
, /* mEpilogueTileM */ 128
, /* mEpilogueTileN */ 8
, /* mGridTriggerSecondaryA */ 0
, /* mGridTriggerSecondaryB */ 0
, /* mGridWaitForPrimaryEarlyExit */ 1
, /* mGridWaitForPrimaryA */ 1
, /* mGridWaitForPrimaryB */ 1
, /* mHoistLoadTaskInit */ 1
, /* mHoistMmaTaskTryWaits */ 0
, /* mK */ 2048
, /* mKernelTraits */ {}
, /* mLayoutA */ gemm::MatrixLayout(0)
, /* mLayoutB */ gemm::MatrixLayout(0)
, /* mM */ 256
, /* mMmaK */ 32
, /* mMmaKind */ trtllm::gen::MmaKind(2)
, /* mMmaM */ 128
, /* mMmaN */ 8
, /* mMockAllReduce */ 0
, /* mN */ 256
, /* mNumSlicesForSplitK */ 1
, /* mNumSlicesForSliceK */ 1
, /* mNumStages */ 3
, /* mNumStagesMma */ 1
, /* mNumStagesMmaWithinWorkTile */ 1
, /* mNumStagesMmaAcrossWorkTile */ 1
, /* mNumStagesWorkId */ 3
, /* mOutputDebugTensors */ 0
, /* mPatchF2fp */ 0
, /* mUseShuffledMatrixA */ 1
, /* mSliceK */ 0
, /* mSplitK */ gemm::SplitK(0)
, /* mTransposeMmaOutput */ 1
, /* mTileM */ 128
, /* mTileN */ 8
, /* mTileK */ 256
, /* mUseUnrollLoop2xForMma */ 1
, /* mUseCustomMmaSchedule */ 1
, /* mUseHoistTryWaitForCustomMmaSchedule */ 0
, /* mUseDeepSeekFp8 */ 0
, /* mUsePerTokenSfA */ 0
, /* mUsePerTokenSfB */ 0
, /* mUseTmaStore */ 1
, /* mUseTwoTmaLoadWarps */ 1
, /* mUseTwoMmaWarps */ 0
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(3)
, /* mSfLayoutC */ trtllm::gen::SfLayout(3)
, /* mSfReshapeFactor */ 1
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, /* mActType */ gemmGatedAct::ActType(0)
, /* mClampBeforeAct */ 0
 }, gemm::SmVersion::Sm100a},
{GemmGatedActKernel_E4m3_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100a_cubin, GemmGatedActKernel_E4m3_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100a_cubin_len, 110592, "gemmGatedActKernel_E4m3_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100a", 224, "41477079616a5969c5e3521238b1ec32d8f6a09680324b7a732d144ff3b18d07", { { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
, /* mBiasType */ gemm::BiasType(0)
, /* mBlockK */ -1
, /* mClusterDimX */ 1
, /* mClusterDimY */ 1
, /* mClusterDimZ */ 4
, /* mDtypeAcc */ trtllm::gen::Dtype(1056776)
, /* mDtypeA */ trtllm::gen::Dtype(1050629)
, /* mDtypeB */ trtllm::gen::Dtype(1050629)
, /* mDtypeC */ trtllm::gen::Dtype(1050629)
, /* mDtypeMmaA */ trtllm::gen::Dtype(1050629)
, /* mDtypeMmaB */ trtllm::gen::Dtype(1050629)
, /* mEnablesEarlyExit */ 0
, /* mEnablesDelayedEarlyExit */ 0
, /* mEnablesGlobalPtxKnobs */ 1
, /* mEpilogueLdtmDps */ 16
, /* mEpilogueLdtmBits */ 256
, /* mEpilogueTileM */ 128
, /* mEpilogueTileN */ 8
, /* mGridTriggerSecondaryA */ 0
, /* mGridTriggerSecondaryB */ 0
, /* mGridWaitForPrimaryEarlyExit */ 1
, /* mGridWaitForPrimaryA */ 1
, /* mGridWaitForPrimaryB */ 1
, /* mHoistLoadTaskInit */ 1
, /* mHoistMmaTaskTryWaits */ 0
, /* mK */ 2048
, /* mKernelTraits */ {}
, /* mLayoutA */ gemm::MatrixLayout(0)
, /* mLayoutB */ gemm::MatrixLayout(0)
, /* mM */ 256
, /* mMmaK */ 32
, /* mMmaKind */ trtllm::gen::MmaKind(2)
, /* mMmaM */ 128
, /* mMmaN */ 8
, /* mMockAllReduce */ 0
, /* mN */ 256
, /* mNumSlicesForSplitK */ 4
, /* mNumSlicesForSliceK */ 1
, /* mNumStages */ 3
, /* mNumStagesMma */ 1
, /* mNumStagesMmaWithinWorkTile */ 1
, /* mNumStagesMmaAcrossWorkTile */ 1
, /* mNumStagesWorkId */ 3
, /* mOutputDebugTensors */ 0
, /* mPatchF2fp */ 0
, /* mUseShuffledMatrixA */ 1
, /* mSliceK */ 0
, /* mSplitK */ gemm::SplitK(2)
, /* mTransposeMmaOutput */ 1
, /* mTileM */ 128
, /* mTileN */ 8
, /* mTileK */ 256
, /* mUseUnrollLoop2xForMma */ 1
, /* mUseCustomMmaSchedule */ 1
, /* mUseHoistTryWaitForCustomMmaSchedule */ 0
, /* mUseDeepSeekFp8 */ 0
, /* mUsePerTokenSfA */ 0
, /* mUsePerTokenSfB */ 0
, /* mUseTmaStore */ 1
, /* mUseTwoTmaLoadWarps */ 1
, /* mUseTwoMmaWarps */ 0
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(3)
, /* mSfLayoutC */ trtllm::gen::SfLayout(3)
, /* mSfReshapeFactor */ 1
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, /* mActType */ gemmGatedAct::ActType(0)
, /* mClampBeforeAct */ 0
 }, gemm::SmVersion::Sm100a},
{GemmGatedActKernel_Fp16_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100a_cubin, GemmGatedActKernel_Fp16_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100a_cubin_len, 86016, "gemmGatedActKernel_Fp16_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100a", 448, "754cf37e731c40683a27223bc2e0c3bd8dfb39b308f0ef16a9e9d857479f000e", { { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
, /* mBiasType */ gemm::BiasType(0)
, /* mBlockK */ -1
, /* mClusterDimX */ 1
, /* mClusterDimY */ 1
, /* mClusterDimZ */ 4
, /* mDtypeAcc */ trtllm::gen::Dtype(1056776)
, /* mDtypeA */ trtllm::gen::Dtype(17826818)
, /* mDtypeB */ trtllm::gen::Dtype(17826818)
, /* mDtypeC */ trtllm::gen::Dtype(1052679)
, /* mDtypeMmaA */ trtllm::gen::Dtype(17826818)
, /* mDtypeMmaB */ trtllm::gen::Dtype(17826818)
, /* mEnablesEarlyExit */ 0
, /* mEnablesDelayedEarlyExit */ 0
, /* mEnablesGlobalPtxKnobs */ 1
, /* mEpilogueLdtmDps */ 16
, /* mEpilogueLdtmBits */ 256
, /* mEpilogueTileM */ 128
, /* mEpilogueTileN */ 8
, /* mGridTriggerSecondaryA */ 0
, /* mGridTriggerSecondaryB */ 0
, /* mGridWaitForPrimaryEarlyExit */ 1
, /* mGridWaitForPrimaryA */ 1
, /* mGridWaitForPrimaryB */ 1
, /* mHoistLoadTaskInit */ 1
, /* mHoistMmaTaskTryWaits */ 0
, /* mK */ 2048
, /* mKernelTraits */ {}
, /* mLayoutA */ gemm::MatrixLayout(0)
, /* mLayoutB */ gemm::MatrixLayout(0)
, /* mM */ 256
, /* mMmaK */ 64
, /* mMmaKind */ trtllm::gen::MmaKind(4)
, /* mMmaM */ 128
, /* mMmaN */ 8
, /* mMockAllReduce */ 0
, /* mN */ 256
, /* mNumSlicesForSplitK */ 4
, /* mNumSlicesForSliceK */ 1
, /* mNumStages */ 4
, /* mNumStagesMma */ 1
, /* mNumStagesMmaWithinWorkTile */ 1
, /* mNumStagesMmaAcrossWorkTile */ 1
, /* mNumStagesWorkId */ 3
, /* mOutputDebugTensors */ 0
, /* mPatchF2fp */ 0
, /* mUseShuffledMatrixA */ 1
, /* mSliceK */ 0
, /* mSplitK */ gemm::SplitK(2)
, /* mTransposeMmaOutput */ 1
, /* mTileM */ 128
, /* mTileN */ 8
, /* mTileK */ 256
, /* mUseUnrollLoop2xForMma */ 1
, /* mUseCustomMmaSchedule */ 1
, /* mUseHoistTryWaitForCustomMmaSchedule */ 0
, /* mUseDeepSeekFp8 */ 0
, /* mUsePerTokenSfA */ 0
, /* mUsePerTokenSfB */ 0
, /* mUseTmaStore */ 1
, /* mUseTwoTmaLoadWarps */ 1
, /* mUseTwoMmaWarps */ 0
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(1)
, /* mSfLayoutC */ trtllm::gen::SfLayout(1)
, /* mSfReshapeFactor */ 1
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, /* mActType */ gemmGatedAct::ActType(0)
, /* mClampBeforeAct */ 0
 }, gemm::SmVersion::Sm100a},
{GemmGatedActKernel_Fp16_E4m3E4m3_Fp32_t128x128x256u2_s2_et128x128_m128x128x32_cga1x1x1_16dp256b_TN_schedS_swiGlu_sm100a_cubin, GemmGatedActKernel_Fp16_E4m3E4m3_Fp32_t128x128x256u2_s2_et128x128_m128x128x32_cga1x1x1_16dp256b_TN_schedS_swiGlu_sm100a_cubin_len, 168960, "gemmGatedActKernel_Fp16_E4m3E4m3_Fp32_t128x128x256u2_s2_et128x128_m128x128x32_cga1x1x1_16dp256b_TN_schedS_swiGlu_sm100a", 224, "76ccade87557dfae73216d5531b6ef8ed098af960daca62ec7eabac3d1fb72df", { { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
, /* mBiasType */ gemm::BiasType(0)
, /* mBlockK */ -1
, /* mClusterDimX */ 1
, /* mClusterDimY */ 1
, /* mClusterDimZ */ 1
, /* mDtypeAcc */ trtllm::gen::Dtype(1056776)
, /* mDtypeA */ trtllm::gen::Dtype(1050629)
, /* mDtypeB */ trtllm::gen::Dtype(1050629)
, /* mDtypeC */ trtllm::gen::Dtype(1052679)
, /* mDtypeMmaA */ trtllm::gen::Dtype(1050629)
, /* mDtypeMmaB */ trtllm::gen::Dtype(1050629)
, /* mEnablesEarlyExit */ 0
, /* mEnablesDelayedEarlyExit */ 0
, /* mEnablesGlobalPtxKnobs */ 1
, /* mEpilogueLdtmDps */ 16
, /* mEpilogueLdtmBits */ 256
, /* mEpilogueTileM */ 128
, /* mEpilogueTileN */ 128
, /* mGridTriggerSecondaryA */ 0
, /* mGridTriggerSecondaryB */ 0
, /* mGridWaitForPrimaryEarlyExit */ 1
, /* mGridWaitForPrimaryA */ 1
, /* mGridWaitForPrimaryB */ 1
, /* mHoistLoadTaskInit */ 1
, /* mHoistMmaTaskTryWaits */ 0
, /* mK */ 2048
, /* mKernelTraits */ {}
, /* mLayoutA */ gemm::MatrixLayout(0)
, /* mLayoutB */ gemm::MatrixLayout(0)
, /* mM */ 256
, /* mMmaK */ 32
, /* mMmaKind */ trtllm::gen::MmaKind(2)
, /* mMmaM */ 128
, /* mMmaN */ 128
, /* mMockAllReduce */ 0
, /* mN */ 256
, /* mNumSlicesForSplitK */ 1
, /* mNumSlicesForSliceK */ 1
, /* mNumStages */ 2
, /* mNumStagesMma */ 1
, /* mNumStagesMmaWithinWorkTile */ 1
, /* mNumStagesMmaAcrossWorkTile */ 1
, /* mNumStagesWorkId */ 3
, /* mOutputDebugTensors */ 0
, /* mPatchF2fp */ 0
, /* mUseShuffledMatrixA */ 0
, /* mSliceK */ 0
, /* mSplitK */ gemm::SplitK(0)
, /* mTransposeMmaOutput */ 0
, /* mTileM */ 128
, /* mTileN */ 128
, /* mTileK */ 256
, /* mUseUnrollLoop2xForMma */ 1
, /* mUseCustomMmaSchedule */ 1
, /* mUseHoistTryWaitForCustomMmaSchedule */ 0
, /* mUseDeepSeekFp8 */ 0
, /* mUsePerTokenSfA */ 0
, /* mUsePerTokenSfB */ 0
, /* mUseTmaStore */ 1
, /* mUseTwoTmaLoadWarps */ 1
, /* mUseTwoMmaWarps */ 0
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(3)
, /* mSfLayoutC */ trtllm::gen::SfLayout(3)
, /* mSfReshapeFactor */ 1
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, /* mActType */ gemmGatedAct::ActType(0)
, /* mClampBeforeAct */ 0
 }, gemm::SmVersion::Sm100a},
{GemmGatedActKernel_Fp16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x1_16dp256b_TN_transOut_schedS_swiGlu_sm100a_cubin, GemmGatedActKernel_Fp16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x1_16dp256b_TN_transOut_schedS_swiGlu_sm100a_cubin_len, 112640, "gemmGatedActKernel_Fp16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x1_16dp256b_TN_transOut_schedS_swiGlu_sm100a", 224, "394e76b48c40fd0514773ca4078cf01274e2d0bff804bd176bab44d2a832021b", { { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
, /* mBiasType */ gemm::BiasType(0)
, /* mBlockK */ -1
, /* mClusterDimX */ 1
, /* mClusterDimY */ 1
, /* mClusterDimZ */ 1
, /* mDtypeAcc */ trtllm::gen::Dtype(1056776)
, /* mDtypeA */ trtllm::gen::Dtype(1050629)
, /* mDtypeB */ trtllm::gen::Dtype(1050629)
, /* mDtypeC */ trtllm::gen::Dtype(1052679)
, /* mDtypeMmaA */ trtllm::gen::Dtype(1050629)
, /* mDtypeMmaB */ trtllm::gen::Dtype(1050629)
, /* mEnablesEarlyExit */ 0
, /* mEnablesDelayedEarlyExit */ 0
, /* mEnablesGlobalPtxKnobs */ 1
, /* mEpilogueLdtmDps */ 16
, /* mEpilogueLdtmBits */ 256
, /* mEpilogueTileM */ 128
, /* mEpilogueTileN */ 8
, /* mGridTriggerSecondaryA */ 0
, /* mGridTriggerSecondaryB */ 0
, /* mGridWaitForPrimaryEarlyExit */ 1
, /* mGridWaitForPrimaryA */ 1
, /* mGridWaitForPrimaryB */ 1
, /* mHoistLoadTaskInit */ 1
, /* mHoistMmaTaskTryWaits */ 0
, /* mK */ 2048
, /* mKernelTraits */ {}
, /* mLayoutA */ gemm::MatrixLayout(0)
, /* mLayoutB */ gemm::MatrixLayout(0)
, /* mM */ 256
, /* mMmaK */ 32
, /* mMmaKind */ trtllm::gen::MmaKind(2)
, /* mMmaM */ 128
, /* mMmaN */ 8
, /* mMockAllReduce */ 0
, /* mN */ 256
, /* mNumSlicesForSplitK */ 1
, /* mNumSlicesForSliceK */ 1
, /* mNumStages */ 3
, /* mNumStagesMma */ 1
, /* mNumStagesMmaWithinWorkTile */ 1
, /* mNumStagesMmaAcrossWorkTile */ 1
, /* mNumStagesWorkId */ 3
, /* mOutputDebugTensors */ 0
, /* mPatchF2fp */ 0
, /* mUseShuffledMatrixA */ 1
, /* mSliceK */ 0
, /* mSplitK */ gemm::SplitK(0)
, /* mTransposeMmaOutput */ 1
, /* mTileM */ 128
, /* mTileN */ 8
, /* mTileK */ 256
, /* mUseUnrollLoop2xForMma */ 1
, /* mUseCustomMmaSchedule */ 1
, /* mUseHoistTryWaitForCustomMmaSchedule */ 0
, /* mUseDeepSeekFp8 */ 0
, /* mUsePerTokenSfA */ 0
, /* mUsePerTokenSfB */ 0
, /* mUseTmaStore */ 1
, /* mUseTwoTmaLoadWarps */ 1
, /* mUseTwoMmaWarps */ 0
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(3)
, /* mSfLayoutC */ trtllm::gen::SfLayout(3)
, /* mSfReshapeFactor */ 1
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, /* mActType */ gemmGatedAct::ActType(0)
, /* mClampBeforeAct */ 0
 }, gemm::SmVersion::Sm100a},
{GemmGatedActKernel_Fp16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100a_cubin, GemmGatedActKernel_Fp16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100a_cubin_len, 110592, "gemmGatedActKernel_Fp16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100a", 224, "22842a61b00e9b98e5d2435e53779c43140031479c75f1a7a9fae5673d3a9d50", { { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
, /* mBiasType */ gemm::BiasType(0)
, /* mBlockK */ -1
, /* mClusterDimX */ 1
, /* mClusterDimY */ 1
, /* mClusterDimZ */ 4
, /* mDtypeAcc */ trtllm::gen::Dtype(1056776)
, /* mDtypeA */ trtllm::gen::Dtype(1050629)
, /* mDtypeB */ trtllm::gen::Dtype(1050629)
, /* mDtypeC */ trtllm::gen::Dtype(1052679)
, /* mDtypeMmaA */ trtllm::gen::Dtype(1050629)
, /* mDtypeMmaB */ trtllm::gen::Dtype(1050629)
, /* mEnablesEarlyExit */ 0
, /* mEnablesDelayedEarlyExit */ 0
, /* mEnablesGlobalPtxKnobs */ 1
, /* mEpilogueLdtmDps */ 16
, /* mEpilogueLdtmBits */ 256
, /* mEpilogueTileM */ 128
, /* mEpilogueTileN */ 8
, /* mGridTriggerSecondaryA */ 0
, /* mGridTriggerSecondaryB */ 0
, /* mGridWaitForPrimaryEarlyExit */ 1
, /* mGridWaitForPrimaryA */ 1
, /* mGridWaitForPrimaryB */ 1
, /* mHoistLoadTaskInit */ 1
, /* mHoistMmaTaskTryWaits */ 0
, /* mK */ 2048
, /* mKernelTraits */ {}
, /* mLayoutA */ gemm::MatrixLayout(0)
, /* mLayoutB */ gemm::MatrixLayout(0)
, /* mM */ 256
, /* mMmaK */ 32
, /* mMmaKind */ trtllm::gen::MmaKind(2)
, /* mMmaM */ 128
, /* mMmaN */ 8
, /* mMockAllReduce */ 0
, /* mN */ 256
, /* mNumSlicesForSplitK */ 4
, /* mNumSlicesForSliceK */ 1
, /* mNumStages */ 3
, /* mNumStagesMma */ 1
, /* mNumStagesMmaWithinWorkTile */ 1
, /* mNumStagesMmaAcrossWorkTile */ 1
, /* mNumStagesWorkId */ 3
, /* mOutputDebugTensors */ 0
, /* mPatchF2fp */ 0
, /* mUseShuffledMatrixA */ 1
, /* mSliceK */ 0
, /* mSplitK */ gemm::SplitK(2)
, /* mTransposeMmaOutput */ 1
, /* mTileM */ 128
, /* mTileN */ 8
, /* mTileK */ 256
, /* mUseUnrollLoop2xForMma */ 1
, /* mUseCustomMmaSchedule */ 1
, /* mUseHoistTryWaitForCustomMmaSchedule */ 0
, /* mUseDeepSeekFp8 */ 0
, /* mUsePerTokenSfA */ 0
, /* mUsePerTokenSfB */ 0
, /* mUseTmaStore */ 1
, /* mUseTwoTmaLoadWarps */ 1
, /* mUseTwoMmaWarps */ 0
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(3)
, /* mSfLayoutC */ trtllm::gen::SfLayout(3)
, /* mSfReshapeFactor */ 1
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, /* mActType */ gemmGatedAct::ActType(0)
, /* mClampBeforeAct */ 0
 }, gemm::SmVersion::Sm100a},
{GemmGatedActKernel_Fp32_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100a_cubin, GemmGatedActKernel_Fp32_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100a_cubin_len, 86016, "gemmGatedActKernel_Fp32_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100a", 448, "feb2bb2aa2e88f8f3dfd8de985601474e5147b926b6d067884ed67730eda4b66", { { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
, /* mBiasType */ gemm::BiasType(0)
, /* mBlockK */ -1
, /* mClusterDimX */ 1
, /* mClusterDimY */ 1
, /* mClusterDimZ */ 4
, /* mDtypeAcc */ trtllm::gen::Dtype(1056776)
, /* mDtypeA */ trtllm::gen::Dtype(17826818)
, /* mDtypeB */ trtllm::gen::Dtype(17826818)
, /* mDtypeC */ trtllm::gen::Dtype(1056776)
, /* mDtypeMmaA */ trtllm::gen::Dtype(17826818)
, /* mDtypeMmaB */ trtllm::gen::Dtype(17826818)
, /* mEnablesEarlyExit */ 0
, /* mEnablesDelayedEarlyExit */ 0
, /* mEnablesGlobalPtxKnobs */ 1
, /* mEpilogueLdtmDps */ 16
, /* mEpilogueLdtmBits */ 256
, /* mEpilogueTileM */ 128
, /* mEpilogueTileN */ 8
, /* mGridTriggerSecondaryA */ 0
, /* mGridTriggerSecondaryB */ 0
, /* mGridWaitForPrimaryEarlyExit */ 1
, /* mGridWaitForPrimaryA */ 1
, /* mGridWaitForPrimaryB */ 1
, /* mHoistLoadTaskInit */ 1
, /* mHoistMmaTaskTryWaits */ 0
, /* mK */ 2048
, /* mKernelTraits */ {}
, /* mLayoutA */ gemm::MatrixLayout(0)
, /* mLayoutB */ gemm::MatrixLayout(0)
, /* mM */ 256
, /* mMmaK */ 64
, /* mMmaKind */ trtllm::gen::MmaKind(4)
, /* mMmaM */ 128
, /* mMmaN */ 8
, /* mMockAllReduce */ 0
, /* mN */ 256
, /* mNumSlicesForSplitK */ 4
, /* mNumSlicesForSliceK */ 1
, /* mNumStages */ 4
, /* mNumStagesMma */ 1
, /* mNumStagesMmaWithinWorkTile */ 1
, /* mNumStagesMmaAcrossWorkTile */ 1
, /* mNumStagesWorkId */ 3
, /* mOutputDebugTensors */ 0
, /* mPatchF2fp */ 0
, /* mUseShuffledMatrixA */ 1
, /* mSliceK */ 0
, /* mSplitK */ gemm::SplitK(2)
, /* mTransposeMmaOutput */ 1
, /* mTileM */ 128
, /* mTileN */ 8
, /* mTileK */ 256
, /* mUseUnrollLoop2xForMma */ 1
, /* mUseCustomMmaSchedule */ 1
, /* mUseHoistTryWaitForCustomMmaSchedule */ 0
, /* mUseDeepSeekFp8 */ 0
, /* mUsePerTokenSfA */ 0
, /* mUsePerTokenSfB */ 0
, /* mUseTmaStore */ 1
, /* mUseTwoTmaLoadWarps */ 1
, /* mUseTwoMmaWarps */ 0
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(1)
, /* mSfLayoutC */ trtllm::gen::SfLayout(1)
, /* mSfReshapeFactor */ 1
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, /* mActType */ gemmGatedAct::ActType(0)
, /* mClampBeforeAct */ 0
 }, gemm::SmVersion::Sm100a},
#endif // EXCLUDE_SM_100
};
// clang-format on
} // namespace kernels
} // namespace tensorrt_llm
} // namespace gemmGatedAct
