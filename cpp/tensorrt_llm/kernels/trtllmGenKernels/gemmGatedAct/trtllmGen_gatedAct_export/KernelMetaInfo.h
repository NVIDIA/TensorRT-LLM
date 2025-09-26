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

#define TLLM_GEN_COMMIT "541a9315-dirty"
#define TLLM_GEN_EXPORT_VERSION "7.0.4.0"

static constexpr size_t tllmGenGemmGatedActListLen = 13;

#ifndef EXCLUDE_SM_100
extern unsigned char GemmGatedActKernel_Bfloat16_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100f_cubin[];
extern unsigned char GemmGatedActKernel_Bfloat16_E4m3E4m3_Fp32_t128x128x256u2_s2_et128x128_m128x128x32_cga1x1x1_16dp256b_TN_schedS_swiGlu_sm100f_cubin[];
extern unsigned char GemmGatedActKernel_Bfloat16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x1_16dp256b_TN_transOut_schedS_swiGlu_sm100f_cubin[];
extern unsigned char GemmGatedActKernel_Bfloat16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100f_cubin[];
extern unsigned char GemmGatedActKernel_E2m1_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100f_cubin[];
extern unsigned char GemmGatedActKernel_E4m3_E4m3E4m3_Fp32_t128x128x256u2_s2_et128x128_m128x128x32_cga1x1x1_16dp256b_TN_schedS_swiGlu_sm100f_cubin[];
extern unsigned char GemmGatedActKernel_E4m3_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x1_16dp256b_TN_transOut_schedS_swiGlu_sm100f_cubin[];
extern unsigned char GemmGatedActKernel_E4m3_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100f_cubin[];
extern unsigned char GemmGatedActKernel_Fp16_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100f_cubin[];
extern unsigned char GemmGatedActKernel_Fp16_E4m3E4m3_Fp32_t128x128x256u2_s2_et128x128_m128x128x32_cga1x1x1_16dp256b_TN_schedS_swiGlu_sm100f_cubin[];
extern unsigned char GemmGatedActKernel_Fp16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x1_16dp256b_TN_transOut_schedS_swiGlu_sm100f_cubin[];
extern unsigned char GemmGatedActKernel_Fp16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100f_cubin[];
extern unsigned char GemmGatedActKernel_Fp32_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100f_cubin[];
#endif // EXCLUDE_SM_100

#ifndef EXCLUDE_SM_100
extern unsigned int GemmGatedActKernel_Bfloat16_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100f_cubin_len;
extern unsigned int GemmGatedActKernel_Bfloat16_E4m3E4m3_Fp32_t128x128x256u2_s2_et128x128_m128x128x32_cga1x1x1_16dp256b_TN_schedS_swiGlu_sm100f_cubin_len;
extern unsigned int GemmGatedActKernel_Bfloat16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x1_16dp256b_TN_transOut_schedS_swiGlu_sm100f_cubin_len;
extern unsigned int GemmGatedActKernel_Bfloat16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100f_cubin_len;
extern unsigned int GemmGatedActKernel_E2m1_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100f_cubin_len;
extern unsigned int GemmGatedActKernel_E4m3_E4m3E4m3_Fp32_t128x128x256u2_s2_et128x128_m128x128x32_cga1x1x1_16dp256b_TN_schedS_swiGlu_sm100f_cubin_len;
extern unsigned int GemmGatedActKernel_E4m3_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x1_16dp256b_TN_transOut_schedS_swiGlu_sm100f_cubin_len;
extern unsigned int GemmGatedActKernel_E4m3_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100f_cubin_len;
extern unsigned int GemmGatedActKernel_Fp16_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100f_cubin_len;
extern unsigned int GemmGatedActKernel_Fp16_E4m3E4m3_Fp32_t128x128x256u2_s2_et128x128_m128x128x32_cga1x1x1_16dp256b_TN_schedS_swiGlu_sm100f_cubin_len;
extern unsigned int GemmGatedActKernel_Fp16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x1_16dp256b_TN_transOut_schedS_swiGlu_sm100f_cubin_len;
extern unsigned int GemmGatedActKernel_Fp16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100f_cubin_len;
extern unsigned int GemmGatedActKernel_Fp32_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100f_cubin_len;
#endif // EXCLUDE_SM_100


static const gemmGatedAct::GemmGatedActConfig tllmGenGemmGatedActList[] = {
#ifndef EXCLUDE_SM_100
{GemmGatedActKernel_Bfloat16_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100f_cubin, GemmGatedActKernel_Bfloat16_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100f_cubin_len, 86016, "gemmGatedActKernel_Bfloat16_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100f", 448, "9676717a6339c1f0ed39de9935975d4d60468440174326380990d70d75cf357a", { { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
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
, /* mSfBlockSizeA */ std::nullopt
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(1)
, /* mSfLayoutC */ trtllm::gen::SfLayout(1)
, /* mSfReshapeFactor */ 1
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, /* mActType */ gemmGatedAct::ActType(0)
, /* mClampBeforeAct */ 0
 }, gemm::SmVersion::Sm100a},
{GemmGatedActKernel_Bfloat16_E4m3E4m3_Fp32_t128x128x256u2_s2_et128x128_m128x128x32_cga1x1x1_16dp256b_TN_schedS_swiGlu_sm100f_cubin, GemmGatedActKernel_Bfloat16_E4m3E4m3_Fp32_t128x128x256u2_s2_et128x128_m128x128x32_cga1x1x1_16dp256b_TN_schedS_swiGlu_sm100f_cubin_len, 168960, "gemmGatedActKernel_Bfloat16_E4m3E4m3_Fp32_t128x128x256u2_s2_et128x128_m128x128x32_cga1x1x1_16dp256b_TN_schedS_swiGlu_sm100f", 224, "ec67e15fd7862d51b19dda2176f8129e4592a748eefb3e48b6a3fbc41868d6f2", { { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
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
, /* mSfBlockSizeA */ std::nullopt
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(3)
, /* mSfLayoutC */ trtllm::gen::SfLayout(3)
, /* mSfReshapeFactor */ 1
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, /* mActType */ gemmGatedAct::ActType(0)
, /* mClampBeforeAct */ 0
 }, gemm::SmVersion::Sm100a},
{GemmGatedActKernel_Bfloat16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x1_16dp256b_TN_transOut_schedS_swiGlu_sm100f_cubin, GemmGatedActKernel_Bfloat16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x1_16dp256b_TN_transOut_schedS_swiGlu_sm100f_cubin_len, 112640, "gemmGatedActKernel_Bfloat16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x1_16dp256b_TN_transOut_schedS_swiGlu_sm100f", 224, "caa8817305ecc3e91818c767f6a7989b7db55b6c49232c9a2e32d1c907228684", { { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
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
, /* mSfBlockSizeA */ std::nullopt
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(3)
, /* mSfLayoutC */ trtllm::gen::SfLayout(3)
, /* mSfReshapeFactor */ 1
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, /* mActType */ gemmGatedAct::ActType(0)
, /* mClampBeforeAct */ 0
 }, gemm::SmVersion::Sm100a},
{GemmGatedActKernel_Bfloat16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100f_cubin, GemmGatedActKernel_Bfloat16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100f_cubin_len, 110592, "gemmGatedActKernel_Bfloat16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100f", 224, "48fc7f954cbb918fab79ff2ee54159c00da0687ee4ca60036b9fe5c746d97a2c", { { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
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
, /* mSfBlockSizeA */ std::nullopt
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(3)
, /* mSfLayoutC */ trtllm::gen::SfLayout(3)
, /* mSfReshapeFactor */ 1
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, /* mActType */ gemmGatedAct::ActType(0)
, /* mClampBeforeAct */ 0
 }, gemm::SmVersion::Sm100a},
{GemmGatedActKernel_E2m1_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100f_cubin, GemmGatedActKernel_E2m1_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100f_cubin_len, 86016, "gemmGatedActKernel_E2m1_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100f", 448, "0d61dc1cfefc5ca6d5e90afa70347394930cbf4c79eea20c1518de8e1222da54", { { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
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
, /* mSfBlockSizeA */ std::nullopt
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(1)
, /* mSfLayoutC */ trtllm::gen::SfLayout(1)
, /* mSfReshapeFactor */ 1
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, /* mActType */ gemmGatedAct::ActType(0)
, /* mClampBeforeAct */ 0
 }, gemm::SmVersion::Sm100a},
{GemmGatedActKernel_E4m3_E4m3E4m3_Fp32_t128x128x256u2_s2_et128x128_m128x128x32_cga1x1x1_16dp256b_TN_schedS_swiGlu_sm100f_cubin, GemmGatedActKernel_E4m3_E4m3E4m3_Fp32_t128x128x256u2_s2_et128x128_m128x128x32_cga1x1x1_16dp256b_TN_schedS_swiGlu_sm100f_cubin_len, 152576, "gemmGatedActKernel_E4m3_E4m3E4m3_Fp32_t128x128x256u2_s2_et128x128_m128x128x32_cga1x1x1_16dp256b_TN_schedS_swiGlu_sm100f", 224, "403cbe5700ce7b2749b49c595661ceeedbf6e50c580709a37a610cc54a239b4d", { { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
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
, /* mSfBlockSizeA */ std::nullopt
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(3)
, /* mSfLayoutC */ trtllm::gen::SfLayout(3)
, /* mSfReshapeFactor */ 1
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, /* mActType */ gemmGatedAct::ActType(0)
, /* mClampBeforeAct */ 0
 }, gemm::SmVersion::Sm100a},
{GemmGatedActKernel_E4m3_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x1_16dp256b_TN_transOut_schedS_swiGlu_sm100f_cubin, GemmGatedActKernel_E4m3_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x1_16dp256b_TN_transOut_schedS_swiGlu_sm100f_cubin_len, 111616, "gemmGatedActKernel_E4m3_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x1_16dp256b_TN_transOut_schedS_swiGlu_sm100f", 224, "c2d4a41c6e16594d2c46fece2ee0a15e783d4c914bf698c4b7fd7ac0c14c2dfc", { { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
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
, /* mSfBlockSizeA */ std::nullopt
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(3)
, /* mSfLayoutC */ trtllm::gen::SfLayout(3)
, /* mSfReshapeFactor */ 1
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, /* mActType */ gemmGatedAct::ActType(0)
, /* mClampBeforeAct */ 0
 }, gemm::SmVersion::Sm100a},
{GemmGatedActKernel_E4m3_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100f_cubin, GemmGatedActKernel_E4m3_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100f_cubin_len, 110592, "gemmGatedActKernel_E4m3_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100f", 224, "20c18093aeca9040c7914535312f7980bf9cfbb77430dc0ef9660d7a8b2f3340", { { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
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
, /* mSfBlockSizeA */ std::nullopt
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(3)
, /* mSfLayoutC */ trtllm::gen::SfLayout(3)
, /* mSfReshapeFactor */ 1
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, /* mActType */ gemmGatedAct::ActType(0)
, /* mClampBeforeAct */ 0
 }, gemm::SmVersion::Sm100a},
{GemmGatedActKernel_Fp16_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100f_cubin, GemmGatedActKernel_Fp16_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100f_cubin_len, 86016, "gemmGatedActKernel_Fp16_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100f", 448, "375daed8ff4ada0d0117299a9fd882c0598feeff34924a9d1479b1bc2a0c473e", { { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
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
, /* mSfBlockSizeA */ std::nullopt
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(1)
, /* mSfLayoutC */ trtllm::gen::SfLayout(1)
, /* mSfReshapeFactor */ 1
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, /* mActType */ gemmGatedAct::ActType(0)
, /* mClampBeforeAct */ 0
 }, gemm::SmVersion::Sm100a},
{GemmGatedActKernel_Fp16_E4m3E4m3_Fp32_t128x128x256u2_s2_et128x128_m128x128x32_cga1x1x1_16dp256b_TN_schedS_swiGlu_sm100f_cubin, GemmGatedActKernel_Fp16_E4m3E4m3_Fp32_t128x128x256u2_s2_et128x128_m128x128x32_cga1x1x1_16dp256b_TN_schedS_swiGlu_sm100f_cubin_len, 168960, "gemmGatedActKernel_Fp16_E4m3E4m3_Fp32_t128x128x256u2_s2_et128x128_m128x128x32_cga1x1x1_16dp256b_TN_schedS_swiGlu_sm100f", 224, "d0abb1065fc2517d1208f38e0716cf6fa177a165ae5c560033fd2e1e81bffe16", { { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
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
, /* mSfBlockSizeA */ std::nullopt
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(3)
, /* mSfLayoutC */ trtllm::gen::SfLayout(3)
, /* mSfReshapeFactor */ 1
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, /* mActType */ gemmGatedAct::ActType(0)
, /* mClampBeforeAct */ 0
 }, gemm::SmVersion::Sm100a},
{GemmGatedActKernel_Fp16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x1_16dp256b_TN_transOut_schedS_swiGlu_sm100f_cubin, GemmGatedActKernel_Fp16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x1_16dp256b_TN_transOut_schedS_swiGlu_sm100f_cubin_len, 112640, "gemmGatedActKernel_Fp16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x1_16dp256b_TN_transOut_schedS_swiGlu_sm100f", 224, "22fefb0cbe12f8af7719f7dd3e9271d7386f20e57d8dd2809587d0e825d13681", { { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
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
, /* mSfBlockSizeA */ std::nullopt
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(3)
, /* mSfLayoutC */ trtllm::gen::SfLayout(3)
, /* mSfReshapeFactor */ 1
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, /* mActType */ gemmGatedAct::ActType(0)
, /* mClampBeforeAct */ 0
 }, gemm::SmVersion::Sm100a},
{GemmGatedActKernel_Fp16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100f_cubin, GemmGatedActKernel_Fp16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100f_cubin_len, 110592, "gemmGatedActKernel_Fp16_E4m3E4m3_Fp32_t128x8x256u2_s3_et128x8_m128x8x32_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100f", 224, "48101b606e52817efdcfc9a17ad91b224fa8fd6c36283f5f370b8f6ae729fadc", { { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
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
, /* mSfBlockSizeA */ std::nullopt
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(3)
, /* mSfLayoutC */ trtllm::gen::SfLayout(3)
, /* mSfReshapeFactor */ 1
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, /* mActType */ gemmGatedAct::ActType(0)
, /* mClampBeforeAct */ 0
 }, gemm::SmVersion::Sm100a},
{GemmGatedActKernel_Fp32_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100f_cubin, GemmGatedActKernel_Fp32_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100f_cubin_len, 86016, "gemmGatedActKernel_Fp32_E2m1E2m1_Fp32_t128x8x256u2_s4_et128x8_m128x8x64_cga1x1x4_16dp256b_splitK4_TN_transOut_schedS_swiGlu_sm100f", 448, "0e648979b852a9c612006fa871fc73dc427ceacbd79201c7d1b4e52a0d64aec7", { { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
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
, /* mSfBlockSizeA */ std::nullopt
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
