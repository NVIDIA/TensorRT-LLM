/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION &
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

#include <optional>
#include <set>
#include <sstream>

#include "Enums.h"
#include "KernelParams.h"
#include "KernelTraits.h"
#include "trtllm/gen/CudaArchDecl.h"
#include "trtllm/gen/DtypeDecl.h"
#include "trtllm/gen/MmaDecl.h"
#include "trtllm/gen/SfLayoutDecl.h"
#include "trtllm/gen/SparsityDecl.h"
#ifndef TLLM_GEN_EXPORT_INTERFACE
#include "trtllm/gen/CudaRunner.h"
#include "trtllm/gen/GenCtx.h"
#else
#ifdef TLLM_GEN_EXPORT_FLASHINFER
#include <string>

namespace flashinfer::trtllm_cubin_loader
{
std::string getCubin(std::string const& kernelName, std::string const& sha256);
}
#endif // TLLM_GEN_EXPORT_FLASHINFER
#include <iostream>
namespace batchedGemm
{

template <typename T>
void printArgs(T arg)
{
#ifdef TLLM_GEN_DEBUG
    std::cout << arg;
#endif
}

template <typename T, typename... Args>
void printArgs(T first, Args... args)
{
    printArgs(first);
    if constexpr (sizeof...(args) > 0)
    {
        printArgs(", ");
        printArgs(args...);
    }
}

#define TLLM_CHECK_ERROR(cond, ...)                                                                                    \
    if (!(cond))                                                                                                       \
    {                                                                                                                  \
        printArgs(__VA_ARGS__);                                                                                        \
        printArgs("\n");                                                                                               \
        return false;                                                                                                  \
    }

#define TLLM_LOG_ERROR(...) TLLM_CHECK_ERROR(false, __VA_ARGS__)

#define TLLM_CHECK_ERROR_FMT(cond, ...) TLLM_CHECK_ERROR(cond, __VA_ARGS__)

#define TLLM_CHECK_WARNING(cond, ...)                                                                                  \
    if (!(cond))                                                                                                       \
    {                                                                                                                  \
        printArgs(__VA_ARGS__);                                                                                        \
        printArgs("\n");                                                                                               \
        return false;                                                                                                  \
    }

#define TLLM_LOG_WARNING(...) TLLM_CHECK_WARNING(false, __VA_ARGS__)

#define TLLM_LOG_INFO(...) TLLM_CHECK_WARNING(false, __VA_ARGS__)

#endif // TLLM_GEN_EXPORT_INTERFACE

#define GEMM_UPDATE_OR_ERROR(OPTION, VALUE)                                                                            \
    if (updateOptions)                                                                                                 \
    {                                                                                                                  \
        OPTION = VALUE;                                                                                                \
    }                                                                                                                  \
    else                                                                                                               \
        return false

namespace trtllm
{
namespace gen
{
class CudaRunner;
class GenCfg;
} // namespace gen
} // namespace trtllm

namespace gemm
{

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

// NOTE: when adding new parameters, please update the dumpOptions function and
// gemm_export_config.json for cubin export.
struct GemmOptions
{
#ifndef TLLM_GEN_EXPORT_INTERFACE
    // allow safely down-casting
    virtual ~GemmOptions() = default;
#endif

    GemmOptions() = default;

    GemmOptions(AllReduceAlgo allReduceAlgo, BiasType biasType, int blockK, bool clcFastDrain, int clusterDimX,
        int clusterDimY, int clusterDimZ, CtaSwizzleType ctaSwizzleType, tg::Dtype dtypeAcc, tg::Dtype dtypeA,
        tg::Dtype dtypeB, tg::Dtype dtypeC, tg::Dtype dtypeMmaA, tg::Dtype dtypeMmaB, EltwiseActType eltwiseActType,
        bool enablesEarlyExit, bool enablesDelayedEarlyExit, bool enablesGlobalPtxKnobs, int epilogueLdtmDps,
        int epilogueLdtmBits, int epilogueTileM, int epilogueTileN, bool fuseUtccpWithUtcmma,
        bool gridTriggerSecondaryA, bool gridTriggerSecondaryB, bool gridWaitForPrimaryEarlyExit,
        bool gridWaitForPrimaryA, bool gridWaitForPrimaryB, bool hoistLoadTaskInit, bool hoistMmaTaskTryWaits, int k,
        KernelTraits kernelTraits, MatrixLayout layoutA, MatrixLayout layoutB, int m, int mmaK, tg::MmaKind mmaKind,
        int mmaM, int mmaN, bool mockAllReduce, int n, int numEpilogueWarps, int numRegsCastAWarps,
        int numRegsCopySfLdsSttm, int numRegsCopySparsityInfo, int numRegsPerThreadEpilogueWarp,
        int numRegsPerThreadNonEpilogueWarp, int numSlicesForSplitK, int numSlicesForSliceK, int numStages,
        int numStagesMma, int numStagesMmaWithinWorkTile, int numStagesMmaAcrossWorkTile, int numStagesWorkId,
        bool outputDebugTensors, bool patchF2fp, int32_t sfBlockSizeA, int32_t sfBlockSizeB, int32_t sfBlockSizeC,
        tg::SfLayout sfLayoutA, tg::SfLayout sfLayoutB, tg::SfLayout sfLayoutC, int sfReshapeFactor, bool sliceK,
        tg::Sparsity sparsityA, SplitK splitK, int tileK, int tileM, int tileN, TileScheduler tileScheduler,
        bool transposeMmaOutput, bool useCustomMmaSchedule, bool useDeepSeekFp8,
        bool useHoistTryWaitForCustomMmaSchedule, bool useMaxTmemOverlap, bool usePerTokenSfA, bool usePerTokenSfB,
        bool useShuffledMatrix, bool useTmaStore, bool useTwoTmaLoadWarps, bool useTwoMmaWarps,
        bool useUnrollLoop2xForMma, int validM, int validN, int validK, int worldSize)
        : mAllReduceAlgo{allReduceAlgo}
        , mBiasType{biasType}
        , mBlockK(blockK)
        , mClcFastDrain{clcFastDrain}
        , mClusterDimX{clusterDimX}
        , mClusterDimY{clusterDimY}
        , mClusterDimZ{clusterDimZ}
        , mCtaSwizzleType{ctaSwizzleType}
        , mDtypeAcc{dtypeAcc}
        , mDtypeA{dtypeA}
        , mDtypeB{dtypeB}
        , mDtypeC{dtypeC}
        , mDtypeMmaA{dtypeMmaA}
        , mDtypeMmaB{dtypeMmaB}
        , mEltwiseActType{eltwiseActType}
        , mEnablesEarlyExit{enablesEarlyExit}
        , mEnablesDelayedEarlyExit{enablesDelayedEarlyExit}
        , mEnablesGlobalPtxKnobs{enablesGlobalPtxKnobs}
        , mEpilogueLdtmDps{epilogueLdtmDps}
        , mEpilogueLdtmBits{epilogueLdtmBits}
        , mEpilogueTileM{epilogueTileM}
        , mEpilogueTileN{epilogueTileN}
        , mFuseUtccpWithUtcmma{fuseUtccpWithUtcmma}
        , mGridTriggerSecondaryA{gridTriggerSecondaryA}
        , mGridTriggerSecondaryB{gridTriggerSecondaryB}
        , mGridWaitForPrimaryEarlyExit{gridWaitForPrimaryEarlyExit}
        , mGridWaitForPrimaryA{gridWaitForPrimaryA}
        , mGridWaitForPrimaryB{gridWaitForPrimaryB}
        , mHoistLoadTaskInit{hoistLoadTaskInit}
        , mHoistMmaTaskTryWaits{hoistMmaTaskTryWaits}
        , mK{k}
        , mKernelTraits{kernelTraits}
        , mLayoutA{layoutA}
        , mLayoutB{layoutB}
        , mM{m}
        , mMmaK{mmaK}
        , mMmaKind{mmaKind}
        , mMmaM{mmaM}
        , mMmaN{mmaN}
        , mMockAllReduce{mockAllReduce}
        , mN{n}
        , mNumEpilogueWarps{numEpilogueWarps}
        , mNumRegsCastAWarps(numRegsCastAWarps)
        , mNumRegsCopySfLdsSttm(numRegsCopySfLdsSttm)
        , mNumRegsCopySparsityInfo(numRegsCopySparsityInfo)
        , mNumRegsPerThreadEpilogueWarp(numRegsPerThreadEpilogueWarp)
        , mNumRegsPerThreadNonEpilogueWarp(numRegsPerThreadNonEpilogueWarp)
        , mNumSlicesForSplitK{numSlicesForSplitK}
        , mNumSlicesForSliceK{numSlicesForSliceK}
        , mNumStages{numStages}
        , mNumStagesMma{numStagesMma}
        , mNumStagesMmaWithinWorkTile{numStagesMmaWithinWorkTile}
        , mNumStagesMmaAcrossWorkTile{numStagesMmaAcrossWorkTile}
        , mNumStagesWorkId{numStagesWorkId}
        , mOutputDebugTensors{outputDebugTensors}
        , mPatchF2fp{patchF2fp}
        , mSfBlockSizeA{sfBlockSizeA}
        , mSfBlockSizeB{sfBlockSizeB}
        , mSfBlockSizeC{sfBlockSizeC}
        , mSfLayoutA{sfLayoutA}
        , mSfLayoutB{sfLayoutB}
        , mSfLayoutC{sfLayoutC}
        , mSfReshapeFactor{sfReshapeFactor}
        , mSliceK{sliceK}
        , mSparsityA{sparsityA}
        , mSplitK{splitK}
        , mTileK{tileK}
        , mTileM{tileM}
        , mTileN{tileN}
        , mTileScheduler{tileScheduler}
        , mTransposeMmaOutput{transposeMmaOutput}
        , mUseCustomMmaSchedule{useCustomMmaSchedule}
        , mUseDeepSeekFp8{useDeepSeekFp8}
        , mUseHoistTryWaitForCustomMmaSchedule{useHoistTryWaitForCustomMmaSchedule}
        , mUseMaxTmemOverlap{useMaxTmemOverlap}
        , mUsePerTokenSfA{usePerTokenSfA}
        , mUsePerTokenSfB{usePerTokenSfB}
        , mUseShuffledMatrix{useShuffledMatrix}
        , mUseTmaStore{useTmaStore}
        , mUseTwoTmaLoadWarps{useTwoTmaLoadWarps}
        , mUseTwoMmaWarps{useTwoMmaWarps}
        , mUseUnrollLoop2xForMma{useUnrollLoop2xForMma}
        , mValidM{validM}
        , mValidN{validN}
        , mValidK{validK}
        , mWorldSize{worldSize}
    {
    }

    // The all-reduce algorithm.
    AllReduceAlgo mAllReduceAlgo{AllReduceAlgo::None};
    // The type of bias.
    BiasType mBiasType{BiasType::None};
    // Block size in the K dimension
    int mBlockK{-1};
    // Whether to enable CLC fast drain for early exit in SM100 CLC-based scheduler.
    bool mClcFastDrain{true};
    // Cluster size in X dim.
    int mClusterDimX{1};
    // Cluster size in Y dim.
    int mClusterDimY{1};
    // Cluster size in Z dim.
    int mClusterDimZ{1};
    // The type of CTA swizzle.
    CtaSwizzleType mCtaSwizzleType{CtaSwizzleType::RasterizeAlongM};
    // Data type of the accumulators.
    tg::Dtype mDtypeAcc{tg::Dtype::Fp32};
    // Data type of the A matrix.
    tg::Dtype mDtypeA{tg::Dtype::Fp16};
    // Data type of the B matrix.
    tg::Dtype mDtypeB{tg::Dtype::Void};
    // Data type of the outputs.
    tg::Dtype mDtypeC{tg::Dtype::Void};
    // Data type of the A matrix for the MMA, if different from the input type.
    tg::Dtype mDtypeMmaA{tg::Dtype::Void};
    // Data type of the B matrix for the MMA, if different from the input type.
    tg::Dtype mDtypeMmaB{tg::Dtype::Void};
    // The type of activation.
    EltwiseActType mEltwiseActType{EltwiseActType::None};
    // Whether to enable early exit.
    bool mEnablesEarlyExit{false};
    // Whether to enable delayed early exit to overlap
    // numNonExitingCtas loading with the other instructions.
    bool mEnablesDelayedEarlyExit{false};
    // Whether to enable the global PTX knobs for guiding the compiler optimizations.
    bool mEnablesGlobalPtxKnobs{true};
    // The epilogue supports multiple LDTM shapes, although not every shape is applicable in every
    // case. In particular:
    // - On Hopper: must be 16dp256bit.
    // - Transposed output: must be 16dp256bit.
    // - Non-transposed output:
    //     - NvFp4 with fused activation: must be 32dp32bit.
    //     - Else it can be either 16dp256bit or 32dp32bit.
    // The number of DP lanes in the epilogue LDTM.
    int mEpilogueLdtmDps{16};
    // The number of bits in the epilogue LDTM.
    int mEpilogueLdtmBits{256};
    // Tile size for the epilogue in M dimension.
    int mEpilogueTileM{128};
    // Tile size for the epilogue in N dimension.
    int mEpilogueTileN{32};
    // Whether fuse UTCCP with UTC*MMA.
    bool mFuseUtccpWithUtcmma{false};
    // Whether load task A triggers the next grid.
    bool mGridTriggerSecondaryA{false};
    // Whether load task B triggers the next grid.
    bool mGridTriggerSecondaryB{false};
    // Whether the loads that check for an early exit should wait on a grid dependency.
    bool mGridWaitForPrimaryEarlyExit{true};
    // Whether the load of A should wait on a grid dependency.
    bool mGridWaitForPrimaryA{true};
    // Whether the load of B should wait on a grid dependency.
    bool mGridWaitForPrimaryB{true};
    // Whether to hoist the initialization of the loading tasks.
    bool mHoistLoadTaskInit{true};
    // Whether to hoist the mbarrier try_waits (e.g., mma.prodAcq, smemAb.consWait) in the MMA task.
    bool mHoistMmaTaskTryWaits{false};
    // The K dimension of GEMM.
    int mK{16 * 16};
    // Traits of the kernel.
    KernelTraits mKernelTraits{};
    // Layout of A matrix
    MatrixLayout mLayoutA{MatrixLayout::MajorK};
    // Layout of B matrix
    MatrixLayout mLayoutB{MatrixLayout::MajorK};
    // The M dimension of GEMM.
    int mM{128 * 2};
    // Size of the MMA instruction in the K dimension.
    int mMmaK{16};
    // The kind of MMA instruction to use.
    tg::MmaKind mMmaKind{tg::MmaKind::Auto};
    // Size of the MMA instruction in the M dimension.
    int mMmaM{64};
    // Size of the MMA instruction in the N dimension.
    int mMmaN{16};
    // Whether to mock all-reduce code for single-GPU debugging.
    bool mMockAllReduce{false};
    // The N dimension of GEMM.
    int mN{64 * 4};
    // Number of Epilogue Warps
    int mNumEpilogueWarps{4};
    // Number of registers for the cast A warps.
    int mNumRegsCastAWarps{0};
    // Number of registers for the LDS+STTM warps.
    int mNumRegsCopySfLdsSttm{0};
    // Number of registers per thread to copy sparsity info with LDS+STTM.
    int mNumRegsCopySparsityInfo{0};
    // Number of registers per thread for epilogue warps
    int mNumRegsPerThreadEpilogueWarp{0};
    // Number of registers per thread for non-epilogue warps
    int mNumRegsPerThreadNonEpilogueWarp{0};
    // Number of partitions along the K dimension. When mNumSlicesForSplitK > 1,
    // the problem is distributed across several SMs, where each CTA works on its local K slice.
    // Partial results are accumulated afterwards using either GMEM or DSMEM (in CGA)
    // to exchange the data between CTAs.
    int mNumSlicesForSplitK{1};
    // Number of slices for slice-K along K dimension.
    int mNumSlicesForSliceK{1};
    // The depth of the mainloop pipeline.
    int mNumStages{2};
    // The depth of the mma pipeline. Equals numStagesMmaWithinWorkTile * numStagesMmaAcrossWorkTile.
    int mNumStagesMma{1};
    // The depth of the mma pipeline within work tile. Only GmemC classes with "WithAccInReg" suffix
    // are allowed to be greater than 1.
    int mNumStagesMmaWithinWorkTile{-1};
    // The depth of the mma pipeline across work tiles in the persistent loop.
    int mNumStagesMmaAcrossWorkTile{-1};
    // The depth of the work id pipeline and the work throttle pipeline.
    int mNumStagesWorkId{3};
    // Whether to output debug tensors.
    bool mOutputDebugTensors{false};
    // Patch float conversions.
    bool mPatchF2fp{false};
    // Block size of A, for block-scaled types.
    int mSfBlockSizeA{-1};
    // Block size of B, for block-scaled types.
    int mSfBlockSizeB{-1};
    // Block size of C, for block-scaled types.
    int mSfBlockSizeC{-1};
    // Scale factors layout for A.
    tg::SfLayout mSfLayoutA{tg::SfLayout::R128c4};
    // Scale factors layout for B.
    tg::SfLayout mSfLayoutB{tg::SfLayout::R128c4};
    // Scale factors layout for C.
    tg::SfLayout mSfLayoutC{tg::SfLayout::R128c4};
    // Number of "repeats", i.e. reshaping factor, to fold hidden dimension into SfBlock dimension.
    // As result, the hidden dimension of the SF tensor must be a multiple of NumRepeats *
    // numEltsPerSf * 4. This reduces the problem shape space that the kernel is able to run.
    // But it reduces the number of L2 requests under the hood and potentially improves perf.
    // Applies to layout 8x4 only.
    int mSfReshapeFactor{1};
    // Slice-K implementation to use TileM dimension for TileK.
    bool mSliceK{false};
    // Sparsity of A.
    tg::Sparsity mSparsityA{tg::Sparsity::Dense};
    // The location of the exchange for split-K (it's None when split-K is disabled).
    SplitK mSplitK{SplitK::None};
    // K tile dimension of GEMM.
    int mTileK{16};
    // M tile dimension of GEMM.
    int mTileM{128};
    // N tile dimension of GEMM.
    int mTileN{32};
    // Tile scheduler type.
    TileScheduler mTileScheduler{TileScheduler::Static};
    // Save output of MMA in M-major format.
    bool mTransposeMmaOutput{false};
    // Use custom MMA schedule optimized for low-latency.
    bool mUseCustomMmaSchedule{false};
    // Use DeepSeek Fp8.
    bool mUseDeepSeekFp8{false};
    // The purpose of hoisting trywaits is to opportunistically peek at the availability of the next
    // k-block. It benefits when the next k-block is already available and thus sustaining the
    // momentum, but it adds latency to the first k-block for smaller k-loop.
    bool mUseHoistTryWaitForCustomMmaSchedule{false};
    // Whether use the max Tmem overlap trick.
    bool mUseMaxTmemOverlap{false};
    // Apply per-token scales from A
    bool mUsePerTokenSfA{false};
    // Apply per-token scales from B
    bool mUsePerTokenSfB{false};
    // Reorder rows/cols in the A matrix (when TransposeMmaOutput is true, otherwise B matrix) for the
    // better memory accesses in the M-major epilogue.
    bool mUseShuffledMatrix{false};
    // Use TMA to store the result.
    bool mUseTmaStore{true};
    // Use two different warps for A and B matrix load.
    bool mUseTwoTmaLoadWarps{false};
    // Use two different warps for MMA tasks. Applicable only to DeepSeek FP8.
    bool mUseTwoMmaWarps{false};
    // Whether to unroll the loop by 2x.
    bool mUseUnrollLoop2xForMma{true};
    // The valid range of M/N/K dimension of GEMM without padding values.
    // Used to opportunistically remove memory traffic from the padding due to rigid SF shape
    // constraint or TMA constraint. Such as:
    // 1. outputDim % (4 * sfBlockSize) == 0; as 4x SFs are packed into 4 bytes
    // 2. MxFp4 x Fp8 mmaType requires bespoke TMA load which requires hiddenDim % 128 == 0
    // 3. TMA requires 16B alignment for each row
    int mValidM{-1};
    int mValidN{-1};
    int mValidK{-1};
    // World size for all-reduce.
    int mWorldSize{1};
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SmVersion = tg::CudaArch;

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// GemmConfig
//
////////////////////////////////////////////////////////////////////////////////////////////////////

struct GemmConfig
{
    uint8_t const* mData{nullptr};
    uint32_t mSize{0};
    uint32_t mSharedMemSize{0};
    char const* mFunctionName{nullptr};
    uint32_t mNumThreadsPerCTA{0};
    char const* mHash{nullptr};
    std::string mGenCfgJsonStr{""};
    char const* mExecPath{nullptr};
    trtllm::gen::CudaRunner* mCudaRunner{nullptr};
    trtllm::gen::GenCfg* mGenCfg{nullptr};
    int32_t mInstanceIdx{0};

    GemmOptions mOptions{};
    tg::CudaArch mSm{tg::CudaArch::Sm100a};
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Serialization helpers.
template <typename T>
inline std::string toString(T e)
{
    return std::to_string(e);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline std::string toString(trtllm::gen::Dtype e)
{
    return trtllm::gen::dtypeToString(e);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline std::string toString(trtllm::gen::MmaKind e)
{
    return trtllm::gen::mmaKindToString(e);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline std::string toString(CtaSwizzleType e)
{
    switch (e)
    {
    case CtaSwizzleType::RasterizeAlongM: return "RasterizeAlongM";
    case CtaSwizzleType::RasterizeAlongN: return "RasterizeAlongN";
    case CtaSwizzleType::ZigZagAlongM2: return "ZigZagAlongM2";
    case CtaSwizzleType::ZigZagAlongN2: return "ZigZagAlongN2";
    case CtaSwizzleType::ZigZagAlongM4: return "ZigZagAlongM4";
    case CtaSwizzleType::ZigZagAlongN4: return "ZigZagAlongN4";
    default: return std::to_string(static_cast<int32_t>(e));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline std::string dumpOptions(GemmOptions const& options, bool dumpRuntimeParams = true)
{
    std::stringstream ss;
    ss << "mAllReduceAlgo="
       << "gemm::AllReduceAlgo(" << static_cast<int32_t>(options.mAllReduceAlgo) << ")"
       << "," << std::endl;
    ss << "mBiasType="
       << "gemm::BiasType(" << static_cast<int32_t>(options.mBiasType) << ")"
       << "," << std::endl;
    ss << "mBlockK=" << options.mBlockK << "," << std::endl;
    ss << "mClcFastDrain=" << options.mClcFastDrain << "," << std::endl;
    ss << "mClusterDimX=" << options.mClusterDimX << "," << std::endl;
    ss << "mClusterDimY=" << options.mClusterDimY << "," << std::endl;
    ss << "mClusterDimZ=" << options.mClusterDimZ << "," << std::endl;
    ss << "mCtaSwizzleType="
       << "gemm::CtaSwizzleType(" << static_cast<int32_t>(options.mCtaSwizzleType) << ")"
       << "," << std::endl;
    ss << "mDtypeAcc="
       << "trtllm::gen::Dtype(" << static_cast<int32_t>(options.mDtypeAcc) << ")"
       << "," << std::endl;
    ss << "mDtypeA="
       << "trtllm::gen::Dtype(" << static_cast<int32_t>(options.mDtypeA) << ")"
       << "," << std::endl;
    ss << "mDtypeB="
       << "trtllm::gen::Dtype(" << static_cast<int32_t>(options.mDtypeB) << ")"
       << "," << std::endl;
    ss << "mDtypeC="
       << "trtllm::gen::Dtype(" << static_cast<int32_t>(options.mDtypeC) << ")"
       << "," << std::endl;
    ss << "mDtypeMmaA="
       << "trtllm::gen::Dtype(" << static_cast<int32_t>(options.mDtypeMmaA) << ")"
       << "," << std::endl;
    ss << "mDtypeMmaB="
       << "trtllm::gen::Dtype(" << static_cast<int32_t>(options.mDtypeMmaB) << ")"
       << "," << std::endl;
    ss << "mEltwiseActType="
       << "gemm::EltwiseActType(" << static_cast<int32_t>(options.mEltwiseActType) << ")"
       << "," << std::endl;
    ss << "mEnablesEarlyExit=" << options.mEnablesEarlyExit << "," << std::endl;
    ss << "mEnablesDelayedEarlyExit=" << options.mEnablesDelayedEarlyExit << "," << std::endl;
    ss << "mEnablesGlobalPtxKnobs=" << options.mEnablesGlobalPtxKnobs << "," << std::endl;
    ss << "mEpilogueLdtmDps=" << options.mEpilogueLdtmDps << "," << std::endl;
    ss << "mEpilogueLdtmBits=" << options.mEpilogueLdtmBits << "," << std::endl;
    ss << "mEpilogueTileM=" << options.mEpilogueTileM << "," << std::endl;
    ss << "mEpilogueTileN=" << options.mEpilogueTileN << "," << std::endl;
    ss << "mFuseUtccpWithUtcmma=" << options.mFuseUtccpWithUtcmma << "," << std::endl;
    ss << "mGridTriggerSecondaryA=" << options.mGridTriggerSecondaryA << "," << std::endl;
    ss << "mGridTriggerSecondaryB=" << options.mGridTriggerSecondaryB << "," << std::endl;
    ss << "mGridWaitForPrimaryEarlyExit=" << options.mGridWaitForPrimaryEarlyExit << "," << std::endl;
    ss << "mGridWaitForPrimaryA=" << options.mGridWaitForPrimaryA << "," << std::endl;
    ss << "mGridWaitForPrimaryB=" << options.mGridWaitForPrimaryB << "," << std::endl;
    ss << "mHoistLoadTaskInit=" << options.mHoistLoadTaskInit << "," << std::endl;
    ss << "mHoistMmaTaskTryWaits=" << options.mHoistMmaTaskTryWaits << "," << std::endl;
    if (dumpRuntimeParams)
    {
        ss << "mK=" << options.mK << "," << std::endl;
    }
    ss << "mKernelTraits={}"
       << "," << std::endl;
    ss << "mLayoutA=gemm::MatrixLayout(" << static_cast<int32_t>(options.mLayoutA) << ")"
       << "," << std::endl;
    ss << "mLayoutB=gemm::MatrixLayout(" << static_cast<int32_t>(options.mLayoutB) << ")"
       << "," << std::endl;
    if (dumpRuntimeParams)
    {
        ss << "mM=" << options.mM << "," << std::endl;
    }
    ss << "mMmaK=" << options.mMmaK << "," << std::endl;
    ss << "mMmaKind="
       << "trtllm::gen::MmaKind(" << static_cast<int32_t>(options.mMmaKind) << ")"
       << "," << std::endl;
    ss << "mMmaM=" << options.mMmaM << "," << std::endl;
    ss << "mMmaN=" << options.mMmaN << "," << std::endl;
    ss << "mMockAllReduce=" << options.mMockAllReduce << "," << std::endl;
    if (dumpRuntimeParams)
    {
        ss << "mN=" << options.mN << "," << std::endl;
    }
    ss << "mNumEpilogueWarps=" << options.mNumEpilogueWarps << "," << std::endl;
    ss << "mNumRegsCastAWarps=" << options.mNumRegsCastAWarps << "," << std::endl;
    ss << "mNumRegsCopySfLdsSttm=" << options.mNumRegsCopySfLdsSttm << "," << std::endl;
    ss << "mNumRegsCopySparsityInfo=" << options.mNumRegsCopySparsityInfo << "," << std::endl;
    ss << "mNumRegsPerThreadEpilogueWarp=" << options.mNumRegsPerThreadEpilogueWarp << "," << std::endl;
    ss << "mNumRegsPerThreadNonEpilogueWarp=" << options.mNumRegsPerThreadNonEpilogueWarp << "," << std::endl;
    ss << "mNumSlicesForSplitK=" << options.mNumSlicesForSplitK << "," << std::endl;
    ss << "mNumSlicesForSliceK=" << options.mNumSlicesForSliceK << "," << std::endl;
    ss << "mNumStages=" << options.mNumStages << "," << std::endl;
    ss << "mNumStagesMma=" << options.mNumStagesMma << "," << std::endl;
    ss << "mNumStagesMmaWithinWorkTile=" << options.mNumStagesMmaWithinWorkTile << "," << std::endl;
    ss << "mNumStagesMmaAcrossWorkTile=" << options.mNumStagesMmaAcrossWorkTile << "," << std::endl;
    ss << "mNumStagesWorkId=" << options.mNumStagesWorkId << "," << std::endl;
    ss << "mOutputDebugTensors=" << options.mOutputDebugTensors << "," << std::endl;
    ss << "mPatchF2fp=" << options.mPatchF2fp << "," << std::endl;
    ss << "mSfBlockSizeA=" << options.mSfBlockSizeA << "," << std::endl;
    ss << "mSfBlockSizeB=" << options.mSfBlockSizeB << "," << std::endl;
    ss << "mSfBlockSizeC=" << options.mSfBlockSizeC << "," << std::endl;
    ss << "mSfLayoutA="
       << "trtllm::gen::SfLayout(" << static_cast<int32_t>(options.mSfLayoutA) << ")"
       << "," << std::endl;
    ss << "mSfLayoutB="
       << "trtllm::gen::SfLayout(" << static_cast<int32_t>(options.mSfLayoutB) << ")"
       << "," << std::endl;
    ss << "mSfLayoutC="
       << "trtllm::gen::SfLayout(" << static_cast<int32_t>(options.mSfLayoutC) << ")"
       << "," << std::endl;
    ss << "mSfReshapeFactor=" << options.mSfReshapeFactor << "," << std::endl;
    ss << "mSliceK=" << options.mSliceK << "," << std::endl;
    ss << "mSparsityA="
       << "trtllm::gen::Sparsity(" << static_cast<int32_t>(options.mSparsityA) << ")"
       << "," << std::endl;
    ss << "mSplitK="
       << "gemm::SplitK(" << static_cast<int32_t>(options.mSplitK) << ")"
       << "," << std::endl;
    ss << "mTileK=" << options.mTileK << "," << std::endl;
    ss << "mTileM=" << options.mTileM << "," << std::endl;
    ss << "mTileN=" << options.mTileN << "," << std::endl;
    ss << "mTileScheduler="
       << "gemm::TileScheduler(" << static_cast<int32_t>(options.mTileScheduler) << ")"
       << "," << std::endl;
    ss << "mTransposeMmaOutput=" << options.mTransposeMmaOutput << "," << std::endl;
    ss << "mUseCustomMmaSchedule=" << options.mUseCustomMmaSchedule << "," << std::endl;
    ss << "mUseDeepSeekFp8=" << options.mUseDeepSeekFp8 << "," << std::endl;
    ss << "mUseHoistTryWaitForCustomMmaSchedule=" << options.mUseHoistTryWaitForCustomMmaSchedule << "," << std::endl;
    ss << "mUseMaxTmemOverlap=" << options.mUseMaxTmemOverlap << "," << std::endl;
    ss << "mUsePerTokenSfA=" << options.mUsePerTokenSfA << "," << std::endl;
    ss << "mUsePerTokenSfB=" << options.mUsePerTokenSfB << "," << std::endl;
    ss << "mUseShuffledMatrix=" << options.mUseShuffledMatrix << "," << std::endl;
    ss << "mUseTmaStore=" << options.mUseTmaStore << "," << std::endl;
    ss << "mUseTwoTmaLoadWarps=" << options.mUseTwoTmaLoadWarps << "," << std::endl;
    ss << "mUseTwoMmaWarps=" << options.mUseTwoMmaWarps << "," << std::endl;
    ss << "mUseUnrollLoop2xForMma=" << options.mUseUnrollLoop2xForMma << "," << std::endl;
    if (dumpRuntimeParams)
    {
        ss << "mValidM=" << options.mValidM << "," << std::endl;
        ss << "mValidN=" << options.mValidN << "," << std::endl;
        ss << "mValidK=" << options.mValidK << "," << std::endl;
        ss << "mWorldSize=" << options.mWorldSize << std::endl;
    }
    return ss.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline T divUp(T a, T b)
{
    return (a + b - 1) / b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline T divUpMul(T a, T b)
{
    return gemm::divUp(a, b) * b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// clang-format off
inline std::vector<int> srcToDstBlk16RowMap =
      {
        0,  8,
        1,  9,
        2, 10,
        3, 11,
        4, 12,
        5, 13,
        6, 14,
        7, 15
      };
inline std::vector<int> srcToDstBlk32RowMap =
      {
        0,  8, 16, 24,
        1,  9, 17, 25,
        2, 10, 18, 26,
        3, 11, 19, 27,
        4, 12, 20, 28,
        5, 13, 21, 29,
        6, 14, 22, 30,
        7, 15, 23, 31
      };

// clang-format on

////////////////////////////////////////////////////////////////////////////////////////////////////

inline int32_t getShuffleBlockSize(int epilogueTileM)
{
    int shuffleBlockSize = 16;
    if (epilogueTileM % 128 == 0)
    {
        shuffleBlockSize = 32;
    }
    return shuffleBlockSize;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline std::vector<int> const& getShuffleIndices(int epilogueTileM)
{
    auto const shuffleBlockSize = getShuffleBlockSize(epilogueTileM);
    return shuffleBlockSize == 16 ? srcToDstBlk16RowMap : srcToDstBlk32RowMap;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Check if the options are valid or not.
inline bool checkAndUpdateGemmOptions(
    GemmOptions& options, tg::CudaArch cudaArch, int tpGrpSize, bool updateOptions = true)
{
    options.mWorldSize = tpGrpSize;

    bool isBlackwell = tg::isArchBlackwell(cudaArch);

    // If dtypeB is unspecified (Dtype::Void), assign to dtypeA.
    if (options.mDtypeB == tg::Dtype::Void)
    {
        if (updateOptions)
        {
            options.mDtypeB = options.mDtypeA;
        }
        else
        {
            return false;
        }
    }
    // If dtypeC is unspecified (Dtype::Void), assign to dtypeA.
    if (options.mDtypeC == tg::Dtype::Void)
    {
        TLLM_LOG_INFO("Setting dtypeC to ", tg::dtypeToString(options.mDtypeA));
        if (updateOptions)
        {
            options.mDtypeC = options.mDtypeA;
        }
        else
        {
            return false;
        }
    }

    // If not specified, used the input dtypes as MMA dtypes (no cast required).
    if (options.mDtypeMmaA == tg::Dtype::Void)
    {
        if (updateOptions)
        {
            options.mDtypeMmaA = options.mDtypeA;
        }
        else
        {
            return false;
        }
    }
    if (options.mDtypeMmaB == tg::Dtype::Void)
    {
        if (updateOptions)
        {
            options.mDtypeMmaB = options.mDtypeB;
        }
        else
        {
            return false;
        }
    }

    // If validM/N/K is not specified, then assume the full range of the dimension is valid.
    if (options.mValidM < 0 || options.mValidN < 0 || options.mValidK < 0)
    {
        if (updateOptions)
        {
            options.mValidM = options.mValidM < 0 ? options.mM : options.mValidM;
            options.mValidN = options.mValidN < 0 ? options.mN : options.mValidN;
            options.mValidK = options.mValidK < 0 ? options.mK : options.mValidK;
        }
        else
        {
            return false;
        }
    }

    // It must not exceed the padded dimensions.
    if (options.mValidM > options.mM || options.mValidN > options.mN || options.mValidK > options.mK)
    {
        TLLM_LOG_WARNING("ValidM, ValidN, and ValidK must be less than or equal to M, N, and K respectively.");
        if (updateOptions)
        {
            options.mValidM = std::min(options.mValidM, options.mM);
            options.mValidN = std::min(options.mValidN, options.mN);
            options.mValidK = std::min(options.mValidK, options.mK);
        }
        else
        {
            return false;
        }
    }

    // BlockMajorK layout does not support validM, validN, validK parameters
    if (options.mLayoutA == gemm::MatrixLayout::BlockMajorK || options.mLayoutB == gemm::MatrixLayout::BlockMajorK)
    {
        bool hasValidParams = (options.mValidM != -1 && options.mValidM != options.mM)
            || (options.mValidN != -1 && options.mValidN != options.mN)
            || (options.mValidK != -1 && options.mValidK != options.mK);
        TLLM_CHECK_ERROR(!hasValidParams,
            "BlockMajorK layout does not support validM/validN/validK parameters due to swizzled layout. "
            "Found validM=",
            options.mValidM, " validN=", options.mValidN, " validK=", options.mValidK);
    }

#ifdef TLLM_PUBLIC_RELEASE
    if (options.mDtypeA == tg::Dtype::E2m1 && options.mDtypeMmaA == tg::Dtype::E4m3)
    {
        TLLM_CHECK_ERROR(false, "E2m1 x E4m3 is not supported for JIT compile. Use cubins instead.");
    }
#endif // TLLM_PUBLIC_RELEASE

    // Check that the A cast is supported.
    // Currently, we only support {MxFp4, NvFp4, MxInt4} -> Bf16.
    TLLM_CHECK_ERROR((options.mDtypeA == options.mDtypeMmaA)
            || ((options.mDtypeA == tg::Dtype::MxE2m1 || options.mDtypeA == tg::Dtype::E2m1
                    || options.mDtypeA == tg::Dtype::MxInt4)
                && options.mDtypeMmaA == tg::Dtype::Bfloat16)
            || (options.mDtypeA == tg::Dtype::E2m1 && options.mDtypeMmaA == tg::Dtype::E4m3),
        "Unsupported cast for A: ", tg::dtypeToString(options.mDtypeA), " -> ", tg::dtypeToString(options.mDtypeMmaA));

    // Check that the B cast is supported.
    // Currently, we only support Fp8 -> MxFp8.
    // TODO: add same support for A (no transpose)
    TLLM_CHECK_ERROR((options.mDtypeB == options.mDtypeMmaB)
            || (options.mDtypeB == tg::Dtype::E4m3 && options.mDtypeMmaB == tg::Dtype::MxE4m3),
        "Unsupported cast for B: ", tg::dtypeToString(options.mDtypeB), " -> ", tg::dtypeToString(options.mDtypeMmaB));

    if (options.mDtypeA != options.mDtypeMmaA)
    {
        TLLM_CHECK_ERROR(options.mTileM == 128, "TileM must be 128 when casting the input matrix A before the MMA.");
    }

    if (options.mPatchF2fp)
    {
        TLLM_CHECK_ERROR(options.mDtypeA == tg::Dtype::MxE2m1 && options.mDtypeMmaA == tg::Dtype::Bfloat16,
            "PatchF2fp is only supported for MxFp4 to Bf16 casts.");
    }
#ifdef TLLM_PUBLIC_RELEASE
    options.mPatchF2fp = false;
#endif // TLLM_PUBLIC_RELEASE

    // FIXME: We do not support different dtypes for A and B when not on Blackwell.
    if (!isBlackwell)
    {
        TLLM_CHECK_ERROR(
            options.mDtypeMmaA == options.mDtypeMmaB, "For non-Blackwell, A and B must have the same dtype.");
    }

    // Check that the different dtypes for A and B are supported by the tensor core
    // kind::f8f6f4
    if (options.mDtypeMmaA == tg::Dtype::E4m3 || options.mDtypeMmaA == tg::Dtype::E2m1)
    {
        TLLM_CHECK_ERROR(options.mDtypeMmaB == tg::Dtype::E4m3 || options.mDtypeMmaB == tg::Dtype::E2m1,
            "For dtypeMmaA = E4m3/E2m1 A, dtypeMmaB must also be E4m3/E2m1.");
    }

    // kind::mxf8f6f4
    if (options.mDtypeMmaA == tg::Dtype::MxE4m3 || options.mDtypeMmaA == tg::Dtype::MxE2m1)
    {
        TLLM_CHECK_ERROR(options.mDtypeMmaB == tg::Dtype::MxE4m3 || options.mDtypeMmaB == tg::Dtype::MxE2m1,
            "For dtypeMmaA = MxE4m3 or MxE2m1, dtypeMmaB must also be MxE4m3 or MxE2m1.");
    }
    if (options.mDtypeMmaB == tg::Dtype::MxE4m3 || options.mDtypeMmaB == tg::Dtype::MxE2m1)
    {
        TLLM_CHECK_ERROR(options.mDtypeMmaA == tg::Dtype::MxE4m3 || options.mDtypeMmaA == tg::Dtype::MxE2m1,
            "For dtypeMmaB = MxE4m3 or MxE2m1, dtypeMmaA must also be MxE4m3 or MxE2m1.");
    }

    // kind::f16
    if (options.mDtypeMmaA == tg::Dtype::Fp16 || options.mDtypeMmaA == tg::Dtype::Bfloat16)
    {
        TLLM_CHECK_ERROR(options.mDtypeMmaB == options.mDtypeMmaA,
            "For dtypeMmaA = Fp16/Bfloat16, dtypeMmaB must be the same as dtypeMmaA.");
    }

    // When one of the inputs needs to be cast, we must use two load warps.
    if ((options.mDtypeMmaA != options.mDtypeA || options.mDtypeMmaB != options.mDtypeB)
        && !options.mUseTwoTmaLoadWarps)
    {
        TLLM_LOG_WARNING("Two TMA load warps must be enabled if any of the inputs needs to be cast.");
    }

    // When different dtypes are used for A and B, we must use different tiles to do the loading.
    // It is not strictly required, but current implementation of SmemAb requires that.
    if (options.mDtypeA != options.mDtypeB)
    {
        TLLM_CHECK_ERROR(
            options.mUseTwoTmaLoadWarps, "Two TMA load warps must be enabled for different input types of A and B.");
    }

    // Get the mma kind for the input types.
    if (options.mMmaKind == tg::MmaKind::Auto)
    {
        if (updateOptions)
        {
            options.mMmaKind = dtypeGetMmaKind(options.mDtypeMmaA, options.mDtypeMmaB);
        }
        else
        {
            return false;
        }
    }

    // Check that the sparsity mode of A is supported, and compatible with the MMA kind.
    // Note: trtllm-gen currently does not support sparsity with tf32, fp16, bf16.
    switch (options.mSparsityA)
    {
    case tg::Sparsity::Dense:
        // Always supported.
        break;
    case tg::Sparsity::Any_1_2: TLLM_LOG_ERROR("1:2 sparsity is not supported."); break;
    case tg::Sparsity::Any_2_4:
    {
        bool isSupported_2_4
            = (options.mMmaKind == tg::MmaKind::Fp8Fp6Fp4 || options.mMmaKind == tg::MmaKind::MxFp8Fp6Fp4);
        TLLM_CHECK_ERROR(isSupported_2_4, "2:4 sparsity is not supported for MMA kind ",
            tg::mmaKindToString(options.mMmaKind), " on target ", tg::cudaArchToString(cudaArch));
        break;
    }
    case tg::Sparsity::Pairwise_4_8:
        TLLM_CHECK_ERROR(options.mMmaKind == tg::MmaKind::MxFp4NvFp4,
            "Pairwise 4:8 sparsity is only supported for MMA kind MxFp4NvFp4.");
        break;
    default: TLLM_CHECK_ERROR(false, "Unsupported sparsityA: ", tg::sparsityToString(options.mSparsityA)); break;
    }

    // Is A sparse?
    bool const isSparseA = tg::isSparse(options.mSparsityA);

    // Requirements specific to sparsity, and compatibility with other features.
    if (isSparseA)
    {
        TLLM_CHECK_ERROR(isBlackwell, "Sparsity is only supported on Blackwell");
        // The following requirement is for TMA load: the box width must be a multiple of 16B.
        TLLM_CHECK_ERROR(tg::getNumBytesSparsityInfo(options.mSparsityA, options.mTileK) % 16 == 0,
            "The sparsity information for one tile row must be a multiple of 16B. Use larger tileK.");
        TLLM_CHECK_ERROR(options.mDtypeA == options.mDtypeMmaA, "Sparsity is not supported with on-the-fly upcasting.");
        TLLM_CHECK_ERROR(!options.mUseDeepSeekFp8, "Sparsity is not supported with DeepSeek Fp8.");
        TLLM_CHECK_ERROR(!options.mSliceK, "Sparsity is not supported with slice-k.");
    }

    if (options.mMmaKind == tg::MmaKind::Fp8Fp6Fp4)
    {
        int mmaK = isSparseA ? 64 : 32;

        if (options.mMmaK != mmaK)
        {
            TLLM_LOG_WARNING("Unsupported MmaK (", options.mMmaK, ") for MmaKind=", gemm::toString(options.mMmaKind),
                " and sparsity=", tg::sparsityToString(options.mSparsityA), ". Setting MmaK to ", mmaK);
            if (updateOptions)
            {
                options.mMmaK = mmaK;
                options.mTileK = std::max(options.mMmaK, options.mTileK);
            }
            else
            {
                return false;
            }
        }
    }

    // Check LDTM shape.
    if (isBlackwell)
    {
        TLLM_CHECK_ERROR((options.mEpilogueLdtmDps == 16 && options.mEpilogueLdtmBits == 256)
                || (options.mEpilogueLdtmDps == 32 && options.mEpilogueLdtmBits == 32),
            "Unsupported LDTM shape: ", options.mEpilogueLdtmDps, "dp", options.mEpilogueLdtmBits, "bit.");
        if (options.mEpilogueTileM == 64)
        {
            TLLM_CHECK_ERROR(options.mEpilogueLdtmDps == 16,
                "Unsupported LDTM shape for epilogueTileM=64: ", options.mEpilogueLdtmDps, "dp",
                options.mEpilogueLdtmBits, "bit.");
        }
        if (options.mTransposeMmaOutput)
        {
            // We can't use 32dp32bit LDTM for transposed outputs because we need each thread to own
            // multiple consecutive output elements.
            TLLM_CHECK_ERROR((options.mEpilogueLdtmDps == 16 && options.mEpilogueLdtmBits == 256),
                "Only 16dp256bit LDTM is supported for transposed outputs.");
        }
    }
    else
    {
        TLLM_CHECK_ERROR(options.mEpilogueLdtmDps == 16 && options.mEpilogueLdtmBits == 256,
            "Hopper does not use TMEM. The register layout corresponds to 16dp256bit. Got ", options.mEpilogueLdtmDps,
            "dp", options.mEpilogueLdtmBits, "bit.");
    }

    // Constraints for NvFp4, MxFp8, and MxFp4.
    if ((options.mMmaKind == tg::MmaKind::MxFp4NvFp4 || options.mMmaKind == tg::MmaKind::MxFp8Fp6Fp4
            || options.mDtypeC == tg::Dtype::MxE4m3)
        && options.mMmaM != 128)
    {

        if (options.mClusterDimX == 1)
        {
            // MMA M must be 128 when the input uses block scaling, or when the output is an Mx format.
            int newTileM = 128 * divUp(options.mTileM, 128);
            TLLM_LOG_WARNING("Unsupported MmaM (", options.mMmaM, ") for MmaKind=", gemm::toString(options.mMmaKind),
                ". Setting MmaM to 128 and TileM to ", newTileM);
            if (updateOptions)
            {
                options.mMmaM = 128;
                options.mTileM = newTileM;
            }
            else
            {
                return false;
            }
        }
        else
        {
            TLLM_CHECK_ERROR(options.mMmaM == 256 && options.mTileM == 128,
                "2CTA UTCxMMA only supports mmaM = 256 and tileM = 128.");
        }
    }
    if (options.mClusterDimX > 1)
    {
        TLLM_CHECK_ERROR(options.mLayoutB != MatrixLayout::BlockMajorK,
            "layoutB == MatrixLayout::BlockMajorK is not supported for now");
    }
    if (options.mMmaKind == tg::MmaKind::MxFp4NvFp4 || options.mMmaKind == tg::MmaKind::MxFp8Fp6Fp4)
    {
        TLLM_CHECK_ERROR(isBlackwell, "Block scaling is only supported on Blackwell");

        int mmaK = isSparseA ? 64 : 32;
        if (options.mMmaKind == tg::MmaKind::MxFp4NvFp4)
        {
            mmaK = isSparseA ? 128 : 64;
            if (options.mMmaK == 96 && !isSparseA)
            {
                mmaK = 96;
                TLLM_CHECK_ERROR(options.mTileK == 768, "When mmaK == 96, only tileK == 768 is supported");
                TLLM_CHECK_ERROR(options.mTileN <= 128, "When mmaK == 96, only tileN <= 128 is supported");
            }
        }
        if (options.mMmaK != mmaK)
        {
            int newTileK = mmaK * divUp(options.mTileK, mmaK);
            TLLM_LOG_WARNING("Unsupported MmaK (", options.mMmaK, ") for MmaKind=", gemm::toString(options.mMmaKind),
                ". Setting MmaK to ", mmaK, " and TileK to ", newTileK);
            if (updateOptions)
            {
                options.mMmaK = mmaK;
                options.mTileK = newTileK;
            }
            else
            {
                return false;
            }
        }

        // The MMA N may only be smaller than 64 if it is equal to the tile N.
        TLLM_CHECK_ERROR(options.mMmaN >= 64 || options.mMmaN == options.mTileN, "MmaN (", options.mMmaN,
            ") must be >= 64 or equal to TileN (", options.mTileN, ")");
    }

    // Note: the logic for selecting/checking the correct block size based on dtypes and sparsity is
    // centralized here, to avoid error-prone code duplication and make it a more explicit "contract"
    // with the user who is providing inputs in this format.
    // Additionally, in some cases, multiple values are possible:
    // - When we use type casting before the MMA (e.g. e2m1 x e4m3).
    // - For output C, based on whether the consumer will use sparsity.

    // SF block size for A.
    if (options.mDtypeA == tg::Dtype::E2m1 && options.mDtypeB == tg::Dtype::E4m3)
    {
        // Note that the type conversion needs to happen before TCs.
        // For example, convert e2m1 to e4m3 inside TmemCastA.
        if (!(options.mSfBlockSizeA == 16 || options.mSfBlockSizeA == 32))
        {
            TLLM_LOG_WARNING("sfBlockSizeA must be 16 or 32 for e2m1 x e4m3, got ", options.mSfBlockSizeA);
            GEMM_UPDATE_OR_ERROR(options.mSfBlockSizeA, 16);
        }
    }
    else if (options.mDtypeA == tg::Dtype::E2m1)
    {
        if (!((options.mSfBlockSizeA == 16 && !isSparseA) || (options.mSfBlockSizeA == 32 && isSparseA)))
        {
            TLLM_LOG_WARNING(
                "sfBlockSizeA must be 16 (dense) or 32 (sparse) for dtypeA=e2m1, got ", options.mSfBlockSizeA);
            GEMM_UPDATE_OR_ERROR(options.mSfBlockSizeA, isSparseA ? 32 : 16);
        }
    }
    else if (options.mDtypeA == tg::Dtype::MxE2m1 || options.mDtypeA == tg::Dtype::MxE4m3
        || options.mDtypeA == tg::Dtype::MxInt4)
    {
        if (!((options.mSfBlockSizeA == 32 && !isSparseA) || (options.mSfBlockSizeA == 64 && isSparseA)))
        {
            TLLM_LOG_WARNING("sfBlockSizeA must be 32 (dense) or 64 (sparse) for dtypeA=mx{e2m1,e4m3,int4}, got ",
                options.mSfBlockSizeA);
            GEMM_UPDATE_OR_ERROR(options.mSfBlockSizeA, isSparseA ? 64 : 32);
        }
    }
    else if (options.mSfBlockSizeA > 0)
    {
        TLLM_LOG_WARNING("Got sfBlockSizeA=", options.mSfBlockSizeA, " but dtypeA=", tg::dtypeToString(options.mDtypeA),
            " does not use block scales");
        GEMM_UPDATE_OR_ERROR(options.mSfBlockSizeA, -1);
    }
    // SF block size for B.
    if (options.mDtypeB == tg::Dtype::E2m1)
    {
        if (!((options.mSfBlockSizeB == 16 && !isSparseA) || (options.mSfBlockSizeB == 32 && isSparseA)))
        {
            TLLM_LOG_WARNING(
                "sfBlockSizeB must be 16 (dense) or 32 (sparse) for dtypeB=e2m1, got ", options.mSfBlockSizeB);
            GEMM_UPDATE_OR_ERROR(options.mSfBlockSizeB, isSparseA ? 32 : 16);
        }
    }
    else if (options.mDtypeB == tg::Dtype::MxE2m1 || options.mDtypeB == tg::Dtype::MxE4m3
        || (options.mDtypeB == tg::Dtype::E4m3 && options.mDtypeMmaB == tg::Dtype::MxE4m3))
    {
        if (!((options.mSfBlockSizeB == 32 && !isSparseA) || (options.mSfBlockSizeB == 64 && isSparseA)))
        {
            TLLM_LOG_WARNING(
                "sfBlockSizeB must be 32 (dense) or 64 (sparse) for dtypeB=mx{e2m1,e4m3}, got ", options.mSfBlockSizeB);
            GEMM_UPDATE_OR_ERROR(options.mSfBlockSizeB, isSparseA ? 64 : 32);
        }
    }
    else if (options.mSfBlockSizeB > 0)
    {
        TLLM_LOG_WARNING("Got sfBlockSizeB=", options.mSfBlockSizeB, " but dtypeB=", tg::dtypeToString(options.mDtypeB),
            " does not use block scales");
        GEMM_UPDATE_OR_ERROR(options.mSfBlockSizeB, -1);
    }
    // SF block size for C.
    if (options.mDtypeC == tg::Dtype::E2m1)
    {
        if (!(options.mSfBlockSizeC == 16 || options.mSfBlockSizeC == 32))
        {
            TLLM_LOG_WARNING("sfBlockSizeC must be 16 or 32 for dtypeC=e2m1, got ", options.mSfBlockSizeC);
            GEMM_UPDATE_OR_ERROR(options.mSfBlockSizeC, 16);
        }
    }
    else if (options.mDtypeC == tg::Dtype::MxE2m1 || options.mDtypeC == tg::Dtype::MxE4m3)
    {
        if (!(options.mSfBlockSizeC == 32 || options.mSfBlockSizeC == 64))
        {
            TLLM_LOG_WARNING("sfBlockSizeC must be 32 or 64 for dtypeC=mx{e2m1,e4m3}, got ", options.mSfBlockSizeC);
            GEMM_UPDATE_OR_ERROR(options.mSfBlockSizeC, 32);
        }
    }
    else if (options.mSfBlockSizeC > 0)
    {
        TLLM_LOG_WARNING("Got sfBlockSizeC=", options.mSfBlockSizeC, " but dtypeC=", tg::dtypeToString(options.mDtypeC),
            " does not use block scales");
        GEMM_UPDATE_OR_ERROR(options.mSfBlockSizeC, -1);
    }

    if (tg::dtypeIsBlockFmt(options.mDtypeA))
    {
        int numEltsPerSfA = options.mSfBlockSizeA;
        TLLM_CHECK_ERROR(options.mTileK % (4 * numEltsPerSfA) == 0, "TileK (", options.mTileK,
            ") must be a multiple of ", (4 * numEltsPerSfA), " for typeA ", gemm::toString(options.mDtypeA));
        auto const numEltsPerSfAInK = options.mK / numEltsPerSfA;
        TLLM_CHECK_ERROR(numEltsPerSfAInK % 4 == 0, "K dimension of scaling factors for A (", numEltsPerSfAInK,
            ") must be a multiple of 4");
    }
    if (tg::dtypeIsBlockFmt(options.mDtypeB))
    {
        TLLM_CHECK_ERROR(options.mSfLayoutB == tg::SfLayout::R128c4 || options.mSfLayoutB == tg::SfLayout::R8c4
                || options.mSfLayoutB == tg::SfLayout::Linear,
            "Only the 128x4 and 8x4 SF layouts are supported for B, got ", tg::sfLayoutToString(options.mSfLayoutB));

        // TileN must be a multiple of the number of rows per SF tile.
        int const numSfTileRowsB = options.mSfLayoutB == tg::SfLayout::R128c4 ? 128 : 8;
        TLLM_CHECK_ERROR(options.mTileN % numSfTileRowsB == 0, "TileN (", options.mTileN, ") must be a multiple of ",
            numSfTileRowsB, " for B SF layout ", tg::sfLayoutToString(options.mSfLayoutB));

        int numEltsPerSfB = options.mSfBlockSizeB;
        TLLM_CHECK_ERROR(options.mTileK % (4 * numEltsPerSfB) == 0, "TileK (", options.mTileK,
            ") must be a multiple of ", (4 * numEltsPerSfB), " for typeB ", gemm::toString(options.mDtypeB));
        auto const numEltsPerSfBInK = options.mK / numEltsPerSfB;
        TLLM_CHECK_ERROR(numEltsPerSfBInK % 4 == 0, "K dimension of scaling factors for B (", numEltsPerSfBInK,
            ") must be a multiple of 4");
    }

    int32_t padMultiplierA = 1;
    int32_t padMultiplierB = 1;
    if (options.mMmaKind == tg::MmaKind::MxFp8Fp6Fp4)
    {
        if (options.mDtypeA == tg::Dtype::MxE2m1)
        {
            padMultiplierA = 2;
        }
        if (options.mDtypeB == tg::Dtype::MxE2m1)
        {
            padMultiplierB = 2;
        }
    }
    TLLM_CHECK_ERROR((padMultiplierA * tg::dtypeGetNumBits(options.mDtypeA) * options.mK / 8) % 16 == 0,
        "K dimension of A must be aligned to 16 bytes.");
    TLLM_CHECK_ERROR((padMultiplierB * tg::dtypeGetNumBits(options.mDtypeB) * options.mK / 8) % 16 == 0,
        "K dimension of B must be aligned to 16 bytes.");

    if (tg::dtypeIsBlockFmt(options.mDtypeC))
    {
        TLLM_CHECK_ERROR(isBlackwell, "Block scaling is only supported on Blackwell");

        TLLM_CHECK_ERROR(options.mSfLayoutC == tg::SfLayout::R128c4 || options.mSfLayoutC == tg::SfLayout::R8c4,
            "Only the 128x4 and 8x4 SF layouts are supported for C.");
        if (!options.mTransposeMmaOutput)
        {
            TLLM_CHECK_ERROR(options.mEpilogueTileN % options.mSfBlockSizeC == 0,
                "EpilogueTileN must be a multiple of the number of elements per SF for C");
        }
        int const numSfTileRowsC = options.mSfLayoutC == tg::SfLayout::R128c4 ? 128 : 8;
        int const tileTokenDim = options.mTransposeMmaOutput ? options.mTileN : options.mTileM;
        TLLM_CHECK_ERROR_FMT(tileTokenDim % numSfTileRowsC == 0,
            "Tile%s (%d) must be a multiple of %d for C SF layout %s", options.mTransposeMmaOutput ? "N" : "M",
            tileTokenDim, numSfTileRowsC, tg::sfLayoutToString(options.mSfLayoutC).c_str());

        int numEltsPerSfC = options.mSfBlockSizeC;
        int const hiddenDim = options.mTransposeMmaOutput ? options.mM : options.mN;
        int const hiddenGranularity = 4 * numEltsPerSfC;
        TLLM_CHECK_ERROR(hiddenDim % hiddenGranularity == 0, "Hidden dim (", hiddenDim, ") must be a multiple of ",
            hiddenGranularity, " for block-scaled outputs.");
        int const validHiddenDim = options.mTransposeMmaOutput ? options.mValidM : options.mValidN;
        TLLM_CHECK_ERROR(validHiddenDim % numEltsPerSfC == 0, "Valid hidden dim (", validHiddenDim,
            ") must be a multiple of ", numEltsPerSfC, " for block-scaled outputs.");
        TLLM_CHECK_ERROR(!options.mTransposeMmaOutput || options.mUseShuffledMatrix,
            "Transposing block-scaled outputs requires shuffled matrix.");
    }

    // Set epilogue tile sizes to the output tile sizes, when epilogue tile sizes are incorrect.
    if (options.mTileM % options.mEpilogueTileM != 0)
    {
        TLLM_LOG_WARNING("TileM (", options.mTileM, ") must be divisible by EpilogueTileM (", options.mEpilogueTileM,
            "). Setting EpilogueTileM to TileM");
        if (updateOptions)
        {
            options.mEpilogueTileM = options.mTileM;
        }
        else
        {
            return false;
        }
    }

    if (options.mTileN % options.mEpilogueTileN != 0)
    {
        TLLM_LOG_WARNING("TileN (", options.mTileN, ") must be divisible by EpilogueTileN (", options.mEpilogueTileN,
            "). Setting EpilogueTileN to TileN");
        if (updateOptions)
        {
            options.mEpilogueTileN = options.mTileN;
        }
        else
        {
            return false;
        }
    }

    // On Hopper, epilogue tile sizes are the same as output tiles.
    if (!isBlackwell && (options.mEpilogueTileM != options.mTileM || options.mEpilogueTileN != options.mTileN))
    {
        TLLM_LOG_WARNING("Overwriting epilogueTileM and epilogueTileN to match tileM and tileN respectively");
        if (updateOptions)
        {
            options.mEpilogueTileM = options.mTileM;
            options.mEpilogueTileN = options.mTileN;
        }
        else
        {
            return false;
        }
    }

    // Unsupported epilogue tile size.
    if (options.mMmaM == 128 && options.mEpilogueTileM != options.mTileM)
    {
        TLLM_LOG_WARNING("When MmaM = 128, EpilogueTileM must be equal to TileM. Setting EpilogueTileM to TileM");
        if (updateOptions)
        {
            options.mEpilogueTileM = options.mTileM;
        }
        else
        {
            return false;
        }
    }

    TLLM_CHECK_ERROR(options.mM > 0 && options.mN > 0 && options.mK > 0, "M, N and K must be larger than 0");
    TLLM_CHECK_ERROR(options.mNumSlicesForSplitK > 0, "Split K must be larger than 0.");

    if (options.mUseShuffledMatrix)
    {
        auto const shuffleBlockSize = getShuffleBlockSize(options.mEpilogueTileM);
        TLLM_CHECK_ERROR(options.mM % shuffleBlockSize == 0 && options.mValidM % shuffleBlockSize == 0,
            "M/validM must be a multiple of shuffle block size (", shuffleBlockSize, ") when useShuffledMatrix");
    }

    if (!options.mSliceK)
    {
        TLLM_CHECK_ERROR(options.mMmaM / options.mClusterDimX <= options.mEpilogueTileM,
            "EpilogueTileM must be larger or equal than mmaM.");
    }
    else
    {
        // FIXME: this is not necessary limitation. Simply fixing num repeats in TmemSliceKA should be
        // enough.
        TLLM_CHECK_ERROR(
            (options.mTileN & (options.mTileN - 1)) == 0, "For Slice-K TileN is required to be a power of 2");
    }

    if (options.mClusterDimX == 2)
    {
        TLLM_CHECK_ERROR(options.mMmaM == 256, "Only mmaM = 256 is supported for 2CTA UTCMMA.");
        TLLM_CHECK_ERROR(options.mMmaN % 16 == 0, "mmaN needs to be multiple of 16 for 2CTA UTCMMA.");
    }

    TLLM_CHECK_ERROR(options.mTileM % options.mEpilogueTileM == 0 && options.mTileN % options.mEpilogueTileN == 0,
        "TileM and TileN must be divisible by EpilogueTileM and EpilogueTileN respectively.");
    TLLM_CHECK_ERROR((options.mClusterDimX == 1 || options.mClusterDimX == 2) && options.mClusterDimY == 1,
        "GEMM does not support cluster in X and Y dimensions.");
    TLLM_CHECK_ERROR(
        options.mClusterDimZ == 1 || options.mNumSlicesForSplitK > 1, "Cluster DimZ is only allowed for split-k.");
    TLLM_CHECK_ERROR(options.mTileM <= 128, "GEMM does not support TileM > 128.");

    // FIXME: this is a bug in DeepSeek Fp8.
    if (options.mUseDeepSeekFp8)
    {
        TLLM_CHECK_ERROR(options.mK % (options.mNumSlicesForSplitK * options.mTileK) == 0,
            "K must be a multiple of TileK * numSlicesForSplitK for DeepSeekFp8");
    }

    // Check all-reduce options.
    if (options.mAllReduceAlgo == AllReduceAlgo::OneShot)
    {
        // One shot is implemented with PTX cp.reduce.async.bulk.tensor which supports only the
        // following types for reduce add: u32, s32, u64, f32, f16, bf16.
        //
        // See: https://docs.nvidia.com/cuda/parallel-thread-execution/
        //   #data-movement-and-conversion-instructions-cp-reduce-async-bulk-tensor
        std::set<tg::Dtype> dtypeSupported{tg::Dtype::UInt32, tg::Dtype::Int32, tg::Dtype::UInt64, tg::Dtype::Fp32,
            tg::Dtype::Fp16, tg::Dtype::Bfloat16};
        TLLM_CHECK_ERROR(dtypeSupported.find(options.mDtypeC) != dtypeSupported.end(), "Unsupported output dtype ",
            tg::dtypeToString(options.mDtypeC));
    }
    else if (options.mAllReduceAlgo == AllReduceAlgo::TwoShot)
    {
        // TODO(anchengc):
        // Input dtype == output dtype -> can perform all-reduce in-place.
        // Input dtype != output dtype -> must perform all-reduce out of place.
        TLLM_CHECK_ERROR_FMT(options.mDtypeC == options.mDtypeAcc,
            "Not implemented - mixed dtype (dtypeC (%s) != dtypeAcc (%s)) requires out of place update",
            tg::dtypeToString(options.mDtypeC).c_str(), tg::dtypeToString(options.mDtypeAcc).c_str());
    }
    if (options.mAllReduceAlgo != AllReduceAlgo::None)
    {
        TLLM_CHECK_ERROR(options.mUseTmaStore, "Non-TMA store with all-reduce is not implemented");
    }

    if (updateOptions)
    {
        if (options.mNumSlicesForSplitK == 1)
        {
            // No split-k.
            options.mSplitK = SplitK::None;
        }
        else if (options.mNumSlicesForSplitK > 1 && options.mClusterDimZ == 1)
        {
            // Split-k with exchange through gmem.
            options.mSplitK = SplitK::Gmem;
        }
        else
        {
            // Split-k with exchange through Dsmem.
            options.mSplitK = SplitK::Dsmem;
        }
    }
    // For GMEM-based split-K, we write 4 elements at once.
    if (options.mSplitK == SplitK::Gmem)
    {
        TLLM_CHECK_ERROR((options.mM * options.mN) % 4 == 0, "M * N must be a multiple of 4 for Split-K");
    }

    if (options.mNumSlicesForSplitK > 1)
    {
        if ((options.mEpilogueTileM != options.mTileM || options.mEpilogueTileN != options.mTileN)
            && !options.mUseDeepSeekFp8)
        {
            TLLM_LOG_WARNING("Overwriting epilogueTileM and epilogueTileN to match tileM and tileN respectively");
            if (updateOptions)
            {
                options.mEpilogueTileM = options.mTileM;
                options.mEpilogueTileN = options.mTileN;
            }
            else
            {
                return false;
            }
        }
    }
    if (options.mSplitK == SplitK::Dsmem)
    {
        TLLM_CHECK_ERROR(options.mClusterDimZ == options.mNumSlicesForSplitK,
            "CGA size must be equal to the number of slices in split-k");
    }

    if (options.mUseShuffledMatrix && !options.mTransposeMmaOutput)
    {
        TLLM_CHECK_ERROR(!options.mUseDeepSeekFp8,
            "DeepSeek Fp8 is not supported when using shuffled matrix and non-transposed mma output");
        TLLM_CHECK_ERROR(options.mEpilogueLdtmBits == 32,
            "EpilogueLdtmBits must be 32 when using shuffled matrix and non-transposed mma output");
        TLLM_CHECK_ERROR(options.mEpilogueLdtmDps == 32,
            "EpilogueLdtmDps must be 32 when using shuffled matrix and non-transposed mma output");
        TLLM_CHECK_ERROR(
            options.mUseTmaStore, "TMA store is required when using shuffled matrix and non-transposed mma output");
        TLLM_CHECK_ERROR(
            !options.mSliceK, "Slice-K is not supported when using shuffled matrix and non-transposed mma output");
        // When doing unshuffle in the epilogue, one fragment of epilogue tile must have at least one
        // shuffle block.
        auto minEpilogueTileN = getShuffleBlockSize(options.mEpilogueTileM);
        TLLM_CHECK_ERROR_FMT(options.mEpilogueTileN >= minEpilogueTileN,
            "EpilogueTileN (%d) must be a larger than the shuffle block size (%d) "
            "when using shuffled matrix and non-transposed mma output",
            options.mEpilogueTileN, minEpilogueTileN);
    }

    // Maps numStagesMma to (stagesWithinWorkTile, stagesAcrossWorkTile) if not already set.
    // If (-1, -1) -> (numStagesMma / min(2, numStagesMma), min(2, numStagesMma))
    // If ( m, -1) -> (m, numStagesMma / m)
    // If (-1,  n) -> (numStagesMma / n, n)
    if (options.mNumStagesMmaWithinWorkTile == -1 && options.mNumStagesMmaAcrossWorkTile == -1)
    {
        if (updateOptions)
        {
            options.mNumStagesMmaAcrossWorkTile = std::min(2, options.mNumStagesMma);
            options.mNumStagesMmaWithinWorkTile = options.mNumStagesMma / options.mNumStagesMmaAcrossWorkTile;
        }
        else
        {
            return false;
        }
    }
    else if (options.mNumStagesMmaWithinWorkTile == -1)
    {
        if (updateOptions)
        {
            options.mNumStagesMmaWithinWorkTile = options.mNumStagesMma / options.mNumStagesMmaAcrossWorkTile;
        }
        else
        {
            return false;
        }
    }
    else if (options.mNumStagesMmaAcrossWorkTile == -1)
    {
        if (updateOptions)
        {
            options.mNumStagesMmaAcrossWorkTile = options.mNumStagesMma / options.mNumStagesMmaWithinWorkTile;
        }
        else
        {
            return false;
        }
    }
    // Check mma stages.
    TLLM_CHECK_ERROR_FMT(
        options.mNumStagesMmaWithinWorkTile * options.mNumStagesMmaAcrossWorkTile == options.mNumStagesMma
            && options.mNumStagesMmaAcrossWorkTile <= 2,
        "Condition numStagesMmaWithinWorkTile (%d) * numStagesMmaAcrossWorkTile "
        "(%d) == numStagesMma (%d) && numStagesMmaAcrossWorkTile (%d) <= 2 must be "
        "satisfied. Check arguments.",
        options.mNumStagesMmaWithinWorkTile, options.mNumStagesMmaAcrossWorkTile, options.mNumStagesMma,
        options.mNumStagesMmaAcrossWorkTile);
    // Mma stage must be 1 for pre-Hopper.
    TLLM_CHECK_ERROR(
        isBlackwell || options.mNumStagesMma == 1, "Mma stage must be 1 for pre-Hopper. Found ", options.mNumStagesMma);
    // DeepSeek Fp8
    if (!options.mUseDeepSeekFp8)
    {
        TLLM_CHECK_ERROR(
            options.mNumStagesMmaWithinWorkTile == 1, "Non-DeepSeekFp8 requires numStagesMmaWithinWorkTile == 1");
        if (options.mNumStagesMma > 1)
        {
            TLLM_CHECK_ERROR(isPersistentScheduler(options.mTileScheduler),
                "Non-DeepSeekFp8 requires persistent scheduler when using numStagesMma >1");
        }
    }
    if (options.mUseDeepSeekFp8)
    {
        TLLM_CHECK_ERROR(options.mClusterDimX == 1, "2CTA Gemm is not supported for DeepSeekFp8");
    }
    if (options.mUseDeepSeekFp8)
    {
        TLLM_CHECK_ERROR(options.mDtypeA == tg::Dtype::E4m3 && options.mDtypeB == tg::Dtype::E4m3,
            "A and B dtype must be E4m3 for DeepSeek Fp8. Found dtypeA=", tg::dtypeToString(options.mDtypeA),
            " dtypeB=", tg::dtypeToString(options.mDtypeB));

        TLLM_CHECK_ERROR(isBlackwell, "DeepSeek Fp8 is not supported for Hopper");
        TLLM_CHECK_ERROR(options.mAllReduceAlgo == AllReduceAlgo::None, "DeepSeek Fp8 does not support AllReduce");

        // Check that TileK = 128 for correct scaling of every 128 channels.
        TLLM_CHECK_ERROR(options.mTileK == 128, "Tile-K must be equal to 128 for DeepSeek Fp8");
        TLLM_CHECK_ERROR(options.mK % options.mTileK == 0, "K must be a multiple of TileK");
        // Tile sizes of the output hidden dimension.
        auto hiddenDimPerOutputTile = options.mTransposeMmaOutput ? options.mTileM : options.mTileN;
        auto hiddenDimPerEpilogueTile = options.mTransposeMmaOutput ? options.mEpilogueTileM : options.mEpilogueTileN;
        auto hiddenDimPerMma = options.mTransposeMmaOutput ? options.mMmaM : options.mMmaN;
        auto hiddenDimName = options.mTransposeMmaOutput ? "M" : "N";
        TLLM_CHECK_WARNING(options.mNumStagesMmaWithinWorkTile > 1,
            "DeepSeekFp8 recommends setting \"-numStagesMmaWithinWorkTile 2\".");
        // Update the number of stages of the MMA accumulator pipeline. TODO: enable by default for
        // deepseek.
        // options.mNumStagesMma = 2;
        // Use two MMA warps to reduce mbar trywait latency. TODO: enable by default for deepseek.
        // options.mUseTwoMmaWarps = true;

        // Make sure the GEMM-K dimension is a multiple of 128 when using DeepSeek FP8.
        TLLM_CHECK_ERROR(options.mK % 128 == 0 && options.mValidK % 128 == 0,
            "GEMM-K and validK must be a multiple of 128 when using DeepSeek Fp8. Found ", options.mK,
            " and validK=", options.mValidK);

        // Check that the output tile N can be processed with the epilogue tile granularity.
        TLLM_CHECK_ERROR((hiddenDimPerOutputTile / 2) % hiddenDimPerEpilogueTile == 0, "DeepSeek Fp8 requires Tile",
            hiddenDimName, " / 2 (", hiddenDimPerOutputTile / 2, ") being a multiple of EpilogueTile", hiddenDimName,
            " (", hiddenDimPerEpilogueTile, ")");
        // Check that the output tile N can be processed with the epilogue tile granularity.
        TLLM_CHECK_ERROR((hiddenDimPerOutputTile / 2) % hiddenDimPerMma == 0, "DeepSeek Fp8 requires Tile",
            hiddenDimName, " / 2 (", hiddenDimPerOutputTile / 2, ") being a multiple of mma", hiddenDimName, " (",
            hiddenDimPerMma, ")");
    }

    TLLM_CHECK_ERROR(
        options.mNumEpilogueWarps == 4 || options.mNumEpilogueWarps == 8, "mNumEpilogueWarps has to be either 4 or 8.");

    if (options.mSliceK)
    {
        TLLM_CHECK_ERROR(isBlackwell, "Slice-K is not supported on Hopper");

        TLLM_CHECK_ERROR(!options.mUseDeepSeekFp8, "DeepSeek Fp8 GEMM is not supported for slice-K");
        TLLM_CHECK_ERROR(options.mUseTwoTmaLoadWarps, "Slice-K requires two warp load for A and B");
        TLLM_CHECK_ERROR(options.mTransposeMmaOutput, "Slice-K requires transpose mma output");
        TLLM_CHECK_ERROR(options.mUseShuffledMatrix, "Slice-K requires shuffled matrix");
        TLLM_CHECK_ERROR(options.mTileK % 128 == 0, "Slice-K requires TileK be a multiple of 128");
        TLLM_CHECK_ERROR(options.mMmaM == 128, "Slice-K requires MmaM == 128");
        TLLM_CHECK_ERROR(options.mTileN == options.mEpilogueTileN, "TileN must be equal to EpilogueTileN for slice-K");

        TLLM_LOG_WARNING("Overwriting TileM and EpilogueTileM to 32 for slice-K");
        if (options.mTileM != 32 || options.mEpilogueTileM != 32)
        {
            if (updateOptions)
            {
                // FIXME: it is possible to remove this restriction.
                options.mTileM = 32;
                options.mEpilogueTileM = 32;
            }
            else
            {
                return false;
            }
        }
        TLLM_CHECK_ERROR(options.mDtypeA == tg::Dtype::E4m3 && options.mDtypeB == tg::Dtype::E4m3,
            "Slice-K requires e4m3 input dtype");

        if (options.mNumSlicesForSliceK != 4)
        {
            if (updateOptions)
            {
                options.mNumSlicesForSliceK = 4;
            }
            else
            {
                return false;
            }
        }
        TLLM_CHECK_ERROR((options.mTileK / options.mMmaK) % options.mNumSlicesForSliceK == 0, "TileK (", options.mTileK,
            ") / MmaK (", options.mMmaK, ") must be a multiple of mNumSlicesForSliceK (", options.mNumSlicesForSliceK,
            ")");
    }

    // Number of iterations in K dimension after padding.
    // Note the perCtaK in each CTA in the splitK group are padded to the same number of iterations.
    // E.g., K = 512, TileK = 128, numSlicesForSplitK = 3. Then the padded K is
    //
    //   ceil(512 / (128*3)) * (128*3) = 768
    //
    int const paddedK = divUpMul(options.mK, options.mTileK * options.mNumSlicesForSplitK);
    int const perCtaK = paddedK / options.mNumSlicesForSplitK;
    // However, number of iterations is clamped to multiples of tileK within individual CTAs
    // E.g., K = 448, TileK = 64, numSlicesForSplitK = 4.
    //
    //   paddedK                        = 512
    //   perCtaK                        = 128
    //   clampedPerCtaK for CTA 0, 1, 2 = 128
    //   clampedPerCtaK for CTA 3       = 64
    int const paddingForK = paddedK - options.mK;
    int const clampedAndPaddedPerCtaK = divUpMul(perCtaK - paddingForK, options.mTileK);
    if (options.mUseUnrollLoop2xForMma)
    {
        // Check that the padded K and clamped padded K (K rounded to next multiple of tileK) is a
        // multiple of 2*TileK when UnrollLoop2x is enabled. This is to avoid deadlock when mma runs
        // even-numbered loop while the other warps run odd-numbered loop.
        //
        bool notSupported
            = (perCtaK % (options.mTileK * 2) != 0) || (clampedAndPaddedPerCtaK % (options.mTileK * 2) != 0);
        if (notSupported)
        {
            TLLM_LOG_WARNING("Size K / splitK must be a multiple of TileK * 2. Found TileK=", options.mTileK,
                " and K=", options.mK, " (paddedK=", paddedK, " clampedAndPaddedPerCtaK=", clampedAndPaddedPerCtaK,
                ") and numSlicesForSplitK=", options.mNumSlicesForSplitK, ". Disabling unrollLoop2xForMma.");
            if (updateOptions)
            {
                options.mUseUnrollLoop2xForMma = false;
            }
            else
            {
                return false;
            }
        }
    }
    if (options.mNumSlicesForSplitK > 1)
    {
        TLLM_CHECK_ERROR(perCtaK * (options.mNumSlicesForSplitK - 1) < options.mK,
            "K must be greater than perCtaK * (numSlicesForSplitK - 1) to ensure each CTA has work");
    }

    if (!isBlackwell && options.mTileScheduler == TileScheduler::Persistent)
    {
        TLLM_LOG_WARNING(
            "Persistent scheduling is not supported on Hopper. Use StaticPersistent or "
            "PersistentSm90 instead. Fallback to Static scheduling.");
        if (updateOptions)
        {
            options.mTileScheduler = TileScheduler::Static;
        }
        else
        {
            return false;
        }
    }

    if (isBlackwell && !options.mUseCustomMmaSchedule && !options.mUseDeepSeekFp8
        && isPersistentScheduler(options.mTileScheduler))
    {
        if (updateOptions)
        {
            options.mUseCustomMmaSchedule = true;
        }
        else
        {
            TLLM_CHECK_ERROR(false, "Persistent scheduler and !UseCustomMmaSchedule is not supported.");
        }
    }

    if (options.mEnablesDelayedEarlyExit && options.mEnablesEarlyExit)
    {
        TLLM_LOG_WARNING(
            "Only one of early exit and delayed early exit should be enabled. Disabling "
            "delayed early exit");
        if (updateOptions)
        {
            options.mEnablesDelayedEarlyExit = false;
        }
        else
        {
            return false;
        }
    }

    // This check prevents the triggering of the secondary (PREEXIT) from executing before the wait
    // for primary (ACQBULK). This could lead to the following confusing situation, which we want to
    // avoid:
    //
    // Kernel 3 is written with the assumption that it can read the output of
    // kernel 1 *without* ACQBULK and the output of kernel 2 *with* ACQBULK.
    // However, when we allow PREEXIT and ACQBULK to be executed out of order,
    // this is not guaranteed.
    //
    // Time:      ---->
    //
    // Kernel 1:  ----PREEXIT-----------FLUSH
    // Kernel 2:      -------PREEXIT----ACQBULK---FLUSH
    // Kernel 3:  Warp 0:           ---- (!) Output of 1,2 is not yet visible
    // -----------------------
    //            Warp 1:           ---- (!) We normally assume that 1 is visible is not yet
    //            visible- Warp 2:           -------------------ACQBULK-- Kernel 1,2 output visible
    //            ----------
    TLLM_CHECK_ERROR((options.mGridWaitForPrimaryA || !options.mGridTriggerSecondaryA),
        "A: If a task triggers a secondary kernel, it must also wait for primary kernel.");
    TLLM_CHECK_ERROR((options.mGridWaitForPrimaryB || !options.mGridTriggerSecondaryB),
        "B: If a task triggers a secondary kernel, it must also wait for primary kernel.");

    if (options.mUsePerTokenSfA || options.mUsePerTokenSfB)
    {
        // Checks applicable to both MetaFP8 and RoutingScalesOnInput
        TLLM_CHECK_ERROR(!options.mUseDeepSeekFp8, "DeepSeek FP8 and per-token scaling are not compatible");
        TLLM_CHECK_ERROR(isBlackwell, "Per-token scaling is not supported for Hopper");
        if (options.mUsePerTokenSfA && options.mUsePerTokenSfB)
        {
            // MetaFP8 case
            TLLM_CHECK_ERROR(options.mDtypeA == tg::Dtype::E4m3 && options.mDtypeB == tg::Dtype::E4m3,
                "A and B dtype must be E4m3 for Meta Fp8. Found dtypeA=", tg::dtypeToString(options.mDtypeA),
                " dtypeB=", tg::dtypeToString(options.mDtypeB));
        }
        else
        {
            // RoutingScalesOnInput case
            TLLM_CHECK_ERROR((options.mUsePerTokenSfA && !options.mTransposeMmaOutput)
                    || (options.mUsePerTokenSfB && options.mTransposeMmaOutput),
                "In RoutingScalesOnInput mode, perToken scales must be used on activations");
        }
    }

    // The generation should support non K-major layouts for both A and B; however, it is unclear if
    // there is a use-case
    TLLM_CHECK_ERROR((options.mLayoutA == MatrixLayout::MajorK) || (options.mLayoutB == MatrixLayout::MajorK),
        "At least one matrix must be in k-major layout");

    // Some features are currently only support when both matrices are in K-major format
    if (options.mLayoutA != MatrixLayout::MajorK || options.mLayoutB != MatrixLayout::MajorK)
    {
        TLLM_CHECK_ERROR(isBlackwell, "Non K-major layouts are only supported on Blackwell");
        TLLM_CHECK_ERROR(options.mSplitK == SplitK::None, "Non K-major layouts do not support split K");
    }
    if (options.mLayoutA == MatrixLayout::MajorMn)
    {
        TLLM_CHECK_ERROR(tg::dtypeGetNumBits(options.mDtypeA) >= 8, "Subbyte types do not support m-major layout");
    }
    if (options.mLayoutB == MatrixLayout::MajorMn)
    {
        TLLM_CHECK_ERROR(tg::dtypeGetNumBits(options.mDtypeB) >= 8, "Subbyte types do not support n-major layout");
    }

    if ((options.mLayoutA == MatrixLayout::BlockMajorK) || (options.mLayoutB == MatrixLayout::BlockMajorK))
    {
        bool const isBlockA = options.mLayoutA == MatrixLayout::BlockMajorK;

        // Block K size must be 128B.
        // TODO Leaving this as an option for now in case we want to expertiment with other block sizes
        // As the user is not expected to set this, do not fail if updateOptions is false
        int32_t const elemSizeInBits
            = (isBlockA) ? tg::dtypeGetNumBits(options.mDtypeA) : tg::dtypeGetNumBits(options.mDtypeB);
        int32_t const elemsIn128B = 128 * 8 /* Bits in byte */ / elemSizeInBits;

        if (options.mBlockK != elemsIn128B)
        {
            if (updateOptions)
            {
                options.mBlockK = elemsIn128B;
            }
            else
            {
                return false;
            }
        }

        if (options.mBlockK > options.mTileK)
        {
            TLLM_CHECK_ERROR(options.mBlockK % options.mTileK == 0,
                "If block size is greater than tile size, block size must be a multiple of tile size");
        }
        else if (options.mBlockK < options.mTileK)
        {
            TLLM_CHECK_ERROR(options.mTileK % options.mBlockK == 0,
                "If tile size is greater than block size, tile size must be a multiple of block size");
        }
    }

    if (!isBiasTypeNone(options.mBiasType))
    {
        TLLM_CHECK_ERROR(!isBiasTypeMn(options.mBiasType), "BiasType::Mn is not supported");
        TLLM_CHECK_ERROR(!options.mUseDeepSeekFp8, "Bias is not supported for DeepSeek Fp8");
        TLLM_CHECK_ERROR(!(options.mUsePerTokenSfA && options.mUsePerTokenSfB), "Bias is not supported for Meta Fp8");
    }

    if (options.mUseMaxTmemOverlap)
    {
        TLLM_CHECK_ERROR(options.mUseTmaStore, "mUseMaxTmemOverlap only works with TMA store");
        TLLM_CHECK_ERROR(options.mFuseUtccpWithUtcmma, "mUseMaxTmemOverlap only works with mFuseUtccpWithUtcmma");
        TLLM_CHECK_ERROR(options.mNumSlicesForSplitK == 1, "mUseMaxTmemOverlap does not work with splitK");
        TLLM_CHECK_ERROR(options.mNumSlicesForSliceK == 1, "mUseMaxTmemOverlap does not work with sliceK");
        TLLM_CHECK_ERROR(!options.mUseDeepSeekFp8, "mUseMaxTmemOverlap does not work with mUseDeepSeekFp8");
        TLLM_CHECK_ERROR(
            !options.mUseUnrollLoop2xForMma, "mUseMaxTmemOverlap does not work with mUseUnrollLoop2xForMma");
    }

    if (options.mNumEpilogueWarps > 4)
    {
        TLLM_CHECK_ERROR(options.mUseTmaStore, "Using more than 4 warps for epilogue only works with TMA store");
        TLLM_CHECK_ERROR(
            options.mNumSlicesForSplitK == 1, "Using more than 4 warps for epilogue does not work with splitK");
        TLLM_CHECK_ERROR(
            options.mNumSlicesForSliceK == 1, "Using more than 4 warps for epilogue does not work with sliceK");
        TLLM_CHECK_ERROR(
            !options.mUseDeepSeekFp8, "Using more than 4 warps for epilogue does not work with mUseDeepSeekFp8");

        auto const numEpilogueWrpGrps = options.mNumEpilogueWarps / 4;
        TLLM_CHECK_ERROR(options.mTileN % (options.mEpilogueTileN * numEpilogueWrpGrps) == 0,
            "TileN must be a multiple of EpilogueTileN * numEpilogueWrpGrps");
    }

    if (updateOptions)
    {
        // Init kernel traits.
        options.mKernelTraits = KernelTraits(options.mDtypeA, options.mDtypeB, options.mDtypeC, options.mDtypeAcc,
            options.mDtypeMmaA, options.mDtypeMmaB, options.mMmaKind, options.mSparsityA, options.mMmaK, options.mTileM,
            options.mTileN, options.mTileK, options.mEpilogueTileM, options.mEpilogueTileN, options.mSfBlockSizeA,
            options.mSfBlockSizeB, options.mNumStages, options.mNumStagesMma, options.mNumSlicesForSplitK,
            options.mNumSlicesForSliceK, options.mSplitK, options.mUseTmaStore, options.mTransposeMmaOutput,
            options.mAllReduceAlgo, options.mFuseUtccpWithUtcmma, options.mUseMaxTmemOverlap, options.mNumEpilogueWarps,
            isPersistentScheduler(options.mTileScheduler), options.mUseDeepSeekFp8, options.mUsePerTokenSfA,
            options.mUsePerTokenSfB,
            /* useTwoCtas*/ options.mClusterDimX == 2, options.mBiasType);
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline bool getDoesScaleC(tg::Dtype dtypeC)
{
    // Need to scale/quantize the output C matrix when the output type is Fp8 or NvFp4.
    return dtypeC == tg::Dtype::E4m3 || dtypeC == tg::Dtype::E2m1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline bool getDoesScaleAb(tg::Dtype dtypeA, tg::Dtype dtypeB, bool useDeepSeekFp8)
{
    // Need to scale/dequantize the input A/B matrices when the input type is Fp8 or NvFp4 and
    // DeepSeekFp8 is not used.
    bool const doesScaleAb{dtypeA == tg::Dtype::E2m1 || dtypeB == tg::Dtype::E2m1
        || ((dtypeA == tg::Dtype::E4m3 || dtypeB == tg::Dtype::E4m3) && !useDeepSeekFp8)};
    return doesScaleAb;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

inline bool getDoesScaleAct(tg::Dtype dtypeA, tg::Dtype dtypeB, bool useDeepSeekFp8, EltwiseActType eltwiseActType)
{
    // Only non-linear activations require separate scaleAct.
    bool const isLinearAct = eltwiseActType == EltwiseActType::None;
    return !isLinearAct && getDoesScaleAb(dtypeA, dtypeB, useDeepSeekFp8);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline bool getKernelDoesScaleC(tg::Dtype dtypeA, tg::Dtype dtypeB, tg::Dtype dtypeC, bool useDeepSeekFp8)
{
    // In the Gemm/BatchedGemm kernels, dequantScaleAb and quantScaleC are combined into one single
    // scaling factor (called scaleC). As a result, we combine the logic for getDoesScaleAb and
    // getDoesScaleC.
    return getDoesScaleC(dtypeC) || getDoesScaleAb(dtypeA, dtypeB, useDeepSeekFp8);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Config>
inline CUresult loadCubinData(CUmodule* module, Config const& config)
{
    // Trtllm links the cubin into the executable while Flashinfer loads the cubin from storage.
#ifdef TLLM_GEN_EXPORT_FLASHINFER
#ifdef TLLM_GEN_GEMM_CUBIN_PATH
    static const std::string tllm_gen_gemm_cubin_path = std::string(TLLM_GEN_GEMM_CUBIN_PATH);
    const std::string sha256 = config.mHash ? config.mHash : "";
    std::string fileName = config.mFunctionName;
    if (!fileName.empty())
    {
        fileName[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(fileName[0])));
    }
    std::string const& data
        = flashinfer::trtllm_cubin_loader::getCubin(tllm_gen_gemm_cubin_path + "/" + fileName + ".cubin", sha256);
    CUresult result = cuModuleLoadData(module, data.c_str());
#else
    static_assert(false, "TLLM_GEN_GEMM_CUBIN_PATH macro is not defined when compiling");
#endif // TLLM_GEN_GEMM_CUBIN_PATH
#else
    CUresult result = cuModuleLoadData(module, config.mData);
#endif // TLLM_GEN_EXPORT_FLASHINFER
    return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace gemm

#ifdef TLLM_GEN_EXPORT_INTERFACE

#undef TLLM_CHECK_ERROR
#undef TLLM_CHECK_ERROR_FMT
#undef TLLM_CHECK_WARNING
#undef TLLM_LOG_WARNING
#undef TLLM_LOG_INFO
#undef TLLM_LOG_ERROR

#endif // TLLM_GEN_EXPORT_INTERFACE

} // namespace batchedGemm
