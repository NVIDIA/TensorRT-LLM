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

#include <set>
#include <sstream>

#include "Enums.h"
#include "KernelParams.h"
#include "KernelTraits.h"
#include "trtllm/gen/DtypeDecl.h"
#include "trtllm/gen/SfLayoutDecl.h"
#ifndef TLLM_GEN_EXPORT_INTERFACE
#include "trtllm/gen/CudaRunner.h"
#include "trtllm/gen/GenCtx.h"
#else
#include <iostream>

template <typename T, typename... Args>
void printArgs(T first, Args... args)
{
    std::cout << first;
    if constexpr (sizeof...(args) > 0)
    {
        std::cout << " ";
        printArgs(args...);
    }
}

#define TLLM_CHECK_ERROR(cond, ...)                                                                                    \
    if (!(cond))                                                                                                       \
    {                                                                                                                  \
        printArgs(__VA_ARGS__);                                                                                        \
        return false;                                                                                                  \
    }

#define TLLM_LOG_ERROR(...) TLLM_CHECK_ERROR(false, __VA_ARGS__)

#define TLLM_CHECK_ERROR_FMT(cond, ...) TLLM_CHECK_ERROR(cond, __VA_ARGS__)

#define TLLM_CHECK_WARNING(cond, ...)                                                                                  \
    if (!(cond))                                                                                                       \
    {                                                                                                                  \
        printArgs(__VA_ARGS__);                                                                                        \
        return false;                                                                                                  \
    }

#define TLLM_LOG_WARNING(...) TLLM_CHECK_WARNING(false, __VA_ARGS__)

#define TLLM_LOG_INFO(...) TLLM_CHECK_WARNING(false, __VA_ARGS__)

#endif

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

    // The all-reduce algorithm.
    AllReduceAlgo mAllReduceAlgo{AllReduceAlgo::None};

    // Cluster size in X dim.
    int mClusterDimX{1};
    // Cluster size in Y dim.
    int mClusterDimY{1};
    // Cluster size in Z dim.
    int mClusterDimZ{1};
    // Data type of the accumulators.
    tg::Dtype mDtypeAcc{tg::Dtype::Fp32};
    // Data type of the inputs.
    tg::Dtype mDtypeElt{tg::Dtype::Fp16};
    // Data type of the outputs.
    tg::Dtype mDtypeC{tg::Dtype::Void};
    // Whether to enable early exit.
    bool mEnablesEarlyExit{false};
    // Whether to enable early exit.
    bool mEnablesDelayedEarlyExit{false};
    // Whether to enable the global PTX knobs for guiding the compiler optimizations.
    bool mEnablesGlobalPtxKnobs{true};
    // Tile size for the epilogue in M dimension.
    int mEpilogueTileM{128};
    // Tile size for the epilogue in N dimension.
    int mEpilogueTileN{32};
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
    // Whether to hoist the mbarrier try_waits (e.g., mma.prodAcq, smemAb.consWait) in the MMA task.
    bool mHoistMmaTaskTryWaits{false};
    // The K dimension of GEMM.
    int mK{16 * 16};
    // Traits of the kernel.
    KernelTraits mKernelTraits{};
    // The M dimension of GEMM.
    int mM{128 * 2};
    // Size of the MMA instruction in the K dimension.
    int mMmaK{16};
    // Size of the MMA instruction in the M dimension.
    int mMmaM{64};
    // Size of the MMA instruction in the N dimension.
    int mMmaN{16};
    // Whether to mock all-reduce code for single-GPU debugging.
    bool mMockAllReduce{false};
    // The N dimension of GEMM.
    int mN{64 * 4};
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
    // Reorder rows/cols in the A matrix for the better memory accesses in the M-major epilogue.
    bool mUseShuffledMatrixA{false};
    // Slice-K implementation to use TileM dimension for TileK.
    bool mSliceK{false};
    // The location of the exchange for split-K (it's None when split-K is disabled).
    SplitK mSplitK{SplitK::None};
    // Save output of MMA in M-major format.
    bool mTransposeMmaOutput{false};
    // M tile dimension of GEMM.
    int mTileM{128};
    // N tile dimension of GEMM.
    int mTileN{32};
    // K tile dimension of GEMM.
    int mTileK{16};
    // Whether to unroll the loop by 2x.
    bool mUseUnrollLoop2xForMma{true};
    // Use custom MMA schedule optimized for low-latency.
    bool mUseCustomMmaSchedule{false};
    // The purpose of hoisting trywaits is to opportunistically peek at the availability of the next
    // k-block. It benefits when the next k-block is already available and thus sustaining the
    // momentum, but it adds latency to the first k-block for smaller k-loop.
    bool mUseHoistTryWaitForCustomMmaSchedule{false};
    // Use DeepSeek Fp8.
    bool mUseDeepSeekFp8{false};
    // Apply per-token scales from A
    bool mUsePerTokenSfA{false};
    // Apply per-token scales from B
    bool mUsePerTokenSfB{false};
    // Use TMA to store the result.
    bool mUseTmaStore{true};
    // Use two different warps for A and B matrix load.
    bool mUseTwoTmaLoadWarps{false};
    // Use two different warps for MMA tasks. Applicable only to DeepSeek FP8.
    bool mUseTwoMmaWarps{false};
    // Scale factors layout for A.
    tg::SfLayout mSfLayoutA{tg::SfLayout::R128c4};
    // Scale factors layout for B.
    tg::SfLayout mSfLayoutB{tg::SfLayout::R128c4};
    // Scale factors layout for C.
    tg::SfLayout mSfLayoutC{tg::SfLayout::R128c4};
    // Tile scheduler type.
    TileScheduler mTileScheduler{TileScheduler::Static};
};

////////////////////////////////////////////////////////////////////////////////////////////////////

enum class SmVersion
{
    Sm90a,
    Sm100a
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// GemmConfig
//
////////////////////////////////////////////////////////////////////////////////////////////////////

struct GemmConfig
{
    // When TRT-LLM Gen is exported to the other frameworks, the TLLM_GEN_EXPORT_INTERFACE must be
    // defined. In this case, the cubins will be loaded from the provided data and function name.
    // Otherwise, the kernel will be loaded from the CudaRunner.
#ifdef TLLM_GEN_EXPORT_INTERFACE
    uint8_t const* mData{nullptr};
    uint32_t const mSize{0};
    uint32_t const mSharedMemSize{0};
    char const* mFunctionName{nullptr};
    uint32_t const mNumThreadsPerCTA{0};
#else
    trtllm::gen::CudaRunner* mCudaRunner{nullptr};
#endif

    GemmOptions mOptions{};
    SmVersion mSm{SmVersion::Sm100a};
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

inline std::string dumpOptions(GemmOptions const& options)
{
    std::stringstream ss;
    ss << "mAllReduceAlgo="
       << "gemm::AllReduceAlgo(" << static_cast<int32_t>(options.mAllReduceAlgo) << ")"
       << "," << std::endl;
    ss << "mClusterDimX=" << options.mClusterDimX << "," << std::endl;
    ss << "mClusterDimY=" << options.mClusterDimY << "," << std::endl;
    ss << "mClusterDimZ=" << options.mClusterDimZ << "," << std::endl;
    ss << "mDtypeAcc="
       << "trtllm::gen::Dtype(" << static_cast<int32_t>(options.mDtypeAcc) << ")"
       << "," << std::endl;
    ss << "mDtypeElt="
       << "trtllm::gen::Dtype(" << static_cast<int32_t>(options.mDtypeElt) << ")"
       << "," << std::endl;
    ss << "mDtypeC="
       << "trtllm::gen::Dtype(" << static_cast<int32_t>(options.mDtypeC) << ")"
       << "," << std::endl;
    ss << "mEnablesEarlyExit=" << options.mEnablesEarlyExit << "," << std::endl;
    ss << "mEnablesDelayedEarlyExit=" << options.mEnablesDelayedEarlyExit << "," << std::endl;
    ss << "mEnablesGlobalPtxKnobs=" << options.mEnablesGlobalPtxKnobs << "," << std::endl;
    ss << "mEpilogueTileM=" << options.mEpilogueTileM << "," << std::endl;
    ss << "mEpilogueTileN=" << options.mEpilogueTileN << "," << std::endl;
    ss << "mGridTriggerSecondaryA=" << options.mGridTriggerSecondaryA << "," << std::endl;
    ss << "mGridTriggerSecondaryB=" << options.mGridTriggerSecondaryB << "," << std::endl;
    ss << "mGridWaitForPrimaryEarlyExit=" << options.mGridWaitForPrimaryEarlyExit << "," << std::endl;
    ss << "mGridWaitForPrimaryA=" << options.mGridWaitForPrimaryA << "," << std::endl;
    ss << "mGridWaitForPrimaryB=" << options.mGridWaitForPrimaryB << "," << std::endl;
    ss << "mHoistMmaTaskTryWaits=" << options.mHoistMmaTaskTryWaits << "," << std::endl;
    ss << "mK=" << options.mK << "," << std::endl;
    ss << "mKernelTraits={}"
       << "," << std::endl;
    ss << "mM=" << options.mM << "," << std::endl;
    ss << "mMmaK=" << options.mMmaK << "," << std::endl;
    ss << "mMmaM=" << options.mMmaM << "," << std::endl;
    ss << "mMmaN=" << options.mMmaN << "," << std::endl;
    ss << "mMockAllReduce=" << options.mMockAllReduce << "," << std::endl;
    ss << "mN=" << options.mN << "," << std::endl;
    ss << "mNumSlicesForSplitK=" << options.mNumSlicesForSplitK << "," << std::endl;
    ss << "mNumSlicesForSliceK=" << options.mNumSlicesForSliceK << "," << std::endl;
    ss << "mNumStages=" << options.mNumStages << "," << std::endl;
    ss << "mNumStagesMma=" << options.mNumStagesMma << "," << std::endl;
    ss << "mNumStagesMmaWithinWorkTile=" << options.mNumStagesMmaWithinWorkTile << "," << std::endl;
    ss << "mNumStagesMmaAcrossWorkTile=" << options.mNumStagesMmaAcrossWorkTile << "," << std::endl;
    ss << "mNumStagesWorkId=" << options.mNumStagesWorkId << "," << std::endl;
    ss << "mOutputDebugTensors=" << options.mOutputDebugTensors << "," << std::endl;
    ss << "mUseShuffledMatrixA=" << options.mUseShuffledMatrixA << "," << std::endl;
    ss << "mSliceK=" << options.mSliceK << "," << std::endl;
    ss << "mSplitK="
       << "gemm::SplitK(" << static_cast<int32_t>(options.mSplitK) << ")"
       << "," << std::endl;
    ss << "mTransposeMmaOutput=" << options.mTransposeMmaOutput << "," << std::endl;
    ss << "mTileM=" << options.mTileM << "," << std::endl;
    ss << "mTileN=" << options.mTileN << "," << std::endl;
    ss << "mTileK=" << options.mTileK << "," << std::endl;
    ss << "mUseUnrollLoop2xForMma=" << options.mUseUnrollLoop2xForMma << "," << std::endl;
    ss << "mUseCustomMmaSchedule=" << options.mUseCustomMmaSchedule << "," << std::endl;
    ss << "mUseHoistTryWaitForCustomMmaSchedule=" << options.mUseHoistTryWaitForCustomMmaSchedule << "," << std::endl;
    ss << "mUseDeepSeekFp8=" << options.mUseDeepSeekFp8 << "," << std::endl;
    ss << "mUsePerTokenSfA=" << options.mUsePerTokenSfA << "," << std::endl;
    ss << "mUsePerTokenSfB=" << options.mUsePerTokenSfB << "," << std::endl;
    ss << "mUseTmaStore=" << options.mUseTmaStore << "," << std::endl;
    ss << "mUseTwoTmaLoadWarps=" << options.mUseTwoTmaLoadWarps << "," << std::endl;
    ss << "mUseTwoMmaWarps=" << options.mUseTwoMmaWarps << "," << std::endl;
    ss << "mSfLayoutA="
       << "trtllm::gen::SfLayout(" << static_cast<int32_t>(options.mSfLayoutA) << ")"
       << "," << std::endl;
    ss << "mSfLayoutB="
       << "trtllm::gen::SfLayout(" << static_cast<int32_t>(options.mSfLayoutB) << ")"
       << "," << std::endl;
    ss << "mSfLayoutC="
       << "trtllm::gen::SfLayout(" << static_cast<int32_t>(options.mSfLayoutC) << ")"
       << "," << std::endl;
    ss << "mTileScheduler="
       << "gemm::TileScheduler(" << static_cast<int32_t>(options.mTileScheduler) << ")" << std::endl;
    return ss.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline T divUp(T a, T b)
{
    return (a + b - 1) / b;
}

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

// Check if the options are valid or not.
inline bool checkAndUpdateGemmOptions(
    GemmOptions& options, bool isBlackwell, int /* tpGrpSize */, bool updateOptions = true)
{
    if (options.mDtypeElt == tg::Dtype::E4m3 && options.mMmaK != 32)
    {
        TLLM_LOG_WARNING(
            "Unsupported MmaK (", options.mMmaK, ") for ", gemm::toString(options.mDtypeElt), ". Setting MmaK to 32");
        if (updateOptions)
        {
            options.mMmaK = 32;
            options.mTileK = std::max(options.mMmaK, options.mTileK);
        }
        else
        {
            return false;
        }
    }

    // Constraints for NvFp4 and MxFp8.
    if ((options.mDtypeElt == tg::Dtype::E2m1 || options.mDtypeElt == tg::Dtype::MxE4m3
            || options.mDtypeC == tg::Dtype::MxE4m3)
        && options.mMmaM != 128)
    {
        // MMA M must be 128 when the input uses block scaling, or when the output is an Mx format.
        int newTileM = 128 * divUp(options.mTileM, 128);
        TLLM_LOG_WARNING("Unsupported MmaM (", options.mMmaM, ") for dtypeElt=", gemm::toString(options.mDtypeElt),
            ", dtypeC=", gemm::toString(options.mDtypeC), ". Setting MmaM to 128 and TileM to ", newTileM);
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
    if (options.mDtypeElt == tg::Dtype::E2m1 || options.mDtypeElt == tg::Dtype::MxE4m3)
    {
        TLLM_CHECK_ERROR(isBlackwell, "Block scaling is only supported on Blackwell");

        TLLM_CHECK_ERROR(options.mSfLayoutB == tg::SfLayout::R128c4 || options.mSfLayoutB == tg::SfLayout::R8c4,
            "Only the 128x4 and 8x4 SF layouts are supported for B, got ", tg::sfLayoutToString(options.mSfLayoutB));

        int const mmaK = (options.mDtypeElt == tg::Dtype::E2m1) ? 64 : 32;
        if (options.mMmaK != mmaK)
        {
            int newTileK = mmaK * divUp(options.mTileK, mmaK);
            TLLM_LOG_WARNING("Unsupported MmaK (", options.mMmaK, ") for ", gemm::toString(options.mDtypeElt),
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

        // TileN must be a multiple of the number of rows per SF tile.
        int const numSfTileRowsB = options.mSfLayoutB == tg::SfLayout::R128c4 ? 128 : 8;
        TLLM_CHECK_ERROR(options.mTileN % numSfTileRowsB == 0, "TileN (", options.mTileN, ") must be a multiple of ",
            numSfTileRowsB, " for B SF layout ", tg::sfLayoutToString(options.mSfLayoutB));
        // The MMA N may only be smaller than 64 if it is equal to the tile N.
        TLLM_CHECK_ERROR(options.mMmaN >= 64 || options.mMmaN == options.mTileN, "MmaN (", options.mMmaN,
            ") must be >= 64 or equal to TileN (", options.mTileN, ") for ", gemm::toString(options.mDtypeElt));

        int numEltsPerSf = tg::dtypeNumEltsPerSf(options.mDtypeElt);
        TLLM_CHECK_ERROR(options.mTileK % (4 * numEltsPerSf) == 0, "TileK (", options.mTileK,
            ") must be a multiple of ", (4 * numEltsPerSf), " for type ", gemm::toString(options.mDtypeElt));
    }
    if (options.mDtypeC == tg::Dtype::E2m1 || options.mDtypeC == tg::Dtype::MxE4m3)
    {
        TLLM_CHECK_ERROR(isBlackwell, "Block scaling is only supported on Blackwell");

        TLLM_CHECK_ERROR(options.mSfLayoutC == tg::SfLayout::R128c4 || options.mSfLayoutC == tg::SfLayout::R8c4,
            "Only the 128x4 and 8x4 SF layouts are supported for C.");
        int const numSfTileRowsC = options.mSfLayoutC == tg::SfLayout::R128c4 ? 128 : 8;
        TLLM_CHECK_ERROR(options.mTileN % numSfTileRowsC == 0, "TileN (", options.mTileN, ") must be a multiple of ",
            numSfTileRowsC, " for C SF layout ", tg::sfLayoutToString(options.mSfLayoutC));

        int const hiddenDim = options.mTransposeMmaOutput ? options.mM : options.mN;
        int const hiddenGranularity = 4 * tg::dtypeNumEltsPerSf(options.mDtypeC);
        TLLM_CHECK_ERROR(hiddenDim % hiddenGranularity == 0, "Hidden dim (", hiddenDim, ") must be a multiple of ",
            hiddenGranularity, " for block-scaled outputs.");
        TLLM_CHECK_ERROR(!options.mTransposeMmaOutput || options.mUseShuffledMatrixA,
            "Transposing block-scaled outputs requires shuffled A.");
    }

    // If dtypeC is unspecified (Dtype::Void), assign to the input dtype.
    if (options.mDtypeC == tg::Dtype::Void)
    {
        TLLM_LOG_INFO("Setting dtypeC to ", tg::dtypeToString(options.mDtypeElt));
        if (updateOptions)
        {
            options.mDtypeC = options.mDtypeElt;
        }
        else
        {
            return false;
        }
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
    TLLM_CHECK_ERROR(options.mK % options.mNumSlicesForSplitK == 0, "K must be divisible by NumSlicesForSplitK.");
    TLLM_CHECK_ERROR((options.mK / options.mNumSlicesForSplitK) % options.mTileK == 0,
        "K / NumSlicesForSplitK must be divisible by TileK. Found TileK=", options.mTileK, " and K=", options.mK,
        " and NumSlicesForSplitK=", options.mNumSlicesForSplitK);

    if (options.mUseShuffledMatrixA)
    {
        auto const shuffleBlockSize = getShuffleBlockSize(options.mEpilogueTileM);
        TLLM_CHECK_ERROR(options.mM % shuffleBlockSize == 0, "M must be a multiple of shuffle block size (",
            shuffleBlockSize, ") when useShuffledMatrixA");
    }

    TLLM_CHECK_ERROR(options.mMmaM <= options.mEpilogueTileM && options.mMmaN <= options.mEpilogueTileN,
        "EpilogueTileM and EpilogueTileN must be larger or equal than the respective atom sizes.");
    TLLM_CHECK_ERROR(options.mTileM % options.mEpilogueTileM == 0 && options.mTileN % options.mEpilogueTileN == 0,
        "TileM and TileN must be divisible by EpilogueTileM and EpilogueTileN respectively.");
    TLLM_CHECK_ERROR(
        options.mClusterDimX == 1 && options.mClusterDimY == 1, "GEMM does not support cluster in X and Y dimensions.");
    TLLM_CHECK_ERROR(
        options.mClusterDimZ == 1 || options.mNumSlicesForSplitK > 1, "Cluster DimZ is only allowed for split-k.");
    TLLM_CHECK_ERROR(options.mTileM <= 128, "GEMM does not support TileM > 128.");

    // When the A-matrix is shuffled, the output must be transposed.
    if (options.mUseShuffledMatrixA)
    {
        // TODO add matrix shuffle for N-major epilogue.
        TLLM_CHECK_ERROR(options.mTransposeMmaOutput,
            "Shuffled matrix A is only supported with M-major epilogue. Set -transposeMmaOutput");
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
    }
    if (options.mUseDeepSeekFp8)
    {
        TLLM_CHECK_ERROR(options.mDtypeElt == tg::Dtype::E4m3, "A and B dtype must be E4m3 for DeepSeek Fp8. Found ",
            tg::dtypeToString(options.mDtypeElt));

        TLLM_CHECK_ERROR(isBlackwell, "DeepSeek Fp8 is not supported for Hopper");
        TLLM_CHECK_ERROR(options.mAllReduceAlgo == AllReduceAlgo::None, "DeepSeek Fp8 does not support AllReduce");

        // Check that TileK = 128 for correct scaling of every 128 channels.
        TLLM_CHECK_ERROR(options.mTileK == 128, "Tile-K must be equal to 128 for DeepSeek Fp8");
        // Tile sizes of the output hidden dimension.
        auto hiddenDim = options.mTransposeMmaOutput ? options.mM : options.mN;
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

        // Make sure the GEMM-M/N dimension is a multiple of 128 when using DeepSeek FP8.
        TLLM_CHECK_ERROR(hiddenDim % 128 == 0, "GEMM-", hiddenDimName,
            " must be a multiple of 128 when using DeepSeek Fp8. Found ", hiddenDim);
        // Make sure the GEMM-K dimension is a multiple of 128 when using DeepSeek FP8.
        TLLM_CHECK_ERROR(
            options.mK % 128 == 0, "GEMM-K must be a multiple of 128 when using DeepSeek Fp8. Found ", options.mK);

        // Check that the output tile N can be processed with the epilogue tile granularity.
        TLLM_CHECK_ERROR((hiddenDimPerOutputTile / 2) % hiddenDimPerEpilogueTile == 0, "DeepSeek Fp8 requires Tile",
            hiddenDimName, " / 2 (", hiddenDimPerOutputTile / 2, ") being a multiple of EpilogueTile", hiddenDimName,
            " (", hiddenDimPerEpilogueTile, ")");
        // Check that the output tile N can be processed with the epilogue tile granularity.
        TLLM_CHECK_ERROR((hiddenDimPerOutputTile / 2) % hiddenDimPerMma == 0, "DeepSeek Fp8 requires Tile",
            hiddenDimName, " / 2 (", hiddenDimPerOutputTile / 2, ") being a multiple of mma", hiddenDimName, " (",
            hiddenDimPerMma, ")");
    }

    if (options.mSliceK)
    {
        TLLM_CHECK_ERROR(isBlackwell, "Slice-K is not supported on Hopper");

        TLLM_CHECK_ERROR(!options.mUseDeepSeekFp8, "DeepSeek Fp8 GEMM is not supported for slice-K");
        TLLM_CHECK_ERROR(options.mUseTwoTmaLoadWarps, "Slice-K requires two warp load for A and B");
        TLLM_CHECK_ERROR(options.mTransposeMmaOutput, "Slice-K requires transpose mma output");
        TLLM_CHECK_ERROR(options.mUseShuffledMatrixA, "Slice-K requires shuffled matrix A");
        TLLM_CHECK_ERROR(options.mTileK % 128 == 0, "Slice-K requires TileK be a multiple of 128");
        TLLM_CHECK_ERROR(options.mMmaM == 128, "Slice-K requires MmaM == 128");
        TLLM_CHECK_ERROR(options.mTileN == options.mEpilogueTileN, "TileN must be equal to EpilogueTileN for slice-K");

        TLLM_LOG_WARNING("Overwriting TileM and EpilogueTileM to 32 for slice-K");
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
        TLLM_CHECK_ERROR(options.mDtypeElt == tg::Dtype::E4m3, "Slice-K requires e4m3 input dtype");

        if (updateOptions)
        {
            options.mNumSlicesForSliceK = 4;
        }
        else
        {
            return false;
        }
        TLLM_CHECK_ERROR((options.mTileK / options.mMmaK) % options.mNumSlicesForSliceK == 0, "TileK (", options.mTileK,
            ") / MmaK (", options.mMmaK, ") must be a multiple of mNumSlicesForSliceK (", options.mNumSlicesForSliceK,
            ")");
    }

    if (options.mUseUnrollLoop2xForMma)
    {
        bool notSupported = (options.mK / options.mNumSlicesForSplitK) % (options.mTileK * 2) != 0;
        // Check that the 2*TileK is a multiple of MmaK when UnrollLoop2x is enabled.
        // This is to avoid deadlock when mma runs even-numbered loop while the other warps run
        // odd-numbered loop.
        if (notSupported)
        {
            TLLM_LOG_WARNING("Size K / splitK must be a multiple of TileK * 2. Found TileK=", options.mTileK,
                " and K=", options.mK, " and numSlicesForSplitK=", options.mNumSlicesForSplitK,
                ". Disabling unrollLoop2xForMma.");
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

    if (!isBlackwell && options.mTileScheduler == TileScheduler::Persistent)
    {
        // TODO(anchengc): will be supported in upcoming MRs.
        TLLM_LOG_WARNING("Persistent scheduling is not supported on Hopper. Using Static scheduling.");
        if (updateOptions)
        {
            options.mTileScheduler = TileScheduler::Static;
        }
        else
        {
            return false;
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
    // Kernel 3:  Warp 0:           ---- (!) Output of 1,2 is not yet visible -----------------------
    //            Warp 1:           ---- (!) We normally assume that 1 is visible is not yet visible-
    //            Warp 2:           -------------------ACQBULK-- Kernel 1,2 output visible ----------
    TLLM_CHECK_ERROR((options.mGridWaitForPrimaryA || !options.mGridTriggerSecondaryA),
        "A: If a task triggers a secondary kernel, it must also wait for primary kernel.");
    TLLM_CHECK_ERROR((options.mGridWaitForPrimaryB || !options.mGridTriggerSecondaryB),
        "B: If a task triggers a secondary kernel, it must also wait for primary kernel.");

    if (updateOptions)
    {
        // Init kernel traits.
        options.mKernelTraits = KernelTraits(options.mDtypeElt, options.mDtypeC, options.mDtypeAcc, options.mTileM,
            options.mTileN, options.mTileK, options.mEpilogueTileM, options.mEpilogueTileN, options.mNumStages,
            options.mNumStagesMma, options.mNumSlicesForSplitK, options.mNumSlicesForSliceK, options.mSplitK,
            options.mUseTmaStore, options.mTransposeMmaOutput, options.mAllReduceAlgo,
            options.mTileScheduler == TileScheduler::Persistent, options.mUseDeepSeekFp8, options.mUsePerTokenSfA,
            options.mUsePerTokenSfB);
    }

    return true;
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
