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

#include "BatchedGemmEnums.h"
#include "GemmGatedActOptions.h"
#include "GemmOptions.h"

#include <vector>

#ifndef TLLM_GEN_EXPORT_INTERFACE
#include "trtllm/gen/CudaRunner.h"
#include "trtllm/gen/GenCtx.h"
#else
#include <iostream>

#define TLLM_CHECK_ERROR(cond, ...)                                                                                    \
    if (!(cond))                                                                                                       \
    {                                                                                                                  \
        printArgs(__VA_ARGS__);                                                                                        \
        return false;                                                                                                  \
    }

#define TLLM_LOG_ERROR(...) TLLM_CHECK_ERROR(false, __VA_ARGS__)

#define TLLM_CHECK_ERROR_FMT(...) TLLM_CHECK_ERROR(false, __VA_ARGS__)

#define TLLM_CHECK_WARNING(cond, ...)                                                                                  \
    if (!(cond))                                                                                                       \
    {                                                                                                                  \
        printArgs(__VA_ARGS__);                                                                                        \
        return false;                                                                                                  \
    }

#define TLLM_LOG_WARNING(...) TLLM_CHECK_WARNING(false, __VA_ARGS__)

#define TLLM_LOG_INFO(...) TLLM_CHECK_WARNING(false, __VA_ARGS__)

#endif

namespace batchedGemm
{

namespace batchedGemm
{

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

// We do not differentiate between BatchedGemmOptions and BatchedGemmGatedActOptions for simplicity.
// We inherit from GemmGatedActOptions, which is inherited from
// GemmOptions to get GemmOptions and GemmGatedActOptions at the same time.
struct BatchedGemmOptions : public gemmGatedAct::GemmGatedActOptions
{

    // Dtor. Allow down-casting.
    virtual ~BatchedGemmOptions() = default;

    enum class BatchMode
    {
        BatchM,
        BatchN
    };

    BatchedGemmOptions() = default;

    // FIXME We create explicit constructor with all options to WAR stubgen issue in TRT-LLM.
    BatchedGemmOptions(gemm::AllReduceAlgo allReduceAlgo, gemm::BiasType biasType, int blockK, int clusterDimX,
        int clusterDimY, int clusterDimZ, tg::Dtype dtypeAcc, tg::Dtype dtypeA, tg::Dtype dtypeB, tg::Dtype dtypeC,
        tg::Dtype dtypeMmaA, tg::Dtype dtypeMmaB, bool enablesEarlyExit, bool enablesDelayedEarlyExit,
        bool enablesGlobalPtxKnobs, int epilogueLdtmDps, int epilogueLdtmBits, int epilogueTileM, int epilogueTileN,
        bool gridTriggerSecondaryA, bool gridTriggerSecondaryB, bool gridWaitForPrimaryEarlyExit,
        bool gridWaitForPrimaryA, bool gridWaitForPrimaryB, bool hoistLoadTaskInit, bool hoistMmaTaskTryWaits, int k,
        gemm::KernelTraits kernelTraits, gemm::MatrixLayout layoutA, gemm::MatrixLayout layoutB, int m, int mmaK,
        tg::MmaKind mmaKind, int mmaM, int mmaN, bool mockAllReduce, int n, int numSlicesForSplitK,
        int numSlicesForSliceK, int numStages, int numStagesMma, int numStagesMmaWithinWorkTile,
        int numStagesMmaAcrossWorkTile, int numStagesWorkId, bool outputDebugTensors, bool patchF2fp,
        bool useShuffledMatrixA, bool sliceK, gemm::SplitK splitK, bool transposeMmaOutput, int tileM, int tileN,
        int tileK, bool useUnrollLoop2xForMma, bool useCustomMmaSchedule, bool useHoistTryWaitForCustomMmaSchedule,
        bool useDeepSeekFp8, bool usePerTokenSfA, bool usePerTokenSfB, bool useTmaStore, bool useTwoTmaLoadWarps,
        bool useTwoMmaWarps, tg::SfLayout sfLayoutA, tg::SfLayout sfLayoutB, tg::SfLayout sfLayoutC,
        int32_t sfReshapeFactor, gemm::TileScheduler tileScheduler, gemmGatedAct::ActType actType,
        std::vector<int> batchedM, std::vector<int> batchedN, BatchMode batchMode, int numBatches, bool isStaticBatch,
        int numTokens, RouteImpl routeImpl, bool gridWaitForPrimaryRouting, bool fusedAct,
        int numRegsPerThreadNonEpilogueWarp, int numRegsPerThreadEpilogueWarp, int numRegsCastAWarps)
        : gemmGatedAct::GemmGatedActOptions(
            gemm::GemmOptions(allReduceAlgo, biasType, blockK, clusterDimX, clusterDimY, clusterDimZ, dtypeAcc, dtypeA,
                dtypeB, dtypeC, dtypeMmaA, dtypeMmaB, enablesEarlyExit, enablesDelayedEarlyExit, enablesGlobalPtxKnobs,
                epilogueLdtmDps, epilogueLdtmBits, epilogueTileM, epilogueTileN, gridTriggerSecondaryA,
                gridTriggerSecondaryB, gridWaitForPrimaryEarlyExit, gridWaitForPrimaryA, gridWaitForPrimaryB,
                hoistLoadTaskInit, hoistMmaTaskTryWaits, k, kernelTraits, layoutA, layoutB, m, mmaK, mmaKind, mmaM,
                mmaN, mockAllReduce, n, numSlicesForSplitK, numSlicesForSliceK, numStages, numStagesMma,
                numStagesMmaWithinWorkTile, numStagesMmaAcrossWorkTile, numStagesWorkId, outputDebugTensors, patchF2fp,
                useShuffledMatrixA, sliceK, splitK, transposeMmaOutput, tileM, tileN, tileK, useUnrollLoop2xForMma,
                useCustomMmaSchedule, useHoistTryWaitForCustomMmaSchedule, useDeepSeekFp8, usePerTokenSfA,
                usePerTokenSfB, useTmaStore, useTwoTmaLoadWarps, useTwoMmaWarps, sfLayoutA, sfLayoutB, sfLayoutC,
                sfReshapeFactor, tileScheduler),
            actType)
        , mBatchedM(batchedM)
        , mBatchedN(batchedN)
        , mBatchMode(BatchMode(batchMode))
        , mNumBatches(numBatches)
        , mIsStaticBatch(isStaticBatch)
        , mNumTokens(numTokens)
        , mRouteImpl(routeImpl)
        , mGridWaitForPrimaryRouting(gridWaitForPrimaryRouting)
        , mFusedAct(fusedAct)
        , mNumRegsPerThreadNonEpilogueWarp(numRegsPerThreadNonEpilogueWarp)
        , mNumRegsPerThreadEpilogueWarp(numRegsPerThreadEpilogueWarp)
        , mNumRegsCastAWarps(numRegsCastAWarps)
    {
    }

    // Batched M-dimensions of GEMM.
    std::vector<int> mBatchedM;
    // Batched N-dimensions of GEMM.
    std::vector<int> mBatchedN;
    // Whether batching M or N.
    BatchMode mBatchMode{BatchMode::BatchM};
    // Number of Gemm batches.
    int mNumBatches;

    // Whether the batch size is static (i.e. known at kernel launch time).
    bool mIsStaticBatch{true};
    // Total number of tokens.
    int mNumTokens{32};
    // Whether load the input tokens and do routing.
    RouteImpl mRouteImpl{RouteImpl::NoRoute};
    // Whether the loads that load from ptrRouteMap, ptrTotalNumPaddedTokens,
    // ptrCtaIdxXyToBatchIdx, etc.. should wait on a grid dependency.
    bool mGridWaitForPrimaryRouting{true};

    // Whether to perform a fused gated activation.
    bool mFusedAct{false};

    // Number of registers per thread for non-epilogue warps
    int mNumRegsPerThreadNonEpilogueWarp{0};
    // Number of registers per thread for epilogue warps
    int mNumRegsPerThreadEpilogueWarp{0};
    // Number of registers for the cast A warps.
    int mNumRegsCastAWarps{0};
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Check if the options are valid or not.
bool checkAndUpdateBatchedGemmOptions(BatchedGemmOptions& options, bool isBlackwell, bool updateOptions = true)
{

    bool isValid = true;
    if (options.mFusedAct)
    {
        // ensure that we check the fused options as well
        isValid = gemmGatedAct::checkAndUpdateGemmGatedActOptions(options, isBlackwell, updateOptions);
    }
    else
    {
        isValid = gemm::checkAndUpdateGemmOptions(options, isBlackwell, 1 /* tpGrpSize */, updateOptions);
    }

    bool batchM = options.mBatchMode == BatchedGemmOptions::BatchMode::BatchM;
    if (updateOptions)
    {
        if (batchM)
        {
            if (options.mBatchedM.empty())
            {
                options.mBatchedM.push_back(128);
                options.mBatchedM.push_back(256);
            }
            options.mNumBatches = options.mBatchedM.size();
        }
        else
        {
            if (options.mBatchedN.empty())
            {
                options.mBatchedN.push_back(128);
                options.mBatchedN.push_back(256);
            }
            options.mNumBatches = options.mBatchedN.size();
        }
    }

    for (int b = 0; b < options.mNumBatches; b++)
    {
        if (batchM)
        {
            TLLM_CHECK_ERROR(options.mN > 0 && options.mK > 0, "N and K must be larger than 0");
            TLLM_CHECK_ERROR(options.mN >= options.mTileN, "N must be equal or larger than TileN.");
            TLLM_CHECK_ERROR(options.mN % options.mTileN == 0, "N must be divisible by TileN.");
            TLLM_CHECK_ERROR(!options.mTransposeMmaOutput, "When batchM the MMA output has to be in row-major.");
        }
        else
        {
            TLLM_CHECK_ERROR(options.mM > 0 && options.mK > 0, "M and K must be larger than 0");
            TLLM_CHECK_ERROR(options.mM >= options.mTileM, "N must be equal or larger than tileN.");
            TLLM_CHECK_ERROR(options.mM % options.mTileM == 0, "M must be divisible by TileM.");
            TLLM_CHECK_ERROR(options.mTransposeMmaOutput, "When batchN the MMA output has to be in column-major.");
        }
    }

    if (options.mUseDeepSeekFp8)
    {
        if (batchM)
        {
            // Make sure the GEMM-K dimension is a multiple of 128 when using DeepSeek FP8.
            TLLM_CHECK_ERROR(
                options.mN % 128 == 0, "GEMM-N must be a multiple of 128 when using DeepSeek Fp8. Found ", options.mN);
        }
        else
        {
            // Make sure the GEMM-K dimension is a multiple of 128 when using DeepSeek FP8.
            TLLM_CHECK_ERROR(
                options.mM % 128 == 0, "GEMM-N must be a multiple of 128 when using DeepSeek Fp8. Found ", options.mN);
        }
        // Make sure the GEMM-K dimension is a multiple of 128 when using DeepSeek FP8.
        TLLM_CHECK_ERROR(
            options.mK % 128 == 0, "GEMM-K must be a multiple of 128 when using DeepSeek Fp8. Found ", options.mK);

        TLLM_CHECK_ERROR(options.mDtypeC != tg::Dtype::E2m1 && options.mDtypeA == tg::Dtype::E4m3
                && options.mDtypeB == tg::Dtype::E4m3,
            "E2m1 is not supported with DeepSeek FP8");
    }

    if (batchM)
    {
        if (options.mDtypeA == tg::Dtype::MxE2m1 && options.mMmaKind == tg::MmaKind::MxFp8Fp6Fp4)
        {
            TLLM_CHECK_ERROR(doesRouteImplUseNoRoute(options.mRouteImpl),
                "RouteAct is not supported with dtypeA = MxE2m1 and MxFp8Fp6Fp4.");
        }
    }
    else
    {
        if (options.mDtypeB == tg::Dtype::MxE2m1 && options.mMmaKind == tg::MmaKind::MxFp8Fp6Fp4)
        {
            TLLM_CHECK_ERROR(doesRouteImplUseNoRoute(options.mRouteImpl),
                "RouteAct is not supported with dtypeB = MxE2m1 and MxFp8Fp6Fp4.");
        }
    }

    TLLM_CHECK_ERROR(options.mUseTmaStore, "Only TMA store is supported.");
    if (batchM)
    {
        TLLM_CHECK_ERROR(options.mLayoutA == gemm::MatrixLayout::MajorK, "Activations must be in k-major format");
    }
    else
    {
        TLLM_CHECK_ERROR(options.mLayoutB == gemm::MatrixLayout::MajorK, "Activations must be in k-major format");
    }

    if (tg::mmaKindIsBlockFmt(options.mMmaKind) && !options.mUseDeepSeekFp8)
    {
        if (!doesRouteImplUseNoRoute(options.mRouteImpl))
        {
            if (batchM)
            {
                TLLM_CHECK_ERROR(
                    options.mSfLayoutA == tg::SfLayout::Linear, "Tokens need use SF linear layout when being routed");
            }
            else
            {
                // Note: if B is cast from a non-block format to a block format, there are no SFs to load.
                TLLM_CHECK_ERROR(options.mSfLayoutB == tg::SfLayout::Linear || !tg::dtypeIsBlockFmt(options.mDtypeB),
                    "Tokens need use SF linear layout when being routed");
            }
        }

        if (doesRouteImplUseTma(options.mRouteImpl))
        {
            TLLM_CHECK_ERROR(!batchM, "UTMALDG.GATHER4 only supported for batch N.");

            if (tg::mmaKindIsBlockFmt(options.mMmaKind))
            {
                auto dtypeRoute = batchM ? options.mDtypeA : options.mDtypeB;
                TLLM_CHECK_ERROR(options.mTileK % tg::dtypeNumEltsPerSf(dtypeRoute) == 0,
                    "tileK needs to be a multiple of 16 * tg::dtypeNumEltsPerSf(dtypeA).");
                TLLM_CHECK_ERROR(options.mTileK % (tg::dtypeNumEltsPerSf(dtypeRoute) * 16) == 0,
                    "tileK needs to be a multiple of 16 * tg::dtypeNumEltsPerSf(dtypeA).");
            }
        }

        if (!batchM || doesRouteImplUseNoRoute(options.mRouteImpl))
        {
            TLLM_CHECK_ERROR(options.mSfLayoutA == tg::SfLayout::R128c4,
                "options.mSfLayoutA has to be tg::SfLayout::R128c4 when not being routed");
        }
    }

    if (!gemm::isBiasTypeNone(options.mBiasType))
    {
        TLLM_CHECK_ERROR(
            (gemm::isBiasTypeN(options.mBiasType) && options.mBatchMode == BatchedGemmOptions::BatchMode::BatchM)
                || (gemm::isBiasTypeM(options.mBiasType)
                    && options.mBatchMode == BatchedGemmOptions::BatchMode::BatchN),
            "BatchedGemm supports only per channel bias.");
    }

    // We do not handle the case where K is not a multiple of TileK.
    // TMA based load handles the case transparently.
    if (doesRouteImplUseLdgsts(options.mRouteImpl))
    {
        TLLM_CHECK_ERROR(options.mK % options.mTileK == 0, "K must be a multiple of TileK");
    }

    return isValid;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// BatchedGemmConfig
//
////////////////////////////////////////////////////////////////////////////////////////////////////

struct BatchedGemmConfig
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
    char const* mHash{nullptr};
#else
    trtllm::gen::CudaRunner* mCudaRunner{nullptr};
#endif

    BatchedGemmOptions mOptions;
    gemm::SmVersion mSm{gemm::SmVersion::Sm100a};
};

////////////////////////////////////////////////////////////////////////////////////////////////////

inline std::string dumpOptions(BatchedGemmOptions const& options)
{
    std::stringstream ss;
    ss << gemmGatedAct::dumpOptions(options) << ", ";
    ss << "mBatchedM={}," << std::endl;
    ss << "mBatchedN={}," << std::endl;
    ss << "mBatchMode=batchedGemm::BatchedGemmOptions::BatchMode(" << static_cast<int32_t>(options.mBatchMode) << "),"
       << std::endl;
    ss << "mNumBatches=" << options.mNumBatches << "," << std::endl;
    ss << "mIsStaticBatch=" << options.mIsStaticBatch << "," << std::endl;
    ss << "mNumTokens=" << options.mNumTokens << "," << std::endl;
    ss << "mRouteImpl=batchedGemm::RouteImpl(" << static_cast<int32_t>(options.mRouteImpl) << ")," << std::endl;
    ss << "mGridWaitForPrimaryRouting=" << options.mGridWaitForPrimaryRouting << "," << std::endl;
    ss << "mFusedAct=" << options.mFusedAct << "," << std::endl;
    ss << "mNumRegsPerThreadNonEpilogueWarp=" << options.mNumRegsPerThreadNonEpilogueWarp << "," << std::endl;
    ss << "mNumRegsPerThreadEpilogueWarp=" << options.mNumRegsPerThreadEpilogueWarp << "," << std::endl;
    ss << "mNumRegsCastAWarps=" << options.mNumRegsCastAWarps << std::endl;
    return ss.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace batchedGemm

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef TLLM_GEN_EXPORT_INTERFACE

#undef TLLM_CHECK_ERROR
#undef TLLM_CHECK_ERROR_FMT
#undef TLLM_CHECK_WARNING
#undef TLLM_LOG_WARNING
#undef TLLM_LOG_INFO
#undef TLLM_LOG_ERROR

#endif // TLLM_GEN_EXPORT_INTERFACE

} // namespace batchedGemm
