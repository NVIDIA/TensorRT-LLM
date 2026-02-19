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

#include "BatchedGemmEnums.h"
#include "GemmGatedActOptions.h"
#include "GemmOptions.h"

#include <algorithm>
#include <cstdint>
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

namespace batchedGemm
{

namespace trtllm
{
namespace gen
{
class CudaRunner;
class GenCfg;
} // namespace gen
} // namespace trtllm

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
    BatchedGemmOptions(gemm::AllReduceAlgo allReduceAlgo, gemm::BiasType biasType, int blockK, bool clcFastDrain,
        int clusterDimX, int clusterDimY, int clusterDimZ, gemm::CtaSwizzleType ctaSwizzleType, tg::Dtype dtypeAcc,
        tg::Dtype dtypeA, tg::Dtype dtypeB, tg::Dtype dtypeC, tg::Dtype dtypeMmaA, tg::Dtype dtypeMmaB,
        gemm::EltwiseActType eltwiseActType, bool enablesEarlyExit, bool enablesDelayedEarlyExit,
        bool enablesGlobalPtxKnobs, int epilogueLdtmDps, int epilogueLdtmBits, int epilogueTileM, int epilogueTileN,
        bool fuseUtccpWithUtcmma, bool gridTriggerSecondaryA, bool gridTriggerSecondaryB,
        bool gridWaitForPrimaryEarlyExit, bool gridWaitForPrimaryA, bool gridWaitForPrimaryB, bool hoistLoadTaskInit,
        bool hoistMmaTaskTryWaits, int k, gemm::KernelTraits kernelTraits, gemm::MatrixLayout layoutA,
        gemm::MatrixLayout layoutB, int m, int mmaK, tg::MmaKind mmaKind, int mmaM, int mmaN, bool mockAllReduce, int n,
        int numEpilogueWarps, int numRegsCastAWarps, int numRegsCopySfLdsSttm, int numRegsCopySparsityInfo,
        int numRegsPerThreadEpilogueWarp, int numRegsPerThreadNonEpilogueWarp, int numSlicesForSplitK,
        int numSlicesForSliceK, int numStages, int numStagesMma, int numStagesMmaWithinWorkTile,
        int numStagesMmaAcrossWorkTile, int numStagesWorkId, bool outputDebugTensors, bool patchF2fp,
        int32_t sfBlockSizeA, int32_t sfBlockSizeB, int32_t sfBlockSizeC, tg::SfLayout sfLayoutA,
        tg::SfLayout sfLayoutB, tg::SfLayout sfLayoutC, int32_t sfReshapeFactor, bool sliceK, tg::Sparsity sparsityA,
        gemm::SplitK splitK, int tileK, int tileM, int tileN, gemm::TileScheduler tileScheduler,
        bool transposeMmaOutput, bool useCustomMmaSchedule, bool useDeepSeekFp8,
        bool useHoistTryWaitForCustomMmaSchedule, bool useMaxTmemOverlap, bool usePerTokenSfA, bool usePerTokenSfB,
        bool useShuffledMatrix, bool useTmaStore, bool useTwoTmaLoadWarps, bool useTwoMmaWarps,
        bool useUnrollLoop2xForMma, int validM, int validN, int validK, int worldSize,
        // GemmGatedActOptions
        gemmGatedAct::ActType actType, bool clampBeforeAct,
        // BatchedGemmOptions
        std::vector<int> batchedM, std::vector<int> batchedN, BatchMode batchMode, int32_t batchStrideInTokens,
        bool fusedAct, bool gridWaitForPrimaryRouting, bool isStaticBatch, bool isUniformNumTokensPerBatch,
        int numBatches, int numRegsPerThreadLoadA, int numRegsPerThreadLoadB, int numRegsPerThreadLoadSfA,
        int numRegsPerThreadLoadSfB, int numTokens, int numWarpsLoadA, int numWarpsLoadB, int numWarpsLoadSfA,
        int numWarpsLoadSfB, RouteImpl routeImpl, std::optional<RouteImpl> routeSfsImpl, bool useTmaOobOpt)
        : gemmGatedAct::GemmGatedActOptions(
            gemm::GemmOptions(allReduceAlgo, biasType, blockK, clcFastDrain, clusterDimX, clusterDimY, clusterDimZ,
                ctaSwizzleType, dtypeAcc, dtypeA, dtypeB, dtypeC, dtypeMmaA, dtypeMmaB, eltwiseActType,
                enablesEarlyExit, enablesDelayedEarlyExit, enablesGlobalPtxKnobs, epilogueLdtmDps, epilogueLdtmBits,
                epilogueTileM, epilogueTileN, fuseUtccpWithUtcmma, gridTriggerSecondaryA, gridTriggerSecondaryB,
                gridWaitForPrimaryEarlyExit, gridWaitForPrimaryA, gridWaitForPrimaryB, hoistLoadTaskInit,
                hoistMmaTaskTryWaits, k, kernelTraits, layoutA, layoutB, m, mmaK, mmaKind, mmaM, mmaN, mockAllReduce, n,
                numEpilogueWarps, numRegsCastAWarps, numRegsCopySfLdsSttm, numRegsCopySparsityInfo,
                numRegsPerThreadEpilogueWarp, numRegsPerThreadNonEpilogueWarp, numSlicesForSplitK, numSlicesForSliceK,
                numStages, numStagesMma, numStagesMmaWithinWorkTile, numStagesMmaAcrossWorkTile, numStagesWorkId,
                outputDebugTensors, patchF2fp, sfBlockSizeA, sfBlockSizeB, sfBlockSizeC, sfLayoutA, sfLayoutB,
                sfLayoutC, sfReshapeFactor, sliceK, sparsityA, splitK, tileK, tileM, tileN, tileScheduler,
                transposeMmaOutput, useCustomMmaSchedule, useDeepSeekFp8, useHoistTryWaitForCustomMmaSchedule,
                useMaxTmemOverlap, usePerTokenSfA, usePerTokenSfB, useShuffledMatrix, useTmaStore, useTwoTmaLoadWarps,
                useTwoMmaWarps, useUnrollLoop2xForMma, validM, validN, validK, worldSize),
            actType, clampBeforeAct)
        , mBatchedM(batchedM)
        , mBatchedN(batchedN)
        , mBatchMode(BatchMode(batchMode))
        , mBatchStrideInTokens(batchStrideInTokens)
        , mFusedAct(fusedAct)
        , mGridWaitForPrimaryRouting(gridWaitForPrimaryRouting)
        , mIsStaticBatch(isStaticBatch)
        , mIsUniformNumTokensPerBatch(isUniformNumTokensPerBatch)
        , mNumBatches(numBatches)
        , mNumRegsPerThreadLoadA{numRegsPerThreadLoadA}
        , mNumRegsPerThreadLoadB{numRegsPerThreadLoadB}
        , mNumRegsPerThreadLoadSfA{numRegsPerThreadLoadSfA}
        , mNumRegsPerThreadLoadSfB{numRegsPerThreadLoadSfB}
        , mNumTokens(numTokens)
        , mNumWarpsLoadA{numWarpsLoadA}
        , mNumWarpsLoadB{numWarpsLoadB}
        , mNumWarpsLoadSfA{numWarpsLoadSfA}
        , mNumWarpsLoadSfB{numWarpsLoadSfB}
        , mRouteImpl(routeImpl)
        , mRouteSfsImpl(routeSfsImpl)
        , mUseTmaOobOpt(useTmaOobOpt)
    {
    }

    // Batched M-dimensions of GEMM.
    std::vector<int> mBatchedM;
    // Batched N-dimensions of GEMM.
    std::vector<int> mBatchedN;
    // Whether batching M or N.
    BatchMode mBatchMode{BatchMode::BatchM};
    // Stride between batches in tokens dimension for input matrix.
    int32_t mBatchStrideInTokens{-1};
    // Whether to perform a fused gated activation.
    bool mFusedAct{false};
    // Whether the loads that load from ptrRouteMap, ptrTotalNumPaddedTokens,
    // ptrCtaIdxXyToBatchIdx, etc.. should wait on a grid dependency.
    bool mGridWaitForPrimaryRouting{true};
    // Whether the batch size is static (i.e. known at kernel launch time).
    bool mIsStaticBatch{true};
    // Whether the number of tokens in each entry of the batch is the same.
    bool mIsUniformNumTokensPerBatch{false};
    // Number of Gemm batches.
    int mNumBatches;
    // Number of registers per thread for load A
    int mNumRegsPerThreadLoadA{0};
    // Number of registers per thread for load B
    int mNumRegsPerThreadLoadB{0};
    // Number of registers per thread for load SfA
    int mNumRegsPerThreadLoadSfA{0};
    // Number of registers per thread for load SfB
    int mNumRegsPerThreadLoadSfB{0};
    // Total number of tokens.
    int mNumTokens{32};
    // Number of warps for load A
    int mNumWarpsLoadA{0};
    // Number of warps for load B
    int mNumWarpsLoadB{0};
    // Number of warps for load SfA
    int mNumWarpsLoadSfA{0};
    // Number of warps for load SfB
    int mNumWarpsLoadSfB{0};
    // Whether load the input tokens and do routing.
    RouteImpl mRouteImpl{RouteImpl::NoRoute};
    // Routing logic for scaling factors. If not specified, mRouteImpl is used.
    std::optional<RouteImpl> mRouteSfsImpl{std::nullopt};
    // Whether to use TMA out-of-bounds optimization to reduce wasted traffic. See details in
    // BatchedGemm/KernelParamsDecl.h.
    bool mUseTmaOobOpt{false};
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Check if the options are valid or not.
inline bool checkAndUpdateBatchedGemmOptions(
    BatchedGemmOptions& options, tg::CudaArch cudaArch, bool updateOptions = true)
{
    bool isValid = true;
    if (options.mUseTmaOobOpt && !options.mUseTwoTmaLoadWarps)
    {
        if (updateOptions)
        {
            // Since any routing (mRouteAct != NoRoute) requires mUseTwoTmaLoadWarps == true.
            // Single TMA load warp is not the target use case for OOB optimization.
            options.mUseTmaOobOpt = false;
        }
        else if (!options.mUseTwoTmaLoadWarps)
        {
            TLLM_CHECK_ERROR(false, "TMA OOB optimization requires two TMA load warps.");
            return false;
        }
    }
    if (options.mFusedAct)
    {
        // ensure that we check the fused options as well
        isValid = gemmGatedAct::checkAndUpdateGemmGatedActOptions(options, cudaArch, updateOptions);
    }
    else
    {
        isValid = gemm::checkAndUpdateGemmOptions(options, cudaArch, 1 /* tpGrpSize */, updateOptions);
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

    if (batchM)
    {
        TLLM_CHECK_ERROR(!tg::isSparse(options.mSparsityA), "Sparsity is not supported with batchM.");
        TLLM_CHECK_ERROR(options.mN > 0 && options.mK > 0, "N and K must be larger than 0");
        TLLM_CHECK_ERROR(options.mN >= options.mTileN, "N must be equal or larger than TileN.");
        TLLM_CHECK_ERROR(options.mN % options.mTileN == 0, "N must be divisible by TileN.");
        TLLM_CHECK_ERROR(!options.mTransposeMmaOutput, "When batchM the MMA output has to be in row-major.");
    }
    else
    {
        TLLM_CHECK_ERROR(options.mM > 0 && options.mK > 0, "M and K must be larger than 0");
        TLLM_CHECK_ERROR(options.mM >= options.mTileM, "M must be equal or larger than TileM.");
        TLLM_CHECK_ERROR(options.mM % options.mTileM == 0, "M must be divisible by TileM.");
        TLLM_CHECK_ERROR(options.mTransposeMmaOutput, "When batchN the MMA output has to be in column-major.");
    }

    if (options.mUseDeepSeekFp8)
    {
        if (batchM)
        {
            // Make sure the GEMM-K dimension is a multiple of 128 when using DeepSeek FP8.
            TLLM_CHECK_ERROR(options.mN % 128 == 0 && options.mValidN % 128 == 0,
                "GEMM-N and validN must be a multiple of 128 when using DeepSeek Fp8. Found ", options.mN,
                " and validN=", options.mValidN);
        }
        else
        {
            // Make sure the GEMM-K dimension is a multiple of 128 when using DeepSeek FP8.
            TLLM_CHECK_ERROR(options.mM % 128 == 0 && options.mValidM % 128 == 0,
                "GEMM-M and validM must be a multiple of 128 when using DeepSeek Fp8. Found ", options.mM,
                " and validM=", options.mValidM);
        }
        // Make sure the GEMM-K dimension is a multiple of 128 when using DeepSeek FP8.
        TLLM_CHECK_ERROR(options.mK % 128 == 0 && options.mValidK % 128 == 0,
            "GEMM-K and validK must be a multiple of 128 when using DeepSeek Fp8. Found ", options.mK,
            " and validK=", options.mValidK);

        TLLM_CHECK_ERROR(options.mDtypeC != tg::Dtype::E2m1 && options.mDtypeA == tg::Dtype::E4m3
                && options.mDtypeB == tg::Dtype::E4m3,
            "E2m1 is not supported with DeepSeek FP8");
    }

    if (options.mRouteSfsImpl.has_value() && options.mRouteSfsImpl.value() != options.mRouteImpl)
    {
        TLLM_CHECK_ERROR((options.mRouteSfsImpl.value() == RouteImpl::Ldgsts
                             || options.mRouteSfsImpl.value() == RouteImpl::LdgPlusSts)
                && options.mRouteImpl == RouteImpl::Tma,
            "RouteSfsImpl must be equal to RouteImpl, or Ldgsts/LdgPlusSts, when RouteImpl is Tma");
    }
    else if (!options.mRouteSfsImpl.has_value())
    {
        if (updateOptions)
        {
            options.mRouteSfsImpl = options.mRouteImpl;
        }
        else
        {
            TLLM_LOG_ERROR("RouteSfsImpl must be specified");
            return false;
        }
    }

    TLLM_CHECK_ERROR(options.mRouteImpl != RouteImpl::LdgPlusSts, "LdgPlusSts does not support routing the tokens");

    if (options.mRouteSfsImpl.has_value() && options.mRouteSfsImpl.value() == RouteImpl::LdgPlusSts)
    {
        TLLM_CHECK_ERROR(
            options.mTileK <= 512 && options.mTileK >= 128, "LdgPlusSts only supports 128 <= tileK <= 512");
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

        if (doesRouteImplUseTma(options.mRouteSfsImpl.value()))
        {
            TLLM_CHECK_ERROR(!batchM, "UTMALDG.GATHER4 only supported for batch N.");

            if (tg::mmaKindIsBlockFmt(options.mMmaKind))
            {
                int const numEltsPerSfRoute = batchM ? options.mSfBlockSizeA : options.mSfBlockSizeB;
                TLLM_CHECK_ERROR(options.mTileK % (numEltsPerSfRoute * 16) == 0,
                    "tileK needs to be a multiple of 16 * numEltsPerSf (", numEltsPerSfRoute,
                    ") = ", numEltsPerSfRoute * 16);
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
    if (doesRouteImplUseLdgsts(options.mRouteImpl) && doesRouteImplUseLdgPlusSts(options.mRouteSfsImpl.value()))
    {
        TLLM_CHECK_ERROR(
            options.mK % options.mTileK == 0, "K must be a multiple of TileK when using Ldg based routing");
    }

    if (options.mRouteSfsImpl.has_value()
        && (doesRouteImplUseLdgsts(options.mRouteSfsImpl.value())
            || doesRouteImplUseLdgPlusSts(options.mRouteSfsImpl.value())))
    {
        TLLM_CHECK_ERROR(
            options.mK % options.mTileK == 0, "K must be a multiple of tileK when using Ldg based SF routing");
    }

    if (options.mClusterDimX > 1 && batchM && options.mRouteSfsImpl.has_value())
    {
        TLLM_CHECK_ERROR(options.mRouteSfsImpl.value() != RouteImpl::Tma,
            "2CTA BatchedGemm does not support routing Sf along M dimension with TMA.");
    }

    // Check if all elements in mBatchedM or mBatchedN are the same (uniform tokens per batch) and
    // set mIsUniformNumTokensPerBatch and mBatchStride.
    if (options.mIsUniformNumTokensPerBatch)
    {
        int32_t firstValue = 0;
        bool isUniformNumTokensPerBatch = false;
        if (batchM && !options.mBatchedM.empty())
        {
            firstValue = options.mBatchedM[0];
            isUniformNumTokensPerBatch = std::all_of(options.mBatchedM.begin(), options.mBatchedM.end(),
                [firstValue](int32_t v) { return v == firstValue; });
        }
        else if (!batchM && !options.mBatchedN.empty())
        {
            firstValue = options.mBatchedN[0];
            isUniformNumTokensPerBatch = std::all_of(options.mBatchedN.begin(), options.mBatchedN.end(),
                [firstValue](int32_t v) { return v == firstValue; });
        }
        else
        {
            TLLM_CHECK_ERROR(false, "mBatchedM or mBatchedN must be specified when using uniform tokens per batch.");
        }
        auto tileTokensDim = batchM ? options.mTileM : options.mTileN;
        TLLM_CHECK_ERROR(isUniformNumTokensPerBatch,
            "All elements in mBatchedM or mBatchedN must be the same when using uniform "
            "tokens per batch.");
        TLLM_CHECK_ERROR(options.mBatchStrideInTokens >= 0,
            "Batch stride in tokens must be greater or equal to 0 when using uniform "
            "tokens per batch.");
        TLLM_CHECK_ERROR_FMT(options.mBatchStrideInTokens == 0
                || options.mBatchStrideInTokens == gemm::divUpMul(firstValue, tileTokensDim),
            "Batch stride in tokens must be a 0 or a multiple of %s {%d} when using "
            "uniform tokens per batch.",
            batchM ? "TileM" : "TileN", tileTokensDim);
        TLLM_CHECK_ERROR(
            !options.mUseDeepSeekFp8, "Uniform number of tokens per batch is not supported when using DeepSeek Fp8.");
        TLLM_CHECK_ERROR(!options.mUsePerTokenSfA && !options.mUsePerTokenSfB,
            "Uniform number of tokens per batch is not supported when using per-token SF.");
        TLLM_CHECK_ERROR(options.mBiasType == gemm::BiasType::None,
            "Uniform number of tokens per batch is not supported when using bias.");
        TLLM_CHECK_ERROR(options.mRouteImpl == RouteImpl::NoRoute,
            "Uniform number of tokens per batch is not supported when using routing.");
        TLLM_CHECK_ERROR(!options.mFusedAct,
            "Uniform number of tokens per batch is not supported when using fused gated activation.");
        TLLM_CHECK_ERROR(!tg::dtypeIsBlockFmt(options.mDtypeA) && !tg::dtypeIsBlockFmt(options.mDtypeB)
                && !tg::dtypeIsBlockFmt(options.mDtypeC),
            "Uniform number of tokens per batch is not supported when using block "
            "format for dtypeA, dtypeB, or dtypeC.");
    }
    else if (options.mBatchStrideInTokens >= 0)
    {
        TLLM_LOG_WARNING("Batch stride in tokens is set to ", options.mBatchStrideInTokens,
            " but it is not used when not using uniform tokens per batch.");
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

    BatchedGemmOptions mOptions;
    tg::CudaArch mSm{tg::CudaArch::Sm100a};
};

////////////////////////////////////////////////////////////////////////////////////////////////////

inline std::string dumpOptions(BatchedGemmOptions const& options, bool dumpRuntimeParams = true)
{
    std::stringstream ss;
    ss << gemmGatedAct::dumpOptions(options, dumpRuntimeParams) << ", ";
    if (dumpRuntimeParams)
    {
        ss << "mBatchedM={}," << std::endl;
        ss << "mBatchedN={}," << std::endl;
    }
    ss << "mBatchMode=batchedGemm::BatchedGemmOptions::BatchMode(" << static_cast<int32_t>(options.mBatchMode) << "),"
       << std::endl;
    if (dumpRuntimeParams)
    {
        ss << "mBatchStrideInTokens=" << options.mBatchStrideInTokens << "," << std::endl;
    }
    ss << "mFusedAct=" << options.mFusedAct << "," << std::endl;
    ss << "mGridWaitForPrimaryRouting=" << options.mGridWaitForPrimaryRouting << "," << std::endl;
    ss << "mIsStaticBatch=" << options.mIsStaticBatch << "," << std::endl;
    ss << "mIsUniformNumTokensPerBatch=" << options.mIsUniformNumTokensPerBatch << "," << std::endl;
    if (dumpRuntimeParams)
    {
        ss << "mNumBatches=" << options.mNumBatches << "," << std::endl;
    }
    ss << "mNumRegsPerThreadLoadA=" << options.mNumRegsPerThreadLoadA << "," << std::endl;
    ss << "mNumRegsPerThreadLoadB=" << options.mNumRegsPerThreadLoadB << "," << std::endl;
    ss << "mNumRegsPerThreadLoadSfA=" << options.mNumRegsPerThreadLoadSfA << "," << std::endl;
    ss << "mNumRegsPerThreadLoadSfB=" << options.mNumRegsPerThreadLoadSfB << "," << std::endl;
    if (dumpRuntimeParams)
    {
        ss << "mNumTokens=" << options.mNumTokens << "," << std::endl;
    }
    ss << "mNumWarpsLoadA=" << options.mNumWarpsLoadA << "," << std::endl;
    ss << "mNumWarpsLoadB=" << options.mNumWarpsLoadB << "," << std::endl;
    ss << "mNumWarpsLoadSfA=" << options.mNumWarpsLoadSfA << "," << std::endl;
    ss << "mNumWarpsLoadSfB=" << options.mNumWarpsLoadSfB << "," << std::endl;
    ss << "mRouteImpl=batchedGemm::RouteImpl(" << static_cast<int32_t>(options.mRouteImpl) << ")," << std::endl;
    ss << "mRouteSfsImpl={batchedGemm::RouteImpl(" << static_cast<int32_t>(options.mRouteSfsImpl.value()) << ")},"
       << std::endl;
    ss << "mUseTmaOobOpt=" << options.mUseTmaOobOpt << std::endl;
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
