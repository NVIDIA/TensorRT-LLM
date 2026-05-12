/*
 * Copyright (c) 2020-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "cuda_runtime_api.h"
#include "tensorrt_llm/common/config.h"
#include <cfloat>
#include <cstring>
#include <filesystem>
#include <linux/limits.h>
#include <memory>
#include <mutex>
#include <regex>
#include <sstream>
#include <tuple>
#include <unistd.h>
#include <unordered_map>

#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/kernels/multiHeadAttentionCommon.h"

#include "cubin/kernelMetaInfo.h"
#include "fmhaReduction.h"
#include "fmhaRunnerParams.h"
#include "prepareCustomMask.h"

// Switch to streaming-style TLLM_LOG_* macros for trtllm-gen export headers,
// which use streaming syntax (e.g., TLLM_LOG_INFO("val=", x)) instead of
// TRT-LLM's printf-style (e.g., TLLM_LOG_INFO("val=%d", x)).
#include "trtllmGen_fmha_export/FmhaAutoTuner.h"
#include "trtllmGen_fmha_export/FmhaInterface.h"
#include "trtllmGen_fmha_export/FmhaOptions.h"
#include "trtllmGen_fmha_export/KernelParams.h"
#include "trtllmGen_fmha_export/trtllmGenLogCompat.h"
// Restore original printf-style TLLM_LOG_* macros for the rest of this file.
#include "trtllmGen_fmha_export/trtllmGenLogCompatEnd.h"

namespace
{
namespace tc = tensorrt_llm::common;
namespace tg = trtllm::gen;
} // namespace

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////

// Check if two SM values are family/specific versions of the same architecture
// Returns true only if one is a family version and the other is a compatible specific version
constexpr bool isFamilySpecificSMPair(int sm1, int sm2)
{
    if ((sm1 == kSM_100f && (sm2 == kSM_100 || sm2 == kSM_103))
        || (sm2 == kSM_100f && (sm1 == kSM_100 || sm1 == kSM_103)))
    {
        return true;
    }
    return false;
}

constexpr bool isSMCompatible(int gpuSM, int kernelSM)
{
    if (gpuSM == kSM_103)
    {
        return kernelSM == kSM_100f || kernelSM == kSM_103;
    }
    else if (gpuSM == kSM_100)
    {
        return kernelSM == kSM_100f || kernelSM == kSM_100;
    }

    return gpuSM == kernelSM;
}

class TllmGenFmhaKernel
{

public:
    // The parameters for launching the kernel.
    // maxNumCtasQ, maxNumCtasKv, numCtasX, numCtasY, numCtasZ, clusterDimX
    struct CtaLaunchParams
    {
        // The maximum number of CTAs in Q dimension.
        int mMaxNumCtasQ;
        // The maximum number of CTAs in Kv dimension.
        int mMaxNumCtasKv;
        // The number of CTAs in X dimension.
        int mNumCtasX;
        // The number of CTAs in Y dimension.
        int mNumCtasY;
        // The number of CTAs in Z dimension.
        int mNumCtasZ;
        // The cluster size in the X dimension.
        int mClusterDimX;
    };

public:
    using KernelMeta = TllmGenFmhaKernelMetaInfo;
    using RunnerParams = TllmGenFmhaRunnerParams;
    using SelectKernelParams = TllmGenSelectKernelParams;
    using FmhaOptions = fmha::FmhaOptions;
    using FmhaOptionsFromArgs = fmha::FmhaOptionsFromArgs;
    using FmhaAutoTuner = fmha::FmhaAutoTuner;
    using FmhaInterface = fmha::FmhaInterface;
    using FmhaConfig = fmha::FmhaConfig;
    using FmhaData = fmha::FmhaData;
    using KernelParams = fmha::KernelParams;

    // Ctor.
    TllmGenFmhaKernel(KernelMeta const* pMetaStart, unsigned int nMetaCount, Data_type dtypeQ, Data_type dtypeK,
        Data_type dtypeV, Data_type dtypeOut, unsigned int smArch, int numEltsPerSageAttnBlkQ = 0,
        int numEltsPerSageAttnBlkK = 0, int numEltsPerSageAttnBlkP = 0, int numEltsPerSageAttnBlkV = 0)
        : mDtypeQ(dtypeQ)
        , mDtypeK(dtypeK)
        , mDtypeV(dtypeV)
        , mDtypeOut(dtypeOut)
        , mDriver(tensorrt_llm::common::CUDADriverWrapper::getInstance())
        , mKernelMeta(pMetaStart)
        , mKernelMetaCount(nMetaCount)
        , mSM(smArch)
        , mNumEltsPerSageAttnBlkQ(numEltsPerSageAttnBlkQ)
        , mNumEltsPerSageAttnBlkK(numEltsPerSageAttnBlkK)
        , mNumEltsPerSageAttnBlkP(numEltsPerSageAttnBlkP)
        , mNumEltsPerSageAttnBlkV(numEltsPerSageAttnBlkV)
    {
    }

    void loadKernels()
    {
        // Build a lookup map for all kernels.
        for (unsigned int i = 0; i < mKernelMetaCount; ++i)
        {
            auto const& kernelMeta = mKernelMeta[i];
            if (isSMCompatible(mSM, kernelMeta.mSM) && kernelMeta.mDataTypeQ == mDtypeQ
                && kernelMeta.mDataTypeK == mDtypeK && kernelMeta.mDataTypeV == mDtypeV
                && kernelMeta.mDataTypeO == mDtypeOut && kernelMeta.mNumEltsPerSageAttnBlkQ == mNumEltsPerSageAttnBlkQ
                && kernelMeta.mNumEltsPerSageAttnBlkK == mNumEltsPerSageAttnBlkK
                && kernelMeta.mNumEltsPerSageAttnBlkP == mNumEltsPerSageAttnBlkP
                && kernelMeta.mNumEltsPerSageAttnBlkV == mNumEltsPerSageAttnBlkV)
            {
                // Load CUmodules
                CUmodule hmod{0};
                auto findModuleIter = mModules.find(kernelMeta.mCubin);
                if (findModuleIter != mModules.end())
                {
                    hmod = findModuleIter->second;
                }
                else
                {
                    TLLM_CU_CHECK(mDriver->cuModuleLoadData(&hmod, kernelMeta.mCubin));
                    mModules.insert(std::make_pair(kernelMeta.mCubin, hmod));
                }
                // Build a hash map, which maps from kernel meta info to kernel index
                KernelInfo funcInfo;
                funcInfo.mMetaInfoIndex = i;
                TLLM_CU_CHECK(mDriver->cuModuleGetFunction(&funcInfo.mDeviceFunction, hmod, kernelMeta.mFuncName));
                if (kernelMeta.mSharedMemBytes >= 48 * 1024)
                {
                    auto const result = mDriver->cuFuncSetAttribute(funcInfo.mDeviceFunction,
                        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, kernelMeta.mSharedMemBytes);
                    if (result != CUDA_SUCCESS)
                    {
                        char const* errorName = nullptr;
                        char const* errorString = nullptr;
                        mDriver->cuGetErrorName(result, &errorName);
                        mDriver->cuGetErrorString(result, &errorString);
                        TLLM_LOG_WARNING("Skipping FMHA kernel due to cuFuncSetAttribute failure: "
                            + std::string(kernelMeta.mFuncName) + ", smem=" + std::to_string(kernelMeta.mSharedMemBytes)
                            + ", error=" + std::string(errorName != nullptr ? errorName : "unknown") + ": "
                            + std::string(errorString != nullptr ? errorString : "unknown"));
                        continue;
                    }
                }
                // Make sure the hashIds are not duplicated.
                // Except for the case where we have both family version and specific version of the same config.
                auto const hash = hashID(kernelMeta);
                auto it = mFunctions.find(hash);
                if (it != mFunctions.end())
                {
                    auto const& existingKernelMeta = mKernelMeta[it->second.mMetaInfoIndex];
                    TLLM_CHECK_WITH_INFO(isFamilySpecificSMPair(existingKernelMeta.mSM, kernelMeta.mSM),
                        "The kernel's hashId has conflicts with others.");
                    // Prefer specific SM version over family version
                    if (existingKernelMeta.mSM == kSM_100f)
                    {
                        it->second = funcInfo;
                    }
                }
                else
                {
                    mFunctions[hash] = funcInfo;
                }
            }
        }
    }

    static bool shouldUseNvrtc(FmhaOptions const& options)
    {
        return options.mFmhaKernelType == FmhaKernelType::SwapsMmaAbForGeneration
            && options.mDtypeKv != tg::Dtype::E2m1;
    }

    std::pair<bool, std::string> checkIfKernelExist(RunnerParams const& params) const
    {
        // Some conditions to check if the kernel is supported.
        // This is meant to avoid occupying unnecessary hashId bits.
        if (params.mHeadDimQk % 8 != 0 || params.mHeadDimV % 8 != 0)
        {
            return std::make_pair(false, "HeadDimQk and HeadDimV must be divisible by 8");
        }

        if (params.mMaxSeqLenQ == 0 || params.mBatchSize == 0
            || (!isContextKernel(params.mKernelType) && params.mMaxSeqLenKv == 0))
        {
            return std::make_pair(false, "Empty batch or zero sequence length");
        }

        // The selectKernelParams that might be updated.
        SelectKernelParams selectKernelParams{params};

        int32_t ctaDim = 512;
        FmhaOptions options;
        FmhaOptionsFromArgs optionsFromArgs;
        parseOptionsFromRunnerParams(params, options);
        options.mCudaArch = intToCudaArch(mSM);

        FmhaAutoTuner autoTuner(options, optionsFromArgs, params.mMultiProcessorCount);
        std::tie(options, optionsFromArgs, ctaDim) = autoTuner.selectKernel();

        // Check if the options are valid or not.
        checkFmhaOptions(options, optionsFromArgs);
        // Update the options if needed.
        updateFmhaOptions(options, optionsFromArgs);

        // The number of CtasQ and CtasKv per sequence, Ctas in the Y dimension, and Ctas in the Z
        // dimension.
        computeNumCtas(options, params.mMultiProcessorCount);

        if (shouldUseNvrtc(options))
        {
            // For the NVRTC path, we return supported as long as autotuner successfully selected a kernel config.
            return std::make_pair(true, "NVRTC path is supported");
        }

        // Check if a precompiled cubin exists for this configuration (same lookup as run()).
        // If not, return (false, info) so the dispatcher can fall back to unfused MHA like on main.
        algoFilterForCubinPath(options);
        auto [hashId, info] = hashFromFmhaOptions(options);

        if (mFunctions.find(hashId) == mFunctions.end())
        {
            TLLM_LOG_WARNING("Trtllm-gen kernels not found: " + info);
            return std::make_pair(false, info);
        }
        TLLM_LOG_DEBUG("TRTLLM-Gen kernel traits: %s", info.c_str());

        return std::make_pair(true, info);
    }

    void algoFilterForCubinPath(FmhaOptions& options) const
    {
        if (!isContextKernel(options.mFmhaKernelType) && options.mMaskType == TrtllmGenAttentionMaskType::Dense
            && !options.mIsMlaGen && !isTokenSparse(options.mSparseType))
        {
            options.mMaskType = TrtllmGenAttentionMaskType::Causal;
        }
    }

    void run(RunnerParams const& params)
    {
        if (params.mMaxSeqLenQ == 0 || params.mBatchSize == 0
            || (!isContextKernel(params.mKernelType) && params.mMaxSeqLenKv == 0))
        {
            return;
        }

        int32_t ctaDim = 512;
        FmhaOptions options;
        FmhaOptionsFromArgs optionsFromArgs;
        parseOptionsFromRunnerParams(params, options);
        options.mCudaArch = intToCudaArch(mSM);

        FmhaAutoTuner autoTuner(options, optionsFromArgs, params.mMultiProcessorCount);
        std::tie(options, optionsFromArgs, ctaDim) = autoTuner.selectKernel();

        // Check if the options are valid or not.
        checkFmhaOptions(options, optionsFromArgs);
        // Update the options if needed.
        updateFmhaOptions(options, optionsFromArgs);

        // Any caller that selects MultiCtasKvMode must supply the partial-reduction scratch pool
        // and per-CTA counter; fail fast here instead of silently falling back to Disabled.
        if (options.mMultiCtasKvMode == tensorrt_llm::kernels::MultiCtasKvMode::GmemReduction
            || options.mMultiCtasKvMode == tensorrt_llm::kernels::MultiCtasKvMode::GmemReductionWithSeparateKernel)
        {
            TLLM_CHECK_WITH_INFO(params.multiCtasKvScratchPtr != nullptr && params.multiCtasKvCounterPtr != nullptr,
                "MultiCtasKvScratchPtr/MultiCtasKvCounterPtr must be non-null when fmha kernel uses gmem-based "
                "multi-CTA reduction. "
                "The dispatcher must allocate and pass these buffers.");
        }

        // The number of CtasQ and CtasKv per sequence, Ctas in the Y dimension, and Ctas in the Z
        // dimension.
        auto [numCtasX, numCtasY, numCtasZ] = computeNumCtas(options, params.mMultiProcessorCount);

        // Set the launch grid size.
        tg::CudaRunner::Grid grid{numCtasX, numCtasY, numCtasZ};

        // Prepare custom mask for spec-decoding generation kernels if needed.
        if (params.mLayerIdx == 0 && params.mIsSpecDecTree)
        {
            int32_t stepQ = options.mTileSizeQ * options.mNumInstsQ;
            int32_t stepKv = options.mTileSizeKv * options.mNumInstsKv;
            runPrepareCustomMask(
                params, options.mFmhaKernelType, stepQ, stepKv, options.mTileSizeQ, options.mTileSizeKv, params.stream);
        }

        FmhaData fmhaData;
        setFmhaData(params, options, fmhaData);

        if (shouldUseNvrtc(options))
        {
            // nvrtc path - uses mFmhaInterface member for kernel caching
            FmhaConfig fmhaConfig;
            fmhaConfig.mOptions = options;
            std::ostringstream sstream;
            populateJsonConfig(options, sstream);
            fmhaConfig.mGenCfgJsonStr = sstream.str();

            fmhaConfig.mExecPath = getExecPath().c_str();
            fmhaConfig.mCtaDim = ctaDim;
            fmhaConfig.mGrid = grid;
            mFmhaInterface.generateAndCompileKernel(fmhaConfig);
            mFmhaInterface.run(fmhaConfig, fmhaData, params.stream, params.mMultiProcessorCount, 0);
        }
        else
        {
            algoFilterForCubinPath(options);
            auto [hashId, info] = hashFromFmhaOptions(options);

            // load from cubin
            auto const findIter = mFunctions.find(hashId);
            // Add debug info when kernels are not found.
            TLLM_CHECK_WITH_INFO(findIter != mFunctions.end(), "Trtllm-gen kernels not found: " + info);

            auto const& kernelMeta = mKernelMeta[findIter->second.mMetaInfoIndex];
            const CUfunction func = findIter->second.mDeviceFunction;

            // mGroupsHeadsQ and mGroupsTokensHeadsQ are not part of the hashID, so they don't
            // affect kernel lookup. Use cubin-side values from kernelMeta instead of AutoTuner
            // output, since cubins are exported with enableAutotuner=false.
            options.mGroupsHeadsQ = kernelMeta.mGroupsHeadsQ;
            options.mGroupsTokensHeadsQ = kernelMeta.mGroupsTokensHeadsQ;

            KernelParams kernelParams = fmha::KernelParamsSetup::setKernelParams(options, grid[0], grid[1], grid[2],
                fmhaData.mMetaData.cumSeqLensQPtrD, fmhaData.mMetaData.cumSeqLensKvPtrD, fmhaData.mMetaData.seqLensKvD,
                fmhaData.mInputBuffers.qBasePtr, fmhaData.mInputBuffers.kBasePtr, fmhaData.mInputBuffers.vBasePtr,
                fmhaData.mScales.kSfBasePtr, fmhaData.mScales.vSfBasePtr,
                fmhaData.mInputBuffers.slidingWindowKvPoolBasePtr, fmhaData.mMetaData.kvPageIdxD,
                fmhaData.mScales.outputScaleD, fmhaData.mScales.scaleSoftmaxLog2D, fmhaData.mScales.kvSfScaleD,
                fmhaData.mScales.oSfScaleD, fmhaData.mInputBuffers.customMaskPtrD,
                fmhaData.mInputBuffers.customMaskOffsetsPtrD, fmhaData.mMetaData.firstSparseMaskOffsetsKvPtrD,
                fmhaData.mMetaData.sparseMlaTopKLensPtrD, fmhaData.mScales.sageAttnSfsQPtrD,
                fmhaData.mScales.sageAttnSfsKPtrD, fmhaData.mScales.sageAttnSfsPPtrD, fmhaData.mScales.sageAttnSfsVPtrD,
                fmhaData.mInputBuffers.attentionSinksPtrD, fmhaData.mOutputBuffers.oPtrD, fmhaData.mScales.oSfPtrD,
                fmhaData.mOutputBuffers.multiCtasKvCounterPtrD, fmhaData.mOutputBuffers.partialOPtrD,
                fmhaData.mOutputBuffers.partialStatsPtrD, fmhaData.mOutputBuffers.skipSoftmaxStatsPtrD,
                fmhaData.mOutputBuffers.softmaxStatsD, fmhaData.mOutputBuffers.oDebugPtrD,
                fmhaData.mScales.softmaxScale, fmhaData.mMetaData.inflateMax, fmhaData.mScales.kvSfScale,
                fmhaData.mScales.oSfScale, fmhaData.mMetaData.startTokenIdxSfO, options.mUseBlockSparseAttention,
                options.mUsesSharedPagedKvIdx);

            launchFmhaKernel(kernelParams, kernelMeta, func, grid, options, params.stream);
            // Run the separate reduction kernel if needed.
            runFmhaReduction(kernelMeta, kernelParams, params.mMultiProcessorCount, params.stream);
        }
    }

private:
    inline uint64_t hashID(int qkvLayout, int maskType, int kernelType, int scheduler, int multiCtasKvMode,
        int headDimPerCtaV, int headDimQk, int headDimV, int tileSizeQ, int tileSizeKv, int numTokensPerPage,
        bool reuseSmemKForV, bool uses2CtaMma, int sparseAttention, bool skipsSoftmax) const
    {
        TLLM_CHECK_WITH_INFO((headDimPerCtaV >= 32) && (headDimQk >= 32) && (headDimV >= 32) && (headDimPerCtaV <= 1024)
                && (headDimQk <= 1024) && (headDimV <= 1024),
            "Expect (32 <= headDim <= 1024), got headDimPerCtaV=%d, headDimQk=%d, "
            "headDimV=%d",
            headDimPerCtaV, headDimQk, headDimV);
        // The numTokensPerPage must be 0 (non-paged layouts) or a power of 2 (paged layouts).
        TLLM_CHECK_WITH_INFO(numTokensPerPage == 0 || (numTokensPerPage & (numTokensPerPage - 1)) == 0,
            "The numTokensPerPage must be 0 or power of 2, got %d.", numTokensPerPage);
        TLLM_CHECK_WITH_INFO(tileSizeQ <= 128 && tileSizeKv <= 128, "The tileSizeQ and tileSizeKv must be <= 128.");
        TLLM_CHECK_WITH_INFO((tileSizeQ & (tileSizeQ - 1)) == 0 && (tileSizeKv & (tileSizeKv - 1)) == 0,
            "The tileSizeQ and tileSizeKv must be power of 2.");
        TLLM_CHECK_WITH_INFO(tileSizeKv == 64 || tileSizeKv == 128, "The tileSizeKv must be 64 or 128.");
        // Format of the hash key:
        // Bit 0  - 3 : qkvLayout.
        // Bit 4  - 7 : maskType.
        // Bit 8  - 11: kernelType.
        // Bit 12 - 15: tileScheduler.
        // Bit 16 - 17: multiCtasKvMode.
        // Bit 18 - 25: (headDimPerCtaV >> 3).
        // Bit 26 - 33: (headDimQk >> 3).
        // Bit 34 - 41: (headDimV >> 3).
        // Bit 42 - 43: (tileSizeKv >> 6).
        // Bit 44 - 48: (log2(numTokensPerPage)).
        // Bit 49 - 52: (log2(tileSizeQ)).
        // Bit 53 - 53: reuseSmemKForV.
        // Bit 54 - 54: uses2CtaMma.
        // Bit 55 - 56: sparseAttention.
        // Bit 57 - 57: skipsSoftmax.
        return (static_cast<uint64_t>(qkvLayout) << 0) | (static_cast<uint64_t>(maskType) << 4)
            | (static_cast<uint64_t>(kernelType) << 8) | (static_cast<uint64_t>(scheduler) << 12)
            | (static_cast<uint64_t>(multiCtasKvMode) << 16) | (static_cast<uint64_t>(headDimPerCtaV >> 3) << 18)
            | (static_cast<uint64_t>(headDimQk >> 3) << 26) | (static_cast<uint64_t>(headDimV >> 3) << 34)
            | (static_cast<uint64_t>(tileSizeKv >> 6) << 42)
            | (static_cast<uint64_t>(numTokensPerPage > 0 ? static_cast<int>(log2(numTokensPerPage)) : 0) << 44)
            | (static_cast<uint64_t>(log2(tileSizeQ)) << 49) | (static_cast<uint64_t>(reuseSmemKForV) << 53)
            | (static_cast<uint64_t>(uses2CtaMma) << 54) | (static_cast<uint64_t>(sparseAttention) << 55)
            | (static_cast<uint64_t>(skipsSoftmax) << 57);
    }

    uint64_t hashID(KernelMeta const& kernelMeta) const
    {
        return hashID(kernelMeta.mQkvLayout, kernelMeta.mMaskType, kernelMeta.mKernelType, kernelMeta.mTileScheduler,
            kernelMeta.mMultiCtasKvMode, kernelMeta.mHeadDimPerCtaV, kernelMeta.mHeadDimQk, kernelMeta.mHeadDimV,
            kernelMeta.mTileSizeQ, kernelMeta.mTileSizeKv, kernelMeta.mNumTokensPerPage, kernelMeta.mReuseSmemKForV,
            kernelMeta.m2CtaMma, kernelMeta.mSparseAttn, kernelMeta.mSkipsSoftmaxWhenPossible);
    }

    std::pair<uint64_t, std::string> hashFromFmhaOptions(FmhaOptions const& options) const
    {
        // uses2CtaMma: "2CTA MMA kernel variant" (MLA KeepsMmaAb with clusterDimX=2).
        // CGA scaling (clusterDimX *= mMaxNumCtasKv) is applied only at launch time in launchFmhaKernel,
        // so options.mClusterDimX here is the unscaled value from the autotuner.
        bool uses2CtaMma = (options.mClusterDimX == 2);
        // Debug info.
        std::string info = "dtypeQ=" + std::to_string(static_cast<int>(mDtypeQ)) + ", dtypeK="
            + std::to_string(static_cast<int>(mDtypeK)) + ", dtypeV=" + std::to_string(static_cast<int>(mDtypeV))
            + ", dtypeOut=" + std::to_string(static_cast<int>(mDtypeOut)) + ", sm=" + std::to_string(mSM)
            + ", mNumEltsPerSageAttnBlkQ=" + std::to_string(mNumEltsPerSageAttnBlkQ)
            + ", mNumEltsPerSageAttnBlkK=" + std::to_string(mNumEltsPerSageAttnBlkK)
            + ", mNumEltsPerSageAttnBlkP=" + std::to_string(mNumEltsPerSageAttnBlkP)
            + ", mNumEltsPerSageAttnBlkV=" + std::to_string(mNumEltsPerSageAttnBlkV)
            + ", qkvLayout=" + std::to_string(static_cast<int>(options.mQkvLayout))
            + ", maskType=" + std::to_string(static_cast<int>(options.mMaskType))
            + ", kernelType=" + std::to_string(static_cast<int>(options.mFmhaKernelType))
            + ", tileScheduler=" + std::to_string(static_cast<int>(options.mTileScheduler))
            + ", multiCtasKvMode=" + std::to_string(static_cast<int>(options.mMultiCtasKvMode)) + ", headDimPerCtaV="
            + std::to_string(options.mHeadDimPerCtaV) + ", headDimQk=" + std::to_string(options.mHeadDimQk)
            + ", headDimV=" + std::to_string(options.mHeadDimV) + ", tileSizeQ=" + std::to_string(options.mTileSizeQ)
            + ", tileSizeKv=" + std::to_string(options.mTileSizeKv) + ", numTokensPerPage="
            + std::to_string(options.mNumTokensPerPage) + ", reuseSmemKForV=" + std::to_string(options.mReuseSmemKForV)
            + ", uses2CtaMma=" + std::to_string(uses2CtaMma)
            + ", sparseType=" + std::to_string(static_cast<int>(options.mSparseType))
            + ", skipsSoftmax=" + std::to_string(options.mSkipsSoftmaxWhenPossible);

        TLLM_LOG_DEBUG("Searching for kernel traits: " + info);
        return std::make_pair(hashID(static_cast<int>(options.mQkvLayout), static_cast<int>(options.mMaskType),
                                  static_cast<int>(options.mFmhaKernelType), static_cast<int>(options.mTileScheduler),
                                  static_cast<int>(options.mMultiCtasKvMode), static_cast<int>(options.mHeadDimPerCtaV),
                                  static_cast<int>(options.mHeadDimQk), static_cast<int>(options.mHeadDimV),
                                  static_cast<int>(options.mTileSizeQ), static_cast<int>(options.mTileSizeKv),
                                  static_cast<int>(options.mNumTokensPerPage), options.mReuseSmemKForV, uses2CtaMma,
                                  static_cast<int>(options.mSparseType), options.mSkipsSoftmaxWhenPossible),
            info);
    }

    std::string const& getExecPath() const
    {
        static std::string execPathStr;
        if (execPathStr.empty())
        {
            // Get build directory relative path from CMake macro
            // This is the relative path from project root to build directory
            std::string buildRelPath = "cpp/build/tensorrt_llm/kernels/trtllmGenKernels/fmha";
#ifdef TRTLLM_FMHA_BUILD_DIR
            // Try to extract relative path from absolute path
            // TRTLLM_FMHA_BUILD_DIR is the absolute build directory path
            std::string buildAbsDir = TRTLLM_FMHA_BUILD_DIR;
            // Extract relative path by finding "cpp/build" in the absolute path
            size_t pos = buildAbsDir.find("cpp/build");
            if (pos != std::string::npos)
            {
                buildRelPath = buildAbsDir.substr(pos);
            }
#endif

            // Always use pip show to find installation location at runtime
            char const* cmd = "pip show tensorrt_llm 2>/dev/null";

            // Buffer to store the output
            std::array<char, 128> buffer;
            std::string result;
            // Open pipe to command
#ifdef _MSC_VER
            FILE* pipe = _popen(cmd, "r");
#else
            FILE* pipe = popen(cmd, "r");
#endif
            if (pipe)
            {
                // Read the output
                while (fgets(buffer.data(), buffer.size(), pipe) != nullptr)
                {
                    result += buffer.data();
                }
// Close the pipe
#ifdef _MSC_VER
                _pclose(pipe);
#else
                pclose(pipe);
#endif
                // Parse the location using regex
                // `pip show tensorrt_llm` will output something like:
                // Location: /usr/local/lib/python3.12/dist-packages
                // Editable project location: /code
                std::regex locationRegex("(Location|Editable project location): (.+)");
                // Find all matches
                auto match_begin = std::sregex_iterator(result.begin(), result.end(), locationRegex);
                auto match_end = std::sregex_iterator();

                // Get the number of matches
                auto match_count = std::distance(match_begin, match_end);

                if (match_count > 0)
                {
                    std::string location;
                    bool foundEditable = false;

                    // First, try to find "Editable project location" (preferred)
                    for (auto it = match_begin; it != match_end; ++it)
                    {
                        std::string matchType = it->str(1);
                        if (matchType == "Editable project location")
                        {
                            location = it->str(2);
                            foundEditable = true;
                            break;
                        }
                    }

                    // If not found, use "Location" as fallback
                    if (!foundEditable)
                    {
                        for (auto it = match_begin; it != match_end; ++it)
                        {
                            std::string matchType = it->str(1);
                            if (matchType == "Location")
                            {
                                location = it->str(2);
                                break;
                            }
                        }
                    }

                    // If still not found, use the last match as fallback
                    if (location.empty())
                    {
                        TLLM_LOG_WARNING("No location found, using the last match as fallback.");
                        auto last_match_iter = match_begin;
                        std::advance(last_match_iter, match_count - 1);
                        location = last_match_iter->str(2);
                    }

                    // Trim whitespace
                    location.erase(location.find_last_not_of(" \n\r\t") + 1);

                    // Build the exec path: try candidate paths in order of priority.
                    std::vector<std::filesystem::path> candidatePaths;
                    if (foundEditable)
                    {
                        candidatePaths.push_back(std::filesystem::path(location) / buildRelPath);
                    }
                    auto fmhaIncludeSuffix
                        = std::filesystem::path("tensorrt_llm") / "include" / "trtllm_gen_kernels" / "fmha";
                    candidatePaths.push_back(std::filesystem::path(location) / fmhaIncludeSuffix);

                    bool pathFound = false;
                    for (auto const& candidate : candidatePaths)
                    {
                        if (std::filesystem::exists(candidate))
                        {
                            execPathStr = (candidate / "numb").string();
                            pathFound = true;
                            break;
                        }
                    }
                    TLLM_CHECK_WITH_INFO(pathFound,
                        "FMHA NVRTC kernel headers not found in any candidate path. "
                        "FMHA JIT compilation may fail. Please check the installation of TensorRT-LLM.");
                }
            }
            else
            {
                TLLM_LOG_WARNING("Failed to find TensorRT-LLM installation, NVRTC FMHA path will be unavailable.");
            }
        }
        return execPathStr;
    }

    // Prepare pointers for TMA descriptors.
    static std::tuple<void const*, void const*, void const*> getDevicePtrs(
        TllmGenFmhaRunnerParams const& runnerParams, int32_t bitsPerElt)
    {
        // Declare the q, k, v ptrs.
        void const *qPtr{runnerParams.qPtr}, *kPtr{runnerParams.kPtr}, *vPtr{runnerParams.vPtr};

        // Set Q, K and V pointer from packed QKV tensor.
        if (isPackedQkv(runnerParams.mQkvLayout))
        {
            qPtr = runnerParams.qkvPtr;
            kPtr = reinterpret_cast<void const*>(reinterpret_cast<char const*>(runnerParams.qkvPtr)
                + runnerParams.mNumHeadsQ * runnerParams.mHeadDimQk * bitsPerElt / 8 /*bits*/);
            vPtr = reinterpret_cast<void const*>(reinterpret_cast<char const*>(runnerParams.qkvPtr)
                + (runnerParams.mNumHeadsQ + runnerParams.mNumHeadsKv) * runnerParams.mHeadDimQk * bitsPerElt
                    / 8 /*bits*/);
        }
        // Set K and V pointer from pagedKv tensor.
        else if (isPagedKv(runnerParams.mQkvLayout))
        {
            // Note that the offsets will be fully handled by the pageIdx buffer.
            kPtr = runnerParams.kvPtr;
            vPtr = runnerParams.kvPtr;
        }
        // Set K and V pointer from contiguousQAnddKv tensor.
        else if (isContiguousKv(runnerParams.mQkvLayout))
        {
            kPtr = runnerParams.kvPtr;
            // The maximum headDim of K and V.
            // Note that contiguousKv or pagedKv will pad K and V to maxHeadDimKv.
            int32_t const maxHeadDimKv{std::max(runnerParams.mHeadDimQk, runnerParams.mHeadDimV)};
            vPtr = reinterpret_cast<void const*>(reinterpret_cast<char const*>(runnerParams.kvPtr)
                + runnerParams.mNumHeadsKv * runnerParams.mMaxSeqLenCacheKv * maxHeadDimKv * bitsPerElt / 8 /*bits*/);
        }

        // Return the pointers.
        return std::make_tuple(qPtr, kPtr, vPtr);
    }

    void setFmhaData(RunnerParams const& params, FmhaOptions const& options, FmhaData& fmhaData) const
    {
        // Fill MetaData
        fmhaData.mMetaData.cumSeqLensQPtrD = params.cumSeqLensQPtr;
        fmhaData.mMetaData.cumSeqLensKvPtrD = params.cumSeqLensKvPtr;
        fmhaData.mMetaData.seqLensKvD = params.seqLensKvPtr;
        fmhaData.mMetaData.firstSparseMaskOffsetsKvPtrD = params.firstSparseMaskOffsetsKvPtr;
        fmhaData.mMetaData.sparseMlaTopKLensPtrD = params.ptrSparseMlaTopKLens;
        fmhaData.mMetaData.kvPageIdxD = params.kvPageIdxPtr;
        fmhaData.mMetaData.inflateMax = 0.0F; // Default value for inflate max
        fmhaData.mMetaData.startTokenIdxSfO = params.mSfStartTokenIdx;

        // Fill Scales
        fmhaData.mScales.kSfBasePtr = params.kvSfPtr;
        fmhaData.mScales.vSfBasePtr = params.kvSfPtr;
        fmhaData.mScales.scaleSoftmaxLog2D = params.scaleSoftmaxLog2Ptr;
        fmhaData.mScales.outputScaleD = params.outputScalePtr;
        fmhaData.mScales.kvSfScaleD = params.kvSfScalePtr;
        fmhaData.mScales.oSfScaleD = params.oSfScalePtr;
        // Sage Attention scaling factors
        fmhaData.mScales.sageAttnSfsQPtrD = params.sageAttnSfsQPtr;
        fmhaData.mScales.sageAttnSfsKPtrD = params.sageAttnSfsKPtr;
        fmhaData.mScales.sageAttnSfsPPtrD = params.sageAttnSfsPPtr;
        fmhaData.mScales.sageAttnSfsVPtrD = params.sageAttnSfsVPtr;
        // Host-side scale values (from params)
        TLLM_CHECK_WITH_INFO(params.mScaleQ != 0.f, "mScaleQ must not be zero (used as divisor in softmaxScale).");
        fmhaData.mScales.softmaxScale
            = (1.f / (std::sqrt(static_cast<float>(params.mHeadDimQk)) * params.mScaleQ)) * M_LOG2E;

        fmhaData.mScales.kvSfScale = 1.f; // Default value; per-token scale from kvSfScaleD when set
        fmhaData.mScales.oSfScale = 1.f;  // Default value; per-token scale from oSfScaleD (e2m1/NVFP4 O) when set
        fmhaData.mScales.oSfPtrD = params.oSfPtr;

        // Get qkv pointers.
        auto [qPtr, kPtr, vPtr] = getDevicePtrs(params, tg::dtypeGetNumBits(options.mDtypeK));
        // Fill InputBuffers
        fmhaData.mInputBuffers.qBasePtr = qPtr;
        fmhaData.mInputBuffers.kBasePtr = kPtr;
        fmhaData.mInputBuffers.vBasePtr = vPtr;
        fmhaData.mInputBuffers.slidingWindowKvPoolBasePtr = params.slidingWindowKvPoolBasePtr;
        fmhaData.mInputBuffers.attentionSinksPtrD = params.attentionSinksPtr;
        fmhaData.mInputBuffers.customMaskPtrD = params.customMaskPtr;
        fmhaData.mInputBuffers.customMaskOffsetsPtrD = params.customMaskOffsetsPtr;

        // Fill OutputBuffers
        fmhaData.mOutputBuffers.oPtrD = params.oPtr;
        fmhaData.mOutputBuffers.multiCtasKvCounterPtrD = params.multiCtasKvCounterPtr;
        // Split multiCtasKvScratchPtr into partialStats and partialO
        // Note: The split calculation is done in setKernelParams with kernelMeta.mStepQ
        // For FmhaData, we store the base pointer and let the kernel params handle the split
        fmhaData.mOutputBuffers.partialStatsPtrD = reinterpret_cast<float2*>(params.multiCtasKvScratchPtr);
        // The partial buffers' pointers when the multiCtasKv mode is enabled.
        int64_t partialStatsBufferSize = params.mMultiProcessorCount * options.mTileSizeQ * options.mNumInstsQ;
        // partialO comes after partialStats, but exact offset depends on kernelMeta
        // This will be properly set in setKernelParams
        fmhaData.mOutputBuffers.partialOPtrD = fmhaData.mOutputBuffers.partialStatsPtrD + partialStatsBufferSize;
        fmhaData.mOutputBuffers.skipSoftmaxStatsPtrD = nullptr; // Not available in params (would need to be added)
        fmhaData.mOutputBuffers.softmaxStatsD = params.softmaxStatsPtr;
        fmhaData.mOutputBuffers.oDebugPtrD = nullptr;           // Debug output not supported in TensorRT-LLM

        // Print all primitive type variables in FmhaData for debugging
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    // Convert Data_type to trtllm::gen::Dtype
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    tg::Dtype dataTypeToDtype(Data_type dataType) const
    {
        switch (dataType)
        {
        case DATA_TYPE_BOOL: return tg::Dtype::Bool;
        case DATA_TYPE_FP16: return tg::Dtype::Fp16;
        case DATA_TYPE_FP32: return tg::Dtype::Fp32;
        case DATA_TYPE_INT4:
            // Note: DATA_TYPE_INT4 could map to MxInt4 (block format)
            // Adjust based on your use case
            return tg::Dtype::MxInt4;
        case DATA_TYPE_INT8: return tg::Dtype::Int8;
        case DATA_TYPE_INT32: return tg::Dtype::Int32;
        case DATA_TYPE_BF16: return tg::Dtype::Bfloat16;
        case DATA_TYPE_E2M1: return tg::Dtype::E2m1;
        case DATA_TYPE_E4M3: return tg::Dtype::E4m3;
        case DATA_TYPE_E5M2: return tg::Dtype::E5m2;
        default:
            // Fallback or throw error
            return tg::Dtype::Void;
        }
    }

    void parseOptionsFromRunnerParams(RunnerParams const& params, FmhaOptions& options) const
    {
        // Basic dimensions
        options.mBatchSize = params.mBatchSize;
        options.mMaxSeqLenQ = params.mMaxSeqLenQ;
        options.mMinSeqLenQ = params.mMaxSeqLenQ;
        // For context self-attention without prior KV cache, TRT-LLM passes
        // mMaxSeqLenKv=0. trtllm-gen requires mMaxSeqLenKv>0, so use mMaxSeqLenQ
        // as the effective KV length (Q produces KV of equal length).
        options.mMaxSeqLenKv = (isContextKernel(params.mKernelType) && params.mMaxSeqLenKv == 0) ? params.mMaxSeqLenQ
                                                                                                 : params.mMaxSeqLenKv;
        options.mMinSeqLenKv = options.mMaxSeqLenKv;

        // Variable sequence length support
        options.mSumOfSeqLensQ = params.mSumOfSeqLensQ;
        options.mSumOfSeqLensKv = params.mSumOfSeqLensKv;

        // Head configuration
        options.mNumHeadsQ = params.mNumHeadsQ;
        options.mNumHeadsKv = params.mNumHeadsKv;
        options.mNumHeadsQPerKv = params.mNumHeadsQPerKv;
        options.mHeadDimQk = params.mHeadDimQk;
        options.mHeadDimV = params.mHeadDimV;

        // Layout and mask configuration
        options.mQkvLayout = params.mQkvLayout;
        options.mMaskType = params.mMaskType;
        options.mFmhaKernelType = params.mKernelType;
        options.mTileScheduler = params.mTileScheduler;

        // Multi-CTA KV mode (bool -> enum conversion)
        options.mMultiCtasKvMode = params.mMultiCtasKvMode ? tensorrt_llm::kernels::MultiCtasKvMode::GmemReduction
                                                           : tensorrt_llm::kernels::MultiCtasKvMode::Disabled;
        options.mStoresSoftmaxStats = true;

        // Attention features
        options.mUseBlockSparseAttention = params.mUseBlockSparseAttention;
        options.mAttentionWindowSize = params.mAttentionWindowSize;
        options.mChunkedAttentionSize = params.mChunkedAttentionSize == INT_MAX ? 0 : params.mChunkedAttentionSize;

        // Sparse attention (MLA / MQA / GQA)
        options.mSparseType = params.mSparseAttention;
        options.mSparseAttnTopK = params.mSparseTopK;
        options.mHasSlidingWindowKvPool = isMlaGenKernel(params) && isDynamicTokenSparse(params.mSparseAttention);

        // Softmax optimization
        options.mSkipSoftmaxThresholdScaleFactor = params.mSkipSoftmaxThresholdScaleFactor;
        options.mSkipsSoftmaxWhenPossible = params.mSkipSoftmaxThresholdScaleFactor != 0.0f;

        // Paged KV cache
        options.mMaxNumPagesPerSeqKv = params.mMaxNumPagesPerSeqKv;
        options.mNumTokensPerPage = params.mNumTokensPerPage;
        options.mNumPagesInMemPool = params.mNumPagesInMemPool;

        options.mEnablesAutoTuner = true;
        options.mIsMlaGen = isMlaGenKernel(params);
        options.mDtypeQ = dataTypeToDtype(mDtypeQ);
        options.mDtypeKv = dataTypeToDtype(mDtypeK);
        options.mDtypeK = dataTypeToDtype(mDtypeK);
        options.mDtypeV = dataTypeToDtype(mDtypeV);
        options.mDtypeOut = dataTypeToDtype(mDtypeOut);
        options.mNumEltsPerSageAttnBlkQ = mNumEltsPerSageAttnBlkQ;
        options.mNumEltsPerSageAttnBlkK = mNumEltsPerSageAttnBlkK;
        options.mNumEltsPerSageAttnBlkP = mNumEltsPerSageAttnBlkP;
        options.mNumEltsPerSageAttnBlkV = mNumEltsPerSageAttnBlkV;
        options.mSupportsVarSeqLens = true;
        if (options.mQkvLayout != QkvLayout::PackedQkv)
        {
            options.mSupportsDiffSeqLensForQAndKv = true;
        }

        // Enables the optimization to skip the correction step when possible.
        options.mSkipsCorrWhenPossible = true;

        // Enables interleaveSfV by default.
        options.mInterleaveSfV = true;

        // Enables PDL if specified.
        options.mEnablesPdl = tensorrt_llm::common::getEnvEnablePDL();

        // spec-decoding
        bool isContext = params.mKernelType == FmhaKernelType::Context;
        options.mIsCustomSpecDecodingGen = !isContext && params.mMaxSeqLenQ > 1 && params.mIsSpecDecTree;
        options.mIsCausalSpecDecodingGen = !isContext && params.mMaxSeqLenQ > 1 && !params.mIsSpecDecTree;
        options.mNumSpecDecodingTokens = !isContext && params.mMaxSeqLenQ > 1 ? params.mMaxSeqLenQ : 0;

        options.mIsTrtllmLayout = true;
    }

    void populateJsonConfig(FmhaOptions const& options, std::ostringstream& sstream) const
    {
        sstream << "{\n";
        sstream << "\"clusterDimX\": " << options.mClusterDimX << ",\n";
        // The 2CTA UTCMMA is used by default if the clusterDimX is set to 2.
        if (options.mClusterDimX == 2)
        {
            sstream << "\"usesTwoCtasForMma\": true,\n";
        }
        // Use dynamic cluster dimensions if the CGA reduction is used.
        if (isCgaSmemReduction(options.mMultiCtasKvMode))
        {
            sstream << "\"usesDynamicClusterDims\": true,\n";
        }

        // Disable checksTaskSchedules as there are multiple acquire/commit, wait/release steps in one
        // loop. And the number of loops are not the same in different tasks.
        sstream << "\"checksTaskSchedules\": false,\n";

        if (options.mIsExportingCubin)
        {
            sstream << "\"compileDefs\": [\"-DTLLM_EXPORT_CUBIN\"],\n";
        }

        // Set compile flags for E2M1 KV kernel benchmark.
        // NOTE(tizheng): This is to be removed after compiler fixes PTX exposure of QMUL4. See Fp4Utils.h for details.
        if (options.mChecksResults == 0 && options.mDtypeKv == tg::Dtype::E2m1)
        {
            TLLM_LOG_INFO("Forcing -DTLLM_BENCHMARK_E2M1_KV_CACHE for E2m1 Kv. The results are not correct.");
            sstream << "\"compileDefs\": [\"-DTLLM_BENCHMARK_E2M1_KV_CACHE\"],\n";
        }

        // Enable programmatic dependent launch.
        sstream << "\"enablesPdl\": " << ((options.mEnablesPdl) ? "true" : "false") << ",\n";

        // Postpone the waitsForPrimaryGrid to loading smemQ/smemKv in order to hide the latency as much as possible.
        // This avoids adding the cudaGridDependencySynchronize in the very beginning of the kernel.
        sstream << "\"gridWaitForPrimaryEarlyExit\": false,\n";

        // This forces the schedule to be printed and for the generator to look for deadlocks. It may
        // detect false-positive. If that's the case, disable that feature.
        sstream << "\"printsFullSchedule\": false,\n";

        sstream << "\"skipsKernelGen\": " << ((options.mSkipsKernelGen) ? "true" : "false") << ",\n";
        sstream << "\"smVersion\": \"" << tg::cudaArchToString(options.mCudaArch) << "\",\n";

        // Loads scales from gmem when it is quantized.
        if (options.mDtypeQ == tg::Dtype::E4m3 || options.mDtypeKv == tg::Dtype::E4m3
            || options.mDtypeOut == tg::Dtype::E4m3 || options.mDtypeQ == tg::Dtype::E2m1
            || options.mDtypeKv == tg::Dtype::E2m1 || options.mDtypeOut == tg::Dtype::E2m1)
        {
            sstream << "\"loadsScalesFromGmem\": true,\n";
        }

        // The code uses CUDA PTX.
        sstream << "\"usesCudaPtx\": true,\n";

        if (options.mTileScheduler == TileScheduler::Persistent)
        {
            sstream << "\"skipsFirstProdAcquiresLastConsReleases\": false,\n";
        }
        // Reduce the number of anti-dependencies in the generated code.
        // See https://nvbugs/4940327
        // WARNING: Increase this might lead to more register usage and even spills. Please finetune it carefully.
        sstream << "\"antiDepWeight\": 25,\n";

        sstream << "\"usesNvRtc\": "
                << "true"
                << ",\n";
        sstream << "\"usesTma\": true\n";
        sstream << "}\n";
    }

    void launchFmhaKernel(KernelParams const& kernelParams, KernelMeta const& kernelMeta, CUfunction const& func,
        tg::CudaRunner::Grid const& grid, FmhaOptions const& options, CUstream stream) const
    {
        int launchedClusterDimX = options.mClusterDimX;
        if (isCgaSmemReduction(options.mMultiCtasKvMode))
        {
            launchedClusterDimX *= options.mMaxNumCtasKv;
        }

        // Prepare kernel parameters list for cuLaunchKernelEx.
        void* kernelParamsList[] = {const_cast<void*>(static_cast<void const*>(&kernelParams))};
        CUlaunchConfig launch_config;
        launch_config.blockDimX = kernelMeta.mThreadsPerCTA;
        launch_config.blockDimY = 1;
        launch_config.blockDimZ = 1;
        launch_config.gridDimX = grid[0];
        launch_config.gridDimY = grid[1];
        launch_config.gridDimZ = grid[2];
        launch_config.hStream = stream;
        launch_config.sharedMemBytes = kernelMeta.mSharedMemBytes;

        // Debug info.
        TLLM_LOG_DEBUG("TRTLLM-Gen launch info: kernelName = %s", kernelMeta.mFuncName);
        TLLM_LOG_DEBUG(
            "TRTLLM-Gen launch info: maxSeqLenQ = %d, "
            "maxSeqLenKv = %d, "
            "numHeadsQ = %d, "
            "numHeadsKv = %d, batchSize = %d, kernelType = %d",
            options.mMaxSeqLenQ, options.mMaxSeqLenKv, options.mNumHeadsQ, options.mNumHeadsKv, options.mBatchSize,
            static_cast<int>(options.mFmhaKernelType));
        TLLM_LOG_DEBUG("TRTLLM-Gen launch info: numCtasX = %d, numCtasY = %d, numCtasZ = %d, clusterDimX = %d",
            launch_config.gridDimX, launch_config.gridDimY, launch_config.gridDimZ, launchedClusterDimX);

        CUlaunchAttribute launch_attribute[3];
        launch_attribute[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
        launch_attribute[0].value.clusterDim.x = launchedClusterDimX;
        launch_attribute[0].value.clusterDim.y = 1;
        launch_attribute[0].value.clusterDim.z = 1;
        launch_attribute[1].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
        launch_attribute[1].value.clusterSchedulingPolicyPreference
            = launchedClusterDimX > 1 ? CU_CLUSTER_SCHEDULING_POLICY_SPREAD : CU_CLUSTER_SCHEDULING_POLICY_DEFAULT;
        launch_attribute[2].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
        launch_attribute[2].value.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();

        launch_config.attrs = launch_attribute;
        launch_config.numAttrs = 3;

        // Add setting for non-portable cluster size.
        if (launchedClusterDimX > 8)
        {
            TLLM_CU_CHECK(mDriver->cuFuncSetAttribute(func, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
                1 // Enable non-portable cluster sizes
                ));
        }

        TLLM_CU_CHECK(mDriver->cuLaunchKernelEx(&launch_config, func, kernelParamsList, nullptr));
    }

    std::pair<uint64_t, std::string> hashFromRunnerParams(
        RunnerParams const& params, SelectKernelParams const& selectKernelParams) const
    {

        // Debug info.
        std::string info = "dtypeQ=" + std::to_string(static_cast<int>(mDtypeQ)) + ", dtypeK="
            + std::to_string(static_cast<int>(mDtypeK)) + ", dtypeV=" + std::to_string(static_cast<int>(mDtypeV))
            + ", dtypeOut=" + std::to_string(static_cast<int>(mDtypeOut)) + ", sm=" + std::to_string(mSM)
            + ", mNumEltsPerSageAttnBlkQ=" + std::to_string(mNumEltsPerSageAttnBlkQ)
            + ", mNumEltsPerSageAttnBlkK=" + std::to_string(mNumEltsPerSageAttnBlkK)
            + ", mNumEltsPerSageAttnBlkP=" + std::to_string(mNumEltsPerSageAttnBlkP)
            + ", mNumEltsPerSageAttnBlkV=" + std::to_string(mNumEltsPerSageAttnBlkV)
            + ", qkvLayout=" + std::to_string(static_cast<int>(params.mQkvLayout))
            + ", maskType=" + std::to_string(static_cast<int>(selectKernelParams.mMaskType))
            + ", kernelType=" + std::to_string(static_cast<int>(selectKernelParams.mKernelType))
            + ", tileScheduler=" + std::to_string(static_cast<int>(selectKernelParams.mTileScheduler))
            + ", multiCtasKvMode=" + std::to_string(static_cast<int>(selectKernelParams.mMultiCtasKvMode))
            + ", headDimPerCtaV=" + std::to_string(selectKernelParams.mHeadDimPerCtaV)
            + ", headDimQk=" + std::to_string(params.mHeadDimQk) + ", headDimV=" + std::to_string(params.mHeadDimV)
            + ", tileSizeQ=" + std::to_string(selectKernelParams.mTileSizeQ)
            + ", tileSizeKv=" + std::to_string(selectKernelParams.mTileSizeKv)
            + ", numTokensPerPage=" + std::to_string(selectKernelParams.mNumTokensPerPage)
            + ", reuseSmemKForV=" + std::to_string(selectKernelParams.mReuseSmemKForV)
            + ", uses2CtaMma=" + std::to_string(selectKernelParams.mUses2CtaMma)
            + ", sparseAttention=" + std::to_string(static_cast<int>(params.mSparseAttention))
            + ", skipsSoftmax=" + std::to_string(selectKernelParams.mSkipsSoftmaxWhenPossible);

        TLLM_LOG_DEBUG("Searching for kernel traits: " + info);

        return std::make_pair(
            hashID(static_cast<int>(params.mQkvLayout), static_cast<int>(selectKernelParams.mMaskType),
                static_cast<int>(selectKernelParams.mKernelType), static_cast<int>(selectKernelParams.mTileScheduler),
                static_cast<int>(selectKernelParams.mMultiCtasKvMode), selectKernelParams.mHeadDimPerCtaV,
                params.mHeadDimQk, params.mHeadDimV, selectKernelParams.mTileSizeQ, selectKernelParams.mTileSizeKv,
                selectKernelParams.mNumTokensPerPage, selectKernelParams.mReuseSmemKForV,
                selectKernelParams.mUses2CtaMma, static_cast<int>(params.mSparseAttention),
                selectKernelParams.mSkipsSoftmaxWhenPossible),
            info);
    }

    tg::CudaArch intToCudaArch(int smVersion) const
    {
        switch (smVersion)
        {
        case 90: return tg::CudaArch::Sm90a;
        case 100: return tg::CudaArch::Sm100a;
        case 103: return tg::CudaArch::Sm103a;
        default: assert(false && "Unsupported CUDA architecture"); return tg::CudaArch::Sm100a;
        }
    }

    int cudaArchToInt(tg::CudaArch cudaArch) const
    {
        switch (cudaArch)
        {
        case tg::CudaArch::Sm90a: return 90;
        case tg::CudaArch::Sm100a: return 100;
        case tg::CudaArch::Sm100f: return 100;
        case tg::CudaArch::Sm103a: return 103;
        default: assert(false && "Unsupported CUDA architecture"); return 100;
        }
    }

    // Is it MLA generation kernel ?
    inline bool isMlaGenKernel(RunnerParams const& params) const
    {
        return (params.mHeadDimQk == 576 && params.mHeadDimV == 512)
            || (isTokenSparse(params.mSparseAttention) && params.mHeadDimQk == 512 && params.mHeadDimV == 512);
    }

    // Compute the number of CTAs in X, Y and Z dimension and the cluster size in the X dimension.
    using CtaInfo = std::tuple<int, int, int, int, int, int>;

    Data_type mDtypeQ, mDtypeK, mDtypeV, mDtypeOut;
    std::shared_ptr<tensorrt_llm::common::CUDADriverWrapper> mDriver;
    KernelMeta const* mKernelMeta;
    unsigned int mKernelMetaCount;
    unsigned int mSM;
    int mNumEltsPerSageAttnBlkQ;
    int mNumEltsPerSageAttnBlkK;
    int mNumEltsPerSageAttnBlkP;
    int mNumEltsPerSageAttnBlkV;
    std::unordered_map<unsigned char const*, CUmodule> mModules;

    struct KernelInfo
    {
        unsigned int mMetaInfoIndex;
        CUfunction mDeviceFunction;
    };

    std::unordered_map<uint64_t, KernelInfo> mFunctions;

    FmhaInterface mFmhaInterface{false, 1};
};

////////////////////////////////////////////////////////////////////////////////////////////////////

class TllmFmhaKernelFactory
{
public:
    using KernelType = TllmGenFmhaKernel;

    KernelType* getKernels(const typename KernelType::KernelMeta* pKernelList, unsigned int nbKernels, Data_type dtypeQ,
        Data_type dtypeK, Data_type dtypeV, Data_type dtypeOut, unsigned int sm, int numEltsPerSageAttnBlkQ = 0,
        int numEltsPerSageAttnBlkK = 0, int numEltsPerSageAttnBlkP = 0, int numEltsPerSageAttnBlkV = 0)
    {
        static std::mutex s_mutex;
        std::lock_guard<std::mutex> lg(s_mutex);
        TLLM_CHECK_WITH_INFO(numEltsPerSageAttnBlkQ <= 64 && numEltsPerSageAttnBlkK <= 64
                && numEltsPerSageAttnBlkP <= 64 && numEltsPerSageAttnBlkV <= 64,
            "SageAttention allows numEltsPerSageAttnBlk up to 64.");

        auto const id = hashID(dtypeQ, dtypeK, dtypeV, dtypeOut, sm, numEltsPerSageAttnBlkQ, numEltsPerSageAttnBlkK,
            numEltsPerSageAttnBlkP, numEltsPerSageAttnBlkV);
        auto const findIter = mKernels.find(id);
        if (findIter == mKernels.end())
        {
            KernelType* newKernel = new KernelType{pKernelList, nbKernels, dtypeQ, dtypeK, dtypeV, dtypeOut, sm,
                numEltsPerSageAttnBlkQ, numEltsPerSageAttnBlkK, numEltsPerSageAttnBlkP, numEltsPerSageAttnBlkV};
            newKernel->loadKernels();
            mKernels.insert(std::make_pair(id, std::unique_ptr<KernelType>(newKernel)));
            return newKernel;
        }
        return findIter->second.get();
    }

    static TllmFmhaKernelFactory& Get()
    {
        static std::unique_ptr<TllmFmhaKernelFactory> sFactory[32] = {nullptr};
        int const deviceId = tensorrt_llm::common::getDevice();
        TLLM_CHECK_WITH_INFO(deviceId < 32, "Invalid deviceId %d (must be < 32)", deviceId);
        if (sFactory[deviceId] == nullptr)
        {
            sFactory[deviceId] = std::make_unique<TllmFmhaKernelFactory>(TllmFmhaKernelFactory());
        }
        return *(sFactory[deviceId]);
    }

private:
    TllmFmhaKernelFactory() = default;

    inline uint64_t hashID(Data_type dtypeQ, Data_type dtypeK, Data_type dtypeV, Data_type dtypeOut, unsigned int sm,
        int numEltsPerSageAttnBlkQ, int numEltsPerSageAttnBlkK, int numEltsPerSageAttnBlkP,
        int numEltsPerSageAttnBlkV) const
    {
        auto const computeLog2BlockSizePlus1 = [](int blockSize) -> int
        {
            if (blockSize <= 0)
            {
                return 0;
            }
            TLLM_CHECK_WITH_INFO((blockSize & (blockSize - 1)) == 0, "SageAttn block size must be a power of 2.");
            return __builtin_ctz(static_cast<unsigned int>(blockSize)) + 1;
        };
        // Format of the hash key:
        // Bit 0  - 15: smVer
        // Bit 16 - 19: dtypeQ
        // Bit 20 - 23: dtypeK
        // Bit 24 - 27: dtypeV
        // Bit 28 - 31: dtypeOut
        // Bit 32 - 34: log2NumEltsPerSageAttnBlkQ + 1 -- 0 for non-sage, max numEltsPerSageAttnBlkQ is 64.
        // Bit 35 - 37: log2NumEltsPerSageAttnBlkK + 1 -- 0 for non-sage, max numEltsPerSageAttnBlkK is 64.
        // Bit 38 - 40: log2NumEltsPerSageAttnBlkP + 1 -- 0 for non-sage, max numEltsPerSageAttnBlkP is 64.
        // Bit 41 - 43: log2NumEltsPerSageAttnBlkV + 1 -- 0 for non-sage, max numEltsPerSageAttnBlkV is 64.
        return static_cast<uint64_t>(sm) | static_cast<uint64_t>(dtypeQ) << 16 | static_cast<uint64_t>(dtypeK) << 20
            | static_cast<uint64_t>(dtypeV) << 24 | static_cast<uint64_t>(dtypeOut) << 28
            | (static_cast<uint64_t>(computeLog2BlockSizePlus1(numEltsPerSageAttnBlkQ)) << 32)
            | (static_cast<uint64_t>(computeLog2BlockSizePlus1(numEltsPerSageAttnBlkK)) << 35)
            | (static_cast<uint64_t>(computeLog2BlockSizePlus1(numEltsPerSageAttnBlkP)) << 38)
            | (static_cast<uint64_t>(computeLog2BlockSizePlus1(numEltsPerSageAttnBlkV)) << 41);
    }

    std::unordered_map<uint64_t, const std::unique_ptr<KernelType>> mKernels;
};

inline TllmGenFmhaKernel* getTllmFmhaKernels(Data_type dtypeQ, Data_type dtypeK, Data_type dtypeV, Data_type dtypeOut,
    unsigned int sm, int numEltsPerSageAttnBlkQ = 0, int numEltsPerSageAttnBlkK = 0, int numEltsPerSageAttnBlkP = 0,
    int numEltsPerSageAttnBlkV = 0)
{

#ifndef EXCLUDE_SM_100F
    return TllmFmhaKernelFactory::Get().getKernels(sTllmGenFmhaKernelMetaInfos,
        sizeof(sTllmGenFmhaKernelMetaInfos) / sizeof(sTllmGenFmhaKernelMetaInfos[0]), dtypeQ, dtypeK, dtypeV, dtypeOut,
        sm, numEltsPerSageAttnBlkQ, numEltsPerSageAttnBlkK, numEltsPerSageAttnBlkP, numEltsPerSageAttnBlkV);
#else
    return nullptr;
#endif // EXCLUDE_SM_100F
}

} // namespace kernels

TRTLLM_NAMESPACE_END
