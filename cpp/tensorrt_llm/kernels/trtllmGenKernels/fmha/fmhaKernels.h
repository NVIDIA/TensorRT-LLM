/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION. All rights reserved.
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
#include <memory>
#include <mutex>
#include <unordered_map>

#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"

#include "cubin/kernelMetaInfo.h"
#include "fmhaReduction.h"
#include "fmhaRunnerParams.h"
#include "kernelParams.h"
#include "prepareCustomMask.h"
#include "tensorrt_llm/kernels/multiHeadAttentionCommon.h"

namespace tc = tensorrt_llm::common;

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

    // Ctor.
    TllmGenFmhaKernel(KernelMeta const* pMetaStart, unsigned int nMetaCount, Data_type dtypeQ, Data_type dtypeKv,
        Data_type dtypeOut, unsigned int smArch)
        : mDtypeQ(dtypeQ)
        , mDtypeKv(dtypeKv)
        , mDtypeOut(dtypeOut)
        , mDriver(tensorrt_llm::common::CUDADriverWrapper::getInstance())
        , mKernelMeta(pMetaStart)
        , mKernelMetaCount(nMetaCount)
        , mSM(smArch)
    {
    }

    void loadKernels()
    {
        // Build a lookup map for all kernels.
        for (unsigned int i = 0; i < mKernelMetaCount; ++i)
        {
            auto const& kernelMeta = mKernelMeta[i];
            if (isSMCompatible(mSM, kernelMeta.mSM) && kernelMeta.mDataTypeQ == mDtypeQ
                && kernelMeta.mDataTypeKv == mDtypeKv && kernelMeta.mDataTypeO == mDtypeOut)
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
                    TLLM_CU_CHECK(mDriver->cuFuncSetAttribute(funcInfo.mDeviceFunction,
                        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, kernelMeta.mSharedMemBytes));
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

    inline uint64_t hashID(int qkvLayout, int maskType, int kernelType, int scheduler, int multiCtasKvMode,
        int headDimPerCtaV, int headDimQk, int headDimV, int tileSizeQ, int tileSizeKv, int numTokensPerPage,
        bool reuseSmemKForV, bool uses2CtaMma, bool sparseMla, bool skipsSoftmax) const
    {
        TLLM_CHECK_WITH_INFO((headDimPerCtaV >= 32) && (headDimQk >= 32) && (headDimV >= 32) && (headDimPerCtaV <= 1024)
                && (headDimQk <= 1024) && (headDimV <= 1024),
            "Expect (32 <= headDim <= 1024), got headDimPerCtaV=%d, headDimQk=%d, "
            "headDimV=%d",
            headDimPerCtaV, headDimQk, headDimV);
        // The numTokensPerPage must be power of 2.
        TLLM_CHECK_WITH_INFO(
            (numTokensPerPage & (numTokensPerPage - 1)) == 0, "The numTokensPerPage must be power of 2.");
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
        // Bit 55 - 55: sparseMla.
        // Bit 56 - 56: skipsSoftmax.
        return (static_cast<uint64_t>(qkvLayout) << 0) | (static_cast<uint64_t>(maskType) << 4)
            | (static_cast<uint64_t>(kernelType) << 8) | (static_cast<uint64_t>(scheduler) << 12)
            | (static_cast<uint64_t>(multiCtasKvMode) << 16) | (static_cast<uint64_t>(headDimPerCtaV >> 3) << 18)
            | (static_cast<uint64_t>(headDimQk >> 3) << 26) | (static_cast<uint64_t>(headDimV >> 3) << 34)
            | (static_cast<uint64_t>(tileSizeKv >> 6) << 42) | (static_cast<uint64_t>(log2(numTokensPerPage)) << 44)
            | (static_cast<uint64_t>(log2(tileSizeQ)) << 49) | (static_cast<uint64_t>(reuseSmemKForV) << 53)
            | (static_cast<uint64_t>(uses2CtaMma) << 54) | (static_cast<uint64_t>(sparseMla) << 55)
            | (static_cast<uint64_t>(skipsSoftmax) << 56);
    }

    uint64_t hashID(KernelMeta const& kernelMeta) const
    {
        return hashID(kernelMeta.mQkvLayout, kernelMeta.mMaskType, kernelMeta.mKernelType, kernelMeta.mTileScheduler,
            kernelMeta.mMultiCtasKvMode, kernelMeta.mHeadDimPerCtaV, kernelMeta.mHeadDimQk, kernelMeta.mHeadDimV,
            kernelMeta.mTileSizeQ, kernelMeta.mTileSizeKv, kernelMeta.mNumTokensPerPage, kernelMeta.mReuseSmemKForV,
            kernelMeta.m2CtaMma, kernelMeta.mSparseMla, kernelMeta.mSkipsSoftmaxWhenPossible);
    }

    std::pair<bool, std::string> checkIfKernelExist(RunnerParams const& params) const
    {
        // Some conditions to check if the kernel is supported.
        // This is meant to avoid occupying unnecessary hashId bits.
        if (params.mHeadDimQk % 8 != 0 || params.mHeadDimV % 8 != 0)
        {
            return std::make_pair(false, "HeadDimQk and HeadDimV must be divisible by 8");
        }

        // The selectKernelParams that might be updated.
        SelectKernelParams selectKernelParams{params};
        // Select the kernel.
        selectKernel(params, selectKernelParams);
        // Hash the runner params.
        auto [hashId, info] = hashFromRunnerParams(params, selectKernelParams);
        return std::make_pair(mFunctions.find(hashId) != mFunctions.end(), info);
    }

    std::pair<CUfunction, KernelMeta> loadKernel(
        RunnerParams const& params, SelectKernelParams const& selectKernelParams) const
    {
        auto [hashId, info] = hashFromRunnerParams(params, selectKernelParams);
        auto const findIter = mFunctions.find(hashId);

        // Add debug info when kernels are not found.
        TLLM_CHECK_WITH_INFO(findIter != mFunctions.end(), "Trtllm-gen kernels not found: " + info);

        auto const& kernelMeta = mKernelMeta[findIter->second.mMetaInfoIndex];
        const CUfunction func = findIter->second.mDeviceFunction;
        // Return the kernel function and kernel meta.
        return std::make_pair(func, kernelMeta);
    }

    void run(RunnerParams const& params) const
    {
        // The selectKernelParams that might be updated.
        SelectKernelParams selectKernelParams{params};
        // The parameters for launching the kernel.
        CtaLaunchParams ctaLaunchParams;
        // The iteration index (used to detect a deadlock of selecting new kernels).
        int selectKernelIter = 0;
        // While loop.
        while (true)
        {
            // Any value >= 2 should work here, but we set it larger in case that we might have more complicated
            // heuristic in the future.
            TLLM_CHECK_WITH_INFO(selectKernelIter < 8, "A deadlock is detected when selecting trtllm-gen kernels.");

            // Select the kernel.
            selectKernel(params, selectKernelParams);
            // Load the kernel.
            auto [func, kernelMeta] = loadKernel(params, selectKernelParams);

            // Compute the number of CTAs in X, Y and Z dimension and the cluster size in the X dimension.
            computeNumCtas(ctaLaunchParams, params, kernelMeta, selectKernelParams);

            // Need to select a new kernel if mSelectNewKernel is true.
            if (selectKernelParams.mSelectNewKernel)
            {
                selectKernelIter++;
                continue;
            }
            // Prepare custom mask for spec-decoding generation kernels.
            if (params.mLayerIdx == 0 && params.mIsSpecDecTree)
            {
                runPrepareCustomMask(kernelMeta, params, params.stream);
            }

            // Prepare the kernel parameters.
            auto kernelParams = KernelParams::setKernelParams(
                params, kernelMeta, ctaLaunchParams.mMaxNumCtasQ, ctaLaunchParams.mMaxNumCtasKv);

            // Prepare kernel parameters list for cuLaunchKernelEx.
            void* kernelParamsList[] = {&kernelParams};
            CUlaunchConfig launch_config;
            launch_config.blockDimX = kernelMeta.mThreadsPerCTA;
            launch_config.blockDimY = 1;
            launch_config.blockDimZ = 1;
            launch_config.gridDimX = ctaLaunchParams.mNumCtasX;
            launch_config.gridDimY = ctaLaunchParams.mNumCtasY;
            launch_config.gridDimZ = ctaLaunchParams.mNumCtasZ;
            launch_config.hStream = params.stream;
            launch_config.sharedMemBytes = kernelMeta.mSharedMemBytes;

            // Debug info.
            TLLM_LOG_DEBUG("TRTLLM-Gen launch info: kernelName = %s", kernelMeta.mFuncName);
            TLLM_LOG_DEBUG(
                "TRTLLM-Gen launch info: maxSeqLenQ = %d, "
                "maxSeqLenKv = %d, "
                "numHeadsQ = %d, "
                "numHeadsKv = %d, batchSize = %d, kernelType = %d",
                params.mMaxSeqLenQ, params.mMaxSeqLenKv, params.mNumHeadsQ, params.mNumHeadsKv, params.mBatchSize,
                static_cast<int>(params.mKernelType));
            TLLM_LOG_DEBUG("TRTLLM-Gen launch info: numCtasX = %d, numCtasY = %d, numCtasZ = %d, clusterDimX = %d",
                ctaLaunchParams.mNumCtasX, ctaLaunchParams.mNumCtasY, ctaLaunchParams.mNumCtasZ,
                ctaLaunchParams.mClusterDimX);

            CUlaunchAttribute launch_attribute[3];
            launch_attribute[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
            launch_attribute[0].value.clusterDim.x = ctaLaunchParams.mClusterDimX;
            launch_attribute[0].value.clusterDim.y = 1;
            launch_attribute[0].value.clusterDim.z = 1;
            launch_attribute[1].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
            launch_attribute[1].value.clusterSchedulingPolicyPreference = ctaLaunchParams.mClusterDimX > 1
                ? CU_CLUSTER_SCHEDULING_POLICY_SPREAD
                : CU_CLUSTER_SCHEDULING_POLICY_DEFAULT;
            launch_attribute[2].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
            launch_attribute[2].value.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();

            launch_config.attrs = launch_attribute;
            launch_config.numAttrs = 3;

            // Add setting for non-portable cluster size.
            if (ctaLaunchParams.mClusterDimX > 8)
            {
                TLLM_CU_CHECK(mDriver->cuFuncSetAttribute(func, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
                    1 // Enable non-portable cluster sizes
                    ));
            }

            // Force using GmemReduction for the multiCtasKvMode if the CgaSmemReduction needs more than one wave (due
            // to the cluster occupancy limit).
            // TODO: find a better heuristic of using CgaSmemReduction.
            if (isCgaSmemReduction(selectKernelParams.mMultiCtasKvMode))
            {
                // The maximum number of active clusters that could co-exist.
                int maxActiveClusters = 1;
                TLLM_CU_CHECK(mDriver->cuOccupancyMaxActiveClusters(&maxActiveClusters, func, &launch_config));
                // Use the GmemReduction instead if it needs more than one wave.
                if (maxActiveClusters * ctaLaunchParams.mClusterDimX
                    < (ctaLaunchParams.mNumCtasX * ctaLaunchParams.mNumCtasY * ctaLaunchParams.mNumCtasZ))
                {
                    selectKernelParams.mForceGmemReduction = true;
                    selectKernelParams.mMultiCtasKvMode = MultiCtasKvMode::GmemReduction;
                    // continue to select a new kernel.
                    continue;
                }
            }

            TLLM_CU_CHECK(mDriver->cuLaunchKernelEx(&launch_config, func, kernelParamsList, nullptr));

            // Run the separate reduction kernel if needed.
            runFmhaReduction(kernelMeta, kernelParams, params.mMultiProcessorCount, params.stream);

            // Break the while op.
            break;
        }
    }

private:
    // Is it MLA generation kernel ?
    inline bool isMlaGenKernel(RunnerParams const& params) const
    {
        return params.mHeadDimQk == 576 && params.mHeadDimV == 512;
    }

    void computeNumCtas(CtaLaunchParams& ctaLaunchParams, RunnerParams const& params, KernelMeta const& kernelMeta,
        SelectKernelParams& selectKernelParams) const
    {
        bool isDsv3MinLatencyMode = params.mBatchSize == 1 && params.mMaxSeqLenQ >= 1 && params.mMaxSeqLenQ <= 16
            && params.mHeadDimQk == 576 && params.mHeadDimV == 512;
        // Do we need to select a new kernel ?
        selectKernelParams.mSelectNewKernel = false;

        // The number of Ctas per Q sequence.
        int numCtasPerSeqQ = (params.mMaxSeqLenQ + kernelMeta.mStepQ - 1) / kernelMeta.mStepQ;
        // The generation-phase kernels might need to group both tokensQ and headsQ into one CTA.
        if (params.mMaxSeqLenQ > 1 && !isContextKernel(params.mKernelType))
        {
            // Each CTA handles one tokenQ by default for spec-decoding generation kernel.
            if (!kernelMeta.mGroupsTokensHeadsQ)
            {
                numCtasPerSeqQ = params.mMaxSeqLenQ;
            }
            else
            {
                // Compute numTokensPerCtaQ where each CTA must process complete numGroupedHeadsQ.
                // Note that each CTA must process complete numHeadsQPerKv.
                int numTokensPerCtaQ = kernelMeta.mStepQ / params.mNumHeadsQPerKv;
                // Group both headsQ and tokensQ into one CTA.
                numCtasPerSeqQ = tc::divUp(params.mMaxSeqLenQ, numTokensPerCtaQ);
            }
        }

        // Compute the grid dimension Y.
        int numHeadsPerCta = kernelMeta.mGroupsHeadsQ ? std::min(params.mNumHeadsQPerKv, kernelMeta.mStepQ) : 1;
        int numCtasForAllHeadsQ = params.mNumHeadsQ / numHeadsPerCta;
        TLLM_CHECK_WITH_INFO(
            numHeadsPerCta * numCtasForAllHeadsQ == params.mNumHeadsQ, "The numHeadsQ/numHeadsKv is not supported.");
        // Take the number of headDim CTAs.
        TLLM_CHECK_WITH_INFO(
            kernelMeta.mHeadDimV % selectKernelParams.mHeadDimPerCtaV == 0, "The headDimPerCtaV is not supported.");
        int numCtasPerHeadDim = kernelMeta.mHeadDimV / selectKernelParams.mHeadDimPerCtaV;
        // Compute the current numCtasX.
        int numCtasX = numCtasPerSeqQ;
        // Update the numCtasY.
        int numCtasY = numCtasForAllHeadsQ * numCtasPerHeadDim;
        // Compute the grid dimension Z.
        int numCtasZ = params.mBatchSize;
        // The 2CtaMma kernels will use 2 Ctas in the x dimension (only used by MLA generation kernels) for heads,
        // so numCtasPerHeadDim and numCtasForAllHeadsQ will be handled by the 2Ctas in the x dimension.
        if (isMlaGenKernel(params) && selectKernelParams.mUses2CtaMma)
        {
            TLLM_CHECK_WITH_INFO(
                numCtasForAllHeadsQ == 2 && numCtasPerHeadDim == 2, "Internal error: numCtasPerHeadDim should be 2.");
            numCtasX *= 2;
            numCtasY /= (numCtasForAllHeadsQ * numCtasPerHeadDim);
        }

        // First split the seqLenKv into multiple CTAs if the utilization is not full.
        // The number of Ctas per KV sequence.
        int numCtasPerSeqKv = 1;
        // Consider the multiCtasKvMode for better GPU utilization.
        if (isMultiCtasKvEnabled(selectKernelParams.mMultiCtasKvMode))
        {
            // The maximum attention window (the maximum number of tokensKv that will be attended to).
            int maxAttentionWindow{params.mMaxSeqLenKv};
            // The sparseMla only selects topK tokensKv.
            if (params.mSparseMla)
            {
                maxAttentionWindow = std::min(params.mMaxSeqLenKv, params.mSparseMlaTopK);
            }
            // Some of the tilesKv will be skipped if the sliding window attention or chunked attention is used.
            if (isSlidingOrChunkedCausalMask(selectKernelParams.mMaskType))
            {
                if (params.mMaxSeqLenKv > params.mAttentionWindowSize)
                {
                    // Consider that the first tileKv might contain tokensKv that is out of the attention window.
                    maxAttentionWindow
                        = std::min(params.mMaxSeqLenKv, params.mAttentionWindowSize + kernelMeta.mTileSizeKv - 1);
                }
                else
                {
                    maxAttentionWindow = std::min(params.mMaxSeqLenKv, params.mChunkedAttentionSize);
                }
            }

            // The maximum number Ctas per Kv sequence, which makes sure that each CtaKv has work to do.
            // The factor of 2 is applied here to ensure the reduction overhead does not outweigh the benefits of a
            // shorter mainloop.
            int const maxNumCtasPerSeqKv = (maxAttentionWindow + 2 * kernelMeta.mStepKv - 1) / (2 * kernelMeta.mStepKv);
            // Compute numCtasPerSeqKv.
            numCtasPerSeqKv = std::min(maxNumCtasPerSeqKv,
                std::max(1, int32_t(params.mMultiProcessorCount / (numCtasX * numCtasY * numCtasZ))));
            // Update the numCtasX.
            numCtasX *= numCtasPerSeqKv;
            // The current total number of CTAs.
            int totalNumCtas = numCtasX * numCtasZ * numCtasY;
            // Disable the multiCtasKvMode if there is only one CtaKv.
            if (numCtasPerSeqKv <= 1)
            {
                selectKernelParams.mMultiCtasKvMode = MultiCtasKvMode::Disabled;
                // Enable the persistent scheduler for better performance.
                selectKernelParams.mTileScheduler = TileScheduler::Persistent;
                // Need to select a different kernel.
                selectKernelParams.mSelectNewKernel = true;
            }
            else if (totalNumCtas < params.mMultiProcessorCount && isMlaGenKernel(params) && !params.mSparseMla
                && selectKernelParams.mTileSizeKv == 128 && tensorrt_llm::common::getEnvUseTileSizeKv64ForTrtllmGen())
            {
                // Use smaller tileSizeKv to fully utilize the SMs.
                selectKernelParams.mTileSizeKv = 64;
                // Need to select a different kernel.
                selectKernelParams.mSelectNewKernel = true;
            }

            // Enable the CgaSmemReduction if the numCtasPerSeqKv <= 16 as the maximum cluster dimension is 16.
            // Only the swapsMmaAbForGeneration kernel supports the CgaSmemReduction for now.
            if (!isDsv3MinLatencyMode && numCtasPerSeqKv > 1 && numCtasPerSeqKv <= 16
                && isSwapsMmaAbForGenerationKernel(selectKernelParams.mKernelType)
                && isGmemReduction(selectKernelParams.mMultiCtasKvMode) && !selectKernelParams.mForceGmemReduction)
            {
                selectKernelParams.mMultiCtasKvMode = MultiCtasKvMode::CgaSmemReduction;
                // Need to select a different kernel.
                selectKernelParams.mSelectNewKernel = true;
            }

            // Disable skipsSoftmax when the sequence length per CTA is less than 1K or the data type is E4M3.
            if (selectKernelParams.mSkipsSoftmaxWhenPossible && !isContextKernel(selectKernelParams.mKernelType))
            {
                // Compute the sequence length per CTA.
                int const seqLenPerCta = (params.mMaxSeqLenKv + numCtasPerSeqKv - 1) / numCtasPerSeqKv;

                if (seqLenPerCta < 1024)
                {
                    // Disable skipsSoftmax.
                    selectKernelParams.mSkipsSoftmaxWhenPossible = false;
                    // Need to select a different kernel.
                    selectKernelParams.mSelectNewKernel = true;
                }
            }

            // Add the debug info when multiCtasKvMode is enabled.
            if (numCtasPerSeqKv > 1)
            {
                TLLM_LOG_DEBUG(
                    "TRTLLM-Gen launch info: multiCtasKvMode is enabled with tileSizeKv = %d, numCtasPerSeqKv = %d, "
                    "numCtasPerSeqQ = "
                    "%d, numCtasY = %d, numCtasZ = %d",
                    selectKernelParams.mTileSizeKv, numCtasPerSeqKv, numCtasPerSeqQ, numCtasY, numCtasZ);
            }
        }

        // The cluster size in the X dimension.
        int clusterDimX = selectKernelParams.mUses2CtaMma ? 2 : 1;
        if (isCgaSmemReduction(selectKernelParams.mMultiCtasKvMode))
        {
            // Note 2CtaMma and CgaSmemReduction cannot be used together currently.
            clusterDimX *= numCtasPerSeqKv;
        }

        // Compute the current number of CTAs in total.
        int totalNumCtas = numCtasX * numCtasZ * numCtasY;

        // Then split the headDimV into multiple CTAs if there are still unused SMs.
        if (isMlaGenKernel(params) && !selectKernelParams.mReuseSmemKForV && !selectKernelParams.mSelectNewKernel
            && !selectKernelParams.mUses2CtaMma)
        {
            // Split the headDimV into multiple CTAs if the utilization is not full.
            // It doesn't work with reuseSmemKForV currently.
            // TODO: find better heuristic of splitting headDimV across multiple CTAs.

            int corrFactor = isDsv3MinLatencyMode ? 1 : 2;
            if (selectKernelParams.mHeadDimPerCtaV == 512 && totalNumCtas * corrFactor <= params.mMultiProcessorCount)
            {
                // Use smaller headDimPerCtaV to fully utilize the SMs.
                selectKernelParams.mHeadDimPerCtaV
                    = totalNumCtas * 2 * corrFactor <= params.mMultiProcessorCount ? 128 : 256;
                // Need to select a different kernel.
                selectKernelParams.mSelectNewKernel = true;
            }
        }

        // Update the parameters for launching the kernel.
        ctaLaunchParams.mMaxNumCtasQ = numCtasPerSeqQ;
        ctaLaunchParams.mMaxNumCtasKv = numCtasPerSeqKv;
        ctaLaunchParams.mNumCtasX = numCtasX;
        ctaLaunchParams.mNumCtasY = numCtasY;
        ctaLaunchParams.mNumCtasZ = numCtasZ;
        ctaLaunchParams.mClusterDimX = clusterDimX;
    }

    // Determine if we should use the SwapsMmaAbForGeneration kernel for MLA generation.
    bool useSwapsMmaAbMlaGenKernel(RunnerParams const& params) const
    {
        // Use the SwapsMmaAbForGeneration kernel for MLA generation when the following conditions are met:
        // 1. The seqLenPerCtaKv <= 1024 based on the benchmark results (this might be fine-tuned later).
        // 2. The numCtas (after splitting the heads across multiple CTAs) <= params.mMultiProcessorCount.

        // The maximum number Ctas per Kv sequence, which makes sure that each CtaKv has work to do.
        // Here we assume the stepKv is 256.
        int const maxNumCtasPerSeqKv = (params.mMaxSeqLenKv + 256 - 1) / 256;
        // The number of Ctas.
        int const numCtas
            = static_cast<int32_t>(params.mBatchSize * params.mMaxSeqLenQ * tc::divUp(params.mNumHeadsQPerKv, 16));
        if (numCtas == 0)
        {
            return false;
        }
        // Compute numCtasPerSeqKv.
        int const numCtasPerSeqKv
            = std::min(maxNumCtasPerSeqKv, std::max(1, int32_t(params.mMultiProcessorCount / numCtas)));
        // Compute the seqLenPerCtaKv.
        int const seqLenPerCtaKv = (params.mMaxSeqLenKv + numCtasPerSeqKv - 1) / numCtasPerSeqKv;
        // Whether we should use the SwapsMmaAbForGeneration kernel for MLA generation.
        return seqLenPerCtaKv <= 1024 && numCtas <= params.mMultiProcessorCount;
    }

    // Selects a heuristic kernel for MLA generation.
    void selectMlaGenerationKernel(RunnerParams const& params, SelectKernelParams& selectKernelParams) const
    {
        // We use the low-latency kernel (SwapsMmaAbForGeneration with tileSizeQ = 16) when any of the following
        // conditions are met:
        // 1. The number of headsQPerKv is <= 32.
        // 2. The number of headsQPerKv is < 128 for sparseMla.
        // 3. The seqLenPerCtaKv <= 1024 based on the benchmark results (this might be fine-tuned later) and
        //    the numCtas (after splitting the heads across multiple CTAs) <= params.mMultiProcessorCount.
        // The sparseMla kernel will always use the 2CTA high-throughput kernel.

        // The kernel type.
        FmhaKernelType& kernelType = selectKernelParams.mKernelType;
        // The tile size for Q.
        int& tileSizeQ = selectKernelParams.mTileSizeQ;

        // Check the conditions.
        if (params.mNumHeadsQPerKv <= 32 || (params.mSparseMla && params.mNumHeadsQPerKv < 128)
            || useSwapsMmaAbMlaGenKernel(params))
        {
            kernelType = FmhaKernelType::SwapsMmaAbForGeneration;
            // Currently, only tileSizeQ = 8 or 16 are supported.
            tileSizeQ = params.mNumHeadsQPerKv <= 8 ? 8 : 16;
        }
        else
        {
            // Otherwise, we use the high-throughput kernel.
            kernelType = FmhaKernelType::KeepsMmaAbForGeneration;
            // Use the tileSizeQ = 64 for MLA high-throughput generation kernels.
            tileSizeQ = 64;
            // Always use the separate reduction kernel.
            if (isMultiCtasKvEnabled(selectKernelParams.mMultiCtasKvMode))
            {
                selectKernelParams.mMultiCtasKvMode = MultiCtasKvMode::GmemReductionWithSeparateKernel;
            }
            // The keepsMmaAbForGeneration sparseMla kernels only support numHeadsQPerKv = 128.
            TLLM_CHECK_WITH_INFO(!params.mSparseMla || params.mNumHeadsQPerKv == 128,
                "The keepsMmaAbForGeneration sparseMla kernels only support numHeadsQPerKv = 128, got %d",
                params.mNumHeadsQPerKv);
            // The 2CTA keepsMmaAbForGeneration kernel is used when the numHeadsQPerKv is 128.
            if (params.mNumHeadsQPerKv == 128)
            {
                selectKernelParams.mUses2CtaMma = true;
                // Each Cta only handles 256 headDimV.
                selectKernelParams.mHeadDimPerCtaV = 256;
            }
        }
    }

    // Selects a heuristic tileSizeQ if groupsTokensHeadsQ is true.
    void selectTileSizeQForGqaGeneration(RunnerParams const& params, SelectKernelParams& selectKernelParams) const
    {

        // Define the per-tile mainloop cost model for different tileSizeQ choices.
        std::unordered_map<int, float> kernelMainloopCost = {
            {128, 2.2}, // Cost factor when tileSizeQ = 128
            {64, 1.68}, // Cost factor when tileSizeQ = 64
            {32, 1.48}, // Cost factor when tileSizeQ = 32
            {16, 1.2},  // Cost factor when tileSizeQ = 16
            {8, 1.0}    // Cost factor when tileSizeQ = 8
        };

        // Define the per-tile reduction cost model for different tileSizeQ choices.
        std::unordered_map<int, float> kernelReductionCost = {
            {128, 1.32}, // Reduction cost factor when tileSizeQ = 128
            {64, 1.2},   // Reduction cost factor when tileSizeQ = 64
            {32, 1.08},  // Reduction cost factor when tileSizeQ = 32
            {16, 1.03},  // Reduction cost factor when tileSizeQ = 16
            {8, 1.0}     // Reduction cost factor when tileSizeQ = 8
        };

        // The reduction cost emulated as a sequence length factor.
        float const kernelReductionSeqLenFactor = 128.0f;

        // The parameters for launching the kernel.
        CtaLaunchParams ctaLaunchParams;
        // The copy of the selectKernelParams, which makes sure it won't modify the original selectKernelParams when
        // computing the number of CTAs.
        SelectKernelParams selectKernelParamsCopy = selectKernelParams;
        // Load the kernel.
        auto [func, kernelMeta] = loadKernel(params, selectKernelParamsCopy);
        // Compute numCtasX, numCtasY and numCtasZ.
        computeNumCtas(ctaLaunchParams, params, kernelMeta, selectKernelParamsCopy);

        // If there are no free SMs or tileSizeQ is already the smallest one, skip the heuristic selection.
        if (ctaLaunchParams.mNumCtasX * ctaLaunchParams.mNumCtasY * ctaLaunchParams.mNumCtasZ * 2
                > params.mMultiProcessorCount
            || selectKernelParamsCopy.mTileSizeQ <= 8)
        {
            // No need to select the kernel further.
            return;
        }

        // Candidate tile sizes for tileSizeQ to explore.
        int const candidateTileSizesQ[] = {128, 64, 32, 16, 8};

        // The default tileSizeQ.
        int defaultTileSizeQ = selectKernelParamsCopy.mTileSizeQ;
        // The selected tileSizeQ.
        int selectedTileSizeQ = selectKernelParamsCopy.mTileSizeQ;

        // The minimum modeling kernel time.
        float globalModelingKernelTime = FLT_MAX;
        // Loop over each candidate tile size.
        for (int tileSizeQ : candidateTileSizesQ)
        {
            // Only consider candidates <= default tileSizeQ.
            if (tileSizeQ > defaultTileSizeQ)
            {
                continue;
            }

            // Update the tileSizeQ.
            selectKernelParamsCopy.mTileSizeQ = tileSizeQ;
            if (tileSizeQ >= 64)
            {
                selectKernelParamsCopy.mKernelType = FmhaKernelType::KeepsMmaAbForGeneration;
            }
            else
            {
                selectKernelParamsCopy.mKernelType = FmhaKernelType::SwapsMmaAbForGeneration;
            }
            // Load the kernel.
            std::tie(func, kernelMeta) = loadKernel(params, selectKernelParamsCopy);

            // Compute the number of CTAs.
            computeNumCtas(ctaLaunchParams, params, kernelMeta, selectKernelParamsCopy);

            // Compute the seqLenPerCtaKv.
            int32_t seqLenPerCtaKv
                = tc::divUp(tc::divUp(params.mMaxSeqLenKv, kernelMeta.mStepKv), ctaLaunchParams.mMaxNumCtasKv)
                * kernelMeta.mStepKv;

            // Compute the modeling kernel time = mainloop cost + reduction cost.
            float modelingKernelTime = kernelMainloopCost[tileSizeQ] * seqLenPerCtaKv
                + kernelReductionCost[tileSizeQ] * kernelReductionSeqLenFactor * ctaLaunchParams.mMaxNumCtasKv;

            // Compute the total number of CTAs.
            int32_t numCtas = ctaLaunchParams.mNumCtasX * ctaLaunchParams.mNumCtasY * ctaLaunchParams.mNumCtasZ;
            // Compute the number of waves.
            int32_t numWaves = tc::divUp(numCtas, params.mMultiProcessorCount);
            // Compute the total modeling kernel time.
            modelingKernelTime *= numWaves;

            // If this candidate has a lower time than the global minimum, update the global minimum.
            if (modelingKernelTime < globalModelingKernelTime)
            {
                globalModelingKernelTime = modelingKernelTime;
                selectedTileSizeQ = tileSizeQ;
            }
        }

        // Update the tileSizeQ.
        selectKernelParams.mTileSizeQ = selectedTileSizeQ;
        // Update the kernel type.
        if (selectKernelParams.mTileSizeQ >= 64)
        {
            selectKernelParams.mKernelType = FmhaKernelType::KeepsMmaAbForGeneration;
        }
        else
        {
            selectKernelParams.mKernelType = FmhaKernelType::SwapsMmaAbForGeneration;
        }
    }

    // Selects a heuristic kernel for GQA generation.
    void selectGqGenerationKernel(RunnerParams const& params, SelectKernelParams& selectKernelParams) const
    {

        // The kernel type.
        FmhaKernelType& kernelType = selectKernelParams.mKernelType;
        // The tile size for Q.
        int& tileSizeQ = selectKernelParams.mTileSizeQ;

        // Check the conditions.
        if (params.mIsSpecDecTree)
        {

            bool isSupported = params.mNumHeadsQPerKv <= 16 && (params.mHeadDimQk == 64 || params.mHeadDimQk == 128);
            if (isSupported)
            {
                kernelType = FmhaKernelType::KeepsMmaAbForGeneration;
                // Only support tileSizeQ = 128 for tree-based speculative decoding.
                tileSizeQ = 128;
            }
            else
            {
                TLLM_LOG_ERROR(
                    "Tree-based speculative decoding is not supported with numHeadsQPerKv = %d and headDimQk = %d "
                    "by TRTLLM-GEN",
                    params.mNumHeadsQPerKv, params.mHeadDimQk);
            }

            // No need to select the kernel further.
            return;
        }

        // Mixed precision kernels don't work with groupsTokensHeadsQ = true for now.
        if (mDtypeQ != mDtypeKv || mDtypeOut == DATA_TYPE_E2M1)
        {
            tileSizeQ = params.mNumHeadsQPerKv <= 8 ? 8 : 16;
            kernelType = FmhaKernelType::SwapsMmaAbForGeneration;
            return;
        }

        // The number of tokensQ and headsQ that can be grouped into one CTA.
        int numTokensHeadsQ = params.mNumHeadsQPerKv * params.mMaxSeqLenQ;
        // When numHeadsQPerKv >= 64, use KeepsMmaAbForGeneration kernel.
        if (numTokensHeadsQ <= 8)
        {
            tileSizeQ = 8;
            kernelType = FmhaKernelType::SwapsMmaAbForGeneration;
        }
        else if (numTokensHeadsQ <= 16)
        {
            tileSizeQ = 16;
            kernelType = FmhaKernelType::SwapsMmaAbForGeneration;
        }
        else if (numTokensHeadsQ <= 32)
        {
            tileSizeQ = 32;
            kernelType = FmhaKernelType::SwapsMmaAbForGeneration;
        }
        else if (numTokensHeadsQ <= 64)
        {
            tileSizeQ = 64;
            kernelType = FmhaKernelType::KeepsMmaAbForGeneration;
        }
        else
        {
            tileSizeQ = 128;
            kernelType = FmhaKernelType::KeepsMmaAbForGeneration;
        }

        // When maxSeqLenQ > 1, use an experimental kernel-timing model to select the best kernel that groups both
        // tokensQ and headsQ into one CTA.
        if (params.mMaxSeqLenQ > 1)
        {
            selectTileSizeQForGqaGeneration(params, selectKernelParams);
        }
    }

    // Select a kernel based on the heuristic.
    void selectKernel(RunnerParams const& params, SelectKernelParams& selectKernelParams) const
    {

        // Select the kernel based on the kernel type.
        if (isGenerationKernel(params.mKernelType) && isMlaGenKernel(params))
        {
            selectMlaGenerationKernel(params, selectKernelParams);
        }
        else if (isGenerationKernel(params.mKernelType))
        {
            selectGqGenerationKernel(params, selectKernelParams);
        }

        // Enable sliding window or chunked causal if the max kv sequence length exceeds attention window size or
        // chunked attention size.
        // This is supported by causal-mask context kernels and generation-phase kernels.
        if ((selectKernelParams.mMaskType == TrtllmGenAttentionMaskType::Causal || !isContextKernel(params.mKernelType))
            && (params.mMaxSeqLenKv > params.mAttentionWindowSize || params.mChunkedAttentionSize != INT_MAX))
        {
            TLLM_CHECK_WITH_INFO(params.mMaxSeqLenKv <= params.mAttentionWindowSize
                    || params.mMaxSeqLenKv <= params.mChunkedAttentionSize,
                "Sliding window attention and chunked attention should not be used together");
            selectKernelParams.mMaskType = TrtllmGenAttentionMaskType::SlidingOrChunkedCausal;
        }

        // SparseMla kernels use a fixed numTokensPerPage = 1.
        if (params.mSparseMla)
        {
            selectKernelParams.mNumTokensPerPage = 1;
        }
        else if (!isPagedKv(params.mQkvLayout))
        {
            // NumTokensPerPage is set to 0 when not selecting pagedKv-layout kernels.
            selectKernelParams.mNumTokensPerPage = 0;
        }
    }

    std::pair<uint64_t, std::string> hashFromRunnerParams(
        RunnerParams const& params, SelectKernelParams const& selectKernelParams) const
    {

        // Debug info.
        std::string info = "dtypeQ=" + std::to_string(static_cast<int>(mDtypeQ)) + ", dtypeKv="
            + std::to_string(static_cast<int>(mDtypeKv)) + ", dtypeOut=" + std::to_string(static_cast<int>(mDtypeOut))
            + ", sm=" + std::to_string(mSM) + ", qkvLayout=" + std::to_string(static_cast<int>(params.mQkvLayout))
            + ", maskType=" + std::to_string(static_cast<int>(selectKernelParams.mMaskType))
            + ", kernelType=" + std::to_string(static_cast<int>(selectKernelParams.mKernelType))
            + ", tileScheduler=" + std::to_string(static_cast<int>(selectKernelParams.mTileScheduler))
            + ", multiCtasKvMode=" + std::to_string(static_cast<int>(selectKernelParams.mMultiCtasKvMode))
            + ", headDimPerCtaV=" + std::to_string(selectKernelParams.mHeadDimPerCtaV)
            + ", headDimQk=" + std::to_string(params.mHeadDimQk) + ", headDimV=" + std::to_string(params.mHeadDimV)
            + ", tileSizeQ=" + std::to_string(selectKernelParams.mTileSizeQ)
            + ", tileSizeKv=" + std::to_string(selectKernelParams.mTileSizeKv)
            + ", numTokensPerPage=" + std::to_string(selectKernelParams.mNumTokensPerPage)
            + ", reuseSmemKForV=" + std::to_string(selectKernelParams.mReuseSmemKForV) + ", uses2CtaMma="
            + std::to_string(selectKernelParams.mUses2CtaMma) + ", sparseMla=" + std::to_string(params.mSparseMla)
            + ", skipsSoftmax=" + std::to_string(selectKernelParams.mSkipsSoftmaxWhenPossible);

        TLLM_LOG_DEBUG("Searching for kernel traits: " + info);

        return std::make_pair(
            hashID(static_cast<int>(params.mQkvLayout), static_cast<int>(selectKernelParams.mMaskType),
                static_cast<int>(selectKernelParams.mKernelType), static_cast<int>(selectKernelParams.mTileScheduler),
                static_cast<int>(selectKernelParams.mMultiCtasKvMode), selectKernelParams.mHeadDimPerCtaV,
                params.mHeadDimQk, params.mHeadDimV, selectKernelParams.mTileSizeQ, selectKernelParams.mTileSizeKv,
                selectKernelParams.mNumTokensPerPage, selectKernelParams.mReuseSmemKForV,
                selectKernelParams.mUses2CtaMma, params.mSparseMla, selectKernelParams.mSkipsSoftmaxWhenPossible),
            info);
    }

    Data_type mDtypeQ, mDtypeKv, mDtypeOut;
    std::shared_ptr<tensorrt_llm::common::CUDADriverWrapper> mDriver;
    KernelMeta const* mKernelMeta;
    unsigned int mKernelMetaCount;
    unsigned int mSM;
    std::unordered_map<unsigned char const*, CUmodule> mModules;

    struct KernelInfo
    {
        unsigned int mMetaInfoIndex;
        CUfunction mDeviceFunction;
    };

    std::unordered_map<uint64_t, KernelInfo> mFunctions;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

class TllmFmhaKernelFactory
{
public:
    using KernelType = TllmGenFmhaKernel;

    KernelType const* getKernels(const typename KernelType::KernelMeta* pKernelList, unsigned int nbKernels,
        Data_type dtypeQ, Data_type dtypeKv, Data_type dtypeOut, unsigned int sm)
    {
        static std::mutex s_mutex;
        std::lock_guard<std::mutex> lg(s_mutex);

        auto const id = hashID(dtypeQ, dtypeKv, dtypeOut, sm);
        auto const findIter = mKernels.find(id);
        if (findIter == mKernels.end())
        {
            KernelType* newKernel = new KernelType{pKernelList, nbKernels, dtypeQ, dtypeKv, dtypeOut, sm};
            newKernel->loadKernels();
            mKernels.insert(std::make_pair(id, std::unique_ptr<KernelType>(newKernel)));
            return newKernel;
        }
        return findIter->second.get();
    }

    static TllmFmhaKernelFactory& Get()
    {
        int deviceId;
        cudaGetDevice(&deviceId);
        static std::unique_ptr<TllmFmhaKernelFactory> sFactory[32] = {nullptr};
        if (sFactory[deviceId] == nullptr)
        {
            TLLM_CHECK_WITH_INFO(deviceId < 32, "Invalid deviceId %d", deviceId);
            sFactory[deviceId] = std::make_unique<TllmFmhaKernelFactory>(TllmFmhaKernelFactory());
        }
        return *(sFactory[deviceId]);
    }

private:
    TllmFmhaKernelFactory() = default;

    inline uint64_t hashID(Data_type dtypeQ, Data_type dtypeKv, Data_type dtypeOut, unsigned int sm) const
    {
        return static_cast<uint64_t>(sm) | static_cast<uint64_t>(dtypeQ) << 16 | static_cast<uint64_t>(dtypeKv) << 20
            | static_cast<uint64_t>(dtypeOut) << 24;
    }

    std::unordered_map<uint64_t, const std::unique_ptr<KernelType>> mKernels;
};

inline TllmGenFmhaKernel const* getTllmFmhaKernels(
    Data_type dtypeQ, Data_type dtypeKv, Data_type dtypeOut, unsigned int sm)
{

#if !defined(EXCLUDE_SM_100) || !defined(EXCLUDE_SM_103)
    return TllmFmhaKernelFactory::Get().getKernels(sTllmGenFmhaKernelMetaInfos,
        sizeof(sTllmGenFmhaKernelMetaInfos) / sizeof(sTllmGenFmhaKernelMetaInfos[0]), dtypeQ, dtypeKv, dtypeOut, sm);
#else
    return nullptr;
#endif // EXCLUDE_SM_100
}

} // namespace kernels

TRTLLM_NAMESPACE_END
