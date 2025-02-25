/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
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
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/logger.h"

#include "cubin/kernelMetaInfo.h"
#include "fmhaRunnerParams.h"
#include "kernelParams.h"

namespace tensorrt_llm
{
namespace kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////

class TllmGenFmhaKernel
{
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
            if (static_cast<unsigned int>(kernelMeta.mSM) == mSM && kernelMeta.mDataTypeQ == mDtypeQ
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
                TLLM_CHECK_WITH_INFO(mFunctions.find(hashID(kernelMeta)) == mFunctions.end(),
                    "The kernel's hashId has conflicts with others.");
                mFunctions.insert(std::make_pair(hashID(kernelMeta), funcInfo));
            }
        }
    }

    inline uint64_t hashID(int qkvLayout, int maskType, int kernelType, int scheduler, int multiCtasKvMode,
        int headDimPerCtaV, int headDimQk, int headDimV, int numTokensPerPage, int maxNumHeadsQPerKvInCta,
        bool reuseSmemKForV) const
    {
        TLLM_CHECK_WITH_INFO((headDimPerCtaV >= 32) && (headDimQk >= 32) && (headDimV >= 32) && (headDimPerCtaV <= 2048)
                && (headDimQk <= 2048) && (headDimV <= 2048) && (numTokensPerPage <= 128),
            "Expect (32 <= headDim <= 2048) && (numTokensPerPage <= 128), got headDimPerCtaV=%d, headDimQk=%d, "
            "headDimV=%d, numTokensPerPage=%d",
            headDimPerCtaV, headDimQk, headDimV, numTokensPerPage);
        TLLM_CHECK_WITH_INFO(maxNumHeadsQPerKvInCta <= 128, "The maxNumHeadsQPerKvInCta <= 128 is required.");
        // Format of the hash key:
        // Bit 0  - 3 : qkvLayout.
        // Bit 4  - 7 : maskType.
        // Bit 8  - 11: kernelType.
        // Bit 12 - 15: tileScheduler.
        // Bit 16 - 16: multiCtasKvMode.
        // Bit 17 - 23: (headDimPerCtaV >> 5).
        // Bit 24 - 30: (headDimQk >> 5).
        // Bit 31 - 37: (headDimV >> 5).
        // Bit 38 - 45: numTokensPerPage
        // Bit 46 - 53: maxNumHeadsQPerKvInCta.
        // Bit 53 - 54: reuseSmemKForV.
        return (static_cast<uint64_t>(qkvLayout) << 0) | (static_cast<uint64_t>(maskType) << 4)
            | (static_cast<uint64_t>(kernelType) << 8) | (static_cast<uint64_t>(scheduler) << 12)
            | (static_cast<uint64_t>(multiCtasKvMode) << 16) | (static_cast<uint64_t>(headDimPerCtaV >> 5) << 17)
            | (static_cast<uint64_t>(headDimQk >> 5) << 24) | (static_cast<uint64_t>(headDimV >> 5) << 31)
            | (static_cast<uint64_t>(numTokensPerPage) << 38) | (static_cast<uint64_t>(maxNumHeadsQPerKvInCta) << 46)
            | (static_cast<uint64_t>(reuseSmemKForV) << 53);
    }

    uint64_t hashID(KernelMeta const& kernelMeta) const
    {
        return hashID(kernelMeta.mQkvLayout, kernelMeta.mMaskType, kernelMeta.mKernelType, kernelMeta.mTileScheduler,
            kernelMeta.mMultiCtasKvMode, kernelMeta.mHeadDimPerCtaV, kernelMeta.mHeadDimQk, kernelMeta.mHeadDimV,
            kernelMeta.mNumTokensPerPage, kernelMeta.mMaxNumHeadsQPerKvInCta, kernelMeta.mReuseSmemKForV);
    }

    std::pair<bool, std::string> checkIfKernelExist(RunnerParams const& params) const
    {
        // The selectKernelParams that might be updated.
        SelectKernelParams selectKernelParams{params};
        auto [hashId, info] = hashFromRunnerParams(params, selectKernelParams);
        return std::make_pair(mFunctions.find(hashId) != mFunctions.end(), info);
    }

    void run(RunnerParams const& params) const
    {
        // The selectKernelParams that might be updated.
        SelectKernelParams selectKernelParams{params};
        // The iteration index (used to detect a deadlock of selecting new kernels).
        int selectKernelIter = 0;
        // While loop.
        while (true)
        {
            // Any value >= 2 should work here, but we set it larger in case that we might have more complicated
            // heuristic in the future.
            TLLM_CHECK_WITH_INFO(selectKernelIter < 8, "A deadlock is detected when selecting trtllm-gen kernels.");
            auto [hashId, info] = hashFromRunnerParams(params, selectKernelParams);
            auto const findIter = mFunctions.find(hashId);

            // Add debug info when kernels are not found.
            TLLM_CHECK_WITH_INFO(findIter != mFunctions.end(), "Trtllm-gen kernels not found: " + info);

            auto const& kernelMeta = mKernelMeta[findIter->second.mMetaInfoIndex];
            const CUfunction func = findIter->second.mDeviceFunction;

            // Compute the number of CTAs in X, Y and Z dimension.
            auto [numCtasX, numCtasY, numCtasZ] = computeNumCtas(params, kernelMeta, selectKernelParams);

            // Need to select a new kernel if mSelectNewKernel is true.
            if (selectKernelParams.mSelectNewKernel)
            {
                selectKernelIter++;
                continue;
            }

            // Prepare the kernel parameters.
            auto kernelParams = KernelParams::setKernelParams(params, kernelMeta);

            // Prepare kernel parameters list for cuLaunchKernelEx.
            void* kernelParamsList[] = {&kernelParams};
            CUlaunchConfig launch_config;
            launch_config.blockDimX = kernelMeta.mThreadsPerCTA;
            launch_config.blockDimY = 1;
            launch_config.blockDimZ = 1;
            launch_config.gridDimX = numCtasX;
            launch_config.gridDimY = numCtasY;
            launch_config.gridDimZ = numCtasZ;
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

            CUlaunchAttribute launch_attribute[2];
            launch_attribute[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
            launch_attribute[0].value.clusterDim.x = 1;
            launch_attribute[0].value.clusterDim.y = 1;
            launch_attribute[0].value.clusterDim.z = 1;
            launch_attribute[1].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
            launch_attribute[1].value.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_DEFAULT;

            launch_config.attrs = launch_attribute;
            launch_config.numAttrs = 2;

            TLLM_CU_CHECK(mDriver->cuLaunchKernelEx(&launch_config, func, kernelParamsList, nullptr));
            // Break the while op.
            break;
        }
    }

private:
    // Is it MLA generation kernel ?
    inline bool isMlaGenKernel(KernelMeta const& kernelMeta) const
    {
        return kernelMeta.mHeadDimQk == 576 && kernelMeta.mHeadDimV == 512;
    }

    // Compute the number of CTAs in X, Y and Z dimension.
    std::tuple<int, int, int> computeNumCtas(
        RunnerParams const& params, KernelMeta const& kernelMeta, SelectKernelParams& selectKernelParams) const
    {
        // Do we need to select a new kernel ?
        selectKernelParams.mSelectNewKernel = false;

        // The number of Ctas per Q sequence.
        int numCtasPerSeqQ = (params.mMaxSeqLenQ + kernelMeta.mStepQ - 1) / kernelMeta.mStepQ;

        // Compute the grid dimension Y.
        int numHeadsPerCta
            = kernelMeta.mGroupsHeadsQ ? std::min(params.mNumHeadsQPerKv, kernelMeta.mMaxNumHeadsQPerKvInCta) : 1;
        int numCtasForAllHeadsQ = params.mNumHeadsQ / numHeadsPerCta;
        TLLM_CHECK_WITH_INFO(
            numHeadsPerCta * numCtasForAllHeadsQ == params.mNumHeadsQ, "The numHeadsQ/numHeadsKv is not supported.");
        // Take the number of headDim CTAs.
        TLLM_CHECK_WITH_INFO(
            kernelMeta.mHeadDimV % selectKernelParams.mHeadDimPerCtaV == 0, "The headDimPerCtaV is not supported.");
        int numCtasPerHeadDim = kernelMeta.mHeadDimV / selectKernelParams.mHeadDimPerCtaV;
        // Update the numCtasY.
        int numCtasY = numCtasForAllHeadsQ * numCtasPerHeadDim;

        // Compute the grid dimension Z.
        int numCtasZ = params.mBatchSize;
        // MTP kernels use different blockY to process MTP tokens.
        if (isMlaGenKernel(kernelMeta) && params.mMaxSeqLenQ > 1)
        {
            numCtasZ *= params.mMaxSeqLenQ;
            numCtasPerSeqQ = 1;
        }

        // Compute the current number of CTAs in total.
        int totalNumCtasYz = numCtasZ * numHeadsPerCta;

        // The heuristic to select MLA kernels.
        if (isMlaGenKernel(kernelMeta) && !selectKernelParams.mReuseSmemKForV)
        {
            // Split the headDimV into multiple CTAs if the utilization is not full.
            // TODO: find better heuristic of splitting headDimV across multiple CTAs.
            if (selectKernelParams.mHeadDimPerCtaV == 512 && totalNumCtasYz * 4 < params.mMultiProcessorCount)
            {
                selectKernelParams.mHeadDimPerCtaV = 128;
                // Need to select a different kernel.
                selectKernelParams.mSelectNewKernel = true;
            }
            // TODO: find better heuristic of enabling reuseSmemKForV.
            else if (selectKernelParams.mHeadDimPerCtaV == 512 && numCtasForAllHeadsQ == 1)
            {
                // It seems that enabling reuseSmemKForV has worse perf when there are multiple CTAs for different
                // headsQ.
                selectKernelParams.mReuseSmemKForV = true;
                // Need to select a different kernel.
                selectKernelParams.mSelectNewKernel = true;
            }
        }

        // The number of Ctas per KV sequence.
        int numCtasPerSeqKv = 1;
        // Consider the multiCtasKvMode for better GPU utilization.
        if (selectKernelParams.mMultiCtasKvMode)
        {
            // The maximum number Ctas per Kv sequence, which makes sure that each CtaKv has work to do.
            int const maxNumCtasPerSeqKv = (params.mMaxSeqLenKv + kernelMeta.mStepKv - 1) / kernelMeta.mStepKv;
            // Compute numCtasPerSeqKv.
            numCtasPerSeqKv = std::min(
                maxNumCtasPerSeqKv, int32_t(params.mMultiProcessorCount / (numCtasPerSeqQ * numCtasY * numCtasZ)));
            // Reset multiCtasKvMode to false.
            if (numCtasPerSeqKv <= 1)
            {
                selectKernelParams.mMultiCtasKvMode = false;
                // Enable the persistent scheduler for better performance.
                selectKernelParams.mTileScheduler = TileScheduler::Persistent;
                // Need to select a different kernel.
                selectKernelParams.mSelectNewKernel = true;
            }
            else
            {
                TLLM_LOG_DEBUG(
                    "TRTLLM-Gen launch info: multiCtasKvMode is enabled with numCtasPerSeqKv = %d, numCtasPerSeqQ = "
                    "%d, numCtasY = %d, numCtasZ = %d",
                    numCtasPerSeqKv, numCtasPerSeqQ, numCtasY, numCtasZ);
            }
        }

        // Compute the grid dimension X.
        int numCtasX = numCtasPerSeqQ * numCtasPerSeqKv;

        // Return the number of CTAs for X, Y and Z dimension.
        return std::make_tuple(numCtasX, numCtasY, numCtasZ);
    }

    std::pair<uint64_t, std::string> hashFromRunnerParams(
        RunnerParams const& params, SelectKernelParams& selectKernelParams) const
    {
        // The updated kernel type.
        FmhaKernelType kernelType = params.mKernelType;
        // Generation kernelType will use either SwapsMmaAbForGeneration or KeepsMmaAbForGeneration.
        if (isGenerationKernel(params.mKernelType))
        {
            kernelType = (params.mNumHeadsQPerKv <= 16 && params.mHeadDimQk != 32)
                ? FmhaKernelType::SwapsMmaAbForGeneration
                : FmhaKernelType::KeepsMmaAbForGeneration;
        }

        // The maximum number of headsQPerKv that the kernel can support in one Cta.
        int& maxNumHeadsQPerKvInCta = selectKernelParams.mMaxNumHeadsQPerKvInCta;
        if (isSwapsMmaAbForGenerationKernel(kernelType))
        {
            // Set maxNumHeadsQPerKvInCta automatically if not set.
            if (maxNumHeadsQPerKvInCta == 1)
            {
                maxNumHeadsQPerKvInCta = (params.mNumHeadsQPerKv <= 8) ? 8 : 16;
            }
            TLLM_CHECK_WITH_INFO((maxNumHeadsQPerKvInCta == 8 || maxNumHeadsQPerKvInCta == 16)
                    && (params.mNumHeadsQPerKv < maxNumHeadsQPerKvInCta
                        || params.mNumHeadsQPerKv % maxNumHeadsQPerKvInCta == 0),
                "Not supported");
        }
        else if (isKeepsMmaAbForGenerationKernel(kernelType))
        {
            // Set maxNumHeadsQPerKvInCta automatically if not set.
            if (maxNumHeadsQPerKvInCta == 1)
            {
                maxNumHeadsQPerKvInCta = 32;
            }
            TLLM_CHECK_WITH_INFO(maxNumHeadsQPerKvInCta == 32
                    && (params.mNumHeadsQPerKv < maxNumHeadsQPerKvInCta
                        || params.mNumHeadsQPerKv % maxNumHeadsQPerKvInCta == 0),
                "Not supported");
        }
        else if (isContextKernel(kernelType))
        {
            TLLM_CHECK_WITH_INFO(maxNumHeadsQPerKvInCta == 1, "Not supported");
        }

        // The mask type.
        TrtllmGenAttentionMaskType maskType = params.mMaskType;
        // Enable sliding window causal if the max kv sequence length exceeds attention window size.
        if (params.mAttentionWindowSize < params.mMaxSeqLenKv && maskType == TrtllmGenAttentionMaskType::Causal)
        {
            maskType = TrtllmGenAttentionMaskType::SlidingWindowCausal;
        }
        // NumTokensPerPage is set to 0 when not selecting pagedKv-layout kernels.
        int numTokensPerPage = (!isPagedKv(params.mQkvLayout)) ? 0 : params.mNumTokensPerPage;

        // Debug info.
        std::string info = "qkvLayout=" + std::to_string(static_cast<int>(params.mQkvLayout))
            + ", maskType=" + std::to_string(static_cast<int>(maskType))
            + ", kernelType=" + std::to_string(static_cast<int>(kernelType))
            + ", tileScheduler=" + std::to_string(static_cast<int>(selectKernelParams.mTileScheduler))
            + ", multiCtasKvMode=" + std::to_string(selectKernelParams.mMultiCtasKvMode) + ", headDimPerCtaV="
            + std::to_string(selectKernelParams.mHeadDimPerCtaV) + ", headDimQk=" + std::to_string(params.mHeadDimQk)
            + ", headDimV=" + std::to_string(params.mHeadDimV) + ", numTokensPerPage="
            + std::to_string(numTokensPerPage) + ", maxNumHeadsQPerKvInCta=" + std::to_string(maxNumHeadsQPerKvInCta)
            + ", reuseSmemKForV=" + std::to_string(selectKernelParams.mReuseSmemKForV);
        TLLM_LOG_DEBUG("Searching for kernel traits: " + info);

        return std::make_pair(
            hashID(static_cast<int>(params.mQkvLayout), static_cast<int>(maskType), static_cast<int>(kernelType),
                static_cast<int>(selectKernelParams.mTileScheduler), selectKernelParams.mMultiCtasKvMode,
                selectKernelParams.mHeadDimPerCtaV, params.mHeadDimQk, params.mHeadDimV, numTokensPerPage,
                maxNumHeadsQPerKvInCta, selectKernelParams.mReuseSmemKForV),
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

#ifndef EXCLUDE_SM_100
    return TllmFmhaKernelFactory::Get().getKernels(sTllmGenFmhaKernelMetaInfos,
        sizeof(sTllmGenFmhaKernelMetaInfos) / sizeof(sTllmGenFmhaKernelMetaInfos[0]), dtypeQ, dtypeKv, dtypeOut, sm);
#else
    return nullptr;
#endif // EXCLUDE_SM_100
}

} // namespace kernels
} // namespace tensorrt_llm
