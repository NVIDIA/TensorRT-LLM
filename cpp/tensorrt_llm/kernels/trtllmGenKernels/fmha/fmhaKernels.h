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
                mFunctions.insert(std::make_pair(hashID(kernelMeta), funcInfo));
            }
        }
    }

    inline uint64_t hashID(int qkvLayout, int maskType, int kernelType, int scheduler, int multiCtasKvMode, int headDim,
        int numTokensPerPage, int maxNumHeadsQPerKv) const
    {
        TLLM_CHECK_WITH_INFO((headDim <= 2048) && (numTokensPerPage <= 128),
            "Expect (headDim <= 2048) && (numTokensPerPage <= 128), got headDim=%d, numTokensPerPage=%d", headDim,
            numTokensPerPage);
        TLLM_CHECK_WITH_INFO(maxNumHeadsQPerKv <= 128, "The maxNumHeadsQPerKv <= 128 is required.");
        // Format of the hash key:
        // Bit 0  - 3 : qkvLayout.
        // Bit 4  - 7 : maskType.
        // Bit 8  - 11: kernelType.
        // Bit 12 - 15: tileScheduler.
        // Bit 16 - 16: multiCtasKvMode.
        // Bit 17 - 27: (headDim - 1).
        // Bit 28 - 35: numTokensPerPage
        // Bit 36 - 43: maxNumHeadsQPerKv.
        return (static_cast<uint64_t>(qkvLayout) << 0) | (static_cast<uint64_t>(maskType) << 4)
            | (static_cast<uint64_t>(kernelType) << 8) | (static_cast<uint64_t>(scheduler) << 12)
            | (static_cast<uint64_t>(multiCtasKvMode) << 16) | (static_cast<uint64_t>(headDim - 1) << 17)
            | (static_cast<uint64_t>(numTokensPerPage) << 28) | (static_cast<uint64_t>(maxNumHeadsQPerKv) << 36);
    }

    uint64_t hashID(KernelMeta const& kernelMeta) const
    {
        return hashID(kernelMeta.mQkvLayout, kernelMeta.mMaskType, kernelMeta.mKernelType, kernelMeta.mTileScheduler,
            kernelMeta.mMultiCtasKvMode, kernelMeta.mHeadDim, kernelMeta.mNumTokensPerPage,
            kernelMeta.mMaxNumHeadsQPerKv);
    }

    std::pair<bool, std::string> checkIfKernelExist(RunnerParams const& params) const
    {
        auto [hashId, info] = hashFromRunnerParams(params);
        return std::make_pair(mFunctions.find(hashId) != mFunctions.end(), info);
    }

    void run(RunnerParams const& params) const
    {
        auto [hashId, info] = hashFromRunnerParams(params);
        auto const findIter = mFunctions.find(hashId);

        // Add debug info when kernels are not found.
        TLLM_CHECK_WITH_INFO(findIter != mFunctions.end(), "Trtllm-gen kernels not found: " + info);

        auto const& kernelMeta = mKernelMeta[findIter->second.mMetaInfoIndex];
        const CUfunction func = findIter->second.mDeviceFunction;

        // Prepare the kernel parameters.
        auto kernelParams = KernelParams::setKernelParams(params, kernelMeta);

        // Prepare kernel parameters list for cuLaunchKernelEx.
        void* kernelParamsList[] = {&kernelParams};
        CUlaunchConfig launch_config;
        launch_config.blockDimX = kernelMeta.mThreadsPerCTA;
        launch_config.blockDimY = 1;
        launch_config.blockDimZ = 1;
        launch_config.gridDimY = kernelMeta.mGroupsHeadsQ ? params.mNumHeadsKv : params.mNumHeadsQ;
        launch_config.gridDimZ = params.mBatchSize;
        launch_config.hStream = params.stream;
        launch_config.sharedMemBytes = kernelMeta.mSharedMemBytes;

        // The number of Ctas per Q sequence.
        int numCtasPerSeqQ = (params.mMaxSeqLenQ + kernelMeta.mStepQ - 1) / kernelMeta.mStepQ;
        int numCtasPerSeqKv = 1;
        // Ccompute the numCtasPerSeqKv.
        if (params.mMultiCtasKvMode)
        {
            // The maximum number Ctas per Kv sequence, which makes sure that each CtaKv has work to do.
            int const maxNumCtasPerSeqKv = (params.mMaxSeqLenKv + kernelMeta.mStepKv - 1) / kernelMeta.mStepKv;
            // Compute numCtasPerSeqKv.
            numCtasPerSeqKv = std::min(maxNumCtasPerSeqKv,
                int32_t(params.mMaxNumCtas / (numCtasPerSeqQ * launch_config.gridDimY * launch_config.gridDimZ)));
            // Reset params.mMultiCtasKvMode to false.
            if (numCtasPerSeqKv <= 1)
            {
                RunnerParams updatedParams = params;
                updatedParams.mMultiCtasKvMode = false;
                // Enable the persistent scheduler for better performance.
                updatedParams.mTileScheduler = TileScheduler::Persistent;
                // Relaunch the run function.
                run(updatedParams);
                return;
            }
        }

        // Set gridDim.x = numCtasPerSeqQ * numCtasPerSeqKv.
        launch_config.gridDimX = numCtasPerSeqQ * numCtasPerSeqKv;
        TLLM_LOG_DEBUG(
            "TRTLLM-Gen launch info: kernelName = %s, numCtasPerSeqQ = %d, numCtasPerSeqKv = %d, maxSeqLenQ = %d, "
            "maxSeqLenKv = %d, "
            "numHeadsQ = %d, "
            "numHeadsKv = %d, batchSize = %d, kernelType = %d",
            kernelMeta.mFuncName, numCtasPerSeqQ, numCtasPerSeqKv, params.mMaxSeqLenQ, params.mMaxSeqLenKv,
            params.mNumHeadsQ, params.mNumHeadsKv, params.mBatchSize, static_cast<int>(params.mKernelType));

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
    }

private:
    std::pair<uint64_t, std::string> hashFromRunnerParams(RunnerParams const& params) const
    {
        // The updated kernel type.
        FmhaKernelType kernelType = params.mKernelType;
        // Generation kernelType will use either SwapsMmaAbForGeneration or KeepsMmaAbForGeneration.
        if (isGenerationKernel(params.mKernelType))
        {
            kernelType = (params.mNumHeadsQPerKv <= 16 && params.mHeadDim != 32)
                ? FmhaKernelType::SwapsMmaAbForGeneration
                : FmhaKernelType::KeepsMmaAbForGeneration;
        }

        // The maximum number of headsQPerKv that the kernel needs to support.
        // Default 0 means that it can support any value of numHeadsQPerKv.
        int maxNumHeadsQPerKv = 0;
        if (isSwapsMmaAbForGenerationKernel(kernelType))
        {
            TLLM_CHECK_WITH_INFO(params.mNumHeadsQPerKv <= 16, "Not supported.");
            maxNumHeadsQPerKv = (params.mNumHeadsQPerKv <= 8) ? 8 : 16;
        }
        else if (isKeepsMmaAbForGenerationKernel(kernelType))
        {
            TLLM_CHECK_WITH_INFO(params.mNumHeadsQPerKv <= 32, "Not supported.");
            maxNumHeadsQPerKv = 32;
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
            + ", tileScheduler=" + std::to_string(static_cast<int>(params.mTileScheduler))
            + ", multiCtasKvMode=" + std::to_string(params.mMultiCtasKvMode)
            + ", headDim=" + std::to_string(params.mHeadDim) + ", numTokensPerPage=" + std::to_string(numTokensPerPage)
            + ", maxNumHeadsQPerKv=" + std::to_string(maxNumHeadsQPerKv);
        TLLM_LOG_DEBUG("Searching for kernel traits: " + info);

        return std::make_pair(hashID(static_cast<int>(params.mQkvLayout), static_cast<int>(maskType),
                                  static_cast<int>(kernelType), static_cast<int>(params.mTileScheduler),
                                  params.mMultiCtasKvMode, params.mHeadDim, numTokensPerPage, maxNumHeadsQPerKv),
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
