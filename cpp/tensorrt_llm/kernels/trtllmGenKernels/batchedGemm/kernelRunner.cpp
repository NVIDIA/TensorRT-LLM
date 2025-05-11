/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "kernelRunner.h"
#include "kernelList.h"
#include "kernelParams.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/kernels/multiHeadAttentionCommon.h"

namespace tensorrt_llm::kernels
{

TrtllmGenBatchedGemmRunner::TrtllmGenBatchedGemmRunner(
    Data_type outputType, int64_t gemmBatchSize, int64_t tileSize, bool useDeepSeekFp8, bool batchModeM)
    : mOutputType(outputType)
    , mGemmBatchSize(gemmBatchSize)
    , mUseDeepSeekFp8(useDeepSeekFp8)
    , mBatchMode(batchModeM ? BatchMode::BatchM : BatchMode::BatchN)
    , mBatchModeM(batchModeM)
    , mTileSize(tileSize)
{
    std::vector<int32_t> selectedIndex;
    for (size_t i = 0; i < trtllmGenBatchedStridedGemmInfo.size(); i++)
    {
        if (mBatchModeM)
        {
            if (trtllmGenBatchedStridedGemmInfo[i].dtypeC == outputType
                && trtllmGenBatchedStridedGemmInfo[i].useDeepSeekFp8 == useDeepSeekFp8
                && trtllmGenBatchedStridedGemmInfo[i].batchMode == mBatchMode
                && trtllmGenBatchedStridedGemmInfo[i].tileN == mTileSize)
            {
                selectedIndex.push_back(i);
            }
        }
        else
        {
            if (trtllmGenBatchedStridedGemmInfo[i].dtypeC == outputType
                && trtllmGenBatchedStridedGemmInfo[i].useDeepSeekFp8 == useDeepSeekFp8
                && trtllmGenBatchedStridedGemmInfo[i].batchMode == mBatchMode
                && trtllmGenBatchedStridedGemmInfo[i].tileM == mTileSize)
            {
                selectedIndex.push_back(i);
            }
        }
    }
    TLLM_CHECK_WITH_INFO(!selectedIndex.empty(), "No kernel found for the given output type and gemmBatchSize");
    TLLM_CHECK_WITH_INFO(
        selectedIndex.size() == 1, "Multiple kernels found for the given output type and gemmBatchSize");

    mKernelInfo = &trtllmGenBatchedStridedGemmInfo[selectedIndex.front()];
    mDriver = tensorrt_llm::common::CUDADriverWrapper::getInstance();
    TLLM_CU_CHECK(mDriver->cuModuleLoadData(&mModule, mKernelInfo->data));
    TLLM_CHECK_WITH_INFO(mModule != nullptr, "No module");
    TLLM_CU_CHECK(mDriver->cuModuleGetFunction(&mFunction, mModule, mKernelInfo->functionName));
    TLLM_CU_CHECK(mDriver->cuFuncSetAttribute(
        mFunction, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, mKernelInfo->sharedMemSize));
}

void TrtllmGenBatchedGemmRunner::run(int32_t m, int32_t n, int32_t k, void* a, void* b, void* c, float const* cScale,
    float* dDqSfsA, float* dDqSfsB, float* dDqSfsC, std::vector<int32_t> const& batchedM,
    std::vector<int32_t> const& batchedN, CUstream stream)
{
    // Shuffle the A matrix (if needed). TODO: on-device function
    if (mKernelInfo->useShuffledMatrixA)
    {
        TLLM_LOG_WARNING("useShuffledMatrixA enabled, shuffling matrix A on host");

        // Keep source tensor intact
        void* hShuffledPtrA{nullptr};
        auto numBytesA = mGemmBatchSize * m * k * get_size_in_bytes(mKernelInfo->dtypeElt);
        hShuffledPtrA = (void*) malloc(numBytesA);
        cudaMemcpy(hShuffledPtrA, a, numBytesA, cudaMemcpyDeviceToHost);

        // Reorder matrix rows for the wide stores in the epilogue.
        this->template shuffleMatrixA<Data_type::DATA_TYPE_E4M3>(
            hShuffledPtrA, hShuffledPtrA, mGemmBatchSize, m, k, mKernelInfo->epilogueTileM);

        // Copy shuffled matrix to the device instead of the original matrix.
        cudaMemcpy(a, hShuffledPtrA, numBytesA, cudaMemcpyHostToDevice);
        free(hShuffledPtrA);
    }

    auto params = TrtllmGenBatchedGemmKernelParams::setKernelParams(mGemmBatchSize, mKernelInfo->numTokens, mBatchModeM,
        m, n, k, batchedM, batchedN, mKernelInfo->tileM, mKernelInfo->tileN, mKernelInfo->tileK,
        mKernelInfo->epilogueTileM, mKernelInfo->epilogueTileN, mUseDeepSeekFp8, mKernelInfo->useTmaStore,
        mKernelInfo->transposeMmaOutput, false, mKernelInfo->dtypeElt, mKernelInfo->dtypeC, a, b, c, cScale, dDqSfsA,
        dDqSfsB, dDqSfsC, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    CUlaunchConfig launch_config{};
    launch_config.blockDimX = mKernelInfo->threadsPerCTA;
    launch_config.blockDimY = 1;
    launch_config.blockDimZ = 1;
    launch_config.gridDimX
        = mBatchModeM ? common::divUp(mGemmBatchSize * m, mKernelInfo->tileM) : common::divUp(m, mKernelInfo->tileM);
    launch_config.gridDimY
        = mBatchModeM ? common::divUp(n, mKernelInfo->tileN) : common::divUp(mGemmBatchSize * n, mKernelInfo->tileN);
    launch_config.gridDimZ = mKernelInfo->numSlicesForSplitK;
    launch_config.hStream = stream;
    launch_config.sharedMemBytes = mKernelInfo->sharedMemSize;
    CUlaunchAttribute launch_attribute[2];
    launch_attribute[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    launch_attribute[0].value.clusterDim.x = 1;
    launch_attribute[0].value.clusterDim.y = 1;
    launch_attribute[0].value.clusterDim.z = 1;
    launch_attribute[1].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
    launch_attribute[1].value.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_DEFAULT;
    launch_config.attrs = launch_attribute;
    launch_config.numAttrs = 2;
    void* kernelParamsList[] = {&params};

    TLLM_CU_CHECK(mDriver->cuLaunchKernelEx(&launch_config, mFunction, kernelParamsList, nullptr));
}
} // namespace tensorrt_llm::kernels
