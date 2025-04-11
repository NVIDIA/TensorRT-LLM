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
#include "tensorrt_llm/common/envUtils.h"
#include <iostream>

namespace tensorrt_llm
{
namespace kernels
{
TrtllmGenBlockScaleGemmRunner::TrtllmGenBlockScaleGemmRunner(Data_type outputType)
    : mOutputType(outputType)
{
    std::vector<int32_t> selectedIndex;
    for (size_t i = 0; i < trtllmGenBlockScaleGemmInfo.size(); i++)
    {
        if (trtllmGenBlockScaleGemmInfo[i].dtypeC == outputType)
        {
            selectedIndex.push_back(i);
        }
    }
    TLLM_CHECK_WITH_INFO(selectedIndex.size() != 0, "No kernel found for the given output type");
    TLLM_CHECK_WITH_INFO(selectedIndex.size() == 1, "Multiple kernels found for the given output type");

    mKernelInfo = &trtllmGenBlockScaleGemmInfo[selectedIndex[0]];
    mDriver = tensorrt_llm::common::CUDADriverWrapper::getInstance();
    TLLM_CU_CHECK(mDriver->cuModuleLoadData(&mModule, mKernelInfo->data));
    TLLM_CU_CHECK(mDriver->cuModuleGetFunction(&mFunction, mModule, mKernelInfo->functionName));
    if (mKernelInfo->sharedMemSize >= 48 * 1024)
    {
        TLLM_CU_CHECK(mDriver->cuFuncSetAttribute(
            mFunction, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, mKernelInfo->sharedMemSize));
    }
}

struct TrtllmGenBlockScaleGemmOptions
{
    int32_t mM{-1};
    int32_t mN{-1};
    int32_t mK{-1};
    int32_t mTileM{-1};
    int32_t mTileN{-1};
    int32_t mTileK{-1};
    int32_t mNumSlicesForSplitK{-1};
    int32_t mEpilogueTileM{-1};
    int32_t mEpilogueTileN{-1};
    int32_t mMmaM{-1};
    int32_t mMmaN{-1};
    bool mUseTmaStore{false};
    int32_t mNumStages{-1};
    Data_type mDtypeElt;
    Data_type mDtypeC;
    Data_type mDtypeAcc;
    bool mTransposeMmaOutput{false};
    TrtllmGenBlockScaleGemmKernelParams::AllReduceAlgo mAllReduceAlgo{
        TrtllmGenBlockScaleGemmKernelParams::AllReduceAlgo::None};
    bool mSliceK{false};
};

void TrtllmGenBlockScaleGemmRunner::run(int32_t m, int32_t n, int32_t k, void const* a, float const* aScale,
    void const* b, float const* bScale, void* c, float* cScale, CUstream stream)
{

    TrtllmGenBlockScaleGemmOptions options;
    options.mM = m;
    options.mN = n;
    options.mK = k;
    options.mTileM = mKernelInfo->tileM;
    options.mTileN = mKernelInfo->tileN;
    options.mTileK = mKernelInfo->tileK;
    options.mNumSlicesForSplitK = mKernelInfo->numSlicesForSplitK;
    options.mEpilogueTileM = mKernelInfo->epilogueTileM;
    options.mEpilogueTileN = mKernelInfo->epilogueTileN;
    options.mMmaM = mKernelInfo->mmaM;
    options.mMmaN = mKernelInfo->mmaN;
    options.mUseTmaStore = mKernelInfo->useTmaStore;
    options.mNumStages = mKernelInfo->numStages;
    options.mDtypeElt = mKernelInfo->dtypeElt;
    options.mDtypeC = mKernelInfo->dtypeC;
    options.mDtypeAcc = mKernelInfo->dtypeAcc;
    options.mTransposeMmaOutput = mKernelInfo->transposeMmaOutput;
    options.mAllReduceAlgo = TrtllmGenBlockScaleGemmKernelParams::AllReduceAlgo::None;
    options.mSliceK = mKernelInfo->sliceK;

    auto params = TrtllmGenBlockScaleGemmKernelParams::setKernelParams(options, a, aScale, b, bScale, c,
        nullptr /* ptrSfc */, nullptr /* multimemC */, cScale /* ptrScaleC */, nullptr /* ptrPartialSumsForSplitK */,
        nullptr /* ptrTileBars */, nullptr /* multimemTileBars */, nullptr /* ptrCompletionBars */,
        nullptr /* multimemCompletionBars */, nullptr /* ptrSplitKCompletionBars */, 0, 1);
    TLLM_CHECK_WITH_INFO(sizeof(params) == 832, "Size of mismatch between trtllm-gen and trtllm");

    CUlaunchConfig launch_config;
    launch_config.blockDimX = mKernelInfo->threadsPerCTA;
    launch_config.blockDimY = 1;
    launch_config.blockDimZ = 1;
    launch_config.gridDimX = (options.mM + options.mTileM - 1) / options.mTileM;
    launch_config.gridDimY = (options.mN + options.mTileN - 1) / options.mTileN;
    launch_config.gridDimZ = options.mNumSlicesForSplitK;
    launch_config.hStream = stream;
    launch_config.sharedMemBytes = mKernelInfo->sharedMemSize;
    CUlaunchAttribute launch_attribute[3];
    launch_attribute[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    launch_attribute[0].value.clusterDim.x = 1;
    launch_attribute[0].value.clusterDim.y = 1;
    launch_attribute[0].value.clusterDim.z = 1;
    launch_attribute[1].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
    launch_attribute[1].value.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_DEFAULT;
    launch_attribute[2].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
    launch_attribute[2].value.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    launch_config.attrs = launch_attribute;
    launch_config.numAttrs = 3;
    void* kernelParamsList[] = {&params};
    TLLM_CU_CHECK(mDriver->cuLaunchKernelEx(&launch_config, mFunction, kernelParamsList, nullptr));
}
} // namespace kernels
} // namespace tensorrt_llm
