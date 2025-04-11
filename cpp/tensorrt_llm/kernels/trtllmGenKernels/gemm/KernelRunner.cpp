/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <vector>

#include "KernelRunner.h"
#include "tensorrt_llm/common/assert.h"
#include "trtllmGen_export/GemmInterface.h"

namespace tensorrt_llm
{
namespace kernels
{

TrtllmGenGemmRunner::TrtllmGenGemmRunner(tg::Dtype eltType, tg::Dtype outputType)
    : mEltType(eltType)
    , mOutputType(outputType)
{
    // Select a GEMM kernel config to use
    auto const gemm = gemm::GemmInterface();
    auto const configs = gemm.getGemmConfigs();

    std::vector<int32_t> selectedIndex;

    for (size_t i = 0; i < gemm.getNumGemmConfigs(); ++i)
    {
        auto const options = configs[i].mOptions;

        // When we include low-latency kernels we can set transposeMmaOutput via constructor
        if (options.mDtypeElt == eltType && options.mDtypeC == outputType && !options.mTransposeMmaOutput)
        {
            selectedIndex.push_back(i);
        }
    }

    TLLM_CHECK_WITH_INFO(selectedIndex.size() != 0, "No kernel found for the given output type");
    TLLM_CHECK_WITH_INFO(selectedIndex.size() == 1, "Multiple kernels found for the given output type");

    mGemmConfig = &configs[selectedIndex[0]];
}

size_t TrtllmGenGemmRunner::getWorkspaceSizeInBytes(
    int32_t m, int32_t n, int32_t k, tg::Dtype eltType, tg::Dtype outputType) const
{
    gemm::GemmData gemmData;
    gemmData.mProblemDimensions.mM = m;
    gemmData.mProblemDimensions.mN = n;
    gemmData.mProblemDimensions.mK = k;

    auto gemm = gemm::GemmInterface();

    return gemm.getWorkspaceSizeInBytes(*mGemmConfig, gemmData);
}

void TrtllmGenGemmRunner::run(int32_t m, int32_t n, int32_t k, void const* a, float const* aScale, void const* b,
    float const* bScale, void* c, float* cScale, void* workspace, CUstream stream, int device)
{
    auto gemm = gemm::GemmInterface();

    gemm::GemmData gemmData;

    // Dims
    gemmData.mProblemDimensions.mM = m;
    gemmData.mProblemDimensions.mN = n;
    gemmData.mProblemDimensions.mK = k;

    // Inputs
    gemmData.mInputBuffers.mPtrA = a;
    gemmData.mInputBuffers.mPtrSfA = aScale;
    gemmData.mInputBuffers.mPtrB = b;
    gemmData.mInputBuffers.mPtrSfB = bScale;
    gemmData.mInputBuffers.mPtrScaleC = cScale;

    // Outputs
    gemmData.mOutputBuffers.mPtrC = c;

    auto isValidConfig = gemm.isValidConfig(*mGemmConfig, gemmData);
    TLLM_CHECK_WITH_INFO(isValidConfig, "Invalid GEMM config selected!");

    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, device);

    // FIXME once we start using all-reduce in the epilogue of the gemm this can be moved elsewhere
    gemm.runInitBeforeWorldSync(*mGemmConfig, gemmData, static_cast<void*>(stream));

    auto const err = gemm.run(*mGemmConfig, workspace, gemmData, static_cast<void*>(stream), deviceProperties);

    TLLM_CHECK_WITH_INFO(err == 0, "Error occurred when running GEMM!");
}

} // namespace kernels
} // namespace tensorrt_llm
