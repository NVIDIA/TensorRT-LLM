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
#include "trtllmGen_gatedAct_export/GemmGatedActInterface.h"
#include "trtllmGen_gatedAct_export/GemmOptions.h"
#include "trtllmGen_gatedAct_export/trtllm/gen/DtypeDecl.h"

namespace tensorrt_llm
{
namespace kernels
{
using namespace gemmGatedAct::gemmGatedAct;
static GemmGatedActInterface::ModuleCache globalTrtllmGenGemmGatedActModuleCache;

TrtllmGenGemmGatedActRunner::TrtllmGenGemmGatedActRunner(TrtllmGenGemmGatedActRunnerOptions const& options_)
    : mOptions(options_)
{
    // Select a GEMM kernel config to use
    auto const gemm = GemmGatedActInterface();
    auto const configs = gemm.getGemmConfigs();

    mPassingConfigIndices.clear();

    for (size_t i = 0; i < gemm.getNumGemmConfigs(); ++i)
    {
        auto const options = configs[i].mOptions;

        // When we include low-latency kernels we can set transposeMmaOutput via constructor
        if (options.mDtypeA == mOptions.eltType && options.mDtypeC == mOptions.outputType
            && options.mUseDeepSeekFp8 == mOptions.deepSeekFp8
            && options.mTransposeMmaOutput == mOptions.transposeMmaOutput)
        {
            mPassingConfigIndices.push_back(i);
        }
    }

    TLLM_CHECK_WITH_INFO(mPassingConfigIndices.size() != 0, "No kernel found for the given output type");
}

size_t TrtllmGenGemmGatedActRunner::getWorkspaceSizeInBytes(int32_t m, int32_t n, int32_t k)
{
    GemmGatedActData gemmData;
    gemmData.mProblemDimensions.mM = mOptions.transposeMmaOutput ? n : m;
    gemmData.mProblemDimensions.mN = mOptions.transposeMmaOutput ? m : n;
    gemmData.mProblemDimensions.mK = k;

    selectGemmConfig(m, n, k);

    auto gemm = GemmGatedActInterface();
    auto const configs = gemm.getGemmConfigs();
    TLLM_CHECK_WITH_INFO(
        mSelectedConfigIndex.has_value(), "No valid kernel found for given param config and problem size");
    auto const config = configs[mSelectedConfigIndex.value()];

    return gemm.getWorkspaceSizeInBytes(config, gemmData);
}

void TrtllmGenGemmGatedActRunner::run(int32_t m, int32_t n, int32_t k, void const* a, float const* aScale,
    void const* b, float const* bScale, void* c, float* cScale, float* cScaleGate, void* workspace, CUstream stream,
    int device)
{
    auto gemm = GemmGatedActInterface();

    GemmGatedActData gemmData;

    auto const configs = gemm.getGemmConfigs();
    TLLM_CHECK_WITH_INFO(
        mSelectedConfigIndex.has_value(), "No valid kernel found for given param config and problem size");
    auto const& config = configs[mSelectedConfigIndex.value()];

    // Dims
    gemmData.mProblemDimensions.mM = mOptions.transposeMmaOutput ? n : m;
    gemmData.mProblemDimensions.mN = mOptions.transposeMmaOutput ? m : n;
    gemmData.mProblemDimensions.mK = k;

    // Inputs
    gemmData.mInputBuffers.mPtrA = mOptions.transposeMmaOutput ? b : a;
    gemmData.mInputBuffers.mPtrSfA = mOptions.transposeMmaOutput ? bScale : aScale;
    gemmData.mInputBuffers.mPtrB = mOptions.transposeMmaOutput ? a : b;
    gemmData.mInputBuffers.mPtrSfB = mOptions.transposeMmaOutput ? aScale : bScale;
    gemmData.mInputBuffers.mPtrScaleC = cScale;
    gemmData.mInputBuffers.mPtrScaleGate = cScaleGate;
    // Outputs
    gemmData.mOutputBuffers.mPtrC = c;

    int32_t multiProcessorCount;
    cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, device);

    // FIXME once we start using all-reduce in the epilogue of the gemm this can be moved elsewhere
    gemm.runInitBeforeWorldSync(config, gemmData, static_cast<void*>(stream));

    auto const err = gemm.run(config, workspace, gemmData, static_cast<void*>(stream), multiProcessorCount,
        /*usePdl=*/true, globalTrtllmGenGemmGatedActModuleCache);

    TLLM_CHECK_WITH_INFO(err == 0, "Error occurred when running GEMM!");
}

void TrtllmGenGemmGatedActRunner::run(int32_t m, int32_t n, int32_t k, void const* a, void const* b, void* c,
    float* cScale, float* cScaleGate, void* workspace, CUstream stream, int device)
{
    run(m, n, k, a, /*aScale*/ nullptr, b, /*bScale*/ nullptr, c, cScale, cScaleGate, workspace, stream, device);
}

void TrtllmGenGemmGatedActRunner::selectGemmConfig(int32_t m, int32_t n, int32_t k)
{
    auto const gemm = GemmGatedActInterface();
    auto const configs = gemm.getGemmConfigs();

    GemmGatedActData gemmData;
    // Dims
    gemmData.mProblemDimensions.mM = mOptions.transposeMmaOutput ? n : m;
    gemmData.mProblemDimensions.mN = mOptions.transposeMmaOutput ? m : n;
    gemmData.mProblemDimensions.mK = k;

    for (auto const& configIndex : mPassingConfigIndices)
    {
        auto const& config = configs[configIndex];
        // FIXME: We select the first valid config,
        // but must instead choose the "best" config based on some heruistics.
        auto isValidConfig = gemm.isValidConfig(config, gemmData);
        if (isValidConfig)
        {
            mSelectedConfigIndex = configIndex;
            return;
        }
    }
}

} // namespace kernels
} // namespace tensorrt_llm
