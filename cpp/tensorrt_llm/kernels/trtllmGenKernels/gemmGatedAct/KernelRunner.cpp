/*
 * Copyright (c) 2020-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/config.h"
#include "trtllmGen_gatedAct_export/GemmGatedActInterface.h"
#include "trtllmGen_gatedAct_export/GemmOptions.h"
#include "trtllmGen_gatedAct_export/trtllm/gen/DtypeDecl.h"
// Include after trtllm-gen export headers to avoid TLLM_LOG_* macro redefinition warnings
#include "tensorrt_llm/common/cudaUtils.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
using namespace gemmGatedAct::gemmGatedAct;
static GemmGatedActInterface::ModuleCache globalTrtllmGenGemmGatedActModuleCache;

using SmVersion = gemmGatedAct::gemm::SmVersion;

constexpr bool isSMCompatible(int gpuSM, SmVersion kernelSM)
{
    if (gpuSM == 107)
    {
        // SM107 (Rubin) can run SM100f family kernels and SM107a-specific kernels
        return kernelSM == SmVersion::Sm107a || kernelSM == SmVersion::Sm100f;
    }
    else if (gpuSM == 103)
    {
        return kernelSM == SmVersion::Sm103a || kernelSM == SmVersion::Sm100f;
    }
    else if (gpuSM == 100)
    {
        return kernelSM == SmVersion::Sm100a || kernelSM == SmVersion::Sm100f;
    }
    else if (gpuSM == 90)
    {
        return kernelSM == SmVersion::Sm90a;
    }
    // Redirect SM100 family (major version 10) to family kernels
    else if (tensorrt_llm::common::isSM100Family(gpuSM))
    {
        return kernelSM == SmVersion::Sm100f;
    }
    return true;
}

TrtllmGenGemmGatedActRunner::TrtllmGenGemmGatedActRunner(TrtllmGenGemmGatedActRunnerOptions const& options_)
    : mOptions(options_)
{
    // Select a GEMM kernel config to use
    auto const gemm = GemmGatedActInterface();
    auto const configs = gemm.getGemmConfigs();

    mPassingConfigIndices.clear();

    int const gpuSM = tensorrt_llm::common::getSMVersion();

    for (size_t i = 0; i < gemm.getNumGemmConfigs(); ++i)
    {
        auto const& config = configs[i];
        auto const options = config.mOptions;

        // Filter by SM compatibility first
        if (!isSMCompatible(gpuSM, config.mSm))
        {
            continue;
        }

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
    // Set valid dimensions to full range (same as M/N/K when no padding)
    gemmData.mProblemDimensions.mValidM = mOptions.transposeMmaOutput ? n : m;
    gemmData.mProblemDimensions.mValidN = mOptions.transposeMmaOutput ? m : n;
    gemmData.mProblemDimensions.mValidK = k;

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
    // Set valid dimensions to full range (same as M/N/K when no padding)
    gemmData.mProblemDimensions.mValidM = mOptions.transposeMmaOutput ? n : m;
    gemmData.mProblemDimensions.mValidN = mOptions.transposeMmaOutput ? m : n;
    gemmData.mProblemDimensions.mValidK = k;

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
    // Set valid dimensions to full range (same as M/N/K when no padding)
    gemmData.mProblemDimensions.mValidM = mOptions.transposeMmaOutput ? n : m;
    gemmData.mProblemDimensions.mValidN = mOptions.transposeMmaOutput ? m : n;
    gemmData.mProblemDimensions.mValidK = k;

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

TRTLLM_NAMESPACE_END
