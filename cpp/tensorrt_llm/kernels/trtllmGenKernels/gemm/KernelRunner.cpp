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

// clang-format off
#include "trtllmGen_gemm_export/GemmInterface.h"
#include "trtllmGen_gemm_export/GemmOptions.h"
#include "trtllmGen_gemm_export/trtllm/gen/DtypeDecl.h"
// clang-format on

#include "KernelRunner.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"

namespace tensorrt_llm
{
namespace kernels
{

namespace tg = gemm::trtllm::gen;
using namespace gemm::gemm;

static GemmInterface::ModuleCache globalTrtllmGenGemmModuleCache;

constexpr bool isSMCompatible(int gpuSM, SmVersion kernelSM)
{
    if (gpuSM == 103)
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
    return true;
}

TrtllmGenGemmRunner::TrtllmGenGemmRunner(TrtllmGenGemmRunnerOptions const& options_)
    : mOptions(options_)
{
    // Select a GEMM kernel config to use
    auto const gemm = GemmInterface();
    auto const configs = gemm.getGemmConfigs();

    mPassingConfigIndices.clear();
    int gpuNativeSmVersion = tensorrt_llm::common::getSMVersion();
    for (size_t i = 0; i < gemm.getNumGemmConfigs(); ++i)
    {
        auto const options = configs[i].mOptions;

        // When we include low-latency kernels we can set transposeMmaOutput via constructor
        if (options.mDtypeA == mOptions.eltTypeA && options.mDtypeC == mOptions.outputType
            && options.mUseDeepSeekFp8 == mOptions.deepSeekFp8
            && options.mTransposeMmaOutput == mOptions.transposeMmaOutput
            && (mOptions.eltTypeB == gemm::trtllm::gen::Dtype::Void || options.mDtypeB == mOptions.eltTypeB)
            && isSMCompatible(gpuNativeSmVersion, configs[i].mSm))
        {
            mPassingConfigIndices.push_back(i);
        }
    }

    TLLM_CHECK_WITH_INFO(mPassingConfigIndices.size() != 0, "No kernel found for the given output type");
}

size_t TrtllmGenGemmRunner::getWorkspaceSizeInBytes(int32_t m, int32_t n, int32_t k)
{
    GemmData gemmData;
    gemmData.mProblemDimensions.mM = mOptions.transposeMmaOutput ? n : m;
    gemmData.mProblemDimensions.mN = mOptions.transposeMmaOutput ? m : n;
    gemmData.mProblemDimensions.mK = k;
    gemmData.mProblemDimensions.mRank = 0;
    gemmData.mProblemDimensions.mWorldSize = 1;

    selectGemmConfig(m, n, k);

    auto gemm = GemmInterface();
    auto const configs = gemm.getGemmConfigs();
    TLLM_CHECK_WITH_INFO(
        mSelectedConfigIndex.has_value(), "No valid kernel found for given param config and problem size");
    auto const config = configs[mSelectedConfigIndex.value()];

    return gemm.getWorkspaceSizeInBytes(config, gemmData);
}

void TrtllmGenGemmRunner::run(int32_t m, int32_t n, int32_t k, void const* a, float const* aScale, void const* b,
    float const* bScale, void* c, float* cScale, float* cScalePtr, void* workspace, CUstream stream, int device)
{
    auto gemm = GemmInterface();

    GemmData gemmData;

    auto const configs = gemm.getGemmConfigs();
    TLLM_CHECK_WITH_INFO(
        mSelectedConfigIndex.has_value(), "No valid kernel found for given param config and problem size");
    auto const& config = configs[mSelectedConfigIndex.value()];

    // Dims
    gemmData.mProblemDimensions.mM = mOptions.transposeMmaOutput ? n : m;
    gemmData.mProblemDimensions.mN = mOptions.transposeMmaOutput ? m : n;
    gemmData.mProblemDimensions.mK = k;
    gemmData.mProblemDimensions.mRank = 0;
    gemmData.mProblemDimensions.mWorldSize = 1;

    // Inputs
    gemmData.mInputBuffers.mPtrA = mOptions.transposeMmaOutput ? b : a;
    gemmData.mInputBuffers.mPtrSfA = mOptions.transposeMmaOutput ? bScale : aScale;
    gemmData.mInputBuffers.mPtrB = mOptions.transposeMmaOutput ? a : b;
    gemmData.mInputBuffers.mPtrSfB = mOptions.transposeMmaOutput ? aScale : bScale;
    gemmData.mInputBuffers.mPtrScaleC = cScale;

    // Outputs
    gemmData.mOutputBuffers.mPtrC = c;
    gemmData.mOutputBuffers.mPtrSfC = cScalePtr;

    int32_t multiProcessorCount;
    cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, device);

    // FIXME once we start using all-reduce in the epilogue of the gemm this can be moved elsewhere
    gemm.runInitBeforeWorldSync(config, gemmData, static_cast<void*>(stream));

    auto const err = gemm.run(config, workspace, gemmData, static_cast<void*>(stream), multiProcessorCount,
        tensorrt_llm::common::getEnvEnablePDL(), globalTrtllmGenGemmModuleCache);

    TLLM_CHECK_WITH_INFO(err == 0, "Error occurred when running GEMM!");
}

void TrtllmGenGemmRunner::run(int32_t m, int32_t n, int32_t k, void const* a, void const* b, void* c, float* cScale,
    void* workspace, CUstream stream, int device)
{
    run(m, n, k, a, /*aScale*/ nullptr, b, /*bScale*/ nullptr, c, cScale, /*cScalePtr*/ nullptr, workspace, stream,
        device);
}

void TrtllmGenGemmRunner::selectGemmConfig(int32_t m, int32_t n, int32_t k)
{
    auto const gemm = GemmInterface();
    auto const configs = gemm.getGemmConfigs();

    GemmData gemmData;
    // Dims
    gemmData.mProblemDimensions.mM = mOptions.transposeMmaOutput ? n : m;
    gemmData.mProblemDimensions.mN = mOptions.transposeMmaOutput ? m : n;
    gemmData.mProblemDimensions.mK = k;
    gemmData.mProblemDimensions.mRank = 0;
    gemmData.mProblemDimensions.mWorldSize = 1;

    std::vector<int32_t> sortedIndices = mPassingConfigIndices;
    std::sort(sortedIndices.begin(), sortedIndices.end(),
        [&configs, &gemmData](int32_t idx0, int32_t idx1)
        {
            auto const& optionsA = configs[idx0].mOptions;
            auto const& optionsB = configs[idx1].mOptions;

            // Choose the tileN that is closest to the problem N. Also if one tileN is larger and the other is smaller,
            // prefer the larger one. This is the batch size dimension for low latency (transposeMmaOutput) case;
            if (optionsA.mTileN != optionsB.mTileN)
            {
                auto const N = gemmData.mProblemDimensions.mN;
                auto const tileA = optionsA.mTileN;
                auto const tileB = optionsB.mTileN;

                // If one tile is larger than N and one is smaller, prefer the larger one
                if ((tileA >= N) != (tileB >= N))
                {
                    return tileA > tileB;
                }

                // Otherwise, choose the closest to N
                return abs(N - tileA) < abs(N - tileB);
            }

            // Sort by tileK sizes
            if (optionsA.mTileK != optionsB.mTileK)
            {
                return optionsA.mTileK > optionsB.mTileK;
            }

            // Then by unroll loop 2x for mma
            if (optionsA.mUseUnrollLoop2xForMma != optionsB.mUseUnrollLoop2xForMma)
            {
                return optionsA.mUseUnrollLoop2xForMma;
            }

            // Sort by tileM sizes
            // This is the batch size dimension for throughput (non-transposeMmaOutput) case;
            if (optionsA.mTileM != optionsB.mTileM)
            {
                return optionsA.mTileM > optionsB.mTileM;
            }

            // Then by splitK sizes
            if (optionsA.mNumSlicesForSplitK != optionsB.mNumSlicesForSplitK)
            {
                return optionsA.mNumSlicesForSplitK > optionsB.mNumSlicesForSplitK;
            }

            return true;
        });

    for (auto const& configIndex : sortedIndices)
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
