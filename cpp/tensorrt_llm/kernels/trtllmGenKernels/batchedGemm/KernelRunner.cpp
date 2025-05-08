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
#include "trtllmGen_bmm_export/BatchedGemmInterface.h"
#include "trtllmGen_bmm_export/trtllm/gen/DtypeDecl.h"

namespace tensorrt_llm
{
namespace kernels
{

TrtllmGenBatchedGemmRunner::TrtllmGenBatchedGemmRunner(TrtllmGenBatchedGemmRunnerOptions const& options_)
    : mOptions(options_)
{
    // Select a GEMM kernel config to use
    auto const bmm = batchedGemm::BatchedGemmInterface();
    auto const configs = bmm.getBatchedGemmConfigs();

    mPassingConfigIndices.clear();

    for (size_t i = 0; i < bmm.getNumBatchedGemmConfigs(); ++i)
    {
        auto const options = configs[i].mOptions;
        auto const tileSize = mOptions.transposeMmaOutput ? options.mTileN : options.mTileM;
        // When we include low-latency kernels we can set transposeMmaOutput via constructor
        if (options.mDtypeElt == mOptions.eltType && options.mDtypeC == mOptions.outputType
            && options.mUseDeepSeekFp8 == mOptions.deepSeekFp8
            && options.mTransposeMmaOutput == mOptions.transposeMmaOutput && options.mRouteAct == mOptions.routeAct
            && options.mFusedAct == mOptions.fusedAct && options.mIsStaticBatch == mOptions.staticBatch
            && tileSize == mOptions.tileSize)
        {
            if (mOptions.transposeMmaOutput && options.mEpilogueTileM == mOptions.epilogueTileM)
            {
                mPassingConfigIndices.push_back(i);
            }
        }
    }

    TLLM_CHECK_WITH_INFO(mPassingConfigIndices.size() != 0, "No kernel found for the given output type");
}

size_t TrtllmGenBatchedGemmRunner::getWorkspaceSizeInBytes(int32_t m, int32_t n, int32_t k,
    std::vector<int32_t> const& batchedTokens, int32_t numTokens, int32_t numBatches, int32_t maxNumCtasInBatchDim)
{
    batchedGemm::BatchedGemmData gemmData;
    gemmData.mProblemDimensions.mNumBatches = numBatches;
    gemmData.mProblemDimensions.mNumTokens = numTokens;
    gemmData.mProblemDimensions.mBatchM = !mOptions.transposeMmaOutput;
    gemmData.mProblemDimensions.mBatchedM = mOptions.transposeMmaOutput ? std::vector<int32_t>{} : batchedTokens;
    gemmData.mProblemDimensions.mBatchedN = mOptions.transposeMmaOutput ? batchedTokens : std::vector<int32_t>{};
    gemmData.mProblemDimensions.mM = mOptions.transposeMmaOutput ? n : m;
    gemmData.mProblemDimensions.mN = mOptions.transposeMmaOutput ? m : n;
    gemmData.mProblemDimensions.mK = k;
    gemmData.mProblemDimensions.mRank = 0;
    gemmData.mProblemDimensions.mWorldSize = 1;
    gemmData.mProblemDimensions.mMaxNumCtasInTokenDim = maxNumCtasInBatchDim;

    selectGemmConfig(m, n, k, batchedTokens, numTokens, numBatches, maxNumCtasInBatchDim);

    auto bmm = batchedGemm::BatchedGemmInterface();
    auto const configs = bmm.getBatchedGemmConfigs();
    TLLM_CHECK_WITH_INFO(
        mSelectedConfigIndex.has_value(), "No valid kernel found for given param config and problem size");
    auto const& config = configs[mSelectedConfigIndex.value()];
    return bmm.getWorkspaceSizeInBytes(config, gemmData);
}

void TrtllmGenBatchedGemmRunner::run(int32_t m, int32_t n, int32_t k, std::vector<int32_t> const& batchedTokens,
    int32_t numTokens, int32_t numBatches, int32_t maxNumCtasInBatchDim, void const* a, void const* sfA, void const* b,
    void const* sfB, void const* perTokensSfA, void const* perTokensSfB, float const* scaleC, float const* scaleGateC,
    void* c, void* outSfC, int32_t const* routeMap, int32_t const* totalNumPaddedTokens,
    int32_t const* ctaIdxXyToBatchIdx, int32_t const* ctaIdxXyToMnLimit, int32_t const* numNonExitingCtas,
    void* workspace, CUstream stream, int device)
{
    auto bmm = batchedGemm::BatchedGemmInterface();

    batchedGemm::BatchedGemmData gemmData;

    auto const configs = bmm.getBatchedGemmConfigs();
    TLLM_CHECK_WITH_INFO(
        mSelectedConfigIndex.has_value(), "No valid kernel found for given param config and problem size");
    auto const& config = configs[mSelectedConfigIndex.value()];

    TLLM_CHECK_WITH_INFO(numBatches > 0, "Batched GEMM requires numBatches > 0");
    if (!mOptions.staticBatch)
    {
        TLLM_CHECK_WITH_INFO(totalNumPaddedTokens, "Batched GEMM with dynamic batching requires totalNumPaddedTokens");
        TLLM_CHECK_WITH_INFO(ctaIdxXyToBatchIdx, "Batched GEMM with dynamic batching requires ctaIdxXyToBatchIdx");
        TLLM_CHECK_WITH_INFO(ctaIdxXyToMnLimit, "Batched GEMM with dynamic batching requires ctaIdxXyToMnLimit");
        TLLM_CHECK_WITH_INFO(numNonExitingCtas, "Batched GEMM with dynamic batching requires numNonExitingCtas");
    }

    if (!mOptions.staticBatch && numTokens != 0)
    {
        TLLM_CHECK_WITH_INFO(
            maxNumCtasInBatchDim > 0, "Batched GEMM with dynamic batching requires maxNumCtasInBatchDim > 0");
    }

    if (mOptions.routeAct)
    {
        TLLM_CHECK_WITH_INFO(routeMap, "Batched GEMM with routeAct requires routeMap");
        TLLM_CHECK_WITH_INFO(numTokens > 0, "Batched GEMM with routeAct requires numTokens > 0");
    }

    // Dims
    gemmData.mProblemDimensions.mNumBatches = numBatches;
    gemmData.mProblemDimensions.mNumTokens = numTokens;
    gemmData.mProblemDimensions.mBatchM = !mOptions.transposeMmaOutput;
    gemmData.mProblemDimensions.mBatchedM = mOptions.transposeMmaOutput ? std::vector<int32_t>{} : batchedTokens;
    gemmData.mProblemDimensions.mBatchedN = mOptions.transposeMmaOutput ? batchedTokens : std::vector<int32_t>{};
    gemmData.mProblemDimensions.mM = mOptions.transposeMmaOutput ? n : m;
    gemmData.mProblemDimensions.mN = mOptions.transposeMmaOutput ? m : n;
    gemmData.mProblemDimensions.mK = k;
    gemmData.mProblemDimensions.mRank = 0;
    gemmData.mProblemDimensions.mWorldSize = 1;

    // Inputs
    gemmData.mInputBuffers.mPtrA = mOptions.transposeMmaOutput ? b : a;
    gemmData.mInputBuffers.mPtrSfA = mOptions.transposeMmaOutput ? sfB : sfA;
    gemmData.mInputBuffers.mPtrB = mOptions.transposeMmaOutput ? a : b;
    gemmData.mInputBuffers.mPtrSfB = mOptions.transposeMmaOutput ? sfA : sfB;
    gemmData.mInputBuffers.mPtrScaleC = scaleC;
    gemmData.mInputBuffers.mPtrScaleGate = scaleGateC;
    gemmData.mInputBuffers.mPtrPerTokenSfA = mOptions.transposeMmaOutput ? perTokensSfB : perTokensSfA;
    gemmData.mInputBuffers.mPtrPerTokenSfB = mOptions.transposeMmaOutput ? perTokensSfA : perTokensSfB;

    gemmData.mInputBuffers.mPtrRouteMap = routeMap;

    gemmData.mProblemDimensions.mMaxNumCtasInTokenDim = maxNumCtasInBatchDim;

    // Pointer to total number of padded tokens
    gemmData.mInputBuffers.mPtrTotalNumPaddedTokens = totalNumPaddedTokens;
    gemmData.mInputBuffers.mPtrCtaIdxXyToBatchIdx = ctaIdxXyToBatchIdx;
    gemmData.mInputBuffers.mPtrCtaIdxXyToMnLimit = ctaIdxXyToMnLimit;
    gemmData.mInputBuffers.mPtrNumNonExitingCtas = numNonExitingCtas;

    // Outputs
    gemmData.mOutputBuffers.mPtrC = c;
    gemmData.mOutputBuffers.mPtrSfC = outSfC;

    int32_t multiProcessorCount;
    cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, device);

    // FIXME once we start using all-reduce in the epilogue of the bmm this can be moved elsewhere
    bmm.runInitBeforeWorldSync(config, gemmData, static_cast<void*>(stream));

    auto const err = bmm.run(config, workspace, gemmData, static_cast<void*>(stream), multiProcessorCount);

    TLLM_CHECK_WITH_INFO(err == 0, "Error occurred when running GEMM!");
}

void TrtllmGenBatchedGemmRunner::run(int32_t m, int32_t n, int32_t k, std::vector<int32_t> const& batchedTokens,
    void const* a, void const* sfA, void const* b, void const* sfB, void* c, void* outSfC, void* workspace,
    CUstream stream, int device)
{
    // Dispatch with block scaling factors and with static batching.
    run(m, n, k, batchedTokens, /* numTokens */ 0, batchedTokens.size(), /* maxNumCtasInBatchDim */ 0, a, sfA, b, sfB,
        /* perTokensSfA */ nullptr, /* perTokensSfB */ nullptr,
        /* scaleC */ nullptr, /* scaleGateC */ nullptr, c, outSfC,
        /* routeMap */ nullptr, /* totalNumPaddedTokens */ nullptr,
        /* ctaIdxXyToBatchIdx */ nullptr, /* ctaIdxXyToMnLimit */ nullptr,
        /* numNonExitingCtas */ nullptr, workspace, stream, device);
}

void TrtllmGenBatchedGemmRunner::run(int32_t m, int32_t n, int32_t k, std::vector<int32_t> const& batchedTokens,
    void const* a, void const* b, float const* scaleC, float const* scaleGateC, void* c, void* workspace,
    CUstream stream, int device)
{
    // Dispatch with block scaling factors and with static batching.
    run(m, n, k, batchedTokens, /* numTokens */ 0, batchedTokens.size(), /* maxNumCtasInBatchDim */ 0, a,
        /* sfA */ nullptr, b, /* sfB */ nullptr, /* perTokensSfA */ nullptr, /* perTokensSfB */ nullptr, scaleC,
        scaleGateC, c, /* outSfC */ nullptr,
        /* routeMap */ nullptr, /* totalNumPaddedTokens */ nullptr,
        /* ctaIdxXyToBatchIdx */ nullptr, /* ctaIdxXyToMnLimit */ nullptr,
        /* numNonExitingCtas */ nullptr, workspace, stream, device);
}

void TrtllmGenBatchedGemmRunner::selectGemmConfig(int32_t m, int32_t n, int32_t k,
    std::vector<int32_t> const& batchedTokens, int32_t numTokens, int32_t numBatches, int32_t maxNumCtasInBatchDim)
{
    auto const bmm = batchedGemm::BatchedGemmInterface();
    auto const configs = bmm.getBatchedGemmConfigs();

    batchedGemm::BatchedGemmData gemmData;
    // Dims
    gemmData.mProblemDimensions.mNumBatches = numBatches;
    gemmData.mProblemDimensions.mNumTokens = numTokens;
    gemmData.mProblemDimensions.mBatchM = !mOptions.transposeMmaOutput;
    gemmData.mProblemDimensions.mBatchedM = mOptions.transposeMmaOutput ? std::vector<int32_t>{} : batchedTokens;
    gemmData.mProblemDimensions.mBatchedN = mOptions.transposeMmaOutput ? batchedTokens : std::vector<int32_t>{};
    gemmData.mProblemDimensions.mM = mOptions.transposeMmaOutput ? n : m;
    gemmData.mProblemDimensions.mN = mOptions.transposeMmaOutput ? m : n;
    gemmData.mProblemDimensions.mK = k;
    gemmData.mProblemDimensions.mRank = 0;
    gemmData.mProblemDimensions.mWorldSize = 1;
    gemmData.mProblemDimensions.mMaxNumCtasInTokenDim = maxNumCtasInBatchDim;
    // Sort configs by options
    std::vector<int32_t> sortedIndices = mPassingConfigIndices;
    std::sort(sortedIndices.begin(), sortedIndices.end(),
        [&configs](int32_t idx0, int32_t idx1)
        {
            auto const& optionsA = configs[idx0].mOptions;
            auto const& optionsB = configs[idx1].mOptions;

            // Sort by tileK sizes first
            if (optionsA.mTileK != optionsB.mTileK)
            {
                return optionsA.mTileK > optionsB.mTileK;
            }

            // Then by unroll loop 2x for mma
            return optionsA.mUseUnrollLoop2xForMma;
        });

    for (auto const& configIndex : sortedIndices)
    {
        auto const& config = configs[configIndex];
        auto isValidConfig = bmm.isValidConfig(config, gemmData);
        if (isValidConfig)
        {
            mSelectedConfigIndex = configIndex;
            return;
        }
    }
}

} // namespace kernels
} // namespace tensorrt_llm
