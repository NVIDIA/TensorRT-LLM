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
#include "tensorrt_llm/common/envUtils.h"
#include "trtllmGen_bmm_export/BatchedGemmInterface.h"
#include "trtllmGen_bmm_export/trtllm/gen/DtypeDecl.h"

namespace tensorrt_llm
{
namespace kernels
{

using namespace batchedGemm::batchedGemm;
using namespace batchedGemm::gemm;
using namespace batchedGemm::trtllm::gen;

std::vector<int64_t> prioritizePredefinedConfigs(int m, int n, int k, std::vector<int64_t> const& sortedIndices,
    batchedGemm::batchedGemm::BatchedGemmConfig const* configs)
{

    // Function to bubble up the pre-determined config.
    auto bubbleUpConfig = [&configs](std::vector<int64_t> const& sortedIndices, auto&& pred) -> std::vector<int64_t>
    {
        std::vector<int64_t> prioritizedIndices_;
        // Copy matching configs to new vector
        std::copy_if(sortedIndices.begin(), sortedIndices.end(), std::back_inserter(prioritizedIndices_),
            [&configs, &pred](int idx)
            {
                BatchedGemmConfig const& config = configs[idx];
                return (pred(config));
            });
        // Copy the rest of the configs to new vector, if not already copied
        std::copy_if(sortedIndices.begin(), sortedIndices.end(), std::back_inserter(prioritizedIndices_),
            [&prioritizedIndices_](int idx) {
                return std::find(prioritizedIndices_.begin(), prioritizedIndices_.end(), idx)
                    == prioritizedIndices_.end();
            });
        return prioritizedIndices_;
    };

    // Init empty vector
    std::vector<int64_t> prioritizedIndices;

    //
    // Qwen3
    //

    // Qwen3_235B_TP1_EP8_MoE_FC1 m=3072 k=4096
    if (n /* out_dim */ == 3072 && k /* in_dim */ == 4096)
    {
        auto pred = [](BatchedGemmConfig const& config)
        {
            BatchedGemmOptions const& options = config.mOptions;
            return options.mNumStages == 4 && options.mNumStagesMma == 1 && options.mTileK == 512
                && options.mTileScheduler == TileScheduler::Static;
        };
        prioritizedIndices = bubbleUpConfig(sortedIndices, pred);
    }
    // Qwen3_235B_TP1_EP8_MoE_FC2 m=4096 k=1536
    else if (n /* out_dim */ == 4096 && k /* in_dim */ == 1536)
    {
        auto pred = [](BatchedGemmConfig const& config)
        {
            BatchedGemmOptions const& options = config.mOptions;
            return options.mNumStages == 4 && options.mNumStagesMma == 1 && options.mTileK == 512
                && options.mTileScheduler == TileScheduler::Static;
        };
        prioritizedIndices = bubbleUpConfig(sortedIndices, pred);
    }
    // Qwen3_235B_TP2_EP4_MoE_FC1 m=1536 k=4096
    else if (n /* out_dim */ == 1536 && k /* in_dim */ == 4096)
    {
        auto pred = [](BatchedGemmConfig const& config)
        {
            BatchedGemmOptions const& options = config.mOptions;
            return options.mNumStages == 4 && options.mNumStagesMma == 1 && options.mTileK == 512
                && options.mTileScheduler == TileScheduler::Static;
        };
        prioritizedIndices = bubbleUpConfig(sortedIndices, pred);
    }
    // Qwen3_235B_TP2_EP4_MoE_FC2 m=4096 k=768
    else if (n /* out_dim */ == 4096 && k /* in_dim */ == 768)
    {
        auto pred = [](BatchedGemmConfig const& config)
        {
            BatchedGemmOptions const& options = config.mOptions;
            return options.mNumStages == 4 && options.mNumStagesMma == 2 && options.mTileK == 512
                && options.mTileScheduler == TileScheduler::Persistent;
        };
        prioritizedIndices = bubbleUpConfig(sortedIndices, pred);
    }
    // Qwen3_235B_TP4_EP2_MoE_FC1 m=768 k=4096
    else if (n /* out_dim */ == 768 && k /* in_dim */ == 4096)
    {
        auto pred = [](BatchedGemmConfig const& config)
        {
            BatchedGemmOptions const& options = config.mOptions;
            return options.mNumStages == 4 && options.mNumStagesMma == 1 && options.mTileK == 512
                && options.mTileScheduler == TileScheduler::Static;
        };
        prioritizedIndices = bubbleUpConfig(sortedIndices, pred);
    }
    // Qwen3_235B_TP4_EP2_MoE_FC2 m=4096 k=384
    else if (n /* out_dim */ == 4096 && k /* in_dim */ == 384)
    {
        auto pred = [](BatchedGemmConfig const& config)
        {
            BatchedGemmOptions const& options = config.mOptions;
            return options.mNumStages == 4 && options.mNumStagesMma == 2 && options.mTileK == 512
                && options.mTileScheduler == TileScheduler::Persistent;
        };
        prioritizedIndices = bubbleUpConfig(sortedIndices, pred);
    }
    // Qwen3_235B_TP8_EP1_MoE_FC1 m=384 k=4096
    else if (n /* out_dim */ == 384 && k /* in_dim */ == 4096)
    {
        auto pred = [](BatchedGemmConfig const& config)
        {
            BatchedGemmOptions const& options = config.mOptions;
            return options.mNumStages == 4 && options.mNumStagesMma == 1 && options.mTileK == 512
                && options.mTileScheduler == TileScheduler::Static;
        };
        prioritizedIndices = bubbleUpConfig(sortedIndices, pred);
    }
    // Qwen3_235B_TP8_EP1_MoE_FC2 m=4096 k=192
    else if (n /* out_dim */ == 4096 && k /* in_dim */ == 192)
    {
        auto pred = [](BatchedGemmConfig const& config)
        {
            BatchedGemmOptions const& options = config.mOptions;
            return options.mNumStages == 4 && options.mNumStagesMma == 2 && options.mTileK == 256
                && options.mTileScheduler == TileScheduler::Persistent;
        };
        prioritizedIndices = bubbleUpConfig(sortedIndices, pred);
    }
    //
    // Fall back
    //
    else
    {
        prioritizedIndices = sortedIndices;
    }

    return prioritizedIndices;
}

TrtllmGenBatchedGemmRunner::TrtllmGenBatchedGemmRunner(TrtllmGenBatchedGemmRunnerOptions const& options_)
    : mOptions(options_)
{
    // Select a GEMM kernel config to use
    auto const bmm = BatchedGemmInterface();
    auto const configs = bmm.getBatchedGemmConfigs();

    mPassingConfigIndices.clear();

    for (size_t i = 0; i < bmm.getNumBatchedGemmConfigs(); ++i)
    {
        auto const options = configs[i].mOptions;
        auto const tileSize = mOptions.transposeMmaOutput ? options.mTileN : options.mTileM;
        // When we include low-latency kernels we can set transposeMmaOutput via constructor
        if (options.mDtypeA == mOptions.dtypeA && options.mDtypeB == mOptions.dtypeB
            && options.mDtypeC == mOptions.dtypeC && options.mUseDeepSeekFp8 == mOptions.deepSeekFp8
            && options.mTransposeMmaOutput == mOptions.transposeMmaOutput
            && (!doesRouteImplUseNoRoute(options.mRouteImpl)) == mOptions.routeAct
            && options.mFusedAct == mOptions.fusedAct && options.mIsStaticBatch == mOptions.staticBatch
            && tileSize == mOptions.tileSize)
        {
            if (mOptions.transposeMmaOutput && options.mEpilogueTileM == mOptions.epilogueTileM)
            {
                mPassingConfigIndices.push_back(i);
            }
        }
    }

    TLLM_CHECK_WITH_INFO(!mPassingConfigIndices.empty(), "No kernel found for the given options");
}

size_t TrtllmGenBatchedGemmRunner::getWorkspaceSizeInBytes(int32_t m, int32_t n, int32_t k,
    std::vector<int32_t> const& batchedTokens, int32_t numTokens, int32_t numBatches, int32_t maxNumCtasInBatchDim,
    int32_t configIndex) const
{
    BatchedGemmData gemmData;
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

    auto bmm = BatchedGemmInterface();

    auto const configs = bmm.getBatchedGemmConfigs();

    auto const& config = configs[configIndex];

    return bmm.getWorkspaceSizeInBytes(config, gemmData);
}

void TrtllmGenBatchedGemmRunner::run(int32_t m, int32_t n, int32_t k, std::vector<int32_t> const& batchedTokens,
    int32_t numTokens, int32_t numBatches, int32_t maxNumCtasInBatchDim, void const* a, void const* sfA, void const* b,
    void const* sfB, void const* perTokensSfA, void const* perTokensSfB, float const* scaleC, float const* scaleGateC,
    float const* ptrBias, float const* ptrSwiGluAlpha, float const* ptrSwiGluBeta, void* c, void* outSfC,
    int32_t const* routeMap, int32_t const* totalNumPaddedTokens, int32_t const* ctaIdxXyToBatchIdx,
    int32_t const* ctaIdxXyToMnLimit, int32_t const* numNonExitingCtas, void* workspace, CUstream stream, int device,
    int32_t configIndex)
{
    auto bmm = BatchedGemmInterface();

    BatchedGemmData gemmData;

    auto const configs = bmm.getBatchedGemmConfigs();

    auto const& config = configs[configIndex];

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
    gemmData.mInputBuffers.mPtrBias = ptrBias;
    gemmData.mInputBuffers.mPtrSwiGluAlpha = ptrSwiGluAlpha;
    gemmData.mInputBuffers.mPtrSwiGluBeta = ptrSwiGluBeta;

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

    auto const enablePdl = tensorrt_llm::common::getEnvEnablePDL();
    auto const err = bmm.run(config, workspace, gemmData, static_cast<void*>(stream), multiProcessorCount, enablePdl);

    TLLM_CHECK_WITH_INFO(err == 0, "Error occurred when running GEMM!");
}

void TrtllmGenBatchedGemmRunner::run(int32_t m, int32_t n, int32_t k, std::vector<int32_t> const& batchedTokens,
    void const* a, void const* sfA, void const* b, void const* sfB, void* c, void* outSfC, void* workspace,
    CUstream stream, int device, int32_t configIndex)
{
    // Dispatch with block scaling factors and with static batching.
    run(m, n, k, batchedTokens, /* numTokens */ 0, batchedTokens.size(), /* maxNumCtasInBatchDim */ 0, a, sfA, b, sfB,
        /* perTokensSfA */ nullptr, /* perTokensSfB */ nullptr,
        /* scaleC */ nullptr, /* scaleGateC */ nullptr, /* ptrBias */ nullptr, /* ptrSwiGluAlpha */ nullptr,
        /* ptrSwiGluBeta */ nullptr, c, outSfC,
        /* routeMap */ nullptr, /* totalNumPaddedTokens */ nullptr,
        /* ctaIdxXyToBatchIdx */ nullptr, /* ctaIdxXyToMnLimit */ nullptr,
        /* numNonExitingCtas */ nullptr, workspace, stream, device, configIndex);
}

void TrtllmGenBatchedGemmRunner::run(int32_t m, int32_t n, int32_t k, std::vector<int32_t> const& batchedTokens,
    void const* a, void const* sfA, void const* b, void const* sfB, float const* ptrBias, float const* ptrSwiGluAlpha,
    float const* ptrSwiGluBeta, void* c, void* outSfC, void* workspace, CUstream stream, int device,
    std::optional<int32_t> configIndex)
{
    // Dispatch with block scaling factors and with static batching.
    run(m, n, k, batchedTokens, /* numTokens */ 0, batchedTokens.size(), /* maxNumCtasInBatchDim */ 0, a, sfA, b, sfB,
        /* perTokensSfA */ nullptr, /* perTokensSfB */ nullptr,
        /* scaleC */ nullptr, /* scaleGateC */ nullptr, ptrBias, ptrSwiGluAlpha, ptrSwiGluBeta, c, outSfC,
        /* routeMap */ nullptr, /* totalNumPaddedTokens */ nullptr,
        /* ctaIdxXyToBatchIdx */ nullptr, /* ctaIdxXyToMnLimit */ nullptr,
        /* numNonExitingCtas */ nullptr, workspace, stream, device, configIndex);
}

void TrtllmGenBatchedGemmRunner::run(int32_t m, int32_t n, int32_t k, std::vector<int32_t> const& batchedTokens,
    void const* a, void const* b, float const* scaleC, float const* scaleGateC, void* c, void* workspace,
    CUstream stream, int device, int32_t configIndex)
{
    // Dispatch with block scaling factors and with static batching.
    run(m, n, k, batchedTokens, /* numTokens */ 0, batchedTokens.size(), /* maxNumCtasInBatchDim */ 0, a,
        /* sfA */ nullptr, b, /* sfB */ nullptr, /* perTokensSfA */ nullptr, /* perTokensSfB */ nullptr, scaleC,
        scaleGateC, /* ptrBias */ nullptr, /* ptrSwiGluAlpha */ nullptr, /* ptrSwiGluBeta */ nullptr, c,
        /* outSfC */ nullptr,
        /* routeMap */ nullptr, /* totalNumPaddedTokens */ nullptr,
        /* ctaIdxXyToBatchIdx */ nullptr, /* ctaIdxXyToMnLimit */ nullptr,
        /* numNonExitingCtas */ nullptr, workspace, stream, device, configIndex);
}

std::vector<int64_t> TrtllmGenBatchedGemmRunner::getValidConfigIndices(int32_t m, int32_t n, int32_t k,
    std::vector<int32_t> const& batchedTokens, int32_t numTokens, int32_t numBatches,
    int32_t maxNumCtasInBatchDim) const
{
    auto const bmm = BatchedGemmInterface();
    auto const configs = bmm.getBatchedGemmConfigs();

    BatchedGemmData gemmData;
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
    std::vector<int64_t> sortedIndices = mPassingConfigIndices;
    std::sort(sortedIndices.begin(), sortedIndices.end(),
        [&configs](int64_t idx0, int64_t idx1)
        {
            auto const& optionsA = configs[idx0].mOptions;
            auto const& optionsB = configs[idx1].mOptions;

            // Sort by tileK sizes first
            if (optionsA.mTileK != optionsB.mTileK)
            {
                return optionsA.mTileK > optionsB.mTileK;
            }

            // Then by unroll loop 2x for mma
            if (optionsA.mUseUnrollLoop2xForMma != optionsB.mUseUnrollLoop2xForMma)
            {
                return optionsA.mUseUnrollLoop2xForMma;
            }

            // Then by tile scheduler (persistent scheduler is better for FC2 in MoE)
            if (doesRouteImplUseNoRoute(optionsA.mRouteImpl))
            {
                return optionsA.mTileScheduler == batchedGemm::gemm::TileScheduler::Persistent;
            }

            return optionsA.mTileM > optionsB.mTileM;
        });

    std::vector<int64_t> prioritizedIndices = prioritizePredefinedConfigs(m, n, k, sortedIndices, configs);
    std::vector<int64_t> validConfigIndices;
    for (auto const& configIndex : prioritizedIndices)
    {
        auto const& config = configs[configIndex];
        auto isValidConfig = bmm.isValidConfig(config, gemmData);
        if (isValidConfig)
        {
            validConfigIndices.push_back(configIndex);
        }
    }

    TLLM_CHECK_WITH_INFO(!validConfigIndices.empty(), "No valid config found for the given problem shape");

    return validConfigIndices;
}

int64_t TrtllmGenBatchedGemmRunner::getDefaultValidConfigIndex(int32_t m, int32_t n, int32_t k,
    std::vector<int32_t> const& batchedTokens, int32_t numTokens, int32_t numBatches,
    int32_t maxNumCtasInBatchDim) const
{
    auto const validConfigIndices
        = getValidConfigIndices(m, n, k, batchedTokens, numTokens, numBatches, maxNumCtasInBatchDim);

    return validConfigIndices[0];
}

bool TrtllmGenBatchedGemmRunner::isValidConfigIndex(int32_t configIndex, int32_t m, int32_t n, int32_t k,
    std::vector<int32_t> const& batchedTokens, int32_t numTokens, int32_t numBatches,
    int32_t maxNumCtasInBatchDim) const
{
    auto const bmm = BatchedGemmInterface();
    auto const configs = bmm.getBatchedGemmConfigs();

    BatchedGemmData gemmData;
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

    auto const& config = configs[configIndex];

    return bmm.isValidConfig(config, gemmData);
}

} // namespace kernels
} // namespace tensorrt_llm
