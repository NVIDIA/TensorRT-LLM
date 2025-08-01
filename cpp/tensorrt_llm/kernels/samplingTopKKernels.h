/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/runtime/common.h"
#include <curand_kernel.h>

namespace tensorrt_llm::kernels
{

static constexpr runtime::SizeType32 TOP_K_MAX = 1024;

template <typename T>
struct TopKSamplingKernelParams
{

    //! Input buffer [batchSize, maxTokensPerStep, vocabSizePadded].
    //! Log probabilities of each token in the vocab. If logitsHasProbs is true,
    //! logProbs must contain **just** probabilities instead of log probabilities.
    T const* logProbs{nullptr};
    //! input buffer [batchSize][tokensPerStep, vocabSizePadded] array of pointers to logits.
    //! If nullptr, logProbs is used.
    T const* const* logProbsPtrs{nullptr};

    //! output buffer [maxBatchSize][maxSeqLen], optional. Contains pointers to rows
    //! with output tokens per request. If nullptr, outputIds must be provided.
    runtime::TokenIdType** outputIdsPtrs{nullptr};
    //! output buffer [maxBatchSize, maxSeqLen], optional. Tensor to store output tokens.
    //! Not used if outputIdsPtrs != nullptr
    runtime::TokenIdType* outputIds{nullptr};

    //! Required. Pointer to the workspace of size returned by getTopKWorkspaceSize.
    //! Has to be pre-allocated by caller.
    //! Function does not take ownership of the buffer
    void* workspace{nullptr};

    //! input buffer [maxBatchSize], optional. EOS token ids per request
    runtime::TokenIdType const* endIds{nullptr};

    //! input/output buffer [maxBatchSize], optional. If nullptr, seqLen is 0
    //! Current sequence length of the request. Set up to, but excluding endId token.
    runtime::SizeType32* sequenceLengths{nullptr};
    //! input buffer[batchSize], optional. Indices of rows of data in memory pool.
    //! Linear indexing (batchIdx) is used if nullptr.
    runtime::SizeType32 const* batchSlots{nullptr};
    //! input buffer [maxBatchSize], optional. Number of tokens per step for each request.
    //! It is assumed that all requests have maxTokensPerStep tokens per step if nullptr.
    runtime::SizeType32 const* tokensPerStep{nullptr};

    //! input buffer [maxBatchSize], optional. If true, request exits early.
    FinishedState const* finishedInput{nullptr};
    //! output buffer [maxBatchSize], optional.
    //! Set to true if sequence has finished (if finished || outputId == endId).
    FinishedState* finishedOutput{nullptr};
    //! input buffer [maxBatchSize]. Flags whether to skip decoding per request
    bool const* skipDecode{nullptr};

    //! input/output buffer [maxBatchSize], optional.
    //! Cumulative log probability of selected tokens. Ignored if nullptr
    float* cumLogProbs{nullptr};
    //! output buffer
    //! [maxBatchSize, maxTopK] when returnAllSelectedTokens, otherwise [maxSeqLen, maxBatchSize]
    //! Log probs is the probability induced by the top-k sampling.
    //! If normalizeLogProbs is true, we normalize the probability 'expLogit' of the selected token
    //! by the probability 's_sum' of a set of top-k tokens, meaning the logProb is the probability
    //! of the selected token, conditioned on the event that it is selected,
    //! i.e., log_prob = log P(i | i is in top-k) = log(expLogit / s_sum).
    //! Ignored if nullptr.
    float* outputLogProbs{nullptr};

    //! input buffer [maxBatchSize], optional. Initialized curand states.
    //! If nullptr, 1 is always used.
    curandState_t* curandState{nullptr};
    //! input buffer [maxBatchSize]. K for topK sampling per request.
    //! Supported K is in range [1; 1024]. Where K=1 is greedy search.
    //! If nullptr maxTopK is used for all requests.
    runtime::SizeType32 const* topKs{nullptr};
    //! input buffer [maxBatchSize]. Probability for topP sampling per request.
    //! Supported P is in range (0.0, 1.0]. If nullptr, topP is used for all requests
    float const* topPs{nullptr};
    //! maximum among all topKs K for topK sampling
    runtime::SizeType32 maxTopK{TOP_K_MAX};
    //! probability for topP sampling.
    float maxTopP{1.0f};

    runtime::SizeType32 batchSize{-1};
    runtime::SizeType32 maxBatchSize{-1};
    runtime::SizeType32 vocabSizePadded{-1};
    runtime::SizeType32 maxTokensPerStep{-1};
    runtime::SizeType32 maxSeqLen{-1};

    //! when set to True outputLogProbs are normalized to TopK
    bool normalizeLogProbs{false};
    //! flag to highlight that logProbs contains probabilities
    bool logitsHasProbs{false};
    //! flag to return all selected TopK results
    bool returnAllSelectedTokens{false};
    //! flag to set strict TopP boundary.
    //! If true, when randNum <=0.0f, the selection is completed, even if K draft tokens are not reached.
    //! If false, when randNum <=0.0f, the selection will continue until it reaches K tokens.
    bool strictTopPBoundary{true};

    //! flag to return all selected TopK results per request.
    bool const* returnAllSelectedTokensPerSlot{nullptr};

    //! output buffer [maxBatchSize], optional.
    //! Store the multinomial sampled target token id in TopK/TopP sampled tokens when returnAllSelectedTokens==True.
    //! Only return when skipOutputIdCurrentStep != nullptr && skipOutputIdCurrentStep == False
    runtime::TokenIdType* outputIdCurrentStep{nullptr};
    //! input buffer [maxBatchSize]. Determine if multinomial sampling is required when returnAllSelectedTokens==True.
    bool const* skipOutputIdCurrentStep{nullptr};

    void checkParams() const
    {
        TLLM_CHECK(batchSize > 0);
        TLLM_CHECK(maxBatchSize > 0);
        TLLM_CHECK(maxBatchSize >= batchSize);
        TLLM_CHECK(vocabSizePadded > 0);
        TLLM_CHECK(maxTokensPerStep > 0);

        TLLM_CHECK(logProbs || logProbsPtrs);
        TLLM_CHECK(outputIds || outputIdsPtrs);

        if (maxTokensPerStep > 1)
        {
            TLLM_CHECK(tokensPerStep);
        }

        if (outputIds)
        {
            TLLM_CHECK(maxSeqLen > 0);
        }

        TLLM_CHECK(workspace);

        TLLM_CHECK(maxTokensPerStep != 1 || returnAllSelectedTokens || sequenceLengths);
        TLLM_CHECK(maxTokensPerStep != 1 || returnAllSelectedTokens || endIds);
        if (cumLogProbs != nullptr || outputLogProbs != nullptr)
        {
            TLLM_CHECK(maxTokensPerStep == 1);
            if (cumLogProbs != nullptr)
            {
                TLLM_CHECK(!returnAllSelectedTokens);
            }
        }

        TLLM_CHECK(((finishedOutput == nullptr) ^ (endIds == nullptr)) == 0);
        TLLM_CHECK_WITH_INFO(0 < maxTopP && maxTopP <= 1.f, "maxTopP (%f) is out of range", maxTopP);
        TLLM_CHECK_WITH_INFO(0 <= maxTopK && maxTopK <= TOP_K_MAX, "maxTopK (%d) is out of range", maxTopK);
        TLLM_CHECK((skipOutputIdCurrentStep && outputIdCurrentStep && returnAllSelectedTokens)
            || (skipOutputIdCurrentStep == nullptr && outputIdCurrentStep == nullptr));
    }
};

// clang-format off
//! \brief Given logProbs, performs top K **and** top P sampling at the same time. Fills sampled tokens to outputIds.
//! Computes sequenceLength, finished state, cumLogProbs inplace.
//! Sampling per request can be controlled using skipDecode, topPs and topKs parameters.
//! Function sets workspaceSize and exits early if workspace is nullptr.
//! If logits are Nan, we set output token to be the last in the vocabulary.
// clang-format on
template <typename T>
void invokeBatchTopKSampling(TopKSamplingKernelParams<T> const& params, cudaStream_t stream);

template <typename T>
[[nodiscard]] std::vector<size_t> getTopKWorkspaceSizes(runtime::SizeType32 batchSize,
    runtime::SizeType32 maxTokensPerStep, runtime::SizeType32 maxTopK, runtime::SizeType32 vocabSizePadded)
{
    runtime::SizeType32 constexpr maxBlockPerBeam = 8;
    auto const tempLogProbsBufSize = sizeof(T) * batchSize * maxTokensPerStep * vocabSizePadded;         // type T
    auto const topKTmpIdsBufSize
        = sizeof(runtime::SizeType32) * batchSize * maxTokensPerStep * maxTopK * maxBlockPerBeam;        // type int
    auto const topKTmpValBufSize = sizeof(T) * batchSize * maxTokensPerStep * maxTopK * maxBlockPerBeam; // type T

    return {tempLogProbsBufSize, topKTmpIdsBufSize, topKTmpValBufSize};
}

[[nodiscard]] inline std::vector<size_t> getTopKInitWorkspaceSizes(runtime::SizeType32 batchSize)
{
    auto const tempTopKsBufSize = batchSize * sizeof(runtime::SizeType32);
    auto const tempTopPsBufSize = batchSize * sizeof(float);

    return {tempTopKsBufSize, tempTopPsBufSize};
}

//! \brief Returns workspace size in bytes needed for sampling TopK computation
//! \param batchSize batch size
//! \param maxTokensPerStep maximum number of tokens per computed per step
//! \param maxTopK maximum among all topKs K for topK sampling
//! \param vocabSizePadded size of padded vocab
template <typename T>
[[nodiscard]] size_t getTopKWorkspaceSize(runtime::SizeType32 batchSize, runtime::SizeType32 maxTokensPerStep,
    runtime::SizeType32 maxTopK, runtime::SizeType32 vocabSizePadded)
{
    auto const workspaceSizes = getTopKWorkspaceSizes<T>(batchSize, maxTokensPerStep, maxTopK, vocabSizePadded);
    auto const initWorkspaceSizes = getTopKInitWorkspaceSizes(batchSize);
    return std::max(tensorrt_llm::common::calcAlignedSize(workspaceSizes, 256),
        tensorrt_llm::common::calcAlignedSize(initWorkspaceSizes, 256));
}

void invokeSetupTopKRuntimeArgs(runtime::SizeType32 batchSize, ScatterDecodingParamEntry<runtime::SizeType32> topK,
    ScatterDecodingParamEntry<float> topP, bool* skipDecodePtr, runtime::SizeType32 const* batchSlotsPtr, bool onDevice,
    cudaStream_t stream = nullptr);

void invokeSetupTopKTopPRuntimeArgs(runtime::SizeType32 batchSize, ScatterDecodingParamEntry<runtime::SizeType32> topK,
    ScatterDecodingParamEntry<float> topP, bool* skipDecodeTopKPtr, bool* skipDecodeTopPPtr,
    runtime::SizeType32 const* batchSlotsPtr, bool onDevice, cudaStream_t stream = nullptr);

inline bool clampTopP(float& topP)
{
    if (topP < 0.f || topP > 1.0f)
    {
        TLLM_LOG_WARNING("TopP (%f) is out of range ([0.0, 1.0f]). Clip to closest number.", topP);
        topP = std::clamp(topP, 0.f, 1.f);
        return true;
    }

    return false;
}

inline bool clampTopK(runtime::SizeType32& topK)
{
    if (topK < 0 || topK > TOP_K_MAX)
    {
        TLLM_LOG_WARNING(
            "TopK (%d) is larger than max supported number (%d). Clip to max supported number.", topK, TOP_K_MAX);
        topK = std::clamp(topK, 0, TOP_K_MAX);
        return true;
    }

    return false;
}

inline bool regularizeTopKTopP(runtime::SizeType32& topK, float& topP)
{
    bool modified = false;
    if (topK == 0 && topP == 0.0f)
    {
        // TensorRT-LLM's topp implementation does not support topp = 0.0f, but it
        // equivalent to greedy search. So, we set the topk = 1 as an alternative
        // solution.
        topK = 1;
        modified = true;
    }
    if (topK > 0 && topP == 0.0f)
    {
        // This case corresponds to the old topk sampling, which is equivalent to
        // the old topk_topp sampling with topp=1.0f. TopKSamplingLayer and
        // TopKTopPSamplingLayer are now merged by TopKSamplingLayer. Thus, we
        // replace the case topk>0 and topp=0.0f by topk>0 and topp=1.0f for the
        // compatibility.
        topP = 1.0f;
        modified = true;
    }

    return modified;
}

__device__ __host__ inline void setupTopKTopPRuntimeArgOne(runtime::SizeType32 batchIndex,
    ScatterDecodingParamEntry<runtime::SizeType32> topK, ScatterDecodingParamEntry<float> topP,
    runtime::SizeType32 const* batchSlots, bool* skipDecodeTopK, bool* skipDecodeTopP, float* initialTopPBuf)
{
    auto const batchSlot = batchSlots[batchIndex];
    auto const k = topK.mVector == nullptr ? topK.mScalar : topK.mVector[batchIndex];
    auto const p = topP.mVector == nullptr ? topP.mScalar : topP.mVector[batchIndex];
    if (topK.mTarget != nullptr)
    {
        topK.mTarget[batchSlot] = k;
    }
    if (topP.mTarget != nullptr)
    {
        topP.mTarget[batchSlot] = p;
    }
    if (skipDecodeTopK != nullptr)
    {
        skipDecodeTopK[batchSlot] = k == 0;
    }
    if (skipDecodeTopP != nullptr)
    {
        skipDecodeTopP[batchSlot] = k != 0;
    }
    if (initialTopPBuf != nullptr)
    {
        initialTopPBuf[batchSlot] = p;
    }
}

} // namespace tensorrt_llm::kernels
