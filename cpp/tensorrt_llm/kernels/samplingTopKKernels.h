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
    //! input buffer [batchSize][vocabSizePadded] array of pointers to logits.
    //! If nullptr, logProbs is used. Only maxTokensPerStep == 1 is supported.
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
    //! output buffer [maxBatchSize]. Log probs is the probability induced by the top-k sampling.
    //! If normalizeLogProbs is true, we normalize the probability 'expLogit' of the selected token
    //! by the probability 's_sum' of a set of top-k tokens, meaning the logProb is the probability
    //! of the selected token, conditioned on the event that it is selected,
    //! i.e., log_prob = log P(i | i is in top-k) = log(expLogit / s_sum).
    //! Ignored if nullptr.
    float* outputLogProbs{nullptr};

    //! input buffer [maxBatchSize]. Initialized curand states
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
    //! flag to return all selectedTopK results
    bool returnAllTopK{false};

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
        TLLM_CHECK(curandState);

        TLLM_CHECK(maxTokensPerStep != 1 || returnAllTopK || sequenceLengths);
        TLLM_CHECK(maxTokensPerStep != 1 || returnAllTopK || endIds);
        if (cumLogProbs != nullptr || outputLogProbs != nullptr)
        {
            TLLM_CHECK(maxTokensPerStep == 1 && !returnAllTopK);
        }
        TLLM_CHECK(((finishedOutput == nullptr) ^ (endIds == nullptr)) == 0);

        TLLM_CHECK(0 < maxTopP && maxTopP <= 1.f);
        TLLM_CHECK(0 <= maxTopK && maxTopK <= TOP_K_MAX);
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
    return tensorrt_llm::common::calcAlignedSize(workspaceSizes, 256);
}

void invokeSetupTopKRuntimeArgs(runtime::SizeType32 batchSize, runtime::SizeType32 topK,
    runtime::SizeType32* runtimeTopKDevicePtr, runtime::SizeType32 runtimeTopKSize, float topP,
    float* runtimeTopPDevicePtr, runtime::SizeType32 runtimeTopPSize, bool* skipDecodeDevicePtr,
    runtime::SizeType32 const* batchSlotsDevicePtr, cudaStream_t stream);

} // namespace tensorrt_llm::kernels
