/*
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
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/runtime/common.h"
#include <curand_kernel.h>

namespace tensorrt_llm::kernels
{
template <typename T>
struct MinPSamplingKernelParams
{
    //! input buffer [batchSize, vocabSizePadded], required. Probabilities of each token in the vocab.
    T const* probs{nullptr};

    //! output buffer [maxBatchSize][maxSeqLen]. Contains pointers to rows with output tokens per request.
    //! If nullptr, outputIds must be provided.
    runtime::TokenIdType** outputIdsPtrs{nullptr};

    //! output buffer [maxBatchSize, maxSeqLen], optional. Tensor to store output tokens.
    //! Not used if outputIdsPtrs != nullptr
    runtime::TokenIdType* outputIds{nullptr};

    //! pointer to the workspace. Has to be pre-allocated by caller.
    //! Function does not take ownership of the buffer.
    void* workspace{nullptr};

    //! input buffer [maxBatchSize]. P for MinP sampling per request. Supported P is in range [0.0; 1.0].
    //! 1.0 will always select the token with the highest probability.
    //! 0.0 will disable the MinP filter and sample from all tokens.
    //! If nullptr, MinP of 0.0 is used for all requests.
    float const* minPs{nullptr};

    //! input buffer [maxBatchSize]. Temperature per request for late temperature adjustment.
    //! If nullptr, temperature of 1.0 is used for all requests.
    float const* temperatures{nullptr};

    //! input/output buffer [maxBatchSize], required. Current sequence length of the request up to, but excluding endId
    //! token.
    runtime::SizeType32* sequenceLength{nullptr};
    //! input buffer [maxBatchSize], optional. EOS token ids per request
    runtime::TokenIdType const* endIds{nullptr};
    //! input buffer[batchSize], optional. Indices of rows of data in memory pool.
    runtime::SizeType32 const* batchSlots{nullptr};

    //! input buffer [maxBatchSize], optional. Exit early if true.
    FinishedState const* finishedInput{nullptr};
    //! output buffer [maxBatchSize], optional. Set flag if sequence has finished (if finished || outputId == endId).
    FinishedState* finishedOutput{nullptr};
    //! input buffer [maxBatchSize], optional. Flags whether to skip decoding per request
    bool const* skipDecode{nullptr};

    //! input/output buffer [maxBatchSize], optional. Cumulative log probability of selected tokens. Ignored if nullptr.
    float* cumLogProbs{nullptr};
    //! output buffer [maxBatchSize], optional. Log probs is the probability induced by the MinP sampling.
    //! I.e., log_prob = log P(i | i is in vocab).
    float* outputLogProbs{nullptr};
    //! input buffer [maxBatchSize], optional. Curand states properly initialized using
    //! invokeCurandInitialize per request. Either curandState or randomVals should be specified.
    curandState_t* curandState{nullptr};
    //! input buffer [maxBatchSize], optional. Precomputed random values per request.
    //! Either curandState or randomVals should be specified.
    float const* randomVals{nullptr};

    runtime::SizeType32 batchSize{-1};
    runtime::SizeType32 maxBatchSize{-1};
    runtime::SizeType32 vocabSizePadded{-1};
    runtime::SizeType32 maxSeqLen{-1};

    bool returnAllSelectedTokens{false};

    //! output buffer [maxBatchSize], optional.
    //! Store the multinomial sampled target token id in TopK/MinP sampled tokens when returnAllSelectedTokens==True.
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
        TLLM_CHECK(probs);
        TLLM_CHECK(outputIds || outputIdsPtrs);
        TLLM_CHECK(workspace);
        TLLM_CHECK((curandState != nullptr) || (randomVals != nullptr));
        TLLM_CHECK(((curandState != nullptr) & (randomVals != nullptr)) == 0);
        TLLM_CHECK(minPs);
        TLLM_CHECK(temperatures);

        if (outputIds)
        {
            TLLM_CHECK(maxSeqLen > 0);
        }

        TLLM_CHECK(((finishedOutput == nullptr) ^ (endIds == nullptr)) == 0);
        TLLM_CHECK((skipOutputIdCurrentStep && outputIdCurrentStep && returnAllSelectedTokens)
            || (skipOutputIdCurrentStep == nullptr && outputIdCurrentStep == nullptr));
    }
};

//! \brief Returns workspace size in bytes needed for sampling MinP computation
//! \param batchSize batch size
//! \param vocabSizePadded size of padded vocab
template <typename T>
[[nodiscard]] size_t getMinPWorkspaceSize(runtime::SizeType32 batchSize, runtime::SizeType32 vocabSizePadded);

//! \brief Returns workspace size in bytes needed for initialization of sampling MinP
//! \param batchSize batch size
template <typename T>
[[nodiscard]] std::vector<size_t> getMinPInitWorkspaceSizes(runtime::SizeType32 batchSize);

//! \brief Given probs, performs Min P sampling. Fills sampled tokens to outputIds.
//! Updates sequenceLength, finished state, cumLogProbs inplace.
//! Sampling per request can be controlled using MinPs parameter.
template <typename T>
void invokeBatchMinPSampling(MinPSamplingKernelParams<T> const& params, cudaStream_t stream);

void invokeSetMinPRuntimeArgs(runtime::SizeType32 batchSize, ScatterDecodingParamEntry<float> minP,
    ScatterDecodingParamEntry<float> temperature, runtime::SizeType32 const* batchSlotsPtr,
    cudaStream_t stream = nullptr);

} // namespace tensorrt_llm::kernels
