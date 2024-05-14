/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/runtime/common.h"
#include <curand_kernel.h>

namespace tensorrt_llm
{
namespace kernels
{
template <typename T>
struct TopPSamplingKernelParams
{
    //! input buffer [batchSize, vocabSizePadded], required. Probabilities of each token in the vocab.
    T const* probs{nullptr};

    //! output buffer [maxBatchSize][maxSeqLen], required. Contains pointers to rows with output tokens per request.
    runtime::TokenIdType** outputIds{nullptr};

    //! pointer to the workspace. Has to be pre-allocated by caller.
    //! Function does not take ownership of the buffer.
    void* workspace{nullptr};

    //! input buffer [maxBatchSize], required. P for topP sampling per request. Supported P is in range (0.0; 1.0].
    //! If nullptr maxTopP is used for all requests.
    float const* topPs{nullptr};

    //! input/output buffer [maxBatchSize], required. Current sequence length of the request up to, but excluding endId
    //! token.
    runtime::SizeType32* sequenceLength{nullptr};
    //! input buffer [maxBatchSize], optional. EOS token ids per request
    runtime::TokenIdType const* endIds{nullptr};
    //! input buffer[batchSize], optional. Indices of rows of data in memory pool.
    runtime::SizeType32 const* batchSlots;

    //! input buffer [maxBatchSize], optional. Exit early if true.
    FinishedState const* finishedInput{nullptr};
    //! output buffer [maxBatchSize], optional. Set flag if sequence has finished (if finished || outputId == endId).
    FinishedState* finishedOutput{nullptr};
    //! input buffer [maxBatchSize], optional. Flags whether to skip decoding per request
    bool const* skipDecode{nullptr};

    //! input/output buffer [maxBatchSize], optional. Cumulative log probability of selected tokens. Ignored if nullptr.
    float* cumLogProbs{nullptr};
    //! output buffer [maxBatchSize], optional. Log probs is the probability induced by the TopP sampling.
    //! I.e., log_prob = log P(i | i is in vocab).
    float* outputLogProbs{nullptr};
    //! input buffer [maxBatchSize], required. Curand states properly initialized using
    //! invokeCurandInitialize per request.
    curandState_t* curandState{nullptr};

    //! The appropriate block configuration calculated based on the number of multiprocessors, occupancy,
    //! batchSize and vocabSizePadded. Required for AirTopP
    runtime::SizeType32 blockNum{-1};
    //! bool, optional. Default value is false.
    //! When isDeterministic==true, the result is reproducible. Required for AirTopP
    bool isDeterministic{true};

    runtime::SizeType32 batchSize{-1};
    runtime::SizeType32 maxBatchSize{-1};
    runtime::SizeType32 vocabSizePadded{-1};

    void checkParams() const
    {
        TLLM_CHECK(batchSize > 0);
        TLLM_CHECK(maxBatchSize > 0);
        TLLM_CHECK(maxBatchSize >= batchSize);
        TLLM_CHECK(vocabSizePadded > 0);
        TLLM_CHECK(probs);
        TLLM_CHECK(outputIds);
        TLLM_CHECK(workspace);
        TLLM_CHECK(sequenceLength);
        TLLM_CHECK(curandState);
        TLLM_CHECK(topPs);

        TLLM_CHECK(((finishedOutput == nullptr) ^ (endIds == nullptr)) == 0);
    }
};

//! \brief Returns workspace size in bytes needed for sampling TopP computation
//! \param batchSize batch size
//! \param vocabSizePadded size of padded vocab
template <typename T>
[[nodiscard]] size_t getTopPWorkspaceSize(runtime::SizeType32 batchSize, runtime::SizeType32 vocabSizePadded);

// clang-format off
//! \brief Given probs, performs Top P sampling. Fills sampled tokens to outputIds.
//! Updates sequenceLength, finished state, cumLogProbs inplace.
//! Sampling per request can be controlled using skipDecode and topPs parameters.
// clang-format on
template <typename T>
void invokeBatchTopPSampling(TopPSamplingKernelParams<T> const& params, cudaStream_t stream);

//! \brief Compute the topp decay by https://arxiv.org/pdf/2206.04624.pdf
//!        In short, the formula is
//!          runtimeTopP = max(runtimeTopP * topPDecay, topPMin)
//!        If generating the topPResetIds, then reset the runtimeTopP.
//!
//! \param runtimeTopP
//! \param runtimeInitialTopP
//! \param outputIds
//! \param topPDecay
//! \param topPMin
//! \param topPResetIds
//! \param sequenceLengths
//! \param batchSlots input buffer[batchSize], optional. Indices of rows of data in memory pool
//! \param localBatchSize
void invokeComputeToppDecay(float* runtimeTopP, float const* runtimeInitialTopP, runtime::TokenIdType const** outputIds,
    float const* topPDecay, float const* topPMin, runtime::TokenIdType const* topPResetIds,
    runtime::SizeType32 const* sequenceLengths, runtime::SizeType32 const* batchSlots,
    runtime::SizeType32 localBatchSize, cudaStream_t stream);

//! \brief Given probs, performs top P sampling.
//! Note different from invokeTopPSampling() and invokeBatchTopPSampling() there two functions invokeAirTopPSampling
//! and invokeBatchAirTopPSampling is non-deterministic.
//! Fills sampled tokens to outputIds. Computes sequenceLength, finished state, cumLogProbs inplace.
//! Sampling per request can be controlled using skipDecode and topPs parameters.
//! Function sets workspaceSize  and exits early if workspace is nullptr.
//! When isDeterministic==true, the result is reproducible.
template <typename T>
void invokeBatchAirTopPSampling(TopPSamplingKernelParams<T> const& params, cudaStream_t stream);

//! \brief  Calculate the number of blocks based on the number of multiprocessors, batchSize and vocabSize.
//! \tparam T the data type of value
//! \param batchSize
//! \param len the number of candidates for each case
//! \param smCnt number of multiprocessors on device
//! \param isDeterministic bool, optional. Default value is false.
//! When isDeterministic==true, the result is reproducible.
template <typename T>
uint32_t calcAirTopPBlockNum(int batchSize, int len, int smCnt, bool isDeterministic = false);

//! \brief Returns workspace size in bytes needed for sampling Air TopP computation
//! \param batchSize batch size
//! \param vocabSizePadded size of padded vocab
//! \param isDeterministic bool, optional. Default value is false.
//! When isDeterministic==true, the result is reproducible.
template <typename T>
[[nodiscard]] size_t getAirTopPWorkspaceSize(int32_t batchSize, int32_t vocabSizePadded, bool isDeterministic = false);

} // namespace kernels
} // namespace tensorrt_llm
