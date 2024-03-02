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

#include "tensorrt_llm/kernels/decodingCommon.h"
#include <curand_kernel.h>

namespace tensorrt_llm
{
namespace kernels
{
// clang-format off
//! \brief Given logProbs, performs top K **and** top P sampling at the same time. Fills sampled tokens to outputIds.
//! Computes sequenceLength, finished state, cumLogProbs inplace.
//! Sampling per request can be controlled using skipDecode, topPs and topKs parameters.
//! Function sets workspaceSize and exits early if workspace is nullptr.
//! If logits are Nan, we set output token to be the last in the vocabulary.
//!
//! \param workspace pointer to the workspace. Has to be pre-allocated by caller. Function does not take ownership of the
//! buffer.
//! \param workspaceSize size of the workspace in bytes
//! \param logProbs input buffer [batchSize x vocabSizePadded].
//! Log probabilities of each token in the vocab. If logitsHasProbs is true,
//! logProbs must contain **just** probabilities instead of log probabilities.
//! \param outputIds output buffer [maxBatchSize][maxSeqLen]. Contains pointers to rows with output tokens per request
//! \param sequenceLength input/output buffer [maxBatchSize]. Current sequence length of the request up to, but excluding endId token
//! \param finishedInput input buffer [maxBatchSize]. If true, request exits early.
//! \param finishedOutput output buffer [maxBatchSize]. Set flag if sequence has finished (if finished || outputId == endId).
//! \param cumLogProbs input/output buffer [maxBatchSize]. Cumulative log probability of selected tokens. Ignored if nullptr
//! \param outputLogProbs output buffer [maxBatchSize]. Log probs is the probability induced by the top-k sampling.
//! If normalizeLogProbs is true, we normalize the probability 'expLogit' of the selected token by the probability 's_sum' of a set of top-k
//! tokens, meaning the logProb is the probability of the selected token, conditioned on the event that it is selected,
//! i.e., log_prob = log P(i | i is in top-k) = log(expLogit / s_sum).
//! Ignored if nullptr.
//! \param curandstate input buffer [maxBatchSize]. Curand states properly
//! initialized using invokeCurandInitialize per request.
//! \param maxTopK maximum among all topKs K for topK sampling
//! \param topKs input buffer [maxBatchSize]. K for topK sampling per request.
//! Supported K is in range [1; 1024]. Where K=1 is greedy search. If nullptr maxTopK is used for all requests.
//! \param topP probability for topP sampling.
//! \param topPs input buffer [maxBatchSize]. Probability for topP sampling per request.
//! Supported P is in range (0.0, 1.0]. If nullptr, topP is used for all requests
//! \param vocabSizePadded size of padded vocab
//! \param endIds input buffer [maxBatchSize]. EOS token ids per request
//! \param batchSlots input buffer[batchSize], optional. Indices of rows of data in memory pool
//! \param stream cuda stream
//! \param batchSize batch size
//! \param maxBatchSize maximum batch size
//! \param skipDecode input buffer [maxBatchSize]. Flags whether to skip decoding per request
//! \param normalizeLogProbs when set to True outputLogProbs are normalized to TopK
//! \param logitsHasProbs flag to highlight that logProbs contains probabilities
// clang-format on
template <typename T>
void invokeBatchTopKSampling(void* workspace, size_t& workspaceSize, const T* logProbs, int** ids, int* sequenceLengths,
    const FinishedState* finishedInput, FinishedState* finishedOutput, float* cumLogProbs, float* outputLogProbs,
    curandState_t* curandstate, const int maxTopK, const int* topKs, const float topP, const float* topPs,
    const int vocabSizePadded, const int* endIds, const int* batchSlots, cudaStream_t stream, const int batchSize,
    int maxBatchSize, const bool* skipDecode, const bool normalizeLogProbs, const bool logitsHasProbs);

//! \brief Specialization of invokeBatchTopKSampling with topPs=nullptr and topKs=nullptr
template <typename T>
void invokeTopKSampling(void* workspace, size_t& workspaceSize, const T* logProbs, int** outputIds, int* sequenceLength,
    const FinishedState* finishedInput, FinishedState* finishedOutput, float* cumLogProbs, float* outputLogProbs,
    curandState_t* curandstate, const int topK, const float topP, const int vocabSizePadded, const int* endIds,
    const int* batchSlots, cudaStream_t stream, const int batchSize, int maxBatchSize, const bool* skipDecode,
    const bool normalizeLogProbs, const bool logitsHasProbs);

} // namespace kernels
} // namespace tensorrt_llm
