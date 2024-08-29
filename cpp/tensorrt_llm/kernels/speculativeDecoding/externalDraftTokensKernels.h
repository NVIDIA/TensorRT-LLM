/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/kernels/speculativeDecoding/common.h"
#include "tensorrt_llm/runtime/common.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace tensorrt_llm::kernels::speculative_decoding
{

//! \brief Accepts or rejects draft tokens based on the equality of draft and target tokens
//! for speculative decoding. Target token is accepted if targetToken == draftToken.
//! If number of accepted tokens N < maxDraftTokens, then function accepts N + 1 tokens of target model.
//! sequenceLengths, finishedSum and finishedFinal are modified accordingly.
//!
//! \param draftIds input buffer [batchSize, maxDraftTokens].
//! Indices of the draft tokens.
//! \param targetIds input buffer [batchSize, maxSeqLen]. Indices of the tokens decoded by the target model
//! \param contextLengths input buffer [batchSize]. Context lengths of the requests without draft tokens
//! \param numsDraftTokens input buffer [batchSize]. Number of draft tokens per request
//! \param sequenceLengths input/output buffer [batchSize] sequence lengths of the requests in batch
//! Modified in-place according to the accepted/rejected tokens
//! \param finished input buffer [maxDraftTokens + 1, batchSize] finished states at each decoding iteration
//! \param finishedFinal output buffer [batchSize] finished states after accepting/rejecting tokens
//! \param finishedSum output buffer [1] total number of requests in batch that finished the execution
//! \param batchSlots input buffer [batchSize], address map from local index
//! to global index [0, batchSize] -> [0, maxBatchSize]
//! \param batchSize current batch size
//! \param maxBatchSize maximum batch size
//! \param beamWidth beam width
//! \param maxSeqLen maximum sequence length
//! \param maxDraftTokens maximum number of draft tokens
//! \param stream stream
void invokeAcceptDraftTokensByIds(runtime::TokenIdType const* draftIds, runtime::TokenIdType const* targetIds,
    runtime::SizeType32 const* contextLengths, runtime::SizeType32 const* numsDraftTokens,
    runtime::SizeType32* sequenceLengths, FinishedState const* finished, FinishedState* finishedFinal,
    runtime::SizeType32* finishedSum, runtime::SizeType32 const* batchSlots, runtime::SizeType32 batchSize,
    runtime::SizeType32 maxBatchSize, runtime::SizeType32 beamWidth, runtime::SizeType32 maxSeqLen,
    runtime::SizeType32 maxDraftTokens, cudaStream_t stream);

//! \brief Performs probabilistic acceptance of draft tokens based on their probability distributions.
//! Corrects targetLogits for the next to the last accepted token
//! according to https://openreview.net/pdf?id=C9NEblP8vS
//!
//! \param draftLogits input/output buffer [draftTokens, batchSize, beamWidth, vocabSize].
//! Initially contains token logits of the draft model.
//! \param targetLogits input/output buffer [batchSize][draftTokens+1, beamWidth, vocabSize].
//! Vector of pointers to the logits.
//! Initially contains token logits of the target model.
//! It is modified in-place for next to the last accepted token such as
//! P'(x) = norm(max(0, P_{n+1}(x) - Q_{n+1}(x))), where N < maxDraftTokens is number of accepted tokens.
//! \param draftProbs output buffer [maxDraftTokens, batchSize, beamWidth, vocabSize].
//! Workspace buffer for token probabilities of the draft model.
//! \param targetProbs output buffer [maxDraftTokens+1, batchSize, beamWidth, vocabSize].
//! Workspace buffer for token probabilities of the target model.
//! \param numsDraftTokens input buffer [batchSize]. Number of draft tokens per request
//! \param finished output buffer [draftTokens, batchSize, beamWidth].
//! At each step sets to NOT_FINISHED if token is accepted or SKIP_DECODING if token is not accepted
//! \param curandState input buffer [batchSize]. Curand states properly
//! initialized using invokeCurandInitialize per request.
//! \param batchSlots input buffer [batchSize], address map from local index
//! to global index [0, batchSize] -> [0, maxBatchSize]
//! \param batchSize current batch size
//! \param maxBatchSize maximum batch size
//! \param beamWidth beam width
//! \param vocabSize unpadded vocab size
//! \param vocabSizePadded padded vocab size
//! \param maxDraftTokens maximum number of draft tokens
//! \param randomThreshold True if use uniformly sampled threshold for token acceptance
//! \param constantThreshold threshold used to accept tokens if randomThreshold is false
//! \param stream stream
template <typename T>
void acceptDraftTokensByLogits(T* draftLogits, T** targetLogits, T* draftProbs, T* targetProbs,
    runtime::SizeType32 const* numsDraftTokens, FinishedState* finished, curandState_t* curandState,
    runtime::SizeType32 const* batchSlots, runtime::SizeType32 batchSize, runtime::SizeType32 maxBatchSize,
    runtime::SizeType32 beamWidth, runtime::SizeType32 vocabSize, runtime::SizeType32 vocabSizePadded,
    runtime::SizeType32 maxDraftTokens, bool randomThreshold, float constantThreshold, cudaStream_t stream);

struct Candidate // Hold probability maximum and rate of target / dfraft, used in `acceptDraftTokensByLogits`
{
    float maxProb{0.f};
    float rateQP{0.f};
};

__device__ __forceinline__ Candidate reduce_op(Candidate const& a, Candidate const& b)
{
    // Max-reduce operator of Candidate
    return (a.maxProb > b.maxProb) ? a : b;
}

} // namespace tensorrt_llm::kernels::speculative_decoding
