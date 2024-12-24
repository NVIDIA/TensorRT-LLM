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

//! \brief Accepts or rejects draft tokens based on their probability distributions or the equality of draft and target
//! tokens. Corrects targetLogits for the last accepted token
//! according to https://openreview.net/pdf?id=C9NEblP8vS
//!
//! \param batchSize current batch size
//! \param draftProbs output buffer [maxDraftTokens, batchSize, beamWidth, vocabSize].
//! Workspace buffer for token probabilities of the draft model.
//! \param targetProbs output buffer [maxDraftTokens+1, batchSize, beamWidth, vocabSize].
//! Workspace buffer for token probabilities of the target model.
//! \param numsDraftTokens input buffer [batchSize]. Number of draft tokens per request
//! \param batchUseDraftLogits input buffer [batchSize]. Acceptance logic using draft logits or not, per request
//! \param draftIds input buffer [batchSize, draftTokens]. Pointer to draft token ids.
//! \param finishedInput input buffer [batchSize, beamWidth].
//! \param finishedOutput output buffer [batchSize, beamWidth]. At each step sets SKIP_DECODING if token is not
//! accepted.
//! \param curandState input buffer [batchSize]. Curand states properly initialized using invokeCurandInitialize
//! per request.
//! \param batchSlots input buffer [batchSize], address map from local index to global index [0, batchSize] ->
//! [0, maxBatchSize].
//! \param maxDraftTokens maximum number of draft tokens
//! \param beamWidth beam width (only beamWidth == 1 supported)
//! \param vocabSizePadded padded vocab size
//! \param randomThreshold True if use uniformly sampled threshold for token acceptance
//! \param constantThreshold threshold used to accept tokens if randomThreshold is false
//! \param step The current step of decoding (draft token id index)
//! \param batchIsAccepted output buffer [batchSize]. Stores acceptance result for multinomial sampling later or
//! forwarding next step.
//! \param targetOutputIds input/output buffer [batchSize]. Stores target sampling output ids for acceptById
//! logics.
//! \param stream stream
template <typename T>
void invokeAcceptDraftTokens(runtime::SizeType32 batchSize, T* draftProbs, T* targetProbs,
    runtime::SizeType32 const* numsDraftTokens, bool const* batchUseDraftLogits, runtime::TokenIdType const* draftIds,
    FinishedState const* finishedInput, FinishedState* finishedOutput, curandState_t* curandState,
    runtime::SizeType32 const* batchSlots, runtime::SizeType32 maxDraftTokens, runtime::SizeType32 beamWidth,
    runtime::SizeType32 vocabSizePadded, bool randomThreshold, float constantThreshold, runtime::SizeType32 step,
    bool* batchIsAccepted, runtime::SizeType32* targetOutputIds, cudaStream_t stream);

//! \brief Mask the target logits with -inf for unselected topK/topP token ids.
//! according to
//! https://github.com/huggingface/transformers/blob/2e24ee4dfa39cc0bc264b89edbccc373c8337086/src/transformers/generation/utils.py#L4064
//!
//! \param batchSize current batch size
//! \param targetLogits input/output buffer [batchSize][draftTokens+1, beamWidth, vocabSize].
//! Vector of pointers to the logits. (beamWidth == 1)
//! Initially contains token logits of the target model.
//! \param batchSlots input buffer [batchSize], address map from local index to global index [0, batchSize] ->
//! [0, maxBatchSize].
//! \param beamWidth beam width (only beamWidth == 1 supported)
//! \param vocabSizePadded padded vocab size
//! \param finishedInput input buffer [batchSize, beamWidth].
//! \param maxBatchSize maximum batch size
//! \param outputIdsAfterSampling input buffer [batchSize, vocabSize]. Stores all selected IDs from sampling for
//! masking.
//! \param numsDraftTokens input buffer [batchSize]. Number of draft tokens per request
//! \param runtimeTopKDevicePtr input buffer [batchSize] the topks in sampling step, for porting topK ids out.
//! \param maskBuffer input buffer [batchSize, vocabSize] for masking calculation (index value to position).
//! \param stream stream
template <typename T>
void invokeMaskTargetLogits(runtime::SizeType32 batchSize, T* targetLogits, runtime::SizeType32 const* batchSlots,
    runtime::SizeType32 beamWidth, runtime::SizeType32 vocabSizePadded, FinishedState const* finishedInput,
    runtime::SizeType32 maxBatchSize, runtime::SizeType32* outputIdsAfterSampling,
    runtime::SizeType32* runtimeTopKDevicePtr, bool* maskBuffer, cudaStream_t stream);

void invokeForwardAcceptedTokens(runtime::SizeType32 batchSize, runtime::SizeType32 const* batchSlots,
    bool* batchIsAccepted, runtime::SizeType32* outputSequenceLengths, runtime::TokenIdType const* draftIds,
    runtime::TokenIdType** idsPtrs, runtime::SizeType32 step, runtime::SizeType32 maxDraftTokens,
    runtime::TokenIdType const* endIds, FinishedState* finishedOutput, cudaStream_t stream);

} // namespace tensorrt_llm::kernels::speculative_decoding
