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

//! \brief Sets pointers to logits in logitsPtrs according to the draftDecodingTokens.
//! \param logitsPtrs [batchSize][vocabSizePadded]
//! \param decodingTokens [batchSize], on GPU. draftDecodingTokens + 1.
//! \param logits [numTokens, vocabSizePadded], on GPU. Continuous logits in memory.
//! \param draftDecodingTokens [batchSize], on GPU. 0 for context requests, and actual draft len for gen requests
//! \param batchSize batch size. Only batch size <= 512 is supported at the moment
//! \param maxDecodingTokens maximum number of decoding tokens per step per request
//! \param vocabSizePadded vocab size of the logits
//! \param stream cuda stream
template <typename T>
void invokeAssembleTargetLogitsOffsets(T const** logitsPtrs, runtime::SizeType32* decodingTokens, T const* logits,
    runtime::SizeType32 const* draftDecodingTokens, runtime::SizeType32 batchSize,
    runtime::SizeType32 maxDecodingTokens, runtime::SizeType32 vocabSizePadded, cudaStream_t stream);

//! \brief Sets last accepted token ids and computes inclusive sum of the indices of the last accepted tokens in
//! flattened input_ids tensor.
//! \param lastAcceptedTokenIds [batchSize], on GPU. Token ids of the last accepted tokens.
//! \param exclusiveSumLastAcceptedIndices [batchSize], on GPU. Exclusive sum of the positions of the last accepted
//! tokens in the original flattened draft sequence.
//! \param draftDecodingTokens [batchSize], on GPU. 0 for context
//! requests, and actual draft len for gen requests.
//! \param acceptedTokenIds [batchSize, maxPathLen], on GPU. Ids of the
//! accepted tokens per request.
//! \param acceptedLengths [batchSize], on GPU. Lengths of the accepted draft sequences
//! per request.
//! \param bestPathIds [batchSize], on GPU. Selected path id per request
//! \param paths [batchSize,
//! maxDecodingTokens, maxPathLen], on GPU. Indices of the draft sequences
//! \param batchSize batch size. Only batch size
//! <= 512 is supported at the moment
//! \param maxDecodingTokens maximum number of decoding tokens per step per request
//! \param maxPathLen maximum path len of the draft sequence
//! \param stream cuda stream
void invokeSelectLastAccTokenAndComputeIndicesCumSum(runtime::TokenIdType* lastAcceptedTokenIds,
    runtime::SizeType32* exclusiveSumLastAcceptedIndices, runtime::SizeType32 const* draftDecodingTokens,
    runtime::TokenIdType const* acceptedTokenIds, runtime::SizeType32 const* acceptedLengths,
    runtime::SizeType32 const* bestPathIds, runtime::SizeType32 const* paths, runtime::SizeType32 batchSize,
    runtime::SizeType32 maxDecodingTokens, runtime::SizeType32 maxPathLen, cudaStream_t stream);

} // namespace tensorrt_llm::kernels::speculative_decoding
