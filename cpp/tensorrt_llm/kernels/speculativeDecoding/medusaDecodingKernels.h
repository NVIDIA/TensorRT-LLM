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

//! \brief verifies draft medusa tokens given target tokens. Modifies outputIds tensor accordingly filling it with
//! accepted tokens. Fills logitsPtrs tensor with the pointers to the respective medusa logits tensor according
//! to the next after the last accepted token.
//!
//! \param outputIds output buffer [maxBatchSize, maxSeqLen], input tokens.
//! \param draftIds input buffer [maxBatchSize, maxDecodingTokens], draft tokens
//! \param targetIds input buffer [maxBatchSize, maxDecodingTokens], tokens predicted from the target medusa head
//! \param sequenceLengths input/output buffer [maxBatchSize], length of the data in outputIds without draft tokens
//! Incrememnted according to the accepted length
//! \param acceptedLengths output buffer [maxBatchSize], length of the data accepted tokens
//! \param finishedFinal input buffer [maxBatchSize], finished states per request
//! \param batchSlots input buffer [batchSize], address map from local index
//! to global index [0, batchSize] -> [0, maxBatchSize]
//! \param paths input buffer [maxBatchSize, maxDecodingTokens, maxNumHeads+1],
//! paths to restore sequences from outputIds and targetIds. Should be filled with -1 for everything that is not path.
//! \param endIds input buffer [maxBatchSize], EOS ids per request
//! \param medusaLogits input buffer [maxNumHeads, maxBatchSize, maxDecodingTokens, vocabSize], pointer
//! to the logits from medusa heads
//! \param logitsPtrs output buffer [batchSize, maxNumHeads], contains pointers to the
//! respective rows of the medusaLogits for the next after the accepted token
//! \param curTokensPerStep current tokens to compute per step will be updated to
//! targetTokensPerStep if curTokensPerStep == 1
//! \param targetTokensPerStep target values of tokens to compute per step
//! \param bestPathIds output buffer [maxBatchSize], indices of the selected paths
//! \param batchSize current batch size
//! \param maxBatchSize maximum batch size
//! \param vocabSize vocab size
//! \param maxSeqLen maximum sequence length of output ids
//! \param maxNumHeads maximum number of medusa heads
//! \param maxDecodingTokens maximum number of tokens per step configured in the system
//! \param stream stream
template <typename T>
void acceptDraftTokensByIdsWithPaths(runtime::TokenIdType* outputIds, runtime::TokenIdType const* draftIds,
    runtime::TokenIdType const* targetIds, runtime::SizeType32* sequenceLengths, runtime::SizeType32* acceptedLengths,
    FinishedState* finishedFinal, runtime::SizeType32 const* batchSlots, runtime::SizeType32 const* paths,
    runtime::TokenIdType const* endIds, T const** medusaLogits, T const** logitsPtrs,
    runtime::SizeType32* curTokensPerStep, runtime::SizeType32 const* targetTokensPerStep,
    runtime::SizeType32* bestPathIds, runtime::SizeType32 batchSize, runtime::SizeType32 maxBatchSize,
    runtime::SizeType32 vocabSize, runtime::SizeType32 maxSeqLen, runtime::SizeType32 maxNumHeads,
    runtime::SizeType32 maxDecodingTokens, cudaStream_t stream);

//! \brief assembles draft tokens to treeDraftIds from sourceDraftIds using indices of treeIds
//!
//! \param treeDraftIds output buffer [maxBatchSize, maxDecodingTokens-1], output draft tokens
//! scattered from sourceDraftIds according to treeIds111
//! \param sourceDraftIds input buffer [maxBatchSize, maxDecodingTokens], draft tokens saved leanearly after
//! sampling from Medusa heads with TopK.
//! \param treeIds input buffer [maxBatchSize, maxDecodingTokens-1], address map from sourceDraftIds to treeDraftIds
//! [0, unqiueDraftTokens] -> [0, maxDecodingTokens], where unqiueDraftTokens = sum(MedusaHeadsTopK)
//! unqiueDraftTokens <= maxDraftTokens
//! \param tokensPerStep input buffer [maxBatchSize], number of output draft tokens
//! \param batchSlots input buffer [maxBatchSize], address map from local index
//! to global index [0, batchSize] -> [0, maxBatchSize]
//! \param maxDecodingTokens maximum number of tokens per step configured in the system
//! \param batchSize current batch size
//! \param stream cuda stream
void scatterMedusaDraftTokens(runtime::TokenIdType* treeDraftIds, runtime::TokenIdType const* sourceDraftIds,
    runtime::SizeType32 const* treeIds, runtime::SizeType32 const* tokensPerStep, runtime::SizeType32 const* batchSlots,
    runtime::SizeType32 maxDecodingTokens, runtime::SizeType32 batchSize, cudaStream_t stream);

} // namespace tensorrt_llm::kernels::speculative_decoding
