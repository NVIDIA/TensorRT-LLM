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
#include "tensorrt_llm/runtime/common.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace tensorrt_llm::kernels::speculative_decoding
{

//! \brief Linearly packs accepted paths in memory according to the accceptedLengths and bestPathIds
//!
//! \param acceptedLengthsCumSum input buffer [maxBatchSize + 1], exclusive sum of accepted lengths
//! (indexed linearly in memory).
//! \param pathsOffsets input buffer [maxBatchSize * maxDraftLen], slices of accepted paths packed in memory
//! \param acceptedLengths input buffer [maxBatchSize], length of the data accepted tokens
//! \param bestPathIds input buffer [maxBatchSize], indices of the selected paths
//! \param paths input buffer [batchSize, numPaths, maxPathLen] if isPathsLinearBatchIdx else [maxBatchSize, numPaths,
//! maxPathLen], paths to restore sequences from outputIds and targetIds. Should be filled with -1 for everything that
//! is not path. \param batchSlots input buffer [batchSize], address map from local index to global index [0, batchSize]
//! -> [0, maxBatchSize] \param batchSize current batch size \param numPaths maximum number of tokens per step
//! configured in the system \param maxPathLen maximum sequence length of the sequence containing draft tokens \param
//! isPathsLinearBatchIdx \param stream stream
void invokePackAcceptedPaths(runtime::SizeType32* acceptedLengthsCumSum, runtime::SizeType32* pathsOffsets,
    runtime::SizeType32 const* acceptedLengths, runtime::SizeType32 const* bestPathIds,
    runtime::SizeType32 const* paths, runtime::SizeType32 const* batchSlots, runtime::SizeType32 batchSize,
    runtime::SizeType32 numPaths, runtime::SizeType32 maxPathLen, bool isPathsLinearBatchIdx, cudaStream_t stream);

template <typename T>
struct AcceptDraftTokensByIdsWithPathsParams
{
    //! output buffer [maxBatchSize, maxSeqLen], input tokens.
    runtime::TokenIdType* outputIds{nullptr};
    //! input buffer [maxBatchSize, maxDecodingTokens], draft tokens
    runtime::TokenIdType const* draftIds{nullptr};
    //! input buffer [maxBatchSize, maxDecodingTokens], tokens predicted from the target medusa head
    runtime::TokenIdType const* targetIds{nullptr};
    //! input/output buffer [maxBatchSize], optional.
    //! Length of the data in outputIds without draft tokens.
    //! If set, incrememnted according to the accepted length.
    runtime::SizeType32* sequenceLengths{nullptr};
    //! output buffer [maxBatchSize], length of the data accepted tokens
    runtime::SizeType32* acceptedLengths{nullptr};
    //! input buffer [maxBatchSize], optional. Finished states per request
    FinishedState* finishedFinal{nullptr};
    //! input buffer [batchSize], optional. Address map from local index
    //! to global index [0, batchSize] -> [0, maxBatchSize].
    //! If nullptr, batchIdx is used.
    runtime::SizeType32 const* batchSlots{nullptr};
    //! input buffer [maxBatchSize, maxDecodingTokens, maxDraftPathLen+1],
    //! paths to restore sequences from outputIds and targetIds. Should be filled with -1 for everything that is not
    //! path.
    runtime::SizeType32 const* paths{nullptr};
    //! input buffer [maxBatchSize], optional. EOS ids per request.
    //! No EOS checks if nullptr.
    runtime::TokenIdType const* endIds{nullptr};
    //! input buffer [maxDraftPathLen, maxBatchSize, maxDecodingTokens, vocabSize], optional.
    //! Pointer to the logits from medusa heads.
    T const** medusaLogits{nullptr};
    //! output buffer [batchSize, maxDraftPathLen], optional. Contains pointers to the
    //! respective rows of the medusaLogits for the next after the accepted token
    T const** logitsPtrs{nullptr};
    //! current tokens to compute per step will be updated to
    //! targetTokensPerStep if curTokensPerStep == 1
    runtime::SizeType32* curTokensPerStep{nullptr};
    //! target values of tokens to compute per step
    runtime::SizeType32 const* targetTokensPerStep{nullptr};
    //! output buffer [maxBatchSize], indices of the selected paths
    runtime::SizeType32* bestPathIds{nullptr};
    //! current batch size
    runtime::SizeType32 batchSize{0};
    //! maximum batch size
    runtime::SizeType32 maxBatchSize{0};
    //! vocab size
    runtime::SizeType32 vocabSize{0};
    //! maximum sequence length of output ids
    runtime::SizeType32 maxSeqLen{0};
    //! maximum number of medusa heads
    runtime::SizeType32 maxDraftPathLen{0};
    //! maximum number of tokens per step configured in the system
    runtime::SizeType32 maxDecodingTokens{0};
    //! stream
    cudaStream_t stream;

    void checkParams() const
    {
        TLLM_CHECK(outputIds);
        TLLM_CHECK(draftIds);
        TLLM_CHECK(targetIds);
        TLLM_CHECK(acceptedLengths);
        TLLM_CHECK(paths);
        TLLM_CHECK(bestPathIds);
        TLLM_CHECK((curTokensPerStep == nullptr) ^ (targetTokensPerStep == nullptr) == 0);
        TLLM_CHECK((medusaLogits == nullptr) ^ (logitsPtrs == nullptr) == 0);

        TLLM_CHECK(batchSize > 0);
        TLLM_CHECK(batchSize <= maxBatchSize);
        TLLM_CHECK(vocabSize > 0);
        TLLM_CHECK(maxSeqLen > 0);
        TLLM_CHECK(maxDraftPathLen > 0);
        TLLM_CHECK(maxDecodingTokens > 0);
    }
};

//! \brief verifies draft medusa tokens given target tokens. Modifies outputIds tensor accordingly filling it with
//! accepted tokens. Fills logitsPtrs tensor with the pointers to the respective medusa logits tensor according
//! to the next after the last accepted token.
template <typename T>
void acceptDraftTokensByIdsWithPaths(AcceptDraftTokensByIdsWithPathsParams<T> const&);

} // namespace tensorrt_llm::kernels::speculative_decoding
