/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_runtime_api.h>

#include "tensorrt_llm/common/config.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::speculative_decoding::ngram
{

//! Parameters for the fused NGram history-extend and draft-lookup kernels.
//!
//! The pool keeps one token history per slot: ``historyTokens[slot * maxSeqLen + i]`` is the i-th token
//! of the sequence assigned to ``slot`` and ``historyLens[slot]`` is its current length. Every batch row
//! whose ``rowMask`` entry is non-zero first appends its accepted tokens to its slot, then proposes the
//! tokens that followed the longest (at most ``maxNgramSize``) earlier occurrence of its suffix.
struct NGramDraftParams
{
    int batchSize{0};                   //!< Number of rows (context + generation requests) in the batch.
    int draftLen{0};                    //!< Draft tokens emitted per row; 0 only extends the histories.
    int acceptedStride{0};              //!< Row stride of ``acceptedTokens``.
    int maxNgramSize{0};                //!< Longest suffix (in tokens) matched against the history.
    int numSlots{0};                    //!< Rows of the history pool.
    int maxSeqLen{0};                   //!< Columns of the history pool.
    bool useOldest{true};               //!< Draft from the earliest occurrence of the matched suffix (else the latest).
    bool publicPool{false};             //!< Also search the histories of the other slots when they are non-empty.
    int const* slotIds{nullptr};        //!< [batchSize] history slot of each row.
    int const* rowMask{nullptr};        //!< [batchSize] 1 to extend and draft the row, 0 to skip it.
    int const* acceptedTokens{nullptr}; //!< [batchSize, acceptedStride] tokens accepted this step.
    int const* numAcceptedTokens{nullptr}; //!< [batchSize] number of accepted tokens per row.
    int* historyTokens{nullptr};           //!< [numSlots, maxSeqLen] token history pool, updated in place.
    int* historyLens{nullptr};             //!< [numSlots] history length per slot, updated in place.
    int* draftTokensOut{nullptr};          //!< [batchSize, draftLen] proposed draft tokens, zero padded.
    int* matchLenOut{nullptr};             //!< [batchSize] matched suffix length, 0 when nothing was found.
};

//! Largest value ``NGramDraftParams::numSlots`` may take (the slot index is packed into the match key).
constexpr int kNGramMaxSlots = 1 << 16;
//! Largest value ``NGramDraftParams::maxNgramSize`` may take (the match length is packed into the match key).
constexpr int kNGramMaxNgramSize = (1 << 15) - 1;

//! Append the accepted tokens to each row's history and look up the next draft tokens. Both launches are
//! enqueued on ``stream`` and are CUDA graph capturable.
void invokeNGramExtendAndDraft(NGramDraftParams const& params, cudaStream_t stream);

} // namespace kernels::speculative_decoding::ngram

TRTLLM_NAMESPACE_END
