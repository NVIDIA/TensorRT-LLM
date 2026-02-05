/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 *
 * Adapted from Baseten's sa_spec library (Apache-2.0)
 * https://github.com/basetenlabs/sa_spec
 */

#pragma once

#include "suffixAutomaton.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::speculative_decoding::suffix_automaton
{

//! \brief Parameters for the suffix automaton extend kernel
struct SuffixAutomatonExtendParams
{
    //! Number of sequences in the batch
    int batchSize{0};

    //! Number of draft tokens to generate per sequence
    int draftLength{0};

    //! Pointer to the suffix automaton states on GPU [maxSlots]
    SuffixAutomaton* slots{nullptr};

    //! Batch indices mapping external batch idx to workspace slot [batchSize]
    int const* batchIndices{nullptr};

    //! Output: match lengths for each sequence [batchSize]
    int* matchLenOut{nullptr};

    //! Output: draft tokens for each sequence [batchSize, draftLength]
    int* draftTokensOut{nullptr};

    //! Input: accepted tokens for each sequence [batchSize, draftLength + 1]
    int const* acceptedTokensIn{nullptr};

    //! Input: number of accepted tokens for each sequence [batchSize]
    int const* acceptedLensIn{nullptr};

    void checkParams() const
    {
        TLLM_CHECK(batchSize > 0);
        TLLM_CHECK(draftLength > 0);
        TLLM_CHECK(slots != nullptr);
        TLLM_CHECK(batchIndices != nullptr);
        TLLM_CHECK(matchLenOut != nullptr);
        TLLM_CHECK(draftTokensOut != nullptr);
        TLLM_CHECK(acceptedTokensIn != nullptr);
        TLLM_CHECK(acceptedLensIn != nullptr);
    }
};

//! \brief Invokes the suffix automaton extend kernel
//!
//! This kernel updates the suffix automaton states for each sequence in the batch
//! with the newly accepted tokens, then performs a lookup to find the longest
//! suffix match and generates draft tokens based on that match.
//!
//! \param params The parameters for the kernel
//! \param stream The CUDA stream to run the kernel on
void invokeSuffixAutomatonExtend(SuffixAutomatonExtendParams const& params, cudaStream_t stream);

//! \brief Parameters for the suffix automaton extend kernel with ngram support
struct SuffixAutomatonExtendNgramParams
{
    //! Number of sequences in the batch
    int batchSize{0};

    //! Number of draft tokens to generate per sequence
    int draftLength{0};

    //! Maximum ngram size for matching, or -1 for longest match mode
    int maxNgramSize{-1};

    //! Pointer to the suffix automaton states on GPU [maxSlots]
    SuffixAutomaton* slots{nullptr};

    //! Batch indices mapping external batch idx to workspace slot [batchSize]
    int const* batchIndices{nullptr};

    //! Output: match lengths for each sequence [batchSize]
    int* matchLenOut{nullptr};

    //! Output: draft tokens for each sequence [batchSize, draftLength]
    int* draftTokensOut{nullptr};

    //! Input: accepted tokens for each sequence [batchSize, draftLength + 1]
    int const* acceptedTokensIn{nullptr};

    //! Input: number of accepted tokens for each sequence [batchSize]
    int const* acceptedLensIn{nullptr};

    void checkParams() const
    {
        TLLM_CHECK(batchSize > 0);
        TLLM_CHECK(draftLength > 0);
        TLLM_CHECK(slots != nullptr);
        TLLM_CHECK(batchIndices != nullptr);
        TLLM_CHECK(matchLenOut != nullptr);
        TLLM_CHECK(draftTokensOut != nullptr);
        TLLM_CHECK(acceptedTokensIn != nullptr);
        TLLM_CHECK(acceptedLensIn != nullptr);
    }
};

//! \brief Invokes the suffix automaton extend kernel with ngram support
//!
//! This kernel updates the suffix automaton states for each sequence in the batch
//! with the newly accepted tokens, then performs lookup based on maxNgramSize:
//! - If maxNgramSize == -1: finds the longest suffix match
//! - If maxNgramSize > 0: tries ngram sizes from maxNgramSize down to 1 until a match is found
//!
//! This kernel is CUDA graph compatible.
//!
//! \param params The parameters for the kernel
//! \param stream The CUDA stream to run the kernel on
void invokeSuffixAutomatonExtendNgram(SuffixAutomatonExtendNgramParams const& params, cudaStream_t stream);

//! \brief Get the size in bytes of a single SuffixAutomaton state
//! \return Size in bytes
size_t getSuffixAutomatonStateSize();

//! \brief Get the maximum number of slots supported
//! \return Maximum number of slots
size_t getSuffixAutomatonMaxSlots();

//! \brief Get the maximum sequence length supported
//! \return Maximum sequence length
size_t getSuffixAutomatonMaxSeqLen();

//! \brief Initialize a SuffixAutomaton using placement new
//! \param sa Pointer to allocated memory for the SuffixAutomaton
void initAutomaton(SuffixAutomaton* sa);

//! \brief Build a suffix automaton by extending with the given tokens
//! \param sa Pointer to an initialized SuffixAutomaton
//! \param tokens Array of token IDs
//! \param numTokens Number of tokens in the array
void buildAutomatonFromTokens(SuffixAutomaton* sa, int const* tokens, int numTokens);

} // namespace kernels::speculative_decoding::suffix_automaton

TRTLLM_NAMESPACE_END
