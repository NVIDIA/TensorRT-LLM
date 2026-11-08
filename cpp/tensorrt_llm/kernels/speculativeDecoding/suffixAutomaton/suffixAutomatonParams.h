/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Lightweight header containing only param structs and function declarations.
// Safe to include from non-CUDA translation units (e.g. nanobind bindings)
// because it does NOT include suffixAutomaton.h, which redefines cudaStream_t
// to int via saCudaCallable.h when __CUDACC__ is not defined.

#pragma once

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::speculative_decoding::suffix_automaton
{

// Forward declaration — full definition lives in suffixAutomaton.h (CUDA-only).
struct SuffixAutomaton;

// Max suffix tokens loaded into shared memory per block in the global search kernel.
// Caps the suffix length used for cross-SA matching to bound shared-memory usage.
static constexpr int kMaxGlobalSuffixLen = 64;

// =====================================================================
// Param structs
// =====================================================================

//! \brief Parameters for the suffix automaton extend kernel
struct SuffixAutomatonExtendParams
{
    //! Number of sequences in the batch
    int batchSize{0};

    //! Number of draft tokens to generate per sequence
    int draftLength{0};

    //! Maximum number of slots in the workspace
    int maxSlots{0};

    //! Maximum sequence length (runtime configurable)
    int maxSeqLen{0};

    //! Pointer to the suffix automaton workspace on GPU (raw bytes)
    void* slots{nullptr};

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
        TLLM_CHECK(maxSlots > 0);
        TLLM_CHECK(maxSeqLen > 0);
        TLLM_CHECK(slots != nullptr);
        TLLM_CHECK(batchIndices != nullptr);
        TLLM_CHECK(matchLenOut != nullptr);
        TLLM_CHECK(draftTokensOut != nullptr);
        TLLM_CHECK(acceptedTokensIn != nullptr);
        TLLM_CHECK(acceptedLensIn != nullptr);
    }
};

//! \brief Parameters for the suffix automaton extend kernel with ngram support
struct SuffixAutomatonExtendNgramParams
{
    //! Number of sequences in the batch
    int batchSize{0};

    //! Number of draft tokens to generate per sequence
    int draftLength{0};

    //! Maximum ngram size for matching, or -1 for longest match mode
    int maxNgramSize{-1};

    //! Maximum number of slots in the workspace
    int maxSlots{0};

    //! Maximum sequence length (runtime configurable)
    int maxSeqLen{0};

    //! Pointer to the suffix automaton workspace on GPU (raw bytes)
    void* slots{nullptr};

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
        TLLM_CHECK(maxSlots > 0);
        TLLM_CHECK(maxSeqLen > 0);
        TLLM_CHECK(slots != nullptr);
        TLLM_CHECK(batchIndices != nullptr);
        TLLM_CHECK(matchLenOut != nullptr);
        TLLM_CHECK(draftTokensOut != nullptr);
        TLLM_CHECK(acceptedTokensIn != nullptr);
        TLLM_CHECK(acceptedLensIn != nullptr);
    }
};

//! \brief Parameters for the global search kernel (cross-request pattern sharing)
//!
//! Limitation: maxSlots must be <= 1024 because the search kernel maps one
//! CUDA thread per slot within a single block. The suffix matched per request
//! is capped at 64 tokens (kMaxGlobalSuffixLen) due to shared-memory storage.
struct SuffixAutomatonGlobalSearchParams
{
    //! Number of sequences in the batch
    int batchSize{0};

    //! Number of draft tokens to generate per sequence
    int draftLength{0};

    //! Maximum ngram size for matching, or -1 for longest match mode
    int maxNgramSize{-1};

    //! Maximum number of slots in the workspace (must be <= 1024)
    int maxSlots{0};

    //! Maximum sequence length (runtime configurable)
    int maxSeqLen{0};

    //! Pointer to the suffix automaton workspace on GPU (raw bytes)
    void* slots{nullptr};

    //! Batch indices mapping external batch idx to workspace slot [batchSize]
    int const* batchIndices{nullptr};

    //! Active slot mask: 1=active, 0=inactive [maxSlots]
    int const* activeSlotMask{nullptr};

    //! Output: match lengths for each sequence [batchSize]
    int* matchLenOut{nullptr};

    //! Output: source slot index for the best match [batchSize]
    int* matchSlotOut{nullptr};

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
        TLLM_CHECK(maxSlots > 0);
        TLLM_CHECK_WITH_INFO(
            maxSlots <= 1024, "Global search kernel supports at most 1024 slots (one CUDA thread per slot)");
        TLLM_CHECK_WITH_INFO(maxNgramSize == -1 || (maxNgramSize >= 1 && maxNgramSize <= kMaxGlobalSuffixLen),
            "maxNgramSize must be -1 (longest match) or in [1, %d], got %d", kMaxGlobalSuffixLen, maxNgramSize);
        TLLM_CHECK(maxSeqLen > 0);
        TLLM_CHECK(slots != nullptr);
        TLLM_CHECK(batchIndices != nullptr);
        TLLM_CHECK(activeSlotMask != nullptr);
        TLLM_CHECK(matchLenOut != nullptr);
        TLLM_CHECK(matchSlotOut != nullptr);
        TLLM_CHECK(draftTokensOut != nullptr);
        TLLM_CHECK(acceptedTokensIn != nullptr);
        TLLM_CHECK(acceptedLensIn != nullptr);
    }
};

// =====================================================================
// Function declarations
// =====================================================================

//! \brief Invokes the suffix automaton extend kernel
//!
//! This kernel updates the suffix automaton states for each sequence in the batch
//! with the newly accepted tokens, then performs a lookup to find the longest
//! suffix match and generates draft tokens based on that match.
//!
//! \param params The parameters for the kernel
//! \param stream The CUDA stream to run the kernel on
void invokeSuffixAutomatonExtend(SuffixAutomatonExtendParams const& params, cudaStream_t stream);

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

//! \brief Invokes the global search kernel for cross-request pattern sharing
//!
//! This launches two kernels on the same stream:
//! 1. Extend kernel: updates each request's SA with accepted tokens
//! 2. Search kernel: each request searches all active SA states in parallel
//!    and reduces to the best match (longest match -> own slot -> longest continuation)
//!
//! The kernel launch boundary acts as a device-wide barrier, ensuring all extends
//! complete before any search begins. CUDA graph compatible.
//!
//! \param params The parameters for the kernel
//! \param stream The CUDA stream to run the kernel on
void invokeSuffixAutomatonGlobalSearch(SuffixAutomatonGlobalSearchParams const& params, cudaStream_t stream);

//! \brief Get the size in bytes of a single SuffixAutomaton state for a given max sequence length
//! \param maxSeqLen Maximum sequence length
//! \return Size in bytes
size_t getSuffixAutomatonStateSize(size_t maxSeqLen);

//! \brief Initialize a SuffixAutomaton at the given memory location
//! \param memory Pointer to allocated memory for the SuffixAutomaton
//! \param maxSeqLen Maximum sequence length
void initAutomaton(void* memory, size_t maxSeqLen);

//! \brief Build a suffix automaton by extending with the given tokens
//! \param sa Pointer to an initialized SuffixAutomaton
//! \param tokens Array of token IDs
//! \param numTokens Number of tokens in the array
void buildAutomatonFromTokens(SuffixAutomaton* sa, int const* tokens, int numTokens);

//! \brief Relocate a SuffixAutomaton's internal pointers for GPU copy
//!
//! WARNING: This function MUTATES the SuffixAutomaton in-place, making it SINGLE-USE.
//! It rebases internal pointers from oldBase to newBase directly in the host buffer.
//! After this call, the host buffer's internal pointer graph is relative to newBase
//! (the GPU destination), so the host buffer is effectively corrupted for any subsequent
//! use (e.g., copying to a different GPU slot). If a caller invokes this twice with
//! the same host buffer but a different newBase, the second operation will produce
//! incorrectly relocated pointers.
//!
//! To copy the same automaton state to multiple GPU slots, callers must rebuild the
//! automaton from scratch via initAutomaton() + buildAutomatonFromTokens() for each
//! destination.
//!
//! \param sa Pointer to the SuffixAutomaton (mutated in-place)
//! \param oldBase The current base address (host address)
//! \param newBase The target base address (GPU address)
void relocateAutomaton(SuffixAutomaton* sa, void* oldBase, void* newBase);

} // namespace kernels::speculative_decoding::suffix_automaton

TRTLLM_NAMESPACE_END
