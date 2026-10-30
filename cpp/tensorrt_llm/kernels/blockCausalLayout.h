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

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::blockCausalLayout
{

//! Compute per-causal-block attention layouts on the GPU from the cache's page
//! table, for one causal block size.
//!
//! Cached tokens occupy logical positions [0, past): first fixedTokens pinned
//! tokens, then history. The forward writes numCausalBlocks * causalBlockSize
//! new tokens starting at past.
//!
//! Causal block i starts at past + i * causalBlockSize. It sees the pinned
//! tokens, up to windowTokens preceding tokens (including earlier causal
//! blocks), and its own tokens. Its page list contains:
//!   1. Resident pages that are fully visible.
//!   2. Private pages containing visible fragments of partial pages, followed
//!      by the causal block's own tokens.
//!
//! There are at most three partial pages: where the pinned prefix ends,
//! where the window starts, and where the causal block starts.
//!
//! Indices use these units:
//!   viewPage: a layer-0 page index stored in the page table.
//!   slot id:  viewPage * tokensPerPage + slot.
//!   pool row: viewPage * rowsPerPage + slot.
struct BlockCausalLayoutParams
{
    int32_t const* table;        //!< [numPages] maps logical pages to viewPage indices.
    int32_t const* privatePages; //!< [numCausalBlocks, numPrivatePages] private viewPage indices.
    int64_t numPages;
    int64_t numPrivatePages;
    int64_t numCausalBlocks;
    int64_t causalBlockSize;
    int64_t tokensPerPage;
    int pageShift; //!< log2(tokensPerPage); tokensPerPage is a power of two
    int64_t past;
    int64_t fixedTokens;
    int64_t windowTokens;
    //! Number of pages rotated out of history by this commit; 0 if none. If the pinned prefix
    //! ends mid-page, that page also holds history. Rotation replaces it at the prefix/history
    //! boundary. The original page, now at `numPages - dropPages`, still holds the pinned tail
    //! and acts as the source of its tokens.
    int64_t dropPages;
    int64_t kvFactor; //!< offset encoding of the TRTLLM op: ``viewPage * kvFactor`` is the K plane
    int64_t kvOffset; //!< and ``+ kvOffset`` the V plane
    int64_t rowsPerPage;

    //! [numCausalBlocks, maxPages] each causal block's page list; unused entries are 0.
    int32_t* causalBlockPages;
    int64_t maxPages;
    //! [numCausalBlocks, 2, maxPages] the page lists in the TRTLLM op's offset encoding.
    int32_t* causalBlockOffsets;
    //! [numCausalBlocks] key token count: `full-page + partial-page + causal block's own`
    int32_t* seqLenKv;
    //! [numCausalBlocks * causalBlockSize] slot ids of each causal block's own tokens
    int64_t* ownSlots;
    //! current_private: copy indices for earlier causal blocks' tokens from this forward that belong
    //! in a causal block's private partial-page copies. Written every forward. Each causal block has
    //! ``2 * (tokensPerPage - 1)`` entries: currentPrivateSrc holds staged-token indices;
    //! currentPrivateDst holds slot ids. Unused entries repeat the causal block's first own token.
    int64_t* currentPrivateSrc;
    int64_t* currentPrivateDst;
    //! committed_private: committed tokens on a causal block's partial pages, copied into its private
    //! pages at commit. ``3 * (tokensPerPage - 1)`` entries per causal block, (source pool row,
    //! destination pool row), -1 past the used ones.
    int64_t* committedPrivateSrc;
    int64_t* committedPrivateDst;

    //! Optional, computed once per commit by the launch that gets them: slot ids of the
    //! ``numStaged`` staged tokens at ``past``, and the fixed tail's refill, ``tokensPerPage``
    //! entries (-1 past the tail, all -1 when nothing was dropped).
    int64_t* stagedSlots;
    int64_t numStaged;
    int64_t* refillSrc;
    int64_t* refillDst;
};

void invokeBlockCausalLayout(BlockCausalLayoutParams const& params, cudaStream_t stream);

//! Copy pool rows: entry ``e`` copies the token at pool row ``src[e]`` to pool row ``dst[e]`` in
//! every layer, K and V, every head: rows ``+ layer * rowsPerPage + (kv * numHeads + head) *
//! tokensPerPage``. Entries with a negative ``src`` are skipped. ``pool`` is the whole buffer as
//! rows of ``headDim`` elements.
struct CopyKvSlotsParams
{
    void* pool;
    int64_t const* src;
    int64_t const* dst;
    int64_t numEntries;
    int64_t numLayers;
    int64_t numHeads;
    int64_t headDim;
    int64_t tokensPerPage;
    int64_t rowsPerPage;
};

void invokeCopyKvSlots(CopyKvSlotsParams const& params, int elemSize, cudaStream_t stream);

} // namespace kernels::blockCausalLayout

TRTLLM_NAMESPACE_END
