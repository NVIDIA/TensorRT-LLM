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

namespace kernels
{

//! The per-causal-block tensors of one block size for a causal K/V cache, computed on the device
//! from the cache's page table. The cache holds logical tokens ``[0, past)``: ``fixedTokens`` fixed
//! ones, then history; a forward stages ``numBlocks * blockSize`` tokens at ``past``. Block ``i``
//! starts at ``past + i * blockSize`` and sees the fixed region, the ``windowTokens`` before it, the
//! earlier blocks and itself. Its row lists every page it sees in full, then its private region:
//! the visible slots of the pages it sees only partly (at most three: where the fixed region ends,
//! where the window starts, where the block starts), then its own tokens.
//!
//! A view page is a layer-0 page index as the table stores it; a slot id is
//! ``viewPage * tokensPerPage + slot``; a pool row is ``viewPage * rowsPerPage + slot``.
struct CausalLayoutParams
{
    int32_t const* table;   //!< [numPages] view page of each logical page
    int32_t const* regions; //!< [numBlocks, regionPages] private view pages of each block
    int64_t numPages;
    int64_t regionPages;
    int64_t numBlocks;
    int64_t blockSize;
    int64_t tokensPerPage;
    int64_t past;
    int64_t fixedTokens;
    int64_t windowTokens;
    //! Pages dropped by the rotation this commit (0 if none). The page the fixed region's tail
    //! shares with the history then has a new view page; its old one, now at logical page
    //! ``numPages - dropPages``, still holds the fixed tail and is the source for it.
    int64_t dropPages;
    int64_t kvFactor; //!< block-offset encoding: ``viewPage * kvFactor`` is the K plane
    int64_t kvOffset; //!< and ``+ kvOffset`` the V plane
    int64_t rowsPerPage;

    int32_t* rows;         //!< [numBlocks, rowLen] view pages, 0-padded
    int64_t rowLen;
    int32_t* blockOffsets; //!< [numBlocks, 2, rowLen]
    int32_t* seqLenKv;     //!< [numBlocks] keys in the row: whole pages, partial slots, the block
    int64_t* ownSlots;     //!< [numBlocks * blockSize] slot ids of each block's own tokens
    //! Earlier blocks' staged tokens on a block's partial pages: ``2 * (tokensPerPage - 1)`` entries
    //! per block, (staged token index, slot id), padded with the block's own first token.
    int64_t* extraSrc;
    int64_t* extraDst;
    //! Fixed and history tokens copied into a block's private region: ``3 * (tokensPerPage - 1)``
    //! entries per block, (source pool row, destination pool row), -1 past the used ones.
    int64_t* pieceSrc;
    int64_t* pieceDst;

    //! Optional, computed once per commit by the launch that gets them: slot ids of the
    //! ``numStaged`` staged tokens at ``past``, and the fixed tail's refill, ``tokensPerPage``
    //! piece entries (-1 past the tail, all -1 when nothing was dropped).
    int64_t* stagedSlots;
    int64_t numStaged;
    int64_t* refillSrc;
    int64_t* refillDst;
};

void invokeCausalLayout(CausalLayoutParams const& params, cudaStream_t stream);

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

} // namespace kernels

TRTLLM_NAMESPACE_END
