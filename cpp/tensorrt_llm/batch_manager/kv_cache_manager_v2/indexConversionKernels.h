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

#include <cuda.h>

#include <cstdint>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

//! One converted index table, as the conversion sees it.
//!
//! A table holds `numBuffers` values per (row, block): one per buffer of the specs it
//! serves, differing only by the offset baked into them. Merged specs share these values
//! and are told apart by their base pointers, which the conversion never sees.
struct IndexTableDesc
{
    //! Raw slot ids for this table's life cycle: [numRows][rawRowStride].
    int32_t const* raw;
    //! Destination: [numRows][numBuffers][maxBlocksPerRow], so a row's buffers are adjacent.
    int32_t* converted;
    //! Offset added to each buffer's indices, in page-index units. Length numBuffers.
    int32_t const* offsets;
    int32_t numBuffers;
    //! Slot id to page index, i.e. how many buffers the pool coalesces.
    int32_t scale;

    //! Scratch metadata for this table's life cycle, null when it cannot hold scratch.
    //!
    //! A scratch block's slot does not come from the page table at all: it comes from the
    //! request's own scratch slots, indexed by how far into the scratch range the block
    //! sits.
    int32_t const* scratchBegs = nullptr;  //!< [numRows]
    int32_t const* scratchEnds = nullptr;  //!< [numRows]
    int32_t const* scratchSlots = nullptr; //!< [numRows][scratchSlotStride]
    int32_t scratchSlotStride = 0;
    //! Sub-pages a scratch block occupies, which is what advances the slot cursor.
    int32_t scratchPagesPerBlock = 1;
};

//! Reset one row's entries to kBadPageIndex in every table.
//!
//! The conversion stops at a row's capacity, so what an earlier occupant wrote above it
//! would otherwise outlive it. Clearing on the way out is what keeps the entries past a
//! capacity reading as absent.
void launchClearIndexRow(IndexTableDesc const* descs, int32_t numTables, int32_t maxBuffers, int32_t row,
    int32_t maxBlocksPerRow, int32_t smCount, CUstream stream);

//! Convert raw slot ids into kernel page indices for every table at once.
//!
//! One launch covers every table: under PER_LAYER there is a table per layer, and per-launch
//! driver cost would dominate the few arithmetic operations per entry.
//!
//! `entry = slot * scale + offset`, with kBadPageIndex passed through unchanged -- a sparse
//! table numbers pure-history and input-token entries in different levels' pools, so only
//! the sentinel separates an absent block from slot zero.
//!
//! A row is converted only as far as its capacity reaches; the entries past it read as
//! absent because a vacated row is cleared.
//!
//! @param descs        device array of numTables descriptors
//! @param maxBuffers    widest table's numBuffers; narrower tables return immediately
//! @param maxBlocksPerRow  entries a row reserves, and the row stride of `converted`. No
//!                      capacity may exceed what it holds; the caller is required to have
//!                      checked that, since the kernels index on it unguarded.
//! @param rawRowStride  elements between consecutive rows of a raw table
//! @param capacity      per-row token watermark, numRows entries, on the launch stream
//! @param tokensPerBlock  must be a power of two; the launcher rejects anything else so the
//!                      kernel divides by shifting
//! @param smCount       multiprocessors to fill; a row's blocks are split across CTAs only
//!                      until the grid covers them, since a prefill batch has few rows and
//!                      few buffers and would otherwise leave the machine idle
void launchIndexConversion(IndexTableDesc const* descs, int32_t numTables, int32_t maxBuffers, int32_t numRows,
    int32_t maxBlocksPerRow, int32_t rawRowStride, int32_t const* capacity, int32_t tokensPerBlock, int32_t smCount,
    CUstream stream);

//! The same, for tables whose life cycle substitutes scratch slots.
//!
//! A scratch block takes its slot from the request instead of the page table, and wraps the
//! layer's offset within the slot rather than adding it.
//!
//! Scratch is a PER_LAYER-only concern. Under SHARED every layer of a life cycle reads one
//! table, which substitution would break.
void launchIndexConversionWithScratch(IndexTableDesc const* descs, int32_t numTables, int32_t maxBuffers,
    int32_t numRows, int32_t maxBlocksPerRow, int32_t rawRowStride, int32_t const* capacity, int32_t tokensPerBlock,
    int32_t smCount, CUstream stream);

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
