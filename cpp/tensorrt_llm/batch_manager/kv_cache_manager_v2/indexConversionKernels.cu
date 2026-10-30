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

#include "indexConversionKernels.h"

#include "common.h"
#include "kv_cache_manager_v2/utils/math.h"

#include "tensorrt_llm/common/cudaUtils.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cuda_runtime_api.h>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

namespace
{

constexpr int kThreads = 256;

//! Blocks each thread converts, as a shift, chosen so the grid lands near one wave.
//!
//! Only the block extent scales with the sequence; the others are the batch's rows and its
//! buffers, and a prefill step leaves both tiny -- one row of a million tokens is two CTAs
//! if a CTA takes a whole row. Splitting a row across CTAs fixes that, and giving each
//! thread more blocks folds the split back once the rows and buffers already fill the
//! machine, so a decode batch keeps the single-chunk shape it has now.
int32_t lg2BlocksPerThread(int32_t numTables, int32_t maxBuffers, int32_t numRows, int32_t maxBlocksPerRow,
    int32_t smCount, int32_t ctasPerSm) noexcept
{
    int64_t const perChunk = int64_t{numTables} * maxBuffers * numRows;
    int64_t const wave = int64_t{std::max(smCount, 1)} * std::max(ctasPerSm, 1);
    int64_t const chunks = std::max<int64_t>(1, wave / std::max<int64_t>(perChunk, 1));
    int64_t const blocks = divUp(int64_t{maxBlocksPerRow}, chunks);
    // Smallest shift with kThreads << lg2 >= blocks, which is ceil(log2()) of the blocks a
    // thread would take.
    int64_t const perThread = divUp(blocks, int64_t{kThreads});
    return perThread <= 1 ? 0 : std::bit_width(static_cast<uint64_t>(perThread - 1));
}

//! Resident CTAs per SM for `kernel`, as the compiler built it.
//!
//! Asked rather than assumed: the answer folds in the register pressure a constant cannot
//! know, and it is what says when splitting a row stops buying occupancy. Each launcher
//! caches its own, since the two kernels need not agree.
int32_t queryCtasPerSm(void const* kernel) noexcept
{
    int blocks = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, kThreads, 0) != cudaSuccess)
    {
        return 1;
    }
    return std::max(blocks, 1);
}

//! CTAs the block extent needs once each thread takes `1 << lg2` of them.
int32_t chunkCount(int32_t maxBlocksPerRow, int32_t lg2) noexcept
{
    return divUp(maxBlocksPerRow, kThreads << lg2);
}

//! Shift behind a power-of-two tokens-per-block, rejecting anything the kernel cannot divide by.
int32_t lg2TokensPerBlock(int32_t tokensPerBlock)
{
    TLLM_CHECK_WITH_INFO(tokensPerBlock > 0 && std::popcount(static_cast<uint32_t>(tokensPerBlock)) == 1,
        "index conversion: tokensPerBlock must be a power of two, got %d", tokensPerBlock);
    return std::countr_zero(static_cast<uint32_t>(tokensPerBlock));
}

//! Blocks a row's tokens occupy, which is as far as a consumer may read it.
__device__ inline int32_t rowBlockCount(int32_t capacity, int32_t lg2Tpb)
{
    return static_cast<int32_t>((static_cast<uint32_t>(capacity) + (1U << lg2Tpb) - 1U) >> lg2Tpb);
}

//! One CTA per (table, row, buffer); threads walk the blocks of that row.
//!
//! Buffer is the x extent because CTAs are rasterised there first: the buffers of one row
//! read the same raw row, and write spans that abut, where rows-first would stream the whole
//! raw table between reuses and stride the writes by a buffer's worth of blocks.
__global__ __launch_bounds__(kThreads) void indexConversionKernel(IndexTableDesc const* descs, int32_t numRows,
    int32_t maxBuffers, int32_t maxBlocksPerRow, int32_t rawRowStride, int32_t const* capacity, int32_t lg2Tpb,
    int32_t lg2Blocks)
{
    assert(gridDim.z == static_cast<unsigned>(numRows));
    int32_t const table = static_cast<int32_t>(blockIdx.x) / maxBuffers;
    int32_t const buffer = static_cast<int32_t>(blockIdx.x) % maxBuffers;
    auto const& desc = descs[table];
    if (buffer >= desc.numBuffers)
    {
        return;
    }
    int32_t const row = static_cast<int32_t>(blockIdx.z);

    int32_t const* raw = desc.raw + static_cast<ptrdiff_t>(row) * rawRowStride;
    // Row-major in the entries: a row's entries are adjacent, so a request's K and V
    // sit together the way an attention kernel's page list wants them.
    int32_t* out = desc.converted + (static_cast<ptrdiff_t>(row) * desc.numBuffers + buffer) * maxBlocksPerRow;
    int32_t const offset = desc.offsets[buffer];

    int32_t const rowBlocks = rowBlockCount(capacity[row], lg2Tpb);
    assert(rowBlocks <= maxBlocksPerRow);
    int32_t const begin = static_cast<int32_t>(blockIdx.y) * (kThreads << lg2Blocks);
    int32_t const end = min(begin + (kThreads << lg2Blocks), rowBlocks);
    for (int32_t block = begin + static_cast<int32_t>(threadIdx.x); block < end; block += kThreads)
    {
        int32_t const slot = raw[block];
        // An absent block stays absent: folding it to a safe slot would read page zero.
        out[block] = slot == kBadPageIndex.value() ? kBadPageIndex.value() : slot * desc.scale + offset;
    }
}

//! One CTA per (table, row, buffer), laid out as above, but scratch blocks take their slot
//! from the request rather than the page table.
__global__ __launch_bounds__(kThreads) void indexConversionScratchKernel(IndexTableDesc const* descs, int32_t numRows,
    int32_t maxBuffers, int32_t maxBlocksPerRow, int32_t rawRowStride, int32_t const* capacity, int32_t lg2Tpb,
    int32_t lg2Blocks)
{
    assert(gridDim.z == static_cast<unsigned>(numRows));
    int32_t const table = static_cast<int32_t>(blockIdx.x) / maxBuffers;
    int32_t const buffer = static_cast<int32_t>(blockIdx.x) % maxBuffers;
    auto const& desc = descs[table];
    if (buffer >= desc.numBuffers)
    {
        return;
    }
    int32_t const row = static_cast<int32_t>(blockIdx.z);

    int32_t const* raw = desc.raw + static_cast<ptrdiff_t>(row) * rawRowStride;
    int32_t* out = desc.converted + (static_cast<ptrdiff_t>(row) * desc.numBuffers + buffer) * maxBlocksPerRow;
    int32_t const offset = desc.offsets[buffer];

    int32_t const scratchBeg = desc.scratchBegs[row];
    int32_t const scratchEnd = desc.scratchEnds[row];
    int32_t const* scratchSlots = desc.scratchSlots + static_cast<ptrdiff_t>(row) * desc.scratchSlotStride;

    int32_t const rowBlocks = rowBlockCount(capacity[row], lg2Tpb);
    assert(rowBlocks <= maxBlocksPerRow);
    int32_t const begin = static_cast<int32_t>(blockIdx.y) * (kThreads << lg2Blocks);
    int32_t const end = min(begin + (kThreads << lg2Blocks), rowBlocks);
    for (int32_t block = begin + static_cast<int32_t>(threadIdx.x); block < end; block += kThreads)
    {
        if (block >= scratchBeg && block < scratchEnd)
        {
            // The slot cursor advances by the sub-pages each scratch block occupies. The
            // layer's own offset wraps inside the slot: a scratch slot is shared, so the
            // sub-page positions past it are already spoken for.
            int32_t const total = (block - scratchBeg) * desc.scratchPagesPerBlock;
            int32_t const slotId = scratchSlots[total / desc.scale];
            out[block] = slotId == kBadPageIndex.value()
                ? kBadPageIndex.value()
                : slotId * desc.scale + (total % desc.scale + offset) % desc.scale;
            continue;
        }
        int32_t const slot = raw[block];
        out[block] = slot == kBadPageIndex.value() ? kBadPageIndex.value() : slot * desc.scale + offset;
    }
}

//! Reset one row across every table, laid out like the conversion above.
__global__ __launch_bounds__(kThreads) void clearIndexRowKernel(
    IndexTableDesc const* descs, int32_t maxBuffers, int32_t row, int32_t maxBlocksPerRow, int32_t lg2Blocks)
{
    int32_t const table = static_cast<int32_t>(blockIdx.x) / maxBuffers;
    int32_t const buffer = static_cast<int32_t>(blockIdx.x) % maxBuffers;
    auto const& desc = descs[table];
    if (buffer >= desc.numBuffers)
    {
        return;
    }

    int32_t* out = desc.converted + (static_cast<ptrdiff_t>(row) * desc.numBuffers + buffer) * maxBlocksPerRow;
    int32_t const begin = static_cast<int32_t>(blockIdx.y) * (kThreads << lg2Blocks);
    int32_t const end = min(begin + (kThreads << lg2Blocks), maxBlocksPerRow);
    for (int32_t block = begin + static_cast<int32_t>(threadIdx.x); block < end; block += kThreads)
    {
        out[block] = kBadPageIndex.value();
    }
}

} // namespace

void launchClearIndexRow(IndexTableDesc const* descs, int32_t numTables, int32_t maxBuffers, int32_t row,
    int32_t maxBlocksPerRow, int32_t smCount, CUstream stream)
{
    if (numTables <= 0 || maxBuffers <= 0 || maxBlocksPerRow <= 0)
    {
        return;
    }
    // One launch for every table: under PER_LAYER a memset apiece would cost more in driver
    // calls than the row is worth in bytes.
    static int32_t const ctasPerSm = queryCtasPerSm(reinterpret_cast<void const*>(clearIndexRowKernel));
    int32_t const lg2Blocks = lg2BlocksPerThread(numTables, maxBuffers, 1, maxBlocksPerRow, smCount, ctasPerSm);
    dim3 const grid(static_cast<unsigned>(numTables * maxBuffers),
        static_cast<unsigned>(chunkCount(maxBlocksPerRow, lg2Blocks)), 1U);
    clearIndexRowKernel<<<grid, kThreads, 0, stream>>>(descs, maxBuffers, row, maxBlocksPerRow, lg2Blocks);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

void launchIndexConversion(IndexTableDesc const* descs, int32_t numTables, int32_t maxBuffers, int32_t numRows,
    int32_t maxBlocksPerRow, int32_t rawRowStride, int32_t const* capacity, int32_t tokensPerBlock, int32_t smCount,
    CUstream stream)
{
    if (numTables <= 0 || maxBuffers <= 0 || numRows <= 0 || maxBlocksPerRow <= 0)
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(capacity != nullptr, "launchIndexConversion: capacity watermarks are required");
    int32_t const lg2Tpb = lg2TokensPerBlock(tokensPerBlock);
    // Table and buffer share the x extent so that a table's buffers are adjacent CTAs: they
    // read the same raw row, and the reuse is then two CTAs apart whatever the sequence
    // length. The buffer extent is the widest table's, since descs lives on the device and
    // the host cannot read it back to size the grid per table; a narrower table's surplus
    // CTAs return at once. Rows are slowest, sharing only the descriptor.
    static int32_t const ctasPerSm = queryCtasPerSm(reinterpret_cast<void const*>(indexConversionKernel));
    int32_t const lg2Blocks = lg2BlocksPerThread(numTables, maxBuffers, numRows, maxBlocksPerRow, smCount, ctasPerSm);
    dim3 const grid(static_cast<unsigned>(numTables * maxBuffers),
        static_cast<unsigned>(chunkCount(maxBlocksPerRow, lg2Blocks)), static_cast<unsigned>(numRows));
    indexConversionKernel<<<grid, kThreads, 0, stream>>>(
        descs, numRows, maxBuffers, maxBlocksPerRow, rawRowStride, capacity, lg2Tpb, lg2Blocks);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

void launchIndexConversionWithScratch(IndexTableDesc const* descs, int32_t numTables, int32_t maxBuffers,
    int32_t numRows, int32_t maxBlocksPerRow, int32_t rawRowStride, int32_t const* capacity, int32_t tokensPerBlock,
    int32_t smCount, CUstream stream)
{
    if (numTables <= 0 || maxBuffers <= 0 || numRows <= 0 || maxBlocksPerRow <= 0)
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(capacity != nullptr, "launchIndexConversionWithScratch: capacity watermarks are required");
    int32_t const lg2Tpb = lg2TokensPerBlock(tokensPerBlock);
    static int32_t const ctasPerSm = queryCtasPerSm(reinterpret_cast<void const*>(indexConversionScratchKernel));
    int32_t const lg2Blocks = lg2BlocksPerThread(numTables, maxBuffers, numRows, maxBlocksPerRow, smCount, ctasPerSm);
    dim3 const grid(static_cast<unsigned>(numTables * maxBuffers),
        static_cast<unsigned>(chunkCount(maxBlocksPerRow, lg2Blocks)), static_cast<unsigned>(numRows));
    indexConversionScratchKernel<<<grid, kThreads, 0, stream>>>(
        descs, numRows, maxBuffers, maxBlocksPerRow, rawRowStride, capacity, lg2Tpb, lg2Blocks);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
