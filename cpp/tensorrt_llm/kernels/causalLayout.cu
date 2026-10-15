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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/causalLayout.h"

#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_scan.cuh>

using namespace tensorrt_llm::common;

TRTLLM_NAMESPACE_BEGIN

namespace kernels::causalLayout
{
namespace
{

namespace constants
{
constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kMaxRanges = 2;                                // visible ranges: the fixed tokens and the window
constexpr int kMaxBoundaryPages = 2 * kMaxRanges;            // each range's first and last page
constexpr int kMaxSegments = kMaxBoundaryPages * kMaxRanges; // one per (boundary page, range), a lane each
static_assert(kMaxSegments <= kWarpSize, "one warp plans a block");
} // namespace constants

// What one causal block sees before its own tokens, in logical token positions.
//
// The block sees one or two visible ranges of positions: the fixed tokens [0, fixedEnd), then the
// window [windowBegin, blockBegin); a single range [0, blockBegin) when the window reaches the
// fixed tokens. A page that lies fully inside a range goes into the block's row as it is. A page a
// range covers only partly cannot: a page table cannot start or stop a key sequence mid-page. Those
// positions form segments (one per partial page and range), which are copied, in ascending position
// order, into the block's private pages, slot by slot; the block's own tokens follow them there.
//
// Example, 32-token pages, 40 fixed tokens, a 100-token window, a block starting at 200: the ranges
// are [0, 40) and [100, 200). Full pages: 0, then 4 and 5. Segments, in order: positions 32-39 on
// page 1 (the fixed tail), 100-127 on page 3 (where the window starts) and 192-199 on page 6 (just
// before the block). They fill private slots 0-7, 8-35 and 36-43, and the block's own tokens start
// at private slot 44.
struct BlockPlan
{
    // Logical pages [fullPageBegin[r], fullPageEnd[r]) lie fully inside visible range r; the row
    // lists range 0's, then range 1's.
    int64_t fullPageBegin[constants::kMaxRanges];
    int64_t fullPageEnd[constants::kMaxRanges];
    // Segment s holds positions [segmentBegin[s], segmentEnd[s]), ascending across segments, and
    // starts at private slot segmentPrivateSlot[s].
    int64_t segmentBegin[constants::kMaxSegments];
    int64_t segmentEnd[constants::kMaxSegments];
    int64_t segmentPrivateSlot[constants::kMaxSegments];
    // Row entries before the private pages.
    int64_t numFullPages;
    // Private slots before the block's own tokens: the segments' total length.
    int64_t numPartialSlots;
    // Of those, the leading ones that hold committed tokens (position < past). They are copied
    // into the private pages at commit; the rest hold earlier blocks' tokens of this forward, which
    // every forward writes into the private pages.
    int64_t numCommittedSlots;
    int numRanges; // 0 for a block at position 0, else 1 or 2
    int numSegments;
};

__device__ int64_t pageOf(CausalLayoutParams const& p, int64_t pos)
{
    return pos >> p.pageShift;
}

__device__ int64_t slotOf(CausalLayoutParams const& p, int64_t pos)
{
    return pos & (p.tokensPerPage - 1);
}

// Appends the visible range [begin, end) unless it is empty.
__device__ void addRange(int64_t (&rangeBegin)[constants::kMaxRanges], int64_t (&rangeEnd)[constants::kMaxRanges],
    int& numRanges, int64_t begin, int64_t end)
{
    if (end > begin)
    {
        rangeBegin[numRanges] = begin;
        rangeEnd[numRanges++] = end;
    }
}

// Boundary page b: the first page of range b / 2 for even b, its last page for odd b. The ranges
// are disjoint and ordered, so boundary pages ascend with b.
__device__ int64_t boundaryPage(CausalLayoutParams const& p, int64_t const (&rangeBegin)[constants::kMaxRanges],
    int64_t const (&rangeEnd)[constants::kMaxRanges], int b)
{
    return b % 2 == 0 ? pageOf(p, rangeBegin[b / 2]) : pageOf(p, rangeEnd[b / 2] - 1);
}

// Warp 0 plans block i. A partial page is a boundary page that is not a full page; each lane takes
// one (boundary page, range) pair and measures that range's segment on that page. Lanes are in
// position order, so an exclusive sum of the segment lengths gives each segment its private slot,
// and an exclusive sum over the lanes that have a segment gives each segment its index.
__device__ void planBlock(CausalLayoutParams const& p, int64_t i, BlockPlan& plan)
{
    __shared__ typename cub::WarpScan<int64_t>::TempStorage privateSlotScan;
    __shared__ typename cub::WarpScan<int>::TempStorage segmentIndexScan;
    __shared__ typename cub::WarpReduce<int64_t>::TempStorage committedSum;
    int const lane = threadIdx.x % constants::kWarpSize;
    int64_t const tpb = p.tokensPerPage;
    int64_t const blockBegin = p.past + i * p.blockSize;
    int64_t const fixedEnd = p.fixedTokens;
    int64_t const windowBegin = max(fixedEnd, blockBegin - p.windowTokens);

    // a contiguous range of token position [begin, end)
    // holds tokens the block is allowed to attend to
    int64_t rangeBegin[constants::kMaxRanges];
    int64_t rangeEnd[constants::kMaxRanges];
    int numRanges = 0;
    if (windowBegin <= fixedEnd)
    {
        addRange(rangeBegin, rangeEnd, numRanges, 0, blockBegin);
    }
    else // history exceeds the window
    {
        addRange(rangeBegin, rangeEnd, numRanges, 0, fixedEnd);
        addRange(rangeBegin, rangeEnd, numRanges, windowBegin, blockBegin);
    }

    int64_t fullPageBegin[constants::kMaxRanges]; // first full page of the range
    int64_t fullPageEnd[constants::kMaxRanges];   // one past the last full page
    for (int r = 0; r < numRanges; ++r)
    {
        fullPageBegin[r] = pageOf(p, rangeBegin[r] + tpb - 1);
        fullPageEnd[r] = max(fullPageBegin[r], pageOf(p, rangeEnd[r]));
    }

    // figure out attended ranges within non-full pages
    int const boundary = lane / constants::kMaxRanges;
    int const range = lane % constants::kMaxRanges;
    int64_t segmentBegin = 0;
    int64_t segmentLength = 0;
    int64_t committedLength = 0;
    if (boundary < 2 * numRanges && range < numRanges)
    {
        int64_t const page = boundaryPage(p, rangeBegin, rangeEnd, boundary);
        bool const sameAsPrevious = boundary > 0 && boundaryPage(p, rangeBegin, rangeEnd, boundary - 1) == page;
        bool isFullPage = false;
        for (int r = 0; r < numRanges; ++r)
        {
            isFullPage |= fullPageBegin[r] <= page && page < fullPageEnd[r];
        }
        if (!sameAsPrevious && !isFullPage)
        {
            segmentBegin = max(rangeBegin[range], page * tpb);
            int64_t const segmentEnd = min(rangeEnd[range], (page + 1) * tpb);
            segmentLength = max(int64_t{0}, segmentEnd - segmentBegin);
            committedLength = max(int64_t{0}, min(segmentEnd, p.past) - segmentBegin);
        }
    }

    // where does the segment start in private pages (destination)
    int64_t privateSlot = 0;
    int64_t numPartialSlots = 0;
    cub::WarpScan<int64_t>(privateSlotScan).ExclusiveSum(segmentLength, privateSlot, numPartialSlots);
    // which entry of the plan's segment list the lane writes
    int segmentIndex = 0;
    int numSegments = 0;
    cub::WarpScan<int>(segmentIndexScan).ExclusiveSum(segmentLength > 0 ? 1 : 0, segmentIndex, numSegments);

    // publish result
    int64_t const numCommittedSlots = cub::WarpReduce<int64_t>(committedSum).Sum(committedLength); // lane 0 only
    if (segmentLength > 0)
    {
        plan.segmentBegin[segmentIndex] = segmentBegin;
        plan.segmentEnd[segmentIndex] = segmentBegin + segmentLength;
        plan.segmentPrivateSlot[segmentIndex] = privateSlot;
    }
    if (lane == 0)
    {
        plan.numRanges = numRanges;
        plan.numFullPages = 0;
        for (int r = 0; r < numRanges; ++r)
        {
            plan.fullPageBegin[r] = fullPageBegin[r];
            plan.fullPageEnd[r] = fullPageEnd[r];
            plan.numFullPages += fullPageEnd[r] - fullPageBegin[r];
        }
        plan.numSegments = numSegments;
        plan.numPartialSlots = numPartialSlots;
        plan.numCommittedSlots = numCommittedSlots;
    }
}

// The logical position copied into private slot privateSlot (< numPartialSlots).
__device__ int64_t positionAtPrivateSlot(BlockPlan const& plan, int64_t privateSlot)
{
    int s = 0;
    while (s + 1 < plan.numSegments && privateSlot >= plan.segmentPrivateSlot[s + 1])
    {
        ++s;
    }
    return plan.segmentBegin[s] + (privateSlot - plan.segmentPrivateSlot[s]);
}

// The view page holding logical position pos. After a rotation the fixed tail on the page it
// shares with the history is still read from that page's old view, now at the ring's tail.
__device__ int32_t viewOf(CausalLayoutParams const& p, int64_t pos)
{
    int64_t const page = pageOf(p, pos);
    if (p.dropPages > 0 && pos < p.fixedTokens && page == pageOf(p, p.fixedTokens))
    {
        return p.table[p.numPages - p.dropPages];
    }
    return p.table[page];
}

// The home slot id (view page * tokensPerPage + slot) of each token the next forward writes, at
// positions past, past + 1, ...; the chunk write puts each token's K/V there.
__device__ void writeStagedHomeSlots(CausalLayoutParams const& p)
{
    for (int64_t t = threadIdx.x; t < p.numStaged; t += blockDim.x)
    {
        int64_t const pos = p.past + t;
        p.stagedSlots[t] = static_cast<int64_t>(p.table[pageOf(p, pos)]) * p.tokensPerPage + slotOf(p, pos);
    }
}

// Copies (layer-0 pool rows) that move the fixed tokens' tail after a rotation. When the fixed
// tokens end mid-page, their last page is shared with history; a rotation moves that page to the
// ring's tail, so its fixed slots are copied to the page that now follows the full fixed pages.
// Unused entries are -1: no rotation this commit, or the fixed tokens end on a page boundary.
__device__ void writeFixedTailRefill(CausalLayoutParams const& p)
{
    int64_t const tailLength = p.dropPages > 0 ? slotOf(p, p.fixedTokens) : 0;
    int64_t const tailPage = pageOf(p, p.fixedTokens);
    for (int64_t slot = threadIdx.x; slot < p.tokensPerPage; slot += blockDim.x)
    {
        if (slot < tailLength)
        {
            p.refillSrc[slot] = static_cast<int64_t>(p.table[p.numPages - p.dropPages]) * p.rowsPerPage + slot;
            p.refillDst[slot] = static_cast<int64_t>(p.table[tailPage]) * p.rowsPerPage + slot;
        }
        else
        {
            p.refillSrc[slot] = -1;
            p.refillDst[slot] = -1;
        }
    }
}

__global__ void causalLayoutKernel(CausalLayoutParams p)
{
    __shared__ BlockPlan plan;
    int64_t const i = blockIdx.x;
    if (i >= p.numBlocks)
    {
        // The commit block: work done once per commit, not per causal block.
        if (p.stagedSlots != nullptr)
        {
            writeStagedHomeSlots(p);
        }
        if (p.refillSrc != nullptr)
        {
            writeFixedTailRefill(p);
        }
        return;
    }
    if (threadIdx.x < constants::kWarpSize)
    {
        planBlock(p, i, plan);
    }
    __syncthreads();

    int64_t const tpb = p.tokensPerPage;
    int32_t const* privatePages = p.privatePages + i * p.numPrivatePages;

    // The row: full pages in logical order, then the private pages, then zeros.
    int32_t* row = p.rows + i * p.rowLen;
    int32_t* offK = p.blockOffsets + i * 2 * p.rowLen;
    int32_t* offV = offK + p.rowLen;
    int64_t const fullPagesOfRange0 = plan.numRanges > 0 ? plan.fullPageEnd[0] - plan.fullPageBegin[0] : 0;
    for (int64_t j = threadIdx.x; j < p.rowLen; j += blockDim.x)
    {
        int32_t view = 0;
        if (j < plan.numFullPages)
        {
            view = p.table[j < fullPagesOfRange0 ? plan.fullPageBegin[0] + j
                                                 : plan.fullPageBegin[1] + (j - fullPagesOfRange0)];
        }
        else if (j < plan.numFullPages + p.numPrivatePages)
        {
            view = privatePages[j - plan.numFullPages];
        }
        row[j] = view;
        offK[j] = static_cast<int32_t>(view * p.kvFactor);
        offV[j] = static_cast<int32_t>(view * p.kvFactor + p.kvOffset);
    }
    if (threadIdx.x == 0)
    {
        p.seqLenKv[i] = static_cast<int32_t>(plan.numFullPages * tpb + plan.numPartialSlots + p.blockSize);
    }

    // The block's own tokens follow the partial slots in its private pages.
    int64_t* own = p.ownSlots + i * p.blockSize;
    for (int64_t t = threadIdx.x; t < p.blockSize; t += blockDim.x)
    {
        int64_t const k = plan.numPartialSlots + t;
        own[t] = static_cast<int64_t>(privatePages[pageOf(p, k)]) * tpb + slotOf(p, k);
    }

    // Partial slots: committed tokens are copied in at commit (pieces); this forward's tokens of
    // earlier blocks are written in by every forward (extras).
    int64_t const pieceCap = 3 * (tpb - 1);
    int64_t const extraCap = 2 * (tpb - 1);
    int64_t* pieceSrc = p.pieceSrc + i * pieceCap;
    int64_t* pieceDst = p.pieceDst + i * pieceCap;
    for (int64_t k = threadIdx.x; k < pieceCap; k += blockDim.x)
    {
        if (k < plan.numCommittedSlots)
        {
            int64_t const pos = positionAtPrivateSlot(plan, k);
            pieceSrc[k] = static_cast<int64_t>(viewOf(p, pos)) * p.rowsPerPage + slotOf(p, pos);
            pieceDst[k] = static_cast<int64_t>(privatePages[pageOf(p, k)]) * p.rowsPerPage + slotOf(p, k);
        }
        else
        {
            pieceSrc[k] = -1;
            pieceDst[k] = -1;
        }
    }
    int64_t* extraSrc = p.extraSrc + i * extraCap;
    int64_t* extraDst = p.extraDst + i * extraCap;
    int64_t const ownFirst
        = static_cast<int64_t>(privatePages[pageOf(p, plan.numPartialSlots)]) * tpb + slotOf(p, plan.numPartialSlots);
    for (int64_t t = threadIdx.x; t < extraCap; t += blockDim.x)
    {
        int64_t const k = plan.numCommittedSlots + t;
        if (k < plan.numPartialSlots)
        {
            int64_t const pos = positionAtPrivateSlot(plan, k);
            extraSrc[t] = pos - p.past;
            extraDst[t] = static_cast<int64_t>(privatePages[pageOf(p, k)]) * tpb + slotOf(p, k);
        }
        else
        {
            extraSrc[t] = i * p.blockSize;
            extraDst[t] = ownFirst;
        }
    }
}

// One CTA per (entry, layer); threads over the K and V rows of every head, one vector each.
template <typename Vec>
__global__ void copyKvSlotsKernel(Vec* __restrict__ pool, int64_t const* __restrict__ src,
    int64_t const* __restrict__ dst, int64_t numRows, int64_t vecsPerRow, int64_t tokensPerPage, int64_t rowsPerPage)
{
    int64_t const entry = blockIdx.x;
    int64_t const from = src[entry];
    if (from < 0)
    {
        return;
    }
    int64_t const to = dst[entry];
    int64_t const layerRow = static_cast<int64_t>(blockIdx.y) * rowsPerPage;
    int64_t const total = numRows * vecsPerRow;
    for (int64_t idx = threadIdx.x; idx < total; idx += blockDim.x)
    {
        int64_t const r = idx / vecsPerRow;
        int64_t const v = idx % vecsPerRow;
        int64_t const off = (layerRow + r * tokensPerPage) * vecsPerRow + v;
        pool[to * vecsPerRow + off] = pool[from * vecsPerRow + off];
    }
}

int pickVecBytes(CopyKvSlotsParams const& p, int elemSize)
{
    for (int bytes : {16, 8, 4, 2, 1})
    {
        if (bytes < elemSize)
        {
            break;
        }
        if (p.headDim * elemSize % bytes == 0 && reinterpret_cast<int64_t>(p.pool) % bytes == 0)
        {
            return bytes;
        }
    }
    return elemSize;
}

template <typename Vec>
void launchCopy(CopyKvSlotsParams const& p, int elemSize, cudaStream_t stream)
{
    int64_t const vecsPerRow = p.headDim * elemSize / static_cast<int64_t>(sizeof(Vec));
    dim3 const grid(static_cast<unsigned>(p.numEntries), static_cast<unsigned>(p.numLayers));
    copyKvSlotsKernel<Vec><<<grid, 128, 0, stream>>>(
        static_cast<Vec*>(p.pool), p.src, p.dst, 2 * p.numHeads, vecsPerRow, p.tokensPerPage, p.rowsPerPage);
    check_cuda_error(cudaGetLastError());
}

} // namespace

void invokeCausalLayout(CausalLayoutParams const& params, cudaStream_t stream)
{
    bool const shared = params.stagedSlots != nullptr || params.refillSrc != nullptr;
    int64_t const blocks = params.numBlocks + (shared ? 1 : 0);
    if (blocks <= 0)
    {
        return;
    }
    causalLayoutKernel<<<static_cast<unsigned>(blocks), constants::kThreads, 0, stream>>>(params);
    check_cuda_error(cudaGetLastError());
}

void invokeCopyKvSlots(CopyKvSlotsParams const& params, int elemSize, cudaStream_t stream)
{
    if (params.numEntries <= 0 || params.numLayers <= 0 || params.numHeads <= 0 || params.headDim <= 0)
    {
        return;
    }
    switch (pickVecBytes(params, elemSize))
    {
    case 16: launchCopy<uint4>(params, elemSize, stream); break;
    case 8: launchCopy<uint2>(params, elemSize, stream); break;
    case 4: launchCopy<uint32_t>(params, elemSize, stream); break;
    case 2: launchCopy<uint16_t>(params, elemSize, stream); break;
    case 1: launchCopy<uint8_t>(params, elemSize, stream); break;
    default: TLLM_THROW("copyKvSlots: unsupported element size %d bytes", elemSize);
    }
}

} // namespace kernels::causalLayout

TRTLLM_NAMESPACE_END
