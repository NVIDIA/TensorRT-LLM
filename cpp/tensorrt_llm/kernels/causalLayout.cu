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

using namespace tensorrt_llm::common;

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

constexpr int kThreads = 256;
constexpr int kMaxSpans = 2;              // the fixed region and the window
constexpr int kMaxRanges = 3 * kMaxSpans; // three partial pages, each cut by at most two spans

// What one causal block sees before its own tokens: whole logical pages as runs, and the visible
// positions of its partial pages as ascending ranges, each with the index of its first slot in the
// block's private region.
struct BlockPlan
{
    int64_t runA[kMaxSpans];
    int64_t runB[kMaxSpans];
    int64_t rangeLo[kMaxRanges];
    int64_t rangeHi[kMaxRanges];
    int64_t rangeK[kMaxRanges];
    int64_t whole;     // whole pages
    int64_t partial;   // visible positions on partial pages
    int64_t numStatic; // of those, the resident ones (position < past); they come first
    int numRuns;
    int numRanges;
};

__device__ void planBlock(CausalLayoutParams const& p, int64_t i, BlockPlan& plan)
{
    int64_t const tpb = p.tokensPerPage;
    int64_t const start = p.past + i * p.blockSize;
    int64_t const fixed = p.fixedTokens;
    int64_t const winStart = max(fixed, start - p.windowTokens);
    int64_t spanLo[kMaxSpans];
    int64_t spanHi[kMaxSpans];
    int numSpans = 0;
    if (winStart <= fixed)
    {
        if (start > 0)
        {
            spanLo[numSpans] = 0;
            spanHi[numSpans++] = start;
        }
    }
    else
    {
        if (fixed > 0)
        {
            spanLo[numSpans] = 0;
            spanHi[numSpans++] = fixed;
        }
        spanLo[numSpans] = winStart;
        spanHi[numSpans++] = start;
    }

    plan.whole = 0;
    plan.numRuns = numSpans;
    for (int s = 0; s < numSpans; ++s)
    {
        int64_t const a = (spanLo[s] + tpb - 1) / tpb;
        int64_t const b = max(a, spanHi[s] / tpb);
        plan.runA[s] = a;
        plan.runB[s] = b;
        plan.whole += b - a;
    }

    // The spans' edge pages, sorted; those no run covers are the partial pages.
    int64_t edges[2 * kMaxSpans];
    int numEdges = 0;
    for (int s = 0; s < numSpans; ++s)
    {
        edges[numEdges++] = spanLo[s] / tpb;
        edges[numEdges++] = (spanHi[s] - 1) / tpb;
    }
    for (int x = 1; x < numEdges; ++x)
    {
        int64_t const v = edges[x];
        int y = x - 1;
        for (; y >= 0 && edges[y] > v; --y)
        {
            edges[y + 1] = edges[y];
        }
        edges[y + 1] = v;
    }
    plan.numRanges = 0;
    plan.partial = 0;
    plan.numStatic = 0;
    for (int x = 0; x < numEdges; ++x)
    {
        int64_t const page = edges[x];
        if (x > 0 && edges[x - 1] == page)
        {
            continue;
        }
        bool covered = false;
        for (int r = 0; r < plan.numRuns; ++r)
        {
            covered |= plan.runA[r] <= page && page < plan.runB[r];
        }
        if (covered)
        {
            continue;
        }
        for (int s = 0; s < numSpans; ++s)
        {
            int64_t const lo = max(spanLo[s], page * tpb);
            int64_t const hi = min(spanHi[s], (page + 1) * tpb);
            if (hi <= lo)
            {
                continue;
            }
            plan.rangeLo[plan.numRanges] = lo;
            plan.rangeHi[plan.numRanges] = hi;
            plan.rangeK[plan.numRanges] = plan.partial;
            ++plan.numRanges;
            plan.partial += hi - lo;
            plan.numStatic += max(int64_t{0}, min(hi, p.past) - lo);
        }
    }
}

// The logical position of the k-th visible slot on the block's partial pages.
__device__ int64_t positionAt(BlockPlan const& plan, int64_t k)
{
    int r = 0;
    while (r + 1 < plan.numRanges && k >= plan.rangeK[r + 1])
    {
        ++r;
    }
    return plan.rangeLo[r] + (k - plan.rangeK[r]);
}

// The view page holding logical position pos. After a rotation the fixed tail on the page it
// shares with the history is still read from that page's old view, now at the ring's tail.
__device__ int32_t viewOf(CausalLayoutParams const& p, int64_t pos)
{
    int64_t const page = pos / p.tokensPerPage;
    if (p.dropPages > 0 && pos < p.fixedTokens && page == p.fixedTokens / p.tokensPerPage)
    {
        return p.table[p.numPages - p.dropPages];
    }
    return p.table[page];
}

__device__ void sharedWork(CausalLayoutParams const& p)
{
    int64_t const tpb = p.tokensPerPage;
    for (int64_t t = threadIdx.x; t < p.numStaged; t += blockDim.x)
    {
        int64_t const pos = p.past + t;
        p.stagedSlots[t] = static_cast<int64_t>(p.table[pos / tpb]) * tpb + pos % tpb;
    }
    if (p.refillSrc == nullptr)
    {
        return;
    }
    int64_t const tail = p.dropPages > 0 ? p.fixedTokens % tpb : 0;
    int64_t const fixedPages = p.fixedTokens / tpb;
    for (int64_t slot = threadIdx.x; slot < tpb; slot += blockDim.x)
    {
        if (slot < tail)
        {
            p.refillSrc[slot] = static_cast<int64_t>(p.table[p.numPages - p.dropPages]) * p.rowsPerPage + slot;
            p.refillDst[slot] = static_cast<int64_t>(p.table[fixedPages]) * p.rowsPerPage + slot;
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
        sharedWork(p);
        return;
    }
    if (threadIdx.x == 0)
    {
        planBlock(p, i, plan);
    }
    __syncthreads();

    int64_t const tpb = p.tokensPerPage;
    int32_t const* region = p.regions + i * p.regionPages;

    // The row: whole pages in logical order, then the private region, then zeros.
    int32_t* row = p.rows + i * p.rowLen;
    int32_t* offK = p.blockOffsets + i * 2 * p.rowLen;
    int32_t* offV = offK + p.rowLen;
    int64_t const len0 = plan.numRuns > 0 ? plan.runB[0] - plan.runA[0] : 0;
    for (int64_t j = threadIdx.x; j < p.rowLen; j += blockDim.x)
    {
        int32_t view = 0;
        if (j < plan.whole)
        {
            view = p.table[j < len0 ? plan.runA[0] + j : plan.runA[1] + (j - len0)];
        }
        else if (j < plan.whole + p.regionPages)
        {
            view = region[j - plan.whole];
        }
        row[j] = view;
        offK[j] = static_cast<int32_t>(view * p.kvFactor);
        offV[j] = static_cast<int32_t>(view * p.kvFactor + p.kvOffset);
    }
    if (threadIdx.x == 0)
    {
        p.seqLenKv[i] = static_cast<int32_t>(plan.whole * tpb + plan.partial + p.blockSize);
    }

    // The block's own tokens follow the partial slots in its region.
    int64_t* own = p.ownSlots + i * p.blockSize;
    for (int64_t t = threadIdx.x; t < p.blockSize; t += blockDim.x)
    {
        int64_t const k = plan.partial + t;
        own[t] = static_cast<int64_t>(region[k / tpb]) * tpb + k % tpb;
    }

    // Partial slots: resident positions are copied now (pieces), staged ones are written by
    // every forward (extras).
    int64_t const pieceCap = 3 * (tpb - 1);
    int64_t const extraCap = 2 * (tpb - 1);
    int64_t* pieceSrc = p.pieceSrc + i * pieceCap;
    int64_t* pieceDst = p.pieceDst + i * pieceCap;
    for (int64_t k = threadIdx.x; k < pieceCap; k += blockDim.x)
    {
        if (k < plan.numStatic)
        {
            int64_t const pos = positionAt(plan, k);
            pieceSrc[k] = static_cast<int64_t>(viewOf(p, pos)) * p.rowsPerPage + pos % tpb;
            pieceDst[k] = static_cast<int64_t>(region[k / tpb]) * p.rowsPerPage + k % tpb;
        }
        else
        {
            pieceSrc[k] = -1;
            pieceDst[k] = -1;
        }
    }
    int64_t* extraSrc = p.extraSrc + i * extraCap;
    int64_t* extraDst = p.extraDst + i * extraCap;
    int64_t const ownFirst = static_cast<int64_t>(region[plan.partial / tpb]) * tpb + plan.partial % tpb;
    for (int64_t t = threadIdx.x; t < extraCap; t += blockDim.x)
    {
        int64_t const k = plan.numStatic + t;
        if (k < plan.partial)
        {
            int64_t const pos = positionAt(plan, k);
            extraSrc[t] = pos - p.past;
            extraDst[t] = static_cast<int64_t>(region[k / tpb]) * tpb + k % tpb;
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
    causalLayoutKernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(params);
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

} // namespace kernels

TRTLLM_NAMESPACE_END
