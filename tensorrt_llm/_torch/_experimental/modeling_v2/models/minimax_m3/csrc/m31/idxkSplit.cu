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

// IDXK_SPLIT: index-K pages split over the TP8 ranks with an exact top-16 merge. See idxkSplit.h.
//
// The FSEL contract reproduced here (fselTopkSm107.cu, fsel_kernel, identity page table):
//   nraw = n_valid[row]; nvv = min(nraw, numCols)
//   nvv <= 16:  out[k] = k < nvv ? k : -1
//   cap = min(nvv, 9216); fend = nraw - local; flo = min(init, cap); fhi = max(0, cap - max(fend, flo));
//   need = 16 - flo - fhi
//   need <= 0:  out[k] = k < flo ? k : cap - fhi + (k - flo)
//   else:       out[0..flo) = 0..flo; out[16 - fhi..16) = cap - fhi..cap; out[flo..flo + need) = the top `need` of
//               the blocks p in [init, min(cap, fend)) by (okey(score bits) desc, p asc), in ascending p.
// okey(raw) = raw & sign ? ~raw : raw | 0x80000000 (the fp32 bit order; FSEL's `okey`).
#include "minimax_m3/csrc/m31/idxkSplit.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3
{
namespace idxkSplit
{
namespace
{

constexpr uint32_t kFull = 0xffffffffu;
constexpr int kTopkThreads = 256;
constexpr int kTopkWarps = kTopkThreads / 32;
constexpr int kKpt = (kMaxLocal + kTopkThreads - 1) / kTopkThreads; // 5 compact blocks per thread at 9216
constexpr int kMergeThreads = kWorld * kTopk;                       // one candidate per thread
constexpr uint32_t kTagMask = 0x3FFFFu;
constexpr uint32_t kPageMask = 0x3FFFu;
static_assert(kMaxBlocks <= static_cast<int>(kPageMask) + 1, "block ids must fit the 14-bit page field");
static_assert(kMergeThreads == 128, "the merge ranks 8 x 16 candidates with 4 warps");

__device__ __forceinline__ void pdlWait()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void pdlTrigger()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

__device__ __forceinline__ uint32_t okey(uint32_t raw)
{
    return (raw & 0x80000000u) ? ~raw : (raw | 0x80000000u);
}

__device__ __forceinline__ unsigned long long ldVolatile(unsigned long long const* p)
{
    return *reinterpret_cast<unsigned long long const volatile*>(p);
}

__device__ __forceinline__ void stVolatile(unsigned long long* p, unsigned long long v)
{
    *reinterpret_cast<unsigned long long volatile*>(p) = v;
}

// Store one word at `off` of every rank's layer workspace.
__device__ __forceinline__ void publishWord(Dest const& d, int64_t off, unsigned long long v)
{
    if (d.mc != nullptr)
    {
        stVolatile(d.mc + off, v);
        return;
    }
#pragma unroll
    for (int r = 0; r < kWorld; ++r)
        if (r < d.world)
            stVolatile(d.dst[r] + off, v);
}

__device__ __forceinline__ unsigned long long warpMax(unsigned long long v)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
    {
        unsigned long long const t = __shfl_xor_sync(kFull, v, o);
        v = t > v ? t : v;
    }
    return v;
}

__device__ __forceinline__ int64_t qOffset(int half, int head, int row, int rowsMax)
{
    return ((static_cast<int64_t>(half) * kHeads + head) * rowsMax + row) * kQWords;
}

__device__ __forceinline__ int64_t cOffset(int half, int src, int head, int row, int rowsMax)
{
    return static_cast<int64_t>(2) * kHeads * rowsMax * kQWords
        + (((static_cast<int64_t>(half) * kWorld + src) * kHeads + head) * rowsMax + row) * kTopk;
}

// The FSEL row geometry of one query row. ranked: the row takes a rank (need > 0 and more than 16 valid blocks).
struct RowGeom
{
    int nvv, cap, fend, flo, fhi, need;
    bool shortRow, allForced;
};

__device__ __forceinline__ RowGeom rowGeom(int nraw, int numCols, int init, int local)
{
    RowGeom g;
    g.nvv = min(nraw, numCols);
    g.shortRow = g.nvv <= kTopk;
    g.cap = min(g.nvv, kMaxBlocks);
    g.fend = nraw - local;
    g.flo = min(init, g.cap);
    g.fhi = max(0, g.cap - max(g.fend, g.flo));
    g.need = kTopk - g.flo - g.fhi;
    g.allForced = g.need <= 0;
    return g;
}

__global__ void compactKernel(int32_t const* __restrict__ bt, int64_t btStride, int32_t btWidth,
    int32_t const* __restrict__ seqLens, int32_t* __restrict__ cbt, int64_t cbtStride, int32_t cbtWidth,
    int32_t* __restrict__ scoreLens, int32_t* __restrict__ epoch, int32_t rank, int32_t qlen)
{
    int const b = blockIdx.x;
    int const seq = seqLens[b];
    int const hist = min(max((seq - 1) / kPage, 0), btWidth);
    int nLocal = hist > rank ? (hist - rank + kWorld - 1) / kWorld : 0;
    nLocal = min(nLocal, cbtWidth - 1);
    int32_t const* const row = bt + static_cast<int64_t>(b) * btStride;
    int32_t const first = row[0];
    for (int j = threadIdx.x; j < cbtWidth; j += blockDim.x)
        cbt[static_cast<int64_t>(b) * cbtStride + j] = j < nLocal ? row[j * kWorld + rank] : first;
    if (threadIdx.x == 0)
    {
        scoreLens[b] = nLocal * kPage + qlen;
        if (b == 0)
            epoch[0] += 1;
    }
}

// One warp per query row: 32 words of 4 FP8 bytes each.
__global__ void qxKernel(uint8_t const* __restrict__ idxQ, int64_t qRowStride, int32_t rowsMax, Dest dest,
    unsigned long long const* __restrict__ local, uint8_t* __restrict__ qTile, int32_t const* __restrict__ epochPtr,
    int32_t rank, bool publish, bool gather)
{
    int const row = blockIdx.x, lane = threadIdx.x;
    pdlWait();
    uint32_t const epoch = static_cast<uint32_t>(*epochPtr);
    int const half = static_cast<int>(epoch & 1u);
    if (publish && (rank & 1) == 0)
    {
        uint32_t const v = reinterpret_cast<uint32_t const*>(idxQ + row * qRowStride)[lane];
        publishWord(
            dest, qOffset(half, rank >> 1, row, rowsMax) + lane, (static_cast<unsigned long long>(epoch) << 32) | v);
    }
    if (gather)
    {
        uint32_t* const tile = reinterpret_cast<uint32_t*>(qTile) + static_cast<int64_t>(row) * kHeads * kQWords;
#pragma unroll
        for (int h = 0; h < kHeads; ++h)
        {
            unsigned long long const* const src = local + qOffset(half, h, row, rowsMax) + lane;
            unsigned long long w;
            do
            {
                w = ldVolatile(src);
            } while (static_cast<uint32_t>(w >> 32) != epoch);
            tile[h * kQWords + lane] = static_cast<uint32_t>(w);
        }
    }
    __syncwarp();
    pdlTrigger();
}

// One CTA per (query row, head): this rank's exact top 16 by the packed key, published to every rank.
__global__ void __launch_bounds__(kTopkThreads) topkKernel(float const* __restrict__ scores, int64_t headStride,
    int64_t rowStride, int64_t blockStride, int32_t const* __restrict__ nValid, int32_t rowsMax, int32_t numCols,
    int32_t init, int32_t localBlocks, Dest dest, int32_t const* __restrict__ epochPtr, int32_t rank)
{
    __shared__ unsigned long long sWarp[kTopkWarps][kTopk];
    int const row = blockIdx.x, head = blockIdx.y, tid = threadIdx.x;
    int const warp = tid >> 5, lane = tid & 31;
    // Pre-wait: the valid-block count is final before the step's first layer (a plain launch writes it).
    RowGeom const g = rowGeom(__ldg(nValid + row), numCols, init, localBlocks);
    pdlWait();
    pdlTrigger();
    if (g.shortRow || g.allForced)
        return;
    uint32_t const epoch = static_cast<uint32_t>(*epochPtr);
    int const half = static_cast<int>(epoch & 1u);
    unsigned long long const tagBits = static_cast<unsigned long long>(epoch & kTagMask) << 14;
    int const hi = min(g.cap, g.fend);
    int const jlo = init > rank ? (init - rank + kWorld - 1) / kWorld : 0;
    int const jhi = hi > rank ? (hi - rank + kWorld - 1) / kWorld : 0;
    float const* const srow = scores + head * headStride + row * rowStride;
    unsigned long long k[kKpt];
#pragma unroll
    for (int i = 0; i < kKpt; ++i)
    {
        int const j = jlo + tid + kTopkThreads * i;
        k[i] = 0ull;
        if (j < jhi)
        {
            uint32_t const raw = __float_as_uint(__ldcg(srow + j * blockStride));
            int const p = j * kWorld + rank;
            k[i] = (static_cast<unsigned long long>(okey(raw)) << 32) | tagBits
                | static_cast<unsigned long long>(kPageMask - static_cast<uint32_t>(p));
        }
    }
    // Per warp: 16 rounds of the warp maximum; the (unique) holder retires it.
#pragma unroll 1
    for (int r = 0; r < kTopk; ++r)
    {
        unsigned long long best = 0ull;
#pragma unroll
        for (int i = 0; i < kKpt; ++i)
            best = k[i] > best ? k[i] : best;
        unsigned long long const m = warpMax(best);
        if (m != 0ull && best == m)
        {
#pragma unroll
            for (int i = 0; i < kKpt; ++i)
                k[i] = k[i] == m ? 0ull : k[i];
        }
        if (lane == 0)
            sWarp[warp][r] = m;
    }
    __syncthreads();
    if (warp != 0)
        return;
    // Warp 0 over the 8 x 16 warp winners: 4 per lane, 16 more rounds.
    unsigned long long c[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        c[i] = sWarp[(lane + 32 * i) / kTopk][(lane + 32 * i) % kTopk];
    int64_t const off = cOffset(half, rank, head, row, rowsMax);
#pragma unroll 1
    for (int r = 0; r < kTopk; ++r)
    {
        unsigned long long best = 0ull;
#pragma unroll
        for (int i = 0; i < 4; ++i)
            best = c[i] > best ? c[i] : best;
        unsigned long long const m = warpMax(best);
        if (m != 0ull && best == m)
        {
#pragma unroll
            for (int i = 0; i < 4; ++i)
                c[i] = c[i] == m ? 0ull : c[i];
        }
        // An empty slot (fewer than 16 candidates) carries the tag with key 0: it never outranks a real candidate.
        if (lane == r)
            publishWord(dest, off + r, m != 0ull ? m : tagBits);
    }
}

// One CTA per query row: the FSEL table of `head` from the 8 ranks' candidates.
__global__ void __launch_bounds__(kMergeThreads) mergeKernel(unsigned long long* __restrict__ local,
    int32_t const* __restrict__ nValid, int32_t rowsMax, int32_t numCols, int32_t init, int32_t localBlocks,
    int32_t head, int32_t* __restrict__ out, int64_t outRowStride, int32_t const* __restrict__ epochPtr, int32_t trig)
{
    __shared__ unsigned long long sCand[kMergeThreads];
    __shared__ int sWin[kTopk];
    int const row = blockIdx.x, tid = threadIdx.x;
    int32_t* const orw = out + row * outRowStride;
    RowGeom const g = rowGeom(__ldg(nValid + row), numCols, init, localBlocks);
    pdlWait();
    if (trig == 0)
        pdlTrigger();
    if (g.shortRow)
    {
        if (tid < kTopk)
            orw[tid] = tid < g.nvv ? tid : -1;
        if (trig != 0)
            pdlTrigger();
        return;
    }
    if (g.allForced)
    {
        if (tid < kTopk)
            orw[tid] = tid < g.flo ? tid : g.cap - g.fhi + (tid - g.flo);
        if (trig != 0)
            pdlTrigger();
        return;
    }
    if (tid < g.flo)
        orw[tid] = tid;
    else if (tid >= kTopk - g.fhi && tid < kTopk)
        orw[tid] = g.cap - (kTopk - tid);
    uint32_t const epoch = static_cast<uint32_t>(*epochPtr);
    int const half = static_cast<int>(epoch & 1u);
    uint32_t const tag = epoch & kTagMask;
    int const src = tid / kTopk, slot = tid % kTopk;
    unsigned long long* const p = local + cOffset(half, src, head, row, rowsMax) + slot;
    unsigned long long v;
    do
    {
        v = ldVolatile(p);
    } while (((static_cast<uint32_t>(v) >> 14) & kTagMask) != tag);
    sCand[tid] = v;
    // Re-arm the consumed slot with a tag of the other parity: this half only ever carries epochs of this parity, so
    // the slot can never match again until its producer rewrites it (epoch + 2 at the earliest, which needs this rank's
    // next-step topk, stream-ordered after this kernel). Without it the 18-bit tag aliases after 2^18 steps on a row
    // that was idle since (a batch that shrank and regrew), and the merge would take a stale candidate.
    stVolatile(p, static_cast<unsigned long long>((epoch + 1u) & kTagMask) << 14);
    __syncthreads();
    if (tid >= 32)
        return;
    int const lane = tid;
    unsigned long long c[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        c[i] = sCand[lane + 32 * i];
    int pg = 0x7fffffff;
#pragma unroll 1
    for (int r = 0; r < g.need; ++r)
    {
        unsigned long long best = 0ull;
#pragma unroll
        for (int i = 0; i < 4; ++i)
            best = c[i] > best ? c[i] : best;
        unsigned long long const m = warpMax(best);
        if (best == m)
        {
#pragma unroll
            for (int i = 0; i < 4; ++i)
                c[i] = c[i] == m ? 0ull : c[i];
        }
        if (lane == r)
            pg = static_cast<int>(kPageMask - (static_cast<uint32_t>(m) & kPageMask));
    }
    if (lane < kTopk)
        sWin[lane] = pg;
    __syncwarp();
    int pos = 0;
#pragma unroll
    for (int j = 0; j < kTopk; ++j)
        pos += sWin[j] < pg;
    if (lane < g.need)
        orw[g.flo + pos] = pg;
    if (trig != 0)
        pdlTrigger();
}

cudaLaunchConfig_t pdlConfig(dim3 grid, dim3 block, bool usePdl, cudaStream_t stream, cudaLaunchAttribute* attr)
{
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.stream = stream;
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = usePdl ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cfg;
}

} // namespace

cudaError_t invokeCompact(int32_t const* blockTable, int64_t btStride, int32_t btWidth, int32_t const* seqLens,
    int32_t batch, int32_t* cbt, int64_t cbtStride, int32_t cbtWidth, int32_t* scoreLens, int32_t* epoch, int32_t rank,
    int32_t qlen, cudaStream_t stream)
{
    if (batch <= 0 || cbtWidth < 1 || btWidth < 1 || rank < 0 || rank >= kWorld || qlen < 1 || qlen > kPage)
        return cudaErrorInvalidValue;
    // A plain launch: everything after it in the step reads its outputs before its own grid wait.
    compactKernel<<<batch, 256, 0, stream>>>(
        blockTable, btStride, btWidth, seqLens, cbt, cbtStride, cbtWidth, scoreLens, epoch, rank, qlen);
    return cudaGetLastError();
}

cudaError_t invokeQx(uint8_t const* idxQ, int64_t qRowStride, int32_t rows, int32_t rowsMax, Dest const& dest,
    unsigned long long const* local, uint8_t* qTile, int32_t const* epoch, int32_t rank, bool publish, bool gather,
    bool usePdl, cudaStream_t stream)
{
    if (rows <= 0 || rows > rowsMax || rank < 0 || rank >= kWorld || (qRowStride % 4) != 0
        || (dest.mc == nullptr && dest.world != kWorld))
        return cudaErrorInvalidValue;
    cudaLaunchAttribute attr[1];
    cudaLaunchConfig_t cfg = pdlConfig(dim3(rows), dim3(32), usePdl, stream, attr);
    return cudaLaunchKernelEx(
        &cfg, qxKernel, idxQ, qRowStride, rowsMax, dest, local, qTile, epoch, rank, publish, gather);
}

cudaError_t invokeTopk(float const* scores, int64_t headStride, int64_t rowStride, int64_t blockStride,
    int32_t const* nValidBlocks, int32_t rows, int32_t rowsMax, int32_t numCols, int32_t initBlocks,
    int32_t localBlocks, Dest const& dest, int32_t const* epoch, int32_t rank, bool usePdl, cudaStream_t stream)
{
    if (rows <= 0 || rows > rowsMax || numCols < 1 || rank < 0 || rank >= kWorld || initBlocks < 0 || localBlocks < 0
        || (dest.mc == nullptr && dest.world != kWorld))
        return cudaErrorInvalidValue;
    cudaLaunchAttribute attr[1];
    cudaLaunchConfig_t cfg = pdlConfig(dim3(rows, kHeads), dim3(kTopkThreads), usePdl, stream, attr);
    return cudaLaunchKernelEx(&cfg, topkKernel, scores, headStride, rowStride, blockStride, nValidBlocks, rowsMax,
        numCols, initBlocks, localBlocks, dest, epoch, rank);
}

cudaError_t invokeMerge(unsigned long long* local, int32_t const* nValidBlocks, int32_t rows, int32_t rowsMax,
    int32_t numCols, int32_t initBlocks, int32_t localBlocks, int32_t head, int32_t* out, int64_t outRowStride,
    int32_t const* epoch, bool usePdl, int32_t triggerPhase, cudaStream_t stream)
{
    if (rows <= 0 || rows > rowsMax || numCols < 1 || head < 0 || head >= kHeads || initBlocks < 0 || localBlocks < 0
        || (triggerPhase != 0 && triggerPhase != 2))
        return cudaErrorInvalidValue;
    cudaLaunchAttribute attr[1];
    cudaLaunchConfig_t cfg = pdlConfig(dim3(rows), dim3(kMergeThreads), usePdl, stream, attr);
    return cudaLaunchKernelEx(&cfg, mergeKernel, local, nValidBlocks, rowsMax, numCols, initBlocks, localBlocks, head,
        out, outRowStride, epoch, triggerPhase);
}

} // namespace idxkSplit
} // namespace minimaxM3
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
