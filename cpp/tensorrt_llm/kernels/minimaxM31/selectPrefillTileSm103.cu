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

// MiniMax-M3.1 GB300 (sm_103) target-local copy of the #20629 prefill tile select
// (cpp/tensorrt_llm/kernels/minimaxM3/selectPrefillTile.* @ cf8a77e816); namespace and op name are M3.1's.

// Prefill top-16 block select over head-major scores [num_kv_heads, num_blocks, total_q] with unit query stride.
//
// One CTA per (KV head, 8 consecutive queries). The 8 queries of one block are one 32-byte sector, so the CTA reads
// each sector it needs exactly once, as two 16-byte loads. The tile's rows are staged in shared memory as ordered keys
// of the forced scores. A long tile's key rows fill most of shared memory, so one CTA runs per SM; its loads then come
// from extra staging warps (kWideThreads from kWideMinBlocks blocks), which exit before the selection. Each of the
// first 8 warps then selects one row:
//   - the 32 lane-stride group minima are sorted in the warp and the 16th of them bounds the row's 16th smallest key,
//     so every key at or below that bound is a candidate; the candidates (compacted in ascending block order) are
//     ranked exactly by (key, block);
//   - a row with more than kMaxCand candidates (heavy ties) falls back to an exact 4 x 8-bit radix select.
// Ordering is the descending ordered key of the float bits used by minimax_m3_select_blocks, ties to the smaller
// block, so the result is the same top-16 set.

#include "tensorrt_llm/kernels/minimaxM31/selectPrefillTileSm103.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"

#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <mutex>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::minimaxM31Sm103
{

namespace
{
constexpr int kTopK = 16;
constexpr int kTileQ = 8;
constexpr int kThreads = kTileQ * 32;
// From this many blocks on, the tile is staged by kWideThreads threads with kWideUnroll blocks in flight each.
constexpr int kWideMinBlocks = 2600;
constexpr int kWideThreads = 1024;
constexpr int kWideUnroll = 4;
constexpr uint32_t kFull = 0xFFFFFFFFu;
constexpr float kInitScore = 1.0e30f;
constexpr float kLocalScore = 1.0e29f;
constexpr uint32_t kNegInfKey = 0xFF800000u;
// Per-warp scratch: the radix histogram (256 bins) or the candidate list (keys then blocks).
constexpr int kMaxCand = 256;
constexpr int kWarpScratch = 2 * kMaxCand;
// Blocks each thread has in flight while staging a narrow tile.
constexpr int kStageUnroll = 8;

__device__ __forceinline__ uint32_t orderedKey(float x)
{
    uint32_t const bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? bits : (~bits & 0x7fffffffu);
}

// Row pitch in keys: >= num_blocks and == 4 (mod 32), so the staging writes of
// 8 rows x 4 consecutive blocks per warp hit 32 distinct banks.
__host__ __device__ __forceinline__ int rowPitch(int numBlocks)
{
    int p = (numBlocks + 31) / 32 * 32 + 4;
    return p;
}

// kNt staging threads with kUnroll blocks in flight each; warps 0 .. kTileQ - 1 select. kIlp builds each row's 32
// stride-class minima while staging (a staging thread only ever sees one class) and issues the candidate scan's
// shared-memory loads four at a time; the minima, the bound and every decision are the same as the scalar scans'.
template <int kNt, int kUnroll, bool kIlp>
__global__ void __launch_bounds__(kNt) selectPrefillTileKernel(float const* __restrict__ scores, int64_t headStride,
    int64_t blockStride, int32_t const* __restrict__ nValid, int32_t* __restrict__ out, int64_t outQueryStride,
    int64_t outHeadStride, int numBlocks, int totalQ, int initBlocks, int localBlocks)
{
    extern __shared__ uint32_t smem[];
    __shared__ uint32_t sClassMin[kTileQ * 32]; // [row][b % 32] (kIlp)
    int const pitch = rowPitch(numBlocks);
    uint32_t* keys = smem;                      // [kTileQ][pitch]
    uint32_t* hist = smem + kTileQ * pitch;     // [kTileQ][kWarpScratch]

    int const head = blockIdx.y;
    int const q0 = blockIdx.x * kTileQ;
    int const tid = threadIdx.x;
    int const warp = tid / 32;
    int const lane = tid % 32;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    // ---- stage: thread loads queries 4 * jh .. 4 * jh + 3 of block b = tid / 2 + (kNt / 2) * i as one float4 ----
    {
        int const jh = tid % 2;
        int rowEnd[4];
        int64_t localStart[4];
        int tileEnd = 0;
#pragma unroll
        for (int c = 0; c < 4; ++c)
        {
            int const q = q0 + 4 * jh + c;
            rowEnd[c] = 0;
            localStart[c] = 0;
            if (q < totalQ)
            {
                int const raw = nValid[q];
                rowEnd[c] = max(0, min(raw, numBlocks));
                localStart[c] = max(static_cast<int64_t>(raw) - localBlocks, static_cast<int64_t>(0));
            }
            tileEnd = max(tileEnd, rowEnd[c]);
        }
        tileEnd = max(tileEnd, __shfl_xor_sync(kFull, tileEnd, 1));
        if constexpr (kIlp)
        {
            if (tid < kTileQ * 32)
            {
                sClassMin[tid] = 0xFFFFFFFFu;
            }
        }
        __shared__ int sTileEnd[kNt / 32];
        if (lane == 0)
        {
            sTileEnd[warp] = tileEnd;
        }
        __syncthreads();
        tileEnd = 0;
#pragma unroll
        for (int w = 0; w < kNt / 32; ++w)
        {
            tileEnd = max(tileEnd, sTileEnd[w]);
        }
        // Two threads read a tile's 8 queries of one block, one aligned 32-byte sector, when the base and the
        // strides keep 16-byte alignment; otherwise, and for a partial last tile, they read one float at a time.
        bool const fullVec = (reinterpret_cast<uintptr_t>(scores) & 15) == 0 && ((headStride | blockStride) & 3) == 0
            && (q0 + 4 * jh + 3) < totalQ;
        float const* base = scores + static_cast<int64_t>(head) * headStride + q0 + 4 * jh;
        constexpr int kBlocksPerIter = kNt / 2;
        static_assert(kBlocksPerIter % 32 == 0, "a staging thread stays in one stride class b % 32");
        uint32_t classMin[4] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
        for (int b0 = tid / 2; b0 < tileEnd; b0 += kBlocksPerIter * kUnroll)
        {
            float4 v[kUnroll];
#pragma unroll
            for (int u = 0; u < kUnroll; ++u)
            {
                int const b = b0 + u * kBlocksPerIter;
                float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
                if (b < tileEnd)
                {
                    float const* p = base + static_cast<int64_t>(b) * blockStride;
                    if (fullVec)
                    {
                        x = __ldg(reinterpret_cast<float4 const*>(p));
                    }
                    else
                    {
                        x.x = b < rowEnd[0] ? __ldg(p) : 0.f;
                        x.y = b < rowEnd[1] ? __ldg(p + 1) : 0.f;
                        x.z = b < rowEnd[2] ? __ldg(p + 2) : 0.f;
                        x.w = b < rowEnd[3] ? __ldg(p + 3) : 0.f;
                    }
                }
                v[u] = x;
            }
#pragma unroll
            for (int u = 0; u < kUnroll; ++u)
            {
                int const b = b0 + u * kBlocksPerIter;
                if (b < tileEnd)
                {
                    float const vals[4] = {v[u].x, v[u].y, v[u].z, v[u].w};
#pragma unroll
                    for (int c = 0; c < 4; ++c)
                    {
                        uint32_t key = kNegInfKey;
                        if (b < rowEnd[c])
                        {
                            float s = vals[c];
                            if (b < initBlocks)
                            {
                                s = kInitScore;
                            }
                            if (b >= localStart[c])
                            {
                                s = kLocalScore;
                            }
                            key = orderedKey(s);
                            if constexpr (kIlp)
                            {
                                classMin[c] = min(classMin[c], key);
                            }
                        }
                        keys[(4 * jh + c) * pitch + b] = key;
                    }
                }
            }
        }
        if constexpr (kIlp)
        {
            int const cls = (tid / 2) % 32;
#pragma unroll
            for (int c = 0; c < 4; ++c)
            {
                if (classMin[c] != 0xFFFFFFFFu)
                {
                    atomicMin(&sClassMin[(4 * jh + c) * 32 + cls], classMin[c]);
                }
            }
        }
    }
    __syncthreads();
    if (warp >= kTileQ)
    {
        return;
    }

    // ---- select: warp w owns query q0 + w ----
    int const q = q0 + warp;
    if (q >= totalQ)
    {
        return;
    }
    int const raw = nValid[q];
    int const len = max(0, min(raw, numBlocks));
    uint32_t const* row = keys + warp * pitch;
    uint32_t* h = hist + warp * kWarpScratch;
    int32_t* dst = out + static_cast<int64_t>(q) * outQueryStride + static_cast<int64_t>(head) * outHeadStride;

    // Threshold key K and how many keys equal to K to take (tieTake). Keys < K are all in.
    if (len > kTopK)
    {
        // Fast path. Group g = lanes' stride classes b % 32: each lane's best (smallest) key.
        // T = the 16th smallest of the 32 group bests, so at least 16 keys are <= T and the
        // row's 16th smallest key is <= T: every top-16 key (and every key tied with one) is a
        // candidate. Candidates are compacted in ascending block order and ranked exactly.
        uint32_t best = 0xFFFFFFFFu;
        bool const any = lane < len;
        if constexpr (kIlp)
        {
            best = sClassMin[warp * 32 + lane];
        }
        else
        {
            for (int b = lane; b < len; b += 32)
            {
                best = min(best, row[b]);
            }
        }
        // Ascending bitonic sort of the 32 group bests; empty groups sort last.
        uint32_t v = any ? best : 0xFFFFFFFFu;
        bool vEmpty = !any;
#pragma unroll
        for (int size = 2; size <= 32; size *= 2)
        {
#pragma unroll
            for (int stride = size / 2; stride > 0; stride /= 2)
            {
                uint32_t const pv = __shfl_xor_sync(kFull, v, stride);
                bool const pe = __shfl_xor_sync(kFull, vEmpty, stride);
                bool const up = ((lane & size) == 0) == ((lane & stride) == 0);
                // order: (empty, value); smaller first when 'up'
                bool const pLess = (!pe && vEmpty) || (pe == vEmpty && pv < v);
                bool const pGreater = (pe && !vEmpty) || (pe == vEmpty && pv > v);
                if ((up && pLess) || (!up && pGreater))
                {
                    v = pv;
                    vEmpty = pe;
                }
            }
        }
        uint32_t const bound = __shfl_sync(kFull, v, kTopK - 1);
        uint32_t* candKey = h;
        uint32_t* candBlk = h + kMaxCand;
        int nCand = 0;
        int cb0 = 0;
        if constexpr (kIlp)
        {
            for (; cb0 + 128 <= len; cb0 += 128)
            {
                uint32_t k[4];
#pragma unroll
                for (int j = 0; j < 4; ++j)
                {
                    k[j] = row[cb0 + 32 * j + lane];
                }
#pragma unroll
                for (int j = 0; j < 4; ++j)
                {
                    bool const c = k[j] <= bound;
                    uint32_t const m = __ballot_sync(kFull, c);
                    int const slot = nCand + __popc(m & ((1u << lane) - 1u));
                    if (c && slot < kMaxCand)
                    {
                        candKey[slot] = k[j];
                        candBlk[slot] = static_cast<uint32_t>(cb0 + 32 * j + lane);
                    }
                    nCand += __popc(m);
                }
            }
        }
        for (int b0 = cb0; b0 < len; b0 += 32)
        {
            int const b = b0 + lane;
            bool const c = b < len && row[b] <= bound;
            uint32_t const m = __ballot_sync(kFull, c);
            int const slot = nCand + __popc(m & ((1u << lane) - 1u));
            if (c && slot < kMaxCand)
            {
                candKey[slot] = row[b];
                candBlk[slot] = static_cast<uint32_t>(b);
            }
            nCand += __popc(m);
        }
        __syncwarp();
        if (nCand <= kMaxCand)
        {
            // Exact rank by (key, block); candidates are already in ascending block order, so a
            // second ballot compaction of the rank < 16 ones writes the output ascending.
            int written = 0;
            for (int i0 = 0; i0 < nCand; i0 += 32)
            {
                int const i = i0 + lane;
                bool sel = false;
                uint32_t k = 0xFFFFFFFFu;
                uint32_t blk = 0;
                if (i < nCand)
                {
                    k = candKey[i];
                    blk = candBlk[i];
                    int rank = 0;
                    for (int jdx = 0; jdx < nCand; ++jdx)
                    {
                        uint32_t const kj = candKey[jdx];
                        rank += (kj < k) || (kj == k && jdx < i);
                    }
                    sel = rank < kTopK;
                }
                uint32_t const m = __ballot_sync(kFull, sel && k != kNegInfKey);
                int const slot = written + __popc(m & ((1u << lane) - 1u));
                if (sel && k != kNegInfKey)
                {
                    dst[slot] = static_cast<int32_t>(blk);
                }
                written += __popc(m);
            }
            for (int r = written + lane; r < kTopK; r += 32)
            {
                dst[r] = -1;
            }
            return;
        }
        __syncwarp();
    }

    uint32_t thresh = 0xFFFFFFFFu;
    int tieTake = kTopK;
    if (len > kTopK)
    {
        uint32_t prefix = 0;
        int remaining = kTopK; // rank (1-based) of the wanted key among keys matching prefix
#pragma unroll 1
        for (int pass = 0; pass < 4; ++pass)
        {
            int const shift = 24 - 8 * pass;
            uint32_t const hiMask = pass == 0 ? 0u : (0xFFFFFFFFu << (shift + 8));
            for (int i = lane; i < 256; i += 32)
            {
                h[i] = 0;
            }
            __syncwarp();
            for (int b0 = 0; b0 < len; b0 += 32)
            {
                int const b = b0 + lane;
                bool const active = b < len;
                uint32_t const k = active ? row[b] : 0u;
                bool const match = active && ((k & hiMask) == prefix);
                uint32_t const digit = (k >> shift) & 0xFFu;
                uint32_t const m = __ballot_sync(kFull, match);
                if (match)
                {
                    uint32_t const peers = __match_any_sync(m, digit);
                    if ((__ffs(peers) - 1) == lane)
                    {
                        atomicAdd(&h[digit], static_cast<uint32_t>(__popc(peers)));
                    }
                }
            }
            __syncwarp();
            // Find the digit holding the remaining-th smallest key: each lane owns 8 bins.
            uint32_t c[8];
            uint32_t local = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                c[i] = h[lane * 8 + i];
                local += c[i];
            }
            uint32_t incl = local;
#pragma unroll
            for (int o = 1; o < 32; o *= 2)
            {
                uint32_t const v = __shfl_up_sync(kFull, incl, o);
                if (lane >= o)
                {
                    incl += v;
                }
            }
            uint32_t const excl = incl - local;
            bool const mine = excl < static_cast<uint32_t>(remaining) && incl >= static_cast<uint32_t>(remaining);
            uint32_t const owner = __ballot_sync(kFull, mine);
            int const ownerLane = __ffs(owner) - 1;
            int digitSel = 0;
            uint32_t below = 0;
            if (lane == ownerLane)
            {
                uint32_t acc = excl;
#pragma unroll
                for (int i = 0; i < 8; ++i)
                {
                    if (acc + c[i] >= static_cast<uint32_t>(remaining))
                    {
                        digitSel = lane * 8 + i;
                        below = acc;
                        break;
                    }
                    acc += c[i];
                }
            }
            digitSel = __shfl_sync(kFull, digitSel, ownerLane);
            below = __shfl_sync(kFull, below, ownerLane);
            prefix |= static_cast<uint32_t>(digitSel) << shift;
            remaining -= static_cast<int>(below);
            __syncwarp();
        }
        thresh = prefix;
        tieTake = remaining; // keys equal to thresh to take, smallest blocks first
    }

    // ---- collect in ascending block order ----
    // len <= 16: every valid block is a pick (thresh = max, tieTake = 16 covers all).
    int count = 0;
    int ties = 0;
    for (int b0 = 0; b0 < len && count < kTopK; b0 += 32)
    {
        int const b = b0 + lane;
        bool const active = b < len;
        uint32_t const k = active ? row[b] : 0xFFFFFFFFu;
        bool const strict = active && (len <= kTopK || k < thresh);
        bool const tie = active && len > kTopK && k == thresh;
        uint32_t const tieMask = __ballot_sync(kFull, tie);
        int const tieRank = ties + __popc(tieMask & ((1u << lane) - 1u));
        bool const take = strict || (tie && tieRank < tieTake);
        uint32_t const takeMask = __ballot_sync(kFull, take);
        int const slot = count + __popc(takeMask & ((1u << lane) - 1u));
        if (take && slot < kTopK)
        {
            dst[slot] = (k == kNegInfKey) ? -1 : b;
        }
        count += __popc(takeMask);
        ties += __popc(tieMask);
    }
    __syncwarp();
    // Picks forced to -inf were written as -1 in place; compact the valid ids to the front, -1 tail.
    if (lane == 0)
    {
        int32_t ids[kTopK];
        int n = 0;
        int const picks = min(count, kTopK);
        for (int i = 0; i < picks; ++i)
        {
            int32_t const v = dst[i];
            if (v >= 0)
            {
                ids[n++] = v;
            }
        }
        for (int i = 0; i < kTopK; ++i)
        {
            dst[i] = i < n ? ids[i] : -1;
        }
    }
}
} // namespace

cudaError_t invokeMinimaxM3SelectPrefillTile(float const* scores, int64_t headStride, int64_t blockStride,
    int32_t const* nValidBlocks, int32_t* out, int64_t outQueryStride, int64_t outHeadStride, int32_t numKvHeads,
    int32_t numBlocks, int32_t totalQ, int32_t initBlocks, int32_t localBlocks, cudaStream_t stream)
{
    if (totalQ == 0 || numKvHeads == 0)
    {
        return cudaSuccess;
    }
    if (numBlocks > kMinimaxM3SelectPrefillTileMaxBlocks)
    {
        return cudaErrorInvalidValue;
    }
    // TRTLLM_M3_SELECT_TILE_V2=1 keeps the previous launch: 256 threads for every tile, one load at a time per scan.
    static bool const v2Only = common::getBoolEnv("TRTLLM_M3_SELECT_TILE_V2");
    static std::once_flag logged;
    std::call_once(logged,
        []
        {
            TLLM_LOG_INFO(v2Only
                    ? "minimaxM3SelectPrefillTile: staging reads each block's sector as two 16-byte loads"
                    : "minimaxM3SelectPrefillTile: v4, stride-class minima while staging, 4-wide candidate "
                      "scan and 1024 staging threads from 2600 blocks");
        });
    size_t const smemBytes
        = (static_cast<size_t>(kTileQ) * rowPitch(numBlocks) + kTileQ * kWarpScratch) * sizeof(uint32_t);
    bool const wide = !v2Only && numBlocks >= kWideMinBlocks;
    auto kernel = v2Only ? selectPrefillTileKernel<kThreads, kStageUnroll, false>
        : wide           ? selectPrefillTileKernel<kWideThreads, kWideUnroll, true>
                         : selectPrefillTileKernel<kThreads, kStageUnroll, true>;
    cudaError_t err
        = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smemBytes));
    if (err != cudaSuccess)
    {
        return err;
    }
    dim3 const grid((totalQ + kTileQ - 1) / kTileQ, numKvHeads);
    common::launchWithPdlWhenEnabled("minimaxM3SelectPrefillTile", kernel, grid, dim3(wide ? kWideThreads : kThreads),
        smemBytes, stream, scores, headStride, blockStride, nValidBlocks, out, outQueryStride, outHeadStride, numBlocks,
        totalQ, initBlocks, localBlocks);
    return cudaGetLastError();
}

} // namespace kernels::minimaxM31Sm103

TRTLLM_NAMESPACE_END
