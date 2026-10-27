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

// Top-16 block select for MiniMax-M3 decode rows of 129 to 9216 blocks.
//
// One 1024-thread CTA per (query, KV head) row holds the row's ordered score keys in registers (2, 4 or 9 per
// thread, by the row's valid count). The weakest of the 16 maxima of the interleaved groups `block % 16` is a lower
// bound on the 16th-largest key, so the top 16 is among the keys at or above it; those are compacted to shared
// memory and ranked exactly by (key, block). More than 1024 keys tied at the bound (degenerate rows) rank every
// candidate against the whole row (re-read). Rows of at most 64 candidates (the common case) skip the all-thread rank:
// warp 0 sorts them in registers (a 64-key bitonic network), keeps the 16 smallest (key, block) pairs and emits them,
// and the other warps exit. The row's valid count is a step input and is read before the grid dependency wait.

#include "tensorrt_llm/kernels/minimaxM3Sm107Tp4/selectLong2.h"

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3Sm107Tp4
{

namespace
{

constexpr float kInitScore = 1.0e30f;
constexpr float kLocalScore = 1.0e29f;
constexpr uint32_t kNegInfKey = 0xff800000u;
constexpr int kItems = 9;           // keys per thread: rows up to 9216 blocks (1M tokens + draft tokens)
constexpr int kWarpCandidates = 64; // candidates warp 0 ranks alone (2 per lane)

// Descending order key: a smaller key is a larger score (NaN above +inf), as in the reference op.
__device__ __forceinline__ uint32_t orderedKey(float x)
{
    uint32_t const b = __float_as_uint(x);
    return (b & 0x80000000u) ? b : (~b & 0x7fffffffu);
}

__device__ __forceinline__ float applyForcing(float raw, int block, int initBlocks, long long localStart)
{
    if (block >= localStart)
        return kLocalScore;
    if (block < initBlocks)
        return kInitScore;
    return raw;
}

struct SelectSmem
{
    uint32_t groupMin[16];
    unsigned long long cand[1024];
    int numCand;
    unsigned long long picked[16];
};

// One row with up to kItems * 1024 valid blocks; the caller picks kItems from the row's valid count.
// Returns -1 when sm.picked holds the row's picks, else the candidate count (at most kWarpCandidates, compacted to
// sm.cand and not yet ranked). The value is uniform across the CTA.
template <int kItems, bool kFast>
__device__ __forceinline__ int selectRow(SelectSmem& sm, float const* rowBase, long long blockStride, int valid,
    int initBlocks, long long localStart, bool loadedTrigger)
{
    int const tid = threadIdx.x, lane = tid & 31;
    uint32_t key[kItems];
    uint32_t localMin = 0xffffffffu;
#pragma unroll
    for (int s = 0; s < kItems; ++s)
    {
        int const block = s * 1024 + tid;
        key[s] = block < valid ? orderedKey(applyForcing(rowBase[block * blockStride], block, initBlocks, localStart))
                               : 0xffffffffu;
        localMin = min(localMin, key[s]);
    }
    if (valid <= 16)
    {
        if (tid < valid)
            sm.picked[tid] = (static_cast<unsigned long long>(key[0]) << 32) | static_cast<unsigned>(tid);
        if (loadedTrigger)
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        return -1;
    }
    // Group g = block % 16 = tid % 16: lanes l and l ^ 16 of every warp.
    localMin = min(localMin, __shfl_xor_sync(0xffffffffu, localMin, 16));
    if (lane < 16)
        atomicMin(&sm.groupMin[lane], localMin);
    __syncthreads();
    // Every score of the row is loaded: release the sparse decode, so its launch and its pre-wait prologue overlap
    // the rank below. This comes after this grid's own dependency wait, which the sparse decode relies on: it reads
    // q and the seq_lens / block-table row before its own wait (sparseDecodeLc2.cu), safe only while every kernel
    // between the producer and it releases its dependents after its own wait.
    if (loadedTrigger)
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    uint32_t bound = 0;
#pragma unroll
    for (int g = 0; g < 16; ++g)
        bound = max(bound, sm.groupMin[g]);
#pragma unroll
    for (int s = 0; s < kItems; ++s)
    {
        int const block = s * 1024 + tid;
        bool const isCand = block < valid && key[s] <= bound;
        unsigned const ballot = __ballot_sync(0xffffffffu, isCand);
        int base = 0;
        if (lane == 0 && ballot)
            base = atomicAdd(&sm.numCand, __popc(ballot));
        base = __shfl_sync(0xffffffffu, base, 0);
        if (isCand)
        {
            int const i = base + __popc(ballot & ((1u << lane) - 1));
            if (i < 1024)
                sm.cand[i] = (static_cast<unsigned long long>(key[s]) << 32) | static_cast<unsigned>(block);
        }
    }
    __syncthreads();
    int const n = sm.numCand;
    if (kFast && n <= kWarpCandidates)
        return n;
    if (n <= 1024)
    {
        for (int i = tid; i < n; i += 1024)
        {
            unsigned long long const mine = sm.cand[i];
            int rank = 0;
            for (int j = 0; j < n; ++j)
                rank += sm.cand[j] < mine;
            if (rank < 16)
                sm.picked[rank] = mine;
        }
        return -1;
    }
    // More than 1024 keys tied at the bound (degenerate rows): rank every candidate against the whole row.
#pragma unroll
    for (int s = 0; s < kItems; ++s)
    {
        int const block = s * 1024 + tid;
        if (block < valid && key[s] <= bound)
        {
            int rank = 0;
            for (int j = 0; j < valid; ++j)
            {
                uint32_t const other = orderedKey(applyForcing(rowBase[j * blockStride], j, initBlocks, localStart));
                rank += other < key[s] || (other == key[s] && j < block);
            }
            if (rank < 16)
                sm.picked[rank] = (static_cast<unsigned long long>(key[s]) << 32) | static_cast<unsigned>(block);
        }
    }
    return -1;
}

// Warp 0: the n <= kWarpCandidates candidates of sm.cand sorted ascending as 64 keys (element r * 32 + lane in
// register r); returns this lane's element (lanes 0..15 then hold the 16 smallest).
__device__ __forceinline__ unsigned long long warpSmallest(SelectSmem const& sm, int n, int lane)
{
    unsigned long long a = lane < n ? sm.cand[lane] : ~0ull;
    unsigned long long b = lane + 32 < n ? sm.cand[lane + 32] : ~0ull;
#pragma unroll
    for (int size = 2; size <= 64; size <<= 1)
    {
#pragma unroll
        for (int stride = size >> 1; stride > 0; stride >>= 1)
        {
            if (stride == 32)
            {
                unsigned long long const lo = min(a, b), hi = max(a, b);
                a = lo;
                b = hi;
            }
            else
            {
                unsigned long long const oa = __shfl_xor_sync(0xffffffffu, a, stride);
                unsigned long long const ob = __shfl_xor_sync(0xffffffffu, b, stride);
                bool const lower = (lane & stride) == 0;
                bool const ascA = (lane & size) == 0, ascB = ((lane + 32) & size) == 0;
                a = (lower == ascA) ? min(a, oa) : max(a, oa);
                b = (lower == ascB) ? min(b, ob) : max(b, ob);
            }
        }
    }
    return a;
}

// Warp 0: the picks (lane < 16 holds pick `lane`, ~0ull when absent) as ascending block ids, -1 for forced -inf picks
// and missing picks, padded at the tail.
__device__ __forceinline__ void emitPicks(unsigned long long c, int lane, int* outRow)
{
    uint32_t blockKey = ~0u;
    if (lane < 16 && c != ~0ull && static_cast<uint32_t>(c >> 32) != kNegInfKey)
        blockKey = static_cast<uint32_t>(c & 0xffffffffu);
#pragma unroll
    for (int size = 2; size <= 32; size <<= 1)
    {
#pragma unroll
        for (int stride = size >> 1; stride > 0; stride >>= 1)
        {
            uint32_t const other = __shfl_xor_sync(0xffffffffu, blockKey, stride);
            bool const ascending = (lane & size) == 0, lower = (lane & stride) == 0;
            blockKey = (lower == ascending) ? min(blockKey, other) : max(blockKey, other);
        }
    }
    if (lane < 16)
        outRow[lane] = blockKey == ~0u ? -1 : static_cast<int>(blockKey);
}

// kFast = false is the select before the warp-0 tail (dispatch.SELECT_LONG_SL3 = False, the kill value).
template <bool kFast>
__global__ void __launch_bounds__(1024) m3_select_long2_kernel(float const* __restrict__ scores, long long headStride,
    long long blockStride, long long queryStride, int const* __restrict__ nValid, int* __restrict__ out,
    long long outQueryStride, long long outHeadStride, int numKvHeads, int numBlocks, int initBlocks, int localBlocks,
    bool earlyTrigger, bool loadedTrigger)
{
    __shared__ SelectSmem sm;
    int const row = blockIdx.x, query = row / numKvHeads, head = row % numKvHeads;
    int const tid = threadIdx.x, lane = tid & 31;
    if (tid < 16)
    {
        sm.groupMin[tid] = 0xffffffffu;
        sm.picked[tid] = ~0ull;
    }
    if (tid == 0)
        sm.numCand = 0;
    // With kFast the row's valid count is read before this grid's dependency wait: a step input (written between
    // replays or at the step head), final before the wait, as for the sparse decode's pre-wait seq_lens / block-table
    // reads.
    int raw = 0;
    if (kFast)
    {
        raw = nValid[query];
        __syncthreads();
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize();
    // The sparse decode that follows stages its table-independent inputs before its own grid wait; past this grid's
    // wait those inputs are final, so it may launch now rather than at this grid's exit.
    if (earlyTrigger)
        cudaTriggerProgrammaticLaunchCompletion();
#endif
    if (!kFast)
    {
        raw = nValid[query];
        __syncthreads();
    }
    int const valid = max(0, min(raw, numBlocks));
    long long const localStart = max(static_cast<long long>(raw) - localBlocks, 0LL);
    float const* rowBase = scores + head * headStride + query * queryStride;
    int n;
    if (valid <= 2048)
        n = selectRow<2, kFast>(sm, rowBase, blockStride, valid, initBlocks, localStart, loadedTrigger);
    else if (valid <= 4096)
        n = selectRow<4, kFast>(sm, rowBase, blockStride, valid, initBlocks, localStart, loadedTrigger);
    else
        n = selectRow<kItems, kFast>(sm, rowBase, blockStride, valid, initBlocks, localStart, loadedTrigger);
    int* const outRow = out + query * outQueryStride + head * outHeadStride;
    if (n >= 0)
    {
        if (tid >= 32)
            return;
        emitPicks(warpSmallest(sm, n, lane), lane, outRow);
        return;
    }
    __syncthreads();
    if (tid >= 32)
        return;
    emitPicks(lane < 16 ? sm.picked[lane] : ~0ull, lane, outRow);
}

} // namespace

cudaError_t invokeSelectLong2(float const* scores, int64_t headStride, int64_t blockStride, int64_t queryStride,
    int32_t const* nValidBlocks, int32_t* out, int64_t outQueryStride, int64_t outHeadStride, int32_t numKvHeads,
    int32_t numBlocks, int32_t totalQ, int32_t initBlocks, int32_t localBlocks, bool usePdl, bool earlyTrigger,
    int32_t triggerPhase, bool fastTail, cudaStream_t stream)
{
    if (numBlocks > kSelectLong2MaxBlocks || triggerPhase < 0 || triggerPhase > 1)
    {
        return cudaErrorInvalidValue;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(totalQ * numKvHeads);
    cfg.blockDim = dim3(1024);
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = usePdl ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, fastTail ? m3_select_long2_kernel<true> : m3_select_long2_kernel<false>, scores,
        static_cast<long long>(headStride), static_cast<long long>(blockStride), static_cast<long long>(queryStride),
        nValidBlocks, out, static_cast<long long>(outQueryStride), static_cast<long long>(outHeadStride), numKvHeads,
        numBlocks, initBlocks, localBlocks, earlyTrigger, !earlyTrigger && triggerPhase == 1);
}

} // namespace minimaxM3Sm107Tp4
} // namespace kernels

TRTLLM_NAMESPACE_END
