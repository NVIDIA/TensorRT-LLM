/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// Adapted from SGLang's Apache-2.0 licensed load_cache_to_device_buffer_kernel:
// https://github.com/sgl-project/sglang/blob/0163f8ff74c3d8f32e364e2b812d24dc715af039/
// python/sglang/kernels/jit/csrc/kvcacheio/hisparse.cuh
// This port contains the CUDA linear-cache path, without the TVM launcher,
// DSv4 page layout, miss-plan recording, or SkipIO profiling specialization.

#include "tensorrt_llm/common/config.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::hisparse
{

namespace detail
{

constexpr int kWarpSize = 32;
constexpr unsigned int kFullWarpMask = 0xFFFFFFFFU;
constexpr int32_t kTokenHit = -1;
constexpr int32_t kHashEmpty = -1;

__device__ __forceinline__ int hashSlot(int32_t key, int hashSize)
{
    constexpr uint32_t kHashMultiplier = 2654435761U;
    return (static_cast<uint32_t>(key) * kHashMultiplier) % static_cast<uint32_t>(hashSize);
}

// One warp copies one item. Retain the upstream non-caching loads and L2-cached
// stores, but only issue vector loads at their required 16-byte alignment.
// Scalar and byte tails also cover items whose stride is not a multiple of 16.
__device__ __forceinline__ void transferItemWarp(
    int laneId, void const* srcAddress, void* dstAddress, int64_t itemSizeBytes)
{
    auto const* src = static_cast<char const*>(srcAddress);
    auto* dst = static_cast<char*>(dstAddress);
    auto const alignment = reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst);
    int64_t copiedBytes = 0;
    constexpr int64_t kVectorBytes = 16;
    constexpr int64_t kWordBytes = sizeof(uint64_t);
    if ((alignment % kVectorBytes) == 0)
    {
        int64_t const pairs = itemSizeBytes / kVectorBytes;
        for (int64_t i = laneId; i < pairs; i += kWarpSize)
        {
            uint64_t lo, hi;
            auto const* source = src + i * kVectorBytes;
            auto* destination = dst + i * kVectorBytes;
            asm volatile("ld.global.nc.v2.b64 {%0,%1},[%2];" : "=l"(lo), "=l"(hi) : "l"(source) : "memory");
            asm volatile("st.global.cg.v2.b64 [%0],{%1,%2};" ::"l"(destination), "l"(lo), "l"(hi) : "memory");
        }
        copiedBytes = pairs * kVectorBytes;
    }
    if ((alignment % kWordBytes) == 0)
    {
        int64_t const words = (itemSizeBytes - copiedBytes) / kWordBytes;
        for (int64_t i = laneId; i < words; i += kWarpSize)
        {
            uint64_t value;
            auto const* source = src + copiedBytes + i * kWordBytes;
            auto* destination = dst + copiedBytes + i * kWordBytes;
            asm volatile("ld.global.nc.b64 %0,[%1];" : "=l"(value) : "l"(source) : "memory");
            asm volatile("st.global.cg.b64 [%0],%1;" ::"l"(destination), "l"(value) : "memory");
        }
        copiedBytes += words * kWordBytes;
    }
    for (int64_t i = copiedBytes + laneId; i < itemSizeBytes; i += kWarpSize)
    {
        dst[i] = src[i];
    }
}

__device__ __forceinline__ int warpInclusiveScan(int32_t* data, int laneId, int offset, int end, int accumulator)
{
    int const index = offset + laneId;
    int value = index < end ? data[index] : 0;
#pragma unroll
    for (int delta = 1; delta < kWarpSize; delta *= 2)
    {
        int const other = __shfl_up_sync(kFullWarpMask, value, delta);
        if (laneId >= delta)
        {
            value += other;
        }
    }
    value += accumulator;
    if (index < end)
    {
        data[index] = value;
    }
    return __shfl_sync(kFullWarpMask, value, kWarpSize - 1);
}

} // namespace detail

//! Dynamic shared memory required by loadCacheToDeviceBufferKernel.
template <int NumTopK, int HotBufferSize>
struct SmemLayout
{
    static_assert(NumTopK > 0 && NumTopK <= HotBufferSize);
    static_assert(HotBufferSize <= static_cast<int>(std::numeric_limits<int16_t>::max()) + 1);
    static constexpr int kHashSize = NumTopK * 2;
    static constexpr int kNumBufferChunks = (HotBufferSize + detail::kWarpSize - 1) / detail::kWarpSize;
    // Top-k scratch, two prefix sums, hash keys, and hit counters, then LRU slots and hash values.
    static constexpr int kTotalInt32 = NumTopK + 2 * (kNumBufferChunks + 1) + kHashSize + 2;
    static constexpr int kTotalInt16 = HotBufferSize + kHashSize;
    static constexpr size_t kBytes = kTotalInt32 * sizeof(int32_t) + kTotalInt16 * sizeof(int16_t);
};

//! Resolve sparse attention indices, update LRU state, and refetch missing KV items in one block per request.
//!
//! Launch with BlockSize threads and SmemLayout<NumTopK, HotBufferSize>::kBytes dynamic shared memory.
//! The caller must opt in with cudaFuncSetAttribute when this exceeds the default shared-memory limit.
//! All metadata pointers are device pointers; host caches must be GPU-accessible aliases of mapped pinned memory.
//! Host and device KV caches use the same linear itemSizeBytes stride. IsMla uses only K; V may then be null.
//!
//! topKTokens and topKDeviceLocs have batch rows of NumTopK entries and independent row strides (in elements).
//! seqLens and reqPoolIndices have batch entries, independently int32_t or int64_t; lengths must fit int32_t.
//! numRealReqs is a device scalar. Padded rows only write -1 outputs, without reading request metadata.
//! Request IDs in the active batch must be distinct; their physical device cache locations must not overlap.
//! deviceBufferTokens and deviceBufferLocs share bufferStride rows indexed by request ID, containing at least
//! HotBufferSize + 1 entries. Slot HotBufferSize holds the newest token and is excluded from LRU replacement.
//! hostCacheLocs has hostStride entries per request, mapping logical token IDs to physical host items.
//! lruSlots has lruStride entries per request; its first HotBufferSize entries must be a permutation of the
//! hot slots ordered LRU to MRU. Resident token IDs must be unique; -1 denotes an empty slot.
//!
//! For seqLen <= HotBufferSize, all tokens are resident in token order and only min(seqLen, NumTopK) selections
//! are read; -1 selections and remaining outputs are invalid. No cache bytes or LRU state are changed.
//! Longer sequences require NumTopK distinct token IDs in [0, seqLen). The newest token is already resident
//! in the reserved slot and must not also appear in a hot slot. Misses replace the oldest non-hit slots in
//! top-k order. The new LRU order is remaining non-hits, refetched misses, then hits in their previous LRU order.
template <int BlockSize, int NumTopK, int HotBufferSize, bool IsMla, typename SeqLensT, typename ReqPoolIndicesT>
__global__ void loadCacheToDeviceBufferKernel(int32_t const* __restrict__ topKTokens,
    int32_t* __restrict__ deviceBufferTokens, int64_t const* __restrict__ hostCacheLocs,
    int32_t const* __restrict__ deviceBufferLocs, void const* __restrict__ hostCacheK,
    void const* __restrict__ hostCacheV, void* __restrict__ deviceBufferK, void* __restrict__ deviceBufferV,
    int32_t* __restrict__ topKDeviceLocs, ReqPoolIndicesT const* __restrict__ reqPoolIndices,
    SeqLensT const* __restrict__ seqLens, int16_t* __restrict__ lruSlots, int32_t const* __restrict__ numRealReqs,
    int64_t bufferStride, int64_t hostStride, int64_t lruStride, int64_t topKTokensStride, int64_t topKDeviceLocsStride,
    int64_t itemSizeBytes)
{
    using namespace detail;
    static_assert(BlockSize >= kWarpSize && BlockSize <= 1024 && BlockSize % kWarpSize == 0);
    static_assert(std::is_same_v<SeqLensT, int32_t> || std::is_same_v<SeqLensT, int64_t>);
    static_assert(std::is_same_v<ReqPoolIndicesT, int32_t> || std::is_same_v<ReqPoolIndicesT, int64_t>);
    using Layout = SmemLayout<NumTopK, HotBufferSize>;
    constexpr int kNumWarps = BlockSize / kWarpSize;
    constexpr int kNumTokenChunks = (NumTopK + kWarpSize - 1) / kWarpSize;
    constexpr int kNumBufferChunks = Layout::kNumBufferChunks;
    constexpr int kHashSize = Layout::kHashSize;

    int const batchIndex = blockIdx.x;
    int const tid = threadIdx.x;
    auto* reqTopKDeviceLocs = topKDeviceLocs + batchIndex * topKDeviceLocsStride;
    if (batchIndex >= numRealReqs[0])
    {
        for (int i = tid; i < NumTopK; i += BlockSize)
        {
            reqTopKDeviceLocs[i] = -1;
        }
        return;
    }

    int const warpId = tid / kWarpSize;
    int const laneId = tid % kWarpSize;
    unsigned int const lanesBefore = (1U << laneId) - 1U;
    int64_t const requestId = reqPoolIndices[batchIndex];
    int64_t const seqLen = seqLens[batchIndex];
    auto const* reqTopKTokens = topKTokens + batchIndex * topKTokensStride;
    auto* reqDeviceBufferTokens = deviceBufferTokens + requestId * bufferStride;
    auto const* reqDeviceBufferLocs = deviceBufferLocs + requestId * bufferStride;
    auto const* reqHostCacheLocs = hostCacheLocs + requestId * hostStride;
    auto* reqLruSlots = lruSlots + requestId * lruStride;

    if (seqLen <= HotBufferSize)
    {
        int const count = seqLen < NumTopK ? static_cast<int>(seqLen) : NumTopK;
        for (int i = tid; i < NumTopK; i += BlockSize)
        {
            int32_t location = -1;
            if (i < count)
            {
                int32_t const token = reqTopKTokens[i];
                if (token >= 0 && token < seqLen)
                {
                    location = reqDeviceBufferLocs[token];
                }
            }
            reqTopKDeviceLocs[i] = location;
        }
        return;
    }

    extern __shared__ int32_t smem[];
    auto* selectedTokens = smem;
    auto* chunkOffsets = selectedTokens + NumTopK;
    auto* evictChunkOffsets = chunkOffsets + kNumBufferChunks + 1;
    auto* hashKeys = evictChunkOffsets + kNumBufferChunks + 1;
    auto& totalHits = hashKeys[kHashSize];
    auto& newestHit = hashKeys[kHashSize + 1];
    auto* orderedSlots = reinterpret_cast<int16_t*>(smem + Layout::kTotalInt32);
    auto* hashValues = orderedSlots + HotBufferSize;
    if (tid == 0)
    {
        totalHits = 0;
        newestHit = 0;
    }
    for (int i = tid; i < kHashSize; i += BlockSize)
    {
        hashKeys[i] = kHashEmpty;
    }
    for (int i = tid; i <= kNumBufferChunks; i += BlockSize)
    {
        chunkOffsets[i] = 0;
        evictChunkOffsets[i] = 0;
    }
    __syncthreads();

    // Hash the selections, then probe each resident token in LRU order.
    for (int i = tid; i < NumTopK; i += BlockSize)
    {
        int32_t const token = reqTopKTokens[i];
        if (token == seqLen - 1)
        {
            selectedTokens[i] = kTokenHit;
            reqTopKDeviceLocs[i] = reqDeviceBufferLocs[HotBufferSize];
            newestHit = 1;
        }
        else
        {
            int slot = hashSlot(token, kHashSize);
            while (atomicCAS(&hashKeys[slot], kHashEmpty, token) != kHashEmpty)
            {
                slot = (slot + 1) % kHashSize;
            }
            hashValues[slot] = static_cast<int16_t>(i);
            selectedTokens[i] = token;
        }
    }
    __syncthreads();

    int hitAccumulator = 0;
    int evictAccumulator = 0;
    constexpr int kBufferIterations = (kNumBufferChunks + kNumWarps - 1) / kNumWarps;
    for (int iter = 0; iter < kBufferIterations; ++iter)
    {
        int const chunk = warpId + iter * kNumWarps;
        int const lruIndex = chunk * kWarpSize + laneId;
        bool const validSlot = lruIndex < HotBufferSize;
        int16_t const bufferSlot = validSlot ? reqLruSlots[lruIndex] : -1;
        int32_t const token = validSlot ? reqDeviceBufferTokens[bufferSlot] : -1;
        int selectedIndex = -1;
        if (token >= 0)
        {
            int slot = hashSlot(token, kHashSize);
            while (hashKeys[slot] != kHashEmpty)
            {
                if (hashKeys[slot] == token)
                {
                    selectedIndex = hashValues[slot];
                    break;
                }
                slot = (slot + 1) % kHashSize;
            }
        }
        bool const hit = selectedIndex >= 0;
        bool const evictable = validSlot && !hit;
        if (hit)
        {
            selectedTokens[selectedIndex] = kTokenHit;
            reqTopKDeviceLocs[selectedIndex] = reqDeviceBufferLocs[bufferSlot];
        }
        auto const hitMask = __ballot_sync(kFullWarpMask, hit);
        auto const evictMask = __ballot_sync(kFullWarpMask, evictable);
        if (laneId == 0 && chunk < kNumBufferChunks)
        {
            chunkOffsets[chunk + 1] = __popc(hitMask);
            evictChunkOffsets[chunk + 1] = __popc(evictMask);
        }
        __syncthreads();
        if (warpId == 0)
        {
            int const offset = iter * kNumWarps + 1;
            // Only scan this iteration's counts. A full warp window would overwrite
            // future counts when BlockSize < 1024 and multiple iterations are needed.
            int const end = min(offset + kNumWarps, kNumBufferChunks + 1);
            hitAccumulator = warpInclusiveScan(chunkOffsets, laneId, offset, end, hitAccumulator);
            evictAccumulator = warpInclusiveScan(evictChunkOffsets, laneId, offset, end, evictAccumulator);
            if (tid == 0)
            {
                totalHits = hitAccumulator;
            }
        }
        __syncthreads();
        if (hit)
        {
            orderedSlots[chunkOffsets[chunk] + __popc(hitMask & lanesBefore)] = bufferSlot;
        }
        if (evictable)
        {
            orderedSlots[HotBufferSize - 1 - evictChunkOffsets[chunk] - __popc(evictMask & lanesBefore)] = bufferSlot;
        }
    }
    __syncthreads();

    for (int i = tid; i <= kNumTokenChunks; i += BlockSize)
    {
        chunkOffsets[i] = 0;
    }
    __syncthreads();

    // Compact misses in top-k order and assign the oldest non-hit slots.
    int missAccumulator = 0;
    constexpr int kTokenIterations = (kNumTokenChunks + kNumWarps - 1) / kNumWarps;
    for (int iter = 0; iter < kTokenIterations; ++iter)
    {
        int const chunk = warpId + iter * kNumWarps;
        int const selectedIndex = chunk * kWarpSize + laneId;
        int32_t const token = selectedIndex < NumTopK ? selectedTokens[selectedIndex] : kTokenHit;
        bool const miss = token != kTokenHit;
        auto const missMask = __ballot_sync(kFullWarpMask, miss);
        if (laneId == 0 && chunk < kNumTokenChunks)
        {
            chunkOffsets[chunk + 1] = __popc(missMask);
        }
        __syncthreads();
        if (warpId == 0)
        {
            int const offset = iter * kNumWarps + 1;
            int const end = min(offset + kNumWarps, kNumTokenChunks + 1);
            missAccumulator = warpInclusiveScan(chunkOffsets, laneId, offset, end, missAccumulator);
        }
        __syncthreads();
        if (miss)
        {
            int const missIndex = chunkOffsets[chunk] + __popc(missMask & lanesBefore);
            int16_t const evictSlot = orderedSlots[HotBufferSize - 1 - missIndex];
            // Every thread has read its token before this compacted write. The
            // destination cannot overrun selections read by a later iteration.
            selectedTokens[missIndex] = token;
            reqTopKDeviceLocs[selectedIndex] = reqDeviceBufferLocs[evictSlot];
            reqDeviceBufferTokens[evictSlot] = token;
        }
    }
    __syncthreads();

    int const totalMisses = NumTopK - totalHits - newestHit;
    int const totalEvictable = HotBufferSize - totalHits;
    for (int i = tid; i < HotBufferSize; i += BlockSize)
    {
        if (i < totalMisses)
        {
            reqLruSlots[totalEvictable - totalMisses + i] = orderedSlots[HotBufferSize - 1 - i];
        }
        else if (i < totalEvictable)
        {
            reqLruSlots[i - totalMisses] = orderedSlots[HotBufferSize - 1 - i];
        }
        else
        {
            reqLruSlots[i] = orderedSlots[i - totalEvictable];
        }
    }

    for (int missIndex = warpId; missIndex < totalMisses; missIndex += kNumWarps)
    {
        int32_t const token = selectedTokens[missIndex];
        int16_t const evictSlot = orderedSlots[HotBufferSize - 1 - missIndex];
        int64_t const sourceOffset = reqHostCacheLocs[token] * itemSizeBytes;
        int64_t const destinationOffset = static_cast<int64_t>(reqDeviceBufferLocs[evictSlot]) * itemSizeBytes;
        transferItemWarp(laneId, static_cast<char const*>(hostCacheK) + sourceOffset,
            static_cast<char*>(deviceBufferK) + destinationOffset, itemSizeBytes);
        if constexpr (!IsMla)
        {
            transferItemWarp(laneId, static_cast<char const*>(hostCacheV) + sourceOffset,
                static_cast<char*>(deviceBufferV) + destinationOffset, itemSizeBytes);
        }
    }
}

} // namespace kernels::hisparse

TRTLLM_NAMESPACE_END
