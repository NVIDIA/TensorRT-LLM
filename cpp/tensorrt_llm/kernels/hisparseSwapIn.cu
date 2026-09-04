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

// This kernel is a block/page-granular port of SGLang HiSparse's
// load_cache_to_device_buffer_kernel (kvcacheio/hisparse.cuh). The hit-detection,
// LRU ordering and miss-eviction logic follow the original closely; the compile-time
// template parameters (NUM_TOP_K / HOT_BUFFER_SIZE / BLOCK_SIZE) are lowered to runtime
// arguments with dynamic shared memory, and the per-item copy moves a whole KV page for
// separate K and V pools instead of a single MLA token.

#include "tensorrt_llm/kernels/hisparseSwapIn.h"

#include "tensorrt_llm/common/cudaUtils.h"

#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

constexpr int kWarpSize = 32;
constexpr uint32_t kFullWarpMask = 0xFFFFFFFFu;
constexpr int32_t kTokenHit = static_cast<int32_t>(0xFFFFFFFF); // -1 sentinel: "this selected block is resident"
constexpr int32_t kHashEmpty = -1;

// Knuth multiplicative hash for the open-addressing table of size hashSize.
__device__ __forceinline__ int hashSlot(int32_t key, int hashSize)
{
    return static_cast<int>((static_cast<uint32_t>(key) * 2654435761u) % static_cast<uint32_t>(hashSize));
}

// Copy one KV page (itemSizeBytes) host->device with a single warp. 128-bit bulk
// transfer via paired 64-bit loads; non-temporal streaming load, cached store for the
// device buffer the attention kernel reads next. itemSizeBytes must be a multiple of 8.
__device__ __forceinline__ void transferItemWarp(
    int32_t laneId, void const* srcAddr, void* dstAddr, int64_t itemSizeBytes)
{
    int const totalPairs = static_cast<int>(itemSizeBytes / 16);
    {
        uint64_t const* __restrict__ src = static_cast<uint64_t const*>(srcAddr);
        uint64_t* __restrict__ dst = static_cast<uint64_t*>(dstAddr);
        for (int j = laneId; j < totalPairs; j += kWarpSize)
        {
            uint64_t lo, hi;
            uint64_t const* s = src + j * 2;
            asm volatile("ld.global.nc.v2.b64 {%0,%1},[%2];" : "=l"(lo), "=l"(hi) : "l"(s) : "memory");
            uint64_t* d = dst + j * 2;
            asm volatile("st.global.cg.v2.b64 [%0],{%1,%2};" ::"l"(d), "l"(lo), "l"(hi) : "memory");
        }
    }

    // Tail: 64-bit for the remaining 8-byte chunk when itemSizeBytes is not a multiple of 16.
    int const tail8B = static_cast<int>((itemSizeBytes - static_cast<int64_t>(totalPairs) * 16) / 8);
    if (tail8B > 0 && laneId < tail8B)
    {
        uint64_t const* __restrict__ src8
            = reinterpret_cast<uint64_t const*>(static_cast<char const*>(srcAddr) + totalPairs * 16);
        uint64_t* __restrict__ dst8 = reinterpret_cast<uint64_t*>(static_cast<char*>(dstAddr) + totalPairs * 16);
        uint64_t tmp;
        asm volatile("ld.global.nc.b64 %0,[%1];" : "=l"(tmp) : "l"(src8 + laneId) : "memory");
        asm volatile("st.global.cg.b64 [%0],%1;" ::"l"(dst8 + laneId), "l"(tmp) : "memory");
    }
}

__device__ __forceinline__ void copyMissItem(int32_t laneId, char const* __restrict__ hostCacheK,
    char const* __restrict__ hostCacheV, char* __restrict__ deviceBufferK, char* __restrict__ deviceBufferV,
    int64_t srcLoc, int64_t dstLoc, int64_t itemSizeBytes)
{
    char const* srcK = hostCacheK + srcLoc * itemSizeBytes;
    char* dstK = deviceBufferK + dstLoc * itemSizeBytes;
    transferItemWarp(laneId, srcK, dstK, itemSizeBytes);

    char const* srcV = hostCacheV + srcLoc * itemSizeBytes;
    char* dstV = deviceBufferV + dstLoc * itemSizeBytes;
    transferItemWarp(laneId, srcV, dstV, itemSizeBytes);
}

__device__ __forceinline__ int warpInclusiveScan(int* sData, int laneId, int offset, int count, int accumulator)
{
    int idx = laneId + offset;
    int val = (idx < count) ? sData[idx] : 0;
#pragma unroll
    for (int i = 1; i < kWarpSize; i *= 2)
    {
        int n = __shfl_up_sync(kFullWarpMask, val, i);
        if (laneId >= i)
        {
            val += n;
        }
    }
    val += accumulator;
    if (idx < count)
    {
        sData[idx] = val;
    }
    accumulator = __shfl_sync(kFullWarpMask, val, kWarpSize - 1);
    return accumulator;
}

// One CUDA block per request. See invokeHiSparseSwapInBlocks for the tensor contract.
template <typename SeqLensT, typename ReqPoolIndicesT>
__global__ void hiSparseSwapInBlocksKernel(int32_t const* __restrict__ topKTokens,
    int32_t* __restrict__ deviceBufferTokens, int64_t const* __restrict__ hostCacheLocs,
    int32_t const* __restrict__ deviceBufferLocs, char const* __restrict__ hostCacheK,
    char const* __restrict__ hostCacheV, char* __restrict__ deviceBufferK, char* __restrict__ deviceBufferV,
    int32_t* __restrict__ topKDeviceLocs, ReqPoolIndicesT const* __restrict__ reqPoolIndices,
    SeqLensT const* __restrict__ seqLens, int16_t* __restrict__ lruSlots, int32_t const* __restrict__ numRealReqs,
    int32_t numTopK, int32_t hotBufferSize, int64_t bufferStride0, int64_t hostStride, int64_t lruSlotStride0,
    int64_t topKTokensStride, int64_t topKDeviceLocsStride, int64_t itemSizeBytes)
{
    int const blockSize = static_cast<int>(blockDim.x);
    int const numWarps = blockSize / kWarpSize;
    int const numTokenChunks = (numTopK + kWarpSize - 1) / kWarpSize;
    int const numBufferChunks = (hotBufferSize + kWarpSize - 1) / kWarpSize;

    int const bid = blockIdx.x;
    int const tid = threadIdx.x;
    int32_t* reqTopKDeviceLocs = topKDeviceLocs + static_cast<int64_t>(bid) * topKDeviceLocsStride;

    // CUDA graph pads the batch to a captured size. Keep padded output rows invalid.
    if (bid >= numRealReqs[0])
    {
        for (int i = tid; i < numTopK; i += blockSize)
        {
            reqTopKDeviceLocs[i] = -1;
        }
        return;
    }

    int const warpId = tid / kWarpSize;
    int const laneId = tid % kWarpSize;
    uint32_t const lanesBefore = (1u << laneId) - 1u;

    int64_t const rid = static_cast<int64_t>(reqPoolIndices[bid]);
    int64_t const seqLen = static_cast<int64_t>(seqLens[bid]);

    int32_t const* reqTopKTokens = topKTokens + static_cast<int64_t>(bid) * topKTokensStride;
    int64_t const bufferOffset = rid * bufferStride0;
    int32_t* reqDeviceBufferTokens = deviceBufferTokens + bufferOffset;
    int32_t const* reqDeviceBufferLocs = deviceBufferLocs + bufferOffset;
    int64_t const* reqHostCacheLocs = hostCacheLocs + rid * hostStride;
    int16_t* reqLruSlots = lruSlots + rid * lruSlotStride0;

    // Fast path: short sequences have all blocks resident in the device buffer in order.
    if (seqLen <= hotBufferSize)
    {
        int const count = (seqLen < numTopK) ? static_cast<int>(seqLen) : numTopK;
        for (int i = tid; i < numTopK; i += blockSize)
        {
            int32_t deviceLoc = -1;
            if (i < count)
            {
                int32_t tokenPos = reqTopKTokens[i];
                if (tokenPos >= 0)
                {
                    deviceLoc = reqDeviceBufferLocs[tokenPos];
                }
            }
            reqTopKDeviceLocs[i] = deviceLoc;
        }
        return;
    }

    // Dynamic shared memory layout: int32 arrays first, then int16 arrays.
    extern __shared__ char smemRaw[];
    int const hashSize = numTopK * 2;

    int32_t* sTopKTokens = reinterpret_cast<int32_t*>(smemRaw);
    int32_t* sChunkOffset = sTopKTokens + numTopK;
    int32_t* sEvictChunkOffset = sChunkOffset + (numBufferChunks + 1);
    int32_t* sHashKeys = sEvictChunkOffset + (numBufferChunks + 1);
    int32_t* sTotalHits = sHashKeys + hashSize;
    int32_t* sNewestHit = sTotalHits + 1;
    int const totalInt32 = numTopK + 2 * (numBufferChunks + 1) + hashSize + 2;

    int16_t* smemI16 = reinterpret_cast<int16_t*>(smemRaw + static_cast<size_t>(totalInt32) * sizeof(int32_t));
    int16_t* sLruSlotsOut = smemI16;
    int16_t* sHashVals = sLruSlotsOut + hotBufferSize;

    if (tid == 0)
    {
        sTotalHits[0] = 0;
        sNewestHit[0] = 0;
    }
    for (int i = tid; i < hashSize; i += blockSize)
    {
        sHashKeys[i] = kHashEmpty;
    }
    for (int i = tid; i < numBufferChunks + 1; i += blockSize)
    {
        sChunkOffset[i] = 0;
        sEvictChunkOffset[i] = 0;
    }
    __syncthreads();

    int const newestSlot = hotBufferSize;
    int32_t const newestToken = static_cast<int32_t>(seqLen - 1);

    // Insert selected blocks into the shared-memory hash table.
    for (int i = tid; i < numTopK; i += blockSize)
    {
        int32_t tokenIdx = reqTopKTokens[i];
        if (tokenIdx == newestToken)
        {
            // The newest (current) block is bound to the reserved newest slot and marked a hit.
            sTopKTokens[i] = kTokenHit;
            reqTopKDeviceLocs[i] = reqDeviceBufferLocs[newestSlot];
            sNewestHit[0] = 1;
        }
        else
        {
            int slot = hashSlot(tokenIdx, hashSize);
            while (true)
            {
                int32_t old = atomicCAS(&sHashKeys[slot], kHashEmpty, tokenIdx);
                if (old == kHashEmpty || old == tokenIdx)
                {
                    sHashVals[slot] = static_cast<int16_t>(i);
                    break;
                }
                slot = (slot + 1) % hashSize;
            }
            sTopKTokens[i] = tokenIdx;
        }
    }
    __syncthreads();

    int const iterationsPerWarpBuffer = (numBufferChunks + numWarps - 1) / numWarps;
    int totalHitCount = 0;
    int totalEvictCount = 0;
    for (int iter = 0; iter < iterationsPerWarpBuffer; iter++)
    {
        int chunkIdx = warpId + iter * numWarps;
        bool hasValidChunk = chunkIdx < numBufferChunks;

        int const slotIdx = chunkIdx * kWarpSize + laneId;
        bool const hasValidSlot = hasValidChunk && (slotIdx < hotBufferSize);
        int16_t const bufSlot = hasValidSlot ? reqLruSlots[slotIdx] : -1;
        int32_t myBufferToken = (bufSlot >= 0) ? reqDeviceBufferTokens[bufSlot] : -1;
        int myFoundTopKIdx = -1;
        if (myBufferToken >= 0)
        {
            int h = hashSlot(myBufferToken, hashSize);
            while (true)
            {
                int32_t k = sHashKeys[h];
                if (k == myBufferToken)
                {
                    myFoundTopKIdx = static_cast<int32_t>(sHashVals[h]);
                    break;
                }
                if (k == kHashEmpty)
                {
                    break;
                }
                h = (h + 1) % hashSize;
            }
        }
        bool isHit = myFoundTopKIdx >= 0;
        bool isEvictable = hasValidSlot && !isHit;

        if (isHit)
        {
            sTopKTokens[myFoundTopKIdx] = kTokenHit;
            reqTopKDeviceLocs[myFoundTopKIdx] = reqDeviceBufferLocs[bufSlot];
        }

        int localHitOffset = 0;
        int localEvictOffset = 0;
        if (hasValidChunk)
        {
            uint32_t const hitMask = __ballot_sync(kFullWarpMask, isHit);
            uint32_t const evictMask = __ballot_sync(kFullWarpMask, isEvictable);
            localHitOffset = __popc(hitMask & lanesBefore);
            localEvictOffset = __popc(evictMask & lanesBefore);
            if (laneId == 0)
            {
                sChunkOffset[chunkIdx + 1] = __popc(hitMask);
                sEvictChunkOffset[chunkIdx + 1] = __popc(evictMask);
            }
        }
        __syncthreads();

        if (warpId == 0)
        {
            totalHitCount
                = warpInclusiveScan(sChunkOffset, laneId, chunkIdx + 1, numBufferChunks + 1, totalHitCount);
            totalEvictCount = warpInclusiveScan(
                sEvictChunkOffset, laneId, chunkIdx + 1, numBufferChunks + 1, totalEvictCount);
            if (tid == 0)
            {
                sTotalHits[0] = totalHitCount;
            }
        }
        __syncthreads();

        // Hits grow forward from index 0.
        if (isHit)
        {
            int hitOffset = sChunkOffset[chunkIdx] + localHitOffset;
            sLruSlotsOut[hitOffset] = bufSlot;
        }
        // Evictables grow backward from hotBufferSize - 1.
        if (isEvictable)
        {
            int evictOffset = sEvictChunkOffset[chunkIdx] + localEvictOffset;
            sLruSlotsOut[hotBufferSize - 1 - evictOffset] = bufSlot;
        }
    }
    __syncthreads();

    // Reset offsets for the miss-counting phase (only numTokenChunks + 1 entries needed).
    for (int i = tid; i < numTokenChunks + 1; i += blockSize)
    {
        sChunkOffset[i] = 0;
    }
    __syncthreads();

    // Identify misses and their evictable slots.
    int totalMisses = 0;
    int const iterationsPerWarpToken = (numTokenChunks + numWarps - 1) / numWarps;
    for (int iter = 0; iter < iterationsPerWarpToken; iter++)
    {
        int chunkIdx = warpId + iter * numWarps;
        bool hasValidChunk = chunkIdx < numTokenChunks;

        int const chunkTokenStart = chunkIdx * kWarpSize;
        int const myTokenIdx = chunkTokenStart + laneId;
        bool const hasValidToken = hasValidChunk && (myTokenIdx < numTopK);

        int32_t myToken = 0;
        bool isMiss = false;
        int localMissOffset = 0;

        if (hasValidToken)
        {
            isMiss = sTopKTokens[myTokenIdx] != kTokenHit;
            if (isMiss)
            {
                myToken = sTopKTokens[myTokenIdx];
            }
        }

        if (hasValidChunk)
        {
            uint32_t const missMask = __ballot_sync(kFullWarpMask, isMiss);
            localMissOffset = __popc(missMask & lanesBefore);
            int const warpMissCount = __popc(missMask);
            if (laneId == 0)
            {
                sChunkOffset[chunkIdx + 1] = warpMissCount;
            }
        }
        __syncthreads();

        if (warpId == 0)
        {
            totalMisses = warpInclusiveScan(sChunkOffset, laneId, chunkIdx + 1, numTokenChunks + 1, totalMisses);
        }
        __syncthreads();

        if (isMiss)
        {
            int missOffset = sChunkOffset[chunkIdx] + localMissOffset;
            int16_t evictSlot = sLruSlotsOut[hotBufferSize - 1 - missOffset];
            // Reuse sTopKTokens as miss scratch: missOffset < myTokenIdx always holds
            // (hits are skipped), so compacted writes never overrun pending reads.
            sTopKTokens[missOffset] = myToken;
            reqTopKDeviceLocs[myTokenIdx] = reqDeviceBufferLocs[evictSlot];
            reqDeviceBufferTokens[evictSlot] = myToken;
        }
    }
    __syncthreads();

    totalMisses = numTopK - sTotalHits[0] - sNewestHit[0];

    // Write back LRU order: evictables at front (LRU), hits at back (MRU).
    {
        int const totalEvictable = hotBufferSize - sTotalHits[0];
        for (int i = tid; i < hotBufferSize; i += blockSize)
        {
            if (i < totalMisses)
            {
                // Misses: just loaded from host, placed right before hits.
                reqLruSlots[totalEvictable - totalMisses + i] = sLruSlotsOut[hotBufferSize - 1 - i];
            }
            else if (i < totalEvictable)
            {
                // Remaining evictables: truly stale, destination at LRU front.
                reqLruSlots[i - totalMisses] = sLruSlotsOut[hotBufferSize - 1 - i];
            }
            else
            {
                // Hits: source at forward end, destination at MRU back.
                reqLruSlots[i] = sLruSlotsOut[i - totalEvictable];
            }
        }
    }

    // Each warp copies one miss page (K then V) host->device.
    for (int missIdx = warpId; missIdx < totalMisses; missIdx += numWarps)
    {
        int32_t const missToken = sTopKTokens[missIdx];
        int16_t const evictSlot = sLruSlotsOut[hotBufferSize - 1 - missIdx];

        int64_t const srcLoc = reqHostCacheLocs[missToken];
        int64_t const dstLoc = static_cast<int64_t>(reqDeviceBufferLocs[evictSlot]);

        copyMissItem(laneId, hostCacheK, hostCacheV, deviceBufferK, deviceBufferV, srcLoc, dstLoc, itemSizeBytes);
    }
}

// Shared-memory bytes for one request, matching the dynamic layout in the kernel.
size_t smemBytes(int32_t numTopK, int32_t hotBufferSize)
{
    int const hashSize = numTopK * 2;
    int const numBufferChunks = (hotBufferSize + kWarpSize - 1) / kWarpSize;
    int const totalInt32 = numTopK + 2 * (numBufferChunks + 1) + hashSize + 2;
    int const totalInt16 = hotBufferSize + hashSize;
    return static_cast<size_t>(totalInt32) * sizeof(int32_t) + static_cast<size_t>(totalInt16) * sizeof(int16_t);
}

template <typename SeqLensT, typename ReqPoolIndicesT>
void launch(int32_t const* topKBlocks, int32_t* deviceBufferBlocks, int64_t const* hostBlockLocs,
    int32_t const* deviceBufferLocs, char const* hostCacheK, char const* hostCacheV, char* deviceBufferK,
    char* deviceBufferV, int32_t* topKDeviceLocs, ReqPoolIndicesT const* reqPoolIndices, SeqLensT const* seqLensBlocks,
    int16_t* lruSlots, int32_t const* numRealReqs, int32_t numReqs, int32_t numTopK, int32_t hotBufferSize,
    int64_t bufferStride0, int64_t hostStride, int64_t lruSlotStride0, int64_t topKStride, int64_t topKDeviceLocsStride,
    int64_t itemSizeBytes, int32_t cudaBlockSize, cudaStream_t stream)
{
    size_t const smem = smemBytes(numTopK, hotBufferSize);
    auto kernel = hiSparseSwapInBlocksKernel<SeqLensT, ReqPoolIndicesT>;
    if (smem > 48u * 1024u)
    {
        TLLM_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    }
    kernel<<<numReqs, cudaBlockSize, smem, stream>>>(topKBlocks, deviceBufferBlocks, hostBlockLocs, deviceBufferLocs,
        hostCacheK, hostCacheV, deviceBufferK, deviceBufferV, topKDeviceLocs, reqPoolIndices, seqLensBlocks, lruSlots,
        numRealReqs, numTopK, hotBufferSize, bufferStride0, hostStride, lruSlotStride0, topKStride, topKDeviceLocsStride,
        itemSizeBytes);
}

} // namespace

void invokeHiSparseSwapInBlocks(int32_t const* topKBlocks, int32_t* deviceBufferBlocks, int64_t const* hostBlockLocs,
    int32_t const* deviceBufferLocs, void const* hostCacheK, void const* hostCacheV, void* deviceBufferK,
    void* deviceBufferV, int32_t* topKDeviceLocs, void const* reqPoolIndices, bool reqPoolIndicesIsInt64,
    void const* seqLensBlocks, bool seqLensIsInt64, int16_t* lruSlots, int32_t const* numRealReqs, int32_t numReqs,
    int32_t numTopK, int32_t hotBufferSize, int64_t bufferStride0, int64_t hostStride, int64_t lruSlotStride0,
    int64_t topKStride, int64_t topKDeviceLocsStride, int64_t itemSizeBytes, int32_t cudaBlockSize, cudaStream_t stream)
{
    if (numReqs == 0)
    {
        return;
    }
    auto const* hostK = static_cast<char const*>(hostCacheK);
    auto const* hostV = static_cast<char const*>(hostCacheV);
    auto* devK = static_cast<char*>(deviceBufferK);
    auto* devV = static_cast<char*>(deviceBufferV);

    auto dispatch = [&](auto seqTag, auto rpiTag)
    {
        using SeqT = decltype(seqTag);
        using RpiT = decltype(rpiTag);
        launch<SeqT, RpiT>(topKBlocks, deviceBufferBlocks, hostBlockLocs, deviceBufferLocs, hostK, hostV, devK, devV,
            topKDeviceLocs, static_cast<RpiT const*>(reqPoolIndices), static_cast<SeqT const*>(seqLensBlocks), lruSlots,
            numRealReqs, numReqs, numTopK, hotBufferSize, bufferStride0, hostStride, lruSlotStride0, topKStride,
            topKDeviceLocsStride, itemSizeBytes, cudaBlockSize, stream);
    };

    if (seqLensIsInt64 && reqPoolIndicesIsInt64)
    {
        dispatch(int64_t{}, int64_t{});
    }
    else if (seqLensIsInt64 && !reqPoolIndicesIsInt64)
    {
        dispatch(int64_t{}, int32_t{});
    }
    else if (!seqLensIsInt64 && reqPoolIndicesIsInt64)
    {
        dispatch(int32_t{}, int64_t{});
    }
    else
    {
        dispatch(int32_t{}, int32_t{});
    }
}

} // namespace kernels

TRTLLM_NAMESPACE_END
