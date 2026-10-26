/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include "moeAlltoAllTileNvfp4V2.h"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::moe_comm::tile_nvfp4_tma
{

constexpr int kThreads = 512;
constexpr int kWarpsPerCta = kThreads / 32;
constexpr int kTopK = 8;
constexpr int kHiddenElements = 7168;
constexpr int kTileElements = 64;
constexpr int kScaleBlockElements = 16;
constexpr int kTilesPerToken = kHiddenElements / kTileElements;
constexpr int kBlocksPerTile = kTileElements / kScaleBlockElements;
constexpr int kPairsPerBlock = kScaleBlockElements / 2;
constexpr int kPairsPerStore = 4;
constexpr int kQBytesPerTile = 32;
constexpr int kQBytesPerBlock = kScaleBlockElements / 2;
constexpr int kSfBytesPerTile = 8;
constexpr int kScaleCacheStride = kTilesPerToken + 2;
constexpr int kScaleCacheRouteStrideBytes = kScaleCacheStride * static_cast<int>(sizeof(std::uint64_t));
constexpr int kQCacheStride = kTilesPerToken + 1;
constexpr int kQCacheRouteStrideBytes = kQCacheStride * kQBytesPerTile;
constexpr int kBf16TokenBytes = 14336;
constexpr int kQPayloadBytesPerToken = 3584;
constexpr int kSfPayloadBytesPerToken = 896;
constexpr int kStageRows = 8;
constexpr int kScaleCacheBytes = kStageRows * kScaleCacheRouteStrideBytes;
constexpr int kQCacheBytes = kStageRows * kQCacheRouteStrideBytes;
constexpr unsigned int kFullWarpMask = 0xFFFFFFFFU;
constexpr std::uint64_t kTimeoutCycles = 600000000000ULL;

static_assert(kWarpsPerCta == 16);
static_assert(kTilesPerToken == 112);
static_assert(kBlocksPerTile * kTopK == 32);
static_assert(kPairsPerBlock == 8);
static_assert(kQBytesPerBlock == 8);
static_assert(kQPayloadBytesPerToken == kTilesPerToken * kQBytesPerTile);
static_assert(kSfPayloadBytesPerToken == kTilesPerToken * kSfBytesPerTile);
static_assert(kStageRows == 8);
static_assert(kScaleCacheStride == 114);
static_assert(kScaleCacheRouteStrideBytes == 912);
static_assert(kScaleCacheRouteStrideBytes % 16 == 0);
static_assert((kScaleCacheRouteStrideBytes / 4) % 32 == 4);
static_assert(kScaleCacheBytes == 7296);
static_assert(kQCacheStride == 113);
static_assert(kQCacheRouteStrideBytes == 3616);
static_assert(kQCacheRouteStrideBytes == kQPayloadBytesPerToken + kQBytesPerTile);
static_assert(kQCacheRouteStrideBytes % 32 == 0);
static_assert((kQCacheRouteStrideBytes / 4) % 32 == 8);
static_assert(kQCacheBytes == 28928);

// This POD block mirrors the per-EP combine variants field for
// field. The per-peer arrays are sized by the tileNvfp4-local kMaxRanksTile
// (128) so the unified combine ABI covers EP up to 128; the shared dispatch
// kMaxRanks (64) is deliberately untouched.
struct TileNvfp4CombinePointersV2
{
    std::uint8_t const* qPeer[kMaxRanksTile];
    std::uint8_t const* sfPeer[kMaxRanksTile];
    std::uint32_t* completionFlags[kMaxRanksTile];
    std::uint32_t const* flagVal;
    int const* topkTargetRanks;
    int const* topkSendIndices;
    __nv_bfloat16* output;
};

struct RoutePointers
{
    std::uint8_t const* qRecord;
    std::uint8_t const* sfRecord;
};

union alignas(16) CachedRoute
{
    RoutePointers pointer;
    uint4 vector;
};

static_assert(sizeof(CachedRoute) == 16);
static_assert(alignof(CachedRoute) == 16);

struct alignas(32) B256
{
    uint4 a;
    uint4 b;
};

static_assert(sizeof(B256) == 32);
static_assert(alignof(B256) == 32);
static_assert(kQBytesPerTile == sizeof(B256));
static_assert(kBf16TokenBytes % alignof(B256) == 0);

union Bf16PairBits
{
    __nv_bfloat162 value;
    std::uint32_t bits;
};

// Internal, width-clean controls consumed by every TMA instantiation.
// bulkPeerMaskLo/Hi form a 128-bit per-peer mask held as two scalar words
// (NOT an array -- runtime array indexing would defeat SROA and spill the
// whole controls block to the local stack): bit r
// = (r < 64 ? bulkPeerMaskLo : bulkPeerMaskHi) >> (r & 63). A uint32 mask is
// zero-extended into Lo (Hi = 0).
struct TmaRuntimeControls
{
    std::uint64_t bulkPeerMaskLo;
    std::uint64_t bulkPeerMaskHi;
    std::int32_t poisonMode;
    std::uint32_t poisonEpoch;
    std::int32_t skewProfile;
    std::int32_t skewStragglerRoute;
    std::int32_t delayedDmaCycles;
    unsigned long long* debugBarTimestamps;
};

static __device__ __forceinline__ void publishAndAcquireReadiness(
    TileNvfp4CombinePointersV2 const& ptrs, int epRank, int epSize)
{
    if (static_cast<int>(threadIdx.x) >= 32)
    {
        return;
    }
    int const lane = static_cast<int>(threadIdx.x);
    std::uint32_t const expected = *ptrs.flagVal;

    if (blockIdx.x == 0)
    {
        asm volatile("fence.release.sys;" ::: "memory");
#pragma unroll 1
        for (int peer = lane; peer < epSize; peer += 32)
        {
            std::uint32_t* address = &ptrs.completionFlags[peer][epRank];
            asm volatile("st.relaxed.sys.u32 [%0], %1;" ::"l"(address), "r"(expected) : "memory");
        }
    }

#pragma unroll 1
    for (int peer = lane; peer < epSize; peer += 32)
    {
        std::uint32_t* address = &ptrs.completionFlags[epRank][peer];
        std::uint64_t const start = clock64();
        bool ready = false;
        do
        {
            std::uint32_t observed;
            asm volatile("ld.relaxed.sys.u32 %0, [%1];" : "=r"(observed) : "l"(address) : "memory");
            ready = observed == expected;
        } while (!ready && (clock64() - start) <= kTimeoutCycles);
        if (__builtin_expect(!ready, 0))
        {
            asm volatile("trap;");
            return;
        }
    }
    asm volatile("fence.acquire.sys;" ::: "memory");
}

static __device__ __forceinline__ std::uint64_t loadScalePackOncePerRoute(
    CachedRoute const& cached, int tile, int blockInTile, int routePosition)
{
    // Lanes 0..7 are block16==0. Each of them reads one route's full 8-byte
    // scale pack. The pack is then broadcast vertically to lanes route+8*n.
    std::uint64_t pack = 0;
    if (blockInTile == 0 && cached.pointer.qRecord != nullptr)
    {
        pack = *reinterpret_cast<std::uint64_t const*>(
            cached.pointer.sfRecord + static_cast<std::size_t>(tile) * kSfBytesPerTile);
    }
    std::uint32_t const low = __shfl_sync(kFullWarpMask, static_cast<std::uint32_t>(pack), routePosition);
    std::uint32_t const high = __shfl_sync(kFullWarpMask, static_cast<std::uint32_t>(pack >> 32), routePosition);
    return static_cast<std::uint64_t>(low) | (static_cast<std::uint64_t>(high) << 32);
}

// Packed (kTileNvfp4Compact) sibling of loadScalePackOncePerRoute. The packed
// record stitches Q and SF into ONE buffer whose base is cached.pointer.qRecord,
// so the 8-byte SF pack for `tile` lives at qRecord + Window::sfByteOffset(tile)
// (FullToken: qRecord + 3584 + tile*8, identical to the split-plane
// sfRecord + tile*8; Half: the interleaved SF_h0/SF_h1 offset). Everything else
// (the blockInTile==0 read gated on a non-null record + the low/high shfl
// broadcast to lanes route+8*n) is byte-identical to the split-plane loader.
template <class Window>
static __device__ __forceinline__ std::uint64_t loadScalePackOncePerRoutePacked(
    CachedRoute const& cached, int tile, int blockInTile, int routePosition)
{
    std::uint64_t pack = 0;
    if (blockInTile == 0 && cached.pointer.qRecord != nullptr)
    {
        pack = *reinterpret_cast<std::uint64_t const*>(cached.pointer.qRecord + Window::sfByteOffset(tile));
    }
    std::uint32_t const low = __shfl_sync(kFullWarpMask, static_cast<std::uint32_t>(pack), routePosition);
    std::uint32_t const high = __shfl_sync(kFullWarpMask, static_cast<std::uint32_t>(pack >> 32), routePosition);
    return static_cast<std::uint64_t>(low) | (static_cast<std::uint64_t>(high) << 32);
}

static __device__ __forceinline__ float reduceRoute8Frozen(float value, int routePosition)
{
    // Distributed form of the frozen K8 tree:
    // ((v0+v1)+(v2+v3)) + ((v4+v5)+(v6+v7)).
    float const adjacent = __shfl_down_sync(kFullWarpMask, value, 1, kTopK);
    if ((routePosition & 1) == 0)
    {
        value = __fadd_rn(value, adjacent);
    }

    float const pair = __shfl_down_sync(kFullWarpMask, value, 2, kTopK);
    if ((routePosition & 3) == 0)
    {
        value = __fadd_rn(value, pair);
    }

    float const half = __shfl_down_sync(kFullWarpMask, value, 4, kTopK);
    if (routePosition == 0)
    {
        value = __fadd_rn(value, half);
    }
    return value;
}

static __device__ __forceinline__ std::uint32_t packBf16Pair(float x, float y)
{
    Bf16PairBits result;
    result.value = __floats2bfloat162_rn(x, y);
    return result.bits;
}

// Generic K8 fallback. Keep this non-template and __noinline__: it is the
// fail-closed path for every routing policy.
static __device__ __noinline__ void decodeReduceAndStoreTile(
    CachedRoute const& cached, __nv_bfloat16* outputRecord, int tile, int blockInTile, int routePosition)
{
    std::uint64_t const scalePack = loadScalePackOncePerRoute(cached, tile, blockInTile, routePosition);
    float scaledBlock = 0.0F;
    if (cached.pointer.qRecord != nullptr)
    {
        std::uint8_t const rawBlockScale = static_cast<std::uint8_t>(scalePack >> (blockInTile * 8));
        float const globalD = __uint_as_float(static_cast<std::uint32_t>(scalePack >> 32));
        __nv_fp8_e4m3 blockScale;
        blockScale.__x = rawBlockScale;
        scaledBlock = __fmul_rn(static_cast<float>(blockScale), globalD);
    }

    // Each lane reads one aligned 8-byte Q block. The address and load remain
    // inside the valid branch so invalid routes cannot touch peer payload.
    std::uint64_t packedQ = 0;
    if (cached.pointer.qRecord != nullptr)
    {
        std::uint8_t const* qAddress
            = cached.pointer.qRecord + static_cast<std::size_t>(tile) * kQBytesPerTile + blockInTile * kQBytesPerBlock;
        packedQ = *reinterpret_cast<std::uint64_t const*>(qAddress);
    }

#pragma unroll
    for (int batch = 0; batch < kPairsPerBlock / kPairsPerStore; ++batch)
    {
        uint4 words = make_uint4(0U, 0U, 0U, 0U);
#pragma unroll
        for (int pairInBatch = 0; pairInBatch < kPairsPerStore; ++pairInBatch)
        {
            int const pair = batch * kPairsPerStore + pairInBatch;
            float x = 0.0F;
            float y = 0.0F;
            if (cached.pointer.qRecord != nullptr)
            {
                __nv_fp4x2_e2m1 qPair;
                qPair.__x = static_cast<__nv_fp4x2_storage_t>((packedQ >> (pair * 8)) & 0xFFU);
                float2 const q = static_cast<float2>(qPair);
                x = __fmul_rn(q.x, scaledBlock);
                y = __fmul_rn(q.y, scaledBlock);
            }

            x = reduceRoute8Frozen(x, routePosition);
            y = reduceRoute8Frozen(y, routePosition);
            if (routePosition == 0)
            {
                std::uint32_t const packed = packBf16Pair(x, y);
                if (pairInBatch == 0)
                {
                    words.x = packed;
                }
                else if (pairInBatch == 1)
                {
                    words.y = packed;
                }
                else if (pairInBatch == 2)
                {
                    words.z = packed;
                }
                else
                {
                    words.w = packed;
                }
            }
        }

        // Lanes 0/8/16/24 are routePosition==0 and each owns one block16.
        if (routePosition == 0)
        {
            __nv_bfloat16* output = outputRecord + static_cast<std::size_t>(tile) * kTileElements
                + blockInTile * kScaleBlockElements + batch * (kPairsPerStore * 2);
            *reinterpret_cast<uint4*>(output) = words;
        }
    }
}

} // namespace kernels::moe_comm::tile_nvfp4_tma

TRTLLM_NAMESPACE_END
