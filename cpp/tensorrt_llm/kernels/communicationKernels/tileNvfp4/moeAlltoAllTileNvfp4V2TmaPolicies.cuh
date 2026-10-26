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

#include "moeAlltoAllTileNvfp4V2TmaCommon.cuh"
#include "moeAlltoAllTileNvfp4V2TmaFormats.cuh"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::moe_comm::tile_nvfp4_tma
{

// Defined in TmaTemplate.cuh. Policies select a compile-time transport
// geometry but never own the transport schedule. The Format template
// parameter is defaulted on the definitions in TmaTemplate.cuh.
template <int RowsPerToken, class Format, bool Packed>
struct HalfRecordWindow;

template <int RecordsPerToken, class Format, bool Packed>
struct FullTokenWindow;

// ---------------------------------------------------------------------------
// EP8/EP32: one token, eight route-position records, half-record window.
// ---------------------------------------------------------------------------
template <class FormatT, bool Packed = false>
struct Route8HalfPolicyT
{
    using Format = FormatT;
    using Window = HalfRecordWindow<8, Format, Packed>;

    struct Shared
    {
        CachedRoute* routeCache;
        std::uint64_t (*scaleCache)[kScaleCacheStride];
        typename Format::QGranule (*qCache)[Format::kQCacheStride];
        std::uint32_t* routeValidMask;
        std::uint32_t* routeBulkMask;
        std::uint64_t* tmaBar;
        // Packed [Q|SF] record cache (kTileNvfp4Compact); nullptr on the
        // split-plane path. Added last so the split-plane aggregate inits
        // (which omit it) keep compiling and remain byte-identical.
        std::uint8_t* packedCache = nullptr;
    };

    struct DecodeState
    {
        int work[1];
    };

    static constexpr int kTokensPerCta = 1;
    static constexpr int kStageRowsPerToken = kTopK;
    static constexpr int kQHalfTiles = kTilesPerToken / 2;
    static constexpr int kQHalfBytes = kQHalfTiles * Format::kQBytesPerTile;
    static constexpr unsigned kBar0BytesPerRoute = static_cast<unsigned>(kQHalfBytes + Format::kSfPayloadBytesPerToken);
    static constexpr unsigned kBar1BytesPerRoute = static_cast<unsigned>(kQHalfBytes);

    static_assert(kQHalfTiles == 56);

    __device__ __forceinline__ static void buildRouteCacheAndMasks(TileNvfp4CombinePointersV2 const& ptrs,
        int localNumTokens, int epRank, int epSize, int runtimeMaxTokensPerRank, int graphStaticMaxTokensPerRank,
        std::uint64_t bulkPeerMaskLo, std::uint64_t bulkPeerMaskHi, Shared const& shared)
    {
        int const routeWorker = static_cast<int>(threadIdx.x);
        bool routeValid = false;
        if (localNumTokens != 0 && routeWorker >= 32 && routeWorker < 32 + kTopK)
        {
            int const routePosition = routeWorker - 32;
            int const localToken = static_cast<int>(blockIdx.x);
            int const route = localToken * kTopK + routePosition;
            int const dstIdx = ptrs.topkSendIndices[route];
            CachedRoute cached{};
            if (dstIdx >= 0)
            {
                asm volatile("" ::: "memory");
                int const targetRank = ptrs.topkTargetRanks[route];
                if (__builtin_expect(targetRank < 0 || targetRank >= epSize || dstIdx >= runtimeMaxTokensPerRank, 0))
                {
                    asm volatile("trap;");
                    return;
                }

                std::size_t const qSlot
                    = static_cast<std::size_t>(epRank) * static_cast<std::size_t>(runtimeMaxTokensPerRank)
                    + static_cast<std::size_t>(dstIdx);
                std::size_t const sfSlot
                    = static_cast<std::size_t>(epRank) * static_cast<std::size_t>(graphStaticMaxTokensPerRank)
                    + static_cast<std::size_t>(dstIdx);
                if constexpr (Window::kPacked)
                {
                    // Packed plane rides sfPeer; slot = epRank*R + dstIdx
                    // (runtime R, matching the packed prepare grid). qRecord is
                    // the packed record base; sfRecord = base + q_aligned.
                    cached.pointer.qRecord = ptrs.sfPeer[targetRank] + qSlot * Format::kPackedRecordStrideBytes;
                    cached.pointer.sfRecord = cached.pointer.qRecord + Format::kQAlignedBytes;
                }
                else
                {
                    cached.pointer.qRecord = ptrs.qPeer[targetRank] + qSlot * kBf16TokenBytes;
                    cached.pointer.sfRecord = ptrs.sfPeer[targetRank] + sfSlot * Format::kSfPayloadBytesPerToken;
                }
                routeValid = true;
            }
            shared.routeCache[routePosition].vector = cached.vector;
        }

        int const setupWarp = static_cast<int>(threadIdx.x) >> 5;
        int const setupLane = static_cast<int>(threadIdx.x) & 31;
        if (setupWarp == 1)
        {
            std::uint32_t const routeValidMask = __ballot_sync(kFullWarpMask, routeValid);
            bool bulkEligible = false;
            if (routeValid)
            {
                int const route = static_cast<int>(blockIdx.x) * kTopK + setupLane;
                int const targetRank = ptrs.topkTargetRanks[route];
                bulkEligible
                    = (((targetRank < 64 ? bulkPeerMaskLo : bulkPeerMaskHi) >> (targetRank & 63)) & 1ULL) != 0ULL;
            }
            std::uint32_t const routeBulkMask = __ballot_sync(kFullWarpMask, bulkEligible);
            if (setupLane == 0)
            {
                *shared.routeValidMask = routeValidMask & 0xFFU;
                *shared.routeBulkMask = routeBulkMask & 0xFFU;
            }
        }
    }

    __device__ __forceinline__ static bool fastEligible(Shared const& shared, int tokensInCta)
    {
        return tokensInCta == kTokensPerCta && *shared.routeValidMask == 0xFFU;
    }

    __device__ __forceinline__ static std::uint32_t bulkStageMask(Shared const& shared)
    {
        return *shared.routeBulkMask;
    }

    __device__ __forceinline__ static CachedRoute stageRoute(Shared const& shared, int stageRow)
    {
        CachedRoute cached;
        cached.vector = shared.routeCache[stageRow].vector;
        return cached;
    }

    __device__ __forceinline__ static CachedRoute fallbackRoute(Shared const& shared, int tokenSlot, int routePosition)
    {
        (void) tokenSlot;
        CachedRoute cached;
        cached.vector = shared.routeCache[routePosition].vector;
        return cached;
    }

    __device__ __forceinline__ static DecodeState makeDecodeState(int warp)
    {
        return DecodeState{{warp}};
    }

    __device__ __forceinline__ static float collapse(float value, int routePosition)
    {
        return reduceRoute8Frozen(value, routePosition);
    }

    template <int Phase>
    __device__ __forceinline__ static void decodeFastPhase(
        TileNvfp4CombinePointersV2 const& ptrs, Shared const& shared, int ctaBase, int tokenSlot, int tile, int lane)
    {
        static_assert(Phase == 0 || Phase == 1);
        (void) tokenSlot;
        int const blockInTile = lane / kTopK;
        int const routePosition = lane % kTopK;
        __nv_bfloat16* outputRecord = ptrs.output + static_cast<std::size_t>(ctaBase) * kHiddenElements;

        std::uint64_t scalePack = 0;
        std::uint8_t const* qBlock;
        if constexpr (Window::kPacked)
        {
            std::uint8_t const* rec
                = shared.packedCache + static_cast<std::size_t>(routePosition) * Format::kPackedRecordStrideBytes;
            if constexpr (Format::kHasScalePlane)
            {
                scalePack = *reinterpret_cast<std::uint64_t const*>(rec + Window::sfByteOffset(tile));
            }
            qBlock = rec + Window::qByteOffset(tile) + blockInTile * Format::kQBytesPerBlock;
        }
        else
        {
            if constexpr (Format::kHasScalePlane)
            {
                scalePack = shared.scaleCache[routePosition][tile];
            }
            std::uint8_t const* qTile = reinterpret_cast<std::uint8_t const*>(&shared.qCache[routePosition][tile]);
            qBlock = qTile + blockInTile * Format::kQBytesPerBlock;
        }
        DecodedBlock16 const decoded = Format::decodeBlock16(qBlock, scalePack, blockInTile);

#pragma unroll
        for (int batch = 0; batch < kPairsPerBlock / kPairsPerStore; ++batch)
        {
            uint4 words = make_uint4(0U, 0U, 0U, 0U);
#pragma unroll
            for (int pairInBatch = 0; pairInBatch < kPairsPerStore; ++pairInBatch)
            {
                int const pair = batch * kPairsPerStore + pairInBatch;
                float x = decoded.pair[pair].x;
                float y = decoded.pair[pair].y;
                x = collapse(x, routePosition);
                y = collapse(y, routePosition);
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

            if (routePosition == 0)
            {
                __nv_bfloat16* output = outputRecord + static_cast<std::size_t>(tile) * kTileElements
                    + blockInTile * kScaleBlockElements + batch * (kPairsPerStore * 2);
                *reinterpret_cast<uint4*>(output) = words;
            }
        }
    }
};

// ---------------------------------------------------------------------------
// EP4: two tokens, low-four route-position records, half-record window.
// ---------------------------------------------------------------------------
template <class FormatT, bool Packed = false>
struct Route4HalfPolicyT
{
    using Format = FormatT;
    using Window = HalfRecordWindow<4, Format, Packed>;

    struct Shared
    {
        CachedRoute* routeCache;
        std::uint64_t (*scaleCache)[kScaleCacheStride];
        typename Format::QGranule (*qCache)[Format::kQCacheStride];
        std::uint32_t* routeValidMask;
        std::uint32_t* routeBulkMask;
        std::uint64_t* tmaBar;
        // Packed [Q|SF] record cache (kTileNvfp4Compact); nullptr on the
        // split-plane path. Added last for aggregate-init compatibility.
        std::uint8_t* packedCache = nullptr;
    };

    struct DecodeState
    {
        int work[2];
    };

    static constexpr int kTokensPerCta = 2;
    static constexpr int kFastRoutePositions = 4;
    static constexpr int kStageRowsPerToken = kFastRoutePositions;
    static constexpr int kStagePairs = kTokensPerCta * kFastRoutePositions;
    static constexpr int kFastQBytesPerLane = 4;
    static constexpr int kQHalfTiles = kTilesPerToken / 2;
    static constexpr int kQHalfBytes = kQHalfTiles * Format::kQBytesPerTile;
    static constexpr unsigned kBar0BytesPerRoute = static_cast<unsigned>(kQHalfBytes + Format::kSfPayloadBytesPerToken);
    static constexpr unsigned kBar1BytesPerRoute = static_cast<unsigned>(kQHalfBytes);

    static_assert(kStagePairs == kStageRows);

    __device__ __forceinline__ static void buildRouteCacheAndMasks(TileNvfp4CombinePointersV2 const& ptrs,
        int localNumTokens, int epRank, int epSize, int runtimeMaxTokensPerRank, int graphStaticMaxTokensPerRank,
        std::uint64_t bulkPeerMaskLo, std::uint64_t bulkPeerMaskHi, Shared const& shared)
    {
        int const routeWorker = static_cast<int>(threadIdx.x);
        bool routeValid = false;
        if (localNumTokens != 0 && routeWorker >= 32 && routeWorker < 32 + kTokensPerCta * kTopK)
        {
            int const tokenInCta = (routeWorker - 32) / kTopK;
            int const routePosition = (routeWorker - 32) % kTopK;
            int const localToken = static_cast<int>(blockIdx.x) * kTokensPerCta + tokenInCta;
            CachedRoute cached{};
            if (localToken < localNumTokens)
            {
                int const routeIndex = localToken * kTopK + routePosition;
                int const dstIdx = ptrs.topkSendIndices[routeIndex];
                if (dstIdx >= 0)
                {
                    asm volatile("" ::: "memory");
                    int const targetRank = ptrs.topkTargetRanks[routeIndex];
                    if (__builtin_expect(
                            targetRank < 0 || targetRank >= epSize || dstIdx >= runtimeMaxTokensPerRank, 0))
                    {
                        asm volatile("trap;");
                        return;
                    }
                    std::size_t const qSlot
                        = static_cast<std::size_t>(epRank) * static_cast<std::size_t>(runtimeMaxTokensPerRank)
                        + static_cast<std::size_t>(dstIdx);
                    std::size_t const sfSlot
                        = static_cast<std::size_t>(epRank) * static_cast<std::size_t>(graphStaticMaxTokensPerRank)
                        + static_cast<std::size_t>(dstIdx);
                    if constexpr (Window::kPacked)
                    {
                        cached.pointer.qRecord = ptrs.sfPeer[targetRank] + qSlot * Format::kPackedRecordStrideBytes;
                        cached.pointer.sfRecord = cached.pointer.qRecord + Format::kQAlignedBytes;
                    }
                    else
                    {
                        cached.pointer.qRecord = ptrs.qPeer[targetRank] + qSlot * kBf16TokenBytes;
                        cached.pointer.sfRecord = ptrs.sfPeer[targetRank] + sfSlot * Format::kSfPayloadBytesPerToken;
                    }
                    routeValid = true;
                }
            }
            shared.routeCache[tokenInCta * kTopK + routePosition].vector = cached.vector;
        }

        int const setupWarp = static_cast<int>(threadIdx.x) >> 5;
        int const setupLane = static_cast<int>(threadIdx.x) & 31;
        if (setupWarp == 1)
        {
            std::uint32_t const routeValidMask = __ballot_sync(kFullWarpMask, routeValid);
            bool bulkEligible = false;
            if (routeValid)
            {
                int const tokenInCta = setupLane / kTopK;
                int const routePosition = setupLane % kTopK;
                int const localToken = static_cast<int>(blockIdx.x) * kTokensPerCta + tokenInCta;
                int const routeIndex = localToken * kTopK + routePosition;
                int const targetRank = ptrs.topkTargetRanks[routeIndex];
                bulkEligible
                    = (((targetRank < 64 ? bulkPeerMaskLo : bulkPeerMaskHi) >> (targetRank & 63)) & 1ULL) != 0ULL;
            }
            std::uint32_t const bulkBallot = __ballot_sync(kFullWarpMask, bulkEligible);
            if (setupLane == 0)
            {
                *shared.routeValidMask = routeValidMask & 0xFFFFU;
                *shared.routeBulkMask = (bulkBallot & 0x0FU) | (((bulkBallot >> 8) & 0x0FU) << 4);
            }
        }
    }

    __device__ __forceinline__ static bool fastEligible(Shared const& shared, int tokensInCta)
    {
        std::uint32_t const expectedFastMask = tokensInCta == kTokensPerCta ? 0x0F0FU : 0x000FU;
        return *shared.routeValidMask == expectedFastMask;
    }

    __device__ __forceinline__ static std::uint32_t bulkStageMask(Shared const& shared)
    {
        return *shared.routeBulkMask;
    }

    __device__ __forceinline__ static CachedRoute stageRoute(Shared const& shared, int stageRow)
    {
        CachedRoute cached;
        int const tokenSlot = stageRow >> 2;
        int const routePosition = stageRow & 3;
        cached.vector = shared.routeCache[tokenSlot * kTopK + routePosition].vector;
        return cached;
    }

    __device__ __forceinline__ static CachedRoute fallbackRoute(Shared const& shared, int tokenSlot, int routePosition)
    {
        CachedRoute cached;
        cached.vector = shared.routeCache[tokenSlot * kTopK + routePosition].vector;
        return cached;
    }

    __device__ __forceinline__ static DecodeState makeDecodeState(int warp)
    {
        DecodeState state{};
#pragma unroll
        for (int tokenSlot = 0; tokenSlot < kTokensPerCta; ++tokenSlot)
        {
            state.work[tokenSlot] = warp;
        }
        return state;
    }

    __device__ __forceinline__ static float collapse(float value, int route4)
    {
        std::uint32_t const bits = __float_as_uint(value);
        std::uint32_t adjacentBits;
        asm volatile("shfl.sync.down.b32 %0, %1, 1, 0x1c1f, 0xffffffff;" : "=r"(adjacentBits) : "r"(bits));
        float const adjacent = __uint_as_float(adjacentBits);
        float const adjacentSum = __fadd_rn(value, adjacent);
        value = (route4 & 1) == 0 ? adjacentSum : value;

        std::uint32_t const pairInputBits = __float_as_uint(value);
        std::uint32_t pairBits;
        asm volatile("shfl.sync.down.b32 %0, %1, 2, 0x1c1f, 0xffffffff;" : "=r"(pairBits) : "r"(pairInputBits));
        float const pair = __uint_as_float(pairBits);
        float const left = __fadd_rn(value, pair);
        float const positiveZero = 0.0F;
        float const invalid45 = __fadd_rn(positiveZero, positiveZero);
        float const invalid67 = __fadd_rn(positiveZero, positiveZero);
        float const invalid47 = __fadd_rn(invalid45, invalid67);
        float const reduced = __fadd_rn(left, invalid47);
        return route4 == 0 ? reduced : value;
    }

    template <int Phase>
    __device__ __forceinline__ static void decodeFastPhase(
        TileNvfp4CombinePointersV2 const& ptrs, Shared const& shared, int ctaBase, int tokenSlot, int tile, int lane)
    {
        static_assert(Phase == 0 || Phase == 1);
        int const blockInTile = lane >> 3;
        int const half = (lane >> 2) & 1;
        int const route4 = lane & 3;
        int const stageRow = tokenSlot * kFastRoutePositions + route4;
        __nv_bfloat16* outputRecord = ptrs.output + static_cast<std::size_t>(ctaBase + tokenSlot) * kHiddenElements;
        std::uint64_t scalePack = 0;
        std::uint8_t const* qBlock;
        if constexpr (Window::kPacked)
        {
            std::uint8_t const* rec
                = shared.packedCache + static_cast<std::size_t>(stageRow) * Format::kPackedRecordStrideBytes;
            if constexpr (Format::kHasScalePlane)
            {
                scalePack = *reinterpret_cast<std::uint64_t const*>(rec + Window::sfByteOffset(tile));
            }
            qBlock = rec + Window::qByteOffset(tile) + blockInTile * Format::kQBytesPerBlock;
        }
        else
        {
            if constexpr (Format::kHasScalePlane)
            {
                scalePack = shared.scaleCache[stageRow][tile];
            }
            std::uint8_t const* qTile = reinterpret_cast<std::uint8_t const*>(&shared.qCache[stageRow][tile]);
            qBlock = qTile + blockInTile * Format::kQBytesPerBlock;
        }
        // Decode ONLY the kFastQBytesPerLane (4) pairs this half-lane stores
        // ([half*4, half*4+4)) instead of the full 8-pair decodeBlock16 whose
        // other half is discarded -- the two half-lanes of a block16 no longer
        // each decode all 8 pairs (2x decode redundancy removed). Byte-exact:
        // decodeHalfBlock16.pair[p] == decodeBlock16.pair[half*4+p] bit-for-bit.
        DecodedBlock16Half const decoded
            = Format::decodeHalfBlock16(qBlock, scalePack, blockInTile, half * kFastQBytesPerLane);

        uint4 words = make_uint4(0U, 0U, 0U, 0U);
#pragma unroll
        for (int pair = 0; pair < kFastQBytesPerLane; ++pair)
        {
            float x = decoded.pair[pair].x;
            float y = decoded.pair[pair].y;
            x = collapse(x, route4);
            y = collapse(y, route4);
            if (route4 == 0)
            {
                std::uint32_t const packed = packBf16Pair(x, y);
                if (pair == 0)
                {
                    words.x = packed;
                }
                else if (pair == 1)
                {
                    words.y = packed;
                }
                else if (pair == 2)
                {
                    words.z = packed;
                }
                else
                {
                    words.w = packed;
                }
            }
        }

        if (route4 == 0)
        {
            __nv_bfloat16* output = outputRecord + static_cast<std::size_t>(tile) * kTileElements
                + blockInTile * kScaleBlockElements + half * (kFastQBytesPerLane * 2);
            *reinterpret_cast<uint4*>(output) = words;
        }
    }
};

// ---------------------------------------------------------------------------
// EP2: four tokens, two compacted records per token, full-token window.
// ---------------------------------------------------------------------------
template <class FormatT, bool Packed = false>
struct Route2FullTokenPolicyT
{
    using Format = FormatT;
    using Window = FullTokenWindow<2, Format, Packed>;

    struct Shared
    {
        CachedRoute* routeCache;
        CachedRoute* recordCache;
        std::uint64_t (*scaleCache)[kScaleCacheStride];
        typename Format::QGranule (*qCache)[Format::kQCacheStride];
        std::uint32_t* routeValidMask;
        std::uint32_t* tokenFastMask;
        std::uint32_t* recordBulkMask;
        std::uint64_t* tmaBar;
        // Packed [Q|SF] record cache (kTileNvfp4Compact); nullptr on the
        // split-plane path. Added last for aggregate-init compatibility.
        std::uint8_t* packedCache = nullptr;
    };

    struct DecodeState
    {
        int work[1];
    };

    static constexpr int kTokensPerCta = 4;
    static constexpr int kRecordsPerToken = 2;
    static constexpr int kStageRowsPerToken = kRecordsPerToken;
    static constexpr int kStageRecords = kTokensPerCta * kRecordsPerToken;
    static constexpr int kTilesPerQuad = 4;
    static constexpr int kQuadsPerToken = kTilesPerToken / kTilesPerQuad;
    static constexpr unsigned kRecordBulkBytes
        = static_cast<unsigned>(Format::kQPayloadBytesPerToken + Format::kSfPayloadBytesPerToken);

    static_assert(kStageRecords == kStageRows);
    static_assert(kQuadsPerToken == 28);

    __device__ __forceinline__ static void buildRouteCacheAndMasks(TileNvfp4CombinePointersV2 const& ptrs,
        int localNumTokens, int epRank, int epSize, int runtimeMaxTokensPerRank, int graphStaticMaxTokensPerRank,
        std::uint64_t bulkPeerMaskLo, std::uint64_t bulkPeerMaskHi, Shared const& shared)
    {
        int const worker = static_cast<int>(threadIdx.x);
        if (worker < 32 || worker >= 64)
        {
            return;
        }
        int const lane = worker - 32;
        int const tokenSlot = lane >> 3;
        int const routePosition = lane & 7;
        int const localToken = static_cast<int>(blockIdx.x) * kTokensPerCta + tokenSlot;

        CachedRoute cached{};
        bool valid = false;
        int targetRank = -1;
        if (localNumTokens != 0 && localToken < localNumTokens)
        {
            int const routeIndex = localToken * kTopK + routePosition;
            int const dstIdx = ptrs.topkSendIndices[routeIndex];
            if (dstIdx >= 0)
            {
                asm volatile("" ::: "memory");
                targetRank = ptrs.topkTargetRanks[routeIndex];
                if (__builtin_expect(targetRank < 0 || targetRank >= epSize || dstIdx >= runtimeMaxTokensPerRank, 0))
                {
                    asm volatile("trap;");
                    return;
                }
                std::size_t const qSlot
                    = static_cast<std::size_t>(epRank) * static_cast<std::size_t>(runtimeMaxTokensPerRank)
                    + static_cast<std::size_t>(dstIdx);
                if constexpr (Window::kPacked)
                {
                    // Packed plane rides sfPeer; slot = epRank*R + dstIdx
                    // (== qSlot, runtime R). qRecord = packed record base,
                    // sfRecord = base + q_aligned (3584).
                    cached.pointer.qRecord = ptrs.sfPeer[targetRank] + qSlot * Format::kPackedRecordStrideBytes;
                    cached.pointer.sfRecord = cached.pointer.qRecord + Format::kQAlignedBytes;
                }
                else
                {
                    std::size_t const sfSlot
                        = static_cast<std::size_t>(epRank) * static_cast<std::size_t>(graphStaticMaxTokensPerRank)
                        + static_cast<std::size_t>(dstIdx);
                    cached.pointer.qRecord = ptrs.qPeer[targetRank] + qSlot * kBf16TokenBytes;
                    cached.pointer.sfRecord = ptrs.sfPeer[targetRank] + sfSlot * Format::kSfPayloadBytesPerToken;
                }
                valid = true;
            }
        }
        shared.routeCache[tokenSlot * kTopK + routePosition].vector = cached.vector;

        std::uint32_t const validMask = __ballot_sync(kFullWarpMask, valid);
        std::uint32_t const tokenByte = (validMask >> (tokenSlot * 8)) & 0xFFU;
        int const validBefore = __popc(tokenByte & ((1U << routePosition) - 1U));
        bool const ownsRecord = valid && validBefore < kRecordsPerToken;
        int const record = tokenSlot * kRecordsPerToken + validBefore;
        if (ownsRecord)
        {
            shared.recordCache[record].vector = cached.vector;
        }

        std::uint32_t folded = 0;
        if (ownsRecord && (((targetRank < 64 ? bulkPeerMaskLo : bulkPeerMaskHi) >> (targetRank & 63)) & 1ULL) != 0ULL)
        {
            folded |= (1U << record);
        }
        if (routePosition == 0 && __popc(tokenByte) == kRecordsPerToken)
        {
            folded |= (1U << (16 + tokenSlot));
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            folded |= __shfl_down_sync(kFullWarpMask, folded, offset);
        }
        if (lane == 0)
        {
            *shared.routeValidMask = validMask;
            *shared.recordBulkMask = folded & 0xFFU;
            *shared.tokenFastMask = (folded >> 16) & 0xFU;
        }
    }

    __device__ __forceinline__ static bool fastEligible(Shared const& shared, int tokensInCta)
    {
        std::uint32_t const tokenFastMask = *shared.tokenFastMask;
        std::uint32_t const inRangeMask = (1U << tokensInCta) - 1U;
        return (tokenFastMask & inRangeMask) == inRangeMask;
    }

    __device__ __forceinline__ static std::uint32_t bulkStageMask(Shared const& shared)
    {
        return *shared.recordBulkMask;
    }

    __device__ __forceinline__ static CachedRoute stageRoute(Shared const& shared, int stageRow)
    {
        CachedRoute cached;
        cached.vector = shared.recordCache[stageRow].vector;
        return cached;
    }

    __device__ __forceinline__ static CachedRoute fallbackRoute(Shared const& shared, int tokenSlot, int routePosition)
    {
        CachedRoute cached;
        cached.vector = shared.routeCache[tokenSlot * kTopK + routePosition].vector;
        return cached;
    }

    __device__ __forceinline__ static DecodeState makeDecodeState(int warp)
    {
        return DecodeState{{warp}};
    }

    __device__ __forceinline__ static float collapse(float value, int routePosition)
    {
        float const partner = __shfl_down_sync(kFullWarpMask, value, 1, kRecordsPerToken);
        if (routePosition == 0)
        {
            value = __fadd_rn(value, partner);
        }
        return value;
    }

    template <int Phase>
    __device__ __forceinline__ static void decodeFastPhase(TileNvfp4CombinePointersV2 const& ptrs, Shared const& shared,
        int ctaBase, int tokenSlot, int quadInToken, int lane)
    {
        static_assert(Phase == 0 || Phase == 1);
        int const tileInQuad = lane >> 3;
        int const blockInTile = (lane & 7) >> 1;
        int const routePosition = lane & 1;
        int const tile = quadInToken * kTilesPerQuad + tileInQuad;
        int const record = tokenSlot * kRecordsPerToken + routePosition;
        __nv_bfloat16* outputRecord = ptrs.output + static_cast<std::size_t>(ctaBase + tokenSlot) * kHiddenElements;

        std::uint64_t scalePack = 0;
        std::uint8_t const* qBlock;
        if constexpr (Window::kPacked)
        {
            std::uint8_t const* rec
                = shared.packedCache + static_cast<std::size_t>(record) * Format::kPackedRecordStrideBytes;
            if constexpr (Format::kHasScalePlane)
            {
                scalePack = *reinterpret_cast<std::uint64_t const*>(rec + Window::sfByteOffset(tile));
            }
            qBlock = rec + Window::qByteOffset(tile) + blockInTile * Format::kQBytesPerBlock;
        }
        else
        {
            if constexpr (Format::kHasScalePlane)
            {
                scalePack = shared.scaleCache[record][tile];
            }
            std::uint8_t const* qTile = reinterpret_cast<std::uint8_t const*>(&shared.qCache[record][tile]);
            qBlock = qTile + blockInTile * Format::kQBytesPerBlock;
        }
        DecodedBlock16 const decoded = Format::decodeBlock16(qBlock, scalePack, blockInTile);

#pragma unroll
        for (int batch = 0; batch < kPairsPerBlock / kPairsPerStore; ++batch)
        {
            uint4 words = make_uint4(0U, 0U, 0U, 0U);
#pragma unroll
            for (int pairInBatch = 0; pairInBatch < kPairsPerStore; ++pairInBatch)
            {
                int const pair = batch * kPairsPerStore + pairInBatch;
                float x = decoded.pair[pair].x;
                float y = decoded.pair[pair].y;
                x = collapse(x, routePosition);
                y = collapse(y, routePosition);
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

            if (routePosition == 0)
            {
                __nv_bfloat16* output = outputRecord + static_cast<std::size_t>(tile) * kTileElements
                    + blockInTile * kScaleBlockElements + batch * (kPairsPerStore * 2);
                *reinterpret_cast<uint4*>(output) = words;
            }
        }
    }
};

// Concrete NVFP4 policy aliases: the templated *T policies instantiated with
// Nvfp4Format (decodeBlock16 reproduces the exact math).
using Route8HalfPolicy = Route8HalfPolicyT<Nvfp4Format>;
using Route4HalfPolicy = Route4HalfPolicyT<Nvfp4Format>;
using Route2FullTokenPolicy = Route2FullTokenPolicyT<Nvfp4Format>;

// Packed-interleaved [Q|SF] (kTileNvfp4Compact) NVFP4 EP2 policy. One contiguous
// 4480-B record per stage row -> ONE cp.async.bulk per record (2 TMA/token vs
// the split-plane 4 TMA/token). The split-plane aliases above keep Packed=false
// and stay byte-identical.
using Route2FullTokenPackedPolicy = Route2FullTokenPolicyT<Nvfp4Format, true>;

// Packed-interleaved [Q_h0|SF_h0|Q_h1|SF_h1] NVFP4 EP4/EP8 Half policies. Each stage
// row is ONE contiguous 4480-B record staged as two 2240-B phase slices (one
// cp.async.bulk per phase, 2 TMA/record vs the split-plane 3). The *T policy
// bodies are already kPacked-correct (decodeFastPhase reads packedCache with the
// interleaved Window offsets); the split-plane Half aliases above keep
// Packed=false and stay byte-identical.
using Route4HalfPackedPolicy = Route4HalfPolicyT<Nvfp4Format, true>;
using Route8HalfPackedPolicy = Route8HalfPolicyT<Nvfp4Format, true>;

} // namespace kernels::moe_comm::tile_nvfp4_tma

TRTLLM_NAMESPACE_END
