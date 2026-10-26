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

#include "moeAlltoAllTileNvfp4V2TmaFormats.cuh"
#include "moeAlltoAllTileNvfp4V2TmaPolicies.cuh"
#include "moeAlltoAllTileNvfp4V2TmaPrimitives.cuh"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::moe_comm::tile_nvfp4_tma
{

struct DecodeCoord
{
    int tokenSlot;
    int unit;
};

// Two barriers divide every stage row at the Q-record midpoint. Phase zero
// transfers Q[0:56] and the complete scale record; phase one transfers
// Q[56:112]. Rows remain active in both phases. The Q byte geometry and the
// presence of a scale record are sourced from the payload Format; NVFP4 keeps
// its original numbers, FP8/BF16 widen Q and drop the scale plane.
template <int RowsPerToken, class Format = Nvfp4Format, bool Packed = false>
struct HalfRecordWindow
{
    static_assert(RowsPerToken == 4 || RowsPerToken == 8);
    static_assert(kStageRows % RowsPerToken == 0);

    static constexpr bool kPacked = Packed;
    static constexpr int kTokensPerCta = kStageRows / RowsPerToken;
    static constexpr int kDecodeSpans = kTokensPerCta;
    static constexpr int kQHalfTiles = kTilesPerToken / 2;
    static constexpr unsigned kQHalfBytes = static_cast<unsigned>(kQHalfTiles * Format::kQBytesPerTile);

    __device__ __forceinline__ static int activeRows(int tokensInCta)
    {
        return tokensInCta * RowsPerToken;
    }

    __device__ __forceinline__ static std::uint32_t activeRowsMask(int tokensInCta)
    {
        return (1U << activeRows(tokensInCta)) - 1U;
    }

    template <int Phase>
    __device__ __forceinline__ static std::uint32_t phaseRowsMask(std::uint32_t activeMask)
    {
        static_assert(Phase == 0 || Phase == 1);
        return activeMask;
    }

    template <int Phase>
    static constexpr __host__ __device__ int qTileBegin()
    {
        static_assert(Phase == 0 || Phase == 1);
        return Phase == 0 ? 0 : kQHalfTiles;
    }

    template <int Phase>
    static constexpr __host__ __device__ unsigned qBytes()
    {
        static_assert(Phase == 0 || Phase == 1);
        return kQHalfBytes;
    }

    template <int Phase>
    static constexpr __host__ __device__ unsigned sfBytes()
    {
        static_assert(Phase == 0 || Phase == 1);
        // Split-plane Half loads the WHOLE SF record (896 B) in phase 0 only.
        // Packed Half interleaves the SF halves into the two phase slices, so
        // each phase carries exactly kSfHalfBytes (448 B).
        if constexpr (Packed)
        {
            return static_cast<unsigned>(Format::kSfHalfBytes);
        }
        else
        {
            return Phase == 0 ? static_cast<unsigned>(Format::kSfPayloadBytesPerToken) : 0U;
        }
    }

    // Packed record byte layout (per stage row):
    //   [ Q_h0(1792) | SF_h0(448) | Q_h1(1792) | SF_h1(448) ]
    // phase 0 owns the [0, 2240) slice, phase 1 the [2240, 4480) slice.
    template <int Phase>
    static constexpr __host__ __device__ unsigned phaseByteBegin()
    {
        static_assert(Phase == 0 || Phase == 1);
        return Phase == 0 ? 0U : static_cast<unsigned>(Format::kQHalfAlignedBytes + Format::kSfHalfBytes);
    }

    static constexpr __host__ __device__ int qByteOffset(int tile)
    {
        return tile < kQHalfTiles
            ? tile * Format::kQBytesPerTile
            : (Format::kQHalfAlignedBytes + Format::kSfHalfBytes) + (tile - kQHalfTiles) * Format::kQBytesPerTile;
    }

    static constexpr __host__ __device__ int sfByteOffset(int tile)
    {
        return tile < kQHalfTiles
            ? Format::kQHalfAlignedBytes + tile * Format::kSfBytesPerTile
            : (2 * Format::kQHalfAlignedBytes + Format::kSfHalfBytes) + (tile - kQHalfTiles) * Format::kSfBytesPerTile;
    }

    template <int Phase>
    __device__ __forceinline__ static int decodeSpanCount(int tokensInCta)
    {
        static_assert(Phase == 0 || Phase == 1);
        return tokensInCta;
    }

    template <int Phase>
    __device__ __forceinline__ static int decodeBegin(int span, int tokensInCta)
    {
        static_assert(Phase == 0 || Phase == 1);
        (void) span;
        (void) tokensInCta;
        return Phase == 0 ? 0 : kQHalfTiles;
    }

    template <int Phase>
    __device__ __forceinline__ static int decodeEnd(int span, int tokensInCta)
    {
        static_assert(Phase == 0 || Phase == 1);
        (void) span;
        (void) tokensInCta;
        return Phase == 0 ? kQHalfTiles : kTilesPerToken;
    }

    template <int Phase>
    __device__ __forceinline__ static DecodeCoord decodeCoord(int span, int work)
    {
        static_assert(Phase == 0 || Phase == 1);
        return DecodeCoord{span, work};
    }
};

// Each stage row is one complete Q+scale record. Barrier zero owns rows 0..3
// (tokens 0..1); barrier one owns rows 4..7 (tokens 2..3).
template <int RecordsPerToken, class Format = Nvfp4Format, bool Packed = false>
struct FullTokenWindow
{
    static_assert(RecordsPerToken == 2);
    static_assert(kStageRows % RecordsPerToken == 0);

    static constexpr bool kPacked = Packed;
    static constexpr int kTokensPerCta = kStageRows / RecordsPerToken;
    static constexpr int kDecodeSpans = 1;
    static constexpr int kQuadsPerToken = 28;

    __device__ __forceinline__ static int activeRows(int tokensInCta)
    {
        return tokensInCta * RecordsPerToken;
    }

    __device__ __forceinline__ static std::uint32_t activeRowsMask(int tokensInCta)
    {
        return (1U << activeRows(tokensInCta)) - 1U;
    }

    template <int Phase>
    __device__ __forceinline__ static std::uint32_t phaseRowsMask(std::uint32_t activeMask)
    {
        static_assert(Phase == 0 || Phase == 1);
        return activeMask & (Phase == 0 ? 0x0FU : 0xF0U);
    }

    template <int Phase>
    static constexpr __host__ __device__ int qTileBegin()
    {
        static_assert(Phase == 0 || Phase == 1);
        return 0;
    }

    template <int Phase>
    static constexpr __host__ __device__ unsigned qBytes()
    {
        static_assert(Phase == 0 || Phase == 1);
        return static_cast<unsigned>(Format::kQPayloadBytesPerToken);
    }

    template <int Phase>
    static constexpr __host__ __device__ unsigned sfBytes()
    {
        static_assert(Phase == 0 || Phase == 1);
        return static_cast<unsigned>(Format::kSfPayloadBytesPerToken);
    }

    // Packed full-token record is one contiguous [ Q(3584) | SF(896) ] plane
    // transferred whole per stage row; the two phases split by ROWS (0..3 /
    // 4..7), not by byte offset, so both phase slices start at record byte 0.
    template <int Phase>
    static constexpr __host__ __device__ unsigned phaseByteBegin()
    {
        static_assert(Phase == 0 || Phase == 1);
        return 0U;
    }

    static constexpr __host__ __device__ int qByteOffset(int tile)
    {
        return tile * Format::kQBytesPerTile;
    }

    static constexpr __host__ __device__ int sfByteOffset(int tile)
    {
        return Format::kQAlignedBytes + tile * Format::kSfBytesPerTile;
    }

    template <int Phase>
    __device__ __forceinline__ static int decodeSpanCount(int tokensInCta)
    {
        static_assert(Phase == 0 || Phase == 1);
        (void) tokensInCta;
        return kDecodeSpans;
    }

    template <int Phase>
    __device__ __forceinline__ static int decodeBegin(int span, int tokensInCta)
    {
        static_assert(Phase == 0 || Phase == 1);
        (void) span;
        int const phaseBoundary = (tokensInCta < 2 ? tokensInCta : 2) * kQuadsPerToken;
        return Phase == 0 ? 0 : phaseBoundary;
    }

    template <int Phase>
    __device__ __forceinline__ static int decodeEnd(int span, int tokensInCta)
    {
        static_assert(Phase == 0 || Phase == 1);
        (void) span;
        int const phaseBoundary = (tokensInCta < 2 ? tokensInCta : 2) * kQuadsPerToken;
        return Phase == 0 ? phaseBoundary : tokensInCta * kQuadsPerToken;
    }

    template <int Phase>
    __device__ __forceinline__ static DecodeCoord decodeCoord(int span, int work)
    {
        static_assert(Phase == 0 || Phase == 1);
        (void) span;
        return DecodeCoord{work / kQuadsPerToken, work % kQuadsPerToken};
    }
};

template <int Phase, class RoutingPolicy>
__device__ __forceinline__ void issueTmaPhase(
    typename RoutingPolicy::Shared const& shared, std::uint32_t bulkStageMask, TmaRuntimeControls const& controls)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    using Window = typename RoutingPolicy::Window;
    using Format = typename RoutingPolicy::Format;
    std::uint32_t const phaseMask = Window::template phaseRowsMask<Phase>(bulkStageMask);
    if (phaseMask == 0U)
    {
        return;
    }

    unsigned const bytesPerStage = Window::template qBytes<Phase>() + Window::template sfBytes<Phase>();
    mbarArriveExpectTx(&shared.tmaBar[Phase], static_cast<unsigned>(__popc(phaseMask)) * bytesPerStage);

#pragma unroll
    for (int stageRow = 0; stageRow < kStageRows; ++stageRow)
    {
        if (((phaseMask >> stageRow) & 1U) == 0U)
        {
            continue;
        }
        spinCycles(static_cast<long long>(
                       skewRank(stageRow, controls.skewProfile, controls.skewStragglerRoute, controls.poisonEpoch))
            * kSkewCycleUnit);
        CachedRoute const cached = RoutingPolicy::stageRoute(shared, stageRow);
        if constexpr (Window::kPacked)
        {
            // ONE bulk per phase slice: the packed record stitches Q and SF
            // into a single contiguous buffer. cached.pointer.qRecord is the
            // packed record base; the phase slice begins at phaseByteBegin and
            // is exactly qBytes + sfBytes long (FullToken: 4480 at offset 0;
            // Half: 2240 at offset 0 / 2240). The dst offset within packedCache
            // mirrors the src offset within the record (same layout).
            constexpr unsigned phaseBegin = Window::template phaseByteBegin<Phase>();
            std::uint8_t* rec
                = shared.packedCache + static_cast<std::size_t>(stageRow) * Format::kPackedRecordStrideBytes;
            bulk1d(rec + phaseBegin, cached.pointer.qRecord + phaseBegin, bytesPerStage, &shared.tmaBar[Phase]);
        }
        else
        {
            constexpr int qTileBegin = Window::template qTileBegin<Phase>();
            bulk1d(&shared.qCache[stageRow][qTileBegin],
                cached.pointer.qRecord + static_cast<std::size_t>(qTileBegin) * Format::kQBytesPerTile,
                Window::template qBytes<Phase>(), &shared.tmaBar[Phase]);
            if constexpr (Window::template sfBytes<Phase>() != 0U)
            {
                bulk1d(&shared.scaleCache[stageRow][0], cached.pointer.sfRecord, Window::template sfBytes<Phase>(),
                    &shared.tmaBar[Phase]);
            }
        }
    }
#else
    (void) shared;
    (void) bulkStageMask;
    (void) controls;
#endif
}

template <class RoutingPolicy>
__device__ __forceinline__ void stageNonBulkRowsSynchronously(
    typename RoutingPolicy::Shared const& shared, int activeRows, std::uint32_t bulkStageMask, int warp, int lane)
{
    using Format = typename RoutingPolicy::Format;
    int const stageRow = warp & (kStageRows - 1);
    if (stageRow >= activeRows || ((bulkStageMask >> stageRow) & 1U) != 0U)
    {
        return;
    }

    int const wave = warp >> 3;
    using Window = typename RoutingPolicy::Window;
    CachedRoute const cached = RoutingPolicy::stageRoute(shared, stageRow);
    asm volatile("" ::: "memory");
    if constexpr (Window::kPacked)
    {
        // The packed record has the SAME internal byte layout in global memory
        // and in packedCache (both tight stride kPackedRecordStrideBytes), so
        // the synchronous stage is a same-offset copy of every Q granule and SF
        // pack. cached.pointer.qRecord is the packed record base.
        std::uint8_t* rec = shared.packedCache + static_cast<std::size_t>(stageRow) * Format::kPackedRecordStrideBytes;
        std::uint8_t const* src = cached.pointer.qRecord;
#pragma unroll 1
        for (int stageTile = wave * 32 + lane; stageTile < kTilesPerToken; stageTile += 64)
        {
            int const qOff = Window::qByteOffset(stageTile);
            *reinterpret_cast<typename Format::QGranule*>(rec + qOff)
                = *reinterpret_cast<typename Format::QGranule const*>(src + qOff);
            if constexpr (Format::kHasScalePlane)
            {
                int const sfOff = Window::sfByteOffset(stageTile);
                *reinterpret_cast<std::uint64_t*>(rec + sfOff) = *reinterpret_cast<std::uint64_t const*>(src + sfOff);
            }
        }
    }
    else
    {
#pragma unroll 1
        for (int stageTile = wave * 32 + lane; stageTile < kTilesPerToken; stageTile += 64)
        {
            std::uint8_t const* qAddress
                = cached.pointer.qRecord + static_cast<std::size_t>(stageTile) * Format::kQBytesPerTile;
            shared.qCache[stageRow][stageTile] = *reinterpret_cast<typename Format::QGranule const*>(qAddress);
            if constexpr (Format::kHasScalePlane)
            {
                std::uint8_t const* sfAddress
                    = cached.pointer.sfRecord + static_cast<std::size_t>(stageTile) * Format::kSfBytesPerTile;
                shared.scaleCache[stageRow][stageTile] = *reinterpret_cast<std::uint64_t const*>(sfAddress);
            }
        }
    }
}

// Format-parameterized generic fallback (non-staged, reads peer memory
// directly). Mirrors decodeReduceAndStoreTile but sources the dtype-specific
// unpack from Format::decodeBlock16. Invalid routes (qRecord == nullptr)
// contribute zeros yet must still traverse reduceRoute8Frozen so the K8 warp
// shuffle stays convergent across all eight route positions.
template <class Window, class Format>
__device__ __noinline__ void decodeFallbackTile(
    CachedRoute const& cached, __nv_bfloat16* outputRecord, int tile, int blockInTile, int routePosition)
{
    std::uint64_t scalePack = 0;
    if constexpr (Format::kHasScalePlane)
    {
        // Packed records interleave the SF halves inside the qRecord buffer, so
        // the SF pack must be read at Window::sfByteOffset(tile); split-plane
        // reads the separate sfRecord at tile*kSfBytesPerTile. FullToken packed
        // reproduces the split-plane address exactly (qRecord + 3584 + tile*8 ==
        // sfRecord + tile*8), so the EP2 t94_packed output stays identical.
        if constexpr (Window::kPacked)
        {
            scalePack = loadScalePackOncePerRoutePacked<Window>(cached, tile, blockInTile, routePosition);
        }
        else
        {
            scalePack = loadScalePackOncePerRoute(cached, tile, blockInTile, routePosition);
        }
    }

    DecodedBlock16 decoded;
    if (cached.pointer.qRecord != nullptr)
    {
        // Packed Q for a tile is at the interleaved Window::qByteOffset (Half:
        // tiles>=56 live in the second phase slice; FullToken: tile*32 ==
        // linear). Split-plane keeps the byte-identical linear tile stride.
        std::uint8_t const* qBlock;
        if constexpr (Window::kPacked)
        {
            qBlock = cached.pointer.qRecord + Window::qByteOffset(tile) + blockInTile * Format::kQBytesPerBlock;
        }
        else
        {
            qBlock = cached.pointer.qRecord + static_cast<std::size_t>(tile) * Format::kQBytesPerTile
                + blockInTile * Format::kQBytesPerBlock;
        }
        decoded = Format::decodeBlock16(qBlock, scalePack, blockInTile);
    }
    else
    {
#pragma unroll
        for (int pair = 0; pair < kPairsPerBlock; ++pair)
        {
            decoded.pair[pair] = make_float2(0.0F, 0.0F);
        }
    }

#pragma unroll
    for (int batch = 0; batch < kPairsPerBlock / kPairsPerStore; ++batch)
    {
        uint4 words = make_uint4(0U, 0U, 0U, 0U);
#pragma unroll
        for (int pairInBatch = 0; pairInBatch < kPairsPerStore; ++pairInBatch)
        {
            int const pair = batch * kPairsPerStore + pairInBatch;
            float x = reduceRoute8Frozen(decoded.pair[pair].x, routePosition);
            float y = reduceRoute8Frozen(decoded.pair[pair].y, routePosition);
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

template <class RoutingPolicy>
__device__ __forceinline__ void decodeFallback(TileNvfp4CombinePointersV2 const& ptrs,
    typename RoutingPolicy::Shared const& shared, int ctaBase, int tokensInCta, int warp, int lane)
{
    using Format = typename RoutingPolicy::Format;
    using Window = typename RoutingPolicy::Window;
    int const blockInTile = lane / kTopK;
    int const routePosition = lane % kTopK;
#pragma unroll 1
    for (int tokenSlot = 0; tokenSlot < tokensInCta; ++tokenSlot)
    {
        CachedRoute const cached = RoutingPolicy::fallbackRoute(shared, tokenSlot, routePosition);
        asm volatile("" ::: "memory");
        __nv_bfloat16* outputRecord = ptrs.output + static_cast<std::size_t>(ctaBase + tokenSlot) * kHiddenElements;
        for (int tile = warp; tile < kTilesPerToken; tile += kWarpsPerCta)
        {
            decodeFallbackTile<Window, Format>(cached, outputRecord, tile, blockInTile, routePosition);
        }
    }
}

template <int Phase, class RoutingPolicy>
__device__ __forceinline__ void scheduleFastDecodePhase(TileNvfp4CombinePointersV2 const& ptrs,
    typename RoutingPolicy::Shared const& shared, int ctaBase, int tokensInCta,
    typename RoutingPolicy::DecodeState& state, int lane)
{
    using Window = typename RoutingPolicy::Window;
    int const spanCount = Window::template decodeSpanCount<Phase>(tokensInCta);
#pragma unroll
    for (int span = 0; span < Window::kDecodeSpans; ++span)
    {
        if (span >= spanCount)
        {
            break;
        }
        int const begin = Window::template decodeBegin<Phase>(span, tokensInCta);
        int const end = Window::template decodeEnd<Phase>(span, tokensInCta);
        int& work = state.work[span];
        while (work < begin)
        {
            work += kWarpsPerCta;
        }
        for (; work < end; work += kWarpsPerCta)
        {
            DecodeCoord const coord = Window::template decodeCoord<Phase>(span, work);
            RoutingPolicy::template decodeFastPhase<Phase>(ptrs, shared, ctaBase, coord.tokenSlot, coord.unit, lane);
        }
    }
}

// The single TMA family body. EP, token batching, route mapping, reduction
// collapse, and window geometry are compile-time-specialized.
template <int EP, int TOKENS_PER_CTA, class RoutingPolicy>
__device__ __forceinline__ void tileNvfp4CombineTma(TileNvfp4CombinePointersV2 const& ptrs, int localNumTokens,
    int epRank, int epSize, int runtimeMaxTokensPerRank, int graphStaticMaxTokensPerRank,
    TmaRuntimeControls const& controls, typename RoutingPolicy::Shared const& shared)
{
    using Window = typename RoutingPolicy::Window;
    using Format = typename RoutingPolicy::Format;
    static_assert(EP == 2 || EP == 4 || EP == 8 || EP == 32, "v1 supports exactly EP {2,4,8,32}");
    static_assert(TOKENS_PER_CTA == RoutingPolicy::kTokensPerCta);
    static_assert(
        TOKENS_PER_CTA == Window::kTokensPerCta, "kernel shell, routing policy, and window disagree on tokens/CTA");

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
    if (threadIdx.x == 0)
    {
        mbarInit(&shared.tmaBar[0], 1);
        mbarInit(&shared.tmaBar[1], 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
#endif

    RoutingPolicy::buildRouteCacheAndMasks(ptrs, localNumTokens, epRank, epSize, runtimeMaxTokensPerRank,
        graphStaticMaxTokensPerRank, controls.bulkPeerMaskLo, controls.bulkPeerMaskHi, shared);
    publishAndAcquireReadiness(ptrs, epRank, epSize);

    __syncthreads();
    asm volatile("fence.acquire.sys;" ::: "memory");
    if (localNumTokens == 0)
    {
        return;
    }

    int const ctaBase = static_cast<int>(blockIdx.x) * TOKENS_PER_CTA;
    int const remainingTokens = localNumTokens - ctaBase;
    int const tokensInCta = remainingTokens < TOKENS_PER_CTA ? remainingTokens : TOKENS_PER_CTA;
    if (tokensInCta <= 0)
    {
        return;
    }

    int const warp = static_cast<int>(threadIdx.x) >> 5;
    int const lane = static_cast<int>(threadIdx.x) & 31;
    if (!RoutingPolicy::fastEligible(shared, tokensInCta))
    {
        decodeFallback<RoutingPolicy>(ptrs, shared, ctaBase, tokensInCta, warp, lane);
        return;
    }

    int const activeRows = Window::activeRows(tokensInCta);
    std::uint32_t const activeRowsMask = Window::activeRowsMask(tokensInCta);
    std::uint32_t const bulkStageMask = RoutingPolicy::bulkStageMask(shared) & activeRowsMask;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    if constexpr (Format::kSupportsPoison)
    {
        if (controls.poisonMode != 0)
        {
            if constexpr (Window::kPacked)
            {
                stampPoisonAllStageRowsPacked<Window, Format>(shared.packedCache, warp, lane, controls.poisonEpoch);
            }
            else
            {
                stampPoisonAllStageRows(shared.scaleCache, shared.qCache, warp, lane, controls.poisonEpoch);
            }
            __syncthreads();
        }
    }

    if (threadIdx.x == 0 && bulkStageMask != 0U)
    {
        asm volatile("fence.proxy.async;" ::: "memory");
        spinCycles(static_cast<long long>(controls.delayedDmaCycles));
        issueTmaPhase<0, RoutingPolicy>(shared, bulkStageMask, controls);
        issueTmaPhase<1, RoutingPolicy>(shared, bulkStageMask, controls);
        if (controls.debugBarTimestamps != nullptr)
        {
            controls.debugBarTimestamps[static_cast<std::size_t>(blockIdx.x) * 4]
                = static_cast<unsigned long long>(clock64());
        }
    }

    if (bulkStageMask != activeRowsMask)
    {
        stageNonBulkRowsSynchronously<RoutingPolicy>(shared, activeRows, bulkStageMask, warp, lane);
    }
    __syncthreads();

    typename RoutingPolicy::DecodeState decodeState = RoutingPolicy::makeDecodeState(warp);
    std::uint32_t const phase0Mask = Window::template phaseRowsMask<0>(bulkStageMask);
    if (phase0Mask != 0U)
    {
        mbarWaitParityAllConsumers(&shared.tmaBar[0], 0U);
        if (controls.debugBarTimestamps != nullptr && threadIdx.x == 0)
        {
            controls.debugBarTimestamps[static_cast<std::size_t>(blockIdx.x) * 4 + 1]
                = static_cast<unsigned long long>(clock64());
        }
    }
    scheduleFastDecodePhase<0, RoutingPolicy>(ptrs, shared, ctaBase, tokensInCta, decodeState, lane);

    std::uint32_t const phase1Mask = Window::template phaseRowsMask<1>(bulkStageMask);
    if (phase1Mask != 0U)
    {
        mbarWaitParityAllConsumers(&shared.tmaBar[1], 0U);
        if (controls.debugBarTimestamps != nullptr && threadIdx.x == 0)
        {
            controls.debugBarTimestamps[static_cast<std::size_t>(blockIdx.x) * 4 + 2]
                = static_cast<unsigned long long>(clock64());
        }
    }
    scheduleFastDecodePhase<1, RoutingPolicy>(ptrs, shared, ctaBase, tokensInCta, decodeState, lane);

    if (controls.debugBarTimestamps != nullptr && threadIdx.x == 0 && bulkStageMask != 0U)
    {
        controls.debugBarTimestamps[static_cast<std::size_t>(blockIdx.x) * 4 + 3]
            = static_cast<unsigned long long>(clock64());
    }
#else
    stageNonBulkRowsSynchronously<RoutingPolicy>(shared, activeRows, 0U, warp, lane);
    __syncthreads();
    typename RoutingPolicy::DecodeState decodeState = RoutingPolicy::makeDecodeState(warp);
    scheduleFastDecodePhase<0, RoutingPolicy>(ptrs, shared, ctaBase, tokensInCta, decodeState, lane);
    scheduleFastDecodePhase<1, RoutingPolicy>(ptrs, shared, ctaBase, tokensInCta, decodeState, lane);
#endif
}

} // namespace kernels::moe_comm::tile_nvfp4_tma

TRTLLM_NAMESPACE_END
