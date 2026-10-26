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

TRTLLM_NAMESPACE_BEGIN

namespace kernels::moe_comm::tile_nvfp4_tma
{

constexpr long long kSkewCycleUnit = 512;
constexpr std::uint32_t kQNibbleFloor = 0x11111111U;
constexpr std::uint32_t kSfLowFloor = 0x01010101U;
constexpr std::uint32_t kSfHighQNaN = 0x7FC00000U;
constexpr std::uint32_t kEpochStride = 0x9E3779B9U;

static __device__ __forceinline__ void spinCycles(long long cycles)
{
    if (cycles <= 0)
    {
        return;
    }
    long long const start = static_cast<long long>(clock64());
    while (static_cast<long long>(clock64()) - start < cycles)
    {
        asm volatile("" ::: "memory");
    }
}

// All current TMA policies expose exactly eight stage rows. Keeping that
// invariant explicit avoids a runtime strategy branch and preserves the
// skew schedule.
static __device__ __forceinline__ int skewRank(int stageRow, int skewProfile, int stragglerRow, unsigned epoch)
{
    switch (skewProfile)
    {
    case 1: return stageRow;
    case 2: return (kStageRows - 1) - stageRow;
    case 3: return stageRow == stragglerRow ? kStageRows : 0;
    case 4:
    {
        unsigned h = (static_cast<unsigned>(stageRow) * 2654435761U) ^ (epoch * 40503U);
        return static_cast<int>(h & 7U);
    }
    default: return 0;
    }
}

static __device__ __forceinline__ void stampPoison(
    B256* qDst, std::uint64_t* sfDst, int stageRow, int globalTile, unsigned epoch)
{
    unsigned const cell
        = (static_cast<unsigned>(stageRow) * static_cast<unsigned>(kTilesPerToken) + static_cast<unsigned>(globalTile))
            * 4U
        + epoch * kEpochStride;
    unsigned const s0 = (cell + 0U) | kQNibbleFloor;
    unsigned const s1 = (cell + 1U) | kQNibbleFloor;
    unsigned const s2 = (cell + 2U) | kQNibbleFloor;
    unsigned const s3 = (cell + 3U) | kQNibbleFloor;
    qDst->a = make_uint4(s0, s0, s1, s1);
    qDst->b = make_uint4(s2, s2, s3, s3);

    unsigned const sfLow
        = (static_cast<unsigned>(stageRow) * static_cast<unsigned>(kTilesPerToken) + static_cast<unsigned>(globalTile))
        | epoch | kSfLowFloor;
    *sfDst = static_cast<std::uint64_t>(sfLow) | (static_cast<std::uint64_t>(kSfHighQNaN) << 32);
}

static __device__ __forceinline__ void stampPoisonAllStageRows(
    std::uint64_t (*scaleCache)[kScaleCacheStride], B256 (*qCache)[kQCacheStride], int warp, int lane, unsigned epoch)
{
    int const stageRow = warp & (kStageRows - 1);
    int const wave = warp >> 3;
#pragma unroll 1
    for (int stageTile = wave * 32 + lane; stageTile < kTilesPerToken; stageTile += 64)
    {
        stampPoison(&qCache[stageRow][stageTile], &scaleCache[stageRow][stageTile], stageRow, stageTile, epoch);
    }
}

// Packed-layout poison stamper (NVFP4-only). Same per-(stageRow,tile) monotone
// Q stamp + quiet-NaN SF-high stamp as stampPoison, but written into the ONE
// contiguous packed record via the window's byte offsets: the 32-B Q granule at
// rec + Window::qByteOffset(tile) and the 8-B scale pack at rec +
// Window::sfByteOffset(tile). Window carries the FullToken/Half offset geometry.
template <class Window>
static __device__ __forceinline__ void stampPoisonPackedRecord(
    std::uint8_t* rec, int stageRow, int globalTile, unsigned epoch)
{
    unsigned const cell
        = (static_cast<unsigned>(stageRow) * static_cast<unsigned>(kTilesPerToken) + static_cast<unsigned>(globalTile))
            * 4U
        + epoch * kEpochStride;
    unsigned const s0 = (cell + 0U) | kQNibbleFloor;
    unsigned const s1 = (cell + 1U) | kQNibbleFloor;
    unsigned const s2 = (cell + 2U) | kQNibbleFloor;
    unsigned const s3 = (cell + 3U) | kQNibbleFloor;
    B256* qDst = reinterpret_cast<B256*>(rec + Window::qByteOffset(globalTile));
    qDst->a = make_uint4(s0, s0, s1, s1);
    qDst->b = make_uint4(s2, s2, s3, s3);

    unsigned const sfLow
        = (static_cast<unsigned>(stageRow) * static_cast<unsigned>(kTilesPerToken) + static_cast<unsigned>(globalTile))
        | epoch | kSfLowFloor;
    *reinterpret_cast<std::uint64_t*>(rec + Window::sfByteOffset(globalTile))
        = static_cast<std::uint64_t>(sfLow) | (static_cast<std::uint64_t>(kSfHighQNaN) << 32);
}

template <class Window, class Format>
static __device__ __forceinline__ void stampPoisonAllStageRowsPacked(
    std::uint8_t* packedCache, int warp, int lane, unsigned epoch)
{
    int const stageRow = warp & (kStageRows - 1);
    int const wave = warp >> 3;
    std::uint8_t* rec = packedCache + static_cast<std::size_t>(stageRow) * Format::kPackedRecordStrideBytes;
#pragma unroll 1
    for (int stageTile = wave * 32 + lane; stageTile < kTilesPerToken; stageTile += 64)
    {
        stampPoisonPackedRecord<Window>(rec, stageRow, stageTile, epoch);
    }
}

// This block is byte-identical across the per-EP combine variants.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

static __device__ __forceinline__ unsigned smemAddr(void const* p)
{
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

static __device__ __forceinline__ void mbarInit(std::uint64_t* mbar, unsigned count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smemAddr(mbar)), "r"(count));
}

static __device__ __forceinline__ void mbarArriveExpectTx(std::uint64_t* mbar, unsigned txBytes)
{
    std::uint64_t state;
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], %2;"
                 : "=l"(state)
                 : "r"(smemAddr(mbar)), "r"(txBytes));
    (void) state;
}

static __device__ __forceinline__ void bulk1d(void* dstSmem, void const* gSrc, unsigned bytes, std::uint64_t* mbar)
{
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];" ::"r"(smemAddr(dstSmem)),
        "l"(gSrc), "r"(bytes), "r"(smemAddr(mbar)));
}

static __device__ __forceinline__ void mbarWaitParityAllConsumers(std::uint64_t* mbar, unsigned phaseParity)
{
    std::uint64_t const start = clock64();
    unsigned done = 0;
    do
    {
        asm volatile(
            "{.reg .pred p;"
            " mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;"
            " selp.b32 %0, 1, 0, p;}"
            : "=r"(done)
            : "r"(smemAddr(mbar)), "r"(phaseParity)
            : "memory");
        if (done == 0U)
        {
            __nanosleep(64);
            if (__builtin_expect((clock64() - start) > kTimeoutCycles, 0))
            {
                asm volatile("trap;");
                return;
            }
        }
    } while (done == 0U);
}

#endif // __CUDA_ARCH__ >= 900

} // namespace kernels::moe_comm::tile_nvfp4_tma

TRTLLM_NAMESPACE_END
