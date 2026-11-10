/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

// Kernels and launchers of the sentinel transport of the NVLink one-sided MoE all-to-all. The
// protocol is described in moeAlltoAllSentinel.h.

#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/moe/communication/moeAlltoAllSentinel.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

#ifndef DISABLE_TIMEOUT
#define DISABLE_TIMEOUT 0
#endif

// Device code for the 10.x targets carries the kernels; other targets get kernels that trap. TRT-LLM
// builds sm_10x as the sm_100f family by default, so usually only the __CUDA_ARCH__ == 1000 branch is
// compiled, and GPUs of compute capability 10.3 and 10.7 run that image; 1030 and 1070 apply to
// builds that list those architectures (e.g. --cuda_architectures "107-real"). Which GPUs may use the
// transport is decided at run time by moeA2ASentinelKernelsAvailable(), not here. (FlashInfer
// builds native sm_107a for these kernels, noting that the driver rejects the sm_100f fallback on
// SM107; if it does, the availability probe below fails and the transport stays off.)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070)
#define MOE_A2A_SENTINEL_COMPILE 1
#else
#define MOE_A2A_SENTINEL_COMPILE 0
#endif

TRTLLM_NAMESPACE_BEGIN

namespace kernels::moe_comm::sentinel
{

namespace
{

#if MOE_A2A_SENTINEL_COMPILE
// ================================================================================ PTX helpers
__device__ __forceinline__ uint4 loadVolatile(void const* ptr)
{
    uint4 v;
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(ptr)
                 : "memory");
    return v;
}

__device__ __forceinline__ int loadVolatileInt(int const* ptr)
{
    int v;
    asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(v) : "l"(ptr) : "memory");
    return v;
}

__device__ __forceinline__ void storeVolatile(void* ptr, uint4 v)
{
    asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}

// Programmatic dependent launch: every kernel waits for its predecessor before it touches a
// buffer or the epoch, and lets its successor launch once its own stores are issued.
__device__ __forceinline__ void waitForPreviousKernel()
{
    asm volatile("griddepcontrol.wait;" ::: "memory");
}

__device__ __forceinline__ void allowNextKernel()
{
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

__device__ __forceinline__ uint4 splat(uint32_t word)
{
    return make_uint4(word, word, word, word);
}

__device__ __forceinline__ bool hasArrived(uint4 v, uint32_t sentinelWord)
{
    return v.x != sentinelWord && v.y != sentinelWord && v.z != sentinelWord && v.w != sentinelWord;
}

// Row (source, token) of the landing regions: a token slot's rows from every source are adjacent.
__device__ __forceinline__ int64_t landingRow(int epSize, int source, int token)
{
    return static_cast<int64_t>(token) * epSize + source;
}

// Polls the landing vectors whose bits are set in `pending` until each has arrived, every load in
// flight before any check; address(i) is vector i's address, and beforeEachSpin() runs at the
// start of every spin (e.g. to re-read address(i)'s inputs from shared memory, all before the
// first load). Returns the bits still pending when the wait timed out (0: all arrived).
template <int NUM_VECTORS, typename BeforeEachSpin, typename Address>
__device__ __forceinline__ uint32_t pollUntilArrived(uint32_t pending, uint32_t sentinelWord,
    BeforeEachSpin const& beforeEachSpin, Address const& address, uint4 (&vectors)[NUM_VECTORS], long long waitStart,
    int64_t timeoutCycles)
{
    for (int spins = 1; pending; ++spins)
    {
        beforeEachSpin();
#pragma unroll
        for (int i = 0; i < NUM_VECTORS; ++i)
        {
            if ((pending >> i) & 1U)
            {
                vectors[i] = loadVolatile(address(i));
            }
        }
#pragma unroll
        for (int i = 0; i < NUM_VECTORS; ++i)
        {
            if (((pending >> i) & 1U) && hasArrived(vectors[i], sentinelWord))
            {
                pending &= ~(1U << i);
            }
        }
        // A -DDISABLE_TIMEOUT=1 debug build waits forever, as the fence kernels do.
        if (!DISABLE_TIMEOUT && pending && spins % 64 == 0 && clock64() - waitStart > timeoutCycles)
        {
            break;
        }
    }
    return pending;
}

// A peer that never arrives: report once per launch, then trap (the CUDA context is lost, as in
// the fence kernels). [[noreturn]]: callers keep no register live
// across the call.
[[noreturn]] static __device__ __noinline__ void reportAndTrap(int32_t* roundState, int epRank, char const* what)
{
    if (atomicExch(roundState + kTimeoutReported, 1) == 0)
    {
        printf("sentinel transport: rank %d %s\n", epRank, what);
    }
    __threadfence_system();
    __trap();
}

// The rank mask is passed by value, as the fence kernels' is. Nothing is stored to or polled
// from a masked rank; a route to it is dropped like a repeated one.
template <bool ENABLE_RANK_MASK>
__device__ __forceinline__ bool isRankAlive(uint64_t activeRankMask, int rank)
{
    if constexpr (ENABLE_RANK_MASK)
    {
        return (activeRankMask >> rank) & 1ULL;
    }
    else
    {
        return true;
    }
}

// Removes the lowest rank from a rank mask and returns it: successive calls walk the ranks in
// ascending order, at a cost per rank in the mask rather than per rank of the EP group.
__device__ __forceinline__ int popLowestRank(uint64_t& ranks)
{
    int const rank = __ffsll(static_cast<long long>(ranks)) - 1;
    ranks &= ranks - 1;
    return rank;
}

// Row vector `vec` (top-k slots 4 vec .. 4 vec + 3) into a receive row of topK words.
__device__ __forceinline__ void storeTopkWords(uint32_t* row, int topK, int vec, uint4 v)
{
    uint32_t const words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int w = 0; w < 4; ++w)
    {
        if (4 * vec + w < topK)
        {
            row[4 * vec + w] = words[w];
        }
    }
}

// Routing weights of row vector `vec` into a receive row of topK weights. bf16 weights travel as
// the fp32 words of their values (bits << 16), so the high halves are the bf16 values exactly.
__device__ __forceinline__ void storeTopkWeights(DispatchParams const& p, int64_t recvRow, int topK, int vec, uint4 v)
{
    if (!p.weightsBf16)
    {
        storeTopkWords(p.recvTopkWeights + recvRow * topK, topK, vec, v);
        return;
    }
    uint16_t* const row = reinterpret_cast<uint16_t*>(p.recvTopkWeights) + recvRow * topK;
    uint32_t const words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int w = 0; w < 4; ++w)
    {
        if (4 * vec + w < topK)
        {
            row[4 * vec + w] = static_cast<uint16_t>(words[w] >> 16);
        }
    }
}

// ============================================== sender-side rewrites: no wire word is a sentinel
// BF16 0xFFFFFFFF (two NaNs) -> 0x7FFF7FFF; FP8 data and fp32 / e4m3 scale or weight words
// 0xFFFFFFFF -> 0x7FFFFFFF (NaN stays NaN); an id equal to kIdsSentinel (an invalid id) ->
// invalidExpertId. MXFP8 scales travel as 16-bit lanes 0x00XX, which never match.
__device__ __forceinline__ uint32_t canonicalizeBf16x2(uint32_t word)
{
    return word == kDataSentinel ? 0x7FFF7FFFU : word;
}

__device__ __forceinline__ uint32_t canonicalizeWord(uint32_t word)
{
    return word == kDataSentinel ? 0x7FFFFFFFU : word;
}

__device__ __forceinline__ uint32_t canonicalizeExpertId(uint32_t id, uint32_t invalidId)
{
    return id == kIdsSentinel ? invalidId : id;
}

// NVFP4: every -0 nibble (1000b) becomes +0; the wire word is this XOR kNvfp4WireXor.
__device__ __forceinline__ uint32_t canonicalizeFp4x8(uint32_t word)
{
    uint32_t const magnitude = word & 0x77777777U;
    uint32_t const nonzero = (magnitude | (magnitude >> 1) | (magnitude >> 2)) & 0x11111111U;
    return word & (0x77777777U | (nonzero << 3));
}

// Wire <-> raw NVFP4 vector (an involution).
__device__ __forceinline__ uint4 nvfp4WireXor(uint4 v)
{
    return make_uint4(v.x ^ kNvfp4WireXor, v.y ^ kNvfp4WireXor, v.z ^ kNvfp4WireXor, v.w ^ kNvfp4WireXor);
}

// Combine partials: every bf16 lane 0xFFFF (a NaN) -> 0x7FFF.
__device__ __forceinline__ uint32_t canonicalizeBf16Lanes(uint32_t word)
{
    if ((word & 0xFFFFU) == 0xFFFFU)
    {
        word = (word & 0xFFFF0000U) | 0x7FFFU;
    }
    if ((word >> 16) == 0xFFFFU)
    {
        word = (word & 0x0000FFFFU) | 0x7FFF0000U;
    }
    return word;
}

__device__ __forceinline__ uint4 canonicalizePartial(uint4 v)
{
    return make_uint4(
        canonicalizeBf16Lanes(v.x), canonicalizeBf16Lanes(v.y), canonicalizeBf16Lanes(v.z), canonicalizeBf16Lanes(v.w));
}

// MXFP8 scales: 8 UE8M0 bytes (raw) <-> 8 16-bit lanes 0x00XX (wire).
__device__ __forceinline__ uint4 mxScalesToWire(uint2 raw)
{
    return make_uint4((raw.x & 0xFFU) | ((raw.x & 0xFF00U) << 8), ((raw.x >> 16) & 0xFFU) | ((raw.x >> 8) & 0xFF0000U),
        (raw.y & 0xFFU) | ((raw.y & 0xFF00U) << 8), ((raw.y >> 16) & 0xFFU) | ((raw.y >> 8) & 0xFF0000U));
}

__device__ __forceinline__ uint2 mxScalesFromWire(uint4 wire)
{
    return make_uint2(
        (wire.x & 0xFFU) | ((wire.x >> 8) & 0xFF00U) | ((wire.y & 0xFFU) << 16) | (((wire.y >> 16) & 0xFFU) << 24),
        (wire.z & 0xFFU) | ((wire.z >> 8) & 0xFF00U) | ((wire.w & 0xFFU) << 16) | (((wire.w >> 16) & 0xFFU) << 24));
}

// ============================================================================ dispatch phases
// D1, warp 0, lane k = top-k slot k. Returns the token's target-rank mask (0 for a padding slot,
// token >= localNumTokens). Stages the token's wire ids row for D3 in sIdsRowWords
// (canonical ids, invalidExpertId past topK) and, if topkTargetRanksRow is set, writes the
// rank of every slot at its first occurrence (else -1).
template <int TOP_K_CAPACITY, bool FIXED_TOP_K, bool ENABLE_RANK_MASK>
__device__ __forceinline__ uint64_t routeToken(
    DispatchParams const& p, int token, int lane, uint32_t* sIdsRowWords, int32_t* topkTargetRanksRow)
{
    int const topK = FIXED_TOP_K ? TOP_K_CAPACITY : p.topK;
    int expertId = -1, rank = -1;
    if (token < p.localNumTokens && lane < topK)
    {
        expertId = p.tokenSelectedExperts[static_cast<int64_t>(token) * topK + lane];
        // Experts split evenly (checked on the host). The fence kernels' compute_target_rank_id also
        // handles uneven splits, but its extra branch measurably slows this dispatch at decode sizes.
        rank = (expertId >= 0 && expertId < p.numExperts) ? expertId / p.expertsPerRank : -1;
    }
    uint32_t const sameRankLanes = __match_any_sync(0xFFFFFFFFU, rank);
    int targetRank = (rank >= 0 && __ffs(sameRankLanes) - 1 == lane) ? rank : -1;
    uint32_t const lowRanks = __reduce_or_sync(0xFFFFFFFFU, rank >= 0 && rank < 32 ? 1U << rank : 0U);
    uint32_t const highRanks = __reduce_or_sync(0xFFFFFFFFU, rank >= 32 ? 1U << (rank - 32) : 0U);
    uint64_t targetMask = static_cast<uint64_t>(lowRanks) | static_cast<uint64_t>(highRanks) << 32;
    if constexpr (ENABLE_RANK_MASK)
    {
        targetMask &= p.activeRankMask;
        if (targetRank >= 0 && !isRankAlive<true>(p.activeRankMask, targetRank))
        {
            targetRank = -1;
        }
    }
    uint32_t const invalidId = static_cast<uint32_t>(p.invalidExpertId);
    if (lane < 4 * topkRowVectors(TOP_K_CAPACITY))
    {
        sIdsRowWords[lane] = lane < topK ? canonicalizeExpertId(static_cast<uint32_t>(expertId), invalidId) : invalidId;
    }
    if (lane < topK && topkTargetRanksRow != nullptr && token < p.localNumTokens)
    {
        topkTargetRanksRow[lane] = targetRank;
    }
    return targetMask;
}

// D2 / D5: where vector `vec` of a wire row lives (payload vectors first, then scale vectors).
struct WireVector
{
    bool isPayload;
    int64_t regionOffset; // the landing region
    int rowBytes;         // landing row
    int offsetInRow;      // bytes
};

__device__ __forceinline__ WireVector locateWireVector(DispatchParams const& p, int vec)
{
    if (vec < p.payloadVectors)
    {
        return {true, p.landing.payloadOffset, p.payloadRowBytes, vec * 16};
    }
    return {false, p.landing.scaleOffset, p.wireScaleRowBytes, (vec - p.payloadVectors) * 16};
}

// D2: vector `vec` of the token's wire row, canonicalised.
template <PayloadKind PAYLOAD_KIND>
__device__ __forceinline__ uint4 loadWireVector(DispatchParams const& p, int token, WireVector const& wv)
{
    if (wv.isPayload)
    {
        uint4 const v = *reinterpret_cast<uint4 const*>(
            p.hiddenStates + static_cast<int64_t>(token) * p.payloadRowBytes + wv.offsetInRow);
        if constexpr (PAYLOAD_KIND == PayloadKind::kBf16)
        {
            return make_uint4(
                canonicalizeBf16x2(v.x), canonicalizeBf16x2(v.y), canonicalizeBf16x2(v.z), canonicalizeBf16x2(v.w));
        }
        else if constexpr (PAYLOAD_KIND == PayloadKind::kFp8)
        {
            return make_uint4(
                canonicalizeWord(v.x), canonicalizeWord(v.y), canonicalizeWord(v.z), canonicalizeWord(v.w));
        }
        else
        {
            return nvfp4WireXor(make_uint4(
                canonicalizeFp4x8(v.x), canonicalizeFp4x8(v.y), canonicalizeFp4x8(v.z), canonicalizeFp4x8(v.w)));
        }
    }
    uint8_t const* scaleRow = p.hiddenStatesScale + static_cast<int64_t>(token) * p.scaleRowBytes;
    if (p.mxScales) // a wire vector = 8 raw UE8M0 bytes
    {
        return mxScalesToWire(*reinterpret_cast<uint2 const*>(scaleRow + wv.offsetInRow / 2));
    }
    uint4 const v = *reinterpret_cast<uint4 const*>(scaleRow + wv.offsetInRow);
    return make_uint4(canonicalizeWord(v.x), canonicalizeWord(v.y), canonicalizeWord(v.z), canonicalizeWord(v.w));
}

// D2 / D5: store a wire vector (own or received) into receive row `recvRow`, in raw format.
template <PayloadKind PAYLOAD_KIND>
__device__ __forceinline__ void storeReceivedVector(
    DispatchParams const& p, int64_t recvRow, WireVector const& wv, uint4 wire)
{
    if (wv.isPayload)
    {
        *reinterpret_cast<uint4*>(p.recvHiddenStates + recvRow * p.payloadRowBytes + wv.offsetInRow)
            = PAYLOAD_KIND == PayloadKind::kNvfp4 ? nvfp4WireXor(wire) : wire;
        return;
    }
    uint8_t* scaleRow = p.recvHiddenStatesScale + recvRow * p.scaleRowBytes;
    if (p.mxScales)
    {
        *reinterpret_cast<uint2*>(scaleRow + wv.offsetInRow / 2) = mxScalesFromWire(wire);
    }
    else
    {
        *reinterpret_cast<uint4*>(scaleRow + wv.offsetInRow) = wire;
    }
}

// D2: store `wire` into the landing row (epRank, token) of every target rank; this rank's own
// copy goes straight into its receive row `recvRow`.
template <PayloadKind PAYLOAD_KIND>
__device__ __forceinline__ void pushVector(
    DispatchParams const& p, uint64_t targetMask, int token, int64_t recvRow, WireVector const& wv, uint4 wire)
{
    int64_t const offset = wv.regionOffset + landingRow(p.epSize, p.epRank, token) * wv.rowBytes + wv.offsetInRow;
    for (uint64_t targets = targetMask; targets;)
    {
        int const target = popLowestRank(targets);
        if (target == p.epRank)
        {
            storeReceivedVector<PAYLOAD_KIND>(p, recvRow, wv, wire);
        }
        else
        {
            storeVolatile(p.rankWorkspaces[target] + offset, wire);
        }
    }
}

// D3, chunk-0 block: the ids row (epRank, token) goes to EVERY alive peer (all invalidExpertId
// where the peer is no target; it still signals arrival), the weights row to the targets only;
// own rows go straight to the receive buffers. The (peer, vector) items are spread over the
// block's threads. sIdsRow / sWeightsRow: the rows staged in D1.
template <bool ENABLE_RANK_MASK>
__device__ __forceinline__ void pushIdsAndWeights(DispatchParams const& p, int topK, int idsRowVectors,
    uint64_t targetMask, int token, int64_t recvRow, int64_t idsSetOffset, uint4 const* sIdsRow,
    uint4 const* sWeightsRow)
{
    uint4 const noTargetIds = splat(static_cast<uint32_t>(p.invalidExpertId));
    int64_t const offset = landingRow(p.epSize, p.epRank, token) * (16 * idsRowVectors); // ids / weights row
    for (int item = threadIdx.x; item < p.epSize * idsRowVectors; item += kNumThreads)
    {
        int const peer = item / idsRowVectors, vec = item % idsRowVectors;
        bool const isTarget = (targetMask >> peer) & 1ULL;
        uint4 const ids = isTarget ? sIdsRow[vec] : noTargetIds;
        bool const sendsWeights = isTarget && p.hasTopkWeights;
        if (peer == p.epRank)
        {
            storeTopkWords(reinterpret_cast<uint32_t*>(p.recvTopkIds) + recvRow * topK, topK, vec, ids);
            if (sendsWeights)
            {
                storeTopkWeights(p, recvRow, topK, vec, sWeightsRow[vec]);
            }
            continue;
        }
        if (!isRankAlive<ENABLE_RANK_MASK>(p.activeRankMask, peer))
        {
            continue;
        }
        uint8_t* const peerWorkspace = p.rankWorkspaces[peer] + offset + vec * 16;
        storeVolatile(peerWorkspace + p.landing.idsOffset + idsSetOffset, ids);
        if (sendsWeights)
        {
            storeVolatile(peerWorkspace + p.landing.weightsOffset, sWeightsRow[vec]);
        }
    }
}

// D4: record one received ids vector (source, vec): a source with any id other than
// invalidExpertId sent a token; the chunk-0 block copies the ids out.
__device__ __forceinline__ void acceptIdsVector(
    DispatchParams const& p, int topK, int token, int chunk, int source, int vec, uint4 ids, uint64_t& sRecvSourceMask)
{
    uint32_t const invalidId = static_cast<uint32_t>(p.invalidExpertId);
    if (ids.x != invalidId || ids.y != invalidId || ids.z != invalidId || ids.w != invalidId)
    {
        atomicOr(reinterpret_cast<unsigned long long*>(&sRecvSourceMask), 1ULL << source);
    }
    if (chunk != 0)
    {
        return;
    }
    int64_t const recvRow = static_cast<int64_t>(source) * p.maxTokensPerRank + token;
    storeTopkWords(reinterpret_cast<uint32_t*>(p.recvTopkIds) + recvRow * topK, topK, vec, ids);
}

// D4, a masked source: never polled, its row holds invalid ids.
__device__ __forceinline__ void rejectMaskedSource(
    DispatchParams const& p, int topK, int token, int chunk, int source, int vec)
{
    if (chunk != 0)
    {
        return;
    }
    int64_t const recvRow = static_cast<int64_t>(source) * p.maxTokensPerRank + token;
    storeTopkWords(reinterpret_cast<uint32_t*>(p.recvTopkIds) + recvRow * topK, topK, vec,
        splat(static_cast<uint32_t>(p.invalidExpertId)));
}

// D4: poll the ids rows (source, token) of every peer: ITEMS_PER_THREAD ids vectors (items) per
// thread, epSize * idsRowVectors items in all.
template <int ITEMS_PER_THREAD, bool ENABLE_RANK_MASK>
__device__ __forceinline__ void pollIdsRows(DispatchParams const& p, int topK, int idsRowVectors, int token, int chunk,
    int64_t idsSetOffset, long long waitStart, uint64_t& sRecvSourceMask, bool& timedOut)
{
    uint8_t const* idsRows = p.rankWorkspaces[p.epRank] + p.landing.idsOffset + idsSetOffset;
    int const numItems = p.epSize * idsRowVectors; // item = (source, vec)
    int64_t const idsRowBytes = 16 * idsRowVectors;
    uint32_t pending = 0;
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        int const item = threadIdx.x + i * kNumThreads;
        int const source = item / idsRowVectors;
        if (item >= numItems || source == p.epRank)
        {
            continue;
        }
        if (!isRankAlive<ENABLE_RANK_MASK>(p.activeRankMask, source))
        {
            rejectMaskedSource(p, topK, token, chunk, source, item % idsRowVectors);
            continue;
        }
        pending |= 1U << i;
    }
    uint32_t const polled = pending;
    uint4 ids[ITEMS_PER_THREAD];
    auto address = [&](int i)
    {
        int const item = threadIdx.x + i * kNumThreads;
        return idsRows + landingRow(p.epSize, item / idsRowVectors, token) * idsRowBytes + (item % idsRowVectors) * 16;
    };
    pending = pollUntilArrived(
        pending, kIdsSentinel, [] {}, address, ids, waitStart, p.timeoutCycles);
    timedOut = pending != 0;
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        if (!(((polled & ~pending) >> i) & 1U))
        {
            continue;
        }
        int const item = threadIdx.x + i * kNumThreads;
        acceptIdsVector(p, topK, token, chunk, item / idsRowVectors, item % idsRowVectors, ids[i], sRecvSourceMask);
    }
}

// D4 -> D5, warp 0: the ranks of sourceMask in ascending order into sSources[0, n); lane 0
// stores n.
__device__ __forceinline__ void listSources(uint64_t sourceMask, int* sSources, int* sNumSources)
{
    int const lane = threadIdx.x;
    uint32_t const low = static_cast<uint32_t>(sourceMask), high = sourceMask >> 32;
    uint32_t const below = (1U << lane) - 1U;
    if ((low >> lane) & 1U)
    {
        sSources[__popc(low & below)] = lane;
    }
    if ((high >> lane) & 1U)
    {
        sSources[__popc(low) + __popc(high & below)] = 32 + lane;
    }
    if (lane == 0)
    {
        *sNumSources = __popc(low) + __popc(high);
    }
}

// D5 / D6: poll, copy out (wire -> raw) and re-arm this thread's wire vector of every source in
// sSources[0, numSources), 8 sources per round with every load in flight: n sources take
// ceil(n / 8) rounds whatever their ranks. A round's sources are re-read from shared memory before
// every spin's loads and before its copy-out (holding them across the round would exceed the
// dispatch's register bound). A thread
// with pollsWeights (chunk 0, t < 8 * idsRowVectors) also polls, as the round's vector 8,
// weights vector t % idsRowVectors of the round's source t / idsRowVectors.
template <PayloadKind PAYLOAD_KIND>
__device__ __forceinline__ void receiveVectors(DispatchParams const& p, int topK, int idsRowVectors, int token, int vec,
    bool pollsWeights, int const* sSources, int numSources, long long waitStart, bool& timedOut)
{
    constexpr int kWeights = 8; // the round's vector 8: this thread's weights vector
    bool const hasVector = vec < p.payloadVectors + p.wireScaleVectors;
    WireVector const wv = locateWireVector(p, vec);
    uint8_t* const landing = p.rankWorkspaces[p.epRank];
    uint8_t* const vectorColumn = landing + wv.regionOffset + wv.offsetInRow;
    int const weightsRoundSlot = pollsWeights ? static_cast<int>(threadIdx.x) / idsRowVectors : kWeights; // 8: none
    int const weightsVec = static_cast<int>(threadIdx.x) % idsRowVectors;
    int64_t const idsRowBytes = 16 * idsRowVectors;
    uint8_t* const weightsColumn = landing + p.landing.weightsOffset + weightsVec * 16;
    for (int first = 0; first < numSources && !timedOut; first += 8)
    {
        int const roundSize = min(8, numSources - first);
        uint32_t pending = hasVector ? (1U << roundSize) - 1U : 0U;
        int weightsSource = 0;
        if (weightsRoundSlot < roundSize)
        {
            weightsSource = sSources[first + weightsRoundSlot];
            pending |= 1U << kWeights;
        }
        uint32_t const polled = pending;
        uint4 vectors[kWeights + 1];
        int sources[8]; // the round's sources (past roundSize: unused)
        auto readSources = [&]
        {
#pragma unroll
            for (int j = 0; j < 8; ++j)
                sources[j] = sSources[first + j];
        };
        auto address = [&](int i)
        {
            if (i == kWeights)
            {
                return weightsColumn + landingRow(p.epSize, weightsSource, token) * idsRowBytes;
            }
            return vectorColumn + landingRow(p.epSize, sources[i], token) * wv.rowBytes;
        };
        pending = pollUntilArrived(pending, kDataSentinel, readSources, address, vectors, waitStart, p.timeoutCycles);
        timedOut = pending != 0;
        uint32_t const received = polled & ~pending;
        readSources();
#pragma unroll
        for (int j = 0; j < kWeights; ++j)
        {
            if (!((received >> j) & 1U))
            {
                continue;
            }
            int64_t const recvRow = static_cast<int64_t>(sources[j]) * p.maxTokensPerRank + token;
            storeReceivedVector<PAYLOAD_KIND>(p, recvRow, wv, vectors[j]);
            storeVolatile(address(j), splat(kDataSentinel));
        }
        if ((received >> kWeights) & 1U)
        {
            int64_t const recvRow = static_cast<int64_t>(weightsSource) * p.maxTokensPerRank + token;
            storeTopkWeights(p, recvRow, topK, weightsVec, vectors[kWeights]);
            storeVolatile(address(kWeights), splat(kDataSentinel));
        }
    }
}

// D4-D6 of every dispatch kernel, after the block's own pushes.
template <PayloadKind PAYLOAD_KIND, bool ENABLE_RANK_MASK>
__device__ __forceinline__ void receiveTokenSlot(DispatchParams const& p, int topK, int idsRowVectors, int token,
    int chunk, int vec, uint64_t targetMask, int64_t idsSetOffset, uint64_t& sRecvSourceMask)
{
    __shared__ __align__(16) int sSources[kMaxRanks];
    __shared__ int sNumSources;
    long long const waitStart = clock64();
    bool timedOut = false;
    if (threadIdx.x == 0 && ((targetMask >> p.epRank) & 1ULL))
    {
        atomicOr(reinterpret_cast<unsigned long long*>(&sRecvSourceMask), 1ULL << p.epRank);
    }
    if (p.epSize * idsRowVectors <= kNumThreads)
    { // D4
        pollIdsRows<1, ENABLE_RANK_MASK>(
            p, topK, idsRowVectors, token, chunk, idsSetOffset, waitStart, sRecvSourceMask, timedOut);
    }
    else
    {
        pollIdsRows<4, ENABLE_RANK_MASK>(
            p, topK, idsRowVectors, token, chunk, idsSetOffset, waitStart, sRecvSourceMask, timedOut);
    }
    __syncthreads();
    uint64_t const recvSourceMask = sRecvSourceMask;
    if (chunk == 0 && threadIdx.x == 0)
    {
        p.recvSourceMask[token] = recvSourceMask;
    }
    if (threadIdx.x < 32) // this rank's own row was received in D2
    {
        listSources(recvSourceMask & ~(1ULL << p.epRank), sSources, &sNumSources);
    }
    __syncthreads();
    bool const pollsWeights = chunk == 0 && p.hasTopkWeights && static_cast<int>(threadIdx.x) < 8 * idsRowVectors;
    if ((vec < p.payloadVectors + p.wireScaleVectors || pollsWeights) && !timedOut) // D5, D6
    {
        receiveVectors<PAYLOAD_KIND>(
            p, topK, idsRowVectors, token, vec, pollsWeights, sSources, sNumSources, waitStart, timedOut);
    }
    if (timedOut)
    {
        reportAndTrap(p.roundState, p.epRank, "timed out waiting for a peer (dispatch)");
    }
}
#endif // MOE_A2A_SENTINEL_COMPILE

// ============================================================================= dispatch kernel
// Any payload format and top-k (fixed at compile time, or at most TOP_K_CAPACITY).
template <int TOP_K_CAPACITY, bool FIXED_TOP_K, PayloadKind PAYLOAD_KIND, bool ENABLE_RANK_MASK>
__global__ void __launch_bounds__(kNumThreads, kDispatchBlocksPerSM)
    moeA2ASentinelDispatchKernel(DispatchParams const p)
{
#if MOE_A2A_SENTINEL_COMPILE
    constexpr int kRowCapacity = topkRowVectors(TOP_K_CAPACITY);
    int const topK = FIXED_TOP_K ? TOP_K_CAPACITY : p.topK;
    int const idsRowVectors = topkRowVectors(topK);
    int const token = blockIdx.x / p.blocksPerToken, chunk = blockIdx.x % p.blocksPerToken;
    int const vec = chunk * kNumThreads + threadIdx.x;
    __shared__ uint64_t sTargetMask, sRecvSourceMask;
    __shared__ int sLandingSet;
    __shared__ uint4 sIdsRow[kRowCapacity], sWeightsRow[kRowCapacity]; // wire rows for D3
    waitForPreviousKernel();
    if (threadIdx.x < 32)
    { // D1
        int const lane = threadIdx.x;
        int epoch = 0;
        if (lane == 0)
        {
            epoch = loadVolatileInt(p.roundState + kEpoch);
        }
        uint32_t weight = 0; // loaded before routing, so that it is in flight with the expert id
        if (chunk == 0 && p.hasTopkWeights && token < p.localNumTokens && lane < topK)
        {
            int64_t const index = static_cast<int64_t>(token) * topK + lane;
            weight = p.weightsBf16
                ? static_cast<uint32_t>(reinterpret_cast<uint16_t const*>(p.topkWeights)[index]) << 16
                : p.topkWeights[index];
        }
        uint64_t const targetMask = routeToken<TOP_K_CAPACITY, FIXED_TOP_K, ENABLE_RANK_MASK>(p, token, lane,
            reinterpret_cast<uint32_t*>(sIdsRow),
            chunk == 0 ? p.topkTargetRanks + static_cast<int64_t>(token) * topK : nullptr);
        if (lane < 4 * kRowCapacity) // padding: weight 0
        {
            reinterpret_cast<uint32_t*>(sWeightsRow)[lane] = lane < topK ? canonicalizeWord(weight) : 0U;
        }
        if (lane == 0)
        {
            sLandingSet = epoch & 1;
            sTargetMask = targetMask;
            sRecvSourceMask = 0;
        }
    }
    __syncthreads();
    uint64_t const targetMask = sTargetMask;
    int64_t const idsSetOffset = sLandingSet * p.landing.idsSetBytes;
    int64_t const recvRow = static_cast<int64_t>(p.epRank) * p.maxTokensPerRank + token;
    if (targetMask && vec < p.payloadVectors + p.wireScaleVectors)
    { // D2
        WireVector const wv = locateWireVector(p, vec);
        pushVector<PAYLOAD_KIND>(p, targetMask, token, recvRow, wv, loadWireVector<PAYLOAD_KIND>(p, token, wv));
    }
    if (chunk == 0) // D3
    {
        pushIdsAndWeights<ENABLE_RANK_MASK>(
            p, topK, idsRowVectors, targetMask, token, recvRow, idsSetOffset, sIdsRow, sWeightsRow);
    }
    allowNextKernel();
    receiveTokenSlot<PAYLOAD_KIND, ENABLE_RANK_MASK>(
        p, topK, idsRowVectors, token, chunk, vec, targetMask, idsSetOffset, sRecvSourceMask);
#else
    __trap(); // not built for this architecture (see MOE_A2A_SENTINEL_COMPILE)
#endif
}

#if MOE_A2A_SENTINEL_COMPILE
// ============================================================================= combine phases
// C1, warp 0: the ranks of the token's target slots (topkTargetRanks >= 0) in top-k slot
// order into sSources[0, n); lane 0 stores n.
__device__ __forceinline__ void listTargetRanks(
    CombineParams const& p, int token, int topK, int* sSources, int* sNumSources)
{
    int const lane = threadIdx.x;
    int const rank
        = (lane < topK && token < p.localNumTokens) ? p.topkTargetRanks[static_cast<int64_t>(token) * topK + lane] : -1;
    uint32_t const sourceLanes = __ballot_sync(0xFFFFFFFFU, rank >= 0);
    if (rank >= 0)
    {
        sSources[__popc(sourceLanes & ((1U << lane) - 1U))] = rank;
    }
    if (lane == 0)
    {
        *sNumSources = __popc(sourceLanes);
    }
}

// C2: push the expert output of row (source, token) to source's combine landing row
// (epRank, token), for every source that sent this slot a token but this rank (whose partial
// C3 reads directly). 8 sources at a time, every load issued before the stores (a store's
// "memory" clobber would otherwise wait for one load after another).
__device__ __forceinline__ void pushPartials(CombineParams const& p, int token, int vec, uint64_t recvSourceMask)
{
    int64_t const rowBytes = static_cast<int64_t>(p.hiddenSize) * 2;
    int64_t const landingOffset
        = p.landing.combineOffset + (static_cast<int64_t>(p.epRank) * p.maxTokensPerRank + token) * rowBytes + vec * 16;
    for (uint64_t unsent = recvSourceMask & ~(1ULL << p.epRank); unsent;)
    {
        int sources[8]; // the round's sources, -1 past the last
        uint4 partials[8];
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            sources[j] = unsent ? popLowestRank(unsent) : -1;
            if (sources[j] >= 0)
            {
                partials[j] = *reinterpret_cast<uint4 const*>(p.expertOutput
                    + (static_cast<int64_t>(sources[j]) * p.maxTokensPerRank + token) * rowBytes + vec * 16);
            }
        }
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            if (sources[j] >= 0)
            {
                storeVolatile(p.rankWorkspaces[sources[j]] + landingOffset, canonicalizePartial(partials[j]));
            }
        }
    }
}

// Fence-combine order of up to 8 sources (zero-filled): s0 + s1 for n <= 2, else the pairwise
// tree ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7)). __fadd_rn is never fused and, like the
// fence combine's adds, flushes subnormals exactly when the module builds with -ftz (fast math).
__device__ __forceinline__ float sumGroup(float const (&values)[8], int numSources)
{
    if (numSources <= 2)
    {
        return __fadd_rn(values[0], values[1]);
    }
    return __fadd_rn(__fadd_rn(__fadd_rn(values[0], values[1]), __fadd_rn(values[2], values[3])),
        __fadd_rn(__fadd_rn(values[4], values[5]), __fadd_rn(values[6], values[7])));
}

// C3, sources 8 * group .. 8 * group + 7 of the slot-ordered list: poll their partials of this
// rank's token (every load in flight), re-arm them, and sum each of the vector's 8 elements.
__device__ __forceinline__ void receiveGroup(CombineParams const& p, int token, int vec, int group, int numSources,
    int const* sSources, long long waitStart, bool& timedOut, float (&sums)[8])
{
#pragma unroll
    for (int e = 0; e < 8; ++e)
    {
        sums[e] = 0.F;
    }
    if (8 * group >= numSources)
    {
        return;
    }
    int64_t const rowBytes = static_cast<int64_t>(p.hiddenSize) * 2;
    uint8_t* combineRows = p.rankWorkspaces[p.epRank] + p.landing.combineOffset + vec * 16;
    int ranks[8];
#pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        ranks[j] = 8 * group + j < numSources ? sSources[8 * group + j] : -1;
    }
    uint4 partials[8];
    uint32_t pending = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        partials[j] = make_uint4(0, 0, 0, 0); // a missing source adds +0
        if (ranks[j] == p.epRank)
        {
            partials[j] = canonicalizePartial(*reinterpret_cast<uint4 const*>(
                p.expertOutput + (static_cast<int64_t>(p.epRank) * p.maxTokensPerRank + token) * rowBytes + vec * 16));
        }
        else if (ranks[j] >= 0)
        {
            pending |= 1U << j;
        }
    }
    uint32_t const polled = pending;
    auto address
        = [&](int j) { return combineRows + (static_cast<int64_t>(ranks[j]) * p.maxTokensPerRank + token) * rowBytes; };
    if (pollUntilArrived(
            pending, kDataSentinel, [] {}, address, partials, waitStart, p.timeoutCycles))
    {
        timedOut = true;
    }
#pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        if ((polled >> j) & 1U)
        {
            storeVolatile(address(j), splat(kDataSentinel));
        }
    }
#pragma unroll
    for (int e = 0; e < 8; ++e)
    {
        float values[8];
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            values[j] = __bfloat162float(reinterpret_cast<__nv_bfloat16 const*>(&partials[j])[e]);
        }
        sums[e] = sumGroup(values, numSources);
    }
}

// C3: join the group sums as the fence combine joins its lanes (n <= 8: g0; n <= 16: g0 + g1;
// else (g0 + g1) + (g2 + g3)) and round to bf16.
template <int NUM_GROUPS>
__device__ __forceinline__ uint4 joinGroups(float const (&groupSums)[NUM_GROUPS][8], int numSources)
{
    uint4 out;
    __nv_bfloat16* outElements = reinterpret_cast<__nv_bfloat16*>(&out);
#pragma unroll
    for (int e = 0; e < 8; ++e)
    {
        float sum = groupSums[0][e];
        if constexpr (NUM_GROUPS > 1)
        {
            if (numSources > 8)
            {
                sum = __fadd_rn(sum, groupSums[1][e]);
            }
        }
        if constexpr (NUM_GROUPS > 2)
        {
            if (numSources > 16)
            {
                float last = 0.F;
                if constexpr (NUM_GROUPS > 3)
                {
                    last = groupSums[3][e];
                }
                sum = __fadd_rn(sum, __fadd_rn(groupSums[2][e], last));
            }
        }
        outElements[e] = __float2bfloat16_rn(sum);
    }
    return out;
}

// C4: re-arm this slot's ids rows of the round's landing set (their readers, the dispatch
// blocks, have all ended), then the last block closes the round and advances the epoch (every
// block read it first).
__device__ __forceinline__ void rearmIdsAndAdvanceEpoch(
    CombineParams const& p, int token, int chunk, int idsRowVectors, int landingSet)
{
    if (chunk == 0)
    {
        uint8_t* idsRows = p.rankWorkspaces[p.epRank] + p.landing.idsOffset + landingSet * p.landing.idsSetBytes;
        int64_t const idsRowBytes = 16 * idsRowVectors;
        for (int item = threadIdx.x; item < p.epSize * idsRowVectors; item += kNumThreads)
        {
            int const source = item / idsRowVectors, vec = item % idsRowVectors;
            if (source == p.epRank)
            {
                continue; // never written: own rows go to the receive buffers
            }
            storeVolatile(idsRows + landingRow(p.epSize, source, token) * idsRowBytes + vec * 16, splat(kIdsSentinel));
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        __threadfence();
        int const finishedBefore = atomicAdd(p.roundState + kFinishedCombineBlocks, 1);
        if (finishedBefore == static_cast<int>(gridDim.x) - 1)
        {
            atomicExch(p.roundState + kFinishedCombineBlocks, 0);
            atomicAdd(p.roundState + kEpoch, 1);
        }
    }
}
#endif // MOE_A2A_SENTINEL_COMPILE

// ============================================================================== combine kernel
template <int TOP_K_CAPACITY, bool FIXED_TOP_K, bool ENABLE_RANK_MASK>
__global__ void __launch_bounds__(kNumThreads) moeA2ASentinelCombineKernel(CombineParams const p)
{
#if MOE_A2A_SENTINEL_COMPILE
    constexpr int kNumGroups = (TOP_K_CAPACITY + 7) / 8;
    int const topK = FIXED_TOP_K ? TOP_K_CAPACITY : p.topK;
    int const token = blockIdx.x / p.blocksPerToken, chunk = blockIdx.x % p.blocksPerToken;
    int const vec = chunk * kNumThreads + threadIdx.x;
    __shared__ uint64_t sRecvSourceMask;
    __shared__ int sSources[TOP_K_CAPACITY];
    __shared__ int sNumSources, sLandingSet;
    waitForPreviousKernel(); // the expert output, and the dispatch's routing state
    if (threadIdx.x < 32)
    {                        // C1
        listTargetRanks(p, token, topK, sSources, &sNumSources);
        if (threadIdx.x == 0)
        {
            sLandingSet = loadVolatileInt(p.roundState + kEpoch) & 1;
            uint64_t mask = p.recvSourceMask[token];
            if constexpr (ENABLE_RANK_MASK)
            {
                mask &= p.activeRankMask; // never push to a masked rank
            }
            sRecvSourceMask = mask;
        }
    }
    __syncthreads();
    if (vec < p.rowVectors)
    {
        pushPartials(p, token, vec, sRecvSourceMask); // C2
    }
    allowNextKernel();
    bool timedOut = false;
    if (token < p.localNumTokens && vec < p.rowVectors)
    { // C3
        long long const waitStart = clock64();
        float groupSums[kNumGroups][8];
#pragma unroll
        for (int group = 0; group < kNumGroups; ++group)
        {
            receiveGroup(p, token, vec, group, sNumSources, sSources, waitStart, timedOut, groupSums[group]);
        }
        *reinterpret_cast<uint4*>(p.output + (static_cast<int64_t>(token) * p.hiddenSize) * 2 + vec * 16)
            = joinGroups<kNumGroups>(groupSums, sNumSources);
    }
    if (timedOut)
    {
        reportAndTrap(p.roundState, p.epRank, "timed out waiting for a peer (combine)");
    }
    rearmIdsAndAdvanceEpoch(p, token, chunk, topkRowVectors(topK), sLandingSet); // C4
#else
    __trap(); // not built for this architecture (see MOE_A2A_SENTINEL_COMPILE)
#endif
}

// Runs the statements __VA_ARGS__ with TOP_K_CAPACITY and FIXED_TOP_K of the kernels serving
// topK: topK 6, 8, 10, 16 and 22 have their own kernels, the others share a kernel with a
// runtime topK of at most 8, 16 or 24.
#define MOE_A2A_SENTINEL_TOP_K_CASE(CAPACITY, FIXED, ...)                                                              \
    {                                                                                                                  \
        constexpr int TOP_K_CAPACITY = CAPACITY;                                                                       \
        constexpr bool FIXED_TOP_K = FIXED;                                                                            \
        __VA_ARGS__;                                                                                                   \
    }
#define MOE_A2A_SENTINEL_SWITCH_TOP_K(topK, ...)                                                                       \
    if ((topK) == 6)                                                                                                   \
        MOE_A2A_SENTINEL_TOP_K_CASE(6, true, __VA_ARGS__)                                                              \
    else if ((topK) == 8)                                                                                              \
        MOE_A2A_SENTINEL_TOP_K_CASE(8, true, __VA_ARGS__)                                                              \
    else if ((topK) == 10)                                                                                             \
        MOE_A2A_SENTINEL_TOP_K_CASE(10, true, __VA_ARGS__)                                                             \
    else if ((topK) == 16)                                                                                             \
        MOE_A2A_SENTINEL_TOP_K_CASE(16, true, __VA_ARGS__)                                                             \
    else if ((topK) == 22)                                                                                             \
        MOE_A2A_SENTINEL_TOP_K_CASE(22, true, __VA_ARGS__)                                                             \
    else if ((topK) <= 8)                                                                                              \
        MOE_A2A_SENTINEL_TOP_K_CASE(8, false, __VA_ARGS__)                                                             \
    else if ((topK) <= 16)                                                                                             \
        MOE_A2A_SENTINEL_TOP_K_CASE(16, false, __VA_ARGS__)                                                            \
    else                                                                                                               \
        MOE_A2A_SENTINEL_TOP_K_CASE(24, false, __VA_ARGS__)

using tensorrt_llm::common::launchWithPdlWhenEnabled;

// 1 in the code built for an architecture the kernels support.
__device__ int gKernelsCompiled = MOE_A2A_SENTINEL_COMPILE;

// Only the all-ranks-active kernels (ENABLE_RANK_MASK = false) are instantiated: the ops reject a
// rank mask on this transport, since the kernels never drain or re-arm a masked peer's rows.
template <int TOP_K_CAPACITY, bool FIXED_TOP_K>
void launchDispatch(PayloadKind payloadKind, DispatchParams const& params, cudaStream_t stream)
{
    void (*kernel)(DispatchParams) = payloadKind == PayloadKind::kBf16
        ? moeA2ASentinelDispatchKernel<TOP_K_CAPACITY, FIXED_TOP_K, PayloadKind::kBf16, false>
        : payloadKind == PayloadKind::kFp8
        ? moeA2ASentinelDispatchKernel<TOP_K_CAPACITY, FIXED_TOP_K, PayloadKind::kFp8, false>
        : moeA2ASentinelDispatchKernel<TOP_K_CAPACITY, FIXED_TOP_K, PayloadKind::kNvfp4, false>;
    launchWithPdlWhenEnabled("moeA2ASentinelDispatchKernel", kernel,
        dim3(params.maxTokensPerRank * params.blocksPerToken), dim3(kNumThreads), 0, stream, params);
}

template <int TOP_K_CAPACITY, bool FIXED_TOP_K>
void launchCombine(CombineParams const& params, cudaStream_t stream)
{
    launchWithPdlWhenEnabled("moeA2ASentinelCombineKernel",
        moeA2ASentinelCombineKernel<TOP_K_CAPACITY, FIXED_TOP_K, false>,
        dim3(params.maxTokensPerRank * params.blocksPerToken), dim3(kNumThreads), 0, stream, params);
}

} // namespace

bool moeA2ASentinelKernelsAvailable()
{
    // The GPUs the transport is validated on.
    int device = 0;
    int major = 0;
    int minor = 0;
    if (cudaGetDevice(&device) != cudaSuccess
        || cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess
        || cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess)
    {
        cudaGetLastError();
        return false;
    }
    if (major != 10 || (minor != 0 && minor != 3 && minor != 7))
    {
        return false;
    }
    int compiled = 0;
    if (cudaMemcpyFromSymbol(&compiled, gKernelsCompiled, sizeof(compiled)) != cudaSuccess)
    {
        cudaGetLastError(); // no code for this device at all
        return false;
    }
    return compiled != 0;
}

void moeA2ASentinelDispatchLaunch(PayloadKind payloadKind, DispatchParams const& params, cudaStream_t stream)
{
    MOE_A2A_SENTINEL_SWITCH_TOP_K(
        params.topK, { launchDispatch<TOP_K_CAPACITY, FIXED_TOP_K>(payloadKind, params, stream); })
}

void moeA2ASentinelCombineLaunch(CombineParams const& params, cudaStream_t stream)
{
    MOE_A2A_SENTINEL_SWITCH_TOP_K(params.topK, { launchCombine<TOP_K_CAPACITY, FIXED_TOP_K>(params, stream); })
}

#undef MOE_A2A_SENTINEL_SWITCH_TOP_K
#undef MOE_A2A_SENTINEL_TOP_K_CASE

} // namespace kernels::moe_comm::sentinel

TRTLLM_NAMESPACE_END
