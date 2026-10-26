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

/*
 * K4 push-combine consumer top-k reduce, the CONSUMER-side payload
 * kernel of the packed-NVFP4 tileNvfp4 V2 push-combine path. Producer pushes
 * (K1) land packed [Q|SF] 4480-B records into THIS consumer's LOCAL landing
 * plane; K4 gates on the K3->K4 hand-off, TMA-loads the local records, dequants
 * and top-k reduces them to BF16 -- reusing the family decode + reduce.
 *
 * Strict contract, all enforced structurally below:
 *   - Each CTA performs EXACTLY ONE non-spin acquire load of localGate[C]
 *     (ld.acquire.gpu) as the K3->K4 hand-off, then proceeds. There is NO spin,
 *     NO compare, NO clock64 deadline on the gate.
 *   - K4 reads ONLY the consumer-LOCAL packed landing plane (packed_local_base
 *     at readSlot = P*r_runtime + t, mirroring K1's landing address). It holds
 *     NO remote peer pointers, reads NO completionFlags / readiness words, and
 *     has NO direct-global decode fallback: decodeFallback and
 *     stageNonBulkRowsSynchronously are never called, and the only reused
 *     clock64/timeout trap (mbarWaitParityAllConsumers) is unreachable in
 *     practice because a local TMA always drains far under kTimeoutCycles.
 *   - EP2 uses Route2FullTokenPacked (4 tokens/CTA, <=2 records/token, ONE 4480-B
 *     local TMA per valid record, split by rows: phase0 rows 0..3 / phase1 rows
 *     4..7). EP4/8/16/32 use Route8HalfPacked (1 token/CTA, <=8 route
 *     positions). EP4 has at most four valid records, but must still inspect all
 *     eight positions because dispatch deduplicates in place rather than
 *     compacting. Each Half record is split into phase0 2240-B + phase1 2240-B.
 *     All variants issue phase0 then phase1 and keep phase1 in flight while decoding
 *     phase0. mbarrier expected-bytes are computed from the ACTUAL issued rows.
 *
 * Byte-exact against the PULL combine (tileNvfp4CombineTma, untouched):
 *   - The route/record cache (dedup, validity, per-record bulk mask) is built by
 *     the family's own Route{2,8} buildRouteCacheAndMasks, fed direct
 *     producer-band bases with address-rank zero so the family's
 *     (epRank*R + dstIdx) slot math resolves to the LOCAL landing slot
 *     (targetRank*R + dstIdx). See makeLocalPointers.
 *   - Every record row of the shared packed cache is zero-initialised before the
 *     TMA. Only VALID (post-dedup) records are transferred; invalid / deduped /
 *     out-of-range record rows stay all-zero, and an all-zero packed record
 *     decodes through Nvfp4Format::decodeBlock16 to EXACT +0.0f -- bit-identical
 *     to the PULL combine's decodeFallback nullptr-route contribution. U == 0 (a token
 *     with no valid record) issues no payload TMA and stores an exact-zero token.
 *   - The default staging stride (kK1SelectedSharedStrideDefaultBytes == 4480 ==
 *     Nvfp4Format::kPackedRecordStrideBytes) reuses the family issueTmaPhase and
 *     decodeFastPhase store loops unchanged. The 4496 stride option only widens the
 *     SHARED record-row stride (bank/occupancy tuning); it decodes the SAME
 *     4480 payload bytes via a stride-parameterised mirror of those loops.
 *
 * The one legal launch width is 224 threads (7 warps); the family decode is
 * tile-parallel and its per-element arithmetic is warp-count independent, so the
 * 7-warp K4 schedule below covers 112 / 56 / 28 units per phase (all divisible
 * by 7) and produces bit-identical output to the 16-warp PULL decode.
 */

#include "moeAlltoAllTileNvfp4V2PushFlags.h"
#include "moeAlltoAllTileNvfp4V2PushK1K4.h"

#include "moeAlltoAllTileNvfp4V2TmaTemplate.cuh"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::moe_comm
{
namespace
{

using namespace tile_nvfp4_tma;

// The one legal launch width (kK1BaselineThreads); 7 warps. The family decode is
// warp-count independent per output element, so a 7-warp tile schedule covers
// every 112 / 56 / 28-unit phase span (all divisible by 7).
constexpr int kK4Threads = kK1BaselineThreads; // 224
constexpr int kK4Warps = kK4Threads / 32;      // 7
static_assert(kK4Threads == 224, "K4 baseline launch width is frozen at 224");
static_assert(kK4Warps == 7, "K4 runs seven warps (112/56/28 all divisible by 7)");
static_assert(kK4Threads % 32 == 0, "K4 launch width must be a whole warp count");
static_assert(kTilesPerToken % kK4Warps == 0, "112 tiles must split evenly over 7 warps");
static_assert((kTilesPerToken / 2) % kK4Warps == 0, "56 half-record tiles must split over 7");

// The redeclared packed record byte count (control-plane-adjacent header) must
// equal the family's authoritative Nvfp4Format geometry.
static_assert(kK1PackedRecordBytes == Nvfp4Format::kPackedRecordBytes,
    "K1/K4 packed record byte count must mirror Nvfp4Format::kPackedRecordBytes");
static_assert(kK1PackedRecordBytes == Nvfp4Format::kPackedRecordStrideBytes,
    "K1/K4 packed record stride must mirror Nvfp4Format::kPackedRecordStrideBytes");

// The two supported SHARED staging strides. The default is the natural packed
// record stride; the padded option pads each staged record row to the next 16-byte
// multiple. Both keep the record PAYLOAD at 4480 B; only the smem row stride
// differs (bank/occupancy tuning).
static_assert(kK1SelectedSharedStrideDefaultBytes == 4480, "default staging stride is 4480");
static_assert(kK1SelectedSharedStrideAbBytes == 4496, "A/B staging stride is 4496");
static_assert(kK1SelectedSharedStrideDefaultBytes % 16 == 0, "staging stride must be 16-aligned");
static_assert(kK1SelectedSharedStrideAbBytes % 16 == 0, "staging stride must be 16-aligned");
static_assert(kK1SelectedSharedStrideAbBytes >= Nvfp4Format::kPackedRecordBytes,
    "A/B staging stride must still hold the whole 4480-B record");

// -----------------------------------------------------------------------------
// Single non-spin acquire of this consumer's K3->K4 hand-off word localGate[C].
//
// `controlRowBase` is this consumer's OWN control row, ALREADY hopped by the host
// (control_base == base + rank_id*stride(0)); it is the SAME address K3 released
// localGate on (control_peer_base[rank_id]). K3 released localGate[C]
// (fence.release.gpu + st.relaxed.gpu) after acquiring every producer's pushed
// payload; a single ld.acquire.gpu here re-establishes that ordering so the local
// landing bytes this CTA is about to read are visible. It is a plain load: no
// spin, no compare against the generation, no deadline -- K4 is launched on the
// owner stream only AFTER K3, so the gate is already set. The loaded value is
// deliberately discarded.
// -----------------------------------------------------------------------------
__device__ __forceinline__ void k4AcquireLocalGateOnce(std::uint8_t* controlRowBase)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    std::uint64_t* gate = pushLocalGatePtr(controlRowBase);
    std::uint64_t observed;
    asm volatile("ld.acquire.gpu.u64 %0, [%1];" : "=l"(observed) : "l"(gate) : "memory");
    (void) observed;
#else
    (void) controlRowBase;
#endif
}

// Zero every record row of the shared packed cache (all kStageRows rows at the
// selected staging stride). Invalid / deduped / out-of-range record rows are
// never TMA'd, so this zero is what makes them decode to exact +0.0f. Valid rows
// are overwritten by the subsequent (ordered) TMA. Both strides are a whole
// number of 16-B granules, so the zeroing is uint4-wide.
template <int kSharedStride>
__device__ __forceinline__ void k4ZeroPackedCache(std::uint8_t* packedCache)
{
    constexpr int kZeroWords = (kStageRows * kSharedStride) / 16;
    static_assert(
        kZeroWords * 16 == kStageRows * kSharedStride, "packed cache byte span must be a whole number of 16-B words");
    uint4* words = reinterpret_cast<uint4*>(packedCache);
#pragma unroll 1
    for (int i = static_cast<int>(threadIdx.x); i < kZeroWords; i += static_cast<int>(blockDim.x))
    {
        words[i] = make_uint4(0U, 0U, 0U, 0U);
    }
}

// Zero ONLY the record rows NOT in keepRowMask (the rows the ordered TMA
// will NOT overwrite -- invalid / deduped / out-of-range routes). Those rows carry
// no payload and must decode to exact +0.0f, which the zeroed bytes provide. Valid
// rows in keepRowMask are fully overwritten by the TMA, so skipping their zero is
// bitwise-identical to clearing all kStageRows; in the normal all-valid path
// keepRowMask covers every active row, so this clears NOTHING and zeroed_bytes/token
// collapses from kStageRows*stride (35840 B) to 0 (zero-leaf
// equivalence). Caller must publish keepRowMask (a __syncthreads after route
// resolution) first; keepRowMask is CTA-uniform, so the row loop never diverges.
template <int kSharedStride>
__device__ __forceinline__ void k4ZeroInvalidRows(std::uint8_t* packedCache, std::uint32_t keepRowMask)
{
    constexpr int kWordsPerRow = kSharedStride / 16;
    static_assert(kWordsPerRow * 16 == kSharedStride, "staging stride must be a whole number of 16-B words");
#pragma unroll 1
    for (int r = 0; r < kStageRows; ++r)
    {
        if ((keepRowMask >> r) & 1U)
        {
            continue; // valid row: the ordered TMA overwrites it in full
        }
        uint4* row = reinterpret_cast<uint4*>(packedCache + static_cast<std::size_t>(r) * kSharedStride);
#pragma unroll 1
        for (int i = static_cast<int>(threadIdx.x); i < kWordsPerRow; i += static_cast<int>(blockDim.x))
        {
            row[i] = make_uint4(0U, 0U, 0U, 0U);
        }
    }
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
// -----------------------------------------------------------------------------
// Padded-staging-stride mirror of issueTmaPhase's packed branch. Byte-for-
// byte the same transfer as the family issueTmaPhase when kSharedStride == 4480,
// with two differences for the 4496 knob: (a) the shared record ROW base uses
// kSharedStride, and (b) there is no skew/delay spin (K4 carries no skew
// profile). The GLOBAL source is the tight 4480-B record (its phase slice), the
// mbarrier expected-tx is popc(issued rows) * bytesPerStage.
// -----------------------------------------------------------------------------
template <int Phase, class Policy, int kSharedStride>
__device__ __forceinline__ void k4IssueTmaPhasePadded(typename Policy::Shared const& shared, std::uint32_t issueMask)
{
    using Window = typename Policy::Window;
    std::uint32_t const phaseMask = Window::template phaseRowsMask<Phase>(issueMask);
    if (phaseMask == 0U)
    {
        return;
    }
    unsigned const bytesPerStage = Window::template qBytes<Phase>() + Window::template sfBytes<Phase>();
    mbarArriveExpectTx(&shared.tmaBar[Phase], static_cast<unsigned>(__popc(phaseMask)) * bytesPerStage);

    constexpr unsigned phaseBegin = Window::template phaseByteBegin<Phase>();
#pragma unroll
    for (int stageRow = 0; stageRow < kStageRows; ++stageRow)
    {
        if (((phaseMask >> stageRow) & 1U) == 0U)
        {
            continue;
        }
        CachedRoute const cached = Policy::stageRoute(shared, stageRow);
        std::uint8_t* rec = shared.packedCache + static_cast<std::size_t>(stageRow) * kSharedStride;
        // Source stride is the tight 4480-B landing record; dst stride is the
        // (possibly padded) shared row. Both slices begin at phaseBegin.
        bulk1d(rec + phaseBegin, cached.pointer.qRecord + phaseBegin, bytesPerStage, &shared.tmaBar[Phase]);
    }
}
#endif // __CUDA_ARCH__ >= 900

// -----------------------------------------------------------------------------
// Padded-staging-stride mirror of Route8HalfPacked /
// Route2FullTokenPacked
// decodeFastPhase for ONE decode unit. These are byte-for-byte the family store
// loops (Policies.cuh) with the single change that the shared record ROW base
// uses kSharedStride; the per-tile Q/SF byte offsets, the decodeBlock16 unpack,
// the frozen reduce (reduceRoute8Frozen / Route2 collapse), packBf16Pair, and
// the uint4 store are all reused verbatim. They are only instantiated for the
// non-default (4496) knob; the default (4480) path calls decodeFastPhase itself.
// -----------------------------------------------------------------------------
template <int kSharedStride>
__device__ __forceinline__ void k4Route8DecodeTilePadded(
    TileNvfp4CombinePointersV2 const& ptrs, std::uint8_t const* packedCache, int ctaBase, int tile, int lane)
{
    using Window = Route8HalfPackedPolicy::Window;
    using Format = Nvfp4Format;
    int const blockInTile = lane / kTopK;
    int const routePosition = lane % kTopK;
    __nv_bfloat16* outputRecord = ptrs.output + static_cast<std::size_t>(ctaBase) * kHiddenElements;

    std::uint8_t const* rec = packedCache + static_cast<std::size_t>(routePosition) * kSharedStride;
    std::uint64_t const scalePack = *reinterpret_cast<std::uint64_t const*>(rec + Window::sfByteOffset(tile));
    std::uint8_t const* qBlock = rec + Window::qByteOffset(tile) + blockInTile * Format::kQBytesPerBlock;
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
        if (routePosition == 0)
        {
            __nv_bfloat16* output = outputRecord + static_cast<std::size_t>(tile) * kTileElements
                + blockInTile * kScaleBlockElements + batch * (kPairsPerStore * 2);
            *reinterpret_cast<uint4*>(output) = words;
        }
    }
}

template <int kSharedStride>
__device__ __forceinline__ void k4Route2DecodeQuadPadded(TileNvfp4CombinePointersV2 const& ptrs,
    std::uint8_t const* packedCache, int ctaBase, int tokenSlot, int quadInToken, int lane)
{
    using Policy = Route2FullTokenPackedPolicy;
    using Window = Policy::Window;
    using Format = Nvfp4Format;
    int const tileInQuad = lane >> 3;
    int const blockInTile = (lane & 7) >> 1;
    int const routePosition = lane & 1;
    int const tile = quadInToken * Policy::kTilesPerQuad + tileInQuad;
    int const record = tokenSlot * Policy::kRecordsPerToken + routePosition;
    __nv_bfloat16* outputRecord = ptrs.output + static_cast<std::size_t>(ctaBase + tokenSlot) * kHiddenElements;

    std::uint8_t const* rec = packedCache + static_cast<std::size_t>(record) * kSharedStride;
    std::uint64_t const scalePack = *reinterpret_cast<std::uint64_t const*>(rec + Window::sfByteOffset(tile));
    std::uint8_t const* qBlock = rec + Window::qByteOffset(tile) + blockInTile * Format::kQBytesPerBlock;
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
            x = Policy::collapse(x, routePosition);
            y = Policy::collapse(y, routePosition);
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

// -----------------------------------------------------------------------------
// 7-warp phase schedules. Each warp fully decodes one unit (a tile for
// EP4/8/16/32, a 4-tile quad for EP2); the family per-unit decode is warp-count
// independent, so striding by kK4Warps over the phase's [begin,end) unit span
// (every span length divisible by 7) covers every unit exactly once with
// bit-identical results.
// The DEFAULT stride reuses decodeFastPhase verbatim; the 4496 knob uses the
// stride-parameterised mirror above.
// -----------------------------------------------------------------------------
// Register-fold frozen K8 tree. EXACT same operand grouping/order as the
// lane-0 result of reduceRoute8Frozen (Common.cuh): ((v0+v1)+(v2+v3)) +
// ((v4+v5)+(v6+v7)) with __fadd_rn. IEEE fadd is deterministic under fixed
// association, so this is bitwise-identical to the shuffle tree -- with ZERO
// __shfl.
__device__ __forceinline__ float k4FoldRoute8Frozen(
    float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7)
{
    float const v01 = __fadd_rn(v0, v1);
    float const v23 = __fadd_rn(v2, v3);
    float const v45 = __fadd_rn(v4, v5);
    float const v67 = __fadd_rn(v6, v7);
    return __fadd_rn(__fadd_rn(v01, v23), __fadd_rn(v45, v67));
}

// Register-fold decode: replaces the 8-lane shuffle reduction. Each lane owns
// ONE (tile, blockInTile, pairHalf) output slot (a uint4 = 4 pairs = 8 BF16), reads
// ALL kTopK record rows at that (tile, blockInTile) offset, decodes the 8 routes'
// pairs LOCALLY (byte-identical to decodeBlock16), folds them with the frozen K8
// tree in registers, and stores -- EVERY lane stores (vs 1/8 in a shuffle tree), NO __shfl.
// Persistent hot set is scaledBlock[8] + qHalf[8] (this half's 4 packed-Q bytes);
// each pair's fp4x2->float2 dequant is recomputed on the fly to bound registers.
// Handles both the 4480 default and 4496 padded stride via kSharedStride (the row
// base), so K4 does not call the family decodeFastPhase.
template <int kSharedStride>
__device__ __forceinline__ void k4Route8DecodeTileRegFold(
    TileNvfp4CombinePointersV2 const& ptrs, std::uint8_t const* packedCache, int ctaBase, int tile, int lane)
{
    using Window = Route8HalfPackedPolicy::Window;
    using Format = Nvfp4Format;
    int const pairHalf = lane & 1;           // which 4-pair half of the block16
    int const blockInTile = (lane >> 1) & 3; // 0..3
    __nv_bfloat16* outputRecord = ptrs.output + static_cast<std::size_t>(ctaBase) * kHiddenElements;

    float scaledBlock[kTopK];
    std::uint32_t qHalf[kTopK];
#pragma unroll
    for (int r = 0; r < kTopK; ++r)
    {
        std::uint8_t const* rec = packedCache + static_cast<std::size_t>(r) * kSharedStride;
        std::uint64_t const scalePack = *reinterpret_cast<std::uint64_t const*>(rec + Window::sfByteOffset(tile));
        std::uint8_t const rawBlockScale = static_cast<std::uint8_t>(scalePack >> (blockInTile * 8));
        float const globalD = __uint_as_float(static_cast<std::uint32_t>(scalePack >> 32));
        __nv_fp8_e4m3 blockScale;
        blockScale.__x = rawBlockScale;
        scaledBlock[r] = __fmul_rn(static_cast<float>(blockScale), globalD);
        std::uint8_t const* qBlock = rec + Window::qByteOffset(tile) + blockInTile * Format::kQBytesPerBlock;
        // this half's 4 packed-Q bytes (pairs [pairHalf*4, pairHalf*4+4))
        qHalf[r] = *reinterpret_cast<std::uint32_t const*>(qBlock + pairHalf * 4);
    }

    uint4 words;
    std::uint32_t* w = &words.x;
#pragma unroll
    for (int p = 0; p < kPairsPerBlock / 2; ++p) // 4 pairs in this half
    {
        float xs[kTopK];
        float ys[kTopK];
#pragma unroll
        for (int r = 0; r < kTopK; ++r)
        {
            __nv_fp4x2_e2m1 qPair;
            qPair.__x = static_cast<__nv_fp4x2_storage_t>((qHalf[r] >> (p * 8)) & 0xFFU);
            float2 const q = static_cast<float2>(qPair);
            xs[r] = __fmul_rn(q.x, scaledBlock[r]);
            ys[r] = __fmul_rn(q.y, scaledBlock[r]);
        }
        float const x = k4FoldRoute8Frozen(xs[0], xs[1], xs[2], xs[3], xs[4], xs[5], xs[6], xs[7]);
        float const y = k4FoldRoute8Frozen(ys[0], ys[1], ys[2], ys[3], ys[4], ys[5], ys[6], ys[7]);
        w[p] = packBf16Pair(x, y);
    }

    __nv_bfloat16* output = outputRecord + static_cast<std::size_t>(tile) * kTileElements
        + blockInTile * kScaleBlockElements + pairHalf * (kPairsPerStore * 2); // half offset: 0 or 8 elements
    *reinterpret_cast<uint4*>(output) = words;
}

template <int Phase, int kSharedStride>
__device__ __forceinline__ void k4Route8Schedule(TileNvfp4CombinePointersV2 const& ptrs,
    Route8HalfPackedPolicy::Shared const& shared, int ctaBase, int tokensInCta, int warp, int lane)
{
    using Policy = Route8HalfPackedPolicy;
    using Window = Policy::Window;
    int const begin = Window::template decodeBegin<Phase>(0, tokensInCta);
    int const end = Window::template decodeEnd<Phase>(0, tokensInCta);
    // Register-fold lane map: each lane owns (tile, blockInTile, pairHalf); a warp
    // spans 4 tiles per iteration (tileInGroup = lane>>3) and strides by 4*kK4Warps.
    // The 56-tile phase (56 = 2*28 = 2*4*kK4Warps) is covered exactly once, no
    // shuffles, so per-lane tail predication is safe (no cross-lane dependence).
    int const tileInGroup = lane >> 3; // 0..3
    for (int tileBase = begin + warp * 4; tileBase < end; tileBase += kK4Warps * 4)
    {
        int const tile = tileBase + tileInGroup;
        if (tile < end)
        {
            k4Route8DecodeTileRegFold<kSharedStride>(ptrs, shared.packedCache, ctaBase, tile, lane);
        }
    }
}

template <int Phase, int kSharedStride>
__device__ __forceinline__ void k4Route2Schedule(TileNvfp4CombinePointersV2 const& ptrs,
    Route2FullTokenPackedPolicy::Shared const& shared, int ctaBase, int tokensInCta, int warp, int lane)
{
    using Policy = Route2FullTokenPackedPolicy;
    using Window = Policy::Window;
    int const begin = Window::template decodeBegin<Phase>(0, tokensInCta);
    int const end = Window::template decodeEnd<Phase>(0, tokensInCta);
    for (int work = begin + warp; work < end; work += kK4Warps)
    {
        int const tokenSlot = work / Policy::kQuadsPerToken;
        int const quadInToken = work % Policy::kQuadsPerToken;
        if constexpr (kSharedStride == kK1SelectedSharedStrideDefaultBytes)
        {
            Policy::template decodeFastPhase<Phase>(ptrs, shared, ctaBase, tokenSlot, quadInToken, lane);
        }
        else
        {
            k4Route2DecodeQuadPadded<kSharedStride>(ptrs, shared.packedCache, ctaBase, tokenSlot, quadInToken, lane);
        }
    }
}

// -----------------------------------------------------------------------------
// EP4/8/16/32 Route8HalfPacked K4 kernel. One token per CTA and all eight route
// positions are inspected; invalid/deduplicated positions decode as zero.
// -----------------------------------------------------------------------------
// Resource envelope: hold 5 resident CTAs/SM (65536 / (5*224) = 58 reg/thread
// ceiling) for the register-fold decode path, which trades the shuffle tree for a
// slightly larger per-lane hot set; it must not force a register spill.
template <int kSharedStride>
__global__ __launch_bounds__(kK4Threads, 5) void moeA2APushK4Route8Kernel(TileNvfp4CombinePointersV2 ptrs,
    std::uint8_t* controlBase, int localNumTokens, int rankId, int epSize, int rRuntime)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    using Policy = Route8HalfPackedPolicy;
    using Window = Policy::Window;

    // Wait on the programmatic dependency (K3 on the owner stream) before
    // touching any local landing byte.
    cudaGridDependencySynchronize();

    __shared__ CachedRoute routeCache[kTopK];
    __shared__ alignas(128) std::uint8_t packedCache[kStageRows * kSharedStride];
    __shared__ std::uint32_t routeValidMaskShared;
    __shared__ std::uint32_t routeBulkMaskShared;
    __shared__ std::uint64_t tmaBar[2];

    Policy::Shared shared{routeCache, /*scaleCache=*/nullptr, /*qCache=*/nullptr, &routeValidMaskShared,
        &routeBulkMaskShared, tmaBar, packedCache};

    if (threadIdx.x == 0)
    {
        mbarInit(&tmaBar[0], 1);
        mbarInit(&tmaBar[1], 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
        // EXACTLY ONE non-spin acquire load of localGate[C]. controlBase is this
        // consumer's already-hopped own row (== control_peer_base[rank_id]).
        k4AcquireLocalGateOnce(controlBase);
    }

    // Every valid route is bulk-eligible (there is no per-peer bulk gating on the
    // push path): a full peer mask makes routeBulkMask == the valid-route mask,
    // so the family builder yields issueMask == all valid local records.
    std::uint64_t const fullPeerMaskLo
        = epSize >= 64 ? ~static_cast<std::uint64_t>(0) : ((static_cast<std::uint64_t>(1) << epSize) - 1);
    std::uint64_t const fullPeerMaskHi = 0;
    Policy::buildRouteCacheAndMasks(ptrs, localNumTokens, /*epRank=*/0, epSize, rRuntime,
        /*graphStaticMaxTokensPerRank=*/rRuntime, fullPeerMaskLo, fullPeerMaskHi, shared);
    (void) rankId;

    // Publish the route masks the family builder just wrote (routeBulkMask),
    // then clear ONLY the invalid rows below -- not the unconditional 35840 B clear.
    __syncthreads();
    asm volatile("fence.acquire.sys;" ::: "memory");

    if (localNumTokens == 0)
    {
        return;
    }
    int const ctaBase = static_cast<int>(blockIdx.x) * Policy::kTokensPerCta;
    int const remainingTokens = localNumTokens - ctaBase;
    int const tokensInCta = remainingTokens < Policy::kTokensPerCta ? remainingTokens : Policy::kTokensPerCta;
    if (tokensInCta <= 0)
    {
        return;
    }

    int const warp = static_cast<int>(threadIdx.x) >> 5;
    int const lane = static_cast<int>(threadIdx.x) & 31;
    std::uint32_t const activeRowsMask = Window::activeRowsMask(tokensInCta);
    std::uint32_t const issueMask = Policy::bulkStageMask(shared) & activeRowsMask;

    // Clear only the rows the TMA will NOT overwrite (invalid routes -> exact +0.0f
    // leaf). All-valid tokens clear nothing; this removes the per-token 35840 B memset
    // that was exposed preparation work in the baseline.
    k4ZeroInvalidRows<kSharedStride>(packedCache, issueMask);
    __syncthreads();

    if (threadIdx.x == 0 && issueMask != 0U)
    {
        asm volatile("fence.proxy.async;" ::: "memory");
        if constexpr (kSharedStride == kK1SelectedSharedStrideDefaultBytes)
        {
            // Default stride: reuse the family TMA issue verbatim (zeroed controls
            // => no skew/delay spin; poison untouched).
            TmaRuntimeControls const controls{fullPeerMaskLo, fullPeerMaskHi, 0, 0U, 0, 0, 0, nullptr};
            issueTmaPhase<0, Policy>(shared, issueMask, controls);
            issueTmaPhase<1, Policy>(shared, issueMask, controls);
        }
        else
        {
            k4IssueTmaPhasePadded<0, Policy, kSharedStride>(shared, issueMask);
            k4IssueTmaPhasePadded<1, Policy, kSharedStride>(shared, issueMask);
        }
    }
    __syncthreads();

    std::uint32_t const phase0Mask = Window::template phaseRowsMask<0>(issueMask);
    if (phase0Mask != 0U)
    {
        mbarWaitParityAllConsumers(&tmaBar[0], 0U);
    }
    k4Route8Schedule<0, kSharedStride>(ptrs, shared, ctaBase, tokensInCta, warp, lane);

    std::uint32_t const phase1Mask = Window::template phaseRowsMask<1>(issueMask);
    if (phase1Mask != 0U)
    {
        mbarWaitParityAllConsumers(&tmaBar[1], 0U);
    }
    k4Route8Schedule<1, kSharedStride>(ptrs, shared, ctaBase, tokensInCta, warp, lane);
#else
    (void) ptrs;
    (void) controlBase;
    (void) localNumTokens;
    (void) rankId;
    (void) epSize;
    (void) rRuntime;
#endif
}

// -----------------------------------------------------------------------------
// EP2 Route2FullTokenPacked K4 kernel. Four tokens per CTA, up to two local
// records per token; each valid record is one whole 4480-B local TMA, phase-split
// by rows (0..3 / 4..7).
// -----------------------------------------------------------------------------
template <int kSharedStride>
__global__ __launch_bounds__(kK4Threads) void moeA2APushK4Route2Kernel(TileNvfp4CombinePointersV2 ptrs,
    std::uint8_t* controlBase, int localNumTokens, int rankId, int epSize, int rRuntime)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    using Policy = Route2FullTokenPackedPolicy;
    using Window = Policy::Window;

    cudaGridDependencySynchronize();

    __shared__ CachedRoute routeCache[Policy::kTokensPerCta][kTopK];
    __shared__ CachedRoute recordCache[kStageRows];
    __shared__ alignas(128) std::uint8_t packedCache[kStageRows * kSharedStride];
    __shared__ std::uint32_t routeValidMaskShared;
    __shared__ std::uint32_t tokenFastMaskShared;
    __shared__ std::uint32_t recordBulkMaskShared;
    __shared__ std::uint64_t tmaBar[2];

    Policy::Shared shared{&routeCache[0][0], recordCache, /*scaleCache=*/nullptr,
        /*qCache=*/nullptr, &routeValidMaskShared, &tokenFastMaskShared, &recordBulkMaskShared, tmaBar, packedCache};

    if (threadIdx.x == 0)
    {
        mbarInit(&tmaBar[0], 1);
        mbarInit(&tmaBar[1], 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
        // EXACTLY ONE non-spin acquire of localGate[C]; controlBase is the
        // already-hopped own row (== control_peer_base[rank_id]).
        k4AcquireLocalGateOnce(controlBase);
    }

    std::uint64_t const fullPeerMaskLo
        = epSize >= 64 ? ~static_cast<std::uint64_t>(0) : ((static_cast<std::uint64_t>(1) << epSize) - 1);
    std::uint64_t const fullPeerMaskHi = 0;
    Policy::buildRouteCacheAndMasks(ptrs, localNumTokens, /*epRank=*/0, epSize, rRuntime,
        /*graphStaticMaxTokensPerRank=*/rRuntime, fullPeerMaskLo, fullPeerMaskHi, shared);
    (void) rankId;

    // Publish the route masks the family builder just wrote (routeBulkMask),
    // then clear ONLY the invalid rows below -- not the unconditional 35840 B clear.
    __syncthreads();
    asm volatile("fence.acquire.sys;" ::: "memory");

    if (localNumTokens == 0)
    {
        return;
    }
    int const ctaBase = static_cast<int>(blockIdx.x) * Policy::kTokensPerCta;
    int const remainingTokens = localNumTokens - ctaBase;
    int const tokensInCta = remainingTokens < Policy::kTokensPerCta ? remainingTokens : Policy::kTokensPerCta;
    if (tokensInCta <= 0)
    {
        return;
    }

    int const warp = static_cast<int>(threadIdx.x) >> 5;
    int const lane = static_cast<int>(threadIdx.x) & 31;
    std::uint32_t const activeRowsMask = Window::activeRowsMask(tokensInCta);
    std::uint32_t const issueMask = Policy::bulkStageMask(shared) & activeRowsMask;

    // Clear only the rows the TMA will NOT overwrite (invalid routes -> exact +0.0f
    // leaf). All-valid tokens clear nothing; this removes the per-token 35840 B memset
    // that was exposed preparation work in the baseline.
    k4ZeroInvalidRows<kSharedStride>(packedCache, issueMask);
    __syncthreads();

    if (threadIdx.x == 0 && issueMask != 0U)
    {
        asm volatile("fence.proxy.async;" ::: "memory");
        if constexpr (kSharedStride == kK1SelectedSharedStrideDefaultBytes)
        {
            TmaRuntimeControls const controls{fullPeerMaskLo, fullPeerMaskHi, 0, 0U, 0, 0, 0, nullptr};
            issueTmaPhase<0, Policy>(shared, issueMask, controls);
            issueTmaPhase<1, Policy>(shared, issueMask, controls);
        }
        else
        {
            k4IssueTmaPhasePadded<0, Policy, kSharedStride>(shared, issueMask);
            k4IssueTmaPhasePadded<1, Policy, kSharedStride>(shared, issueMask);
        }
    }
    __syncthreads();

    std::uint32_t const phase0Mask = Window::template phaseRowsMask<0>(issueMask);
    if (phase0Mask != 0U)
    {
        mbarWaitParityAllConsumers(&tmaBar[0], 0U);
    }
    k4Route2Schedule<0, kSharedStride>(ptrs, shared, ctaBase, tokensInCta, warp, lane);

    std::uint32_t const phase1Mask = Window::template phaseRowsMask<1>(issueMask);
    if (phase1Mask != 0U)
    {
        mbarWaitParityAllConsumers(&tmaBar[1], 0U);
    }
    k4Route2Schedule<1, kSharedStride>(ptrs, shared, ctaBase, tokensInCta, warp, lane);
#else
    (void) ptrs;
    (void) controlBase;
    (void) localNumTokens;
    (void) rankId;
    (void) epSize;
    (void) rRuntime;
#endif
}

// =============================================================================
// K4 v2 for the Route8 (EP4/8/16/32) wire, TRTLLM_PUSH_K4_V2=1 (default off).
//
// Same inputs, same gate acquire, same bytes staged by TMA, bitwise-identical
// output. What it removes from the stock kernel's per-CTA chain:
//   - the route arrays are read with independent loads (send index and target
//     rank together, one latency instead of two dependent ones), by one warp that
//     also performs the gate acquire, initialises the mbarriers and issues the TMA:
//     one __syncthreads before the decode instead of three;
//   - those route loads (L2, ld.global.cg) and the mbarrier init come before the
//     grid-dependency wait. They read only this layer's dispatch output: a K4 CTA
//     starts no earlier than K23 lets it (at K23's exit, or after K23's own wait on
//     K1 under TRTLLM_PUSH_K23_PDL), so K1, and before it the dispatch, have
//     completed. The gate acquire and every landing-plane byte stay after the wait;
//   - no zeroing of the invalid (deduplicated) record rows: they are never read;
//   - the decode skips the invalid positions. The stock register fold reduces all
//     eight positions with the frozen tree ((v0+v1)+(v2+v3))+((v4+v5)+(v6+v7)),
//     an invalid position contributing exact +0.0f (its zeroed row decodes to
//     +0 * +0). Here an invalid position's eight values are the constant +0.0f
//     and the same tree of __fadd_rn runs on the same operands, so every output
//     bit is the stock's (including the sign of zero);
//   - 192 threads and the max shared-memory carveout: with 32 resident warps per SM
//     the stock 224-thread K4 fits four CTAs per SM; at 6 warps five CTAs fit, so
//     up to five tokens per SM run in one wave.
//     The 28 four-tile groups of a token (14 per phase) go round-robin over the six
//     warps; the output slots and the per-slot arithmetic are unchanged.
// =============================================================================
constexpr int kK4V2Threads = 192;
constexpr int kK4V2Warps = kK4V2Threads / 32;
constexpr int kK4V2TileGroups = kTilesPerToken / 4;    // 28 groups of four tiles
constexpr int kK4V2Phase0Groups = kK4V2TileGroups / 2; // groups [0, 14) are phase 0

// One position's eight values of this lane's (tile, blockInTile, pairHalf) output
// slot, decoded from its staged record row: byte-for-byte the stock register
// fold's per-position arithmetic (k4Route8DecodeTileRegFold).
__device__ __forceinline__ void k4v2DecodePosition(
    std::uint8_t const* rec, int tile, int blockInTile, int pairHalf, float (&v)[kPairsPerStore * 2])
{
    using Window = Route8HalfPackedPolicy::Window;
    using Format = Nvfp4Format;
    std::uint64_t const scalePack = *reinterpret_cast<std::uint64_t const*>(rec + Window::sfByteOffset(tile));
    std::uint8_t const rawBlockScale = static_cast<std::uint8_t>(scalePack >> (blockInTile * 8));
    float const globalD = __uint_as_float(static_cast<std::uint32_t>(scalePack >> 32));
    __nv_fp8_e4m3 blockScale;
    blockScale.__x = rawBlockScale;
    float const scaledBlock = __fmul_rn(static_cast<float>(blockScale), globalD);
    std::uint8_t const* qBlock = rec + Window::qByteOffset(tile) + blockInTile * Format::kQBytesPerBlock;
    std::uint32_t const qHalf = *reinterpret_cast<std::uint32_t const*>(qBlock + pairHalf * 4);
#pragma unroll
    for (int p = 0; p < kPairsPerBlock / 2; ++p)
    {
        __nv_fp4x2_e2m1 qPair;
        qPair.__x = static_cast<__nv_fp4x2_storage_t>((qHalf >> (p * 8)) & 0xFFU);
        float2 const q = static_cast<float2>(qPair);
        v[2 * p] = __fmul_rn(q.x, scaledBlock);
        v[2 * p + 1] = __fmul_rn(q.y, scaledBlock);
    }
}

// Level one of the frozen tree for positions (2j, 2j+1): __fadd_rn(v_2j, v_2j+1)
// per value, an invalid position being +0.0f. validMask is CTA-uniform, so the
// branches never diverge and an invalid position costs no decode.
__device__ __forceinline__ void k4v2Level1(std::uint8_t const* packedCache, std::uint32_t validMask, int j, int tile,
    int blockInTile, int pairHalf, float (&out)[kPairsPerStore * 2])
{
    float a[kPairsPerStore * 2];
    float b[kPairsPerStore * 2];
    if ((validMask >> (2 * j)) & 1U)
    {
        k4v2DecodePosition(
            packedCache + static_cast<std::size_t>(2 * j) * kK1PackedRecordBytes, tile, blockInTile, pairHalf, a);
    }
    else
    {
#pragma unroll
        for (int e = 0; e < kPairsPerStore * 2; ++e)
        {
            a[e] = 0.0F;
        }
    }
    if ((validMask >> (2 * j + 1)) & 1U)
    {
        k4v2DecodePosition(
            packedCache + static_cast<std::size_t>(2 * j + 1) * kK1PackedRecordBytes, tile, blockInTile, pairHalf, b);
    }
    else
    {
#pragma unroll
        for (int e = 0; e < kPairsPerStore * 2; ++e)
        {
            b[e] = 0.0F;
        }
    }
#pragma unroll
    for (int e = 0; e < kPairsPerStore * 2; ++e)
    {
        out[e] = __fadd_rn(a[e], b[e]);
    }
}

// One lane's 16-byte output slot: ((L0 + L1) + (L2 + L3)) with Lj the level-one
// sums, exactly k4FoldRoute8Frozen's association per value.
__device__ __forceinline__ void k4v2DecodeTile(
    std::uint8_t const* packedCache, std::uint32_t validMask, __nv_bfloat16* outputRecord, int tile, int lane)
{
    int const pairHalf = lane & 1;
    int const blockInTile = (lane >> 1) & 3;
    float lo[kPairsPerStore * 2];
    float hi[kPairsPerStore * 2];
    float left[kPairsPerStore * 2];
    k4v2Level1(packedCache, validMask, 0, tile, blockInTile, pairHalf, lo);
    k4v2Level1(packedCache, validMask, 1, tile, blockInTile, pairHalf, hi);
#pragma unroll
    for (int e = 0; e < kPairsPerStore * 2; ++e)
    {
        left[e] = __fadd_rn(lo[e], hi[e]);
    }
    k4v2Level1(packedCache, validMask, 2, tile, blockInTile, pairHalf, lo);
    k4v2Level1(packedCache, validMask, 3, tile, blockInTile, pairHalf, hi);
    uint4 words;
    std::uint32_t* w = &words.x;
#pragma unroll
    for (int p = 0; p < kPairsPerStore; ++p)
    {
        float const x = __fadd_rn(left[2 * p], __fadd_rn(lo[2 * p], hi[2 * p]));
        float const y = __fadd_rn(left[2 * p + 1], __fadd_rn(lo[2 * p + 1], hi[2 * p + 1]));
        w[p] = packBf16Pair(x, y);
    }
    __nv_bfloat16* output = outputRecord + static_cast<std::size_t>(tile) * kTileElements
        + blockInTile * kScaleBlockElements + pairHalf * (kPairsPerStore * 2);
    *reinterpret_cast<uint4*>(output) = words;
}

__global__ __launch_bounds__(kK4V2Threads, 5) void moeA2APushK4V2Route8Kernel(std::uint8_t const* packedLocalBase,
    std::uint8_t* controlBase, int const* topkTargetRanks, int const* topkSendIndices, __nv_bfloat16* output,
    int localNumTokens, int epSize, int rRuntime)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    constexpr unsigned kPhaseBytes = Nvfp4Format::kQHalfAlignedBytes + Nvfp4Format::kSfHalfBytes; // 2240
    static_assert(2 * kPhaseBytes == kK1PackedRecordBytes, "two phase slices tile the packed record");

    __shared__ alignas(128) std::uint8_t packedCache[kTopK * kK1PackedRecordBytes];
    __shared__ std::uint64_t tmaBar[2];
    __shared__ std::uint32_t validMaskShared;

    int const warp = static_cast<int>(threadIdx.x) >> 5;
    int const lane = static_cast<int>(threadIdx.x) & 31;
    int const token = static_cast<int>(blockIdx.x);

    if (warp == 0)
    {
        int dstIdx = -1;
        int targetRank = 0;
        if (localNumTokens != 0 && lane < kTopK)
        {
            int const route = token * kTopK + lane;
            dstIdx = __ldcg(topkSendIndices + route);
            targetRank = __ldcg(topkTargetRanks + route);
        }
        if (lane == 0)
        {
            mbarInit(&tmaBar[0], 1);
            mbarInit(&tmaBar[1], 1);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
        bool const valid = lane < kTopK && dstIdx >= 0;
        if (valid && __builtin_expect(targetRank < 0 || targetRank >= epSize || dstIdx >= rRuntime, 0))
        {
            asm volatile("trap;");
        }
        std::uint32_t const validMask = __ballot_sync(kFullWarpMask, valid) & 0xFFU;
        std::uint8_t const* record = valid ? packedLocalBase
                + (static_cast<std::size_t>(targetRank) * static_cast<std::size_t>(rRuntime)
                      + static_cast<std::size_t>(dstIdx))
                    * kK1PackedRecordBytes
                                           : nullptr;
        std::uint8_t const* records[kTopK];
#pragma unroll
        for (int r = 0; r < kTopK; ++r)
        {
            records[r] = reinterpret_cast<std::uint8_t const*>(
                __shfl_sync(kFullWarpMask, reinterpret_cast<unsigned long long>(record), r));
        }
        // The landing bytes and the gate come from K23 and the K1s before it: wait on
        // the programmatic dependency (K23 on the owner stream) first, as the stock K4.
        cudaGridDependencySynchronize();
        if (lane == 0)
        {
            // EXACTLY ONE non-spin acquire of localGate[C] (the K3->K4 hand-off),
            // as the stock K4; controlBase is this consumer's already-hopped row.
            k4AcquireLocalGateOnce(controlBase);
            validMaskShared = validMask;
            if (validMask != 0U)
            {
                // The stock K4's acquire scope and proxy order, on the issuing thread:
                // gate acquire (above) -> fence.acquire.sys -> fence.proxy.async -> TMA.
                asm volatile("fence.acquire.sys;" ::: "memory");
                asm volatile("fence.proxy.async;" ::: "memory");
                unsigned const rows = static_cast<unsigned>(__popc(validMask));
                mbarArriveExpectTx(&tmaBar[0], rows * kPhaseBytes);
                mbarArriveExpectTx(&tmaBar[1], rows * kPhaseBytes);
#pragma unroll
                for (int r = 0; r < kTopK; ++r)
                {
                    if ((validMask >> r) & 1U)
                    {
                        std::uint8_t* row = packedCache + static_cast<std::size_t>(r) * kK1PackedRecordBytes;
                        bulk1d(row, records[r], kPhaseBytes, &tmaBar[0]);
                        bulk1d(row + kPhaseBytes, records[r] + kPhaseBytes, kPhaseBytes, &tmaBar[1]);
                    }
                }
            }
        }
    }
    else
    {
        cudaGridDependencySynchronize();
    }
    __syncthreads(); // publishes the mbarrier init and validMaskShared to every warp

    if (localNumTokens == 0)
    {
        return;
    }
    std::uint32_t const validMask = validMaskShared;
    __nv_bfloat16* outputRecord = output + static_cast<std::size_t>(token) * kHiddenElements;
    int const tileInGroup = lane >> 3; // 0..3

    // Groups g = warp, warp + 6, ... (< 28) in increasing order: a warp's phase-0
    // groups (g < 14, tiles [0, 56)) come first, then it waits for phase 1.
    if (validMask != 0U)
    {
        mbarWaitParityAllConsumers(&tmaBar[0], 0U);
    }
    bool phase1Ready = validMask == 0U;
#pragma unroll 1
    for (int group = warp; group < kK4V2TileGroups; group += kK4V2Warps)
    {
        if (group >= kK4V2Phase0Groups && !phase1Ready)
        {
            mbarWaitParityAllConsumers(&tmaBar[1], 0U);
            phase1Ready = true;
        }
        k4v2DecodeTile(packedCache, validMask, outputRecord, group * 4 + tileInGroup, lane);
    }
#else
    (void) packedLocalBase;
    (void) controlBase;
    (void) topkTargetRanks;
    (void) topkSendIndices;
    (void) output;
    (void) localNumTokens;
    (void) epSize;
    (void) rRuntime;
#endif
}

static_assert(Route8HalfPackedPolicy::Window::kQHalfTiles == kK4V2Phase0Groups * 4,
    "phase 0 is exactly the first 14 groups of four tiles");

// TRTLLM_PUSH_K4_V2=1 selects K4 v2 (read at each launch, so a captured
// graph bakes the choice; default off). Printed once per process when it engages.
bool k4V2Requested()
{
    char const* env = std::getenv("TRTLLM_PUSH_K4_V2");
    return env != nullptr && env[0] == '1';
}

// Build the family combine-pointers view for K4. The push path holds NO remote
// peer pointers and NO completion flags. Each sfPeer entry is instead the direct
// base of a producer band in this consumer's LOCAL landing plane. The policy
// builders receive address-rank zero, so their slot expression is:
//     sfPeer[producer] + (0*R + dstIdx) * 4480
//   = packed_local_base + (producer*R + dstIdx) * 4480    (== K1's landing slot)
// This avoids forming a pointer before packed_local_base for nonzero consumers.
// The GLOBAL landing/read stride stays the tight 4480-B record (matching K1);
// selected_shared_stride_bytes affects only the SHARED staging row.
inline TileNvfp4CombinePointersV2 makeLocalPointers(K4Params const& params)
{
    TileNvfp4CombinePointersV2 pointers{};
    for (int producerRank = 0; producerRank < params.ep_size; ++producerRank)
    {
        std::size_t const producerBandBytes = static_cast<std::size_t>(producerRank)
            * static_cast<std::size_t>(params.r_runtime) * static_cast<std::size_t>(kK1PackedRecordBytes);
        pointers.sfPeer[producerRank] = params.packed_local_base + producerBandBytes;
    }
    // qPeer / completionFlags / flagVal stay null: the packed fast path never
    // reads them, and K4 must not touch the readiness plane.
    pointers.topkTargetRanks = params.topk_target_ranks;
    pointers.topkSendIndices = params.topk_send_indices;
    pointers.output = static_cast<__nv_bfloat16*>(params.output_data);
    return pointers;
}

// Host-side parameter validation, mirroring the validateCombineParamsPacked<EP>
// patterns adapted to the K4Params surface.
void validateK4Params(K4Params const& params)
{
    // Frozen launch width: any other block width is a HOST_REJECT.
    TLLM_CHECK_WITH_INFO(params.geometry.threads == kK1BaselineThreads,
        "K4 requires the frozen baseline launch width (%d); got geometry.threads=%d "
        "(HOST_REJECT)",
        kK1BaselineThreads, params.geometry.threads);

    bool const supportedEp = params.ep_size == 2 || params.ep_size == 4 || params.ep_size == 8 || params.ep_size == 16
        || params.ep_size == 32;
    TLLM_CHECK_WITH_INFO(supportedEp, "K4 supports EP in {2,4,8,16,32}; got ep_size=%d", params.ep_size);
    TLLM_CHECK_WITH_INFO(params.rank_id >= 0 && params.rank_id < params.ep_size,
        "K4 consumer rank_id %d out of [0, ep_size=%d)", params.rank_id, params.ep_size);
    TLLM_CHECK_WITH_INFO(params.top_k == kTopK, "K4 requires TOP_K=%d; got %d", kTopK, params.top_k);
    TLLM_CHECK_WITH_INFO(params.r_runtime > 0, "K4 requires r_runtime > 0; got %d", params.r_runtime);
    TLLM_CHECK_WITH_INFO(
        params.local_num_tokens >= 0, "K4 local_num_tokens must be nonnegative; got %d", params.local_num_tokens);

    // records_per_token pins the wire layout. K4 uses Route2 for the two-record
    // FullToken wire and Route8 for both four/eight-record Half wires.
    int const expectedRecords = params.top_k < params.ep_size ? params.top_k : params.ep_size;
    TLLM_CHECK_WITH_INFO(params.records_per_token == expectedRecords,
        "K4 records_per_token must be min(top_k, ep_size)=%d; got %d", expectedRecords, params.records_per_token);

    // Packed layout identity must match the producing K1's 4480-B [Q|SF] record.
    TLLM_CHECK_WITH_INFO(
        params.packed_layout_version == static_cast<int>(MoeA2ACombinePayloadLayout::kTileNvfp4Compact),
        "K4 packed_layout_version must be the compact (packed-interleaved) layout");

    // The staging stride option: default 4480 or padded 4496; nothing else.
    TLLM_CHECK_WITH_INFO(params.selected_shared_stride_bytes == kK1SelectedSharedStrideDefaultBytes
            || params.selected_shared_stride_bytes == kK1SelectedSharedStrideAbBytes,
        "K4 selected_shared_stride_bytes must be %d (default) or %d (A/B); got %d", kK1SelectedSharedStrideDefaultBytes,
        kK1SelectedSharedStrideAbBytes, params.selected_shared_stride_bytes);

    // The control plane (K3 hand-off) and the local landing plane are always
    // required; the routing arrays and output only when there are output tokens.
    TLLM_CHECK_WITH_INFO(
        params.control_base != nullptr, "K4 requires a non-null compact_row_v2 control_base for the K3 gate acquire");
    TLLM_CHECK_WITH_INFO(reinterpret_cast<std::uintptr_t>(params.control_base) % alignof(std::uint64_t) == 0,
        "K4 control_base must be u64-aligned for the raw-u64 flag ABI");
    TLLM_CHECK_WITH_INFO(params.packed_local_base != nullptr, "K4 requires a non-null local packed landing plane");
    TLLM_CHECK_WITH_INFO(reinterpret_cast<std::uintptr_t>(params.packed_local_base) % 128 == 0,
        "K4 local packed landing plane must be 128-byte aligned");

    if (params.local_num_tokens > 0)
    {
        TLLM_CHECK_WITH_INFO(
            params.output_data != nullptr && params.topk_target_ranks != nullptr && params.topk_send_indices != nullptr,
            "K4 requires output and TOP_K routing arrays for nonzero local tokens");
        TLLM_CHECK_WITH_INFO(
            reinterpret_cast<std::uintptr_t>(params.output_data) % 16 == 0, "K4 BF16 output must be 16-byte aligned");
    }
}

} // namespace

// -----------------------------------------------------------------------------
// Host launch. Validate, build direct local producer-band pointers, and dispatch
// by EP family: EP2 -> Route2; EP4/8/16/32 -> Route8. EP4 deliberately scans
// all top-k positions because valid first occurrences are not compacted.
// Both staging-stride instantiations use PDL (the kernel performs the dependency
// wait at entry). K4 must be enqueued on the owner stream only after K3.
// -----------------------------------------------------------------------------
void launchK4TopkReduce(K4Params const& params, cudaStream_t stream)
{
    validateK4Params(params);

    TileNvfp4CombinePointersV2 const pointers = makeLocalPointers(params);

    int const tokensPerCta = params.records_per_token == 2 ? Route2FullTokenPackedPolicy::kTokensPerCta
                                                           : Route8HalfPackedPolicy::kTokensPerCta;
    int const grid = params.local_num_tokens == 0 ? 1 : (params.local_num_tokens + tokensPerCta - 1) / tokensPerCta;

    if (params.records_per_token != 2 && params.selected_shared_stride_bytes == kK1SelectedSharedStrideDefaultBytes
        && k4V2Requested())
    {
        // K4 v2 (Route8 wire, default staging stride only; every other
        // shape keeps the stock kernel below).
        static bool const configured = []
        {
            TLLM_CUDA_CHECK(cudaFuncSetAttribute(moeA2APushK4V2Route8Kernel,
                cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
            return true;
        }();
        (void) configured;
        static bool const announced = [&]
        {
            TLLM_LOG_INFO(
                "[TRTLLM_PUSH_K4_V2] push-combine K4 v2 active: one-warp route/gate/TMA prologue, route loads "
                "before the dependency wait, invalid positions neither zeroed nor decoded (same fold, bitwise), "
                "192 threads x 5 CTAs/SM");
            return true;
        }();
        (void) announced;
        tensorrt_llm::common::launchWithPdlWhenEnabled("moeA2APushK4V2Route8Kernel", moeA2APushK4V2Route8Kernel,
            dim3(grid), dim3(kK4V2Threads), 0, stream, static_cast<std::uint8_t const*>(params.packed_local_base),
            params.control_base, params.topk_target_ranks, params.topk_send_indices,
            static_cast<__nv_bfloat16*>(params.output_data), params.local_num_tokens, params.ep_size, params.r_runtime);
    }
    else if (params.records_per_token != 2)
    {
        if (params.selected_shared_stride_bytes == kK1SelectedSharedStrideDefaultBytes)
        {
            tensorrt_llm::common::launchWithPdlWhenEnabled("moeA2APushK4Route8Kernel",
                moeA2APushK4Route8Kernel<kK1SelectedSharedStrideDefaultBytes>, dim3(grid), dim3(kK4Threads), 0, stream,
                pointers, params.control_base, params.local_num_tokens, params.rank_id, params.ep_size,
                params.r_runtime);
        }
        else
        {
            tensorrt_llm::common::launchWithPdlWhenEnabled("moeA2APushK4Route8KernelAb",
                moeA2APushK4Route8Kernel<kK1SelectedSharedStrideAbBytes>, dim3(grid), dim3(kK4Threads), 0, stream,
                pointers, params.control_base, params.local_num_tokens, params.rank_id, params.ep_size,
                params.r_runtime);
        }
    }
    else
    {
        if (params.selected_shared_stride_bytes == kK1SelectedSharedStrideDefaultBytes)
        {
            tensorrt_llm::common::launchWithPdlWhenEnabled("moeA2APushK4Route2Kernel",
                moeA2APushK4Route2Kernel<kK1SelectedSharedStrideDefaultBytes>, dim3(grid), dim3(kK4Threads), 0, stream,
                pointers, params.control_base, params.local_num_tokens, params.rank_id, params.ep_size,
                params.r_runtime);
        }
        else
        {
            tensorrt_llm::common::launchWithPdlWhenEnabled("moeA2APushK4Route2KernelAb",
                moeA2APushK4Route2Kernel<kK1SelectedSharedStrideAbBytes>, dim3(grid), dim3(kK4Threads), 0, stream,
                pointers, params.control_base, params.local_num_tokens, params.rank_id, params.ep_size,
                params.r_runtime);
        }
    }
    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels::moe_comm

TRTLLM_NAMESPACE_END
