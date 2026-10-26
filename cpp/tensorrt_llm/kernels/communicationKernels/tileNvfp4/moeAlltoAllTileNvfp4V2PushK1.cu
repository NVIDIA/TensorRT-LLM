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
 * K1 producer-push payload kernel for the packed-NVFP4 tileNvfp4 V2 family.
 *
 * K1 is the standalone PRODUCER step: this rank P quantizes each of its BF16
 * activation tokens into the family's packed [Q|SF] 4480-B record with the
 * family NVFP4 quantizer (loadPackedVec16 / packedAbsMax / fp32VecToE2m1 /
 * quantizeSixteen / quantizeAndStoreTile, defined below) and PUSHES each whole
 * record straight into the destination
 * consumer's packed landing plane with a single strong system-scope raw-TMA
 * bulk S2G (cp_async_bulk_s2g_sys, moeAlltoAllTileNvfp4V2PushAsyncOps.cuh).
 *
 * Contract highlights (moeAlltoAllTileNvfp4V2PushK1K4.h):
 *   * Persistent, imbalance-safe, replay-fixed grid. The physical grid width is
 *     the resolved per-wave CTA ceiling cta_cap (fixed at launch, so it is baked
 *     into any captured CUDA graph and identical across replays). CTAs walk the
 *     true_compact_prefix_v2 work domain: this producer builds an
 *     E-entry band prefix over the selected destination permutation perm_P[b]
 *     (prefix[b+1]=prefix[b]+recv_counters[perm_P[b]], Nrec=prefix[E]) and iterates
 *     ONLY the compact ordinal u in [0, Nrec) with a CTA-granular grid stride of
 *     cta_cap. Each u maps by inverse-prefix to (band b, t) with C=perm_P[b],
 *     t=u-prefix[b]; NO padding slot is ever scanned (padding_slot_scan_count==0),
 *     unlike an E*R+skip scan. CTAs with blockIdx.x >= Nrec read the
 *     prefix and early-exit with no coverage, so live CTAs are exactly
 *     min(Nrec, cta_cap) with deterministic, replay-stable coverage.
 *   * The destination-band order perm_P[b] is one of TWO fixed
 *     candidates (never runtime-selected per replay): linear_destination_band_v1
 *     (perm_P[b]=b) or producer_rank_rotated_remote_first_v1 (perm_P[b]=(P+1+b) mod
 *     E). The rotated candidate issues the slowest cross-node PUSHes FIRST so their
 *     fabric drain overlaps maximally and the K3 consumer-gate exposed wait (the
 *     fabric-drain tail) collapses -- this is the anti-incast lever. For any fixed
 *     band b, P -> perm_P[b] is a bijection over [0,E), so synchronized producers
 *     never co-hammer one consumer. Coverage is permutation-invariant: source=C*R+t
 *     and landing=P*R+t are unchanged, so the bytewise/bitwise output is identical.
 *   * blockDim.x == 224 is the MANDATORY baseline ("deterministic_224_baseline_v1");
 *     any other width is HOST_REJECTed. 224 == 56 tile groups * 4 lanes is exactly
 *     the family packed-prepare shape, so each token quantizes in two 56-tile waves
 *     with one 4-lane group per tile per wave.
 *   * Each staged record is built in an alignas(128) __shared__ double buffer
 *     packedStage[2][4480] (NOT global). FullToken (EP2, records_per_token==2)
 *     stages the contiguous [ Q@0 | SF@3584 ] layout; Half (EP4/EP8/EP16/EP32,
 *     records_per_token in {4,8}) stages the interleaved
 *     [ Q_h0@0 | SF_h0@1792 | Q_h1@2240 | SF_h1@4032 ] layout -- byte-identical to
 *     the family FullToken / Half packed prepares.
 *   * Publication of one staged record: EVERY writer issues
 *     fence_proxy_async_shared_cta() (order the generic-proxy quant stores ahead of
 *     the async-proxy bulk read), then __syncthreads(), then a SINGLE elected
 *     issuer (threadIdx.x == 0) performs EXACTLY ONE cp_async_bulk_s2g_sys(dst,
 *     stage, 4480) + cp_async_bulk_commit_group() to
 *     dst = packed_peer_base[C] + (P*r_runtime + t)*4480. The publication unit is
 *     the whole 4480-B record; Q / SF / half slices are never pushed separately.
 *   * S_alloc == S_active == 2 double buffer so quantize(n+1) overlaps push(n):
 *     before reusing a stage buffer the issuer drains with
 *     cp_async_bulk_wait_group<S_active-1>() and the CTA barrier republishes the
 *     freed buffer to every writer. The tail drains with cp_async_bulk_wait_group<0>().
 *
 * K1 touches ONLY payload bytes: it never writes the raw-u64 flag plane (that is
 * K2's job, moeAlltoAllTileNvfp4V2PushFlags.h) and never polls. The self record
 * (C == P) is quantized and pushed like any other -- there is no BF16 bypass.
 *
 * The quant math below defines the packed record bytes bit for bit; K4 decodes
 * them with the family Nvfp4Format.
 */

#include "moeAlltoAllTileNvfp4V2PushAsyncOps.cuh"
#include "moeAlltoAllTileNvfp4V2PushK1K4.h"
#include "moeAlltoAllTileNvfp4V2TmaFormats.cuh"

#include "tensorrt_llm/kernels/cudaAsyncOps.cuh"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::moe_comm
{
namespace
{

// -----------------------------------------------------------------------------
// Packed-record + quant geometry of the packed [Q|SF] record format, declared
// locally (not shared through a header); the values -- and the quant helpers below -- must stay
// bit-for-bit identical to the format K4 decodes. static_asserts below pin them
// against the Nvfp4Format definition in TmaFormats.cuh.
// -----------------------------------------------------------------------------
constexpr int kWarpSize = 32;
constexpr int kHiddenElements = 7168;
constexpr int kTileElements = 64;
constexpr int kScaleBlockElements = 16;
constexpr int kTilesPerToken = 112;
constexpr int kQBytesPerTile = 32;
constexpr int kSfBytesPerTile = 8;
constexpr int kBf16TokenBytes = 14336;
constexpr int kQPayloadBytesPerToken = 3584;
constexpr int kSfPayloadBytesPerToken = 896;
constexpr int kQAlignedBytes = kQPayloadBytesPerToken;                                     // 3584
constexpr int kPackedRecordStrideBytes = kQPayloadBytesPerToken + kSfPayloadBytesPerToken; // 4480
constexpr int kQHalfTiles = kTilesPerToken / 2;                                            // 56
constexpr int kQHalfAlignedBytes = kQHalfTiles * kQBytesPerTile;                           // 1792
constexpr int kSfHalfBytes = kQHalfTiles * kSfBytesPerTile;                                // 448
constexpr int kPackedHalfBytes = kQHalfAlignedBytes + kSfHalfBytes;                        // 2240
constexpr int kLanesPerTile = 4;
constexpr int kElementsPerLane = 16;
constexpr float kGlobalDenominator = 2688.0F;
constexpr std::uint32_t kMinimumEffectiveAmaxBits = 0x06280000U;

// The K1 record layout must equal the family packed record contract byte-for-byte.
static_assert(kPackedRecordStrideBytes == kK1PackedRecordBytes,
    "K1 packed record stride must mirror kK1PackedRecordBytes (4480)");
static_assert(kPackedRecordStrideBytes == tile_nvfp4_tma::Nvfp4Format::kPackedRecordBytes,
    "K1 packed record must mirror Nvfp4Format::kPackedRecordBytes");
static_assert(kQAlignedBytes == tile_nvfp4_tma::Nvfp4Format::kQAlignedBytes,
    "K1 FullToken SF start must mirror Nvfp4Format::kQAlignedBytes (3584)");
static_assert(kQHalfAlignedBytes == tile_nvfp4_tma::Nvfp4Format::kQHalfAlignedBytes,
    "K1 Half Q slice must mirror Nvfp4Format::kQHalfAlignedBytes (1792)");
static_assert(kSfHalfBytes == tile_nvfp4_tma::Nvfp4Format::kSfHalfBytes,
    "K1 Half SF slice must mirror Nvfp4Format::kSfHalfBytes (448)");
static_assert(
    kPackedHalfBytes * 2 == kPackedRecordStrideBytes, "two Half phase slices must tile the packed record exactly");
static_assert(kPackedRecordStrideBytes % 16 == 0, "raw-TMA b128 bulk S2G requires a 16-byte-multiple record size");

// -----------------------------------------------------------------------------
// K1 launch geometry (baseline). blockDim.x == 224 == 56 tile groups * 4 lanes
// is the family packed-prepare shape (two 56-tile waves cover the 112 tiles).
// S_alloc == S_active == 2 pins the shared double buffer.
// -----------------------------------------------------------------------------
constexpr int kK1Threads = kK1BaselineThreads;                         // 224
constexpr int kK1PrepareWaves = 2;
constexpr int kK1PrepareTileGroups = kTilesPerToken / kK1PrepareWaves; // 56
constexpr int kK1StageAlloc = 2;
constexpr int kK1StageActive = 2;
constexpr int kK1TopK = 8; // DeepSeek-R1 route slots per token

static_assert(kK1Threads == kK1PrepareTileGroups * kLanesPerTile,
    "224 threads == 56 tile groups * 4 lanes (one 4-lane group per tile per wave)");
static_assert(kK1PrepareTileGroups == kQHalfTiles,
    "wave 0 IS Half phase 0 (tiles 0..55); wave 1 IS Half phase 1 (tiles 56..111)");
static_assert(
    kK1StageAlloc == kK1BaselineGeometry.s_alloc, "K1 shared double buffer must match the frozen baseline s_alloc (2)");
static_assert(kK1StageActive == kK1BaselineGeometry.s_active,
    "K1 active pipeline depth must match the frozen baseline s_active (2)");
static_assert(kK1StageActive >= 1 && kK1StageActive <= kK1StageAlloc,
    "active pipeline depth must fit the allocated stage buffers");

// =============================================================================
// NVFP4 quant helpers of the packed record format. Do not edit: they define
// the record bytes bit for bit. Every store lands in generic address space, so the
// SAME helpers write the __shared__ stage record instead of the global plane.
// =============================================================================

struct alignas(32) B256
{
    uint4 a;
    uint4 b;
};

static_assert(sizeof(B256) == 32, "B256 must cover sixteen BF16 values");
static_assert(alignof(B256) == 32, "B256 must carry a 32-byte alignment contract");

union alignas(32) PackedVec16Bf16
{
    B256 wide;
    __nv_bfloat162 elts[8];
};

static_assert(sizeof(PackedVec16Bf16) == 32, "sixteen BF16 values must occupy 32 bytes");
static_assert(alignof(PackedVec16Bf16) == 32, "wide source view requires 32-byte alignment");

__device__ __forceinline__ void loadPackedVec16(__nv_bfloat16 const* input, PackedVec16Bf16& result)
{
    result.wide = *reinterpret_cast<B256 const*>(input);
}

__device__ __forceinline__ float minimumEffectiveAmax()
{
    return __uint_as_float(kMinimumEffectiveAmaxBits);
}

__device__ __forceinline__ float packedAbsMax(PackedVec16Bf16 const& vec)
{
    __nv_bfloat162 localMax = __habs2(vec.elts[0]);
#pragma unroll
    for (int i = 1; i < 8; ++i)
    {
        localMax = __hmax2(localMax, __habs2(vec.elts[i]));
    }
    return __bfloat162float(__hmax(localMax.x, localMax.y));
}

__device__ __forceinline__ float reciprocalApproximateFtz(float value)
{
    float result;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(result) : "f"(value));
    return result;
}

__device__ __forceinline__ std::uint32_t fp32VecToE2m1(float2 (&values)[4])
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    std::uint32_t result;
    asm volatile(
        "{\n"
        ".reg .b8 byte0;\n"
        ".reg .b8 byte1;\n"
        ".reg .b8 byte2;\n"
        ".reg .b8 byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte3, %8, %7;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "}"
        : "=r"(result)
        : "f"(values[0].x), "f"(values[0].y), "f"(values[1].x), "f"(values[1].y), "f"(values[2].x), "f"(values[2].y),
        "f"(values[3].x), "f"(values[3].y));
    return result;
#else
    return 0U;
#endif
}

__device__ __forceinline__ std::uint64_t quantizeSixteen(
    PackedVec16Bf16 const& vec, float blockAmax, float multiplier, std::uint8_t& scaleByteOut)
{
    float const scaleValueWide = multiplier * (blockAmax * reciprocalApproximateFtz(6.0F));
    __nv_fp8_e4m3 scaleValueNarrow = __nv_fp8_e4m3(scaleValueWide);
    scaleByteOut = scaleValueNarrow.__x;
    float const decodedScale = static_cast<float>(scaleValueNarrow);
    float const outputScale
        = blockAmax != 0.0F ? reciprocalApproximateFtz(decodedScale * reciprocalApproximateFtz(multiplier)) : 0.0F;

    std::uint32_t qWords[2];
    float2 fp32Values[4];
#pragma unroll
    for (int half = 0; half < 2; ++half)
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            fp32Values[i] = __bfloat1622float2(vec.elts[half * 4 + i]);
            fp32Values[i].x *= outputScale;
            fp32Values[i].y *= outputScale;
        }
        qWords[half] = fp32VecToE2m1(fp32Values);
    }
    return static_cast<std::uint64_t>(qWords[0]) | (static_cast<std::uint64_t>(qWords[1]) << 32);
}

__device__ __forceinline__ void quantizeAndStoreTile(
    PackedVec16Bf16 const& vec, int tile, int laneInTile, std::uint8_t* tokenQ, std::uint8_t* tokenSf)
{
    int const warpLane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    int const tileLaneBase = warpLane & ~(kLanesPerTile - 1);
    unsigned int const tileMask = 0xFU << tileLaneBase;

    // One 4-lane thread owns exactly one block16; compute its amax only once.
    float const blockAmax = packedAbsMax(vec);
    float tileAmax = blockAmax;
    tileAmax = fmaxf(tileAmax, __shfl_xor_sync(tileMask, tileAmax, 1));
    tileAmax = fmaxf(tileAmax, __shfl_xor_sync(tileMask, tileAmax, 2));

    std::uint8_t* tileQ = tokenQ + tile * kQBytesPerTile;
    std::uint8_t* tileSf = tokenSf + tile * kSfBytesPerTile;

    if (tileAmax == 0.0F)
    {
        *reinterpret_cast<std::uint64_t*>(tileQ + laneInTile * sizeof(std::uint64_t)) = 0U;
        if (laneInTile == 0)
        {
            constexpr std::uint64_t kCanonicalZeroScalePack = static_cast<std::uint64_t>(0x3F800000U) << 32;
            *reinterpret_cast<std::uint64_t*>(tileSf) = kCanonicalZeroScalePack;
        }
        return;
    }

    float const effectiveAmax = fmaxf(tileAmax, minimumEffectiveAmax());
    float const multiplier = __fdiv_rn(kGlobalDenominator, effectiveAmax);
    float const globalDequantScale = __fdiv_rn(effectiveAmax, kGlobalDenominator);

    std::uint8_t scaleByte;
    std::uint64_t const qWord = quantizeSixteen(vec, blockAmax, multiplier, scaleByte);
    *reinterpret_cast<std::uint64_t*>(tileQ + laneInTile * sizeof(std::uint64_t)) = qWord;

    std::uint32_t scaleGather = static_cast<std::uint32_t>(scaleByte) << (laneInTile * 8);
    scaleGather |= __shfl_xor_sync(tileMask, scaleGather, 1);
    scaleGather |= __shfl_xor_sync(tileMask, scaleGather, 2);
    if (laneInTile == 0)
    {
        std::uint64_t const scalePack = static_cast<std::uint64_t>(scaleGather)
            | (static_cast<std::uint64_t>(__float_as_uint(globalDequantScale)) << 32);
        *reinterpret_cast<std::uint64_t*>(tileSf) = scalePack;
    }
}

// =============================================================================
// K1 device args + kernel.
// =============================================================================

// POD kernel args (mirrors the family idiom of passing a pointer-table struct by
// value, e.g. TileNvfp4CombinePointersV2). Built on the host from K1PushParams.
struct K1PushDeviceArgs
{
    std::uint8_t* packed_peer_base[kMaxRanksTile];
    int const* recv_counters;          // per-consumer valid-band counts recv_counters[C]
    std::uint32_t const* a2a_flag_val; // NVLinkOneSided round flag selecting the recv_counters bank (nullable)
    void const* source_activation;
    std::uint64_t source_token_stride_bytes;
    int ep_size;      // E; the compact recv-enumeration outer axis
    int rank_id;      // producer index P
    int r_runtime;    // R; the P*R + t landing-slot stride factor
    int cta_cap;      // fixed physical grid width == CTA-granular grid stride
    int dest_perm_id; // band permutation: 0=linear_destination_band_v1,
                      // 1=producer_rank_rotated_remote_first_v1 (fixed per launch)
};

// Destination-band permutation perm_P[b]. Only two registered
// candidates; the choice is fixed per launch, never selected per
// replay:
//   linear_destination_band_v1:              perm_P[b] = b
//   producer_rank_rotated_remote_first_v1:   perm_P[b] = (P + 1 + b) mod E
// The rotated candidate's +1 makes band 0 the first REMOTE peer and self the last
// band, so the slowest cross-node PUSHes are issued first and drain under maximal
// overlap -> the K3 exposed wait collapses. For any fixed band b, P -> perm_P[b]
// is a bijection over [0, E), so synchronized producers never co-hammer one
// consumer (anti-incast). Coverage is permutation-invariant, so the output is identical.
__device__ __forceinline__ int k1PermBand(int b, int producerP, int epSize, int permId)
{
    return permId == 1 ? ((producerP + 1 + b) % epSize) : b;
}

// Stage one token's packed record into the __shared__ double-buffer row.
// UseHalf selects the interleaved Half layout (EP4/EP8/EP16/EP32, records_per_token in
// {4,8}); otherwise the contiguous FullToken layout (EP2, records_per_token==2).
// The two-wave load + per-tile quantize is byte-for-byte the family packed
// prepare; only the destination is __shared__ instead of the global plane.
template <bool UseHalf>
__device__ __forceinline__ void quantizeTokenIntoStage(
    __nv_bfloat16 const* tokenInput, std::uint8_t* stageBase, int tileGroup, int laneInTile)
{
    // 2-deep load ILP: issue both independent B256 source loads up front, then
    // quantize. At H7168 both waves are
    // always active (56 + 56 == 112 tiles), but keep the guard.
    int const secondTile = tileGroup + kK1PrepareTileGroups;
    bool const hasSecond = secondTile < kTilesPerToken;
    PackedVec16Bf16 firstVec;
    PackedVec16Bf16 secondVec;
    loadPackedVec16(tokenInput + tileGroup * kTileElements + laneInTile * kElementsPerLane, firstVec);
    if (hasSecond)
    {
        loadPackedVec16(tokenInput + secondTile * kTileElements + laneInTile * kElementsPerLane, secondVec);
    }

    if (UseHalf)
    {
        {
            int const half = tileGroup >= kQHalfTiles ? 1 : 0;
            int const localTile = tileGroup - half * kQHalfTiles;
            std::uint8_t* halfQ = stageBase + half * kPackedHalfBytes;
            std::uint8_t* halfSf = stageBase + kQHalfAlignedBytes + half * kPackedHalfBytes;
            quantizeAndStoreTile(firstVec, localTile, laneInTile, halfQ, halfSf);
        }
        if (hasSecond)
        {
            int const half = secondTile >= kQHalfTiles ? 1 : 0;
            int const localTile = secondTile - half * kQHalfTiles;
            std::uint8_t* halfQ = stageBase + half * kPackedHalfBytes;
            std::uint8_t* halfSf = stageBase + kQHalfAlignedBytes + half * kPackedHalfBytes;
            quantizeAndStoreTile(secondVec, localTile, laneInTile, halfQ, halfSf);
        }
    }
    else
    {
        std::uint8_t* tokenQ = stageBase;
        std::uint8_t* tokenSf = stageBase + kQAlignedBytes;
        quantizeAndStoreTile(firstVec, tileGroup, laneInTile, tokenQ, tokenSf);
        if (hasSecond)
        {
            quantizeAndStoreTile(secondVec, secondTile, laneInTile, tokenQ, tokenSf);
        }
    }
}

template <bool UseHalf>
__global__ __launch_bounds__(kK1Threads) void moeA2APushK1Kernel(K1PushDeviceArgs args)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // Honor a producer-of-activations programmatic dependency without triggering
    // downstream completion (K1 publishes no flag). Mirrors the family prepares.
    cudaGridDependencySynchronize();
#endif

    // alignas(128) covers the >=16-byte source alignment the bulk S2G requires and
    // keeps each staged record on a 128-byte boundary (record stride is 128-aligned).
    alignas(128) __shared__ std::uint8_t packedStage[kK1StageAlloc][kK1PackedRecordBytes];

    int const producerP = args.rank_id;
    int const epSize = args.ep_size;
    int const rRuntime = args.r_runtime;
    int const permId = args.dest_perm_id;
    long long const gridStride = static_cast<long long>(args.cta_cap);

    int const tileGroup = static_cast<int>(threadIdx.x) / kLanesPerTile;
    int const laneInTile = static_cast<int>(threadIdx.x) % kLanesPerTile;

    // --- true_compact_prefix_v2 ------------------------------------------------
    // Build this producer's E-entry band prefix over the selected permutation,
    // then iterate ONLY the compact record domain [0, Nrec). Unlike an
    // E*R+skip scan, no padding slot is ever visited (padding_slot_scan_count==0):
    //   prefix[0]=0; prefix[b+1]=prefix[b]+recv_counters[perm_P[b]]; Nrec=prefix[E].
    // E <= kMaxRanksTile, so the prefix is a few hundred bytes of __shared__ built
    // once by the issuer thread; every thread then reads it uniformly.
    __shared__ int s_prefix[kMaxRanksTile + 1];
    if (threadIdx.x == 0)
    {
        // The NVLinkOneSided dispatch writes this round's counts into the
        // recv_counters[parity] bank; parity is derived on device from the A2A
        // round flag (2N+1 after dispatch, 2N+2 after the D0 combine-open both
        // map to round N), so a captured graph re-selects the bank on replay.
        int const* recvCounters = args.recv_counters;
        if (args.a2a_flag_val != nullptr)
        {
            std::uint32_t const parity = ((*args.a2a_flag_val - 1U) >> 1U) & 1U;
            recvCounters += static_cast<int>(parity) * epSize;
        }
        int acc = 0;
        s_prefix[0] = 0;
        for (int b = 0; b < epSize; ++b)
        {
            acc += recvCounters[k1PermBand(b, producerP, epSize, permId)];
            s_prefix[b + 1] = acc;
        }
    }
    __syncthreads();
    long long const Nrec = static_cast<long long>(s_prefix[epSize]);

    int stage = 0;
    long long recordsPushed = 0;

    // Persistent CTA-granular grid stride over the compact record domain [0, Nrec).
    // Live CTAs are exactly min(Nrec, cta_cap): a CTA with blockIdx.x >= Nrec skips
    // the loop entirely (it read the prefix, then exits at the tail draining
    // nothing) and contributes no coverage. Grid width is the fixed cta_cap so
    // coverage is replay-stable. u strictly increases by cta_cap each step, so the
    // inverse-prefix band index only advances forward -> O(E + records) lookups per
    // CTA. u, band, consumerC and tokenT are all CTA-uniform (no divergence).
    int band = 0;
    for (long long u = static_cast<long long>(blockIdx.x); u < Nrec; u += gridStride)
    {
        // inverse_prefix: unique band with s_prefix[band] <= u < s_prefix[band+1].
        // Zero-count bands (s_prefix[band+1]==s_prefix[band]) are stepped over here
        // and never produce a payload issue.
        while (band + 1 < epSize && u >= static_cast<long long>(s_prefix[band + 1]))
        {
            ++band;
        }
        int const consumerC = k1PermBand(band, producerP, epSize, permId);
        int const tokenT = static_cast<int>(u - static_cast<long long>(s_prefix[band]));

        // --- reuse barrier: free the stage buffer we are about to overwrite ---
        // Only the issuer (threadIdx.x == 0) committed the outstanding groups, so
        // only it drains; the CTA barrier then republishes the freed buffer to
        // every writer before any thread rewrites it (WAR against the bulk read).
        if (recordsPushed >= kK1StageActive)
        {
            if (threadIdx.x == 0)
            {
                cp_async_bulk_wait_group<kK1StageActive - 1>();
            }
        }
        __syncthreads();

        // --- quantize token t into packedStage[stage] ---
        // Source is the [ep_size*R, 7168] BF16 band plane: the record from band C
        // at send index t lives at slot C*R + t (moeAlltoAllKernels.cu 975-983).
        std::uint8_t* stageBase = &packedStage[stage][0];
        std::size_t const sourceSlot = static_cast<std::size_t>(consumerC) * static_cast<std::size_t>(rRuntime)
            + static_cast<std::size_t>(tokenT);
        auto const* tokenBytes
            = static_cast<std::uint8_t const*>(args.source_activation) + sourceSlot * args.source_token_stride_bytes;
        auto const* tokenInput = reinterpret_cast<__nv_bfloat16 const*>(tokenBytes);
        quantizeTokenIntoStage<UseHalf>(tokenInput, stageBase, tileGroup, laneInTile);

        // --- publication: every writer fences, one issuer pushes exactly one record ---
        fence_proxy_async_shared_cta();
        __syncthreads();
        if (threadIdx.x == 0)
        {
            std::size_t const landingSlot = static_cast<std::size_t>(producerP) * static_cast<std::size_t>(rRuntime)
                + static_cast<std::size_t>(tokenT);
            std::uint8_t* dst
                = args.packed_peer_base[consumerC] + landingSlot * static_cast<std::size_t>(kK1PackedRecordBytes);
            cp_async_bulk_s2g_sys(dst, stageBase, kK1PackedRecordBytes);
            cp_async_bulk_commit_group();
        }

        ++recordsPushed;
        stage ^= 1;
    }

    // --- K1 tail: drain every outstanding push before the CTA frees its stage ---
    if (threadIdx.x == 0)
    {
        cp_async_bulk_wait_group<0>();
    }
    __syncthreads();
}

// -----------------------------------------------------------------------------
// Host validation (validate4LaneW256Packed-style HOST_REJECT). Rejects any
// non-baseline geometry, bad packed identity, or missing pointers.
// -----------------------------------------------------------------------------
void validateK1PushParams(K1PushParams const& params)
{
    // Geometry: the ONLY legal block width is the frozen 224 baseline.
    TLLM_CHECK_WITH_INFO(params.geometry.threads == kK1BaselineThreads,
        "K1 requires the deterministic_224_baseline_v1 geometry (blockDim.x == 224); got %d", params.geometry.threads);
    TLLM_CHECK_WITH_INFO(params.geometry.s_alloc == kK1StageAlloc && params.geometry.s_active == kK1StageActive,
        "K1 baseline pins s_alloc == s_active == 2 (the shared double buffer)");

    // Route / topology shape.
    TLLM_CHECK_WITH_INFO(params.ep_size > 0 && params.ep_size <= kMaxRanksTile, "K1 ep_size %d out of (0, %d]",
        params.ep_size, kMaxRanksTile);
    TLLM_CHECK_WITH_INFO(params.rank_id >= 0 && params.rank_id < params.ep_size, "K1 rank_id %d out of [0, ep_size=%d)",
        params.rank_id, params.ep_size);
    TLLM_CHECK_WITH_INFO(params.top_k == kK1TopK, "K1 requires TOP_K == 8 (DeepSeek-R1); got %d", params.top_k);
    TLLM_CHECK_WITH_INFO(params.r_runtime > 0, "K1 requires R (r_runtime) > 0");
    TLLM_CHECK_WITH_INFO(params.local_num_tokens >= 0, "K1 local_num_tokens must be nonnegative");
    // The compact recv enumeration reads recv_counters[C] per consumer band; it is
    // required (may be null only on a degenerate all-zero launch, guarded below).
    TLLM_CHECK_WITH_INFO(
        params.recv_counters != nullptr, "K1 recv_counters is required for the compact recv enumeration");

    // Packed identity: records_per_token pins the FullToken vs Half stage layout
    // and must equal min(top_k, ep_size), restricted to the family axis {2,4,8}.
    int const expectedRecordsPerToken = params.top_k < params.ep_size ? params.top_k : params.ep_size;
    TLLM_CHECK_WITH_INFO(params.records_per_token == expectedRecordsPerToken,
        "K1 records_per_token %d != min(top_k=%d, ep_size=%d)", params.records_per_token, params.top_k, params.ep_size);
    TLLM_CHECK_WITH_INFO(
        params.records_per_token == 2 || params.records_per_token == 4 || params.records_per_token == 8,
        "K1 records_per_token must be one of {2,4,8}; got %d", params.records_per_token);
    // packed_layout_version is the SOLE identity axis: it must carry the compact
    // (packed-interleaved [Q|SF]) layout, agreeing with the consuming K4.
    TLLM_CHECK_WITH_INFO(
        params.packed_layout_version == static_cast<int>(MoeA2ACombinePayloadLayout::kTileNvfp4Compact),
        "K1 packed_layout_version must be the compact (packed-interleaved) layout; got %d",
        params.packed_layout_version);

    // Source activation plane (BF16 H7168): required + 32-byte aligned for the
    // wide B256 loads; the token stride is the family BF16 token stride.
    if (params.local_num_tokens > 0)
    {
        TLLM_CHECK_WITH_INFO(
            params.source_activation != nullptr, "K1 source_activation is required for nonzero local tokens");
        TLLM_CHECK_WITH_INFO(reinterpret_cast<std::uintptr_t>(params.source_activation) % 32 == 0,
            "K1 BF16 source_activation must be 32-byte aligned");
        TLLM_CHECK_WITH_INFO(params.topk_target_ranks != nullptr && params.topk_send_indices != nullptr,
            "K1 TOP_K routing arrays are required for nonzero local tokens");
    }
    TLLM_CHECK_WITH_INFO(params.source_token_stride_bytes == static_cast<std::uint64_t>(kBf16TokenBytes),
        "K1 source_token_stride_bytes must be the BF16 token stride (14336)");

    // Destination packed landing planes: one per consumer C in [0, ep_size),
    // required and 128-byte aligned (each record lands on a 128-byte boundary).
    for (int consumerC = 0; consumerC < params.ep_size; ++consumerC)
    {
        TLLM_CHECK_WITH_INFO(
            params.packed_peer_base[consumerC] != nullptr, "K1 packed_peer_base[%d] is required", consumerC);
        TLLM_CHECK_WITH_INFO(reinterpret_cast<std::uintptr_t>(params.packed_peer_base[consumerC]) % 128 == 0,
            "K1 packed_peer_base[%d] must be 128-byte aligned", consumerC);
    }

    // The compact recv enumeration domain (ep_size * R) must stay within int range.
    std::uint64_t const workItems
        = static_cast<std::uint64_t>(params.ep_size) * static_cast<std::uint64_t>(params.r_runtime);
    TLLM_CHECK_WITH_INFO(workItems <= static_cast<std::uint64_t>(std::numeric_limits<int>::max()),
        "K1 (ep_size * r_runtime) work domain exceeds int index range");

    // The landing slot P*R + t must stay addressable as a record index. The
    // compact enum bounds t by recv_counters[C] <= R, so the max slot is
    // P*R + (R-1) == (P+1)*R - 1.
    std::uint64_t const maxSlot
        = static_cast<std::uint64_t>(params.rank_id) * static_cast<std::uint64_t>(params.r_runtime)
        + static_cast<std::uint64_t>(params.r_runtime > 0 ? params.r_runtime - 1 : 0);
    TLLM_CHECK_WITH_INFO(
        maxSlot <= (std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(kK1PackedRecordBytes)),
        "K1 landing-slot byte offset overflows uint64");
}

// Resolve the fixed physical grid width (per-wave CTA ceiling). geometry values
// > 0 are honored verbatim; 0 means "resolve from the live device" (occupancy of
// the launched kernel times the device SM count). The result is a constant baked
// into the launch, so it is stable across CUDA-graph replays.
int resolveCtaCap(K1PushParams const& params, void const* kernelFn)
{
    if (params.geometry.cta_cap > 0)
    {
        return params.geometry.cta_cap;
    }

    int device = -1;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));
    int smCount = params.geometry.target_sm_count;
    if (smCount <= 0)
    {
        TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device));
    }
    int blocksPerSm = 0;
    TLLM_CUDA_CHECK(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSm, kernelFn, kK1Threads, /*dynamicSMemSize*/ 0));
    int ctaCap = blocksPerSm * smCount;
    if (ctaCap < 1)
    {
        ctaCap = 1;
    }
    return ctaCap;
}

// Resolve the destination-band permutation id. K1_DEST_PERM overrides it; it is
// read once at launch/capture and baked into the replay-stable args (NOT a
// per-replay runtime autotune -- graph capture bakes the resolved value). Default
// is the rotated remote-first candidate (the anti-incast schedule that minimizes
// the K3 exposed wait). K1_DEST_PERM=0|linear selects the linear order.
int resolveDestPermId()
{
    char const* env = std::getenv("K1_DEST_PERM");
    if (env != nullptr && env[0] != '\0')
    {
        if (env[0] == '0' || env[0] == 'l' || env[0] == 'L')
        {
            return 0; // linear_destination_band_v1
        }
        if (env[0] == '1' || env[0] == 'r' || env[0] == 'R')
        {
            return 1; // producer_rank_rotated_remote_first_v1
        }
    }
    return 1; // default: producer_rank_rotated_remote_first_v1
}

K1PushDeviceArgs makeDeviceArgs(K1PushParams const& params, int ctaCap)
{
    K1PushDeviceArgs args{};
    for (int consumerC = 0; consumerC < params.ep_size; ++consumerC)
    {
        args.packed_peer_base[consumerC] = params.packed_peer_base[consumerC];
    }
    args.recv_counters = params.recv_counters;
    args.a2a_flag_val = params.a2a_flag_val;
    args.source_activation = params.source_activation;
    args.source_token_stride_bytes = params.source_token_stride_bytes;
    args.ep_size = params.ep_size;
    args.rank_id = params.rank_id;
    args.r_runtime = params.r_runtime;
    args.cta_cap = ctaCap;
    args.dest_perm_id = resolveDestPermId();
    return args;
}

} // namespace

void launchK1Push(K1PushParams const& params, cudaStream_t stream)
{
    validateK1PushParams(params);

    // records_per_token == 2 -> FullToken layout; {4,8} -> interleaved Half layout.
    bool const useHalf = params.records_per_token != 2;

    // The physical grid width is fixed at launch (replay-stable). It is resolved
    // against the exact kernel instantiation that will run so occupancy matches.
    auto const* kernelFn = useHalf ? reinterpret_cast<void const*>(&moeA2APushK1Kernel<true>)
                                   : reinterpret_cast<void const*>(&moeA2APushK1Kernel<false>);
    int const ctaCap = resolveCtaCap(params, kernelFn);
    K1PushDeviceArgs const args = makeDeviceArgs(params, ctaCap);

    // The common TensorRT-LLM helper opts into programmatic stream serialization according
    // to getEnvEnablePDL(). Grid is the fixed cta_cap;
    // block is the mandatory 224 baseline; no dynamic shared memory (the double
    // buffer is a static __shared__ array).
    if (useHalf)
    {
        tensorrt_llm::common::launchWithPdlWhenEnabled(
            "moeA2APushK1Kernel<Half>", moeA2APushK1Kernel<true>, dim3(ctaCap), dim3(kK1Threads), 0, stream, args);
    }
    else
    {
        tensorrt_llm::common::launchWithPdlWhenEnabled("moeA2APushK1Kernel<FullToken>", moeA2APushK1Kernel<false>,
            dim3(ctaCap), dim3(kK1Threads), 0, stream, args);
    }
}

} // namespace kernels::moe_comm

TRTLLM_NAMESPACE_END
