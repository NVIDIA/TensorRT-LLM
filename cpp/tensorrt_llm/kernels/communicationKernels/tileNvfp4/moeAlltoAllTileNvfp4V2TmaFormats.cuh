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

// Compile-time payload-format traits for the tile-wise TMA MoE combine.
//
// The TMA transport (bulk copy + mbarrier rendezvous + route cache), the K8
// route reduction (reduceRoute8Frozen, fp32), and the bf16 store (packBf16Pair)
// are all payload-format agnostic. Only three things change per precision:
//   1. the wire byte geometry (bytes per tile / per token, scale plane size),
//   2. the shared-memory Q-cache granule type (one aligned element per tile),
//   3. the per-block16 decode: raw payload bytes (+ scale) -> 16 dequantized
//      fp32 activations, packed as 8 float2 pairs for the shared reduce+store.
//
// Each Format below supplies (1)/(2) as constants/typedefs and (3) as a single
// __forceinline__ decodeBlock16() so the routing-policy fast paths and the
// generic fallback share one decode implementation per precision.
//
// Recipes (all reduce in fp32, store bf16; hidden=7168, tile=64, block16=16):
//   Nvfp4Format : 4-bit E2M1 + per-16 E4M3 block scale + per-tile fp32 globalD.
//                 32 B/tile Q + 8 B/tile SF  (3584 + 896 B/token).  [existing]
//   Fp8E4m3Format: 8-bit E4M3 scale-free (direct bf16<->e4m3 saturating cast,
//                 matching the production low-precision combine convention).
//                 64 B/tile Q + no SF plane  (7168 B/token).
//   Bf16Format  : 16-bit bf16 passthrough, no quantization, no scale plane.
//                 128 B/tile Q + 0 B/tile SF (14336 + 0 B/token).

#include "moeAlltoAllTileNvfp4V2.h"
#include "moeAlltoAllTileNvfp4V2TmaCommon.cuh"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::moe_comm::tile_nvfp4_tma
{

// 64-byte / 128-byte aligned smem Q-cache granules (one per tile) for the FP8
// and BF16 payloads, mirroring B256 (32 B) used by the NVFP4 payload.
struct alignas(64) B512
{
    uint4 a;
    uint4 b;
    uint4 c;
    uint4 d;
};

static_assert(sizeof(B512) == 64);
static_assert(alignof(B512) == 64);

struct alignas(128) B1024
{
    uint4 a;
    uint4 b;
    uint4 c;
    uint4 d;
    uint4 e;
    uint4 f;
    uint4 g;
    uint4 h;
};

static_assert(sizeof(B1024) == 128);
static_assert(alignof(B1024) == 128);

// One block16's decoded activations: 16 fp32 values as 8 (x,y) pairs, matching
// the existing decode inner loop granularity (kPairsPerBlock == 8).
struct DecodedBlock16
{
    float2 pair[kPairsPerBlock];
};

// The Route4Half fast path stores only 4 of a block16's 8 pairs per lane
// ([half*4, half*4+4)); decodeHalfBlock16 decodes exactly those 4 so the other
// four pair-decodes are never issued. Byte-exact: the kept pairs run the
// identical per-pair arithmetic as the corresponding decodeBlock16 pairs.
struct DecodedBlock16Half
{
    float2 pair[kPairsPerBlock / 2];
};

static_assert(kPairsPerBlock == 8);
static_assert(kScaleBlockElements == 16);

// ---------------------------------------------------------------------------
// NVFP4 (existing production recipe). decodeBlock16 reproduces, byte-for-byte,
// the math in decodeReduceAndStoreTile()/the policy fast paths:
//   scaledBlock = float(e4m3 blockScaleByte) * fp32 globalD
//   for each fp4x2 byte: (qx, qy) = float2(e2m1x2); out = (qx*scaledBlock,
//   qy*scaledBlock).
// ---------------------------------------------------------------------------
struct Nvfp4Format
{
    static constexpr MoeA2ACombineQuantFormat kQuantFormat = MoeA2ACombineQuantFormat::kTileNvfp4;
    static constexpr bool kHasScalePlane = true;
    // Only NVFP4 carries the adversarial poison-stamp corpus (it stamps the
    // scale plane + 32-B Q granule). FP8/BF16 opt out so the granule-typed
    // stampPoisonAllStageRows is never instantiated for them.
    static constexpr bool kSupportsPoison = true;

    static constexpr int kQBytesPerTile = 32;                                      // 64 e2m1 (4-bit) = 32 B
    static constexpr int kSfBytesPerTile = 8;                                      // 4 e4m3 block scales + fp32 globalD
    static constexpr int kQBytesPerBlock = kQBytesPerTile / kBlocksPerTile;        // 8
    static constexpr int kQPayloadBytesPerToken = kTilesPerToken * kQBytesPerTile; // 3584
    static constexpr int kSfPayloadBytesPerToken = kTilesPerToken * kSfBytesPerTile; // 896
    static constexpr int kQCacheStride = kTilesPerToken + 1;                         // 113
    static constexpr int kScaleCacheStride = kTilesPerToken + 2;                     // 114

    // Packed-interleaved layout geometry (payload_layout == kTileNvfp4Compact):
    // the Q body and the SF record are stitched into ONE contiguous per-record
    // buffer so a single cp.async.bulk transfers both. Only NVFP4 carries a SF
    // plane, so only this Format defines the packed constants; the FP8/BF16
    // Formats are scale-free and excluded from the packed path.
    //   FullToken (EP2): [ Q(3584) | SF(896) ] = 4480 B, q_aligned = 3584.
    //   Half (EP4/EP8):  [ Q_h0(1792) | SF_h0(448) | Q_h1(1792) | SF_h1(448) ].
    static constexpr int kQAlignedBytes = kQPayloadBytesPerToken;                               // 3584
    static constexpr int kQHalfAlignedBytes = (kTilesPerToken / 2) * kQBytesPerTile;            // 1792
    static constexpr int kSfHalfBytes = (kTilesPerToken / 2) * kSfBytesPerTile;                 // 448
    static constexpr int kPackedRecordBytes = kQPayloadBytesPerToken + kSfPayloadBytesPerToken; // 4480
    static constexpr int kPackedRecordStrideBytes = kPackedRecordBytes;                         // 4480

    static_assert(kQAlignedBytes == 3584);
    static_assert(kQHalfAlignedBytes == 1792);
    static_assert(kSfHalfBytes == 448);
    static_assert(kPackedRecordBytes == 4480);
    static_assert(kPackedRecordStrideBytes % 128 == 0); // every record 128-aligned
    static_assert(kQHalfAlignedBytes + kSfHalfBytes == kPackedRecordStrideBytes / 2);

    using QGranule = B256;              // 32 B smem q-cache element per tile
    using ScaleGranule = std::uint64_t; // 8 B smem scale-cache element per tile

    // scalePack: the 8-byte SF record for this tile (4 e4m3 block scales in the
    // low 32 bits, fp32 globalD in the high 32 bits). qBlock: this block16's
    // kQBytesPerBlock (8) Q bytes, already staged.
    __device__ __forceinline__ static DecodedBlock16 decodeBlock16(
        std::uint8_t const* qBlock, std::uint64_t scalePack, int blockInTile)
    {
        std::uint8_t const rawBlockScale = static_cast<std::uint8_t>(scalePack >> (blockInTile * 8));
        float const globalD = __uint_as_float(static_cast<std::uint32_t>(scalePack >> 32));
        __nv_fp8_e4m3 blockScale;
        blockScale.__x = rawBlockScale;
        float const scaledBlock = __fmul_rn(static_cast<float>(blockScale), globalD);

        std::uint64_t const packedQ = *reinterpret_cast<std::uint64_t const*>(qBlock);
        DecodedBlock16 out;
#pragma unroll
        for (int pair = 0; pair < kPairsPerBlock; ++pair)
        {
            __nv_fp4x2_e2m1 qPair;
            qPair.__x = static_cast<__nv_fp4x2_storage_t>((packedQ >> (pair * 8)) & 0xFFU);
            float2 const q = static_cast<float2>(qPair);
            out.pair[pair].x = __fmul_rn(q.x, scaledBlock);
            out.pair[pair].y = __fmul_rn(q.y, scaledBlock);
        }
        return out;
    }

    // Decode only the 4 pairs [startPair, startPair+4) of this block16. Every
    // kept pair executes the identical scaledBlock prologue + fp4x2 unpack +
    // __fmul_rn as decodeBlock16().pair[startPair+p], so the output bits are
    // byte-for-byte identical; the discarded four pairs are simply never
    // computed (used by the Route4Half half-lane store, which keeps 4 pairs).
    __device__ __forceinline__ static DecodedBlock16Half decodeHalfBlock16(
        std::uint8_t const* qBlock, std::uint64_t scalePack, int blockInTile, int startPair)
    {
        std::uint8_t const rawBlockScale = static_cast<std::uint8_t>(scalePack >> (blockInTile * 8));
        float const globalD = __uint_as_float(static_cast<std::uint32_t>(scalePack >> 32));
        __nv_fp8_e4m3 blockScale;
        blockScale.__x = rawBlockScale;
        float const scaledBlock = __fmul_rn(static_cast<float>(blockScale), globalD);

        std::uint64_t const packedQ = *reinterpret_cast<std::uint64_t const*>(qBlock);
        DecodedBlock16Half out;
#pragma unroll
        for (int p = 0; p < kPairsPerBlock / 2; ++p)
        {
            int const pair = startPair + p;
            __nv_fp4x2_e2m1 qPair;
            qPair.__x = static_cast<__nv_fp4x2_storage_t>((packedQ >> (pair * 8)) & 0xFFU);
            float2 const q = static_cast<float2>(qPair);
            out.pair[p].x = __fmul_rn(q.x, scaledBlock);
            out.pair[p].y = __fmul_rn(q.y, scaledBlock);
        }
        return out;
    }
};

// ---------------------------------------------------------------------------
// FP8 E4M3 recipe -- SCALE-FREE, matching the production low-precision combine
// (moeAlltoAllKernels.cu vectorized_quant/vectorized_combine): prepare does a
// direct bf16->e4m3 saturating cast (cvt.rn.satfinite.e4m3x2.bf16x2, clips at
// +/-448, no scale), combine casts e4m3->fp32 straight back. No SF plane, the
// same precision as the reference FP8 combine.
//   value = float(e4m3 byte)
// ---------------------------------------------------------------------------
struct Fp8E4m3Format
{
    static constexpr MoeA2ACombineQuantFormat kQuantFormat = MoeA2ACombineQuantFormat::kFp8E4M3;
    static constexpr bool kHasScalePlane = false;
    static constexpr bool kSupportsPoison = false;

    static constexpr int kQBytesPerTile = 64;                                      // 64 e4m3 (8-bit) = 64 B
    static constexpr int kSfBytesPerTile = 0;                                      // scale-free: no SF plane
    static constexpr int kQBytesPerBlock = kQBytesPerTile / kBlocksPerTile;        // 16
    static constexpr int kQPayloadBytesPerToken = kTilesPerToken * kQBytesPerTile; // 7168
    static constexpr int kSfPayloadBytesPerToken = 0;
    static constexpr int kQCacheStride = kTilesPerToken + 1;
    static constexpr int kScaleCacheStride = 0;

    using QGranule = B512;     // 64 B smem q-cache element per tile
    using ScaleGranule = void; // no scale plane

    // scalePack/blockInTile unused (scale-free). qBlock: this block16's 16 e4m3
    // bytes.
    __device__ __forceinline__ static DecodedBlock16 decodeBlock16(
        std::uint8_t const* qBlock, std::uint64_t scalePack, int blockInTile)
    {
        (void) scalePack;
        (void) blockInTile;
        DecodedBlock16 out;
#pragma unroll
        for (int pair = 0; pair < kPairsPerBlock; ++pair)
        {
            __nv_fp8_e4m3 e0;
            __nv_fp8_e4m3 e1;
            e0.__x = qBlock[pair * 2];
            e1.__x = qBlock[pair * 2 + 1];
            out.pair[pair].x = static_cast<float>(e0);
            out.pair[pair].y = static_cast<float>(e1);
        }
        return out;
    }

    // 4-pair [startPair, startPair+4) subset (Route4Half half-lane store);
    // byte-exact vs decodeBlock16 (same per-pair e4m3->float cast).
    __device__ __forceinline__ static DecodedBlock16Half decodeHalfBlock16(
        std::uint8_t const* qBlock, std::uint64_t scalePack, int blockInTile, int startPair)
    {
        (void) scalePack;
        (void) blockInTile;
        DecodedBlock16Half out;
#pragma unroll
        for (int p = 0; p < kPairsPerBlock / 2; ++p)
        {
            int const pair = startPair + p;
            __nv_fp8_e4m3 e0;
            __nv_fp8_e4m3 e1;
            e0.__x = qBlock[pair * 2];
            e1.__x = qBlock[pair * 2 + 1];
            out.pair[p].x = static_cast<float>(e0);
            out.pair[p].y = static_cast<float>(e1);
        }
        return out;
    }
};

// ---------------------------------------------------------------------------
// BF16 passthrough recipe. No quantization, no scale plane. Q is the raw bf16
// token; decode is a straight bf16->fp32 widen (the K8 reduce still runs in
// fp32, then the result is stored bf16, exactly as for the quantized recipes).
// This is the accuracy ceiling / bandwidth upper bound of the comparison.
// ---------------------------------------------------------------------------
struct Bf16Format
{
    static constexpr MoeA2ACombineQuantFormat kQuantFormat = MoeA2ACombineQuantFormat::kNone;
    static constexpr bool kHasScalePlane = false;
    static constexpr bool kSupportsPoison = false;

    static constexpr int kQBytesPerTile = 128;                                     // 64 bf16 (16-bit) = 128 B
    static constexpr int kSfBytesPerTile = 0;
    static constexpr int kQBytesPerBlock = kQBytesPerTile / kBlocksPerTile;        // 32
    static constexpr int kQPayloadBytesPerToken = kTilesPerToken * kQBytesPerTile; // 14336
    static constexpr int kSfPayloadBytesPerToken = 0;
    static constexpr int kQCacheStride = kTilesPerToken + 1;
    static constexpr int kScaleCacheStride = 0;

    using QGranule = B1024;    // 128 B smem q-cache element per tile
    using ScaleGranule = void; // no scale plane

    // scalePack/blockInTile unused. qBlock: this block16's 32 bytes = 16 bf16.
    __device__ __forceinline__ static DecodedBlock16 decodeBlock16(
        std::uint8_t const* qBlock, std::uint64_t scalePack, int blockInTile)
    {
        (void) scalePack;
        (void) blockInTile;
        __nv_bfloat16 const* elems = reinterpret_cast<__nv_bfloat16 const*>(qBlock);
        DecodedBlock16 out;
#pragma unroll
        for (int pair = 0; pair < kPairsPerBlock; ++pair)
        {
            out.pair[pair].x = __bfloat162float(elems[pair * 2]);
            out.pair[pair].y = __bfloat162float(elems[pair * 2 + 1]);
        }
        return out;
    }

    // 4-pair [startPair, startPair+4) subset (Route4Half half-lane store);
    // byte-exact vs decodeBlock16 (same per-pair bf16->float widen).
    __device__ __forceinline__ static DecodedBlock16Half decodeHalfBlock16(
        std::uint8_t const* qBlock, std::uint64_t scalePack, int blockInTile, int startPair)
    {
        (void) scalePack;
        (void) blockInTile;
        __nv_bfloat16 const* elems = reinterpret_cast<__nv_bfloat16 const*>(qBlock);
        DecodedBlock16Half out;
#pragma unroll
        for (int p = 0; p < kPairsPerBlock / 2; ++p)
        {
            int const pair = startPair + p;
            out.pair[p].x = __bfloat162float(elems[pair * 2]);
            out.pair[p].y = __bfloat162float(elems[pair * 2 + 1]);
        }
        return out;
    }
};

} // namespace kernels::moe_comm::tile_nvfp4_tma

TRTLLM_NAMESPACE_END
