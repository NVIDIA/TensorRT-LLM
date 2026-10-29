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

#include "tensorrt_llm/kernels/glmMoeDsaRubin/glmSparseMlaFlashDecode.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa
{
namespace
{

// The MMAs run 8 head columns: the kHeads query heads in columns 0 .. kHeads - 1, zero Q rows in the others.
constexpr int kMmaHeads = 8;
constexpr int kHeadDimQk = kGlmSparseMlaHeadDimQk;
constexpr int kHeadDimV = kGlmSparseMlaHeadDimV;
// One cluster of kSplits CTAs per query token; CTA s owns indices [s * kTokensPerCta, (s + 1) * kTokensPerCta).
constexpr int kSplits = 16;
constexpr int kTokensPerCta = kGlmSparseMlaMaxTopK / kSplits;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kTokensPerWarp = kTokensPerCta / kWarps;
static_assert(kTokensPerWarp == 16, "a warp gathers one 16-token MMA tile, one token row per lane");
// A token row: 576 E4M3 values. In shared memory the rows are 592 bytes apart, so the eight 16-byte rows of every
// ldmatrix phase fall into distinct banks.
constexpr int kRowBytes = kHeadDimQk;
constexpr int kSmemRowBytes = kRowBytes + 16;
// P (E4M3) per MMA head column, padded the same way.
constexpr int kPRowBytes = kTokensPerCta + 16;
// Output columns merged by each CTA of the cluster.
constexpr int kDvPerCta = kHeadDimV / kSplits;
static_assert(kDvPerCta % 16 == 0, "each output slice holds whole 16-column MMA blocks");
// P is scaled to the E4M3 range before the second MMA; the normalization divides it out.
constexpr float kPScale = 448.f;

// v_b BMM: CTA r of the cluster owns v in [r * kVPerCta, (r + 1) * kVPerCta) of every head.
constexpr int kVHeadDim = kGlmSparseMlaVHeadDim;
constexpr int kVPerCta = kVHeadDim / kSplits;
static_assert(kVPerCta == 16, "a warp computes one head's 16 rows of the v_b slice as one m16n8k16 row tile");
// The v_b_proj rows [h][r * 16, r * 16 + 16)[0, 512) of one head: 16 KiB, contiguous in global memory.
constexpr int kVbHeadBytes = kVPerCta * kHeadDimV * 2;

// Shared memory: token rows, Q rows, P, the per-warp max / sum, the receive buffers of the cluster merge, the v_b
// slice, the CTA's merged output slice, the row's gathered BF16 output and the mbarriers.
constexpr int kOffQ = kTokensPerCta * kSmemRowBytes;
constexpr int kOffP = kOffQ + kMmaHeads * kSmemRowBytes;
constexpr int kOffRed = kOffP + kMmaHeads * kPRowBytes;
constexpr int kRedFloats = 2 * kWarps * kMmaHeads;
constexpr int kOffRecvO = kOffRed + kRedFloats * 4;
constexpr int kRecvOFloats = kSplits * kMmaHeads * kDvPerCta;
constexpr int kOffRecvStats = kOffRecvO + kRecvOFloats * 4;
constexpr int kRecvStatsFloats = kSplits * kMmaHeads * 2;
constexpr int kRecvBytes = (kRecvOFloats + kRecvStatsFloats) * 4;
constexpr int kOffVb = kOffRecvStats + kRecvStatsFloats * 4;
// Four heads use dedicated merge buffers. Eight heads reuse P and KV storage only after every
// CTA has finished P.V, as ordered by the post-P.V statistics transfer.
constexpr int kSliceBytes = kMmaHeads * kDvPerCta * 2;
constexpr int kGatheredOBytes = kMmaHeads * kHeadDimV * 2;
constexpr int kNumBars = kWarps + 4;

template <int kHeads>
struct FoldedLayout
{
    static_assert(kHeads == kGlmSparseMlaNumHeads || kHeads == kGlmSparseMlaNumHeadsTp8);
    static constexpr bool kOwnMergeBuffers = kHeads < kMmaHeads;
    static constexpr int kVbBytes = kHeads * kVbHeadBytes;
    static constexpr int kOffSlice = kOwnMergeBuffers ? kOffVb + kVbBytes : kOffP;
    static constexpr int kOffGatheredO = kOwnMergeBuffers ? kOffSlice + kSliceBytes : 0;
    static constexpr int kOffBars = kOwnMergeBuffers ? kOffGatheredO + kGatheredOBytes : kOffVb + kVbBytes;
    static constexpr int kBytes = kOffBars + kNumBars * 8;
    static_assert(kOffSlice % 16 == 0 && kOffGatheredO % 16 == 0 && kOffBars % 8 == 0, "alignment");
    static_assert(kBytes <= 227 * 1024, "shared memory per CTA");
};

static_assert(kOffQ % 16 == 0 && kOffP % 16 == 0 && kOffRecvO % 16 == 0 && kOffVb % 16 == 0, "alignment");
static_assert(kSplits * kMmaHeads <= kThreads, "one max / sum transfer per thread");
static_assert(kMmaHeads * kDvPerCta == kThreads, "the merge maps one output element to each thread");
static_assert(kDvPerCta == 32 && kSplits <= 32, "the merge maps warp h to head h of the slice, lane r to partial r");
static_assert(kGatheredOBytes <= kOffQ && kSliceBytes <= kMmaHeads * kPRowBytes, "merge buffers fit reused storage");
static_assert(FoldedLayout<8>::kBytes == 230752, "eight-head shared-memory layout");

// The 16-head kernel (glmSparseMlaFlashDecodeHeads16Kernel): one cluster of kSplits CTAs per query token, CTA s owns
// indices [s * kTokensPerCta, (s + 1) * kTokensPerCta); each of its warps gathers one 16-token tile, as above.
// 16 splits (128 keys per CTA) while every cluster fits one wave; 8 splits (256 keys per CTA) for more query tokens.
template <int kSplits_>
struct Heads16
{
    static constexpr int kHeads = kGlmSparseMlaNumHeadsWide;
    // 8-head MMA tiles (the n dimension of every QK and P.V MMA).
    static constexpr int kHeadTiles = kHeads / kMmaHeads;
    static constexpr int kSplits = kSplits_;
    static constexpr int kTokensPerCta = kGlmSparseMlaMaxTopK / kSplits;
    static constexpr int kWarps = kTokensPerCta / kTokensPerWarp;
    static constexpr int kThreads = kWarps * 32;
    static constexpr int kPRowBytes = kTokensPerCta + 16;
    // Output columns merged by each CTA of the cluster, and the 16-column P.V blocks of each warp.
    static constexpr int kDvPerCta = kHeadDimV / kSplits;
    static constexpr int kBlocksPerWarp = kHeadDimV / 16 / kWarps;
    // The merge: thread t merges columns 2 (t % kColPairs) and 2 (t % kColPairs) + 1 of head t / kColPairs.
    static constexpr int kColPairs = kDvPerCta / 2;

    // Shared memory: token rows, Q rows, P, the per-warp max / sum, the receive buffers of the cluster merge,
    // mbarriers.
    static constexpr int kOffQ = kTokensPerCta * kSmemRowBytes;
    static constexpr int kOffP = kOffQ + kHeads * kSmemRowBytes;
    static constexpr int kOffRed = kOffP + kHeads * kPRowBytes;
    static constexpr int kRedFloats = 2 * kWarps * kHeads;
    static constexpr int kOffRecvO = kOffRed + kRedFloats * 4;
    static constexpr int kRecvOFloats = kSplits * kHeads * kDvPerCta;
    static constexpr int kOffRecvStats = kOffRecvO + kRecvOFloats * 4;
    static constexpr int kRecvStatsFloats = kSplits * kHeads * 2;
    static constexpr int kRecvBytes = (kRecvOFloats + kRecvStatsFloats) * 4;
    static constexpr int kOffBars = kOffRecvStats + kRecvStatsFloats * 4;
    // One gather barrier per warp, the Q barrier and the receive barrier.
    static constexpr int kNumBars = kWarps + 2;
    static constexpr int kBytes = kOffBars + kNumBars * 8;

    static_assert(kTokensPerCta % kTokensPerWarp == 0 && kHeadDimV % (16 * kWarps) == 0, "whole MMA tiles per warp");
    static_assert(kDvPerCta % 16 == 0, "each output slice holds whole 16-column MMA blocks");
    static_assert(kOffQ % 16 == 0 && kOffP % 16 == 0 && kOffRecvO % 16 == 0 && kOffBars % 8 == 0, "alignment");
    static_assert(kSplits * kHeads <= kThreads, "one max / sum transfer per thread");
    static_assert(kHeads * kDvPerCta == 2 * kThreads, "the merge maps two output columns to each thread");
    static_assert(kBytes <= 227 * 1024, "shared memory per CTA");
};

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

__device__ __forceinline__ uint32_t smemAddress(void const* ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void mbarrierInit(uint32_t bar, uint32_t count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count) : "memory");
}

__device__ __forceinline__ void mbarrierArriveExpectTx(uint32_t bar, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbarrierWait(uint32_t bar, uint32_t parity)
{
    asm volatile(
        "{\n"
        ".reg .pred complete;\n"
        "GLM_SPARSE_MLA_FD_WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 complete, [%0], %1;\n"
        "@!complete bra GLM_SPARSE_MLA_FD_WAIT_%=;\n"
        "}\n" ::"r"(bar),
        "r"(parity)
        : "memory");
}

// Wait with cluster-scope acquire: the receive barriers complete on bytes that other CTAs of the cluster wrote.
__device__ __forceinline__ void mbarrierWaitCluster(uint32_t bar, uint32_t parity)
{
    asm volatile(
        "{\n"
        ".reg .pred complete;\n"
        "GLM_SPARSE_MLA_FD_WAIT_CLUSTER_%=:\n"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 complete, [%0], %1;\n"
        "@!complete bra GLM_SPARSE_MLA_FD_WAIT_CLUSTER_%=;\n"
        "}\n" ::"r"(bar),
        "r"(parity)
        : "memory");
}

__device__ __forceinline__ void bulkCopyGlobalToShared(uint32_t dst, void const* src, uint32_t bytes, uint32_t bar)
{
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(dst),
                 "l"(src), "r"(bytes), "r"(bar)
                 : "memory");
}

// Two floats into the shared memory of a CTA of the cluster; the bytes complete on that CTA's barrier.
__device__ __forceinline__ void storeRemote(uint32_t remoteAddr, float a, float b, uint32_t remoteBar)
{
    asm volatile(
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1, %2}, [%3];\n" ::"r"(remoteAddr),
        "f"(a), "f"(b), "r"(remoteBar)
        : "memory");
}

// Sixteen bytes into the shared memory of a CTA of the cluster; the bytes complete on that CTA's barrier.
__device__ __forceinline__ void storeRemote16(uint32_t remoteAddr, uint4 value, uint32_t remoteBar)
{
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];\n" ::"r"(
                     remoteAddr),
                 "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w), "r"(remoteBar)
                 : "memory");
}

__device__ __forceinline__ uint32_t mapToRank(uint32_t addr, uint32_t rank)
{
    uint32_t mapped;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(mapped) : "r"(addr), "r"(rank));
    return mapped;
}

__device__ __forceinline__ void clusterArriveRelaxed()
{
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void clusterWait()
{
    asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void ldmatrixX4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrixX4Trans(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrixX2(uint32_t addr, uint32_t& r0, uint32_t& r1)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(addr));
}

// D += A B with A 16 x 16 (row major) and B 16 x 8 (column major), BF16 inputs and fp32 accumulation.
__device__ __forceinline__ void mmaBf16(
    float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// D += A B with A 16 x 16 (row major) and B 16 x 8 (column major), FP16 inputs and fp32 accumulation.
__device__ __forceinline__ void mmaF16(
    float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// The two E4M3 values in the low 16 bits of `pair` as two FP16 values (exact).
__device__ __forceinline__ uint32_t e4m3x2ToHalf2(uint32_t pair)
{
    __half2_raw const h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(pair & 0xffffU), __NV_E4M3);
    return static_cast<uint32_t>(h.x) | (static_cast<uint32_t>(h.y) << 16);
}

// The B fragment of an E4M3 m16n8k32 MMA ({b0, b1}) unpacked for the two FP16 MMAs of mmaE4m3AsF16: low halves, then
// high halves.
__device__ __forceinline__ void unpackE4m3B(uint32_t (&bh)[4], uint32_t b0, uint32_t b1)
{
    bh[0] = e4m3x2ToHalf2(b0);
    bh[1] = e4m3x2ToHalf2(b1);
    bh[2] = e4m3x2ToHalf2(b0 >> 16);
    bh[3] = e4m3x2ToHalf2(b1 >> 16);
}

// D += A B for an E4M3 m16n8k32 MMA with A in its E4M3 fragment registers and B unpacked beforehand (unpackE4m3B):
// an FP16 m16n8k16 MMA on the low 16-bit halves of the A registers, then one on the high halves, chained on D (the
// SM100 family has no E4M3 mma.sync; this is the sequence the compiler lowers one to).
__device__ __forceinline__ void mmaE4m3AsF16(
    float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t const (&bh)[4])
{
    mmaF16(d, e4m3x2ToHalf2(a0), e4m3x2ToHalf2(a1), e4m3x2ToHalf2(a2), e4m3x2ToHalf2(a3), bh[0], bh[1]);
    mmaF16(d, e4m3x2ToHalf2(a0 >> 16), e4m3x2ToHalf2(a1 >> 16), e4m3x2ToHalf2(a2 >> 16), e4m3x2ToHalf2(a3 >> 16), bh[2],
        bh[3]);
}

// The A fragment of an E4M3 m16n8k32 MMA unpacked for two FP16 MMAs: ah[0..3] the low 16-bit halves of a0..a3,
// ah[4..7] the high halves.
__device__ __forceinline__ void unpackE4m3A(uint32_t (&ah)[8], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3)
{
    ah[0] = e4m3x2ToHalf2(a0);
    ah[1] = e4m3x2ToHalf2(a1);
    ah[2] = e4m3x2ToHalf2(a2);
    ah[3] = e4m3x2ToHalf2(a3);
    ah[4] = e4m3x2ToHalf2(a0 >> 16);
    ah[5] = e4m3x2ToHalf2(a1 >> 16);
    ah[6] = e4m3x2ToHalf2(a2 >> 16);
    ah[7] = e4m3x2ToHalf2(a3 >> 16);
}

// mmaE4m3AsF16 with A unpacked beforehand too (unpackE4m3A), so that one A fragment feeds several B fragments.
__device__ __forceinline__ void mmaE4m3AsF16Unpacked(float (&d)[4], uint32_t const (&ah)[8], uint32_t const (&bh)[4])
{
    mmaF16(d, ah[0], ah[1], ah[2], ah[3], bh[0], bh[1]);
    mmaF16(d, ah[4], ah[5], ah[6], ah[7], bh[2], bh[3]);
}

__device__ __forceinline__ uint8_t toE4m3(float x)
{
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}

// Position of the CTA's token t in the k order of the P.V MMA: within every 16 tokens, k slots 4c .. 4c + 3 hold the
// tokens {2c, 2c + 1, 2c + 8, 2c + 9}, the order in which ldmatrix.trans of the token-major V rows plus two byte
// permutes deliver a V^T fragment.
__device__ __forceinline__ int permutedSlot(int t)
{
    int const u = t & 15;
    int const slot16 = (u < 8) ? ((u >> 1) * 4 + (u & 1)) : (((u - 8) >> 1) * 4 + 2 + (u & 1));
    return (t & ~15) + slot16;
}

// The row of one lane of a warp's gather, found before the grid-dependency wait (kGatherBeforeWait).
struct EarlyGatherRow
{
    int64_t row;
    bool valid;
    // A position of this step's tokens: its row is written by the KV append this grid depends on.
    bool fresh;
};

// Lane l < 16 of the warp maps index slot `slot` of query token `tokenQ` (a position) to its pool row with the remap
// arithmetic, and bulk-copies the row unless it holds one of this step's tokens. Masked lanes copy row 0, as after the
// wait. Every read here (indices, block table, KV length, rows of earlier steps) precedes this grid's launch.
__device__ __forceinline__ EarlyGatherRow gatherRowBeforeWait(
    GlmSparseMlaFlashDecodeParams const& params, int tokenQ, int slot, int lane, uint32_t dst, uint32_t barGather)
{
    int position = -1;
    if (lane < kTokensPerWarp && slot < params.topK)
    {
        position = __ldg(params.indices + static_cast<int64_t>(tokenQ) * params.indicesStride + slot);
    }
    int const seqIdx = tokenQ / params.seqLenQ;
    int const tokenInSeq = tokenQ - seqIdx * params.seqLenQ;
    int const block = position / params.remapTokensPerBlock;
    int page = -1;
    if (position >= 0 && block < params.remapMaxBlocks)
    {
        page = __ldg(params.remapBlockTable + seqIdx * params.remapBlockTableStride + block);
    }
    int const inBlock = position - block * params.remapTokensPerBlock;
    int const rowRaw = page >= 0 ? page * params.remapStrideFactor + inBlock + params.remapLayerOffset : -1;
    int const seqLenKv = __ldg(params.seqLensKv + seqIdx);
    int const numValid = min(seqLenKv - (params.seqLenQ - 1 - tokenInSeq), params.topK);
    EarlyGatherRow out;
    out.valid = lane < kTokensPerWarp && slot < numValid && rowRaw >= 0;
    out.row = out.valid ? rowRaw : 0;
    out.fresh = out.valid && position >= seqLenKv - params.seqLenQ;
    if (lane == 0)
    {
        mbarrierArriveExpectTx(barGather, kTokensPerWarp * kRowBytes);
    }
    __syncwarp();
    if (lane < kTokensPerWarp && !out.fresh)
    {
        bulkCopyGlobalToShared(
            dst, static_cast<uint8_t const*>(params.kvPool) + out.row * kRowBytes, kRowBytes, barGather);
    }
    return out;
}

#endif // defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

// kRemap: the indices are positions, mapped to pool rows through the block table. kGatherBeforeWait (with kRemap): the
// indices, the block table and the KV length are read, and the rows of the positions before this step's tokens
// gathered, before the grid-dependency wait; the rows of this step's tokens after it.
template <int kHeads, bool kRemap, bool kGatherBeforeWait>
__global__ void __launch_bounds__(kThreads, 1) glmSparseMlaFlashDecodeKernel(GlmSparseMlaFlashDecodeParams const params)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static_assert(kRemap || !kGatherBeforeWait, "the early gather tells this step's tokens by their positions");
    using Layout = FoldedLayout<kHeads>;
    constexpr bool kStatsAfterPv = !Layout::kOwnMergeBuffers;
    extern __shared__ __align__(1024) uint8_t smem[];
    uint8_t* sKv = smem;
    uint8_t* sQ = smem + kOffQ;
    uint8_t* sP = smem + kOffP;
    float* sRedMax = reinterpret_cast<float*>(smem + kOffRed);
    float* sRedSum = sRedMax + kWarps * kMmaHeads;
    float* recvO = reinterpret_cast<float*>(smem + kOffRecvO);
    float* recvStats = reinterpret_cast<float*>(smem + kOffRecvStats);
    uint8_t* sSlice = smem + Layout::kOffSlice;
    uint8_t* sGatheredO = smem + Layout::kOffGatheredO;
    uint64_t* bars = reinterpret_cast<uint64_t*>(smem + Layout::kOffBars);

    int const tid = threadIdx.x;
    int const warp = tid / 32;
    int const lane = tid % 32;
    // The cluster spans gridDim.x, so the split index is the CTA's rank in the cluster.
    int const split = blockIdx.x;
    int const tokenQ = blockIdx.y;

    uint32_t const barGather = smemAddress(bars + warp);
    uint32_t const barQ = smemAddress(bars + kWarps);
    uint32_t const barRecv = smemAddress(bars + kWarps + 1);
    uint32_t const barVb = smemAddress(bars + kWarps + 2);
    uint32_t const barGatheredO = smemAddress(bars + kWarps + 3);
    {
        // Q rows kHeads .. 7 feed the MMAs' unused head columns: zero, before the barrier below orders them for the QK.
        constexpr int kZeroChunks = (kMmaHeads - kHeads) * kRowBytes / 16;
        static_assert(kZeroChunks <= kThreads && kRowBytes % 16 == 0, "one 16-byte zero store per thread");
        if (tid < kZeroChunks)
        {
            int const r = kHeads + tid / (kRowBytes / 16);
            int const chunk = tid % (kRowBytes / 16);
            *reinterpret_cast<uint4*>(sQ + r * kSmemRowBytes + 16 * chunk) = make_uint4(0, 0, 0, 0);
        }
    }
    if (tid == 0)
    {
        for (int i = 0; i < kNumBars; ++i)
        {
            mbarrierInit(smemAddress(bars + i), 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
        mbarrierArriveExpectTx(barRecv, kRecvBytes);
        mbarrierArriveExpectTx(barGatheredO, kGatheredOBytes);
    }
    __syncthreads();
    // Paired with the wait before the first remote store: no CTA writes into a receive buffer whose barrier other
    // CTAs of the cluster have not initialized yet.
    clusterArriveRelaxed();

    [[maybe_unused]] EarlyGatherRow early{};
    if constexpr (kGatherBeforeWait)
    {
        if (warp == 0)
        {
            // The block-table row of the token's sequence into L2 while the indices load: the remap below reads it
            // next. The CTAs of a cluster prefetch interleaved 128-byte lines of the row.
            char const* tableRow = reinterpret_cast<char const*>(
                params.remapBlockTable + (tokenQ / params.seqLenQ) * params.remapBlockTableStride);
            int64_t const rowBytes = static_cast<int64_t>(params.remapMaxBlocks) * sizeof(int32_t);
            for (int64_t off = (static_cast<int64_t>(split) * 32 + lane) * 128; off < rowBytes;
                 off += kSplits * 32 * 128)
            {
                asm volatile("prefetch.global.L2 [%0];\n" ::"l"(tableRow + off));
            }
        }
        int const slotEarly = split * kTokensPerCta + warp * kTokensPerWarp + lane;
        early = gatherRowBeforeWait(params, tokenQ, slotEarly, lane,
            smemAddress(sKv + (warp * kTokensPerWarp + lane) * kSmemRowBytes), barGather);
    }

    // Q, the indices, the KV pool rows and the scales come from earlier kernels.
    cudaGridDependencySynchronize();

    if (warp == 0)
    {
        if (lane == 0)
        {
            mbarrierArriveExpectTx(barQ, kHeads * kRowBytes);
        }
        __syncwarp();
        if (lane < kHeads)
        {
            bulkCopyGlobalToShared(smemAddress(sQ + lane * kSmemRowBytes),
                static_cast<uint8_t const*>(params.q) + (static_cast<int64_t>(tokenQ) * kHeads + lane) * kRowBytes,
                kRowBytes, barQ);
        }
    }

    // Lane l of warp w gathers token tokBase + l of the CTA. The index is read before the KV length is known, so the
    // two loads share one round trip.
    int const tokBase = warp * kTokensPerWarp;
    uint32_t validMask;
    if constexpr (kGatherBeforeWait)
    {
        // The rows of this step's tokens, which the KV append wrote.
        if (lane < kTokensPerWarp && early.fresh)
        {
            bulkCopyGlobalToShared(smemAddress(sKv + (tokBase + lane) * kSmemRowBytes),
                static_cast<uint8_t const*>(params.kvPool) + early.row * kRowBytes, kRowBytes, barGather);
        }
        validMask = __ballot_sync(0xffffffffU, early.valid);
    }
    else
    {
        int const slot = split * kTokensPerCta + tokBase + lane;
        int indexRaw = -1;
        if (lane < kTokensPerWarp && slot < params.topK)
        {
            indexRaw = __ldg(params.indices + static_cast<int64_t>(tokenQ) * params.indicesStride + slot);
        }
        int const seqIdx = tokenQ / params.seqLenQ;
        int const tokenInSeq = tokenQ - seqIdx * params.seqLenQ;
        if constexpr (kRemap)
        {
            // The index is a position in the sequence: map it to its pool row through the block table.
            int const block = indexRaw / params.remapTokensPerBlock;
            int page = -1;
            if (indexRaw >= 0 && block < params.remapMaxBlocks)
            {
                page = __ldg(params.remapBlockTable + seqIdx * params.remapBlockTableStride + block);
            }
            int const inBlock = indexRaw - block * params.remapTokensPerBlock;
            indexRaw = page >= 0 ? page * params.remapStrideFactor + inBlock + params.remapLayerOffset : -1;
        }
        int const numValid = min(__ldg(params.seqLensKv + seqIdx) - (params.seqLenQ - 1 - tokenInSeq), params.topK);
        bool const valid = lane < kTokensPerWarp && slot < numValid && indexRaw >= 0;
        // Masked lanes still load a row (row 0 of the pool) so that every gather barrier expects the same bytes.
        int64_t const row = valid ? indexRaw : 0;
        if (lane == 0)
        {
            mbarrierArriveExpectTx(barGather, kTokensPerWarp * kRowBytes);
        }
        __syncwarp();
        if (lane < kTokensPerWarp)
        {
            bulkCopyGlobalToShared(smemAddress(sKv + (tokBase + lane) * kSmemRowBytes),
                static_cast<uint8_t const*>(params.kvPool) + row * kRowBytes, kRowBytes, barGather);
        }
        validMask = __ballot_sync(0xffffffffU, valid);
    }
    float const scaleSoftmaxLog2 = __ldg(params.scaleSoftmaxLog2);
    float const outputScale = __ldg(params.outputScale);

    // Fragment coordinates of mma.sync: g = row group, c = column pair.
    int const g = lane / 4;
    int const c = lane % 4;

    // S^T = K Q^T for the warp's 16 tokens and the 8 head columns; two accumulators over alternating k steps. The
    // query fragments of every k step are unpacked once, while the warp's token rows are in flight.
    mbarrierWait(barQ, 0);
    uint32_t qh[kHeadDimQk / 32][4];
    {
        uint32_t const qAddr = smemAddress(sQ + (lane % 8) * kSmemRowBytes + 16 * ((lane / 8) % 2));
#pragma unroll
        for (int kd = 0; kd < kHeadDimQk / 32; ++kd)
        {
            uint32_t b0, b1;
            ldmatrixX2(qAddr + 32 * kd, b0, b1);
            unpackE4m3B(qh[kd], b0, b1);
        }
    }
    mbarrierWait(barGather, 0);
    float s[4];
    {
        float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        uint32_t const kAddr
            = smemAddress(sKv + (tokBase + lane % 8 + 8 * ((lane / 8) % 2)) * kSmemRowBytes + 16 * (lane / 16));
#pragma unroll
        for (int kd = 0; kd < kHeadDimQk / 32; ++kd)
        {
            uint32_t a0, a1, a2, a3;
            ldmatrixX4(kAddr + 32 * kd, a0, a1, a2, a3);
            mmaE4m3AsF16(acc[kd % 2], a0, a1, a2, a3, qh[kd]);
        }
        bool const valid0 = (validMask >> g) & 1U;
        bool const valid1 = (validMask >> (g + 8)) & 1U;
        s[0] = valid0 ? (acc[0][0] + acc[1][0]) * scaleSoftmaxLog2 : -INFINITY;
        s[1] = valid0 ? (acc[0][1] + acc[1][1]) * scaleSoftmaxLog2 : -INFINITY;
        s[2] = valid1 ? (acc[0][2] + acc[1][2]) * scaleSoftmaxLog2 : -INFINITY;
        s[3] = valid1 ? (acc[0][3] + acc[1][3]) * scaleSoftmaxLog2 : -INFINITY;
    }

    // The CTA's max per head column; this thread holds columns 2c and 2c + 1.
    float max0 = fmaxf(s[0], s[2]);
    float max1 = fmaxf(s[1], s[3]);
#pragma unroll
    for (int offset = 4; offset < 32; offset *= 2)
    {
        max0 = fmaxf(max0, __shfl_xor_sync(0xffffffffU, max0, offset));
        max1 = fmaxf(max1, __shfl_xor_sync(0xffffffffU, max1, offset));
    }
    if (g == 0)
    {
        sRedMax[warp * kMmaHeads + 2 * c] = max0;
        sRedMax[warp * kMmaHeads + 2 * c + 1] = max1;
    }
    // Every warp observed its own gather barrier before its QK; this barrier orders all token rows before the P.V
    // reads of the other warps.
    __syncthreads();
    if (warp == 1)
    {
        // v_b_proj[h][16 r, 16 r + 16)[0, 512) for every head h: a weight, so it may be read at any point. It is
        // issued once every token row has landed and arrives during QK / P.V.
        if (lane == 0)
        {
            mbarrierArriveExpectTx(barVb, Layout::kVbBytes);
        }
        __syncwarp();
        if (lane < kHeads)
        {
            bulkCopyGlobalToShared(smemAddress(smem + kOffVb + lane * kVbHeadBytes),
                static_cast<uint8_t const*>(params.vbProj)
                    + (static_cast<int64_t>(lane) * kVHeadDim + split * kVPerCta) * kHeadDimV * 2,
                kVbHeadBytes, barVb);
        }
    }
    float ctaMax0 = -INFINITY;
    float ctaMax1 = -INFINITY;
#pragma unroll
    for (int w = 0; w < kWarps; ++w)
    {
        ctaMax0 = fmaxf(ctaMax0, sRedMax[w * kMmaHeads + 2 * c]);
        ctaMax1 = fmaxf(ctaMax1, sRedMax[w * kMmaHeads + 2 * c + 1]);
    }
    // A CTA without valid tokens keeps P = 0.
    float const offset0 = ctaMax0 == -INFINITY ? 0.f : ctaMax0;
    float const offset1 = ctaMax1 == -INFINITY ? 0.f : ctaMax1;

    // P in E4M3 at the permuted k slots, and the per-head sums of the fp32 P.
    {
        float const p00 = exp2f(s[0] - offset0);
        float const p01 = exp2f(s[1] - offset1);
        float const p10 = exp2f(s[2] - offset0);
        float const p11 = exp2f(s[3] - offset1);
        int const slot0 = permutedSlot(tokBase + g);
        int const slot1 = permutedSlot(tokBase + g + 8);
        sP[(2 * c) * kPRowBytes + slot0] = toE4m3(p00 * kPScale);
        sP[(2 * c + 1) * kPRowBytes + slot0] = toE4m3(p01 * kPScale);
        sP[(2 * c) * kPRowBytes + slot1] = toE4m3(p10 * kPScale);
        sP[(2 * c + 1) * kPRowBytes + slot1] = toE4m3(p11 * kPScale);
        float sum0 = p00 + p10;
        float sum1 = p01 + p11;
#pragma unroll
        for (int offset = 4; offset < 32; offset *= 2)
        {
            sum0 += __shfl_xor_sync(0xffffffffU, sum0, offset);
            sum1 += __shfl_xor_sync(0xffffffffU, sum1, offset);
        }
        if (g == 0)
        {
            sRedSum[warp * kMmaHeads + 2 * c] = sum0;
            sRedSum[warp * kMmaHeads + 2 * c + 1] = sum1;
        }
    }
    __syncthreads();

    // Dedicated merge buffers permit early statistics. Reused P/KV storage must wait until
    // all P.V reads in the cluster are complete before any CTA can merge into it.
    clusterWait();
    if constexpr (!kStatsAfterPv)
    {
        if (tid < kSplits * kMmaHeads)
        {
            int const dst = tid / kMmaHeads;
            int const h = tid % kMmaHeads;
            float ctaMax = -INFINITY;
            float ctaSum = 0.f;
#pragma unroll
            for (int w = 0; w < kWarps; ++w)
            {
                ctaMax = fmaxf(ctaMax, sRedMax[w * kMmaHeads + h]);
                ctaSum += sRedSum[w * kMmaHeads + h];
            }
            storeRemote(mapToRank(smemAddress(recvStats + (split * kMmaHeads + h) * 2), dst), ctaMax, ctaSum,
                mapToRank(barRecv, dst));
        }
    }

    // O^T = V^T P^T: warp w owns the 16-column output blocks 4w .. 4w + 3, one block at a time over all k steps, each
    // block sent to its merging CTA while the next one is computed. ldmatrix.trans of 8 tokens x 16 bytes gives
    // (token 2c, columns 2g and 2g + 1) and (token 2c + 1, same columns); the byte permutes regroup them into a V^T row
    // (one column, four tokens), so fragment row g holds column 2g and row g + 8 holds column 2g + 1.
    {
        uint32_t const pAddr = smemAddress(sP + (lane % 8) * kPRowBytes + 16 * ((lane / 8) % 2));
        uint32_t const vAddr = smemAddress(sKv + lane * kSmemRowBytes + 16 * (4 * warp));
        uint32_t pb[kTokensPerCta / 32][4];
#pragma unroll
        for (int kStep = 0; kStep < kTokensPerCta / 32; ++kStep)
        {
            uint32_t b0, b1;
            ldmatrixX2(pAddr + 32 * kStep, b0, b1);
            unpackE4m3B(pb[kStep], b0, b1);
        }
#pragma unroll
        for (int bi = 0; bi < 4; ++bi)
        {
            float ob[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int kStep = 0; kStep < kTokensPerCta / 32; ++kStep)
            {
                uint32_t r0, r1, r2, r3;
                ldmatrixX4Trans(vAddr + 32 * kStep * kSmemRowBytes + 16 * bi, r0, r1, r2, r3);
                uint32_t const a0 = __byte_perm(r0, r1, 0x6420);
                uint32_t const a1 = __byte_perm(r0, r1, 0x7531);
                uint32_t const a2 = __byte_perm(r2, r3, 0x6420);
                uint32_t const a3 = __byte_perm(r2, r3, 0x7531);
                mmaE4m3AsF16(ob, a0, a1, a2, a3, pb[kStep]);
            }
            // Columns [32 r, 32 r + 32) of the partial output go to CTA r.
            int const col = 16 * (4 * warp + bi) + 2 * g;
            int const dst = col / kDvPerCta;
            int const colInSlice = col % kDvPerCta;
            uint32_t const remoteBar = mapToRank(barRecv, dst);
            storeRemote(mapToRank(smemAddress(recvO + (split * kMmaHeads + 2 * c) * kDvPerCta + colInSlice), dst),
                ob[0], ob[2], remoteBar);
            storeRemote(mapToRank(smemAddress(recvO + (split * kMmaHeads + 2 * c + 1) * kDvPerCta + colInSlice), dst),
                ob[1], ob[3], remoteBar);
        }
    }
    if constexpr (kStatsAfterPv)
    {
        __syncthreads();
        if (tid < kSplits * kMmaHeads)
        {
            int const dst = tid / kMmaHeads;
            int const h = tid % kMmaHeads;
            float ctaMax = -INFINITY;
            float ctaSum = 0.f;
#pragma unroll
            for (int w = 0; w < kWarps; ++w)
            {
                ctaMax = fmaxf(ctaMax, sRedMax[w * kMmaHeads + h]);
                ctaSum += sRedSum[w * kMmaHeads + h];
            }
            storeRemote(mapToRank(smemAddress(recvStats + (split * kMmaHeads + h) * 2), dst), ctaMax, ctaSum,
                mapToRank(barRecv, dst));
        }
    }
    // The dependents (the o_proj GEMM) wait for the whole grid before reading out; launching them now lets them stage
    // their weights during the merge and the v_b product.
    cudaTriggerProgrammaticLaunchCompletion();

    // Merge the kSplits partials of this CTA's output slice: thread (h, column) of the 8 x 32 slice. Warp h merges head
    // column h; lane r < kSplits computes the weight of partial r once, and the loop takes the weights from the lanes
    // in partial order.
    mbarrierWaitCluster(barRecv, 0);
    int const h = tid / kDvPerCta;
    int const colInSlice = tid % kDvPerCta;
    float value;
    {
        float const laneMax = lane < kSplits ? recvStats[(lane * kMmaHeads + h) * 2] : -INFINITY;
        float mergedMax = laneMax;
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
        {
            mergedMax = fmaxf(mergedMax, __shfl_xor_sync(0xffffffffU, mergedMax, offset));
        }
        float const laneAlpha = laneMax == -INFINITY ? 0.f : exp2f(laneMax - mergedMax);
        float mergedSum = 0.f;
        float acc = 0.f;
#pragma unroll
        for (int r = 0; r < kSplits; ++r)
        {
            float const alpha = __shfl_sync(0xffffffffU, laneAlpha, r);
            mergedSum += alpha * recvStats[(r * kMmaHeads + h) * 2 + 1];
            acc += alpha * recvO[(r * kMmaHeads + h) * kDvPerCta + colInSlice];
        }
        value = mergedSum > 0.f ? acc / (mergedSum * kPScale) * outputScale : 0.f;
    }

    // The merged slice in BF16 (the attention output the v_b product reads) goes to every CTA of the cluster. The merge
    // maps thread tid to head column tid / 32, so each warp holds and sends one head row.
    reinterpret_cast<__nv_bfloat16*>(sSlice)[h * kDvPerCta + colInSlice] = __float2bfloat16_rn(value);
    __syncwarp();
    {
        int const chunk = lane % 4;
        uint4 const packed = *reinterpret_cast<uint4 const*>(sSlice + (h * kDvPerCta + 8 * chunk) * 2);
        uint32_t const localAddr = smemAddress(sGatheredO + (h * kHeadDimV + split * kDvPerCta + 8 * chunk) * 2);
#pragma unroll
        for (int k = 0; k < 2; ++k)
        {
            uint32_t const dst = lane / 4 + 8 * k;
            storeRemote16(mapToRank(localAddr, dst), packed, mapToRank(barGatheredO, dst));
        }
    }
    mbarrierWaitCluster(barGatheredO, 0);
    mbarrierWait(barVb, 0);
    if (h >= kHeads)
    {
        // A head column without a v_b slice (warp-uniform: the merge maps warp h to head column h).
        return;
    }

    // out[t][h * 256 + 16 r + v] = sum_d O[h][d] v_b[h][16 r + v][d] for v = 0 .. 15: warp h, one m16n8k16 row tile of
    // the 16 v rows (1 KiB apart). Per pair of k steps a thread loads 16 bytes of its rows g and g + 8; rows with odd g
    // take the other 32-column half of every 64-column block, so the eight rows of a load cover all banks. B column n
    // follows the same parity as rows with g = n, so the dot product of row g is D[g][g % 2] (thread c = 0).
    {
        int const parity = g % 2;
        uint8_t const* rowG = smem + kOffVb + h * kVbHeadBytes + g * (kHeadDimV * 2) + 16 * c;
        uint8_t const* rowG8 = rowG + 8 * (kHeadDimV * 2);
        uint8_t const* oRow = sGatheredO + h * (kHeadDimV * 2) + 16 * c;
        float acc[4][4];
#pragma unroll
        for (int a = 0; a < 4; ++a)
        {
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                acc[a][i] = 0.f;
            }
        }
#pragma unroll
        for (int kPair = 0; kPair < kHeadDimV / 32; ++kPair)
        {
            int const block = (kPair ^ parity) * 64;
            uint4 const ra = *reinterpret_cast<uint4 const*>(rowG + block);
            uint4 const rb = *reinterpret_cast<uint4 const*>(rowG8 + block);
            uint4 const ob = *reinterpret_cast<uint4 const*>(oRow + block);
            mmaBf16(acc[(2 * kPair) % 4], ra.x, rb.x, ra.y, rb.y, ob.x, ob.y);
            mmaBf16(acc[(2 * kPair + 1) % 4], ra.z, rb.z, ra.w, rb.w, ob.z, ob.w);
        }
        if (c == 0)
        {
            float sums[4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                sums[i] = ((acc[0][i] + acc[1][i]) + acc[2][i]) + acc[3][i];
            }
            __nv_bfloat16* outRow = static_cast<__nv_bfloat16*>(params.out)
                + static_cast<int64_t>(tokenQ) * params.outStride + h * kVHeadDim + split * kVPerCta;
            outRow[g] = __float2bfloat16_rn(parity ? sums[1] : sums[0]);
            outRow[g + 8] = __float2bfloat16_rn(parity ? sums[3] : sums[2]);
        }
    }
#endif // defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

// 16 query heads per token (Heads16 above): the 8-head kernel's gather, softmax and merge with two 8-head MMA tiles on
// every K / V fragment. QK and P.V issue the FP16 MMAs that the SM100 family runs for the E4M3 MMA (mmaE4m3AsF16),
// with the K / V fragment unpacked once for both head tiles. kGatherBeforeWait: as in sparseMlaFlashDecodeKernel.
template <int kSplits16, bool kRemap, bool kGatherBeforeWait>
__global__ void __launch_bounds__(Heads16<kSplits16>::kThreads, 1)
    glmSparseMlaFlashDecodeHeads16Kernel(GlmSparseMlaFlashDecodeParams const params)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static_assert(kRemap || !kGatherBeforeWait, "the early gather tells this step's tokens by their positions");
    using h16 = Heads16<kSplits16>;
    extern __shared__ __align__(1024) uint8_t smem[];
    uint8_t* sKv = smem;
    uint8_t* sQ = smem + h16::kOffQ;
    uint8_t* sP = smem + h16::kOffP;
    float* sRedMax = reinterpret_cast<float*>(smem + h16::kOffRed);
    float* sRedSum = sRedMax + h16::kWarps * h16::kHeads;
    float* recvO = reinterpret_cast<float*>(smem + h16::kOffRecvO);
    float* recvStats = reinterpret_cast<float*>(smem + h16::kOffRecvStats);
    uint64_t* bars = reinterpret_cast<uint64_t*>(smem + h16::kOffBars);

    int const tid = threadIdx.x;
    int const warp = tid / 32;
    int const lane = tid % 32;
    // The cluster spans gridDim.x, so the split index is the CTA's rank in the cluster.
    int const split = blockIdx.x;
    int const tokenQ = blockIdx.y;

    uint32_t const barGather = smemAddress(bars + warp);
    uint32_t const barQ = smemAddress(bars + h16::kWarps);
    uint32_t const barRecv = smemAddress(bars + h16::kWarps + 1);
    if (tid == 0)
    {
        for (int i = 0; i < h16::kNumBars; ++i)
        {
            mbarrierInit(smemAddress(bars + i), 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
        mbarrierArriveExpectTx(barRecv, h16::kRecvBytes);
    }
    __syncthreads();
    // Paired with the wait before the first remote store: no CTA writes into a receive buffer whose barrier other
    // CTAs of the cluster have not initialized yet.
    clusterArriveRelaxed();

    int const tokBase = warp * kTokensPerWarp;
    [[maybe_unused]] EarlyGatherRow early{};
    if constexpr (kGatherBeforeWait)
    {
        if (warp == 0)
        {
            // The block-table row of the token's sequence into L2 while the indices load: the remap below reads it
            // next. The CTAs of a cluster prefetch interleaved 128-byte lines of the row.
            char const* tableRow = reinterpret_cast<char const*>(
                params.remapBlockTable + (tokenQ / params.seqLenQ) * params.remapBlockTableStride);
            int64_t const rowBytes = static_cast<int64_t>(params.remapMaxBlocks) * sizeof(int32_t);
            for (int64_t off = (static_cast<int64_t>(split) * 32 + lane) * 128; off < rowBytes;
                 off += h16::kSplits * 32 * 128)
            {
                asm volatile("prefetch.global.L2 [%0];\n" ::"l"(tableRow + off));
            }
        }
        early = gatherRowBeforeWait(params, tokenQ, split * h16::kTokensPerCta + tokBase + lane, lane,
            smemAddress(sKv + (tokBase + lane) * kSmemRowBytes), barGather);
    }

    // Q, the indices, the KV pool rows and the scales come from earlier kernels.
    cudaGridDependencySynchronize();
    if (params.triggerAfterWait != 0)
    {
        // The dependent reads this grid's output only after its own grid-dependency wait, which waits for this
        // whole grid; its CTAs may land and stage their weights while this kernel runs.
        cudaTriggerProgrammaticLaunchCompletion();
    }

    if (warp == 0)
    {
        if (lane == 0)
        {
            mbarrierArriveExpectTx(barQ, h16::kHeads * kRowBytes);
        }
        __syncwarp();
        if (lane < h16::kHeads)
        {
            bulkCopyGlobalToShared(smemAddress(sQ + lane * kSmemRowBytes),
                static_cast<uint8_t const*>(params.q) + (static_cast<int64_t>(tokenQ) * h16::kHeads + lane) * kRowBytes,
                kRowBytes, barQ);
        }
    }

    // Lane l of warp w gathers token tokBase + l of the CTA, as in sparseMlaFlashDecodeKernel.
    uint32_t validMask;
    if constexpr (kGatherBeforeWait)
    {
        // The rows of this step's tokens, which the KV append wrote.
        if (lane < kTokensPerWarp && early.fresh)
        {
            bulkCopyGlobalToShared(smemAddress(sKv + (tokBase + lane) * kSmemRowBytes),
                static_cast<uint8_t const*>(params.kvPool) + early.row * kRowBytes, kRowBytes, barGather);
        }
        validMask = __ballot_sync(0xffffffffU, early.valid);
    }
    else
    {
        int const slot = split * h16::kTokensPerCta + tokBase + lane;
        int indexRaw = -1;
        if (lane < kTokensPerWarp && slot < params.topK)
        {
            indexRaw = __ldg(params.indices + static_cast<int64_t>(tokenQ) * params.indicesStride + slot);
        }
        int const seqIdx = tokenQ / params.seqLenQ;
        int const tokenInSeq = tokenQ - seqIdx * params.seqLenQ;
        if constexpr (kRemap)
        {
            // The index is a position in the sequence: map it to its pool row through the block table.
            int const block = indexRaw / params.remapTokensPerBlock;
            int page = -1;
            if (indexRaw >= 0 && block < params.remapMaxBlocks)
            {
                page = __ldg(params.remapBlockTable + seqIdx * params.remapBlockTableStride + block);
            }
            int const inBlock = indexRaw - block * params.remapTokensPerBlock;
            indexRaw = page >= 0 ? page * params.remapStrideFactor + inBlock + params.remapLayerOffset : -1;
        }
        int const numValid = min(__ldg(params.seqLensKv + seqIdx) - (params.seqLenQ - 1 - tokenInSeq), params.topK);
        bool const valid = lane < kTokensPerWarp && slot < numValid && indexRaw >= 0;
        // Masked lanes still load a row (row 0 of the pool) so that every gather barrier expects the same bytes.
        int64_t const row = valid ? indexRaw : 0;
        if (lane == 0)
        {
            mbarrierArriveExpectTx(barGather, kTokensPerWarp * kRowBytes);
        }
        __syncwarp();
        if (lane < kTokensPerWarp)
        {
            bulkCopyGlobalToShared(smemAddress(sKv + (tokBase + lane) * kSmemRowBytes),
                static_cast<uint8_t const*>(params.kvPool) + row * kRowBytes, kRowBytes, barGather);
        }
        validMask = __ballot_sync(0xffffffffU, valid);
    }
    float const scaleSoftmaxLog2 = __ldg(params.scaleSoftmaxLog2);
    float const outputScale = __ldg(params.outputScale);

    // Fragment coordinates of mma.sync: g = row group, c = column pair.
    int const g = lane / 4;
    int const c = lane % 4;

    // S^T = K Q^T for the warp's 16 tokens and the 16 heads (head tile t = heads 8 t .. 8 t + 7); two accumulators per
    // tile over alternating k steps.
    mbarrierWait(barQ, 0);
    mbarrierWait(barGather, 0);
    float s[h16::kHeadTiles][4];
    {
        float acc[h16::kHeadTiles][2][4];
#pragma unroll
        for (int t = 0; t < h16::kHeadTiles; ++t)
        {
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                acc[t][0][i] = 0.f;
                acc[t][1][i] = 0.f;
            }
        }
        uint32_t const qAddr = smemAddress(sQ + (lane % 8) * kSmemRowBytes + 16 * ((lane / 8) % 2));
        uint32_t const kAddr
            = smemAddress(sKv + (tokBase + lane % 8 + 8 * ((lane / 8) % 2)) * kSmemRowBytes + 16 * (lane / 16));
#pragma unroll
        for (int kd = 0; kd < kHeadDimQk / 32; ++kd)
        {
            uint32_t a0, a1, a2, a3;
            ldmatrixX4(kAddr + 32 * kd, a0, a1, a2, a3);
            uint32_t ah[8];
            unpackE4m3A(ah, a0, a1, a2, a3);
#pragma unroll
            for (int t = 0; t < h16::kHeadTiles; ++t)
            {
                uint32_t b0, b1;
                ldmatrixX2(qAddr + t * kMmaHeads * kSmemRowBytes + 32 * kd, b0, b1);
                uint32_t bh[4];
                unpackE4m3B(bh, b0, b1);
                mmaE4m3AsF16Unpacked(acc[t][kd % 2], ah, bh);
            }
        }
        bool const valid0 = (validMask >> g) & 1U;
        bool const valid1 = (validMask >> (g + 8)) & 1U;
#pragma unroll
        for (int t = 0; t < h16::kHeadTiles; ++t)
        {
            s[t][0] = valid0 ? (acc[t][0][0] + acc[t][1][0]) * scaleSoftmaxLog2 : -INFINITY;
            s[t][1] = valid0 ? (acc[t][0][1] + acc[t][1][1]) * scaleSoftmaxLog2 : -INFINITY;
            s[t][2] = valid1 ? (acc[t][0][2] + acc[t][1][2]) * scaleSoftmaxLog2 : -INFINITY;
            s[t][3] = valid1 ? (acc[t][0][3] + acc[t][1][3]) * scaleSoftmaxLog2 : -INFINITY;
        }
    }

    // The CTA's max per head; this thread holds heads 8 t + 2 c and 8 t + 2 c + 1.
#pragma unroll
    for (int t = 0; t < h16::kHeadTiles; ++t)
    {
        float max0 = fmaxf(s[t][0], s[t][2]);
        float max1 = fmaxf(s[t][1], s[t][3]);
#pragma unroll
        for (int offset = 4; offset < 32; offset *= 2)
        {
            max0 = fmaxf(max0, __shfl_xor_sync(0xffffffffU, max0, offset));
            max1 = fmaxf(max1, __shfl_xor_sync(0xffffffffU, max1, offset));
        }
        if (g == 0)
        {
            sRedMax[warp * h16::kHeads + kMmaHeads * t + 2 * c] = max0;
            sRedMax[warp * h16::kHeads + kMmaHeads * t + 2 * c + 1] = max1;
        }
    }
    // Every warp observed its own gather barrier before its QK; this barrier orders all token rows before the P.V
    // reads of the other warps.
    __syncthreads();

    // P in E4M3 at the permuted k slots, and the per-head sums of the fp32 P.
    {
        int const slot0 = permutedSlot(tokBase + g);
        int const slot1 = permutedSlot(tokBase + g + 8);
#pragma unroll
        for (int t = 0; t < h16::kHeadTiles; ++t)
        {
            int const head0 = kMmaHeads * t + 2 * c;
            float ctaMax0 = -INFINITY;
            float ctaMax1 = -INFINITY;
#pragma unroll
            for (int w = 0; w < h16::kWarps; ++w)
            {
                ctaMax0 = fmaxf(ctaMax0, sRedMax[w * h16::kHeads + head0]);
                ctaMax1 = fmaxf(ctaMax1, sRedMax[w * h16::kHeads + head0 + 1]);
            }
            // A CTA without valid tokens keeps P = 0.
            float const offset0 = ctaMax0 == -INFINITY ? 0.f : ctaMax0;
            float const offset1 = ctaMax1 == -INFINITY ? 0.f : ctaMax1;
            float const p00 = exp2f(s[t][0] - offset0);
            float const p01 = exp2f(s[t][1] - offset1);
            float const p10 = exp2f(s[t][2] - offset0);
            float const p11 = exp2f(s[t][3] - offset1);
            sP[head0 * h16::kPRowBytes + slot0] = toE4m3(p00 * kPScale);
            sP[(head0 + 1) * h16::kPRowBytes + slot0] = toE4m3(p01 * kPScale);
            sP[head0 * h16::kPRowBytes + slot1] = toE4m3(p10 * kPScale);
            sP[(head0 + 1) * h16::kPRowBytes + slot1] = toE4m3(p11 * kPScale);
            float sum0 = p00 + p10;
            float sum1 = p01 + p11;
#pragma unroll
            for (int offset = 4; offset < 32; offset *= 2)
            {
                sum0 += __shfl_xor_sync(0xffffffffU, sum0, offset);
                sum1 += __shfl_xor_sync(0xffffffffU, sum1, offset);
            }
            if (g == 0)
            {
                sRedSum[warp * h16::kHeads + head0] = sum0;
                sRedSum[warp * h16::kHeads + head0 + 1] = sum1;
            }
        }
    }
    __syncthreads();

    // This CTA's max and sum per head go to every CTA of the cluster; the partial output follows after P.V.
    clusterWait();
    if (tid < h16::kSplits * h16::kHeads)
    {
        int const dst = tid / h16::kHeads;
        int const h = tid % h16::kHeads;
        float ctaMax = -INFINITY;
        float ctaSum = 0.f;
#pragma unroll
        for (int w = 0; w < h16::kWarps; ++w)
        {
            ctaMax = fmaxf(ctaMax, sRedMax[w * h16::kHeads + h]);
            ctaSum += sRedSum[w * h16::kHeads + h];
        }
        storeRemote(mapToRank(smemAddress(recvStats + (split * h16::kHeads + h) * 2), dst), ctaMax, ctaSum,
            mapToRank(barRecv, dst));
    }

    // O^T = V^T P^T: warp w owns the 16-column output blocks kBlocksPerWarp * w + bi, for both head tiles (see
    // sparseMlaFlashDecodeKernel for the fragment layout). Each V fragment is unpacked once for the two tiles.
    float o[h16::kBlocksPerWarp][h16::kHeadTiles][4];
#pragma unroll
    for (int bi = 0; bi < h16::kBlocksPerWarp; ++bi)
    {
#pragma unroll
        for (int t = 0; t < h16::kHeadTiles; ++t)
        {
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                o[bi][t][i] = 0.f;
            }
        }
    }
    {
        uint32_t const pAddr = smemAddress(sP + (lane % 8) * h16::kPRowBytes + 16 * ((lane / 8) % 2));
        uint32_t const vAddr = smemAddress(sKv + lane * kSmemRowBytes + 16 * (h16::kBlocksPerWarp * warp));
#pragma unroll
        for (int kStep = 0; kStep < h16::kTokensPerCta / 32; ++kStep)
        {
            uint32_t ph[h16::kHeadTiles][4];
#pragma unroll
            for (int t = 0; t < h16::kHeadTiles; ++t)
            {
                uint32_t pb0, pb1;
                ldmatrixX2(pAddr + t * kMmaHeads * h16::kPRowBytes + 32 * kStep, pb0, pb1);
                unpackE4m3B(ph[t], pb0, pb1);
            }
#pragma unroll
            for (int bi = 0; bi < h16::kBlocksPerWarp; ++bi)
            {
                uint32_t r0, r1, r2, r3;
                ldmatrixX4Trans(vAddr + 32 * kStep * kSmemRowBytes + 16 * bi, r0, r1, r2, r3);
                uint32_t ah[8];
                unpackE4m3A(ah, __byte_perm(r0, r1, 0x6420), __byte_perm(r0, r1, 0x7531), __byte_perm(r2, r3, 0x6420),
                    __byte_perm(r2, r3, 0x7531));
#pragma unroll
                for (int t = 0; t < h16::kHeadTiles; ++t)
                {
                    mmaE4m3AsF16Unpacked(o[bi][t], ah, ph[t]);
                }
            }
        }
    }

    // Columns [kDvPerCta r, kDvPerCta (r + 1)) of the partial output go to CTA r.
#pragma unroll
    for (int bi = 0; bi < h16::kBlocksPerWarp; ++bi)
    {
        int const col = 16 * (h16::kBlocksPerWarp * warp + bi) + 2 * g;
        int const dst = col / h16::kDvPerCta;
        int const colInSlice = col % h16::kDvPerCta;
        uint32_t const remoteBar = mapToRank(barRecv, dst);
#pragma unroll
        for (int t = 0; t < h16::kHeadTiles; ++t)
        {
            int const head0 = kMmaHeads * t + 2 * c;
            storeRemote(
                mapToRank(smemAddress(recvO + (split * h16::kHeads + head0) * h16::kDvPerCta + colInSlice), dst),
                o[bi][t][0], o[bi][t][2], remoteBar);
            storeRemote(
                mapToRank(smemAddress(recvO + (split * h16::kHeads + head0 + 1) * h16::kDvPerCta + colInSlice), dst),
                o[bi][t][1], o[bi][t][3], remoteBar);
        }
    }

    // Merge the kSplits partials of this CTA's output slice: thread t merges two columns of head t / kColPairs.
    mbarrierWaitCluster(barRecv, 0);
    {
        int const h = tid / h16::kColPairs;
        int const colInSlice = 2 * (tid % h16::kColPairs);
        float mergedMax = -INFINITY;
#pragma unroll
        for (int r = 0; r < h16::kSplits; ++r)
        {
            mergedMax = fmaxf(mergedMax, recvStats[(r * h16::kHeads + h) * 2]);
        }
        float mergedSum = 0.f;
        float acc0 = 0.f;
        float acc1 = 0.f;
#pragma unroll
        for (int r = 0; r < h16::kSplits; ++r)
        {
            float const partMax = recvStats[(r * h16::kHeads + h) * 2];
            float const alpha = partMax == -INFINITY ? 0.f : exp2f(partMax - mergedMax);
            mergedSum += alpha * recvStats[(r * h16::kHeads + h) * 2 + 1];
            float2 const part
                = *reinterpret_cast<float2 const*>(recvO + (r * h16::kHeads + h) * h16::kDvPerCta + colInSlice);
            acc0 += alpha * part.x;
            acc1 += alpha * part.y;
        }
        float const value0 = mergedSum > 0.f ? acc0 / (mergedSum * kPScale) * outputScale : 0.f;
        float const value1 = mergedSum > 0.f ? acc1 / (mergedSum * kPScale) * outputScale : 0.f;
        *reinterpret_cast<__nv_bfloat162*>(static_cast<__nv_bfloat16*>(params.out)
            + static_cast<int64_t>(tokenQ) * params.outStride + h * kHeadDimV + split * h16::kDvPerCta + colInSlice)
            = __floats2bfloat162_rn(value0, value1);
    }
    // The dependents may launch once every CTA has stored its slice.
    cudaTriggerProgrammaticLaunchCompletion();
#endif // defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

template <int kHeads, bool kRemap, bool kGatherBeforeWait>
void launchGlmSparseMlaFlashDecode(GlmSparseMlaFlashDecodeParams const& params, cudaStream_t stream)
{
    using Layout = FoldedLayout<kHeads>;
    auto const kernel = glmSparseMlaFlashDecodeKernel<kHeads, kRemap, kGatherBeforeWait>;
    static bool const configured = [kernel]()
    {
        TLLM_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Layout::kBytes));
        TLLM_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        return true;
    }();
    (void) configured;

    cudaLaunchConfig_t config;
    config.gridDim = dim3(kSplits, params.numTokensQ, 1);
    config.blockDim = dim3(kThreads, 1, 1);
    config.dynamicSmemBytes = Layout::kBytes;
    config.stream = stream;
    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = kSplits;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[1].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.attrs = attrs;
    config.numAttrs = 2;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, params));
}

template <int kSplits16, bool kRemap, bool kGatherBeforeWait>
void launchGlmSparseMlaFlashDecodeHeads16(GlmSparseMlaFlashDecodeParams const& params, cudaStream_t stream)
{
    using Layout = Heads16<kSplits16>;
    auto const kernel = glmSparseMlaFlashDecodeHeads16Kernel<kSplits16, kRemap, kGatherBeforeWait>;
    static bool const configured = [kernel]()
    {
        TLLM_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Layout::kBytes));
        TLLM_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        return true;
    }();
    (void) configured;

    cudaLaunchConfig_t config;
    config.gridDim = dim3(Layout::kSplits, params.numTokensQ, 1);
    config.blockDim = dim3(Layout::kThreads, 1, 1);
    config.dynamicSmemBytes = Layout::kBytes;
    config.stream = stream;
    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = Layout::kSplits;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[1].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.attrs = attrs;
    config.numAttrs = 2;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, params));
}

template <int kSplits16>
void launchGlmSparseMlaFlashDecodeHeads16Splits(
    GlmSparseMlaFlashDecodeParams const& params, bool remap, cudaStream_t stream)
{
    if (remap && params.gatherBeforeWait != 0)
    {
        launchGlmSparseMlaFlashDecodeHeads16<kSplits16, true, true>(params, stream);
    }
    else if (remap)
    {
        launchGlmSparseMlaFlashDecodeHeads16<kSplits16, true, false>(params, stream);
    }
    else
    {
        launchGlmSparseMlaFlashDecodeHeads16<kSplits16, false, false>(params, stream);
    }
}

template <int kHeads>
void launchGlmSparseMlaFlashDecodeFolded(GlmSparseMlaFlashDecodeParams const& params, bool remap, cudaStream_t stream)
{
    if (remap && params.gatherBeforeWait)
    {
        launchGlmSparseMlaFlashDecode<kHeads, true, true>(params, stream);
    }
    else if (remap)
    {
        launchGlmSparseMlaFlashDecode<kHeads, true, false>(params, stream);
    }
    else
    {
        launchGlmSparseMlaFlashDecode<kHeads, false, false>(params, stream);
    }
}

} // namespace

void invokeGlmSparseMlaFlashDecode(GlmSparseMlaFlashDecodeParams const& params, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(params.topK > 0 && params.topK <= kGlmSparseMlaMaxTopK,
        "GLM sparse MLA flash decode supports TopK up to %d, got %d", kGlmSparseMlaMaxTopK, params.topK);
    TLLM_CHECK_WITH_INFO(params.seqLenQ > 0 && params.indicesStride >= params.topK && params.numTokensQ >= 0
            && params.numTokensQ % params.seqLenQ == 0,
        "Invalid GLM sparse MLA geometry");
    TLLM_CHECK_WITH_INFO(params.numHeadsQ == 4 || params.numHeadsQ == 8, "GLM folded sparse MLA requires 4 or 8 heads");
    TLLM_CHECK_WITH_INFO(
        params.vbProj != nullptr && params.out != nullptr && params.outStride >= params.numHeadsQ * kVHeadDim,
        "GLM sparse MLA flash decode needs v_b_proj and an output of %d columns per token",
        params.numHeadsQ * kVHeadDim);
    if (params.numTokensQ == 0)
    {
        return;
    }
    bool const remap = params.remapBlockTable != nullptr;
    if (remap)
    {
        TLLM_CHECK_WITH_INFO(params.remapTokensPerBlock > 0 && params.remapMaxBlocks > 0
                && params.remapBlockTableStride >= params.remapMaxBlocks,
            "Invalid GLM sparse MLA index remap geometry");
    }
    TLLM_CHECK_WITH_INFO(!params.gatherBeforeWait || remap,
        "The GLM sparse MLA flash decode gathers before its wait only with the index remap");
    if (params.numHeadsQ == kGlmSparseMlaNumHeads)
    {
        launchGlmSparseMlaFlashDecodeFolded<kGlmSparseMlaNumHeads>(params, remap, stream);
    }
    else
    {
        launchGlmSparseMlaFlashDecodeFolded<kGlmSparseMlaNumHeadsTp8>(params, remap, stream);
    }
}

void invokeGlmSparseMlaFlashDecodeLatent(GlmSparseMlaFlashDecodeParams const& params, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(params.topK > 0 && params.topK <= kGlmSparseMlaMaxTopK,
        "GLM sparse MLA flash decode supports TopK up to %d, got %d", kGlmSparseMlaMaxTopK, params.topK);
    TLLM_CHECK_WITH_INFO(params.seqLenQ > 0 && params.indicesStride >= params.topK && params.numTokensQ >= 0
            && params.numTokensQ % params.seqLenQ == 0,
        "Invalid GLM sparse MLA geometry");
    TLLM_CHECK_WITH_INFO(params.numTokensQ <= kGlmSparseMlaMaxWideTokens,
        "GLM latent sparse MLA supports at most %d query tokens", kGlmSparseMlaMaxWideTokens);
    TLLM_CHECK_WITH_INFO(
        params.vbProj == nullptr && params.out != nullptr && params.outStride >= kGlmSparseMlaNumHeadsWide * kHeadDimV,
        "GLM latent sparse MLA requires an unfused [tokens, 16, 512] output");
    if (params.numTokensQ == 0)
    {
        return;
    }
    bool const remap = params.remapBlockTable != nullptr;
    if (remap)
    {
        TLLM_CHECK_WITH_INFO(params.remapTokensPerBlock > 0 && params.remapMaxBlocks > 0
                && params.remapBlockTableStride >= params.remapMaxBlocks,
            "Invalid GLM sparse MLA index remap geometry");
    }
    TLLM_CHECK_WITH_INFO(!params.gatherBeforeWait || remap,
        "The GLM sparse MLA flash decode gathers before its wait only with the index remap");
    if (params.numTokensQ <= 8)
    {
        launchGlmSparseMlaFlashDecodeHeads16Splits<16>(params, remap, stream);
    }
    else
    {
        launchGlmSparseMlaFlashDecodeHeads16Splits<8>(params, remap, stream);
    }
}

} // namespace kernels::glm_moe_dsa

TRTLLM_NAMESPACE_END
