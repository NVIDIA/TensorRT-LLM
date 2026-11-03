/*
 * MIT License
 *
 * Copyright (c) 2025 DeepSeek
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
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
 *
 * reference: https://github.com/deepseek-ai/FlashMLA (csrc/sm100/decode/head64)
 */

// CSA2 sparse decode attention for the SM100 family (tcgen05), derived from FlashMLA's SM100 head64 sparse decode.
//
// KV sources are the CSA2 packed pools, read directly:
//   - SWA blocks  = SWA pool rows (528 B: 512 x e4m3 + 16 x ue8m0 per-32 scales), int64 indices
//   - main blocks = main pool rows (288 B: 256 B e2m1 + 32 x e4m3 per-16 scales), int32 physical indices, or int32
//                   logical positions mapped through a page table by the index warp
// Whole rows are gathered with TMA gather4 and dequantized exactly to BF16 (64 keys x 512 channels, SW128 K-major)
// with native cvt.rn.bf16x2.{e4m3x2,e2m1x2,ue8m0x2} and exact bf16x2 scale multiplies; K and V are the same tile.
// MMA math follows FlashMLA: BF16 UMMA, M=64 dual-GEMM QK^T, FP32 accumulation, P rounded to BF16.
//
// Work split (static, one persistent CTA per SM): CTA p processes F whole tokens; the KV blocks of the remaining
// M - F * numCtas tokens are cut into contiguous chunks over remCtas CTAs. A token cut by a chunk boundary is split:
// each part writes a normalized FP32 partial and its LSE to its CTA's slot and counts itself in a per-token arrival
// counter, and a combine kernel merges the parts in a fixed order (deterministic) as soon as all of them have
// arrived. Whole tokens are written directly (BF16 or MXFP8).
//
// PDL: the prologue (barriers, TMEM, schedule, role setup) runs before the grid dependency wait; every role that
// touches global memory waits first. Split items are processed first, so under PDL the combine kernel merges them
// beside the attention CTAs that are still running whole tokens; it completes only after the attention grid.
//
// Warp roles (512 threads):
//   WG0  softmax                   WG1  w4 QK^T issuer, w5 raw-row TMA producer, w6 PV issuer, w7 indices
//   WG2  dequant (raw -> BF16)     WG3  Q loader (TMA -> smem -> regs -> TMEM) and output epilogue
// Token transitions do not drain the pipeline. P has two TMEM buffers and S two smem buffers; the softmax exchanges the
// dual-GEMM partial logits in two rounds through S, and PV-hi of block g is queued behind QK^T of block g + 1.

#pragma once

#include "tensorrt_llm/kernels/csa2SparseDecode/csa2SparseDecode.h"
#include "tensorrt_llm/kernels/csa2SparseDecode/csa2SparseDecodeUtils.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/barrier.h>
#include <math_constants.h>

// The kernels use tcgen05, TMA gather4 and the native FP4/FP8 conversions of the SM100 family. Other targets (and
// placeholder builds) compile a trapping body; the host side rejects unsupported devices before launching.
#if defined(__CUDA_ARCH__)
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL)                                         \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__)                                                                         \
        && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000 || __CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))
#define CSA2_SPARSE_DECODE_ARCH_SUPPORTED 1
#endif
#endif

TRTLLM_NAMESPACE_BEGIN

namespace kernels::csa2SparseDecode
{

//! Upper bound on the CTAs sharing one split token: the most parts makeSchedule may cut one token into.
constexpr int kMaxCombineSplits = 8;

// Device-side arguments of both kernels (FlashMLA naming).
struct KernelParams
{
    int num_tokens;

    int64_t const* swa_indices;
    int64_t stride_swa_idx;
    int swa_topk;
    int64_t swa_rows;

    int32_t const* main_indices;
    int64_t stride_main_idx;
    int main_topk;
    int64_t main_rows;

    // Logical main selections (main_page_table != nullptr).
    int32_t const* main_page_table;
    int64_t main_page_table_stride;
    int main_page_table_rows;
    int main_page_table_cols;
    int main_page_shift;
    int64_t main_max_positions;
    int64_t const* token_requests;
    int64_t const* token_visible;

    float const* attn_sink;
    float sm_scale_log2;
    int q_fp8;     // Q is E4M3: every token's Q goes through the Q loader warpgroup, which converts it to BF16
    void const* q; // BF16 Q of the later tokens, read by the Q loader warpgroup
    int64_t stride_q_tok;
    int64_t stride_q_head;

    bf16* out;
    int64_t stride_o_tok;
    int64_t stride_o_head;

    // MXFP8 output instead of out (see Csa2SparseDecodeParams::outFp8).
    uint8_t* out_fp8;
    uint8_t* out_scale;
    int32_t const* positions;
    float const* rope_cos_sin;

    // Split-KV partials: o_accum [2 * num_ctas, 64, 512] fp32, lse_accum [2 * num_ctas, 64] fp32.
    float* o_accum;
    float* lse_accum;

    int num_ctas;
    int full_per_cta;
    int rem_ctas;
    int pdl; // launched with programmatic dependent launch: the combine kernel overlaps the attention kernel
    // Host-computed reciprocals ceil(2^64 / d) for the divisions of the static schedule (0 encodes d == 1).
    uint64_t magic_rem_ctas;   // d = max(rem_ctas, 1)
    uint64_t magic_rem_blocks; // d = max(remainder tokens * blocks per token, 1)
    uint64_t magic_nb;         // d = blocks per token
};

using namespace cute;
using cutlass::arch::fence_view_async_shared;
using cutlass::arch::NamedBarrier;

enum NamedBarriers : uint32_t
{
    wg0_sync = 1,
    wg0_warp02_sync = 2,
    wg0_warp13_sync = 3,
    wg3_sync = 5,
    dequant_sync = 6,
};

// ---------------------------------------------------------------------------------------------------------------
// Small device helpers
// ---------------------------------------------------------------------------------------------------------------

// Native exact conversions: one F2FP per pair of values.
__device__ __forceinline__ uint32_t e4m3x2ToBf16x2(uint32_t v16)
{
    uint32_t r;
    asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"((uint16_t) v16));
    return r;
}

__device__ __forceinline__ uint32_t ue8m0x2ToBf16x2(uint32_t v16)
{
    uint32_t r;
    asm("cvt.rn.bf16x2.ue8m0x2 %0, %1;" : "=r"(r) : "h"((uint16_t) v16));
    return r;
}

// 8 x e2m1 (4 bytes, low nibble = even channel) -> 4 x bf16x2 (exact)
__device__ __forceinline__ void e2m1x8ToBf16x2x4(uint32_t v, uint32_t (&h)[4])
{
    asm("{\n"
        ".reg .b8 byte0, byte1, byte2, byte3;\n"
        "mov.b32 {byte0, byte1, byte2, byte3}, %4;\n"
        "cvt.rn.bf16x2.e2m1x2 %0, byte0;\n"
        "cvt.rn.bf16x2.e2m1x2 %1, byte1;\n"
        "cvt.rn.bf16x2.e2m1x2 %2, byte2;\n"
        "cvt.rn.bf16x2.e2m1x2 %3, byte3;\n"
        "}\n"
        : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3])
        : "r"(v));
}

// bf16x2 multiply. Exact here: e2m1 x e4m3 has <= 6 significant bits, e4m3 x 2^k is a power-of-two scaling.
__device__ __forceinline__ uint32_t bf16x2Mul(uint32_t a, uint32_t b)
{
    uint32_t r;
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

__device__ __forceinline__ float fmax3f(float a, float b, float c)
{
    float r;
    asm("max.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
    return r;
}

__device__ __forceinline__ uint32_t float2ToBf16x2(float2 v)
{
    __nv_bfloat162 b = __float22bfloat162_rn(v);
    return *reinterpret_cast<uint32_t*>(&b);
}

__device__ __forceinline__ void stSharedV4(uint32_t addr, uint4 const& v)
{
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n" ::"r"(addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
}

__device__ __forceinline__ void bulkWaitGroupRead0()
{
    asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
}

__device__ __forceinline__ void bulkWaitGroup0()
{
    asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
}

__device__ __forceinline__ void bulkCommitGroup()
{
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void fenceProxyAsyncSmem()
{
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

// ---------------------------------------------------------------------------------------------------------------
// MXFP8 output: projection.inverse_rope_mxfp8 of the BF16 output, bit for bit
// ---------------------------------------------------------------------------------------------------------------

constexpr int kHeadsPerGroup = kNumHeads / kOutputGroups;

#if defined(CSA2_SPARSE_DECODE_TMEM_LOAD_MAX)
constexpr bool kTmemLoadMax = true; // O blocks come out of TMEM with their maximum magnitude
#else
constexpr bool kTmemLoadMax = false;
#endif

//! A BF16 pair as two floats (exact).
__device__ __forceinline__ float2 bf16PairToFloat2(uint32_t w)
{
    return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xFFFF0000U));
}

//! The inverse rotation of a GPT-J channel pair (a, b): (a * cos + b * sin, b * cos - a * sin) with the products
//! b * sin and a * sin rounded on their own, rounded to BF16.
__device__ __forceinline__ float2 inverseRopePair(float2 x, float cosv, float sinv)
{
    return bf16PairToFloat2(float2ToBf16x2(
        make_float2(__fmaf_rn(x.x, cosv, __fmul_rn(x.y, sinv)), __fmaf_rn(x.y, cosv, -__fmul_rn(x.x, sinv)))));
}

//! UE8M0 exponent of a 32-channel block with maximum magnitude amax (2^(e - 127) >= amax / 448) and the factor
//! 2^(127 - e) that scales the block into the E4M3 range (0 for an all-zero block).
__device__ __forceinline__ uint32_t mxfp8Exponent(float amax, float& factor)
{
    float const scale = amax * (1.F / 448.F);
    uint32_t const bits = __float_as_uint(scale);
    uint32_t exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0);
    exponent = scale <= 0x1p-127F ? 0 : min(exponent, 254U);
    factor = amax == 0.F ? 0.F : __uint_as_float((254 - exponent) << 23);
    return exponent;
}

//! Two floats as an E4M3 pair (x in the low byte), rounded to nearest with saturation.
__device__ __forceinline__ uint32_t toE4m3x2(float x, float y)
{
    uint16_t r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(y), "f"(x));
    return r;
}

//! Byte offset of the 4-scale word (blocks 4 * word .. 4 * word + 3) of row `row` of `group` in the CUTLASS
//! 128x4 layout of [kOutputGroups][pad128(numTokens), 128] UE8M0 scales.
__device__ __forceinline__ int64_t scaleWordOffset(int group, int row, int word, int numTokens)
{
    int const sfRows = (numTokens + 127) / 128 * 128;
    return (int64_t) group * sfRows * 128 + (row / 128) * 16384 + (row % 32) * 16 + (row % 128) / 32 * 4 + word * 512;
}

// ---------------------------------------------------------------------------------------------------------------
// Static work split: contiguous, balanced ranges of the flattened (token, 64-key block) space
// ---------------------------------------------------------------------------------------------------------------

// floor(n / d) for n < 2^32 with magic = ceil(2^64 / d) (d < 2^32; magic 0 encodes d == 1).
__device__ __forceinline__ uint32_t divMagic(uint32_t n, uint64_t magic)
{
    return magic == 0 ? n : (uint32_t) __umul64hi((uint64_t) n, magic);
}

struct WorkSplit
{
    int nb;          // 64-key blocks per token
    int num_orig;    // SWA blocks per token (first blocks of a token)
    int full_tokens; // tokens processed whole: CTA p owns [p * F, (p + 1) * F)
    int TR;          // remainder blocks: (M - full_tokens) * nb, cut into PR contiguous chunks
    int PR;          // CTAs sharing the remainder
    uint64_t magic_pr, magic_tr;

    // The host checks TR * PR < 2^32, so the products below fit 32 bits.
    __device__ __forceinline__ int rem_begin(int p) const
    {
        return (int) divMagic((uint32_t) p * (uint32_t) TR, magic_pr);
    }

    // CTA that owns remainder block b
    __device__ __forceinline__ int rem_owner(int b) const
    {
        return (int) divMagic(((uint32_t) b + 1u) * (uint32_t) PR - 1u, magic_tr);
    }
};

//! Per remainder token: [0] parts of the token published by the attention kernel, [1] combine CTAs done with it.
//! They are zero when the module loads, and the last combine CTA of a token resets both, so every launch starts and
//! ends with all counters at zero. Like the workspace, they are shared by the launches of a device, which therefore
//! must not run concurrently.
constexpr int kMaxSplitTokens = 2048;
constexpr int kCombineHeadsPerCta = 8;
__device__ int csa2SplitArrivals[kMaxSplitTokens][2];

struct Csa2Kernel
{
    static constexpr int D_Q = kHeadDim;
    static constexpr int D_K = kHeadDim;
    static constexpr int D_V = kHeadDim;

    static constexpr int B_H = kNumHeads;
    static constexpr int B_TOPK = 64;
    // Three K/V buffers: a block's raw rows are gathered into the head of its buffer and dequantized in place.
    static constexpr int NUM_KV_BUFS = 3;
    static constexpr int NUM_INDEX_BUFS = 8;
    static constexpr int NUM_THREADS = 128 * 4;
    static constexpr float MAX_INIT_VAL = -1e30f;

    static constexpr int SOFTMAX_REGS = 176;
    static constexpr int PRODUCER_REGS = 48;
    static constexpr int DEQUANT_REGS = 120;
    static constexpr int IO_REGS = 168;
    static_assert(SOFTMAX_REGS + PRODUCER_REGS + DEQUANT_REGS + IO_REGS == 512);

    // TMA destinations must be 128-B aligned: each gather4 group of 4 rows starts on a 128-B boundary.
    static constexpr int SWA_GROUP_PITCH = (4 * kSwaRowBytes + 127) / 128 * 128;   // 2176
    static constexpr int MAIN_GROUP_PITCH = (4 * kMainRowBytes + 127) / 128 * 128; // 1152
    static_assert((B_TOPK / 4) * SWA_GROUP_PITCH <= B_TOPK * kHeadDim * 2);        // the raw rows fit their K/V buffer
    // A main block's raw rows lie in the buffer's channels [0, 256): only PV-lo of the buffer's previous block has to
    // be done before they are gathered.
    static_assert((B_TOPK / 4) * MAIN_GROUP_PITCH <= B_TOPK * kHeadDim);

    // WG3 staging (8 KB): per-warp epilogue tiles (32 rows x 64 B, SW64: bf16 32 channels or fp32 16 channels) and,
    // between epilogues, one 64x128 e4m3 SW128 Q box.
    static constexpr int EPI_WARP_TILE_BYTES = 32 * 64;
    static constexpr int EPI_BYTES = 4 * EPI_WARP_TILE_BYTES;

    // Tensor memory columns: O [0, 256), Q [256, 384), P[2] [384, 512)
    struct tmem_cols
    {
        static constexpr int O = 0;
        static constexpr int Q = 256;
        static constexpr int P = 384;
    };

    using SmemLayoutS
        = decltype(tile_to_shape(UMMA::Layout_K_INTER_Atom<bf16>{}, Shape<Int<B_H>, Int<B_TOPK>>{}, Step<_1, _2>{}));

    template <int NUM_TILES>
    using SmemLayoutKTiles_SW128 = decltype(coalesce(
        tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<B_H>, Int<64 * NUM_TILES>>{}, Step<_1, _2>{}),
        Shape<_1, _1>{}));

    template <int NUM_TILES>
    using SmemLayoutKTiles_DualGemm_SW128 = decltype(coalesce(
        tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<B_H * 2>, Int<64 * NUM_TILES>>{}, Step<_1, _2>{}),
        Shape<_1, _1>{}));

    template <int NUM_TILES>
    using SmemLayoutKTilesTransposed_SW128 = decltype(composition(SmemLayoutKTiles_SW128<NUM_TILES>{},
        Layout<Shape<Int<64 * NUM_TILES>, Int<B_TOPK>>, Stride<Int<B_TOPK>, _1>>{}));

    struct SharedMemoryPlan
    {
        array_aligned<bf16, B_TOPK * D_K, 1024> kv[NUM_KV_BUFS]; // 8 SW128 tiles of 64 x 64 bf16 each
        array_aligned<uint8_t, EPI_BYTES, 1024> epi;
        // S (bf16 P) double buffer; the dual-GEMM partial-sum exchange of block g uses s[g % 2] (in two rounds) before
        // S(g) is written
        array_aligned<bf16, cosize_v<SmemLayoutS>, 1024> s[2];
        float rowwise_max_buf[2][128];
        float rowwise_li_buf[128];
        float stats_scale[2][B_H];
        float stats_lse[2][B_H];
        char is_token_valid[NUM_INDEX_BUFS][B_TOPK / 8];
        CUTE_ALIGNAS(16) int tma_coord[NUM_INDEX_BUFS][B_TOPK];
        array_aligned<uint32_t, 1> tmem_start_addr;
        transac_bar_t bar_index_ready[NUM_INDEX_BUFS], bar_index_free[NUM_INDEX_BUFS];
        transac_bar_t bar_raw_ready[NUM_KV_BUFS];
        transac_bar_t bar_kv_ready_lo[NUM_KV_BUFS], bar_kv_ready[NUM_KV_BUFS], bar_sv_done[NUM_KV_BUFS];
        transac_bar_t bar_sv_done_lo[NUM_KV_BUFS]; // PV over V channels [0, 256) done: that half may be rewritten
        transac_bar_t bar_qk_done[2], bar_p_free[2];
        // QK^T of block g >= 1 fully issued (buffer g % 2): PV-hi(g - 1) may queue behind it
        transac_bar_t bar_qkhi_issued[2];
        transac_bar_t bar_so_ready[2]; // per S buffer: softmax(g+1) may finish before PV(g) is issued
        transac_bar_t bar_q_ready, bar_q_free;
        transac_bar_t bar_qstage_full, bar_qstage_free, bar_qload;
        transac_bar_t bar_o_full, bar_o_free_lo, bar_o_free_hi;
        transac_bar_t bar_stats_ready[2], bar_stats_free[2];
    };

    static constexpr size_t SMEM_SIZE = sizeof(SharedMemoryPlan);
    static_assert(SMEM_SIZE <= 232448, "shared memory plan exceeds 227 KB");

    using TiledMMA_P = decltype(make_tiled_mma(
        MmaWsTsBf16NoElect<bf16, bf16, float, B_H, B_TOPK * 2, UMMA::Major::K, UMMA::Major::K>{}));

    using TiledMMA_O
        = decltype(make_tiled_mma(MmaWsSsBf16NoElect<bf16, bf16, float, B_H, 256, UMMA::Major::K, UMMA::Major::MN>{}));

    struct TmaParams
    {
        CUtensorMap tensor_map_swa;
        CUtensorMap tensor_map_main;
        CUtensorMap tensor_map_q;           // bf16 [M, 64, 512], box {64, 64, 1}, SW128
        CUtensorMap tensor_map_out;         // bf16 [M, 64, 512], box {32, 32, 1}, SW64
        CUtensorMap tensor_map_oaccum;      // fp32 [2 * num_ctas, 64, 512], box {16, 32, 1}, SW64
        CUtensorMap tensor_map_out_wide;    // bf16, box {64, 32, 1}, SW128 (last item, staged in the free KV buffers)
        CUtensorMap tensor_map_oaccum_wide; // fp32, box {32, 32, 1}, SW128
    };

    static __device__ void devfunc(KernelParams const& params, TmaParams const& tma_params);
};

// ---------------------------------------------------------------------------------------------------------------
// Kernel body
// ---------------------------------------------------------------------------------------------------------------

__device__ void Csa2Kernel::devfunc(KernelParams const& params, TmaParams const& tma_params)
{
#if defined(CSA2_SPARSE_DECODE_ARCH_SUPPORTED)
    int const warpgroup_idx = cutlass::canonical_warp_group_idx();
    int const idx_in_warpgroup = threadIdx.x % 128;
    int const warp_idx = cutlass::canonical_warp_idx_sync();
    int const lane_idx = threadIdx.x % 32;

    extern __shared__ __align__(1024) char wksp_buf[];
    SharedMemoryPlan& plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);

    if (warp_idx == 0)
    {
        if (elect_one_sync())
        {
            cute::prefetch_tma_descriptor(&tma_params.tensor_map_swa);
            cute::prefetch_tma_descriptor(&tma_params.tensor_map_main);
            cute::prefetch_tma_descriptor(&tma_params.tensor_map_q);
            cute::prefetch_tma_descriptor(&tma_params.tensor_map_out);
            cute::prefetch_tma_descriptor(&tma_params.tensor_map_oaccum);
            cute::prefetch_tma_descriptor(&tma_params.tensor_map_out_wide);
            cute::prefetch_tma_descriptor(&tma_params.tensor_map_oaccum_wide);
            for (int i = 0; i < NUM_INDEX_BUFS; ++i)
            {
                plan.bar_index_ready[i].init(32);
                plan.bar_index_free[i].init(128 + 16); // softmax WG + the 16 raw-producer lanes that read coordinates
            }
            for (int i = 0; i < NUM_KV_BUFS; ++i)
            {
                plan.bar_raw_ready[i].init(1);
                plan.bar_kv_ready_lo[i].init(128);
                plan.bar_kv_ready[i].init(128);
                plan.bar_sv_done[i].init(1);
                plan.bar_sv_done_lo[i].init(1);
            }
            for (int i = 0; i < 2; ++i)
            {
                plan.bar_qk_done[i].init(1);
                plan.bar_qkhi_issued[i].init(1);
                plan.bar_p_free[i].init(128);
                plan.bar_stats_ready[i].init(128);
                plan.bar_stats_free[i].init(128);
            }
            plan.bar_so_ready[0].init(128);
            plan.bar_so_ready[1].init(128);
            plan.bar_q_ready.init(128);
            plan.bar_q_free.init(1);
            plan.bar_qstage_full.init(1);
            plan.bar_qstage_free.init(1);
            plan.bar_qload.init(1);
            plan.bar_o_full.init(1);
            plan.bar_o_free_lo.init(128);
            plan.bar_o_free_hi.init(128);
            cutlass::arch::fence_barrier_init();
        }
        cute::TMEM::Allocator1Sm().allocate(512, plan.tmem_start_addr.data());
        if (plan.tmem_start_addr.data()[0] != 0)
        {
            asm volatile("trap;");
        }
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }
    __syncthreads();
    // PDL: the prologue, the schedule and the role setup overlap the previous kernel's tail. Every role that touches
    // global memory (inputs, outputs, workspace, split counters) waits for the previous kernel first (role entry
    // points below); thread 0 releases the dependent combine kernel only after its own wait.

    // ------------------------------------------------------------------ static schedule of this CTA
    // Remainder (split) tokens first, so that every split token is finished early by all of its CTAs and its combine
    // overlaps the whole tokens that follow.
    int const swa_blocks = max((params.swa_topk + B_TOPK - 1) / B_TOPK, 1);
    int const main_blocks = (params.main_topk + B_TOPK - 1) / B_TOPK;
    int const nb = swa_blocks + main_blocks;
    int const full_tokens = params.full_per_cta * (int) gridDim.y;
    WorkSplit const ws{nb, swa_blocks, full_tokens, (params.num_tokens - full_tokens) * nb, max(params.rem_ctas, 1),
        params.magic_rem_ctas, params.magic_rem_blocks};
    int const cta = blockIdx.y;
    bool const has_rem = cta < params.rem_ctas && ws.TR > 0;
    int const r0 = has_rem ? ws.rem_begin(cta) : 0;
    int const r1 = has_rem ? ws.rem_begin(cta + 1) : 0;
    int const r0_tok = (int) divMagic((uint32_t) r0, params.magic_nb);
    int const nrem = r1 > r0 ? (int) divMagic((uint32_t) (r1 - 1), params.magic_nb) - r0_tok + 1 : 0;
    int const rem_first = full_tokens + r0_tok;
    int const num_items = nrem + params.full_per_cta;
    bool const has_work = num_items > 0;
    int const total_blocks = (nrem > 0 ? r1 - r0 : 0) + params.full_per_cta * nb;

    struct ItemArgs
    {
        int token_idx, start_block_idx, end_block_idx;
        bool is_split;
        int slot; // partial slot of this CTA for a split token
        int swa_length, main_length, num_orig_kv_blocks;
    };

    auto get_item = [&](int k) -> ItemArgs
    {
        if (k < nrem)
        {
            int const tok = rem_first + k;
            int const tb = (tok - full_tokens) * nb;
            bool const split = ws.rem_owner(tb) != ws.rem_owner(tb + nb - 1);
            return ItemArgs{tok, max(r0, tb) - tb, min(r1, tb + nb) - tb, split, 2 * cta + (k == 0 ? 0 : 1),
                params.swa_topk, params.main_topk, ws.num_orig};
        }
        int const tok = cta * params.full_per_cta + (k - nrem);
        return ItemArgs{tok, 0, nb, false, 0, params.swa_topk, params.main_topk, ws.num_orig};
    };

    auto for_each_item = [&](auto f)
    {
#pragma unroll 1
        for (int k = 0; k < num_items; ++k)
        {
            f(get_item(k), k);
        }
    };

    if (warpgroup_idx == 0)
    {
        // ======================================= Softmax warpgroup =======================================
        cutlass::arch::warpgroup_reg_alloc<SOFTMAX_REGS>();
        cudaGridDependencySynchronize();
        if (threadIdx.x == 0)
        {
            cudaTriggerProgrammaticLaunchCompletion(); // lets the combine kernel be scheduled
        }

        float2 const scale = float2{params.sm_scale_log2, params.sm_scale_log2};
        int const sS_off = lane_idx * 8 + (warp_idx & 1) * (B_H / 2) * 8 + (warp_idx / 2) * B_H * (B_TOPK / 2);
        float const attn_sink = __ldg(params.attn_sink + (idx_in_warpgroup % 64)) * CUDART_L2E_F;

        int g = 0;
        for_each_item(
            [&](ItemArgs const& args, int k)
            {
                float mi = MAX_INIT_VAL;
                float li = 0.0f;
                float real_mi = -CUDART_INF_F;

                CUTE_NO_UNROLL
                for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx, ++g)
                {
                    int const ib = g % NUM_INDEX_BUFS;
                    int const pb = g % 2;
                    float4* xchg = reinterpret_cast<float4*>(plan.s[pb].data()); // [4 warps][4 * 32] float4
                    // No block-entry barrier: the row-max exchange buffer is double buffered (g % 2), and s[pb] is
                    // only rewritten two blocks later, after every thread passed this block's row-max barrier.
                    plan.bar_index_ready[ib].wait((g / NUM_INDEX_BUFS) & 1);
                    plan.bar_qk_done[pb].wait((g / 2) & 1);
                    tcgen05AfterThreadSync();

                    float p[B_TOPK / 2], p_peer[B_TOPK / 2];
                    int const pcol = tmem_cols::P + pb * 64;
                    if (warp_idx < 2)
                    {
                        tmemLoad32dp32b32x(pcol, p);
                        tmemLoad32dp32b32x(pcol + 32, p_peer);
                    }
                    else
                    {
                        tmemLoad32dp32b32x(pcol, p_peer);
                        tmemLoad32dp32b32x(pcol + 32, p);
                    }
                    cutlass::arch::fence_view_async_tmem_load();
                    tcgen05BeforeThreadSync();
                    plan.bar_p_free[pb].arrive();

                    // Dual-GEMM partial-sum exchange with the partner warp (warp ^ 2) in two rounds of 16 values,
                    // through s[pb] (which PV(g-2) must have finished reading).
                    if (g >= 2)
                    {
                        plan.bar_sv_done[(g - 2) % NUM_KV_BUFS].wait(((g - 2) / NUM_KV_BUFS) & 1);
                    }
                    CUTE_UNROLL
                    for (int r = 0; r < 2; ++r)
                    {
                        if (r == 1)
                        {
                            // the partner has read round 0
                            NamedBarrier::arrive_and_wait(64, NamedBarriers::wg0_warp02_sync + (warp_idx & 1));
                        }
                        CUTE_UNROLL
                        for (int i = 0; i < 4; ++i)
                        {
                            xchg[(warp_idx ^ 2) * 128 + i * 32 + lane_idx] = *(float4*) (p_peer + (4 * r + i) * 4);
                        }
                        NamedBarrier::arrive_and_wait(64, NamedBarriers::wg0_warp02_sync + (warp_idx & 1));
                        CUTE_UNROLL
                        for (int i = 0; i < 4; ++i)
                        {
                            float2 t2[2];
                            *(float4*) t2 = xchg[warp_idx * 128 + i * 32 + lane_idx];
                            float2* cur_p = (float2*) (p + (4 * r + i) * 4);
                            cur_p[0] = float2Add(cur_p[0], t2[0]);
                            cur_p[1] = float2Add(cur_p[1], t2[1]);
                        }
                    }

                    uint32_t valid_mask = *((uint32_t*) plan.is_token_valid[ib] + (idx_in_warpgroup >= 64 ? 1 : 0));
                    if (valid_mask != 0xffffffffu)
                    {
                        CUTE_UNROLL
                        for (int i = 0; i < B_TOPK / 2; i += 1)
                        {
                            if (!(valid_mask >> i & 1))
                            {
                                p[i] = -CUDART_INF_F;
                            }
                        }
                    }

                    float cur_pi_max = fmax3f(p[0], p[1], p[2]);
                    CUTE_UNROLL
                    for (int i = 3; i < (B_TOPK / 2) - 1; i += 2)
                    {
                        cur_pi_max = fmax3f(cur_pi_max, p[i], p[i + 1]);
                    }
                    cur_pi_max = max(cur_pi_max, p[B_TOPK / 2 - 1]);
                    cur_pi_max *= params.sm_scale_log2;

                    plan.rowwise_max_buf[pb][idx_in_warpgroup] = cur_pi_max;
                    NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync); // also ends exchange reads of s[pb]
                    plan.bar_index_free[ib].arrive();
                    cur_pi_max = max(cur_pi_max, plan.rowwise_max_buf[pb][idx_in_warpgroup ^ 64]);
                    real_mi = max(real_mi, cur_pi_max);
                    bool const should_scale_o = __any_sync(0xffffffff, cur_pi_max - mi > 6.0f);

                    float new_max, scale_for_old;
                    if (!should_scale_o)
                    {
                        scale_for_old = 1.0f;
                        new_max = mi;
                    }
                    else
                    {
                        new_max = max(cur_pi_max, mi);
                        scale_for_old = exp2f(mi - new_max);
                    }
                    mi = new_max;

                    __nv_bfloat162 s[(B_TOPK / 2) / 2];
                    float2 const neg_new_max = float2{-new_max, -new_max};
                    float2 cur_sum = float2{0.0f, 0.0f};
                    CUTE_UNROLL
                    for (int i = 0; i < (B_TOPK / 2) / 2; i += 1)
                    {
                        float2 d = float2Fma(float2{p[i * 2], p[i * 2 + 1]}, scale, neg_new_max);
                        d.x = exp2f(d.x);
                        d.y = exp2f(d.y);
                        cur_sum = float2Add(cur_sum, d);
                        s[i] = __float22bfloat162_rn(d);
                    }
                    li = fma(li, scale_for_old, (cur_sum.x + cur_sum.y));

                    bf16* sS_base = plan.s[pb].data() + sS_off;
                    CUTE_UNROLL
                    for (int i = 0; i < B_TOPK / 2 / 8; i += 1)
                    {
                        *(uint128_t*) (sS_base + B_H * 8 * i) = *(uint128_t*) (s + i * 4);
                    }

                    if (block_idx != args.start_block_idx && should_scale_o)
                    {
                        // Lazy O rescale (rare): PV(g-1) must have completed; PV(g) waits for so_ready(g).
                        plan.bar_sv_done[(g - 1) % NUM_KV_BUFS].wait(((g - 1) / NUM_KV_BUFS) & 1);
                        tcgen05AfterThreadSync();
                        float2 const scale_for_old_float2 = float2{scale_for_old, scale_for_old};
                        static constexpr int CHUNK_SIZE = 32;
                        float2 o[CHUNK_SIZE / 2];
                        CUTE_UNROLL
                        for (int chunk_idx = 0; chunk_idx < (D_V / 2) / CHUNK_SIZE; ++chunk_idx)
                        {
                            tmemLoad32dp32b32x(tmem_cols::O + chunk_idx * CHUNK_SIZE, o);
                            cutlass::arch::fence_view_async_tmem_load();
                            CUTE_UNROLL
                            for (int i = 0; i < CHUNK_SIZE / 2; ++i)
                            {
                                o[i] = float2Mul(o[i], scale_for_old_float2);
                            }
                            tmemStore32dp32b32x(tmem_cols::O + chunk_idx * CHUNK_SIZE, o);
                        }
                        cutlass::arch::fence_view_async_tmem_store();
                    }
                    tcgen05BeforeThreadSync();
                    fence_view_async_shared();
                    plan.bar_so_ready[pb].arrive();
                }

                if (real_mi == -CUDART_INF_F)
                {
                    li = 0.0f;
                    mi = -CUDART_INF_F;
                }
                plan.rowwise_li_buf[idx_in_warpgroup] = li;
                NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
                li += plan.rowwise_li_buf[idx_in_warpgroup ^ 64];

                int const sb = k % 2;
                plan.bar_stats_free[sb].wait(((k / 2) & 1) ^ 1);
                if (idx_in_warpgroup < B_H)
                {
                    float o_scale, lse;
                    if (!args.is_split)
                    {
                        o_scale = li == 0.0f ? 0.0f : __fdividef(1.0f, li + exp2f(attn_sink - mi));
                        lse = 0.0f;
                    }
                    else
                    {
                        o_scale = li == 0.0f ? 0.0f : __fdividef(1.0f, li);
                        lse = li == 0.0f ? -CUDART_INF_F : log2f(li) + mi;
                    }
                    plan.stats_scale[sb][idx_in_warpgroup] = o_scale;
                    plan.stats_lse[sb][idx_in_warpgroup] = lse;
                }
                plan.bar_stats_ready[sb].arrive();
            });
    }
    else if (warpgroup_idx == 1)
    {
        cutlass::arch::warpgroup_reg_dealloc<PRODUCER_REGS>();
        int const warp_idx = cutlass::canonical_warp_idx_sync();

        if (warp_idx == 4 || warp_idx == 6)
        {
            // ======================================= MMA issuers =======================================
            // Two independent issuing threads: warp 4 issues QK^T (and the first token's Q UTCCP), warp 6 issues PV.
            // Each blocks on its own barriers; tcgen05.commit only tracks the issuing thread's MMAs, so ordering
            // between the two streams is expressed with explicit barriers (sv_done, p_free, o_free).
            if (elect_one_sync() && has_work)
            {
                TiledMMA tiled_mma_P = TiledMMA_P{};
                TiledMMA tiled_mma_O = TiledMMA_O{};
                Tensor tP = partition_fragment_C(tiled_mma_P, Shape<Int<B_H>, _128>{});
                Tensor tO_lo = partition_fragment_C(tiled_mma_O, Shape<Int<B_H>, Int<D_V / 2>>{});
                Tensor tO_hi = partition_fragment_C(tiled_mma_O, Shape<Int<B_H>, Int<D_V / 2>>{});
                tO_lo.data().get() = tmem_cols::O;
                tO_hi.data().get() = tmem_cols::O + 128;
                // QK^T is issued in two K halves (channels [0,256) and [256,512)) so that the first half overlaps the
                // dequantization of the second half of the KV tile.
                Tensor tQ_lo = tiled_mma_P.get_slice(_0{}).make_fragment_A(
                    partition_shape_A(tiled_mma_P, Shape<Int<B_H>, Int<D_Q / 4>>{}));
                Tensor tQ_hi = tiled_mma_P.get_slice(_0{}).make_fragment_A(
                    partition_shape_A(tiled_mma_P, Shape<Int<B_H>, Int<D_Q / 4>>{}));
                tQ_lo.data().get() = tmem_cols::Q;
                tQ_hi.data().get() = tmem_cols::Q + 64;

                if (warp_idx == 4 && params.q_fp8)
                {
                    // KV buffer 1 stages no Q.
                    plan.bar_qstage_free.arrive();
                }
                if (warp_idx == 4)
                {
                    cudaGridDependencySynchronize();
                    // Q of the first BF16 token: TMA into the (still unused) KV buffer 1, then UTCCP into TMEM.
                    int const tok0 = get_item(0).token_idx;
                    CUTE_UNROLL
                    for (int t = 0; t < D_Q / 64 && !params.q_fp8; ++t)
                    {
                        cute::SM90_TMA_LOAD_3D::copy(&tma_params.tensor_map_q, (uint64_t*) &plan.bar_qstage_full,
                            (uint64_t) TMA::CacheHintSm90::EVICT_FIRST, plan.kv[1].data() + t * 64 * 64, t * 64, 0,
                            tok0);
                    }
                    if (!params.q_fp8)
                    {
                        plan.bar_qstage_full.arrive_and_expect_tx(B_H * D_Q * sizeof(bf16));
                        plan.bar_qstage_full.wait(0);
                        tcgen05AfterThreadSync();
                        UMMA::SmemDescriptor sQ_desc
                            = UMMA::make_umma_desc<UMMA::Major::K>(make_tensor(make_smem_ptr(plan.kv[1].data()),
                                tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<B_H * 2>, Int<64>>{})));
                        CUTE_UNROLL
                        for (int tile_idx = 0; tile_idx < D_Q / 128; ++tile_idx)
                        {
                            CUTE_UNROLL
                            for (int subtile_idx = 0; subtile_idx < 64 / 16; ++subtile_idx)
                            {
                                SM100_UTCCP_128dp256bit_1cta::copy(
                                    sQ_desc + (tile_idx * (B_H * 128) + subtile_idx * 16) * 2 / 16,
                                    tmem_cols::Q + tile_idx * 32 + subtile_idx * 8);
                            }
                        }
                        ummaArriveNoElect(plan.bar_qstage_free);
                    }
                    plan.bar_qstage_free.wait(0);
                    tcgen05AfterThreadSync();

                    int g = 0;
                    for_each_item(
                        [&](ItemArgs const& args, int k)
                        {
                            CUTE_NO_UNROLL
                            for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx, ++g)
                            {
                                bool const first = block_idx == args.start_block_idx;
                                bool const last = block_idx == args.end_block_idx - 1;
                                int const pb = g % 2;
                                int const kb = g % NUM_KV_BUFS;
                                if (first && (k > 0 || params.q_fp8))
                                {
                                    plan.bar_q_ready.wait((params.q_fp8 ? k : k - 1) & 1);
                                }
                                plan.bar_p_free[pb].wait(((g / 2) & 1) ^ 1);
                                plan.bar_kv_ready_lo[kb].wait((g / NUM_KV_BUFS) & 1);
                                tcgen05AfterThreadSync();
                                tP.data().get() = tmem_cols::P + pb * 64;
                                {
                                    Tensor sK = make_tensor(make_smem_ptr(plan.kv[kb].data()),
                                        SmemLayoutKTiles_DualGemm_SW128<D_K / 64 / 4>{});
                                    utcmmaTs(tiled_mma_P, tQ_lo, sK, tP, true);
                                }
                                plan.bar_kv_ready[kb].wait((g / NUM_KV_BUFS) & 1);
                                tcgen05AfterThreadSync();
                                {
                                    Tensor sK = make_tensor(make_smem_ptr(plan.kv[kb].data() + B_TOPK * (D_K / 2)),
                                        SmemLayoutKTiles_DualGemm_SW128<D_K / 64 / 4>{});
                                    utcmmaTs(tiled_mma_P, tQ_hi, sK, tP, false);
                                }
                                ummaArriveNoElect(plan.bar_qk_done[pb]);
                                if (g > 0)
                                {
                                    plan.bar_qkhi_issued[pb].arrive();
                                }
                                if (last)
                                {
                                    ummaArriveNoElect(plan.bar_q_free);
                                }
                            }
                        });
                }
                else
                {
                    int g = 0;
                    for_each_item(
                        [&](ItemArgs const& args, int k)
                        {
                            CUTE_NO_UNROLL
                            for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx, ++g)
                            {
                                bool const first = block_idx == args.start_block_idx;
                                bool const last = block_idx == args.end_block_idx - 1;
                                int const pb = g % 2;
                                int const kb = g % NUM_KV_BUFS;
                                plan.bar_so_ready[pb].wait((g / 2) & 1);
                                tcgen05AfterThreadSync();
                                Tensor sS = make_tensor(make_smem_ptr(plan.s[pb].data()), SmemLayoutS{});
                                bf16* v_base = plan.kv[kb].data();
                                Tensor sV_lo = make_tensor(
                                    make_smem_ptr(v_base), SmemLayoutKTilesTransposed_SW128<D_V / 64 / 2>{});
                                Tensor sV_hi = make_tensor(make_smem_ptr(v_base + B_TOPK * (D_V / 2)),
                                    SmemLayoutKTilesTransposed_SW128<D_V / 64 / 2>{});
                                if (first && k > 0)
                                {
                                    plan.bar_o_free_lo.wait((k - 1) & 1);
                                    tcgen05AfterThreadSync();
                                }
                                utcmmaSs(tiled_mma_O, sS, sV_lo, tO_lo, first);
                                ummaArriveNoElect(plan.bar_sv_done_lo[kb]);
                                // Let QK^T of the next block (on the critical softmax path) run before PV-hi of this
                                // one.
                                if (g + 1 < total_blocks)
                                {
                                    plan.bar_qkhi_issued[(g + 1) % 2].wait((g >> 1) & 1);
                                }
                                if (first && k > 0)
                                {
                                    plan.bar_o_free_hi.wait((k - 1) & 1);
                                    tcgen05AfterThreadSync();
                                }
                                utcmmaSs(tiled_mma_O, sS, sV_hi, tO_hi, first);
                                ummaArriveNoElect(plan.bar_sv_done[kb]);
                                if (last)
                                {
                                    ummaArriveNoElect(plan.bar_o_full);
                                }
                            }
                        });
                }
            }
        }
        else if (warp_idx == 5)
        {
            cudaGridDependencySynchronize();
            // ======================================= Raw packed-row producer ===================================
            // TMA issue throughput is per issuing thread (~230 cycles per gather4 from one thread), so each 64-row
            // block is issued by 16 lanes (one gather4 each). The raw rows land in the head of the block's K/V buffer.
            int g = 0;
            for_each_item(
                [&](ItemArgs const& args, int k)
                {
                    CUTE_NO_UNROLL
                    for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx, ++g)
                    {
                        bool const is_swa = block_idx < args.num_orig_kv_blocks;
                        int const ib = g % NUM_INDEX_BUFS;
                        int const kb = g % NUM_KV_BUFS;
                        plan.bar_index_ready[ib].wait((g / NUM_INDEX_BUFS) & 1);
                        // PV of the buffer's previous block must be done reading the rows the gather overwrites (main
                        // raw rows stay in channels [0, 256)); buffer 1 first stages the first token's BF16 Q.
                        if (g >= NUM_KV_BUFS)
                        {
                            int const phase = ((g - NUM_KV_BUFS) / NUM_KV_BUFS) & 1;
                            if (is_swa)
                            {
                                plan.bar_sv_done[kb].wait(phase);
                            }
                            else
                            {
                                plan.bar_sv_done_lo[kb].wait(phase);
                            }
                        }
                        else if (g == 1)
                        {
                            plan.bar_qstage_free.wait(0);
                        }
                        uint8_t* const raw = reinterpret_cast<uint8_t*>(plan.kv[kb].data());
                        if (lane_idx < 16)
                        {
                            int4 const coords = *(int4*) (plan.tma_coord[ib] + 4 * lane_idx);
                            if (is_swa)
                            {
                                tmaGather4(&tma_params.tensor_map_swa, plan.bar_raw_ready[kb],
                                    raw + SWA_GROUP_PITCH * lane_idx, 0, coords,
                                    (int64_t) TMA::CacheHintSm90::EVICT_LAST);
                            }
                            else
                            {
                                tmaGather4(&tma_params.tensor_map_main, plan.bar_raw_ready[kb],
                                    raw + MAIN_GROUP_PITCH * lane_idx, 0, coords,
                                    (int64_t) TMA::CacheHintSm90::EVICT_LAST);
                            }
                        }
                        __syncwarp();
                        if (lane_idx == 0)
                        {
                            plan.bar_raw_ready[kb].arrive_and_expect_tx(
                                B_TOPK * (is_swa ? kSwaRowBytes : kMainRowBytes));
                        }
                        if (lane_idx < 16)
                        {
                            plan.bar_index_free[ib].arrive(); // each lane that read coordinates releases them
                        }
                    }
                });
        }
        else if (warp_idx == 7)
        {
            cudaGridDependencySynchronize();
            // ======================================= Index warp: coordinates and validity masks =========
            bool const logical_main = params.main_page_table != nullptr;
            int64_t const page_mask = (int64_t{1} << params.main_page_shift) - 1;
            int g = 0;
            for_each_item(
                [&](ItemArgs const& args, int k)
                {
                    // Logical main selections: this query's page-table row and visible bound, hoisted per token.
                    int64_t map_request = 0;
                    int64_t map_limit = 0;
                    if (logical_main && args.main_length > 0)
                    {
                        map_request = __ldg(params.token_requests + args.token_idx);
                        map_limit = min(__ldg(params.token_visible + args.token_idx), params.main_max_positions);
                        if (map_request < 0 || map_request >= params.main_page_table_rows
                            || params.main_page_table_cols <= 0)
                        {
                            map_limit = 0;
                        }
                    }
                    int32_t const* map_row = params.main_page_table + map_request * params.main_page_table_stride;

                    CUTE_NO_UNROLL
                    for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx, ++g)
                    {
                        bool const is_swa = block_idx < args.num_orig_kv_blocks;
                        int const ib = g % NUM_INDEX_BUFS;
                        int64_t const rows = is_swa ? params.swa_rows : params.main_rows;
                        int const length = is_swa ? args.swa_length : args.main_length;
                        int abs_pos;
                        int my_indices[2];
                        if (is_swa)
                        {
                            abs_pos = block_idx * B_TOPK + lane_idx * 2;
                            int64_t const* ptr = params.swa_indices + params.stride_swa_idx * args.token_idx + abs_pos;
                            longlong2 v = abs_pos < length ? __ldg((longlong2 const*) ptr) : make_longlong2(-1, -1);
                            my_indices[0] = (v.x < 0 || v.x >= rows) ? -1 : (int) v.x;
                            my_indices[1] = (v.y < 0 || v.y >= rows) ? -1 : (int) v.y;
                        }
                        else
                        {
                            abs_pos = (block_idx - args.num_orig_kv_blocks) * B_TOPK + lane_idx * 2;
                            int const* ptr = params.main_indices + params.stride_main_idx * args.token_idx + abs_pos;
                            int2 v = abs_pos < length ? __ldg((int2 const*) ptr) : make_int2(-1, -1);
                            my_indices[0] = v.x;
                            my_indices[1] = v.y;
                            if (logical_main)
                            {
                                // Both page-table reads are issued before either result is used.
                                int physical[2];
                                CUTE_UNROLL
                                for (int i = 0; i < 2; ++i)
                                {
                                    int64_t const logical = my_indices[i];
                                    int64_t const page = logical >> params.main_page_shift;
                                    bool const mapped
                                        = logical >= 0 && logical < map_limit && page < params.main_page_table_cols;
                                    physical[i] = mapped ? __ldg(map_row + page) : -1;
                                }
                                CUTE_UNROLL
                                for (int i = 0; i < 2; ++i)
                                {
                                    int64_t const slot = (int64_t{physical[i]} << params.main_page_shift)
                                        | (int64_t{my_indices[i]} & page_mask);
                                    my_indices[i] = (physical[i] >= 0 && slot < rows) ? (int) slot : -1;
                                }
                            }
                        }

                        int tma_coords[2];
                        char valid_mask = 0;
                        CUTE_UNROLL
                        for (int i = 0; i < 2; ++i)
                        {
                            bool const is_token_valid
                                = my_indices[i] >= 0 && my_indices[i] < rows && (abs_pos + i < length);
                            valid_mask |= is_token_valid << i;
                            tma_coords[i] = is_token_valid ? my_indices[i] : -1;
                        }
                        valid_mask <<= lane_idx % 4 * 2;
                        valid_mask |= __shfl_xor_sync(0xFFFFFFFF, valid_mask, 0x1);
                        valid_mask |= __shfl_xor_sync(0xFFFFFFFF, valid_mask, 0x2);
                        plan.bar_index_free[ib].wait(((g / NUM_INDEX_BUFS) & 1) ^ 1);
                        *(int2*) (plan.tma_coord[ib] + lane_idx * 2) = *(int2*) tma_coords;
                        if (lane_idx % 4 == 0)
                        {
                            plan.is_token_valid[ib][lane_idx / 4] = valid_mask;
                        }
                        plan.bar_index_ready[ib].arrive();
                    }
                });
        }
    }
    else if (warpgroup_idx == 2)
    {
        // ======================================= Dequant warpgroup =======================================
        cutlass::arch::warpgroup_reg_dealloc<DEQUANT_REGS>();

        // Main rows (e2m1): thread (group = t / 8, q = t % 8) converts rows group + 16 i (i < 4), 8-channel chunk q of
        // each 64-channel column tile; a warp's 4-byte loads cover 4 rows x 32 B and its 16-byte stores 4 full rows.
        // SWA rows (e4m3): thread (w = t / 32, r2 = lane / 16, j = lane % 16) converts rows 16 w + 2 s + r2 (s < 8),
        // channels [16 j, 16 j + 16) of each half; a warp's 16-byte loads cover 2 rows x 256 B. Both mappings keep
        // the shared-memory loads and the SW128 stores at the minimum number of wavefronts.
        constexpr int NUM_GROUPS = 16;
        int const group_idx = idx_in_warpgroup / 8, idx_in_group = idx_in_warpgroup % 8;
        Tensor kv0 = make_tensor(make_smem_ptr(plan.kv[0].data()), SmemLayoutKTiles_SW128<D_K / 64>{});
        uint32_t const kv0_base = cute::cast_smem_ptr_to_uint(&kv0(group_idx, idx_in_group * 8));
        uint32_t const kv_raw0 = cute::cast_smem_ptr_to_uint(plan.kv[0].data());
        uint32_t const kv_buf_stride
            = cute::cast_smem_ptr_to_uint(plan.kv[1].data()) - cute::cast_smem_ptr_to_uint(plan.kv[0].data());
        uint32_t const main_q = idx_in_group >> 1;
        int const swa_w = idx_in_warpgroup / 32, swa_r2 = (idx_in_warpgroup / 16) % 2, swa_j = idx_in_warpgroup % 16;
        uint32_t const swa_b = (swa_j / 2) % 4;

        int g = 0;
        for_each_item(
            [&](ItemArgs const& args, int k)
            {
                CUTE_NO_UNROLL
                for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx, ++g)
                {
                    bool const is_swa = block_idx < args.num_orig_kv_blocks;
                    int const kb = g % NUM_KV_BUFS;
                    plan.bar_raw_ready[kb].wait((g / NUM_KV_BUFS) & 1);
                    uint32_t const dst_main = kv0_base + kb * kv_buf_stride;
                    uint32_t const dst_buf = kv_raw0 + kb * kv_buf_stride;
                    uint8_t const* raw0 = reinterpret_cast<uint8_t const*>(plan.kv[kb].data());

                    // The bf16 rows overwrite the raw rows in place: every thread reads all of its raw bytes and scales
                    // (both channel halves) into registers before the warpgroup writes anything back.
                    auto begin_half = [&](int khalf)
                    {
                        if (khalf == 0)
                        {
                            plan.bar_sv_done_lo[kb].wait(((g / NUM_KV_BUFS) & 1) ^ 1);
                            if (g == 1)
                            {
                                plan.bar_qstage_free.wait(0); // KV buffer 1 staged the first token's Q
                            }
                        }
                        else
                        {
                            plan.bar_sv_done[kb].wait(((g / NUM_KV_BUFS) & 1) ^ 1);
                        }
                    };
                    auto end_half = [&](int khalf)
                    {
                        cutlass::arch::fence_view_async_shared();
                        if (khalf == 0)
                        {
                            plan.bar_kv_ready_lo[kb].arrive();
                        }
                        else
                        {
                            plan.bar_kv_ready[kb].arrive();
                        }
                    };
                    if (is_swa)
                    {
                        uint4 v[2][8];
                        uint32_t sb[2]
                                   [2]; // the thread's scale byte of row sidx in byte sidx % 4 of sb[khalf][sidx / 4]
                        CUTE_UNROLL
                        for (int khalf = 0; khalf < 2; ++khalf)
                        {
                            CUTE_UNROLL
                            for (int sidx = 0; sidx < 8; ++sidx)
                            {
                                int const row = 16 * swa_w + 2 * sidx + swa_r2;
                                uint8_t const* rowp = raw0 + (row >> 2) * SWA_GROUP_PITCH + (row & 3) * kSwaRowBytes;
                                v[khalf][sidx] = *(uint4 const*) (rowp + 256 * khalf + 16 * swa_j);
                                uint2 const sc = *(uint2 const*) (rowp + 512 + 8 * khalf);
                                uint32_t const byte = __byte_perm(swa_j >= 8 ? sc.y : sc.x, 0, swa_b | 0x4440);
                                sb[khalf][sidx / 4]
                                    = sidx % 4 == 0 ? byte : sb[khalf][sidx / 4] | (byte << (8 * (sidx % 4)));
                            }
                        }
                        NamedBarrier::arrive_and_wait(128, NamedBarriers::dequant_sync);
                        CUTE_UNROLL
                        for (int khalf = 0; khalf < 2; ++khalf)
                        {
                            begin_half(khalf);
                            CUTE_UNROLL
                            for (int sidx = 0; sidx < 8; ++sidx)
                            {
                                int const row = 16 * swa_w + 2 * sidx + swa_r2;
                                uint4 const x = v[khalf][sidx];
                                uint32_t const sc2 = ue8m0x2ToBf16x2(
                                    __byte_perm(sb[khalf][sidx / 4], 0, (sidx % 4) | ((sidx % 4) << 4)));
                                uint4 o0, o1;
                                o0.x = bf16x2Mul(e4m3x2ToBf16x2(x.x & 0xFFFF), sc2);
                                o0.y = bf16x2Mul(e4m3x2ToBf16x2(x.x >> 16), sc2);
                                o0.z = bf16x2Mul(e4m3x2ToBf16x2(x.y & 0xFFFF), sc2);
                                o0.w = bf16x2Mul(e4m3x2ToBf16x2(x.y >> 16), sc2);
                                o1.x = bf16x2Mul(e4m3x2ToBf16x2(x.z & 0xFFFF), sc2);
                                o1.y = bf16x2Mul(e4m3x2ToBf16x2(x.z >> 16), sc2);
                                o1.z = bf16x2Mul(e4m3x2ToBf16x2(x.w & 0xFFFF), sc2);
                                o1.w = bf16x2Mul(e4m3x2ToBf16x2(x.w >> 16), sc2);
                                uint32_t const tile = dst_buf + (4 * khalf + swa_j / 4) * (B_TOPK * 128) + row * 128;
                                int const c0 = 2 * (swa_j % 4);
                                stSharedV4(tile + (((c0) ^ (row & 7)) << 4), o0);
                                stSharedV4(tile + (((c0 + 1) ^ (row & 7)) << 4), o1);
                            }
                            end_half(khalf);
                        }
                    }
                    else
                    {
                        uint32_t v[2][4][4];
                        uint32_t sb[2][4]; // byte cc: the thread's scale of channel tile 4 khalf + cc of row i
                        CUTE_UNROLL
                        for (int i = 0; i < 4; ++i)
                        {
                            int const row_idx = i * NUM_GROUPS + group_idx;
                            uint8_t const* row
                                = raw0 + (row_idx >> 2) * MAIN_GROUP_PITCH + (row_idx & 3) * kMainRowBytes;
                            CUTE_UNROLL
                            for (int khalf = 0; khalf < 2; ++khalf)
                            {
                                uint4 const sc = *(uint4 const*) (row + 256 + 16 * khalf);
                                // channels [c*64 + q*8, +8), c = 4*khalf + cc: scale byte 4 cc + q / 2 of the half
                                sb[khalf][i] = __byte_perm(__byte_perm(sc.x, sc.y, main_q | ((main_q + 4) << 4)),
                                    __byte_perm(sc.z, sc.w, main_q | ((main_q + 4) << 4)), 0x5410);
                                CUTE_UNROLL
                                for (int cc = 0; cc < 4; ++cc)
                                {
                                    v[khalf][i][cc]
                                        = *(uint32_t const*) (row + (4 * khalf + cc) * 32 + idx_in_group * 4);
                                }
                            }
                        }
                        NamedBarrier::arrive_and_wait(128, NamedBarriers::dequant_sync);
                        CUTE_UNROLL
                        for (int khalf = 0; khalf < 2; ++khalf)
                        {
                            begin_half(khalf);
                            CUTE_UNROLL
                            for (int i = 0; i < 4; ++i)
                            {
                                CUTE_UNROLL
                                for (int cc = 0; cc < 4; ++cc)
                                {
                                    uint32_t hh[4];
                                    e2m1x8ToBf16x2x4(v[khalf][i][cc], hh);
                                    uint32_t const sc2 = e4m3x2ToBf16x2(__byte_perm(sb[khalf][i], 0, cc | (cc << 4)));
                                    uint4 o;
                                    o.x = bf16x2Mul(hh[0], sc2);
                                    o.y = bf16x2Mul(hh[1], sc2);
                                    o.z = bf16x2Mul(hh[2], sc2);
                                    o.w = bf16x2Mul(hh[3], sc2);
                                    stSharedV4(
                                        dst_main + 2 * (i * NUM_GROUPS * 64 + (4 * khalf + cc) * B_TOPK * 64), o);
                                }
                            }
                            end_half(khalf);
                        }
                    }
                }
            });
    }
    else
    {
        // ======================================= Q loader + epilogue warpgroup ===============================
        cutlass::arch::warpgroup_reg_alloc<IO_REGS>();
        cudaGridDependencySynchronize();
        int const t = idx_in_warpgroup;
        int const head = t % 64;
        int const half = t / 64;     // TMEM lanes 64..127 hold channels +64 (Q) / +128 (O) of the same head
        int const wq = (t / 32) % 2; // this warp's 32 heads: [32 * wq, 32 * wq + 32)
        int const hrow = t % 32;     // row within this warp's staging tile
        uint32_t const epi_base = cute::cast_smem_ptr_to_uint(plan.epi.data());
        if (params.out_fp8 != nullptr)
        {
            // The scale rows numTokens .. pad128(numTokens) - 1 are zeroed by all CTAs together.
            int const pad = (params.num_tokens + 127) / 128 * 128 - params.num_tokens;
            for (int i = blockIdx.y * 128 + t; i < kOutputGroups * pad * 32; i += gridDim.y * 128)
            {
                int const group = i / (pad * 32);
                int const row = params.num_tokens + i / 32 % pad;
                *reinterpret_cast<uint32_t*>(params.out_scale + scaleWordOffset(group, row, i % 32, params.num_tokens))
                    = 0;
            }
        }

        // Q of a later token (of every token when Q is E4M3), this thread's head and 64-channel half of each
        // 128-channel tile into registers as BF16. E4M3: 4 rounds of one 64x128 SW128 TMA box into the staging buffer
        // (8 KB), read conflict-free under the swizzle and converted exactly. BF16: 16-byte global loads, one tile at a
        // time (each tile's addresses depend on the previous tile's data, so at most 16 KB of loads per CTA are in
        // flight). The registers are written to TMEM once the previous token's QK^T is done.
        uint32_t qreg[128];
        int qload_ctr = 0;
        uint32_t const q_zero = static_cast<uint32_t>(params.num_tokens) >> 31; // 0, unknown to the compiler
        auto load_q = [&](int token_idx)
        {
            if (!params.q_fp8)
            {
                bf16 const* const src = static_cast<bf16 const*>(params.q) + token_idx * params.stride_q_tok
                    + head * params.stride_q_head + 64 * half;
                uint32_t dep = 0;
                CUTE_UNROLL
                for (int tile = 0; tile < 4; ++tile)
                {
                    CUTE_UNROLL
                    for (int c = 0; c < 8; ++c)
                    {
                        uint4 const v = __ldg(reinterpret_cast<uint4 const*>(src + tile * 128 + c * 8 + dep));
                        qreg[tile * 32 + c * 4 + 0] = v.x;
                        qreg[tile * 32 + c * 4 + 1] = v.y;
                        qreg[tile * 32 + c * 4 + 2] = v.z;
                        qreg[tile * 32 + c * 4 + 3] = v.w;
                    }
                    dep = (qreg[tile * 32] ^ qreg[tile * 32 + 31]) & q_zero;
                }
                return;
            }
            if (lane_idx == 0)
            {
                bulkWaitGroupRead0(); // this warp's epilogue stores have finished reading the staging buffer
            }
            NamedBarrier::arrive_and_wait(128, NamedBarriers::wg3_sync);
            CUTE_UNROLL
            for (int tile = 0; tile < 4; ++tile)
            {
                if (t == 0)
                {
                    cute::SM90_TMA_LOAD_3D::copy(&tma_params.tensor_map_q, (uint64_t*) &plan.bar_qload,
                        (uint64_t) TMA::CacheHintSm90::EVICT_FIRST, plan.epi.data(), tile * 128, 0, token_idx);
                    plan.bar_qload.arrive_and_expect_tx(64 * 128);
                }
                plan.bar_qload.wait(qload_ctr & 1);
                ++qload_ctr;
                uint32_t const row_base = epi_base + head * 128;
                CUTE_UNROLL
                for (int c = 0; c < 4; ++c)
                {
                    uint4 v;
                    asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n"
                                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                                 : "r"(row_base + (((4 * half + c) ^ (head % 8)) << 4)));
                    uint32_t const words[4] = {v.x, v.y, v.z, v.w};
                    CUTE_UNROLL
                    for (int i = 0; i < 4; ++i)
                    {
                        qreg[tile * 32 + c * 8 + 2 * i] = e4m3x2ToBf16x2(words[i] & 0xFFFF);
                        qreg[tile * 32 + c * 8 + 2 * i + 1] = e4m3x2ToBf16x2(words[i] >> 16);
                    }
                }
                // The next round's TMA (async proxy) rewrites the staging buffer: order these generic reads before it.
                fenceProxyAsyncSmem();
                NamedBarrier::arrive_and_wait(128, NamedBarriers::wg3_sync);
            }
        };
        auto store_q = [&]()
        {
            CUTE_UNROLL
            for (int tile = 0; tile < 4; ++tile)
            {
                tmemStore32dp32b32x(tmem_cols::Q + tile * 32, qreg + tile * 32);
            }
            cutlass::arch::fence_view_async_tmem_store();
            tcgen05BeforeThreadSync();
            plan.bar_q_ready.arrive();
        };

        // Per-warp epilogue staging: one tile of 32 rows x 64 B, SW64. Each warp TMA-stores its own 32 heads, so no
        // warpgroup-wide barriers are needed; bulk groups are tracked by lane 0.
        uint32_t const my_stage = epi_base + (t / 32) * EPI_WARP_TILE_BYTES;
        int const swz = (hrow >> 1) & 3;
        auto stage_and_store = [&](uint4 const(&chunk)[4], bool to_out, int c0, int c2)
        {
            if (lane_idx == 0)
            {
                bulkWaitGroupRead0(); // the previous store has finished reading the tile
            }
            __syncwarp();
            uint32_t const tile_addr = my_stage;
            CUTE_UNROLL
            for (int i = 0; i < 4; ++i)
            {
                stSharedV4(tile_addr + hrow * 64 + ((i ^ swz) << 4), chunk[i]);
            }
            fenceProxyAsyncSmem();
            __syncwarp();
            if (lane_idx == 0)
            {
                void const* tmap = to_out ? &tma_params.tensor_map_out : &tma_params.tensor_map_oaccum;
                SM90_TMA_STORE_3D::copy(tmap, plan.epi.data() + (my_stage - epi_base), c0, 32 * wq, c2);
                bulkCommitGroup();
            }
        };

        // MXFP8 output of a whole token: this thread's 128 channels of one half of O (TMEM columns 128 * hpart, o) in
        // 4 blocks of 32. The owners of the head's last 64 channels (hpart 1, half 1, blocks 2 and 3) rotate them back
        // first.
        int const group = head / kHeadsPerGroup;
        auto store_fp8 = [&](int token, int hpart, uint32_t const(&o)[128], float const(&o_max)[4], float2 o_scale2)
        {
            uint8_t* const dst = params.out_fp8
                + ((int64_t) group * params.num_tokens + token) * (kHeadsPerGroup * kHeadDim)
                + (head % kHeadsPerGroup) * kHeadDim + 256 * hpart + 128 * half;
            float const* const cos_sin = params.rope_cos_sin + (int64_t) __ldg(params.positions + token) * 64;
            uint32_t exponents = 0;
            CUTE_UNROLL
            for (int j = 0; j < 4; ++j)
            {
                float2 x[16];
                CUTE_UNROLL
                for (int e = 0; e < 16; ++e)
                {
                    float2 const v = {__uint_as_float(o[j * 32 + 2 * e]), __uint_as_float(o[j * 32 + 2 * e + 1])};
                    x[e] = bf16PairToFloat2(float2ToBf16x2(float2Mul(v, o_scale2)));
                }
                if (hpart == 1 && half == 1 && j >= 2)
                {
                    CUTE_UNROLL
                    for (int e = 0; e < 16; e += 4)
                    {
                        // Channel pair 448 + 2i, 449 + 2i takes cos/sin i.
                        float4 const c = __ldg(reinterpret_cast<float4 const*>(cos_sin + (j - 2) * 16 + e));
                        float4 const sn = __ldg(reinterpret_cast<float4 const*>(cos_sin + 32 + (j - 2) * 16 + e));
                        x[e] = inverseRopePair(x[e], c.x, sn.x);
                        x[e + 1] = inverseRopePair(x[e + 1], c.y, sn.y);
                        x[e + 2] = inverseRopePair(x[e + 2], c.z, sn.z);
                        x[e + 3] = inverseRopePair(x[e + 3], c.w, sn.w);
                    }
                }
                // BF16 rounding is monotonic in the magnitude, so the block's largest |x| is the BF16 of the largest
                // |o| times the scale (on SM103 the loads' hardware maximum); rotated blocks and the SM100-family
                // build take the maximum here.
                float amax = 0.F;
                if (kTmemLoadMax && !(hpart == 1 && half == 1 && j >= 2))
                {
                    amax = __bfloat162float(__float2bfloat16_rn(o_max[j] * o_scale2.x));
                }
                else
                {
                    CUTE_UNROLL
                    for (int e = 0; e < 16; ++e)
                    {
                        amax = fmax3f(amax, fabsf(x[e].x), fabsf(x[e].y));
                    }
                }
                float factor;
                exponents |= mxfp8Exponent(amax, factor) << (8 * j);
                float2 const factor2 = {factor, factor};
                uint32_t q[8];
                CUTE_UNROLL
                for (int i = 0; i < 8; ++i)
                {
                    float2 a, b; // x * factor with the scalar fast-math multiply's flushing, two at a time
                    asm("mul.ftz.f32x2 %0, %1, %2;\n"
                        : "=l"(reinterpret_cast<uint64_t&>(a))
                        : "l"(reinterpret_cast<uint64_t const&>(x[2 * i])),
                        "l"(reinterpret_cast<uint64_t const&>(factor2)));
                    asm("mul.ftz.f32x2 %0, %1, %2;\n"
                        : "=l"(reinterpret_cast<uint64_t&>(b))
                        : "l"(reinterpret_cast<uint64_t const&>(x[2 * i + 1])),
                        "l"(reinterpret_cast<uint64_t const&>(factor2)));
                    q[i] = toE4m3x2(a.x, a.y) | (toE4m3x2(b.x, b.y) << 16);
                }
                asm volatile("st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" ::"l"(dst + 32 * j), "r"(q[0]),
                             "r"(q[1]), "r"(q[2]), "r"(q[3]), "r"(q[4]), "r"(q[5]), "r"(q[6]), "r"(q[7])
                             : "memory");
            }
            *reinterpret_cast<uint32_t*>(params.out_scale
                + scaleWordOffset(group, token, (head % kHeadsPerGroup) * 4 + 2 * hpart + half, params.num_tokens))
                = exponents;
        };

        // MXFP8 epilogue of a whole token. One copy of the code serves both halves of O; unrolled, it would crowd
        // the instruction cache of the other roles. The CTA's last item frees no O half (nothing waits for it).
        auto epilogue_fp8 = [&](ItemArgs const& args, int k, bool last)
        {
            int const sb = k % 2;
            plan.bar_stats_ready[sb].wait((k / 2) & 1);
            float const o_scale = plan.stats_scale[sb][head];
            plan.bar_stats_free[sb].arrive();
            float2 const o_scale2 = {o_scale, o_scale};
            plan.bar_o_full.wait(k & 1);
            tcgen05AfterThreadSync();
#pragma unroll 1
            for (int hpart = 0; hpart < 2; ++hpart)
            {
                uint32_t o[128];
                float o_max[4];
                CUTE_UNROLL
                for (int j = 0; j < 4; ++j)
                {
                    if constexpr (kTmemLoadMax)
                    {
                        tmemLoad32dp32b32xMaxAbs(tmem_cols::O + hpart * 128 + j * 32, o + j * 32, o_max[j]);
                    }
                    else
                    {
                        tmemLoad32dp32b32x(tmem_cols::O + hpart * 128 + j * 32, o + j * 32);
                    }
                }
                cutlass::arch::fence_view_async_tmem_load();
                if (!last)
                {
                    tcgen05BeforeThreadSync();
                    (hpart == 0 ? plan.bar_o_free_lo : plan.bar_o_free_hi).arrive();
                }
                store_fp8(args.token_idx, hpart, o, o_max, o_scale2);
            }
        };

        auto epilogue = [&](ItemArgs const& args, int k)
        {
            int const sb = k % 2;
            plan.bar_stats_ready[sb].wait((k / 2) & 1);
            float const o_scale = plan.stats_scale[sb][head];
            float const lse = plan.stats_lse[sb][head];
            plan.bar_stats_free[sb].arrive();
            float2 const o_scale2 = {o_scale, o_scale};
            if (args.is_split && t < B_H)
            {
                params.lse_accum[(int64_t) args.slot * B_H + t] = lse;
            }
            plan.bar_o_full.wait(k & 1);
            tcgen05AfterThreadSync();
            CUTE_UNROLL
            for (int hpart = 0; hpart < 2; ++hpart)
            {
                // TMEM cols [128*hpart, +128): channels 256*hpart + 128*half + [0, 128)
                uint32_t o[128];
                CUTE_UNROLL
                for (int j = 0; j < 4; ++j)
                {
                    tmemLoad32dp32b32x(tmem_cols::O + hpart * 128 + j * 32, o + j * 32);
                }
                cutlass::arch::fence_view_async_tmem_load();
                tcgen05BeforeThreadSync();
                if (hpart == 0)
                {
                    plan.bar_o_free_lo.arrive();
                }
                else
                {
                    plan.bar_o_free_hi.arrive();
                }
                int const ch0 = 256 * hpart + 128 * half; // channel base of this thread's 128 values
                if (!args.is_split)
                {
                    CUTE_UNROLL
                    for (int j = 0; j < 4; ++j)
                    {
                        uint4 chunk[4];
                        uint32_t* cw = (uint32_t*) chunk;
                        CUTE_UNROLL
                        for (int e = 0; e < 16; ++e)
                        {
                            float2 v = {__uint_as_float(o[j * 32 + 2 * e]), __uint_as_float(o[j * 32 + 2 * e + 1])};
                            cw[e] = float2ToBf16x2(float2Mul(v, o_scale2));
                        }
                        stage_and_store(chunk, true, ch0 + 32 * j, args.token_idx);
                    }
                }
                else
                {
                    CUTE_UNROLL
                    for (int j = 0; j < 8; ++j)
                    {
                        uint4 chunk[4];
                        float* cf = (float*) chunk;
                        CUTE_UNROLL
                        for (int e = 0; e < 16; e += 2)
                        {
                            float2 v = {__uint_as_float(o[j * 16 + e]), __uint_as_float(o[j * 16 + e + 1])};
                            *(float2*) (cf + e) = float2Mul(v, o_scale2);
                        }
                        stage_and_store(chunk, false, ch0 + 16 * j, args.slot);
                    }
                }
            }
        };

        // Publishes this CTA's part of a split token (partial output and LSE) device-wide, then counts it for the
        // combine kernel, which may already be waiting for it (it runs alongside this kernel's tail under PDL).
        auto publish = [&](ItemArgs const& a)
        {
            if (lane_idx == 0)
            {
                bulkWaitGroup0(); // this warp's partial-output stores are complete
                asm volatile("fence.proxy.async.global;" ::: "memory");
            }
            __syncwarp();
            __threadfence();
            NamedBarrier::arrive_and_wait(128, NamedBarriers::wg3_sync);
            if (t == 0)
            {
                asm volatile(
                    "red.release.gpu.global.add.s32 [%0], 1;" ::"l"(&csa2SplitArrivals[a.token_idx - full_tokens][0])
                    : "memory");
            }
        };

        // Epilogue of the CTA's last item: every MMA has completed, so the KV buffers are free and each warp stages its
        // whole 32-head slice (bf16: 4 boxes of 64 channels, fp32: 8 boxes of 32 channels, SW128) for few, unblocked
        // TMA stores.
        auto epilogue_last = [&](ItemArgs const& args, int k)
        {
            int const sb = k % 2;
            plan.bar_stats_ready[sb].wait((k / 2) & 1);
            float const o_scale = plan.stats_scale[sb][head];
            float const lse = plan.stats_lse[sb][head];
            plan.bar_stats_free[sb].arrive();
            float2 const o_scale2 = {o_scale, o_scale};
            if (args.is_split && t < B_H)
            {
                params.lse_accum[(int64_t) args.slot * B_H + t] = lse;
            }
            plan.bar_o_full.wait(k & 1);
            tcgen05AfterThreadSync();
            uint32_t const stage = cute::cast_smem_ptr_to_uint(plan.kv[0].data()) + (t / 32) * 32768;
            bf16* const stage_ptr = plan.kv[0].data() + (t / 32) * 16384;
            int const rsw = hrow % 8;
            CUTE_UNROLL
            for (int hpart = 0; hpart < 2; ++hpart)
            {
                uint32_t o[128];
                CUTE_UNROLL
                for (int j = 0; j < 4; ++j)
                {
                    tmemLoad32dp32b32x(tmem_cols::O + hpart * 128 + j * 32, o + j * 32);
                }
                cutlass::arch::fence_view_async_tmem_load();
                if (!args.is_split)
                {
                    // 128 channels = 2 boxes of 64; box b2 of this hpart at stage + (2 hpart + b2) * 4 KB
                    CUTE_UNROLL
                    for (int b2 = 0; b2 < 2; ++b2)
                    {
                        uint32_t const box = stage + (2 * hpart + b2) * 4096 + hrow * 128;
                        CUTE_UNROLL
                        for (int c = 0; c < 8; ++c)
                        {
                            uint32_t w[4];
                            CUTE_UNROLL
                            for (int e = 0; e < 4; ++e)
                            {
                                int const i = b2 * 64 + c * 8 + 2 * e;
                                float2 v = {__uint_as_float(o[i]), __uint_as_float(o[i + 1])};
                                w[e] = float2ToBf16x2(float2Mul(v, o_scale2));
                            }
                            stSharedV4(box + ((c ^ rsw) << 4), make_uint4(w[0], w[1], w[2], w[3]));
                        }
                    }
                }
                else
                {
                    // 128 channels = 4 boxes of 32 fp32; box b4 of this hpart at stage + (4 hpart + b4) * 4 KB
                    CUTE_UNROLL
                    for (int b4 = 0; b4 < 4; ++b4)
                    {
                        uint32_t const box = stage + (4 * hpart + b4) * 4096 + hrow * 128;
                        CUTE_UNROLL
                        for (int c = 0; c < 8; ++c)
                        {
                            int const i = b4 * 32 + c * 4;
                            float2 v0 = float2Mul({__uint_as_float(o[i]), __uint_as_float(o[i + 1])}, o_scale2);
                            float2 v1 = float2Mul({__uint_as_float(o[i + 2]), __uint_as_float(o[i + 3])}, o_scale2);
                            stSharedV4(box + ((c ^ rsw) << 4),
                                make_uint4(__float_as_uint(v0.x), __float_as_uint(v0.y), __float_as_uint(v1.x),
                                    __float_as_uint(v1.y)));
                        }
                    }
                }
            }
            fenceProxyAsyncSmem();
            __syncwarp();
            if (lane_idx == 0)
            {
                if (!args.is_split)
                {
                    CUTE_UNROLL
                    for (int bx = 0; bx < 4; ++bx)
                    {
                        // box bx: hpart = bx / 2, channels 256 (bx / 2) + 128 half + 64 (bx % 2)
                        SM90_TMA_STORE_3D::copy(&tma_params.tensor_map_out_wide, stage_ptr + bx * 2048,
                            256 * (bx / 2) + 128 * half + 64 * (bx % 2), 32 * wq, args.token_idx);
                    }
                }
                else
                {
                    CUTE_UNROLL
                    for (int bx = 0; bx < 8; ++bx)
                    {
                        SM90_TMA_STORE_3D::copy(&tma_params.tensor_map_oaccum_wide, stage_ptr + bx * 2048,
                            256 * (bx / 4) + 128 * half + 32 * (bx % 4), 32 * wq, args.slot);
                    }
                }
                bulkCommitGroup();
            }
        };

        if (has_work)
        {
            // Q of item 0 is staged by the MMA warp (TMA + UTCCP) unless it is E4M3; later items go through the
            // staging buffer.
            ItemArgs prev = get_item(0);
            if (params.q_fp8)
            {
                load_q(prev.token_idx);
                store_q();
            }
#pragma unroll 1
            for (int k = 1; k < num_items; ++k)
            {
                ItemArgs const cur = get_item(k);
                load_q(cur.token_idx);
                plan.bar_q_free.wait((k - 1) & 1);
                tcgen05AfterThreadSync();
                store_q();
                if (!prev.is_split && params.out_fp8 != nullptr)
                {
                    epilogue_fp8(prev, k - 1, false);
                }
                else
                {
                    epilogue(prev, k - 1);
                }
                if (prev.is_split)
                {
                    publish(prev);
                }
                prev = cur;
            }
            if (!prev.is_split && params.out_fp8 != nullptr)
            {
                epilogue_fp8(prev, num_items - 1, true);
            }
            else
            {
                epilogue_last(prev, num_items - 1);
            }
            if (prev.is_split)
            {
                publish(prev);
            }
            if (lane_idx == 0)
            {
                bulkWaitGroupRead0(); // the staged tiles must stay valid until the stores have read them
            }
        }
    }

    __syncthreads();
    if (warp_idx == 0)
    {
        cute::TMEM::Allocator1Sm().free(0, 512);
    }
#else
    asm volatile("trap;");
#endif
}

__global__ void __launch_bounds__(Csa2Kernel::NUM_THREADS, 1, 1) csa2SparseDecodeKernel(
    __grid_constant__ KernelParams const params, __grid_constant__ Csa2Kernel::TmaParams const tma_params)
{
    Csa2Kernel::devfunc(params, tma_params);
}

// ---------------------------------------------------------------------------------------------------------------
// Split-KV combine: one CTA per (remainder token, kCombineHeadsPerCta heads), one warp per head at a time; lane owns
// channels i * 128 + lane * 4 + [0, 4) for i = 0..3, so every load / store instruction of a warp is one contiguous
// segment. out = sum_s w_s O_s with w_s = 2^(lse_s - L) / (sum_s 2^(lse_s - L) + 2^(sink*log2e - L)); fixed order.
// The splits of a token are the contiguous CTAs p_first..p_last. A CTA p > p_first starts its chunk inside this
// token, so the token is its first remainder item (slot 2p); p_first uses slot 2p_first + first_odd.
// With PDL this kernel starts once every attention CTA is running (they trigger at their start), so it may wait on the
// per-token arrival counters instead of the whole attention grid: a token is combined as soon as its parts are out.
// ---------------------------------------------------------------------------------------------------------------

__global__ void __launch_bounds__(256, 4) csa2SparseDecodeCombineKernel(__grid_constant__ KernelParams const params)
{
#if defined(CSA2_SPARSE_DECODE_ARCH_SUPPORTED)
    int const swa_blocks = max((params.swa_topk + 63) / 64, 1);
    int const nb = swa_blocks + (params.main_topk + 63) / 64;
    int const full_tokens = params.full_per_cta * params.num_ctas;
    WorkSplit const ws{nb, swa_blocks, full_tokens, (params.num_tokens - full_tokens) * nb, max(params.rem_ctas, 1),
        params.magic_rem_ctas, params.magic_rem_blocks};
    int const r = blockIdx.x;
    int const tok = full_tokens + r;
    int const tb = r * nb;
    int const p_first = ws.rem_owner(tb);
    int const nsplit = ws.rem_owner(tb + nb - 1) - p_first + 1;
    if (nsplit <= 1)
    {
        return; // written directly by the attention kernel
    }
    int const first_odd = ws.rem_begin(p_first) != tb ? 1 : 0;
    int const lane = threadIdx.x % 32;
    // MXFP8 output: cos/sin of the RoPE channel pairs 2 * lane - 32 and next that lanes 16..31 hold, read ahead.
    float2 rope_cos{}, rope_sin{};
    if (params.out_fp8 != nullptr && lane >= 16)
    {
        float const* const cos_sin = params.rope_cos_sin + (int64_t) __ldg(params.positions + tok) * 64;
        rope_cos = __ldg(reinterpret_cast<float2 const*>(cos_sin + 2 * lane - 32));
        rope_sin = __ldg(reinterpret_cast<float2 const*>(cos_sin + 2 * lane));
    }

    if (params.pdl && threadIdx.x == 0)
    {
        // Wait until every part of the token is published (acquire), then let the CTA read the partials. Without PDL
        // the attention kernel has completed before this kernel starts.
        int* const arrived = &csa2SplitArrivals[r][0];
        int v;
        while (true)
        {
            asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(arrived) : "memory");
            if (v >= nsplit)
            {
                break;
            }
            __nanosleep(64);
        }
    }
    __syncthreads();

    constexpr int MAXS = kMaxCombineSplits;
    for (int h = blockIdx.y * kCombineHeadsPerCta + threadIdx.x / 32; h < (blockIdx.y + 1) * kCombineHeadsPerCta;
         h += 8)
    {
        float w[MAXS];
#pragma unroll
        for (int s = 0; s < MAXS; ++s)
        {
            int const slot = 2 * (p_first + s) + (s == 0 ? first_odd : 0);
            w[s] = s < nsplit ? __ldcg(params.lse_accum + (int64_t) slot * kNumHeads + h) : -CUDART_INF_F;
        }
        float lmax = -CUDART_INF_F;
#pragma unroll
        for (int s = 0; s < MAXS; ++s)
        {
            lmax = fmaxf(lmax, w[s]);
        }
        float total = 0.0f;
#pragma unroll
        for (int s = 0; s < MAXS; ++s)
        {
            w[s] = lmax == -CUDART_INF_F ? 0.0f : exp2f(w[s] - lmax);
            total += w[s];
        }
        if (lmax != -CUDART_INF_F)
        {
            total += exp2f(params.attn_sink[h] * CUDART_L2E_F - lmax);
        }
        float const inv = lmax == -CUDART_INF_F ? 0.0f : __fdividef(1.0f, total);

        float4 acc[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            acc[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
#pragma unroll
        for (int s0 = 0; s0 < MAXS; s0 += 4)
        {
            if (s0 < nsplit)
            {
                float4 v[4][4];
#pragma unroll
                for (int j = 0; j < 4; ++j)
                {
                    int const slot = 2 * (p_first + s0 + j) + (s0 + j == 0 ? first_odd : 0);
                    float4 const* src
                        = (float4 const*) (params.o_accum + ((int64_t) slot * kNumHeads + h) * kHeadDim) + lane;
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                    {
                        v[j][i] = (s0 + j < nsplit) ? __ldcg(src + i * 32) : make_float4(0.f, 0.f, 0.f, 0.f);
                    }
                }
#pragma unroll
                for (int j = 0; j < 4; ++j)
                {
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                    {
                        acc[i].x = fmaf(w[s0 + j], v[j][i].x, acc[i].x);
                        acc[i].y = fmaf(w[s0 + j], v[j][i].y, acc[i].y);
                        acc[i].z = fmaf(w[s0 + j], v[j][i].z, acc[i].z);
                        acc[i].w = fmaf(w[s0 + j], v[j][i].w, acc[i].w);
                    }
                }
            }
        }
        if (params.out_fp8 != nullptr)
        {
            // Lanes 8b .. 8b + 7 hold 32-channel block 4i + b of the head; the head's last 64 channels (i = 3, lanes
            // 16..31) are rotated back first.
            int const group = h / kHeadsPerGroup;
            uint8_t* const dst = params.out_fp8
                + ((int64_t) group * params.num_tokens + tok) * (kHeadsPerGroup * kHeadDim)
                + (h % kHeadsPerGroup) * kHeadDim + lane * 4;
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                float2 a = bf16PairToFloat2(float2ToBf16x2({acc[i].x * inv, acc[i].y * inv}));
                float2 b = bf16PairToFloat2(float2ToBf16x2({acc[i].z * inv, acc[i].w * inv}));
                if (i == 3 && lane >= 16)
                {
                    a = inverseRopePair(a, rope_cos.x, rope_sin.x);
                    b = inverseRopePair(b, rope_cos.y, rope_sin.y);
                }
                float amax = fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(b.x), fabsf(b.y)));
#pragma unroll
                for (int m = 1; m < 8; m *= 2)
                {
                    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFU, amax, m));
                }
                float factor;
                uint32_t const exponent = mxfp8Exponent(amax, factor);
                *reinterpret_cast<uint32_t*>(dst + i * 128)
                    = toE4m3x2(a.x * factor, a.y * factor) | (toE4m3x2(b.x * factor, b.y * factor) << 16);
                uint32_t const word = __shfl_sync(0xFFFFFFFFU, exponent, 0)
                    | (__shfl_sync(0xFFFFFFFFU, exponent, 8) << 8) | (__shfl_sync(0xFFFFFFFFU, exponent, 16) << 16)
                    | (__shfl_sync(0xFFFFFFFFU, exponent, 24) << 24);
                if (lane == 0)
                {
                    *reinterpret_cast<uint32_t*>(
                        params.out_scale + scaleWordOffset(group, tok, (h % kHeadsPerGroup) * 4 + i, params.num_tokens))
                        = word;
                }
            }
        }
        else
        {
            bf16* dst
                = params.out + (int64_t) tok * params.stride_o_tok + (int64_t) h * params.stride_o_head + lane * 4;
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                *(uint2*) (dst + i * 128) = make_uint2(
                    float2ToBf16x2({acc[i].x * inv, acc[i].y * inv}), float2ToBf16x2({acc[i].z * inv, acc[i].w * inv}));
            }
        }
    }

    if (!params.pdl)
    {
        // Nobody else reads the token's counter in this launch: reset it for the next one.
        if (blockIdx.y == 0 && threadIdx.x == 0)
        {
            csa2SplitArrivals[r][0] = 0;
        }
        return;
    }
    // The last CTA of the token resets its counters for the next launch (all of the token's CTAs passed the wait).
    __syncthreads();
    // Complete only after the attention kernel has (its whole tokens are written), so that a kernel that depends on
    // this one through PDL observes the complete output.
    cudaGridDependencySynchronize();
    if (threadIdx.x == 0)
    {
        int const done = atomicAdd(&csa2SplitArrivals[r][1], 1);
        if (done == kNumHeads / kCombineHeadsPerCta - 1)
        {
            csa2SplitArrivals[r][0] = 0;
            csa2SplitArrivals[r][1] = 0;
        }
    }
#else
    asm volatile("trap;");
#endif
}

} // namespace kernels::csa2SparseDecode

TRTLLM_NAMESPACE_END
