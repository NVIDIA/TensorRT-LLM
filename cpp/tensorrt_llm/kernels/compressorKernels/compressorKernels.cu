/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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

// ============================================================================
// Compressor Kernels — DeepSeek-V4 KV Cache Compression
// ============================================================================
//
// This file implements CUDA kernels for KV cache compression in the DeepSeek-V4
// sparse attention system. The compressor reduces sequences of input tokens
// into fewer compressed tokens via learned weighted averaging (online softmax),
// then post-processes and scatters results into a paged KV cache.
//
// Three kernels are provided:
//
//   1. pagedKvCompressKernel  — Decode path (single/few new tokens per batch).
//      Loads prior compressor state from paged memory, performs online softmax
//      with the new token(s), writes updated state back, and emits a compressed
//      output token when compress_ratio tokens have been accumulated.
//
//   2. prefillReductionKernel — Prefill path (many tokens per batch).
//      Processes full chunks of compress_ratio tokens in one shot via online
//      softmax reduction over the input sequence. Also saves compressor state
//      for any remainder tokens that don't form a complete chunk.
//
//   3. postProcessScatterKernel — Fused post-processing + paged cache write.
//      Takes compressed output tokens and applies: RMSNorm → RoPE → Hadamard
//      transform → optional V4-Pro QDQ → scatter to paged KV cache. Supports
//      default, FP8, and V4-Pro MXFP8/MXFP4 QDQ cache modes.
//      Keeps all intermediate values in float32 registers to avoid extra DRAM
//      round-trips.
//
// Vectorization strategy:
//   All kernels use 128-bit vectorized loads/stores (float4 / 8×bf16).
//   VEC = number of elements per thread, chosen so that NTHRD = HEAD_DIM/VEC >= 32.
//   For HEAD_DIM=128, bf16: VEC=4, NTHRD=32.  For HEAD_DIM=512, bf16: VEC=8, NTHRD=64.
//
// Overlap mode (compress_ratio=4):
//   When enabled, state_dim = 2*head_dim and the compressor uses overlapping
//   windows: each compressed output is derived from both the previous and current
//   chunk of compress_ratio tokens (previous chunk → first head_dim features,
//   current chunk → second head_dim features). This doubles the state stored
//   per position but improves compression quality.
//
// Template parameters:
//   HEAD_DIM            — Head dimension (128 or 512)
//   KV_SCORE_ELEM_BYTES — kv_score element size (2=bf16, 4=fp32)
//   STATE_ELEM_BYTES    — compressor state element size (2=bf16, 4=fp32)
//   SCALE_TYPE          — Output cache scale/dtype for postProcessScatterKernel
// ============================================================================

#include "tensorrt_llm/kernels/compressorKernels/compressorKernels.h"

#include "tensorrt_llm/common/assert.h"
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <type_traits>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::compressor
{

// ============================================================================
// Helper functions
// ============================================================================

// Full-warp butterfly reductions via __shfl_xor_sync (all 32 lanes participate).
__device__ inline float warpReduceSum(float val)
{
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    return val;
}

__device__ inline float warpReduceMax(float val)
{
    for (int mask = 16; mask > 0; mask >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, mask));
    return val;
}

// Runtime-dispatched element load (bf16 or fp32 → float). Used in the decode
// kernel where elem_bytes is a runtime parameter from paged state buffers.
__device__ inline float loadAsFloat(void const* base, int64_t offset, int elem_bytes)
{
    if (elem_bytes == 2)
        return __bfloat162float(reinterpret_cast<__nv_bfloat16 const*>(base)[offset]);
    else
        return reinterpret_cast<float const*>(base)[offset];
}

__device__ inline void storeFromFloat(void* base, int64_t offset, float val, int elem_bytes)
{
    if (elem_bytes == 2)
        reinterpret_cast<__nv_bfloat16*>(base)[offset] = __float2bfloat16_rn(val);
    else
        reinterpret_cast<float*>(base)[offset] = val;
}

// Bit-hack ceil(log2(x)) for x>0: equivalent to V4 reference fast_log2_ceil.
__device__ inline int fastLog2Ceil(float x)
{
    uint32_t const bits = __float_as_uint(x);
    int const exp_part = static_cast<int>((bits >> 23) & 0xFFu) - 127;
    uint32_t const man_bits = bits & 0x007FFFFFu;
    return exp_part + (man_bits != 0u ? 1 : 0);
}

// Bit-hack 2^n for integer n: equivalent to V4 reference fast_pow2.
__device__ inline float fastPow2(int n)
{
    uint32_t const bits = static_cast<uint32_t>(n + 127) << 23;
    return __uint_as_float(bits);
}

// V4-Pro fast_round_scale: 2^ceil(log2(amax * max_value_inv)).  Operand order
// (multiplication vs `amax / max_value`) matches V4 byte-for-byte; using fp32
// bit hacks avoids log2f/exp2f rounding so the resulting power-of-2 is exact.
__device__ inline float roundedPow2Scale(float amax, float max_value_inv, float min_amax)
{
    float const clamped_amax = fmaxf(amax, min_amax);
    return fastPow2(fastLog2Ceil(clamped_amax * max_value_inv));
}

// Hardware FP4 (e2m1) round-trip via __nv_fp4_e2m1 (cuda_fp4.h).  The
// constructor uses the SM100 PTX `cvt.rn.satfinite.e2m1x2.f32` cast
// (round-to-nearest-even with finite saturation), matching V4-Pro's
// Cast(FP4) byte-for-byte.  Verified against the V4-Pro reference LUT
// (e2m1 levels {0, 0.5, 1, 1.5, 2, 3, 4, 6} with midpoint thresholds
// {0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0}) over 1.1M sweep inputs covering
// every exact level, every midpoint ±2 ULPs, out-of-range, subnormals, and
// dense random fp32 -- zero mismatches.  The Python test reference still
// uses the explicit LUT so the kernel is checked against an independent
// software model rather than the HW cast itself.
__device__ inline uint8_t toUe8m0(float val)
{
    __nv_fp8_e8m0 out;
    out.__x = __nv_cvt_float_to_e8m0(val, __NV_SATFINITE, cudaRoundPosInf);
    return out.__x;
}

__device__ inline uint8_t packE2M1x2(float lo, float hi)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    uint32_t val;
    asm volatile(
        "{\n"
        ".reg .b8 byte0;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"
        "mov.b32 %0, {byte0, byte0, byte0, byte0};\n"
        "}"
        : "=r"(val)
        : "f"(lo), "f"(hi));
    return static_cast<uint8_t>(val);
#else
    return 0;
#endif
}

// Vectorized load/store types: maps byte-width to CUDA vector type.
template <int V>
struct VecType;

template <>
struct VecType<4>
{
    using type = unsigned int;
}; //  32-bit: 2 bf16 or 1 fp32

template <>
struct VecType<8>
{
    using type = uint2;
}; //  64-bit: 4 bf16 or 2 fp32

template <>
struct VecType<16>
{
    using type = uint4;
}; // 128-bit: 8 bf16 or 4 fp32

// Cache scale / pre-store quantization type for postProcessScatterKernel.
// The store dtype is implied: kNone keeps the input dtype (elem_bytes
// controls bf16 vs fp32), kFP8* writes one byte per element, and
// kMXFP4Blockwise writes packed FP4 (two values per byte) plus per-32
// UE8M0 scale bytes.
enum class CacheScaleType
{
    kNone = 0,
    kFP8PerTensor = 1,  // FP8 E4M3 with static scale=1.0
    kFP8Blockwise = 2,  // FP8 E4M3 with per-128-element fp32 scales
    kMXFP4Blockwise = 3 // packed FP4 E2M1 with per-32 UE8M0 scales
};

// ============================================================================
// Decode Kernel: pagedKvCompressKernel
//
// Template: <HEAD_DIM, KV_SCORE_ELEM_BYTES, STATE_ELEM_BYTES, NEXT_N>
//   NEXT_N: number of new tokens per sequence in this decode step (1-4)
//
// Grid:  (batch_size) — one block per batch element
// Block: (NTHRD) where NTHRD = HEAD_DIM / VEC (>= 32 threads)
//
// Algorithm per batch element:
//   For each new token in the decode step:
//     1. Load existing compressor state (partial kv/score) from paged cache
//     2. Perform online softmax: accumulate new token's contribution using
//        the numerically stable running max + weighted sum formulation
//     3. Write updated state back to paged cache
//     4. If compress_ratio tokens accumulated → emit compressed output,
//        reset state for next compression window
//
// Each thread handles VEC contiguous elements of head_dim. In overlap mode
// (state_dim = 2*head_dim), Phase 1 iterates over 2 column halves.
//
// Memory layout:
//   kv_score:   [total_tokens, 2 * state_dim] — interleaved KV and score projections
//   paged_kv:   paged cache for compressor KV state
//   paged_score: paged cache for compressor score state (with APE bias)
//   output:     [total_comp_tokens, head_dim] — compressed output tokens
// ============================================================================

// Helper: vectorized online softmax step reading from paged KV/score state.
// Loads one position's KV and score from paged memory and updates the running
// online softmax accumulators (rmax, rsum, rwsum) per element.
// APE is already baked into paged_score (added during Phase 1), so no APE
// addition is performed here.
template <int HEAD_DIM, int STATE_ELEM_BYTES, int VEC>
__device__ __forceinline__ void decodeSoftmaxVec(void const* __restrict__ paged_kv_raw,
    void const* __restrict__ paged_score_raw,
    int64_t page_sd, // page_size * state_dim (in elements)
    int state_dim,
    int phys_kv,     // physical page index for kv
    int phys_sc,     // physical page index for score
    int blk_off,     // offset within page
    int kv_col_off,  // column offset (0 or HEAD_DIM)
    int tid, float* __restrict__ rmax, float* __restrict__ rsum, float* __restrict__ rwsum)
{
    using StateElemT = typename std::conditional<STATE_ELEM_BYTES == 2, __nv_bfloat16, float>::type;
    using StateVecT = typename VecType<VEC * STATE_ELEM_BYTES>::type;

    auto const* kv = reinterpret_cast<StateElemT const*>(paged_kv_raw);
    auto const* sc = reinterpret_cast<StateElemT const*>(paged_score_raw);

    int64_t base_kv = static_cast<int64_t>(phys_kv) * page_sd + blk_off * state_dim + kv_col_off;
    int64_t base_sc = static_cast<int64_t>(phys_sc) * page_sd + blk_off * state_dim + kv_col_off;

    StateVecT k_raw = reinterpret_cast<StateVecT const*>(&kv[base_kv])[tid];
    StateVecT s_raw = reinterpret_cast<StateVecT const*>(&sc[base_sc])[tid];
    StateElemT const* ke = reinterpret_cast<StateElemT const*>(&k_raw);
    StateElemT const* se = reinterpret_cast<StateElemT const*>(&s_raw);

#pragma unroll
    for (int i = 0; i < VEC; i += 4)
    {
        float kf[4] = {static_cast<float>(ke[i]), static_cast<float>(ke[i + 1]), static_cast<float>(ke[i + 2]),
            static_cast<float>(ke[i + 3])};
        // score already includes APE (added during Phase 1 store)
        float sf[4] = {static_cast<float>(se[i]), static_cast<float>(se[i + 1]), static_cast<float>(se[i + 2]),
            static_cast<float>(se[i + 3])};
        // Online softmax: maintain running (max, sum_exp, weighted_sum) per element.
        // nm = new max, sc_f = rescale factor for old accumulators, tm = exp(score - new_max).
        // Final output: rwsum / rsum = weighted average of KV values.
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            float nm = fmaxf(rmax[i + j], sf[j]);
            float sc_f = expf(rmax[i + j] - nm);
            float tm = expf(sf[j] - nm);
            rsum[i + j] = rsum[i + j] * sc_f + tm;
            rwsum[i + j] = rwsum[i + j] * sc_f + kf[j] * tm;
            rmax[i + j] = nm;
        }
    }
}

template <int HEAD_DIM, int KV_SCORE_ELEM_BYTES, int STATE_ELEM_BYTES, int COMPRESS_RATIO, int NEXT_N,
    int NUM_RED_WARPS = 1>
__global__ void pagedKvCompressKernel(void const* __restrict__ kv_score_raw, float const* __restrict__ ape,
    void* __restrict__ paged_kv_raw, void* __restrict__ paged_score_raw, int32_t const* __restrict__ block_table_kv,
    int32_t const* __restrict__ block_table_score, void* __restrict__ output_raw, int32_t const* __restrict__ kv_lens,
    int32_t const* __restrict__ cu_seq_lens, int32_t const* __restrict__ cu_kv_comp, int page_size, int max_blocks,
    int out_elem_bytes)
{
    using KvScoreElemT = typename std::conditional<KV_SCORE_ELEM_BYTES == 2, __nv_bfloat16, float>::type;
    using StateElemT = typename std::conditional<STATE_ELEM_BYTES == 2, __nv_bfloat16, float>::type;
    // DeepSeek-V4 model configures compress_ratio to be 4 or 128.
    static_assert(COMPRESS_RATIO == 4 || COMPRESS_RATIO == 128, "Unsupported COMPRESS_RATIO");
    constexpr bool IS_OVERLAP = (COMPRESS_RATIO == 4);
    constexpr int ELEM_BYTES_FOR_VEC
        = (KV_SCORE_ELEM_BYTES > STATE_ELEM_BYTES) ? KV_SCORE_ELEM_BYTES : STATE_ELEM_BYTES;
    constexpr int MAX_VEC = 16 / ELEM_BYTES_FOR_VEC;
    constexpr int VEC = (HEAD_DIM / MAX_VEC >= 32) ? MAX_VEC : (HEAD_DIM / 32);
    using KvScoreVecT = typename VecType<VEC * KV_SCORE_ELEM_BYTES>::type;
    using StateVecT = typename VecType<VEC * STATE_ELEM_BYTES>::type;
    static_assert(VEC >= 4, "VEC must be >= 4 for float4 ape loads");

    // HEAD_BLOCKS: split head_dim across blockIdx.y for better SM utilisation.
    // For HD=512 and max elem size 2: NTHRD_BASE=64 → HEAD_BLOCKS=2, NTHRD_INNER=32.
    // For HD=128 and max elem size 2/4: NTHRD_BASE=32 → HEAD_BLOCKS=1, NTHRD_INNER=32.
    // For HD=512 and max elem size 4: NTHRD_BASE=128 → HEAD_BLOCKS=4, NTHRD_INNER=32.
    constexpr int NTHRD_BASE = HEAD_DIM / VEC;
    constexpr int HEAD_BLOCKS = (NTHRD_BASE > 32) ? (NTHRD_BASE / 32) : 1;
    constexpr int NTHRD_INNER = NTHRD_BASE / HEAD_BLOCKS; // always <= 32
    // ELEM_PER_BLOCK: head_dim elements handled by one blockIdx.y block.
    // = NTHRD_INNER * VEC = HEAD_DIM / HEAD_BLOCKS.
    // Used for the multi-warp shared-memory merge layout so that each block
    // only allocates storage for its own head slice, not the full HEAD_DIM.
    constexpr int ELEM_PER_BLOCK = NTHRD_INNER * VEC;

    // state_dim is fully determined by template parameters.
    constexpr int STATE_DIM = IS_OVERLAP ? 2 * HEAD_DIM : HEAD_DIM;
    constexpr int64_t TWO_SD = 2 * STATE_DIM;
    constexpr int COFF = IS_OVERLAP ? 2 : 1;

    int const tid = threadIdx.x % NTHRD_INNER;
    int const warp_id = threadIdx.x / NTHRD_INNER; // 0..NUM_RED_WARPS-1
    int const batch_idx = blockIdx.x;
    int const head_blk = blockIdx.y;
    int const eff_tid = head_blk * NTHRD_INNER + tid;

    int const kv_len = kv_lens[batch_idx];
    int const sp = kv_len - NEXT_N;
    int const in_off = cu_seq_lens[batch_idx];
    int const out_off = cu_kv_comp[batch_idx];
    int64_t const page_sd = static_cast<int64_t>(page_size) * STATE_DIM;

    auto const* kv_score = reinterpret_cast<KvScoreElemT const*>(kv_score_raw);
    auto* paged_kv = reinterpret_cast<StateElemT*>(paged_kv_raw);
    auto* paged_score = reinterpret_cast<StateElemT*>(paged_score_raw);

    // ================================================================
    // Phase 1: Write NEXT_N new tokens' KV and score state to paged cache.
    //
    // Only warp 0 participates (all warps share the same eff_tid mapping).
    // When NUM_RED_WARPS == 1, the guard compiles away.
    // ================================================================
    if (warp_id == 0)
    {
#pragma unroll
        for (int t = 0; t < NEXT_N; t++)
        {
            int token_idx = sp + t;
            if (token_idx < kv_len)
            {
                int ape_idx = token_idx % COMPRESS_RATIO;
                int log_blk = token_idx / page_size;
                int blk_off = token_idx % page_size;
                int phys_kv = block_table_kv[batch_idx * max_blocks + log_blk];
                int phys_sc = block_table_score[batch_idx * max_blocks + log_blk];

                for (int col_idx = 0; col_idx < COFF; col_idx++)
                {
                    int const col = col_idx * HEAD_DIM;
                    int64_t const src = static_cast<int64_t>(in_off + t) * TWO_SD + col;
                    int64_t const dkv = static_cast<int64_t>(phys_kv) * page_sd + blk_off * STATE_DIM + col;
                    int64_t const dsc = static_cast<int64_t>(phys_sc) * page_sd + blk_off * STATE_DIM + col;

                    KvScoreVecT kv_raw = reinterpret_cast<KvScoreVecT const*>(&kv_score[src])[eff_tid];
                    KvScoreVecT sc_raw = reinterpret_cast<KvScoreVecT const*>(&kv_score[src + STATE_DIM])[eff_tid];

                    KvScoreElemT const* kv_e = reinterpret_cast<KvScoreElemT const*>(&kv_raw);
                    StateVecT kv_out;
                    StateElemT* kv_o = reinterpret_cast<StateElemT*>(&kv_out);
#pragma unroll
                    for (int i = 0; i < VEC; i++)
                    {
                        kv_o[i] = static_cast<StateElemT>(static_cast<float>(kv_e[i]));
                    }
                    reinterpret_cast<StateVecT*>(&paged_kv[dkv])[eff_tid] = kv_out;

                    KvScoreElemT const* sc_e = reinterpret_cast<KvScoreElemT const*>(&sc_raw);
                    StateVecT sc_out;
                    StateElemT* sc_o = reinterpret_cast<StateElemT*>(&sc_out);
#pragma unroll
                    for (int i = 0; i < VEC; i += 4)
                    {
                        float4 av
                            = *reinterpret_cast<float4 const*>(&ape[ape_idx * STATE_DIM + col + eff_tid * VEC + i]);
                        sc_o[i] = static_cast<StateElemT>(static_cast<float>(sc_e[i]) + av.x);
                        sc_o[i + 1] = static_cast<StateElemT>(static_cast<float>(sc_e[i + 1]) + av.y);
                        sc_o[i + 2] = static_cast<StateElemT>(static_cast<float>(sc_e[i + 2]) + av.z);
                        sc_o[i + 3] = static_cast<StateElemT>(static_cast<float>(sc_e[i + 3]) + av.w);
                    }
                    reinterpret_cast<StateVecT*>(&paged_score[dsc])[eff_tid] = sc_out;
                }
            }
        }
    }

    if constexpr (NUM_RED_WARPS > 1)
    {
        __syncthreads();
    }

    // ================================================================
    // Phase 2: Count how many complete compression windows finished.
    // ================================================================
    int last_token_idx = sp + NEXT_N - 1;
    int num_compressions = (last_token_idx + 1) / COMPRESS_RATIO - sp / COMPRESS_RATIO;

    // ================================================================
    // Phase 3: Online softmax reduction over each complete chunk.
    //
    // When NUM_RED_WARPS > 1, the compress_ratio positions are split
    // across warps. Each warp reduces its partition independently, then
    // partial (rmax, rsum, rwsum) accumulators are merged via shared
    // memory using the log-sum-exp identity.
    // ================================================================
    for (int c = 0; c < NEXT_N; c++)
    {
        if (c >= num_compressions)
            break;

        int compress_idx = sp / COMPRESS_RATIO + c;
        int curr_chunk_start = compress_idx * COMPRESS_RATIO;

        float rmax[VEC], rsum[VEC], rwsum[VEC];
#pragma unroll
        for (int i = 0; i < VEC; i++)
        {
            rmax[i] = -INFINITY;
            rsum[i] = 0.0f;
            rwsum[i] = 0.0f;
        }

        constexpr int positions_per_warp = COMPRESS_RATIO / NUM_RED_WARPS;
        int const my_r_start = warp_id * positions_per_warp;
        int const my_r_end = (warp_id == NUM_RED_WARPS - 1) ? COMPRESS_RATIO : (my_r_start + positions_per_warp);

        if constexpr (IS_OVERLAP)
        {
            int prev_start = curr_chunk_start - COMPRESS_RATIO;
            if (prev_start >= 0)
            {
                if (page_size >= COMPRESS_RATIO)
                {
                    int log_blk_prev = prev_start / page_size;
                    int phys_kv_prev = block_table_kv[batch_idx * max_blocks + log_blk_prev];
                    int phys_sc_prev = block_table_score[batch_idx * max_blocks + log_blk_prev];
                    int chunk_off_prev = prev_start % page_size;
                    for (int r = my_r_start; r < my_r_end; r++)
                    {
                        decodeSoftmaxVec<HEAD_DIM, STATE_ELEM_BYTES, VEC>(paged_kv_raw, paged_score_raw, page_sd,
                            STATE_DIM, phys_kv_prev, phys_sc_prev, chunk_off_prev + r, 0, eff_tid, rmax, rsum, rwsum);
                    }
                }
                else
                {
                    for (int r = my_r_start; r < my_r_end; r++)
                    {
                        int pos = prev_start + r;
                        int log_blk = pos / page_size;
                        int blk_off = pos % page_size;
                        int phys_kv = block_table_kv[batch_idx * max_blocks + log_blk];
                        int phys_sc = block_table_score[batch_idx * max_blocks + log_blk];
                        decodeSoftmaxVec<HEAD_DIM, STATE_ELEM_BYTES, VEC>(paged_kv_raw, paged_score_raw, page_sd,
                            STATE_DIM, phys_kv, phys_sc, blk_off, 0, eff_tid, rmax, rsum, rwsum);
                    }
                }
            }

            if (page_size >= COMPRESS_RATIO)
            {
                int log_blk_cur = curr_chunk_start / page_size;
                int phys_kv_cur = block_table_kv[batch_idx * max_blocks + log_blk_cur];
                int phys_sc_cur = block_table_score[batch_idx * max_blocks + log_blk_cur];
                int chunk_off_cur = curr_chunk_start % page_size;
                for (int r = my_r_start; r < my_r_end; r++)
                {
                    decodeSoftmaxVec<HEAD_DIM, STATE_ELEM_BYTES, VEC>(paged_kv_raw, paged_score_raw, page_sd, STATE_DIM,
                        phys_kv_cur, phys_sc_cur, chunk_off_cur + r, HEAD_DIM, eff_tid, rmax, rsum, rwsum);
                }
            }
            else
            {
                for (int r = my_r_start; r < my_r_end; r++)
                {
                    int pos = curr_chunk_start + r;
                    int log_blk = pos / page_size;
                    int blk_off = pos % page_size;
                    int phys_kv = block_table_kv[batch_idx * max_blocks + log_blk];
                    int phys_sc = block_table_score[batch_idx * max_blocks + log_blk];
                    decodeSoftmaxVec<HEAD_DIM, STATE_ELEM_BYTES, VEC>(paged_kv_raw, paged_score_raw, page_sd, STATE_DIM,
                        phys_kv, phys_sc, blk_off, HEAD_DIM, eff_tid, rmax, rsum, rwsum);
                }
            }
        }
        else
        {
            if (page_size >= COMPRESS_RATIO)
            {
                int log_blk = curr_chunk_start / page_size;
                int phys_kv = block_table_kv[batch_idx * max_blocks + log_blk];
                int phys_sc = block_table_score[batch_idx * max_blocks + log_blk];
                int chunk_off = curr_chunk_start % page_size;
                for (int r = my_r_start; r < my_r_end; r++)
                {
                    decodeSoftmaxVec<HEAD_DIM, STATE_ELEM_BYTES, VEC>(paged_kv_raw, paged_score_raw, page_sd, STATE_DIM,
                        phys_kv, phys_sc, chunk_off + r, 0, eff_tid, rmax, rsum, rwsum);
                }
            }
            else
            {
                for (int r = my_r_start; r < my_r_end; r++)
                {
                    int pos = curr_chunk_start + r;
                    int log_blk = pos / page_size;
                    int blk_off = pos % page_size;
                    int phys_kv = block_table_kv[batch_idx * max_blocks + log_blk];
                    int phys_sc = block_table_score[batch_idx * max_blocks + log_blk];
                    decodeSoftmaxVec<HEAD_DIM, STATE_ELEM_BYTES, VEC>(paged_kv_raw, paged_score_raw, page_sd, STATE_DIM,
                        phys_kv, phys_sc, blk_off, 0, eff_tid, rmax, rsum, rwsum);
                }
            }
        }

        // Multi-warp merge epilogue (compiled away when NUM_RED_WARPS == 1).
        if constexpr (NUM_RED_WARPS > 1)
        {
            // Shared-memory layout: [NUM_RED_WARPS * ELEM_PER_BLOCK] per array.
            // ELEM_PER_BLOCK = NTHRD_INNER * VEC = HEAD_DIM / HEAD_BLOCKS, i.e.
            // the number of head_dim elements covered by this block (blockIdx.y).
            // Using ELEM_PER_BLOCK (not HEAD_DIM) avoids 4x over-allocation when
            // HEAD_BLOCKS > 1 (e.g. HD=512 fp32 has HEAD_BLOCKS=4).
            extern __shared__ float smem[];
            float* s_rmax = smem;
            float* s_rsum = s_rmax + NUM_RED_WARPS * ELEM_PER_BLOCK;
            float* s_rwsum = s_rsum + NUM_RED_WARPS * ELEM_PER_BLOCK;

            // local_elem: index within this block's head slice [0, ELEM_PER_BLOCK).
            // = tid * VEC + i   (tid = threadIdx.x % NTHRD_INNER, same as eff_tid - head_blk*NTHRD_INNER)
#pragma unroll
            for (int i = 0; i < VEC; i++)
            {
                int const local_elem = tid * VEC + i;
                s_rmax[warp_id * ELEM_PER_BLOCK + local_elem] = rmax[i];
                s_rsum[warp_id * ELEM_PER_BLOCK + local_elem] = rsum[i];
                s_rwsum[warp_id * ELEM_PER_BLOCK + local_elem] = rwsum[i];
            }
            __syncthreads();

            if (warp_id == 0)
            {
                for (int w = 1; w < NUM_RED_WARPS; w++)
                {
#pragma unroll
                    for (int i = 0; i < VEC; i++)
                    {
                        int const local_elem = tid * VEC + i;
                        float const m2 = s_rmax[w * ELEM_PER_BLOCK + local_elem];
                        float const s2 = s_rsum[w * ELEM_PER_BLOCK + local_elem];
                        float const ws2 = s_rwsum[w * ELEM_PER_BLOCK + local_elem];

                        float const nm = fmaxf(rmax[i], m2);
                        float const sc1 = expf(rmax[i] - nm);
                        float const sc2 = expf(m2 - nm);
                        rsum[i] = rsum[i] * sc1 + s2 * sc2;
                        rwsum[i] = rwsum[i] * sc1 + ws2 * sc2;
                        rmax[i] = nm;
                    }
                }
            }
            __syncthreads();
        }

        bool const should_write = (NUM_RED_WARPS == 1) || (warp_id == 0);
        if (should_write)
        {
            int64_t const out_base = static_cast<int64_t>(out_off + c) * HEAD_DIM + eff_tid * VEC;
            if (out_elem_bytes == 2)
            {
                __nv_bfloat16 packed[VEC];
#pragma unroll
                for (int i = 0; i < VEC; i++)
                    packed[i] = __float2bfloat16_rn(rwsum[i] / rsum[i]);
                using OutVecT = typename VecType<VEC * 2>::type;
                *reinterpret_cast<OutVecT*>(&reinterpret_cast<__nv_bfloat16*>(output_raw)[out_base])
                    = *reinterpret_cast<OutVecT const*>(packed);
            }
            else
            {
                float result[VEC];
#pragma unroll
                for (int i = 0; i < VEC; i++)
                    result[i] = rwsum[i] / rsum[i];
#pragma unroll
                for (int i = 0; i < VEC; i += 4)
                    *reinterpret_cast<float4*>(&reinterpret_cast<float*>(output_raw)[out_base + i])
                        = *reinterpret_cast<float4 const*>(&result[i]);
            }
        }
    }
}

// Explicit instantiations for decode kernel (NUM_RED_WARPS defaults to 1)
#define INST_DECODE(HD, KV_EB, STATE_EB, CR, NN, NRW)                                                                  \
    template __global__ void pagedKvCompressKernel<HD, KV_EB, STATE_EB, CR, NN, NRW>(void const*, float const*, void*, \
        void*, int32_t const*, int32_t const*, void*, int32_t const*, int32_t const*, int32_t const*, int, int, int);

#define INST_DECODE_NN(HD, KV_EB, STATE_EB, CR)                                                                        \
    INST_DECODE(HD, KV_EB, STATE_EB, CR, 1, 1)                                                                         \
    INST_DECODE(HD, KV_EB, STATE_EB, CR, 2, 1)                                                                         \
    INST_DECODE(HD, KV_EB, STATE_EB, CR, 3, 1) INST_DECODE(HD, KV_EB, STATE_EB, CR, 4, 1)

#define INST_DECODE_DTYPES(HD, CR)                                                                                     \
    INST_DECODE_NN(HD, 2, 2, CR)                                                                                       \
    INST_DECODE_NN(HD, 2, 4, CR) INST_DECODE_NN(HD, 4, 2, CR) INST_DECODE_NN(HD, 4, 4, CR)

INST_DECODE_DTYPES(128, 4)
INST_DECODE_DTYPES(128, 128)
INST_DECODE_DTYPES(512, 4)
INST_DECODE_DTYPES(512, 128)

// 4-warp parallel reduction variants for large compress_ratio.
// HD=128: ELEM_PER_BLOCK=128, smem = 3*4*128*4 = 6 KB.
// HD=512 bf16: ELEM_PER_BLOCK=256, smem = 3*4*256*4 = 12 KB.
// HD=512 fp32: ELEM_PER_BLOCK=128, smem = 3*4*128*4 = 6 KB.
// NEXT_N=1 (single-token decode) and NEXT_N=2 (MTP speculative decode).
INST_DECODE(128, 2, 2, 128, 1, 4)
INST_DECODE(128, 2, 4, 128, 1, 4)
INST_DECODE(128, 4, 2, 128, 1, 4)
INST_DECODE(128, 4, 4, 128, 1, 4)
INST_DECODE(128, 2, 2, 128, 2, 4)
INST_DECODE(128, 2, 4, 128, 2, 4)
INST_DECODE(128, 4, 2, 128, 2, 4)
INST_DECODE(128, 4, 4, 128, 2, 4)
INST_DECODE(512, 2, 2, 128, 1, 4)
INST_DECODE(512, 2, 4, 128, 1, 4)
INST_DECODE(512, 4, 2, 128, 1, 4)
INST_DECODE(512, 4, 4, 128, 1, 4)
INST_DECODE(512, 2, 2, 128, 2, 4)
INST_DECODE(512, 2, 4, 128, 2, 4)
INST_DECODE(512, 4, 2, 128, 2, 4)
INST_DECODE(512, 4, 4, 128, 2, 4)

#undef INST_DECODE_DTYPES
#undef INST_DECODE_NN
#undef INST_DECODE

// ============================================================================
// Decode Launch Wrapper
//
// Dispatches to the correct template instantiation based on head_dim, elem_bytes,
// and next_n (number of new tokens per decode step, capped at 4).
// Grid is 2D: (batch_size, head_blocks) where head_blocks = NTHRD_BASE / 32.
// For HD=512 bf16: head_blocks=2; for HD=128 bf16: head_blocks=1.
// ============================================================================

// Forward declaration (defined in prefill section below).
static inline int prefillNthreads(int head_dim, int elem_bytes_for_vec);

void pagedKvCompressLaunch(void const* kv_score, float const* ape, void* paged_kv, void* paged_score,
    int32_t const* block_table_kv, int32_t const* block_table_score, void* output, int32_t const* kv_lens,
    int32_t const* cu_seq_lens, int32_t const* cu_kv_comp, int batch_size, int page_size, int max_blocks, int head_dim,
    int compress_ratio, int next_n, int kv_score_elem_bytes, int state_elem_bytes, int out_elem_bytes,
    cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(
        compress_ratio == 4 || compress_ratio == 128, "pagedKvCompressLaunch only supports compress_ratio 4 or 128");
    TLLM_CHECK_WITH_INFO(
        (kv_score_elem_bytes == 2 || kv_score_elem_bytes == 4) && (state_elem_bytes == 2 || state_elem_bytes == 4),
        "pagedKvCompressLaunch only supports bf16/fp32 kv_score and paged state");

    // Compute HEAD_BLOCKS: mirrors the compile-time constant in the kernel.
    // VEC = max_vec if HEAD_DIM/max_vec >= 32, else HEAD_DIM/32.
    // NTHRD_BASE = HEAD_DIM / VEC; HEAD_BLOCKS = NTHRD_BASE / 32 (or 1 if <= 32).
    // This spreads the head_dim across multiple blocks for better SM utilisation.
    int const elem_bytes_for_vec = max(kv_score_elem_bytes, state_elem_bytes);
    int const max_vec_elem = 16 / elem_bytes_for_vec;
    int const vec = (head_dim / max_vec_elem >= 32) ? max_vec_elem : (head_dim / 32);
    int const nthrd_base = head_dim / vec;               // mirrors kernel's NTHRD_BASE = HEAD_DIM / VEC
    int const head_blocks = (nthrd_base > 32) ? (nthrd_base / 32) : 1;
    int const nthreads_inner = nthrd_base / head_blocks; // = min(32, nthrd_base) = always 32

    // For large compress_ratio, use 4-warp parallel reduction to cut the serial
    // softmax loop from COMPRESS_RATIO iterations to COMPRESS_RATIO/4 per warp.
    // Supported configs: CR=128, (HD=128 or HD=512), NEXT_N=1 or NEXT_N=2.
    //
    // smem per block = 3 * MULTI_WARP * ELEM_PER_BLOCK * sizeof(float)
    //   where ELEM_PER_BLOCK = nthreads_inner * vec = HEAD_DIM / HEAD_BLOCKS.
    // HD=128: ELEM_PER_BLOCK=128 → 6 KB.
    // HD=512 with max elem size 2 (vec=8, HEAD_BLOCKS=2): ELEM_PER_BLOCK=256 → 12 KB.
    // HD=512 with max elem size 4 (vec=4, HEAD_BLOCKS=4): ELEM_PER_BLOCK=128 →  6 KB.
    constexpr int MULTI_WARP = 4;
    bool const use_multi_warp = (compress_ratio == 128 && next_n <= 2);
    int const num_red_warps = use_multi_warp ? MULTI_WARP : 1;
    int const nthreads = nthreads_inner * num_red_warps;
    int const elem_per_block = nthreads_inner * vec; // = HEAD_DIM / HEAD_BLOCKS
    int const smem_bytes = use_multi_warp ? (3 * MULTI_WARP * elem_per_block * static_cast<int>(sizeof(float))) : 0;

    dim3 grid(batch_size, head_blocks);

#define LAUNCH_DECODE(HD, KV_EB, STATE_EB, CR, NN)                                                                     \
    pagedKvCompressKernel<HD, KV_EB, STATE_EB, CR, NN><<<grid, nthreads, smem_bytes, stream>>>(kv_score, ape,          \
        paged_kv, paged_score, block_table_kv, block_table_score, output, kv_lens, cu_seq_lens, cu_kv_comp, page_size, \
        max_blocks, out_elem_bytes)

#define LAUNCH_DECODE_MW(HD, KV_EB, STATE_EB, CR, NN)                                                                  \
    pagedKvCompressKernel<HD, KV_EB, STATE_EB, CR, NN, MULTI_WARP><<<grid, nthreads, smem_bytes, stream>>>(kv_score,   \
        ape, paged_kv, paged_score, block_table_kv, block_table_score, output, kv_lens, cu_seq_lens, cu_kv_comp,       \
        page_size, max_blocks, out_elem_bytes)

#define DISPATCH_NN_MW(HD, KV_EB, STATE_EB, CR)                                                                        \
    switch (next_n)                                                                                                    \
    {                                                                                                                  \
    case 2: LAUNCH_DECODE_MW(HD, KV_EB, STATE_EB, CR, 2); break;                                                       \
    default: LAUNCH_DECODE_MW(HD, KV_EB, STATE_EB, CR, 1); break;                                                      \
    }

#define DISPATCH_NN(HD, KV_EB, STATE_EB, CR)                                                                           \
    switch (next_n)                                                                                                    \
    {                                                                                                                  \
    case 1: LAUNCH_DECODE(HD, KV_EB, STATE_EB, CR, 1); break;                                                          \
    case 2: LAUNCH_DECODE(HD, KV_EB, STATE_EB, CR, 2); break;                                                          \
    case 3: LAUNCH_DECODE(HD, KV_EB, STATE_EB, CR, 3); break;                                                          \
    default: LAUNCH_DECODE(HD, KV_EB, STATE_EB, CR, 4); break;                                                         \
    }

#define DISPATCH_DTYPE(HD, CR, DISPATCH_MACRO)                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        if (kv_score_elem_bytes == 4 && state_elem_bytes == 4)                                                         \
        {                                                                                                              \
            DISPATCH_MACRO(HD, 4, 4, CR);                                                                              \
        }                                                                                                              \
        else if (kv_score_elem_bytes == 2 && state_elem_bytes == 4)                                                    \
        {                                                                                                              \
            DISPATCH_MACRO(HD, 2, 4, CR);                                                                              \
        }                                                                                                              \
        else if (kv_score_elem_bytes == 4 && state_elem_bytes == 2)                                                    \
        {                                                                                                              \
            DISPATCH_MACRO(HD, 4, 2, CR);                                                                              \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            DISPATCH_MACRO(HD, 2, 2, CR);                                                                              \
        }                                                                                                              \
    } while (false)

    if (use_multi_warp)
    {
        // Multi-warp path: HD=128 or HD=512, NEXT_N=1 or NEXT_N=2, CR=128.
        if (head_dim == 512)
        {
            DISPATCH_DTYPE(512, 128, DISPATCH_NN_MW);
        }
        else
        {
            DISPATCH_DTYPE(128, 128, DISPATCH_NN_MW);
        }
    }
    else if (compress_ratio == 4)
    {
        if (head_dim == 512)
        {
            DISPATCH_DTYPE(512, 4, DISPATCH_NN);
        }
        else
        {
            DISPATCH_DTYPE(128, 4, DISPATCH_NN);
        }
    }
    else
    {
        if (head_dim == 512)
        {
            DISPATCH_DTYPE(512, 128, DISPATCH_NN);
        }
        else
        {
            DISPATCH_DTYPE(128, 128, DISPATCH_NN);
        }
    }

#undef DISPATCH_DTYPE
#undef DISPATCH_NN
#undef DISPATCH_NN_MW
#undef LAUNCH_DECODE_MW
#undef LAUNCH_DECODE
}

// ============================================================================
// Prefill Kernel: prefillReductionKernel
//
// Template: <HEAD_DIM, KV_SCORE_ELEM_BYTES, STATE_ELEM_BYTES>
//
// Grid:  (batch_size, max_outputs_per_batch) — one block per compressed output
// Block: (NTHRD) where NTHRD = HEAD_DIM / VEC (>= 32 threads)
//
// Unlike the decode kernel (which operates token-by-token from paged state),
// the prefill kernel processes the full input sequence at once. Each block
// reads compress_ratio consecutive input rows from kv_score and reduces them
// via online softmax to produce one compressed output.
//
// The last block (local_output_idx == num_outputs - 1) also handles saving
// compressor state for any remainder tokens that don't form a full chunk.
// This state is written to paged kv/score caches for use in future decode steps.
//
// Memory layout:
//   kv_score:    [total_tokens, 2*state_dim] — interleaved KV and score from linear projection
//   paged_kv:    paged cache for compressor state (remainder)
//   paged_score: paged cache for compressor score state (remainder, with APE)
//   output:      [total_comp_tokens, head_dim] — compressed output tokens
// ============================================================================

// Per-element online softmax step on VEC elements via 128-bit vectorized loads.
// Reads directly from the kv_score input buffer (not paged state) since prefill
// has the full sequence available.
template <int HEAD_DIM, int KV_SCORE_ELEM_BYTES, int VEC>
__device__ __forceinline__ void prefillSoftmaxVec(void const* __restrict__ kv_score_raw, float const* __restrict__ ape,
    int64_t row_elem, // (input_offset + row_idx) * two_sd
    int kv_col_off,   // column offset into kv_score row (0 or HEAD_DIM)
    int ape_base,     // r * state_dim + ape_col_off
    int state_dim, int tid, float* __restrict__ rmax, float* __restrict__ rsum, float* __restrict__ rwsum)
{
    using KvScoreElemT = typename std::conditional<KV_SCORE_ELEM_BYTES == 2, __nv_bfloat16, float>::type;
    using KvScoreVecT = typename VecType<VEC * KV_SCORE_ELEM_BYTES>::type;

    auto const* kv = reinterpret_cast<KvScoreElemT const*>(kv_score_raw);

    KvScoreVecT k_raw = reinterpret_cast<KvScoreVecT const*>(&kv[row_elem + kv_col_off])[tid];
    KvScoreVecT s_raw = reinterpret_cast<KvScoreVecT const*>(&kv[row_elem + state_dim + kv_col_off])[tid];
    KvScoreElemT const* ke = reinterpret_cast<KvScoreElemT const*>(&k_raw);
    KvScoreElemT const* se = reinterpret_cast<KvScoreElemT const*>(&s_raw);

#pragma unroll
    for (int i = 0; i < VEC; i += 4)
    {
        float4 av = *reinterpret_cast<float4 const*>(&ape[ape_base + tid * VEC + i]);
        float kf[4] = {static_cast<float>(ke[i]), static_cast<float>(ke[i + 1]), static_cast<float>(ke[i + 2]),
            static_cast<float>(ke[i + 3])};
        float sf[4] = {static_cast<float>(se[i]) + av.x, static_cast<float>(se[i + 1]) + av.y,
            static_cast<float>(se[i + 2]) + av.z, static_cast<float>(se[i + 3]) + av.w};
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            float nm = fmaxf(rmax[i + j], sf[j]);
            float sc = expf(rmax[i + j] - nm);
            float tm = expf(sf[j] - nm);
            rsum[i + j] = rsum[i + j] * sc + tm;
            rwsum[i + j] = rwsum[i + j] * sc + kf[j] * tm;
            rmax[i + j] = nm;
        }
    }
}

template <int HEAD_DIM, int KV_SCORE_ELEM_BYTES, int STATE_ELEM_BYTES, int COMPRESS_RATIO>
__global__ void prefillReductionKernel(void const* __restrict__ kv_score_raw, float const* __restrict__ ape,
    void* __restrict__ paged_kv_raw, void* __restrict__ paged_score_raw, int32_t const* __restrict__ block_table_kv,
    int32_t const* __restrict__ block_table_score, void* __restrict__ output_raw, int32_t const* __restrict__ kv_lens,
    int32_t const* __restrict__ start_pos_arr, int32_t const* __restrict__ cu_seq_lens,
    int32_t const* __restrict__ cu_kv_comp, int page_size, int state_dim, int max_blocks, int out_elem_bytes)
{
    using KvScoreElemT = typename std::conditional<KV_SCORE_ELEM_BYTES == 2, __nv_bfloat16, float>::type;
    using StateElemT = typename std::conditional<STATE_ELEM_BYTES == 2, __nv_bfloat16, float>::type;
    static_assert(COMPRESS_RATIO == 4 || COMPRESS_RATIO == 128, "Unsupported COMPRESS_RATIO");
    constexpr bool IS_OVERLAP = (COMPRESS_RATIO == 4);

    constexpr int ELEM_BYTES_FOR_VEC
        = (KV_SCORE_ELEM_BYTES > STATE_ELEM_BYTES) ? KV_SCORE_ELEM_BYTES : STATE_ELEM_BYTES;
    constexpr int MAX_VEC = 16 / ELEM_BYTES_FOR_VEC;
    constexpr int VEC = (HEAD_DIM / MAX_VEC >= 32) ? MAX_VEC : (HEAD_DIM / 32);
    using KvScoreVecT = typename VecType<VEC * KV_SCORE_ELEM_BYTES>::type;
    using StateVecT = typename VecType<VEC * STATE_ELEM_BYTES>::type;
    static_assert(VEC >= 4, "VEC must be >= 4 for float4 ape loads");

    int const tid = threadIdx.x;
    int const batch_idx = blockIdx.x;
    int const local_output_idx = blockIdx.y;

    int const sp = start_pos_arr[batch_idx];
    int const kv_len = kv_lens[batch_idx];
    int const input_offset = cu_seq_lens[batch_idx];
    int const output_offset = cu_kv_comp[batch_idx];

    // Absolute compression index range.  Window k covers [k*R, (k+1)*R).
    // We emit one output per complete window that gains at least one new token.
    int const first_abs_idx = sp / COMPRESS_RATIO;
    int const last_abs_idx = kv_len / COMPRESS_RATIO; // exclusive
    int const actual_num_outputs = last_abs_idx - first_abs_idx;
    // Keep at least one block so the last CTA can still persist remainder state
    // even when this chunk has no full compression window.
    int const num_outputs = max(actual_num_outputs, 1);

    if (local_output_idx >= num_outputs)
        return;

    constexpr int coff = IS_OVERLAP ? 2 : 1;
    bool const should_compress = (local_output_idx < actual_num_outputs);

    // Absolute window for this output block.
    int const abs_idx = first_abs_idx + local_output_idx;
    int const win_start = abs_idx * COMPRESS_RATIO;

    auto const* kv_score = reinterpret_cast<KvScoreElemT const*>(kv_score_raw);
    auto* paged_kv = reinterpret_cast<StateElemT*>(paged_kv_raw);
    auto* paged_score = reinterpret_cast<StateElemT*>(paged_score_raw);

    int64_t const two_sd = 2 * state_dim;
    int64_t const page_sd = static_cast<int64_t>(page_size) * state_dim;

    // ================================================================
    // Phase 1: State Update (all output blocks, for block reuse support)
    //
    // 1a runs on every block that has a full chunk (should_compress), writing
    // each block's own window to paged cache so any prefix slice is valid for
    // block reuse. 1b (remainder) still runs only on the last block.
    // ================================================================

    // Helper: write a contiguous block of positions to paged KV/score cache.
    // Only new positions (>= sp) are written; positions already persisted from
    // prior calls are skipped by starting the loop at write_r_start.
    // APE index is r (position within the compression window), matching the
    // index used during the original decode/prefill that first established
    // the window alignment.
    auto write_to_paged = [&](int range_start, int range_end, int write_r_start)
    {
        for (int r = write_r_start; r < range_end - range_start; r++)
        {
            int const pos = range_start + r;
            int const input_row = pos - sp; // >= 0 since pos >= sp (loop starts at write_r_start)
            int const log_blk = pos / page_size;
            int const blk_off = pos % page_size;
            int const phys_kv = block_table_kv[batch_idx * max_blocks + log_blk];
            int const phys_sc = block_table_score[batch_idx * max_blocks + log_blk];

            for (int col_idx = 0; col_idx < coff; col_idx++)
            {
                int const col = col_idx * HEAD_DIM;
                int64_t const src = static_cast<int64_t>(input_offset + input_row) * two_sd + col;
                int64_t const dkv = static_cast<int64_t>(phys_kv) * page_sd + blk_off * state_dim + col;
                int64_t const dsc = static_cast<int64_t>(phys_sc) * page_sd + blk_off * state_dim + col;

                KvScoreVecT kv_raw = reinterpret_cast<KvScoreVecT const*>(&kv_score[src])[tid];
                KvScoreVecT sc_raw = reinterpret_cast<KvScoreVecT const*>(&kv_score[src + state_dim])[tid];

                KvScoreElemT const* kv_e = reinterpret_cast<KvScoreElemT const*>(&kv_raw);
                StateVecT kv_out;
                StateElemT* kv_o = reinterpret_cast<StateElemT*>(&kv_out);
#pragma unroll
                for (int i = 0; i < VEC; i++)
                {
                    kv_o[i] = static_cast<StateElemT>(static_cast<float>(kv_e[i]));
                }
                reinterpret_cast<StateVecT*>(&paged_kv[dkv])[tid] = kv_out;

                KvScoreElemT const* sc_e = reinterpret_cast<KvScoreElemT const*>(&sc_raw);
                StateVecT sc_out;
                StateElemT* sc_o = reinterpret_cast<StateElemT*>(&sc_out);
#pragma unroll
                for (int i = 0; i < VEC; i += 4)
                {
                    float4 av = *reinterpret_cast<float4 const*>(&ape[r * state_dim + col + tid * VEC + i]);
                    sc_o[i] = static_cast<StateElemT>(static_cast<float>(sc_e[i]) + av.x);
                    sc_o[i + 1] = static_cast<StateElemT>(static_cast<float>(sc_e[i + 1]) + av.y);
                    sc_o[i + 2] = static_cast<StateElemT>(static_cast<float>(sc_e[i + 2]) + av.z);
                    sc_o[i + 3] = static_cast<StateElemT>(static_cast<float>(sc_e[i + 3]) + av.w);
                }
                reinterpret_cast<StateVecT*>(&paged_score[dsc])[tid] = sc_out;
            }
        }
    };

    // 1a. Full chunk for this output block.
    // Positions [win_start, sp) are already in paged cache from a prior call;
    // start the loop at write_r_start to skip them without a per-iteration branch.
    if (should_compress)
    {
        int const write_r_start = (win_start < sp) ? (sp - win_start) : 0;
        write_to_paged(win_start, win_start + COMPRESS_RATIO, write_r_start);
    }

    // 1b. Remainder tokens (last block only).
    // Tokens past the last complete window are persisted so a later call can
    // continue the same compression window.
    // rem_start_pos < sp when the chunk has no full window (actual_num_outputs == 0);
    // rem_write_start skips those already-paged positions without a per-iteration branch.
    if (local_output_idx == num_outputs - 1)
    {
        int const rem_start_pos = last_abs_idx * COMPRESS_RATIO;
        int const rem_count = kv_len - rem_start_pos;
        int const rem_write_start = (rem_start_pos < sp) ? (sp - rem_start_pos) : 0;
        write_to_paged(rem_start_pos, rem_start_pos + rem_count, rem_write_start);
    }

    // ================================================================
    // Phase 2: Online softmax reduction (vectorized)
    //
    // Each block reduces compress_ratio rows into one output via per-element
    // online softmax: output[d] = sum_r(kv[r,d] * softmax(score[r,d] + ape[r,d]))
    // In overlap mode, combines previous chunk's first-half and current chunk's
    // second-half features (same as decode kernel).
    // ================================================================
    if (!should_compress)
        return;

    float rmax[VEC], rsum[VEC], rwsum[VEC];
#pragma unroll
    for (int i = 0; i < VEC; i++)
    {
        rmax[i] = -INFINITY;
        rsum[i] = 0.0f;
        rwsum[i] = 0.0f;
    }

    // Helper: online-softmax reduction over a contiguous window of COMPRESS_RATIO positions.
    // Positions [range_start, range_start + new_start) come from paged cache (APE already
    // fused during the call that wrote them); positions [range_start + new_start, range_start
    // + COMPRESS_RATIO) are new tokens read from kv_score with live APE addition.
    // Two branch-free loops avoid a per-iteration if/else on pos < sp.
    //   kv_col_off  — column offset into kv_score / paged state (0 or HEAD_DIM for overlap)
    //   ape_col_off — column offset into APE table for the score column (same as kv_col_off)
    auto reduce_window = [&](int range_start, int kv_col_off, int ape_col_off)
    {
        // Precompute split once for the whole window.
        int const new_start = (range_start < sp) ? min(sp - range_start, COMPRESS_RATIO) : 0;

        // Paged portion — APE already fused when tokens were stored.
        for (int r = 0; r < new_start; r++)
        {
            int const pos = range_start + r;
            int log_blk = pos / page_size;
            int blk_off = pos % page_size;
            int phys_kv = block_table_kv[batch_idx * max_blocks + log_blk];
            int phys_sc = block_table_score[batch_idx * max_blocks + log_blk];
            decodeSoftmaxVec<HEAD_DIM, STATE_ELEM_BYTES, VEC>(paged_kv_raw, paged_score_raw, page_sd, state_dim,
                phys_kv, phys_sc, blk_off, kv_col_off, tid, rmax, rsum, rwsum);
        }

        // New-token portion — read from kv_score and add APE live.
        for (int r = new_start; r < COMPRESS_RATIO; r++)
        {
            int const pos = range_start + r;
            int const input_row = pos - sp;
            int64_t const row = static_cast<int64_t>(input_offset + input_row) * two_sd;
            prefillSoftmaxVec<HEAD_DIM, KV_SCORE_ELEM_BYTES, VEC>(
                kv_score_raw, ape, row, kv_col_off, r * state_dim + ape_col_off, state_dim, tid, rmax, rsum, rwsum);
        }
    };

    if constexpr (IS_OVERLAP)
    {
        // Overlap mode: each output combines
        //   prev-segment (first head_dim,  kv_col=0)    from window (abs_idx-1)
        //   curr-segment (second head_dim, kv_col=HD)   from window abs_idx
        if (abs_idx > 0)
            reduce_window((abs_idx - 1) * COMPRESS_RATIO, 0, 0); // prev window, first half
        reduce_window(win_start, HEAD_DIM, HEAD_DIM);            // curr window, second half
    }
    else
    {
        // Non-overlap mode: reduce one ratio-sized window.
        reduce_window(win_start, 0, 0);
    }

    // ================================================================
    // Store output (vectorized)
    // ================================================================
    int64_t const out_base = static_cast<int64_t>(output_offset + local_output_idx) * HEAD_DIM + tid * VEC;

    if (out_elem_bytes == 2)
    {
        __nv_bfloat16 packed[VEC];
#pragma unroll
        for (int i = 0; i < VEC; i++)
            packed[i] = __float2bfloat16_rn(rwsum[i] / rsum[i]);
        // VEC * 2 bytes: VEC=4 → 8B (uint2), VEC=8 → 16B (uint4)
        using OutVecT = typename VecType<VEC * 2>::type;
        *reinterpret_cast<OutVecT*>(&reinterpret_cast<__nv_bfloat16*>(output_raw)[out_base])
            = *reinterpret_cast<OutVecT const*>(packed);
    }
    else
    {
        float result[VEC];
#pragma unroll
        for (int i = 0; i < VEC; i++)
            result[i] = rwsum[i] / rsum[i];
            // VEC * 4 bytes: VEC=4 → 16B (uint4/float4), VEC=8 → 32B (2×float4)
#pragma unroll
        for (int i = 0; i < VEC; i += 4)
            *reinterpret_cast<float4*>(&reinterpret_cast<float*>(output_raw)[out_base + i])
                = *reinterpret_cast<float4 const*>(&result[i]);
    }
}

// Explicit instantiations
#define INST_PREFILL(HD, KV_EB, STATE_EB, CR)                                                                          \
    template __global__ void prefillReductionKernel<HD, KV_EB, STATE_EB, CR>(void const*, float const*, void*, void*,  \
        int32_t const*, int32_t const*, void*, int32_t const*, int32_t const*, int32_t const*, int32_t const*, int,    \
        int, int, int);

#define INST_PREFILL_DTYPES(HD, CR)                                                                                    \
    INST_PREFILL(HD, 2, 2, CR)                                                                                         \
    INST_PREFILL(HD, 2, 4, CR) INST_PREFILL(HD, 4, 2, CR) INST_PREFILL(HD, 4, 4, CR)

INST_PREFILL_DTYPES(128, 4)
INST_PREFILL_DTYPES(128, 128)
INST_PREFILL_DTYPES(512, 4)
INST_PREFILL_DTYPES(512, 128)
#undef INST_PREFILL_DTYPES
#undef INST_PREFILL

// ============================================================================
// Prefill Launch Wrapper
//
// Grid is (batch_size, max(max_outputs, 1)). Blocks for local_output_idx >= num_outputs
// early-exit inside the kernel.
// ============================================================================

// Compute threads per block: mirrors compile-time NTHRD = HEAD_DIM / VEC.
static inline int prefillNthreads(int head_dim, int elem_bytes_for_vec)
{
    int max_vec = 16 / elem_bytes_for_vec;
    int vec = (head_dim / max_vec >= 32) ? max_vec : (head_dim / 32);
    return head_dim / vec;
}

void prefillReductionLaunch(void const* kv_score, float const* ape, void* paged_kv, void* paged_score,
    int32_t const* block_table_kv, int32_t const* block_table_score, void* output, int32_t const* kv_lens,
    int32_t const* start_pos, int32_t const* cu_seq_lens, int32_t const* cu_kv_comp, int batch_size, int page_size,
    int max_blocks, int head_dim, int compress_ratio, int max_outputs, int kv_score_elem_bytes, int state_elem_bytes,
    int out_elem_bytes, cudaStream_t stream)
{
    bool const overlap = (compress_ratio == 4);
    TLLM_CHECK_WITH_INFO(
        compress_ratio == 4 || compress_ratio == 128, "prefillReductionLaunch only supports compress_ratio 4 or 128");
    TLLM_CHECK_WITH_INFO(
        (kv_score_elem_bytes == 2 || kv_score_elem_bytes == 4) && (state_elem_bytes == 2 || state_elem_bytes == 4),
        "prefillReductionLaunch only supports bf16/fp32 kv_score and paged state");
    int const elem_bytes_for_vec = max(kv_score_elem_bytes, state_elem_bytes);
    int const nthreads = prefillNthreads(head_dim, elem_bytes_for_vec);
    int const coff = overlap ? 2 : 1;
    int const state_dim = coff * head_dim;
    dim3 grid(batch_size, max(max_outputs, 1));

#define LAUNCH_PREFILL(HD, KV_EB, STATE_EB, CR)                                                                        \
    prefillReductionKernel<HD, KV_EB, STATE_EB, CR><<<grid, nthreads, 0, stream>>>(kv_score, ape, paged_kv,            \
        paged_score, block_table_kv, block_table_score, output, kv_lens, start_pos, cu_seq_lens, cu_kv_comp,           \
        page_size, state_dim, max_blocks, out_elem_bytes)

#define DISPATCH_PREFILL_DTYPE(HD, CR)                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        if (kv_score_elem_bytes == 4 && state_elem_bytes == 4)                                                         \
        {                                                                                                              \
            LAUNCH_PREFILL(HD, 4, 4, CR);                                                                              \
        }                                                                                                              \
        else if (kv_score_elem_bytes == 2 && state_elem_bytes == 4)                                                    \
        {                                                                                                              \
            LAUNCH_PREFILL(HD, 2, 4, CR);                                                                              \
        }                                                                                                              \
        else if (kv_score_elem_bytes == 4 && state_elem_bytes == 2)                                                    \
        {                                                                                                              \
            LAUNCH_PREFILL(HD, 4, 2, CR);                                                                              \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            LAUNCH_PREFILL(HD, 2, 2, CR);                                                                              \
        }                                                                                                              \
    } while (false)

    if (head_dim == 512)
    {
        if (compress_ratio == 4)
            DISPATCH_PREFILL_DTYPE(512, 4);
        else
            DISPATCH_PREFILL_DTYPE(512, 128);
    }
    else
    {
        if (compress_ratio == 4)
            DISPATCH_PREFILL_DTYPE(128, 4);
        else
            DISPATCH_PREFILL_DTYPE(128, 128);
    }

#undef DISPATCH_PREFILL_DTYPE
#undef LAUNCH_PREFILL
}

// ============================================================================
// Postprocess + Scatter Kernel: postProcessScatterKernel
//
// Template: <HEAD_DIM, ELEM_BYTES, SCALE_TYPE>
//
// Grid:  (total_tokens) — one block per compressed token
// Block: (NTHRD = HEAD_DIM / VEC) threads, always >= 32
// Smem:  HEAD_DIM * sizeof(float) — used for cross-warp Hadamard butterfly
//
// This kernel fuses all post-compression processing with the paged cache write
// into a single kernel launch, keeping data in float32 registers throughout.
// This eliminates the DRAM round-trip that a split postprocess+scatter would need.
//
// SCALE_TYPE alone determines the output cache layout:
//   - kNone:              ELEM_BYTES per value (bf16 / fp32) into kv_cache.
//   - kFP8PerTensor:      one fp8 byte per value, implicit scale=1.0.
//   - kFP8Blockwise:      one fp8 byte per value + one fp32 scale per 128 values.
//   - kMXFP4Blockwise:    packed fp4 (two values per byte) + one ue8m0 byte
//                         per 32 values.
//
// Pipeline (10 steps, all in fp32 registers):
//   1. Vectorized load compressed token from kv_comp
//   2. RMSNorm: compute sum-of-squares → cross-warp reduce → rsqrt → scale
//   3. Apply RMSNorm weights
//   4. RoPE: interleaved even/odd rotation on rope_dim elements (skip nope_dim)
//   5. Hadamard butterfly transform (3 phases: local → warp shuffle → shared mem)
//   6. Scale by 1/sqrt(HEAD_DIM) (Hadamard normalization)
//   7. Optionally write postprocessed result to kv_out (for callers that need it)
//   8. Binary search cu_kv_comp to find batch_idx for this token
//   9. Compute paged cache destination (logical→physical block via block table)
//  10. Store to cache (layout by SCALE_TYPE).
// ============================================================================

template <int HEAD_DIM, int ELEM_BYTES, CacheScaleType SCALE_TYPE = CacheScaleType::kNone,
    bool ROTATE_ACTIVATION = true>
__global__ void postProcessScatterKernel(void const* __restrict__ kv_comp, // [total_tokens, head_dim] input
    void* __restrict__ kv_out,                // [total_tokens, head_dim] postprocessed output (may be nullptr)
    void const* __restrict__ rms_weight,      // [head_dim]
    float rms_eps,
    float const* __restrict__ cos_sin_table,  // [max_pos, 2, rope_dim/2]
    int32_t const* __restrict__ position_ids, // [total_tokens]
    int nope_dim, int rope_dim,
    // scatter params
    void* __restrict__ kv_cache, // paged cache buffer
    int32_t const* __restrict__ num_outputs_arr, int32_t const* __restrict__ cu_kv_comp,
    int32_t const* __restrict__ start_pos_arr, int32_t const* __restrict__ block_offsets,
    bool const* __restrict__ compressed_mask, int batch_size, int tokens_per_block, int max_blocks,
    int cache_stride_blk_bytes, int total_tokens, int num_scale_blocks, void* __restrict__ quant_output,
    void* __restrict__ scale_output)
{
    using ElemT = typename std::conditional<ELEM_BYTES == 2, __nv_bfloat16, float>::type;
    constexpr int MAX_VEC = 16 / ELEM_BYTES;
    constexpr int VEC = (HEAD_DIM / MAX_VEC >= 32) ? MAX_VEC : (HEAD_DIM / 32);
    constexpr int NTHRD = HEAD_DIM / VEC;
    constexpr int VEC_BYTES = VEC * ELEM_BYTES;
    using VecT = typename VecType<VEC_BYTES>::type;

    int const token_idx = blockIdx.x;
    if (token_idx >= total_tokens)
        return;

    // Per-token mask: precomputed on host, skips padded generation slots
    // and batches that produced no compressed tokens.
    if (!compressed_mask[token_idx])
        return;

    // ================================================================
    // Step 0: Find owning batch via binary search on cu_kv_comp.
    // ================================================================
    int batch_idx, local_output_idx;
    if (batch_size <= 1)
    {
        batch_idx = 0;
        local_output_idx = token_idx;
    }
    else
    {
        int lo = 0, hi = batch_size;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (cu_kv_comp[mid + 1] <= token_idx)
                lo = mid + 1;
            else
                hi = mid;
        }
        batch_idx = lo;
        if (batch_idx >= batch_size)
            return;
        local_output_idx = token_idx - cu_kv_comp[batch_idx];
    }

    if (local_output_idx >= num_outputs_arr[batch_idx])
        return;

    int const tid = threadIdx.x;
    extern __shared__ float smem[];

    // ================================================================
    // Step 1: Vectorized load from kv_comp
    // ================================================================
    auto const* src = reinterpret_cast<VecT const*>(
        reinterpret_cast<ElemT const*>(kv_comp) + static_cast<int64_t>(token_idx) * HEAD_DIM);
    VecT raw_in = src[tid];
    ElemT const* in_elems = reinterpret_cast<ElemT const*>(&raw_in);

    float v[VEC];
#pragma unroll
    for (int i = 0; i < VEC; i++)
        v[i] = static_cast<float>(in_elems[i]);

    // ================================================================
    // Step 2: RMSNorm
    // ================================================================
    float local_sq = 0.f;
#pragma unroll
    for (int i = 0; i < VEC; i++)
        local_sq += v[i] * v[i];

    float warp_sum = warpReduceSum(local_sq);

    constexpr int NUM_WARPS = (NTHRD + 31) / 32;
    int const warp_id = tid / 32;
    int const lane_id = tid % 32;

    if (lane_id == 0)
        smem[warp_id] = warp_sum;
    __syncthreads();

    if (warp_id == 0)
    {
        float s = (lane_id < NUM_WARPS) ? smem[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            s += __shfl_xor_sync(0xFFFFFFFF, s, offset);
        if (lane_id == 0)
            smem[0] = s;
    }
    __syncthreads();
    float const rms_scale = rsqrtf(smem[0] / static_cast<float>(HEAD_DIM) + rms_eps);

    // ================================================================
    // Step 3: Load weight, apply RMSNorm
    // ================================================================
    auto const* wt_src = reinterpret_cast<VecT const*>(reinterpret_cast<ElemT const*>(rms_weight));
    VecT raw_w = wt_src[tid];
    ElemT const* w_elems = reinterpret_cast<ElemT const*>(&raw_w);

#pragma unroll
    for (int i = 0; i < VEC; i++)
        v[i] = v[i] * rms_scale * static_cast<float>(w_elems[i]);

    // ================================================================
    // Step 4: RoPE (Rotary Positional Embedding)
    //
    // Applied only to elements in [nope_dim, nope_dim+rope_dim).
    // Uses interleaved even/odd pairs: (x_even, x_odd) → rotated by (cos, sin).
    // cos_sin_table layout: [max_pos, rope_dim] where first half is cos, second is sin.
    // ================================================================
    int const half_rope = rope_dim / 2;
    int const pos_id = position_ids[token_idx];

#pragma unroll
    for (int i = 0; i < VEC; i += 2)
    {
        int const elem_idx = tid * VEC + i;
        if (elem_idx >= nope_dim)
        {
            int const rope_idx = elem_idx - nope_dim;
            int const d = rope_idx >> 1;
            float const cos_v = cos_sin_table[pos_id * rope_dim + d];
            float const sin_v = cos_sin_table[pos_id * rope_dim + half_rope + d];
            float const x_even = v[i];
            float const x_odd = v[i + 1];
            v[i] = x_even * cos_v - x_odd * sin_v;
            v[i + 1] = x_odd * cos_v + x_even * sin_v;
        }
    }

    // ================================================================
    // Step 5: Hadamard butterfly transform (rotate activation)
    //
    // Implements the Walsh-Hadamard transform H_n * v via butterfly network.
    // H_n has the recursive structure: H_n = [[H_{n/2}, H_{n/2}], [H_{n/2}, -H_{n/2}]]
    // which decomposes into log2(HEAD_DIM) butterfly stages.
    //
    // Three phases handle increasing stride lengths:
    //   A) Local: strides < VEC — within each thread's register file
    //   B) Warp shuffle: strides VEC..32*VEC-1 — via __shfl_xor_sync
    //   C) Shared memory: strides >= 32*VEC — via XOR-swizzled smem
    //
    // The XOR swizzle pattern `idx ^ ((idx >> 3) & 0x1F)` ensures bank-conflict-
    // free access to shared memory across all butterfly stride patterns.
    //
    // Skipped entirely when ROTATE_ACTIVATION=false.
    // ================================================================
    if constexpr (ROTATE_ACTIVATION)
    {

        // Phase A: local butterfly (strides 1..VEC-1, within each thread's VEC registers)
#pragma unroll
        for (int stride = 1; stride < VEC; stride <<= 1)
        {
#pragma unroll
            for (int i = 0; i < VEC; i++)
            {
                if ((i & stride) == 0)
                {
                    float a = v[i], b = v[i ^ stride];
                    v[i] = a + b;
                    v[i ^ stride] = a - b;
                }
            }
        }

        // Phase B: warp shuffle butterfly (strides VEC..32*VEC-1, within a single warp)
        if constexpr (NTHRD > 1)
        {
            constexpr int SHFL_END = (NTHRD <= 32) ? NTHRD : 32;
#pragma unroll
            for (int ts = 1; ts < SHFL_END; ts <<= 1)
            {
                int const stride = ts * VEC;
#pragma unroll
                for (int i = 0; i < VEC; i++)
                {
                    float partner = __shfl_xor_sync(0xFFFFFFFF, v[i], ts);
                    int const elem_idx = tid * VEC + i;
                    v[i] = (elem_idx & stride) ? (partner - v[i]) : (v[i] + partner);
                }
            }
        }

        // Phase C: cross-warp butterfly via XOR-swizzled shared memory (strides >= 32*VEC)
        // Only needed when NTHRD > 32 (i.e., multiple warps, e.g., HEAD_DIM=512, VEC=8 → 64 threads)
        if constexpr (NTHRD > 32)
        {
#pragma unroll
            for (int i = 0; i < VEC; i++)
            {
                int const idx = tid * VEC + i;
                smem[idx ^ ((idx >> 3) & 0x1F)] = v[i];
            }
            __syncthreads();

            for (int stride = 32 * VEC; stride < HEAD_DIM; stride <<= 1)
            {
#pragma unroll
                for (int i = 0; i < VEC; i++)
                {
                    int const idx = tid * VEC + i;
                    int const partner_idx = idx ^ stride;
                    float const a = smem[idx ^ ((idx >> 3) & 0x1F)];
                    float const b = smem[partner_idx ^ ((partner_idx >> 3) & 0x1F)];
                    v[i] = (idx & stride) ? (b - a) : (a + b);
                }
                __syncthreads();
#pragma unroll
                for (int i = 0; i < VEC; i++)
                {
                    int const idx = tid * VEC + i;
                    smem[idx ^ ((idx >> 3) & 0x1F)] = v[i];
                }
                __syncthreads();
            }
        }

        // ================================================================
        // Step 6: Scale by Hadamard factor
        // ================================================================
        float const had_scale = rsqrtf(static_cast<float>(HEAD_DIM));

#pragma unroll
        for (int i = 0; i < VEC; i++)
            v[i] *= had_scale;

    } // ROTATE_ACTIVATION

    // ================================================================
    // Step 7: Write postprocessed output to kv_out (if requested)
    // ================================================================
    if (kv_out != nullptr)
    {
        VecT raw_out;
        ElemT* out_elems = reinterpret_cast<ElemT*>(&raw_out);
#pragma unroll
        for (int i = 0; i < VEC; i++)
            out_elems[i] = static_cast<ElemT>(v[i]);

        auto* dst
            = reinterpret_cast<VecT*>(reinterpret_cast<ElemT*>(kv_out) + static_cast<int64_t>(token_idx) * HEAD_DIM);
        dst[tid] = raw_out;
    }

    // ================================================================
    // Step 9: Compute paged cache destination address.
    // Map (batch_idx, local_output_idx) → logical block → physical block
    // via the block table. block_base points to the start of the physical
    // page; token_offset is the slot within that page.
    // ================================================================
    int const start_pos = start_pos_arr[batch_idx];
    int const cache_pos = start_pos + local_output_idx;
    int const logical_block = cache_pos / tokens_per_block;
    int const token_offset = cache_pos % tokens_per_block;
    int const phys_block = block_offsets[batch_idx * max_blocks + logical_block];

    uint8_t* block_base
        = reinterpret_cast<uint8_t*>(kv_cache) + static_cast<int64_t>(phys_block) * cache_stride_blk_bytes;

    // ================================================================
    // Step 11: Store to cache (compile-time dispatch on cache dtype/scale type)
    //
    // Cache addressing is byte-based: block_base points to the start of
    // the physical block, cache_stride_blk_bytes is the total block size.
    // ================================================================
    if constexpr (SCALE_TYPE == CacheScaleType::kNone)
    {
        // Default mode: float→bf16/fp32 pack + vectorized store.
        // Cache layout per block: [tokens_per_block * HEAD_DIM] elements of ElemT.
        VecT raw_out;
        ElemT* out_elems = reinterpret_cast<ElemT*>(&raw_out);
#pragma unroll
        for (int i = 0; i < VEC; i++)
            out_elems[i] = static_cast<ElemT>(v[i]);

        ElemT* row_base = reinterpret_cast<ElemT*>(block_base) + token_offset * HEAD_DIM;
        reinterpret_cast<VecT*>(row_base)[tid] = raw_out;
    }
    else if constexpr (SCALE_TYPE == CacheScaleType::kFP8PerTensor)
    {
        // FP8 per-tensor: direct float→fp8_e4m3fn cast (implicit scale=1.0).
        // Cache layout per block: [tokens_per_block * HEAD_DIM] bytes of fp8.
        uint8_t fp8_bytes[VEC];
#pragma unroll
        for (int i = 0; i < VEC; i++)
        {
            __nv_fp8_e4m3 fp8_val(v[i]);
            fp8_bytes[i] = *reinterpret_cast<uint8_t const*>(&fp8_val);
        }

        using Fp8VecT = typename VecType<VEC>::type;
        uint8_t* fp8_dst = block_base + token_offset * HEAD_DIM + tid * VEC;
        *reinterpret_cast<Fp8VecT*>(fp8_dst) = *reinterpret_cast<Fp8VecT const*>(fp8_bytes);
    }
    else if constexpr (SCALE_TYPE == CacheScaleType::kFP8Blockwise)
    {
        // FP8 blockwise: per-128-element quantization with explicit scales.
        // GROUP_SIZE = number of threads that share one scale factor.
        // For HD=512, VEC=8: GROUP_SIZE=16 threads → 128 elements per scale block.
        //
        // Cache layout per block:
        //   [fp8_data: tokens_per_block * HEAD_DIM bytes]
        //   [scales:   tokens_per_block * (HEAD_DIM/128) * 4 bytes]
        constexpr int GROUP_SIZE = 128 / VEC;

        // Step 11a: Compute per-group amax via warp shuffle reduction.
        // GROUP_SIZE <= 16 (< warp), so shuffle is sufficient (no smem needed).
        float local_amax = 0.f;
#pragma unroll
        for (int i = 0; i < VEC; i++)
            local_amax = fmaxf(local_amax, fabsf(v[i]));

#pragma unroll
        for (int offset = GROUP_SIZE / 2; offset > 0; offset >>= 1)
            local_amax = fmaxf(local_amax, __shfl_xor_sync(0xFFFFFFFF, local_amax, offset));

        // Step 11b: Compute scale and inverse scale for quantization.
        // 448.0 is the max representable value for fp8_e4m3fn.
        float const scale = local_amax / 448.0f;
        float const inv_scale = (local_amax > 0.f) ? (448.0f / local_amax) : 1.0f;

        // Step 11c: Quantize to FP8 and store data.
        uint8_t fp8_bytes[VEC];
#pragma unroll
        for (int i = 0; i < VEC; i++)
        {
            __nv_fp8_e4m3 fp8_val(v[i] * inv_scale);
            fp8_bytes[i] = *reinterpret_cast<uint8_t const*>(&fp8_val);
        }

        using Fp8VecT = typename VecType<VEC>::type;
        uint8_t* fp8_dst = block_base + token_offset * HEAD_DIM + tid * VEC;
        *reinterpret_cast<Fp8VecT*>(fp8_dst) = *reinterpret_cast<Fp8VecT const*>(fp8_bytes);

        // Step 11d: Store scale factor (one thread per 128-element group writes it).
        if (tid % GROUP_SIZE == 0)
        {
            int const scale_idx = tid / GROUP_SIZE;
            float* scale_dst = reinterpret_cast<float*>(block_base + tokens_per_block * HEAD_DIM
                + (token_offset * num_scale_blocks + scale_idx) * sizeof(float));
            *scale_dst = scale;
        }

        // Step 11e: Optionally write FP8 data and scales to output buffers.
        // Used by the indexer compressor which returns (fp8_data, scales) to Python
        // for downstream sparse attention indexing.
        if (quant_output != nullptr)
        {
            uint8_t* fp8_out_dst
                = reinterpret_cast<uint8_t*>(quant_output) + static_cast<int64_t>(token_idx) * HEAD_DIM + tid * VEC;
            *reinterpret_cast<Fp8VecT*>(fp8_out_dst) = *reinterpret_cast<Fp8VecT const*>(fp8_bytes);
        }
        if (scale_output != nullptr && tid % GROUP_SIZE == 0)
        {
            int const scale_idx = tid / GROUP_SIZE;
            reinterpret_cast<float*>(scale_output)[static_cast<int64_t>(token_idx) * num_scale_blocks + scale_idx]
                = scale;
        }
    }
    else if constexpr (SCALE_TYPE == CacheScaleType::kMXFP4Blockwise)
    {
        constexpr int GROUP_SIZE = 32 / VEC;
        constexpr int PACKED_VEC_BYTES = VEC / 2;
        constexpr float kFp4Max = 6.0f;
        constexpr float kFp4MaxInv = 1.0f / kFp4Max;
        constexpr float kFp4MinAmax = kFp4Max * 1.1754943508222875e-38f;

        float local_amax = 0.f;
#pragma unroll
        for (int i = 0; i < VEC; i++)
            local_amax = fmaxf(local_amax, fabsf(v[i]));

#pragma unroll
        for (int offset = GROUP_SIZE / 2; offset > 0; offset >>= 1)
            local_amax = fmaxf(local_amax, __shfl_xor_sync(0xFFFFFFFF, local_amax, offset));

        float const scale = roundedPow2Scale(local_amax, kFp4MaxInv, kFp4MinAmax);

        uint8_t fp4_bytes[PACKED_VEC_BYTES];
#pragma unroll
        for (int i = 0; i < VEC; i += 2)
        {
            fp4_bytes[i / 2] = packE2M1x2(v[i] / scale, v[i + 1] / scale);
        }

        int const packed_head_dim = HEAD_DIM / 2;
        uint8_t* fp4_dst = block_base + token_offset * packed_head_dim + tid * PACKED_VEC_BYTES;
#pragma unroll
        for (int i = 0; i < PACKED_VEC_BYTES; ++i)
            fp4_dst[i] = fp4_bytes[i];

        if (tid % GROUP_SIZE == 0)
        {
            int const scale_idx = tid / GROUP_SIZE;
            uint8_t* scale_dst
                = block_base + tokens_per_block * packed_head_dim + token_offset * num_scale_blocks + scale_idx;
            *scale_dst = toUe8m0(scale);
        }

        if (quant_output != nullptr)
        {
            uint8_t* fp4_out_dst = reinterpret_cast<uint8_t*>(quant_output)
                + static_cast<int64_t>(token_idx) * packed_head_dim + tid * PACKED_VEC_BYTES;
#pragma unroll
            for (int i = 0; i < PACKED_VEC_BYTES; ++i)
                fp4_out_dst[i] = fp4_bytes[i];
        }
        if (scale_output != nullptr && tid % GROUP_SIZE == 0)
        {
            int const scale_idx = tid / GROUP_SIZE;
            reinterpret_cast<uint8_t*>(scale_output)[static_cast<int64_t>(token_idx) * num_scale_blocks + scale_idx]
                = toUe8m0(scale);
        }
    }
}

// Explicit instantiations — fused postprocess+scatter.
// kNone supports bf16 (EB=2) and fp32 (EB=4) input types; the quantized
// scale types only support bf16 input since the compressor output is bf16.
// Each combination is instantiated with ROTATE_ACTIVATION=true and false.
#define INST_PPS(HD, EB, CST, AR)                                                                                      \
    template __global__ void postProcessScatterKernel<HD, EB, CST, AR>(void const*, void*, void const*, float,         \
        float const*, int32_t const*, int, int, void*, int32_t const*, int32_t const*, int32_t const*, int32_t const*, \
        bool const*, int, int, int, int, int, int, void*, void*);

#define INST_PPS_AR(HD, EB, CST)                                                                                       \
    INST_PPS(HD, EB, CST, true)                                                                                        \
    INST_PPS(HD, EB, CST, false)

INST_PPS_AR(128, 2, CacheScaleType::kNone)
INST_PPS_AR(128, 4, CacheScaleType::kNone)
INST_PPS_AR(512, 2, CacheScaleType::kNone)
INST_PPS_AR(512, 4, CacheScaleType::kNone)
INST_PPS_AR(128, 2, CacheScaleType::kFP8PerTensor)
INST_PPS_AR(512, 2, CacheScaleType::kFP8PerTensor)
INST_PPS_AR(128, 2, CacheScaleType::kFP8Blockwise)
INST_PPS_AR(512, 2, CacheScaleType::kFP8Blockwise)
INST_PPS_AR(128, 2, CacheScaleType::kMXFP4Blockwise)
INST_PPS_AR(512, 2, CacheScaleType::kMXFP4Blockwise)
#undef INST_PPS_AR
#undef INST_PPS

// ============================================================================
// Postprocess + Scatter Launch Wrapper
//
// Derives cache layout parameters (cache_stride_blk_bytes, num_scale_blocks)
// from cache dtype / scale type, then dispatches to the appropriate template instantiation.
// ============================================================================

// Compute number of threads per block, mirroring the compile-time VEC/NTHRD logic.
// Ensures NTHRD >= 32 by reducing VEC when HEAD_DIM is small.
static inline int compressorNthreads(int head_dim, int elem_bytes)
{
    int max_vec = 16 / elem_bytes;
    int vec = (head_dim / max_vec >= 32) ? max_vec : (head_dim / 32);
    return head_dim / vec;
}

void postProcessScatterLaunch(void const* kv_comp, void* kv_out, void const* rms_weight, float rms_eps,
    float const* cos_sin_table, int32_t const* position_ids, int nope_dim, int rope_dim, void* kv_cache,
    int32_t const* num_outputs, int32_t const* cu_kv_comp, int32_t const* start_pos, int32_t const* block_offsets,
    bool const* compressed_mask, int batch_size, int tokens_per_block, int head_dim, int max_blocks_per_seq,
    int elem_bytes, int total_tokens, int cache_scale_type, bool rotate_activation, void* quant_output,
    void* scale_output, cudaStream_t stream)
{
    if (total_tokens == 0)
    {
        return;
    }

    TLLM_CHECK_WITH_INFO(
        cache_scale_type >= 0 && cache_scale_type <= 3, "Invalid cache_scale_type: %d", cache_scale_type);
    auto const cst = static_cast<CacheScaleType>(cache_scale_type);

    bool const is_quantized_store = (cst != CacheScaleType::kNone);
    int const nthreads = compressorNthreads(head_dim, elem_bytes);
    int const smem_bytes = head_dim * sizeof(float);
    TLLM_CHECK_WITH_INFO(cst != CacheScaleType::kMXFP4Blockwise || head_dim % 32 == 0,
        "MXFP4 cache requires head_dim divisible by 32, got %d", head_dim);
    TLLM_CHECK_WITH_INFO(cst != CacheScaleType::kMXFP4Blockwise || head_dim % 2 == 0,
        "FP4 packed cache requires even head_dim, got %d", head_dim);
    TLLM_CHECK_WITH_INFO(!is_quantized_store || elem_bytes == 2,
        "Quantized cache modes require bf16 compressor output, got elem_bytes=%d", elem_bytes);

    // Derive cache block stride (in bytes) and scale block count from the
    // scale type.  Each physical cache block stores tokens_per_block tokens:
    //   none:             tpb * HD * elem_bytes
    //   fp8 pertensor:    tpb * HD
    //   fp8 blockwise:    tpb * HD + tpb * (HD/128)*4
    //   mxfp4:            tpb * (HD/2) + tpb * (HD/32)
    int num_scale_blocks = 0;
    int cache_stride_blk_bytes = 0;
    switch (cst)
    {
    case CacheScaleType::kFP8PerTensor: cache_stride_blk_bytes = tokens_per_block * head_dim; break;
    case CacheScaleType::kFP8Blockwise:
        num_scale_blocks = head_dim / 128;
        cache_stride_blk_bytes = tokens_per_block * head_dim + tokens_per_block * num_scale_blocks * 4;
        break;
    case CacheScaleType::kMXFP4Blockwise:
        num_scale_blocks = head_dim / 32;
        cache_stride_blk_bytes = tokens_per_block * (head_dim / 2) + tokens_per_block * num_scale_blocks;
        break;
    default: cache_stride_blk_bytes = tokens_per_block * head_dim * elem_bytes; break;
    }

#define LAUNCH_PPS(HD, EB, CST, AR)                                                                                    \
    postProcessScatterKernel<HD, EB, CST, AR><<<total_tokens, nthreads, smem_bytes, stream>>>(kv_comp, kv_out,         \
        rms_weight, rms_eps, cos_sin_table, position_ids, nope_dim, rope_dim, kv_cache, num_outputs, cu_kv_comp,       \
        start_pos, block_offsets, compressed_mask, batch_size, tokens_per_block, max_blocks_per_seq,                   \
        cache_stride_blk_bytes, total_tokens, num_scale_blocks, quant_output, scale_output)

#define DISPATCH_ROTATE(HD, EB, CST)                                                                                   \
    if (rotate_activation)                                                                                             \
    {                                                                                                                  \
        LAUNCH_PPS(HD, EB, CST, true);                                                                                 \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        LAUNCH_PPS(HD, EB, CST, false);                                                                                \
    }
#define DISPATCH_HD_EB(CST)                                                                                            \
    if (elem_bytes == 4)                                                                                               \
    {                                                                                                                  \
        switch (head_dim)                                                                                              \
        {                                                                                                              \
        case 128: DISPATCH_ROTATE(128, 4, CST); break;                                                                 \
        default: DISPATCH_ROTATE(512, 4, CST); break;                                                                  \
        }                                                                                                              \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        switch (head_dim)                                                                                              \
        {                                                                                                              \
        case 128: DISPATCH_ROTATE(128, 2, CST); break;                                                                 \
        default: DISPATCH_ROTATE(512, 2, CST); break;                                                                  \
        }                                                                                                              \
    }
#define DISPATCH_HD_BF16(CST)                                                                                          \
    switch (head_dim)                                                                                                  \
    {                                                                                                                  \
    case 128: DISPATCH_ROTATE(128, 2, CST); break;                                                                     \
    default: DISPATCH_ROTATE(512, 2, CST); break;                                                                      \
    }

    if (cst == CacheScaleType::kFP8PerTensor)
    {
        DISPATCH_HD_BF16(CacheScaleType::kFP8PerTensor);
    }
    else if (cst == CacheScaleType::kFP8Blockwise)
    {
        DISPATCH_HD_BF16(CacheScaleType::kFP8Blockwise);
    }
    else if (cst == CacheScaleType::kMXFP4Blockwise)
    {
        DISPATCH_HD_BF16(CacheScaleType::kMXFP4Blockwise);
    }
    else
    {
        DISPATCH_HD_EB(CacheScaleType::kNone);
    }

#undef DISPATCH_HD_BF16
#undef DISPATCH_HD_EB
#undef DISPATCH_ROTATE
#undef LAUNCH_PPS
}

} // namespace kernels::compressor

TRTLLM_NAMESPACE_END
