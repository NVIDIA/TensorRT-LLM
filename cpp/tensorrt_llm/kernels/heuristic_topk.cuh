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

// ============================================================================
// heuristic_topk.cuh — Heuristic-Guided Top-K (Sort-Free, Histogram-Based)
//
// Outer name: "heuristic" (algorithm family + public dispatcher / launchers).
// Inner name: "gvr" (Guess-Verify-Refine) — the single-CTA single-row
//             micro-kernel implementing the algorithm of:
//   "Guess-Verify-Refine: Data-Aware Top-K for Sparse-Attention Decoding
//    on Blackwell via Temporal Correlation"
//
// Optimised for NVIDIA B200 (Blackwell, sm_100), single thread-block kernel.
//
// GVR phase mapping:
//   P1 (preIdx stats)               ┐  Guess: estimate the K-th-value
//   P2 (secant threshold search)    ┘  threshold from previous-step top-K
//                                      indices, then refine the guess via
//                                      count-only secant iterations.
//   P3 (collect)                    — Verify: scatter the elements that
//                                      pass the guessed threshold into
//                                      shared memory and confirm the
//                                      candidate count is in the safe band.
//   P4 (histogram snap + partition) — Refine: 2048-bin histogram snap to
//                                      the exact K-th value, then partition
//                                      the candidates into the output set.
// ============================================================================

#pragma once

#include <cfloat>
#include <cstdint>
#include <cstdio>      // for snap convergence warning printf
#include <cstdlib>     // for std::getenv (PDL launcher env-gate)
#include <cuda_bf16.h> // 4d: bf16 input dtype support
#include <cuda_fp16.h> // 4d: fp16 input dtype support
#include <cuda_runtime.h>
#include <type_traits> // 4d: std::is_same_v in launchHeuristicTopK dispatch

namespace heuristic_topk
{

// ============================================================================
// Multi-dtype Trait Layer (4d_multi_dtype_unified, 2026-05-06)
// ============================================================================
// Encapsulates dtype-specific cvt intrinsics + vector load width so the
// kernel body can be templated cleanly. For fp32 the path is byte-
// equivalent to the original V2e kernel (VEC_W=4, identity cvt).
//
// Tier-1 fp32 arithmetic (threshold, accumulators, bin index) — kernel-
// internal, not in trait. Tier-2 InputT containers (HBM input, smem keys,
// outputValues) — dtype-driven. Tier-3 ALU stays fp32 (GVR is HBM-bound).

template <typename T>
struct GvrDtypeTraits;

template <>
struct GvrDtypeTraits<float>
{
    using SmemKey = float;
    static constexpr int VEC_W = 4; // int4 = 4 × fp32
    static constexpr int SMEM_KEY_BYTES = 4;

    __device__ static __forceinline__ float to_fp32(float v)
    {
        return v;
    }

    __device__ static __forceinline__ float from_fp32(float v)
    {
        return v;
    }

    __device__ static __forceinline__ void unpack4(int4 raw, float* out)
    {
        out[0] = __int_as_float(raw.x);
        out[1] = __int_as_float(raw.y);
        out[2] = __int_as_float(raw.z);
        out[3] = __int_as_float(raw.w);
    }
};

template <>
struct GvrDtypeTraits<__nv_bfloat16>
{
    using SmemKey = __nv_bfloat16;
    static constexpr int VEC_W = 8; // int4 = 8 × bf16 = 4 × bf162
    static constexpr int SMEM_KEY_BYTES = 2;

    __device__ static __forceinline__ float to_fp32(__nv_bfloat16 v)
    {
        return __bfloat162float(v);
    }

    __device__ static __forceinline__ __nv_bfloat16 from_fp32(float v)
    {
        return __float2bfloat16_rn(v);
    }

    __device__ static __forceinline__ void unpack8(int4 raw, float* out)
    {
        auto* p = reinterpret_cast<__nv_bfloat162*>(&raw);
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            out[2 * j] = __low2float(p[j]);
            out[2 * j + 1] = __high2float(p[j]);
        }
    }
};

template <>
struct GvrDtypeTraits<__half>
{
    using SmemKey = __half;
    static constexpr int VEC_W = 8; // int4 = 8 × fp16 = 4 × half2
    static constexpr int SMEM_KEY_BYTES = 2;

    __device__ static __forceinline__ float to_fp32(__half v)
    {
        return __half2float(v);
    }

    __device__ static __forceinline__ __half from_fp32(float v)
    {
        return __float2half_rn(v);
    }

    __device__ static __forceinline__ void unpack8(int4 raw, float* out)
    {
        auto* p = reinterpret_cast<__half2*>(&raw);
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            out[2 * j] = __low2float(p[j]);
            out[2 * j + 1] = __high2float(p[j]);
        }
    }
};

// ============================================================================
// Configuration Constants
// ============================================================================

constexpr int BLOCK_SIZE = 512;
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

constexpr int TOP_K = 2048;
constexpr int HEURISTIC_SIZE = 2048;
constexpr int SAFETY_MARGIN = 2048;
constexpr int MAX_CANDIDATES = TOP_K + SAFETY_MARGIN * 2; // 6144

constexpr int MAX_REFINE_ITERS = 15;
constexpr int NUM_BINS = 2048;

static_assert(TOP_K % BLOCK_SIZE == 0);
static_assert(MAX_CANDIDATES % BLOCK_SIZE == 0);

// ============================================================================
// Shared Memory Layout
// ============================================================================
// fp32 instantiation (KernelSmem = KernelSmemTpl<float>): ~59 KB
// bf16/fp16 instantiation (KernelSmemTpl<__nv_bfloat16/__half>): ~47 KB
// (4d_multi_dtype_unified, 2026-05-06: templatized on SmemKey for trait path)

template <typename SmemKey>
struct KernelSmemTpl
{
    alignas(16) SmemKey keys[MAX_CANDIDATES]; // 24 KB (fp32) / 12 KB (bf16/fp16)
    alignas(16) int vals[MAX_CANDIDATES];     // 24 KB

    int warp_counts[NUM_WARPS];               // 64 B
    int histogram[NUM_BINS];                  // 8 KB
    int per_thread_counts[BLOCK_SIZE];        // 2 KB — OPT7: cached from last blockCountGE

    float threshold;
    int cand_count;
    int done;

    float val_lo, val_hi;
    int cnt_lo, cnt_hi;

    float pmax_saved;
    int out_count;
};

// fp32 alias preserved so the original gvrTopKJob / gvrTopKKernel /
// blockCountGE / blockFusedSnapIter and the TRT-LLM multi-row caller
// `using heuristic_topk::gvrTopKJob` keep their byte-equivalent signature.
using KernelSmem = KernelSmemTpl<float>;

// ============================================================================
// ★ OPT4: Warp-Level Reduction Primitives
// ============================================================================

#if __CUDA_ARCH__ >= 800

__device__ __forceinline__ int warpReduceSum(int val)
{
    return __reduce_add_sync(0xffffffffu, val);
}

__device__ __forceinline__ unsigned floatToOrderedUint(float f)
{
    unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ float orderedUintToFloat(unsigned u)
{
    return __uint_as_float((u & 0x80000000u) ? (u & ~0x80000000u) : ~u);
}

__device__ __forceinline__ float warpReduceMin(float val)
{
    unsigned u = floatToOrderedUint(val);
    u = __reduce_min_sync(0xffffffffu, u);
    return orderedUintToFloat(u);
}

__device__ __forceinline__ float warpReduceMax(float val)
{
    unsigned u = floatToOrderedUint(val);
    u = __reduce_max_sync(0xffffffffu, u);
    return orderedUintToFloat(u);
}

#else

__device__ __forceinline__ int warpReduceSum(int val)
{
#pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffffu, val, off);
    return val;
}

__device__ __forceinline__ float warpReduceMin(float val)
{
#pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        val = fminf(val, __shfl_xor_sync(0xffffffffu, val, off));
    return val;
}

__device__ __forceinline__ float warpReduceMax(float val)
{
#pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffffu, val, off));
    return val;
}

#endif

// ============================================================================
// Device: Block count ≥ threshold in GLOBAL memory  (1-sync pattern)
// ============================================================================

__device__ __forceinline__ void blockCountGE(
    float const* __restrict__ input, int N, float threshold, KernelSmem* smem, int tid, int warp_id, int lane)
{
    int c = 0;
    for (int i = tid * 4; i + 3 < N; i += BLOCK_SIZE * 4)
    {
        float4 v4 = __ldg(reinterpret_cast<float4 const*>(input + i));
        c += (v4.x >= threshold) + (v4.y >= threshold) + (v4.z >= threshold) + (v4.w >= threshold);
    }
    for (int i = (N & ~3) + tid; i < N; i += BLOCK_SIZE)
        c += (__ldg(&input[i]) >= threshold);

    // OPT7: cache per-thread count for Phase 3 sub-pass 1 reuse
    smem->per_thread_counts[tid] = c;

    c = warpReduceSum(c);

    if (lane == 0)
        smem->warp_counts[warp_id] = c;
    __syncthreads();

    if (tid == 0)
    {
        int t = 0;
        for (int w = 0; w < NUM_WARPS; w++)
            t += smem->warp_counts[w];
        smem->cand_count = t;
    }
}

// ============================================================================
// Fused snap iteration (2 syncs per call)
// ============================================================================

__device__ __forceinline__ void blockFusedSnapIter(KernelSmem* smem, int count, int tid, int warp_id, int lane)
{
    float const thr = smem->threshold;

    int lge = 0, lgt = 0;
    float s_up = FLT_MAX, s_down = -FLT_MAX;

    for (int i = tid; i < count; i += BLOCK_SIZE)
    {
        float v = smem->keys[i];
        lge += (v >= thr);
        lgt += (v > thr);
        if (v > thr)
            s_up = fminf(s_up, v);
        if (v < thr)
            s_down = fmaxf(s_down, v);
    }

    int packed = (lge << 16) | lgt;
    packed = warpReduceSum(packed);
    s_up = warpReduceMin(s_up);
    s_down = warpReduceMax(s_down);

    if (lane == 0)
    {
        smem->warp_counts[warp_id] = packed;
        smem->histogram[warp_id] = __float_as_int(s_up);
        smem->histogram[NUM_WARPS + warp_id] = __float_as_int(s_down);
    }
    __syncthreads();

    if (tid == 0)
    {
        int tp = 0;
        float total_up = FLT_MAX, total_down = -FLT_MAX;
        for (int w = 0; w < NUM_WARPS; w++)
        {
            tp += smem->warp_counts[w];
            total_up = fminf(total_up, __int_as_float(smem->histogram[w]));
            total_down = fmaxf(total_down, __int_as_float(smem->histogram[NUM_WARPS + w]));
        }
        smem->cnt_lo = tp >> 16;
        smem->cnt_hi = tp & 0xFFFF;

        int cge = smem->cnt_lo;
        int cgt = smem->cnt_hi;

        if (cgt >= TOP_K)
        {
            if (total_up < FLT_MAX)
                smem->threshold = total_up;
        }
        else if (cge < TOP_K)
        {
            if (total_down > -FLT_MAX)
                smem->threshold = total_down;
        }
    }
    __syncthreads();
}

// ============================================================================
// Dtype-templated helpers (4d_multi_dtype_unified)
// ============================================================================
// Mirror blockCountGE / blockFusedSnapIter for bf16/fp16 inputs. fp32
// path keeps the originals untouched (byte-equivalent to V2e baseline).
//
// blockCountGEDtype:
//   - 8-wide vector load (int4 = 8 × 16-bit) instead of 4-wide (float4 = 4 × fp32)
//   - up-cast each element to fp32 before threshold compare
//
// blockFusedSnapIterDtype:
//   - reads SmemKey (bf16 / fp16) from smem->keys, up-casts to fp32 for
//     min/max/count work; threshold remains fp32 (Tier-1 invariant).

template <typename InputT>
__device__ __forceinline__ void blockCountGEDtype(InputT const* __restrict__ input, int N, float threshold,
    KernelSmemTpl<typename GvrDtypeTraits<InputT>::SmemKey>* smem, int tid, int warp_id, int lane)
{
    using Trait = GvrDtypeTraits<InputT>;
    static_assert(Trait::VEC_W == 8, "blockCountGEDtype is for bf16/fp16 (8-wide vector); use blockCountGE for fp32");

    int c = 0;
    for (int i = tid * 8; i + 7 < N; i += BLOCK_SIZE * 8)
    {
        int4 raw = __ldg(reinterpret_cast<int4 const*>(input + i));
        float v[8];
        Trait::unpack8(raw, v);
#pragma unroll
        for (int j = 0; j < 8; j++)
            c += (v[j] >= threshold);
    }
    for (int i = (N & ~7) + tid; i < N; i += BLOCK_SIZE)
        c += (Trait::to_fp32(__ldg(&input[i])) >= threshold);

    smem->per_thread_counts[tid] = c;

    c = warpReduceSum(c);

    if (lane == 0)
        smem->warp_counts[warp_id] = c;
    __syncthreads();

    if (tid == 0)
    {
        int t = 0;
        for (int w = 0; w < NUM_WARPS; w++)
            t += smem->warp_counts[w];
        smem->cand_count = t;
    }
}

template <typename SmemKey>
__device__ __forceinline__ void blockFusedSnapIterDtype(
    KernelSmemTpl<SmemKey>* smem, int count, int tid, int warp_id, int lane)
{
    using Trait = GvrDtypeTraits<SmemKey>;

    float const thr = smem->threshold;

    int lge = 0, lgt = 0;
    float s_up = FLT_MAX, s_down = -FLT_MAX;

    for (int i = tid; i < count; i += BLOCK_SIZE)
    {
        float v = Trait::to_fp32(smem->keys[i]);
        lge += (v >= thr);
        lgt += (v > thr);
        if (v > thr)
            s_up = fminf(s_up, v);
        if (v < thr)
            s_down = fmaxf(s_down, v);
    }

    int packed = (lge << 16) | lgt;
    packed = warpReduceSum(packed);
    s_up = warpReduceMin(s_up);
    s_down = warpReduceMax(s_down);

    if (lane == 0)
    {
        smem->warp_counts[warp_id] = packed;
        smem->histogram[warp_id] = __float_as_int(s_up);
        smem->histogram[NUM_WARPS + warp_id] = __float_as_int(s_down);
    }
    __syncthreads();

    if (tid == 0)
    {
        int tp = 0;
        float total_up = FLT_MAX, total_down = -FLT_MAX;
        for (int w = 0; w < NUM_WARPS; w++)
        {
            tp += smem->warp_counts[w];
            total_up = fminf(total_up, __int_as_float(smem->histogram[w]));
            total_down = fmaxf(total_down, __int_as_float(smem->histogram[NUM_WARPS + w]));
        }
        smem->cnt_lo = tp >> 16;
        smem->cnt_hi = tp & 0xFFFF;

        int cge = smem->cnt_lo;
        int cgt = smem->cnt_hi;

        if (cgt >= TOP_K)
        {
            if (total_up < FLT_MAX)
                smem->threshold = total_up;
        }
        else if (cge < TOP_K)
        {
            if (total_down > -FLT_MAX)
                smem->threshold = total_down;
        }
    }
    __syncthreads();
}

// ============================================================================
// Device function: algorithm body (independently optimized by ptxas)
// __noinline__ ensures ptxas allocates registers and schedules instructions
// for this function independently from the caller, matching standalone SASS.
// ============================================================================

__device__ __noinline__ void gvrTopKJob(float const* __restrict__ input, int const N, int const* __restrict__ preIdx,
    int const M, int const topK, float* __restrict__ outputValues, int* __restrict__ outputIndices, KernelSmem* smem,
    int const preIdxOffset = 0)
{
    int const tid = threadIdx.x;
    int const warp_id = tid / WARP_SIZE;
    int const lane = tid & (WARP_SIZE - 1);
    unsigned const full_mask = 0xffffffffu;

    {
        // ================================================================
        // Phase 1 (GVR Guess, part 1) — Min/Max/Mean of pre-indexed values
        // ================================================================

        float local_min = FLT_MAX;
        float local_max = -FLT_MAX;
        float local_sum = 0.0f;
        int local_cnt = 0;
        for (int i = tid; i < M; i += BLOCK_SIZE)
        {
            int idx = __ldg(&preIdx[i]) + preIdxOffset;
            if (idx >= 0 && idx < N)
            {
                float v = __ldg(&input[idx]);
                local_min = fminf(local_min, v);
                local_max = fmaxf(local_max, v);
                local_sum += v;
                local_cnt++;
            }
        }

        float wmin = warpReduceMin(local_min);
        float wmax = warpReduceMax(local_max);
        float wsum = local_sum;
#pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
            wsum += __shfl_down_sync(0xffffffffu, wsum, off);
        int wcnt = warpReduceSum(local_cnt);

        if (lane == 0)
        {
            smem->histogram[warp_id] = __float_as_int(wmin);
            smem->histogram[NUM_WARPS + warp_id] = __float_as_int(wmax);
            smem->histogram[NUM_WARPS * 2 + warp_id] = __float_as_int(wsum);
            smem->histogram[NUM_WARPS * 3 + warp_id] = wcnt;
        }
        __syncthreads();

        if (tid == 0)
        {
            float pmin = FLT_MAX, pmax = -FLT_MAX, psum = 0.0f;
            int pcnt = 0;
            for (int w = 0; w < NUM_WARPS; w++)
            {
                pmin = fminf(pmin, __int_as_float(smem->histogram[w]));
                pmax = fmaxf(pmax, __int_as_float(smem->histogram[NUM_WARPS + w]));
                psum += __int_as_float(smem->histogram[NUM_WARPS * 2 + w]);
                pcnt += smem->histogram[NUM_WARPS * 3 + w];
            }
            float pmean = (pcnt > 0) ? psum / (float) pcnt : (pmin + pmax) * 0.5f;

            smem->pmax_saved = pmax;
            smem->threshold = pmean;
            smem->val_lo = pmin;
            smem->val_hi = pmax;
            smem->cnt_lo = M + M / 4;
            smem->cnt_hi = 1;
            smem->done = 0;
        }
        __syncthreads();

        if (smem->val_hi <= -FLT_MAX || smem->val_lo >= smem->val_hi)
        {
            if (tid == 0)
                for (int i = 0; i < topK && i < N; i++)
                {
                    outputIndices[i] = i;
                    outputValues[i] = input[i];
                }
            return;
        }

        // ================================================================
        // Phase 2 (GVR Guess, part 2) — Secant-interpolation threshold search
        // ================================================================

        blockCountGE(input, N, smem->threshold, smem, tid, warp_id, lane);

        if (tid == 0)
        {
            int c = smem->cand_count;
            if (c >= TOP_K && c <= MAX_CANDIDATES)
                smem->done = 1;
            else if (c > MAX_CANDIDATES)
            {
                smem->val_lo = smem->threshold;
                smem->cnt_lo = c;
            }
            else
            {
                smem->val_hi = smem->threshold;
                smem->cnt_hi = c;
            }
        }
        __syncthreads();

        for (int iter = 0; iter < MAX_REFINE_ITERS; iter++)
        {
            if (smem->done)
                break;
            if (tid == 0)
            {
                float vlo = smem->val_lo, vhi = smem->val_hi;
                int clo = smem->cnt_lo, chi = smem->cnt_hi;
                int target = TOP_K + SAFETY_MARGIN / 2;
                float range = vhi - vlo;
                float nv;
                if (clo > chi && range > 1e-10f)
                {
                    float f = (float) (clo - target) / (float) (clo - chi);
                    f = fmaxf(0.05f, fminf(0.95f, f));
                    if (iter == 0)
                        f = fminf(f, 0.50f);
                    nv = vlo + range * f;
                }
                else
                    nv = (vlo + vhi) * 0.5f;
                if (nv <= vlo)
                    nv = vlo + range * 0.05f;
                if (nv >= vhi)
                    nv = vhi - range * 0.05f;
                if (nv == vlo || nv == vhi)
                {
                    nv = (vlo + vhi) * 0.5f;
                    if (nv == vlo || nv == vhi)
                    {
                        smem->threshold = vlo;
                        smem->done = 2;
                    }
                    else
                        smem->threshold = nv;
                }
                else
                    smem->threshold = nv;
            }
            __syncthreads();
            if (smem->done)
                break;
            blockCountGE(input, N, smem->threshold, smem, tid, warp_id, lane);
            if (tid == 0)
            {
                int c = smem->cand_count;
                if (c >= TOP_K && c <= MAX_CANDIDATES)
                    smem->done = 1;
                else if (c > MAX_CANDIDATES)
                {
                    smem->val_lo = smem->threshold;
                    smem->cnt_lo = c;
                }
                else
                {
                    smem->val_hi = smem->threshold;
                    smem->cnt_hi = c;
                }
            }
            __syncthreads();
        }

        if (tid == 0 && !smem->done)
        {
            if (smem->cnt_lo <= MAX_CANDIDATES * 2)
                smem->threshold = smem->val_lo;
            else
                smem->threshold = smem->val_hi;
            smem->done = 2;
        }
        __syncthreads();
    } // end of P1+P2 scope

    // ================================================================
    // Phase 3 (GVR Verify) — Ballot-free candidate collect
    // ================================================================

    // OPT5: when done==1, Phase 2 already verified count in
    //        [TOP_K, MAX_CANDIDATES]; skip the redundant full-N
    //        blockCountGE re-check entirely.
    if (smem->done != 1)
    {
        blockCountGE(input, N, smem->threshold, smem, tid, warp_id, lane);
        if (tid == 0 && smem->cand_count > MAX_CANDIDATES)
            smem->val_lo = smem->threshold;
        __syncthreads();

        for (int retry = 0; retry < 10 && smem->cand_count > MAX_CANDIDATES; retry++)
        {
            if (tid == 0)
            {
                float lo = smem->val_lo, hi = smem->val_hi;
                float mid = (lo + hi) * 0.5f;
                if (mid == lo)
                    mid = hi;
                smem->threshold = mid;
            }
            __syncthreads();
            blockCountGE(input, N, smem->threshold, smem, tid, warp_id, lane);
            if (tid == 0)
            {
                int c = smem->cand_count;
                if (c > MAX_CANDIDATES)
                    smem->val_lo = smem->threshold;
                else if (c < TOP_K)
                    smem->val_hi = smem->threshold;
            }
            __syncthreads();
        }
    }

    // OPT7: reuse per-thread counts cached by the last blockCountGE call;
    //        saves one full N-scan (blockCountGE's __syncthreads guarantees visibility).
    int my_total_qual = smem->per_thread_counts[tid];

    int thread_prefix = my_total_qual;
#pragma unroll
    for (int off = 1; off < WARP_SIZE; off *= 2)
    {
        int other = __shfl_up_sync(full_mask, thread_prefix, off);
        if (lane >= off)
            thread_prefix += other;
    }
    int my_excl_offset = thread_prefix - my_total_qual;
    int warp_total_qual = __shfl_sync(full_mask, thread_prefix, WARP_SIZE - 1);

    if (lane == 0)
        smem->warp_counts[warp_id] = warp_total_qual;
    __syncthreads();

    if (tid == 0)
    {
        int total = 0;
        for (int w = 0; w < NUM_WARPS; w++)
        {
            int cnt = smem->warp_counts[w];
            smem->warp_counts[w] = total;
            total += cnt;
        }
        smem->cand_count = total;
    }
    __syncthreads();

    int my_write_pos = smem->warp_counts[warp_id] + my_excl_offset;

    {
        float const thr = smem->threshold;
        for (int i = tid * 4; i + 3 < N; i += BLOCK_SIZE * 4)
        {
            float4 v4 = __ldg(reinterpret_cast<float4 const*>(input + i));
#pragma unroll
            for (int j = 0; j < 4; j++)
            {
                float val = (&v4.x)[j];
                if (val >= thr && my_write_pos < MAX_CANDIDATES)
                {
                    smem->keys[my_write_pos] = val;
                    smem->vals[my_write_pos] = i + j;
                    my_write_pos++;
                }
            }
        }
        for (int i = (N & ~3) + tid; i < N; i += BLOCK_SIZE)
        {
            float val = __ldg(&input[i]);
            if (val >= thr && my_write_pos < MAX_CANDIDATES)
            {
                smem->keys[my_write_pos] = val;
                smem->vals[my_write_pos] = i;
                my_write_pos++;
            }
        }
    }
    __syncthreads();

    // ================================================================
    // Phase 4 (GVR Refine) — Histogram-based selection + partition
    // ================================================================

    int const cand_count = min(smem->cand_count, MAX_CANDIDATES);

    if (cand_count == TOP_K)
    {
        for (int i = tid; i < TOP_K; i += BLOCK_SIZE)
        {
            outputValues[i] = smem->keys[i];
            outputIndices[i] = smem->vals[i];
        }
        return;
    }

    if (cand_count > TOP_K)
    {
        float cmin = FLT_MAX, cmax = -FLT_MAX;
        for (int i = tid; i < cand_count; i += BLOCK_SIZE)
        {
            float v = smem->keys[i];
            cmin = fminf(cmin, v);
            cmax = fmaxf(cmax, v);
        }
        cmin = warpReduceMin(cmin);
        cmax = warpReduceMax(cmax);
        if (lane == 0)
        {
            smem->warp_counts[warp_id] = __float_as_int(cmin);
            smem->histogram[warp_id] = __float_as_int(cmax);
        }
        __syncthreads();

        float block_min = FLT_MAX, block_max = -FLT_MAX;
        for (int w = 0; w < NUM_WARPS; w++)
        {
            block_min = fminf(block_min, __int_as_float(smem->warp_counts[w]));
            block_max = fmaxf(block_max, __int_as_float(smem->histogram[w]));
        }
        if (block_max <= block_min)
            block_max = block_min + 1e-6f;

        for (int i = tid; i < NUM_BINS; i += BLOCK_SIZE)
            smem->histogram[i] = 0;
        __syncthreads();

        float range1 = block_max - block_min;
        float inv1 = (range1 > 0.0f) ? ((float) (NUM_BINS - 1) + 0.99f) / range1 : 0.0f;

        for (int i = tid; i < cand_count; i += BLOCK_SIZE)
        {
            int bin = (int) ((smem->keys[i] - block_min) * inv1);
            bin = min(max(bin, 0), NUM_BINS - 1);
            atomicAdd(&smem->histogram[bin], 1);
        }
        __syncthreads();

        // OPT6: Parallel K-th bin search (2-step).
        // Each warp sums BINS_PER_WARP consecutive bins (high→low); tid=0 locates the
        // target warp in NUM_WARPS steps; one thread in that warp scans BINS_PER_WARP bins.
        // Total serial depth: NUM_WARPS + BINS_PER_WARP = 16 + 128 = 144 steps vs 2048.
        {
            constexpr int BINS_PER_WARP = NUM_BINS / NUM_WARPS;
            static_assert(NUM_BINS % NUM_WARPS == 0, "NUM_BINS must be divisible by NUM_WARPS");
            // Step 1: each warp accumulates its slice of bins (high→low)
            int warp_bin_sum = 0;
            for (int j = 0; j < BINS_PER_WARP; j++)
                warp_bin_sum += smem->histogram[NUM_BINS - 1 - warp_id * BINS_PER_WARP - j];
            if (lane == 0)
                smem->warp_counts[warp_id] = warp_bin_sum;
        }
        __syncthreads(); // S-4b3a

        // Step 2: tid=0 finds which warp contains the K-th element
        if (tid == 0)
        {
            int cum = 0, tw = NUM_WARPS - 1;
            for (int w = 0; w < NUM_WARPS; w++)
            {
                cum += smem->warp_counts[w];
                if (cum >= TOP_K)
                {
                    tw = w;
                    break;
                }
            }
            // Recompute prefix before target warp for step 3
            cum = 0;
            for (int w = 0; w < tw; w++)
                cum += smem->warp_counts[w];
            smem->cnt_lo = cum; // prefix count before target warp
            smem->cnt_hi = tw;  // target warp index
        }
        __syncthreads();        // S-4b3b

        // Step 3: one thread in target warp scans its BINS_PER_WARP bins
        if (warp_id == smem->cnt_hi && lane == 0)
        {
            constexpr int BINS_PER_WARP = NUM_BINS / NUM_WARPS;
            int base_cum = smem->cnt_lo;
            float thr = block_min;
            for (int j = 0; j < BINS_PER_WARP; j++)
            {
                int b = NUM_BINS - 1 - smem->cnt_hi * BINS_PER_WARP - j;
                base_cum += smem->histogram[b];
                if (base_cum >= TOP_K)
                {
                    thr = block_min + (float) b * range1 / (float) NUM_BINS;
                    break;
                }
            }
            smem->threshold = thr;
        }
        __syncthreads(); // S-4b3c

        bool snap_converged = false;
        int snap_limit = (cand_count > 128 ? cand_count / 4 : 32);
        for (int si = 0; si < snap_limit; si++)
        {
            blockFusedSnapIter(smem, cand_count, tid, warp_id, lane);
            int cge = smem->cnt_lo;
            int cgt = smem->cnt_hi;
            if (cgt < TOP_K && cge >= TOP_K)
            {
                snap_converged = true;
                break;
            }
        }
        (void) snap_converged;

        float sel_thr = smem->threshold;
        if (tid == 0)
            smem->out_count = 0;
        __syncthreads();

        // Opt-M fix: separate gt and eq into two sequential passes so that
        // strictly-greater values are never dropped in favor of tie-values.
        // The v0 interleaved version had a correctness bug when ties at the
        // K-th value cross TOP_K; this fix makes the selection rank-stable.

        // Pass 1: strictly greater than sel_thr
        for (int base = warp_id * WARP_SIZE; base < cand_count; base += BLOCK_SIZE)
        {
            int i = base + lane;
            float v = (i < cand_count) ? smem->keys[i] : -FLT_MAX;

            bool emit_gt = (i < cand_count) && (v > sel_thr);
            unsigned mask_gt = __ballot_sync(full_mask, emit_gt);
            if (mask_gt)
            {
                int cnt = __popc(mask_gt);
                int moff = __popc(mask_gt & ((1u << lane) - 1u));
                int bp = 0;
                if (lane == 0)
                    bp = atomicAdd(&smem->out_count, cnt);
                bp = __shfl_sync(full_mask, bp, 0);
                if (emit_gt && bp + moff < TOP_K)
                {
                    outputValues[bp + moff] = v;
                    outputIndices[bp + moff] = smem->vals[i];
                }
            }
        }
        __syncthreads();

        // Pass 2: equal to sel_thr (fills remaining slots)
        for (int base = warp_id * WARP_SIZE; base < cand_count; base += BLOCK_SIZE)
        {
            int i = base + lane;
            float v = (i < cand_count) ? smem->keys[i] : -FLT_MAX;

            bool emit_eq = (i < cand_count) && (v == sel_thr);
            unsigned mask_eq = __ballot_sync(full_mask, emit_eq);
            if (mask_eq)
            {
                int cnt = __popc(mask_eq);
                int moff = __popc(mask_eq & ((1u << lane) - 1u));
                int bp = 0;
                if (lane == 0)
                    bp = atomicAdd(&smem->out_count, cnt);
                bp = __shfl_sync(full_mask, bp, 0);
                if (emit_eq && bp + moff < TOP_K)
                {
                    outputValues[bp + moff] = v;
                    outputIndices[bp + moff] = smem->vals[i];
                }
            }
        }
        __syncthreads();

        int filled = min(smem->out_count, TOP_K);
        for (int i = filled + tid; i < TOP_K; i += BLOCK_SIZE)
        {
            outputValues[i] = -FLT_MAX;
            outputIndices[i] = -1;
        }
        return;
    }

    for (int i = tid; i < cand_count; i += BLOCK_SIZE)
    {
        outputValues[i] = smem->keys[i];
        outputIndices[i] = smem->vals[i];
    }
    for (int i = cand_count + tid; i < TOP_K; i += BLOCK_SIZE)
    {
        outputValues[i] = -FLT_MAX;
        outputIndices[i] = -1;
    }
}

// ============================================================================
// gvrTopKKernel — single-row global wrapper (1 CTA, 1 row).
// Calls gvrTopKJob (independently-optimized device function).
// For multi-row decode launches, see heuristicTopKMultiRowKernel in
// heuristicTopKDecode.cu — both share the same micro-kernel job.
// ============================================================================

__global__ void __launch_bounds__(BLOCK_SIZE, 1)
    gvrTopKKernel(float const* __restrict__ input, int const N, int const* __restrict__ preIdx, int const M,
        int const topK, float* __restrict__ outputValues, int* __restrict__ outputIndices, int const thresholdPos)
{
    extern __shared__ unsigned char smem_raw[];
    auto* smem = reinterpret_cast<KernelSmem*>(smem_raw);

    gvrTopKJob(input, N, preIdx, M, topK, outputValues, outputIndices, smem, /*preIdxOffset=*/0);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// ============================================================================
// gvrTopKJobDtype<InputT> — bf16/fp16 device function (4d_multi_dtype_unified)
// ============================================================================
// Mirror of gvrTopKJob with trait-driven dtype substitutions:
//   - HBM input read   : Trait::to_fp32(__ldg(&input[i]))
//   - smem keys write  : Trait::from_fp32(v)   (Tier-2 SmemKey container)
//   - smem keys read   : Trait::to_fp32(keys[i]) for arithmetic
//   - outputValues     : InputT (SmemKey == InputT in templated path)
//   - vector load      : 8-wide via Trait::unpack8
//   - blockCountGE     : blockCountGEDtype<InputT>
//   - blockFusedSnapIter: blockFusedSnapIterDtype<SmemKey>
// All Tier-1 arithmetic stays fp32 (threshold, accumulators, bin index).
// fp32 path is NOT routed here — fp32 keeps the original gvrTopKJob (option A,
// guaranteed zero-regression). This template only instantiates for bf16/fp16.

template <typename InputT>
__device__ __noinline__ void gvrTopKJobDtype(InputT const* __restrict__ input, int const N,
    int const* __restrict__ preIdx, int const M, int const topK, InputT* __restrict__ outputValues,
    int* __restrict__ outputIndices, KernelSmemTpl<typename GvrDtypeTraits<InputT>::SmemKey>* smem,
    int const preIdxOffset = 0)
{
    using Trait = GvrDtypeTraits<InputT>;
    using SmemKey = typename Trait::SmemKey;
    static_assert(Trait::VEC_W == 8, "gvrTopKJobDtype is for bf16/fp16 (8-wide); fp32 uses gvrTopKJob");

    int const tid = threadIdx.x;
    int const warp_id = tid / WARP_SIZE;
    int const lane = tid & (WARP_SIZE - 1);
    unsigned const full_mask = 0xffffffffu;

    {
        // ================================================================
        // Phase 1 — Min/Max/Mean of pre-indexed values
        // ================================================================

        float local_min = FLT_MAX;
        float local_max = -FLT_MAX;
        float local_sum = 0.0f;
        int local_cnt = 0;
        for (int i = tid; i < M; i += BLOCK_SIZE)
        {
            int idx = __ldg(&preIdx[i]) + preIdxOffset;
            if (idx >= 0 && idx < N)
            {
                float v = Trait::to_fp32(__ldg(&input[idx]));
                local_min = fminf(local_min, v);
                local_max = fmaxf(local_max, v);
                local_sum += v;
                local_cnt++;
            }
        }

        float wmin = warpReduceMin(local_min);
        float wmax = warpReduceMax(local_max);
        float wsum = local_sum;
#pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
            wsum += __shfl_down_sync(0xffffffffu, wsum, off);
        int wcnt = warpReduceSum(local_cnt);

        if (lane == 0)
        {
            smem->histogram[warp_id] = __float_as_int(wmin);
            smem->histogram[NUM_WARPS + warp_id] = __float_as_int(wmax);
            smem->histogram[NUM_WARPS * 2 + warp_id] = __float_as_int(wsum);
            smem->histogram[NUM_WARPS * 3 + warp_id] = wcnt;
        }
        __syncthreads();

        if (tid == 0)
        {
            float pmin = FLT_MAX, pmax = -FLT_MAX, psum = 0.0f;
            int pcnt = 0;
            for (int w = 0; w < NUM_WARPS; w++)
            {
                pmin = fminf(pmin, __int_as_float(smem->histogram[w]));
                pmax = fmaxf(pmax, __int_as_float(smem->histogram[NUM_WARPS + w]));
                psum += __int_as_float(smem->histogram[NUM_WARPS * 2 + w]);
                pcnt += smem->histogram[NUM_WARPS * 3 + w];
            }
            float pmean = (pcnt > 0) ? psum / (float) pcnt : (pmin + pmax) * 0.5f;

            smem->pmax_saved = pmax;
            smem->threshold = pmean;
            smem->val_lo = pmin;
            smem->val_hi = pmax;
            smem->cnt_lo = M + M / 4;
            smem->cnt_hi = 1;
            smem->done = 0;
        }
        __syncthreads();

        if (smem->val_hi <= -FLT_MAX || smem->val_lo >= smem->val_hi)
        {
            if (tid == 0)
                for (int i = 0; i < topK && i < N; i++)
                {
                    outputIndices[i] = i;
                    outputValues[i] = __ldg(&input[i]); // both InputT, no convert
                }
            return;
        }

        // ================================================================
        // Phase 2 — Secant-interpolation threshold search
        // ================================================================

        blockCountGEDtype<InputT>(input, N, smem->threshold, smem, tid, warp_id, lane);

        if (tid == 0)
        {
            int c = smem->cand_count;
            if (c >= TOP_K && c <= MAX_CANDIDATES)
                smem->done = 1;
            else if (c > MAX_CANDIDATES)
            {
                smem->val_lo = smem->threshold;
                smem->cnt_lo = c;
            }
            else
            {
                smem->val_hi = smem->threshold;
                smem->cnt_hi = c;
            }
        }
        __syncthreads();

        for (int iter = 0; iter < MAX_REFINE_ITERS; iter++)
        {
            if (smem->done)
                break;
            if (tid == 0)
            {
                float vlo = smem->val_lo, vhi = smem->val_hi;
                int clo = smem->cnt_lo, chi = smem->cnt_hi;
                int target = TOP_K + SAFETY_MARGIN / 2;
                float range = vhi - vlo;
                float nv;
                if (clo > chi && range > 1e-10f)
                {
                    float f = (float) (clo - target) / (float) (clo - chi);
                    f = fmaxf(0.05f, fminf(0.95f, f));
                    if (iter == 0)
                        f = fminf(f, 0.50f);
                    nv = vlo + range * f;
                }
                else
                    nv = (vlo + vhi) * 0.5f;
                if (nv <= vlo)
                    nv = vlo + range * 0.05f;
                if (nv >= vhi)
                    nv = vhi - range * 0.05f;
                if (nv == vlo || nv == vhi)
                {
                    nv = (vlo + vhi) * 0.5f;
                    if (nv == vlo || nv == vhi)
                    {
                        smem->threshold = vlo;
                        smem->done = 2;
                    }
                    else
                        smem->threshold = nv;
                }
                else
                    smem->threshold = nv;
            }
            __syncthreads();
            if (smem->done)
                break;
            blockCountGEDtype<InputT>(input, N, smem->threshold, smem, tid, warp_id, lane);
            if (tid == 0)
            {
                int c = smem->cand_count;
                if (c >= TOP_K && c <= MAX_CANDIDATES)
                    smem->done = 1;
                else if (c > MAX_CANDIDATES)
                {
                    smem->val_lo = smem->threshold;
                    smem->cnt_lo = c;
                }
                else
                {
                    smem->val_hi = smem->threshold;
                    smem->cnt_hi = c;
                }
            }
            __syncthreads();
        }

        if (tid == 0 && !smem->done)
        {
            if (smem->cnt_lo <= MAX_CANDIDATES * 2)
                smem->threshold = smem->val_lo;
            else
                smem->threshold = smem->val_hi;
            smem->done = 2;
        }
        __syncthreads();
    } // end of P1+P2 scope

    // ================================================================
    // Phase 3 — Ballot-free candidate collect
    // ================================================================

    if (smem->done != 1)
    {
        blockCountGEDtype<InputT>(input, N, smem->threshold, smem, tid, warp_id, lane);
        if (tid == 0 && smem->cand_count > MAX_CANDIDATES)
            smem->val_lo = smem->threshold;
        __syncthreads();

        for (int retry = 0; retry < 10 && smem->cand_count > MAX_CANDIDATES; retry++)
        {
            if (tid == 0)
            {
                float lo = smem->val_lo, hi = smem->val_hi;
                float mid = (lo + hi) * 0.5f;
                if (mid == lo)
                    mid = hi;
                smem->threshold = mid;
            }
            __syncthreads();
            blockCountGEDtype<InputT>(input, N, smem->threshold, smem, tid, warp_id, lane);
            if (tid == 0)
            {
                int c = smem->cand_count;
                if (c > MAX_CANDIDATES)
                    smem->val_lo = smem->threshold;
                else if (c < TOP_K)
                    smem->val_hi = smem->threshold;
            }
            __syncthreads();
        }
    }

    int my_total_qual = smem->per_thread_counts[tid];

    int thread_prefix = my_total_qual;
#pragma unroll
    for (int off = 1; off < WARP_SIZE; off *= 2)
    {
        int other = __shfl_up_sync(full_mask, thread_prefix, off);
        if (lane >= off)
            thread_prefix += other;
    }
    int my_excl_offset = thread_prefix - my_total_qual;
    int warp_total_qual = __shfl_sync(full_mask, thread_prefix, WARP_SIZE - 1);

    if (lane == 0)
        smem->warp_counts[warp_id] = warp_total_qual;
    __syncthreads();

    if (tid == 0)
    {
        int total = 0;
        for (int w = 0; w < NUM_WARPS; w++)
        {
            int cnt = smem->warp_counts[w];
            smem->warp_counts[w] = total;
            total += cnt;
        }
        smem->cand_count = total;
    }
    __syncthreads();

    int my_write_pos = smem->warp_counts[warp_id] + my_excl_offset;

    {
        float const thr = smem->threshold;
        // 8-wide vector load (int4 = 8 × bf16/fp16)
        for (int i = tid * 8; i + 7 < N; i += BLOCK_SIZE * 8)
        {
            int4 raw = __ldg(reinterpret_cast<int4 const*>(input + i));
            float v[8];
            Trait::unpack8(raw, v);
#pragma unroll
            for (int j = 0; j < 8; j++)
            {
                float val = v[j];
                if (val >= thr && my_write_pos < MAX_CANDIDATES)
                {
                    smem->keys[my_write_pos] = Trait::from_fp32(val);
                    smem->vals[my_write_pos] = i + j;
                    my_write_pos++;
                }
            }
        }
        // Tail loop (N % 8)
        for (int i = (N & ~7) + tid; i < N; i += BLOCK_SIZE)
        {
            float val = Trait::to_fp32(__ldg(&input[i]));
            if (val >= thr && my_write_pos < MAX_CANDIDATES)
            {
                smem->keys[my_write_pos] = Trait::from_fp32(val);
                smem->vals[my_write_pos] = i;
                my_write_pos++;
            }
        }
    }
    __syncthreads();

    // ================================================================
    // Phase 4 — Histogram-based selection + partition
    // ================================================================

    int const cand_count = min(smem->cand_count, MAX_CANDIDATES);

    if (cand_count == TOP_K)
    {
        for (int i = tid; i < TOP_K; i += BLOCK_SIZE)
        {
            outputValues[i] = smem->keys[i]; // both InputT, no convert
            outputIndices[i] = smem->vals[i];
        }
        return;
    }

    if (cand_count > TOP_K)
    {
        float cmin = FLT_MAX, cmax = -FLT_MAX;
        for (int i = tid; i < cand_count; i += BLOCK_SIZE)
        {
            float v = Trait::to_fp32(smem->keys[i]);
            cmin = fminf(cmin, v);
            cmax = fmaxf(cmax, v);
        }
        cmin = warpReduceMin(cmin);
        cmax = warpReduceMax(cmax);
        if (lane == 0)
        {
            smem->warp_counts[warp_id] = __float_as_int(cmin);
            smem->histogram[warp_id] = __float_as_int(cmax);
        }
        __syncthreads();

        float block_min = FLT_MAX, block_max = -FLT_MAX;
        for (int w = 0; w < NUM_WARPS; w++)
        {
            block_min = fminf(block_min, __int_as_float(smem->warp_counts[w]));
            block_max = fmaxf(block_max, __int_as_float(smem->histogram[w]));
        }
        if (block_max <= block_min)
            block_max = block_min + 1e-6f;

        for (int i = tid; i < NUM_BINS; i += BLOCK_SIZE)
            smem->histogram[i] = 0;
        __syncthreads();

        float range1 = block_max - block_min;
        float inv1 = (range1 > 0.0f) ? ((float) (NUM_BINS - 1) + 0.99f) / range1 : 0.0f;

        for (int i = tid; i < cand_count; i += BLOCK_SIZE)
        {
            int bin = (int) ((Trait::to_fp32(smem->keys[i]) - block_min) * inv1);
            bin = min(max(bin, 0), NUM_BINS - 1);
            atomicAdd(&smem->histogram[bin], 1);
        }
        __syncthreads();

        // Parallel K-th bin search (2-step)
        {
            constexpr int BINS_PER_WARP = NUM_BINS / NUM_WARPS;
            static_assert(NUM_BINS % NUM_WARPS == 0, "NUM_BINS must be divisible by NUM_WARPS");
            int warp_bin_sum = 0;
            for (int j = 0; j < BINS_PER_WARP; j++)
                warp_bin_sum += smem->histogram[NUM_BINS - 1 - warp_id * BINS_PER_WARP - j];
            if (lane == 0)
                smem->warp_counts[warp_id] = warp_bin_sum;
        }
        __syncthreads();

        if (tid == 0)
        {
            int cum = 0, tw = NUM_WARPS - 1;
            for (int w = 0; w < NUM_WARPS; w++)
            {
                cum += smem->warp_counts[w];
                if (cum >= TOP_K)
                {
                    tw = w;
                    break;
                }
            }
            cum = 0;
            for (int w = 0; w < tw; w++)
                cum += smem->warp_counts[w];
            smem->cnt_lo = cum;
            smem->cnt_hi = tw;
        }
        __syncthreads();

        if (warp_id == smem->cnt_hi && lane == 0)
        {
            constexpr int BINS_PER_WARP = NUM_BINS / NUM_WARPS;
            int base_cum = smem->cnt_lo;
            float thr = block_min;
            for (int j = 0; j < BINS_PER_WARP; j++)
            {
                int b = NUM_BINS - 1 - smem->cnt_hi * BINS_PER_WARP - j;
                base_cum += smem->histogram[b];
                if (base_cum >= TOP_K)
                {
                    thr = block_min + (float) b * range1 / (float) NUM_BINS;
                    break;
                }
            }
            smem->threshold = thr;
        }
        __syncthreads();

        bool snap_converged = false;
        int snap_limit = (cand_count > 128 ? cand_count / 4 : 32);
        for (int si = 0; si < snap_limit; si++)
        {
            blockFusedSnapIterDtype<SmemKey>(smem, cand_count, tid, warp_id, lane);
            int cge = smem->cnt_lo;
            int cgt = smem->cnt_hi;
            if (cgt < TOP_K && cge >= TOP_K)
            {
                snap_converged = true;
                break;
            }
        }
        (void) snap_converged;

        float sel_thr = smem->threshold;
        if (tid == 0)
            smem->out_count = 0;
        __syncthreads();

        // Pass 1: strictly greater than sel_thr
        for (int base = warp_id * WARP_SIZE; base < cand_count; base += BLOCK_SIZE)
        {
            int i = base + lane;
            float v = (i < cand_count) ? Trait::to_fp32(smem->keys[i]) : -FLT_MAX;

            bool emit_gt = (i < cand_count) && (v > sel_thr);
            unsigned mask_gt = __ballot_sync(full_mask, emit_gt);
            if (mask_gt)
            {
                int cnt = __popc(mask_gt);
                int moff = __popc(mask_gt & ((1u << lane) - 1u));
                int bp = 0;
                if (lane == 0)
                    bp = atomicAdd(&smem->out_count, cnt);
                bp = __shfl_sync(full_mask, bp, 0);
                if (emit_gt && bp + moff < TOP_K)
                {
                    outputValues[bp + moff] = Trait::from_fp32(v);
                    outputIndices[bp + moff] = smem->vals[i];
                }
            }
        }
        __syncthreads();

        // Pass 2: equal to sel_thr (fills remaining slots)
        for (int base = warp_id * WARP_SIZE; base < cand_count; base += BLOCK_SIZE)
        {
            int i = base + lane;
            float v = (i < cand_count) ? Trait::to_fp32(smem->keys[i]) : -FLT_MAX;

            bool emit_eq = (i < cand_count) && (v == sel_thr);
            unsigned mask_eq = __ballot_sync(full_mask, emit_eq);
            if (mask_eq)
            {
                int cnt = __popc(mask_eq);
                int moff = __popc(mask_eq & ((1u << lane) - 1u));
                int bp = 0;
                if (lane == 0)
                    bp = atomicAdd(&smem->out_count, cnt);
                bp = __shfl_sync(full_mask, bp, 0);
                if (emit_eq && bp + moff < TOP_K)
                {
                    outputValues[bp + moff] = Trait::from_fp32(v);
                    outputIndices[bp + moff] = smem->vals[i];
                }
            }
        }
        __syncthreads();

        int filled = min(smem->out_count, TOP_K);
        InputT const neg_max = Trait::from_fp32(-FLT_MAX);
        for (int i = filled + tid; i < TOP_K; i += BLOCK_SIZE)
        {
            outputValues[i] = neg_max;
            outputIndices[i] = -1;
        }
        return;
    }

    // cand_count < TOP_K fallback
    for (int i = tid; i < cand_count; i += BLOCK_SIZE)
    {
        outputValues[i] = smem->keys[i]; // both InputT
        outputIndices[i] = smem->vals[i];
    }
    InputT const neg_max = Trait::from_fp32(-FLT_MAX);
    for (int i = cand_count + tid; i < TOP_K; i += BLOCK_SIZE)
    {
        outputValues[i] = neg_max;
        outputIndices[i] = -1;
    }
}

// ============================================================================
// gvrTopKKernelDtype<InputT> — bf16/fp16 single-row global wrapper
// ============================================================================

template <typename InputT>
__global__ void __launch_bounds__(BLOCK_SIZE, 1)
    gvrTopKKernelDtype(InputT const* __restrict__ input, int const N, int const* __restrict__ preIdx, int const M,
        int const topK, InputT* __restrict__ outputValues, int* __restrict__ outputIndices, int const thresholdPos)
{
    using SmemKey = typename GvrDtypeTraits<InputT>::SmemKey;
    extern __shared__ unsigned char smem_raw[];
    auto* smem = reinterpret_cast<KernelSmemTpl<SmemKey>*>(smem_raw);

    gvrTopKJobDtype<InputT>(input, N, preIdx, M, topK, outputValues, outputIndices, smem,
        /*preIdxOffset=*/0);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// ============================================================================
// Launch Wrapper
// ============================================================================

template <typename T, typename IdxT = int>
cudaError_t launchHeuristicTopK(T const* input, int N, IdxT const* preIdx, int M, int topK, T* outputValues,
    IdxT* outputIndices, cudaStream_t stream = 0, int thresholdPos = -1)
{
    static_assert(sizeof(IdxT) == sizeof(int), "launchHeuristicTopK only supports 32-bit indices");
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, __half>,
        "launchHeuristicTopK supports only fp32 / bf16 / fp16");

    if (topK != TOP_K)
        return cudaErrorInvalidValue;

    // 4d: smem size depends on InputT. fp32 path uses the original
    // KernelSmem (~59 KB); bf16/fp16 use KernelSmemTpl<T> (~47 KB).
    using SmemKey = typename GvrDtypeTraits<T>::SmemKey;
    size_t smemSize = sizeof(KernelSmemTpl<SmemKey>);

    // Resolve target kernel function pointer at compile time.
    auto kfn = []()
    {
        if constexpr (std::is_same_v<T, float>)
            return gvrTopKKernel;
        else
            return gvrTopKKernelDtype<T>;
    }();

    // Opt-in to extended shared memory. cudaFuncSetAttribute is device-scoped
    // and cheap — call unconditionally to be safe across multi-GPU processes.
    if (smemSize > 48u * 1024u)
    {
        int device;
        cudaGetDevice(&device);
        int maxSmem;
        cudaDeviceGetAttribute(&maxSmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        if (smemSize > static_cast<size_t>(maxSmem))
            return cudaErrorInvalidConfiguration;
        cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smemSize));
    }

    // Use cudaLaunchKernelEx with PDL attribute so the kernel epilogue's
    // cudaTriggerProgrammaticLaunchCompletion() takes effect on Hopper+. The
    // attribute is a no-op on pre-Hopper architectures. Honor the standard
    // TRTLLM_ENABLE_PDL env var (default on; set "0" to disable) without
    // depending on TRT-LLM common headers, since this launcher is also reused
    // by the standalone JIT-compiled PyTorch extension under
    // ablation_study/gvr_phase_timing/.
    bool enablePDL = true;
    if (char const* env = std::getenv("TRTLLM_ENABLE_PDL"))
    {
        if (env[0] == '0' && env[1] == '\0')
            enablePDL = false;
    }

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(1);
    config.blockDim = dim3(BLOCK_SIZE);
    config.dynamicSmemBytes = smemSize;
    config.stream = stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = enablePDL ? 1 : 0;
    config.attrs = attrs;
    config.numAttrs = 1;

    cudaLaunchKernelEx(&config, kfn, input, N, preIdx, M, topK, outputValues, outputIndices, thresholdPos);

    return cudaGetLastError();
}

// Explicit instantiations — fp32 (V2e baseline path) + bf16/fp16 (4d dtype path)
template cudaError_t launchHeuristicTopK<float, int>(
    float const*, int, int const*, int, int, float*, int*, cudaStream_t, int);
template cudaError_t launchHeuristicTopK<__nv_bfloat16, int>(
    __nv_bfloat16 const*, int, int const*, int, int, __nv_bfloat16*, int*, cudaStream_t, int);
template cudaError_t launchHeuristicTopK<__half, int>(
    __half const*, int, int const*, int, int, __half*, int*, cudaStream_t, int);

} // namespace heuristic_topk
