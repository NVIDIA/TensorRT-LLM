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
#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace heuristic_topk
{

// ============================================================================
// Multi-dtype Trait Layer
// ============================================================================
// Encapsulates dtype-specific cvt intrinsics + vector load width so the
// kernel body can be templated cleanly. For fp32 the trait is identity.
//
// Arithmetic (threshold, accumulators, bin index) is always fp32; only
// the HBM input container, smem keys, and output values follow InputT.
// GVR is HBM-bandwidth-bound, so fp32 ALU has no measurable cost.

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
// Multi-K Trait Layer
// ============================================================================
// Per-(InputT, TopK) compile-time trait encoding the secant-search target
// `kFTarget`, candidate-buffer cap `kC`, and Phase-4 histogram bin count
// `kNumBins`.
//
// kFTarget under V3.2-decode preIdx semantics (preIdx = top-K of prev row):
// Phase-1 pmean lands near the right tail of the prev-row top-K, so the
// Phase-2 secant initial bracket is biased high. A tighter K-proportional
// target converges faster than the M=2048 era's flat target.
//
// `kFTarget` is the secant solver's **soft steering target**, not the
// convergence condition. The Phase-2 loop converges whenever the
// candidate count falls within `[kK, kCC]` (see the `done = 1` check at
// the end of `gvrTopKJob`'s P1+P2 scope). A `kFTarget` below `kK` is
// intentional and useful for small K — with preIdx-seeded P1 landing
// the initial threshold near the right tail of the prev-row top-K, the
// secant's first interpolation often overshoots; biasing the target
// below kK pulls the next iteration's threshold down more aggressively
// and reaches the legal `[kK, kCC]` band in fewer secant steps. The
// concrete multipliers below were tuned empirically over V4 M=K cells
// (commit 8 in this PR):
//   K=512:  0.75K = 384
//   K=1024: 2.5K  = 2560
//   K=2048: kept at 1.5K (3072) for fp32 to preserve SASS byte-identity
//          with the V2e production hot path. bf16/fp16 K=2048 use 2K
//          (4096), where there is no prior production baseline to honor.
//
// kC = 5120 for K=512/1024 across all dtypes — drops smem footprint enough
// to leave headroom for 4-5 CTA/SM theoretical occupancy when register
// allocation permits. fp32 K=2048 keeps kC=6144 to preserve V2e SASS
// byte-identity (production fp32 K=2048 hot path is the correctness
// floor; smem layout change would alter SASS).
//
// kNumBins varies per (T, K) by atomic-contention vs Phase-4 setup-cost
// trade-off:
//   - Below ~1024 bins, atomicAdd contention on the bin-counter array
//     dominates Phase-4 cost.
//   - Above 1024 bins, the Phase-4 histogram clear+scan setup cost
//     dominates over the contention savings.
//   The optimum varies because (a) candidate-buffer size kC scales the
//   atomic-contention denominator, and (b) bf16/fp16 paths read smem keys
//   through Trait::to_fp32, shifting the Phase-3 ↔ Phase-4 ratio. Hence
//   per-(T, K) tabulation rather than a closed-form rule.
//
// Primary template intentionally left undefined: any unsupported (T, K)
// combination triggers a compile-time error rather than a runtime fall-
// through.
template <typename InputT, int TopK>
struct GvrParams; // primary undefined → compile-time error for bad combos

template <>
struct GvrParams<float, 512>
{
    static constexpr int kFTarget = 384;
    static constexpr int kC = 5120;
    static constexpr int kNumBins = 1024;
};

template <>
struct GvrParams<float, 1024>
{
    static constexpr int kFTarget = 2560;
    static constexpr int kC = 5120;
    static constexpr int kNumBins = 1024;
};

// fp32 K=2048 preserves V2e SASS byte-identity with the production hot path.
// Changing kFTarget or kC would alter ptxas output for the existing
// `gvrTopKJob<2048>` / `heuristicTopKMultiRowKernel<2048>` instantiations.
template <>
struct GvrParams<float, 2048>
{
    static constexpr int kFTarget = 3072;
    static constexpr int kC = 6144;
    static constexpr int kNumBins = NUM_BINS;
};

template <>
struct GvrParams<__nv_bfloat16, 512>
{
    static constexpr int kFTarget = 384;
    static constexpr int kC = 5120;
    static constexpr int kNumBins = 512;
};

template <>
struct GvrParams<__nv_bfloat16, 1024>
{
    static constexpr int kFTarget = 2560;
    static constexpr int kC = 5120;
    static constexpr int kNumBins = 512;
};

template <>
struct GvrParams<__nv_bfloat16, 2048>
{
    static constexpr int kFTarget = 4096;
    static constexpr int kC = 5120;
    static constexpr int kNumBins = NUM_BINS;
};

template <>
struct GvrParams<__half, 512>
{
    static constexpr int kFTarget = 384;
    static constexpr int kC = 5120;
    static constexpr int kNumBins = 512;
};

template <>
struct GvrParams<__half, 1024>
{
    static constexpr int kFTarget = 2560;
    static constexpr int kC = 5120;
    static constexpr int kNumBins = 1024;
};

template <>
struct GvrParams<__half, 2048>
{
    static constexpr int kFTarget = 4096;
    static constexpr int kC = 5120;
    static constexpr int kNumBins = NUM_BINS;
};

// kC must remain divisible by BLOCK_SIZE (vector loads).
static_assert(GvrParams<float, 512>::kC % BLOCK_SIZE == 0);
static_assert(GvrParams<float, 1024>::kC % BLOCK_SIZE == 0);
static_assert(GvrParams<float, 2048>::kC % BLOCK_SIZE == 0);

// ============================================================================
// Shared Memory Layout
// ============================================================================
// Templated on (SmemKey, candidate-cap, num-bins). Default
// (MAX_CANDIDATES=6144, NUM_BINS=2048) sizes:
//   fp32      : ~59 KB
//   bf16/fp16 : ~47 KB
// K=512/1024 instantiations cap candidates at kC=5120 (~51 KB fp32 /
//   ~41 KB bf16/fp16) per GvrParams<T, K>::kC.

template <typename SmemKey, int CCap = MAX_CANDIDATES, int NumBinsT = NUM_BINS>
struct KernelSmemTplK
{
    alignas(16) SmemKey keys[CCap];    // CCap × sizeof(SmemKey) (4B fp32 / 2B bf16/fp16)
    alignas(16) int vals[CCap];        // CCap × 4B

    int warp_counts[NUM_WARPS];        // 64 B
    int histogram[NumBinsT];           // NumBinsT × 4B (default 2048 → 8 KB)
    int per_thread_counts[BLOCK_SIZE]; // cached from the most recent blockCountGE call (Phase-3 reuse)

    float threshold;
    int cand_count;
    int done;

    float val_lo, val_hi;
    int cnt_lo, cnt_hi;

    float pmax_saved;
    int out_count;
};

// Convenience alias for the default-cap layout (kC=6144, kNumBins=2048),
// used by the K=2048 instantiations.
template <typename SmemKey>
using KernelSmemTpl = KernelSmemTplK<SmemKey>;

using KernelSmem = KernelSmemTpl<float>;

// ============================================================================
// Warp-Level Reduction Primitives
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

// Templated on SmemT so K=512/1024 paths can pass a kC=5120 layout.
// Default (KernelSmem) targets the K=2048 kC=6144 layout.
template <typename SmemT = KernelSmem>
__device__ __forceinline__ void blockCountGE(
    float const* __restrict__ input, int N, float threshold, SmemT* smem, int tid, int warp_id, int lane)
{
    int c = 0;
    for (int i = tid * 4; i + 3 < N; i += BLOCK_SIZE * 4)
    {
        float4 v4 = __ldg(reinterpret_cast<float4 const*>(input + i));
        c += (v4.x >= threshold) + (v4.y >= threshold) + (v4.z >= threshold) + (v4.w >= threshold);
    }
    for (int i = (N & ~3) + tid; i < N; i += BLOCK_SIZE)
        c += (__ldg(&input[i]) >= threshold);

    // cache per-thread count for Phase 3 sub-pass 1 reuse
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

// Templated on (TopK, SmemT) so K=512/1024 paths reuse the same helper.
template <int TopK = TOP_K, typename SmemT = KernelSmem>
__device__ __forceinline__ void blockFusedSnapIter(SmemT* smem, int count, int tid, int warp_id, int lane)
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

        if (cgt >= TopK)
        {
            if (total_up < FLT_MAX)
                smem->threshold = total_up;
        }
        else if (cge < TopK)
        {
            if (total_down > -FLT_MAX)
                smem->threshold = total_down;
        }
    }
    __syncthreads();
}

// ============================================================================
// Dtype-templated helpers (bf16 / fp16)
// ============================================================================
// Mirror of `blockCountGE` for bf16/fp16 inputs (8-wide vector load via
// `Trait::unpack8` + fp32 up-cast before threshold compare). The Phase-4
// snap iter is NOT mirrored: `gvrTopKJobDtype` stores smem `keys[]` as
// fp32 even on bf16/fp16 paths (deferred-conversion optimization, see
// the `gvrTopKJobDtype` comment block below), so it reuses the fp32
// `blockFusedSnapIter<TopK>` helper directly.

// Templated on SmemT so K=512/1024 dtype paths can pass a kC=5120 layout.
// Default targets the K=2048 kC=6144 layout.
template <typename InputT, typename SmemT = KernelSmemTpl<typename GvrDtypeTraits<InputT>::SmemKey>>
__device__ __forceinline__ void blockCountGEDtype(
    InputT const* __restrict__ input, int N, float threshold, SmemT* smem, int tid, int warp_id, int lane)
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

// ============================================================================
// Device function: algorithm body (independently optimized by ptxas)
// __noinline__ ensures ptxas allocates registers and schedules instructions
// for this function independently from the caller, matching standalone SASS.
// ============================================================================

// Templated on TopK so K=512/1024/2048 fp32 paths share this body. The
// runtime `topK` parameter is kept for header-API compatibility with
// callers that pass it; the kernel asserts at entry that it matches
// the template instantiation.
template <int TopK = TOP_K>
__device__ __noinline__ void gvrTopKJob(float const* __restrict__ input, int const N, int const* __restrict__ preIdx,
    int const M, int const topK, float* __restrict__ outputValues, int* __restrict__ outputIndices,
    KernelSmemTplK<float, GvrParams<float, TopK>::kC, GvrParams<float, TopK>::kNumBins>* smem,
    int const preIdxOffset = 0)
{
    using Params = GvrParams<float, TopK>;
    constexpr int kK = TopK;
    constexpr int kCC = Params::kC;
    constexpr int kBins = Params::kNumBins;
    constexpr int kFTarget = Params::kFTarget;

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
            if (c >= kK && c <= kCC)
                smem->done = 1;
            else if (c > kCC)
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
                constexpr int target = kFTarget;
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
                if (c >= kK && c <= kCC)
                    smem->done = 1;
                else if (c > kCC)
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
            if (smem->cnt_lo <= kCC * 2)
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

    // When done==1, Phase 2 already verified the candidate count is in
    // [kK, kCC]; skip the redundant full-N blockCountGE re-check.
    if (smem->done != 1)
    {
        blockCountGE(input, N, smem->threshold, smem, tid, warp_id, lane);
        if (tid == 0 && smem->cand_count > kCC)
            smem->val_lo = smem->threshold;
        __syncthreads();

        for (int retry = 0; retry < 10 && smem->cand_count > kCC; retry++)
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
                if (c > kCC)
                    smem->val_lo = smem->threshold;
                else if (c < kK)
                    smem->val_hi = smem->threshold;
            }
            __syncthreads();
        }
    }

    // Reuse per-thread counts cached by the last blockCountGE call (saves
    // one full N-scan; blockCountGE's __syncthreads guarantees visibility).
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
                if (val >= thr && my_write_pos < kCC)
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
            if (val >= thr && my_write_pos < kCC)
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

    int const cand_count = min(smem->cand_count, kCC);

    if (cand_count == kK)
    {
        for (int i = tid; i < kK; i += BLOCK_SIZE)
        {
            outputValues[i] = smem->keys[i];
            outputIndices[i] = smem->vals[i];
        }
        return;
    }

    if (cand_count > kK)
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

        for (int i = tid; i < kBins; i += BLOCK_SIZE)
            smem->histogram[i] = 0;
        __syncthreads();

        float range1 = block_max - block_min;
        float inv1 = (range1 > 0.0f) ? ((float) (kBins - 1) + 0.99f) / range1 : 0.0f;

        for (int i = tid; i < cand_count; i += BLOCK_SIZE)
        {
            int bin = (int) ((smem->keys[i] - block_min) * inv1);
            bin = min(max(bin, 0), kBins - 1);
            atomicAdd(&smem->histogram[bin], 1);
        }
        __syncthreads();

        // Parallel K-th bin search (3-step).
        // Step 1: each warp sums BINS_PER_WARP consecutive bins (high→low).
        // Step 2: tid=0 locates the target warp in NUM_WARPS steps.
        // Step 3: one thread in that warp scans its BINS_PER_WARP bins.
        // Total serial depth: NUM_WARPS + BINS_PER_WARP steps vs full kBins.
        {
            constexpr int BINS_PER_WARP = kBins / NUM_WARPS;
            static_assert(kBins % NUM_WARPS == 0, "kBins must be divisible by NUM_WARPS");
            // Step 1: each warp accumulates its slice of bins (high→low)
            int warp_bin_sum = 0;
            for (int j = 0; j < BINS_PER_WARP; j++)
                warp_bin_sum += smem->histogram[kBins - 1 - warp_id * BINS_PER_WARP - j];
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
                if (cum >= kK)
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
            constexpr int BINS_PER_WARP = kBins / NUM_WARPS;
            int base_cum = smem->cnt_lo;
            float thr = block_min;
            for (int j = 0; j < BINS_PER_WARP; j++)
            {
                int b = kBins - 1 - smem->cnt_hi * BINS_PER_WARP - j;
                base_cum += smem->histogram[b];
                if (base_cum >= kK)
                {
                    thr = block_min + (float) b * range1 / (float) kBins;
                    break;
                }
            }
            smem->threshold = thr;
        }
        __syncthreads(); // S-4b3c

        // snap_limit must equal cand_count to guarantee convergence: each
        // iteration either strictly decreases cgt (raise thr to next
        // distinct value above) or strictly increases cge (lower thr to
        // next distinct value below) by >= 1, so worst-case convergence
        // takes cand_count - kK + 1 iters. The older bound `cand_count/4`
        // silently accepted a non-converged threshold; Pass 1 then picked
        // K elements in scan order from `cgt > kK` candidates, missing
        // some true top-K members (~0.09 % intermittent at small-kNumBins
        // mean-zero distributions). Common path still converges in 1-3
        // iters; the higher upper bound only affects the long-tail cells.
        bool snap_converged = false;
        int snap_limit = cand_count;
        for (int si = 0; si < snap_limit; si++)
        {
            blockFusedSnapIter<TopK>(smem, cand_count, tid, warp_id, lane);
            int cge = smem->cnt_lo;
            int cgt = smem->cnt_hi;
            if (cgt < kK && cge >= kK)
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

        // Two-pass selection: pass 1 emits strictly-greater-than-threshold
        // candidates, pass 2 fills remaining slots with tie-values. An
        // interleaved single-pass implementation would be unstable across
        // rows with many ties at the K-th rank — equal-valued candidates
        // could displace strictly-greater ones depending on their relative
        // order in the candidate buffer. Splitting the passes makes the
        // selection deterministic regardless of buffer ordering.

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
                if (emit_gt && bp + moff < kK)
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
                if (emit_eq && bp + moff < kK)
                {
                    outputValues[bp + moff] = v;
                    outputIndices[bp + moff] = smem->vals[i];
                }
            }
        }
        __syncthreads();

        int filled = min(smem->out_count, kK);
        for (int i = filled + tid; i < kK; i += BLOCK_SIZE)
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
    for (int i = cand_count + tid; i < kK; i += BLOCK_SIZE)
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

// Templated on TopK so the launcher can dispatch K=512/1024/2048 to the
// same kernel template.
//
// __launch_bounds__ uses the single-arg form (no minBlocksPerSM hint) so
// nvcc applies the same register heuristic as `heuristicTopKMultiRowKernel`
// in heuristicTopKDecode.cu. Adding `, 1` would lower theoretical occupancy
// from 75% (REG=40) to 50% (REG=64) for the K=2048 fp32 path.
template <int TopK = TOP_K>
__global__ void __launch_bounds__(BLOCK_SIZE)
    gvrTopKKernel(float const* __restrict__ input, int const N, int const* __restrict__ preIdx, int const M,
        int const topK, float* __restrict__ outputValues, int* __restrict__ outputIndices)
{
    using SmemT = KernelSmemTplK<float, GvrParams<float, TopK>::kC, GvrParams<float, TopK>::kNumBins>;
    extern __shared__ unsigned char smem_raw[];
    auto* smem = reinterpret_cast<SmemT*>(smem_raw);

    gvrTopKJob<TopK>(input, N, preIdx, M, topK, outputValues, outputIndices, smem, /*preIdxOffset=*/0);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// ============================================================================
// gvrTopKJobDtype<InputT, TopK> — bf16/fp16 device function
// ============================================================================
// Mirror of gvrTopKJob with trait-driven dtype substitutions:
//   - HBM input read   : Trait::to_fp32(__ldg(&input[i]))
//   - outputValues     : InputT (Trait::from_fp32 at writeback)
//   - vector load      : 8-wide via Trait::unpack8
//   - blockCountGE     : blockCountGEDtype<InputT>
// `blockFusedSnapIter<TopK>` is reused directly from the fp32 path —
// smem `keys[]` are stored as fp32 here (see deferred-conversion note
// below) so no dtype-specialized snap helper is needed.
// All arithmetic (threshold, accumulators, bin index) stays fp32. The fp32
// dtype path uses gvrTopKJob (above), not this template; instantiated only
// for bf16 and fp16.
//
// smem `keys[]` are stored as fp32 even on the bf16/fp16 paths: deferring
// the down-conversion out of the Phase-3 collect loop saves more than the
// extra ~10 KB of smem costs. The fp32 keys move conversion to the output
// writeback (one cvt per surviving candidate) instead of every smem store.
template <typename InputT, int TopK = TOP_K>
__device__ __noinline__ void gvrTopKJobDtype(InputT const* __restrict__ input, int const N,
    int const* __restrict__ preIdx, int const M, int const topK, InputT* __restrict__ outputValues,
    int* __restrict__ outputIndices,
    KernelSmemTplK<float, GvrParams<InputT, TopK>::kC, GvrParams<InputT, TopK>::kNumBins>* smem,
    int const preIdxOffset = 0)
{
    using Trait = GvrDtypeTraits<InputT>;
    using SmemKey = float; // keys stay fp32; conversion deferred to output writeback
    using Params = GvrParams<InputT, TopK>;
    constexpr int kK = TopK;
    constexpr int kCC = Params::kC;
    constexpr int kBins = Params::kNumBins;
    constexpr int kFTarget = Params::kFTarget;
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
            if (c >= kK && c <= kCC)
                smem->done = 1;
            else if (c > kCC)
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
                constexpr int target = kFTarget;
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
                if (c >= kK && c <= kCC)
                    smem->done = 1;
                else if (c > kCC)
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
            if (smem->cnt_lo <= kCC * 2)
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
        if (tid == 0 && smem->cand_count > kCC)
            smem->val_lo = smem->threshold;
        __syncthreads();

        for (int retry = 0; retry < 10 && smem->cand_count > kCC; retry++)
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
                if (c > kCC)
                    smem->val_lo = smem->threshold;
                else if (c < kK)
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
                if (val >= thr && my_write_pos < kCC)
                {
                    smem->keys[my_write_pos] = val; // P0: defer convert to output
                    smem->vals[my_write_pos] = i + j;
                    my_write_pos++;
                }
            }
        }
        // Tail loop (N % 8)
        for (int i = (N & ~7) + tid; i < N; i += BLOCK_SIZE)
        {
            float val = Trait::to_fp32(__ldg(&input[i]));
            if (val >= thr && my_write_pos < kCC)
            {
                smem->keys[my_write_pos] = val; // P0: defer convert to output
                smem->vals[my_write_pos] = i;
                my_write_pos++;
            }
        }
    }
    __syncthreads();

    // ================================================================
    // Phase 4 — Histogram-based selection + partition
    // ================================================================

    int const cand_count = min(smem->cand_count, kCC);

    if (cand_count == kK)
    {
        for (int i = tid; i < kK; i += BLOCK_SIZE)
        {
            outputValues[i] = Trait::from_fp32(smem->keys[i]); // P0: convert at output
            outputIndices[i] = smem->vals[i];
        }
        return;
    }

    if (cand_count > kK)
    {
        float cmin = FLT_MAX, cmax = -FLT_MAX;
        for (int i = tid; i < cand_count; i += BLOCK_SIZE)
        {
            float v = smem->keys[i]; // P0: keys already fp32
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

        for (int i = tid; i < kBins; i += BLOCK_SIZE)
            smem->histogram[i] = 0;
        __syncthreads();

        float range1 = block_max - block_min;
        float inv1 = (range1 > 0.0f) ? ((float) (kBins - 1) + 0.99f) / range1 : 0.0f;

        for (int i = tid; i < cand_count; i += BLOCK_SIZE)
        {
            int bin = (int) ((smem->keys[i] - block_min) * inv1); // P0: keys fp32
            bin = min(max(bin, 0), kBins - 1);
            atomicAdd(&smem->histogram[bin], 1);
        }
        __syncthreads();

        // Parallel K-th bin search (2-step)
        {
            constexpr int BINS_PER_WARP = kBins / NUM_WARPS;
            static_assert(kBins % NUM_WARPS == 0, "kBins must be divisible by NUM_WARPS");
            int warp_bin_sum = 0;
            for (int j = 0; j < BINS_PER_WARP; j++)
                warp_bin_sum += smem->histogram[kBins - 1 - warp_id * BINS_PER_WARP - j];
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
                if (cum >= kK)
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
            constexpr int BINS_PER_WARP = kBins / NUM_WARPS;
            int base_cum = smem->cnt_lo;
            float thr = block_min;
            for (int j = 0; j < BINS_PER_WARP; j++)
            {
                int b = kBins - 1 - smem->cnt_hi * BINS_PER_WARP - j;
                base_cum += smem->histogram[b];
                if (base_cum >= kK)
                {
                    thr = block_min + (float) b * range1 / (float) kBins;
                    break;
                }
            }
            smem->threshold = thr;
        }
        __syncthreads();

        // snap_limit must equal cand_count to guarantee convergence; see
        // detailed rationale in the fp32 `gvrTopKJob` path.
        bool snap_converged = false;
        int snap_limit = cand_count;
        for (int si = 0; si < snap_limit; si++)
        {
            blockFusedSnapIter<TopK>(smem, cand_count, tid, warp_id, lane); // P0: keys fp32 → use fp32 snap helper
            int cge = smem->cnt_lo;
            int cgt = smem->cnt_hi;
            if (cgt < kK && cge >= kK)
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
            float v = (i < cand_count) ? smem->keys[i] : -FLT_MAX; // P0: keys fp32

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
                if (emit_gt && bp + moff < kK)
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
            float v = (i < cand_count) ? smem->keys[i] : -FLT_MAX; // P0: keys fp32

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
                if (emit_eq && bp + moff < kK)
                {
                    outputValues[bp + moff] = Trait::from_fp32(v);
                    outputIndices[bp + moff] = smem->vals[i];
                }
            }
        }
        __syncthreads();

        int filled = min(smem->out_count, kK);
        InputT const neg_max = Trait::from_fp32(-FLT_MAX);
        for (int i = filled + tid; i < kK; i += BLOCK_SIZE)
        {
            outputValues[i] = neg_max;
            outputIndices[i] = -1;
        }
        return;
    }

    // cand_count < kK fallback
    for (int i = tid; i < cand_count; i += BLOCK_SIZE)
    {
        outputValues[i] = Trait::from_fp32(smem->keys[i]); // P0: convert at output
        outputIndices[i] = smem->vals[i];
    }
    InputT const neg_max = Trait::from_fp32(-FLT_MAX);
    for (int i = cand_count + tid; i < kK; i += BLOCK_SIZE)
    {
        outputValues[i] = neg_max;
        outputIndices[i] = -1;
    }
}

// ============================================================================
// gvrTopKKernelDtype<InputT, TopK> — bf16/fp16 single-row global wrapper
// ============================================================================
// Templated on (InputT, TopK). __launch_bounds__ uses the single-arg form
// so nvcc applies the same register heuristic as the multi-row dtype kernel
// in heuristicTopKDecode.cu. See `gvrTopKKernel<TopK>` note above.

template <typename InputT, int TopK = TOP_K>
__global__ void __launch_bounds__(BLOCK_SIZE)
    gvrTopKKernelDtype(InputT const* __restrict__ input, int const N, int const* __restrict__ preIdx, int const M,
        int const topK, InputT* __restrict__ outputValues, int* __restrict__ outputIndices)
{
    using SmemKey = typename GvrDtypeTraits<InputT>::SmemKey;
    using SmemT
        = KernelSmemTplK<float, GvrParams<InputT, TopK>::kC, GvrParams<InputT, TopK>::kNumBins>; // dtype keys fp32
    extern __shared__ unsigned char smem_raw[];
    auto* smem = reinterpret_cast<SmemT*>(smem_raw);

    gvrTopKJobDtype<InputT, TopK>(input, N, preIdx, M, topK, outputValues, outputIndices, smem,
        /*preIdxOffset=*/0);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// ============================================================================
// Explicit kernel instantiations — 9 (T × K) combos.
// ============================================================================
// Mirrors the same pattern used in heuristicTopKDecode.cu for the multi-row
// kernels. Forces nvcc to emit each `gvrTopKKernel<K>` / `gvrTopKKernelDtype
// <T,K>` host-side wrapper stub *before* `launchHeuristicTopK` takes their
// address via `cudaLaunchKernelEx`. Without these declarations, certain nvcc
// stubgen versions (CI containers on sm_89 / sm_120f) emit the implicit stub
// inside the kernel body and then conflict with their own subsequent
// "explicit specialization of __wrapper__device_stub_*<K>" pass — surfacing
// as a "specialization after instantiation" build error against
// cudafe1.stub.c. Header is included only by heuristicTopKDecode.cu (one TU)
// so no ODR concern.
template __global__ void gvrTopKKernel<512>(float const*, int, int const*, int, int, float*, int*);
template __global__ void gvrTopKKernel<1024>(float const*, int, int const*, int, int, float*, int*);
template __global__ void gvrTopKKernel<2048>(float const*, int, int const*, int, int, float*, int*);
template __global__ void gvrTopKKernelDtype<__nv_bfloat16, 512>(
    __nv_bfloat16 const*, int, int const*, int, int, __nv_bfloat16*, int*);
template __global__ void gvrTopKKernelDtype<__nv_bfloat16, 1024>(
    __nv_bfloat16 const*, int, int const*, int, int, __nv_bfloat16*, int*);
template __global__ void gvrTopKKernelDtype<__nv_bfloat16, 2048>(
    __nv_bfloat16 const*, int, int const*, int, int, __nv_bfloat16*, int*);
template __global__ void gvrTopKKernelDtype<__half, 512>(__half const*, int, int const*, int, int, __half*, int*);
template __global__ void gvrTopKKernelDtype<__half, 1024>(__half const*, int, int const*, int, int, __half*, int*);
template __global__ void gvrTopKKernelDtype<__half, 2048>(__half const*, int, int const*, int, int, __half*, int*);

// ============================================================================
// Launch Wrapper
// ============================================================================

namespace detail
{
// Per-(T, TopK) launcher implementation. Hoisted out of `launchHeuristicTopK`
// (was a C++20 templated lambda `[&]<int TopK>()`) because that pattern
// confuses nvcc's cudafe1 stub generator: taking the address of
// `gvrTopKKernel<TopK>` / `gvrTopKKernelDtype<T, TopK>` from inside a
// templated capturing lambda triggers an "explicit specialization of
// `__wrapper__device_stub_gvrTopKKernel<K>` after instantiation" error
// against the auto-generated host wrapper stub. A regular function template
// avoids the quirk and stays in C++17 (no templated-lambda extension warning).
//
// Kernel body, GvrParams traits, kfn selection, opt-in smem, PDL attr, and
// cudaLaunchKernelEx call are byte-identical to the previous lambda body —
// SASS is unchanged for all 9 (T, K) instantiations.
template <typename T, int TopK>
cudaError_t launchHeuristicTopKImpl(T const* input, int N, int const* preIdx, int M, int topK, T* outputValues,
    int* outputIndices, cudaStream_t stream, bool enablePDL)
{
    // dtype path uses fp32 smem keys (deferred convert). Launcher
    // allocates smem with float keys regardless of input dtype.
    using SmemT = KernelSmemTplK<float, GvrParams<T, TopK>::kC, GvrParams<T, TopK>::kNumBins>;
    size_t const smemSize = sizeof(SmemT);

    // Resolve target kernel function pointer at compile time.
    auto kfn = []()
    {
        if constexpr (std::is_same_v<T, float>)
            return gvrTopKKernel<TopK>;
        else
            return gvrTopKKernelDtype<T, TopK>;
    }();

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

    cudaLaunchKernelEx(&config, kfn, input, N, preIdx, M, topK, outputValues, outputIndices);
    return cudaGetLastError();
}
} // namespace detail

template <typename T, typename IdxT = int>
cudaError_t launchHeuristicTopK(T const* input, int N, IdxT const* preIdx, int M, int topK, T* outputValues,
    IdxT* outputIndices, cudaStream_t stream = 0)
{
    static_assert(sizeof(IdxT) == sizeof(int), "launchHeuristicTopK only supports 32-bit indices");
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, __half>,
        "launchHeuristicTopK supports only fp32 / bf16 / fp16");

    // GvrParams specializations cover K ∈ {512, 1024, 2048}; reject others.
    if (topK != 512 && topK != 1024 && topK != 2048)
        return cudaErrorInvalidValue;

    // Dispatch on (T, topK) → 9 distinct kernel-pointer paths. Each
    // instantiation captures its own (kFTarget, kC, kNumBins) tuple via
    // GvrParams<T, TopK> so all values are compile-time constants inside
    // the kernel body. Opt-in smem + cudaLaunchKernelEx + PDL handling is
    // shared across all 9 paths via `detail::launchHeuristicTopKImpl`.

    // Honor the standard TRTLLM_ENABLE_PDL env var (default on; set "0" to
    // disable).
    bool enablePDL = true;
    if (char const* env = std::getenv("TRTLLM_ENABLE_PDL"))
    {
        if (env[0] == '0' && env[1] == '\0')
            enablePDL = false;
    }

    switch (topK)
    {
    case 512:
        return detail::launchHeuristicTopKImpl<T, 512>(
            input, N, preIdx, M, topK, outputValues, outputIndices, stream, enablePDL);
    case 1024:
        return detail::launchHeuristicTopKImpl<T, 1024>(
            input, N, preIdx, M, topK, outputValues, outputIndices, stream, enablePDL);
    case 2048:
        return detail::launchHeuristicTopKImpl<T, 2048>(
            input, N, preIdx, M, topK, outputValues, outputIndices, stream, enablePDL);
    default: return cudaErrorInvalidValue;
    }
}

// Explicit instantiations — fp32 + bf16/fp16
template cudaError_t launchHeuristicTopK<float, int>(
    float const*, int, int const*, int, int, float*, int*, cudaStream_t);
template cudaError_t launchHeuristicTopK<__nv_bfloat16, int>(
    __nv_bfloat16 const*, int, int const*, int, int, __nv_bfloat16*, int*, cudaStream_t);
template cudaError_t launchHeuristicTopK<__half, int>(
    __half const*, int, int const*, int, int, __half*, int*, cudaStream_t);

} // namespace heuristic_topk
