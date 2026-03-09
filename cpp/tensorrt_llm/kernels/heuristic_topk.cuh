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
// heuristic_topk.cuh — V2d: V2c + skip 4a + skip Pass A + 256-bin + snap3
// Heuristic-Guided TopK — Sort-Free, Histogram-Based Selection
// Optimised for NVIDIA B200 (Blackwell, sm_100), single thread-block kernel
//
// V2d: ballot-free Phase 3, skip-4a, skip-PassA, 256-bin, snap≤3
//      OPT3 (__ldg), OPT4 (redux.sync)
//
// Define HEURISTIC_TOPK_PROFILE before including to enable per-phase printf.
// Shared memory: ~50 KB (no CUB dependency)
// ============================================================================

#pragma once

#include <cfloat>
#include <cstdint>
#include <cstdio> // for snap convergence warning printf
#include <cuda_runtime.h>

namespace heuristic_topk
{

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
constexpr int NUM_BINS = 256;

static_assert(TOP_K % BLOCK_SIZE == 0);
static_assert(MAX_CANDIDATES % BLOCK_SIZE == 0);

// ============================================================================
// Shared Memory Layout (~26 KB)
// ============================================================================

struct KernelSmem
{
    alignas(16) float keys[MAX_CANDIDATES]; // 12 KB
    alignas(16) int vals[MAX_CANDIDATES];   // 12 KB

    int warp_counts[NUM_WARPS]; // 64 B
    int histogram[NUM_BINS];    // 1 KB

    float threshold;
    int cand_count;
    int done;

    float val_lo, val_hi;
    int cnt_lo, cnt_hi;

    float pmax_saved;
    int out_count;
};

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
// Main Kernel
// ============================================================================

__global__ void __launch_bounds__(BLOCK_SIZE, 1)
    heuristicTopKKernel(float const* __restrict__ input, int const N, int const* __restrict__ preIdx, int const M,
        int const topK, float* __restrict__ outputValues, int* __restrict__ outputIndices, int const thresholdPos)
{
    extern __shared__ unsigned char smem_raw[];
    auto* smem = reinterpret_cast<KernelSmem*>(smem_raw);

    int const tid = threadIdx.x;
    int const warp_id = tid / WARP_SIZE;
    int const lane = tid & (WARP_SIZE - 1);
    unsigned const full_mask = 0xffffffffu;

#ifdef HEURISTIC_TOPK_PROFILE
    long long prof_t0 = 0, prof_t1 = 0, prof_t2 = 0, prof_t3 = 0, prof_t4 = 0;
    int prof_p2_iters = 0, prof_p2_first_count = 0, prof_p2_done_type = 0;
    if (tid == 0)
        prof_t0 = clock64();
#endif

    // ================================================================
    // Phase 1 — Min/Max/Mean of pre-indexed values
    // ================================================================

    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;
    int local_cnt = 0;
    for (int i = tid; i < M; i += BLOCK_SIZE)
    {
        int idx = __ldg(&preIdx[i]);
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
    __syncthreads(); // S1

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
    __syncthreads(); // S2

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
    // Phase 2 — Interpolation threshold search (count-only)
    // ================================================================

#ifdef HEURISTIC_TOPK_PROFILE
    if (tid == 0)
        prof_t1 = clock64();
#endif

    blockCountGE(input, N, smem->threshold, smem, tid, warp_id, lane);

    if (tid == 0)
    {
        int c = smem->cand_count;
#ifdef HEURISTIC_TOPK_PROFILE
        prof_p2_first_count = c;
        prof_p2_iters = 1;
#endif
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
                {
                    smem->threshold = nv;
                }
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
#ifdef HEURISTIC_TOPK_PROFILE
            prof_p2_iters++;
#endif
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
        // Fallback: prefer val_lo (count > MAX_CANDIDATES, overshoot) since
        // Phase 4 can select from the excess.  With MAX_CANDIDATES=4096,
        // the overshoot is much less likely to drop true top-K elements.
        // If val_lo's count would massively exceed MAX_CANDIDATES, use val_hi
        // (undershoot) and let the fill step pad with (-FLT_MAX, -1).
        if (smem->cnt_lo <= MAX_CANDIDATES * 2)
            smem->threshold = smem->val_lo;
        else
            smem->threshold = smem->val_hi;
        smem->done = 2;
    }
#ifdef HEURISTIC_TOPK_PROFILE
    if (tid == 0)
    {
        prof_p2_done_type = smem->done;
        prof_t2 = clock64();
    }
#endif
    __syncthreads();

    // ================================================================
    // Phase 3 — Ballot-free collect
    // ================================================================

    // Safety net: if count exceeds MAX_CANDIDATES, tighten threshold by
    // bisecting toward val_hi (which gave count < TOP_K) until it fits.
    // Each retry is one L2-cached blockCountGE (~6μs worst case).
    // Normal path (done=1, count already in range): 1 count + break.
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

    // Sub-pass 1: per-thread qualifying count
    int my_total_qual = 0;
    {
        float const thr = smem->threshold;
        for (int i = tid * 4; i + 3 < N; i += BLOCK_SIZE * 4)
        {
            float4 v4 = __ldg(reinterpret_cast<float4 const*>(input + i));
            my_total_qual += (v4.x >= thr) + (v4.y >= thr) + (v4.z >= thr) + (v4.w >= thr);
        }
        for (int i = (N & ~3) + tid; i < N; i += BLOCK_SIZE)
            my_total_qual += (__ldg(&input[i]) >= thr);
    }

    // Warp-level inclusive prefix sum of per-thread counts
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

    // Cross-warp exclusive prefix sum
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

    // Sub-pass 2: per-thread write (ballot-free)
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

    int const cand_count = min(smem->cand_count, MAX_CANDIDATES);

#ifdef HEURISTIC_TOPK_PROFILE
    if (tid == 0)
        prof_t3 = clock64();
#endif

    // ================================================================
    // Phase 4 — Histogram-based selection + partition
    // ================================================================

    if (cand_count == TOP_K)
    {
        for (int i = tid; i < TOP_K; i += BLOCK_SIZE)
        {
            outputValues[i] = smem->keys[i];
            outputIndices[i] = smem->vals[i];
        }
#ifdef HEURISTIC_TOPK_PROFILE
        if (tid == 0)
        {
            prof_t4 = clock64();
            printf(
                "[topK prof] P1=%lld P2=%lld P3=%lld P4=%lld total=%lld | "
                "P2_iters=%d first_count=%d done_type=%d cands=%d (exact)\n",
                prof_t1 - prof_t0, prof_t2 - prof_t1, prof_t3 - prof_t2, prof_t4 - prof_t3, prof_t4 - prof_t0,
                prof_p2_iters, prof_p2_first_count, prof_p2_done_type, cand_count);
        }
#endif
        return;
    }

    if (cand_count > TOP_K)
    {

        // ---- 4a: Candidate Min/Max (restored for correctness) ----
        // pmax_saved is from preIdx and may underestimate the true candidate
        // max when non-preIdx elements exceed pmax.  Use actual scan.
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
        __syncthreads(); // S-4a

        float block_min = FLT_MAX, block_max = -FLT_MAX;
        for (int w = 0; w < NUM_WARPS; w++)
        {
            block_min = fminf(block_min, __int_as_float(smem->warp_counts[w]));
            block_max = fmaxf(block_max, __int_as_float(smem->histogram[w]));
        }
        if (block_max <= block_min)
            block_max = block_min + 1e-6f;

        // ---- 4b: 256-bin histogram ----
        for (int i = tid; i < NUM_BINS; i += BLOCK_SIZE)
            smem->histogram[i] = 0;
        __syncthreads(); // S-4b1

        float range1 = block_max - block_min;
        float inv1 = (range1 > 0.0f) ? ((float) (NUM_BINS - 1) + 0.99f) / range1 : 0.0f;

        for (int i = tid; i < cand_count; i += BLOCK_SIZE)
        {
            int bin = (int) ((smem->keys[i] - block_min) * inv1);
            bin = min(max(bin, 0), NUM_BINS - 1);
            atomicAdd(&smem->histogram[bin], 1);
        }
        __syncthreads(); // S-4b2

        if (tid == 0)
        {
            int cum = 0;
            float thr = block_min;
            for (int b = NUM_BINS - 1; b >= 0; b--)
            {
                cum += smem->histogram[b];
                if (cum >= TOP_K)
                {
                    thr = block_min + (float) b * range1 / (float) NUM_BINS;
                    break;
                }
            }
            smem->threshold = thr;
        }
        __syncthreads(); // S-4b3

#ifdef HEURISTIC_TOPK_PROFILE
        long long prof_4b = 0, prof_4d = 0;
        int prof_snap_iters = 0;
        if (tid == 0)
            prof_4b = clock64();
#endif

        // ---- 4d: Snap iterations ----
        // Each snap moves threshold by one distinct value.  The limit must
        // exceed the number of distinct values in the histogram's target bin.
        // With cand_count up to MAX_CANDIDATES and 256 bins, worst case is
        // ~cand_count/NUM_BINS ≈ 24.  But the histogram range [block_min,
        // block_max] may be much wider than the target bin, concentrating
        // many candidates in a few bins.  Use cand_count/4 as a safe limit
        // (each snap is cheap: 2 syncs + 1 smem scan of cand_count elements).
        bool snap_converged = false;
        int snap_limit = (cand_count > 128 ? cand_count / 4 : 32);
        int snap_iters_done = 0;
        for (int si = 0; si < snap_limit; si++)
        {
            blockFusedSnapIter(smem, cand_count, tid, warp_id, lane);
            snap_iters_done = si + 1;
#ifdef HEURISTIC_TOPK_PROFILE
            if (tid == 0)
                prof_snap_iters++;
#endif
            int cge = smem->cnt_lo;
            int cgt = smem->cnt_hi;
            if (cgt < TOP_K && cge >= TOP_K)
            {
                snap_converged = true;
                break;
            }
        }

        if (!snap_converged && tid == 0)
        {
            printf(
                "[topK WARN] snap did NOT converge after %d/%d iters! "
                "cands=%d count_ge=%d count_gt=%d thr=%.6f\n",
                snap_iters_done, snap_limit, cand_count, smem->cnt_lo, smem->cnt_hi, smem->threshold);
        }

#ifdef HEURISTIC_TOPK_PROFILE
        if (tid == 0)
            prof_4d = clock64();
#endif

        // ---- 4e: Partition ----
        float sel_thr = smem->threshold;
        if (tid == 0)
            smem->out_count = 0;
        __syncthreads(); // S-4e1

        // ---- Partition: emit > and == sel_thr from candidate set ----
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

        // Fill any remaining output slots if partition underflowed
        {
            int filled = min(smem->out_count, TOP_K);
            for (int i = filled + tid; i < TOP_K; i += BLOCK_SIZE)
            {
                outputValues[i] = -FLT_MAX;
                outputIndices[i] = -1;
            }
        }
#ifdef HEURISTIC_TOPK_PROFILE
        if (tid == 0)
        {
            prof_t4 = clock64();
            long long p4_hist = prof_4b - prof_t3;
            long long p4_snap = prof_4d - prof_4b;
            long long p4_part = prof_t4 - prof_4d;
            printf(
                "[topK prof] P1=%lld P2=%lld P3=%lld P4=%lld total=%lld | "
                "P2_iters=%d first_count=%d done_type=%d cands=%d | "
                "P4: hist=%lld snap=%lld(x%d) part=%lld\n",
                prof_t1 - prof_t0, prof_t2 - prof_t1, prof_t3 - prof_t2, prof_t4 - prof_t3, prof_t4 - prof_t0,
                prof_p2_iters, prof_p2_first_count, prof_p2_done_type, cand_count, p4_hist, p4_snap, prof_snap_iters,
                p4_part);
        }
#endif
        return;
    }

    // Edge case: cand_count < TOP_K
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
#ifdef HEURISTIC_TOPK_PROFILE
    if (tid == 0)
    {
        prof_t4 = clock64();
        printf(
            "[topK prof] P1=%lld P2=%lld P3=%lld P4=%lld total=%lld | "
            "P2_iters=%d first_count=%d done_type=%d cands=%d (underflow)\n",
            prof_t1 - prof_t0, prof_t2 - prof_t1, prof_t3 - prof_t2, prof_t4 - prof_t3, prof_t4 - prof_t0,
            prof_p2_iters, prof_p2_first_count, prof_p2_done_type, cand_count);
    }
#endif
}

// ============================================================================
// Launch Wrapper
// ============================================================================

template <typename T, typename IdxT = int>
cudaError_t launchHeuristicTopK(T const* input, int N, IdxT const* preIdx, int M, int topK, T* outputValues,
    IdxT* outputIndices, cudaStream_t stream = 0, int thresholdPos = -1)
{
    size_t smemSize = sizeof(KernelSmem);

    static bool configured = false;
    if (!configured)
    {
        int device;
        cudaGetDevice(&device);
        int maxSmem;
        cudaDeviceGetAttribute(&maxSmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        if (smemSize > static_cast<size_t>(maxSmem))
            return cudaErrorInvalidConfiguration;
        if (smemSize > 48u * 1024u)
            cudaFuncSetAttribute(
                heuristicTopKKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smemSize));
        configured = true;
    }

    heuristicTopKKernel<<<1, BLOCK_SIZE, smemSize, stream>>>(reinterpret_cast<float const*>(input), N,
        reinterpret_cast<int const*>(preIdx), M, topK, reinterpret_cast<float*>(outputValues),
        reinterpret_cast<int*>(outputIndices), thresholdPos);

    return cudaGetLastError();
}

template cudaError_t launchHeuristicTopK<float, int>(
    float const*, int, int const*, int, int, float*, int*, cudaStream_t, int);
template cudaError_t launchHeuristicTopK<float, int64_t>(
    float const*, int, int64_t const*, int, int, float*, int64_t*, cudaStream_t, int);

} // namespace heuristic_topk
