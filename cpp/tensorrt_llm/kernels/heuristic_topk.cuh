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
// heuristic_topk.cuh — V2e: V2d + OPT5 (safety-guard done!=1) +
//                           OPT6 (NUM_BINS=2048 + parallel Phase 4b K-th bin search) +
//                           OPT7 (Phase 3 sub-pass 1 elimination via per-thread count cache)
// Heuristic-Guided TopK — Sort-Free, Histogram-Based Selection
// Optimised for NVIDIA B200 (Blackwell, sm_100), single thread-block kernel
//
// V2d: ballot-free Phase 3, skip-4a, skip-PassA, 256-bin, snap≤3
//      OPT3 (__ldg), OPT4 (redux.sync)
// V2e: +OPT5 (skip Phase 3 blockCountGE re-scan when done==1)
//      +OPT6 (NUM_BINS=2048 + parallel 2-step K-th bin search in Phase 4b)
//      +OPT7 (reuse per-thread counts cached by last blockCountGE; eliminates
//             Phase 3 sub-pass 1 full-N rescan)
//
// Define HEURISTIC_TOPK_PROFILE before including to enable per-phase printf.
// Shared memory: ~59 KB (no CUB dependency)
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
constexpr int NUM_BINS = 2048;

static_assert(TOP_K % BLOCK_SIZE == 0);
static_assert(MAX_CANDIDATES % BLOCK_SIZE == 0);

// ============================================================================
// Shared Memory Layout (~59 KB)
// ============================================================================

struct KernelSmem
{
    alignas(16) float keys[MAX_CANDIDATES]; // 24 KB
    alignas(16) int vals[MAX_CANDIDATES];   // 24 KB

    int warp_counts[NUM_WARPS];             // 64 B
    int histogram[NUM_BINS];                // 8 KB
    int per_thread_counts[BLOCK_SIZE];      // 2 KB — OPT7: cached from last blockCountGE

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
// Device function: algorithm body (independently optimized by ptxas)
// __noinline__ ensures ptxas allocates registers and schedules instructions
// for this function independently from the caller, matching standalone SASS.
// ============================================================================

__device__ __noinline__ void heuristicTopKJob(float const* __restrict__ input, int const N,
    int const* __restrict__ preIdx, int const M, int const topK, float* __restrict__ outputValues,
    int* __restrict__ outputIndices, KernelSmem* smem, int const preIdxOffset = 0)
{
    int const tid = threadIdx.x;
    int const warp_id = tid / WARP_SIZE;
    int const lane = tid & (WARP_SIZE - 1);
    unsigned const full_mask = 0xffffffffu;

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
    // Phase 2 — Interpolation threshold search
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

    // ================================================================
    // Phase 3 — Ballot-free collect
    // ================================================================

    // OPT5: when done==1, Phase 2 already verified count in [TOP_K, MAX_CANDIDATES];
    //        skip the redundant full-N blockCountGE re-check entirely.
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
    // Phase 4 — Histogram-based selection + partition
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
// Main Kernel (calls heuristicTopKJob as an independently-optimized device fn)
// ============================================================================

__global__ void __launch_bounds__(BLOCK_SIZE, 1)
    heuristicTopKKernel(float const* __restrict__ input, int const N, int const* __restrict__ preIdx, int const M,
        int const topK, float* __restrict__ outputValues, int* __restrict__ outputIndices, int const thresholdPos)
{
    extern __shared__ unsigned char smem_raw[];
    auto* smem = reinterpret_cast<KernelSmem*>(smem_raw);

    heuristicTopKJob(input, N, preIdx, M, topK, outputValues, outputIndices, smem);
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
