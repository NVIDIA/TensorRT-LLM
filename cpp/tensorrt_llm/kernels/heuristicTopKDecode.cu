/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/heuristicTopKDecode.h"

#include "tensorrt_llm/common/config.h"

// Reuse all __device__ helpers and types from the standalone heuristic kernel.
// This ensures identical SASS quality for the hot loops (blockCountGE, etc.).
#include "tensorrt_llm/kernels/heuristic_topk.cuh"

#include <cfloat>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

using heuristic_topk::BLOCK_SIZE;
using heuristic_topk::blockCountGE;
using heuristic_topk::blockFusedSnapIter;
using heuristic_topk::KernelSmem;
using heuristic_topk::MAX_CANDIDATES;
using heuristic_topk::MAX_REFINE_ITERS;
using heuristic_topk::NUM_BINS;
using heuristic_topk::NUM_WARPS;
using heuristic_topk::SAFETY_MARGIN;
using heuristic_topk::TOP_K;
using heuristic_topk::WARP_SIZE;
using heuristic_topk::warpReduceMax;
using heuristic_topk::warpReduceMin;
using heuristic_topk::warpReduceSum;

// Multi-row kernel: one thread block per row, algorithm body identical to
// heuristic_topk::heuristicTopKKernel to produce the same SASS quality.
__global__ void __launch_bounds__(BLOCK_SIZE, 1) heuristicTopKMultiRowKernel(
    float const* __restrict__ logits, int const* __restrict__ seqLens, int const* __restrict__ preIdx,
    float* __restrict__ scratchValues, int* __restrict__ outIndices, int stride0, int next_n, int topK,
    int preIdxStride, int preIdxCount)
{
    int const rowIdx = blockIdx.x;

    // Per-row effective length (same logic as topKPerRowDecode)
    int const seq_len = seqLens[rowIdx / next_n];
    int const N = seq_len - next_n + (rowIdx % next_n) + 1;

    // Per-row pointers
    float const* __restrict__ input = logits + static_cast<int64_t>(rowIdx) * stride0;
    int const* __restrict__ rowPreIdx = preIdx + static_cast<int64_t>(rowIdx) * preIdxStride;
    float* __restrict__ outputValues = scratchValues + static_cast<int64_t>(rowIdx) * topK;
    int* __restrict__ outputIndices = outIndices + static_cast<int64_t>(rowIdx) * topK;
    int const M = preIdxCount;

    extern __shared__ unsigned char smem_raw[];
    auto* smem = reinterpret_cast<KernelSmem*>(smem_raw);

    int const tid = threadIdx.x;
    int const warp_id = tid / WARP_SIZE;
    int const lane = tid & (WARP_SIZE - 1);
    unsigned const full_mask = 0xffffffffu;

    // Edge case: row length <= topK — copy all valid indices, pad with -1
    if (N <= topK)
    {
        for (int i = tid; i < N; i += BLOCK_SIZE)
        {
            outputValues[i] = input[i];
            outputIndices[i] = i;
        }
        for (int i = N + tid; i < topK; i += BLOCK_SIZE)
        {
            outputValues[i] = -FLT_MAX;
            outputIndices[i] = -1;
        }
        return;
    }

    // ================================================================
    // Phase 1 — Min/Max/Mean of pre-indexed values
    // ================================================================

    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;
    int local_cnt = 0;
    for (int i = tid; i < M; i += BLOCK_SIZE)
    {
        int idx = __ldg(&rowPreIdx[i]);
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
    // Phase 2 — Interpolation threshold search (count-only)
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
        __syncthreads();

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
}

} // anonymous namespace

void launchHeuristicTopKDecode(float const* logits, int const* seqLens, int const* preIdx, int* outIndices,
    int stride0, int next_n, int topK, int preIdxStride, int preIdxCount, int numRows, cudaStream_t stream)
{
    size_t smemSize = sizeof(KernelSmem);

    static bool configured = false;
    if (!configured)
    {
        int device = 0;
        cudaGetDevice(&device);
        int maxSmem = 0;
        cudaDeviceGetAttribute(&maxSmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        if (smemSize > 48u * 1024u && smemSize <= static_cast<size_t>(maxSmem))
        {
            cudaFuncSetAttribute(heuristicTopKMultiRowKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smemSize));
        }
        configured = true;
    }

    // Scratch buffer for outputValues (kernel writes both values + indices for SASS quality)
    static float* scratchValues = nullptr;
    static int scratchCapacity = 0;
    int const needed = numRows * topK;
    if (scratchValues == nullptr || scratchCapacity < needed)
    {
        if (scratchValues)
        {
            cudaFreeAsync(scratchValues, stream);
        }
        cudaMallocAsync(&scratchValues, static_cast<size_t>(needed) * sizeof(float), stream);
        scratchCapacity = needed;
    }

    heuristicTopKMultiRowKernel<<<numRows, BLOCK_SIZE, smemSize, stream>>>(
        logits, seqLens, preIdx, scratchValues, outIndices, stride0, next_n, topK, preIdxStride, preIdxCount);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
