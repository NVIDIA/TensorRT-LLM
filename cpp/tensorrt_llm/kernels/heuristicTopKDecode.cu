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

#include <cfloat>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

constexpr int BLOCK_SIZE = 512;
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

constexpr int TOP_K = 2048;
constexpr int SAFETY_MARGIN = 2048;
constexpr int MAX_CANDIDATES = TOP_K + SAFETY_MARGIN * 2;

constexpr int MAX_REFINE_ITERS = 15;
constexpr int NUM_BINS = 256;

static_assert(TOP_K % BLOCK_SIZE == 0);
static_assert(MAX_CANDIDATES % BLOCK_SIZE == 0);

struct KernelSmem
{
    alignas(16) float keys[MAX_CANDIDATES];
    alignas(16) int vals[MAX_CANDIDATES];

    int warp_counts[NUM_WARPS];
    int histogram[NUM_BINS];

    float threshold;
    int cand_count;
    int done;

    float val_lo, val_hi;
    int cnt_lo, cnt_hi;

    float pmax_saved;
    int out_count;
};

__host__ __device__ constexpr size_t getSmemSize()
{
    return sizeof(KernelSmem);
}

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
    {
        val += __shfl_down_sync(0xffffffffu, val, off);
    }
    return val;
}

__device__ __forceinline__ float warpReduceMin(float val)
{
#pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
    {
        val = fminf(val, __shfl_xor_sync(0xffffffffu, val, off));
    }
    return val;
}

__device__ __forceinline__ float warpReduceMax(float val)
{
#pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
    {
        val = fmaxf(val, __shfl_xor_sync(0xffffffffu, val, off));
    }
    return val;
}
#endif

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
    {
        c += (__ldg(&input[i]) >= threshold);
    }

    c = warpReduceSum(c);

    if (lane == 0)
    {
        smem->warp_counts[warp_id] = c;
    }
    __syncthreads();

    if (tid == 0)
    {
        int t = 0;
        for (int w = 0; w < NUM_WARPS; ++w)
        {
            t += smem->warp_counts[w];
        }
        smem->cand_count = t;
    }
}

__device__ __forceinline__ void blockFusedSnapIter(KernelSmem* smem, int count, int tid, int warp_id, int lane)
{
    float const thr = smem->threshold;

    int lge = 0;
    int lgt = 0;
    float s_up = FLT_MAX;
    float s_down = -FLT_MAX;

    for (int i = tid; i < count; i += BLOCK_SIZE)
    {
        float v = smem->keys[i];
        lge += (v >= thr);
        lgt += (v > thr);
        if (v > thr)
        {
            s_up = fminf(s_up, v);
        }
        if (v < thr)
        {
            s_down = fmaxf(s_down, v);
        }
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
        float total_up = FLT_MAX;
        float total_down = -FLT_MAX;
        for (int w = 0; w < NUM_WARPS; ++w)
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
            {
                smem->threshold = total_up;
            }
        }
        else if (cge < TOP_K)
        {
            if (total_down > -FLT_MAX)
            {
                smem->threshold = total_down;
            }
        }
    }
    __syncthreads();
}

__global__ void __launch_bounds__(BLOCK_SIZE, 1) heuristicTopKDecodeKernel(float const* __restrict__ logits,
    int const* __restrict__ seqLens, int const* __restrict__ preIdx, int* __restrict__ outIndices, int stride0,
    int next_n, int topK, int preIdxStride, int preIdxCount)
{
    int const rowIdx = blockIdx.x;
    int const seq_len = seqLens[rowIdx / next_n];
    int const N = seq_len - next_n + (rowIdx % next_n) + 1;

    float const* __restrict__ input = logits + static_cast<int64_t>(rowIdx) * stride0;
    int const* __restrict__ rowPreIdx = preIdx + static_cast<int64_t>(rowIdx) * preIdxStride;
    int* __restrict__ rowOut = outIndices + static_cast<int64_t>(rowIdx) * topK;

    extern __shared__ unsigned char smem_raw[];
    auto* smem = reinterpret_cast<KernelSmem*>(smem_raw);

    int const tid = threadIdx.x;
    int const warp_id = tid / WARP_SIZE;
    int const lane = tid & (WARP_SIZE - 1);
    unsigned const full_mask = 0xffffffffu;
    int const M = preIdxCount;

    // Phase 1: Min/Max/Mean of pre-indexed values
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
    {
        wsum += __shfl_down_sync(0xffffffffu, wsum, off);
    }
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
        float pmin = FLT_MAX;
        float pmax = -FLT_MAX;
        float psum = 0.0f;
        int pcnt = 0;
        for (int w = 0; w < NUM_WARPS; ++w)
        {
            pmin = fminf(pmin, __int_as_float(smem->histogram[w]));
            pmax = fmaxf(pmax, __int_as_float(smem->histogram[NUM_WARPS + w]));
            psum += __int_as_float(smem->histogram[NUM_WARPS * 2 + w]);
            pcnt += smem->histogram[NUM_WARPS * 3 + w];
        }
        float pmean = (pcnt > 0) ? psum / static_cast<float>(pcnt) : (pmin + pmax) * 0.5f;

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
        {
            for (int i = 0; i < topK && i < N; ++i)
            {
                rowOut[i] = i;
            }
        }
        return;
    }

    // Phase 2: Interpolation threshold search
    blockCountGE(input, N, smem->threshold, smem, tid, warp_id, lane);

    if (tid == 0)
    {
        int c = smem->cand_count;
        if (c >= TOP_K && c <= MAX_CANDIDATES)
        {
            smem->done = 1;
        }
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

    for (int iter = 0; iter < MAX_REFINE_ITERS; ++iter)
    {
        if (smem->done)
        {
            break;
        }
        if (tid == 0)
        {
            float vlo = smem->val_lo;
            float vhi = smem->val_hi;
            int clo = smem->cnt_lo;
            int chi = smem->cnt_hi;
            int target = TOP_K + SAFETY_MARGIN / 2;
            float range = vhi - vlo;
            float nv;
            if (clo > chi && range > 1e-10f)
            {
                float f = static_cast<float>(clo - target) / static_cast<float>(clo - chi);
                f = fmaxf(0.05f, fminf(0.95f, f));
                if (iter == 0)
                {
                    f = fminf(f, 0.50f);
                }
                nv = vlo + range * f;
            }
            else
            {
                nv = (vlo + vhi) * 0.5f;
            }
            if (nv <= vlo)
            {
                nv = vlo + range * 0.05f;
            }
            if (nv >= vhi)
            {
                nv = vhi - range * 0.05f;
            }
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
            {
                smem->threshold = nv;
            }
        }
        __syncthreads();
        if (smem->done)
        {
            break;
        }
        blockCountGE(input, N, smem->threshold, smem, tid, warp_id, lane);
        if (tid == 0)
        {
            int c = smem->cand_count;
            if (c >= TOP_K && c <= MAX_CANDIDATES)
            {
                smem->done = 1;
            }
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
        {
            smem->threshold = smem->val_lo;
        }
        else
        {
            smem->threshold = smem->val_hi;
        }
        smem->done = 2;
    }
    __syncthreads();

    // Phase 3: Collect candidates
    blockCountGE(input, N, smem->threshold, smem, tid, warp_id, lane);
    if (tid == 0 && smem->cand_count > MAX_CANDIDATES)
    {
        smem->val_lo = smem->threshold;
    }
    __syncthreads();

    for (int retry = 0; retry < 10 && smem->cand_count > MAX_CANDIDATES; ++retry)
    {
        if (tid == 0)
        {
            float lo = smem->val_lo;
            float hi = smem->val_hi;
            float mid = (lo + hi) * 0.5f;
            if (mid == lo)
            {
                mid = hi;
            }
            smem->threshold = mid;
        }
        __syncthreads();
        blockCountGE(input, N, smem->threshold, smem, tid, warp_id, lane);
        if (tid == 0)
        {
            int c = smem->cand_count;
            if (c > MAX_CANDIDATES)
            {
                smem->val_lo = smem->threshold;
            }
            else if (c < TOP_K)
            {
                smem->val_hi = smem->threshold;
            }
        }
        __syncthreads();
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
        {
            my_total_qual += (__ldg(&input[i]) >= thr);
        }
    }

    int thread_prefix = my_total_qual;
#pragma unroll
    for (int off = 1; off < WARP_SIZE; off *= 2)
    {
        int other = __shfl_up_sync(full_mask, thread_prefix, off);
        if (lane >= off)
        {
            thread_prefix += other;
        }
    }
    int my_excl_offset = thread_prefix - my_total_qual;
    int warp_total_qual = __shfl_sync(full_mask, thread_prefix, WARP_SIZE - 1);

    if (lane == 0)
    {
        smem->warp_counts[warp_id] = warp_total_qual;
    }
    __syncthreads();

    if (tid == 0)
    {
        int total = 0;
        for (int w = 0; w < NUM_WARPS; ++w)
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
            for (int j = 0; j < 4; ++j)
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

    // Phase 4: Selection from candidates
    int const cand_count = min(smem->cand_count, MAX_CANDIDATES);
    if (cand_count == TOP_K)
    {
        for (int i = tid; i < TOP_K; i += BLOCK_SIZE)
        {
            rowOut[i] = smem->vals[i];
        }
        return;
    }

    if (cand_count > TOP_K)
    {
        float cmin = FLT_MAX;
        float cmax = -FLT_MAX;
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

        float block_min = FLT_MAX;
        float block_max = -FLT_MAX;
        for (int w = 0; w < NUM_WARPS; ++w)
        {
            block_min = fminf(block_min, __int_as_float(smem->warp_counts[w]));
            block_max = fmaxf(block_max, __int_as_float(smem->histogram[w]));
        }
        if (block_max <= block_min)
        {
            block_max = block_min + 1e-6f;
        }

        for (int i = tid; i < NUM_BINS; i += BLOCK_SIZE)
        {
            smem->histogram[i] = 0;
        }
        __syncthreads();

        float range1 = block_max - block_min;
        float inv1 = (range1 > 0.0f) ? (static_cast<float>(NUM_BINS - 1) + 0.99f) / range1 : 0.0f;
        for (int i = tid; i < cand_count; i += BLOCK_SIZE)
        {
            int bin = static_cast<int>((smem->keys[i] - block_min) * inv1);
            bin = min(max(bin, 0), NUM_BINS - 1);
            atomicAdd(&smem->histogram[bin], 1);
        }
        __syncthreads();

        if (tid == 0)
        {
            int cum = 0;
            float thr = block_min;
            for (int b = NUM_BINS - 1; b >= 0; --b)
            {
                cum += smem->histogram[b];
                if (cum >= TOP_K)
                {
                    thr = block_min + static_cast<float>(b) * range1 / static_cast<float>(NUM_BINS);
                    break;
                }
            }
            smem->threshold = thr;
        }
        __syncthreads();

        bool snap_converged = false;
        int snap_limit = (cand_count > 128 ? cand_count / 4 : 32);
        for (int si = 0; si < snap_limit; ++si)
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
        {
            smem->out_count = 0;
        }
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
                {
                    bp = atomicAdd(&smem->out_count, cnt);
                }
                bp = __shfl_sync(full_mask, bp, 0);
                if (emit_gt && bp + moff < TOP_K)
                {
                    rowOut[bp + moff] = smem->vals[i];
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
                {
                    bp = atomicAdd(&smem->out_count, cnt);
                }
                bp = __shfl_sync(full_mask, bp, 0);
                if (emit_eq && bp + moff < TOP_K)
                {
                    rowOut[bp + moff] = smem->vals[i];
                }
            }
        }
        __syncthreads();

        int filled = min(smem->out_count, TOP_K);
        for (int i = filled + tid; i < TOP_K; i += BLOCK_SIZE)
        {
            rowOut[i] = -1;
        }
        return;
    }

    for (int i = tid; i < cand_count; i += BLOCK_SIZE)
    {
        rowOut[i] = smem->vals[i];
    }
    for (int i = cand_count + tid; i < TOP_K; i += BLOCK_SIZE)
    {
        rowOut[i] = -1;
    }
}

} // anonymous namespace

void launchHeuristicTopKDecode(float const* logits, int const* seqLens, int const* preIdx, int* outIndices,
    int stride0, int next_n, int topK, int preIdxStride, int preIdxCount, int numRows, cudaStream_t stream)
{
    static bool configured = false;
    if (!configured)
    {
        size_t smemSize = getSmemSize();
        int device = 0;
        cudaGetDevice(&device);
        int maxSmem = 0;
        cudaDeviceGetAttribute(&maxSmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        if (smemSize > 48u * 1024u && smemSize <= static_cast<size_t>(maxSmem))
        {
            cudaFuncSetAttribute(
                heuristicTopKDecodeKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smemSize));
        }
        configured = true;
    }

    heuristicTopKDecodeKernel<<<numRows, BLOCK_SIZE, getSmemSize(), stream>>>(
        logits, seqLens, preIdx, outIndices, stride0, next_n, topK, preIdxStride, preIdxCount);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
