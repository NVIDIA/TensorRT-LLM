/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include "moeTopKFuncs.cuh"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/heuristicTopKDecode.h"
#include "tensorrt_llm/kernels/noAuxTcKernels.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;
using namespace tensorrt_llm::common;

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{
namespace heuristic_topk
{
constexpr int BLOCK_SIZE = 512;
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

constexpr int TOP_K = 2048;
constexpr int HEURISTIC_SIZE = 2048;
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

__host__ __device__ constexpr size_t getKernelSmemSize()
{
    return sizeof(KernelSmem);
}

inline void configureHeuristicTopKKernel(const void* kernel)
{
    size_t smemSize = getKernelSmemSize();
    int device = 0;
    cudaGetDevice(&device);
    int maxSmem = 0;
    cudaDeviceGetAttribute(&maxSmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (smemSize > 48u * 1024u && smemSize <= static_cast<size_t>(maxSmem))
    {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smemSize));
    }
}

__device__ __forceinline__ void storeOutput(float* outputValues, int* outputIndices, int idx, float value, int index)
{
    if (outputValues)
    {
        outputValues[idx] = value;
    }
    outputIndices[idx] = index;
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

__device__ __forceinline__ void heuristicTopKRowJob(float const* __restrict__ input, int N, int const* __restrict__ preIdx,
    int M, int topK, float* __restrict__ outputValues, int* __restrict__ outputIndices, int const /*thresholdPos*/,
    KernelSmem* smem)
{
    int const tid = threadIdx.x;
    int const warp_id = tid / WARP_SIZE;
    int const lane = tid & (WARP_SIZE - 1);
    unsigned const full_mask = 0xffffffffu;

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
                storeOutput(outputValues, outputIndices, i, input[i], i);
            }
        }
        return;
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

    int const cand_count = min(smem->cand_count, MAX_CANDIDATES);
    if (cand_count == TOP_K)
    {
        for (int i = tid; i < TOP_K; i += BLOCK_SIZE)
        {
            storeOutput(outputValues, outputIndices, i, smem->keys[i], smem->vals[i]);
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
                    storeOutput(outputValues, outputIndices, bp + moff, v, smem->vals[i]);
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
                    storeOutput(outputValues, outputIndices, bp + moff, v, smem->vals[i]);
                }
            }
        }
        __syncthreads();

        int filled = min(smem->out_count, TOP_K);
        for (int i = filled + tid; i < TOP_K; i += BLOCK_SIZE)
        {
            storeOutput(outputValues, outputIndices, i, -FLT_MAX, -1);
        }
        return;
    }

    for (int i = tid; i < cand_count; i += BLOCK_SIZE)
    {
        storeOutput(outputValues, outputIndices, i, smem->keys[i], smem->vals[i]);
    }
    for (int i = cand_count + tid; i < TOP_K; i += BLOCK_SIZE)
    {
        storeOutput(outputValues, outputIndices, i, -FLT_MAX, -1);
    }
}
} // namespace heuristic_topk

template <int step>
static inline __device__ uint32_t extractBinIdx(float x)
{
    if constexpr (step == 0)
    {
        __half hx = __float2half(x);
        uint16_t bits = __half_as_ushort(hx);
        bits = (bits & 0x8000) ? bits : ~bits & 0x7fff;
        return bits >> 5;
    }
    else
    {
        uint32_t bits = __float_as_uint(x);
        bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;

        if constexpr (step == 1)
        {
            return bits >> 21;
        }
        else if constexpr (step == 2)
        {
            return (bits >> 10) & 0x7ff;
        }
        else if constexpr (step == 3)
        {
            return bits & 0x3ff;
        }
    }
}

template <int shift>
static inline __device__ bool isPartialMatch(float x, uint32_t pattern)
{
    if constexpr (shift == 0)
    {
        return true;
    }
    uint32_t bits = __float_as_uint(x);
    bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;
    return (bits ^ pattern) >> shift == 0;
}

/**
 * Map a Func over the input data, using vectorized load instructions if
 * possible.
 *
 * @tparam T element type
 * @tparam IdxT indexing type
 * @tparam Func void (T x, IdxT idx)
 *
 * @param thread_rank rank of the calling thread among all participating threads
 * @param num_threads number of the threads that participate in processing
 * @param in the input data
 * @param len the number of elements to read
 * @param f the lambda taking two arguments (T x, IdxT idx)
 */
template <typename T, typename idxT, typename Func>
__device__ void vectorized_process(size_t thread_rank, size_t num_threads, T const* in, idxT len, Func f)
{
    constexpr int WARP_SIZE = 32;
    using WideT = float4;
    if constexpr (sizeof(T) >= sizeof(WideT))
    {
        for (idxT i = thread_rank; i < len; i += num_threads)
        {
            f(in[i], i);
        }
    }
    else
    {
        static_assert(sizeof(WideT) % sizeof(T) == 0);
        constexpr int items_per_scalar = sizeof(WideT) / sizeof(T);

        // TODO: it's UB
        union
        {
            WideT scalar;
            T array[items_per_scalar];
        } wide;

        int skip_cnt = (reinterpret_cast<size_t>(in) % sizeof(WideT))
            ? ((sizeof(WideT) - reinterpret_cast<size_t>(in) % sizeof(WideT)) / sizeof(T))
            : 0;
        if (skip_cnt > len)
        {
            skip_cnt = len;
        }
        WideT const* in_cast = reinterpret_cast<decltype(in_cast)>(in + skip_cnt);
        idxT const len_cast = (len - skip_cnt) / items_per_scalar;

        for (idxT i = thread_rank; i < len_cast; i += num_threads)
        {
            wide.scalar = in_cast[i];
            idxT const real_i = skip_cnt + i * items_per_scalar;
#pragma unroll
            for (int j = 0; j < items_per_scalar; ++j)
            {
                f(wide.array[j], real_i + j);
            }
        }

        static_assert(WARP_SIZE >= items_per_scalar);
        // and because items_per_scalar > skip_cnt, WARP_SIZE > skip_cnt
        // no need to use loop
        if (thread_rank < skip_cnt)
        {
            f(in[thread_rank], thread_rank);
        }
        // because len_cast = (len - skip_cnt) / items_per_scalar,
        // len_cast * items_per_scalar + items_per_scalar > len - skip_cnt;
        // and so
        // len - (skip_cnt + len_cast * items_per_scalar) < items_per_scalar <=
        // WARP_SIZE no need to use loop
        idxT const remain_i = skip_cnt + len_cast * items_per_scalar + thread_rank;
        if (remain_i < len)
        {
            f(in[remain_i], remain_i);
        }
    }
}

template <int step, int kNumThreadsPerBlock, int kNumBins, int kNumFinalItems, bool multipleBlocksPerRow,
    bool mergeBlocks, typename SmemFinalType, typename SmemOutputType>
__device__ bool processHistogramStep(int const* indices, float const* logits, int rowEnd, uint32_t& logitPattern,
    int& thresholdBinIdx, SmemOutputType& smemOutput, int* smemThresholdBinIdx, int* smemFinalDstIdx,
    int* smemFinalBinSize, int* smemFoundTopKValues, SmemFinalType& smemFinal, int stride1, int rowStart, int topK)
{
    // Clear the histogram.
#pragma unroll
    for (int idx = threadIdx.x; idx < kNumBins; idx += kNumThreadsPerBlock)
    {
        smemFinal.histo.data[idx] = 0;
    }

    // Make sure the histogram is ready.
    __syncthreads();

    // Update pattern
    constexpr auto patternShift = step < 2 ? 0 : step == 2 ? 21 : 10;
    if constexpr (step == 2)
    {
        logitPattern = static_cast<uint32_t>(thresholdBinIdx & 0x7ff) << patternShift;
    }
    else if constexpr (step == 3)
    {
        logitPattern |= static_cast<uint32_t>(thresholdBinIdx & 0x7ff) << patternShift;
    }

    auto distributeToBins = [&](float logit, int /* idx */ = 0)
    {
        if (isPartialMatch<patternShift>(logit, logitPattern))
        {
            uint32_t binIdx = extractBinIdx<step>(logit);
            atomicAdd(&smemFinal.histo.data[binIdx], 1);
        }
    };

    // Distribute the elements to the histogram bins.
    if (stride1 == 1)
    {
        vectorized_process(threadIdx.x, kNumThreadsPerBlock, logits + rowStart, rowEnd - rowStart, distributeToBins);
    }
    else
    {
        for (int idx = rowStart + threadIdx.x; idx < rowEnd; idx += kNumThreadsPerBlock)
        {
            float logit = logits[idx * stride1];
            distributeToBins(logit, idx);
        }
    }
    // Make sure the histogram is ready.
    __syncthreads();

    // Reads the value of the starting position in the smemOutput array
    int lastValue = smemFoundTopKValues[0];

    for (int round = 0; round < kNumBins / kNumThreadsPerBlock; round++)
    {
        // Read the values from SMEM.
        int idx = threadIdx.x + kNumThreadsPerBlock * round;
        int binCount{0};
        binCount = smemFinal.histo.data[idx];

        // Make sure each thread has read its value.
        __syncthreads();

        // Compute the prefix sum.
        int prefixSum{0}, totalSum{0};
        using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;
        Scan(smemFinal.histo.scan).ExclusiveSum(binCount, prefixSum, totalSum);

        // Update the histogram with the prefix sums.
        prefixSum += lastValue;
        totalSum += lastValue;
        smemFinal.histo.data[idx] = prefixSum;

        // Make sure the data is in shared memory.
        __syncthreads();

        // Find the last valid bin.
        bool foundThreshold = false;
        if (prefixSum < topK)
        {
            int nextPrefixSum = threadIdx.x == kNumThreadsPerBlock - 1 ? totalSum : smemFinal.histo.data[idx + 1];

            if (nextPrefixSum >= topK)
            {
                smemThresholdBinIdx[0] = idx;
                smemFinalBinSize[0] = nextPrefixSum - prefixSum;
                foundThreshold = true;
            }
        }

        // Early exit: if any thread found the threshold, we can skip remaining
        // rounds
        if (__syncthreads_or(foundThreshold))
        {
            break;
        }

        lastValue = totalSum;
    }

    // Make sure the data is in shared memory.
    __syncthreads();

    // The threshold bin.
    thresholdBinIdx = smemThresholdBinIdx[0];

    auto processBins = [&](float logit, int idx)
    {
        if (isPartialMatch<patternShift>(logit, logitPattern))
        {
            uint32_t binIdx = extractBinIdx<step>(logit);
            if (binIdx < thresholdBinIdx)
            {
                // The element is part of the top-k selection
                int dstIdx = atomicAdd(&smemFoundTopKValues[0], 1);

                if constexpr (mergeBlocks)
                {
                    smemOutput[dstIdx] = indices[idx];
                }
                else if constexpr (multipleBlocksPerRow)
                {
                    smemOutput[dstIdx] = idx + rowStart;
                    reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = logit;
                }
                else
                {
                    smemOutput[dstIdx] = idx;
                }
            }
            if constexpr (step < 3)
            {
                // Only fill the final items for sorting if the threshold bin fits
                if (binIdx == thresholdBinIdx && smemFinalBinSize[0] <= kNumFinalItems)
                {
                    int dstIdx = atomicAdd(&smemFinalDstIdx[0], 1);
                    smemFinal.items.logits[dstIdx] = logit;
                    if constexpr (mergeBlocks)
                    {
                        smemFinal.items.indices[dstIdx] = indices[idx];
                    }
                    else if constexpr (multipleBlocksPerRow)
                    {
                        smemFinal.items.indices[dstIdx] = idx + rowStart;
                    }
                    else
                    {
                        smemFinal.items.indices[dstIdx] = idx;
                    }
                }
            }
            else
            {
                if (binIdx == thresholdBinIdx)
                {
                    // The elements in the threshold bin share the same 32 bits at step 3
                    int dstIdx = atomicAdd(&smemFinal.histo.data[binIdx], 1);
                    if (dstIdx < topK)
                    {
                        if constexpr (mergeBlocks)
                        {
                            smemOutput[dstIdx] = indices[idx];
                        }
                        else if constexpr (multipleBlocksPerRow)
                        {
                            smemOutput[dstIdx] = idx + rowStart;
                            reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = logit;
                        }
                        else
                        {
                            smemOutput[dstIdx] = idx;
                        }
                    }
                }
            }
        }
    };

    if (stride1 == 1)
    {
        vectorized_process(threadIdx.x, kNumThreadsPerBlock, logits + rowStart, rowEnd - rowStart, processBins);
    }
    else
    {
        for (int idx = rowStart + threadIdx.x; idx < rowEnd; idx += kNumThreadsPerBlock)
        {
            float logit = logits[idx * stride1];
            processBins(logit, idx);
        }
    }

    // Make sure the elements are in shared memory.
    __syncthreads();

    // Check if we should continue to next step
    return smemFinalBinSize[0] > kNumFinalItems;
}

// Follows half - 11 - 11 - 10 bit iterations
template <int kNumThreadsPerBlock, int kNumBins, bool useRadixSort, bool multipleBlocksPerRow = false,
    bool mergeBlocks = false>
static __device__ void topKPerRowJob(int const* indices, float const* logits, int rowStart, int rowEnd, int* outIndices,
    float* outLogits, int stride1, int topK)
{
    // The number of slots for the final pass.
    static constexpr int kNumFinalItems = 2048;
    // The number of elements per thread for the final sort.
    static constexpr int kNumFinalItemsPerThread = kNumFinalItems / kNumThreadsPerBlock;
    // The class to sort the elements during the final pass.
    using FinalSort = cub::BlockRadixSort<float, kNumThreadsPerBlock, kNumFinalItemsPerThread, int>;
    using FinalSortTempStorage = std::conditional_t<useRadixSort, typename FinalSort::TempStorage, int>;
    // The class to compute the inclusive prefix-sum over the histogram.
    using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;

    // The structure to store the final items (for the final pass).
    struct FinalItems
    {
        // Shared memory to store the indices for the final pass.
        int indices[kNumFinalItems];
        // Shared memory to store the logits for the final pass.
        float logits[kNumFinalItems];
    };

    struct Histogram
    {
        typename Scan::TempStorage scan;
        int data[kNumBins];
    };

    // Shared memory to compute the block sort.
    __shared__ union
    {
        FinalItems items;
        FinalSortTempStorage finalSort;
        Histogram histo;
    } smemFinal;

    // Shared memory to store the selected indices.
    // If we are processing using multiple blocks, we need to store the logits and
    // indices.
    extern __shared__ int32_t smemOutput[];

    // Shared memory to store the threshold bin.
    __shared__ int smemThresholdBinIdx[1];
    // Shared memory counter to register the candidates for the final phase.
    __shared__ int smemFinalDstIdx[1];
    // Shared memory to determine if the threshold bin fits in the final items.
    __shared__ int smemFinalBinSize[1];
    // Shared memory to keep track of the top-k values found so far by the
    // previous iterations
    __shared__ int smemFoundTopKValues[1];

    // The length of the row.
    int rowLen = rowEnd - rowStart;

    // Shortcut if the length of the row is smaller than Top-K. Indices are not
    // sorted by their corresponding logit.
    if (rowLen <= topK)
    {
        for (int rowIt = threadIdx.x; rowIt < rowLen; rowIt += kNumThreadsPerBlock)
        {
            if constexpr (multipleBlocksPerRow)
            {
                outIndices[rowIt] = rowIt + rowStart;
                outLogits[rowIt] = logits[rowIt + rowStart];
            }
            else
            {
                outIndices[rowIt] = rowIt;
            }
        }
        for (int rowIt = rowLen + threadIdx.x; rowIt < topK; rowIt += kNumThreadsPerBlock)
        {
            outIndices[rowIt] = -1;
            if constexpr (multipleBlocksPerRow)
            {
                outLogits[rowIt] = -FLT_MAX;
            }
        }

        return;
    }
    // Initialize values
    if (threadIdx.x == 0)
    {
        smemFinalDstIdx[0] = 0;
        smemFoundTopKValues[0] = 0;
    }
    __syncthreads();
    int thresholdBinIdx = -1;
    uint32_t logitPattern = 0;

    // Step 0: Process first 11 bits of half representation
    bool continueToNextStep
        = processHistogramStep<0, kNumThreadsPerBlock, kNumBins, kNumFinalItems, multipleBlocksPerRow, mergeBlocks>(
            indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput, smemThresholdBinIdx, smemFinalDstIdx,
            smemFinalBinSize, smemFoundTopKValues, smemFinal, stride1, rowStart, topK);

    if (continueToNextStep)
    {
        // Step 1: Process next 11 bits
        continueToNextStep
            = processHistogramStep<1, kNumThreadsPerBlock, kNumBins, kNumFinalItems, multipleBlocksPerRow, mergeBlocks>(
                indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput, smemThresholdBinIdx,
                smemFinalDstIdx, smemFinalBinSize, smemFoundTopKValues, smemFinal, stride1, rowStart, topK);
    }

    if (continueToNextStep)
    {
        // Step 2: Process next 11 bits
        continueToNextStep
            = processHistogramStep<2, kNumThreadsPerBlock, kNumBins, kNumFinalItems, multipleBlocksPerRow, mergeBlocks>(
                indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput, smemThresholdBinIdx,
                smemFinalDstIdx, smemFinalBinSize, smemFoundTopKValues, smemFinal, stride1, rowStart, topK);
    }

    if (continueToNextStep)
    {
        // Step 3: Process last 10 bits
        processHistogramStep<3, kNumThreadsPerBlock, kNumBins, kNumFinalItems, multipleBlocksPerRow, mergeBlocks>(
            indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput, smemThresholdBinIdx, smemFinalDstIdx,
            smemFinalBinSize, smemFoundTopKValues, smemFinal, stride1, rowStart, topK);
    }

    if (!continueToNextStep)
    {
        // The histogram did not proceed to the final 10 bits, therefore we need to
        // sort the final items The logits of the elements to be sorted in the final
        // pass.
        if constexpr (useRadixSort)
        {
            // Sorting with radix sort
            float finalLogits[kNumFinalItemsPerThread];
            // The indices of the elements to be sorted in the final pass.
            int finalIndices[kNumFinalItemsPerThread];

#pragma unroll
            for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii)
            {
                finalLogits[ii] = -FLT_MAX;
            }

            // Read the elements from SMEM.
#pragma unroll
            for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii)
            {
                int srcIdx = ii * kNumThreadsPerBlock + threadIdx.x;
                if (srcIdx < smemFinalDstIdx[0])
                {
                    finalLogits[ii] = smemFinal.items.logits[srcIdx];
                    finalIndices[ii] = smemFinal.items.indices[srcIdx];
                }
            }
            // Make sure the shared memory has been read.
            __syncthreads();

            // Sort the elements.
            FinalSort(smemFinal.finalSort).SortDescendingBlockedToStriped(finalLogits, finalIndices);

            // Copy the data back to the shared memory storage.
            int baseIdx = smemFoundTopKValues[0];

#pragma unroll
            for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii)
            {
                int srcIdx = ii * kNumThreadsPerBlock + threadIdx.x;
                int dstIdx = baseIdx + srcIdx;

                if (dstIdx < topK)
                {
                    smemOutput[dstIdx] = finalIndices[ii];
                    if constexpr (multipleBlocksPerRow)
                    {
                        reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = finalLogits[ii];
                    }
                }
            }
        }
        else
        {
            // Sorting with insertion sort
            auto baseIdx = smemFoundTopKValues[0];
            for (int i = threadIdx.x; i < smemFinalDstIdx[0]; i += kNumThreadsPerBlock)
            {
                int outIndex = 0;
                auto logit = smemFinal.items.logits[i];
                for (int j = 0; j < smemFinalDstIdx[0]; j++)
                {
                    auto otherLogit = smemFinal.items.logits[j];
                    if (logit < otherLogit || (logit == otherLogit && i < j))
                    {
                        outIndex++;
                    }
                }
                // Store if outIndex is in bounds
                if (outIndex + baseIdx < topK)
                {
                    smemOutput[outIndex + baseIdx] = smemFinal.items.indices[i];
                    if constexpr (multipleBlocksPerRow)
                    {
                        reinterpret_cast<float*>(smemOutput + topK)[outIndex + baseIdx] = smemFinal.items.logits[i];
                    }
                }
            }
        }
        __syncthreads();
    }

    // Store to global memory.
    for (int i = threadIdx.x; i < topK; i += kNumThreadsPerBlock)
    {
        if constexpr (multipleBlocksPerRow)
        {
            outIndices[i] = smemOutput[i];
            outLogits[i] = reinterpret_cast<float*>(smemOutput + topK)[i];
        }
        else
        {
            if (stride1 == 1)
            {
                // stride1 == 1 will use vectorized_process, which indexes already skip the rowStart.
                outIndices[i] = smemOutput[i];
            }
            else
            {
                outIndices[i] = smemOutput[i] - rowStart;
            }
        }
    }
}
} // namespace

template <int kNumThreadsPerBlock, bool useRadixSort>
static __global__ __launch_bounds__(kNumThreadsPerBlock) void topKPerRowPrefill(float const* logits,
    int const* rowStarts, int const* rowEnds, int* outIndices, int stride0, int stride1, int const topK,
    int const offsetIndex)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif
    // The number of bins in the histogram.
    static constexpr int kNumBins = 2048;

    // The row computed by this block.
    int rowIdx = blockIdx.x + offsetIndex;

    // The range of logits within the row.
    int rowStart = rowStarts[rowIdx];
    int rowEnd = rowEnds[rowIdx];

    // Local pointers to this block
    outIndices += static_cast<int64_t>(rowIdx) * topK;
    logits += static_cast<int64_t>(rowIdx) * stride0;

    topKPerRowJob<kNumThreadsPerBlock, kNumBins, useRadixSort>(
        nullptr, logits, rowStart, rowEnd, outIndices, nullptr, stride1, topK);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <int kNumThreadsPerBlock, bool useRadixSort, bool multipleBlocksPerRow = false, bool mergeBlocks = false>
static __global__ __launch_bounds__(kNumThreadsPerBlock) void topKPerRowDecode(float const* logits, int const* seqLens,
    int* outIndices, int stride0, int stride1, int const topK, int next_n, float* outLogits = nullptr,
    int const numBlocksToMerge = 0, int const* indices = nullptr)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif
    // The number of bins in the histogram.
    static constexpr int kNumBins = 2048;

    // The row computed by this block.
    int rowIdx = blockIdx.x;

    // The range of logits within the row.
    int rowStart = 0;
    int seq_len = seqLens[rowIdx / next_n];
    int rowEnd = seq_len - next_n + (rowIdx % next_n) + 1;

    // Local pointers to this block
    if constexpr (!multipleBlocksPerRow && !mergeBlocks)
    {
        outIndices += static_cast<int64_t>(rowIdx) * topK;
    }
    else if constexpr (multipleBlocksPerRow)
    {
        auto const blockSize = rowEnd / gridDim.y; // 16384 / 2 = 8192
        rowStart = blockSize * blockIdx.y;         // 8192 * 1 = 8192
        rowEnd = gridDim.y == blockIdx.y + 1 ? rowEnd : rowStart + blockSize;
        outIndices += static_cast<int64_t>(rowIdx) * gridDim.y * topK + blockIdx.y * topK;
        outLogits += static_cast<int64_t>(rowIdx) * gridDim.y * topK + blockIdx.y * topK;
    }
    else if constexpr (mergeBlocks)
    {
        rowEnd = numBlocksToMerge * topK;
        indices += static_cast<int64_t>(rowIdx) * numBlocksToMerge * topK;
        outIndices += static_cast<int64_t>(rowIdx) * topK;
    }
    logits += static_cast<int64_t>(rowIdx) * stride0;

    topKPerRowJob<kNumThreadsPerBlock, kNumBins, useRadixSort, multipleBlocksPerRow, mergeBlocks>(
        indices, logits, rowStart, rowEnd, outIndices, outLogits, stride1, topK);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

static __global__ __launch_bounds__(heuristic_topk::BLOCK_SIZE, 1) void topKPerRowDecodeHeuristic(
    float const* logits, int const* seqLens, int const* preIdx, int* outIndices, int stride0, int next_n, int const topK,
    int const preIdxStride, int const preIdxCount)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif
    int rowIdx = blockIdx.x;
    int seq_len = seqLens[rowIdx / next_n];
    int rowEnd = seq_len - next_n + (rowIdx % next_n) + 1;

    logits += static_cast<int64_t>(rowIdx) * stride0;
    preIdx += static_cast<int64_t>(rowIdx) * preIdxStride;
    outIndices += static_cast<int64_t>(rowIdx) * topK;

    extern __shared__ unsigned char smem_raw[];
    auto* smem = reinterpret_cast<heuristic_topk::KernelSmem*>(smem_raw);
    heuristic_topk::heuristicTopKRowJob(logits, rowEnd, preIdx, preIdxCount, topK, nullptr, outIndices, -1, smem);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

void invokeIndexerTopKDecode(float const* logits, int const* seqLens, int* indices, float* outLogitsAux,
    int* outIndicesAux, int const splitWorkThreshold, int const numRows, int const numColumns, int const stride0,
    int const stride1, int const next_n, int const topK, int const* preIdx, int const preIdxStride,
    int const preIdxCount, cudaStream_t const stream)
{

    constexpr int kSortingAlgorithmThreshold = 12288;
    constexpr int kDefaultSplitWorkThreshold = 200 * 1000;
    constexpr int kNumThreadsPerBlock = 512;
    int const effectiveSplitWorkThreshold = splitWorkThreshold > 0 ? splitWorkThreshold : kDefaultSplitWorkThreshold;
    bool const canUseHeuristic = preIdx != nullptr && stride1 == 1 && topK == heuristic_topk::TOP_K
        && preIdxCount == heuristic_topk::HEURISTIC_SIZE && preIdxStride >= preIdxCount
        && numColumns < effectiveSplitWorkThreshold;

    if (canUseHeuristic)
    {
        auto* kernel_instance = &topKPerRowDecodeHeuristic;
        heuristic_topk::configureHeuristicTopKKernel(reinterpret_cast<const void*>(kernel_instance));

        cudaLaunchConfig_t config;
        config.gridDim = numRows;
        config.blockDim = heuristic_topk::BLOCK_SIZE;
        config.dynamicSmemBytes = heuristic_topk::getKernelSmemSize();
        config.stream = stream;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
        config.numAttrs = 1;
        config.attrs = attrs;

        cudaLaunchKernelEx(
            &config, kernel_instance, logits, seqLens, preIdx, indices, stride0, next_n, topK, preIdxStride, preIdxCount);
    }
    else if (numColumns < kSortingAlgorithmThreshold)
    {
        // Use insertion sort
        auto* kernel_instance = &topKPerRowDecode<kNumThreadsPerBlock, false>;

        cudaLaunchConfig_t config;
        config.gridDim = numRows;
        config.blockDim = kNumThreadsPerBlock;
        config.dynamicSmemBytes = topK * sizeof(int32_t);
        config.stream = stream;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
        config.numAttrs = 1;
        config.attrs = attrs;

        cudaLaunchKernelEx(
            &config, kernel_instance, logits, seqLens, indices, stride0, stride1, topK, next_n, nullptr, 0, nullptr);
    }
    else if (numColumns < effectiveSplitWorkThreshold)
    {
        // From this threshold, use radix sort instead
        auto* kernel_instance = &topKPerRowDecode<kNumThreadsPerBlock, true>;

        cudaLaunchConfig_t config;
        config.gridDim = numRows;
        config.blockDim = kNumThreadsPerBlock;
        config.dynamicSmemBytes = topK * sizeof(int32_t);
        config.stream = stream;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
        config.numAttrs = 1;
        config.attrs = attrs;

        cudaLaunchKernelEx(
            &config, kernel_instance, logits, seqLens, indices, stride0, stride1, topK, next_n, nullptr, 0, nullptr);
    }
    else
    {
        // Long sequences are run in two steps
        constexpr auto multipleBlocksPerRowConfig = 10;
        auto* kernel_instance_part1 = &topKPerRowDecode<kNumThreadsPerBlock, true, true>;
        cudaLaunchConfig_t config_part1;
        config_part1.gridDim = dim3(numRows, multipleBlocksPerRowConfig);
        config_part1.blockDim = kNumThreadsPerBlock;
        config_part1.dynamicSmemBytes = 2 * topK * sizeof(int32_t);
        config_part1.stream = stream;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
        config_part1.numAttrs = 1;
        config_part1.attrs = attrs;

        cudaLaunchKernelEx(&config_part1, kernel_instance_part1, logits, seqLens, outIndicesAux, stride0, stride1, topK,
            next_n, outLogitsAux, 0, nullptr);

        constexpr int kNumThreadsPerBlockMerge = 1024;
        auto* kernel_instance_part2 = &topKPerRowDecode<kNumThreadsPerBlockMerge, true, false, true>;
        cudaLaunchConfig_t config_part2;
        config_part2.gridDim = numRows;
        config_part2.blockDim = kNumThreadsPerBlockMerge;
        config_part2.dynamicSmemBytes = topK * sizeof(int32_t);
        config_part2.stream = stream;
        // Reuse attrs array since part1 kernel has already been launched
        config_part2.numAttrs = 1;
        config_part2.attrs = attrs;

        cudaLaunchKernelEx(&config_part2, kernel_instance_part2, outLogitsAux, seqLens, indices,
            multipleBlocksPerRowConfig * topK, 1, topK, next_n, nullptr, multipleBlocksPerRowConfig, outIndicesAux);
    }
    sync_check_cuda_error(stream);
}

void invokeIndexerTopKPrefill(float const* logits, int const* rowStarts, int const* rowEnds, int* indices,
    int const numRows, int const numColumns, int const stride0, int const stride1, int const topK,
    cudaStream_t const stream)
{
    constexpr int kSortingAlgorithmThreshold = 12288;
    constexpr int kNumThreadsPerBlock = 512;

    int numInsertionBlocks = std::min(numRows, kSortingAlgorithmThreshold);
    topKPerRowPrefill<kNumThreadsPerBlock, false>
        <<<numInsertionBlocks, kNumThreadsPerBlock, topK * sizeof(int32_t), stream>>>(
            logits, rowStarts, rowEnds, indices, stride0, stride1, topK, 0);

    if (numRows > kSortingAlgorithmThreshold)
    {
        int numRadixBlocks = numRows - kSortingAlgorithmThreshold;
        topKPerRowPrefill<kNumThreadsPerBlock, true>
            <<<numRadixBlocks, kNumThreadsPerBlock, topK * sizeof(int32_t), stream>>>(
                logits, rowStarts, rowEnds, indices, stride0, stride1, topK, kSortingAlgorithmThreshold);
    }

    sync_check_cuda_error(stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
