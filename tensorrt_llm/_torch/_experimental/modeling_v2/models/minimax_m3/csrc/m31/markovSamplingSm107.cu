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

// U-A70's DSpark Markov-block sampler (speculative/markov_sampling.py, Triton) as CUDA C++, bit for bit.
// v2: the draw batches its 4 quarters' block reductions (2 barriers instead of 24) and vectorizes the 4-lane accesses.
// v3: the proposal hoists its loads above its reductions; optional pick fused into the draw's last chunk CTA.
// v5: pick without the thread-0 broadcast (every thread derives the token; 2-wide embedding copy); proposal kernel
// templated on the vector path and on chunks per CTA (one statistics reduction per kChunksPerCta chunks).
// v4: logits * (1 / T) is mul.rn.ftz as Triton's libdevice.mul_rn (draw + proposal); optional PDL (wait at entry,
// trigger at exit); optional proposal of position pos - 1 inside the draw of position pos (same chunk CTA work).
// Triton's libdevice log / log1p are compiled FTZ; their domain here (u >= 1.16e-10) never has a subnormal input or
// intermediate, so the non-FTZ logf / log1pf give the same bits.
// The instruction choices follow the Triton 3.8 PTX of the three kernels (dumped from the cut-3.3.7 image):
//   1 / T                 div.full.f32
//   exp(x)                mul.f32 x, log2(e) ; ex2.approx.f32 (no ftz)
//   log, log1p            libdevice (__nv_logf / __nv_log1pf = CUDA logf / log1pf; build with -fmad=false)
//   Philox                4x32-10, key (seed lo, seed hi), counter (lane, TAG, offset lo, offset hi)
//   sums                  in-thread pairwise ((e0 + e2) + (e1 + e3)), warp butterfly 16..1, cross-warp
//                         ((w0 + w2) + (w1 + w3)) through shared memory
//   running sum           fma(exp(run_max - new_max), run_sum, lane_sum)
//   proposal total        first butterfly step fma(e, run_sum, partner's product), then adds
//   argmax                ties to the lowest index; across the 4 lane groups a strict '>' keeps the earlier group
// Layout as Triton's: 4 warps (128 threads); the draw / proposal tiles are [512] x 4 with 4 consecutive lanes per
// thread (sizePerThread 4); the pick / total tiles are [128] with one lane per thread.
// In-tree (kernels verbatim; the unused v4 proposal kernel dropped): compiled with -fmad=false
// (kernels/CMakeLists.txt), as measured, so logf / log1pf keep libdevice's unfused sequence. Every
// vocab runs here. Bit for bit the Triton sampler on the served class (vocab % 16 == 0, 65..128 chunks of 2048); the
// pick / fused pick / proposal statistics also take more than 128 chunks (each thread first folds its chunks t,
// t + 128, ... in chunk order), which on <= 128 chunks is the one-element path unchanged.
#include "minimax_m3/csrc/m31/markovSamplingSm107.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cuda_bf16.h>
#include <math.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{
namespace
{

constexpr int kThreads = 128;
constexpr int kBlock = 2048;
constexpr int kQuarter = kBlock / 4; // 512 lanes per Philox call
constexpr uint32_t kTag = 0x9E3779B9u;
constexpr float kLog2e = 1.4426950408889634f;

__device__ __forceinline__ float ex2Approx(float x)
{
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float tritonExp(float x)
{
    return ex2Approx(__fmul_rn(x, kLog2e));
}

__device__ __forceinline__ float mulRnFtz(float a, float b)
{
    float y;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

__device__ __forceinline__ void pdlWait(bool pdl)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    if (pdl)
        asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void pdlTrigger(bool pdl)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    if (pdl)
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

__device__ __forceinline__ float divFull(float a, float b)
{
    float y;
    asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

__device__ __forceinline__ void philox(
    uint64_t seed, uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3, uint32_t out[4])
{
    uint32_t k0 = static_cast<uint32_t>(seed & 0xFFFFFFFFull);
    uint32_t k1 = static_cast<uint32_t>((seed >> 32) & 0xFFFFFFFFull);
#pragma unroll
    for (int r = 0; r < 10; ++r)
    {
        uint32_t const p0 = c0, p2 = c2;
        c0 = __umulhi(0xCD9E8D57u, p2) ^ c1 ^ k0;
        c2 = __umulhi(0xD2511F53u, p0) ^ c3 ^ k1;
        c1 = 0xCD9E8D57u * p2;
        c3 = 0xD2511F53u * p0;
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    out[0] = c0;
    out[1] = c1;
    out[2] = c2;
    out[3] = c3;
}

//! Sum over the CTA's 128 threads in Triton's order; every thread gets the result.
__device__ __forceinline__ float blockSum(float v, float* smem)
{
    int const lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1)
        v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, m));
    __syncthreads();
    if (lane == 0)
        smem[warp] = v;
    __syncthreads();
    float w = smem[lane & 3];
    w = __fadd_rn(w, __shfl_xor_sync(0xFFFFFFFFu, w, 2));
    w = __fadd_rn(w, __shfl_xor_sync(0xFFFFFFFFu, w, 1));
    return w;
}

__device__ __forceinline__ float blockMax(float v, float* smem)
{
    int const lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, m));
    __syncthreads();
    if (lane == 0)
        smem[warp] = v;
    __syncthreads();
    float w = smem[lane & 3];
    w = fmaxf(w, __shfl_xor_sync(0xFFFFFFFFu, w, 2));
    w = fmaxf(w, __shfl_xor_sync(0xFFFFFFFFu, w, 1));
    return w;
}

//! (key, index) argmax over the CTA, ties to the lowest index (NaN never wins: keys here are finite or -inf).
__device__ __forceinline__ void blockArgmax(float& key, int& idx, float* fsm, int* ism)
{
    int const lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1)
    {
        float const k2 = __shfl_xor_sync(0xFFFFFFFFu, key, m);
        int const i2 = __shfl_xor_sync(0xFFFFFFFFu, idx, m);
        if (k2 > key || (k2 == key && i2 < idx))
        {
            key = k2;
            idx = i2;
        }
    }
    __syncthreads();
    if (lane == 0)
    {
        fsm[warp] = key;
        ism[warp] = idx;
    }
    __syncthreads();
    key = fsm[0];
    idx = ism[0];
#pragma unroll
    for (int w = 1; w < 4; ++w)
        if (fsm[w] > key || (fsm[w] == key && ism[w] < idx))
        {
            key = fsm[w];
            idx = ism[w];
        }
}

//! This thread's (key, chunk) over chunks t, t + 128, ... of a row's per-chunk argmax keys: strict '>' keeps the
//! earlier chunk on ties. With numChunks <= 128 it is the one-element load (key = keys[t] or -inf, index t). kCg reads
//! through L2 (another CTA of this grid wrote the keys).
template <bool kCg>
__device__ __forceinline__ void threadChunkArgmax(float const* keys, int64_t numChunks, float& key, int& idx)
{
    key = -INFINITY;
    idx = threadIdx.x;
    for (int64_t c = threadIdx.x; c < numChunks; c += kThreads)
    {
        float const k = kCg ? __ldcg(keys + c) : keys[c];
        if (k > key)
        {
            key = k;
            idx = static_cast<int>(c);
        }
    }
}

//! blockArgmax on fresh shared arrays: no entry barrier.
__device__ __forceinline__ void blockArgmaxFresh(float& key, int& idx, float* fsm, int* ism)
{
    int const lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1)
    {
        float const k2 = __shfl_xor_sync(0xFFFFFFFFu, key, m);
        int const i2 = __shfl_xor_sync(0xFFFFFFFFu, idx, m);
        if (k2 > key || (k2 == key && i2 < idx))
        {
            key = k2;
            idx = i2;
        }
    }
    if (lane == 0)
    {
        fsm[warp] = key;
        ism[warp] = idx;
    }
    __syncthreads();
    key = fsm[0];
    idx = ism[0];
#pragma unroll
    for (int w = 1; w < 4; ++w)
        if (fsm[w] > key || (fsm[w] == key && ism[w] < idx))
        {
            key = fsm[w];
            idx = ism[w];
        }
}

template <typename T>
struct Pair;

template <>
struct Pair<__nv_bfloat16>
{
    using type = __nv_bfloat162;
};

template <>
struct Pair<float>
{
    using type = float2;
};

//! emb[row, :rank] = valid ? w1[token, :rank] : 0, two elements per access when rank and the stride are even.
template <typename TW>
__device__ __forceinline__ void copyEmbRow(TW const* __restrict__ w1, int64_t w1Stride, TW* __restrict__ emb,
    int64_t row, int64_t rank, int64_t token, bool valid)
{
    using P = typename Pair<TW>::type;
    TW const* src = w1 + token * w1Stride;
    TW* dst = emb + row * rank;
    if (rank % 2 == 0 && w1Stride % 2 == 0)
    {
        P zero;
        *reinterpret_cast<TW*>(&zero) = TW(0.0f);
        reinterpret_cast<TW*>(&zero)[1] = TW(0.0f);
        for (int64_t r = 2 * threadIdx.x; r < rank; r += 2 * kThreads)
            *reinterpret_cast<P*>(dst + r) = valid ? *reinterpret_cast<P const*>(src + r) : zero;
    }
    else
        for (int64_t r = threadIdx.x; r < rank; r += kThreads)
            dst[r] = valid ? src[r] : TW(0.0f);
}

__device__ __forceinline__ float ldf(float const* p)
{
    return *p;
}

__device__ __forceinline__ float ldf(__nv_bfloat16 const* p)
{
    return __bfloat162float(*p);
}

template <typename T>
__device__ __forceinline__ void load4(T const* p, float out[4]);

template <>
__device__ __forceinline__ void load4<float>(float const* p, float out[4])
{
    float4 const v = *reinterpret_cast<float4 const*>(p);
    out[0] = v.x;
    out[1] = v.y;
    out[2] = v.z;
    out[3] = v.w;
}

template <>
__device__ __forceinline__ void load4<__nv_bfloat16>(__nv_bfloat16 const* p, float out[4])
{
    uint2 const u = *reinterpret_cast<uint2 const*>(p);
    __nv_bfloat162 const lo = *reinterpret_cast<__nv_bfloat162 const*>(&u.x);
    __nv_bfloat162 const hi = *reinterpret_cast<__nv_bfloat162 const*>(&u.y);
    out[0] = __low2float(lo);
    out[1] = __high2float(lo);
    out[2] = __low2float(hi);
    out[3] = __high2float(hi);
}

__device__ __forceinline__ void proposalChunk(int64_t row, int64_t posIdx, int64_t rowPos, int64_t chunk,
    float const* __restrict__ biased, float const* __restrict__ temps, float const* __restrict__ maxIn,
    float const* __restrict__ sumIn, int64_t const* __restrict__ slots, float* __restrict__ probs,
    int64_t probsSlotStride, int64_t probsPosStride, int64_t vocab, int64_t numChunks, bool vecPath, float* maxS,
    float* sumS);

//! The draw of position `pos`: grid (numGens, numChunks), 128 threads. Mirrors _draw_kernel. v2: the four quarters'
//! block reductions are batched (one shared-memory round for the argmaxes and maxes, one for the sums; each value
//! keeps its v1 / Triton reduction order), and with vecPath the 4 lanes of a thread move as one 16-byte (fp32) or
//! 8-byte (bf16) access.
template <typename TLogit, typename TBias>
__global__ void __launch_bounds__(kThreads) drawKernel(TLogit const* __restrict__ logits, float* __restrict__ biased,
    TBias const* __restrict__ bias, float const* __restrict__ temps, int64_t const* __restrict__ seeds,
    int64_t const* __restrict__ offsets, int64_t const* __restrict__ slots, float* __restrict__ keyOut,
    int32_t* __restrict__ indexOut, float* __restrict__ maxOut, float* __restrict__ sumOut, int64_t pos, int64_t numPos,
    int64_t offsetStep, int64_t paddingSlot, int64_t vocab, int64_t numChunks, bool vecPath,
    int32_t* __restrict__ arrivals, int32_t* __restrict__ tokensOut, TBias const* __restrict__ w1, int64_t w1Stride,
    TBias* __restrict__ emb, int64_t rank, bool pdl, float* __restrict__ probs, int64_t probsSlotStride,
    int64_t probsPosStride, bool propVec)
{
    __shared__ float keyS[4][4], maxS[4][4], sumS[4][4];
    __shared__ float pMaxS[4], pSumS[4];
    __shared__ bool isLast;
    pdlWait(pdl);
    // v4 fused proposal (probs != nullptr): this chunk CTA also writes q of position pos - 1 (its statistics are
    // final).
    if (probs != nullptr && pos > 0)
        proposalChunk(blockIdx.x, pos - 1, static_cast<int64_t>(blockIdx.x) * numPos + pos - 1, blockIdx.y, biased,
            temps, maxOut, sumOut, slots, probs, probsSlotStride, probsPosStride, vocab, numChunks, propVec, pMaxS,
            pSumS);
    __shared__ int idxS[4][4];
    int const lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int64_t const row = blockIdx.x;
    int64_t const chunk = blockIdx.y;
    float const invT = divFull(1.0f, temps[row]);
    bool const isPadding = slots[row] == paddingSlot;
    int64_t const offset = offsets[row] + pos * offsetStep;
    uint64_t const seed = static_cast<uint64_t>(seeds[row]);
    uint32_t const offLo = static_cast<uint32_t>(offset & 0xFFFFFFFFll);
    uint32_t const offHi = static_cast<uint32_t>((offset >> 32) & 0xFFFFFFFFll);
    // Thread t holds lanes 4t .. 4t + 3 of each 512-lane quarter.
    uint32_t bits[4][4]; // [lane e][quarter j]
#pragma unroll
    for (int e = 0; e < 4; ++e)
        philox(seed, static_cast<uint32_t>(chunk * kQuarter + 4 * threadIdx.x + e), kTag, offLo, offHi, bits[e]);
    int64_t const rowStart = (row * numPos + pos) * vocab;
    TBias const* biasRow = bias + row * vocab;
    float scaled[4][4];
    float qKey[4], qMax[4];
    int qIdx[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        int64_t const c0 = chunk * kBlock + j * kQuarter + 4 * threadIdx.x;
        float base[4], bv[4], lg[4];
        if (vecPath && c0 + 3 < vocab)
        {
            load4(logits + rowStart + c0, base);
            load4(biasRow + c0, bv);
#pragma unroll
            for (int e = 0; e < 4; ++e)
                lg[e] = __fadd_rn(isPadding ? 0.0f : base[e], bv[e]);
            *reinterpret_cast<float4*>(biased + rowStart + c0) = make_float4(lg[0], lg[1], lg[2], lg[3]);
        }
        else
        {
#pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                bool const mask = c0 + e < vocab;
                float b0 = mask ? ldf(logits + rowStart + c0 + e) : 0.0f;
                if (isPadding)
                    b0 = 0.0f;
                lg[e] = __fadd_rn(b0, mask ? ldf(biasRow + c0 + e) : 0.0f);
                if (mask)
                    biased[rowStart + c0 + e] = lg[e];
            }
        }
        float laneKey = -INFINITY, laneMax = -INFINITY;
        int laneIdx = 0x7FFFFFFF;
#pragma unroll
        for (int e = 0; e < 4; ++e)
        {
            int64_t const col = c0 + e;
            bool const mask = col < vocab;
            scaled[j][e] = mask ? mulRnFtz(lg[e], invT) : -INFINITY;
            float const u
                = fminf(__fmul_rn(__fadd_rn(__uint2float_rn(bits[e][j]), 0.5f), 2.3283064365386963e-10f), 0.99999994f);
            float const key = __fsub_rn(scaled[j][e], logf(-log1pf(-u)));
            if (key > laneKey || (key == laneKey && static_cast<int>(col) < laneIdx))
            {
                laneKey = key;
                laneIdx = static_cast<int>(col);
            }
            laneMax = fmaxf(laneMax, scaled[j][e]);
        }
        qKey[j] = laneKey;
        qIdx[j] = laneIdx;
        qMax[j] = laneMax;
    }
    // Round 1: the 4 quarters' (key, index) argmaxes and maxes, each in blockArgmax / blockMax order.
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1)
    {
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            float const k2 = __shfl_xor_sync(0xFFFFFFFFu, qKey[j], m);
            int const i2 = __shfl_xor_sync(0xFFFFFFFFu, qIdx[j], m);
            if (k2 > qKey[j] || (k2 == qKey[j] && i2 < qIdx[j]))
            {
                qKey[j] = k2;
                qIdx[j] = i2;
            }
            qMax[j] = fmaxf(qMax[j], __shfl_xor_sync(0xFFFFFFFFu, qMax[j], m));
        }
    }
    if (lane == 0)
    {
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            keyS[j][warp] = qKey[j];
            idxS[j][warp] = qIdx[j];
            maxS[j][warp] = qMax[j];
        }
    }
    __syncthreads();
    float bestKey = -INFINITY;
    int bestIndex = 0;
    float runMax = -INFINITY;
    float prevMax[4], newMax[4], local[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        float k = keyS[j][0];
        int i = idxS[j][0];
#pragma unroll
        for (int w = 1; w < 4; ++w)
            if (keyS[j][w] > k || (keyS[j][w] == k && idxS[j][w] < i))
            {
                k = keyS[j][w];
                i = idxS[j][w];
            }
        if (k > bestKey)
            bestIndex = i;
        bestKey = fmaxf(bestKey, k);
        float mw = maxS[j][lane & 3];
        mw = fmaxf(mw, __shfl_xor_sync(0xFFFFFFFFu, mw, 2));
        mw = fmaxf(mw, __shfl_xor_sync(0xFFFFFFFFu, mw, 1));
        prevMax[j] = runMax;
        newMax[j] = fmaxf(runMax, mw);
        runMax = newMax[j];
        float ex[4];
#pragma unroll
        for (int e = 0; e < 4; ++e)
            ex[e] = tritonExp(__fsub_rn(scaled[j][e], newMax[j]));
        local[j] = __fadd_rn(__fadd_rn(ex[0], ex[2]), __fadd_rn(ex[1], ex[3]));
    }
    // Round 2: the 4 quarters' sums, each in blockSum order.
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            local[j] = __fadd_rn(local[j], __shfl_xor_sync(0xFFFFFFFFu, local[j], m));
    if (lane == 0)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            sumS[j][warp] = local[j];
    __syncthreads();
    float runSum = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        float w = sumS[j][lane & 3];
        w = __fadd_rn(w, __shfl_xor_sync(0xFFFFFFFFu, w, 2));
        w = __fadd_rn(w, __shfl_xor_sync(0xFFFFFFFFu, w, 1));
        runSum = __fmaf_rn(tritonExp(__fsub_rn(prevMax[j], newMax[j])), runSum, w);
    }
    if (threadIdx.x == 0)
    {
        keyOut[row * numChunks + chunk] = bestKey;
        indexOut[row * numChunks + chunk] = bestIndex;
        int64_t const stat = (row * numPos + pos) * numChunks + chunk;
        maxOut[stat] = runMax;
        sumOut[stat] = runSum;
    }
    if (arrivals == nullptr)
    {
        pdlTrigger(pdl);
        return;
    }
    // v3 fused pick (arrivals != nullptr): the row's last-arriving chunk CTA runs pickKernel's fromDraw body, so the
    // block needs no pick launch per position. arrivals[row] is zero on entry and is reset by that CTA.
    if (threadIdx.x == 0)
    {
        __threadfence();
        isLast = atomicAdd(arrivals + row, 1) == static_cast<int32_t>(numChunks) - 1;
    }
    __syncthreads();
    if (!isLast)
    {
        pdlTrigger(pdl);
        return;
    }
    __threadfence();
    __shared__ float fsm[4];
    __shared__ int ism[4];
    float key;
    int idx;
    threadChunkArgmax<true>(keyOut + row * numChunks, numChunks, key, idx);
    blockArgmaxFresh(key, idx, fsm, ism);
    int64_t const token = __ldcg(indexOut + row * numChunks + idx);
    if (threadIdx.x == 0)
    {
        tokensOut[row * numPos + pos] = static_cast<int32_t>(token);
        arrivals[row] = 0;
    }
    copyEmbRow(w1, w1Stride, emb, row, rank, token, token >= 0);
    pdlTrigger(pdl);
}

//! The pick: grid numGens, 128 threads. fromDraw: the token = the argmax chunk's index, stored at [row, pos];
//! else the anchor (out-of-vocab -> no bias). Then emb[row, :rank] = w1[token, :rank] (0 when invalid).
template <typename TW, typename TA>
__global__ void __launch_bounds__(kThreads)
    pickKernel(float const* __restrict__ keys, int32_t const* __restrict__ indices, int32_t* __restrict__ tokensOut,
        TA const* __restrict__ anchors, int64_t pos, int64_t numPos, TW const* __restrict__ w1, int64_t w1Stride,
        TW* __restrict__ emb, int64_t rank, int64_t vocab, int64_t numChunks, bool fromDraw, bool pdl)
{
    __shared__ float fsm[4];
    __shared__ int ism[4];
    pdlWait(pdl);
    int64_t const row = blockIdx.x;
    int64_t token;
    bool valid;
    if (fromDraw)
    {
        float key;
        int idx;
        threadChunkArgmax<false>(keys + row * numChunks, numChunks, key, idx);
        blockArgmaxFresh(key, idx, fsm, ism);
        token = indices[row * numChunks + idx];
        if (threadIdx.x == 0)
            tokensOut[row * numPos + pos] = static_cast<int32_t>(token);
        valid = token >= 0;
    }
    else
    {
        token = static_cast<int64_t>(anchors[row]);
        valid = token >= 0 && token < vocab;
        token = valid ? token : 0;
    }
    copyEmbRow(w1, w1Stride, emb, row, rank, token, valid);
    pdlTrigger(pdl);
}

//! One proposal CTA's work: q_k of (row, posIdx) over column chunk `chunk` into probs[slot, posIdx, :vocab]. v3: the
//! row's scaled logits are loaded before the statistics reductions; separate shared arrays (2 barriers).
__device__ __forceinline__ void proposalChunk(int64_t row, int64_t posIdx, int64_t rowPos, int64_t chunk,
    float const* __restrict__ biased, float const* __restrict__ temps, float const* __restrict__ maxIn,
    float const* __restrict__ sumIn, int64_t const* __restrict__ slots, float* __restrict__ probs,
    int64_t probsSlotStride, int64_t probsPosStride, int64_t vocab, int64_t numChunks, bool vecPath, float* maxS,
    float* sumS)
{
    constexpr int kReps = kBlock / (4 * kThreads);
    int const lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float const invT = divFull(1.0f, temps[row]);
    bool const in = threadIdx.x < numChunks;
    float const rm = in ? maxIn[rowPos * numChunks + threadIdx.x] : -INFINITY;
    float const rs = in ? sumIn[rowPos * numChunks + threadIdx.x] : 0.0f;
    int64_t const dst = slots[row] * probsSlotStride + posIdx * probsPosStride;
    float sc[kReps][4];
#pragma unroll
    for (int rep = 0; rep < kReps; ++rep)
    {
        int64_t const c0 = chunk * kBlock + rep * (4 * kThreads) + 4 * threadIdx.x;
        float lg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (vecPath && c0 + 3 < vocab)
            load4(biased + rowPos * vocab + c0, lg);
        else
#pragma unroll
            for (int e4 = 0; e4 < 4; ++e4)
                if (c0 + e4 < vocab)
                    lg[e4] = biased[rowPos * vocab + c0 + e4];
#pragma unroll
        for (int e4 = 0; e4 < 4; ++e4)
            sc[rep][e4] = mulRnFtz(lg[e4], invT);
    }
    // The row max (blockMax order).
    float mx = rm;
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, m));
    if (lane == 0)
        maxS[warp] = mx;
    __syncthreads();
    float rowMax = maxS[lane & 3];
    rowMax = fmaxf(rowMax, __shfl_xor_sync(0xFFFFFFFFu, rowMax, 2));
    rowMax = fmaxf(rowMax, __shfl_xor_sync(0xFFFFFFFFu, rowMax, 1));
    float const e = tritonExp(__fsub_rn(rm, rowMax));
    float const p = __fmul_rn(e, rs);
    // Triton: the first butterfly step contracts the product into an FMA with the partner's rounded product.
    float v = __fmaf_rn(e, rs, __shfl_xor_sync(0xFFFFFFFFu, p, 16));
#pragma unroll
    for (int m = 8; m >= 1; m >>= 1)
        v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, m));
    if (lane == 0)
        sumS[warp] = v;
    __syncthreads();
    float total = sumS[lane & 3];
    total = __fadd_rn(total, __shfl_xor_sync(0xFFFFFFFFu, total, 2));
    total = __fadd_rn(total, __shfl_xor_sync(0xFFFFFFFFu, total, 1));
#pragma unroll
    for (int rep = 0; rep < kReps; ++rep)
    {
        int64_t const c0 = chunk * kBlock + rep * (4 * kThreads) + 4 * threadIdx.x;
        float q[4];
#pragma unroll
        for (int e4 = 0; e4 < 4; ++e4)
            q[e4] = divFull(tritonExp(__fsub_rn(sc[rep][e4], rowMax)), total);
        if (vecPath && c0 + 3 < vocab)
            *reinterpret_cast<float4*>(probs + dst + c0) = make_float4(q[0], q[1], q[2], q[3]);
        else
#pragma unroll
            for (int e4 = 0; e4 < 4; ++e4)
                if (c0 + e4 < vocab)
                    probs[dst + c0 + e4] = q[e4];
    }
}

//! v5 proposal: grid (numGens * numPos | numGens, ceil(numChunks / kCpc)), 128 threads. The scaled logits of the CTA's
//! kCpc chunks are loaded first, then one statistics reduction (proposalChunk's order), then kCpc chunk writes.
//! kVec: 16-byte loads / stores only (the launcher guarantees vocab % 4 == 0 and aligned rows; the tail is masked per
//! 4-lane group, which vocab % 4 == 0 keeps whole).
template <bool kVec, int kCpc>
__global__ void __launch_bounds__(kThreads)
    proposalKernelV5(float const* __restrict__ biased, float const* __restrict__ temps, float const* __restrict__ maxIn,
        float const* __restrict__ sumIn, int64_t const* __restrict__ slots, float* __restrict__ probs,
        int64_t probsSlotStride, int64_t probsPosStride, int vocab, int numChunks, int numPos, int onlyPos, bool pdl)
{
    constexpr int kReps = kBlock / (4 * kThreads);
    __shared__ float maxS[4], sumS[4];
    int const lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int const b = static_cast<int>(blockIdx.x);
    int const row = onlyPos >= 0 ? b : b / numPos;
    int const posIdx = onlyPos >= 0 ? onlyPos : b - row * numPos;
    int64_t const rowPos = static_cast<int64_t>(row) * numPos + posIdx;
    pdlWait(pdl);
    float const invT = divFull(1.0f, temps[row]);
    bool const in = static_cast<int>(threadIdx.x) < numChunks;
    float const rm = in ? maxIn[rowPos * numChunks + threadIdx.x] : -INFINITY;
    float const rs = in ? sumIn[rowPos * numChunks + threadIdx.x] : 0.0f;
    float const* src = biased + rowPos * vocab;
    float sc[kCpc][kReps][4];
#pragma unroll
    for (int c = 0; c < kCpc; ++c)
#pragma unroll
        for (int rep = 0; rep < kReps; ++rep)
        {
            int const c0 = (blockIdx.y * kCpc + c) * kBlock + rep * (4 * kThreads) + 4 * threadIdx.x;
            float lg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (kVec)
            {
                if (c0 < vocab)
                    load4(src + c0, lg);
            }
            else
#pragma unroll
                for (int e4 = 0; e4 < 4; ++e4)
                    if (c0 + e4 < vocab)
                        lg[e4] = src[c0 + e4];
#pragma unroll
            for (int e4 = 0; e4 < 4; ++e4)
                sc[c][rep][e4] = mulRnFtz(lg[e4], invT);
        }
    // numChunks > 128 (outside the served class): each thread first folds its chunks t, t + 128, ... in chunk order.
    bool const many = numChunks > kThreads;
    float mx = rm;
    if (many)
        for (int c = threadIdx.x + kThreads; c < numChunks; c += kThreads)
            mx = fmaxf(mx, maxIn[rowPos * numChunks + c]);
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, m));
    if (lane == 0)
        maxS[warp] = mx;
    __syncthreads();
    float rowMax = maxS[lane & 3];
    rowMax = fmaxf(rowMax, __shfl_xor_sync(0xFFFFFFFFu, rowMax, 2));
    rowMax = fmaxf(rowMax, __shfl_xor_sync(0xFFFFFFFFu, rowMax, 1));
    float v;
    if (!many)
    {
        float const e = tritonExp(__fsub_rn(rm, rowMax));
        float const p = __fmul_rn(e, rs);
        v = __fmaf_rn(e, rs, __shfl_xor_sync(0xFFFFFFFFu, p, 16));
    }
    else
    {
        float t = __fmul_rn(tritonExp(__fsub_rn(rm, rowMax)), rs);
        for (int c = threadIdx.x + kThreads; c < numChunks; c += kThreads)
            t = __fmaf_rn(
                tritonExp(__fsub_rn(maxIn[rowPos * numChunks + c], rowMax)), sumIn[rowPos * numChunks + c], t);
        v = __fadd_rn(t, __shfl_xor_sync(0xFFFFFFFFu, t, 16));
    }
#pragma unroll
    for (int m = 8; m >= 1; m >>= 1)
        v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, m));
    if (lane == 0)
        sumS[warp] = v;
    __syncthreads();
    float total = sumS[lane & 3];
    total = __fadd_rn(total, __shfl_xor_sync(0xFFFFFFFFu, total, 2));
    total = __fadd_rn(total, __shfl_xor_sync(0xFFFFFFFFu, total, 1));
    float* dst = probs + slots[row] * probsSlotStride + static_cast<int64_t>(posIdx) * probsPosStride;
#pragma unroll
    for (int c = 0; c < kCpc; ++c)
#pragma unroll
        for (int rep = 0; rep < kReps; ++rep)
        {
            int const c0 = (blockIdx.y * kCpc + c) * kBlock + rep * (4 * kThreads) + 4 * threadIdx.x;
            float q[4];
#pragma unroll
            for (int e4 = 0; e4 < 4; ++e4)
                q[e4] = divFull(tritonExp(__fsub_rn(sc[c][rep][e4], rowMax)), total);
            if (kVec)
            {
                if (c0 < vocab)
                    *reinterpret_cast<float4*>(dst + c0) = make_float4(q[0], q[1], q[2], q[3]);
            }
            else
#pragma unroll
                for (int e4 = 0; e4 < 4; ++e4)
                    if (c0 + e4 < vocab)
                        dst[c0 + e4] = q[e4];
        }
    pdlTrigger(pdl);
}

template <typename... Exp, typename... Act>
cudaError_t launch(void (*kernel)(Exp...), dim3 grid, cudaStream_t stream, bool pdl, Act&&... args)
{
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(kThreads);
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, kernel, static_cast<Exp>(args)...);
}

bool aligned(void const* p, int64_t bytes)
{
    return reinterpret_cast<uintptr_t>(p) % bytes == 0;
}

template <typename TW, typename TA>
cudaError_t pick(MarkovPickArgs const& a, cudaStream_t stream)
{
    return launch(pickKernel<TW, TA>, dim3(static_cast<unsigned>(a.numGens)), stream, a.pdl, a.keys, a.indices,
        a.tokens, static_cast<TA const*>(a.anchors), a.pos, a.numPos, static_cast<TW const*>(a.w1), a.w1Stride,
        static_cast<TW*>(a.emb), a.rank, a.vocab, a.numChunks, a.fromDraw, a.pdl);
}

template <typename TLogit, typename TBias>
cudaError_t draw(MarkovDrawArgs const& a, bool vec, cudaStream_t stream)
{
    dim3 const grid(static_cast<unsigned>(a.numGens), static_cast<unsigned>(a.numChunks));
    return launch(drawKernel<TLogit, TBias>, grid, stream, a.pdl, static_cast<TLogit const*>(a.logits), a.biased,
        static_cast<TBias const*>(a.bias), a.temps, a.seeds, a.offsets, a.slots, a.keys, a.indices, a.runMax, a.runSum,
        a.pos, a.numPos, a.offsetStep, a.paddingSlot, a.vocab, a.numChunks, vec, a.arrivals, a.tokens,
        static_cast<TBias const*>(a.w1), a.w1Stride, static_cast<TBias*>(a.emb), a.rank, a.pdl,
        static_cast<float*>(nullptr), int64_t{0}, int64_t{0}, false);
}

} // namespace

cudaError_t invokeMarkovPick(MarkovPickArgs const& args, cudaStream_t stream)
{
    if (args.numGens <= 0)
    {
        return cudaSuccess;
    }
    if (args.w1Bf16)
    {
        return args.anchors64 ? pick<__nv_bfloat16, int64_t>(args, stream) : pick<__nv_bfloat16, int32_t>(args, stream);
    }
    return args.anchors64 ? pick<float, int64_t>(args, stream) : pick<float, int32_t>(args, stream);
}

cudaError_t invokeMarkovDraw(MarkovDrawArgs const& args, cudaStream_t stream)
{
    if (args.numGens <= 0)
    {
        return cudaSuccess;
    }
    if (args.numChunks > 65535 || (args.arrivals != nullptr) != (args.tokens != nullptr)
        || (args.arrivals != nullptr) != (args.w1 != nullptr) || (args.arrivals != nullptr) != (args.emb != nullptr))
    {
        return cudaErrorInvalidValue;
    }
    int const logitBytes = args.logitsBf16 ? 2 : 4, biasBytes = args.biasBf16 ? 2 : 4;
    // 4 lanes per thread as one access: rows start on 4-element boundaries and the bases are aligned to the access.
    bool const vec = args.vocab % 4 == 0 && aligned(args.logits, 4 * logitBytes) && aligned(args.bias, 4 * biasBytes)
        && aligned(args.biased, 16);
    if (args.logitsBf16)
    {
        return args.biasBf16 ? draw<__nv_bfloat16, __nv_bfloat16>(args, vec, stream)
                             : draw<__nv_bfloat16, float>(args, vec, stream);
    }
    return args.biasBf16 ? draw<float, __nv_bfloat16>(args, vec, stream) : draw<float, float>(args, vec, stream);
}

cudaError_t invokeMarkovProposal(MarkovProposalArgs const& a, cudaStream_t stream)
{
    if (a.numGens <= 0 || a.numPos <= 0)
    {
        return cudaSuccess;
    }
    if (a.numChunks > 65535 || a.vocab >= (int64_t{1} << 31))
    {
        return cudaErrorInvalidValue;
    }
    bool const vec = a.vocab % 4 == 0 && aligned(a.biased, 16) && aligned(a.probs, 16) && a.probsSlotStride % 4 == 0
        && a.probsPosStride % 4 == 0;
    dim3 const grid(static_cast<unsigned>(a.numGens * a.numPos), static_cast<unsigned>(a.numChunks));
    int const v = static_cast<int>(a.vocab), nc = static_cast<int>(a.numChunks), np = static_cast<int>(a.numPos);
    if (vec)
    {
        return launch(proposalKernelV5<true, 1>, grid, stream, a.pdl, a.biased, a.temps, a.runMax, a.runSum, a.slots,
            a.probs, a.probsSlotStride, a.probsPosStride, v, nc, np, -1, a.pdl);
    }
    return launch(proposalKernelV5<false, 1>, grid, stream, a.pdl, a.biased, a.temps, a.runMax, a.runSum, a.slots,
        a.probs, a.probsSlotStride, a.probsPosStride, v, nc, np, -1, a.pdl);
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
