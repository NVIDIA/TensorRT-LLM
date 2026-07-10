/*
 * Copyright (c) 2020-2026, NVIDIA CORPORATION.  All rights reserved.
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

// =============================================================================
// Fused T5 attention kernel — encoder self-attention with additive relative
// position bias, fused as a single CUDA kernel.
//
// See fusedT5AttentionKernels.h for capability/interface documentation.
//
// This file contains two device paths:
//   1. WMMA fast path       — SM80+ (tensor cores), templated on head_size.
//   2. SIMT reference path  — SM70/75 fallback, single kernel, any dtype.
//
// The runner picks the path from the current SM version.
// =============================================================================

#include "tensorrt_llm/kernels/fusedT5AttentionKernels.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_fp16.h>
#include <mma.h>

#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

// ---------------------------------------------------------------------------
// Compile-time tile / warp constants shared by the WMMA fast path.
// ---------------------------------------------------------------------------
constexpr int kWmmaM       = 16;
constexpr int kWmmaN       = 16;
constexpr int kWmmaK       = 16;
constexpr int kQTileRows   = 32;   // Bq
constexpr int kKvTileRows  = 64;   // Bk
constexpr int kBlockSize   = 128;  // 4 warps
constexpr int kWarpSize    = 32;
constexpr int kWarpsPerBlk = kBlockSize / kWarpSize;
constexpr float kFloatMaskV = -FLT_MAX;

// ---------------------------------------------------------------------------
// Type conversion helpers.
// ---------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ float toFloatVal(T val);

template <>
__device__ __forceinline__ float toFloatVal<half>(half val)
{
    return __half2float(val);
}

template <typename T>
__device__ __forceinline__ T fromFloatVal(float val);

template <>
__device__ __forceinline__ half fromFloatVal<half>(float val)
{
    return __float2half(val);
}

#ifdef ENABLE_BF16
template <>
__device__ __forceinline__ float toFloatVal<__nv_bfloat16>(__nv_bfloat16 val)
{
    return __bfloat162float(val);
}

template <>
__device__ __forceinline__ __nv_bfloat16 fromFloatVal<__nv_bfloat16>(float val)
{
    return __float2bfloat16(val);
}
#endif

// ---------------------------------------------------------------------------
// Explicit-bias extraction kernel.
//
// For each (head, bucket) pair, find any (qi, ki) whose (ki - qi) hashes to
// that bucket and copy the corresponding element out of the dense
// [1, H, S, S] table. This uses the bucket table directly to look up a
// representative offset, so it is agnostic to the T5 bucketing formula
// details.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void extractExplicitBiasKernel(T const* __restrict__ explicitTable, T* __restrict__ bucketBiasOut,
    int16_t const* __restrict__ bucketTable, int const numHeads, int const maxSeqLen, int const numBuckets)
{
    int const idx = blockIdx.x * blockDim.x + threadIdx.x;
    int const total = numHeads * numBuckets;
    if (idx >= total)
    {
        return;
    }

    int const headId   = idx / numBuckets;
    int const bucketId = idx % numBuckets;

    // Scan the bucket table once to find any (qi, ki) whose relative position
    // maps to `bucketId`. Deterministic (first match wins).
    int const tableLen = 2 * maxSeqLen - 1;
    int delta = 0;
    bool found = false;
    for (int i = 0; i < tableLen; ++i)
    {
        if (static_cast<int>(bucketTable[i]) == bucketId)
        {
            delta = i - (maxSeqLen - 1);
            found = true;
            break;
        }
    }

    if (!found)
    {
        bucketBiasOut[idx] = fromFloatVal<T>(0.f);
        return;
    }

    int qi, ki;
    if (delta >= 0)
    {
        qi = 0;
        ki = delta;
    }
    else
    {
        qi = -delta;
        ki = 0;
    }

    if (qi < maxSeqLen && ki < maxSeqLen)
    {
        int const tableIdx = headId * maxSeqLen * maxSeqLen + qi * maxSeqLen + ki;
        bucketBiasOut[idx] = explicitTable[tableIdx];
    }
    else
    {
        bucketBiasOut[idx] = fromFloatVal<T>(0.f);
    }
}

// ---------------------------------------------------------------------------
// WMMA fast-path kernel, parameterized on head_size at compile time.
//
// Grid:  (num_q_tiles, batch_size, num_heads)
// Block: kBlockSize = 128 threads.
//
// The layout matches the previous fused_t5_attention_v4 kernel; the only
// structural change is that HEAD_SIZE is now a template parameter and the
// T1 bucket table is passed as a global pointer (rather than __constant__).
// ---------------------------------------------------------------------------
using namespace nvcuda;

template <typename T, int HeadSize>
__global__ void __launch_bounds__(kBlockSize, 4) fusedT5AttentionWmmaKernel(T const* __restrict__ qkv,
    T const* __restrict__ bucketBiasAll, int16_t const* __restrict__ bucketTable, T* __restrict__ out,
    int const* __restrict__ inputLengths, int const* __restrict__ cuSeqlens, int const numHeads, int const maxSeqLen,
    int const numBuckets, float const qkScale, bool const removePadding)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    static_assert(HeadSize % kWmmaK == 0, "HeadSize must be a multiple of WMMA_K (=16)");
    static_assert(kQTileRows % kWmmaM == 0, "Bq must be a multiple of WMMA_M (=16)");
    static_assert(kKvTileRows % kWmmaN == 0, "Bk must be a multiple of WMMA_N (=16)");

    constexpr int kQTilesPerBlock = kQTileRows / kWmmaM; // 2

    int const qTileIdx = blockIdx.x;
    int const batchIdx = blockIdx.y;
    int const headIdx  = blockIdx.z;
    int const tid      = threadIdx.x;
    int const warpId   = tid / kWarpSize;
    int const laneId   = tid % kWarpSize;
    int const qStart   = qTileIdx * kQTileRows;

    int const actualLen  = inputLengths[batchIdx];
    int const tokenStart = removePadding ? cuSeqlens[batchIdx] : (batchIdx * maxSeqLen);

    if (qStart >= actualLen)
    {
        return;
    }

    int const numKTiles = (actualLen + kKvTileRows - 1) / kKvTileRows;

    int const qkvStride = 3 * numHeads * HeadSize;
    int const qHeadOff  = headIdx * HeadSize;
    int const kHeadOff  = numHeads * HeadSize + headIdx * HeadSize;
    int const vHeadOff  = 2 * numHeads * HeadSize + headIdx * HeadSize;
    int const outStride = numHeads * HeadSize;

    // Shared-memory layout. `numBuckets` is dynamic → sBias is sized at launch.
    extern __shared__ char smem[];
    char* ptr = smem;

    T* sQ = reinterpret_cast<T*>(ptr);
    ptr += kQTileRows * HeadSize * sizeof(T);
    T* sKV = reinterpret_cast<T*>(ptr);
    ptr += kKvTileRows * HeadSize * sizeof(T);
    float* sS = reinterpret_cast<float*>(ptr);
    ptr += kQTileRows * kKvTileRows * sizeof(float);
    float* sO = reinterpret_cast<float*>(ptr);
    ptr += kQTileRows * HeadSize * sizeof(float);
    T* sBias = reinterpret_cast<T*>(ptr);
    ptr += numBuckets * sizeof(T);
    ptr = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(ptr) + 15) & ~static_cast<uintptr_t>(15));
    float* sRowMax = reinterpret_cast<float*>(ptr);
    ptr += kQTileRows * sizeof(float);
    float* sRowSum = reinterpret_cast<float*>(ptr);

    using Vec4 = uint4;
    constexpr int kElemsPerVec = sizeof(Vec4) / sizeof(T); // 8 for fp16/bf16

    // ---- Load Q tile ----
    int const qVecCount = (kQTileRows * HeadSize) / kElemsPerVec;
    for (int i = tid; i < qVecCount; i += kBlockSize)
    {
        int const elemBase = i * kElemsPerVec;
        int const r        = elemBase / HeadSize;
        int const c        = elemBase % HeadSize;
        int const seqPos   = qStart + r;
        if (seqPos < actualLen)
        {
            int const tok = tokenStart + seqPos;
            reinterpret_cast<Vec4*>(sQ)[i]
                = reinterpret_cast<Vec4 const*>(qkv + tok * qkvStride + qHeadOff + c)[0];
        }
        else
        {
            reinterpret_cast<Vec4*>(sQ)[i] = make_uint4(0, 0, 0, 0);
        }
    }

    // ---- Load per-head bucket bias (T2) into shared memory ----
    for (int i = tid; i < numBuckets; i += kBlockSize)
    {
        sBias[i] = bucketBiasAll[headIdx * numBuckets + i];
    }

    // ---- Zero output accumulator ----
    int const oF4Count = (kQTileRows * HeadSize) / 4;
    for (int i = tid; i < oF4Count; i += kBlockSize)
    {
        reinterpret_cast<float4*>(sO)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    }

    if (tid < kQTileRows)
    {
        sRowMax[tid] = kFloatMaskV;
        sRowSum[tid] = 0.f;
    }

    __syncthreads();

    // ---- Preload Q WMMA fragments (register cache, reused across all K tiles) ----
    int const myQTile = warpId / (kQTilesPerBlock);   // warp -> Q-tile-half
    int const myQRow  = myQTile * kWmmaM;

    wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, T, wmma::row_major> qFrag[HeadSize / kWmmaK];
    #pragma unroll
    for (int kk = 0; kk < HeadSize; kk += kWmmaK)
    {
        wmma::load_matrix_sync(qFrag[kk / kWmmaK], sQ + myQRow * HeadSize + kk, HeadSize);
    }

    int const seqLenTableOffset = maxSeqLen - 1;
    int const kvVecCount = (kKvTileRows * HeadSize) / kElemsPerVec;

    for (int kt = 0; kt < numKTiles; ++kt)
    {
        int const kStart = kt * kKvTileRows;

        // ---- Load K tile ----
        for (int i = tid; i < kvVecCount; i += kBlockSize)
        {
            int const elemBase = i * kElemsPerVec;
            int const r        = elemBase / HeadSize;
            int const c        = elemBase % HeadSize;
            int const seqPos   = kStart + r;
            if (seqPos < actualLen)
            {
                int const tok = tokenStart + seqPos;
                reinterpret_cast<Vec4*>(sKV)[i]
                    = reinterpret_cast<Vec4 const*>(qkv + tok * qkvStride + kHeadOff + c)[0];
            }
            else
            {
                reinterpret_cast<Vec4*>(sKV)[i] = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        // ---- Phase 1: QK GEMM ----
        {
            int const myKHalf = warpId % kQTilesPerBlock;
            int const nStart  = myKHalf * (kKvTileRows / kQTilesPerBlock);
            int const nEnd    = nStart + (kKvTileRows / kQTilesPerBlock);

            for (int n = nStart; n < nEnd; n += kWmmaN)
            {
                wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> acc;
                wmma::fill_fragment(acc, 0.f);

                #pragma unroll
                for (int kk = 0; kk < HeadSize; kk += kWmmaK)
                {
                    wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, T, wmma::col_major> kf;
                    wmma::load_matrix_sync(kf, sKV + n * HeadSize + kk, HeadSize);
                    wmma::mma_sync(acc, qFrag[kk / kWmmaK], kf, acc);
                }

                wmma::store_matrix_sync(sS + myQRow * kKvTileRows + n, acc, kKvTileRows, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // ---- Phase 2: scale + T5 bias + online softmax ----
        {
            int const row     = tid >> 2;           // 4 threads per row × 32 rows
            int const quarter = tid & 3;
            int const col0    = quarter * 16;
            int const qi      = qStart + row;

            float lmax = kFloatMaskV;
            float vals[16];

            #pragma unroll
            for (int c = 0; c < 16; ++c)
            {
                int const ki = kStart + col0 + c;
                float s = sS[row * kKvTileRows + col0 + c] * qkScale;

                if (qi < actualLen && ki < actualLen)
                {
                    int const rel   = ki - qi;
                    int const bkt   = static_cast<int>(bucketTable[rel + seqLenTableOffset]);
                    s += toFloatVal(sBias[bkt]);
                }
                else
                {
                    s = kFloatMaskV;
                }
                vals[c] = s;
                lmax = fmaxf(lmax, s);
            }

            lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffff, lmax, 1));
            lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffff, lmax, 2));
            float const tileMax = lmax;

            float const oldMax = sRowMax[row];
            float const newMax = fmaxf(oldMax, tileMax);
            // If oldMax is still the sentinel (no valid data yet), the
            // correction factor is 0 → this preserves the invariant that
            // sO starts at zero.
            float const corr = (oldMax == kFloatMaskV) ? 0.f : expf(oldMax - newMax);

            float lsum = 0.f;
            #pragma unroll
            for (int c = 0; c < 16; ++c)
            {
                vals[c] = (vals[c] == kFloatMaskV) ? 0.f : expf(vals[c] - newMax);
                lsum += vals[c];
            }
            lsum += __shfl_xor_sync(0xffffffff, lsum, 1);
            lsum += __shfl_xor_sync(0xffffffff, lsum, 2);

            // Reuse the trailing region of sS (which is float-sized and larger
            // than sP=T) as a per-row scratch for `corr`. sP occupies
            // kQTileRows * kKvTileRows * sizeof(T) bytes at the start of sS;
            // sRowMax/sRowSum are separate allocations at the end of smem.
            // We use kQTileRows floats immediately after sP.
            float* sCorr = reinterpret_cast<float*>(
                reinterpret_cast<char*>(sS) + kQTileRows * kKvTileRows * sizeof(T));

            float const newSum = sRowSum[row] * corr + lsum;
            if (quarter == 0)
            {
                sRowMax[row] = newMax;
                sRowSum[row] = newSum;
                sCorr[row]   = corr;
            }

            T* sP = reinterpret_cast<T*>(sS);
            #pragma unroll
            for (int c = 0; c < 16; ++c)
            {
                sP[row * kKvTileRows + col0 + c] = fromFloatVal<T>(vals[c]);
            }
        }
        __syncthreads();

        // Cooperative rescale of sO by per-row corr. Full [kQTileRows, HeadSize]
        // sweep, works for HeadSize in {32, 64, 128}.
        {
            float* sCorr = reinterpret_cast<float*>(
                reinterpret_cast<char*>(sS) + kQTileRows * kKvTileRows * sizeof(T));
            int const oCount = kQTileRows * HeadSize;
            for (int i = tid; i < oCount; i += kBlockSize)
            {
                int const r = i / HeadSize;
                sO[i] *= sCorr[r];
            }
        }
        __syncthreads();

        // ---- Load V tile (overwrites K in sKV) ----
        for (int i = tid; i < kvVecCount; i += kBlockSize)
        {
            int const elemBase = i * kElemsPerVec;
            int const r        = elemBase / HeadSize;
            int const c        = elemBase % HeadSize;
            int const seqPos   = kStart + r;
            if (seqPos < actualLen)
            {
                int const tok = tokenStart + seqPos;
                reinterpret_cast<Vec4*>(sKV)[i]
                    = reinterpret_cast<Vec4 const*>(qkv + tok * qkvStride + vHeadOff + c)[0];
            }
            else
            {
                reinterpret_cast<Vec4*>(sKV)[i] = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        // ---- Phase 3: SV GEMM ----
        {
            int const myVHalf = warpId % kQTilesPerBlock;
            T* sP             = reinterpret_cast<T*>(sS);

            float* warpTmp = reinterpret_cast<float*>(reinterpret_cast<char*>(sS) + kQTileRows * kKvTileRows * sizeof(T))
                + warpId * kWmmaM * kWmmaN;

            int const dStart = myVHalf * (HeadSize / kQTilesPerBlock);
            int const dEnd   = dStart + (HeadSize / kQTilesPerBlock);

            for (int d = dStart; d < dEnd; d += kWmmaN)
            {
                wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> sv;
                wmma::fill_fragment(sv, 0.f);

                #pragma unroll
                for (int kk = 0; kk < kKvTileRows; kk += kWmmaK)
                {
                    wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, T, wmma::row_major> pf;
                    wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, T, wmma::row_major> vf;
                    wmma::load_matrix_sync(pf, sP + myQRow * kKvTileRows + kk, kKvTileRows);
                    wmma::load_matrix_sync(vf, sKV + kk * HeadSize + d, HeadSize);
                    wmma::mma_sync(sv, pf, vf, sv);
                }

                wmma::store_matrix_sync(warpTmp, sv, kWmmaN, wmma::mem_row_major);

                for (int i = laneId; i < kWmmaM * kWmmaN; i += kWarpSize)
                {
                    int const r = i / kWmmaN;
                    int const c = i % kWmmaN;
                    sO[(myQRow + r) * HeadSize + d + c] += warpTmp[i];
                }
            }
        }
        __syncthreads();
    }

    // ---- Final: normalize by row_sum + write output ----
    for (int i = tid; i < qVecCount; i += kBlockSize)
    {
        int const elemBase = i * kElemsPerVec;
        int const r        = elemBase / HeadSize;
        int const c        = elemBase % HeadSize;
        int const seqPos   = qStart + r;
        if (seqPos >= actualLen)
        {
            continue;
        }
        float const rowSum = sRowSum[r];
        // Guard against a fully-masked row (all -inf). This is only possible
        // if actual_len == 0 for this batch, which we handle by writing zeros.
        float const invSum = (rowSum > 0.f) ? __fdividef(1.f, rowSum) : 0.f;
        int const tok = tokenStart + seqPos;

        float4 f0 = reinterpret_cast<float4*>(sO + elemBase)[0];
        float4 f1 = reinterpret_cast<float4*>(sO + elemBase + 4)[0];

        T packed[8];
        packed[0] = fromFloatVal<T>(f0.x * invSum);
        packed[1] = fromFloatVal<T>(f0.y * invSum);
        packed[2] = fromFloatVal<T>(f0.z * invSum);
        packed[3] = fromFloatVal<T>(f0.w * invSum);
        packed[4] = fromFloatVal<T>(f1.x * invSum);
        packed[5] = fromFloatVal<T>(f1.y * invSum);
        packed[6] = fromFloatVal<T>(f1.z * invSum);
        packed[7] = fromFloatVal<T>(f1.w * invSum);

        *reinterpret_cast<Vec4*>(out + tok * outStride + qHeadOff + c) = *reinterpret_cast<Vec4*>(packed);
    }
#endif // __CUDA_ARCH__ >= 800
}

// ---------------------------------------------------------------------------
// SIMT reference fallback (SM70 / SM75).
//
// One block per (batch, head, q_row). One thread computes one output row of
// dimension head_size. No tensor cores, no online softmax tricks — this is
// the numerically simplest possible implementation and its sole purpose is
// to preserve correctness on older architectures. Performance is not a
// design goal.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void fusedT5AttentionSimtKernel(T const* __restrict__ qkv, T const* __restrict__ bucketBiasAll,
    int16_t const* __restrict__ bucketTable, T* __restrict__ out, int const* __restrict__ inputLengths,
    int const* __restrict__ cuSeqlens, int const numHeads, int const headSize, int const maxSeqLen,
    int const numBuckets, float const qkScale, bool const removePadding)
{
    int const batchIdx = blockIdx.y;
    int const headIdx  = blockIdx.z;
    int const qi       = blockIdx.x;
    int const tid      = threadIdx.x;

    int const actualLen  = inputLengths[batchIdx];
    int const tokenStart = removePadding ? cuSeqlens[batchIdx] : (batchIdx * maxSeqLen);
    if (qi >= actualLen)
    {
        return;
    }

    int const qkvStride = 3 * numHeads * headSize;
    int const qHeadOff  = headIdx * headSize;
    int const kHeadOff  = numHeads * headSize + headIdx * headSize;
    int const vHeadOff  = 2 * numHeads * headSize + headIdx * headSize;
    int const outStride = numHeads * headSize;

    // Shared-memory scratch: bias table (numBuckets * sizeof(T)) + scores buffer.
    extern __shared__ char smem[];
    T* sBias      = reinterpret_cast<T*>(smem);
    float* scores = reinterpret_cast<float*>(smem + numBuckets * sizeof(T));

    for (int i = tid; i < numBuckets; i += blockDim.x)
    {
        sBias[i] = bucketBiasAll[headIdx * numBuckets + i];
    }
    __syncthreads();

    // Score pass — one thread computes multiple ki entries.
    for (int kj = tid; kj < actualLen; kj += blockDim.x)
    {
        int const qTok = tokenStart + qi;
        int const kTok = tokenStart + kj;
        float dot = 0.f;
        for (int d = 0; d < headSize; ++d)
        {
            float const qv = toFloatVal(qkv[qTok * qkvStride + qHeadOff + d]);
            float const kv = toFloatVal(qkv[kTok * qkvStride + kHeadOff + d]);
            dot += qv * kv;
        }
        dot *= qkScale;
        int const rel = kj - qi;
        int const bkt = static_cast<int>(bucketTable[rel + (maxSeqLen - 1)]);
        dot += toFloatVal(sBias[bkt]);
        scores[kj] = dot;
    }
    __syncthreads();

    // Row-max reduction (thread 0 only — simple, this is a fallback).
    if (tid == 0)
    {
        float m = kFloatMaskV;
        for (int i = 0; i < actualLen; ++i)
        {
            m = fmaxf(m, scores[i]);
        }
        float sum = 0.f;
        for (int i = 0; i < actualLen; ++i)
        {
            scores[i] = expf(scores[i] - m);
            sum += scores[i];
        }
        float const inv = (sum > 0.f) ? 1.f / sum : 0.f;
        for (int i = 0; i < actualLen; ++i)
        {
            scores[i] *= inv;
        }
    }
    __syncthreads();

    // Weighted-V pass — each thread handles a subset of head_size dims.
    for (int d = tid; d < headSize; d += blockDim.x)
    {
        float acc = 0.f;
        for (int kj = 0; kj < actualLen; ++kj)
        {
            int const kTok = tokenStart + kj;
            float const vv = toFloatVal(qkv[kTok * qkvStride + vHeadOff + d]);
            acc += scores[kj] * vv;
        }
        int const qTok = tokenStart + qi;
        out[qTok * outStride + qHeadOff + d] = fromFloatVal<T>(acc);
    }
}

// ---------------------------------------------------------------------------
// SM gate.
// ---------------------------------------------------------------------------
bool headSizeSupported(int headSize)
{
    return headSize == kFusedT5HeadSize32 || headSize == kFusedT5HeadSize64 || headSize == kFusedT5HeadSize128;
}

// Configure the opt-in dynamic shared-memory limit for a given kernel
// specialization. cudaFuncSetAttribute mutates global driver state for the
// function, so calling it concurrently from multiple host threads (which the
// runner allows) is a data race. We serialize the configuration behind a
// per-specialization guard and only touch the driver when the requested
// footprint grows beyond what we have already set.
template <typename KernelT>
void configureMaxDynamicSmem(KernelT kernel, int smemBytes)
{
    static std::mutex sMutex;
    static int sConfiguredBytes = -1;
    std::lock_guard<std::mutex> lock(sMutex);
    if (smemBytes > sConfiguredBytes)
    {
        TLLM_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));
        sConfiguredBytes = smemBytes;
    }
}

// Dispatch to the SM-specific WMMA specialization when available, otherwise
// to the SIMT fallback.
template <typename T>
void launchDispatch(FusedT5AttentionParams const& params, T const* qkv, T const* bucketBias,
    int16_t const* bucketTable, int const* inputLengths, int const* cuSeqlens, T* out, cudaStream_t stream)
{
    int const sm = common::getSMVersion();
    bool const useWmma = sm >= 80 && !params.forceSimt;

    if (useWmma)
    {
        int const numQTiles = (params.maxSeqLen + kQTileRows - 1) / kQTileRows;
        dim3 grid(numQTiles, params.batchSize, params.numHeads);
        dim3 block(kBlockSize);

        // Shared-memory footprint. Round `numBuckets * sizeof(T)` region up
        // to a 16-byte boundary so the following float arrays stay aligned.
        int const biasBytes = ((params.numBuckets * static_cast<int>(sizeof(T))) + 15) & ~15;
        int const smemBytes = kQTileRows * params.headSize * sizeof(T)              // sQ
            + kKvTileRows * params.headSize * sizeof(T)                              // sKV
            + kQTileRows * kKvTileRows * sizeof(float)                               // sS
            + kQTileRows * params.headSize * sizeof(float)                           // sO
            + biasBytes                                                              // sBias (aligned)
            + kQTileRows * sizeof(float)                                             // sRowMax
            + kQTileRows * sizeof(float);                                            // sRowSum

        auto launchOne = [&](auto headSizeConst) {
            constexpr int kHeadSize = decltype(headSizeConst)::value;
            auto kernel = fusedT5AttentionWmmaKernel<T, kHeadSize>;
            configureMaxDynamicSmem(kernel, smemBytes);
            kernel<<<grid, block, smemBytes, stream>>>(qkv, bucketBias, bucketTable, out, inputLengths, cuSeqlens,
                params.numHeads, params.maxSeqLen, params.numBuckets, params.qkScale, params.removePadding);
        };

        switch (params.headSize)
        {
        case kFusedT5HeadSize32:
            launchOne(std::integral_constant<int, kFusedT5HeadSize32>{});
            break;
        case kFusedT5HeadSize64:
            launchOne(std::integral_constant<int, kFusedT5HeadSize64>{});
            break;
        case kFusedT5HeadSize128:
            launchOne(std::integral_constant<int, kFusedT5HeadSize128>{});
            break;
        default:
            TLLM_CHECK_WITH_INFO(false, "FusedT5Attention: unsupported head_size=%d", params.headSize);
        }
        sync_check_cuda_error(stream);
        return;
    }

    // SIMT fallback.
    dim3 grid(params.maxSeqLen, params.batchSize, params.numHeads);
    // 128 threads gives us enough parallelism for the ki-loop and the d-loop.
    dim3 block(128);
    int const smemBytes = params.numBuckets * static_cast<int>(sizeof(T)) + params.maxSeqLen * sizeof(float);
    fusedT5AttentionSimtKernel<T><<<grid, block, smemBytes, stream>>>(qkv, bucketBias, bucketTable, out, inputLengths,
        cuSeqlens, params.numHeads, params.headSize, params.maxSeqLen, params.numBuckets, params.qkScale,
        params.removePadding);
    sync_check_cuda_error(stream);
}

} // anonymous namespace

// ===========================================================================
// Public API.
// ===========================================================================

int hostT5RelativeBucket(int relativePosition, int numBuckets, int maxDistance, bool bidirectional)
{
    int relativeBuckets = 0;

    if (bidirectional)
    {
        int const halfBuckets = numBuckets / 2;
        if (relativePosition > 0)
        {
            relativeBuckets = halfBuckets;
        }
        else
        {
            relativePosition = -relativePosition;
        }

        int const maxExact = halfBuckets / 2;
        if (relativePosition < maxExact)
        {
            relativeBuckets += relativePosition;
        }
        else
        {
            float const logRatio = std::log(static_cast<float>(relativePosition) / static_cast<float>(maxExact))
                / std::log(static_cast<float>(maxDistance) / static_cast<float>(maxExact));
            int bucket = maxExact + static_cast<int>(logRatio * static_cast<float>(halfBuckets - maxExact));
            bucket = (bucket < halfBuckets - 1) ? bucket : (halfBuckets - 1);
            relativeBuckets += bucket;
        }
    }
    else
    {
        // Causal / unidirectional T5 bucketing (kept for completeness).
        relativePosition = -std::min(relativePosition, 0);
        int const maxExact = numBuckets / 2;
        if (relativePosition < maxExact)
        {
            relativeBuckets = relativePosition;
        }
        else
        {
            float const logRatio = std::log(static_cast<float>(relativePosition) / static_cast<float>(maxExact))
                / std::log(static_cast<float>(maxDistance) / static_cast<float>(maxExact));
            int bucket = maxExact + static_cast<int>(logRatio * static_cast<float>(numBuckets - maxExact));
            bucket = (bucket < numBuckets - 1) ? bucket : (numBuckets - 1);
            relativeBuckets = bucket;
        }
    }
    return relativeBuckets;
}

void hostBuildT5BucketTable(
    int16_t* table, int maxSeqLen, int numBuckets, int maxDistance, bool bidirectional)
{
    int const tableLen = 2 * maxSeqLen - 1;
    for (int i = 0; i < tableLen; ++i)
    {
        int const relPos = i - (maxSeqLen - 1);
        table[i] = static_cast<int16_t>(hostT5RelativeBucket(relPos, numBuckets, maxDistance, bidirectional));
    }
}

void initFusedT5BucketTable(int16_t* bucketTable, int maxSeqLen, int numBuckets, int maxDistance, bool bidirectional,
    cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(bucketTable != nullptr, "FusedT5Attention: bucketTable must not be null");
    TLLM_CHECK_WITH_INFO(maxSeqLen > 0 && maxSeqLen <= kFusedT5MaxSeqLen,
        "FusedT5Attention: maxSeqLen=%d out of range (1..%d)", maxSeqLen, kFusedT5MaxSeqLen);
    TLLM_CHECK_WITH_INFO(numBuckets > 0 && (numBuckets % 2) == 0 && numBuckets <= kFusedT5MaxNumBuckets,
        "FusedT5Attention: numBuckets=%d must be even and in (0, %d]", numBuckets, kFusedT5MaxNumBuckets);
    TLLM_CHECK_WITH_INFO(maxDistance > 0, "FusedT5Attention: maxDistance must be > 0");

    int const tableLen = 2 * maxSeqLen - 1;
    std::vector<int16_t> hostTable(tableLen);
    hostBuildT5BucketTable(hostTable.data(), maxSeqLen, numBuckets, maxDistance, bidirectional);
    TLLM_CUDA_CHECK(cudaMemcpyAsync(bucketTable, hostTable.data(), tableLen * sizeof(int16_t),
        cudaMemcpyHostToDevice, stream));
    // The user typically follows this call with kernel launches on the same
    // stream, so we do not synchronize here.
}

template <typename T>
void extractExplicitBiasToTable2(T const* explicitTable, T* bucketBiasOut, int16_t const* bucketTable, int numHeads,
    int maxSeqLen, int numBuckets, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(explicitTable != nullptr && bucketBiasOut != nullptr && bucketTable != nullptr,
        "FusedT5Attention: extractExplicitBiasToTable2 got a null pointer");
    TLLM_CHECK_WITH_INFO(numHeads > 0 && numBuckets > 0 && numBuckets <= kFusedT5MaxNumBuckets,
        "FusedT5Attention: bad numHeads/numBuckets");

    int const totalThreads = numHeads * numBuckets;
    int const block = (totalThreads < 256) ? totalThreads : 256;
    int const grid  = (totalThreads + block - 1) / block;
    extractExplicitBiasKernel<T><<<grid, block, 0, stream>>>(
        explicitTable, bucketBiasOut, bucketTable, numHeads, maxSeqLen, numBuckets);
    sync_check_cuda_error(stream);
}

bool FusedT5AttentionRunner::isShapeSupported(FusedT5AttentionParams const& params)
{
    if (!params.isBidirectional)
    {
        // Causal path is intentionally not fused yet — caller should use the
        // generic attention op for T5 decoder self-attention.
        return false;
    }
    if (!headSizeSupported(params.headSize))
    {
        return false;
    }
    if (params.maxSeqLen <= 0 || params.maxSeqLen > kFusedT5MaxSeqLen)
    {
        return false;
    }
    if (params.numBuckets <= 0 || (params.numBuckets % 2) != 0 || params.numBuckets > kFusedT5MaxNumBuckets)
    {
        return false;
    }
    if (params.batchSize <= 0 || params.numHeads <= 0)
    {
        return false;
    }

    // Architectural gate. WMMA path needs SM80+, SIMT fallback needs SM70+.
    int const sm = common::getSMVersion();
    if (sm < 70)
    {
        return false;
    }

    // Shared-memory sanity for the WMMA path (48KB default, 100KB opt-in).
    if (sm >= 80)
    {
        int const biasBytes = ((params.numBuckets * 2) + 15) & ~15; // 2 = sizeof(half)/bf16
        int const smemBytes = kQTileRows * params.headSize * 2 // sQ
            + kKvTileRows * params.headSize * 2                 // sKV
            + kQTileRows * kKvTileRows * 4                      // sS
            + kQTileRows * params.headSize * 4                  // sO
            + biasBytes
            + kQTileRows * 4                                    // sRowMax
            + kQTileRows * 4;                                   // sRowSum
        // 163 KB is the SM89 max; give ourselves a comfortable headroom.
        if (smemBytes > 100 * 1024)
        {
            return false;
        }
    }
    return true;
}

bool FusedT5AttentionRunner::isSupported(FusedT5AttentionParams const& params)
{
    if (!isShapeSupported(params))
    {
        return false;
    }
    if (params.forceEnable)
    {
        return true;
    }
    return common::getEnvEnableFusedT5Attention();
}

template <typename T>
void FusedT5AttentionRunner::run(FusedT5AttentionParams const& params, T const* qkv, T const* bucketBias,
    int16_t const* bucketTable, int const* inputLengths, int const* cuSeqlens, T* out, cudaStream_t stream) const
{
    TLLM_CHECK_WITH_INFO(isSupported(params),
        "FusedT5Attention::run called with unsupported params (head_size=%d, max_seq_len=%d, num_buckets=%d, "
        "bidirectional=%d, force=%d).",
        params.headSize, params.maxSeqLen, params.numBuckets, static_cast<int>(params.isBidirectional),
        static_cast<int>(params.forceEnable));
    TLLM_CHECK_WITH_INFO(qkv != nullptr && bucketBias != nullptr && bucketTable != nullptr && out != nullptr
            && inputLengths != nullptr,
        "FusedT5Attention::run got a null pointer");
    if (params.removePadding)
    {
        TLLM_CHECK_WITH_INFO(cuSeqlens != nullptr, "FusedT5Attention: cuSeqlens required when removePadding=true");
    }

    launchDispatch<T>(params, qkv, bucketBias, bucketTable, inputLengths, cuSeqlens, out, stream);
}

// ---------------------------------------------------------------------------
// Explicit template instantiations.
// ---------------------------------------------------------------------------
template void extractExplicitBiasToTable2<half>(
    half const*, half*, int16_t const*, int, int, int, cudaStream_t);
template void FusedT5AttentionRunner::run<half>(FusedT5AttentionParams const&, half const*, half const*,
    int16_t const*, int const*, int const*, half*, cudaStream_t) const;

#ifdef ENABLE_BF16
template void extractExplicitBiasToTable2<__nv_bfloat16>(
    __nv_bfloat16 const*, __nv_bfloat16*, int16_t const*, int, int, int, cudaStream_t);
template void FusedT5AttentionRunner::run<__nv_bfloat16>(FusedT5AttentionParams const&, __nv_bfloat16 const*,
    __nv_bfloat16 const*, int16_t const*, int const*, int const*, __nv_bfloat16*, cudaStream_t) const;
#endif

} // namespace kernels

TRTLLM_NAMESPACE_END
