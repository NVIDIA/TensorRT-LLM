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
// TriAttention folded score kernels
// ============================================================================
//
// Scores every cached decode token of every scored layer for KV eviction.
// The per-round trigonometry is folded into coefficient tables first
// (foldScoreCoefficientsLaunch), so the hot kernel is a pure fused
// multiply-add stream over paged KV:
//
//   score(t, h) = sum_f K_re(t,f)*c_re + K_im(t,f)*c_im + |K(t,f)|*c_mlr
//
// One thread scores one token across all frequencies; a 128-thread CTA covers
// 128 consecutive tokens of one (request, layer, KV-head) segment. There are
// no shuffles, no shared memory, and no barriers: each thread keeps one fused
// accumulator per query head of its GQA group ("mean" aggregation) or one
// partial sum per offset plane ("max" aggregation, where max over offsets
// does not commute through the frequency sum). Coefficient loads are
// lane-uniform 16-byte reads served by L1 broadcast; K loads are 16-byte
// chunks of 8 frequencies when the pool layout allows it, otherwise a fully
// strided scalar path runs the same math.
//
// The kernel accumulates the frequency reduction in sequential chunks (not a
// block-wide tree), so results are tolerance-equal, not bit-equal, against
// the unit tests' PyTorch oracle (the in-tree reference; the original Triton
// score kernel has been deleted). The valid-width side store IS exact: first
// tile, first head, first segment of each request, thread 0, before any
// early-out.
//
// This file must NOT be compiled with --use_fast_math: the fold kernel's
// cosf/sinf and the scalar-path precision are part of the accuracy contract.
// The approximate square root below is an explicit, scoped opt-in instead.
// ============================================================================

#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/triAttentionScoreKernels/triAttentionScoreKernels.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::tri_attention_score
{

namespace
{

// |K| uses the hardware approximate square root (one MUFU op), gated by the
// unit suite's tolerance comparison against the PyTorch oracle (this matched
// the deleted Triton score kernel's tl.sqrt lowering). Define
// TRTLLM_TRI_ATTENTION_IEEE_SQRT to restore the IEEE sqrtf sequence if a
// future geometry needs the extra bits.
__device__ __forceinline__ float triSqrtApprox(float x)
{
#ifdef TRTLLM_TRI_ATTENTION_IEEE_SQRT
    return sqrtf(x);
#else
    float y;
    asm("sqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
#endif
}

template <typename T>
__device__ __forceinline__ float toFloat(T value);

template <>
__device__ __forceinline__ float toFloat<__nv_bfloat16>(__nv_bfloat16 value)
{
    return __bfloat162float(value);
}

template <>
__device__ __forceinline__ float toFloat<half>(half value)
{
    return __half2float(value);
}

template <>
__device__ __forceinline__ float toFloat<float>(float value)
{
    return value;
}

// Quantized pool elements are converted RAW (no scale applied): the per-layer
// dequantization scale is folded into the coefficient tables at fold time, so
// the hot loop stays a pure convert-and-FMA stream.
template <>
__device__ __forceinline__ float toFloat<__nv_fp8_e4m3>(__nv_fp8_e4m3 value)
{
    return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float toFloat<int8_t>(int8_t value)
{
    return static_cast<float>(value);
}

// Unpack one 16-byte K chunk (8 consecutive frequencies) to fp32.
template <typename T>
__device__ __forceinline__ void unpackChunk8(uint4 v, float* dst);

template <>
__device__ __forceinline__ void unpackChunk8<__nv_bfloat16>(uint4 v, float* dst)
{
    auto const* p = reinterpret_cast<__nv_bfloat162 const*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        float2 f2 = __bfloat1622float2(p[i]);
        dst[2 * i] = f2.x;
        dst[2 * i + 1] = f2.y;
    }
}

template <>
__device__ __forceinline__ void unpackChunk8<half>(uint4 v, float* dst)
{
    auto const* p = reinterpret_cast<__half2 const*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        float2 f2 = __half22float2(p[i]);
        dst[2 * i] = f2.x;
        dst[2 * i + 1] = f2.y;
    }
}

// Predicated 16-byte loads of one 8-frequency chunk of one token row. Row
// layout with unit frequency stride is [re f0..F-1 | im f0..F-1], so the
// imaginary half of chunk c sits sizeof(T) * numFreqs bytes further.
template <typename T>
__device__ __forceinline__ void scoreLoadChunk(char const* row, bool valid, int numFreqs, int c, uint4& re4, uint4& im4)
{
    re4 = make_uint4(0u, 0u, 0u, 0u);
    im4 = make_uint4(0u, 0u, 0u, 0u);
    if (valid)
    {
        int const fByte = c * 8 * static_cast<int>(sizeof(T));
        re4 = __ldg(reinterpret_cast<uint4 const*>(row + fByte));
        im4 = __ldg(reinterpret_cast<uint4 const*>(row + static_cast<int64_t>(sizeof(T)) * numFreqs + fByte));
    }
}

// Accumulate one 8-frequency chunk into this thread's per-head accumulators.
// coff0 = flat index of (request, layer, first head of this block, chunk
// frequency 0) in the c_re/c_im tables; cMlr = the matching chunk pointer
// into the static request-independent [layer, head, freq] c_mlr table,
// pre-offset by the caller. Passing the resolved pointer (instead of a
// second flat offset) keeps this loop's live set at the tuned ~72-register
// baseline: one pointer register replaces the base + request-strided offset
// pair the c_mlr reads consumed before the table went static. All
// coefficient reads are lane-uniform 16-byte loads. |K| is computed once per
// (token, frequency) BEFORE the head loop so the GROUP heads share it from
// registers.
template <typename T, int GROUP, bool USE_MAX>
__device__ __forceinline__ void scoreComputeChunk(FoldedScoreParams const& a, int64_t coff0, float const* cMlr,
    int64_t planeStride, uint4 re4, uint4 im4, float* accMean, float* accMlr, float* accPos)
{
    float kRe[8], kIm[8], kMag[8];
    unpackChunk8<T>(re4, kRe);
    unpackChunk8<T>(im4, kIm);
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        kMag[i] = triSqrtApprox(kRe[i] * kRe[i] + kIm[i] * kIm[i]);
    }
#pragma unroll
    for (int hg = 0; hg < GROUP; ++hg)
    {
        int64_t const coff = coff0 + static_cast<int64_t>(hg) * a.numFreqs;
        float const* cmp = cMlr + static_cast<int64_t>(hg) * a.numFreqs;
        float4 const cm0 = __ldg(reinterpret_cast<float4 const*>(cmp));
        float4 const cm1 = __ldg(reinterpret_cast<float4 const*>(cmp + 4));
        float const cml[8] = {cm0.x, cm0.y, cm0.z, cm0.w, cm1.x, cm1.y, cm1.z, cm1.w};
        if constexpr (USE_MAX)
        {
            // The |K| term is offset independent: keep it in its own
            // accumulator and add it after the offset max at store time.
            float m = accMlr[hg];
#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                m = fmaf(kMag[i], cml[i], m);
            }
            accMlr[hg] = m;
            // Unrolled with a live-plane guard so accPos indexing stays
            // static (register-resident) despite the runtime offset count.
#pragma unroll
            for (int o = 0; o < kMaxScoreOffsets; ++o)
            {
                if (o < a.numOffsets)
                {
                    float const* crp = a.cRe + o * planeStride + coff;
                    float const* cip = a.cIm + o * planeStride + coff;
                    float4 const cr0 = __ldg(reinterpret_cast<float4 const*>(crp));
                    float4 const cr1 = __ldg(reinterpret_cast<float4 const*>(crp + 4));
                    float4 const ci0 = __ldg(reinterpret_cast<float4 const*>(cip));
                    float4 const ci1 = __ldg(reinterpret_cast<float4 const*>(cip + 4));
                    float const cre[8] = {cr0.x, cr0.y, cr0.z, cr0.w, cr1.x, cr1.y, cr1.z, cr1.w};
                    float const cim[8] = {ci0.x, ci0.y, ci0.z, ci0.w, ci1.x, ci1.y, ci1.z, ci1.w};
                    float p = accPos[hg * kMaxScoreOffsets + o];
#pragma unroll
                    for (int i = 0; i < 8; ++i)
                    {
                        p = fmaf(kRe[i], cre[i], fmaf(kIm[i], cim[i], p));
                    }
                    accPos[hg * kMaxScoreOffsets + o] = p;
                }
            }
        }
        else
        {
            float const* crp = a.cRe + coff;
            float const* cip = a.cIm + coff;
            float4 const cr0 = __ldg(reinterpret_cast<float4 const*>(crp));
            float4 const cr1 = __ldg(reinterpret_cast<float4 const*>(crp + 4));
            float4 const ci0 = __ldg(reinterpret_cast<float4 const*>(cip));
            float4 const ci1 = __ldg(reinterpret_cast<float4 const*>(cip + 4));
            float const cre[8] = {cr0.x, cr0.y, cr0.z, cr0.w, cr1.x, cr1.y, cr1.z, cr1.w};
            float const cim[8] = {ci0.x, ci0.y, ci0.z, ci0.w, ci1.x, ci1.y, ci1.z, ci1.w};
            float t = accMean[hg];
#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                // 3 fused multiply-adds per (token, frequency, head): the
                // position and |K| terms share one accumulator chain, valid
                // because the mean path has a single coefficient plane.
                t = fmaf(kRe[i], cre[i], fmaf(kIm[i], cim[i], fmaf(kMag[i], cml[i], t)));
            }
            accMean[hg] = t;
        }
    }
}

// Vectorized token-per-thread score kernel. Grid: x = 128-token tiles of the
// page-aligned decode span, y = (request, layer) segment, z = KV head (or one
// query head when a.zIsQueryHead covers GQA group sizes with no dedicated
// GROUP instantiation). STATIC_CHUNKS == 8 pins the production 64-frequency
// shape at compile time (fully unrolled chunk loop, the tuned register
// budget); STATIC_CHUNKS == 0 loops numFreqs / 8 chunks at runtime.
//
// minBlocksPerMultiprocessor = 7: tighter caps force ptxas into ~48-56
// registers with stack spills in the fully unrolled inner loops; 7 CTAs/SM
// admits the ~72-register spill-free allocation this kernel was tuned at.
template <typename T, int GROUP, int STATIC_CHUNKS, bool USE_MAX>
__global__ void __launch_bounds__(kScoreBlockThreads, 7) triScoreVectorizedKernel(FoldedScoreParams a)
{
    int const seg = blockIdx.y;
    int const reqId = a.segRequestIds[seg];
    int const seqLen = a.requestSeqLens[reqId];
    int const tokenStart = a.requestTokenStarts[reqId];
    // Valid-width side store: evaluated before any early-out, so it fires
    // exactly once per request even when the decode region is empty.
    if (blockIdx.x == 0 && blockIdx.z == 0 && (seg % a.numLayers) == 0 && threadIdx.x == 0)
    {
        a.validWidthOut[reqId] = seqLen - tokenStart;
    }
    // Tiles start on the first page of the decode region: with 32-token pages
    // every warp then covers exactly one page and its lane-identical page
    // lookup collapses to one L1 broadcast.
    int const alignedStart = (tokenStart / a.tokensPerBlock) * a.tokensPerBlock;
    if (alignedStart + static_cast<int>(blockIdx.x) * kScoreBlockThreads >= seqLen)
    {
        return; // CTA-uniform: the whole tile is past this sequence
    }
    int const absT = alignedStart + static_cast<int>(blockIdx.x) * kScoreBlockThreads + static_cast<int>(threadIdx.x);
    bool const valid = absT >= tokenStart && absT < seqLen;

    int kvHead;
    int headBase;
    if (a.zIsQueryHead)
    {
        headBase = static_cast<int>(blockIdx.z);
        kvHead = headBase / (a.numQueryHeads / a.numKvHeads);
    }
    else
    {
        kvHead = static_cast<int>(blockIdx.z);
        headBase = kvHead * GROUP;
    }

    int const layerId = a.segLayerIds[seg];
    int const page = absT / a.tokensPerBlock;
    int const slot = absT - page * a.tokensPerBlock;
    // Threads past the sequence tail must not touch the page table (their
    // page ordinal may exceed the staged row); their K loads are predicated
    // off below, so page 0 is a safe placeholder.
    int encoded = 0;
    if (absT < seqLen)
    {
        encoded = a.blockOffsets[a.segPageOffsets[seg] + page];
    }
    // Page-table entries count K/V role pages; kvFactor converts to the
    // layer pool page holding the K plane.
    auto const physPage = static_cast<int64_t>(encoded / a.kvFactor);
    auto const* layerBase = reinterpret_cast<char const*>(a.layerBaseAddrs[layerId]);
    char const* row = layerBase
        + static_cast<int64_t>(sizeof(T))
            * (physPage * a.stridePage + static_cast<int64_t>(kvHead) * a.strideKvHead
                + static_cast<int64_t>(slot) * a.strideSlot);

    int64_t const coff0 = (static_cast<int64_t>(reqId) * a.numCalibratedLayers + layerId) * a.numQueryHeads * a.numFreqs
        + static_cast<int64_t>(headBase) * a.numFreqs;
    // The MLR coefficient is position independent, so its table is folded once
    // at initialization without a request axis: [layer, head, freq]. Hoist the
    // row pointer here (no request stride) so the fully unrolled chunk loop
    // sees a single pre-offset pointer, not an extra live flat offset.
    float const* const cMlrRow = a.cMlr + (static_cast<int64_t>(layerId) * a.numQueryHeads + headBase) * a.numFreqs;
    int64_t const planeStride = static_cast<int64_t>(a.numRequests) * a.numCalibratedLayers * a.numQueryHeads
        * static_cast<int64_t>(a.numFreqs);

    float accMean[GROUP];
    float accMlr[USE_MAX ? GROUP : 1];
    float accPos[USE_MAX ? GROUP * kMaxScoreOffsets : 1];
#pragma unroll
    for (int hg = 0; hg < GROUP; ++hg)
    {
        accMean[hg] = 0.0f;
    }
    if constexpr (USE_MAX)
    {
#pragma unroll
        for (int hg = 0; hg < GROUP; ++hg)
        {
            accMlr[hg] = 0.0f;
        }
#pragma unroll
        for (int i = 0; i < GROUP * kMaxScoreOffsets; ++i)
        {
            accPos[i] = 0.0f;
        }
    }

    if constexpr (STATIC_CHUNKS > 0)
    {
#pragma unroll
        for (int c = 0; c < STATIC_CHUNKS; ++c)
        {
            uint4 re4, im4;
            scoreLoadChunk<T>(row, valid, a.numFreqs, c, re4, im4);
            scoreComputeChunk<T, GROUP, USE_MAX>(
                a, coff0 + c * 8, cMlrRow + c * 8, planeStride, re4, im4, accMean, accMlr, accPos);
        }
    }
    else
    {
        int const chunkCount = a.numFreqs / 8;
        for (int c = 0; c < chunkCount; ++c)
        {
            uint4 re4, im4;
            scoreLoadChunk<T>(row, valid, a.numFreqs, c, re4, im4);
            scoreComputeChunk<T, GROUP, USE_MAX>(
                a, coff0 + c * 8, cMlrRow + c * 8, planeStride, re4, im4, accMean, accMlr, accPos);
        }
    }

    // Store: per (tile, head) the CTA writes one contiguous fp32 row; the
    // predicate replicates the reference token mask and decode-region clip.
    int const tDec = absT - tokenStart;
    if (tDec >= 0 && absT < seqLen)
    {
#pragma unroll
        for (int hg = 0; hg < GROUP; ++hg)
        {
            float score;
            if constexpr (USE_MAX)
            {
                float best = -INFINITY;
#pragma unroll
                for (int o = 0; o < kMaxScoreOffsets; ++o)
                {
                    if (o < a.numOffsets)
                    {
                        best = fmaxf(best, accPos[hg * kMaxScoreOffsets + o]);
                    }
                }
                score = best + accMlr[hg];
            }
            else
            {
                score = accMean[hg];
            }
            a.out[(static_cast<int64_t>(seg) * a.numQueryHeads + headBase + hg) * a.outputWidth + tDec] = score;
        }
    }
}

// Strided scalar score kernel: same token-per-thread mapping and math as the
// vectorized kernel, but every K element is loaded through the full runtime
// stride set (any frequency count, any element stride, fp32 pools included).
// The GQA head loop runs at runtime, so any group size is covered. |K| is
// recomputed per head from the same loads — bit-identical to hoisting it.
template <typename T, bool USE_MAX>
__global__ void __launch_bounds__(kScoreBlockThreads) triScoreScalarKernel(FoldedScoreParams a)
{
    int const seg = blockIdx.y;
    int const reqId = a.segRequestIds[seg];
    int const seqLen = a.requestSeqLens[reqId];
    int const tokenStart = a.requestTokenStarts[reqId];
    // Valid-width side store: identical contract to the vectorized kernel.
    if (blockIdx.x == 0 && blockIdx.z == 0 && (seg % a.numLayers) == 0 && threadIdx.x == 0)
    {
        a.validWidthOut[reqId] = seqLen - tokenStart;
    }
    int const alignedStart = (tokenStart / a.tokensPerBlock) * a.tokensPerBlock;
    if (alignedStart + static_cast<int>(blockIdx.x) * kScoreBlockThreads >= seqLen)
    {
        return;
    }
    int const absT = alignedStart + static_cast<int>(blockIdx.x) * kScoreBlockThreads + static_cast<int>(threadIdx.x);
    bool const valid = absT >= tokenStart && absT < seqLen;

    int const kvHead = blockIdx.z;
    int const groupSize = a.numQueryHeads / a.numKvHeads;
    int const layerId = a.segLayerIds[seg];
    int const page = absT / a.tokensPerBlock;
    int const slot = absT - page * a.tokensPerBlock;
    int encoded = 0;
    if (absT < seqLen)
    {
        encoded = a.blockOffsets[a.segPageOffsets[seg] + page];
    }
    auto const physPage = static_cast<int64_t>(encoded / a.kvFactor);
    auto const* row = reinterpret_cast<T const*>(a.layerBaseAddrs[layerId]) + physPage * a.stridePage
        + static_cast<int64_t>(kvHead) * a.strideKvHead + static_cast<int64_t>(slot) * a.strideSlot;

    int64_t const planeStride = static_cast<int64_t>(a.numRequests) * a.numCalibratedLayers * a.numQueryHeads
        * static_cast<int64_t>(a.numFreqs);
    // The static request-independent [layer, head, freq] c_mlr table: hoist
    // this block's first-head row pointer once (no request stride), mirroring
    // the vectorized kernel; the head loop advances it by numFreqs per head.
    float const* const cMlrRow
        = a.cMlr + (static_cast<int64_t>(layerId) * a.numQueryHeads + kvHead * groupSize) * a.numFreqs;
    int const tDec = absT - tokenStart;
    bool const store = tDec >= 0 && absT < seqLen;

    for (int hg = 0; hg < groupSize; ++hg)
    {
        int const h = kvHead * groupSize + hg;
        int64_t const coff
            = (static_cast<int64_t>(reqId) * a.numCalibratedLayers + layerId) * a.numQueryHeads * a.numFreqs
            + static_cast<int64_t>(h) * a.numFreqs;
        float const* const cml = cMlrRow + static_cast<int64_t>(hg) * a.numFreqs;
        float acc = 0.0f;
        float accMlr = 0.0f;
        float accPos[kMaxScoreOffsets];
#pragma unroll
        for (int o = 0; o < kMaxScoreOffsets; ++o)
        {
            accPos[o] = 0.0f;
        }
        for (int f = 0; f < a.numFreqs; ++f)
        {
            float kRe = 0.0f;
            float kIm = 0.0f;
            if (valid)
            {
                kRe = toFloat<T>(row[static_cast<int64_t>(f) * a.strideDim]);
                kIm = toFloat<T>(row[static_cast<int64_t>(a.numFreqs + f) * a.strideDim]);
            }
            float const kMag = triSqrtApprox(kRe * kRe + kIm * kIm);
            if constexpr (USE_MAX)
            {
                accMlr = fmaf(kMag, cml[f], accMlr);
#pragma unroll
                for (int o = 0; o < kMaxScoreOffsets; ++o)
                {
                    if (o < a.numOffsets)
                    {
                        accPos[o] = fmaf(kRe, a.cRe[o * planeStride + coff + f],
                            fmaf(kIm, a.cIm[o * planeStride + coff + f], accPos[o]));
                    }
                }
            }
            else
            {
                acc = fmaf(kRe, a.cRe[coff + f], fmaf(kIm, a.cIm[coff + f], fmaf(kMag, cml[f], acc)));
            }
        }
        if (store)
        {
            float score;
            if constexpr (USE_MAX)
            {
                float best = -INFINITY;
#pragma unroll
                for (int o = 0; o < kMaxScoreOffsets; ++o)
                {
                    if (o < a.numOffsets)
                    {
                        best = fmaxf(best, accPos[o]);
                    }
                }
                score = best + accMlr;
            }
            else
            {
                score = acc;
            }
            a.out[(static_cast<int64_t>(seg) * a.numQueryHeads + h) * a.outputWidth + tDec] = score;
        }
    }
}

// Per-round coefficient fold: one thread per (request, layer, head, freq)
// element. On the max path each thread additionally writes one c_re/c_im
// value per offset plane (planes are `total` elements apart). The production
// mean path no longer runs this kernel (see the rotation kernel below); the
// c_mlr rows written here are request-identical, and the score kernels read
// only the leading [layer, head, freq] block of that buffer.
__global__ void triFoldScoreCoefficientsKernel(float const* __restrict__ qReal, float const* __restrict__ qImag,
    float const* __restrict__ mlrCoef, float const* __restrict__ freqScaleSq, float const* __restrict__ meanCos,
    float const* __restrict__ meanSin, float const* __restrict__ omega, float const* __restrict__ offsets,
    int32_t const* __restrict__ roundStarts, float const* __restrict__ kvScales, float* __restrict__ cRe,
    float* __restrict__ cIm, float* __restrict__ cMlr, int32_t numCalibratedLayers, int32_t numQueryHeads,
    int32_t numFreqs, int32_t numOffsets, bool useMax, int64_t total)
{
    int64_t const idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
    {
        return;
    }
    auto const f = static_cast<int32_t>(idx % numFreqs);
    int64_t rest = idx / numFreqs;
    auto const h = static_cast<int32_t>(rest % numQueryHeads);
    rest /= numQueryHeads;
    auto const l = static_cast<int32_t>(rest % numCalibratedLayers);
    auto const req = static_cast<int32_t>(rest / numCalibratedLayers);
    // Calibration tables are per (layer, head, freq); the request axis only
    // enters through the phase terms below.
    int64_t const cIdx = (static_cast<int64_t>(l) * numQueryHeads + h) * numFreqs + f;
    float const qre = qReal[cIdx];
    float const qim = qImag[cIdx];
    float s = freqScaleSq[f];
    // Quantized-pool dequantization fold: K_real = scale_l * K_quant, so
    // multiplying scale_l into ALL coefficient tables (c_mlr and every
    // c_re/c_im plane below, since they all carry s) lets the score kernel
    // read raw quantized elements. The |K| term relies on scale > 0
    // (validated host-side). Guarded (not "* 1.0f") so the float-pool path is
    // instruction-identical to before this parameter existed.
    if (kvScales != nullptr)
    {
        s *= kvScales[l];
    }
    cMlr[idx] = mlrCoef[cIdx] * s;
    if (!useMax)
    {
        float const mc = meanCos[static_cast<int64_t>(req) * numFreqs + f];
        float const ms = meanSin[static_cast<int64_t>(req) * numFreqs + f];
        cRe[idx] = s * (qre * mc - qim * ms);
        cIm[idx] = s * (qim * mc + qre * ms);
    }
    else
    {
        float const om = omega[f];
        auto const rs = static_cast<float>(roundStarts[req]);
        for (int32_t o = 0; o < numOffsets; ++o)
        {
            float const phase = (rs + offsets[o]) * om;
            float const cp = cosf(phase);
            float const sp = sinf(phase);
            cRe[o * total + idx] = s * (qre * cp - qim * sp);
            cIm[o * total + idx] = s * (qim * cp + qre * sp);
        }
    }
}

// Per-round mean-path replacement for the fold above: all trigonometry is
// tabulated once at initialization (RoPE-style position tables), so the round
// reduces to one table-row gather per request plus four multiplies and two
// adds per element. With
//     phaseCos[pos, f] = freq_scale_sq[f] * (1/O) * sum_o cos((pos + offset_o) * omega_f)
//     phaseSin[pos, f] = freq_scale_sq[f] * (1/O) * sum_o sin((pos + offset_o) * omega_f)
//     qRealScaled / qImagScaled = kv_scale_l * q   (identity for float pools)
// the rotation
//     c_re = qRealScaled * phaseCos[rs] - qImagScaled * phaseSin[rs]
//     c_im = qImagScaled * phaseCos[rs] + qRealScaled * phaseSin[rs]
// equals the mean fold's freq_scale_sq * kv_scale_l * (q rotated by the
// offset-mean phase at round start rs), because both scale factors distribute
// over the complex product. The position-independent c_mlr fold is fully
// static (folded once at initialization, request axis removed), so this
// kernel never writes it. Grid mirrors the fold kernel: one thread per
// (request, layer, head, freq) element. The host wrapper guarantees every
// round start indexes inside the tables.
// Design: Fanrong Li (torch-graph review, 2026-07-20).
__global__ void triRotateMeanScoreCoefficientsKernel(float const* __restrict__ qRealScaled,
    float const* __restrict__ qImagScaled, float const* __restrict__ phaseCos, float const* __restrict__ phaseSin,
    int32_t const* __restrict__ roundStarts, float* __restrict__ cRe, float* __restrict__ cIm,
    int32_t numCalibratedLayers, int32_t numQueryHeads, int32_t numFreqs, int64_t total)
{
    int64_t const idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
    {
        return;
    }
    auto const f = static_cast<int32_t>(idx % numFreqs);
    int64_t const rest = idx / numFreqs;
    int64_t const calibrationRows = static_cast<int64_t>(numCalibratedLayers) * numQueryHeads;
    auto const req = static_cast<int32_t>(rest / calibrationRows);
    // Calibration tables are per (layer, head, freq); the request-major flat
    // index reduces onto them modulo the calibration extent.
    int64_t const cIdx = (rest % calibrationRows) * numFreqs + f;
    int64_t const phaseIdx = static_cast<int64_t>(roundStarts[req]) * numFreqs + f;
    float const pc = phaseCos[phaseIdx];
    float const ps = phaseSin[phaseIdx];
    float const qre = qRealScaled[cIdx];
    float const qim = qImagScaled[cIdx];
    cRe[idx] = qre * pc - qim * ps;
    cIm[idx] = qim * pc + qre * ps;
}

template <typename T>
void launchVectorized(FoldedScoreParams const& params, int32_t groupSize, dim3 grid, bool useMax, cudaStream_t stream)
{
    bool const staticChunks = params.numFreqs == 64;
    int32_t const effectiveGroup = params.zIsQueryHead ? 1 : groupSize;
#define TRTLLM_TRI_SCORE_LAUNCH_GROUP(GROUP_V)                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        if (staticChunks)                                                                                              \
        {                                                                                                              \
            if (useMax)                                                                                                \
                triScoreVectorizedKernel<T, GROUP_V, 8, true><<<grid, kScoreBlockThreads, 0, stream>>>(params);        \
            else                                                                                                       \
                triScoreVectorizedKernel<T, GROUP_V, 8, false><<<grid, kScoreBlockThreads, 0, stream>>>(params);       \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            if (useMax)                                                                                                \
                triScoreVectorizedKernel<T, GROUP_V, 0, true><<<grid, kScoreBlockThreads, 0, stream>>>(params);        \
            else                                                                                                       \
                triScoreVectorizedKernel<T, GROUP_V, 0, false><<<grid, kScoreBlockThreads, 0, stream>>>(params);       \
        }                                                                                                              \
    } while (0)
    switch (effectiveGroup)
    {
    case 1: TRTLLM_TRI_SCORE_LAUNCH_GROUP(1); break;
    case 2: TRTLLM_TRI_SCORE_LAUNCH_GROUP(2); break;
    case 4: TRTLLM_TRI_SCORE_LAUNCH_GROUP(4); break;
    case 8: TRTLLM_TRI_SCORE_LAUNCH_GROUP(8); break;
    default:
        TLLM_CHECK_WITH_INFO(false,
            "tri_attention_score: vectorized GQA group size must be 1/2/4/8 or use the per-query-head mapping (got "
            "%d)",
            effectiveGroup);
    }
#undef TRTLLM_TRI_SCORE_LAUNCH_GROUP
}

template <typename T>
void launchScalar(FoldedScoreParams const& params, dim3 grid, bool useMax, cudaStream_t stream)
{
    if (useMax)
    {
        triScoreScalarKernel<T, true><<<grid, kScoreBlockThreads, 0, stream>>>(params);
    }
    else
    {
        triScoreScalarKernel<T, false><<<grid, kScoreBlockThreads, 0, stream>>>(params);
    }
}

// Launch flavor for bf16/fp16 pools, the only element types owning both load
// paths (the vectorized 16-byte chunk kernel and the strided scalar kernel).
template <typename T>
void launchVectorizedOrScalar(
    FoldedScoreParams const& params, int32_t groupSize, dim3 grid, bool useVectorized, bool useMax, cudaStream_t stream)
{
    if (useVectorized)
    {
        launchVectorized<T>(params, groupSize, grid, useMax, stream);
    }
    else
    {
        launchScalar<T>(params, grid, useMax, stream);
    }
}

// Quantized pools are functional-only: no vectorized instantiation exists for
// them by design (their dequant scale is folded into the coefficients, so
// only the scalar load path knows how to read them).
template <typename T>
void launchQuantizedScalar(
    FoldedScoreParams const& params, dim3 grid, bool useVectorized, bool useMax, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(!useVectorized, "tri_attention_score: quantized pools must use the scalar path");
    launchScalar<T>(params, grid, useMax, stream);
}

} // namespace

void foldScoreCoefficientsLaunch(float const* qReal, float const* qImag, float const* mlrCoef, float const* freqScaleSq,
    float const* meanCos, float const* meanSin, float const* omega, float const* offsets, int32_t const* roundStarts,
    float const* kvScales, float* cRe, float* cIm, float* cMlr, int32_t numRequests, int32_t numCalibratedLayers,
    int32_t numQueryHeads, int32_t numFreqs, int32_t numOffsets, bool useMax, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(useMax ? (omega != nullptr && offsets != nullptr && roundStarts != nullptr)
                                : (meanCos != nullptr && meanSin != nullptr),
        "tri_attention_score_fold: aggregation-path inputs are missing");
    int64_t const total = static_cast<int64_t>(numRequests) * numCalibratedLayers * numQueryHeads * numFreqs;
    int32_t const threads = 256;
    auto const blocks = static_cast<uint32_t>((total + threads - 1) / threads);
    triFoldScoreCoefficientsKernel<<<blocks, threads, 0, stream>>>(qReal, qImag, mlrCoef, freqScaleSq, meanCos, meanSin,
        omega, offsets, roundStarts, kvScales, cRe, cIm, cMlr, numCalibratedLayers, numQueryHeads, numFreqs, numOffsets,
        useMax, total);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

void rotateMeanScoreCoefficientsLaunch(float const* qRealScaled, float const* qImagScaled, float const* phaseCos,
    float const* phaseSin, int32_t const* roundStarts, float* cRe, float* cIm, int32_t numRequests,
    int32_t numCalibratedLayers, int32_t numQueryHeads, int32_t numFreqs, cudaStream_t stream)
{
    int64_t const total = static_cast<int64_t>(numRequests) * numCalibratedLayers * numQueryHeads * numFreqs;
    int32_t const threads = 256;
    auto const blocks = static_cast<uint32_t>((total + threads - 1) / threads);
    triRotateMeanScoreCoefficientsKernel<<<blocks, threads, 0, stream>>>(qRealScaled, qImagScaled, phaseCos, phaseSin,
        roundStarts, cRe, cIm, numCalibratedLayers, numQueryHeads, numFreqs, total);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

void foldedScoreLaunch(FoldedScoreParams const& params, PoolElementType poolType, int32_t groupSize,
    int32_t numSegments, bool useVectorized, bool useMax, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(numSegments > 0 && numSegments <= 65535,
        "tri_attention_score: request*layer segment count exceeds the CUDA grid limit");
    TLLM_CHECK_WITH_INFO(params.numOffsets >= 1 && params.numOffsets <= kMaxScoreOffsets,
        "tri_attention_score: offset planes exceed the per-thread accumulator budget");
    TLLM_CHECK_WITH_INFO(!useVectorized || (params.numFreqs % 8 == 0 && params.strideDim == 1),
        "tri_attention_score: vectorized path requires 8-frequency chunks with unit stride");
    // Tile count covers the decode span plus the worst-case page-alignment
    // slack (tokenStart may sit up to tokensPerBlock - 1 tokens into a page).
    auto const tiles = static_cast<uint32_t>(
        (params.outputWidth + params.tokensPerBlock - 1 + kScoreBlockThreads - 1) / kScoreBlockThreads);
    uint32_t const headBlocks = params.zIsQueryHead && useVectorized ? params.numQueryHeads : params.numKvHeads;
    dim3 const grid(tiles, static_cast<uint32_t>(numSegments), headBlocks);
    switch (poolType)
    {
    case PoolElementType::kBFloat16:
        launchVectorizedOrScalar<__nv_bfloat16>(params, groupSize, grid, useVectorized, useMax, stream);
        break;
    case PoolElementType::kHalf:
        launchVectorizedOrScalar<half>(params, groupSize, grid, useVectorized, useMax, stream);
        break;
    case PoolElementType::kFloat32:
        // fp32 pools have 32-byte 8-frequency rows; the 16-byte chunk path
        // does not apply, so they always take the strided scalar kernel.
        TLLM_CHECK_WITH_INFO(!useVectorized, "tri_attention_score: fp32 pools must use the scalar path");
        launchScalar<float>(params, grid, useMax, stream);
        break;
    case PoolElementType::kFloat8E4M3:
        launchQuantizedScalar<__nv_fp8_e4m3>(params, grid, useVectorized, useMax, stream);
        break;
    case PoolElementType::kInt8: launchQuantizedScalar<int8_t>(params, grid, useVectorized, useMax, stream); break;
    }
    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels::tri_attention_score

TRTLLM_NAMESPACE_END
