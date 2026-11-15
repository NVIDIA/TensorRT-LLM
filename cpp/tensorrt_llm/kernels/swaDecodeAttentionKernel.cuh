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

#pragma once

// Device code of the dedicated sliding-window decode attention kernel (swaDecodeAttention.h). It has no TRT-LLM
// includes so that it can be compiled on its own; the KV cache accessor is a template parameter (KVBlockArray in the
// library).
//
// One CTA (4 warps) handles one (request, KV head): the rows are the request's query tokens times the Q heads that
// share the KV head (row = token * headsPerKv + head), at most 16 * MT rows. The keys are the union of the rows'
// windows, [L - qLen - window + 1, L - 1], at most kMaxKeys. Q, K and V are E4M3 (as the FMHA reads them), staged in
// shared memory and widened to FP16 (exact) for mma.sync m16n8k16 with FP32 accumulation; the softmax is FP32 in the
// log2 domain with the same device scales and attention-sink semantics as the trtllm-gen FMHA; P is FP16; O is BF16.

#include <cstdint>

namespace swa_decode
{

constexpr int kHeadDim = 64;
// Keys staged per CTA: window + qLen - 1, rounded up to 16 (the PV k-step), must fit.
constexpr int kMaxKeys = 160;
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
// Bytes per staged Q/K/V row: 64 + 16 of padding, so that the 8 rows a fragment load touches fall in distinct banks.
constexpr int kRowBytes = 80;
// FP16 elements per staged P row (padding as above: 84 words per row).
constexpr int kPRow = kMaxKeys + 8;
// Staging loads per thread: K and V, kMaxKeys rows of four 16-byte chunks each.
constexpr int kKvChunkIters = (2 * kMaxKeys * 4 + kThreads - 1) / kThreads;
// QK n-tiles (8 keys) per warp.
constexpr int kNTilesPerWarp = kMaxKeys / 8 / kWarps;
constexpr float kLog2e = 1.4426950408889634074f;

static_assert(kMaxKeys % 16 == 0 && (kMaxKeys / 8) % kWarps == 0, "key tiling");
static_assert(kHeadDim == 16 * kWarps, "PV: each warp owns 16 output dims");
static_assert(2 * kMaxKeys * 4 == kKvChunkIters * kThreads, "staging loop covers K and V exactly");

template <typename KV>
struct Params
{
    uint8_t const* q;              // [numTokens, numHeadsQ, 64] E4M3 (the FMHA's Q input after preprocessing)
    uint16_t* out;                 // [numTokens, numHeadsQ, 64] BF16
    KV kv;                         // paged E4M3 KV cache, HND pages
    int const* seqLensKv;          // [batch], KV length including the current tokens
    int const* cuSeqLensQ;         // [batch + 1] or nullptr (then every request has maxSeqLenQ tokens)
    float const* scaleSoftmaxLog2; // device: qScale * kScale * softmaxScale * log2(e)
    float const* outputScale;      // device: vScale / oScale
    float const* sinks;            // [numHeadsQ] attention sinks (natural-log units) or nullptr
    int numHeadsQ;
    int headsPerKv;
    int maxSeqLenQ;
    int window; // keys j visible to position p: p - window < j <= p
};

template <int MT>
struct Smem
{
    alignas(16) uint8_t k[kMaxKeys * kRowBytes];
    alignas(16) uint8_t v[kMaxKeys * kRowBytes];
    alignas(16) uint8_t q[MT * 16 * kRowBytes];
    alignas(16) uint16_t p[MT * 16 * kPRow];
    float red[2][kWarps][MT * 16];
};

__device__ __forceinline__ uint32_t e4m3x2ToHalf2(uint16_t x)
{
    // Low byte -> low half (cuda_fp8.hpp __nv_cvt_fp8x2_to_halfraw2 uses the same instruction). Exact.
    uint32_t r;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 890)
    // No E4M3 conversion instruction before SM89 (the dispatcher only selects this kernel on SM100-family GPUs; this
    // keeps the library's older targets assembling). Exact: E4M3 values are FP16 values.
    auto const one = [](uint32_t v) -> uint32_t
    {
        uint32_t const e = (v >> 3) & 0xFu;
        uint32_t const m = v & 0x7u;
        float f = e == 0 ? static_cast<float>(m) * 0.001953125f : __uint_as_float(((e + 120u) << 23) | (m << 20));
        f = (e == 0xFu && m == 0x7u) ? __uint_as_float(0x7fc00000u) : f;
        f = (v & 0x80u) ? -f : f;
        uint16_t h;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
        return h;
    };
    r = one(x & 0xFFu) | (one(static_cast<uint32_t>(x) >> 8) << 16);
#else
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(x));
#endif
    return r;
}

__device__ __forceinline__ uint16_t floatToHalfBits(float x)
{
    uint16_t r;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float halfBitsToFloat(uint16_t x)
{
    float r;
    asm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(x));
    return r;
}

__device__ __forceinline__ uint32_t packBf16x2(float lo, float hi)
{
    uint32_t r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

// D = A(16x16, row) * B(16x8, col) + D, FP16 inputs, FP32 accumulators.
__device__ __forceinline__ void mma16816(float (&d)[4], uint32_t const (&a)[4], uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint16_t ld16(uint8_t const* p)
{
    return *reinterpret_cast<uint16_t const*>(p);
}

template <int MT, typename KV>
__global__ void __launch_bounds__(kThreads, 4) swaDecodeFmhaKernel(Params<KV> const prm)
{
    __shared__ Smem<MT> sm;

    int const kvh = blockIdx.x;
    int const b = blockIdx.y;
    int const tid = threadIdx.x;
    int const warp = tid >> 5;
    int const lane = tid & 31;
    int const g = lane >> 2; // mma group: rows g and g + 8
    int const t4 = lane & 3; // thread in group

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // PDL: the preprocessing kernel wrote Q, the new KV and the scales.
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif

    int q0, qLen;
    if (prm.cuSeqLensQ != nullptr)
    {
        q0 = prm.cuSeqLensQ[b];
        qLen = prm.cuSeqLensQ[b + 1] - q0;
    }
    else
    {
        q0 = b * prm.maxSeqLenQ;
        qLen = prm.maxSeqLenQ;
    }
    qLen = qLen < prm.maxSeqLenQ ? qLen : prm.maxSeqLenQ; // the launch sized MT for maxSeqLenQ
    int const L = prm.seqLensKv[b];
    int const hpk = prm.headsPerKv;
    int const rows = qLen > 0 ? qLen * hpk : 0;
    if (rows == 0)
    {
        return; // uniform across the CTA
    }
    int const pos0 = L - qLen; // position of the request's first query token
    int const window = prm.window;
    int const first = pos0 - window + 1 > 0 ? pos0 - window + 1 : 0;
    int nKeys = L - first; // = min(L, qLen + window - 1)
    nKeys = nKeys < 0 ? 0 : (nKeys > kMaxKeys ? kMaxKeys : nKeys);
    int const nK16 = (nKeys + 15) >> 4; // PV k-steps; keys [0, 16 * nK16) are staged (zeros past nKeys)
    int const nStaged = nK16 * 16;

    // ---- Stage Q, K and V (E4M3) in shared memory. Every load is issued before any is used.
    uint4 qv = make_uint4(0u, 0u, 0u, 0u);
    int const qRow = tid >> 2;
    int const qPart = tid & 3;
    if (qRow < MT * 16 && qRow < rows)
    {
        int const tok = q0 + qRow / hpk;
        int const head = kvh * hpk + qRow % hpk;
        qv = *reinterpret_cast<uint4 const*>(
            prm.q + (static_cast<int64_t>(tok) * prm.numHeadsQ + head) * kHeadDim + qPart * 16);
    }
    uint4 kvBuf[kKvChunkIters];
#pragma unroll
    for (int i = 0; i < kKvChunkIters; ++i)
    {
        int const c = tid + i * kThreads;
        int const isV = c >= kMaxKeys * 4;
        int const w = c - isV * kMaxKeys * 4;
        int const key = w >> 2;
        int const part = w & 3;
        kvBuf[i] = make_uint4(0u, 0u, 0u, 0u);
        if (key < nKeys)
        {
            int const j = first + key;
            uint8_t const* blk
                = static_cast<uint8_t const*>(isV ? prm.kv.getVBlockPtr(b, j) : prm.kv.getKBlockPtr(b, j));
            kvBuf[i] = *reinterpret_cast<uint4 const*>(blk + prm.kv.getKVLocalIdx(j, kvh, kHeadDim, part * 16));
        }
    }
    if (qRow < MT * 16)
    {
        *reinterpret_cast<uint4*>(&sm.q[qRow * kRowBytes + qPart * 16]) = qv;
    }
#pragma unroll
    for (int i = 0; i < kKvChunkIters; ++i)
    {
        int const c = tid + i * kThreads;
        int const isV = c >= kMaxKeys * 4;
        int const w = c - isV * kMaxKeys * 4;
        int const key = w >> 2;
        int const part = w & 3;
        if (key < nStaged)
        {
            uint8_t* dst = isV ? sm.v : sm.k;
            *reinterpret_cast<uint4*>(&dst[key * kRowBytes + part * 16]) = kvBuf[i];
        }
    }
    __syncthreads();

    // ---- S = Q K^T. Warp w owns the 8-key n-tiles w, w + 4, ...
    uint32_t qa[MT][4][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
    {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
        {
            uint8_t const* r0 = &sm.q[(mt * 16 + g) * kRowBytes + ks * 16 + 2 * t4];
            uint8_t const* r1 = r0 + 8 * kRowBytes;
            qa[mt][ks][0] = e4m3x2ToHalf2(ld16(r0));
            qa[mt][ks][1] = e4m3x2ToHalf2(ld16(r1));
            qa[mt][ks][2] = e4m3x2ToHalf2(ld16(r0 + 8));
            qa[mt][ks][3] = e4m3x2ToHalf2(ld16(r1 + 8));
        }
    }
    int const nT8 = 2 * nK16;
    float s[MT][kNTilesPerWarp][4];
#pragma unroll
    for (int i = 0; i < kNTilesPerWarp; ++i)
    {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
        {
#pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                s[mt][i][e] = 0.f;
            }
        }
        int const nt = warp + kWarps * i;
        if (nt < nT8)
        {
#pragma unroll
            for (int ks = 0; ks < 4; ++ks)
            {
                uint8_t const* kr = &sm.k[(nt * 8 + g) * kRowBytes + ks * 16 + 2 * t4];
                uint32_t const b0 = e4m3x2ToHalf2(ld16(kr));
                uint32_t const b1 = e4m3x2ToHalf2(ld16(kr + 8));
#pragma unroll
                for (int mt = 0; mt < MT; ++mt)
                {
                    mma16816(s[mt][i], qa[mt][ks], b0, b1);
                }
            }
        }
    }

    // ---- Mask, row max (log2 domain).
    float const scaleLog2 = *prm.scaleSoftmaxLog2;
    float const kNegInf = __int_as_float(static_cast<int>(0xff800000u));
    float mx[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
    {
        mx[mt][0] = kNegInf;
        mx[mt][1] = kNegInf;
#pragma unroll
        for (int i = 0; i < kNTilesPerWarp; ++i)
        {
            int const nt = warp + kWarps * i;
#pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                int const hi = e >> 1;
                int const row = mt * 16 + g + 8 * hi;
                int const key = nt * 8 + 2 * t4 + (e & 1);
                int const j = first + key;
                int const p = pos0 + row / hpk;
                bool const valid = nt < nT8 && row < rows && key < nKeys && j <= p && j > p - window;
                float const x = valid ? s[mt][i][e] * scaleLog2 : kNegInf;
                s[mt][i][e] = x;
                mx[mt][hi] = fmaxf(mx[mt][hi], x);
            }
        }
#pragma unroll
        for (int hi = 0; hi < 2; ++hi)
        {
            mx[mt][hi] = fmaxf(mx[mt][hi], __shfl_xor_sync(0xffffffffu, mx[mt][hi], 1));
            mx[mt][hi] = fmaxf(mx[mt][hi], __shfl_xor_sync(0xffffffffu, mx[mt][hi], 2));
            if (t4 == 0)
            {
                sm.red[0][warp][mt * 16 + g + 8 * hi] = mx[mt][hi];
            }
        }
    }
    __syncthreads();

    // ---- P = exp2(S - m) in FP16 (to shared memory), row sums; the sink joins the max and the denominator.
    float sinkL2[MT][2];
    float sum[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
    {
#pragma unroll
        for (int hi = 0; hi < 2; ++hi)
        {
            int const row = mt * 16 + g + 8 * hi;
            float m = sm.red[0][0][row];
#pragma unroll
            for (int w = 1; w < kWarps; ++w)
            {
                m = fmaxf(m, sm.red[0][w][row]);
            }
            float const sk = (prm.sinks != nullptr && row < rows) ? prm.sinks[kvh * hpk + row % hpk] * kLog2e : kNegInf;
            m = fmaxf(m, sk);
            mx[mt][hi] = (m == kNegInf) ? 0.f : m;
            sinkL2[mt][hi] = sk;
            sum[mt][hi] = 0.f;
        }
#pragma unroll
        for (int i = 0; i < kNTilesPerWarp; ++i)
        {
            int const nt = warp + kWarps * i;
            if (nt < nT8)
            {
#pragma unroll
                for (int hi = 0; hi < 2; ++hi)
                {
                    int const row = mt * 16 + g + 8 * hi;
                    float const x0 = s[mt][i][2 * hi];
                    float const x1 = s[mt][i][2 * hi + 1];
                    uint16_t const h0 = floatToHalfBits(x0 == kNegInf ? 0.f : exp2f(x0 - mx[mt][hi]));
                    uint16_t const h1 = floatToHalfBits(x1 == kNegInf ? 0.f : exp2f(x1 - mx[mt][hi]));
                    sum[mt][hi] += halfBitsToFloat(h0) + halfBitsToFloat(h1);
                    *reinterpret_cast<uint32_t*>(&sm.p[row * kPRow + nt * 8 + 2 * t4])
                        = static_cast<uint32_t>(h0) | (static_cast<uint32_t>(h1) << 16);
                }
            }
        }
#pragma unroll
        for (int hi = 0; hi < 2; ++hi)
        {
            sum[mt][hi] += __shfl_xor_sync(0xffffffffu, sum[mt][hi], 1);
            sum[mt][hi] += __shfl_xor_sync(0xffffffffu, sum[mt][hi], 2);
            if (t4 == 0)
            {
                sm.red[1][warp][mt * 16 + g + 8 * hi] = sum[mt][hi];
            }
        }
    }
    __syncthreads();

    // ---- O = P V. Warp w owns output dims [16 w, 16 w + 16).
    float o[MT][2][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
    {
#pragma unroll
        for (int jn = 0; jn < 2; ++jn)
        {
#pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                o[mt][jn][e] = 0.f;
            }
        }
    }
    for (int ks = 0; ks < nK16; ++ks)
    {
        int const k0 = ks * 16 + 2 * t4;
        uint32_t vb[2][2];
#pragma unroll
        for (int jn = 0; jn < 2; ++jn)
        {
            int const n = warp * 16 + jn * 8 + g;
            uint8_t const* vc = &sm.v[n];
            vb[jn][0] = e4m3x2ToHalf2(
                static_cast<uint16_t>(vc[k0 * kRowBytes] | (static_cast<uint32_t>(vc[(k0 + 1) * kRowBytes]) << 8)));
            vb[jn][1] = e4m3x2ToHalf2(static_cast<uint16_t>(
                vc[(k0 + 8) * kRowBytes] | (static_cast<uint32_t>(vc[(k0 + 9) * kRowBytes]) << 8)));
        }
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
        {
            uint16_t const* p0 = &sm.p[(mt * 16 + g) * kPRow + k0];
            uint16_t const* p1 = p0 + 8 * kPRow;
            uint32_t pa[4];
            pa[0] = *reinterpret_cast<uint32_t const*>(p0);
            pa[1] = *reinterpret_cast<uint32_t const*>(p1);
            pa[2] = *reinterpret_cast<uint32_t const*>(p0 + 8);
            pa[3] = *reinterpret_cast<uint32_t const*>(p1 + 8);
#pragma unroll
            for (int jn = 0; jn < 2; ++jn)
            {
                mma16816(o[mt][jn], pa, vb[jn][0], vb[jn][1]);
            }
        }
    }

    // ---- Normalise and store BF16.
    float const outScale = *prm.outputScale;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
    {
#pragma unroll
        for (int hi = 0; hi < 2; ++hi)
        {
            int const row = mt * 16 + g + 8 * hi;
            if (row >= rows)
            {
                continue;
            }
            float den = 0.f;
#pragma unroll
            for (int w = 0; w < kWarps; ++w)
            {
                den += sm.red[1][w][row];
            }
            if (sinkL2[mt][hi] != kNegInf)
            {
                den += exp2f(sinkL2[mt][hi] - mx[mt][hi]);
            }
            float const inv = den > 0.f ? outScale / den : 0.f;
            int const tok = q0 + row / hpk;
            int const head = kvh * hpk + row % hpk;
            uint16_t* dst = prm.out + (static_cast<int64_t>(tok) * prm.numHeadsQ + head) * kHeadDim;
#pragma unroll
            for (int jn = 0; jn < 2; ++jn)
            {
                int const d = warp * 16 + jn * 8 + 2 * t4;
                *reinterpret_cast<uint32_t*>(dst + d)
                    = packBf16x2(o[mt][jn][2 * hi] * inv, o[mt][jn][2 * hi + 1] * inv);
            }
        }
    }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

} // namespace swa_decode
