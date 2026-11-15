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

// Device code of the second sliding-window decode attention kernel (swaDecodeAttention.h, TLLM_FMHA_SWA_DECODE_V2=1).
// Same inputs, mask, sinks and scales as swaDecodeFmhaKernel (swaDecodeAttentionKernel.cuh); it does the same
// attention with fewer instructions per key:
//  - Q and K stay E4M3 in shared memory (cp.async, no register staging) and QK^T runs on E4M3 mma.sync (m16n8k32,
//    FP32 accumulation), so no Q/K conversion is issued;
//  - V is widened to FP16 once while it is staged, and the PV B fragments come from ldmatrix.trans instead of
//    per-byte gathers;
//  - each warp owns 16-key chunks for both products (chunk c goes to warp c % 4): its S accumulators become the FP16
//    A fragments of PV in registers (no P round trip through shared memory), with a warp-local row max;
//  - the four partial (max, sum, O) per row are combined once at the end, with the attention sink.
// One CTA (4 warps) handles one (request, KV head) with up to 16 rows (query token x Q head of the KV group) and up
// to kMaxKeys keys. It has no TRT-LLM includes, so it compiles on its own; the KV accessor is a template parameter
// (KVBlockArray in the library). SWA_DECODE2_HOST_EMULATION replaces the few instructions below with a host
// emulation for CPU tests.

#include <cstdint>

namespace swa_decode2
{

constexpr int kHeadDim = 64;
// Keys staged per CTA: window + qLen - 1, rounded up to the 16-key chunk, must fit (GPT-OSS: 128 + 2 - 1 = 129).
constexpr int kMaxKeys = 144;
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
// Rows per CTA: query tokens x Q heads per KV head (one m16 tile).
constexpr int kRows = 16;
constexpr int kChunkKeys = 16;
constexpr int kMaxChunks = kMaxKeys / kChunkKeys;
constexpr int kChunksPerWarp = (kMaxChunks + kWarps - 1) / kWarps;
// Bytes per staged E4M3 Q/K row: 64 + 16 of padding, so the 8 rows a fragment load touches fall in distinct banks.
constexpr int kRowBytes8 = 80;
// Bytes per staged FP16 V row: 128 + 16 of padding, so the 8 rows an ldmatrix phase reads fall in distinct banks.
constexpr int kRowBytes16 = 144;
// Floats per row of a warp's partial O (64 + 4 of padding).
constexpr int kOStride = 68;
// 16-byte K/V chunks per thread while staging (kMaxKeys keys x 4 chunks over kThreads threads).
constexpr int kStageIters = (kMaxKeys * 4 + kThreads - 1) / kThreads;
constexpr float kLog2e = 1.4426950408889634074f;

static_assert(kMaxKeys % kChunkKeys == 0, "chunk tiling");
static_assert(kRows * kWarps * kOStride * 4 <= kMaxKeys * kRowBytes16, "partial O fits over the staged V");

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

struct Smem
{
    union
    {
        struct
        {
            alignas(16) uint8_t v[kMaxKeys * kRowBytes16]; // FP16 [key][64]
            alignas(16) uint8_t k[kMaxKeys * kRowBytes8];  // E4M3 [key][64]
        } kv;

        alignas(16) float o[kWarps][kRows][kOStride]; // partial O per warp, written after every warp's PV
    } u;

    alignas(16) uint8_t q[kRows * kRowBytes8]; // E4M3 [row][64]
    float m[kWarps][kRows];                    // warp-local row max (log2 domain), -inf if no visible key
    float l[kWarps][kRows];                    // warp-local row sum of the FP16-rounded P
};

#ifndef SWA_DECODE2_HOST_EMULATION

__device__ __forceinline__ uint32_t smemAddr(void const* p)
{
    uint32_t a;
    asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(a) : "l"(p));
    return a;
}

// 16 bytes global -> shared (32-bit shared address), or 16 zero bytes when !valid (nothing is read then).
__device__ __forceinline__ void cpAsync16(uint32_t dst, void const* src, bool valid)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(valid ? 16 : 0)
                 : "memory");
}

__device__ __forceinline__ void cpAsyncWaitAll()
{
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}

// D = A(16x32, row) * B(32x8, col) + D, E4M3 inputs, FP32 accumulators. SM89 and later (the kernel traps before).
__device__ __forceinline__ void mmaE4m3(float (&d)[4], uint32_t const (&a)[4], uint32_t b0, uint32_t b1)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    (void) d;
    (void) a;
    (void) b0;
    (void) b1;
    __trap();
#endif
}

// Two E4M3 values (low byte -> low half) to FP16x2. Exact. SM89 and later (the kernel traps before).
__device__ __forceinline__ uint32_t e4m3x2ToHalf2(uint16_t x)
{
    uint32_t r = 0u;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(x));
#else
    (void) x;
    __trap();
#endif
    return r;
}

// D = A(16x16, row) * B(16x8, col) + D, FP16 inputs, FP32 accumulators.
__device__ __forceinline__ void mmaF16(float (&d)[4], uint32_t const (&a)[4], uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Four transposed 8x8 b16 matrices; lane i gives the (32-bit shared) address of row i % 8 of matrix i / 8.
__device__ __forceinline__ void ldmatrixX4Trans(uint32_t (&r)[4], uint32_t rowAddr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(rowAddr));
}

__device__ __forceinline__ float shflXor(float v, int laneMask)
{
    return __shfl_xor_sync(0xffffffffu, v, laneMask);
}

// (lo, hi) -> FP16x2, round to nearest even; lo in the low half.
__device__ __forceinline__ uint32_t packHalf2(float lo, float hi)
{
    uint32_t r;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

__device__ __forceinline__ uint32_t packBf16x2(float lo, float hi)
{
    uint32_t r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

__device__ __forceinline__ void pdlWait()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void pdlLaunchDependents()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

#endif // SWA_DECODE2_HOST_EMULATION

__device__ __forceinline__ uint32_t lds32(uint8_t const* p)
{
    return *reinterpret_cast<uint32_t const*>(p);
}

template <typename KV>
__global__ void __launch_bounds__(kThreads, 6) swaDecodeFmhaKernelV2(Params<KV> const prm)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 890)
    // No E4M3 mma.sync before SM89; the dispatcher only selects this kernel on SM100-family GPUs.
    (void) prm;
    __trap();
#else
    __shared__ Smem sm;
#ifdef SWA_DECODE2_HOST_EMULATION
    swa2EmuPoisonShared(&sm, sizeof(sm));
#endif

    int const kvh = blockIdx.x;
    int const b = blockIdx.y;
    int const tid = threadIdx.x;
    int const warp = tid >> 5;
    int const lane = tid & 31;
    int const g = lane >> 2; // mma group: rows g and g + 8
    int const t4 = lane & 3; // thread in group
    float const kNegInf = __int_as_float(static_cast<int>(0xff800000u));

    // PDL: the preprocessing kernel wrote Q, the new KV and the scales.
    pdlWait();

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
    qLen = qLen < prm.maxSeqLenQ ? qLen : prm.maxSeqLenQ; // the launch checked maxSeqLenQ x headsPerKv <= kRows
    int const L = prm.seqLensKv[b];
    int const hpk = prm.headsPerKv;
    int const rows = qLen > 0 ? qLen * hpk : 0;
    if (rows == 0)
    {
        return;                // uniform across the CTA
    }
    int const pos0 = L - qLen; // position of the request's first query token
    int const window = prm.window;
    int const first = pos0 - window + 1 > 0 ? pos0 - window + 1 : 0;
    int nKeys = L - first; // = min(L, qLen + window - 1)
    nKeys = nKeys < 0 ? 0 : (nKeys > kMaxKeys ? kMaxKeys : nKeys);
    int const nChunks = (nKeys + kChunkKeys - 1) / kChunkKeys;
    int const nStaged = nChunks * kChunkKeys; // keys [nKeys, nStaged) are staged as zeros

    // ---- Stage. Q and K stay E4M3 (cp.async straight to shared memory; rows and keys past the request's are
    //      zero-filled); V is loaded to registers and widened to FP16 on the way. Every load is issued before any
    //      conversion.
    uint32_t const qBase = smemAddr(sm.q);
    uint32_t const kBase = smemAddr(sm.u.kv.k);
    uint32_t const vBase = smemAddr(sm.u.kv.v);
    if (tid < kRows * 4)
    {
        int const r = tid >> 2;
        int const part = tid & 3;
        bool const ok = r < rows;
        uint8_t const* src = prm.q;
        if (ok)
        {
            int const tok = q0 + r / hpk;
            int const head = kvh * hpk + r % hpk;
            src = prm.q + (static_cast<int64_t>(tok) * prm.numHeadsQ + head) * kHeadDim + part * 16;
        }
        cpAsync16(qBase + r * kRowBytes8 + part * 16, src, ok);
    }
    uint4 vBuf[kStageIters];
#pragma unroll
    for (int i = 0; i < kStageIters; ++i)
    {
        int const c = tid + i * kThreads;
        int const key = c >> 2;
        int const part = c & 3;
        vBuf[i] = make_uint4(0u, 0u, 0u, 0u);
        if (key < nStaged)
        {
            bool const ok = key < nKeys;
            uint8_t const* kSrc = prm.q;
            if (ok)
            {
                int const j = first + key;
                int const off = prm.kv.getKVLocalIdx(j, kvh, kHeadDim, part * 16);
                kSrc = static_cast<uint8_t const*>(prm.kv.getKBlockPtr(b, j)) + off;
                vBuf[i] = *reinterpret_cast<uint4 const*>(static_cast<uint8_t const*>(prm.kv.getVBlockPtr(b, j)) + off);
            }
            cpAsync16(kBase + key * kRowBytes8 + part * 16, kSrc, ok);
        }
    }
#pragma unroll
    for (int i = 0; i < kStageIters; ++i)
    {
        int const c = tid + i * kThreads;
        int const key = c >> 2;
        int const part = c & 3;
        if (key < nStaged)
        {
            uint32_t const w[4] = {vBuf[i].x, vBuf[i].y, vBuf[i].z, vBuf[i].w};
            uint32_t h[8];
#pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                h[2 * e] = e4m3x2ToHalf2(static_cast<uint16_t>(w[e] & 0xFFFFu));
                h[2 * e + 1] = e4m3x2ToHalf2(static_cast<uint16_t>(w[e] >> 16));
            }
            uint4* dst = reinterpret_cast<uint4*>(&sm.u.kv.v[key * kRowBytes16 + part * 32]);
            dst[0] = make_uint4(h[0], h[1], h[2], h[3]);
            dst[1] = make_uint4(h[4], h[5], h[6], h[7]);
        }
    }
    cpAsyncWaitAll();
    __syncthreads();

    // ---- Q fragments (A of m16n8k32, E4M3): rows g and g + 8, dims [32 ks, 32 ks + 32).
    uint32_t qa[2][4];
#pragma unroll
    for (int ks = 0; ks < 2; ++ks)
    {
        uint8_t const* r0 = &sm.q[g * kRowBytes8 + ks * 32 + t4 * 4];
        uint8_t const* r1 = r0 + 8 * kRowBytes8;
        qa[ks][0] = lds32(r0);
        qa[ks][1] = lds32(r1);
        qa[ks][2] = lds32(r0 + 16);
        qa[ks][3] = lds32(r1 + 16);
    }
    // Visible staged keys of the lane's rows g and g + 8 (row r: position p = pos0 + r / hpk sees keys j with
    // p - window < j <= p, i.e. staged keys [p - window + 1 - first, p - first], and only the first nKeys exist).
    int lo[2], hi[2];
#pragma unroll
    for (int h = 0; h < 2; ++h)
    {
        int const r = g + 8 * h;
        int const p = pos0 + r / hpk;
        int const l0 = p - window + 1 - first;
        int const h0 = p - first;
        lo[h] = r < rows ? (l0 > 0 ? l0 : 0) : 1;
        hi[h] = r < rows ? (h0 < nKeys - 1 ? h0 : nKeys - 1) : 0;
    }
    // Keys every row sees (the bounds grow with the row): chunks inside need no per-element mask. CTA-uniform.
    int loAll, hiAll;
    {
        int const pLast = pos0 + (rows - 1) / hpk;
        int const l0 = pLast - window + 1 - first;
        int const h0 = pos0 - first;
        loAll = l0 > 0 ? l0 : 0;
        hiAll = h0 < nKeys - 1 ? h0 : nKeys - 1;
    }
    float const scaleLog2 = *prm.scaleSoftmaxLog2;

    // ---- S = Q K^T on the warp's chunks (chunk c = warp + 4 i: keys [16 c, 16 c + 16), n-tiles jj = 0, 1 of 8 keys),
    //      masked and scaled to the log2 domain, and the warp's row max.
    float s[kChunksPerWarp][2][4];
    float mx[2] = {kNegInf, kNegInf};
#pragma unroll
    for (int i = 0; i < kChunksPerWarp; ++i)
    {
        int const c = warp + kWarps * i;
#pragma unroll
        for (int jj = 0; jj < 2; ++jj)
        {
#pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                s[i][jj][e] = 0.f;
            }
        }
        if (c < nChunks)
        {
#pragma unroll
            for (int jj = 0; jj < 2; ++jj)
            {
                uint8_t const* kr = &sm.u.kv.k[(c * kChunkKeys + jj * 8 + g) * kRowBytes8 + t4 * 4];
#pragma unroll
                for (int ks = 0; ks < 2; ++ks)
                {
                    mmaE4m3(s[i][jj], qa[ks], lds32(kr + ks * 32), lds32(kr + ks * 32 + 16));
                }
            }
            if (c * kChunkKeys >= loAll && c * kChunkKeys + kChunkKeys - 1 <= hiAll)
            {
                // Every key of the chunk is visible to every row (padding rows past `rows` are never stored).
#pragma unroll
                for (int jj = 0; jj < 2; ++jj)
                {
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                    {
                        s[i][jj][e] *= scaleLog2;
                        mx[e >> 1] = fmaxf(mx[e >> 1], s[i][jj][e]);
                    }
                }
            }
            else
            {
#pragma unroll
                for (int jj = 0; jj < 2; ++jj)
                {
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                    {
                        int const key = c * kChunkKeys + jj * 8 + 2 * t4 + (e & 1);
                        int const h = e >> 1;
                        float const x = (key >= lo[h] && key <= hi[h]) ? s[i][jj][e] * scaleLog2 : kNegInf;
                        s[i][jj][e] = x;
                        mx[h] = fmaxf(mx[h], x);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int h = 0; h < 2; ++h)
    {
        mx[h] = fmaxf(mx[h], shflXor(mx[h], 1));
        mx[h] = fmaxf(mx[h], shflXor(mx[h], 2));
    }
    float const mBase[2] = {mx[0] == kNegInf ? 0.f : mx[0], mx[1] == kNegInf ? 0.f : mx[1]};

    // ---- P = exp2(S - m_warp), rounded to FP16 (exp2 of a masked -inf is +0): the S accumulators of a chunk's two
    //      n-tiles are the chunk's PV A fragment (a0/a1: keys 2 t4.., rows g / g + 8; a2/a3: keys 8 + 2 t4..).
    uint32_t pa[kChunksPerWarp][4];
#pragma unroll
    for (int i = 0; i < kChunksPerWarp; ++i)
    {
#pragma unroll
        for (int jj = 0; jj < 2; ++jj)
        {
            pa[i][2 * jj] = packHalf2(exp2f(s[i][jj][0] - mBase[0]), exp2f(s[i][jj][1] - mBase[0]));
            pa[i][2 * jj + 1] = packHalf2(exp2f(s[i][jj][2] - mBase[1]), exp2f(s[i][jj][3] - mBase[1]));
        }
    }

    // ---- O_warp = P V over the warp's chunks: 8 n-tiles of 8 dims. Matrix m of an ldmatrix.trans x4 load holds
    //      keys [16 c + 8 (m & 1), +8) x dims [16 u + 8 (m >> 1), +8): m = 0, 1 are n-tile 2 u's B fragment, m = 2, 3
    //      those of n-tile 2 u + 1. A ninth n-tile against a column of FP16 ones gives the row sums of the rounded P
    //      (FP32, in every lane of the quad): rs[0] for row g, rs[2] for row g + 8.
    float o[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j)
    {
#pragma unroll
        for (int e = 0; e < 4; ++e)
        {
            o[j][e] = 0.f;
        }
    }
    float rs[4] = {0.f, 0.f, 0.f, 0.f};
    uint32_t const kOnes = 0x3C003C00u; // FP16x2 (1, 1)
    uint32_t const ldmRow = vBase + ((lane >> 3 & 1) * 8 + (lane & 7)) * kRowBytes16 + (lane >> 4) * 16;
#pragma unroll
    for (int i = 0; i < kChunksPerWarp; ++i)
    {
        int const c = warp + kWarps * i;
        if (c < nChunks)
        {
#pragma unroll
            for (int u = 0; u < 4; ++u)
            {
                uint32_t vb[4];
                ldmatrixX4Trans(vb, ldmRow + c * kChunkKeys * kRowBytes16 + u * 32);
                mmaF16(o[2 * u], pa[i], vb[0], vb[1]);
                mmaF16(o[2 * u + 1], pa[i], vb[2], vb[3]);
            }
            mmaF16(rs, pa[i], kOnes, kOnes);
        }
    }

    // ---- The warps' partial (max, sum, O) to shared memory, over the staged V once every warp is done with it.
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 8; ++j)
    {
#pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            *reinterpret_cast<float2*>(&sm.u.o[warp][g + 8 * h][j * 8 + 2 * t4])
                = make_float2(o[j][2 * h], o[j][2 * h + 1]);
        }
    }
    if (t4 == 0)
    {
#pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            sm.m[warp][g + 8 * h] = mx[h];
            sm.l[warp][g + 8 * h] = rs[2 * h];
        }
    }
    __syncthreads();

    // ---- Combine: thread -> (row, 8 dims). m = max(warp maxima, sink); O = sum_w 2^(m_w - m) O_w /
    //      (sum_w 2^(m_w - m) l_w + 2^(sink - m)), times the output scale; BF16.
    {
        int const row = tid >> 3;
        int const d0 = (tid & 7) * 8;
        if (row < rows)
        {
            float const sk = prm.sinks != nullptr ? prm.sinks[kvh * hpk + row % hpk] * kLog2e : kNegInf;
            float m = sk;
#pragma unroll
            for (int w = 0; w < kWarps; ++w)
            {
                m = fmaxf(m, sm.m[w][row]);
            }
            float const mb = m == kNegInf ? 0.f : m;
            float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            float den = 0.f;
#pragma unroll
            for (int w = 0; w < kWarps; ++w)
            {
                float const mw = sm.m[w][row];
                float const f = mw == kNegInf ? 0.f : exp2f(mw - mb);
                den += f * sm.l[w][row];
                float4 const a = *reinterpret_cast<float4 const*>(&sm.u.o[w][row][d0]);
                float4 const c = *reinterpret_cast<float4 const*>(&sm.u.o[w][row][d0 + 4]);
                acc[0] += f * a.x;
                acc[1] += f * a.y;
                acc[2] += f * a.z;
                acc[3] += f * a.w;
                acc[4] += f * c.x;
                acc[5] += f * c.y;
                acc[6] += f * c.z;
                acc[7] += f * c.w;
            }
            if (sk != kNegInf)
            {
                den += exp2f(sk - mb);
            }
            float const inv = den > 0.f ? *prm.outputScale / den : 0.f;
            int const tok = q0 + row / hpk;
            int const head = kvh * hpk + row % hpk;
            uint16_t* dst = prm.out + (static_cast<int64_t>(tok) * prm.numHeadsQ + head) * kHeadDim + d0;
            *reinterpret_cast<uint4*>(dst)
                = make_uint4(packBf16x2(acc[0] * inv, acc[1] * inv), packBf16x2(acc[2] * inv, acc[3] * inv),
                    packBf16x2(acc[4] * inv, acc[5] * inv), packBf16x2(acc[6] * inv, acc[7] * inv));
        }
    }
    pdlLaunchDependents();
#endif
}

} // namespace swa_decode2
