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

//
// MiniMax-M3 block-sparse GQA verify decode for GB300 (SM103), packed NVFP4 K/V (torch.ops.trtllm.
// minimax_m3_cute_sparse_decode). CUDA C++ on CuTe atoms: cute::SM80_16x8x16_F32F16F16F32_TN (mma.sync HMMA),
// cute::SM90_BULK_COPY_G2S (1D bulk copy with mbarrier complete_tx), cute::cluster_sync. Define CSD_RAW_PTX to use the
// identical raw PTX instead. The kernel body compiles on the SM10x passes only; other passes get an empty kernel.
//
// Shape (TP2 per rank): KV heads H (2), 16 q heads per KV head, head_dim 128, page = sparse block = 128 tokens,
// top-k <= 32 (16 in M3), FP8-e4m3 q, packed NVFP4 K/V (E2M1 nibbles + E4M3 scale per 16 elements; K scales
// token-major, V scales in 4x4 token-quad order), per-tensor fp32 K/V scales, bf16 output.
//
// Design (2c40 blueprint, adapted to NVFP4):
//   grid (S, H, total_q), cluster (S, 1, 1): one CTA per (top-k split, KV head, query row); 5 warps.
//   warp 4 = producer: lanes fetch the split's top-k ids and pages once, lane 0 streams each selected page's
//            K (8 KB), V (8 KB), K scales (1 KB), V scales (1 KB) with 4 bulk copies into an ST-stage smem ring
//            guarded by full/empty mbarriers.
//   warps 0-3 = consumers: warp w owns keys 32w..32w+31 of every block. QK and PV run as m16n8k16 HMMA with fp32
//            accumulation, M = the 16 q heads of the row. NVFP4 is dequantized in registers straight into the MMA
//            B fragments (K: one LDS.128 per key per thread; V: 2x2 transpose by PRMT, V scale pairs come for free
//            from the token-quad scale order). Each warp keeps its own exact online softmax (linear recurrence, as
//            stock's USE_LINEAR_SOFTMAX) over its key slice: no per-block cross-warp synchronisation.
//   epilogue: the 4 warps' (m, l, O) are merged in smem (fixed order); with S > 1 the S split CTAs merge their
//            normalised fp32 partials over DSMEM with stock's LSE-weight formula; one launch per layer, no global
//            scratch, no merge kernel. PDL: griddepcontrol.wait before the first global read, launch_dependents
//            after the last store.
//   DQ = 0: NVFP4 nibbles -> f16 with cvt.rn.f16x2.e2m1x2 (as stock).  DQ = 1: integer bit placement (value * 2^-14,
//            the 2^14 folded exactly as 2^7 into q and 2^7 into the block scales; products bit-identical to stock's).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#include "tensorrt_llm/kernels/minimaxM3/cuteSparseDecode.h"

#ifndef CSD_RAW_PTX
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/arch/util.hpp>
// The CuTe atoms silently become CUTE_INVALID_CONTROL_PATH when their arch macro is off: fail the build instead.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)                                                                  \
    && (!defined(CUTE_ARCH_TMA_SM90_ENABLED) || !defined(CUTE_ARCH_MMA_SM80_ENABLED)                                   \
        || !defined(CUTE_ARCH_CLUSTER_SM90_ENABLED))
#error                                                                                                                 \
    "cute-sparse-decode: CuTe SM80 MMA / SM90 bulk copy / cluster atoms are not enabled for this arch (build with -DCSD_RAW_PTX)"
#endif
#endif

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace csd
{

constexpr int kPage = 128;
constexpr int kHd = 128;
constexpr int kGrp = 16;
constexpr int kNcw = 4;
constexpr int kThreads = 32 * (kNcw + 1);
constexpr int kKB = kPage * kHd / 2;           // 8192 packed bytes per (page, KV head)
constexpr int kSB = kPage * kHd / 16;          // 1024 scale bytes per (page, KV head)
constexpr int kStage = 2 * kKB + 2 * kSB;      // 18432
constexpr int kORow = 132;                     // padded fp32 row of the merge buffer (bank spread)
constexpr int kOBuf = kNcw * kGrp * kORow * 4; // 33792 bytes, reuses the ring after the main loop
constexpr uint32_t kF16x2_128 = 0x58005800u;   // {128.0, 128.0} as f16x2

using Params = MinimaxM3CuteSparseDecodeParams;

__host__ __device__ constexpr int bar_bytes(int stages)
{
    return 64 * ((2 * stages * 8 + 63) / 64);
}

__host__ __device__ constexpr int smem_bytes(int stages)
{
    return stages * kStage + bar_bytes(stages) + 2 * kNcw * kGrp * 4 + kGrp * 4;
}

static_assert(2 * kStage >= kOBuf, "merge buffer must fit in a 2-stage ring");

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
// ------------------------------------------------------------------------------------------------ device helpers
__device__ __forceinline__ uint32_t smem_u32(void const* p)
{
#ifndef CSD_RAW_PTX
    return cute::cast_smem_ptr_to_uint(p);
#else
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
#endif
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}

__device__ __forceinline__ void fence_mbar_init()
{
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar)
{
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity)
{
    asm volatile(
        "{\n\t.reg .pred P1;\n\t"
        "CSD_WAIT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n\t"
        "@P1 bra CSD_DONE;\n\t"
        "bra CSD_WAIT;\n\t"
        "CSD_DONE:\n\t}" ::"r"(smem_u32(bar)),
        "r"(parity)
        : "memory");
}

__device__ __forceinline__ void bulk_g2s(void* dst, void const* src, uint32_t bytes, uint64_t* bar)
{
#ifndef CSD_RAW_PTX
    cute::SM90_BULK_COPY_G2S::copy(src, bar, dst, static_cast<int32_t>(bytes));
#else
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(smem_u32(dst)),
        "l"(src), "r"(bytes), "r"(smem_u32(bar))
        : "memory");
#endif
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1)
{
#ifndef CSD_RAW_PTX
    cute::SM80_16x8x16_F32F16F16F32_TN::fma(
        d[0], d[1], d[2], d[3], a[0], a[1], a[2], a[3], b0, b1, d[0], d[1], d[2], d[3]);
#else
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void cl_sync()
{
#ifndef CSD_RAW_PTX
    cute::cluster_sync();
#else
    asm volatile("barrier.cluster.arrive.aligned;\n\tbarrier.cluster.wait.aligned;" ::: "memory");
#endif
}

__device__ __forceinline__ uint32_t map_rank(void const* p, uint32_t rank)
{
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(smem_u32(p)), "r"(rank));
    return r;
}

__device__ __forceinline__ float ld_dsm_f(uint32_t a)
{
    float v;
    asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(a) : "memory");
    return v;
}

__device__ __forceinline__ float4 ld_dsm_f4(uint32_t a)
{
    float4 v;
    asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                 : "r"(a)
                 : "memory");
    return v;
}

__device__ __forceinline__ void griddep_wait()
{
    asm volatile("griddepcontrol.wait;" ::: "memory");
}

__device__ __forceinline__ void griddep_launch()
{
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

__device__ __forceinline__ void named_bar_sync(int id, int n)
{
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}

__device__ __forceinline__ float ex2(float x)
{
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float lg2(float x)
{
    float y;
    asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b)
{
    uint32_t d;
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}

__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi)
{
    uint32_t d;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
    return d;
}

// two E4M3 bytes (low 16 bits; low byte -> low half) -> f16x2, exact
__device__ __forceinline__ uint32_t cvt_e4m3x2(uint32_t v)
{
    uint32_t d;
    unsigned short const s = static_cast<unsigned short>(v & 0xffffu);
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(d) : "h"(s));
    return d;
}

// four packed E2M1x2 bytes of w -> four f16x2 (byte b -> o[b]; low nibble -> low half), exact
__device__ __forceinline__ void cvt_e2m1x8(uint32_t w, uint32_t (&o)[4])
{
    asm("{\n\t.reg .b8 b0, b1, b2, b3;\n\t"
        "mov.b32 {b0, b1, b2, b3}, %4;\n\t"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\n\t"
        "cvt.rn.f16x2.e2m1x2 %1, b1;\n\t"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\n\t"
        "cvt.rn.f16x2.e2m1x2 %3, b3;\n\t}"
        : "=r"(o[0]), "=r"(o[1]), "=r"(o[2]), "=r"(o[3])
        : "r"(w));
}

// DQ = 1: nibble i of each 16-bit half of u -> f16 bits (sign at 15, magnitude at 11..9) = E2M1 value * 2^-14, exact
__device__ __forceinline__ uint32_t e2m1_bits(uint32_t u, int i)
{
    const uint32_t x = u >> (4 * i);
    return ((x << 9) & 0x0E000E00u) | ((x << 12) & 0x80008000u);
}

#endif // defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

// ------------------------------------------------------------------------------------------------ kernel
template <int S, int ST, int DQ>
__global__ void __launch_bounds__(kThreads, 2) csd_kernel(const __grid_constant__ Params p)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* ring = smem;
    uint64_t* full = reinterpret_cast<uint64_t*>(smem + ST * kStage);
    uint64_t* empty = full + ST;
    float* sm_m = reinterpret_cast<float*>(smem + ST * kStage + bar_bytes(ST));
    float* sm_l = sm_m + kNcw * kGrp;
    float* sm_lse = sm_l + kNcw * kGrp;
    float* obuf = reinterpret_cast<float*>(smem); // [kNcw][kGrp][kORow] fp32, after the main loop

    int const split = blockIdx.x;                 // == rank in the (S,1,1) cluster
    int const h = blockIdx.y;
    int const t = blockIdx.z;
    int const warp = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;

    if (threadIdx.x == 0)
    {
#pragma unroll
        for (int s = 0; s < ST; ++s)
        {
            mbar_init(&full[s], 1);
            mbar_init(&empty[s], kNcw);
        }
        fence_mbar_init();
    }
    __syncthreads();

    // PDL: no global input is read before the producer grid has completed.
    griddep_wait();

    int const req = t / p.dql;
    int const qoff = t - req * p.dql;
    int const seq_len = __ldg(p.seq_lens + req);
    int const kv_len = max(seq_len - p.dql + qoff + 1, 0);
    int const nblk = (kv_len + kPage - 1) / kPage;
    int const real = min(p.max_topk, nblk);
    int const chunk = (p.max_topk + S - 1) / S;
    int const c0 = split * chunk;
    int const n = max(min(c0 + chunk, real) - c0, 0);

    int my_blk = 0;
    if (lane < n)
    {
        my_blk = __ldg(p.topk + (long long) h * p.st_h + (long long) t * p.st_n + (long long) (c0 + lane) * p.st_k);
    }

    if (warp == kNcw)
    {
        // ---------------------------------------------------------------- producer warp
        long long page = 0;
        if (lane < n)
            page = (long long) __ldg(p.bt + (long long) req * p.sbt_b + my_blk);
        for (int it = 0; it < n; ++it)
        {
            long long const pg = __shfl_sync(0xffffffffu, page, it);
            if (lane == 0)
            {
                int const st = it % ST;
                const uint32_t ph = (it / ST) & 1;
                mbar_wait(&empty[st], ph ^ 1);
                uint8_t* dst = ring + st * kStage;
                mbar_arrive_expect_tx(&full[st], kStage);
                bulk_g2s(dst, p.k + pg * p.sk_blk + h * p.sk_h, kKB, &full[st]);
                bulk_g2s(dst + kKB, p.v + pg * p.sv_blk + h * p.sv_h, kKB, &full[st]);
                bulk_g2s(dst + 2 * kKB, p.ksc + pg * p.sks_blk + h * p.sks_h, kSB, &full[st]);
                bulk_g2s(dst + 2 * kKB + kSB, p.vsc + pg * p.svs_blk + h * p.svs_h, kSB, &full[st]);
            }
            __syncwarp();
        }
    }
    else
    {
        // ---------------------------------------------------------------- consumer warps
        int const w = warp;
        int const c = lane & 3;
        int const r = lane >> 2;
        float const sl = p.sm_scale * 1.4426950409f * __ldg(p.kgs);

        // Q fragments (A of QK), rows r and r+8, this thread's 32 dims 32c..32c+31, permuted to match the K fragments.
        uint32_t qa[8][4];
        {
            uint8_t const* q0 = p.q + (long long) t * p.sq_n + (long long) (h * kGrp + r) * p.sq_h + 32 * c;
            uint8_t const* q1 = q0 + 8 * p.sq_h;
            const uint4 a0 = __ldg(reinterpret_cast<uint4 const*>(q0));
            const uint4 a1 = __ldg(reinterpret_cast<uint4 const*>(q0 + 16));
            const uint4 b0 = __ldg(reinterpret_cast<uint4 const*>(q1));
            const uint4 b1 = __ldg(reinterpret_cast<uint4 const*>(q1 + 16));
            const uint32_t w0[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const uint32_t w1[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int ks = 0; ks < 8; ++ks)
            {
                if constexpr (DQ == 0)
                {
                    // k-step ks, reg q -> dims 32c + 4ks + 2q + {0,1}
                    qa[ks][0] = cvt_e4m3x2(w0[ks]);
                    qa[ks][1] = cvt_e4m3x2(w1[ks]);
                    qa[ks][2] = cvt_e4m3x2(w0[ks] >> 16);
                    qa[ks][3] = cvt_e4m3x2(w1[ks] >> 16);
                }
                else
                {
                    // k-step ks, reg q -> element pair (8m + i, 8m + i + 4), m = ks/2, i = 2(ks&1) + q; scaled by 2^7
                    // (exact)
                    int const m = ks >> 1;
                    const uint32_t i0 = 2u * (ks & 1);
                    const uint32_t sel0 = i0 | ((i0 + 4u) << 4);
                    const uint32_t sel1 = (i0 + 1u) | ((i0 + 5u) << 4);
                    qa[ks][0] = hmul2(cvt_e4m3x2(__byte_perm(w0[2 * m], w0[2 * m + 1], sel0)), kF16x2_128);
                    qa[ks][1] = hmul2(cvt_e4m3x2(__byte_perm(w1[2 * m], w1[2 * m + 1], sel0)), kF16x2_128);
                    qa[ks][2] = hmul2(cvt_e4m3x2(__byte_perm(w0[2 * m], w0[2 * m + 1], sel1)), kF16x2_128);
                    qa[ks][3] = hmul2(cvt_e4m3x2(__byte_perm(w1[2 * m], w1[2 * m + 1], sel1)), kF16x2_128);
                }
            }
        }

        // O accumulators (C of PV): n-tile tt, c0/c1 = row r, dims 32c + tt / 32c + 16 + tt; c2/c3 = row r + 8.
        float o[16][4];
#pragma unroll
        for (int a = 0; a < 16; ++a)
        {
            o[a][0] = 0.f;
            o[a][1] = 0.f;
            o[a][2] = 0.f;
            o[a][3] = 0.f;
        }
        float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
        int const odd = c & 1; // odd quads read the V key pair swapped (smem bank spread); PRMT selectors undo it
        const uint32_t selLo = odd ? 0x1054u : 0x5410u;
        const uint32_t selHi = odd ? 0x3276u : 0x7632u;

        for (int it = 0; it < n; ++it)
        {
            int const st = it % ST;
            const uint32_t ph = (it / ST) & 1;
            int const blk = __shfl_sync(0xffffffffu, my_blk, it);
            int const base = blk * kPage;
            bool const part = base + kPage > kv_len;
            mbar_wait(&full[st], ph);
            uint8_t const* sK = ring + st * kStage;
            uint8_t const* sV = sK + kKB;
            uint8_t const* sKs = sK + 2 * kKB;
            uint8_t const* sVs = sKs + kSB;

            // ---- S = Q K^T for keys 32w + 8j + (0..7), j = 0..3
            float s[4][4];
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                s[j][0] = 0.f;
                s[j][1] = 0.f;
                s[j][2] = 0.f;
                s[j][3] = 0.f;
                int const key = 32 * w + 8 * j + r;
                const uint4 kw = *reinterpret_cast<uint4 const*>(sK + key * 64 + 16 * c);
                uint32_t sc = cvt_e4m3x2(*reinterpret_cast<uint16_t const*>(sKs + key * 8 + 2 * c));
                if constexpr (DQ == 1)
                    sc = hmul2(sc, kF16x2_128);
                const uint32_t sA = __byte_perm(sc, 0, 0x1010); // group 2c   (dims 32c .. 32c+15)
                const uint32_t sB = __byte_perm(sc, 0, 0x3232); // group 2c+1 (dims 32c+16 .. 32c+31)
                const uint32_t kwv[4] = {kw.x, kw.y, kw.z, kw.w};
#pragma unroll
                for (int mm = 0; mm < 4; ++mm)
                {
                    const uint32_t sg = mm < 2 ? sA : sB;
                    uint32_t bq[4];
                    if constexpr (DQ == 0)
                    {
                        cvt_e2m1x8(kwv[mm], bq);
                    }
                    else
                    {
#pragma unroll
                        for (int i = 0; i < 4; ++i)
                            bq[i] = e2m1_bits(kwv[mm], i);
                    }
                    mma16816(s[j], qa[2 * mm], hmul2(bq[0], sg), hmul2(bq[1], sg));
                    mma16816(s[j], qa[2 * mm + 1], hmul2(bq[2], sg), hmul2(bq[3], sg));
                }
            }

            // ---- online softmax (log2 domain, linear recurrence), per warp over its key slice
            float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
#pragma unroll
                for (int e = 0; e < 4; ++e)
                {
                    float x = s[j][e] * sl;
                    if (part)
                    {
                        int const pos = base + 32 * w + 8 * j + 2 * c + (e & 1);
                        if (pos >= kv_len)
                            x = -INFINITY;
                    }
                    s[j][e] = x;
                }
                mx0 = fmaxf(mx0, fmaxf(s[j][0], s[j][1]));
                mx1 = fmaxf(mx1, fmaxf(s[j][2], s[j][3]));
            }
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
            float const mn0 = fmaxf(m0, mx0);
            float const mn1 = fmaxf(m1, mx1);
            float const mu0 = mn0 == -INFINITY ? 0.f : mn0;
            float const mu1 = mn1 == -INFINITY ? 0.f : mn1;
            float const al0 = ex2(m0 - mu0);
            float const al1 = ex2(m1 - mu1);
            m0 = mn0;
            m1 = mn1;
            float rs0 = 0.f, rs1 = 0.f;
            uint32_t pa[2][4];
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                float const p0 = ex2(s[j][0] - mu0);
                float const p1 = ex2(s[j][1] - mu0);
                float const p2 = ex2(s[j][2] - mu1);
                float const p3 = ex2(s[j][3] - mu1);
                rs0 += p0 + p1;
                rs1 += p2 + p3;
                pa[j >> 1][(j & 1) * 2 + 0] = pack_f16x2(p0, p1);
                pa[j >> 1][(j & 1) * 2 + 1] = pack_f16x2(p2, p3);
            }
            l0 = l0 * al0 + rs0;
            l1 = l1 * al1 + rs1;
#pragma unroll
            for (int a = 0; a < 16; ++a)
            {
                o[a][0] *= al0;
                o[a][1] *= al0;
                o[a][2] *= al1;
                o[a][3] *= al1;
            }

            // ---- O += P V, k-steps kk = 0, 1 (keys 32w + 16kk + ...); n-tile tt <-> dim 16r + tt for the B fragment
#pragma unroll
            for (int kk = 0; kk < 2; ++kk)
            {
                int const k0 = 32 * w + 16 * kk + 2 * c;
                const uint2 vX = *reinterpret_cast<uint2 const*>(sV + (k0 + odd) * 64 + 8 * r);
                const uint2 vY = *reinterpret_cast<uint2 const*>(sV + (k0 + 1 - odd) * 64 + 8 * r);
                const uint2 vZ = *reinterpret_cast<uint2 const*>(sV + (k0 + 8 + odd) * 64 + 8 * r);
                const uint2 vW = *reinterpret_cast<uint2 const*>(sV + (k0 + 9 - odd) * 64 + 8 * r);
                uint32_t s01 = cvt_e4m3x2(*reinterpret_cast<uint16_t const*>(sVs + (k0 >> 2) * 32 + 4 * r + (k0 & 3)));
                uint32_t s23 = cvt_e4m3x2(
                    *reinterpret_cast<uint16_t const*>(sVs + ((k0 + 8) >> 2) * 32 + 4 * r + ((k0 + 8) & 3)));
                if constexpr (DQ == 1)
                {
                    s01 = hmul2(s01, kF16x2_128);
                    s23 = hmul2(s23, kF16x2_128);
                }
                if (part)
                { // keys past kv_len: zero their V scales (stale page tails may hold NaN scale bytes)
                    int const pk = base + k0;
                    s01 &= (pk < kv_len ? 0x0000ffffu : 0u) | (pk + 1 < kv_len ? 0xffff0000u : 0u);
                    s23 &= (pk + 8 < kv_len ? 0x0000ffffu : 0u) | (pk + 9 < kv_len ? 0xffff0000u : 0u);
                }
                const uint32_t X[2] = {vX.x, vX.y};
                const uint32_t Y[2] = {vY.x, vY.y};
                const uint32_t Z[2] = {vZ.x, vZ.y};
                const uint32_t Wv[2] = {vW.x, vW.y};
#pragma unroll
                for (int wi = 0; wi < 2; ++wi)
                {
                    if constexpr (DQ == 0)
                    {
                        uint32_t dx[4], dy[4], dz[4], dw[4];
                        cvt_e2m1x8(X[wi], dx);
                        cvt_e2m1x8(Y[wi], dy);
                        cvt_e2m1x8(Z[wi], dz);
                        cvt_e2m1x8(Wv[wi], dw);
#pragma unroll
                        for (int bb = 0; bb < 4; ++bb)
                        {
                            int const jb = 4 * wi + bb; // dims 16r + 2jb, +1 -> n-tiles 2jb, 2jb + 1
                            const uint32_t b0l = hmul2(__byte_perm(dx[bb], dy[bb], selLo), s01);
                            const uint32_t b1l = hmul2(__byte_perm(dz[bb], dw[bb], selLo), s23);
                            const uint32_t b0h = hmul2(__byte_perm(dx[bb], dy[bb], selHi), s01);
                            const uint32_t b1h = hmul2(__byte_perm(dz[bb], dw[bb], selHi), s23);
                            mma16816(o[2 * jb], pa[kk], b0l, b1l);
                            mma16816(o[2 * jb + 1], pa[kk], b0h, b1h);
                        }
                    }
                    else
                    {
                        const uint32_t u0l = __byte_perm(X[wi], Y[wi], selLo);
                        const uint32_t u0h = __byte_perm(X[wi], Y[wi], selHi);
                        const uint32_t u1l = __byte_perm(Z[wi], Wv[wi], selLo);
                        const uint32_t u1h = __byte_perm(Z[wi], Wv[wi], selHi);
#pragma unroll
                        for (int i = 0; i < 4; ++i)
                        { // elements 8wi + i (lo) and 8wi + 4 + i (hi) -> n-tiles of the same index
                            mma16816(
                                o[8 * wi + i], pa[kk], hmul2(e2m1_bits(u0l, i), s01), hmul2(e2m1_bits(u1l, i), s23));
                            mma16816(o[8 * wi + 4 + i], pa[kk], hmul2(e2m1_bits(u0h, i), s01),
                                hmul2(e2m1_bits(u1h, i), s23));
                        }
                    }
                }
            }
            __syncwarp();
            if (lane == 0)
                mbar_arrive(&empty[st]);
        }

        // ---- merge the 4 consumer warps (fixed order) in smem
        l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
        l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
        l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
        l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
        if (c == 0)
        {
            sm_m[w * kGrp + r] = m0;
            sm_m[w * kGrp + r + 8] = m1;
            sm_l[w * kGrp + r] = l0;
            sm_l[w * kGrp + r + 8] = l1;
        }
        named_bar_sync(1, kNcw * 32); // all consumers are done with the ring; m / l visible
        float f0, f1;
        {
            float M0 = -INFINITY, M1 = -INFINITY;
#pragma unroll
            for (int u = 0; u < kNcw; ++u)
            {
                M0 = fmaxf(M0, sm_m[u * kGrp + r]);
                M1 = fmaxf(M1, sm_m[u * kGrp + r + 8]);
            }
            float const Mu0 = M0 == -INFINITY ? 0.f : M0;
            float const Mu1 = M1 == -INFINITY ? 0.f : M1;
            float L0 = 0.f, L1 = 0.f;
#pragma unroll
            for (int u = 0; u < kNcw; ++u)
            {
                L0 += sm_l[u * kGrp + r] * ex2(sm_m[u * kGrp + r] - Mu0);
                L1 += sm_l[u * kGrp + r + 8] * ex2(sm_m[u * kGrp + r + 8] - Mu1);
            }
            f0 = L0 > 0.f ? ex2(m0 - Mu0) / L0 : 0.f;
            f1 = L1 > 0.f ? ex2(m1 - Mu1) / L1 : 0.f;
        }
        {
            float* ob = obuf + (w * kGrp) * kORow;
#pragma unroll
            for (int k4 = 0; k4 < 4; ++k4)
            {
                *reinterpret_cast<float4*>(ob + r * kORow + 32 * c + 4 * k4) = make_float4(
                    o[4 * k4][0] * f0, o[4 * k4 + 1][0] * f0, o[4 * k4 + 2][0] * f0, o[4 * k4 + 3][0] * f0);
                *reinterpret_cast<float4*>(ob + r * kORow + 32 * c + 16 + 4 * k4) = make_float4(
                    o[4 * k4][1] * f0, o[4 * k4 + 1][1] * f0, o[4 * k4 + 2][1] * f0, o[4 * k4 + 3][1] * f0);
                *reinterpret_cast<float4*>(ob + (r + 8) * kORow + 32 * c + 4 * k4) = make_float4(
                    o[4 * k4][2] * f1, o[4 * k4 + 1][2] * f1, o[4 * k4 + 2][2] * f1, o[4 * k4 + 3][2] * f1);
                *reinterpret_cast<float4*>(ob + (r + 8) * kORow + 32 * c + 16 + 4 * k4) = make_float4(
                    o[4 * k4][3] * f1, o[4 * k4 + 1][3] * f1, o[4 * k4 + 2][3] * f1, o[4 * k4 + 3][3] * f1);
            }
        }
        named_bar_sync(1, kNcw * 32);
        {
            int const tid = threadIdx.x; // 0..127
            int const row = tid >> 3;
            int const d0 = (tid & 7) * 16;
            float acc[16];
#pragma unroll
            for (int q4 = 0; q4 < 4; ++q4)
            {
                const float4 v = *reinterpret_cast<float4 const*>(obuf + row * kORow + d0 + 4 * q4);
                acc[4 * q4 + 0] = v.x;
                acc[4 * q4 + 1] = v.y;
                acc[4 * q4 + 2] = v.z;
                acc[4 * q4 + 3] = v.w;
            }
#pragma unroll
            for (int u = 1; u < kNcw; ++u)
            {
#pragma unroll
                for (int q4 = 0; q4 < 4; ++q4)
                {
                    const float4 v = *reinterpret_cast<float4 const*>(obuf + (u * kGrp + row) * kORow + d0 + 4 * q4);
                    acc[4 * q4 + 0] += v.x;
                    acc[4 * q4 + 1] += v.y;
                    acc[4 * q4 + 2] += v.z;
                    acc[4 * q4 + 3] += v.w;
                }
            }
            float const osc = __ldg(p.vgs) * (DQ == 1 ? 128.f : 1.f);
#pragma unroll
            for (int e = 0; e < 16; ++e)
                acc[e] *= osc;
            if constexpr (S == 1)
            {
                __nv_bfloat16* dst = p.out + (long long) t * p.so_n + (long long) (h * kGrp + row) * p.so_h + d0;
                uint32_t pk[8];
#pragma unroll
                for (int e = 0; e < 8; ++e)
                {
                    __nv_bfloat162 b2 = __floats2bfloat162_rn(acc[2 * e], acc[2 * e + 1]);
                    pk[e] = *reinterpret_cast<uint32_t*>(&b2);
                }
                reinterpret_cast<uint4*>(dst)[0] = make_uint4(pk[0], pk[1], pk[2], pk[3]);
                reinterpret_cast<uint4*>(dst)[1] = make_uint4(pk[4], pk[5], pk[6], pk[7]);
            }
            else
            {
#pragma unroll
                for (int q4 = 0; q4 < 4; ++q4)
                {
                    *reinterpret_cast<float4*>(obuf + row * kORow + d0 + 4 * q4)
                        = make_float4(acc[4 * q4], acc[4 * q4 + 1], acc[4 * q4 + 2], acc[4 * q4 + 3]);
                }
                if ((tid & 7) == 0)
                {
                    float M = -INFINITY;
#pragma unroll
                    for (int u = 0; u < kNcw; ++u)
                        M = fmaxf(M, sm_m[u * kGrp + row]);
                    float const Mu = M == -INFINITY ? 0.f : M;
                    float L = 0.f;
#pragma unroll
                    for (int u = 0; u < kNcw; ++u)
                        L += sm_l[u * kGrp + row] * ex2(sm_m[u * kGrp + row] - Mu);
                    sm_lse[row] = L > 0.f ? Mu + lg2(L) : -INFINITY;
                }
            }
        }
    }

    if constexpr (S > 1)
    {
        // ---- split merge over DSMEM (stock merge formula); CTA rank `split` writes heads [split*HPS, (split+1)*HPS)
        __syncwarp(); // barrier.cluster.*.aligned: every warp converged
        cl_sync();
        constexpr int HPS = kGrp / S;
        for (int qq = threadIdx.x; qq < HPS * 32; qq += kThreads)
        {
            int const row = split * HPS + qq / 32;
            int const d4 = (qq % 32) * 4;
            float lk[S];
            float lmax = -INFINITY;
#pragma unroll
            for (int k = 0; k < S; ++k)
            {
                lk[k] = ld_dsm_f(map_rank(sm_lse + row, k));
                lmax = fmaxf(lmax, lk[k]);
            }
            if (lmax == -INFINITY)
                lmax = 0.f;
            float den = 0.f;
#pragma unroll
            for (int k = 0; k < S; ++k)
            {
                lk[k] = ex2(lk[k] - lmax);
                den += lk[k];
            }
            float const dd = den > 0.f ? den : 1.f;
            float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
            for (int k = 0; k < S; ++k)
            {
                float const wk = lk[k] / dd;
                const float4 v = ld_dsm_f4(map_rank(obuf + row * kORow + d4, k));
                acc.x += v.x * wk;
                acc.y += v.y * wk;
                acc.z += v.z * wk;
                acc.w += v.w * wk;
            }
            __nv_bfloat16* dst = p.out + (long long) t * p.so_n + (long long) (h * kGrp + row) * p.so_h + d4;
            __nv_bfloat162 lo = __floats2bfloat162_rn(acc.x, acc.y);
            __nv_bfloat162 hi = __floats2bfloat162_rn(acc.z, acc.w);
            *reinterpret_cast<uint2*>(dst)
                = make_uint2(*reinterpret_cast<uint32_t*>(&lo), *reinterpret_cast<uint32_t*>(&hi));
        }
        cl_sync(); // peers keep their smem alive until every CTA has read it
    }
    griddep_launch();
#endif // defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

// ------------------------------------------------------------------------------------------------ host side
template <int S, int ST, int DQ>
static cudaError_t launch_t(Params const& p, int pdl, cudaStream_t stream)
{
    auto kern = csd_kernel<S, ST, DQ>;
    int const smem = smem_bytes(ST);
    static int attr_dev = -1;
    int dev = 0;
    cudaGetDevice(&dev);
    if (attr_dev != dev)
    {
        cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (e != cudaSuccess)
            return e;
        attr_dev = dev;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(S, p.num_kv_heads, p.total_q);
    cfg.blockDim = dim3(kThreads, 1, 1);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute at[2];
    int na = 0;
    if (S > 1)
    {
        at[na].id = cudaLaunchAttributeClusterDimension;
        at[na].val.clusterDim.x = S;
        at[na].val.clusterDim.y = 1;
        at[na].val.clusterDim.z = 1;
        ++na;
    }
    if (pdl)
    {
        at[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        at[na].val.programmaticStreamSerializationAllowed = 1;
        ++na;
    }
    cfg.attrs = at;
    cfg.numAttrs = na;
    return cudaLaunchKernelEx(&cfg, kern, p);
}

template <int S, int ST, int DQ>
static cudaError_t attrs_t(int* out)
{
    auto kern = csd_kernel<S, ST, DQ>;
    int const smem = smem_bytes(ST);
    cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (e != cudaSuccess)
        return e;
    cudaFuncAttributes fa;
    e = cudaFuncGetAttributes(&fa, kern);
    if (e != cudaSuccess)
        return e;
    int nb = 0;
    e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kern, kThreads, smem);
    if (e != cudaSuccess)
        return e;
    out[0] = fa.numRegs;
    out[1] = (int) fa.localSizeBytes;
    out[2] = smem;
    out[3] = nb;
    out[4] = fa.binaryVersion;
    return cudaSuccess;
}

#define CSD_DISPATCH(FN, ...)                                                                                          \
    switch (S * 100 + ST * 10 + DQ)                                                                                    \
    {                                                                                                                  \
    case 120: return FN<1, 2, 0>(__VA_ARGS__);                                                                         \
    case 130: return FN<1, 3, 0>(__VA_ARGS__);                                                                         \
    case 140: return FN<1, 4, 0>(__VA_ARGS__);                                                                         \
    case 220: return FN<2, 2, 0>(__VA_ARGS__);                                                                         \
    case 230: return FN<2, 3, 0>(__VA_ARGS__);                                                                         \
    case 240: return FN<2, 4, 0>(__VA_ARGS__);                                                                         \
    case 420: return FN<4, 2, 0>(__VA_ARGS__);                                                                         \
    case 430: return FN<4, 3, 0>(__VA_ARGS__);                                                                         \
    case 440: return FN<4, 4, 0>(__VA_ARGS__);                                                                         \
    case 820: return FN<8, 2, 0>(__VA_ARGS__);                                                                         \
    case 830: return FN<8, 3, 0>(__VA_ARGS__);                                                                         \
    case 840: return FN<8, 4, 0>(__VA_ARGS__);                                                                         \
    case 121: return FN<1, 2, 1>(__VA_ARGS__);                                                                         \
    case 131: return FN<1, 3, 1>(__VA_ARGS__);                                                                         \
    case 141: return FN<1, 4, 1>(__VA_ARGS__);                                                                         \
    case 221: return FN<2, 2, 1>(__VA_ARGS__);                                                                         \
    case 231: return FN<2, 3, 1>(__VA_ARGS__);                                                                         \
    case 241: return FN<2, 4, 1>(__VA_ARGS__);                                                                         \
    case 421: return FN<4, 2, 1>(__VA_ARGS__);                                                                         \
    case 431: return FN<4, 3, 1>(__VA_ARGS__);                                                                         \
    case 441: return FN<4, 4, 1>(__VA_ARGS__);                                                                         \
    case 821: return FN<8, 2, 1>(__VA_ARGS__);                                                                         \
    case 831: return FN<8, 3, 1>(__VA_ARGS__);                                                                         \
    case 841: return FN<8, 4, 1>(__VA_ARGS__);                                                                         \
    default: return cudaErrorInvalidValue;                                                                             \
    }

} // namespace csd

cudaError_t invokeMinimaxM3CuteSparseDecode(
    MinimaxM3CuteSparseDecodeParams const& p, int splits, int stages, int dequant, bool pdl, cudaStream_t stream)
{
    if (p.total_q <= 0)
    {
        return cudaSuccess;
    }
    int const S = splits, ST = stages, DQ = dequant;
    using namespace csd;
    CSD_DISPATCH(launch_t, p, pdl ? 1 : 0, stream)
}

cudaError_t minimaxM3CuteSparseDecodeAttrs(int splits, int stages, int dequant, int* out5)
{
    int const S = splits, ST = stages, DQ = dequant;
    using namespace csd;
    CSD_DISPATCH(attrs_t, out5)
}

} // namespace kernels

TRTLLM_NAMESPACE_END
