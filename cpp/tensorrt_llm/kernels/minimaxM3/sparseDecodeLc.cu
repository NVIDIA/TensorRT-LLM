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

// MiniMax-M3 long-context sparse decode attention in one launch.
//
// Grid: one cluster of 16 CTAs per query token; CTA `r` attends the token's `r`-th selected block (the Triton
// producer's chunk `r` at 16 chunks), keeps the partial in shared memory, and after a cluster barrier merges head
// `r` from all 16 partials over DSMEM, writing the BF16 row and, optionally, its MXFP8 copy and swizzled UE8M0
// scales exactly as the Triton merge's `_store_mxfp8_head` lays them out.
//
// Per-block math follows the Triton producer: NVFP4 K/V dequantized to FP16 (exact), FP8 q widened to FP16, QK and
// PV on FP16 tensor cores with FP32 accumulation, log2-domain softmax with either the linear denominator or the
// log-domain recurrence.

#include "tensorrt_llm/kernels/minimaxM3/sparseDecodeLc.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

constexpr int kHeads = 16;
constexpr int kDim = 128;
constexpr int kPage = 128;
constexpr int kTopk = 16;
constexpr int kThreads = 128;
constexpr int kPitch = kDim + 8; // halves per smem row: 272 B, conflict-free ldmatrix
constexpr float kNegInf = -__builtin_huge_valf();

using Params = MinimaxM3SparseDecodeLcParams;

struct Smem
{
    __half qs[kHeads * kPitch];
    __half vs[kPage * kPitch];
    __half ps[kHeads * kPitch];
    float recv[kTopk][kDim]; // head `rank` partials pushed by the 16 CTAs, chunk-major
    float recv_lse[kTopk];
    float red[4][kHeads];
    float rowm[kHeads];
    float rowl[kHeads];
    alignas(8) unsigned long long rbar; // the split-K partials of head `rank` landed (complete_tx)
};

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__device__ __forceinline__ uint32_t smem_u32(void const* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ uint32_t e2m1x2_to_f16x2(uint32_t byte)
{
    uint32_t out;
    asm("{\n .reg .b8 s;\n cvt.u8.u32 s, %1;\n cvt.rn.f16x2.e2m1x2 %0, s;\n}" : "=r"(out) : "r"(byte));
    return out;
}

__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint16_t pair)
{
    uint32_t out;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(out) : "h"(pair));
    return out;
}

__device__ __forceinline__ float ex2(float x)
{
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float lg2(float x)
{
    float y;
    asm("lg2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], uint32_t addr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void mma(float (&c)[4], uint32_t const (&a)[4], uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t map_cluster(void const* local, uint32_t rank)
{
    uint32_t remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(smem_u32(local)), "r"(rank));
    return remote;
}

// Posted DSMEM stores that complete_tx on the receiving CTA's mbarrier `bar`.
__device__ __forceinline__ void st_async_v2(uint32_t addr, float a, float b, uint32_t bar)
{
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1, %2}, [%3];" ::"r"(addr),
                 "f"(a), "f"(b), "r"(bar)
                 : "memory");
}

__device__ __forceinline__ void st_async_f32(uint32_t addr, float a, uint32_t bar)
{
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(addr),
                 "r"(__float_as_uint(a)), "r"(bar)
                 : "memory");
}

// One packed NVFP4 row (64 bytes, 128 values) with its 8 E4M3 scales -> 128 fp16 values in smem.
__device__ __forceinline__ void dequant_row(__half* dst, uint8_t const* packed, uint8_t const (&sc)[8], bool valid)
{
    uint4 const* src = reinterpret_cast<uint4 const*>(packed);
    uint32_t* d = reinterpret_cast<uint32_t*>(dst);
#pragma unroll
    for (int c = 0; c < 4; ++c)
    {
        uint4 w = valid ? __ldg(src + c) : make_uint4(0, 0, 0, 0);
        uint32_t words[4] = {w.x, w.y, w.z, w.w};
        // 16 bytes = 32 values = scale columns 2c and 2c + 1.
#pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            uint8_t const s = valid ? sc[2 * c + h] : 0;
            uint32_t const sh = e4m3x2_to_f16x2(static_cast<uint16_t>(s | (s << 8)));
            __half2 const scale = *reinterpret_cast<__half2 const*>(&sh);
#pragma unroll
            for (int j = 0; j < 2; ++j)
            {
                uint32_t const word = words[2 * h + j];
#pragma unroll
                for (int b = 0; b < 4; ++b)
                {
                    uint32_t const v2 = e2m1x2_to_f16x2((word >> (8 * b)) & 0xFF);
                    __half2 const x = __hmul2(*reinterpret_cast<__half2 const*>(&v2), scale);
                    d[c * 16 + h * 8 + j * 4 + b] = *reinterpret_cast<uint32_t const*>(&x);
                }
            }
        }
    }
}
#endif // defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

template <bool LINEAR, bool MXFP8, bool EARLY>
__global__ void __launch_bounds__(kThreads) m3_lc_attn_kernel(__grid_constant__ Params const p)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    extern __shared__ __align__(128) uint8_t smem_raw[];
    Smem& s = *reinterpret_cast<Smem*>(smem_raw);
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    int const tok = blockIdx.x / kTopk;
    uint32_t rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));

    if (tid == 0)
    {
        // One arrival (here) plus every cluster CTA's partial row (kDim floats) and LSE for head `rank`.
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_u32(&s.rbar)) : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(&s.rbar)),
                     "r"(kTopk * (kDim + 1) * 4)
                     : "memory");
    }
    // Every peer must be running before its shared memory takes a DSMEM store: arrive now, wait before pushing.
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

    // Everything that does not depend on the top-16 table is read before the grid dependency: the request's length
    // (written before the step's graph runs) and the query rows (written by the producer, which completed before the
    // select ahead of this grid could release it: the select triggers its dependents only after its own grid wait).
    int const req = tok / p.dql;
    int const qoff = tok - req * p.dql;
    int const seq = p.seq_lens[req];
    int const kv_len = max(seq - p.dql + qoff + 1, 0);
    int const nblocks = (kv_len + kPage - 1) / kPage;
    int const real_topk = min(kTopk, nblocks);
    // q: 16 heads x 128 e4m3 -> fp16; thread = (head, 16-value chunk).
    {
        int const h = tid >> 3;
        int const c = (tid & 7) * 16;
        uint4 const w = *reinterpret_cast<uint4 const*>(p.q + tok * p.q_tok + h * p.q_head + c);
        uint32_t const words[4] = {w.x, w.y, w.z, w.w};
        uint32_t* d = reinterpret_cast<uint32_t*>(s.qs + h * kPitch + c);
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            d[2 * i] = e4m3x2_to_f16x2(static_cast<uint16_t>(words[i] & 0xFFFF));
            d[2 * i + 1] = e4m3x2_to_f16x2(static_cast<uint16_t>(words[i] >> 16));
        }
    }

    asm volatile("griddepcontrol.wait;" ::: "memory");
    if constexpr (EARLY)
    {
        asm volatile("griddepcontrol.launch_dependents;");
    }
    int blk = static_cast<int>(rank) < real_topk ? p.topk[tok * p.t_tok + rank * p.t_k] : -1;
    bool const active = blk >= 0;
    int64_t const page_pw = !active ? 0 : p.block_table[req * p.bt_stride + blk];

    float m_row[2] = {kNegInf, kNegInf};
    float acc[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i)
            acc[j][i] = 0.F;
    float l_row[2] = {0.F, 0.F};

    // K stays packed in registers: the MMA B fragment of token n = lane / 4 at k-step s is the f16x2 pair of
    // packed bytes 8s + q and 8s + 4 + q (q = lane % 4), both under scale column s.
    uint32_t kw[4][16];
    uint2 ksc[4];
    int64_t page = 0;
    if (active)
    {
        page = page_pw;
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            int const n = warp * 32 + j * 8 + (lane >> 2);
            bool const valid = blk * kPage + n < kv_len;
            uint4 const* src = reinterpret_cast<uint4 const*>(p.k + page * p.k_blk + n * 64);
#pragma unroll
            for (int c = 0; c < 4; ++c)
            {
                uint4 const w = valid ? __ldg(src + c) : make_uint4(0, 0, 0, 0);
                kw[j][4 * c] = w.x;
                kw[j][4 * c + 1] = w.y;
                kw[j][4 * c + 2] = w.z;
                kw[j][4 * c + 3] = w.w;
            }
            ksc[j] = valid ? __ldg(reinterpret_cast<uint2 const*>(p.ksc + page * p.ksc_blk + n * 8)) : make_uint2(0, 0);
        }
        bool const valid = blk * kPage + tid < kv_len;
        uint8_t vsc[8];
        if (valid)
        {
            uint8_t const* vb = p.vsc + page * p.vsc_blk + (tid >> 2) * 32 + (tid & 3);
#pragma unroll
            for (int c = 0; c < 8; ++c)
                vsc[c] = __ldg(vb + 4 * c);
        }
        dequant_row(s.vs + tid * kPitch, p.v + page * p.v_blk + tid * 64, vsc, valid);
    }
    __syncthreads();

    if (active)
    {
        float const scale_log2 = p.sm_scale * 1.4426950409F * __ldg(p.kgs);
        int const q4 = (lane & 3) * 8;
        // S = q K^T for this warp's 32 tokens: rows g / g + 8, columns 2 * (lane % 4) + {0, 1} of each n-tile.
        float sacc[4][4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i)
                sacc[j][i] = 0.F;
#pragma unroll
        for (int ks = 0; ks < kDim / 16; ++ks)
        {
            uint32_t a[4];
            ldsm_x4(a, smem_u32(s.qs + (lane & 15) * kPitch + ks * 16 + (lane >> 4) * 8));
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                uint32_t const scw = ks < 4 ? ksc[j].x : ksc[j].y;
                uint8_t const sc = static_cast<uint8_t>(scw >> (8 * (ks & 3)));
                uint32_t const sh = e4m3x2_to_f16x2(static_cast<uint16_t>(sc | (sc << 8)));
                __half2 const scale = *reinterpret_cast<__half2 const*>(&sh);
                uint32_t const r0 = e2m1x2_to_f16x2((kw[j][2 * ks] >> q4) & 0xFF);
                uint32_t const r1 = e2m1x2_to_f16x2((kw[j][2 * ks + 1] >> q4) & 0xFF);
                __half2 const h0 = __hmul2(*reinterpret_cast<__half2 const*>(&r0), scale);
                __half2 const h1 = __hmul2(*reinterpret_cast<__half2 const*>(&r1), scale);
                mma(sacc[j], a, *reinterpret_cast<uint32_t const*>(&h0), *reinterpret_cast<uint32_t const*>(&h1));
            }
        }
        int const g = lane >> 2;
        float mloc[2] = {kNegInf, kNegInf};
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                int const t = warp * 32 + j * 8 + 2 * (lane & 3) + (i & 1);
                float const v = blk * kPage + t < kv_len ? 0.F + sacc[j][i] * scale_log2 : kNegInf;
                sacc[j][i] = v;
                mloc[i >> 1] = fmaxf(mloc[i >> 1], v);
            }
#pragma unroll
        for (int r = 0; r < 2; ++r)
        {
            mloc[r] = fmaxf(mloc[r], __shfl_xor_sync(0xffffffff, mloc[r], 1));
            mloc[r] = fmaxf(mloc[r], __shfl_xor_sync(0xffffffff, mloc[r], 2));
        }
        if ((lane & 3) == 0)
        {
            s.red[warp][g] = mloc[0];
            s.red[warp][g + 8] = mloc[1];
        }
        __syncthreads();
        if (tid < kHeads)
            s.rowm[tid] = fmaxf(fmaxf(s.red[0][tid], s.red[1][tid]), fmaxf(s.red[2][tid], s.red[3][tid]));
        __syncthreads();
        m_row[0] = s.rowm[g];
        m_row[1] = s.rowm[g + 8];
        float lloc[2] = {0.F, 0.F};
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int i = 0; i < 4; i += 2)
            {
                float const p0 = ex2(sacc[j][i] - m_row[i >> 1]);
                float const p1 = ex2(sacc[j][i + 1] - m_row[i >> 1]);
                lloc[i >> 1] += p0 + p1;
                int const row = g + (i >> 1) * 8;
                int const t = warp * 32 + j * 8 + 2 * (lane & 3);
                *reinterpret_cast<__half2*>(s.ps + row * kPitch + t) = __floats2half2_rn(p0, p1);
            }
#pragma unroll
        for (int r = 0; r < 2; ++r)
        {
            lloc[r] += __shfl_xor_sync(0xffffffff, lloc[r], 1);
            lloc[r] += __shfl_xor_sync(0xffffffff, lloc[r], 2);
        }
        __syncthreads(); // rowm reads done before red is reused; ps complete
        if ((lane & 3) == 0)
        {
            s.red[warp][g] = lloc[0];
            s.red[warp][g + 8] = lloc[1];
        }
        // O[:, warp's 32 dims] = P V over all 128 tokens.
#pragma unroll
        for (int kk = 0; kk < kPage; kk += 16)
        {
            uint32_t a[4];
            ldsm_x4(a, smem_u32(s.ps + (lane & 15) * kPitch + kk + (lane >> 4) * 8));
#pragma unroll
            for (int jp = 0; jp < 2; ++jp)
            {
                uint32_t b[4];
                int const t = kk + (lane & 7) + ((lane >> 3) & 1) * 8;
                int const d = warp * 32 + jp * 16 + (lane >> 4) * 8;
                ldsm_x4_t(b, smem_u32(s.vs + t * kPitch + d));
                mma(acc[2 * jp], a, b[0], b[1]);
                mma(acc[2 * jp + 1], a, b[2], b[3]);
            }
        }
        __syncthreads();
        if (tid < kHeads)
            s.rowl[tid] = (s.red[0][tid] + s.red[1][tid]) + (s.red[2][tid] + s.red[3][tid]);
        __syncthreads();
        l_row[0] = s.rowl[g];
        l_row[1] = s.rowl[g + 8];
    }

    // Normalize the partial and push head h's row with its LSE (log2) into CTA h's recv[rank] slot, as the Triton
    // producer stores chunk `rank`; st.async stores that complete_tx on CTA h's barrier.
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    {
        int const g = lane >> 2;
        float const vgs = __ldg(p.vgs);
#pragma unroll
        for (int r = 0; r < 2; ++r)
        {
            float scale, lse;
            if constexpr (LINEAR)
            {
                scale = l_row[r] > 0.F ? 1.0F / l_row[r] : 0.F;
                lse = l_row[r] > 0.F ? m_row[r] + lg2(l_row[r]) : kNegInf;
            }
            else
            {
                float const lse_i = active ? m_row[r] + lg2(ex2(kNegInf - m_row[r]) + l_row[r]) : kNegInf;
                scale = lse_i > kNegInf ? ex2(m_row[r] - lse_i) : 0.F;
                lse = lse_i;
            }
            scale = scale * vgs;
            int const row = g + r * 8;
            uint32_t const dst = map_cluster(&s.recv[rank][0], row);
            uint32_t const rb = map_cluster(&s.rbar, row);
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                int const d = warp * 32 + j * 8 + 2 * (lane & 3);
                st_async_v2(dst + d * 4, acc[j][2 * r] * scale, acc[j][2 * r + 1] * scale, rb);
            }
            if (warp == 0 && (lane & 3) == 0)
                st_async_f32(map_cluster(&s.recv_lse[rank], row), lse, rb);
        }
    }
    {
        // Every CTA's partial row and LSE for head `rank` landed (complete_tx on this CTA's barrier).
        uint32_t done = 0;
        while (!done)
        {
            asm volatile(
                "{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
                " selp.u32 %0, 1, 0, p;\n}"
                : "=r"(done)
                : "r"(smem_u32(&s.rbar))
                : "memory");
        }
    }

    // CTA `rank` merges head `rank` across the 16 partials; thread = output dim.
    int const h = static_cast<int>(rank);
    float lse_c[kTopk];
    float lse_max = kNegInf;
#pragma unroll
    for (int c = 0; c < kTopk; ++c)
    {
        lse_c[c] = s.recv_lse[c];
        lse_max = fmaxf(lse_max, lse_c[c]);
    }
    lse_max = lse_max == kNegInf ? 0.F : lse_max;
    float denom = 0.F;
#pragma unroll
    for (int c = 0; c < kTopk; ++c)
    {
        lse_c[c] = ex2(lse_c[c] - lse_max);
        denom += lse_c[c];
    }
    denom = denom > 0.F ? denom : 1.F;
    float o = 0.F;
#pragma unroll
    for (int c = 0; c < kTopk; ++c)
        o += s.recv[c][tid] * __fdiv_rn(lse_c[c], denom);
    __nv_bfloat16 const ob = __float2bfloat16_rn(o);
    p.out[tok * p.out_stride + h * kDim + tid] = ob;

    if constexpr (MXFP8)
    {
        float const x = __bfloat162float(ob);
        float amax = fabsf(x);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
        float rcp;
        asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(rcp) : "f"(448.0F));
        float const sfv = amax * rcp;
        int const bits = __float_as_int(sfv);
        int e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0);
        e = min(e, 254);
        float const qs = amax != 0.F ? __int_as_float((254 - e) << 23) : 0.F;
        float const xq = x * qs;
        uint16_t q2;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(q2) : "f"(0.F), "f"(xq));
        p.q8[static_cast<int64_t>(tok) * (p.num_sf_k_tiles * 128) + h * kDim + tid] = static_cast<uint8_t>(q2 & 0xFF);
        if (lane == 0)
        {
            int const k_sf = h * 4 + warp;
            int64_t const off = static_cast<int64_t>(tok / 128) * p.num_sf_k_tiles * 512 + (k_sf / 4) * 512
                + (tok % 32) * 16 + ((tok % 128) / 32) * 4 + k_sf % 4;
            p.sf[off] = static_cast<uint8_t>(e);
        }
    }
    if constexpr (!EARLY)
    {
        asm volatile("griddepcontrol.launch_dependents;");
    }
#endif // defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

template <bool LINEAR, bool MXFP8, bool EARLY>
cudaError_t configure()
{
    auto kernel = m3_lc_attn_kernel<LINEAR, MXFP8, EARLY>;
    cudaError_t err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Smem));
    if (err != cudaSuccess)
    {
        return err;
    }
    return cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
}

template <bool LINEAR, bool MXFP8, bool EARLY>
cudaError_t launch(Params const& prm, int num_tokens, bool use_pdl, cudaStream_t stream)
{
    // Kernel attributes are set once per process, on the first launch (outside CUDA-graph capture: warmup runs
    // every variant eagerly first).
    static cudaError_t const configured = configure<LINEAR, MXFP8, EARLY>();
    if (configured != cudaSuccess)
    {
        return configured;
    }
    auto kernel = m3_lc_attn_kernel<LINEAR, MXFP8, EARLY>;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(num_tokens * kTopk);
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = sizeof(Smem);
    cfg.stream = stream;
    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = kTopk;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[1].val.programmaticStreamSerializationAllowed = use_pdl ? 1 : 0;
    cfg.attrs = attrs;
    cfg.numAttrs = 2;
    return cudaLaunchKernelEx(&cfg, kernel, prm);
}

} // namespace

cudaError_t invokeMinimaxM3SparseDecodeLc(MinimaxM3SparseDecodeLcParams const& params, int32_t numTokens,
    bool linearSoftmax, bool mxfp8, bool earlyTrigger, bool usePdl, cudaStream_t stream)
{
    if (!earlyTrigger)
    {
        return cudaErrorNotSupported;
    }
    if (linearSoftmax)
    {
        return mxfp8 ? launch<true, true, true>(params, numTokens, usePdl, stream)
                     : launch<true, false, true>(params, numTokens, usePdl, stream);
    }
    return mxfp8 ? launch<false, true, true>(params, numTokens, usePdl, stream)
                 : launch<false, false, true>(params, numTokens, usePdl, stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
