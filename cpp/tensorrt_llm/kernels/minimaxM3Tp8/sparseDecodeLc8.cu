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

// MiniMax-M3 TP8 long-context sparse decode attention in one launch (8 query heads on one KV head).
//
// Grid: one cluster of 16 CTAs per query token; CTA `r` attends the token's `r`-th selected block. The QK and PV
// MMAs are transposed (S^T = K q^T, O^T = V^T P^T) so the 8 heads fill the MMA n dimension. Each CTA pushes its
// normalized partial over DSMEM so that CTA `r` holds (head r / 2, dims (r % 2) * 64 + [0, 64)) of all 16 partials,
// merges them, and writes the BF16 row and, optionally, its MXFP8 copy and swizzled UE8M0 scales.
//
// Per-block math: NVFP4 K/V dequantized to FP16 (exact), FP8 q widened to FP16, QK and PV on FP16 tensor cores with
// FP32 accumulation, log2-domain softmax with either the linear denominator or the log-domain recurrence.
//
// PREWAIT stages everything that does not depend on the top-16 table before the grid dependency wait: the request's
// seq_lens entry, the q rows and the request's block-table row (in dynamic shared memory after Smem8, when it has at
// most kBtCap entries). The arithmetic is the same in both variants.

#include "tensorrt_llm/kernels/minimaxM3Tp8/sparseDecodeLc8.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

constexpr int kDim = 128;
constexpr int kPage = 128;
constexpr int kTopk = 16;
constexpr int kThreads = 128;
constexpr int kPitch = kDim + 8; // halves per smem row: 272 B, conflict-free ldmatrix
constexpr float kNegInf = -__builtin_huge_valf();
constexpr int kHeads8 = 8;
constexpr int kHalfDim = kDim / 2;
constexpr int kBtCap = 8192; // block-table entries staged in shared memory (8192 x 128 = 1M tokens)

using Params = MinimaxM3Tp8SparseDecodeLc8Params;

struct Smem8
{
    __half qs[kHeads8 * kPitch];
    __half vs[kPage * kPitch];
    __half ps[kHeads8 * kPitch]; // P^T: [head][token]
    float recv[kTopk][kHalfDim]; // (head rank / 2, dims (rank % 2) * 64 + [0, 64)) partials of the 16 CTAs
    float recv_lse[kTopk];
    float part[kHalfDim];
    float red[4][kHeads8];
    float rowm[kHeads8];
    float rowl[kHeads8];
};

template <bool PREWAIT>
constexpr size_t smemBytes()
{
    return PREWAIT ? sizeof(Smem8) + kBtCap * sizeof(int) : sizeof(Smem8);
}

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

__device__ __forceinline__ void cluster_sync()
{
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}

__device__ __forceinline__ uint32_t map_cluster(void const* local, uint32_t rank)
{
    uint32_t remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(smem_u32(local)), "r"(rank));
    return remote;
}

__device__ __forceinline__ void st_cluster_f32(uint32_t addr, float a)
{
    asm volatile("st.shared::cluster.f32 [%0], %1;" ::"r"(addr), "f"(a) : "memory");
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

// q: 8 heads x 128 e4m3 -> fp16; threads 0..63 = (head, 16-value chunk).
__device__ __forceinline__ void load_q(Smem8& s, Params const& p, int tok, int tid)
{
    if (tid < kHeads8 * 8)
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
}
#endif // defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

template <bool LINEAR, bool MXFP8, bool PREWAIT>
__global__ void __launch_bounds__(kThreads) m3_tp8_lc8_attn_kernel(__grid_constant__ Params const p)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    extern __shared__ __align__(128) uint8_t smem_raw[];
    Smem8& s = *reinterpret_cast<Smem8*>(smem_raw);
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    int const g = lane >> 2;
    int const qd = lane & 3;
    int const tok = blockIdx.x / kTopk;
    uint32_t rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));

    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    if constexpr (!PREWAIT)
    {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        asm volatile("griddepcontrol.launch_dependents;");
    }

    int const req = tok / p.dql;
    int const qoff = tok - req * p.dql;
    int const seq = p.seq_lens[req];
    int const kv_len = max(seq - p.dql + qoff + 1, 0);
    int const nblocks = (kv_len + kPage - 1) / kPage;
    int const real_topk = min(kTopk, nblocks);

    // The request's block-table row, staged before the grid dependency (written before the graph runs).
    int* const s_bt = reinterpret_cast<int*>(smem_raw + sizeof(Smem8));
    bool bt_smem = false;
    if constexpr (PREWAIT)
    {
        int const nbt = (max(seq, 0) + kPage - 1) / kPage;
        bt_smem = nbt <= kBtCap;
        if (bt_smem)
        {
            for (int i = tid; i < nbt; i += kThreads)
            {
                s_bt[i] = __ldg(p.block_table + req * p.bt_stride + i);
            }
        }
        load_q(s, p, tok, tid);
        asm volatile("griddepcontrol.wait;" ::: "memory");
        asm volatile("griddepcontrol.launch_dependents;");
    }
    int blk = static_cast<int>(rank) < real_topk ? p.topk[tok * p.t_tok + rank * p.t_k] : -1;
    bool const active = blk >= 0;
    if constexpr (PREWAIT)
    {
        __syncthreads(); // s_bt and the q rows are complete
    }
    else
    {
        load_q(s, p, tok, tid);
    }

    // Lane holds heads 2 * qd + {0, 1}; acc[mt]: dims warp * 32 + mt * 16 + g + {0, 8}.
    float m_row[2] = {kNegInf, kNegInf};
    float l_row[2] = {0.F, 0.F};
    float acc[2][4];
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i)
            acc[j][i] = 0.F;

    // K rows n = warp * 32 + j * 8 + g stay packed in registers. As the MMA A operand, m-tile mt takes rows
    // j = 2 mt (row g) and 2 mt + 1 (row g + 8); k-step ks takes packed bytes 8 ks + qd and 8 ks + 4 + qd.
    uint32_t kw[4][16];
    uint2 ksc[4];
    if (active)
    {
        int64_t const page = bt_smem ? s_bt[blk] : p.block_table[req * p.bt_stride + blk];
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            int const n = warp * 32 + j * 8 + g;
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
        int const q4 = qd * 8;
        // S^T = K q^T for this warp's 32 tokens: rows (tokens) g / g + 8 of each m-tile, columns (heads) 2 qd + {0, 1}.
        float sacc[2][4];
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i)
                sacc[j][i] = 0.F;
#pragma unroll
        for (int ks2 = 0; ks2 < kDim / 16; ks2 += 2)
        {
            uint32_t bq[4];
            ldsm_x4(bq, smem_u32(s.qs + (lane & 7) * kPitch + ks2 * 16 + (lane >> 3) * 8));
#pragma unroll
            for (int u = 0; u < 2; ++u)
            {
                int const ks = ks2 + u;
#pragma unroll
                for (int mt = 0; mt < 2; ++mt)
                {
                    uint32_t a[4];
#pragma unroll
                    for (int hi = 0; hi < 2; ++hi)
                    {
                        int const j = 2 * mt + hi;
                        uint32_t const scw = ks < 4 ? ksc[j].x : ksc[j].y;
                        uint8_t const sc = static_cast<uint8_t>(scw >> (8 * (ks & 3)));
                        uint32_t const sh = e4m3x2_to_f16x2(static_cast<uint16_t>(sc | (sc << 8)));
                        __half2 const scale = *reinterpret_cast<__half2 const*>(&sh);
                        uint32_t const r0 = e2m1x2_to_f16x2((kw[j][2 * ks] >> q4) & 0xFF);
                        uint32_t const r1 = e2m1x2_to_f16x2((kw[j][2 * ks + 1] >> q4) & 0xFF);
                        __half2 const h0 = __hmul2(*reinterpret_cast<__half2 const*>(&r0), scale);
                        __half2 const h1 = __hmul2(*reinterpret_cast<__half2 const*>(&r1), scale);
                        a[hi] = *reinterpret_cast<uint32_t const*>(&h0);
                        a[2 + hi] = *reinterpret_cast<uint32_t const*>(&h1);
                    }
                    mma(sacc[mt], a, bq[2 * u], bq[2 * u + 1]);
                }
            }
        }
        float mloc[2] = {kNegInf, kNegInf};
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                int const t = warp * 32 + mt * 16 + g + 8 * (i >> 1);
                float const v = blk * kPage + t < kv_len ? 0.F + sacc[mt][i] * scale_log2 : kNegInf;
                sacc[mt][i] = v;
                mloc[i & 1] = fmaxf(mloc[i & 1], v);
            }
#pragma unroll
        for (int e = 0; e < 2; ++e)
        {
            mloc[e] = fmaxf(mloc[e], __shfl_xor_sync(0xffffffff, mloc[e], 4));
            mloc[e] = fmaxf(mloc[e], __shfl_xor_sync(0xffffffff, mloc[e], 8));
            mloc[e] = fmaxf(mloc[e], __shfl_xor_sync(0xffffffff, mloc[e], 16));
        }
        if (g == 0)
        {
            s.red[warp][2 * qd] = mloc[0];
            s.red[warp][2 * qd + 1] = mloc[1];
        }
        __syncthreads();
        if (tid < kHeads8)
            s.rowm[tid] = fmaxf(fmaxf(s.red[0][tid], s.red[1][tid]), fmaxf(s.red[2][tid], s.red[3][tid]));
        __syncthreads();
        m_row[0] = s.rowm[2 * qd];
        m_row[1] = s.rowm[2 * qd + 1];
        float lloc[2] = {0.F, 0.F};
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                float const pv = ex2(sacc[mt][i] - m_row[i & 1]);
                lloc[i & 1] += pv;
                int const t = warp * 32 + mt * 16 + g + 8 * (i >> 1);
                s.ps[(2 * qd + (i & 1)) * kPitch + t] = __float2half_rn(pv);
            }
#pragma unroll
        for (int e = 0; e < 2; ++e)
        {
            lloc[e] += __shfl_xor_sync(0xffffffff, lloc[e], 4);
            lloc[e] += __shfl_xor_sync(0xffffffff, lloc[e], 8);
            lloc[e] += __shfl_xor_sync(0xffffffff, lloc[e], 16);
        }
        __syncthreads(); // rowm reads done before red is reused; ps complete
        if (g == 0)
        {
            s.red[warp][2 * qd] = lloc[0];
            s.red[warp][2 * qd + 1] = lloc[1];
        }
        // O^T[warp's 32 dims, 8 heads] = V^T P^T over all 128 tokens.
#pragma unroll
        for (int kk = 0; kk < kPage; kk += 32)
        {
            uint32_t b[4];
            ldsm_x4(b, smem_u32(s.ps + (lane & 7) * kPitch + kk + (lane >> 3) * 8));
#pragma unroll
            for (int u = 0; u < 2; ++u)
            {
#pragma unroll
                for (int mt = 0; mt < 2; ++mt)
                {
                    uint32_t a[4];
                    int const t = kk + u * 16 + (lane & 7) + ((lane >> 4) & 1) * 8;
                    int const d = warp * 32 + mt * 16 + ((lane >> 3) & 1) * 8;
                    ldsm_x4_t(a, smem_u32(s.vs + t * kPitch + d));
                    mma(acc[mt], a, b[2 * u], b[2 * u + 1]);
                }
            }
        }
        __syncthreads();
        if (tid < kHeads8)
            s.rowl[tid] = (s.red[0][tid] + s.red[1][tid]) + (s.red[2][tid] + s.red[3][tid]);
        __syncthreads();
        l_row[0] = s.rowl[2 * qd];
        l_row[1] = s.rowl[2 * qd + 1];
    }

    // Normalize; push (head, dim) to CTA 2 * head + dim / 64, slot recv[rank][dim % 64], plus each head's LSE (log2)
    // to both of its CTAs. Posted DSMEM stores, made visible by the cluster barrier.
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    {
        float const vgs = __ldg(p.vgs);
        float scale[2], lse[2];
#pragma unroll
        for (int e = 0; e < 2; ++e)
        {
            if constexpr (LINEAR)
            {
                scale[e] = l_row[e] > 0.F ? 1.0F / l_row[e] : 0.F;
                lse[e] = l_row[e] > 0.F ? m_row[e] + lg2(l_row[e]) : kNegInf;
            }
            else
            {
                float const lse_i = active ? m_row[e] + lg2(ex2(kNegInf - m_row[e]) + l_row[e]) : kNegInf;
                scale[e] = lse_i > kNegInf ? ex2(m_row[e] - lse_i) : 0.F;
                lse[e] = lse_i;
            }
            scale[e] = scale[e] * vgs;
        }
        uint32_t dst[2];
#pragma unroll
        for (int e = 0; e < 2; ++e)
            dst[e] = map_cluster(&s.recv[rank][0], (2 * qd + e) * 2 + (warp >> 1));
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                int const dl = (warp & 1) * 32 + mt * 16 + g + 8 * (i >> 1);
                st_cluster_f32(dst[i & 1] + dl * 4, acc[mt][i] * scale[i & 1]);
            }
        if (warp == 0 && g == 0)
        {
#pragma unroll
            for (int e = 0; e < 2; ++e)
#pragma unroll
                for (int hf = 0; hf < 2; ++hf)
                    st_cluster_f32(map_cluster(&s.recv_lse[rank], (2 * qd + e) * 2 + hf), lse[e]);
        }
    }
    cluster_sync();

    // CTA `rank` merges head rank / 2, dims (rank % 2) * 64 + dl; thread = (chunk half, dl).
    int const h = static_cast<int>(rank) >> 1;
    int const dh = static_cast<int>(rank) & 1;
    int const dl = tid & (kHalfDim - 1);
    int const hc = tid >> 6;
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
    for (int c = 0; c < kTopk / 2; ++c)
    {
        int const cc = hc * (kTopk / 2) + c;
        o += s.recv[cc][dl] * __fdiv_rn(lse_c[cc], denom);
    }
    if (hc)
        s.part[dl] = o;
    __syncthreads();
    if (!hc)
    {
        o += s.part[dl];
        int const col = h * kDim + dh * kHalfDim + dl;
        __nv_bfloat16 const ob = __float2bfloat16_rn(o);
        p.out[tok * p.out_stride + col] = ob;
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
            p.q8[static_cast<int64_t>(tok) * (p.num_sf_k_tiles * 128) + col] = static_cast<uint8_t>(q2 & 0xFF);
            if (lane == 0)
            {
                int const k_sf = col / 32;
                int64_t const off = static_cast<int64_t>(tok / 128) * p.num_sf_k_tiles * 512 + (k_sf / 4) * 512
                    + (tok % 32) * 16 + ((tok % 128) / 32) * 4 + k_sf % 4;
                p.sf[off] = static_cast<uint8_t>(e);
            }
        }
    }
#endif // defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

template <bool LINEAR, bool MXFP8, bool PREWAIT>
cudaError_t configure()
{
    auto kernel = m3_tp8_lc8_attn_kernel<LINEAR, MXFP8, PREWAIT>;
    cudaError_t err = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smemBytes<PREWAIT>()));
    if (err != cudaSuccess)
    {
        return err;
    }
    return cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
}

template <bool LINEAR, bool MXFP8, bool PREWAIT>
cudaError_t launch(Params const& prm, int num_tokens, bool use_pdl, cudaStream_t stream)
{
    // Kernel attributes are set once per process, on the first launch (outside CUDA-graph capture: warmup runs
    // every variant eagerly first).
    static cudaError_t const configured = configure<LINEAR, MXFP8, PREWAIT>();
    if (configured != cudaSuccess)
    {
        return configured;
    }
    auto kernel = m3_tp8_lc8_attn_kernel<LINEAR, MXFP8, PREWAIT>;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(num_tokens * kTopk);
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = smemBytes<PREWAIT>();
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

template <bool LINEAR, bool MXFP8>
cudaError_t launchPrewait(Params const& prm, int num_tokens, bool prewait, bool use_pdl, cudaStream_t stream)
{
    return prewait ? launch<LINEAR, MXFP8, true>(prm, num_tokens, use_pdl, stream)
                   : launch<LINEAR, MXFP8, false>(prm, num_tokens, use_pdl, stream);
}

} // namespace

cudaError_t invokeMinimaxM3Tp8SparseDecodeLc8(MinimaxM3Tp8SparseDecodeLc8Params const& params, int32_t numTokens,
    bool linearSoftmax, bool mxfp8, bool prewait, bool usePdl, cudaStream_t stream)
{
    if (linearSoftmax)
    {
        return mxfp8 ? launchPrewait<true, true>(params, numTokens, prewait, usePdl, stream)
                     : launchPrewait<true, false>(params, numTokens, prewait, usePdl, stream);
    }
    return mxfp8 ? launchPrewait<false, true>(params, numTokens, prewait, usePdl, stream)
                 : launchPrewait<false, false>(params, numTokens, prewait, usePdl, stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
