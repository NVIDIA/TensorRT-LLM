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

// MiniMax-M3 long-context sparse decode, SM107 TP2 geometry, LC2-split: LC2 (sparseDecodeLc2.cu) without its thread-
// block cluster. Grid 1 (tokens * 16, kv heads): CTA `r` attends the token's `r`-th selected block exactly as LC2 does
// and stores its normalized partial rows and LSE (LC2's pushed values) to a workspace, head-major. Grid 2 (tokens, kv
// heads): one warp per head combines the 16 partials with LC2's merge weights in slot order and writes the BF16 row
// and, optionally, LC2's MXFP8 copy and swizzled UE8M0 scales. Bit-identical to LC2.
//
// Per-block math follows the Triton producer: NVFP4 K/V dequantized to FP16 (exact), FP8 q widened to FP16, QK and
// PV on FP16 tensor cores with FP32 accumulation, log2-domain softmax with either the linear denominator or the
// log-domain recurrence.

#include "tensorrt_llm/kernels/minimaxM3Sm107Tp2/sparseDecodeLc2Split.h"

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
constexpr int kBtCap = 4096;     // block-table entries staged in smem by PREWAIT (512K tokens)

using Params = MinimaxM3Sm107Tp2SparseDecodeLcParams;

struct Smem
{
    __half qs[kHeads * kPitch];
    __half vs[kPage * kPitch];
    __half ps[kHeads * kPitch];
    float red[4][kHeads];
    float rowm[kHeads];
    float rowl[kHeads];
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

template <bool PREWAIT>
constexpr size_t smemBytes()
{
    return PREWAIT ? sizeof(Smem) + kBtCap * sizeof(int) : sizeof(Smem);
}

template <bool LINEAR, bool MXFP8, bool EARLY, bool PREWAIT>
__global__ void __launch_bounds__(kThreads)
    m3_lcs_attn_kernel(__grid_constant__ Params const p, float* const s_part, float* const s_lse)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    extern __shared__ __align__(128) uint8_t smem_raw[];
    Smem& s = *reinterpret_cast<Smem*>(smem_raw);
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    int const tok = blockIdx.x / kTopk;
    int const kvh = blockIdx.y;
    uint32_t rank;
    rank = blockIdx.x % kTopk; // LC2-split: no cluster; slot = the CTA's position in its 16

    // Pre-wait read contract: everything that does not depend on the top-16 table is read before the grid dependency.
    // - seq_lens: written inside the step's graph by msa_kv_update (the KV-length correction). That kernel is a plain
    //   launch, not a programmatic dependent, so it has completed before any later kernel of the step starts.
    // - the query rows: written by the producer, two grids back. This holds only while every kernel between the
    //   producer and this grid (the long-row select) triggers its dependents after its own griddepcontrol.wait, so
    //   this grid cannot start before the producer completed. A kernel inserted into that chain, or a select that
    //   triggers before its wait, breaks it: these reads then move after the griddepcontrol.wait below.
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
        uint4 const w = *reinterpret_cast<uint4 const*>(p.q + tok * p.q_tok + (kvh * kHeads + h) * p.q_head + c);
        uint32_t const words[4] = {w.x, w.y, w.z, w.w};
        uint32_t* d = reinterpret_cast<uint32_t*>(s.qs + h * kPitch + c);
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            d[2 * i] = e4m3x2_to_f16x2(static_cast<uint16_t>(words[i] & 0xFFFF));
            d[2 * i + 1] = e4m3x2_to_f16x2(static_cast<uint16_t>(words[i] >> 16));
        }
    }

    // PREWAIT: the request's block-table row (written before the graph runs) is staged before the grid dependency.
    int* const s_bt = reinterpret_cast<int*>(smem_raw + sizeof(Smem));
    bool bt_smem = false;
    if constexpr (PREWAIT)
    {
        int const nbt = (max(seq, 0) + kPage - 1) / kPage;
        bt_smem = nbt <= kBtCap;
        if (bt_smem)
        {
            for (int i = tid; i < nbt; i += kThreads)
                s_bt[i] = __ldg(p.block_table + req * p.bt_stride + i);
        }
    }
    // N1: the block this (KV head, row, slot) selected at the layer's previous decode step recurs ~2/3 of the time;
    // pull its K/V page and scales into L2 while the select runs. The ids were written by this kernel one step back
    // (complete before the graph's next replay) and the block-table row before the graph runs. Read and written by
    // the same thread, so the record below cannot overtake this read. Out-of-range ids (stale rows) are skipped.
    bool const n1 = p.n1_ids != nullptr && tok < p.n1_rows;
    int* const n1_slot = n1 ? p.n1_ids + kvh * p.n1_h + tok * kTopk + rank : nullptr;
    if (n1 && tid == 64)
    {
        int const nb = *n1_slot;
        if (nb >= 0 && nb < nblocks)
        {
            int64_t const pg = __ldg(p.block_table + req * p.bt_stride + nb);
            if (pg >= 0 && pg < p.n1_pages)
            {
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p.k + pg * p.k_blk + kvh * p.k_h),
                             "r"(kPage * 64)
                             : "memory");
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p.v + pg * p.v_blk + kvh * p.v_h),
                             "r"(kPage * 64)
                             : "memory");
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p.ksc + pg * p.ksc_blk + kvh * p.ksc_h),
                             "r"(kPage * 8)
                             : "memory");
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p.vsc + pg * p.vsc_blk + kvh * p.vsc_h),
                             "r"(kPage * 8)
                             : "memory");
            }
        }
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if constexpr (EARLY)
    {
        asm volatile("griddepcontrol.launch_dependents;");
    }
    int blk = static_cast<int>(rank) < real_topk ? p.topk[kvh * p.t_h + tok * p.t_tok + rank * p.t_k] : -1;
    if (n1 && tid == 64)
    {
        *n1_slot = blk;
    }
    bool const active = blk >= 0;
    if constexpr (PREWAIT)
    {
        __syncthreads(); // s_bt complete
    }
    int64_t const page_pw = !active ? 0 : (bt_smem ? s_bt[blk] : p.block_table[req * p.bt_stride + blk]);

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
            uint4 const* src = reinterpret_cast<uint4 const*>(p.k + page * p.k_blk + kvh * p.k_h + n * 64);
#pragma unroll
            for (int c = 0; c < 4; ++c)
            {
                uint4 const w = valid ? __ldg(src + c) : make_uint4(0, 0, 0, 0);
                kw[j][4 * c] = w.x;
                kw[j][4 * c + 1] = w.y;
                kw[j][4 * c + 2] = w.z;
                kw[j][4 * c + 3] = w.w;
            }
            ksc[j] = valid ? __ldg(reinterpret_cast<uint2 const*>(p.ksc + page * p.ksc_blk + kvh * p.ksc_h + n * 8))
                           : make_uint2(0, 0);
        }
        bool const valid = blk * kPage + tid < kv_len;
        uint8_t vsc[8];
        if (valid)
        {
            uint8_t const* vb = p.vsc + page * p.vsc_blk + kvh * p.vsc_h + (tid >> 2) * 32 + (tid & 3);
#pragma unroll
            for (int c = 0; c < 8; ++c)
                vsc[c] = __ldg(vb + 4 * c);
        }
        dequant_row(s.vs + tid * kPitch, p.v + page * p.v_blk + kvh * p.v_h + tid * 64, vsc, valid);
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

    // LC2-split: store this CTA's normalized partial rows and LSE (LC2's pushed values, the same expressions) to the
    // workspace, head-major: part[row][kvh][head][slot][128], lse[row][kvh][head][slot]. The merge grid combines them.
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
            int64_t const hs = ((static_cast<int64_t>(tok) * p.num_kv_heads + kvh) * kHeads + row) * kTopk + rank;
            float* const dst = s_part + hs * kDim;
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                int const d = warp * 32 + j * 8 + 2 * (lane & 3);
                *reinterpret_cast<float2*>(dst + d) = make_float2(acc[j][2 * r] * scale, acc[j][2 * r + 1] * scale);
            }
            if (warp == 0 && (lane & 3) == 0)
                s_lse[hs] = lse;
        }
    }
    if constexpr (!EARLY)
    {
        asm volatile("griddepcontrol.launch_dependents;");
    }
#endif // defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

template <bool LINEAR, bool MXFP8, bool EARLY, bool PREWAIT>
cudaError_t configure()
{
    auto kernel = m3_lcs_attn_kernel<LINEAR, MXFP8, EARLY, PREWAIT>;
    cudaError_t err = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smemBytes<PREWAIT>()));
    if (err != cudaSuccess)
    {
        return err;
    }
    return cudaSuccess;
}

template <bool LINEAR, bool MXFP8, bool EARLY, bool PREWAIT>
cudaError_t launch(Params const& prm, int num_tokens, bool use_pdl, float* part, float* lse, cudaStream_t stream)
{
    // Kernel attributes are set once per process, on the first launch (outside CUDA-graph capture: warmup runs
    // every variant eagerly first).
    static cudaError_t const configured = configure<LINEAR, MXFP8, EARLY, PREWAIT>();
    if (configured != cudaSuccess)
    {
        return configured;
    }
    auto kernel = m3_lcs_attn_kernel<LINEAR, MXFP8, EARLY, PREWAIT>;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(num_tokens * kTopk, prm.num_kv_heads);
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
    cfg.attrs = attrs + 1; // no cluster
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, kernel, prm, part, lse);
}

template <bool MXFP8>
__global__ void __launch_bounds__(kHeads * 32)
    m3_lcs_merge_kernel(__grid_constant__ Params const p, float const* const s_part, float const* const s_lse)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // One CTA per (row, KV head), one warp per head; lane l owns dims 4l .. 4l + 3. LC2's merge per head: the 16 slot
    // weights (exp2 of each LSE minus the largest, divided by their sum formed in slot order), then the weighted sum of
    // the 16 partials in slot order, the BF16 row and the MXFP8 hand-off, value for value.
    int const tok = blockIdx.x;
    int const kvh = blockIdx.y;
    int const h = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int64_t const hs = ((static_cast<int64_t>(tok) * p.num_kv_heads + kvh) * kHeads + h) * kTopk;
    float const lse_c = lane < kTopk ? s_lse[hs + lane] : kNegInf;
    float lse_max = kNegInf;
#pragma unroll
    for (int c = 0; c < kTopk; ++c)
        lse_max = fmaxf(lse_max, __shfl_sync(0xffffffff, lse_c, c));
    lse_max = lse_max == kNegInf ? 0.F : lse_max;
    float const e_c = ex2(lse_c - lse_max);
    float denom = 0.F;
#pragma unroll
    for (int c = 0; c < kTopk; ++c)
        denom += __shfl_sync(0xffffffff, e_c, c);
    denom = denom > 0.F ? denom : 1.F;
    float const w_c = __fdiv_rn(e_c, denom);
    float o[4] = {0.F, 0.F, 0.F, 0.F};
    float4 const* const src = reinterpret_cast<float4 const*>(s_part + hs * kDim) + lane;
#pragma unroll
    for (int c = 0; c < kTopk; ++c)
    {
        float const w = __shfl_sync(0xffffffff, w_c, c);
        float4 const v = src[c * (kDim / 4)];
        o[0] += v.x * w;
        o[1] += v.y * w;
        o[2] += v.z * w;
        o[3] += v.w * w;
    }
    __nv_bfloat16 ob[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        ob[i] = __float2bfloat16_rn(o[i]);
    __nv_bfloat16* const orow = p.out + tok * p.out_stride + (kvh * kHeads + h) * kDim + 4 * lane;
    *reinterpret_cast<uint2*>(orow) = *reinterpret_cast<uint2 const*>(ob);
    if constexpr (MXFP8)
    {
        // LC2's per-32-dim group: amax over the group (8 lanes here), the UE8M0 exponent and the E4M3 values.
        float amax = fabsf(__bfloat162float(ob[0]));
#pragma unroll
        for (int i = 1; i < 4; ++i)
            amax = fmaxf(amax, fabsf(__bfloat162float(ob[i])));
#pragma unroll
        for (int off = 4; off > 0; off >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
        float rcp;
        asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(rcp) : "f"(448.0F));
        float const sfv = amax * rcp;
        int const bits = __float_as_int(sfv);
        int e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0);
        e = min(e, 254);
        float const qs = amax != 0.F ? __int_as_float((254 - e) << 23) : 0.F;
        uint8_t q8v[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            float const xq = __bfloat162float(ob[i]) * qs;
            uint16_t q2;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(q2) : "f"(0.F), "f"(xq));
            q8v[i] = static_cast<uint8_t>(q2 & 0xFF);
        }
        *reinterpret_cast<uint32_t*>(
            p.q8 + static_cast<int64_t>(tok) * (p.num_sf_k_tiles * 128) + (kvh * kHeads + h) * kDim + 4 * lane)
            = *reinterpret_cast<uint32_t const*>(q8v);
        if ((lane & 7) == 0)
        {
            int const warp = lane >> 3; // LC2's warp index = the 32-dim group
            int const k_sf = (kvh * kHeads + h) * 4 + warp;
            int64_t const off = static_cast<int64_t>(tok / 128) * p.num_sf_k_tiles * 512 + (k_sf / 4) * 512
                + (tok % 32) * 16 + ((tok % 128) / 32) * 4 + k_sf % 4;
            p.sf[off] = static_cast<uint8_t>(e);
        }
    }
#endif
}

template <bool MXFP8>
cudaError_t launchMerge(
    Params const& prm, int num_tokens, bool use_pdl, float const* part, float const* lse, cudaStream_t stream)
{
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(num_tokens, prm.num_kv_heads);
    cfg.blockDim = dim3(kHeads * 32);
    cfg.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = use_pdl ? 1 : 0;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, m3_lcs_merge_kernel<MXFP8>, prm, part, lse);
}

} // namespace

size_t minimaxM3Sm107Tp2SparseDecodeLcSplitWorkspaceBytes(int32_t maxTokens, int32_t numKvHeads)
{
    size_t const hs = static_cast<size_t>(maxTokens) * numKvHeads * kHeads * kTopk;
    return hs * kDim * sizeof(float) + hs * sizeof(float);
}

cudaError_t invokeMinimaxM3Sm107Tp2SparseDecodeLcSplit(MinimaxM3Sm107Tp2SparseDecodeLcParams const& params,
    int32_t numTokens, bool linearSoftmax, bool mxfp8, bool prewait, bool usePdl, void* workspace,
    size_t workspaceBytes, cudaStream_t stream)
{
    size_t const hs = static_cast<size_t>(numTokens) * params.num_kv_heads * kHeads * kTopk;
    if (workspace == nullptr || workspaceBytes < hs * (kDim + 1) * sizeof(float))
    {
        return cudaErrorInvalidValue;
    }
    float* const part = static_cast<float*>(workspace);
    float* const lse = part + hs * kDim;
    // The compute grid triggers its dependents at its end (EARLY off), so no merge CTA waits on a slot it needs.
    cudaError_t err;
    if (prewait)
    {
        err = linearSoftmax ? (mxfp8 ? launch<true, true, false, true>(params, numTokens, usePdl, part, lse, stream)
                                     : launch<true, false, false, true>(params, numTokens, usePdl, part, lse, stream))
                            : (mxfp8 ? launch<false, true, false, true>(params, numTokens, usePdl, part, lse, stream)
                                     : launch<false, false, false, true>(params, numTokens, usePdl, part, lse, stream));
    }
    else
    {
        err = linearSoftmax
            ? (mxfp8 ? launch<true, true, false, false>(params, numTokens, usePdl, part, lse, stream)
                     : launch<true, false, false, false>(params, numTokens, usePdl, part, lse, stream))
            : (mxfp8 ? launch<false, true, false, false>(params, numTokens, usePdl, part, lse, stream)
                     : launch<false, false, false, false>(params, numTokens, usePdl, part, lse, stream));
    }
    if (err != cudaSuccess)
    {
        return err;
    }
    return mxfp8 ? launchMerge<true>(params, numTokens, usePdl, part, lse, stream)
                 : launchMerge<false>(params, numTokens, usePdl, part, lse, stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
