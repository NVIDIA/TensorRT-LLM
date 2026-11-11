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
// One-shot all-reduce of a decode step's partial rows fused with the mHC boundary that follows it (GLM-5.3-Flash at
// tp4: hidden 4096, hc_mult 4, at most 8 rows), as one cluster of 16 CTAs.
//
// All-reduce: TensorRT-LLM's Lamport one-shot protocol on the all-reduce workspace (the LamportComm layout: three
// rotating receive buffers per rank, the flag, the counter and the clear size), so calls interleave with the
// library's one-shot all-reduce on the same workspace. Each CTA pushes its 256 columns of the rows to every rank
// (-0.0 marks a word not yet arrived, so a -0.0 value is sent as +0.0), clears its share of the buffer of the call
// before last, polls until the four ranks' slices are there, sums them in rank order in FP32 and rounds to BF16,
// like the library's one-shot kernel.
// mHC boundary (post-mapping of the previous block, pre-mapping of the next one): every CTA computes the new streams
// of its columns, its partial mixer products (TF32 MMA, the streams are BF16 and exact in TF32) and Gram entries;
// the 16 CTAs exchange their sums through distributed shared memory and each adds them in CTA order
// (deterministic). Each CTA then derives the coefficients (the aux warp runs the Sinkhorn normalization) and writes
// its columns of the streams and of the layer input, RMS-normalized through the Gram matrix (no second reduction).
// The CTA's 96 KB slice of the mixer weights is loaded before the grid-dependency wait.
#include "glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/csrc/mhcArBoundary.h"

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

namespace cg = cooperative_groups;

constexpr int kN = 4, kMix = (2 + kN) * kN, kGram = 10, kAcc = kMix + kGram;              // 24 + 10 sums per row
constexpr int kH = 4096, kNH = kN * kH, kRanks = 4;
constexpr int kCtas = 16, kCols = kH / kCtas, kWarps = kCols / 32, kThreads = kCols + 32; // + aux warp
constexpr int kMaxRows = kMhcArBoundaryMaxRows;
constexpr int kChunks = kCols / 8;                        // 16-byte chunks of a row slice
constexpr int kSlice = kCols + 8, kTok = kN * kSlice + 8; // padded (bank-conflict free MMA fragment loads)
constexpr int kFnRow = kN * kCols + 4;                    // fn rows (o, j) at o * kFnRow + j * kCols (floats)
constexpr int kFnBytes = kMix * kFnRow * 4;
constexpr int kResBytes = kMaxRows * kTok * 2;            // this CTA's columns of the residual streams
constexpr int kSmem = kResBytes + kWarps * kMaxRows * kAcc * 4 + kMaxRows * kCols * 2 + kMaxRows * 20 * 4 + kFnBytes;

struct Params
{
    void** workspace;             // TensorRT-LLM all-reduce workspace
    int rank;
    __nv_bfloat16 const* partial; // [M, H] this rank's partial block output
    __nv_bfloat16 const* res;     // [M, N, H]
    float const* post_prev;       // [M, N]
    float const* comb_prev;       // [M, N, N]
    float const* fn;              // [24, N * H]
    float const* scale;           // [3]
    float const* base;            // [24]
    __nv_bfloat16 const* w;       // [H]
    __nv_bfloat16* res_out;       // [M, N, H]
    float* post_out;              // [M, N]
    float* comb_out;              // [M, N, N]
    __nv_bfloat16* li_out;        // [M, H]
    int M, sk_iters;
    float norm_eps, hc_eps, sk_eps, post_mult, li_eps;
};

__device__ __forceinline__ constexpr int gram_idx(int j, int k)
{
    return j * kN - j * (j - 1) / 2 + (k - j);
}

__device__ __forceinline__ bool is_neg_zero(uint32_t v)
{
    return v == 0x80000000u;
}

__device__ __forceinline__ uint4 ld_volatile(uint4 const* p)
{
    uint4 v;
    asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ uint32_t to_tf32(float v)
{
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(v));
    return r;
}

// D += A (16 x 8, row) * B (8 x 8, col), TF32 inputs, FP32 accumulation
__device__ __forceinline__ void mma_tf32(
    float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t f32_bits(float v)
{
    return __float_as_uint(v);
}

__device__ __forceinline__ uint32_t smem_u32(void const* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__global__ void __cluster_dims__(kCtas, 1, 1) __launch_bounds__(kThreads, 1) mhc_ar_kernel(Params p)
{
    extern __shared__ __align__(128) uint8_t smem[];
    __nv_bfloat16* s_res = reinterpret_cast<__nv_bfloat16*>(smem); // [rows][4][256], then the new streams
    float* s_warp = reinterpret_cast<float*>(smem + kResBytes);    // [warps][rows][34]
    __nv_bfloat16* s_x
        = reinterpret_cast<__nv_bfloat16*>(smem + kResBytes + kWarps * kMaxRows * kAcc * 4); // [rows][256]
    float* s_pc = reinterpret_cast<float*>(smem + kResBytes + kWarps * kMaxRows * kAcc * 4 + kMaxRows * kCols * 2);
    // s_pc: post_prev [rows][4], then comb_prev [rows][16]
    float* s_fn = s_pc + kMaxRows * 20;
    __shared__ float s_all[kCtas * kMaxRows * kAcc]; // every CTA's sums, slot c written by CTA c (DSMEM stores)
    __shared__ float s_tot[kMaxRows * kAcc];         // the 16 CTAs' totals
    cg::cluster_group cluster = cg::this_cluster();
    __shared__ float s_sk[2][kN + kN * kN];          // this CTA's rows' post and comb (aux warp), stored after the wait
    __shared__ __align__(8) uint64_t in_bar, fn_bar;

    int const cta = blockIdx.x, t = threadIdx.x, warp = t / 32, lane = t % 32;
    int const c0 = cta * kCols, h = c0 + t, M = p.M;
    bool const col_thread = warp < kWarps, aux = warp == kWarps;
    __shared__ float s_coef[kMaxRows][kN]; // layer_input = sum_j coef[m][j] r_j * w

    // before the PDL wait: weights (this column's 96 mixer weights, kept in registers for every row, and its norm
    // weight) and the all-reduce workspace's pointer table and buffer size, which never change
    // mixer K slice of this warp: stream w / 2, columns [128 (w % 2), + 128), 16 k-steps of 8 (B from s_fn)
    int const g = lane >> 2, tig = lane & 3;
    int const mj = warp >> 1, mcol = (warp & 1) * 128;
    float const wh = col_thread ? __bfloat162float(p.w[h]) : 0.f;
    void** ws = p.workspace;
    int* counter_ptr = reinterpret_cast<int*>(ws[kRanks * 3]);
    int* flag_ptr = counter_ptr + 2;
    long long* clear_ptr = reinterpret_cast<long long*>(ws[kRanks * 3 + 1]);
    long long const comm_size = clear_ptr[1];
    uint8_t* bufs[kRanks];
#pragma unroll
    for (int r = 0; r < kRanks; r++)
        bufs[r] = reinterpret_cast<uint8_t*>(ws[2 * kRanks + r]);
    if (t == kCols)
    {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_u32(&in_bar)));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_u32(&fn_bar)));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(&fn_bar)),
                     "r"(kMix * kN * kCols * 4)
                     : "memory");
    }
    __syncthreads();
    // the fn slice (weights) before the wait: 96 rows of 1 KB
    if (aux)
    {
        for (int i = lane; i < kMix * kN; i += 32)
        {
            int const o = i / kN, j = i % kN;
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                    smem_u32(s_fn + o * kFnRow + j * kCols)),
                "l"(p.fn + (size_t) o * kNH + j * kH + c0), "r"(kCols * 4), "r"(smem_u32(&fn_bar))
                : "memory");
        }
    }

    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); // this CTA has started
    // the producer of the partial rows (and the previous boundary, the last user of the Lamport state) has completed
    cudaGridDependencySynchronize();

    // aux warp: stage this CTA's columns of the residual streams and the rows' post / comb (overlaps the all-reduce)
    if (aux)
    {
        if (lane == 0)
        {
            const uint32_t bytes = M * kN * kCols * 2 + M * 20 * 4;
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(&in_bar)), "r"(bytes)
                         : "memory");
        }
        __syncwarp();
        for (int i = lane; i < M * kN + 2; i += 32)
        {
            bool const res_copy = i < M * kN;
            const uint32_t dst = res_copy ? smem_u32(s_res + (i / kN) * kTok + (i % kN) * kSlice)
                                          : smem_u32(s_pc + (i == M * kN ? 0 : kMaxRows * 4));
            void const* src = res_copy
                ? static_cast<void const*>(p.res + (size_t) i * kH + c0)
                : (i == M * kN ? static_cast<void const*>(p.post_prev) : static_cast<void const*>(p.comb_prev));
            const uint32_t bytes = res_copy ? kCols * 2 : (i == M * kN ? M * 16 : M * 64);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
                "l"(src), "r"(bytes), "r"(smem_u32(&in_bar))
                : "memory");
        }
    }
    // Lamport state (TensorRT-LLM LamportComm): buffers rotate by flag % 3, the call before last's is cleared.
    // The clear size is the bytes that call wrote, whatever kernel or dtype it was; it is cleared in whole
    // 16-byte stores, rounded up.
    int const flag = *flag_ptr;
    long long const clear_size = clear_ptr[0];
    if (t == 0)
        atomicAdd(counter_ptr, 1);
    uint8_t* data_bufs[kRanks];
#pragma unroll
    for (int r = 0; r < kRanks; r++)
        data_bufs[r] = bufs[r] + (flag % 3) * comm_size;
    uint4* clear_buf = reinterpret_cast<uint4*>(bufs[p.rank] + ((flag + 2) % 3) * comm_size);
    int const tot = M * kH / 8; // 16-byte accesses in one rank's message

    // push this CTA's columns of the partial rows to every rank (-0.0 words sent as +0.0)
    for (int i = t; i < M * kChunks; i += kThreads)
    {
        int const m = i / kChunks, q = i % kChunks, idx = (m * kH + c0) / 8 + q;
        uint4 v = reinterpret_cast<uint4 const*>(p.partial)[idx];
        v.x = is_neg_zero(v.x) ? 0u : v.x;
        v.y = is_neg_zero(v.y) ? 0u : v.y;
        v.z = is_neg_zero(v.z) ? 0u : v.z;
        v.w = is_neg_zero(v.w) ? 0u : v.w;
#pragma unroll
        for (int r = 0; r < kRanks; r++)
            reinterpret_cast<uint4*>(data_bufs[r])[p.rank * tot + idx] = v;
    }
    // clear this CTA's share of the call-before-last's buffer
    const uint4 sentinel = make_uint4(0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u);
    for (long long i = (long long) cta * kThreads + t; i < (clear_size + 15) / 16; i += (long long) kCtas * kThreads)
        clear_buf[i] = sentinel;
    // poll and sum the four ranks' slices in rank order (FP32), rounded to BF16 like the library's all-reduce
    for (int i = t; i < M * kChunks; i += kThreads)
    {
        int const m = i / kChunks, q = i % kChunks, idx = (m * kH + c0) / 8 + q;
        uint4 v[kRanks];
        bool done;
        do
        {
            done = true;
#pragma unroll
            for (int r = 0; r < kRanks; r++)
            {
                v[r] = ld_volatile(reinterpret_cast<uint4 const*>(data_bufs[p.rank]) + r * tot + idx);
                done &= !(is_neg_zero(v[r].x) || is_neg_zero(v[r].y) || is_neg_zero(v[r].z) || is_neg_zero(v[r].w));
            }
        } while (!done);
        float acc[8];
#pragma unroll
        for (int e = 0; e < 8; e++)
            acc[e] = 0.f;
#pragma unroll
        for (int r = 0; r < kRanks; r++)
        {
            __nv_bfloat16 const* bv = reinterpret_cast<__nv_bfloat16 const*>(&v[r]);
#pragma unroll
            for (int e = 0; e < 8; e++)
                acc[e] += __bfloat162float(bv[e]);
        }
        uint4 o;
        __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&o);
#pragma unroll
        for (int e = 0; e < 8; e++)
            ob[e] = __float2bfloat16(acc[e]);
        reinterpret_cast<uint4*>(s_x + m * kCols)[q] = o;
    }
    asm volatile(
        "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n @!p bra WAIT_%=;\n}" ::
            "r"(smem_u32(&in_bar))
        : "memory");
    __syncthreads();

    // post-mapping of column h: the new streams, in place; four rows per group, the group's inputs loaded first
    float4 const* s_post4 = reinterpret_cast<float4 const*>(s_pc);
    float4 const* s_comb4 = reinterpret_cast<float4 const*>(s_pc + kMaxRows * 4);
    if (col_thread)
    {
        for (int m0 = 0; m0 < M; m0 += 4)
        {
            float xv[4], rp[4][kN];
            float4 po[4], cm[4][kN];
#pragma unroll
            for (int k = 0; k < 4; k++)
            {
                int const m = min(m0 + k, M - 1);
                xv[k] = __bfloat162float(s_x[m * kCols + t]);
#pragma unroll
                for (int i = 0; i < kN; i++)
                    rp[k][i] = __bfloat162float(s_res[m * kTok + i * kSlice + t]);
                po[k] = s_post4[m];
#pragma unroll
                for (int i = 0; i < kN; i++)
                    cm[k][i] = s_comb4[m * kN + i];
            }
#pragma unroll
            for (int k = 0; k < 4; k++)
            {
                int const m = m0 + k;
                if (m < M)
                {
                    float const pj[kN] = {po[k].x, po[k].y, po[k].z, po[k].w};
#pragma unroll
                    for (int j = 0; j < kN; j++)
                    {
                        float a = pj[j] * xv[k];
#pragma unroll
                        for (int i = 0; i < kN; i++)
                        {
                            float const c = j == 0 ? cm[k][i].x
                                : j == 1           ? cm[k][i].y
                                : j == 2           ? cm[k][i].z
                                                   : cm[k][i].w;
                            a += c * rp[k][i];
                        }
                        const __nv_bfloat16 bb = __float2bfloat16(a);
                        s_res[m * kTok + j * kSlice + t] = bb;
                    }
                }
            }
        }
    }
    __syncthreads();
    // this warp's partial mixer outputs (K slice above) and Gram entries (columns [32 w, 32 w + 32)) on the
    // tensor cores; A operands from the new streams (BF16, exact in TF32)
    asm volatile(
        "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n @!p bra WAIT_%=;\n}" ::
            "r"(smem_u32(&fn_bar))
        : "memory");
    if (col_thread)
    {
        auto rv = [&](int m, int j, int col) -> uint32_t
        { return m < M ? f32_bits(__bfloat162float(s_res[m * kTok + j * kSlice + col])) : 0u; };
        for (int mt = 0; mt * 16 < M; mt++)
        {
            float acc[3][4] = {};
            int const r0 = 16 * mt + g, r1 = r0 + 8;
#pragma unroll
            for (int st = 0; st < 16; st++)
            {
                int const col = mcol + 8 * st + tig;
                const uint32_t a0 = rv(r0, mj, col), a1 = rv(r1, mj, col), a2 = rv(r0, mj, col + 4),
                               a3 = rv(r1, mj, col + 4);
#pragma unroll
                for (int nt = 0; nt < 3; nt++)
                {
                    float const* fb = s_fn + (nt * 8 + g) * kFnRow + mj * kCols + col;
                    mma_tf32(acc[nt], a0, a1, a2, a3, to_tf32(fb[0]), to_tf32(fb[4]));
                }
            }
#pragma unroll
            for (int nt = 0; nt < 3; nt++)
            {
                int const n = nt * 8 + 2 * tig;
                if (r0 < M)
                {
                    s_warp[(warp * kMaxRows + r0) * kAcc + n] = acc[nt][0];
                    s_warp[(warp * kMaxRows + r0) * kAcc + n + 1] = acc[nt][1];
                }
                if (r1 < M)
                {
                    s_warp[(warp * kMaxRows + r1) * kAcc + n] = acc[nt][2];
                    s_warp[(warp * kMaxRows + r1) * kAcc + n + 1] = acc[nt][3];
                }
            }
        }
        // Gram: rows (token 4 mg + r / 4, stream r % 4) x columns (token 4 mg + 2 q + n / 4, stream n % 4)
        for (int mg = 0; mg * 4 < M; mg++)
        {
            float ga[2][4] = {};
            int const ta0 = 4 * mg + (g >> 2), ta1 = ta0 + 2, sa = g & 3;
#pragma unroll
            for (int st = 0; st < 4; st++)
            {
                int const col = 32 * warp + 8 * st + tig;
                const uint32_t a0 = rv(ta0, sa, col), a1 = rv(ta1, sa, col), a2 = rv(ta0, sa, col + 4),
                               a3 = rv(ta1, sa, col + 4);
#pragma unroll
                for (int q = 0; q < 2; q++)
                {
                    int const tb = 4 * mg + 2 * q + (g >> 2);
                    mma_tf32(ga[q], a0, a1, a2, a3, rv(tb, sa, col), rv(tb, sa, col + 4));
                }
            }
#pragma unroll
            for (int q = 0; q < 2; q++)
            {
                int const tb = 2 * q + (tig >> 1), kb0 = 2 * (tig & 1);
#pragma unroll
                for (int c = 0; c < 4; c++)
                {
                    int const ta = (c < 2 ? 0 : 2) + (g >> 2), kb = kb0 + (c & 1), token = 4 * mg + ta;
                    if (ta == tb && sa <= kb && token < M)
                        s_warp[(warp * kMaxRows + token) * kAcc + kMix + gram_idx(sa, kb)] = ga[q][c];
                }
            }
        }
    }
    __syncthreads();
    // this CTA's sums into slot `cta` of every CTA's s_all, then each CTA adds the 16 slots in CTA order
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); // every CTA of the cluster has started
    for (int i = t; i < M * kAcc; i += kThreads)
    {
        float s3 = 0.f;
#pragma unroll
        for (int w8 = 0; w8 < kWarps; w8++)
            s3 += s_warp[(w8 * kMaxRows) * kAcc + i];
#pragma unroll
        for (int c = 0; c < kCtas; c++)
            cluster.map_shared_rank(s_all, c)[cta * kMaxRows * kAcc + i] = s3;
    }
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    for (int i = t; i < M * kAcc; i += kThreads)
    {
        float s3 = 0.f;
#pragma unroll
        for (int c = 0; c < kCtas; c++)
            s3 += s_all[c * kMaxRows * kAcc + i];
        s_tot[i] = s3;
    }
    __syncthreads();
    // per row: the collapse coefficients (pre_j and the RMSNorm scale of the collapsed row from the Gram matrix)
    if (warp == 0 && lane < M)
    {
        float const* a = s_tot + lane * kAcc;
        float G[kN][kN];
#pragma unroll
        for (int j = 0; j < kN; j++)
#pragma unroll
            for (int i = j; i < kN; i++)
                G[j][i] = G[i][j] = a[kMix + gram_idx(j, i)];
        float const rs = rsqrtf((G[0][0] + G[1][1] + G[2][2] + G[3][3]) / (float) kNH + p.norm_eps);
        float pre[kN];
#pragma unroll
        for (int j = 0; j < kN; j++)
            pre[j] = 1.f / (1.f + __expf(-(a[j] * rs * p.scale[0] + p.base[j]))) + p.hc_eps;
        float q = 0.f;
#pragma unroll
        for (int j = 0; j < kN; j++)
#pragma unroll
            for (int i = 0; i < kN; i++)
                q += pre[j] * pre[i] * G[j][i];
        float const sc = rsqrtf(q / (float) kH + p.li_eps);
#pragma unroll
        for (int j = 0; j < kN; j++)
            s_coef[lane][j] = pre[j] * sc;
    }
    __syncthreads();
    if (col_thread)
    {
        // layer_input of column h for every row
#pragma unroll
        for (int m = 0; m < kMaxRows; m++)
        {
            if (m < M)
            {
                float y = 0.f;
#pragma unroll
                for (int j = 0; j < kN; j++)
                    y += s_coef[m][j] * __bfloat162float(s_res[m * kTok + j * kSlice + t]);
                s_x[m * kCols + t] = __float2bfloat16(y * wh); // s_x (the all-reduced rows) is free again
            }
        }
    }
    else if (lane < (kMaxRows + kCtas - 1) / kCtas && cta + kCtas * lane < M)
    {
        // aux warp: post and the Sinkhorn-normalized comb of the rows m with m % 16 == cta, one lane each
        int const m = cta + kCtas * lane;
        float const* a = s_tot + m * kAcc;
        float d[kN];
#pragma unroll
        for (int j = 0; j < kN; j++)
            d[j] = a[kMix + gram_idx(j, j)];
        float const rs = rsqrtf((d[0] + d[1] + d[2] + d[3]) / (float) kNH + p.norm_eps);
#pragma unroll
        for (int j = 0; j < kN; j++)
            s_sk[lane][j] = p.post_mult / (1.f + __expf(-(a[kN + j] * rs * p.scale[1] + p.base[kN + j])));
        float c[kN][kN];
#pragma unroll
        for (int i = 0; i < kN; i++)
        {
            float mx = -3.4e38f;
#pragma unroll
            for (int j = 0; j < kN; j++)
            {
                c[i][j] = a[2 * kN + i * kN + j] * rs * p.scale[2] + p.base[2 * kN + i * kN + j];
                mx = fmaxf(mx, c[i][j]);
            }
            float s3 = 0.f;
#pragma unroll
            for (int j = 0; j < kN; j++)
            {
                c[i][j] = __expf(c[i][j] - mx);
                s3 += c[i][j];
            }
            float const inv = 1.f / s3;
#pragma unroll
            for (int j = 0; j < kN; j++)
                c[i][j] = c[i][j] * inv + p.sk_eps;
        }
        for (int it = 0; it < p.sk_iters; it++)
        {
            if (it > 0)
            {
#pragma unroll
                for (int i = 0; i < kN; i++)
                {
                    float const inv = 1.f / (c[i][0] + c[i][1] + c[i][2] + c[i][3] + p.sk_eps);
#pragma unroll
                    for (int j = 0; j < kN; j++)
                        c[i][j] *= inv;
                }
            }
#pragma unroll
            for (int j = 0; j < kN; j++)
            {
                float const inv = 1.f / (c[0][j] + c[1][j] + c[2][j] + c[3][j] + p.sk_eps);
#pragma unroll
                for (int i = 0; i < kN; i++)
                    c[i][j] *= inv;
            }
        }
#pragma unroll
        for (int i = 0; i < kN; i++)
#pragma unroll
            for (int j = 0; j < kN; j++)
                s_sk[lane][kN + i * kN + j] = c[i][j];
    }
    __syncthreads();
    // 16-byte stores: this CTA's 256 columns of every residual stream row and layer_input row
    for (int i = t; i < M * (kN + 1) * kChunks; i += kThreads)
    {
        int const sl = i / kChunks, q = i % kChunks, m = sl / (kN + 1), j = sl % (kN + 1);
        if (j < kN)
            reinterpret_cast<uint4*>(p.res_out + ((size_t) m * kN + j) * kH + c0)[q]
                = reinterpret_cast<uint4 const*>(s_res + m * kTok + j * kSlice)[q];
        else
            reinterpret_cast<uint4*>(p.li_out + (size_t) m * kH + c0)[q]
                = reinterpret_cast<uint4 const*>(s_x + m * kCols)[q];
    }
    if (col_thread)
    {
    }
    else if (lane < (kMaxRows + kCtas - 1) / kCtas && cta + kCtas * lane < M)
    {
        int const m = cta + kCtas * lane;
#pragma unroll
        for (int j = 0; j < kN; j++)
            p.post_out[m * kN + j] = s_sk[lane][j];
#pragma unroll
        for (int i = 0; i < kN * kN; i++)
            p.comb_out[m * kN * kN + i] = s_sk[lane][kN + i];
    }
    cudaTriggerProgrammaticLaunchCompletion(); // after the outputs: the next kernel's CTAs do not crowd them
    // Lamport rotation (LamportComm::update): once every CTA has counted in
    if (cta == 0 && t == 0)
    {
        while (*reinterpret_cast<int volatile*>(counter_ptr) != kCtas)
        {
        }
        *flag_ptr = (flag + 1) % 3;
        clear_ptr[0] = (long long) M * kH * kRanks * (long long) sizeof(__nv_bfloat16); // bytes, as LamportComm
        *counter_ptr = 0;
    }
}

} // namespace

cudaError_t mhcArBoundary(MhcArBoundaryArgs const& a, cudaStream_t stream)
{
    if (a.rows < 1 || a.rows > kMaxRows)
        return cudaErrorInvalidValue;
    constexpr int kMaxDevices = 64;
    static bool attributes_set[kMaxDevices] = {};
    int dev = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess)
        return err;
    if (dev >= kMaxDevices)
        return cudaErrorNotSupported;
    if (!attributes_set[dev])
    {
        err = cudaFuncSetAttribute(mhc_ar_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
        if (err == cudaSuccess)
            err = cudaFuncSetAttribute(mhc_ar_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
        if (err != cudaSuccess)
            return err;
        attributes_set[dev] = true;
    }
    Params p{a.workspace, a.rank, static_cast<__nv_bfloat16 const*>(a.partial),
        static_cast<__nv_bfloat16 const*>(a.residual), a.post_prev, a.comb_prev, a.fn, a.scale, a.base,
        static_cast<__nv_bfloat16 const*>(a.norm_weight), static_cast<__nv_bfloat16*>(a.residual_out), a.post_out,
        a.comb_out, static_cast<__nv_bfloat16*>(a.layer_input), a.rows, a.sinkhorn_iters, a.norm_eps, a.hc_eps,
        a.sinkhorn_eps, a.post_mult, a.layer_input_eps};
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(kCtas);
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = kSmem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = a.pdl ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, mhc_ar_kernel, p);
}

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
