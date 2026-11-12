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

#include "tensorrt_llm/kernels/minimaxM3/draftStep.h"

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// One pass of the fused greedy Eagle3 draft loop of MiniMax-M3 (GB300, TP4, <= 8 requests per pass).
//
// One kernel replaces the draft lm_head GEMM, the logits all-gather / float cast / argmax of the in-tree greedy draft
// sampling and the draft loop's per-pass bookkeeping, at the in-tree precision:
//
// * The vocab-sharded BF16 lm_head weight is stored in mma.m16n8k16 A-fragment order (the same values): per 16-row
//   tile and k16 step, 512 bytes; lane (g, j) holds 16 bytes = {row g k 2j..2j+1, row g+8 k 2j.., row g k 2j+8..,
//   row g+8 k 2j+8..}.
// * Or, by default, in a lossless 12-bit coding of those fragments (75% of the bytes; the target head's coding, see
//   targetHead.cu): per 16-row tile and unit of 4 k16 steps a lane reads three 16-byte words, the sign/mantissa bytes
//   of steps 0-1, of steps 2-3, then 4 code words (exponent - base, 4 bits; 15 = an exception). The decoded values
//   are the checkpoint's BF16 bits; an exception decodes to +0 and the segment's flush adds the exact product x * w
//   from the tile's list (products of BF16 values are exact in FP32).
// * A persistent grid (one 16-warp CTA per SM) streams whole tiles over the full K with BF16 MMAs and FP32
//   accumulation; the BF16 head input rows sit in shared memory in B-fragment order, unchanged.
// * Each logit is rounded to BF16 (the in-tree logits are the BF16 GEMM output) and becomes a 64-bit key
//   ordered(value) << 32 | ~global_index, with -0 taken as +0: the larger value wins and ties pick the lowest index, as
//   torch.max / torch.argmax over the gathered logits do. Keys are max-reduced per row: warp shuffles, then shared
//   memory, then atomicMax.
// * The last CTA to finish pushes the rank's per-row key into every rank's exchange slot through a multicast mapping
//   (Lamport: 0 = empty, a key is never 0), polls its own slots for all ranks, takes the global argmax, re-arms the
//   slots, and runs the pass's bookkeeping. Call n uses slot half n % 2.

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3
{
namespace draftStep
{
namespace
{
constexpr int kUnroll = 8;       // k16 chunks in flight per warp
constexpr int kChunkBytes = 512; // one tile's k16 step: 32 lanes x 16 bytes

// Coded units (4 k16 steps, 3 x 16 bytes per lane) in flight per warp: 3 up to 4 rows, 2 at 8 rows.
template <int MP>
constexpr int codedUnroll()
{
    return MP >= 8 ? 2 : 3;
}

__device__ __forceinline__ void mma_bf16(
    float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// BF16 pair p (values 2p, 2p + 1) of a lane's 8 coded values: sign/mantissa bytes sm (4 per word), codes word cw.
__device__ __forceinline__ uint32_t coded_decode_pair(uint32_t sm_word, int byte0, uint32_t cw, int p, uint32_t base)
{
    uint32_t const b = __byte_perm(sm_word, 0u, byte0 == 0 ? 0x4140 : 0x4342); // {b0, 0, b1, 0}
    uint32_t const q = (cw >> (4 * p)) & 0x000f000fu;
    uint32_t const e = q * 128u + base;                                        // exponent field of both halves
    // Code 15 (an exception, sign/mantissa byte 0) decodes to +0.
    return ((b & 0x007f007fu) | ((b << 8) & 0x80008000u) | e) & __vcmpne2(q, 0x000f000fu);
}

// One coded unit (4 k16 steps) against the BF16 B fragments: steps alternate between the two accumulators, as the
// k16 steps of the plain BF16 head do.
template <int MP>
__device__ __forceinline__ void chunk_mma_coded(uint4 const& sm01, uint4 const& sm23, uint4 const& cw, uint32_t base,
    uint2 const* xs, int unit, int j, int g, bool b_active, float (&da)[4], float (&db)[4])
{
    uint32_t const smw[8] = {sm01.x, sm01.y, sm01.z, sm01.w, sm23.x, sm23.y, sm23.z, sm23.w};
    uint32_t const cws[4] = {cw.x, cw.y, cw.z, cw.w};
#pragma unroll
    for (int ci = 0; ci < 4; ++ci)
    {
        uint2 b = make_uint2(0u, 0u);
        if (b_active)
            b = xs[((unit * 4 + ci) * 4 + j) * MP + g];
        uint32_t const w0 = coded_decode_pair(smw[2 * ci], 0, cws[ci], 0, base);
        uint32_t const w1 = coded_decode_pair(smw[2 * ci], 2, cws[ci], 1, base);
        uint32_t const w2 = coded_decode_pair(smw[2 * ci + 1], 0, cws[ci], 2, base);
        uint32_t const w3 = coded_decode_pair(smw[2 * ci + 1], 2, cws[ci], 3, base);
        if (ci & 1)
            mma_bf16(db, w0, w1, w2, w3, b.x, b.y);
        else
            mma_bf16(da, w0, w1, w2, w3, b.x, b.y);
    }
}

// Exceptions of `tile` with k in [kb, ke): v += x * w (exact) for the owner lanes (row g or g + 8; tokens 2j,
// 2j + 1 < M), x from the staged B fragments. Warp-uniform (shuffles).
template <int MP>
__device__ __forceinline__ void coded_exceptions(
    CodedArgs const& cd, int tile, int kb, int ke, uint2 const* xs, int M, int lane, float (&v)[4])
{
    int const g = lane >> 2, j = lane & 3;
    int const e0 = __ldg(cd.exc_off + tile), e1 = __ldg(cd.exc_off + tile + 1);
    for (int b0 = e0; b0 < e1; b0 += 32)
    {
        uint32_t pos = 0xffffffffu;
        float2 wv = make_float2(0.f, 0.f);
        if (b0 + lane < e1)
        {
            pos = __ldg(cd.exc_pos + b0 + lane);
            wv = __ldg(cd.exc_w + b0 + lane);
        }
        int const cnt = min(32, e1 - b0);
        for (int q = 0; q < cnt; ++q)
        {
            uint32_t const pq = __shfl_sync(0xffffffffu, pos, q);
            float const w0 = __shfl_sync(0xffffffffu, wv.x, q);
            int const r = static_cast<int>(pq >> 16), k = static_cast<int>(pq & 0xffffu);
            if (k >= kb && k < ke && (r & 7) == g)
            {
                int const st = k >> 4, rr = k & 15;
                bool const hi = r >= 8;
#pragma unroll
                for (int q2 = 0; q2 < 2; ++q2)
                {
                    int const n = 2 * j + q2;
                    if (n < M)
                    {
                        uint2 const bw = xs[(st * 4 + ((rr & 7) >> 1)) * MP + n];
                        uint32_t const word = rr >= 8 ? bw.y : bw.x;
                        float const xv = __uint_as_float((rr & 1) ? (word & 0xffff0000u) : (word << 16));
                        float const d = __fmul_rn(xv, w0);
                        if (hi)
                            v[2 + q2] += d;
                        else
                            v[q2] += d;
                    }
                }
            }
            __syncwarp();
        }
    }
}

__device__ __forceinline__ uint4 ldg_stream(uint4 const* p)
{
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "l"(p));
    return r;
}

// The argmax value of a logit: its BF16 rounding (the in-tree logits' dtype), -0 as +0.
__device__ __forceinline__ float key_value(float v)
{
    float const r = __bfloat162float(__float2bfloat16(v));
    return r == 0.f ? 0.f : r;
}

__device__ __forceinline__ unsigned long long make_key(float v, uint32_t index)
{
    uint32_t u = __float_as_uint(v);
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return (static_cast<unsigned long long>(u) << 32) | static_cast<unsigned long long>(0xFFFFFFFFu - index);
}

__device__ __forceinline__ unsigned long long shfl_max(unsigned long long k, int o)
{
    unsigned long long const ok = __shfl_xor_sync(0xffffffffu, k, o);
    return ok > k ? ok : k;
}

// One k16 step of a tile: the lane's A fragment w against the activation B fragment of step `chunk` (rows g < MP).
// Even and odd steps accumulate separately (two independent MMA chains); the tile's sum is da + db.
template <int MP>
__device__ __forceinline__ void chunk_mma(
    uint4 const& w, uint2 const* xs, int chunk, int j, int g, bool b_active, float (&da)[4], float (&db)[4])
{
    uint2 b = make_uint2(0u, 0u);
    if (b_active)
        b = xs[(chunk * 4 + j) * MP + g];
    if (chunk & 1)
        mma_bf16(db, w.x, w.y, w.z, w.w, b.x, b.y);
    else
        mma_bf16(da, w.x, w.y, w.z, w.w, b.x, b.y);
}

// Add a warp's segment of a tile into the CTA's accumulator (with a coded head, the segment's exceptions first).
template <int MP, bool CODED>
__device__ __forceinline__ void flush_tile(
    float4* acc, float (&da)[4], float (&db)[4], Params const& p, int tile, int kb, int ke, uint2 const* xs, int lane)
{
    float v[4] = {da[0] + db[0], da[1] + db[1], da[2] + db[2], da[3] + db[3]};
    if constexpr (CODED)
        coded_exceptions<MP>(p.cd, tile, kb, ke, xs, p.M, lane, v);
    float* a = reinterpret_cast<float*>(acc);
    atomicAdd(a + 0, v[0]);
    atomicAdd(a + 1, v[1]);
    atomicAdd(a + 2, v[2]);
    atomicAdd(a + 3, v[3]);
#pragma unroll
    for (int e = 0; e < 4; ++e)
    {
        da[e] = 0.f;
        db[e] = 0.f;
    }
}

// Balanced streaming GEMV over the fragment-ordered weight, persistent grid of 16-warp CTAs. CTA c owns tiles
// [t_begin, t_end); its (tile, chunk) space is split into 16 equal contiguous ranges, one per warp, so every warp
// streams the same number of bytes. A tile straddles at most two warps; their partial sums are added into shared
// memory (two addends: the order does not change the sum). MP = rows padded to 1 / 2 / 4 / 8. A chunk is one k16
// step of the BF16 head, or one coded unit (4 k16 steps) of the coded head (CODED: no shared-memory prologue).
template <int MP, bool CODED>
__global__ void __launch_bounds__(kWarps * 32, 1)
    draft_step_kernel(Params const p, int kchunks, int pf_chunks, int pf_off)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    extern __shared__ __align__(16) uint2 xs[]; // [K/16 steps][4 (j)][MP (g)], then the weight prologue
    __shared__ float4 acc_s[kMaxTilesPerCta][32];
    __shared__ unsigned long long red[kWarps][kMaxTokens];
    __shared__ __align__(8) unsigned long long pf_bar[kWarps];
    __shared__ bool s_last;
    __shared__ int s_tok[kMaxTokens];

    int const warp = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
    int const g = lane >> 2;
    int const j = lane & 3;
    constexpr int kChunkK = CODED ? 64 : 16; // k per chunk
    constexpr int kWords = CODED ? 3 : 1;    // 16-byte words per lane and chunk
    constexpr int KU = CODED ? codedUnroll<MP>() : kUnroll;
    int const K = kchunks * kChunkK;
    int const M = p.M;
    int const t_begin = static_cast<int>((static_cast<long long>(blockIdx.x) * p.n_tiles) / gridDim.x);
    int const t_end = static_cast<int>((static_cast<long long>(blockIdx.x + 1) * p.n_tiles) / gridDim.x);
    int const n_local = t_end - t_begin;
    int const span = n_local * kchunks;
    int const q_begin = warp * span / kWarps;
    int const q_end = (warp + 1) * span / kWarps;
    long long const base = static_cast<long long>(t_begin) * kchunks; // first chunk of this CTA
    uint4 const* const wf = p.wf;

    // Weights are static: start this warp's stream before waiting on the producer of x. The warp's first pf_chunks
    // chunks come in by one bulk copy into shared memory (dynamic offset pf_off), which fills the grid-dependency
    // wait and the staging; the register ring starts after them.
    int const n_pf = CODED ? 0 : min(pf_chunks, q_end - q_begin);
    uint4* const wpf = reinterpret_cast<uint4*>(reinterpret_cast<char*>(xs) + pf_off) + warp * pf_chunks * 32;
    if (n_pf > 0 && lane == 0)
    {
        uint32_t const bar = static_cast<uint32_t>(__cvta_generic_to_shared(&pf_bar[warp]));
        uint32_t const bytes = static_cast<uint32_t>(n_pf) * kChunkBytes;
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bar) : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                         static_cast<uint32_t>(__cvta_generic_to_shared(wpf))),
                     "l"(wf + (base + q_begin) * 32), "r"(bytes), "r"(bar)
                     : "memory");
    }
    __syncwarp();
    int const q_reg = q_begin + n_pf;
    uint4 buf[KU][kWords];
#pragma unroll
    for (int u = 0; u < KU; ++u)
    {
        if (q_reg + u < q_end)
        {
#pragma unroll
            for (int l = 0; l < kWords; ++l)
                buf[u][l] = ldg_stream(wf + ((base + q_reg + u) * kWords + l) * 32 + lane);
        }
    }
    for (int i = threadIdx.x; i < n_local * 32; i += blockDim.x)
        acc_s[i / 32][i % 32] = make_float4(0.f, 0.f, 0.f, 0.f);

    asm volatile("griddepcontrol.wait;" ::: "memory");

    // Stage the head input rows in B-fragment order: item (k16 step S, row gg) = two 16-byte loads of x, four
    // fragment words {x[2jj], x[2jj + 1]}, {x[2jj + 8], x[2jj + 9]} (BF16 pairs, moved unchanged). Rows >= M are 0.
    constexpr int kSU = MP >= 4 ? 4 : 2;
    int const n_items = (K / 16) * MP;
    for (int i0 = threadIdx.x; i0 < n_items; i0 += kSU * blockDim.x)
    {
        uint4 xv[kSU][2];
#pragma unroll
        for (int u = 0; u < kSU; ++u)
        {
            int const i = i0 + u * blockDim.x;
            int const gg = i % MP;
            int const S = i / MP;
            xv[u][0] = make_uint4(0u, 0u, 0u, 0u);
            xv[u][1] = make_uint4(0u, 0u, 0u, 0u);
            if (i < n_items && gg < M)
            {
                long long const xr = p.x_rows != nullptr ? p.x_rows[gg] : gg;
                uint4 const* xp = reinterpret_cast<uint4 const*>(p.x + xr * K + 16 * S);
                xv[u][0] = xp[0];
                xv[u][1] = xp[1];
            }
        }
#pragma unroll
        for (int u = 0; u < kSU; ++u)
        {
            int const i = i0 + u * blockDim.x;
            if (i >= n_items)
                continue;
            int const gg = i % MP;
            int const S = i / MP;
            uint32_t const lo[4] = {xv[u][0].x, xv[u][0].y, xv[u][0].z, xv[u][0].w};
            uint32_t const hi[4] = {xv[u][1].x, xv[u][1].y, xv[u][1].z, xv[u][1].w};
#pragma unroll
            for (int jj = 0; jj < 4; ++jj)
                xs[(S * 4 + jj) * MP + gg] = make_uint2(lo[jj], hi[jj]);
        }
    }
    __syncthreads();

    bool const b_active = g < MP;
    float da[4] = {0.f, 0.f, 0.f, 0.f};
    float db[4] = {0.f, 0.f, 0.f, 0.f};
    int tl = q_begin / kchunks;     // local tile of the current accumulator
    int c = q_begin - tl * kchunks; // chunk within that tile
    int cs = c;                     // first chunk of the warp's segment of that tile
    if (n_pf > 0)
    {
        uint32_t const bar = static_cast<uint32_t>(__cvta_generic_to_shared(&pf_bar[warp]));
        asm volatile(
            "{\n .reg .pred p;\n PF_WAIT%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n"
            " @!p bra PF_WAIT%=;\n}\n" ::"r"(bar)
            : "memory");
        for (int q = q_begin; q < q_reg; ++q)
        {
            chunk_mma<MP>(wpf[(q - q_begin) * 32 + lane], xs, c, j, g, b_active, da, db);
            if (++c == kchunks)
            {
                flush_tile<MP, CODED>(&acc_s[tl][lane], da, db, p, t_begin + tl, cs * kChunkK, K, xs, lane);
                c = 0;
                cs = 0;
                ++tl;
            }
        }
    }
    for (int q = q_reg; q < q_end; q += KU)
    {
        uint4 nxt[KU][kWords];
#pragma unroll
        for (int u = 0; u < KU; ++u)
        {
            if (q + KU + u < q_end)
            {
#pragma unroll
                for (int l = 0; l < kWords; ++l)
                    nxt[u][l] = ldg_stream(wf + ((base + q + KU + u) * kWords + l) * 32 + lane);
            }
        }
#pragma unroll
        for (int u = 0; u < KU; ++u)
        {
            if (q + u < q_end)
            {
                if constexpr (CODED)
                    chunk_mma_coded<MP>(buf[u][0], buf[u][kWords > 1 ? 1 : 0], buf[u][kWords > 2 ? 2 : 0], p.cd.base,
                        xs, c, j, g, b_active, da, db);
                else
                    chunk_mma<MP>(buf[u][0], xs, c, j, g, b_active, da, db);
                if (++c == kchunks)
                {
                    flush_tile<MP, CODED>(&acc_s[tl][lane], da, db, p, t_begin + tl, cs * kChunkK, K, xs, lane);
                    c = 0;
                    cs = 0;
                    ++tl;
                }
            }
        }
#pragma unroll
        for (int u = 0; u < KU; ++u)
#pragma unroll
            for (int l = 0; l < kWords; ++l)
                buf[u][l] = nxt[u][l];
    }
    if (c != 0)
        flush_tile<MP, CODED>(&acc_s[tl][lane], da, db, p, t_begin + tl, cs * kChunkK, c * kChunkK, xs, lane);
    __syncthreads();

    // Accumulator lane (g, j) of a tile: rows g / g + 8 (x, y / z, w) of tokens 2j / 2j + 1 (x, z / y, w).
    unsigned long long best0 = 0ull, best1 = 0ull; // tokens 2j, 2j + 1
    for (int t = warp; t < n_local; t += kWarps)
    {
        float4 const v = acc_s[t][lane];
        int const r0 = (t_begin + t) * 16 + g;
        uint32_t const i0 = static_cast<uint32_t>(p.vocab_offset + r0);
        unsigned long long k0 = make_key(key_value(v.x), i0);
        unsigned long long const k0b = make_key(key_value(v.z), i0 + 8);
        unsigned long long k1 = make_key(key_value(v.y), i0);
        unsigned long long const k1b = make_key(key_value(v.w), i0 + 8);
        k0 = k0b > k0 ? k0b : k0;
        k1 = k1b > k1 ? k1b : k1;
        best0 = k0 > best0 ? k0 : best0;
        best1 = k1 > best1 ? k1 : best1;
    }
#pragma unroll
    for (int o = 4; o < 32; o <<= 1)
    {
        best0 = shfl_max(best0, o);
        best1 = shfl_max(best1, o);
    }
    asm volatile("griddepcontrol.launch_dependents;");
    if (lane < 4)
    {
        red[warp][2 * lane] = best0;
        red[warp][2 * lane + 1] = best1;
    }
    __syncthreads();
    if (threadIdx.x < M)
    {
        unsigned long long b = 0ull;
#pragma unroll
        for (int w = 0; w < kWarps; ++w)
            b = red[w][threadIdx.x] > b ? red[w][threadIdx.x] : b;
        if (b != 0ull)
            atomicMax(p.keys + threadIdx.x, b);
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0)
    {
        unsigned int const prev = atomicAdd(p.counters, 1u);
        s_last = prev == gridDim.x - 1;
    }
    __syncthreads();
    if (!s_last)
        return;

    __threadfence();
    unsigned int const call = *reinterpret_cast<unsigned int volatile*>(p.counters + 1);
    int const half = static_cast<int>(call & 1u);
    if (threadIdx.x < M)
    {
        int const t = threadIdx.x;
        unsigned long long const mine = atomicExch(p.keys + t, 0ull);
        unsigned long long* slot_mc = p.ws_mc + (half * kMaxTokens + t) * p.world;
        unsigned long long* slot_uc = p.ws_uc + (half * kMaxTokens + t) * p.world;
        slot_mc[p.rank] = mine;
        unsigned long long best = 0ull;
        for (int r = 0; r < p.world; ++r)
        {
            unsigned long long v;
            do
            {
                v = *reinterpret_cast<unsigned long long volatile*>(slot_uc + r);
            } while (v == 0ull);
            best = v > best ? v : best;
            slot_uc[r] = 0ull;
        }
        s_tok[t] = static_cast<int>(0xFFFFFFFFu - static_cast<uint32_t>(best & 0xFFFFFFFFull));
        if (p.out_keys != nullptr)
            p.out_keys[t] = best;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        p.counters[0] = 0u;
        p.counters[1] = call + 1u;
    }
    if (threadIdx.x < M)
    {
        StepArgs const& st = p.st;
        int const b = threadIdx.x;
        int const tok = s_tok[b];
        st.draft_out[b * st.draft_stride + st.step] = tok;
        st.ids_next[b] = tok;
        long long const r = p.x_rows != nullptr ? p.x_rows[b] : b;
        st.pos_next[b] = st.pos_src[r] + 1;
        if (st.step == 0 && st.num_acc != nullptr)
        {
            st.kv_lens[b] -= st.draft_len - st.num_acc[b];
            st.seq_lens[b] = 1;
        }
        else if (st.step > 0)
        {
            st.kv_lens[b] += 1;
        }
    }
#endif
}

template <int MP, bool CODED>
cudaError_t launch(Params const& p, int K, int grid, bool pdl, cudaStream_t stream)
{
    auto kern = draft_step_kernel<MP, CODED>;
    size_t const xs_bytes = static_cast<size_t>(K / 16) * 4 * MP * sizeof(uint2);
    // Shared-memory weight prologue: chunks per warp that fit next to the kernel's static + dynamic shared memory.
    static int const sStaticSmem = []
    {
        cudaFuncAttributes fa{};
        return cudaFuncGetAttributes(&fa, draft_step_kernel<MP, CODED>) == cudaSuccess
            ? static_cast<int>(fa.sharedSizeBytes)
            : -1;
    }();
    static int const sMaxSmemPerBlock = []
    {
        int dev = 0, maxSmem = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&maxSmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        return maxSmem;
    }();
    int pf = 0;
    if (!CODED && sStaticSmem >= 0)
    {
        long long const room
            = static_cast<long long>(sMaxSmemPerBlock) - sStaticSmem - static_cast<long long>(xs_bytes) - 1024;
        long long const want = static_cast<long long>(kPrefetchKb) * 1024;
        pf = static_cast<int>(std::max(0ll, std::min(room, want)) / (kWarps * kChunkBytes));
    }
    size_t const smem = xs_bytes + static_cast<size_t>(pf) * kWarps * kChunkBytes;
    cudaError_t err = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
    if (err != cudaSuccess)
        return err;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(static_cast<unsigned>(grid));
    cfg.blockDim = dim3(kWarps * 32);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, kern, p, K / (CODED ? 64 : 16), pf, static_cast<int>(xs_bytes));
}

template <bool CODED>
cudaError_t dispatchRows(Params const& p, int K, int grid, bool pdl, cudaStream_t stream)
{
    if (p.M <= 1)
        return launch<1, CODED>(p, K, grid, pdl, stream);
    if (p.M <= 2)
        return launch<2, CODED>(p, K, grid, pdl, stream);
    if (p.M <= 4)
        return launch<4, CODED>(p, K, grid, pdl, stream);
    return launch<8, CODED>(p, K, grid, pdl, stream);
}

} // namespace

cudaError_t invokeDraftStep(Params const& p, int K, int grid, bool pdl, cudaStream_t stream)
{
    bool const coded = p.cd.exc_off != nullptr;
    if (p.M < 1 || p.M > kMaxTokens || K % (coded ? 64 : 16) != 0 || grid < 1)
        return cudaErrorInvalidValue;
    return coded ? dispatchRows<true>(p, K, grid, pdl, stream) : dispatchRows<false>(p, K, grid, pdl, stream);
}

} // namespace draftStep
} // namespace minimaxM3
} // namespace kernels

TRTLLM_NAMESPACE_END
