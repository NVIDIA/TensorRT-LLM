/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

// Lamport boundary device functions (tagged-word exchange), for producers that fuse the attention boundary into their
// own epilogue (o_proj GEMV -> push -> poll -> reduce + residual + Gemma norm + MXFP8).
// Math = FlashInfer's one-shot AR + norm (fp32 acc, ftz); lb_kernel (allreduce_norm_quant_ext.cu) is bit-exact to it.
//
// Exchange word (32 bit): (tag << 16) | bf16. tag = round % 65535 + 1, round = the caller's per-site counter (same on
// every rank; read it before griddepcontrol.wait, store round back at the end). Receive area per site:
// [world][MAXT=16][6144] words; the slice of (src rank r, row t, col c) is at ((r * 16 + t) * 6144 + c).
// A site must not be reused before every rank has finished polling it (one use per decode step is safe).
//
// The norm needs the row's sum of squares over all 6144 columns. Two ways to get it across CTAs:
//  - cluster (lb_kernel): CTAs of one row in a cluster, st.async + mbarrier exchange;
//  - global, any grid shape: rowsq_publish / rowsq_gather below. Each CTA owning a column tile publishes its partial
//    sum of squares per row as a tagged 64-bit word; every CTA polls all tiles' words and sums them in tile order.
//    One intra-GPU hop, no flags, no fences, no re-arm.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace lb
{
constexpr int H = 6144;
constexpr int MAXT = 16;
constexpr int MAXW = 8;

__device__ __forceinline__ float add_f(float a, float b)
{
    float d;
    asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

__device__ __forceinline__ float mul_f(float a, float b)
{
    float d;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

__device__ __forceinline__ float fma_f(float a, float b, float c)
{
    float d;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
    return d;
}

__device__ __forceinline__ float div_f(float a, float b)
{
    float d;
    asm("div.full.ftz.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

__device__ __forceinline__ float rsqrt_f(float a)
{
    float d;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a));
    return d;
}

__device__ __forceinline__ float rcp_f(float a)
{
    float d;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a));
    return d;
}

__device__ __forceinline__ float bf16_round(float a)
{
    return __bfloat162float(__float2bfloat16_rn(a));
}

__device__ __forceinline__ uint32_t tag_of(unsigned round)
{
    return (round % 65535u) + 1u;
}

__device__ __forceinline__ uint4 ldv4(uint32_t const* p)
{
    uint4 v;
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p)
                 : "memory");
    return v;
}

__device__ __forceinline__ void st_sys_v4(uint32_t* p, uint4 v)
{
    asm volatile("st.relaxed.sys.global.v4.b32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}

// Push 8 bf16 (one uint4) of this rank's partial, row t, columns [c0, c0 + 8), into every other rank's area.
__device__ __forceinline__ void push8(
    uint32_t* const* recv_peer, int rank, int world, int t, int c0, uint4 x, uint32_t tag)
{
    uint32_t const xw[4] = {x.x, x.y, x.z, x.w};
    uint4 const w0 = make_uint4((tag << 16) | (xw[0] & 0xFFFFu), (tag << 16) | (xw[0] >> 16),
        (tag << 16) | (xw[1] & 0xFFFFu), (tag << 16) | (xw[1] >> 16));
    uint4 const w1 = make_uint4((tag << 16) | (xw[2] & 0xFFFFu), (tag << 16) | (xw[2] >> 16),
        (tag << 16) | (xw[3] & 0xFFFFu), (tag << 16) | (xw[3] >> 16));
    int64_t const off = ((int64_t) rank * MAXT + t) * H + c0;
#pragma unroll
    for (int r = 0; r < MAXW; ++r)
        if (r < world && r != rank)
        {
            st_sys_v4(recv_peer[r] + off, w0);
            st_sys_v4(recv_peer[r] + off + 4, w1);
        }
}

// Poll every other rank's 8 columns, then the rank-ordered fp32 sum (stock order) of all ranks' bf16 values.
__device__ __forceinline__ void poll_sum8(
    uint32_t const* recv_local, int rank, int world, int t, int c0, uint4 own, uint32_t tag, float s[8])
{
    uint4 v0[MAXW], v1[MAXW];
    uint32_t pending = ((1u << world) - 1u) & ~(1u << rank);
    long long spins = 0;
    while (pending)
    {
#pragma unroll
        for (int r = 0; r < MAXW; ++r)
            if ((pending >> r) & 1u)
            {
                uint32_t const* src = recv_local + ((int64_t) r * MAXT + t) * H + c0;
                v0[r] = ldv4(src);
                v1[r] = ldv4(src + 4);
            }
#pragma unroll
        for (int r = 0; r < MAXW; ++r)
            if (((pending >> r) & 1u) && (v0[r].x >> 16) == tag && (v0[r].y >> 16) == tag && (v0[r].z >> 16) == tag
                && (v0[r].w >> 16) == tag && (v1[r].x >> 16) == tag && (v1[r].y >> 16) == tag && (v1[r].z >> 16) == tag
                && (v1[r].w >> 16) == tag)
                pending &= ~(1u << r);
        if (++spins > (1LL << 31))
            __trap();
    }
#pragma unroll
    for (int r = 0; r < MAXW; ++r)
    {
        if (r >= world)
            break;
        float f[8];
        if (r == rank)
        {
            uint32_t const xw[4] = {own.x, own.y, own.z, own.w};
#pragma unroll
            for (int q = 0; q < 4; ++q)
            {
                f[2 * q] = __uint_as_float(xw[q] << 16);
                f[2 * q + 1] = __uint_as_float(xw[q] & 0xFFFF0000u);
            }
        }
        else
        {
            uint32_t const ww[8] = {v0[r].x, v0[r].y, v0[r].z, v0[r].w, v1[r].x, v1[r].y, v1[r].z, v1[r].w};
#pragma unroll
            for (int e = 0; e < 8; ++e)
                f[e] = __uint_as_float(ww[e] << 16);
        }
#pragma unroll
        for (int e = 0; e < 8; ++e)
            s[e] = r == 0 ? f[e] : add_f(s[e], f[e]);
    }
}

// s (rank sum) + residual -> bf16 residual_out (returned); s becomes the rounded updated value; sq accumulates.
__device__ __forceinline__ uint4 residual_add8(float s[8], uint4 residual, float& sq)
{
    __nv_bfloat162 const* rh = reinterpret_cast<__nv_bfloat162 const*>(&residual);
    uint32_t upd[4];
#pragma unroll
    for (int q = 0; q < 4; ++q)
    {
        float2 const rf = __bfloat1622float2(rh[q]);
        __nv_bfloat162 const u
            = __floats2bfloat162_rn(add_f(bf16_round(s[2 * q]), rf.x), add_f(bf16_round(s[2 * q + 1]), rf.y));
        upd[q] = *reinterpret_cast<uint32_t const*>(&u);
        float2 const uf = __bfloat1622float2(u);
        s[2 * q] = uf.x;
        s[2 * q + 1] = uf.y;
        sq = fma_f(uf.y, uf.y, fma_f(uf.x, uf.x, sq));
    }
    return make_uint4(upd[0], upd[1], upd[2], upd[3]);
}

// inv = rsqrt(total / H + eps), total = the row's sum of squares (all 6144 columns).
__device__ __forceinline__ float inv_rms(float total, float eps)
{
    return rsqrt_f(add_f(div_f(total, (float) H), eps));
}

// Gemma norm of 8 values: bf16(s * inv * (1 + w)); s becomes the rounded normed values (input to MXFP8).
__device__ __forceinline__ uint4 norm8(float s[8], float inv, uint4 gamma, int gemma)
{
    __nv_bfloat162 const* wv = reinterpret_cast<__nv_bfloat162 const*>(&gamma);
    uint32_t o[4];
#pragma unroll
    for (int q = 0; q < 4; ++q)
    {
        float2 wf = __bfloat1622float2(wv[q]);
        if (gemma)
        {
            wf.x = add_f(1.0f, wf.x);
            wf.y = add_f(1.0f, wf.y);
        }
        __nv_bfloat162 const n
            = __floats2bfloat162_rn(mul_f(mul_f(s[2 * q], inv), wf.x), mul_f(mul_f(s[2 * q + 1], inv), wf.y));
        o[q] = *reinterpret_cast<uint32_t const*>(&n);
        float2 const nf = __bfloat1622float2(n);
        s[2 * q] = nf.x;
        s[2 * q + 1] = nf.y;
    }
    return make_uint4(o[0], o[1], o[2], o[3]);
}

// MXFP8 (trtllm mxfp8_quantize(normed, swizzled)): the 4 lanes (lane & ~3 .. +3) holding one 32-column block must call
// this together (warp-synchronous, full mask). Writes 8 e4m3 bytes and, from lane % 4 == 0, the UE8M0 scale into the
// F8_128x4 swizzled layout (sf_k_tiles = 6144 / 128 = 48).
__device__ __forceinline__ void mxfp8_8(
    float const s[8], bool active, int t, int c0, uint8_t* out_q, uint8_t* out_sf, int sf_k_tiles)
{
    float amax = 0.f;
    if (active)
    {
#pragma unroll
        for (int e = 0; e < 8; ++e)
            amax = fmaxf(amax, fabsf(s[e]));
    }
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    if (!active)
        return;
    __nv_fp8_e8m0 code;
    code.__x = __nv_cvt_float_to_e8m0(amax * rcp_f(448.0f), __NV_SATFINITE, cudaRoundPosInf);
    float const out_scale = amax != 0.f ? rcp_f(static_cast<float>(code)) : 0.0f;
    uint32_t packed[2];
#pragma unroll
    for (int h = 0; h < 2; ++h)
    {
        __nv_fp8x2_e4m3 const lo(make_float2(s[4 * h] * out_scale, s[4 * h + 1] * out_scale));
        __nv_fp8x2_e4m3 const hi(make_float2(s[4 * h + 2] * out_scale, s[4 * h + 3] * out_scale));
        packed[h] = static_cast<uint32_t>(lo.__x) | (static_cast<uint32_t>(hi.__x) << 16);
    }
    *reinterpret_cast<uint2*>(out_q + (int64_t) t * H + c0) = make_uint2(packed[0], packed[1]);
    if ((threadIdx.x & 3) == 0)
    {
        int const kb = c0 / 32;
        out_sf[(int64_t) (t / 128) * sf_k_tiles * 512 + (kb / 4) * 512 + (t % 32) * 16 + ((t % 128) / 32) * 4 + kb % 4]
            = code.__x;
    }
}

// Global row sum-of-squares exchange for any grid shape. slots: [MAXT][ntiles] 64-bit words on this GPU, one per
// (row, column tile). The tile owner publishes (tag << 32 | float bits); every reader waits for all ntiles words of
// the row and sums them in tile order (deterministic). One thread per (row) call; spread rows/tiles over lanes.
__device__ __forceinline__ void rowsq_publish(
    unsigned long long* slots, int ntiles, int t, int tile, float part, uint32_t tag)
{
    unsigned long long const w = ((unsigned long long) tag << 32) | __float_as_uint(part);
    asm volatile("st.relaxed.gpu.global.u64 [%0], %1;" ::"l"(slots + (int64_t) t * ntiles + tile), "l"(w) : "memory");
}

__device__ __forceinline__ float rowsq_gather(unsigned long long const* slots, int ntiles, int t, uint32_t tag)
{
    float total = 0.f;
    for (int i = 0; i < ntiles; ++i)
    {
        unsigned long long w;
        do
        {
            asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];"
                         : "=l"(w)
                         : "l"(slots + (int64_t) t * ntiles + i)
                         : "memory");
        } while ((uint32_t) (w >> 32) != tag);
        total = add_f(total, __uint_as_float((uint32_t) w));
    }
    return total;
}

} // namespace lb
