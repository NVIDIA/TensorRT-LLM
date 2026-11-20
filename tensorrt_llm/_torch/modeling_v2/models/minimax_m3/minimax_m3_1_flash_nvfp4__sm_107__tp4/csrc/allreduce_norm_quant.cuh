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

// Prepushed all-reduce + residual + Gemma RMSNorm (+ MXFP8) consumer kernel.
#pragma once
#include "lamport_exchange.cuh"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>
using namespace lb;
#ifndef H9_GT_DEFINED
#define H9_GT_DEFINED

__device__ __forceinline__ unsigned long long gt()
{
    unsigned long long t;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
    return t;
}
#endif

// Prepushed consumer: a producer (o_proj GEMV, MoE engine) already pushed every rank's partial into
// the prx workspace (Lamport sentinel words, per-(row, quarter) call parity). Cluster of 8 per row, st.async RMS
// exchange; the CTA pair sharing a quarter-row parity flag flips it once both have polled (after the exchange).
struct PushParamsV3
{
    int* peers[8];
    int const* flags;
    int rank, world, maxt, cluster;
    int ndst;
};

constexpr int kEmpty = static_cast<int>(0x80000000u);

struct V3Params
{
    int* ex;
    int* flags;
    int maxt;
    __nv_bfloat16 const* residual;
    __nv_bfloat16 const* gamma;
    __nv_bfloat16* out_res;
    __nv_bfloat16* out_norm;
    uint8_t* out_q;
    uint8_t* out_sf;
    float eps;
    int rows, world, gemma, sf_k_tiles, rank;
    int poll_mode;  // 0: spin; 1: __nanosleep backoff between polls (fewer L2 polls while the reference is late)
    unsigned long long*
        dbg;        // optional per-CTA stamps, ring of 16 calls: [16][ctas][4] = start, own ready, peer ready, end
    unsigned* dcnt; // per-CTA call counters for the ring
};

__device__ __forceinline__ int4 ldv_i4(int const* p)
{
    int4 v;
    asm volatile("ld.volatile.global.v4.s32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p)
                 : "memory");
    return v;
}

template <int CL, int TH>
__global__ void __launch_bounds__(TH) lbv3_kernel(const __grid_constant__ V3Params p)
{
    static_assert(CL % 4 == 0, "a quarter-row parity slice must be whole CTAs");
    __shared__ float warp_sq[TH / 32];
    __shared__ alignas(16) float csq[CL];
    __shared__ alignas(8) unsigned long long xbar;
    if (threadIdx.x == 0)
    {
        uint32_t const b = static_cast<uint32_t>(__cvta_generic_to_shared(&xbar));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(b) : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    int crank;
    asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(crank));
    int const t = blockIdx.x / CL;
    constexpr int cols = H / CL;
    int const c0 = crank * cols + threadIdx.x * 8;
    bool const active = threadIdx.x * 8 < cols && t < p.rows;
    int const fidx = t * 4 + crank / (CL / 4);
    uint4 wraw = make_uint4(0, 0, 0, 0), rr = make_uint4(0, 0, 0, 0);
    if (active)
    {
        wraw = *reinterpret_cast<uint4 const*>(p.gamma + c0);
        rr = *reinterpret_cast<uint4 const*>(
            p.residual + (int64_t) t * H + c0); // written grids earlier (prx prepushed 2)
    }
    asm volatile("griddepcontrol.launch_dependents;");
    unsigned long long const ts0 = p.dbg ? gt() : 0ull;
    unsigned long long ts_own = 0, ts_peer = 0;
    int const parity = *reinterpret_cast<int volatile*>(p.flags + fidx) & 1;
    int64_t const half = (int64_t) p.maxt * p.world * (H / 2);
    int* const p0 = p.ex + parity * half + (int64_t) t * p.world * (H / 2) + c0 / 2;
    float s[8];
    float sq = 0.f;
    if (active)
    {
        int4 v[MAXW];
        uint32_t pending = (1u << p.world) - 1u;
        long long spins = 0;
        unsigned ns = 32;
        while (pending)
        {
#pragma unroll
            for (int r = 0; r < MAXW; ++r)
                if ((pending >> r) & 1u)
                    v[r] = ldv_i4(p0 + (int64_t) r * (H / 2));
#pragma unroll
            for (int r = 0; r < MAXW; ++r)
                if (((pending >> r) & 1u) && v[r].x != kEmpty && v[r].y != kEmpty && v[r].z != kEmpty
                    && v[r].w != kEmpty)
                {
                    pending &= ~(1u << r);
                    if (p.dbg && threadIdx.x == 0)
                    {
                        if (r == p.rank)
                            ts_own = gt();
                        else
                            ts_peer = gt();
                    }
                }
            if (++spins > (1LL << 31))
                __trap();
            if (pending && p.poll_mode == 1)
            {
                __nanosleep(ns);
                ns = ns < 256 ? ns * 2 : 256;
            }
        }
#pragma unroll
        for (int r = 0; r < MAXW; ++r)
        {
            if (r >= p.world)
                break;
            *reinterpret_cast<int4*>(p0 + (int64_t) r * (H / 2)) = make_int4(kEmpty, kEmpty, kEmpty, kEmpty);
            uint32_t const ww[4] = {(uint32_t) v[r].x, (uint32_t) v[r].y, (uint32_t) v[r].z, (uint32_t) v[r].w};
#pragma unroll
            for (int q = 0; q < 4; ++q)
            {
                float const lo = __uint_as_float(ww[q] << 16), hi = __uint_as_float(ww[q] & 0xFFFF0000u);
                s[2 * q] = r == 0 ? lo : add_f(s[2 * q], lo);
                s[2 * q + 1] = r == 0 ? hi : add_f(s[2 * q + 1], hi);
            }
        }
        uint4 const u = residual_add8(s, rr, sq);
        *reinterpret_cast<uint4*>(p.out_res + (int64_t) t * H + c0) = u;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        sq = add_f(sq, __shfl_xor_sync(0xffffffffu, sq, o));
    if ((threadIdx.x & 31) == 0)
        warp_sq[threadIdx.x / 32] = sq;
    __syncthreads();
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    if (threadIdx.x == 0)
    {
        float tot = 0.f;
#pragma unroll
        for (int w = 0; w < TH / 32; ++w)
            tot = add_f(tot, warp_sq[w]);
        uint32_t const b = static_cast<uint32_t>(__cvta_generic_to_shared(&xbar));
        uint32_t const d = static_cast<uint32_t>(__cvta_generic_to_shared(&csq[crank]));
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(CL * 4) : "memory");
        for (int c = 0; c < CL; ++c)
        {
            uint32_t rd, rb;
            asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rd) : "r"(d), "r"(c));
            asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rb) : "r"(b), "r"(c));
            asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(rd),
                         "r"(__float_as_uint(tot)), "r"(rb)
                         : "memory");
        }
    }
    {
        uint32_t const b = static_cast<uint32_t>(__cvta_generic_to_shared(&xbar));
        uint32_t done = 0;
        while (!done)
            asm volatile(
                "{\n .reg .pred q;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 q, [%1], 0;\n selp.u32 "
                "%0, 1, 0, q;\n}"
                : "=r"(done)
                : "r"(b)
                : "memory");
    }
    // every CTA of the row has polled (each sent its sum after its poll): flip the quarter's parity once
    if (threadIdx.x == 0 && crank % (CL / 4) == 0 && t < p.rows)
        p.flags[fidx] = parity ^ 1;
    if (active)
    {
        float total = 0.f;
#pragma unroll
        for (int c = 0; c < CL; ++c)
            total = add_f(total, csq[c]);
        uint4 const n = norm8(s, inv_rms(total, p.eps), wraw, p.gemma);
        if (p.out_norm)
            *reinterpret_cast<uint4*>(p.out_norm + (int64_t) t * H + c0) = n;
    }
    if (p.out_q != nullptr)
        mxfp8_8(s, active, t, c0, p.out_q, p.out_sf, p.sf_k_tiles);
    if (p.dbg && threadIdx.x == 0)
    {
        unsigned const c = p.dcnt[blockIdx.x]++;
        unsigned long long* d = p.dbg + ((int64_t) (c & 15) * gridDim.x + blockIdx.x) * 4;
        d[0] = ts0;
        d[1] = ts_own;
        d[2] = ts_peer;
        d[3] = gt();
    }
}
