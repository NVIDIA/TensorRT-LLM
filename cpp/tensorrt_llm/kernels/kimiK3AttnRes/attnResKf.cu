/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

// Attention-residual mix + residual add + RMSNorm kernel for H = 7168 (kimi_k3_attn_res_add_rmsnorm_h7168),
// clang-formatted; added are only its header include, the PLACEHOLDER_KERNELS guard and the trtllm
// entry point at the end (invokeAttnResKfFwd; the original attnres_launch keeps internal linkage).
// Built without --use_fast_math, as validated (kimi_k3_attn_res_kf_src in CMakeLists.txt).

// Kimi K3 attention-residual mix + optional residual add + output RMSNorm, H = 7168, BF16.
//
// Register-resident kernel: each CTA owns 7168 / C columns of one token row (C = cluster size).
// All N candidate slices are loaded up front with 32-byte streaming loads, the per-candidate
// sum of squares and score dot products are reduced across the CTA (warp shuffles + smem) and
// across the cluster (DSMEM pushes + one cluster barrier), then the softmax mix and the output
// RMSNorm (second reduction) are applied in FP32 and the result is rounded to BF16 once.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "tensorrt_llm/kernels/kimiK3AttnRes/attnResKf.h"

#ifndef PLACEHOLDER_KERNELS // set_cuda_architectures found no sm_100-family architecture

namespace
{

constexpr int H = 7168;
constexpr int VEC = 16; // bf16 elements per thread per candidate (one 32-byte load)

struct alignas(32) U8
{
    uint32_t v[8];
};

__device__ __forceinline__ void ld_stream(U8& r, void const* p)
{
    asm volatile(
        "ld.global.nc.L1::no_allocate.L2::evict_first.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3]), "=r"(r.v[4]), "=r"(r.v[5]), "=r"(r.v[6]), "=r"(r.v[7])
        : "l"(p));
}

__device__ __forceinline__ void ld_keep(U8& r, void const* p)
{
    asm volatile(
        "ld.global.nc.L2::evict_last.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3]), "=r"(r.v[4]), "=r"(r.v[5]), "=r"(r.v[6]), "=r"(r.v[7])
        : "l"(p));
}

__device__ __forceinline__ void st_stream(void* p, const U8& r)
{
    asm volatile("st.global.L1::no_allocate.v8.u32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" ::"l"(p), "r"(r.v[0]), "r"(r.v[1]),
                 "r"(r.v[2]), "r"(r.v[3]), "r"(r.v[4]), "r"(r.v[5]), "r"(r.v[6]), "r"(r.v[7])
                 : "memory");
}

__device__ __forceinline__ float bf_lo(uint32_t w)
{
    return __uint_as_float(w << 16);
}

__device__ __forceinline__ float bf_hi(uint32_t w)
{
    return __uint_as_float(w & 0xffff0000u);
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi)
{
    uint32_t r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

__device__ __forceinline__ uint32_t cluster_ctarank()
{
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void cluster_arrive_relaxed()
{
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}

__device__ __forceinline__ void cluster_arrive_release()
{
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
}

__device__ __forceinline__ void cluster_wait()
{
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

__device__ __forceinline__ void st_cluster_f32(float const* local, uint32_t rank, float v)
{
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(local));
    uint32_t ra;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(a), "r"(rank));
    asm volatile("st.shared::cluster.f32 [%0], %1;" ::"r"(ra), "f"(v) : "memory");
}

__device__ __forceinline__ uint32_t mapa_u32(uint32_t a, uint32_t rank)
{
    uint32_t ra;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(a), "r"(rank));
    return ra;
}

__device__ __forceinline__ void st_async_f32(uint32_t remote_addr, float v, uint32_t remote_bar)
{
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];" ::"r"(remote_addr), "f"(v),
                 "r"(remote_bar)
                 : "memory");
}

__device__ __forceinline__ void mb_init(uint32_t bar, uint32_t count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

__device__ __forceinline__ void mb_expect_tx(uint32_t bar, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mb_wait_cluster(uint32_t bar, uint32_t parity)
{
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "WAITC_%=:\n"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1;\n"
        "@!P1 bra WAITC_%=;\n"
        "}\n" ::"r"(bar),
        "r"(parity)
        : "memory");
}

// Score pass on one packed bf16 pair: (s0, s1) += v * v and (d0, d1) += v * (wl, wh), packed FP32x2.
// The bf16 -> fp32 unpack lives inside the asm so it is not CSE'd with the mix pass (which would keep
// every candidate live as FP32 and blow up the register count).
__device__ __forceinline__ void stat2(uint32_t w, float wl, float wh, float& s0, float& s1, float& d0, float& d1)
{
    asm("{\n"
        ".reg .b32 lo, hi;\n"
        ".reg .b64 v, s, d, c;\n"
        "shl.b32 lo, %4, 16;\n"
        "and.b32 hi, %4, 0xFFFF0000;\n"
        "mov.b64 v, {lo, hi};\n"
        "mov.b64 s, {%0, %1};\n"
        "fma.rn.f32x2 s, v, v, s;\n"
        "mov.b64 {%0, %1}, s;\n"
        "mov.b64 c, {%5, %6};\n"
        "mov.b64 d, {%2, %3};\n"
        "fma.rn.f32x2 d, v, c, d;\n"
        "mov.b64 {%2, %3}, d;\n"
        "}"
        : "+f"(s0), "+f"(s1), "+f"(d0), "+f"(d1)
        : "r"(w), "f"(wl), "f"(wh));
}

// Mix pass on one packed bf16 pair: (a0, a1) += p * v.
__device__ __forceinline__ void mix2(uint32_t w, float p, float& a0, float& a1)
{
    asm("{\n"
        ".reg .b32 lo, hi;\n"
        ".reg .b64 v, a, q;\n"
        "and.b32 hi, %2, 0xFFFF0000;\n"
        "shl.b32 lo, %2, 16;\n"
        "mov.b64 v, {lo, hi};\n"
        "mov.b64 q, {%3, %3};\n"
        "mov.b64 a, {%0, %1};\n"
        "fma.rn.f32x2 a, q, v, a;\n"
        "mov.b64 {%0, %1}, a;\n"
        "}"
        : "+f"(a0), "+f"(a1)
        : "r"(w), "f"(p));
}

__device__ __forceinline__ void mb_wait_cta(uint32_t bar, uint32_t parity)
{
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "WAITA_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@!P1 bra WAITA_%=;\n"
        "}\n" ::"r"(bar),
        "r"(parity)
        : "memory");
}

__device__ __forceinline__ void unpack_pair(uint32_t w, float& lo, float& hi)
{
    asm("prmt.b32 %0, %2, 0, 0x1044;\n\t"
        "prmt.b32 %1, %2, 0, 0x3244;"
        : "=r"(*reinterpret_cast<uint32_t*>(&lo)), "=r"(*reinterpret_cast<uint32_t*>(&hi))
        : "r"(w));
}

__device__ __forceinline__ void ffma2(float& c0, float& c1, float a0, float a1, float b0, float b1)
{
    asm("{\n"
        ".reg .b64 a, b, c;\n"
        "mov.b64 a, {%2, %3};\n"
        "mov.b64 b, {%4, %5};\n"
        "mov.b64 c, {%0, %1};\n"
        "fma.rn.f32x2 c, a, b, c;\n"
        "mov.b64 {%0, %1}, c;\n"
        "}"
        : "+f"(c0), "+f"(c1)
        : "f"(a0), "f"(a1), "f"(b0), "f"(b1));
}

__host__ __device__ constexpr int next_pow2(int x)
{
    return x <= 1 ? 1 : 2 * next_pow2((x + 1) / 2);
}

__host__ __device__ constexpr int ilog2(int x)
{
    return x <= 1 ? 0 : 1 + ilog2(x / 2);
}

// Reduce NP values across a warp; afterwards lane L holds the warp sum of value L >> (5 - log2 NP).
template <int n, int o, int NP>
__device__ __forceinline__ void warp_multi(float (&a)[NP], int lane)
{
    if constexpr (o >= 1)
    {
        if constexpr (n > 1)
        {
            constexpr int h = n / 2;
            bool const up = (lane & o) != 0;
#pragma unroll
            for (int k = 0; k < h; ++k)
            {
                float const send = up ? a[k] : a[k + h];
                float const keep = up ? a[k + h] : a[k];
                a[k] = keep + __shfl_xor_sync(0xffffffffu, send, o);
            }
            warp_multi<h, o / 2, NP>(a, lane);
        }
        else
        {
            a[0] += __shfl_xor_sync(0xffffffffu, a[0], o);
            warp_multi<1, o / 2, NP>(a, lane);
        }
    }
}

__device__ __forceinline__ float warp_sum(float v)
{
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

struct Params
{
    __nv_bfloat16 const* layer;
    __nv_bfloat16 const* add;
    __nv_bfloat16 const* block;
    __nv_bfloat16 const* res_w;
    __nv_bfloat16 const* rms_w;
    __nv_bfloat16 const* out_w;
    __nv_bfloat16* updated;
    __nv_bfloat16* out;
    float eps1;
    float eps2;
    int M;
    int async_xch;
    int gram;
    int dec_mode;   // decode kernel v2: 0 = LDG, 1 = TMA snapshots
    int gram_max;   // decode kernel v2: largest N using the Gram exchange
    int ring_slots; // ring kernel: slots in use (<= RING_S)
    int hold;       // ring kernel: keep block candidates in smem until the mix (re-read)
    int pace;       // producer: issue row j only after slot (pace - 1) of row j-1 landed (0: off)
};

template <int N, bool ADD, int C, int T>
__global__ void __launch_bounds__(T) attnres_reg_kernel(const Params p)
{
    constexpr int CHUNKS = H / VEC / C; // 16-element chunks per CTA per candidate
    static_assert(CHUNKS <= T, "one chunk per thread");
    constexpr int W = T / 32;
    constexpr int NV = 2 * N;         // (sumsq, dot) per candidate
    constexpr int NP = next_pow2(NV); // padded for the warp reduction
    constexpr int SH = 5 - ilog2(NP); // lane >> SH = value index after warp_multi
    static_assert(NP <= 32, "too many candidates");

    __shared__ __align__(16) float red[W][NP];
    __shared__ __align__(16) float xch[C][NP];
    __shared__ __align__(16) float tot_s[NP];
    __shared__ float red2[W];
    __shared__ float xch2[C];
    __shared__ __align__(8) uint64_t xbar[2];

    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    const uint32_t rank = (C > 1) ? cluster_ctarank() : 0u;
    int const row = blockIdx.x / C;
    bool const active = tid < CHUNKS;
    int const col = (static_cast<int>(rank) * CHUNKS + tid) * VEC;

    const uint32_t xb1 = static_cast<uint32_t>(__cvta_generic_to_shared(&xbar[0]));
    const uint32_t xb2 = xb1 + 8;
    if constexpr (C > 1)
    {
        if (p.async_xch && tid == 0)
        {
            mb_init(xb1, 1);
            mb_init(xb2, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            mb_expect_tx(xb1, C * NV * 4);
            mb_expect_tx(xb2, C * 4);
        }
        // all threads arrive; the barrier inits are published by fence.mbarrier_init above. Waited on
        // (acquire) before the first DSMEM access.
        if (p.async_xch)
            cluster_arrive_relaxed();
        else
            cluster_arrive_release();
    }

    U8 x[N];
    U8 xadd;
    U8 rw, sw, ow;
    if (active)
    {
#pragma unroll
        for (int i = 0; i < N - 1; ++i)
            ld_stream(x[i], p.block + (static_cast<size_t>(i) * p.M + row) * H + col);
        ld_stream(x[N - 1], p.layer + static_cast<size_t>(row) * H + col);
        if constexpr (ADD)
            ld_stream(xadd, p.add + static_cast<size_t>(row) * H + col);
        ld_keep(rw, p.rms_w + col);
        ld_keep(sw, p.res_w + col);
        ld_keep(ow, p.out_w + col);
    }
    else
    {
#pragma unroll
        for (int i = 0; i < N; ++i)
#pragma unroll
            for (int e = 0; e < 8; ++e)
                x[i].v[e] = 0u;
#pragma unroll
        for (int e = 0; e < 8; ++e)
        {
            xadd.v[e] = 0u;
            rw.v[e] = 0u;
            sw.v[e] = 0u;
            ow.v[e] = 0u;
        }
    }

    if constexpr (ADD)
    {
#pragma unroll
        for (int e = 0; e < 8; ++e)
        {
            float const lo = __fadd_rn(bf_lo(x[N - 1].v[e]), bf_lo(xadd.v[e]));
            float const hi = __fadd_rn(bf_hi(x[N - 1].v[e]), bf_hi(xadd.v[e]));
            x[N - 1].v[e] = pack_bf16(lo, hi);
        }
    }
    // updated = layer + add (or the layer_residual copy when has_add == 0, as the reference clones it)
    if (active)
        st_stream(p.updated + static_cast<size_t>(row) * H + col, x[N - 1]);

    // score weights in fp32: rms_w * res_w
    float scw[VEC];
#pragma unroll
    for (int e = 0; e < 8; ++e)
    {
        scw[2 * e] = bf_lo(rw.v[e]) * bf_lo(sw.v[e]);
        scw[2 * e + 1] = bf_hi(rw.v[e]) * bf_hi(sw.v[e]);
    }

    float a[NP];
#pragma unroll
    for (int j = 0; j < NP; ++j)
        a[j] = 0.f;
#pragma unroll
    for (int i = 0; i < N; ++i)
    {
        float ss0 = 0.f, ss1 = 0.f, d0 = 0.f, d1 = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e)
            stat2(x[i].v[e], scw[2 * e], scw[2 * e + 1], ss0, ss1, d0, d1);
        a[2 * i] = ss0 + ss1;
        a[2 * i + 1] = d0 + d1;
    }

    // ---- reduction 1: per-candidate sumsq and dot ----
    warp_multi<NP, 16, NP>(a, lane);
    if ((lane & ((1 << SH) - 1)) == 0)
        red[warp][lane >> SH] = a[0];
    __syncthreads();

    float tot[NP];
    if constexpr (C == 1)
    {
        // per-candidate totals -> logits in threads 0..N-1 (tot[i] = logit_i below)
        if (tid < N)
        {
            float ss = 0.f, d = 0.f;
#pragma unroll
            for (int w = 0; w < W; ++w)
            {
                ss += red[w][2 * tid];
                d += red[w][2 * tid + 1];
            }
            tot_s[tid] = d * rsqrtf(ss * (1.0f / H) + p.eps1);
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < N; j += 4)
        {
            const float4 t = *reinterpret_cast<float4 const*>(&tot_s[j]);
            tot[j] = t.x;
            if (j + 1 < NP)
                tot[j + 1] = t.y;
            if (j + 2 < NP)
                tot[j + 2] = t.z;
            if (j + 3 < NP)
                tot[j + 3] = t.w;
        }
    }
    else
    {
        cluster_wait(); // all CTAs of the cluster are running: DSMEM stores are legal
        if (tid < NV)
        {
            float s = 0.f;
#pragma unroll
            for (int w = 0; w < W; ++w)
                s += red[w][tid];
            if (p.async_xch)
            {
                const uint32_t la = static_cast<uint32_t>(__cvta_generic_to_shared(&xch[rank][tid]));
#pragma unroll
                for (int r = 0; r < C; ++r)
                    st_async_f32(mapa_u32(la, r), s, mapa_u32(xb1, r));
            }
            else
            {
#pragma unroll
                for (int r = 0; r < C; ++r)
                    st_cluster_f32(&xch[rank][tid], r, s);
            }
        }
        if (p.async_xch)
        {
            mb_wait_cta(xb1, 0);
        }
        else
        {
            cluster_arrive_release();
            cluster_wait();
        }
#pragma unroll
        for (int j = 0; j < NP; ++j)
            tot[j] = 0.f;
#pragma unroll
        for (int r = 0; r < C; ++r)
        {
#pragma unroll
            for (int j = 0; j < NP; j += 4)
            {
                const float4 t = *reinterpret_cast<float4 const*>(&xch[r][j]);
                tot[j] += t.x;
                tot[j + 1] += t.y;
                tot[j + 2] += t.z;
                tot[j + 3] += t.w;
            }
        }
    }

    // ---- softmax over the candidates ----
    float lg[N];
    float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < N; ++i)
    {
        if constexpr (C == 1)
        {
            lg[i] = tot[i]; // logits computed above
        }
        else
        {
            float const rs = rsqrtf(tot[2 * i] * (1.0f / H) + p.eps1);
            lg[i] = tot[2 * i + 1] * rs;
        }
        mx = fmaxf(mx, lg[i]);
    }
    float se = 0.f;
#pragma unroll
    for (int i = 0; i < N; ++i)
    {
        lg[i] = __expf(lg[i] - mx);
        se += lg[i];
    }
    float const inv = 1.0f / se;

    // ---- mix ----
    float acc[VEC];
#pragma unroll
    for (int e = 0; e < VEC; ++e)
        acc[e] = 0.f;
#pragma unroll
    for (int i = 0; i < N; ++i)
    {
        float const pi = lg[i] * inv;
#pragma unroll
        for (int e = 0; e < 8; ++e)
        {
            mix2(x[i].v[e], pi, acc[2 * e], acc[2 * e + 1]);
        }
    }
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < VEC; ++e)
        ss = fmaf(acc[e], acc[e], ss);

    // ---- reduction 2: sumsq of the mix ----
    ss = warp_sum(ss);
    if (lane == 0)
        red2[warp] = ss;
    __syncthreads();
    float ss_tot = 0.f;
    if constexpr (C == 1)
    {
#pragma unroll
        for (int w = 0; w < W; ++w)
            ss_tot += red2[w];
    }
    else
    {
        if (tid == 0)
        {
            float s = 0.f;
#pragma unroll
            for (int w = 0; w < W; ++w)
                s += red2[w];
            if (p.async_xch)
            {
                const uint32_t la = static_cast<uint32_t>(__cvta_generic_to_shared(&xch2[rank]));
#pragma unroll
                for (int r = 0; r < C; ++r)
                    st_async_f32(mapa_u32(la, r), s, mapa_u32(xb2, r));
            }
            else
            {
#pragma unroll
                for (int r = 0; r < C; ++r)
                    st_cluster_f32(&xch2[rank], r, s);
            }
        }
        if (p.async_xch)
        {
            mb_wait_cta(xb2, 0);
        }
        else
        {
            cluster_arrive_release();
            cluster_wait();
        }
#pragma unroll
        for (int r = 0; r < C; ++r)
            ss_tot += xch2[r];
    }
    float const r2 = rsqrtf(ss_tot * (1.0f / H) + p.eps2);

    if (active)
    {
        U8 o;
#pragma unroll
        for (int e = 0; e < 8; ++e)
        {
            float const lo = (acc[2 * e] * r2) * bf_lo(ow.v[e]);
            float const hi = (acc[2 * e + 1] * r2) * bf_hi(ow.v[e]);
            o.v[e] = pack_bf16(lo, hi);
        }
        st_stream(p.out + static_cast<size_t>(row) * H + col, o);
    }
}

// ---------------------------------------------------------------------------------------------
// Persistent TMA-ring kernel for large M. One producer warp streams whole candidate rows
// (14336 B each) with cp.async.bulk into a ring of RING_S shared-memory slots; 14 consumer warps
// copy their 2 x 16 B of each slot into registers, release the slot immediately and keep the
// N candidates of the current token in registers. The ring therefore stays full of in-flight
// loads while the consumers reduce, mix and normalize.
// ---------------------------------------------------------------------------------------------
#ifndef RING_SLOTS
#define RING_SLOTS 15
#endif
constexpr int RING_S = RING_SLOTS;
constexpr int RING_CW = 14;                // consumer warps
constexpr int RING_CT = RING_CW * 32;      // 448 consumer threads, 16 elements each
constexpr int RING_THREADS = RING_CT + 32; // + producer warp
constexpr int ROW_BYTES = H * 2;
constexpr int RING_SMEM = RING_S * ROW_BYTES + 2 * RING_S * 8;

__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity)
{
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@!P1 bra WAIT_%=;\n"
        "}\n" ::"r"(bar),
        "r"(parity)
        : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint32_t bar)
{
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}

__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}

__device__ __forceinline__ void bulk_g2s(uint32_t dst, void const* src, uint32_t bytes, uint32_t bar, uint64_t policy)
{
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], "
        "%2, [%3], %4;" ::"r"(dst),
        "l"(src), "r"(bytes), "r"(bar), "l"(policy)
        : "memory");
}

__device__ __forceinline__ void consumer_sync()
{
    asm volatile("bar.sync 1, %0;" ::"n"(RING_CT) : "memory");
}

__device__ __forceinline__ uint4 ld_shared16(uint32_t a)
{
    uint4 r;
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "r"(a)
                 : "memory");
    return r;
}

__device__ __forceinline__ uint4 ldg16_keep(void const* p)
{
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "l"(p));
    return r;
}

__device__ __forceinline__ void stg16(void* p, uint4 const& v)
{
    asm volatile(
        "st.global.L1::no_allocate.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
        : "memory");
}

__device__ __forceinline__ uint32_t u4get(uint4 const& v, int e)
{
    return e == 0 ? v.x : e == 1 ? v.y : e == 2 ? v.z : v.w;
}

template <int N, bool ADD, bool HOLD>
__global__ void __launch_bounds__(RING_THREADS, 1) attnres_ring_kernel(const Params p)
{
    constexpr int K = N - 1;
    constexpr int L = ADD ? K + 2 : K + 1; // loads per row
    constexpr int NV = 2 * N;
    constexpr int NP = next_pow2(NV);
    constexpr int SH = 5 - ilog2(NP);
    constexpr int HALF = H / 2;

    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ __align__(16) float red[RING_CW][NP];
    __shared__ __align__(16) float tot_s[NP];
    __shared__ __align__(16) float red2[16];

    const uint32_t smem_base = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    const uint32_t full_bar = smem_base + RING_S * ROW_BYTES;
    const uint32_t empty_bar = full_bar + RING_S * 8;
    int const nslots = p.ring_slots; // slots in use (<= RING_S)
    int const tid = threadIdx.x;
    int const warp = tid >> 5;
    int const lane = tid & 31;
    int const M = p.M;
    // rows of this CTA (round-robin)
    int const row_begin = blockIdx.x, row_end = M, row_step = gridDim.x;

    if (tid == 0)
    {
        for (int s = 0; s < RING_S; ++s)
        {
            mbar_init(full_bar + 8 * s, 1);
            mbar_init(empty_bar + 8 * s, RING_CW);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if (warp == RING_CW)
    {
        // ---------------- producer ----------------
        if (lane == 0)
        {
            uint64_t policy;
            asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
            int slot = 0;
            uint32_t phase = 0;
            // Decode pacing: request row j+1 only after slot `pace_l` of row j has landed, so each CTA's
            // rows arrive in order and the compute of one row overlaps the next row's loads.
            int const pace_l = p.pace > 0 ? (p.pace - 1 < L - 1 ? p.pace - 1 : L - 1) : -1;
            int prev_slot = -1;
            uint32_t prev_phase = 0;
            for (int row = row_begin; row < row_end; row += row_step)
            {
                if (pace_l >= 0 && prev_slot >= 0)
                    mbar_wait(full_bar + 8 * prev_slot, prev_phase);
#pragma unroll
                for (int l = 0; l < L; ++l)
                {
                    if (l == pace_l)
                    {
                        prev_slot = slot;
                        prev_phase = phase;
                    }
                    __nv_bfloat16 const* src = (l < K) ? p.block + (static_cast<size_t>(l) * M + row) * H
                        : (l == K)                     ? p.layer + static_cast<size_t>(row) * H
                                                       : p.add + static_cast<size_t>(row) * H;
                    mbar_wait(empty_bar + 8 * slot, phase ^ 1u);
                    mbar_expect_tx(full_bar + 8 * slot, ROW_BYTES);
                    bulk_g2s(smem_base + slot * ROW_BYTES, src, ROW_BYTES, full_bar + 8 * slot, policy);
                    if (++slot == nslots)
                    {
                        slot = 0;
                        phase ^= 1u;
                    }
                }
            }
        }
        return;
    }

    // ---------------- consumers ----------------
    int const c0 = 8 * tid;        // first 8 columns
    int const c1 = HALF + 8 * tid; // second 8 columns
    float scw[16];
    uint4 ow0, ow1;
    {
        const uint4 r0 = ldg16_keep(p.rms_w + c0), r1 = ldg16_keep(p.rms_w + c1);
        const uint4 s0 = ldg16_keep(p.res_w + c0), s1 = ldg16_keep(p.res_w + c1);
        ow0 = ldg16_keep(p.out_w + c0);
        ow1 = ldg16_keep(p.out_w + c1);
#pragma unroll
        for (int e = 0; e < 4; ++e)
        {
            scw[2 * e] = bf_lo(u4get(r0, e)) * bf_lo(u4get(s0, e));
            scw[2 * e + 1] = bf_hi(u4get(r0, e)) * bf_hi(u4get(s0, e));
            scw[8 + 2 * e] = bf_lo(u4get(r1, e)) * bf_lo(u4get(s1, e));
            scw[8 + 2 * e + 1] = bf_hi(u4get(r1, e)) * bf_hi(u4get(s1, e));
        }
    }
    const uint32_t my_off0 = 16u * tid;
    const uint32_t my_off1 = HALF * 2u + 16u * tid;

    int slot = 0;
    uint32_t phase = 0;
    for (int row = row_begin; row < row_end; row += row_step)
    {
        uint4 x[N][2];
        float a[NP];
#pragma unroll
        for (int j = 0; j < NP; ++j)
            a[j] = 0.f;

        int held_slot = 0;        // layer slot kept until the add slot has been consumed (ADD only)
        int bslot[K > 0 ? K : 1]; // HOLD: ring slots of the block candidates (released after the mix)
#pragma unroll
        for (int l = 0; l < L; ++l)
        {
            mbar_wait(full_bar + 8 * slot, phase);
            const uint32_t sb = smem_base + slot * ROW_BYTES;
            const uint4 v0 = ld_shared16(sb + my_off0);
            const uint4 v1 = ld_shared16(sb + my_off1);
            int ci = -1; // candidate whose stats are complete after this load
            if (l < K)
            {
                x[l][0] = v0;
                x[l][1] = v1;
                ci = l;
                if (HOLD)
                    bslot[l] = slot;
            }
            else if (l == K)
            {
                x[K][0] = v0;
                x[K][1] = v1;
                if (!ADD)
                    ci = K;
            }
            else
            {
                uint32_t* u0 = reinterpret_cast<uint32_t*>(&x[K][0]);
                uint32_t* u1 = reinterpret_cast<uint32_t*>(&x[K][1]);
#pragma unroll
                for (int e = 0; e < 4; ++e)
                {
                    const uint32_t a0 = u4get(v0, e), a1 = u4get(v1, e);
                    u0[e] = pack_bf16(__fadd_rn(bf_lo(u0[e]), bf_lo(a0)), __fadd_rn(bf_hi(u0[e]), bf_hi(a0)));
                    u1[e] = pack_bf16(__fadd_rn(bf_lo(u1[e]), bf_lo(a1)), __fadd_rn(bf_hi(u1[e]), bf_hi(a1)));
                }
                ci = K;
            }
            if (ci >= 0)
            {
                uint32_t const* w0 = reinterpret_cast<uint32_t const*>(&x[ci][0]);
                uint32_t const* w1 = reinterpret_cast<uint32_t const*>(&x[ci][1]);
                float ss0 = 0.f, ss1 = 0.f, d0 = 0.f, d1 = 0.f;
#pragma unroll
                for (int e = 0; e < 4; ++e)
                {
                    stat2(w0[e], scw[2 * e], scw[2 * e + 1], ss0, ss1, d0, d1);
                    stat2(w1[e], scw[8 + 2 * e], scw[8 + 2 * e + 1], ss0, ss1, d0, d1);
                }
                a[2 * ci] = ss0 + ss1;
                a[2 * ci + 1] = d0 + d1;
            }
            // Release the slot only after its data has been consumed by arithmetic, so the shared
            // loads have certainly completed before the producer may overwrite the slot.
            if (ADD && l == K)
            {
                held_slot = slot;
            }
            else if (HOLD && l < K)
            {
                // kept in shared memory until the mix re-reads it
            }
            else
            {
                __syncwarp();
                if (lane == 0)
                {
                    mbar_arrive(empty_bar + 8 * slot);
                    if (ADD && l == K + 1)
                        mbar_arrive(empty_bar + 8 * held_slot);
                }
            }
            if (++slot == nslots)
            {
                slot = 0;
                phase ^= 1u;
            }
        }
        // updated (layer + add, or the layer copy when has_add == 0)
        {
            __nv_bfloat16* up = p.updated + static_cast<size_t>(row) * H;
            stg16(up + c0, x[K][0]);
            stg16(up + c1, x[K][1]);
        }

        // ---- reduction 1 ----
        warp_multi<NP, 16, NP>(a, lane);
        if ((lane & ((1 << SH) - 1)) == 0)
            red[warp][lane >> SH] = a[0];
        consumer_sync();
        if (tid < N)
        { // per-candidate totals -> logit
            float ss = 0.f, d = 0.f;
#pragma unroll
            for (int w = 0; w < RING_CW; ++w)
            {
                ss += red[w][2 * tid];
                d += red[w][2 * tid + 1];
            }
            tot_s[tid] = d * rsqrtf(ss * (1.0f / H) + p.eps1);
        }
        consumer_sync();
        float lg[(N + 3) & ~3];
#pragma unroll
        for (int j = 0; j < N; j += 4)
        {
            const float4 t = *reinterpret_cast<float4 const*>(&tot_s[j]);
            lg[j] = t.x;
            lg[j + 1] = t.y;
            lg[j + 2] = t.z;
            lg[j + 3] = t.w;
        }
        float mx = lg[0];
#pragma unroll
        for (int i = 1; i < N; ++i)
            mx = fmaxf(mx, lg[i]);
        float se = 0.f;
#pragma unroll
        for (int i = 0; i < N; ++i)
        {
            lg[i] = __expf(lg[i] - mx);
            se += lg[i];
        }
        {
            float const inv = 1.0f / se;
#pragma unroll
            for (int i = 0; i < N; ++i)
                lg[i] *= inv;
            se = 1.0f;
        }

        // acc = sum_i e_i v_i (unnormalised); the 1/se cancels in the output RMSNorm except in eps
        float acc[16];
#pragma unroll
        for (int e = 0; e < 16; ++e)
            acc[e] = 0.f;
#pragma unroll
        for (int i = 0; i < N; ++i)
        {
            uint4 m0 = x[i][0], m1 = x[i][1];
            if (HOLD && i < K)
            {
                const uint32_t sb = smem_base + bslot[i] * ROW_BYTES;
                m0 = ld_shared16(sb + my_off0);
                m1 = ld_shared16(sb + my_off1);
            }
            uint32_t const* w0 = reinterpret_cast<uint32_t const*>(&m0);
            uint32_t const* w1 = reinterpret_cast<uint32_t const*>(&m1);
#pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                mix2(w0[e], lg[i], acc[2 * e], acc[2 * e + 1]);
                mix2(w1[e], lg[i], acc[8 + 2 * e], acc[8 + 2 * e + 1]);
            }
            if (HOLD && i < K)
            {
                __syncwarp();
                if (lane == 0)
                    mbar_arrive(empty_bar + 8 * bslot[i]);
            }
        }
        float sq0 = 0.f, sq1 = 0.f;
#pragma unroll
        for (int e = 0; e < 16; e += 2)
            ffma2(sq0, sq1, acc[e], acc[e + 1], acc[e], acc[e + 1]);
        float const ss = warp_sum(sq0 + sq1);
        if (lane == 0)
            red2[warp] = ss;
        consumer_sync();
        float ss_tot = 0.f;
#pragma unroll
        for (int w = 0; w < 16; w += 4)
        {
            const float4 t = *reinterpret_cast<float4 const*>(&red2[w]);
            ss_tot += (w + 0 < RING_CW ? t.x : 0.f) + (w + 1 < RING_CW ? t.y : 0.f) + (w + 2 < RING_CW ? t.z : 0.f)
                + (w + 3 < RING_CW ? t.w : 0.f);
        }
        float const r2 = rsqrtf(ss_tot * (1.0f / H) + p.eps2 * se * se);

        uint4 o0, o1;
        uint32_t* q0 = reinterpret_cast<uint32_t*>(&o0);
        uint32_t* q1 = reinterpret_cast<uint32_t*>(&o1);
#pragma unroll
        for (int e = 0; e < 4; ++e)
        {
            q0[e] = pack_bf16((acc[2 * e] * r2) * bf_lo(u4get(ow0, e)), (acc[2 * e + 1] * r2) * bf_hi(u4get(ow0, e)));
            q1[e] = pack_bf16(
                (acc[8 + 2 * e] * r2) * bf_lo(u4get(ow1, e)), (acc[8 + 2 * e + 1] * r2) * bf_hi(u4get(ow1, e)));
        }
        __nv_bfloat16* op = p.out + static_cast<size_t>(row) * H;
        stg16(op + c0, o0);
        stg16(op + c1, o1);
    }
}

int num_sms()
{
    static int n = 0;
    if (n == 0)
    {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
        if (n <= 0)
            n = 148;
    }
    return n;
}

template <int N, bool ADD>
void launch_ring(Params const& p, cudaStream_t stream)
{
    auto kern = p.hold ? attnres_ring_kernel<N, ADD, true> : attnres_ring_kernel<N, ADD, false>;
    static bool init = false;
    if (!init)
    {
        cudaFuncSetAttribute(attnres_ring_kernel<N, ADD, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, RING_SMEM);
        cudaFuncSetAttribute(
            attnres_ring_kernel<N, ADD, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, RING_SMEM);
        init = true;
    }
    int const grid = p.M < num_sms() ? p.M : num_sms();
    kern<<<grid, RING_THREADS, RING_SMEM, stream>>>(p);
}

// ---------------------------------------------------------------------------------------------
// Cluster decode kernel with a single cluster exchange. Besides the score dots d_i, every CTA
// accumulates the candidate Gram matrix G_ij = sum_h v_ih v_jh over its columns; after one DSMEM
// exchange, sum_h mixed_h^2 = p^T G p, so the output RMSNorm needs no second reduction.
// ---------------------------------------------------------------------------------------------
template <int N, bool ADD, int C, int T>
__global__ void __launch_bounds__(T) attnres_gram_kernel(const Params p)
{
    constexpr int CHUNKS = H / VEC / C;
    static_assert(CHUNKS <= T, "one chunk per thread");
    static_assert(C > 1, "cluster kernel");
    constexpr int W = T / 32;
    constexpr int NG = N * (N + 1) / 2; // upper-triangular Gram entries
    constexpr int NV = NG + N;          // + score dots
    constexpr int NP = next_pow2(NV);
    static_assert(NP <= 64, "too many values");
    constexpr int R = NP >= 32 ? NP / 32 : 1; // values per lane after warp_multi
    constexpr int SH = NP >= 32 ? 0 : 5 - ilog2(NP);

    __shared__ __align__(16) float red[W][NP];
    __shared__ __align__(16) float xch[C][NP];
    __shared__ __align__(16) float tot_s[NP];
    __shared__ __align__(8) uint64_t xbar;

    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    const uint32_t rank = cluster_ctarank();
    int const row = blockIdx.x / C;
    bool const active = tid < CHUNKS;
    int const col = (static_cast<int>(rank) * CHUNKS + tid) * VEC;

    const uint32_t xb = static_cast<uint32_t>(__cvta_generic_to_shared(&xbar));
    if (tid == 0)
    {
        mb_init(xb, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        mb_expect_tx(xb, C * NV * 4);
    }
    cluster_arrive_relaxed();

    U8 x[N];
    U8 xadd;
    U8 rw, sw, ow;
    if (active)
    {
#pragma unroll
        for (int i = 0; i < N - 1; ++i)
            ld_stream(x[i], p.block + (static_cast<size_t>(i) * p.M + row) * H + col);
        ld_stream(x[N - 1], p.layer + static_cast<size_t>(row) * H + col);
        if constexpr (ADD)
            ld_stream(xadd, p.add + static_cast<size_t>(row) * H + col);
        ld_keep(rw, p.rms_w + col);
        ld_keep(sw, p.res_w + col);
        ld_keep(ow, p.out_w + col);
    }
    else
    {
#pragma unroll
        for (int i = 0; i < N; ++i)
#pragma unroll
            for (int e = 0; e < 8; ++e)
                x[i].v[e] = 0u;
#pragma unroll
        for (int e = 0; e < 8; ++e)
        {
            xadd.v[e] = 0u;
            rw.v[e] = 0u;
            sw.v[e] = 0u;
            ow.v[e] = 0u;
        }
    }
    if constexpr (ADD)
    {
#pragma unroll
        for (int e = 0; e < 8; ++e)
        {
            float const lo = __fadd_rn(bf_lo(x[N - 1].v[e]), bf_lo(xadd.v[e]));
            float const hi = __fadd_rn(bf_hi(x[N - 1].v[e]), bf_hi(xadd.v[e]));
            x[N - 1].v[e] = pack_bf16(lo, hi);
        }
    }
    if (active)
        st_stream(p.updated + static_cast<size_t>(row) * H + col, x[N - 1]);

    // ---- Gram matrix + score dots over this thread's 16 columns (packed FP32x2 accumulators) ----
    float g0[NV], g1[NV];
#pragma unroll
    for (int j = 0; j < NV; ++j)
        g0[j] = g1[j] = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e)
    {
        float const w0 = bf_lo(rw.v[e]) * bf_lo(sw.v[e]);
        float const w1 = bf_hi(rw.v[e]) * bf_hi(sw.v[e]);
        float vl[N], vh[N];
#pragma unroll
        for (int i = 0; i < N; ++i)
            unpack_pair(x[i].v[e], vl[i], vh[i]);
#pragma unroll
        for (int i = 0; i < N; ++i)
        {
#pragma unroll
            for (int j = i; j < N; ++j)
            {
                int const gi = i * N - (i * (i - 1)) / 2 + (j - i);
                ffma2(g0[gi], g1[gi], vl[i], vh[i], vl[j], vh[j]);
            }
            ffma2(g0[NG + i], g1[NG + i], vl[i], vh[i], w0, w1);
        }
    }
    float a[NP];
#pragma unroll
    for (int j = 0; j < NP; ++j)
        a[j] = (j < NV) ? g0[j] + g1[j] : 0.f;

    // ---- one reduction: warp -> CTA -> cluster (DSMEM st.async + mbarrier) ----
    warp_multi<NP, 16, NP>(a, lane);
    if constexpr (NP >= 32)
    {
#pragma unroll
        for (int k = 0; k < R; ++k)
            red[warp][lane * R + k] = a[k];
    }
    else
    {
        if ((lane & ((1 << SH) - 1)) == 0)
            red[warp][lane >> SH] = a[0];
    }
    __syncthreads();
    cluster_wait(); // all CTAs of the cluster are running (mbarriers initialised)
    for (int j = tid; j < NV; j += T)
    {
        float s = 0.f;
#pragma unroll
        for (int w = 0; w < W; ++w)
            s += red[w][j];
        const uint32_t la = static_cast<uint32_t>(__cvta_generic_to_shared(&xch[rank][j]));
#pragma unroll
        for (int r = 0; r < C; ++r)
            st_async_f32(mapa_u32(la, r), s, mapa_u32(xb, r));
    }
    mb_wait_cta(xb, 0);
    for (int j = tid; j < NV; j += T)
    {
        float s = 0.f;
#pragma unroll
        for (int r = 0; r < C; ++r)
            s += xch[r][j];
        tot_s[j] = s;
    }
    __syncthreads();

    // ---- softmax over the candidates and the output scale from p^T G p ----
    float pr[N];
    float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < N; ++i)
    {
        int const gi = i * N - (i * (i - 1)) / 2;
        float const rs = rsqrtf(tot_s[gi] * (1.0f / H) + p.eps1);
        pr[i] = tot_s[NG + i] * rs;
        mx = fmaxf(mx, pr[i]);
    }
    float se = 0.f;
#pragma unroll
    for (int i = 0; i < N; ++i)
    {
        pr[i] = __expf(pr[i] - mx);
        se += pr[i];
    }
    float const inv = 1.0f / se;
#pragma unroll
    for (int i = 0; i < N; ++i)
        pr[i] *= inv;
    float msq = 0.f;
#pragma unroll
    for (int i = 0; i < N; ++i)
    {
        int const gi = i * N - (i * (i - 1)) / 2;
        float t = pr[i] * tot_s[gi];
#pragma unroll
        for (int j = i + 1; j < N; ++j)
            t = fmaf(2.0f * pr[j], tot_s[gi + (j - i)], t);
        msq = fmaf(pr[i], t, msq);
    }
    float const r2 = rsqrtf(fmaxf(msq, 0.f) * (1.0f / H) + p.eps2);

    // ---- mix and output ----
    float acc[VEC];
#pragma unroll
    for (int e = 0; e < VEC; ++e)
        acc[e] = 0.f;
#pragma unroll
    for (int i = 0; i < N; ++i)
    {
#pragma unroll
        for (int e = 0; e < 8; ++e)
            mix2(x[i].v[e], pr[i], acc[2 * e], acc[2 * e + 1]);
    }
    if (active)
    {
        U8 o;
#pragma unroll
        for (int e = 0; e < 8; ++e)
        {
            float const lo = (acc[2 * e] * r2) * bf_lo(ow.v[e]);
            float const hi = (acc[2 * e + 1] * r2) * bf_hi(ow.v[e]);
            o.v[e] = pack_bf16(lo, hi);
        }
        st_stream(p.out + static_cast<size_t>(row) * H + col, o);
    }
}

// ---------------------------------------------------------------------------------------------
// Decode kernel v2: each token row is split over a cluster of C CTAs (C up to 16), 8 columns
// (16 B) per thread. MODE 0: all candidates via 16-byte LDG; MODE 1: block snapshots via one
// cp.async.bulk per candidate slice into shared memory (layer / add rows via LDG).
// GRAM: one cluster exchange (Gram matrix -> sum(mixed^2) = p^T G p), else two exchanges.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ uint4 ldg16_stream(void const* p)
{
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "l"(p));
    return r;
}

template <int C>
struct DecCfg
{
    static constexpr int COLS = H / C;            // columns per CTA
    static constexpr int NT = COLS / 8;           // threads with data
    static constexpr int T = (NT + 31) / 32 * 32; // block size
};

template <int N, bool ADD, int C, int MODE, bool GRAM>
__global__ void __launch_bounds__(DecCfg<C>::T) attnres_dec_kernel(const Params p)
{
    constexpr int COLS = DecCfg<C>::COLS;
    constexpr int NT = DecCfg<C>::NT;
    constexpr int T = DecCfg<C>::T;
    constexpr int W = T / 32;
    constexpr int K = N - 1;
    constexpr int NG = GRAM ? N * (N + 1) / 2 : 0;
    constexpr int NV = GRAM ? NG + N : 2 * N;
    constexpr int NP = next_pow2(NV);
    static_assert(NP <= 64, "too many values");
    constexpr int R = NP >= 32 ? NP / 32 : 1;
    constexpr int SH = NP >= 32 ? 0 : 5 - ilog2(NP);
    constexpr bool TWO_LEVEL = (C * NP > 64);

    extern __shared__ __align__(128) uint8_t dsmem[]; // MODE 1: K candidate slices
    __shared__ __align__(16) float red[W][NP];
    __shared__ __align__(16) float xch[C][NP];
    __shared__ __align__(16) float tot_s[NP];
    __shared__ float red2[W];
    __shared__ float xch2[C];
    __shared__ __align__(8) uint64_t bars[3]; // data, exchange 1, exchange 2

    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    const uint32_t rank = cluster_ctarank();
    int const row = blockIdx.x / C;
    bool const active = tid < NT;
    int const col0 = static_cast<int>(rank) * COLS;
    int const col = col0 + 8 * tid;
    const uint32_t dbar = static_cast<uint32_t>(__cvta_generic_to_shared(&bars[0]));
    const uint32_t xb1 = dbar + 8, xb2 = dbar + 16;
    const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(dsmem));

    if (tid == 0)
    {
        mb_init(dbar, 1);
        mb_init(xb1, 1);
        mb_init(xb2, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        mb_expect_tx(xb1, C * NV * 4);
        if (!GRAM)
            mb_expect_tx(xb2, C * 4);
        if constexpr (MODE >= 1)
        {
            uint64_t policy;
            asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
            mbar_expect_tx(dbar, K * COLS * 2);
#pragma unroll
            for (int i = 0; i < K; ++i)
                bulk_g2s(sbase + i * COLS * 2, p.block + (static_cast<size_t>(i) * p.M + row) * H + col0, COLS * 2,
                    dbar, policy);
        }
    }
    cluster_arrive_relaxed();

    uint4 x[N];
    if (active)
    {
        if constexpr (MODE == 0)
        {
#pragma unroll
            for (int i = 0; i < K; ++i)
                x[i] = ldg16_stream(p.block + (static_cast<size_t>(i) * p.M + row) * H + col);
        }
        x[K] = ldg16_stream(p.layer + static_cast<size_t>(row) * H + col);
    }
    uint4 xa = make_uint4(0u, 0u, 0u, 0u);
    if constexpr (ADD)
    {
        if (active)
            xa = ldg16_stream(p.add + static_cast<size_t>(row) * H + col);
    }
    uint4 rw = make_uint4(0u, 0u, 0u, 0u), sw = rw, ow = rw;
    if (active)
    {
        rw = ldg16_keep(p.rms_w + col);
        sw = ldg16_keep(p.res_w + col);
        ow = ldg16_keep(p.out_w + col);
    }
    if (!active)
    {
#pragma unroll
        for (int i = 0; i < N; ++i)
            x[i] = make_uint4(0u, 0u, 0u, 0u);
    }
    if constexpr (ADD)
    {
        uint32_t* u = reinterpret_cast<uint32_t*>(&x[K]);
#pragma unroll
        for (int e = 0; e < 4; ++e)
        {
            const uint32_t a = u4get(xa, e);
            u[e] = pack_bf16(__fadd_rn(bf_lo(u[e]), bf_lo(a)), __fadd_rn(bf_hi(u[e]), bf_hi(a)));
        }
    }
    if (active)
        stg16(p.updated + static_cast<size_t>(row) * H + col, x[K]);
    float wl[4], wh[4];
#pragma unroll
    for (int e = 0; e < 4; ++e)
    {
        wl[e] = bf_lo(u4get(rw, e)) * bf_lo(u4get(sw, e));
        wh[e] = bf_hi(u4get(rw, e)) * bf_hi(u4get(sw, e));
    }
    if constexpr (MODE >= 1)
    {
        __syncthreads(); // barrier init visible to every waiting thread
        mb_wait_cta(dbar, 0);
        if (active && (MODE == 1 || GRAM))
        {
#pragma unroll
            for (int i = 0; i < K; ++i)
                x[i] = ld_shared16(sbase + (i * COLS + 8 * tid) * 2);
        }
    }
    // MODE 2: block candidates are re-read from shared memory in each pass (low register count)
    auto xv = [&](int i) -> uint4
    {
        if constexpr (MODE == 2 && !GRAM)
        {
            if (i < K)
                return active ? ld_shared16(sbase + (i * COLS + 8 * tid) * 2) : make_uint4(0u, 0u, 0u, 0u);
        }
        return x[i];
    };

    // ---- per-thread partial statistics ----
    float a[NP];
#pragma unroll
    for (int j = 0; j < NP; ++j)
        a[j] = 0.f;
    if constexpr (GRAM)
    {
        float g0[NV], g1[NV];
#pragma unroll
        for (int j = 0; j < NV; ++j)
            g0[j] = g1[j] = 0.f;
#pragma unroll
        for (int e = 0; e < 4; ++e)
        {
            float vl[N], vh[N];
#pragma unroll
            for (int i = 0; i < N; ++i)
                unpack_pair(u4get(x[i], e), vl[i], vh[i]);
#pragma unroll
            for (int i = 0; i < N; ++i)
            {
#pragma unroll
                for (int j = i; j < N; ++j)
                {
                    int const gi = i * N - (i * (i - 1)) / 2 + (j - i);
                    ffma2(g0[gi], g1[gi], vl[i], vh[i], vl[j], vh[j]);
                }
                ffma2(g0[NG + i], g1[NG + i], vl[i], vh[i], wl[e], wh[e]);
            }
        }
#pragma unroll
        for (int j = 0; j < NV; ++j)
            a[j] = g0[j] + g1[j];
    }
    else
    {
#pragma unroll
        for (int i = 0; i < N; ++i)
        {
            const uint4 xi = xv(i);
            float ss0 = 0.f, ss1 = 0.f, d0 = 0.f, d1 = 0.f;
#pragma unroll
            for (int e = 0; e < 4; ++e)
                stat2(u4get(xi, e), wl[e], wh[e], ss0, ss1, d0, d1);
            a[2 * i] = ss0 + ss1;
            a[2 * i + 1] = d0 + d1;
        }
    }

    // ---- exchange 1: warp -> CTA -> cluster ----
    warp_multi<NP, 16, NP>(a, lane);
    if constexpr (NP >= 32)
    {
#pragma unroll
        for (int k = 0; k < R; ++k)
            red[warp][lane * R + k] = a[k];
    }
    else
    {
        if ((lane & ((1 << SH) - 1)) == 0)
            red[warp][lane >> SH] = a[0];
    }
    __syncthreads();
    cluster_wait();
    for (int j = tid; j < NV; j += T)
    {
        float sacc = 0.f;
#pragma unroll
        for (int w = 0; w < W; ++w)
            sacc += red[w][j];
        const uint32_t la = static_cast<uint32_t>(__cvta_generic_to_shared(&xch[rank][j]));
#pragma unroll
        for (int r = 0; r < C; ++r)
            st_async_f32(mapa_u32(la, r), sacc, mapa_u32(xb1, r));
    }
    mb_wait_cta(xb1, 0);
    float tot[NP];
    if constexpr (TWO_LEVEL)
    {
        for (int j = tid; j < NV; j += T)
        {
            float sacc = 0.f;
#pragma unroll
            for (int r = 0; r < C; ++r)
                sacc += xch[r][j];
            tot_s[j] = sacc;
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < NP; j += 4)
        {
            const float4 t4 = *reinterpret_cast<float4 const*>(&tot_s[j]);
            tot[j] = t4.x;
            tot[j + 1] = t4.y;
            tot[j + 2] = t4.z;
            tot[j + 3] = t4.w;
        }
    }
    else
    {
#pragma unroll
        for (int j = 0; j < NP; ++j)
            tot[j] = 0.f;
#pragma unroll
        for (int r = 0; r < C; ++r)
        {
#pragma unroll
            for (int j = 0; j < NP; j += 4)
            {
                const float4 t4 = *reinterpret_cast<float4 const*>(&xch[r][j]);
                tot[j] += t4.x;
                tot[j + 1] += t4.y;
                tot[j + 2] += t4.z;
                tot[j + 3] += t4.w;
            }
        }
    }

    // ---- softmax ----
    float pr[N];
    float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < N; ++i)
    {
        float const ssi = GRAM ? tot[i * N - (i * (i - 1)) / 2] : tot[2 * i];
        float const di = GRAM ? tot[NG + i] : tot[2 * i + 1];
        pr[i] = di * rsqrtf(ssi * (1.0f / H) + p.eps1);
        mx = fmaxf(mx, pr[i]);
    }
    float se = 0.f;
#pragma unroll
    for (int i = 0; i < N; ++i)
    {
        pr[i] = __expf(pr[i] - mx);
        se += pr[i];
    }
    float const inv = 1.0f / se;
#pragma unroll
    for (int i = 0; i < N; ++i)
        pr[i] *= inv;

    float acc[8];
#pragma unroll
    for (int e = 0; e < 8; ++e)
        acc[e] = 0.f;
#pragma unroll
    for (int i = 0; i < N; ++i)
    {
        const uint4 xi = xv(i);
#pragma unroll
        for (int e = 0; e < 4; ++e)
            mix2(u4get(xi, e), pr[i], acc[2 * e], acc[2 * e + 1]);
    }

    float r2;
    if constexpr (GRAM)
    {
        float msq = 0.f;
#pragma unroll
        for (int i = 0; i < N; ++i)
        {
            int const gi = i * N - (i * (i - 1)) / 2;
            float t = pr[i] * tot[gi];
#pragma unroll
            for (int j = i + 1; j < N; ++j)
                t = fmaf(2.0f * pr[j], tot[gi + (j - i)], t);
            msq = fmaf(pr[i], t, msq);
        }
        r2 = rsqrtf(fmaxf(msq, 0.f) * (1.0f / H) + p.eps2);
    }
    else
    {
        float q0 = 0.f, q1 = 0.f;
#pragma unroll
        for (int e = 0; e < 8; e += 2)
            ffma2(q0, q1, acc[e], acc[e + 1], acc[e], acc[e + 1]);
        float const ssq = warp_sum(q0 + q1);
        if (lane == 0)
            red2[warp] = ssq;
        __syncthreads();
        if (tid == 0)
        {
            float sacc = 0.f;
#pragma unroll
            for (int w = 0; w < W; ++w)
                sacc += red2[w];
            const uint32_t la = static_cast<uint32_t>(__cvta_generic_to_shared(&xch2[rank]));
#pragma unroll
            for (int r = 0; r < C; ++r)
                st_async_f32(mapa_u32(la, r), sacc, mapa_u32(xb2, r));
        }
        mb_wait_cta(xb2, 0);
        float ss_tot = 0.f;
#pragma unroll
        for (int r = 0; r < C; ++r)
            ss_tot += xch2[r];
        r2 = rsqrtf(ss_tot * (1.0f / H) + p.eps2);
    }
    if (active)
    {
        uint4 o;
        uint32_t* q = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
        for (int e = 0; e < 4; ++e)
            q[e] = pack_bf16((acc[2 * e] * r2) * bf_lo(u4get(ow, e)), (acc[2 * e + 1] * r2) * bf_hi(u4get(ow, e)));
        stg16(p.out + static_cast<size_t>(row) * H + col, o);
    }
}

// Mid-M decode: token split over a C-CTA cluster, V 16 B vectors per thread (fewer, wider CTAs so
// every CTA of the grid is resident), block snapshots staged by cp.async.bulk and re-read for the
// mix, two st.async + mbarrier cluster exchanges.
template <int N, bool ADD, int C, int V>
__global__ void __launch_bounds__(H / C / 8 / V) attnres_dec2_kernel(const Params p)
{
    constexpr int COLS = H / C;
    constexpr int T = COLS / 8 / V;
    static_assert(T % 32 == 0, "whole warps");
    constexpr int W = T / 32;
    constexpr int K = N - 1;
    constexpr int NV = 2 * N;
    constexpr int NP = next_pow2(NV);
    constexpr int SH = 5 - ilog2(NP);

    extern __shared__ __align__(128) uint8_t dsmem[];
    __shared__ __align__(16) float red[W][NP];
    __shared__ __align__(16) float xch[C][NP];
    __shared__ float red2[W];
    __shared__ float xch2[C];
    __shared__ __align__(8) uint64_t bars[3];

    int const tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const uint32_t rank = cluster_ctarank();
    int const row = blockIdx.x / C;
    int const col0 = static_cast<int>(rank) * COLS;
    const uint32_t dbar = static_cast<uint32_t>(__cvta_generic_to_shared(&bars[0]));
    const uint32_t xb1 = dbar + 8, xb2 = dbar + 16;
    const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(dsmem));
    if (tid == 0)
    {
        mb_init(dbar, 1);
        mb_init(xb1, 1);
        mb_init(xb2, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        mb_expect_tx(xb1, C * NV * 4);
        mb_expect_tx(xb2, C * 4);
        uint64_t policy;
        asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
        mbar_expect_tx(dbar, K * COLS * 2);
#pragma unroll
        for (int i = 0; i < K; ++i)
            bulk_g2s(sbase + i * COLS * 2, p.block + (static_cast<size_t>(i) * p.M + row) * H + col0, COLS * 2, dbar,
                policy);
    }
    cluster_arrive_relaxed();

    uint4 cur[V], ow[V];
    float wl[V][4], wh[V][4];
#pragma unroll
    for (int v = 0; v < V; ++v)
        cur[v] = ldg16_stream(p.layer + static_cast<size_t>(row) * H + col0 + 8 * (tid + v * T));
    if constexpr (ADD)
    {
        uint4 ad[V];
#pragma unroll
        for (int v = 0; v < V; ++v)
            ad[v] = ldg16_stream(p.add + static_cast<size_t>(row) * H + col0 + 8 * (tid + v * T));
#pragma unroll
        for (int v = 0; v < V; ++v)
        {
            uint32_t* u = reinterpret_cast<uint32_t*>(&cur[v]);
#pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                const uint32_t a = u4get(ad[v], e);
                u[e] = pack_bf16(__fadd_rn(bf_lo(u[e]), bf_lo(a)), __fadd_rn(bf_hi(u[e]), bf_hi(a)));
            }
        }
    }
#pragma unroll
    for (int v = 0; v < V; ++v)
    {
        int const c = col0 + 8 * (tid + v * T);
        const uint4 r = ldg16_keep(p.rms_w + c), w = ldg16_keep(p.res_w + c);
        ow[v] = ldg16_keep(p.out_w + c);
#pragma unroll
        for (int e = 0; e < 4; ++e)
        {
            wl[v][e] = bf_lo(u4get(r, e)) * bf_lo(u4get(w, e));
            wh[v][e] = bf_hi(u4get(r, e)) * bf_hi(u4get(w, e));
        }
        stg16(p.updated + static_cast<size_t>(row) * H + c, cur[v]);
    }

    float a[NP];
#pragma unroll
    for (int j = 0; j < NP; ++j)
        a[j] = 0.f;
    {
        float ss0 = 0.f, ss1 = 0.f, d0 = 0.f, d1 = 0.f;
#pragma unroll
        for (int v = 0; v < V; ++v)
#pragma unroll
            for (int e = 0; e < 4; ++e)
                stat2(u4get(cur[v], e), wl[v][e], wh[v][e], ss0, ss1, d0, d1);
        a[2 * K] = ss0 + ss1;
        a[2 * K + 1] = d0 + d1;
    }
    __syncthreads(); // barrier inits visible to all waiting threads
    mb_wait_cta(dbar, 0);
#pragma unroll
    for (int i = 0; i < K; ++i)
    {
        float ss0 = 0.f, ss1 = 0.f, d0 = 0.f, d1 = 0.f;
#pragma unroll
        for (int v = 0; v < V; ++v)
        {
            const uint4 xv = ld_shared16(sbase + (i * COLS + 8 * (tid + v * T)) * 2);
#pragma unroll
            for (int e = 0; e < 4; ++e)
                stat2(u4get(xv, e), wl[v][e], wh[v][e], ss0, ss1, d0, d1);
        }
        a[2 * i] = ss0 + ss1;
        a[2 * i + 1] = d0 + d1;
    }

    warp_multi<NP, 16, NP>(a, lane);
    if ((lane & ((1 << SH) - 1)) == 0)
        red[warp][lane >> SH] = a[0];
    __syncthreads();
    cluster_wait();
    if (tid < NV)
    {
        float sacc = 0.f;
#pragma unroll
        for (int w = 0; w < W; ++w)
            sacc += red[w][tid];
        const uint32_t la = static_cast<uint32_t>(__cvta_generic_to_shared(&xch[rank][tid]));
#pragma unroll
        for (int r = 0; r < C; ++r)
            st_async_f32(mapa_u32(la, r), sacc, mapa_u32(xb1, r));
    }
    mb_wait_cta(xb1, 0);
    float pr[N];
    float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < N; ++i)
    {
        float ss = 0.f, d = 0.f;
#pragma unroll
        for (int r = 0; r < C; ++r)
        {
            ss += xch[r][2 * i];
            d += xch[r][2 * i + 1];
        }
        pr[i] = d * rsqrtf(ss * (1.0f / H) + p.eps1);
        mx = fmaxf(mx, pr[i]);
    }
    float se = 0.f;
#pragma unroll
    for (int i = 0; i < N; ++i)
    {
        pr[i] = __expf(pr[i] - mx);
        se += pr[i];
    }
    float const inv = 1.0f / se;
#pragma unroll
    for (int i = 0; i < N; ++i)
        pr[i] *= inv;

    float acc[V][8];
#pragma unroll
    for (int v = 0; v < V; ++v)
#pragma unroll
        for (int e = 0; e < 8; ++e)
            acc[v][e] = 0.f;
#pragma unroll
    for (int v = 0; v < V; ++v)
#pragma unroll
        for (int e = 0; e < 4; ++e)
            mix2(u4get(cur[v], e), pr[K], acc[v][2 * e], acc[v][2 * e + 1]);
#pragma unroll
    for (int i = 0; i < K; ++i)
    {
#pragma unroll
        for (int v = 0; v < V; ++v)
        {
            const uint4 xv = ld_shared16(sbase + (i * COLS + 8 * (tid + v * T)) * 2);
#pragma unroll
            for (int e = 0; e < 4; ++e)
                mix2(u4get(xv, e), pr[i], acc[v][2 * e], acc[v][2 * e + 1]);
        }
    }
    float q0 = 0.f, q1 = 0.f;
#pragma unroll
    for (int v = 0; v < V; ++v)
#pragma unroll
        for (int e = 0; e < 8; e += 2)
            ffma2(q0, q1, acc[v][e], acc[v][e + 1], acc[v][e], acc[v][e + 1]);
    float const ssq = warp_sum(q0 + q1);
    if (lane == 0)
        red2[warp] = ssq;
    __syncthreads();
    if (tid == 0)
    {
        float sacc = 0.f;
#pragma unroll
        for (int w = 0; w < W; ++w)
            sacc += red2[w];
        const uint32_t la = static_cast<uint32_t>(__cvta_generic_to_shared(&xch2[rank]));
#pragma unroll
        for (int r = 0; r < C; ++r)
            st_async_f32(mapa_u32(la, r), sacc, mapa_u32(xb2, r));
    }
    mb_wait_cta(xb2, 0);
    float ss_tot = 0.f;
#pragma unroll
    for (int r = 0; r < C; ++r)
        ss_tot += xch2[r];
    float const r2 = rsqrtf(ss_tot * (1.0f / H) + p.eps2);
#pragma unroll
    for (int v = 0; v < V; ++v)
    {
        uint4 o;
        uint32_t* q = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
        for (int e = 0; e < 4; ++e)
            q[e] = pack_bf16(
                (acc[v][2 * e] * r2) * bf_lo(u4get(ow[v], e)), (acc[v][2 * e + 1] * r2) * bf_hi(u4get(ow[v], e)));
        stg16(p.out + static_cast<size_t>(row) * H + col0 + 8 * (tid + v * T), o);
    }
}

template <int N, bool ADD, int C, int V>
void launch_dec2(Params const& p, cudaStream_t stream)
{
    auto kern = attnres_dec2_kernel<N, ADD, C, V>;
    constexpr int SMEM = (N - 1) * (H / C) * 2;
    static bool init = false;
    if (!init)
    {
        if (SMEM > 48 * 1024)
            cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        init = true;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(p.M * C);
    cfg.blockDim = dim3(H / C / 8 / V);
    cfg.dynamicSmemBytes = SMEM;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = C;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, kern, p);
}

template <int N, bool ADD, int C, int MODE, bool GRAM>
void launch_dec(Params const& p, cudaStream_t stream)
{
    auto kern = attnres_dec_kernel<N, ADD, C, MODE, GRAM>;
    constexpr int SMEM = MODE >= 1 ? (N - 1) * DecCfg<C>::COLS * 2 : 0;
    static bool init = false;
    if (!init)
    {
        if (C > 8)
            cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
        if (SMEM > 48 * 1024)
            cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        init = true;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(p.M * C);
    cfg.blockDim = dim3(DecCfg<C>::T);
    cfg.dynamicSmemBytes = SMEM;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = C;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, kern, p);
}

template <int N, bool ADD, int C>
void launch_dec_mode(Params const& p, cudaStream_t stream)
{
    constexpr bool kGram = N <= 6; // Gram exchange (single cluster round trip) for few candidates
    bool const gram = p.gram && N <= p.gram_max;
    if (p.dec_mode == 1)
    {
        if (gram)
            launch_dec<N, ADD, C, 1, kGram>(p, stream);
        else
            launch_dec<N, ADD, C, 1, false>(p, stream);
    }
    else
    {
        if (gram)
            launch_dec<N, ADD, C, 0, kGram>(p, stream);
        else
            launch_dec<N, ADD, C, 0, false>(p, stream);
    }
}

template <int N, bool ADD, int C>
void launch_c(Params const& p, cudaStream_t stream)
{
    constexpr int T = (C == 1) ? 448 : (C == 2) ? 224 : (C == 4) ? 128 : 64;
    auto kern = attnres_reg_kernel<N, ADD, C, T>;
    if constexpr (C > 1 && N <= 5)
    { // Gram (single exchange) only pays off for few candidates
        if (p.gram)
            kern = attnres_gram_kernel<N, ADD, C, T>;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(p.M * C);
    cfg.blockDim = dim3(T);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    int na = 0;
    if constexpr (C > 1)
    {
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim.x = C;
        attr[0].val.clusterDim.y = 1;
        attr[0].val.clusterDim.z = 1;
        na = 1;
    }
    cfg.attrs = attr;
    cfg.numAttrs = na;
    cudaLaunchKernelEx(&cfg, kern, p);
}

template <int N, bool ADD>
void launch_n(Params const& p, int C, cudaStream_t stream)
{
    if (C == 0)
    {
        launch_ring<N, ADD>(p, stream);
        return;
    }
    if (C >= 100)
    { // only the 7-CTA configuration is dispatched
        launch_dec_mode<N, ADD, 7>(p, stream);
        return;
    }
    if (C == -3)
    { // 2-CTA cluster, 224 threads x 2 vectors
        launch_dec2<N, ADD, 2, 2>(p, stream);
        return;
    }

    switch (C)
    {
    case 2: launch_c<N, ADD, 2>(p, stream); break;
    default: launch_c<N, ADD, 1>(p, stream); break;
    }
}

template <bool ADD>
void launch_add(Params const& p, int N, int C, cudaStream_t stream)
{
    switch (N)
    {
    case 2: launch_n<2, ADD>(p, C, stream); break;
    case 3: launch_n<3, ADD>(p, C, stream); break;
    case 4: launch_n<4, ADD>(p, C, stream); break;
    case 5: launch_n<5, ADD>(p, C, stream); break;
    case 6: launch_n<6, ADD>(p, C, stream); break;
    case 7: launch_n<7, ADD>(p, C, stream); break;
    case 8: launch_n<8, ADD>(p, C, stream); break;
    case 9: launch_n<9, ADD>(p, C, stream); break;
    default: break;
    }
}

} // namespace

namespace
{

void attnres_launch(void const* layer, void const* add, void const* block, void const* res_w, void const* rms_w,
    void const* out_w, void* updated, void* out, int M, int K, int has_add, float eps1, float eps2, cudaStream_t stream)
{
    if (M <= 0)
        return;
    Params p;
    p.layer = static_cast<__nv_bfloat16 const*>(layer);
    p.add = static_cast<__nv_bfloat16 const*>(add);
    p.block = static_cast<__nv_bfloat16 const*>(block);
    p.res_w = static_cast<__nv_bfloat16 const*>(res_w);
    p.rms_w = static_cast<__nv_bfloat16 const*>(rms_w);
    p.out_w = static_cast<__nv_bfloat16 const*>(out_w);
    p.updated = static_cast<__nv_bfloat16*>(updated);
    p.out = static_cast<__nv_bfloat16*>(out);
    p.eps1 = eps1;
    p.eps2 = eps2;
    p.M = M;
    p.async_xch = 1;
    p.gram = 1;
    p.ring_slots = RING_S;
    p.hold = 0;
    p.dec_mode = 1;
    p.gram_max = 6;
    p.pace = (M <= 4 * num_sms()) ? 99 : 0; // decode: strict pacing (previous row fully landed)
    int const N = K + 1;
    // Dispatch. C > 100: decode kernel, row split over a (C - 100)-CTA cluster;
    // C = 1 / 2: register kernel (2: 2-CTA Gram cluster); C = 0: persistent TMA ring; C = -3: mid-M
    // 2-CTA cluster kernel with smem-staged snapshots.
    int const sms = num_sms();
    int const L = N + has_add; // candidate rows loaded per token
    int C;
    if (M <= 100)
    {                                  // decode: each row over a 7-CTA cluster (1024 columns)
        C = 107;
        p.dec_mode = (N >= 7) ? 1 : 0; // many candidates: TMA snapshots into smem
    }
    else if (M <= sms)
    {
        C = 1; // one row per SM
    }
    else if (M <= 2 * sms && N <= 3)
    {
        C = 2; // few candidates: all rows resident at once
    }
    else if (M <= 3 * sms && M % sms != 0 && M % sms <= sms / 2)
    {
        C = -3; // short last wave: split rows over 2 SMs instead
    }
    else
    {
        C = 0; // persistent TMA ring
    }
    if (C == 0)
    {
        if (p.pace || N >= 9)
        {
            // decode (paced) and 9-candidate prefill: keep block rows in the full-depth ring until the mix
            p.hold = 1;
        }
        else
        {
            // prefill: about one row of loads in flight per SM beats a deep ring (less DRAM contention)
            int const slots = L + 1 < 6 ? 6 : L + 1;
            p.ring_slots = slots < RING_S ? slots : RING_S;
        }
    }
    if (has_add)
        launch_add<true>(p, N, C, stream);
    else
        launch_add<false>(p, N, C, stream);
}

} // namespace

#else
#include "tensorrt_llm/common/assert.h"
#endif // PLACEHOLDER_KERNELS

TRTLLM_NAMESPACE_BEGIN

namespace kernels::kimiK3AttnRes
{

void invokeAttnResKfFwd(void const* layer, void const* add, void const* block, void const* resW, void const* rmsW,
    void const* outW, void* updated, void* out, int M, int numSnapshots, bool hasAdd, float rmsEps, float outputRmsEps,
    cudaStream_t stream)
{
#ifndef PLACEHOLDER_KERNELS
    attnres_launch(layer, add, block, resW, rmsW, outW, updated, out, M, numSnapshots, hasAdd ? 1 : 0, rmsEps,
        outputRmsEps, stream);
#else
    TLLM_THROW("invokeAttnResKfFwd: this build has no sm_100-family CUDA architecture");
#endif
}

} // namespace kernels::kimiK3AttnRes

TRTLLM_NAMESPACE_END
