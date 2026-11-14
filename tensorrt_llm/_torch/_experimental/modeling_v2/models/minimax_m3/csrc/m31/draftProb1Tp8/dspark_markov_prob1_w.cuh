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

// dspark_markov_prob1_w: dspark_markov_prob1.cuh (shipped, unchanged) templated on the exchange world W (see
// dspark_markov_prob_w.cuh Lim<W>); the only non-sizing change is the warp-0 peer-word max, which reduces over the W
// source lanes of a row (xor 1, 2 at W 4 as shipped; xor 1, 2, 4 at W 8). Original header comment:
// dspark_markov_prob1: all k probabilistic DSpark Markov steps of one draft in ONE launch, bitwise equal to k launches
// of dk_mk_prob (dspark_markov_prob.cuh). Per step every CTA does exactly dk_mk_prob's work for its blockIdx (same
// element mapping, same partial slot) and the last CTA to finish is that step's leader with dk_mk_prob's reduction and
// cross-rank exchange (same words, same epoch values); it also stores its own word in this rank's slot, so every CTA
// takes the step's token as the max of the world words it reads (unpack(pack(key)) == key for ids < 2^18) instead of
// waiting on a kernel boundary or on the leader. Not persistent: the grid exits after step k-1.
// Requires every CTA co-resident (grid <= SMs at 1 CTA/SM, checked by the host); waits are bounded and set err.
#pragma once
#include "minimax_m3/csrc/m31/draftProb1Tp8/dspark_markov_prob_w.cuh"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

TRTLLM_MV2_NAMESPACE_BEGIN

namespace dkmkp1w
{
using dkmkpw::kMaxGrid;
using dkmkpw::Lim;
using dkmkpw::kR;
using dkmkpw::kUnroll;
using dkmkpw::MS;
using dkmkpw::ms_merge;
using dkmkpw::umax;
constexpr int kMaxK = 16;
constexpr long long kSpin = 1ll << 35;

template <int W>
struct Args
{
    static constexpr int kMaxW = Lim<W>::kMaxW;
    __nv_bfloat16 const* base; // row (r*k + step) of this rank's shard, row stride ld
    int64_t ld;
    __nv_bfloat16 const* w1;
    __nv_bfloat16 const* w2s;
    int Vs, voff;
    int C, k;
    int64_t const* prev_in; // step-0 anchors unless prev32 is set
    int32_t const* prev32;
    int prev32_stride;
    int64_t* prev_buf[2]; // step i writes prev_buf[i & 1], as dk_mk_prob's ping-pong
    int64_t* tok_out;
    int tok_ld;
    int32_t const* idx_map;
    int64_t const* idx_map64;
    int64_t const* sample_pos;
    float const* temperature;
    int64_t const* seeds;
    __nv_bfloat16* caches[kMaxW];
    int ncache;
    int64_t cs0, cs1;
    float* dstat;
    unsigned long long* part;
    float* part_ms;
    unsigned int* counter;
    unsigned long long* epoch;
    unsigned long long* mine;
    unsigned long long* peers[kMaxW];
    int* err;
    int rank, world;
};

__device__ __forceinline__ void st_rel(unsigned long long* p, unsigned long long v)
{
    asm volatile("st.release.gpu.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

template <int W, int C>
__global__ void __launch_bounds__(dkmkpw::Cfg<C>::kNT, 1) dk_mk_prob1_kernel(Args<W> a)
{
    constexpr int kMaxC = Lim<W>::kMaxC;
    constexpr int kMaxW = Lim<W>::kMaxW;
    static_assert(C >= 1 && C <= kMaxC);
    constexpr int kWarps = dkmkpw::Cfg<C>::kNW;
    constexpr int kThreads = dkmkpw::Cfg<C>::kNT;
    __shared__ float s_e[C][kR];
    __shared__ unsigned long long s_k[kWarps][C];
    __shared__ MS s_ms[kWarps][C];
    __shared__ float s_t[kMaxK][C];
    __shared__ uint32_t s_gs[kMaxK][C];
    __shared__ int s_rs[kMaxK][C];
    __shared__ int64_t s_tok[C];
    __shared__ unsigned long long s_ep0;
    __shared__ bool s_last, s_abort;
    int const tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    // every step's sampling state up front: it does not depend on the drafted tokens
    for (int i = tid; i < C * a.k; i += kThreads)
    {
        int const r = i / a.k, st = i % a.k, row = r * a.k + st;
        int const rs = a.idx_map64 != nullptr ? (int) a.idx_map64[row] : a.idx_map[row];
        s_t[st][r] = rs >= 0 ? a.temperature[rs] : 0.f;
        s_gs[st][r] = dkr::gumbel_seed(rs >= 0 ? a.seeds[rs] : 0, a.sample_pos[row] - 1, true);
        s_rs[st][r] = rs;
    }
    if (tid == 0)
    {
        // no leader can advance the epoch before every CTA has done step 0, so this is the launch's base value
        s_ep0 = *(unsigned long long volatile*) a.epoch;
        s_abort = false;
    }
    __syncthreads();
    unsigned long long const ep0 = s_ep0;
    constexpr int kRowsPerLane = (C + 3) / 4;
    constexpr bool kPre = C == 1 || C >= 4; // C2/C3 sit at the 128-register cap of 512 threads: no early tile there
    int const ncache = a.ncache;
    int const gw = blockIdx.x * kWarps + warp, nw = gridDim.x * kWarps;
    int const u_own = ((lane >> 4) & 1) * 4 + ((lane >> 3) & 1) * 2 + ((lane >> 2) & 1);
    for (int step = 0; step < a.k; ++step)
    {
        // this warp's first tile and base values do not depend on the token: issue them before the wait
        int const v00 = gw * kUnroll;
        uint4 w0[kPre ? kUnroll : 1];
        float bl0[kPre ? kRowsPerLane : 1];
        if constexpr (kPre)
        {
#pragma unroll
            for (int u = 0; u < kUnroll; ++u)
            {
                int const v = v00 + u;
                w0[u] = v < a.Vs ? __ldg(reinterpret_cast<uint4 const*>(a.w2s + (int64_t) v * kR) + lane)
                                 : make_uint4(0, 0, 0, 0);
            }
#pragma unroll
            for (int i = 0; i < kRowsPerLane; ++i)
            {
                int const r = (lane & 3) + 4 * i, v = v00 + u_own;
                bl0[i] = (r < C && v < a.Vs) ? __bfloat162float(a.base[(int64_t) (r * a.k + step) * a.ld + v]) : 0.f;
            }
        }
        if (step > 0)
        {
            static_assert(kMaxC * kMaxW == 32, "one warp lane per (row, source rank)");
            if (warp == 0)
            {
                unsigned long long const ep = ep0 + (unsigned long long) step; // the previous step's words
                int const r = lane / kMaxW, s = lane % kMaxW;
                unsigned long long key = 0ull;
                bool ok = true;
                if (r < C && s < a.world)
                {
                    unsigned long long const* src = a.mine + ((size_t) (ep & 1ull) * kMaxC + r) * kMaxW + s;
                    long long const t0 = clock64();
                    unsigned long long pw;
                    while (!dkr::word_ready(pw = dkmkpw::ld_rlx(src), ep))
                        if (clock64() - t0 > kSpin)
                        {
                            atomicExch(a.err, 2);
                            ok = false;
                            break;
                        }
                    if (ok)
                        key = dkr::unpack_word(pw);
                }
                key = umax(key, __shfl_xor_sync(0xffffffffu, key, 1));
                key = umax(key, __shfl_xor_sync(0xffffffffu, key, 2));
                if constexpr (kMaxW > 4) // W 8: the row's other four source lanes (W 4 is the shipped two-step max)
                    key = umax(key, __shfl_xor_sync(0xffffffffu, key, 4));
                bool const bad = __any_sync(0xffffffffu, !ok);
                if (s == 0 && r < C)
                    s_tok[r] = (int64_t) dkr::key_id(key);
                if (lane == 0)
                    s_abort = bad;
            }
            __syncthreads();
            if (s_abort)
                return;
        }
        for (int i = tid; i < C * kR; i += kThreads)
        {
            int const r = i / kR, j = i % kR;
            const int64_t p = step > 0 ? s_tok[r]
                : a.prev32             ? (int64_t) a.prev32[(int64_t) r * a.prev32_stride]
                                       : a.prev_in[r];
            s_e[r][j] = __bfloat162float(a.w1[p * kR + j]);
        }
        __syncthreads();
        float e[C][8];
#pragma unroll
        for (int r = 0; r < C; ++r)
#pragma unroll
            for (int q = 0; q < 8; ++q)
                e[r][q] = s_e[r][lane * 8 + q];
        unsigned long long best[kRowsPerLane];
        MS ms[kRowsPerLane];
        float tl_[kRowsPerLane];
        uint32_t gs_[kRowsPerLane];
        int rs_[kRowsPerLane];
#pragma unroll
        for (int i = 0; i < kRowsPerLane; ++i)
        {
            int const r = (lane & 3) + 4 * i;
            best[i] = 0ull;
            ms[i] = MS{-INFINITY, 0.f};
            tl_[i] = r < C ? s_t[step][r] : 0.f;
            gs_[i] = r < C ? s_gs[step][r] : 0u;
            rs_[i] = r < C ? s_rs[step][r] : -1;
        }
        bool first = kPre;
        for (int v0 = v00; v0 < a.Vs; v0 += nw * kUnroll)
        {
            float bl[kRowsPerLane];
            uint4 w[kUnroll];
            if (first)
            {
                if constexpr (kPre)
                {
#pragma unroll
                    for (int i = 0; i < kRowsPerLane; ++i)
                        bl[i] = bl0[i];
#pragma unroll
                    for (int u = 0; u < kUnroll; ++u)
                        w[u] = w0[u];
                }
                first = false;
            }
            else
            {
#pragma unroll
                for (int i = 0; i < kRowsPerLane; ++i)
                {
                    int const r = (lane & 3) + 4 * i, v = v0 + u_own;
                    bl[i] = (r < C && v < a.Vs) ? __bfloat162float(a.base[(int64_t) (r * a.k + step) * a.ld + v]) : 0.f;
                }
#pragma unroll
                for (int u = 0; u < kUnroll; ++u)
                {
                    int const v = v0 + u;
                    w[u] = v < a.Vs ? __ldg(reinterpret_cast<uint4 const*>(a.w2s + (int64_t) v * kR) + lane)
                                    : make_uint4(0, 0, 0, 0);
                }
            }
#pragma unroll
            for (int r = 0; r < C; ++r)
            {
                float p[kUnroll];
#pragma unroll
                for (int u = 0; u < kUnroll; ++u)
                {
                    __nv_bfloat16 const* wb = reinterpret_cast<__nv_bfloat16 const*>(&w[u]);
                    float s = 0.f;
#pragma unroll
                    for (int q = 0; q < 8; ++q)
                        s = fmaf(__bfloat162float(wb[q]), e[r][q], s);
                    p[u] = s;
                }
                bool const b16 = lane & 16, b8 = lane & 8, b4 = lane & 4;
                float q4[4];
#pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    float const send = b16 ? p[i] : p[i + 4];
                    float const keep = b16 ? p[i + 4] : p[i];
                    q4[i] = keep + __shfl_xor_sync(0xffffffffu, send, 16);
                }
                float q2[2];
#pragma unroll
                for (int i = 0; i < 2; ++i)
                {
                    float const send = b8 ? q4[i] : q4[i + 2];
                    float const keep = b8 ? q4[i + 2] : q4[i];
                    q2[i] = keep + __shfl_xor_sync(0xffffffffu, send, 8);
                }
                float s;
                {
                    float const send = b4 ? q2[0] : q2[1];
                    float const keep = b4 ? q2[1] : q2[0];
                    s = keep + __shfl_xor_sync(0xffffffffu, send, 4);
                }
                s += __shfl_xor_sync(0xffffffffu, s, 2);
                s += __shfl_xor_sync(0xffffffffu, s, 1);
                int const v = v0 + u_own;
                if ((r & 3) == (lane & 3) && v < a.Vs)
                {
                    int const i = r >> 2;
                    const __nv_bfloat16 bias = __float2bfloat16_rn(s);
                    const __nv_bfloat16 lgb = __float2bfloat16_rn(bl[i] + __bfloat162float(bias));
                    float const lg = __bfloat162float(lgb);
                    const uint32_t gid = (uint32_t) (a.voff + v);
                    if (rs_[i] >= 0)
                    {
                        const int64_t off = (int64_t) rs_[i] * a.cs0 + (int64_t) step * a.cs1 + gid;
                        for (int c = 0; c < ncache; ++c)
                            a.caches[c][off] = lgb;
                    }
                    float const val = dkr::gumbel_value(lg, tl_[i], gs_[i], gid, true);
                    if (!(val != val))
                        best[i] = umax(best[i], dkr::make_key(val, gid));
                    if (a.dstat && tl_[i] != 0.f)
                        ms[i] = ms_merge(ms[i], MS{dkr::div_full(lg, tl_[i]), 1.f});
                }
            }
        }
        bool const last_step = step == a.k - 1;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        // only after every CTA is known resident (step k-1 starts after all CTAs finished step k-2)
        if (last_step)
            asm volatile("griddepcontrol.launch_dependents;");
#endif
#pragma unroll
        for (int i = 0; i < kRowsPerLane; ++i)
        {
            unsigned long long kk = best[i];
            MS m = ms[i];
#pragma unroll
            for (int o = 16; o >= 4; o >>= 1)
            {
                kk = umax(kk, __shfl_xor_sync(0xffffffffu, kk, o));
                MS o_{__shfl_xor_sync(0xffffffffu, m.m, o), __shfl_xor_sync(0xffffffffu, m.s, o)};
                m = (lane & o) ? ms_merge(o_, m) : ms_merge(m, o_);
            }
            int const r = (lane & 3) + 4 * i;
            if (lane < 4 && r < C)
            {
                s_k[warp][r] = kk;
                s_ms[warp][r] = m;
            }
        }
        __syncthreads();
        if (tid < C)
        {
            unsigned long long kk = 0;
            MS m{-INFINITY, 0.f};
            for (int w = 0; w < kWarps; ++w)
            {
                kk = umax(kk, s_k[w][tid]);
                m = ms_merge(m, s_ms[w][tid]);
            }
            a.part[tid * kMaxGrid + blockIdx.x] = kk;
            a.part_ms[(tid * kMaxGrid + blockIdx.x) * 2] = m.m;
            a.part_ms[(tid * kMaxGrid + blockIdx.x) * 2 + 1] = m.s;
        }
        // mirror peer stores of every step become system-visible here, before the last step's exchange publishes
        if (ncache > 1 && last_step)
            __threadfence_system();
        else
            __threadfence();
        __syncthreads();
        if (tid == 0)
            s_last = atomicAdd(a.counter, 1u) == (unsigned int) (step + 1) * gridDim.x - 1; // reset once, in step k-1
        __syncthreads();
        if (!s_last)
            continue;
        __threadfence();
        unsigned long long const ep = ep0 + (unsigned long long) step + 1ull;
        if (warp < C)
        {
            int const r = warp;
            unsigned long long kk = 0;
            constexpr int kPer = kMaxGrid / 32;
            unsigned long long pk[kPer];
#pragma unroll
            for (int j = 0; j < kPer; ++j)
            {
                int const b = lane + 32 * j;
                pk[j] = b < (int) gridDim.x ? __ldcg(&a.part[r * kMaxGrid + b]) : 0ull;
            }
#pragma unroll
            for (int j = 0; j < kPer; ++j)
                kk = umax(kk, pk[j]);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                kk = umax(kk, __shfl_xor_sync(0xffffffffu, kk, o));
            if (a.dstat && lane == 0 && s_rs[step][r] >= 0 && s_t[step][r] != 0.f)
            {
                MS m{-INFINITY, 0.f};
                for (int b = 0; b < (int) gridDim.x; ++b)
                    m = ms_merge(m,
                        MS{__ldcg(&a.part_ms[(r * kMaxGrid + b) * 2]), __ldcg(&a.part_ms[(r * kMaxGrid + b) * 2 + 1])});
                float* d = a.dstat + ((int64_t) s_rs[step][r] * a.k + step) * 2;
                d[0] = m.m;
                d[1] = m.s;
            }
            int const par = (int) (ep & 1ull);
            if (lane == 0) // this rank's own word: what every local CTA reads beside the peers'
                dkmkpw::st_relaxed(a.mine + ((size_t) par * kMaxC + r) * kMaxW + a.rank, dkr::pack_word(kk, ep));
            unsigned long long g = kk;
            if (a.world > 1)
            {
                int const s = lane;
                bool const xch = s < a.world && s != a.rank;
                unsigned long long xk = 0ull;
                if (xch)
                {
                    dkmkpw::st_relaxed(
                        a.peers[s] + ((size_t) par * kMaxC + r) * kMaxW + a.rank, dkr::pack_word(kk, ep));
                    unsigned long long const* src = a.mine + ((size_t) par * kMaxC + r) * kMaxW + s;
                    long long const t0 = clock64();
                    unsigned long long pw;
                    bool ok = true;
                    while (!dkr::word_ready(pw = dkmkpw::ld_rlx(src), ep))
                        if (clock64() - t0 > kSpin)
                        {
                            atomicExch(a.err, 1);
                            ok = false;
                            break;
                        }
                    if (ok)
                        xk = dkr::unpack_word(pw);
                }
#pragma unroll
                for (int o = 16; o > 0; o >>= 1)
                    xk = umax(xk, __shfl_xor_sync(0xffffffffu, xk, o));
                g = umax(g, xk);
            }
            if (lane == 0)
            {
                const int64_t tok = (int64_t) dkr::key_id(g);
                a.tok_out[(int64_t) r * a.tok_ld + step] = tok;
                a.prev_buf[step & 1][r] = tok;
            }
        }
        __syncthreads();
        if (tid == 0)
        {
            if (last_step)
                *a.counter = 0u; // every CTA's last arrival is in; dk_mk_prob and the next launch start from 0
            __threadfence();
            st_rel(a.epoch, ep); // dk_mk_prob's epoch value after this step
        }
    }
}

template <int W, int C>
inline cudaError_t launch_c(cudaLaunchConfig_t const& cfg, Args<W> const& a)
{
    if constexpr (C <= Lim<W>::kMaxC)
        return cudaLaunchKernelEx(&cfg, dk_mk_prob1_kernel<W, C>, a);
    else
        return cudaErrorInvalidValue;
}

template <int W>
inline cudaError_t launch(Args<W> const& a, int grid, bool pdl, cudaStream_t s)
{
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(a.C <= 3 ? 512 : 256);
    cfg.stream = s;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    switch (a.C)
    {
    case 1: return launch_c<W, 1>(cfg, a);
    case 2: return launch_c<W, 2>(cfg, a);
    case 3: return launch_c<W, 3>(cfg, a);
    case 4: return launch_c<W, 4>(cfg, a);
    case 5: return launch_c<W, 5>(cfg, a);
    case 6: return launch_c<W, 6>(cfg, a);
    case 7: return launch_c<W, 7>(cfg, a);
    case 8: return launch_c<W, 8>(cfg, a);
    default: return cudaErrorInvalidValue;
    }
}

template <int W, int C>
inline int ctas_per_sm_c()
{
    int n = 0;
    if constexpr (C <= Lim<W>::kMaxC)
    {
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, dk_mk_prob1_kernel<W, C>, C <= 3 ? 512 : 256, 0)
            != cudaSuccess)
            n = 0;
    }
    return n;
}

// CTAs of this kernel that fit on one SM at C rows (the host keeps grid <= SMs * this, so every CTA is co-resident)
template <int W>
inline int ctas_per_sm(int C)
{
    switch (C)
    {
    case 1: return ctas_per_sm_c<W, 1>();
    case 2: return ctas_per_sm_c<W, 2>();
    case 3: return ctas_per_sm_c<W, 3>();
    case 4: return ctas_per_sm_c<W, 4>();
    case 5: return ctas_per_sm_c<W, 5>();
    case 6: return ctas_per_sm_c<W, 6>();
    case 7: return ctas_per_sm_c<W, 7>();
    case 8: return ctas_per_sm_c<W, 8>();
    default: return 0;
    }
}
} // namespace dkmkp1w

TRTLLM_MV2_NAMESPACE_END
