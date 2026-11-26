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

// dspark_markov_prob: one DSpark Markov step for draft_sample_method=probabilistic, vocab-sharded, TP world 1/2/4.
// GEMV, bias rounding, base prefetch and grid are dspark_markov_sample's; the exchange is dspark_markov_sample_tp4's
// (one slot per source rank). Per element of this rank's shard, stock semantics:
//   lg   = bf16(base + bias)                                  dspark/speculator.py:187-188 (bf16 add)
//   draft_logits[rs, step, voff + v] = lg                     gumbel.py:159-175 (pre-temperature, valid rows only)
//   val  = T == 0 ? lg : div.full(lg, T) - log(-log1p(-u))    gumbel.py:108-128, u = rand(randint(seed, pos + 2^30),
//   id) tok  = argmax val over the full vocab, ties -> lowest id  gumbel.py:130 + :304-305
// plus this rank's (max, sum exp(lg/T - max)) per row, for a sharded rejection (reject/dkpb).
#pragma once
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "minimax_m3/csrc/m31/draftProb1Tp4/dspark_rng.h"

TRTLLM_MV2_NAMESPACE_BEGIN

namespace dkmkp
{
constexpr int kR = 256;
constexpr int kMaxC = 8;
constexpr int kMaxGrid = 512;
constexpr int kUnroll = 8;
constexpr int kMaxW = 4;

struct Args
{
    __nv_bfloat16 const* base; // this rank's shard of the draft logits; row (r*k + step), row stride ld
    int64_t ld;
    __nv_bfloat16 const* w1;   // [V, 256]
    __nv_bfloat16 const* w2s;  // [Vs, 256]
    int Vs, voff;
    int C, k, step;
    int64_t const* prev_in;
    int32_t const* prev32; // step 0: anchors from input_ids (prev32[r * prev32_stride])
    int prev32_stride;
    int64_t* prev_out;
    int64_t* tok_out;          // tok_out[r * tok_ld + step]
    int tok_ld;
    __nv_bfloat16* bias_dbg;   // optional [C, Vs]
                               // sampling state (speculator buffers, written by the prep kernel before this launch)
    int32_t const* idx_map;    // sample_idx_mapping [C*k]: request-state row of (r, step); fe1d int32
    int64_t const* idx_map64;  // the same map as int64 (the reference drafter's dflash/speculator.py); used when set
    int64_t const* sample_pos; // [C*k]; the draw position is sample_pos - 1 (dspark/speculator.py:136-143)
    float const* temperature;  // [max_num_reqs]
    int64_t const* seeds;      // [max_num_reqs]
    float const* noise_feed;   // optional [C, Vs] stock noise for this step (gate mode): val = lg/T + noise
    // draft_logits cache [max_num_reqs, k, V]: caches[0] = this rank's; mirror mode also writes every peer's copy
    __nv_bfloat16* caches[kMaxW];
    int ncache;
    int64_t cs0, cs1;
    float* dstat; // optional [max_num_reqs, k, 2] this rank's shard (max, sumexp) of lg/T
    // exchange workspace (dk_mk4 layout)
    unsigned long long* part; // [kMaxC][kMaxGrid] keys
    float* part_ms;           // [kMaxC][kMaxGrid][2]
    unsigned int* counter;
    unsigned long long* epoch;
    unsigned long long* mine; // [2 parity][kMaxC][kMaxW src]
    unsigned long long* peers[kMaxW];
    int* err;
    int rank, world;
};

__device__ __forceinline__ unsigned long long ld_rlx(unsigned long long const* p)
{
    unsigned long long v;
    asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void st_relaxed(unsigned long long* p, unsigned long long v)
{
    asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

__device__ __forceinline__ unsigned long long umax(unsigned long long a, unsigned long long b)
{
    return a > b ? a : b;
}

// (max, sum exp(x - max)) online merge with Triton's exp (ex2.approx(x * log2e)), as eglue_core.h ms_merge
struct MS
{
    float m, s;
};

__device__ __forceinline__ MS ms_merge(MS a, MS b)
{
    if (b.m == -INFINITY)
        return a;
    if (a.m == -INFINITY)
        return b;
    if (a.m >= b.m)
        return MS{a.m, dkr::add_rn(a.s, dkr::mul_rn(b.s, dkr::tl_exp(dkr::sub_rn(b.m, a.m))))};
    return MS{b.m, dkr::add_rn(b.s, dkr::mul_rn(a.s, dkr::tl_exp(dkr::sub_rn(a.m, b.m))))};
}

template <int C>
struct Cfg
{
    static constexpr int kNT = C <= 3 ? 512 : 256; // C4 at 512 threads spills 24 B with the sampling state
    static constexpr int kNW = kNT / 32;
};

template <int C>
__global__ void __launch_bounds__(Cfg<C>::kNT, 1) dk_mk_prob_kernel(Args a)
{
    constexpr int kWarps = Cfg<C>::kNW;
    constexpr int kThreads = Cfg<C>::kNT;
    __shared__ float s_e[C][kR];
    __shared__ unsigned long long s_k[kWarps][C];
    __shared__ MS s_ms[kWarps][C];
    __shared__ float s_t[C];
    __shared__ uint32_t s_gs[C];
    __shared__ int s_rs[C];
    __shared__ bool s_last;
    int const tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    for (int i = tid; i < C * kR; i += kThreads)
    {
        int const r = i / kR, j = i % kR;
        const int64_t p = a.prev32 ? (int64_t) a.prev32[(int64_t) r * a.prev32_stride] : a.prev_in[r];
        s_e[r][j] = __bfloat162float(a.w1[p * kR + j]);
    }
    if (tid < C)
    {
        // gumbel.py:154-158, 177-178: invalid rows (idx -1) draw at temperature 0 and are never cached
        int const row = tid * a.k + a.step;
        int const rs = a.idx_map64 != nullptr ? (int) a.idx_map64[row] : a.idx_map[row];
        float const t = rs >= 0 ? a.temperature[rs] : 0.f;
        s_t[tid] = t;
        long long const seed = rs >= 0 ? a.seeds[rs] : 0;
        s_gs[tid] = dkr::gumbel_seed(seed, a.sample_pos[row] - 1, true);
        s_rs[tid] = rs;
    }
    __syncthreads();
    float e[C][8];
#pragma unroll
    for (int r = 0; r < C; ++r)
#pragma unroll
        for (int q = 0; q < 8; ++q)
            e[r][q] = s_e[r][lane * 8 + q];
    constexpr int kRowsPerLane = (C + 3) / 4;
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
        tl_[i] = r < C ? s_t[r] : 0.f;
        gs_[i] = r < C ? s_gs[r] : 0u;
        rs_[i] = r < C ? s_rs[r] : -1;
    }
    int const ncache = a.ncache;
    int const gw = blockIdx.x * kWarps + warp, nw = gridDim.x * kWarps;
    int const u_own = ((lane >> 4) & 1) * 4 + ((lane >> 3) & 1) * 2 + ((lane >> 2) & 1);
    for (int v0 = gw * kUnroll; v0 < a.Vs; v0 += nw * kUnroll)
    {
        float bl[kRowsPerLane];
#pragma unroll
        for (int i = 0; i < kRowsPerLane; ++i)
        {
            int const r = (lane & 3) + 4 * i, v = v0 + u_own;
            bl[i] = (r < C && v < a.Vs) ? __bfloat162float(a.base[(int64_t) (r * a.k + a.step) * a.ld + v]) : 0.f;
        }
        uint4 w[kUnroll];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u)
        {
            int const v = v0 + u;
            w[u] = v < a.Vs ? __ldg(reinterpret_cast<uint4 const*>(a.w2s + (int64_t) v * kR) + lane)
                            : make_uint4(0, 0, 0, 0);
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
                if (a.bias_dbg)
                    a.bias_dbg[(int64_t) r * a.Vs + v] = bias;
                const uint32_t gid = (uint32_t) (a.voff + v);
                if (rs_[i] >= 0)
                {
                    const int64_t off = (int64_t) rs_[i] * a.cs0 + (int64_t) a.step * a.cs1 + gid;
                    for (int c = 0; c < ncache; ++c)
                        a.caches[c][off] = lgb;
                }
                float val;
                if (a.noise_feed)
                    val = tl_[i] == 0.f ? lg
                                        : dkr::add_rn(dkr::div_full(lg, tl_[i]), a.noise_feed[(int64_t) r * a.Vs + v]);
                else
                    val = dkr::gumbel_value(lg, tl_[i], gs_[i], gid, true);
                if (!(val != val))
                    best[i] = umax(best[i], dkr::make_key(val, gid));
                if (a.dstat && tl_[i] != 0.f)
                {
                    float const xt = dkr::div_full(lg, tl_[i]);
                    ms[i] = ms_merge(ms[i], MS{xt, 1.f});
                }
            }
        }
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;");
#endif
    // lanes of one request slot are lane = r & 3 (mod 4): reduce over the 8 lanes that share (lane & 3)
#pragma unroll
    for (int i = 0; i < kRowsPerLane; ++i)
    {
        unsigned long long k = best[i];
        MS m = ms[i];
#pragma unroll
        for (int o = 16; o >= 4; o >>= 1)
        {
            k = umax(k, __shfl_xor_sync(0xffffffffu, k, o));
            MS o_{__shfl_xor_sync(0xffffffffu, m.m, o), __shfl_xor_sync(0xffffffffu, m.s, o)};
            m = (lane & o) ? ms_merge(o_, m) : ms_merge(m, o_); // fixed (low lane, high lane) order
        }
        int const r = (lane & 3) + 4 * i;
        if (lane < 4 && r < C)
        {
            s_k[warp][r] = k;
            s_ms[warp][r] = m;
        }
    }
    __syncthreads();
    if (tid < C)
    {
        unsigned long long k = 0;
        MS m{-INFINITY, 0.f};
        for (int w = 0; w < kWarps; ++w)
        {
            k = umax(k, s_k[w][tid]);
            m = ms_merge(m, s_ms[w][tid]);
        }
        a.part[tid * kMaxGrid + blockIdx.x] = k;
        a.part_ms[(tid * kMaxGrid + blockIdx.x) * 2] = m.m;
        a.part_ms[(tid * kMaxGrid + blockIdx.x) * 2 + 1] = m.s;
    }
    // mirror-mode peer cache stores must be visible system-wide before the exchange publishes this step
    if (ncache > 1)
        __threadfence_system();
    else
        __threadfence();
    __syncthreads();
    if (tid == 0)
        s_last = atomicAdd(a.counter, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!s_last)
        return;
    __threadfence();
    if (warp < C)
    {
        int const r = warp;
        unsigned long long k = 0;
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
            k = umax(k, pk[j]);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            k = umax(k, __shfl_xor_sync(0xffffffffu, k, o));
        if (a.dstat && lane == 0 && s_rs[r] >= 0 && s_t[r] != 0.f)
        {
            MS m{-INFINITY, 0.f};
            for (int b = 0; b < (int) gridDim.x; ++b) // CTA order: deterministic
                m = ms_merge(
                    m, MS{__ldcg(&a.part_ms[(r * kMaxGrid + b) * 2]), __ldcg(&a.part_ms[(r * kMaxGrid + b) * 2 + 1])});
            float* d = a.dstat + ((int64_t) s_rs[r] * a.k + a.step) * 2;
            d[0] = m.m;
            d[1] = m.s;
        }
        unsigned long long const ep = *a.epoch + 1;
        int const par = (int) (ep & 1ull);
        unsigned long long g = k;
        if (a.world > 1)
        {
            int const s = lane;
            bool const xch = s < a.world && s != a.rank;
            unsigned long long xk = 0ull;
            if (xch)
            {
                st_relaxed(a.peers[s] + ((size_t) par * kMaxC + r) * kMaxW + a.rank, dkr::pack_word(k, ep));
                unsigned long long const* src = a.mine + ((size_t) par * kMaxC + r) * kMaxW + s;
                long long t0 = clock64();
                unsigned long long pw;
                bool ok = true;
                while (!dkr::word_ready(pw = ld_rlx(src), ep))
                {
                    if (clock64() - t0 > (1ll << 35))
                    {
                        atomicExch(a.err, 1);
                        ok = false;
                        break;
                    }
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
            a.tok_out[(int64_t) r * a.tok_ld + a.step] = tok;
            a.prev_out[r] = tok;
        }
    }
    __syncthreads();
    if (tid == 0)
    {
        *a.counter = 0u;
        *a.epoch = *a.epoch + 1;
    }
}

inline cudaError_t launch(Args const& a, int grid, bool pdl, cudaStream_t s)
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
    case 1: return cudaLaunchKernelEx(&cfg, dk_mk_prob_kernel<1>, a);
    case 2: return cudaLaunchKernelEx(&cfg, dk_mk_prob_kernel<2>, a);
    case 3: return cudaLaunchKernelEx(&cfg, dk_mk_prob_kernel<3>, a);
    case 4: return cudaLaunchKernelEx(&cfg, dk_mk_prob_kernel<4>, a);
    case 5: return cudaLaunchKernelEx(&cfg, dk_mk_prob_kernel<5>, a);
    case 6: return cudaLaunchKernelEx(&cfg, dk_mk_prob_kernel<6>, a);
    case 7: return cudaLaunchKernelEx(&cfg, dk_mk_prob_kernel<7>, a);
    case 8: return cudaLaunchKernelEx(&cfg, dk_mk_prob_kernel<8>, a);
    default: return cudaErrorInvalidValue;
    }
}
} // namespace dkmkp

TRTLLM_MV2_NAMESPACE_END
