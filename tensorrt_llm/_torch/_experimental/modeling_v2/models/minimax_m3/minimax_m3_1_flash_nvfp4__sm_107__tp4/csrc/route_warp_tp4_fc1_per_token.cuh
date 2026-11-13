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

// X4 == 2 is mode Q's per-token NVFP4 tail: the row amax rides the sum-of-squares cluster message (float4 per CTA: sq,
// then the CTA's argmax |s * w| pair), the row quantizer matches nvfp4_row_quant, and s_t goes to q.x4_pts[t].
// lbv3rp_kernel routes on an added warp: rx poll + sums before inv, top-4 after inv, one relaxed 16 B store per lane.
// Prepushed all-reduce + residual + Gemma RMSNorm + quant, plus MoE routing from the o_proj router CTAs' partials.
//
// Per row t, the cluster's crank-0 CTA, warp 0 (lane l owns experts 4l..4l+3):
//   0. at entry (before the prx poll) issue the rx loads: rx[t][src][part][E] u64 words (tag << 32 | fp32), the
//      o_proj router role pushed them with tag = its per-CTA call counter + 1; nothing consumes them before the
//      cluster exchange;
//   1. after the row's sum of squares: accept a block only when all its words carry tag == rxc[t] + 1 (no sentinel,
//      no re-arm: a stale word never matches), re-poll the rest until a globaltimer deadline; bump rxc[t];
//   2. S_e = fixed (src, part) order sum, logits_e = inv_rms * S_e: same bits on every rank;
//   3. MiniMax2 top-4 (the engine's routing, operation for operation; owner lane of expert e is e / 4 here);
//   4. publish 4 tagged u64 route words per replica line (tag = rctr[t] + 1), then the normed / MXFP8 tail.
// q.route == nullptr: no routing block (the prepushed post's math only).
#pragma once
#include "allreduce_norm_quant.cuh"

namespace rh
{
constexpr int E = 128;
constexpr int TOPK = 4;
constexpr int LINE_U64 = 16;    // one 128-byte line per (replica, token)
constexpr int RCTR_STRIDE = 32; // int32 words per row counter (own line)
constexpr int MAXRP = 6;        // world * npart polled blocks per row (TP2 x (a + 2 r parts))
constexpr int TDBG_W = 4;       // tdbg words per (call, row): entry, routing start, published, CTA end
constexpr float ROUTE_SCALE = 2.0f;
constexpr float SUM_EPS = 1e-20f;

struct RhParams
{
    unsigned long long* rx;    // this rank's rx slot: u64 [maxt][world][npart][E] = (tag << 32) | fp32 bits
    int* rxc;                  // [maxt][RCTR_STRIDE]: word 0 = tag of the last consumed rx call of row t;
                               //   HANG_DBG words 1..3 = deadline hits, expected tag, an observed stale tag
    float const* bias;         // e_score_correction_bias fp32 [E]
    unsigned long long* route; // [repl][maxt][LINE_U64]; words 0..3 = (tag << 32) | (id << 16) | bf16(w); null = off
    int* rctr;                 // [maxt][RCTR_STRIDE]: word 0 = tag of the last publish of row t
    float* lg_dbg;             // optional [maxt][E] hoisted logits (route-check builds), nullptr = off
    unsigned long long* tdbg;  // optional [16 calls][maxt][TDBG_W] globaltimer, nullptr = off
    unsigned const* tab;       // skeleton tier B: [maxt][4] learned (id << 16 | bf16 w); non-null = publish it instead
    int npart, repl, delay_ns; // delay_ns: tier B's modelled routing time before the publish
    int rmaxt;                 // row stride of route and tdbg (the RouteBuf's maxt; rx uses the prx bucket's maxt)
    unsigned long long deadline_ns; // rx poll deadline (globaltimer)
    int hang_dbg;                   // 1: on the deadline record into rxc words 1..3 and continue; 0: __trap()
    // X4: the engine's B1 activation, written by the post after normed2
    uint8_t* x4_q;        // [maxt][H / 2] packed e2m1 pairs, row-major (low nibble = even column)
    uint8_t* x4_sf;       // [maxt][H / 16] e4m3 block-16 scales, row-major
    float const* x4_sfsc; // [1] fc31_input_scale (static)
    float* x4_pts;        // X4 2: [>= rows] fp32 per-token FC1 dequant scale s_t (the engine folds it into FC1)
};

// IEEE ops without ftz: the engine's _add_rn / _mul_rn / _fma_rn / _sub_rn / _div_rn spellings (asm: no contraction)
__device__ __forceinline__ float fadd(float a, float b)
{
    float d;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

__device__ __forceinline__ float fsub(float a, float b)
{
    float d;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

__device__ __forceinline__ float fmul(float a, float b)
{
    float d;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

__device__ __forceinline__ float ffma(float a, float b, float c)
{
    float d;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
    return d;
}

__device__ __forceinline__ float fdiv(float a, float b)
{
    float d;
    asm("div.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

__device__ __forceinline__ unsigned redux_max(unsigned v)
{
    unsigned d;
    asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(d) : "r"(v));
    return d;
}

__device__ __forceinline__ unsigned twiddle(float x)
{
    unsigned const b = __float_as_uint(x);
    return (int) b < 0 ? ~b : (b | 0x80000000u);
}

// Must lower like the engine's cute.math.tanh (precise vs tanh.approx); not yet confirmed against the engine's SASS.
__device__ __forceinline__ float tanh_engine(float x)
{
    return tanhf(x);
}

__device__ __forceinline__ void st_relaxed_v2u64(unsigned long long* p, unsigned long long a, unsigned long long b)
{
    asm volatile("st.relaxed.gpu.global.v2.u64 [%0], {%1, %2};" ::"l"(p), "l"(a), "l"(b) : "memory");
}

__device__ __forceinline__ ulonglong2 ldv_u64x2(unsigned long long const* p)
{
    ulonglong2 v;
    asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ bool tag2(ulonglong2 v, unsigned tag)
{
    return (unsigned) (v.x >> 32) == tag && (unsigned) (v.y >> 32) == tag;
}

// 0 / re-poll: issue the lane's two 16-byte loads (experts 4 lane .. +3) of every pending (src, part) block; block
// i = src * npart + part sits at row + i * E. Loads only: the tag check is rx_check, after the cluster exchange.
__device__ __forceinline__ void rx_issue(
    unsigned long long const* row, int lane, int n, unsigned pending, ulonglong2 v[MAXRP][2])
{
#pragma unroll
    for (int i = 0; i < MAXRP; ++i)
        if (i < n && ((pending >> i) & 1u))
        {
            v[i][0] = ldv_u64x2(row + i * E + 4 * lane);
            v[i][1] = ldv_u64x2(row + i * E + 4 * lane + 2);
        }
}

// 0 (entry): every block, unconditionally (blocks >= n re-read block n - 1 and are never checked). A predicated load
// makes ptxas merge it into the live value with a predicated MOV that waits for the data (sm_107a), stalling warp 0
// before the RMS exchange.
__device__ __forceinline__ void rx_issue_all(unsigned long long const* row, int lane, int n, ulonglong2 v[MAXRP][2])
{
#pragma unroll
    for (int i = 0; i < MAXRP; ++i)
    {
        unsigned long long const* const b = row + (i < n ? i : n - 1) * E + 4 * lane;
        v[i][0] = ldv_u64x2(b);
        v[i][1] = ldv_u64x2(b + 2);
    }
}

__device__ __forceinline__ void rx_check(int n, unsigned tag, const ulonglong2 v[MAXRP][2], unsigned& pending)
{
#pragma unroll
    for (int i = 0; i < MAXRP; ++i)
        if (i < n && ((pending >> i) & 1u) && tag2(v[i][0], tag) && tag2(v[i][1], tag))
            pending &= ~(1u << i);
}

// 1 + 2: finish the poll (exact tag match, globaltimer deadline), lane's 4 hoisted logits in the fixed (src, part)
// order. Returns the observed tag of a still-pending word on a deadline hit (0 otherwise) through `stale`.
// ROLL: one check / issue copy (the first check folded into the loop); same accepted words, same sums.
template <int ROLL>
__device__ __forceinline__ bool rx_finish(RhParams const& q, unsigned long long const* row, int lane, int world,
    unsigned tag, float inv, ulonglong2 v[MAXRP][2], float lg[4], unsigned& stale)
{
    int const n = world * q.npart;
    unsigned pending = (1u << n) - 1u;
    bool late = false;
    stale = 0u;
    if constexpr (ROLL != 0)
    {
        unsigned long long t0 = 0ull; // globaltimer is never 0
#pragma unroll 1
        for (;;)
        {
            rx_check(n, tag, v, pending);
            if (!pending)
                break;
            unsigned long long const now = gt();
            if (t0 == 0ull)
                t0 = now;
            else if (now - t0 > q.deadline_ns)
            {
                late = true;
                break;
            }
            rx_issue(row, lane, n, pending, v);
        }
    }
    else
    {
        rx_check(n, tag, v, pending);
        if (pending)
        {
            unsigned long long const t0 = gt();
            while (pending)
            {
                rx_issue(row, lane, n, pending, v);
                rx_check(n, tag, v, pending);
                if (pending && gt() - t0 > q.deadline_ns)
                {
                    late = true;
                    break;
                }
            }
        }
    }
    if (late)
    {
#pragma unroll
        for (int i = MAXRP - 1; i >= 0; --i)
            if (i < n && ((pending >> i) & 1u))
                stale = (unsigned) (v[i][0].x >> 32);
    }
    if constexpr (ROLL != 0)
    { // world 2: the flat walk's order written out per npart (same adds, same operands)
#pragma unroll
        for (int e = 0; e < 4; ++e)
        {
            auto f = [&](int i) {
                return __uint_as_float(
                    (unsigned) (e < 2 ? (e == 0 ? v[i][0].x : v[i][0].y) : (e == 2 ? v[i][1].x : v[i][1].y)));
            };
            float sv;
            if (q.npart == 1)
                sv = fadd(f(0), f(1));
            else if (q.npart == 2)
                sv = fadd(fadd(f(0), f(1)), fadd(f(2), f(3)));
            else
                sv = fadd(fadd(fadd(f(0), f(1)), f(2)), fadd(fadd(f(3), f(4)), f(5)));
            lg[e] = fmul(inv, sv);
        }
        return late;
    }
    float s[4] = {0.f, 0.f, 0.f, 0.f}, pr[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int i = 0; i < MAXRP; ++i)
    {
        if (i >= n)
            break;
        int const r = i / q.npart, j = i - r * q.npart;
        float const f[4] = {__uint_as_float((unsigned) v[i][0].x), __uint_as_float((unsigned) v[i][0].y),
            __uint_as_float((unsigned) v[i][1].x), __uint_as_float((unsigned) v[i][1].y)};
#pragma unroll
        for (int e = 0; e < 4; ++e)
        {
            pr[e] = j == 0 ? f[e] : fadd(pr[e], f[e]);
            if (j == q.npart - 1)
                s[e] = r == 0 ? pr[e] : fadd(s[e], pr[e]);
        }
    }
#pragma unroll
    for (int e = 0; e < 4; ++e)
        lg[e] = fmul(inv, s[e]);
    return late;
}

// 3: MiniMax2 top-4 of the warp's 128 logits. Lane k < 4 returns rank-k's (id, bf16-rounded weight).
// ROLL: the 4 tanh/key evaluations and the 4 selection rounds are rolled loops (select chains keep every value in
// registers); every arithmetic op, operand and order is the unrolled form's, so ids and weights are bit-equal.
template <int ROLL>
__device__ __forceinline__ void route_top4(float const lg[4], float const* bias, int lane, int& id_out, float& w_out)
{
    if constexpr (ROLL != 0)
    {
        float k0 = 0.f, k1 = 0.f, k2 = 0.f, k3 = 0.f;
#pragma unroll 1
        for (int i = 0; i < 4; ++i)
        {
            float const x = i == 0 ? lg[0] : i == 1 ? lg[1] : i == 2 ? lg[2] : lg[3];
            float const kk = fadd(ffma(0.5f, tanh_engine(fmul(0.5f, x)), 0.5f), bias[4 * lane + i]);
            k0 = i == 0 ? kk : k0;
            k1 = i == 1 ? kk : k1;
            k2 = i == 2 ? kk : k2;
            k3 = i == 3 ? kk : k3;
        }
        float const keys[4] = {k0, k1, k2, k3};
        unsigned hi[4];
        bool taken[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            hi[i] = twiddle(keys[i]);
            taken[i] = false;
        }
        int el = 0;
        float ktl = 0.f;
#pragma unroll 1
        for (int k = 0; k < TOPK; ++k)
        {
            unsigned bh = 0, bl = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                unsigned const lo = 65535u - (unsigned) (4 * lane + i);
                unsigned const hv = taken[i] ? 0u : hi[i], lv = taken[i] ? 0u : lo;
                bool const better = hv > bh || (hv == bh && lv > bl);
                bh = better ? hv : bh;
                bl = better ? lv : bl;
            }
            unsigned const mh = redux_max(bh);
            unsigned const ml = redux_max(bh == mh ? bl : 0u);
            int const e = 65535 - (int) ml;
            float kv = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                taken[i] = taken[i] || (e == 4 * lane + i);
                kv = (e == 4 * lane + i) ? keys[i] : kv;
            }
            float const kk = __shfl_sync(0xffffffffu, kv, e >> 2);
            el = lane == k ? e : el;
            ktl = lane == k ? kk : ktl;
        }
        float const sig = lane < TOPK ? fsub(ktl, bias[el]) : 0.f;
        float total = sig;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            total = fadd(total, __shfl_xor_sync(0xffffffffu, total, off));
        id_out = el;
        w_out = bf16_round(fdiv(fmul(sig, ROUTE_SCALE), fadd(total, SUM_EPS)));
        return;
    }
    float keys[4];
    unsigned hi[4];
    bool taken[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        float const sg = ffma(0.5f, tanh_engine(fmul(0.5f, lg[i])), 0.5f);
        keys[i] = fadd(sg, bias[4 * lane + i]);
        hi[i] = twiddle(keys[i]);
        taken[i] = false;
    }
    int ids[TOPK];
    float kt[TOPK];
#pragma unroll
    for (int k = 0; k < TOPK; ++k)
    {
        unsigned bh = 0, bl = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            unsigned const lo = 65535u - (unsigned) (4 * lane + i); // ties -> the lower id
            unsigned const hv = taken[i] ? 0u : hi[i], lv = taken[i] ? 0u : lo;
            bool const better = hv > bh || (hv == bh && lv > bl);
            bh = better ? hv : bh;
            bl = better ? lv : bl;
        }
        unsigned const mh = redux_max(bh);
        unsigned const ml = redux_max(bh == mh ? bl : 0u);
        int const e = 65535 - (int) ml;
        float kv = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            taken[i] = taken[i] || (e == 4 * lane + i);
            kv = (e == 4 * lane + i) ? keys[i] : kv;
        }
        ids[k] = e;
        kt[k] = __shfl_sync(0xffffffffu, kv, e >> 2);
    }
    float sig = 0.f;
    int el = 0;
#pragma unroll
    for (int k = 0; k < TOPK; ++k)
    {
        float const sk = fsub(kt[k], bias[ids[k]]);
        sig = lane == k ? sk : sig;
        el = lane == k ? ids[k] : el;
    }
    float total = sig;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        total = fadd(total, __shfl_xor_sync(0xffffffffu, total, off));
    id_out = el;
    w_out = bf16_round(fdiv(fmul(sig, ROUTE_SCALE), fadd(total, SUM_EPS)));
}

// 4: lanes < repl write replica `lane`'s line of row t (two 16-byte stores; every word carries the tag).
__device__ __forceinline__ void publish(RhParams const& q, int maxt, int t, int lane, unsigned tag, int id, float w)
{
    unsigned const bw = __float_as_uint(w) >> 16; // w is bf16-exact
    unsigned const lo = ((unsigned) id << 16) | bw;
    unsigned long long wd[TOPK];
#pragma unroll
    for (int k = 0; k < TOPK; ++k)
        wd[k] = ((unsigned long long) tag << 32) | (unsigned long long) __shfl_sync(0xffffffffu, lo, k);
    if (lane < q.repl)
    {
        unsigned long long* const l = q.route + ((int64_t) lane * maxt + t) * LINE_U64;
        st_relaxed_v2u64(l, wd[0], wd[1]);
        st_relaxed_v2u64(l + 2, wd[2], wd[3]);
    }
}
} // namespace rh

// X4 tail: x = normed2 quantised exactly as the engine's _build_b1_fused (fp4_quantize(x, sf_scale, 16): e4m3 block-16
// scale = e4m3(sfsc * amax / 6), values e2m1 of x * rcp(e4m3_to_f32(scale) / sfsc)).
namespace rhx
{
// The engine's exact PTX spellings (_rcp / _mul_rn / _e4m3_bits / _e4m3_to_f32 / _e2m1x2; abs/max non-ftz)
__device__ __forceinline__ float rcp_ftz(float a)
{
    float d;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a));
    return d;
}

__device__ __forceinline__ float fabs_(float a)
{
    float d;
    asm("abs.f32 %0, %1;" : "=f"(d) : "f"(a));
    return d;
}

__device__ __forceinline__ float fmax_(float a, float b)
{
    float d;
    asm("max.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

__device__ __forceinline__ unsigned e4m3_bits(float x)
{
    unsigned d;
    asm("{\n.reg .b16 t;\ncvt.rn.satfinite.e4m3x2.f32 t, 0f00000000, %1;\ncvt.u32.u16 %0, t;\n}" : "=r"(d) : "f"(x));
    return d & 0xFFu;
}

__device__ __forceinline__ float e4m3_to_f32(unsigned b)
{
    float d;
    asm("{\n.reg .b16 t, l, u;\n.reg .b32 h;\ncvt.u16.u32 t, %1;\ncvt.rn.f16x2.e4m3x2 h, t;\nmov.b32 {l, u}, h;\n"
        "cvt.f32.f16 %0, l;\n}"
        : "=f"(d)
        : "r"(b));
    return d;
}

__device__ __forceinline__ unsigned e2m1x2(float hi, float lo)
{
    unsigned d;
    asm("{\n.reg .b8 t;\ncvt.rn.satfinite.e2m1x2.f32 t, %1, %2;\ncvt.u32.u8 %0, t;\n}" : "=r"(d) : "f"(hi), "f"(lo));
    return d & 0xFFu;
}

// One NVFP4 block = 16 columns = this lane's 8 (c0 .. c0 + 7) and lane ^ 1's; c0 % 16 == 0 on even lanes. s: the
// bf16-rounded normed values (norm8's output). Warp-synchronous, full mask, like mxfp8_8.
__device__ __forceinline__ void x4_8(float const s[8], bool active, int t, int c0, float sc, uint8_t* xq, uint8_t* xsf)
{
    float am = 0.f;
    if (active)
    { // the engine's chain: |x0|, then max (all-NaN blocks stay NaN, as in _build_b1_fused)
        am = fabs_(s[0]);
#pragma unroll
        for (int e = 1; e < 8; ++e)
            am = fmax_(am, fabs_(s[e]));
    }
    float const vmax = fmax_(am, __shfl_xor_sync(0xffffffffu, am, 1));
    if (!active)
        return;
    unsigned const sfb = e4m3_bits(rh::fmul(sc, rh::fmul(vmax, rcp_ftz(6.0f))));
    float const osc = vmax != 0.f ? rcp_ftz(rh::fmul(e4m3_to_f32(sfb), rcp_ftz(sc))) : 0.f;
    unsigned w = 0u;
#pragma unroll
    for (int z = 0; z < 4; ++z)
        w |= e2m1x2(rh::fmul(s[2 * z + 1], osc), rh::fmul(s[2 * z], osc)) << (8 * z);
    *reinterpret_cast<unsigned*>(xq + (int64_t) t * (H / 2) + c0 / 2) = w;
    if ((threadIdx.x & 1) == 0)
        xsf[(int64_t) t * (H / 16) + c0 / 16] = (uint8_t) sfb;
}

// X4 2 (nvfp4_row_quant.py): enc = div.full(2688, amax), x_norm = bf16(x * enc), sf = e4m3(blockmax(x_norm) * rcp(6)),
// code = e2m1(x_norm * rcp(sf)); s_t = bf16(div.full(amax, 2688)).
__device__ __forceinline__ float div_full(float a, float b)
{
    float d;
    asm("div.full.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

__device__ __forceinline__ float bf16r(float a)
{
    return __bfloat162float(__float2bfloat16_rn(a));
}

__device__ __forceinline__ void x4_8_qat(
    float const s[8], bool active, int t, int c0, float enc, uint8_t* xq, uint8_t* xsf)
{
    float xn[8];
    float am = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e)
    {
        xn[e] = active ? bf16r(rh::fmul(s[e], enc)) : 0.f;
        am = fmax_(am, fabs_(xn[e]));
    }
    float const bm = fmax_(am, __shfl_xor_sync(0xffffffffu, am, 1));
    if (!active)
        return;
    unsigned const sfb = e4m3_bits(rh::fmul(bm, rcp_ftz(6.0f)));
    float const sff = e4m3_to_f32(sfb);
    float const inv = sff == 0.f ? 0.f : rcp_ftz(sff);
    unsigned w = 0u;
#pragma unroll
    for (int z = 0; z < 4; ++z)
        w |= e2m1x2(rh::fmul(xn[2 * z + 1], inv), rh::fmul(xn[2 * z], inv)) << (8 * z);
    *reinterpret_cast<unsigned*>(xq + (int64_t) t * (H / 2) + c0 / 2) = w;
    if ((threadIdx.x & 1) == 0)
        xsf[(int64_t) t * (H / 16) + c0 / 16] = (uint8_t) sfb;
}

// The argmax |s * w| (s, w) of this thread's 8 columns (w = norm8's weight, same ops): norm8 rounds (s * inv) * w, so
// the row's |normed| max is taken over each CTA's candidate after inv (exact up to fp32 near-ties between columns).
__device__ __forceinline__ void amax_cand8(
    float const s[8], bool active, uint4 gamma, int gemma, float& bp, float& bs, float& bw)
{
    bp = -1.f;
    bs = 0.f;
    bw = 0.f;
    if (!active)
        return;
    __nv_bfloat162 const* wv = reinterpret_cast<__nv_bfloat162 const*>(&gamma);
#pragma unroll
    for (int q = 0; q < 4; ++q)
    {
        float2 wf = __bfloat1622float2(wv[q]);
        if (gemma)
        {
            wf.x = add_f(1.0f, wf.x);
            wf.y = add_f(1.0f, wf.y);
        }
        float const p0 = fabs_(rh::fmul(s[2 * q], wf.x)), p1 = fabs_(rh::fmul(s[2 * q + 1], wf.y));
        if (p0 > bp)
        {
            bp = p0;
            bs = s[2 * q];
            bw = wf.x;
        }
        if (p1 > bp)
        {
            bp = p1;
            bs = s[2 * q + 1];
            bw = wf.y;
        }
    }
}

__device__ __forceinline__ void amax_cand_shfl(float& bp, float& bs, float& bw, int o)
{
    float const p = __shfl_xor_sync(0xffffffffu, bp, o), a = __shfl_xor_sync(0xffffffffu, bs, o),
                w = __shfl_xor_sync(0xffffffffu, bw, o);
    if (p > bp)
    {
        bp = p;
        bs = a;
        bw = w;
    }
}
} // namespace rhx

// Routing-warp post.
namespace prr
{
using namespace rh;
constexpr int W = 4;                        // world 4
constexpr int NP = 1;                       // rx parts per rank: lr4 pushes a + r as one part
constexpr int NB = W * NP;                  // world 4 blocks per row
constexpr unsigned long long HALF_NS = 384; // second rx round issued half a remote-L2 round after the first

__device__ __forceinline__ void rx_ld(unsigned long long const* row, int lane, ulonglong2 v[NB][2])
{
#pragma unroll
    for (int i = 0; i < NB; ++i)
    { // unconditional: a predicated reload stalls on the old value (see rx_issue_all)
        v[i][0] = ldv_u64x2(row + i * E + 4 * lane);
        v[i][1] = ldv_u64x2(row + i * E + 4 * lane + 2);
    }
}

__device__ __forceinline__ void rx_take(unsigned tag, const ulonglong2 v[NB][2], unsigned& pending, float acc[NB][4])
{
#pragma unroll
    for (int i = 0; i < NB; ++i)
        if (((pending >> i) & 1u) && tag2(v[i][0], tag) && tag2(v[i][1], tag))
        {
            pending &= ~(1u << i);
            acc[i][0] = __uint_as_float((unsigned) v[i][0].x);
            acc[i][1] = __uint_as_float((unsigned) v[i][0].y);
            acc[i][2] = __uint_as_float((unsigned) v[i][1].x);
            acc[i][3] = __uint_as_float((unsigned) v[i][1].y);
        }
}

__device__ __forceinline__ unsigned rx_stale(unsigned pending, const ulonglong2 v[NB][2])
{
    unsigned st = 0u;
#pragma unroll
    for (int i = NB - 1; i >= 0; --i)
        if ((pending >> i) & 1u)
            st = (unsigned) (v[i][0].x >> 32);
    return st;
}

// Before inv: poll the lane's rx words (exact tag match, globaltimer deadline as rx_finish) with two rounds in flight,
// then S_e in rx_finish<1>'s fixed (src, part) order. Returns late; stale = an observed tag of a pending word.
template <int RX2>
__device__ __forceinline__ bool rx_poll_sum(
    RhParams const& q, unsigned long long const* row, int lane, unsigned tag, float S[4], unsigned& stale)
{
    unsigned pending = (1u << NB) - 1u;
    float acc[NB][4];
    ulonglong2 a[NB][2], b[NB][2];
    bool late = false;
    unsigned long long t0 = 0ull;
    stale = 0u;
    rx_ld(row, lane, a);
    if constexpr (RX2 != 0)
    {
        unsigned long long const ti = gt();
        while (gt() - ti < HALF_NS)
        {
        }
#pragma unroll 1
        for (;;)
        {
            rx_ld(row, lane, b);
            rx_take(tag, a, pending, acc);
            if (!pending)
                break;
            unsigned long long now = gt();
            if (t0 == 0ull)
                t0 = now;
            else if (now - t0 > q.deadline_ns)
            {
                late = true;
                stale = rx_stale(pending, a);
                break;
            }
            rx_ld(row, lane, a);
            rx_take(tag, b, pending, acc);
            if (!pending)
                break;
            now = gt();
            if (now - t0 > q.deadline_ns)
            {
                late = true;
                stale = rx_stale(pending, b);
                break;
            }
        }
    }
    else
    {
#pragma unroll 1
        for (;;)
        {
            rx_take(tag, a, pending, acc);
            if (!pending)
                break;
            unsigned long long const now = gt();
            if (t0 == 0ull)
                t0 = now;
            else if (now - t0 > q.deadline_ns)
            {
                late = true;
                stale = rx_stale(pending, a);
                break;
            }
            rx_ld(row, lane, a);
        }
    }
    if (late)
    { // rx_finish accepts whatever the last read held; keep the same (garbage-tolerant) values
#pragma unroll
        for (int i = 0; i < NB; ++i)
            if ((pending >> i) & 1u)
            {
                acc[i][0] = __uint_as_float((unsigned) a[i][0].x);
                acc[i][1] = __uint_as_float((unsigned) a[i][0].y);
                acc[i][2] = __uint_as_float((unsigned) a[i][1].x);
                acc[i][3] = __uint_as_float((unsigned) a[i][1].y);
            }
    }
#pragma unroll
    for (int e = 0; e < 4; ++e)
        S[e] = fadd(fadd(fadd(acc[0][e], acc[1][e]), acc[2][e]), acc[3][e]); // world 4 npart 1: rx_finish<0>'s order
    return late;
}

struct Cand
{
    unsigned hi, lo;
    float sg;
};

__device__ __forceinline__ bool cgt(Cand const& a, Cand const& b)
{
    return a.hi > b.hi || (a.hi == b.hi && a.lo > b.lo);
}

__device__ __forceinline__ void cex(Cand& a, Cand& b)
{
    bool const s = cgt(b, a);
    Cand const x = s ? b : a, y = s ? a : b;
    a = x;
    b = y;
}

// After inv: route_top4<1> / route_top4_f's ids and weights, then publish's words. Lane l: 4 keys (unrolled), local
// descending sort of (twiddle(key), 65535 - id), 4 rounds over the lanes' heads only (the max head = the max not-taken
// entry: ids are unique), winner's sig broadcast by the round's shuffle. sum = (s0 + s2) + (s1 + s3) = the butterfly's
// value (its other levels add +0). Lane l stores words 2(l&1), 2(l&1)+1 of replica l >> 1 (16 replicas per pass).
__device__ __forceinline__ void route_pub(RhParams const& q, int maxt, int t, int lane, unsigned tag, float inv,
    float const S[4], float const bb[4], unsigned long long* ts_sel)
{
    Cand c[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        float const x = fmul(inv, S[i]);
        float const kk = fadd(ffma(0.5f, tanh_engine(fmul(0.5f, x)), 0.5f), bb[i]);
        c[i].hi = twiddle(kk);
        c[i].lo = 65535u - (unsigned) (4 * lane + i);
        c[i].sg = fsub(kk, bb[i]);
    }
    cex(c[0], c[1]);
    cex(c[2], c[3]);
    cex(c[0], c[2]);
    cex(c[1], c[3]);
    cex(c[1], c[2]);
    unsigned ek[TOPK];
    float sk[TOPK];
#pragma unroll
    for (int k = 0; k < TOPK; ++k)
    {
        unsigned const mh = redux_max(c[0].hi);
        unsigned const ml = redux_max(c[0].hi == mh ? c[0].lo : 0u);
        ek[k] = 65535u - ml;
        sk[k] = __shfl_sync(0xffffffffu, c[0].sg, (int) (ek[k] >> 2));
        if (c[0].lo == ml)
        {
            c[0] = c[1];
            c[1] = c[2];
            c[2] = c[3];
            c[3].hi = 0u;
            c[3].lo = 0u;
            c[3].sg = 0.f;
        }
    }
    if (ts_sel)
        *ts_sel = gt();
    float const total = fadd(fadd(sk[0], sk[2]), fadd(sk[1], sk[3]));
    float const den = fadd(total, SUM_EPS);
    int const h = lane & 1;
    float const s0 = h ? sk[2] : sk[0], s1 = h ? sk[3] : sk[1];
    unsigned const e0 = h ? ek[2] : ek[0], e1 = h ? ek[3] : ek[1];
    float const w0 = bf16_round(fdiv(fmul(s0, ROUTE_SCALE), den));
    float const w1 = bf16_round(fdiv(fmul(s1, ROUTE_SCALE), den));
    unsigned long long const hi = (unsigned long long) tag << 32;
    unsigned long long const d0 = hi | (e0 << 16) | (__float_as_uint(w0) >> 16);
    unsigned long long const d1 = hi | (e1 << 16) | (__float_as_uint(w1) >> 16);
#pragma unroll 1
    for (int r = lane >> 1; r < q.repl; r += 16)
        st_relaxed_v2u64(q.route + ((int64_t) r * maxt + t) * LINE_U64 + 2 * h, d0, d1);
}
} // namespace prr

// The prepushed post (rolled row path, X4 tail) + one routing warp (threadIdx >= TH_A, crank 0): rx poll and sums
// before inv, routing after inv, while warps 0..TH_A/32-1 run the row path (exchange, norm, MXFP8 / X4 tail).
// World 4, npart 1.
// The routing warp skips the CTA barrier (named barrier 1 over TH_A threads) and joins only the cluster barrier wait
// and the RMS mbarrier, so it never delays the row path. Stamps: [inv, rx done (before inv), selected, published]. X4:
// 0 off, 1 static x4, 2 per-token (M3.1 QAT) x4 + s_t. PW (DEV, M3LL_DEV_OPROJ_PTRIG=1): thread 0 of every CTA runs
// griddepcontrol.wait as its last instruction, after every poll, store and flip, so this post completes only after its
// o_proj grid has completed (rule 6 A6: a waiting, polling consumer). That lets the lr4 o_proj trigger after its wait
// (ptrig 1) instead of after its push. PW 0 is the shipped kernel (no wait anywhere).
template <int CL, int TH_A, int X4, int RX2, int PW = 0>
__global__ void __launch_bounds__(TH_A + 32, 1)
    lbv3rp_kernel(const __grid_constant__ V3Params p, const __grid_constant__ rh::RhParams q)
{
    static_assert(CL % 4 == 0, "a quarter-row parity slice must be whole CTAs");
    static_assert(TH_A % 32 == 0 && TH_A >= 32, "row warps");
    if (p.world != prr::W || (q.route != nullptr && q.tab == nullptr && q.npart != prr::NP))
        __trap();
    __shared__ float warp_sq[TH_A / 32];
    constexpr int XW = X4 == 2 ? 4 : 1; // exchange words per CTA: sq (+ X4 2: argmax |s w|, its s, its w)
    __shared__ alignas(16) float csq[CL * XW];
    __shared__ alignas(16) float warp_c[X4 == 2 ? 3 * (TH_A / 32) : 1];
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
    bool const rwarp = threadIdx.x >= TH_A;
    int const c0 = crank * cols + threadIdx.x * 8;
    bool const active = !rwarp && threadIdx.x * 8 < cols && t < p.rows;
    bool const router = rwarp && crank == 0 && t < p.rows && q.route != nullptr;
    bool const poll_rx = router && q.tab == nullptr;
    int const lane = threadIdx.x & 31;
    bool const stamp = router && q.tdbg != nullptr && lane == 0;
    int const fidx = t * 4 + crank / (CL / 4);
    uint4 wraw = make_uint4(0, 0, 0, 0), rr = make_uint4(0, 0, 0, 0);
    if (active)
    {
        wraw = *reinterpret_cast<uint4 const*>(p.gamma + c0);
        rr = *reinterpret_cast<uint4 const*>(p.residual + (int64_t) t * H + c0);
    }
    float x4sc = 0.f;
    if constexpr (X4 == 1)
    {
        if (!rwarp)
            x4sc = *q.x4_sfsc;
    }
    float bb[4] = {0.f, 0.f, 0.f, 0.f};
    unsigned prev_r = 0u, prev_x = 0u;
    if (poll_rx)
    {
        float4 const b4 = *reinterpret_cast<float4 const*>(q.bias + 4 * lane);
        bb[0] = b4.x;
        bb[1] = b4.y;
        bb[2] = b4.z;
        bb[3] = b4.w;
    }
    if (router && lane == 0)
    {
        prev_r = (unsigned) q.rctr[t * rh::RCTR_STRIDE];
        if (poll_rx)
            prev_x = (unsigned) q.rxc[t * rh::RCTR_STRIDE];
    }
    asm volatile("griddepcontrol.launch_dependents;");
    float s[8];
    float S[4] = {0.f, 0.f, 0.f, 0.f};
    unsigned long long ts_rx = 0ull;
    bool late = false;
    unsigned rtag = 0u, stale = 0u;
    int parity = 0;
    if (!rwarp)
    {
        parity = *reinterpret_cast<int volatile*>(p.flags + fidx) & 1;
        int64_t const half = (int64_t) p.maxt * p.world * (H / 2);
        int* const p0 = p.ex + parity * half + (int64_t) t * p.world * (H / 2) + c0 / 2;
        float sq = 0.f;
        if (active)
        {
            int4 v[prr::W];
            uint32_t pending = (1u << prr::W) - 1u;
            long long spins = 0;
            while (pending)
            {
#pragma unroll
                for (int r = 0; r < prr::W; ++r)
                    if ((pending >> r) & 1u)
                        v[r] = ldv_i4(p0 + (int64_t) r * (H / 2));
#pragma unroll
                for (int r = 0; r < prr::W; ++r)
                    if (((pending >> r) & 1u) && v[r].x != kEmpty && v[r].y != kEmpty && v[r].z != kEmpty
                        && v[r].w != kEmpty)
                        pending &= ~(1u << r);
                if (++spins > (1LL << 31))
                    __trap();
            }
#pragma unroll
            for (int r = 0; r < prr::W; ++r)
            {
                *reinterpret_cast<int4*>(p0 + (int64_t) r * (H / 2)) = make_int4(kEmpty, kEmpty, kEmpty, kEmpty);
                uint32_t const ww[4] = {(uint32_t) v[r].x, (uint32_t) v[r].y, (uint32_t) v[r].z, (uint32_t) v[r].w};
#pragma unroll
                for (int qq = 0; qq < 4; ++qq)
                {
                    float const lo = __uint_as_float(ww[qq] << 16), hi = __uint_as_float(ww[qq] & 0xFFFF0000u);
                    s[2 * qq] = r == 0 ? lo : add_f(s[2 * qq], lo);
                    s[2 * qq + 1] = r == 0 ? hi : add_f(s[2 * qq + 1], hi);
                }
            }
            uint4 const u = residual_add8(s, rr, sq);
            *reinterpret_cast<uint4*>(p.out_res + (int64_t) t * H + c0) = u;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            sq = add_f(sq, __shfl_xor_sync(0xffffffffu, sq, o));
        if (lane == 0)
            warp_sq[threadIdx.x / 32] = sq;
        if constexpr (X4 == 2)
        {
            float bp, bs, bw;
            rhx::amax_cand8(s, active, wraw, p.gemma, bp, bs, bw);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                rhx::amax_cand_shfl(bp, bs, bw, o);
            if (lane == 0)
            {
                int const w = threadIdx.x / 32;
                warp_c[3 * w] = bp;
                warp_c[3 * w + 1] = bs;
                warp_c[3 * w + 2] = bw;
            }
        }
        asm volatile("bar.sync 1, %0;" ::"n"(TH_A) : "memory");
    }
    else if (poll_rx)
    {
        rtag = __shfl_sync(0xffffffffu, prev_x, 0) + 1u;
        unsigned long long const* const rxrow = q.rx + (int64_t) t * prr::W * prr::NP * rh::E;
        late = prr::rx_poll_sum<RX2>(q, rxrow, lane, rtag, S, stale);
        unsigned const lm = __ballot_sync(0xffffffffu, late);
        if (lm != 0u && !q.hang_dbg)
            __trap();
        if (lane == 0)
        {
            int* const xr = q.rxc + t * rh::RCTR_STRIDE;
            xr[0] = (int) rtag; // consumed: only this path bumps rxc
            if (lm != 0u)
            {
                xr[1] += 1;
                xr[2] = (int) rtag;
            }
        }
        unsigned const st = __shfl_sync(0xffffffffu, stale, lm ? __ffs(lm) - 1 : 0);
        if (lane == 0 && lm != 0u)
            q.rxc[t * rh::RCTR_STRIDE + 3] = (int) st;
        ts_rx = stamp ? gt() : 0ull;
    }
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    if (threadIdx.x == 0)
    {
        float tot = 0.f;
#pragma unroll
        for (int w = 0; w < TH_A / 32; ++w)
            tot = add_f(tot, warp_sq[w]);
        uint32_t const b = static_cast<uint32_t>(__cvta_generic_to_shared(&xbar));
        uint32_t const d = static_cast<uint32_t>(__cvta_generic_to_shared(&csq[crank * XW]));
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(CL * 4 * XW) : "memory");
        if constexpr (X4 == 2)
        {
            float bp = warp_c[0], bs = warp_c[1], bw = warp_c[2];
#pragma unroll
            for (int w = 1; w < TH_A / 32; ++w)
                if (warp_c[3 * w] > bp)
                {
                    bp = warp_c[3 * w];
                    bs = warp_c[3 * w + 1];
                    bw = warp_c[3 * w + 2];
                }
            for (int c = 0; c < CL; ++c)
            {
                uint32_t rd, rb;
                asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rd) : "r"(d), "r"(c));
                asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rb) : "r"(b), "r"(c));
                asm volatile(
                    "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(
                        rd),
                    "r"(__float_as_uint(tot)), "r"(__float_as_uint(bp)), "r"(__float_as_uint(bs)),
                    "r"(__float_as_uint(bw)), "r"(rb)
                    : "memory");
            }
        }
        else
        {
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
    }
    if (rwarp && !router)
        return; // routing warp of crank != 0 / idle rows: nothing after the cluster barrier
    {
        uint32_t const b = static_cast<uint32_t>(__cvta_generic_to_shared(&xbar));
        uint32_t done = 0;
        while (!done)
            asm volatile(
                "{\n .reg .pred qq;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 qq, [%1], 0;\n selp.u32 "
                "%0, 1, 0, qq;\n}"
                : "=r"(done)
                : "r"(b)
                : "memory");
    }
    if (threadIdx.x == 0 && crank % (CL / 4) == 0 && t < p.rows)
        p.flags[fidx] = parity ^ 1;
    float total = 0.f;
#pragma unroll
    for (int c = 0; c < CL; ++c)
        total = add_f(total, csq[c * XW]);
    float const inv = inv_rms(total, p.eps);
    if (rwarp)
    { // == router
        unsigned long long const ts1 = stamp ? gt() : 0ull;
        unsigned long long ts_t4 = 0ull;
        unsigned tag = __shfl_sync(0xffffffffu, prev_r, 0) + 1u;
        tag = tag == 0u || tag >= 0x80000000u ? 1u : tag; // pack_route's 0 < tag < 2^31
        if (q.tab)
        {
            unsigned const v = lane < rh::TOPK ? q.tab[t * rh::TOPK + lane] : 0u;
            int const id = (int) (v >> 16);
            float const w = __uint_as_float(v << 16);
            if (q.delay_ns > 0)
            {
                unsigned long long const d0 = gt();
                while (gt() - d0 < (unsigned long long) q.delay_ns)
                {
                }
            }
            rh::publish(q, q.rmaxt, t, lane, tag, id, w);
        }
        else
        {
            if (q.lg_dbg)
                *reinterpret_cast<float4*>(q.lg_dbg + (int64_t) t * rh::E + 4 * lane)
                    = make_float4(rh::fmul(inv, S[0]), rh::fmul(inv, S[1]), rh::fmul(inv, S[2]), rh::fmul(inv, S[3]));
            prr::route_pub(q, q.rmaxt, t, lane, tag, inv, S, bb, stamp ? &ts_t4 : nullptr);
        }
        if (lane == 0)
            q.rctr[t * rh::RCTR_STRIDE] = (int) tag;
        if (stamp)
        {
            unsigned long long* const d = q.tdbg + ((int64_t) ((tag - 1u) & 15u) * q.rmaxt + t) * rh::TDBG_W;
            d[0] = ts1;
            d[1] = ts_rx;
            d[2] = ts_t4;
            d[3] = gt(); // inv, rx done (pre-inv), selected, published
        }
        return;
    }
    if (active)
    {
        uint4 const n = norm8(s, inv, wraw, p.gemma);
        if (p.out_norm)
            *reinterpret_cast<uint4*>(p.out_norm + (int64_t) t * H + c0) = n;
    }
    if constexpr (X4 == 1)
        rhx::x4_8(s, active, t, c0, x4sc, q.x4_q, q.x4_sf);
    if constexpr (X4 == 2)
    { // amax = max over the CTAs' candidates of |bf16((s * inv) * w)| (norm8's ops)
        float ra = 0.f;
#pragma unroll
        for (int c = 0; c < CL; ++c)
            if (csq[c * XW + 1] >= 0.f)
                ra = rhx::fmax_(ra, rhx::fabs_(rhx::bf16r(mul_f(mul_f(csq[c * XW + 2], inv), csq[c * XW + 3]))));
        ra = rhx::fmax_(ra, 1e-12f);
        rhx::x4_8_qat(s, active, t, c0, rhx::div_full(2688.0f, ra), q.x4_q, q.x4_sf);
        if (threadIdx.x == 0 && crank == 0 && t < p.rows)
            q.x4_pts[t] = rhx::bf16r(rhx::div_full(ra, 2688.0f));
    }
    if (p.out_q != nullptr)
        mxfp8_8(s, active, t, c0, p.out_q, p.out_sf, p.sf_k_tiles);
    if constexpr (PW != 0)
    {
        if (threadIdx.x == 0)
            asm volatile("griddepcontrol.wait;" ::: "memory");
    }
}
