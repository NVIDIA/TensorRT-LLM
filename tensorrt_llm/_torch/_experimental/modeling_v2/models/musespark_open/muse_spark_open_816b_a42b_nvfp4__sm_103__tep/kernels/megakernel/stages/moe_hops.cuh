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

// The MoE side of a decoder layer around the experts, as megakernel stages over 8 ranks (one-shot hops in the kernel,
// data as its own flag; layout in moe_hops_layout.cuh):
//
//   MhOproj (stub)  this rank's o_proj partial [8, 8192] BF16 -> slot `rank` of every rank's hop1 area
//   MhH1            hop1 consumer: the 8 slots summed in rank order from +0 (one BF16 rounding), the weightless post-
//                   attention norm, the gated FP32 residual, the Gemma pre-FFN norm -> x_ffn (arena); r' stays in
//                   shared memory for MhF1 on the same CTA
//   MhRouter        one expert row (FP32 [8192], 32 KiB pre-staged): 8 logits x 1/sqrt(32) -> every rank's logit area
//   MhPool          a 16-row tile of the rank's pool slice, one K half (128 KiB pre-staged): FP32 partial [8][16]
//                   -> every rank's pool area (the reader sums the two halves and rounds once to BF16)
//   MhExpStub (stub) the experts' place: waits for its share of the gathered pool output and of the logits (F2'),
//                   then publishes this rank's given weighted-sum partial into every rank's finalize area
//   MhFinal         finalize all-reduce: the 8 partials summed in rank order from +0, one BF16 rounding -> m (arena)
//   MhUnpool        a 16-row tile of the rank's unpool slice (128 KiB pre-staged), BF16 -> every rank's F1 area
//   MhF1            F1 consumer: the weight-first post-FFN norm, the gated residual, the next layer's pre-QKV Gemma
//                   norm -> x_attn, the residual, and the DFlash capture bf16(residual) on capture layers
//   MhClear         the previous launch's copy of the arena and of the F2' areas
//
// The two boundaries (MhH1, MhF1) are the stock boundary kernel (musespark_open_boundary._ClusterBoundaryKernel at 8
// rows: cluster 8 x 128 threads, 8 consecutive columns per thread) statement for statement, given its FMA contraction
// (MSO_MK_MH_FMA = 1 matches the stock's bits). A row has U = kMhRowU units of W = 8192 / U columns; vector j of thread
// t covers columns q W + 8 (t + 256 j), so its warp butterfly is the stock's warp slot 32 q / U + 8 j + w, and the
// row's 32 slots are summed by the stock's second butterfly (in shared memory for U = 1, through an arena row
// otherwise). Their coefficients are unit weights, pre-staged in shared memory at kernel entry.
//
// Ownership (launch n, copy c = n % 3): every element of the hop1 / fin / f1 areas has one writer (the source rank's
// unit, once) and one reader on each rank (the consumer thread owning those columns), which resets it after its output;
// the next write into copy c is launch n + 3, ordered after the reset by kernel boundaries on both ranks. The pool /
// logit areas and the arena may be read by several units (the routing reads every logit on every CTA): the MhClear
// units of launch n empty copy (n + 2) % 3, whose readers (launch n - 1) completed before launch n's dependency wait
// and whose next writers (launch n + 2 on any rank) start only after this launch completed on this rank. MhH1(u) and
// MhF1(u) run on the same CTA with the same thread map, and r' passes between them in the CTA's shared memory.
//
// Publication: one store through the multicast address, or (kFlagMhUnicast) one store into each rank's unicast mapping.

#pragma once

#include "moe_hops_xchg.cuh"

// ---- rank-local router units (stageMhRouterLocal): unit = kMhRouterLocalBit + 64 h + q ----
constexpr int kMhRouterLocalBit = 512;
constexpr int kMhRouterLocalExperts = 4; // experts per unit (4 q .. 4 q + 3 of all 256)

// ---- end rank-local router units ----

// The boundary's per-thread arithmetic, with the contraction the stock kernel's compiler applies (MSO_MK_MH_FMA).
__device__ __forceinline__ float mhSq(float total, float v)
{
#if MSO_MK_MH_FMA == 0
    return __fadd_rn(total, __fmul_rn(v, v));
#else
    return __fmaf_rn(v, v, total);
#endif
}

__device__ __forceinline__ float mhGated(float a, float r, float b, float v)
{
#if MSO_MK_MH_FMA == 0
    return __fadd_rn(__fmul_rn(a, r), __fmul_rn(b, v));
#elif MSO_MK_MH_FMA == 1
    return __fmaf_rn(a, r, __fmul_rn(b, v));
#else
    return __fmaf_rn(b, v, __fmul_rn(a, r));
#endif
}

__device__ __forceinline__ float mhRsqrt(float total, float eps)
{
#if MSO_MK_FZ
    // the mean as total * 2^-13, bitwise equal to total / 8192 for every input (scaling by a power of two: both are
    // the correctly rounded value of the same real number), so the boundaries call no shared division subroutine,
    // whose register assignment every caller of it shares
    return rsqrtf(__fadd_rn(__fmul_rn(total, 1.f / 8192.f), eps));
#else
    return rsqrtf(__fadd_rn(__fdiv_rn(total, 8192.f), eps));
#endif
}

__device__ __forceinline__ uint8_t* mhUc(Ctx const& ctx, int copy)
{
    return ptr<uint8_t>(ctx, kPMhUc) + static_cast<size_t>(copy) * kMhCopyBytes;
}

__device__ __forceinline__ uint8_t* mhArena(Ctx const& ctx, int copy)
{
    return ptr<uint8_t>(ctx, kPMhArena) + static_cast<size_t>(copy) * kMhArenaBytes;
}

__device__ __forceinline__ int mhRank(Ctx const& ctx)
{
    return static_cast<int>(ctx.p[kPMhRank]);
}

// ---- canary-gated polls (kIFlagT3Canary*) ----
// The CTA waits until warp 0 has seen word a (lane l: the canary of producer unit l, nullptr: none) non-empty on every
// lane, then the barrier. Only a heuristic gate in front of the consumer's full poll: that poll still checks every
// word, so any canary word is correct; each canary is the last 16-byte store of a producer thread (the unit's last
// columns).
__device__ __forceinline__ void t3Canary(Ctx const& ctx, uint8_t const* a, int code)
{
    if ((threadIdx.x >> 5) == 0)
    {
        poll1(ctx, a, code);
    }
    __syncthreads();
}

// The canary of boundary unit u = (row u / U, part u % U) in a BF16 [8][8192] output written by its thread map (x_ffn
// by MhH1, x_attn by MhF1): the last word of its last vector. nullptr unless the unit's rows / columns meet
// [row0, row1) x [col0, col1).
__device__ __forceinline__ uint8_t const* t3RowCanary(
    uint8_t const* base, int u, int row0, int row1, int col0, int col1)
{
    int const row = u / kMhRowU;
    int const c0 = (u % kMhRowU) * kMhRowW;
    bool const meets = u < kMhBoundaryUnits && row >= row0 && row < row1 && c0 < col1 && c0 + kMhRowW > col0;
    return meets ? base + (static_cast<size_t>(row) * 8192 + c0 + kMhRowW - 8) * 2 + 12 : nullptr;
}

// ---- end canary-gated polls ----

// One 16-byte vector at byte `off` of this launch's copy, on every rank.
__device__ __forceinline__ void mhPublish(Ctx const& ctx, size_t off, uint4 v)
{
    size_t const o = static_cast<size_t>(ctx.copy) * kMhCopyBytes + off;
    if (MSO_MK_KFLAGS(ctx.flags) & kFlagMhUnicast) // MSO_MK_LD_KF
    {
        int64_t const* peers = ptr<int64_t const>(ctx, kPMhPeers);
#pragma unroll
        for (int r = 0; r < kMhRanks; ++r)
        {
            st128(reinterpret_cast<uint8_t*>(__ldg(peers + r)) + o, v);
        }
    }
    else
    {
        st128(ptr<uint8_t>(ctx, kPMhMc) + o, v);
    }
}

__device__ __forceinline__ void ld8f(float (&v)[8], float const* p)
{
    float4 const a = ldcgF4(p);
    float4 const b = ldcgF4(p + 4);
    v[0] = a.x, v[1] = a.y, v[2] = a.z, v[3] = a.w, v[4] = b.x, v[5] = b.y, v[6] = b.z, v[7] = b.w;
}

// 8 FP32 at a shared address.
__device__ __forceinline__ void lds8f(float (&v)[8], uint32_t a)
{
    uint4 const x = lds128(a);
    uint4 const y = lds128(a + 16);
    v[0] = __uint_as_float(x.x), v[1] = __uint_as_float(x.y), v[2] = __uint_as_float(x.z), v[3] = __uint_as_float(x.w);
    v[4] = __uint_as_float(y.x), v[5] = __uint_as_float(y.y), v[6] = __uint_as_float(y.z), v[7] = __uint_as_float(y.w);
}

__device__ __forceinline__ uint4 f4u(float a, float b, float c, float d)
{
    return make_uint4(__float_as_uint(a), __float_as_uint(b), __float_as_uint(c), __float_as_uint(d));
}

__device__ __forceinline__ void st8f(float* p, float const (&v)[8])
{
    st128(p, f4u(v[0], v[1], v[2], v[3]));
    st128(p + 4, f4u(v[4], v[5], v[6], v[7]));
}

__device__ __forceinline__ void unpack8(float (&v)[8], uint4 w)
{
    uint32_t const u[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        v[2 * j] = bf16Lo(u[j]);
        v[2 * j + 1] = bf16Hi(u[j]);
    }
}

__device__ __forceinline__ uint4 pack8(float const (&v)[8])
{
    return make_uint4(packBf16(v[0], v[1]), packBf16(v[2], v[3]), packBf16(v[4], v[5]), packBf16(v[6], v[7]));
}

__device__ __forceinline__ float* mhSsqRow(Ctx const& ctx, int phase, int row)
{
    return reinterpret_cast<float*>(mhArena(ctx, ctx.copy) + kMhArSsq) + (phase * 8 + row) * 32;
}

// Vector j of this thread in a boundary unit: local column 8 (t + 256 j) of the unit's kMhRowW.
__device__ __forceinline__ int mhLocalCol(int j)
{
    return 8 * (static_cast<int>(threadIdx.x) + 256 * j);
}

// The stock boundary's row sum of one norm phase. part[j] = this thread's stock-thread partial of vector j (its 8
// values in order); per j the warp butterfly (16, 8, 4, 2, 1) is stock warp slot 32 q / U + 8 j + w; every warp then
// reads the row's 32 slots (lane l: 0 + slot l) and butterflies them again. All 256 threads call it.
__device__ __forceinline__ float mhRowSum(Ctx const& ctx, float const (&part)[kMhRowV], int row, int q, int phase)
{
    int const lane = threadIdx.x & 31;
    int const warp = threadIdx.x >> 5;
    float* slots
        = kMhRowU == 1 ? reinterpret_cast<float*>(ctx.smem + kSmemRed) + 32 * phase : mhSsqRow(ctx, phase, row);
    float wv[kMhRowV];
    int ws[kMhRowV];
#pragma unroll
    for (int j = 0; j < kMhRowV; ++j)
    {
        float v = part[j];
#pragma unroll
        for (int m = 16; m >= 1; m >>= 1)
        {
            v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, m));
        }
        int const slot = (32 / kMhRowU) * q + 8 * j + warp;
        wv[j] = v;
        ws[j] = slot;
        bool const t3Vec = kMhRowU > 1 && kMhXchg == 1 && (icPIFlags(ctx.p[kPIFlags]) & kIFlagT3XchgVec);
        if (lane == 0 && (kMhRowU == 1 || kMhXchg != 2) && !t3Vec)
        {
            if constexpr (kMhRowU == 1)
            {
                slots[slot] = v;
            }
            else
            {
                st32(slots + slot, lamport(__float_as_uint(v)));
            }
        }
    }
    float t;
    if constexpr (kMhRowU == 1)
    {
        __syncthreads();
        t = __fadd_rn(0.f, slots[lane]);
    }
    else if constexpr (kMhXchg == 2)
    {
#if !MSO_MK_LEAN
        if (clusterCtaRank() != static_cast<uint32_t>(q))
        {
            __trap();
        }
#endif
        t = __fadd_rn(0.f, mhXchgDsmem<kMhRowV>(ctx, wv, ws, phase));
    }
    else if constexpr (kMhXchg == 1)
    {
        float* stage = reinterpret_cast<float*>(ctx.smem + kSmemRed) + 32 * phase;
        // ---- kIFlagT3XchgVec: the CTA's 32 / U slots gathered in shared memory first, then published by warp 0 as
        // 16-byte vectors (2 stores per CTA instead of 8 single words into the row's line) and polled as 8 vectors ----
        if (icPIFlags(ctx.p[kPIFlags]) & kIFlagT3XchgVec)
        {
            constexpr int kOwn = 32 / kMhRowU;
            float* own = reinterpret_cast<float*>(ctx.smem + kSmemRed) + 128 + kOwn * phase;
            if (lane == 0)
            {
#pragma unroll
                for (int j = 0; j < kMhRowV; ++j)
                {
                    own[8 * j + warp] = wv[j];
                }
            }
            __syncthreads();
            if (warp == 0)
            {
                if (lane < kOwn / 4)
                {
                    uint4 const w = *reinterpret_cast<uint4 const*>(own + 4 * lane);
                    st128(slots + kOwn * q + 4 * lane, lamport4(w));
                }
                uint8_t const* const a[1] = {lane < 8 ? reinterpret_cast<uint8_t const*>(slots + 4 * lane) : nullptr};
                uint4 r[1];
                pollVecs<1>(ctx, r, a, kCodeMhSsq);
                if (lane < 8)
                {
                    *reinterpret_cast<uint4*>(stage + 4 * lane) = r[0];
                }
            }
        }
        else
            // ---- end kIFlagT3XchgVec ----
            if (warp == 0)
            {
                stage[lane] = __uint_as_float(poll1(ctx, slots + lane, kCodeMhSsq));
            }
        __syncthreads();
        t = __fadd_rn(0.f, stage[lane]);
    }
    else
    {
        t = __fadd_rn(0.f, __uint_as_float(poll1(ctx, slots + lane, kCodeMhSsq)));
    }
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1)
    {
        t = __fadd_rn(t, __shfl_xor_sync(0xffffffffu, t, m));
    }
    return t;
}

#if MSO_MK_NB
// ---- MSO_MK_NB: the one-exchange boundary (bit 1 at MhH1, phase 0; bit 2 at MhF1, phase 2) ----
// One row exchange of four sums: .x = the stock first-norm sum (part[j]: the stock thread partial of vector j, the warp
// butterfly and slot (32 / U) q + 8 j + warp exactly as mhRowSum, so the row total is bitwise the stock one), and .y,
// .z, .w = the row sums of this thread's a, b, c (they ride in vector j = 0's slot; the other slots carry 0). Each slot
// is one 16-byte Lamport vector in the arena's kMhArAbc rows of boundary bnd; warp 0 of each of the row's U units polls
// the row's 32 vectors (lane l: slot l) and passes them on in shared memory ([kSmemRed, +512), free while the unit
// runs, as for mhRowSum). Ownership: every slot has one writer (lane 0 of one warp of one row unit, once per launch)
// and U readers (warp 0 of each row unit, the same launch); MhClear of the next launch empties the copy (the slots lie
// inside [0, kMhArenaBytes), the rule of mhSsqRow's slots). All 256 threads call it.
__device__ __forceinline__ float4 nbRowSum4(
    Ctx const& ctx, float const (&part)[kMhRowV], float a, float b, float c, int row, int q, int bnd)
{
    int const lane = threadIdx.x & 31;
    int const warp = threadIdx.x >> 5;
    float4* const slots = reinterpret_cast<float4*>(mhArena(ctx, ctx.copy) + kMhArAbc) + (bnd * 8 + row) * 32;
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1)
    {
        a = __fadd_rn(a, __shfl_xor_sync(0xffffffffu, a, m));
        b = __fadd_rn(b, __shfl_xor_sync(0xffffffffu, b, m));
        c = __fadd_rn(c, __shfl_xor_sync(0xffffffffu, c, m));
    }
#pragma unroll
    for (int j = 0; j < kMhRowV; ++j)
    {
        float v = part[j];
#pragma unroll
        for (int m = 16; m >= 1; m >>= 1)
        {
            v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, m));
        }
        if (lane == 0)
        {
            int const slot = (32 / kMhRowU) * q + 8 * j + warp;
            st128(slots + slot, lamport4(f4u(v, j == 0 ? a : 0.f, j == 0 ? b : 0.f, j == 0 ? c : 0.f)));
        }
    }
    float4* const stage = reinterpret_cast<float4*>(ctx.smem + kSmemRed);
    if (warp == 0)
    {
        uint8_t const* const p[1] = {reinterpret_cast<uint8_t const*>(slots + lane)};
        uint4 r[1];
        pollVecs<1>(ctx, r, p, kCodeMhSsq);
        stage[lane] = make_float4(
            __uint_as_float(r[0].x), __uint_as_float(r[0].y), __uint_as_float(r[0].z), __uint_as_float(r[0].w));
    }
    __syncthreads();
    float4 const s = stage[lane];
    float t = __fadd_rn(0.f, s.x);
    float y = s.y, z = s.z, w = s.w;
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1)
    {
        t = __fadd_rn(t, __shfl_xor_sync(0xffffffffu, t, m));
        y = __fadd_rn(y, __shfl_xor_sync(0xffffffffu, y, m));
        z = __fadd_rn(z, __shfl_xor_sync(0xffffffffu, z, m));
        w = __fadd_rn(w, __shfl_xor_sync(0xffffffffu, w, m));
    }
    return make_float4(t, y, z, w);
}

// ---- end MSO_MK_NB ----
#endif
#if MSO_MK_RL
// ---- MSO_MK_RL builds (msoMegakernel.cu): word i of the layer's rl block (kPRl) as a pointer ----
template <typename T>
__device__ __forceinline__ T* rlBlockPtr(Ctx const& ctx, int i)
{
    return reinterpret_cast<T*>(__ldg(reinterpret_cast<long long const*>(ctx.p[kPRl]) + i));
}

// ---- end MSO_MK_RL ----
#endif
#if MSO_MK_RL & 1
// ---- MSO_MK_RL bit 1 ----
// The rl unit of MhRouter's row-split unit code (expert row e, row half h): i = e + 32 h.
__device__ __forceinline__ int rlUnitOf(int unit)
{
    return (unit & (kMhLocalExperts - 1)) + kMhLocalExperts * ((unit >> 5) & 1);
}

// ---- end MSO_MK_RL bit 1 ----
#endif

#if MSO_MK_BH
// ---- MSO_MK_BH: two-term consumers at H1 (moe_hops_layout.cuh). MhH1 publishes the residual term rA = (alpha1 r)
// (w + 1) before its hop-1 poll (bit 2) and the branch term xB = (beta1 x)(w + 1) right after its rank sum (in x_ffn's
// words), each as a BF16 plane (bit 4: and a lo plane bf16(t - bf16(t))); MhPool / MhRouter publish
// inv2 (L(rA) + inv1 L(xB)) with inv1 / inv2 recomputed from the boundary's one-exchange slots. Ownership: every plane
// word has one writer (the MhH1 thread owning the columns, once per launch) and the rank's pool / router units as
// readers in the same launch; the planes lie in the arena copy, which the next launch's MhClear empties (the rule of
// x_ffn's words). ----

// The H1 boundary's inv1 / inv2 from the row's 32 one-exchange slot vectors (lane l: slot l of the row): nbRowSum4's
// second butterfly and mhBoundary's formula statement for statement, so the scalars are MhH1's bits. All lanes of the
// warp call it and get them.
__device__ __forceinline__ float2 bhInvs(uint4 r)
{
    float t = __fadd_rn(0.f, __uint_as_float(r.x));
    float y = __uint_as_float(r.y), z = __uint_as_float(r.z), w = __uint_as_float(r.w);
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1)
    {
        t = __fadd_rn(t, __shfl_xor_sync(0xffffffffu, t, m));
        y = __fadd_rn(y, __shfl_xor_sync(0xffffffffu, y, m));
        z = __fadd_rn(z, __shfl_xor_sync(0xffffffffu, z, m));
        w = __fadd_rn(w, __shfl_xor_sync(0xffffffffu, w, m));
    }
    float const inv = mhRsqrt(t, 1e-8f);
    float const s2 = __fmaf_rn(__fmul_rn(inv, inv), w, __fmaf_rn(__fmul_rn(2.f, inv), z, y));
    return make_float2(inv, mhRsqrt(fmaxf(s2, 0.f), 1e-5f));
}

// Slot `lane` of row `row` of the H1 boundary's one-exchange rows (kMhArAbc, boundary 0) of this launch's copy.
__device__ __forceinline__ uint8_t const* bhSlot(Ctx const& ctx, int row, int lane)
{
    return mhArena(ctx, ctx.copy) + kMhArAbc + (static_cast<size_t>(row) * 32 + lane) * 16;
}

// One plane vector of a term (8 FP32 at columns [col, col + 8) of row `row`): bf16(t) into the plane at hiOff and, with
// loOff >= 0, bf16(t - bf16(t)) into the plane at loOff (t - bf16(t) is exact in FP32), as data-as-flag words.
__device__ __forceinline__ void bhPublishTerm(
    Ctx const& ctx, float const (&t)[8], int row, int col, int hiOff, int loOff)
{
    uint8_t* const ar = mhArena(ctx, ctx.copy);
    size_t const off = (static_cast<size_t>(row) * 8192 + col) * 2;
    uint4 const hi = pack8(t);
    st128(ar + hiOff + off, lamport4(hi));
    if (loOff >= 0)
    {
        float h[8], lo[8];
        unpack8(h, hi);
#pragma unroll
        for (int v = 0; v < 8; ++v)
        {
            lo[v] = __fsub_rn(t[v], h[v]);
        }
        st128(ar + loOff + off, lamport4(pack8(lo)));
    }
}

// MhH1's term of vector j: (c[v] x[v]) (w[v] + 1) with c at the shared address cS (alpha1 or beta1, FP32) and the
// pre-FFN weight w at wS (BF16), both at the thread's local columns.
__device__ __forceinline__ void bhTerm(float (&t)[8], float const (&x)[8], uint32_t cS, uint32_t wS, int lc)
{
    float c[8], w[8];
    lds8f(c, cS + 4 * lc);
    unpack8(w, lds128(wS + 2 * lc));
#pragma unroll
    for (int v = 0; v < 8; ++v)
    {
        t[v] = __fmul_rn(__fmul_rn(c[v], x[v]), __fadd_rn(w[v], 1.f));
    }
}

// The pool's B fragment addresses of one plane (8 chunks of K half h; warp w: chunks 64 h + w + 8 i, as mhPollB) into
// a[base, base + 16).
template <int kN>
__device__ __forceinline__ void bhPoolAddrs(uint8_t const* (&a)[kN], int base, uint8_t const* plane, int h)
{
    int const lane = threadIdx.x & 31;
    int const warp = threadIdx.x >> 5;
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        int const col = (64 * h + warp + 8 * i) * kChunkCols + 16 * (lane & 3);
        a[base + 2 * i] = plane + (static_cast<size_t>(lane >> 2) * 8192 + col) * 2;
        a[base + 2 * i + 1] = a[base + 2 * i] + 16;
    }
}

// The pool tile's 8 chunks (warp w: chunks w + 8 i of the unit's weights) times the B fragments v[base, base + 16),
// into the 4 interleaved partials of mhGemvTile.
template <int kN>
__device__ __forceinline__ void bhPoolMma(
    Ctx const& ctx, uint32_t wOff, uint4 const (&v)[kN], int base, float (&part)[4][4])
{
    int const warp = threadIdx.x >> 5;
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        uint4 const b[2] = {v[base + 2 * i], v[base + 2 * i + 1]};
        mmaChunkBf16(part[i & 3], ctx.smemBase + wOff + static_cast<uint32_t>(warp + 8 * i) * kBf16Block, b);
    }
}

// ---- MSO_MK_BH's pool polls under MSO_MK_PB bit 2 (MhPool's input polls). bhStageMhPool replaces stageMhPool's
// mhPollB, so with MSO_MK_BH the PB bit 2 site is the pool's plane polls: the 16 B-fragment vectors of a plane
// (bhPoolAddrs: pairs a[2 i], a[2 i] + 16, 32-byte aligned) as one 256-bit load per fragment (pollVecs256, mhPollB's
// kPb: the same words, lanes, registers and strength; a plane word is written once per copy and cleared only in
// another copy than the one the launch polls), and the xB poll's 17th vector (the row's one-exchange slot) as one
// 16-byte load in the same rounds and under the same exit vote (pollVecs256Tail). Value-identical by construction. ----
static_assert(kMhArX % 32 == 0 && kMhArRA % 32 == 0 && kMhPlaneBytes % 32 == 0 && kMhArenaBytes % 32 == 0
        && (kChunkCols * 2) % 32 == 0,
    "MSO_MK_BH with MSO_MK_PB bit 2: plane vectors in 32-byte pairs");

// pollVecs256<kN> over a[0, 2 kN) and one 16-byte vector a[2 kN] (or nullptr), polled in the same rounds: the exit
// test is the conjunction over all 2 kN + 1 vectors, as pollVecs<2 kN + 1> over the same addresses. The whole warp
// calls it.
template <int kN>
__device__ __forceinline__ void pollVecs256Tail(
    Ctx const& ctx, uint4 (&v)[2 * kN + 1], uint8_t const* const (&a)[2 * kN + 1], int code)
{
#if MSO_MK_SY_POLL
    // the one-loop form of pollVecs (MSO_MK_SY_POLL = 1)
#pragma unroll
    for (int i = 0; i < kN; ++i)
    {
        v[2 * i] = a[2 * i] != nullptr ? emptyVec() : make_uint4(0, 0, 0, 0);
        v[2 * i + 1] = v[2 * i];
    }
    v[2 * kN] = a[2 * kN] != nullptr ? emptyVec() : make_uint4(0, 0, 0, 0);
    uint64_t* const sy = syProbe(ctx);
    uint64_t syC0 = 0;
    uint64_t syC1 = 0;
    uint32_t syRounds = 0;
    uint64_t t0 = 0;
    while (true)
    {
#pragma unroll
        for (int i = 0; i < kN; ++i)
        {
            if (!(full4(v[2 * i]) & full4(v[2 * i + 1])))
            {
                ldPoll256(a[2 * i], v[2 * i], v[2 * i + 1]);
            }
        }
        if (!full4(v[2 * kN]))
        {
            v[2 * kN] = ldPoll128(a[2 * kN]);
        }
        if (sy != nullptr)
        {
            if (syRounds == 0)
            {
                syC0 = clock64();
            }
            else if (syRounds == 1)
            {
                syC1 = clock64();
            }
        }
        ++syRounds;
        bool ok = true;
#pragma unroll
        for (int i = 0; i < 2 * kN + 1; ++i)
        {
            ok &= full4(v[i]);
        }
        if (warpAll(ok))
        {
            break;
        }
        spinCheck(ctx, t0, code, 2 * kN + 1, 0);
    }
    syProbeEnd(sy, syClass(true, code), syC0, syC1, syRounds, code);
#else
#pragma unroll
    for (int i = 0; i < kN; ++i)
    {
        if (a[2 * i] != nullptr)
        {
            ldPoll256(a[2 * i], v[2 * i], v[2 * i + 1]);
        }
        else
        {
            v[2 * i] = make_uint4(0, 0, 0, 0);
            v[2 * i + 1] = v[2 * i];
        }
    }
    v[2 * kN] = a[2 * kN] != nullptr ? ldPoll128(a[2 * kN]) : make_uint4(0, 0, 0, 0);
    uint64_t* const sy = syProbe(ctx);                      // poll probe
    uint64_t const syC0 = sy != nullptr ? clock64() : 0ull; // poll probe
    uint64_t syC1 = 0;                                      // poll probe
    uint32_t syRounds = 1;                                  // poll probe
    uint64_t t0 = 0;
    while (true)
    {
        bool ok = true;
#pragma unroll
        for (int i = 0; i < 2 * kN + 1; ++i)
        {
            ok &= full4(v[i]);
        }
        if (warpAll(ok))
        {
            break;
        }
        spinCheck(ctx, t0, code, 2 * kN + 1, 0);
#pragma unroll
        for (int i = 0; i < kN; ++i)
        {
            if (!(full4(v[2 * i]) & full4(v[2 * i + 1])))
            {
                ldPoll256(a[2 * i], v[2 * i], v[2 * i + 1]);
            }
        }
        if (!full4(v[2 * kN]))
        {
            v[2 * kN] = ldPoll128(a[2 * kN]);
        }
        syProbeRound(sy, syRounds, syC1);                            // poll probe
        ++syRounds;                                                  // poll probe
    }
    syProbeEnd(sy, syClass(true, code), syC0, syC1, syRounds, code); // poll probe
#endif
}

// One plane's 16 pool fragment vectors (bhPoolAddrs), polled: 256-bit fragments with MSO_MK_PB bit 2, else pollVecs.
__device__ __forceinline__ void bhPollPlane(Ctx const& ctx, uint4 (&v)[16], uint8_t const* const (&a)[16])
{
#if MSO_MK_PB & 2
    pollVecs256<8>(ctx, v, a, kCodeMhX);
#else
    pollVecs<16>(ctx, v, a, kCodeMhX);
#endif
}

// ---- end MSO_MK_PB bit 2 plane polls ----

// MhPool under MSO_MK_BH (unit = (tile t, K half h), as stageMhPool): the weights, then (bit 2) the residual pass on rA
// (bit 4: + its lo plane), then one poll of xB and of the row's slots (warp w: row w), the branch pass (bit 4: + xB's
// lo plane, polled next), inv1 / inv2 of row w to shared memory, one reduction of both terms, and inv2 (PA + inv1 PB)
// to every rank's pool area.
__device__ __forceinline__ void bhStageMhPool(Ctx const& ctx, int t, int h, uint32_t wOff, uint32_t bar)
{
    int const lane = threadIdx.x & 31;
    int const warp = threadIdx.x >> 5;
    uint8_t const* const ar = mhArena(ctx, ctx.copy);
    waitPieces(ctx, bar, kCodeWeights);
    float pa[4][4] = {};
    float pb[4][4] = {};
#if MSO_MK_BH & 2
    {
        uint8_t const* a[16];
        uint4 v[16];
        bhPoolAddrs(a, 0, ar + kMhArRA, h);
        bhPollPlane(ctx, v, a); // MSO_MK_PB bit 2
        bhPoolMma(ctx, wOff, v, 0, pa);
#if MSO_MK_BH & 4
        bhPoolAddrs(a, 0, ar + kMhArRALo, h);
        bhPollPlane(ctx, v, a); // MSO_MK_PB bit 2
        bhPoolMma(ctx, wOff, v, 0, pa);
#endif
    }
#endif
    uint8_t const* a[17];
    bhPoolAddrs(a, 0, ar + kMhArX, h);
    a[16] = bhSlot(ctx, warp, lane);
    uint4 v[17];
#if MSO_MK_PB & 2 // 256-bit fragments + the slot vector in the same rounds
    pollVecs256Tail<8>(ctx, v, a, kCodeMhX);
#else
    pollVecs<17>(ctx, v, a, kCodeMhX);
#endif
    stampInput(ctx);
    bhPoolMma(ctx, wOff, v, 0, pb);
#if MSO_MK_BH & 4 // the lo plane in a second poll (published with the hi plane, so normally complete at once)
    {
        uint8_t const* al[16];
        uint4 vl[16];
        bhPoolAddrs(al, 0, ar + kMhArXLo, h);
        bhPollPlane(ctx, vl, al); // MSO_MK_PB bit 2
        bhPoolMma(ctx, wOff, vl, 0, pb);
    }
#endif
    float2 const iv = bhInvs(v[16]);
    float* const sc = reinterpret_cast<float*>(ctx.smem + kSmemRed + 9216);
    if (lane == 0)
    {
        sc[warp] = iv.x;
        sc[8 + warp] = iv.y;
    }
    float acc[2][4];
#pragma unroll
    for (int r = 0; r < 4; ++r)
    {
        acc[0][r] = (pa[0][r] + pa[1][r]) + (pa[2][r] + pa[3][r]);
        acc[1][r] = (pb[0][r] + pb[1][r]) + (pb[2][r] + pb[3][r]);
    }
    float* const out = reinterpret_cast<float*>(ctx.smem + kSmemRed + 8192);
    reduceWarpsTokRow<2>(ctx, acc, out);
    int const i = threadIdx.x;
    if (i < 32)
    {
        int const row = i / 4;
        int const r4 = 4 * (i % 4);
        float4 const sa = *reinterpret_cast<float4 const*>(out + row * 16 + r4);
        float4 const sb = *reinterpret_cast<float4 const*>(out + 128 + row * 16 + r4);
        float const inv1 = sc[row];
        float const inv2 = sc[8 + row];
        mhPublish(ctx,
            kMhPoolOff
                + ((static_cast<size_t>(h) * 8 + row) * kMhPoolWidth + kMhPoolLocal * mhRank(ctx) + 16 * t + r4) * 4,
            lamport4(f4u(__fmul_rn(inv2, __fmaf_rn(inv1, sb.x, sa.x)), __fmul_rn(inv2, __fmaf_rn(inv1, sb.y, sa.y)),
                __fmul_rn(inv2, __fmaf_rn(inv1, sb.z, sa.z)), __fmul_rn(inv2, __fmaf_rn(inv1, sb.w, sa.w)))));
    }
}

// One row half of an MhRouter unit on the vectors v[0, 16) (rows 4 half .. + 4, 8 columns per thread and j): the
// per-thread products, pair sums and warp butterfly of stageMhRouter; the 16 warp-half sums of row 4 half + rr into
// part[(tid >> 4) * 8 + 4 half + rr].
template <int kN>
__device__ __forceinline__ void bhRouterHalf(Ctx const& ctx, uint4 const (&v)[kN], uint32_t wOff, int half, float* part)
{
    int const tid = threadIdx.x;
    int const lane = tid & 31;
#pragma unroll
    for (int rr = 0; rr < 4; ++rr)
    {
        float q[2][2][4]; // [stock thread h][parity c][j]
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            float wv[8];
            lds8f(wv, ctx.smemBase + wOff + 4 * (8 * tid + 2048 * j));
            float xv[8];
            unpack8(xv, v[4 * rr + j]);
#pragma unroll
            for (int h = 0; h < 2; ++h)
            {
#pragma unroll
                for (int c = 0; c < 2; ++c)
                {
                    q[h][c][j] = __fadd_rn(
                        __fmul_rn(wv[4 * h + c], xv[4 * h + c]), __fmul_rn(wv[4 * h + c + 2], xv[4 * h + c + 2]));
                }
            }
        }
        float s[2];
#pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            float a[2];
#pragma unroll
            for (int c = 0; c < 2; ++c)
            {
                a[c] = __fadd_rn(__fadd_rn(q[h][c][0], q[h][c][1]), __fadd_rn(q[h][c][2], q[h][c][3]));
            }
            s[h] = __fadd_rn(a[0], a[1]);
        }
#pragma unroll
        for (int m = 8; m >= 1; m >>= 1)
        {
#pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                s[h] = __fadd_rn(s[h], __shfl_xor_sync(0xffffffffu, s[h], m));
            }
        }
        float const w = __fadd_rn(s[0], s[1]);
        if ((lane & 15) == 0)
        {
            part[(tid >> 4) * 8 + 4 * half + rr] = w;
        }
    }
}

// MhRouter under MSO_MK_BH (unit codes and row halves as stageMhRouter): per half, (bit 2) the residual pass on rA
// into red[256, 384), then one poll of xB and of the half's 4 rows' slots (warps 0-3), the branch pass into red[0, 128)
// and inv1 / inv2 to red[512 + row] / red[520 + row]; then per row the two 16-sum butterflies of stageMhRouter and
// inv2 (RA + inv1 RB) x scale to every rank's logit area. The kIFlagT3CanaryX canary (a timing heuristic) is not taken.
__device__ __forceinline__ void bhStageMhRouter(Ctx const& ctx, int unit, uint32_t wOff, uint32_t bar)
{
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    uint8_t const* const ar = mhArena(ctx, ctx.copy);
    float* red = reinterpret_cast<float*>(ctx.smem + kSmemRed);
    int const split = unit >> 8;
    int const h0 = split == 0 ? 0 : (unit >> 5) & 1;
    int const h1 = split == 0 ? 2 : h0 + 1;
    waitPieces(ctx, bar, kCodeWeights);
#pragma unroll
    for (int half = 0; half < 2; ++half)
    {
        if (half < h0 || half >= h1)
        {
            continue;
        }
#if MSO_MK_BH & 2
        {
            uint8_t const* a[16];
#pragma unroll
            for (int k = 0; k < 16; ++k)
            {
                int const row = 4 * half + k / 4;
                a[k] = ar + kMhArRA + (static_cast<size_t>(row) * 8192 + 8 * tid + 2048 * (k % 4)) * 2;
            }
            uint4 v[16];
            pollVecs<16>(ctx, v, a, kCodeMhX);
            bhRouterHalf(ctx, v, wOff, half, red + 256);
        }
#endif
        uint8_t const* a[17];
#pragma unroll
        for (int k = 0; k < 16; ++k)
        {
            int const row = 4 * half + k / 4;
            a[k] = ar + kMhArX + (static_cast<size_t>(row) * 8192 + 8 * tid + 2048 * (k % 4)) * 2;
        }
        a[16] = warp < 4 ? bhSlot(ctx, 4 * half + warp, lane) : nullptr;
        uint4 v[17];
        pollVecs<17>(ctx, v, a, kCodeMhX);
        if (half == h1 - 1)
        {
            stampInput(ctx);
        }
        if (warp < 4)
        {
            float2 const iv = bhInvs(v[16]);
            if (lane == 0)
            {
                red[512 + 4 * half + warp] = iv.x;
                red[520 + 4 * half + warp] = iv.y;
            }
        }
        bhRouterHalf(ctx, v, wOff, half, red);
    }
    __syncthreads();
    if (tid >= 4 * h0 && tid < 4 * h1)
    {
        float b[16], c[16];
#pragma unroll
        for (int i = 0; i < 16; ++i)
        {
            b[i] = red[i * 8 + tid];
            c[i] = (kMhBh & 2) ? red[256 + i * 8 + tid] : 0.f;
        }
#pragma unroll
        for (int n = 8; n >= 1; n >>= 1)
        {
#pragma unroll
            for (int i = 0; i < n; ++i)
            {
                b[i] = __fadd_rn(b[i], b[i ^ n]);
                c[i] = __fadd_rn(c[i], c[i ^ n]);
            }
        }
        float const inv1 = red[512 + tid];
        float const inv2 = red[520 + tid];
        red[128 + tid] = __fmul_rn(
            __fmul_rn(inv2, __fmaf_rn(inv1, b[0], c[0])), __int_as_float(static_cast<int>(ctx.p[kPMhScale])));
    }
    __syncthreads();
    if (tid >= h0 && tid < h1)
    {
        float4 const s = *reinterpret_cast<float4 const*>(red + 128 + 4 * tid);
        int const e = kMhLocalExperts * mhRank(ctx) + (unit & (kMhLocalExperts - 1)); // mhRouterExpert
        mhPublish(ctx, kMhLogitOff + (static_cast<size_t>(e) * 8 + 4 * tid) * 4, lamport4(f4u(s.x, s.y, s.z, s.w)));
    }
}

// ---- end MSO_MK_BH ----
#endif

#if MSO_MK_FZ
// ---- MSO_MK_FZ: zero-exchange F1 (moe_hops_layout.cuh). MhH1 also writes r' of this rank's unpool columns into
// the rank-local r' area of its fz block (FP32, data as its own flag, from mhBoundary's pass 2). An MhUnpool thread's 8
// columns of one row are exactly one MhF1 stock thread's 8 columns, so after publishing b it forms that thread's
// partials of nbRowSum4's four F1 row sums (the leaves of its tree) from the b it published and r', alpha2, beta2,
// gain_pf of those columns, and publishes them into every rank's leaf area once per reading quarter. MhF1 polls its
// row's leaves with its b and sums them in nbRowSum4's tree (fzRowSum4), so inv1 / inv2 are nbRowSum4's bits and no
// row exchange is left.
// Ownership: every r' word and every leaf word has one writer and, on each rank, one reader (the unpool thread of the
// columns; the MhF1 unit (row, quarter)), which resets it after use: the F1 area's rule (the next write is launch n +
// C, after kernel boundaries on both ranks); MhClear, the arena and the hop copies are unchanged. ----

// The fz block of copy c: after the room of kMhFzMaxCopies hop copies.
__device__ __forceinline__ size_t fzOff(int copy)
{
    return static_cast<size_t>(kMhFzMaxCopies) * kMhCopyBytes + static_cast<size_t>(copy) * kMhFzBytes;
}

// This rank's unicast view of this launch's fz block.
__device__ __forceinline__ uint8_t* fzUc(Ctx const& ctx)
{
    return ptr<uint8_t>(ctx, kPMhUc) + fzOff(ctx.copy);
}

// One 16-byte vector at byte `off` of this launch's fz block, on every rank (mhPublish's two paths).
__device__ __forceinline__ void fzPublish(Ctx const& ctx, size_t off, uint4 v)
{
    size_t const o = fzOff(ctx.copy) + off;
    if (MSO_MK_KFLAGS(ctx.flags) & kFlagMhUnicast) // MSO_MK_LD_KF
    {
        int64_t const* peers = ptr<int64_t const>(ctx, kPMhPeers);
#pragma unroll
        for (int r = 0; r < kMhRanks; ++r)
        {
            st128(reinterpret_cast<uint8_t*>(__ldg(peers + r)) + o, v);
        }
    }
    else
    {
        st128(ptr<uint8_t>(ctx, kPMhMc) + o, v);
    }
}

// The prefetch slot of unpool thread i < 16 in the reduction area (free while the unit runs; mhGemvTile uses [0, 4 KiB)
// and [8 KiB, +512 B)): alpha2 [8] FP32 at +0, beta2 [8] FP32 at +32, gain_pf [8] BF16 at +64.
constexpr int kFzSlotOff = kSmemRed + 10240;
constexpr int kFzSlotBytes = 128;
static_assert(kFzSlotOff + 16 * kFzSlotBytes <= kSmemRed + kSmemRedBytes, "fz: prefetch slots in the reduction area");

__device__ __forceinline__ void fzCp16(uint32_t dst, void const* src)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}

// Unpool thread i < 16 (row i / 2, columns [col, col + 8)): the F1 coefficients of its columns (part col / W of the
// coefficient blob, read-only during the launch) into its slot, as asynchronous copies (no register stays live).
__device__ __forceinline__ void fzPrefetchCoef(Ctx const& ctx, int i, int col)
{
    uint8_t const* const cf = ptr<uint8_t const>(ctx, kPMhCoefF1) + kMhCoefF1GainBytes
        + static_cast<size_t>(col / kMhRowW) * kMhCoefF1Bytes;
    int const lc = col % kMhRowW;
    uint32_t const s = ctx.smemBase + kFzSlotOff + kFzSlotBytes * i;
    fzCp16(s, cf + 4 * lc);
    fzCp16(s + 16, cf + 4 * lc + 16);
    fzCp16(s + 32, cf + 4 * kMhRowW + 4 * lc);
    fzCp16(s + 48, cf + 4 * kMhRowW + 4 * lc + 16);
    fzCp16(s + 64, cf + 8 * kMhRowW + 2 * lc);
    asm volatile("cp.async.commit_group;" ::: "memory");
}

// MhUnpool thread i < 16, after b, with its r' complete in r (loaded before the m poll; the caller polls any word still
// empty): resets its r' words, then forms the leaf of its columns from its published b vector bv (the bits MhF1 reads)
// exactly as mhBoundary's kNbOneX pass 1 forms the stock thread's partials (sigma = bf16(b gain), mhSq from +0, a / b
// / c fma chains from +0, columns in order), and publishes it into every rank's leaf area, once per reading quarter.
__device__ __forceinline__ void fzPublishLeaf(
    Ctx const& ctx, uint4 (&r)[2], uint8_t const* r1, int row, int col, uint4 bv)
{
    int const i = threadIdx.x;
    if (i < 16)
    {
        st128(const_cast<uint8_t*>(r1), emptyVec()); // its one reader resets it
        st128(const_cast<uint8_t*>(r1) + 16, emptyVec());
        asm volatile("cp.async.wait_all;" ::: "memory");
        uint4 const* const s = reinterpret_cast<uint4 const*>(ctx.smem + kFzSlotOff + kFzSlotBytes * i);
        float b[8], g[8], rr[8], al[8], be[8];
        unpack8(b, bv);
        unpack8(g, s[4]);
        uint4 const al0 = s[0], al1 = s[1], be0 = s[2], be1 = s[3];
        uint32_t const ru[8] = {r[0].x, r[0].y, r[0].z, r[0].w, r[1].x, r[1].y, r[1].z, r[1].w};
        uint32_t const au[8] = {al0.x, al0.y, al0.z, al0.w, al1.x, al1.y, al1.z, al1.w};
        uint32_t const bu[8] = {be0.x, be0.y, be0.z, be0.w, be1.x, be1.y, be1.z, be1.w};
#pragma unroll
        for (int v = 0; v < 8; ++v)
        {
            rr[v] = __uint_as_float(ru[v]);
            al[v] = __uint_as_float(au[v]);
            be[v] = __uint_as_float(bu[v]);
        }
        float p = 0.f, sa = 0.f, sb = 0.f, sc = 0.f;
#pragma unroll
        for (int v = 0; v < 8; ++v)
        {
            float const x = bf16Round(__fmul_rn(b[v], g[v]));
            p = mhSq(p, x);
            float const ar = __fmul_rn(al[v], rr[v]);
            float const bx = __fmul_rn(be[v], x);
            sa = __fmaf_rn(ar, ar, sa);
            sb = __fmaf_rn(ar, bx, sb);
            sc = __fmaf_rn(bx, bx, sc);
        }
        uint4 const leaf = lamport4(f4u(p, sa, sb, sc));
#pragma unroll
        for (int qr = 0; qr < 4; ++qr) // one copy per reading quarter
        {
            fzPublish(ctx, kMhFzLeafOff + ((static_cast<size_t>(qr) * 8 + row) * 1024 + col / 8) * 16, leaf);
        }
    }
}

// MhF1's four row sums from its row's leaves, in nbRowSum4's tree: warp w lane l holds in v[q'] the leaf of stock
// warp (q', w) lane l (columns 2048 q' + 256 w + 8 l). The transpose butterfly reduces the 16 values (q', statistic) of
// the warp: at level m = 16, 8, 4, 2 each lane keeps the half of its values selected by its lane bit m, sends the other
// half to lane l ^ m and adds what it receives (the butterfly's pairs (l, l ^ m) of each value, so the butterfly's
// bits); level 1 is a plain step. Lane l then holds the warp sum of value l >> 1 = 4 q' + statistic, i.e. component
// statistic of slot 8 q' + w; the even lanes write it to [kSmemRed, +512) (nbRowSum4's staging), then its second
// butterfly. Returns (S1, A, B, C) as nbRowSum4. All 256 threads call it.
__device__ __forceinline__ float4 fzRowSum4(Ctx const& ctx, uint4 const (&v)[4])
{
    int const lane = threadIdx.x & 31;
    int const warp = threadIdx.x >> 5;
    float x[16];
#pragma unroll
    for (int q = 0; q < 4; ++q)
    {
        x[4 * q] = __uint_as_float(v[q].x);
        x[4 * q + 1] = __uint_as_float(v[q].y);
        x[4 * q + 2] = __uint_as_float(v[q].z);
        x[4 * q + 3] = __uint_as_float(v[q].w);
    }
#pragma unroll
    for (int n = 8; n >= 1; n >>= 1) // level m = 2 n (16, 8, 4, 2): x[0, 2 n) -> x[0, n)
    {
        bool const hi = (lane & (2 * n)) != 0;
#pragma unroll
        for (int k = 0; k < n; ++k)
        {
            float const keep = hi ? x[k + n] : x[k];
            float const send = hi ? x[k] : x[k + n];
            x[k] = __fadd_rn(keep, __shfl_xor_sync(0xffffffffu, send, 2 * n));
        }
    }
    x[0] = __fadd_rn(x[0], __shfl_xor_sync(0xffffffffu, x[0], 1));
    float* const stage = reinterpret_cast<float*>(ctx.smem + kSmemRed);
    if ((lane & 1) == 0)
    {
        int const k = lane >> 1;
        stage[(8 * (k >> 2) + warp) * 4 + (k & 3)] = x[0];
    }
    __syncthreads();
    float4 const s = reinterpret_cast<float4 const*>(stage)[lane];
    float t = __fadd_rn(0.f, s.x);
    float y = s.y, z = s.z, w = s.w;
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1)
    {
        t = __fadd_rn(t, __shfl_xor_sync(0xffffffffu, t, m));
        y = __fadd_rn(y, __shfl_xor_sync(0xffffffffu, y, m));
        z = __fadd_rn(z, __shfl_xor_sync(0xffffffffu, z, m));
        w = __fadd_rn(w, __shfl_xor_sync(0xffffffffu, w, m));
    }
    return make_float4(t, y, z, w);
}

// ---- end MSO_MK_FZ ----
#endif

// The weights of a unit (the weight table and the device-side requests use this). The boundary coefficients are
// per column part: every row's unit q (= unit % U) reads part q of the packed blob (moe_hops.pack_coef).
__device__ __forceinline__ bool mhUnitWeights(
    Ctx const& ctx, int stage, int unit, uint8_t const*& src0, uint32_t& n0, uint8_t const*& src1, uint32_t& n1)
{
    switch (stage)
    {
    case kStMhRouter: // the expert row is the unit's low bits (mhRouterExpert); a rank-local unit (512 + ...)
                      // reads 4 rows of the full [256][8192] table (stageMhRouterLocal)
#if MSO_MK_RL & 1
        if (kRlOk && (ctx.p[kPIFlags] & kIFlagRlOn)) // MSO_MK_RL: the rl unit's weight slab
        {
            src0 = rlBlockPtr<uint8_t const>(ctx, kRlBlockW) + static_cast<size_t>(rlUnitOf(unit)) * kRlUnitBytes;
            n0 = kRlUnitBytes;
            return true;
        }
#endif
        if (unit & kMhRouterLocalBit)
        {
            src0 = ptr<uint8_t const>(ctx, kPMhWRouter)
                + static_cast<size_t>(unit & 63) * kMhRouterLocalExperts * kMhRouterUnitBytes;
            n0 = kMhRouterLocalExperts * kMhRouterUnitBytes;
            return true;
        }
        src0 = ptr<uint8_t const>(ctx, kPMhWRouter)
            + static_cast<size_t>(unit & (kMhLocalExperts - 1)) * kMhRouterUnitBytes;
        n0 = kMhRouterUnitBytes;
        return true;
    case kStMhPool:
        src0 = ptr<uint8_t const>(ctx, kPMhWPool) + static_cast<size_t>(unit) * kMhPoolUnitBytes;
        n0 = kMhPoolUnitBytes;
        return true;
    case kStMhUnpool:
        src0 = ptr<uint8_t const>(ctx, kPMhWUnpool) + static_cast<size_t>(unit) * kMhUnpoolUnitBytes;
        n0 = kMhUnpoolUnitBytes;
        return true;
    case kStMhH1:
        src0 = ptr<uint8_t const>(ctx, kPMhCoefH1) + static_cast<size_t>(unit % kMhRowU) * kMhCoefH1Bytes;
        n0 = kMhCoefH1Bytes;
        return true;
    case kStMhF1:
        src0 = ptr<uint8_t const>(ctx, kPMhCoefF1) + kMhCoefF1GainBytes
            + static_cast<size_t>(unit % kMhRowU) * kMhCoefF1Bytes;
        n0 = kMhCoefF1Bytes;
        if constexpr (kMhF1Full)
        {
            src1 = ptr<uint8_t const>(ctx, kPMhCoefF1);
            n1 = kMhCoefF1GainBytes;
        }
        return true;
    default: return false;
    }
}

// MhOproj (stub of the o_proj epilogue): unit u = (row u / 4, columns [2048 (u % 4), +2048)); thread t: 8 columns.
__device__ __noinline__ void stageMhOproj(Ctx const& ctx, int unit)
{
    int const row = unit / 4;
    int const col = 2048 * (unit % 4) + 8 * threadIdx.x;
    uint4 const v = ldcg128(ptr<__nv_bfloat16 const>(ctx, kPMhInOproj) + row * 8192 + col);
    mhPublish(ctx, kMhH1Off + ((static_cast<size_t>(mhRank(ctx)) * 8 + row) * 8192 + col) * 2, lamport4(v));
}

// The boundary after its branch values x (BF16 values, the gain applied). Pass 1: the post norm's variance; pass 2:
// the normalized branch and the gated residual u = fma(alpha, r, beta * value) (alpha, beta at shared addresses aS, bS;
// r from rG (MhH1, registers) or the shared address rS (MhF1)); MhH1 (rIsGlobal) leaves u in its alpha region for MhF1;
// u also to uOut / uCheck (global, may be null) and bf16(u) to capture (may be null); pass 3: the next Gemma norm with
// the BF16 weight at wNextS, returned packed in o.
// kNbOneX (MSO_MK_NB builds): pass 1 also sums (alpha r)^2, (alpha r)(beta x), (beta x)^2 and carries them in the
// first norm's exchange (nbRowSum4); the second norm's variance is then sum u^2 = A + 2 inv1 B + inv1^2 C, u unrounded,
// so pass 2 has no exchange.
#if MSO_MK_NB
#if MSO_MK_FZ // kFzGiven = MhF1's inv1 / inv2 from its leaves (no pass-1 sums, no exchange)
template <bool kNbOneX, bool kFzGiven = false>
#else
template <bool kNbOneX>
#endif
#endif
__device__ __forceinline__ void mhBoundary(Ctx const& ctx, float (&x)[kMhRowV][8], uint32_t aS, uint32_t bS,
    uint32_t rS, bool rIsGlobal, float const (&rG)[kMhRowV][8], uint32_t wNextS, int row, int q, int phase, float* uOut,
    __nv_bfloat16* capture, float* uCheck, uint4 (&o)[kMhRowV], int subBase, float inv1Given = -1.f,
    float wBias = 1.f,      // wBias 0 = a plain (non-Gemma) next norm, the final norm after layer 61
    float* invOut = nullptr // MSO_MK_RL: the two norms' 1 / rms (s1, s2) of the row, or null
#if MSO_MK_FZ               // MhF1's four row sums from its leaves (kFzGiven); MhH1's r' area (pass 2 writes u into it)
    ,
    float4 fzSums = float4{}, uint8_t* r1Out = nullptr
#endif
)
{
    // inv1Given >= 0: the first norm's 1 / rms, already summed by the caller (MhF1 with kMhF1Full).
    float part[kMhRowV];
    float inv = inv1Given;
#if MSO_MK_NB
    float nbInv2 = 0.f; // kNbOneX: the second norm's 1 / rms from the one exchange
#endif
#if MSO_MK_FZ
    if constexpr (kFzGiven) // the one exchange's statements on MhF1's sums from its leaves
    {
        inv = mhRsqrt(fzSums.x, 1e-8f);
        float const s2 = __fmaf_rn(__fmul_rn(inv, inv), fzSums.w, __fmaf_rn(__fmul_rn(2.f, inv), fzSums.z, fzSums.y));
        nbInv2 = mhRsqrt(fmaxf(s2, 0.f), 1e-5f);
    }
    else
#endif
        if (inv1Given < 0.f)
    {
#pragma unroll
        for (int j = 0; j < kMhRowV; ++j)
        {
            float t = 0.f;
#pragma unroll
            for (int v = 0; v < 8; ++v)
            {
                t = mhSq(t, x[j][v]);
            }
            part[j] = t;
        }
#if MSO_MK_NB
        if constexpr (kNbOneX)
        {
            float sa = 0.f, sb = 0.f, sc = 0.f;
#pragma unroll
            for (int j = 0; j < kMhRowV; ++j)
            {
                int const lc = mhLocalCol(j);
                float al[8], be[8], r[8];
                lds8f(al, aS + 4 * lc);
                lds8f(be, bS + 4 * lc);
                if (rIsGlobal)
                {
#pragma unroll
                    for (int v = 0; v < 8; ++v)
                    {
                        r[v] = rG[j][v];
                    }
                }
                else
                {
                    lds8f(r, rS + 4 * lc);
                }
#pragma unroll
                for (int v = 0; v < 8; ++v)
                {
                    float const ar = __fmul_rn(al[v], r[v]);
                    float const bx = __fmul_rn(be[v], x[j][v]);
                    sa = __fmaf_rn(ar, ar, sa);
                    sb = __fmaf_rn(ar, bx, sb);
                    sc = __fmaf_rn(bx, bx, sc);
                }
            }
            float4 const t4 = nbRowSum4(ctx, part, sa, sb, sc, row, q, phase / 2);
            inv = mhRsqrt(t4.x, 1e-8f);
            float const s2 = __fmaf_rn(__fmul_rn(inv, inv), t4.w, __fmaf_rn(__fmul_rn(2.f, inv), t4.z, t4.y));
            nbInv2 = mhRsqrt(fmaxf(s2, 0.f), 1e-5f);
        }
        else
#endif
            inv = mhRsqrt(mhRowSum(ctx, part, row, q, phase), 1e-8f);
    }
#if MSO_MK_RL & 1
    if (invOut != nullptr) // MSO_MK_RL
    {
        invOut[0] = inv;
    }
#endif
    subStamp(ctx, subBase + 1);
#pragma unroll
    for (int j = 0; j < kMhRowV; ++j)
    {
        int const lc = mhLocalCol(j);
        float al[8], be[8], r[8];
        lds8f(al, aS + 4 * lc);
        lds8f(be, bS + 4 * lc);
        if (rIsGlobal)
        {
#pragma unroll
            for (int v = 0; v < 8; ++v)
            {
                r[v] = rG[j][v];
            }
        }
        else
        {
            lds8f(r, rS + 4 * lc);
        }
        float u[8];
        float t = 0.f;
#pragma unroll
        for (int v = 0; v < 8; ++v)
        {
            float const value = bf16Round(__fmul_rn(x[j][v], inv));
            u[v] = mhGated(al[v], r[v], be[v], value);
            x[j][v] = bf16Round(u[v]);
            t = mhSq(t, x[j][v]);
        }
        part[j] = t;
        int const col = q * kMhRowW + lc;
        if (rIsGlobal)
        {
            // r' into the alpha slots this thread read above.
            float4* s = reinterpret_cast<float4*>(ctx.smem + (aS - ctx.smemBase) + 4 * lc);
            s[0] = make_float4(u[0], u[1], u[2], u[3]);
            s[1] = make_float4(u[4], u[5], u[6], u[7]);
#if MSO_MK_FZ
            // MSO_MK_FZ: r' of this rank's unpool columns into its r' area for the unpool units, as data-as-flag words
            uint32_t const c1 = static_cast<uint32_t>(col - kMhUnpoolLocal * mhRank(ctx));
            if (r1Out != nullptr && c1 < static_cast<uint32_t>(kMhUnpoolLocal))
            {
                uint8_t* const p = r1Out + (static_cast<size_t>(row) * kMhUnpoolLocal + c1) * 4;
                st128(p, lamport4(f4u(u[0], u[1], u[2], u[3])));
                st128(p + 16, lamport4(f4u(u[4], u[5], u[6], u[7])));
            }
#endif
        }
        if (uOut != nullptr)
        {
            st8f(uOut + row * 8192 + col, u);
        }
        if (uCheck != nullptr)
        {
            st8f(uCheck + row * 8192 + col, u);
        }
        if (capture != nullptr)
        {
            st128(capture + static_cast<size_t>(row) * ctx.p[kPMhCaptureStride] + col, pack8(u));
        }
    }
#if MSO_MK_NB
    if constexpr (kNbOneX)
    {
        inv = nbInv2;
    }
    else
#endif
        inv = mhRsqrt(mhRowSum(ctx, part, row, q, phase + 1), 1e-5f);
#if MSO_MK_RL & 1
    if (invOut != nullptr) // MSO_MK_RL
    {
        invOut[1] = inv;
    }
#endif
    subStamp(ctx, subBase + 2);
#pragma unroll
    for (int j = 0; j < kMhRowV; ++j)
    {
        float w[8];
        unpack8(w, lds128(wNextS + 2 * mhLocalCol(j)));
        float y[8];
#pragma unroll
        for (int v = 0; v < 8; ++v)
        {
            y[v] = __fmul_rn(__fmul_rn(x[j][v], inv), __fadd_rn(w[v], wBias)); // wBias 0: a plain norm
        }
        o[j] = pack8(y);
    }
}

#if MSO_MK_RL & 6
// ---- MSO_MK_RL record / oracle (kIFlagRlRecord / kIFlagRlOracleLg / kIFlagRlOraclePool, msoMegakernel.cu): boundary
// unit (row, q)'s slice of the F2' areas: thread t's 8 pool words of K half t / 128 at columns 1024 q + 8 (t % 128)
// (two 16-byte vectors), and for t < 64 the logit word of expert 64 q + t (byte offsets in the F2' layout, from
// kMhPoolOff; kRlOracleBytes, kRlOk: stages/moe_hops_layout.cuh) ----
__device__ __forceinline__ size_t rlPoolOff(int row, int q)
{
    int const t = threadIdx.x;
    return ((static_cast<size_t>(t >> 7) * 8 + row) * kMhPoolWidth + 1024 * q + 8 * (t & 127)) * 4;
}

__device__ __forceinline__ size_t rlLogitOff(int row, int q)
{
    return static_cast<size_t>(kMhPoolBytes) + ((static_cast<size_t>(64 * q + threadIdx.x)) * 8 + row) * 4;
}

#endif
#if MSO_MK_RL & 2
// Recording (MhF1 unit (row, q), all threads): poll the slice in this launch's F2' areas, copy it to the record.
__device__ __noinline__ void rlRecord(Ctx const& ctx, int row, int q)
{
    uint8_t const* const f2 = mhUc(ctx, ctx.copy) + kMhPoolOff;
    uint8_t* const o = rlBlockPtr<uint8_t>(ctx, kRlBlockRec);
    size_t const po = rlPoolOff(row, q);
    uint8_t const* const a[2] = {f2 + po, f2 + po + 16};
    uint4 v[2];
    pollVecs<2>(ctx, v, a, kCodeMhF2);
    size_t const lo = rlLogitOff(row, q);
    uint32_t const w = poll1(ctx, threadIdx.x < 64 ? f2 + lo : nullptr, kCodeMhF2);
    st128(o + po, v[0]);
    st128(o + po + 16, v[1]);
    if (threadIdx.x < 64)
    {
        st32(o + lo, w);
    }
}

// Record, MhH1 unit (row, q) (kIFlagRlRecord; the record layout kRlRec* is in stages/moe_hops_layout.cuh): its residual
// slice and the row's gated o_proj input columns [256 q, +256) (threads 0-31).
__device__ __noinline__ void rlRecordH1(Ctx const& ctx, int row, int q, float const (&res)[kMhRowV][8])
{
    uint8_t* const rec = rlBlockPtr<uint8_t>(ctx, kRlBlockRec);
#pragma unroll
    for (int j = 0; j < kMhRowV; ++j)
    {
        st8f(reinterpret_cast<float*>(rec + kRlRecResOff) + row * 8192 + q * kMhRowW + mhLocalCol(j), res[j]);
    }
    if (threadIdx.x < 32)
    {
        size_t const o = (static_cast<size_t>(row) * kRlOCols + 256 * q + 8 * threadIdx.x) * 2;
        st128(rec + kRlRecOOff + o, ldcg128(ctx.arena + kArAhO + o));
    }
}

#endif
#if MSO_MK_RL & 1
__device__ __forceinline__ uint8_t* rlMc(Ctx const& ctx)
{
    return rlBlockPtr<uint8_t>(ctx, kRlBlockMc) + static_cast<size_t>(ctx.copy) * kRlCopyBytes;
}

__device__ __forceinline__ uint8_t* rlUc(Ctx const& ctx)
{
    return rlBlockPtr<uint8_t>(ctx, kRlBlockUc) + static_cast<size_t>(ctx.copy) * kRlCopyBytes;
}

// rl unit i (MhRouter's slot, kIFlagRlOn): this rank's logit partials of experts [4 i, +4) for the 8 rows,
//   A[row][e] = sum_k W_R diag(gamma_a)[4 i + e][1024 rank + k] r[row][1024 rank + k]   (the layer's FP32 residual)
//   B[row][e] = sum_k D_r[4 i + e][k] o[row][k]                                          (the gated o_proj input)
// Weights (32 KiB at wOff): the A part then the D part, each [4 vectors j][256 threads][4] FP32, thread t = 4 ks + e
// holding columns 16 ks + 4 j .. + 3 of expert 4 i + e. Thread t: FP32 FMAs over its 16 columns in order for every
// row; the warp's 8 column slices summed by the butterfly over lanes 4, 8, 16; the 8 warps in order. The unit runs
// after the grid-dependency wait (a MoE-half unit). Out: [term][rank][row][4 i, +4) of every rank's rl area
// (multicast).
__device__ __noinline__ void stageRlLogit(Ctx const& ctx, int unit, uint32_t wOff, uint32_t bar, Io const& io)
{
    int const tid = threadIdx.x;
    int const warp = tid >> 5;
    int const lane = tid & 31;
    int const i = rlUnitOf(unit);
    int const ks = tid >> 2;
    int const rank = mhRank(ctx);
    float acc[2][8];
#pragma unroll
    for (int t = 0; t < 8; ++t)
    {
        acc[0][t] = 0.f;
        acc[1][t] = 0.f;
    }
    waitPieces(ctx, bar, kCodeWeights);
    {
        float w[16];
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            uint4 const v = lds128(ctx.smemBase + wOff + static_cast<uint32_t>((j * kThreads + tid) * 16));
            w[4 * j] = __uint_as_float(v.x), w[4 * j + 1] = __uint_as_float(v.y);
            w[4 * j + 2] = __uint_as_float(v.z), w[4 * j + 3] = __uint_as_float(v.w);
        }
        float const* res = static_cast<float const*>(io.res) + kRlOCols * rank + 16 * ks;
#pragma unroll
        for (int t = 0; t < 8; ++t)
        {
            float x[16];
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                float4 const v = ldcgF4(res + t * 8192 + 4 * j);
                x[4 * j] = v.x, x[4 * j + 1] = v.y, x[4 * j + 2] = v.z, x[4 * j + 3] = v.w;
            }
#pragma unroll
            for (int k = 0; k < 16; ++k)
            {
                acc[0][t] = fmaf(w[k], x[k], acc[0][t]);
            }
        }
    }
    {
        int const reps = ctx.p[kPAhOReplicas] > 0 ? static_cast<int>(ctx.p[kPAhOReplicas]) : 1;
        uint8_t const* in = ctx.arena + kArAhO + (blockIdx.x % reps) * (kTok * kRlOCols * 2);
        uint8_t const* a[16];
#pragma unroll
        for (int t = 0; t < 8; ++t)
        {
            a[2 * t] = in + (static_cast<size_t>(t) * kRlOCols + 16 * ks) * 2;
            a[2 * t + 1] = a[2 * t] + 16;
        }
        uint4 v[16];
        pollVecs<16>(ctx, v, a, kCodeMhX);
        stampInput(ctx);
        float w[16];
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            uint4 const u
                = lds128(ctx.smemBase + wOff + static_cast<uint32_t>(kRlPartBytes + (j * kThreads + tid) * 16));
            w[4 * j] = __uint_as_float(u.x), w[4 * j + 1] = __uint_as_float(u.y);
            w[4 * j + 2] = __uint_as_float(u.z), w[4 * j + 3] = __uint_as_float(u.w);
        }
#pragma unroll
        for (int t = 0; t < 8; ++t)
        {
            float xl[8];
            float xh[8];
            unpack8(xl, v[2 * t]);
            unpack8(xh, v[2 * t + 1]);
#pragma unroll
            for (int k = 0; k < 8; ++k)
            {
                acc[1][t] = fmaf(w[k], xl[k], acc[1][t]);
            }
#pragma unroll
            for (int k = 0; k < 8; ++k)
            {
                acc[1][t] = fmaf(w[8 + k], xh[k], acc[1][t]);
            }
        }
    }
    float* red = reinterpret_cast<float*>(ctx.smem + kSmemRed + 12288);
    static_assert(12288 + kWarps * 2 * 8 * 4 * 4 <= kSmemRedBytes, "rl staging");
#pragma unroll
    for (int p = 0; p < 2; ++p)
    {
#pragma unroll
        for (int t = 0; t < 8; ++t)
        {
            float v = acc[p][t];
            v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 4));
            v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 8));
            v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 16));
            if (lane < 4)
            {
                red[((warp * 2 + p) * 8 + t) * 4 + lane] = v;
            }
        }
    }
    __syncthreads();
    if (tid < 64)
    {
        int const p = tid >> 5;
        int const t = (tid >> 2) & 7;
        int const e = tid & 3;
        float s = 0.f;
#pragma unroll
        for (int ww = 0; ww < kWarps; ++ww)
        {
            s = __fadd_rn(s, red[((ww * 2 + p) * 8 + t) * 4 + e]);
        }
        float const s1 = __shfl_down_sync(0xffffffffu, s, 1);
        float const s2 = __shfl_down_sync(0xffffffffu, s, 2);
        float const s3 = __shfl_down_sync(0xffffffffu, s, 3);
        if (e == 0)
        {
            size_t const off = static_cast<size_t>(p) * kRlTermBytes
                + ((static_cast<size_t>(rank) * 8 + t) * kMhExperts + kRlExpertsPerUnit * i) * 4;
            st128(rlMc(ctx) + off, lamport4(f4u(s, s1, s2, s3)));
        }
    }
}

// rlFinalize, MhH1 unit (row, q), all threads: thread t's vector of the partials, term t / 128, source rank (t / 16) %
// 8, experts 64 q + 4 (t % 16) .. + 3 (MhH1 issues the load after its hop-1 poll; rlFinalize completes it).
__device__ __forceinline__ uint8_t* rlPartAddr(Ctx const& ctx, int row, int q)
{
    int const t = threadIdx.x;
    return rlUc(ctx) + static_cast<size_t>(t >> 7) * kRlTermBytes
        + ((static_cast<size_t>((t >> 4) & 7) * 8 + row) * kMhExperts + 64 * q + 4 * (t & 15)) * 4;
}

// After the boundary: the partials (complete the poll), then for expert 64 q + t (t < 64) the 8 sources' A and B
// partials summed in rank order from +0, l = (mu s2) (A + s1 B), into this rank's F2' logit area (plain stores; the
// routing polls them there); then every word read is emptied (single reader per rank).
__device__ __noinline__ void rlFinalize(Ctx const& ctx, int row, int q, float s1, float s2, uint4 v)
{
    uint8_t* const pa = rlPartAddr(ctx, row, q);
    {
        uint8_t const* const a[1] = {pa};
        uint4 w[1] = {v};
        uint64_t t0 = 0;
        while (!warpAll(full4(w[0])))
        {
            spinCheck(ctx, t0, kCodeMhF2, 1, 0);
            if (!full4(w[0]))
            {
                w[0] = ldPoll128(a[0]);
            }
        }
        v = w[0];
    }
    int const t = threadIdx.x;
    float* red = reinterpret_cast<float*>(ctx.smem + kSmemRed + 12288);
    // [term][source][64 experts]
    *reinterpret_cast<uint4*>(red + ((t >> 7) * 8 + ((t >> 4) & 7)) * 64 + 4 * (t & 15)) = v;
    __syncthreads();
    if (t < 64)
    {
        float sa = 0.f;
        float sb = 0.f;
#pragma unroll
        for (int r = 0; r < kMhRanks; ++r)
        {
            sa = __fadd_rn(sa, red[r * 64 + t]);
            sb = __fadd_rn(sb, red[(8 + r) * 64 + t]);
        }
        float const mu = __int_as_float(static_cast<int>(ctx.p[kPMhScale]));
        float const l = __fmul_rn(__fmul_rn(mu, s2), __fadd_rn(sa, __fmul_rn(s1, sb)));
        st32(mhUc(ctx, ctx.copy) + kMhLogitOff + ((static_cast<size_t>(64 * q + t)) * 8 + row) * 4,
            lamport(__float_as_uint(l)));
    }
    st128(pa, emptyVec());
}

#if MSO_MK_RL & 8
#if !(MSO_MK_NB & 1)
#error "rl bit 8 takes MhH1's norm scalars from nb's one exchange: it needs MSO_MK_NB bit 1"
#endif
#if !(MSO_MK_RP & 1)
#error "rl bit 8 forms the logits in the experts stage's publisher CTA: it needs MSO_MK_RP bit 1"
#endif
// MSO_MK_RL bit 8 (msoMegakernel.cu): the rank's logits formed once, by the experts stage's publisher CTA (MSO_MK_RP)
// at its entry, instead of by the 32 MhH1 units (rlFinalize, which makes MhH1 wait for the slowest rank's partials).
// Warp w: the norm scalars s1, s2 of rows 2 (w / 2), + 1 from MhH1's one exchange (nbRowSum4's slots of boundary 0; the
// same sums in the same order as nbRowSum4 and mhBoundary<true>, so the same s1, s2 bitwise). Thread t: experts
// 4 (t % 64) .. + 3 of rows 2 (t / 64), + 1; the 8 source ranks' A and B partials summed in rank order from +0 and
// l = (mu s2) (A + s1 B) (rlFinalize's arithmetic) into this rank's F2' logit area, where route() polls them. Every
// partial word read is emptied after its round (the publisher is the single reader of its rank's rl area, as the MhH1
// units are without this bit). All 256 threads; no shared memory.
__device__ __noinline__ void rlPublisherLogits(Ctx const& ctx)
{
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const e0 = 4 * (tid & 63);
    int const r0 = 2 * (tid >> 6);
    float s1[2], s2[2];
#pragma unroll
    for (int rr = 0; rr < 2; ++rr)
    {
        float4 const* const slots
            = reinterpret_cast<float4 const*>(mhArena(ctx, ctx.copy) + kMhArAbc) + (0 * 8 + r0 + rr) * 32;
        uint8_t const* const p[1] = {reinterpret_cast<uint8_t const*>(slots + lane)};
        uint4 r[1];
        pollVecs<1>(ctx, r, p, kCodeMhSsq);
        float t = __fadd_rn(0.f, __uint_as_float(r[0].x));
        float y = __uint_as_float(r[0].y), z = __uint_as_float(r[0].z), w = __uint_as_float(r[0].w);
#pragma unroll
        for (int m = 16; m >= 1; m >>= 1)
        {
            t = __fadd_rn(t, __shfl_xor_sync(0xffffffffu, t, m));
            y = __fadd_rn(y, __shfl_xor_sync(0xffffffffu, y, m));
            z = __fadd_rn(z, __shfl_xor_sync(0xffffffffu, z, m));
            w = __fadd_rn(w, __shfl_xor_sync(0xffffffffu, w, m));
        }
        float const inv = mhRsqrt(t, 1e-8f);
        float const q2 = __fmaf_rn(__fmul_rn(inv, inv), w, __fmaf_rn(__fmul_rn(2.f, inv), z, y));
        s1[rr] = inv;
        s2[rr] = mhRsqrt(fmaxf(q2, 0.f), 1e-5f);
    }
    uint8_t* const uc = rlUc(ctx);
    float acc[2][2][4]; // [term][row][expert]
#pragma unroll
    for (int p = 0; p < 2; ++p)
    {
#pragma unroll
        for (int rr = 0; rr < 2; ++rr)
        {
            uint8_t const* a[kMhRanks];
#pragma unroll
            for (int src = 0; src < kMhRanks; ++src)
            {
                a[src] = uc + static_cast<size_t>(p) * kRlTermBytes
                    + ((static_cast<size_t>(src) * 8 + r0 + rr) * kMhExperts + e0) * 4;
            }
            uint4 v[kMhRanks];
            pollVecs<kMhRanks>(ctx, v, a, kCodeMhF2);
#pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                acc[p][rr][k] = 0.f;
            }
#pragma unroll
            for (int src = 0; src < kMhRanks; ++src)
            {
                acc[p][rr][0] = __fadd_rn(acc[p][rr][0], __uint_as_float(v[src].x));
                acc[p][rr][1] = __fadd_rn(acc[p][rr][1], __uint_as_float(v[src].y));
                acc[p][rr][2] = __fadd_rn(acc[p][rr][2], __uint_as_float(v[src].z));
                acc[p][rr][3] = __fadd_rn(acc[p][rr][3], __uint_as_float(v[src].w));
                st128(const_cast<uint8_t*>(a[src]), emptyVec());
            }
        }
    }
    float const mu = __int_as_float(static_cast<int>(ctx.p[kPMhScale]));
    uint8_t* const lg = mhUc(ctx, ctx.copy) + kMhLogitOff;
#pragma unroll
    for (int rr = 0; rr < 2; ++rr)
    {
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            float const l
                = __fmul_rn(__fmul_rn(mu, s2[rr]), __fadd_rn(acc[0][rr][k], __fmul_rn(s1[rr], acc[1][rr][k])));
            st32(lg + ((static_cast<size_t>(e0 + k)) * 8 + r0 + rr) * 4, lamport(__float_as_uint(l)));
        }
    }
    __syncthreads();
}
#endif
// ---- end MSO_MK_RL record / oracle ----
#endif

// MhH1: unit u = (row u / U, part q = u % U). Coefficients (unit weights at wOff): alpha1 [W] FP32, beta1 [W] FP32,
// w_preffn [W] BF16. The hop1 slots of the thread's vectors are polled in rounds of at most two vectors (16 loads),
// summed in rank order from +0 and rounded once; the residual comes from global memory, loaded before the poll.
__device__ __noinline__ void stageMhH1(Ctx const& ctx, int unit, uint32_t wOff, uint32_t bar, Io const& io)
{
    int const row = unit / kMhRowU;
    int const q = unit % kMhRowU;
    uint8_t* const h1 = mhUc(ctx, ctx.copy) + kMhH1Off;
    float const* resRow = static_cast<float const*>(io.res) + row * 8192 + q * kMhRowW;
    float res[kMhRowV][8];
#pragma unroll
    for (int j = 0; j < kMhRowV; ++j)
    {
        ld8f(res[j], resRow + mhLocalCol(j));
    }
    float x[kMhRowV][8];
#if MSO_MK_RL & 4
    // ---- the recorded F2' slice, loaded before the hop-1 poll (kIFlagRlOracleLg / kIFlagRlOraclePool) ----
    bool const rlLg = kRlOk && (ctx.p[kPIFlags] & kIFlagRlOracleLg) != 0;
    bool const rlPool = kRlOk && (ctx.p[kPIFlags] & kIFlagRlOraclePool) != 0;
    uint4 rlV[2] = {};
    uint32_t rlW = 0u;
    if (rlLg || rlPool)
    {
        uint8_t const* const o = rlBlockPtr<uint8_t const>(ctx, kRlBlockRec);
        if (rlPool)
        {
            size_t const po = rlPoolOff(row, q);
            rlV[0] = ldcg128(o + po);
            rlV[1] = ldcg128(o + po + 16);
        }
        if (rlLg && threadIdx.x < 64)
        {
            rlW = __ldcg(reinterpret_cast<unsigned int const*>(o + rlLogitOff(row, q)));
        }
    }
    // ---- end recorded slice ----
#endif
#if (MSO_MK_RL & 1) && !(MSO_MK_RL & 8)
    bool const rlOn = kRlOk && (ctx.p[kPIFlags] & kIFlagRlOn) != 0; // this unit forms its slice of the logits
#endif
#if MSO_MK_BH & 2 // the residual term needs the unit weights before the poll
    bool const t3Wf = true;
#else
    bool const t3Wf = (icPIFlags(ctx.p[kPIFlags]) & kIFlagT3WaitFirst) != 0; // weights first
#endif
    if (t3Wf)
    {
        waitPieces(ctx, bar, kCodeWeights);
    }
#if MSO_MK_BH & 2
    // ---- the residual term rA = (alpha1 r)(w + 1) of the thread's columns, published before the hop-1 poll ----
    {
        uint32_t const cb = ctx.smemBase + wOff;
#pragma unroll
        for (int j = 0; j < kMhRowV; ++j)
        {
            float t[8];
            bhTerm(t, res[j], cb, cb + 8 * kMhRowW, mhLocalCol(j));
            bhPublishTerm(ctx, t, row, q * kMhRowW + mhLocalCol(j), kMhArRA, (kMhBh & 4) ? kMhArRALo : -1);
        }
    }
    // ---- end residual term ----
#endif
    constexpr int kJr = kMhRowV < 2 ? kMhRowV : 2;
#pragma unroll
    for (int j0 = 0; j0 < kMhRowV; j0 += kJr)
    {
        uint8_t const* a[kJr * kMhRanks];
#pragma unroll
        for (int jj = 0; jj < kJr; ++jj)
        {
#pragma unroll
            for (int r = 0; r < kMhRanks; ++r)
            {
                a[jj * kMhRanks + r]
                    = h1 + ((static_cast<size_t>(r) * 8 + row) * 8192 + q * kMhRowW + mhLocalCol(j0 + jj)) * 2;
            }
        }
        uint4 v[kJr * kMhRanks];
        pollVecs<kJr * kMhRanks>(ctx, v, a, kCodeMhHop1);
#pragma unroll
        for (int jj = 0; jj < kJr; ++jj)
        {
            float s[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int r = 0; r < kMhRanks; ++r)
            {
                float p[8];
                unpack8(p, v[jj * kMhRanks + r]);
#pragma unroll
                for (int k = 0; k < 8; ++k)
                {
                    s[k] = __fadd_rn(s[k], p[k]);
                }
            }
#pragma unroll
            for (int k = 0; k < 8; ++k)
            {
                x[j0 + jj][k] = bf16Round(s[k]);
            }
        }
    }
#if (MSO_MK_RL & 1) && !(MSO_MK_RL & 8)
    // ---- the logit partials, requested now (they land under the boundary arithmetic; rlFinalize completes) ----
    uint4 rlPv = make_uint4(0u, 0u, 0u, 0u);
    if (rlOn)
    {
        rlPv = ldPoll128(rlPartAddr(ctx, row, q));
    }
    // ---- end logit partials ----
#endif
#if MSO_MK_RL & 2
    if (kRlOk && (ctx.p[kPIFlags] & kIFlagRlRecord) != 0) // the record's residual and o_proj input slices
    {
        rlRecordH1(ctx, row, q, res);
    }
#endif
    stampInput(ctx);
    subStamp(ctx, 0);
    // ---- stamped builds only: sub 16 = every warp of the CTA has its hop-1 data ----
    if (MSO_MK_STAMPS_DETAIL(ctx) != nullptr)
    {
        __syncthreads();
        subStamp(ctx, 16);
    }
    // ---- end stamped builds only ----
    if (!t3Wf)
    {
        waitPieces(ctx, bar, kCodeWeights);
    }
    uint32_t const cb = ctx.smemBase + wOff;
#if MSO_MK_BH
    // ---- the branch term xB = (beta1 x)(w + 1) into x_ffn's words, before the boundary's exchange ----
#pragma unroll
    for (int j = 0; j < kMhRowV; ++j)
    {
        float t[8];
        bhTerm(t, x[j], cb + 4 * kMhRowW, cb + 8 * kMhRowW, mhLocalCol(j));
        bhPublishTerm(ctx, t, row, q * kMhRowW + mhLocalCol(j), kMhArX, (kMhBh & 4) ? kMhArXLo : -1);
    }
    // ---- end branch term ----
#endif
    uint4 o[kMhRowV];
#if (MSO_MK_RL & 1) && !(MSO_MK_RL & 8)
    float rlInv[2] = {0.f, 0.f}; // MSO_MK_RL: s1, s2
#endif
    mhBoundary
#if MSO_MK_NB
        <(MSO_MK_NB & 1) != 0> // MSO_MK_NB bit 1: MhH1's one exchange (MSO_MK_RL's s2 then comes from it)
#endif
        (ctx, x, cb, cb + 4 * kMhRowW, cb, true, res, cb + 8 * kMhRowW, row, q, 0, nullptr, nullptr,
            ctx.p[kPMhOutRes1] != 0 ? ptr<float>(ctx, kPMhOutRes1) : nullptr, o, 0
#if (MSO_MK_RL & 1) && !(MSO_MK_RL & 8)
            ,
            -1.f, 1.f, rlOn ? rlInv : nullptr // MSO_MK_RL
#elif MSO_MK_FZ
            ,
            -1.f, 1.f, nullptr
#endif
#if MSO_MK_FZ // r' of this rank's unpool columns also into its r' area
            ,
            float4{}, fzUc(ctx) + kMhFzR1Off
#endif
        );
#if MSO_MK_RL & 4
    // ---- the recorded slice into this launch's F2' areas of this rank, before x_ffn ----
    if (rlLg || rlPool)
    {
        uint8_t* const f2 = mhUc(ctx, ctx.copy) + kMhPoolOff;
        if (rlLg && threadIdx.x < 64)
        {
            st32(f2 + rlLogitOff(row, q), rlW);
        }
        if (rlPool)
        {
            size_t const po = rlPoolOff(row, q);
            st128(f2 + po, rlV[0]);
            st128(f2 + po + 16, rlV[1]);
        }
    }
    // ---- end recorded slice ----
#endif
    // MhF1 of this CTA finds r' at this unit's region.
    reinterpret_cast<int*>(ctx.smem + kSmemSmall + kSmallScalars)[100] = static_cast<int>(wOff);
#pragma unroll
    for (int j = 0; j < kMhRowV; ++j)
    {
        int const col = q * kMhRowW + mhLocalCol(j);
#if !MSO_MK_BH // with MSO_MK_BH x_ffn's arena words hold xB; x_ffn goes to the check output only
        st128(mhArena(ctx, ctx.copy) + kMhArX + (static_cast<size_t>(row) * 8192 + col) * 2, lamport4(o[j]));
#endif
        if (MSO_MK_CKP(ctx.p[kPMhOutXffn]) != 0) // MSO_MK_LD_CK
        {
            st128(ptr<__nv_bfloat16>(ctx, kPMhOutXffn) + row * 8192 + col, o[j]);
        }
    }
#if !(MSO_MK_XP & (128 | 256)) // MSO_MK_XP bits 128 / 256: the next launch's MhClear empties the hop-1 area
#pragma unroll
    for (int j = 0; j < kMhRowV; ++j)
    {
#pragma unroll
        for (int r = 0; r < kMhRanks; ++r)
        {
            st128(h1 + ((static_cast<size_t>(r) * 8 + row) * 8192 + q * kMhRowW + mhLocalCol(j)) * 2, emptyVec());
        }
    }
#endif
#if (MSO_MK_RL & 1) && !(MSO_MK_RL & 8)
    if (rlOn) // MSO_MK_RL: the unit's logits into this rank's F2' logit area
    {
        rlFinalize(ctx, row, q, rlInv[0], rlInv[1], rlPv);
    }
#endif
    subStamp(ctx, 3);
}

// The B fragments of kN chunks (warp w: chunks c0 + w + 8 i) of a BF16 [8][ld] activation in the arena, polled.
// kPb: one 256-bit load per fragment (pollVecs256, MSO_MK_PB); x and ld * 2 are 32-byte aligned.
template <int kN, bool kPb = false>
__device__ __forceinline__ void mhPollB(Ctx const& ctx, uint4 (&b)[kN][2], uint8_t const* x, int ld, int c0, int code)
{
    int const lane = threadIdx.x & 31;
    int const warp = threadIdx.x >> 5;
    uint8_t const* a[2 * kN];
#pragma unroll
    for (int i = 0; i < kN; ++i)
    {
        int const col = (c0 + warp + 8 * i) * kChunkCols + 16 * (lane & 3);
        a[2 * i] = x + (static_cast<size_t>(lane >> 2) * ld + col) * 2;
        a[2 * i + 1] = a[2 * i] + 16;
    }
    uint4 v[2 * kN];
    if constexpr (kPb) // one 256-bit load per fragment
    {
        static_assert(kMhArX % 32 == 0 && kMhArM % 32 == 0 && kMhArenaBytes % 32 == 0 && (kMhPoolWidth * 2) % 32 == 0,
            "MSO_MK_PB (kPb): 32-byte pairs");
        pollVecs256<kN>(ctx, v, a, code);
    }
    else
    {
        pollVecs<2 * kN>(ctx, v, a, code);
    }
#pragma unroll
    for (int i = 0; i < kN; ++i)
    {
        b[i][0] = v[2 * i];
        b[i][1] = v[2 * i + 1];
    }
}

// 64 chunks of a packed tile in the unit's region (warp w: chunks w + 8 i) times the polled B fragments; the warps'
// sums in warp order as FP32 [8 rows][16 tile rows] at the reduction area + 8 KiB.
__device__ __forceinline__ float* mhGemvTile(Ctx const& ctx, uint32_t wOff, uint4 const (&b)[8][2])
{
    int const warp = threadIdx.x >> 5;
    float part[4][4] = {};
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        mmaChunkBf16(part[i & 3], ctx.smemBase + wOff + static_cast<uint32_t>(warp + 8 * i) * kBf16Block, b[i]);
    }
    float acc[1][4];
#pragma unroll
    for (int r = 0; r < 4; ++r)
    {
        acc[0][r] = (part[0][r] + part[1][r]) + (part[2][r] + part[3][r]);
    }
    float* out = reinterpret_cast<float*>(ctx.smem + kSmemRed + 8192);
    reduceWarpsTokRow<1>(ctx, acc, out);
    return out;
}

// MhPool: unit u = (tile t = u / 2, K half h = u % 2) of the rank's [512, 8192] slice; its 64 chunks are the unit's
// weights. FP32 partial [8][16] -> pool[h][row][512 rank + 16 t + ...] of every rank.
__device__ __noinline__ void stageMhPool(Ctx const& ctx, int unit, uint32_t wOff, uint32_t bar)
{
    int const t = unit / 2;
    int const h = unit % 2;
#if MSO_MK_BH // the two-term pool
    bhStageMhPool(ctx, t, h, wOff, bar);
    return;
#endif
    uint4 b[8][2];
    if (icPIFlags(ctx.p[kPIFlags]) & kIFlagT3CanaryX) // canary: the MhH1 units of columns [4096 h, +4096)
    {
        t3Canary(ctx, t3RowCanary(mhArena(ctx, ctx.copy) + kMhArX, threadIdx.x & 31, 0, 8, 4096 * h, 4096 * h + 4096),
            kCodeMhX);
    }
    bool const t3Wf = (icPIFlags(ctx.p[kPIFlags]) & kIFlagT3WaitFirst) != 0; // weights first
    if (t3Wf)
    {
        waitPieces(ctx, bar, kCodeWeights);
    }
    mhPollB<8, (MSO_MK_PB & 2) != 0>(ctx, b, mhArena(ctx, ctx.copy) + kMhArX, 8192, 64 * h, kCodeMhX); // PB bit 2
    stampInput(ctx);
    if (!t3Wf)
    {
        waitPieces(ctx, bar, kCodeWeights);
    }
    float const* out = mhGemvTile(ctx, wOff, b);
    int const i = threadIdx.x;
    if (i < 32)
    {
        int const row = i / 4;
        int const r4 = 4 * (i % 4);
        float4 const s = *reinterpret_cast<float4 const*>(out + row * 16 + r4);
        mhPublish(ctx,
            kMhPoolOff
                + ((static_cast<size_t>(h) * 8 + row) * kMhPoolWidth + kMhPoolLocal * mhRank(ctx) + 16 * t + r4) * 4,
            lamport4(f4u(s.x, s.y, s.z, s.w)));
    }
}

// MhRouter: unit e = expert 32 rank + e; the FP32 row (unit weights) times x_ffn in the stock summation order of
// router_shard (one Triton program per row and expert, 16 warps, the [1, 8192] tile in 4-column FP32 vectors, so each
// of its 512 threads holds p(j, v) = w * x at column 2048 j + 4 vt + v, j, v < 4; its PTX sums them as f32x2 pairs:
// per parity c, ((q0 + q1) + (q2 + q3)) with q_j = p(j, c) + p(j, c + 2), then parity 0 + parity 1). Thread t plays
// the stock threads vt = 2t and 2t + 1 (columns 2048 j + 8t + [0, 8): one 16-byte x_ffn vector per row and j); then
// the stock warp butterfly (xor 16, 8, 4, 2, 1 over the stock lanes = xor 8, 4, 2, 1 over this warp's lanes, then
// the thread's two stock threads); the 16 stock warp sums (stock warp t / 16) in the stock inter-warp order (xor 8, 4,
// 2, 1); then x scale. 8 FP32 -> logits[expert][0..8) of every rank.
//
// ---- row-split router units ----
// unit = e + 32 h + 256 split (e < 32: the expert row; split 0: all 8 rows, h = 0; split 1: rows [4 h, 4 h + 4), h <
// 2). Every row's logit is computed by the same threads in the same order whatever the split (the rows are
// independent), so the logits are the same bits; a split unit polls and multiplies half of x_ffn.
__device__ __forceinline__ int mhRouterExpert(int unit)
{
    return unit & (kMhLocalExperts - 1);
}

// ---- end row-split router units ----
#ifndef MSO_MK_LY_PAD
#define MSO_MK_LY_PAD 0
#endif
#ifndef MSO_MK_LY_PADX
#define MSO_MK_LY_PADX 0
#endif
#if MSO_MK_LY_PAD > 0 || MSO_MK_LY_PADX > 0
// ---- attribution only, never served: layout-only twins. ptxas lays the kernel's callees out in the order of their
// mangled names. MSO_MK_LY_PAD: that many NANOSLEEP instructions (16 B each) at the head of the router-free
// stageMhRouter, which no unit of these builds runs, so every function after it (stageMhUnpool .. stageMhH1: the
// experts functions, ahMerge, stageAhSwaAttn, stageMhF1, stageMhH1) moves by the pad. MSO_MK_LY_PADX: the same pad in
// p4a::lyPadEpilog (stages/experts.cuh), which sorts between p4a::finalUnitPx and p4a::epilogueLoop and is called only
// from that never-run path, so the experts' epilogue / producer / MMA loops (and everything after them) move against
// the loader loop. No executed instruction changes in either. ----
#if !((MSO_MK_BH & 8) || (MSO_MK_RL & 16))
#error "MSO_MK_LY_PAD / MSO_MK_LY_PADX need the router-free stageMhRouter (MSO_MK_BH bit 8 or MSO_MK_RL bit 16)"
#endif
#if MSO_MK_LY_PADX > 0
namespace p4a
{
__device__ __noinline__ void lyPadEpilog();
} // namespace p4a
#endif
template <int kN>
__device__ __forceinline__ void lyPad()
{
#pragma unroll
    for (int i = 0; i < kN; ++i)
    {
        asm volatile("nanosleep.u32 0;" ::: "memory");
    }
}

// ---- end layout-only twins ----
#endif
__device__ __noinline__ void stageMhRouter(Ctx const& ctx, int unit, uint32_t wOff, uint32_t bar)
{
#if (MSO_MK_BH & 8) || (MSO_MK_RL & 16) // no router code (MSO_MK_BH pool-only consumers / MSO_MK_RL router-free): every
    // router slot runs the rl units (kIFlagRlOn on every layer)
#if MSO_MK_LY_PAD > 0
    lyPad<MSO_MK_LY_PAD>(); // layout-only pad (attribution)
#endif
#if MSO_MK_LY_PADX > 0
    p4a::lyPadEpilog(); // keeps the experts-block pad in the kernel's call graph (never runs)
#endif
    __trap();
    return;
#elif MSO_MK_BH // the two-term router
    bhStageMhRouter(ctx, unit, wOff, bar);
    return;
#endif
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    uint8_t const* x = mhArena(ctx, ctx.copy) + kMhArX;
    float* red = reinterpret_cast<float*>(ctx.smem + kSmemRed);
    // ---- the unit's row halves [h0, h1) ----
    int const split = unit >> 8;
    int const h0 = split == 0 ? 0 : (unit >> 5) & 1;
    int const h1 = split == 0 ? 2 : h0 + 1;
    // ---- end row halves ----
    waitPieces(ctx, bar, kCodeWeights);
    if (icPIFlags(ctx.p[kPIFlags]) & kIFlagT3CanaryX) // canary: the MhH1 units of rows [4 h0, 4 h1)
    {
        t3Canary(ctx, t3RowCanary(x, threadIdx.x & 31, 4 * h0, 4 * h1, 0, 8192), kCodeMhX);
    }
#pragma unroll
    for (int half = 0; half < 2; ++half)
    {
        if (half < h0 || half >= h1) // not one of the unit's row halves
        {
            continue;
        }
        uint8_t const* a[16];
#pragma unroll
        for (int k = 0; k < 16; ++k)
        {
            int const row = 4 * half + k / 4;
            a[k] = x + (static_cast<size_t>(row) * 8192 + 8 * tid + 2048 * (k % 4)) * 2;
        }
        uint4 v[16];
        pollVecs<16>(ctx, v, a, kCodeMhX);
        if (half == h1 - 1) // the unit's last row half
        {
            stampInput(ctx);
        }
#pragma unroll
        for (int rr = 0; rr < 4; ++rr)
        {
            float q[2][2][4]; // [stock thread h][parity c][j]
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                float wv[8];
                lds8f(wv, ctx.smemBase + wOff + 4 * (8 * tid + 2048 * j));
                float xv[8];
                unpack8(xv, v[4 * rr + j]);
#pragma unroll
                for (int h = 0; h < 2; ++h)
                {
#pragma unroll
                    for (int c = 0; c < 2; ++c)
                    {
                        q[h][c][j] = __fadd_rn(
                            __fmul_rn(wv[4 * h + c], xv[4 * h + c]), __fmul_rn(wv[4 * h + c + 2], xv[4 * h + c + 2]));
                    }
                }
            }
            float s[2];
#pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                float a[2];
#pragma unroll
                for (int c = 0; c < 2; ++c)
                {
                    a[c] = __fadd_rn(__fadd_rn(q[h][c][0], q[h][c][1]), __fadd_rn(q[h][c][2], q[h][c][3]));
                }
                s[h] = __fadd_rn(a[0], a[1]);
            }
#pragma unroll
            for (int m = 8; m >= 1; m >>= 1)
            {
#pragma unroll
                for (int h = 0; h < 2; ++h)
                {
                    s[h] = __fadd_rn(s[h], __shfl_xor_sync(0xffffffffu, s[h], m));
                }
            }
            float const w = __fadd_rn(s[0], s[1]);
            if ((lane & 15) == 0)
            {
                red[(tid >> 4) * 8 + 4 * half + rr] = w;
            }
        }
    }
    __syncthreads();
    if (tid >= 4 * h0 && tid < 4 * h1) // one thread per row of the unit's halves
    {
        float b[16];
#pragma unroll
        for (int i = 0; i < 16; ++i)
        {
            b[i] = red[i * 8 + tid];
        }
#pragma unroll
        for (int n = 8; n >= 1; n >>= 1)
        {
#pragma unroll
            for (int i = 0; i < n; ++i)
            {
                b[i] = __fadd_rn(b[i], b[i ^ n]);
            }
        }
        red[128 + tid] = __fmul_rn(b[0], __int_as_float(static_cast<int>(ctx.p[kPMhScale])));
    }
    __syncthreads();
    if (tid >= h0 && tid < h1) // one thread per row half of the unit
    {
        float4 const s = *reinterpret_cast<float4 const*>(red + 128 + 4 * tid);
        int const e = kMhLocalExperts * mhRank(ctx) + mhRouterExpert(unit); // the unit's expert row
        mhPublish(ctx, kMhLogitOff + (static_cast<size_t>(e) * 8 + 4 * tid) * 4, lamport4(f4u(s.x, s.y, s.z, s.w)));
    }
}

// ---- rank-local router ----
// Every rank computes all 256 logits from its own x_ffn (which every rank holds after hop 1), so the logits need no
// cross-rank hop: unit = 512 + 64 h + q computes experts 4 q .. 4 q + 3 for rows [4 h, 4 h + 4) from the full FP32
// router table (unit weights: the 4 rows, 128 KiB). Per expert and row: the same threads, products, butterflies and
// inter-warp order as stageMhRouter (whose weights are re-read per row; reading them once per j is the same values), so
// the same bits. The logits go into this rank's own copy of the logit area only (plain stores, kEmpty-polled there).
__device__ __noinline__ void stageMhRouterLocal(Ctx const& ctx, int unit, uint32_t wOff, uint32_t bar)
{
#if MSO_MK_BH // x_ffn's arena words hold xB; the rank-local router has no two-term form
    __trap();
#endif
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const half = (unit >> 6) & 1;
    int const q = unit & 63;
    uint8_t const* x = mhArena(ctx, ctx.copy) + kMhArX;
    float* red = reinterpret_cast<float*>(ctx.smem + kSmemRed);
    waitPieces(ctx, bar, kCodeWeights);
    uint8_t const* a[16];
#pragma unroll
    for (int k = 0; k < 16; ++k)
    {
        int const row = 4 * half + k / 4;
        a[k] = x + (static_cast<size_t>(row) * 8192 + 8 * tid + 2048 * (k % 4)) * 2;
    }
    uint4 v[16];
    pollVecs<16>(ctx, v, a, kCodeMhX);
    stampInput(ctx);
    // Two experts at a time (independent chains for latency hiding; the per-expert arithmetic is unchanged).
#pragma unroll 2
    for (int i = 0; i < kMhRouterLocalExperts; ++i)
    {
        float wv[4][8];
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            lds8f(wv[j], ctx.smemBase + wOff + i * kMhRouterUnitBytes + 4 * (8 * tid + 2048 * j));
        }
#pragma unroll
        for (int rr = 0; rr < 4; ++rr)
        {
            float qv[2][2][4]; // [stock thread h][parity c][j]
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                float xv[8];
                unpack8(xv, v[4 * rr + j]);
#pragma unroll
                for (int h = 0; h < 2; ++h)
                {
#pragma unroll
                    for (int c = 0; c < 2; ++c)
                    {
                        qv[h][c][j] = __fadd_rn(__fmul_rn(wv[j][4 * h + c], xv[4 * h + c]),
                            __fmul_rn(wv[j][4 * h + c + 2], xv[4 * h + c + 2]));
                    }
                }
            }
            float sum[2];
#pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                float pa[2];
#pragma unroll
                for (int c = 0; c < 2; ++c)
                {
                    pa[c] = __fadd_rn(__fadd_rn(qv[h][c][0], qv[h][c][1]), __fadd_rn(qv[h][c][2], qv[h][c][3]));
                }
                sum[h] = __fadd_rn(pa[0], pa[1]);
            }
#pragma unroll
            for (int m = 8; m >= 1; m >>= 1)
            {
#pragma unroll
                for (int h = 0; h < 2; ++h)
                {
                    sum[h] = __fadd_rn(sum[h], __shfl_xor_sync(0xffffffffu, sum[h], m));
                }
            }
            float const w = __fadd_rn(sum[0], sum[1]);
            if ((lane & 15) == 0)
            {
                red[i * 128 + (tid >> 4) * 8 + 4 * half + rr] = w;
            }
        }
    }
    __syncthreads();
    if (tid < 4 * kMhRouterLocalExperts)
    {
        int const i = tid / 4;
        int const row = 4 * half + tid % 4;
        float b[16];
#pragma unroll
        for (int k = 0; k < 16; ++k)
        {
            b[k] = red[i * 128 + k * 8 + row];
        }
#pragma unroll
        for (int n = 8; n >= 1; n >>= 1)
        {
#pragma unroll
            for (int k = 0; k < n; ++k)
            {
                b[k] = __fadd_rn(b[k], b[k ^ n]);
            }
        }
        red[512 + 4 * i + tid % 4] = __fmul_rn(b[0], __int_as_float(static_cast<int>(ctx.p[kPMhScale])));
    }
    __syncthreads();
    if (tid < kMhRouterLocalExperts)
    {
        float4 const r4 = *reinterpret_cast<float4 const*>(red + 512 + 4 * tid);
        int const e = kMhRouterLocalExperts * q + tid;
        st128(mhUc(ctx, ctx.copy) + kMhLogitOff + (static_cast<size_t>(e) * 8 + 4 * half) * 4,
            lamport4(f4u(r4.x, r4.y, r4.z, r4.w)));
    }
}

// ---- end rank-local router ----

// F2' reader helpers (the experts stage's inputs): this launch's local copy of the gathered pool output and logits.
// mhPoolPtrs: the 4 vectors (2 halves x 2 x 4 FP32) of row `row`, pooled columns [col, col + 8).
__device__ __forceinline__ void mhPoolPtrs(Ctx const& ctx, uint8_t const* (&a)[4], int row, int col)
{
    uint8_t const* base = mhUc(ctx, ctx.copy) + kMhPoolOff;
#pragma unroll
    for (int h = 0; h < 2; ++h)
    {
        a[2 * h] = base + ((static_cast<size_t>(h) * 8 + row) * kMhPoolWidth + col) * 4;
        a[2 * h + 1] = a[2 * h] + 16;
    }
}

// The 8 BF16 pooled values from the 4 polled vectors: bf16(half0 + half1).
__device__ __forceinline__ uint4 mhPoolValues(uint4 const (&v)[4])
{
    float s[8];
    uint32_t const p0[8] = {v[0].x, v[0].y, v[0].z, v[0].w, v[1].x, v[1].y, v[1].z, v[1].w};
    uint32_t const p1[8] = {v[2].x, v[2].y, v[2].z, v[2].w, v[3].x, v[3].y, v[3].z, v[3].w};
#pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        s[j] = __fadd_rn(__uint_as_float(p0[j]), __uint_as_float(p1[j]));
    }
    return pack8(s);
}

// Logit vector k (0..511): experts k / 2, rows 4 (k % 2) .. + 4, FP32.
__device__ __forceinline__ uint8_t const* mhLogitPtr(Ctx const& ctx, int k)
{
    return mhUc(ctx, ctx.copy) + kMhLogitOff + 16 * static_cast<size_t>(k);
}

// MhExpStub: unit u = (row u / 4, pooled columns [1024 (u % 4), +1024)). Warps 0-3 (thread t: columns + 8 t) poll the
// gathered pool output, warp 4 lanes 0-15 the unit's 16 logit vectors (16 u + lane); then threads 0-127 publish the
// given partial's columns into slot `rank` of every rank's finalize area.
__device__ __noinline__ void stageMhExpStub(Ctx const& ctx, int unit)
{
    int const tid = threadIdx.x;
    int const row = unit / 4;
    int const col = 1024 * (unit % 4) + 8 * tid;
    if (tid < 128)
    {
        uint8_t const* a[4];
        mhPoolPtrs(ctx, a, row, col);
        uint4 v[4];
        pollVecs<4>(ctx, v, a, kCodeMhF2);
        uint4 const xp = mhPoolValues(v);
        if (MSO_MK_CKP(ctx.p[kPMhOutPool]) != 0) // MSO_MK_LD_CK
        {
            st128(ptr<__nv_bfloat16>(ctx, kPMhOutPool) + row * kMhPoolWidth + col, xp);
        }
    }
    else if (tid < 160)
    {
        int const lane = tid - 128;
        int const k = 16 * unit + lane;
        uint8_t const* a[1] = {lane < 16 ? mhLogitPtr(ctx, k) : nullptr};
        uint4 v[1];
        pollVecs<1>(ctx, v, a, kCodeMhF2);
        if (lane < 16 && MSO_MK_CKP(ctx.p[kPMhOutLogits]) != 0) // MSO_MK_LD_CK
        {
            st128(ptr<float>(ctx, kPMhOutLogits) + 4 * k, v[0]);
        }
    }
    stampInput(ctx);
    __syncthreads();
    if (tid < 128)
    {
        uint4 const p = ldcg128(ptr<__nv_bfloat16 const>(ctx, kPMhInExpert) + row * kMhPoolWidth + col);
        mhPublish(
            ctx, kMhFinOff + ((static_cast<size_t>(mhRank(ctx)) * 8 + row) * kMhPoolWidth + col) * 2, lamport4(p));
    }
}

// MhFinal: unit u = (row u / 4, columns [1024 (u % 4), +1024)), threads 0-127: 8 columns each; the 8 partials summed in
// rank order from +0, one BF16 rounding -> m (arena), then the slots reset.
// ---- kIFlagT3Final256: MhFinal over 256 threads, 4 columns (8 bytes) each ----
template <int kN>
__device__ __forceinline__ void t3PollV2(Ctx const& ctx, uint2 (&v)[kN], uint8_t const* const (&a)[kN], int code)
{
#pragma unroll
    for (int i = 0; i < kN; ++i)
    {
        asm volatile("ld.volatile.global.v2.u32 {%0, %1}, [%2];" : "=r"(v[i].x), "=r"(v[i].y) : "l"(a[i]));
    }
    uint64_t t0 = 0;
    while (true)
    {
        bool ok = true;
#pragma unroll
        for (int i = 0; i < kN; ++i)
        {
            ok &= (v[i].x != kEmpty) & (v[i].y != kEmpty);
        }
        if (warpAll(ok))
        {
            break;
        }
        spinCheck(ctx, t0, code, kN, 0);
#pragma unroll
        for (int i = 0; i < kN; ++i)
        {
            if (v[i].x == kEmpty || v[i].y == kEmpty)
            {
                asm volatile("ld.volatile.global.v2.u32 {%0, %1}, [%2];" : "=r"(v[i].x), "=r"(v[i].y) : "l"(a[i]));
            }
        }
    }
}

__device__ __forceinline__ void t3St64(void* p, uint32_t x, uint32_t y)
{
    asm volatile("st.global.v2.b32 [%0], {%1, %2};" ::"l"(p), "r"(x), "r"(y) : "memory");
}

// The same per-column sums as stageMhFinal (8 partials in rank order from +0, one BF16 rounding), 4 columns per thread.
__device__ __noinline__ void t3StageMhFinal256(Ctx const& ctx, int unit)
{
    int const row = unit / 4;
    int const col = 1024 * (unit % 4) + 4 * static_cast<int>(threadIdx.x);
    uint8_t* const fin = mhUc(ctx, ctx.copy) + kMhFinOff;
    uint8_t const* a[kMhRanks];
#pragma unroll
    for (int r = 0; r < kMhRanks; ++r)
    {
        a[r] = fin + ((static_cast<size_t>(r) * 8 + row) * kMhPoolWidth + col) * 2;
    }
    uint2 v[kMhRanks];
    t3PollV2<kMhRanks>(ctx, v, a, kCodeMhFin);
    stampInput(ctx);
    float s[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int r = 0; r < kMhRanks; ++r)
    {
        s[0] = __fadd_rn(s[0], bf16Lo(v[r].x));
        s[1] = __fadd_rn(s[1], bf16Hi(v[r].x));
        s[2] = __fadd_rn(s[2], bf16Lo(v[r].y));
        s[3] = __fadd_rn(s[3], bf16Hi(v[r].y));
    }
    uint32_t const m0 = packBf16(s[0], s[1]);
    uint32_t const m1 = packBf16(s[2], s[3]);
    t3St64(mhArena(ctx, ctx.copy) + kMhArM + (static_cast<size_t>(row) * kMhPoolWidth + col) * 2, lamport(m0),
        lamport(m1));
    if (MSO_MK_CKP(ctx.p[kPMhOutM]) != 0) // MSO_MK_LD_CK
    {
        t3St64(ptr<__nv_bfloat16>(ctx, kPMhOutM) + row * kMhPoolWidth + col, m0, m1);
    }
#pragma unroll
    for (int r = 0; r < kMhRanks; ++r)
    {
        t3St64(const_cast<uint8_t*>(a[r]), kEmpty, kEmpty);
    }
}

// ---- end kIFlagT3Final256 ----

__device__ __noinline__ void stageMhFinal(Ctx const& ctx, int unit)
{
    if (icPIFlags(ctx.p[kPIFlags]) & kIFlagT3Final256) // 4 columns per thread
    {
        t3StageMhFinal256(ctx, unit);
        return;
    }
    int const tid = threadIdx.x;
    if (tid >= 128)
    {
        return;
    }
    int const row = unit / 4;
    int const col = 1024 * (unit % 4) + 8 * tid;
    uint8_t* const fin = mhUc(ctx, ctx.copy) + kMhFinOff;
    uint8_t const* a[kMhRanks];
#pragma unroll
    for (int r = 0; r < kMhRanks; ++r)
    {
        a[r] = fin + ((static_cast<size_t>(r) * 8 + row) * kMhPoolWidth + col) * 2;
    }
    uint4 v[kMhRanks];
    pollVecs<kMhRanks>(ctx, v, a, kCodeMhFin);
    stampInput(ctx);
    float s[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int r = 0; r < kMhRanks; ++r)
    {
        float p[8];
        unpack8(p, v[r]);
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            s[j] = __fadd_rn(s[j], p[j]);
        }
    }
    uint4 const m = pack8(s);
    st128(mhArena(ctx, ctx.copy) + kMhArM + (static_cast<size_t>(row) * kMhPoolWidth + col) * 2, lamport4(m));
    if (MSO_MK_CKP(ctx.p[kPMhOutM]) != 0) // MSO_MK_LD_CK
    {
        st128(ptr<__nv_bfloat16>(ctx, kPMhOutM) + row * kMhPoolWidth + col, m);
    }
#pragma unroll
    for (int r = 0; r < kMhRanks; ++r)
    {
        st128(const_cast<uint8_t*>(a[r]), emptyVec());
    }
}

// MhUnpool: unit t = tile t of the rank's [1024, 4096] slice (64 chunks = the unit's weights) times m (polled); BF16
// [8][16] -> f1[row][1024 rank + 16 t + ...] of every rank.
__device__ __noinline__ void stageMhUnpool(Ctx const& ctx, int unit, uint32_t wOff, uint32_t bar)
{
    uint4 b[8][2];
    // kIFlagT3CanaryM: lane l = MhFinal unit l's last vector (row l / 4, column 1024 (l % 4) + 1016)
    if (icPIFlags(ctx.p[kPIFlags]) & kIFlagT3CanaryM)
    {
        int const l = threadIdx.x & 31;
        t3Canary(ctx,
            mhArena(ctx, ctx.copy) + kMhArM + (static_cast<size_t>(l / 4) * kMhPoolWidth + 1024 * (l % 4) + 1016) * 2
                + 12,
            kCodeMhM);
    }
    bool const t3Wf = (icPIFlags(ctx.p[kPIFlags]) & kIFlagT3WaitFirst) != 0; // weights first
    if (t3Wf)
    {
        waitPieces(ctx, bar, kCodeWeights);
    }
#if MSO_MK_FZ
    // ---- thread i < 16 (row i / 2, columns 1024 rank + 16 t + 8 (i % 2)): its r' loads issued before the m poll
    // (checked after the GEMV), its F1 coefficients copied into its slot. Placed after the weights wait, where the
    // stage already holds its shared-memory base and memory descriptor, so the stage's uniform registers stay the
    // base build's (the kernel-wide allocation depends on them). ----
    int const fzRow = (threadIdx.x & 15) / 2;
    int const fzCol = kMhUnpoolLocal * mhRank(ctx) + 16 * unit + 8 * (threadIdx.x & 1);
    uint8_t const* const fzR1 = fzUc(ctx) + kMhFzR1Off
        + (static_cast<size_t>(fzRow) * kMhUnpoolLocal + 16 * unit + 8 * (threadIdx.x & 1)) * 4;
    uint4 fzR[2] = {emptyVec(), emptyVec()};
    if (threadIdx.x < 16)
    {
        fzR[0] = ldPoll128(fzR1);
        fzR[1] = ldPoll128(fzR1 + 16);
        fzPrefetchCoef(ctx, threadIdx.x, fzCol);
    }
    // ---- end MSO_MK_FZ prefetch ----
#endif
    mhPollB<8, (MSO_MK_PB & 1) != 0>(ctx, b, mhArena(ctx, ctx.copy) + kMhArM, kMhPoolWidth, 0, kCodeMhM); // PB bit 1
    stampInput(ctx);
    if (!t3Wf)
    {
        waitPieces(ctx, bar, kCodeWeights);
    }
    float const* out = mhGemvTile(ctx, wOff, b);
    int const i = threadIdx.x;
#if MSO_MK_FZ
    uint4 fzB = make_uint4(0, 0, 0, 0); // the published b vector
#endif
    if (i < 16)
    {
        int const row = i / 2;
        int const c8 = 8 * (i % 2);
        float s[8];
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            s[j] = out[row * 16 + c8 + j];
        }
#if MSO_MK_FZ
        fzB = lamport4(pack8(s));
        mhPublish(
            ctx, kMhF1Off + (static_cast<size_t>(row) * 8192 + kMhUnpoolLocal * mhRank(ctx) + 16 * unit + c8) * 2, fzB);
#else
        mhPublish(ctx, kMhF1Off + (static_cast<size_t>(row) * 8192 + kMhUnpoolLocal * mhRank(ctx) + 16 * unit + c8) * 2,
            lamport4(pack8(s)));
#endif
    }
#if MSO_MK_FZ
    {
        // thread i < 16 completes its r' (a word still empty is polled now). Every thread of the CTA calls the poll
        // here, where the warps are converged (the others pass nullptr): a warp collective inside warp 0's branch needs
        // divergence checks that re-partition the kernel-wide register allocation.
        uint8_t const* const ar[2]
            = {(i < 16 && !full4(fzR[0])) ? fzR1 : nullptr, (i < 16 && !full4(fzR[1])) ? fzR1 + 16 : nullptr};
        uint4 w[2];
        pollVecs<2>(ctx, w, ar, kCodeMhM);
        if (ar[0] != nullptr)
        {
            fzR[0] = w[0];
        }
        if (ar[1] != nullptr)
        {
            fzR[1] = w[1];
        }
    }
    if (i < 16) // lanes < 16 of warp 0, after b: the leaves
    {
        fzPublishLeaf(ctx, fzR, fzR1, fzRow, fzCol, fzB);
    }
#endif
}

// MhF1: unit u = (row u / U, part q), the thread map of MhH1 (whose region on this CTA holds r'). Coefficients (unit
// weights at wOff): alpha2 [W] FP32, beta2 [W] FP32, gain_pf [W] BF16, w_next [W] BF16. The gathered branch b,
// sigma = bf16(b * gain), the boundary; the residual r'' to global memory (the next launch's input), x_attn, capture.
__device__ __noinline__ void stageMhF1(Ctx const& ctx, int unit, uint32_t wOff, uint32_t bar, Io const& io)
{
    int const row = unit / kMhRowU;
    int const q = unit % kMhRowU;
    uint8_t* const f1 = mhUc(ctx, ctx.copy) + kMhF1Off;
    // kMhF1Full: the whole row (thread t's vector jf covers columns 8 (t + 256 jf), so this unit's vector j is jf =
    // q kMhRowV + j); else this unit's columns only.
    constexpr int kP = kMhF1Full ? 8192 / (8 * kThreads) : kMhRowV;
    int const jOwn = kMhF1Full ? q * kMhRowV : 0;
#if MSO_MK_FZ // + the row's leaves of stock warps (q', w), lane l (columns 2048 q' + 256 w + 8 l), q' = 0..3
    uint8_t const* a[kP + 4];
#else
    uint8_t const* a[kP];
#endif
#pragma unroll
    for (int j = 0; j < kP; ++j)
    {
        a[j] = f1 + (static_cast<size_t>(row) * 8192 + (kMhF1Full ? 0 : q * kMhRowW) + mhLocalCol(j)) * 2;
    }
#if MSO_MK_FZ
#pragma unroll
    for (int k = 0; k < 4; ++k)
    {
        a[kP + k] = fzUc(ctx) + kMhFzLeafOff + ((static_cast<size_t>(q) * 8 + row) * 1024 + 256 * k + threadIdx.x) * 16;
    }
    uint4 v[kP + 4];
#else
    uint4 v[kP];
#endif
    // ---- weights and r' before the poll (kIFlagT3WaitFirst, kIFlagT3EarlyRes1) ----
    bool const t3Wf = (icPIFlags(ctx.p[kPIFlags]) & kIFlagT3WaitFirst) != 0;
    if (t3Wf)
    {
        waitPieces(ctx, bar, kCodeWeights);
    }
    bool const rGlobal = (icPIFlags(ctx.p[kPIFlags]) & kIFlagMhRes1Global) != 0;
    bool const t3R1 = rGlobal && (icPIFlags(ctx.p[kPIFlags]) & kIFlagT3EarlyRes1) != 0;
    float unused[kMhRowV][8] = {};
    if (t3R1)
    {
        float const* r1 = ptr<float const>(ctx, kPMhOutRes1) + row * 8192 + q * kMhRowW;
#pragma unroll
        for (int j = 0; j < kMhRowV; ++j)
        {
            ld8f(unused[j], r1 + mhLocalCol(j));
        }
    }
    // ---- end weights and r' first ----
#if MSO_MK_FZ
    pollVecs<kP + 4>(ctx, v, a, kCodeMhF1);
#else
    pollVecs<kP>(ctx, v, a, kCodeMhF1);
#endif
#if MSO_MK_RL & 2
    if (kRlOk && (ctx.p[kPIFlags] & kIFlagRlRecord)) // the recording launch's copy of the F2' slice
    {
        rlRecord(ctx, row, q);
    }
#endif
    stampInput(ctx);
    subStamp(ctx, 4);
    // ---- stamped builds only: sub 17 = every warp of the CTA has its F1 data ----
    if (MSO_MK_STAMPS_DETAIL(ctx) != nullptr)
    {
        __syncthreads();
        subStamp(ctx, 17);
    }
    // ---- end stamped builds only ----
#if MSO_MK_FZ
    // ---- the row's four sums from its leaves in nbRowSum4's tree (mhBoundary takes inv1 / inv2 from them); the leaf
    // words reset by their one reader ----
    float4 fzSums;
    {
        uint4 const lv[4] = {v[kP], v[kP + 1], v[kP + 2], v[kP + 3]};
        fzSums = fzRowSum4(ctx, lv);
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            st128(const_cast<uint8_t*>(a[kP + k]), emptyVec());
        }
    }
    // ---- end leaf sums ----
#endif
    if (!t3Wf)
    {
        waitPieces(ctx, bar, kCodeWeights);
    }
    uint32_t const cb = ctx.smemBase + wOff;
    uint32_t const h1 = ctx.smemBase
        + static_cast<uint32_t>(reinterpret_cast<int const*>(ctx.smem + kSmemSmall + kSmallScalars)[100]);
    float x[kMhRowV][8];
    uint4 vOwn[kMhRowV]; // this unit's vectors (selected in unrolled loops: no run-time index into v)
    float inv1 = -1.f;
    if constexpr (kMhF1Full)
    {
        // The stock first norm over the whole row, in this CTA: per vector the stock thread partial and the warp
        // butterfly (stock slot 8 jf + w), the 32 slots through shared memory, then the second butterfly.
        uint32_t const gainS = cb + kMhCoefF1Bytes;
        float* slots = reinterpret_cast<float*>(ctx.smem + kSmemRed) + 32 * 2;
        int const lane = threadIdx.x & 31;
        int const warp = threadIdx.x >> 5;
#pragma unroll
        for (int jf = 0; jf < kP; ++jf)
        {
            float b[8], g[8], xf[8];
            unpack8(b, v[jf]);
            unpack8(g, lds128(gainS + 2 * mhLocalCol(jf)));
            float t = 0.f;
#pragma unroll
            for (int k = 0; k < 8; ++k)
            {
                xf[k] = bf16Round(__fmul_rn(b[k], g[k]));
                t = mhSq(t, xf[k]);
            }
#pragma unroll
            for (int m = 16; m >= 1; m >>= 1)
            {
                t = __fadd_rn(t, __shfl_xor_sync(0xffffffffu, t, m));
            }
            if (lane == 0)
            {
                slots[8 * jf + warp] = t;
            }
#pragma unroll
            for (int j = 0; j < kMhRowV; ++j)
            {
                if (jf == jOwn + j)
                {
                    vOwn[j] = v[jf];
#pragma unroll
                    for (int k = 0; k < 8; ++k)
                    {
                        x[j][k] = xf[k];
                    }
                }
            }
        }
        __syncthreads();
        float t = __fadd_rn(0.f, slots[lane]);
#pragma unroll
        for (int m = 16; m >= 1; m >>= 1)
        {
            t = __fadd_rn(t, __shfl_xor_sync(0xffffffffu, t, m));
        }
        inv1 = mhRsqrt(t, 1e-8f);
    }
    else
    {
#pragma unroll
        for (int j = 0; j < kMhRowV; ++j)
        {
            float b[8], g[8];
            vOwn[j] = v[j];
            unpack8(b, v[j]);
            unpack8(g, lds128(cb + 8 * kMhRowW + 2 * mhLocalCol(j)));
#pragma unroll
            for (int k = 0; k < 8; ++k)
            {
                x[j][k] = bf16Round(__fmul_rn(b[k], g[k]));
            }
        }
    }
    // ---- kIFlagMhRes1Global: r' from global memory; r'' to kPIResOut ----
    if (rGlobal && !t3R1) // with kIFlagT3EarlyRes1 loaded before the poll
    {
        float const* r1 = ptr<float const>(ctx, kPMhOutRes1) + row * 8192 + q * kMhRowW;
#pragma unroll
        for (int j = 0; j < kMhRowV; ++j)
        {
            ld8f(unused[j], r1 + mhLocalCol(j));
        }
    }
    float* const rOut
        = ctx.p[kPIResOut] != 0 ? ptr<float>(ctx, kPIResOut) : static_cast<float*>(const_cast<void*>(io.res));
    // ---- end kIFlagMhRes1Global ----
    uint4 o[kMhRowV];
    mhBoundary
#if MSO_MK_NB
#if MSO_MK_FZ
        <(MSO_MK_NB & 2) != 0, true> // MSO_MK_FZ: MhF1's sums from its leaves
#else
        <(MSO_MK_NB & 2) != 0> // MSO_MK_NB bit 2: MhF1's one exchange
#endif
#endif
        (ctx, x, cb, cb + 4 * kMhRowW, h1, rGlobal, unused, cb + 10 * kMhRowW, row, q, 2, rOut,
            ctx.p[kPMhCapture] != 0 ? ptr<__nv_bfloat16>(ctx, kPMhCapture) : nullptr,
            MSO_MK_CKP(ctx.p[kPMhOutRes2]) != 0 ? ptr<float>(ctx, kPMhOutRes2) : nullptr, o, 4, inv1, // MSO_MK_LD_CK
            (icPIFlags(ctx.p[kPIFlags]) & kIFlagZ1FinalNorm) ? 0.f : 1.f // layer 61's final norm (raw weight)
#if MSO_MK_FZ
            ,
            nullptr, fzSums // the row's four sums from the leaves (no rl norm scalars at MhF1)
#endif
        );
#pragma unroll
    for (int j = 0; j < kMhRowV; ++j)
    {
        int const col = q * kMhRowW + mhLocalCol(j);
        if (ctx.p[kPMhOutXattn] != 0)
        {
            st128(ptr<__nv_bfloat16>(ctx, kPMhOutXattn) + row * 8192 + col, o[j]);
        }
        // ---- kIFlagXOut: early x for the next launch (copy n + 1 of kPIXBuf, Lamport words) ----
        if (icPIFlags(ctx.p[kPIFlags]) & kIFlagXOut)
        {
            st128(ptr<uint8_t>(ctx, kPIXBuf) + static_cast<size_t>(MSO_MK_COPY_NEXT(ctx.copy, ctx.numCopies)) * kIXBytes
                    + (static_cast<size_t>(row) * 8192 + col) * 2,
                lamport4(o[j]));
        }
        // ---- end kIFlagXOut ----
        if (MSO_MK_CKP(ctx.p[kPMhOutB]) != 0) // MSO_MK_LD_CK
        {
            st128(ptr<__nv_bfloat16>(ctx, kPMhOutB) + row * 8192 + col, vOwn[j]);
        }
        // This unit's columns only. With kMhF1Full the row's other units read them too: each did so before it
        // published the second norm's slots, which this unit has received.
        st128(f1 + (static_cast<size_t>(row) * 8192 + col) * 2, emptyVec());
    }
    subStamp(ctx, 7);
}

#if MSO_MK_XP & 128
// MSO_MK_XP bit 128: the previous copy's hop-1 area in kXpH1ClearUnits slices of whole vectors (MhClear units 0 ..).
constexpr int kXpH1ClearUnits = 16;
constexpr int kXpH1ClearUnitBytes = static_cast<int>(kMhH1Bytes / kXpH1ClearUnits);
static_assert(kXpH1ClearUnits <= kMhClearUnits && kMhH1Bytes % (16 * kThreads * kXpH1ClearUnits) == 0,
    "xp: the hop-1 clear slices are whole rounds of vectors");
#endif
#if MSO_MK_XP & 256
// MSO_MK_XP bit 256: MhClear's one byte range is the arena copy, then the multicast copy's hop-1 + pool + logit areas
// (contiguous from kMhH1Off), in units of kXpClearUnitBytes (whole vectors; the last unit ends at the span's end).
constexpr int64_t kXpClearSpan = kMhArenaBytes + kMhH1Bytes + kMhPoolBytes + kMhLogitBytes;
constexpr int64_t kXpClearUnitBytes = (kXpClearSpan / 16 + kMhClearUnits - 1) / kMhClearUnits * 16;
static_assert(kMhH1Off + kMhH1Bytes == kMhPoolOff && kMhPoolOff + kMhPoolBytes == kMhLogitOff && kXpClearSpan % 16 == 0
        && kXpClearUnitBytes * kMhClearUnits >= kXpClearSpan && kXpClearUnitBytes * (kMhClearUnits - 1) < kXpClearSpan,
    "xp: the hop-1, pool and logit areas are one range, covered once by the clear units");
#endif
// MhClear: bytes [k U, (k + 1) U) of (the arena copy, then the pool + logit areas of the multicast copy) of the copy
// (n + 2) % 3 = (n - 1) % 3 (see the ownership note above), evict-first. MSO_MK_XP bit 256: of (the arena copy, then
// the hop-1 + pool + logit areas of the multicast copy), the hop-1 slots' readers (launch n - 1's MhH1 units) having
// completed before this launch's dependency wait and their next writers (launch n + C - 1's hop-1 senders on any
// rank) starting only after this launch completed on this rank, as for the pool / logit areas.
__device__ __noinline__ void stageMhClear(Ctx const& ctx, int unit)
{
    int const prev = MSO_MK_COPY_PREV(ctx.copy, ctx.numCopies);
    uint64_t const policy = policyEvictFirst();
    uint8_t* const ar = mhArena(ctx, prev);
#if MSO_MK_XP & 256
    uint8_t* const f2 = mhUc(ctx, prev) + kMhH1Off;
    int64_t const endU = static_cast<int64_t>(unit + 1) * kXpClearUnitBytes;
    int64_t const end = endU < kXpClearSpan ? endU : kXpClearSpan;
    for (int64_t off = static_cast<int64_t>(unit) * kXpClearUnitBytes + 16 * threadIdx.x; off < end;
         off += 16 * kThreads)
#else
    uint8_t* const f2 = mhUc(ctx, prev) + kMhPoolOff;
    int64_t const end = static_cast<int64_t>(unit + 1) * kMhClearUnitBytes;
    for (int64_t off = static_cast<int64_t>(unit) * kMhClearUnitBytes + 16 * threadIdx.x; off < end;
         off += 16 * kThreads)
#endif
    {
        uint8_t* p = off < kMhArenaBytes ? ar + off : f2 + (off - kMhArenaBytes);
        st128Hint(p, emptyVec(), policy);
    }
#if MSO_MK_XP & 128
    // ---- MSO_MK_XP bit 128: units [0, kXpH1ClearUnits) also empty slice `unit` of the previous copy's hop-1 area,
    // whose MhH1 readers do not empty their slots under this bit. Its readers (launch n - 1's MhH1 units) completed
    // before this launch's dependency wait; its next writers (launch n + C - 1's hop-1 senders on any rank) start only
    // after this launch completed on this rank (the pool / logit area rule above). ----
    if (unit < kXpH1ClearUnits)
    {
        uint8_t* const h1 = mhUc(ctx, prev) + kMhH1Off + static_cast<int64_t>(unit) * kXpH1ClearUnitBytes;
#pragma unroll 4
        for (int off = 16 * static_cast<int>(threadIdx.x); off < kXpH1ClearUnitBytes; off += 16 * kThreads)
        {
            st128Hint(h1 + off, emptyVec(), policy);
        }
    }
    // ---- end MSO_MK_XP bit 128 ----
#endif
}

// The stage switch of this file (runUnit's default case); false: not a stage of this file.
__device__ __forceinline__ bool mhRunUnit(Ctx const& c, int stage, int unit, uint32_t off, uint32_t bar, Io const& io)
{
    switch (stage)
    {
#if !MSO_MK_LD_DS // a stub stage (tests only)
    case kStMhOproj: stageMhOproj(c, unit); return true;
#endif
    case kStMhH1: stageMhH1(c, unit, off, bar, io); return true;
#if MSO_MK_LD_DS // no rank-local router units in the served plans
    case kStMhRouter:
#if MSO_MK_RL & 1
        if (kRlOk && (c.p[kPIFlags] & kIFlagRlOn)) // MSO_MK_RL: the rl unit in the router unit's slot
        {
            stageRlLogit(c, unit, off, bar, io);
            return true;
        }
#endif
        if (unit & kMhRouterLocalBit)
        {
            __trap();
        }
        stageMhRouter(c, unit, off, bar);
        return true;
#else
    case kStMhRouter:                              // and the rank-local units
#if MSO_MK_RL & 1
        if (kRlOk && (c.p[kPIFlags] & kIFlagRlOn)) // MSO_MK_RL
        {
            stageRlLogit(c, unit, off, bar, io);
        }
        else
#endif
            if (unit & kMhRouterLocalBit)
        {
            stageMhRouterLocal(c, unit, off, bar);
        }
        else
        {
            stageMhRouter(c, unit, off, bar);
        }
        return true;
#endif
    case kStMhPool: stageMhPool(c, unit, off, bar); return true;
#if !MSO_MK_LD_DS // a stub stage (tests only)
    case kStMhExpStub: stageMhExpStub(c, unit); return true;
#endif
    case kStMhFinal: stageMhFinal(c, unit); return true;
    case kStMhUnpool: stageMhUnpool(c, unit, off, bar); return true;
    case kStMhF1: stageMhF1(c, unit, off, bar, io); return true;
    case kStMhClear: stageMhClear(c, unit); return true;
    default: return false;
    }
}
