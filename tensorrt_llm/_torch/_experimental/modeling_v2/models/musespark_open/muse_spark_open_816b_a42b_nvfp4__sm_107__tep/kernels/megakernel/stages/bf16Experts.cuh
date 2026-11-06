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

// BF16 MoE experts stage of the megakernel (namespace z1): the routed experts of the two unquantized MoE layers (0 and
// 61) on this rank's two-window stack [own | r - 1] (ring_buddy_assign's placement), from the router logits and the
// pooled input of the 8 rows to this rank's [8, 4096] BF16 partial of the expert output. Included inside namespace
// mso_mk and the device-code guard after stages/experts.cuh, whose helpers it uses (namespace p4a).
//
// The stage is the NVFP4 stage's skeleton (one unit per CTA, the same warp roles and cross-CTA contracts) with:
//
//   1. routing: p4a::route unchanged (the stock BF16 layers route with the same _full_routing_kernel);
//   2. placement: two windows (ring_buddy_assign: the smallest feasible maximum over arcs of n blocks on n + 1
//      ranks, the pointwise smallest flow, the x[b] present experts with the highest local index move to b + 1);
//      local ids own -> i, predecessor's -> 32 + i;
//   3. the pre-expert norm: flashinfer's Gemma RMSNorm into BF16 (x rstd (1 + w), FP32), spread over the CTAs as the
//      NVFP4 stage spreads its quantization (each CTA normalizes its 64-column groups and publishes them as Lamport
//      words), then every CTA gathers the 8 BF16 rows into the 64 KiB B area (SW128, [64 k-blocks][8 rows][128 B]);
//   4. the weight stream: a unit is one BlockMajorK slab (128 rows x 64 K of one expert, 16 KiB contiguous), one TMA
//      box through a tensor map over the stock stack seen as [64 experts * 64 k-blocks * R rows][128 B]; the slots'
//      units are cut over the CTAs as in the NVFP4 stage (E * 4096 FC1 units, E * 2048 FC2 units);
//   5. MMAs: tcgen05.mma kind::f16 (BF16 x BF16 -> FP32), M = 128 weight rows, N = 8 tokens, K = 16, four per unit;
//      FC1 tile t of a slot (128 gate/up rows) is exactly h k-block t of its FC2 (64 intermediate columns);
//   6. FC1 epilogue: SwiGLU (silu(gate) up, FP32) rounded to BF16, 1 KiB of h per tile into the workspace; the h
//      loader moves each k-block of the CTA's FC2 slots into the B area once the CTA's FC1 MMAs have completed (the B
//      area held x until then: a tcgen05.commit barrier);
//   7. FC2 epilogue: y = bf16(acc) and the per-tile sums of (y (1 + w_post))^2 in the NVFP4 stage's layout, so
//      p4a::finalUnit computes the weighted post-norm sum unchanged.
//
// Synchronization as in the NVFP4 stage: cross-CTA data is data-as-flag (kEmpty words), polled with all loads in
// flight and an __all_sync exit; the workspace has C copies, launch n uses copy n % C and clears copy (n + 1) % C; the
// split-tile partial slots are emptied by their single reader.

namespace z1
{
using namespace p4a;

// ---- PTX ------------------------------------------------------------------------------------------------------------
// kind::f16 instruction descriptor: D FP32 (bit 4), A and B BF16 (bits 7, 10), both K-major, N >> 3 at 17, M >> 4
// at 24.
constexpr uint32_t kIdescBf16
    = (1u << 4) | (1u << 7) | (1u << 10) | ((static_cast<uint32_t>(kBfN) >> 3) << 17) | ((128u >> 4) << 24);
// The B descriptor: N = 8 is one 8-row SW128 group; N = 16 reads the same group twice (stride 0 between groups).
constexpr uint64_t kDescBBf = kBfN == 8 ? kDescSw128 : ((1ull << 16) | (1ull << 46) | (2ull << 61));
constexpr uint32_t kDescBBfHi = static_cast<uint32_t>(kDescBBf >> 32);

// D (+)= A B for one K step of 16 (32 bytes of each SW128 row); a / b are the low descriptor words.
__device__ __forceinline__ void mmaBf16Lo(uint32_t d, uint32_t aLo, uint32_t bLo, uint32_t acc)
{
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        ".reg .b64 da, db;\n"
        "mov.b64 da, {%1, %3};\n"
        "mov.b64 db, {%2, %4};\n"
        "setp.ne.b32 p, %6, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], da, db, %5, p;\n"
        "}\n" ::"r"(d),
        "r"(aLo), "r"(bLo), "r"(kDescSw128Hi), "r"(kDescBBfHi), "r"(kIdescBf16), "r"(acc)
        : "memory");
}

// One box of a 3-D tensor map ([planes][rows][128 B]: x = byte, y = row, z = k-block plane) into shared memory.
__device__ __forceinline__ void tmaLoad3d(
    uint32_t dst, void const* tmap, int x, int y, int z, uint32_t bar, uint64_t policy)
{
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(dst),
        "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(bar), "l"(policy)
        : "memory");
}

__device__ __forceinline__ uint32_t bbar(Sm const& s, int i)
{
    return s.redS + kBfRdBars + 8 * i;
}

// ---- the cut (identical in every CTA of the rank): p4a's with 64 k-blocks per tile ----------------------------------
struct BCut
{
    int E;
    bool tileCut;
    int a[2], b[2];
    int U[2];
};

__device__ __forceinline__ int bListStart(BCut const& c, int fc, int cta)
{
    if (c.tileCut)
    {
        int const tiles = c.U[fc] / kBfUpt;
        return static_cast<int>((static_cast<int64_t>(tiles) * cta) / kCtas) * kBfUpt;
    }
    return static_cast<int>((static_cast<int64_t>(c.U[fc]) * cta) / kCtas);
}

__device__ __forceinline__ int bCtaOfUnit(BCut const& c, int fc, int u)
{
    if (c.tileCut)
    {
        int const tiles = c.U[fc] / kBfUpt;
        int const t = u / kBfUpt;
        return static_cast<int>((static_cast<int64_t>(kCtas) * (t + 1) - 1) / tiles);
    }
    return static_cast<int>((static_cast<int64_t>(kCtas) * (u + 1) - 1) / c.U[fc]);
}

__device__ __forceinline__ void bMakeCut(BCut& c, int E, bool tileCut, int cta)
{
    c.E = E;
    c.tileCut = tileCut;
    c.U[0] = E * kBfFc1Units;
    c.U[1] = E * kBfFc2Units;
#pragma unroll
    for (int fc = 0; fc < 2; ++fc)
    {
        c.a[fc] = c.U[fc] == 0 ? 0 : bListStart(c, fc, cta);
        c.b[fc] = c.U[fc] == 0 ? 0 : bListStart(c, fc, cta + 1);
    }
}

__device__ __forceinline__ int bUnitsOf(BCut const& c)
{
    return (c.b[0] - c.a[0]) + (c.b[1] - c.a[1]);
}

// Unit j of the CTA's part of list fc: the part's tiles from the last to the first, k-blocks in order (p4a::listUnit).
__device__ __forceinline__ int bListUnit(BCut const& c, int fc, int j)
{
    int const a = c.a[fc];
    int const lastLo = ((c.b[fc] - 1) / kBfUpt) * kBfUpt;
    int const lastFrom = lastLo > a ? lastLo : a;
    int const nLast = c.b[fc] - lastFrom;
    if (j < nLast)
    {
        return lastFrom + j;
    }
    int const jj = j - nLast;
    int const lo = lastLo - kBfUpt * (1 + jj / kBfUpt);
    return (lo > a ? lo : a) + jj % kBfUpt;
}

__device__ __forceinline__ void bUnitAt(BCut const& c, int j, int& fc, int& u)
{
    int const n1 = c.b[0] - c.a[0];
    fc = j < n1 ? 0 : 1;
    u = bListUnit(c, fc, j < n1 ? j : j - n1);
}

__device__ __forceinline__ void bDecodeUnit(int fc, int u, int& slot, int& mt, int& kb)
{
    kb = u & (kBfUpt - 1);
    mt = fc == 0 ? (u / kBfUpt) & (kBfFc1Tiles - 1) : (u / kBfUpt) & (kBfFc2Tiles - 1);
    slot = fc == 0 ? u / kBfFc1Units : u / kBfFc2Units;
}

// ---- 2. placement (ring_buddy_assign, two windows) and the slot list
// ------------------------------------------------------ Called by every thread after route() and a barrier; leaves E
// in the scalars, the slot tables and tokSlot.
__device__ __noinline__ void place2(Ctx const& ctx, Sm const s)
{
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const rank = static_cast<int>(dw(s, kMdRank));
    uint32_t* pres = rd<uint32_t>(s, kMoeRdPres); // [0, 8) block presence, [8, 10) local presence
    int* xs = rd<int>(s, kMoeRdMoved);            // x[b]: experts of block b that run on b + 1
    int const* sel = rd<int const>(s, kMoeRdSel);
    if (tid < kTok * kMoeTopK)
    {
        int const e = sel[tid];
        atomicOr(pres + (e >> 5), 1u << (e & 31));
    }
    __syncthreads();
    if (tid < 32)
    {
        int const load = lane < kMoeRanks ? __popc(pres[lane]) : 0;
        int const D = static_cast<int>(__reduce_add_sync(0xffffffffu, static_cast<uint32_t>(load)));
        // The smallest feasible maximum: ceil(D / 8) and every arc of n blocks over its n + 1 ranks.
        int tr = 0;
        int arc = 0;
#pragma unroll
        for (int n = 1; n < kMoeRanks; ++n)
        {
            arc += __shfl_sync(0xffffffffu, load, (lane - n + 1) & 7);
            tr = max(tr, (arc + n) / (n + 1));
        }
        int const T = max((D + kMoeRanks - 1) / kMoeRanks,
            static_cast<int>(__reduce_max_sync(0xffffffffu, static_cast<uint32_t>(lane < kMoeRanks ? tr : 0))));
        // x[r]: the pointwise smallest flow r -> r + 1 of maximum T.
        int c = 0;
        arc = 0;
#pragma unroll
        for (int n = 1; n <= kMoeRanks; ++n)
        {
            arc += __shfl_sync(0xffffffffu, load, (lane - n + 1) & 7);
            c = max(c, arc - n * T);
        }
        int const prev = __shfl_sync(0xffffffffu, c, (lane - 1) & 7);
        if (lane < kMoeRanks)
        {
            xs[lane] = c;
            uint8_t* dbg = dptr<uint8_t>(s, kMdDbg);
            if (dbg != nullptr && blockIdx.x == 0)
            {
                reinterpret_cast<int*>(dbg + kMoeDbgLoads)[lane] = load;
                reinterpret_cast<int*>(dbg + kMoeDbgLoads)[kMoeRanks + lane] = load - c + prev;
            }
        }
    }
    __syncthreads();
    int* local = rd<int>(s, kMoeRdLocal);
    if (tid < kTok * kMoeTopK)
    {
        int const e = sel[tid];
        int const b = e >> 5;
        int const i = e & 31;
        int const above = __popc(static_cast<uint32_t>(static_cast<uint64_t>(pres[b]) >> (i + 1)));
        int const runsOn = above < xs[b] ? ((b + 1) & 7) : b;
        int const id = runsOn == rank ? ((rank - b) & 7) * kMoeBlock + i : kBfWindow + e % (kMoeExperts - kBfWindow);
        local[tid] = id;
        if (id < kBfWindow)
        {
            atomicOr(pres + 8 + (id >> 5), 1u << (id & 31));
        }
        uint8_t* dbg = dptr<uint8_t>(s, kMdDbg);
        if (dbg != nullptr && blockIdx.x == 0)
        {
            reinterpret_cast<int*>(dbg + kMoeDbgLocal)[tid] = id;
        }
    }
    __syncthreads();
    uint32_t const lp0 = pres[8], lp1 = pres[9];
    auto prefix = [&](int id)
    {
        uint32_t const below = (1u << (id & 31)) - 1u;
        return (id >= 32 ? __popc(lp0) : 0) + __popc((id < 32 ? lp0 : lp1) & below);
    };
    int const E = __popc(lp0) + __popc(lp1);
    int* slotExpert = rd<int>(s, kMoeRdSlotExpert);
    uint32_t* slotMask = rd<uint32_t>(s, kMoeRdSlotMask);
    int8_t* slotK = rd<int8_t>(s, kMoeRdSlotK);
    int8_t* tokSlot = rd<int8_t>(s, kMoeRdTokSlot);
    if (tid < kBfWindow)
    {
        uint32_t const w = tid < 32 ? lp0 : lp1;
        if ((w >> (tid & 31)) & 1u)
        {
            slotExpert[prefix(tid)] = tid;
        }
    }
    if (tid < kTok * kMoeTopK)
    {
        int const id = local[tid];
        int const t = tid >> 3;
        int const k = tid & 7;
        if (id < kBfWindow)
        {
            int const slot = prefix(id);
            tokSlot[tid] = static_cast<int8_t>(slot);
            atomicOr(slotMask + slot, 1u << t);
            slotK[slot * kTok + t] = static_cast<int8_t>(k);
        }
        else
        {
            tokSlot[tid] = -1;
        }
    }
    if (tid == 0)
    {
        rd<int>(s, kMoeRdScal)[kMoeScE] = E;
    }
    __syncthreads();
    uint8_t* dbg = dptr<uint8_t>(s, kMdDbg);
    if (dbg != nullptr && blockIdx.x == 0 && tid <= kMoeMaxSlots)
    {
        reinterpret_cast<int*>(dbg + kMoeDbgSlots)[tid] = tid == 0 ? E : (tid - 1 < E ? slotExpert[tid - 1] : -1);
    }
}

// ---- 3. the pre-expert Gemma RMSNorm, spread over the CTAs
// ------------------------------------------------------------ The 512 (row, 64-column) groups are split over the CTAs
// as the NVFP4 stage splits its quantization (p4a::shareRange: at most two rows per CTA). Thread 32 w + l holds columns
// [16 b, 16 b + 16) of each of the CTA's rows (b = 32 w + l): the row's sum of squares in the block-per-thread order of
// p4a::shareQuant, rstd = rsqrt(ss / 4096 + eps), then for the CTA's own groups bf16(x rstd (1 + w)) published as
// Lamport words.
__device__ __noinline__ void shareNorm(Ctx const& ctx, Sm const s, uint8_t* wsCur)
{
    int const t = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
    int const b = 32 * t + lane;
    int lo, hi, r0, nr;
    shareRange(lo, hi, r0, nr);
    uint8_t const* pool = dptr<uint8_t const>(s, kMdPool);
    float* red = rd<float>(s, kMoeRdRed);
    uint4 x0 = make_uint4(0, 0, 0, 0), x1 = make_uint4(0, 0, 0, 0);
    int myRow = -1;
#pragma unroll 1
    for (int rr = 0; rr < nr; ++rr)
    {
        int const row = r0 + rr;
        uint4 v[2];
#if MSO_MK_LANES & 8
        if (dw(s, kMdFlags) & kMoeFlagMhF2)
        {
            // The pooled row from the F2' area's two FP32 K halves (bf16(h0 + h1), as the NVFP4 stage).
            uint8_t const* ap[8];
            uint8_t const* a4[4];
            mhPoolPtrs(ctx, a4, row, b * 16);
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                ap[i] = a4[i];
            }
            mhPoolPtrs(ctx, a4, row, b * 16 + 8);
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                ap[4 + i] = a4[i];
            }
            uint4 pv[8];
            pollVecs<8>(ctx, pv, ap, kCodeBfPool);
            uint4 const q0[4] = {pv[0], pv[1], pv[2], pv[3]};
            uint4 const q1[4] = {pv[4], pv[5], pv[6], pv[7]};
            v[0] = mhPoolValues(q0);
            v[1] = mhPoolValues(q1);
        }
        else
#endif
        {
            uint8_t const* a[2] = {pool + row * (kMoeHidden * 2) + b * 32, pool + row * (kMoeHidden * 2) + b * 32 + 16};
            pollVecs<2>(ctx, v, a, kCodeBfPool);
        }
        uint32_t const w[8] = {v[0].x, v[0].y, v[0].z, v[0].w, v[1].x, v[1].y, v[1].z, v[1].w};
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            float2 const x2 = make_float2(bf16Lo(w[i]), bf16Hi(w[i]));
            float2 const sq = mulRn2(x2, x2);
            ss = addRn(ss, sq.x);
            ss = addRn(ss, sq.y);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
        {
            ss = addRn(ss, __shfl_xor_sync(0xffffffffu, ss, o));
        }
        if (lane == 0)
        {
            red[rr * kWarps + t] = ss;
        }
        int const q = row * (kMoeHidden / 64) + (b >> 2);
        if (q >= lo && q < hi)
        {
            myRow = rr;
            x0 = v[0];
            x1 = v[1];
        }
    }
    __syncthreads();
    // The gain (and the warm-up boxes into ring stage 0) are in for every thread.
    waitBar(ctx, bbar(s, kBfBarGain), 0, kCodeBfBar);
    if (myRow >= 0)
    {
        float tot = 0.f;
#pragma unroll
        for (int j = 0; j < kWarps; ++j)
        {
            tot = addRn(tot, red[myRow * kWarps + j]);
        }
        float const eps = __uint_as_float(static_cast<uint32_t>(dw(s, kMdEps)));
        float const rstd = rsqrtFtz(addRn(mulRn(tot, 1.f / kMoeHidden), eps));
        uint4 const g0 = p4a::lds128(s.base + kBfSmGain + b * 32);
        uint4 const g1 = p4a::lds128(s.base + kBfSmGain + b * 32 + 16);
        uint32_t const xw[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
        uint32_t const gw[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
        uint32_t o[8];
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            float const lo0 = mulRn(mulRn(bf16Lo(xw[i]), rstd), addRn(1.f, bf16Lo(gw[i])));
            float const hi0 = mulRn(mulRn(bf16Hi(xw[i]), rstd), addRn(1.f, bf16Hi(gw[i])));
            o[i] = bf16Bits(lo0) | (bf16Bits(hi0) << 16);
        }
        uint8_t* dst = wsCur + kBfWsX + static_cast<size_t>(r0 + myRow) * (kMoeHidden * 2) + b * 32;
        st128(dst, lamport4(make_uint4(o[0], o[1], o[2], o[3])));
        st128(dst + 16, lamport4(make_uint4(o[4], o[5], o[6], o[7])));
    }
    if (threadIdx.x == 0)
    {
        moeStamp(ctx, kMoeStRstd);
    }
}

// ---- 4. the weight stream
// ----------------------------------------------------------------------------------------------
__device__ __forceinline__ void issueUnitB(Sm const& s, BCut const& c, int j, uint64_t policy)
{
    int fc, u, slot, mt, kb;
    bUnitAt(c, j, fc, u);
    bDecodeUnit(fc, u, slot, mt, kb);
    int const st = j % kBfStages;
    uint32_t const full = bbar(s, kBfBarFull + st);
    uint32_t const dst = s.base + kBfSmRing + st * kBfUnitBytes;
    int const expert = rd<int const>(s, kMoeRdSlotExpert)[slot];
    void const* tmap = reinterpret_cast<void const*>(dw(s, fc == 0 ? kMdTmapW13 : kMdTmapW2));
    mbarExpectTx(full, kBfUnitBytes);
    tmaLoad3d(dst, tmap, 0, mt * 128, expert * kBfKb + kb * kBfUkb, full, policy);
}

// Every CTA: lane 0 of warp t < nEarly requests ring unit t once the warp's x loads are out, then the 8 normalized rows
// are polled from the workspace into the SW128 B area (two rounds of 8 vectors per thread).
__device__ __noinline__ void gatherXB(Ctx const& ctx, Sm const s, BCut const c, int nEarly, uint8_t const* wsCur)
{
    int const t = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
    constexpr int kV = kBfWsXBytes / 16; // 4096 vectors
    constexpr int kRound = 8;
    static_assert(kV == 2 * kRound * kThreads, "two rounds of 8 vectors per thread");
    uint8_t* dbg = dptr<uint8_t>(s, kMdDbg);
    bool const dbgOn = dbg != nullptr && blockIdx.x == 0;
#pragma unroll 1
    for (int round = 0; round < 2; ++round)
    {
        uint4 v[kRound];
        uint8_t const* a[kRound];
#pragma unroll
        for (int k = 0; k < kRound; ++k)
        {
            int const vi = threadIdx.x + kThreads * (kRound * round + k);
            a[k] = wsCur + kBfWsX + 16 * vi;
            v[k] = ldPoll128(a[k]);
        }
        if (round == 0)
        {
            if (lane == 0 && t < nEarly)
            {
                issueUnitB(s, c, t, policyEvictFirst());
            }
            if (threadIdx.x == 0)
            {
                moeStamp(ctx, kMoeStEarly);
            }
        }
        uint64_t t0 = 0;
        while (true)
        {
            bool ok = true;
#pragma unroll
            for (int k = 0; k < kRound; ++k)
            {
                ok &= full4(v[k]);
            }
            if (warpAll(ok))
            {
                break;
            }
            spinCheck(ctx, t0, kCodeBfX, round, 0);
#pragma unroll
            for (int k = 0; k < kRound; ++k)
            {
                if (!full4(v[k]))
                {
                    v[k] = ldPoll128(a[k]);
                }
            }
        }
#pragma unroll
        for (int k = 0; k < kRound; ++k)
        {
            int const vi = threadIdx.x + kThreads * (kRound * round + k);
            int const r = vi >> 9;
            int const ci = vi & 511;
            int const kb = ci >> 3;
            int const c16 = ci & 7;
            p4a::sts128(s.base + kBfSmB + kb * 1024 + r * 128 + ((c16 ^ r) << 4), v[k]);
            if (dbgOn)
            {
                *reinterpret_cast<uint4*>(dbg + kBfDbgX + 16 * vi) = v[k];
            }
        }
    }
    if (threadIdx.x == 0)
    {
        moeStamp(ctx, kMoeStPool);
    }
}

// ---- 5a. MMA issue (warp 1)
// ------------------------------------------------------------------------------------------
__device__ __noinline__ void mmaLoopB(Ctx const& ctx, Sm const s, BCut const c, uint32_t tmem)
{
    int const n = bUnitsOf(c);
    int const n1 = c.b[0] - c.a[0];
    tcFenceAfter();
    bool const noMma = (dw(s, kMdFlags) & kMoeFlagDbgNoMma) != 0;
    int ti = -1;
    int prevKey = -1;
    int prevSlot2 = -1;
    int slotLoad = -1;
    uint64_t hDone = 0;
    bool first = false;
    if (n1 == 0 && (threadIdx.x & 31) == 0)
    {
        mbarArrive(bbar(s, kBfBarFc1Done));
    }
#pragma unroll 1
    for (int j = 0; j < n; ++j)
    {
        int fc, u, slot, mt, kb;
        bUnitAt(c, j, fc, u);
        bDecodeUnit(fc, u, slot, mt, kb);
        int const key = (fc << 20) | (slot << 8) | mt;
        if (key != prevKey)
        {
            ++ti;
            prevKey = key;
            first = true;
            if (ti >= kBfAccs)
            {
                waitBar(ctx, bbar(s, kBfBarAccEmpty + (ti & (kBfAccs - 1))), ((ti / kBfAccs) - 1) & 1, kCodeBfBar);
                tcFenceAfter();
            }
        }
        if (fc == 1)
        {
            if (slot != prevSlot2)
            {
                if (prevSlot2 >= 0)
                {
                    if (electOne())
                    {
                        commit(bbar(s, kBfBarHFree));
                    }
                    __syncwarp();
                }
                ++slotLoad;
                hDone = 0;
                prevSlot2 = slot;
            }
#pragma unroll
            for (int i = 0; i < kBfUkb; ++i)
            {
                int const hk = kb * kBfUkb + i;
                if (!((hDone >> hk) & 1ull))
                {
                    waitBar(ctx, bbar(s, kBfBarHReady + hk), slotLoad & 1, kCodeBfBar);
                    if (j == n1 && slotLoad == 0 && (threadIdx.x & 31) == 0)
                    {
                        moeStamp(ctx, kMoeStFirstH);
                    }
                    tcFenceAfter();
                    hDone |= 1ull << hk;
                }
            }
        }
        int const st = j % kBfStages;
        waitBar(ctx, bbar(s, kBfBarFull + st), (j / kBfStages) & 1, kCodeBfBar);
        bool const lastOfTile = kb == kBfUpt - 1 || u == c.b[fc] - 1;
        if ((threadIdx.x & 31) == 0)
        {
            if (j == 0)
            {
                moeStamp(ctx, kMoeStFirstFull);
            }
            if (j == n1 - 1)
            {
                moeStamp(ctx, kMoeStFc1LastFull);
            }
            if (j == n - 1)
            {
                moeStamp(ctx, kMoeStLastFull);
            }
        }
        tcFenceAfter();
        int const acc = ti & (kBfAccs - 1);
        if (electOne())
        {
            if (!noMma)
            {
                uint32_t const d = tmem + static_cast<uint32_t>(acc * kBfN);
#pragma unroll
                for (int i = 0; i < kBfUkb; ++i)
                {
                    uint32_t const aLo = kDescSw128Lo | descLo(s.base + kBfSmRing + st * kBfUnitBytes + i * 16384);
                    uint32_t const bLo = kDescSw128Lo | descLo(s.base + kBfSmB + (kb * kBfUkb + i) * 1024);
#pragma unroll
                    for (int ks = 0; ks < 4; ++ks)
                    {
                        mmaBf16Lo(d, aLo + 2 * ks, bLo + 2 * ks, (first && i == 0 && ks == 0) ? 0u : 1u);
                    }
                }
            }
            commit(bbar(s, kBfBarEmpty + st));
            if (lastOfTile)
            {
                commit(bbar(s, kBfBarAccFull + acc));
            }
            if (j == n1 - 1)
            {
                // The B area may take h once these MMAs (the CTA's last reads of x) have completed.
                commit(bbar(s, kBfBarFc1Done));
            }
        }
        __syncwarp();
        first = false;
        if (j == n1 - 1 && (threadIdx.x & 31) == 0)
        {
            moeStamp(ctx, kMoeStFc1Issued);
        }
        if (j == n1 && (threadIdx.x & 31) == 0)
        {
            moeStamp(ctx, kMoeStFc2First);
        }
    }
    if ((threadIdx.x & 31) == 0)
    {
        moeStamp(ctx, kMoeStMmaDone);
    }
}

// ---- 5b. the weight stream after the initial requests (warp 0 lane 0)
// -----------------------------------------------------
__device__ __noinline__ void produceLoopB(Ctx const& ctx, Sm const s, BCut const c, int from)
{
    uint64_t const policy = policyEvictFirst();
    int const n = bUnitsOf(c);
    int const n1 = c.b[0] - c.a[0];
    // kMoeFlagFc2TailL2: the FC2 units past the ring (their stages are freed only once h has arrived) into L2 first.
    bool const tailL2 = (dw(s, kMdFlags) & kMoeFlagFc2TailL2) != 0;
#pragma unroll 1
    for (int j = from; j < n; ++j)
    {
        int const st = j % kBfStages;
        if (tailL2 && j == n1 + kBfStages)
        {
            uint64_t const pf = policyEvictNormal();
#pragma unroll 1
            for (int jj = j; jj < n; ++jj)
            {
                int fc, u, slot, mt, kb;
                bUnitAt(c, jj, fc, u);
                bDecodeUnit(fc, u, slot, mt, kb);
                int const expert = rd<int const>(s, kMoeRdSlotExpert)[slot];
                void const* tmap = reinterpret_cast<void const*>(dw(s, kMdTmapW2));
                asm volatile(
                    "cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::cache_hint [%0, {%1, %2, %3}], %4;" ::"l"(
                        tmap),
                    "r"(0), "r"(mt * 128), "r"(expert * kBfKb + kb * kBfUkb), "l"(pf)
                    : "memory");
            }
        }
        if (j >= kBfStages)
        {
            waitBar(ctx, bbar(s, kBfBarEmpty + st), ((j / kBfStages) - 1) & 1, kCodeBfBar);
        }
        issueUnitB(s, c, j, policy);
        if (j == n - 1)
        {
            moeStamp(ctx, kMoeStPreIssue);
        }
    }
}

// ---- 5c. warps 2-3: clear the next copy's workspace slice, then fetch h of the CTA's FC2 slots into the B area
// ----------
__device__ __noinline__ void loaderLoopB(
    Ctx const& ctx, Sm const s, BCut const c, uint8_t const* wsCur, uint8_t* wsNext, uint8_t* nvNext)
{
    int const t64 = threadIdx.x - 64;
    {
        constexpr int kGranules = kBfWsBytes / 16;
        int const g0 = static_cast<int>((static_cast<int64_t>(kGranules) * blockIdx.x) / kCtas);
        int const g1 = static_cast<int>((static_cast<int64_t>(kGranules) * (blockIdx.x + 1)) / kCtas);
        uint64_t const policy = policyEvictFirst();
        for (int g = g0 + t64; g < g1; g += 64)
        {
            st128Hint(wsNext + 16 * static_cast<size_t>(g), emptyVec(), policy);
        }
        // In a forward that mixes the two stages, the NVFP4 stage's next workspace copy as well (that stage clears
        // copy n + 1 in every launch it runs; this launch stands in for it).
        if (nvNext != nullptr)
        {
            constexpr int kNvGranules = kMoeWsBytes / 16;
            int const h0 = static_cast<int>((static_cast<int64_t>(kNvGranules) * blockIdx.x) / kCtas);
            int const h1 = static_cast<int>((static_cast<int64_t>(kNvGranules) * (blockIdx.x + 1)) / kCtas);
            for (int g = h0 + t64; g < h1; g += 64)
            {
                st128Hint(nvNext + 16 * static_cast<size_t>(g), emptyVec(), policy);
            }
        }
    }
    int const n1 = c.b[0] - c.a[0];
    int const n = bUnitsOf(c);
    int const lane = threadIdx.x & 31;
    int const r = t64 >> 3;
    int const c16 = t64 & 7;
    if (n > n1)
    {
        // The B area holds x until the CTA's FC1 MMAs have completed.
        waitBarIdle(ctx, bbar(s, kBfBarFc1Done), 0, kCodeBfBar, (dw(s, kMdFlags) & kMoeFlagSpinSleep) ? 256u : 0u);
    }
    int si = -1;
    int j = n1;
#pragma unroll 1
    while (j < n)
    {
        int fc, u, slot, mt, kb;
        bUnitAt(c, j, fc, u);
        bDecodeUnit(fc, u, slot, mt, kb);
        ++si;
        if (si > 0)
        {
            // The previous slot's MMAs have read its operand.
            waitBar(ctx, bbar(s, kBfBarHFree), (si - 1) & 1, kCodeBfBar);
        }
        uint64_t need = 0;
        int jEnd = j;
#pragma unroll 1
        for (; jEnd < n; ++jEnd)
        {
            int f2, u2, sl2, mt2, kb2;
            bUnitAt(c, jEnd, f2, u2);
            bDecodeUnit(f2, u2, sl2, mt2, kb2);
            if (sl2 != slot)
            {
                break;
            }
            need |= ((1ull << kBfUkb) - 1ull) << (kb2 * kBfUkb);
        }
        // The k-blocks this slot never uses complete their phase now (one arrival per loader warp).
#pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            int const k = lane + 32 * h;
            if (!((need >> k) & 1ull))
            {
                mbarArrive(bbar(s, kBfBarHReady + k));
            }
        }
        uint8_t const* hd = wsCur + kBfWsH + (static_cast<size_t>(slot) * kBfKb * kTok) * 128 + 16 * t64;
        uint64_t left = need;
#pragma unroll 1
        while (left != 0)
        {
            int kbs[4];
            uint4 v[4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                kbs[i] = left != 0 ? __ffsll(static_cast<long long>(left)) - 1 : -1;
                if (kbs[i] >= 0)
                {
                    left &= left - 1;
                }
                v[i] = kbs[i] >= 0 ? ldPoll128(hd + kbs[i] * (kTok * 128)) : make_uint4(0, 0, 0, 0);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                if (kbs[i] < 0)
                {
                    break;
                }
                int const k = kbs[i];
                uint64_t t0 = 0;
                while (!warpAll(full4(v[i])))
                {
                    spinCheck(ctx, t0, kCodeBfH, k, 0);
                    if (!full4(v[i]))
                    {
                        v[i] = ldPoll128(hd + k * (kTok * 128));
                    }
                }
                p4a::sts128(s.base + kBfSmB + k * 1024 + r * 128 + ((c16 ^ r) << 4), v[i]);
                fenceProxyAsyncSmem();
                __syncwarp();
                if (lane == 0)
                {
                    mbarArrive(bbar(s, kBfBarHReady + k));
                }
            }
        }
        j = jEnd;
    }
    if (t64 == 0 && n > n1)
    {
        moeStamp(ctx, kMoeStHAll);
    }
}

// ---- 5d. the epilogue (warps 4-7: TMEM lane quadrant q = warp - 4, row r = 32 q + lane)
// ------------------------------------ FC1 epilogue of a finished tile mt (128 gate/up rows = intermediate columns [64
// mt, 64 mt + 64) = h k-block mt): lane l < 8 / 16 <= l < 24 holds up rows of columns 16 q + 2 l / 16 q + 2 (l - 16) +
// 1, the partner gate row at l + 8; h = bf16(up silu(gate)) (FP32: gate / (1 + 2^(-gate log2 e)) with the approximate
// exp2 and reciprocal), word 8 q + l of the token's 32 = (h[l], h[l + 16]).
__device__ __forceinline__ void fc1EpilogueB(Sm const& s, float const (&v)[8], int slot, int mt, uint8_t* wsCur)
{
    int const q = (threadIdx.x >> 5) - 4;
    int const lane = threadIdx.x & 31;
    int const e = threadIdx.x - 128;
    uint32_t const mask = rd<uint32_t const>(s, kMoeRdSlotMask)[slot];
    uint32_t* hst = rd<uint32_t>(s, kBfRdEpi);
#pragma unroll
    for (int nTok = 0; nTok < kTok; ++nTok)
    {
        float const g = __shfl_down_sync(0xffffffffu, v[nTok], 8);
        float const x = mulRn(g, -kLog2e);
        float const d = addRn(ex2Ftz(x), 1.f);
        float const gsig = mulRn(g, rcpFtz(d));
        float const h = mulRn(v[nTok], gsig);
        uint32_t const hb = bf16Bits(h);
        uint32_t const partner = __shfl_down_sync(0xffffffffu, hb, 16);
        uint32_t word = hb | (partner << 16);
        if (!((mask >> nTok) & 1u))
        {
            word = 0;
        }
        if (lane < 8)
        {
            hst[nTok * 32 + q * 8 + lane] = word;
        }
    }
    namedSync(1, 128);
    if (e < 64)
    {
        uint4 const w = *reinterpret_cast<uint4 const*>(hst + (e >> 3) * 32 + 4 * (e & 7));
        st128(wsCur + kBfWsH + ((static_cast<size_t>(slot) * kBfKb + mt) * kTok + (e >> 3)) * 128 + 16 * (e & 7),
            lamport4(w));
    }
    namedSync(1, 128);
}

// FC2 epilogue of a finished tile: rows y = bf16(acc) of the routed tokens and their per-tile sums of (y (1 +
// w_post))^2 (p4a::fc2Epilogue without the NVFP4 output scale).
__device__ __forceinline__ void fc2EpilogueB(Sm const& s, float const (&v)[8], int slot, int mt, uint8_t* wsCur)
{
    int const q = (threadIdx.x >> 5) - 4;
    int const lane = threadIdx.x & 31;
    int const e = threadIdx.x - 128;
    uint32_t const mask = rd<uint32_t const>(s, kMoeRdSlotMask)[slot];
    int8_t const* slotK = rd<int8_t const>(s, kMoeRdSlotK) + slot * kTok;
    int const col = mt * 128 + 32 * q + 4 * (lane & 7) + (lane >> 3);
    __nv_bfloat16 const* wPost = dptr<__nv_bfloat16 const>(s, kMdGainPost);
    float const gain = addRn(__bfloat162float(wPost[col]), 1.f);
    __nv_bfloat16* rowsDbg = dptr<__nv_bfloat16>(s, kMdRows);
    float* sq = rd<float>(s, kMoeRdEpiSq);
#pragma unroll
    for (int nTok = 0; nTok < kTok; ++nTok)
    {
        bool const routed = (mask >> nTok) & 1u;
        uint32_t const yb = bf16Bits(v[nTok]);
        int const tk = nTok * kMoeTopK + (routed ? slotK[nTok] : 0);
        uint32_t const y1 = __shfl_down_sync(0xffffffffu, yb, 8);
        uint32_t const y2 = __shfl_down_sync(0xffffffffu, yb, 16);
        uint32_t const y3 = __shfl_down_sync(0xffffffffu, yb, 24);
        if (routed)
        {
            if (lane < 8)
            {
                uint8_t* dst = wsCur + kBfWsY + (static_cast<size_t>(tk) * kMoeHidden + col) * 2;
                st64(dst, make_uint2(lamport(yb | (y1 << 16)), lamport(y2 | (y3 << 16))));
            }
            if (rowsDbg != nullptr)
            {
                reinterpret_cast<uint16_t*>(rowsDbg)[static_cast<size_t>(tk) * kMoeHidden + col]
                    = static_cast<uint16_t>(yb);
            }
        }
        float const rho = mulRn(__uint_as_float(yb << 16), gain);
        float ssq = routed ? mulRn(rho, rho) : 0.f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
        {
            ssq = addRn(ssq, __shfl_xor_sync(0xffffffffu, ssq, o));
        }
        if (lane == 0)
        {
            sq[q * kTok + nTok] = ssq;
        }
    }
    namedSync(1, 128);
    if (e < kTok && ((mask >> e) & 1u))
    {
        float const tot = addRn(addRn(addRn(sq[e], sq[kTok + e]), sq[2 * kTok + e]), sq[3 * kTok + e]);
        int const tk = e * kMoeTopK + slotK[e];
        st32(wsCur + kBfWsSsq + (static_cast<size_t>(tk) * kMoeFc2Tiles + mt) * 4, lamport(__float_as_uint(tot)));
    }
    namedSync(1, 128);
}

__device__ __noinline__ void epilogueLoopB(Ctx const& ctx, Sm const s, BCut const c, uint32_t tmem, uint8_t* wsCur)
{
    int const q = (threadIdx.x >> 5) - 4;
    int const lane = threadIdx.x & 31;
    int const r = 32 * q + lane;
    float* part = dptr<float>(s, kMdPart);
    uint32_t const idleNs = (dw(s, kMdFlags) & kMoeFlagSpinSleep) ? 128u : 0u;
    int ti = -1;
#pragma unroll 1
    for (int fc = 0; fc < 2; ++fc)
    {
        if (c.b[fc] <= c.a[fc])
        {
            continue;
        }
        int const g0 = c.a[fc] / kBfUpt;
        int const g1 = (c.b[fc] - 1) / kBfUpt;
#pragma unroll 1
        for (int g = g1; g >= g0; --g)
        {
            ++ti;
            int const acc = ti & (kBfAccs - 1);
            waitBarIdle(ctx, bbar(s, kBfBarAccFull + acc), (ti / kBfAccs) & 1, kCodeBfBar, idleNs);
            if (fc == 1 && g == g1 && threadIdx.x == 128)
            {
                moeStamp(ctx, kMoeStFc2EpiIn);
            }
            tcFenceAfter();
            float v[8];
            tmemLd8(tmem + (static_cast<uint32_t>(32 * q) << 16) + static_cast<uint32_t>(acc * kBfN), v);
            tcFenceBefore();
            __syncwarp();
            if (lane == 0)
            {
                mbarArrive(bbar(s, kBfBarAccEmpty + acc));
            }
            bool const finished = kBfUpt * g + kBfUpt - 1 < c.b[fc];
            if (!finished)
            {
                float* dst = part + ((static_cast<size_t>(blockIdx.x) * 2 + fc) * 128 + r) * kTok;
                st128(dst,
                    lamport4(make_uint4(
                        __float_as_uint(v[0]), __float_as_uint(v[1]), __float_as_uint(v[2]), __float_as_uint(v[3]))));
                st128(dst + 4,
                    lamport4(make_uint4(
                        __float_as_uint(v[4]), __float_as_uint(v[5]), __float_as_uint(v[6]), __float_as_uint(v[7]))));
                continue;
            }
            int const first = bCtaOfUnit(c, fc, kBfUpt * g);
            if (first < static_cast<int>(blockIdx.x))
            {
                float sum[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
                for (int cc = first; cc < static_cast<int>(blockIdx.x); ++cc)
                {
                    float const* src = part + ((static_cast<size_t>(cc) * 2 + fc) * 128 + r) * kTok;
                    uint4 pv[2];
                    uint8_t const* const pa[2]
                        = {reinterpret_cast<uint8_t const*>(src), reinterpret_cast<uint8_t const*>(src + 4)};
                    pollVecs<2>(ctx, pv, pa, kCodeBfPart);
                    uint32_t const w[8] = {pv[0].x, pv[0].y, pv[0].z, pv[0].w, pv[1].x, pv[1].y, pv[1].z, pv[1].w};
#pragma unroll
                    for (int i = 0; i < 8; ++i)
                    {
                        sum[i] = cc == first ? __uint_as_float(w[i]) : addRn(sum[i], __uint_as_float(w[i]));
                    }
                }
#pragma unroll
                for (int i = 0; i < 8; ++i)
                {
                    v[i] = addRn(sum[i], v[i]);
                }
            }
            int slot, mt, kb;
            bDecodeUnit(fc, kBfUpt * g, slot, mt, kb);
            if (fc == 0)
            {
                fc1EpilogueB(s, v, slot, mt, wsCur);
            }
            else
            {
                fc2EpilogueB(s, v, slot, mt, wsCur);
            }
            for (int cc = first; cc < static_cast<int>(blockIdx.x); ++cc)
            {
                float* src = part + ((static_cast<size_t>(cc) * 2 + fc) * 128 + r) * kTok;
                st128(src, emptyVec());
                st128(src + 4, emptyVec());
            }
        }
        if (fc == 0 && threadIdx.x == 128)
        {
            moeStamp(ctx, kMoeStFc1EpiDone);
        }
    }
    if (threadIdx.x == 128)
    {
        moeStamp(ctx, kMoeStEpiDone);
    }
}

} // namespace z1

// ---- the stage
// ---------------------------------------------------------------------------------------------------------
__device__ __noinline__ void stageMoeExpertsBf16(Ctx const& ctx, int unit, uint32_t wOff)
{
    using namespace z1;
    int const tid = threadIdx.x;
    int const warp = tid >> 5;
    Sm s;
    s.redS = ctx.smemBase + kSmemRed;
    s.base = ctx.smemBase + wOff;
    int64_t* sd = rd<int64_t>(s, kMoeRdDesc);
    int64_t const* desc = ptr<int64_t const>(ctx, kPMoe);
    if (tid < kMdNumWords)
    {
        sd[tid] = desc[tid];
    }
    uint32_t* pres = rd<uint32_t>(s, kMoeRdPres);
    if (tid < 16)
    {
        pres[tid] = 0;
    }
    if (tid >= 32 && tid < 32 + kMoeMaxSlots)
    {
        rd<uint32_t>(s, kMoeRdSlotMask)[tid - 32] = 0;
    }
    if (tid == 0)
    {
        for (int i = 0; i < kBfNumBars; ++i)
        {
            uint32_t const count = (i >= kBfBarAccEmpty && i < kBfBarAccEmpty + kBfAccs) ? 4u
                : (i >= kBfBarHReady && i < kBfBarHReady + kBfKb)                        ? 2u
                                                                                         : 1u;
            mbarInit(bbar(s, i), count);
        }
        fenceMbarInit();
    }
    if (warp == 1)
    {
        tmemAlloc(s.redS + kMoeRdScal + 4 * kMoeScTmem, kBfTmemCols);
    }
    tcFenceBefore();
    __syncthreads();
    tcFenceAfter();
    if (sd[kMdMagic] != kBfMagic)
    {
        __trap();
    }
    if (tid == 0)
    {
        moeStamp(ctx, kMoeStDesc);
    }
    if (tid == 32)
    {
        prefetchTmap(sd[kMdTmapW13]);
        prefetchTmap(sd[kMdTmapW2]);
        bool const warmTma = !(sd[kMdFlags] & kMoeFlagNoWarmTma);
        mbarExpectTx(bbar(s, kBfBarGain), kMoeHidden * 2 + (warmTma ? 2 * kBfUnitBytes : 0));
        bulkCopy(
            s.base + kBfSmGain, reinterpret_cast<void const*>(sd[kMdGainPre]), kMoeHidden * 2, bbar(s, kBfBarGain));
        if (warmTma)
        {
            // One box through each stack's tensor map into ring stage 0 (requested only after the norm has waited on
            // this barrier): the launch's first tensor copy per map is not one the weight stream waits for.
            uint64_t const policy = policyEvictFirst();
            tmaLoad3d(s.base + kBfSmRing, reinterpret_cast<void const*>(sd[kMdTmapW13]), 0, 0, 0, bbar(s, kBfBarGain),
                policy);
            tmaLoad3d(
                s.base + kBfSmRing, reinterpret_cast<void const*>(sd[kMdTmapW2]), 0, 0, 0, bbar(s, kBfBarGain), policy);
        }
    }
    // The workspace copy: the launch rotation (kMdReserved = 0), or the layer's own copy of its own rotation
    // (kMdReserved = (1 + copy) | copies << 8: the BF16 layers of a forward take turns, each clearing the next one's).
    int64_t const own = sd[kMdNumWords - 1];
    int const cp = own != 0 ? static_cast<int>(own & 0xff) - 1 : ctx.copy;
    int const ncp = own != 0 ? static_cast<int>((own >> 8) & 0xff) : ctx.numCopies;
    uint8_t* ws = dptr<uint8_t>(s, kMdWs);
    uint8_t* wsCur = ws + static_cast<size_t>(cp) * kBfWsBytes;
    uint8_t* wsNext = ws + static_cast<size_t>((cp + 1) % ncp) * kBfWsBytes;
    // kMdGs: the NVFP4 stage's workspace (its copy n + 1 is cleared here), or 0.
    uint8_t* nvNext = sd[kMdGs] != 0
        ? reinterpret_cast<uint8_t*>(sd[kMdGs]) + static_cast<size_t>((ctx.copy + 1) % ctx.numCopies) * kMoeWsBytes
        : nullptr;
    bool const normFirst = (sd[kMdFlags] & kMoeFlagQuantFirst) != 0;
    if (normFirst)
    {
        shareNorm(ctx, s, wsCur);
    }
    route(ctx, s);
    __syncthreads();
    if (tid == 0)
    {
        moeStamp(ctx, kMoeStTop8);
    }
    if (!normFirst)
    {
        shareNorm(ctx, s, wsCur);
    }
    place2(ctx, s);
    if (tid == 0)
    {
        moeStamp(ctx, kMoeStRouted);
    }
    int const E = rd<int const>(s, kMoeRdScal)[kMoeScE];
    uint32_t const tmem = rd<uint32_t const>(s, kMoeRdScal)[kMoeScTmem];
    BCut c;
    bMakeCut(c, E, (sd[kMdFlags] & kMoeFlagTileCut) != 0, static_cast<int>(blockIdx.x));
    int const n = bUnitsOf(c);
    int const issued = n < kBfStages ? n : kBfStages;
    if (tid == 0)
    {
        moeStamp(ctx, kMoeStCut);
        if (MSO_MK_STAMPS_DETAIL(ctx) != nullptr)
        {
            MSO_MK_STAMPS(ctx)[kStampI2aRaw6] = (static_cast<uint64_t>(E) << 32) | static_cast<uint32_t>(n);
        }
    }
    if (E > 0)
    {
        gatherXB(ctx, s, c, issued, wsCur);
    }
    if (tid == 0)
    {
        moeStamp(ctx, kMoeStQuant);
    }
    fenceProxyAsyncSmem();
    tcFenceBefore();
    __syncthreads();
    tcFenceAfter();
    if (warp == 0)
    {
        if (tid == 0)
        {
            produceLoopB(ctx, s, c, issued);
        }
    }
    else if (warp == 1)
    {
        mmaLoopB(ctx, s, c, tmem);
    }
    else if (warp < 4)
    {
        loaderLoopB(ctx, s, c, wsCur, wsNext, nvNext);
    }
    else
    {
        epilogueLoopB(ctx, s, c, tmem, wsCur);
    }
    tcFenceBefore();
    __syncthreads();
    tcFenceAfter();
    if (warp == 1)
    {
        tmemDealloc(tmem, kBfTmemCols);
    }
    if (unit > 0)
    {
        finalUnit(ctx, s, unit - 1, wsCur + kBfWsY - kMoeWsY);
    }
}
