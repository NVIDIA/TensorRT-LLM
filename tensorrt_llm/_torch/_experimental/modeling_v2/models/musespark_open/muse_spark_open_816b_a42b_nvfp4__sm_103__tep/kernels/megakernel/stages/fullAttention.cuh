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

// Full-attention (NoPE, all-context) stages of a full-attention layer's decode step: the 8 verify rows of one
// sequence, 16 query heads and 2 KV heads of 64 on one rank, the FP8 paged cache (32-token pages, K and V of a page in
// separate 4 KiB blocks [2 heads][32 tokens][64]). Included by msoMegakernel.cu inside namespace mso_mk.
//
// FaAttn (one unit per CTA, U units per layer): unit u owns pages [floor(u P / eff), floor((u + 1) P / eff)) of the
// sequence's P = ceil(L / 32) pages, eff = min(U, ceil(P / minPages)), all computed on the device from kv_lens[0], so a
// captured graph serves every context. The unit's pages are taken in tiles of 4 (128 keys); the stock split kernel's
// structure (full_attention_split_kv_kernel.py) in the megakernel's 256 threads: warpgroup wg (warps 4 wg .. 4 wg + 3)
// takes tiles wg, wg + 2, ... with its own running (m, l), its TMEM accumulator O_wg (columns 128 wg) and S buffer S_wg
// (columns 256 + 128 wg); TMEM lane r = query row r = 64 h + 8 tok + gq (KV head h, token tok, query head 8 h + gq),
// and thread r of the warpgroup holds row r's 128 scores. Per tile: S = Q K^T as 8 tcgen05.mma kind::f8f6f4 M128 N128
// K32 (Q block-diagonal over the two heads, as Q_hi + Q_lo, two E4M3 terms, K-major SW64 in shared memory; B = the
// tile's 128 keys of one head, K-major SW64); the stock softmax rule (scale log2(e) / 8, P = exp2(s sc + 4 - m sc)
// rounded to E4M3 against a running max m raised only when the tile max exceeds it by more than 2^4 in log2 units; the
// denominator sums the unrounded P); P (E4M3, 32 columns) overwrites S_wg; then per page one tcgen05.mma M128 N128 K32
// with A = P from TMEM and B = the page's V of both heads (MN-major SW64) accumulates O_wg (row r's head's 64 columns
// are its own). The epilogue merges the two warpgroups' (m, l, O) per row as the stock does; the unit's normalized O
// (FP16) and log2-sum-exp (FP32) of every row go to single-reader Lamport slots.
//
// FaMerge (128 units, one per (token, query head) row): polls the eff units' slots of its row, merges them with weights
// 2^(lse - max) in a fixed order, rounds to BF16 and applies the attention norm-gate (bf16(rmsnorm(a)) *
// bf16(sigmoid(gate))), writing the o_proj input [8, 1024] BF16.
//
// Ownership and progress.
// * Pages: unit u reads only its range. KV pages are requested by TMA at kernel entry, before griddepcontrol.wait (the
//   cache below the step's first new position, the block table and kv_lens are immutable during the step); the step's
//   own 8 positions L - 8 .. L - 1 (stale in the landed copy) are computed by the issuer warp of the tile's warpgroup
//   after the wait, written to the pool (the stock bytes e4m3_rn_satfinite(bf16(x * bf16(s)))) and patched into the
//   landed ring page, whose V rows at positions >= L are zeroed, then fence.proxy.async.shared::cta before that tile's
//   QK is issued. No other unit reads those pages in this launch; later steps read the pool after this kernel
//   completed.
// * MMAs: warp 4 wg issues every MMA of its warpgroup's tiles, warp-collectively (elect.sync in the statement, so the
//   descriptors stay warp-uniform): after the warpgroup's P barrier of tile t, PV(t) and the commits on empty[slot] and
//   odone[wg], then QK(t + 2) into S_wg (after PV(t) in the same thread's order) and its commit on s_full[wg]. So when
//   the softmax of tile t sees s_full, PV(t - 2) has completed too: the (rare) O rescale reads O_wg without a wait.
//   While one warpgroup issues and waits, the other runs its softmax.
// * Ring: kSlots tile slots of 4 pages, stored as four parts [K head 0 | K head 1 | V head 0 | V head 1] of kRing 2 KiB
//   boxes, so a slot's 128 keys of one head are contiguous. The first kSlots tiles are requested at entry; tile t's
//   slot is refilled with tile t + kSlots by tile t's warpgroup (one page per warp, after empty[slot] = PV(t) complete;
//   the issuer warp arms full[slot]) in its idle time before S(t + 2). full / empty phases are waited in tile order.
// * Partials: row R of unit u is written once by one thread pair (32 FP16 each; the LSE by the wg 0 thread), read and
//   emptied by merge unit R only. Merge units are second on their CTAs and wait only on attention units, which wait on
//   nothing but their own TMA, MMAs and the predecessor: progress holds with every CTA resident (one per SM).

// The constants (namespace fa) are in fullAttentionConfig.cuh, included before the device-only part.

// ---------------------------------------------------------------------------------------------------------------
// PTX helpers (tcgen05, TMA tensor copies, named barriers)
// ---------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ float faNegInf()
{
    return __int_as_float(static_cast<int>(0xff800000u));
}

__device__ __forceinline__ void faTmaLoad2d(
    uint32_t dst, void const* tmap, int c0, int c1, uint32_t bar, uint64_t policy)
{
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, "
        "{%2, "
        "%3}], [%4], %5;" ::"r"(dst),
        "l"(tmap), "r"(c0), "r"(c1), "r"(bar), "l"(policy)
        : "memory");
}

__device__ __forceinline__ void faPrefetchTmap(void const* tmap)
{
    asm volatile("prefetch.tensormap [%0];" ::"l"(tmap) : "memory");
}

// Shared-memory matrix descriptor (sm_100 UMMA): start, LBO, SBO in 16 B units, version 1, layout SWIZZLE_64B (4).
__device__ __forceinline__ uint64_t faDesc(uint32_t addr, uint32_t lbo, uint32_t sbo)
{
    return static_cast<uint64_t>((addr >> 4) & 0x3fffu) | (static_cast<uint64_t>(lbo & 0x3fffu) << 16)
        | (static_cast<uint64_t>(sbo & 0x3fffu) << 32) | (1ull << 46) | (4ull << 61);
}

__device__ __forceinline__ void faMmaSS(uint32_t dTmem, uint64_t aDesc, uint64_t bDesc, uint32_t idesc, uint32_t acc)
{
    asm volatile(
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}\n" ::"r"(dTmem),
        "l"(aDesc), "l"(bDesc), "r"(idesc), "r"(acc)
        : "memory");
}

__device__ __forceinline__ void faMmaTS(uint32_t dTmem, uint32_t aTmem, uint64_t bDesc, uint32_t idesc, uint32_t acc)
{
    asm volatile(
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, p;\n\t}\n" ::"r"(dTmem),
        "r"(aTmem), "l"(bDesc), "r"(idesc), "r"(acc)
        : "memory");
}

__device__ __forceinline__ void faCommit(uint32_t bar)
{
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(bar) : "memory");
}

__device__ __forceinline__ void faFenceBefore()
{
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void faFenceAfter()
{
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void faLd32(uint32_t taddr, float (&v)[32])
{
    uint32_t r[32];
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]),
        "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]),
        "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
        : "r"(taddr)
        : "memory");
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
    for (int i = 0; i < 32; ++i)
    {
        v[i] = __uint_as_float(r[i]);
    }
}

__device__ __forceinline__ void faSt32(uint32_t taddr, float const (&v)[32])
{
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
        "%15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};" ::"r"(taddr),
        "r"(__float_as_uint(v[0])), "r"(__float_as_uint(v[1])), "r"(__float_as_uint(v[2])), "r"(__float_as_uint(v[3])),
        "r"(__float_as_uint(v[4])), "r"(__float_as_uint(v[5])), "r"(__float_as_uint(v[6])), "r"(__float_as_uint(v[7])),
        "r"(__float_as_uint(v[8])), "r"(__float_as_uint(v[9])), "r"(__float_as_uint(v[10])),
        "r"(__float_as_uint(v[11])), "r"(__float_as_uint(v[12])), "r"(__float_as_uint(v[13])),
        "r"(__float_as_uint(v[14])), "r"(__float_as_uint(v[15])), "r"(__float_as_uint(v[16])),
        "r"(__float_as_uint(v[17])), "r"(__float_as_uint(v[18])), "r"(__float_as_uint(v[19])),
        "r"(__float_as_uint(v[20])), "r"(__float_as_uint(v[21])), "r"(__float_as_uint(v[22])),
        "r"(__float_as_uint(v[23])), "r"(__float_as_uint(v[24])), "r"(__float_as_uint(v[25])),
        "r"(__float_as_uint(v[26])), "r"(__float_as_uint(v[27])), "r"(__float_as_uint(v[28])),
        "r"(__float_as_uint(v[29])), "r"(__float_as_uint(v[30])), "r"(__float_as_uint(v[31]))
        : "memory");
}

// ---- 16-column TMEM load / store (the MSO_MK_FA_RC == 16 rescale) ----
__device__ __forceinline__ void faLd16(uint32_t taddr, float (&v)[16])
{
    uint32_t r[16];
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
        "%15}, "
        "[%16];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
        : "r"(taddr)
        : "memory");
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        v[i] = __uint_as_float(r[i]);
    }
}

__device__ __forceinline__ void faSt16f(uint32_t taddr, float const (&v)[16])
{
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
        "%15, %16};" ::"r"(taddr),
        "r"(__float_as_uint(v[0])), "r"(__float_as_uint(v[1])), "r"(__float_as_uint(v[2])), "r"(__float_as_uint(v[3])),
        "r"(__float_as_uint(v[4])), "r"(__float_as_uint(v[5])), "r"(__float_as_uint(v[6])), "r"(__float_as_uint(v[7])),
        "r"(__float_as_uint(v[8])), "r"(__float_as_uint(v[9])), "r"(__float_as_uint(v[10])),
        "r"(__float_as_uint(v[11])), "r"(__float_as_uint(v[12])), "r"(__float_as_uint(v[13])),
        "r"(__float_as_uint(v[14])), "r"(__float_as_uint(v[15]))
        : "memory");
}

// ---- end of the 16-column load / store ----

__device__ __forceinline__ void faSt8(uint32_t taddr, uint32_t const (&w)[8])
{
    asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" ::"r"(taddr),
                 "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3]), "r"(w[4]), "r"(w[5]), "r"(w[6]), "r"(w[7])
                 : "memory");
}

__device__ __forceinline__ void faWaitSt()
{
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ float faMulRn(float a, float b)
{
    float d;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

__device__ __forceinline__ float faAddRn(float a, float b)
{
    float d;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

__device__ __forceinline__ float faSubRn(float a, float b)
{
    float d;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

__device__ __forceinline__ float faRsqrtApprox(float x)
{
    float d;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(x));
    return d;
}

__device__ __forceinline__ float faEx2(float x)
{
    float d;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(x));
    return d;
}

__device__ __forceinline__ float faLg2(float x)
{
    float d;
    asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(x));
    return d;
}

__device__ __forceinline__ float faRcp(float x)
{
    float d;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(x));
    return d;
}

// Four f32 -> four E4M3 (RN, satfinite), a in the low byte.
__device__ __forceinline__ uint32_t faE4m3x4(float a, float b, float c, float d)
{
    uint32_t w;
    asm("{\n\t.reg .b16 lo, hi;\n\tcvt.rn.satfinite.e4m3x2.f32 lo, %2, %1;\n\tcvt.rn.satfinite.e4m3x2.f32 hi, %4, "
        "%3;\n\t"
        "mov.b32 %0, {lo, hi};\n\t}\n"
        : "=r"(w)
        : "f"(a), "f"(b), "f"(c), "f"(d));
    return w;
}

// Veltkamp split: the nearest value with 4 significant bits (explicit .rn, nothing reassociated).
__device__ __forceinline__ float faSplitHi(float x)
{
    float d;
    asm("{\n\t.reg .f32 c, t;\n\tmul.rn.f32 c, %1, 0f49800008;\n\tsub.rn.f32 t, c, %1;\n\tsub.rn.f32 %0, c, t;\n\t}\n"
        : "=f"(d)
        : "f"(x));
    return d;
}

__device__ __forceinline__ float faFmax3(float a, float b, float c)
{
    float d;
    asm("max.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
    return d;
}

__device__ __forceinline__ uint32_t faLdS32(uint32_t addr)
{
    uint32_t v;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(addr));
    return v;
}

__device__ __forceinline__ void faStS128(uint32_t addr, uint4 v)
{
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}

// Thread 0's accumulated clock64 cycles per activity (stamped builds; the sub-stamp map is above faStampWg).
struct FaProf
{
    long long full = 0, empty = 0, drain = 0, sm = 0, sw = 0, bar = 0, tma = 0, mma = 0, ld = 0, ar = 0, st = 0;
};

__device__ __forceinline__ long long faClk(Ctx const& ctx)
{
    return MSO_MK_STAMPS_DETAIL(ctx) != nullptr ? clock64() : 0;
}

__device__ __forceinline__ void faStampVal(Ctx const& ctx, int k, long long v)
{
    if (MSO_MK_STAMPS_DETAIL(ctx) != nullptr && threadIdx.x == 0)
    {
        MSO_MK_STAMPS(ctx)[kStampSub + k] = static_cast<uint64_t>(v);
    }
}

// Byte offset of 16 B chunk c of row `row` of a SWIZZLE_64B tile of 64 B rows (512 B aligned): the TMA / UMMA pattern.
__device__ __forceinline__ uint32_t faSw64(int row, int c)
{
    return static_cast<uint32_t>(row * 64 + 16 * (c ^ ((row >> 1) & 3)));
}

// ---------------------------------------------------------------------------------------------------------------
// Geometry (device): pages of the context, keyed units, the unit's page range.
// ---------------------------------------------------------------------------------------------------------------
struct FaGeom
{
    int L, P, eff, p0, p1;
};

// ---- MSO_MK_LD_FG: the served full layers' geometry fields as constants (else MSO_MK_FG_*(v) is the field v) ----
#if MSO_MK_LD_FG
#define MSO_MK_FG_UNITS(v) kCtas
#define MSO_MK_FG_MINPAGES(v) 4
#define MSO_MK_FG_ENTRY(v) fa::kRing
#define MSO_MK_FG_PREFETCH(v) 0
#else
#define MSO_MK_FG_UNITS(v) v
#define MSO_MK_FG_MINPAGES(v) v
#define MSO_MK_FG_ENTRY(v) v
#define MSO_MK_FG_PREFETCH(v) v
#endif
#if MSO_MK_LD_FG
__device__ __forceinline__ void ldFgCheck(Ctx const& ctx)
{
    if (ctx.p[kPFaUnits] != kCtas || ctx.p[kPFaMinPages] != 4 || ctx.p[kPFaEntryPages] != fa::kRing
        || ctx.p[kPFaPrefetch] != 0)
    {
        __trap(); // MSO_MK_LD_FG: the build fixes the full layers' geometry fields to the served ones
    }
}
#endif
// ---- end MSO_MK_LD_FG ----

__device__ __forceinline__ FaGeom faGeom(int L, int units, int minPages, int u)
{
    FaGeom g;
    g.L = L;
    g.P = (L + fa::kPage - 1) / fa::kPage;
    int const want = (g.P + minPages - 1) / minPages;
    g.eff = want < units ? want : units;
    if (u < g.eff)
    {
        g.p0 = static_cast<int>(static_cast<int64_t>(u) * g.P / g.eff);
        g.p1 = static_cast<int>(static_cast<int64_t>(u + 1) * g.P / g.eff);
    }
    else
    {
        g.p0 = g.p1 = 0;
    }
    return g;
}

// ---- kIFlagFaLateLight: the page split weighted 16 (early slots) : kFaLateWeight (the last kFaLateSlots) ----
__device__ __forceinline__ int64_t faSplitW(int s, int w) // the late weight w as an argument
{
    int const e0 = kCtas - kFaLateSlots;
    return s <= e0 ? 16ll * s : 16ll * e0 + static_cast<int64_t>(w) * (s - e0);
}

__device__ __forceinline__ FaGeom faGeomW(Ctx const& ctx, int L, int units, int minPages, int u)
{
    FaGeom g = faGeom(L, units, minPages, u);
    if ((icFaFlags(ctx.p[kPIFlags]) & kIFlagFaLateLight) && g.eff == kCtas && units == kCtas && u < g.eff)
    {
        int const w = kFaLateWeight - static_cast<int>((icFaFlags(ctx.p[kPIFlags]) >> kF3FaLateWShift) & 3);
        int64_t const tot = faSplitW(g.eff, w);
        g.p0 = static_cast<int>(faSplitW(u, w) * g.P / tot);
        g.p1 = static_cast<int>(faSplitW(u + 1, w) * g.P / tot);
    }
    return g;
}

// ---- end kIFlagFaLateLight ----

__device__ __forceinline__ int faReadL(Ctx const& ctx)
{
    return __ldg(ptr<int const>(ctx, kPFaKvLens)); // immutable during the launch
}

// ---- the unit a FaAttn role slot computes (kIFlagFaRevUnits: keyed slots reversed) ----
__device__ __forceinline__ int faUnitOfSlot(Ctx const& ctx, int slot, int eff)
{
    return ((icFaFlags(ctx.p[kPIFlags]) & kIFlagFaRevUnits) && slot < eff) ? eff - 1 - slot : slot;
}

// ---- end kIFlagFaRevUnits ----

// Box `part` (< kBoxesPerPage) of the unit page whose ids are (blkK, blkV) into ring page p, on barrier bar: parts 0, 1
// = K head 0, 1 (2 KiB, map K); with kVBox part 2 = V of both heads (4 KiB, map V), else parts 2, 3 = V head 0, 1.
__device__ __forceinline__ void faIssueBox(
    Ctx const& ctx, uint32_t ring, int p, int part, int blkK, int blkV, uint32_t bar, uint64_t policy)
{
    uint8_t const* maps = ptr<uint8_t const>(ctx, kPFaTmap);
    if (part < 2)
    {
        faTmaLoad2d(ring + part * fa::kRingPart + p * fa::kBox, maps, 0, blkK * 64 + part * 32, bar, policy);
    }
    else if (fa::kVBox)
    {
        faTmaLoad2d(ring + fa::kOffV + p * fa::kVPage, maps + 128, 0, blkV * 64, bar, policy);
    }
    else
    {
        faTmaLoad2d(ring + fa::kOffV + (part - 2) * fa::kRingPart + p * fa::kBox, maps, 0, blkV * 64 + (part - 2) * 32,
            bar, policy);
    }
}

// The byte offset in the ring of row `row` (0 .. 31), 16-byte chunk c, of V head hk of ring page p (SW64 pattern).
__device__ __forceinline__ uint32_t faVAt(int p, int hk, int row, int c)
{
    return fa::kVBox ? static_cast<uint32_t>(fa::kOffV + p * fa::kVPage + hk * fa::kBox) + faSw64(row, c)
                     : static_cast<uint32_t>(fa::kOffV + hk * fa::kRingPart + p * fa::kBox) + faSw64(row, c);
}

// Tile t (the unit's pages 4 t .. 4 t + 3 that are < n) into slot t % kSlots: kBoxesPerPage boxes per page on
// full[slot]. Warp-collective: lane 0 arms the barrier, lane 4 q + part < 4 np issues box part of page q.
__device__ __forceinline__ void faIssueTile(
    Ctx const& ctx, uint32_t base, int const* ids, int t, int n, uint64_t policy)
{
    int const lane = threadIdx.x & 31;
    int const slot = t % fa::kSlots;
    int const left = n - fa::kTilePages * t;
    int const np = left < fa::kTilePages ? left : fa::kTilePages;
    uint32_t const bar = base + fa::kOffFull + 8 * slot;
    if (lane == 0)
    {
        mbarExpectTx(bar, static_cast<uint32_t>(np * fa::kStageBytes));
    }
    __syncwarp();
    int const q = lane >> 2;
    int const part = lane & 3;
    if (q < np && part < fa::kBoxesPerPage)
    {
        int const rel = fa::kTilePages * t + q;
        faIssueBox(ctx, base + fa::kOffRing, fa::kTilePages * slot + q, part, ids[rel], ids[fa::kMaxUnitPages + rel],
            bar, policy);
    }
}

// L2 prefetch of tile t's K and V blocks (warp-collective; pages beyond the unit are skipped).
__device__ __forceinline__ void faPrefetchTile(Ctx const& ctx, int const* ids, int t, int n)
{
    int const lane = threadIdx.x & 31;
    int const q = lane >> 1;
    int const rel = fa::kTilePages * t + q;
    if (q < fa::kTilePages && rel < n)
    {
        int const blk = ids[(lane & 1) * fa::kMaxUnitPages + rel];
        prefetchBulkL2(ptr<uint8_t const>(ctx, kPFaPool) + static_cast<size_t>(blk) * 4096, 4096);
    }
}

// Tiles requested at entry: ceil(FaEntryPages / 4), at most the ring and the unit.
__device__ __forceinline__ int faEntryTiles(Ctx const& ctx, int nt)
{
    int const entry = static_cast<int>(MSO_MK_FG_ENTRY(ctx.p[kPFaEntryPages])); // MSO_MK_LD_FG
    int first = (entry + fa::kTilePages - 1) / fa::kTilePages;
    first = first < fa::kSlots ? first : fa::kSlots;
    first = first < nt ? first : nt;
    return first > 0 ? first : 0;
}

// faPrologue's tail. Warp 0, after scal and ids are in the region: the ring / S / O barriers, the TMEM allocation
// and the L2 prefetches of the FaPrefetch tiles past the ring.
__device__ __forceinline__ void faPrologueTail(Ctx const& ctx, uint32_t base, int const* ids, int n)
{
    int const lane = threadIdx.x & 31;
    if (lane == 0)
    {
        for (int s = 0; s < fa::kSlots; ++s)
        {
            mbarInit(base + fa::kOffFull + 8 * s, 1);
            mbarInit(base + fa::kOffEmpty + 8 * s, 1);
        }
        for (int b = 0; b < 2; ++b) // per warpgroup
        {
            mbarInit(base + fa::kOffSFull + 8 * b, 1);
            mbarInit(base + fa::kOffODone + 8 * b, 1);
        }
        fenceMbarInit();
    }
    // TMEM: all 512 columns for a keyed unit (the unit is the CTA's first; freed after its epilogue).
    if (n > 0)
    {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" ::"r"(base + fa::kOffTmem)
                     : "memory");
    }
    __syncwarp();
    int const nt = (n + fa::kTilePages - 1) / fa::kTilePages;
    if (faEntryTiles(ctx, nt) > 0)
    {
        int const pf = static_cast<int>(MSO_MK_FG_PREFETCH(ctx.p[kPFaPrefetch])); // MSO_MK_LD_FG
        for (int t = fa::kSlots; t < fa::kSlots + pf && t < nt; ++t)
        {
            faPrefetchTile(ctx, ids, t, n);
        }
    }
}

// Warp 0, before griddepcontrol.wait: the unit's geometry and page ids into shared memory, the barriers, the TMEM
// allocation and L2 prefetches of the FaPrefetch tiles past the ring (the cache, the block table and kv_lens are
// immutable during the step). The ring's first tiles follow in faPrologueIssue.
__device__ __noinline__ void faPrologue(Ctx const& ctx, int unit, uint32_t off)
{
    icFaCheck(ctx.p[kPIFlags]); // the full-attention kPIFlags bits this build fixes
#if MSO_MK_LD_FG
    ldFgCheck(ctx);
#endif
    int const lane = threadIdx.x & 31;
    uint32_t const base = ctx.smemBase + off;
    uint8_t* region = ctx.smem + off;
    int* scal = reinterpret_cast<int*>(region + fa::kOffScal);
    int* ids = reinterpret_cast<int*>(region + fa::kOffIds);
    FaGeom g = faGeom(faReadL(ctx), static_cast<int>(MSO_MK_FG_UNITS(ctx.p[kPFaUnits])),
        static_cast<int>(MSO_MK_FG_MINPAGES(ctx.p[kPFaMinPages])), unit);                           // MSO_MK_LD_FG
    g = faGeomW(ctx, g.L, static_cast<int>(MSO_MK_FG_UNITS(ctx.p[kPFaUnits])),
        static_cast<int>(MSO_MK_FG_MINPAGES(ctx.p[kPFaMinPages])), faUnitOfSlot(ctx, unit, g.eff)); // MSO_MK_LD_FG
    int const n = g.p1 - g.p0;
    if (n > fa::kMaxUnitPages)
    {
        __trap(); // the host checks the context bound; this is the device guard
    }
    if (lane == 0)
    {
        faPrefetchTmap(ptr<void const>(ctx, kPFaTmap));
    }
    int const* table = ptr<int const>(ctx, kPFaTable);
    int64_t const stride = ctx.p[kPFaTableStride];
    // every load in flight before the stores (one table round trip instead of one per 32 pages)
    constexpr int kPerLane = fa::kMaxUnitPages / 32;
    int vk[kPerLane], vv[kPerLane];
#pragma unroll
    for (int k = 0; k < kPerLane; ++k)
    {
        int const i = lane + 32 * k;
        if (i < n)
        {
            vk[k] = table[g.p0 + i];
            vv[k] = table[stride + g.p0 + i];
        }
    }
#pragma unroll
    for (int k = 0; k < kPerLane; ++k)
    {
        int const i = lane + 32 * k;
        if (i < n)
        {
            ids[i] = vk[k];
            ids[fa::kMaxUnitPages + i] = vv[k];
        }
    }
    if (lane == 0)
    {
        scal[0] = g.L;
        scal[1] = g.p0;
        scal[2] = g.p1;
        scal[3] = g.eff;
        scal[4] = g.P;
        scal[5] = __float_as_int(__ldg(ptr<float const>(ctx, kPFaKvScale))); // immutable during the step
    }
    faPrologueTail(ctx, base, ids, n); // the barriers, TMEM and L2 prefetches (shared with faPrologueMeta)
}

// ---- kIFlagFaEarlyMeta ----
// Warp 0, a deferred FaAttn unit (after the CTA's QKV run) whose meta area faEarlyMeta filled at kernel entry: scal and
// the page ids from the meta area into the region, then faPrologueTail.
__device__ __noinline__ void faPrologueMeta(Ctx const& ctx, uint32_t off)
{
    int const lane = threadIdx.x & 31;
    uint8_t* region = ctx.smem + off;
    int const* meta = reinterpret_cast<int const*>(ctx.smem + kSmemFaMeta);
    int const* metaIds = reinterpret_cast<int const*>(ctx.smem + kSmemFaMeta + fa::kMetaIds);
    int* scal = reinterpret_cast<int*>(region + fa::kOffScal);
    int* ids = reinterpret_cast<int*>(region + fa::kOffIds);
    int const n = meta[2] - meta[1];
    for (int i = lane; i < n; i += 32)
    {
        ids[i] = metaIds[i];
        ids[fa::kMaxUnitPages + i] = metaIds[fa::kMetaIdsCap + i];
    }
    if (lane < 6)
    {
        scal[lane] = meta[lane];
    }
    __syncwarp();
    faPrologueTail(ctx, ctx.smemBase + off, ids, n);
}

// ---- end kIFlagFaEarlyMeta ----

// Every warp of the CTA, after faPrologue and a CTA barrier, before griddepcontrol.wait: the ring's first tiles (tile t
// < kSlots in slot t), warp w the pages w, w + kWarps, ... (lanes 0 .. 3: the page's four boxes); warp 0 lane t arms
// tile t's barrier.
__device__ __noinline__ void faPrologueIssue(Ctx const& ctx, uint32_t off)
{
    int const lane = threadIdx.x & 31;
    int const warp = threadIdx.x >> 5;
    uint32_t const base = ctx.smemBase + off;
    uint8_t const* region = ctx.smem + off;
    int const* scal = reinterpret_cast<int const*>(region + fa::kOffScal);
    int const* ids = reinterpret_cast<int const*>(region + fa::kOffIds);
    int const n = scal[2] - scal[1];
    int const nt = (n + fa::kTilePages - 1) / fa::kTilePages;
    int const first = faEntryTiles(ctx, nt);
    int const pages = fa::kTilePages * first < n ? fa::kTilePages * first : n;
    if (warp == 0 && lane < first)
    {
        int const left = n - fa::kTilePages * lane;
        int const np = left < fa::kTilePages ? left : fa::kTilePages;
        mbarExpectTx(base + fa::kOffFull + 8 * lane, static_cast<uint32_t>(np * fa::kStageBytes));
    }
    uint64_t const policy = policyEvictFirst();
    for (int pg = warp; pg < pages; pg += kWarps)
    {
        if (lane < fa::kBoxesPerPage)
        {
            faIssueBox(ctx, base + fa::kOffRing, pg, lane, ids[pg], ids[fa::kMaxUnitPages + pg],
                base + fa::kOffFull + 8 * (pg / fa::kTilePages), policy);
        }
    }
}

// ---- the one-launch full layer ----
// Warp 0 at kernel entry, for a FaAttn unit that is not the CTA's first (its prologue runs after the QKV run): the
// unit's block-table entries and the step scalars into L2, so the deferred prologue's loads hit L2.
__device__ __noinline__ void faPrefetchMeta(Ctx const& ctx, int slot)
{
    icFaCheck(ctx.p[kPIFlags]); // the full-attention kPIFlags bits this build fixes
    int const lane = threadIdx.x & 31;
    int const units = static_cast<int>(MSO_MK_FG_UNITS(ctx.p[kPFaUnits]));          // MSO_MK_LD_FG
    int const minPages = static_cast<int>(MSO_MK_FG_MINPAGES(ctx.p[kPFaMinPages])); // MSO_MK_LD_FG
    FaGeom g = faGeom(faReadL(ctx), units, minPages, slot);
    g = faGeomW(ctx, g.L, units, minPages, faUnitOfSlot(ctx, slot, g.eff));
    int const* table = ptr<int const>(ctx, kPFaTable);
    int64_t const stride = ctx.p[kPFaTableStride];
    int const n = g.p1 - g.p0;
    int const lines = (n * 4 + 127) / 128 + 1; // <= 9 lines of 128 B per table row
    if (n > 0 && lane < 2 * lines)
    {
        int const* row = table + (lane < lines ? 0 : stride) + g.p0;
        int const l = lane < lines ? lane : lane - lines;
        asm volatile("prefetch.global.L2 [%0];" ::"l"(reinterpret_cast<char const*>(row) + 128 * l));
    }
    if (lane == 0)
    {
        asm volatile("prefetch.global.L2 [%0];" ::"l"(ptr<float const>(ctx, kPFaKvScale)));
        faPrefetchTmap(ptr<void const>(ctx, kPFaTmap));
    }
}

// ---- kIFlagFaQPrep ----
// QkvSum of a q block (threads 0-127, all of them; query head hq): this thread's 4 columns [c0, c0 + 4) of token tok's
// BF16 row o through FaAttn's Q prep, the norm's sum in its order (chunks of 8 accumulated from (x0^2 + x1^2), the
// pair sums, then ((l0 + l1) + (l2 + l3)) + ((l4 + l5) + (l6 + l7))): a chunk is two neighbouring lanes, a pair four,
// and the four pairs of a row are the four warps (QkvSum thread t: tile t / 32 of the block, token (t % 32) / 4,
// columns 16 (t / 32) + 4 (t % 4)). Q_hi / Q_lo words of FaAttn row r = (hq / 8) 64 + 8 tok + hq % 8 at kArAhQ + 64 r +
// c0 and kArAhQ + 8192 + 64 r + c0, as Lamport words.
__device__ __noinline__ void faQPrepStore(Ctx const& ctx, int tok, int c0, int hq, uint2 o)
{
    int const tid = threadIdx.x;
    // The words FaAttn would read from kArFaQkv (lamport: a word equal to kEmpty is published as +0).
    uint32_t const w0 = lamport(o.x), w1 = lamport(o.y);
    float const x[4] = {bf16Lo(w0), bf16Hi(w0), bf16Lo(w1), bf16Hi(w1)};
    float half = faAddRn(faMulRn(x[0], x[0]), faMulRn(x[1], x[1]));
    half = faAddRn(half, faMulRn(x[2], x[2]));
    half = faAddRn(half, faMulRn(x[3], x[3]));
    float const prev = __shfl_up_sync(0xffffffffu, half, 1); // lane - 1: the chunk's first half
    float chunk = half;
    if (c0 & 4)                                              // the chunk's second half continues the first half's sum
    {
        chunk = faAddRn(prev, faMulRn(x[0], x[0]));
        chunk = faAddRn(chunk, faMulRn(x[1], x[1]));
        chunk = faAddRn(chunk, faMulRn(x[2], x[2]));
        chunk = faAddRn(chunk, faMulRn(x[3], x[3]));
    }
    float const lower = __shfl_up_sync(0xffffffffu, chunk, 2);    // lane - 2: the pair's first chunk
    float* pairs = reinterpret_cast<float*>(ctx.smem + kSmemRed); // [8 tokens][4 warps]
    if ((c0 & 12) == 12)
    {
        pairs[tok * 4 + (tid >> 5)] = faAddRn(lower, chunk);
    }
    asm volatile("bar.sync 1, 128;" ::: "memory"); // threads 0-127
    float const* pr = pairs + tok * 4;
    float const total = faAddRn(faAddRn(pr[0], pr[1]), faAddRn(pr[2], pr[3]));
    float const eps = __int_as_float(static_cast<int>(ctx.p[kPFaEps]));
    float const qs = __int_as_float(static_cast<int>(ctx.p[kPFaQScale]));
    float const inv = faRsqrtApprox(faAddRn(faMulRn(total, 1.0f / fa::kD), eps));
    float v[4], hv[4];
#pragma unroll
    for (int e = 0; e < 4; ++e)
    {
        v[e] = bf16Round(faMulRn(bf16Round(faMulRn(x[e], inv)), qs));
        hv[e] = faSplitHi(v[e]);
    }
    uint32_t const hi = faE4m3x4(hv[0], hv[1], hv[2], hv[3]);
    uint32_t const lo = faE4m3x4(v[0] - hv[0], v[1] - hv[1], v[2] - hv[2], v[3] - hv[3]);
    int const r = (hq / fa::kG) * 64 + tok * 8 + hq % fa::kG;
    st32(ctx.arena + kArAhQ + 64 * r + c0, lamport(hi));
    st32(ctx.arena + kArAhQ + 8192 + 64 * r + c0, lamport(lo));
}

// ---- end kIFlagFaQPrep ----

// ---- kIFlagFaEarlyMeta ----
// Warp 1 at kernel entry (before the dependency wait; kv_lens, the block table and the KV scale are immutable during
// the step), for a FaAttn unit that is not the CTA's first: scal and the unit's page ids into the meta area
// (kSmemFaMeta) and scal[7] = kMetaValid, or scal[7] = 0 when the unit has more than kMetaIdsCap pages (the deferred
// prologue then takes the global path).
__device__ __noinline__ void faEarlyMeta(Ctx const& ctx, int slot)
{
    icFaCheck(ctx.p[kPIFlags]); // the full-attention kPIFlags bits this build fixes
    int const lane = threadIdx.x & 31;
    int* meta = reinterpret_cast<int*>(ctx.smem + kSmemFaMeta);
    int* metaIds = reinterpret_cast<int*>(ctx.smem + kSmemFaMeta + fa::kMetaIds);
    int const units = static_cast<int>(MSO_MK_FG_UNITS(ctx.p[kPFaUnits]));          // MSO_MK_LD_FG
    int const minPages = static_cast<int>(MSO_MK_FG_MINPAGES(ctx.p[kPFaMinPages])); // MSO_MK_LD_FG
    FaGeom g = faGeom(faReadL(ctx), units, minPages, slot);
    g = faGeomW(ctx, g.L, units, minPages, faUnitOfSlot(ctx, slot, g.eff));
    int const n = g.p1 - g.p0;
    bool const ok = n <= fa::kMetaIdsCap;
    if (ok)
    {
        int const* table = ptr<int const>(ctx, kPFaTable);
        int64_t const stride = ctx.p[kPFaTableStride];
        constexpr int kPerLane = (fa::kMetaIdsCap + 31) / 32;
        int vk[kPerLane], vv[kPerLane];
#pragma unroll
        for (int k = 0; k < kPerLane; ++k)
        {
            int const i = lane + 32 * k;
            if (i < n)
            {
                vk[k] = table[g.p0 + i];
                vv[k] = table[stride + g.p0 + i];
            }
        }
        int const kvs = lane == 0 ? __float_as_int(__ldg(ptr<float const>(ctx, kPFaKvScale))) : 0;
#pragma unroll
        for (int k = 0; k < kPerLane; ++k)
        {
            int const i = lane + 32 * k;
            if (i < n)
            {
                metaIds[i] = vk[k];
                metaIds[fa::kMetaIdsCap + i] = vv[k];
            }
        }
        if (lane == 0)
        {
            meta[0] = g.L;
            meta[1] = g.p0;
            meta[2] = g.p1;
            meta[3] = g.eff;
            meta[4] = g.P;
            meta[5] = kvs;
        }
    }
    if (lane == 0)
    {
        meta[7] = ok ? fa::kMetaValid : 0;
    }
    __syncwarp();
}

// Thread 0: the deferred-prologue stamps (kStampFa + k: 0 prologue start, 1 warp 0's part done, 2 ring issued).
__device__ __forceinline__ void faStampFa(Ctx const& ctx, int k)
{
    if (MSO_MK_STAMPS_DETAIL(ctx) != nullptr && threadIdx.x == 0)
    {
        MSO_MK_STAMPS(ctx)[kStampFa + k] = stampTime(); // MSO_MK_SY_CLK picks the clock
    }
}

// ---- end kIFlagFaEarlyMeta ----

// A FaAttn unit that is not the CTA's first (it follows the CTA's QkvRun and maybe region-less units such as QkvSum):
// its prologue (warp 0) and first ring tiles (every warp) right after the unit before it (role i) ends, i.e. once the
// run's weights are consumed (the caller's __syncthreads); before the dependency wait when the run polls early x. Only
// region-less units may sit between the run and the FaAttn unit (the planner checks it).
__device__ __forceinline__ void faAfterQkvRun(Ctx const& ctx, int const* sRoles, int i, int nRoles)
{
    int const j = i + 1;
    if (j < nRoles && (sRoles[j] >> 16) == kStFaAttn)
    {
        uint32_t const off = static_cast<uint32_t>(sRoles[kMaxRoles + j]);
        icFaCheck(ctx.p[kPIFlags]); // the full-attention kPIFlags bits this build fixes
        // kIFlagFaEarlyMeta: the meta area faEarlyMeta filled at entry (written whenever the flag is set and the
        // CTA has a FaAttn unit after its first)
        bool const meta = (icFaFlags(ctx.p[kPIFlags]) & kIFlagFaEarlyMeta)
            && reinterpret_cast<int const volatile*>(ctx.smem + kSmemFaMeta)[7] == fa::kMetaValid;
        faStampFa(ctx, 0); // prologue start
        if (threadIdx.x < 32)
        {
            if (meta) // kIFlagFaEarlyMeta
            {
                faPrologueMeta(ctx, off);
            }
            else
            {
                faPrologue(ctx, sRoles[j] & 0xffff, off);
            }
        }
        faStampFa(ctx, 1); // warp 0's part done
        __syncthreads();
        faPrologueIssue(ctx, off);
        faStampFa(ctx, 2); // ring issued
    }
}

// The raw projection's 16-byte vectors at a[] (kIFlagQkvArena: Lamport words in the arena, polled; else plain loads
// from io.qkv). The whole warp.
template <int kN>
__device__ __forceinline__ void faLoadQkv(Ctx const& ctx, uint4 (&v)[kN], uint8_t const* const (&a)[kN], bool polled)
{
    if (polled)
    {
        pollVecs<kN>(ctx, v, a, kCodeFaQkv);
    }
    else
    {
#pragma unroll
        for (int i = 0; i < kN; ++i)
        {
            v[i] = *reinterpret_cast<uint4 const*>(a[i]);
        }
    }
}

// ---- end of the one-launch full layer ----

// The step's K' (the prep's weightless RMSNorm, BF16) and V of its 8 rows as the stock prep writes them
// (e4m3_rn_satfinite(bf16(x * bf16(s)))): into the staging area stg [8 rows][2 heads][K, V][64 B] and, for the rows on
// this unit's pages, into the pool. The whole warp (lane = (row, quarter of 16 elements); both KV heads); called at the
// stage's start, so its qkv loads overlap the Q loads.
__device__ __forceinline__ void faAppendPrep(
    Ctx const& ctx, FaGeom const& g, __nv_bfloat16 const* qkv, int const* ids, uint8_t* stg, float kvScale)
{
    int const lane = threadIdx.x & 31;
    int const row = lane >> 2;
    int const qm = lane & 3;
    int const pos = g.L - kTok + row;
    int const page = pos / fa::kPage;
    bool const mine = page >= g.p0 && page < g.p1;
    int const slot = pos % fa::kPage;
    float const eps = __int_as_float(static_cast<int>(ctx.p[kPFaEps]));
    float const sc = bf16Round(kvScale);
    uint8_t* pool = ptr<uint8_t>(ctx, kPFaPool);
    uint4 kr[fa::kHkv][2], vr[fa::kHkv][2];
    {
        // the eight vectors through faLoadQkv (polled with kIFlagQkvArena)
        uint8_t const* a[4 * fa::kHkv];
#pragma unroll
        for (int hk = 0; hk < fa::kHkv; ++hk)
        {
            __nv_bfloat16 const* kp = qkv + static_cast<size_t>(row) * fa::kQkvCols + fa::kKCol + hk * fa::kD + 16 * qm;
            __nv_bfloat16 const* vp = qkv + static_cast<size_t>(row) * fa::kQkvCols + fa::kVCol + hk * fa::kD + 16 * qm;
            a[4 * hk + 0] = reinterpret_cast<uint8_t const*>(kp);
            a[4 * hk + 1] = reinterpret_cast<uint8_t const*>(kp + 8);
            a[4 * hk + 2] = reinterpret_cast<uint8_t const*>(vp);
            a[4 * hk + 3] = reinterpret_cast<uint8_t const*>(vp + 8);
        }
        uint4 w[4 * fa::kHkv];
        faLoadQkv<4 * fa::kHkv>(ctx, w, a, (icFaFlags(ctx.p[kPIFlags]) & kIFlagQkvArena) != 0);
#pragma unroll
        for (int hk = 0; hk < fa::kHkv; ++hk)
        {
            kr[hk][0] = w[4 * hk + 0];
            kr[hk][1] = w[4 * hk + 1];
            vr[hk][0] = w[4 * hk + 2];
            vr[hk][1] = w[4 * hk + 3];
        }
    }
#pragma unroll
    for (int hk = 0; hk < fa::kHkv; ++hk)
    {
        uint32_t const kw[8]
            = {kr[hk][0].x, kr[hk][0].y, kr[hk][0].z, kr[hk][0].w, kr[hk][1].x, kr[hk][1].y, kr[hk][1].z, kr[hk][1].w};
        uint32_t const vw[8]
            = {vr[hk][0].x, vr[hk][0].y, vr[hk][0].z, vr[hk][0].w, vr[hk][1].x, vr[hk][1].y, vr[hk][1].z, vr[hk][1].w};
        float kf[16], vf[16];
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            kf[2 * i] = bf16Lo(kw[i]);
            kf[2 * i + 1] = bf16Hi(kw[i]);
            vf[2 * i] = bf16Lo(vw[i]);
            vf[2 * i + 1] = bf16Hi(vw[i]);
        }
        float sums[2];
#pragma unroll
        for (int ln = 0; ln < 2; ++ln)
        {
            float acc = faAddRn(faMulRn(kf[8 * ln], kf[8 * ln]), faMulRn(kf[8 * ln + 1], kf[8 * ln + 1]));
#pragma unroll
            for (int e = 2; e < 8; ++e)
            {
                acc = faAddRn(acc, faMulRn(kf[8 * ln + e], kf[8 * ln + e]));
            }
            sums[ln] = acc;
        }
        float const u = faAddRn(sums[0], sums[1]);
        float const v = faAddRn(u, __shfl_xor_sync(0xffffffffu, u, 1));
        float const total = faAddRn(v, __shfl_xor_sync(0xffffffffu, v, 2));
        float const inv = faRsqrtApprox(faAddRn(faMulRn(total, 1.0f / fa::kD), eps));
        uint32_t kb[4], vb[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            float z[4], y[4];
#pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                z[e] = bf16Round(faMulRn(bf16Round(faMulRn(kf[4 * j + e], inv)), sc));
                y[e] = bf16Round(faMulRn(vf[4 * j + e], sc));
            }
            kb[j] = faE4m3x4(z[0], z[1], z[2], z[3]);
            vb[j] = faE4m3x4(y[0], y[1], y[2], y[3]);
        }
        uint4 const k4 = make_uint4(kb[0], kb[1], kb[2], kb[3]);
        uint4 const v4 = make_uint4(vb[0], vb[1], vb[2], vb[3]);
        *reinterpret_cast<uint4*>(stg + ((row * fa::kHkv + hk) * 2 + 0) * 64 + 16 * qm) = k4;
        *reinterpret_cast<uint4*>(stg + ((row * fa::kHkv + hk) * 2 + 1) * 64 + 16 * qm) = v4;
        if (mine)
        {
            int const rel = page - g.p0;
            size_t const o = static_cast<size_t>(hk * fa::kPage + slot) * 64 + 16 * qm;
            st128(pool + static_cast<size_t>(ids[rel]) * 4096 + o, k4);
            st128(pool + static_cast<size_t>(ids[fa::kMaxUnitPages + rel]) * 4096 + o, v4);
        }
    }
}

// The staged K / V rows on unit page `page` into its landed ring page `rp`; V rows at positions >= L zeroed (the last
// page). The whole warp; then the proxy fence before the MMAs' (async proxy) reads and the slot's next TMA refill.
__device__ __forceinline__ void faPatchPage(uint32_t ring, int rp, int page, FaGeom const& g, uint8_t const* stg)
{
    int const lane = threadIdx.x & 31;
    int const row = lane >> 2;
    int const qm = lane & 3;
    int const pos = g.L - kTok + row;
    int const slot = pos % fa::kPage;
    if (pos / fa::kPage == page)
    {
        uint32_t const at = static_cast<uint32_t>(rp * fa::kBox) + faSw64(slot, qm);
#pragma unroll
        for (int hk = 0; hk < fa::kHkv; ++hk)
        {
            uint4 const k4 = *reinterpret_cast<uint4 const*>(stg + ((row * fa::kHkv + hk) * 2 + 0) * 64 + 16 * qm);
            uint4 const v4 = *reinterpret_cast<uint4 const*>(stg + ((row * fa::kHkv + hk) * 2 + 1) * 64 + 16 * qm);
            faStS128(ring + hk * fa::kRingPart + at, k4);
            faStS128(ring + faVAt(rp, hk, slot, qm), v4);
        }
    }
    if (page == g.P - 1)
    {
        int const tail = g.L - page * fa::kPage; // valid slots of the last page
        int const rowsZero = fa::kPage - tail;
        for (int i = lane; i < 2 * rowsZero * 4; i += 32)
        {
            int const hk = i / (rowsZero * 4);
            int const rr = tail + (i % (rowsZero * 4)) / 4;
            int const c = i % 4;
            faStS128(ring + faVAt(rp, hk, rr, c), make_uint4(0, 0, 0, 0)); // the whole row: any chunk order
        }
    }
    fenceProxyAsyncSmem();
    __syncwarp();
}

// A warp-uniform mbarrier wait: every lane polls and the warp leaves together (the code after it stays converged).
__device__ __forceinline__ void faWaitW(Ctx const& ctx, uint32_t bar, uint32_t parity, int code)
{
    if (!__all_sync(0xffffffffu, mbarTryWait(bar, parity)))
    {
        uint64_t t0 = 0;
        while (!__all_sync(0xffffffffu, mbarTryWait(bar, parity)))
        {
            spinCheck(ctx, t0, code, static_cast<int>(bar), static_cast<int>(parity));
        }
    }
}

// The warpgroup's 128-thread named barrier (ids 1 and 2; id 0 is __syncthreads).
__device__ __forceinline__ void faBarWg(int wg)
{
    asm volatile("bar.sync %0, 128;" ::"r"(1 + wg) : "memory");
}

// ---- kIFlagF3FaPingPong ----
// Named barrier 3 + wg (256 threads: warpgroup wg's 128 syncing, the other warpgroup's 128 arriving) gates warpgroup
// wg's softmax of tile t >= 1 on the other warpgroup's softmax of tile t - 1. Every sync has exactly one matching
// arrival set: the other warpgroup arrives after its tile t - 1 only when tile t exists, so the barrier is idle again
// when the unit ends.
__device__ __forceinline__ void faPpWait(int wg)
{
    asm volatile("bar.sync %0, 256;" ::"r"(3 + wg) : "memory");
}

__device__ __forceinline__ void faPpPass(int wg)
{
    asm volatile("bar.arrive %0, 256;" ::"r"(4 - wg) : "memory");
}

// ---- end kIFlagF3FaPingPong ----

// ---- kIFlagFaIssueOrder ----
// Named barrier 9 + wg (64 threads: warpgroup wg's issuer warp syncing, the other issuer warp arriving) gates chain k
// (issued by warpgroup k % 2 = wg) on chain k - 1 having been issued. Chain k passes only when chain k + 1 exists, and
// chain k + 1 always waits, so each sync has exactly one arrival set and both barriers are idle when the unit ends. A
// phase cannot be re-armed early: the next arrival on barrier 9 + wg comes from chain k + 1, issued after its own wait
// on chain k's pass, which follows chain k's sync.
__device__ __forceinline__ void faIoWait(int wg)
{
    asm volatile("bar.sync %0, 64;" ::"r"(9 + wg) : "memory");
}

__device__ __forceinline__ void faIoPass(int wg)
{
    asm volatile("bar.arrive %0, 64;" ::"r"(10 - wg) : "memory");
}

// MSO_MK_FA_XQ: named barrier 11 + wg (160 threads: warpgroup wg's 128 arriving once their S(t) is in registers, the
// other warpgroup's issuer warp syncing before it issues QK(t + 2) into S_wg). Warpgroup wg arrives at tile t iff
// t + 2 < nt, and the other issuer syncs at its tile t + 1 iff t + 1 >= 1 and t + 2 < nt: one arrival set per sync. The
// next arrival (tile t + 2) follows warpgroup wg's wait for S(t + 2), which the syncing issuer commits after its sync,
// so a phase is never re-armed early.
__device__ __forceinline__ void faXqArrive(int wg)
{
    asm volatile("bar.arrive %0, 160;" ::"r"(11 + wg) : "memory");
}

__device__ __forceinline__ void faXqWait(int wg)
{
    asm volatile("bar.sync %0, 160;" ::"r"(11 + wg) : "memory");
}

// ---- end kIFlagFaIssueOrder / MSO_MK_FA_XQ ----

// Warp-collective tcgen05 issue: every lane of a converged warp executes the statement and elect.sync picks the one
// lane that issues (always the same lane, so its commits track its MMAs). The operands are warp-uniform values.
__device__ __forceinline__ void faMmaSSw(uint32_t dTmem, uint64_t aDesc, uint64_t bDesc, uint32_t idesc, uint32_t acc)
{
    asm volatile(
        "{\n\t.reg .pred p, e;\n\t.reg .b32 x;\n\tsetp.ne.b32 p, %4, 0;\n\telect.sync x|e, 0xffffffff;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}\n" ::"r"(dTmem),
        "l"(aDesc), "l"(bDesc), "r"(idesc), "r"(acc)
        : "memory");
}

__device__ __forceinline__ void faMmaTSw(uint32_t dTmem, uint32_t aTmem, uint64_t bDesc, uint32_t idesc, uint32_t acc)
{
    asm volatile(
        "{\n\t.reg .pred p, e;\n\t.reg .b32 x;\n\tsetp.ne.b32 p, %4, 0;\n\telect.sync x|e, 0xffffffff;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, p;\n\t}\n" ::"r"(dTmem),
        "r"(aTmem), "l"(bDesc), "r"(idesc), "r"(acc)
        : "memory");
}

__device__ __forceinline__ void faCommitW(uint32_t bar)
{
    asm volatile(
        "{\n\t.reg .pred e;\n\t.reg .b32 x;\n\telect.sync x|e, 0xffffffff;\n\t"
        "@e tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n\t}\n" ::"r"(bar)
        : "memory");
}

#if MSO_MK_KM
// ---- MSO_MK_KM (fullAttentionConfig.cuh): one warp-collective statement per MMA batch ----
// A compile-time tile slot for the batch statements below.
template <int kS>
struct KmSlot
{
    static constexpr int kValue = kS;
};

// f(KmSlot<slot>{}) for the run-time slot (warp-uniform, < fa::kSlots): the slot's operand offsets become constants.
template <int kS = 0, class F>
__device__ __forceinline__ void kmForSlot(int slot, F const& f)
{
    if constexpr (kS + 1 < fa::kSlots)
    {
        if (slot == kS)
        {
            f(KmSlot<kS>{});
        }
        else
        {
            kmForSlot<kS + 1>(slot, f);
        }
    }
    else
    {
        f(KmSlot<kS>{});
    }
}

// PV(t) of a full tile in slot kS (4 MMAs M128 N128 K32, A = P at TMEM columns sW + 8 q, B = page q's V of both heads)
// accumulating O at oW (the first one iff acc0 != 0), then the commits on `empty` and `odone`: faMmaTSw's statements
// and faCommitW's, in their order, under one elect.sync (the same lane issues all of them).
template <int kS>
__device__ __forceinline__ void kmPv4(
    uint32_t oW, uint32_t sW, uint64_t dV, uint32_t acc0, uint32_t empty, uint32_t odone)
{
    constexpr uint32_t kV = static_cast<uint32_t>(fa::kTilePages * kS * fa::kVPage);
    asm volatile(
        "{\n\t.reg .pred p, t, e;\n\t.reg .b32 x;\n\tsetp.ne.b32 p, %9, 0;\n\tsetp.eq.b32 t, %9, %9;\n\t"
        "elect.sync x|e, 0xffffffff;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %5, %10, p;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%2], %6, %10, t;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%3], %7, %10, t;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%4], %8, %10, t;\n\t"
        "@e tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%11];\n\t"
        "@e tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%12];\n\t}\n" ::"r"(oW),
        "r"(sW), "r"(sW + 8), "r"(sW + 16), "r"(sW + 24), "l"(dV + static_cast<uint64_t>(kV >> 4)),
        "l"(dV + static_cast<uint64_t>((kV + fa::kVPage) >> 4)),
        "l"(dV + static_cast<uint64_t>((kV + 2 * fa::kVPage) >> 4)),
        "l"(dV + static_cast<uint64_t>((kV + 3 * fa::kVPage) >> 4)), "r"(acc0), "r"(fa::kIdescPv), "r"(empty),
        "r"(odone)
        : "memory");
}

// QK of a tile in slot kS into S at sDst (8 MMAs M128 N128 K32: Q_hi then Q_lo, K steps 0 .. 3, the first one
// overwriting), then the commit on sFullDst: the per-MMA statements of stageFaAttn's qkTile under one elect.sync.
template <int kS>
__device__ __forceinline__ void kmQk8(uint32_t sDst, uint64_t dQ, uint64_t dK, uint32_t sFullDst)
{
    constexpr uint32_t kK = static_cast<uint32_t>(fa::kTilePages * kS * fa::kBox);
    auto a = [&](int j)
    { return dQ + static_cast<uint64_t>(((j >> 2) * 16384 + ((j & 3) >> 1) * 8192 + (j & 1) * 32) >> 4); };
    auto b
        = [&](int j) { return dK + static_cast<uint64_t>((kK + ((j & 3) >> 1) * fa::kRingPart + (j & 1) * 32) >> 4); };
    asm volatile(
        "{\n\t.reg .pred f, t, e;\n\t.reg .b32 x;\n\tsetp.ne.b32 f, %17, %17;\n\tsetp.eq.b32 t, %17, %17;\n\t"
        "elect.sync x|e, 0xffffffff;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %9, %17, f;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %2, %10, %17, t;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %3, %11, %17, t;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %4, %12, %17, t;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %5, %13, %17, t;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %6, %14, %17, t;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %7, %15, %17, t;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %8, %16, %17, t;\n\t"
        "@e tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%18];\n\t}\n" ::"r"(sDst),
        "l"(a(0)), "l"(a(1)), "l"(a(2)), "l"(a(3)), "l"(a(4)), "l"(a(5)), "l"(a(6)), "l"(a(7)), "l"(b(0)), "l"(b(1)),
        "l"(b(2)), "l"(b(3)), "l"(b(4)), "l"(b(5)), "l"(b(6)), "l"(b(7)), "r"(fa::kIdescQk), "r"(sFullDst)
        : "memory");
}

// kmPv4's statement for any slot (MSO_MK_KM bit 8): dVs = the slot's first V page descriptor; the pages q = 1 .. 3
// (kVPage apart) and the A columns sW + 8 q are added inside the statement.
__device__ __forceinline__ void kmPv4r(
    uint32_t oW, uint32_t sW, uint64_t dVs, uint32_t acc0, uint32_t empty, uint32_t odone)
{
    static_assert(fa::kVPage % 16 == 0, "kmPv4r: V pages are 16 B aligned descriptor steps");
#if MSO_MK_KM & 32 // bit 32: each MMA's operands formed right before it in one temporary pair (low register pressure)
    asm volatile(
        "{\n\t.reg .pred p, t, e;\n\t.reg .b32 x, a;\n\t.reg .b64 d;\n\t"
        "setp.ne.b32 p, %3, 0;\n\tsetp.eq.b32 t, %3, %3;\n\t"
        "elect.sync x|e, 0xffffffff;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %4, p;\n\t"
        "add.u32 a, %1, 8;\n\tadd.s64 d, %2, %7;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [a], d, %4, t;\n\t"
        "add.u32 a, %1, 16;\n\tadd.s64 d, %2, %8;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [a], d, %4, t;\n\t"
        "add.u32 a, %1, 24;\n\tadd.s64 d, %2, %9;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [a], d, %4, t;\n\t"
        "@e tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%5];\n\t"
        "@e tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%6];\n\t}\n" ::"r"(oW),
        "r"(sW), "l"(dVs), "r"(acc0), "r"(fa::kIdescPv), "r"(empty), "r"(odone), "n"(fa::kVPage >> 4),
        "n"(2 * fa::kVPage >> 4), "n"(3 * fa::kVPage >> 4)
        : "memory");
#else
    asm volatile(
        "{\n\t.reg .pred p, t, e;\n\t.reg .b32 x, a1, a2, a3;\n\t.reg .b64 d1, d2, d3;\n\t"
        "setp.ne.b32 p, %3, 0;\n\tsetp.eq.b32 t, %3, %3;\n\t"
        "add.u32 a1, %1, 8;\n\tadd.u32 a2, %1, 16;\n\tadd.u32 a3, %1, 24;\n\t"
        "add.s64 d1, %2, %7;\n\tadd.s64 d2, %2, %8;\n\tadd.s64 d3, %2, %9;\n\t"
        "elect.sync x|e, 0xffffffff;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %4, p;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [a1], d1, %4, t;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [a2], d2, %4, t;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [a3], d3, %4, t;\n\t"
        "@e tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%5];\n\t"
        "@e tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%6];\n\t}\n" ::"r"(oW),
        "r"(sW), "l"(dVs), "r"(acc0), "r"(fa::kIdescPv), "r"(empty), "r"(odone), "n"(fa::kVPage >> 4),
        "n"(2 * fa::kVPage >> 4), "n"(3 * fa::kVPage >> 4)
        : "memory");
#endif
}

// kmPv4r's statement for any tile (MSO_MK_KM bit 64): MMA q (q = 1 .. 3) only when q < np (1 <= np <= 4, the tile's
// pages), as the per-MMA path issues them; operands formed right before each MMA. No branch for a tail tile.
__device__ __forceinline__ void kmPv4n(
    uint32_t oW, uint32_t sW, uint64_t dVs, uint32_t acc0, uint32_t empty, uint32_t odone, int np)
{
    static_assert(fa::kVPage % 16 == 0, "kmPv4n: V pages are 16 B aligned descriptor steps");
    asm volatile(
        "{\n\t.reg .pred p, t, e, e1, e2, e3, q;\n\t.reg .b32 x, a;\n\t.reg .b64 d;\n\t"
        "setp.ne.b32 p, %3, 0;\n\tsetp.eq.b32 t, %3, %3;\n\t"
        "elect.sync x|e, 0xffffffff;\n\t"
        "setp.gt.s32 q, %10, 1;\n\tand.pred e1, e, q;\n\t"
        "setp.gt.s32 q, %10, 2;\n\tand.pred e2, e, q;\n\t"
        "setp.gt.s32 q, %10, 3;\n\tand.pred e3, e, q;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %4, p;\n\t"
        "add.u32 a, %1, 8;\n\tadd.s64 d, %2, %7;\n\t"
        "@e1 tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [a], d, %4, t;\n\t"
        "add.u32 a, %1, 16;\n\tadd.s64 d, %2, %8;\n\t"
        "@e2 tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [a], d, %4, t;\n\t"
        "add.u32 a, %1, 24;\n\tadd.s64 d, %2, %9;\n\t"
        "@e3 tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [a], d, %4, t;\n\t"
        "@e tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%5];\n\t"
        "@e tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%6];\n\t}\n" ::"r"(oW),
        "r"(sW), "l"(dVs), "r"(acc0), "r"(fa::kIdescPv), "r"(empty), "r"(odone), "n"(fa::kVPage >> 4),
        "n"(2 * fa::kVPage >> 4), "n"(3 * fa::kVPage >> 4), "r"(np)
        : "memory");
}

// kmQk8's statement for any slot (MSO_MK_KM bit 16): dKs = the slot's first K descriptor; Q's eight descriptors
// (dQ plus constants) and K's four (dKs plus constants, Q_hi and Q_lo read the same K steps) are added inside the
// statement.
__device__ __forceinline__ void kmQk8r(uint32_t sDst, uint64_t dQ, uint64_t dKs, uint32_t sFullDst)
{
    static_assert(fa::kRingPart % 16 == 0, "kmQk8r: K parts are 16 B aligned descriptor steps");
#if MSO_MK_KM & 32 // bit 32: each MMA's operands formed right before it, one temporary pair each
    asm volatile(
        "{\n\t.reg .pred f, t, e;\n\t.reg .b32 x;\n\t.reg .b64 a, b;\n\t"
        "setp.ne.b32 f, %3, %3;\n\tsetp.eq.b32 t, %3, %3;\n\t"
        "elect.sync x|e, 0xffffffff;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, f;\n\t"
        "add.s64 a, %1, 2;\n\tadd.s64 b, %2, 2;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], a, b, %3, t;\n\t"
        "add.s64 a, %1, 512;\n\tadd.s64 b, %2, %5;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], a, b, %3, t;\n\t"
        "add.s64 a, %1, 514;\n\tadd.s64 b, %2, %6;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], a, b, %3, t;\n\t"
        "add.s64 a, %1, 1024;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], a, %2, %3, t;\n\t"
        "add.s64 a, %1, 1026;\n\tadd.s64 b, %2, 2;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], a, b, %3, t;\n\t"
        "add.s64 a, %1, 1536;\n\tadd.s64 b, %2, %5;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], a, b, %3, t;\n\t"
        "add.s64 a, %1, 1538;\n\tadd.s64 b, %2, %6;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], a, b, %3, t;\n\t"
        "@e tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%4];\n\t}\n" ::"r"(sDst),
        "l"(dQ), "l"(dKs), "r"(fa::kIdescQk), "r"(sFullDst), "n"(fa::kRingPart >> 4), "n"((fa::kRingPart + 32) >> 4)
        : "memory");
#else
    asm volatile(
        "{\n\t.reg .pred f, t, e;\n\t.reg .b32 x;\n\t.reg .b64 a1, a2, a3, a4, a5, a6, a7, b1, b2, b3;\n\t"
        "setp.ne.b32 f, %3, %3;\n\tsetp.eq.b32 t, %3, %3;\n\t"
        "add.s64 a1, %1, 2;\n\tadd.s64 a2, %1, 512;\n\tadd.s64 a3, %1, 514;\n\tadd.s64 a4, %1, 1024;\n\t"
        "add.s64 a5, %1, 1026;\n\tadd.s64 a6, %1, 1536;\n\tadd.s64 a7, %1, 1538;\n\t"
        "add.s64 b1, %2, 2;\n\tadd.s64 b2, %2, %5;\n\tadd.s64 b3, %2, %6;\n\t"
        "elect.sync x|e, 0xffffffff;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, f;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], a1, b1, %3, t;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], a2, b2, %3, t;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], a3, b3, %3, t;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], a4, %2, %3, t;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], a5, b1, %3, t;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], a6, b2, %3, t;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], a7, b3, %3, t;\n\t"
        "@e tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%4];\n\t}\n" ::"r"(sDst),
        "l"(dQ), "l"(dKs), "r"(fa::kIdescQk), "r"(sFullDst), "n"(fa::kRingPart >> 4), "n"((fa::kRingPart + 32) >> 4)
        : "memory");
#endif
}

// ---- end MSO_MK_KM ----
#endif

#if MSO_MK_KM_TL
// ---- MSO_MK_KM_TL: the attribution timeline (fullAttentionConfig.cuh; stamped builds) ----
constexpr int kKmTlU0 = 2;    // the first stamped iteration of a warpgroup
constexpr int kKmTlIters = 6; // stamped iterations per warpgroup (2 x 6 x 8 words <= kStampE3Words)
static_assert(2 * kKmTlIters * 8 <= kStampE3Words, "km timeline exceeds the stamp words");

__device__ __forceinline__ void kmTl(Ctx const& ctx, int wg, int u, int k)
{
    if (MSO_MK_STAMPS_DETAIL(ctx) != nullptr && (threadIdx.x & 127) == 0 && u >= kKmTlU0 && u < kKmTlU0 + kKmTlIters)
    {
        MSO_MK_STAMPS(ctx)[kStampE3 + (wg * kKmTlIters + u - kKmTlU0) * 8 + k] = static_cast<uint64_t>(clock64());
    }
}
#if MSO_MK_KM_TL == 2
// MSO_MK_KM_TL = 2: 20 points of iterations 3 and 4 (kKmTl2U0 ..): S seen, softmax start, P stored, wait::st, barrier,
// each PV MMA issued (4), PV commits issued, QK start, each QK MMA issued (8), QK commit issued.
constexpr int kKmTl2U0 = 3;
constexpr int kKmTl2Pts = 20;
static_assert(2 * 2 * kKmTl2Pts <= kStampE3Words, "km timeline 2 exceeds the stamp words");

__device__ __forceinline__ void kmTl2(Ctx const& ctx, int wg, int u, int k)
{
    if (MSO_MK_STAMPS_DETAIL(ctx) != nullptr && (threadIdx.x & 127) == 0 && u >= kKmTl2U0 && u < kKmTl2U0 + 2)
    {
        MSO_MK_STAMPS(ctx)[kStampE3 + (wg * 2 + u - kKmTl2U0) * kKmTl2Pts + k] = static_cast<uint64_t>(clock64());
    }
}

#define KM_TL(k)                                                                                                       \
    if ((k) <= 4)                                                                                                      \
    {                                                                                                                  \
        kmTl2(ctx, wg, u, (k));                                                                                        \
    }
#define KM_TL2(u_, k) kmTl2(ctx, wg, (u_), (k))
#else
#define KM_TL(k) kmTl(ctx, wg, u, k)
#define KM_TL2(u_, k)
#endif
// ---- end MSO_MK_KM_TL ----
#else
#define KM_TL(k)
#define KM_TL2(u_, k)
#endif

#if MSO_MK_FA_ST
// ---- MSO_MK_FA_ST: the FaAttn loop record (fullAttentionConfig.cuh; per-unit stamped builds), raw clock64 words
// at kStampE3 + k except the anchor's %globaltimer ----
namespace fa
{
constexpr int kStU = 12;                  // stamped iterations u < kStU of each warpgroup
constexpr int kStWg = 3 * kStU;           // words of warpgroup wg at wg * kStWg: S seen [u], barrier passed [kStU + u],
                                          // the issuer's slot-full wait return of QK(t + 2) [2 kStU + u]
constexpr int kStAnchor = 2 * kStWg;      // 72 .. 74: thread 0 at Q ready: clock64, %globaltimer, clock64 after it
constexpr int kStLoopEnd = kStAnchor + 3; // 75 + wg: the warpgroup's loop end (thread 128 wg)
constexpr int kStDrain = kStLoopEnd + 2;  // 77: thread 0: the epilogue's PV wait returned
constexpr int kStParts = kStDrain + 1;    // 78: thread 0: the partials stored
constexpr int kStGeom = kStParts + 1;     // 79: nt | n << 16 | unit << 32 (raw)
constexpr int kStFull0 = kStGeom + 1;     // 80 + wg: the slot-full wait return of QK(wg), before the loop
constexpr int kStEmpty = kStFull0 + 2;    // 82 + wg: lane 0 of warp 4 wg + 1: its refill empty-wait cycles, summed
constexpr int kStWords = kStEmpty + 2;    // 84
// The FaAttn epilogue's end and the FaMerge unit that follows FaAttn on the same CTA (same SM clock as the anchor).
constexpr int kStFaSync = kStWords;       // 84: FaAttn thread 0 after the epilogue's __syncthreads
constexpr int kStFaFree = kStFaSync + 1;  // 85: FaAttn warp 0 lane 0 after the TMEM dealloc
constexpr int kStFmStart = kStFaFree + 1; // 86: FaMerge thread 0 at entry
constexpr int kStFmGate = kStFmStart + 1; // 87: FaMerge warp 0 lane 0 with its row's gate word
constexpr int kStFmPolled
    = kStFmGate + 1; // 88 + w: FaMerge warp w lane 0, its polls complete; bits 56 .. 63: its rounds
constexpr int kStFmBar = kStFmPolled + kWarps; // 96: FaMerge thread 0 after the polls' barrier
constexpr int kStFmSum = kStFmBar + 1;         // 97: FaMerge thread 0 after the warps' sums barrier
constexpr int kStFmEnd = kStFmSum + 1;         // 98: FaMerge thread 0 at the end (slots emptied)
constexpr int kStFmTicket = kStFmEnd + 1;      // 99: MSO_MK_LM bit 1: thread 0 with its arrival ticket (bits 56 .. 63)
constexpr int kStWordsAll = kStFmTicket + 1;   // 100
} // namespace fa

static_assert(fa::kStWordsAll <= kStampE3Words, "MSO_MK_FA_ST: the FaAttn record exceeds the stamp words");

__device__ __forceinline__ void faSt(Ctx const& ctx, int k, uint64_t v)
{
    MSO_MK_STAMPS(ctx)[kStampE3 + k] = v;
}

// ---- end MSO_MK_FA_ST ----
#endif

// Two floats as one 64-bit operand of the packed f32x2 instructions.
__device__ __forceinline__ uint64_t faPack2(float a, float b)
{
    uint64_t r;
    asm("mov.b64 %0, {%1, %2};" : "=l"(r) : "f"(a), "f"(b));
    return r;
}

// (a, b) = (a, b) * s2 + c2, one rounding per element (fma.rn.f32x2: the bits of two fma.rn.f32).
__device__ __forceinline__ void faFma2(float& a, float& b, uint64_t s2, uint64_t c2)
{
    asm("{\n\t.reg .b64 x;\n\tmov.b64 x, {%0, %1};\n\tfma.rn.f32x2 x, x, %2, %3;\n\tmov.b64 {%0, %1}, x;\n\t}\n"
        : "+f"(a), "+f"(b)
        : "l"(s2), "l"(c2));
}

// (a, b) += (c, d) (add.rn.f32x2: the bits of two add.rn.f32).
__device__ __forceinline__ void faAdd2(float& a, float& b, float c, float d)
{
    asm("{\n\t.reg .b64 x, y;\n\tmov.b64 x, {%0, %1};\n\tmov.b64 y, {%2, %3};\n\tadd.rn.f32x2 x, x, y;\n\t"
        "mov.b64 {%0, %1}, x;\n\t}\n"
        : "+f"(a), "+f"(b)
        : "f"(c), "f"(d));
}

__device__ __forceinline__ void faSt16(uint32_t taddr, uint32_t const (&w)[16])
{
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
        "%15, %16};" ::"r"(taddr),
        "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3]), "r"(w[4]), "r"(w[5]), "r"(w[6]), "r"(w[7]), "r"(w[8]), "r"(w[9]),
        "r"(w[10]), "r"(w[11]), "r"(w[12]), "r"(w[13]), "r"(w[14]), "r"(w[15])
        : "memory");
}

// 128 columns of this warp's 32 TMEM lanes (one tcgen05.ld 32x32b.x128), waited in the same statement.
__device__ __forceinline__ void faLd128(uint32_t taddr, float (&v)[128])
{
    uint32_t r[128];
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x128.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
        "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, "
        "%33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, "
        "%53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, "
        "%73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, "
        "%93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, "
        "%110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, "
        "%126, %127}, [%128];\n\ttcgen05.wait::ld.sync.aligned;"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]),
        "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]),
        "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]), "=r"(r[32]),
        "=r"(r[33]), "=r"(r[34]), "=r"(r[35]), "=r"(r[36]), "=r"(r[37]), "=r"(r[38]), "=r"(r[39]), "=r"(r[40]),
        "=r"(r[41]), "=r"(r[42]), "=r"(r[43]), "=r"(r[44]), "=r"(r[45]), "=r"(r[46]), "=r"(r[47]), "=r"(r[48]),
        "=r"(r[49]), "=r"(r[50]), "=r"(r[51]), "=r"(r[52]), "=r"(r[53]), "=r"(r[54]), "=r"(r[55]), "=r"(r[56]),
        "=r"(r[57]), "=r"(r[58]), "=r"(r[59]), "=r"(r[60]), "=r"(r[61]), "=r"(r[62]), "=r"(r[63]), "=r"(r[64]),
        "=r"(r[65]), "=r"(r[66]), "=r"(r[67]), "=r"(r[68]), "=r"(r[69]), "=r"(r[70]), "=r"(r[71]), "=r"(r[72]),
        "=r"(r[73]), "=r"(r[74]), "=r"(r[75]), "=r"(r[76]), "=r"(r[77]), "=r"(r[78]), "=r"(r[79]), "=r"(r[80]),
        "=r"(r[81]), "=r"(r[82]), "=r"(r[83]), "=r"(r[84]), "=r"(r[85]), "=r"(r[86]), "=r"(r[87]), "=r"(r[88]),
        "=r"(r[89]), "=r"(r[90]), "=r"(r[91]), "=r"(r[92]), "=r"(r[93]), "=r"(r[94]), "=r"(r[95]), "=r"(r[96]),
        "=r"(r[97]), "=r"(r[98]), "=r"(r[99]), "=r"(r[100]), "=r"(r[101]), "=r"(r[102]), "=r"(r[103]), "=r"(r[104]),
        "=r"(r[105]), "=r"(r[106]), "=r"(r[107]), "=r"(r[108]), "=r"(r[109]), "=r"(r[110]), "=r"(r[111]), "=r"(r[112]),
        "=r"(r[113]), "=r"(r[114]), "=r"(r[115]), "=r"(r[116]), "=r"(r[117]), "=r"(r[118]), "=r"(r[119]), "=r"(r[120]),
        "=r"(r[121]), "=r"(r[122]), "=r"(r[123]), "=r"(r[124]), "=r"(r[125]), "=r"(r[126]), "=r"(r[127])
        : "r"(taddr)
        : "memory");
#pragma unroll
    for (int i = 0; i < 128; ++i)
    {
        v[i] = __uint_as_float(r[i]);
    }
}

// Stamped builds (thread 0 = warpgroup 0's row 0, lane 0 of its issuer warp): sub-stamps 0 Q ready, 1 first QKs
// issued, 2 + t the S-wait return of tile t < 10 (by the tile's warpgroup), 14 / 24 warpgroup 0 / 1 loop end, 15
// partials stored; accumulated clock64 values 12 MMA issue (PV + QK, warpgroup 0), 13 TMA refill issue, 16 waits for a
// slot's TMA, 17 refill waits (PV complete), 18 rescales, 19 the epilogue's PV wait, 20 tiles, 21 softmax (S loaded ->
// P stored), 22 S waits, 23 warpgroup barrier.
__device__ __forceinline__ void faStampWg(Ctx const& ctx, int k)
{
    if (MSO_MK_STAMPS_DETAIL(ctx) != nullptr && (threadIdx.x & 127) == 0)
    {
        MSO_MK_STAMPS(ctx)[kStampSub + k] = stampTime(); // MSO_MK_SY_CLK picks the clock
    }
}

// FaAttn: one unit's pages in tiles of 4, the two warpgroups taking alternate tiles (see the file comment).
__device__ __noinline__ void stageFaAttn(Ctx const& ctx, int unit, uint32_t off, Io const& io)
{
    icFaCheck(ctx.p[kPIFlags]); // the full-attention kPIFlags bits this build fixes
#if MSO_MK_LD_FG
    ldFgCheck(ctx);
#endif
    int const tid = threadIdx.x;
    // Warp-uniform values (broadcast from lane 0) wherever an MMA or TMA operand derives from them.
    int const warp = __shfl_sync(0xffffffffu, tid >> 5, 0);
    int const wg = warp >> 2;
    int const r = tid & 127;
    int const lane = tid & 31;
    uint32_t const base = __shfl_sync(0xffffffffu, ctx.smemBase + off, 0);
    uint8_t* region = ctx.smem + off;
    int const* scal = reinterpret_cast<int const*>(region + fa::kOffScal);
    int const* ids = reinterpret_cast<int const*>(region + fa::kOffIds);
    float* xch = reinterpret_cast<float*>(ctx.smem + kSmemRed); // [2][128] Q norm halves, then [2 wg][m, l][128]
    FaGeom g;
    g.L = __shfl_sync(0xffffffffu, scal[0], 0);
    g.p0 = __shfl_sync(0xffffffffu, scal[1], 0);
    g.p1 = __shfl_sync(0xffffffffu, scal[2], 0);
    g.eff = scal[3];
    g.P = __shfl_sync(0xffffffffu, scal[4], 0);
    int const n = g.p1 - g.p0;
    if (n == 0)
    {
        return; // a key-less unit: the merge reads only the eff keyed units' slots
    }
    unit = faUnitOfSlot(ctx, unit, __shfl_sync(0xffffffffu, g.eff, 0));    // the partial slots of the unit computed
    int const nt = (n + fa::kTilePages - 1) / fa::kTilePages;
    int const units = static_cast<int>(MSO_MK_FG_UNITS(ctx.p[kPFaUnits])); // MSO_MK_LD_FG
    // ---- the raw projection in the arena (kIFlagQkvArena, polled) or io.qkv ----
    bool const qkvPolled = (icFaFlags(ctx.p[kPIFlags]) & kIFlagQkvArena) != 0;
    __nv_bfloat16 const* qkv = qkvPolled ? reinterpret_cast<__nv_bfloat16 const*>(ctx.arena + kArFaQkv)
                                         : static_cast<__nv_bfloat16 const*>(io.qkv);
    // ---- end kIFlagQkvArena ----
    int const h = r >> 6;
    int const tok = (r & 63) >> 3;
    int const hq = h * fa::kG + (r & 7);
    float const kvScale = __int_as_float(scal[5]);
    float const dequant = 1.0f / kvScale;
    float const sc = fa::kSmScaleLog2 * dequant;
    subStamp(ctx, 25);

    uint8_t* stg = ctx.smem + kSmemRed + 4096;            // the step's staged K / V rows (faAppendPrep), 2 KiB
    bool const appends = g.p1 > (g.L - kTok) / fa::kPage; // this unit's range holds a page of the step's rows

    // ---- kIFlagFaEntryAtQ: warp 0 requests the ring tiles the prologue left (FaEntryPages < the ring) as soon as
    // its Q words landed, before the norm ----
    bool const entryAtQ = (icFaFlags(ctx.p[kPIFlags]) & kIFlagFaEntryAtQ) != 0;
    auto entryRest = [&]()
    {
        if (entryAtQ && warp == 0)
        {
            int const ntq = (n + fa::kTilePages - 1) / fa::kTilePages;
            uint64_t const pol = policyEvictFirst();
            for (int t = faEntryTiles(ctx, ntq); t < fa::kSlots && t < ntq; ++t)
            {
                faIssueTile(ctx, base, ids, t, n, pol);
            }
        }
    };
    // ---- end kIFlagFaEntryAtQ ----
    // Q prep (the stock fold kernel's statements): thread (row r, half wg) loads its whole row (the norm's sum in the
    // stock's order: lane sums of 8, the two halves of 32, their sum) and keeps elements [32 wg, +32).
    {
        __nv_bfloat16 const* qp = qkv + static_cast<size_t>(tok) * fa::kQkvCols + hq * (2 * fa::kD);
        // kIFlagFaQRep: the replica, [tok][head][64]
        if (qkvPolled && (icFaFlags(ctx.p[kPIFlags]) & kIFlagFaQRep) && (blockIdx.x & 1))
        {
            qp = reinterpret_cast<__nv_bfloat16 const*>(ctx.arena + kArAhQ) + (tok * fa::kHq + hq) * fa::kD;
        }
        // A chunk of 8 BF16: its sum of squares in the stock's order.
        auto chunkSq = [](uint4 q)
        {
            uint32_t const ws[4] = {q.x, q.y, q.z, q.w};
            float x[8];
#pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                x[2 * e] = bf16Lo(ws[e]);
                x[2 * e + 1] = bf16Hi(ws[e]);
            }
            float acc = faAddRn(faMulRn(x[0], x[0]), faMulRn(x[1], x[1]));
#pragma unroll
            for (int e = 2; e < 8; ++e)
            {
                acc = faAddRn(acc, faMulRn(x[e], x[e]));
            }
            return acc;
        };
        uint4 qh[4]; // this thread's half row: chunks 4 wg .. 4 wg + 3
        float total;
        // ---- kIFlagFaQPrep: the prepared Q_hi / Q_lo bytes of row r, chunks 2 wg, 2 wg + 1 ----
        bool const qPrep = qkvPolled && (icFaFlags(ctx.p[kPIFlags]) & kIFlagFaQPrep);
        if (qPrep)
        {
            uint4 w[4];
            {
                uint8_t const* qa = ctx.arena + kArAhQ + 64 * r + 32 * wg;
                uint8_t const* a[4] = {qa, qa + 16, qa + 8192, qa + 8192 + 16};
                faLoadQkv<4>(ctx, w, a, true);
            }
            stampInput(ctx);
            entryRest();
            if (appends && warp == kWarps - 1)
            {
                faAppendPrep(ctx, g, qkv, ids, stg, kvScale);
            }
            subStamp(ctx, 26);
#pragma unroll
            for (int t = 0; t < 2; ++t)
            {
                uint32_t const qb = base + fa::kOffQ + t * 16384;
#pragma unroll
                for (int cl = 0; cl < 2; ++cl)
                {
                    int const c = 2 * wg + cl;
                    faStS128(qb + h * 8192 + faSw64(r, c), w[2 * t + cl]);
                    faStS128(qb + (1 - h) * 8192 + faSw64(r, c), make_uint4(0, 0, 0, 0));
                }
            }
            fenceProxyAsyncSmem();
        }
        else if ((icFaFlags(ctx.p[kPIFlags]) & kIFlagF3FaQHalf) == 0) // ---- end kIFlagFaQPrep ----
        {
            uint4 qw[8];
            {
                uint8_t const* a[8]; // through faLoadQkv
#pragma unroll
                for (int c = 0; c < 8; ++c)
                {
                    a[c] = reinterpret_cast<uint8_t const*>(qp + 8 * c);
                }
                faLoadQkv<8>(ctx, qw, a, qkvPolled);
            }
            stampInput(ctx); // the unit's Q words landed (thread 0's view)
            entryRest();     // kIFlagFaEntryAtQ
            if (appends && warp == kWarps - 1)
            {
                faAppendPrep(ctx, g, qkv, ids, stg, kvScale);
            }
            float ls[8];
#pragma unroll
            for (int ln = 0; ln < 8; ++ln)
            {
                ls[ln] = chunkSq(qw[ln]);
            }
            total = faAddRn(faAddRn(faAddRn(ls[0], ls[1]), faAddRn(ls[2], ls[3])),
                faAddRn(faAddRn(ls[4], ls[5]), faAddRn(ls[6], ls[7])));
#pragma unroll
            for (int c = 0; c < 4; ++c)
            {
                qh[c] = wg == 0 ? qw[c] : qw[4 + c];
            }
        }
        else // ---- kIFlagF3FaQHalf ----
        {
            {
                uint8_t const* a[4];
#pragma unroll
                for (int c = 0; c < 4; ++c)
                {
                    a[c] = reinterpret_cast<uint8_t const*>(qp + 8 * (4 * wg + c));
                }
                faLoadQkv<4>(ctx, qh, a, qkvPolled);
            }
            stampInput(ctx);
            entryRest(); // kIFlagFaEntryAtQ
            float const half
                = faAddRn(faAddRn(chunkSq(qh[0]), chunkSq(qh[1])), faAddRn(chunkSq(qh[2]), chunkSq(qh[3])));
            xch[wg * fa::kRows + r] = half;
            asm volatile("bar.sync %0, 64;" ::"r"(5 + (warp & 3)) : "memory"); // warps w and w + 4
            float const other = xch[(1 - wg) * fa::kRows + r];
            total = wg == 0 ? faAddRn(half, other) : faAddRn(other, half);
            if (appends && warp == kWarps - 1)
            {
                faAppendPrep(ctx, g, qkv, ids, stg, kvScale);
            }
        }           // ---- end kIFlagF3FaQHalf ----
        if (!qPrep) // the raw q's prep (with kIFlagFaQPrep QkvSum does it)
        {
            if (MSO_MK_STAMPS_DETAIL(ctx) != nullptr && tid == 0 && total == -1.f)
            {
                MSO_MK_STAMPS(ctx)[kStampSub + 31] = 1; // never: keeps the stamp below after the loads' use
            }
            subStamp(ctx, 26);
            float xf[32];
#pragma unroll
            for (int c = 0; c < 4; ++c)
            {
                uint4 const w = qh[c]; // this thread's half
                uint32_t const ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
                for (int e = 0; e < 4; ++e)
                {
                    xf[8 * c + 2 * e] = bf16Lo(ws[e]);
                    xf[8 * c + 2 * e + 1] = bf16Hi(ws[e]);
                }
            }
            float const eps = __int_as_float(static_cast<int>(ctx.p[kPFaEps]));
            float const qs = __int_as_float(static_cast<int>(ctx.p[kPFaQScale]));
            float const inv = faRsqrtApprox(faAddRn(faMulRn(total, 1.0f / fa::kD), eps));
            uint32_t hi[8], lo[8];
#pragma unroll
            for (int e4 = 0; e4 < 8; ++e4)
            {
                float v[4], hv[4];
#pragma unroll
                for (int e = 0; e < 4; ++e)
                {
                    v[e] = bf16Round(faMulRn(bf16Round(faMulRn(xf[4 * e4 + e], inv)), qs));
                    hv[e] = faSplitHi(v[e]);
                }
                hi[e4] = faE4m3x4(hv[0], hv[1], hv[2], hv[3]);
                lo[e4] = faE4m3x4(v[0] - hv[0], v[1] - hv[1], v[2] - hv[2], v[3] - hv[3]);
            }
            // Row r: K atom h holds the row's 64 bytes, atom 1 - h zeros (block-diagonal); this thread's chunks 2 wg,
            // 2 wg + 1.
#pragma unroll
            for (int t = 0; t < 2; ++t)
            {
                uint32_t const qb = base + fa::kOffQ + t * 16384;
                uint32_t const* w = t == 0 ? hi : lo;
#pragma unroll
                for (int cl = 0; cl < 2; ++cl)
                {
                    int const c = 2 * wg + cl;
                    faStS128(qb + h * 8192 + faSw64(r, c),
                        make_uint4(w[4 * cl], w[4 * cl + 1], w[4 * cl + 2], w[4 * cl + 3]));
                    faStS128(qb + (1 - h) * 8192 + faSw64(r, c), make_uint4(0, 0, 0, 0));
                }
            }
            fenceProxyAsyncSmem();
        }
    }
    faFenceBefore();
    __syncthreads();
    faFenceAfter();
    uint32_t const tmem = __shfl_sync(0xffffffffu, *reinterpret_cast<uint32_t volatile*>(region + fa::kOffTmem), 0);
    subStamp(ctx, 0);
    stampWeights(ctx); // Q ready in shared memory (the unit's "weights" stamp slot, otherwise unused by FaAttn)
#if MSO_MK_FA_ST       // the record's anchor and geometry
    bool const faStOn = MSO_MK_STAMPS(ctx) != nullptr;
    if (faStOn && tid == 0)
    {
        uint64_t const c0 = clock64();
        uint64_t const gt = globalTimer();
        uint64_t const c1 = clockAfter(gt);
        faSt(ctx, fa::kStAnchor, c0);
        faSt(ctx, fa::kStAnchor + 1, gt);
        faSt(ctx, fa::kStAnchor + 2, c1);
        faSt(ctx, fa::kStGeom,
            static_cast<uint64_t>(nt) | (static_cast<uint64_t>(n) << 16) | (static_cast<uint64_t>(unit) << 32));
    }
    uint64_t faStEmpty = 0; // lane 0 of warp 4 wg + 1: its refill empty-wait cycles
#endif
    FaProf prof;

    bool const issuerWarp = (warp & 3) == 0; // warp 4 wg: its warpgroup's MMAs
    int const firstNew = (g.L - kTok) / fa::kPage;
    uint64_t const policy = policyEvictFirst();
    int const pf = static_cast<int>(MSO_MK_FG_PREFETCH(ctx.p[kPFaPrefetch])); // MSO_MK_LD_FG
    uint32_t const ring = base + fa::kOffRing;
    // TMEM: O_wg at columns 128 wg, S_wg at 256 + 128 wg (P: its first 32 columns).
    uint32_t const oW = tmem + 128 * wg;
    uint32_t const sW = tmem + 256 + 128 * wg;
    uint32_t const laneBase = static_cast<uint32_t>(32 * (warp & 3)) << 16;
    uint64_t const dQ = faDesc(base + fa::kOffQ, 1, 32);
    uint64_t const dK = faDesc(ring, 1, 32);
    // V (MN-major, SW64): the two heads' 64-column atoms LBO apart, 8-key groups 512 B apart.
    uint64_t const dV = faDesc(ring + fa::kOffV, (fa::kVBox ? fa::kBox : fa::kRingPart) >> 4, 512 >> 4);
    uint32_t const sFull = base + fa::kOffSFull + 8 * wg;

    // QK of tile t into S of warpgroup t % 2 (an issuer warp, converged: the tile's own warpgroup's, or with
    // MSO_MK_FA_XQ the other one's): its slot landed, its step pages patched, then 8 MMAs (Q_hi then Q_lo, K steps
    // 0 .. 3: the stock's order) and the commit on that warpgroup's s_full.
#if MSO_MK_KM_TL == 2
    int kmTlU = -1; // MSO_MK_KM_TL = 2: the caller's iteration while it issues QK(t + 2)
#endif
    auto qkTile = [&](int t)
    {
        uint32_t const sDst = tmem + 256 + 128 * (t & 1);             // = sW for the tile's own warpgroup
        uint32_t const sFullDst = base + fa::kOffSFull + 8 * (t & 1); // = sFull likewise
        int const slot = t % fa::kSlots;
        long long const tf = faClk(ctx);
        faWaitW(ctx, base + fa::kOffFull + 8 * slot, (t / fa::kSlots) & 1, kCodeFaFull);
#if MSO_MK_FA_ST // record: the issuer lane of the tile's warpgroup (t & 1); QK(t) follows iteration (t >> 1) - 1's PV
        if (faStOn && (threadIdx.x & 127) == 0)
        {
            int const uu = (t >> 1) - 1;
            if (uu < 0)
            {
                faSt(ctx, fa::kStFull0 + (t & 1), clock64());
            }
            else if (uu < fa::kStU)
            {
                faSt(ctx, (t & 1) * fa::kStWg + 2 * fa::kStU + uu, clock64());
            }
        }
#endif
        KM_TL2(kmTlU, 10);
        long long const tq = faClk(ctx);
        prof.full += tq - tf;
        int const left = n - fa::kTilePages * t;
        int const np = left < fa::kTilePages ? left : fa::kTilePages;
#if MSO_MK_KM & 16 // bit 16: the tile's pages are consecutive: one compare for "some page is at or past firstNew"
        if (np > 0 && g.p0 + fa::kTilePages * t + np - 1 >= firstNew)
#endif
        {
            for (int q = 0; q < np; ++q)
            {
                int const rel = fa::kTilePages * t + q;
                if (g.p0 + rel >= firstNew)
                {
                    faPatchPage(ring, fa::kTilePages * slot + q, g.p0 + rel, g, stg);
                }
            }
        }
        __syncwarp();
#if (MSO_MK_FA_DBG & 512) == 0 // attribution build (timing only): bit 9 drops the two tcgen05 fences before the MMAs
        faFenceAfter();
#endif
#if MSO_MK_KM & 16 // bit 16: the 8 MMAs and the commit as one statement, one code copy for every slot
        static_assert(fa::kBox % 16 == 0, "km: the slot's K offset is a whole descriptor step");
        kmQk8r(sDst, dQ, dK + static_cast<uint64_t>((fa::kTilePages * slot * fa::kBox) >> 4), sFullDst);
#elif MSO_MK_KM & 2 // bit 2: the 8 MMAs and the commit as one statement
        kmForSlot(slot, [&](auto ks) { kmQk8<decltype(ks)::kValue>(sDst, dQ, dK, sFullDst); });
#else
        uint32_t const kOff = static_cast<uint32_t>(fa::kTilePages * slot * fa::kBox);
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            int const tt = j >> 2;
            int const ks = j & 3;
            uint64_t const a = dQ + static_cast<uint64_t>((tt * 16384 + (ks >> 1) * 8192 + (ks & 1) * 32) >> 4);
            uint64_t const b = dK + static_cast<uint64_t>((kOff + (ks >> 1) * fa::kRingPart + (ks & 1) * 32) >> 4);
#if (MSO_MK_FA_DBG & 1) == 0
            if ((MSO_MK_FA_DBG & 8) == 0 || tt == 0) // attribution build: bit 3 skips the Q_lo MMAs
            {
                faMmaSSw(sDst, a, b, fa::kIdescQk, j != 0 ? 1u : 0u);
            }
#endif
            KM_TL2(kmTlU, 11 + j);
        }
        faCommitW(sFullDst);
        KM_TL2(kmTlU, 19);
#endif // MSO_MK_KM
        prof.mma += faClk(ctx) - tq;
    };
    bool const issueOrder = (icFaFlags(ctx.p[kPIFlags]) & kIFlagFaIssueOrder) != 0 && nt >= 2 && !MSO_MK_FA_XQ;
#if MSO_MK_FA_XQ // P_wg in shared memory
    uint32_t const pW = base + fa::kOffP + fa::kPBytes * wg;
    uint64_t const dP = faDesc(pW, 1, 32);
    uint32_t const oDoneW = base + fa::kOffODone + 8 * wg;
#endif
    if (issuerWarp)
    {
        if (wg == 0 && !entryAtQ) // with kIFlagFaEntryAtQ already requested after the Q words landed
        {
            for (int t = faEntryTiles(ctx, nt); t < fa::kSlots && t < nt; ++t)
            {
                faIssueTile(ctx, base, ids, t, n, policy);
            }
        }
        if (wg < nt)
        {
            if (issueOrder && wg == 1) // kIFlagFaIssueOrder: chain 1 after chain 0
            {
                faIoWait(wg);
            }
            qkTile(wg);
            if (issueOrder) // chains 0 and 1 always have a successor (chains 2 and 3 exist for nt >= 2)
            {
                faIoPass(wg);
            }
        }
    }
    subStamp(ctx, 1);

    // Keys [k0, k0 + 128) of tile t are visible up to the row's position and inside the unit.
    int const vis = g.L - kTok + tok < fa::kPage * g.p1 - 1 ? g.L - kTok + tok : fa::kPage * g.p1 - 1;
    int const myN = nt > wg ? (nt - wg + 1) / 2 : 0;
    uint64_t const sc2 = faPack2(sc, sc);
    bool const pingPong = (icFaFlags(ctx.p[kPIFlags]) & kIFlagF3FaPingPong) != 0 && nt >= kF3FaPpMinTiles;
    bool const ppLate = pingPong && (icFaFlags(ctx.p[kPIFlags]) & kIFlagF3FaPpLate) != 0;
    bool const ppEarly = pingPong && !ppLate;
    bool const refillSib = (icFaFlags(ctx.p[kPIFlags]) & kIFlagF3FaRefillSib) != 0;
    float m = faNegInf();
    float l = 0.f;
    int rescales = 0;
    for (int u = 0; u < myN; ++u)
    {
        int const t = wg + 2 * u;
        int const slot = t % fa::kSlots;
        long long const ts = faClk(ctx);
        faWaitW(ctx, sFull, u & 1, kCodeFaS);
#if MSO_MK_FA_ST // the loop record
        if (faStOn && r == 0 && u < fa::kStU)
        {
            faSt(ctx, wg * fa::kStWg + u, clock64());
        }
#endif
        long long const ts1 = faClk(ctx);
        prof.sw += ts1 - ts;
        if (t < 7)
        {
            faStampWg(ctx, 2 + t);
        }
        KM_TL(0);
#if MSO_MK_FA_XQ // the other warpgroup's QK(t + 1) once its S(t - 1) is in registers (it runs that softmax now)
        if (issuerWarp && t >= 1 && t + 1 < nt)
        {
            faXqWait(1 - wg);
            faFenceAfter();
            qkTile(t + 1);
        }
#endif
        if (ppEarly && t > 0) // after the other warpgroup's softmax of tile t - 1
        {
            faPpWait(wg);
        }
        KM_TL(1);
        faFenceAfter();
        float v[128];
#if (MSO_MK_FA_DBG & 32) == 0
        faLd128(sW + laneBase, v);
#else // attribution build (timing only, wrong results): no S load, row-dependent stand-in scores
#pragma unroll
        for (int k = 0; k < fa::kTileKeys; ++k)
        {
            v[k] = static_cast<float>((r + 3 * k + t) & 15) * 0.25f;
        }
#endif
#if MSO_MK_FA_XQ // S_wg is free for QK(t + 2) (P goes to shared memory)
        if (t + 2 < nt)
        {
            faFenceBefore();
            faXqArrive(wg);
        }
#endif
        long long const tl = faClk(ctx);
        prof.ld += tl - ts1;
#if (MSO_MK_FA_DBG & 2) != 0
#pragma unroll
        for (int k = 0; k < fa::kTileKeys; ++k)
        {
            v[k] = 0.f; // attribution build: the softmax below reduces to its stores
        }
#endif
        int const lim = vis - (fa::kPage * g.p0 + fa::kTileKeys * t);
        if (lim < fa::kTileKeys - 1)
        {
            asm volatile("" ::: "memory"); // keeps this a branch: the masking runs on edge tiles only
#pragma unroll
            for (int k = 0; k < fa::kTileKeys; ++k)
            {
                v[k] = k <= lim ? v[k] : faNegInf();
            }
        }
        float mx0 = faFmax3(v[0], v[1], v[2]);
        float mx1 = faFmax3(v[3], v[4], v[5]);
        float mx2 = faFmax3(v[6], v[7], v[8]);
        float mx3 = faFmax3(v[9], v[10], v[11]);
#pragma unroll
        for (int j = 12; j < fa::kTileKeys - 4; j += 8)
        {
            mx0 = faFmax3(mx0, v[j], v[j + 1]);
            mx1 = faFmax3(mx1, v[j + 2], v[j + 3]);
            mx2 = faFmax3(mx2, v[j + 4], v[j + 5]);
            mx3 = faFmax3(mx3, v[j + 6], v[j + 7]);
        }
        mx0 = faFmax3(mx0, v[fa::kTileKeys - 4], v[fa::kTileKeys - 3]);
        mx1 = faFmax3(mx1, v[fa::kTileKeys - 2], v[fa::kTileKeys - 1]);
        float const mNew = faFmax3(faFmax3(mx0, mx1, mx2), mx3, m);
        bool const need = (mNew - m) * sc > fa::kTau;
        bool const rescale = need && m != faNegInf();
        // O_wg holds this warpgroup's tiles < t: PV(t - 2) completed before S(t) was committed (same issuing thread).
#if MSO_MK_FA_XQ // S(t) came from the other issuer, so wait for PV(t - 2) (O_wg and P_wg are then free)
        if (u > 0)
        {
            faWaitW(ctx, oDoneW, (u - 1) & 1, kCodeFaEmpty);
            faFenceAfter();
        }
#endif
        // tcgen05.ld / st are warp-collective: the whole warp takes the branch; other rows multiply by 1.
        if (__any_sync(0xffffffffu, rescale))
        {
            float const factor = rescale ? faEx2((m - mNew) * sc) : 1.f;
#if MSO_MK_FA_RC == 16 // 16-column chunks (fewer live registers; the same multiply per element)
#pragma unroll
            for (int c = 0; c < 4; ++c)
            {
                float o[16];
                uint32_t const ta = oW + 64 * h + 16 * c + laneBase;
                faLd16(ta, o);
#pragma unroll
                for (int j = 0; j < 16; ++j)
                {
                    o[j] *= factor;
                }
                faSt16f(ta, o);
            }
#else
#pragma unroll
            for (int c = 0; c < 2; ++c)
            {
                float o[32];
                uint32_t const ta = oW + 64 * h + 32 * c + laneBase;
                faLd32(ta, o);
#pragma unroll
                for (int j = 0; j < 32; ++j)
                {
                    o[j] *= factor;
                }
                faSt32(ta, o);
            }
#endif
            l *= factor;
            ++rescales;
        }
        m = need ? mNew : m;
        float const mSafe = m == faNegInf() ? 0.f : m;
        float const bias = fa::kPre - mSafe * sc;
        if (ppLate && t > 0) // the exps after the other warpgroup's exps of tile t - 1
        {
            faPpWait(wg);
        }
        uint64_t const b2 = faPack2(bias, bias);
        // P = exp2(s sc + bias), E4M3, in two halves of 64 keys (S columns 0 .. 15, then 16 .. 31: keys 4 c .. 4 c + 3
        // in column c); the denominator sums the unrounded P in the stock's order.
#pragma unroll
        for (int hh = 0; hh < 2; ++hh)
        {
#pragma unroll
            for (int j = 64 * hh; j < 64 * hh + 64; j += 2)
            {
                faFma2(v[j], v[j + 1], sc2, b2);
#if (MSO_MK_FA_DBG & 4) != 0 // attribution build (timing only, wrong results): the exps as one FMUL each
                v[j] = faMulRn(v[j], 0.5f);
                v[j + 1] = faMulRn(v[j + 1], 0.5f);
#else
                v[j] = faEx2(v[j]);
                v[j + 1] = faEx2(v[j + 1]);
#endif
            }
            uint32_t pw[16];
#pragma unroll
            for (int w = 0; w < 16; ++w)
            {
                int const k = 64 * hh + 4 * w;
                pw[w] = faE4m3x4(v[k], v[k + 1], v[k + 2], v[k + 3]);
            }
            if (hh == 0)
            {
                prof.ar += faClk(ctx) - tl;
            }
#if MSO_MK_FA_XQ // P_wg atom hh, row r (the same E4M3 words)
#pragma unroll
            for (int c = 0; c < 4; ++c)
            {
                faStS128(
                    pW + 8192 * hh + faSw64(r, c), make_uint4(pw[4 * c], pw[4 * c + 1], pw[4 * c + 2], pw[4 * c + 3]));
            }
#else
#if (MSO_MK_FA_DBG & 64) == 0
            faSt16(sW + 16 * hh + laneBase, pw);
#else // attribution build (timing only, wrong results): no P store; the words stay live
            if (pw[0] == 0x7f7f7f7fu && pw[15] == 0x7f7f7f7fu)
            {
                asm volatile("" ::"r"(pw[1]), "r"(pw[2]), "r"(pw[3]), "r"(pw[4]), "r"(pw[5]), "r"(pw[6]), "r"(pw[7]),
                    "r"(pw[8]), "r"(pw[9]), "r"(pw[10]), "r"(pw[11]), "r"(pw[12]), "r"(pw[13]), "r"(pw[14]));
            }
#endif
#endif
        }
        KM_TL(2);
        if (pingPong && t + 1 < nt) // the other warpgroup's softmax of tile t + 1 may start
        {
            faPpPass(wg);
        }
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
        for (int j = 0; j < fa::kTileKeys; j += 4)
        {
            faAdd2(a0, a1, v[j], v[j + 1]);
            faAdd2(a2, a3, v[j + 2], v[j + 3]);
        }
        l = l + ((a0 + a1) + (a2 + a3));
        long long const tw = faClk(ctx);
#if MSO_MK_FA_XQ // P in shared memory, read by the async proxy
        fenceProxyAsyncSmem();
#elif (MSO_MK_FA_DBG & 64) == 0
        faWaitSt();
#endif
        KM_TL(3);
        long long const tp = faClk(ctx);
        prof.st += tp - tw;
        prof.sm += tp - ts1;
        faFenceBefore();
        faBarWg(wg);
#if MSO_MK_FA_ST // the loop record
        if (faStOn && r == 0 && u < fa::kStU)
        {
            faSt(ctx, wg * fa::kStWg + fa::kStU + u, clock64());
        }
#endif
        KM_TL(4);
        if (u == 0)
        {
            subStamp(ctx, 27);
        }
        long long const tb = faClk(ctx);
        prof.bar += tb - tp;
        if (issuerWarp)
        {
            // PV(t) (A = P in S_wg, B = the page's V of both heads), the slot's and O's commits, then QK(t + 2) into
            // S_wg (after PV(t) in this thread's issue order).
            __syncwarp();
            if (issueOrder) // kIFlagFaIssueOrder: chain t + 2 after chain t + 1 (the other warpgroup's)
            {
                faIoWait(wg);
            }
#if (MSO_MK_FA_DBG & 512) == 0
            faFenceAfter();
#endif
            int const left = n - fa::kTilePages * t;
            int const np = left < fa::kTilePages ? left : fa::kTilePages;
#if MSO_MK_KM & 64 // bit 64: the tile's MMAs (np of them) and both commits as one statement, for every tile
            kmPv4n(oW, sW, dV + static_cast<uint64_t>((fa::kTilePages * slot * fa::kVPage) >> 4), u != 0 ? 1u : 0u,
                base + fa::kOffEmpty + 8 * slot, base + fa::kOffODone + 8 * wg, np);
            if (false)
#elif MSO_MK_KM & 8 // bit 8: a full tile's 4 MMAs and both commits as one statement, one code copy for every slot
            if (np == fa::kTilePages)
            {
                kmPv4r(oW, sW, dV + static_cast<uint64_t>((fa::kTilePages * slot * fa::kVPage) >> 4), u != 0 ? 1u : 0u,
                    base + fa::kOffEmpty + 8 * slot, base + fa::kOffODone + 8 * wg);
            }
            else
#elif MSO_MK_KM & 1 // bit 1: a full tile's 4 MMAs and both commits as one statement
            if (np == fa::kTilePages)
            {
                uint32_t const acc0 = u != 0 ? 1u : 0u;
                uint32_t const empty = base + fa::kOffEmpty + 8 * slot;
                uint32_t const odone = base + fa::kOffODone + 8 * wg;
                kmForSlot(slot, [&](auto ks) { kmPv4<decltype(ks)::kValue>(oW, sW, dV, acc0, empty, odone); });
            }
            else
#endif
            {
                uint32_t const vOff = static_cast<uint32_t>(fa::kTilePages * slot * fa::kVPage);
#pragma unroll
                for (int q = 0; q < fa::kTilePages; ++q)
                {
                    if (q < np && (MSO_MK_FA_DBG & 17) == 0) // attribution build: bit 4 skips the PV MMAs
                    {
#if MSO_MK_FA_XQ // A = P_wg in shared memory (key step q: atom q / 2, bytes 32 (q % 2))
                        faMmaSSw(oW, dP + static_cast<uint64_t>(((q >> 1) * 8192 + (q & 1) * 32) >> 4),
                            dV + static_cast<uint64_t>((vOff + q * fa::kVPage) >> 4), fa::kIdescPv,
                            (u | q) != 0 ? 1u : 0u);
#else
                        faMmaTSw(oW, sW + 8 * q, dV + static_cast<uint64_t>((vOff + q * fa::kVPage) >> 4), fa::kIdescPv,
                            (u | q) != 0 ? 1u : 0u);
#endif
                    }
                    KM_TL2(u, 5 + q);
                }
                faCommitW(base + fa::kOffEmpty + 8 * slot);
                faCommitW(base + fa::kOffODone + 8 * wg);
            } // the per-MMA path
            prof.mma += faClk(ctx) - tb;
            KM_TL(5);
            KM_TL2(u, 9);
#if MSO_MK_KM_TL == 2
            kmTlU = u;
#endif
#if MSO_MK_KM & 4 // bit 4: QK(t + 2) into a drained queue: after PV(t) completed (odone[wg] phase u)
            if (t + 2 < nt)
            {
                faWaitW(ctx, base + fa::kOffODone + 8 * wg, u & 1, kCodeFaEmpty);
                faFenceAfter();
            }
#endif
            KM_TL(6);
            if (t + 2 < nt && !MSO_MK_FA_XQ) // with MSO_MK_FA_XQ the other issuer issues it
            {
                qkTile(t + 2);
            }
#if MSO_MK_KM_TL == 2
            kmTlU = -1;
#endif
            KM_TL(7);
            if (issueOrder && t + 1 < nt) // kIFlagFaIssueOrder: chain t + 3 exists (tile t + 1's)
            {
                faIoPass(wg);
            }
        }
        // Refill of tile t's slot with tile t + kSlots once PV(t) completed: the warpgroup's warps one page each
        // (lanes 0 .. 3: its four boxes); the issuer warp's lane 0 arms the slot's barrier.
        if (t + fa::kSlots < nt && !(refillSib && issuerWarp)) // with kIFlagF3FaRefillSib warps 1 .. 3 only
        {
            int const tn = t + fa::kSlots;
            int const left = n - fa::kTilePages * tn;
            int const np = left < fa::kTilePages ? left : fa::kTilePages;
            long long const te = faClk(ctx);
#if MSO_MK_FA_ST // the loop record
            uint64_t const faStE0 = faStOn ? clock64() : 0;
#endif
#if (MSO_MK_FA_DBG & 256) == 0 // attribution build (with bit 7 only): bit 8 skips the refill warps' empty wait
            faWaitW(ctx, base + fa::kOffEmpty + 8 * slot, (t / fa::kSlots) & 1, kCodeFaEmpty);
#endif
#if MSO_MK_FA_ST // the loop record
            faStEmpty += faStOn ? clock64() - faStE0 : 0;
#endif
            long long const ti = faClk(ctx);
            prof.empty += ti - te;
            uint32_t const full = base + fa::kOffFull + 8 * slot;
            if ((refillSib ? (warp & 3) == 1 : issuerWarp) && lane == 0)
            {
                // attribution build (timing only, wrong results): bit 7 arms the slot without its TMA (stale keys)
                mbarExpectTx(full, (MSO_MK_FA_DBG & 128) != 0 ? 0u : static_cast<uint32_t>(np * fa::kStageBytes));
            }
            // with kIFlagF3FaRefillSib warp 1 + w (w < 3) issues pages w and w + 3 (lanes 4 k + part: page w + 3 k)
            int const q = refillSib ? (warp & 3) - 1 + 3 * (lane >> 2) : warp & 3;
            int const part = refillSib ? (lane & 3) : lane;
            if (q < np && part < fa::kBoxesPerPage && (refillSib ? lane < 8 : true) && (MSO_MK_FA_DBG & 128) == 0)
            {
                int const rel = fa::kTilePages * tn + q;
                faIssueBox(
                    ctx, ring, fa::kTilePages * slot + q, part, ids[rel], ids[fa::kMaxUnitPages + rel], full, policy);
            }
            if (pf > 0 && issuerWarp && tn + pf < nt)
            {
                faPrefetchTile(ctx, ids, tn + pf, n);
            }
            prof.tma += faClk(ctx) - ti;
        }
    }
    faStampWg(ctx, wg == 0 ? 14 : 24);
#if MSO_MK_FA_ST // the loop record
    if (faStOn && r == 0)
    {
        faSt(ctx, fa::kStLoopEnd + wg, clock64());
    }
    if (faStOn && r == 32)
    {
        faSt(ctx, fa::kStEmpty + wg, faStEmpty);
    }
#endif

    // Epilogue (the stock's): this warpgroup's PVs complete (its last S wait implies all but the last), (m, l) of both
    // warpgroups exchanged, then this thread's 32 output columns [32 wg, +32) of its head from both accumulators.
    long long const td = faClk(ctx);
    if (myN > 0)
    {
        faWaitW(ctx, base + fa::kOffODone + 8 * wg, (myN - 1) & 1, kCodeFaDrain);
    }
    prof.drain += faClk(ctx) - td;
#if MSO_MK_FA_ST // the loop record
    if (faStOn && tid == 0)
    {
        faSt(ctx, fa::kStDrain, clock64());
    }
#endif
    subStamp(ctx, 28);
    float* xml = xch + 2 * fa::kRows;
    xml[wg * 2 * fa::kRows + r] = m;
    xml[wg * 2 * fa::kRows + fa::kRows + r] = l;
    faFenceBefore();
    __syncthreads();
    faFenceAfter();
    subStamp(ctx, 29);
    float const m0 = xml[r];
    float const l0 = xml[fa::kRows + r];
    float const m1 = xml[2 * fa::kRows + r];
    float const l1 = xml[3 * fa::kRows + r];
    int const n0 = (nt + 1) / 2;
    int const n1 = nt / 2;
    float const mx = fmaxf(m0, m1);
    float const mxSafe = mx == faNegInf() ? 0.f : mx;
    float const w0 = n0 > 0 && m0 != faNegInf() ? faEx2((m0 - mxSafe) * sc) : 0.f;
    float const w1 = n1 > 0 && m1 != faNegInf() ? faEx2((m1 - mxSafe) * sc) : 0.f;
    float const lsum = w0 * l0 + w1 * l1;
    float const inv = (lsum > 0.f ? faRcp(lsum) : 0.f) * dequant;
    float const c0 = w0 * inv;
    float const c1 = w1 * inv;
    int const rowOut = tok * fa::kHq + hq;
    if (n > 0)
    {
        float o0[32], o1[32];
        faLd32(tmem + 64 * h + 32 * wg + laneBase, o0);
        faLd32(tmem + 128 + 64 * h + 32 * wg + laneBase, o1);
        uint32_t pk[16];
#pragma unroll
        for (int j = 0; j < 16; ++j)
        {
            float const x0 = o0[2 * j];
            float const x1 = o0[2 * j + 1];
            float const y0 = n1 > 0 ? o1[2 * j] : 0.f;
            float const y1 = n1 > 0 ? o1[2 * j + 1] : 0.f;
            __half2 const hv = __floats2half2_rn(c0 * x0 + c1 * y0, c0 * x1 + c1 * y1);
            pk[j] = *reinterpret_cast<uint32_t const*>(&hv);
        }
        float const lse = lsum > 0.f ? mxSafe * sc - fa::kPre + faLg2(lsum) : faNegInf();
        uint8_t* po
            = ptr<uint8_t>(ctx, kPFaPartO) + ((static_cast<size_t>(rowOut) * units + unit) * fa::kD + 32 * wg) * 2;
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            st128(po + 16 * j,
                make_uint4(lamport(pk[4 * j]), lamport(pk[4 * j + 1]), lamport(pk[4 * j + 2]), lamport(pk[4 * j + 3])));
        }
        if (wg == 0)
        {
            st32(ptr<uint8_t>(ctx, kPFaPartL) + 4 * (static_cast<size_t>(rowOut) * units + unit),
                lamport(__float_as_uint(lse)));
        }
        // Debug copy of the partials in the stock op's layout (FP16 [128 rows][units][64], FP32 [128 rows][units]).
        if (MSO_MK_CKP(ctx.p[kPFaDbgO]) != 0) // MSO_MK_LD_CK
        {
            uint8_t* dpo
                = ptr<uint8_t>(ctx, kPFaDbgO) + ((static_cast<size_t>(rowOut) * units + unit) * fa::kD + 32 * wg) * 2;
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                st128(dpo + 16 * j, make_uint4(pk[4 * j], pk[4 * j + 1], pk[4 * j + 2], pk[4 * j + 3]));
            }
            if (wg == 0)
            {
                ptr<float>(ctx, kPFaDbgL)[static_cast<size_t>(rowOut) * units + unit] = lse;
            }
        }
    }
    subStamp(ctx, 15);
#if MSO_MK_FA_ST // the loop record
    if (faStOn && tid == 0)
    {
        faSt(ctx, fa::kStParts, clock64());
    }
#endif
    faStampVal(ctx, 16, prof.full);
    faStampVal(ctx, 17, prof.empty);
    faStampVal(ctx, 18, rescales);
    faStampVal(ctx, 19, prof.drain);
    faStampVal(ctx, 20, nt);
    faStampVal(ctx, 21, prof.sm);
    faStampVal(ctx, 22, prof.sw);
    faStampVal(ctx, 23, prof.bar);
    faStampVal(ctx, 13, prof.tma);
    faStampVal(ctx, 12, prof.mma);
    faStampVal(ctx, 9, prof.ld); // S-wait slots of tiles 7-9 hold these values
    faStampVal(ctx, 10, prof.ar);
    faStampVal(ctx, 11, prof.st);
    faFenceBefore();
    __syncthreads();
#if MSO_MK_FA_ST // the loop record
    if (faStOn && tid == 0)
    {
        faSt(ctx, fa::kStFaSync, clock64());
    }
#endif
    if (warp == 0)
    {
        faFenceAfter();
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" ::"r"(tmem) : "memory");
#if MSO_MK_FA_ST // the loop record
        if (faStOn && lane == 0)
        {
            faSt(ctx, fa::kStFaFree, clock64());
        }
#endif
    }
}

#if MSO_MK_FT_GATE
// ---- MSO_MK_FT_GATE: FaMerge's own gate (one-launch full layer, kIFlagQkvArena). Warp 0, lane l: columns 2 l,
// 2 l + 1 of gate head hq for token tok, i.e. tile 4 b + l / 8 of the gate's weight block b (20 + hq with
// kIFlagQkvHeadsFirst, else 2 hq + 1), float4 q = 4 tok + (l % 8) / 2 of every run partial that covers the tile, its
// elements 2 (l % 2) and + 1.
// The sum is stageQkvSum's (the slots in ascending k from +0, one BF16 rounding; the served one-launch layer has no
// kFlagXPrime scale), so the word equals the one the gate's QkvSum unit would have published, lamport() included.
// Single reader: row (tok, hq) is the only reader of token tok's float4s of the gate head's tiles; the even lane of a
// pair empties the float4 once the whole warp holds its values. The host plan holds no gate QkvSum unit. ----
__device__ __forceinline__ uint32_t ftMergeGate(Ctx const& ctx, int tok, int hq, Io const& io)
{
    int const lane = threadIdx.x & 31;
#if !MSO_MK_LEAN
    if (MSO_MK_KFLAGS(ctx.flags) & kFlagXPrime) // MSO_MK_LD_KF
    {
        __trap();                               // the gate would need the 1 / rms scale of QkvSum's kFlagXPrime path
    }
#endif
#if MSO_MK_LD_FL
    int const b = kQkvBlocks - fa::kHq + hq; // MSO_MK_LD_FL: the heads-first pack (FaMerge traps on another word)
#else
    int const b = (icFaFlags(ctx.p[kPIFlags]) & kIFlagQkvHeadsFirst) ? kQkvBlocks - fa::kHq + hq : 2 * hq + 1;
#endif
    int const tile = kQkvTilesPerBlock * b + lane / 8;
    int const q = 4 * tok + (lane & 7) / 2;
    uint8_t const* base = ptr<uint8_t const>(ctx, kPQkvPart);
    uint8_t const* a[kQkvMaxRuns] = {};
#pragma unroll
    for (int sl = 0; sl < kQkvSlices; ++sl)
    {
        int const g = sl * kQkvTiles + tile;
        int const cFirst = qkvCtaOfChunk(g * kSegChunks);
        int const cLast = qkvCtaOfChunk(g * kSegChunks + kSegChunks - 1);
#pragma unroll
        for (int k = 0; k < kQkvRunsPerSeg; ++k)
        {
            int const cr = cFirst + k;
            if (cr <= cLast)
            {
                int const slot = g - qkvRunStart(cr) / kSegChunks;
                a[sl * kQkvRunsPerSeg + k]
                    = base + (static_cast<size_t>(cr) * kQkvMaxSlots + slot) * kQkvPartSlotBytes + 16 * q;
            }
        }
    }
    uint4 v[kQkvMaxRuns];
    pollVecs<kQkvMaxRuns>(ctx, v, a, kCodeFaQkv);
    bool const hi = (lane & 1) != 0;
    float s0 = 0.f;
    float s1 = 0.f;
#pragma unroll
    for (int k = 0; k < kQkvMaxRuns; ++k)
    {
        if (a[k] != nullptr)
        {
            s0 += __uint_as_float(hi ? v[k].z : v[k].x);
            s1 += __uint_as_float(hi ? v[k].w : v[k].y);
        }
    }
    if ((MSO_MK_KFLAGS(ctx.flags) & kFlagCheck) && io.qkvF32 != nullptr) // MSO_MK_LD_KF
    {
        *reinterpret_cast<float2*>(
            static_cast<float*>(io.qkvF32) + static_cast<size_t>(tok) * kQkvRows + kQkvBlockRows * b + 2 * lane)
            = make_float2(s0, s1);
    }
    if (!hi)
    {
#pragma unroll
        for (int k = 0; k < kQkvMaxRuns; ++k)
        {
            if (a[k] != nullptr)
            {
                st128(const_cast<uint8_t*>(a[k]), emptyVec());
            }
        }
    }
    return lamport(packBf16(s0, s1));
}

// ---- end MSO_MK_FT_GATE ----
#endif

#if MSO_MK_LM
// ---- MSO_MK_LM (fullAttentionConfig.cuh): FaMerge's arrival ticket (bit 1) and the gate spread over the warps
// (bit 2). Shared memory inside FaMerge's reduction area (kSmemRed): the ticket word and the per-warp gate values, both
// beyond the merge's own words (red[0 .. 16 + 64 * 9)). ----
namespace lm
{
constexpr int kRows = kTok * fa::kHq; // FaMerge rows: the first kRows tickets merge
constexpr int kTicketWord = 1600;     // uint32 index in kSmemRed: thread 0's ticket, read by every thread
constexpr int kGateOff = 4096;        // bytes in kSmemRed: float2 [kWarps][32], warp w's values of gate slot w
static_assert(16 + 64 * 9 <= kGateOff / 4 && kGateOff + kWarps * 32 * 8 <= 4 * kTicketWord
        && 4 * kTicketWord + 4 <= kSmemRedBytes,
    "lm: the ticket word and the gate values sit beyond the merge's reduction words");
static_assert(kQkvMaxRuns <= kWarps, "lm S: one gate-run slot per warp");
static_assert(kCtas >= kRows, "lm T: at least one FaMerge candidate per row");
} // namespace lm

// The relaxed arrival ticket (the atomic only elects the row; the partials are data-as-flag words, so no ordering
// with them is needed).
__device__ __forceinline__ uint32_t lmAtomAdd(uint32_t* p, uint32_t v)
{
    uint32_t old;
    asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}

// The weight block of gate head hq in the QKV pack (ftMergeGate's b).
__device__ __forceinline__ int lmGateBlock(Ctx const& ctx, int hq)
{
#if MSO_MK_LD_FL
    return kQkvBlocks - fa::kHq + hq; // MSO_MK_LD_FL: the heads-first pack (FaMerge traps on another launch word)
#else
    return (icFaFlags(ctx.p[kPIFlags]) & kIFlagQkvHeadsFirst) ? kQkvBlocks - fa::kHq + hq : 2 * hq + 1;
#endif
}

// The float4 that ftMergeGate's lane `lane` loads for run slot k (nullptr where it loads none): tile 4 b + lane / 8,
// float4 q = 4 tok + (lane % 8) / 2 of run partial slot k = sl * kQkvRunsPerSeg + r (K slice sl, the r-th run over it).
__device__ __forceinline__ uint8_t const* lmGateAddr(Ctx const& ctx, int tok, int hq, int k, int lane)
{
    if (k >= kQkvMaxRuns)
    {
        return nullptr;
    }
    int const b = lmGateBlock(ctx, hq);
    int const tile = kQkvTilesPerBlock * b + lane / 8;
    int const q = 4 * tok + (lane & 7) / 2;
    int const gg = (k / kQkvRunsPerSeg) * kQkvTiles + tile;
    int const cFirst = qkvCtaOfChunk(gg * kSegChunks);
    int const cLast = qkvCtaOfChunk(gg * kSegChunks + kSegChunks - 1);
    int const cr = cFirst + k % kQkvRunsPerSeg;
    if (cr > cLast)
    {
        return nullptr;
    }
    int const slot = gg - qkvRunStart(cr) / kSegChunks;
    return ptr<uint8_t const>(ctx, kPQkvPart) + (static_cast<size_t>(cr) * kQkvMaxSlots + slot) * kQkvPartSlotBytes
        + 16 * q;
}

// ---- end MSO_MK_LM ----
#endif

// FaMerge: unit = output row R = tok * 16 + hq. Thread (chunk c = tid % 8 of 8 FP16, slot lane sl = tid / 8) polls the
// slots u = sl + 32 k < eff of its row; weights 2^(lse_u - max), fixed-order sums (thread, warp butterflies, warps in
// order); BF16; the norm-gate on warp 0 (lane: columns 2 lane, 2 lane + 1); then the read slots are emptied.
// With MSO_MK_FT_DRY, pass 0 runs the same code on zeros, without the gate, the polls and any global store, so pass 1
// (the real merge, after the last partial) runs from a warm instruction cache.
__device__ __noinline__ void stageFaMerge(Ctx const& ctx, int unit, Io const& io)
{
    icFaCheck(ctx.p[kPIFlags]); // the full-attention kPIFlags bits this build fixes
#if MSO_MK_LD_FL
    if (!(ctx.p[kPIFlags] & kIFlagQkvHeadsFirst) || ctx.p[kPAhOReplicas] != 1)
    {
        __trap(); // MSO_MK_LD_FL: the served one-launch full layers (heads-first pack, one o_proj replica)
    }
#endif
    constexpr int kMaxK = (kCtas + 31) / 32;
    int const tid = threadIdx.x;
    int const warp = tid >> 5;
    int const lane = tid & 31;
    int const c = tid & 7;
    int const sl = tid >> 3;
#if MSO_MK_LD_FG
    ldFgCheck(ctx);
#endif
    int const units = static_cast<int>(MSO_MK_FG_UNITS(ctx.p[kPFaUnits])); // MSO_MK_LD_FG
    FaGeom const g = faGeom(faReadL(ctx), units, static_cast<int>(MSO_MK_FG_MINPAGES(ctx.p[kPFaMinPages])), 0);
#if !(MSO_MK_LM & 1)
    int const tok = unit / fa::kHq;
    int const hq = unit % fa::kHq;
#endif
#if MSO_MK_FA_ST // the FaMerge part of the record
    bool const fmStOn = MSO_MK_STAMPS(ctx) != nullptr;
    if (fmStOn && tid == 0)
    {
        faSt(ctx, fa::kStFmStart, clock64());
    }
    uint64_t fmRounds = 0;
#endif
#if MSO_MK_LM & 1 // bit 1: the row is this CTA's arrival ticket; tickets from lm::kRows on merge nothing
    {
        uint32_t* const sT = reinterpret_cast<uint32_t*>(ctx.smem + kSmemRed) + lm::kTicketWord;
        if (tid == 0)
        {
            *sT = lmAtomAdd(ctx.counters + kCLmTicket, 1u);
        }
        __syncthreads();
        unit = static_cast<int>(*sT);
#if MSO_MK_FA_ST // the record: the ticket (bits 56 .. 63) when thread 0 holds it
        if (fmStOn && tid == 0)
        {
            faSt(ctx, fa::kStFmTicket,
                (static_cast<uint64_t>(clock64()) & ((1ull << 56) - 1)) | (static_cast<uint64_t>(unit & 255) << 56));
        }
#endif
        if (unit >= lm::kRows)
        {
            return;
        }
    }
    int const tok = unit / fa::kHq;
    int const hq = unit % fa::kHq;
#endif
#if MSO_MK_FT_DRY // a dry pass, then the real one
#pragma unroll 1
    for (int pass = 0; pass < 2; ++pass)
    {
        int passV = pass;
        asm volatile("" : "+r"(passV)); // opaque: one copy of the code serves both passes
        bool const dry = passV == 0;
#else
    {
        constexpr bool dry = false;
#endif
        // The gate of this row (warp 0, lane: columns 2 lane, 2 lane + 1), loaded now so its latency overlaps the
        // polls.
        // ---- polled in the arena with kIFlagQkvArena (warp 0 as a whole) ----
        uint32_t gw = 0u;
#if MSO_MK_LM & 2 // bit 2: with the arena, warp w polls gate-run slot w with its partials; warp 0 sums them below
        bool const lmS = (icFaFlags(ctx.p[kPIFlags]) & kIFlagQkvArena) != 0;
        uint8_t const* const ag = lmS && !dry ? lmGateAddr(ctx, tok, hq, warp, lane) : nullptr;
        if (warp == 0 && !dry && !lmS)
#else
        if (warp == 0 && !dry)
#endif
        {
            size_t const gi = static_cast<size_t>(tok) * fa::kQkvCols + hq * (2 * fa::kD) + fa::kD + 2 * lane;
#if MSO_MK_FT_GATE && !(MSO_MK_LM & 2)
            if (icFaFlags(ctx.p[kPIFlags])
                & kIFlagQkvArena) // MSO_MK_FT_GATE: from the gate runs' partials (ftMergeGate)
            {
                gw = ftMergeGate(ctx, tok, hq, io);
            }
#else
            if (icFaFlags(ctx.p[kPIFlags]) & kIFlagQkvArena)
            {
                gw = poll1(ctx, ctx.arena + kArFaQkv + 2 * gi, kCodeFaQkv);
            }
#endif
            else
            {
                gw = *reinterpret_cast<uint32_t const*>(static_cast<__nv_bfloat16 const*>(io.qkv) + gi);
            }
#if MSO_MK_FA_ST // the loop record
            if (fmStOn && lane == 0)
            {
                faSt(ctx, fa::kStFmGate, clock64());
            }
#endif
        }
        // ---- end kIFlagQkvArena ----
        uint8_t* partO = ptr<uint8_t>(ctx, kPFaPartO) + static_cast<size_t>(unit) * units * fa::kD * 2;
        uint8_t* partL = ptr<uint8_t>(ctx, kPFaPartL) + static_cast<size_t>(unit) * units * 4;
        uint8_t const* ao[kMaxK];
        uint8_t const* al[kMaxK];
#pragma unroll
        for (int k = 0; k < kMaxK; ++k)
        {
            int const u = sl + 32 * k;
            bool const on = u < g.eff;
            ao[k] = on ? partO + (static_cast<size_t>(u) * fa::kD + 8 * c) * 2 : nullptr;
            al[k] = on ? partL + 4 * u : nullptr;
        }
#if MSO_MK_LM & 2 // bit 2: this thread's gate float4 (nullptr: none, which reads as full)
        uint4 vg = ag != nullptr ? ldPoll128(ag) : make_uint4(0, 0, 0, 0);
#endif
        uint4 vo[kMaxK];
        uint32_t vl[kMaxK];
#pragma unroll
        for (int k = 0; k < kMaxK; ++k)
        {
            vo[k] = ao[k] != nullptr && !dry ? ldPoll128(ao[k]) : make_uint4(0, 0, 0, 0);
            vl[k] = al[k] != nullptr && !dry ? ldPoll32(al[k]) : 0u;
        }
        uint64_t t0 = 0;
        while (!dry)
        {
            bool ok = true;
#pragma unroll
            for (int k = 0; k < kMaxK; ++k)
            {
                ok &= full4(vo[k]) & (vl[k] != kEmpty);
            }
#if MSO_MK_LM & 2 // bit 2
            ok &= full4(vg);
#endif
            if (warpAll(ok))
            {
                break;
            }
#if MSO_MK_FA_ST // the loop record
            ++fmRounds;
#endif
            spinCheck(ctx, t0, kCodeFaPart, unit, 0);
#pragma unroll
            for (int k = 0; k < kMaxK; ++k)
            {
                if (!full4(vo[k]))
                {
                    vo[k] = ldPoll128(ao[k]);
                }
                if (vl[k] == kEmpty)
                {
                    vl[k] = ldPoll32(al[k]);
                }
            }
#if MSO_MK_LM & 2 // bit 2
            if (!full4(vg))
            {
                vg = ldPoll128(ag);
            }
#endif
        }
        if (!dry)
        {
            stampInput(ctx);
        }
#if MSO_MK_LM & 2 // bit 2: the whole warp holds its values (the loop left through warpAll): this lane's two FP32 values
        // of gate slot `warp` into shared memory for warp 0, and the pair's even lane empties the float4 (single
        // reader)
        float2* const lmG = reinterpret_cast<float2*>(ctx.smem + kSmemRed + lm::kGateOff);
        if (lmS && !dry)
        {
            bool const hi = (lane & 1) != 0;
            lmG[warp * 32 + lane] = make_float2(__uint_as_float(hi ? vg.z : vg.x), __uint_as_float(hi ? vg.w : vg.y));
            if (!hi && ag != nullptr)
            {
                st128(const_cast<uint8_t*>(ag), emptyVec());
            }
        }
#endif
#if MSO_MK_FA_ST // this warp's polls complete, its poll rounds in bits 56 .. 63
        if (fmStOn && lane == 0 && !dry)
        {
            uint64_t const rounds = fmRounds < 255 ? fmRounds : 255;
            faSt(ctx, fa::kStFmPolled + warp, (static_cast<uint64_t>(clock64()) & ((1ull << 56) - 1)) | (rounds << 56));
        }
#endif
        float* red = reinterpret_cast<float*>(ctx.smem + kSmemRed);
        // The row max of the log2-sum-exps.
        float mloc = faNegInf();
#pragma unroll
        for (int k = 0; k < kMaxK; ++k)
        {
            if (al[k] != nullptr)
            {
                mloc = fmaxf(mloc, __uint_as_float(vl[k]));
            }
        }
        mloc = fmaxf(mloc, __shfl_xor_sync(0xffffffffu, mloc, 8));
        mloc = fmaxf(mloc, __shfl_xor_sync(0xffffffffu, mloc, 16));
        if (lane == 0)
        {
            red[warp] = mloc;
        }
        __syncthreads();
        if (!dry)
        {
            subStamp(ctx, 30); // every warp's polls complete
            stampWeights(ctx); // the same moment as the unit's "weights" stamp (stamped builds; FaMerge has no weights)
#if MSO_MK_FA_ST               // the loop record
            if (fmStOn && tid == 0)
            {
                faSt(ctx, fa::kStFmBar, clock64());
            }
#endif
        }
        float M = red[0];
#pragma unroll
        for (int w = 1; w < kWarps; ++w)
        {
            M = fmaxf(M, red[w]);
        }
        float acc[8] = {};
        float ws = 0.f;
#pragma unroll
        for (int k = 0; k < kMaxK; ++k)
        {
            float const lse = __uint_as_float(vl[k]);
            if (al[k] != nullptr && lse != faNegInf())
            {
                float const w = faEx2(lse - M);
                ws += w;
                uint32_t const o[4] = {vo[k].x, vo[k].y, vo[k].z, vo[k].w};
#pragma unroll
                for (int e = 0; e < 4; ++e)
                {
                    __half2 const hv = *reinterpret_cast<__half2 const*>(&o[e]);
                    float2 const f = __half22float2(hv);
                    acc[2 * e] = fmaf(w, f.x, acc[2 * e]);
                    acc[2 * e + 1] = fmaf(w, f.y, acc[2 * e + 1]);
                }
            }
        }
#pragma unroll
        for (int msk = 8; msk <= 16; msk <<= 1)
        {
            ws += __shfl_xor_sync(0xffffffffu, ws, msk);
#pragma unroll
            for (int e = 0; e < 8; ++e)
            {
                acc[e] += __shfl_xor_sync(0xffffffffu, acc[e], msk);
            }
        }
        __syncthreads();
        if (lane < 8)
        {
            // red[16 + (warp * 8 + c) * 9 + e]: this warp's sums of chunk c = lane.
#pragma unroll
            for (int e = 0; e < 8; ++e)
            {
                red[16 + (warp * 8 + lane) * 9 + e] = acc[e];
            }
            red[16 + (warp * 8 + lane) * 9 + 8] = ws;
        }
        __syncthreads();
#if MSO_MK_FA_ST // the loop record
        if (fmStOn && tid == 0 && !dry)
        {
            faSt(ctx, fa::kStFmSum, clock64());
        }
#endif
        if (warp == 0)
        {
            int const d0 = 2 * lane;
            int const cc = d0 / 8;
            int const e0 = d0 % 8;
            float s0 = 0.f, s1 = 0.f, sw = 0.f;
#pragma unroll
            for (int w = 0; w < kWarps; ++w)
            {
                float const* q = red + 16 + (w * 8 + cc) * 9;
                s0 += q[e0];
                s1 += q[e0 + 1];
                sw += q[8];
            }
            float const a0 = s0 / sw;
            float const a1 = s1 / sw;
            size_t const oc = static_cast<size_t>(tok) * fa::kOutCols + hq * fa::kD + d0;
            if (MSO_MK_CKP(ctx.p[kPFaOutF32]) != 0 && !dry) // MSO_MK_LD_CK
            {
                *reinterpret_cast<float2*>(ptr<float>(ctx, kPFaOutF32) + oc) = make_float2(a0, a1);
            }
            float const x0 = bf16Round(a0);
            float const x1 = bf16Round(a1);
            float ss = x0 * x0 + x1 * x1;
#pragma unroll
            for (int msk = 16; msk >= 1; msk >>= 1)
            {
                ss += __shfl_xor_sync(0xffffffffu, ss, msk);
            }
            float const eps = __int_as_float(static_cast<int>(ctx.p[kPFaEps]));
            float const rs = rsqrtf(ss / fa::kD + eps);
#if MSO_MK_LM & 2 // bit 2: ftMergeGate's sum (slots ascending from +0, one BF16 rounding) from the warps' values
            if (lmS && !dry)
            {
#if !MSO_MK_LEAN
                if (MSO_MK_KFLAGS(ctx.flags) & kFlagXPrime) // MSO_MK_LD_KF
                {
                    __trap(); // the gate would need the 1 / rms scale of QkvSum's kFlagXPrime path
                }
#endif
                float s0g = 0.f;
                float s1g = 0.f;
#pragma unroll
                for (int k = 0; k < kWarps; ++k)
                {
                    float2 const v = lmG[k * 32 + lane];
                    s0g += v.x;
                    s1g += v.y;
                }
                if ((MSO_MK_KFLAGS(ctx.flags) & kFlagCheck) && io.qkvF32 != nullptr) // MSO_MK_LD_KF
                {
                    *reinterpret_cast<float2*>(static_cast<float*>(io.qkvF32) + static_cast<size_t>(tok) * kQkvRows
                        + kQkvBlockRows * lmGateBlock(ctx, hq) + 2 * lane)
                        = make_float2(s0g, s1g);
                }
                gw = lamport(packBf16(s0g, s1g));
#if MSO_MK_FA_ST // the record: warp 0 holds its gate word
                if (fmStOn && lane == 0)
                {
                    faSt(ctx, fa::kStFmGate, clock64());
                }
#endif
            }
#endif
            float const g0 = bf16Lo(gw);
            float const g1 = bf16Hi(gw);
            float const sg0 = bf16Round(1.f / (1.f + __expf(-g0)));
            float const sg1 = bf16Round(1.f / (1.f + __expf(-g1)));
            float const n0v = bf16Round(x0 * rs);
            float const n1v = bf16Round(x1 * rs);
            uint32_t const ow = packBf16(n0v * sg0, n1v * sg1);
            if (MSO_MK_CKP(ctx.p[kPFaOut]) != 0 && !dry) // MSO_MK_LD_CK
            {
                st32(ptr<uint8_t>(ctx, kPFaOut) + oc * 2, ow);
            }
            // ---- kIFlagFaOutArena: the o_proj input for AhOproj in the same launch ----
            if ((icFaFlags(ctx.p[kPIFlags]) & kIFlagFaOutArena) && !dry)
            {
#if MSO_MK_LD_FL
                st32(ctx.arena + kArAhO + oc * 2, lamport(ow)); // MSO_MK_LD_FL: one replica (checked at entry)
#else
                int const reps = static_cast<int>(ctx.p[kPAhOReplicas]);
                for (int rep = 0; rep < reps; ++rep)
                {
                    st32(ctx.arena + kArAhO + rep * (kTok * fa::kOutCols * 2) + oc * 2, lamport(ow));
                }
#endif
            }
            // ---- end kIFlagFaOutArena ----
        }
        // Single reader: empty what this thread polled.
#pragma unroll
        for (int k = 0; k < kMaxK; ++k)
        {
            if (ao[k] != nullptr && !dry)
            {
                st128(const_cast<uint8_t*>(ao[k]), emptyVec());
                if (c == 0)
                {
                    st32(const_cast<uint8_t*>(al[k]), kEmpty);
                }
            }
        }
#if MSO_MK_FT_DRY // the dry pass's reads of the reduction buffer end before the real pass writes it
        if (dry)
        {
            __syncthreads();
        }
#endif
#if MSO_MK_FA_ST // the loop record
        if (fmStOn && tid == 0 && !dry)
        {
            faSt(ctx, fa::kStFmEnd, clock64());
        }
#endif
    }
}
