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

// The attention half of a sliding-window decoder layer (one rank of TP8: 16 query heads, 2 KV heads of width 64), for
// the 8 verify rows, after the QKV run units of msoMegakernel.cu (K-slice order, MSO_MK_QKV_SLICES = 4):
//
//   AhQkvFin  (36 units, one per 64-row block = one head of the projection): the runs' FP32 partials summed and rounded
//             once to BF16 (the projection), then the stock prep statements on that head: q / k: the weightless RMSNorm
//             (eight lanes of eight elements, add.rn in order, then lanes paired over bits 0, 1, 2; rsqrt.approx.ftz),
//             bf16(bf16(x inv) scale), GPT-J RoPE (fma.rn.ftz / mul.rn.ftz), BF16; k / v: e4m3(bf16(x bf16(s))) into
//             the paged pool (raw bytes) and the arena; q: bf16(e4m3(bf16(q bf16(s)))) into the arena (q_fp8); the gate
//             head: bf16(sigmoid(g)) into the arena.
//   AhSwaAttn (66 units = 2 KV heads x 33 blocks of 64 keys, two 32-key pages): the pages of keys before the step,
//             requested at kernel entry (before griddepcontrol.wait); the step's rows from the arena; keys outside
//             [lo, L) zeroed. 8 warps = 4 m tiles of 16 query rows (row = 8 token + query head of the group) x 2 key
//             halves of 32 (the stock split's 32-key block); S and P V with mma.sync BF16 (FP8 -> BF16 is exact);
//             base-2 softmax from m = -1e30 per half; the two halves merged in the unit; the partial (O / s, m, l)
//             stored as Lamport words (single reader). Units b < 32 then merge rows {2b, 2b + 1} of their KV head over
//             the 33 partials (w = exp2(m - max m), 0 where l = 0), round to BF16, apply the norm-gate
//             (bf16(bf16(a rsqrt(mean(a^2) + eps)) bf16(sigmoid(g)))) and store the gated o_proj input into the
//             arena; the partials they read are emptied after that output.
//   AhOproj   (212 units, 512 tiles of 16 output rows: 2 or 3 per CTA): the gated input (K = 1024) times the CTA's
//             o_proj tiles, one BF16 rounding (cvt.rn, which never yields the 0xFFFE half the hop1 consumer polls
//             for), stored once per element into this rank's hop1 slot (the multicast address of the stock direct
//             transport), whose stock consumer hop then runs unchanged.
//
// Every pre-wait read is a graph input or immutable: the sequence length, the positions, the block-table entries, the
// RoPE rows and the pool pages of keys before the step. Included from msoMegakernel.cu inside namespace mso_mk, after
// its stages (helpers, Ctx, Lamport family, GEMV blocks, QKV cut).

#pragma once

// ---------------------------------------------------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------------------------------------------------
constexpr int kAhQHeads = 16;
constexpr int kAhKvHeads = 2;
constexpr int kAhGroup = kAhQHeads / kAhKvHeads; // query heads per KV head
constexpr int kAhD = 64;
constexpr int kAhPageTok = 32;
constexpr int kAhPageBytes = kAhKvHeads * kAhPageTok * kAhD; // one pool block (FP8)
constexpr int kAhBlockKeys = 64;                             // keys of an attention unit (two pages)
constexpr int kAhBlocks = 33;                                // blocks per KV head: cdiv(2048 + 8 - 2 + 31, 64)
constexpr int kAhAttnUnits = kAhKvHeads * kAhBlocks;
constexpr int kAhRows = kTok * kAhGroup;                     // query rows per KV head (8 tokens x 8 heads)
constexpr int kAhOTiles = kHidden / 16;                      // o_proj [8192, 1024]: 512 tiles
constexpr int kAhOChunks = (kAhQHeads * kAhD) / kChunkCols;  // 16
constexpr int kAhOTileBytes = kAhOChunks * kBf16Block;       // 32 KiB
constexpr int kAhOCols = kAhQHeads * kAhD;                   // 1024
constexpr int kAhOMaxTiles = 4;                              // o_proj tiles of one unit (4 only without SwaAttn)
static_assert(kAhOChunks == 2 * kWarps, "two o_proj chunks per warp");

// Shared memory: the entry copies (pages or RoPE rows) after the largest QKV run region, the attention unit's BF16
// tiles after the o_proj weights.
constexpr int kAhEntryOff = 178176;
constexpr int kAhEntryBytes = 8192;
constexpr int kAhScratchOff = 98304;
constexpr int kAhRowStride = (kAhD + 8) * 2;      // padded BF16 row (144 B: ldmatrix rows on distinct banks)
constexpr int kAhTileBytes = 64 * kAhRowStride;
constexpr int kAhScratchBytes = 3 * kAhTileBytes; // Q, K, V
static_assert(kAhEntryOff >= kQkvMaxRun * kBf16Block, "the entry copies sit after the QKV run region");
static_assert(kAhEntryOff + kAhEntryBytes <= kSmemMain, "entry region");
static_assert(kAhScratchOff >= 3 * kAhOTileBytes && kAhScratchOff + kAhScratchBytes <= kAhEntryOff, "scratch");

// The single-reader attention partials (kPAhPart): O FP32 [2][33][64 rows][64], then (m, l) FP32 [2][33][64][2].
constexpr int kAhPartOBytes = kAhAttnUnits * kAhRows * kAhD * 4;
constexpr int kAhPartBytes = kAhPartOBytes + kAhAttnUnits * kAhRows * 2 * 4;
static_assert(kAhPartBytes == 2 * 33 * 64 * 64 * 4 + 2 * 33 * 64 * 2 * 4, "msoMkAhPartBytes");

// Per-CTA metadata the entry hook leaves in the small area (a CTA holds at most one QkvFin or SwaAttn unit).
constexpr int kAhMetaOff = kSmallScalars + 64;

struct AhMeta
{
    int L;       // kv length including the step
    int pages;   // SwaAttn: bit i = landing slot i (K p0, K p1, V p0, V p1) was copied
    int page[8]; // QkvFin k / v: the pool block of each step token
    int pos[8];  // QkvFin q / k: the RoPE position of each step token
};

static_assert(kAhMetaOff + static_cast<int>(sizeof(AhMeta)) <= kSmallParams, "metadata slot");

// kPAhFlags bits.
constexpr int64_t kAhFlagOprojCanary = 1; // o_proj: lane 0 of warp 0 waits on one word of the gated input first

enum AhWaitCode : int
{
    kCodeAhRuns = 140,  // QkvFin: the runs' partials
    kCodeAhRope = 141,  // QkvFin: the RoPE rows
    kCodeAhQ = 142,     // SwaAttn: Q and the step's K / V rows
    kCodeAhPages = 143, // SwaAttn: the pool pages
    kCodeAhMerge = 144, // merge: the partials and the gate
    kCodeAhOIn = 145,   // Oproj: the gated input
};

__device__ __forceinline__ AhMeta* ahMeta(Ctx const& ctx)
{
    return reinterpret_cast<AhMeta*>(ctx.smem + kSmemSmall + kAhMetaOff);
}

__device__ __forceinline__ float f32Param(Ctx const& ctx, int idx)
{
    return __int_as_float(static_cast<int>(ctx.p[idx]));
}

// ---------------------------------------------------------------------------------------------------------------------
// Arithmetic with the stock kernels' instruction choices
// ---------------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ float rsqrtApproxFtz(float x)
{
    float y;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float mulRnFtz(float a, float b)
{
    float y;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

__device__ __forceinline__ float fmaRnFtz(float a, float b, float c)
{
    float y;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(y) : "f"(a), "f"(b), "f"(c));
    return y;
}

__device__ __forceinline__ float ex2Approx(float x)
{
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// Two FP32 values to E4M3 (RN, satfinite), lo in the low byte.
__device__ __forceinline__ uint32_t e4m3x2(float lo, float hi)
{
    uint16_t r;
    asm("{\n.reg .b16 t;\ncvt.rn.satfinite.e4m3x2.f32 t, %1, %2;\nmov.b16 %0, t;\n}" : "=h"(r) : "f"(hi), "f"(lo));
    return r;
}

// Two E4M3 bytes (lo in the low byte) to a BF16 pair (exact).
__device__ __forceinline__ uint32_t e4m3x2ToBf16x2(uint32_t v)
{
    uint32_t h2;
    uint16_t const b = static_cast<uint16_t>(v);
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(b));
    __half2 const h = *reinterpret_cast<__half2 const*>(&h2);
    float2 const f = __half22float2(h);
    return packBf16(f.x, f.y);
}

// The stock KV-cache store: e4m3(bf16(x * bf16(s))) (x a BF16 value).
__device__ __forceinline__ float toCacheScaled(float x, float sBf16)
{
    return bf16Round(__fmul_rn(x, sBf16));
}

__device__ __forceinline__ float e4m3Value(float x)
{
    uint32_t const b = e4m3x2(x, 0.f);
    uint32_t const p = e4m3x2ToBf16x2(b);
    return bf16Lo(p);
}

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t addr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], uint32_t addr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void sts128(uint32_t addr, uint4 v)
{
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}

// ---------------------------------------------------------------------------------------------------------------------
// The window of a step (stock swa_block): keys [lo, L), blocks of 64 keys from base = 32 floor(lo / 32).
// ---------------------------------------------------------------------------------------------------------------------
struct AhWindow
{
    int L, past, lo, base, nBlocks;
};

__device__ __forceinline__ AhWindow ahWindow(int L, int window)
{
    AhWindow w;
    w.L = L;
    w.past = L - kTok;
    w.lo = w.past - window + 1 > 0 ? w.past - window + 1 : 0;
    w.base = (w.lo / kAhPageTok) * kAhPageTok;
    w.nBlocks = (L - w.base + kAhBlockKeys - 1) / kAhBlockKeys;
    return w;
}

// ---------------------------------------------------------------------------------------------------------------------
// The packed QKV weight's 64-row blocks, heads first (attn_half.pack_qkv_heads_first): q_0 .. q_15 | k_0 k_1 |
// v_0 v_1 | g_0 .. g_15. The gate rows are needed only at the merge, so their runs can sit on the CTAs that start last.
// ---------------------------------------------------------------------------------------------------------------------
constexpr int kAhBlkK = kAhQHeads;
constexpr int kAhBlkV = kAhQHeads + kAhKvHeads;
constexpr int kAhBlkG = kAhQHeads + 2 * kAhKvHeads;

// The checkpoint column (order q_0 | g_0 | ... | q_15 | g_15 | k_0 | k_1 | v_0 | v_1) of packed block b.
__device__ __forceinline__ int ahCkptCol(int b)
{
    return b < kAhBlkK ? 2 * kAhD * b
        : b < kAhBlkG  ? 2 * kAhQHeads * kAhD + kAhD * (b - kAhBlkK)
                       : 2 * kAhD * (b - kAhBlkG) + kAhD;
}

// ---------------------------------------------------------------------------------------------------------------------
// Entry hook (thread 0, before griddepcontrol.wait): the pool pages of an attention unit, the RoPE rows and the
// block-table entries of a QkvFin unit. Only graph inputs and immutable data are read here.
// ---------------------------------------------------------------------------------------------------------------------
// ---- kIFlagI2bEntryLite: a gate QkvFin block reads no metadata, so it has no entry hook (no entry round trip on the
// CTA's path to its first unit) ----
// MSO_MK_LD_SW: the served SWA layers set kIFlagI2bEntryLite (ahEntryIssue traps on a launch that does not).
#if MSO_MK_LD_SW
#define MSO_MK_SW_ENTRY_LITE(f) kIFlagI2bEntryLite
#else
#define MSO_MK_SW_ENTRY_LITE(f) f
#endif
__device__ __forceinline__ bool ahEntryHookless(Ctx const& ctx, int role)
{
    return (MSO_MK_SW_ENTRY_LITE(icPIFlags(ctx.p[kPIFlags])) & kIFlagI2bEntryLite) && (role >> 16) == kStAhQkvFin
        && (role & 0xffff) >= kAhBlkG;
}

// kIFlagI2bEntryLite, a QkvFin q / k block: the sequence length and the positions in one round trip (the generic stores
// into the metadata otherwise order the positions' loads after the block-table loads).
__device__ __noinline__ void ahEntryIssueFinLite(Ctx const& ctx, int b, uint32_t bar)
{
    AhMeta* meta = ahMeta(ctx);
    int const* positions = ptr<int const>(ctx, kPAhPositions);
    float const* cs = ptr<float const>(ctx, kPAhCosSin);
    int const* bt = ptr<int const>(ctx, kPAhBlockTable);
    int const btStride = static_cast<int>(ctx.p[kPAhBtStride]);
    bool const rope = b < kAhBlkV;
    bool const cache = b >= kAhBlkK && b < kAhBlkG;
    int const L = __ldg(ptr<int const>(ctx, kPAhSeqLen));
    int pos[kTok];
#pragma unroll
    for (int t = 0; t < kTok; ++t)
    {
        pos[t] = rope ? __ldg(positions + t) : 0;
    }
    int page[kTok];
    int const kv = b < kAhBlkV ? 0 : 1;
#pragma unroll
    for (int t = 0; t < kTok; ++t)
    {
        page[t] = cache ? __ldg(bt + kv * btStride + (L - kTok + t) / kAhPageTok) : 0;
    }
    if (rope)
    {
        uint32_t const dst = ctx.smemBase + kAhEntryOff;
        mbarExpectTx(bar, kTok * 2 * 32 * 4);
#pragma unroll
        for (int t = 0; t < kTok; ++t)
        {
            bulkCopy(dst + t * 256, cs + static_cast<size_t>(pos[t]) * 64, 256, bar);
        }
    }
    meta->L = L;
#pragma unroll
    for (int t = 0; t < kTok; ++t)
    {
        meta->page[t] = page[t];
        meta->pos[t] = pos[t];
    }
}

// ---- end kIFlagI2bEntryLite ----

__device__ __noinline__ void ahEntryIssue(Ctx const& ctx, int role, uint32_t bar)
{
    // ---- kIFlagI2bEntryLite ----
#if MSO_MK_LD_SW
    if (!(ctx.p[kPIFlags] & kIFlagI2bEntryLite))
    {
        __trap(); // MSO_MK_LD_SW: the build fixes kIFlagI2bEntryLite to the served SWA layers' 1
    }
#endif
    if ((MSO_MK_SW_ENTRY_LITE(icPIFlags(ctx.p[kPIFlags])) & kIFlagI2bEntryLite) && (role >> 16) == kStAhQkvFin)
    {
        ahEntryIssueFinLite(ctx, role & 0xffff, bar);
        return;
    }
    // ---- end kIFlagI2bEntryLite ----
    int const stage = role >> 16;
    int const unit = role & 0xffff;
    AhMeta* meta = ahMeta(ctx);
    bool const evictFirst = (MSO_MK_KFLAGS(ctx.flags) & kFlagEvictFirst) != 0; // MSO_MK_LD_KF
    uint64_t const policy = policyEvictFirst();
    int const L = *ptr<int const>(ctx, kPAhSeqLen);
    meta->L = L;
    int const* bt = ptr<int const>(ctx, kPAhBlockTable);
    int const btStride = static_cast<int>(ctx.p[kPAhBtStride]);
    uint8_t const* pool = ptr<uint8_t const>(ctx, kPAhPool);
    uint32_t const dst = ctx.smemBase + kAhEntryOff;
    if (stage == kStAhSwaAttn)
    {
        int const h = unit / kAhBlocks;
        int const b = unit % kAhBlocks;
        AhWindow const w = ahWindow(L, static_cast<int>(ctx.p[kPAhWindow]));
        int const p0 = (w.base + kAhBlockKeys * b) / kAhPageTok;
        uint8_t const* src[4] = {};
        uint32_t bytes = 0;
        int mask = 0;
        if (b < w.nBlocks)
        {
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                int const p = p0 + (i & 1);
                // Pages holding a key before the step; the step's own rows come from the arena.
                if (p * kAhPageTok < w.past)
                {
                    int const blk = bt[(i >> 1) * btStride + p];
                    src[i] = pool + static_cast<size_t>(blk) * kAhPageBytes + h * (kAhPageTok * kAhD);
                    mask |= 1 << i;
                    bytes += kAhPageTok * kAhD;
                }
            }
        }
        meta->pages = mask;
        mbarExpectTx(bar, bytes);
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            if (mask & (1 << i))
            {
                uint32_t const d = dst + i * (kAhPageTok * kAhD);
                if (evictFirst)
                {
                    bulkCopyHint(d, src[i], kAhPageTok * kAhD, bar, policy);
                }
                else
                {
                    bulkCopy(d, src[i], kAhPageTok * kAhD, bar);
                }
            }
        }
    }
    else if (stage == kStAhQkvFin)
    {
        int const b = unit;
        bool const rope = b < kAhBlkV;
        bool const cache = b >= kAhBlkK && b < kAhBlkG;
        if (cache)
        {
            int const kv = b < kAhBlkV ? 0 : 1;
#pragma unroll
            for (int t = 0; t < kTok; ++t)
            {
                meta->page[t] = bt[kv * btStride + (L - kTok + t) / kAhPageTok];
            }
        }
        if (rope)
        {
            int const* positions = ptr<int const>(ctx, kPAhPositions);
            float const* cs = ptr<float const>(ctx, kPAhCosSin);
            int pos[kTok];
#pragma unroll
            for (int t = 0; t < kTok; ++t)
            {
                pos[t] = positions[t];
                meta->pos[t] = pos[t];
            }
            mbarExpectTx(bar, kTok * 2 * 32 * 4);
#pragma unroll
            for (int t = 0; t < kTok; ++t)
            {
                bulkCopy(dst + t * 256, cs + static_cast<size_t>(pos[t]) * 64, 256, bar);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// AhQkvFin: unit b = packed projection block b (heads first, see kAhBlkK)
// ---------------------------------------------------------------------------------------------------------------------
__device__ __noinline__ void stageAhQkvFin(Ctx const& ctx, int unit, uint32_t bar, Io const& io)
{
    int const tid = threadIdx.x;
    AhMeta const* meta = ahMeta(ctx);
    float* X = reinterpret_cast<float*>(ctx.smem + kSmemRed); // [8 tokens][64] BF16 values of the head
    uint4 v[kQkvMaxRuns];
    uint8_t const* a[kQkvMaxRuns] = {};
    if (tid < 128)
    {
        int const tile = kQkvTilesPerBlock * unit + tid / 32;
        int const q = tid % 32;
        uint8_t const* base = ptr<uint8_t const>(ctx, kPQkvPart);
#pragma unroll
        for (int sl = 0; sl < kQkvSlices; ++sl)
        {
            int const g = sl * kQkvTiles + tile;
            int const cFirst = qkvCtaOfChunk(g * kSegChunks);
            int const cLast = qkvCtaOfChunk(g * kSegChunks + kSegChunks - 1);
#pragma unroll
            for (int k = 0; k < kQkvRunsPerSeg; ++k)
            {
                int const c = cFirst + k;
                if (c <= cLast)
                {
                    int const slot = g - qkvRunStart(c) / kSegChunks;
                    a[sl * kQkvRunsPerSeg + k]
                        = base + (static_cast<size_t>(c) * kQkvMaxSlots + slot) * kQkvPartSlotBytes + 16 * q;
                }
            }
        }
        pollVecs<kQkvMaxRuns>(ctx, v, a, kCodeAhRuns);
        stampInput(ctx);
        float sum[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int k = 0; k < kQkvMaxRuns; ++k)
        {
            if (a[k] != nullptr)
            {
                sum[0] += __uint_as_float(v[k].x);
                sum[1] += __uint_as_float(v[k].y);
                sum[2] += __uint_as_float(v[k].z);
                sum[3] += __uint_as_float(v[k].w);
            }
        }
        int const tok = q / 4;
        int const col = 16 * (tid / 32) + 4 * (q % 4);
        float4 const r = make_float4(bf16Round(sum[0]), bf16Round(sum[1]), bf16Round(sum[2]), bf16Round(sum[3]));
        *reinterpret_cast<float4*>(X + tok * kAhD + col) = r;
        if (MSO_MK_CKIO(io.qkv) != nullptr) // MSO_MK_LD_CK
        {
            uint2 const o = make_uint2(packBf16(r.x, r.y), packBf16(r.z, r.w));
            st64(static_cast<uint8_t*>(io.qkv) + (static_cast<size_t>(tok) * kQkvRows + ahCkptCol(unit) + col) * 2, o);
        }
    }
    __syncthreads();
    int const b = unit;
    bool const isQ = b < kAhBlkK;
    bool const isGate = b >= kAhBlkG;
    bool const isK = b >= kAhBlkK && b < kAhBlkV;
    float const sBf16 = bf16Round(f32Param(ctx, kPAhKvScale));
    if ((isQ || isK) && tid < 64)
    {
        waitBar(ctx, bar, 0, kCodeAhRope);
    }
    if (tid < 64)
    {
        int const t = tid / 8;
        int const l = tid % 8;
        float x[8];
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            x[i] = X[t * kAhD + 8 * l + i];
        }
        uint8_t* arenaBase = ctx.arena;
        if (isQ || isK)
        {
            // The norm in the stock thread order: eight elements in order, then lane bits 0, 1, 2.
            float s = __fmul_rn(x[0], x[0]);
#pragma unroll
            for (int i = 1; i < 8; ++i)
            {
                s = __fadd_rn(s, __fmul_rn(x[i], x[i]));
            }
            s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, 1));
            s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, 2));
            s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, 4));
            float const inv = rsqrtApproxFtz(__fadd_rn(__fmul_rn(s, 1.f / 64.f), f32Param(ctx, kPAhPrepEps)));
            float const scale = isQ ? f32Param(ctx, kPAhQScale) : 1.f;
            float y[8];
#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                y[i] = bf16Round(__fmul_rn(bf16Round(__fmul_rn(x[i], inv)), scale));
            }
            float const* cs = reinterpret_cast<float const*>(ctx.smem + kAhEntryOff + t * 256);
#pragma unroll
            for (int p = 0; p < 4; ++p)
            {
                int const pair = 4 * l + p;
                float const c = cs[pair];
                float const sn = cs[32 + pair];
                float const e = y[2 * p];
                float const o = y[2 * p + 1];
                y[2 * p] = bf16Round(fmaRnFtz(e, c, mulRnFtz(-o, sn)));
                y[2 * p + 1] = bf16Round(fmaRnFtz(o, c, mulRnFtz(e, sn)));
            }
            if (isQ)
            {
                int const qh = b;
#if MSO_MK_LD_SW
                if (ctx.p[kPAhQFp8] == 0)
                {
                    __trap(); // MSO_MK_LD_SW: the build fixes kPAhQFp8 to the served 1
                }
                bool const qFp8 = true;
#else
                bool const qFp8 = ctx.p[kPAhQFp8] != 0;
#endif
                float qv[8];
#pragma unroll
                for (int i = 0; i < 8; ++i)
                {
                    qv[i] = qFp8 ? e4m3Value(toCacheScaled(y[i], sBf16)) : y[i];
                }
                uint4 const w = make_uint4(
                    packBf16(qv[0], qv[1]), packBf16(qv[2], qv[3]), packBf16(qv[4], qv[5]), packBf16(qv[6], qv[7]));
                int const row = t * kAhGroup + qh % kAhGroup;
                st128(arenaBase + kArAhQ + ((qh / kAhGroup) * kAhRows + row) * kAhD * 2 + 16 * l, lamport4(w));
            }
            else
            {
#pragma unroll
                for (int i = 0; i < 8; ++i)
                {
                    x[i] = y[i];
                }
            }
        }
        if (isK || (!isQ && !isGate))
        {
            int const kv = isK ? 0 : 1;
            int const h = b - kAhBlkK - kv * kAhKvHeads;
            // Bytes 0-3 from elements 0-3, bytes 4-7 from elements 4-7.
            uint32_t const w0 = e4m3x2(toCacheScaled(x[0], sBf16), toCacheScaled(x[1], sBf16))
                | (e4m3x2(toCacheScaled(x[2], sBf16), toCacheScaled(x[3], sBf16)) << 16);
            uint32_t const w1 = e4m3x2(toCacheScaled(x[4], sBf16), toCacheScaled(x[5], sBf16))
                | (e4m3x2(toCacheScaled(x[6], sBf16), toCacheScaled(x[7], sBf16)) << 16);
            int const pos = meta->L - kTok + t;
            uint8_t* pool = ptr<uint8_t>(ctx, kPAhPool) + static_cast<size_t>(meta->page[t]) * kAhPageBytes
                + (h * kAhPageTok + pos % kAhPageTok) * kAhD + 8 * l;
            st64(pool, make_uint2(w0, w1));
            st64(arenaBase + (kv == 0 ? kArAhKNew : kArAhVNew) + (h * kTok + t) * kAhD + 8 * l,
                make_uint2(lamport(w0), lamport(w1)));
        }
        else if (isGate)
        {
            int const qh = b - kAhBlkG;
            float sg[8];
#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                sg[i] = 1.f / (1.f + expf(-x[i]));
            }
            uint4 const w = make_uint4(
                packBf16(sg[0], sg[1]), packBf16(sg[2], sg[3]), packBf16(sg[4], sg[5]), packBf16(sg[6], sg[7]));
            st128(arenaBase + kArAhSig + (t * kAhOCols + qh * kAhD) * 2 + 16 * l, lamport4(w));
        }
    }
    // The run partials this unit read, emptied after its outputs (single reader).
    if (tid < 128)
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
}

// ---------------------------------------------------------------------------------------------------------------------
// AhSwaAttn + merge + norm-gate
// ---------------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ uint8_t* ahPartO(Ctx const& ctx, int h, int b, int row)
{
    return ptr<uint8_t>(ctx, kPAhPart) + ((static_cast<size_t>(h * kAhBlocks + b) * kAhRows + row) * kAhD) * 4;
}

__device__ __forceinline__ uint8_t* ahPartMl(Ctx const& ctx, int h, int b, int row)
{
    return ptr<uint8_t>(ctx, kPAhPart) + kAhPartOBytes
        + ((static_cast<size_t>(h * kAhBlocks + b) * kAhRows + row) * 2) * 4;
}

// ---- Attribution stamps of the SWA block pass and its merge, written by thread `who` in the stamped builds with
// sub-stamps (not lite, not lean; no code in the timed builds). In an SWA launch a SwaAttn CTA also runs MoeExperts,
// which writes sub-stamps 5 and 8-31 (experts_defs.cuh kMoeSt*), and no MhH1 / MhF1 / FaAttn unit, so the free words
// are sub-stamp 7 and the four FaAttn prologue words (kStampFa): word k < 0 is kStampFa + (-1 - k), k >= 0 sub-stamp k.
// Map: sub-stamp 7 warp 7's merge poll done (thread 224: GM's gate partials, or the gate fin's sig); kStampFa + 0 the
// merge's entry (after the block's barrier); kStampFa + 1 after the merge's first barrier (weights + sig in shared
// memory); kStampFa + 2 after its second barrier (split-group sums); kStampFa + 3 the block's P V done (the barrier
// before the half merge). ----
__device__ __forceinline__ void ah2Sub(Ctx const& ctx, int k, int who)
{
    if (MSO_MK_STAMPS_DETAIL(ctx) != nullptr && threadIdx.x == who)
    {
        MSO_MK_STAMPS(ctx)[k < 0 ? kStampFa - 1 - k : kStampSub + k] = stampTime(); // %globaltimer or clock64
    }
}

// ---- end attribution stamps ----

#if MSO_MK_AH & 128
// ---- GS (MSO_MK_AH bit 128): qkvRunStart / qkvCtaOfChunk in 32-bit unsigned arithmetic (the products stay below
// 2^31, so the same integers as the 64-bit forms), and the merge's poll: pollVecs (its default form) with kW 32-bit
// words w[i] at aw[i] (nullptr: none, value 0) in flight together with the vectors, the same exit test over both. ----
static_assert(static_cast<int64_t>(kQkvChunks) * kCtas < (int64_t{1} << 31), "GS: 32-bit run arithmetic");

__device__ __forceinline__ int ahRunStart32(int c)
{
    return static_cast<int>(
        (static_cast<uint32_t>(kQkvChunks) * static_cast<uint32_t>(c)) / static_cast<uint32_t>(kCtas));
}

__device__ __forceinline__ int ahCtaOfChunk32(int j)
{
    return static_cast<int>(
        (static_cast<uint32_t>(kCtas) * static_cast<uint32_t>(j + 1) - 1u) / static_cast<uint32_t>(kQkvChunks));
}

template <int kN, int kW>
__device__ __forceinline__ void ahPollVecsWords(Ctx const& ctx, uint4 (&v)[kN], uint8_t const* const (&a)[kN],
    uint32_t (&w)[kW], uint8_t const* const (&aw)[kW], int code)
{
#pragma unroll
    for (int i = 0; i < kN; ++i)
    {
        v[i] = a[i] != nullptr ? ldPoll128(a[i]) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int i = 0; i < kW; ++i)
    {
        w[i] = aw[i] != nullptr ? ldPoll32(aw[i]) : 0u;
    }
    uint64_t* const sy = syProbe(ctx);                      // poll probe
    uint64_t const syC0 = sy != nullptr ? clock64() : 0ull; // poll probe
    uint64_t syC1 = 0;                                      // poll probe
    uint32_t syRounds = 1;                                  // poll probe
    uint64_t t0 = 0;
    while (true)
    {
        bool ok = true;
#pragma unroll
        for (int i = 0; i < kN; ++i)
        {
            ok &= full4(v[i]);
        }
#pragma unroll
        for (int i = 0; i < kW; ++i)
        {
            ok &= w[i] != kEmpty;
        }
        if (warpAll(ok))
        {
            break;
        }
        spinCheck(ctx, t0, code, kN + kW, 0);
#pragma unroll
        for (int i = 0; i < kN; ++i)
        {
            if (!full4(v[i]))
            {
                v[i] = ldPoll128(a[i]);
            }
        }
#pragma unroll
        for (int i = 0; i < kW; ++i)
        {
            if (w[i] == kEmpty)
            {
                w[i] = ldPoll32(aw[i]);
            }
        }
        syProbeRound(sy, syRounds, syC1);                            // poll probe
        ++syRounds;                                                  // poll probe
    }
    syProbeEnd(sy, syClass(true, code), syC0, syC1, syRounds, code); // poll probe
}

// ---- end GS ----
#endif

#if !(MSO_MK_AH & 512) // ME (MSO_MK_AH bit 512): the half-1 merge below
// Rows {2b, 2b + 1} of KV head h over the 33 partials, then the norm-gate, into the arena's gated input.
// Warp 0 holds the 33 splits' (m, l) (lane s; lane 0 also split 32) and forms max m, the weights w_s and the
// denominators with shuffles; every thread then weighs its partial vectors.
// WD (MSO_MK_AH bit 16): `dry` runs the same code with every polled address null (pollVecs returns zeros at once),
// no global store and no stamp; it writes only its shared-memory scratch (kSmemRed), which the real pass rewrites
// before it reads it, so the real pass that follows runs from a warm instruction cache.
#if MSO_MK_AH & 16
__device__ __noinline__ void ahMerge(Ctx const& ctx, int h, int b, Io const& io, bool const dry)
#else
__device__ __noinline__ void ahMerge(Ctx const& ctx, int h, int b, Io const& io)
#endif
{
#if !(MSO_MK_AH & 16)
    constexpr bool dry = false;
#endif
    int const tid = threadIdx.x;
    int const sg = tid >> 5;       // split group: splits sg, sg + 8, ...
    int const rr = (tid >> 4) & 1; // row of the pair
    int const dq = tid & 15;       // dims 4 dq .. 4 dq + 3
    constexpr int kPer = (kAhBlocks + 7) / 8;
    constexpr int kN = kPer + 2;
    static_assert(kAhBlocks == 33, "warp 0 holds splits 0-31, lane 0 also split 32");
    if (!dry)
    {
        ah2Sub(ctx, -1, 0); // merge entry (kStampFa + 0)
    }
    int const row0 = 2 * b;
    int const t = row0 / kAhGroup;
    int const g0 = row0 % kAhGroup;
    int const qh0 = h * kAhGroup + g0;
    // ---- GM (MSO_MK_AH bit 1): the gate of rows (t, qh0) and (t, qh0 + 1) from the gate runs' partials, polled with
    // the 33 partials. Warp 7, lane j: gate head qh0 + j / 16, tile (j / 4) % 4 of its weight block kAhBlkG + qh,
    // float4 q = 4 t + j % 4 (rows 4 (j % 4) .. + 3 of the tile for token t) of every run partial that covers the tile:
    // the addresses and the sum order of stageAhQkvFin. Single reader: (t, qh) is merged by this unit only, and the
    // host plan holds no gate AhQkvFin unit. ----
#if (MSO_MK_AH & 1) && !(MSO_MK_AH & 128)
    constexpr int kGv = kQkvMaxRuns;
#else
    constexpr int kGv = 0;
#endif
#if MSO_MK_AH & 1024 // MX (MSO_MK_AH bit 1024): slots kN .. kN + kPer - 1 of thread 128 + p, thread p's O vectors
    constexpr int kMe = kPer;
#else
    constexpr int kMe = 0;
#endif
    uint4 v[kN + kGv + kMe];
    uint8_t const* a[kN + kGv + kMe] = {};
#if MSO_MK_AH & 128 // GS: threads 128-255, the 8 run-partial words of gate value (r, d)
    int const gr = (tid >> 6) & 1;
    int const gd = tid & 63;
    uint32_t gw[kQkvMaxRuns];
    uint8_t const* aw[kQkvMaxRuns] = {};
#endif
#pragma unroll
    for (int k = 0; k < kPer; ++k)
    {
        int const s = sg + 8 * k;
        if (s < kAhBlocks)
        {
            a[k] = ahPartO(ctx, h, s, row0 + rr) + 16 * dq;
        }
    }
    if (tid < 32)
    {
        a[kPer] = ahPartMl(ctx, h, tid, row0);
        if (tid == 0)
        {
            a[kPer + 1] = ahPartMl(ctx, h, 32, row0);
        }
    }
#if MSO_MK_AH & 128
    // ---- GS: thread (gr, gd) of threads 128-255 holds gate value (t, qh0 + gr, dim gd): tile gd / 16 of the gate
    // head's weight block, float4 q = 4 t + (gd / 4) % 4, word gd % 4 (warp 7 lane j = 16 gr + 4 (gd / 16) + (gd / 4) %
    // 4 of the GM form below, component gd % 4). ----
    else if (tid >= 128)
    {
        int const tile = kQkvTilesPerBlock * (kAhBlkG + qh0 + gr) + (gd >> 4);
        int const q = 4 * t + ((gd >> 2) & 3);
        uint8_t const* base = ptr<uint8_t const>(ctx, kPQkvPart);
#pragma unroll
        for (int sl = 0; sl < kQkvSlices; ++sl)
        {
            int const g = sl * kQkvTiles + tile;
            int const cFirst = ahCtaOfChunk32(g * kSegChunks);
            int const cLast = ahCtaOfChunk32(g * kSegChunks + kSegChunks - 1);
#pragma unroll
            for (int k = 0; k < kQkvRunsPerSeg; ++k)
            {
                int const c = cFirst + k;
                if (c <= cLast)
                {
                    int const slot = g - ahRunStart32(c) / kSegChunks;
                    aw[sl * kQkvRunsPerSeg + k] = base
                        + (static_cast<size_t>(c) * kQkvMaxSlots + slot) * kQkvPartSlotBytes + 16 * q + 4 * (gd & 3);
                }
            }
        }
    }
    // ---- end GS ----
#elif MSO_MK_AH & 1
    else if (tid >= 224)
    {
        int const j = tid - 224;
        int const tile = kQkvTilesPerBlock * (kAhBlkG + qh0 + (j >> 4)) + ((j >> 2) & 3);
        int const q = 4 * t + (j & 3);
        uint8_t const* base = ptr<uint8_t const>(ctx, kPQkvPart);
#pragma unroll
        for (int sl = 0; sl < kQkvSlices; ++sl)
        {
            int const g = sl * kQkvTiles + tile;
            int const cFirst = qkvCtaOfChunk(g * kSegChunks);
            int const cLast = qkvCtaOfChunk(g * kSegChunks + kSegChunks - 1);
#pragma unroll
            for (int k = 0; k < kQkvRunsPerSeg; ++k)
            {
                int const c = cFirst + k;
                if (c <= cLast)
                {
                    int const slot = g - qkvRunStart(c) / kSegChunks;
                    a[kN + sl * kQkvRunsPerSeg + k]
                        = base + (static_cast<size_t>(c) * kQkvMaxSlots + slot) * kQkvPartSlotBytes + 16 * q;
                }
            }
        }
    }
#else
    else if (tid >= 240)
    {
        a[kPer + 1] = ctx.arena + kArAhSig + (t * kAhOCols + qh0 * kAhD) * 2 + 16 * (tid - 240);
    }
#endif
    // ---- end GM ----
#if MSO_MK_AH & 1024
    // ---- MX: half 1 polls for half 0. Thread 128 + p holds thread p's slots (split group sg - 4, the same rr and
    // dq); warp 4 lane l also warp 0 lane l's (m, l) slots (lane 0: split 32's too). Half 0 polls nothing. ----
    if (tid < 128)
    {
#pragma unroll
        for (int k = 0; k < kN; ++k)
        {
            a[k] = nullptr;
        }
    }
    else
    {
#pragma unroll
        for (int k = 0; k < kPer; ++k)
        {
            int const s = sg - 4 + 8 * k;
            if (s < kAhBlocks)
            {
                a[kN + k] = ahPartO(ctx, h, s, row0 + rr) + 16 * dq;
            }
        }
        if (tid < 160)
        {
            a[kPer] = ahPartMl(ctx, h, tid - 128, row0);
            if (tid == 128)
            {
                a[kPer + 1] = ahPartMl(ctx, h, 32, row0);
            }
        }
    }
    // ---- end MX ----
#endif
#if MSO_MK_AH & 16 // WD: the dry pass polls nothing (and so resets nothing)
    if (dry)
    {
#pragma unroll
        for (int k = 0; k < kN + kGv; ++k)
        {
            a[k] = nullptr;
        }
    }
#endif
#if MSO_MK_AH & 128 // GS
    ahPollVecsWords<kN + kMe, kQkvMaxRuns>(ctx, v, a, gw, aw, kCodeAhMerge);
#else
    pollVecs<kN + kGv>(ctx, v, a, kCodeAhMerge);
#endif
    if (!dry)
    {
#if !(MSO_MK_AH & 1024)      // MX: after the hand-off below
        subStamp(ctx, 5);    // merge inputs in (sub-stamps 0-2 belong to the QKV run)
#endif
        ah2Sub(ctx, 7, 224); // warp 7's poll done (sub-stamp 7)
    }
    float* wts = reinterpret_cast<float*>(ctx.smem + kSmemRed);        // [33][2] weights, then den[2]
    float* sig = reinterpret_cast<float*>(ctx.smem + kSmemRed + 1024); // [2][64]
    float* red = reinterpret_cast<float*>(ctx.smem + kSmemRed + 2048); // [8][2][64]
    float* ss = reinterpret_cast<float*>(ctx.smem + kSmemRed + 6144);  // [4]
#if MSO_MK_AH & 1024
    // ---- MX: the hand-off. hv [k][p] holds thread p's O vector k (k = kPer - 1: p < 32 only), hv[kHvMl + l]
    // warp 0 lane l's (m, l) and hv[kHvMl + 32] split 32's, in the reduction area after ss (free while the unit runs;
    // every word is written before the barrier and read after it, within this merge). ----
    {
        constexpr int kHvOff = 6400;
        constexpr int kHvMl = 128 * (kPer - 1) + 32;
        static_assert(kHvOff >= 6144 + 16 && kHvOff + 16 * (kHvMl + 33) <= kSmemRedBytes, "ME hand-off area");
        uint4* const hv = reinterpret_cast<uint4*>(ctx.smem + kSmemRed + kHvOff);
        if (tid >= 128)
        {
#pragma unroll
            for (int k = 0; k < kPer; ++k)
            {
                if (a[kN + k] != nullptr)
                {
                    hv[128 * k + tid - 128] = v[kN + k];
                }
            }
            if (tid < 160)
            {
                hv[kHvMl + tid - 128] = v[kPer];
                if (tid == 128)
                {
                    hv[kHvMl + 32] = v[kPer + 1];
                }
            }
        }
        __syncthreads();
        if (tid < 128)
        {
#pragma unroll
            for (int k = 0; k < kPer; ++k)
            {
                if (sg + 8 * k < kAhBlocks)
                {
                    v[k] = hv[128 * k + tid];
                }
            }
            if (tid < 32)
            {
                v[kPer] = hv[kHvMl + tid];
                if (tid == 0)
                {
                    v[kPer + 1] = hv[kHvMl + 32];
                }
            }
        }
        subStamp(ctx, 5); // merge inputs in
    }
    // ---- end MX ----
#endif
    if (tid < 32)
    {
        uint4 const s0 = v[kPer];
        uint4 const s1
            = tid == 0 ? v[kPer + 1] : make_uint4(__float_as_uint(-1.0e30f), 0u, __float_as_uint(-1.0e30f), 0u);
#pragma unroll
        for (int r = 0; r < 2; ++r)
        {
            float const m0 = __uint_as_float(r == 0 ? s0.x : s0.z);
            float const l0 = __uint_as_float(r == 0 ? s0.y : s0.w);
            float const m1 = __uint_as_float(r == 0 ? s1.x : s1.z);
            float const l1 = __uint_as_float(r == 0 ? s1.y : s1.w);
            float mx = fmaxf(l0 > 0.f ? m0 : -1.0e30f, l1 > 0.f ? m1 : -1.0e30f);
#pragma unroll
            for (int m = 16; m >= 1; m >>= 1)
            {
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, m));
            }
            float const w0 = l0 > 0.f ? ex2Approx(m0 - mx) : 0.f;
            float const w1 = l1 > 0.f ? ex2Approx(m1 - mx) : 0.f;
            float den = w0 * l0 + w1 * l1;
#pragma unroll
            for (int m = 16; m >= 1; m >>= 1)
            {
                den += __shfl_xor_sync(0xffffffffu, den, m);
            }
            wts[2 * tid + r] = w0;
            if (tid == 0)
            {
                wts[2 * 32 + r] = w1;
                wts[2 * 33 + r] = den;
            }
        }
    }
#if MSO_MK_AH & 128
    else if (tid >= 128) // GS: GM's gate statements for the one value (gr, gd)
    {
        float sum = 0.f;
#pragma unroll
        for (int k = 0; k < kQkvMaxRuns; ++k)
        {
            if (aw[k] != nullptr)
            {
                sum += __uint_as_float(gw[k]);
            }
        }
        float const xg = bf16Round(sum);
        sig[gr * kAhD + gd] = bf16Round(1.f / (1.f + expf(-xg)));
        if (MSO_MK_CKIO(io.qkv) != nullptr) // MSO_MK_LD_CK (the gate projection, as the gate fin stored it)
        {
            float const xgHi = __shfl_down_sync(0xffffffffu, xg, 1);
            if ((gd & 1) == 0)
            {
                st32(static_cast<uint8_t*>(io.qkv)
                        + (static_cast<size_t>(t) * kQkvRows + ahCkptCol(kAhBlkG + qh0 + gr) + gd) * 2,
                    packBf16(xg, xgHi));
            }
        }
    }
#elif MSO_MK_AH & 1
    else if (tid >= 224) // GM: stageAhQkvFin's gate statements (sum from +0 in slot order, BF16, sigmoid, BF16)
    {
        int const j = tid - 224;
        float sum[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int k = 0; k < kGv; ++k)
        {
            if (a[kN + k] != nullptr)
            {
                sum[0] += __uint_as_float(v[kN + k].x);
                sum[1] += __uint_as_float(v[kN + k].y);
                sum[2] += __uint_as_float(v[kN + k].z);
                sum[3] += __uint_as_float(v[kN + k].w);
            }
        }
        int const col = 16 * ((j >> 2) & 3) + 4 * (j & 3);
        float xg[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            xg[i] = bf16Round(sum[i]);
            sig[(j >> 4) * kAhD + col + i] = bf16Round(1.f / (1.f + expf(-xg[i])));
        }
        if (MSO_MK_CKIO(io.qkv) != nullptr && !dry) // MSO_MK_LD_CK (the gate projection, as the gate fin stored it); WD
        {
            uint2 const o = make_uint2(packBf16(xg[0], xg[1]), packBf16(xg[2], xg[3]));
            st64(static_cast<uint8_t*>(io.qkv)
                    + (static_cast<size_t>(t) * kQkvRows + ahCkptCol(kAhBlkG + qh0 + (j >> 4)) + col) * 2,
                o);
        }
    }
#else
    else if (tid >= 240)
    {
        uint4 const s = v[kPer + 1];
        int const e = 8 * (tid - 240);
        sig[e + 0] = bf16Lo(s.x);
        sig[e + 1] = bf16Hi(s.x);
        sig[e + 2] = bf16Lo(s.y);
        sig[e + 3] = bf16Hi(s.y);
        sig[e + 4] = bf16Lo(s.z);
        sig[e + 5] = bf16Hi(s.z);
        sig[e + 6] = bf16Lo(s.w);
        sig[e + 7] = bf16Hi(s.w);
    }
#endif
    __syncthreads();
    if (!dry)
    {
        ah2Sub(ctx, -2, 0); // weights and gate in shared memory (kStampFa + 1)
    }
    float4 num = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int k = 0; k < kPer; ++k)
    {
        int const s = sg + 8 * k;
        if (s < kAhBlocks)
        {
            float const w = wts[2 * s + rr];
            num.x += w * __uint_as_float(v[k].x);
            num.y += w * __uint_as_float(v[k].y);
            num.z += w * __uint_as_float(v[k].z);
            num.w += w * __uint_as_float(v[k].w);
        }
    }
    *reinterpret_cast<float4*>(red + (sg * 2 + rr) * kAhD + 4 * dq) = num;
    __syncthreads();
    if (!dry)
    {
        ah2Sub(ctx, -3, 0); // split-group sums in shared memory (kStampFa + 2)
    }
    if (tid < 128)
    {
        int const r = tid >> 6;
        int const d = tid & 63;
        float n = 0.f;
#pragma unroll
        for (int k = 0; k < 8; ++k)
        {
            n += red[(k * 2 + r) * kAhD + d];
        }
        float const att = bf16Round(n / wts[2 * 33 + r]);
        int const qh = qh0 + r;
        uint8_t* attnOut = MSO_MK_CKPTR(uint8_t, ctx, kPAhAttnOut); // MSO_MK_LD_CK
        float sq = att * att;
#pragma unroll
        for (int m = 16; m >= 1; m >>= 1)
        {
            sq += __shfl_xor_sync(0xffffffffu, sq, m);
        }
        if ((tid & 31) == 0)
        {
            ss[tid >> 5] = sq;
        }
        // The two warps of the row (named barrier over threads 0-127).
        asm volatile("bar.sync 1, 128;" ::: "memory");
        float const var = (ss[2 * r] + ss[2 * r + 1]) * (1.f / 64.f);
        float const normed = bf16Round(att * rsqrtf(var + f32Param(ctx, kPAhGateEps)));
        float const out = bf16Round(normed * sig[r * kAhD + d]);
        float const outHi = __shfl_down_sync(0xffffffffu, out, 1);
        float const attHi = __shfl_down_sync(0xffffffffu, att, 1);
        if ((d & 1) == 0 && !dry) // WD: the dry pass stores nothing
        {
            int const reps = static_cast<int>(ctx.p[kPAhOReplicas]);
            uint32_t const word = lamport(packBf16(out, outHi));
#pragma unroll 1
            for (int rep = 0; rep < reps; ++rep)
            {
                st32(ctx.arena + kArAhO + rep * (kTok * kAhOCols * 2) + (t * kAhOCols + qh * kAhD + d) * 2, word);
            }
            if (attnOut != nullptr)
            {
                st32(attnOut + (t * kAhOCols + qh * kAhD + d) * 2, packBf16(att, attHi));
            }
        }
    }
    __syncthreads();
    if (dry) // WD: no stamp, no reset
    {
        return;
    }
    subStamp(ctx, 6); // gated output stored
    // The partials this unit read, emptied after its output (single reader); slot kPer + 1 of thread 0 is split 32's
    // (m, l), of threads 240+ the gate (arena, not cleared here).
#if MSO_MK_AH & 1024 // MX: each slot emptied by the thread that polled it (threads 128-255; half 0 holds none)
#pragma unroll
    for (int k = 0; k < kN + kMe; ++k)
    {
        if (a[k] != nullptr)
        {
            st128(const_cast<uint8_t*>(a[k]), emptyVec());
        }
    }
#else
#pragma unroll
    for (int k = 0; k <= kPer; ++k)
    {
        if (a[k] != nullptr)
        {
            st128(const_cast<uint8_t*>(a[k]), emptyVec());
        }
    }
    if (tid == 0)
    {
        st128(const_cast<uint8_t*>(a[kPer + 1]), emptyVec());
    }
#endif
#if MSO_MK_AH & 128 // GS: each gate word emptied by the thread that read it (the bytes of GM's 16-byte resets)
#pragma unroll
    for (int k = 0; k < kQkvMaxRuns; ++k)
    {
        if (aw[k] != nullptr)
        {
            st32(const_cast<uint8_t*>(aw[k]), kEmpty);
        }
    }
#elif MSO_MK_AH & 1 // GM: the gate partials, emptied by their single reader (as stageAhQkvFin does without GM)
#pragma unroll
    for (int k = 0; k < kGv; ++k)
    {
        if (a[kN + k] != nullptr)
        {
            st128(const_cast<uint8_t*>(a[kN + k]), emptyVec());
        }
    }
#endif
}

#else
// ---- ME (MSO_MK_AH bit 512, with GS): the merge on key half 1 alone (threads 128-255), which enters it right after
// the block pass's key-half staging barrier while half 0 combines and publishes (stageAhSwaAttn), with the statements
// of the merge above on the same values: thread 128 + p does thread p's part and its own. Thread (q, rr, dq),
// q = warp - 4: rows 2b + rr, dims 4 dq .. 4 dq + 3 of split groups q + 4 (splits q + 4 + 8 k, k < 4: its own part in
// the merge above) and q (splits q + 8 k, k < 4, and split 32 for q = 0: thread p's part); warp 4 does warp 0's (m, l)
// trees (lane l: split l, lane 0 also split 32); GS's gate value (gr, gd) of thread 128 + 64 gr + gd is the sig of the
// final value (r, d) = (gr, gd) the same thread forms, so it stays in a register. Barriers: bar 1 over the 128 threads
// (half 0 waits in stageAhSwaAttn's closing __syncthreads, so no other bar 1 user runs meanwhile). Each slot is emptied
// by the thread that polled it (single reader). ----
__device__ __noinline__ void ahMerge(Ctx const& ctx, int h, int b, Io const& io)
{
    int const tid = threadIdx.x;
    int const q = (tid >> 5) - 4;  // split groups q (thread tid - 128's in the merge above) and q + 4 (this thread's)
    int const rr = (tid >> 4) & 1; // row of the pair
    int const dq = tid & 15;       // dims 4 dq .. 4 dq + 3
    int const lane = tid & 31;
    constexpr int kOwn = 4;        // group q + 4: splits q + 4 + 8 k, k < 4 (all < 32)
    constexpr int kPart = (kAhBlocks + 7) / 8; // group q: splits q + 8 k, k < 5 (k = 4: split 32, q = 0 only)
    constexpr int kMl = kOwn + kPart;          // warp 4: (m, l) of split lane, lane 0 also split 32
    constexpr int kN = kMl + 2;
    static_assert(kAhBlocks == 33, "warp 4 holds splits 0-31, lane 0 also split 32");
    ah2Sub(ctx, -1, 128); // merge entry (kStampFa + 0)
    int const row0 = 2 * b;
    int const t = row0 / kAhGroup;
    int const g0 = row0 % kAhGroup;
    int const qh0 = h * kAhGroup + g0;
    uint4 v[kN];
    uint8_t const* a[kN] = {};
    int const gr = (tid >> 6) & 1;
    int const gd = tid & 63;
    uint32_t gw[kQkvMaxRuns];
    uint8_t const* aw[kQkvMaxRuns] = {};
#pragma unroll
    for (int k = 0; k < kOwn; ++k)
    {
        a[k] = ahPartO(ctx, h, q + 4 + 8 * k, row0 + rr) + 16 * dq;
    }
#pragma unroll
    for (int k = 0; k < kPart; ++k)
    {
        if (q + 8 * k < kAhBlocks)
        {
            a[kOwn + k] = ahPartO(ctx, h, q + 8 * k, row0 + rr) + 16 * dq;
        }
    }
    if (q == 0)
    {
        a[kMl] = ahPartMl(ctx, h, lane, row0);
        if (lane == 0)
        {
            a[kMl + 1] = ahPartMl(ctx, h, 32, row0);
        }
    }
    {
        // GS's gate addresses of value (gr, gd)
        int const tile = kQkvTilesPerBlock * (kAhBlkG + qh0 + gr) + (gd >> 4);
        int const qv = 4 * t + ((gd >> 2) & 3);
        uint8_t const* base = ptr<uint8_t const>(ctx, kPQkvPart);
#pragma unroll
        for (int sl = 0; sl < kQkvSlices; ++sl)
        {
            int const g = sl * kQkvTiles + tile;
            int const cFirst = ahCtaOfChunk32(g * kSegChunks);
            int const cLast = ahCtaOfChunk32(g * kSegChunks + kSegChunks - 1);
#pragma unroll
            for (int k = 0; k < kQkvRunsPerSeg; ++k)
            {
                int const c = cFirst + k;
                if (c <= cLast)
                {
                    int const slot = g - ahRunStart32(c) / kSegChunks;
                    aw[sl * kQkvRunsPerSeg + k] = base
                        + (static_cast<size_t>(c) * kQkvMaxSlots + slot) * kQkvPartSlotBytes + 16 * qv + 4 * (gd & 3);
                }
            }
        }
    }
    ahPollVecsWords<kN, kQkvMaxRuns>(ctx, v, a, gw, aw, kCodeAhMerge);
    ah2Sub(ctx, 5, 128);                                               // merge inputs in (sub-stamp 5)
    ah2Sub(ctx, 7, 224);                                               // warp 7's poll done (sub-stamp 7)
    float* wts = reinterpret_cast<float*>(ctx.smem + kSmemRed);        // [33][2] weights, then den[2]
    float* red = reinterpret_cast<float*>(ctx.smem + kSmemRed + 2048); // [8][2][64]
    float* ss = reinterpret_cast<float*>(ctx.smem + kSmemRed + 6144);  // [4]
    if (q == 0)                                                        // warp 0's statements of the merge above
    {
        uint4 const s0 = v[kMl];
        uint4 const s1
            = lane == 0 ? v[kMl + 1] : make_uint4(__float_as_uint(-1.0e30f), 0u, __float_as_uint(-1.0e30f), 0u);
#pragma unroll
        for (int r = 0; r < 2; ++r)
        {
            float const m0 = __uint_as_float(r == 0 ? s0.x : s0.z);
            float const l0 = __uint_as_float(r == 0 ? s0.y : s0.w);
            float const m1 = __uint_as_float(r == 0 ? s1.x : s1.z);
            float const l1 = __uint_as_float(r == 0 ? s1.y : s1.w);
            float mx = fmaxf(l0 > 0.f ? m0 : -1.0e30f, l1 > 0.f ? m1 : -1.0e30f);
#pragma unroll
            for (int m = 16; m >= 1; m >>= 1)
            {
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, m));
            }
            float const w0 = l0 > 0.f ? ex2Approx(m0 - mx) : 0.f;
            float const w1 = l1 > 0.f ? ex2Approx(m1 - mx) : 0.f;
            float den = w0 * l0 + w1 * l1;
#pragma unroll
            for (int m = 16; m >= 1; m >>= 1)
            {
                den += __shfl_xor_sync(0xffffffffu, den, m);
            }
            wts[2 * lane + r] = w0;
            if (lane == 0)
            {
                wts[2 * 32 + r] = w1;
                wts[2 * 33 + r] = den;
            }
        }
    }
    float sig; // GS's statements for the one value (gr, gd)
    {
        float sum = 0.f;
#pragma unroll
        for (int k = 0; k < kQkvMaxRuns; ++k)
        {
            if (aw[k] != nullptr)
            {
                sum += __uint_as_float(gw[k]);
            }
        }
        float const xg = bf16Round(sum);
        sig = bf16Round(1.f / (1.f + expf(-xg)));
        if (MSO_MK_CKIO(io.qkv) != nullptr) // MSO_MK_LD_CK (the gate projection, as the gate fin stored it)
        {
            float const xgHi = __shfl_down_sync(0xffffffffu, xg, 1);
            if ((gd & 1) == 0)
            {
                st32(static_cast<uint8_t*>(io.qkv)
                        + (static_cast<size_t>(t) * kQkvRows + ahCkptCol(kAhBlkG + qh0 + gr) + gd) * 2,
                    packBf16(xg, xgHi));
            }
        }
    }
    asm volatile("bar.sync 1, 128;" ::: "memory");
    ah2Sub(ctx, -2, 128); // weights and gate in (kStampFa + 1)
    {
        // group q (thread tid - 128's part), splits q + 8 k
        float4 num = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int k = 0; k < kPart; ++k)
        {
            int const s = q + 8 * k;
            if (s < kAhBlocks)
            {
                float const w = wts[2 * s + rr];
                num.x += w * __uint_as_float(v[kOwn + k].x);
                num.y += w * __uint_as_float(v[kOwn + k].y);
                num.z += w * __uint_as_float(v[kOwn + k].z);
                num.w += w * __uint_as_float(v[kOwn + k].w);
            }
        }
        *reinterpret_cast<float4*>(red + (q * 2 + rr) * kAhD + 4 * dq) = num;
    }
    {
        // group q + 4 (this thread's part), splits q + 4 + 8 k
        float4 num = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int k = 0; k < kOwn; ++k)
        {
            float const w = wts[2 * (q + 4 + 8 * k) + rr];
            num.x += w * __uint_as_float(v[k].x);
            num.y += w * __uint_as_float(v[k].y);
            num.z += w * __uint_as_float(v[k].z);
            num.w += w * __uint_as_float(v[k].w);
        }
        *reinterpret_cast<float4*>(red + ((q + 4) * 2 + rr) * kAhD + 4 * dq) = num;
    }
    asm volatile("bar.sync 1, 128;" ::: "memory");
    ah2Sub(ctx, -3, 128); // split-group sums in shared memory (kStampFa + 2)
    {
        int const r = gr;
        int const d = gd;
        float n = 0.f;
#pragma unroll
        for (int k = 0; k < 8; ++k)
        {
            n += red[(k * 2 + r) * kAhD + d];
        }
        float const att = bf16Round(n / wts[2 * 33 + r]);
        int const qh = qh0 + r;
        uint8_t* attnOut = MSO_MK_CKPTR(uint8_t, ctx, kPAhAttnOut); // MSO_MK_LD_CK
        float sq = att * att;
#pragma unroll
        for (int m = 16; m >= 1; m >>= 1)
        {
            sq += __shfl_xor_sync(0xffffffffu, sq, m);
        }
        if (lane == 0)
        {
            ss[q] = sq;
        }
        // The two warps of each row (bar 1 over the 128 threads).
        asm volatile("bar.sync 1, 128;" ::: "memory");
        float const var = (ss[2 * r] + ss[2 * r + 1]) * (1.f / 64.f);
        float const normed = bf16Round(att * rsqrtf(var + f32Param(ctx, kPAhGateEps)));
        float const out = bf16Round(normed * sig);
        float const outHi = __shfl_down_sync(0xffffffffu, out, 1);
        float const attHi = __shfl_down_sync(0xffffffffu, att, 1);
        if ((d & 1) == 0)
        {
            int const reps = static_cast<int>(ctx.p[kPAhOReplicas]);
            uint32_t const word = lamport(packBf16(out, outHi));
#pragma unroll 1
            for (int rep = 0; rep < reps; ++rep)
            {
                st32(ctx.arena + kArAhO + rep * (kTok * kAhOCols * 2) + (t * kAhOCols + qh * kAhD + d) * 2, word);
            }
            if (attnOut != nullptr)
            {
                st32(attnOut + (t * kAhOCols + qh * kAhD + d) * 2, packBf16(att, attHi));
            }
        }
    }
    asm volatile("bar.sync 1, 128;" ::: "memory");
    ah2Sub(ctx, 6, 128); // gated output stored (sub-stamp 6)
    // The partials and gate words this thread read, emptied after the output (single reader).
#pragma unroll
    for (int k = 0; k < kN; ++k)
    {
        if (a[k] != nullptr)
        {
            st128(const_cast<uint8_t*>(a[k]), emptyVec());
        }
    }
#pragma unroll
    for (int k = 0; k < kQkvMaxRuns; ++k)
    {
        if (aw[k] != nullptr)
        {
            st32(const_cast<uint8_t*>(aw[k]), kEmpty);
        }
    }
}

// ---- end ME ----
#endif

__device__ __noinline__ void stageAhSwaAttn(Ctx const& ctx, int unit, uint32_t bar, Io const& io)
{
    int const tid = threadIdx.x;
    int const warp = tid >> 5;
    int const lane = tid & 31;
    int const h = unit / kAhBlocks;
    int const b = unit % kAhBlocks;
#if (MSO_MK_AH & 48) == 48
    // ---- WD-Q (MSO_MK_AH bit 32): a merging unit runs its merge's code once dry first, in the Q wait (the unit starts
    // right after the CTA's QKV run; Q needs the q / k / v fins, which start after the last run that feeds them) ----
    if (b < kAhBlocks - 1)
    {
        ahMerge(ctx, h, b, io, true);
    }
    // ---- end WD-Q ----
#endif
    AhMeta const* meta = ahMeta(ctx);
    AhWindow const w = ahWindow(meta->L, static_cast<int>(ctx.p[kPAhWindow]));
    int const j0 = w.base + kAhBlockKeys * b;
    float const dequant = 1.f / f32Param(ctx, kPAhKvScale);
    if (b >= w.nBlocks)
    {
        // An empty block: l = 0, which the merge weighs as zero.
        for (int i = tid; i < kAhRows * kAhD / 4; i += kThreads)
        {
            st128(ahPartO(ctx, h, b, 0) + 16 * i, make_uint4(0, 0, 0, 0));
        }
        if (tid < kAhRows)
        {
            st64(ahPartMl(ctx, h, b, tid), make_uint2(__float_as_uint(-1.0e30f), 0u));
        }
    }
    else
    {
        uint32_t const qs = ctx.smemBase + kAhScratchOff;
        uint32_t const ks = qs + kAhTileBytes;
        uint32_t const vs = ks + kAhTileBytes;
        uint32_t const land = ctx.smemBase + kAhEntryOff;
        bool const fresh = j0 + kAhBlockKeys > w.past;
        // ---- kIFlagW1SwaPreConvert: the pages' FP8 -> BF16 tiles before the Q poll (the pages are entry copies of
        // keys before the step); after Q only the step's own rows are converted, straight into the tiles ----
#if MSO_MK_LD_SW
        if (!(ctx.p[kPIFlags] & kIFlagW1SwaPreConvert) || ctx.p[kPAhQFp8] == 0)
        {
            __trap(); // MSO_MK_LD_SW: the build fixes kIFlagW1SwaPreConvert and kPAhQFp8 to the served SWA layers' 1
        }
        bool const pre = true;
#else
        bool const pre = (icPIFlags(ctx.p[kPIFlags]) & kIFlagW1SwaPreConvert) != 0;
#endif
        if (pre)
        {
            waitBar(ctx, bar, 0, kCodeAhPages);
            int const key = tid / 4;
            int const part = tid % 4;
            int const j = j0 + key;
            bool const live = j >= w.lo && j < w.L;
            uint32_t const off = (key < 32 ? key * kAhD : 2048 + (key - 32) * kAhD) + 16 * part;
#pragma unroll
            for (int kv = 0; kv < 2; ++kv)
            {
                uint4 const raw = lds128(land + kv * 4096 + off);
                uint4 lo4 = make_uint4(0, 0, 0, 0), hi4 = make_uint4(0, 0, 0, 0);
                if (live)
                {
                    lo4 = make_uint4(e4m3x2ToBf16x2(raw.x), e4m3x2ToBf16x2(raw.x >> 16), e4m3x2ToBf16x2(raw.y),
                        e4m3x2ToBf16x2(raw.y >> 16));
                    hi4 = make_uint4(e4m3x2ToBf16x2(raw.z), e4m3x2ToBf16x2(raw.z >> 16), e4m3x2ToBf16x2(raw.w),
                        e4m3x2ToBf16x2(raw.w >> 16));
                }
                uint32_t const t0 = (kv == 0 ? ks : vs) + key * kAhRowStride + part * 32;
                sts128(t0, lo4);
                sts128(t0 + 16, hi4);
            }
            __syncthreads(); // the step's rows below overwrite rows other threads converted here
        }
        // ---- end kIFlagW1SwaPreConvert ----
        // Q (8 KiB of BF16 rows of this KV head) and, for the step's blocks, its 8 K / V rows.
        uint4 v[3];
        uint8_t const* a[3] = {};
        a[0] = ctx.arena + kArAhQ + h * kAhRows * kAhD * 2 + 16 * tid;
        a[1] = a[0] + 16 * kThreads;
        if (fresh && tid < 64)
        {
            a[2] = ctx.arena + (tid < 32 ? kArAhKNew : kArAhVNew) + h * kTok * kAhD + 16 * (tid & 31);
        }
        pollVecs<3>(ctx, v, a, kCodeAhQ);
        stampInput(ctx);
#pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            int const vec = tid + kThreads * i;
            sts128(qs + (vec / 8) * kAhRowStride + (vec % 8) * 16, v[i]);
        }
        if (!pre) // without the pre-conversion
        {
            waitBar(ctx, bar, 0, kCodeAhPages);
        }
        stampWeights(ctx);
        subStamp(ctx, 3); // Q and pages in
        if (a[2] != nullptr)
        {
            int const t = (tid & 31) / 4;
            int const part = tid & 3;
            int const key = w.past + t - j0;
            if (key >= 0 && key < kAhBlockKeys)
            {
                uint32_t const off = (key < 32 ? key * kAhD : 2048 + (key - 32) * kAhD) + 16 * part;
                if (pre) // the step's row straight into the BF16 tile (a step key is live: lo <= past <= j < L)
                {
                    uint4 const raw = v[2];
                    int const j = j0 + key;
                    bool const live = j >= w.lo && j < w.L;
                    uint4 lo4 = make_uint4(0, 0, 0, 0), hi4 = make_uint4(0, 0, 0, 0);
                    if (live)
                    {
                        lo4 = make_uint4(e4m3x2ToBf16x2(raw.x), e4m3x2ToBf16x2(raw.x >> 16), e4m3x2ToBf16x2(raw.y),
                            e4m3x2ToBf16x2(raw.y >> 16));
                        hi4 = make_uint4(e4m3x2ToBf16x2(raw.z), e4m3x2ToBf16x2(raw.z >> 16), e4m3x2ToBf16x2(raw.w),
                            e4m3x2ToBf16x2(raw.w >> 16));
                    }
                    uint32_t const t0 = (tid < 32 ? ks : vs) + key * kAhRowStride + part * 32;
                    sts128(t0, lo4);
                    sts128(t0 + 16, hi4);
                }
                else
                {
                    sts128(land + (tid < 32 ? 0 : 4096) + off, v[2]);
                }
            }
        }
        __syncthreads();
        // FP8 -> BF16 tiles, keys outside [lo, L) zeroed.
        if (!pre) // without the pre-conversion
        {
            int const key = tid / 4;
            int const part = tid % 4;
            int const j = j0 + key;
            bool const live = j >= w.lo && j < w.L;
            uint32_t const off = (key < 32 ? key * kAhD : 2048 + (key - 32) * kAhD) + 16 * part;
#pragma unroll
            for (int kv = 0; kv < 2; ++kv)
            {
                uint4 const raw = lds128(land + kv * 4096 + off);
                uint4 lo4 = make_uint4(0, 0, 0, 0), hi4 = make_uint4(0, 0, 0, 0);
                if (live)
                {
                    lo4 = make_uint4(e4m3x2ToBf16x2(raw.x), e4m3x2ToBf16x2(raw.x >> 16), e4m3x2ToBf16x2(raw.y),
                        e4m3x2ToBf16x2(raw.y >> 16));
                    hi4 = make_uint4(e4m3x2ToBf16x2(raw.z), e4m3x2ToBf16x2(raw.z >> 16), e4m3x2ToBf16x2(raw.w),
                        e4m3x2ToBf16x2(raw.w >> 16));
                }
                uint32_t const t0 = (kv == 0 ? ks : vs) + key * kAhRowStride + part * 32;
                sts128(t0, lo4);
                sts128(t0 + 16, hi4);
            }
        }
        if (!pre) // with the pre-conversion the barrier after the step's rows already ordered the tiles
        {
            __syncthreads();
        }
        int const mt = warp & 3;
        int const kh = warp >> 2;
        int const g = lane >> 2;
        int const c = lane & 3;
#if MSO_MK_LD_SW
        bool const qFp8 = true; // MSO_MK_LD_SW (checked above)
#else
        bool const qFp8 = ctx.p[kPAhQFp8] != 0;
#endif
        float const qkScale = 1.4426950408889634f / 8.f * dequant * (qFp8 ? dequant : 1.f);
        // S = Q K^T over the warp's 16 rows x 32 keys.
        uint32_t af[4][4];
#pragma unroll
        for (int ks4 = 0; ks4 < 4; ++ks4)
        {
            ldsm4(af[ks4], qs + (16 * mt + (lane & 15)) * kAhRowStride + 32 * ks4 + 16 * (lane >> 4));
        }
        float s[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
        {
            s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
            for (int k2 = 0; k2 < 2; ++k2)
            {
                uint32_t bf[4];
                ldsm4(bf, ks + (32 * kh + 8 * nt + (lane & 7)) * kAhRowStride + 64 * k2 + 16 * (lane >> 3));
                mmaBf16(s[nt], af[2 * k2][0], af[2 * k2][1], af[2 * k2][2], af[2 * k2][3], bf[0], bf[1]);
                mmaBf16(
                    s[nt], af[2 * k2 + 1][0], af[2 * k2 + 1][1], af[2 * k2 + 1][2], af[2 * k2 + 1][3], bf[2], bf[3]);
            }
        }
        // Scale, mask, base-2 softmax over the half (rows g -> token 2 mt, g + 8 -> token 2 mt + 1).
        int const window = static_cast<int>(ctx.p[kPAhWindow]);
        float mx[2] = {-1.0e30f, -1.0e30f};
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
        {
#pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                int const j = j0 + 32 * kh + 8 * nt + 2 * c + (e & 1);
                int const qPos = w.past + 2 * mt + (e >> 1);
                bool const keep = j >= w.lo && j <= qPos && j > qPos - window;
                float const x = keep ? s[nt][e] * qkScale : __uint_as_float(0xff800000u);
                s[nt][e] = x;
                mx[e >> 1] = fmaxf(mx[e >> 1], x);
            }
        }
#pragma unroll
        for (int r = 0; r < 2; ++r)
        {
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
        }
        float l[2] = {0.f, 0.f};
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
        {
#pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                float const p = ex2Approx(s[nt][e] - mx[e >> 1]);
                s[nt][e] = p;
                l[e >> 1] += p;
            }
        }
#pragma unroll
        for (int r = 0; r < 2; ++r)
        {
            l[r] += __shfl_xor_sync(0xffffffffu, l[r], 1);
            l[r] += __shfl_xor_sync(0xffffffffu, l[r], 2);
        }
        // O = P V (P rounded to BF16), 16 rows x 64 dims.
        float o[8][4];
#pragma unroll
        for (int dt = 0; dt < 8; ++dt)
        {
            o[dt][0] = o[dt][1] = o[dt][2] = o[dt][3] = 0.f;
        }
#pragma unroll
        for (int kk = 0; kk < 2; ++kk)
        {
            uint32_t const pa0 = packBf16(s[2 * kk][0], s[2 * kk][1]);
            uint32_t const pa1 = packBf16(s[2 * kk][2], s[2 * kk][3]);
            uint32_t const pa2 = packBf16(s[2 * kk + 1][0], s[2 * kk + 1][1]);
            uint32_t const pa3 = packBf16(s[2 * kk + 1][2], s[2 * kk + 1][3]);
#pragma unroll
            for (int dp = 0; dp < 4; ++dp)
            {
                uint32_t bv[4];
                ldsm4t(bv,
                    vs + (32 * kh + 16 * kk + (lane & 7) + 8 * ((lane >> 3) & 1)) * kAhRowStride
                        + 16 * (2 * dp + (lane >> 4)));
                mmaBf16(o[2 * dp], pa0, pa1, pa2, pa3, bv[0], bv[1]);
                mmaBf16(o[2 * dp + 1], pa0, pa1, pa2, pa3, bv[2], bv[3]);
            }
        }
        __syncthreads();
        ah2Sub(ctx, -4, 0); // P V done (kStampFa + 3)
        // The two key halves merged: half 1 stages (m, l, O) in the scratch, half 0 combines and publishes.
        // ---- SP (MSO_MK_AH bit 256): o1's rows padded to kAhD + 8 floats. A warp's float2 accesses cover rows
        // 16 mt + g (g = lane / 4, 8 rows) at the same columns; with 64-float rows the 8 rows fall on the same two
        // banks (8 wavefronts per access), with 72-float rows on four bank octets (2 wavefronts, the minimum for 256
        // B). ----
#if MSO_MK_AH & 256
        constexpr int kO1Stride = kAhD + 8;
#else
        constexpr int kO1Stride = kAhD;
#endif
        static_assert(4 * kAhRows * kO1Stride + 4 * kAhRows * 2 <= kAhScratchBytes, "o1 + ml1 in the scratch");
        float* o1 = reinterpret_cast<float*>(ctx.smem + kAhScratchOff); // [64 rows][kO1Stride]
        float* ml1 = reinterpret_cast<float*>(ctx.smem + kAhScratchOff + 4 * kAhRows * kO1Stride); // [64 rows][2]
        if (kh == 1)
        {
#pragma unroll
            for (int dt = 0; dt < 8; ++dt)
            {
                *reinterpret_cast<float2*>(o1 + (16 * mt + g) * kO1Stride + 8 * dt + 2 * c)
                    = make_float2(o[dt][0], o[dt][1]);
                *reinterpret_cast<float2*>(o1 + (16 * mt + g + 8) * kO1Stride + 8 * dt + 2 * c)
                    = make_float2(o[dt][2], o[dt][3]);
            }
            if (c == 0)
            {
                ml1[2 * (16 * mt + g)] = mx[0];
                ml1[2 * (16 * mt + g) + 1] = l[0];
                ml1[2 * (16 * mt + g + 8)] = mx[1];
                ml1[2 * (16 * mt + g + 8) + 1] = l[1];
            }
        }
        __syncthreads();
        if (kh == 0)
        {
#pragma unroll
            for (int r = 0; r < 2; ++r)
            {
                int const row = 16 * mt + g + 8 * r;
                float const m1 = ml1[2 * row];
                float const l1 = ml1[2 * row + 1];
                float const m = fmaxf(mx[r], m1);
                float const a0 = ex2Approx(mx[r] - m);
                float const a1 = ex2Approx(m1 - m);
                float const lt = a0 * l[r] + a1 * l1;
                uint8_t* dst = ahPartO(ctx, h, b, row);
#pragma unroll
                for (int dt = 0; dt < 8; ++dt)
                {
                    float2 const p1 = *reinterpret_cast<float2 const*>(o1 + row * kO1Stride + 8 * dt + 2 * c);
                    float const x0 = (a0 * o[dt][2 * r] + a1 * p1.x) * dequant;
                    float const x1 = (a0 * o[dt][2 * r + 1] + a1 * p1.y) * dequant;
                    st64(dst + (8 * dt + 2 * c) * 4,
                        make_uint2(lamport(__float_as_uint(x0)), lamport(__float_as_uint(x1))));
                }
                if (c == 0)
                {
                    st64(ahPartMl(ctx, h, b, row),
                        make_uint2(lamport(__float_as_uint(m)), lamport(__float_as_uint(lt))));
                }
            }
        }
    }
    subStamp(ctx, 4); // partial published
#if MSO_MK_AH & 512
    // ---- ME: half 1 skips the combine above and merges here while half 0 combines and publishes; half 0 then waits
    // for it, so the unit ends together. The branch tests a REDUX result (a uniform register: the same flags in every
    // thread of a warp, the half being warp >> 2), so ptxas sees a warp-convergent call, and nothing is live across it
    // (a call under a thread-dependent branch is compiled for a split warp and moves the kernel-wide uniform-register
    // allocation). ----
    unsigned const meU = __reduce_or_sync(0xffffffffu, (tid >= 128 ? 1u : 0u) | (b < kAhBlocks - 1 ? 2u : 0u));
    if (meU & 2u)
    {
#if MSO_MK_LY & 8 // MSO_MK_LY bit 8: ME's attribution twin, the merge entered after half 0's publication
        __syncthreads();
#endif
        if (meU & 1u)
        {
            ahMerge(ctx, h, b, io);
        }
        __syncthreads();
    }
    // ---- end ME ----
#else
    if (b < kAhBlocks - 1)
    {
#if !(MSO_MK_AH & 1024) // MX: no pre-merge barrier: half 1 (no combine) enters first, the hand-off barrier joins half 0
        __syncthreads();
#endif
#if MSO_MK_AH & 16
        ahMerge(ctx, h, b, io, false);
#else
        ahMerge(ctx, h, b, io);
#endif
    }
#endif
}

// ---------------------------------------------------------------------------------------------------------------------
// AhOproj: unit = t0 | nt << 10 (output tiles [t0, t0 + nt), 16 rows each)
// ---------------------------------------------------------------------------------------------------------------------
__device__ __noinline__ void stageAhOproj(Ctx const& ctx, int unit, uint32_t wOff, uint32_t bar, Io const& io)
{
    int const tid = threadIdx.x;
    int const warp = tid >> 5;
    int const lane = tid & 31;
    int const t0 = unit & 1023;
    int const nt = unit >> 10;
    int const reps = static_cast<int>(ctx.p[kPAhOReplicas]);
#if MSO_MK_LD_SC
    // MSO_MK_LD_SC: the served replica counts (SWA 4, full 1) are powers of two: the replica index is a mask
    if (reps <= 0 || (reps & (reps - 1)) != 0)
    {
        __trap();
    }
    uint8_t const* in = ctx.arena + kArAhO + (blockIdx.x & static_cast<unsigned>(reps - 1)) * (kTok * kAhOCols * 2);
#else
    uint8_t const* in = ctx.arena + kArAhO + (blockIdx.x % reps) * (kTok * kAhOCols * 2);
#endif
    bool const canary = (ctx.p[kPAhFlags] & kAhFlagOprojCanary) != 0;
    // ---- kIFlagW1OprojAfterQ: a unit whose weights runRole did not request after the QKV run (they would land inside
    // the q / k / v hand-off window) requests them here, once one word of attention Q (QkvFin's output) is published.
    // The unit's barriers are initialised in the prologue; its weight wait below orders their use. ----
    if (icPIFlags(ctx.p[kPIFlags]) & kIFlagW1OprojAfterQ)
    {
        int const* roles = reinterpret_cast<int const*>(ctx.smem + kSmemSmall + kSmallRoles);
        int const j = ctx.stampRole;
        int const at = roles[2 * kMaxRoles + j];
        if (at >= 0 && (roles[at] >> 16) == kStQkvRun && warp == 0)
        {
            int const qh = static_cast<int>(blockIdx.x) % kAhQHeads;
            poll1(ctx,
                lane == 0 ? ctx.arena + kArAhQ + ((qh / kAhGroup) * kAhRows + qh % kAhGroup) * kAhD * 2 : nullptr,
                kCodeAhOIn);
            if (lane == 0)
            {
                issueUnitWeights(ctx, j, bar);
            }
        }
    }
    // ---- end kIFlagW1OprojAfterQ ----
    // ---- kIFlagW1WarmOproj: the canary's first load decides; while it reads empty the CTA runs a warm pass ----
    bool warm = false;
    bool canaryFull = false;
    if (canary && (icPIFlags(ctx.p[kPIFlags]) & kIFlagW1WarmOproj))
    {
        uint32_t empty = 0u;
        if (tid == 0)
        {
            int const c = static_cast<int>(blockIdx.x) % (kAhKvHeads * (kAhBlocks - 1));
            int const row = 2 * (c % (kAhBlocks - 1)) + 1;
            int const qh = (c / (kAhBlocks - 1)) * kAhGroup + row % kAhGroup;
            empty = ldPoll32(in + ((row / kAhGroup) * kAhOCols + qh * kAhD + kAhD - 2) * 2) == kEmpty ? 1u : 0u;
        }
        warm = __syncthreads_or(empty) != 0;
        canaryFull = !warm;
    }
#pragma unroll 1
    for (int pass = warm ? 0 : 1; pass < 2; ++pass)
    {
        bool const dry = pass == 0;
        // ---- end kIFlagW1WarmOproj setup ----
        if (canary && !dry
            && !canaryFull) // not in the warm pass; no second canary poll once the first load read it full
        {
            // Canary: one word of merge unit (h, b) = (c / 32, c % 32), c = blockIdx.x % 64 (its row 2 b + 1, dims
            // 62-63), so a waiting CTA spins on one line instead of re-polling its 16 KiB copy; the full poll below
            // still checks every word it reads.
            if (warp == 0)
            {
                int const c = static_cast<int>(blockIdx.x) % (kAhKvHeads * (kAhBlocks - 1));
                int const row = 2 * (c % (kAhBlocks - 1)) + 1;
                int const qh = (c / (kAhBlocks - 1)) * kAhGroup + row % kAhGroup;
                poll1(ctx, lane == 0 ? in + ((row / kAhGroup) * kAhOCols + qh * kAhD + kAhD - 2) * 2 : nullptr,
                    kCodeAhOIn);
            }
            __syncthreads();
        }
        uint4 v[4];
        uint8_t const* a[4];
#pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            uint8_t const* src = in + ((lane >> 2) * kAhOCols + (2 * warp + i) * kChunkCols + 16 * (lane & 3)) * 2;
            a[2 * i] = dry ? nullptr : src; // the warm pass loads nothing
            a[2 * i + 1] = dry ? nullptr : src + 16;
        }
        // ---- kIFlagT3WaitFirstOproj: the weights before the input poll ----
        bool const t3Wf = (icPIFlags(ctx.p[kPIFlags]) & kIFlagT3WaitFirstOproj) != 0;
        if (t3Wf && !dry) // not in the warm pass
        {
            waitPieces(ctx, bar, kCodeWeights);
        }
#if MSO_MK_PB & 8 // MSO_MK_PB bit 8: one 256-bit load per fragment
        static_assert(kArAhO % 32 == 0 && kArenaBytes % 32 == 0 && (kTok * kAhOCols * 2) % 32 == 0,
            "MSO_MK_PB bit 8: 32-byte pairs");
        pollVecs256<2>(ctx, v, a, kCodeAhOIn);
#else
        pollVecs<4>(ctx, v, a, kCodeAhOIn);
#endif
        if (!dry) // not in the warm pass
        {
            stampInput(ctx);
        }
        if (!t3Wf && !dry) // not in the warm pass
        {
            waitPieces(ctx, bar, kCodeWeights);
        }
        // ---- end kIFlagT3WaitFirstOproj ----
        // Up to kAhOMaxTiles tiles (a 4-tile unit on a CTA without an attention unit, schedule_attn oproj_off_merge)
        float acc[kAhOMaxTiles][2][4] = {};
#pragma unroll
        for (int i = 0; i < kAhOMaxTiles; ++i)
        {
            if (i < nt)
            {
#pragma unroll
                for (int k = 0; k < 2; ++k)
                {
                    uint4 const bfr[2] = {v[2 * k], v[2 * k + 1]};
                    mmaChunkBf16(acc[i][k],
                        ctx.smemBase + wOff + static_cast<uint32_t>(i * kAhOChunks + 2 * warp + k) * kBf16Block, bfr);
                }
            }
        }
        // Staging [warp][tile][lane][4] FP32 (16 KiB for 4 tiles); each output thread then sums its 8 rows over the
        // warps itself (warp 0 .. 7 from 0.f), so no second staging area or barrier is needed.
        float* red = reinterpret_cast<float*>(ctx.smem + kSmemRed);
        static_assert(kWarps * kAhOMaxTiles * 32 * 4 * 4 <= kSmemRedBytes, "o_proj staging");
#pragma unroll
        for (int i = 0; i < kAhOMaxTiles; ++i)
        {
            if (i < nt)
            {
                *reinterpret_cast<float4*>(red + ((warp * kAhOMaxTiles + i) * 32 + lane) * 4)
                    = make_float4(acc[i][0][0] + acc[i][1][0], acc[i][0][1] + acc[i][1][1], acc[i][0][2] + acc[i][1][2],
                        acc[i][0][3] + acc[i][1][3]);
            }
        }
        __syncthreads();
        if (tid < nt * 16)
        {
            int const i = tid / 16;
            int const tok = (tid % 16) / 2;
            int const hf = tid % 2;
            float x[8];
#pragma unroll
            for (int k = 0; k < 8; ++k)
            {
                // Row 8 hf + k of tile i, token tok: lane (row & 7) * 4 + tok / 2, element (row >= 8 ? 2 : 0) + (tok &
                // 1).
                int const l = k * 4 + tok / 2;
                int const r = 2 * hf + (tok & 1);
                float s = 0.f;
#pragma unroll
                for (int ww = 0; ww < kWarps; ++ww)
                {
                    s += red[((ww * kAhOMaxTiles + i) * 32 + l) * 4 + r];
                }
                x[k] = s;
            }
            uint4 const w4
                = make_uint4(packBf16(x[0], x[1]), packBf16(x[2], x[3]), packBf16(x[4], x[5]), packBf16(x[6], x[7]));
            if (!dry) // the warm pass stores nothing
            {
                // ---- MSO_MK_LANES bit 8: into the MoE hop workspace's hop1 area ----
#if MSO_MK_LANES & 8
                if (icPIFlags(ctx.p[kPIFlags]) & kIFlagOprojToMh)
                {
                    mhPublish(ctx, i1Hop1Off(ctx, tok, 16 * (t0 + i) + 8 * hf), lamport4(w4));
                }
                else
#endif
                    // ---- end MoE hop hand-off ----
                    st128(
                        ptr<uint8_t>(ctx, kPAhSlot) + (static_cast<size_t>(tok) * kHidden + 16 * (t0 + i) + 8 * hf) * 2,
                        w4);
            }
        }
        // ---- warm pass: the real pass rewrites the staging area ----
        if (dry)
        {
            __syncthreads();
        }
        // ---- end warm pass ----
    } // warm / real pass
}
