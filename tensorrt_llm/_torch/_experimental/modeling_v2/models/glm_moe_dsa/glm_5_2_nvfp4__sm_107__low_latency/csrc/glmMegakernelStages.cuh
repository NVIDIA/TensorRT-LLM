// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Device stages included inside the glm52 namespace by glmMegakernel.cu.

// ----------------------------------------------------------------------------------------------------------------
// Stages
// ----------------------------------------------------------------------------------------------------------------

// kFlagXPrime (fused-A's warp 0): 1 / rms of token (lane % 8) from the previous launch's four H2 partials of it
// (kArXSsq), v their first load; one exit test for the whole warp.
__device__ __forceinline__ float xRcp(Ctx const& ctx, uint4 v, int code)
{
    float const* p = arena<float const>(ctx, kArXSsq) + 4 * (threadIdx.x & 7);
    uint64_t t0 = 0;
    while (!warpAll(full4(v)))
    {
        spinCheck(ctx, t0, code, 0, 0);
        if (!full4(v))
        {
            v = ldPoll128(p);
        }
    }
    float const S
        = (((0.f + __uint_as_float(v.x)) + __uint_as_float(v.y)) + __uint_as_float(v.z)) + __uint_as_float(v.w);
    return rsqrtf(S / static_cast<float>(kHidden) + kEps);
}

// fused-A: rows [16 tile, +16) of [q_a | c_kv | k_pe] = x W_A^T, K = 6144.
__device__ __noinline__ void stageFusedA(Ctx const& ctx, int tile, uint32_t wOff, uint32_t bar, __nv_bfloat16 const* x)
{
    int const warp = threadIdx.x >> 5;
    if constexpr ((kSpecializedStaticFlags & kFlagLandProbe) != 0)
    {
        if (GLM52_STAMPS(ctx) != nullptr && threadIdx.x == 0)
        {
            while (!mbarTryWait(bar, 0))
            {
            }
            subStamp(ctx, 129);
        }
    }
    constexpr int kPer = kFaChunks / kWarps;
    uint4 b[kPer][2];
    if (GLM52_FLAGS(ctx) & kFlagEarlyX)
    {
        // Before the dependency wait the previous launch may still be running: its clear units empty this arena copy's
        // x (left non-empty from four launches back) before its H2 writes the new one, and the rows this unit stores
        // (kArY, kArIdxKw), and nothing else orders those clears before this read and those stores. So first its clear
        // count, as its H2 waits for it before publishing (its copy's counters were zeroed before it began and are not
        // zeroed again until this launch is done); then the poll.
        if (threadIdx.x == 0)
        {
            uint32_t const* c = reinterpret_cast<uint32_t const*>(ctx.p[kPCounters])
                + ((ctx.parity + ctx.numCopies - 1) % ctx.numCopies) * kNumCounters + kCClear;
            uint64_t t0 = 0;
            uint32_t v;
            while ((v = ldAcquire(c)) < static_cast<uint32_t>(kCountedClear))
            {
                spinCheck(ctx, t0, 99, kCClear, static_cast<int>(v));
            }
        }
        __syncthreads();
        if constexpr ((kSpecializedStaticFlags & kFlagXFrag) != 0)
        {
            pollBFrag<kPer>(ctx, b, arena<uint8_t const>(ctx, kArX), warp * kPer, 100);
        }
        else
        {
            pollB<kPer>(ctx, b, arena<uint8_t const>(ctx, kArX), kHidden, warp * kPer, 100);
        }
    }
    else
    {
#pragma unroll
        for (int i = 0; i < kPer; ++i)
        {
            loadB(b[i], x, kHidden, warp * kPer + i);
        }
    }
    // kFlagXPrime: the arena x is x' = res' x gamma; warp 0 takes the tokens' 1 / rms (lane l: token l % 8) from the
    // previous launch's H2 partials, first load now (they follow their x' by ~0.1 us), the rest after its MMAs.
    bool const xPrime = (kSpecializedStaticFlags & kFlagXPrime) != 0 && (GLM52_FLAGS(ctx) & kFlagEarlyX) && warp == 0;
    uint4 xSsq = make_uint4(kEmpty, kEmpty, kEmpty, kEmpty);
    if (xPrime)
    {
        xSsq = ldPoll128(arena<float const>(ctx, kArXSsq) + 4 * (threadIdx.x & 7));
    }
    float acc[1][4];
    bool const idxqEarly = ctx.leader && (GLM52_FLAGS(ctx) & kFlagIdxQEarly) && threadIdx.x == 0;
    stampInput(ctx);
    waitBar(ctx, bar, 0, 101);
    if (idxqEarly)
    {
        prefetchDeferred(ctx, reinterpret_cast<int const*>(ctx.smem + kSmemSmall + kSmallRoles));
    }
    // Four interleaved accumulator chains (one chain of 48 dependent MMAs was the unit's critical path).
    float part[4][4] = {};
#pragma unroll
    for (int i = 0; i < kPer; ++i)
    {
        mmaChunkBf16(part[i & 3], ctx.smemBase + wOff + (warp * kPer + i) * kBf16Block, b[i]);
    }
#pragma unroll
    for (int r = 0; r < 4; ++r)
    {
        acc[0][r] = (part[0][r] + part[1][r]) + (part[2][r] + part[3][r]);
    }
    float* sRcp = reinterpret_cast<float*>(ctx.smem + kSmemRed + 8704);
    if (xPrime)
    {
        float const rcp = xRcp(ctx, xSsq, 148);
        if (threadIdx.x < kTok)
        {
            sRcp[threadIdx.x] = rcp;
        }
    }
    float* out = reinterpret_cast<float*>(ctx.smem + kSmemRed + 8192);
    reduceWarps<1>(ctx, acc, out);
    float const* scale
        = (kSpecializedStaticFlags & kFlagXPrime) != 0 && (GLM52_FLAGS(ctx) & kFlagEarlyX) ? sRcp : nullptr;
    if (idxqEarly)
    {
        // Every warp's MMAs read the weight slot before the reduction's barrier: the IdxQ unit it blocks gets it now.
        issueBlockedEarly(ctx, 0, kStIdxQ);
    }
    // Tiles 164-173 (leader layers): the indexer's K and head-weight rows.
    if (tile < kFaRows / 16)
    {
        storeTileBf16(out, arena<uint8_t>(ctx, kArY), kFaRows, 16 * tile, scale);
    }
    else
    {
        storeTileBf16(out, arena<uint8_t>(ctx, kArIdxKw), kIdxKwCols, 16 * tile - kFaRows, scale);
    }
}

// q_b': rows [48 u, +48) of Q = rstd_q * (q_a W'^T), RoPE on the rope columns, E4M3.
__device__ __noinline__ void stageQb(Ctx const& ctx, int unit, uint32_t wOff, uint32_t bar, int const* kvLenPtr)
{
    int const warp = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
    if constexpr ((kSpecializedStaticFlags & kFlagLandProbe) != 0)
    {
        if (GLM52_STAMPS(ctx) != nullptr && threadIdx.x == 0)
        {
            while (!mbarTryWait(bar, 0))
            {
            }
            subStamp(ctx, 129);
        }
    }
    float* sc = reinterpret_cast<float*>(ctx.smem + kSmemSmall + kSmallScalars);
    // The RoPE factors of this thread's epilogue rows first: their loads overlap the q_a poll.
    float4 rope = make_float4(1.f, 0.f, 1.f, 0.f);
    {
        int const col = (48 * unit + 4 * (threadIdx.x % 12)) % 576;
        if (threadIdx.x < 12 * kTok && col >= 512)
        {
            int const pos = ldPinnedS32(kvLenPtr) - kTok + static_cast<int>(threadIdx.x) / 12;
            rope = ldPinnedF4(ptr<float const>(ctx, kPRope) + static_cast<size_t>(pos) * 128 + (col - 512));
        }
    }
    constexpr int kPer = kQbChunks / kWarps;
    uint4 b[kPer][2];
    pollB<kPer>(ctx, b, arena<uint8_t const>(ctx, kArY), kFaRows, warp * kPer, 102);
    stampInput(ctx);
    subStamp(ctx, 0);
    // Sum of squares of q_a per row: thread (g, c) holds 16 of row g's elements per chunk.
    float ssq = 0.f;
#pragma unroll
    for (int i = 0; i < kPer; ++i)
    {
#pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            uint32_t const w4[4] = {b[i][h].x, b[i][h].y, b[i][h].z, b[i][h].w};
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                float const lo = bf16Lo(w4[j]);
                float const hi = bf16Hi(w4[j]);
                ssq = ssq + lo * lo + hi * hi;
            }
        }
    }
    ssq += __shfl_xor_sync(0xffffffffu, ssq, 1);
    ssq += __shfl_xor_sync(0xffffffffu, ssq, 2);
    if ((lane & 3) == 0)
    {
        sc[warp * kTok + (lane >> 2)] = ssq;
    }
    waitBar(ctx, bar, 0, 103);
    subStamp(ctx, 1);
    float acc[kQbUnitTiles][4];
#pragma unroll
    for (int t = 0; t < kQbUnitTiles; ++t)
    {
        acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
    }
#pragma unroll
    for (int i = 0; i < kPer; ++i)
    {
#pragma unroll
        for (int t = 0; t < kQbUnitTiles; ++t)
        {
            mmaChunkBf16(acc[t], ctx.smemBase + wOff + (t * kQbChunks + warp * kPer + i) * kBf16Block, b[i]);
        }
    }
    subStamp(ctx, 2);
    float* out = reinterpret_cast<float*>(ctx.smem + kSmemRed + 12288);
    reduceWarps<kQbUnitTiles>(ctx, acc, out);
    subStamp(ctx, 3);
    if (threadIdx.x < kTok)
    {
        float s = 0.f;
        for (int w = 0; w < kWarps; ++w)
        {
            s += sc[w * kTok + threadIdx.x];
        }
        sc[64 + threadIdx.x] = rsqrtf(s / 2048.f + kEps);
    }
    __syncthreads();
    // Epilogue: thread (token, 4 rows): 8 x 12 = 96 threads, one word of 4 E4M3 values each.
    if (threadIdx.x < 12 * kTok)
    {
        int const tok = threadIdx.x / 12;
        int const q4 = threadIdx.x % 12;
        int const row0 = 48 * unit + 4 * q4;
        float const r = sc[64 + tok];
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            v[j] = out[(4 * q4 + j) * kTok + tok] * r;
        }
        if (row0 % 576 >= 512)
        {
            float const cs[4] = {rope.x, rope.y, rope.z, rope.w};
#pragma unroll
            for (int pp = 0; pp < 2; ++pp)
            {
                float const cosv = cs[2 * pp];
                float const sinv = cs[2 * pp + 1];
                float const n0 = cosv * v[2 * pp] - sinv * v[2 * pp + 1];
                float const n1 = cosv * v[2 * pp + 1] + sinv * v[2 * pp];
                v[2 * pp] = n0;
                v[2 * pp + 1] = n1;
            }
        }
        uint32_t const w = static_cast<uint32_t>(toE4m3(v[0])) | (static_cast<uint32_t>(toE4m3(v[1])) << 8)
            | (static_cast<uint32_t>(toE4m3(v[2])) << 16) | (static_cast<uint32_t>(toE4m3(v[3])) << 24);
        st32(arena<uint8_t>(ctx, kArQ8) + tok * kQbRows + row0, lamport(w));
    }
    subStamp(ctx, 4);
}

// KV append: this step's 8 KV rows (kv_a RMSNorm, k_pe RoPE, E4M3) into the pool and into the arena (this step's
// attention reads them from there); warp t = row t.
__device__ __noinline__ void stageKvAppend(Ctx const& ctx, int const* kvLenPtr, int const* blockTable)
{
    int const warp = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
    int const tok = warp;
    uint8_t const* y = arena<uint8_t const>(ctx, kArY) + (tok * kFaRows + 2048) * 2;
    // Step constants first (their loads overlap the poll of y): position, page, RoPE factors, the lane's gammas.
    int const L = __ldg(kvLenPtr);
    int const pos = L - kTok + tok;
    int const page = __ldg(blockTable + (pos >> 6));
    float4 cs0 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 cs1 = cs0;
    if (lane < 8)
    {
        float4 const* cs
            = reinterpret_cast<float4 const*>(ptr<float const>(ctx, kPRope) + static_cast<size_t>(pos) * 128)
            + 2 * lane;
        cs0 = __ldg(cs);
        cs1 = __ldg(cs + 1);
    }
    __nv_bfloat16 const* gamma = ptr<__nv_bfloat16 const>(ctx, kPGammaKv);
    uint4 const gm0 = __ldg(reinterpret_cast<uint4 const*>(gamma + 8 * lane));
    uint4 const gm1 = __ldg(reinterpret_cast<uint4 const*>(gamma + 8 * (lane + 32)));
    // Lane l: latent vectors l and l + 32 (8 values each), then the rope vector l (lanes < 8; others re-read l).
    uint4 v[3];
    {
        uint8_t const* srcs[3] = {y + 16 * lane, y + 16 * (lane + 32), y + 1024 + 16 * (lane & 7)};
#pragma unroll
        for (int i = 0; i < 3; ++i)
        {
            v[i] = ldPoll128(srcs[i]);
        }
        uint64_t t0 = 0;
        while (!warpAll(full4(v[0]) & full4(v[1]) & full4(v[2])))
        {
            spinCheck(ctx, t0, 104, tok, 0);
#pragma unroll
            for (int i = 0; i < 3; ++i)
            {
                if (!full4(v[i]))
                {
                    v[i] = ldPoll128(srcs[i]);
                }
            }
        }
    }
    stampInput(ctx);
    float lat[16];
    {
        uint32_t const w[8] = {v[0].x, v[0].y, v[0].z, v[0].w, v[1].x, v[1].y, v[1].z, v[1].w};
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            lat[2 * j] = bf16Lo(w[j]);
            lat[2 * j + 1] = bf16Hi(w[j]);
        }
    }
    float ssq = 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j)
    {
        ssq += lat[j] * lat[j];
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
    {
        ssq += __shfl_xor_sync(0xffffffffu, ssq, off);
    }
    float const rstd = rsqrtf(ssq / 512.f + kEps);
    int64_t const row = static_cast<int64_t>(page) * ctx.p[kPStrideRows] + (pos & 63) + ctx.p[kPLayerRows];
    uint8_t* dst = ptr<uint8_t>(ctx, kPKvBase) + row * kRowBytes;
    uint8_t* dstA = arena<uint8_t>(ctx, kArKv) + tok * kRowBytes;
    float gms[16];
    {
        uint32_t const gw[8] = {gm0.x, gm0.y, gm0.z, gm0.w, gm1.x, gm1.y, gm1.z, gm1.w};
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            gms[2 * j] = bf16Lo(gw[j]);
            gms[2 * j + 1] = bf16Hi(gw[j]);
        }
    }
    uint32_t out[4];
#pragma unroll
    for (int q = 0; q < 4; ++q)
    {
        uint32_t w = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e)
        {
            int const j = 4 * q + e;
            w |= static_cast<uint32_t>(toE4m3(lat[j] * (rstd * (gms[j] * 1.f)))) << (8 * e);
        }
        out[q] = lamport(w);
    }
    st64(dst + 8 * lane, make_uint2(out[0], out[1]));
    st64(dst + 8 * (lane + 32), make_uint2(out[2], out[3]));
    st64(dstA + 8 * lane, make_uint2(out[0], out[1]));
    st64(dstA + 8 * (lane + 32), make_uint2(out[2], out[3]));
    if (lane < 8)
    {
        float k[8];
        uint32_t const w[4] = {v[2].x, v[2].y, v[2].z, v[2].w};
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            k[2 * j] = bf16Lo(w[j]);
            k[2 * j + 1] = bf16Hi(w[j]);
        }
        float const cs[8] = {cs0.x, cs0.y, cs0.z, cs0.w, cs1.x, cs1.y, cs1.z, cs1.w};
        uint32_t ro[2] = {0, 0};
#pragma unroll
        for (int pp = 0; pp < 4; ++pp)
        {
            float const cosv = cs[2 * pp];
            float const sinv = cs[2 * pp + 1];
            uint32_t const r0 = toE4m3((cosv * k[2 * pp] - sinv * k[2 * pp + 1]) * 1.f);
            uint32_t const r1 = toE4m3((cosv * k[2 * pp + 1] + sinv * k[2 * pp]) * 1.f);
            ro[pp >> 1] |= (r0 | (r1 << 8)) << (16 * (pp & 1));
        }
        uint2 const rw = make_uint2(lamport(ro[0]), lamport(ro[1]));
        st64(dst + 512 + 8 * lane, rw);
        st64(dstA + 512 + 8 * lane, rw);
    }
}

// ----------------------------------------------------------------------------------------------------------------
// Leader layers: the DSA indexer (numerics as the stock fused prologue / Q tail: FP32 math on BF16-rounded
// intermediates, GPT-J RoPE on dims 0-63, UE8M0 E4M3 scales with the MUFU reciprocal)
// ----------------------------------------------------------------------------------------------------------------
// This step's 8 indexer keys (warp t = row t, lane l = dims 4l .. 4l + 3): LayerNorm (eps 1e-6), RoPE, E4M3 with the
// row's scale; into the cache at the row's position and into the arena (the logits of the page holding it read it
// there: the page may have been read before this store).
__device__ __noinline__ void stageIdxK(Ctx const& ctx, int const* kvLenPtr)
{
    int const t = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
    int const pos = __ldg(kvLenPtr) - kTok + t;
    // Constants first (their loads overlap the poll): LayerNorm weight and bias, cos / sin of pairs 2l, 2l + 1.
    float4 const lw = __ldg(reinterpret_cast<float4 const*>(ptr<float const>(ctx, kPIdxKnW)) + lane);
    float4 const lb = __ldg(reinterpret_cast<float4 const*>(ptr<float const>(ctx, kPIdxKnB)) + lane);
    float2 cs = make_float2(1.f, 1.f);
    float2 sn = make_float2(0.f, 0.f);
    if (lane < 16)
    {
        float const* row = ptr<float const>(ctx, kPIdxRope) + static_cast<size_t>(pos) * 64;
        cs = __ldg(reinterpret_cast<float2 const*>(row) + lane);
        sn = __ldg(reinterpret_cast<float2 const*>(row + 32) + lane);
    }
    int const blk = __ldg(ptr<int const>(ctx, kPIdxBlocks) + (pos >> 6));
    uint8_t const* src = arena<uint8_t const>(ctx, kArIdxKw) + (t * kIdxKwCols + 4 * lane) * 2;
    uint2 v = ldPoll64(src);
    {
        uint64_t t0 = 0;
        while (!__all_sync(0xffffffffu, full2(v)))
        {
            spinCheck(ctx, t0, 150, t, 0);
            if (!full2(v))
            {
                v = ldPoll64(src);
            }
        }
    }
    stampInput(ctx);
    float const x[4] = {bf16Lo(v.x), bf16Hi(v.x), bf16Lo(v.y), bf16Hi(v.y)};
    float sum = (x[0] + x[1]) + (x[2] + x[3]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
    {
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
    }
    float const mean = sum / static_cast<float>(kIdxDim);
    float kc[4];
    float var = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        kc[i] = x[i] - mean;
        var = fmaf(kc[i], kc[i], var);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
    {
        var += __shfl_xor_sync(0xffffffffu, var, off);
    }
    float const rstd = rsqrtf(var / static_cast<float>(kIdxDim) + 1e-6f);
    float const w[4] = {lw.x, lw.y, lw.z, lw.w};
    float const b[4] = {lb.x, lb.y, lb.z, lb.w};
    float k[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        k[i] = bf16Round(fmaf(kc[i] * rstd, w[i], b[i]));
    }
    if (lane < 16)
    {
        // GPT-J pairs (4l, 4l + 1), (4l + 2, 4l + 3): x cos -/+ partner sin, as fma(x, cos, rn(+-partner sin)).
        float const r0 = bf16Round(fmaf(k[0], cs.x, __fmul_rn(-k[1], sn.x)));
        float const r1 = bf16Round(fmaf(k[1], cs.x, __fmul_rn(k[0], sn.x)));
        float const r2 = bf16Round(fmaf(k[2], cs.y, __fmul_rn(-k[3], sn.y)));
        float const r3 = bf16Round(fmaf(k[3], cs.y, __fmul_rn(k[2], sn.y)));
        k[0] = r0;
        k[1] = r1;
        k[2] = r2;
        k[3] = r3;
    }
    float amax = fmaxf(fmaxf(fabsf(k[0]), fabsf(k[1])), fmaxf(fabsf(k[2]), fabsf(k[3])));
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
    {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    }
    float const scale = ue8m0Scale(amax);
    float const inv = rcpApproxFtz(scale);
    uint32_t const word = static_cast<uint32_t>(toE4m3(__fmul_rn(k[0], inv)))
        | (static_cast<uint32_t>(toE4m3(__fmul_rn(k[1], inv))) << 8)
        | (static_cast<uint32_t>(toE4m3(__fmul_rn(k[2], inv))) << 16)
        | (static_cast<uint32_t>(toE4m3(__fmul_rn(k[3], inv))) << 24);
    uint8_t* page = ptr<uint8_t>(ctx, kPIdxPool) + static_cast<size_t>(blk) * ctx.p[kPIdxBlockStride];
    st32(page + (pos & 63) * kIdxDim + 4 * lane, word);
    st32(arena<uint8_t>(ctx, kArIdxKNew) + t * kIdxDim + 4 * lane, lamport(word));
    if (lane == 0)
    {
        st32(page + kIdxPageKeys * kIdxDim + (pos & 63) * 4, __float_as_uint(scale));
        st32(arena<uint8_t>(ctx, kArIdxKNewS) + 4 * t, lamport(__float_as_uint(scale)));
    }
}

__device__ __noinline__ void selEpilogue(
    uint32_t taddr, float const* sW, float* sLog, int kt, int hh, float ks, int tMin, float* dbg, int64_t dbgLd);

// kFlagPageBalance: each IdxQ CTA holds kIdxBalDeficit pages fewer than each CTA without an IdxQ unit [kIdxQUnits,
// kCtas): the first kBalFree * kIdxBalDeficit pages round-robin over the latter, the rest round-robin over all CTAs.
// The CTAs without an IdxQ unit start IdxSel, and so their page loads, ~3.5 us earlier; with 7 pages more their logits
// end with the IdxQ CTAs' at 102k-900k keys (mk-t15 stamps: the earlier split, 14 pages on them and then 16 on each
// IdxQ CTA, left them 1.4 us late at 102k and the IdxQ CTAs 3.5 us late at 335k). Else CTA c holds the pages c, c +
// 212, ...
constexpr bool kPageBal = (kSpecializedStaticFlags & kFlagPageBalance) != 0;
constexpr int kBalFree = kCtas - kIdxQUnits;
static_assert(
    kIdxBalDeficit + (kIdxMaxKeys / kIdxPageKeys - kBalFree * kIdxBalDeficit + kCtas - 1) / kCtas <= kIdxMaxPagesRs
        && (kIdxMaxKeys / kIdxPageKeys + kCtas - 1) / kCtas <= kIdxMaxPagesRs,
    "the indexer's longest context: every CTA's logits fit");

// kFlagPageMap2: the key pages water-filled over four CTA groups in the order of their IdxSel starts (CTAs 128-163 at
// ~1.7 us, 174-211 at ~5.1, 164-173 at ~5.7, the IdxQ CTAs 0-127 at ~6.3; mk-t09 stamps, leader-front.md P3): group g
// takes T - kMapS(g) pages per CTA for the largest T the pages fill, the rest go one each to the CTAs in group order.
// Pages are numbered group by group (CTA-interleaved within a group), then the rest.
constexpr bool kPageMap2 = (kSpecializedStaticFlags & kFlagPageMap2) != 0;

// Group g's CTA count, first CTA, first CTA in group order, and pages fewer than group 0's (its IdxSel start's lead
// over group 0 / ~0.24 us per page).
__host__ __device__ constexpr int kMapN(int g)
{
    return g == 0 ? 36 : g == 1 ? 38 : g == 2 ? 10 : 128;
}

__host__ __device__ constexpr int kMapC0(int g)
{
    return g == 0 ? 128 : g == 1 ? 174 : g == 2 ? 164 : 0;
}

__host__ __device__ constexpr int kMapOrd(int g)
{
    return g == 0 ? 0 : g == 1 ? 36 : g == 2 ? 74 : 84;
}

__host__ __device__ constexpr int kMapS(int g)
{
    return g == 0 ? 0 : g == 1 ? 2 : g == 2 ? 5 : 7;
}

static_assert(kMapN(0) + kMapN(1) + kMapN(2) + kMapN(3) == kCtas && kMapC0(3) == 0 && kMapN(3) == kIdxQUnits,
    "the page map's groups");
static_assert((kIdxMaxKeys / kIdxPageKeys + 38 * 2 + 10 * 5 + 128 * 7) / kCtas + 1 <= kIdxMaxPagesRs,
    "the page map at the indexer's longest context");

struct IdxPageMap
{
    int T; // group g holds max(0, T - kMapS(g)) pages per CTA
    int B; // pages in the group blocks
    int R; // the rest: one page each for the first R CTAs in group order
};

__device__ __forceinline__ int idxMapGroup(int cta)
{
    return cta < 128 ? 3 : cta < 164 ? 0 : cta < 174 ? 2 : 1;
}

__device__ __forceinline__ IdxPageMap idxPageMap(int nPages)
{
    int g = 0;
    int A = kMapN(0);
    int S = kMapN(0) * kMapS(0);
    int T = (nPages + S) / A;
    while (g < 3 && T > kMapS(g + 1))
    {
        ++g;
        A += kMapN(g);
        S += kMapN(g) * kMapS(g);
        T = (nPages + S) / A;
    }
    int b = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        b += kMapN(i) * max(0, T - kMapS(i));
    }
    return IdxPageMap{T, b, nPages - b};
}

// Group g's pages per CTA and its first page.
__device__ __forceinline__ int idxMapK(IdxPageMap const& m, int g)
{
    return max(0, m.T - kMapS(g));
}

__device__ __forceinline__ int idxMapBase(IdxPageMap const& m, int g)
{
    int b = 0;
#pragma unroll
    for (int i = 0; i < 3; ++i)
    {
        b += i < g ? kMapN(i) * max(0, m.T - kMapS(i)) : 0;
    }
    return b;
}

// The global page of the CTA's page j (kPageBal: a CTA without an IdxQ unit holds the pages (cta - kIdxQUnits) +
// kBalFree j for j < kIdxBalDeficit, then every CTA the pages kBalFree * kIdxBalDeficit + cta, + 212, ...).
__device__ __forceinline__ int idxPageOf(int cta, int j, int nPages)
{
    if constexpr (kPageMap2)
    {
        IdxPageMap const m = idxPageMap(nPages);
        int const g = idxMapGroup(cta);
        return j < idxMapK(m, g) ? idxMapBase(m, g) + j * kMapN(g) + (cta - kMapC0(g))
                                 : m.B + kMapOrd(g) + (cta - kMapC0(g));
    }
    if constexpr (kPageBal)
    {
        if (cta >= kIdxQUnits && j < kIdxBalDeficit)
        {
            return (cta - kIdxQUnits) + kBalFree * j;
        }
        return kBalFree * kIdxBalDeficit + cta + kCtas * (j - (cta >= kIdxQUnits ? kIdxBalDeficit : 0));
    }
    return cta + kCtas * j;
}

// The CTA's index of global page p, or -1 when another CTA holds it.
__device__ __forceinline__ int idxPageSlot(int cta, int p, int nPages)
{
    if constexpr (kPageMap2)
    {
        IdxPageMap const m = idxPageMap(nPages);
        int const g = idxMapGroup(cta);
        if (p >= m.B)
        {
            return p - m.B == kMapOrd(g) + (cta - kMapC0(g)) ? idxMapK(m, g) : -1;
        }
        int const q = p - idxMapBase(m, g);
        return q >= 0 && q < idxMapK(m, g) * kMapN(g) && q % kMapN(g) == cta - kMapC0(g) ? q / kMapN(g) : -1;
    }
    if constexpr (kPageBal)
    {
        int const F = kBalFree * kIdxBalDeficit;
        if (p < F)
        {
            return cta >= kIdxQUnits && p % kBalFree == cta - kIdxQUnits ? p / kBalFree : -1;
        }
        return (p - F) % kCtas == cta ? (cta >= kIdxQUnits ? kIdxBalDeficit : 0) + (p - F) / kCtas : -1;
    }
    return p >= cta && (p - cta) % kCtas == 0 ? (p - cta) / kCtas : -1;
}

// The CTA's IdxSel pages for L keys.
__device__ __forceinline__ int idxPagesOf(int cta, int nPages)
{
    if constexpr (kPageMap2)
    {
        IdxPageMap const m = idxPageMap(nPages);
        int const g = idxMapGroup(cta);
        return idxMapK(m, g) + (kMapOrd(g) + (cta - kMapC0(g)) < m.R ? 1 : 0);
    }
    if constexpr (kPageBal)
    {
        int const F = min(nPages, kBalFree * kIdxBalDeficit);
        int const X = nPages - kBalFree * kIdxBalDeficit; // the round-robin pages
        int const extra = cta < X ? (X - 1 - cta) / kCtas + 1 : 0;
        if (cta >= kIdxQUnits)
        {
            int const c = cta - kIdxQUnits;
            return (c < F ? (F - 1 - c) / kBalFree + 1 : 0) + extra;
        }
        return extra;
    }
    return cta < nPages ? (nPages - 1 - cta) / kCtas + 1 : 0;
}

__device__ __forceinline__ int idxSelMyPages(int L)
{
    return idxPagesOf(blockIdx.x, (L + kIdxPageKeys - 1) / kIdxPageKeys);
}

// The CTA's IdxSel key pages [first, first + count) (SWIZZLE_128B boxes of the pool's tensor map) and their scales into
// the IdxSel region at a0 (page first + j into slot slot0 + j), on bar (0: kBarIdxK): warp 0, lane j takes page
// first + j, its block index from sBlk (the CTA's table) or loaded (in parallel with the others'). Read once per
// leader: L2 evict-first, as the weights (with the default policy these ~19.5 MB and the KV rows pushed what the next
// layers reuse out of L2: -46 us per slice, mk-v75a).
__device__ __noinline__ void idxSelIssuePages(Ctx const& ctx, uint32_t a0, int first, int count, int nPages,
    int slot0 = 0, uint32_t bar = 0, int const* sBlk = nullptr)
{
    int const lane = threadIdx.x & 31;
    int const cta = blockIdx.x;
    if (threadIdx.x >= 32 || count <= 0)
    {
        return;
    }
    uint32_t const barK = bar != 0 ? bar : ctx.smemBase + kSmemSmall + kSmallBars + kBarIdxK * 8;
    int const blk = lane >= count ? 0
        : sBlk != nullptr         ? sBlk[first + lane]
                                  : __ldg(ptr<int const>(ctx, kPIdxBlocks) + idxPageOf(cta, first + lane, nPages));
    if (lane == 0)
    {
        fenceProxyAsyncSmem();
        mbarExpectTx(barK, static_cast<uint32_t>(count) * (8192u + 256u));
    }
    __syncwarp();
    if (lane < count)
    {
        int64_t const stride = ctx.p[kPIdxBlockStride];
        int const y = static_cast<int>(static_cast<int64_t>(blk) * (stride / 128));
        uint8_t const* scales
            = ptr<uint8_t const>(ctx, kPIdxPool) + static_cast<size_t>(blk) * stride + kIdxPageKeys * kIdxDim;
        uint64_t const policy = policyEvictFirst();
        tmaLoad2dHint(a0 + kSelK + (slot0 + lane) * 8192, ptr<void const>(ctx, kPIdxTmap), 0, y, barK, policy);
        bulkCopyHint(a0 + kSelKs + (slot0 + lane) * 256, scales, 256, barK, policy);
    }
}

// Leader: the context fits in the top-k (row t has L - 7 + t <= L keys), so the top-k is every valid key whatever the
// logits: no indexer Q, no key pages, logits or selection; the attention takes keys [0, valid) itself (the stock top-k
// as a set). The new keys still go into the indexer cache (IdxK).
__device__ __forceinline__ bool idxAllKeys(int L)
{
    return L <= kTopK;
}

// Indexer Q rows [32 u, +32) (a quarter of head u / 4): rstd(q_a) (q_a Wf^T) rounded to BF16 (the stock wq_b output;
// Wf = wq_b with the q_a norm weight folded in), RoPE on the head's dims 0-63 (GPT-J pairs are adjacent rows), BF16;
// then the head's E4M3 scale from its four quarters' |max| (the head's other units run alongside), the quarter in
// E4M3, and (quarter 0) the head weight.
__device__ __noinline__ void stageIdxQ(Ctx const& ctx, int unit, uint32_t wOff, uint32_t bar, int const* kvLenPtr)
{
    if (idxAllKeys(__ldg(kvLenPtr)))
    {
        // The unit's weights were requested: they land before a later unit takes the slot.
        waitBar(ctx, bar, 0, 152);
        return;
    }
    int const warp = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
    float* sc = reinterpret_cast<float*>(ctx.smem + kSmemSmall + kSmallScalars);
    // Epilogue thread (row t = tid / 16, row pair p = tid % 16): its cos / sin first (their loads overlap the poll).
    int const et = static_cast<int>(threadIdx.x) / 16;
    int const ep = static_cast<int>(threadIdx.x) % 16;
    int const row0 = kIdxQUnitRows * unit + 2 * ep;
    int const dim0 = row0 % kIdxDim;
    float cosv = 1.f;
    float sinv = 0.f;
    if (threadIdx.x < 16 * kTok && dim0 < 64)
    {
        float const* cs = ptr<float const>(ctx, kPIdxRope) + static_cast<size_t>(__ldg(kvLenPtr) - kTok + et) * 64;
        cosv = __ldg(cs + dim0 / 2);
        sinv = __ldg(cs + 32 + dim0 / 2);
    }
    constexpr int kPer = kIdxQChunks / kWarps;
    uint4 b[kPer][2];
    pollB<kPer>(ctx, b, arena<uint8_t const>(ctx, kArY), kFaRows, warp * kPer, 151);
    stampInput(ctx);
    // Quarter 0: the head's weight word now (fused-A wrote it with q_a), its round trip under this unit's work.
    int const head = kIdxQUnitRows * unit / kIdxDim;
    int const quarter = (kIdxQUnitRows * unit % kIdxDim) / kIdxQUnitRows;
    uint8_t const* const kwWord = arena<uint8_t const>(ctx, kArIdxKw) + (et * kIdxKwCols + 128 + (head & ~1)) * 2;
    uint32_t const kwEarly = threadIdx.x < 16 * kTok && quarter == 0 && ep == 0 ? ldPoll32(kwWord) : kEmpty;
    // Sum of squares of q_a per row: thread (g, c) holds 16 of row g's elements per chunk.
    float ssq = 0.f;
#pragma unroll
    for (int i = 0; i < kPer; ++i)
    {
#pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            uint32_t const w4[4] = {b[i][h].x, b[i][h].y, b[i][h].z, b[i][h].w};
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                float const lo = bf16Lo(w4[j]);
                float const hi = bf16Hi(w4[j]);
                ssq = ssq + lo * lo + hi * hi;
            }
        }
    }
    ssq += __shfl_xor_sync(0xffffffffu, ssq, 1);
    ssq += __shfl_xor_sync(0xffffffffu, ssq, 2);
    if ((lane & 3) == 0)
    {
        sc[warp * kTok + (lane >> 2)] = ssq;
    }
    // The leader's logits epilogue once on a dummy tile while this unit's weights land (it runs cold in the CTA's
    // IdxSel unit otherwise): TMEM columns no MMA has written, the results into the reduction area.
    tcFenceAfter();
#if GLM52_FLAGS_MODE != 4 && GLM52_FLAGS_MODE != 5
    selEpilogue(*reinterpret_cast<uint32_t const*>(ctx.smem + kSmemSmall + kSmallScalars + 4 * 120) + 256u
            + (static_cast<uint32_t>(32 * (warp & 3)) << 16) + 128u * static_cast<uint32_t>(warp >> 2),
        reinterpret_cast<float const*>(ctx.smem + kSmemRed + 4096), reinterpret_cast<float*>(ctx.smem + kSmemRed),
        32 * (warp & 3) + lane, warp >> 2, 0.f, 0, nullptr, 0);
#endif
    __syncthreads(); // finish dummy scratch accesses before real reduction writes
    waitBar(ctx, bar, 0, 152);
    float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
    for (int i = 0; i < kPer; ++i)
    {
#pragma unroll
        for (int tl = 0; tl < 2; ++tl)
        {
            mmaChunkBf16(acc[tl], ctx.smemBase + wOff + (tl * kIdxQChunks + warp * kPer + i) * kBf16Block, b[i]);
        }
    }
    float* out = reinterpret_cast<float*>(ctx.smem + kSmemRed + 12288);
    reduceWarps<2>(ctx, acc, out);
    subStamp(ctx, 69);
    if (threadIdx.x < kTok)
    {
        float s = 0.f;
        for (int w = 0; w < kWarps; ++w)
        {
            s += sc[w * kTok + threadIdx.x];
        }
        sc[64 + threadIdx.x] = rsqrtf(s / 2048.f + kEps);
    }
    __syncthreads();
    if (threadIdx.x < 16 * kTok)
    {
        float const r = sc[64 + et];
        float x0 = bf16Round(out[(2 * ep) * kTok + et] * r);
        float x1 = bf16Round(out[(2 * ep + 1) * kTok + et] * r);
        if (dim0 < 64)
        {
            float const y0 = bf16Round(fmaf(x0, cosv, __fmul_rn(-x1, sinv)));
            float const y1 = bf16Round(fmaf(x1, cosv, __fmul_rn(x0, sinv)));
            x0 = y0;
            x1 = y1;
        }
        if ((GLM52_FLAGS(ctx) & kFlagIdxAllRows) == 0)
        {
            // The row shard: BF16 Q and the head's raw weight; the IdxSel CTAs take the head's E4M3 scale from its 128
            // values (the four quarters' |max| exchange, one L2 round trip, is off this unit's path).
            st32(arena<uint8_t>(ctx, kArIdxQb) + ((et * kIdxHeads + head) * kIdxDim + dim0) * 2,
                lamport(packBf16(x0, x1)));
            if (quarter == 0 && ep == 0)
            {
                uint32_t const wr = kwEarly != kEmpty ? kwEarly : poll1(ctx, kwWord, 154);
                st32(arena<float>(ctx, kArIdxW) + et * kIdxHeads + head,
                    lamport(__float_as_uint((head & 1) ? bf16Hi(wr) : bf16Lo(wr))));
            }
            subStamp(ctx, 72);
            return;
        }
        // The quarter's |max| of row et: its 16 lanes (lanes 0-15 hold row 2w, 16-31 row 2w + 1).
        float m = fmaxf(fabsf(x0), fabsf(x1));
#pragma unroll
        for (int off = 8; off > 0; off >>= 1)
        {
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
        }
        float* amaxRow = arena<float>(ctx, kArIdxAmax) + (et * kIdxHeads + head) * 4;
        if (ep == 0)
        {
            st32(amaxRow + quarter, lamport(__float_as_uint(m)));
        }
        subStamp(ctx, 70);
        uint4 const am = poll4(ctx, amaxRow, 153);
        subStamp(ctx, 71);
        float const scale = ue8m0Scale(fmaxf(
            fmaxf(__uint_as_float(am.x), __uint_as_float(am.y)), fmaxf(__uint_as_float(am.z), __uint_as_float(am.w))));
        float const inv = rcpApproxFtz(scale);
        // Lane pairs (ep even, ep + 1) make the 32-bit word of dims dim0 .. dim0 + 3.
        uint32_t const two = toE4m3x2(__fmul_rn(x0, inv), __fmul_rn(x1, inv));
        uint32_t const next = __shfl_down_sync(0xffffffffu, two, 1);
        if ((ep & 1) == 0)
        {
            st32(arena<uint8_t>(ctx, kArIdxQ8) + (et * kIdxHeads + head) * kIdxDim + dim0, lamport(two | (next << 16)));
        }
        if (quarter == 0 && ep == 0)
        {
            uint32_t const wr = kwEarly != kEmpty ? kwEarly : poll1(ctx, kwWord, 154);
            float const wraw = (head & 1) ? bf16Hi(wr) : bf16Lo(wr);
            st32(arena<float>(ctx, kArIdxW) + et * kIdxHeads + head,
                lamport(__float_as_uint(__fmul_rn(__fmul_rn(wraw, scale), kIdxWeightScale))));
        }
    }
    subStamp(ctx, 72);
}

// Block-wide, over a 2048-bin histogram: the largest bin whose count from the top reaches need (1 <= need <= the
// total), the count strictly above it and its own count, to every thread (thread j sums bins [8 j, +8); sSel[2..5)
// and sSel[8..16) are scratch).
__device__ __noinline__ void blockCrossBin(
    uint32_t const* hist, uint32_t need, uint32_t* sSel, int& bin, uint32_t& above, uint32_t& inBin)
{
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    uint4 const v0 = reinterpret_cast<uint4 const*>(hist)[2 * tid];
    uint4 const v1 = reinterpret_cast<uint4 const*>(hist)[2 * tid + 1];
    uint32_t const c[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
    uint32_t s = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        s += c[i];
    }
    // Inclusive suffix sum over the warp's lanes [lane, 32) (the higher lanes hold the larger bins), then the warps'.
    uint32_t incl = s;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1)
    {
        uint32_t const v = __shfl_down_sync(0xffffffffu, incl, off);
        incl += lane + off < 32 ? v : 0u;
    }
    if (lane == 0)
    {
        sSel[8 + warp] = incl;
    }
    __syncthreads();
    uint32_t higher = 0;
#pragma unroll
    for (int w = 0; w < kWarps; ++w)
    {
        higher += w > warp ? sSel[8 + w] : 0u;
    }
    uint32_t const aboveMe = higher + incl - s;
    if (aboveMe < need && aboveMe + s >= need)
    {
        uint32_t acc = aboveMe;
#pragma unroll
        for (int i = 7; i >= 0; --i)
        {
            if (acc + c[i] >= need)
            {
                sSel[2] = static_cast<uint32_t>(8 * tid + i);
                sSel[3] = acc;
                sSel[4] = c[i];
                break;
            }
            acc += c[i];
        }
    }
    __syncthreads();
    bin = static_cast<int>(sSel[2]);
    above = sSel[3];
    inBin = sSel[4];
}

// Block-wide exclusive scan of two per-thread counts in thread order (scratch: 16 words at shared address scratch; one
// barrier inside).
__device__ __forceinline__ void blockScan2(uint32_t a, uint32_t b, uint32_t scratch, uint32_t& baseA, uint32_t& baseB)
{
    int const lane = threadIdx.x & 31;
    int const warp = threadIdx.x >> 5;
    uint32_t ia = a;
    uint32_t ib = b;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1)
    {
        uint32_t const va = __shfl_up_sync(0xffffffffu, ia, off);
        uint32_t const vb = __shfl_up_sync(0xffffffffu, ib, off);
        ia += lane >= off ? va : 0u;
        ib += lane >= off ? vb : 0u;
    }
    if (lane == 31)
    {
        sts32(scratch + 4 * warp, ia);
        sts32(scratch + 4 * (kWarps + warp), ib);
    }
    __syncthreads();
    uint32_t pa = 0;
    uint32_t pb = 0;
#pragma unroll
    for (int w = 0; w < kWarps; ++w)
    {
        uint32_t const sa = lds32(scratch + 4 * w);
        uint32_t const sb = lds32(scratch + 4 * (kWarps + w));
        pa += w < warp ? sa : 0u;
        pb += w < warp ? sb : 0u;
    }
    baseA = pa + ia - a;
    baseB = pb + ib - b;
}

// Check mode: 128 accumulator columns of the warp's TMEM lane quarter from taddr into dd (this lane's row).
__device__ __noinline__ void dumpTileD(uint32_t taddr, float* dd)
{
#pragma unroll 1
    for (int c = 0; c < 4; ++c)
    {
        uint32_t r[32];
        tmemLd32(taddr + 32u * static_cast<uint32_t>(c), r);
#pragma unroll
        for (int j = 0; j < 32; ++j)
        {
            dd[32 * c + j] = __uint_as_float(r[j]);
        }
    }
}

// IdxSel epilogue of one tile for this thread (key kt of the tile, rows 4 hh .. + 3, accumulator columns [taddr,
// +128)): logit(t) = ks * sum_h w(t, h) relu(D[t][h]) into sLog[t][kt], -inf for rows t < tMin (the key is past the
// row's last); dbg != nullptr: also into dbg[t * dbgLd] (check mode). Out of line: the same code warms up on a dummy
// tile.
__device__ __noinline__ void selEpilogue(
    uint32_t taddr, float const* sW, float* sLog, int kt, int hh, float ks, int tMin, float* dbg, int64_t dbgLd)
{
    uint32_t r[128];
    tmemLd128(taddr, r);
#pragma unroll
    for (int tt = 0; tt < 4; ++tt)
    {
        int const t = 4 * hh + tt;
        // Four independent chains over the heads (h mod 4), then their sum.
        float4 const* w4 = reinterpret_cast<float4 const*>(sW + 32 * t);
        float c0 = 0.f;
        float c1 = 0.f;
        float c2 = 0.f;
        float c3 = 0.f;
#pragma unroll
        for (int h4 = 0; h4 < 8; ++h4)
        {
            float4 const wv = w4[h4];
            c0 = fmaf(wv.x, fmaxf(__uint_as_float(r[32 * tt + 4 * h4]), 0.f), c0);
            c1 = fmaf(wv.y, fmaxf(__uint_as_float(r[32 * tt + 4 * h4 + 1]), 0.f), c1);
            c2 = fmaf(wv.z, fmaxf(__uint_as_float(r[32 * tt + 4 * h4 + 2]), 0.f), c2);
            c3 = fmaf(wv.w, fmaxf(__uint_as_float(r[32 * tt + 4 * h4 + 3]), 0.f), c3);
        }
        float const acc = (c0 + c1) + (c2 + c3);
        float const lgt = t >= tMin ? acc * ks : -INFINITY;
        sLog[t * 128 + kt] = lgt;
        if (dbg != nullptr)
        {
            dbg[t * dbgLd] = lgt;
        }
    }
}

// IdxSel epilogue of one tile under the row shard (warps 0-3; key kt of the tile, the rank's row: accumulator columns
// [taddr, +32) = its 32 heads): logit = ks * sum_h w(h) relu(D[h]) into sLog[kt], -inf when the key is past the row's
// last (valid false); dbg != nullptr: also into *dbg (check mode). Returns the logit.
__device__ __noinline__ float selEpilogue1(
    uint32_t taddr, float const* sW, float* sLog, int kt, float ks, bool valid, float* dbg)
{
    uint32_t r[32];
    tmemLd32(taddr, r);
    float4 const* w4 = reinterpret_cast<float4 const*>(sW);
    float c0 = 0.f;
    float c1 = 0.f;
    float c2 = 0.f;
    float c3 = 0.f;
#pragma unroll
    for (int h4 = 0; h4 < 8; ++h4)
    {
        float4 const wv = w4[h4];
        c0 = fmaf(wv.x, fmaxf(__uint_as_float(r[4 * h4]), 0.f), c0);
        c1 = fmaf(wv.y, fmaxf(__uint_as_float(r[4 * h4 + 1]), 0.f), c1);
        c2 = fmaf(wv.z, fmaxf(__uint_as_float(r[4 * h4 + 2]), 0.f), c2);
        c3 = fmaf(wv.w, fmaxf(__uint_as_float(r[4 * h4 + 3]), 0.f), c3);
    }
    float const acc = (c0 + c1) + (c2 + c3);
    float const lgt = valid ? acc * ks : -INFINITY;
    sLog[kt] = lgt;
    if (dbg != nullptr)
    {
        *dbg = lgt;
    }
    return lgt;
}

// IdxSel tile MMAs (one warp): D = K8 Q8^T over the 4 K steps of 32, committed on bar (with commit: one commit after
// several tiles covers them all).
__device__ __noinline__ void selTileMma(
    uint32_t d, uint32_t aK, uint32_t aQ, uint32_t bar, uint32_t idesc, bool commit = true)
{
    tcFenceAfter();
    if (electOne())
    {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
        {
            mmaF8(d, aK + 32 * ks, aQ + 32 * ks, ks > 0 ? 1u : 0u, idesc);
        }
        if (commit)
        {
            commit1(bar);
        }
    }
    __syncwarp();
}

#if GLM52_SEL_BATCH_MMA
// One row-shard batch, after pagesIn's proxy fence and CTA synchronization.
// All tiles use disjoint TMEM columns; keep each tile's original four-K-step chain.
// One elected lane issues every tile and the existing single completion commit.
__device__ __noinline__ void selBatchMma(uint32_t d, uint32_t aK, uint32_t aQ, uint32_t bar, int nt)
{
    tcFenceAfter();
    if (electOne())
    {
#pragma unroll 1
        for (int i = 0; i < nt; ++i)
        {
#pragma unroll
            for (int ks = 0; ks < 4; ++ks)
            {
                mmaF8(d + 32u * static_cast<uint32_t>(i), aK + 16384u * static_cast<uint32_t>(i) + 32 * ks,
                    aQ + 32 * ks, ks > 0 ? 1u : 0u, kIdescF8N32);
            }
        }
        if (nt > 0)
        {
            commit1(bar);
        }
    }
    __syncwarp();
}
#endif

// Leader top-k helpers, row t = warp t, lane l over the CTA's local keys l + 32 m. Logit (row t, local key k) of the
// CTA: tile k / 128's first 4 KiB, [t][k % 128], from shared address a0 + kSelK (t < 0, the row shard: the row's array
// at a0 + kSelLog), as an order key (0 = none).
__device__ __forceinline__ uint32_t selKeyAt(uint32_t a0, int t, int k)
{
    float const f = __uint_as_float(
        lds32(t < 0 ? a0 + kSelLog + 4 * k : a0 + kSelK + (k >> 7) * 16384 + (t * 128 + (k & 127)) * 4));
    return f == -INFINITY ? 0u : orderKey(f);
}

// Over the nBins bins of the shared histogram h (a multiple of kThreads; bin b counts keys whose digit is b): the bin
// holding the need-th largest key (counted from the top bin down) into scal[8], the keys in the bins above it into
// scal[9]. scal[0, 8): the warps' sums. Every thread calls it; the bins must sum to at least need.
__device__ __forceinline__ void idxBinFromTop(uint32_t const* h, int nBins, uint32_t need, uint32_t* scal)
{
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    int const per = nBins / kThreads;
    int const hi = nBins - per * tid; // this thread's bins: [hi - per, hi), the top chunk at thread 0
    uint32_t s = 0;
    for (int j = 0; j < per; ++j)
    {
        s += h[hi - 1 - j];
    }
    uint32_t incl = s;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1)
    {
        uint32_t const v = __shfl_up_sync(0xffffffffu, incl, off);
        incl += lane >= off ? v : 0u;
    }
    if (lane == 31)
    {
        scal[warp] = incl;
    }
    __syncthreads();
    uint32_t before = incl - s;
    for (int w = 0; w < warp; ++w)
    {
        before += scal[w];
    }
    if (before < need && before + s >= need)
    {
        for (int j = 0; j < per; ++j)
        {
            uint32_t const c = h[hi - 1 - j];
            if (before + c >= need)
            {
                scal[8] = static_cast<uint32_t>(hi - 1 - j);
                scal[9] = before;
                break;
            }
            before += c;
        }
    }
    __syncthreads();
}

// Every CTA, row t = warp t, lane l over the CTA's local keys l + 32 m (in shared memory). The 10th largest of
// the lanes' four largest keys (ten keys of the CTA are >= it, so it is at most the CTA's 10th-largest key; it is that
// key unless a lane holds five of the top ten) into the arena's word of (CTA, row). From the words, per row: T(t) =
// their minimum, leaving out up to 7 CTAs whose word is not positive (ReLU logits: a CTA can hold fewer than ten keys
// above 0): the other >= 205 CTAs' 2050 keys are >= T(t), so T(t) <= the row's 2048th largest; E(t) = the geometric
// mean of the positive words (the mean of their bits), an estimate of the 2048th without a certificate (within ~1 % on
// the c1 rows), when at least half the CTAs have one; and T2(t) = max(T(t), E(t) - kTightGap). The CTA's keys >= T2(t)
// go into row t's list A, its keys in [T(t), T2(t)) into list B: the selector takes list A alone when it holds need0
// keys (then the row's 2048 are among them), A and B otherwise. Keys equal to T2 (list A) and to T (list B) are taken
// kTieFree freely, more from the list's pool of kTopK (equal keys are interchangeable, and the pool covers any row's
// need). One 64-bit atomic per row takes the CTA's slots in both lists and counts the CTA in ((1 << 24) | A count, then
// B count and the CTA's keys in list A's histogram << 18), then the (key, position) pairs as single 64-bit stores (the
// selector polls them; key 0 = not written yet). List A's keys in [T2, T2 + kHistSpan) also go into the row's histogram
// (a relaxed add per key). sMin: shared scratch; T(t) and T2(t) left at sMin + 768 and + 896 (+ 4 t).
constexpr uint32_t kBndZero = 0x007fffffu;  // the word of a +0.0 bound (= kEmpty): orderKey(-inf), never a bound
constexpr int kIdxTCta = kIdxSelBase;       // the CTA that takes T from the words (its logits end among the first)
constexpr uint32_t kTieFree = 16;
constexpr int kUnionWords = kCtas * 64;     // per row: every CTA lane's two largest keys
constexpr int kUnionVecs = kUnionWords / 4; // as 16-byte vectors
constexpr int kUnionPer = (kUnionVecs + kThreads - 1) / kThreads; // a thread's vectors: tid + 256 u
// Per row after the union: every CTA's summary of its 64 words (two 16-byte vectors, idxUnionBound).
constexpr int kUnionSumWords = kCtas * 8;
static_assert((kUnionWords + kUnionSumWords) * 4 <= (kCandCap - kCandList) * 8, "the union's words past list A");

// Inserts key x into a lane's two largest (m1 >= m2; 0 = none). Equal keys are distinct entries.
__device__ __forceinline__ void top2Insert(uint32_t& m1, uint32_t& m2, uint32_t x)
{
    if (x > m1)
    {
        m2 = m1;
        m1 = x;
    }
    else if (x > m2)
    {
        m2 = x;
    }
}

// One 256-bin pass of the union's keys (kUnionWords words at shared address sU): keys k with k - lo <= lim (the window
// [lo, lo + lim], lim < 256 << sh; k < lo wraps past lim since lo >= 1 and lim <= 0xffffffff - lo; 0 is no key) to bin
// (k - lo) >> sh, the others to dummy bins, into a histogram with a copy per lane (bin b of lane l at word 32 b + l,
// the dummies past bin 255: a warp's adds never share a bank, and none branches: a conditional add compiled to a branch
// with a reconvergence barrier per key), then bin t's count in thread t and the bin c holding the need-th largest
// (count(>= its lower edge) >= need): c = (bins with that property) - 1, -1 when even bin 0 holds fewer. Returns c;
// cnt: bin c's count; abv: the keys in bins above c. sHist: (256 + kUnionDummies) x 32 words; sRed: 16 words. Not
// inlined and rolled: it runs once or twice per launch from a cold instruction cache (the unrolled register version
// took 6.2 us per pass in the kernel against 0.75 us warm alone, mk-u05a). Two vectors per iteration, both loaded
// before the adds, and the adds without a memory clobber: 0.61 us per 13,568 keys warm, against ~1.0 one vector at a
// time.
constexpr int kUnionDummies = 8;
// The first pass's key window below the union's largest key (log2; order keys: 2^23 per binade).
#ifndef GLM52_UNION_WIN
#define GLM52_UNION_WIN 28
#endif
__device__ __noinline__ int unionPass(uint32_t sU, uint32_t lo, int sh, uint32_t lim, uint32_t need, uint32_t sHist,
    uint32_t sRed, uint32_t& cnt, uint32_t& abv)
{
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
#pragma unroll 1
    for (int i = tid; i < 256 * 32 / 4; i += kThreads)
    {
        sts128(sHist + 16 * i, make_uint4(0u, 0u, 0u, 0u));
    }
    __syncthreads();
    uint32_t const laneOff = sHist + 4 * lane;
    auto add = [&](uint32_t k, uint32_t dummy)
    {
        uint32_t const d = k - lo;
        redsAddUnordered(laneOff + 128 * (d <= lim ? d >> sh : dummy), 1u);
    };
#pragma unroll 1
    for (int i = tid; i < kUnionVecs; i += 2 * kThreads)
    {
        bool const two = i + kThreads < kUnionVecs;
        uint4 const q = lds128(sU + 16 * i);
        uint4 r = lds128(sU + 16 * min(i + kThreads, kUnionVecs - 1));
        r = two ? r : make_uint4(0u, 0u, 0u, 0u);
        uint32_t const dummy = 256u + static_cast<uint32_t>(i & (kUnionDummies - 1));
        add(q.x, dummy);
        add(q.y, dummy);
        add(q.z, dummy);
        add(q.w, dummy);
        add(r.x, dummy ^ 1u);
        add(r.y, dummy ^ 1u);
        add(r.z, dummy ^ 1u);
        add(r.w, dummy ^ 1u);
    }
    __syncthreads();
    // Bin tid's 32 copies as 8 vectors, rotated by tid so a warp's loads spread over the banks.
    uint32_t s = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        uint4 const q = lds128(sHist + 16 * (8 * tid + ((j + tid) & 7)));
        s += (q.x + q.y) + (q.z + q.w);
    }
    // Suffix sums from bin 255 down: within the warp, then the warps above (their 8 totals in two loads).
    uint32_t suf = s;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1)
    {
        uint32_t const x = __shfl_down_sync(0xffffffffu, suf, off);
        suf += lane + off < 32 ? x : 0u;
    }
    if (lane == 0)
    {
        sts32(sRed + 4 * warp, suf);
    }
    __syncthreads();
    {
        uint4 const a = lds128(sRed);
        uint4 const b = lds128(sRed + 16);
        uint32_t const tw[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
        for (int w = 1; w < kWarps; ++w)
        {
            suf += w > warp ? tw[w] : 0u;
        }
    }
    int const c = __syncthreads_count(suf >= need) - 1;
    if (tid == c)
    {
        sts32(sRed + 4 * 8, s);
        sts32(sRed + 4 * 9, suf - s);
    }
    __syncthreads();
    cnt = c >= 0 ? lds32(sRed + 4 * 8) : 0u;
    abv = c >= 0 ? lds32(sRed + 4 * 9) : 0u;
    return c;
}

// unionPass's lim: the window of 256 bins of 2^sh from lo, up to cap.
__device__ __forceinline__ uint32_t unionLim(uint32_t lo, int sh, uint32_t cap)
{
    uint64_t const end = static_cast<uint64_t>(lo) + (256ull << sh) - 1ull;
    return static_cast<uint32_t>((end < cap ? end : static_cast<uint64_t>(cap)) - lo);
}

// The union's count and range (its nonzero keys; kFlagUnionSide: the positive keys, order key >= 0x80000000, their
// smallest, and the largest negative key), from the CTAs' summaries of their 64 words (idxUnionBound).
struct UnionStats
{
    uint32_t tot;
    uint32_t lo;
    uint32_t hi;
    uint32_t np;
    uint32_t minPos;
    uint32_t maxNeg;
};

// The need0-th largest of the union's keys (kUnionWords words at shared address sU, 0 = none; st: their count and
// range), bounded from below: two 256-bin passes (unionPass): over the top 2^GLM52_UNION_WIN keys of the range (the
// row's need0-th largest lies far closer to the largest key; the whole range when that holds too few), then over the
// cut bin. T = the last cut bin's lower edge: at least need0 keys are >= T, within max(1, 2^-16 of the range) of the
// need0-th largest. With fewer than need0 keys: 1 (every key of the row). hsh: list A's histogram bin width (log2) for
// ~4 of the union's keys per bin at T's density (14 when the first cut bin held < 16). sHist: (256 + kUnionDummies) x
// 32 words; sRed: 16 words. Every thread calls it.
__device__ __noinline__ uint32_t idxUnionKth(
    Ctx const& ctx, uint32_t sU, int need0, uint32_t sHist, uint32_t sRed, uint32_t& hsh, UnionStats st)
{
    constexpr bool kSide = (kSpecializedStaticFlags & kFlagUnionSide) != 0;
    hsh = 14u;
    if (st.tot < static_cast<uint32_t>(need0))
    {
        return 1u;
    }
    uint32_t const need = static_cast<uint32_t>(need0);
    // kFlagUnionSide: pass 1 over the side of zero that holds the need-th largest key: [minPos, hi] with need when
    // there are need positive keys, else [lo, maxNeg] with need - np (every positive key is above it; keys above
    // maxNeg go to the dummy bins in both passes).
    uint32_t top = st.hi;
    uint32_t bot = st.lo;
    uint32_t needS = need;
    uint32_t cap = 0xffffffffu;
    if constexpr (kSide)
    {
        if (st.np >= need)
        {
            bot = st.minPos;
        }
        else if (st.np > 0)
        {
            top = st.maxNeg;
            needS = need - st.np;
            cap = st.maxNeg;
        }
    }
    uint32_t base = top - bot > (1u << GLM52_UNION_WIN) ? top - (1u << GLM52_UNION_WIN) : bot;
    // (top - base) >> sh < 256. __clz's result is unsigned in this toolchain: without the cast 24 - __clz(.) wraps
    // below a range of 128 and max(0, .) compares unsigned, leaving sh negative.
    int sh = max(0, 24 - static_cast<int>(__clz(top - base)));
    uint32_t cnt;
    uint32_t abv;
    int b = unionPass(sU, base, sh, unionLim(base, sh, cap), needS, sHist, sRed, cnt, abv);
    if (b < 0)
    {
        // The top 2^26 keys of the range hold fewer than need0: the whole range (one pass, coarser).
        base = st.lo;
        sh = max(0, 24 - static_cast<int>(__clz(st.hi - st.lo)));
        needS = need;
        cap = 0xffffffffu;
        b = max(0, unionPass(sU, base, sh, unionLim(base, sh, cap), need, sHist, sRed, cnt, abv));
    }
    if (cnt >= 16u)
    {
        hsh = static_cast<uint32_t>(min(16, max(8, sh + 2 - (31 - static_cast<int>(__clz(cnt))))));
    }
    base += static_cast<uint32_t>(b) << sh;
    subStamp(ctx, 133);
    if (sh > 0)
    {
        // Pass 2 counts the cut bin [base, base + 2^sh) only: for sh < 8 its 256 unit bins reach past the bin into keys
        // already in abv.
        int const sh2 = max(0, sh - 8);
        uint32_t const binTop = base > 0xffffffffu - ((1u << sh) - 1u) ? 0xffffffffu : base + ((1u << sh) - 1u);
        int const b2
            = unionPass(sU, base, sh2, unionLim(base, sh2, min(cap, binTop)), needS - abv, sHist, sRed, cnt, abv);
        base += static_cast<uint32_t>(max(b2, 0)) << sh2;
    }
    subStamp(ctx, 134);
    return max(base, 1u);
}

// The leader's bound for the rows this CTA's warps handle (GLM52_TOPK_UNION). Every lane of every CTA puts the two
// largest of the keys it holds (64 per CTA and row) into the row's union, kUnionWords words past list A (key + 1: 0 =
// not written; the row's candidates use list A alone). CTA kIdxTCta polls a row's union as 16-byte vectors into
// registers, zeroes the words for the next launch of this copy and takes T, a lower bound of the union's need0-th
// largest key within one histogram bin (idxUnionKth): at least need0 of the row's keys, the union's, are >= T. T = T2
// (one list, no estimate) is published to every CTA. With two keys per lane, T sits within a few keys of the row's
// need0-th largest unless a lane holds three of the row's top keys (with one, T was 30-50 % below it at 32k-65k
// keys: 3.6-3.8k candidates, and list A's fine bins missed the cut). (The minimum of the CTAs' bounds and the estimate
// from the positive ones assumed positive logits spread over the CTAs: served rows are negative, and one CTA holds up
// to ~66 of a row's top 2048. The union's exact need0-th largest from three radix passes over warp-aggregated adds
// (__match_any_sync) took 17 us from the last CTA's words to T, one histogram pass 6.6 us: 723.6 -> 685.3 us per slice
// at 102k, mk-t08.)
__device__ __noinline__ void idxUnionBound(Ctx const& ctx, uint32_t a0, int nPer, int cta, int t, int tL, bool rs,
    bool active, uint64_t* cand, uint32_t sMin, uint32_t* bnd, int L)
{
    int const warp = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
    if (active)
    {
        uint32_t m1 = 0;
        uint32_t m2 = 0;
        if (rs && !(GLM52_FLAGS(ctx) & (kFlagIdxTilePairs | kFlagBoundLoop)))
        {
            // The row shard's tile epilogues kept them: the two largest of the 8 warps' pairs of this lane.
#pragma unroll
            for (int w = 0; w < kWarps; ++w)
            {
                top2Insert(m1, m2, lds32(sMin + kSelLaneMax + 4 * (32 * w + lane)));
                top2Insert(m1, m2, lds32(sMin + kSelLaneMax2 + 4 * (32 * w + lane)));
            }
        }
        else
        {
#pragma unroll 4
            for (int m = 0; m < nPer; ++m)
            {
                top2Insert(m1, m2, selKeyAt(a0, tL, lane + 32 * m));
            }
        }
        subStamp(ctx, 64);
        // key + 1: 0 = not written (orderKey(+FLT_MAX) + 1 does not wrap).
        uint32_t* const uni = reinterpret_cast<uint32_t*>(cand + static_cast<size_t>(t) * kCandCap + kCandList);
        st32(uni + 64 * cta + lane, m1 + 1u);
        st32(uni + 64 * cta + 32 + lane, m2 + 1u);
        // The CTA's summary of its 64 words (m1 >= m2, 0 = none) for the T CTA's count and range (a pass over the union
        // took ~1 us there): the largest and smallest key, the smallest positive and largest negative key, the
        // positive keys and all keys. Every word is nonzero once written (0xffffffff: none; the others + 1).
        bool const p1 = m1 >= 0x80000000u;
        bool const p2 = m2 >= 0x80000000u;
        uint32_t const hiC = __reduce_max_sync(0xffffffffu, m1);
        uint32_t const loC = __reduce_min_sync(0xffffffffu, m2 != 0u ? m2 : m1 != 0u ? m1 : 0xffffffffu);
        uint32_t const minPosC = __reduce_min_sync(0xffffffffu, p2 ? m2 : p1 ? m1 : 0xffffffffu);
        uint32_t const maxNegC = __reduce_max_sync(0xffffffffu, !p1 ? m1 : !p2 ? m2 : 0u);
        uint32_t const npC = __reduce_add_sync(0xffffffffu, (m1 >> 31) + (m2 >> 31));
        uint32_t const cC = __reduce_add_sync(0xffffffffu, (m1 != 0u ? 1u : 0u) + (m2 != 0u ? 1u : 0u));
        if (lane == 0)
        {
            uint32_t* const sum = uni + kUnionWords + 8 * cta;
            st128(sum,
                make_uint4(hiC + 1u, loC == 0xffffffffu ? loC : loC + 1u,
                    minPosC == 0xffffffffu ? minPosC : minPosC + 1u, maxNegC + 1u));
            st128(sum + 4, make_uint4(npC + 1u, cC + 1u, 1u, 1u));
        }
    }
    subStamp(ctx, 65);
    uint32_t* const tw = bnd + kCtas * kTok;
    // kFlagUnionLean: CTA 127 takes T while it holds fewer key pages than CTAs 128-211 (with kFlagPageBalance, always:
    // kIdxBalDeficit fewer): its logits end early, so its union poll starts before the last CTAs' stores (CTA 204, then
    // a 14-page CTA, was one of the last).
    constexpr bool kLean = (kSpecializedStaticFlags & kFlagUnionLean) != 0;
    int const nPagesT = (L + kIdxPageKeys - 1) / kIdxPageKeys;
    int const tCta
        = kLean && idxPagesOf(kIdxQUnits - 1, nPagesT) < idxPagesOf(kIdxQUnits, nPagesT) ? kIdxQUnits - 1 : kIdxTCta;
    if (cta == tCta)
    {
        // The histogram over Q8 and the key scales and its scratch after it (dead after the MMAs; the CTA's logits are
        // past them).
        uint32_t const sU = a0 + kSelK;
        uint32_t const sHist = a0 + kSelQ8;
        uint32_t const sRed = sHist + 4 * (256 + kUnionDummies) * 32;
        static_assert(kSelK + 4 * kUnionWords <= kSelQ8, "the union in the K region");
        static_assert(kSelQ8 + 4 * (256 + kUnionDummies) * 32 + 4 * 48 <= kSelLog, "the T CTA's histogram and scratch");
#pragma unroll 1
        for (int r = 0; r < kTok; ++r)
        {
            if (rs && r != t)
            {
                continue;
            }
            // The union's vectors into registers, all in flight at once; each round re-polls every vector of the
            // thread not complete yet (one round trip for them all: one vector at a time took a round trip each), and
            // the warp leaves the loop together (warpAll: a split warp takes the slow path of every warp collective
            // after it).
            uint32_t* const uni = reinterpret_cast<uint32_t*>(cand + static_cast<size_t>(r) * kCandCap + kCandList);
            // Thread c < kCtas: CTA c's summary too (vectors kUnionPer and kUnionPer + 1).
            uint32_t* const sum = uni + kUnionWords + 8 * threadIdx.x;
            bool const hasSum = threadIdx.x < kCtas;
            uint4 v[kUnionPer + 2];
#pragma unroll
            for (int u = 0; u < kUnionPer; ++u)
            {
                int const i = threadIdx.x + u * kThreads;
                v[u] = i < kUnionVecs ? ldPoll128(uni + 4 * i) : make_uint4(1u, 1u, 1u, 1u);
            }
            v[kUnionPer] = hasSum ? ldPoll128(sum) : make_uint4(1u, 1u, 1u, 1u);
            v[kUnionPer + 1] = hasSum ? ldPoll128(sum + 4) : make_uint4(1u, 1u, 1u, 1u);
            auto pending = [&]()
            {
                uint32_t p = 0;
#pragma unroll
                for (int u = 0; u < kUnionPer + 2; ++u)
                {
                    p |= (v[u].x == 0u) | (v[u].y == 0u) | (v[u].z == 0u) | (v[u].w == 0u) ? 1u << u : 0u;
                }
                return p;
            };
            uint32_t p = pending();
            uint64_t t0 = 0;
            while (!warpAll(p == 0u))
            {
                spinCheck(ctx, t0, 181, threadIdx.x, r);
#pragma unroll
                for (int u = 0; u < kUnionPer; ++u)
                {
                    if ((p >> u) & 1u)
                    {
                        v[u] = ldPoll128(uni + 4 * (threadIdx.x + u * kThreads));
                    }
                }
                if ((p >> kUnionPer) & 1u)
                {
                    v[kUnionPer] = ldPoll128(sum);
                }
                if ((p >> (kUnionPer + 1)) & 1u)
                {
                    v[kUnionPer + 1] = ldPoll128(sum + 4);
                }
                p = pending();
            }
            // Into shared memory at the start of the K region (dead after the MMAs: the CTA's logits are past kSelLog),
            // the slots back to zero for the next launch of this copy.
#pragma unroll
            for (int u = 0; u < kUnionPer; ++u)
            {
                int const i = threadIdx.x + u * kThreads;
                if (i < kUnionVecs)
                {
                    sts128(sU + 16 * i, make_uint4(v[u].x - 1u, v[u].y - 1u, v[u].z - 1u, v[u].w - 1u));
                    if constexpr (!kLean)
                    {
                        st128(uni + 4 * i, make_uint4(0u, 0u, 0u, 0u));
                    }
                }
            }
            if (!kLean && hasSum)
            {
                st128(sum, make_uint4(0u, 0u, 0u, 0u));
                st128(sum + 4, make_uint4(0u, 0u, 0u, 0u));
            }
            // The count and range: the summaries' (8 warps' partials through sRed, then every thread sums them).
            UnionStats st;
            {
                uint4 const a = v[kUnionPer];
                uint4 const b = v[kUnionPer + 1];
                uint32_t const hiS = hasSum ? a.x - 1u : 0u;
                uint32_t const loS = hasSum && a.y != 0xffffffffu ? a.y - 1u : 0xffffffffu;
                uint32_t const minPosS = hasSum && a.z != 0xffffffffu ? a.z - 1u : 0xffffffffu;
                uint32_t const maxNegS = hasSum ? a.w - 1u : 0u;
                uint32_t const npS = hasSum ? b.x - 1u : 0u;
                uint32_t const cS = hasSum ? b.y - 1u : 0u;
                uint32_t const warpV[6] = {__reduce_add_sync(0xffffffffu, cS), __reduce_min_sync(0xffffffffu, loS),
                    __reduce_max_sync(0xffffffffu, hiS), __reduce_add_sync(0xffffffffu, npS),
                    __reduce_min_sync(0xffffffffu, minPosS), __reduce_max_sync(0xffffffffu, maxNegS)};
                if (lane < 6)
                {
                    uint32_t x = warpV[0];
#pragma unroll
                    for (int j = 1; j < 6; ++j)
                    {
                        x = lane == j ? warpV[j] : x;
                    }
                    sts32(sRed + 4 * (8 * lane + warp), x);
                }
                __syncthreads();
                // Stat j's 8 warp values at sRed + 32 j.
                auto sum8 = [&](int j)
                {
                    uint4 const x = lds128(sRed + 32 * j);
                    uint4 const y = lds128(sRed + 32 * j + 16);
                    return ((x.x + x.y) + (x.z + x.w)) + ((y.x + y.y) + (y.z + y.w));
                };
                auto min8 = [&](int j)
                {
                    uint4 const x = lds128(sRed + 32 * j);
                    uint4 const y = lds128(sRed + 32 * j + 16);
                    return min(min(min(x.x, x.y), min(x.z, x.w)), min(min(y.x, y.y), min(y.z, y.w)));
                };
                auto max8 = [&](int j)
                {
                    uint4 const x = lds128(sRed + 32 * j);
                    uint4 const y = lds128(sRed + 32 * j + 16);
                    return max(max(max(x.x, x.y), max(x.z, x.w)), max(max(y.x, y.y), max(y.z, y.w)));
                };
                st = UnionStats{sum8(0), min8(1), max8(2), sum8(3), min8(4), max8(5)};
            }
            __syncthreads();
            subStamp(ctx, 132);
            // List A's 64 histogram bins from T get ~4 of the union's keys each at T's density (hsh): T sits within
            // a few keys of the row's need0-th largest, so the cut lands in the first bins and holds few keys (a fixed
            // 2^14 put up to ~60 keys in it on served rows, ranked one by one).
            uint32_t hsh;
            uint32_t const T = idxUnionKth(ctx, sU, min(kTopK, L - kTok + r + 1), sHist, sRed, hsh, st);
            if (threadIdx.x == 0)
            {
                uint32_t const enc = T == kEmpty ? kBndZero : T;
                stU64(tw + 2 * r, (static_cast<uint64_t>(hsh) << 32) | enc);
            }
            if constexpr (kLean)
            {
                // The slots back to zero for the next launch of this copy, after T's publish.
#pragma unroll
                for (int u = 0; u < kUnionPer; ++u)
                {
                    int const i = threadIdx.x + u * kThreads;
                    if (i < kUnionVecs)
                    {
                        st128(uni + 4 * i, make_uint4(0u, 0u, 0u, 0u));
                    }
                }
                if (hasSum)
                {
                    st128(sum, make_uint4(0u, 0u, 0u, 0u));
                    st128(sum + 4, make_uint4(0u, 0u, 0u, 0u));
                }
            }
            __syncthreads(); // the histogram again for the next row
        }
        subStamp(ctx, 77);
    }
    if (warp == 0)
    {
        bool const want = rs ? lane == t : lane < kTok;
        uint64_t v = want ? ldPollU64(tw + 2 * lane) : 0ull;
        uint64_t t0 = 0;
        while (!warpAll(!want || static_cast<uint32_t>(v) != kEmpty))
        {
            spinCheck(ctx, t0, 167, lane, 0);
            if (want && static_cast<uint32_t>(v) == kEmpty)
            {
                v = ldPollU64(tw + 2 * lane);
            }
        }
        if (want)
        {
            uint32_t const lo = static_cast<uint32_t>(v);
            uint32_t const T = lo == kBndZero ? kEmpty : lo;
            sts32(sMin + 768 + 4 * lane, T);
            sts32(sMin + 896 + 4 * lane, T);
            sts32(sMin + 800 + 4 * lane, 1u);                             // the selector's E: none
            sts32(sMin + 864 + 4 * lane, static_cast<uint32_t>(v >> 32)); // list A's histogram bin width (log2)
        }
    }
}

__device__ __noinline__ void idxBoundCands(
    Ctx const& ctx, uint32_t a0, int nTiles, int cta, uint32_t* gc, uint64_t* cand, uint32_t sMin, uint32_t* bnd, int L)
{
    int const warp = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
    // The row shard: warp 0 alone handles the rank's row (the tiles hold its logits as row 0).
    bool const rs = (GLM52_FLAGS(ctx) & kFlagIdxAllRows) == 0;
    int const t = rs ? ctx.rank >> 1 : warp; // the row (words, lists, counters)
    int const tL = rs ? -1 : warp;           // its logits' row in the tiles (-1: the row shard's array)
    bool const active = !rs || warp == 0;
    int const nPer = nTiles * 4;
    // The loops are rolled (by four, the keys read from shared memory each time): this code runs once per leader launch
    // from cold instruction caches, ~25-55 ns per 128-byte line, which made a fully unrolled 32-key version 3x slower.
#if GLM52_TOPK_UNION
    idxUnionBound(ctx, a0, nPer, cta, t, tL, rs, active, cand, sMin, bnd, L);
#else
    if (!(GLM52_FLAGS(ctx) & kFlagTopkTCta))
    {
        // Each lane's largest key; the warp's 32 maxima in descending order (bitonic): the 10th is a certified bound
        // (ten lanes hold a key >= it, so ten of the CTA's keys are), the 9th the estimate's sample (~ the CTA's
        // 10th-largest key). One 64-bit word per (row, CTA) (rows' words contiguous): 9th << 32 | 10th. Every CTA then
        // takes T, T2 and E of its row from the 212 words itself (one hop; the active warp, seven words per lane). (The
        // lanes' top four and ten warp rounds took ~1.5 us, and a T CTA a second hop.)
        if (active)
        {
            uint32_t m1 = 0;
            if (rs && !(GLM52_FLAGS(ctx) & (kFlagIdxTilePairs | kFlagBoundLoop)))
            {
                // The row shard's tile epilogues kept them: the maximum of the 8 warps' words of this lane.
#pragma unroll
                for (int w = 0; w < kWarps; ++w)
                {
                    m1 = max(m1, lds32(sMin + kSelLaneMax + 4 * (32 * w + lane)));
                }
            }
            else
            {
#pragma unroll 4
                for (int m = 0; m < nPer; ++m)
                {
                    m1 = max(m1, selKeyAt(a0, tL, lane + 32 * m));
                }
            }
            subStamp(ctx, 64);
#pragma unroll
            for (int k = 2; k <= 32; k <<= 1)
            {
#pragma unroll
                for (int j = k >> 1; j > 0; j >>= 1)
                {
                    uint32_t const o = __shfl_xor_sync(0xffffffffu, m1, j);
                    m1 = ((lane & k) == 0) == ((lane & j) == 0) ? max(m1, o) : min(m1, o);
                }
            }
            uint32_t const tenth = __shfl_sync(0xffffffffu, m1, 9);
            uint32_t const ninth = __shfl_sync(0xffffffffu, m1, 8);
            uint64_t* const rowW = reinterpret_cast<uint64_t*>(bnd) + t * kCtas;
            if (lane == 0)
            {
                stU64(rowW + cta, (static_cast<uint64_t>(ninth) << 32) | (tenth == kEmpty ? kBndZero : tenth));
            }
            subStamp(ctx, 65);
            uint64_t w[7];
#pragma unroll
            for (int i = 0; i < 7; ++i)
            {
                w[i] = lane + 32 * i < kCtas ? ldPollU64(rowW + lane + 32 * i) : 0ull;
            }
            uint64_t t0 = 0;
            while (true)
            {
                bool miss = false;
#pragma unroll
                for (int i = 0; i < 7; ++i)
                {
                    miss |= static_cast<uint32_t>(w[i]) == kEmpty;
                }
                if (warpAll(!miss))
                {
                    break;
                }
                spinCheck(ctx, t0, 166, lane, 0);
#pragma unroll
                for (int i = 0; i < 7; ++i)
                {
                    if (static_cast<uint32_t>(w[i]) == kEmpty)
                    {
                        w[i] = ldPollU64(rowW + lane + 32 * i);
                    }
                }
            }
            // T: the minimum of the 10ths, leaving out up to 7 CTAs whose 10th is not positive; E: the geometric mean
            // of the positive 9ths (the mean of their bits), when at least half the CTAs have one.
            uint32_t mnAll = ~0u;
            uint32_t mnPos = ~0u;
            uint32_t nNon = 0;
            uint32_t sumE = 0;
            uint32_t nPosE = 0;
#pragma unroll
            for (int i = 0; i < 7; ++i)
            {
                bool const valid = lane + 32 * i < kCtas;
                uint32_t const lw = static_cast<uint32_t>(w[i]);
                uint32_t const tv = lw == kBndZero ? kEmpty : lw;
                uint32_t const ev = static_cast<uint32_t>(w[i] >> 32);
                bool const posT = valid && tv > kEmpty;
                bool const posE = valid && ev > kEmpty;
                mnAll = valid ? min(mnAll, tv) : mnAll;
                mnPos = posT ? min(mnPos, tv) : mnPos;
                nNon += valid && !posT ? 1u : 0u;
                sumE += posE ? (ev - kEmpty) >> 8 : 0u;
                nPosE += posE ? 1u : 0u;
            }
            mnAll = __reduce_min_sync(0xffffffffu, mnAll);
            mnPos = __reduce_min_sync(0xffffffffu, mnPos);
            nNon = __reduce_add_sync(0xffffffffu, nNon);
            sumE = __reduce_add_sync(0xffffffffu, sumE);
            nPosE = __reduce_add_sync(0xffffffffu, nPosE);
            uint32_t const Tr = max(nNon <= 7 && mnPos != ~0u ? mnPos : mnAll, 1u);
            uint32_t const est = 2 * nPosE >= static_cast<uint32_t>(kCtas) && !(GLM52_FLAGS(ctx) & kFlagTopkOneList)
                ? kEmpty + ((sumE / nPosE) << 8)
                : 0u;
            uint32_t const T2r = est > Tr + kTightGap ? est - kTightGap : Tr;
            if (lane == 0)
            {
                sts32(sMin + 768 + 4 * t, Tr);
                sts32(sMin + 896 + 4 * t, T2r);
                sts32(sMin + 800 + 4 * t, est > Tr && est != kEmpty ? est : 1u); // the selector's E (1: none)
            }
        }
    }
    else
    {
        uint32_t m1 = 0;
        uint32_t m2 = 0;
        uint32_t m3 = 0;
        uint32_t m4 = 0;
#pragma unroll 4
        for (int m = 0; m < (active ? nPer : 0); ++m)
        {
            uint32_t const v = selKeyAt(a0, tL, lane + 32 * m);
            uint32_t const r1 = min(m1, v);
            uint32_t const r2 = min(m2, r1);
            m4 = max(m4, min(m3, r2));
            m3 = max(m3, r2);
            m2 = max(m2, r1);
            m1 = max(m1, v);
        }
        subStamp(ctx, 64);
        // Ten rounds of the warp maximum over the lanes' heads (a lane's next key follows its head).
        uint32_t tenth = 0;
#pragma unroll 1
        for (int round = 0; round < (active ? 10 : 0); ++round)
        {
            uint32_t const wmax = __reduce_max_sync(0xffffffffu, m1);
            uint32_t const who = __ballot_sync(0xffffffffu, m1 == wmax);
            if (lane == __ffs(who) - 1)
            {
                m1 = m2;
                m2 = m3;
                m3 = m4;
                m4 = 0;
            }
            tenth = wmax;
        }
        if (active && lane == 0)
        {
            st32(bnd + cta * kTok + t, tenth == kEmpty ? kBndZero : tenth);
        }
        subStamp(ctx, 65);
        // CTA kIdxTCta polls the CTAs' words (thread c < kCtas: CTA c's 8), takes T, T2 and E per row (each from the
        // warp minima, counts and sums of the words, then over the warps) and publishes them right after the words: per
        // row one 64-bit word T | T2 << 32, then the 8 E words (for the selectors). Every CTA's warp 0 polls the 8
        // rows' words (all warps polling the same lines, three words each, slowed the hand-off by ~0.8 us). One CTA
        // polling 212 x 8 words instead of all of them: every CTA taking T itself (one hop) came out even (mk-v73a).
        uint32_t* const tw = bnd + kCtas * kTok;
        if (cta == kIdxTCta)
        {
            uint32_t w[kTok];
            {
                int const c = threadIdx.x;
                uint4 b0 = make_uint4(~0u, ~0u, ~0u, ~0u);
                uint4 b1 = b0;
                if (rs)
                {
                    // Row t only: its word of every CTA, at its place among b0 / b1 (the other rows stay ~0u).
                    uint32_t v = c < kCtas ? ldPoll32(bnd + c * kTok + t) : ~0u;
                    uint64_t t0 = 0;
                    while (!warpAll(v != kEmpty))
                    {
                        spinCheck(ctx, t0, 166, c, 0);
                        v = c < kCtas && v == kEmpty ? ldPoll32(bnd + c * kTok + t) : v;
                    }
                    uint32_t* const bw = t < 4 ? &b0.x : &b1.x;
                    bw[t & 3] = v;
                }
                else
                {
                    if (c < kCtas)
                    {
                        b0 = ldPoll128(bnd + c * kTok);
                        b1 = ldPoll128(bnd + c * kTok + 4);
                    }
                    uint64_t t0 = 0;
                    while (!warpAll(full4(b0) && full4(b1)))
                    {
                        spinCheck(ctx, t0, 166, c, 0);
                        if (c < kCtas && !full4(b0))
                        {
                            b0 = ldPoll128(bnd + c * kTok);
                        }
                        if (c < kCtas && !full4(b1))
                        {
                            b1 = ldPoll128(bnd + c * kTok + 4);
                        }
                    }
                }
                w[0] = b0.x;
                w[1] = b0.y;
                w[2] = b0.z;
                w[3] = b0.w;
                w[4] = b1.x;
                w[5] = b1.y;
                w[6] = b1.z;
                w[7] = b1.w;
            }
            subStamp(ctx, 78);
            // Per (warp, row) at sMin + 1024 (the selector's scratch, not in use yet): the minimum of all words and of
            // the positive ones, the count of the others and the sum of the positive ones' bits >> 8 (< 2^31 over 212).
            uint32_t const sRed = sMin + 1024;
#pragma unroll
            for (int r = 0; r < kTok; ++r)
            {
                if (rs && r != t)
                {
                    continue;
                }
                uint32_t const v = w[r] == kBndZero ? kEmpty : w[r];
                bool const pos = v > kEmpty && v != ~0u;
                uint32_t const mnAll = __reduce_min_sync(0xffffffffu, v);
                uint32_t const mnPos = __reduce_min_sync(0xffffffffu, pos ? v : ~0u);
                uint32_t const nNon = __reduce_add_sync(0xffffffffu, threadIdx.x < kCtas && !pos ? 1u : 0u);
                uint32_t const sum = __reduce_add_sync(0xffffffffu, pos ? (v - kEmpty) >> 8 : 0u);
                if (lane == 0)
                {
                    sts128(sRed + 16 * (kTok * warp + r), make_uint4(mnAll, mnPos, nNon, sum));
                }
            }
            __syncthreads();
            if (lane == 0 && (!rs || warp == 0))
            {
                uint32_t mnAll = ~0u;
                uint32_t mnPos = ~0u;
                uint32_t nNon = 0;
                uint32_t sum = 0;
#pragma unroll
                for (int x = 0; x < kWarps; ++x)
                {
                    uint4 const v = lds128(sRed + 16 * (kTok * x + t));
                    mnAll = min(mnAll, v.x);
                    mnPos = min(mnPos, v.y);
                    nNon += v.z;
                    sum += v.w;
                }
                uint32_t const Tr = max(nNon <= 7 && mnPos != ~0u ? mnPos : mnAll, 1u);
                uint32_t const nPos = static_cast<uint32_t>(kCtas) - nNon;
                uint32_t const est = 2 * nPos >= static_cast<uint32_t>(kCtas) && !(GLM52_FLAGS(ctx) & kFlagTopkOneList)
                    ? kEmpty + ((sum / nPos) << 8)
                    : 0u;
                uint32_t const T2r = est > Tr + kTightGap ? est - kTightGap : Tr;
                stU64(tw + 2 * t,
                    (static_cast<uint64_t>(T2r == kEmpty ? kBndZero : T2r) << 32) | (Tr == kEmpty ? kBndZero : Tr));
                st32(tw + 2 * kTok + t, est > Tr && est != kEmpty ? est : 1u); // 1: no estimate (below every bound)
            }
            subStamp(ctx, 77);
        }
        if (warp == 0)
        {
            bool const want = rs ? lane == t : lane < kTok;
            uint64_t v = want ? ldPollU64(tw + 2 * lane) : 0ull;
            uint64_t t0 = 0;
            while (!warpAll(!want || static_cast<uint32_t>(v) != kEmpty))
            {
                spinCheck(ctx, t0, 167, lane, 0);
                if (want && static_cast<uint32_t>(v) == kEmpty)
                {
                    v = ldPollU64(tw + 2 * lane);
                }
            }
            if (want)
            {
                uint32_t const lo = static_cast<uint32_t>(v);
                uint32_t const hi = static_cast<uint32_t>(v >> 32);
                sts32(sMin + 768 + 4 * lane, lo == kBndZero ? kEmpty : lo);
                sts32(sMin + 896 + 4 * lane, hi == kBndZero ? kEmpty : hi);
            }
        }
    }
#endif
    __syncthreads();
    // kFlagCandWarps (row shard): every warp counts and writes the candidates of its share of the lanes' keys.
    bool const split = (kSpecializedStaticFlags & kFlagCandWarps) != 0 && rs;
    if (!active && !split)
    {
        return;
    }
    int const mBeg = split ? warp * nPer / kWarps : 0;
    int const mEnd = split ? (warp + 1) * nPer / kWarps : nPer;
    uint32_t const T = lds32(sMin + 768 + 4 * t);
    uint32_t const T2 = lds32(sMin + 896 + 4 * t);
#if GLM52_TOPK_UNION
    uint32_t const hsh = lds32(sMin + 864 + 4 * t); // the bin width the T CTA chose (log2)
#else
    uint32_t const hsh = kHistShift;
#endif
    uint32_t const hspan = static_cast<uint32_t>(kHistBins) << hsh;
    subStamp(ctx, 44);
    // The lane's keys of each class (A: > T2, == T2; B: in (T, T2), == T when T2 > T), their exclusive prefixes over
    // the lanes and the warp's totals (two classes per 32-bit word, sums <= 768): each lane then writes its own
    // candidates (slots per list: the keys above its threshold lane by lane, then the first ties lane by lane), with no
    // warp collective per key (a ballot pair per key, the keys of a lane ~3% candidates, took ~3 us). The count keeps
    // the indices m of the lane's last eight keys >= T (8 bits each): with at most eight (a lane has ~0.7 on average),
    // the writes visit only those. (A second pass over all keys and their stores took ~1.4 us; with four kept, ~15 % of
    // the CTAs still had a lane in that pass, and the last CTA sets the selectors' start.)
    bool const two = T2 > T;
    bool const hist = !(GLM52_FLAGS(ctx) & kFlagTopkNoHist);
    uint32_t* const hs = gc + kCIdxHist + t * kHistBins;
    uint32_t cntA = 0;
    uint32_t cntB = 0;
    uint32_t cntH = 0;
    uint64_t rec = 0;
    // The keys for the histogram are only counted here (their adds go with their candidates' stores below: in this
    // loop they cost +1.35 us per slice even with the selector's window on the critical path, mk-v144a).
#if GLM52_CAND_FILTER_ROLL
#pragma unroll 1
#else
#pragma unroll 4
#endif
    for (int m = mBeg; m < mEnd; ++m)
    {
        uint32_t const v = selKeyAt(a0, tL, lane + 32 * m);
        cntA += v > T2 ? 1u : v == T2 ? 1u << 16 : 0u;
        cntB += v > T && v < T2 ? 1u : two && v == T ? 1u << 16 : 0u;
        rec = v >= T ? (rec << 8) | static_cast<uint64_t>(m) : rec;
        cntH += hist && v > T2 && v - T2 < hspan ? 1u : 0u;
    }
    uint32_t preA = cntA;
    uint32_t preB = cntB;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1)
    {
        uint32_t const va = __shfl_up_sync(0xffffffffu, preA, off);
        uint32_t const vb = __shfl_up_sync(0xffffffffu, preB, off);
        preA += lane >= off ? va : 0u;
        preB += lane >= off ? vb : 0u;
    }
    uint32_t totA = __shfl_sync(0xffffffffu, preA, 31);
    uint32_t totB = __shfl_sync(0xffffffffu, preB, 31);
    preA -= cntA;
    preB -= cntB;
    uint32_t cntHw = hist ? __reduce_add_sync(0xffffffffu, cntH) : 0u;
    uint32_t const scr = sMin + 1024; // kFlagCandWarps: the warps' totals [0, 24), the CTA's results [24, 30)
    if (split)
    {
        // The warps' totals; the CTA's (and the earlier warps' share of them) after one barrier.
        if (lane == 0)
        {
            sts32(scr + 4 * warp, totA);
            sts32(scr + 4 * (8 + warp), totB);
            sts32(scr + 4 * (16 + warp), cntHw);
        }
        __syncthreads();
        uint32_t pA = 0;
        uint32_t pB = 0;
        totA = 0;
        totB = 0;
        cntHw = 0;
#pragma unroll
        for (int w = 0; w < kWarps; ++w)
        {
            uint32_t const a = lds32(scr + 4 * w);
            uint32_t const b = lds32(scr + 4 * (8 + w));
            pA += w < warp ? a : 0u;
            pB += w < warp ? b : 0u;
            totA += a;
            totB += b;
            cntHw += lds32(scr + 4 * (16 + w));
        }
        preA += pA;
        preB += pB;
    }
    uint32_t const gtA = totA & 0xffffu;
    uint32_t const eqA = totA >> 16;
    uint32_t const gtB = totB & 0xffffu;
    uint32_t const eqB = totB >> 16;
    uint32_t tieA = min(eqA, kTieFree);
    uint32_t tieB = min(eqB, kTieFree);
    // One lane per CTA (the row shard's warp 0, or every warp of its row without it) takes the extra ties, the
    // histogram's tie add and the slot atomic.
    bool const leadW = !split || warp == 0;
    if (leadW && (eqA > kTieFree || eqB > kTieFree))
    {
        uint32_t extraA = 0;
        uint32_t extraB = 0;
        if (lane == 0 && eqA > kTieFree)
        {
            uint32_t const used = atomicAdd(gc + kCIdxTies + t, eqA - kTieFree);
            extraA = used < static_cast<uint32_t>(kTopK) ? min(eqA - kTieFree, kTopK - used) : 0u;
        }
        if (lane == 0 && eqB > kTieFree)
        {
            uint32_t const used = atomicAdd(gc + kCIdxTiesB + t, eqB - kTieFree);
            extraB = used < static_cast<uint32_t>(kTopK) ? min(eqB - kTieFree, kTopK - used) : 0u;
        }
        tieA += __shfl_sync(0xffffffffu, extraA, 0);
        tieB += __shfl_sync(0xffffffffu, extraB, 0);
    }
    // The CTA's keys in the histogram: the ones counted above (added with their stores) and its keys equal to T2 taken
    // (bin 0). Their count rides on the slot atomic: the selector waits until the bins' sum equals the row's (every add
    // landed).
    uint32_t const totH = hist ? cntHw + tieA : 0u;
    if (hist && leadW && lane == 0 && tieA != 0)
    {
        redRelaxedAdd(hs, tieA);
    }
    subStamp(ctx, 67);
    uint32_t baseA = 0;
    uint32_t baseB = 0;
    if (leadW && lane == 0)
    {
        unsigned long long const add
            = (static_cast<unsigned long long>((totH << 18) | (gtB + tieB)) << 32) | ((1u << 24) | (gtA + tieA));
        unsigned long long const old = atomicAdd(reinterpret_cast<unsigned long long*>(gc + kCIdxCand + 2 * t), add);
        baseA = static_cast<uint32_t>(old) & 0xffffffu;
        baseB = static_cast<uint32_t>(old >> 32) & 0x3ffffu;
        if (split)
        {
            sts32(scr + 4 * 24, baseA);
            sts32(scr + 4 * 25, baseB);
            sts32(scr + 4 * 26, tieA);
            sts32(scr + 4 * 27, tieB);
        }
    }
    if (split)
    {
        __syncthreads();
        baseA = lds32(scr + 4 * 24);
        baseB = lds32(scr + 4 * 25);
        tieA = lds32(scr + 4 * 26);
        tieB = lds32(scr + 4 * 27);
    }
    else
    {
        baseA = __shfl_sync(0xffffffffu, baseA, 0);
        baseB = __shfl_sync(0xffffffffu, baseB, 0);
    }
    subStamp(ctx, 68);
    if (cntA + cntB == 0)
    {
        return;
    }
    uint64_t* const rowA = cand + static_cast<size_t>(t) * kCandCap;
    uint32_t hiA = baseA + (preA & 0xffffu);
    uint32_t tA = preA >> 16;
    uint32_t hiB = baseB + (preB & 0xffffu);
    uint32_t tB = preB >> 16;
    auto put = [&](int m)
    {
        uint32_t const v = selKeyAt(a0, tL, lane + 32 * m);
        uint32_t slot = ~0u;
        if (v > T2)
        {
            slot = hiA++;
            if (hist && v - T2 < hspan)
            {
                redRelaxedAdd(hs + ((v - T2) >> hsh), 1u);
            }
        }
        else if (v == T2)
        {
            slot = tA < tieA ? baseA + gtA + tA : ~0u;
            ++tA;
        }
        else if (v > T)
        {
            slot = kCandList + hiB++;
        }
        else
        {
            slot = tB < tieB ? kCandList + baseB + gtB + tB : ~0u;
            ++tB;
        }
        // Each list holds every key of the row (kCandList).
        if (slot < (v >= T2 ? kCandList : kCandCap))
        {
            // Local key k: tile k / 128, page 2 (k / 128) + (k % 128) / 64 of the CTA's, row k % 64.
            int const k = lane + 32 * m;
            int const page = 2 * (k >> 7) + ((k & 127) >> 6);
            uint32_t const pos = static_cast<uint32_t>(
                idxPageOf(cta, page, (L + kIdxPageKeys - 1) / kIdxPageKeys) * kIdxPageKeys + (k & 63));
            stU64(rowA + slot, (static_cast<uint64_t>(pos) << 32) | v);
        }
    };
    int const nRec = static_cast<int>((cntA & 0xffffu) + (cntA >> 16) + (cntB & 0xffffu) + (cntB >> 16));
    if (nRec <= 8)
    {
#pragma unroll 1
        for (int i = 0; i < nRec; ++i)
        {
            put(static_cast<int>((rec >> (8 * i)) & 0xffu));
        }
    }
    else
    {
#if GLM52_CAND_FILTER_ROLL
#pragma unroll 1
#else
#pragma unroll 4
#endif
        for (int m = mBeg; m < mEnd; ++m)
        {
            if (selKeyAt(a0, tL, lane + 32 * m) >= T)
            {
                put(m);
            }
        }
    }
}

// A row's top need0 from its n0 > kSelSmall candidates in lst0 ((key, position) pairs, keys >= lo, the largest hiKey;
// not modified; lst1: kSelSmall pairs of scratch; hist: 2048 words; sSel: 32 words). Each round scans lst0 restricted
// to the window
// [lo, lo + range] (all of it at first): a 2048-bin histogram of key - lo (bins as wide as the window needs), the keys
// above the crossing bin to the output (slots by a block-wide scan of the threads' counts), the crossing bin as the
// next window. Once the window holds at most kSelSmall keys they are gathered into lst1 (in that round's writes) and
// ranked against each other (larger key first, then the earlier index); with single-key bins the crossing key's ties
// are taken directly. Outputs: the positions into the shared int32 [need0] at outS (written out coalesced by the
// caller; per-thread slots in global memory were scattered stores). (Adding once per distinct bin of a warp with
// match.any made the histogram pass 2x slower.)
__device__ __noinline__ void idxSelect(Ctx const& ctx, uint2 const* lst0, uint2* lst1, uint32_t* hist, uint32_t* sSel,
    int n0, int need0, uint32_t lo, uint32_t hiKey, uint32_t outS)
{
    int const tid = threadIdx.x;
    uint32_t const l0 = smemU32(lst0);
    uint32_t const l1 = smemU32(lst1);
    uint32_t const hs = smemU32(hist);
    uint32_t range = hiKey > lo ? hiKey - lo : 0u;
    int need = need0;
    int emitted = 0;
    int nWin = n0;
    int nSmall = -1;
#pragma unroll 1
    while (nWin > kSelSmall)
    {
        int const s = range < 2048u ? 0 : 21 - __clz(static_cast<int>(range));
#pragma unroll 1
        for (int b = tid; b < 2048 / 4; b += kThreads)
        {
            reinterpret_cast<uint4*>(hist)[b] = make_uint4(0u, 0u, 0u, 0u);
        }
        __syncthreads();
#pragma unroll 4
        for (int i = tid; i < n0; i += kThreads)
        {
            uint32_t const d = lds32(l0 + 8 * i) - lo;
            if (d <= range)
            {
                redsAdd(hs + 4 * (d >> s), 1u);
            }
        }
        __syncthreads();
        int bstar;
        uint32_t above;
        uint32_t inBin;
        blockCrossBin(hist, static_cast<uint32_t>(need), sSel, bstar, above, inBin);
        if (emitted == 0)
        {
            subStamp(ctx, 62);
        }
        int const ties = need - static_cast<int>(above);
        bool const gather = s != 0 && inBin <= static_cast<uint32_t>(kSelSmall);
        uint32_t nHi = 0;
        uint32_t nEq = 0;
#pragma unroll 4
        for (int i = tid; i < n0; i += kThreads)
        {
            uint32_t const d = lds32(l0 + 8 * i) - lo;
            if (d <= range)
            {
                int const b = static_cast<int>(d >> s);
                nHi += b > bstar ? 1u : 0u;
                nEq += b == bstar ? 1u : 0u;
            }
        }
        uint32_t baseH;
        uint32_t baseE;
        blockScan2(nHi, nEq, smemU32(sSel + 16), baseH, baseE);
        uint32_t const o = outS + 4 * emitted;
#pragma unroll 4
        for (int i = tid; i < n0; i += kThreads)
        {
            uint2 const e = lds64(l0 + 8 * i);
            uint32_t const d = e.x - lo;
            if (d <= range)
            {
                int const b = static_cast<int>(d >> s);
                if (b > bstar)
                {
                    sts32(o + 4 * baseH++, e.y);
                }
                else if (b == bstar)
                {
                    if (s == 0)
                    {
                        if (static_cast<int>(baseE) < ties)
                        {
                            sts32(o + 4 * (above + baseE), e.y);
                        }
                    }
                    else if (gather)
                    {
                        sts64(l1 + 8 * baseE, e);
                    }
                    ++baseE;
                }
            }
        }
        __syncthreads();
        if (emitted == 0)
        {
            subStamp(ctx, 63);
        }
        emitted += static_cast<int>(above);
        need = ties;
        if (s == 0)
        {
            nWin = 0;
            break;
        }
        lo += static_cast<uint32_t>(bstar) << s;
        range = (1u << s) - 1u;
        nWin = static_cast<int>(inBin);
        if (gather)
        {
            nSmall = nWin;
            break;
        }
    }
    // n0 > kSelSmall: the rounds end gathered into lst1, or with single-key bins (nothing left to rank).
    uint32_t const src = l1;
    int const n = nSmall >= 0 ? nSmall : 0;
#pragma unroll 1
    for (int j = tid; j < n; j += kThreads)
    {
        uint2 const e = lds64(src + 8 * j);
        int rank = 0;
#pragma unroll 4
        for (int i = 0; i < n; ++i)
        {
            uint32_t const k = lds32(src + 8 * i);
            rank += (k > e.x || (k == e.x && i < j)) ? 1 : 0;
        }
        if (rank < need)
        {
            sts32(outS + 4 * (emitted + rank), e.y);
        }
    }
}

static_assert(
    kIdxSelBase + kTok == kCtas && kIdxSelBase >= kIdxQUnits, "selector CTAs: the last 8, past the IdxQ ones");
static_assert(kIdxQUnits * kIdxQUnitRows == kIdxHeads * kIdxDim, "IdxQ units");
constexpr int kSelOut = kCandSmem * 8 + kSelSmall * 8 + 2048 * 4; // the selection's output, int32 [2048]
static_assert(32 * kThreads * 8 + 2 * 32 * kThreads * 4 <= kSelOut && 32 * kThreads > kSelSmall,
    "selector: pairs, then packed keys and positions; idxSelect's n0");
static_assert(kSelOut + kTopK * 4 <= kSelBytes, "selection buffers");
static_assert(1024 + 64 + 8 * kSelWin <= kSmemRedBytes, "idxSelectBin: the bin's keys");

// The selector of a row with nc <= 32 x 256 candidates (keys >= lo at shared address sKey, zero past nc to a multiple
// of four; their positions at sPos): X = the largest value with at least need0 keys >= X, by
// a search over [lo, the largest key], or over the window win from list A's histogram (win.z != 0) (a step: each
// thread's count of keys >= a probe, one block-wide sum; the first probes est and est +- kGallop, kGallop doubling,
// until both sides are known, when est is in the range, then bisection) until the window holds at most kSelWin keys (~8
// steps instead of ~25), then by ranking the window's keys against each other; the keys > X and need0 - count(> X) of
// the keys == X (any) are the row's selection, staged in shared memory and written to topkOut coalesced (the -1 fill
// past the row's valid keys included). Thread tid holds the keys of groups tid + 256 j (four keys each) in registers.
// The code is kept small: it runs once per leader launch on 8 SMs from cold instruction caches, where straight-line
// passes over the 32 registers (13.5 KB of code) took 11 us against 7 warm. So the counts of keys > X and == X reuse
// the search's count (probes X + 1 and X), and the scatter is a rolled loop. sSel: shared scratch, 64 words, then the
// staged output (int32 [2048]) from sSel + 512.
__device__ __noinline__ void idxSelectRow(Ctx const& ctx, int nc, int need0, uint32_t lo, uint32_t est, uint4 win,
    int* topkOut, int* arenaOut, int* mcOut, uint32_t sSel, uint32_t sKey, uint32_t sPos)
{
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    int const nq = (nc + 3) >> 2;
    int const ng = (nq + kThreads - 1) / kThreads; // the thread's groups of four that can hold keys (the rest are zero)
    uint32_t mx = 0;
    uint32_t k[32];
#pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        int const q = tid + kThreads * j;
        uint4 const g = q < nq ? lds128(sKey + 16 * q) : make_uint4(0u, 0u, 0u, 0u);
        k[4 * j] = g.x;
        k[4 * j + 1] = g.y;
        k[4 * j + 2] = g.z;
        k[4 * j + 3] = g.w;
        mx = max(mx, max(max(g.x, g.y), max(g.z, g.w)));
    }
    mx = __reduce_max_sync(0xffffffffu, mx);
    if (lane == 0)
    {
        sts32(sSel + 4 * (16 + warp), mx);
    }
    __syncthreads();
#pragma unroll
    for (int x = 0; x < kWarps; ++x)
    {
        mx = max(mx, lds32(sSel + 4 * (16 + x)));
    }
    subStamp(ctx, 61);
    // Search invariant: count(>= a) = ca >= need0 > cb = count(> b), and the answer is in [a, b]. Stage 0 bisects
    // (probe mid), stage 3 ranks the window's ca - cb <= kSelWin keys (gathered into the output staging area), stage 1
    // takes the thread's count of keys >= X + 1 (= > X), stage 2 its count of keys >= X; X + 1 does not wrap (order
    // keys of finite logits stay below 0xff800000).
    uint32_t a = lo;
    uint32_t b = max(mx, lo);
    uint32_t ca = static_cast<uint32_t>(nc);
    uint32_t cb = 0;
    if (win.z != 0)
    {
        // The first window from list A's histogram (the row shard): count(>= win.x) = win.z >= need0 > win.w =
        // count(> win.y); win.y = 0: up to the largest key.
        a = win.x;
        b = win.y != 0 ? win.y : b;
        ca = win.z;
        cb = win.w;
    }
    int stage = a < b ? (ca - cb > static_cast<uint32_t>(kSelWin) ? 0 : 3) : 1;
    // g: 3 = est is the next probe, 1 / 2 = galloping up / down from it, 0 = bisection.
    int g = est > a && est <= b ? 3 : 0;
    uint32_t gd = kGallop;
    uint32_t probe = stage == 0 ? (g != 0 ? est : a + (b - a) / 2 + 1) : a + 1;
    int parity = 0;
    int steps = 0; // bisection steps, stored once (stamped runs; a global add per step stalled thread 0)
    uint32_t nHi = 0;
    uint32_t nEq = 0;
    uint32_t const sWin = sSel + 512;
    uint32_t const sWinN = sSel + 4 * 40;
#pragma unroll 1
    while (true)
    {
        if (stage == 3)
        {
            subStamp(ctx, 66);
            // The window's keys gather at sWin, zero past them to a multiple of four (zero is below every key).
            if (tid == 0)
            {
                sts32(sWinN, 0u);
            }
#pragma unroll 1
            for (int i = tid; i < kSelWin + 4; i += kThreads)
            {
                sts32(sWin + 4 * i, 0u);
            }
            __syncthreads();
#pragma unroll 1
            for (int q = tid; q < nq; q += kThreads)
            {
                uint4 const g = lds128(sKey + 16 * q);
                uint32_t const w[4] = {g.x, g.y, g.z, g.w};
#pragma unroll
                for (int e = 0; e < 4; ++e)
                {
                    if (w[e] >= a && w[e] <= b)
                    {
                        sts32(sWin + 4 * atomsAdd(sWinN, 1u), w[e]);
                    }
                }
            }
            __syncthreads();
            int const nW = static_cast<int>(lds32(sWinN));
            // The largest window key v with cb + (window keys >= v) >= need0.
            uint32_t best = 0;
#pragma unroll 1
            for (int j = tid; j < nW; j += kThreads)
            {
                uint32_t const v = lds32(sWin + 4 * j);
                uint32_t ge = 0;
#pragma unroll 2
                for (int i = 0; i < nW; i += 4)
                {
                    uint4 const w = lds128(sWin + 4 * i);
                    ge += (w.x >= v ? 1u : 0u) + (w.y >= v ? 1u : 0u) + (w.z >= v ? 1u : 0u) + (w.w >= v ? 1u : 0u);
                }
                best = cb + ge >= static_cast<uint32_t>(need0) ? max(best, v) : best;
            }
            best = __reduce_max_sync(0xffffffffu, best);
            if (lane == 0)
            {
                sts32(sSel + 4 * (8 * parity + warp), best);
            }
            __syncthreads();
            best = 0;
#pragma unroll
            for (int x = 0; x < kWarps; ++x)
            {
                best = max(best, lds32(sSel + 4 * (8 * parity + x)));
            }
            parity ^= 1;
            a = best;
            stage = 1;
            probe = a + 1;
            continue;
        }
        uint32_t c0 = 0;
        uint32_t c1 = 0;
        uint32_t c2 = 0;
        uint32_t c3 = 0;
#pragma unroll
        for (int j = 0; j < 32; j += 4)
        {
            if (j < 4 * ng)
            {
                c0 += k[j] >= probe ? 1u : 0u;
                c1 += k[j + 1] >= probe ? 1u : 0u;
                c2 += k[j + 2] >= probe ? 1u : 0u;
                c3 += k[j + 3] >= probe ? 1u : 0u;
            }
        }
        uint32_t const c = (c0 + c1) + (c2 + c3);
        if (stage == 2)
        {
            nEq = c - nHi;
            break;
        }
        if (stage == 1)
        {
            nHi = c;
            stage = 2;
            probe = a;
            continue;
        }
        uint32_t const cw = __reduce_add_sync(0xffffffffu, c);
        if (lane == 0)
        {
            sts32(sSel + 4 * (8 * parity + warp), cw);
        }
        __syncthreads();
        uint32_t tot = 0;
#pragma unroll
        for (int x = 0; x < kWarps; ++x)
        {
            tot += lds32(sSel + 4 * (8 * parity + x));
        }
        parity ^= 1;
        ++steps;
        if (steps <= 5)
        {
            subStamp(ctx, 90 + steps); // stamped runs: the first five steps' ends (sub-stamps 91-95)
        }
        bool const up = tot >= static_cast<uint32_t>(need0);
        if (up)
        {
            a = probe;
            ca = tot;
        }
        else
        {
            b = probe - 1;
            cb = tot;
        }
        g = g == 3 ? (up ? 1 : 2) : (g == 1 && !up) || (g == 2 && up) ? 0 : g;
        stage = a < b ? (ca - cb > static_cast<uint32_t>(kSelWin) ? 0 : 3) : 1;
        if (stage == 0 && g != 0 && b - a > gd)
        {
            probe = g == 1 ? a + gd : b + 1 - gd;
            gd <<= 1;
        }
        else
        {
            g = 0;
            probe = stage == 0 ? a + (b - a) / 2 + 1 : a + 1;
        }
    }
    uint32_t const X = a;
    subStamp(ctx, 62);
    if (GLM52_STAMPS(ctx) != nullptr && tid == 0)
    {
        GLM52_STAMPS(ctx)[kMaxRoles * 4 + 4 + 90] = static_cast<uint64_t>(steps);
    }
    uint32_t baseH;
    uint32_t baseE;
    blockScan2(nHi, nEq, sSel + 4 * 24, baseH, baseE);
    uint32_t totHi = 0;
#pragma unroll
    for (int x = 0; x < kWarps; ++x)
    {
        totHi += lds32(sSel + 4 * (24 + x));
    }
    uint32_t const ties = static_cast<uint32_t>(need0) > totHi ? static_cast<uint32_t>(need0) - totHi : 0u;
    uint32_t const sOut = sSel + 512;
    // The thread's keys in the order counted above: groups tid + 256 j, four keys and their positions per 16-byte load
    // (keys past nc are zero, below X).
#pragma unroll 1
    for (int q = tid; q < nq; q += kThreads)
    {
        uint4 const kk = lds128(sKey + 16 * q);
        uint4 const pp = lds128(sPos + 16 * q);
        uint32_t const key[4] = {kk.x, kk.y, kk.z, kk.w};
        uint32_t const pos[4] = {pp.x, pp.y, pp.z, pp.w};
#pragma unroll
        for (int e = 0; e < 4; ++e)
        {
            if (key[e] > X)
            {
                sts32(sOut + 4 * baseH++, pos[e]);
            }
            else if (key[e] == X)
            {
                if (baseE < ties)
                {
                    sts32(sOut + 4 * (totHi + baseE), pos[e]);
                }
                ++baseE;
            }
        }
    }
    subStamp(ctx, 63);
    __syncthreads();
    int const filled = min(nc, need0);
#pragma unroll
    for (int v = 0; v < kTopK / 4 / kThreads; ++v)
    {
        int const i = 4 * (tid + kThreads * v);
        uint4 o = lds128(sOut + 4 * i);
        o.x = i < filled ? o.x : ~0u;
        o.y = i + 1 < filled ? o.y : ~0u;
        o.z = i + 2 < filled ? o.z : ~0u;
        o.w = i + 3 < filled ? o.w : ~0u;
        if (mcOut != nullptr)
        {
            st128(mcOut + i, o); // the top-k exchange first: the other ranks wait on it
        }
        if (topkOut != nullptr)
        {
            st128(topkOut + i, o);
            st128(arenaOut + i, o);
        }
    }
}

// The selector of a row with more than 32 x 256 candidates ((key, position) pairs at shared address a0 = s0): its
// largest key, then the histogram rounds of idxSelect; the result out of shared memory.
__device__ __noinline__ void idxSelectLarge(Ctx const& ctx, int nc, int need0, uint32_t lo, uint8_t* s0, uint32_t a0,
    uint32_t* sSel, int* topkOut, int* arenaOut, int* mcOut)
{
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    uint32_t mx = 0;
    for (int i = tid; i < nc; i += kThreads)
    {
        mx = max(mx, lds32(a0 + 8 * i));
    }
    mx = __reduce_max_sync(0xffffffffu, mx);
    uint32_t const sMx = smemU32(sSel) + 4 * 48;
    if (lane == 0)
    {
        sts32(sMx + 4 * warp, mx);
    }
    __syncthreads();
#pragma unroll
    for (int x = 0; x < kWarps; ++x)
    {
        mx = max(mx, lds32(sMx + 4 * x));
    }
    uint32_t const outS = a0 + kSelOut;
    idxSelect(ctx, reinterpret_cast<uint2 const*>(s0), reinterpret_cast<uint2*>(s0 + kCandSmem * 8),
        reinterpret_cast<uint32_t*>(s0 + kCandSmem * 8 + kSelSmall * 8), sSel, nc, need0, lo, mx, outS);
    __syncthreads();
    int const filled = min(nc, need0);
#pragma unroll 4
    for (int i = tid; i < kTopK; i += kThreads)
    {
        int const v = i < filled ? static_cast<int>(lds32(outS + 4 * i)) : -1;
        if (mcOut != nullptr)
        {
            mcOut[i] = v;
        }
        if (topkOut != nullptr)
        {
            topkOut[i] = v;
            arenaOut[i] = v;
        }
    }
}

// The selector when list A's histogram puts the row's need0-th key in a bin [a, b] of at most kSelWin keys
// (count(> b) = cb, count(>= a) = ca; list A's nc (key, position) pairs at shared address a0, copied from list, or with
// fromList read from list by the pass itself, the next four pairs per lane in flight while it handles four): one
// pass, four pairs per lane at a time, writes the positions of the keys > b (the row's cb surest) straight into the
// row's output out (and out2 unless null: the exchange row, or the top-k buffer and the arena), slots from one
// warp-aggregated shared counter per four pairs (keys > b | the bin's << 16), and gathers the bin's keys; a pair the
// copy read before it landed (key 0) is polled from list. Warp 0 then ranks the bin (larger key first, then the
// earlier entry) and writes the need0 - cb largest after them, while the other warps clear list for the next leader
// launch (every slot has landed). No search, no packing. (Staging the row in shared memory for 16-byte stores at the
// end was 4.2 us slower per slice, mk-v125a: the attention then waits for the whole row.)
__device__ __noinline__ void idxSelectBin(Ctx const& ctx, int nc, int need0, uint4 win, uint32_t a0, uint32_t sSel,
    int* out, int* out2, uint64_t* list, bool fromList, uint2 const* pre)
{
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    uint32_t const a = win.x;
    uint32_t const b = win.y;
    uint32_t const cb = win.w;
    uint32_t const sWinK = sSel + 64;
    uint32_t const sWinP = sWinK + 4 * kSelWin;
    if (tid == 0)
    {
        sts32(sSel, 0u);
    }
    __syncthreads();
    subStamp(ctx, 61);
    unsigned const below = (1u << lane) - 1u;
    auto load = [&](int i0, uint2(&kp)[4])
    {
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            int const i = i0 + lane + 32 * k;
            if (i >= nc)
            {
                kp[k] = make_uint2(0u, 0u);
            }
            else if (fromList && (GLM52_FLAGS(ctx) & kFlagBinVolatile))
            {
                uint64_t const v = ldPollU64(list + i);
                kp[k] = make_uint2(static_cast<uint32_t>(v), static_cast<uint32_t>(v >> 32));
            }
            else if (fromList)
            {
                // Plain L2 loads (ld.global.cg): free to overlap the processing of the four before (volatile loads
                // kept the pass at ~0.35 us per four pairs); a pair read before it landed is polled below.
                kp[k] = __ldcg(reinterpret_cast<uint2 const*>(list + i));
            }
            else
            {
                kp[k] = lds64(a0 + 8 * i);
            }
        }
    };
    uint2 kp[4];
    uint2 nxt[4] = {make_uint2(0u, 0u), make_uint2(0u, 0u), make_uint2(0u, 0u), make_uint2(0u, 0u)};
    if (pre != nullptr)
    {
        // The pass's first four pairs per lane, loaded by the caller before the window (in flight through its polls).
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            kp[k] = pre[k];
        }
    }
    else
    {
        load(128 * warp, kp);
    }
#pragma unroll 1
    for (int i0 = 128 * warp; i0 < nc; i0 += 128 * kWarps)
    {
        if (i0 + 128 * kWarps < nc)
        {
            load(i0 + 128 * kWarps, nxt);
        }
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            int const i = i0 + lane + 32 * k;
            uint64_t t0 = 0;
            while (i < nc && kp[k].x == 0u)
            {
                uint64_t const v = ldPollU64(list + i);
                kp[k] = make_uint2(static_cast<uint32_t>(v), static_cast<uint32_t>(v >> 32));
                if (kp[k].x == 0u)
                {
                    spinCheck(ctx, t0, 173, i, 0);
                }
            }
        }
        unsigned ms[4];
        unsigned mb[4];
        uint32_t n = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            ms[k] = __ballot_sync(0xffffffffu, kp[k].x > b);
            mb[k] = __ballot_sync(0xffffffffu, kp[k].x >= a && kp[k].x <= b);
            n += static_cast<uint32_t>(__popc(ms[k])) | (static_cast<uint32_t>(__popc(mb[k])) << 16);
        }
        uint32_t base = 0;
        if (lane == 0 && n != 0)
        {
            base = atomsAdd(sSel, n);
        }
        base = __shfl_sync(0xffffffffu, base, 0);
        uint32_t bs = base & 0xffffu;
        uint32_t bb = base >> 16;
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            uint32_t const s = bs + static_cast<uint32_t>(__popc(ms[k] & below));
            uint32_t const w = bb + static_cast<uint32_t>(__popc(mb[k] & below));
            if (kp[k].x > b && s < cb)
            {
                st32(out + s, kp[k].y);
                if (out2 != nullptr)
                {
                    st32(out2 + s, kp[k].y);
                }
            }
            else if (kp[k].x >= a && kp[k].x <= b && w < static_cast<uint32_t>(kSelWin))
            {
                sts32(sWinK + 4 * w, kp[k].x);
                sts32(sWinP + 4 * w, kp[k].y);
            }
            bs += static_cast<uint32_t>(__popc(ms[k]));
            bb += static_cast<uint32_t>(__popc(mb[k]));
        }
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            kp[k] = nxt[k];
        }
    }
    __syncthreads();
    subStamp(ctx, 62);
    uint32_t const cnt = lds32(sSel);
    uint32_t const nSure = cnt & 0xffffu;
    uint32_t const nBin = cnt >> 16;
    if (nSure != cb || nBin != win.z - cb)
    {
        fail(ctx, 177, static_cast<int>(nSure), static_cast<int>(nBin)); // the histogram and the list disagree
    }
    if (warp == 0)
    {
        int const want = need0 - static_cast<int>(cb);
#pragma unroll 1
        for (int j = lane; j < static_cast<int>(nBin); j += 32)
        {
            uint32_t const v = lds32(sWinK + 4 * j);
            int r = 0;
#pragma unroll 1
            for (int x = 0; x < static_cast<int>(nBin); ++x)
            {
                uint32_t const u = lds32(sWinK + 4 * x);
                r += u > v || (u == v && x < j) ? 1 : 0;
            }
            if (r < want)
            {
                uint32_t const pos = lds32(sWinP + 4 * j);
                st32(out + cb + r, pos);
                if (out2 != nullptr)
                {
                    st32(out2 + cb + r, pos);
                }
            }
        }
    }
    else
    {
        // The list back to zero for the next leader launch (two pairs per store).
#pragma unroll 1
        for (int i = 2 * (tid - 32); i < nc; i += 2 * (kThreads - 32))
        {
            st128(list + i, make_uint4(0u, 0u, 0u, 0u));
        }
    }
    subStamp(ctx, 63);
    // Past the row's valid keys (short rows only): -1.
#pragma unroll 1
    for (int i = need0 + tid; i < kTopK; i += kThreads)
    {
        st32(out + i, ~0u);
        if (out2 != nullptr)
        {
            st32(out2 + i, ~0u);
        }
    }
}

// Optional row-shard bin pass. CTA 204 publishes either fallback (state 1) or a release-ordered descriptor (state 2).
// CTAs 204..211 own disjoint contiguous ranges of 128 pairs; each pair is polled before use and before its clear.
// A single packed global allocator assigns disjoint sure-output and boundary slots. The CTA completion releases
// order all boundary stores before CTA 204's acquire; only CTA 204 ranks the boundary and counts the row done.
// The new counter tail is cleared with the next copy, after its previous readers finished, just like the histogram.
__device__ __noinline__ void idxSelectBinParallel(Ctx const& ctx, int need0, uint64_t* list, int* out)
{
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    int const part = static_cast<int>(blockIdx.x) - kIdxSelBase;
    uint32_t* const gc = ctx.counters;
    uint32_t const sSel = ctx.smemBase + kSmemRed + 1024;
    if (tid == 0)
    {
        uint64_t t0 = 0;
        uint32_t state;
        while ((state = ldAcquire(gc + kCIdxParState)) == 0u)
        {
            spinCheck(ctx, t0, 178, part, 0);
        }
        sts32(sSel, state);
    }
    __syncthreads();
    if (lds32(sSel) == 1u)
    {
        return; // the original selector owns every candidate and output; helpers have read neither
    }
    int const nc = static_cast<int>(ldPoll32(gc + kCIdxParN));
    uint4 const win = ldPoll128(gc + kCIdxParWin);
    uint32_t const a = win.x;
    uint32_t const b = win.y;
    uint32_t const cb = win.w;
    int const chunks = (nc + 127) / 128;
    int const begin = 128 * (chunks * part / kTok);
    int const end = min(nc, 128 * (chunks * (part + 1) / kTok));
    unsigned const below = (1u << lane) - 1u;
    subStamp(ctx, 61);
#pragma unroll 1
    for (int i0 = begin + 128 * warp; i0 < end; i0 += 128 * kWarps)
    {
        uint2 kp[4];
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            int const i = i0 + lane + 32 * k;
            if (i >= end)
            {
                kp[k] = make_uint2(0u, 0u);
            }
            else if (GLM52_FLAGS(ctx) & kFlagBinVolatile)
            {
                uint64_t const v = ldPollU64(list + i);
                kp[k] = make_uint2(static_cast<uint32_t>(v), static_cast<uint32_t>(v >> 32));
            }
            else
            {
                kp[k] = __ldcg(reinterpret_cast<uint2 const*>(list + i));
            }
        }
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            int const i = i0 + lane + 32 * k;
            uint64_t t0 = 0;
            while (i < end && kp[k].x == 0u)
            {
                uint64_t const v = ldPollU64(list + i);
                kp[k] = make_uint2(static_cast<uint32_t>(v), static_cast<uint32_t>(v >> 32));
                if (kp[k].x == 0u)
                {
                    spinCheck(ctx, t0, 173, i, part);
                }
            }
        }
        unsigned ms[4];
        unsigned mb[4];
        uint32_t n = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            ms[k] = __ballot_sync(0xffffffffu, kp[k].x > b);
            mb[k] = __ballot_sync(0xffffffffu, kp[k].x >= a && kp[k].x <= b);
            n += static_cast<uint32_t>(__popc(ms[k])) | (static_cast<uint32_t>(__popc(mb[k])) << 16);
        }
        uint32_t base = 0;
        if (lane == 0 && n != 0u)
        {
            // cb < need0 <= 2048 and the boundary has <=64 entries: neither packed field can carry into the other.
            base = atomicAdd(gc + kCIdxParCount, n);
        }
        base = __shfl_sync(0xffffffffu, base, 0);
        uint32_t bs = base & 0xffffu;
        uint32_t bb = base >> 16;
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            int const i = i0 + lane + 32 * k;
            uint32_t const s = bs + static_cast<uint32_t>(__popc(ms[k] & below));
            uint32_t const w = bb + static_cast<uint32_t>(__popc(mb[k] & below));
            if (kp[k].x > b && s < cb)
            {
                st32(out + s, kp[k].y); // streaming multicast, so attention may consume completed slots now
            }
            else if (kp[k].x >= a && kp[k].x <= b && w < static_cast<uint32_t>(kSelWin))
            {
                st32(gc + kCIdxParKey + w, kp[k].x);
                st32(gc + kCIdxParPos + w, kp[k].y);
            }
            if (i < end)
            {
                stU64(list + i, 0ull); // this unique owner has observed the producer's single-copy 64-bit store
            }
            bs += static_cast<uint32_t>(__popc(ms[k]));
            bb += static_cast<uint32_t>(__popc(mb[k]));
        }
    }
    signalCounter(ctx, kCIdxParDone);
    if (part != 0)
    {
        return;
    }
    waitCounter(ctx, kCIdxParDone, kTok, 179);
    subStamp(ctx, 62);
    uint32_t const count = ldPoll32(gc + kCIdxParCount);
    uint32_t const nSure = count & 0xffffu;
    uint32_t const nBin = count >> 16;
    if (nSure != cb || nBin != win.z - cb || nBin > static_cast<uint32_t>(kSelWin))
    {
        fail(ctx, 180, static_cast<int>(nSure), static_cast<int>(nBin));
    }
    uint32_t const sWinK = sSel + 64;
    uint32_t const sWinP = sWinK + 4 * kSelWin;
    if (tid < static_cast<int>(nBin))
    {
        sts32(sWinK + 4 * tid, ldPoll32(gc + kCIdxParKey + tid));
        sts32(sWinP + 4 * tid, ldPoll32(gc + kCIdxParPos + tid));
    }
    __syncthreads();
    if (warp == 0)
    {
        int const want = need0 - static_cast<int>(cb);
#pragma unroll 1
        for (int j = lane; j < static_cast<int>(nBin); j += 32)
        {
            uint32_t const v = lds32(sWinK + 4 * j);
            int r = 0;
#pragma unroll 1
            for (int x = 0; x < static_cast<int>(nBin); ++x)
            {
                uint32_t const u = lds32(sWinK + 4 * x);
                r += u > v || (u == v && x < j) ? 1 : 0;
            }
            if (r < want)
            {
                st32(out + cb + r, lds32(sWinP + 4 * j));
            }
        }
    }
#pragma unroll 1
    for (int i = need0 + tid; i < kTopK; i += kThreads)
    {
        st32(out + i, ~0u);
    }
    subStamp(ctx, 63);
}

// kFlagSelFast: the parallel bin pass without its descriptor and completion counters (each a release, ~1 us). Every
// selector CTA took the window itself; the boundary keys go out as single 64-bit (key, position) stores into slots
// zeroed with the counters, which CTA kIdxSelBase polls (it knows their number from the window).
__device__ __noinline__ void idxSelectBinParallelFast(
    Ctx const& ctx, int need0, uint64_t* list, int* out, int const nc, uint4 const win)
{
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    int const part = static_cast<int>(blockIdx.x) - kIdxSelBase;
    uint32_t* const gc = ctx.counters;
    uint32_t const sSel = ctx.smemBase + kSmemRed + 1024;
    uint32_t const a = win.x;
    uint32_t const b = win.y;
    uint32_t const cb = win.w;
    int const chunks = (nc + 127) / 128;
    int const begin = 128 * (chunks * part / kTok);
    int const end = min(nc, 128 * (chunks * (part + 1) / kTok));
    unsigned const below = (1u << lane) - 1u;
    subStamp(ctx, 61);
#pragma unroll 1
    for (int i0 = begin + 128 * warp; i0 < end; i0 += 128 * kWarps)
    {
        uint2 kp[4];
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            int const i = i0 + lane + 32 * k;
            if (i >= end)
            {
                kp[k] = make_uint2(0u, 0u);
            }
            else if (GLM52_FLAGS(ctx) & kFlagBinVolatile)
            {
                uint64_t const v = ldPollU64(list + i);
                kp[k] = make_uint2(static_cast<uint32_t>(v), static_cast<uint32_t>(v >> 32));
            }
            else
            {
                kp[k] = __ldcg(reinterpret_cast<uint2 const*>(list + i));
            }
        }
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            int const i = i0 + lane + 32 * k;
            uint64_t t0 = 0;
            while (i < end && kp[k].x == 0u)
            {
                uint64_t const v = ldPollU64(list + i);
                kp[k] = make_uint2(static_cast<uint32_t>(v), static_cast<uint32_t>(v >> 32));
                if (kp[k].x == 0u)
                {
                    spinCheck(ctx, t0, 173, i, part);
                }
            }
        }
        unsigned ms[4];
        unsigned mb[4];
        uint32_t n = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            ms[k] = __ballot_sync(0xffffffffu, kp[k].x > b);
            mb[k] = __ballot_sync(0xffffffffu, kp[k].x >= a && kp[k].x <= b);
            n += static_cast<uint32_t>(__popc(ms[k])) | (static_cast<uint32_t>(__popc(mb[k])) << 16);
        }
        uint32_t base = 0;
        if (lane == 0 && n != 0u)
        {
            // cb < need0 <= 2048 and the boundary has <=64 entries: neither packed field can carry into the other.
            base = atomicAdd(gc + kCIdxParCount, n);
        }
        base = __shfl_sync(0xffffffffu, base, 0);
        uint32_t bs = base & 0xffffu;
        uint32_t bb = base >> 16;
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            int const i = i0 + lane + 32 * k;
            uint32_t const s = bs + static_cast<uint32_t>(__popc(ms[k] & below));
            uint32_t const w = bb + static_cast<uint32_t>(__popc(mb[k] & below));
            if (kp[k].x > b && s < cb)
            {
                st32(out + s, kp[k].y); // streaming multicast, so attention may consume completed slots now
            }
            else if (kp[k].x >= a && kp[k].x <= b && w < static_cast<uint32_t>(kSelWin))
            {
                stU64(reinterpret_cast<uint64_t*>(gc + kCIdxParKey) + w,
                    (static_cast<uint64_t>(kp[k].y) << 32) | kp[k].x);
            }
            if (i < end)
            {
                stU64(list + i, 0ull); // this unique owner has observed the producer's single-copy 64-bit store
            }
            bs += static_cast<uint32_t>(__popc(ms[k]));
            bb += static_cast<uint32_t>(__popc(mb[k]));
        }
    }
    if (part != 0)
    {
        return;
    }
    // The window's boundary keys (as many as the histogram counted), each one 64-bit store (key != 0): polled.
    uint32_t const nBin = win.z - cb;
    if (nBin > static_cast<uint32_t>(kSelWin))
    {
        fail(ctx, 180, static_cast<int>(cb), static_cast<int>(nBin));
    }
    subStamp(ctx, 62);
    uint32_t const sWinK = sSel + 64;
    uint32_t const sWinP = sWinK + 4 * kSelWin;
    if (tid < static_cast<int>(nBin))
    {
        uint64_t const* const slot = reinterpret_cast<uint64_t const*>(gc + kCIdxParKey) + tid;
        uint64_t v = ldPollU64(slot);
        uint64_t t0 = 0;
        while (static_cast<uint32_t>(v) == 0u)
        {
            spinCheck(ctx, t0, 179, tid, 0);
            v = ldPollU64(slot);
        }
        sts32(sWinK + 4 * tid, static_cast<uint32_t>(v));
        sts32(sWinP + 4 * tid, static_cast<uint32_t>(v >> 32));
    }
    __syncthreads();
    if (warp == 0)
    {
        int const want = need0 - static_cast<int>(cb);
#pragma unroll 1
        for (int j = lane; j < static_cast<int>(nBin); j += 32)
        {
            uint32_t const v = lds32(sWinK + 4 * j);
            int r = 0;
#pragma unroll 1
            for (int x = 0; x < static_cast<int>(nBin); ++x)
            {
                uint32_t const u = lds32(sWinK + 4 * x);
                r += u > v || (u == v && x < j) ? 1 : 0;
            }
            if (r < want)
            {
                st32(out + cb + r, lds32(sWinP + 4 * j));
            }
        }
    }
#pragma unroll 1
    for (int i = need0 + tid; i < kTopK; i += kThreads)
    {
        st32(out + i, ~0u);
    }
    subStamp(ctx, 63);
}

// The selector when a list it takes holds more than kCandSmemList candidates (more than the shared-memory selection
// copies: an estimate E far from the row's 2048th key, or keys spread unevenly over the CTAs): an exact selection over
// the nA + nB candidates in global memory (list A, then list B when taken; nB = 0 otherwise). The first pass polls
// every slot until its candidate landed (key 0: not yet) and counts the keys' bits 31-20 into a shared histogram at s0;
// two more passes count bits 19-8 and 7-0 of the keys in the bin found so far: X = the row's need0-th largest key. The
// keys > X and need0 - count(> X) of the keys == X (equal keys are interchangeable) are the row's selection, out in any
// order (the attention takes them as a set; every thread handles the same slots in every pass, so it zeroes its own for
// the next leader launch as it reads them last); entries past min(nc, need0) get -1 (up to outEnd). Without select (the
// row shard's odd rank) the lists are only polled and zeroed.
__device__ __noinline__ void idxSelectGlobal(Ctx const& ctx, uint64_t* listA, int nA, uint64_t* listB, int nB,
    int need0, bool select, uint8_t* s0, int* topkOut, int* arenaOut, int* mcOut, int outEnd = kTopK)
{
    int const tid = threadIdx.x;
    int const nc = nA + nB;
    uint32_t* const hist = reinterpret_cast<uint32_t*>(s0); // 4096 bins
    uint32_t* const scal = hist + 4096;
    auto at = [&](int i) { return i < nA ? listA + i : listB + (i - nA); };
    bool const all = nc <= need0; // every candidate is selected
    for (int i = tid; i < 4096; i += kThreads)
    {
        hist[i] = 0u;
    }
    if (tid == 0)
    {
        scal[10] = 0u;
        scal[11] = 0u;
    }
    __syncthreads();
    // Pass 1: every slot landed; bits 31-20.
#pragma unroll 1
    for (int base = tid; base < nc; base += 8 * kThreads)
    {
        uint64_t v[8];
#pragma unroll
        for (int u = 0; u < 8; ++u)
        {
            int const i = base + u * kThreads;
            v[u] = i < nc ? ldPollU64(at(i)) : 0ull;
        }
#pragma unroll
        for (int u = 0; u < 8; ++u)
        {
            int const i = base + u * kThreads;
            if (i < nc)
            {
                uint64_t t0 = 0;
                while (static_cast<uint32_t>(v[u]) == 0u)
                {
                    spinCheck(ctx, t0, 177, i, nc);
                    v[u] = ldPollU64(at(i));
                }
                if (select && !all)
                {
                    atomicAdd(&hist[static_cast<uint32_t>(v[u]) >> 20], 1u);
                }
            }
        }
    }
    __syncthreads();
    uint32_t X = 0;
    uint32_t gt = 0;
    if (select && !all)
    {
        uint32_t const need = static_cast<uint32_t>(need0);
        idxBinFromTop(hist, 4096, need, scal);
        uint32_t const b1 = scal[8];
        uint32_t const above1 = scal[9];
        for (int i = tid; i < 4096; i += kThreads)
        {
            hist[i] = 0u;
        }
        __syncthreads();
        // Pass 2: bits 19-8 of the keys in bin b1.
#pragma unroll 4
        for (int i = tid; i < nc; i += kThreads)
        {
            uint32_t const key = static_cast<uint32_t>(__ldcg(reinterpret_cast<unsigned long long const*>(at(i))));
            if ((key >> 20) == b1)
            {
                atomicAdd(&hist[(key >> 8) & 0xfffu], 1u);
            }
        }
        __syncthreads();
        idxBinFromTop(hist, 4096, need - above1, scal);
        uint32_t const b2 = scal[8];
        uint32_t const above2 = scal[9];
        uint32_t const hi24 = (b1 << 12) | b2;
        for (int i = tid; i < 256; i += kThreads)
        {
            hist[i] = 0u;
        }
        __syncthreads();
        // Pass 3: bits 7-0 of the keys in bin hi24.
#pragma unroll 4
        for (int i = tid; i < nc; i += kThreads)
        {
            uint32_t const key = static_cast<uint32_t>(__ldcg(reinterpret_cast<unsigned long long const*>(at(i))));
            if ((key >> 8) == hi24)
            {
                atomicAdd(&hist[key & 0xffu], 1u);
            }
        }
        __syncthreads();
        idxBinFromTop(hist, 256, need - above1 - above2, scal);
        X = (hi24 << 8) | scal[8];
        gt = above1 + above2 + scal[9];
    }
    // Output: the keys > X (slots from 0), then need0 - gt keys == X; every slot zeroed after its last read.
    uint32_t const ties = select && !all ? static_cast<uint32_t>(need0) - gt : 0u;
#pragma unroll 4
    for (int i = tid; i < nc; i += kThreads)
    {
        uint64_t* const p = at(i);
        uint64_t const v = __ldcg(reinterpret_cast<unsigned long long const*>(p));
        if (select)
        {
            uint32_t const key = static_cast<uint32_t>(v);
            int slot = -1;
            if (all || key > X)
            {
                slot = static_cast<int>(atomicAdd(&scal[10], 1u));
            }
            else if (key == X)
            {
                uint32_t const t = atomicAdd(&scal[11], 1u);
                slot = t < ties ? static_cast<int>(gt + t) : -1;
            }
            if (slot >= 0)
            {
                int const pos = static_cast<int>(v >> 32);
                if (mcOut != nullptr)
                {
                    mcOut[slot] = pos;
                }
                if (topkOut != nullptr)
                {
                    topkOut[slot] = pos;
                    arenaOut[slot] = pos;
                }
            }
        }
        stU64(p, 0ull);
    }
    if (select)
    {
        int const filled = min(nc, need0);
#pragma unroll 4
        for (int i = filled + tid; i < outEnd; i += kThreads)
        {
            if (mcOut != nullptr)
            {
                mcOut[i] = -1;
            }
            if (topkOut != nullptr)
            {
                topkOut[i] = -1;
                arenaOut[i] = -1;
            }
        }
    }
}

// The selector's last part: the row counted done, then list B, when it was not taken, zeroed slot by slot once each
// candidate landed (a zero stored before the candidate lands would leave a stale key).
__device__ __forceinline__ void idxTopKDone(Ctx const& ctx, uint32_t* gc, uint64_t* listB, int nB, bool useB, int row)
{
    int const tid = threadIdx.x;
    __syncthreads();
    subStamp(ctx, 47);
    if (tid == 0)
    {
        redReleaseAdd(gc + kCIdxDone, 1u);
    }
    if (!useB)
    {
        int const n = min(nB, kCandList);
#pragma unroll 1
        for (int i = tid; i < n; i += kThreads)
        {
            uint64_t t1 = 0;
            while (static_cast<uint32_t>(ldPollU64(listB + i)) == 0u)
            {
                spinCheck(ctx, t1, 174, row, i);
            }
            stU64(listB + i, 0ull);
        }
    }
}

// The top-2048 of every row from the logits in the CTA's tiles (stageIdxSel's last part): T(t), T2(t) and the CTA's
// candidates (idxBoundCands: one 64-bit (key, position) store each into zeroed lists A and B, so key 0 = not written
// yet). CTA kIdxSelBase + t, once every CTA has made its candidates' stores (a written count, released), copies the
// row's list A, and list B too when A holds fewer than need0 keys (A, a zero pair when A's count is odd, then B), into
// shared memory at a0 (bulk copies on barK, phase parity barParity), polls the slots it read as zero (not landed yet)
// from global memory, zeroes them for the next leader launch, packs the keys and positions for the selection, picks
// the row's 2048 (slots past the row's valid keys get -1; the search's first probe is E(t)), counts the row done and
// then zeroes list B when it did not take it (polling each slot first: a zero stored before the candidate lands would
// leave a stale key). (Copying as soon as every CTA held its slots, before their stores, left 2-5 us of polling.)
__device__ __noinline__ void idxTopK(Ctx const& ctx, uint32_t a0, uint8_t* s0, int nTiles, int L, uint64_t* cand,
    int* topkBase, uint32_t barK, uint32_t barParity)
{
    int const tid = threadIdx.x;
    int const cta = blockIdx.x;
    uint32_t* gc = ctx.counters;
    uint32_t const sScr = ctx.smemBase + kSmemRed;
    idxBoundCands(ctx, a0, nTiles, cta, gc, cand, sScr, arena<uint32_t>(ctx, kArIdxBnd), L);
    // The CTA's candidate stores, then the written count as a release (after the join: it covers every thread's
    // stores), which the selector's acquire pairs with before it bulk-copies the lists: a relaxed count (~1 us less)
    // left the copy racing the stores, and a torn record (new key, old position) would pass its key-0 re-poll.
    __syncthreads();
    if (tid == 0)
    {
        redReleaseAdd(gc + kCIdxWritten, 1u);
    }
    subStamp(ctx, 45);
    // The row shard: CTA kIdxSelBase selects the rank's row (the top-k exchange takes the even rank's to every rank).
    bool const rs = (GLM52_FLAGS(ctx) & kFlagIdxAllRows) == 0;
    bool const parallel = rs && (ctx.rank & 1) == 0 && (GLM52_FLAGS(ctx) & kFlagIdxParallel);
    if (rs ? cta < kIdxSelBase || (cta != kIdxSelBase && !parallel) : cta < kIdxSelBase)
    {
        return;
    }
    int const row = rs ? ctx.rank >> 1 : cta - kIdxSelBase;
    int const need0 = min(kTopK, L - kTok + row + 1);
    // The row into the top-k buffer and into this launch's arena (cleared to kEmpty: the leader's attention polls those
    // words). The row shard: into the top-k exchange instead, from the even rank of the pair alone (two selections of
    // a row list it in different orders); the attention copies every row from there into the top-k buffer. The odd
    // rank only clears its lists.
    int* topkOut = rs ? nullptr : topkBase + row * kTopK;
    int* arenaOut = rs ? nullptr : arena<int>(ctx, kArIdxTopk) + row * kTopK;
    int* mcOut = rs && (ctx.rank & 1) == 0
        ? reinterpret_cast<int*>(ptr<uint8_t>(ctx, kPTkMc) + static_cast<size_t>(ctx.parity) * kTkBytes) + row * kTopK
        : nullptr;
    bool const select = topkOut != nullptr || mcOut != nullptr;
    uint64_t* const listA = cand + static_cast<size_t>(row) * kCandCap;
    uint64_t* const listB = listA + kCandList;
    constexpr bool kSelFast = (kSpecializedStaticFlags & kFlagSelFast) != 0;
    if (parallel && cta != kIdxSelBase && !kSelFast)
    {
        idxSelectBinParallel(ctx, need0, listA, mcOut);
        return;
    }
    // As idxBoundCands took them; E(row) from the arena (published with them).
    uint32_t const T = lds32(sScr + 768 + 4 * row);
    uint32_t const T2 = lds32(sScr + 896 + 4 * row);
    bool const hist = select && !(GLM52_FLAGS(ctx) & kFlagTopkNoHist);
    uint32_t const sWin = sScr + 848;
#if GLM52_TOPK_UNION
    // The written count, the counts and the window in one poll (warp 1): every CTA has issued its candidates' stores,
    // the slot atomics counted every CTA (bits 24-31: the counts are final), and the bins' sum equals the binned count
    // that rode on those atomics. (One after the other they took three round trips. Starting before the written count
    // left 2-5 us of polling for slots not landed yet.)
    uint32_t const hsh = lds32(sScr + 864 + 4 * row);
    uint32_t const hspan = static_cast<uint32_t>(kHistBins) << hsh;
    if ((tid >> 5) == 1)
    {
        int const lane = tid & 31;
        uint32_t const* const hs = gc + kCIdxHist + row * kHistBins;
        uint32_t wr = ldAcquire(gc + kCIdxWritten);
        uint64_t nn = ldPollU64(gc + kCIdxCand + 2 * row);
        uint32_t h0 = hist ? ldPoll32(hs + 2 * lane) : 0u;
        uint32_t h1 = hist ? ldPoll32(hs + 2 * lane + 1) : 0u;
        uint64_t t1 = 0;
        while (true)
        {
            bool const all = wr >= static_cast<uint32_t>(kCtas)
                && ((static_cast<uint32_t>(nn) >> 24) & 0xffu) == static_cast<uint32_t>(kCtas);
            uint32_t const sum = __reduce_add_sync(0xffffffffu, h0 + h1);
            if (warpAll(all && (!hist || sum == static_cast<uint32_t>(nn >> 50))))
            {
                break;
            }
            spinCheck(ctx, t1, 168, row, lane);
            wr = ldAcquire(gc + kCIdxWritten);
            nn = ldPollU64(gc + kCIdxCand + 2 * row);
            if (hist)
            {
                h0 = ldPoll32(hs + 2 * lane);
                h1 = ldPoll32(hs + 2 * lane + 1);
            }
        }
        uint32_t const cA = static_cast<uint32_t>(nn) & 0xffffffu;
        uint32_t const binned = static_cast<uint32_t>(nn >> 50);
        if (lane == 0)
        {
            sts32(sScr + 832, cA);
            sts32(sScr + 836, static_cast<uint32_t>(nn >> 32) & 0x3ffffu);
            sts32(sScr + 844, binned); // list A's keys in the histogram
        }
        if (hist)
        {
            // The window: the bin holding the row's need0-th key (count(>= its lower edge) >= need0 > count(> its upper
            // edge)), or the keys above the bins when they are that many; none when list A is short (list B taken).
            uint4 w = make_uint4(0u, 0u, 0u, 0u);
            if (cA >= static_cast<uint32_t>(need0))
            {
                uint32_t const above = cA - binned;
                uint32_t suf = h0 + h1;
#pragma unroll
                for (int off = 1; off < 32; off <<= 1)
                {
                    uint32_t const x = __shfl_down_sync(0xffffffffu, suf, off);
                    suf += lane + off < 32 ? x : 0u;
                }
                uint32_t const s0 = above + suf;
                uint32_t const s1 = s0 - h0;
                uint32_t const need = static_cast<uint32_t>(need0);
                unsigned const hit0 = __ballot_sync(0xffffffffu, s0 >= need);
                unsigned const hit1 = __ballot_sync(0xffffffffu, s1 >= need);
                if (above >= need)
                {
                    w = make_uint4(T2 + hspan, 0u, above, 0u); // up to the largest key
                }
                else
                {
                    int const c = max(2 * (31 - __clz(hit0)), hit1 != 0 ? 2 * (31 - __clz(hit1)) + 1 : 0);
                    uint32_t const sc = __shfl_sync(0xffffffffu, (c & 1) ? s1 : s0, c >> 1);
                    uint32_t const hc = __shfl_sync(0xffffffffu, (c & 1) ? h1 : h0, c >> 1);
                    uint32_t const edge = T2 + (static_cast<uint32_t>(c) << hsh);
                    w = make_uint4(edge, edge + (1u << hsh) - 1u, sc, sc - hc);
                }
            }
            if (lane == 0)
            {
                sts128(sWin, w);
            }
        }
    }
    if (tid == 0)
    {
        sts32(sScr + 840, lds32(sScr + 800 + 4 * row));
        st32(gc + kCIdxT + row, ~T);
    }
#else
    if (tid == 0)
    {
        // E's load first (published with T: in by now), its round trip under the written count's wait.
        // E: from the T CTA's word, or as this CTA took it with T (never kEmpty then).
        uint8_t const* const eWord = arena<uint8_t const>(ctx, kArIdxBnd) + (kCtas * kTok + 2 * kTok + row) * 4;
        uint32_t const e = (GLM52_FLAGS(ctx) & kFlagTopkTCta) ? ldPoll32(eWord) : lds32(sScr + 800 + 4 * row);
        uint64_t t0 = 0;
        while (ldAcquire(gc + kCIdxWritten) < static_cast<uint32_t>(kCtas))
        {
            spinCheck(ctx, t0, 168, row, 0);
        }
        // The slot atomics are done in L2 before the counted stores: one 64-bit read of both lists' counts.
        uint64_t const nn = ldPollU64(gc + kCIdxCand + 2 * row);
        sts32(sScr + 832, static_cast<uint32_t>(nn) & 0xffffffu);
        sts32(sScr + 836, static_cast<uint32_t>(nn >> 32) & 0x3ffffu);
        sts32(sScr + 844, static_cast<uint32_t>(nn >> 50)); // list A's keys in the histogram
        sts32(sScr + 840, e != kEmpty ? e : poll1(ctx, eWord, 175));
        st32(gc + kCIdxT + row, ~T);
    }
#endif
    // Every thread's shared-memory accesses before the bulk copy rewrites the region (the CTA's own logits), and its
    // own candidates' stores before the async-proxy read.
    fenceProxyAsyncSmem();
    __syncthreads();
    subStamp(ctx, 46);
    int const nA = static_cast<int>(lds32(sScr + 832));
    int const nB = static_cast<int>(lds32(sScr + 836));
    uint32_t const E = lds32(sScr + 840);
    bool const useB = nA < need0;
    int const pad = useB ? (nA & 1) : 0;
    int const nc = useB ? nA + pad + nB : nA;
    uint32_t const lo = useB ? T : T2;
    if (nA > kCandList || (useB && nB > kCandList))
    {
        fail(ctx, 169, row, nA + (useB ? nB << 16 : 0));
    }
    if (nA > kCandSmemList || (useB && nB > kCandSmemList))
    {
        // More candidates than the shared-memory selection copies: the exact selection in global memory, by the
        // original selector alone (the helpers took the same counts and leave).
        if (parallel && cta != kIdxSelBase)
        {
            return;
        }
        if (parallel && !kSelFast && tid == 0)
        {
            redReleaseAdd(gc + kCIdxParState, 1u);
        }
        idxSelectGlobal(ctx, listA, nA, listB, useB ? nB : 0, need0, select, s0, topkOut, arenaOut, mcOut);
        idxTopKDone(ctx, gc, listB, nB, useB, row);
        return;
    }
    auto issueCopy = [&]()
    {
        if (tid == 0)
        {
            asm volatile("fence.proxy.async.global;" ::: "memory");
            uint32_t const bytesA = static_cast<uint32_t>(useB ? nA + pad : (nA + 1) & ~1) * 8u;
            uint32_t const bytesB = useB ? static_cast<uint32_t>((nB + 1) & ~1) * 8u : 0u;
            mbarExpectTx(barK, bytesA + bytesB);
            if (bytesA != 0)
            {
                bulkCopy(a0, listA, bytesA, barK);
            }
            if (bytesB != 0)
            {
                bulkCopy(a0 + bytesA, listB, bytesB, barK);
            }
        }
    };
    // With the histogram and list A alone, no copy unless the window is not a small bin: the bin pass reads list A
    // itself (the copy took ~2 us after the written count, mk-v134a L10).
    bool const direct = select && !(GLM52_FLAGS(ctx) & kFlagTopkNoHist) && !useB && !(GLM52_FLAGS(ctx) & kFlagBinCopy);
    if (parallel && !direct && tid == 0 && !kSelFast)
    {
        redReleaseAdd(gc + kCIdxParState, 1u); // helpers leave without touching the original selector's lists
    }
    if (kSelFast && parallel && !direct && cta != kIdxSelBase)
    {
        return; // the original selector owns every candidate and output; the helpers took the same decision
    }
    if (!direct)
    {
        issueCopy();
    }
    // The direct pass's first four pairs per lane go out now, in flight while warp 1 takes the window.
    uint2 pre[4];
#pragma unroll
    for (int k = 0; k < 4; ++k)
    {
        int const i = 128 * (tid >> 5) + (tid & 31) + 32 * k;
        pre[k] = direct && !parallel && i < nA ? __ldcg(reinterpret_cast<uint2 const*>(listA + i)) : make_uint2(0u, 0u);
    }
    // While the copy lands (warp 1): the search's first window from list A's histogram, once the bins' sum equals the
    // count that rode on the slot atomics (every CTA's adds landed): the bin holding the row's need0-th key (count(>=
    // its lower edge) >= need0 > count(> its upper edge)), or the keys above the bins when they are that many.
    if (!GLM52_TOPK_UNION && hist && (tid >> 5) == 1)
    {
        int const lane = tid & 31;
        uint4 w = make_uint4(0u, 0u, 0u, 0u); // ca = 0: no window (the search starts from the whole list)
        if (!useB)
        {
            uint32_t const binned = lds32(sScr + 844);
            uint32_t const* const hs = gc + kCIdxHist + row * kHistBins;
            uint32_t h0 = ldPoll32(hs + 2 * lane);
            uint32_t h1 = ldPoll32(hs + 2 * lane + 1);
            uint64_t t1 = 0;
            while (__reduce_add_sync(0xffffffffu, h0 + h1) != binned)
            {
                spinCheck(ctx, t1, 176, lane, static_cast<int>(binned));
                h0 = ldPoll32(hs + 2 * lane);
                h1 = ldPoll32(hs + 2 * lane + 1);
            }
            // List A's keys >= the lower edge of bin 2 lane (s0) and of bin 2 lane + 1 (s1).
            uint32_t const above = static_cast<uint32_t>(nA) - binned;
            uint32_t suf = h0 + h1;
#pragma unroll
            for (int off = 1; off < 32; off <<= 1)
            {
                uint32_t const v = __shfl_down_sync(0xffffffffu, suf, off);
                suf += lane + off < 32 ? v : 0u;
            }
            uint32_t const s0 = above + suf;
            uint32_t const s1 = s0 - h0;
            uint32_t const need = static_cast<uint32_t>(need0);
            unsigned const hit0 = __ballot_sync(0xffffffffu, s0 >= need);
            unsigned const hit1 = __ballot_sync(0xffffffffu, s1 >= need);
            if (above >= need)
            {
                w = make_uint4(T2 + kHistSpan, 0u, above, 0u); // up to the largest key
            }
            else
            {
                // The largest such bin (lane 0's s0 = nA >= need0).
                int const c = max(2 * (31 - __clz(hit0)), hit1 != 0 ? 2 * (31 - __clz(hit1)) + 1 : 0);
                uint32_t const sc = __shfl_sync(0xffffffffu, (c & 1) ? s1 : s0, c >> 1);
                uint32_t const hc = __shfl_sync(0xffffffffu, (c & 1) ? h1 : h0, c >> 1);
                uint32_t const edge = T2 + (static_cast<uint32_t>(c) << kHistShift);
                w = make_uint4(edge, edge + (1u << kHistShift) - 1u, sc, sc - hc);
            }
        }
        if (lane == 0)
        {
            sts128(sWin, w);
        }
    }
    bool binDirect = false;
    if (direct)
    {
        __syncthreads(); // warp 1's window to every thread
        uint4 const w = lds128(sWin);
        binDirect = w.y != 0 && w.z - w.w <= static_cast<uint32_t>(kSelWin) && !(GLM52_FLAGS(ctx) & kFlagTopkNoBin);
        if (binDirect)
        {
            subStamp(ctx, 79);
            if (parallel && kSelFast)
            {
                idxSelectBinParallelFast(ctx, need0, listA, mcOut, nc, w);
                if (cta != kIdxSelBase)
                {
                    return;
                }
            }
            else if (parallel)
            {
                if (tid == 0)
                {
                    st32(gc + kCIdxParN, static_cast<uint32_t>(nc));
                    st128(gc + kCIdxParWin, w);
                    redReleaseAdd(gc + kCIdxParState, 2u);
                }
                idxSelectBinParallel(ctx, need0, listA, mcOut);
            }
            else
            {
                idxSelectBin(ctx, nc, need0, w, a0, sScr + 1024, rs ? mcOut : topkOut, rs ? nullptr : arenaOut, listA,
                    true, (GLM52_FLAGS(ctx) & kFlagBinNoPreload) ? nullptr : pre);
            }
        }
        else
        {
            if (kSelFast && parallel && cta != kIdxSelBase)
            {
                return;
            }
            if (parallel && tid == 0 && !kSelFast)
            {
                redReleaseAdd(gc + kCIdxParState, 1u);
            }
            issueCopy();
        }
    }
    if (!binDirect)
    {
        waitBar(ctx, barK, barParity, 171);
        subStamp(ctx, 79);
        if (hist)
        {
            __syncthreads(); // warp 1's window to every thread
        }
        uint4 const win = hist ? lds128(sWin) : make_uint4(0u, 0u, 0u, 0u);
        bool const bin
            = win.y != 0 && win.z - win.w <= static_cast<uint32_t>(kSelWin) && !(GLM52_FLAGS(ctx) & kFlagTopkNoBin);
        if (bin)
        {
            // A bin window (list A only): one pass, into the exchange (the row shard) or the top-k buffer and the
            // arena; it polls the slots the copy read before they landed, then clears the list.
            idxSelectBin(ctx, nc, need0, win, a0, sScr + 1024, rs ? mcOut : topkOut, rs ? nullptr : arenaOut, listA,
                false, nullptr);
        }
        else
        {
            // Slots the copy read before their candidate landed (key 0; thread tid's slots tid + 256 k, k < 64, the pad
            // excepted): polled from global memory, up to 8 loads in flight per round trip (one at a time took ~2 us
            // more); per lane, the barrier below rejoins the warps.
            auto src = [&](int i) { return i < nA ? listA + i : listB + (i - nA - pad); };
            uint64_t zero = 0;
#pragma unroll 1
            for (int k = 0; k < 64 && tid + kThreads * k < nc; ++k)
            {
                int const i = tid + kThreads * k;
                zero |= lds32(a0 + 8 * i) == 0u && !(pad && i == nA) ? 1ull << k : 0ull;
            }
            uint64_t t0 = 0;
#pragma unroll 1
            while (zero != 0)
            {
                uint64_t v[8];
                int kk[8];
#pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    uint64_t const rest = j == 0 ? zero : zero & ~((2ull << kk[j - 1]) - 1);
                    kk[j] = rest != 0 ? __ffsll(static_cast<long long>(rest)) - 1 : -1;
                    v[j] = kk[j] >= 0 ? ldPollU64(src(tid + kThreads * kk[j])) : 0ull;
                }
#pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    if (kk[j] >= 0 && static_cast<uint32_t>(v[j]) != 0u)
                    {
                        sts64(a0 + 8 * (tid + kThreads * kk[j]),
                            make_uint2(static_cast<uint32_t>(v[j]), static_cast<uint32_t>(v[j] >> 32)));
                        zero &= ~(1ull << kk[j]);
                    }
                }
                if (zero != 0)
                {
                    spinCheck(ctx, t0, 173, row, tid);
                }
            }
            __syncthreads();
            // The lists taken are in shared memory now: back to zero for the next leader launch, the stores draining
            // while the selection computes (zeroed after it, they delayed this CTA's next units by ~1.2 us; at every
            // CTA's exit, the next launch by ~2.7 us; after the selection in mk-v98a: 0.7 us slower per slice).
#pragma unroll 1
            for (int i = 2 * tid; i < nA; i += 2 * kThreads)
            {
                st128(listA + i, make_uint4(0u, 0u, 0u, 0u));
            }
            if (useB)
            {
#pragma unroll 1
                for (int i = 2 * tid; i < nB; i += 2 * kThreads)
                {
                    st128(listB + i, make_uint4(0u, 0u, 0u, 0u));
                }
            }
            if (select && nc <= 32 * kThreads)
            {
                // Keys and positions packed four per 16 bytes (keys past nc zero) after the pairs.
                uint32_t const sKey = a0 + 32 * kThreads * 8;
                uint32_t const sPos = sKey + 32 * kThreads * 4;
#pragma unroll 1
                for (int q = tid; q < (nc + 3) >> 2; q += kThreads)
                {
                    int const i = 4 * q;
                    uint4 const p0 = lds128(a0 + 32 * q);
                    uint4 const p1 = lds128(a0 + 32 * q + 16);
                    sts128(sKey + 16 * q,
                        make_uint4(p0.x, i + 1 < nc ? p0.z : 0u, i + 2 < nc ? p1.x : 0u, i + 3 < nc ? p1.z : 0u));
                    sts128(sPos + 16 * q, make_uint4(p0.y, p0.w, p1.y, p1.w));
                }
                // Each thread reads back only the groups it packed.
                idxSelectRow(ctx, nc, need0, lo, E, win, topkOut, arenaOut, mcOut, sScr + 1024, sKey, sPos);
            }
            else if (select)
            {
                idxSelectLarge(ctx, nc, need0, lo, s0, a0, reinterpret_cast<uint32_t*>(ctx.smem + kSmemRed + 2048),
                    topkOut, arenaOut, mcOut);
            }
        }
    }
    idxTopKDone(ctx, gc, listB, nB, useB, row);
}

#if GLM52_TOPK_RSH
// ----------------------------------------------------------------------------------------------------------------
// RS-H: the row shard's exact top-2048 above kTopK keys from histograms of the row's keys (no bound T, candidate lists
// or selectors). Rank 2 t scores row t over all keys (the row shard) and publishes it:
//  - each CTA counts its valid keys per bin (rsBin) in shared memory in its tile epilogues;
//  - right after its logits and one scan it adds its counts of the bins from the one holding its kRsLocalK-th largest
//    key up (B) to the row's global histogram H (one relaxed red.add per nonzero bin; H: 6144 u32 in the union words'
//    place past list A, zero when unused) and each warp publishes its count added (and B if in its bins) in its word
//    of Pw (past H); then it sorts its keys of the bins >= B by bin (descending) in shared memory;
//  - the row CTA (kIdxSelBase) polls H and Pw until every word of the CTAs with keys is in and H's bins sum to their
//    counts (relaxed adds only raise the sum: every add landed; no fence), and takes the cut bin b* (the largest bin
//    with count(>= b*) >= need), the count above it and its own: exact when b* >= every CTA's B (H holds every key of
//    those bins); else (kRsMore) every CTA adds its other keys too and the row CTA takes the cut on the whole
//    histogram. It publishes a descriptor D (the counters' kCIdxParWin; after kRsMore, kCIdxParKey; nonzero words);
//  - each CTA reads its counts above b* and in b* from its sorted bins (two shared loads), takes its slots with one
//    64-bit atomic (the row's kCIdxCand pair: above | in-bin << 32), stores the positions of its keys above b*
//    straight into the multicast top-k row, and its keys in b* there too when b* is taken whole or holds equal keys
//    (+0.0 / -0.0), else as (key, position) pairs into list A;
//  - the row CTA ranks the pairs (larger key first, then the smaller position: torch.topk's choice among equal keys)
//    and stores the first need - C_above at [C_above, need); more than kRsPairs pairs (a cut in a tail bin, heavy ties)
//    go through the exact global selection. [need, kTopK) gets -1. It zeroes H and Pw.
// Every slot of [0, need) is written once (the bins partition the keys by order key, so the slot counts match). The
// code between a CTA's logits and its stores runs once per launch from a cold instruction cache (~0.4 us per KiB):
// the work before the adds and after D is kept to a few rolled loops.
// ----------------------------------------------------------------------------------------------------------------
static_assert(
    (kCandCap - kCandList) * 2 >= kRsBins + kCtas * kWarps, "RS-H's histogram and words in the union's place");
constexpr int kRsLh = kSelQ8 + 32 * 128 + 512; // the CTA's per-bin counts (u32 [kRsBins]) past the row's Q and blocks
static_assert(4 * kIdxMaxPagesRs <= 512 && kRsLh + 4 * kRsBins <= kSelKs, "RS-H's counts in the Q region");
constexpr int kRsMaxKeys = 128 * (kIdxMaxPagesRs / 2); // a CTA's local keys
constexpr int kRsKey = 0;                              // the CTA's sorted keys (u32), in the K region (dead after the
constexpr int kRsPos = kRsKey + 4 * kRsMaxKeys;        // logits), then their positions
static_assert(kRsPos + 4 * kRsMaxKeys <= kSelQ8, "RS-H's sorted keys in the K region");

// The bin of order key key (!= 0), without branches (it runs once per key in the tile epilogues): the offset into the
// sign's window clamped to [-1, kRsW] (-1: below it, kRsW: above), from bin 1 (negative) or kRsW + 5 (positive); then
// -0.0 and +0.0.
__device__ __forceinline__ uint32_t rsBin(uint32_t key)
{
    bool const pos = key >= 0x80000000u;
    int const d = min(max(static_cast<int>(key >> 16) - static_cast<int>(pos ? kRsP0 : kRsN0), -1), kRsW);
    uint32_t const b = static_cast<uint32_t>((pos ? kRsW + 5 : 1) + d);
    return key == 0x80000000u ? kRsBinPZ : key == 0x7fffffffu ? kRsBinNZ : b;
}

// The smallest order key of bin b (keys >= it lie in the bins >= b).
__device__ __forceinline__ uint32_t rsLo(uint32_t b)
{
    return b == 0u       ? 1u
        : b <= kRsW + 1u ? (kRsN0 + b - 1u) << 16
        : b == kRsBinNZ  ? 0x7fffffffu
        : b == kRsBinPZ  ? 0x80000000u
        : b == kRsW + 4u ? 0x80000001u
                         : (kRsP0 + b - kRsW - 5u) << 16;
}

// Bin b's word in the global histogram (thread b / kRsPer's polls hold it).
__device__ __forceinline__ uint32_t rsWord(uint32_t b)
{
    uint32_t const t = b / kRsPer;
    uint32_t const r = b - t * kRsPer;
    return 4u * (t + kThreads * (r >> 2)) + (r & 3u);
}

// The tile epilogue's count of one logit in the CTA's per-bin counts at shared address sLh (-inf: a key past the row's
// last, not counted).
__device__ __forceinline__ void rsCount(uint32_t sLh, float lgt)
{
    if (lgt != -INFINITY)
    {
        redsAddUnordered(sLh + 4u * rsBin(orderKey(lgt)), 1u);
    }
}

// After the CTA's logits (at a0 + kSelLog) and per-bin counts (at a0 + kRsLh; thread t's bins [kRsPer t, + kRsPer)):
// its counts of the bins >= B (the bin of its kRsLocalK-th largest key, 0 when it has fewer) into H after one scan
// (thread t adds at its own poll words), each warp's word Pw[cta][warp] = (its count added + 1) | (B, if in its bins)
// << 16; then per bin the CTA's keys in the bins above it (in place, rsSort's starts). Returns B. sScr: 64 bytes.
__device__ __noinline__ uint32_t rsFlush(Ctx const& ctx, uint32_t a0, uint32_t* H, uint32_t* Pw, uint32_t sScr)
{
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    uint32_t const sLh = a0 + kRsLh + 4u * kRsPer * static_cast<uint32_t>(tid);
    uint32_t s = 0;
#pragma unroll 1
    for (int j = 0; j < kRsPer / 4; ++j)
    {
        uint4 const q = lds128(sLh + 16 * j);
        s += (q.x + q.y) + (q.z + q.w);
    }
    // The keys in the bins above this thread's: its warp's higher lanes, then the warps above.
    uint32_t incl = s;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1)
    {
        uint32_t const x = __shfl_down_sync(0xffffffffu, incl, off);
        incl += lane + off < 32 ? x : 0u;
    }
    if (lane == 0)
    {
        sts32(sScr + 4 * warp, incl);
    }
    __syncthreads();
    uint32_t above = incl - s;
#pragma unroll 1
    for (int w = warp + 1; w < kWarps; ++w)
    {
        above += lds32(sScr + 4 * w);
    }
    // From this thread's top bin down: a bin is added while fewer than kRsLocalK keys lie above it.
    uint32_t acc = above;
    uint32_t added = 0;
    uint32_t bK = 0;
#pragma unroll 1
    for (int j = kRsPer / 4 - 1; j >= 0; --j)
    {
        uint4 const q = lds128(sLh + 16 * j);
        uint32_t const c[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int i = 3; i >= 0; --i)
        {
            if (c[i] != 0u && acc < kRsLocalK)
            {
                asm volatile(
                    "red.relaxed.gpu.global.add.u32 [%0], %1;" ::"l"(H + 4 * (tid + kThreads * j) + i), "r"(c[i]));
                added += c[i];
                bK = acc + c[i] >= kRsLocalK ? static_cast<uint32_t>(kRsPer * tid + 4 * j + i) : bK;
            }
            acc += c[i];
        }
    }
    added = __reduce_add_sync(0xffffffffu, added);
    bK = __reduce_max_sync(0xffffffffu, bK);
    if (lane == 0)
    {
        st32(Pw + kWarps * blockIdx.x + warp, (added + 1u) | (bK << 16));
        sts32(sScr + 32 + 4 * warp, bK);
    }
    subStamp(ctx, 65);
    // Per bin the CTA's keys in the bins above it, in place.
    acc = above;
#pragma unroll 1
    for (int j = kRsPer / 4 - 1; j >= 0; --j)
    {
        uint4 const q = lds128(sLh + 16 * j);
        uint4 st;
        st.w = acc;
        acc += q.w;
        st.z = acc;
        acc += q.z;
        st.y = acc;
        acc += q.y;
        st.x = acc;
        acc += q.x;
        sts128(sLh + 16 * j, st);
    }
    __syncthreads();
    uint32_t bC = 0;
#pragma unroll 1
    for (int w = 0; w < kWarps; ++w)
    {
        bC = max(bC, lds32(sScr + 32 + 4 * w));
    }
    return bC;
}

// The CTA's keys of the bins >= bC (or < bC with less) sorted by bin, descending, after rsFlush: keys at a0 + kRsKey,
// positions at a0 + kRsPos (each bin's start then becomes the end of its keys). With H, each key also goes into H
// (kRsMore: the keys rsFlush left out, one red.add each).
__device__ __noinline__ void rsSort(
    Ctx const& ctx, uint32_t a0, int nPer, int nPagesL, uint32_t bC, bool less, uint32_t* H)
{
    int const nKeys = 32 * nPer;
    uint32_t const lo = rsLo(bC); // the keys of the bins >= bC: one compare per key
    // The scan only records the thread's last eight selected keys (iteration i: key tid + 256 i); they are placed after
    // it (placing them inside put a divergent atomic and its stores into most iterations).
    uint32_t n = 0;
    uint64_t rec = 0;
#pragma unroll 1
    for (int i = 0; threadIdx.x + kThreads * i < nKeys; ++i)
    {
        uint32_t const v = selKeyAt(a0, -1, threadIdx.x + kThreads * i);
        bool const sel = less ? v != 0u && v < lo : v >= lo;
        n += sel ? 1u : 0u;
        rec = sel ? (rec << 8) | static_cast<uint64_t>(i) : rec;
    }
    auto place = [&](int k)
    {
        uint32_t const v = selKeyAt(a0, -1, k);
        if (less ? v != 0u && v < lo : v >= lo)
        {
            uint32_t const b = rsBin(v);
            if (H != nullptr)
            {
                redRelaxedAdd(H + rsWord(b), 1u);
            }
            uint32_t const idx = atomsAdd(a0 + kRsLh + 4 * b, 1u);
            sts32(a0 + kRsKey + 4 * idx, v);
            sts32(a0 + kRsPos + 4 * idx,
                static_cast<uint32_t>(idxPageOf(blockIdx.x, k >> 6, nPagesL) * kIdxPageKeys + (k & 63)));
        }
    };
    if (n <= 8u)
    {
#pragma unroll 1
        for (uint32_t j = 0; j < n; ++j)
        {
            place(static_cast<int>(threadIdx.x) + kThreads * static_cast<int>((rec >> (8 * j)) & 0xffu));
        }
    }
    else
    {
#pragma unroll 1
        for (int k = threadIdx.x; k < nKeys; k += kThreads)
        {
            place(k);
        }
    }
    __syncthreads();
}

// The row CTA: polls H (thread t: its bins [kRsPer t, + kRsPer), six 16-byte vectors) and, with Pw, the words of the
// CTAs holding keys (thread c: CTA c's eight) until every such word is in and H's bins sum to their counts (without
// Pw: to n), then the cut for need: (b*, the keys above b*, the keys in b*, the largest B of the CTAs). Every round
// runs the whole cut on what it read, so its code is warm when the last add lands. sScr: 272 bytes.
__device__ __noinline__ uint4 rsRowCut(
    Ctx const& ctx, uint32_t const* H, uint32_t const* Pw, int nPagesL, uint32_t n, uint32_t need, uint32_t sScr)
{
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    bool const mine = Pw != nullptr && tid < kCtas && idxPagesOf(tid, nPagesL) > 0;
    uint4 p0 = mine ? ldPoll128(Pw + kWarps * tid) : make_uint4(1u, 1u, 1u, 1u);
    uint4 p1 = mine ? ldPoll128(Pw + kWarps * tid + 4) : make_uint4(1u, 1u, 1u, 1u);
    uint4 v[kRsPer / 4];
#pragma unroll
    for (int j = 0; j < kRsPer / 4; ++j)
    {
        v[j] = ldPoll128(H + 4 * (tid + kThreads * j));
    }
    uint64_t t0 = 0;
    uint32_t par = 0;
#pragma unroll 1
    while (true)
    {
        uint32_t h[kRsPer];
#pragma unroll
        for (int j = 0; j < kRsPer / 4; ++j)
        {
            h[4 * j] = v[j].x;
            h[4 * j + 1] = v[j].y;
            h[4 * j + 2] = v[j].z;
            h[4 * j + 3] = v[j].w;
        }
        uint32_t s = 0;
#pragma unroll
        for (int i = 0; i < kRsPer; ++i)
        {
            s += h[i];
        }
        // Inclusive suffix sums over the warp's lanes (the higher lanes hold the larger bins), then the warps above;
        // the CTAs' words missing, their counts and their largest B.
        uint32_t incl = s;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1)
        {
            uint32_t const x = __shfl_down_sync(0xffffffffu, incl, off);
            incl += lane + off < 32 ? x : 0u;
        }
        uint32_t const pv[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
        bool have = true;
        uint32_t kc = 0;
        uint32_t bc = 0;
#pragma unroll
        for (int w = 0; w < kWarps; ++w)
        {
            have &= pv[w] != 0u;
            kc += (pv[w] & 0xffffu) - 1u;
            bc = max(bc, pv[w] >> 16);
        }
        uint32_t const missW = __reduce_add_sync(0xffffffffu, have ? 0u : 1u);
        uint32_t const kW = __reduce_add_sync(0xffffffffu, have ? kc : 0u);
        uint32_t const bW = __reduce_max_sync(0xffffffffu, have ? bc : 0u);
        if (lane == 0)
        {
            sts128(sScr + 128 * par + 16 * warp, make_uint4(incl, missW, kW, bW));
        }
        __syncthreads();
        uint32_t higher = 0;
        uint32_t total = 0;
        uint32_t miss = 0;
        uint32_t expect = 0;
        uint32_t bMax = 0;
#pragma unroll
        for (int w = 0; w < kWarps; ++w)
        {
            uint4 const x = lds128(sScr + 128 * par + 16 * w);
            higher += w > warp ? x.x : 0u;
            total += x.x;
            miss += x.y;
            expect += x.z;
            bMax = max(bMax, x.w);
        }
        expect = Pw != nullptr ? expect : n;
        // This thread's bins from its top: the one holding the need-th largest key, if any.
        uint32_t acc = higher + incl - s;
        uint32_t cut = 0xffffffffu;
        uint32_t above = 0;
        uint32_t inBin = 0;
#pragma unroll
        for (int i = kRsPer - 1; i >= 0; --i)
        {
            bool const hit = acc < need && acc + h[i] >= need;
            cut = hit ? static_cast<uint32_t>(kRsPer * tid + i) : cut;
            above = hit ? acc : above;
            inBin = hit ? h[i] : inBin;
            acc += h[i];
        }
        if (miss == 0u && total == expect)
        {
            if (cut != 0xffffffffu)
            {
                sts128(sScr + 256, make_uint4(cut, above, inBin, 0u));
            }
            __syncthreads();
            uint4 const c = lds128(sScr + 256);
            return make_uint4(c.x, c.y, c.z, bMax);
        }
        if (miss == 0u && total > expect)
        {
            fail(ctx, 191, static_cast<int>(total),
                static_cast<int>(expect)); // a stale histogram (not zeroed after use)
        }
        spinCheck(ctx, t0, 190, static_cast<int>(total), static_cast<int>(expect));
        if (!have)
        {
            p0 = ldPoll128(Pw + kWarps * tid);
            p1 = ldPoll128(Pw + kWarps * tid + 4);
        }
#pragma unroll
        for (int j = 0; j < kRsPer / 4; ++j)
        {
            v[j] = ldPoll128(H + 4 * (tid + kThreads * j));
        }
        par ^= 1u;
    }
}

// The CTA's emission for descriptor d (thread 0's; b* + 1; 1; C_above + 1 | need << 12; n_bin | mode << 24): its
// counts above b* and in b* from its sorted bins, one 64-bit slot atomic, then every thread stores its share: the
// keys in b* first (the row CTA ranks them), then the keys above. Returns false (no emission) for kRsMore.
// sScr: 32 bytes.
__device__ __noinline__ bool rsEmit(
    Ctx const& ctx, uint32_t a0, uint4 d, uint32_t sScr, int* tk, uint64_t* list, unsigned long long* slots)
{
    int const tid = threadIdx.x;
    if (tid == 0)
    {
        uint4 cnt = make_uint4(0u, 0u, 0u, 0u); // slot bases above / in b*, the keys above / in b*
        if ((d.w >> 24) != kRsMore)
        {
            cnt.z = lds32(a0 + kRsLh + 4 * d.x); // bin b* + 1's end: the keys above b*
            cnt.w = lds32(a0 + kRsLh + 4 * d.x - 4) - cnt.z;
            if (cnt.z + cnt.w != 0u)
            {
                unsigned long long const old = atomicAdd(slots, (static_cast<unsigned long long>(cnt.w) << 32) | cnt.z);
                cnt.x = static_cast<uint32_t>(old);
                cnt.y = static_cast<uint32_t>(old >> 32);
            }
        }
        sts128(sScr, cnt);
        sts128(sScr + 16, d);
    }
    __syncthreads();
    subStamp(ctx, 68);
    uint4 const cnt = lds128(sScr);
    uint4 const dd = lds128(sScr + 16);
    if ((dd.w >> 24) == kRsMore)
    {
        return false;
    }
    uint32_t const cAbove = (dd.z & 0xfffu) - 1u;
    uint32_t const want = (dd.z >> 12) - cAbove;
    bool const direct = (dd.w >> 24) == kRsDirect;
#pragma unroll 1
    for (uint32_t i = tid; i < cnt.w; i += kThreads)
    {
        uint32_t const x = cnt.z + i;
        uint32_t const pos = lds32(a0 + kRsPos + 4 * x);
        uint32_t const j = cnt.y + i;
        if (!direct)
        {
            stU64(list + j, (static_cast<uint64_t>(pos) << 32) | lds32(a0 + kRsKey + 4 * x));
        }
        else if (j < want)
        {
            st32(tk + cAbove + j, pos);
        }
    }
#pragma unroll 1
    for (uint32_t i = tid; i < cnt.z; i += kThreads)
    {
        st32(tk + cnt.x + i, lds32(a0 + kRsPos + 4 * i));
    }
    return true;
}

// The row CTA's part after its emission: the cut bin's pairs ranked (larger key first, then the smaller position) into
// [C_above, need) of tk, or (kRsGlobal) selected exactly in global memory; list's slots back to zero. sScr: 4 KiB.
__device__ __noinline__ void rsBoundary(Ctx const& ctx, uint4 d, uint64_t* list, int* tk, uint32_t sScr, uint8_t* s0)
{
    int const tid = threadIdx.x;
    uint32_t const cAbove = (d.z & 0xfffu) - 1u;
    uint32_t const want = (d.z >> 12) - cAbove;
    int const nBin = static_cast<int>(d.w & 0xffffffu);
    if ((d.w >> 24) == kRsGlobal)
    {
        // Its histogram at s0 is the sorted keys' place (a0 + kRsKey), zeroed before its first barrier: every
        // thread's rsEmit reads of them first (rsEmit returns without a join; d is the CTA's own).
        __syncthreads();
        idxSelectGlobal(ctx, list, nBin, nullptr, 0, static_cast<int>(want), true, s0, nullptr, nullptr, tk + cAbove,
            static_cast<int>(want));
        return;
    }
    // Pairs tid and tid + 256 (key != 0 once stored), polled together; the warp leaves together.
    uint64_t p[2];
#pragma unroll
    for (int u = 0; u < 2; ++u)
    {
        p[u] = tid + kThreads * u < nBin ? ldPollU64(list + tid + kThreads * u) : 1ull;
    }
    uint64_t t0 = 0;
    while (!warpAll(static_cast<uint32_t>(p[0]) != 0u && static_cast<uint32_t>(p[1]) != 0u))
    {
        spinCheck(ctx, t0, 193, tid, nBin);
#pragma unroll
        for (int u = 0; u < 2; ++u)
        {
            if (static_cast<uint32_t>(p[u]) == 0u)
            {
                p[u] = ldPollU64(list + tid + kThreads * u);
            }
        }
    }
    // Pair i as one 64-bit value, key << 32 | ~position: a larger key, or the same key and a smaller position, is the
    // larger value (torch.topk's choice among equal keys; the positions are distinct). Pairs [nBin, nPad) are 0, below
    // every pair (its key is nonzero).
#pragma unroll
    for (int u = 0; u < 2; ++u)
    {
        if (tid + kThreads * u < nBin)
        {
            sts64(sScr + 8 * (tid + kThreads * u),
                make_uint2(~static_cast<uint32_t>(p[u] >> 32), static_cast<uint32_t>(p[u])));
        }
    }
    int const nPad = (nBin + 7) & ~7;
    if (tid < nPad - nBin)
    {
        sts64(sScr + 8 * (nBin + tid), make_uint2(0u, 0u));
    }
    __syncthreads();
    subStamp(ctx, 62);
    // Pair i's rank (the pairs above it), by thread i % 256: eight pairs per step (four 16-byte loads in flight) into
    // four independent counts (one chain of dependent adds took ~9 cycles per pair). Warps without a pair skip it.
#pragma unroll 1
    for (int u = 0; u < 2; ++u)
    {
        int const i = tid + kThreads * u;
        if ((i & ~31) < nBin)
        {
            uint2 const mm = i < nBin ? lds64(sScr + 8 * i) : make_uint2(0u, 0u);
            uint64_t const m = static_cast<uint64_t>(mm.y) << 32 | mm.x;
            uint32_t r[4] = {0u, 0u, 0u, 0u};
#pragma unroll 1
            for (int x = 0; x < nPad; x += 8)
            {
#pragma unroll
                for (int k = 0; k < 4; ++k)
                {
                    // Two values as 64-bit words straight into register pairs (one 64-bit compare each, no moves).
                    uint64_t q0;
                    uint64_t q1;
                    asm volatile("ld.shared.v2.u64 {%0, %1}, [%2];" : "=l"(q0), "=l"(q1) : "r"(sScr + 8 * x + 16 * k));
                    r[k] += (q0 > m ? 1u : 0u) + (q1 > m ? 1u : 0u);
                }
            }
            uint32_t const rank = (r[0] + r[1]) + (r[2] + r[3]);
            if (i < nBin && rank < want)
            {
                st32(tk + cAbove + rank, ~mm.x);
            }
        }
    }
    subStamp(ctx, 63);
#pragma unroll
    for (int u = 0; u < 2; ++u)
    {
        if (tid + kThreads * u < nBin)
        {
            stU64(list + tid + kThreads * u, 0ull);
        }
    }
}

// The top-k part of stageIdxSel under RS-H, after the CTA's logits (its row array at a0 + kSelLog, its per-bin counts
// at a0 + kRsLh): the CTA adds its top bins to H and sorts its top keys, the row CTA takes the cut and publishes D,
// every CTA emits, the row CTA resolves the cut bin and zeroes H and Pw. L: the context; tr: the row; nPer: the CTA's
// local keys / 32; list: the row's list A (H and Pw past it); tk: the row of the multicast top-k exchange. Every thread
// of the even rank's CTAs with key pages (and of the row CTA) calls it.
__device__ __noinline__ void rsTopK(
    Ctx const& ctx, uint32_t a0, uint8_t* s0, int nPer, int L, int tr, uint64_t* list, int* tk)
{
    int const tid = threadIdx.x;
    uint32_t* const gc = ctx.counters;
    uint32_t const sScr = ctx.smemBase + kSmemRed;
    uint32_t* const H = reinterpret_cast<uint32_t*>(list + kCandList);
    uint32_t* const Pw = H + kRsBins;
    unsigned long long* const slots = reinterpret_cast<unsigned long long*>(gc + kCIdxCand + 2 * tr);
    int const nPagesL = (L + kIdxPageKeys - 1) / kIdxPageKeys;
    bool const row = blockIdx.x == kIdxSelBase;
    uint32_t const need = min(static_cast<uint32_t>(kTopK), static_cast<uint32_t>(L - kTok + tr + 1));
    uint32_t const bC = rsFlush(ctx, a0, H, Pw, sScr);
    if (!row)
    {
        rsSort(ctx, a0, nPer, nPagesL, bC, false, nullptr); // the row CTA sorts once D is out
    }
    subStamp(ctx, 64);
    uint4 d = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll 1
    for (int phase = 0; phase < 2; ++phase)
    {
        // Phase 0: D in kCIdxParWin; after kRsMore (every CTA added the rest of its keys) D in kCIdxParKey.
        uint32_t* const dAddr = gc + (phase == 0 ? kCIdxParWin : kCIdxParKey);
        if (row)
        {
            // Its scratch past rsFlush's B words (sScr + 32: slower warps may still read them, with no join after).
            uint4 const c = rsRowCut(
                ctx, H, phase == 0 ? Pw : nullptr, nPagesL, static_cast<uint32_t>(L - kTok + tr + 1), need, sScr + 64);
            subStamp(ctx, phase == 0 ? 132 : 134);
            if (tid == 0)
            {
                uint32_t const mode = c.x < c.w                               ? kRsMore
                    : need - c.y == c.z || c.x == kRsBinNZ || c.x == kRsBinPZ ? kRsDirect
                    : c.z <= kRsPairs                                         ? kRsRank
                                                                              : kRsGlobal;
                d = make_uint4(c.x + 1u, 1u, (c.y + 1u) | (need << 12), c.z | (mode << 24));
                st128(dAddr, d);
            }
            subStamp(ctx, 133);
            if (phase == 0)
            {
                rsSort(ctx, a0, nPer, nPagesL, bC, false, nullptr);
            }
        }
        else if (tid < 32)
        {
            d = ldPoll128(dAddr);
            uint64_t t0 = 0;
            while (!warpAll(d.x != 0u && d.y != 0u && d.z != 0u && d.w != 0u))
            {
                spinCheck(ctx, t0, 192, phase, 0);
                d = ldPoll128(dAddr);
            }
            subStamp(ctx, 44);
        }
        if (rsEmit(ctx, a0, d, sScr + 512, tk, list, slots))
        {
            break;
        }
        rsSort(ctx, a0, nPer, nPagesL, bC, true, H); // kRsMore: the CTA's other keys into H and its sorted keys
    }
    subStamp(ctx, 45);
    if (row)
    {
        d = lds128(sScr + 512 + 16);
#pragma unroll 1
        for (int i = static_cast<int>(need) + tid; i < kTopK; i += kThreads)
        {
            st32(tk + i, ~0u);
        }
        if ((d.w >> 24) != kRsDirect)
        {
            rsBoundary(ctx, d, list, tk, sScr + 1024, s0);
        }
        // H and Pw back to zero for the next leader launch of this copy (every add landed: its sum was the expected).
#pragma unroll
        for (int j = 0; j < kRsPer / 4; ++j)
        {
            st128(H + 4 * (tid + kThreads * j), make_uint4(0u, 0u, 0u, 0u));
        }
        for (int i = tid; i < kCtas * kWarps / 4; i += kThreads)
        {
            st128(Pw + 4 * i, make_uint4(0u, 0u, 0u, 0u));
        }
        subStamp(ctx, 47);
    }
}
#endif

// Leader: the indexer logits of this CTA's key pages (pages c, c + 212, ... of the request's ceil(L / 64)) for the 8
// rows, then the exact top-2048 of every row with all the other CTAs. Logit of key s for row t (s <= L - 8 + t):
// scale(s) * sum_h w(t, h) relu(Q8(t, h) . K8(s)), w = weights_proj x q_scale x 128^-0.5 x 32^-0.5 (stock numerics).
// Top-k: a lower bound T(t) of row t's 2048th-largest logit from every CTA's ten largest (each CTA polls the others'
// bounds), every CTA's keys >= T(t) into row t's candidate list, and CTA kIdxSelBase + t selects the row's 2048 from
// the list.
__device__ __noinline__ void stageIdxSel(Ctx const& ctx, uint32_t wOff, int const* kvLenPtr)
{
    int const tid = threadIdx.x;
    int const warp = tid >> 5;
    int const lane = tid & 31;
    int const cta = blockIdx.x;
    uint8_t* s0 = ctx.smem + wOff;
    uint32_t const a0 = ctx.smemBase + wOff;
    float* sKs = reinterpret_cast<float*>(s0 + kSelKs);
    float* sW = reinterpret_cast<float*>(s0 + kSelW);
    uint32_t const barK = ctx.smemBase + kSmemSmall + kSmallBars + kBarIdxK * 8;
    uint32_t const barS0 = barK + 8;
    uint32_t const barS1 = barK + 16;
    uint32_t const barK2 = ctx.smemBase + kSmemSmall + kSmallBars + kBarIdxK2 * 8;
    uint32_t* sTmem = reinterpret_cast<uint32_t*>(ctx.smem + kSmemSmall + kSmallScalars + 4 * 120);
    int const L = __ldg(kvLenPtr);
    if (idxAllKeys(L))
    {
        return;
    }
    int const nMine = idxSelMyPages(L);
    int const nPagesL = (L + kIdxPageKeys - 1) / kIdxPageKeys;
    // The row shard: this rank's row tr (ranks 2 tr and 2 tr + 1 both score it; the even one publishes its selection).
    bool const rs = (GLM52_FLAGS(ctx) & kFlagIdxAllRows) == 0;
    int const tr = ctx.rank >> 1;
#if GLM52_TOPK_RSH
    static_assert(
        (kSpecializedStaticFlags & (kFlagIdxAllRows | kFlagIdxTilePairs)) == 0, "RS-H: the row shard's rounds");
    // RS-H: the even rank's CTAs score, count and emit row tr; the odd rank's selection would be unused, and a CTA
    // without key pages (the IdxQ CTAs below ~37k keys) has nothing to score or emit (the row CTA still takes the cut).
    if ((ctx.rank & 1) != 0 || (nMine == 0 && cta != kIdxSelBase))
    {
        return;
    }
    // The CTA's per-bin counts start at zero (the epilogues add to them; pagesIn's barrier comes first).
#pragma unroll
    for (int j = 0; j < kRsPer / 4; ++j)
    {
        sts128(a0 + kRsLh + 16 * (tid + kThreads * j), make_uint4(0u, 0u, 0u, 0u));
    }
#endif
    // The row shard takes its pages in rounds of kIdxMaxPages (the logits into the row's array), the 8 rows in one.
    bool const rounds = rs && !(GLM52_FLAGS(ctx) & kFlagIdxTilePairs);
    if (nMine > (rounds ? kIdxMaxPagesRs : kIdxMaxPages))
    {
        fail(ctx, 160, nMine, L);
    }
    int const nTiles = (nMine + 1) / 2;
    uint64_t* cand = ptr<uint64_t>(ctx, kPIdxCand) + static_cast<size_t>(ctx.parity) * kTok * kCandCap;
    int* const topkBase = ptr<int>(ctx, kPTopkOut);
    // 1. The key pages and their scales (the first round's). (Issued by the CTA's IdxQ unit once its MMAs had read its
    // weights, they held up that unit's epilogue: warp 0 takes ~1-2 us to issue 22 requests; IdxQ ended 2 us later,
    // mk-v98a.)
    // The row shard: in batches of kSelBatchPages, batch k in slots [8 (k & 1), + 8) on kBarIdxK / kBarIdxK2, two in
    // flight (a batch's MMAs go out once its 8 pages are in, and batch k + 2 once batch k's MMAs are done: rounds of 16
    // pages, the next round's pages after the round's MMAs, took 3.4-3.6 us per round at 900k against ~2.7 at the HBM
    // cap, and the first 8 pages' epilogues now run while the next 8 load).
    // The row shard: the block indices of the CTA's later pages into a table (past Q's 32 rows), loaded while Q is on
    // its way: a batch's issue then reads them from shared memory (each issue's own block-index loads, one L2 round
    // trip, sat on warp 0's path between a batch's MMAs and the next batch's requests).
    int* const sBlk = reinterpret_cast<int*>(s0 + kSelQ8 + 32 * 128);
    static_assert(32 * 128 + 4 * kIdxMaxPagesRs <= 256 * 128, "IdxSel block table");
    if (rounds)
    {
        idxSelIssuePages(ctx, a0, 0, min(nMine, kSelBatchPages), nPagesL, 0, barK);
        idxSelIssuePages(ctx, a0, kSelBatchPages, min(max(nMine - kSelBatchPages, 0), kSelBatchPages), nPagesL,
            kSelBatchPages, barK2);
        for (int j = 2 * kSelBatchPages + tid; j < nMine; j += kThreads)
        {
            sBlk[j] = __ldg(ptr<int const>(ctx, kPIdxBlocks) + idxPageOf(cta, j, nPagesL));
        }
    }
    else
    {
        idxSelIssuePages(ctx, a0, 0, min(nMine, kIdxMaxPages), nPagesL);
    }
    // The logits epilogue once on a dummy tile while Q is on its way (its first real tile took ~2 us against ~0.7 for
    // the next ones): TMEM columns no MMA of this stage has written yet, the results into the Q8 region (rewritten
    // below after the barrier); the CTAs that ran an IdxQ unit did it in that unit's weight wait. (Dry runs of the
    // top-k steps cost 6-15 us here, more than their real runs: dropped.)
    {
        uint8_t* scratch = s0 + kSelQ8;
        if (cta >= kIdxQUnits && !rs)
        {
            tcFenceAfter();
            selEpilogue(*sTmem + 256u + (static_cast<uint32_t>(32 * (warp & 3)) << 16)
                    + 128u * static_cast<uint32_t>(warp >> 2),
                sW, reinterpret_cast<float*>(scratch), 32 * (warp & 3) + lane, warp >> 2, 0.f, 0, nullptr, 0);
        }
        __syncthreads();
    }
    // 2. Q in E4M3 (row R = 32 t + h of [256][128 B], SWIZZLE_128B) and the head weights: 16-byte vector c = tid + 256
    // i of Q (coalesced) to row c / 8, chunk (c % 8) ^ (row % 8); weight R = tid. The row shard: row tr's 32 heads only
    // (rows 0-31, one vector per thread; weights 0-31).
    if (rs)
    {
        // Row tr's BF16 Q (thread: head R = tid / 8, dims [16 (tid % 8), +16)) and the head's raw weight (thread 8 R);
        // the head's E4M3 scale from its 128 values (its 8 threads), then the E4M3 chunk and the weight, as IdxQ did.
        int const R = tid >> 3;
        uint8_t const* qsrc = arena<uint8_t const>(ctx, kArIdxQb) + (tr * 32 * kIdxDim + 16 * tid) * 2;
        uint8_t const* wsrc = arena<uint8_t const>(ctx, kArIdxW) + 4 * (32 * tr + R);
        bool const wThread = (tid & 7) == 0;
        uint4 v0 = ldPoll128(qsrc);
        uint4 v1 = ldPoll128(qsrc + 16);
        uint32_t wv = wThread ? ldPoll32(wsrc) : 0u;
        uint64_t t0 = 0;
        while (!warpAll((!wThread || wv != kEmpty) && full4(v0) && full4(v1)))
        {
            spinCheck(ctx, t0, 161, tid, 0);
            if (wThread && wv == kEmpty)
            {
                wv = ldPoll32(wsrc);
            }
            if (!full4(v0))
            {
                v0 = ldPoll128(qsrc);
            }
            if (!full4(v1))
            {
                v1 = ldPoll128(qsrc + 16);
            }
        }
        uint32_t const qw[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
        float m = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            m = fmaxf(m, fmaxf(fabsf(bf16Lo(qw[j])), fabsf(bf16Hi(qw[j]))));
        }
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 4));
        float const scale = ue8m0Scale(m);
        float const inv = rcpApproxFtz(scale);
        uint32_t q8[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            uint32_t const a = toE4m3x2(__fmul_rn(bf16Lo(qw[2 * j]), inv), __fmul_rn(bf16Hi(qw[2 * j]), inv));
            uint32_t const b = toE4m3x2(__fmul_rn(bf16Lo(qw[2 * j + 1]), inv), __fmul_rn(bf16Hi(qw[2 * j + 1]), inv));
            q8[j] = a | (b << 16);
        }
        *reinterpret_cast<uint4*>(s0 + kSelQ8 + R * 128 + 16 * ((tid & 7) ^ (R & 7)))
            = make_uint4(q8[0], q8[1], q8[2], q8[3]);
        if (wThread)
        {
            sW[R] = __fmul_rn(__fmul_rn(__uint_as_float(wv), scale), kIdxWeightScale);
        }
    }
    else
    {
        uint8_t const* qsrc = arena<uint8_t const>(ctx, kArIdxQ8);
        uint8_t const* wsrc = arena<uint8_t const>(ctx, kArIdxW) + 4 * tid;
        uint4 v[8];
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            v[i] = ldPoll128(qsrc + 16 * (tid + kThreads * i));
        }
        uint32_t wv = ldPoll32(wsrc);
        uint64_t t0 = 0;
        while (true)
        {
            bool ok = wv != kEmpty;
#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                ok &= full4(v[i]);
            }
            if (warpAll(ok))
            {
                break;
            }
            spinCheck(ctx, t0, 161, tid, 0);
            if (wv == kEmpty)
            {
                wv = ldPoll32(wsrc);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                if (!full4(v[i]))
                {
                    v[i] = ldPoll128(qsrc + 16 * (tid + kThreads * i));
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            int const c = tid + kThreads * i;
            int const R = c >> 3;
            *reinterpret_cast<uint4*>(s0 + kSelQ8 + R * 128 + 16 * ((c & 7) ^ (R & 7))) = v[i];
        }
        sW[tid] = __uint_as_float(wv);
    }
    stampInput(ctx);
    // 3. The CTA's pages [p0, p0 + np) in (slots [slot0, + np), on bar with parity ph); this step's keys among them
    // from the arena (the cache rows may not have been written when the pages were read).
    auto pagesIn = [&](int p0, int np, int slot0, uint32_t bar, uint32_t ph)
    {
        if (np > 0)
        {
            waitBar(ctx, bar, ph, 162);
        }
        if (tid < kTok * 8)
        {
            int const i = tid >> 3;
            int const ch = tid & 7;
            int const pf = L - kTok + i;
            int const page = pf / kIdxPageKeys;
            int const j = idxPageSlot(cta, page, nPagesL) - p0;
            if (j >= 0 && j < np)
            {
                int const r = pf % kIdxPageKeys;
                uint4 const v = poll4(ctx, arena<uint8_t const>(ctx, kArIdxKNew) + i * kIdxDim + 16 * ch, 163);
                *reinterpret_cast<uint4*>(s0 + kSelK + (slot0 + j) * 8192 + r * 128 + 16 * (ch ^ (r & 7))) = v;
                if (ch == 0)
                {
                    sKs[(slot0 + j) * kIdxPageKeys + r]
                        = __uint_as_float(poll1(ctx, arena<uint8_t const>(ctx, kArIdxKNewS) + 4 * i, 164));
                }
            }
        }
        fenceProxyAsyncSmem();
        tcFenceBefore();
        __syncthreads();
        tcFenceAfter();
    };
    if (!rounds)
    {
        pagesIn(0, min(nMine, kIdxMaxPages), 0, barK, 0u);
        subStamp(ctx, 48);
    }
    if (!GLM52_LEAN && ctx.p[kPIdxDebug2] != 0 && cta == 0 && !rs)
    {
        uint8_t* dbg = ptr<uint8_t>(ctx, kPIdxDebug2);
        for (int v = tid; v < (64 + 256) * 8; v += kThreads)
        {
            int const r = v >> 3;
            int const c = v & 7;
            uint8_t const* src = r < 64 ? s0 + kSelK + r * 128 : s0 + kSelQ8 + (r - 64) * 128;
            *reinterpret_cast<uint4*>(dbg + r * 128 + 16 * c)
                = *reinterpret_cast<uint4 const*>(src + 16 * (c ^ ((r < 64 ? r : r - 64) & 7)));
        }
    }
    // 4. Per tile of two pages (128 keys): D[key][32 t + h] = K8 Q8^T into TMEM buffer (tile & 1) (two tiles in
    // flight), then the epilogue: warp w takes keys 32 (w % 4) + lane and rows 4 (w / 4) .. + 3 (one 128-column TMEM
    // read); the tile's logits, FP32 [8][128] (masked: -inf), into its first 4 KiB (its MMAs are done).
    uint32_t const tmem = *sTmem;
    bool const dbg = !GLM52_LEAN && ctx.p[kPIdxDebug] != 0;
    bool const dbg2 = !GLM52_LEAN && ctx.p[kPIdxDebug2] != 0 && cta == 0 && !rs;
    int64_t const dbgLd = ctx.p[kPIdxDebugLd];
    if (rounds)
    {
        // The row shard: N = 32, so every tile of a batch has 32 TMEM columns of its own (tile t at 32 t, batch k's
        // tiles [4 (k & 1), + 4)): the batch's MMAs at once (one commit, kBarIdxK + 1 / + 2 by k & 1), then the
        // epilogues two tiles at a time, warps 0-3 the even tiles and warps 4-7 the odd ones, the logits into the row's
        // array (local key k at kSelLog + 4 k). (Two buffers of 256 columns, as for the 8 rows, took ~0.65 us per tile:
        // each tile's MMAs waited for the epilogue two tiles back, and warps 4-7 idled.)
        int const q = warp & 3;
        int const kt = 32 * q + lane;
        float* const sLog = reinterpret_cast<float*>(s0 + kSelLog);
#if GLM52_TOPK_RSH
        // RS-H: every valid key counted per bin in shared memory (rsTopK; zeroed at the stage's start).
        uint32_t const rsLh = a0 + kRsLh;
#else
        // The thread's two largest keys as order keys (selKeyAt's: -inf = 0), for the bound's union: local key 128 i +
        // kt is lane kt % 32 = lane's (idxUnionBound takes the two largest of the 8 warps' pairs per lane).
        uint32_t lm1 = 0;
        uint32_t lm2 = 0;
#endif
        int const nBatches = (nMine + kSelBatchPages - 1) / kSelBatchPages;
        // Batch k: its pages in (all threads), then its MMAs (warp 1, one commit on barS0 / barS1 by k & 1).
        auto batchIn = [&](int k)
        {
            int const p0 = kSelBatchPages * k;
            int const np = min(nMine - p0, kSelBatchPages);
            int const nt = (np + 1) / 2;
            int const sb = k & 1;
            int const t0 = kSelBatchPages / 2 * sb;
            pagesIn(p0, np, kSelBatchPages * sb, sb ? barK2 : barK, static_cast<uint32_t>((k >> 1) & 1));
            if (warp == 1)
            {
#if GLM52_SEL_BATCH_MMA
                selBatchMma(tmem + 32u * static_cast<uint32_t>(t0), a0 + kSelK + t0 * 16384, a0 + kSelQ8,
                    sb ? barS1 : barS0, nt);
#else
                for (int i = 0; i < nt; ++i)
                {
                    selTileMma(tmem + 32u * static_cast<uint32_t>(t0 + i), a0 + kSelK + (t0 + i) * 16384, a0 + kSelQ8,
                        sb ? barS1 : barS0, kIdescF8N32, i + 1 == nt);
                }
#endif
            }
        };
        if (nBatches > 0)
        {
            batchIn(0);
            subStamp(ctx, 48);
        }
        // Per batch k: its scales into registers, its MMAs done -> batch k + 2's pages into its slots, then batch
        // k + 1's pages and MMAs (they run during this batch's epilogues), then its epilogues (warps 0-3 the even
        // tiles, 4-7 the odd ones). (Batch k + 1's MMAs after batch k's epilogues left the tensor cores idle through
        // them.)
#pragma unroll 1
        for (int k = 0; k < nBatches; ++k)
        {
            int const p0 = kSelBatchPages * k;
            int const np = min(nMine - p0, kSelBatchPages);
            int const nt = (np + 1) / 2;
            int const sb = k & 1; // the batch's slots [8 sb, + 8), tiles [4 sb, + 4)
            int const t0 = kSelBatchPages / 2 * sb;
            float ks[2];
#pragma unroll
            for (int j = 0; j < 2; ++j)
            {
                int const i = (warp >> 2) + 2 * j;
                int const page = p0 + 2 * i + (kt >> 6);
                ks[j]
                    = i < nt && page < nMine ? sKs[(kSelBatchPages * sb + page - p0) * kIdxPageKeys + (kt & 63)] : 0.f;
            }
            waitBar(ctx, sb ? barS1 : barS0, static_cast<uint32_t>((k >> 1) & 1), 165);
            tcFenceAfter();
            if (k == 0)
            {
                subStamp(ctx, 49);
            }
            else if (k < 4)
            {
                subStamp(ctx, 47 + 3 * k); // 50, 53, 56: batch k's MMAs done
            }
            if (k + 2 < nBatches)
            {
                int const nf = p0 + 2 * kSelBatchPages;
                __syncthreads(); // every thread's scales of the batch read
                idxSelIssuePages(ctx, a0, nf, min(nMine - nf, kSelBatchPages), nPagesL, kSelBatchPages * sb,
                    sb ? barK2 : barK, sBlk);
            }
            if (k + 1 < nBatches)
            {
                batchIn(k + 1);
                if (k < 3)
                {
                    subStamp(ctx, 51 + 3 * k); // 51, 54, 57: batch k + 1's pages in, its MMAs issued
                }
            }
#pragma unroll
            for (int j = 0; j < 2; ++j)
            {
                int const i = (warp >> 2) + 2 * j;
                if (i < nt)
                {
                    int const page = p0 + 2 * i + (kt >> 6);
                    int const pos = idxPageOf(cta, page, nPagesL) * kIdxPageKeys + (kt & 63);
                    bool const inPage = page < nMine;
                    float const lgt = selEpilogue1(
                        tmem + 32u * static_cast<uint32_t>(t0 + i) + (static_cast<uint32_t>(32 * q) << 16), sW,
                        sLog + 64 * p0 + 128 * i, kt, ks[j], inPage && pos <= L - kTok + tr,
                        dbg && inPage && pos < L && pos < dbgLd ? ptr<float>(ctx, kPIdxDebug) + tr * dbgLd + pos
                                                                : nullptr);
#if GLM52_TOPK_RSH
                    rsCount(rsLh, lgt);
#else
                    top2Insert(lm1, lm2, lgt == -INFINITY ? 0u : orderKey(lgt));
#endif
                }
            }
            if (k < 3)
            {
                subStamp(ctx, 52 + 3 * k); // 52, 55, 58: batch k's epilogues done (thread 0's)
            }
        }
#if !GLM52_TOPK_RSH
        sts32(ctx.smemBase + kSmemRed + kSelLaneMax + 4 * (32 * warp + lane), lm1);
        sts32(ctx.smemBase + kSmemRed + kSelLaneMax2 + 4 * (32 * warp + lane), lm2);
#endif
        tcFenceBefore();
    }
    else
    {
        uint32_t phases = 0; // bit b: the parity buffer b's barrier waits for next
        auto issue = [&](int i)
        {
            if (warp == 1)
            {
                selTileMma(tmem + 256u * static_cast<uint32_t>(i & 1), a0 + kSelK + i * 16384, a0 + kSelQ8,
                    barS0 + 8 * (i & 1), rs ? kIdescF8N32 : kIdescF8);
            }
        };
        for (int i = 0; i < 2 && i < nTiles; ++i)
        {
            issue(i);
        }
        int const q = warp & 3;
        int const hh = warp >> 2;
        for (int i = 0; i < nTiles; ++i)
        {
            waitBar(ctx, barS0 + 8 * (i & 1), (phases >> (i & 1)) & 1u, 165);
            phases ^= 1u << (i & 1);
            tcFenceAfter();
            if (i < 6)
            {
                subStamp(ctx, 49 + 2 * i);
            }
            int const kt = 32 * q + lane;
            int const page = 2 * i + (kt >> 6);
            int const pos = idxPageOf(cta, page, nPagesL) * kIdxPageKeys + (kt & 63);
            bool const inPage = page < nMine;
            float const ks = inPage ? sKs[page * kIdxPageKeys + (kt & 63)] : 0.f;
            uint32_t const taddr
                = tmem + 256u * static_cast<uint32_t>(i & 1) + (static_cast<uint32_t>(32 * q) << 16) + 128u * hh;
            if (dbg2 && i == 0)
            {
                // Before the read below: no accumulator registers live across the call.
                dumpTileD(
                    taddr, reinterpret_cast<float*>(ptr<uint8_t>(ctx, kPIdxDebug2) + 320 * 128) + kt * 256 + 128 * hh);
            }
            // Every MMA of tile i is done: its K rows (read by nothing else now) take its logits.
            if (rs)
            {
                // Row tr only (N = 32): warps 0-3, the tile's logits into the row's array (kSelLog).
                if (warp < 4)
                {
                    selEpilogue1(tmem + 256u * static_cast<uint32_t>(i & 1) + (static_cast<uint32_t>(32 * q) << 16), sW,
                        reinterpret_cast<float*>(s0 + kSelLog) + 128 * i, kt, ks, inPage && pos <= L - kTok + tr,
                        dbg && inPage && pos < L && pos < dbgLd ? ptr<float>(ctx, kPIdxDebug) + tr * dbgLd + pos
                                                                : nullptr);
                }
            }
            else
            {
                selEpilogue(taddr, sW, reinterpret_cast<float*>(s0 + kSelK + i * 16384), kt, hh, ks,
                    inPage ? pos - (L - kTok) : kTok,
                    dbg && inPage && pos < L && pos < dbgLd ? ptr<float>(ctx, kPIdxDebug) + pos : nullptr, dbgLd);
            }
            // Every thread has read buffer (i & 1): it takes tile i + 2.
            tcFenceBefore();
            __syncthreads();
            if (i < 6)
            {
                subStamp(ctx, 50 + 2 * i);
            }
            if (i + 2 < nTiles)
            {
                issue(i + 2);
            }
        }
    }
    __syncthreads();
    subStamp(ctx, 43);
#if GLM52_TOPK_RSH
    // 5. RS-H: row tr's top-2048 from its histogram, into the top-k exchange (rank 2 tr publishes it to every rank).
    (void) topkBase;
    rsTopK(ctx, a0, s0, 4 * nTiles, L, tr, cand + static_cast<size_t>(tr) * kCandCap,
        reinterpret_cast<int*>(ptr<uint8_t>(ctx, kPTkMc) + static_cast<size_t>(ctx.parity) * kTkBytes) + tr * kTopK);
#else
    // 5. Top-2048 of every row (idxTopK); kBarIdxK's parity: its page loads so far (the row shard's even batches).
    int const nLoadsK = rounds ? ((nMine + kSelBatchPages - 1) / kSelBatchPages + 1) / 2 : (nMine > 0 ? 1 : 0);
    idxTopK(ctx, a0, s0, nTiles, L, cand, topkBase, barK, static_cast<uint32_t>(nLoadsK & 1));
#endif
}

// Position of the CTA's key t in the k order of the P.V MMA (production flash decode): within every 16 keys, k slots
// 4c .. 4c + 3 hold keys {2c, 2c + 1, 2c + 8, 2c + 9}.
__device__ __forceinline__ int permutedSlot(int t)
{
    int const u = t & 15;
    int const slot16 = (u < 8) ? ((u >> 1) * 4 + (u & 1)) : (((u - 8) >> 1) * 4 + 2 + (u & 1));
    return (t & ~15) + slot16;
}

// Sparse attention of row t over keys [128 b, +128) of its 2048: partial O, max and sum per head.
// kFlagSliceMerge: attention unit (t, blk), after its block, merges dims [32 blk, +32) of row t's 4 heads (64 threads:
// head tid / 16, two dims each) from the 16 blocks' partials (this unit is their only reader: it empties them) and
// stats (every unit of the row reads them: in the arena, which its copy's clear units empty), stageMerge's arithmetic.
// No merge stage: o_proj takes O from the 128 attention units, each consumer reads 8 KB instead of 32 KB, and nothing
// is cleared behind o_proj's stores.
__device__ __noinline__ void attnSliceMerge(Ctx const& ctx, int t, int blk)
{
    int const tid = threadIdx.x;
    if (tid >= kHeads * 16)
    {
        return;
    }
    int const h = tid >> 4;
    int const d = 32 * blk + 2 * (tid & 15);
    uint8_t const* const sts = arena<uint8_t const>(ctx, kArPStats) + (t * kAttnBlocks * kHeads + h) * 8;
    constexpr int kStatStride = kHeads * 8;
    uint8_t* const base = reinterpret_cast<uint8_t*>(
        ptr<float>(ctx, kPPartO) + (static_cast<size_t>(t * kAttnBlocks) * kHeads + h) * kDv + d);
    constexpr int kStride = kHeads * kDv * 4;
    uint2 o[kAttnBlocks];
    uint2 sb[kAttnBlocks];
#pragma unroll
    for (int b = 0; b < kAttnBlocks; ++b)
    {
        o[b] = ldPoll64(base + b * kStride);
        sb[b] = ldPoll64(sts + b * kStatStride);
    }
    uint64_t t0 = 0;
    while (true)
    {
        bool ok = true;
#pragma unroll
        for (int b = 0; b < kAttnBlocks; ++b)
        {
            ok &= full2(o[b]) & full2(sb[b]);
        }
        if (warpAll(ok))
        {
            break;
        }
        spinCheck(ctx, t0, 108, t, h);
#pragma unroll
        for (int b = 0; b < kAttnBlocks; ++b)
        {
            if (!full2(o[b]))
            {
                o[b] = ldPoll64(base + b * kStride);
            }
            if (!full2(sb[b]))
            {
                sb[b] = ldPoll64(sts + b * kStatStride);
            }
        }
    }
    float merged = -INFINITY;
#pragma unroll
    for (int b = 0; b < kAttnBlocks; ++b)
    {
        merged = fmaxf(merged, __uint_as_float(sb[b].x));
    }
    float mergedSum = 0.f;
    float acc0 = 0.f;
    float acc1 = 0.f;
#pragma unroll
    for (int b = 0; b < kAttnBlocks; ++b)
    {
        float const mb = __uint_as_float(sb[b].x);
        float const alpha = mb == -INFINITY ? 0.f : exp2f(mb - merged);
        mergedSum += alpha * __uint_as_float(sb[b].y);
        acc0 += alpha * __uint_as_float(o[b].x);
        acc1 += alpha * __uint_as_float(o[b].y);
    }
    float const v0 = mergedSum > 0.f ? acc0 / (mergedSum * kPScale) * 1.f : 0.f;
    float const v1 = mergedSum > 0.f ? acc1 / (mergedSum * kPScale) * 1.f : 0.f;
    uint32_t const ow = lamport(packBf16(v0, v1));
    st32(arena<uint8_t>(ctx, kArO) + ((t * kHeads + h) * kDv + d) * 2, ow);
#if GLM52_IO_MODE == 1 || GLM52_IO_MODE == 3
#pragma unroll
    for (int r = 0; r < 3; ++r)
    {
        st32(arena<uint8_t>(ctx, kArORep + r * kORepBytes) + ((t * kHeads + h) * kDv + d) * 2, ow);
    }
#endif
#pragma unroll
    for (int b = 0; b < kAttnBlocks; ++b)
    {
        st64(base + b * kStride, make_uint2(kEmpty, kEmpty));
    }
}

__device__ __noinline__ void stageAttn(
    Ctx const& ctx, int unit, uint32_t wOff, int const* kvLenPtr, int const* blockTable, int const* topk)
{
    int const t = unit / kAttnBlocks;
    int const blk = unit % kAttnBlocks;
    int const tid = threadIdx.x;
    int const warp = tid >> 5;
    int const lane = tid & 31;
    uint8_t* sKv = ctx.smem + wOff;
    uint8_t* sQ = sKv + kAttnKeys * kSmemRowBytes;
    uint8_t* sP = sQ + 8 * kSmemRowBytes;
    float* sRedMax = reinterpret_cast<float*>(sP + 8 * kPRowBytes);
    float* sRedSum = sRedMax + kWarps * 8;
    int* sFreshAll = reinterpret_cast<int*>(sRedSum + kWarps * 8);
    uint32_t const barGather = ctx.smemBase + kSmemSmall + kSmallBars + (kMaxRoles + warp) * 8;

    // The warp's 16 rows: rows of earlier steps from the pool (bulk copies), this step's rows from the arena. A
    // leader's top-k is this launch's: its selectors' copy in the arena (cleared to kEmpty), polled word by word (no
    // count to wait for, no reload). Meanwhile the block table goes into L1 (ld.global.nc, as the lookups below): each
    // index then finds its page there instead of one more L2 round trip.
    int const L = __ldg(kvLenPtr);
    int const slot = blk * kAttnKeys + warp * 16 + lane;
    int const numValid = min(L - (kTok - 1 - t), kTopK);
    auto issueOproj = [&]()
    {
        if (tid == 0)
        {
            // The o_proj weights held back for these gathers: the SM takes in the rows first (the 128 KB request ahead
            // of them held the attention's input back ~0.4 us per plain layer, mk-v149a).
            int* sRoles = reinterpret_cast<int*>(ctx.smem + kSmemSmall + kSmallRoles);
#pragma unroll 1
            for (int j = 0; j < kMaxRoles && sRoles[j] != 0; ++j)
            {
                if (sRoles[2 * kMaxRoles + j] == -4)
                {
#if GLM52_SETUP_MODE == 0
                    issueUnitWeights(
                        ctx, sRoles[j], sRoles[kMaxRoles + j], ctx.smemBase + kSmemSmall + kSmallBars + 8 * j);
#else
                    issueUnitWeights(ctx, j, ctx.smemBase + kSmemSmall + kSmallBars + 8 * j);
#endif
                    sRoles[2 * kMaxRoles + j] = -3;
                }
            }
        }
    };
    // A block past the row's keys (contexts below 2048 + 7): no indices, gathers or Q; the partials of an empty block
    // (O = 0, max -inf, sum 0). (Gathering row 0 of the pool for every slot past the keys put ~16k copies of one row on
    // one L2 slice: the attention took ~10 us more per layer at 32 keys than at 32k, mk-t10.)
    if (!(kSpecializedStaticFlags & kFlagSliceMerge) && blk * kAttnKeys >= numValid)
    {
        issueOproj();
        float* partO = ptr<float>(ctx, kPPartO) + static_cast<size_t>(t * kAttnBlocks + blk) * kHeads * kDv;
        for (int i = tid; i < kHeads * kDv / 2; i += kThreads)
        {
            st64(partO + 2 * i, make_uint2(0u, 0u));
        }
        if (tid < kHeads)
        {
            st64(ptr<float>(ctx, kPPartStats) + ((t * kAttnBlocks + blk) * kHeads + tid) * 2,
                make_uint2(__float_as_uint(-INFINITY), 0u));
        }
        stampInput(ctx);
        return;
    }
    int P = -1;
    if (ctx.leader)
    {
        int const nb = min((L + kIdxPageKeys - 1) / kIdxPageKeys, static_cast<int>(ctx.p[kPMaxBlocks]));
#pragma unroll 1
        for (int b = tid; b < nb; b += kThreads)
        {
            uint32_t dummy;
            asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(dummy) : "l"(blockTable + b));
        }
        // The row shard: every row from this GPU's copy of the top-k exchange (row t published by rank 2 t).
        bool const rs = (GLM52_FLAGS(ctx) & kFlagIdxAllRows) == 0;
        bool const all = idxAllKeys(L);
        if (all)
        {
            P = slot < numValid ? slot : -1;
        }
        else
        {
            uint32_t const* src = (rs ? reinterpret_cast<uint32_t const*>(
                                       ptr<uint8_t const>(ctx, kPTkUc) + static_cast<size_t>(ctx.parity) * kTkBytes)
                                      : arena<uint32_t const>(ctx, kArIdxTopk))
                + t * kTopK + slot;
            uint32_t w = lane < 16 ? ldPoll32(src) : 0u;
            uint64_t t0 = 0;
            while (!warpAll(lane >= 16 || w != kEmpty))
            {
                spinCheck(ctx, t0, 170, t, slot);
                if (lane < 16 && w == kEmpty)
                {
                    w = ldPoll32(src);
                }
            }
            P = static_cast<int>(w);
        }
        if ((rs || all) && lane < 16)
        {
            // The leader's top-k buffer (the next plain layers' input): under the row shard, or with every key, no
            // selector writes it.
            const_cast<int*>(topk)[t * kTopK + slot] = P;
        }
    }
    else if (lane < 16)
    {
        P = __ldcg(topk + t * kTopK + slot);
    }
    bool valid = false;
    bool fresh = false;
    int freshIdx = 0;
    int64_t row = 0;
    if (lane < 16)
    {
        int const block = P >> 6;
        int page = -1;
        if (P >= 0 && block < static_cast<int>(ctx.p[kPMaxBlocks]))
        {
            page = __ldg(blockTable + block);
        }
        valid = slot < numValid && page >= 0;
        fresh = valid && P >= L - kTok;
        freshIdx = P - (L - kTok);
        row = valid ? static_cast<int64_t>(page) * ctx.p[kPStrideRows] + (P & 63) + ctx.p[kPLayerRows] : 0;
    }
    uint32_t const validMask = __ballot_sync(0xffffffffu, valid);
    uint32_t const freshMask = __ballot_sync(0xffffffffu, fresh);
    uint8_t const* kvBase = ptr<uint8_t const>(ctx, kPKvBase);
    if (lane == 0)
    {
        mbarExpectTx(barGather, __popc(validMask & ~freshMask) * kRowBytes);
    }
    __syncwarp();
    if (lane < 16 && !valid)
    {
        // A slot past the row's keys: its row zeroed (its P is 0, its V must be finite), not gathered.
        uint4* z = reinterpret_cast<uint4*>(sKv + (warp * 16 + lane) * kSmemRowBytes);
#pragma unroll 4
        for (int i = 0; i < kRowBytes / 16; ++i)
        {
            z[i] = make_uint4(0, 0, 0, 0);
        }
    }
    if (valid && !fresh)
    {
        fenceProxyAsyncSmem();
        // Read once per layer: L2 evict-first (see the indexer's key pages).
        bulkCopyHint(smemU32(sKv + (warp * 16 + lane) * kSmemRowBytes), kvBase + row * kRowBytes, kRowBytes, barGather,
            policyEvictFirst());
    }
    subStamp(ctx, 5);
    issueOproj();

    // Q of row t (4 heads x 576 E4M3, rows 0-3 of sQ; rows 4-7 zero) and this step's rows among the warp's (from the
    // arena; vector j of the warp's fresh rows is lane j mod 32's), polled in one round trip (the fresh rows are in
    // long before Q).
    {
        constexpr int kVecs = kRowBytes / 16;
        constexpr int kMaxPerLane = (kTok * kVecs + 31) / 32;
        for (int i = kHeads * kVecs + tid; i < 8 * kVecs; i += kThreads)
        {
            *reinterpret_cast<uint4*>(sQ + (i / kVecs) * kSmemRowBytes + 16 * (i % kVecs)) = make_uint4(0, 0, 0, 0);
        }
        int* sFresh = sFreshAll + warp * 16;
        if (fresh)
        {
            int const rank = __popc(freshMask & ((1u << lane) - 1u));
            sFresh[2 * rank] = lane;
            sFresh[2 * rank + 1] = freshIdx;
        }
        __syncwarp();
        int const nVec = __popc(freshMask) * kVecs;
        uint8_t const* kvNew = arena<uint8_t const>(ctx, kArKv);
        auto srcOf = [&](int j)
        {
            int const r = j / kVecs;
            return kvNew + sFresh[2 * r + 1] * kRowBytes + 16 * (j - kVecs * r);
        };
        static_assert(kHeads * kVecs <= kThreads, "one Q vector per thread");
        uint8_t const* qSrc
            = arena<uint8_t const>(ctx, kArQ8) + t * kQbRows + (tid / kVecs) * kRowBytes + 16 * (tid % kVecs);
        bool const hasQ = tid < kHeads * kVecs;
        uint4 qv = hasQ ? ldPoll128(qSrc) : make_uint4(0, 0, 0, 0);
        uint4 v[kMaxPerLane];
#pragma unroll
        for (int i = 0; i < kMaxPerLane; ++i)
        {
            int const j = lane + 32 * i;
            v[i] = j < nVec ? ldPoll128(srcOf(j)) : make_uint4(0, 0, 0, 0);
        }
        uint64_t t0 = 0;
        while (true)
        {
            bool ok = full4(qv);
#pragma unroll
            for (int i = 0; i < kMaxPerLane; ++i)
            {
                ok &= full4(v[i]);
            }
            if (warpAll(ok))
            {
                break;
            }
            spinCheck(ctx, t0, 105, t, nVec);
            if (!full4(qv))
            {
                qv = ldPoll128(qSrc);
            }
#pragma unroll
            for (int i = 0; i < kMaxPerLane; ++i)
            {
                if (!full4(v[i]))
                {
                    v[i] = ldPoll128(srcOf(lane + 32 * i));
                }
            }
        }
        if (hasQ)
        {
            *reinterpret_cast<uint4*>(sQ + (tid / kVecs) * kSmemRowBytes + 16 * (tid % kVecs)) = qv;
        }
#pragma unroll
        for (int i = 0; i < kMaxPerLane; ++i)
        {
            int const j = lane + 32 * i;
            if (j < nVec)
            {
                int const r = j / kVecs;
                *reinterpret_cast<uint4*>(sKv + (warp * 16 + sFresh[2 * r]) * kSmemRowBytes + 16 * (j - kVecs * r))
                    = v[i];
            }
        }
    }
    subStamp(ctx, 6);
    __syncthreads();
    stampInput(ctx);
    subStamp(ctx, 7);

    int const g = lane / 4;
    int const c = lane % 4;
    uint32_t const qAddr = smemU32(sQ + (lane % 8) * kSmemRowBytes + 16 * ((lane / 8) % 2));
    waitBar(ctx, barGather, 0, 107);
    subStamp(ctx, 8);

    float const scaleSoftmaxLog2 = 0.0901684f; // (1 / 16) * log2(e)
    float s[4];
    {
        // kFlagAttnIlp: four accumulator chains (each MMA's two halves into different ones).
        float acc[4][4] = {};
        uint32_t const kAddr
            = smemU32(sKv + (warp * 16 + lane % 8 + 8 * ((lane / 8) % 2)) * kSmemRowBytes + 16 * (lane / 16));
        // Two K steps per iteration (one per accumulator); the query fragments are unpacked in the loop (a fully
        // unrolled loop was ~7 KiB of code, fetched cold once per layer).
#pragma unroll 2
        for (int kd = 0; kd < kRowBytes / 32; ++kd)
        {
            uint32_t b0, b1;
            ldmatrixX2(qAddr + 32 * kd, b0, b1);
            uint32_t a0, a1, a2, a3;
            ldmatrixX4(kAddr + 32 * kd, a0, a1, a2, a3);
            mmaF16(acc[kd % 2], e4m3x2ToHalf2(a0), e4m3x2ToHalf2(a1), e4m3x2ToHalf2(a2), e4m3x2ToHalf2(a3),
                e4m3x2ToHalf2(b0), e4m3x2ToHalf2(b1));
            mmaF16(acc[2 + kd % 2], e4m3x2ToHalf2(a0 >> 16), e4m3x2ToHalf2(a1 >> 16), e4m3x2ToHalf2(a2 >> 16),
                e4m3x2ToHalf2(a3 >> 16), e4m3x2ToHalf2(b0 >> 16), e4m3x2ToHalf2(b1 >> 16));
        }
        bool const valid0 = (validMask >> g) & 1u;
        bool const valid1 = (validMask >> (g + 8)) & 1u;
        s[0] = valid0 ? ((acc[0][0] + acc[1][0]) + (acc[2][0] + acc[3][0])) * scaleSoftmaxLog2 : -INFINITY;
        s[1] = valid0 ? ((acc[0][1] + acc[1][1]) + (acc[2][1] + acc[3][1])) * scaleSoftmaxLog2 : -INFINITY;
        s[2] = valid1 ? ((acc[0][2] + acc[1][2]) + (acc[2][2] + acc[3][2])) * scaleSoftmaxLog2 : -INFINITY;
        s[3] = valid1 ? ((acc[0][3] + acc[1][3]) + (acc[2][3] + acc[3][3])) * scaleSoftmaxLog2 : -INFINITY;
    }
    subStamp(ctx, 9);
    float max0 = fmaxf(s[0], s[2]);
    float max1 = fmaxf(s[1], s[3]);
#pragma unroll
    for (int off = 4; off < 32; off *= 2)
    {
        max0 = fmaxf(max0, __shfl_xor_sync(0xffffffffu, max0, off));
        max1 = fmaxf(max1, __shfl_xor_sync(0xffffffffu, max1, off));
    }
    if (g == 0)
    {
        sRedMax[warp * 8 + 2 * c] = max0;
        sRedMax[warp * 8 + 2 * c + 1] = max1;
    }
    __syncthreads();
    float ctaMax0 = -INFINITY;
    float ctaMax1 = -INFINITY;
#pragma unroll
    for (int w = 0; w < kWarps; ++w)
    {
        ctaMax0 = fmaxf(ctaMax0, sRedMax[w * 8 + 2 * c]);
        ctaMax1 = fmaxf(ctaMax1, sRedMax[w * 8 + 2 * c + 1]);
    }
    float const offset0 = ctaMax0 == -INFINITY ? 0.f : ctaMax0;
    float const offset1 = ctaMax1 == -INFINITY ? 0.f : ctaMax1;
    {
        float const p00 = exp2f(s[0] - offset0);
        float const p01 = exp2f(s[1] - offset1);
        float const p10 = exp2f(s[2] - offset0);
        float const p11 = exp2f(s[3] - offset1);
        int const slot0 = permutedSlot(warp * 16 + g);
        int const slot1 = permutedSlot(warp * 16 + g + 8);
        sP[(2 * c) * kPRowBytes + slot0] = toE4m3(p00 * kPScale);
        sP[(2 * c + 1) * kPRowBytes + slot0] = toE4m3(p01 * kPScale);
        sP[(2 * c) * kPRowBytes + slot1] = toE4m3(p10 * kPScale);
        sP[(2 * c + 1) * kPRowBytes + slot1] = toE4m3(p11 * kPScale);
        float sum0 = p00 + p10;
        float sum1 = p01 + p11;
#pragma unroll
        for (int off = 4; off < 32; off *= 2)
        {
            sum0 += __shfl_xor_sync(0xffffffffu, sum0, off);
            sum1 += __shfl_xor_sync(0xffffffffu, sum1, off);
        }
        if (g == 0)
        {
            sRedSum[warp * 8 + 2 * c] = sum0;
            sRedSum[warp * 8 + 2 * c + 1] = sum1;
        }
    }
    __syncthreads();
    subStamp(ctx, 10);
    // O^T = V^T P^T: warp w owns the 16-column output blocks 4w .. 4w + 3 (production flash decode).
    float* partO = ptr<float>(ctx, kPPartO) + static_cast<size_t>(t * kAttnBlocks + blk) * kHeads * kDv;
    {
        uint32_t const pAddr = smemU32(sP + (lane % 8) * kPRowBytes + 16 * ((lane / 8) % 2));
        uint32_t const vAddr = smemU32(sKv + lane * kSmemRowBytes + 16 * (4 * warp));
        uint32_t pb[kAttnKeys / 32][4];
#pragma unroll
        for (int kStep = 0; kStep < kAttnKeys / 32; ++kStep)
        {
            uint32_t b0, b1;
            ldmatrixX2(pAddr + 32 * kStep, b0, b1);
            pb[kStep][0] = e4m3x2ToHalf2(b0);
            pb[kStep][1] = e4m3x2ToHalf2(b1);
            pb[kStep][2] = e4m3x2ToHalf2(b0 >> 16);
            pb[kStep][3] = e4m3x2ToHalf2(b1 >> 16);
        }
        // kFlagAttnIlp: two output blocks per iteration, two independent accumulator chains.
#pragma unroll 1
        for (int bi0 = 0; bi0 < 4; bi0 += 2)
        {
            float ob2[2][4] = {};
#pragma unroll
            for (int kStep = 0; kStep < kAttnKeys / 32; ++kStep)
            {
#pragma unroll
                for (int u = 0; u < 2; ++u)
                {
                    uint32_t r0, r1, r2, r3;
                    ldmatrixX4Trans(vAddr + 32 * kStep * kSmemRowBytes + 16 * (bi0 + u), r0, r1, r2, r3);
                    uint32_t const a0 = __byte_perm(r0, r1, 0x6420);
                    uint32_t const a1 = __byte_perm(r0, r1, 0x7531);
                    uint32_t const a2 = __byte_perm(r2, r3, 0x6420);
                    uint32_t const a3 = __byte_perm(r2, r3, 0x7531);
                    mmaF16(ob2[u], e4m3x2ToHalf2(a0), e4m3x2ToHalf2(a1), e4m3x2ToHalf2(a2), e4m3x2ToHalf2(a3),
                        pb[kStep][0], pb[kStep][1]);
                    mmaF16(ob2[u], e4m3x2ToHalf2(a0 >> 16), e4m3x2ToHalf2(a1 >> 16), e4m3x2ToHalf2(a2 >> 16),
                        e4m3x2ToHalf2(a3 >> 16), pb[kStep][2], pb[kStep][3]);
                }
            }
#pragma unroll
            for (int u = 0; u < 2; ++u)
            {
                float const* ob = ob2[u];
                int const col = 16 * (4 * warp + bi0 + u) + 2 * g;
                if (c < 2)
                {
                    st64(partO + (2 * c) * kDv + col,
                        make_uint2(lamport(__float_as_uint(ob[0])), lamport(__float_as_uint(ob[2]))));
                    st64(partO + (2 * c + 1) * kDv + col,
                        make_uint2(lamport(__float_as_uint(ob[1])), lamport(__float_as_uint(ob[3]))));
                }
            }
        }
    }
    if (tid < kHeads)
    {
        float m = -INFINITY;
        float sm = 0.f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w)
        {
            m = fmaxf(m, sRedMax[w * 8 + tid]);
            sm += sRedSum[w * 8 + tid];
        }
        float* st = (kSpecializedStaticFlags & kFlagSliceMerge)
            ? reinterpret_cast<float*>(arena<uint8_t>(ctx, kArPStats)) + ((t * kAttnBlocks + blk) * kHeads + tid) * 2
            : ptr<float>(ctx, kPPartStats) + ((t * kAttnBlocks + blk) * kHeads + tid) * 2;
        st64(st, make_uint2(lamport(__float_as_uint(m)), lamport(__float_as_uint(sm))));
    }
    if constexpr ((kSpecializedStaticFlags & kFlagSliceMerge) != 0)
    {
        subStamp(ctx, 135); // the block done; 11: its slice merged
        attnSliceMerge(ctx, t, blk);
    }
    subStamp(ctx, 11);
}

#if (GLM52_STATIC_FLAGS >> 39) & 1
// kFlagMergeDry (a separate source so that the default build stays as it was).
// Merge of the 16 partials of (row t, head h): o in BF16. This unit is the partials' only reader: it empties them.
__device__ __noinline__ void stageMerge(Ctx const& ctx, int unit)
{
    int const t = unit / kHeads;
    int const h = unit % kHeads;
    int const tid = threadIdx.x;
    float* sStats = reinterpret_cast<float*>(ctx.smem + kSmemRed);
    // Thread tid < 32 also polls stats word tid (block tid / 2: max, sum), with its partials in one round trip.
    bool const hasStat = tid < 2 * kAttnBlocks;
    float* st = ptr<float>(ctx, kPPartStats) + ((t * kAttnBlocks + tid / 2) * kHeads + h) * 2 + (tid & 1);
    int const d = 2 * tid;
    uint8_t* base = reinterpret_cast<uint8_t*>(
        ptr<float>(ctx, kPPartO) + (static_cast<size_t>(t * kAttnBlocks) * kHeads + h) * kDv + d);
    constexpr int kStride = kHeads * kDv * 4;
    bool const clearFirst = (GLM52_FLAGS(ctx) & kFlagMergeClearFirst) != 0;
    // The partials back to empty for a later launch: after this unit's output (issued before it, the 4,096 clearing
    // stores held the output up behind them; o_proj saw it 0.7-2 us after the last merge, mk-v105a).
    auto clearPartials = [&]()
    {
        if (hasStat)
        {
            st32(st, kEmpty);
        }
#pragma unroll
        for (int b = 0; b < kAttnBlocks; ++b)
        {
            st64(base + b * kStride, make_uint2(kEmpty, kEmpty));
        }
    };
    // The unit's code once on zeros (no stores) while the partials are still far off, so the real pass after the poll
    // runs from a warm instruction cache (the same loop body serves both passes).
    uint32_t sw = 0u;
    uint2 o[kAttnBlocks];
#pragma unroll 1
    for (int pass = 0; pass < 2; ++pass)
    {
        bool const dry = pass == 0;
        if (dry)
        {
#pragma unroll
            for (int b = 0; b < kAttnBlocks; ++b)
            {
                o[b] = make_uint2(0u, 0u);
            }
        }
        else
        {
            sw = hasStat ? ldPoll32(st) : 0u;
#pragma unroll
            for (int b = 0; b < kAttnBlocks; ++b)
            {
                o[b] = ldPoll64(base + b * kStride);
            }
            uint64_t t0 = 0;
            while (true)
            {
                bool ok = sw != kEmpty;
#pragma unroll
                for (int b = 0; b < kAttnBlocks; ++b)
                {
                    ok &= full2(o[b]);
                }
                if (warpAll(ok))
                {
                    break;
                }
                spinCheck(ctx, t0, 108, t, h);
                if (sw == kEmpty)
                {
                    sw = ldPoll32(st);
                }
#pragma unroll
                for (int b = 0; b < kAttnBlocks; ++b)
                {
                    if (!full2(o[b]))
                    {
                        o[b] = ldPoll64(base + b * kStride);
                    }
                }
            }
        }
        if (hasStat)
        {
            sStats[tid] = __uint_as_float(sw);
        }
        if (!dry && clearFirst)
        {
            clearPartials();
        }
        __syncthreads();
        if (!dry)
        {
            stampInput(ctx);
        }
        float merged = -INFINITY;
#pragma unroll
        for (int b = 0; b < kAttnBlocks; ++b)
        {
            merged = fmaxf(merged, sStats[2 * b]);
        }
        float mergedSum = 0.f;
        float acc0 = 0.f;
        float acc1 = 0.f;
#pragma unroll
        for (int b = 0; b < kAttnBlocks; ++b)
        {
            float const mb = sStats[2 * b];
            float const alpha = mb == -INFINITY ? 0.f : exp2f(mb - merged);
            mergedSum += alpha * sStats[2 * b + 1];
            acc0 += alpha * __uint_as_float(o[b].x);
            acc1 += alpha * __uint_as_float(o[b].y);
        }
        float const v0 = mergedSum > 0.f ? acc0 / (mergedSum * kPScale) * 1.f : 0.f;
        float const v1 = mergedSum > 0.f ? acc1 / (mergedSum * kPScale) * 1.f : 0.f;
        uint32_t const ow = lamport(packBf16(v0, v1));
        if (dry)
        {
            // The dry pass's reads of the stats end before the real pass writes them.
            __syncthreads();
            continue;
        }
        st32(arena<uint8_t>(ctx, kArO) + ((t * kHeads + h) * kDv + d) * 2, ow);
#if GLM52_IO_MODE == 1 || GLM52_IO_MODE == 3
#pragma unroll
        for (int r = 0; r < 3; ++r)
        {
            st32(arena<uint8_t>(ctx, kArORep + r * kORepBytes) + ((t * kHeads + h) * kDv + d) * 2, ow);
        }
#endif
        if (!clearFirst)
        {
            clearPartials();
        }
    }
}
#else
// Merge of the 16 partials of (row t, head h): o in BF16. This unit is the partials' only reader: it empties them.
__device__ __noinline__ void stageMerge(Ctx const& ctx, int unit)
{
    int const t = unit / kHeads;
    int const h = unit % kHeads;
    int const tid = threadIdx.x;
    float* sStats = reinterpret_cast<float*>(ctx.smem + kSmemRed);
    // Thread tid < 32 also polls stats word tid (block tid / 2: max, sum), with its partials in one round trip.
    bool const hasStat = tid < 2 * kAttnBlocks;
    float* st = ptr<float>(ctx, kPPartStats) + ((t * kAttnBlocks + tid / 2) * kHeads + h) * 2 + (tid & 1);
    uint32_t sw = hasStat ? ldPoll32(st) : 0u;
    int const d = 2 * tid;
    uint8_t* base = reinterpret_cast<uint8_t*>(
        ptr<float>(ctx, kPPartO) + (static_cast<size_t>(t * kAttnBlocks) * kHeads + h) * kDv + d);
    constexpr int kStride = kHeads * kDv * 4;
    uint2 o[kAttnBlocks];
#pragma unroll
    for (int b = 0; b < kAttnBlocks; ++b)
    {
        o[b] = ldPoll64(base + b * kStride);
    }
    {
        uint64_t t0 = 0;
        while (true)
        {
            bool ok = sw != kEmpty;
#pragma unroll
            for (int b = 0; b < kAttnBlocks; ++b)
            {
                ok &= full2(o[b]);
            }
            if (warpAll(ok))
            {
                break;
            }
            spinCheck(ctx, t0, 108, t, h);
            if (sw == kEmpty)
            {
                sw = ldPoll32(st);
            }
#pragma unroll
            for (int b = 0; b < kAttnBlocks; ++b)
            {
                if (!full2(o[b]))
                {
                    o[b] = ldPoll64(base + b * kStride);
                }
            }
        }
    }
    if (hasStat)
    {
        sStats[tid] = __uint_as_float(sw);
    }
    bool const clearFirst = (GLM52_FLAGS(ctx) & kFlagMergeClearFirst) != 0;
    // The partials back to empty for a later launch: after this unit's output (issued before it, the 4,096 clearing
    // stores held the output up behind them; o_proj saw it 0.7-2 us after the last merge, mk-v105a).
    auto clearPartials = [&]()
    {
        if (hasStat)
        {
            st32(st, kEmpty);
        }
#pragma unroll
        for (int b = 0; b < kAttnBlocks; ++b)
        {
            st64(base + b * kStride, make_uint2(kEmpty, kEmpty));
        }
    };
    if (clearFirst)
    {
        clearPartials();
    }
    __syncthreads();
    stampInput(ctx);
    // Threads 0-15: block tid's weight exp2(m_b - max) into shared memory, thread 0 the merged sum's reciprocal (once
    // per unit instead of 16 exp2 and two divides in every thread: the path ran cold once per launch).
    float* sAlpha = sStats + 2 * kAttnBlocks;
    if (tid < kAttnBlocks)
    {
        float merged = -INFINITY;
#pragma unroll
        for (int b = 0; b < kAttnBlocks; ++b)
        {
            merged = fmaxf(merged, sStats[2 * b]);
        }
        float const mb = sStats[2 * tid];
        float const alpha = mb == -INFINITY ? 0.f : exp2f(mb - merged);
        sAlpha[tid] = alpha;
        float sum = alpha * sStats[2 * tid + 1];
#pragma unroll
        for (int off = kAttnBlocks / 2; off > 0; off >>= 1)
        {
            sum += __shfl_xor_sync(0x0000ffffu, sum, off);
        }
        if (tid == 0)
        {
            sAlpha[kAttnBlocks] = sum > 0.f ? 1.f / (sum * kPScale) : 0.f;
        }
    }
    __syncthreads();
    float acc0 = 0.f;
    float acc1 = 0.f;
#pragma unroll
    for (int b = 0; b < kAttnBlocks; ++b)
    {
        float const alpha = sAlpha[b];
        acc0 += alpha * __uint_as_float(o[b].x);
        acc1 += alpha * __uint_as_float(o[b].y);
    }
    float const rcp = sAlpha[kAttnBlocks];
    float const v0 = acc0 * rcp;
    float const v1 = acc1 * rcp;
#if GLM52_IO_MODE == 1 || GLM52_IO_MODE == 3
    uint32_t const ow = lamport(packBf16(v0, v1));
    st32(arena<uint8_t>(ctx, kArO) + ((t * kHeads + h) * kDv + d) * 2, ow);
#pragma unroll
    for (int r = 0; r < 3; ++r)
    {
        st32(arena<uint8_t>(ctx, kArORep + r * kORepBytes) + ((t * kHeads + h) * kDv + d) * 2, ow);
    }
#else
    st32(arena<uint8_t>(ctx, kArO) + ((t * kHeads + h) * kDv + d) * 2, lamport(packBf16(v0, v1)));
#endif
    if (!clearFirst)
    {
        clearPartials();
    }
}
#endif

// Byte offset of hop 1's slot (token t, rank r, column c): [token][rank][6144], or with kFlagHop1Lines
// [rank][o_proj unit][token][32 columns].
__device__ __forceinline__ size_t hop1Off(int t, int r, int c)
{
    if constexpr ((kSpecializedStaticFlags & kFlagHop1Lines) != 0)
    {
        return ((static_cast<size_t>(r) * (kHidden / 32) + c / 32) * kTok + t) * 64 + (c % 32) * 2;
    }
    return (static_cast<size_t>(t * kRanks + r) * kHidden + c) * 2;
}

// o_proj: z rows [32 u, +32) = o W_OV^T (o = the 4 heads' merged outputs, K = 2048; W_OV = W_O W_UV per head),
// published into every GPU's attention all-reduce slot of this rank.
__device__ __noinline__ void stageOproj(Ctx const& ctx, int unit, uint32_t wOff, uint32_t bar)
{
    int const warp = threadIdx.x >> 5;
    constexpr int kPer = kOChunks / kWarps;
    if (GLM52_FLAGS(ctx) & kFlagCanaryOproj)
    {
        // The first word of every merge unit (token t, head h).
        uint8_t const* a = arena<uint8_t const>(ctx, kArO);
        waitCanaries(
            ctx, 32, [&](int u) { return a + u * kDv * 2; }, 140);
    }
    uint4 b[kPer][2];
#if GLM52_IO_MODE == 1 || GLM52_IO_MODE == 3
    int const orr = static_cast<int>(blockIdx.x) & 3;
    uint8_t const* const oSrc = arena<uint8_t const>(ctx, orr == 0 ? kArO : kArORep + (orr - 1) * kORepBytes);
#else
    uint8_t const* const oSrc = arena<uint8_t const>(ctx, kArO);
#endif
    float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    pollB<kPer>(ctx, b, oSrc, kHeads * kDv, warp * kPer, 111);
    stampInput(ctx);
    waitBar(ctx, bar, 0, 112);
#pragma unroll
    for (int i = 0; i < kPer; ++i)
    {
#pragma unroll
        for (int t = 0; t < 2; ++t)
        {
            mmaChunkBf16(acc[t], ctx.smemBase + wOff + (t * kOChunks + warp * kPer + i) * kBf16Block, b[i]);
        }
    }
    float* out = reinterpret_cast<float*>(ctx.smem + kSmemRed + 8192);
    reduceWarps<2>(ctx, acc, out);
    // 8 tokens x 4 vectors of 8 columns.
    if (threadIdx.x < 32)
    {
        int const tok = threadIdx.x / 4;
        int const v = threadIdx.x % 4;
        uint32_t w[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            int const r = 8 * v + 2 * j;
            w[j] = lamport(packBf16(out[r * kTok + tok], out[(r + 1) * kTok + tok]));
        }
        uint8_t* mc = ptr<uint8_t>(ctx, kPH1Mc) + static_cast<size_t>(ctx.parity) * kHopBytes;
        st128(mc + hop1Off(tok, ctx.rank, 32 * unit + 8 * v), make_uint4(w[0], w[1], w[2], w[3]));
    }
}

// GLM52_ROUTER_LG: this launch's copy of the partial logits (after the top-k exchange's copies): this GPU's, and its
// multicast address.
__device__ __forceinline__ uint8_t* lgUc(Ctx const& ctx)
{
    return ptr<uint8_t>(ctx, kPTkUc) + static_cast<size_t>(ctx.numCopies) * kTkBytes
        + static_cast<size_t>(ctx.parity) * kLgBytes;
}

__device__ __forceinline__ uint8_t* lgMc(Ctx const& ctx)
{
    return ptr<uint8_t>(ctx, kPTkMc) + static_cast<size_t>(ctx.numCopies) * kTkBytes
        + static_cast<size_t>(ctx.parity) * kLgBytes;
}

// acc += A x B for one 64-wide chunk with the A fragments in registers (pieces 0-3 of the packed chunk, lane's 16 B).
__device__ __forceinline__ void mmaChunkBf16Regs(float (&acc)[4], uint4 const (&p)[4], uint4 const (&b)[2])
{
    mmaBf16(acc, p[0].x, p[2].x, p[0].y, p[2].y, b[0].x, b[0].y);
    mmaBf16(acc, p[0].z, p[2].z, p[0].w, p[2].w, b[0].z, b[0].w);
    mmaBf16(acc, p[1].x, p[3].x, p[1].y, p[3].y, b[1].x, b[1].y);
    mmaBf16(acc, p[1].z, p[3].z, p[1].w, p[3].w, b[1].z, b[1].w);
}

__device__ __noinline__ void moeWarmUp(Ctx const& ctx, uint32_t wOff);

// GLM52_ROUTER_LG, the 16 ranks' partials of logit unit u (warp 0): summed in rank order ((0..7) + (8..15)) into kArLg
// (lane l: token l / 4, experts 16 u + 4 (l % 4) .. + 3), then the 16 slots back to empty (single reader; their next
// writers are the launch numCopies on), after the logits (a CTA's stores leave in issue order).
__device__ __forceinline__ void lgReduce(Ctx const& ctx, int u)
{
    int const lane = threadIdx.x & 31;
    uint8_t* const src = lgUc(ctx) + u * kLgSlotBytes + 16 * lane;
    uint4 v[kRanks];
    pollStrided<kRanks>(ctx, v, src, kLgUnits * kLgSlotBytes, 192);
    float a[4] = {0.f, 0.f, 0.f, 0.f};
    float b[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int r = 0; r < kRanks; ++r)
    {
        uint32_t const w[4] = {v[r].x, v[r].y, v[r].z, v[r].w};
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            if (r < 8)
            {
                a[j] += __uint_as_float(w[j]);
            }
            else
            {
                b[j] += __uint_as_float(w[j]);
            }
        }
    }
    uint4 const o
        = make_uint4(lamport(__float_as_uint((0.f + a[0]) + b[0])), lamport(__float_as_uint((0.f + a[1]) + b[1])),
            lamport(__float_as_uint((0.f + a[2]) + b[2])), lamport(__float_as_uint((0.f + a[3]) + b[3])));
    st128(arena<uint8_t>(ctx, kArLg) + ((lane >> 2) * kExperts + 16 * u + 4 * (lane & 3)) * 4, o);
    if (GLM52_STAMPS(ctx) != nullptr && lane == 0)
    {
        GLM52_STAMPS(ctx)[kMaxRoles * 4 + 4 + 131] = globalTimer();
    }
    uint4 const e = make_uint4(kEmpty, kEmpty, kEmpty, kEmpty);
    uint64_t const policy = policyEvictFirst();
#pragma unroll
    for (int r = 0; r < kRanks; ++r)
    {
        st128Hint(src + r * kLgUnits * kLgSlotBytes, e, policy);
    }
}

// GLM52_ROUTER_LG on CTAs 148-163: o_proj unit `unit` exactly as stageOproj (its hop-1 shard goes out first), then on
// the same polled O logit unit u = unit - kLgOprojFirst: this rank's partial logits of experts [16 u, +16) =
// M_r [O | residual columns [384 r, +384)], one 16-row tile (its O part's 64 KB in shared memory after o_proj's 128 KB;
// warp w < 6 also residual chunk w, its A fragments and the residual loaded from global memory at the unit's start),
// into every GPU's slot [rank][u] as FP32 [8 tokens][16 experts]. Then the CTA's MoE warm-up (kFlagWarmMoe; the MoE
// stage skips it, kMoeWarmed) while the other ranks' partials are on their way, and their sum (lgReduce).
__device__ __noinline__ void stageOprojLg(
    Ctx const& ctx, int unit, uint32_t wOff, uint32_t bar, __nv_bfloat16 const* residual)
{
    int const warp = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
    int const u = unit - kLgOprojFirst;
    constexpr int kPer = kOChunks / kWarps;
    // The residual (the launch's input, in since the dependency wait) and the residual chunk's weights, long before O.
    uint4 br[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
    uint4 ar[4] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
    if (warp < kLgResChunks)
    {
        loadB(br, residual, kHidden, kLgResChunks * ctx.rank + warp);
        uint8_t const* const a = ptr<uint8_t const>(ctx, kPWLg)
            + (static_cast<size_t>(u) * kLgChunks + kOChunks + warp) * kBf16Block + 16 * lane;
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            ar[k] = ldcg128(a + 512 * k);
        }
    }
    uint4 b[kPer][2];
#if GLM52_IO_MODE == 1 || GLM52_IO_MODE == 3
    int const orr = static_cast<int>(blockIdx.x) & 3;
    uint8_t const* const oSrc = arena<uint8_t const>(ctx, orr == 0 ? kArO : kArORep + (orr - 1) * kORepBytes);
#else
    uint8_t const* const oSrc = arena<uint8_t const>(ctx, kArO);
#endif
    float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    pollB<kPer>(ctx, b, oSrc, kHeads * kDv, warp * kPer, 111);
    stampInput(ctx);
    waitBar(ctx, bar, 0, 112);
#pragma unroll
    for (int i = 0; i < kPer; ++i)
    {
#pragma unroll
        for (int t = 0; t < 2; ++t)
        {
            mmaChunkBf16(acc[t], ctx.smemBase + wOff + (t * kOChunks + warp * kPer + i) * kBf16Block, b[i]);
        }
    }
    float* out = reinterpret_cast<float*>(ctx.smem + kSmemRed + 8192);
    reduceWarps<2>(ctx, acc, out);
    if (threadIdx.x < 32)
    {
        int const tok = threadIdx.x / 4;
        int const v = threadIdx.x % 4;
        uint32_t w[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            int const r = 8 * v + 2 * j;
            w[j] = lamport(packBf16(out[r * kTok + tok], out[(r + 1) * kTok + tok]));
        }
        uint8_t* mc = ptr<uint8_t>(ctx, kPH1Mc) + static_cast<size_t>(ctx.parity) * kHopBytes;
        st128(mc + hop1Off(tok, ctx.rank, 32 * unit + 8 * v), make_uint4(w[0], w[1], w[2], w[3]));
    }
    // The logit tile on the same O fragments (reduceWarps<1> writes its warp sums below o_proj's out and syncs before
    // overwriting out, which threads 0-31 read above).
    float acc1[1][4] = {{0.f, 0.f, 0.f, 0.f}};
#pragma unroll
    for (int i = 0; i < kPer; ++i)
    {
        mmaChunkBf16(acc1[0], ctx.smemBase + wOff + (2 * kOChunks + warp * kPer + i) * kBf16Block, b[i]);
    }
    if (warp < kLgResChunks)
    {
        mmaChunkBf16Regs(acc1[0], ar, br);
    }
    reduceWarps<1>(ctx, acc1, out);
    if (threadIdx.x < 32)
    {
        // Lane l: token l / 4, experts 4 (l % 4) .. + 3 of the logit unit.
        int const tok = threadIdx.x >> 2;
        int const e0 = 4 * (threadIdx.x & 3);
        uint4 const w = make_uint4(lamport(__float_as_uint(out[(e0 + 0) * kTok + tok])),
            lamport(__float_as_uint(out[(e0 + 1) * kTok + tok])), lamport(__float_as_uint(out[(e0 + 2) * kTok + tok])),
            lamport(__float_as_uint(out[(e0 + 3) * kTok + tok])));
        st128(lgMc(ctx) + (ctx.rank * kLgUnits + u) * kLgSlotBytes + 16 * threadIdx.x, w);
    }
    subStamp(ctx, 130);
    if (GLM52_FLAGS(ctx) & kFlagWarmMoe)
    {
        moeWarmUp(ctx, 0);
        if (threadIdx.x == 0)
        {
            reinterpret_cast<int*>(ctx.smem + kSmemSmall + kSmallMoe)[kMoeWarmed] = 1;
        }
    }
    if (warp == 0)
    {
        lgReduce(ctx, u);
    }
}

// Empties this GPU's copy of the next launch's buffers (their readers finished before this launch began): the two hop
// buffers (counted: the MoE all-reduce's publish waits for them), then the arena (counted too: kCountedClearUnits).
__device__ __noinline__ void stageClear(Ctx const& ctx, int unit)
{
    int const other = (ctx.parity + 1) % ctx.numCopies;
    uint4 const e = make_uint4(kEmpty, kEmpty, kEmpty, kEmpty);
    // Under the L2 evict-first policy: only a later launch reads these ~5 MB (-1.1 us per slice, mk-v116a).
    uint64_t const policy = policyEvictFirst();
    if (unit < kHopClearUnits)
    {
        int const begin = unit * kClearUnitBytes;
        int const which = begin / kHopBytes;
        uint8_t* base = which == 0 ? ptr<uint8_t>(ctx, kPH1Uc) + static_cast<size_t>(other) * kHopBytes + begin
            : which == 1 ? ptr<uint8_t>(ctx, kPH2Uc) + static_cast<size_t>(other) * kHopBytes + (begin - kHopBytes)
                         : ptr<uint8_t>(ctx, kPTkUc) + static_cast<size_t>(other) * kTkBytes + (begin - 2 * kHopBytes);
        for (int off = 16 * threadIdx.x; off < kClearUnitBytes; off += 16 * kThreads)
        {
            st128Hint(base + off, e, policy);
        }
        signalCounter(ctx, kCClear);
    }
    else
    {
        int const begin = (unit - kHopClearUnits) * kClearUnitBytes;
        int const end = min(begin + kClearUnitBytes, kArenaBytes);
        uint8_t* base = ptr<uint8_t>(ctx, kPArena) + static_cast<size_t>(other) * kArenaBytes;
        for (int off = begin + 16 * threadIdx.x; off < end; off += 16 * kThreads)
        {
            st128Hint(base + off, e, policy);
        }
        signalCounter(ctx, kCClear);
    }
}

// Polls this GPU's copy of the 16 slots of 8 columns (token tok, columns col..col+7; hop 1's layout with first); FP32
// sum in rank order ((0..7) + (8..15)), as the one-shot hop.
__device__ __forceinline__ void pollReduce8(
    Ctx const& ctx, uint8_t const* buf, int tok, int col, float (&sum)[8], int code, bool first)
{
    uint4 v[kRanks];
    if (first)
    {
        pollStrided<kRanks>(
            ctx, v, buf + hop1Off(tok, 0, col), static_cast<int>(hop1Off(tok, 1, col) - hop1Off(tok, 0, col)), code);
    }
    else
    {
        pollStrided<kRanks>(ctx, v, buf + (static_cast<size_t>(tok * kRanks) * kHidden + col) * 2, kHidden * 2, code);
    }
    float a[8], b[8];
#pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        a[j] = 0.f;
        b[j] = 0.f;
    }
#pragma unroll
    for (int r = 0; r < kRanks; ++r)
    {
        uint32_t const w[4] = {v[r].x, v[r].y, v[r].z, v[r].w};
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            if (r < 8)
            {
                a[2 * j] += bf16Lo(w[j]);
                a[2 * j + 1] += bf16Hi(w[j]);
            }
            else
            {
                b[2 * j] += bf16Lo(w[j]);
                b[2 * j + 1] += bf16Hi(w[j]);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        sum[j] = (0.f + a[j]) + b[j];
    }
}

// The two all-reduce consumers: token t, columns [1536 s, +1536). Threads 0-191 own 8 columns each.
// H1 (second false): res = s + residual_in (kept in shared memory), h = RMSNorm_post(res) -> arena (BF16 and NVFP4).
// H2 (second true): res' = s + res, x_next = RMSNorm_next(res'), residual_out = res'. One copy of the code for both
// (the CTAs that run them run both).
__device__ __noinline__ void moeWarmUp(Ctx const& ctx, uint32_t wOff);

// kFlagHPrime: 1 / rms of token t from its four H1 units' sum-of-squares partials in kArSsq1 (H1's own formula),
// polled.
__device__ __forceinline__ float hopRcp(Ctx const& ctx, int t, int code)
{
    uint4 const all = poll4(ctx, arena<float>(ctx, kArSsq1) + t * 4, code);
    float const S
        = (((0.f + __uint_as_float(all.x)) + __uint_as_float(all.y)) + __uint_as_float(all.z)) + __uint_as_float(all.w);
    return rsqrtf(S / static_cast<float>(kHidden) + kEps);
}

// kFlagHopSelfClear: the 16 slots of this launch's hop 1 that H1 unit (t, col) read, and those of the previous
// launch's hop 2 (its H2 read them; their next writers are the launch numCopies on): back to empty, nothing waits for
// them. At H1's end, or with kFlagHopClearLate at the start of the same CTA's H2 unit, which waits for hop 2 anyway.
__device__ __forceinline__ void hopSelfClear(Ctx const& ctx, int t, int col)
{
    uint4 const e = make_uint4(kEmpty, kEmpty, kEmpty, kEmpty);
    uint64_t const policy = policyEvictFirst();
    uint8_t* const h1 = ptr<uint8_t>(ctx, kPH1Uc) + static_cast<size_t>(ctx.parity) * kHopBytes;
    uint8_t* const h2
        = ptr<uint8_t>(ctx, kPH2Uc) + static_cast<size_t>((ctx.parity + ctx.numCopies - 1) % ctx.numCopies) * kHopBytes;
#pragma unroll
    for (int r = 0; r < kRanks; ++r)
    {
        st128Hint(h1 + hop1Off(t, r, col), e, policy);
        st128Hint(h2 + (static_cast<size_t>(t * kRanks + r) * kHidden + col) * 2, e, policy);
    }
}

__device__ __noinline__ void stageHop(
    Ctx const& ctx, int unit, bool second, __nv_bfloat16 const* residualIn, __nv_bfloat16* xOut, __nv_bfloat16* resOut)
{
    int const t = unit / 4;
    int const s4 = unit % 4;
    int const tid = threadIdx.x;
    float* sc = reinterpret_cast<float*>(ctx.smem + kSmemSmall + kSmallScalars);
    uint4* sRes = reinterpret_cast<uint4*>(ctx.smem + kSmemRes);
    bool const active = tid < kColsPerHopUnit / 8;
    int const col = kColsPerHopUnit * s4 + 8 * tid;
    uint8_t const* buf = (second ? ptr<uint8_t const>(ctx, kPH2Uc) : ptr<uint8_t const>(ctx, kPH1Uc))
        + static_cast<size_t>(ctx.parity) * kHopBytes;
    // H2 of a launch whose successor reads x' from its arena (kFlagNextChained): x' and the residual only.
    bool const xOnly = second && (kSpecializedStaticFlags & kFlagXPrime) != 0 && (GLM52_FLAGS(ctx) & kFlagNextChained);
    // The next launch's arena copy (H2: its x, kFlagEarlyX there).
    uint8_t* const nextArena
        = ptr<uint8_t>(ctx, kPArena) + static_cast<size_t>((ctx.parity + 1) % ctx.numCopies) * kArenaBytes;
    uint4 resPrev = make_uint4(0, 0, 0, 0);
    uint4 gam = make_uint4(0, 0, 0, 0);
    // H1: the layer's FC1 activation global scale (Alpha1[kExperts]; 1 for GLM-5.2-NVFP4), before the hop wait.
    float inScale = 1.f;
    if (active)
    {
        resPrev
            = second ? sRes[tid] : *reinterpret_cast<uint4 const*>(residualIn + static_cast<size_t>(t) * kHidden + col);
        gam = __ldg(reinterpret_cast<uint4 const*>(
            (second ? ptr<__nv_bfloat16 const>(ctx, kPGammaNext) : ptr<__nv_bfloat16 const>(ctx, kPGammaPost)) + col));
        if (!second)
        {
            inScale = __ldg(ptr<float const>(ctx, kPAlpha1) + kExperts);
        }
    }
    if constexpr ((kSpecializedStaticFlags & kFlagHopSelfClear) != 0
        && (kSpecializedStaticFlags & kFlagHopClearLate) != 0)
    {
        if (second && active)
        {
            hopSelfClear(ctx, t, col); // this CTA's H1 unit (same t, col) read them
        }
    }
    if constexpr ((kSpecializedStaticFlags & kFlagWarmInHop) != 0)
    {
        if (!second && (GLM52_FLAGS(ctx) & kFlagWarmMoe))
        {
            // The CTA's MoE warm-up while hop 1's data is on its way (its MoE region and TMEM are free in H1): it would
            // otherwise sit between H1 and this CTA's routing (~1.6 us; these CTAs set the MoE tail).
            moeWarmUp(ctx, 0);
        }
    }
    float res[8];
    float ssq = 0.f;
    if (active)
    {
        float sum[8];
        pollReduce8(ctx, buf, t, col, sum, second ? 120 : 113, !second);
        uint32_t const rp[4] = {resPrev.x, resPrev.y, resPrev.z, resPrev.w};
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            float const s0 = bf16Round(sum[2 * j]);
            float const s1 = bf16Round(sum[2 * j + 1]);
            res[2 * j] = bf16Round(s0 + bf16Lo(rp[j]));
            res[2 * j + 1] = bf16Round(s1 + bf16Hi(rp[j]));
        }
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            ssq += bf16Round(res[j] * res[j]);
        }
        if constexpr ((kSpecializedStaticFlags & kFlagHPrime) != 0)
        {
            if (!second)
            {
                // h' = res x gamma for the router and the shared gate_up, before the exchange (they scale by 1 / rms).
                uint32_t const gw[4] = {gam.x, gam.y, gam.z, gam.w};
                uint32_t hp[4];
#pragma unroll
                for (int j = 0; j < 4; ++j)
                {
                    hp[j] = lamport(
                        packBf16(bf16Round(res[2 * j] * bf16Lo(gw[j])), bf16Round(res[2 * j + 1] * bf16Hi(gw[j]))));
                }
                st128(arena<uint8_t>(ctx, kArHp) + (static_cast<size_t>(t) * kHidden + col) * 2,
                    make_uint4(hp[0], hp[1], hp[2], hp[3]));
            }
        }
        if constexpr ((kSpecializedStaticFlags & kFlagXPrime) != 0)
        {
            if (second)
            {
                // The next launch's x' = res' x gamma_next, before the exchange (its fused-A scales by 1 / rms).
                uint32_t const gw[4] = {gam.x, gam.y, gam.z, gam.w};
                uint32_t xp[4];
#pragma unroll
                for (int j = 0; j < 4; ++j)
                {
                    xp[j] = lamport(
                        packBf16(bf16Round(res[2 * j] * bf16Lo(gw[j])), bf16Round(res[2 * j + 1] * bf16Hi(gw[j]))));
                }
                st128(nextArena + kArX + xArOff(t, col), make_uint4(xp[0], xp[1], xp[2], xp[3]));
            }
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
    {
        ssq += __shfl_xor_sync(0xffffffffu, ssq, off);
    }
    int const warp = tid >> 5;
    if ((tid & 31) == 0)
    {
        sc[warp] = ssq;
    }
    __syncthreads();
    // Sum-of-squares partials of the token's 4 units through the arena.
    if (tid == 0)
    {
        float part = 0.f;
        for (int w = 0; w < kWarps; ++w)
        {
            part += sc[w];
        }
        if constexpr ((kSpecializedStaticFlags & kFlagXPrime) != 0)
        {
            if (second)
            {
                st32(nextArena + kArXSsq + (t * 4 + s4) * 4, lamport(__float_as_uint(part)));
            }
        }
        if (!xOnly)
        {
            float* ssqBuf = arena<float>(ctx, second ? kArSsq2 : kArSsq1) + t * 4;
            st32(ssqBuf + s4, lamport(__float_as_uint(part)));
            uint4 const all = poll4(ctx, ssqBuf, second ? 121 : 114);
            sc[8] = __uint_as_float(all.x);
            sc[9] = __uint_as_float(all.y);
            sc[10] = __uint_as_float(all.z);
            sc[11] = __uint_as_float(all.w);
        }
    }
    __syncthreads();
    stampInput(ctx);
    if (xOnly)
    {
        // The next launch takes x' (published above) and this residual: no normalized x, no exchange.
        if (active)
        {
            *reinterpret_cast<uint4*>(resOut + static_cast<size_t>(t) * kHidden + col) = make_uint4(
                packBf16(res[0], res[1]), packBf16(res[2], res[3]), packBf16(res[4], res[5]), packBf16(res[6], res[7]));
        }
        return;
    }
    float const S = (((0.f + sc[8]) + sc[9]) + sc[10]) + sc[11];
    float const rcp = rsqrtf(S / static_cast<float>(kHidden) + kEps);
    if (active)
    {
        uint32_t const gw[4] = {gam.x, gam.y, gam.z, gam.w};
        float hv[8];
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            hv[2 * j] = bf16Round((res[2 * j] * rcp) * bf16Lo(gw[j]));
            hv[2 * j + 1] = bf16Round((res[2 * j + 1] * rcp) * bf16Hi(gw[j]));
        }
        uint4 const hPacked = make_uint4(
            packBf16(hv[0], hv[1]), packBf16(hv[2], hv[3]), packBf16(hv[4], hv[5]), packBf16(hv[6], hv[7]));
        uint4 const rPacked = make_uint4(
            packBf16(res[0], res[1]), packBf16(res[2], res[3]), packBf16(res[4], res[5]), packBf16(res[6], res[7]));
        if (second)
        {
            *reinterpret_cast<uint4*>(xOut + static_cast<size_t>(t) * kHidden + col) = hPacked;
            *reinterpret_cast<uint4*>(resOut + static_cast<size_t>(t) * kHidden + col) = rPacked;
            // The next launch's x (kFlagEarlyX there): its arena copy's slot, emptied before this layer's shards went
            // out (with kFlagXPrime x' went out before the exchange).
            if constexpr ((kSpecializedStaticFlags & kFlagXPrime) == 0)
            {
                st128(nextArena + kArX + xArOff(t, col),
                    make_uint4(lamport(hPacked.x), lamport(hPacked.y), lamport(hPacked.z), lamport(hPacked.w)));
            }
        }
        else
        {
            sRes[tid] = rPacked;
            st128(arena<uint8_t>(ctx, kArH) + (static_cast<size_t>(t) * kHidden + col) * 2,
                make_uint4(lamport(hPacked.x), lamport(hPacked.y), lamport(hPacked.z), lamport(hPacked.w)));
            // NVFP4 of h (16-element blocks: this thread and its neighbour).
            float vmax = 0.f;
#pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                vmax = fmaxf(vmax, fabsf(hv[j]));
            }
            vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, 1));
            // As the stock quantizer: block scales and values of inScale h.
            float const sfValue = inScale * (vmax * rcpApproxFtz(6.f));
            uint32_t const sf = toE4m3(sfValue);
            float const sff = static_cast<float>(*reinterpret_cast<__nv_fp8_e4m3 const*>(&sf));
            float const outScale = vmax != 0.f ? rcpApproxFtz(sff * rcpApproxFtz(inScale)) : 0.f;
            uint32_t packed = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                packed |= toE2m1x2(hv[2 * j] * outScale, hv[2 * j + 1] * outScale) << (8 * j);
            }
            st32(arena<uint8_t>(ctx, kArHfp4) + t * 3072 + col / 2, lamport(packed));
            // Scales: lane 8k packs the blocks of lanes 8k, 8k + 2, 8k + 4, 8k + 6 (the warp is active as a whole).
            uint32_t const s1 = __shfl_down_sync(0xffffffffu, sf, 2);
            uint32_t const s2 = __shfl_down_sync(0xffffffffu, sf, 4);
            uint32_t const s3 = __shfl_down_sync(0xffffffffu, sf, 6);
            if ((tid & 7) == 0)
            {
                st32(arena<uint8_t>(ctx, kArHsf) + t * 384 + col / 16,
                    lamport(sf | (s1 << 8) | (s2 << 16) | (s3 << 24)));
            }
        }
    }
    if constexpr ((kSpecializedStaticFlags & kFlagHopSelfClear) != 0
        && (kSpecializedStaticFlags & kFlagHopClearLate) == 0)
    {
        if (!second && active)
        {
            hopSelfClear(ctx, t, col);
        }
    }
}

// Router GEMV: logits of experts [16 et, +16) over K quarter kq.
__device__ __noinline__ void stageRouterGemv(Ctx const& ctx, int unit, uint32_t wOff, uint32_t bar)
{
    int const et = unit / 4;
    int const kq = unit % 4;
    int const warp = threadIdx.x >> 5;
    constexpr int kPer = 24 / kWarps;
    if (GLM52_FLAGS(ctx) & kFlagCanaryH)
    {
        // The first word of the H1 unit (token t, quarter kq) of every token.
        uint8_t const* a = arena<uint8_t const>(ctx, kArH) + 1536 * kq * 2;
        waitCanaries(
            ctx, kTok, [&](int t) { return a + t * kHidden * 2; }, 141);
    }
    uint4 b[kPer][2];
    pollB<kPer>(ctx, b,
        arena<uint8_t const>(ctx, (kSpecializedStaticFlags & kFlagHPrime) ? kArHp : kArH) + 1536 * kq * 2, kHidden,
        warp * kPer, 115);
    stampInput(ctx);
    waitBar(ctx, bar, 0, 116);
    float acc[1][4] = {{0.f, 0.f, 0.f, 0.f}};
#pragma unroll
    for (int i = 0; i < kPer; ++i)
    {
        mmaChunkBf16(acc[0], ctx.smemBase + wOff + (warp * kPer + i) * kBf16Block, b[i]);
    }
    float* out = reinterpret_cast<float*>(ctx.smem + kSmemRed + 8192);
    reduceWarps<1>(ctx, acc, out);
    if (threadIdx.x < 128)
    {
        int const row = threadIdx.x / kTok;
        int const tok = threadIdx.x % kTok;
        st32(arena<float>(ctx, kArLogit) + (kq * kTok + tok) * kExperts + 16 * et + row,
            lamport(__float_as_uint(out[row * kTok + tok])));
    }
}

__device__ __forceinline__ float sigmoidAccurate(float x)
{
    return 0.5f * tanhf(0.5f * x) + 0.5f;
}

// noaux_tc routing of the 8 tokens (warp t = token t) from the polled router logits: sigmoid + BF16 bias, top-8 (ties
// to the smaller expert), weights 2.5 * sig / sum, into the MoE work list (E, slot experts, slot and weight of every
// (token, k)). With publish, also into the arena (the station check reads it there).
__device__ __forceinline__ void routeTokens(Ctx const& ctx, int* smemMoe, bool publish)
{
    int const warp = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
    int* sTopExpert = reinterpret_cast<int*>(ctx.smem + kSmemRed);
    float* sTopWeight = reinterpret_cast<float*>(ctx.smem + kSmemRed + 256);
    uint32_t* sMask = reinterpret_cast<uint32_t*>(ctx.smem + kSmemRed + 512);
    uint32_t* sTokMask = reinterpret_cast<uint32_t*>(ctx.smem + kSmemRed + 2048); // per expert: the tokens choosing it
    // GLM52_ROUTER_LG == 2: the reduced logits (one word per expert, kArLg); else the router GEMV's K quarters.
    constexpr int kQ = kRouterLg == 2 ? 1 : 4;
    float const* lp = arena<float const>(ctx, kRouterLg == 2 ? kArLg : kArLogit);
    __nv_bfloat16 const* bias = ptr<__nv_bfloat16 const>(ctx, kPRouterBias);
    int const t = warp;
    sTokMask[threadIdx.x] = 0;
    // The lane's 8 bias values first (constants; their loads overlap the poll).
    uint32_t bz[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        bz[i] = ldPinnedU16(bias + 32 * i + lane);
    }
    // kFlagHPrime: the token's 4 sum-of-squares partials (in long before the logits), requested with them (read after
    // the poll, their L2 round trip sat between the logits and the top-8).
    uint4 ssq = make_uint4(0u, 0u, 0u, 0u);
    if constexpr ((kSpecializedStaticFlags & kFlagHPrime) != 0)
    {
        if (lane == 0)
        {
            ssq = ldPoll128(arena<float const>(ctx, kArSsq1) + t * 4);
        }
    }
    // Lane: experts 32 i + lane, the kQ partials of each (8 kQ words, polled together).
    uint32_t lw[8][kQ];
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
#pragma unroll
        for (int k = 0; k < kQ; ++k)
        {
            lw[i][k] = ldPoll32(lp + (k * kTok + t) * kExperts + 32 * i + lane);
        }
    }
    constexpr bool kRstdShared = (kSpecializedStaticFlags & kFlagHPrime) && (kSpecializedStaticFlags & kFlagRstdShared);
    {
        uint64_t t0 = 0;
        while (true)
        {
            bool ok = true;
#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
#pragma unroll
                for (int k = 0; k < kQ; ++k)
                {
                    ok &= lw[i][k] != kEmpty;
                }
            }
            if constexpr (kRstdShared)
            {
                ok &= lane != 0 || full4(ssq);
            }
            if (warpAll(ok))
            {
                break;
            }
            spinCheck(ctx, t0, 117, t, 0);
            if constexpr (kRstdShared)
            {
                if (lane == 0 && !full4(ssq))
                {
                    ssq = ldPoll128(arena<float const>(ctx, kArSsq1) + t * 4);
                }
            }
#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
#pragma unroll
                for (int k = 0; k < kQ; ++k)
                {
                    if (lw[i][k] == kEmpty)
                    {
                        lw[i][k] = ldPoll32(lp + (k * kTok + t) * kExperts + 32 * i + lane);
                    }
                }
            }
        }
    }
    stampInput(ctx);
    subStamp(ctx, 14);
    // kFlagHPrime: the router ran on h' = res x gamma: the token's 1 / rms (hopRcp's).
    float rcpT = 1.f;
    if constexpr ((kSpecializedStaticFlags & kFlagHPrime) != 0)
    {
        if (lane == 0)
        {
            if (!full4(ssq))
            {
                ssq = poll4(ctx, arena<float const>(ctx, kArSsq1) + t * 4, 146);
            }
            float const S = (((0.f + __uint_as_float(ssq.x)) + __uint_as_float(ssq.y)) + __uint_as_float(ssq.z))
                + __uint_as_float(ssq.w);
            rcpT = rsqrtf(S / static_cast<float>(kHidden) + kEps);
            if constexpr (kRstdShared)
            {
                smemMoe[kMoeRcp + t] = __float_as_int(rcpT);
            }
        }
        rcpT = __shfl_sync(0xffffffffu, rcpT, 0);
    }
    // Selection score = sigmoid + bias; the weights use the sigmoid itself.
    float sg[8];
    float score[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        float logit = 0.f;
#pragma unroll
        for (int k = 0; k < kQ; ++k)
        {
            logit += __uint_as_float(lw[i][k]);
        }
        if constexpr ((kSpecializedStaticFlags & kFlagHPrime) != 0)
        {
            logit *= rcpT;
        }
        sg[i] = sigmoidAccurate(logit);
        score[i] = sg[i] + __uint_as_float(bz[i] << 16);
    }
    uint32_t taken = 0;
    float topSig = 0.f;
    int topExpert = 0;
    // Top-8 with the warp reductions: per round the largest score (order-preserving bits), then the smallest expert
    // among the lanes holding it (ties go to the smaller expert, as a sorted top-k).
    uint32_t key[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        uint32_t const b = __float_as_uint(score[i]);
        key[i] = b ^ ((b >> 31) ? 0xffffffffu : 0x80000000u);
    }
#pragma unroll 1
    for (int k = 0; k < 8; ++k)
    {
        // The lane's best remaining candidate (the smaller i on a tie: the smaller expert).
        uint32_t best = 0;
        int bi = 0;
#pragma unroll
        for (int i = 7; i >= 0; --i)
        {
            if (!(taken & (1u << i)) && key[i] >= best)
            {
                best = key[i];
                bi = i;
            }
        }
        uint32_t const wmax = __reduce_max_sync(0xffffffffu, best);
        uint32_t const e = static_cast<uint32_t>(32 * bi + lane);
        uint32_t const win = __reduce_min_sync(0xffffffffu, best == wmax ? e : 0xffffu);
        float s0 = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            s0 = i == bi ? sg[i] : s0;
        }
        float const ws = __shfl_sync(0xffffffffu, s0, static_cast<int>(win & 31u));
        if (static_cast<uint32_t>(lane) == (win & 31u))
        {
            taken |= 1u << (win >> 5);
        }
        if (lane == k)
        {
            topSig = ws;
            topExpert = static_cast<int>(win);
        }
    }
    subStamp(ctx, 15);
    float const sig = lane < 8 ? topSig : 0.f;
    float sum = sig;
    sum += __shfl_xor_sync(0xffffffffu, sum, 4);
    sum += __shfl_xor_sync(0xffffffffu, sum, 2);
    sum += __shfl_xor_sync(0xffffffffu, sum, 1);
    if (lane < 8)
    {
        float const w = bf16Round(sig * 2.5f / (sum + 0.f));
        sTopExpert[t * 8 + lane] = topExpert;
        sTopWeight[t * 8 + lane] = w;
    }
    uint32_t* sSlotMask = reinterpret_cast<uint32_t*>(ctx.smem + kSmemRed + 1024);
    int* sSlotOfExpert = reinterpret_cast<int*>(ctx.smem + kSmemRed + 12288);
    __syncthreads();
    if (lane < 8)
    {
        atomicOr(&sTokMask[topExpert], 1u << t);
    }
    __syncthreads();
    // Thread e (one per expert): the tokens that chose it; the selected experts in index order are the slots, so a
    // warp's ballot and the earlier warps' counts give e's slot.
    {
        int const e = threadIdx.x;
        uint32_t const tokMask = sTokMask[e];
        uint32_t const sel = __ballot_sync(0xffffffffu, tokMask != 0);
        if (lane == 0)
        {
            sMask[warp] = sel;
        }
        __syncthreads();
        int slot = __popc(sel & ((1u << lane) - 1u));
        int total = 0;
#pragma unroll
        for (int w = 0; w < kWarps; ++w)
        {
            int const c = __popc(sMask[w]);
            slot += w < warp ? c : 0;
            total += c;
        }
        if (tokMask != 0)
        {
            smemMoe[kMoeSlotExpert + slot] = e;
            sSlotMask[slot] = tokMask;
            sSlotOfExpert[e] = slot;
        }
        if (threadIdx.x == 0)
        {
            smemMoe[kMoeE] = total;
        }
    }
    __syncthreads();
    if (threadIdx.x < kTok * 8)
    {
        smemMoe[kMoeSlotOfTk + threadIdx.x] = sSlotOfExpert[sTopExpert[threadIdx.x]];
        smemMoe[kMoeWeight + threadIdx.x] = __float_as_int(sTopWeight[threadIdx.x]);
    }
    __syncthreads();
    if (publish && threadIdx.x < kTok * 8)
    {
        int* route = arena<int>(ctx, kArRoute);
        int const E = smemMoe[kMoeE];
        if (threadIdx.x < E)
        {
            st32(route + kRouteSlotExpert + threadIdx.x, static_cast<uint32_t>(smemMoe[kMoeSlotExpert + threadIdx.x]));
        }
        st32(route + kRouteSlotOfTk + threadIdx.x, static_cast<uint32_t>(smemMoe[kMoeSlotOfTk + threadIdx.x]));
        st32(route + kRouteWeight + threadIdx.x, static_cast<uint32_t>(smemMoe[kMoeWeight + threadIdx.x]));
        if (threadIdx.x == 0)
        {
            st32(route + kRouteE, static_cast<uint32_t>(E));
        }
    }
}

// Shared expert gate_up: rows [16 ft, +16) (gate) and [128 + 16 ft, +16) (up) over K quarter kq.
__device__ __noinline__ void stageSharedGu(Ctx const& ctx, int unit, uint32_t wOff, uint32_t bar)
{
    int const ft = unit / 4;
    int const kq = unit % 4;
    int const warp = threadIdx.x >> 5;
    constexpr int kPer = 24 / kWarps;
    if constexpr (kRouterLg == 2)
    {
        if (GLM52_FLAGS(ctx) & kFlagWarmMoe)
        {
            // The CTA's MoE warm-up while h' is on its way (as H1's in its hop wait): with the reduced logits in
            // before h', it would otherwise sit between this unit and the CTA's routing. Its stale operands include
            // this unit's weights (the MMAs' results are discarded), so they land first: the copy's completion is
            // the only edge that orders its writes before the tensor cores' reads.
            waitBar(ctx, bar, 0, 119);
            moeWarmUp(ctx, 0);
            if (threadIdx.x == 0)
            {
                reinterpret_cast<int*>(ctx.smem + kSmemSmall + kSmallMoe)[kMoeWarmed] = 1;
            }
        }
    }
    if (GLM52_FLAGS(ctx) & kFlagCanaryH)
    {
        uint8_t const* a = arena<uint8_t const>(ctx, kArH) + 1536 * kq * 2;
        waitCanaries(
            ctx, kTok, [&](int t) { return a + t * kHidden * 2; }, 142);
    }
    uint4 b[kPer][2];
    pollB<kPer>(ctx, b,
        arena<uint8_t const>(ctx, (kSpecializedStaticFlags & kFlagHPrime) ? kArHp : kArH) + 1536 * kq * 2, kHidden,
        warp * kPer, 118);
    stampInput(ctx);
    waitBar(ctx, bar, 0, 119);
    float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
    for (int i = 0; i < kPer; ++i)
    {
#pragma unroll
        for (int t = 0; t < 2; ++t)
        {
            mmaChunkBf16(acc[t], ctx.smemBase + wOff + (t * 24 + warp * kPer + i) * kBf16Block, b[i]);
        }
    }
    float* out = reinterpret_cast<float*>(ctx.smem + kSmemRed + 8192);
    reduceWarps<2>(ctx, acc, out);
    float* sh = arena<float>(ctx, kArSh) + ((kq * 8 + ft) * 32) * kTok;
    st32(sh + threadIdx.x, lamport(__float_as_uint(out[threadIdx.x])));
}

// The MoE work cut: n pieces into kCtas runs by weight per CTA class (as the role table places the units): kCutWOs for
// the CTAs with a shared-expert unit ([kOsFirst, +kOsCtas): that unit runs before their first piece; with equal runs
// they ended the layer's MoE ~2 us after the others), kCutWSg for the shared gate_up CTAs ([kSgFirst, +kSgCtas)) and
// kCutWH1 for the H1 CTAs ([kH1Cta0, kCtas)): their routing starts after those units' ends (GLM52_ROUTER_LG == 2: the
// others route as soon as the reduced logits and H1's sums of squares are in), kCutW for the rest. The first piece of
// CTA c's run, and the CTA whose run holds piece g (the largest c whose run starts at or before g). OS 2 of 3 since the
// shared expert takes 1 / rms from the routing (kFlagRstdShared; 9:5 matched its four rstd polls). Valid weights keep
// every aligned cut's FC1 runs >= 0 for E 27-64 (gen_cut.py checks; kFc2ListsPacked and kFc2ListsCut come from it).
constexpr int kOsFirst = 100;
constexpr int kOsCtas = 48;
constexpr int kSgFirst = 65; // the shared gate_up units' CTAs (host build_roles: 65 + unit)
constexpr int kSgCtas = 32;
constexpr int kCutW = kFc2ListsCut[0];
constexpr int kCutWOs = kFc2ListsCut[1];
constexpr int kCutWSg = kFc2ListsCut[2];
constexpr int kCutWH1 = kFc2ListsCut[3];

// CTAs of [a, b) below c.
__host__ __device__ constexpr int cutSegment(int c, int a, int b)
{
    return c <= a ? 0 : c >= b ? b - a : c - a;
}

// The cut's weight of CTAs [0, c).
__host__ __device__ constexpr int moeCutWeight(int c)
{
    return kCutW
        * (cutSegment(c, 0, kSgFirst) + cutSegment(c, kSgFirst + kSgCtas, kOsFirst)
            + cutSegment(c, kOsFirst + kOsCtas, kH1Cta0))
        + kCutWSg * cutSegment(c, kSgFirst, kSgFirst + kSgCtas) + kCutWOs * cutSegment(c, kOsFirst, kOsFirst + kOsCtas)
        + kCutWH1 * cutSegment(c, kH1Cta0, kCtas);
}

constexpr int kCutTotal = moeCutWeight(kCtas);
static_assert(kSgFirst + kSgCtas <= kOsFirst && kOsFirst + kOsCtas <= kH1Cta0, "cut segments in CTA order");

__device__ __forceinline__ int moeRunStart(int c, int n)
{
    return moeCutWeight(c) * n / kCutTotal;
}

// (GLM52_LISTS_MODE < 2 only) the largest c whose run starts at or before piece g.
__device__ __forceinline__ int moeCtaOf(int g, int n)
{
    int lo = 0;
    int hi = kCtas - 1;
    while (lo < hi)
    {
        int const mid = (lo + hi + 1) >> 1;
        if (moeRunStart(mid, n) <= g)
        {
            lo = mid;
        }
        else
        {
            hi = mid - 1;
        }
    }
    return lo;
}

// The MoE work cut of E slots: CTA c's FC1 run [f1(c), f1(c + 1)) of the 16 E FC1 pieces and its FC2 run
// [f2(c), f2(c + 1)) of the 8 E FC2 pieces. With E >= 27 (aligned): slot s's 8 FC2 pieces split evenly over CTAs
// [s kCtas / E, (s + 1) kCtas / E) only, and every CTA fills its share of the joint (weighted) cut of all 24 E with FC1
// pieces. No CTA then holds FC2 pieces of two slots: a second FC2 group, with its own wait for partials, SwiGLU and
// quantization, ended such a CTA ~2 us after the median, the layer's last (-8 us per slice, mk-v93a); the totals keep
// the spread of a plain joint cut (checked for E 27-64: FC1 runs monotone). Fewer experts: separate weighted cuts of
// the FC1 and FC2 pieces (a joint cut's FC2 counts could go negative there).
struct MoeCut
{
    int E;
    int const* table; // aligned cut: the FC1 starts f1(c), c <= kCtas, in shared memory; else null

    // The aligned cut's FC2 start of CTA c: its slot s is the last whose first CTA s kCtas / E is at or before c.
    __device__ __forceinline__ static int alignedF2(int c, int E)
    {
        if (c >= kCtas)
        {
            return 8 * E;
        }
        int const s = min(((c + 1) * E - 1) / kCtas, E - 1);
        int const c0 = s * kCtas / E;
        return 8 * s + 8 * (c - c0) / ((s + 1) * kCtas / E - c0);
    }

    __device__ __forceinline__ int f1(int c) const
    {
        return table != nullptr ? table[c] : moeRunStart(c, 16 * E);
    }

    __device__ __forceinline__ int f2(int c) const
    {
        if (table != nullptr)
        {
            return moeRunStart(c, 24 * E) - table[c];
        }
        return E >= 27 ? moeRunStart(c, 24 * E) - moeRunStart(c, 16 * E) : moeRunStart(c, 8 * E);
    }

    // The CTA whose FC1 run holds FC1 piece g (the last CTA starting at or before it).
    __device__ __forceinline__ int ctaOfF1(int g) const
    {
        if (table == nullptr)
        {
            return moeCtaOf(g, 16 * E);
        }
        int lo = 0;
        int hi = kCtas - 1;
        while (lo < hi)
        {
            int const mid = (lo + hi + 1) >> 1;
            if (table[mid] <= g)
            {
                lo = mid;
            }
            else
            {
                hi = mid - 1;
            }
        }
        return lo;
    }
};

#if GLM52_LISTS_MODE >= 1
// Lists for E1..64, each expert's two FC1 tiles. Header4 bits, up to8 ascending9-bit(CTA,run) codes.
// Index E*(E-1)+2*slot+tile;3 aligned32-bit loads per list, no runtime divisions or binary searches.
static_assert(kCtas == 212 && kOsFirst == 100 && kOsCtas == 48 && kSgFirst == 65 && kSgCtas == 32 && kH1Cta0 == 180,
    "regenerate packed FC2 lists for a new cut (gen_cut.py: its CTA classes)");

__device__ __forceinline__ void loadFc2PartialList(int E, int slot, int tile, uint32_t* list)
{
    int const index = E * (E - 1) + 2 * slot + tile;
    uint32_t const a = kFc2ListsPacked[index][0];
    uint32_t const b = kFc2ListsPacked[index][1];
    uint32_t const c = kFc2ListsPacked[index][2];
    uint32_t const nPart = a & 15u;
#pragma unroll
    for (int q = 0; q < 8; ++q)
    {
        if (static_cast<uint32_t>(q) < nPart)
        {
            int const bit = 4 + 9 * q;
            int const word = bit >> 5;
            int const shift = bit & 31;
            uint32_t const lo = word == 0 ? a : word == 1 ? b : c;
            uint32_t const hi = word == 0 ? b : c;
            uint32_t code = lo >> shift;
            if (shift > 23)
            {
                code |= hi << (32 - shift);
            }
            list[1 + q] = code & 511u;
        }
    }
    list[0] = nPart;
}
#endif

// kFlagFc2Direct: a lane's 32 bytes (at base in run 0's P1 block, 4 KiB per run) of each FC1 partial run of its FC2
// tile (list: the count, then the (CTA, run) codes in ascending CTA order), summed into sum[8], polled in one round of
// kN >= count runs: all loads in flight together, then again for the vectors not complete. kN = 8 for a group with a
// tile of 5-8 runs (the OS region's slots at cut weight 2, 7-11 FC1 contributors per slot): in rounds of four the
// second round's loads left only after the first four runs were in, and the latest run is mostly a first-listed one
// (the lowest CTA ends the tile with its second run), one more L2 round trip on the slot's hand-off (+0.3 us stamped vs
// 4 runs). The sums are taken in every turn of the spin from that turn's values (the empty asm keeps them in the loop),
// so that their code is warm when the last run lands (a cold sum after the loop cost ~0.1 us per waiting group). Runs
// past the count stay +0: full, and adding +0 to the +0-started sum is exact, so these are the same q-by-q FP32 adds as
// summing the listed runs.
template <int kN>
__device__ __forceinline__ void pollFc2Runs(
    Ctx const& ctx, uint8_t const* base, uint32_t const* list, int nPart, float (&sum)[8], int slot)
{
    uint4 w[kN][2];
#pragma unroll
    for (int q = 0; q < kN; ++q)
    {
        bool const use = q < nPart;
        uint8_t const* const src = base + static_cast<size_t>(use ? list[1 + q] : 0u) * (128 * kTok * 4);
        w[q][0] = use ? ldPoll128(src) : make_uint4(0, 0, 0, 0);
        w[q][1] = use ? ldPoll128(src + 16) : make_uint4(0, 0, 0, 0);
    }
    uint64_t t0 = 0;
    while (true)
    {
        bool ok = true;
#pragma unroll
        for (int q = 0; q < kN; ++q)
        {
            ok &= full4(w[q][0]) && full4(w[q][1]);
        }
        bool const done = warpAll(ok);
#pragma unroll
        for (int e = 0; e < 8; ++e)
        {
            sum[e] = 0.f;
        }
#pragma unroll
        for (int q = 0; q < kN; ++q)
        {
#pragma unroll
            for (int v = 0; v < 2; ++v)
            {
                sum[4 * v] += __uint_as_float(w[q][v].x);
                sum[4 * v + 1] += __uint_as_float(w[q][v].y);
                sum[4 * v + 2] += __uint_as_float(w[q][v].z);
                sum[4 * v + 3] += __uint_as_float(w[q][v].w);
            }
        }
#pragma unroll
        for (int e = 0; e < 8; ++e)
        {
            asm volatile("" : "+f"(sum[e]));
        }
        if (done)
        {
            break;
        }
        spinCheck(ctx, t0, 124, slot, kN);
#pragma unroll
        for (int q = 0; q < kN; ++q)
        {
            uint8_t const* const src = base + static_cast<size_t>(q < nPart ? list[1 + q] : 0u) * (128 * kTok * 4);
#pragma unroll
            for (int v = 0; v < 2; ++v)
            {
                if (!full4(w[q][v]))
                {
                    w[q][v] = ldPoll128(src + 16 * v);
                }
            }
        }
    }
}

// The source of MoE piece `piece` of this CTA (its FC1 pieces, then its FC2 pieces): 48 KiB of weights then 6 KiB of
// scales, packed together on the host ([expert][tile][8 pieces] for FC1, [expert][8 eighths] for FC2, kPieceBytes
// each).
__device__ __forceinline__ int moePieceSources(
    Ctx const& ctx, int const* smemMoe, int piece, uint8_t const* (&src)[4], uint32_t (&dst)[4], uint32_t (&n)[4])
{
    int const n1 = smemMoe[kMoeF1B] - smemMoe[kMoeF1A];
    dst[0] = 0;
    n[0] = kPieceBytes;
    if (piece < n1)
    {
        int const g = smemMoe[kMoeF1A] + piece;
        size_t const tile = static_cast<size_t>(smemMoe[kMoeSlotExpert + (g >> 4)]) * 2 + ((g >> 3) & 1);
        src[0] = ptr<uint8_t const>(ctx, kPW13) + (tile * 8 + (g & 7)) * kPieceBytes;
        return 1;
    }
    int const j = smemMoe[kMoeF2A] + piece - n1;
    src[0] = ptr<uint8_t const>(ctx, kPW2)
        + (static_cast<size_t>(smemMoe[kMoeSlotExpert + (j >> 3)]) * 8 + (j & 7)) * kPieceBytes;
    return 1;
}

// FC1 MMAs of one piece (warp 1): the 12 K steps' weight scales into TMEM (sfa), then the 12 K steps of 128 rows x 8
// tokens into D (the first one accumulating only with accFirst), committed on bar. hbq: the token operand of the
// piece's first K block; sfb: its first K step's token scales.
__device__ __noinline__ void fc1PieceMma(
    uint32_t tmem, uint32_t base, uint32_t hbq, uint32_t sfa, uint32_t sfb, uint32_t accFirst, uint32_t bar)
{
    tcFenceAfter();
    if (electOne())
    {
#pragma unroll
        for (int st = 0; st < 12; ++st)
        {
            utccp1(sfa + 4 * st, base + kPieceW + 512 * st);
        }
#pragma unroll
        for (int st = 0; st < 12; ++st)
        {
            int const kb = st >> 1;
            int const ks = st & 1;
            mmaNvf4(tmem + kTmemD, base + kb * 8192 + 32 * ks, hbq + kb * 512 + 32 * ks, sfa + 4 * st, sfb + 2 * st,
                (accFirst != 0 || st > 0) ? 1u : 0u);
        }
        commit1(bar);
    }
    __syncwarp();
}

// FC2 MMAs of one piece (warp 1): the token scales (fc2Sfb) and the weight scales into TMEM, then 6 tiles x 2 K steps
// into accumulator columns [d + 8 ti, +8), committed on bar.
__device__ __noinline__ void fc2PieceMma(
    uint32_t tmem, uint32_t base, uint32_t fc2B, uint32_t fc2Sfb, uint32_t sfa, uint32_t bar, uint32_t d = kTmemD)
{
    tcFenceAfter();
    if (electOne())
    {
        utccp1(tmem + kTmemSfb2, fc2Sfb);
#pragma unroll
        for (int st = 0; st < 12; ++st)
        {
            utccp1(sfa + 4 * st, base + kPieceW + 512 * st);
        }
#pragma unroll
        for (int st = 0; st < 12; ++st)
        {
            int const ti = st >> 1;
            int const ks = st & 1;
            mmaNvf4(tmem + d + 8 * ti, base + ti * 8192 + 32 * ks, fc2B + 32 * ks, sfa + 4 * st,
                tmem + kTmemSfb2 + 2 * ks, static_cast<uint32_t>(ks));
        }
        commit1(bar);
    }
    __syncwarp();
}

// Requests MoE piece `piece` of this CTA into a ring slot (thread 0 only). The slot was last read by the tensor cores
// (async proxy) or, for the first pieces, before the stage's proxy fence.
__device__ __noinline__ void issueMoePiece(Ctx const& ctx, int const* smemMoe, int piece, uint32_t slot, uint32_t bar)
{
    uint8_t const* src[4];
    uint32_t dst[4];
    uint32_t n[4];
    int const parts = moePieceSources(ctx, smemMoe, piece, src, dst, n);
    mbarExpectTx(bar, kPieceBytes);
    if (GLM52_FLAGS(ctx) & kFlagEvictFirst)
    {
        uint64_t const policy = policyEvictFirst();
        for (int i = 0; i < parts; ++i)
        {
            bulkCopyHint(slot + dst[i], src[i], n[i], bar, policy);
        }
    }
    else
    {
        for (int i = 0; i < parts; ++i)
        {
            bulkCopy(slot + dst[i], src[i], n[i], bar);
        }
    }
}

// Prefetches MoE piece `piece` of this CTA into L2 (one lane).
__device__ __noinline__ void prefetchMoePiece(Ctx const& ctx, int const* smemMoe, int piece)
{
    uint8_t const* src[4];
    uint32_t dst[4];
    uint32_t n[4];
    int const parts = moePieceSources(ctx, smemMoe, piece, src, dst, n);
    bool const hint = (GLM52_FLAGS(ctx) & kFlagEvictFirst) && !(GLM52_FLAGS(ctx) & kFlagMoeL2Normal);
    uint64_t const policy = hint ? policyEvictFirst() : 0;
    for (int i = 0; i < parts; ++i)
    {
        if (hint)
        {
            prefetchBulkL2Hint(src[i], n[i], policy);
        }
        else
        {
            prefetchBulkL2(src[i], n[i]);
        }
    }
}

// NVFP4(h) into shared memory as the FC1 token operand: E2M1 in 48 SWIZZLE_64B atoms ([K block][8 rows][64 B]) and
// the scales as they are ([8][384]); polled (thread t: vectors t, t + 256, ...).
constexpr int kHVecE = kHfp4Bytes / 16;
constexpr int kHVec = kHVecE + kHsfBytes / 16;
constexpr int kHN = (kHVec + kThreads - 1) / kThreads;

// The thread's 16-byte vectors of NVFP4(h) and its scales (vector tid + kThreads i), loaded without waiting.
__device__ __forceinline__ void loadH(Ctx const& ctx, uint4 (&v)[kHN])
{
    uint8_t const* src = arena<uint8_t const>(ctx, kArHfp4);
#pragma unroll
    for (int i = 0; i < kHN; ++i)
    {
        int const j = threadIdx.x + kThreads * i;
        v[i] = j < kHVec ? ldPoll128(src + 16 * j) : make_uint4(0, 0, 0, 0);
    }
}

// The loaded vectors complete (polled again while empty), then into shared memory (h swizzled for the FC1 MMA).
__device__ __forceinline__ void finishH(Ctx const& ctx, uint4 (&v)[kHN], uint8_t* hb, uint8_t* hsf)
{
    constexpr int kVecE = kHVecE;
    constexpr int kVec = kHVec;
    constexpr int kN = kHN;
    uint8_t const* src = arena<uint8_t const>(ctx, kArHfp4);
    int const tid = threadIdx.x;
    uint64_t t0 = 0;
    while (true)
    {
        bool ok = true;
#pragma unroll
        for (int i = 0; i < kN; ++i)
        {
            ok &= full4(v[i]);
        }
        if (warpAll(ok))
        {
            break;
        }
        spinCheck(ctx, t0, 131, 0, 0);
#pragma unroll
        for (int i = 0; i < kN; ++i)
        {
            if (!full4(v[i]))
            {
                v[i] = ldPoll128(src + 16 * (tid + kThreads * i));
            }
        }
    }
#pragma unroll
    for (int i = 0; i < kN; ++i)
    {
        int const j = tid + kThreads * i;
        if (j < kVecE)
        {
            int const t = j / 192;
            int const c16 = j % 192;
            *reinterpret_cast<uint4*>(hb + (c16 >> 2) * 512 + t * 64 + (((c16 & 3) ^ ((t >> 1) & 3)) << 4)) = v[i];
        }
        else if (j < kVec)
        {
            *reinterpret_cast<uint4*>(hsf + 16 * (j - kVecE)) = v[i];
        }
    }
}

__device__ __noinline__ void pollCopyH(Ctx const& ctx, uint8_t* hb, uint8_t* hsf)
{
    uint4 v[kHN];
    loadH(ctx, v);
    finishH(ctx, v, hb, hsf);
}

// The FC1 token scales of K step pairs [24 h, +24) (h = 0, 1) into TMEM columns kTmemSfb + 96 h + 4 pair: per pair of
// steps 32 rows x 16 B staged in shared memory (12 KiB at stage), row t < 8 = (scales of the even step, 0, scales of
// the odd step, 0); warp 1 copies them (all threads call; the caller waits for the copies before stage is rewritten).
__device__ __forceinline__ void stageTokenScales(
    Ctx const& ctx, uint8_t const* hsf, uint8_t* stage, uint32_t tmem, int h)
{
    for (int v = threadIdx.x; v < 24 * 32; v += kThreads)
    {
        int const pair = 24 * h + v / 32;
        int const row = v % 32;
        uint4 val = make_uint4(0, 0, 0, 0);
        if (row < kTok)
        {
            uint32_t const* sf = reinterpret_cast<uint32_t const*>(hsf + row * 384) + 2 * pair;
            val = make_uint4(sf[0], 0u, sf[1], 0u);
        }
        *reinterpret_cast<uint4*>(stage + 16 * v) = val;
    }
    fenceProxyAsyncSmem();
    __syncthreads();
    if ((threadIdx.x >> 5) == 1)
    {
        tcFenceAfter();
        if (electOne())
        {
            for (int pair = 0; pair < 24; ++pair)
            {
                utccp1(tmem + kTmemSfb + 96 * h + 4 * pair, smemU32(stage + 512 * pair));
            }
        }
        __syncwarp();
    }
}

// NVFP4(h) into shared memory (hb, hsf) and the token scales of all 96 K steps into TMEM, in two halves through 12 KiB
// of the reduction area (free once the routing is done; two phases of barMma from mmaPhase on, the next parity
// returned).
__device__ __noinline__ uint32_t moeStageH(Ctx const& ctx, uint8_t* hb, uint8_t* hsf, uint32_t tmem, uint32_t barMma,
    uint32_t mmaPhase, bool staged, bool scalesStaged
#if GLM52_IO_MODE == 2 || GLM52_IO_MODE == 3
    ,
    bool scaleWork
#endif
)
{
    if (!staged)
    {
        pollCopyH(ctx, hb, hsf);
    }
    __syncthreads();
    subStamp(ctx, 88);
#if GLM52_IO_MODE == 2 || GLM52_IO_MODE == 3
    // Zero-piece CTAs must not issue asynchronous copies: no later MMA completion
    // would drain them before TMEM deallocation. scaleWork is CTA-uniform.
    if (!scaleWork)
    {
        return mmaPhase;
    }
    // Row t < 8 of pair p: (scale word 2 p of token t, 0, word 2 p + 1, 0); 48 pairs x 128 B.
    uint8_t* stage = ctx.smem + kSmemRed + 2048;
    for (int v = threadIdx.x; v < 48 * kTok; v += kThreads)
    {
        int const pair = v / kTok;
        int const row = v % kTok;
        uint32_t const* sf = reinterpret_cast<uint32_t const*>(hsf + row * 384) + 2 * pair;
        *reinterpret_cast<uint4*>(stage + 16 * v) = make_uint4(sf[0], 0u, sf[1], 0u);
    }
    fenceProxyAsyncSmem();
    __syncthreads();
    if ((threadIdx.x >> 5) == 1)
    {
        tcFenceAfter();
        if (electOne())
        {
            for (int pair = 0; pair < 48; ++pair)
            {
                utccp1Rows8(tmem + kTmemSfb + 4 * pair, smemU32(stage + 128 * pair));
            }
        }
        __syncwarp();
    }
    return mmaPhase;
#else
    for (int h = 0; h < (scalesStaged ? 0 : 2); ++h)
    {
        stageTokenScales(ctx, hsf, ctx.smem + kSmemRed + 2048, tmem, h);
        if ((threadIdx.x >> 5) == 1)
        {
            commitElect(barMma);
        }
        waitBar(ctx, barMma, mmaPhase, 136);
        mmaPhase ^= 1;
        __syncthreads();
        if (h == 0)
        {
            subStamp(ctx, 89);
        }
    }
    return mmaPhase;
#endif
}

// Routed experts on the tensor cores. The CTA routes the 8 tokens itself, then runs its FC1 run (the pieces' MMAs
// accumulate per (slot, tile); each run's FP32 partial D[128 rows][8 tokens] goes into the arena) and its FC2 run
// (per slot: the slot's FC1 partials summed, SwiGLU + NVFP4 = the token operand, then per eighth 6 tiles of 128 rows x
// K 128, the rows of the tokens routed to the expert into fc2out). Weights stream through a ring of three 54 KiB
// slots; warp 1 issues the scale copies and the MMAs, warps 0-3 read the accumulators.
__device__ __noinline__ void stageOwnerShared(Ctx const& ctx, int J, int const* smemMoe = nullptr);

// The routed experts of the 8 tokens on this CTA's share of the work list. ownerShared >= 0: the CTA's shared-expert
// unit runs between the first pieces' requests and their arrival (its inputs are in by then; before the routing it
// delayed this CTA's requests).
// kFlagWarmMoe: one FC1 and one FC2 MMA batch on the stale contents of the MoE region's ring slots 0 and 1 (the MMA
// code and pipe warm before the real pieces; worth ~1.5 us per layer): D and the token-scale columns are rewritten
// before use (the first K step of every real tile does not accumulate; the token scales are staged after it). The
// barrier completes two phases, so its parity is 0 again. Every thread calls it.
__device__ __noinline__ void moeWarmUp(Ctx const& ctx, uint32_t wOff)
{
    int const warp = threadIdx.x >> 5;
    constexpr uint32_t kRegion2 = kRingSlots * kPieceBytes;
    uint32_t const hb = ctx.smemBase + wOff + kRegion2;
    uint32_t const fc2B = hb + 32768;
    uint32_t const fc2Sfb = fc2B + 512;
    uint32_t const barMma = ctx.smemBase + kSmemSmall + kSmallBars + (kMaxRoles + 8) * 8 + kRingSlots * 8;
    uint32_t const tmem = *reinterpret_cast<uint32_t const*>(ctx.smem + kSmemSmall + kSmallScalars + 4 * 120);
    fenceProxyAsyncSmem();
    tcFenceBefore();
    __syncthreads();
    tcFenceAfter();
    if (warp == 1)
    {
        fc1PieceMma(tmem, ctx.smemBase + wOff, hb, tmem + kTmemSfa, tmem + kTmemSfb, 0u, barMma);
    }
    waitBar(ctx, barMma, 0, 134);
    __syncthreads();
    tcFenceAfter();
    if (warp == 1)
    {
        fc2PieceMma(tmem, ctx.smemBase + wOff + kPieceBytes, fc2B, fc2Sfb, tmem + kTmemSfa + 48, barMma);
    }
    waitBar(ctx, barMma, 1, 135);
    tcFenceBefore();
    __syncthreads();
    tcFenceAfter();
}

__device__ __noinline__ void stageMoe(Ctx const& ctx, int ownerShared, uint32_t wOff)
{
    int const tid = threadIdx.x;
    int const warp = tid >> 5;
    int const lane = tid & 31;
    int const cta = blockIdx.x;
    constexpr uint32_t kRegion2 = kRingSlots * kPieceBytes; // 32 KiB: h during FC1, the slot's FC1 partials in FC2
    uint8_t* region2 = ctx.smem + wOff + kRegion2;
    uint32_t const hb = ctx.smemBase + wOff + kRegion2;
    uint8_t* hsf = region2 + kHfp4Bytes;
    uint8_t* fc2B = region2 + 32768; // FC2 token operand: one atom, 8 rows x 64 B
    uint8_t* fc2Sfb = fc2B + 512;    // its scales: 32 rows x 16 B (tcgen05.cp source)
    uint32_t const barRing0 = ctx.smemBase + kSmemSmall + kSmallBars + (kMaxRoles + 8) * 8;
    uint32_t const barMma = barRing0 + kRingSlots * 8;
    uint32_t* sTmem = reinterpret_cast<uint32_t*>(ctx.smem + kSmemSmall + kSmallScalars + 4 * 120);
    int* smemMoe = reinterpret_cast<int*>(ctx.smem + kSmemSmall + kSmallMoe);
    if ((GLM52_FLAGS(ctx) & kFlagOsPrefetch) && ownerShared >= 0 && tid == 0)
    {
        // Its 32 KiB of down weights go out ahead of the expert pieces (behind them its loads took ~4 us and its
        // first FC1 MMA came 1.8 us after the other CTAs').
        prefetchBulkL2(
            ptr<uint8_t const>(ctx, kPWD) + static_cast<size_t>(ownerShared) * 8 * 2 * kBf16Block, 8 * 2 * kBf16Block);
    }
    bool const canary = (GLM52_FLAGS(ctx) & kFlagCanaryMoe) != 0;
    if (canary)
    {
        // The first logit word of every router unit (experts [16 et, +16), K quarter kq, token 0).
        uint8_t const* a = arena<uint8_t const>(ctx, kArLogit);
        waitCanaries(
            ctx, 64, [&](int u) { return a + ((u % 4) * kTok * kExperts + 16 * (u / 4)) * 4; }, 144);
    }
    // TMEM (taken at kernel start) and the proxy fence for the first pieces' copies into shared memory used before.
    fenceProxyAsyncSmem();
    tcFenceBefore();
    __syncthreads();
    tcFenceAfter();
    uint32_t const tmem = *sTmem;
    uint32_t mmaPhase = 0;
    // The warm-up (kFlagWarmMoe), unless this CTA's H1 unit ran it in its hop wait (kFlagWarmInHop: CTAs 180-211) or
    // another earlier unit did (kMoeWarmed: GLM52_ROUTER_LG's logit units, and the shared gate_up with mode 2).
    if ((GLM52_FLAGS(ctx) & kFlagWarmMoe) && !((kSpecializedStaticFlags & kFlagWarmInHop) != 0 && cta >= kH1Cta0)
        && !(kRouterLg != 0 && smemMoe[kMoeWarmed] != 0))
    {
        moeWarmUp(ctx, wOff);
    }
    // h's loads out now (h is in since the attention hop): they land while the routing runs (after the first pieces'
    // requests, h's staging held the first MMA back ~1.2 us: mk-v154a 772.15 vs 781.9 us).
    constexpr bool hEarly = true;
    uint4 hv[kHN];
    if (hEarly)
    {
        loadH(ctx, hv);
    }
    // The token scales early (vectors 1536 + tid of h's loads, tid < 192: row tid / 24, K step pairs 2 (tid % 24) and
    // + 1) into TMEM, staged in ring slot 2: off (in two halves after the requests instead).
    constexpr bool scalesEarly = false;
    if (scalesEarly)
    {
        uint8_t* stage = ctx.smem + wOff + 2 * kPieceBytes;
        for (int v = tid; v < 48 * 24; v += kThreads)
        {
            *reinterpret_cast<uint4*>(stage + 512 * (v / 24) + 16 * (kTok + v % 24)) = make_uint4(0, 0, 0, 0);
        }
        if (tid < kHVec - kHVecE)
        {
            uint4& sv = hv[kHN - 1];
            uint8_t const* src = arena<uint8_t const>(ctx, kArHfp4) + 16 * (kHVecE + tid);
            uint64_t t0 = 0;
            while (!full4(sv))
            {
                spinCheck(ctx, t0, 138, tid, 0);
                sv = ldPoll128(src);
            }
            int const row = tid / 24;
            int const pair = 2 * (tid % 24);
            *reinterpret_cast<uint4*>(stage + 512 * pair + 16 * row) = make_uint4(sv.x, 0u, sv.y, 0u);
            *reinterpret_cast<uint4*>(stage + 512 * (pair + 1) + 16 * row) = make_uint4(sv.z, 0u, sv.w, 0u);
        }
        fenceProxyAsyncSmem();
        __syncthreads();
        if (warp == 1)
        {
            tcFenceAfter();
            if (electOne())
            {
                for (int pair = 0; pair < 48; ++pair)
                {
                    utccp1(tmem + kTmemSfb + 4 * pair, smemU32(stage + 512 * pair));
                }
            }
            __syncwarp();
            commitElect(barMma);
        }
        waitBar(ctx, barMma, mmaPhase, 139);
        mmaPhase ^= 1;
        __syncthreads();
        tcFenceAfter();
    }
    // This CTA's run bounds f1(c), f1(c + 1), f2(c), f2(c + 1) for every expert count E = 1 + tid / 4, while the
    // router finishes (the cut after the routing took ~0.7 us before the first request: -1.6 us per slice, mk-v153a).
    int* const sPre = reinterpret_cast<int*>(ctx.smem + kSmemRed + 6144);
    {
        int const e = 1 + (tid >> 2);
        int const c = cta + (tid & 1);
        int v;
        if (e >= 27)
        {
            int const a2 = MoeCut::alignedF2(c, e);
            v = (tid & 2) ? a2 : moeRunStart(c, 24 * e) - a2;
        }
        else
        {
            v = (tid & 2) ? moeRunStart(c, 8 * e) : moeRunStart(c, 16 * e);
        }
        sPre[tid] = v;
    }
    // The routing first: its requests start the layer's longest stream. h and the token scales are staged while the
    // first pieces land (staged before the routing, they outlasted the router's remaining time and held the requests
    // back by 1-2 us).
    subStamp(ctx, 12);
    routeTokens(ctx, smemMoe, cta == 0);
    subStamp(ctx, 13);
    int const E = smemMoe[kMoeE];
    // The aligned cut's FC1 starts, one per thread into a table (its divisions and the partials' lookups otherwise
    // held the CTA ~9 us), in the reduction area's routing-free middle.
    int* const sCut = reinterpret_cast<int*>(ctx.smem + kSmemRed + 4096);
    bool const aligned = E >= 27;
    [[maybe_unused]] MoeCut const cut{E, aligned ? sCut : nullptr};
    int const f1a = sPre[4 * (E - 1)];
    int const f1b = sPre[4 * (E - 1) + 1];
    int const f2a = sPre[4 * (E - 1) + 2];
    int const f2b = sPre[4 * (E - 1) + 3];
    int const nPieces = (f1b - f1a) + (f2b - f2a);
    {
        // The slots' token masks into the list (the routing's scratch is reused by the shared expert).
        uint32_t const* sSlotMask = reinterpret_cast<uint32_t const*>(ctx.smem + kSmemRed + 1024);
        if (tid < 64)
        {
            reinterpret_cast<uint8_t*>(smemMoe + kMoeSlotMask)[tid]
                = static_cast<uint8_t>(tid < E ? sSlotMask[tid] : 0u);
        }
        if (tid == 0)
        {
            smemMoe[kMoePieces] = nPieces;
            smemMoe[kMoeF1A] = f1a;
            smemMoe[kMoeF1B] = f1b;
            smemMoe[kMoeF2A] = f2a;
            smemMoe[kMoeF2B] = f2b;
            subStamp(ctx, 40);
        }
    }
    __syncthreads();
    // Piece 0 first, pieces 1-2 after the FC2 groups' inputs (with all three at once, every CTA's piece 0 landed only
    // with the end of the whole 35 MB burst, ~4.4 us after the request: mk-v140a 793.35 vs 783.95 us per slice; after
    // h is staged, ~1 us later still, +9.1 us, mk-v142a); kFlagMoeAllFirst: all three at once.
    bool const stagger = !(GLM52_FLAGS(ctx) & kFlagMoeAllFirst);
    if (lane == 0 && warp < (stagger ? 1 : kRingSlots) && warp < nPieces)
    {
        issueMoePiece(ctx, smemMoe, warp, ctx.smemBase + wOff + warp * kPieceBytes, barRing0 + 8 * warp);
    }
    if (tid == 0)
    {
        subStamp(ctx, 41);
    }
    fenceProxyAsyncSmem();
#if GLM52_LISTS_MODE < 2
    if (aligned)
    {
        // The table for the FC2 groups' partial lists below.
        for (int c = tid; c <= kCtas; c += kThreads)
        {
            sCut[c] = moeRunStart(c, 24 * E) - MoeCut::alignedF2(c, E);
        }
    }
#endif
    __syncthreads();
    // The FC2 groups' inputs now, off their path (each took an L2 round trip and a loop of divisions there): per group
    // g (the CTA's g-th FC2 slot, at most 4) and tile, the count then the (CTA, run) of the FC1 partials, at sFc2 + 9
    // (2 g + tile); the slot's expert scales alpha1 / alpha2 at sFc2[72 + g] / [76 + g].
    uint32_t* const sFc2 = reinterpret_cast<uint32_t*>(ctx.smem + kSmemRed + 14336);
    {
        int const slot0 = f2a >> 3;
        int const nGroups = f2b > f2a ? ((f2b - 1) >> 3) - slot0 + 1 : 0;
        if (nGroups > 4)
        {
            fail(ctx, 137, nGroups, f2b - f2a);
        }
        if (tid < 2 * nGroups)
        {
            int const tile = tid & 1;
            int const g0 = 16 * (slot0 + (tid >> 1)) + 8 * tile;
            uint32_t* const list = sFc2 + 9 * tid;
#if GLM52_LISTS_MODE >= 1
            loadFc2PartialList(E, slot0 + (tid >> 1), tile, list);
#else
            int nPart = 0;
            int const cEnd = cut.ctaOfF1(g0 + 7);
            for (int c = cut.ctaOfF1(g0); c <= cEnd && nPart < 8; ++c)
            {
                int const a = cut.f1(c);
                if (a < cut.f1(c + 1))
                {
                    list[1 + nPart++] = static_cast<uint32_t>(c * 2 + ((a >> 3) == (g0 >> 3) ? 0 : 1));
                }
            }
            list[0] = static_cast<uint32_t>(nPart);
#endif
        }
        else if (tid >= 32 && tid < 32 + nGroups)
        {
            int const expert = smemMoe[kMoeSlotExpert + slot0 + tid - 32];
            sFc2[72 + tid - 32] = __float_as_uint(__ldg(ptr<float const>(ctx, kPAlpha1) + expert));
            sFc2[76 + tid - 32] = __float_as_uint(__ldg(ptr<float const>(ctx, kPAlpha2) + expert));
        }
    }
    if (stagger && lane == 0 && warp >= 1 && warp < kRingSlots && warp < nPieces)
    {
        issueMoePiece(ctx, smemMoe, warp, ctx.smemBase + wOff + warp * kPieceBytes, barRing0 + 8 * warp);
    }
    // kFlagMoeL2: the rest of the CTA's pieces into L2 behind its ring's requests (one lane per piece), so the ring's
    // refills come from L2.
    if (GLM52_FLAGS(ctx) & kFlagMoeL2)
    {
        __syncthreads();
        if (lane == 0 && kRingSlots + warp < nPieces)
        {
            prefetchMoePiece(ctx, smemMoe, kRingSlots + warp);
        }
    }

    if (GLM52_STAMPS(ctx) != nullptr && tid == 0)
    {
        subStamp(ctx, 42);
        GLM52_STAMPS(ctx)[kMaxRoles * 4 + 3] = globalTimer();
    }
    if (hEarly)
    {
        finishH(ctx, hv, region2, hsf);
    }
#if GLM52_IO_MODE == 2 || GLM52_IO_MODE == 3
    mmaPhase = moeStageH(ctx, region2, hsf, tmem, barMma, mmaPhase, hEarly, scalesEarly, nPieces > 0);
#else
    mmaPhase = moeStageH(ctx, region2, hsf, tmem, barMma, mmaPhase, hEarly, scalesEarly);
#endif
    stampInput(ctx);
#if GLM52_OWNER_FC1_OVERLAP
    // Only the first FC1 piece: its async TCGEN operation can run during the
    // unchanged shared-expert activation/HMMA. No ring refill or new phase is
    // issued until the original FC1 completion wait and CTA join below.
    bool const ownerFirst = ownerShared >= 0 && f1a < f1b;
    if (ownerFirst)
    {
        subStamp(ctx, 80);
        waitBar(ctx, barRing0, 0, 123);
        subStamp(ctx, 16);
        if (warp == 1)
        {
            int const k0 = f1a & 7;
            fc1PieceMma(
                tmem, ctx.smemBase + wOff, hb + 6 * k0 * 512, tmem + kTmemSfa, tmem + kTmemSfb + 24 * k0, 0u, barMma);
        }
    }
#endif
    if (ownerShared >= 0)
    {
        constexpr bool kRstdShared
            = (kSpecializedStaticFlags & kFlagHPrime) && (kSpecializedStaticFlags & kFlagRstdShared);
        stageOwnerShared(ctx, ownerShared, kRstdShared ? smemMoe : nullptr);
        __syncthreads();
    }
    uint8_t* p1Base = arena<uint8_t>(ctx, kArP1);
    bool const fc2Direct = (GLM52_FLAGS(ctx) & kFlagFc2Direct) != 0;
#if GLM52_FLAGS_MODE == 3 || GLM52_FLAGS_MODE == 5
    bool const sparseP1 = fc2Direct && (GLM52_FLAGS(ctx) & kFlagSparseP1);
#endif
    int p = 0;
    // FC1: a run of pieces of one (slot, tile) accumulates in D; at its end the partial goes to P1[cta][run] (the
    // CTA's first or second run).
    int run = 0;
    for (int g = f1a; g < f1b; ++g, ++p)
    {
        int const k = g & 7;
        bool const first = g == f1a || k == 0;
        bool const last = g + 1 == f1b || k == 7;
        int const rs = p % kRingSlots;
        uint32_t const base = ctx.smemBase + wOff + rs * kPieceBytes;
#if GLM52_OWNER_FC1_OVERLAP
        if (!(ownerFirst && p == 0))
        {
#endif
            if (p < 8)
            {
                subStamp(ctx, 80 + p);
            }
            waitBar(ctx, barRing0 + 8 * rs, (p / kRingSlots) & 1, 123);
            if (p < 8)
            {
                subStamp(ctx, 16 + 2 * p);
            }
            if (warp == 1)
            {
                fc1PieceMma(tmem, base, hb + 6 * k * 512, tmem + kTmemSfa + 48 * rs, tmem + kTmemSfb + 24 * k,
                    first ? 0u : 1u, barMma);
            }
#if GLM52_OWNER_FC1_OVERLAP
        }
#endif
        waitBar(ctx, barMma, mmaPhase, 132);
        mmaPhase ^= 1;
        // Every thread has seen this phase before the next commit can complete the barrier's next one (a waiter
        // that slept through both would wait on the parity of a phase that never comes).
        __syncthreads();
        tcFenceAfter();
        if (p < 8)
        {
            subStamp(ctx, 17 + 2 * p);
        }
        // The slot's MMAs are complete: request the piece three ahead into it.
        if (tid == 0 && p + kRingSlots < nPieces)
        {
            issueMoePiece(ctx, smemMoe, p + kRingSlots, base, barRing0 + 8 * rs);
        }
        if (last)
        {
            // The run's partial: warp w < 4 = rows [32 w, +32) of the tile, 8 tokens per row.
            if (warp < 4)
            {
                float v[8];
                tmemLd<8>(tmem + kTmemD + (static_cast<uint32_t>(32 * warp) << 16), v);
                if (fc2Direct)
                {
                    // Old row = 32 b + 16 j + m + 8 gate, feature = 2 m + j. For each token the warp's scalar
                    // stores cover one contiguous 128-byte block, in [16 features][up, gate] order.
                    int const feature = 2 * (lane & 7) + (lane >> 4);
                    int const gate = (lane >> 3) & 1;
                    uint8_t* dst = p1Base + static_cast<size_t>(cta * 2 + run) * 128 * kTok * 4;
#if GLM52_FLAGS_MODE == 3 || GLM52_FLAGS_MODE == 5
                    int const tokenMask
                        = sparseP1 ? reinterpret_cast<uint8_t const*>(smemMoe + kMoeSlotMask)[g >> 4] : 255;
#endif
#pragma unroll
                    for (int t = 0; t < kTok; ++t)
                    {
#if GLM52_FLAGS_MODE == 3 || GLM52_FLAGS_MODE == 5
                        if (!((tokenMask >> t) & 1))
                        {
                            continue; // every reader of this run uses the same expert slot mask
                        }
#endif
                        st32(dst + 4 * (t * 128 + 32 * warp + 2 * feature + gate), lamport(__float_as_uint(v[t])));
                    }
                }
                else
                {
                    uint8_t* dst = p1Base + ((static_cast<size_t>(cta * 2 + run) * 128 + 32 * warp + lane) * kTok) * 4;
                    st128(dst,
                        make_uint4(lamport(__float_as_uint(v[0])), lamport(__float_as_uint(v[1])),
                            lamport(__float_as_uint(v[2])), lamport(__float_as_uint(v[3]))));
                    st128(dst + 16,
                        make_uint4(lamport(__float_as_uint(v[4])), lamport(__float_as_uint(v[5])),
                            lamport(__float_as_uint(v[6])), lamport(__float_as_uint(v[7]))));
                }
            }
            ++run;
            tcFenceBefore();
            __syncthreads();
            tcFenceAfter();
        }
    }
    // FC2: the CTA's eighths slot by slot; per slot the FC1 partials are summed (SwiGLU + NVFP4 = the token operand)
    // once for all its eighths here.
    // The layer's FC2 activation global scale (Alpha2[kExperts]; 1 for GLM-5.2-NVFP4).
    float const fc2InScale = __ldg(ptr<float const>(ctx, kPAlpha2) + kExperts);
    int grp = 0;
    for (int j = f2a; j < f2b; ++grp)
    {
        int const slot = j >> 3;
        int const jEnd = min(f2b, 8 * slot + 8);
        int const mask = reinterpret_cast<uint8_t const*>(smemMoe + kMoeSlotMask)[slot];
        // The partials of each tile (the CTAs whose FC1 run covers pieces [16 slot + 8 tile, +8), their run 0 or 1)
        // and the expert's scales: listed before the FC1 pieces.
        uint32_t const* const lists = sFc2 + 18 * grp;
        float const alphaUp = __uint_as_float(sFc2[72 + grp]);
        float const alphaDown = __uint_as_float(sFc2[76 + grp]);
#if GLM52_FC2_SPARSE_WIDE && (GLM52_FLAGS_MODE == 3 || GLM52_FLAGS_MODE == 5)
        int const sparseTokens = __popc(static_cast<unsigned>(mask));
        bool const useSparseWide = fc2Direct && sparseP1 && sparseTokens >= 1 && sparseTokens <= 2;
        if (useSparseWide)
        {
            // The full token operand and scale padding are cleared before active
            // writers. The original preparation already has this CTA join.
            if (tid < 32)
            {
                *reinterpret_cast<uint4*>(fc2B + 16 * tid) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(fc2Sfb + 16 * tid) = make_uint4(0, 0, 0, 0);
            }
            __syncthreads();
            if (grp < 2)
            {
                subStamp(ctx, 73 + grp);
            }
            int const activeGroup = tid >> 7;
            if (activeGroup < sparseTokens)
            {
                // Four whole warps own one active token, one feature per lane.
                // A warp stays within one 64-feature tile and one producer list.
                unsigned tokenBits = static_cast<unsigned>(mask);
                if (activeGroup != 0)
                {
                    tokenBits &= tokenBits - 1;
                }
                int const t = __ffs(static_cast<int>(tokenBits)) - 1;
                int const f = tid & 127;
                int const tile = f >> 6;
                int const nPart = static_cast<int>(lists[9 * tile]);
                uint32_t const byteOff = 4 * (t * 128 + 2 * (f & 63));
                float up = 0.f;
                float gate = 0.f;
#pragma unroll 1
                for (int q0 = 0; q0 < nPart; q0 += 4)
                {
                    uint8_t const* src[4];
                    uint2 words[4];
                    bool use[4];
#pragma unroll
                    for (int q = 0; q < 4; ++q)
                    {
                        use[q] = q0 + q < nPart;
                        src[q] = nullptr;
                        if (use[q])
                        {
                            src[q]
                                = p1Base + static_cast<size_t>(lists[9 * tile + 1 + q0 + q]) * 128 * kTok * 4 + byteOff;
                        }
                        words[q] = use[q] ? ldPoll64(src[q]) : make_uint2(0, 0);
                    }
                    uint64_t t0 = 0;
                    while (true)
                    {
                        bool ok = true;
#pragma unroll
                        for (int q = 0; q < 4; ++q)
                        {
                            ok &= !use[q] || full2(words[q]);
                        }
                        if (warpAll(ok))
                        {
                            break;
                        }
                        spinCheck(ctx, t0, 124, slot, q0);
#pragma unroll
                        for (int q = 0; q < 4; ++q)
                        {
                            if (use[q] && !full2(words[q]))
                            {
                                words[q] = ldPoll64(src[q]);
                            }
                        }
                    }
                    // Preserve +0 and the original ascending q-by-q FP32 adds.
#pragma unroll
                    for (int q = 0; q < 4; ++q)
                    {
                        if (use[q])
                        {
                            up += __uint_as_float(words[q].x);
                            gate += __uint_as_float(words[q].y);
                        }
                    }
                }
                float const alpha = alphaUp;
                float const gv = fmaf(up, alpha, 0.f)
                    * (gate * rcpApproxFtz(1.f + ex2ApproxFtz(-(gate * (1.4426950408889634f * alpha)))));
                float amax = fabsf(gv);
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 8));
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
                uint32_t const sf = toE4m3(fc2InScale * (amax * alpha * 0.16666667f));
                float const sff = static_cast<float>(*reinterpret_cast<__nv_fp8_e4m3 const*>(&sf));
                float const r = sff != 0.f ? rcpApproxFtz(sff * rcpApproxFtz(fc2InScale)) : 0.f;
                float const scaled = (gv * r) * alpha;
                float const adjacent = __shfl_xor_sync(0xffffffffu, scaled, 1);
                if ((f & 1) == 0)
                {
                    uint32_t const q01 = toE2m1x2(scaled, adjacent);
                    fc2B[t * 64 + (((f >> 5) ^ ((t >> 1) & 3)) << 4) + ((f >> 1) & 15)] = static_cast<uint8_t>(q01);
                }
                if ((f & 15) == 0)
                {
                    int const b = f >> 4;
                    fc2Sfb[16 * t + (b < 4 ? b : 4 + b)] = static_cast<uint8_t>(sf);
                }
            }
            if (grp < 2)
            {
                subStamp(ctx, 75 + grp);
            }
        }
        else
        {
#endif
            if (tid < 32)
            {
                *reinterpret_cast<uint4*>(fc2Sfb + 16 * tid) = make_uint4(0, 0, 0, 0);
            }
            __syncthreads();
            // A thread owns four adjacent features of one token: [up0, gate0, ..., up3, gate3]. The optional P1
            // layout lets it sum those values directly; the baseline first transposes all partial sums through region2.
            int const t = tid / 32;
            int const b = (tid / 4) % 8;
            int const sub = tid % 4;
            float sum[8];
#if GLM52_FLAGS_MODE == 3 || GLM52_FLAGS_MODE == 5
            bool const activeP1 = !sparseP1 || ((mask >> t) & 1); // one token per whole consumer warp
#endif
            if (grp < 2)
            {
                subStamp(ctx, 73 + grp);
            }
            if (fc2Direct)
            {
#if GLM52_FLAGS_MODE == 3 || GLM52_FLAGS_MODE == 5
                if (activeP1)
                {
#endif
                    int const tile = b >> 2;
                    int const nPart = static_cast<int>(lists[9 * tile]);
                    // Both half-warps run the same poll (4 or 8 runs), even when their tiles have different partial
                    // counts. Inactive slots contribute nothing and never load a list entry. All 32 lanes reach each
                    // warpAll.
                    int const nMax = max(static_cast<int>(lists[0]), static_cast<int>(lists[9]));
                    uint8_t const* const base = p1Base + 4 * (t * 128 + 32 * (b & 3) + 8 * sub);
                    if (nMax > 4)
                    {
                        pollFc2Runs<8>(ctx, base, lists + 9 * tile, nPart, sum, slot);
                    }
                    else
                    {
                        pollFc2Runs<4>(ctx, base, lists + 9 * tile, nPart, sum, slot);
                    }
#if GLM52_FLAGS_MODE == 3 || GLM52_FLAGS_MODE == 5
                }
#endif
                if (grp < 2)
                {
                    subStamp(ctx, 75 + grp);
                }
            }
            else
            {
                float* psum = reinterpret_cast<float*>(region2);
                {
                    int const nPart0 = static_cast<int>(lists[0]);
                    int const nPart1 = static_cast<int>(lists[9]);
                    float4 acc[2] = {make_float4(0.f, 0.f, 0.f, 0.f), make_float4(0.f, 0.f, 0.f, 0.f)};
#pragma unroll 1
                    for (int q0 = 0; q0 < max(nPart0, nPart1); q0 += 4)
                    {
                        uint8_t const* src[2][4];
                        uint4 w[2][4];
                        bool use[2][4];
#pragma unroll
                        for (int tile = 0; tile < 2; ++tile)
                        {
                            int const nPart = tile ? nPart1 : nPart0;
#pragma unroll
                            for (int q = 0; q < 4; ++q)
                            {
                                use[tile][q] = q0 + q < nPart;
                                src[tile][q] = p1Base
                                    + static_cast<size_t>(lists[9 * tile + 1 + (use[tile][q] ? q0 + q : 0)]) * 128
                                        * kTok * 4
                                    + 16 * tid;
                                w[tile][q] = use[tile][q] ? ldPoll128(src[tile][q]) : make_uint4(0, 0, 0, 0);
                            }
                        }
                        uint64_t t0 = 0;
                        while (true)
                        {
                            bool ok = true;
#pragma unroll
                            for (int tile = 0; tile < 2; ++tile)
                            {
#pragma unroll
                                for (int q = 0; q < 4; ++q)
                                {
                                    ok &= !use[tile][q] || full4(w[tile][q]);
                                }
                            }
                            if (warpAll(ok))
                            {
                                break;
                            }
                            spinCheck(ctx, t0, 124, slot, q0);
#pragma unroll
                            for (int tile = 0; tile < 2; ++tile)
                            {
#pragma unroll
                                for (int q = 0; q < 4; ++q)
                                {
                                    if (use[tile][q] && !full4(w[tile][q]))
                                    {
                                        w[tile][q] = ldPoll128(src[tile][q]);
                                    }
                                }
                            }
                        }
#pragma unroll
                        for (int tile = 0; tile < 2; ++tile)
                        {
#pragma unroll
                            for (int q = 0; q < 4; ++q)
                            {
                                if (use[tile][q])
                                {
                                    acc[tile].x += __uint_as_float(w[tile][q].x);
                                    acc[tile].y += __uint_as_float(w[tile][q].y);
                                    acc[tile].z += __uint_as_float(w[tile][q].z);
                                    acc[tile].w += __uint_as_float(w[tile][q].w);
                                }
                            }
                        }
                    }
                    *reinterpret_cast<float4*>(psum + 4 * tid) = acc[0];
                    *reinterpret_cast<float4*>(psum + 128 * kTok + 4 * tid) = acc[1];
                }
                if (grp < 2)
                {
                    subStamp(ctx, 75 + grp);
                }
                __syncthreads();
#pragma unroll
                for (int u = 0; u < 4; ++u)
                {
                    int const lf = 4 * sub + u;
                    int const upRow = 32 * b + 16 * (lf & 1) + (lf >> 1);
                    sum[2 * u] = psum[upRow * kTok + t];
                    sum[2 * u + 1] = psum[(upRow + 8) * kTok + t];
                }
            }
            // SwiGLU + NVFP4 is shared by both paths, with identical feature ownership and arithmetic.
#if GLM52_FLAGS_MODE == 3 || GLM52_FLAGS_MODE == 5
            if (activeP1)
#endif
            {
                float const alpha = alphaUp;
                float gv[4];
#pragma unroll
                for (int u = 0; u < 4; ++u)
                {
                    float const x0 = sum[2 * u];
                    float const x1 = sum[2 * u + 1];
                    gv[u] = fmaf(x0, alpha, 0.f)
                        * (x1 * rcpApproxFtz(1.f + ex2ApproxFtz(-(x1 * (1.4426950408889634f * alpha)))));
                }
                float amax = fmaxf(fmaxf(fabsf(gv[0]), fabsf(gv[1])), fmaxf(fabsf(gv[2]), fabsf(gv[3])));
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
                uint32_t const sf = toE4m3(fc2InScale * (amax * alpha * 0.16666667f));
                float const sff = static_cast<float>(*reinterpret_cast<__nv_fp8_e4m3 const*>(&sf));
                float const r = sff != 0.f ? rcpApproxFtz(sff * rcpApproxFtz(fc2InScale)) : 0.f;
                uint32_t const q01 = toE2m1x2((gv[0] * r) * alpha, (gv[1] * r) * alpha);
                uint32_t const q23 = toE2m1x2((gv[2] * r) * alpha, (gv[3] * r) * alpha);
                // Token t's 64 B row: byte 8 b + 2 sub = chunk b / 2 (swizzled), byte 8 (b & 1) + 2 sub in it.
                *reinterpret_cast<uint16_t*>(fc2B + t * 64 + (((b >> 1) ^ ((t >> 1) & 3)) << 4) + 8 * (b & 1) + 2 * sub)
                    = static_cast<uint16_t>(q01 | (q23 << 8));
                if (sub == 0)
                {
                    fc2Sfb[16 * t + (b < 4 ? b : 4 + b)] = static_cast<uint8_t>(sf);
                }
            }
#if GLM52_FLAGS_MODE == 3 || GLM52_FLAGS_MODE == 5
            else
            {
                // This warp's token is absent from the expert. No P1 words exist for it; write a complete zero operand.
                // Each lane owns the same disjoint two bytes as the active quantizer; the scale padding was zeroed
                // above.
                *reinterpret_cast<uint16_t*>(fc2B + t * 64 + (((b >> 1) ^ ((t >> 1) & 3)) << 4) + 8 * (b & 1) + 2 * sub)
                    = 0;
                if (sub == 0)
                {
                    fc2Sfb[16 * t + (b < 4 ? b : 4 + b)] = 0;
                }
            }
#endif
#if GLM52_FC2_SPARSE_WIDE && (GLM52_FLAGS_MODE == 3 || GLM52_FLAGS_MODE == 5)
        }
#endif
        fenceProxyAsyncSmem();
        __syncthreads();
        if (grp < 8)
        {
            subStamp(ctx, 32 + grp);
        }
        for (; j < jEnd; ++j, ++p)
        {
            int const o = j & 7;
            int const rs = p % kRingSlots;
            uint32_t const base = ctx.smemBase + wOff + rs * kPieceBytes;
            if (p < 8)
            {
                subStamp(ctx, 80 + p);
            }
            waitBar(ctx, barRing0 + 8 * rs, (p / kRingSlots) & 1, 125);
            if (p < 8)
            {
                subStamp(ctx, 16 + 2 * p);
            }
            if (warp == 1)
            {
                fc2PieceMma(tmem, base, smemU32(fc2B), smemU32(fc2Sfb), tmem + kTmemSfa + 48 * rs, barMma);
            }
            waitBar(ctx, barMma, mmaPhase, 133);
            mmaPhase ^= 1;
            if (p + kRingSlots < nPieces)
            {
                // Every thread has seen the slot's phase before the refill below can complete its next one (as in
                // FC1: a waiter that slept through both would wait on the parity of a phase that never comes).
                __syncthreads();
            }
            tcFenceAfter();
            if (p < 8)
            {
                subStamp(ctx, 17 + 2 * p);
            }
            if (tid == 0 && p + kRingSlots < nPieces)
            {
                issueMoePiece(ctx, smemMoe, p + kRingSlots, base, barRing0 + 8 * rs);
            }
            // Output rows: warp w reads TMEM lane quarter q = w % 4 (rows [32 q, +32) of each tile) of tiles [3 (w /
            // 4), +3), the three tiles in one wait; row 32 q + l holds feature 32 q + 4 (l & 7) + l / 8, so lane g < 8
            // gathers features 4 g .. 4 g + 3 from lanes g, g + 8, g + 16, g + 24.
            {
                int const q = warp & 3;
                int const h = warp >> 2;
                float const a2 = alphaDown;
                uint8_t* outRows = reinterpret_cast<uint8_t*>(
                    ptr<__nv_bfloat16>(ctx, kPFc2out) + static_cast<size_t>(slot) * kTok * kHidden);
                float vt[3][8];
                tmemLd3x8(tmem + kTmemD + 24 * h + (static_cast<uint32_t>(32 * q) << 16), vt);
                // Tokens outer, the three tiles inner: one mask test per token (the parallel agent's FC2 drain order;
                // tiles outer was ~7 us slower per slice).
#pragma unroll
                for (int t = 0; t < kTok; ++t)
                {
                    if ((GLM52_FLAGS(ctx) & kFlagFc2Masked) && !((mask >> t) & 1))
                    {
                        continue; // identical mask for every lane; all shuffles remain converged
                    }
#pragma unroll
                    for (int i = 0; i < 3; ++i)
                    {
                        int const ti = 3 * h + i;
                        float const* v = vt[i];
                        int const col = 128 * (6 * o + ti) + 32 * q + 4 * (lane & 7);
                        float const x = (v[t] + 0.f) * a2;
                        float const x1 = __shfl_down_sync(0xffffffffu, x, 8);
                        float const x2 = __shfl_down_sync(0xffffffffu, x, 16);
                        float const x3 = __shfl_down_sync(0xffffffffu, x, 24);
                        if (lane < 8 && ((mask >> t) & 1))
                        {
                            st64(outRows + (static_cast<size_t>(t) * kHidden + col) * 2,
                                make_uint2(lamport(packBf16(x, x1)), lamport(packBf16(x2, x3))));
                        }
                    }
                }
            }
            tcFenceBefore();
            __syncthreads();
            tcFenceAfter();
        }
    }
    tcFenceBefore();
    __syncthreads();
    if (warp == 0)
    {
        tcFenceAfter();
        tmemDealloc(tmem, kTmemCols);
    }
}

// Shared expert of output columns [128 J, +128) (SiLU(gate) * up, down), before the CTA's MoE unit or inside it
// (kFlagOsInMoe): BF16 [8 tokens][128] into the CTA's residual area for its owner unit. smemMoe (inside the MoE, with
// kFlagRstdShared): the routing's MoE list, the tokens' 1 / rms in it.
__device__ __noinline__ void stageOwnerShared(Ctx const& ctx, int J, int const* smemMoe)
{
    int const tid = threadIdx.x;
    int const warp = tid >> 5;
    int const lane = tid & 31;
    __nv_bfloat16* sA = reinterpret_cast<__nv_bfloat16*>(ctx.smem + kSmemRed);   // [8][128]
    __nv_bfloat16* sOut = reinterpret_cast<__nv_bfloat16*>(ctx.smem + kSmemRes); // [8][128]
    // The down-projection weights of the warp's tile (output columns [16 w, +16) of the owner's 128, K = 128 in two
    // chunks) straight into the MMA fragment registers: their loads overlap the polls below.
    uint4 wa[2][4];
    {
        uint8_t const* wd
            = ptr<uint8_t const>(ctx, kPWD) + static_cast<size_t>(J * 8 + warp) * 2 * kBf16Block + lane * 16;
#pragma unroll
        for (int ch = 0; ch < 2; ++ch)
        {
#pragma unroll
            for (int pc = 0; pc < 4; ++pc)
            {
                wa[ch][pc] = ldcg128(wd + ch * kBf16Block + pc * 512);
            }
        }
    }
    if (GLM52_FLAGS(ctx) & kFlagCanaryOwner)
    {
        // The first word of every shared gate_up unit (rows [16 ft, +16) x K quarter kq).
        uint8_t const* a = arena<uint8_t const>(ctx, kArSh);
        waitCanaries(
            ctx, 32, [&](int u) { return a + ((u % 4) * 8 + u / 4) * 32 * kTok * 4; }, 143);
    }
    // Shared expert activation: thread (feature f, token half): gate and up of 4 tokens over the 4 K quarters.
    {
        int const f = tid / 2;
        int const half = tid % 2;
        int const ft = f / 16;
        int const r = f % 16;
        uint8_t const* sh = arena<uint8_t const>(ctx, kArSh) + ((ft * 32 + r) * kTok + 4 * half) * 4;
        // Vector 2 kq: gate rows (row r), 2 kq + 1: up rows (row 16 + r); K quarter kq at + kq * 8 * 32 * 8 floats.
        uint4 v[8];
        {
            uint8_t const* srcs[8];
#pragma unroll
            for (int kq = 0; kq < 4; ++kq)
            {
                srcs[2 * kq] = sh + kq * 8 * 32 * kTok * 4;
                srcs[2 * kq + 1] = sh + (kq * 8 * 32 + 16) * kTok * 4;
            }
#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                v[i] = ldPoll128(srcs[i]);
            }
            uint64_t t0 = 0;
            while (true)
            {
                bool ok = true;
#pragma unroll
                for (int i = 0; i < 8; ++i)
                {
                    ok &= full4(v[i]);
                }
                if (warpAll(ok))
                {
                    break;
                }
                spinCheck(ctx, t0, 126, J, 0);
#pragma unroll
                for (int i = 0; i < 8; ++i)
                {
                    if (!full4(v[i]))
                    {
                        v[i] = ldPoll128(srcs[i]);
                    }
                }
            }
        }
        float gate[4] = {0.f, 0.f, 0.f, 0.f};
        float up[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int kq = 0; kq < 4; ++kq)
        {
            uint32_t const gw[4] = {v[2 * kq].x, v[2 * kq].y, v[2 * kq].z, v[2 * kq].w};
            uint32_t const uw[4] = {v[2 * kq + 1].x, v[2 * kq + 1].y, v[2 * kq + 1].z, v[2 * kq + 1].w};
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                gate[j] += __uint_as_float(gw[j]);
                up[j] += __uint_as_float(uw[j]);
            }
        }
        if constexpr ((kSpecializedStaticFlags & kFlagHPrime) != 0)
        {
            // The shared gate_up ran on h' = res x gamma: its tokens' 1 / rms.
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                float const r = smemMoe != nullptr ? __int_as_float(smemMoe[kMoeRcp + 4 * half + j])
                                                   : hopRcp(ctx, 4 * half + j, 147);
                gate[j] *= r;
                up[j] *= r;
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            float const gb = bf16Round(gate[j]);
            float const ub = bf16Round(up[j]);
            float const sig = 1.f / (1.f + __expf(-gb));
            sA[(4 * half + j) * 128 + f] = __float2bfloat16_rn(gb * sig * ub);
        }
    }
    __syncthreads();
    {
        uint4 b[2][2];
        loadBSmem(b[0], sA, 128, 0);
        loadBSmem(b[1], sA, 128, 1);
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int ch = 0; ch < 2; ++ch)
        {
            uint4 const& p0 = wa[ch][0];
            uint4 const& p1 = wa[ch][1];
            uint4 const& p2 = wa[ch][2];
            uint4 const& p3 = wa[ch][3];
            mmaBf16(acc, p0.x, p2.x, p0.y, p2.y, b[ch][0].x, b[ch][0].y);
            mmaBf16(acc, p0.z, p2.z, p0.w, p2.w, b[ch][0].z, b[ch][0].w);
            mmaBf16(acc, p1.x, p3.x, p1.y, p3.y, b[ch][1].x, b[ch][1].y);
            mmaBf16(acc, p1.z, p3.z, p1.w, p3.w, b[ch][1].z, b[ch][1].w);
        }
        int const g = lane >> 2;
        int const c = lane & 3;
        // Tile w = output columns [16 w, +16) of the owner's 128.
        sOut[(2 * c) * 128 + 16 * warp + g] = __float2bfloat16_rn(acc[0]);
        sOut[(2 * c + 1) * 128 + 16 * warp + g] = __float2bfloat16_rn(acc[1]);
        sOut[(2 * c) * 128 + 16 * warp + g + 8] = __float2bfloat16_rn(acc[2]);
        sOut[(2 * c + 1) * 128 + 16 * warp + g + 8] = __float2bfloat16_rn(acc[3]);
    }
}

// Owner of output columns [128 J, +128): the top-8 combine of the routed FC2 rows (which it empties: it is their only
// reader) plus the shared expert's output (stageOwnerShared, same CTA), published as the MoE all-reduce shard of this
// rank into every GPU's slot.
__device__ __noinline__ void stageOwner(Ctx const& ctx, int J)
{
    int const tid = threadIdx.x;
    __nv_bfloat16 const* sOut = reinterpret_cast<__nv_bfloat16 const*>(ctx.smem + kSmemRes); // [8][128]
    // Thread 128: this GPU's hop buffers and x slot of the next launch are emptied before the shard goes out (the CTA's
    // barrier below orders the publish after it), concurrently with the polls of the routed rows.
    if (tid == 128)
    {
        uint32_t const* c = ctx.counters + kCClear;
        uint64_t t0 = 0;
        uint32_t v;
        while ((v = ldAcquire(c)) < static_cast<uint32_t>(kCountedClear))
        {
            spinCheck(ctx, t0, 129, kCClear, static_cast<int>(v));
        }
    }
    // Thread (token, vector of 8 columns): 8 x 16 = 128 threads.
    uint4 shard = make_uint4(0, 0, 0, 0);
    size_t off = 0;
    uint8_t* srcs[8];
    bool const clearFirst = (GLM52_FLAGS(ctx) & kFlagOwnerClearFirst) != 0;
    if (tid < 128)
    {
        int const tok = tid / 16;
        int const v = tid % 16;
        int const col = 128 * J + 8 * v;
        // Slots and weights of the token's top-8: from this CTA's MoE work list (the MoE unit ran before).
        int const* smemMoe = reinterpret_cast<int const*>(ctx.smem + kSmemSmall + kSmallMoe);
        int slots[8];
        float wts[8];
#pragma unroll
        for (int k = 0; k < 8; ++k)
        {
            slots[k] = smemMoe[kMoeSlotOfTk + tok * 8 + k];
            wts[k] = __int_as_float(smemMoe[kMoeWeight + tok * 8 + k]);
        }
        uint8_t* fc2 = reinterpret_cast<uint8_t*>(ptr<__nv_bfloat16>(ctx, kPFc2out));
        uint4 o[8];
#pragma unroll
        for (int k = 0; k < 8; ++k)
        {
            srcs[k] = fc2 + ((static_cast<size_t>(slots[k]) * kTok + tok) * kHidden + col) * 2;
            o[k] = ldPoll128(srcs[k]);
        }
        {
            uint64_t t0 = 0;
            while (true)
            {
                bool ok = true;
#pragma unroll
                for (int k = 0; k < 8; ++k)
                {
                    ok &= full4(o[k]);
                }
                if (warpAll(ok))
                {
                    break;
                }
                spinCheck(ctx, t0, 130, J, tok);
#pragma unroll
                for (int k = 0; k < 8; ++k)
                {
                    if (!full4(o[k]))
                    {
                        o[k] = ldPoll128(srcs[k]);
                    }
                }
            }
        }
        if (clearFirst)
        {
#pragma unroll
            for (int k = 0; k < 8; ++k)
            {
                st128(srcs[k], make_uint4(kEmpty, kEmpty, kEmpty, kEmpty));
            }
        }
        stampInput(ctx);
        float routed[8];
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            routed[j] = 0.f;
        }
#pragma unroll
        for (int k = 0; k < 8; ++k)
        {
            uint32_t const ow[4] = {o[k].x, o[k].y, o[k].z, o[k].w};
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                routed[2 * j] = fmaf(wts[k], bf16Lo(ow[j]), routed[2 * j]);
                routed[2 * j + 1] = fmaf(wts[k], bf16Hi(ow[j]), routed[2 * j + 1]);
            }
        }
        uint32_t pk[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            float const s0 = __bfloat162float(sOut[tok * 128 + 8 * v + 2 * j]);
            float const s1 = __bfloat162float(sOut[tok * 128 + 8 * v + 2 * j + 1]);
            pk[j] = lamport(packBf16(s0 + bf16Round(routed[2 * j]), s1 + bf16Round(routed[2 * j + 1])));
        }
        shard = make_uint4(pk[0], pk[1], pk[2], pk[3]);
        off = (static_cast<size_t>(tok * kRanks + ctx.rank) * kHidden + col) * 2;
    }
    __syncthreads();
    if (tid < 128)
    {
        st128(ptr<uint8_t>(ctx, kPH2Mc) + static_cast<size_t>(ctx.parity) * kHopBytes + off, shard);
        // The routed rows back to empty for a later launch, after the shard (stores leave in issue order: the merge
        // units' clears before their output held o_proj's input up by ~1 us, mk-v111a).
        if (!clearFirst)
        {
#pragma unroll
            for (int k = 0; k < 8; ++k)
            {
                st128(srcs[k], make_uint4(kEmpty, kEmpty, kEmpty, kEmpty));
            }
        }
    }
}
