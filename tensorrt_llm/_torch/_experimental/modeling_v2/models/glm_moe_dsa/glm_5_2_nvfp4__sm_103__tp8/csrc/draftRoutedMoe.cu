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

#include "glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/csrc/draftRoutedMoe.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cfloat>
#include <cstdint>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace draft_routed_moe
{

namespace
{

// Ring depths (2 KB chunks per warp; the deeper pair with kFlagDeepRing) and the FC2 intermediate ring (one entry
// per expert of a tile).
constexpr int kFc1Stages = 12;
constexpr int kFc1StagesDeep = 20;
constexpr int kFc2Stages = 8;
constexpr int kFc2StagesDeep = 12;
constexpr int kFc2HStages = 6;
constexpr int kChunkBytes = 16 * kKChunk * 2;       // 16 rows x 128 bytes
constexpr int kXRowBytes = kHidden * 2;             // 12288
constexpr int kHEntryBytes = kMaxRows * kInter * 2; // 4096
constexpr int kStageBytes = kMaxRows * kTopK * 16 * 2;
constexpr float kSumEpsilon = 1e-20f;
constexpr float kBf16MaxFinite = 3.38953139e38f;
// Bytes of one FC2 weight block of an expert (one K chunk of all 6144 rows) and of one L2 prefetch piece.
constexpr int kFc2BlockBytes = kHidden * kKChunk * 2;
constexpr int kPrefetchPieceBytes = 16384;
constexpr int kPrefetchPiecesPerBlock = kFc2BlockBytes / kPrefetchPieceBytes;
static_assert(kFc2BlockBytes % kPrefetchPieceBytes == 0, "prefetch pieces must tile an FC2 block");
// FC2 intermediate ring: expert s (in the warp's order over its items) has its K chunks at stream positions
// 4s..4s+3 (alone or as the first of a pair) or 4s-3, 4s-1, 4s+1, 4s+3 (second of a pair), so it is last read at
// iteration 4s+7 or earlier, and expert s + kFc2HStages loads into the same entry together with its first chunk, at
// position >= 4(s + kFc2HStages) - 3, issued at the end of iteration (position - kStages + 1). The entry is free when
// that iteration is not earlier than the last read: kStages <= 4 * kFc2HStages - 9.
static_assert(kFc2Stages <= 4 * kFc2HStages - 9 && kFc2StagesDeep <= 4 * kFc2HStages - 9,
    "an FC2 intermediate entry is refilled before its last read");

// ---------------------------------------------------------------------------------------------------------------
// Arithmetic with the exact instructions of the native kernels.
// ---------------------------------------------------------------------------------------------------------------

__device__ __forceinline__ float mulFtz(float a, float b)
{
    float r;
    asm("mul.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float addFtz(float a, float b)
{
    float r;
    asm("add.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float fmaFtz(float a, float b, float c)
{
    float r;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
    return r;
}

__device__ __forceinline__ float tanhApprox(float x)
{
    float r;
    asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float divApproxFtz(float a, float b)
{
    float r;
    asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float ex2ApproxFtz(float x)
{
    float r;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float rcpApproxFtz(float x)
{
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float mulRn(float a, float b)
{
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float addRn(float a, float b)
{
    float r;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

//! FlashInfer ``sigmoid_accurate`` as compiled with ``-use_fast_math``.
__device__ __forceinline__ float routingSigmoid(float x)
{
    return fmaFtz(tanhApprox(mulFtz(x, 0.5f)), 0.5f, 0.5f);
}

//! Round to BF16 like ``F2FP.SATFINITE.BF16.F32``: round to nearest even, overflow saturates, NaN stays NaN.
__device__ __forceinline__ __nv_bfloat16 bf16SatFinite(float x)
{
    __nv_bfloat16 r = __float2bfloat16_rn(x);
    float back = __bfloat162float(r);
    if (back == INFINITY || back == -INFINITY)
    {
        r = __float2bfloat16_rn(x > 0.f ? kBf16MaxFinite : -kBf16MaxFinite);
    }
    return r;
}

//! The native gated epilogue (``swiGlu``, no clamp limit, alpha 1, beta 0): ``(up + 0) * (gate * sigmoid(gate))``
//! with ``sigmoid(g) = rcp.approx(1 + ex2.approx(-(g * log2 e)))``; both inputs are clamped to +-FLT_MAX first.
__device__ __forceinline__ float swiGlu(float up, float gate)
{
    float const g = gate > FLT_MAX ? FLT_MAX : (gate < -FLT_MAX ? -FLT_MAX : gate);
    float const u = up > FLT_MAX ? FLT_MAX : (up < -FLT_MAX ? -FLT_MAX : up);
    float const t = mulRn(mulRn(g, 1.4426950216293334961f), -1.f);
    float const s = rcpApproxFtz(addRn(ex2ApproxFtz(t), 1.f));
    return mulRn(addRn(u, 0.f), mulRn(g, s));
}

// ---------------------------------------------------------------------------------------------------------------
// Data movement and MMA.
// ---------------------------------------------------------------------------------------------------------------

__device__ __forceinline__ uint32_t smemAddr(void const* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cpAsync16(uint32_t dst, void const* src)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(__cvta_generic_to_global(src)) : "memory");
}

__device__ __forceinline__ void cpAsyncCommit()
{
    asm volatile("cp.async.commit_group;" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cpAsyncWait()
{
    asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

__device__ __forceinline__ void ldmatrixX4(uint32_t (&r)[4], uint32_t addr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void mmaBf16(float (&c)[4], uint32_t const (&a)[4], uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void gridDependencyWait()
{
    asm volatile("griddepcontrol.wait;" ::: "memory");
}

__device__ __forceinline__ void gridDependencyTrigger()
{
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

//! Byte offset of 16-byte unit ``u`` of a 128-byte row segment for row ``r`` (XOR swizzle, conflict-free ldmatrix).
__device__ __forceinline__ uint32_t swizzled(int r, int u)
{
    return static_cast<uint32_t>(((u & ~7) | ((u & 7) ^ (r & 7))) << 4);
}

//! Copies one 16-row x 64-element weight chunk (2 KB, contiguous in the BlockMajorK layout) into a ring slot.
__device__ __forceinline__ void issueWeightChunk(uint32_t slot, __nv_bfloat16 const* src, int lane)
{
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        int const idx = lane + 32 * j;
        int const r = idx >> 3;
        int const u = idx & 7;
        cpAsync16(slot + r * 128 + swizzled(r, u), src + r * kKChunk + u * 8);
    }
}

//! A fragments of k-step ``q`` (16 elements) of a chunk: rows 0..15, natural K order (m16n8k16 row-major A).
__device__ __forceinline__ void loadA(uint32_t (&a)[4], uint32_t slot, int q, int lane)
{
    int const m = lane >> 3;
    int const r = (lane & 7) | ((m & 1) << 3);
    int const u = 2 * q + (m >> 1);
    ldmatrixX4(a, slot + r * 128 + swizzled(r, u));
}

//! B fragments of two consecutive k-steps from a [token][k] BF16 tile (rows ``rowBytes`` apart, units swizzled
//! by token); tokens at or above ``rows`` read token 0 (their columns are discarded).
__device__ __forceinline__ void loadB2(uint32_t (&b)[4], uint32_t base, int rowBytes, int unit, int rows, int lane)
{
    int tok = lane & 7;
    tok = tok < rows ? tok : 0;
    int const u = unit + (lane >> 3);
    ldmatrixX4(b, base + tok * rowBytes + swizzled(tok, u));
}

// ---------------------------------------------------------------------------------------------------------------
// Routing (FlashInfer routingCustom: SigmoidBias + ScaledSumNormalize, warp per token).
// ---------------------------------------------------------------------------------------------------------------

//! Selection key: the score's bits mapped to an unsigned order (a NaN score ranks above every number), then the
//! expert id so that ties go to the lower index.
__device__ __forceinline__ uint64_t packKey(float key, int expert)
{
    uint32_t bits = __float_as_uint(key);
    bits = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
    bits = key != key ? 0xFFFFFFFFu : bits;
    return (static_cast<uint64_t>(bits) << 32) | static_cast<uint64_t>(0xFFFF & (65535 - expert));
}

//! Lexicographic warp maximum of 64-bit keys with two ``redux.sync.max.u32`` (as the native top-k reduction).
__device__ __forceinline__ uint64_t warpMax64(uint64_t v)
{
    uint32_t const hi = static_cast<uint32_t>(v >> 32);
    uint32_t const lo = static_cast<uint32_t>(v);
    uint32_t maxHi;
    asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(maxHi) : "r"(hi));
    uint32_t const loContrib = hi == maxHi ? lo : 0u;
    uint32_t maxLo;
    asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(maxLo) : "r"(loContrib));
    return (static_cast<uint64_t>(maxHi) << 32) | maxLo;
}

struct RouteSmem
{
    int topExpert[kMaxRows][kTopK];
    float topWeight[kMaxRows][kTopK]; //!< Value of the BF16 weight.
    int slotDistinct[kMaxRows][kTopK];
    int distinct[kMaxDistinct];
    uint32_t mask[kNumExperts / 32]; //!< Experts selected by any row: bit e % 32 of word e / 32.
    int numDistinct;
};

//! A lane's routing inputs of one token: logits of experts 8l .. 8l + 7 (two 16-byte loads).
struct RouteLogits
{
    float4 l0;
    float4 l1;
};

__device__ __forceinline__ RouteLogits loadRouteLogits(DraftRoutedMoeParams const& p, int token, int lane)
{
    float4 const* logits = reinterpret_cast<float4 const*>(p.logits + token * kNumExperts + lane * (kNumExperts / 32));
    return RouteLogits{logits[0], logits[1]};
}

//! Compare-exchange for a descending order: a[i] >= a[j] afterwards.
__device__ __forceinline__ void keyExchange(uint64_t& hi, uint64_t& lo)
{
    uint64_t const a = hi;
    uint64_t const b = lo;
    hi = a > b ? a : b;
    lo = a > b ? b : a;
}

//! Sorts 8 keys in descending order (Batcher's odd-even merge sort, 19 compare-exchanges).
__device__ __forceinline__ void sortKeysDescending(uint64_t (&k)[8])
{
    keyExchange(k[0], k[1]);
    keyExchange(k[2], k[3]);
    keyExchange(k[4], k[5]);
    keyExchange(k[6], k[7]);
    keyExchange(k[0], k[2]);
    keyExchange(k[1], k[3]);
    keyExchange(k[4], k[6]);
    keyExchange(k[5], k[7]);
    keyExchange(k[1], k[2]);
    keyExchange(k[5], k[6]);
    keyExchange(k[0], k[4]);
    keyExchange(k[1], k[5]);
    keyExchange(k[2], k[6]);
    keyExchange(k[3], k[7]);
    keyExchange(k[2], k[4]);
    keyExchange(k[3], k[5]);
    keyExchange(k[1], k[2]);
    keyExchange(k[3], k[4]);
    keyExchange(k[5], k[6]);
}

//! Top-8 of one token by one warp; lane k < 8 receives slot k (expert, weight). Lane l holds experts 8l .. 8l + 7
//! (its logits and its 16-byte slice of the BF16 bias, ``braw``); the selection key carries the expert id, so the
//! result does not depend on this placement.
__device__ void routeToken(
    DraftRoutedMoeParams const& p, int token, int lane, RouteLogits const& in, uint4 braw, RouteSmem& rs)
{
    constexpr int kPerLane = kNumExperts / 32;
    float const logit[kPerLane] = {in.l0.x, in.l0.y, in.l0.z, in.l0.w, in.l1.x, in.l1.y, in.l1.z, in.l1.w};
    uint32_t const biasPairs[kPerLane / 2] = {braw.x, braw.y, braw.z, braw.w};
    float sig[kPerLane];
    uint64_t key[kPerLane];
#pragma unroll
    for (int i = 0; i < kPerLane; ++i)
    {
        uint32_t const pair = biasPairs[i / 2];
        float const bias = __uint_as_float((i % 2 == 0 ? pair << 16 : pair & 0xFFFF0000u));
        sig[i] = routingSigmoid(logit[i]);
        key[i] = packKey(addFtz(sig[i], bias), lane * kPerLane + i);
    }
    // The lane's keys in descending order; then every round takes the warp maximum of the lanes' leading keys and
    // the owning lane drops its leading key (keys are unique: they carry the expert id). Every round thus selects
    // the largest key not selected yet, as the native top-k reduction does.
    sortKeysDescending(key);
    float mySig = 0.f;
    int myExpert = 0;
#pragma unroll
    for (int k = 0; k < kTopK; ++k)
    {
        uint64_t const best = warpMax64(key[0]);
        bool const pop = key[0] == best;
#pragma unroll
        for (int i = 0; i + 1 < kPerLane; ++i)
        {
            key[i] = pop ? key[i + 1] : key[i];
        }
        key[kPerLane - 1] = pop ? 0 : key[kPerLane - 1];
        int const e = 65535 - static_cast<int>(best & 0xFFFF);
        int const owner = e / kPerLane;
        int const slot = e % kPerLane;
        float s = 0.f;
#pragma unroll
        for (int i = 0; i < kPerLane; ++i)
        {
            s = i == slot ? sig[i] : s;
        }
        s = __shfl_sync(0xffffffffu, s, owner);
        if (lane == k)
        {
            mySig = s;
            myExpert = e;
        }
    }
    // cg::reduce(warp, sigmoid, plus): XOR butterfly 16, 8, 4, 2, 1; lanes >= 8 hold 0.
    float sum = lane < kTopK ? mySig : 0.f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
    {
        sum = addFtz(sum, __shfl_xor_sync(0xffffffffu, sum, off));
    }
    if (lane < kTopK)
    {
        float const w = divApproxFtz(mulFtz(mySig, p.routedScaling), addFtz(sum, kSumEpsilon));
        rs.topExpert[token][lane] = myExpert;
        rs.topWeight[token][lane] = __bfloat162float(__float2bfloat16_rn(w));
        atomicOr(&rs.mask[myExpert >> 5], 1u << (myExpert & 31));
    }
}

//! Routing of all rows plus the distinct-expert list (ascending ids), by the whole CTA. ``tokensDone`` (optional)
//! receives clock64 when this thread's warp has routed its tokens.
template <int kWarps>
__device__ void routeAll(
    DraftRoutedMoeParams const& p, RouteSmem& rs, int warp, int lane, int64_t* tokensDone = nullptr)
{
    if (threadIdx.x < kNumExperts / 32)
    {
        rs.mask[threadIdx.x] = 0;
    }
    // A warp routes rows warp, warp + kWarps, ...: their logit loads are all issued before the first selection.
    constexpr int kPerWarp = (kMaxRows + kWarps - 1) / kWarps;
    uint4 const braw = *reinterpret_cast<uint4 const*>(p.bias + lane * (kNumExperts / 32));
    RouteLogits in[kPerWarp];
#pragma unroll
    for (int i = 0; i < kPerWarp; ++i)
    {
        if (warp + i * kWarps < p.rows)
        {
            in[i] = loadRouteLogits(p, warp + i * kWarps, lane);
        }
    }
    __syncthreads(); // the mask is cleared before any row sets its bits
#pragma unroll
    for (int i = 0; i < kPerWarp; ++i)
    {
        if (warp + i * kWarps < p.rows)
        {
            routeToken(p, warp + i * kWarps, lane, in[i], braw, rs);
        }
    }
    if (tokensDone != nullptr)
    {
        *tokensDone = clock64();
    }
    __syncthreads();
    // Distinct experts in ascending order: the index of expert e is the number of selected experts below it.
    if (warp == 0 && lane < kNumExperts / 32)
    {
        int idx = 0;
        for (int w = 0; w < lane; ++w)
        {
            idx += __popc(rs.mask[w]);
        }
        uint32_t m = rs.mask[lane];
        if (lane == kNumExperts / 32 - 1)
        {
            rs.numDistinct = idx + __popc(m);
        }
        while (m != 0)
        {
            rs.distinct[idx++] = lane * 32 + __ffs(m) - 1;
            m &= m - 1;
        }
    }
    for (int i = threadIdx.x; i < p.rows * kTopK; i += kWarps * 32)
    {
        int const e = rs.topExpert[i / kTopK][i % kTopK];
        int idx = __popc(rs.mask[e >> 5] & ((1u << (e & 31)) - 1u));
        for (int w = 0; w < (e >> 5); ++w)
        {
            idx += __popc(rs.mask[w]);
        }
        rs.slotDistinct[i / kTopK][i % kTopK] = idx;
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------------------------------------------
// FC1: routing + gate/up GEMM + SwiGLU.
// ---------------------------------------------------------------------------------------------------------------

template <int kWarps, bool kPairs, int kStages>
__global__ void __launch_bounds__(kWarps * 32) mtpMoeFc1Kernel(DraftRoutedMoeParams const p)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ RouteSmem rs;
    int const warp = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
    int const rows = p.rows;
    uint8_t* xs = smem;
    uint8_t* ring = smem + rows * kXRowBytes + warp * kStages * kChunkBytes;
    uint32_t const xsAddr = smemAddr(xs);
    uint32_t const ringAddr = smemAddr(ring);

    // Every input below (logits, activation) is produced by the predecessor grid.
    gridDependencyWait();

    // The activation rows, swizzled per 128-byte segment by token.
    for (int i = threadIdx.x; i < rows * (kXRowBytes / 16); i += kWarps * 32)
    {
        int const t = i / (kXRowBytes / 16);
        int const u = i % (kXRowBytes / 16);
        cpAsync16(xsAddr + t * kXRowBytes + swizzled(t, u), p.x + t * kHidden + u * 8);
    }
    cpAsyncCommit();

    routeAll<kWarps>(p, rs, warp, lane);
    int const numDistinct = rs.numDistinct;
    if (blockIdx.x == 0)
    {
        for (int i = threadIdx.x; i < kRouteWords; i += kWarps * 32)
        {
            int v = 0;
            if (i == 0)
            {
                v = numDistinct;
            }
            else if (i < kRouteSlotOffset)
            {
                v = i - kRouteDistinctOffset < numDistinct ? rs.distinct[i - kRouteDistinctOffset] : -1;
            }
            else if (i < kRouteWeightOffset)
            {
                int const j = i - kRouteSlotOffset;
                v = j / kTopK < rows ? rs.slotDistinct[j / kTopK][j % kTopK] : -1;
            }
            else
            {
                int const j = i - kRouteWeightOffset;
                v = j / kTopK < rows ? __float_as_int(rs.topWeight[j / kTopK][j % kTopK]) : 0;
            }
            p.route[i] = v;
        }
    }
    if (p.flags & kFlagEarlyTrigger)
    {
        gridDependencyTrigger();
    }
    cpAsyncWait<0>();
    __syncthreads();

    int const totalWarps = gridDim.x * kWarps;
    int const gw = blockIdx.x * kWarps + warp;
    int const items = numDistinct * kFc1Tiles;
    int const myItems = items > gw ? (items - gw + totalWarps - 1) / totalWarps : 0;
    int const total = myItems * kFc1Chunks;
    // With kPairs, the warp's tiles are taken two at a time and their chunks interleave (A0 B0 A1 B1 ...): two
    // independent accumulation chains per warp, each still in natural K order. An odd last tile runs alone.
    int const pairChunks = kPairs ? (myItems / 2) * 2 * kFc1Chunks : 0;

    // Chunk n of the warp's stream -> (local tile index, K chunk).
    auto chunkOf = [&](int n, int& tile, int& kc)
    {
        if (n < pairChunks)
        {
            int const r = n % (2 * kFc1Chunks);
            tile = 2 * (n / (2 * kFc1Chunks)) + (r & 1);
            kc = r >> 1;
        }
        else
        {
            tile = (pairChunks / kFc1Chunks) + (n - pairChunks) / kFc1Chunks;
            kc = (n - pairChunks) % kFc1Chunks;
        }
    };

    auto issue = [&](int n)
    {
        int tile, kc;
        chunkOf(n, tile, kc);
        int const item = gw + tile * totalWarps;
        int const e = rs.distinct[item / kFc1Tiles];
        int const a = item % kFc1Tiles;
        __nv_bfloat16 const* src = p.w13 + ((static_cast<int64_t>(e) * kFc1Chunks + kc) * kFc1Rows + a * 16) * kKChunk;
        issueWeightChunk(ringAddr + (n % kStages) * kChunkBytes, src, lane);
    };

    auto epilogue = [&](int tile, float const(&acc)[4])
    {
        int const item = gw + tile * totalWarps;
        int const d = item / kFc1Tiles;
        int const a = item % kFc1Tiles;
        int const g = lane >> 2;
        int const t = lane & 3;
        // Rows 0..7 of the tile are up rows, rows 8..15 the gate rows of the same intermediate columns.
        int const j = 16 * (a >> 1) + 2 * g + (a & 1);
#pragma unroll
        for (int c = 0; c < 2; ++c)
        {
            int const tok = 2 * t + c;
            if (tok < rows)
            {
                p.h[(static_cast<int64_t>(d) * kMaxRows + tok) * kInter + j]
                    = bf16SatFinite(swiGlu(acc[c], acc[2 + c]));
            }
        }
    };

#pragma unroll 1
    for (int n = 0; n < kStages - 1; ++n)
    {
        if (n < total)
        {
            issue(n);
        }
        cpAsyncCommit();
    }

    float acc[4] = {-0.f, -0.f, -0.f, -0.f};
    float acc2[4] = {-0.f, -0.f, -0.f, -0.f};
#pragma unroll 1
    for (int n = 0; n < total; ++n)
    {
        // Chunk n has landed once at most kStages - 2 newer groups are pending.
        cpAsyncWait<kStages - 2>();
        __syncwarp();
        int tile, kc;
        chunkOf(n, tile, kc);
        uint32_t const slot = ringAddr + (n % kStages) * kChunkBytes;
        if (n >= pairChunks)
        {
#pragma unroll
            for (int q = 0; q < 4; q += 2)
            {
                uint32_t b[4];
                loadB2(b, xsAddr, kXRowBytes, kc * 8 + 2 * q, rows, lane);
                uint32_t a[4];
                loadA(a, slot, q, lane);
                mmaBf16(acc, a, b[0], b[1]);
                loadA(a, slot, q + 1, lane);
                mmaBf16(acc, a, b[2], b[3]);
            }
            if (kc == kFc1Chunks - 1)
            {
                epilogue(tile, acc);
                acc[0] = acc[1] = acc[2] = acc[3] = -0.f;
            }
        }
        else if (n & 1)
        {
            // Both chunks of this K step of the pair are in: chunk n - 1 (first tile) and n (second tile).
            uint32_t const slot0 = ringAddr + ((n - 1) % kStages) * kChunkBytes;
#pragma unroll
            for (int q = 0; q < 4; q += 2)
            {
                uint32_t b[4];
                loadB2(b, xsAddr, kXRowBytes, kc * 8 + 2 * q, rows, lane);
                uint32_t a0[4];
                uint32_t a1[4];
                loadA(a0, slot0, q, lane);
                loadA(a1, slot, q, lane);
                mmaBf16(acc, a0, b[0], b[1]);
                mmaBf16(acc2, a1, b[0], b[1]);
                loadA(a0, slot0, q + 1, lane);
                loadA(a1, slot, q + 1, lane);
                mmaBf16(acc, a0, b[2], b[3]);
                mmaBf16(acc2, a1, b[2], b[3]);
            }
            if (kc == kFc1Chunks - 1)
            {
                epilogue(tile - 1, acc);
                epilogue(tile, acc2);
                acc[0] = acc[1] = acc[2] = acc[3] = -0.f;
                acc2[0] = acc2[1] = acc2[2] = acc2[3] = -0.f;
            }
        }
        __syncwarp();
        // The slot refilled now held chunk n - 1, consumed in this or the previous iteration.
        if (n + kStages - 1 < total)
        {
            issue(n + kStages - 1);
        }
        cpAsyncCommit();
    }
    cpAsyncWait<0>();

    if (p.flags & kFlagPrefetchFc2)
    {
        int const pieces = numDistinct * kFc2Chunks * kPrefetchPiecesPerBlock;
        for (int piece = gw * 32 + lane; piece < pieces; piece += totalWarps * 32)
        {
            int const d = piece / (kFc2Chunks * kPrefetchPiecesPerBlock);
            int const rem = piece % (kFc2Chunks * kPrefetchPiecesPerBlock);
            int const kc = rem / kPrefetchPiecesPerBlock;
            int const off = rem % kPrefetchPiecesPerBlock;
            uint8_t const* addr = reinterpret_cast<uint8_t const*>(p.w2
                                      + (static_cast<int64_t>(rs.distinct[d]) * kFc2Chunks + kc) * kHidden * kKChunk)
                + static_cast<int64_t>(off) * kPrefetchPieceBytes;
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(__cvta_generic_to_global(addr)),
                         "r"(kPrefetchPieceBytes)
                         : "memory");
        }
    }
    if (!(p.flags & kFlagEarlyTrigger))
    {
        gridDependencyTrigger();
    }
#endif
}

// ---------------------------------------------------------------------------------------------------------------
// FC2: down GEMM per distinct expert + native finalize, 16 hidden rows per work item.
// ---------------------------------------------------------------------------------------------------------------

struct Fc2Smem
{
    float weight[kMaxRows][kTopK];
    int distinct[kMaxDistinct];
    int8_t slotOf[kMaxRows][kMaxDistinct];
    int numDistinct;
};

__device__ __forceinline__ int hiddenOfStoredRow(int s)
{
    return (s & ~31) | ((s & 7) << 2) | ((s & 31) >> 3);
}

template <int kWarps, bool kPairs, int kStages>
__global__ void __launch_bounds__(kWarps * 32) mtpMoeFc2Kernel(DraftRoutedMoeParams const p)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ Fc2Smem fs;
    int const warp = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
    int const rows = p.rows;
    constexpr int kWarpBytes = kStages * kChunkBytes + kFc2HStages * kHEntryBytes + kStageBytes;
    uint8_t* mine = smem + warp * kWarpBytes;
    uint32_t const ringAddr = smemAddr(mine);
    uint32_t const hAddr = ringAddr + kStages * kChunkBytes;
    __nv_bfloat16* stage = reinterpret_cast<__nv_bfloat16*>(mine + kStages * kChunkBytes + kFc2HStages * kHEntryBytes);

    // The routing record and the intermediate are written by the FC1 grid.
    gridDependencyWait();

    int const numDistinct = p.route[0];
    for (int i = threadIdx.x; i < kMaxRows * kMaxDistinct; i += kWarps * 32)
    {
        fs.slotOf[i / kMaxDistinct][i % kMaxDistinct] = -1;
    }
    for (int i = threadIdx.x; i < kMaxDistinct; i += kWarps * 32)
    {
        fs.distinct[i] = p.route[kRouteDistinctOffset + i];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < rows * kTopK; i += kWarps * 32)
    {
        int const t = i / kTopK;
        int const k = i % kTopK;
        fs.slotOf[t][p.route[kRouteSlotOffset + i]] = static_cast<int8_t>(k);
        fs.weight[t][k] = __int_as_float(p.route[kRouteWeightOffset + i]);
    }
    __syncthreads();

    int const totalWarps = gridDim.x * kWarps;
    int const gw = blockIdx.x * kWarps + warp;
    int const perItem = numDistinct * kFc2Chunks;
    int const myItems = kFc2Tiles > gw ? (kFc2Tiles - gw + totalWarps - 1) / totalWarps : 0;
    int const total = myItems * perItem;
    // With kPairs, the experts of an item are taken two at a time and their chunks interleave (two independent
    // accumulation chains, each in natural K order); an odd last expert runs alone.
    int const pairChunks = kPairs ? (numDistinct / 2) * 2 * kFc2Chunks : 0;

    // Chunk m of an item -> (distinct expert index, K chunk).
    auto chunkOf = [&](int m, int& d, int& kc)
    {
        if (m < pairChunks)
        {
            int const r = m % (2 * kFc2Chunks);
            d = 2 * (m / (2 * kFc2Chunks)) + (r & 1);
            kc = r >> 1;
        }
        else
        {
            d = (pairChunks / kFc2Chunks) + (m - pairChunks) / kFc2Chunks;
            kc = (m - pairChunks) % kFc2Chunks;
        }
    };
    // Intermediate entry of expert d of local item il (entries are consumed in this order).
    auto hEntry = [&](int il, int d) { return hAddr + ((il * numDistinct + d) % kFc2HStages) * kHEntryBytes; };

    auto issue = [&](int n)
    {
        int const il = n / perItem;
        int d, kc;
        chunkOf(n % perItem, d, kc);
        int const tile = gw + il * totalWarps;
        int const e = fs.distinct[d];
        __nv_bfloat16 const* src = p.w2 + ((static_cast<int64_t>(e) * kFc2Chunks + kc) * kHidden + tile * 16) * kKChunk;
        issueWeightChunk(ringAddr + (n % kStages) * kChunkBytes, src, lane);
        if (kc == 0)
        {
            uint32_t const entry = hEntry(il, d);
            for (int u = lane; u < rows * (kInter / 8); u += 32)
            {
                int const tok = u / (kInter / 8);
                int const unit = u % (kInter / 8);
                cpAsync16(entry + tok * (kInter * 2) + swizzled(tok, unit),
                    p.h + (static_cast<int64_t>(d) * kMaxRows + tok) * kInter + unit * 8);
            }
        }
    };

    // FC2 rows of expert d (C fragment) -> BF16, into the token's slot of the staging area.
    auto scatter = [&](int tile, int d, float const(&acc)[4])
    {
        int const g = lane >> 2;
        int const t = lane & 3;
#pragma unroll
        for (int c = 0; c < 2; ++c)
        {
            int const tok = 2 * t + c;
            int const k = tok < rows ? fs.slotOf[tok][d] : -1;
            if (k >= 0)
            {
                __nv_bfloat16 const lo = bf16SatFinite(acc[c]);
                __nv_bfloat16 const hi = bf16SatFinite(acc[2 + c]);
                stage[(tok * kTopK + k) * 16 + g] = lo;
                stage[(tok * kTopK + k) * 16 + g + 8] = hi;
                if (p.debugFc2 != nullptr)
                {
                    int64_t const base = (static_cast<int64_t>(tok) * kTopK + k) * kHidden;
                    p.debugFc2[base + hiddenOfStoredRow(tile * 16 + g)] = lo;
                    p.debugFc2[base + hiddenOfStoredRow(tile * 16 + g + 8)] = hi;
                }
            }
        }
    };

    // Native finalize of the item's 16 hidden rows for every token: FP32 fma chain from 0 in slot order.
    auto finalize = [&](int tile)
    {
        __syncwarp();
        for (int pr = lane; pr < rows * 16; pr += 32)
        {
            int const tok = pr >> 4;
            int const row = pr & 15;
            float sum = 0.f;
#pragma unroll
            for (int k = 0; k < kTopK; ++k)
            {
                sum = fmaFtz(fs.weight[tok][k], __bfloat162float(stage[(tok * kTopK + k) * 16 + row]), sum);
            }
            p.out[static_cast<int64_t>(tok) * kHidden + hiddenOfStoredRow(tile * 16 + row)] = __float2bfloat16_rn(sum);
        }
    };

#pragma unroll 1
    for (int n = 0; n < kStages - 1; ++n)
    {
        if (n < total)
        {
            issue(n);
        }
        cpAsyncCommit();
    }

    float acc[4] = {-0.f, -0.f, -0.f, -0.f};
    float acc2[4] = {-0.f, -0.f, -0.f, -0.f};
#pragma unroll 1
    for (int n = 0; n < total; ++n)
    {
        cpAsyncWait<kStages - 2>();
        __syncwarp();
        int const il = n / perItem;
        int const m = n % perItem;
        int const tile = gw + il * totalWarps;
        int d, kc;
        chunkOf(m, d, kc);
        uint32_t const slot = ringAddr + (n % kStages) * kChunkBytes;
        if (m >= pairChunks)
        {
            uint32_t const entry = hEntry(il, d);
#pragma unroll
            for (int q = 0; q < 4; q += 2)
            {
                uint32_t b[4];
                loadB2(b, entry, kInter * 2, kc * 8 + 2 * q, rows, lane);
                uint32_t a[4];
                loadA(a, slot, q, lane);
                mmaBf16(acc, a, b[0], b[1]);
                loadA(a, slot, q + 1, lane);
                mmaBf16(acc, a, b[2], b[3]);
            }
            if (kc == kFc2Chunks - 1)
            {
                scatter(tile, d, acc);
                acc[0] = acc[1] = acc[2] = acc[3] = -0.f;
            }
        }
        else if (m & 1)
        {
            uint32_t const slot0 = ringAddr + ((n - 1) % kStages) * kChunkBytes;
            uint32_t const entry0 = hEntry(il, d - 1);
            uint32_t const entry1 = hEntry(il, d);
#pragma unroll
            for (int q = 0; q < 4; q += 2)
            {
                uint32_t b0[4];
                uint32_t b1[4];
                loadB2(b0, entry0, kInter * 2, kc * 8 + 2 * q, rows, lane);
                loadB2(b1, entry1, kInter * 2, kc * 8 + 2 * q, rows, lane);
                uint32_t a0[4];
                uint32_t a1[4];
                loadA(a0, slot0, q, lane);
                loadA(a1, slot, q, lane);
                mmaBf16(acc, a0, b0[0], b0[1]);
                mmaBf16(acc2, a1, b1[0], b1[1]);
                loadA(a0, slot0, q + 1, lane);
                loadA(a1, slot, q + 1, lane);
                mmaBf16(acc, a0, b0[2], b0[3]);
                mmaBf16(acc2, a1, b1[2], b1[3]);
            }
            if (kc == kFc2Chunks - 1)
            {
                scatter(tile, d - 1, acc);
                scatter(tile, d, acc2);
                acc[0] = acc[1] = acc[2] = acc[3] = -0.f;
                acc2[0] = acc2[1] = acc2[2] = acc2[3] = -0.f;
            }
        }
        if (m == perItem - 1)
        {
            finalize(tile);
        }
        __syncwarp();
        if (n + kStages - 1 < total)
        {
            issue(n + kStages - 1);
        }
        cpAsyncCommit();
    }
    cpAsyncWait<0>();
#endif
}

// ---------------------------------------------------------------------------------------------------------------
// TMA-fed kernels (kFlagTma). Per CTA one producer thread (last warp, lane 0) streams 3D weight boxes (16 rows per
// chain x 64 elements x several K chunks of the BlockMajorK layout, 128-byte swizzle = the layout loadA reads; 32 KB
// per FC1 box, one expert's K = 256 per FC2 box) into a ring of S stages; W MMA warps each own C 16-row accumulation
// chains per tile and consume every stage. Full barriers complete on the transaction count; each MMA warp releases
// a stage (empty barrier, count W) after the MMAs that read it. The arithmetic of every chain (fragments, k-step
// order, accumulator init, epilogues) is that of the kernels above.
// ---------------------------------------------------------------------------------------------------------------

#ifndef GLM_MOE_DSA_DRAFT_ROUTED_MOE_TIMELINE
#define GLM_MOE_DSA_DRAFT_ROUTED_MOE_TIMELINE 0
#endif

//! Probe builds (GLM_MOE_DSA_DRAFT_ROUTED_MOE_TIMELINE=1) record per-CTA events into DraftRoutedMoeParams::timeline;
//! production builds compile none of it.
constexpr bool kTimeline = GLM_MOE_DSA_DRAFT_ROUTED_MOE_TIMELINE != 0;

//! Columns of a CTA's event record: %globaltimer at entry and when MMA warp 0 is done (ns), the SM id, clock64
//! cycles after entry, and counts.
enum TimelineSlot : int
{
    kTlEntryNs = 0,
    kTlDoneNs = 1,
    kTlSmId = 2,
    kTlWait = 3,          //!< griddepcontrol.wait returned
    kTlSetup = 4,         //!< routing (FC1) / routing record (FC2) in shared memory, roles split
    kTlFirst = 5,         //!< MMA warp 0 found its first stage full
    kTlLast = 6,          //!< MMA warp 0 found its last stage full
    kTlDone = 7,          //!< MMA warp 0 done (epilogue / finalize stored)
    kTlProdFirst = 8,     //!< producer issued its first weight copy
    kTlProdLast = 9,      //!< producer issued its last weight copy
    kTlFullBlocked = 10,  //!< MMA warp 0: stage waits whose first poll failed
    kTlEmptyBlocked = 11, //!< producer: re-arm waits whose first poll failed
    kTlChunks = 12,       //!< MMA warp 0: K chunks consumed
    kTlRouteTokens = 13,  //!< FC1 warp 0 has routed its tokens
    kTlRouted = 14,       //!< FC1 routing and distinct-expert list done (before the activation wait)
};

//! K chunks (2 KB per 16-row chain) per stage of the TMA-fed FC1 kernel: one 32 KB 3D box per stage.
static_assert(kTmaStageChunks * kChunkBytes == 32 * 1024, "32 KB FC1 stages");

__device__ __forceinline__ int64_t globalTimerNs()
{
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return static_cast<int64_t>(t);
}

__device__ __forceinline__ int64_t smId()
{
    uint32_t id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
    return id;
}

//! This CTA's event record of kernel ``kernel`` (0: FC1, 1: FC2), or null.
__device__ __forceinline__ int64_t* timelineRow(DraftRoutedMoeParams const& p, int kernel)
{
    if (!kTimeline || p.timeline == nullptr || blockIdx.x >= kTimelineCtas)
    {
        return nullptr;
    }
    return p.timeline + (static_cast<int64_t>(kernel) * kTimelineCtas + blockIdx.x) * kTimelineSlots;
}

__device__ __forceinline__ void mbarInit(uint32_t bar, uint32_t count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

__device__ __forceinline__ void mbarFenceInit()
{
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void mbarArriveExpectTx(uint32_t bar, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbarArrive(uint32_t bar)
{
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}

__device__ __forceinline__ bool mbarTryWait(uint32_t bar, uint32_t parity)
{
    uint32_t done;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, p;\n\t"
        "}"
        : "=r"(done)
        : "r"(bar), "r"(parity)
        : "memory");
    return done != 0;
}

//! Non-blocking probe of the phase of ``bar`` with parity ``parity`` (acquire when it returns true).
__device__ __forceinline__ bool mbarTestWait(uint32_t bar, uint32_t parity)
{
    uint32_t done;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, p;\n\t"
        "}"
        : "=r"(done)
        : "r"(bar), "r"(parity)
        : "memory");
    return done != 0;
}

//! Blocks until the phase of ``bar`` with parity ``parity`` has completed. Every wait is between threads of one CTA
//! (never on another grid), so one that does not complete within 2^28 polls is a protocol error: trap, do not hang.
__device__ __forceinline__ void mbarWait(uint32_t bar, uint32_t parity)
{
    for (uint32_t i = 0; !mbarTryWait(bar, parity); ++i)
    {
        if (i == (1u << 28))
        {
            __trap();
        }
    }
}

//! mbarWait; returns 1 when the first poll failed (probe builds count these), else 0.
[[maybe_unused]] __device__ __forceinline__ int mbarWaitCounted(uint32_t bar, uint32_t parity)
{
    if (mbarTryWait(bar, parity))
    {
        return 0;
    }
    mbarWait(bar, parity);
    return 1;
}

//! Barrier among the W MMA warps (threads 0 .. 32 W - 1) of a warp-specialized kernel; the producer warp does not
//! take part. Named barrier 1 (``__syncthreads`` uses 0).
template <int W>
__device__ __forceinline__ void mmaWarpsSync()
{
    asm volatile("bar.sync 1, %0;" ::"n"(W * 32) : "memory");
}

__device__ __forceinline__ void tmaLoad3d(uint32_t dst, CUtensorMap const* map, int c0, int c1, int c2, uint32_t bar)
{
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], "
        "[%5];" ::"r"(dst),
        "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(bar)
        : "memory");
}

//! tmaLoad3d with an L2 cache policy (createpolicy).
__device__ __forceinline__ void tmaLoad3dHint(
    uint32_t dst, CUtensorMap const* map, int c0, int c1, int c2, uint32_t bar, uint64_t policy)
{
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, "
        "%3, %4}], [%5], %6;" ::"r"(dst),
        "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(bar), "l"(policy)
        : "memory");
}

//! L2 policy that evicts the lines it is attached to first.
__device__ __forceinline__ uint64_t l2EvictFirstPolicy()
{
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
    return policy;
}

__device__ __forceinline__ void tmaPrefetchL2(CUtensorMap const* map, int c0, int c1, int c2)
{
    asm volatile(
        "cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" ::"l"(reinterpret_cast<uint64_t>(map)),
        "r"(c0), "r"(c1), "r"(c2)
        : "memory");
}

__device__ __forceinline__ void bulkLoad(uint32_t dst, void const* src, uint32_t bytes, uint32_t bar)
{
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
                 "l"(__cvta_generic_to_global(src)), "r"(bytes), "r"(bar)
                 : "memory");
}

__device__ __forceinline__ void prefetchTensorMap(CUtensorMap const* map)
{
    asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(map)) : "memory");
}

//! L2 prefetch of this CTA's share of the distinct experts' FC2 weights (16 KB pieces, expert-major, CTA-strided, so
//! the experts the FC2 grid reads first are fetched first). Weights only: legal at any point of the FC1 grid.
__device__ __forceinline__ void prefetchFc2Share(DraftRoutedMoeParams const& p, int const* distinct, int numDistinct)
{
    constexpr int kPiecesPerExpert = kFc2Chunks * kPrefetchPiecesPerBlock;
    int const pieces = numDistinct * kPiecesPerExpert;
    for (int piece = blockIdx.x; piece < pieces; piece += gridDim.x)
    {
        uint8_t const* addr
            = reinterpret_cast<uint8_t const*>(
                  p.w2 + static_cast<int64_t>(distinct[piece / kPiecesPerExpert]) * kFc2Chunks * kHidden * kKChunk)
            + static_cast<int64_t>(piece % kPiecesPerExpert) * kPrefetchPieceBytes;
        asm volatile(
            "cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(__cvta_generic_to_global(addr)), "r"(kPrefetchPieceBytes)
            : "memory");
    }
}

//! Fragments of one 64-element K chunk for C 16-row chains: A of k-steps 0..3 per chain, B of k-steps (0,1), (2,3).
template <int C>
struct ChunkFrags
{
    uint32_t a[C][4][4];
    uint32_t b[2][4];
};

//! The four k-steps of a chunk in natural K order into each chain's own accumulator (the sequence of the kernels
//! above; the chains are interleaved per k-step and never mixed).
template <int C>
__device__ __forceinline__ void mmaChunk(float (&acc)[C][4], ChunkFrags<C> const& f)
{
#pragma unroll
    for (int q = 0; q < 4; ++q)
    {
#pragma unroll
        for (int c = 0; c < C; ++c)
        {
            mmaBf16(acc[c], f.a[c][q], f.b[q >> 1][2 * (q & 1)], f.b[q >> 1][2 * (q & 1) + 1]);
        }
    }
}

//! Per-lane ldmatrix offsets inside one K chunk. ``a[q]``: the lane's row of k-step q in a 2 KB 16-row chunk (as
//! loadA). ``b[h]``: the lane's 16-byte unit of k-steps (2h, 2h + 1) in the first 128-byte K chunk of a [token][k]
//! tile with ``bRowBytes`` per token (as loadB2). The swizzle only permutes the 16-byte units inside a 128-byte K
//! chunk, so K chunk kc of a token row is the same pattern ``kc * 128`` bytes further.
struct LaneOffsets
{
    uint32_t a[4];
    uint32_t b[2];
};

__device__ __forceinline__ LaneOffsets laneOffsets(int rows, int bRowBytes, int lane)
{
    LaneOffsets o;
    int const m = lane >> 3;
    int const r = (lane & 7) | ((m & 1) << 3);
#pragma unroll
    for (int q = 0; q < 4; ++q)
    {
        o.a[q] = r * 128 + swizzled(r, 2 * q + (m >> 1));
    }
    int tok = lane & 7;
    tok = tok < rows ? tok : 0;
#pragma unroll
    for (int h = 0; h < 2; ++h)
    {
        o.b[h] = tok * bRowBytes + swizzled(tok, 4 * h + m);
    }
    return o;
}

//! Per-lane ldmatrix offsets of the B fragments in a streamed activation slice (kFlagStreamX): one TMA box of
//! ``DraftRoutedMoeTensorMaps::x`` is [8 tokens][kXBoxChunks K chunks][128 bytes] with the 128-byte swizzle, so K chunk
//! p of token t is line 2t + p of the box. ``b[p][h]``: the lane's 16-byte unit of k-steps (2h, 2h + 1) of K chunk p.
struct LaneOffsetsX
{
    uint32_t b[2][2];
};

__device__ __forceinline__ LaneOffsetsX laneOffsetsX(int rows, int lane)
{
    static_assert(kXBoxChunks == 2, "two K chunks per activation box");
    LaneOffsetsX o;
    int const m = lane >> 3;
    int tok = lane & 7;
    tok = tok < rows ? tok : 0;
#pragma unroll
    for (int q = 0; q < 2; ++q)
    {
        int const line = kXBoxChunks * tok + q;
#pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            o.b[q][h] = line * 128 + swizzled(line, 4 * h + m);
        }
    }
    return o;
}

//! The fragments of one K chunk for C 16-row chains (chain c's 2 KB chunk follows chain 0's at c * kChunkBytes)
//! from per-stage bases and constant chunk offsets, so the loads of a stage need no address arithmetic.
//! ``aBase[q]``: chain 0 of the stage's first chunk + LaneOffsets::a[q]; ``bBase[h]``: the activation K chunk of the
//! stage's first chunk + LaneOffsets::b[h].
template <int C>
__device__ __forceinline__ void loadFragsAt(
    ChunkFrags<C>& f, uint32_t const (&aBase)[4], uint32_t aOff, uint32_t const (&bBase)[2], uint32_t bOff)
{
#pragma unroll
    for (int c = 0; c < C; ++c)
    {
#pragma unroll
        for (int q = 0; q < 4; ++q)
        {
            ldmatrixX4(f.a[c][q], aBase[q] + aOff + c * kChunkBytes);
        }
    }
#pragma unroll
    for (int h = 0; h < 2; ++h)
    {
        ldmatrixX4(f.b[h], bBase[h] + bOff);
    }
}

//! loadFragsAt for chains ``kChainStride`` bytes apart (the sub-tile boxes of the triples path); chains ``c >= valid``
//! (absent sub-tiles of the grid's last CTA) are not loaded.
template <int C, int kChainStride>
__device__ __forceinline__ void loadFragsStrided(
    ChunkFrags<C>& f, uint32_t const (&aBase)[4], uint32_t aOff, uint32_t const (&bBase)[2], uint32_t bOff, int valid)
{
#pragma unroll
    for (int c = 0; c < C; ++c)
    {
        if (c < valid)
        {
#pragma unroll
            for (int q = 0; q < 4; ++q)
            {
                ldmatrixX4(f.a[c][q], aBase[q] + aOff + c * kChainStride);
            }
        }
    }
#pragma unroll
    for (int h = 0; h < 2; ++h)
    {
        ldmatrixX4(f.b[h], bBase[h] + bOff);
    }
}

//! mmaChunk for the chains ``c < valid`` only (each chain's k-steps in natural order, as mmaChunk).
template <int C>
__device__ __forceinline__ void mmaChunkValid(float (&acc)[C][4], ChunkFrags<C> const& f, int valid)
{
#pragma unroll
    for (int q = 0; q < 4; ++q)
    {
#pragma unroll
        for (int c = 0; c < C; ++c)
        {
            if (c < valid)
            {
                mmaBf16(acc[c], f.a[c][q], f.b[q >> 1][2 * (q & 1)], f.b[q >> 1][2 * (q & 1) + 1]);
            }
        }
    }
}

//! Fragments of one K chunk for E FC2 experts of one 16-row tile: per expert its A (weights) and B (intermediate).
template <int E>
struct ExpertFrags
{
    uint32_t a[E][4][4];
    uint32_t b[E][2][4];
};

//! Byte offset of element ``j`` of token ``tok`` in an intermediate entry stored as its shared-memory image (512-byte
//! token rows, 16-byte units swizzled by token as loadB2 reads them).
__device__ __forceinline__ int hImageElem(int tok, int j)
{
    int const u = j >> 3;
    return tok * kInter + (((u & ~7) | ((u & 7) ^ (tok & 7))) << 3) + (j & 7);
}

//! TMA-fed FC1 kernel: W MMA warps with C chains each, a ring of S stages of 32 KB (one box of ``map``). With S3 > 0
//! (4 MMA warps, kFlagFc1Triples) a call whose tiles exceed the grid but whose 64-row sub-tiles fit three per CTA
//! runs the triples path instead (decided per call from the routing, identically in every CTA): CTA b takes sub-tiles
//! 3b .. 3b + 2 (in distinct-expert order), MMA warp w owns rows 16w .. 16w + 15 of each (three chains), and a stage
//! of its ring of S3 is three boxes of ``map3`` (64 rows x kTripleStageChunks K chunks), one per sub-tile.
//! With X (kFlagStreamX) the activation rows are not kept in shared memory: every stage also carries the activation
//! K chunks it needs (boxes of ``mapX``, kXBoxBytes each, after the weight boxes), so the ring can be deeper.
template <int W, int C, int S, int S3, bool X = false>
__global__ void __launch_bounds__((W + 1) * 32)
    mtpMoeFc1TmaKernel(DraftRoutedMoeParams const p, __grid_constant__ CUtensorMap const map,
        __grid_constant__ CUtensorMap const map3, __grid_constant__ CUtensorMap const mapX)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    constexpr int kChains = W * C;
    constexpr int kKs = kTmaStageChunks / kChains; // K chunks per stage (one TMA box)
    constexpr int kChunkSet = kChains * kChunkBytes;
    constexpr int kStage = kKs * kChunkSet;
    constexpr int kTileRows = 16 * kChains;
    constexpr int kTilesPerExpert = kFc1Rows / kTileRows;
    constexpr int kStagesPerTile = kFc1Chunks / kKs;
    constexpr int kL2Ahead = 2 * S;
    static_assert(kFc1Chunks % kKs == 0, "a tile must be a whole number of stages");
    static_assert(kKs % 2 == 0, "the fragment double buffer restarts at f[0] on every stage");
    // Triples path (S3 > 0): sub-tiles of 16 W rows, three per CTA; a stage holds one box per sub-tile.
    constexpr int kSubRows = 16 * W;
    constexpr int kSubPerExpert = kFc1Rows / kSubRows;
    constexpr int kChunkStride3 = kSubRows * kKChunk * 2; // next K chunk inside a sub-tile box
    constexpr int kBox3 = kTripleStageChunks * kChunkStride3;
    constexpr int kStage3 = kTripleSubTiles * kBox3;
    constexpr int kStagesPerTile3 = kFc1Chunks / kTripleStageChunks;
    static_assert(S3 == 0 || (W == 4 && kSubRows == kTripleRows), "the triples path has 4 MMA warps (64-row boxes)");
    static_assert(kTripleStageChunks == 2, "the triples consumer walks two K chunks per stage");
    // Ring slot strides: the weight boxes, then (X) the activation boxes of the stage's K chunks.
    constexpr int kXBoxes = X ? kKs / kXBoxChunks : 0;
    constexpr int kSlot = kStage + kXBoxes * kXBoxBytes;
    constexpr int kSlot3 = kStage3 + (X ? kXBoxBytes : 0);
    static_assert(!X || (kKs % kXBoxChunks == 0 && kTripleStageChunks == kXBoxChunks), "whole activation boxes");
    static_assert(kSlot % 1024 == 0 && kSlot3 % 1024 == 0, "slots stay aligned to the 128-byte swizzle pattern");
    constexpr int kRingBytes = S * kSlot > S3 * kSlot3 ? S * kSlot : S3 * kSlot3;
    constexpr int kBars = S > S3 ? S : S3;
    extern __shared__ uint8_t smemRaw[];
    __shared__ RouteSmem rs;
    __shared__ __align__(8) uint64_t bars[2 * kBars];
    int64_t const t0 = kTimeline ? clock64() : 0;
    int64_t const entryNs = kTimeline ? globalTimerNs() : 0;
    int const warp = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
    int const rows = p.rows;
    uint32_t const ringAddr = (smemAddr(smemRaw) + 1023u) & ~1023u;
    uint32_t const xsAddr = ringAddr + kRingBytes;
    uint32_t const fullBar = smemAddr(&bars[0]);
    uint32_t const emptyBar = smemAddr(&bars[kBars]);
    if (threadIdx.x == 0)
    {
        for (int s = 0; s < kBars; ++s)
        {
            mbarInit(fullBar + 8 * s, 1);
            mbarInit(emptyBar + 8 * s, W * 32); // every MMA thread releases its own reads
        }
        mbarFenceInit();
        prefetchTensorMap(&map);
        if constexpr (S3 > 0)
        {
            prefetchTensorMap(&map3);
        }
        if constexpr (X)
        {
            prefetchTensorMap(&mapX);
        }
    }

    // Every input below (logits, activation) is produced by the predecessor grid.
    gridDependencyWait();
    int64_t const tWait = kTimeline ? clock64() - t0 : 0;

    // The MMA warps stage the activation rows (they alone read them); the producer warp does not wait for them.
    for (int i = threadIdx.x; i < (X || warp == W ? 0 : rows * (kXRowBytes / 16)); i += W * 32)
    {
        int const t = i / (kXRowBytes / 16);
        int const u = i % (kXRowBytes / 16);
        cpAsync16(xsAddr + t * kXRowBytes + swizzled(t, u), p.x + t * kHidden + u * 8);
    }
    cpAsyncCommit();

    int64_t tTokens = 0;
    // Ends with __syncthreads (also publishes the barrier initialization).
    routeAll<W + 1>(p, rs, warp, lane, kTimeline ? &tTokens : nullptr);
    int64_t const tRouted = kTimeline ? clock64() - t0 : 0;
    int const numDistinct = rs.numDistinct;
    if (blockIdx.x == 0)
    {
        for (int i = threadIdx.x; i < kRouteWords; i += (W + 1) * 32)
        {
            int v = 0;
            if (i == 0)
            {
                v = numDistinct;
            }
            else if (i < kRouteSlotOffset)
            {
                v = i - kRouteDistinctOffset < numDistinct ? rs.distinct[i - kRouteDistinctOffset] : -1;
            }
            else if (i < kRouteWeightOffset)
            {
                int const j = i - kRouteSlotOffset;
                v = j / kTopK < rows ? rs.slotDistinct[j / kTopK][j % kTopK] : -1;
            }
            else
            {
                int const j = i - kRouteWeightOffset;
                v = j / kTopK < rows ? __float_as_int(rs.topWeight[j / kTopK][j % kTopK]) : 0;
            }
            p.route[i] = v;
        }
    }
    int const tiles = numDistinct * kTilesPerExpert;
    bool const noMma = kTimeline && (p.flags & kFlagProbeNoMma) != 0;
    bool const resident = kTimeline && (p.flags & kFlagProbeResident) != 0;
    // The path depends only on numDistinct and the launch, so every CTA takes the same one.
    int const numSub = numDistinct * kSubPerExpert;
    int const sub0 = kTripleSubTiles * static_cast<int>(blockIdx.x);
    bool const triples = S3 > 0 && (p.flags & kFlagFc1Triples) != 0 && !resident && tiles > static_cast<int>(gridDim.x)
        && numSub <= kTripleSubTiles * static_cast<int>(gridDim.x);
    int const valid3 = min(kTripleSubTiles, numSub - sub0); // sub-tiles of this CTA (<= 0: none)
    // Early trigger (after routing): always with kFlagEarlyTrigger, or for calls on the triples path only.
    bool const early = (p.flags & kFlagEarlyTrigger) != 0 || (triples && (p.flags & kFlagEarlyTriggerTriples) != 0);
    if (early)
    {
        gridDependencyTrigger();
    }
    bool const evictFirst = (p.flags & kFlagFc1EvictFirst) != 0;

    if (warp == W)
    {
        if constexpr (S3 > 0)
        {
            if (lane == 0 && triples)
            {
                uint64_t const policy = evictFirst ? l2EvictFirstPolicy() : 0;
                if (p.flags & kFlagPrefetchFc2Early)
                {
                    prefetchFc2Share(p, rs.distinct, numDistinct);
                }
                int emptyBlocked = 0;
                int64_t tFirst = 0;
                for (int u = 0; u < (valid3 > 0 ? kStagesPerTile3 : 0); ++u)
                {
                    int const s = u % S3;
                    if (u >= S3)
                    {
                        if constexpr (kTimeline)
                        {
                            emptyBlocked += mbarWaitCounted(emptyBar + 8 * s, ((u / S3) - 1) & 1);
                        }
                        else
                        {
                            mbarWait(emptyBar + 8 * s, ((u / S3) - 1) & 1);
                        }
                    }
                    mbarArriveExpectTx(fullBar + 8 * s, valid3 * kBox3 + (X ? kXBoxBytes : 0));
                    if constexpr (X)
                    {
                        tmaLoad3d(
                            ringAddr + s * kSlot3 + kStage3, &mapX, 0, u * kTripleStageChunks, 0, fullBar + 8 * s);
                    }
                    for (int c = 0; c < valid3; ++c)
                    {
                        int const sub = sub0 + c;
                        uint32_t const dst = ringAddr + s * kSlot3 + c * kBox3;
                        int const row = (sub % kSubPerExpert) * kSubRows;
                        int const slab = rs.distinct[sub / kSubPerExpert] * kFc1Chunks + u * kTripleStageChunks;
                        if (evictFirst)
                        {
                            tmaLoad3dHint(dst, &map3, 0, row, slab, fullBar + 8 * s, policy);
                        }
                        else
                        {
                            tmaLoad3d(dst, &map3, 0, row, slab, fullBar + 8 * s);
                        }
                    }
                    if (kTimeline && u == 0)
                    {
                        tFirst = clock64() - t0;
                    }
                }
                int64_t const tLast = kTimeline ? clock64() - t0 : 0;
                if ((p.flags & kFlagPrefetchFc2) && !(p.flags & kFlagPrefetchFc2Early))
                {
                    prefetchFc2Share(p, rs.distinct, numDistinct);
                }
                if (int64_t* row = timelineRow(p, 0))
                {
                    row[kTlProdFirst] = tFirst;
                    row[kTlProdLast] = tLast;
                    row[kTlEmptyBlocked] = emptyBlocked;
                }
            }
        }
        if (lane == 0 && !triples)
        {
            int const myTiles
                = static_cast<int>(blockIdx.x) < tiles ? (tiles - 1 - static_cast<int>(blockIdx.x)) / gridDim.x + 1 : 0;
            int const total = myTiles * kStagesPerTile;
            // Box coordinates of this CTA's u-th stage: row of the tile, (expert, first K chunk) slab.
            auto rowOf = [&](int u)
            {
                int const t = blockIdx.x + (u / kStagesPerTile) * gridDim.x;
                return (t % kTilesPerExpert) * kTileRows;
            };
            auto slabOf = [&](int u)
            {
                int const t = blockIdx.x + (u / kStagesPerTile) * gridDim.x;
                return rs.distinct[t / kTilesPerExpert] * kFc1Chunks + (u % kStagesPerTile) * kKs;
            };
            bool const l2 = (p.flags & kFlagL2Prefetch) != 0;
            uint64_t const policy = evictFirst ? l2EvictFirstPolicy() : 0;
            if (p.flags & kFlagPrefetchFc2Early)
            {
                prefetchFc2Share(p, rs.distinct, numDistinct);
            }
            if (l2)
            {
                for (int u = 0; u < kL2Ahead && u < total; ++u)
                {
                    tmaPrefetchL2(&map, 0, rowOf(u), slabOf(u));
                }
            }
            int emptyBlocked = 0;
            int64_t tFirst = 0;
            for (int u = 0; u < total && !(resident && u == S); ++u)
            {
                if (l2 && u + kL2Ahead < total)
                {
                    tmaPrefetchL2(&map, 0, rowOf(u + kL2Ahead), slabOf(u + kL2Ahead));
                }
                int const s = u % S;
                if (u >= S)
                {
                    if constexpr (kTimeline)
                    {
                        emptyBlocked += mbarWaitCounted(emptyBar + 8 * s, ((u / S) - 1) & 1);
                    }
                    else
                    {
                        mbarWait(emptyBar + 8 * s, ((u / S) - 1) & 1);
                    }
                }
                mbarArriveExpectTx(fullBar + 8 * s, kSlot);
                if constexpr (X)
                {
#pragma unroll
                    for (int b = 0; b < kXBoxes; ++b)
                    {
                        tmaLoad3d(ringAddr + s * kSlot + kStage + b * kXBoxBytes, &mapX, 0,
                            (u % kStagesPerTile) * kKs + b * kXBoxChunks, 0, fullBar + 8 * s);
                    }
                }
                if (evictFirst)
                {
                    tmaLoad3dHint(ringAddr + s * kSlot, &map, 0, rowOf(u), slabOf(u), fullBar + 8 * s, policy);
                }
                else
                {
                    tmaLoad3d(ringAddr + s * kSlot, &map, 0, rowOf(u), slabOf(u), fullBar + 8 * s);
                }
                if (kTimeline && u == 0)
                {
                    tFirst = clock64() - t0;
                }
            }
            int64_t const tLast = kTimeline ? clock64() - t0 : 0;
            if ((p.flags & kFlagPrefetchFc2) && !(p.flags & kFlagPrefetchFc2Early))
            {
                prefetchFc2Share(p, rs.distinct, numDistinct);
            }
            if (int64_t* row = timelineRow(p, 0))
            {
                row[kTlProdFirst] = tFirst;
                row[kTlProdLast] = tLast;
                row[kTlEmptyBlocked] = emptyBlocked;
            }
        }
    }
    else
    {
        cpAsyncWait<0>();
        mmaWarpsSync<W>(); // the activation rows of every MMA thread are in shared memory
        int64_t const tSetup = kTimeline ? clock64() - t0 : 0;
        int n = 0;         // K chunks consumed by this warp (all tiles)
        int fullBlocked = 0;
        int64_t tFirst = 0;
        int64_t tLast = 0;
        // Stages are used in ring order; a tile starts on a stage boundary (kFc1Chunks % kKs == 0). The fragment
        // bases point at the stage of ring slot ``fetchSlot`` (``aBase``) and at the K chunk of the activation rows
        // the current stage starts with (``bBase``); both move by constant steps, so no lane offset stays live.
        int fetchSlot = 0;
        uint32_t fetchPhase = 0;
        int releaseSlot = 0;
        int fetchUses = 0;
        uint32_t aBase[4];
        uint32_t bBase[2];
        uint32_t bX[2][2]; // X: K chunk parity p of the activation boxes of ring slot ``fetchSlot``
        {
            LaneOffsets const lo = laneOffsets(rows, kXRowBytes, lane);
            LaneOffsetsX const lx = laneOffsetsX(rows, lane);
#pragma unroll
            for (int q = 0; q < 4; ++q)
            {
                aBase[q] = ringAddr + warp * C * kChunkBytes + lo.a[q];
            }
#pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                bBase[h] = xsAddr + lo.b[h];
                bX[0][h] = ringAddr + kStage + lx.b[0][h];
                bX[1][h] = ringAddr + kStage + lx.b[1][h];
            }
        }
        // Waits for the stage of ring slot ``fetchSlot`` unless an earlier probe found it complete (``ready``).
        auto waitStage = [&](bool ready)
        {
            if constexpr (kTimeline)
            {
                bool const wait = !resident || fetchUses++ < S;
                if (wait && !ready)
                {
                    fullBlocked += mbarWaitCounted(fullBar + 8 * fetchSlot, fetchPhase);
                }
                tLast = clock64() - t0;
                tFirst = tFirst == 0 ? tLast : tFirst;
            }
            else if (!ready)
            {
                mbarWait(fullBar + 8 * fetchSlot, fetchPhase);
            }
        };
        // Moves the A bases (X: and the activation bases) to the next ring slot.
        auto nextSlot = [&]()
        {
            int const step = fetchSlot + 1 == S ? -(S - 1) * kSlot : kSlot;
#pragma unroll
            for (int q = 0; q < 4; ++q)
            {
                aBase[q] += step;
            }
            if constexpr (X)
            {
#pragma unroll
                for (int h = 0; h < 2; ++h)
                {
                    bX[0][h] += step;
                    bX[1][h] += step;
                }
            }
            if (++fetchSlot == S)
            {
                fetchSlot = 0;
                fetchPhase ^= 1;
            }
        };
        if constexpr (S3 > 0)
        {
            if (triples && valid3 > 0)
            {
                // Chain c of this warp: rows 16 warp .. 16 warp + 15 of sub-tile sub0 + c, in box c of each stage.
                uint32_t a3[4];
                uint32_t b3[2];
                uint32_t b3X[2][2]; // X: K chunk 0 / 1 of the stage's activation box
                {
                    LaneOffsets const lo = laneOffsets(rows, kXRowBytes, lane);
                    LaneOffsetsX const lx = laneOffsetsX(rows, lane);
#pragma unroll
                    for (int q = 0; q < 4; ++q)
                    {
                        a3[q] = ringAddr + warp * kChunkBytes + lo.a[q];
                    }
#pragma unroll
                    for (int h = 0; h < 2; ++h)
                    {
                        b3[h] = xsAddr + lo.b[h];
                        b3X[0][h] = ringAddr + kStage3 + lx.b[0][h];
                        b3X[1][h] = ringAddr + kStage3 + lx.b[1][h];
                    }
                }
                // The fragments of K chunk j (0 / 1) of the current stage.
                auto frags3 = [&](ChunkFrags<kTripleSubTiles>& f, int j)
                {
                    if constexpr (X)
                    {
                        loadFragsStrided<kTripleSubTiles, kBox3>(f, a3, j * kChunkStride3, b3X[j], 0, valid3);
                    }
                    else
                    {
                        loadFragsStrided<kTripleSubTiles, kBox3>(f, a3, j * kChunkStride3, b3, j * kKChunk * 2, valid3);
                    }
                };
                float acc[kTripleSubTiles][4];
#pragma unroll
                for (int c = 0; c < kTripleSubTiles; ++c)
                {
                    acc[c][0] = acc[c][1] = acc[c][2] = acc[c][3] = -0.f;
                }
                ChunkFrags<kTripleSubTiles> f[2];
                waitStage(false);
                if (!noMma)
                {
                    frags3(f[0], 0);
                }
#pragma unroll 1
                for (int st = 0; st < kStagesPerTile3; ++st)
                {
                    // K chunk 0 of the stage: fetch chunk 1, MMAs of chunk 0.
                    if (!noMma)
                    {
                        frags3(f[1], 1);
                        mmaChunkValid<kTripleSubTiles>(acc, f[0], valid3);
                    }
                    if (st + 1 < kStagesPerTile3)
                    {
                        // K chunk 1: move to the next stage, probe it, MMAs of chunk 1, then wait only if the probe
                        // failed and fetch the next stage's chunk 0.
                        int const step = fetchSlot + 1 == S3 ? -(S3 - 1) * kSlot3 : kSlot3;
#pragma unroll
                        for (int q = 0; q < 4; ++q)
                        {
                            a3[q] += step;
                        }
                        if (++fetchSlot == S3)
                        {
                            fetchSlot = 0;
                            fetchPhase ^= 1;
                        }
#pragma unroll
                        for (int h = 0; h < 2; ++h)
                        {
                            b3[h] += kTripleStageChunks * kKChunk * 2;
                            b3X[0][h] += step;
                            b3X[1][h] += step;
                        }
                        bool const ready = mbarTestWait(fullBar + 8 * fetchSlot, fetchPhase);
                        if (!noMma)
                        {
                            mmaChunkValid<kTripleSubTiles>(acc, f[1], valid3);
                        }
                        waitStage(ready);
                        if (!noMma)
                        {
                            frags3(f[0], 0);
                        }
                    }
                    else if (!noMma)
                    {
                        mmaChunkValid<kTripleSubTiles>(acc, f[1], valid3);
                    }
                    // Every thread releases its own reads of the stage (their fragments fed the MMAs above).
                    mbarArrive(emptyBar + 8 * releaseSlot);
                    releaseSlot = releaseSlot + 1 == S3 ? 0 : releaseSlot + 1;
                    n += kTripleStageChunks;
                }
                int const g = lane >> 2;
                int const tq = lane & 3;
#pragma unroll
                for (int c = 0; c < kTripleSubTiles; ++c)
                {
                    if (c < valid3)
                    {
                        int const sub = sub0 + c;
                        int const d = sub / kSubPerExpert;
                        int const a = (sub % kSubPerExpert) * W + warp;
                        int const j = 16 * (a >> 1) + 2 * g + (a & 1);
#pragma unroll
                        for (int h = 0; h < 2; ++h)
                        {
                            int const tok = 2 * tq + h;
                            if (tok < rows)
                            {
                                p.h[static_cast<int64_t>(d) * kMaxRows * kInter + hImageElem(tok, j)]
                                    = bf16SatFinite(swiGlu(acc[c][h], acc[c][2 + h]));
                            }
                        }
                    }
                }
            }
        }
        // The regular path (no tile when the triples path ran).
        for (int t = triples ? tiles : static_cast<int>(blockIdx.x); t < tiles; t += gridDim.x)
        {
            float acc[C][4];
#pragma unroll
            for (int c = 0; c < C; ++c)
            {
                acc[c][0] = acc[c][1] = acc[c][2] = acc[c][3] = -0.f;
            }
            // The fragments of chunk j + 1 are loaded before the MMAs of chunk j, across stage boundaries too (the
            // next stage is waited for at the last chunk of the current one); f[0] holds a stage's first chunk.
            ChunkFrags<C> f[2];
            // The fragments of K chunk j of the current stage (X: chunk j is in activation box j / 2, parity j % 2).
            auto frags = [&](ChunkFrags<C>& fr, int j)
            {
                if constexpr (X)
                {
                    loadFragsAt<C>(fr, aBase, j * kChunkSet, bX[j & 1], (j >> 1) * kXBoxBytes);
                }
                else
                {
                    loadFragsAt<C>(fr, aBase, j * kChunkSet, bBase, j * 128);
                }
            };
            if (t != static_cast<int>(blockIdx.x))
            {
                nextSlot();
#pragma unroll
                for (int h = 0; h < 2; ++h)
                {
                    bBase[h] -= (kStagesPerTile - 1) * kKs * 128;
                }
            }
            waitStage(false);
            if (!noMma)
            {
                frags(f[0], 0);
            }
#pragma unroll 1
            for (int st = 0; st < kStagesPerTile; ++st)
            {
#pragma unroll
                for (int j = 0; j < kKs; ++j)
                {
                    if (j + 1 < kKs)
                    {
                        if (!noMma)
                        {
                            frags(f[(j + 1) & 1], j + 1);
                            mmaChunk<C>(acc, f[j & 1]);
                        }
                    }
                    else if (st + 1 < kStagesPerTile)
                    {
                        // Last chunk of the stage: probe the next stage, issue this chunk's MMAs, then wait only if
                        // the probe failed, so the barrier round trip overlaps the MMAs.
                        nextSlot();
#pragma unroll
                        for (int h = 0; h < 2; ++h)
                        {
                            bBase[h] += kKs * 128;
                        }
                        bool const ready = mbarTestWait(fullBar + 8 * fetchSlot, fetchPhase);
                        if (!noMma)
                        {
                            mmaChunk<C>(acc, f[j & 1]);
                        }
                        waitStage(ready);
                        if (!noMma)
                        {
                            frags(f[0], 0);
                        }
                    }
                    else if (!noMma)
                    {
                        mmaChunk<C>(acc, f[j & 1]);
                    }
                }
                // Every thread releases its own reads of the stage (their fragments fed the MMAs above).
                mbarArrive(emptyBar + 8 * releaseSlot);
                releaseSlot = releaseSlot + 1 == S ? 0 : releaseSlot + 1;
                n += kKs;
            }
            int const d = t / kTilesPerExpert;
            int const g = lane >> 2;
            int const tq = lane & 3;
#pragma unroll
            for (int c = 0; c < C; ++c)
            {
                int const a = (t % kTilesPerExpert) * kChains + warp * C + c;
                int const j = 16 * (a >> 1) + 2 * g + (a & 1);
#pragma unroll
                for (int h = 0; h < 2; ++h)
                {
                    int const tok = 2 * tq + h;
                    if (tok < rows)
                    {
                        p.h[static_cast<int64_t>(d) * kMaxRows * kInter + hImageElem(tok, j)]
                            = bf16SatFinite(swiGlu(acc[c][h], acc[c][2 + h]));
                    }
                }
            }
        }
        if (warp == 0 && lane == 0)
        {
            if (int64_t* row = timelineRow(p, 0))
            {
                int64_t const cycles = clock64() - t0;
                row[kTlEntryNs] = entryNs;
                row[kTlDoneNs] = globalTimerNs();
                row[kTlSmId] = smId();
                row[kTlWait] = tWait;
                row[kTlSetup] = tSetup;
                row[kTlFirst] = tFirst;
                row[kTlLast] = tLast;
                row[kTlDone] = cycles;
                row[kTlFullBlocked] = fullBlocked;
                row[kTlChunks] = n;
                row[kTlRouteTokens] = tTokens != 0 ? tTokens - t0 : 0;
                row[kTlRouted] = tRouted;
            }
        }
    }
    if (!early)
    {
        // A CTA counts as triggered as soon as one of its threads triggers: wait for the MMA warps, or the idle
        // producer lanes would release the dependent grid onto free SMs right after routing.
        __syncthreads();
        gridDependencyTrigger();
    }
#endif
}

//! FC2 ring depth for W MMA warps (stages of W * 8 KB weights + one intermediate entry, about 140 KB in all).
template <int W>
constexpr int fc2TmaStages()
{
    return W == 4 ? 4 : (W == 3 ? 5 : (W == 2 ? 7 : 12));
}

template <int W, int S, int E>
__global__ void __launch_bounds__((W + 1) * 32)
    mtpMoeFc2TmaKernel(DraftRoutedMoeParams const p, __grid_constant__ CUtensorMap const map)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    static_assert(E == 1 || E == 2, "one or two experts per step");
    static_assert(S >= 2 * E, "the ring must hold the step being consumed and the next one");
    // One stage per (group, expert): the expert's W * 16 rows x K = 256 weight box (one 3D TMA) and its
    // intermediate entry (one bulk copy), both counted on the stage's full barrier.
    constexpr int kWBytes = W * kFc2Chunks * kChunkBytes;
    constexpr int kStage = kWBytes + kHEntryBytes;
    constexpr int kGroups = kFc2Tiles / W;
    static_assert(kWBytes % 1024 == 0 && kHEntryBytes % 1024 == 0, "stages must stay 1 KB aligned");
    extern __shared__ uint8_t smemRaw[];
    __shared__ Fc2Smem fs;
    __shared__ __align__(8) uint64_t bars[2 * S];
    int64_t const t0 = kTimeline ? clock64() : 0;
    int64_t const entryNs = kTimeline ? globalTimerNs() : 0;
    int const warp = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
    int const rows = p.rows;
    uint32_t const ringAddr = (smemAddr(smemRaw) + 1023u) & ~1023u;
    uint8_t* const stageBase = smemRaw + (ringAddr - smemAddr(smemRaw)) + S * kStage;
    uint32_t const fullBar = smemAddr(&bars[0]);
    uint32_t const emptyBar = smemAddr(&bars[S]);
    if (threadIdx.x == 0)
    {
        for (int s = 0; s < S; ++s)
        {
            mbarInit(fullBar + 8 * s, 1);
            mbarInit(emptyBar + 8 * s, W * 32); // every MMA thread releases its own reads
        }
        mbarFenceInit();
        prefetchTensorMap(&map);
    }

    // The routing record and the intermediate are written by the FC1 grid.
    gridDependencyWait();
    int64_t const tWait = kTimeline ? clock64() - t0 : 0;
    bool const noMma = kTimeline && (p.flags & kFlagProbeNoMma) != 0;
    // Publishes the barrier initialization. Past this point the producer only needs the distinct experts and the MMA
    // warps only the slot and weight tables, so each role reads its part of the routing record by itself.
    __syncthreads();
    int const numDistinct = p.route[0];

    if (warp == W)
    {
        for (int i = lane; i < kMaxDistinct; i += 32)
        {
            fs.distinct[i] = p.route[kRouteDistinctOffset + i];
        }
        __syncwarp();
        if (lane == 0)
        {
            // The intermediate was written with generic stores by the previous grid; bulk copies read it.
            asm volatile("fence.proxy.async.global;" ::: "memory");
            int u = 0;
            int emptyBlocked = 0;
            int64_t tFirst = 0;
            for (int gi = blockIdx.x; gi < kGroups; gi += gridDim.x)
            {
                for (int d = 0; d < numDistinct; ++d, ++u)
                {
                    int const s = u % S;
                    if (u >= S)
                    {
                        if constexpr (kTimeline)
                        {
                            emptyBlocked += mbarWaitCounted(emptyBar + 8 * s, ((u / S) - 1) & 1);
                        }
                        else
                        {
                            mbarWait(emptyBar + 8 * s, ((u / S) - 1) & 1);
                        }
                    }
                    mbarArriveExpectTx(fullBar + 8 * s, kWBytes + rows * kInter * 2);
                    tmaLoad3d(
                        ringAddr + s * kStage, &map, 0, gi * 16 * W, fs.distinct[d] * kFc2Chunks, fullBar + 8 * s);
                    bulkLoad(ringAddr + s * kStage + kWBytes, p.h + static_cast<int64_t>(d) * kMaxRows * kInter,
                        rows * kInter * 2, fullBar + 8 * s);
                    if (kTimeline && u == 0)
                    {
                        tFirst = clock64() - t0;
                    }
                }
            }
            if (int64_t* row = timelineRow(p, 1))
            {
                row[kTlProdFirst] = tFirst;
                row[kTlProdLast] = clock64() - t0;
                row[kTlEmptyBlocked] = emptyBlocked;
            }
        }
    }
    else
    {
        // Slot and weight tables, shared by the W MMA warps only (named barrier 1); first read at the end of a step.
        for (int i = threadIdx.x; i < kMaxRows * kMaxDistinct; i += W * 32)
        {
            fs.slotOf[i / kMaxDistinct][i % kMaxDistinct] = -1;
        }
        mmaWarpsSync<W>();
        for (int i = threadIdx.x; i < rows * kTopK; i += W * 32)
        {
            int const t = i / kTopK;
            int const k = i % kTopK;
            fs.slotOf[t][p.route[kRouteSlotOffset + i]] = static_cast<int8_t>(k);
            fs.weight[t][k] = __int_as_float(p.route[kRouteWeightOffset + i]);
        }
        mmaWarpsSync<W>();
        int64_t const tSetup = kTimeline ? clock64() - t0 : 0;
        __nv_bfloat16* stage = reinterpret_cast<__nv_bfloat16*>(stageBase + warp * kStageBytes);
        int u0 = 0; // stage uses of the previous groups
        int fullBlocked = 0;
        int64_t tFirst = 0;
        int64_t tLast = 0;
        // A group's experts are taken E at a time (a step); step j holds experts E*j .. E*j + E - 1, the last step
        // may hold fewer. Each expert has its own stage and accumulator; their MMAs are interleaved per k-step.
        int const steps = (numDistinct + E - 1) / E;
        int fetchSlot = 0; // stages are used in ring order, one per (group, expert)
        uint32_t fetchPhase = 0;
        int releaseSlot = 0;
        // Fragment bases of the next ring slot to be fetched; they move by constant steps.
        uint32_t slotA[4];
        uint32_t slotB[2];
        {
            LaneOffsets const lo = laneOffsets(rows, kInter * 2, lane);
#pragma unroll
            for (int q = 0; q < 4; ++q)
            {
                slotA[q] = ringAddr + warp * kChunkBytes + lo.a[q];
            }
#pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                slotB[h] = ringAddr + kWBytes + lo.b[h];
            }
        }
        uint32_t aBase[E][4];
        uint32_t bBase[E][2];
        // Points the fragment bases at the stages of step j (one per expert of the step, in ring order) and probes
        // them; finishStep waits for those the probe did not find complete. A step past the last expert reads the
        // step's first expert again (results discarded).
        uint32_t pendBar[E];
        uint32_t pendParity[E];
        bool pendWait[E];
        auto enterStep = [&](int j)
        {
#pragma unroll
            for (int e = 0; e < E; ++e)
            {
                pendWait[e] = false;
                if (E * j + e < numDistinct)
                {
                    pendBar[e] = fullBar + 8 * fetchSlot;
                    pendParity[e] = fetchPhase;
                    pendWait[e] = !mbarTestWait(pendBar[e], pendParity[e]);
#pragma unroll
                    for (int q = 0; q < 4; ++q)
                    {
                        aBase[e][q] = slotA[q];
                    }
#pragma unroll
                    for (int h = 0; h < 2; ++h)
                    {
                        bBase[e][h] = slotB[h];
                    }
                    int const step = fetchSlot + 1 == S ? -(S - 1) * kStage : kStage;
#pragma unroll
                    for (int q = 0; q < 4; ++q)
                    {
                        slotA[q] += step;
                    }
#pragma unroll
                    for (int h = 0; h < 2; ++h)
                    {
                        slotB[h] += step;
                    }
                    if (++fetchSlot == S)
                    {
                        fetchSlot = 0;
                        fetchPhase ^= 1;
                    }
                }
                else
                {
#pragma unroll
                    for (int q = 0; q < 4; ++q)
                    {
                        aBase[e][q] = aBase[0][q];
                    }
#pragma unroll
                    for (int h = 0; h < 2; ++h)
                    {
                        bBase[e][h] = bBase[0][h];
                    }
                }
            }
        };
        auto finishStep = [&]()
        {
#pragma unroll
            for (int e = 0; e < E; ++e)
            {
                if (pendWait[e])
                {
                    if constexpr (kTimeline)
                    {
                        ++fullBlocked;
                    }
                    mbarWait(pendBar[e], pendParity[e]);
                }
            }
            if constexpr (kTimeline)
            {
                tLast = clock64() - t0;
                tFirst = tFirst == 0 ? tLast : tFirst;
            }
        };
        // K chunk kc of every expert of the current step.
        auto load = [&](ExpertFrags<E>& f, int kc)
        {
#pragma unroll
            for (int e = 0; e < E; ++e)
            {
#pragma unroll
                for (int q = 0; q < 4; ++q)
                {
                    ldmatrixX4(f.a[e][q], aBase[e][q] + kc * W * kChunkBytes);
                }
#pragma unroll
                for (int h = 0; h < 2; ++h)
                {
                    ldmatrixX4(f.b[e][h], bBase[e][h] + kc * 128);
                }
            }
        };
        for (int gi = blockIdx.x; gi < kGroups; gi += gridDim.x)
        {
            int const tile = gi * W + warp;
            float acc[E][4];
#pragma unroll
            for (int e = 0; e < E; ++e)
            {
                acc[e][0] = acc[e][1] = acc[e][2] = acc[e][3] = -0.f;
            }
            // The fragments of chunk kc + 1 are loaded before the MMAs of chunk kc, across steps too (the next
            // step's accumulators are independent); f[0] holds a step's first chunk.
            ExpertFrags<E> f[2];
            enterStep(0);
            finishStep();
            if (!noMma)
            {
                load(f[0], 0);
            }
#pragma unroll 1
            for (int j = 0; j < steps; ++j)
            {
#pragma unroll
                for (int kc = 0; kc < kFc2Chunks; ++kc)
                {
                    auto mmas = [&]()
                    {
                        if (!noMma)
                        {
#pragma unroll
                            for (int q = 0; q < 4; ++q)
                            {
#pragma unroll
                                for (int e = 0; e < E; ++e)
                                {
                                    ExpertFrags<E> const& fr = f[kc & 1];
                                    mmaBf16(acc[e], fr.a[e][q], fr.b[e][q >> 1][2 * (q & 1)],
                                        fr.b[e][q >> 1][2 * (q & 1) + 1]);
                                }
                            }
                        }
                    };
                    if (kc + 1 < kFc2Chunks)
                    {
                        if (!noMma)
                        {
                            load(f[(kc + 1) & 1], kc + 1);
                        }
                        mmas();
                    }
                    else if (j + 1 < steps)
                    {
                        // Last chunk of the step: probe the next step's stages, issue this chunk's MMAs, then wait
                        // only for stages the probe did not find complete.
                        enterStep(j + 1);
                        mmas();
                        finishStep();
                        if (!noMma)
                        {
                            load(f[0], 0);
                        }
                    }
                    else
                    {
                        mmas();
                    }
                }
                // Every thread releases its own reads of the step's stages (their fragments fed the MMAs above).
#pragma unroll
                for (int e = 0; e < E; ++e)
                {
                    if (E * j + e < numDistinct)
                    {
                        mbarArrive(emptyBar + 8 * releaseSlot);
                        releaseSlot = releaseSlot + 1 == S ? 0 : releaseSlot + 1;
                    }
                }
                // FC2 rows of each expert -> BF16, into the token's slot of the warp's staging area.
                int const g = lane >> 2;
                int const tq = lane & 3;
#pragma unroll
                for (int e = 0; e < E; ++e)
                {
                    int const d = E * j + e;
#pragma unroll
                    for (int c = 0; c < 2; ++c)
                    {
                        int const tok = 2 * tq + c;
                        int const k = (d < numDistinct && tok < rows) ? fs.slotOf[tok][d] : -1;
                        if (k >= 0)
                        {
                            __nv_bfloat16 const lo16 = bf16SatFinite(acc[e][c]);
                            __nv_bfloat16 const hi16 = bf16SatFinite(acc[e][2 + c]);
                            stage[(tok * kTopK + k) * 16 + g] = lo16;
                            stage[(tok * kTopK + k) * 16 + g + 8] = hi16;
                            if (p.debugFc2 != nullptr)
                            {
                                int64_t const base = (static_cast<int64_t>(tok) * kTopK + k) * kHidden;
                                p.debugFc2[base + hiddenOfStoredRow(tile * 16 + g)] = lo16;
                                p.debugFc2[base + hiddenOfStoredRow(tile * 16 + g + 8)] = hi16;
                            }
                        }
                    }
                    acc[e][0] = acc[e][1] = acc[e][2] = acc[e][3] = -0.f;
                }
            }
            u0 += numDistinct;
            // Native finalize of the tile's 16 hidden rows for every token: FP32 fma chain from 0 in slot order.
            __syncwarp();
            for (int pr = lane; pr < rows * 16; pr += 32)
            {
                int const tok = pr >> 4;
                int const row = pr & 15;
                float sum = 0.f;
#pragma unroll
                for (int k = 0; k < kTopK; ++k)
                {
                    sum = fmaFtz(fs.weight[tok][k], __bfloat162float(stage[(tok * kTopK + k) * 16 + row]), sum);
                }
                p.out[static_cast<int64_t>(tok) * kHidden + hiddenOfStoredRow(tile * 16 + row)]
                    = __float2bfloat16_rn(sum);
            }
            __syncwarp();
        }
        if (warp == 0 && lane == 0)
        {
            if (int64_t* row = timelineRow(p, 1))
            {
                int64_t const cycles = clock64() - t0;
                row[kTlEntryNs] = entryNs;
                row[kTlDoneNs] = globalTimerNs();
                row[kTlSmId] = smId();
                row[kTlWait] = tWait;
                row[kTlSetup] = tSetup;
                row[kTlFirst] = tFirst;
                row[kTlLast] = tLast;
                row[kTlDone] = cycles;
                row[kTlFullBlocked] = fullBlocked;
                row[kTlChunks] = u0 * kFc2Chunks;
            }
        }
    }
#endif
}

template <typename Kernel>
cudaError_t launchOne(Kernel kernel, int ctas, int threads, int smemBytes, bool pdl, DraftRoutedMoeParams const& params,
    cudaStream_t stream)
{
    cudaError_t status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes);
    if (status != cudaSuccess)
    {
        return status;
    }
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(ctas);
    config.blockDim = dim3(threads);
    config.dynamicSmemBytes = smemBytes;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    config.attrs = attrs;
    config.numAttrs = 1;
    return cudaLaunchKernelEx(&config, kernel, params);
}

template <typename Kernel>
cudaError_t launchTma(Kernel kernel, int ctas, int threads, int smemBytes, bool pdl, DraftRoutedMoeParams const& params,
    CUtensorMap const& map, cudaStream_t stream)
{
    cudaError_t status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes);
    if (status != cudaSuccess)
    {
        return status;
    }
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(ctas);
    config.blockDim = dim3(threads);
    config.dynamicSmemBytes = smemBytes;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    config.attrs = attrs;
    config.numAttrs = 1;
    return cudaLaunchKernelEx(&config, kernel, params, map);
}

template <typename Kernel>
cudaError_t launchTma3(Kernel kernel, int ctas, int threads, int smemBytes, bool pdl,
    DraftRoutedMoeParams const& params, CUtensorMap const& map, CUtensorMap const& map3, CUtensorMap const& mapX,
    cudaStream_t stream)
{
    cudaError_t status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes);
    if (status != cudaSuccess)
    {
        return status;
    }
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(ctas);
    config.blockDim = dim3(threads);
    config.dynamicSmemBytes = smemBytes;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    config.attrs = attrs;
    config.numAttrs = 1;
    return cudaLaunchKernelEx(&config, kernel, params, map, map3, mapX);
}

template <typename Kernel>
cudaError_t launchTma2(Kernel kernel, int ctas, int threads, int smemBytes, bool pdl,
    DraftRoutedMoeParams const& params, CUtensorMap const& map, CUtensorMap const& map3, cudaStream_t stream)
{
    return launchTma3(kernel, ctas, threads, smemBytes, pdl, params, map, map3, map, stream);
}

} // namespace

int fc1SmemBytes(int rows, int warps, bool deep)
{
    return rows * kXRowBytes + warps * (deep ? kFc1StagesDeep : kFc1Stages) * kChunkBytes;
}

int fc2SmemBytes(int warps, bool deep)
{
    return warps * ((deep ? kFc2StagesDeep : kFc2Stages) * kChunkBytes + kFc2HStages * kHEntryBytes + kStageBytes);
}

namespace
{

template <int W, int S>
cudaError_t launchFc1(bool pairs, int smem, bool pdl, DraftRoutedMoeParams const& params, cudaStream_t stream)
{
    return pairs ? launchOne(mtpMoeFc1Kernel<W, true, S>, params.fc1Ctas, 32 * W, smem, pdl, params, stream)
                 : launchOne(mtpMoeFc1Kernel<W, false, S>, params.fc1Ctas, 32 * W, smem, pdl, params, stream);
}

template <int W, int S>
cudaError_t launchFc2(bool pairs, int smem, bool pdl, DraftRoutedMoeParams const& params, cudaStream_t stream)
{
    return pairs ? launchOne(mtpMoeFc2Kernel<W, true, S>, params.fc2Ctas, 32 * W, smem, pdl, params, stream)
                 : launchOne(mtpMoeFc2Kernel<W, false, S>, params.fc2Ctas, 32 * W, smem, pdl, params, stream);
}

template <int S>
cudaError_t launchFc1Warps(bool pairs, int smem, bool pdl, DraftRoutedMoeParams const& params, cudaStream_t stream)
{
    switch (params.fc1Warps)
    {
    case 1: return launchFc1<1, S>(pairs, smem, pdl, params, stream);
    case 2: return launchFc1<2, S>(pairs, smem, pdl, params, stream);
    case 4: return launchFc1<4, S>(pairs, smem, pdl, params, stream);
    default: return cudaErrorInvalidValue;
    }
}

template <int S>
cudaError_t launchFc2Warps(bool pairs, int smem, bool pdl, DraftRoutedMoeParams const& params, cudaStream_t stream)
{
    switch (params.fc2Warps)
    {
    case 1: return launchFc2<1, S>(pairs, smem, pdl, params, stream);
    case 2: return launchFc2<2, S>(pairs, smem, pdl, params, stream);
    case 4: return launchFc2<4, S>(pairs, smem, pdl, params, stream);
    default: return cudaErrorInvalidValue;
    }
}

} // namespace

int fc1TmaSmemBytes(int rows, int stages)
{
    return 1024 + stages * kTmaStageChunks * kChunkBytes + rows * kXRowBytes;
}

int fc1TmaTriplesSmemBytes(int rows, int stages, int tripleStages)
{
    int const ring = stages * kTmaStageChunks * kChunkBytes;
    int const ring3 = tripleStages * kTripleSubTiles * kTripleStageChunks * kTripleRows * kKChunk * 2;
    return 1024 + (ring > ring3 ? ring : ring3) + rows * kXRowBytes;
}

int fc1TmaStreamXSmemBytes()
{
    int const ring = 6 * (kTmaStageChunks * kChunkBytes + kXBoxBytes);
    int const ring3 = 4 * (kTripleSubTiles * kTripleRows * kTripleStageChunks * kKChunk * 2 + kXBoxBytes);
    return 1024 + (ring > ring3 ? ring : ring3);
}

int fc2TmaSmemBytes(int warps, int stages)
{
    return 1024 + stages * (warps * kFc2Chunks * kChunkBytes + kHEntryBytes) + warps * kStageBytes;
}

namespace
{

//! Dynamic shared memory budget (the static routing state and barriers take about 3.5 KB of the 227 KB).
constexpr int kMaxDynamicSmem = 220 * 1024;

//! FC1 kernel with ring depth S (and S3 for the triples path, 4 MMA warps only; 0: no triples path). The instantiated
//! (S, S3) pairs are those launchFc1TmaC chooses: (6, 4), (4, 3), (4, 2), (3, 2).
template <int C, int S>
cudaError_t launchFc1TmaCS(int smem, int tripleStages, bool pdl, DraftRoutedMoeParams const& params,
    DraftRoutedMoeTensorMaps const& maps, cudaStream_t stream)
{
    int const n = params.fc1Ctas;
    CUtensorMap const& w = maps.w13;
    switch (params.fc1Warps)
    {
    case 1: return launchTma2(mtpMoeFc1TmaKernel<1, C, S, 0>, n, 64, smem, pdl, params, w, w, stream);
    case 2: return launchTma2(mtpMoeFc1TmaKernel<2, C, S, 0>, n, 96, smem, pdl, params, w, w, stream);
    case 4:
    {
        CUtensorMap const& w3 = maps.w13Triple;
        if (tripleStages == 0)
        {
            return launchTma2(mtpMoeFc1TmaKernel<4, C, S, 0>, n, 160, smem, pdl, params, w, w, stream);
        }
        if constexpr (S == 6)
        {
            if (tripleStages == 4)
            {
                return launchTma2(mtpMoeFc1TmaKernel<4, C, 6, 4>, n, 160, smem, pdl, params, w, w3, stream);
            }
        }
        if constexpr (S == 4)
        {
            if (tripleStages == 3)
            {
                return launchTma2(mtpMoeFc1TmaKernel<4, C, 4, 3>, n, 160, smem, pdl, params, w, w3, stream);
            }
            if (tripleStages == 2)
            {
                return launchTma2(mtpMoeFc1TmaKernel<4, C, 4, 2>, n, 160, smem, pdl, params, w, w3, stream);
            }
        }
        if constexpr (S == 3)
        {
            if (tripleStages == 2)
            {
                return launchTma2(mtpMoeFc1TmaKernel<4, C, 3, 2>, n, 160, smem, pdl, params, w, w3, stream);
            }
        }
        return cudaErrorInvalidValue;
    }
    default: return cudaErrorInvalidValue;
    }
}

//! FC1 ring depth: the deepest of 6 / 4 / 3 stages of 32 KB that fits next to the activation rows; with
//! kFlagFc1Triples and 4 MMA warps also the deepest triples ring (48 KB stages) of the instantiated pairs that fits.
//! kFlagStreamX (4 MMA warps, tile pairs; ignored otherwise): 6 stages of 32 + 2 KB and 4 triples stages of 48 + 2 KB
//! for any row count.
template <int C>
cudaError_t launchFc1TmaC(
    bool pdl, DraftRoutedMoeParams const& params, DraftRoutedMoeTensorMaps const& maps, cudaStream_t stream)
{
    if constexpr (C == 2)
    {
        if ((params.flags & kFlagStreamX) != 0 && params.fc1Warps == 4)
        {
            int const smem = fc1TmaStreamXSmemBytes();
            return launchTma3(mtpMoeFc1TmaKernel<4, 2, 6, 4, true>, params.fc1Ctas, 160, smem, pdl, params, maps.w13,
                maps.w13Triple, maps.x, stream);
        }
    }
    bool const triples = (params.flags & kFlagFc1Triples) != 0 && params.fc1Warps == 4;
    for (int stages : {6, 4, 3})
    {
        int smem = fc1TmaSmemBytes(params.rows, stages);
        if (smem > kMaxDynamicSmem)
        {
            continue;
        }
        int tripleStages = 0;
        if (triples)
        {
            for (int t : {4, 3, 2})
            {
                bool const instantiated
                    = (stages == 6 && t == 4) || (stages == 4 && (t == 3 || t == 2)) || (stages == 3 && t == 2);
                if (instantiated && fc1TmaTriplesSmemBytes(params.rows, stages, t) <= kMaxDynamicSmem)
                {
                    tripleStages = t;
                    smem = fc1TmaTriplesSmemBytes(params.rows, stages, t);
                    break;
                }
            }
            if (tripleStages == 0)
            {
                return cudaErrorInvalidValue;
            }
        }
        switch (stages)
        {
        case 6: return launchFc1TmaCS<C, 6>(smem, tripleStages, pdl, params, maps, stream);
        case 4: return launchFc1TmaCS<C, 4>(smem, tripleStages, pdl, params, maps, stream);
        default: return launchFc1TmaCS<C, 3>(smem, tripleStages, pdl, params, maps, stream);
        }
    }
    return cudaErrorInvalidValue;
}

template <int W>
cudaError_t launchFc2TmaW(bool pdl, DraftRoutedMoeParams const& params, CUtensorMap const& map, cudaStream_t stream)
{
    constexpr int kS = fc2TmaStages<W>();
    int const smem = fc2TmaSmemBytes(W, kS);
    return (params.flags & kFlagFc2Pairs)
        ? launchTma(mtpMoeFc2TmaKernel<W, kS, 2>, params.fc2Ctas, (W + 1) * 32, smem, pdl, params, map, stream)
        : launchTma(mtpMoeFc2TmaKernel<W, kS, 1>, params.fc2Ctas, (W + 1) * 32, smem, pdl, params, map, stream);
}

cudaError_t launchTmaPair(DraftRoutedMoeParams const& params, DraftRoutedMoeTensorMaps const& maps, cudaStream_t stream)
{
    bool const pdl = (params.flags & kFlagPdl) != 0;
    cudaError_t status = (params.flags & kFlagFc1Pairs) ? launchFc1TmaC<2>(pdl, params, maps, stream)
                                                        : launchFc1TmaC<1>(pdl, params, maps, stream);
    if (status != cudaSuccess)
    {
        return status;
    }
    switch (params.fc2Warps)
    {
    case 1: return launchFc2TmaW<1>(pdl, params, maps.w2, stream);
    case 2: return launchFc2TmaW<2>(pdl, params, maps.w2, stream);
    case 3: return launchFc2TmaW<3>(pdl, params, maps.w2, stream);
    case 4: return launchFc2TmaW<4>(pdl, params, maps.w2, stream);
    default: return cudaErrorInvalidValue;
    }
}

} // namespace

cudaError_t invokeDraftRoutedMoe(
    DraftRoutedMoeParams const& params, DraftRoutedMoeTensorMaps const* maps, cudaStream_t stream)
{
    if (params.rows < 1 || params.rows > kMaxRows || params.fc1Ctas < 1 || params.fc2Ctas < 1)
    {
        return cudaErrorInvalidValue;
    }
    if (params.flags & kFlagTma)
    {
        return maps != nullptr ? launchTmaPair(params, *maps, stream) : cudaErrorInvalidValue;
    }
    bool const pdl = (params.flags & kFlagPdl) != 0;
    bool const pairs1 = (params.flags & kFlagFc1Pairs) != 0;
    bool const pairs2 = (params.flags & kFlagFc2Pairs) != 0;
    bool const deep = (params.flags & kFlagDeepRing) != 0;
    int const smem1 = fc1SmemBytes(params.rows, params.fc1Warps, deep);
    cudaError_t status = deep ? launchFc1Warps<kFc1StagesDeep>(pairs1, smem1, pdl, params, stream)
                              : launchFc1Warps<kFc1Stages>(pairs1, smem1, pdl, params, stream);
    if (status != cudaSuccess)
    {
        return status;
    }
    int const smem2 = fc2SmemBytes(params.fc2Warps, deep);
    return deep ? launchFc2Warps<kFc2StagesDeep>(pairs2, smem2, pdl, params, stream)
                : launchFc2Warps<kFc2Stages>(pairs2, smem2, pdl, params, stream);
}

} // namespace draft_routed_moe

TRTLLM_MV2_NAMESPACE_END
