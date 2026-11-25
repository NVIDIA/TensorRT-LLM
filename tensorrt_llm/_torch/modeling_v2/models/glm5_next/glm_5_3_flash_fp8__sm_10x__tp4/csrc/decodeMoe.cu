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
// Decode MoE (M <= 4 rows): one persistent warp-specialized kernel per call, one CTA per SM.
//
// Shapes (GLM-5.3-Flash at moe_tp4): H = 4096, I = 512, E = 288, top-8 + the shared expert.
// w31 [E, 2I, H] e4m3 = [up; gate] rows, s31 [E, 2I/128, H/128]; w2 [E, H, I] e4m3, s2 [E, H/128, I/128].
// In the code, M is numRows, G (the grid) numCtas and U (the expert slots) numSlots.
//
// Every CTA of the grid must be resident at once: FC2 CTAs spin on counters that other CTAs' FC1 epilogues fill,
// and the shared memory admits one CTA per SM. decodeMoe() checks that the device can hold the grid and returns
// cudaErrorCooperativeLaunchTooLarge otherwise; SMs held by other work at launch time are not visible to it.
//
// The op is a stream of U x 6.29 MB of FP8 weights (U = distinct experts, the shared one included). Every CTA
// computes the routing itself and derives the same static schedule from it (no work counter, no inter-CTA
// hand-off of the routing):
//   FC1: a tile = 4 up + 4 gate rows (all of H, 32 KB) of one expert, global tile order (expert slot, tile), the
//        shared expert first. Tiles [0, tl) go round robin over the G CTAs (tile j -> CTA j % G); the tail
//        [tl, U x 128) goes to the G - 128 CTAs without FC2 rows, about the bytes of the FC2 work they skip, less
//        a margin so that the last experts' activations are out before FC2 needs them.
//   FC2: CTA c < 128 owns output rows [32 c, 32 c + 32) and walks all U experts (two per stage), summing them in
//        a fixed order (deterministic, no atomics on the output).
// Routing: warp t < M takes token row t, top-8 by 8 rounds of a warp max (the winner lane drops its key); warp 0
// then numbers the distinct experts in first-occurrence order with match.any (slot 0 = the shared expert). Before
// griddepcontrol.wait (only weights are read there) CTAs c < 128 issue the shared expert's FC1 tile c and
// prefetch 16 KB of its w2 into L2; the first stages after row 0's top-8 are issued before the other rows finish.
// Warp roles: 8 MMA consumer warps, one producer warp (the CTA's whole stage list, FC1 then FC2 weights, through
// a TMA bulk ring: FC2 weights are fetched while FC1 still computes), one act-loader warp, kNEpi FC1-epilogue
// warps. An FC1 tile's epilogue sums the consumer warps' K slices, applies the clamped SwiGLU and the routing
// weight, publishes the M activation rows (f16 hi + lo) to global scratch and bumps the expert slot's counter
// (release); the act loader waits for counter == 128 (acquire) and copies the activation into a small act ring.
// The counters are double-buffered by call parity (per-CTA call epochs): a call zeroes the set the next call uses.
//
// Arithmetic: e4m3 -> f16 (exact, cvt.rn.f16x2.e4m3x2), BF16 x -> f16, mma.sync m16n8k16 f16 -> f32, every
// 128-wide K block's partial scaled by its block scale in f32.
// K permutation: in a 64-wide K block, lane (g = lane / 4, t4 = lane % 4) reads bytes [16 t4, 16 t4 + 16) of
// its weight rows; MMA chunk j takes bytes 16 t4 + 4 j + {0, 1} as A columns {2 t4, 2 t4 + 1} and + {2, 3} as
// {2 t4 + 8, 2 t4 + 9}; the B fragment (column g = token) reads the activation at the same K positions.
#include "glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/csrc/decodeMoe.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

constexpr int kH = kDecodeMoeHidden, kI = kDecodeMoeInter, kE = kDecodeMoeExperts, kTopK = 8;
constexpr int kMaxM = kDecodeMoeMaxRows;
constexpr int kMaxU = 1 + kTopK * kMaxM; // 33
constexpr int kTilesPerExpert = kI / 4;  // 128 FC1 tiles of 4 up + 4 gate rows
constexpr int kCWarps = 8, kConsumers = kCWarps * 32;
constexpr int kProducerWarp = kCWarps, kActWarp = kCWarps + 1, kEpiWarp = kCWarps + 2;
constexpr int kNEpi = 6;           // FC1 epilogue warps (round robin over tiles: their release fences overlap)
constexpr int kThreads = kConsumers + 64 + 32 * kNEpi;
constexpr int kNS = 5;             // weight ring slots
constexpr int kEarly = kNS;        // FC1 stages issued right after row 0's top-8 (expert slots <= 5)
constexpr int kEarlyPicks = 5;     // row 0's picks those stages need (slots 1..5)
constexpr int kFc2Ctas = 128;      // CTAs with FC2 rows (32 each)
constexpr int kMinFc1OnlyCtas = 4; // the fewest CTAs without FC2 rows the tail split runs with
// The largest grid: B200's 148 SMs, the grid the schedule (kTailMargin, kEarly) was tuned and certified on. The
// early stages need it to stay small: they are the CTA's tiles c + s G, s < kEarly, issued once row 0's first
// kEarlyPicks picks are known, so those tiles must lie in expert slots <= kEarlyPicks.
constexpr int kMaxCtas = 148;
constexpr int kTailMargin = 10;               // FC1 tiles kept off the FC1-only CTAs' tail
constexpr int kSlotW = 32768;                 // weight bytes per slot
constexpr int kSlotBytes = kSlotW + 256;      // + scales
constexpr int kNA = 7;                        // activation ring slots (the rest of the shared memory)
constexpr int kActPitch = 520, kActPlane = kMaxM * kActPitch;
constexpr int kActBytes = 2 * kActPlane * 2;  // padded [hi, lo][kMaxM][kActPitch] f16
constexpr int kRed2Stride = 5;
constexpr int kFc2Pitch = 16384;              // one expert's <= 32 rows x 512 in a slot
constexpr int kMaxRows2 = 32;
constexpr int kCntPad = 64;                   // counters 256 B apart
constexpr int kRed1Floats = kCWarps * 16 * 2; // FC1 hand-off: [warp][g * 2 + t4 (< 2)][2]
constexpr int kSmem = kNS * kSlotBytes + kNA * kActBytes + kNEpi * kRed1Floats * 4;
// FC1: a consumer warp's K slice (of all of H), its 64-wide blocks
constexpr int kKW1 = kH / kCWarps, kKB1 = kKW1 / 64;
static_assert(kMaxM * kH * 2 <= kNA * kActBytes, "x is staged in the act ring");
// FC2: warp = (K part kp, m-tile mt), K part width, its 64-wide blocks
static_assert(kKB1 == 8 && kCWarps == 8, "warp split (FC1: 512-wide K slices; FC2: 64-wide K parts)");
static_assert(kCWarps * 32 * kRed2Stride <= kNEpi * kRed1Floats, "FC2 K-part reduction fits the FC1 hand-off buffers");
static_assert(kEarly * kMaxCtas <= (kEarlyPicks + 1) * kTilesPerExpert,
    "the early stages' tiles lie in the expert slots row 0's first kEarlyPicks picks fill");
static_assert(kFc2Ctas + kMinFc1OnlyCtas <= kMaxCtas, "the smallest grid fits under the largest");

//! The kernel's arguments: DecodeMoeArgs' operands plus the device-global scratch.
struct MoeParams
{
    __nv_bfloat16 const* x; //!< [M, H]
    float const* logits;    //!< [M, E]
    float const* bias;      //!< [E]
    uint8_t const* w31;     //!< [E, 2I, H] e4m3, [up; gate]
    float const* s31;       //!< [E, 2I/128, H/128]
    uint8_t const* sw31;    //!< [2I, H] e4m3, the shared expert
    float const* ss31;      //!< [2I/128, H/128]
    uint8_t const* w2;      //!< [E, H, I] e4m3
    float const* s2;        //!< [E, H/128, I/128]
    uint8_t const* sw2;     //!< [H, I] e4m3, the shared expert
    float const* ss2;       //!< [H/128, I/128]
    __half* act;            //!< [kMaxU][2][kMaxM][I]
    //! [2][kMaxU] finished FC1 tiles per expert slot: set (call parity) of this call, the other is reset
    int* cnt;
    unsigned* epoch;    //!< [CTA] calls so far (every CTA runs every call: they agree on the parity)
    __nv_bfloat16* out; //!< [M, H]
    float scaling;      //!< the routed scaling factor
    float limit;        //!< the SwiGLU clamp
    int numRows;        //!< M
};

__device__ __forceinline__ uint32_t e4m3x2ToF16x2(uint32_t v16)
{
    uint32_t r;
    uint16_t const h = static_cast<uint16_t>(v16);
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(h));
    return r;
}

__device__ __forceinline__ void mma16816(
    float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t word(uint4 const& v, int j)
{
    return j == 0 ? v.x : j == 1 ? v.y : j == 2 ? v.z : v.w;
}

__device__ __forceinline__ uint32_t smemU32(void const* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbarInit(uint64_t* bar, int count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smemU32(bar)), "r"(count));
}

__device__ __forceinline__ void mbarInitFence()
{
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void mbarExpectTx(uint64_t* bar, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smemU32(bar)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbarArrive(uint64_t* bar)
{
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smemU32(bar)) : "memory");
}

__device__ __forceinline__ void mbarWait(uint64_t* bar, uint32_t parity)
{
    asm volatile(
        "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}" ::
            "r"(smemU32(bar)),
        "r"(parity)
        : "memory");
}

__device__ __forceinline__ uint64_t evictFirstPolicy()
{
    uint64_t pol;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    return pol;
}

// TMA bulk copy global -> shared, completing `bytes` on bar
// weights: streamed once, evict-first in L2 (keeps the prefetched / published data there)
__device__ __forceinline__ void bulkG2s(void* dst, void const* src, uint32_t bytes, uint64_t* bar, uint64_t pol)
{
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::
            "r"(smemU32(dst)),
        "l"(src), "r"(bytes), "r"(smemU32(bar)), "l"(pol)
        : "memory");
}

__device__ __forceinline__ void bulkG2s(void* dst, void const* src, uint32_t bytes, uint64_t* bar)
{
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(smemU32(dst)),
        "l"(src), "r"(bytes), "r"(smemU32(bar))
        : "memory");
}

__device__ __forceinline__ void l2Prefetch(void const* src, uint32_t bytes)
{
    uint64_t pol;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
    asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" ::"l"(src), "r"(bytes), "l"(pol)
                 : "memory");
}

__device__ __forceinline__ int ldAcquire(int const* p)
{
    int v;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void fenceProxyAsyncGlobal()
{
    asm volatile("fence.proxy.async.global;" ::: "memory");
}

__device__ __forceinline__ void redReleaseAdd(int* p, int v)
{
    asm volatile("red.release.gpu.global.add.s32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void consumerSync()
{
    asm volatile("bar.sync 1, %0;" ::"n"(kConsumers) : "memory");
}

// The largest key not in mask (lowest index on ties; 0 if none): a depth-4 tree.
__device__ __forceinline__ void bestOf9(
    uint32_t const (&key)[9], float const (&sc)[9], unsigned mask, uint32_t& bk, int& bi, float& bs)
{
    uint32_t v[9];
    int ix[9];
    float sv[9];
#pragma unroll
    for (int i = 0; i < 9; i++)
    {
        v[i] = ((mask >> i) & 1u) ? 0u : key[i];
        ix[i] = i;
        sv[i] = sc[i];
    }
#pragma unroll
    for (int st = 1; st < 16; st <<= 1)
    {
#pragma unroll
        for (int i = 0; i + st < 9; i += 2 * st)
        {
            bool const hi = v[i + st] > v[i]; // ties keep the lower index i
            v[i] = hi ? v[i + st] : v[i];
            ix[i] = hi ? ix[i + st] : ix[i];
            sv[i] = hi ? sv[i + st] : sv[i];
        }
    }
    bk = v[0];
    bi = ix[0];
    bs = sv[0];
}

__device__ __forceinline__ uint32_t f2ord(float f)
{
    uint32_t const b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__global__ void __launch_bounds__(kThreads, 1) moeKernel(MoeParams p)
{
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* ring = smem;
    uint8_t* actRing = smem + kNS * kSlotBytes;
    float* red
        = reinterpret_cast<float*>(actRing + kNA * kActBytes); // kNEpi x kRed1Floats (FC1 hand-off; FC2 K-part sums)

    __shared__ __align__(8) uint64_t full[kNS], empty[kNS], actFull[kNA], actEmpty[kNA], redFull[kNEpi],
        redEmpty[kNEpi], xDone, fin;
    __shared__ int sList[kMaxU];             // expert of each slot (-1 = shared)
    __shared__ float sWtab[kMaxU * kMaxM];
    __shared__ int sIds[kMaxM * kTopK + 32]; // + a dummy slot per lane
    __shared__ float sWts[kMaxM * kTopK + 32];
    __shared__ int sNumSlots;
    __shared__ unsigned sParity;

    int const tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, t4 = lane % 4;
    int const numRows = p.numRows, numCtas = gridDim.x, c = blockIdx.x;

    if (tid == kProducerWarp * 32)
    {
        for (int i = 0; i < kNS; i++)
        {
            mbarInit(&full[i], 1);
            mbarInit(&empty[i], kCWarps);
        }
        for (int i = 0; i < kNA; i++)
        {
            mbarInit(&actFull[i], 1); // the act loader's arrive.expect_tx (one bulk copy per act)
            mbarInit(&actEmpty[i], kCWarps);
        }
        for (int i = 0; i < kNEpi; i++)
        {
            mbarInit(&redFull[i], kCWarps);
            mbarInit(&redEmpty[i], 1);
        }
        mbarInit(&xDone, 1);
        mbarInit(&fin, 1 + kNEpi); // the act loader and the epilogue warps are done (FC1 hand-off buffers free)
        mbarInitFence();
    }

    // ---- producer helper (producer lane 0 only) ----
    uint64_t const pol = evictFirstPolicy();
    auto issueFc1 = [&](int s, int j, int e) { // stage s: tile j (expert e, -1 = shared)
        int const slot = s % kNS, t = j % kTilesPerExpert;
        uint8_t const* weights = e < 0 ? p.sw31 : p.w31 + static_cast<size_t>(e) * (2 * kI) * kH;
        float const* scales = e < 0 ? p.ss31 : p.s31 + static_cast<size_t>(e) * (2 * kI / 128) * (kH / 128);
        uint8_t* dst = ring + slot * kSlotBytes;
        uint64_t* bar = &full[slot];
        mbarExpectTx(bar, kSlotW + 256);
        bulkG2s(dst, weights + static_cast<size_t>(4 * t) * kH, 4 * kH, bar, pol);               // up rows 4t .. 4t + 3
        bulkG2s(dst + 4 * kH, weights + static_cast<size_t>(kI + 4 * t) * kH, 4 * kH, bar, pol); // gate rows
        bulkG2s(dst + kSlotW, scales + (t / 32) * (kH / 128), 128, bar);
        bulkG2s(dst + kSlotW + 128, scales + (4 + t / 32) * (kH / 128), 128, bar);
    };

    // The shared expert's tiles are global tiles 0..127: CTAs c < 128 start them before the routing, and
    // prefetch the shared expert's w2 (16 KB each) into L2 for the first FC2 stage.
    if (tid == kProducerWarp * 32 && c < kTilesPerExpert)
    {
        issueFc1(0, c, -1);
        l2Prefetch(p.sw2 + static_cast<size_t>(c) * (kH * kI / kTilesPerExpert), kH * kI / kTilesPerExpert);
    }

    cudaGridDependencySynchronize(); // x, logits, bias are produced by the previous kernel

    // ---- routing, part 1: warp t < M takes token row t (logits loads first) ----
    float sc[9];
    uint32_t key[9];
    if (warp < numRows)
    {
        float lg[9], bs[9];
#pragma unroll
        for (int i = 0; i < 9; i++)
        {
            lg[i] = __ldg(p.logits + warp * kE + lane + 32 * i);
            bs[i] = __ldg(p.bias + lane + 32 * i);
        }
        // The score is the trtllm-gen routing kernels' sigmoid_accurate (RoutingKernel.cuh), the key its sum with
        // the bias as in their SigmoidBias preprocess: the steps above 4 rows route through those kernels, and a
        // different sigmoid could swap two experts within an ulp of each other between the two paths.
#pragma unroll
        for (int i = 0; i < 9; i++)
        {
            sc[i] = 0.5f * tanhf(0.5f * lg[i]) + 0.5f;
            key[i] = f2ord(sc[i] + bs[i]);
        }
    }

    // x -> the (still unused) act ring, for the consumers' B fragments
    __shared__ __align__(8) uint64_t xBar;
    if (tid == kProducerWarp * 32)
    {
        mbarInit(&xBar, 1);
        mbarInitFence();
        mbarExpectTx(&xBar, numRows * kH * 2);
        bulkG2s(actRing, p.x, numRows * kH * 2, &xBar);
    }
    for (int i = tid; i < kMaxU * kMaxM; i += kThreads)
    {
        sWtab[i] = 0.f;
    }

    if (warp < numRows)
    {
        // top-8 of the keys (ties to the lower expert): 8 rounds of a warp max over the lanes' best keys; the winner
        // lane records its expert and score in slot k and replaces its best with the best of its keys not taken yet.
        // That next best (a depth-4 tree over the 9 keys) is computed by every lane while the round's reduction is in
        // flight, so a round's critical path is redux -> ballot -> select.
        unsigned taken = 0u; // bit i: key i is in the top-8
        uint32_t k1;
        int i1;
        float s1;
        bestOf9(key, sc, 0u, k1, i1, s1);
#pragma unroll
        for (int k = 0; k < kTopK; k++)
        {
            uint32_t m;
            // (taken passes through the asm so that the next-best tree below is scheduled after the reduction)
            asm volatile("redux.sync.max.u32 %0, %2, 0xffffffff;" : "=r"(m), "+r"(taken) : "r"(k1));
            uint32_t nk;
            int ni;
            float ns;
            bestOf9(key, sc, taken | (1u << i1), nk, ni, ns);
            asm volatile("" : "+r"(nk), "+r"(ni), "+f"(ns)); // every lane, outside the winner's path
            unsigned const tied = __ballot_sync(0xffffffffu, k1 == m);
            bool win = k1 == m;
            if (tied & (tied - 1))
            { // several lanes hold the max: the lowest expert wins
                uint32_t const cand = win ? static_cast<uint32_t>(lane + 32 * i1) : 0xffffffffu;
                win = cand == __reduce_min_sync(0xffffffffu, cand);
            }
            { // the losers write a private dummy slot (no branch)
                int const o = win ? warp * kTopK + k : kMaxM * kTopK + lane;
                sIds[o] = lane + 32 * i1;
                sWts[o] = s1;
            }
            // row 0's picks 0, 1 (expert slots 1, 2), then 2 .. 4 (slots 3 .. 5): release the producer (one-shot
            // barriers)
            if (warp == 0 && k == 1)
            {
                __syncwarp();
                asm volatile("bar.arrive 3, 64;" ::: "memory");
            }
            if (warp == 0 && k == kEarlyPicks - 1)
            {
                __syncwarp();
                asm volatile("bar.arrive 4, 64;" ::: "memory");
            }
            taken = win ? taken | (1u << i1) : taken;
            k1 = win ? nk : k1;
            i1 = win ? ni : i1;
            s1 = win ? ns : s1;
        }
    }
    if (warp == kProducerWarp)
    {
        // The CTA's first stages after the routing are its tiles c + s G, s < kEarly: expert slots <= 5, i.e. row 0's
        // first picks (slot u = row 0's pick u - 1). Issue them as soon as row 0 is done, before the other rows,
        // the expert numbering and the block barriers.
        int s = c < kTilesPerExpert ? 1 : 0;
        asm volatile("bar.sync 3, 64;" ::: "memory"); // with warp 0: sIds[0, 1] are written
        if (lane == 0)
        {
            for (; s < kEarly && c + s * numCtas < 3 * kTilesPerExpert; s++)
            {
                int const j = c + s * numCtas, u = j / kTilesPerExpert;
                issueFc1(s, j, u == 0 ? -1 : sIds[u - 1]);
            }
        }
        asm volatile("bar.sync 4, 64;" ::: "memory"); // sIds[0 .. kEarlyPicks - 1]
        if (lane == 0)
        {
            for (; s < kEarly; s++)
            {
                int const j = c + s * numCtas, u = j / kTilesPerExpert;
                issueFc1(s, j, u == 0 ? -1 : sIds[u - 1]);
            }
        }
    }
    else
    {
        if (warp == kActWarp && lane == 0)
        { // this call's counter set; the CTA's next call uses the other one
            unsigned const e = p.epoch[c];
            p.epoch[c] = e + 1u;
            sParity = e & 1u;
        }
        // barriers initialized, sWtab zeroed, sIds / sWts written (every warp but the producer's)
        asm volatile("bar.sync 2, %0;" ::"n"(kThreads - 32) : "memory");
    }
    // ---- routing, part 2 (warp 0): distinct experts in first-occurrence order, slot 0 = the shared expert ----
    if (warp == 0)
    {
        int const np = numRows * kTopK; // <= 32: lane = pair (token lane / 8, rank lane % 8)
        int const id = lane < np ? sIds[lane] : -2 - lane;
        float w = lane < np ? sWts[lane] : 0.f;
        float tot = w;
        tot += __shfl_xor_sync(0xffffffffu, tot, 1);
        tot += __shfl_xor_sync(0xffffffffu, tot, 2);
        tot += __shfl_xor_sync(0xffffffffu, tot, 4);
        unsigned const same = __match_any_sync(0xffffffffu, id);
        int const first = __ffs(same) - 1;
        bool const isFirst = first == lane && lane < np;
        unsigned const firsts = __ballot_sync(0xffffffffu, isFirst);
        int const slot = 1 + __popc(firsts & ((1u << first) - 1u));
        if (lane < np)
        {
            sWtab[slot * kMaxM + lane / kTopK] = w / tot * p.scaling;
            if (isFirst)
            {
                sList[slot] = id;
            }
        }
        if (lane == 0)
        {
            sList[0] = -1;
            sNumSlots = 1 + __popc(firsts);
        }
        if (lane < numRows)
        {
            sWtab[lane] = 1.f; // the shared expert, weight 1
        }
    }
    __syncthreads();

    // ---- the CTA's schedule ----
    // FC2: CTAs c < kFc2Ctas own 32 rows each (whole MMA tiles). The other nf CTAs only run FC1: besides their
    // round-robin share of tiles [0, tl) they take the tail [tl, T1) (about the bytes of the FC2 work they skip,
    // less a margin so that the last experts' activations are out before FC2 needs them).
    int const numSlots = sNumSlots;
    int* const cnt = p.cnt + sParity * (kMaxU * kCntPad);
    int const numFc1Tiles = numSlots * kTilesPerExpert;
    int const nf = numCtas - kFc2Ctas;
    int const tl = max(
        numFc1Tiles - max(nf * numSlots / 2 - kTailMargin, 0), kTilesPerExpert); // tiles < 128 go out before routing
    int const nm = c < tl ? (tl - 1 - c) / numCtas + 1 : 0;
    int const ft = c - kFc2Ctas;                                                 // rank among the FC1-only CTAs
    int const nt = (ft >= 0 && tl + ft < numFc1Tiles) ? (numFc1Tiles - 1 - tl - ft) / nf + 1 : 0;
    int const n1 = nm + nt;
    auto tileOf = [&](int k) { return k < nm ? c + k * numCtas : tl + ft + (k - nm) * nf; };
    int const row0 = c < kFc2Ctas ? 32 * c : 0;
    int const n2 = c < kFc2Ctas ? 32 : 0;
    int const numFc1Stages = n1;
    int const numFc2Stages = n2 > 0 ? (numSlots + 1) / 2 : 0;
    int const rb0 = row0 >> 7;
    int const nsc = n2 > 0 ? 1 : 0;

    if (warp == kProducerWarp)
    {
        // ===== producer: the CTA's FC1 stages, then its FC2 stages (two experts each) =====
        if (lane == 0)
        {
            int const s0 = c < kTilesPerExpert ? 1 : 0;
#pragma unroll 1
            for (int s = max(s0, kEarly); s < numFc1Stages + numFc2Stages; s++)
            {
                int const slot = s % kNS;
                if (s >= kNS)
                {
                    mbarWait(&empty[slot], ((s / kNS) - 1) & 1);
                }
                if (s < numFc1Stages)
                {
                    int const j = tileOf(s);
                    issueFc1(s, j, sList[j / kTilesPerExpert]);
                }
                else
                {
                    int const pr = s - numFc1Stages;
                    uint8_t* dst = ring + slot * kSlotBytes;
                    uint64_t* bar = &full[slot];
                    int const ne = (2 * pr + 1 < numSlots) ? 2 : 1;
                    mbarExpectTx(bar, ne * (n2 * 512 + nsc * 16));
#pragma unroll 1
                    for (int i = 0; i < ne; i++)
                    {
                        int const e = sList[2 * pr + i];
                        uint8_t const* weights = (e < 0 ? p.sw2 : p.w2 + static_cast<size_t>(e) * kH * kI)
                            + static_cast<size_t>(row0) * kI;
                        float const* scales = (e < 0 ? p.ss2 : p.s2 + static_cast<size_t>(e) * (kH / 128) * (kI / 128))
                            + rb0 * (kI / 128);
                        bulkG2s(dst + i * kFc2Pitch, weights, n2 * 512, bar, pol);
                        bulkG2s(dst + kSlotW + i * 32, scales, nsc * 16, bar);
                    }
                }
            }
        }
        __syncwarp();
    }
    else if (warp == kActWarp)
    {
        // the other counter set (used by the previous call, used again by the next one) back to zero
        if (c == 0)
        {
            for (int i = lane; i < kMaxU; i += 32)
            {
                p.cnt[(sParity ^ 1u) * (kMaxU * kCntPad) + i * kCntPad] = 0;
            }
        }
        // ===== act loader: lane l polls expert slot l's FC1 counter (lane 0 also slot 32) =====
        if (n2 > 0)
        {
            uint32_t ready = 0u;
            bool ready32 = false;
#pragma unroll 1
            for (int u = 0; u < numSlots; u++)
            {
#pragma unroll 1
                while (!(u < 32 ? ((ready >> u) & 1u) : ready32))
                {
                    bool const mine = lane < numSlots && !((ready >> lane) & 1u);
                    bool const ok = mine && ldAcquire(cnt + lane * kCntPad) >= kTilesPerExpert;
                    ready |= __ballot_sync(0xffffffffu, ok);
                    if (numSlots > 32 && !ready32)
                    {
                        ready32
                            = __shfl_sync(0xffffffffu,
                                  static_cast<int>(lane == 0 && ldAcquire(cnt + 32 * kCntPad) >= kTilesPerExpert), 0)
                            != 0;
                    }
                    __syncwarp();
                    if (!(u < 32 ? ((ready >> u) & 1u) : ready32))
                    {
                        __nanosleep(256);
                    }
                }
                // the whole [hi, lo][kMaxM][I] block (rows >= M only feed discarded MMA columns)
                int const as = u % kNA;
                if (u == 0)
                {
                    mbarWait(&xDone, 0);
                }
                if (u >= kNA)
                {
                    mbarWait(&actEmpty[as], ((u / kNA) - 1) & 1);
                }
                // one bulk copy: the lane that saw the counter complete acquired it, __syncwarp orders that before lane
                // 0; the generic-proxy act stores of the other CTAs' epilogues are then made visible to the async proxy
                if (lane == 0)
                {
                    fenceProxyAsyncGlobal();
                    mbarExpectTx(&actFull[as], kActBytes);
                    bulkG2s(actRing + as * kActBytes, p.act + static_cast<size_t>(u) * 2 * kActPlane, kActBytes,
                        &actFull[as]);
                }
                __syncwarp();
            }
        }
        if (lane == 0)
        {
            mbarArrive(&fin);
        }
    }
    else if (warp >= kEpiWarp)
    {
        // ===== FC1 epilogue: sum the warps' K slices, clamped SwiGLU, times the routing weight, publish =====
        int const ew = warp - kEpiWarp;
#pragma unroll 1
        for (int k = ew; k < n1; k += kNEpi)
        {
            int const j = tileOf(k), u = j / kTilesPerExpert, t = j % kTilesPerExpert;
            int const rb = k % kNEpi;
            mbarWait(&redFull[rb], (k / kNEpi) & 1);
            float const* rd = red + rb * kRed1Floats;
            // lane (g < 4, t4 < 2): output row 4 t + g, tokens 2 t4, 2 t4 + 1; up = MMA row g, gate = MMA row g + 4
            float up[2] = {0.f, 0.f}, gate[2] = {0.f, 0.f};
            bool const mine = g < 4 && t4 < 2;
            if (mine)
            {
#pragma unroll
                for (int w = 0; w < kCWarps; w++)
                {
                    float2 const vu = *reinterpret_cast<float2 const*>(rd + (w * 16 + g * 2 + t4) * 2);
                    float2 const vg = *reinterpret_cast<float2 const*>(rd + (w * 16 + (g + 4) * 2 + t4) * 2);
                    up[0] += vu.x;
                    up[1] += vu.y;
                    gate[0] += vg.x;
                    gate[1] += vg.y;
                }
            }
            __syncwarp();
            if (lane == 0)
            {
                mbarArrive(&redEmpty[rb]);
            }
            if (mine)
            {
#pragma unroll
                for (int cc = 0; cc < 2; cc++)
                {
                    int const tok = 2 * t4 + cc;
                    if (tok < numRows)
                    {
                        float const gv = fminf(gate[cc], p.limit);
                        float const uu = fminf(fmaxf(up[cc], -p.limit), p.limit);
                        float const act = gv / (1.f + expf(-gv)) * uu * sWtab[u * kMaxM + tok];
                        __half const hi = __float2half_rn(act);
                        __half* dst = p.act + static_cast<size_t>(u) * 2 * kActPlane
                            + static_cast<size_t>(tok) * kActPitch + 4 * t + g;
                        dst[0] = hi;
                        dst[kActPlane] = __float2half_rn(act - __half2float(hi));
                    }
                }
            }
            __syncwarp();
            if (lane == 0)
            {
                redReleaseAdd(cnt + u * kCntPad, 1);
            }
        }
        if (lane == 0)
        {
            mbarArrive(&fin);
        }
    }
    else
    {
        // ===== consumers =====
        // B column g of the FC1 MMA: token g & 3 at K half g >> 2 (A rows 0..7: the tile's rows at K half 0, rows
        // 8..15: the same rows at K half 1; only the matching (row half, column half) quadrants are used).
        // This lane's fragments: x[g & 3][2048 (g >> 2) + 256 warp + 64 kb + 16 t4 + 4 j ..] (bf16 -> f16).
        uint32_t xf[4][4][2];
        mbarWait(&xBar, 0);
        {
            __nv_bfloat16 const* xs = reinterpret_cast<__nv_bfloat16 const*>(actRing);
            int const tt = g & 3, hx = g >> 2;
#pragma unroll
            for (int kb = 0; kb < 4; kb++)
            {
#pragma unroll
                for (int j = 0; j < 4; j++)
                {
                    uint2 raw = make_uint2(0u, 0u);
                    if (tt < numRows)
                    {
                        raw = *reinterpret_cast<uint2 const*>(
                            xs + tt * kH + 2048 * hx + 256 * warp + 64 * kb + 16 * t4 + 4 * j);
                    }
                    __half2 const h0
                        = __float22half2_rn(__bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&raw.x)));
                    __half2 const h1
                        = __float22half2_rn(__bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&raw.y)));
                    uint32_t const u0 = *reinterpret_cast<uint32_t const*>(&h0),
                                   u1 = *reinterpret_cast<uint32_t const*>(&h1);
                    xf[kb][j][0] = u0;
                    xf[kb][j][1] = u1;
                }
            }
        }
        consumerSync(); // every consumer has its x fragments: the act ring may be overwritten
        if (tid == 0)
        {
            mbarArrive(&xDone);
        }
        int s = 0;
        // ---- FC1: a tile = 4 up + 4 gate rows (slot rows 0..7) over all of H; warp = K [256 warp, +256) of each half
        // ----
#pragma unroll 1
        for (int k = 0; k < n1; k++, s++)
        {
            int const slot = s % kNS;
            mbarWait(&full[slot], (s / kNS) & 1);
            uint8_t const* base = ring + slot * kSlotBytes;
            float const* sc1 = reinterpret_cast<float const*>(base + kSlotW) + (g < 4 ? 0 : 32);
            uint8_t const* wrow = base + g * kH + 256 * warp + 16 * t4;
            uint4 wl[4], wh[4];
#pragma unroll
            for (int kb = 0; kb < 4; kb++)
            {
                wl[kb] = *reinterpret_cast<uint4 const*>(wrow + 64 * kb);
                wh[kb] = *reinterpret_cast<uint4 const*>(wrow + 2048 + 64 * kb);
            }
            float sl[2], sh[2];
#pragma unroll
            for (int kc = 0; kc < 2; kc++)
            {
                sl[kc] = sc1[2 * warp + kc];
                sh[kc] = sc1[16 + 2 * warp + kc];
            }
            __syncwarp();
            if (lane == 0)
            {
                mbarArrive(&empty[slot]);
            }
            float blk[4][4];
#pragma unroll
            for (int kb = 0; kb < 4; kb++)
            {
#pragma unroll
                for (int i = 0; i < 4; i++)
                {
                    blk[kb][i] = 0.f;
                }
            }
#pragma unroll
            for (int jj = 0; jj < 4; jj++)
            {
#pragma unroll
                for (int kb = 0; kb < 4; kb++)
                {
                    uint32_t const lo = word(wl[kb], jj), hi = word(wh[kb], jj);
                    mma16816(blk[kb], e4m3x2ToF16x2(lo & 0xffffu), e4m3x2ToF16x2(hi & 0xffffu), e4m3x2ToF16x2(lo >> 16),
                        e4m3x2ToF16x2(hi >> 16), xf[kb][jj][0], xf[kb][jj][1]);
                }
            }
            // lane t4 < 2: row g x tokens 2 t4 + {0, 1} at K half 0 (c0, c1); t4 >= 2: K half 1 (c2, c3), tokens 2 t4 -
            // 4 + {0, 1}
            float acc[2] = {0.f, 0.f};
#pragma unroll
            for (int kc = 0; kc < 2; kc++)
            {
                float const s0 = t4 < 2 ? sl[kc] : sh[kc];
                float const v0 = t4 < 2 ? blk[2 * kc][0] + blk[2 * kc + 1][0] : blk[2 * kc][2] + blk[2 * kc + 1][2];
                float const v1 = t4 < 2 ? blk[2 * kc][1] + blk[2 * kc + 1][1] : blk[2 * kc][3] + blk[2 * kc + 1][3];
                acc[0] = fmaf(v0, s0, acc[0]);
                acc[1] = fmaf(v1, s0, acc[1]);
            }
            acc[0] += __shfl_xor_sync(0xffffffffu, acc[0], 2);
            acc[1] += __shfl_xor_sync(0xffffffffu, acc[1], 2);
            // hand the tile's partial sums to the epilogue warp: red[warp][g * 2 + t4][2]
            int const rb = k % kNEpi;
            if (k >= kNEpi)
            {
                mbarWait(&redEmpty[rb], ((k / kNEpi) - 1) & 1);
            }
            float* rd = red + rb * kRed1Floats;
            if (t4 < 2)
            {
                *reinterpret_cast<float2*>(rd + (warp * 16 + g * 2 + t4) * 2) = make_float2(acc[0], acc[1]);
            }
            __syncwarp();
            if (lane == 0)
            {
                mbarArrive(&redFull[rb]);
            }
        }
        // ---- FC2: warp = K part kp = [64 warp, 64 warp + 64) of I, both m-tiles (the CTA's rows 0..15, 16..31) ----
        int const kp = warp, kq = kp >> 1; // kq: its 128-wide K block
        int rbs[2][2];                     // scale block of rows mt * 16 + g (+ 8), relative to rb0
#pragma unroll
        for (int mt = 0; mt < 2; mt++)
        {
#pragma unroll
            for (int h = 0; h < 2; h++)
            {
                rbs[mt][h] = min(((row0 + mt * 16 + 8 * h + g) >> 7) - rb0, max(nsc - 1, 0));
            }
        }
        float acc2[2][4] = {};
        // a stage = up to two experts: wait for both acts, load every fragment, release the slots, then four
        // independent MMA chains (expert x m-tile)
#pragma unroll 1
        for (int pr = 0; pr < numFc2Stages; pr++, s++)
        {
            int const slot = s % kNS;
            int const ne = (2 * pr + 1 < numSlots) ? 2 : 1;
            mbarWait(&full[slot], (s / kNS) & 1);
            uint8_t const* base = ring + slot * kSlotBytes;
            float const* sc2 = reinterpret_cast<float const*>(base + kSlotW);
            uint4 wg[2][2], wg8[2][2], b0v[2], b1v[2];
            float sA[2][2], sB[2][2];
#pragma unroll
            for (int i = 0; i < 2; i++)
            {
                int const u = 2 * pr + i;
                if (i < ne)
                {
                    int const as = u % kNA;
                    mbarWait(&actFull[as], (u / kNA) & 1);
                    uint8_t const* wr = base + i * kFc2Pitch + g * 512 + kp * 64 + 16 * t4;
#pragma unroll
                    for (int mt = 0; mt < 2; mt++)
                    {
                        wg[i][mt] = *reinterpret_cast<uint4 const*>(wr + mt * 16 * 512);
                        wg8[i][mt] = *reinterpret_cast<uint4 const*>(wr + (mt * 16 + 8) * 512);
                        sA[i][mt] = sc2[i * 8 + rbs[mt][0] * 4 + kq];
                        sB[i][mt] = sc2[i * 8 + rbs[mt][1] * 4 + kq];
                    }
                    // B column g: g < 4 -> hi of token g, g >= 4 -> lo of token g - 4 (= act row g of the [hi, lo][4]
                    // block)
                    uint8_t const* ab = actRing + as * kActBytes + (g * kActPitch + kp * 64 + 16 * t4) * 2;
                    b0v[i] = *reinterpret_cast<uint4 const*>(ab);
                    b1v[i] = *reinterpret_cast<uint4 const*>(ab + 16);
                }
                else
                { // odd U: no second expert (zero operands, zero contribution)
#pragma unroll
                    for (int mt = 0; mt < 2; mt++)
                    {
                        wg[i][mt] = make_uint4(0u, 0u, 0u, 0u);
                        wg8[i][mt] = make_uint4(0u, 0u, 0u, 0u);
                        sA[i][mt] = 0.f;
                        sB[i][mt] = 0.f;
                    }
                    b0v[i] = make_uint4(0u, 0u, 0u, 0u);
                    b1v[i] = make_uint4(0u, 0u, 0u, 0u);
                }
            }
            __syncwarp();
            if (lane == 0)
            {
                mbarArrive(&empty[slot]);
                mbarArrive(&actEmpty[(2 * pr) % kNA]);
                if (ne == 2)
                {
                    mbarArrive(&actEmpty[(2 * pr + 1) % kNA]);
                }
            }
            float bh[2][2][4] = {};
#pragma unroll
            for (int jj = 0; jj < 4; jj++)
            {
#pragma unroll
                for (int i = 0; i < 2; i++)
                {
                    uint4 const& bv = jj < 2 ? b0v[i] : b1v[i];
                    uint32_t const bb0 = (jj & 1) ? bv.z : bv.x, bb1 = (jj & 1) ? bv.w : bv.y;
#pragma unroll
                    for (int mt = 0; mt < 2; mt++)
                    {
                        uint32_t const aw = word(wg[i][mt], jj), aw8 = word(wg8[i][mt], jj);
                        mma16816(bh[i][mt], e4m3x2ToF16x2(aw & 0xffffu), e4m3x2ToF16x2(aw8 & 0xffffu),
                            e4m3x2ToF16x2(aw >> 16), e4m3x2ToF16x2(aw8 >> 16), bb0, bb1);
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < 2; i++)
            {
#pragma unroll
                for (int mt = 0; mt < 2; mt++)
                {
                    acc2[mt][0] = fmaf(bh[i][mt][0], sA[i][mt], acc2[mt][0]);
                    acc2[mt][1] = fmaf(bh[i][mt][1], sA[i][mt], acc2[mt][1]);
                    acc2[mt][2] = fmaf(bh[i][mt][2], sB[i][mt], acc2[mt][2]);
                    acc2[mt][3] = fmaf(bh[i][mt][3], sB[i][mt], acc2[mt][3]);
                }
            }
        }
        // hi + lo: lane t4 < 2 ends up with columns 2 t4 + {0, 1} = tokens 2 t4 + {0, 1}
#pragma unroll
        for (int mt = 0; mt < 2; mt++)
        {
#pragma unroll
            for (int x = 0; x < 4; x++)
            {
                acc2[mt][x] += __shfl_xor_sync(0xffffffffu, acc2[mt][x], 2);
            }
        }
        // sum the 8 K parts (fixed order) in the FC1 hand-off buffers, free once every epilogue warp is done
        mbarWait(&fin, 0);
        float* rd = red; // [K part][row 0..31][token, padded to 5]
        if (t4 < 2)
        {
#pragma unroll
            for (int mt = 0; mt < 2; mt++)
            {
                float* d0 = rd + (kp * 32 + mt * 16 + g) * kRed2Stride + 2 * t4;
                float* d1 = d0 + 8 * kRed2Stride;
                d0[0] = acc2[mt][0];
                d0[1] = acc2[mt][1];
                d1[0] = acc2[mt][2];
                d1[1] = acc2[mt][3];
            }
        }
        consumerSync();
        if (tid < kMaxRows2 * numRows)
        {
            int const row = tid % kMaxRows2, tok = tid / kMaxRows2;
            float v = 0.f;
#pragma unroll
            for (int q = 0; q < kCWarps; q++)
            {
                v += rd[(q * 32 + row) * kRed2Stride + tok];
            }
            if (row < n2)
            {
                p.out[static_cast<size_t>(tok) * kH + row0 + row] = __float2bfloat16(v);
            }
        }
    }
    cudaTriggerProgrammaticLaunchCompletion();
}

constexpr int kEpochSlots = 1024; // per-CTA call epochs, indexed by blockIdx.x
static_assert(kMaxCtas <= kEpochSlots, "every CTA of the largest grid has a call epoch");

__device__ __align__(128) __half actScratch[kMaxU * 2 * kActPlane];
__device__ int fc1Counters[2 * kMaxU * kCntPad];
__device__ unsigned callEpochs[kEpochSlots];

} // namespace

cudaError_t decodeMoe(DecodeMoeArgs const& args, cudaStream_t stream)
{
    // Per device, set by the first call that passes the checks: the grid and the scratch symbols' addresses.
    struct DeviceState
    {
        int grid = 0;
        __half* act = nullptr;
        int* cnt = nullptr;
        unsigned* epoch = nullptr;
    };

    constexpr int kMaxDevices = 64;
    static DeviceState states[kMaxDevices];
    int dev = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess)
    {
        return err;
    }
    if (dev >= kMaxDevices)
    {
        return cudaErrorNotSupported;
    }
    DeviceState& st = states[dev];
    if (st.grid == 0)
    {
        int numSms = 0;
        int ctasPerSm = 0;
        void* act = nullptr;
        void* cnt = nullptr;
        void* epoch = nullptr;
        err = cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, dev);
        if (err == cudaSuccess)
        {
            err = cudaFuncSetAttribute(moeKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
        }
        if (err == cudaSuccess)
        {
            err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctasPerSm, moeKernel, kThreads, kSmem);
        }
        if (err == cudaSuccess)
        {
            err = cudaGetSymbolAddress(&act, actScratch);
        }
        if (err == cudaSuccess)
        {
            err = cudaGetSymbolAddress(&cnt, fc1Counters);
        }
        if (err == cudaSuccess)
        {
            err = cudaGetSymbolAddress(&epoch, callEpochs);
        }
        if (err != cudaSuccess)
        {
            return err;
        }
        // Precondition: every CTA of the grid is resident at once. FC2 CTAs spin on counters that the FC1 epilogues
        // of other CTAs fill, so a CTA left waiting for an SM would hang the launch rather than fail it. The grid is
        // kFc2Ctas CTAs with FC2 rows plus at least kMinFc1OnlyCtas without, one per SM, at most kMaxCtas; the
        // occupancy query times the SM count the runtime reports must hold it. SMs that other kernels occupy at
        // launch time are not visible here: the caller must not run work that waits on this launch concurrently.
        if (numSms < kFc2Ctas + kMinFc1OnlyCtas)
        {
            return cudaErrorNotSupported;
        }
        int const grid = numSms < kMaxCtas ? numSms : kMaxCtas;
        if (ctasPerSm * numSms < grid)
        {
            return cudaErrorCooperativeLaunchTooLarge;
        }
        st.act = static_cast<__half*>(act);
        st.cnt = static_cast<int*>(cnt);
        st.epoch = static_cast<unsigned*>(epoch);
        st.grid = grid;
    }
    if (args.rows < 1 || args.rows > kMaxM)
    {
        return cudaErrorInvalidValue;
    }
    MoeParams p{static_cast<__nv_bfloat16 const*>(args.x), args.logits, args.bias, args.w31, args.s31, args.sw31,
        args.ss31, args.w2, args.s2, args.sw2, args.ss2, st.act, st.cnt, st.epoch,
        static_cast<__nv_bfloat16*>(args.out), args.scaling, args.limit, args.rows};
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(st.grid);
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = kSmem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = args.pdl ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, moeKernel, p);
}

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
