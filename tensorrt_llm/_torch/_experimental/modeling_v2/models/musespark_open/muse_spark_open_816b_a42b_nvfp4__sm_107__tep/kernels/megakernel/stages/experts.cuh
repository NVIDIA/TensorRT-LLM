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

// MoE experts stage of the megakernel (namespace p4a), device code. Included inside namespace mso_mk and the
// device-code guard, after the library's PTX helpers, Ctx, the Lamport family and waitBar.
//
// One unit per CTA (stage kStMoeExperts, unit = 1 + final unit index, or 0) runs the routed experts of the 8 rows on
// this rank's three-window stack [own | r-1 | r-2]:
//
//   1. routing, redundantly in every CTA: warp t = token t polls its 256 router logits (Lamport FP32), sigmoid, the
//      selection key (score + bias), top-8, the renormalized weights -- the built-in _full_routing_kernel's
//      instructions on the values that decide bits;
//   2. the three-window ring placement (windows.place, PLACE_BALANCE3) in 8 lanes, the slot list (distinct local
//      experts in ascending window id) and per slot its tokens;
//   3. the byte cut: E slots give E * 1024 FC1 units and E * 512 FC2 units of 18 KiB (a 128-row tile's k-block of 256:
//      16 KiB E2M1 + 2 KiB scales); CTA c takes [U c / kCtas, U (c + 1) / kCtas) of each list (kMoeFlagTileCut: whole
//      tiles). A tile split over CTAs is finished by the CTA holding its last k-block; the others leave FP32 partials
//      in single-reader slots (part[cta][fc]: each CTA has at most one unfinished tile per list, its last);
//   4. the pre-expert RMSNorm + NVFP4 quantization of all 8 rows (flashinfer rmsnorm_fp4quant's arithmetic; warp t =
//      token t, lane l = 16-element blocks l + 32 j), into the swizzled FC1 B operand and the token-scale staging;
//      lanes 0 of warps 0..kMoeStagesEarly - 1 request the first ring stages while their row loads are in flight, the
//      rows wait in the later stages, which the producer requests after the quantization;
//   5. warp roles: warp 0 lane 0 streams the units through a kMoeStages-deep ring (TMA SW128 box of 128 rows x 128 B +
//      a 2 KiB bulk copy of the scales, evict_first); warp 1 copies the scales to TMEM and issues the block-scaled MMAs
//      (M = 128 weight rows, N = 8 tokens, K = 64 per instruction) into kMoeAccs accumulators; warps 4-7 read the
//      accumulators (TMEM lane quadrants 0-3) and run the epilogues (FC1: partial sum, TRTLLM-gen's SwiGLU + NVFP4
//      requantization in its operation order, the h words into the workspace; FC2: fc2_alpha, BF16 rows and the
//      per-tile sums of (y (1 + w_post))^2); warps 2-3 clear the next launch's workspace copy, then fetch each needed
//      k-block of h of the CTA's FC2 slots as soon as its four FC1 tiles are published;
//   6. the final unit (token, 512 columns): this rank's partial bf16(sum_k w_k rsqrt(S_k / 4096 + eps) rho_k), -0
//      cleared.
//
// Synchronization: every cross-CTA buffer is data-as-flag (kEmpty words), polled with all loads in flight and an
// __all_sync exit, no back-off; the workspace (h, rows, sums) has C copies, launch n uses copy n % C and clears copy
// (n + 1) % C; the partial slots are emptied by their single reader. Within the CTA the pipelines are mbarriers, each
// waited at most one phase ahead.

namespace p4a
{

// ---- PTX ------------------------------------------------------------------------------------------------------------
constexpr uint32_t kIdescNvf4 = (1u << 7) | (1u << 10) | ((8u >> 3) << 17) | ((128u >> 4) << 24); // M 128, N 8
// K-major SWIZZLE_128B operand descriptor without its start address: 8-row x 128 B atoms 1024 B apart, version 1.
constexpr uint64_t kDescSw128 = (1ull << 16) | (static_cast<uint64_t>(1024 >> 4) << 32) | (1ull << 46) | (2ull << 61);
// tcgen05.cp 32x128b source descriptor without its start address: 8-row x 16 B core matrices 128 B apart.
constexpr uint64_t kDescSf = (static_cast<uint64_t>(128 >> 4) << 32) | (1ull << 46);
constexpr float kLog2e = 1.44269502162933349609375f;   // 0x3FB8AA3B
constexpr float kRcp6 = 0.16666667163372039794921875f; // 0x3E2AAAAB

__device__ __forceinline__ void tcFenceBefore()
{
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcFenceAfter()
{
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tmemAlloc(uint32_t dst, uint32_t cols)
{
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(dst), "r"(cols) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmemDealloc(uint32_t taddr, uint32_t cols)
{
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(cols) : "memory");
}

__device__ __forceinline__ bool electOne()
{
    uint32_t e;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "elect.sync _|p, 0xffffffff;\n"
        "selp.u32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(e));
    return e != 0;
}

// 512 B of block scales (shared memory, 128x4 layout) into 4 TMEM columns (one thread).
__device__ __forceinline__ void utccp(uint32_t taddr, uint32_t src)
{
    uint64_t const sdesc = kDescSf | ((src & 0x3ffffu) >> 4);
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(taddr), "l"(sdesc) : "memory");
}

// As utccp, all four 8-row groups from the same 128 bytes (stride 0): rows 8-31 repeat rows 0-7 (the N = 8 operand).
__device__ __forceinline__ void utccpRows8(uint32_t taddr, uint32_t src)
{
    uint64_t const sdesc = (1ull << 46) | ((src & 0x3ffffu) >> 4);
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(taddr), "l"(sdesc) : "memory");
}

// Low descriptor words (start address >> 4 plus the constant low bits); the high words are constants.
constexpr uint32_t kDescSw128Lo = static_cast<uint32_t>(kDescSw128);
constexpr uint32_t kDescSw128Hi = static_cast<uint32_t>(kDescSw128 >> 32);
constexpr uint32_t kDescSfHi = static_cast<uint32_t>(kDescSf >> 32);

__device__ __forceinline__ uint32_t descLo(uint32_t smemAddr)
{
    return (smemAddr & 0x3ffffu) >> 4;
}

// As utccp, from the low descriptor word of the source.
__device__ __forceinline__ void utccpLo(uint32_t taddr, uint32_t srcLo)
{
    asm volatile(
        "{\n.reg .b64 d;\nmov.b64 d, {%1, %2};\n"
        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], d;\n}" ::"r"(taddr),
        "r"(srcLo), "r"(kDescSfHi)
        : "memory");
}

// As mmaNvf4, from the low descriptor words of a and b (SW128, K-major).
__device__ __forceinline__ void mmaNvf4Lo(
    uint32_t d, uint32_t aLo, uint32_t bLo, uint32_t sfa, uint32_t sfb, uint32_t acc)
{
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        ".reg .b64 da, db;\n"
        "mov.b64 da, {%1, %3};\n"
        "mov.b64 db, {%2, %3};\n"
        "setp.ne.b32 p, %7, 0;\n"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], da, db, %4, [%5], [%6], p;\n"
        "}\n" ::"r"(d),
        "r"(aLo), "r"(bLo), "r"(kDescSw128Hi), "r"(kIdescNvf4), "r"(sfa), "r"(sfb), "r"(acc)
        : "memory");
}

// D (+)= A B for one K step of 64; a and b are shared addresses of the K step's first byte inside SW128 atoms.
__device__ __forceinline__ void mmaNvf4(uint32_t d, uint32_t a, uint32_t b, uint32_t sfa, uint32_t sfb, uint32_t acc)
{
    uint64_t const da = kDescSw128 | ((a & 0x3ffffu) >> 4);
    uint64_t const db = kDescSw128 | ((b & 0x3ffffu) >> 4);
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %6, 0;\n"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%4], [%5], p;\n"
        "}\n" ::"r"(d),
        "l"(da), "l"(db), "r"(kIdescNvf4), "r"(sfa), "r"(sfb), "r"(acc)
        : "memory");
}

__device__ __forceinline__ void commit(uint32_t bar)
{
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(bar) : "memory");
}

// 8 columns of the warp's TMEM lane quarter (lane = row), waited.
__device__ __forceinline__ void tmemLd8(uint32_t taddr, float (&v)[8])
{
    uint32_t r[8];
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
                 : "r"(taddr)
                 : "memory");
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        v[i] = __uint_as_float(r[i]);
    }
}

// One box of a 2-D tensor map (global tensor-map address) into shared memory on barrier bar, with an L2 policy.
__device__ __forceinline__ void tmaLoad2d(uint32_t dst, void const* tmap, int x, int y, uint32_t bar, uint64_t policy)
{
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0], [%1, {%2, %3}], [%4], %5;" ::"r"(dst),
        "l"(tmap), "r"(x), "r"(y), "r"(bar), "l"(policy)
        : "memory");
}

__device__ __forceinline__ void mbarArrive(uint32_t bar)
{
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}

__device__ __forceinline__ void namedSync(int id, int threads)
{
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

__device__ __forceinline__ void sts64(uint32_t addr, uint32_t a, uint32_t b)
{
    asm volatile("st.shared.v2.b32 [%0], {%1, %2};" ::"r"(addr), "r"(a), "r"(b) : "memory");
}

__device__ __forceinline__ void sts128(uint32_t addr, uint4 v)
{
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}

__device__ __forceinline__ void prefetchTmap(int64_t tmap)
{
    asm volatile("prefetch.tensormap [%0];" ::"l"(tmap) : "memory");
}

__device__ __forceinline__ uint4 lds128(uint32_t addr)
{
    uint4 v;
    asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "r"(addr)
                 : "memory");
    return v;
}

// FP32 arithmetic with its exact rounding (inline PTX: no contraction, no reassociation).
__device__ __forceinline__ float mulRn(float a, float b)
{
    float d;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

__device__ __forceinline__ float mulFtz(float a, float b)
{
    float d;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

__device__ __forceinline__ float addRn(float a, float b)
{
    float d;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

__device__ __forceinline__ float fmaRnZero(float a, float b)
{
    float d;
    asm("fma.rn.f32 %0, %1, %2, 0f00000000;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

__device__ __forceinline__ float divRn(float a, float b)
{
    float d;
    asm("div.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

__device__ __forceinline__ float ex2Ftz(float a)
{
    float d;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a));
    return d;
}

__device__ __forceinline__ float rcpFtz(float a)
{
    float d;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a));
    return d;
}

__device__ __forceinline__ float rsqrtFtz(float a)
{
    float d;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a));
    return d;
}

__device__ __forceinline__ float rsqrtApprox(float a)
{
    float d;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(d) : "f"(a));
    return d;
}

__device__ __forceinline__ float minF(float a, float b)
{
    float d;
    asm("min.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

// The built-in routing sigmoid: 1 / (1 + 2^(x * -log2 e)) with its four instructions.
__device__ __forceinline__ float routeSigmoid(float x)
{
    float s;
    asm("{\n.reg .f32 t, e, d, one;\n"
        "mul.rn.f32 t, %1, 0fBFB8AA3B;\nex2.approx.f32 e, t;\n"
        "add.rn.f32 d, e, 0f3F800000;\nmov.b32 one, 0f3F800000;\n"
        "div.full.f32 %0, one, d;\n}"
        : "=f"(s)
        : "f"(x));
    return s;
}

// The selection key's high word: the order-preserving u32 of +0-canonical(score + bias).
__device__ __forceinline__ uint32_t routeKeyHi(float score, float bias)
{
    uint32_t const rb0 = __float_as_uint(addRn(score, bias));
    uint32_t const rb = (rb0 & 0x7fffffffu) == 0 ? 0u : rb0;
    uint32_t const m = static_cast<int32_t>(rb) > -1 ? 0x80000000u : 0xffffffffu;
    return m ^ rb;
}

__device__ __forceinline__ uint32_t cvtE4m3(float x)
{
    uint32_t r;
    asm("{\n.reg .b16 h;\n.reg .f32 z;\nmov.f32 z, 0f00000000;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h, z, %1;\ncvt.u32.u16 %0, h;\n}"
        : "=r"(r)
        : "f"(x));
    return r & 0xffu;
}

__device__ __forceinline__ float e4m3ToF32(uint32_t byte)
{
    float f;
    asm("{\n.reg .b16 lo, h0, h1;\n.reg .b32 hx2;\n"
        "cvt.u16.u32 lo, %1;\ncvt.rn.f16x2.e4m3x2 hx2, lo;\nmov.b32 {h0, h1}, hx2;\ncvt.f32.f16 %0, h0;\n}"
        : "=f"(f)
        : "r"(byte));
    return f;
}

// Two FP32 -> one E2M1 byte (RN, satfinite), lo in bits 0..3.
__device__ __forceinline__ uint32_t cvtE2m1x2(float lo, float hi)
{
    uint32_t r;
    asm("{\n.reg .b8 b;\n.reg .b32 t;\ncvt.rn.satfinite.e2m1x2.f32 b, %1, %2;\n"
        "mov.b32 t, {b, b, b, b};\nand.b32 %0, t, 255;\n}"
        : "=r"(r)
        : "f"(hi), "f"(lo));
    return r;
}

__device__ __forceinline__ uint32_t bf16Bits(float x)
{
    uint32_t r;
    asm("{\n.reg .b16 h;\ncvt.rn.satfinite.bf16.f32 h, %1;\ncvt.u32.u16 %0, h;\n}" : "=r"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ uint32_t bf16x2Mul(uint32_t a, uint32_t b)
{
    uint32_t r;
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

// max(m, |x|) per bf16 half (exact; a NaN half yields the other operand, as fmaxf does).
__device__ __forceinline__ uint32_t bf16x2AbsMax(uint32_t m, uint32_t x)
{
    uint32_t r;
    asm("{\n.reg .b32 t;\nabs.bf16x2 t, %2;\nmax.bf16x2 %0, %1, t;\n}" : "=r"(r) : "r"(m), "r"(x));
    return r;
}

// (a.x * b.x, a.y * b.y), each rounded like mul.rn.f32.
__device__ __forceinline__ float2 mulRn2(float2 a, float2 b)
{
    float2 d;
    asm("{\n.reg .b64 x, y, z;\nmov.b64 x, {%2, %3};\nmov.b64 y, {%4, %5};\nmul.rn.f32x2 z, x, y;\n"
        "mov.b64 {%0, %1}, z;\n}"
        : "=f"(d.x), "=f"(d.y)
        : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y));
    return d;
}

// waitBar for the warps that only consume (epilogue, h loader): with kMoeFlagSpinSleep each failed try sleeps ns
// nanoseconds, so a waiting warp does not take issue slots from the MMA / producer warp of its SM sub-partition.
__device__ __forceinline__ void waitBarIdle(Ctx const& ctx, uint32_t bar, uint32_t parity, int code, uint32_t ns)
{
    if (!mbarTryWait(bar, parity))
    {
        uint64_t t0 = 0;
        while (!mbarTryWait(bar, parity))
        {
            if (ns != 0)
            {
                __nanosleep(ns);
            }
            spinCheck(ctx, t0, code, static_cast<int>(bar), static_cast<int>(parity));
        }
    }
}

// Stamped builds: a count into sub-stamp k, encoded as the CTA's entry stamp + 1000 count (so the timeline printer's
// "us from the first entry" shows count + the CTA's entry offset).
__device__ __forceinline__ void moeCount(Ctx const& ctx, int k, int count)
{
    if (MSO_MK_STAMPS_DETAIL(ctx) != nullptr)
    {
        MSO_MK_STAMPS(ctx)[kStampSub + k] = MSO_MK_STAMPS(ctx)[kStampEntry] + 1000ull * static_cast<uint64_t>(count);
    }
}

// Any thread: stamp k of the CTA's sub-stamps (stamped builds).
__device__ __forceinline__ void moeStamp(Ctx const& ctx, int k)
{
    if (MSO_MK_STAMPS_DETAIL(ctx) != nullptr)
    {
        MSO_MK_STAMPS(ctx)[kStampSub + k] = stampTime(); // the stamp clock (MSO_MK_SY_CLK)
    }
}

// ---- the stage's shared state
// ----------------------------------------------------------------------------------------
struct Sm
{
    uint32_t redS; // the reduction area's shared address
    uint32_t base; // the unit region's shared address
};

__device__ __forceinline__ uint32_t bar(Sm const& s, int i)
{
    return s.redS + kMoeRdBars + 8 * i;
}

// The reduction area, addressed from the dynamic shared-memory symbol in every function (a pointer passed through a
// struct is generic to the callee, and generic accesses cost an address-space check each).
template <typename T>
__device__ __forceinline__ T* rd(Sm const&, int off)
{
    extern __shared__ __align__(1024) uint8_t smemRaw[];
    return reinterpret_cast<T*>(smemRaw + kSmemRed + off);
}

// Descriptor word idx (its shared copy).
__device__ __forceinline__ int64_t dw(Sm const& s, int idx)
{
    int64_t const v = rd<int64_t const>(s, kMoeRdDesc)[idx];
    return idx == kMdFlags ? icMoeFlags(v) : v; // the build's fixed flag bits (MSO_MK_IC_MOE_FIX; identity by default)
}

template <typename T>
__device__ __forceinline__ T* dptr(Sm const& s, int idx)
{
#if (MSO_MK_RT & 16) && MSO_MK_LEAN
    if (idx == kMdDbg || idx == kMdRows)
    {
        return nullptr; // MSO_MK_RT bit 16: a lean build takes no debug block (the stage traps on a raw non-zero word)
    }
#endif
    return reinterpret_cast<T*>(dw(s, idx));
}

// ---- the ring / B-operand layout of the launch (kMoeFlagE3Ring10), the per-unit stamps ------------------------------
static_assert(kMoeSmEnd10 <= kSmemMain, "the 10-stage layout fits the unit region");
static_assert(kStampE3Words == kE3TrWords, "e3 stamp words");

// The 10-stage ring is for the spread quantization only (the local quantization keeps its rows in the late stages
// and the gain in the FC2 B area).
__device__ __forceinline__ bool e3Ring10(int64_t flags)
{
    return (flags & kMoeFlagE3Ring10) && !(flags & (kMoeFlagLocalQuant | kMoeFlagWarmTwice));
}

template <int kNst>
struct E3Lay
{
    static constexpr int kX = kNst == kMoeStagesMax ? kMoeSmXH10 : kMoeSmX; // FC1 B operand
    static constexpr int kH = kNst == kMoeStagesMax ? kMoeSmXH10 : kMoeSmH; // FC2 B operand
};

__device__ __forceinline__ uint32_t e3SmX(Sm const& s)
{
    return e3Ring10(dw(s, kMdFlags)) ? kMoeSmXH10 : kMoeSmX;
}

__device__ __forceinline__ uint32_t e3SmGain(Sm const& s)
{
    return e3Ring10(dw(s, kMdFlags)) ? kMoeSmGain10 : kMoeSmGain;
}

// Stamped builds: per-unit stamp word k (kE3Tr*) of the CTA.
__device__ __forceinline__ void e3Tr(Ctx const& ctx, int k)
{
    if (MSO_MK_STAMPS_DETAIL(ctx) != nullptr)
    {
        MSO_MK_STAMPS(ctx)[kStampE3 + k] = stampTime(); // the stamp clock (MSO_MK_SY_CLK)
    }
}

// ---- end ring layout and per-unit stamps ----

// ---- the ring's in-flight cap (kMoeW2CapMask; 0 or >= the ring depth = the ring depth) ----
__device__ __forceinline__ int w2Cap(int64_t flags, int nst)
{
    int const cap = static_cast<int>((flags & kMoeW2CapMask) >> kMoeW2CapShift);
    return cap == 0 || cap > nst ? nst : cap;
}

// ---- end in-flight cap ----

// ---- the cut (identical in every CTA)
// ----------------------------------------------------------------------------------
struct Cut
{
    int E;
    bool tileCut;
    int a[2], b[2]; // unit ranges of the FC1 and FC2 lists
    int U[2];       // units of each list
    // ---- kMoeFlagPxQueue (the stock tile queue): a[fc] = 0, b[0] = 16 x the CTA's FC1 tiles, b[1] = 16 x its FC2
    // tiles; unit j is k-block j % 16 of the CTA's tile j / 16, at queue position pxCta + (j / 16) pxG ----
    bool px;
    int pxG;     // working CTAs (stock balance 1)
    int pxEarly; // stock order 1: slots whose FC1 tiles come first, slot major
    int pxCta;
    // ---- end kMoeFlagPxQueue ----
};

__device__ __forceinline__ int listStart(Cut const& c, int fc, int cta)
{
    if (c.tileCut)
    {
        int const tiles = c.U[fc] / kMoeKb;
        return static_cast<int>((static_cast<int64_t>(tiles) * cta) / kCtas) * kMoeKb;
    }
    return static_cast<int>((static_cast<int64_t>(c.U[fc]) * cta) / kCtas);
}

// The CTA whose range holds unit u of list fc.
__device__ __forceinline__ int ctaOfUnit(Cut const& c, int fc, int u)
{
    if (c.tileCut)
    {
        int const tiles = c.U[fc] / kMoeKb;
        int const t = u / kMoeKb;
        return static_cast<int>((static_cast<int64_t>(kCtas) * (t + 1) - 1) / tiles);
    }
    return static_cast<int>((static_cast<int64_t>(kCtas) * (u + 1) - 1) / c.U[fc]);
}

// ---- stock queue: persistent_experts' _schedule / _early_slots (balance 1, order 1) for E slots on kCtas CTAs ----
__device__ __forceinline__ void pxSchedule(int E, int& G, int& early)
{
    int const total = E * (kMoeFc1Tiles + kMoeFc2Tiles);
    int const rounds = (total + kCtas - 1) / kCtas;
    G = rounds > 0 ? (total + rounds - 1) / rounds : kCtas;
    early = 0;
    int const fc1Total = E * kMoeFc1Tiles;
    if (fc1Total > 0)
    {
        int const rMixed = fc1Total / G;
        int const inMixed = G * (rMixed + 1) - fc1Total;
        if (inMixed < G && rMixed + 1 < rounds)
        {
            early = (inMixed + kMoeFc2Tiles - 1) / kMoeFc2Tiles;
            early = early > E ? E : early;
            int const cap = (G * rMixed) / kMoeFc1Tiles;
            early = early > cap ? cap : early;
        }
    }
}

// The CTA's tiles of the stock queue (positions cta, cta + G, ...; FC1 positions below 64 E).
__device__ __forceinline__ Cut makeCutPxBody(int E, int cta)
{
    Cut c;
    c.E = E;
    c.tileCut = true;
    c.U[0] = E * kMoeFc1Units;
    c.U[1] = E * kMoeFc2Units;
    c.px = true;
    c.pxCta = cta;
    pxSchedule(E, c.pxG, c.pxEarly);
    int const total = E * (kMoeFc1Tiles + kMoeFc2Tiles);
    int const fc1Total = E * kMoeFc1Tiles;
    int const nT = cta < c.pxG && cta < total ? (total - cta + c.pxG - 1) / c.pxG : 0;
    int nT1 = cta < c.pxG && cta < fc1Total ? (fc1Total - cta + c.pxG - 1) / c.pxG : 0;
    nT1 = nT1 < nT ? nT1 : nT;
    c.a[0] = 0;
    c.b[0] = kMoeKb * nT1;
    c.a[1] = 0;
    c.b[1] = kMoeKb * (nT - nT1);
    return c;
}

// Out of line for makeCut's run-time choice (the warm-twice pass); the stage's own cut is stageCut<true>'s inline copy.
__device__ __noinline__ Cut makeCutPx(int E, int cta)
{
    return makeCutPxBody(E, cta);
}

// ---- end stock queue ----

__device__ __forceinline__ void makeCut(Cut& c, int E, bool tileCut, int cta, bool px = false) // px: the stock queue
{
    // ---- the stock queue ----
    if (px)
    {
        c = makeCutPx(E, cta);
        return;
    }
    c.px = false;
    c.pxG = kCtas;
    c.pxEarly = 0;
    c.pxCta = cta;
    // ---- end stock queue ----
    c.E = E;
    c.tileCut = tileCut;
    c.U[0] = E * kMoeFc1Units;
    c.U[1] = E * kMoeFc2Units;
#pragma unroll
    for (int fc = 0; fc < 2; ++fc)
    {
        c.a[fc] = c.U[fc] == 0 ? 0 : listStart(c, fc, cta);
        c.b[fc] = c.U[fc] == 0 ? 0 : listStart(c, fc, cta + 1);
    }
}

__device__ __forceinline__ int unitsOf(Cut const& c)
{
    return (c.b[0] - c.a[0]) + (c.b[1] - c.a[1]);
}

// Unit j (0 <= j < b - a) of the CTA's part of list fc. The part's tiles go from the last to the first, k-blocks in
// order within a tile: the last tile, the only one that can continue on the next CTA, publishes its partial before
// this CTA waits for any other CTA's partial (only its first tile can have started on an earlier CTA), so no wait
// chains through the CTAs. The FC2 slots also come in one (descending) run each.
__device__ __forceinline__ int listUnit(Cut const& c, int fc, int j)
{
    int const a = c.a[fc];
    int const lastLo = ((c.b[fc] - 1) / kMoeKb) * kMoeKb;
    int const lastFrom = lastLo > a ? lastLo : a;
    int const nLast = c.b[fc] - lastFrom;
    if (j < nLast)
    {
        return lastFrom + j;
    }
    int const jj = j - nLast;
    int const lo = lastLo - kMoeKb * (1 + jj / kMoeKb);
    return (lo > a ? lo : a) + jj % kMoeKb;
}

// ---- the stock queue: tile t of the CTA (queue position pxCta + t pxG): (fc, the tile's first list unit),
// persistent_experts' _decode with order 1 ----
__device__ __forceinline__ void pxTile(Cut const& c, int t, int& fc, int& u0)
{
    int const p = c.pxCta + t * c.pxG;
    int const fc1Total = c.E * kMoeFc1Tiles;
    if (p < fc1Total)
    {
        int slot = p / kMoeFc1Tiles;
        int mt = p - slot * kMoeFc1Tiles;
        if (p >= c.pxEarly * kMoeFc1Tiles)
        {
            int const q = p - c.pxEarly * kMoeFc1Tiles;
            int const rest = c.E - c.pxEarly;
            mt = q / rest;
            slot = c.pxEarly + (q - mt * rest);
        }
        fc = 0;
        u0 = slot * kMoeFc1Units + mt * kMoeKb;
    }
    else
    {
        int const q = p - fc1Total;
        fc = 1;
        u0 = (q / kMoeFc2Tiles) * kMoeFc2Units + (q % kMoeFc2Tiles) * kMoeKb;
    }
}

// ---- end stock queue ----

// Unit j of the CTA's sequence (its FC1 part, then its FC2 part): (fc, list unit). kPx: the cut is the stock queue
// (c.px), a compile-time choice so that the byte / tile cut loops carry no stock-queue code.
template <bool kPx = false>
__device__ __forceinline__ void unitAt(Cut const& c, int j, int& fc, int& u)
{
    if constexpr (kPx) // the stock queue: k-block j % 16 of the CTA's tile j / 16
    {
        int u0;
        pxTile(c, j >> 4, fc, u0);
        u = u0 + (j & (kMoeKb - 1));
    }
    else
    {
        int const n1 = c.b[0] - c.a[0];
        fc = j < n1 ? 0 : 1;
        u = listUnit(c, fc, j < n1 ? j : j - n1);
    }
}

__device__ __forceinline__ void decodeUnit(int fc, int u, int& slot, int& mt, int& kb)
{
    kb = u & (kMoeKb - 1);
    mt = fc == 0 ? (u >> 4) & (kMoeFc1Tiles - 1) : (u >> 4) & (kMoeFc2Tiles - 1);
    slot = fc == 0 ? u / kMoeFc1Units : u / kMoeFc2Units;
}

// ---- MSO_MK_RT bit 1: 8 distinct keys sorted descending in registers (Batcher's odd-even merge network,
// 19 comparators) ----
#if MSO_MK_RT & 1
__device__ __forceinline__ void rtCas(uint64_t& a, uint64_t& b)
{
    bool const sw = b > a;
    uint64_t const hi = sw ? b : a;
    uint64_t const lo = sw ? a : b;
    a = hi;
    b = lo;
}

__device__ __forceinline__ void rtSort8(uint64_t (&k)[8])
{
    rtCas(k[0], k[1]), rtCas(k[2], k[3]), rtCas(k[4], k[5]), rtCas(k[6], k[7]);
    rtCas(k[0], k[2]), rtCas(k[1], k[3]), rtCas(k[4], k[6]), rtCas(k[5], k[7]);
    rtCas(k[1], k[2]), rtCas(k[5], k[6]);
    rtCas(k[0], k[4]), rtCas(k[1], k[5]), rtCas(k[2], k[6]), rtCas(k[3], k[7]);
    rtCas(k[2], k[4]), rtCas(k[3], k[5]);
    rtCas(k[1], k[2]), rtCas(k[3], k[4]), rtCas(k[5], k[6]);
}
#endif
// ---- MSO_MK_RT bit 8: the X gather (gatherX's loads, polls and stores, same addresses and values) split around
// route()'s picks: with QuantFirst every CTA has published its groups before route(), so the loads go out with the
// logits and land under the picks ----
#if MSO_MK_RT & 8
constexpr int kRtXq = kMoeWsXqBytes / 16;  // 1024 code vectors
constexpr int kRtXs = kMoeWsXsfBytes / 16; // 128 scale vectors
constexpr int kRtXPer = (kRtXq + kRtXs + kThreads - 1) / kThreads;

__device__ __forceinline__ uint8_t const* rtXAddr(uint8_t const* ws, int k)
{
    int const vi = threadIdx.x + kThreads * k;
    return vi < kRtXq ? ws + kMoeWsXq + 16 * vi : vi < kRtXq + kRtXs ? ws + kMoeWsXsf + 16 * (vi - kRtXq) : nullptr;
}

__device__ __forceinline__ void rtXIssue(uint8_t const* ws, uint4 (&v)[kRtXPer])
{
#pragma unroll
    for (int k = 0; k < kRtXPer; ++k)
    {
        uint8_t const* const a = rtXAddr(ws, k);
        v[k] = a != nullptr ? ldPoll128(a) : make_uint4(0, 0, 0, 0);
    }
}

__device__ __forceinline__ void rtXFinish(Ctx const& ctx, Sm const& s, uint8_t const* ws, uint4 (&v)[kRtXPer])
{
    uint64_t t0 = 0;
    while (true)
    {
        bool ok = true;
#pragma unroll
        for (int k = 0; k < kRtXPer; ++k)
        {
            ok &= full4(v[k]);
        }
        if (warpAll(ok))
        {
            break;
        }
        spinCheck(ctx, t0, kCodeMoePool, kRtXPer, 0);
#pragma unroll
        for (int k = 0; k < kRtXPer; ++k)
        {
            if (!full4(v[k]))
            {
                v[k] = ldPoll128(rtXAddr(ws, k));
            }
        }
    }
    if (threadIdx.x == 0)
    {
        moeStamp(ctx, kMoeStPool);
    }
    uint8_t* dbg = dptr<uint8_t>(s, kMdDbg);
    bool const dbgOn = dbg != nullptr && blockIdx.x == 0;
    uint32_t const xOff = e3SmX(s);
#pragma unroll
    for (int k = 0; k < kRtXPer; ++k)
    {
        int const vi = threadIdx.x + kThreads * k;
        if (vi < kRtXq)
        {
            int const r = vi >> 7;
            int const i = (vi & 127) * 2;
            int const kb = i >> 4;
            int const c16 = (i & 15) >> 1;
            sts128(s.base + xOff + kb * 1024 + r * 128 + ((c16 ^ r) << 4), v[k]);
            if (dbgOn)
            {
                *reinterpret_cast<uint4*>(dbg + kMoeDbgXq + 16 * vi) = v[k];
            }
        }
        else if (vi < kRtXq + kRtXs)
        {
            int const vs = vi - kRtXq;
            int const r = vs >> 4;
            int const p0 = 2 * (vs & 15);
            sts128(s.redS + kMoeRdStgX + p0 * 128 + r * 16, make_uint4(v[k].x, 0u, v[k].y, 0u));
            sts128(s.redS + kMoeRdStgX + (p0 + 1) * 128 + r * 16, make_uint4(v[k].z, 0u, v[k].w, 0u));
            if (dbgOn)
            {
                *reinterpret_cast<uint4*>(dbg + kMoeDbgXsf + 16 * vs) = v[k];
            }
        }
    }
    if (threadIdx.x == 0 && MSO_MK_STAMPS(ctx) != nullptr)
    {
        MSO_MK_STAMPS(ctx)[kStampI2aXStaged] = globalTimer();
    }
}
#if MSO_MK_RT & 64
// MSO_MK_RT bit 64: vector k's shared-memory slot: a code vector's SW128 place in the X tile (the one rtXFinish
// stores to), a scale vector's place in the temp (the FC2 scale staging, unused before the roles).
__device__ __forceinline__ uint32_t rtXDst(Sm const& s, int k)
{
    int const vi = threadIdx.x + kThreads * k;
    if (vi < kRtXq)
    {
        int const r = vi >> 7;
        int const i = (vi & 127) * 2;
        int const kb = i >> 4;
        int const c16 = (i & 15) >> 1;
        return s.base + e3SmX(s) + kb * 1024 + r * 128 + ((c16 ^ r) << 4);
    }
    return s.redS + kMoeRdStgH + 16 * (vi - kRtXq);
}

__device__ __forceinline__ bool rtXMine(int k)
{
    return threadIdx.x + kThreads * k < kRtXq + kRtXs;
}

__device__ __forceinline__ void rtXCopy(Sm const& s, uint8_t const* ws, int k)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(rtXDst(s, k)), "l"(rtXAddr(ws, k)) : "memory");
}

__device__ __forceinline__ void rtXIssueA(Sm const& s, uint8_t const* ws)
{
#pragma unroll
    for (int k = 0; k < kRtXPer; ++k)
    {
        if (rtXMine(k))
        {
            rtXCopy(s, ws, k);
        }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

// The copies landed; a vector that still holds the sentinel is copied again (the poll), per warp as in rtXFinish; then
// the scale vectors go from the temp to their tcgen05.cp staging layout.
__device__ __forceinline__ void rtXFinishA(Ctx const& ctx, Sm const& s, uint8_t const* ws)
{
    uint64_t t0 = 0;
    while (true)
    {
        asm volatile("cp.async.wait_all;" ::: "memory");
        uint32_t redo = 0;
#pragma unroll
        for (int k = 0; k < kRtXPer; ++k)
        {
            if (rtXMine(k) && !full4(lds128(rtXDst(s, k))))
            {
                redo |= 1u << k;
            }
        }
        if (warpAll(redo == 0))
        {
            break;
        }
        spinCheck(ctx, t0, kCodeMoePool, kRtXPer, 0);
#pragma unroll
        for (int k = 0; k < kRtXPer; ++k)
        {
            if ((redo >> k) & 1u)
            {
                rtXCopy(s, ws, k);
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    }
    if (threadIdx.x == 0)
    {
        moeStamp(ctx, kMoeStPool);
    }
    uint8_t* dbg = dptr<uint8_t>(s, kMdDbg);
    bool const dbgOn = dbg != nullptr && blockIdx.x == 0;
#pragma unroll
    for (int k = 0; k < kRtXPer; ++k)
    {
        int const vi = threadIdx.x + kThreads * k;
        if (vi < kRtXq)
        {
            if (dbgOn)
            {
                *reinterpret_cast<uint4*>(dbg + kMoeDbgXq + 16 * vi) = lds128(rtXDst(s, k));
            }
        }
        else if (vi < kRtXq + kRtXs)
        {
            uint4 const v = lds128(rtXDst(s, k));
            int const vs = vi - kRtXq;
            int const r = vs >> 4;
            int const p0 = 2 * (vs & 15);
            sts128(s.redS + kMoeRdStgX + p0 * 128 + r * 16, make_uint4(v.x, 0u, v.y, 0u));
            sts128(s.redS + kMoeRdStgX + (p0 + 1) * 128 + r * 16, make_uint4(v.z, 0u, v.w, 0u));
            if (dbgOn)
            {
                *reinterpret_cast<uint4*>(dbg + kMoeDbgXsf + 16 * vs) = v;
            }
        }
    }
    if (threadIdx.x == 0 && MSO_MK_STAMPS(ctx) != nullptr)
    {
        MSO_MK_STAMPS(ctx)[kStampI2aXStaged] = globalTimer();
    }
}
#endif
#endif
// ---- end MSO_MK_RT ----

// ---- 1. routing: warp t = token t
// --------------------------------------------------------------------------------------
#if MSO_MK_RT & 8
// MSO_MK_RT bit 8: xWs != nullptr = the X gather from this workspace copy around the picks.
__device__ __noinline__ void route(Ctx const& ctx, Sm const s, bool pre = false, uint8_t const* xWs = nullptr)
#else
__device__ __noinline__ void route(Ctx const& ctx, Sm const s, bool pre = false) // pre: kMoeFlagW2LogitsPre
#endif
{
    int const t = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
    float const* logits = dptr<float const>(s, kMdLogits) + t * kMoeExperts + 8 * lane;
    float const* biasP = dptr<float const>(s, kMdBias) + 8 * lane;
    float4 const b0 = __ldg(reinterpret_cast<float4 const*>(biasP));
    float4 const b1 = __ldg(reinterpret_cast<float4 const*>(biasP + 4));
    uint32_t raw[8];
    // ---- kMoeFlagMhF2: the logits [256 experts][8 rows] of the F2' area ----
#if MSO_MK_LANES & 8
    // ---- kMoeFlagLogitsCoop: CTA-wide coalesced poll of the logit area, transposed into shared memory ----
    if ((dw(s, kMdFlags) & kMoeFlagMhF2) && (dw(s, kMdFlags) & kMoeFlagLogitsCoop))
    {
        // Thread i polls 16-byte vectors i and i + 256 of the area (vector k: expert k / 2, rows 4 (k % 2) .. + 4), so
        // a warp's loads cover 512 contiguous bytes; the CTA's 512 vectors are stored transposed as FP32 [8][256] in
        // the late ring stages (free until gatherX requests them, after the barrier that follows route()); warp t
        // then reads its token's experts 8 lane .. + 8 as two vectors. Every CTA thread calls route().
        uint8_t const* const base = mhUc(ctx, ctx.copy) + kMhLogitOff;
        // ---- kIFlagT3CanaryLogits: the gate on 32 logit vectors (lane l: source rank l / 4, expert row
        // 8 (l % 4) + 7, row half l % 2) ----
        // (with kMoeFlagW2LogitsPre the gate is skipped: it is only a heuristic in front of the full poll)
        if ((icPIFlags(ctx.p[kPIFlags]) & kIFlagT3CanaryLogits) && !(dw(s, kMdFlags) & kMoeFlagW2LogitsPre))
        {
            int const k = 2 * (kMhLocalExperts * (lane >> 2) + 8 * (lane & 3) + 7) + (lane & 1);
            t3Canary(ctx, base + 16 * k + 12, kCodeMoeLogits);
        }
        // ---- end canary gate ----
        uint8_t const* const a[2] = {base + 16 * threadIdx.x, base + 16 * (threadIdx.x + kThreads)};
        uint4 v[2];
        uint32_t const stg = s.base + kMoeSmQStage;
        // ---- pre: a warp whose prefetched vectors are staged already (pre is warp-uniform) skips its poll ----
        if (!pre)
        {
            pollVecs<2>(ctx, v, a, kCodeMoeLogits);
        }
#pragma unroll
        for (int k = 0; k < 2 && !pre; ++k) // pre: staged already
        {
            int const vi = threadIdx.x + kThreads * k;
            int const e = vi >> 1;
            int const r0 = 4 * (vi & 1);
            uint32_t const w4[4] = {v[k].x, v[k].y, v[k].z, v[k].w};
#pragma unroll
            for (int r = 0; r < 4; ++r)
            {
                asm volatile(
                    "st.shared.u32 [%0], %1;" ::"r"(stg + 4u * static_cast<uint32_t>((r0 + r) * kMoeExperts + e)),
                    "r"(w4[r])
                    : "memory");
            }
        }
        __syncthreads();
        uint4 const x0 = lds128(stg + 4u * static_cast<uint32_t>(t * kMoeExperts + 8 * lane));
        uint4 const x1 = lds128(stg + 4u * static_cast<uint32_t>(t * kMoeExperts + 8 * lane + 4));
        raw[0] = x0.x, raw[1] = x0.y, raw[2] = x0.z, raw[3] = x0.w;
        raw[4] = x1.x, raw[5] = x1.y, raw[6] = x1.z, raw[7] = x1.w;
    }
    else
        // ---- end kMoeFlagLogitsCoop ----
        if (dw(s, kMdFlags) & kMoeFlagMhF2)
        {
            uint8_t const* const base = mhUc(ctx, ctx.copy) + kMhLogitOff;
            uint8_t const* a[8];
#pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                a[j] = base + (static_cast<size_t>(8 * lane + j) * 8 + t) * 4;
            }
            i1PollWords<8>(ctx, raw, a, kCodeMoeLogits);
        }
        else
#endif
        // ---- end kMoeFlagMhF2 ----
        {
            uint4 v[2];
            uint8_t const* const a[2]
                = {reinterpret_cast<uint8_t const*>(logits), reinterpret_cast<uint8_t const*>(logits + 4)};
            pollVecs<2>(ctx, v, a, kCodeMoeLogits);
            uint32_t const r[8] = {v[0].x, v[0].y, v[0].z, v[0].w, v[1].x, v[1].y, v[1].z, v[1].w};
#pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                raw[j] = r[j];
            }
        }
#if (MSO_MK_RT & 8) && (MSO_MK_RT & 64)
    if (xWs != nullptr)
    {
        rtXIssueA(s, xWs); // MSO_MK_RT bit 64: straight to shared memory
    }
#elif MSO_MK_RT & 8
    uint4 xv[kRtXPer];
    if (xWs != nullptr)
    {
        rtXIssue(xWs, xv);
    }
#endif
    float const bias[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
    float score[8];
    uint32_t hi[8];
#pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        score[j] = routeSigmoid(__uint_as_float(raw[j]));
        hi[j] = routeKeyHi(score[j], bias[j]);
    }
    if (threadIdx.x == 0)
    {
        moeStamp(ctx, kMoeStLogits);
    }
    // The lane's keys: hi, then 255 - e (the lower expert id wins a tie); every live key is > 0, a taken one is 0.
    uint64_t key[8];
#pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        key[j] = (static_cast<uint64_t>(hi[j]) << 32) | (255u - static_cast<uint32_t>(8 * lane + j));
    }
#if MSO_MK_RT & 1
    int* selS = rd<int>(s, kMoeRdSel) + t * kMoeTopK;
    float* wS = rd<float>(s, kMoeRdW) + t * kMoeTopK;
    // ---- MSO_MK_RT bit 1: the lane's keys sorted once (descending; the keys are distinct), then per pick one
    // redux of the heads' high words and a ballot: the pick is the head of the lowest lane whose head holds the
    // maximum high word (on a tie of high words the lower expert id wins, and lane l holds experts 8 l .. 8 l + 7;
    // within a lane its head is its largest key). The winner records the pick and shifts its list. ----
    rtSort8(key);
    uint32_t const ltMask = (1u << lane) - 1u;
    uint32_t taken = 0; // bit j: expert 8 lane + j is selected
#pragma unroll 1
    for (int k = 0; k < kMoeTopK; ++k)
    {
        uint32_t const h = static_cast<uint32_t>(key[0] >> 32);
        uint32_t const mh = __reduce_max_sync(0xffffffffu, h);
        bool const mine = h == mh;
        uint32_t const b = __ballot_sync(0xffffffffu, mine);
        bool const win = mine && (b & ltMask) == 0;
        if (win)
        {
            int const e = 255 - static_cast<int>(static_cast<uint32_t>(key[0]));
            int const je = e & 7;
            float const s01 = (je & 1) ? score[1] : score[0];
            float const s23 = (je & 1) ? score[3] : score[2];
            float const s45 = (je & 1) ? score[5] : score[4];
            float const s67 = (je & 1) ? score[7] : score[6];
            float const s03 = (je & 2) ? s23 : s01;
            float const s47 = (je & 2) ? s67 : s45;
            selS[k] = e;
            wS[k] = (je & 4) ? s47 : s03;
            taken |= 1u << je;
        }
#pragma unroll
        for (int j = 0; j < 7; ++j)
        {
            key[j] = win ? key[j + 1] : key[j];
        }
        key[7] = win ? 0ull : key[7];
    }
#else
    uint32_t alive = 0xffu;
    int* selS = rd<int>(s, kMoeRdSel) + t * kMoeTopK;
    float* wS = rd<float>(s, kMoeRdW) + t * kMoeTopK;
    // One round per pick, rolled (run-once code is fetched cold); the picks go to shared memory. The lane's best
    // remaining key is a branch-free 3-level max tree (the keys are distinct within a lane).
#pragma unroll 1
    for (int k = 0; k < kMoeTopK; ++k)
    {
        uint64_t m[8];
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            m[j] = ((alive >> j) & 1u) ? key[j] : 0ull;
        }
#pragma unroll
        for (int w = 1; w < 8; w <<= 1)
        {
#pragma unroll
            for (int j = 0; j < 8; j += 2 * w)
            {
                m[j] = m[j] > m[j + w] ? m[j] : m[j + w];
            }
        }
        uint32_t const bh = static_cast<uint32_t>(m[0] >> 32);
        uint32_t const bl = static_cast<uint32_t>(m[0]);
        uint32_t const mh = __reduce_max_sync(0xffffffffu, bh);
        uint32_t const ml = __reduce_max_sync(0xffffffffu, bh == mh ? bl : 0u);
        int const e = 255 - static_cast<int>(ml);
        int const owner = e >> 3;
        int const je = e & 7;
        // score[je] by its index bits (a select tree, no local array).
        float const s01 = (je & 1) ? score[1] : score[0];
        float const s23 = (je & 1) ? score[3] : score[2];
        float const s45 = (je & 1) ? score[5] : score[4];
        float const s67 = (je & 1) ? score[7] : score[6];
        float const s03 = (je & 2) ? s23 : s01;
        float const s47 = (je & 2) ? s67 : s45;
        float const bs = (je & 4) ? s47 : s03;
        if (lane == owner)
        {
            alive &= ~(1u << je);
        }
        float const pk = __shfl_sync(0xffffffffu, bs, owner);
        if (lane == 0)
        {
            selS[k] = e;
            wS[k] = pk;
        }
    }
#if MSO_MK_RT & 2
    uint32_t const taken = ~alive & 0xffu;
#endif
#endif // MSO_MK_RT bit 1
#if MSO_MK_RT & 2
    // ---- MSO_MK_RT bit 2: the token's presence words (lanes 4 b .. 4 b + 3 hold block b's experts, 8 per lane) ----
    {
        uint32_t v = taken << (8 * (lane & 3));
        v |= __shfl_xor_sync(0xffffffffu, v, 1);
        v |= __shfl_xor_sync(0xffffffffu, v, 2);
        if ((lane & 3) == 0)
        {
            rd<uint32_t>(s, kMoeRdRtPres)[(lane >> 2) * kTok + t] = v;
        }
    }
#endif
    __syncwarp();
    float p[kMoeTopK];
#pragma unroll
    for (int k = 0; k < kMoeTopK; ++k)
    {
        p[k] = wS[k];
    }
    __syncwarp();
    float const sum = addRn(addRn(addRn(p[0], p[1]), addRn(p[4], p[5])), addRn(addRn(p[2], p[3]), addRn(p[6], p[7])));
    float const den = addRn(sum, __uint_as_float(static_cast<uint32_t>(dw(s, kMdRenormEps))));
    if (lane < kMoeTopK)
    {
        int const e = selS[lane];
        float const w = divRn(wS[lane], den);
        wS[lane] = w;
        uint8_t* dbg = dptr<uint8_t>(s, kMdDbg);
        if (dbg != nullptr && blockIdx.x == 0)
        {
            reinterpret_cast<int*>(dbg + kMoeDbgSel)[t * kMoeTopK + lane] = e;
            reinterpret_cast<float*>(dbg + kMoeDbgW)[t * kMoeTopK + lane] = w;
        }
    }
#if (MSO_MK_RT & 8) && (MSO_MK_RT & 64)
    if (xWs != nullptr)
    {
        rtXFinishA(ctx, s, xWs);
    }
#elif MSO_MK_RT & 8
    if (xWs != nullptr)
    {
        rtXFinish(ctx, s, xWs, xv);
    }
#endif
}

// ---- 2. placement (windows.place, three windows) and the slot list
// ------------------------------------------------------- Called by every thread after route() and a barrier; leaves E
// in the scalars, the slot tables and tokSlot.
__device__ __noinline__ void place(Ctx const& ctx, Sm const s)
{
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const rank = static_cast<int>(dw(s, kMdRank));
    uint32_t* pres = rd<uint32_t>(s, kMoeRdPres); // [0, 8) block presence, [8, 11) local presence
    int* m12 = rd<int>(s, kMoeRdMoved);
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
        // The smallest feasible maximum: ceil(D / 8) and every arc of n blocks over its min(n + 2, 8) ranks.
        int tr = 0;
        int arc = 0;
#pragma unroll
        for (int n = 1; n < kMoeRanks; ++n)
        {
            arc += __shfl_sync(0xffffffffu, load, (lane - n + 1) & 7);
            int const reach = n + 2 < kMoeRanks ? n + 2 : kMoeRanks;
            tr = max(tr, (arc + reach - 1) / reach);
        }
        int const T = max((D + kMoeRanks - 1) / kMoeRanks,
            static_cast<int>(__reduce_max_sync(0xffffffffu, static_cast<uint32_t>(lane < kMoeRanks ? tr : 0))));
        // c[r]: experts crossing the edge r -> r + 1 in the smallest flow of maximum T.
        int c = 0;
        arc = 0;
#pragma unroll
        for (int n = 1; n <= kMoeRanks; ++n)
        {
            arc += __shfl_sync(0xffffffffu, load, (lane - n + 1) & 7);
            c = max(c, arc - n * T);
        }
        int const lv0 = min(c, load);
        int const cNext = __shfl_sync(0xffffffffu, c, (lane + 1) & 7);
        int const lv0Next = __shfl_sync(0xffffffffu, lv0, (lane + 1) & 7);
        int const lv1 = min(max(0, cNext - lv0Next), lv0);
        int const moved1 = lv0 - lv1;
        int const moved2 = lv1;
        int const prev1 = __shfl_sync(0xffffffffu, moved1, (lane - 1) & 7);
        int const prev2 = __shfl_sync(0xffffffffu, moved2, (lane - 2) & 7);
        if (lane < kMoeRanks)
        {
            m12[lane] = moved1;
            m12[kMoeRanks + lane] = moved2;
            uint8_t* dbg = dptr<uint8_t>(s, kMdDbg);
            if (dbg != nullptr && blockIdx.x == 0)
            {
                reinterpret_cast<int*>(dbg + kMoeDbgLoads)[lane] = load;
                reinterpret_cast<int*>(dbg + kMoeDbgLoads)[kMoeRanks + lane] = load - moved1 - moved2 + prev1 + prev2;
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
        int const pos = __popc(static_cast<uint32_t>(static_cast<uint64_t>(pres[b]) >> (i + 1)));
        int const mv2 = m12[kMoeRanks + b];
        int const mv1 = m12[b];
        int const runsOn = pos < mv2 ? ((b + 2) & 7) : pos < mv2 + mv1 ? ((b + 1) & 7) : b;
        int const id = runsOn == rank ? ((rank - b) & 7) * kMoeBlock + i : kMoeWindow + e % (kMoeExperts - kMoeWindow);
        local[tid] = id;
        if (id < kMoeWindow)
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
    uint32_t const lp0 = pres[8], lp1 = pres[9], lp2 = pres[10];
    auto prefix = [&](int id)
    {
        int const w = id >> 5;
        uint32_t const below = (1u << (id & 31)) - 1u;
        return (w > 0 ? __popc(lp0) : 0) + (w > 1 ? __popc(lp1) : 0)
            + __popc((w == 0          ? lp0
                             : w == 1 ? lp1
                                      : lp2)
                & below);
    };
    int const E = __popc(lp0) + __popc(lp1) + __popc(lp2);
    int* slotExpert = rd<int>(s, kMoeRdSlotExpert);
    uint32_t* slotMask = rd<uint32_t>(s, kMoeRdSlotMask);
    int8_t* slotK = rd<int8_t>(s, kMoeRdSlotK);
    int8_t* tokSlot = rd<int8_t>(s, kMoeRdTokSlot);
    if (tid < kMoeWindow)
    {
        uint32_t const w = tid < 32 ? lp0 : tid < 64 ? lp1 : lp2;
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
        if (id < kMoeWindow)
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

// ---- kMoeFlagW2PlaceWarp: place() in warp 0 alone, warp-synchronous (no CTA barrier, no shared-memory atomics on
// the presence words). Lane l holds the selections l and l + 32 (token l / 8 resp. 4 + l / 8, k = l % 8); the presence
// words are warp OR-reductions held in every lane, the flow is place()'s code on the same values, so E, the slot list,
// tokSlot, slotMask and slotK are the same. The caller runs one __syncthreads after it. ----
__device__ __forceinline__ uint32_t w2Pick8(uint32_t const (&w)[8], int b)
{
    uint32_t const w01 = (b & 1) ? w[1] : w[0];
    uint32_t const w23 = (b & 1) ? w[3] : w[2];
    uint32_t const w45 = (b & 1) ? w[5] : w[4];
    uint32_t const w67 = (b & 1) ? w[7] : w[6];
    uint32_t const w03 = (b & 2) ? w23 : w01;
    uint32_t const w47 = (b & 2) ? w67 : w45;
    return (b & 4) ? w47 : w03;
}

__device__ __noinline__ void placeWarp(Ctx const& ctx, Sm const s)
{
    int const lane = threadIdx.x & 31;
    int const rank = static_cast<int>(dw(s, kMdRank));
    int const* sel = rd<int const>(s, kMoeRdSel);
    int const ev[2] = {sel[lane], sel[lane + 32]};
    uint32_t w[8];
#pragma unroll
    for (int b = 0; b < kMoeRanks; ++b)
    {
        uint32_t const m
            = ((ev[0] >> 5) == b ? 1u << (ev[0] & 31) : 0u) | ((ev[1] >> 5) == b ? 1u << (ev[1] & 31) : 0u);
        w[b] = __reduce_or_sync(0xffffffffu, m);
    }
    // place()'s flow (lanes 0..7 = blocks), on the same loads.
    int const load = lane < kMoeRanks ? __popc(w2Pick8(w, lane)) : 0;
    int const D = static_cast<int>(__reduce_add_sync(0xffffffffu, static_cast<uint32_t>(load)));
    int tr = 0;
    int arc = 0;
#pragma unroll
    for (int n = 1; n < kMoeRanks; ++n)
    {
        arc += __shfl_sync(0xffffffffu, load, (lane - n + 1) & 7);
        int const reach = n + 2 < kMoeRanks ? n + 2 : kMoeRanks;
        tr = max(tr, (arc + reach - 1) / reach);
    }
    int const T = max((D + kMoeRanks - 1) / kMoeRanks,
        static_cast<int>(__reduce_max_sync(0xffffffffu, static_cast<uint32_t>(lane < kMoeRanks ? tr : 0))));
    int cf = 0;
    arc = 0;
#pragma unroll
    for (int n = 1; n <= kMoeRanks; ++n)
    {
        arc += __shfl_sync(0xffffffffu, load, (lane - n + 1) & 7);
        cf = max(cf, arc - n * T);
    }
    int const lv0 = min(cf, load);
    int const cNext = __shfl_sync(0xffffffffu, cf, (lane + 1) & 7);
    int const lv0Next = __shfl_sync(0xffffffffu, lv0, (lane + 1) & 7);
    int const lv1 = min(max(0, cNext - lv0Next), lv0);
    int const moved1 = lv0 - lv1;
    int const moved2 = lv1;
    uint8_t* dbg = dptr<uint8_t>(s, kMdDbg);
    bool const dbgOn = dbg != nullptr && blockIdx.x == 0;
    {
        int const prev1 = __shfl_sync(0xffffffffu, moved1, (lane - 1) & 7);
        int const prev2 = __shfl_sync(0xffffffffu, moved2, (lane - 2) & 7);
        if (dbgOn && lane < kMoeRanks)
        {
            reinterpret_cast<int*>(dbg + kMoeDbgLoads)[lane] = load;
            reinterpret_cast<int*>(dbg + kMoeDbgLoads)[kMoeRanks + lane] = load - moved1 - moved2 + prev1 + prev2;
        }
    }
    // The window ids of the lane's two selections, and the local presence words.
    int id[2];
#pragma unroll
    for (int h = 0; h < 2; ++h)
    {
        int const e = ev[h];
        int const b = e >> 5;
        int const i = e & 31;
        int const pos = __popc(static_cast<uint32_t>(static_cast<uint64_t>(w2Pick8(w, b)) >> (i + 1)));
        int const mv1 = __shfl_sync(0xffffffffu, moved1, b);
        int const mv2 = __shfl_sync(0xffffffffu, moved2, b);
        int const runsOn = pos < mv2 ? ((b + 2) & 7) : pos < mv2 + mv1 ? ((b + 1) & 7) : b;
        id[h] = runsOn == rank ? ((rank - b) & 7) * kMoeBlock + i : kMoeWindow + e % (kMoeExperts - kMoeWindow);
        if (dbgOn)
        {
            reinterpret_cast<int*>(dbg + kMoeDbgLocal)[lane + 32 * h] = id[h];
        }
    }
    uint32_t lp[3];
#pragma unroll
    for (int j = 0; j < 3; ++j)
    {
        uint32_t const m = (id[0] < kMoeWindow && (id[0] >> 5) == j ? 1u << (id[0] & 31) : 0u)
            | (id[1] < kMoeWindow && (id[1] >> 5) == j ? 1u << (id[1] & 31) : 0u);
        lp[j] = __reduce_or_sync(0xffffffffu, m);
    }
    auto prefix = [&](int x)
    {
        int const wi = x >> 5;
        uint32_t const below = (1u << (x & 31)) - 1u;
        return (wi > 0 ? __popc(lp[0]) : 0) + (wi > 1 ? __popc(lp[1]) : 0)
            + __popc((wi == 0          ? lp[0]
                             : wi == 1 ? lp[1]
                                       : lp[2])
                & below);
    };
    int const E = __popc(lp[0]) + __popc(lp[1]) + __popc(lp[2]);
    int* slotExpert = rd<int>(s, kMoeRdSlotExpert);
    uint32_t* slotMask = rd<uint32_t>(s, kMoeRdSlotMask);
    int8_t* slotK = rd<int8_t>(s, kMoeRdSlotK);
    int8_t* tokSlot = rd<int8_t>(s, kMoeRdTokSlot);
#pragma unroll
    for (int j = 0; j < 3; ++j)
    {
        if ((lp[j] >> lane) & 1u)
        {
            slotExpert[prefix(32 * j + lane)] = 32 * j + lane;
        }
    }
#pragma unroll
    for (int h = 0; h < 2; ++h)
    {
        int const tk = lane + 32 * h; // = 8 t + k
        if (id[h] < kMoeWindow)
        {
            int const slot = prefix(id[h]);
            tokSlot[tk] = static_cast<int8_t>(slot);
            atomicOr(slotMask + slot, 1u << (tk >> 3));
            slotK[slot * kTok + (tk >> 3)] = static_cast<int8_t>(tk & 7);
        }
        else
        {
            tokSlot[tk] = -1;
        }
    }
    if (lane == 0)
    {
        rd<int>(s, kMoeRdScal)[kMoeScE] = E;
    }
    __syncwarp();
}

// ---- end kMoeFlagW2PlaceWarp ----

// ---- 3. the weight stream (one thread)
// ---------------------------------------------------------------------------------
template <int kNst = kMoeStages, bool kPx = false> // kNst: the ring depth; kPx: the stock queue
__device__ __forceinline__ void issueUnit(Sm const& s, Cut const& c, int j, uint64_t policy)
{
    int fc, u, slot, mt, kb;
    unitAt<kPx>(c, j, fc, u);
    decodeUnit(fc, u, slot, mt, kb);
    int const st = j % kNst; // the kNst-stage ring
    uint32_t const full = bar(s, kMoeBarFull + st);
    uint32_t const dst = s.base + kMoeSmRing + st * kMoeUnitBytes;
    int const expert = rd<int const>(s, kMoeRdSlotExpert)[slot];
    int const rows = fc == 0 ? 2 * kMoeInter : kMoeHidden;
    void const* tmap = reinterpret_cast<void const*>(dw(s, fc == 0 ? kMdTmapW13 : kMdTmapW2));
    uint8_t const* sf = reinterpret_cast<uint8_t const*>(dw(s, fc == 0 ? kMdSfW13 : kMdSfW2))
        + (static_cast<size_t>(expert) * rows + static_cast<size_t>(mt) * 128) * (kMoeHidden / 16) + kb * kMoeUnitSf;
    mbarExpectTx(full, kMoeUnitBytes);
    tmaLoad2d(dst, tmap, kb * 128, expert * rows + mt * 128, full, policy);
    bulkCopyHint(dst + kMoeUnitA, sf, kMoeUnitSf, full, policy);
}

#if MSO_MK_XP & 32
// ---- MSO_MK_XP bit 32: the CTA's stock queue cut for every slot count, computed at the stage entry (off the request
// chain). Thread 128 + E (E in [0, kMoeMaxSlots]) stores entry E of kMoeRdXpCut: makeCutPxBody(E, cta)'s pxG, pxEarly
// and tile counts, and tile 0's (fc, first list unit) from pxTile. Each entry has one writer; the readers run after the
// stage entry's CTA barrier. ----
__device__ __noinline__ void xpCutTable(Sm const s)
{
    int const E = static_cast<int>(threadIdx.x) - 128;
    Cut const c = makeCutPxBody(E, static_cast<int>(blockIdx.x));
    int fc0, u0;
    pxTile(c, 0, fc0, u0);
    int const nT1 = c.b[0] / kMoeKb;
    int const nT = nT1 + c.b[1] / kMoeKb;
    rd<uint4>(s, kMoeRdXpCut)[E] = make_uint4(static_cast<uint32_t>(c.pxG), static_cast<uint32_t>(c.pxEarly),
        static_cast<uint32_t>(nT1) | (static_cast<uint32_t>(nT) << 16),
        static_cast<uint32_t>(u0) | (static_cast<uint32_t>(fc0) << 20));
}

static_assert(kCtas <= 0xffff && kMoeMaxSlots * kMoeFc1Units < (1 << 20), "xp: the cut table's packed fields");

// makeCutPxBody(E, cta) from the table (the same fields, read back).
__device__ __forceinline__ Cut makeCutPxTab(Sm const& s, int E, int cta)
{
    uint4 const t = rd<uint4 const>(s, kMoeRdXpCut)[E];
    Cut c;
    c.E = E;
    c.tileCut = true;
    c.U[0] = E * kMoeFc1Units;
    c.U[1] = E * kMoeFc2Units;
    c.px = true;
    c.pxCta = cta;
    c.pxG = static_cast<int>(t.x);
    c.pxEarly = static_cast<int>(t.y);
    int const nT1 = static_cast<int>(t.z & 0xffffu);
    int const nT = static_cast<int>(t.z >> 16);
    c.a[0] = 0;
    c.b[0] = kMoeKb * nT1;
    c.a[1] = 0;
    c.b[1] = kMoeKb * (nT - nT1);
    return c;
}
#endif
#if MSO_MK_XP & 64
// MSO_MK_XP bit 64: issueUnit<kNst, true>(s, c, j, policy) for an early unit j < kMoeKb (k-block j of the CTA's tile
// 0), with tile 0's (fc, first list unit) from the cut table entry of c.E instead of pxTile's decode.
template <int kNst>
__device__ __forceinline__ void issueUnitT0(Sm const& s, Cut const& c, int j, uint64_t policy)
{
    uint32_t const w = rd<uint32_t const>(s, kMoeRdXpCut)[4 * c.E + 3];
    int const fc = static_cast<int>(w >> 20);
    int slot, mt, kb;
    decodeUnit(fc, static_cast<int>(w & 0xfffffu) + j, slot, mt, kb);
    int const st = j % kNst;
    uint32_t const full = bar(s, kMoeBarFull + st);
    uint32_t const dst = s.base + kMoeSmRing + st * kMoeUnitBytes;
    int const expert = rd<int const>(s, kMoeRdSlotExpert)[slot];
    int const rows = fc == 0 ? 2 * kMoeInter : kMoeHidden;
    void const* tmap = reinterpret_cast<void const*>(dw(s, fc == 0 ? kMdTmapW13 : kMdTmapW2));
    uint8_t const* sf = reinterpret_cast<uint8_t const*>(dw(s, fc == 0 ? kMdSfW13 : kMdSfW2))
        + (static_cast<size_t>(expert) * rows + static_cast<size_t>(mt) * 128) * (kMoeHidden / 16) + kb * kMoeUnitSf;
    mbarExpectTx(full, kMoeUnitBytes);
    tmaLoad2d(dst, tmap, kb * 128, expert * rows + mt * 128, full, policy);
    bulkCopyHint(dst + kMoeUnitA, sf, kMoeUnitSf, full, policy);
}

static_assert(kMoeStagesMax <= kMoeKb, "xp bit 64: every early unit is a k-block of tile 0");
#endif
// ---- end MSO_MK_XP ----

__device__ __forceinline__ uint64_t policyEvictNormal()
{
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(p));
    return p;
}

// Unit j's weights into L2 (its A box through the tensor map, its 2 KiB of scales).
template <bool kPx = false> // kPx: the stock queue
__device__ __forceinline__ void prefetchUnit(Sm const& s, Cut const& c, int j, uint64_t policy)
{
    int fc, u, slot, mt, kb;
    unitAt<kPx>(c, j, fc, u);
    decodeUnit(fc, u, slot, mt, kb);
    int const expert = rd<int const>(s, kMoeRdSlotExpert)[slot];
    int const rows = fc == 0 ? 2 * kMoeInter : kMoeHidden;
    void const* tmap = reinterpret_cast<void const*>(dw(s, fc == 0 ? kMdTmapW13 : kMdTmapW2));
    uint8_t const* sf = reinterpret_cast<uint8_t const*>(dw(s, fc == 0 ? kMdSfW13 : kMdSfW2))
        + (static_cast<size_t>(expert) * rows + static_cast<size_t>(mt) * 128) * (kMoeHidden / 16) + kb * kMoeUnitSf;
    asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::cache_hint [%0, {%1, %2}], %3;" ::"l"(tmap),
                 "r"(kb * 128), "r"(expert * rows + mt * 128), "l"(policy)
                 : "memory");
    prefetchBulkL2Hint(sf, kMoeUnitSf, policy);
}

// ---- MSO_MK_RT bit 2: the placement without CTA barriers (place()'s results). Every warp ORs the 8 tokens' presence
// words of block b on lane b, runs place()'s ring flow on lanes 0-7, and builds the three local presence words with
// one ballot each: lane i tests bit i of the block, pos = its set bits above i, and place()'s runsOn == rank is
// pos >= moved1 + moved2 for block rank, moved2 <= pos < moved2 + moved1 for block rank - 1 and pos < moved2 for block
// rank - 2. Each table entry has one writer (warp j < 3 lane i: the slot of window id 32 j + i, its slotExpert and
// slotMask; warp t lane k: the token's pick k, its tokSlot and slotK); the tables are read after a CTA barrier. ----
#if MSO_MK_RT & 2
struct RtPlace
{
    int E;
    uint32_t lp[3]; // local presence: bit i of word j = window id 32 j + i
};

// The window id of slot `slot` (< E): the (slot + 1)-th set bit of the 96-bit local presence.
__device__ __forceinline__ int rtSlotId(RtPlace const& r, int slot)
{
    int const c0 = __popc(r.lp[0]);
    int const c01 = c0 + __popc(r.lp[1]);
    int const j = slot < c0 ? 0 : slot < c01 ? 1 : 2;
    int const n = slot - (j == 0 ? 0 : j == 1 ? c0 : c01);
    uint32_t const w = j == 0 ? r.lp[0] : j == 1 ? r.lp[1] : r.lp[2];
    return 32 * j + static_cast<int>(__fns(w, 0u, n + 1));
}

#if MSO_MK_RT & 4
// issueUnit's request of unit j = (fc, mt, kb) with the slot's window id given (the slot tables are not read).
template <int kNst>
__device__ __forceinline__ void issueUnitAt(Sm const& s, int j, int fc, int mt, int kb, int expert, uint64_t policy)
{
    int const st = j % kNst;
    uint32_t const full = bar(s, kMoeBarFull + st);
    uint32_t const dst = s.base + kMoeSmRing + st * kMoeUnitBytes;
    int const rows = fc == 0 ? 2 * kMoeInter : kMoeHidden;
    void const* tmap = reinterpret_cast<void const*>(dw(s, fc == 0 ? kMdTmapW13 : kMdTmapW2));
    uint8_t const* sf = reinterpret_cast<uint8_t const*>(dw(s, fc == 0 ? kMdSfW13 : kMdSfW2))
        + (static_cast<size_t>(expert) * rows + static_cast<size_t>(mt) * 128) * (kMoeHidden / 16) + kb * kMoeUnitSf;
    mbarExpectTx(full, kMoeUnitBytes);
    tmaLoad2d(dst, tmap, kb * 128, expert * rows + mt * 128, full, policy);
    bulkCopyHint(dst + kMoeUnitA, sf, kMoeUnitSf, full, policy);
}
#endif

// MSO_MK_RT bit 4: the stage's early ring units (stageCut's count for the spread quantization, stock queue, no
// piece 0).
__device__ __forceinline__ int rtEarlyCount(int64_t flags, int n)
{
    int nEarly = (flags & kMoeFlagEarly9) ? kMoeStages : kTok;
    int const capE = static_cast<int>((flags >> kMoeCapEarlyShift) & 15);
    if (capE > 0 && capE < nEarly)
    {
        nEarly = capE;
    }
    int issued = n < nEarly ? n : nEarly;
    int const cap = w2Cap(flags, e3Ring10(flags) ? kMoeStagesMax : kMoeStages);
    issued = issued < cap ? issued : cap;
    if ((flags & kMoeFlagPxRing8) && !e3Ring10(flags))
    {
        issued = issued < kMoeStagesPx8 ? issued : kMoeStagesPx8;
    }
    return issued;
}

__device__ __noinline__ RtPlace rtPlace(Ctx const& ctx, Sm const s, bool early)
{
    (void) early;
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const t = tid >> 5;
    int const rank = static_cast<int>(dw(s, kMdRank));
    uint32_t const presT = s.redS + kMoeRdRtPres;
    uint32_t pb = 0; // lane b < 8: block b's presence
    if (lane < kMoeRanks)
    {
        uint4 const a = lds128(presT + 4 * kTok * lane);
        uint4 const c = lds128(presT + 4 * kTok * lane + 16);
        pb = ((a.x | a.y) | (a.z | a.w)) | ((c.x | c.y) | (c.z | c.w));
    }
    // place()'s flow (lanes 0..7 = blocks), on the same loads.
    int const load = __popc(pb);
    int const D = static_cast<int>(__reduce_add_sync(0xffffffffu, static_cast<uint32_t>(load)));
    int tr = 0;
    int arc = 0;
#pragma unroll
    for (int n = 1; n < kMoeRanks; ++n)
    {
        arc += __shfl_sync(0xffffffffu, load, (lane - n + 1) & 7);
        int const reach = n + 2 < kMoeRanks ? n + 2 : kMoeRanks;
        tr = max(tr, (arc + reach - 1) / reach);
    }
    int const T = max((D + kMoeRanks - 1) / kMoeRanks,
        static_cast<int>(__reduce_max_sync(0xffffffffu, static_cast<uint32_t>(lane < kMoeRanks ? tr : 0))));
    int cf = 0;
    arc = 0;
#pragma unroll
    for (int n = 1; n <= kMoeRanks; ++n)
    {
        arc += __shfl_sync(0xffffffffu, load, (lane - n + 1) & 7);
        cf = max(cf, arc - n * T);
    }
    int const lv0 = min(cf, load);
    int const cNext = __shfl_sync(0xffffffffu, cf, (lane + 1) & 7);
    int const lv0Next = __shfl_sync(0xffffffffu, lv0, (lane + 1) & 7);
    int const lv1 = min(max(0, cNext - lv0Next), lv0);
    int const moved1 = lv0 - lv1;
    int const moved2 = lv1;
    // The three local presence words.
    int const b1 = (rank - 1) & 7;
    int const b2 = (rank - 2) & 7;
    uint32_t const p0 = __shfl_sync(0xffffffffu, pb, rank);
    uint32_t const p1 = __shfl_sync(0xffffffffu, pb, b1);
    uint32_t const p2 = __shfl_sync(0xffffffffu, pb, b2);
    int const keep0 = __shfl_sync(0xffffffffu, lv0, rank);
    int const mv2b1 = __shfl_sync(0xffffffffu, moved2, b1);
    int const mv1b1 = __shfl_sync(0xffffffffu, moved1, b1);
    int const mv2b2 = __shfl_sync(0xffffffffu, moved2, b2);
    int const q0 = __popc(static_cast<uint32_t>(static_cast<uint64_t>(p0) >> (lane + 1)));
    int const q1 = __popc(static_cast<uint32_t>(static_cast<uint64_t>(p1) >> (lane + 1)));
    int const q2 = __popc(static_cast<uint32_t>(static_cast<uint64_t>(p2) >> (lane + 1)));
    RtPlace r;
    r.lp[0] = __ballot_sync(0xffffffffu, ((p0 >> lane) & 1u) != 0 && q0 >= keep0);
    r.lp[1] = __ballot_sync(0xffffffffu, ((p1 >> lane) & 1u) != 0 && q1 >= mv2b1 && q1 < mv2b1 + mv1b1);
    r.lp[2] = __ballot_sync(0xffffffffu, ((p2 >> lane) & 1u) != 0 && q2 < mv2b2);
    r.E = __popc(r.lp[0]) + __popc(r.lp[1]) + __popc(r.lp[2]);
    // ---- MSO_MK_RT bit 4: lane 0 of warp j < issued requests early ring unit j now (its slot's window id from r) ----
#if MSO_MK_RT & 4
    if (early && lane == 0 && r.E > 0)
    {
        Cut const c = makeCutPxBody(r.E, static_cast<int>(blockIdx.x));
        if (t < rtEarlyCount(icMoeFlags(dw(s, kMdFlags)), unitsOf(c)))
        {
            int fc, u, slot, mt, kb;
            unitAt<true>(c, t, fc, u);
            decodeUnit(fc, u, slot, mt, kb);
            issueUnitAt<kMoeStages>(s, t, fc, mt, kb, rtSlotId(r, slot), policyEvictFirst());
        }
    }
#endif
    // ---- end MSO_MK_RT bit 4 ----
    uint8_t* dbg = dptr<uint8_t>(s, kMdDbg);
    bool const dbgOn = dbg != nullptr && blockIdx.x == 0;
    if (dbgOn && t == 0)
    {
        int const prev1 = __shfl_sync(0xffffffffu, moved1, (lane - 1) & 7);
        int const prev2 = __shfl_sync(0xffffffffu, moved2, (lane - 2) & 7);
        if (lane < kMoeRanks)
        {
            reinterpret_cast<int*>(dbg + kMoeDbgLoads)[lane] = load;
            reinterpret_cast<int*>(dbg + kMoeDbgLoads)[kMoeRanks + lane] = load - moved1 - moved2 + prev1 + prev2;
        }
    }
    uint32_t const lp0 = r.lp[0], lp1 = r.lp[1], lp2 = r.lp[2];
    auto prefix = [&](int id)
    {
        int const w = id >> 5;
        uint32_t const below = (1u << (id & 31)) - 1u;
        return (w > 0 ? __popc(lp0) : 0) + (w > 1 ? __popc(lp1) : 0)
            + __popc((w == 0          ? lp0
                             : w == 1 ? lp1
                                      : lp2)
                & below);
    };
    if (t < 3)
    {
        uint32_t const w = t == 0 ? lp0 : t == 1 ? lp1 : lp2;
        if ((w >> lane) & 1u)
        {
            int const id = 32 * t + lane;
            int const slot = prefix(id);
            rd<int>(s, kMoeRdSlotExpert)[slot] = id;
            // The tokens of the expert: bit (id & 31) of its block's 8 token words.
            uint32_t const blk = presT + 4 * kTok * static_cast<uint32_t>((rank - t) & 7);
            uint4 const a = lds128(blk);
            uint4 const c = lds128(blk + 16);
            uint32_t const wt[8] = {a.x, a.y, a.z, a.w, c.x, c.y, c.z, c.w};
            uint32_t m = 0;
#pragma unroll
            for (int tt = 0; tt < kTok; ++tt)
            {
                m |= ((wt[tt] >> lane) & 1u) << tt;
            }
            rd<uint32_t>(s, kMoeRdSlotMask)[slot] = m;
        }
    }
    if (lane < kMoeTopK)
    {
        int const tk = kMoeTopK * t + lane;
        int const e = rd<int const>(s, kMoeRdSel)[tk];
        int const b = e >> 5;
        int const i = e & 31;
        int const j = (rank - b) & 7;
        uint32_t const w = j == 0 ? lp0 : j == 1 ? lp1 : lp2;
        bool const loc = j < 3 && ((w >> i) & 1u) != 0;
        int const id = loc ? kMoeBlock * j + i : kMoeWindow + e % (kMoeExperts - kMoeWindow);
        if (loc)
        {
            int const slot = prefix(id);
            rd<int8_t>(s, kMoeRdTokSlot)[tk] = static_cast<int8_t>(slot);
            rd<int8_t>(s, kMoeRdSlotK)[slot * kTok + t] = static_cast<int8_t>(lane);
        }
        else
        {
            rd<int8_t>(s, kMoeRdTokSlot)[tk] = -1;
        }
        if (dbgOn)
        {
            reinterpret_cast<int*>(dbg + kMoeDbgLocal)[tk] = id;
        }
    }
    if (tid == 0)
    {
        rd<int>(s, kMoeRdScal)[kMoeScE] = r.E;
    }
    if (dbgOn && tid <= kMoeMaxSlots)
    {
        reinterpret_cast<int*>(dbg + kMoeDbgSlots)[tid] = tid == 0 ? r.E : (tid - 1 < r.E ? rtSlotId(r, tid - 1) : -1);
    }
    return r;
}
#endif
// ---- end MSO_MK_RT bit 2 ----

// ---- the pool rows' bulk copy at stage entry reads the BF16 pool buffer kMdPool; with kMoeFlagMhF2 the rows are the
// F2' area's two FP32 K halves, polled in place (shareQuant) ----
__device__ __forceinline__ bool moePoolCopy(int64_t flags)
{
    return !(flags & kMoeFlagNoPoolCopy) && !(flags & kMoeFlagMhF2);
}

// ---- end pool copy ----

// ---- 4. the pre-expert RMSNorm + NVFP4 quantization: warp t = token t, lane l = blocks l + 32 j (j = 0..7)
// ------------ The row is polled into registers and staged in ring stages kMoeStagesEarly.. (requested after the
// quantization), so the loops below stay rolled (this code runs once per launch and is fetched cold). The sum of
// squares keeps the block-per-thread order bit for bit: per block its 16 squares in order, per j the 32 blocks [32 j,
// 32 j + 32) by the xor butterfly, then the 8 partial sums in j order. No cross-warp exchange. Lane 0 of warp w <
// nEarly requests ring unit w while the warp's row loads are in flight.
template <bool kPx = false> // kPx: the stock queue (its early requests inline, as the byte / tile cut's)
__device__ __noinline__ void quantize(Ctx const& ctx, Sm const s, Cut const c, int nEarly)
{
    int const gainOff = kMoeSmGain; // the local quantization keeps the 9-stage layout
    static_assert(kTok == kWarps, "one warp per token");
    int const t = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
    uint8_t const* row = dptr<uint8_t const>(s, kMdPool) + t * (kMoeHidden * 2) + lane * 32;
    uint32_t const stg = s.base + kMoeSmQStage + t * (kMoeHidden * 2) + lane * 32;
    float const gs = *dptr<float const>(s, kMdGs);
    bool const issueFirst = (dw(s, kMdFlags) & kMoeFlagIssueFirst) != 0;
    // ---- the F2' area's pooled rows (kMoeFlagMhF2) are read by the spread quantization only ----
    if (dw(s, kMdFlags) & kMoeFlagMhF2)
    {
        __trap(); // kMoeFlagLocalQuant / kMoeFlagWarmTwice with kMoeFlagMhF2: not supported
    }
    // ---- end kMoeFlagMhF2 ----
    bool const poolCopy = moePoolCopy(dw(s, kMdFlags));
    if ((issueFirst || poolCopy) && lane == 0 && t < nEarly)
    {
        if (t == 0)
        {
            moeStamp(ctx, kMoeStPreIssue);
        }
        issueUnit<kMoeStages, kPx>(s, c, t, policyEvictFirst());
    }
    {
        uint4 v[16];
        uint8_t const* a[16];
        if (poolCopy)
        {
            // The rows were copied into the staging at stage entry (before the routing); the copy is only an early
            // read: a word still kEmpty there is polled from global memory below like any other.
            waitBar(ctx, bar(s, kMoeBarPool), 0, kCodeMoeBar);
        }
#pragma unroll
        for (int i = 0; i < 16; ++i)
        {
            a[i] = row + (i >> 1) * 1024 + (i & 1) * 16;
            v[i] = poolCopy ? lds128(stg + (i >> 1) * 1024 + (i & 1) * 16) : ldPoll128(a[i]);
        }
        if (!issueFirst && !poolCopy && lane == 0 && t < nEarly)
        {
            if (t == 0)
            {
                moeStamp(ctx, kMoeStPreIssue);
            }
            issueUnit<kMoeStages, kPx>(s, c, t, policyEvictFirst());
        }
        if (threadIdx.x == 0)
        {
            moeStamp(ctx, kMoeStEarly);
        }
        __syncwarp();
        uint64_t t0 = 0;
        // Vectors read from global memory (all of them without the pool copy): only these are stored to the staging.
        uint32_t fromGlobal = poolCopy ? 0u : 0xffffu;
        while (true)
        {
            bool ok = true;
#pragma unroll
            for (int i = 0; i < 16; ++i)
            {
                ok &= full4(v[i]);
            }
            if (warpAll(ok))
            {
                break;
            }
            spinCheck(ctx, t0, kCodeMoePool, 16, 0);
#pragma unroll
            for (int i = 0; i < 16; ++i)
            {
                if (!full4(v[i]))
                {
                    v[i] = ldPoll128(a[i]);
                    fromGlobal |= 1u << i;
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 16; ++i)
        {
            if ((fromGlobal >> i) & 1u)
            {
                sts128(stg + (i >> 1) * 1024 + (i & 1) * 16, v[i]);
            }
        }
    }
    // The token's 16-byte rows of the token-scale staging (per K-step pair: step 2p, 0, step 2p + 1, 0): lane = pair.
    sts128(s.redS + kMoeRdStgX + lane * 128 + t * 16, make_uint4(0, 0, 0, 0));
    __syncwarp();
    if (threadIdx.x == 0)
    {
        moeStamp(ctx, kMoeStPool);
    }
    float tot = 0.f;
#pragma unroll 2
    for (int j = 0; j < 8; ++j)
    {
        uint4 const x0 = lds128(stg + j * 1024);
        uint4 const x1 = lds128(stg + j * 1024 + 16);
        uint32_t const w[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
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
        tot = addRn(tot, ss);
    }
    float const eps = __uint_as_float(static_cast<uint32_t>(dw(s, kMdEps)));
    float const rstd = rsqrtFtz(addRn(mulRn(tot, 1.f / kMoeHidden), eps));
    if (threadIdx.x == 0)
    {
        moeStamp(ctx, kMoeStRstd);
    }
    waitBar(ctx, bar(s, kMoeBarGain), 0, kCodeMoeBar);
    float const rcp6 = rcpFtz(6.f);
    uint8_t* dbg = dptr<uint8_t>(s, kMdDbg);
#pragma unroll 2
    for (int j = 0; j < 8; ++j)
    {
        int const b = lane + 32 * j;
        uint4 const x0 = lds128(stg + j * 1024);
        uint4 const x1 = lds128(stg + j * 1024 + 16);
        uint4 const g0 = lds128(s.base + gainOff + b * 32); // the gain at gainOff
        uint4 const g1 = lds128(s.base + gainOff + b * 32 + 16);
        uint32_t const w[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
        uint32_t const gw[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
        uint32_t xw[8];
        uint32_t m2 = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            xw[i] = bf16x2Mul(w[i], gw[i]);
            m2 = bf16x2AbsMax(m2, xw[i]);
        }
        float const amax = fmaxf(bf16Lo(m2), bf16Hi(m2));
        float const maxAbs = mulRn(amax, rstd);
        float const scaleF = minF(mulRn(mulRn(gs, maxAbs), rcp6), 448.f);
        uint32_t const sf = cvtE4m3(scaleF);
        float const sfF = e4m3ToF32(sf);
        float const inv = sfF == 0.f ? 0.f : mulRn(rcpFtz(sfF), gs);
        uint32_t code[2] = {0, 0};
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            float2 const q = mulRn2(
                mulRn2(make_float2(bf16Lo(xw[i]), bf16Hi(xw[i])), make_float2(rstd, rstd)), make_float2(inv, inv));
            code[i >> 2] |= cvtE2m1x2(q.x, q.y) << (8 * (i & 3));
        }
        int const kb = b >> 4;
        int const c16 = (b & 15) >> 1;
        int const step = b >> 2;
        sts64(s.base + kMoeSmX + kb * 1024 + t * 128 + ((c16 ^ t) << 4) + (b & 1) * 8, code[0], code[1]);
        rd<uint8_t>(s, kMoeRdStgX)[(step >> 1) * 128 + t * 16 + (step & 1) * 8 + (b & 3)] = static_cast<uint8_t>(sf);
        if (dbg != nullptr && blockIdx.x == 0)
        {
            *reinterpret_cast<uint2*>(dbg + kMoeDbgXq + t * (kMoeHidden / 2) + b * 8) = make_uint2(code[0], code[1]);
            dbg[kMoeDbgXsf + t * (kMoeHidden / 16) + b] = static_cast<uint8_t>(sf);
        }
    }
}

// ---- 4'. the quantization spread over the CTAs (after the routing): -------------------------------------------------
// The 512 (row, 4-block) groups of X are split over the kCtas CTAs; CTA c quantizes groups [512 c / kCtas,
// 512 (c + 1) / kCtas) (at most two rows) and publishes their codes and scales as Lamport words into the workspace's X
// region; every CTA then gathers the whole X into the FC1 B operand and its token-scale staging. The row's sum of
// squares keeps the block-per-thread order of quantize() bit for bit (thread 32 w + l = block 32 w + l: its 16 squares
// in order, the xor butterfly over the warp, the 8 warps' sums in order), so the codes and scales are quantize()'s.
#if MSO_MK_XP & 512
// MSO_MK_XP bit 512: ceil(2^16 / n), and whether (x * it) >> 16 == x / n for every x in [0, 64 (n + 1)]
// (shareRange's range)
__host__ __device__ constexpr int xpRecip16(int n)
{
    return (65536 + n - 1) / n;
}

__host__ __device__ constexpr bool xpRecip16Exact(int n)
{
    for (int x = 0; x <= (kMoeHidden / 64) * (n + 1); ++x)
    {
        if (((x * xpRecip16(n)) >> 16) != x / n)
        {
            return false;
        }
    }
    return true;
}

static_assert(xpRecip16Exact((kCtas - 1 - ((MSO_MK_XP & 1024) ? 64 : 0)) / kTok)
        && xpRecip16Exact((kCtas - 1 - ((MSO_MK_XP & 1024) ? 64 : 0)) / kTok + 1),
    "xp bit 512: the row split's reciprocal multiplications are exact");
#endif
__device__ __forceinline__ void shareRange(int& lo, int& hi, int& r0, int& nr)
{
    constexpr int kGroups = kTok * (kMoeHidden / 64);
#if MSO_MK_XP & 3
    // MSO_MK_XP bits 1 / 2: the row owners [kXpOwner0, kXpOwner0 + kXpPerRow kTok): owner o takes groups
    // [o G, (o + 1) G) with G = kGroups / (kXpPerRow kTok), i.e. row o / kXpPerRow (or its half o % kXpPerRow); every
    // other CTA none
    constexpr int kOwn = kGroups / (kXpPerRow * kTok);
    int const o = static_cast<int>(blockIdx.x) - kXpOwner0;
    bool const own = o >= 0 && o < kXpPerRow * kTok;
    lo = own ? kOwn * o : kGroups;
    hi = own ? kOwn * (o + 1) : kGroups;
#elif MSO_MK_XP & 512
    // MSO_MK_XP bit 512 (needs MSO_MK_RP): row-aligned shares. The quantizing CTAs [kQ0, kCtas - 1) (bit 1024:
    // kQ0 = 64, the MhPool CTAs [0, 64) quantize none; the routing publisher none) are split over the kTok rows in
    // order, CTA i of them in row r = i kTok / kNq, and the n CTAs of row r split its 64 groups as
    // [64 j / n, 64 (j + 1) / n): no CTA polls two rows
    constexpr int kQ0 = (MSO_MK_XP & 1024) ? 64 : 0;
    constexpr int kNq = kCtas - 1 - kQ0;
    constexpr int kRowG = kMoeHidden / 64;
    int const i = static_cast<int>(blockIdx.x) - kQ0;
    if (i < 0 || i >= kNq)
    {
        lo = kGroups;
        hi = kGroups;
    }
    else
    {
        // n = the row's CTA count, kN0 or kN0 + 1; x / n as (x * M(n)) >> 16 with the selected constant M (exact over
        // the range, static_assert below; a select between two divisions compiles into a run-time 16-bit division)
        constexpr int kN0 = kNq / kTok;
        int const r = (i * kTok) / kNq;
        int const f0 = (r * kNq + kTok - 1) / kTok;
        int const j = i - f0;
        bool const n0 = ((r + 1) * kNq + kTok - 1) / kTok - f0 == kN0;
        constexpr int kM0 = xpRecip16(kN0);
        constexpr int kM1 = xpRecip16(kN0 + 1);
        int const m = n0 ? kM0 : kM1;
        lo = r * kRowG + ((kRowG * j * m) >> 16);
        hi = r * kRowG + ((kRowG * (j + 1) * m) >> 16);
    }
#elif MSO_MK_RP & 1
    // MSO_MK_RP: the groups over CTAs [0, kCtas - 1); the publisher CTA kCtas - 1 quantizes none
    constexpr int kShare = kCtas - 1;
    int const b = static_cast<int>(blockIdx.x);
    lo = b < kShare ? (kGroups * b) / kShare : kGroups;
    hi = b < kShare ? (kGroups * (b + 1)) / kShare : kGroups;
#else
    lo = (kGroups * static_cast<int>(blockIdx.x)) / kCtas;
    hi = (kGroups * (static_cast<int>(blockIdx.x) + 1)) / kCtas;
#endif
    r0 = lo / (kMoeHidden / 64);
    nr = hi > lo ? (hi - 1) / (kMoeHidden / 64) - r0 + 1 : 0;
}

static_assert(kTok * (kMoeHidden / 64) / kCtas + 1 < kMoeHidden / 64, "a CTA's groups span at most two rows");
#if MSO_MK_RP & 1
static_assert(kTok * (kMoeHidden / 64) / (kCtas - 1) + 1 < kMoeHidden / 64, "rp: a CTA's groups span at most two rows");
#endif
#if MSO_MK_XP & 512
static_assert(kTok * (kMoeHidden / 64) <= 512 && (kCtas - 1 - ((MSO_MK_XP & 1024) ? 64 : 0)) / kTok <= kMoeHidden / 64,
    "xp bit 512: every quantizing CTA holds at least one group of one row");
#endif
#if MSO_MK_XP & 3
static_assert((kMoeHidden / 64) % kXpPerRow == 0, "xp: an owner's groups lie in one row");
static_assert(kXpOwner0 + kXpPerRow * kTok <= kCtas - 1, "xp: the owners are not rp's publisher (CTA kCtas - 1)");
#endif

// Returns whether this warp staged the routing's logit vectors (kMoeFlagW2LogitsPre; warp-uniform).
__device__ __noinline__ bool shareQuant(Ctx const& ctx, Sm const s, uint8_t* wsCur)
{
    int const t = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
    int const b = 32 * t + lane; // this thread's block of each row
    int lo, hi, r0, nr;
    shareRange(lo, hi, r0, nr);
    bool const poolCopy = moePoolCopy(dw(s, kMdFlags)); // the rows' bulk copy at stage entry
    uint8_t const* pool = dptr<uint8_t const>(s, kMdPool);
    float* red = rd<float>(s, kMoeRdRed);
    if (poolCopy)
    {
        waitBar(ctx, bar(s, kMoeBarPool), 0, kCodeMoeBar);
    }
    uint4 x0 = make_uint4(0, 0, 0, 0), x1 = make_uint4(0, 0, 0, 0);
    int myRow = -1;
#pragma unroll 1
    for (int rr = 0; rr < nr; ++rr)
    {
        int const row = r0 + rr;
        uint8_t const* a[2] = {pool + row * (kMoeHidden * 2) + b * 32, pool + row * (kMoeHidden * 2) + b * 32 + 16};
        uint4 v[2];
        if (poolCopy)
        {
            uint32_t const stg = s.base + kMoeSmQStage + rr * (kMoeHidden * 2) + b * 32;
            v[0] = lds128(stg);
            v[1] = lds128(stg + 16);
            // The copy is only an early read: a vector still holding kEmpty is polled from global memory.
            uint64_t t0 = 0;
            while (!warpAll(full4(v[0]) & full4(v[1])))
            {
                spinCheck(ctx, t0, kCodeMoePool, 2, 0);
                v[0] = full4(v[0]) ? v[0] : ldPoll128(a[0]);
                v[1] = full4(v[1]) ? v[1] : ldPoll128(a[1]);
            }
        }
        // ---- kMoeFlagMhF2: the pooled row from the F2' area's two FP32 K halves (bf16(h0 + h1), mhPoolValues) ----
#if MSO_MK_LANES & 8
        else if (dw(s, kMdFlags) & kMoeFlagMhF2)
        {
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
            pollVecs<8>(ctx, pv, ap, kCodeMoePool);
            uint4 const q0[4] = {pv[0], pv[1], pv[2], pv[3]};
            uint4 const q1[4] = {pv[4], pv[5], pv[6], pv[7]};
            v[0] = mhPoolValues(q0);
            v[1] = mhPoolValues(q1);
        }
#endif
        // ---- end kMoeFlagMhF2 ----
        else
        {
            pollVecs<2>(ctx, v, a, kCodeMoePool);
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
    // ---- kMoeFlagW2LogitsPre: the logit vectors route() polls, requested now (they come back under the rest) ----
    bool const lpre = (dw(s, kMdFlags) & kMoeFlagW2LogitsPre) && (dw(s, kMdFlags) & kMoeFlagMhF2)
        && (dw(s, kMdFlags) & kMoeFlagLogitsCoop);
    uint4 lv[2] = {emptyVec(), emptyVec()};
#if MSO_MK_LANES & 8
    if (lpre)
    {
        uint8_t const* const lbase = mhUc(ctx, ctx.copy) + kMhLogitOff;
        lv[0] = ldPoll128(lbase + 16 * threadIdx.x);
        lv[1] = ldPoll128(lbase + 16 * (threadIdx.x + kThreads));
    }
#endif
    // ---- end kMoeFlagW2LogitsPre ----
    __syncthreads();
    // The gain (and the warm-up boxes into the X area, which the gather overwrites) are in for every thread.
    waitBar(ctx, bar(s, kMoeBarGain), 0, kCodeMoeBar);
    uint32_t const gainOff = e3SmGain(s); // ring stage 9 with kMoeFlagE3Ring10
    uint32_t sf = 0;
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
        float const gs = *dptr<float const>(s, kMdGs);
        float const rcp6 = rcpFtz(6.f);
        uint4 const g0 = lds128(s.base + gainOff + b * 32); // the gain at gainOff
        uint4 const g1 = lds128(s.base + gainOff + b * 32 + 16);
        uint32_t const w[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
        uint32_t const gw[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
        uint32_t xw[8];
        uint32_t m2 = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            xw[i] = bf16x2Mul(w[i], gw[i]);
            m2 = bf16x2AbsMax(m2, xw[i]);
        }
        float const amax = fmaxf(bf16Lo(m2), bf16Hi(m2));
        float const maxAbs = mulRn(amax, rstd);
        float const scaleF = minF(mulRn(mulRn(gs, maxAbs), rcp6), 448.f);
        sf = cvtE4m3(scaleF);
        float const sfF = e4m3ToF32(sf);
        float const inv = sfF == 0.f ? 0.f : mulRn(rcpFtz(sfF), gs);
        uint32_t code[2] = {0, 0};
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            float2 const qv = mulRn2(
                mulRn2(make_float2(bf16Lo(xw[i]), bf16Hi(xw[i])), make_float2(rstd, rstd)), make_float2(inv, inv));
            code[i >> 2] |= cvtE2m1x2(qv.x, qv.y) << (8 * (i & 3));
        }
        st64(
            wsCur + kMoeWsXq + (r0 + myRow) * (kMoeHidden / 2) + b * 8, make_uint2(lamport(code[0]), lamport(code[1])));
    }
    // The group's four scale bytes as one word (lanes 4 k .. 4 k + 3; never kEmpty: a scale byte is never 0x80).
    uint32_t sfw = sf << (8 * (b & 3));
    sfw |= __shfl_xor_sync(0xffffffffu, sfw, 1);
    sfw |= __shfl_xor_sync(0xffffffffu, sfw, 2);
    if (myRow >= 0 && (b & 3) == 0)
    {
        st32(wsCur + kMoeWsXsf + (r0 + myRow) * (kMoeHidden / 16) + b, sfw);
    }
    if (threadIdx.x == 0)
    {
        moeStamp(ctx, kMoeStRstd);
    }
    // ---- kMoeFlagW2LogitsPre: a warp whose vectors are all full stages them as route() would ----
    bool const pre = lpre && warpAll(full4(lv[0]) & full4(lv[1]));
    if (pre)
    {
        uint32_t const stg = s.base + kMoeSmQStage;
#pragma unroll
        for (int k = 0; k < 2; ++k)
        {
            int const vi = threadIdx.x + kThreads * k;
            int const e = vi >> 1;
            int const r0q = 4 * (vi & 1);
            uint32_t const w4[4] = {lv[k].x, lv[k].y, lv[k].z, lv[k].w};
#pragma unroll
            for (int r = 0; r < 4; ++r)
            {
                asm volatile(
                    "st.shared.u32 [%0], %1;" ::"r"(stg + 4u * static_cast<uint32_t>((r0q + r) * kMoeExperts + e)),
                    "r"(w4[r])
                    : "memory");
            }
        }
    }
    return pre;
    // ---- end kMoeFlagW2LogitsPre ----
}

// Every CTA: lane 0 of warp t < nEarly requests ring unit t, then the whole X is polled from the workspace into the
// SW128 FC1 B operand and the FC1 token-scale staging (the layouts quantize() writes).
template <bool kPx = false> // kPx: the stock queue (its early requests inline, as the byte / tile cut's)
__device__ __noinline__ void gatherX(Ctx const& ctx, Sm const s, Cut const c, int nEarly, uint8_t const* wsCur)
{
    int const t = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
    // The X loads go out before the weight requests (they would queue behind the requests' traffic otherwise);
    // kMoeFlagIssueFirst: the requests first.
    bool const issueFirst = (dw(s, kMdFlags) & kMoeFlagIssueFirst) != 0;
    if (issueFirst && lane == 0 && t < nEarly)
    {
#if MSO_MK_XP & 64
        if constexpr (kPx) // MSO_MK_XP bit 64: tile 0 from the cut table
        {
            issueUnitT0<kMoeStages>(s, c, t, policyEvictFirst());
        }
        else
#endif
            issueUnit<kMoeStages, kPx>(s, c, t, policyEvictFirst());
    }
    constexpr int kVq = kMoeWsXqBytes / 16;  // 1024 code vectors
    constexpr int kVs = kMoeWsXsfBytes / 16; // 128 scale vectors
    constexpr int kPer = (kVq + kVs + kThreads - 1) / kThreads;
    uint4 v[kPer];
    {
        uint8_t const* a[kPer];
#pragma unroll
        for (int k = 0; k < kPer; ++k)
        {
            int const vi = threadIdx.x + kThreads * k;
            a[k] = vi < kVq      ? wsCur + kMoeWsXq + 16 * vi
                : vi < kVq + kVs ? wsCur + kMoeWsXsf + 16 * (vi - kVq)
                                 : nullptr;
        }
#pragma unroll
        for (int k = 0; k < kPer; ++k)
        {
            v[k] = a[k] != nullptr ? ldPoll128(a[k]) : make_uint4(0, 0, 0, 0);
        }
        if (!issueFirst && lane == 0 && t < nEarly)
        {
#if MSO_MK_XP & 64
            if constexpr (kPx) // MSO_MK_XP bit 64: tile 0 from the cut table
            {
                issueUnitT0<kMoeStages>(s, c, t, policyEvictFirst());
            }
            else
#endif
                issueUnit<kMoeStages, kPx>(s, c, t, policyEvictFirst());
        }
        // ---- kMoeFlagEarly9: the ninth ring stage too (nEarly = kMoeStages; warp 0 requests unit 8) ----
        if (lane == 0 && t + kWarps < nEarly)
        {
#if MSO_MK_XP & 64
            if constexpr (kPx) // MSO_MK_XP bit 64: tile 0 from the cut table
            {
                issueUnitT0<kMoeStages>(s, c, t + kWarps, policyEvictFirst());
            }
            else
#endif
                issueUnit<kMoeStages, kPx>(s, c, t + kWarps, policyEvictFirst());
        }
        // ---- end kMoeFlagEarly9 ----
        if (threadIdx.x == 0)
        {
            moeStamp(ctx, kMoeStEarly);
        }
        uint64_t t0 = 0;
        while (true)
        {
            bool ok = true;
#pragma unroll
            for (int k = 0; k < kPer; ++k)
            {
                ok &= full4(v[k]);
            }
            if (warpAll(ok))
            {
                break;
            }
            spinCheck(ctx, t0, kCodeMoePool, kPer, 0);
#pragma unroll
            for (int k = 0; k < kPer; ++k)
            {
                if (!full4(v[k]))
                {
                    v[k] = ldPoll128(a[k]);
                }
            }
        }
    }
    if (threadIdx.x == 0)
    {
        moeStamp(ctx, kMoeStPool);
    }
    uint8_t* dbg = dptr<uint8_t>(s, kMdDbg);
    bool const dbgOn = dbg != nullptr && blockIdx.x == 0;
    uint32_t const xOff = e3SmX(s); // the launch's X area
#pragma unroll
    for (int k = 0; k < kPer; ++k)
    {
        int const vi = threadIdx.x + kThreads * k;
        if (vi < kVq)
        {
            // 16 B = blocks i and i + 1 of row r: one 16-byte chunk of the SW128 operand.
            int const r = vi >> 7;
            int const i = (vi & 127) * 2;
            int const kb = i >> 4;
            int const c16 = (i & 15) >> 1;
            sts128(s.base + xOff + kb * 1024 + r * 128 + ((c16 ^ r) << 4), v[k]); // the launch's X area
            if (dbgOn)
            {
                *reinterpret_cast<uint4*>(dbg + kMoeDbgXq + 16 * vi) = v[k];
            }
        }
        else if (vi < kVq + kVs)
        {
            // 16 scales of row r = K steps 4 m .. 4 m + 3: staging pairs 2 m and 2 m + 1, (step, 0, step + 1, 0) each.
            int const vs = vi - kVq;
            int const r = vs >> 4;
            int const p0 = 2 * (vs & 15);
            sts128(s.redS + kMoeRdStgX + p0 * 128 + r * 16, make_uint4(v[k].x, 0u, v[k].y, 0u));
            sts128(s.redS + kMoeRdStgX + (p0 + 1) * 128 + r * 16, make_uint4(v[k].z, 0u, v[k].w, 0u));
            if (dbgOn)
            {
                *reinterpret_cast<uint4*>(dbg + kMoeDbgXsf + 16 * vs) = v[k];
            }
        }
    }
    // ---- attribution: thread 0's X staged (kStampI2aXStaged) ----
    if (threadIdx.x == 0 && MSO_MK_STAMPS_DETAIL(ctx) != nullptr)
    {
        MSO_MK_STAMPS(ctx)[kStampI2aXStaged] = stampTime(); // its own stamp word; the stamp clock (MSO_MK_SY_CLK)
    }
    // ---- end attribution ----
}

// ---- 5a. MMA issue (warp 1)
// ------------------------------------------------------------------------------------------ The elected lane's tensor
// operations of one unit: its FC1 token scales on first use of the k-block, the 4 weight-scale copies, the 4 MMAs (K =
// 64 each) into accumulator acc.
template <int kNst = kMoeStages> // kNst: the launch's ring layout
__device__ __forceinline__ void issueUnitMmas(Sm const& s, uint32_t tmem, int st, int fc, int kb, int acc, bool first,
    bool needX, bool noMma, bool noSfa, bool cpOnly)
{
    if (noMma)
    {
        return;
    }
    uint32_t const stage = s.base + kMoeSmRing + st * kMoeUnitBytes;
    uint32_t const sfa = tmem + kMoeTmSfa + 16 * st;
    if (needX)
    {
        utccpRows8(tmem + kMoeTmSfbX + 8 * kb, s.redS + kMoeRdStgX + 256 * kb);
        utccpRows8(tmem + kMoeTmSfbX + 8 * kb + 4, s.redS + kMoeRdStgX + 256 * kb + 128);
    }
    uint32_t const sfLo = descLo(stage + kMoeUnitA);
    if (!noSfa)
    {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
        {
            utccpLo(sfa + 4 * ks, sfLo + 32 * ks);
        }
    }
    uint32_t const aLo = kDescSw128Lo | descLo(stage);
    uint32_t const bOff = fc == 0 ? E3Lay<kNst>::kX : E3Lay<kNst>::kH; // the launch's B-operand layout
    uint32_t const bLo = kDescSw128Lo | descLo(s.base + bOff + kb * 1024);
    uint32_t const sfb = tmem + (fc == 0 ? kMoeTmSfbX : kMoeTmSfbH) + 8 * kb;
    uint32_t const d = tmem + kMoeTmD + 8 * acc;
    if (!cpOnly)
    {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
        {
            mmaNvf4Lo(d, aLo + 2 * ks, bLo + 2 * ks, sfa + 4 * ks, sfb + 2 * ks, (first && ks == 0) ? 0u : 1u);
        }
    }
}

template <int kNst, bool kPx = false> // kNst: the ring depth; kPx: the stock queue
__device__ __noinline__ void mmaLoop(Ctx const& ctx, Sm const s, Cut const c, uint32_t tmem)
{
    int const n = unitsOf(c);
    int const n1 = c.b[0] - c.a[0];
    tcFenceAfter();
    // The FC1 token scales of a k-block go to TMEM with its first unit (2 copies), not all 32 before the first MMA.
    uint32_t cpX = 0;
    // ---- kMoeFlagE3XsfPre: every k-block's FC1 token scales to TMEM before the first weights land ----
    if (dw(s, kMdFlags) & kMoeFlagE3XsfPre)
    {
        if (n1 > 0 && electOne())
        {
#pragma unroll 1
            for (int k = 0; k < kMoeKb; ++k)
            {
                utccpRows8(tmem + kMoeTmSfbX + 8 * k, s.redS + kMoeRdStgX + 256 * k);
                utccpRows8(tmem + kMoeTmSfbX + 8 * k + 4, s.redS + kMoeRdStgX + 256 * k + 128);
            }
        }
        __syncwarp();
        cpX = (1u << kMoeKb) - 1u;
    }
    // ---- end kMoeFlagE3XsfPre ----
    bool const noMma = (dw(s, kMdFlags) & kMoeFlagDbgNoMma) != 0;
    bool const noSfa = (dw(s, kMdFlags) & kMoeFlagDbgNoSfa) != 0;
    bool const cpOnly = (dw(s, kMdFlags) & kMoeFlagDbgCopiesOnly) != 0;
    bool const pairs = !(dw(s, kMdFlags) & kMoeFlagNoPairs);
    bool const pairs2 = (dw(s, kMdFlags) & kMoeFlagE3Fc2Pairs) != 0; // FC2 units in pairs too
    // oneElect: the FC2 h-scale copies go into the units' MMA block (one election per unit or pair, not up to three)
    bool const oneElect = (dw(s, kMdFlags) & kMoeFlagW2Fc2OneElect) != 0;
    int ti = -1;
    int prevKey = -1;
    int prevSlot2 = -1;
    int slotLoad = -1;
    uint32_t cpDone = 0;
    bool first = false;
    if (n1 == 0 && (threadIdx.x & 31) == 0)
    {
        mbarArrive(bar(s, kMoeBarFc1Done));
        mbarArrive(bar(s, kMoeBarFc1Cmpl)); // no FC1 units: complete at once
    }
#pragma unroll 1
    for (int j = 0; j < n; ++j)
    {
        int hcp1 = -1, hcp2 = -1; // oneElect: the h k-blocks whose scales the MMA block copies first
        int fc, u, slot, mt, kb;
        unitAt<kPx>(c, j, fc, u);
        decodeUnit(fc, u, slot, mt, kb);
        int const key = (fc << 20) | (slot << 8) | mt;
        if (key != prevKey)
        {
            ++ti;
            prevKey = key;
            first = true;
            if (ti >= kMoeAccs)
            {
                waitBar(ctx, bar(s, kMoeBarAccEmpty + (ti & (kMoeAccs - 1))), ((ti / kMoeAccs) - 1) & 1, kCodeMoeBar);
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
                        commit(bar(s, kMoeBarHFree));
                    }
                    __syncwarp();
                }
                ++slotLoad;
                cpDone = 0;
                prevSlot2 = slot;
            }
            if (!((cpDone >> kb) & 1u))
            {
                waitBar(ctx, bar(s, kMoeBarHReady + kb), slotLoad & 1, kCodeMoeBar);
                if (j == c.b[0] - c.a[0] && slotLoad == 0)
                {
                    if ((threadIdx.x & 31) == 0)
                    {
                        moeStamp(ctx, kMoeStFirstH);
                    }
                }
                tcFenceAfter();
                if (oneElect) // the scales go in the MMA block
                {
                    hcp1 = kb;
                }
                else
                {
                    if (electOne())
                    {
                        utccpRows8(tmem + kMoeTmSfbH + 8 * kb, s.redS + kMoeRdStgH + 256 * kb);
                        utccpRows8(tmem + kMoeTmSfbH + 8 * kb + 4, s.redS + kMoeRdStgH + 256 * kb + 128);
                    }
                    __syncwarp();
                }
                cpDone |= 1u << kb;
            }
            // ---- per-unit stamps ----
            if ((threadIdx.x & 31) == 0 && j - n1 < kE3TrUnits)
            {
                e3Tr(ctx, kE3TrHok + j - n1);
            }
            // ---- end per-unit stamps ----
        }
        int const st = j % kNst; // the kNst-stage ring
        waitBar(ctx, bar(s, kMoeBarFull + st), (j / kNst) & 1, kCodeMoeBar);
        // ---- per-unit stamps ----
        if ((threadIdx.x & 31) == 0)
        {
            if (j >= n1 && j - n1 < kE3TrUnits)
            {
                e3Tr(ctx, kE3TrFull + j - n1);
            }
            else if (j < n1 && j >= n1 - 8)
            {
                e3Tr(ctx, kE3TrFc1Full + j - (n1 - 8));
            }
        }
        // ---- end per-unit stamps ----
        // A tile ends at its k-block 15, or (the part's last tile, walked first) at the part's last unit.
        bool const lastOfTile = kb == kMoeKb - 1 || u == c.b[fc] - 1;
        // FC1 units in pairs: the next unit of the same tile (k-block kb + 1) is issued in the same elected block, so
        // the wait / election / reconvergence of the loop is paid once per two units.
        // kMoeFlagE3Fc2Pairs: FC2 units too (the second unit's h k-block is waited for here).
        bool const pair
            = (pairs && fc == 0 && !lastOfTile && j + 1 < n1) || (pairs2 && fc == 1 && !lastOfTile && j + 1 < n);
        int const st2 = (j + 1) % kNst;                                               // the kNst-stage ring
        int const kb2 = kb + 1;
        bool const lastOfTile2 = pair && (kb2 == kMoeKb - 1 || u + 1 == c.b[fc] - 1); // as lastOfTile
        // ---- the pair's second FC2 unit: its h k-block ----
        if (pair && fc == 1)
        {
            if (!((cpDone >> kb2) & 1u))
            {
                waitBar(ctx, bar(s, kMoeBarHReady + kb2), slotLoad & 1, kCodeMoeBar);
                tcFenceAfter();
                if (oneElect) // the scales go in the MMA block
                {
                    hcp2 = kb2;
                }
                else
                {
                    if (electOne())
                    {
                        utccpRows8(tmem + kMoeTmSfbH + 8 * kb2, s.redS + kMoeRdStgH + 256 * kb2);
                        utccpRows8(tmem + kMoeTmSfbH + 8 * kb2 + 4, s.redS + kMoeRdStgH + 256 * kb2 + 128);
                    }
                    __syncwarp();
                }
                cpDone |= 1u << kb2;
            }
            if ((threadIdx.x & 31) == 0 && j + 1 - n1 < kE3TrUnits)
            {
                e3Tr(ctx, kE3TrHok + j + 1 - n1);
            }
        }
        // ---- end second FC2 unit ----
        if (pair)
        {
            waitBar(ctx, bar(s, kMoeBarFull + st2), ((j + 1) / kNst) & 1, kCodeMoeBar); // the kNst-stage ring
            // ---- per-unit stamps ----
            if ((threadIdx.x & 31) == 0)
            {
                if (j + 1 >= n1 && j + 1 - n1 < kE3TrUnits)
                {
                    e3Tr(ctx, kE3TrFull + j + 1 - n1);
                }
                else if (j + 1 < n1 && j + 1 >= n1 - 8)
                {
                    e3Tr(ctx, kE3TrFc1Full + j + 1 - (n1 - 8));
                }
            }
            // ---- end per-unit stamps ----
        }
        int const jLast = pair ? j + 1 : j;
        if ((threadIdx.x & 31) == 0)
        {
            if (j == 0)
            {
                moeStamp(ctx, kMoeStFirstFull);
            }
            if (jLast == n1 - 1)
            {
                moeStamp(ctx, kMoeStFc1LastFull);
            }
            if (jLast == n - 1)
            {
                moeStamp(ctx, kMoeStLastFull);
            }
        }
        tcFenceAfter();
        bool const needX = fc == 0 && !((cpX >> kb) & 1u);
        bool const needX2 = pair && fc == 0 && !((cpX >> kb2) & 1u); // FC1 only
        int const acc = ti & (kMoeAccs - 1);
        if (electOne())
        {
            if (hcp1 >= 0) // oneElect: the h scale rows first
            {
                utccpRows8(tmem + kMoeTmSfbH + 8 * hcp1, s.redS + kMoeRdStgH + 256 * hcp1);
                utccpRows8(tmem + kMoeTmSfbH + 8 * hcp1 + 4, s.redS + kMoeRdStgH + 256 * hcp1 + 128);
            }
            issueUnitMmas<kNst>(s, tmem, st, fc, kb, acc, first, needX, noMma, noSfa, cpOnly); // the kNst-stage ring
            commit(bar(s, kMoeBarEmpty + st));
            if (lastOfTile)
            {
                commit(bar(s, kMoeBarAccFull + acc));
            }
            if (pair)
            {
                if (hcp2 >= 0) // oneElect: the h scale rows first
                {
                    utccpRows8(tmem + kMoeTmSfbH + 8 * hcp2, s.redS + kMoeRdStgH + 256 * hcp2);
                    utccpRows8(tmem + kMoeTmSfbH + 8 * hcp2 + 4, s.redS + kMoeRdStgH + 256 * hcp2 + 128);
                }
                issueUnitMmas<kNst>(s, tmem, st2, fc, kb2, acc, false, needX2, noMma, noSfa, cpOnly); // the pair's fc
                commit(bar(s, kMoeBarEmpty + st2));
                if (lastOfTile2)
                {
                    commit(bar(s, kMoeBarAccFull + acc));
                }
            }
            // ---- the CTA's FC1 MMAs complete (the shared X / h area of kMoeFlagE3Ring10) ----
            if (jLast == n1 - 1)
            {
                commit(bar(s, kMoeBarFc1Cmpl));
            }
            // ---- end FC1 complete ----
        }
        __syncwarp();
        // ---- per-unit stamps ----
        if ((threadIdx.x & 31) == 0)
        {
            if (j >= n1 && j - n1 < kE3TrUnits)
            {
                e3Tr(ctx, kE3TrIss + j - n1);
            }
            if (pair && j + 1 >= n1 && j + 1 - n1 < kE3TrUnits)
            {
                e3Tr(ctx, kE3TrIss + j + 1 - n1);
            }
        }
        // ---- end per-unit stamps ----
        first = false;
        cpX |= (needX ? (1u << kb) : 0u) | (needX2 ? (1u << kb2) : 0u);
        if (j == n1 && (threadIdx.x & 31) == 0)
        {
            moeStamp(ctx, kMoeStFc2First);
        }
        if (jLast == n1 - 1 && (threadIdx.x & 31) == 0)
        {
            moeStamp(ctx, kMoeStFc1Issued);
            mbarArrive(bar(s, kMoeBarFc1Done));
        }
        if (pair)
        {
            ++j;
        }
    }
    if ((threadIdx.x & 31) == 0)
    {
        moeStamp(ctx, kMoeStMmaDone);
    }
}

// ---- kMoeFlagE3Fc1Quads / kMoeFlagE3Fc2Quads: the MMA warp in groups of up to 4 units -------------------------------
// As mmaLoop, but up to kMoeE3Group consecutive units of the same tile (FC1: up to 4 with kMoeFlagE3Fc1Quads, else 2;
// FC2: up to 4 with kMoeFlagE3Fc2Quads, 2 with kMoeFlagE3Fc2Pairs, else 1) are waited for (weights, and for FC2 their
// h k-blocks), then issued in one elected block, each followed by its own empty-stage commit, the accumulator commit
// after a tile's last unit. A group never crosses a tile, so the slot / accumulator bookkeeping runs at its first unit.
// The MMAs, their order and their accumulators are mmaLoop's.
constexpr int kMoeE3Group = 4;

template <int kNst>
__device__ __noinline__ void mmaLoopG(Ctx const& ctx, Sm const s, Cut const c, uint32_t tmem)
{
    int const n = unitsOf(c);
    int const n1 = c.b[0] - c.a[0];
    tcFenceAfter();
    uint32_t cpX = 0;
    // ---- kMoeFlagE3XsfPre: every k-block's FC1 token scales to TMEM before the first weights land ----
    if (dw(s, kMdFlags) & kMoeFlagE3XsfPre)
    {
        if (n1 > 0 && electOne())
        {
#pragma unroll 1
            for (int k = 0; k < kMoeKb; ++k)
            {
                utccpRows8(tmem + kMoeTmSfbX + 8 * k, s.redS + kMoeRdStgX + 256 * k);
                utccpRows8(tmem + kMoeTmSfbX + 8 * k + 4, s.redS + kMoeRdStgX + 256 * k + 128);
            }
        }
        __syncwarp();
        cpX = (1u << kMoeKb) - 1u;
    }
    // ---- end kMoeFlagE3XsfPre ----
    int64_t const flags = dw(s, kMdFlags);
    bool const noMma = (flags & kMoeFlagDbgNoMma) != 0;
    bool const noSfa = (flags & kMoeFlagDbgNoSfa) != 0;
    bool const cpOnly = (flags & kMoeFlagDbgCopiesOnly) != 0;
    int const g1 = (flags & kMoeFlagE3Fc1Quads) ? 4 : (flags & kMoeFlagNoPairs) ? 1 : 2;
    int const g2 = (flags & kMoeFlagE3Fc2Quads) ? 4 : (flags & kMoeFlagE3Fc2Pairs) ? 2 : 1;
    int ti = -1;
    int prevKey = -1;
    int prevSlot2 = -1;
    int slotLoad = -1;
    uint32_t cpDone = 0;
    bool const lane0 = (threadIdx.x & 31) == 0;
    if (n1 == 0 && lane0)
    {
        mbarArrive(bar(s, kMoeBarFc1Done));
        mbarArrive(bar(s, kMoeBarFc1Cmpl));
    }
#pragma unroll 1
    for (int j = 0; j < n;)
    {
        int fc, u, slot, mt, kb;
        unitAt(c, j, fc, u);
        decodeUnit(fc, u, slot, mt, kb);
        int const key = (fc << 20) | (slot << 8) | mt;
        bool const first = key != prevKey;
        if (first)
        {
            ++ti;
            prevKey = key;
            if (ti >= kMoeAccs)
            {
                waitBar(ctx, bar(s, kMoeBarAccEmpty + (ti & (kMoeAccs - 1))), ((ti / kMoeAccs) - 1) & 1, kCodeMoeBar);
                tcFenceAfter();
            }
        }
        if (fc == 1 && slot != prevSlot2)
        {
            if (prevSlot2 >= 0)
            {
                if (electOne())
                {
                    commit(bar(s, kMoeBarHFree));
                }
                __syncwarp();
            }
            ++slotLoad;
            cpDone = 0;
            prevSlot2 = slot;
        }
        // The group: units j .. j + g - 1 = k-blocks kb .. kb + g - 1 of this tile (a tile ends at its k-block 15 or at
        // the part's last unit; FC1's group stays inside the FC1 part).
        int const lim = fc == 0 ? n1 : n;
        int const gmax = fc == 0 ? g1 : g2;
        int g = 1;
        while (g < gmax && j + g < lim && kb + g - 1 < kMoeKb - 1 && u + g - 1 < c.b[fc] - 1)
        {
            ++g;
        }
        bool const lastOfTile = kb + g - 1 == kMoeKb - 1 || u + g - 1 == c.b[fc] - 1;
#pragma unroll 1
        for (int i = 0; i < g; ++i)
        {
            int const k = kb + i;
            if (fc == 1)
            {
                if (!((cpDone >> k) & 1u))
                {
                    waitBar(ctx, bar(s, kMoeBarHReady + k), slotLoad & 1, kCodeMoeBar);
                    if (j + i == n1 && slotLoad == 0 && lane0)
                    {
                        moeStamp(ctx, kMoeStFirstH);
                    }
                    tcFenceAfter();
                    if (electOne())
                    {
                        utccpRows8(tmem + kMoeTmSfbH + 8 * k, s.redS + kMoeRdStgH + 256 * k);
                        utccpRows8(tmem + kMoeTmSfbH + 8 * k + 4, s.redS + kMoeRdStgH + 256 * k + 128);
                    }
                    __syncwarp();
                    cpDone |= 1u << k;
                }
                if (lane0 && j + i - n1 < kE3TrUnits)
                {
                    e3Tr(ctx, kE3TrHok + j + i - n1);
                }
            }
            waitBar(ctx, bar(s, kMoeBarFull + (j + i) % kNst), ((j + i) / kNst) & 1, kCodeMoeBar);
            if (lane0)
            {
                if (j + i == 0)
                {
                    moeStamp(ctx, kMoeStFirstFull);
                }
                if (j + i == n1 - 1)
                {
                    moeStamp(ctx, kMoeStFc1LastFull);
                }
                if (j + i == n - 1)
                {
                    moeStamp(ctx, kMoeStLastFull);
                }
                if (j + i >= n1 && j + i - n1 < kE3TrUnits)
                {
                    e3Tr(ctx, kE3TrFull + j + i - n1);
                }
                else if (j + i < n1 && j + i >= n1 - 8)
                {
                    e3Tr(ctx, kE3TrFc1Full + j + i - (n1 - 8));
                }
            }
        }
        tcFenceAfter();
        int const acc = ti & (kMoeAccs - 1);
        uint32_t needXm = 0;
        if (fc == 0)
        {
            uint32_t const gm = ((1u << g) - 1u) << kb;
            needXm = gm & ~cpX;
        }
        if (electOne())
        {
#pragma unroll 1
            for (int i = 0; i < g; ++i)
            {
                int const k = kb + i;
                int const st = (j + i) % kNst;
                issueUnitMmas<kNst>(
                    s, tmem, st, fc, k, acc, first && i == 0, ((needXm >> k) & 1u) != 0, noMma, noSfa, cpOnly);
                commit(bar(s, kMoeBarEmpty + st));
            }
            if (lastOfTile)
            {
                commit(bar(s, kMoeBarAccFull + acc));
            }
            if (j + g - 1 == n1 - 1)
            {
                commit(bar(s, kMoeBarFc1Cmpl));
            }
        }
        __syncwarp();
        cpX |= needXm;
        if (lane0)
        {
            if (j <= n1 && n1 < j + g)
            {
                moeStamp(ctx, kMoeStFc2First);
            }
            for (int i = 0; i < g; ++i)
            {
                if (j + i >= n1 && j + i - n1 < kE3TrUnits)
                {
                    e3Tr(ctx, kE3TrIss + j + i - n1);
                }
            }
            if (j + g - 1 == n1 - 1)
            {
                moeStamp(ctx, kMoeStFc1Issued);
                mbarArrive(bar(s, kMoeBarFc1Done));
            }
        }
        j += g;
    }
    if (lane0)
    {
        moeStamp(ctx, kMoeStMmaDone);
    }
}

// ---- end kMoeFlagE3Fc1Quads / kMoeFlagE3Fc2Quads ----

// ---- 5b. the weight stream after the initial requests (warp 0 lane 0)
// -----------------------------------------------------
template <int kNst, bool kPx = false> // kNst: the ring depth; kPx: the stock queue
__device__ __noinline__ void produceLoop(Ctx const& ctx, Sm const s, Cut const c, int from)
{
    uint64_t const policy = policyEvictFirst();
    int const n = unitsOf(c);
    // kMoeFlagFc2TailL2: the units whose stage only an FC2 MMA frees (they wait for h) go into L2 before the wait.
    int const tail = (dw(s, kMdFlags) & kMoeFlagFc2TailL2) ? c.b[0] - c.a[0] + kNst : n; // the kNst-stage ring
    int const n1 = c.b[0] - c.a[0];                                                      // the FC1 units
    int const cap = w2Cap(dw(s, kMdFlags), kNst);                      // the in-flight cap (kMoeW2CapMask)
    bool const capFc1 = (dw(s, kMdFlags) & kMoeFlagW2CapFc1Only) != 0; // the cap for the FC1 units only
    int blocked = 0;
    // ---- the landing cap: at most capD units requested and not landed (their full barriers), 0 = the ring alone ----
    int const capD = static_cast<int>((dw(s, kMdFlags) >> kMoeCapInflightShift) & 15);
    // ---- end landing cap ----
    // ---- kMoeFlagFc2TailEarly: the FC2 tail into L2 at once ----
    if (dw(s, kMdFlags) & kMoeFlagFc2TailEarly)
    {
        uint64_t const pf = policyEvictNormal();
#pragma unroll 1
        for (int jj = c.b[0] - c.a[0] + kNst; jj < n; ++jj) // the kNst-stage ring
        {
            prefetchUnit<kPx>(s, c, jj, pf);
        }
    }
    // ---- end kMoeFlagFc2TailEarly ----
#pragma unroll 1
    for (int j = from; j < n; ++j)
    {
        int const st = j % kNst; // the kNst-stage ring
        if (j == tail)
        {
            uint64_t const pf = policyEvictNormal();
#pragma unroll 1
            for (int jj = j; jj < n; ++jj)
            {
                prefetchUnit<kPx>(s, c, jj, pf);
            }
        }
        // ---- kMoeW2CapMask: at most `cap` units in flight: unit j waits for unit j - cap's MMAs (which implies
        // j - kNst's) ----
        if (cap < kNst && !(capFc1 && j >= n1)) // kMoeFlagW2CapFc1Only lifts the cap for the FC2 units
        {
            if (j >= cap)
            {
                int const jc = j - cap;
                waitBar(ctx, bar(s, kMoeBarEmpty + jc % kNst), (jc / kNst) & 1, kCodeMoeBar);
            }
        }
        else
            // ---- end kMoeW2CapMask ----
            if (j >= kNst) // the kNst-stage ring
            {
                if (j < c.b[0] - c.a[0] && MSO_MK_STAMPS_DETAIL(ctx) != nullptr
                    && !mbarTryWait(bar(s, kMoeBarEmpty + st), ((j / kNst) - 1) & 1))
                {
                    ++blocked;
                }
                waitBar(ctx, bar(s, kMoeBarEmpty + st), ((j / kNst) - 1) & 1, kCodeMoeBar);
            }
        // ---- the landing cap: unit j - capD's copy has landed. Its stage's barrier cannot be past that phase: the
        // next use of the stage is unit j - capD + kNst > j, not yet requested. Every unit below j was requested (by
        // this thread, or by gatherX before the barrier this loop follows), so the wait completes. ----
        if (capD > 0 && capD < kNst && j >= capD)
        {
            int const jj = j - capD;
            waitBar(ctx, bar(s, kMoeBarFull + jj % kNst), (jj / kNst) & 1, kCodeMoeBar);
        }
        // ---- end landing cap ----
        issueUnit<kNst, kPx>(s, c, j, policy); // the kNst-stage ring; kPx: the stock queue
        // ---- per-unit stamps ----
        if (j >= n1 && j - n1 < kE3TrUnits)
        {
            e3Tr(ctx, kE3TrReq + j - n1);
        }
        else if (j < n1 && j >= n1 - 8)
        {
            e3Tr(ctx, kE3TrFc1Req + j - (n1 - 8));
        }
        // ---- end per-unit stamps ----
        if (j == c.b[0] - c.a[0] + kNst) // the kNst-stage ring
        {
            moeStamp(ctx, kMoeStTailIssued);
        }
        // ---- attribution: the CTA's last unit requested (sub-stamp 29, unused by the spread-quantization path) ----
        if (j == n - 1 && !(dw(s, kMdFlags) & (kMoeFlagLocalQuant | kMoeFlagWarmTwice)))
        {
            moeStamp(ctx, kMoeStPreIssue);
        }
        // ---- end attribution ----
    }
    moeCount(ctx, 5, blocked); // FC1 requests that waited for a free stage
}

// ---- the table-driven issue loops (kMoeFlagLxProducer / kMoeFlagLxMma) ----------------------------------------------
// Both loops walk the CTA's unit sequence (unitAt) through a lane-parallel table: lane l of the warp holds the entries
// of units base + l and base + 32 + l of a 64-unit window, and unit j's entry is one shuffle from lane (j - base) & 31.
// Every value the per-unit path uses is broadcast from lane 0 (or comes from such a shuffle), so the compiler keeps it
// warp-uniform. The requests and the tensor operations are produceLoop's and mmaLoop's, in the same order, from one
// elected thread.
__device__ __forceinline__ bool lxProducerOk(int64_t flags, int nst)
{
    return (flags & kMoeFlagLxProducer) && !(flags & (kMoeFlagFc2TailL2 | kMoeFlagFc2TailEarly))
        && !(flags & kLxAttrRequestPath) && w2Cap(flags, nst) == nst;
}

__device__ __forceinline__ bool lxMmaOk(int64_t flags)
{
    return (flags & kMoeFlagLxMma) && (flags & kMoeFlagW2Fc2OneElect) && !(flags & kLxAttrMmaArrive)
        && !(flags & (kMoeFlagE3Fc1Quads | kMoeFlagE3Fc2Quads));
}

// Stamped builds without the per-unit stamp words (-DMSO_MK_GGX_NOTRACE=1).
#if defined(MSO_MK_GGX_NOTRACE) && MSO_MK_GGX_NOTRACE
constexpr bool kLxNoTrace = true;
#else
constexpr bool kLxNoTrace = false;
#endif

// Attribution only (delay injection): spin ns nanoseconds.
__device__ __forceinline__ void lxSpinNs(uint32_t ns)
{
    uint64_t const t0 = globalTimer();
    while (globalTimer() - t0 < ns)
    {
    }
}

template <typename T>
__device__ __forceinline__ T lxBcast(T v)
{
    return __shfl_sync(0xffffffffu, v, 0);
}

// Entry i (0 <= i < 64) of a window held as (lanes' first half a, second half b).
__device__ __forceinline__ uint32_t lxPick(uint32_t a, uint32_t b, int i)
{
    uint32_t const va = __shfl_sync(0xffffffffu, a, i & 31);
    uint32_t const vb = __shfl_sync(0xffffffffu, b, i & 31);
    return i < 32 ? va : vb;
}

// The producer's entry of unit j: the box row y = expert * rows + 128 mt | kb << 20 | fc << 24 (0 past the units).
static_assert(kMoeWindow * 2 * kMoeInter <= (1 << 20), "the box row fits 20 bits");

template <bool kPx> // kPx: the stock queue
__device__ __forceinline__ uint32_t lxProdEntry(Sm const& s, Cut const& c, int j, int n)
{
    if (j >= n)
    {
        return 0;
    }
    int fc, u, slot, mt, kb;
    unitAt<kPx>(c, j, fc, u);
    decodeUnit(fc, u, slot, mt, kb);
    int const expert = rd<int const>(s, kMoeRdSlotExpert)[slot];
    int const rows = fc == 0 ? 2 * kMoeInter : kMoeHidden;
    return static_cast<uint32_t>(expert * rows + mt * 128) | (static_cast<uint32_t>(kb) << 20)
        | (static_cast<uint32_t>(fc) << 24);
}

// The weight stream after the early requests, on the whole of warp 0 (produceLoop's requests, waits and order).
template <int kNst, bool kPx = false> // kPx: the stock queue
__device__ __noinline__ void produceLoopLx(Ctx const& ctx, Sm const s0, Cut const c, int from0)
{
    int const lane = threadIdx.x & 31;
    Sm const s{lxBcast(s0.redS), lxBcast(s0.base)};
    int const from = lxBcast(from0);
    int const n = lxBcast(unitsOf(c));
    int const n1 = lxBcast(c.b[0] - c.a[0]);
    uint64_t const policy = lxBcast(policyEvictFirst());
    int64_t const flags = icMoeFlags(lxBcast(dw(s, kMdFlags))); // the build's fixed flag bits stay constants
    int const capD = static_cast<int>((flags >> kMoeCapInflightShift) & 15);
    int const cap = (kPx && kUeNoCap) || !(capD > 0 && capD < kNst) ? 0 : capD; // 0: the ring alone; MSO_MK_UE bit 4
    bool const noStream = (flags & kLxAttrNoStream) != 0;                       // attribution: plain arrives
    bool const pd = (flags & kLxAttrProdDelay) != 0;                            // attribution: delay injection
    bool const stampPre = !(flags & (kMoeFlagLocalQuant | kMoeFlagWarmTwice));
    int64_t const tmap0 = lxBcast(dw(s, kMdTmapW13));
    int64_t const tmap1 = lxBcast(dw(s, kMdTmapW2));
    int64_t const sf0 = lxBcast(dw(s, kMdSfW13));
    int64_t const sf1 = lxBcast(dw(s, kMdSfW2));
    uint32_t const fullBar = bar(s, kMoeBarFull);
    uint32_t const emptyBar = bar(s, kMoeBarEmpty);
    uint32_t const ring = s.base + kMoeSmRing;
    int blocked = 0;
    // A 32-unit window (lane l: unit base + l), rebuilt every 32 units: one build before the first request.
    int base = from;
    uint32_t wa = lxProdEntry<kPx>(s, c, base + lane, n);
    // Unit j: ring stage st, use parity ph = (j / kNst) & 1; unit j - cap: stage stc, parity phc.
    int st = from % kNst;
    uint32_t ph = (from / kNst) & 1;
    int stc = 0;
    uint32_t phc = 0;
    if (cap > 0 && from >= cap)
    {
        stc = (from - cap) % kNst;
        phc = ((from - cap) / kNst) & 1;
    }
#pragma unroll 1
    for (int j = from; j < n; ++j)
    {
        if (j - base >= 32)
        {
            base = j;
            wa = lxProdEntry<kPx>(s, c, base + lane, n);
        }
        uint32_t const e = __shfl_sync(0xffffffffu, wa, j - base);
        if (j >= kNst)
        {
            if (!kLxNoTrace && j < n1 && MSO_MK_STAMPS_DETAIL(ctx) != nullptr
                && !mbarTryWait(emptyBar + 8 * st, ph ^ 1u))
            {
                ++blocked;
            }
            waitBar(ctx, emptyBar + 8 * st, ph ^ 1u, kCodeMoeBar);
        }
        // the landing cap: unit j - cap's copy has landed (produceLoop's argument: every unit below j was requested).
        if (cap > 0 && j >= cap)
        {
            waitBar(ctx, fullBar + 8 * stc, phc, kCodeMoeBar);
            if (++stc == kNst)
            {
                stc = 0;
                phc ^= 1u;
            }
        }
        if (pd)
        {
            lxSpinNs(kLxAttrDelayNs);
        }
        if (electOne())
        {
            uint32_t const full = fullBar + 8 * st;
            if (noStream)
            {
                mbarArrive(full);
            }
            else
            {
                int const y = static_cast<int>(e & 0xfffffu);
                int const kb = static_cast<int>((e >> 20) & 15u);
                bool const fc2 = ((e >> 24) & 1u) != 0;
                uint32_t const dst = ring + st * kMoeUnitBytes;
                uint8_t const* sf = reinterpret_cast<uint8_t const*>(fc2 ? sf1 : sf0)
                    + (static_cast<size_t>(y) * (kMoeHidden / 16) + kb * kMoeUnitSf);
                mbarExpectTx(full, kMoeUnitBytes);
                tmaLoad2d(dst, reinterpret_cast<void const*>(fc2 ? tmap1 : tmap0), kb * 128, y, full, policy);
                bulkCopyHint(dst + kMoeUnitA, sf, kMoeUnitSf, full, policy);
            }
        }
        __syncwarp();
        if (lane == 0)
        {
            if (j >= n1 && j - n1 < kE3TrUnits)
            {
                e3Tr(ctx, kE3TrReq + j - n1);
            }
            else if (j < n1 && j >= n1 - 8)
            {
                e3Tr(ctx, kE3TrFc1Req + j - (n1 - 8));
            }
            if (j == n1 + kNst)
            {
                moeStamp(ctx, kMoeStTailIssued);
            }
            if (j == n - 1 && stampPre)
            {
                moeStamp(ctx, kMoeStPreIssue);
            }
        }
        if (++st == kNst)
        {
            st = 0;
            ph ^= 1u;
        }
    }
    if (lane == 0)
    {
        moeCount(ctx, 5, blocked);
    }
}

// The MMA warp's entry of a unit (16 bits).
constexpr uint32_t kLxFc = 1u << 4;    // an FC2 unit
constexpr uint32_t kLxLast = 1u << 5;  // the last unit of its tile (the accumulator commit)
constexpr uint32_t kLxNew = 1u << 6;   // the first unit of a tile (a new accumulator)
constexpr uint32_t kLxNeedX = 1u << 7; // FC1: the first use of its k-block (the token-scale copies)
constexpr uint32_t kLxHWait = 1u << 8; // FC2: the first use of its h k-block within the slot (the h wait + scales)
constexpr uint32_t kLxSlot = 1u << 9;  // FC2: the first unit of a slot
constexpr uint32_t kLxPair = 1u << 10; // issued with the next unit (mmaLoop's pair rule)

// ---- the entry of unit j with the stock tile queue (whole tiles in k-block order; the CTA's first FC1 tile uses
// every k-block of X first; every FC2 tile is a new slot: a CTA's FC2 positions are G >= 96 apart, a slot has 32) ----
__device__ __forceinline__ uint32_t pxMmaEntry(int j, int n, int n1, bool pairs, bool pairs2, bool xsfPre)
{
    if (j >= n)
    {
        return 0;
    }
    int const fc = j < n1 ? 0 : 1;
    int const kb = j & (kMoeKb - 1);
    bool const last = kb == kMoeKb - 1;
    bool const newTile = kb == 0;
    bool const pairNext = !last && (fc == 0 ? pairs && j + 1 < n1 : pairs2 && j + 1 < n);
    bool const needX = fc == 0 && !xsfPre && j < kMoeKb;
    bool const hWait = fc == 1;
    bool const slotChange = fc == 1 && newTile;
    return static_cast<uint32_t>(kb) | (fc ? kLxFc : 0u) | (last ? kLxLast : 0u) | (newTile ? kLxNew : 0u)
        | (needX ? kLxNeedX : 0u) | (hWait ? kLxHWait : 0u) | (slotChange ? kLxSlot : 0u) | (pairNext ? kLxPair : 0u);
}

// ---- end stock queue entry ----

// The MMA warp's entry of unit j of the CTA's sequence, in closed form from the cut (listUnit's walk: the part's last
// tile first, from k-block lastFrom % 16, then the lower tiles, each in k-block order). Run r = the part's r-th tile
// walked (run 0: units [lastFrom, b), k-blocks [k0, k1]); every run >= 1 is a whole tile except the part's lowest one,
// which is the last run. So a k-block is first used in run 0, or in run 1 if run 0 did not hold it (mmaLoop's cpX for
// FC1; its cpDone, reset per FC2 slot, for the h k-blocks), and an FC2 slot starts at run 0 and wherever the walk
// crosses into a lower slot (32 tiles per slot).
template <bool kPx> // kPx: the stock queue
__device__ __forceinline__ uint32_t lxMmaEntry(Cut const& c, int j, int n, int n1, bool pairs, bool pairs2, bool xsfPre)
{
    if constexpr (kPx) // the stock queue
    {
        return pxMmaEntry(j, n, n1, pairs, pairs2, xsfPre);
    }
    if (j >= n)
    {
        return 0;
    }
    int const fc = j < n1 ? 0 : 1;
    int const jp = fc == 0 ? j : j - n1;
    int const a = c.a[fc];
    int const b = c.b[fc];
    int const lastLo = ((b - 1) / kMoeKb) * kMoeKb;
    int const lastFrom = lastLo > a ? lastLo : a;
    int const nLast = b - lastFrom;
    int r, u;
    bool newTile;
    if (jp < nLast)
    {
        r = 0;
        u = lastFrom + jp;
        newTile = jp == 0;
    }
    else
    {
        int const jj = jp - nLast;
        r = 1 + jj / kMoeKb;
        int const lo = lastLo - kMoeKb * r;
        u = (lo > a ? lo : a) + jj % kMoeKb;
        newTile = jj % kMoeKb == 0;
    }
    int const kb = u & (kMoeKb - 1);
    bool const last = kb == kMoeKb - 1 || u == b - 1;
    bool const pairNext = !last && (fc == 0 ? pairs && j + 1 < n1 : pairs2 && j + 1 < n);
    bool const notInRun0 = kb < (lastFrom & (kMoeKb - 1)) || kb > ((b - 1) & (kMoeKb - 1));
    bool const firstUse = r == 0 || (r == 1 && notInRun0);
    bool needX = false, hWait = false, slotChange = false;
    if (fc == 0)
    {
        needX = !xsfPre && firstUse;
    }
    else
    {
        int const t = (lastLo / kMoeKb) - r; // the run's tile in the FC2 list (32 per slot)
        bool const segFirst = r == 0 || (t >> 5) != ((t + 1) >> 5);
        hWait = segFirst || firstUse;
        slotChange = newTile && segFirst;
    }
    return static_cast<uint32_t>(kb) | (fc ? kLxFc : 0u) | (last ? kLxLast : 0u) | (newTile ? kLxNew : 0u)
        | (needX ? kLxNeedX : 0u) | (hWait ? kLxHWait : 0u) | (slotChange ? kLxSlot : 0u) | (pairNext ? kLxPair : 0u);
}

static_assert(kMoeFc2Tiles == 32, "lxMmaEntry: 32 FC2 tiles per slot");

// A unit's tensor-op operands (the words issueUnitMmas and the scale-row copies compute), built before the unit's waits
// so that the elected block only issues. rows / rows2: the two 128-byte row groups of the k-block's token scales (FC1,
// needX) or h scales (FC2, first use of the k-block); both land at sfb, sfb + 4.
struct LxOps
{
    uint32_t aLo, bLo, sfLo, sfa, sfb, rows, rows2, empty;
};

template <int kNst>
__device__ __forceinline__ LxOps lxOps(Sm const& s, uint32_t tmem, uint32_t emptyBar, int st, int fc, int kb)
{
    uint32_t const stage = s.base + kMoeSmRing + st * kMoeUnitBytes;
    uint32_t const rowSrc = s.redS + (fc == 0 ? kMoeRdStgX : kMoeRdStgH) + 256 * kb;
    LxOps o;
    o.aLo = kDescSw128Lo | descLo(stage);
    o.bLo = kDescSw128Lo | descLo(s.base + (fc == 0 ? E3Lay<kNst>::kX : E3Lay<kNst>::kH) + kb * 1024);
    o.sfLo = descLo(stage + kMoeUnitA);
    o.sfa = tmem + kMoeTmSfa + 16 * st;
    o.sfb = tmem + (fc == 0 ? kMoeTmSfbX : kMoeTmSfbH) + 8 * kb;
    o.rows = descLo(rowSrc);
    o.rows2 = descLo(rowSrc + 128);
    o.empty = emptyBar + 8 * st;
    // Materialise the words here (the asm is ordered before the waits that follow).
    asm volatile(
        ""
        : "+r"(o.aLo), "+r"(o.bLo), "+r"(o.sfLo), "+r"(o.sfa), "+r"(o.sfb), "+r"(o.rows), "+r"(o.rows2), "+r"(o.empty));
    return o;
}

// utccpRows8 from the low descriptor word of the source.
__device__ __forceinline__ void lxRows8Lo(uint32_t taddr, uint32_t srcLo)
{
    asm volatile(
        "{\n.reg .b64 d;\nmov.b64 d, {%1, %2};\n"
        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], d;\n}" ::"r"(taddr),
        "r"(srcLo), "r"(1u << 14)
        : "memory");
}

// One unit's tensor operations from its operands: the scale rows (rows: FC2 first use of the k-block, or FC1 needX
// unless noMma), then as issueUnitMmas.
__device__ __forceinline__ void lxIssue(
    LxOps const& o, uint32_t d, bool rows, bool first, bool noMma, bool noSfa, bool cpOnly)
{
    if (rows)
    {
        lxRows8Lo(o.sfb, o.rows);
        lxRows8Lo(o.sfb + 4, o.rows2);
    }
    if (noMma)
    {
        return;
    }
    if (!noSfa)
    {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
        {
            utccpLo(o.sfa + 4 * ks, o.sfLo + 32 * ks);
        }
    }
    if (!cpOnly)
    {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
        {
            mmaNvf4Lo(d, o.aLo + 2 * ks, o.bLo + 2 * ks, o.sfa + 4 * ks, o.sfb + 2 * ks, (first && ks == 0) ? 0u : 1u);
        }
    }
}

// mmaLoop (with kMoeFlagW2Fc2OneElect) from the table: the same waits, the same tensor operations and commits from
// the same elected lane, in the same order.
template <int kNst, bool kPx = false> // kPx: the stock queue
__device__ __noinline__ void mmaLoopLx(Ctx const& ctx, Sm const s0, Cut const c, uint32_t tmem0)
{
    int const lane = threadIdx.x & 31;
    Sm const s{lxBcast(s0.redS), lxBcast(s0.base)};
    uint32_t const tmem = lxBcast(tmem0);
    int const n = lxBcast(unitsOf(c));
    int const n1 = lxBcast(c.b[0] - c.a[0]);
    int64_t const flags = icMoeFlags(lxBcast(dw(s, kMdFlags))); // the build's fixed flag bits stay constants
    tcFenceAfter();
    uint32_t cpX = 0;
    if (flags & kMoeFlagE3XsfPre) // as mmaLoop
    {
        if (n1 > 0 && electOne())
        {
#pragma unroll 1
            for (int k = 0; k < kMoeKb; ++k)
            {
                utccpRows8(tmem + kMoeTmSfbX + 8 * k, s.redS + kMoeRdStgX + 256 * k);
                utccpRows8(tmem + kMoeTmSfbX + 8 * k + 4, s.redS + kMoeRdStgX + 256 * k + 128);
            }
        }
        __syncwarp();
        cpX = (1u << kMoeKb) - 1u;
    }
    bool const noMma = (flags & kMoeFlagDbgNoMma) != 0;
    bool const noSfa = (flags & kMoeFlagDbgNoSfa) != 0;
    bool const cpOnly = (flags & kMoeFlagDbgCopiesOnly) != 0;
    bool const pairs = !(flags & kMoeFlagNoPairs);
    bool const pairs2 = (flags & kMoeFlagE3Fc2Pairs) != 0;
    bool const md = (flags & kLxAttrMmaDelay) != 0; // attribution: delay injection
    uint32_t const fullBar = bar(s, kMoeBarFull);
    uint32_t const emptyBar = bar(s, kMoeBarEmpty);
    uint32_t const accFullBar = bar(s, kMoeBarAccFull);
    uint32_t const accEmptyBar = bar(s, kMoeBarAccEmpty);
    uint32_t const hReadyBar = bar(s, kMoeBarHReady);
    if (n1 == 0 && lane == 0)
    {
        mbarArrive(bar(s, kMoeBarFc1Done));
        mbarArrive(bar(s, kMoeBarFc1Cmpl)); // no FC1 units: complete at once
    }
    // The table window [base, base + 64): lane l holds the entries of units base + l and base + 32 + l.
    bool const xsfPre = cpX != 0;
    int base = 0;
    uint32_t wa = lxMmaEntry<kPx>(c, base + lane, n, n1, pairs, pairs2, xsfPre);
    uint32_t wb = lxMmaEntry<kPx>(c, base + 32 + lane, n, n1, pairs, pairs2, xsfPre);
    int ti = -1;
    int slotLoad = -1;
    int st = 0;
    uint32_t ph = 0;
#pragma unroll 1
    for (int j = 0; j < n;)
    {
        int i = j - base;
        if (i >= 63)
        {
            base = j;
            i = 0;
            wa = lxMmaEntry<kPx>(c, base + lane, n, n1, pairs, pairs2, xsfPre);
            wb = lxMmaEntry<kPx>(c, base + 32 + lane, n, n1, pairs, pairs2, xsfPre);
        }
        uint32_t const e1 = lxPick(wa, wb, i);
        uint32_t const e2 = lxPick(wa, wb, i + 1);
        int const kb = static_cast<int>(e1 & 15u);
        int const fc = (e1 & kLxFc) ? 1 : 0;
        bool const pair = (e1 & kLxPair) != 0;
        bool first = false;
        int hcp1 = -1, hcp2 = -1;
        if (e1 & kLxNew)
        {
            ++ti;
            first = true;
            if (ti >= kMoeAccs)
            {
                waitBar(ctx, accEmptyBar + 8 * (ti & (kMoeAccs - 1)), ((ti / kMoeAccs) - 1) & 1, kCodeMoeBar);
                tcFenceAfter();
            }
        }
        if (e1 & kLxSlot)
        {
            if (slotLoad >= 0)
            {
                if (electOne())
                {
                    commit(bar(s, kMoeBarHFree));
                }
                __syncwarp();
            }
            ++slotLoad;
        }
        if (e1 & kLxHWait)
        {
            waitBar(ctx, hReadyBar + 8 * kb, slotLoad & 1, kCodeMoeBar);
            if (j == n1 && slotLoad == 0 && lane == 0)
            {
                moeStamp(ctx, kMoeStFirstH);
            }
            tcFenceAfter();
            hcp1 = kb;
        }
        if (fc == 1 && lane == 0 && j - n1 < kE3TrUnits)
        {
            e3Tr(ctx, kE3TrHok + j - n1);
        }
        int const kb2 = kb + 1;
        int const st2 = st + 1 == kNst ? 0 : st + 1;
        uint32_t const ph2 = st + 1 == kNst ? ph ^ 1u : ph;
        int const acc = ti & (kMoeAccs - 1);
        uint32_t d = tmem + kMoeTmD + 8 * acc;
        uint32_t accBar = accFullBar + 8 * acc;
        asm volatile("" : "+r"(d), "+r"(accBar));
        LxOps const o1 = lxOps<kNst>(s, tmem, emptyBar, st, fc, kb);
        LxOps const o2 = lxOps<kNst>(s, tmem, emptyBar, st2, fc, kb2);
        waitBar(ctx, fullBar + 8 * st, ph, kCodeMoeBar);
        if (lane == 0)
        {
            if (j >= n1 && j - n1 < kE3TrUnits)
            {
                e3Tr(ctx, kE3TrFull + j - n1);
            }
            else if (j < n1 && j >= n1 - 8)
            {
                e3Tr(ctx, kE3TrFc1Full + j - (n1 - 8));
            }
        }
        if (pair && fc == 1)
        {
            if (e2 & kLxHWait)
            {
                waitBar(ctx, hReadyBar + 8 * kb2, slotLoad & 1, kCodeMoeBar);
                tcFenceAfter();
                hcp2 = kb2;
            }
            if (lane == 0 && j + 1 - n1 < kE3TrUnits)
            {
                e3Tr(ctx, kE3TrHok + j + 1 - n1);
            }
        }
        if (pair)
        {
            waitBar(ctx, fullBar + 8 * st2, ph2, kCodeMoeBar);
            if (lane == 0)
            {
                if (j + 1 >= n1 && j + 1 - n1 < kE3TrUnits)
                {
                    e3Tr(ctx, kE3TrFull + j + 1 - n1);
                }
                else if (j + 1 < n1 && j + 1 >= n1 - 8)
                {
                    e3Tr(ctx, kE3TrFc1Full + j + 1 - (n1 - 8));
                }
            }
        }
        int const jLast = pair ? j + 1 : j;
        if (lane == 0)
        {
            if (j == 0)
            {
                moeStamp(ctx, kMoeStFirstFull);
            }
            if (jLast == n1 - 1)
            {
                moeStamp(ctx, kMoeStFc1LastFull);
            }
            if (jLast == n - 1)
            {
                moeStamp(ctx, kMoeStLastFull);
            }
        }
        if (md)
        {
            lxSpinNs(pair ? 2 * kLxAttrDelayNs : kLxAttrDelayNs);
        }
        tcFenceAfter();
        bool const needX = (e1 & kLxNeedX) != 0;
        bool const needX2 = pair && (e2 & kLxNeedX) != 0;
        bool const lastOfTile = (e1 & kLxLast) != 0;
        bool const lastOfTile2 = pair && (e2 & kLxLast) != 0;
        bool const rows1 = fc ? hcp1 >= 0 : needX && !noMma;
        bool const rows2 = fc ? hcp2 >= 0 : needX2 && !noMma;
        if (electOne())
        {
            lxIssue(o1, d, rows1, first, noMma, noSfa, cpOnly);
            commit(o1.empty);
            if (lastOfTile)
            {
                commit(accBar);
            }
            if (pair)
            {
                lxIssue(o2, d, rows2, false, noMma, noSfa, cpOnly);
                commit(o2.empty);
                if (lastOfTile2)
                {
                    commit(accBar);
                }
            }
            if (jLast == n1 - 1)
            {
                commit(bar(s, kMoeBarFc1Cmpl)); // the CTA's FC1 MMAs complete
            }
        }
        __syncwarp();
        if (lane == 0)
        {
            if (j >= n1 && j - n1 < kE3TrUnits)
            {
                e3Tr(ctx, kE3TrIss + j - n1);
            }
            if (pair && j + 1 >= n1 && j + 1 - n1 < kE3TrUnits)
            {
                e3Tr(ctx, kE3TrIss + j + 1 - n1);
            }
            if (j == n1)
            {
                moeStamp(ctx, kMoeStFc2First);
            }
            if (jLast == n1 - 1)
            {
                moeStamp(ctx, kMoeStFc1Issued);
                mbarArrive(bar(s, kMoeBarFc1Done));
            }
        }
        if (pair)
        {
            j += 2;
            st = st2 + 1 == kNst ? 0 : st2 + 1;
            ph = st2 + 1 == kNst ? ph2 ^ 1u : ph2;
        }
        else
        {
            j += 1;
            st = st2;
            ph = ph2;
        }
    }
    if (lane == 0)
    {
        moeStamp(ctx, kMoeStMmaDone);
    }
}

// ---- end table-driven issue loops ----

// ---- kMoeFlagPxMma: the MMA warp of the stock queue in the stock kernel's loop shape --------------------------------
// The CTA's whole tiles in order; per tile its accumulator, then its 16 k-blocks in pairs from running ring counters:
// the pair's operands are built first, then the FC2 h waits, the two full waits and one elected block that issues
// [h scale rows] 4 SFA copies + 4 MMAs + the empty commit per unit and the accumulator commit after k-block 15. The FC1
// token scales of all 16 k-blocks go to TMEM once at the start (every FC1 tile reads all of them). No table, no cut
// decode: the queue gives every tile 16 k-blocks, every FC2 tile a new slot. The tensor operations of a unit and their
// accumulators are mmaLoop's; the token-scale rows are the same bytes in the same TMEM columns, copied earlier.
template <int kNst>
__device__ __noinline__ void mmaLoopPx(Ctx const& ctx, Sm const s0, Cut const c, uint32_t tmem0)
{
    int const lane = threadIdx.x & 31;
    Sm const s{lxBcast(s0.redS), lxBcast(s0.base)};
    uint32_t const tmem = lxBcast(tmem0);
    int const nT = lxBcast(unitsOf(c)) / kMoeKb;
    int const nT1 = lxBcast(c.b[0] - c.a[0]) / kMoeKb;
    int64_t const flags = icMoeFlags(lxBcast(dw(s, kMdFlags))); // the build's fixed flag bits stay constants
    bool const noMma = (flags & kMoeFlagDbgNoMma) != 0;
    bool const noSfa = (flags & kMoeFlagDbgNoSfa) != 0;
    bool const cpOnly = (flags & kMoeFlagDbgCopiesOnly) != 0;
    uint32_t const fullBar = bar(s, kMoeBarFull);
    uint32_t const emptyBar = bar(s, kMoeBarEmpty);
    uint32_t const accFullBar = bar(s, kMoeBarAccFull);
    uint32_t const accEmptyBar = bar(s, kMoeBarAccEmpty);
    uint32_t const hReadyBar = bar(s, kMoeBarHReady);
    uint32_t const ring = s.base + kMoeSmRing;
    tcFenceAfter();
    if (nT1 == 0 && lane == 0)
    {
        mbarArrive(bar(s, kMoeBarFc1Done));
        mbarArrive(bar(s, kMoeBarFc1Cmpl));
    }
    if (nT1 > 0 && !noMma)
    {
        if (electOne())
        {
#pragma unroll 1
            for (int k = 0; k < kMoeKb; ++k)
            {
                utccpRows8(tmem + kMoeTmSfbX + 8 * k, s.redS + kMoeRdStgX + 256 * k);
                utccpRows8(tmem + kMoeTmSfbX + 8 * k + 4, s.redS + kMoeRdStgX + 256 * k + 128);
            }
        }
        __syncwarp();
    }
    int st = 0;      // ring stage of the next unit
    uint32_t ph = 0; // its full-barrier parity
    int slotLoad = -1;
#pragma unroll 1
    for (int t = 0; t < nT; ++t)
    {
        bool const fc2 = t >= nT1;
        int const acc = t & (kMoeAccs - 1);
        if constexpr (kUeLastTile) // MSO_MK_UE bit 2: the h loader's start at the CTA's last FC1 tile (two or more)
        {
            if (lane == 0 && nT1 >= 2 && t == nT1 - 1)
            {
                mbarArrive(bar(s, kMoeBarFc1Done));
            }
        }
        if (t >= kMoeAccs)
        {
            waitBar(ctx, accEmptyBar + 8 * acc, ((t / kMoeAccs) - 1) & 1, kCodeMoeBar);
            tcFenceAfter();
        }
        if (fc2)
        {
            if (slotLoad >= 0)
            {
                if (electOne())
                {
                    commit(bar(s, kMoeBarHFree));
                }
                __syncwarp();
            }
            ++slotLoad;
        }
        uint32_t const hPar = static_cast<uint32_t>(slotLoad) & 1u;
        uint32_t const d = tmem + kMoeTmD + 8 * acc;
        uint32_t const accBar = accFullBar + 8 * acc;
        uint32_t const bLo0 = kDescSw128Lo | descLo(s.base + (fc2 ? E3Lay<kNst>::kH : E3Lay<kNst>::kX));
        uint32_t const sfb0 = tmem + (fc2 ? kMoeTmSfbH : kMoeTmSfbX);
        uint32_t const rows0 = descLo(s.redS + kMoeRdStgH);
        // ---- MSO_MK_UE bit 8: one k-block per elected block (the pair loop's operands, waits, tensor operations and
        // commits for one unit; the same units in the same order) ----
        if constexpr (kUeSingle)
        {
#pragma unroll 1
            for (int kb = 0; kb < kMoeKb; ++kb)
            {
                uint32_t const stg = ring + st * kMoeUnitBytes;
                uint32_t aLo = kDescSw128Lo | descLo(stg);
                uint32_t sfLo = descLo(stg + kMoeUnitA);
                uint32_t sfa = tmem + kMoeTmSfa + 16 * st;
                uint32_t bLo = bLo0 + 64 * kb;
                uint32_t sfb = sfb0 + 8 * kb;
                uint32_t r = rows0 + 16 * kb;
                uint32_t e = emptyBar + 8 * st;
                asm volatile("" : "+r"(aLo), "+r"(sfLo), "+r"(sfa), "+r"(bLo), "+r"(sfb), "+r"(r), "+r"(e));
                bool const ueTr = lane == 0 && (t == nT1 || (t == nT1 - 1 && kb >= kMoeKb - 8));
                if (fc2)
                {
                    waitBar(ctx, hReadyBar + 8 * kb, hPar, kCodeMoeBar);
                    if (ueTr)
                    {
                        e3Tr(ctx, kE3TrHok + kb);
                    }
                    if (MSO_MK_STAMPS_DETAIL(ctx) != nullptr && lane == 0 && kb == 0 && slotLoad == 0)
                    {
                        moeStamp(ctx, kMoeStFirstH);
                    }
                }
                waitBar(ctx, fullBar + 8 * st, ph, kCodeMoeBar);
                if (ueTr)
                {
                    e3Tr(ctx, fc2 ? kE3TrFull + kb : kE3TrFc1Full + kb - (kMoeKb - 8));
                }
                if (MSO_MK_STAMPS_DETAIL(ctx) != nullptr && lane == 0)
                {
                    if (t == 0 && kb == 0)
                    {
                        moeStamp(ctx, kMoeStFirstFull);
                    }
                    if (kb == kMoeKb - 1 && t == nT1 - 1)
                    {
                        moeStamp(ctx, kMoeStFc1LastFull);
                    }
                    if (kb == kMoeKb - 1 && t == nT - 1)
                    {
                        moeStamp(ctx, kMoeStLastFull);
                    }
                }
                tcFenceAfter();
                if (electOne())
                {
                    if (fc2)
                    {
                        lxRows8Lo(sfb, r);
                        lxRows8Lo(sfb + 4, r + 8);
                    }
                    if (!noMma)
                    {
                        if (!noSfa)
                        {
#pragma unroll
                            for (int ks = 0; ks < 4; ++ks)
                            {
                                utccpLo(sfa + 4 * ks, sfLo + 32 * ks);
                            }
                        }
                        if (!cpOnly)
                        {
#pragma unroll
                            for (int ks = 0; ks < 4; ++ks)
                            {
                                mmaNvf4Lo(d, aLo + 2 * ks, bLo + 2 * ks, sfa + 4 * ks, sfb + 2 * ks,
                                    (kb == 0 && ks == 0) ? 0u : 1u);
                            }
                        }
                    }
                    commit(e);
                    if (kb == kMoeKb - 1)
                    {
                        commit(accBar);
                        if (t == nT1 - 1)
                        {
                            commit(bar(s, kMoeBarFc1Cmpl));
                        }
                    }
                }
                __syncwarp();
                if (ueTr && fc2)
                {
                    e3Tr(ctx, kE3TrIss + kb);
                }
                if (++st == kNst)
                {
                    st = 0;
                    ph ^= 1u;
                }
            }
        }
        else
        { // ---- end MSO_MK_UE bit 8 ----
#pragma unroll 1
            for (int kb = 0; kb < kMoeKb; kb += 2)
            {
                int const st2 = st + 1 == kNst ? 0 : st + 1;
                uint32_t const ph2 = st + 1 == kNst ? ph ^ 1u : ph;
                // The pair's operands, ahead of its waits (descLo of a 1 KiB / 256 B step is + 64 / + 16).
                uint32_t const stg1 = ring + st * kMoeUnitBytes;
                uint32_t const stg2 = ring + st2 * kMoeUnitBytes;
                uint32_t aLo1 = kDescSw128Lo | descLo(stg1), aLo2 = kDescSw128Lo | descLo(stg2);
                uint32_t sfLo1 = descLo(stg1 + kMoeUnitA), sfLo2 = descLo(stg2 + kMoeUnitA);
                uint32_t sfa1 = tmem + kMoeTmSfa + 16 * st, sfa2 = tmem + kMoeTmSfa + 16 * st2;
                uint32_t bLo1 = bLo0 + 64 * kb, bLo2 = bLo0 + 64 * (kb + 1);
                uint32_t sfb1 = sfb0 + 8 * kb, sfb2 = sfb0 + 8 * (kb + 1);
                uint32_t r1 = rows0 + 16 * kb, r2 = rows0 + 16 * (kb + 1);
                uint32_t e1 = emptyBar + 8 * st, e2 = emptyBar + 8 * st2;
                asm volatile(""
                             : "+r"(aLo1), "+r"(aLo2), "+r"(sfLo1), "+r"(sfLo2), "+r"(sfa1), "+r"(sfa2), "+r"(bLo1),
                             "+r"(bLo2), "+r"(sfb1), "+r"(sfb2), "+r"(r1), "+r"(r2), "+r"(e1), "+r"(e2));
                // The per-unit stamp words (e3Tr) of the CTA's first FC2 tile (h in, weights in, issued) and of its
                // last 8 FC1 units (weights in); stamped builds only (e3Tr is empty without stamps).
                bool const ueTr = lane == 0 && (t == nT1 || (t == nT1 - 1 && kb >= kMoeKb - 8));
                if (fc2)
                {
                    waitBar(ctx, hReadyBar + 8 * kb, hPar, kCodeMoeBar);
                    if (ueTr)
                    {
                        e3Tr(ctx, kE3TrHok + kb);
                    }
                    waitBar(ctx, hReadyBar + 8 * (kb + 1), hPar, kCodeMoeBar);
                    if (ueTr)
                    {
                        e3Tr(ctx, kE3TrHok + kb + 1);
                    }
                    if (MSO_MK_STAMPS_DETAIL(ctx) != nullptr && lane == 0 && kb == 0 && slotLoad == 0)
                    {
                        moeStamp(ctx, kMoeStFirstH);
                    }
                }
                waitBar(ctx, fullBar + 8 * st, ph, kCodeMoeBar);
                if (ueTr)
                {
                    e3Tr(ctx, fc2 ? kE3TrFull + kb : kE3TrFc1Full + kb - (kMoeKb - 8));
                }
                waitBar(ctx, fullBar + 8 * st2, ph2, kCodeMoeBar);
                if (ueTr)
                {
                    e3Tr(ctx, fc2 ? kE3TrFull + kb + 1 : kE3TrFc1Full + kb + 1 - (kMoeKb - 8));
                }
                if (MSO_MK_STAMPS_DETAIL(ctx) != nullptr && lane == 0)
                {
                    if (t == 0 && kb == 0)
                    {
                        moeStamp(ctx, kMoeStFirstFull);
                    }
                    if (kb == kMoeKb - 2 && t == nT1 - 1)
                    {
                        moeStamp(ctx, kMoeStFc1LastFull);
                    }
                    if (kb == kMoeKb - 2 && t == nT - 1)
                    {
                        moeStamp(ctx, kMoeStLastFull);
                    }
                }
                tcFenceAfter();
                if (electOne())
                {
                    if (fc2)
                    {
                        lxRows8Lo(sfb1, r1);
                        lxRows8Lo(sfb1 + 4, r1 + 8);
                    }
                    if (!noMma)
                    {
                        if (!noSfa)
                        {
#pragma unroll
                            for (int ks = 0; ks < 4; ++ks)
                            {
                                utccpLo(sfa1 + 4 * ks, sfLo1 + 32 * ks);
                            }
                        }
                        if (!cpOnly)
                        {
#pragma unroll
                            for (int ks = 0; ks < 4; ++ks)
                            {
                                mmaNvf4Lo(d, aLo1 + 2 * ks, bLo1 + 2 * ks, sfa1 + 4 * ks, sfb1 + 2 * ks,
                                    (kb == 0 && ks == 0) ? 0u : 1u);
                            }
                        }
                    }
                    commit(e1);
                    if (fc2)
                    {
                        lxRows8Lo(sfb2, r2);
                        lxRows8Lo(sfb2 + 4, r2 + 8);
                    }
                    if (!noMma)
                    {
                        if (!noSfa)
                        {
#pragma unroll
                            for (int ks = 0; ks < 4; ++ks)
                            {
                                utccpLo(sfa2 + 4 * ks, sfLo2 + 32 * ks);
                            }
                        }
                        if (!cpOnly)
                        {
#pragma unroll
                            for (int ks = 0; ks < 4; ++ks)
                            {
                                mmaNvf4Lo(d, aLo2 + 2 * ks, bLo2 + 2 * ks, sfa2 + 4 * ks, sfb2 + 2 * ks, 1u);
                            }
                        }
                    }
                    commit(e2);
                    if (kb == kMoeKb - 2)
                    {
                        commit(accBar);
                        if (t == nT1 - 1)
                        {
                            commit(bar(s, kMoeBarFc1Cmpl));
                        }
                    }
                }
                __syncwarp();
                if (ueTr && fc2) // the pair is issued
                {
                    e3Tr(ctx, kE3TrIss + kb);
                    e3Tr(ctx, kE3TrIss + kb + 1);
                }
                st = st2 + 1 == kNst ? 0 : st2 + 1;
                ph = st2 + 1 == kNst ? ph2 ^ 1u : ph2;
            }
        } // the pair loop
        if (lane == 0)
        {
            if (t == nT1 - 1)
            {
                moeStamp(ctx, kMoeStFc1Issued);
                if (!(kUeLastTile && nT1 >= 2)) // MSO_MK_UE bit 2: arrived at the tile's start
                {
                    mbarArrive(bar(s, kMoeBarFc1Done));
                }
            }
            if (t == nT1)
            {
                moeStamp(ctx, kMoeStFc2First);
            }
        }
    }
    if (lane == 0)
    {
        moeStamp(ctx, kMoeStMmaDone);
    }
}

// ---- end kMoeFlagPxMma ----

// ---- kMoeFlagE3HWide: one slot's h, every needed k-block in flight at once ------------------------------------------
// Thread t64 of warps 2-3 loads the 16 B code chunk (token t64 / 8, chunk t64 % 8) of every needed k-block, and the
// scale vectors (token t64 % 8) of k-blocks t64 / 8 and t64 / 8 + 8. Each warp stores and releases a k-block (one
// arrival on its HReady barrier) as soon as its lanes have their words, walking the k-blocks in first-use order
// (s0 .. 15, then 0 .. s0 - 1) and re-polling only the ones still empty. The stores, the scale staging and the barrier
// protocol are the narrow loader's.
template <int kK>
__device__ __forceinline__ void e3WideStep(Ctx const& ctx, Sm const& s, uint4 (&v)[kMoeKb], uint4 (&sv)[2],
    uint32_t& pending, uint32_t hBase, uint8_t const* hd, uint8_t const* hs, int t64, bool trace)
{
    if (!((pending >> kK) & 1u))
    {
        return;
    }
    bool const own = (t64 >> 3) == (kK & 7);
    bool const ok = full4(v[kK]) && (!own || full4(sv[kK >> 3]));
    if (warpAll(ok))
    {
        int const r = t64 >> 3;
        int const c16 = t64 & 7;
        sts128(s.base + hBase + kK * 1024 + r * 128 + ((c16 ^ r) << 4), v[kK]);
        if (own)
        {
            int const tok = t64 & 7;
            uint4 const w = sv[kK >> 3];
            sts128(s.redS + kMoeRdStgH + 256 * kK + 16 * tok, make_uint4(w.x, 0u, w.y, 0u));
            sts128(s.redS + kMoeRdStgH + 256 * kK + 128 + 16 * tok, make_uint4(w.z, 0u, w.w, 0u));
        }
        fenceProxyAsyncSmem();
        __syncwarp();
        if ((threadIdx.x & 31) == 0)
        {
            mbarArrive(bar(s, kMoeBarHReady + kK));
            if (trace)
            {
                e3Tr(ctx, kE3TrHkb + kK);
            }
        }
        pending &= ~(1u << kK);
        return;
    }
    if (!full4(v[kK]))
    {
        v[kK] = ldPoll128(hd + kK * (kTok * 128));
    }
    if (own && !full4(sv[kK >> 3]))
    {
        sv[kK >> 3] = ldPoll128(hs + kK * (kTok * 16));
    }
}

// The steps of the k-blocks kb with lo <= kb < hi (compile-time indices, run-time range).
template <int kK>
__device__ __forceinline__ void e3WideRange(Ctx const& ctx, Sm const& s, uint4 (&v)[kMoeKb], uint4 (&sv)[2],
    uint32_t& pending, uint32_t hBase, uint8_t const* hd, uint8_t const* hs, int t64, bool trace, int lo, int hi)
{
    if constexpr (kK < kMoeKb)
    {
        if (kK >= lo && kK < hi)
        {
            e3WideStep<kK>(ctx, s, v, sv, pending, hBase, hd, hs, t64, trace);
        }
        e3WideRange<kK + 1>(ctx, s, v, sv, pending, hBase, hd, hs, t64, trace, lo, hi);
    }
}

// hd0 / hs0: the slot's h codes / scales (k-block 0, token 0); need: its k-blocks; s0: the first one used; waitCmpl:
// the h area is FC1's B operand until the CTA's FC1 MMAs are complete (kMoeFlagE3Ring10).
__device__ __forceinline__ void e3LoadSlotWide(Ctx const& ctx, Sm const& s, uint32_t hBase, uint8_t const* hd0,
    uint8_t const* hs0, uint32_t need, int s0, bool trace, bool waitCmpl)
{
    int const t64 = threadIdx.x - 64;
    uint8_t const* hd = hd0 + 16 * t64;
    uint8_t const* hs = hs0 + 16 * (t64 & 7);
    uint4 v[kMoeKb];
    uint4 sv[2];
#pragma unroll
    for (int k = 0; k < kMoeKb; ++k)
    {
        v[k] = ((need >> k) & 1u) ? ldPoll128(hd + k * (kTok * 128)) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int h = 0; h < 2; ++h)
    {
        int const k = (t64 >> 3) + 8 * h;
        sv[h] = ((need >> k) & 1u) ? ldPoll128(hs + k * (kTok * 16)) : make_uint4(0, 0, 0, 0);
    }
    if (waitCmpl)
    {
        waitBar(ctx, bar(s, kMoeBarFc1Cmpl), 0, kCodeMoeBar);
    }
    uint32_t pending = need;
    uint64_t t0 = 0;
    while (pending != 0)
    {
        e3WideRange<0>(ctx, s, v, sv, pending, hBase, hd, hs, t64, trace, s0, kMoeKb);
        e3WideRange<0>(ctx, s, v, sv, pending, hBase, hd, hs, t64, trace, 0, s0);
        if (pending != 0)
        {
            spinCheck(ctx, t0, kCodeMoeH, __popc(pending), 0);
        }
    }
}

// ---- end kMoeFlagE3HWide ----

// ---- 5c. warps 2-3: clear the next copy's workspace slice, then fetch h of the CTA's FC2 slots
// ----------------------------
template <int kNst, bool kPx = false> // kNst: the ring layout; kPx: the stock queue
__device__ __noinline__ void loaderLoop(Ctx const& ctx, Sm const s, Cut const c, uint8_t const* wsCur, uint8_t* wsNext)
{
    bool const hWide = (dw(s, kMdFlags) & kMoeFlagE3HWide) != 0; // every needed k-block at once
    constexpr bool kShared = E3Lay<kNst>::kX == E3Lay<kNst>::kH; // X and h share one area (kMoeFlagE3Ring10)
    int const t64 = threadIdx.x - 64;
    {
        constexpr int kGranules = kMoeWsBytes / 16;
        int const g0 = static_cast<int>((static_cast<int64_t>(kGranules) * blockIdx.x) / kCtas);
        int const g1 = static_cast<int>((static_cast<int64_t>(kGranules) * (blockIdx.x + 1)) / kCtas);
        uint64_t const policy = policyEvictFirst();
#if MSO_MK_CF & 2
#pragma unroll 1 // MSO_MK_CF bit 2: the same stores in the same order, one copy of the loop body
#endif
        for (int g = g0 + t64; g < g1; g += 64)
        {
            st128Hint(wsNext + 16 * static_cast<size_t>(g), emptyVec(), policy);
        }
    }
    int const n1 = c.b[0] - c.a[0];
    int const n = unitsOf(c);
    int const lane = threadIdx.x & 31;
    int const r = t64 >> 3;
    int const c16 = t64 & 7;
    // h cannot be complete before this CTA's own FC1 MMAs are issued (every CTA's FC1 ends at about the same time):
    // no polling traffic during the FC1 stream (kMoeFlagLoaderEarly: poll from the start).
    if (n > n1 && !(kPx && kUeLoaderEarly) && !(dw(s, kMdFlags) & kMoeFlagLoaderEarly)) // MSO_MK_UE bit 1
    {
        waitBarIdle(ctx, bar(s, kMoeBarFc1Done), 0, kCodeMoeBar, (dw(s, kMdFlags) & kMoeFlagSpinSleep) ? 256u : 0u);
    }
    int si = -1;
    int j = n1;
#pragma unroll 1
    while (j < n)
    {
        int fc, u, slot, mt, kb;
        unitAt<kPx>(c, j, fc, u);
        decodeUnit(fc, u, slot, mt, kb);
        ++si;
        if (si > 0)
        {
            // The previous slot's MMAs have read its operand and scales.
            waitBar(ctx, bar(s, kMoeBarHFree), (si - 1) & 1, kCodeMoeBar);
        }
        // The slot's k-blocks in the order of their first use (4 bits each), and where the slot's units end.
        uint32_t need = 0;
        uint64_t order = 0;
        int cnt = 0;
        int jEnd = j;
#if MSO_MK_CF & 4
        constexpr bool kCfScan = !kPx || !MSO_MK_LEAN; // MSO_MK_CF bit 4: kPx's closed form (check builds verify it)
#else
        constexpr bool kCfScan = true;
#endif
        if constexpr (kCfScan)
        {
#pragma unroll 1
            for (; jEnd < n; ++jEnd)
            {
                int f2, u2, sl2, mt2, kb2;
                unitAt<kPx>(c, jEnd, f2, u2);
                decodeUnit(f2, u2, sl2, mt2, kb2);
                if (sl2 != slot)
                {
                    break;
                }
                if (!((need >> kb2) & 1u))
                {
                    need |= 1u << kb2;
                    order |= static_cast<uint64_t>(kb2) << (4 * cnt);
                    ++cnt;
                }
            }
        }
#if MSO_MK_CF & 4
        if constexpr (kPx)
        {
            // MSO_MK_CF bit 4: a stock-queue FC2 tile is one slot, k-blocks 0 .. 15 in order (j is a multiple of 16,
            // n - j >= 16)
            constexpr uint32_t kCfNeed = (1u << kMoeKb) - 1u;
            constexpr uint64_t kCfOrder = 0xfedcba9876543210ull;
            static_assert(kMoeKb == 16, "cf bit 4: 16 k-blocks of 4 bits fill the order word");
            if constexpr (kCfScan)
            {
                if (need != kCfNeed || order != kCfOrder || cnt != kMoeKb || jEnd != j + kMoeKb)
                {
                    __trap(); // MSO_MK_CF bit 4: the closed form differs from the scan (check builds)
                }
            }
            need = kCfNeed;
            order = kCfOrder;
            cnt = kMoeKb;
            jEnd = j + kMoeKb;
        }
#endif
        // The k-blocks this slot never uses complete their phase now (each barrier advances once per slot; two
        // arrivals, one per loader warp).
        if (lane < kMoeKb && !((need >> lane) & 1u))
        {
            mbarArrive(bar(s, kMoeBarHReady + lane));
        }
        // ---- kMoeFlagE3HWide ----
        if (hWide)
        {
            e3LoadSlotWide(ctx, s, E3Lay<kNst>::kH, wsCur + kMoeWsH + (static_cast<size_t>(slot) * kMoeKb * kTok) * 128,
                wsCur + kMoeWsHsf + (static_cast<size_t>(slot) * kMoeKb * kTok) * 16, need,
                static_cast<int>(order & 15u), si == 0 && t64 < 32, kShared && si == 0);
            j = jEnd;
            continue;
        }
        if (kShared && si == 0)
        {
            waitBar(ctx, bar(s, kMoeBarFc1Cmpl), 0, kCodeMoeBar);
        }
        // ---- end kMoeFlagE3HWide ----
        uint8_t const* hd = wsCur + kMoeWsH + (static_cast<size_t>(slot) * kMoeKb * kTok) * 128 + 16 * t64;
        uint8_t const* hs = wsCur + kMoeWsHsf + (static_cast<size_t>(slot) * kMoeKb * kTok) * 16 + 16 * t64;
        // Four k-blocks in flight (per thread: token t64 / 8, chunk t64 % 8 of each, and the scale vector of token
        // t64 < 8); this warp stores and releases each one as soon as its 32 lanes have it.
#pragma unroll 1
        for (int p = 0; p < cnt; p += 4)
        {
            int kbs[4];
            uint4 v[4][2];
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                kbs[i] = p + i < cnt ? static_cast<int>((order >> (4 * (p + i))) & 15u) : -1;
                v[i][0] = kbs[i] >= 0 ? ldPoll128(hd + kbs[i] * (kTok * 128)) : make_uint4(0, 0, 0, 0);
                v[i][1] = kbs[i] >= 0 && t64 < kTok ? ldPoll128(hs + kbs[i] * (kTok * 16)) : make_uint4(0, 0, 0, 0);
            }
#if MSO_MK_CF & 1
            // MSO_MK_CF bit 1: one rolled body for the group's k-blocks; k-block p + i polls and stores through w (its
            // two load registers selected from the four), the statements of the unrolled body below.
#pragma unroll 1
            for (int i = 0; i < 4 && p + i < cnt; ++i)
            {
                int const k = static_cast<int>((order >> (4 * (p + i))) & 15u);
                uint4 w[2];
#pragma unroll
                for (int h = 0; h < 2; ++h)
                {
                    w[h] = i == 0 ? v[0][h] : i == 1 ? v[1][h] : i == 2 ? v[2][h] : v[3][h];
                }
                uint64_t t0 = 0;
                while (!warpAll(full4(w[0]) && full4(w[1])))
                {
                    spinCheck(ctx, t0, kCodeMoeH, k, 0);
                    if (!full4(w[0]))
                    {
                        w[0] = ldPoll128(hd + k * (kTok * 128));
                    }
                    if (t64 < kTok && !full4(w[1]))
                    {
                        w[1] = ldPoll128(hs + k * (kTok * 16));
                    }
                }
                sts128(s.base + E3Lay<kNst>::kH + k * 1024 + r * 128 + ((c16 ^ r) << 4), w[0]); // the h area
                if (t64 < kTok)
                {
                    sts128(s.redS + kMoeRdStgH + 256 * k + 16 * t64, make_uint4(w[1].x, 0u, w[1].y, 0u));
                    sts128(s.redS + kMoeRdStgH + 256 * k + 128 + 16 * t64, make_uint4(w[1].z, 0u, w[1].w, 0u));
                }
                fenceProxyAsyncSmem();
                __syncwarp();
                if (lane == 0)
                {
                    mbarArrive(bar(s, kMoeBarHReady + k));
                    if (si == 0 && t64 < 32)
                    {
                        e3Tr(ctx, kE3TrHkb + k); // per-unit stamp
                    }
                }
            }
#else
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                if (kbs[i] < 0)
                {
                    break;
                }
                int const k = kbs[i];
                uint64_t t0 = 0;
                while (!warpAll(full4(v[i][0]) && full4(v[i][1])))
                {
                    spinCheck(ctx, t0, kCodeMoeH, k, 0);
                    if (!full4(v[i][0]))
                    {
                        v[i][0] = ldPoll128(hd + k * (kTok * 128));
                    }
                    if (t64 < kTok && !full4(v[i][1]))
                    {
                        v[i][1] = ldPoll128(hs + k * (kTok * 16));
                    }
                }
                sts128(s.base + E3Lay<kNst>::kH + k * 1024 + r * 128 + ((c16 ^ r) << 4), v[i][0]); // the h area
                if (t64 < kTok)
                {
                    // pairs 2 kb and 2 kb + 1, row t64: (step 4 kb, 0, step 4 kb + 1, 0), (step 4 kb + 2, 0, ...)
                    sts128(s.redS + kMoeRdStgH + 256 * k + 16 * t64, make_uint4(v[i][1].x, 0u, v[i][1].y, 0u));
                    sts128(s.redS + kMoeRdStgH + 256 * k + 128 + 16 * t64, make_uint4(v[i][1].z, 0u, v[i][1].w, 0u));
                }
                fenceProxyAsyncSmem();
                __syncwarp();
                if (lane == 0)
                {
                    mbarArrive(bar(s, kMoeBarHReady + k));
                    if (si == 0 && t64 < 32)
                    {
                        e3Tr(ctx, kE3TrHkb + k); // per-unit stamp
                    }
                }
            }
#endif
        }
        j = jEnd;
    }
    if (t64 == 0 && n > n1)
    {
        moeStamp(ctx, kMoeStHAll);
    }
}

// ---- 5d. the epilogue (warps 4-7: TMEM lane quadrant q = warp - 4, row r = 32 q + lane)
// ------------------------------------ FC1 epilogue of a finished tile: TRTLLM-gen's SwiGLU + NVFP4 requantization
// (persistent_experts _swiglu_quant_to_stage, operation for operation) into the staging area, then the tile's h words
// into the workspace.
__device__ __forceinline__ void fc1Epilogue(Sm const& s, float const (&v)[8], int slot, int mt, uint8_t* wsCur)
{
    int const q = (threadIdx.x >> 5) - 4;
    int const lane = threadIdx.x & 31;
    int const e = threadIdx.x - 128;
    uint32_t const mask = rd<uint32_t const>(s, kMoeRdSlotMask)[slot];
    float const sG = rd<float const>(s, kMoeRdSlotScal)[slot];
    float const sC = rd<float const>(s, kMoeRdSlotScal + 256)[slot];
    uint8_t* hst = rd<uint8_t>(s, kMoeRdEpi);
    bool const isUp = (lane & 15) < 8;
    float const tNeg = mulFtz(sG, -kLog2e);
#pragma unroll
    for (int nTok = 0; nTok < kTok; ++nTok)
    {
        float const g = __shfl_down_sync(0xffffffffu, v[nTok], 8);
        float const x = mulRn(g, tNeg);
        float const d = addRn(ex2Ftz(x), 1.f);
        float const gsig = mulRn(g, rcpFtz(d));
        float const up = fmaRnZero(v[nTok], sG);
        float const hh = mulRn(up, gsig);
        float const h = isUp ? hh : 0.f;
        float am = fabsf(h);
        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4));
        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 16));
        float const qs = mulRn(mulRn(am, sC), kRcp6);
        uint32_t sf = cvtE4m3(qs);
        float const sfF = e4m3ToF32(sf);
        float const osc = sfF != 0.f ? rcpFtz(sfF) : 0.f;
        float const partner = __shfl_down_sync(0xffffffffu, h, 16);
        uint32_t byte = cvtE2m1x2(mulRn(mulFtz(h, osc), sC), mulRn(mulFtz(partner, osc), sC));
        if (!((mask >> nTok) & 1u))
        {
            byte = 0;
            sf = 0;
        }
        if (lane < 8)
        {
            hst[nTok * 32 + q * 8 + lane] = static_cast<uint8_t>(byte);
        }
        if (lane == 0)
        {
            hst[256 + nTok * 4 + q] = static_cast<uint8_t>(sf);
        }
    }
    namedSync(1, 128);
    // 64 code words (token e / 8, word e % 8) and 8 scale words, each a Lamport word.
    int const kb = mt >> 2;
    int const qq = mt & 3;
    if (e < 64)
    {
        uint32_t const w = *reinterpret_cast<uint32_t const*>(hst + (e >> 3) * 32 + 4 * (e & 7));
        st32(wsCur + kMoeWsH + ((static_cast<size_t>(slot) * kMoeKb + kb) * kTok + (e >> 3)) * 128 + qq * 32
                + 4 * (e & 7),
            lamport(w));
    }
    else if (e < 72)
    {
        int const t = e - 64;
        uint32_t const w = *reinterpret_cast<uint32_t const*>(hst + 256 + 4 * t);
        st32(wsCur + kMoeWsHsf + ((static_cast<size_t>(slot) * kMoeKb + kb) * kTok + t) * 16 + qq * 4, lamport(w));
    }
    namedSync(1, 128);
}

// FC2 epilogue of a finished tile: rows y = bf16(acc fc2_alpha) of the routed tokens and their per-tile sums of
// (y (1 + w_post))^2.
__device__ __forceinline__ float fc2Gain(Sm const& s, int mt)
{
    int const q = (threadIdx.x >> 5) - 4;
    int const lane = threadIdx.x & 31;
    int const col = mt * 128 + 32 * q + 4 * (lane & 7) + (lane >> 3);
    return addRn(__bfloat162float(dptr<__nv_bfloat16 const>(s, kMdGainPost)[col]), 1.f);
}

// gainIn = fc2Gain(s, mt), loaded by the caller (kMoeFlagE3EpiPre: before the tile's accumulator wait).
__device__ __forceinline__ void fc2Epilogue(
    Sm const& s, float const (&v)[8], int slot, int mt, uint8_t* wsCur, float gainIn)
{
    int const q = (threadIdx.x >> 5) - 4;
    int const lane = threadIdx.x & 31;
    int const e = threadIdx.x - 128;
    uint32_t const mask = rd<uint32_t const>(s, kMoeRdSlotMask)[slot];
    int8_t const* slotK = rd<int8_t const>(s, kMoeRdSlotK) + slot * kTok;
    float const a2 = rd<float const>(s, kMoeRdSlotScal + 512)[slot];
    int const col = mt * 128 + 32 * q + 4 * (lane & 7) + (lane >> 3);
    float const gain = gainIn; // fc2Gain(s, mt)
    __nv_bfloat16* rowsDbg = dptr<__nv_bfloat16>(s, kMdRows);
    float* sq = rd<float>(s, kMoeRdEpiSq);
#pragma unroll
    for (int nTok = 0; nTok < kTok; ++nTok)
    {
        bool const routed = (mask >> nTok) & 1u;
        uint32_t const yb = bf16Bits(mulRn(v[nTok], a2));
        int const tk = nTok * kMoeTopK + (routed ? slotK[nTok] : 0);
        // Lane l < 8 gathers columns 4 l .. 4 l + 3 of the warp's 32 (lanes l, l + 8, l + 16, l + 24).
        uint32_t const y1 = __shfl_down_sync(0xffffffffu, yb, 8);
        uint32_t const y2 = __shfl_down_sync(0xffffffffu, yb, 16);
        uint32_t const y3 = __shfl_down_sync(0xffffffffu, yb, 24);
        if (routed)
        {
            if (lane < 8)
            {
                uint8_t* dst = wsCur + kMoeWsY + (static_cast<size_t>(tk) * kMoeHidden + col) * 2;
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
        st32(wsCur + kMoeWsSsq + (static_cast<size_t>(tk) * kMoeFc2Tiles + mt) * 4, lamport(__float_as_uint(tot)));
    }
    namedSync(1, 128);
}

#if MSO_MK_ET & 1
// ---- MSO_MK_ET bit 1: stores under a predicate (no branch region around them) ----
__device__ __forceinline__ void st64If(bool p, void* ptr, uint2 v)
{
    asm volatile("{\n\t.reg .pred q;\n\tsetp.ne.b32 q, %0, 0;\n\t@q st.global.v2.b32 [%1], {%2, %3};\n\t}" ::"r"(
                     static_cast<int>(p)),
                 "l"(ptr), "r"(v.x), "r"(v.y)
                 : "memory");
}

__device__ __forceinline__ void st16If(bool p, void* ptr, uint32_t v)
{
    asm volatile(
        "{\n\t.reg .pred q;\n\tsetp.ne.b32 q, %0, 0;\n\t@q st.global.b16 [%1], %2;\n\t}" ::"r"(static_cast<int>(p)),
        "l"(ptr), "h"(static_cast<uint16_t>(v))
        : "memory");
}

// MSO_MK_ET bit 1: fc2Epilogue of a stock-queue FC2 tile in phases across its 8 tokens. Per token the operations of
// fc2Epilogue on the same operands: yb = bf16(acc alpha2), the 4-column gather (shuffles down 8 / 16 / 24) and its row
// store by lanes 0-7 of a routed token, rho = yb (1 + w_post), ssq = rho^2 (0 if not routed) summed by the butterfly
// 16, 8, 4, 2, 1, lane 0's sum into the warp's slot; then fc2Epilogue's two barriers and its per-tile sum publish.
// gain: fc2Gain.
__device__ __forceinline__ void fc2EpiloguePx(
    Sm const& s, float const (&v)[8], int slot, int mt, uint8_t* wsCur, float gain)
{
    int const q = (threadIdx.x >> 5) - 4;
    int const lane = threadIdx.x & 31;
    int const e = threadIdx.x - 128;
    uint32_t const mask = rd<uint32_t const>(s, kMoeRdSlotMask)[slot];
    int8_t const* slotK = rd<int8_t const>(s, kMoeRdSlotK) + slot * kTok;
    uint2 const kb8 = *reinterpret_cast<uint2 const*>(slotK); // the slot's 8 k bytes
    float const a2 = rd<float const>(s, kMoeRdSlotScal + 512)[slot];
    int const col = mt * 128 + 32 * q + 4 * (lane & 7) + (lane >> 3);
    __nv_bfloat16* rowsDbg = dptr<__nv_bfloat16>(s, kMdRows);
    float* sq = rd<float>(s, kMoeRdEpiSq);
    uint8_t* const yCol = wsCur + kMoeWsY + static_cast<size_t>(col) * 2;
    uint32_t yb[kTok];
    int tk[kTok];
#pragma unroll
    for (int nTok = 0; nTok < kTok; ++nTok)
    {
        bool const routed = (mask >> nTok) & 1u;
        int const k = static_cast<int8_t>(((nTok < 4 ? kb8.x : kb8.y) >> (8 * (nTok & 3))) & 0xffu);
        tk[nTok] = nTok * kMoeTopK + (routed ? k : 0);
        yb[nTok] = bf16Bits(mulRn(v[nTok], a2));
    }
#pragma unroll
    for (int nTok = 0; nTok < kTok; ++nTok)
    {
        bool const routed = (mask >> nTok) & 1u;
        // Lane l < 8 gathers columns 4 l .. 4 l + 3 of the warp's 32 (lanes l, l + 8, l + 16, l + 24).
        uint32_t const y1 = __shfl_down_sync(0xffffffffu, yb[nTok], 8);
        uint32_t const y2 = __shfl_down_sync(0xffffffffu, yb[nTok], 16);
        uint32_t const y3 = __shfl_down_sync(0xffffffffu, yb[nTok], 24);
        st64If(routed && lane < 8, yCol + static_cast<size_t>(tk[nTok]) * (kMoeHidden * 2),
            make_uint2(lamport(yb[nTok] | (y1 << 16)), lamport(y2 | (y3 << 16))));
    }
    if (rowsDbg != nullptr)
    {
#pragma unroll
        for (int nTok = 0; nTok < kTok; ++nTok)
        {
            st16If((mask >> nTok) & 1u, rowsDbg + static_cast<size_t>(tk[nTok]) * kMoeHidden + col, yb[nTok]);
        }
    }
    float ssq[kTok];
#pragma unroll
    for (int nTok = 0; nTok < kTok; ++nTok)
    {
        float const rho = mulRn(__uint_as_float(yb[nTok] << 16), gain);
        ssq[nTok] = ((mask >> nTok) & 1u) ? mulRn(rho, rho) : 0.f;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
    {
#pragma unroll
        for (int nTok = 0; nTok < kTok; ++nTok)
        {
            ssq[nTok] = addRn(ssq[nTok], __shfl_xor_sync(0xffffffffu, ssq[nTok], o));
        }
    }
    if (lane == 0)
    {
#pragma unroll
        for (int nTok = 0; nTok < kTok; ++nTok)
        {
            sq[q * kTok + nTok] = ssq[nTok];
        }
    }
    namedSync(1, 128);
    if (e < kTok && ((mask >> e) & 1u))
    {
        float const tot = addRn(addRn(addRn(sq[e], sq[kTok + e]), sq[2 * kTok + e]), sq[3 * kTok + e]);
        int const tke = e * kMoeTopK + slotK[e];
        st32(wsCur + kMoeWsSsq + (static_cast<size_t>(tke) * kMoeFc2Tiles + mt) * 4, lamport(__float_as_uint(tot)));
    }
    namedSync(1, 128);
}

// ---- end MSO_MK_ET bit 1 ----
#endif

template <bool kPx = false> // kPx: the stock queue
__device__ __noinline__ void epilogueLoop(Ctx const& ctx, Sm const s, Cut const c, uint32_t tmem, uint8_t* wsCur)
{
    int const q = (threadIdx.x >> 5) - 4;
    int const lane = threadIdx.x & 31;
    int const r = 32 * q + lane;
    float* part = dptr<float>(s, kMdPart);
    uint32_t const idleNs = (dw(s, kMdFlags) & kMoeFlagSpinSleep) ? 128u : 0u;
    bool const epiPre = (dw(s, kMdFlags) & kMoeFlagE3EpiPre) != 0; // loads before the accumulator wait
    // ---- the stock queue: the CTA's whole tiles in queue order (the MMA warp's accumulator order); no partials ----
    if constexpr (kPx)
    {
        int const nT = unitsOf(c) / kMoeKb;
        int const nT1 = (c.b[0] - c.a[0]) / kMoeKb;
#pragma unroll 1
        for (int t = 0; t < nT; ++t)
        {
            int fc, u0, slot, mt, kb;
            pxTile(c, t, fc, u0);
            decodeUnit(fc, u0, slot, mt, kb);
            int const acc = t & (kMoeAccs - 1);
#if MSO_MK_ET & 1
            float const gainPre = fc == 1 ? fc2Gain(s, mt) : 0.f; // MSO_MK_ET bit 1: before the accumulator wait
#else
            float const gainPre = epiPre && fc == 1 ? fc2Gain(s, mt) : 0.f;
#endif
            waitBarIdle(ctx, bar(s, kMoeBarAccFull + acc), (t / kMoeAccs) & 1, kCodeMoeBar, idleNs);
            if (t == nT1 && threadIdx.x == 128)
            {
                moeStamp(ctx, kMoeStFc2EpiIn);
            }
            tcFenceAfter();
            float v[8];
            tmemLd8(tmem + (static_cast<uint32_t>(32 * q) << 16) + kMoeTmD + 8 * acc, v);
            tcFenceBefore();
            __syncwarp();
            if (lane == 0)
            {
                mbarArrive(bar(s, kMoeBarAccEmpty + acc));
            }
            if (fc == 0)
            {
                fc1Epilogue(s, v, slot, mt, wsCur);
            }
            else
            {
#if MSO_MK_ET & 1
                fc2EpiloguePx(s, v, slot, mt, wsCur, gainPre); // MSO_MK_ET bit 1
#else
                fc2Epilogue(s, v, slot, mt, wsCur, epiPre ? gainPre : fc2Gain(s, mt));
#endif
            }
            if (threadIdx.x == 128 && t < 8)
            {
                e3Tr(ctx, kE3TrTile + t); // per-unit stamp
            }
            if (t == nT1 - 1 && threadIdx.x == 128)
            {
                moeStamp(ctx, kMoeStFc1EpiDone);
            }
        }
        if (threadIdx.x == 128)
        {
            moeStamp(ctx, kMoeStEpiDone);
        }
        return;
    }
    // ---- end stock queue ----
    int ti = -1;
#pragma unroll 1
    for (int fc = 0; fc < 2; ++fc)
    {
        if (c.b[fc] <= c.a[fc])
        {
            continue;
        }
        int const g0 = c.a[fc] / kMoeKb;
        int const g1 = (c.b[fc] - 1) / kMoeKb;
        // The tiles in the units' order (listUnit): the last first.
#pragma unroll 1
        for (int g = g1; g >= g0; --g)
        {
            ++ti;
            int const acc = ti & (kMoeAccs - 1);
            // ---- kMoeFlagE3EpiPre: the finishing tile's first earlier partial and (FC2) the post gain are
            // requested before the accumulator wait (both were published / are constant long before) ----
            bool const finishedPre = kMoeKb * g + kMoeKb - 1 < c.b[fc];
            int const firstPre = ctaOfUnit(c, fc, kMoeKb * g);
            bool const prePart = epiPre && finishedPre && firstPre < static_cast<int>(blockIdx.x);
            uint4 pre[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
            uint8_t const* preA[2] = {nullptr, nullptr};
            if (prePart)
            {
                float const* src = part + ((static_cast<size_t>(firstPre) * 2 + fc) * 128 + r) * kTok;
                preA[0] = reinterpret_cast<uint8_t const*>(src);
                preA[1] = reinterpret_cast<uint8_t const*>(src + 4);
                pre[0] = ldPoll128(preA[0]);
                pre[1] = ldPoll128(preA[1]);
            }
            float gainPre = 0.f;
            if (epiPre && fc == 1 && finishedPre)
            {
                int sl, mtp, kbp;
                decodeUnit(fc, kMoeKb * g, sl, mtp, kbp);
                gainPre = fc2Gain(s, mtp);
            }
            // ---- end kMoeFlagE3EpiPre ----
            waitBarIdle(ctx, bar(s, kMoeBarAccFull + acc), (ti / kMoeAccs) & 1, kCodeMoeBar, idleNs);
            if (fc == 1 && g == g1 && threadIdx.x == 128)
            {
                moeStamp(ctx, kMoeStFc2EpiIn);
            }
            tcFenceAfter();
            float v[8];
            tmemLd8(tmem + (static_cast<uint32_t>(32 * q) << 16) + kMoeTmD + 8 * acc, v);
            tcFenceBefore();
            __syncwarp();
            if (lane == 0)
            {
                mbarArrive(bar(s, kMoeBarAccEmpty + acc));
            }
            bool const finished = kMoeKb * g + kMoeKb - 1 < c.b[fc];
            if (!finished)
            {
                // The tile continues on the next CTA(s): this CTA's FP32 partial into its slot.
                float* dst = part + ((static_cast<size_t>(blockIdx.x) * 2 + fc) * 128 + r) * kTok;
                st128(dst,
                    lamport4(make_uint4(
                        __float_as_uint(v[0]), __float_as_uint(v[1]), __float_as_uint(v[2]), __float_as_uint(v[3]))));
                st128(dst + 4,
                    lamport4(make_uint4(
                        __float_as_uint(v[4]), __float_as_uint(v[5]), __float_as_uint(v[6]), __float_as_uint(v[7]))));
                if (threadIdx.x == 128 && ti < 8)
                {
                    e3Tr(ctx, kE3TrTile + ti); // per-unit stamp
                }
                continue;
            }
            int const first = ctaOfUnit(c, fc, kMoeKb * g);
            if (first < static_cast<int>(blockIdx.x))
            {
                // Earlier k-blocks' partials, in CTA (= K) order, then this CTA's.
                float sum[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
                for (int cc = first; cc < static_cast<int>(blockIdx.x); ++cc)
                {
                    float const* src = part + ((static_cast<size_t>(cc) * 2 + fc) * 128 + r) * kTok;
                    uint4 pv[2];
                    uint8_t const* const pa[2]
                        = {reinterpret_cast<uint8_t const*>(src), reinterpret_cast<uint8_t const*>(src + 4)};
                    if (prePart && cc == first)
                    {
                        // kMoeFlagE3EpiPre: the pre-wait loads, re-polled only while a word is still empty
                        pv[0] = pre[0];
                        pv[1] = pre[1];
                        uint64_t t0 = 0;
                        while (!warpAll(full4(pv[0]) && full4(pv[1])))
                        {
                            spinCheck(ctx, t0, kCodeMoePart, 2, 0);
                            if (!full4(pv[0]))
                            {
                                pv[0] = ldPoll128(preA[0]);
                            }
                            if (!full4(pv[1]))
                            {
                                pv[1] = ldPoll128(preA[1]);
                            }
                        }
                    }
                    else
                    {
                        pollVecs<2>(ctx, pv, pa, kCodeMoePart);
                    }
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
            decodeUnit(fc, kMoeKb * g, slot, mt, kb);
            if (fc == 0)
            {
                fc1Epilogue(s, v, slot, mt, wsCur);
            }
            else
            {
                fc2Epilogue(s, v, slot, mt, wsCur, epiPre ? gainPre : fc2Gain(s, mt)); // epiPre: the early gain
            }
            if (threadIdx.x == 128 && ti < 8)
            {
                e3Tr(ctx, kE3TrTile + ti); // per-unit stamp
            }
            // The partial slots are emptied by their single reader, after this tile's outputs.
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

// ---- 6. the final unit: token t, columns [512 ch, +512), two per thread
// ---------------------------------------------------
__device__ __noinline__ void finalUnit(Ctx const& ctx, Sm const s, int fu, uint8_t const* wsCur)
{
    int const t = fu >> 3;
    int const ch = fu & 7;
    int const lane = threadIdx.x & 31;
    int const col = ch * 512 + 2 * threadIdx.x;
    __nv_bfloat16 const* wPost = dptr<__nv_bfloat16 const>(s, kMdGainPost);
    float const gain0 = addRn(__bfloat162float(wPost[col]), 1.f);
    float const gain1 = addRn(__bfloat162float(wPost[col + 1]), 1.f);
    float const epsPost = __uint_as_float(static_cast<uint32_t>(static_cast<uint64_t>(dw(s, kMdEps)) >> 32));
    int8_t const* tokSlot = rd<int8_t const>(s, kMoeRdTokSlot) + t * kMoeTopK;
    float const* w = rd<float const>(s, kMoeRdW) + t * kMoeTopK;
    float acc0 = 0.f, acc1 = 0.f;
    bool any = false;
#pragma unroll 1
    for (int k = 0; k < kMoeTopK; ++k)
    {
        if (tokSlot[k] < 0)
        {
            continue;
        }
        int const tk = t * kMoeTopK + k;
        uint4 sv[1];
        uint8_t const* const sa[1]
            = {lane < 8 ? wsCur + kMoeWsSsq + (static_cast<size_t>(tk) * kMoeFc2Tiles + 4 * lane) * 4 : nullptr};
        pollVecs<1>(ctx, sv, sa, kCodeMoeSsq);
        if (!any && threadIdx.x == 0)
        {
            moeStamp(ctx, kMoeStFinalIn);
        }
        any = true;
        float part = addRn(addRn(addRn(__uint_as_float(sv[0].x), __uint_as_float(sv[0].y)), __uint_as_float(sv[0].z)),
            __uint_as_float(sv[0].w));
        part = lane < 8 ? part : 0.f;
        part = addRn(part, __shfl_xor_sync(0xffffffffu, part, 1));
        part = addRn(part, __shfl_xor_sync(0xffffffffu, part, 2));
        part = addRn(part, __shfl_xor_sync(0xffffffffu, part, 4));
        float const tot = __shfl_sync(0xffffffffu, part, 0);
        float const scale = mulRn(w[k], rsqrtApprox(addRn(mulRn(tot, 1.f / kMoeHidden), epsPost)));
        uint32_t const yv = poll1(ctx, wsCur + kMoeWsY + (static_cast<size_t>(tk) * kMoeHidden + col) * 2, kCodeMoeY);
        acc0 = addRn(acc0, mulRn(scale, mulRn(bf16Lo(yv), gain0)));
        acc1 = addRn(acc1, mulRn(scale, mulRn(bf16Hi(yv), gain1)));
    }
    uint32_t lo = bf16Bits(acc0);
    uint32_t hi = bf16Bits(acc1);
    lo = lo == 0x8000u ? 0u : lo;
    hi = hi == 0x8000u ? 0u : hi;
    // ---- kMoeFlagMhF2: four threads' words (8 columns) as one publication into slot rank of the finalize area ----
#if MSO_MK_LANES & 8
    if (dw(s, kMdFlags) & kMoeFlagMhF2)
    {
        uint32_t const word = lo | (hi << 16);
        uint32_t const w1 = __shfl_down_sync(0xffffffffu, word, 1);
        uint32_t const w2 = __shfl_down_sync(0xffffffffu, word, 2);
        uint32_t const w3 = __shfl_down_sync(0xffffffffu, word, 3);
        if ((threadIdx.x & 3) == 0)
        {
            mhPublish(ctx, i1FinOff(ctx, t, col), lamport4(make_uint4(word, w1, w2, w3)));
        }
#if !MSO_MK_LD_SC // MSO_MK_LD_SC: the served build's MhF2 descriptors carry no kMdOut (stageMoeExperts traps on one)
        if (dw(s, kMdOut) != 0)
        {
            st32(dptr<uint8_t>(s, kMdOut) + (static_cast<size_t>(t) * kMoeHidden + col) * 2, word);
        }
#endif
    }
    else
#endif
        // ---- end kMoeFlagMhF2 ----
        st32(dptr<uint8_t>(s, kMdOut) + (static_cast<size_t>(t) * kMoeHidden + col) * 2, lo | (hi << 16));
    if (threadIdx.x == 0)
    {
        moeStamp(ctx, kMoeStFinalDone);
    }
}

#if MSO_MK_ET & 6
// ---- MSO_MK_ET bits 2 / 4: the final unit of the stock-queue path (finalUnit's words, values and arithmetic). The
// token's local picks (k ascending) in windows of up to 4: lanes 8 j .. 8 j + 7 of every warp read pick j's 32
// per-tile sums (4 per lane, the vectors lanes 0-7 read for one pick in finalUnit), every thread the pick's two row
// columns. Bit 2: the window's first pick's sums are polled as in finalUnit; then the window's other loads go out at
// once. Bit 4: every load of the window from the start. Words still empty are re-polled until all are present. Each
// pick's sum (the butterfly 1, 2, 4 stays inside its 8 lanes) and scale (computed by lane 8 j for pick j) and the
// accumulation are finalUnit's, in k order. ----
__device__ __noinline__ void finalUnitPx(Ctx const& ctx, Sm const s, int fu, uint8_t const* wsCur)
{
    int const t = fu >> 3;
    int const ch = fu & 7;
    int const lane = threadIdx.x & 31;
    int const grp = lane >> 3;
    int const col = ch * 512 + 2 * threadIdx.x;
    __nv_bfloat16 const* wPost = dptr<__nv_bfloat16 const>(s, kMdGainPost);
    float const gain0 = addRn(__bfloat162float(wPost[col]), 1.f);
    float const gain1 = addRn(__bfloat162float(wPost[col + 1]), 1.f);
    float const epsPost = __uint_as_float(static_cast<uint32_t>(static_cast<uint64_t>(dw(s, kMdEps)) >> 32));
    uint2 const ts8 = *reinterpret_cast<uint2 const*>(rd<int8_t const>(s, kMoeRdTokSlot) + t * kMoeTopK);
    float const* w = rd<float const>(s, kMoeRdW) + t * kMoeTopK;
    // the local picks (tokSlot[k] >= 0) as a bit mask
    uint32_t local = 0;
#pragma unroll
    for (int k = 0; k < kMoeTopK; ++k)
    {
        uint32_t const b = ((k < 4 ? ts8.x : ts8.y) >> (8 * (k & 3))) & 0xffu;
        local |= (b & 0x80u) ? 0u : (1u << k);
    }
    uint8_t const* const sBase
        = wsCur + kMoeWsSsq + (static_cast<size_t>(t) * kMoeTopK * kMoeFc2Tiles + 4 * (lane & 7)) * 4;
    uint8_t const* const yBase = wsCur + kMoeWsY + (static_cast<size_t>(t) * kMoeTopK * kMoeHidden + col) * 2;
    float acc0 = 0.f, acc1 = 0.f;
    bool any = false;
#pragma unroll 1
    while (local != 0)
    {
        int kw[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            kw[j] = local != 0 ? __ffs(local) - 1 : -1;
            local &= local - 1;
        }
        int const kMine = grp == 0 ? kw[0] : grp == 1 ? kw[1] : grp == 2 ? kw[2] : kw[3];
        uint8_t const* const sa = kMine >= 0 ? sBase + kMine * (kMoeFc2Tiles * 4) : nullptr;
        float const wMine = kMine >= 0 ? w[kMine] : 0.f;
        uint8_t const* ya[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            ya[j] = kw[j] >= 0 ? yBase + kw[j] * (kMoeHidden * 2) : nullptr;
        }
        // Every word of the window starts empty (an absent pick's words count as present); one loop requests the
        // empty words and tests them all, until every word is present. Bit 2: the first pick's sums come from
        // finalUnit's first poll, before anything else is requested.
        uint4 sv = sa != nullptr ? emptyVec() : make_uint4(0, 0, 0, 0);
#if MSO_MK_ET & 2
        {
            uint4 s0[1];
            uint8_t const* const a0[1] = {grp == 0 ? sa : nullptr};
            pollVecs<1>(ctx, s0, a0, kCodeMoeSsq); // the window's first pick's sums, as finalUnit
            if (grp == 0)
            {
                sv = s0[0];
            }
        }
        if (!any && threadIdx.x == 0)
        {
            moeStamp(ctx, kMoeStFinalIn); // as finalUnit: the first local pick's sums arrived
        }
#endif
        uint32_t yv[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            yv[j] = ya[j] != nullptr ? kEmpty : 0u;
        }
        uint64_t t0 = 0;
        while (true)
        {
            if (!full4(sv))
            {
                sv = ldPoll128(sa);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                if (yv[j] == kEmpty)
                {
                    yv[j] = ldPoll32(ya[j]);
                }
            }
            if (warpAll(full4(sv) & (yv[0] != kEmpty) & (yv[1] != kEmpty) & (yv[2] != kEmpty) & (yv[3] != kEmpty)))
            {
                break;
            }
            spinCheck(ctx, t0, full4(sv) ? kCodeMoeY : kCodeMoeSsq, 4, 0);
        }
#if MSO_MK_ET & 4
        if (!any && threadIdx.x == 0)
        {
            moeStamp(ctx, kMoeStFinalIn); // bit 4: the first window complete
        }
#endif
        any = true;
        float part = addRn(
            addRn(addRn(__uint_as_float(sv.x), __uint_as_float(sv.y)), __uint_as_float(sv.z)), __uint_as_float(sv.w));
        part = addRn(part, __shfl_xor_sync(0xffffffffu, part, 1));
        part = addRn(part, __shfl_xor_sync(0xffffffffu, part, 2));
        part = addRn(part, __shfl_xor_sync(0xffffffffu, part, 4));
        // Lane 8 j holds pick j's total as finalUnit's lane 0 does; it computes finalUnit's scale of the pick (all
        // four picks' scales at once), which every lane then takes from it in k order: a rolled loop over the
        // window's picks (kw holds them first, the absent ones last).
        float const scaleMine = mulRn(wMine, rsqrtApprox(addRn(mulRn(part, 1.f / kMoeHidden), epsPost)));
        int k0 = kw[0], k1 = kw[1], k2 = kw[2], k3 = kw[3];
        uint32_t y0 = yv[0], y1 = yv[1], y2 = yv[2], y3 = yv[3];
        int src = 0;
#pragma unroll 1
        while (k0 >= 0)
        {
            float const scale = __shfl_sync(0xffffffffu, scaleMine, src);
            acc0 = addRn(acc0, mulRn(scale, mulRn(bf16Lo(y0), gain0)));
            acc1 = addRn(acc1, mulRn(scale, mulRn(bf16Hi(y0), gain1)));
            k0 = k1;
            k1 = k2;
            k2 = k3;
            k3 = -1;
            y0 = y1;
            y1 = y2;
            y2 = y3;
            src += 8;
        }
    }
    uint32_t lo = bf16Bits(acc0);
    uint32_t hi = bf16Bits(acc1);
    lo = lo == 0x8000u ? 0u : lo;
    hi = hi == 0x8000u ? 0u : hi;
#if MSO_MK_LANES & 8
    if (dw(s, kMdFlags) & kMoeFlagMhF2)
    {
        uint32_t const word = lo | (hi << 16);
        uint32_t const w1 = __shfl_down_sync(0xffffffffu, word, 1);
        uint32_t const w2 = __shfl_down_sync(0xffffffffu, word, 2);
        uint32_t const w3 = __shfl_down_sync(0xffffffffu, word, 3);
        if ((threadIdx.x & 3) == 0)
        {
            mhPublish(ctx, i1FinOff(ctx, t, col), lamport4(make_uint4(word, w1, w2, w3)));
        }
#if !MSO_MK_LD_SC
        if (dw(s, kMdOut) != 0)
        {
            st32(dptr<uint8_t>(s, kMdOut) + (static_cast<size_t>(t) * kMoeHidden + col) * 2, word);
        }
#endif
    }
    else
#endif
        st32(dptr<uint8_t>(s, kMdOut) + (static_cast<size_t>(t) * kMoeHidden + col) * 2, lo | (hi << 16));
    if (threadIdx.x == 0)
    {
        moeStamp(ctx, kMoeStFinalDone);
    }
}

// ---- end MSO_MK_ET bits 2 / 4 ----
#endif

// ---- the warp roles on the launch's ring layout (kMoeFlagE3Ring10: 10 stages, X / h shared). kPx: the cut is the
// stock queue (c.px); its loops are the kPx instantiations, so the byte / tile cut loops carry no stock-queue code ----
template <bool kPx>
__device__ __forceinline__ void expertRoles(Ctx const& ctx, Sm const s, Cut const& c, int64_t flags, int issued,
    uint32_t tmem, uint8_t* wsCur, uint8_t* wsNext, bool pxRing8)
{
    int const tid = threadIdx.x;
    int const warp = tid >> 5;
    bool const ring10 = e3Ring10(flags);
    // ---- the table-driven issue loops (the whole of warp 0 / the table-driven MMA warp) ----
    bool const lxProd = lxProducerOk(flags, ring10 ? kMoeStagesMax : kMoeStages);
    bool const lxMma = lxMmaOk(flags);
    // ---- end table-driven issue loops ----
    // ---- the stock-shape MMA loop (kMoeFlagPxMma) and the 8-stage ring (kMoeFlagPxRing8, needs the first) ----
    if constexpr (kPx)
    {
        bool const pxMma = (flags & kMoeFlagPxMma) != 0 && !ring10;
        if (pxMma && pxRing8)
        {
            if (warp == 0)
            {
                if (lxProducerOk(flags, kMoeStagesPx8))
                {
                    produceLoopLx<kMoeStagesPx8, true>(ctx, s, c, issued);
                }
                else if (tid == 0)
                {
                    produceLoop<kMoeStagesPx8, true>(ctx, s, c, issued);
                }
            }
            else if (warp == 1)
            {
                mmaLoopPx<kMoeStagesPx8>(ctx, s, c, tmem);
            }
            else if (warp < 4)
            {
                loaderLoop<kMoeStagesPx8, true>(ctx, s, c, wsCur, wsNext);
            }
            else
            {
                epilogueLoop<true>(ctx, s, c, tmem, wsCur);
            }
            return;
        }
        if (pxMma && warp == 1)
        {
            mmaLoopPx<kMoeStages>(ctx, s, c, tmem);
            return;
        }
    }
    // ---- end stock queue ----
    if (warp == 0)
    {
        if (lxProd) // kMoeFlagLxProducer
        {
            if (ring10)
            {
                produceLoopLx<kMoeStagesMax, kPx>(ctx, s, c, issued);
            }
            else
            {
                produceLoopLx<kMoeStages, kPx>(ctx, s, c, issued);
            }
        }
        else if (tid == 0)
        {
            if (ring10)
            {
                produceLoop<kMoeStagesMax, kPx>(ctx, s, c, issued);
            }
            else
            {
                produceLoop<kMoeStages, kPx>(ctx, s, c, issued);
            }
        }
    }
    else if (warp == 1)
    {
        // The quad groups (mmaLoopG) walk the byte cut's ranges; the stock queue runs the pairs loops
        bool const groups = !kPx && (flags & (kMoeFlagE3Fc1Quads | kMoeFlagE3Fc2Quads)) != 0;
        if (lxMma) // kMoeFlagLxMma (never with the quad groups)
        {
            if (ring10)
            {
                mmaLoopLx<kMoeStagesMax, kPx>(ctx, s, c, tmem);
            }
            else
            {
                mmaLoopLx<kMoeStages, kPx>(ctx, s, c, tmem);
            }
        }
        else if (ring10)
        {
            if (groups)
            {
                mmaLoopG<kMoeStagesMax>(ctx, s, c, tmem);
            }
            else
            {
                mmaLoop<kMoeStagesMax, kPx>(ctx, s, c, tmem);
            }
        }
        else if (groups)
        {
            mmaLoopG<kMoeStages>(ctx, s, c, tmem);
        }
        else
        {
            mmaLoop<kMoeStages, kPx>(ctx, s, c, tmem);
        }
    }
    else if (warp < 4)
    {
        if (ring10)
        {
            loaderLoop<kMoeStagesMax, kPx>(ctx, s, c, wsCur, wsNext);
        }
        else
        {
            loaderLoop<kMoeStages, kPx>(ctx, s, c, wsCur, wsNext);
        }
    }
    else
    {
        epilogueLoop<kPx>(ctx, s, c, tmem, wsCur);
    }
}

// ---- end warp roles ----

#if MSO_MK_CX
// ---- MSO_MK_CX (attribution only): the oracle slot's loads and stores (experts_defs.cuh, the MSO_MK_CX block). The
// image's loads go out at stage entry with the header's (no ordering is needed between them: a launch reads only a
// slot that an earlier launch completed, see stageMoeExperts). ----
constexpr int kCxImgVecs = (kMoeRdRstd - kMoeRdSlotExpert) / 16; // 120: slot tables, selections, weights, ids, presence
static_assert(kCxImgOff + 16 * kCxImgVecs <= kCxSlotBytes && kCxImgVecs <= kThreads, "cx: the image fits the slot");

__device__ __forceinline__ uint4 cxLd(uint8_t const* p)
{
    uint4 v;
    asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p)
                 : "memory");
    return v;
}

__device__ __forceinline__ void cxSt(uint8_t* p, uint4 v)
{
    asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}

// ---- end MSO_MK_CX ----
#endif

#if MSO_MK_RP & 1
// ---- MSO_MK_RP: the routing state published once per rank (experts_defs.cuh, the MSO_MK_RP block). Owner: the
// publisher CTA kCtas - 1 is the only writer of wsCur's routing area in a launch (one store per vector, after its
// place()); every other CTA of the rank reads it (polls each vector until no word is kEmpty), so no reader clears it:
// the next copy's area is emptied by the loaders' clear of wsNext, as every other word of the workspace, before its
// next writer runs (C >= 3 launches later). ----
static_assert(8 * kRpImgVecs == kMoeRdRstd - kMoeRdSlotExpert, "rp: the image is the reduction area's routing tables");
static_assert(kRpVecs <= kThreads && 16 * kRpVecs <= kRpBytes, "rp: one vector per thread");

__device__ __forceinline__ uint32_t rpLo(uint32_t w)
{
    return (w & 0xffffu) | 0x10000u;
}

__device__ __forceinline__ uint32_t rpHi(uint32_t w)
{
    return (w >> 16) | 0x10000u;
}

__device__ __forceinline__ uint32_t rpJoin(uint32_t lo, uint32_t hi)
{
    return (lo & 0xffffu) | (hi << 16);
}

// The publisher, after place() (its tables are complete for every thread): thread t < kRpImgVecs stores image words
// 2 t and 2 t + 1 as vector t, thread kRpImgVecs stores E.
__device__ __noinline__ void rpPublish(Sm const s, uint8_t* wsCur, int E)
{
    int const t = threadIdx.x;
    uint8_t* const area = wsCur + kMoeWsRp;
    if (t < kRpImgVecs)
    {
        uint32_t const* w = rd<uint32_t const>(s, kMoeRdSlotExpert) + 2 * t;
        uint32_t const w0 = w[0], w1 = w[1];
        st128(area + 16 * t, make_uint4(rpLo(w0), rpHi(w0), rpLo(w1), rpHi(w1)));
    }
    else if (t == kRpImgVecs)
    {
        uint32_t const e = static_cast<uint32_t>(E);
        st128(area + 16 * t, make_uint4(rpLo(e), rpHi(e), 0x10000u, 0x10000u));
    }
}

// Every other CTA, in place of route() and place(): the whole CTA polls the area (thread t: vector t), decodes it into
// the reduction area's tables and E, then a CTA barrier.
__device__ __noinline__ void rpPoll(Ctx const& ctx, Sm const s, uint8_t const* wsCur)
{
    int const t = threadIdx.x;
    uint8_t const* const a[1] = {t < kRpVecs ? wsCur + kMoeWsRp + 16 * t : nullptr};
    uint4 v[1];
    pollVecs<1>(ctx, v, a, kCodeMoeLogits);
    if (t < kRpImgVecs)
    {
        uint32_t* w = rd<uint32_t>(s, kMoeRdSlotExpert) + 2 * t;
        w[0] = rpJoin(v[0].x, v[0].y);
        w[1] = rpJoin(v[0].z, v[0].w);
    }
    else if (t == kRpImgVecs)
    {
        rd<int>(s, kMoeRdScal)[kMoeScE] = static_cast<int>(rpJoin(v[0].x, v[0].y));
    }
    __syncthreads();
}
#if MSO_MK_XP & 8
// ---- MSO_MK_XP bit 8: the publisher's warm-up pass, at its stage entry (it would otherwise wait for the logits):
// route() on zero logits through its staged path (pre = true: no poll; the staging is the late ring stages, unused
// until the publisher's gatherX), then place(); then the words place() accumulates into (the presence words and the
// slot masks) are cleared as at the stage entry. Every other word the pass writes (selections, weights, local ids, slot
// tables, E) is written again by the real pass before any reader, as every launch writes them over the previous
// launch's values; so the real pass computes and publishes the same tables, from code the warm-up pass has just
// fetched. ----
__device__ __noinline__ void xpWarmPublisher(Ctx const& ctx, Sm const s)
{
    uint32_t const stg = s.base + kMoeSmQStage;
    for (int i = threadIdx.x; i < kTok * kMoeExperts / 4; i += kThreads)
    {
        sts128(stg + 16 * i, make_uint4(0u, 0u, 0u, 0u));
    }
    __syncthreads();
    route(ctx, s, true);
    __syncthreads();
    place(ctx, s);
    if (threadIdx.x < 16)
    {
        rd<uint32_t>(s, kMoeRdPres)[threadIdx.x] = 0;
    }
    if (threadIdx.x >= 32 && threadIdx.x < 32 + kMoeMaxSlots)
    {
        rd<uint32_t>(s, kMoeRdSlotMask)[threadIdx.x - 32] = 0;
    }
    __syncthreads();
}
#endif
// ---- end MSO_MK_RP ----
#endif

// ---- the stage from the cut to the end of the warp roles, per cut (kPx: the stock queue). The byte / tile cut's
// instantiation is inline in the stage, the stock queue's is one out-of-line function (stageCutPx): the stage runs once
// per launch from cold instruction memory, and neither path's code then sits in the other's instruction stream ----
template <bool kPx>
__device__ __forceinline__ void stageCut(Ctx const& ctx, Sm const s, int64_t const* sd, int E, uint32_t tmem,
    bool localQuant,
#if MSO_MK_RT & 4
    bool issueAtCut, bool quantAtCut, uint8_t* wsCur, uint8_t* wsNext, bool rtEarly, RtPlace const rp) // MSO_MK_RT & 4
#elif MSO_MK_CX
    bool issueAtCut, bool quantAtCut, uint8_t* wsCur, uint8_t* wsNext, int cxEarly) // MSO_MK_CX: early count, 0 = own
#else
    bool issueAtCut, bool quantAtCut, uint8_t* wsCur, uint8_t* wsNext)
#endif
{
    int const tid = threadIdx.x;
    int const warp = tid >> 5;
    Cut c;
    if constexpr (kPx) // the stock queue
    {
#if MSO_MK_XP & 32
        c = makeCutPxTab(s, E, static_cast<int>(blockIdx.x)); // MSO_MK_XP bit 32: the stage entry's table
#else
        c = makeCutPxBody(E, static_cast<int>(blockIdx.x));
#endif
    }
    else
    {
        makeCut(c, E, (icMoeFlags(sd[kMdFlags]) & kMoeFlagTileCut) != 0, static_cast<int>(blockIdx.x));
    }
    int const n = unitsOf(c);
    // Per-slot scalars (alpha1, scale_c, alpha2): requested now, stored after the quantization (only the epilogue
    // reads them).
    float slotA1 = 0.f, slotSc = 0.f, slotA2 = 0.f;
    int const si = tid - (kThreads - kMoeMaxSlots); // warps 6-7: thread 0 issues the early stages meanwhile
    if (si >= 0 && si < E)
    {
#if MSO_MK_RT & 4
        int const ex = rtEarly ? rtSlotId(rp, si) : rd<int const>(s, kMoeRdSlotExpert)[si]; // no barrier yet
#else
        int const ex = rd<int const>(s, kMoeRdSlotExpert)[si];
#endif
#if MSO_MK_RT & 32
        // MSO_MK_RT bit 32: from the window scalars copied at stage entry
        slotA1 = rd<float const>(s, kMoeRdRtScal)[ex];
        slotSc = rd<float const>(s, kMoeRdRtScal)[kMoeWindow + ex];
        slotA2 = rd<float const>(s, kMoeRdRtScal)[2 * kMoeWindow + ex];
#else
        slotA1 = __ldg(dptr<float const>(s, kMdAlpha1) + ex);
        slotSc = __ldg(dptr<float const>(s, kMdScaleC) + ex);
        slotA2 = __ldg(dptr<float const>(s, kMdAlpha2) + ex);
#endif
    }
    bool const piece0 = (icMoeFlags(sd[kMdFlags]) & kMoeFlagPiece0First) != 0;
    // The early stages (lane 0 of warps 0.., under their row loads); piece 0 first: unit 0 only.
    // (Spread quantization: the staging in the late ring stages is free once shareQuant is done, so one per warp.)
    int nEarly = localQuant                           ? kMoeStagesEarly
        : (icMoeFlags(sd[kMdFlags]) & kMoeFlagEarly9) ? kMoeStages // the ninth stage too
                                                      : kTok;
    // ---- the early cap: fewer early requests with the spread quantization (the producer requests the rest) ----
    {
        int const capE = static_cast<int>((icMoeFlags(sd[kMdFlags]) >> kMoeCapEarlyShift) & 15);
        if (!localQuant && capE > 0 && capE < nEarly)
        {
            nEarly = capE;
        }
    }
    // ---- end early cap ----
#if MSO_MK_CX
    if (cxEarly > 0 && !localQuant)
    {
        nEarly = cxEarly; // MSO_MK_CX: the oracle's early count; the in-flight and ring caps below still hold
    }
#endif
    int issued = piece0 ? (n > 0 ? 1 : 0) : (n < nEarly ? n : nEarly);
    // ---- the early burst is at most the in-flight cap (kMoeW2CapMask) ----
    {
        int const cap
            = w2Cap(icMoeFlags(sd[kMdFlags]), e3Ring10(icMoeFlags(sd[kMdFlags])) ? kMoeStagesMax : kMoeStages);
        issued = issued < cap ? issued : cap;
    }
    // ---- end in-flight cap ----
    // ---- kMoeFlagPxRing8: the early requests fill at most the 8 stages (unit j in stage j % 8 = j) ----
    bool const pxRing8
        = kPx && (icMoeFlags(sd[kMdFlags]) & kMoeFlagPxRing8) != 0 && !e3Ring10(icMoeFlags(sd[kMdFlags]));
    if (pxRing8)
    {
        issued = issued < kMoeStagesPx8 ? issued : kMoeStagesPx8;
    }
    // ---- end kMoeFlagPxRing8 ----
    if (tid == 0)
    {
        moeStamp(ctx, kMoeStCut);
        // ---- attribution: the CTA's slot count E and unit count (raw, in kStampI2aRaw6) ----
        if (MSO_MK_STAMPS_DETAIL(ctx) != nullptr)
        {
            MSO_MK_STAMPS(ctx)[kStampI2aRaw6] = (static_cast<uint64_t>(E) << 32) | static_cast<uint32_t>(n); // raw
        }
        // ---- end attribution ----
    }
    // ---- kMoeFlagIssueAtCut / kMoeFlagQuantAtCut ----
    if (issueAtCut)
    {
        if ((tid & 31) == 0)
        {
            for (int u = warp; u < issued; u += kWarps)
            {
                issueUnit<kMoeStages, kPx>(s, c, u, policyEvictFirst()); // kPx: the stock queue
            }
        }
        if (tid == 0)
        {
            moeStamp(ctx, kMoeStEarly);
        }
        shareQuant(ctx, s, wsCur);
    }
    if (quantAtCut)
    {
        shareQuant(ctx, s, wsCur);
    }
    // ---- end kMoeFlagIssueAtCut / kMoeFlagQuantAtCut ----
#if MSO_MK_RT & 4
    // ---- MSO_MK_RT bit 4: the early units are requested by rtPlace already (the same count: checked in the check
    // builds) ----
#if !MSO_MK_LEAN
    if (rtEarly && E > 0 && issued != rtEarlyCount(icMoeFlags(sd[kMdFlags]), n))
    {
        __trap();
    }
#endif
    if (rtEarly)
    {
        issueAtCut = true; // gatherX requests nothing (the flag is not read after this point)
    }
#endif
#if MSO_MK_RT & 8
    // ---- MSO_MK_RT bit 8: X is in shared memory (route()); the early requests here unless the placement made
    // them ----
    bool const rtX = !localQuant && (icMoeFlags(sd[kMdFlags]) & kMoeFlagQuantFirst) != 0;
    if (rtX && E > 0)
    {
        if (!issueAtCut && (tid & 31) == 0)
        {
            for (int u = warp; u < issued; u += kWarps)
            {
                issueUnit<kMoeStages, kPx>(s, c, u, policyEvictFirst());
            }
        }
        if (tid == 0)
        {
            moeStamp(ctx, kMoeStEarly);
        }
    }
    else
#endif
        if (E > 0 && !localQuant)
    {
        gatherX<kPx>(ctx, s, c, issueAtCut ? 0 : issued, wsCur); // issueAtCut: requests made already; kPx: stock queue
    }
    else if (E > 0)
    {
        quantize<kPx>(ctx, s, c, issued); // kPx: the stock queue
    }
    else
    {
        // The gain (and pool) copies complete before the CTA leaves (nothing else waits for them).
        waitBar(ctx, bar(s, kMoeBarGain), 0, kCodeMoeBar);
        if (moePoolCopy(icMoeFlags(sd[kMdFlags]))) // the stage-entry pool copy
        {
            waitBar(ctx, bar(s, kMoeBarPool), 0, kCodeMoeBar);
        }
    }
    if (si >= 0 && si < E)
    {
        rd<float>(s, kMoeRdSlotScal)[si] = slotA1;
        rd<float>(s, kMoeRdSlotScal + 256)[si] = slotSc;
        rd<float>(s, kMoeRdSlotScal + 512)[si] = slotA2;
    }
    if (tid == 0)
    {
        moeStamp(ctx, kMoeStQuant);
    }
    fenceProxyAsyncSmem();
    tcFenceBefore();
    __syncthreads();
    tcFenceAfter();
    expertRoles<kPx>(ctx, s, c, icMoeFlags(sd[kMdFlags]), issued, tmem, wsCur, wsNext, pxRing8); // kPx: stock queue
}

#if MSO_MK_RT & 4
__device__ __noinline__ void stageCutPx(Ctx const& ctx, Sm const s, int64_t const* sd, int E, uint32_t tmem,
    bool localQuant, bool issueAtCut, bool quantAtCut, uint8_t* wsCur, uint8_t* wsNext, bool rtEarly, RtPlace const rp)
{
    stageCut<true>(ctx, s, sd, E, tmem, localQuant, issueAtCut, quantAtCut, wsCur, wsNext, rtEarly, rp);
}
#elif MSO_MK_CX
__device__ __noinline__ void stageCutPx(Ctx const& ctx, Sm const s, int64_t const* sd, int E, uint32_t tmem,
    bool localQuant, bool issueAtCut, bool quantAtCut, uint8_t* wsCur, uint8_t* wsNext, int cxEarly) // MSO_MK_CX
{
    stageCut<true>(ctx, s, sd, E, tmem, localQuant, issueAtCut, quantAtCut, wsCur, wsNext, cxEarly);
}
#else
__device__ __noinline__ void stageCutPx(Ctx const& ctx, Sm const s, int64_t const* sd, int E, uint32_t tmem,
    bool localQuant, bool issueAtCut, bool quantAtCut, uint8_t* wsCur, uint8_t* wsNext)
{
    stageCut<true>(ctx, s, sd, E, tmem, localQuant, issueAtCut, quantAtCut, wsCur, wsNext);
}
#endif
// ---- end per-cut stage ----

#if MSO_MK_LY_PADX > 0
// ---- the code-placement pad (attribution only, never served): MSO_MK_LY_PADX NANOSLEEPs in a function that only the
// router-free stageMhRouter's never-run path calls (stages/moe_hops.cuh, the MSO_MK_LY_PADX block); its mangled name
// sorts between finalUnitPx and epilogueLoop, so the pad moves epilogueLoop, produceLoopLx, mmaLoopPx (and every later
// callee) against loaderLoop, shareQuant, stageCutPx and finalUnitPx. ----
__device__ __noinline__ void lyPadEpilog()
{
#pragma unroll
    for (int i = 0; i < MSO_MK_LY_PADX; ++i)
    {
        asm volatile("nanosleep.u32 0;" ::: "memory");
    }
    __trap();
}

// ---- end code-placement pad ----
#endif

} // namespace p4a

// ---- the stage
// ---------------------------------------------------------------------------------------------------------
__device__ __noinline__ void stageMoeExperts(Ctx const& ctx, int unit, uint32_t wOff)
{
    using namespace p4a;
    int const tid = threadIdx.x;
    int const warp = tid >> 5;
    Sm s;
    s.redS = ctx.smemBase + kSmemRed;
    s.base = ctx.smemBase + wOff;
    int64_t* sd = rd<int64_t>(s, kMoeRdDesc);
    int64_t const* desc = ptr<int64_t const>(ctx, kPMoe);
#if MSO_MK_CX
    // ---- MSO_MK_CX (attribution only, never served): the layer's oracle slot (descriptor word kMdDbg). Its header
    // and image are loaded here, with no ordering between them: `have` changes only at CTA 0's stage end of a recording
    // launch, after every CTA has passed this point (CTA 0's gatherX has polled every CTA's quantization share, which
    // each CTA publishes after this load), so a launch reads only an image an earlier launch completed. cxUse is
    // uniform over the launch. ----
    uint8_t* const cxSlot = reinterpret_cast<uint8_t*>(desc[kMdDbg]);
    bool cxUse = false;
    int cxLevel = 0, cxEarly = 0, cxE = 0;
    uint32_t cxWant = 0;
    uint4 cxV = make_uint4(0u, 0u, 0u, 0u);
    if (cxSlot != nullptr)
    {
        uint4 const h = cxLd(cxSlot);
        uint4 const hE = cxLd(cxSlot + 16);
        if (tid < kCxImgVecs)
        {
            cxV = cxLd(cxSlot + kCxImgOff + 16 * tid);
        }
        cxWant = h.x;
        cxLevel = static_cast<int>(h.z);
        cxUse = h.y == h.x && cxLevel != 0;
        cxEarly = cxUse ? static_cast<int>(h.w) : 0;
        cxE = static_cast<int>(hE.x);
    }
#if MSO_MK_RP & 1
    // MSO_MK_CX with MSO_MK_RP (attribution only): only the publisher takes a valid slot's state (and publishes it at
    // once); every other CTA polls the publisher's area as in the MSO_MK_RP build. CTA 0 records when the slot is stale
    // (cxValid false).
    bool const cxValid = cxUse;
    cxUse = cxUse && static_cast<int>(blockIdx.x) == kCtas - 1;
    cxEarly = cxUse ? cxEarly : 0;
#endif
    // ---- end MSO_MK_CX ----
#endif
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
#if MSO_MK_XP & 32
    if (tid >= 128 && tid <= 128 + kMoeMaxSlots)
    {
        xpCutTable(s); // MSO_MK_XP bit 32: entry tid - 128 of the cut table (read after the barrier below)
    }
#endif
    if (tid == 0)
    {
        for (int i = 0; i < kMoeNumBars; ++i)
        {
            uint32_t const count = (i >= kMoeBarAccEmpty && i < kMoeBarAccEmpty + kMoeAccs) ? 4u
                : (i >= kMoeBarHReady && i < kMoeBarHReady + kMoeKb)                        ? 2u
                                                                                            : 1u;
            mbarInit(bar(s, i), count);
        }
        fenceMbarInit();
    }
    if (warp == 1)
    {
        tmemAlloc(s.redS + kMoeRdScal + 4 * kMoeScTmem, kMoeTmemCols);
    }
    tcFenceBefore();
    __syncthreads();
    tcFenceAfter();
    if (sd[kMdMagic] != kMoeMagic)
    {
        __trap();
    }
    if (kIcMoeMask != 0 && (sd[kMdFlags] & kIcMoeMask) != kIcMoeVal)
    {
        __trap(); // MSO_MK_IC_MOE_FIX: this launch's experts flags are not the ones the build fixed
    }
#if (MSO_MK_RT & 16) && MSO_MK_LEAN
#if MSO_MK_CX
    if (sd[kMdRows] != 0)
    {
        __trap(); // MSO_MK_RT bit 16: the lean build compiled the debug-block code out (kMdDbg: the MSO_MK_CX slot)
    }
#else
    if (sd[kMdDbg] != 0 || sd[kMdRows] != 0)
    {
        __trap(); // MSO_MK_RT bit 16: this lean build compiled the debug-block code out
    }
#endif
#endif
#if MSO_MK_LD_SC && (MSO_MK_LANES & 8)
    if ((icMoeFlags(sd[kMdFlags]) & kMoeFlagMhF2) && sd[kMdOut] != 0)
    {
        __trap(); // MSO_MK_LD_SC: finalUnit stores no kMdOut copy under MhF2 in this build
    }
#endif
    if (tid == 0)
    {
        moeStamp(ctx, kMoeStDesc);
    }
    if (tid == 32)
    {
        // The stacks' descriptors, before the routing decides which rows their first requests read; the pre-expert
        // gain into the FC2 B area (read by the quantization).
        prefetchTmap(sd[kMdTmapW13]);
        prefetchTmap(sd[kMdTmapW2]);
        if (moePoolCopy(icMoeFlags(sd[kMdFlags]))) // the rows' bulk copy at stage entry
        {
            // Spread quantization: only the (at most two) rows of this CTA's share; local: all 8 rows.
            int lo, hi, r0, nr;
            shareRange(lo, hi, r0, nr);
            if (icMoeFlags(sd[kMdFlags]) & kMoeFlagLocalQuant)
            {
                r0 = 0;
                nr = kTok;
            }
            mbarExpectTx(bar(s, kMoeBarPool), nr * kMoeHidden * 2);
            bulkCopy(s.base + kMoeSmQStage, reinterpret_cast<uint8_t const*>(sd[kMdPool]) + r0 * kMoeHidden * 2,
                nr * kMoeHidden * 2, bar(s, kMoeBarPool));
        }
        bool const warmTma = !(icMoeFlags(sd[kMdFlags]) & kMoeFlagNoWarmTma);
        mbarExpectTx(bar(s, kMoeBarGain), kMoeHidden * 2 + (warmTma ? 2 * kMoeUnitA : 0));
        bool const ring10 = e3Ring10(icMoeFlags(sd[kMdFlags])); // kMoeFlagE3Ring10: the 10-stage layout
        bulkCopy(s.base + (ring10 ? kMoeSmGain10 : kMoeSmGain), reinterpret_cast<void const*>(sd[kMdGainPre]),
            kMoeHidden * 2, bar(s, kMoeBarGain));               // ring stage 9 with kMoeFlagE3Ring10
        if (warmTma)
        {
            // One box through each stack's tensor map (expert 0, rows 0-127, k-block 0; the same box in every CTA)
            // into the FC1 B area, which the quantization overwrites after waiting on this barrier: the launch's
            // first tensor copy per map is not one the weight stream waits for.
            uint64_t const policy = policyEvictFirst();
            uint32_t const xw = s.base + (ring10 ? kMoeSmXH10 : kMoeSmX); // the launch's X area
            tmaLoad2d(xw, reinterpret_cast<void const*>(sd[kMdTmapW13]), 0, 0, bar(s, kMoeBarGain), policy);
            tmaLoad2d(xw, reinterpret_cast<void const*>(sd[kMdTmapW2]), 0, 0, bar(s, kMoeBarGain), policy);
        }
    }
#if MSO_MK_RT & 32
    // ---- MSO_MK_RT bit 32: the window's per-expert scalars into shared memory (waited before the post-route
    // barrier) ----
    if (tid < kMoeWindow)
    {
        uint32_t const dst = s.redS + kMoeRdRtScal + 4 * tid;
        float const* const src[3] = {dptr<float const>(s, kMdAlpha1) + tid, dptr<float const>(s, kMdScaleC) + tid,
            dptr<float const>(s, kMdAlpha2) + tid};
#pragma unroll
        for (int i = 0; i < 3; ++i)
        {
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(dst + 4 * kMoeWindow * i), "l"(src[i])
                         : "memory");
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    }
#endif
    if (icMoeFlags(sd[kMdFlags]) & kMoeFlagWarmTwice)
    {
        // Attribution only: routing, placement, cut and quantization once without requests, then again for real (the
        // second pass runs warm code); the outputs are the same.
        route(ctx, s);
        __syncthreads();
        place(ctx, s);
        {
            int const E0 = rd<int const>(s, kMoeRdScal)[kMoeScE];
            Cut c0;
            makeCut(c0, E0, (icMoeFlags(sd[kMdFlags]) & kMoeFlagTileCut) != 0, static_cast<int>(blockIdx.x),
                (icMoeFlags(sd[kMdFlags]) & kMoeFlagPxQueue) != 0); // kMoeFlagPxQueue: the stock queue
            if (E0 > 0)
            {
                quantize(ctx, s, c0, 0); // no early requests (nEarly 0): either cut
            }
        }
        __syncthreads();
        if (tid < 16)
        {
            pres[tid] = 0;
        }
        if (tid >= 32 && tid < 32 + kMoeMaxSlots)
        {
            rd<uint32_t>(s, kMoeRdSlotMask)[tid - 32] = 0;
        }
        __syncthreads();
        if (tid == 0)
        {
            moeStamp(ctx, kMoeStWarm);
        }
    }
#if MSO_MK_CX
    // ---- MSO_MK_CX (attribution only): a valid slot's routing state into the reduction area, before any reader ----
    if (cxUse)
    {
        if (tid < kCxImgVecs)
        {
            p4a::sts128(s.redS + kMoeRdSlotExpert + 16 * tid, cxV);
        }
        if (tid == 0)
        {
            rd<int>(s, kMoeRdScal)[kMoeScE] = cxE;
        }
        __syncthreads();
    }
    bool const cxEarlyOrder = cxUse && cxLevel == 1; // level 1: the early requests before the quantization
    // ---- end MSO_MK_CX ----
#endif
    bool const localQuant = (icMoeFlags(sd[kMdFlags]) & kMoeFlagLocalQuant) != 0;
    uint8_t* ws = dptr<uint8_t>(s, kMdWs);
    uint8_t* wsCur = ws + static_cast<size_t>(ctx.copy) * kMoeWsBytes;
    uint8_t* wsNext = ws + static_cast<size_t>(MSO_MK_COPY_NEXT(ctx.copy, ctx.numCopies)) * kMoeWsBytes;
    // ---- kMoeFlagQuantFirst quantizes the pooled input before the routing (while the logits are in flight) ----
    bool const quantFirst = !localQuant && (icMoeFlags(sd[kMdFlags]) & kMoeFlagQuantFirst) != 0;
    bool pre = false; // kMoeFlagW2LogitsPre: route() skips its poll
#if MSO_MK_RP & 1
    // MSO_MK_RP: the publisher CTA kCtas - 1 routes first; it quantizes no share (its gain wait follows its publish)
    bool const rpPub = static_cast<int>(blockIdx.x) == kCtas - 1;
#endif
#if MSO_MK_XP & 4
    // MSO_MK_XP bit 4: a CTA other than the publisher that quantizes no group (bit 1 / 2: the non-owners; else every
    // one) polls the state first, then takes stageCut's IssueAtCut path (cut -> early requests -> its shareQuant call
    // -> gatherX)
#if MSO_MK_XP & 3
    bool const xpSf = quantFirst && !rpPub
        && static_cast<unsigned>(static_cast<int>(blockIdx.x) - kXpOwner0) >= static_cast<unsigned>(kXpPerRow * kTok);
#else
    bool const xpSf = quantFirst && !rpPub;
#endif
#endif
#if MSO_MK_CX && (MSO_MK_RP & 1)
    if (quantFirst && !cxEarlyOrder && !rpPub) // MSO_MK_CX with MSO_MK_RP
#elif MSO_MK_CX
    if (quantFirst && !cxEarlyOrder) // MSO_MK_CX: level 1 quantizes after the early requests (the IssueAtCut path)
#elif (MSO_MK_RP & 1) && (MSO_MK_XP & 4)
    if (quantFirst && !rpPub && !xpSf) // MSO_MK_XP bit 4: its share in stageCut's IssueAtCut path
#elif MSO_MK_RP & 1
    if (quantFirst && !rpPub) // MSO_MK_RP: the publisher's call follows its publish
#else
    if (quantFirst)
#endif
    {
        pre = shareQuant(ctx, s, wsCur); // pre: kMoeFlagW2LogitsPre
    }
    // ---- end kMoeFlagQuantFirst ----
#if MSO_MK_RL & 8
    if (rpPub && kRlOk && (ctx.p[kPIFlags] & kIFlagRlOn)) // MSO_MK_RL: the publisher forms the logits (moe_hops.cuh)
    {
        rlPublisherLogits(ctx);
    }
#endif
#if MSO_MK_XP & 8
    if (rpPub)
    {
        xpWarmPublisher(ctx, s); // MSO_MK_XP bit 8: the publisher's warm-up pass before its logit poll
    }
#endif
#if MSO_MK_CX && (MSO_MK_RP & 1)
    if (!cxUse && rpPub) // MSO_MK_CX with MSO_MK_RP: only the publisher routes (unless its slot is valid)
#elif MSO_MK_CX
    if (!cxUse)                      // MSO_MK_CX: a valid slot holds route()'s results
#elif MSO_MK_RP & 1
    if (rpPub)                         // MSO_MK_RP: only the publisher routes
#endif
#if MSO_MK_RT & 8
        route(ctx, s, pre,
            quantFirst ? wsCur : nullptr); // MSO_MK_RT bit 8: the X gather around the picks with QuantFirst
#else
        route(ctx, s, pre);          // pre: kMoeFlagW2LogitsPre
#endif
#if MSO_MK_RT & 32
    if (tid < kMoeWindow)
    {
        asm volatile("cp.async.wait_all;" ::: "memory"); // MSO_MK_RT bit 32: window scalars, visible after the barrier
    }
#endif
    __syncthreads();
    if (tid == 0)
    {
        moeStamp(ctx, kMoeStTop8);
    }
    // kMoeFlagIssueAtCut: the early weight requests right after the cut, then the quantization
    bool const issueAtCut = !localQuant && !quantFirst && (icMoeFlags(sd[kMdFlags]) & kMoeFlagIssueAtCut) != 0;
    bool const quantAtCut
        = !localQuant && !quantFirst && !issueAtCut && (icMoeFlags(sd[kMdFlags]) & kMoeFlagQuantAtCut) != 0;
    if (!localQuant && !quantFirst && !issueAtCut && !quantAtCut) // default: the quantization right after the routing
    {
        shareQuant(ctx, s, wsCur);
    }
#if MSO_MK_RT & 2
    // ---- MSO_MK_RT bit 2: the placement without CTA barriers (kMoeFlagW2PlaceWarp is not taken); bit 4 with the
    // stock queue: the early units are requested inside, and the stage's role barrier is the first one after it ----
    bool const rtEarly = (MSO_MK_RT & 4) != 0 && (icMoeFlags(sd[kMdFlags]) & kMoeFlagPxQueue) != 0 && !localQuant
        && !issueAtCut && !quantAtCut && (icMoeFlags(sd[kMdFlags]) & kMoeFlagPiece0First) == 0;
    RtPlace const rp = rtPlace(ctx, s, rtEarly);
    if (!rtEarly)
    {
        __syncthreads();
    }
    if (tid == 0)
    {
        moeStamp(ctx, kMoeStRouted);
    }
    int const E = rp.E;
#else
#if MSO_MK_RP & 1
    if (!rpPub)
    {
        rpPoll(ctx, s, wsCur); // MSO_MK_RP: the publisher's tables and E, in place of place()
    }
    else
#endif
#if MSO_MK_CX
        if (cxUse)
    {
        // MSO_MK_CX: a valid slot holds place()'s results (stored before the barrier above)
    }
    else
#endif
        // ---- kMoeFlagW2PlaceWarp ----
        if (icMoeFlags(sd[kMdFlags]) & kMoeFlagW2PlaceWarp)
        {
            if (warp == 0)
            {
                placeWarp(ctx, s);
            }
            __syncthreads();
            uint8_t* dbg = dptr<uint8_t>(s, kMdDbg);
            if (dbg != nullptr && blockIdx.x == 0 && tid <= kMoeMaxSlots)
            {
                int const E0 = rd<int const>(s, kMoeRdScal)[kMoeScE];
                reinterpret_cast<int*>(dbg + kMoeDbgSlots)[tid]
                    = tid == 0 ? E0 : (tid - 1 < E0 ? rd<int const>(s, kMoeRdSlotExpert)[tid - 1] : -1);
            }
        }
        else
        {
            place(ctx, s);
        }
    // ---- end kMoeFlagW2PlaceWarp ----
    if (tid == 0)
    {
        moeStamp(ctx, kMoeStRouted);
    }
    int const E = rd<int const>(s, kMoeRdScal)[kMoeScE];
#if MSO_MK_RP & 1
    if (rpPub)
    {
        // ---- MSO_MK_RP: publish, then the publisher's quantization call (no group: its gain wait and barrier) ----
        rpPublish(s, wsCur, E);
        if (quantFirst)
        {
            shareQuant(ctx, s, wsCur);
        }
    }
#endif
#if MSO_MK_CX
    // ---- MSO_MK_CX (attribution only): a recording launch's CTA 0 stores its tables (place() ended with a barrier);
    // a rank with E = 0 never validates its slot (CTA 0 then has no gatherX, so the stage-end rule below would not
    // hold) ----
#if MSO_MK_RP & 1
    if (cxSlot != nullptr && !cxValid && blockIdx.x == 0) // MSO_MK_RP: CTA 0's tables come from the publisher's area
#else
    if (cxSlot != nullptr && !cxUse && blockIdx.x == 0)
#endif
    {
        if (tid < kCxImgVecs)
        {
            cxSt(cxSlot + kCxImgOff + 16 * tid, p4a::lds128(s.redS + kMoeRdSlotExpert + 16 * tid));
        }
        if (tid == kCxImgVecs)
        {
            cxSt(cxSlot + 16, make_uint4(static_cast<uint32_t>(E), 0u, 0u, 0u));
        }
        __threadfence();
    }
    // ---- end MSO_MK_CX ----
#endif
#endif // MSO_MK_RT bit 2
    uint32_t const tmem = rd<uint32_t const>(s, kMoeRdScal)[kMoeScTmem];
    // ---- the cut to the end of the warp roles, per cut (stageCut) ----
#if MSO_MK_RT & 4
    if (icMoeFlags(sd[kMdFlags]) & kMoeFlagPxQueue)
    {
        stageCutPx(ctx, s, sd, E, tmem, localQuant, issueAtCut, quantAtCut, wsCur, wsNext, rtEarly, rp);
    }
    else
    {
        stageCut<false>(ctx, s, sd, E, tmem, localQuant, issueAtCut, quantAtCut, wsCur, wsNext, false, rp);
    }
#elif MSO_MK_CX
    // MSO_MK_CX (attribution only): level 1 takes stageCut's IssueAtCut path (cut -> early requests -> quantization
    // -> gather)
    if (icMoeFlags(sd[kMdFlags]) & kMoeFlagPxQueue)
    {
        stageCutPx(ctx, s, sd, E, tmem, localQuant, issueAtCut || cxEarlyOrder, quantAtCut, wsCur, wsNext, cxEarly);
    }
    else
    {
        stageCut<false>(
            ctx, s, sd, E, tmem, localQuant, issueAtCut || cxEarlyOrder, quantAtCut, wsCur, wsNext, cxEarly);
    }
#elif MSO_MK_XP & 4
    // MSO_MK_XP bit 4: the state-first CTAs take stageCut's IssueAtCut path
    if (icMoeFlags(sd[kMdFlags]) & kMoeFlagPxQueue)
    {
        stageCutPx(ctx, s, sd, E, tmem, localQuant, issueAtCut || xpSf, quantAtCut, wsCur, wsNext);
    }
    else
    {
        stageCut<false>(ctx, s, sd, E, tmem, localQuant, issueAtCut || xpSf, quantAtCut, wsCur, wsNext);
    }
#else
    if (icMoeFlags(sd[kMdFlags]) & kMoeFlagPxQueue)
    {
        stageCutPx(ctx, s, sd, E, tmem, localQuant, issueAtCut, quantAtCut, wsCur, wsNext);
    }
    else
    {
        stageCut<false>(ctx, s, sd, E, tmem, localQuant, issueAtCut, quantAtCut, wsCur, wsNext);
    }
#endif
    // ---- end per-cut stage ----
    tcFenceBefore();
    __syncthreads();
    tcFenceAfter();
#if MSO_MK_CX
    // ---- MSO_MK_CX (attribution only): validate a recorded slot. Every CTA has passed its stage entry by now: CTA
    // 0's gatherX (E > 0) polled every CTA's quantization share, which each CTA publishes after its slot loads ----
#if MSO_MK_RP & 1
    if (cxSlot != nullptr && !cxValid && blockIdx.x == 0 && tid == 0 && E > 0) // MSO_MK_CX with MSO_MK_RP
#else
    if (cxSlot != nullptr && !cxUse && blockIdx.x == 0 && tid == 0 && E > 0)
#endif
    {
        __threadfence();
        asm volatile("st.global.cg.u32 [%0], %1;" ::"l"(cxSlot + 4), "r"(cxWant) : "memory");
    }
    // ---- end MSO_MK_CX ----
#endif
    if (warp == 1)
    {
        tmemDealloc(tmem, kMoeTmemCols);
    }
    if (unit > 0)
    {
#if MSO_MK_ET & 6
        if (icMoeFlags(sd[kMdFlags]) & kMoeFlagPxQueue)
        {
            finalUnitPx(ctx, s, unit - 1, wsCur); // MSO_MK_ET bits 2 / 4: the stock queue's final unit
        }
        else
#endif
            finalUnit(ctx, s, unit - 1, wsCur);
    }
}
