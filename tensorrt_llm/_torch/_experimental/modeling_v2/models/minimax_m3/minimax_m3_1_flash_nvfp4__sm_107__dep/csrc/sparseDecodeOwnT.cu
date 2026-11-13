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

// OWN-K: MiniMax-M3 block-sparse decode over the packed-NVFP4 paged cache with the KeepsAb M = 128 query tile (a
// request's 8 MTP rows x 16 heads), each union block's K / V dequantized once.
//
// Work: the union of the blocks a (request, KV head)'s 8 MTP rows selected, split into equal weighted shares per CTA
// (a block weighs kWBlock + kWUnit per row that selected it, and each item adds kWSeg; a block goes to the CTA whose
// share holds its start). Per union block x: K_x and V_x are dequantized once into smem; S = Q . K_x^T (M 128
// queries, N 128 keys) into S buffer x % 2; softmax WG g takes keys 64 g .. 64 g + 63 with one query per thread (rows
// that did not select the block get P = 0); O_g += P_g . V_x over those keys (P in TMEM). The segment epilogue combines
// the two WGs' O, m and l.
// Main-kernel roles, 20 warps:
//   warps 0-3 / 4-7   softmax WG 0 / 1 (each block's first / last 64 keys): per thread, the max, a lazy rescale of O_g
//                     and l when it exceeds the stale max by 2^8, ex2, l, P (FP16) over S; both run each epilogue;
//   warps 10, 11, 14, 15 K transform (+ each segment's Q tile), on sub-partitions 2 and 3: smem packed row ->
//                        FP16 -> the K tile;
//   warps 8, 9, 12, 13   V transform, on sub-partitions 0 and 1: smem packed row -> FP16 -> the V tile;
//   (each transform's code then fills only its own sub-partitions' instruction caches; within a role, warp
//   8 + 4 b2 + 2 b1 + b0 takes rows 32 (b0 + 2 b2) .. + 31);
//   warp 16           one thread issuing tcgen05 kind::f16: QK of block x, then PV of block x - 1; also allocates
//                     TMEM;
//   warp 17           one thread issuing cp.async.bulk of each block's packed K / V + scales into a 3-stage ring;
//   warps 18-19       idle.
// The merges wait per item, on the range its last CTA publishes once every CTA covering it has written its
// segment, so a finished item's merge runs during the main kernel's tail.
// Numerics: NVFP4 K / V dequantized to FP16 (exact), E4M3 q widened to FP16, QK and PV on FP16 tensor cores with FP32
// accumulation, a base-2 online softmax. Not bitwise LC2: both stay within the float64-reference band.

#include "minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/csrc/sparseDecodeOwnT.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

// tcgen05, setmaxnreg and cp.async.bulk need an arch- or family-specific SM10x target. Every other architecture of
// the build compiles the kernels to empty bodies, which minimaxM3Sm107SparseDecodeOwnTConfigure refuses.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && (__CUDA_ARCH__ < 1100)                                        \
    && (defined(__CUDA_ARCH_SPECIFIC__) || defined(__CUDA_ARCH_FAMILY_SPECIFIC__)                                      \
        || defined(__CUDA_ARCH_FEAT_SM100_ALL))
#define MINIMAX_M3_OWN_T_ENABLED 1
#else
#define MINIMAX_M3_OWN_T_ENABLED 0
#endif

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

using Params = MinimaxM3Sm107SparseDecodeOwnTParams;

constexpr int kMaxItems = kMinimaxM3Sm107SparseDecodeOwnTMaxItems;

constexpr int kHeads = 16;
constexpr int kDim = 128;
constexpr int kPage = 128;
constexpr int kTopk = 16;
constexpr int kRows = kMinimaxM3Sm107SparseDecodeOwnTRows;
constexpr int kQ = kRows * kHeads;
constexpr int kU = kRows * kTopk;
constexpr int kWarps = 20;
constexpr int kThreads = kWarps * 32;
constexpr int kMaxSegs = 16;
constexpr int kMaxBlocks = 256;
// Items whose span meets a share: all own a block start there but at most one, the item holding the share's start.
constexpr int kMaxCands = kMaxSegs + 1;
// Work-partition weights: a union block costs the same whichever of its rows selected it (both WGs take all of its
// keys for every query; a warp whose rows did not select it only zeroes its P), and a segment about four blocks (its
// epilogue). kMaxBlockW bounds one block's weight.
constexpr int kWUnit = 0;
constexpr int kWBlock = 60;
constexpr int kWSeg = 240;
constexpr int kMaxBlockW = kWBlock + kWUnit * kRows;
// Fixed block weights (kWUnit 0): an item's block j starts kWBlock j after the item, which weighs kWBlock count + kWSeg
// (0 without blocks), so the fast plan derives own_t_union's u_wpre / u_wt instead of loading them.
constexpr bool kFixedBlockWeights = kWUnit == 0;
// plan()'s 32-item chunks per lane for up to 64 items (16 requests x 4 KV heads); larger launches take kMaxItems / 32.
constexpr int kSmallPlanPer = 2;
constexpr float kNegInf = -__builtin_huge_valf();
constexpr float kRescaleLog2 = 8.0F;

// setmaxnreg budgets. setmaxnreg.inc only draws on registers released by setmaxnreg.dec, so the five warps of an SMSP
// (one per role group) must fit the launch allocation; minimaxM3Sm107SparseDecodeOwnTConfigure refuses any other
// register count.
constexpr int kRegLaunch = 96;
constexpr int kRegSoftmax = 144;
constexpr int kRegTransform = 64;
constexpr int kRegAux = 64;
static_assert(2 * kRegSoftmax + 2 * kRegTransform + kRegAux <= 5 * kRegLaunch, "setmaxnreg pool");
static_assert(kWarps == 20 && kRegLaunch * kThreads <= 65536, "launch allocation");

constexpr int kPkStages = 3;
constexpr int kPkBytes = 18432; // per stage: K 8192 | K scales 1024 | V 8192 | V scales 1024
constexpr int kPkKsc = 8192;
constexpr int kPkV = 9216;
constexpr int kPkVsc = 17408;
// FP16 tiles in the no-swizzle canonical layout, 128 rows x 128 columns: element (row n, column d) at
// (n / 8) 2048 + (d / 8) 128 + (n % 8) 16 + (d % 8) 2.
constexpr int kSmemPk = 0;
constexpr int kSmemK = kSmemPk + kPkStages * kPkBytes; // 2 x 32768: K (QK B, K-major: row = key, column = dim)
constexpr int kSmemV = kSmemK + 2 * 32768;             // 2 x 32768: V (PV B, MN-major: row = key, column = dim)
constexpr int kSmemQ = kSmemV + 2 * 32768;             // 32768: Q (QK A, K-major: row = query, column = dim)
constexpr int kSmemML = kSmemQ + kQ * kDim * 2;        // [2 WG][128 queries] (m, l) for the epilogue
constexpr int kSmemSeg = kSmemML + 2 * kQ * 8;
constexpr int kSmemBlk = kSmemSeg + kMaxSegs * 40;
constexpr int kSmemBar = kSmemBlk + kMaxBlocks * 16;
constexpr int kSmemMisc = kSmemBar + 512;
constexpr int kSmemBytes = kSmemMisc + 64;
static_assert(kSmemBytes <= 227 * 1024, "the main kernel's dynamic shared memory exceeds the SM10x opt-in limit");

constexpr uint32_t kTmemCols = 512;
// Lane = query (row 16 r + head). S buffer b: fp32, column = key, of the blocks x with x % 2 = b; WG g's P (FP16 pairs,
// keys 2 c, 2 c + 1 in column c) over the first 32 columns of its keys 64 g .. 64 g + 63. O_g: fp32, column = dim,
// over WG g's keys of the segment's blocks.
constexpr uint32_t kColS = 0;   // 2 buffers x 128
constexpr uint32_t kColO = 256; // 2 WG x 128

enum : int
{
    kBarPkFull = 0,
    kBarPkFree = 4,
    kBarKFull = 8,
    kBarKFree = 10,
    kBarVFull = 12,
    kBarVFree = 14,
    kBarQFull = 16,
    kBarQFree = 17,
    kBarSFull = 18, // per S buffer
    kBarPFull = 20, // per S buffer: both WGs' P
    kBarPDone = 22, // per S buffer: the block's PVs are done (P read, O_0 / O_1 written)
    kBarOReady = 24,
    kBarOFree = 25,
    kBarCount = 26
};

static_assert(8 * kBarCount <= kSmemMisc - kSmemBar, "the barriers fit their smem area");

// kind::f16: D F32, A / B F16, M 128, N 128. QK: A (Q) and B (K) K-major in smem. PV: A (P) in TMEM, B (V) MN-major.
constexpr uint32_t kIdescQK = (1U << 4) | (16U << 17) | (8U << 24);
constexpr uint32_t kIdescPV = kIdescQK | (1U << 16);

struct Seg
{
    int item, a, n, split, seg_id, req, kvh, seq;
    int cf, nsg; // the item's CTAs: cf .. cf + nsg - 1
};

static_assert(kMaxCands * (kU * 16 + static_cast<int>(sizeof(Seg))) <= 2 * 32768, "plan()'s staging fits the K tile");

static_assert(sizeof(Seg) == 40, "the segment table's smem stride");

struct Ws
{
    int* u_count;
    int* u_start;
    int* u_page;
    int* u_blk;
    uint8_t* u_mask;
    int* u_wpre; // [items][kU]: the weight of the item's union blocks before block j
    int* u_wt;   // [items]: the item's weight on the work line (0 without blocks)
    float* part_o;
    float* part_m;
    float* part_l;
    int* u_arrive;              // [items]: the item's CTAs whose outputs are written (zeroed by the union kernel)
    unsigned long long* u_desc; // [items]: the item's CTA range for the merge (see pack_desc)
};

size_t align256(size_t x)
{
    return (x + 255) & ~static_cast<size_t>(255);
}

// With base 0 the pointers are the byte offsets.
Ws carve(void* base, int items, int grid, size_t* total)
{
    Ws w;
    uintptr_t const b = reinterpret_cast<uintptr_t>(base);
    size_t off = 0;
    auto take = [&](size_t bytes)
    {
        auto* p = reinterpret_cast<uint8_t*>(b + off);
        off += align256(bytes);
        return p;
    };
    int64_t const segs = static_cast<int64_t>(grid) + items;
    w.u_count = reinterpret_cast<int*>(take(sizeof(int) * items));
    w.u_start = reinterpret_cast<int*>(take(sizeof(int) * (items + 1)));
    w.u_page = reinterpret_cast<int*>(take(sizeof(int) * items * kU));
    w.u_blk = reinterpret_cast<int*>(take(sizeof(int) * items * kU));
    w.u_mask = take(static_cast<size_t>(items) * kU);
    w.u_wpre = reinterpret_cast<int*>(take(sizeof(int) * items * kU));
    w.u_wt = reinterpret_cast<int*>(take(sizeof(int) * items));
    w.part_o = reinterpret_cast<float*>(take(sizeof(float) * segs * kQ * kDim));
    w.part_m = reinterpret_cast<float*>(take(sizeof(float) * segs * kQ));
    w.part_l = reinterpret_cast<float*>(take(sizeof(float) * segs * kQ));
    w.u_arrive = reinterpret_cast<int*>(take(sizeof(int) * items));
    w.u_desc = reinterpret_cast<unsigned long long*>(take(sizeof(unsigned long long) * items));
    if (total != nullptr)
        *total = off;
    return w;
}

__device__ __forceinline__ uint32_t smem_u32(void const* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint32_t a, uint32_t count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(count));
}

__device__ __forceinline__ void mbar_arrive(uint32_t a)
{
    asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared::cta.b64 st, [%0];\n}\n" ::"r"(a) : "memory");
}

__device__ __forceinline__ void mbar_expect_tx(uint32_t a, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(bytes) : "memory");
}

__device__ __forceinline__ bool mbar_try(uint32_t a, uint32_t parity)
{
    uint32_t ok;
    asm volatile(
        "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
        : "=r"(ok)
        : "r"(a), "r"(parity)
        : "memory");
    return ok != 0;
}

// A plain try-wait loop. A diagnostic build (-DMINIMAX_M3_OWN_T_HANG_TRAP) bounds it instead: a wait not complete after
// 2^24 polls names its barrier (tag) and traps rather than hang the stream.
__device__ __forceinline__ void mbar_wait(uint32_t a, uint32_t parity, [[maybe_unused]] int tag)
{
#ifdef MINIMAX_M3_OWN_T_HANG_TRAP
    uint32_t spins = 0;
    while (!mbar_try(a, parity))
    {
        if (++spins == (1U << 24))
        {
            printf("OWNT_HANG tag %d cta %d thread %d parity %u\n", tag, static_cast<int>(blockIdx.x),
                static_cast<int>(threadIdx.x), parity);
            __trap();
        }
    }
#else
    while (!mbar_try(a, parity))
    {
    }
#endif
}

__device__ __forceinline__ void bulk_g2s(uint32_t dst, void const* src, uint32_t bytes, uint32_t bar)
{
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
                 "l"(src), "r"(bytes), "r"(bar)
                 : "memory");
}

__device__ __forceinline__ void tc_fence_before()
{
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tc_fence_after()
{
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void fence_async_smem()
{
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void tc_commit(uint32_t bar)
{
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(bar) : "memory");
}

__device__ __forceinline__ void mma_ts(uint32_t d, uint32_t a, uint64_t bdesc, uint32_t idesc, uint32_t acc)
{
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n" ::
            "r"(d),
        "r"(a), "l"(bdesc), "r"(idesc), "r"(acc));
}

__device__ __forceinline__ void mma_ss(uint32_t d, uint64_t adesc, uint64_t bdesc, uint32_t idesc, uint32_t acc)
{
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n" ::
            "r"(d),
        "l"(adesc), "l"(bdesc), "r"(idesc), "r"(acc));
}

// No-swizzle canonical layout: core matrices of 8 rows x 16 bytes; LBO / SBO in bytes.
__device__ __forceinline__ uint64_t smem_desc(uint32_t addr, uint32_t lbo, uint32_t sbo)
{
    uint64_t d = static_cast<uint64_t>((addr & 0x3FFFF) >> 4);
    d |= static_cast<uint64_t>(lbo >> 4) << 16;
    d |= static_cast<uint64_t>(sbo >> 4) << 32;
    d |= static_cast<uint64_t>(1) << 46;
    return d;
}

__device__ __forceinline__ void tmem_st16(uint32_t taddr, uint32_t const* v)
{
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};" ::"r"(
            taddr),
        "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]), "r"(v[4]), "r"(v[5]), "r"(v[6]), "r"(v[7]), "r"(v[8]), "r"(v[9]),
        "r"(v[10]), "r"(v[11]), "r"(v[12]), "r"(v[13]), "r"(v[14]), "r"(v[15]));
}

__device__ __forceinline__ void tmem_ld16(uint32_t taddr, uint32_t* r)
{
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
        : "r"(taddr));
}

__device__ __forceinline__ void tmem_ld32(uint32_t taddr, uint32_t* r)
{
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,"
        "%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]),
        "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]),
        "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
        : "r"(taddr));
}

__device__ __forceinline__ void tmem_wait_ld()
{
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_wait_st()
{
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void wg_bar(int id)
{
    asm volatile("bar.sync %0, 128;" ::"r"(id) : "memory");
}

// Both softmax WGs (warps 0-7).
__device__ __forceinline__ void pair_bar()
{
    asm volatile("bar.sync 3, 256;" ::: "memory");
}

__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint16_t pair)
{
    uint32_t out;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(out) : "h"(pair));
    return out;
}

__device__ __forceinline__ uint32_t scale_pair(uint32_t byte)
{
    return e4m3x2_to_f16x2(static_cast<uint16_t>(byte | (byte << 8)));
}

// 8 E2M1 values (4 bytes, low nibble = even dim) -> 4 f16x2, scaled (ADD31's byte-select form; exact).
__device__ __forceinline__ void dq8(uint32_t w, uint32_t s2, uint32_t* o)
{
    asm("{\n .reg .b8 b0, b1, b2, b3;\n .reg .b32 t0, t1, t2, t3;\n mov.b32 {b0, b1, b2, b3}, %4;\n"
        " cvt.rn.f16x2.e2m1x2 t0, b0;\n cvt.rn.f16x2.e2m1x2 t1, b1;\n cvt.rn.f16x2.e2m1x2 t2, b2;\n"
        " cvt.rn.f16x2.e2m1x2 t3, b3;\n mul.rn.f16x2 %0, t0, %5;\n mul.rn.f16x2 %1, t1, %5;\n"
        " mul.rn.f16x2 %2, t2, %5;\n mul.rn.f16x2 %3, t3, %5;\n}"
        : "=r"(o[0]), "=r"(o[1]), "=r"(o[2]), "=r"(o[3])
        : "r"(w), "r"(s2));
}

__device__ __forceinline__ float ex2(float x)
{
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// An item's CTA range for the merge: cf in the low 32 bits, cl in the high; kDescUnset until it is published,
// kDescEmpty for an item without blocks.
constexpr uint32_t kDescUnset = 0xFFFFFFFFU;
constexpr uint32_t kDescEmpty = 0xFFFFFFFEU;

__device__ __forceinline__ unsigned long long pack_desc(int cf, int cl)
{
    return (static_cast<unsigned long long>(static_cast<uint32_t>(cl)) << 32) | static_cast<uint32_t>(cf);
}

__device__ __forceinline__ unsigned long long ld_acquire_u64(unsigned long long const* a)
{
    unsigned long long v;
    asm volatile("ld.acquire.gpu.global.b64 %0, [%1];" : "=l"(v) : "l"(a) : "memory");
    return v;
}

__device__ __forceinline__ void st_release_u64(unsigned long long* a, unsigned long long v)
{
    asm volatile("st.release.gpu.global.b64 [%0], %1;" ::"l"(a), "l"(v) : "memory");
}

// Spins until the item's range is published. A wait past ~1 s means a lost arrival: it traps.
__device__ __forceinline__ unsigned long long wait_desc(unsigned long long const* a)
{
    unsigned long long d = ld_acquire_u64(a);
    for (int spin = 0; static_cast<uint32_t>(d) == kDescUnset; ++spin)
    {
        if (spin == (1 << 23))
        {
            printf("OWNT_MERGE_WAIT item %d\n", static_cast<int>(blockIdx.x));
            __trap();
        }
        __nanosleep(128);
        d = ld_acquire_u64(a);
    }
    return d;
}

// A CTA's arrival for a segment its epilogue has written (the wait on the epilogue's OFree orders those stores
// before this thread's fence): the item's last CTA publishes the item's range.
__device__ __forceinline__ void arrive_item(Ws const& ws, Seg const& sg)
{
    __threadfence();
    if (atomicAdd(ws.u_arrive + sg.item, 1) == sg.nsg - 1)
    {
        __threadfence();
        st_release_u64(ws.u_desc + sg.item, pack_desc(sg.cf, sg.cf + sg.nsg - 1));
    }
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi)
{
    __nv_bfloat162 const h = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t const*>(&h);
}

// The max of a 32-key S chunk (keys base .. base + 31; with MASKED only those below nvalid) into 4 accumulators.
template <bool MASKED>
__device__ __forceinline__ void max_chunk(uint32_t const (&s)[32], int base, int nvalid, float (&bm)[4])
{
#pragma unroll
    for (int i = 0; i < 32; ++i)
    {
        float const v = __uint_as_float(s[i]);
        if (MASKED)
            bm[i & 3] = base + i < nvalid ? fmaxf(bm[i & 3], v) : bm[i & 3];
        else
            bm[i & 3] = fmaxf(bm[i & 3], v);
    }
}

// A 32-key chunk of P: p = 2^(s ct + mneg) (0 at keys from nvalid on with MASKED), summed into 4 accumulators, as
// FP16 pairs (the even key in the low half).
template <bool MASKED>
__device__ __forceinline__ void p_chunk(
    uint32_t const (&s)[32], float ct, float mneg, int base, int nvalid, float (&l)[4], uint32_t (&pk)[16])
{
#pragma unroll
    for (int i = 0; i < 32; i += 2)
    {
        float p0 = ex2(fmaf(__uint_as_float(s[i]), ct, mneg));
        float p1 = ex2(fmaf(__uint_as_float(s[i + 1]), ct, mneg));
        if (MASKED)
        {
            p0 = base + i < nvalid ? p0 : 0.F;
            p1 = base + i + 1 < nvalid ? p1 : 0.F;
        }
        l[(i >> 1) & 3] += p0 + p1;
        __half2 const hp = __floats2half2_rn(p0, p1);
        pk[i >> 1] = *reinterpret_cast<uint32_t const*>(&hp);
    }
}

// One block of a softmax thread over its WG's 64 keys kb .. kb + 63 of the block (its query's S half-row at s_col, its
// O_g row at o_col) in a warp holding a selected row: the max over the row's keys (with MASKED only those below
// nvalid); when it exceeds the stale max by 2^8, O_g and l are rescaled once the previous block's PV is done (pdone at
// parity); then P over the half-row's first 32 columns and its sum into l. A thread whose row did not select the
// block (sel 0) writes P = 0 and keeps m and l.
template <bool MASKED>
__device__ __forceinline__ void softmax_block(uint32_t s_col, uint32_t o_col, int kb, float cs, uint32_t sel,
    int nvalid, uint32_t pdone, uint32_t parity, float& m_run, float& l_run)
{
    uint32_t s0[32];
    uint32_t s1[32];
    tmem_ld32(s_col, s0);
    tmem_ld32(s_col + 32, s1);
    tmem_wait_ld();
    float bmx[4] = {kNegInf, kNegInf, kNegInf, kNegInf};
    max_chunk<MASKED>(s0, kb, nvalid, bmx);
    max_chunk<MASKED>(s1, kb + 32, nvalid, bmx);
    float const bm = fmaxf(fmaxf(bmx[0], bmx[1]), fmaxf(bmx[2], bmx[3]));
    float const mc = (sel != 0U && bm != kNegInf) ? bm * cs : kNegInf;
    bool const up = mc > m_run + kRescaleLog2;
    bool const scale = up && m_run != kNegInf;
    // The TMEM accesses are warp-collective: the warp rescales together, f = 1 for its threads that need none.
    if (__any_sync(0xffffffffU, scale))
    {
        float const f = scale ? ex2(m_run - mc) : 1.F;
        mbar_wait(pdone, parity, 10);
        tc_fence_after();
        // 16 columns at a time: the S chunks stay live across this path.
#pragma unroll 1
        for (int c = 0; c < 8; ++c)
        {
            uint32_t o[16];
            tmem_ld16(o_col + 16 * c, o);
            tmem_wait_ld();
#pragma unroll
            for (int i = 0; i < 16; ++i)
                o[i] = __float_as_uint(__uint_as_float(o[i]) * f);
            tmem_st16(o_col + 16 * c, o);
        }
        tmem_wait_st();
        l_run *= f;
    }
    if (up)
        m_run = mc;
    // A key outside the row (an unselected row, or past its length) gets p = 0.
    float const ct = sel != 0U ? cs : 0.F;
    float const mneg = (sel != 0U && m_run != kNegInf) ? -m_run : kNegInf;
    float lacc[4] = {0.F, 0.F, 0.F, 0.F};
    uint32_t pk[16];
    p_chunk<MASKED>(s0, ct, mneg, kb, nvalid, lacc, pk);
    tmem_st16(s_col, pk);
    p_chunk<MASKED>(s1, ct, mneg, kb + 32, nvalid, lacc, pk);
    tmem_st16(s_col + 16, pk);
    l_run += (lacc[0] + lacc[1]) + (lacc[2] + lacc[3]);
}

// trtllm::mxfp8_quantize's scale of a 32-value group from its amax: the UE8M0 byte, and the factor taking the group to
// E4M3 (0 for byte 0: 0 < amax <= 448 * 2^-127 quantizes to zero like an all-zero group).
__device__ __forceinline__ int mxfp8_scale(float amax, float& qs)
{
    float rcp;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(rcp) : "f"(448.0F));
    int const e = __nv_cvt_float_to_e8m0(amax * rcp, __NV_SATFINITE, cudaRoundPosInf);
    qs = amax != 0.F && e != 0 ? __int_as_float((254 - e) << 23) : 0.F;
    return e;
}

__device__ __forceinline__ uint32_t mxfp8_e4m3(float x)
{
    uint16_t q2;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(q2) : "f"(0.F), "f"(x));
    return q2 & 0xFFU;
}

// The byte of row tok's scale k in the 128x4 swizzled layout of trtllm::mxfp8_quantize(x, true).
__device__ __forceinline__ int64_t mxfp8_sf_offset(Params const& p, int tok, int k)
{
    return static_cast<int64_t>(tok / 128) * p.num_sf_k_tiles * 512 + (k / 4) * 512 + (tok % 32) * 16
        + ((tok % 128) / 32) * 4 + k % 4;
}

// The merge's MXFP8 copy of 4 BF16 values: dims 4 lane .. 4 lane + 3 of head `head` in row tok. A 32-value group is
// 8 lanes.
__device__ __forceinline__ void mxfp8_quad(Params const& p, int tok, int head, int lane, uint2 packed)
{
    __nv_bfloat162 const lo = *reinterpret_cast<__nv_bfloat162 const*>(&packed.x);
    __nv_bfloat162 const hi = *reinterpret_cast<__nv_bfloat162 const*>(&packed.y);
    float const v[4] = {__low2float(lo), __high2float(lo), __low2float(hi), __high2float(hi)};
    float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
    for (int off = 4; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffU, amax, off));
    float qs;
    int const e = mxfp8_scale(amax, qs);
    *reinterpret_cast<uint32_t*>(p.q8 + static_cast<int64_t>(tok) * (p.num_sf_k_tiles * 128) + head * kDim + 4 * lane)
        = mxfp8_e4m3(v[0] * qs) | (mxfp8_e4m3(v[1] * qs) << 8) | (mxfp8_e4m3(v[2] * qs) << 16)
        | (mxfp8_e4m3(v[3] * qs) << 24);
    if ((lane & 7) == 0)
        p.sf[mxfp8_sf_offset(p, tok, head * 4 + (lane >> 3))] = static_cast<uint8_t>(e);
}

// The CTAs that take work: every share is wider than any block, so every CTA whose share meets the span from an
// item's first block start to its last owns at least one of its blocks.
__host__ __device__ inline long long work_grid(long long total, long long grid)
{
    long long const g = total / (kMaxBlockW + 1);
    return g < 1 ? 1 : (g < grid ? g : grid);
}

// Warps 0-3: this CTA's equal share [lo, hi) of the weighted work line, its segments and block list. Item i spans
// u_wt[i] from its start; its block j starts at start_i + u_wpre[i][j] and belongs to the CTA whose share holds that
// start. The items whose span meets the share are the candidates: warp w loads candidates w, w + 4, ... (block starts
// and block data in one round trip) and stages their owned blocks in scratch; warp 0 then lays out the segments and
// the block list in item order. CTA 0 publishes the item starts for the merge. kPer = 32-item chunks per lane:
// every chunk's loads, prefix and candidate scan are unrolled, so a launch runs the instance its item count needs.
// kFast (fixed block weights only): the item and block weights are derived, not loaded; the same values.
template <int kPer, bool kFast>
__device__ void plan(Params const& p, Ws const& ws, int items, Seg* segs, int4* blocks, int* misc, uint8_t* scratch)
{
    int const warp = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
    int4* const stage = reinterpret_cast<int4*>(scratch);                     // [kMaxCands][kU]
    Seg* const cands = reinterpret_cast<Seg*>(scratch + kMaxCands * kU * 16); // [kMaxCands]
    static_assert(kPer >= 1 && kPer * 32 <= kMaxItems, "plan chunks");
    static_assert(!kFast || kFixedBlockWeights, "the fast plan derives fixed block weights only");
    int cnt[kPer];
    int wgt[kPer];
    int start[kPer];
#pragma unroll
    for (int k = 0; k < kPer; ++k)
    {
        int const i = k * 32 + lane;
        cnt[k] = i < items ? __ldcg(ws.u_count + i) : 0;
        if constexpr (kFast)
            wgt[k] = cnt[k] > 0 ? cnt[k] * kWBlock + kWSeg : 0;
        else
            wgt[k] = i < items ? __ldcg(ws.u_wt + i) : 0;
    }
    int run = 0;
#pragma unroll
    for (int k = 0; k < kPer; ++k)
    {
        int x = wgt[k];
#pragma unroll
        for (int o = 1; o < 32; o <<= 1)
        {
            int const y = __shfl_up_sync(0xffffffffU, x, o);
            if (lane >= o)
                x += y;
        }
        start[k] = run + x - wgt[k];
        run += __shfl_sync(0xffffffffU, x, 31);
    }
    long long const total = run;
    long long const grid = work_grid(total, gridDim.x);
    long long const c = blockIdx.x;
    if (blockIdx.x == 0 && warp == 0)
    {
#pragma unroll
        for (int k = 0; k < kPer; ++k)
        {
            int const i = k * 32 + lane;
            if (i < items)
            {
                ws.u_start[i] = start[k];
                if (cnt[k] == 0)
                    st_release_u64(ws.u_desc + i, pack_desc(-2, -2));
            }
        }
        if (lane == 0)
            ws.u_start[items] = run;
    }
    long long const lo = total > 0 && c < grid ? c * total / grid : 0;
    long long const hi = total > 0 && c < grid ? (c + 1) * total / grid : 0;
    int nc = 0;
#pragma unroll
    for (int k = 0; k < kPer; ++k)
    {
        int const i = k * 32 + lane;
        long long const s0 = start[k];
        bool const cand = i < items && cnt[k] > 0 && s0 < hi && s0 + wgt[k] > lo;
        unsigned b = __ballot_sync(0xffffffffU, cand);
        while (b != 0U)
        {
            int const src = __ffs(b) - 1;
            b &= b - 1U;
            int const cc = nc++;
            if ((cc & 3) != warp || cc >= kMaxCands)
                continue;
            int const it = k * 32 + src;
            long long const t0 = __shfl_sync(0xffffffffU, s0, src);
            int const n = __shfl_sync(0xffffffffU, cnt[k], src);
            int const req = it / p.num_kv_heads;
            int const sq = __ldg(p.seq_lens + req);
            int wp[kU / 32];
            int pg[kU / 32];
            int bk[kU / 32];
            int mk[kU / 32];
#pragma unroll
            for (int m = 0; m < kU / 32; ++m)
            {
                int const j = m * 32 + lane;
                int const u = it * kU + j;
                bool const in = j < n;
                if constexpr (kFast)
                    wp[m] = in ? kWBlock * j : 0;
                else
                    wp[m] = in ? __ldcg(ws.u_wpre + u) : 0;
                pg[m] = in ? __ldcg(ws.u_page + u) : 0;
                bk[m] = in ? __ldcg(ws.u_blk + u) : 0;
                mk[m] = in ? static_cast<int>(__ldcg(ws.u_mask + u)) : 0;
            }
            // The block starts ascend with j: blocks a .. e - 1 start in [lo, hi); tl is the last block's start.
            int a = 0;
            int e = 0;
            long long tl = t0;
#pragma unroll
            for (int m = 0; m < kU / 32; ++m)
            {
                int const j = m * 32 + lane;
                long long const st = j < n ? t0 + wp[m] : LLONG_MAX;
                a += __popc(__ballot_sync(0xffffffffU, st < lo));
                e += __popc(__ballot_sync(0xffffffffU, st < hi));
                long long const sm = __shfl_sync(0xffffffffU, st, (n - 1) & 31);
                if (m == ((n - 1) >> 5))
                    tl = sm;
            }
#pragma unroll
            for (int m = 0; m < kU / 32; ++m)
            {
                int const j = m * 32 + lane;
                if (j >= a && j < e)
                    stage[cc * kU + j - a] = make_int4(pg[m], bk[m], mk[m], 0);
            }
            if (lane == 0)
            {
                long long const cf = ((t0 + 1) * grid - 1) / total;
                long long const cl = ((tl + 1) * grid - 1) / total;
                Seg sg;
                sg.item = it;
                sg.a = a;
                sg.n = e - a;
                sg.split = cl > cf ? 1 : 0;
                sg.seg_id = static_cast<int>(c) + it;
                sg.req = req;
                sg.kvh = it - req * p.num_kv_heads;
                sg.seq = sq;
                sg.cf = static_cast<int>(cf);
                sg.nsg = static_cast<int>(cl - cf + 1);
                cands[cc] = sg;
            }
        }
    }
    wg_bar(1); // warps 0-3
    if (warp == 0)
    {
        // Candidates owning no block start (only an item's tail weight meets the share) make no segment.
        int nseg = 0;
        int owned = 0;
        int const ncand = nc < kMaxCands ? nc : kMaxCands;
        for (int cc = 0; cc < ncand; ++cc)
        {
            Seg const sg = cands[cc];
            if (sg.n == 0)
                continue;
            if (nseg < kMaxSegs && owned + sg.n <= kMaxBlocks)
            {
                for (int j = lane; j < sg.n; j += 32)
                {
                    int4 v = stage[cc * kU + j];
                    v.w = nseg;
                    blocks[owned + j] = v;
                }
                if (lane == 0)
                    segs[nseg] = sg;
            }
            ++nseg;
            owned += sg.n;
        }
        if (nc > kMaxCands || nseg > kMaxSegs || owned > kMaxBlocks)
        {
            if (lane == 0)
                printf("OWNT_PLAN_OVERFLOW cta %d cands %d segs %d blocks %d\n", static_cast<int>(blockIdx.x), nc, nseg,
                    owned);
            __trap();
        }
        if (lane == 0)
        {
            misc[1] = nseg;
            misc[2] = owned;
        }
    }
    // The scratch is the K tile, which only the K transform's generic stores write next, each followed by that
    // transform's own proxy fence before the MMA reads it: no fence here.
}

template <int kPlanPer, bool kFastPlan>
__global__ void __launch_bounds__(kThreads, 1) own_t_main(__grid_constant__ Params const p, Ws const ws, int items)
{
#if MINIMAX_M3_OWN_T_ENABLED
    extern __shared__ __align__(1024) uint8_t smem[];
    int const tid = threadIdx.x;
    int const warp = tid >> 5;
    int const lane = tid & 31;
    uint32_t const bar0 = smem_u32(smem + kSmemBar);
    auto bar = [&](int i) { return bar0 + 8U * static_cast<uint32_t>(i); };
    int* const misc = reinterpret_cast<int*>(smem + kSmemMisc);
    Seg* const segs = reinterpret_cast<Seg*>(smem + kSmemSeg);
    int4* const blocks = reinterpret_cast<int4*>(smem + kSmemBlk);

    if (tid == 0)
    {
        for (int i = 0; i < kPkStages; ++i)
        {
            mbar_init(bar(kBarPkFull + i), 1);
            mbar_init(bar(kBarPkFree + i), 256);
        }
        for (int i = 0; i < 2; ++i)
        {
            mbar_init(bar(kBarKFull + i), 128);
            mbar_init(bar(kBarKFree + i), 1);
            mbar_init(bar(kBarVFull + i), 128);
            mbar_init(bar(kBarVFree + i), 1);
            mbar_init(bar(kBarSFull + i), 1);
            mbar_init(bar(kBarPFull + i), 256);
            mbar_init(bar(kBarPDone + i), 1);
        }
        mbar_init(bar(kBarQFull), 128);
        mbar_init(bar(kBarQFree), 1);
        mbar_init(bar(kBarOReady), 1);
        mbar_init(bar(kBarOFree), 256);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    if (warp == 16)
    {
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(misc)), "r"(kTmemCols));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;");
    if (warp < 4)
        plan<kPlanPer, kFastPlan>(p, ws, items, segs, blocks, misc, smem + kSmemK);
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    uint32_t const tmem = static_cast<uint32_t>(misc[0]);
    int const nseg = misc[1];
    int const nblk = misc[2];

    // Each role sets its register budget inside its own branch, so no code is compiled under a mix of budgets.
    if (warp == 17)
    {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(kRegAux));
        // Producer: one block's packed K, K scales, V, V scales per ring stage.
        if (lane == 0)
        {
            for (int x = 0; x < nblk; ++x)
            {
                int const st = x % kPkStages;
                uint32_t const ph = (x / kPkStages) & 1;
                mbar_wait(bar(kBarPkFree + st), ph ^ 1, 30);
                int4 const b = blocks[x];
                Seg const& sg = segs[b.w];
                int64_t const page = b.x;
                uint32_t const dst = smem_u32(smem + kSmemPk + st * kPkBytes);
                uint32_t const full = bar(kBarPkFull + st);
                mbar_expect_tx(full, kPkBytes);
                bulk_g2s(dst, p.k + page * p.k_blk + sg.kvh * p.k_h, 8192, full);
                bulk_g2s(dst + kPkKsc, p.ksc + page * p.ksc_blk + sg.kvh * p.ksc_h, 1024, full);
                bulk_g2s(dst + kPkV, p.v + page * p.v_blk + sg.kvh * p.v_h, 8192, full);
                bulk_g2s(dst + kPkVsc, p.vsc + page * p.vsc_blk + sg.kvh * p.vsc_h, 1024, full);
            }
        }
        __syncwarp();
    }
    else if (warp >= 8 && warp < 16 && (warp & 2) != 0)
    {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(kRegTransform));
        // K transform (+ each segment's Q tile): key / query n.
        int const n = (((warp & 1) | ((warp >> 1) & 2)) << 5) + lane;
        for (int x = 0; x < nblk; ++x)
        {
            int4 const b = blocks[x];
            int const k = b.w;
            Seg const sg = segs[k];
            if (x == 0 || blocks[x - 1].w != k)
            {
                if (k > 0)
                    mbar_wait(bar(kBarQFree), (k - 1) & 1, 20);
                // Query n = (row n / 16, head n % 16): E4M3 -> FP16 into row n of the QK A tile (K-major).
                int const r = n >> 4;
                int const h = n & 15;
                uint4 const* src = reinterpret_cast<uint4 const*>(
                    p.q + (sg.req * p.dql + r) * p.q_tok + (sg.kvh * kHeads + h) * p.q_head);
                uint8_t* const qt = smem + kSmemQ + (n >> 3) * 2048 + (n & 7) * 16;
#pragma unroll
                for (int c = 0; c < 8; ++c)
                {
                    uint4 const w = __ldg(src + c);
                    uint32_t const words[4] = {w.x, w.y, w.z, w.w};
                    uint32_t d[8];
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                    {
                        d[2 * i] = e4m3x2_to_f16x2(static_cast<uint16_t>(words[i] & 0xFFFF));
                        d[2 * i + 1] = e4m3x2_to_f16x2(static_cast<uint16_t>(words[i] >> 16));
                    }
                    // dims 16 c .. 16 c + 15 = chunks 2 c, 2 c + 1
                    *reinterpret_cast<uint4*>(qt + (2 * c) * 128) = make_uint4(d[0], d[1], d[2], d[3]);
                    *reinterpret_cast<uint4*>(qt + (2 * c + 1) * 128) = make_uint4(d[4], d[5], d[6], d[7]);
                }
                fence_async_smem();
                mbar_arrive(bar(kBarQFull));
            }
            int const st = x % kPkStages;
            mbar_wait(bar(kBarPkFull + st), (x / kPkStages) & 1, 21);
            uint8_t const* const pk = smem + kSmemPk + st * kPkBytes;
            bool const valid = b.y * kPage + n < sg.seq;
            uint4 kw[4];
#pragma unroll
            for (int c = 0; c < 4; ++c)
                kw[c] = valid ? *reinterpret_cast<uint4 const*>(pk + n * 64 + 16 * c) : make_uint4(0, 0, 0, 0);
            uint2 const ksc = valid ? *reinterpret_cast<uint2 const*>(pk + kPkKsc + n * 8) : make_uint2(0, 0);
            mbar_arrive(bar(kBarPkFree + st));
            int const ks = x & 1;
            mbar_wait(bar(kBarKFree + ks), ((x >> 1) & 1) ^ 1, 22);
            uint32_t const words[16] = {kw[0].x, kw[0].y, kw[0].z, kw[0].w, kw[1].x, kw[1].y, kw[1].z, kw[1].w, kw[2].x,
                kw[2].y, kw[2].z, kw[2].w, kw[3].x, kw[3].y, kw[3].z, kw[3].w};
            // Key n -> row n of the QK B tile (K-major, the V tile's layout).
            uint8_t* const row = smem + kSmemK + ks * 32768 + (n >> 3) * 2048 + (n & 7) * 16;
#pragma unroll
            for (int sci = 0; sci < 8; ++sci)
            {
                uint32_t const scw = sci < 4 ? ksc.x : ksc.y;
                uint32_t const s2 = scale_pair((scw >> (8 * (sci & 3))) & 0xFF);
                // scale column sci = dims 16 sci .. 16 sci + 15 = packed words 2 sci, 2 sci + 1
                uint32_t o[8];
                dq8(words[2 * sci], s2, o);
                dq8(words[2 * sci + 1], s2, o + 4);
                *reinterpret_cast<uint4*>(row + (2 * sci) * 128) = make_uint4(o[0], o[1], o[2], o[3]);
                *reinterpret_cast<uint4*>(row + (2 * sci + 1) * 128) = make_uint4(o[4], o[5], o[6], o[7]);
            }
            fence_async_smem();
            mbar_arrive(bar(kBarKFull + ks));
        }
    }
    else if (warp >= 8 && warp < 16 && (warp & 2) == 0)
    {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(kRegTransform));
        // V transform: key n -> row n of the PV B tile (MN-major).
        int const n = (((warp & 1) | ((warp >> 1) & 2)) << 5) + lane;
        for (int x = 0; x < nblk; ++x)
        {
            int4 const b = blocks[x];
            Seg const& sg = segs[b.w];
            int const st = x % kPkStages;
            mbar_wait(bar(kBarPkFull + st), (x / kPkStages) & 1, 23);
            uint8_t const* const pk = smem + kSmemPk + st * kPkBytes;
            bool const valid = b.y * kPage + n < sg.seq;
            uint4 vw[4];
#pragma unroll
            for (int c = 0; c < 4; ++c)
                vw[c] = valid ? *reinterpret_cast<uint4 const*>(pk + kPkV + n * 64 + 16 * c) : make_uint4(0, 0, 0, 0);
            uint4 const* const vsp = reinterpret_cast<uint4 const*>(pk + kPkVsc + (n >> 2) * 32);
            uint4 const sw0 = valid ? vsp[0] : make_uint4(0, 0, 0, 0);
            uint4 const sw1 = valid ? vsp[1] : make_uint4(0, 0, 0, 0);
            mbar_arrive(bar(kBarPkFree + st));
            int const vs = x & 1;
            mbar_wait(bar(kBarVFree + vs), ((x >> 1) & 1) ^ 1, 24);
            uint32_t const sw[8] = {sw0.x, sw0.y, sw0.z, sw0.w, sw1.x, sw1.y, sw1.z, sw1.w};
            int const sh = 8 * (n & 3);
            uint8_t* const row = smem + kSmemV + vs * 32768 + (n >> 3) * 2048 + (n & 7) * 16;
#pragma unroll
            for (int c = 0; c < 4; ++c)
            {
                uint32_t const words[4] = {vw[c].x, vw[c].y, vw[c].z, vw[c].w};
#pragma unroll
                for (int h = 0; h < 2; ++h)
                {
                    // scale column 2 c + h = dims 32 c + 16 h .. + 15 = packed words 2 h, 2 h + 1 of chunk c
                    uint32_t const s2 = scale_pair((sw[2 * c + h] >> sh) & 0xFF);
                    uint32_t o[8];
                    dq8(words[2 * h], s2, o);
                    dq8(words[2 * h + 1], s2, o + 4);
                    int const d0 = c * 32 + h * 16;
                    *reinterpret_cast<uint4*>(row + (d0 >> 3) * 128) = make_uint4(o[0], o[1], o[2], o[3]);
                    *reinterpret_cast<uint4*>(row + ((d0 >> 3) + 1) * 128) = make_uint4(o[4], o[5], o[6], o[7]);
                }
            }
            fence_async_smem();
            mbar_arrive(bar(kBarVFull + vs));
        }
    }
    else if (warp == 16)
    {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(kRegAux));
        // MMA issue, one thread: for each block x, S buffer x % 2 = Q . K_x^T (M 128 queries, N 128 keys), then for
        // block x - 1 O_g += P_g . V over WG g's 64 keys (M 128 queries, N 128 dims; P in TMEM).
        if (lane == 0)
        {
            uint32_t const q_base = smem_u32(smem + kSmemQ);
            uint32_t const k_base = smem_u32(smem + kSmemK);
            uint32_t const v_base = smem_u32(smem + kSmemV);
            bool fresh = false;    // the segment's first PV overwrites O_0 and O_1
            bool seg_open = false; // the segment's first PV has waited for the previous segment's epilogue
            auto issue_pv = [&](int y, int k)
            {
                int const sb = y & 1;
                int const vs = y & 1;
                mbar_wait(bar(kBarVFull + vs), (y >> 1) & 1, 3);
                mbar_wait(bar(kBarPFull + sb), (y >> 1) & 1, 4);
                if (!seg_open)
                {
                    if (k > 0)
                    {
                        mbar_wait(bar(kBarOFree), (k - 1) & 1, 5);
                        arrive_item(ws, segs[k - 1]);
                    }
                    seg_open = true;
                }
                uint32_t const acc = fresh ? 0U : 1U;
                fresh = false;
                tc_fence_after();
                // O_g += P_g . V over WG g's keys 64 g .. 64 g + 63 (the V tile's rows 64 g ..).
#pragma unroll
                for (int g = 0; g < 2; ++g)
                {
#pragma unroll
                    for (int kk = 0; kk < 4; ++kk)
                    {
                        uint64_t const db = smem_desc(v_base + vs * 32768 + g * 16384 + kk * 4096, 2048, 128);
                        mma_ts(tmem + kColO + 128 * g, tmem + kColS + 128 * sb + 64 * g + 8 * kk, db, kIdescPV,
                            (acc != 0U || kk > 0) ? 1U : 0U);
                    }
                }
                tc_commit(bar(kBarPDone + sb));
                tc_commit(bar(kBarVFree + vs));
            };
            int x = 0;
            for (int k = 0; k < nseg; ++k)
            {
                int const n_k = segs[k].n;
                mbar_wait(bar(kBarQFull), k & 1, 6);
                fresh = true;
                seg_open = false;
                for (int jx = 0; jx < n_k; ++jx, ++x)
                {
                    int const sb = x & 1;
                    int const ks = x & 1;
                    mbar_wait(bar(kBarKFull + ks), (x >> 1) & 1, 7);
                    // S buffer sb holds P of block x - 2 until that block's PV has read it.
                    if (x >= 2)
                        mbar_wait(bar(kBarPDone + sb), ((x >> 1) & 1) ^ 1, 8);
                    tc_fence_after();
#pragma unroll
                    for (int kstep = 0; kstep < 8; ++kstep)
                    {
                        uint64_t const da = smem_desc(q_base + kstep * 256, 128, 2048);
                        uint64_t const db = smem_desc(k_base + ks * 32768 + kstep * 256, 128, 2048);
                        mma_ss(tmem + kColS + 128 * sb, da, db, kIdescQK, kstep > 0 ? 1U : 0U);
                    }
                    tc_commit(bar(kBarSFull + sb));
                    tc_commit(bar(kBarKFree + ks));
                    if (jx == n_k - 1)
                        tc_commit(bar(kBarQFree));
                    if (jx > 0)
                        issue_pv(x - 1, k);
                }
                if (n_k > 0)
                    issue_pv(x - 1, k);
                tc_commit(bar(kBarOReady));
            }
            // Drain: every commit's arrive lands before the CTA's smem is released.
            for (int s = 0; s < 2; ++s)
            {
                int const xs = nblk - 1 - ((nblk - 1 - s) & 1);
                if (xs >= 0)
                {
                    mbar_wait(bar(kBarKFree + s), (xs >> 1) & 1, 12);
                    mbar_wait(bar(kBarVFree + s), (xs >> 1) & 1, 13);
                    mbar_wait(bar(kBarPDone + s), (xs >> 1) & 1, 14);
                }
            }
            if (nseg > 0)
            {
                mbar_wait(bar(kBarQFree), (nseg - 1) & 1, 15);
                mbar_wait(bar(kBarOReady), (nseg - 1) & 1, 16);
                // The last segment's epilogue frees O after its outputs.
                mbar_wait(bar(kBarOFree), (nseg - 1) & 1, 18);
                arrive_item(ws, segs[nseg - 1]);
            }
        }
        __syncwarp();
    }
    else if (warp < 8)
    {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(kRegSoftmax));
        // Softmax WG g, keys 64 g .. 64 g + 63 of every block (S buffer x % 2): thread t = TMEM lane t = query (row
        // t / 16, head t % 16). Online softmax in-thread with a lazy max (O_g and l rescaled only when a block's max
        // exceeds the stale one by 2^8); P (FP16) back over the first 32 columns of the WG's 64.
        int const g = warp >> 2;
        int const wq = warp & 3;
        int const t = tid - 128 * g;
        int const r = t >> 4;
        uint32_t const lq = static_cast<uint32_t>(32 * wq) << 16;
        int const kb = 64 * g; // the WG's keys of every block: kb .. kb + 63
        uint32_t const s_base = tmem + lq + kColS + static_cast<uint32_t>(kb);
        uint32_t const o_col = tmem + lq + kColO + 128 * g;
        float const cs = p.sm_scale * 1.4426950409F * __ldg(p.kgs);
        int x = 0;
        for (int k = 0; k < nseg; ++k)
        {
            Seg const& sg = segs[k];
            int const kv_len = max(sg.seq - p.dql + r + 1, 0);
            int const n_k = sg.n;
            float m_run = kNegInf;
            float l_run = 0.F;
            for (int jx = 0; jx < n_k; ++jx, ++x)
            {
                uint32_t const z = static_cast<uint32_t>(blocks[x].z);
                int const by = blocks[x].y;
                int const sb = x & 1;
                mbar_wait(bar(kBarSFull + sb), (x >> 1) & 1, 9);
                tc_fence_after();
                uint32_t const s_col = s_base + 128U * static_cast<uint32_t>(sb);
                if (((z >> (2 * wq)) & 3U) == 0U)
                {
                    // Neither of this warp's rows selected the block: its P rows are zero.
                    uint32_t zz[16];
#pragma unroll
                    for (int i = 0; i < 16; ++i)
                        zz[i] = 0U;
                    tmem_st16(s_col, zz);
                    tmem_st16(s_col + 16, zz);
                }
                else
                {
                    uint32_t const sel = (z >> r) & 1U;
                    int const nvalid = min(max(kv_len - by * kPage, 0), kPage);
                    // O_g's last PV is the previous block's.
                    uint32_t const pdone = bar(kBarPDone + (sb ^ 1));
                    uint32_t const parity = static_cast<uint32_t>(((x - 1) >> 1) & 1);
                    // Every selected row of the warp holds all of the WG's 64 keys: all blocks but a sequence's last.
                    if (__all_sync(0xffffffffU, sel == 0U || nvalid >= kb + 64))
                        softmax_block<false>(s_col, o_col, kb, cs, sel, nvalid, pdone, parity, m_run, l_run);
                    else
                        softmax_block<true>(s_col, o_col, kb, cs, sel, nvalid, pdone, parity, m_run, l_run);
                }
                tmem_wait_st();
                tc_fence_before();
                mbar_arrive(bar(kBarPFull + sb));
            }
            // Epilogue (both WGs): O = O_0 2^(m_0 - m) + O_1 2^(m_1 - m), l likewise; WG g writes dims 64 g .. + 63.
            // Its values are made here: held across the block loop they would be spilled.
            mbar_wait(bar(kBarOReady), k & 1, 17);
            tc_fence_after();
            float2* const ml = reinterpret_cast<float2*>(smem + kSmemML);
            float const vgs = __ldg(p.vgs);
            int const h = t & 15;
            ml[g * kQ + t] = make_float2(m_run, l_run);
            pair_bar();
            float2 const a0 = ml[t];
            float2 const a1 = ml[kQ + t];
            float const mm = fmaxf(a0.x, a1.x);
            float const f0 = a0.x == kNegInf ? 0.F : ex2(a0.x - mm);
            float const f1 = a1.x == kNegInf ? 0.F : ex2(a1.x - mm);
            float const ll = a0.y * f0 + a1.y * f1;
            int const tok = sg.req * p.dql + r;
            int64_t const q0 = static_cast<int64_t>(sg.seg_id) * kQ + t;
            // The fast division: the IEEE one's slow-path call would spill the registers live across it.
            float const sc = (mm != kNegInf && ll > 0.F) ? __fdividef(vgs, ll) : 0.F;
#pragma unroll
            for (int c = 0; c < 2; ++c)
            {
                int const col = 64 * g + 32 * c;
                uint32_t o0[32];
                uint32_t o1[32];
                tmem_ld32(tmem + lq + kColO + col, o0);
                tmem_ld32(tmem + lq + kColO + 128 + col, o1);
                tmem_wait_ld();
                float v[32];
#pragma unroll
                for (int i = 0; i < 32; ++i)
                {
                    float const x0 = f0 != 0.F ? __uint_as_float(o0[i]) * f0 : 0.F;
                    float const x1 = f1 != 0.F ? __uint_as_float(o1[i]) * f1 : 0.F;
                    v[i] = x0 + x1;
                }
                if (!sg.split)
                {
                    uint4* const dst
                        = reinterpret_cast<uint4*>(p.out + tok * p.out_stride + (sg.kvh * kHeads + h) * kDim + col);
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                        dst[i] = make_uint4(pack_bf16(v[8 * i] * sc, v[8 * i + 1] * sc),
                            pack_bf16(v[8 * i + 2] * sc, v[8 * i + 3] * sc),
                            pack_bf16(v[8 * i + 4] * sc, v[8 * i + 5] * sc),
                            pack_bf16(v[8 * i + 6] * sc, v[8 * i + 7] * sc));
                }
                else
                {
                    // Partial O is [segment][dim / 4][query][4]: a warp's store is 32 queries' 16 B, contiguous.
                    float4* const dst = reinterpret_cast<float4*>(ws.part_o)
                        + (static_cast<int64_t>(sg.seg_id) * (kDim / 4) + (col >> 2)) * kQ + t;
                    bool const live = mm != kNegInf;
#pragma unroll
                    for (int i = 0; i < 8; ++i)
                        __stcg(dst + i * kQ,
                            live ? make_float4(v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3])
                                 : make_float4(0.F, 0.F, 0.F, 0.F));
                }
            }
            if (sg.split && g == 0)
            {
                __stcg(ws.part_m + q0, mm);
                __stcg(ws.part_l + q0, mm != kNegInf ? ll : 0.F);
            }
            tc_fence_before();
            pair_bar(); // ml is rewritten by the next segment
            mbar_arrive(bar(kBarOFree));
        }
    }
    else
    {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(kRegAux));
    }
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    if (warp == 16)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(kTmemCols));
#endif // MINIMAX_M3_OWN_T_ENABLED
}

// One warp per item (request, KV head); lane l holds entries 4 l .. 4 l + 3 = (row l / 4, slots 4 (l % 4) .. + 3).
// LC2's selection: slot < min(16, row blocks) with id >= 0; ids past the request's last block hold no key of any row
// and are dropped. Sorted keys (block, then row) give the ascending union; each block's run ORs its rows. With PDL the
// launch overlaps the previous kernel's tail; griddepcontrol.wait orders the reads after it.
__global__ void __launch_bounds__(128) own_t_union(Params const p, Ws const ws, int items)
{
#if MINIMAX_M3_OWN_T_ENABLED
    asm volatile("griddepcontrol.launch_dependents;");
    asm volatile("griddepcontrol.wait;" ::: "memory");
    __shared__ int keys_s[4][kU];
    int const w = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
    int const item = blockIdx.x * 4 + w;
    if (item >= items)
        return;
    int const req = item / p.num_kv_heads;
    int const kvh = item - req * p.num_kv_heads;
    int const r = lane >> 2;
    int const slot0 = (lane & 3) * 4;
    int const* const tp = p.topk + kvh * p.t_h + (req * p.dql + r) * p.t_tok + slot0 * p.t_k;
    // Every row has kTopk slots, so the slot loads need not wait for seq_lens.
    int raw[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        raw[i] = __ldg(tp + i * p.t_k);
    int const seq = __ldg(p.seq_lens + req);
    int const kv_len = max(seq - p.dql + r + 1, 0);
    int const real = min(kTopk, (kv_len + kPage - 1) / kPage);
    int const nb_max = (max(seq, 0) + kPage - 1) / kPage;
    // Key = block << 10 | row << 7 | the entry's slot in the warp (4 lane + i): the pages are looked up before the
    // sort, and a block's first key takes its page from the slot that loaded it.
    int key[4];
    int pg[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        int const id = slot0 + i < real ? raw[i] : -1;
        bool const sel = id >= 0 && id < nb_max;
        key[i] = sel ? ((id << 10) | (r << 7) | (4 * lane + i)) : INT_MAX;
        pg[i] = sel ? __ldg(p.block_table + req * p.bt_stride + id) : 0;
    }
#pragma unroll
    for (int size = 2; size <= kU; size <<= 1)
    {
#pragma unroll
        for (int stride = size >> 1; stride > 0; stride >>= 1)
        {
            if (stride >= 4)
            {
                int const lx = stride >> 2;
                bool const lower = (lane & lx) == 0;
#pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    int const other = __shfl_xor_sync(0xffffffffU, key[i], lx);
                    bool const up = ((4 * lane + i) & size) == 0;
                    key[i] = (lower == up) ? min(key[i], other) : max(key[i], other);
                }
            }
            else
            {
#pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    if ((i & stride) == 0)
                    {
                        int const jj = i | stride;
                        bool const up = ((4 * lane + i) & size) == 0;
                        int const a = key[i];
                        int const bb = key[jj];
                        if ((a > bb) == up)
                        {
                            key[i] = bb;
                            key[jj] = a;
                        }
                    }
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
        keys_s[w][4 * lane + i] = key[i];
    __syncwarp();
    bool first[4];
    int mask[4];
    int cnt = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        int const e = 4 * lane + i;
        int const kk = key[i];
        first[i] = kk != INT_MAX && (e == 0 || (keys_s[w][e - 1] >> 10) != (kk >> 10));
        mask[i] = 0;
        if (first[i])
        {
            for (int jj = e; jj < kU; ++jj)
            {
                int const kj = keys_s[w][jj];
                if (kj == INT_MAX || (kj >> 10) != (kk >> 10))
                    break;
                mask[i] |= 1 << ((kj >> 7) & 7);
            }
        }
        cnt += first[i] ? 1 : 0;
    }
    int wsum = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i)
        wsum += first[i] ? kWBlock + kWUnit * __popc(mask[i]) : 0;
    int incl = cnt;
    int wincl = wsum;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1)
    {
        int const y = __shfl_up_sync(0xffffffffU, incl, o);
        int const wy = __shfl_up_sync(0xffffffffU, wincl, o);
        if (lane >= o)
        {
            incl += y;
            wincl += wy;
        }
    }
    int pos = incl - cnt;
    int wpos = wincl - wsum;
    int const total = __shfl_sync(0xffffffffU, incl, 31);
    int const wtotal = __shfl_sync(0xffffffffU, wincl, 31);
    // Each key's page from the slot that loaded it (every lane shuffles; only a block's first key uses its page).
    int kpg[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        int const src = key[i] & 127;
        int const a0 = __shfl_sync(0xffffffffU, pg[0], src >> 2);
        int const a1 = __shfl_sync(0xffffffffU, pg[1], src >> 2);
        int const a2 = __shfl_sync(0xffffffffU, pg[2], src >> 2);
        int const a3 = __shfl_sync(0xffffffffU, pg[3], src >> 2);
        int const sl = src & 3;
        kpg[i] = sl == 0 ? a0 : sl == 1 ? a1 : sl == 2 ? a2 : a3;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        if (first[i])
        {
            int const u = item * kU + pos;
            int const blk = key[i] >> 10;
            ws.u_blk[u] = blk;
            ws.u_page[u] = kpg[i];
            ws.u_mask[u] = static_cast<uint8_t>(mask[i]);
            ws.u_wpre[u] = wpos;
            wpos += kWBlock + kWUnit * __popc(mask[i]);
            ++pos;
        }
    }
    if (lane == 0)
    {
        ws.u_count[item] = total;
        ws.u_wt[item] = total > 0 ? wtotal + kWSeg : 0;
        ws.u_arrive[item] = 0;
        ws.u_desc[item] = ~0ULL;
    }
#endif // MINIMAX_M3_OWN_T_ENABLED
}

// Split items: O = sum_c w_c O_c / sum_c w_c l_c, w_c = 2^(m_c - max m), over the CTAs that covered the item (CTAs
// whose row had no key are skipped); items without blocks are zero-filled; items one CTA covered were written by it.
// A CTA takes 32 queries (lane) x 8 dim quads (warp), so the partials ([segment][dim / 4][query][4]) load coalesced;
// every partial's loads are issued before any is used. The BF16 result goes out through smem: thread i writes query
// i / 8's quad i % 8, so a query's 32 dims (one MXFP8 group) are 8 adjacent lanes. With MXFP8 the merge also writes the
// MXFP8 copy of every row, reading back the rows a single CTA wrote, so the main kernel has no MXFP8 code.
template <bool MXFP8>
__global__ void __launch_bounds__(256) own_t_merge(Params const p, Ws const ws, int items, int grid)
{
#if MINIMAX_M3_OWN_T_ENABLED
    __shared__ uint2 stage[32][9]; // [query][dim quad], padded against bank conflicts
    __shared__ unsigned long long desc_s;
    int const item = blockIdx.x;
    int const req = item / p.num_kv_heads;
    int const kvh = item - req * p.num_kv_heads;
    int const qg = (blockIdx.y & 3) * 32;
    int const quad0 = (blockIdx.y >> 2) * 8;
    // The output side of this thread.
    int const qo = qg + (threadIdx.x >> 3);
    int const oquad = quad0 + (threadIdx.x & 7);
    int const tok = req * p.dql + (qo >> 4);
    int const head = kvh * kHeads + (qo & 15);
    __nv_bfloat16* const dst = p.out + tok * p.out_stride + head * kDim + oquad * 4;
    // The item's CTA range, published by its last CTA (by CTA 0 for an item without blocks).
    if (threadIdx.x == 0)
    {
        desc_s = wait_desc(ws.u_desc + item);
    }
    __syncthreads();
    unsigned long long const d = desc_s;
    if (static_cast<uint32_t>(d) == kDescEmpty)
    {
        *reinterpret_cast<uint2*>(dst) = make_uint2(0U, 0U);
        if constexpr (MXFP8)
        {
            mxfp8_quad(p, tok, head, oquad, make_uint2(0U, 0U));
        }
        return;
    }
    long long const cf = static_cast<int>(static_cast<uint32_t>(d));
    long long const cl = static_cast<int>(d >> 32);
    if (cl == cf)
    {
        // The one CTA that covered the item wrote its rows before it published the range.
        if constexpr (MXFP8)
        {
            mxfp8_quad(p, tok, head, oquad, __ldcg(reinterpret_cast<uint2 const*>(dst)));
        }
        return;
    }
    // The load side: query q, dims 4 quad .. 4 quad + 3.
    int const lane = threadIdx.x & 31;
    int const wq = threadIdx.x >> 5;
    int const q = qg + lane;
    int const quad = quad0 + wq;
    constexpr int kB = 8;
    float mmax = kNegInf;
    float den = 0.F;
    float4 acc = make_float4(0.F, 0.F, 0.F, 0.F);
    for (long long c0 = cf; c0 <= cl; c0 += kB)
    {
        float m[kB];
        float l[kB];
        float4 o[kB];
#pragma unroll
        for (int i = 0; i < kB; ++i)
        {
            bool const in = c0 + i <= cl;
            int64_t const sid = c0 + i + item;
            m[i] = in ? __ldcg(ws.part_m + sid * kQ + q) : kNegInf;
            l[i] = in ? __ldcg(ws.part_l + sid * kQ + q) : 0.F;
            o[i] = in ? __ldcg(reinterpret_cast<float4 const*>(ws.part_o) + (sid * (kDim / 4) + quad) * kQ + q)
                      : make_float4(0.F, 0.F, 0.F, 0.F);
        }
        float mb = mmax;
#pragma unroll
        for (int i = 0; i < kB; ++i)
            mb = fmaxf(mb, m[i]);
        float const rs = (mmax == kNegInf) ? 0.F : ex2(mmax - mb);
        den *= rs;
        acc.x *= rs;
        acc.y *= rs;
        acc.z *= rs;
        acc.w *= rs;
#pragma unroll
        for (int i = 0; i < kB; ++i)
        {
            if (m[i] == kNegInf)
                continue;
            float const wgt = ex2(m[i] - mb);
            den += wgt * l[i];
            acc.x += wgt * o[i].x;
            acc.y += wgt * o[i].y;
            acc.z += wgt * o[i].z;
            acc.w += wgt * o[i].w;
        }
        mmax = mb;
    }
    float const f = den > 0.F ? __ldg(p.vgs) / den : 0.F;
    stage[lane][wq] = make_uint2(pack_bf16(acc.x * f, acc.y * f), pack_bf16(acc.z * f, acc.w * f));
    __syncthreads();
    uint2 const packed = stage[threadIdx.x >> 3][threadIdx.x & 7];
    *reinterpret_cast<uint2*>(dst) = packed;
    if constexpr (MXFP8)
    {
        mxfp8_quad(p, tok, head, oquad, packed);
    }
#endif // MINIMAX_M3_OWN_T_ENABLED
}

// own_t_merge for launches whose items span more than 8 CTAs on average: both 8-CTA batches' partials load before
// they are combined in own_t_merge's order, so the result is own_t_merge's.
template <bool MXFP8>
__global__ void __launch_bounds__(256, 2) own_t_merge_wide(Params const p, Ws const ws, int items, int grid)
{
#if MINIMAX_M3_OWN_T_ENABLED
    __shared__ uint2 stage[32][9]; // [query][dim quad], padded against bank conflicts
    __shared__ unsigned long long desc_s;
    int const item = blockIdx.x;
    int const req = item / p.num_kv_heads;
    int const kvh = item - req * p.num_kv_heads;
    int const qg = (blockIdx.y & 3) * 32;
    int const quad0 = (blockIdx.y >> 2) * 8;
    // The output side of this thread.
    int const qo = qg + (threadIdx.x >> 3);
    int const oquad = quad0 + (threadIdx.x & 7);
    int const tok = req * p.dql + (qo >> 4);
    int const head = kvh * kHeads + (qo & 15);
    __nv_bfloat16* const dst = p.out + tok * p.out_stride + head * kDim + oquad * 4;
    // The item's CTA range, published by its last CTA (by CTA 0 for an item without blocks).
    if (threadIdx.x == 0)
    {
        desc_s = wait_desc(ws.u_desc + item);
    }
    __syncthreads();
    unsigned long long const d = desc_s;
    if (static_cast<uint32_t>(d) == kDescEmpty)
    {
        *reinterpret_cast<uint2*>(dst) = make_uint2(0U, 0U);
        if constexpr (MXFP8)
        {
            mxfp8_quad(p, tok, head, oquad, make_uint2(0U, 0U));
        }
        return;
    }
    int const cf = static_cast<int>(static_cast<uint32_t>(d));
    int const cl = static_cast<int>(d >> 32);
    if (cl == cf)
    {
        // The one CTA that covered the item wrote its rows before it published the range.
        if constexpr (MXFP8)
        {
            mxfp8_quad(p, tok, head, oquad, __ldcg(reinterpret_cast<uint2 const*>(dst)));
        }
        return;
    }
    // The load side: query q, dims 4 quad .. 4 quad + 3.
    int const lane = threadIdx.x & 31;
    int const wq = threadIdx.x >> 5;
    int const q = qg + lane;
    int const quad = quad0 + wq;
    constexpr int kB = 8;
    constexpr int kLoad = 2 * kB;
    float mmax = kNegInf;
    float den = 0.F;
    float4 acc = make_float4(0.F, 0.F, 0.F, 0.F);
    for (int c0 = cf; c0 <= cl; c0 += kLoad)
    {
        float m[kLoad];
        float l[kLoad];
        float4 o[kLoad];
#pragma unroll
        for (int i = 0; i < kLoad; ++i)
        {
            bool const in = c0 + i <= cl;
            // A slot past the item's last CTA reads that CTA's partials (in bounds) and is not used.
            int64_t const sid = static_cast<int64_t>(in ? c0 + i : cl) + item;
            m[i] = in ? __ldcg(ws.part_m + sid * kQ + q) : kNegInf;
            l[i] = in ? __ldcg(ws.part_l + sid * kQ + q) : 0.F;
            o[i] = in ? __ldcg(reinterpret_cast<float4 const*>(ws.part_o) + (sid * (kDim / 4) + quad) * kQ + q)
                      : make_float4(0.F, 0.F, 0.F, 0.F);
        }
#pragma unroll
        for (int hb = 0; hb < kLoad / kB; ++hb)
        {
            if (c0 + hb * kB > cl)
                break;
            float mb = mmax;
#pragma unroll
            for (int i = hb * kB; i < hb * kB + kB; ++i)
                mb = fmaxf(mb, m[i]);
            float const rs = (mmax == kNegInf) ? 0.F : ex2(mmax - mb);
            den *= rs;
            acc.x *= rs;
            acc.y *= rs;
            acc.z *= rs;
            acc.w *= rs;
#pragma unroll
            for (int i = hb * kB; i < hb * kB + kB; ++i)
            {
                if (m[i] == kNegInf)
                    continue;
                float const wgt = ex2(m[i] - mb);
                den += wgt * l[i];
                acc.x += wgt * o[i].x;
                acc.y += wgt * o[i].y;
                acc.z += wgt * o[i].z;
                acc.w += wgt * o[i].w;
            }
            mmax = mb;
        }
    }
    float const f = den > 0.F ? __ldg(p.vgs) / den : 0.F;
    stage[lane][wq] = make_uint2(pack_bf16(acc.x * f, acc.y * f), pack_bf16(acc.z * f, acc.w * f));
    __syncthreads();
    uint2 const packed = stage[threadIdx.x >> 3][threadIdx.x & 7];
    *reinterpret_cast<uint2*>(dst) = packed;
    if constexpr (MXFP8)
    {
        mxfp8_quad(p, tok, head, oquad, packed);
    }
#endif // MINIMAX_M3_OWN_T_ENABLED
}

// DEP_OWNT_FASTPLAN: the main kernel's plan derives the union's block weights (kFastPlan) when they are fixed
// (kFixedBlockWeights) and the caller allows it; otherwise (the kill switch, or weights that are not fixed) they are
// loaded. Both give the same split, bitwise.
bool fastPlan(bool allowed)
{
    return kFixedBlockWeights && allowed;
}

// The main kernel's instances: [fast plan][plan chunks]. Without fixed block weights the fast entries are the loading
// instances.
using MainKernel = void (*)(Params, Ws, int);
MainKernel const kMainKernels[2][2] = {{own_t_main<kSmallPlanPer, false>, own_t_main<kMaxItems / 32, false>},
    {own_t_main<kSmallPlanPer, kFixedBlockWeights>, own_t_main<kMaxItems / 32, kFixedBlockWeights>}};

} // namespace

size_t minimaxM3Sm107SparseDecodeOwnTWorkspaceBytes(int32_t numItems, int32_t grid)
{
    size_t total = 0;
    carve(nullptr, numItems, grid, &total);
    return total;
}

cudaError_t minimaxM3Sm107SparseDecodeOwnTConfigure()
{
    for (MainKernel const kernel : {kMainKernels[0][0], kMainKernels[0][1], kMainKernels[1][0], kMainKernels[1][1]})
    {
        void const* const fn = reinterpret_cast<void const*>(kernel);
        cudaError_t err = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
        if (err != cudaSuccess)
        {
            return err;
        }
        cudaFuncAttributes attrs{};
        err = cudaFuncGetAttributes(&attrs, fn);
        if (err != cudaSuccess)
        {
            return err;
        }
        if (attrs.numRegs != kRegLaunch)
        {
            return cudaErrorInvalidDeviceFunction;
        }
    }
    return cudaSuccess;
}

cudaError_t invokeMinimaxM3Sm107SparseDecodeOwnT(MinimaxM3Sm107SparseDecodeOwnTParams const& params, void* workspace,
    size_t workspaceBytes, int32_t grid, bool usePdl, cudaStream_t stream, bool allowFastPlan)
{
    int const items = params.num_requests * params.num_kv_heads;
    if (items <= 0 || items > kMaxItems || params.dql != kRows || grid <= 0)
    {
        return cudaErrorInvalidValue;
    }
    bool const mxfp8 = params.q8 != nullptr;
    if (mxfp8 && (params.sf == nullptr || params.num_sf_k_tiles * 128 < params.num_kv_heads * kHeads * kDim))
    {
        return cudaErrorInvalidValue;
    }
    size_t need = 0;
    Ws const ws = carve(workspace, items, grid, &need);
    if (workspace == nullptr || workspaceBytes < need)
    {
        return cudaErrorInvalidValue;
    }
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((items + 3) / 4);
    cfg.blockDim = dim3(128);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cfg.attrs = usePdl ? attr : nullptr;
    cfg.numAttrs = usePdl ? 1 : 0;
    cudaError_t err = cudaLaunchKernelEx(&cfg, own_t_union, params, ws, items);
    if (err != cudaSuccess)
    {
        return err;
    }
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = kSmemBytes;
    MainKernel const mainKernel = kMainKernels[fastPlan(allowFastPlan) ? 1 : 0][items <= kSmallPlanPer * 32 ? 0 : 1];
    err = cudaLaunchKernelEx(&cfg, mainKernel, params, ws, items);
    if (err != cudaSuccess)
    {
        return err;
    }
    cfg.gridDim = dim3(items, kQ / 8);
    cfg.blockDim = dim3(256);
    cfg.dynamicSmemBytes = 0;
    // An item spans grid / items CTAs on average: past 8, the wide merge loads its two batches at once.
    if (items * 8 < grid)
    {
        err = mxfp8 ? cudaLaunchKernelEx(&cfg, own_t_merge_wide<true>, params, ws, items, grid)
                    : cudaLaunchKernelEx(&cfg, own_t_merge_wide<false>, params, ws, items, grid);
    }
    else
    {
        err = mxfp8 ? cudaLaunchKernelEx(&cfg, own_t_merge<true>, params, ws, items, grid)
                    : cudaLaunchKernelEx(&cfg, own_t_merge<false>, params, ws, items, grid);
    }
    if (err != cudaSuccess)
    {
        return err;
    }
    return cudaGetLastError();
}

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
