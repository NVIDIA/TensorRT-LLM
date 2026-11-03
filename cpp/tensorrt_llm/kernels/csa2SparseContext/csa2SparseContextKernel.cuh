/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

// CSA2 sparse context attention over the shared FP8 KV bank for the SM100 family (tcgen05, TMA gather4).
//
// One persistent 512-thread CTA per SM processes one query token (64 heads) per work item, in 128-key blocks:
//   S = Q K^T  tcgen05.mma.ws M=64 N=128 kind::f8f6f4. A = Q in TMEM, duplicated into both lane halves (lanes h and
//              h + 64 hold head h); B = the key block in shared memory (K-major, 128-byte swizzle). Lanes 0-63
//              receive keys 0-63 of the block, lanes 64-127 keys 64-127.
//   O += P V   tcgen05.mma.ws M=64 N=256 kind::f8f6f4. A = P (E4M3, shared memory, K-major); B = the same key block
//              read MN-major (a bank row is both K and V), once per 256-channel half of O.
// Every bank row a query selects is fetched once per query.
//
// The softmax follows the trtllm-gen FP8 context kernel: an exact running maximum per 128-key block, P scaled by 448
// and rounded to E4M3 for the PV product, FP32 row sums of the unrounded P, and the sink in the denominator.
//
// Warp roles:
//   warps 0-3            softmax, one thread per TMEM lane
//   warps 4-7            O rescale between blocks, then the epilogue (TMEM -> BF16 -> TMA store, or -> the o_a
//                        projection's E4M3 input with the inverse RoPE applied)
//   warp 8               QK issue (Q copy into TMEM, S = Q K^T)
//   warp 12              PV issue
//   warps 9-11, 13-15    producers: Q items and KV blocks through one 3-slot ring. KV rows are TMA-gathered four at
//                        a time; a block of 128 consecutive bank rows (an SWA window) is loaded as four tiles. BF16 Q
//                        is quantized to E4M3 in its ring slot by the producers.
//
// Barrier phases are tracked by parity. No barrier may complete two phases ahead of one of its waiters, so every
// arrive that could otherwise run ahead is ordered after a wait on the consumer of the phase before (alphaEmpty,
// statsEmpty, and the PV wait in the O-rescale skip path). The ring slots have six waiters (the producer warps) but
// some items are loaded by warp 9 alone, so every producer warp also arrives on the slot's last chunk barrier after
// its slotEmpty wait: a slot cannot be filled, consumed and released again before each producer warp has waited on it.
// The O-rescale warps also wait on every pvDone phase.

#pragma once

#include "tensorrt_llm/common/config.h"

#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// The kernel uses tcgen05, TMEM and TMA gather4 of the SM100 family. Other targets compile a trapping body; the host
// side rejects unsupported devices before launching.
#if defined(__CUDA_ARCH__)
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL)                                         \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__)                                                                         \
        && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000 || __CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))
#define CSA2_SPARSE_CONTEXT_ARCH_SUPPORTED 1
#endif
#endif

TRTLLM_NAMESPACE_BEGIN

namespace kernels::csa2SparseContext
{

namespace detail
{

constexpr int kHeads = 64;
constexpr int kDim = 512;
constexpr int kBlockKeys = 128;
constexpr int kNumSlots = 3;
constexpr int kSlotBytes = kBlockKeys * kDim;   //!< One KV block (E4M3) or one BF16 Q item: 64 KB.
constexpr int kKvChunkBytes = kBlockKeys * 128; //!< 128 channels of a KV block.
constexpr int kQChunkBytes = kHeads * 128;      //!< 128 E4M3 channels (or 64 BF16 channels) of Q.
constexpr int kPBytes = kHeads * kBlockKeys;
constexpr int kStageBytes = 32 * 32 * 2;        //!< Epilogue staging tile of one warp: 32 heads x 32 BF16.
constexpr int kNumThreads = 512;
constexpr int kNumProducerWarps = 6;
constexpr int kNumProducerThreads = kNumProducerWarps * 32;
constexpr int kQPrefetchItems = 2;                //!< Q tiles are prefetched into L2 this many items ahead.
constexpr uint32_t kTmemCols = 512;
constexpr uint32_t kTmemO = 0;                    //!< 256 columns: O, two N=256 halves of 64 lanes x 2.
constexpr uint32_t kTmemS = 256;                  //!< 2 x 64 columns: S of two blocks in flight.
constexpr uint32_t kTmemQ = 384;                  //!< 128 columns: Q (512 E4M3 bytes per lane).
constexpr float kPScale = 448.F;                  //!< P is stored as E4M3 of p * 448.
constexpr float kLog2PScale = 8.807354922057604F; //!< log2(448).
constexpr float kLog2e = 1.4426950408889634F;
// Named barriers (0 is __syncthreads): the softmax pairs exchanging row maxima, and the producers.
constexpr int kSoftmaxBarrier = 2;
constexpr int kProducerBarrier = 4;

struct SharedStorage
{
    uint8_t slot[kNumSlots][kSlotBytes]; // 1024-B aligned
    uint8_t p[2][kPBytes];
    uint8_t stage[4][kStageBytes];
    float rowMax[2][128];
    float alpha[2][64];
    float rowSum[2][128];
    float rowMaxFinal[2][64];
    uint64_t slotFull[kNumSlots][4]; // per 128-byte channel chunk, so QK starts on the first chunk; the last
                                     // chunk's barrier also takes one arrival per other producer warp
    uint64_t slotEmpty[kNumSlots];
    uint64_t qRaw[4];                // BF16 Q: channel chunk c (BF16 boxes 2c, 2c + 1) has landed
    uint64_t sFull[2];
    uint64_t sEmpty[2];
    uint64_t pFull[2];
    uint64_t pEmpty[2];
    uint64_t alphaFull[2];  // row rescale factors of block g published (right after the row max)
    uint64_t alphaEmpty[2]; // the rescale warps have read them
    uint64_t oReady[2][2];  // [parity][O half]: O may take PV of the block (every block but a query's first)
    uint64_t pvDone[2][2];  // [block parity][O half]
    uint64_t oEmpty;        // the epilogue has read O of a query
    uint64_t statsFull[2];  // row statistics of a query published
    uint64_t statsEmpty[2]; // the epilogue has read them
    uint32_t tmemBase;
};

constexpr size_t kSmemBytes = sizeof(SharedStorage) + 1024;

} // namespace detail

//! Device-side arguments of the kernel (see Csa2SparseContextParams).
struct KernelParams
{
    int32_t const* indices;
    int64_t indexStride;
    int maxKeys;
    int32_t const* lens;
    float const* bmm1Scale;
    float const* bmm2Scale;
    float const* qScale;
    float const* attnSinks;
    int numTokens;
    int bankRows;
    // kFp8Out only (see Csa2SparseContextParams).
    uint8_t* fp8Out;
    int64_t fp8GroupStride;
    int64_t fp8TokenStride;
    float* fp8Scale;
    int64_t fp8ScaleGroupStride;
    int64_t fp8ScaleBlockStride;
    int fp8ScaleRows;
    int headsPerGroup;
    void const* positions;
    bool positions64;
    float const* ropeCosSin;
};

#if defined(CSA2_SPARSE_CONTEXT_ARCH_SUPPORTED)

namespace detail
{

__device__ __forceinline__ uint32_t smemAddress(void const* ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void mbarInit(uint64_t* bar, uint32_t count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smemAddress(bar)), "r"(count));
}

__device__ __forceinline__ void mbarArrive(uint64_t* bar)
{
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(smemAddress(bar)) : "memory");
}

__device__ __forceinline__ void mbarArriveExpectTx(uint64_t* bar, uint32_t bytes)
{
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(smemAddress(bar)), "r"(bytes)
        : "memory");
}

__device__ __forceinline__ void mbarWait(uint64_t* bar, uint32_t parity)
{
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1;\n"
        "@!p bra WAIT_%=;\n"
        "}\n" ::"r"(smemAddress(bar)),
        "r"(parity)
        : "memory");
}

__device__ __forceinline__ void tcFenceAfter()
{
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcFenceBefore()
{
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcCommit(uint64_t* bar)
{
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(smemAddress(bar))
                 : "memory");
}

__device__ __forceinline__ void fenceProxyAsyncSmem()
{
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void namedBarrierSync(int id, int threads)
{
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

//! Shared-memory matrix descriptor for a 128-byte swizzled operand (SM100: version 1, layout SWIZZLE_128B).
__device__ __forceinline__ uint64_t smemDescriptor(uint32_t address, uint32_t leadingBytes, uint32_t strideBytes)
{
    uint64_t desc = 0;
    desc |= static_cast<uint64_t>((address >> 4) & 0x3FFF);
    desc |= static_cast<uint64_t>((leadingBytes >> 4) & 0x3FFF) << 16;
    desc |= static_cast<uint64_t>((strideBytes >> 4) & 0x3FFF) << 32;
    desc |= uint64_t{1} << 46;
    desc |= uint64_t{2} << 61;
    return desc;
}

//! kind::f8f6f4 instruction descriptor: E4M3 x E4M3 -> FP32, M x N, A / B K-major (0) or MN-major (1).
__host__ __device__ constexpr uint32_t mmaDescriptorF8(int m, int n, int aMnMajor, int bMnMajor)
{
    return (1U << 4) | (static_cast<uint32_t>(aMnMajor) << 15) | (static_cast<uint32_t>(bMnMajor) << 16)
        | (static_cast<uint32_t>(n >> 3) << 17) | (static_cast<uint32_t>(m >> 4) << 24);
}

__device__ __forceinline__ void mmaWsTmemA(uint32_t tmemD, uint32_t tmemA, uint64_t bDesc, uint32_t iDesc, uint32_t acc)
{
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.ws.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, p, 0;\n}\n" ::"r"(tmemD),
        "r"(tmemA), "l"(bDesc), "r"(iDesc), "r"(acc));
}

__device__ __forceinline__ void mmaWsSmemA(uint32_t tmemD, uint64_t aDesc, uint64_t bDesc, uint32_t iDesc, uint32_t acc)
{
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.ws.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p, 0;\n}\n" ::"r"(tmemD),
        "l"(aDesc), "l"(bDesc), "r"(iDesc), "r"(acc));
}

//! 64 rows x 16 bytes from shared memory into TMEM: rows 0-31 -> lanes 0-31 and 64-95, rows 32-63 -> 32-63, 96-127.
__device__ __forceinline__ void tmemCopy64x128Dup(uint32_t tmem, uint64_t desc)
{
    asm volatile("tcgen05.cp.cta_group::1.64x128b.warpx2::02_13 [%0], %1;" ::"r"(tmem), "l"(desc));
}

__device__ __forceinline__ void tmemLoad32(uint32_t address, float* v)
{
    uint32_t* r = reinterpret_cast<uint32_t*>(v);
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]),
        "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]),
        "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
        : "r"(address));
}

__device__ __forceinline__ void tmemStore32(uint32_t address, float const* v)
{
    uint32_t const* r = reinterpret_cast<uint32_t const*>(v);
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,"
        "%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};" ::"r"(address),
        "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]),
        "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]), "r"(r[16]), "r"(r[17]), "r"(r[18]),
        "r"(r[19]), "r"(r[20]), "r"(r[21]), "r"(r[22]), "r"(r[23]), "r"(r[24]), "r"(r[25]), "r"(r[26]), "r"(r[27]),
        "r"(r[28]), "r"(r[29]), "r"(r[30]), "r"(r[31]), "r"(r[32]));
}

__device__ __forceinline__ void tmemLoadWait()
{
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmemStoreWait()
{
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmaLoad2d(CUtensorMap const* map, uint64_t* bar, void* dst, int c0, int c1)
{
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::
            "r"(smemAddress(dst)),
        "l"(map), "r"(c0), "r"(c1), "r"(smemAddress(bar))
        : "memory");
}

__device__ __forceinline__ void tmaLoad3d(CUtensorMap const* map, uint64_t* bar, void* dst, int c0, int c1, int c2)
{
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], "
        "[%5];" ::"r"(smemAddress(dst)),
        "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(smemAddress(bar))
        : "memory");
}

__device__ __forceinline__ void tmaPrefetch3d(CUtensorMap const* map, int c0, int c1, int c2)
{
    asm volatile(
        "cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" ::"l"(map), "r"(c0), "r"(c1), "r"(c2)
        : "memory");
}

__device__ __forceinline__ void tmaGather4(CUtensorMap const* map, uint64_t* bar, void* dst, int col, int4 rows)
{
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, "
        "%5, %6}], [%7];" ::"r"(smemAddress(dst)),
        "l"(map), "r"(col), "r"(rows.x), "r"(rows.y), "r"(rows.z), "r"(rows.w), "r"(smemAddress(bar))
        : "memory");
}

__device__ __forceinline__ void tmaStore3d(CUtensorMap const* map, void const* src, int c0, int c1, int c2)
{
    asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%1, %2, %3}], [%4];" ::"l"(map), "r"(c0),
                 "r"(c1), "r"(c2), "r"(smemAddress(src))
                 : "memory");
}

template <uint32_t kRegisters>
__device__ __forceinline__ void setMaxRegistersInc()
{
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(kRegisters));
}

template <uint32_t kRegisters>
__device__ __forceinline__ void setMaxRegistersDec()
{
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(kRegisters));
}

//! True on one lane of a converged warp. ptxas then issues the guarded tcgen05 instructions without per-lane loops.
__device__ __forceinline__ bool electOne()
{
    uint32_t pred = 0;
    asm volatile("{\n.reg .b32 %%rx;\n.reg .pred %%px;\nelect.sync %%rx|%%px, 0xffffffff;\n@%%px mov.s32 %0, 1;\n}\n"
                 : "+r"(pred));
    return pred != 0;
}

__device__ __forceinline__ float exp2Approx(float x)
{
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

//! Elementwise a * b of two float pairs (round to nearest).
__device__ __forceinline__ float2 mulF32x2(float2 a, float2 b)
{
    float2 c;
    asm("mul.f32x2 %0, %1, %2;"
        : "=l"(reinterpret_cast<uint64_t&>(c))
        : "l"(reinterpret_cast<uint64_t const&>(a)), "l"(reinterpret_cast<uint64_t const&>(b)));
    return c;
}

//! Elementwise a * b + c of float pairs (round to nearest).
__device__ __forceinline__ float2 fmaF32x2(float2 a, float2 b, float2 c)
{
    float2 d;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(reinterpret_cast<uint64_t&>(d))
        : "l"(reinterpret_cast<uint64_t const&>(a)), "l"(reinterpret_cast<uint64_t const&>(b)),
        "l"(reinterpret_cast<uint64_t const&>(c)));
    return d;
}

//! Elementwise a + b of float pairs (round to nearest).
__device__ __forceinline__ float2 addF32x2(float2 a, float2 b)
{
    float2 c;
    asm("add.rn.f32x2 %0, %1, %2;"
        : "=l"(reinterpret_cast<uint64_t&>(c))
        : "l"(reinterpret_cast<uint64_t const&>(a)), "l"(reinterpret_cast<uint64_t const&>(b)));
    return c;
}

//! min(x, y) that propagates NaN (a NaN logit keeps its NaN probability).
__device__ __forceinline__ float minNaN(float x, float y)
{
    float r;
    asm("min.NaN.f32 %0, %1, %2;" : "=f"(r) : "f"(x), "f"(y));
    return r;
}

//! Two floats to E4M3 with round-to-nearest-even and saturation to +-448 (NaN stays NaN); lo in the low byte.
__device__ __forceinline__ uint32_t toE4m3x2(float lo, float hi)
{
    uint16_t r;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
    return r;
}

//! Eight BF16 values times scale to eight E4M3 bytes, exactly as the context FP8 quantizer does: each value is
//! widened to FP32, multiplied by the scale (one rounding) and converted by cvt.rn.satfinite.
__device__ __forceinline__ uint2 quantizeBf16x8(uint4 v, float scale)
{
    uint32_t const words[4] = {v.x, v.y, v.z, v.w};
    uint32_t e[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        float2 const x = mulF32x2(make_float2(__uint_as_float(words[i] << 16), __uint_as_float(words[i] & 0xFFFF0000U)),
            make_float2(scale, scale));
        e[i] = toE4m3x2(x.x, x.y);
    }
    return make_uint2(e[0] | (e[1] << 16), e[2] | (e[3] << 16));
}

//! Quantizes Q row `row` of channel chunk `chunk` (128 channels) in place: the chunk's BF16 values occupy boxes 2c
//! and 2c + 1 of the slot (64 rows x 128 bytes each, 128-byte swizzle), and its E4M3 values replace box 2c with the
//! same swizzle. One thread owns a whole row of box 2c, so each 16-byte unit is overwritten only after this thread
//! has read it: part k reads units {4k', ..} of its source row and writes E4M3 units 2k and 2k + 1, all of which
//! were read by parts <= k.
__device__ __forceinline__ void quantizeQRow(uint8_t* slot, int chunk, int row, float scale)
{
    uint8_t* const dst = slot + (2 * chunk) * kQChunkBytes + row * 128;
    uint8_t const* const upper = slot + (2 * chunk + 1) * kQChunkBytes + row * 128;
    int const swizzle = row & 7;
#pragma unroll
    for (int part = 0; part < 4; ++part)
    {
        uint8_t const* const src = part < 2 ? dst : upper;
        uint4 v[4];
#pragma unroll
        for (int u = 0; u < 4; ++u)
        {
            v[u] = *reinterpret_cast<uint4 const*>(src + (((4 * (part & 1) + u) ^ swizzle) * 16));
        }
#pragma unroll
        for (int w = 0; w < 2; ++w)
        {
            uint2 const a = quantizeBf16x8(v[2 * w], scale);
            uint2 const b = quantizeBf16x8(v[2 * w + 1], scale);
            *reinterpret_cast<uint4*>(dst + (((2 * part + w) ^ swizzle) * 16)) = make_uint4(a.x, a.y, b.x, b.y);
        }
    }
}

//! inverseRopeFp8Quant's rotation of a GPT-J channel pair (a, b): (a * cos + b * sin, b * cos - a * sin), the products
//! b * sin and a * sin rounded on their own.
__device__ __forceinline__ float2 inverseRopePair(float a, float b, float cosv, float sinv)
{
    return make_float2(__fmaf_rn(a, cosv, __fmul_rn(b, sinv)), __fmaf_rn(b, cosv, -__fmul_rn(a, sinv)));
}

//! A BF16 pair as two floats (exact).
__device__ __forceinline__ float2 bf16PairToFloat2(uint32_t w)
{
    return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xFFFF0000U));
}

//! RoPE position of a token.
__device__ __forceinline__ int64_t ropePosition(KernelParams const& p, int token)
{
    return p.positions64 ? __ldg(static_cast<int64_t const*>(p.positions) + token)
                         : __ldg(static_cast<int32_t const*>(p.positions) + token);
}

//! Writes 128-channel block 2 * kHalf + oHalf of `head` (BF16 pairs of an epilogue thread's TMEM columns
//! 128 * kHalf to 128 * kHalf + 127 at packed + 64 * kHalf) as the o_a projection's E4M3 input, exactly as
//! inverseRopeFp8Quant writes it from the BF16 output. Block 3 ends with the head's 64 RoPE channels, rotated back in
//! GPT-J pairs with the token's cos/sin row (cosSin) first. A block's scale is max(amax, 1e-12) / 448, amax ignoring
//! NaN; its values are multiplied by 448 / max(amax, 1e-12) (fast division) and rounded to E4M3 with saturation.
template <int kHalf>
__device__ __forceinline__ void storeFp8Output(
    KernelParams const& p, uint32_t const (&packed)[128], int head, int oHalf, int token, float const* cosSin)
{
    int const group = head / p.headsPerGroup;
    int const headInGroup = head - group * p.headsPerGroup;
    uint8_t* const data = p.fp8Out + group * p.fp8GroupStride + token * p.fp8TokenStride + headInGroup * kDim;
    float* const scales = p.fp8Scale + group * p.fp8ScaleGroupStride + token;
    {
        constexpr int half = kHalf;
        int const block = 2 * half + oHalf;
        uint32_t const* const v = packed + 64 * half;
        bool const rope = half == 1 && oHalf == 1;
        // BF16 maxima are exact and, like fmaxf, skip NaN; the RoPE block's last 32 pairs are rotated FP32 (zeroed so
        // they are defined on the other path too; undefined there, ptxas keeps them live across the token loop).
        float2 rotated[32] = {};
        __nv_bfloat162 m = __floats2bfloat162_rn(0.F, 0.F);
#pragma unroll
        for (int i = 0; i < 32; ++i)
        {
            m = __hmax2(m, __habs2(*reinterpret_cast<__nv_bfloat162 const*>(v + i)));
        }
        float amax;
        if (rope)
        {
#pragma unroll
            for (int i = 0; i < 32; i += 4)
            {
                float4 const c4 = __ldg(reinterpret_cast<float4 const*>(cosSin + i));
                float4 const s4 = __ldg(reinterpret_cast<float4 const*>(cosSin + 32 + i));
                float const c[4] = {c4.x, c4.y, c4.z, c4.w};
                float const s[4] = {s4.x, s4.y, s4.z, s4.w};
#pragma unroll
                for (int k = 0; k < 4; ++k)
                {
                    float2 const x = bf16PairToFloat2(v[32 + i + k]);
                    rotated[i + k] = inverseRopePair(x.x, x.y, c[k], s[k]);
                }
            }
            amax = fmaxf(__low2float(m), __high2float(m));
#pragma unroll
            for (int i = 0; i < 32; ++i)
            {
                amax = fmaxf(amax, fmaxf(fabsf(rotated[i].x), fabsf(rotated[i].y)));
            }
        }
        else
        {
#pragma unroll
            for (int i = 32; i < 64; ++i)
            {
                m = __hmax2(m, __habs2(*reinterpret_cast<__nv_bfloat162 const*>(v + i)));
            }
            amax = fmaxf(__low2float(m), __high2float(m));
        }
        float const blockMax = fmaxf(amax, 1e-12F);
        float const inv = __fdividef(448.F, blockMax);
        scales[(headInGroup * 4 + block) * p.fp8ScaleBlockStride] = blockMax * (1.F / 448.F);
        float2 const inv2 = make_float2(inv, inv);
#pragma unroll
        for (int w = 0; w < 32; w += 8)
        {
            uint32_t q[8];
#pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                int const pair = 2 * (w + j);
                float2 const a = mulF32x2(rope && pair >= 32 ? rotated[pair - 32] : bf16PairToFloat2(v[pair]), inv2);
                float2 const b
                    = mulF32x2(rope && pair >= 32 ? rotated[pair - 31] : bf16PairToFloat2(v[pair + 1]), inv2);
                q[j] = toE4m3x2(a.x, a.y) | (toE4m3x2(b.x, b.y) << 16);
            }
            asm volatile("st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" ::"l"(data + block * 128 + 4 * w),
                         "r"(q[0]), "r"(q[1]), "r"(q[2]), "r"(q[3]), "r"(q[4]), "r"(q[5]), "r"(q[6]), "r"(q[7])
                         : "memory");
        }
    }
}

//! Zeroes the scales of the padding rows numTokens .. fp8ScaleRows - 1, as inverseRopeFp8Quant does; t in [0, 128).
__device__ __forceinline__ void zeroFp8ScalePadding(KernelParams const& p, int t)
{
    int const blocksPerGroup = 4 * p.headsPerGroup;
    int const padding = (p.fp8ScaleRows - p.numTokens) * kHeads * 4;
    for (int i = t; i < padding; i += 128)
    {
        int const row = p.numTokens + i / (kHeads * 4);
        int const block = i % (kHeads * 4);
        p.fp8Scale[(block / blocksPerGroup) * p.fp8ScaleGroupStride + (block % blocksPerGroup) * p.fp8ScaleBlockStride
            + row]
            = 0.F;
    }
}

//! Rescale-warp step of block k of a query (block g overall): observe the block's row rescale factors, rescale O when
//! any changed, and let PV of the block proceed (oReady; r counts the blocks that take it). kChunked keeps 32 O columns
//! in registers instead of 128, for steps taken while an epilogue holds half of its output in registers.
template <bool kChunked>
__device__ __forceinline__ void rescaleStep(SharedStorage& sm, uint32_t oBase, int head, int k, int& g, int& r)
{
    int const sb = g & 1;
    // Observe every alphaFull phase (a skipped phase would alias the parity of the next wait).
    mbarWait(&sm.alphaFull[sb], (g >> 1) & 1);
    bool rescale = false;
    float a = 1.0F;
    if (k > 0)
    {
        a = sm.alpha[sb][head];
        rescale = __any_sync(0xFFFFFFFFU, a != 1.0F);
    }
    mbarArrive(&sm.alphaEmpty[sb]);
    if (rescale)
    {
        float2 const a2 = make_float2(a, a);
#pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            // O half h is final for PV(g - 1) once its half-h MMAs completed.
            mbarWait(&sm.pvDone[(g - 1) & 1][h], ((g - 1) >> 1) & 1);
            tcFenceAfter();
            constexpr int kCols = kChunked ? 32 : 128;
#pragma unroll
            for (int c0 = 0; c0 < 128; c0 += kCols)
            {
                float o[kCols];
#pragma unroll
                for (int c = 0; c < kCols; c += 32)
                {
                    tmemLoad32(oBase + 128 * h + c0 + c, o + c);
                }
                tmemLoadWait();
#pragma unroll
                for (int j = 0; j < kCols; j += 2)
                {
                    float2 const v = mulF32x2(make_float2(o[j], o[j + 1]), a2);
                    o[j] = v.x;
                    o[j + 1] = v.y;
                }
#pragma unroll
                for (int c = 0; c < kCols; c += 32)
                {
                    tmemStore32(oBase + 128 * h + c0 + c, o + c);
                }
            }
            tmemStoreWait();
            tcFenceBefore();
            mbarArrive(&sm.oReady[r & 1][h]);
        }
        ++r;
    }
    else if (k >= 1)
    {
        mbarArrive(&sm.oReady[r & 1][0]);
        mbarArrive(&sm.oReady[r & 1][1]);
        ++r;
        // Nothing above waited for PV(g - 1). Without this wait, oReady of the next block taking it (the
        // parity of block g - 1's) could complete while the PV issuer still waits on its phase for block
        // g - 1, and that parity wait would alias and deadlock; PV(g - 1) completing proves the PV issuer
        // has passed it. Waiting here, like the rescale path's wait before its loads, also observes every
        // pvDone phase; the phase of a query's last block is observed by its epilogue.
        mbarWait(&sm.pvDone[(g - 1) & 1][0], ((g - 1) >> 1) & 1);
        mbarWait(&sm.pvDone[(g - 1) & 1][1], ((g - 1) >> 1) & 1);
    }
    ++g;
}

__device__ __forceinline__ int keyCount(KernelParams const& p, int token)
{
    int const len = __ldg(p.lens + token);
    return len < 1 ? 1 : (len > p.maxKeys ? p.maxKeys : len);
}

__device__ __forceinline__ int blockCount(int keys)
{
    return (keys + kBlockKeys - 1) / kBlockKeys;
}

} // namespace detail

#endif // CSA2_SPARSE_CONTEXT_ARCH_SUPPORTED

//! kBf16Q: Q is BF16 and quantized to E4M3 (x qScale) on load; otherwise Q is E4M3 already.
//! kFp8Out: the epilogue writes the o_a projection's E4M3 input (storeFp8Output) instead of the BF16 output.
template <bool kBf16Q, bool kFp8Out>
__global__ void __launch_bounds__(detail::kNumThreads, 1)
    csa2SparseContextKernel(__grid_constant__ CUtensorMap const tmapBank, __grid_constant__ CUtensorMap const tmapQ,
        __grid_constant__ CUtensorMap const tmapTile, __grid_constant__ CUtensorMap const tmapOut,
        __grid_constant__ KernelParams const p)
{
#if defined(CSA2_SPARSE_CONTEXT_ARCH_SUPPORTED)
    using namespace detail;
    extern __shared__ __align__(1024) uint8_t smemRaw[];
    SharedStorage& sm = *reinterpret_cast<SharedStorage*>(smemRaw + ((1024U - (smemAddress(smemRaw) & 1023U)) & 1023U));
    int const tid = threadIdx.x;
    int const warp = tid / 32;
    int const lane = tid % 32;

    if (tid == 0)
    {
        for (int i = 0; i < kNumSlots; ++i)
        {
            for (int c = 0; c < 4; ++c)
            {
                mbarInit(&sm.slotFull[i][c], c == 3 ? kNumProducerWarps : 1);
            }
            mbarInit(&sm.slotEmpty[i], 1);
        }
        for (int c = 0; c < 4; ++c)
        {
            mbarInit(&sm.qRaw[c], 1);
        }
        for (int i = 0; i < 2; ++i)
        {
            mbarInit(&sm.sFull[i], 1);
            mbarInit(&sm.sEmpty[i], 128);
            mbarInit(&sm.pFull[i], 128);
            mbarInit(&sm.pEmpty[i], 1);
            mbarInit(&sm.alphaFull[i], 128);
            mbarInit(&sm.alphaEmpty[i], 128);
            mbarInit(&sm.oReady[i][0], 128);
            mbarInit(&sm.oReady[i][1], 128);
            mbarInit(&sm.pvDone[i][0], 1);
            mbarInit(&sm.pvDone[i][1], 1);
            mbarInit(&sm.statsFull[i], 128);
            mbarInit(&sm.statsEmpty[i], 128);
        }
        mbarInit(&sm.oEmpty, 128);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 0)
    {
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smemAddress(&sm.tmemBase)),
            "r"(kTmemCols));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    if (tid == 32 * 9)
    {
        asm volatile("prefetch.tensormap [%0];" ::"l"(&tmapBank) : "memory");
        asm volatile("prefetch.tensormap [%0];" ::"l"(&tmapQ) : "memory");
        asm volatile("prefetch.tensormap [%0];" ::"l"(&tmapTile) : "memory");
        if constexpr (!kFp8Out)
        {
            asm volatile("prefetch.tensormap [%0];" ::"l"(&tmapOut) : "memory");
        }
    }
    tcFenceBefore();
    __syncthreads();
    tcFenceAfter();
    uint32_t const tmemBase = sm.tmemBase;

    if (warp < 4)
    {
        // ======================================================================================== softmax
        setMaxRegistersInc<176>();
        int const t = tid; // TMEM lane
        int const head = t & 63;
        int const keyHalf = t >> 6;
        uint32_t const laneOffset = static_cast<uint32_t>(32 * warp) << 16;
        float const scaleLog2 = __ldg(p.bmm1Scale + 1);
        int g = 0;
        int item = 0;
        for (int token = blockIdx.x; token < p.numTokens; token += gridDim.x, ++item)
        {
            int const keys = keyCount(p, token);
            int const blocks = blockCount(keys);
            float mRun = -INFINITY;
            float lRun = 0.F;
            for (int k = 0; k < blocks; ++k, ++g)
            {
                int const sb = g & 1;
                mbarWait(&sm.sFull[sb], (g >> 1) & 1);
                tcFenceAfter();
                float s[64];
                tmemLoad32(tmemBase + laneOffset + kTmemS + 64 * sb, s);
                tmemLoad32(tmemBase + laneOffset + kTmemS + 64 * sb + 32, s + 32);
                tmemLoadWait();
                tcFenceBefore();
                mbarArrive(&sm.sEmpty[sb]);

                int valid = keys - k * kBlockKeys - keyHalf * 64;
                valid = valid < 0 ? 0 : (valid > 64 ? 64 : valid);
                if (valid < 64)
                {
#pragma unroll
                    for (int j = 0; j < 64; ++j)
                    {
                        s[j] = j < valid ? s[j] : -INFINITY;
                    }
                }
                float mx = -INFINITY;
#pragma unroll
                for (int j = 0; j < 64; j += 2)
                {
                    mx = fmaxf(mx, fmaxf(s[j], s[j + 1]));
                }
                mx *= scaleLog2;
                sm.rowMax[sb][t] = mx;
                // The two warps holding the two key halves of these heads.
                namedBarrierSync(kSoftmaxBarrier + (warp & 1), 64);
                mx = fmaxf(mx, sm.rowMax[sb][t ^ 64]);
                float const mNew = fmaxf(mRun, mx);
                float const alpha = exp2Approx(mRun - mNew);
                lRun *= alpha;
                if (g >= 2)
                {
                    mbarWait(&sm.alphaEmpty[sb], ((g >> 1) - 1) & 1);
                }
                if (keyHalf == 0)
                {
                    sm.alpha[sb][head] = alpha;
                }
                mbarArrive(&sm.alphaFull[sb]);
                // p * 448 = exp2(s * scale - m + log2(448)); l is kept in units of 448. The exponent is clamped at
                // log2(448): with |m| >= 2^15 the rounding of log2(448) - m could otherwise lift p above 448, where
                // E4M3 saturates while the row sum would keep the unsaturated value.
                float const negM = kLog2PScale - mNew;
                uint32_t packed[16];
                // The running sums of the score pairs (j, j + 1) and (j + 2, j + 3).
                float2 sums = make_float2(0.F, 0.F);
                float2 const scale2 = make_float2(scaleLog2, scaleLog2);
                float2 const negM2 = make_float2(negM, negM);
#pragma unroll
                for (int j = 0; j < 64; j += 4)
                {
                    float2 const x01 = fmaF32x2(make_float2(s[j + 0], s[j + 1]), scale2, negM2);
                    float2 const x23 = fmaF32x2(make_float2(s[j + 2], s[j + 3]), scale2, negM2);
                    float const p0 = exp2Approx(minNaN(x01.x, kLog2PScale));
                    float const p1 = exp2Approx(minNaN(x01.y, kLog2PScale));
                    float const p2 = exp2Approx(minNaN(x23.x, kLog2PScale));
                    float const p3 = exp2Approx(minNaN(x23.y, kLog2PScale));
                    sums = addF32x2(sums, addF32x2(make_float2(p0, p2), make_float2(p1, p3)));
                    packed[j / 4] = toE4m3x2(p0, p1) | (toE4m3x2(p2, p3) << 16);
                }
                lRun += sums.x + sums.y;
                // P buffer sb is free once PV(g - 2) has completed.
                if (g >= 2)
                {
                    mbarWait(&sm.pEmpty[sb], ((g >> 1) - 1) & 1);
                }
                // K-major, 128-byte swizzle: row r at (r / 8) * 1024 + (r % 8) * 128, 16-byte unit c at (c ^ (r % 8)).
                uint8_t* const pRow = sm.p[sb] + (head >> 3) * 1024 + (head & 7) * 128;
#pragma unroll
                for (int c = 0; c < 4; ++c)
                {
                    int const unit = keyHalf * 4 + c;
                    *reinterpret_cast<uint4*>(pRow + ((unit ^ (head & 7)) * 16))
                        = make_uint4(packed[4 * c], packed[4 * c + 1], packed[4 * c + 2], packed[4 * c + 3]);
                }
                fenceProxyAsyncSmem();
                mbarArrive(&sm.pFull[sb]);
                mRun = mNew;
            }
            // Row statistics for the epilogue (both key halves share mRun).
            if (item >= 2)
            {
                mbarWait(&sm.statsEmpty[item & 1], ((item >> 1) - 1) & 1);
            }
            sm.rowSum[item & 1][t] = lRun;
            if (keyHalf == 0)
            {
                sm.rowMaxFinal[item & 1][head] = mRun;
            }
            mbarArrive(&sm.statsFull[item & 1]);
        }
    }
    else if (warp < 8)
    {
        // ======================================================================================== O rescale + epilogue
        setMaxRegistersInc<224>();
        int const t = tid - 128; // TMEM lane
        int const head = t & 63;
        int const oHalf = t >> 6;
        uint32_t const laneOffset = static_cast<uint32_t>(32 * (warp - 4)) << 16;
        uint32_t const oBase = tmemBase + laneOffset + kTmemO;
        float const bmm2 = __ldg(p.bmm2Scale);
        float const sinkLog2 = p.attnSinks != nullptr ? __ldg(p.attnSinks + head) * kLog2e : -INFINITY;
        // kFp8Out: the RoPE block owners load each token's position one token ahead and prefetch its cos/sin row.
        int64_t position = 0;
        if constexpr (kFp8Out)
        {
            if (blockIdx.x == 0)
            {
                zeroFp8ScalePadding(p, t);
            }
            if (oHalf == 1)
            {
                position = ropePosition(p, blockIdx.x);
            }
        }
        int g = 0;
        int r = 0;      // blocks after a query's first (the only ones that take oReady)
        int item = 0;
        int served = 0; // blocks of this query already stepped in the previous query's epilogue
        int blocks = blockCount(keyCount(p, blockIdx.x));
        int nextBlocks = 0;
        for (int token = blockIdx.x; token < p.numTokens; token += gridDim.x, ++item, blocks = nextBlocks)
        {
            int const next = token + static_cast<int>(gridDim.x);
            nextBlocks = next < p.numTokens ? blockCount(keyCount(p, next)) : 0;
            float const* cosSin = nullptr;
            if constexpr (kFp8Out)
            {
                if (oHalf == 1)
                {
                    cosSin = p.ropeCosSin + position * 64;
                    if (lane < 2)
                    {
                        asm volatile("prefetch.global.L1 [%0];" ::"l"(cosSin + 32 * lane));
                    }
                }
            }
            for (int k = served; k < blocks; ++k)
            {
                rescaleStep<false>(sm, oBase, head, k, g, r);
            }
            served = 0;
            // Epilogue of this query: wait for its last PV and the softmax row statistics.
            int const gLast = g - 1;
            mbarWait(&sm.statsFull[item & 1], (item >> 1) & 1);
            mbarWait(&sm.pvDone[gLast & 1][0], (gLast >> 1) & 1);
            mbarWait(&sm.pvDone[gLast & 1][1], (gLast >> 1) & 1);
            tcFenceAfter();
            float const l = sm.rowSum[item & 1][head] + sm.rowSum[item & 1][head + 64];
            float const m = sm.rowMaxFinal[item & 1][head];
            mbarArrive(&sm.statsEmpty[item & 1]);
            float const oScale = bmm2 / (l + kPScale * exp2Approx(sinkLog2 - m));
            uint32_t packed[128];
            // O in two batches of 128 columns, one TMEM load wait each.
#pragma unroll
            for (int half = 0; half < 2; ++half)
            {
                float o[128];
#pragma unroll
                for (int c = 0; c < 4; ++c)
                {
                    tmemLoad32(oBase + 128 * half + 32 * c, o + 32 * c);
                }
                tmemLoadWait();
#pragma unroll
                for (int j = 0; j < 64; ++j)
                {
                    float2 const x = mulF32x2(make_float2(o[2 * j], o[2 * j + 1]), make_float2(oScale, oScale));
                    __nv_bfloat162 const b2 = __floats2bfloat162_rn(x.x, x.y);
                    packed[64 * half + j] = *reinterpret_cast<uint32_t const*>(&b2);
                }
            }
            tcFenceBefore();
            mbarArrive(&sm.oEmpty);
            if constexpr (kFp8Out)
            {
                if (oHalf == 1 && next < p.numTokens)
                {
                    position = ropePosition(p, next);
                }
                storeFp8Output<0>(p, packed, head, oHalf, token, cosSin);
                // The next query's second PV waits for the rescale of its first block's O, which these warps owe:
                // step its first two blocks between the two halves of this epilogue instead of after it.
                if (nextBlocks >= 2)
                {
                    rescaleStep<true>(sm, oBase, head, 0, g, r);
                    rescaleStep<true>(sm, oBase, head, 1, g, r);
                    served = 2;
                }
                storeFp8Output<1>(p, packed, head, oHalf, token, cosSin);
                continue;
            }
            // O leaves through a per-warp staging tile (32 heads x 32 channels, 64-byte swizzle) and TMA stores. Lane
            // half oHalf holds channels [oHalf * 128, +128) (TMEM columns 0-127) and [256 + oHalf * 128, +128)
            // (columns 128-255).
            uint8_t* const stage = sm.stage[warp - 4];
#pragma unroll
            for (int c = 0; c < 8; ++c)
            {
                if (c > 0 || item > 0)
                {
                    // The previous TMA store from this staging tile must have read it.
                    if (lane == 0)
                    {
                        asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
                    }
                    __syncwarp();
                }
#pragma unroll
                for (int j = 0; j < 4; ++j)
                {
                    *reinterpret_cast<uint4*>(stage + lane * 64 + ((j ^ ((lane >> 1) & 3)) * 16))
                        = make_uint4(packed[16 * c + 4 * j], packed[16 * c + 4 * j + 1], packed[16 * c + 4 * j + 2],
                            packed[16 * c + 4 * j + 3]);
                }
                fenceProxyAsyncSmem();
                __syncwarp();
                if (lane == 0)
                {
                    int const channel = (c >> 2) * 256 + oHalf * 128 + (c & 3) * 32;
                    tmaStore3d(&tmapOut, stage, channel, head & ~31, token);
                    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
                }
            }
        }
        if (lane == 0)
        {
            asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
        }
        __syncwarp();
    }
    else if (warp == 8)
    {
        // ======================================================================================== QK issue
        setMaxRegistersDec<56>();
        if (electOne())
        {
            constexpr uint32_t kQkDesc = mmaDescriptorF8(64, 128, 0, 0);
            // An E4M3 chunk of Q sits at chunk c of its slot, or at BF16 box 2c once quantized in place.
            constexpr int kQChunkStride = kBf16Q ? 2 * kQChunkBytes : kQChunkBytes;
            int it = 0; // ring item
            int g = 0;  // global block
            for (int token = blockIdx.x; token < p.numTokens; token += gridDim.x)
            {
                int const blocks = blockCount(keyCount(p, token));
                {
                    // Q item: shared memory -> TMEM, duplicated into both lane halves; in order after this thread's
                    // last QK, so the previous query's S no longer reads TMEM Q.
                    int const slot = it % kNumSlots;
                    uint32_t const qAddress = smemAddress(sm.slot[slot]);
#pragma unroll
                    for (int c = 0; c < 4; ++c)
                    {
                        mbarWait(&sm.slotFull[slot][c], (it / kNumSlots) & 1);
                        tcFenceAfter();
#pragma unroll
                        for (int s = 0; s < 8; ++s)
                        {
                            tmemCopy64x128Dup(tmemBase + kTmemQ + 32 * c + 4 * s,
                                smemDescriptor(qAddress + c * kQChunkStride + s * 16, 16, 1024));
                        }
                    }
                    tcCommit(&sm.slotEmpty[slot]);
                    ++it;
                }
                for (int k = 0; k < blocks; ++k, ++g)
                {
                    int const slot = it % kNumSlots;
                    int const sb = g & 1;
                    mbarWait(&sm.sEmpty[sb], ((g >> 1) & 1) ^ 1);
                    uint32_t const kAddress = smemAddress(sm.slot[slot]);
#pragma unroll
                    for (int c = 0; c < 4; ++c)
                    {
                        mbarWait(&sm.slotFull[slot][c], (it / kNumSlots) & 1);
                        tcFenceAfter();
#pragma unroll
                        for (int j = 0; j < 4; ++j)
                        {
                            mmaWsTmemA(tmemBase + kTmemS + 64 * sb, tmemBase + kTmemQ + 32 * c + 8 * j,
                                smemDescriptor(kAddress + c * kKvChunkBytes + j * 32, 16, 1024), kQkDesc,
                                (c | j) ? 1U : 0U);
                        }
                    }
                    tcCommit(&sm.sFull[sb]);
                    ++it;
                }
            }
        }
    }
    else if (warp == 12)
    {
        // ======================================================================================== PV issue
        // A second issuing thread lets QK of later blocks fill the tensor core while PV waits for the softmax and
        // the O rescale.
        setMaxRegistersDec<56>();
        if (electOne())
        {
            constexpr uint32_t kPvDesc = mmaDescriptorF8(64, 256, 0, 1);
            int it = 0;
            int g = 0;
            int r = 0;
            int item = 0;
            for (int token = blockIdx.x; token < p.numTokens; token += gridDim.x, ++item)
            {
                int const blocks = blockCount(keyCount(p, token));
                ++it; // Q item
                for (int k = 0; k < blocks; ++k, ++g)
                {
                    int const slot = it % kNumSlots;
                    int const sb = g & 1;
                    bool const first = k == 0;
                    mbarWait(&sm.pFull[sb], (g >> 1) & 1);
                    if (first && item > 0)
                    {
                        mbarWait(&sm.oEmpty, (item - 1) & 1);
                    }
                    uint32_t const pAddress = smemAddress(sm.p[sb]);
                    uint32_t const vAddress = smemAddress(sm.slot[slot]);
#pragma unroll
                    for (int h = 0; h < 2; ++h)
                    {
                        // A query's first PV overwrites O, which the previous epilogue has read (oEmpty above); later
                        // ones wait for the O rescale. So the first PV does not wait for the previous epilogue's
                        // stores, which run on the rescale warps.
                        if (!first)
                        {
                            mbarWait(&sm.oReady[r & 1][h], (r >> 1) & 1);
                        }
                        tcFenceAfter();
#pragma unroll
                        for (int j = 0; j < 4; ++j)
                        {
                            uint64_t const aDesc = smemDescriptor(pAddress + j * 32, 16, 1024);
                            uint64_t const bDesc
                                = smemDescriptor(vAddress + (2 * h) * kKvChunkBytes + j * 4096, kKvChunkBytes, 1024);
                            mmaWsSmemA(tmemBase + kTmemO + 128 * h, aDesc, bDesc, kPvDesc, (first && j == 0) ? 0U : 1U);
                        }
                        tcCommit(&sm.pvDone[sb][h]);
                    }
                    r += first ? 0 : 1;
                    // The slot is free once PV(g) completed (QK(g) completed before P(g) existed).
                    tcCommit(&sm.slotEmpty[slot]);
                    tcCommit(&sm.pEmpty[sb]);
                    ++it;
                }
            }
        }
    }
    else
    {
        // ======================================================================================== producers
        setMaxRegistersDec<56>();
        int const pw = warp < 12 ? warp - 9 : warp - 10;
        int const producerThread = pw * 32 + lane;
        // Gather instruction i = lane * kNumProducerWarps + pw covers row group i % 32 (4 keys) and channel chunk
        // i / 32.
        int const gatherIndex = lane * kNumProducerWarps + pw;
        bool const gathers = gatherIndex < 128;
        int const rowGroup = gatherIndex & 31;
        int const chunk = gatherIndex >> 5;
        float const qScale = kBf16Q ? __ldg(p.qScale) : 1.F;
        // Indices are prefetched one block ahead, so their load latency overlaps the previous block.
        auto loadRows = [&](int token, int k, int keys) -> int4
        {
            int32_t const* const rowIndices = p.indices + static_cast<int64_t>(token) * p.indexStride;
            int const pos = k * kBlockKeys + lane * 4;
            int4 r;
            if (pos + 3 < keys)
            {
                r = __ldg(reinterpret_cast<int4 const*>(rowIndices + pos));
            }
            else
            {
                r.x = (pos + 0 < keys) ? __ldg(rowIndices + pos + 0) : 0;
                r.y = (pos + 1 < keys) ? __ldg(rowIndices + pos + 1) : 0;
                r.z = (pos + 2 < keys) ? __ldg(rowIndices + pos + 2) : 0;
                r.w = (pos + 3 < keys) ? __ldg(rowIndices + pos + 3) : 0;
            }
            return r;
        };
        int it = 0;
        // Every producer warp observes every phase of its slots' slotEmpty barriers (a skipped phase would alias the
        // parity). Warp 9 arms the item's chunk barriers with its bytes; every other producer warp arrives on the last
        // chunk's barrier once all of its lanes have waited (the wait ends in a warp sync, so lane 0 cannot arrive or
        // issue for lanes still waiting). QK reads that chunk last, so even an item that warp 9 loads alone (E4M3 Q, a
        // contiguous block) cannot be consumed and its slot released before every producer lane has waited on the
        // slot; otherwise a lagging lane's parity wait would match the following phase.
        auto waitSlotEmpty = [&](int slot)
        {
            mbarWait(&sm.slotEmpty[slot], ((it / kNumSlots) & 1) ^ 1);
            __syncwarp();
        };
        auto arriveOtherWarp = [&](int slot)
        {
            if (pw != 0 && lane == 0)
            {
                mbarArrive(&sm.slotFull[slot][3]);
            }
        };
        [[maybe_unused]] int qItem = 0; // BF16 Q items converted (parity of the qRaw barriers)
        // Current block (token, k) with `keys` keys; the next block is prefetched into nextRows.
        int token = blockIdx.x;
        int k = 0;
        int keys = token < p.numTokens ? keyCount(p, token) : 1;
        int4 rows = token < p.numTokens ? loadRows(token, 0, keys) : make_int4(0, 0, 0, 0);
        while (token < p.numTokens)
        {
            int const blocks = blockCount(keys);
            int nextToken = token;
            int nextK = k + 1;
            int nextKeys = keys;
            if (nextK >= blocks)
            {
                nextToken = token + gridDim.x;
                nextK = 0;
                nextKeys = nextToken < p.numTokens ? keyCount(p, nextToken) : 1;
            }
            int4 const nextRows
                = nextToken < p.numTokens ? loadRows(nextToken, nextK, nextKeys) : make_int4(0, 0, 0, 0);
            if (k == 0)
            {
                // Q item.
                int const slot = it % kNumSlots;
                waitSlotEmpty(slot);
                int const prefetchToken = token + kQPrefetchItems * gridDim.x;
                if constexpr (kBf16Q)
                {
                    if (producerThread == 0)
                    {
#pragma unroll
                        for (int c = 0; c < 4; ++c)
                        {
                            mbarArriveExpectTx(&sm.qRaw[c], 2 * kQChunkBytes);
                            tmaLoad3d(&tmapQ, &sm.qRaw[c], sm.slot[slot] + (2 * c) * kQChunkBytes, 128 * c, 0, token);
                            tmaLoad3d(&tmapQ, &sm.qRaw[c], sm.slot[slot] + (2 * c + 1) * kQChunkBytes, 128 * c + 64, 0,
                                token);
                        }
                        if (prefetchToken < p.numTokens)
                        {
#pragma unroll
                            for (int c = 0; c < 8; ++c)
                            {
                                tmaPrefetch3d(&tmapQ, 64 * c, 0, prefetchToken);
                            }
                        }
                    }
                    // 256 (chunk, row) items over the producer threads; each converts one row of one chunk in
                    // place. Converting here, before the first KV block is issued, uses the time the producers
                    // would otherwise wait for that block's slot.
                    for (int i = producerThread; i < 4 * kHeads; i += kNumProducerThreads)
                    {
                        int const c = i / kHeads;
                        mbarWait(&sm.qRaw[c], qItem & 1);
                        quantizeQRow(sm.slot[slot], c, i % kHeads, qScale);
                    }
                    fenceProxyAsyncSmem();
                    namedBarrierSync(kProducerBarrier, kNumProducerThreads);
                    if (producerThread == 0)
                    {
#pragma unroll
                        for (int c = 0; c < 4; ++c)
                        {
                            mbarArrive(&sm.slotFull[slot][c]);
                        }
                    }
                    arriveOtherWarp(slot);
                    ++qItem;
                }
                else
                {
                    arriveOtherWarp(slot);
                    if (producerThread == 0)
                    {
#pragma unroll
                        for (int c = 0; c < 4; ++c)
                        {
                            mbarArriveExpectTx(&sm.slotFull[slot][c], kQChunkBytes);
                            tmaLoad3d(
                                &tmapQ, &sm.slotFull[slot][c], sm.slot[slot] + c * kQChunkBytes, 128 * c, 0, token);
                        }
                        if (prefetchToken < p.numTokens)
                        {
#pragma unroll
                            for (int c = 0; c < 4; ++c)
                            {
                                tmaPrefetch3d(&tmapQ, 128 * c, 0, prefetchToken);
                            }
                        }
                    }
                }
                ++it;
            }
            {
                int const slot = it % kNumSlots;
                int4 r = rows;
                r.x = (r.x < 0 || r.x >= p.bankRows) ? 0 : r.x;
                r.y = (r.y < 0 || r.y >= p.bankRows) ? 0 : r.y;
                r.z = (r.z < 0 || r.z >= p.bankRows) ? 0 : r.z;
                r.w = (r.w < 0 || r.w >= p.bankRows) ? 0 : r.w;
                // A block of 128 consecutive bank rows (an SWA window) loads as four dense tiles.
                int const base = __shfl_sync(0xFFFFFFFFU, r.x, 0);
                bool const mine = (r.x == base + 4 * lane) && (r.y == r.x + 1) && (r.z == r.x + 2) && (r.w == r.x + 3)
                    && (base + kBlockKeys - 1 < p.bankRows);
                bool const contiguous = __all_sync(0xFFFFFFFFU, mine);
                int4 const gatherRows
                    = make_int4(__shfl_sync(0xFFFFFFFFU, r.x, rowGroup), __shfl_sync(0xFFFFFFFFU, r.y, rowGroup),
                        __shfl_sync(0xFFFFFFFFU, r.z, rowGroup), __shfl_sync(0xFFFFFFFFU, r.w, rowGroup));
                waitSlotEmpty(slot);
                if (producerThread == 0)
                {
#pragma unroll
                    for (int c = 0; c < 4; ++c)
                    {
                        mbarArriveExpectTx(&sm.slotFull[slot][c], kKvChunkBytes);
                    }
                }
                arriveOtherWarp(slot);
                if (contiguous)
                {
                    if (pw == 0 && lane < 4)
                    {
                        tmaLoad2d(&tmapTile, &sm.slotFull[slot][lane], sm.slot[slot] + lane * kKvChunkBytes, lane * 128,
                            base);
                    }
                }
                else if (gathers)
                {
                    tmaGather4(&tmapBank, &sm.slotFull[slot][chunk],
                        sm.slot[slot] + chunk * kKvChunkBytes + rowGroup * 512, chunk * 128, gatherRows);
                }
                ++it;
            }
            token = nextToken;
            k = nextK;
            keys = nextKeys;
            rows = nextRows;
        }
    }

    // Teardown: every role is done and the last commits have been waited on by their consumers.
    tcFenceBefore();
    __syncthreads();
    if (warp == 0)
    {
        tcFenceAfter();
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmemBase), "r"(kTmemCols));
    }
#else
    asm volatile("trap;");
#endif
}

} // namespace kernels::csa2SparseContext

TRTLLM_NAMESPACE_END
