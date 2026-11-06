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

// One engine runs every kernel of the step. A kernel is an Op: the list of its work items (each a sequence of 16-row
// weight tiles), how to build its resident input rows (the prologue), and what to do with each tile's 16 x 8 fp32
// outputs (the epilogue) and at the end of each item. The engine's producer warp walks the CTA's items and issues one
// cp.async.bulk per weight row into a ring of stages; eight consumer warps
// split each stage's k range, run the MMAs with the weight tile as A and the (up to eight) input rows as B, and the
// warps' partial sums are added in warp order into the tile's outputs.

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/glm5MtpStep/glm5MtpStep.h"
#include "tensorrt_llm/kernels/lmHeadArgmax/lmHeadArgmax.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace glm5_mtp_step
{
namespace
{

constexpr int kConsumerWarps = 8;
constexpr int kConsumers = kConsumerWarps * 32;
constexpr int kThreads = kConsumers + 32; // + the producer warp
constexpr int kTileRows = 16;
constexpr int kSmemMax = 227 * 1024;
constexpr int kMaxStages = 8;
constexpr int kTileOutWords = kTileRows * kMaxRows;
constexpr int kPartialWords = kConsumerWarps * kTileOutWords;
constexpr int kBlock = 128; // FP8 scale block
constexpr float kFp8Max = 448.f;

enum TileKind : int
{
    kBf16 = 0,
    kFp8 = 1,
};

//! One 16-row weight tile: rows 0-7 from w, rows 8-15 from w2 (or continuing w when w2 is null).
struct Tile
{
    uint8_t const* w;
    uint8_t const* w2;
    int64_t wStride;   // bytes between consecutive rows
    int rows;          // valid rows (<= 16)
    int rowBytes;      // the k extent in bytes
    int kind;
    uint32_t in;       // resident input: shared address of row 0 at this tile's first k
    int inStride;      // resident input row stride (bytes)
    int const* rowIdx; // shared: the input row of each token column (null: row t for token t)
    float const* ws;   // FP8: weight block scales along k of rows 0-7
    float const* ws2;  // FP8: of rows 8-15
    float const* xs;   // FP8: input block scales [token][kb]
    int xsStride;
};

//! A tile's outputs: element r * kMaxRows + t (weight row r, token t) is the sum of the consumer warps' partials,
//! taken by the reading thread itself (no barrier between the partials and the epilogue).
struct TileView
{
    float const* partial; // [kConsumerWarps][kTileOutWords]

    __device__ __forceinline__ float operator[](int i) const
    {
        float v = 0.f;
#pragma unroll
        for (int w = 0; w < kConsumerWarps; ++w)
        {
            v += partial[w * kTileOutWords + i];
        }
        return v;
    }
};

__host__ __device__ constexpr size_t align128(size_t x)
{
    return (x + 127) / 128 * 128;
}

template <class Op>
__host__ __device__ constexpr int ringRowBytes()
{
    return Op::kChunkMax + 16;
}

//! FP8 weight block scales of one chunk and row group (rows 0-7 / 8-15), staged after the chunk's rows.
template <class Op>
__host__ __device__ constexpr int stageScaleWords()
{
    return Op::kChunkMax / 128 > 4 ? Op::kChunkMax / 128 : 4; // at least 16 bytes per bulk copy
}

template <class Op>
__host__ __device__ constexpr size_t stageBytes()
{
    return align128(static_cast<size_t>(kTileRows) * ringRowBytes<Op>() + 2 * stageScaleWords<Op>() * sizeof(float));
}

__host__ __device__ constexpr size_t engineFixedBytes()
{
    return 2 * kPartialWords * sizeof(float) + (2 * kMaxStages + 1) * sizeof(uint64_t);
}

template <class Op>
__host__ __device__ inline int chunkBytes(int rowBytes)
{
    int c = Op::kChunkMax;
    while (rowBytes % c)
    {
        c >>= 1;
    }
    return c;
}

//! The routing in shared memory (see computeRouting).
struct alignas(16) RoutingSmem
{
    int sel[kMaxRows * kMaxTopK];
    float weight[kMaxRows * kMaxTopK];
    int slotExpert[kMaxSlots];
    float slotWeight[kMaxSlots * kMaxRows];
    int warpCount[kConsumerWarps];
    int count;
};

__device__ __forceinline__ uint32_t smemAddr(void const* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldmatrixX4(uint32_t* r, uint32_t a)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
#endif
}

__device__ __forceinline__ void ldmatrixX2(uint32_t* r, uint32_t a)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
#endif
}

__device__ __forceinline__ void mmaBf16(float* c, uint32_t const* a, uint32_t const* b)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

// The e4m3 fragments of m16n8k32 hold the same bytes as the bf16 fragments of m16n8k16, so ldmatrix loads both.
__device__ __forceinline__ void mmaFp8(float* c, uint32_t const* a, uint32_t const* b)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

__device__ __forceinline__ void barInit(uint64_t* b, int count)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(smemAddr(b)), "r"(count));
#endif
}

__device__ __forceinline__ void barExpectTx(uint64_t* b, uint32_t bytes)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(smemAddr(b)), "r"(bytes) : "memory");
#endif
}

__device__ __forceinline__ void barArrive(uint64_t* b)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" ::"r"(smemAddr(b)) : "memory");
#endif
}

__device__ __forceinline__ void barWait(uint64_t* b, uint32_t phase)
{
    uint32_t done = 0;
    while (!done)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        asm volatile("{ .reg .pred P; mbarrier.try_wait.parity.shared.b64 P, [%1], %2; selp.u32 %0, 1, 0, P; }\n"
                     : "=r"(done)
                     : "r"(smemAddr(b)), "r"(phase)
                     : "memory");
#endif
    }
}

__device__ __forceinline__ void bulkCopy(void* dst, void const* src, uint32_t bytes, uint64_t* bar)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
                     smemAddr(dst)),
                 "l"(src), "r"(bytes), "r"(smemAddr(bar))
                 : "memory");
#endif
}

__device__ __forceinline__ uint64_t evictFirstPolicy()
{
    uint64_t policy = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(policy));
#endif
    return policy;
}

__device__ __forceinline__ void bulkCopyEvictFirst(
    void* dst, void const* src, uint32_t bytes, uint64_t* bar, uint64_t policy)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], "
        "%4;\n" ::"r"(smemAddr(dst)),
        "l"(src), "r"(bytes), "r"(smemAddr(bar)), "l"(policy)
        : "memory");
#endif
}

__device__ __forceinline__ void consumersSync()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("bar.sync 1, 256;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void pdlWait()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void pdlLaunchDependents()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}

__device__ __forceinline__ float bf(uint16_t x)
{
    return __uint_as_float(static_cast<uint32_t>(x) << 16);
}

__device__ __forceinline__ uint16_t toBf(float x)
{
    return __bfloat16_as_ushort(__float2bfloat16_rn(x));
}

__device__ __forceinline__ float roundBf(float x)
{
    return bf(toBf(x));
}

__device__ __forceinline__ float silu(float x)
{
    return x / (1.f + expf(-x));
}

//! Clamped SwiGLU: silu(min(gate, limit)) * clamp(up, -limit, limit).
__device__ __forceinline__ float swiglu(float gate, float up, float limit)
{
    return silu(fminf(gate, limit)) * fminf(fmaxf(up, -limit), limit);
}

__device__ __forceinline__ float warpSum(float v)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
    {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    return v;
}

__device__ __forceinline__ float warpMax(float v)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
    {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    }
    return v;
}

__device__ __forceinline__ void load8(uint16_t const* src, float* v)
{
    uint4 const x = *reinterpret_cast<uint4 const*>(src);
    uint32_t const w[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        v[2 * i] = bf(static_cast<uint16_t>(w[i] & 0xffffu));
        v[2 * i + 1] = bf(static_cast<uint16_t>(w[i] >> 16));
    }
}

__device__ __forceinline__ void store8(uint16_t* dst, float const* v)
{
    uint4 x;
    x.x = toBf(v[0]) | (static_cast<uint32_t>(toBf(v[1])) << 16);
    x.y = toBf(v[2]) | (static_cast<uint32_t>(toBf(v[3])) << 16);
    x.z = toBf(v[4]) | (static_cast<uint32_t>(toBf(v[5])) << 16);
    x.w = toBf(v[6]) | (static_cast<uint32_t>(toBf(v[7])) << 16);
    *reinterpret_cast<uint4*>(dst) = x;
}

//! Copies m rows of rowBytes from global into shared rows (16-byte multiples), with all consumer threads.
__device__ __forceinline__ void loadRows(
    uint8_t* dst, int dstStride, uint8_t const* src, int64_t srcStride, int m, int rowBytes, int tid)
{
    int const perRow = rowBytes / 16;
    for (int v = tid; v < m * perRow; v += kConsumers)
    {
        int const r = v / perRow;
        int const c = (v % perRow) * 16;
        *reinterpret_cast<uint4*>(dst + r * dstStride + c) = *reinterpret_cast<uint4 const*>(src + r * srcStride + c);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The Lamport all-reduce

__device__ __forceinline__ uint16_t* arSlot(AllReduce const& ar, int peer, int buf, int src, int row)
{
    return static_cast<uint16_t*>(ar.peers[peer])
        + ((static_cast<size_t>(buf) * kMaxRanks + src) * kMaxRows + row) * kHidden;
}

__device__ __forceinline__ uint4 ldVolatile(uint16_t const* p)
{
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
#endif
    return v;
}

__device__ __forceinline__ void stVolatile(uint16_t* p, uint4 v)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
#endif
}

__device__ __forceinline__ bool hasEmpty(uint4 v)
{
    uint32_t const w[4] = {v.x, v.y, v.z, v.w};
    bool e = false;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        e |= (w[i] & 0xffffu) == 0x8000u || (w[i] >> 16) == 0x8000u;
    }
    return e;
}

__device__ __forceinline__ uint32_t positiveZero(uint32_t w)
{
    uint32_t lo = w & 0xffffu;
    uint32_t hi = w >> 16;
    lo = lo == 0x8000u ? 0u : lo;
    hi = hi == 0x8000u ? 0u : hi;
    return lo | (hi << 16);
}

//! Pushes this rank's partial of output columns [n0, n0 + 16) (sPush: [kMaxRows][16] bf16) to every rank's buffer.
__device__ __forceinline__ void arPush(AllReduce const& ar, int buf, int m, int n0, uint16_t const* sPush, int tid)
{
    int const per = 2 * m;
    if (tid < ar.ranks * per)
    {
        int const peer = tid / per;
        int const t = (tid % per) >> 1;
        int const half = tid & 1;
        uint4 v = *reinterpret_cast<uint4 const*>(sPush + t * 16 + half * 8);
        v.x = positiveZero(v.x);
        v.y = positiveZero(v.y);
        v.z = positiveZero(v.z);
        v.w = positiveZero(v.w);
        stVolatile(arSlot(ar, peer, buf, ar.rank, t) + n0 + half * 8, v);
    }
}

//! The all-reduce of buffer buf into the resident input: the ranks' partials summed in fp32 in rank order -> bf16,
//! + residualIn (bf16 add) -> residualOut (CTA 0), then RMSNorm(gamma) in place (and to normOut from CTA 0). Re-marks
//! buffer (buf + 2) % 3 as empty. rowIdx (optional): output row t reads all-reduce row and residualIn row rowIdx[t].
__device__ void arReduceNorm(AllReduce const& ar, int buf, int m, uint16_t const* residualIn, uint16_t* residualOut,
    uint16_t const* gamma, float eps, uint8_t* sIn, int sStride, uint16_t* normOut, int64_t normStride, float* sRed,
    int tid, int const* rowIdx = nullptr)
{
    int const warp = tid >> 5;
    int const lane = tid & 31;
    float ss[kMaxRows];
#pragma unroll
    for (int i = 0; i < 2 * kMaxRows; ++i)
    {
        int const row = i >> 1;
        if ((i & 1) == 0)
        {
            ss[row] = 0.f;
        }
        if (row < m)
        {
            int const col = ((i & 1) * kConsumers + tid) * 8;
            float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            for (int r = 0; r < ar.ranks; ++r)
            {
                uint16_t const* src = arSlot(ar, ar.rank, buf, r, rowIdx != nullptr ? rowIdx[row] : row) + col;
                uint4 v;
                do
                {
                    v = ldVolatile(src);
                } while (hasEmpty(v));
                uint32_t const w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (int j = 0; j < 4; ++j)
                {
                    acc[2 * j] += bf(static_cast<uint16_t>(w[j] & 0xffffu));
                    acc[2 * j + 1] += bf(static_cast<uint16_t>(w[j] >> 16));
                }
            }
#pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                acc[j] = roundBf(acc[j]);
            }
            if (residualIn != nullptr)
            {
                float res[8];
                load8(residualIn + static_cast<size_t>(rowIdx != nullptr ? rowIdx[row] : row) * kHidden + col, res);
#pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    acc[j] = roundBf(acc[j] + res[j]);
                }
            }
            uint16_t* dst = reinterpret_cast<uint16_t*>(sIn + row * sStride) + col;
            store8(dst, acc);
            if (blockIdx.x == 0 && residualOut != nullptr)
            {
                store8(residualOut + static_cast<size_t>(row) * kHidden + col, acc);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                ss[row] += acc[j] * acc[j];
            }
        }
    }
#pragma unroll
    for (int row = 0; row < kMaxRows; ++row)
    {
        if (row < m)
        {
            float const s = warpSum(ss[row]);
            if (lane == 0)
            {
                sRed[warp * kMaxRows + row] = s;
            }
        }
    }
    // Re-mark the buffer of the previous all-reduce (its readers are done) while the sums settle.
    {
        int const old = (buf + 2) % kArBuffers;
        int const vectors = ar.ranks * kMaxRows * kHidden / 8;
        uint4 const mark = make_uint4(0x80008000u, 0x80008000u, 0x80008000u, 0x80008000u);
        for (int v = blockIdx.x * kConsumers + tid; v < vectors; v += gridDim.x * kConsumers)
        {
            int const src = v / (kMaxRows * kHidden / 8);
            int const rest = v % (kMaxRows * kHidden / 8);
            stVolatile(arSlot(ar, ar.rank, old, src, 0) + rest * 8, mark);
        }
    }
    consumersSync();
#pragma unroll
    for (int i = 0; i < 2 * kMaxRows; ++i)
    {
        int const row = i >> 1;
        if (row < m)
        {
            float s = 0.f;
#pragma unroll
            for (int w = 0; w < kConsumerWarps; ++w)
            {
                s += sRed[w * kMaxRows + row];
            }
            float const rstd = rsqrtf(s / kHidden + eps);
            int const col = ((i & 1) * kConsumers + tid) * 8;
            uint16_t* dst = reinterpret_cast<uint16_t*>(sIn + row * sStride) + col;
            float x[8];
            float g[8];
            load8(dst, x);
            load8(gamma + col, g);
#pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                x[j] = x[j] * rstd * g[j];
            }
            store8(dst, x);
            if (blockIdx.x == 0 && normOut != nullptr)
            {
                store8(normOut + row * normStride + col, x);
            }
        }
    }
}

//! RMSNorm of one row of n elements (n % 256 == 0) by one warp: src (global) -> dst (generic), fp32 math.
__device__ void rmsNormRowWarp(uint16_t const* src, int n, uint16_t const* gamma, float eps, uint16_t* dst, int lane)
{
    float ss = 0.f;
    for (int c = lane * 8; c < n; c += 256)
    {
        float x[8];
        load8(src + c, x);
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            ss += x[j] * x[j];
        }
    }
    float const rstd = rsqrtf(warpSum(ss) / n + eps);
    for (int c = lane * 8; c < n; c += 256)
    {
        float x[8];
        float g[8];
        load8(src + c, x);
        load8(gamma + c, g);
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            x[j] = x[j] * rstd * g[j];
        }
        store8(dst + c, x);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Routing (sigmoid scores, top-k by score + correction bias, weights = scores normalized over the top-k x scale),
// computed once per call by the router. This rank's experts that any row selects become slots 0, 1, ... in expert
// order.

//! A float as an unsigned key with the same order.
__device__ __forceinline__ unsigned orderedKey(float v)
{
    unsigned const u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

//! byRow: shared scratch of kConsumers x kMaxRows floats (each local expert's weight per row).
__device__ void computeRouting(Routing const& rt, int m, RoutingSmem* s, float* byRow, int tid)
{
    int const warp = tid >> 5;
    int const lane = tid & 31;
    constexpr int kPerLane = kMaxExperts / 32;
    for (int i = tid; i < rt.localExperts * kMaxRows; i += kConsumers)
    {
        byRow[i] = 0.f;
    }
    if (warp < m)
    {
        // Every load of the lane first (they overlap), then the scores.
        float logit[kPerLane];
        float bias[kPerLane];
#pragma unroll
        for (int j = 0; j < kPerLane; ++j)
        {
            int const e = lane + 32 * j;
            bool const valid = e < rt.experts;
            logit[j] = valid ? __ldcg(rt.logits + warp * rt.experts + e) : 0.f; // written by other CTAs
            bias[j] = valid ? __ldg(rt.bias + e) : 0.f;
        }
        float score[kPerLane];
        float choice[kPerLane];
#pragma unroll
        for (int j = 0; j < kPerLane; ++j)
        {
            score[j] = 1.f / (1.f + expf(-logit[j]));
            choice[j] = lane + 32 * j < rt.experts ? score[j] + bias[j] : -INFINITY;
        }
        // Top-k: the largest choice, ties to the lowest expert, by two hardware warp reductions per pick.
        float sum = 0.f;
        for (int k = 0; k < rt.topK; ++k)
        {
            float lv = -INFINITY;
            int li = 0x7fffffff;
#pragma unroll
            for (int j = 0; j < kPerLane; ++j)
            {
                if (choice[j] > lv) // j ascending: a tie keeps the lower expert
                {
                    lv = choice[j];
                    li = lane + 32 * j;
                }
            }
            unsigned const key = orderedKey(lv);
            unsigned const best = __reduce_max_sync(0xffffffffu, key);
            int const bi = __reduce_min_sync(0xffffffffu, key == best ? li : 0x7fffffff);
            float picked = 0.f;
#pragma unroll
            for (int j = 0; j < kPerLane; ++j)
            {
                if (lane + 32 * j == bi)
                {
                    choice[j] = -INFINITY;
                    picked = score[j];
                }
            }
            picked = __shfl_sync(0xffffffffu, picked, bi & 31);
            sum += picked;
            if (lane == 0)
            {
                s->sel[warp * kMaxTopK + k] = bi;
                s->weight[warp * kMaxTopK + k] = picked;
            }
        }
        __syncwarp();
        if (lane < rt.topK)
        {
            float const w = s->weight[warp * kMaxTopK + lane] / sum * rt.scale;
            s->weight[warp * kMaxTopK + lane] = w;
            int const local = s->sel[warp * kMaxTopK + lane] - rt.localStart;
            if (local >= 0 && local < rt.localExperts)
            {
                byRow[local * kMaxRows + warp] = w;
            }
        }
    }
    consumersSync();
    // Thread j owns local expert localStart + j (localExperts <= kConsumers); its slot is the number of selected local
    // experts before it (a block-wide prefix count).
    bool const mine = tid < rt.localExperts;
    float w[kMaxRows];
    bool used = false;
    if (mine)
    {
        float4 const lo = reinterpret_cast<float4 const*>(byRow + tid * kMaxRows)[0];
        float4 const hi = reinterpret_cast<float4 const*>(byRow + tid * kMaxRows)[1];
        w[0] = lo.x, w[1] = lo.y, w[2] = lo.z, w[3] = lo.w, w[4] = hi.x, w[5] = hi.y, w[6] = hi.z, w[7] = hi.w;
#pragma unroll
        for (int t = 0; t < kMaxRows; ++t)
        {
            used |= w[t] != 0.f;
        }
    }
    unsigned const ballot = __ballot_sync(0xffffffffu, used);
    if (lane == 0)
    {
        s->warpCount[warp] = __popc(ballot);
    }
    consumersSync();
    int slot = __popc(ballot & ((1u << lane) - 1u));
    int total = 0;
#pragma unroll
    for (int v = 0; v < kConsumerWarps; ++v)
    {
        slot += v < warp ? s->warpCount[v] : 0;
        total += s->warpCount[v];
    }
    if (used && slot < kMaxSlots)
    {
        s->slotExpert[slot] = rt.localStart + tid;
#pragma unroll
        for (int t = 0; t < kMaxRows; ++t)
        {
            if (t < m)
            {
                s->slotWeight[slot * kMaxRows + t] = w[t];
            }
        }
    }
    if (tid == 0)
    {
        s->count = min(total, kMaxSlots);
    }
    consumersSync();
    {
    }
}

//! Each slot's tokens (the rows its expert serves), as (slot, token, weight) pairs in slot order; per slot and token
//! the pair's index, or zeroRow when the token does not select the slot's expert.
struct alignas(16) RoutePairs
{
    static constexpr int kMax = kMaxRows * kMaxTopK;
    int rowIdx[kMaxSlots * kMaxRows];
    int slotCount[kMaxSlots];
    int slot[kMax];
    int token[kMax];
    float weight[kMax];
    int total;
};

//! The routing of one call: computed once by the router, read by fc1 and fc2.
struct RouteTable
{
    RoutingSmem routing;
    RoutePairs pairs;
};

static_assert(sizeof(RouteTable) / 4 + 1 <= kRouteWords, "route table too large");

//! The word after the table: the router's count of finished logits tiles.
__host__ __device__ constexpr int routeCounterWord()
{
    return static_cast<int>(sizeof(RouteTable) / 4);
}

//! zeroRow = min(m * topK, RoutePairs::kMax): fc2's activation row of zeros.
__device__ void computePairs(RoutingSmem const* r, RoutePairs* q, int m, int zeroRow, int tid)
{
    int const slots = r->count;
    int count = 0;
    if (tid < slots)
    {
        for (int t = 0; t < m; ++t)
        {
            count += r->slotWeight[tid * kMaxRows + t] != 0.f;
        }
        q->slotCount[tid] = count;
    }
    consumersSync();
    if (tid < slots)
    {
        int j = 0;
        for (int v = 0; v < tid; ++v)
        {
            j += q->slotCount[v];
        }
        int const base = j;
        for (int t = 0; t < kMaxRows; ++t)
        {
            float const w = t < m ? r->slotWeight[tid * kMaxRows + t] : 0.f;
            int row = zeroRow;
            if (w != 0.f && j < zeroRow)
            {
                q->slot[j] = tid;
                q->token[j] = t;
                q->weight[j] = w;
                row = j++;
            }
            q->rowIdx[tid * kMaxRows + t] = row;
        }
        if (tid == slots - 1)
        {
            q->total = min(base + count, zeroRow);
        }
    }
    if (tid == 0 && slots == 0)
    {
        q->total = 0;
    }
    consumersSync();
}

//! Copies bytes (a multiple of 16) between global and shared memory with the consumer threads.
__device__ __forceinline__ void copy16(void* dst, void const* src, int bytes, int tid, bool fromGlobal)
{
    for (int i = tid; i < bytes / 16; i += kConsumers)
    {
        reinterpret_cast<uint4*>(dst)[i]
            = fromGlobal ? __ldcg(reinterpret_cast<uint4 const*>(src) + i) : reinterpret_cast<uint4 const*>(src)[i];
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The engine

template <class Op>
__global__ void __launch_bounds__(kThreads, 1) engineKernel(__grid_constant__ typename Op::Params const p, int ns)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    extern __shared__ __align__(128) uint8_t smem[];
    constexpr int kRingRow = ringRowBytes<Op>();
    constexpr size_t kStage = stageBytes<Op>();
    uint8_t* ring = smem + align128(Op::smemBytes(p));
    float* partial = reinterpret_cast<float*>(ring + static_cast<size_t>(ns) * kStage); // two buffers
    uint64_t* full = reinterpret_cast<uint64_t*>(partial + 2 * kPartialWords);
    uint64_t* empty = full + kMaxStages;
    uint64_t* prologueDone = empty + kMaxStages;

    int const tid = threadIdx.x;
    int const warp = tid >> 5;
    int const lane = tid & 31;
    Op op(p, smem);
    int const m = p.m;

    if (tid == 0)
    {
        for (int s = 0; s < ns; ++s)
        {
            barInit(&full[s], 1);
            barInit(&empty[s], kConsumerWarps);
        }
        barInit(prologueDone, 1);
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();

    if (warp == kConsumerWarps)
    {
        // The producer. A kernel whose tiles do not depend on its inputs starts streaming its weights before the
        // previous kernel finishes; otherwise it waits for the prologue (which waited for the previous kernel).
        uint64_t const policy = Op::kEvictFirst ? evictFirstPolicy() : 0;
        if (!Op::kStaticTiles)
        {
            barWait(prologueDone, 0);
        }
        int i = 0;
        int const items = op.numItems();
        for (int item = blockIdx.x; item < items; item += gridDim.x)
        {
            int const subs = op.numSubs(item);
            for (int sub = 0; sub < subs; ++sub)
            {
                Tile const d = op.tile(item, sub);
                int const cb = chunkBytes<Op>(d.rowBytes);
                int const chunks = d.rowBytes / cb;
                for (int c = 0; c < chunks; ++c, ++i)
                {
                    int const s = i % ns;
                    if (i >= ns)
                    {
                        barWait(&empty[s], ((i / ns) - 1) & 1);
                    }
                    uint8_t* stage = ring + static_cast<size_t>(s) * kStage;
                    // FP8: the chunk's weight block scales of both row groups ride along (16-byte multiples: a chunk
                    // of at least 512 columns).
                    uint32_t const scaleBytes = d.kind != kBf16 ? static_cast<uint32_t>(cb / kBlock) * 4u : 0u;
                    if (lane == 0)
                    {
                        barExpectTx(&full[s], static_cast<uint32_t>(d.rows) * cb + 2u * scaleBytes);
                    }
                    __syncwarp();
                    if (scaleBytes != 0 && (lane == kTileRows || lane == kTileRows + 1))
                    {
                        float const* src = (lane == kTileRows ? d.ws : d.ws2) + c * cb / kBlock;
                        float* dst = reinterpret_cast<float*>(stage + kTileRows * kRingRow)
                            + (lane - kTileRows) * stageScaleWords<Op>();
                        bulkCopy(dst, src, scaleBytes, &full[s]);
                    }
                    if (lane < d.rows)
                    {
                        uint8_t const* src
                            = (lane < 8 || d.w2 == nullptr) ? d.w + lane * d.wStride : d.w2 + (lane - 8) * d.wStride;
                        if (Op::kEvictFirst)
                        {
                            bulkCopyEvictFirst(stage + lane * kRingRow, src + c * cb, cb, &full[s], policy);
                        }
                        else
                        {
                            bulkCopy(stage + lane * kRingRow, src + c * cb, cb, &full[s]);
                        }
                    }
                }
            }
        }
        return;
    }

    pdlWait();
    pdlLaunchDependents();
    op.prologue(tid);
    consumersSync();
    if (tid == 0)
    {
        barArrive(prologueDone);
    }
    if (Op::kLatePrologue)
    {
        // Consumer-side setup that the producer does not need: it overlaps the first weight chunks.
        op.prologueLate(tid);
        consumersSync();
    }
    int const t0 = (lane & 3) * 2;
    int i = 0;
    int pb = 0; // the partial buffer of the next tile
    int const items = op.numItems();
    float acc[4];
    // One barrier per tile output: the partials alternate between two buffers, so the next tile's writes never
    // meet this tile's readers (who pass the next tile's barrier only after reading).
    auto const finishTile = [&](int item, int sub)
    {
        float* buffer = partial + pb * kPartialWords;
        float* mine = buffer + warp * kTileOutWords;
        int const r = lane >> 2;
        mine[r * kMaxRows + t0] = acc[0];
        mine[r * kMaxRows + t0 + 1] = acc[1];
        mine[(r + 8) * kMaxRows + t0] = acc[2];
        mine[(r + 8) * kMaxRows + t0 + 1] = acc[3];
        consumersSync();
        op.epilogue(item, sub, TileView{buffer}, tid);
        pb ^= 1;
    };
    for (int item = blockIdx.x; item < items; item += gridDim.x)
    {
        op.itemBegin(item, tid);
        if (Op::kSyncItemBegin)
        {
            consumersSync();
        }
        int const subs = op.numSubs(item);
        acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
        for (int sub = 0; sub < subs; ++sub)
        {
            Tile const d = op.tile(item, sub);
            int const tok = min(lane & 7, m - 1);
            int const x0 = min(t0, m - 1);
            int const x1 = min(t0 + 1, m - 1);
            int const rowB = d.rowIdx != nullptr ? d.rowIdx[tok] : tok;
            int const row0 = d.rowIdx != nullptr ? d.rowIdx[x0] : x0;
            int const row1 = d.rowIdx != nullptr ? d.rowIdx[x1] : x1;
            int const cb = chunkBytes<Op>(d.rowBytes);
            int const chunks = d.rowBytes / cb;
            int const steps = cb / 32 / kConsumerWarps; // 32-byte k-steps per warp and chunk
            int const ks0 = warp * steps;
            if (!Op::kAccumulateSubs)
            {
                acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
            }
            for (int c = 0; c < chunks; ++c, ++i)
            {
                // FP8: this warp's block scales of the chunk: the two inputs it owns (resident) and the weights' rows
                // 0-7 / 8-15 (staged with the chunk).
                constexpr int kScaleBlocks
                    = Op::kChunkMax / 32 / kConsumerWarps / 4 > 0 ? Op::kChunkMax / 32 / kConsumerWarps / 4 : 1;
                float wTop[kScaleBlocks];
                float wBottom[kScaleBlocks];
                float xs0[kScaleBlocks];
                float xs1[kScaleBlocks];
                if (d.kind != kBf16)
                {
#pragma unroll
                    for (int b = 0; b < kScaleBlocks; ++b)
                    {
                        if (b * 4 < steps)
                        {
                            int const kb = (c * cb + (ks0 + b * 4) * 32) / kBlock;
                            xs0[b] = d.xs[row0 * d.xsStride + kb];
                            xs1[b] = d.xs[row1 * d.xsStride + kb];
                        }
                    }
                }
                int const s = i % ns;
                barWait(&full[s], (i / ns) & 1);
                uint8_t* stage = ring + static_cast<size_t>(s) * kStage;
                if (d.kind != kBf16)
                {
                    float const* staged = reinterpret_cast<float const*>(stage + kTileRows * kRingRow);
#pragma unroll
                    for (int b = 0; b < kScaleBlocks; ++b)
                    {
                        if (b * 4 < steps)
                        {
                            int const kb = (ks0 + b * 4) * 32 / kBlock; // within the chunk
                            wTop[b] = staged[kb];
                            wBottom[b] = staged[stageScaleWords<Op>() + kb];
                        }
                    }
                }
                uint32_t const a0 = smemAddr(stage) + (lane & 15) * kRingRow + (lane >> 4) * 16 + ks0 * 32;
                uint32_t const b0 = d.in + rowB * d.inStride + ((lane >> 3) & 1) * 16 + c * cb + ks0 * 32;
                if (d.kind == kBf16)
                {
#pragma unroll 4
                    for (int q = 0; q < steps; ++q)
                    {
                        uint32_t a[4];
                        uint32_t b[2];
                        ldmatrixX4(a, a0 + q * 32);
                        ldmatrixX2(b, b0 + q * 32);
                        mmaBf16(acc, a, b);
                    }
                }
                else
                {
#pragma unroll
                    for (int b = 0; b < kScaleBlocks; ++b)
                    {
                        if (b * 4 < steps)
                        {
                            float blk[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                            for (int u = 0; u < 4; ++u)
                            {
                                uint32_t a[4];
                                uint32_t bm[2];
                                ldmatrixX4(a, a0 + (b * 4 + u) * 32);
                                ldmatrixX2(bm, b0 + (b * 4 + u) * 32);
                                mmaFp8(blk, a, bm);
                            }
                            acc[0] += blk[0] * wTop[b] * xs0[b];
                            acc[1] += blk[1] * wTop[b] * xs1[b];
                            acc[2] += blk[2] * wBottom[b] * xs0[b];
                            acc[3] += blk[3] * wBottom[b] * xs1[b];
                        }
                    }
                }
                __syncwarp();
                if (lane == 0)
                {
                    barArrive(&empty[s]);
                }
            }
            if (!Op::kAccumulateSubs)
            {
                finishTile(item, sub);
            }
        }
        if (Op::kAccumulateSubs)
        {
            finishTile(item, subs);
        }
        if (Op::kSyncItemEnd)
        {
            consumersSync();
            op.itemEnd(item, tid);
            consumersSync();
        }
    }
    op.finish(tid);
#endif // __CUDA_ARCH__ >= 900
}

// ---------------------------------------------------------------------------------------------------------------------
// The ops. tileOut[r * kMaxRows + t] is output row (weight row) r of the tile for token t; the epilogue threads with
// tid < kTileOutWords own one each (r = tid / kMaxRows, t = tid % kMaxRows).

//! Shared defaults.
struct OpBase
{
    static constexpr bool kStaticTiles = true;
    //! The subs of an item accumulate into one output tile (one epilogue per item, sub = numSubs).
    static constexpr bool kAccumulateSubs = false;
    //! prologueLate runs on the consumers after the producer has been released.
    static constexpr bool kLatePrologue = false;
    static constexpr bool kEvictFirst = false;
    //! Barriers around itemBegin / itemEnd (an op that defines one sets its flag; itemEnd runs only with it).
    static constexpr bool kSyncItemBegin = false;
    static constexpr bool kSyncItemEnd = false;

    __device__ int numSubs(int) const
    {
        return 1;
    }

    __device__ void prologueLate(int) {}

    __device__ void itemBegin(int, int) {}

    __device__ void itemEnd(int, int) {}

    __device__ void finish(int) {}
};

__device__ __forceinline__ Tile bf16Tile(void const* w, int64_t rowElems, int rows, uint32_t in, int inStride)
{
    Tile d{};
    d.w = static_cast<uint8_t const*>(w);
    d.wStride = rowElems * 2;
    d.rows = rows;
    d.rowBytes = static_cast<int>(rowElems * 2);
    d.kind = kBf16;
    d.in = in;
    d.inStride = inStride;
    return d;
}

//! Stores tile outputs as bf16 into out[t * ld + n0 + r].
__device__ __forceinline__ void storeTile(
    TileView const& tileOut, uint16_t* out, int64_t ld, int n0, int nMax, int m, int tid)
{
    if (tid < kTileOutWords)
    {
        int const r = tid / kMaxRows;
        int const t = tid % kMaxRows;
        if (t < m && n0 + r < nMax)
        {
            out[t * ld + n0 + r] = static_cast<uint16_t>(__bfloat16_as_ushort(__float2bfloat16_rn(tileOut[tid])));
        }
    }
}

//! Rounds a tile's outputs to bf16 into sPush ([kMaxRows][16]) for arPush.
__device__ __forceinline__ void stagePush(TileView const& tileOut, uint16_t* sPush, int tid)
{
    if (tid < kTileOutWords)
    {
        int const r = tid / kMaxRows;
        int const t = tid % kMaxRows;
        sPush[t * kTileRows + r] = __bfloat16_as_ushort(__float2bfloat16_rn(tileOut[tid]));
    }
}

struct PreOp : OpBase
{
    using Params = PreParams;
    static constexpr bool kSyncItemEnd = true;
    static constexpr int kChunkMax = 2048;
    static constexpr int kInBytes = kHidden; // kHidden / 2 bf16 columns
    static constexpr int kInStride = kInBytes + 16;

    __host__ __device__ static size_t smemBytes(Params const& p)
    {
        return align128(static_cast<size_t>(p.m) * kInStride) + kConsumerWarps * kMaxRows * sizeof(float)
            + kMaxRows * kTileRows * sizeof(uint16_t);
    }

    Params const& p;
    uint8_t* sIn;
    float* sRed;
    uint16_t* sPush;

    __device__ PreOp(Params const& params, uint8_t* smem)
        : p(params)
        , sIn(smem)
        , sRed(reinterpret_cast<float*>(smem + align128(static_cast<size_t>(params.m) * kInStride)))
        , sPush(reinterpret_cast<uint16_t*>(sRed + kConsumerWarps * kMaxRows))
    {
    }

    __device__ int numItems() const
    {
        return kHidden / kTileRows;
    }

    __device__ Tile tile(int item, int) const
    {
        return bf16Tile(static_cast<uint16_t const*>(p.weight) + static_cast<size_t>(item) * kTileRows * (kHidden / 2),
            kHidden / 2, kTileRows, smemAddr(sIn), kInStride);
    }

    // RMSNorm over the full row of the embedding (or hidden), keeping this rank's half of the columns.
    __device__ void prologue(int tid)
    {

        int const warp = tid >> 5;
        int const lane = tid & 31;
        int const half = p.colStart / (kHidden / 2);
        for (int t = 0; t < p.m; ++t)
        {
            uint16_t const* row = nullptr;
            if (p.useEmbed)
            {
                int const id = p.tokens[t * p.tokensStride];
                row = (id >= 0 && id < p.vocab)
                    ? static_cast<uint16_t const*>(p.embed) + static_cast<size_t>(id) * kHidden
                    : nullptr;
            }
            else
            {
                row = static_cast<uint16_t const*>(p.hidden) + t * p.hiddenStride;
            }
            float x[2][8];
            float ss = 0.f;
#pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                if (row != nullptr)
                {
                    load8(row + h * (kHidden / 2) + tid * 8, x[h]);
                }
                else
                {
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                    {
                        x[h][j] = 0.f;
                    }
                }
#pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    ss += x[h][j] * x[h][j];
                }
            }
            ss = warpSum(ss);
            if (lane == 0)
            {
                sRed[warp * kMaxRows + t] = ss;
            }
            store8(reinterpret_cast<uint16_t*>(sIn + t * kInStride) + tid * 8, x[half]);
        }
        consumersSync();
        for (int t = 0; t < p.m; ++t)
        {
            float s = 0.f;
#pragma unroll
            for (int w = 0; w < kConsumerWarps; ++w)
            {
                s += sRed[w * kMaxRows + t];
            }
            float const rstd = rsqrtf(s / kHidden + p.eps);
            uint16_t* dst = reinterpret_cast<uint16_t*>(sIn + t * kInStride) + tid * 8;
            float x[8];
            float g[8];
            load8(dst, x);
            load8(static_cast<uint16_t const*>(p.gamma) + p.colStart + tid * 8, g);
#pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                x[j] = x[j] * rstd * g[j];
            }
            store8(dst, x);
        }
    }

    __device__ void epilogue(int, int, TileView const& tileOut, int tid)
    {
        stagePush(tileOut, sPush, tid);
    }

    __device__ void itemEnd(int item, int tid)
    {
        arPush(p.ar, 0, p.m, item * kTileRows, sPush, tid);
    }
};

struct QkvAOp : OpBase
{
    using Params = QkvAParams;
    static constexpr int kChunkMax = 2048;
    static constexpr int kInStride = kHidden * 2 + 16;

    __host__ __device__ static size_t smemBytes(Params const& p)
    {
        return align128(static_cast<size_t>(p.m) * kInStride) + kConsumerWarps * kMaxRows * sizeof(float);
    }

    Params const& p;
    uint8_t* sIn;
    float* sRed;

    __device__ QkvAOp(Params const& params, uint8_t* smem)
        : p(params)
        , sIn(smem)
        , sRed(reinterpret_cast<float*>(smem + align128(static_cast<size_t>(params.m) * kInStride)))
    {
    }

    __device__ int numItems() const
    {
        return (p.qkvRows + p.idxRows) / kTileRows;
    }

    __device__ Tile tile(int item, int) const
    {
        int const n0 = item * kTileRows;
        void const* w = n0 < p.qkvRows
            ? static_cast<void const*>(static_cast<uint16_t const*>(p.qkvWeight) + static_cast<size_t>(n0) * kHidden)
            : static_cast<void const*>(
                static_cast<uint16_t const*>(p.idxWeight) + static_cast<size_t>(n0 - p.qkvRows) * kHidden);
        return bf16Tile(w, kHidden, kTileRows, smemAddr(sIn), kInStride);
    }

    __device__ void prologue(int tid)
    {
        arReduceNorm(p.ar, 0, p.m, nullptr, static_cast<uint16_t*>(p.residualOut),
            static_cast<uint16_t const*>(p.gamma), p.eps, sIn, kInStride, nullptr, 0, sRed, tid);
    }

    __device__ void epilogue(int item, int, TileView const& tileOut, int tid)
    {
        storeTile(tileOut, static_cast<uint16_t*>(p.out), p.qkvRows + p.idxRows, item * kTileRows,
            p.qkvRows + p.idxRows, p.m, tid);
    }
};

struct QBOp : OpBase
{
    using Params = QBParams;
    static constexpr int kChunkMax = 1024;

    __host__ __device__ static int inStride(Params const& p)
    {
        return p.qaRank * 2 + 16;
    }

    __host__ __device__ static size_t smemBytes(Params const& p)
    {
        return align128(static_cast<size_t>(p.m) * inStride(p));
    }

    Params const& p;
    uint8_t* sIn;

    __device__ QBOp(Params const& params, uint8_t* smem)
        : p(params)
        , sIn(smem)
    {
    }

    // q_b tiles, then (step 0) the indexer's wq_b tiles over the same normalized q_a rows.
    __device__ int numItems() const
    {
        return (p.qbRows + p.idxQRows) / kTileRows;
    }

    __device__ Tile tile(int item, int) const
    {
        int const qbItems = p.qbRows / kTileRows;
        uint16_t const* w = item < qbItems
            ? static_cast<uint16_t const*>(p.weight) + static_cast<size_t>(item) * kTileRows * p.qaRank
            : static_cast<uint16_t const*>(p.idxQWeight) + static_cast<size_t>(item - qbItems) * kTileRows * p.qaRank;
        return bf16Tile(w, p.qaRank, kTileRows, smemAddr(sIn), inStride(p));
    }

    __device__ void prologue(int tid)
    {

        int const warp = tid >> 5;
        int const lane = tid & 31;
        if (warp < p.m)
        {
            uint16_t const* row = static_cast<uint16_t const*>(p.qkv) + static_cast<size_t>(warp) * p.qkvStride;
            rmsNormRowWarp(row, p.qaRank, static_cast<uint16_t const*>(p.qaGamma), p.eps,
                reinterpret_cast<uint16_t*>(sIn + warp * inStride(p)), lane);
            if (blockIdx.x == 0)
            {
                rmsNormRowWarp(row + p.qaRank, p.kvRank, static_cast<uint16_t const*>(p.kvGamma), p.eps,
                    static_cast<uint16_t*>(p.latentOut) + static_cast<size_t>(warp) * p.kvRank, lane);
                // The indexer key's LayerNorm (biased variance) and the raw gate.
                uint16_t const* key = row + p.qaRank + p.kvRank;
                uint16_t* packed = static_cast<uint16_t*>(p.packedOut) + static_cast<size_t>(warp) * 2 * p.idxDim;
                float sum = 0.f;
                for (int c = lane; c < p.idxDim; c += 32)
                {
                    sum += bf(key[c]);
                }
                float const mean = warpSum(sum) / p.idxDim;
                float var = 0.f;
                for (int c = lane; c < p.idxDim; c += 32)
                {
                    float const d = bf(key[c]) - mean;
                    var += d * d;
                }
                float const rstd = rsqrtf(warpSum(var) / p.idxDim + p.lnEps);
                for (int c = lane; c < p.idxDim; c += 32)
                {
                    float const y = (bf(key[c]) - mean) * rstd * bf(static_cast<uint16_t const*>(p.lnWeight)[c])
                        + bf(static_cast<uint16_t const*>(p.lnBias)[c]);
                    packed[c] = toBf(y);
                    packed[p.idxDim + c] = key[p.idxDim + c];
                }
            }
        }
    }

    __device__ void epilogue(int item, int, TileView const& tileOut, int tid)
    {
        int const qbItems = p.qbRows / kTileRows;
        if (item < qbItems)
        {
            storeTile(tileOut, static_cast<uint16_t*>(p.queryOut), p.qbRows, item * kTileRows, p.qbRows, p.m, tid);
            return;
        }
        storeTile(
            tileOut, static_cast<uint16_t*>(p.idxQOut), p.idxQRows, (item - qbItems) * kTileRows, p.idxQRows, p.m, tid);
    }
};

struct AbsorbOp : OpBase
{
    using Params = AbsorbParams;
    static constexpr int kChunkMax = 512;

    __host__ __device__ static int inStride(Params const& p)
    {
        return p.heads * p.nope * 2 + 16;
    }

    __host__ __device__ static size_t smemBytes(Params const& p)
    {
        return align128(static_cast<size_t>(p.m) * inStride(p));
    }

    Params const& p;
    uint8_t* sIn;

    __device__ AbsorbOp(Params const& params, uint8_t* smem)
        : p(params)
        , sIn(smem)
    {
    }

    __device__ int numItems() const
    {
        return p.heads * p.kvRank / kTileRows;
    }

    __device__ Tile tile(int item, int) const
    {
        int const n0 = item * kTileRows;
        int const head = n0 / p.kvRank;
        return bf16Tile(static_cast<uint16_t const*>(p.wkT) + static_cast<size_t>(n0) * p.nope, p.nope, kTileRows,
            smemAddr(sIn) + head * p.nope * 2, inStride(p));
    }

    __device__ void prologue(int tid)
    {
        loadRows(sIn, inStride(p), static_cast<uint8_t const*>(p.query), p.heads * p.nope * 2, p.m,
            p.heads * p.nope * 2, tid);
    }

    __device__ void epilogue(int item, int, TileView const& tileOut, int tid)
    {
        int const n = p.heads * p.kvRank;
        storeTile(tileOut, static_cast<uint16_t*>(p.out), n, item * kTileRows, n, p.m, tid);
    }
};

struct VUpOp : OpBase
{
    using Params = VUpParams;
    static constexpr int kChunkMax = 1024;

    __host__ __device__ static int inStride(Params const& p)
    {
        return p.heads * p.kvRank * 2 + 16;
    }

    __host__ __device__ static size_t smemBytes(Params const& p)
    {
        return align128(static_cast<size_t>(p.m) * inStride(p));
    }

    Params const& p;
    uint8_t* sIn;

    __device__ VUpOp(Params const& params, uint8_t* smem)
        : p(params)
        , sIn(smem)
    {
    }

    __device__ int numItems() const
    {
        return p.heads * p.vDim / kTileRows;
    }

    __device__ Tile tile(int item, int) const
    {
        int const n0 = item * kTileRows;
        int const head = n0 / p.vDim;
        int const j0 = n0 % p.vDim;
        return bf16Tile(static_cast<uint16_t const*>(p.kvB)
                + static_cast<size_t>(head * (p.nope + p.vDim) + p.nope + j0) * p.kvRank,
            p.kvRank, kTileRows, smemAddr(sIn) + head * p.kvRank * 2, inStride(p));
    }

    __device__ void prologue(int tid)
    {
        loadRows(sIn, inStride(p), static_cast<uint8_t const*>(p.latent), p.latentStride * 2, p.m,
            p.heads * p.kvRank * 2, tid);
    }

    __device__ void epilogue(int item, int, TileView const& tileOut, int tid)
    {
        int const n = p.heads * p.vDim;
        storeTile(tileOut, static_cast<uint16_t*>(p.out), n, item * kTileRows, n, p.m, tid);
    }
};

struct OProjOp : OpBase
{
    using Params = OProjParams;
    static constexpr bool kSyncItemEnd = true;
    static constexpr int kChunkMax = 2048;

    __host__ __device__ static int inStride(Params const& p)
    {
        return p.k * 2 + 16;
    }

    __host__ __device__ static size_t smemBytes(Params const& p)
    {
        return align128(static_cast<size_t>(p.m) * inStride(p)) + kMaxRows * kTileRows * sizeof(uint16_t);
    }

    Params const& p;
    uint8_t* sIn;
    uint16_t* sPush;

    __device__ OProjOp(Params const& params, uint8_t* smem)
        : p(params)
        , sIn(smem)
        , sPush(reinterpret_cast<uint16_t*>(smem + align128(static_cast<size_t>(params.m) * inStride(params))))
    {
    }

    __device__ int numItems() const
    {
        return kHidden / kTileRows;
    }

    __device__ Tile tile(int item, int) const
    {
        return bf16Tile(static_cast<uint16_t const*>(p.weight) + static_cast<size_t>(item) * kTileRows * p.k, p.k,
            kTileRows, smemAddr(sIn), inStride(p));
    }

    __device__ void prologue(int tid)
    {

        loadRows(sIn, inStride(p), static_cast<uint8_t const*>(p.input), p.k * 2, p.m, p.k * 2, tid);
    }

    __device__ void epilogue(int, int, TileView const& tileOut, int tid)
    {
        stagePush(tileOut, sPush, tid);
    }

    __device__ void itemEnd(int item, int tid)
    {
        arPush(p.ar, 1, p.m, item * kTileRows, sPush, tid);
    }
};

struct RouterOp : OpBase
{
    using Params = RouterParams;
    static constexpr bool kSyncItemEnd = true;
    static constexpr int kChunkMax = 2048;
    static constexpr int kInStride = kHidden * 2 + 16;

    __host__ __device__ static size_t smemBytes(Params const& p)
    {
        return align128(static_cast<size_t>(p.m) * kInStride) + align128(kConsumerWarps * kMaxRows * sizeof(float))
            + align128(sizeof(RouteTable)) + align128(kConsumers * kMaxRows * sizeof(float)) + 16;
    }

    Params const& p;
    uint8_t* sIn;
    float* sRed;
    RouteTable* sTable;
    float* sByRow;
    int* sLast;

    __device__ RouterOp(Params const& params, uint8_t* smem)
        : p(params)
        , sIn(smem)
        , sRed(reinterpret_cast<float*>(smem + align128(static_cast<size_t>(params.m) * kInStride)))
        , sTable(reinterpret_cast<RouteTable*>(
              reinterpret_cast<uint8_t*>(sRed) + align128(kConsumerWarps * kMaxRows * sizeof(float))))
        , sByRow(reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(sTable) + align128(sizeof(RouteTable))))
        , sLast(reinterpret_cast<int*>(sByRow + kConsumers * kMaxRows))
    {
    }

    __device__ int routerItems() const
    {
        return (p.experts + kTileRows - 1) / kTileRows;
    }

    __device__ int numItems() const
    {
        return routerItems() + p.sharedInter / 8;
    }

    // Router tiles, then shared-expert tiles of 8 gate rows (0-7) and the matching 8 up rows (8-15).
    __device__ Tile tile(int item, int) const
    {

        uint16_t const* gate = static_cast<uint16_t const*>(p.gateWeight);
        uint16_t const* gateUp = static_cast<uint16_t const*>(p.sharedGateUp);
        if (item < routerItems())
        {
            int const n0 = item * kTileRows;
            return bf16Tile(gate + static_cast<size_t>(n0) * kHidden, kHidden, min(kTileRows, p.experts - n0),
                smemAddr(sIn), kInStride);
        }
        int const j0 = (item - routerItems()) * 8;
        Tile d = bf16Tile(gateUp + static_cast<size_t>(j0) * kHidden, kHidden, kTileRows, smemAddr(sIn), kInStride);
        d.w2 = reinterpret_cast<uint8_t const*>(gateUp + static_cast<size_t>(p.sharedInter + j0) * kHidden);
        return d;
    }

    __device__ void prologue(int tid)
    {
        arReduceNorm(p.ar, 1, p.m, static_cast<uint16_t const*>(p.residualIn), static_cast<uint16_t*>(p.residualOut),
            static_cast<uint16_t const*>(p.gamma), p.eps, sIn, kInStride, static_cast<uint16_t*>(p.normOut), kHidden,
            sRed, tid, p.rowIdx);
    }

    __device__ void epilogue(int item, int, TileView const& tileOut, int tid)
    {

        if (item < routerItems())
        {
            if (tid < kTileOutWords)
            {
                int const r = tid / kMaxRows;
                int const t = tid % kMaxRows;
                int const n = item * kTileRows + r;
                if (t < p.m && n < p.experts)
                {
                    p.logitsOut[t * p.experts + n] = tileOut[tid];
                }
            }
            return;
        }
        if (tid < 8 * kMaxRows)
        {
            int const r = tid / kMaxRows;
            int const t = tid % kMaxRows;
            if (t < p.m)
            {
                // The GEMM's bf16 outputs, then the activation in fp32.
                float const g = roundBf(tileOut[r * kMaxRows + t]);
                float const u = roundBf(tileOut[(r + 8) * kMaxRows + t]);
                int const j = (item - routerItems()) * 8 + r;
                static_cast<uint16_t*>(p.sharedOut)[t * p.sharedInter + j] = toBf(swiglu(g, u, p.swigluLimit));
            }
        }
    }

    // The CTA that finishes the last logits tile computes the routing (all the logits are in global memory then) and
    // writes it to the route table, while the other CTAs still run the shared expert's tiles.
    __device__ void itemEnd(int item, int tid)
    {
        if (item >= routerItems())
        {
            return;
        }
        int* counter = p.route + routeCounterWord();
        if (tid == 0)
        {
            __threadfence();
            int const last = atomicAdd(counter, 1) == routerItems() - 1;
            if (last)
            {
                __threadfence();
                *counter = 0;
            }
            *sLast = last;
        }
        consumersSync();
        if (!*sLast)
        {
            return;
        }
        computeRouting(p.routing, p.m, &sTable->routing, sByRow, tid);
        int const zeroRow = min(p.m * p.routing.topK, RoutePairs::kMax);
        computePairs(&sTable->routing, &sTable->pairs, p.m, zeroRow, tid);
        copy16(p.route, sTable, sizeof(RouteTable), tid, false);
    }
};

struct Fc1Op : OpBase
{
    using Params = Fc1Params;
    static constexpr bool kLatePrologue = true;
    static constexpr bool kStaticTiles = false;
    static constexpr int kChunkMax = 2048;
    static constexpr int kInStride = kHidden + 16; // e4m3
    static constexpr int kBlocks = kHidden / kBlock;
    static constexpr int kInterBlocks = kMoeInter / kBlock;
    //! An expert item: 8 intermediate columns (8 up rows + the matching 8 gate rows) of one slot.
    static constexpr int kItemCols = 8;
    static constexpr int kItemsPerSlot = kMoeInter / kItemCols;

    __host__ __device__ static int sharedStride(Params const& p)
    {
        return p.sharedInter * 2 + 16;
    }

    __host__ __device__ static size_t smemBytes(Params const& p)
    {
        return align128(static_cast<size_t>(p.m) * kInStride) + align128(static_cast<size_t>(p.m) * sharedStride(p))
            + kMaxRows * kBlocks * sizeof(float) + sizeof(RoutingSmem);
    }

    Params const& p;
    uint8_t* sIn;
    uint8_t* sShared;
    float* sXs;
    RoutingSmem* sRoute;

    __device__ Fc1Op(Params const& params, uint8_t* smem)
        : p(params)
        , sIn(smem)
        , sShared(smem + align128(static_cast<size_t>(params.m) * kInStride))
        , sXs(reinterpret_cast<float*>(sShared + align128(static_cast<size_t>(params.m) * sharedStride(params))))
        , sRoute(reinterpret_cast<RoutingSmem*>(sXs + kMaxRows * kBlocks))
    {
    }

    __device__ int expertItems() const
    {
        return sRoute->count * kItemsPerSlot;
    }

    __device__ int numItems() const
    {
        return expertItems() + kHidden / kTileRows;
    }

    // Expert items first (slot-major), then the shared expert's down projection.
    __device__ Tile tile(int item, int) const
    {
        if (item < expertItems())
        {
            int const slot = item / kItemsPerSlot;
            int const up = (item % kItemsPerSlot) * kItemCols;
            int const e = sRoute->slotExpert[slot] - p.routing.localStart;
            uint8_t const* base = static_cast<uint8_t const*>(p.w13) + static_cast<size_t>(e) * 2 * kMoeInter * kHidden;
            float const* scale = p.w13Scale + static_cast<size_t>(e) * (2 * kMoeInter / kBlock) * kBlocks;
            Tile d{};
            d.w = base + static_cast<size_t>(up) * kHidden;
            d.w2 = base + static_cast<size_t>(kMoeInter + up) * kHidden;
            d.wStride = kHidden;
            d.rows = kTileRows;
            d.rowBytes = kHidden;
            d.kind = kFp8;
            d.in = smemAddr(sIn);
            d.inStride = kInStride;
            d.ws = scale + (up / kBlock) * kBlocks;
            d.ws2 = scale + ((kMoeInter + up) / kBlock) * kBlocks;
            d.xs = sXs;
            d.xsStride = kBlocks;
            return d;
        }
        int const n0 = (item - expertItems()) * kTileRows;
        return bf16Tile(static_cast<uint16_t const*>(p.sharedDown) + static_cast<size_t>(n0) * p.sharedInter,
            p.sharedInter, kTileRows, smemAddr(sShared), sharedStride(p));
    }

    // The router's routing (all the producer needs to stream the experts' weights).
    __device__ void prologue(int tid)
    {
        copy16(sRoute, p.route, sizeof(RoutingSmem), tid, true);
    }

    // The input quantized to e4m3 per token and 128 columns (scale = amax / 448); the shared expert's intermediate.
    __device__ void prologueLate(int tid)
    {
        int const warp = tid >> 5;
        int const lane = tid & 31;
        for (int b = warp; b < p.m * kBlocks; b += kConsumerWarps)
        {
            int const t = b / kBlocks;
            int const kb = b % kBlocks;
            uint16_t const* src
                = static_cast<uint16_t const*>(p.input) + static_cast<size_t>(t) * kHidden + kb * kBlock + lane * 4;
            uint2 const raw = *reinterpret_cast<uint2 const*>(src);
            float x[4] = {bf(static_cast<uint16_t>(raw.x & 0xffffu)), bf(static_cast<uint16_t>(raw.x >> 16)),
                bf(static_cast<uint16_t>(raw.y & 0xffffu)), bf(static_cast<uint16_t>(raw.y >> 16))};
            float amax = fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fmaxf(fabsf(x[2]), fabsf(x[3])));
            amax = warpMax(amax);
            float const scale = amax > 0.f ? amax / kFp8Max : 1.f;
            float const inv = 1.f / scale;
            __nv_fp8x4_e4m3 const q(make_float4(x[0] * inv, x[1] * inv, x[2] * inv, x[3] * inv));
            *reinterpret_cast<__nv_fp8x4_e4m3*>(sIn + t * kInStride + kb * kBlock + lane * 4) = q;
            if (lane == 0)
            {
                sXs[t * kBlocks + kb] = scale;
            }
        }
        loadRows(sShared, sharedStride(p), static_cast<uint8_t const*>(p.sharedIn), p.sharedInter * 2, p.m,
            p.sharedInter * 2, tid);
    }

    // Expert item: the clamped SwiGLU of its 8 columns, fp32, per token (fc2 requantizes it). Shared down: bf16.
    __device__ void epilogue(int item, int, TileView const& tileOut, int tid)
    {
        if (item < expertItems())
        {
            if (tid < kItemCols * kMaxRows)
            {
                int const r = tid / kMaxRows;
                int const t = tid % kMaxRows;
                if (t < p.m)
                {
                    int const slot = item / kItemsPerSlot;
                    int const col = (item % kItemsPerSlot) * kItemCols + r;
                    p.inter[(static_cast<size_t>(slot) * kMaxRows + t) * kMoeInter + col]
                        = swiglu(tileOut[(r + 8) * kMaxRows + t], tileOut[r * kMaxRows + t], p.swigluLimit);
                }
            }
            return;
        }
        storeTile(tileOut, static_cast<uint16_t*>(p.sharedDownOut), kHidden, (item - expertItems()) * kTileRows,
            kHidden, p.m, tid);
    }
};

struct Fc2Op : OpBase
{
    using Params = Fc2Params;
    static constexpr bool kSyncItemEnd = true;
    static constexpr bool kStaticTiles = false;
    static constexpr bool kAccumulateSubs = true;
    static constexpr bool kLatePrologue = true;
    static constexpr int kChunkMax = 1024;
    static constexpr int kInterBlocks = kMoeInter / kBlock;
    static constexpr int kActStride = kMoeInter + 16; // e4m3

    __host__ __device__ static int maxPairs(Params const& p)
    {
        int const n = p.m * p.routing.topK;
        return n < RoutePairs::kMax ? n : RoutePairs::kMax;
    }

    __host__ __device__ static size_t smemBytes(Params const& p)
    {
        return align128(sizeof(RouteTable)) + align128((RoutePairs::kMax + 1) * kInterBlocks * sizeof(float))
            + align128(static_cast<size_t>(kMaxRows) * kTileRows * sizeof(uint16_t))
            + static_cast<size_t>(maxPairs(p) + 1) * kActStride;
    }

    Params const& p;
    RouteTable* sTable;
    RoutingSmem* sRoute;
    RoutePairs* sPairs;
    float* sScale; // [pair (+ the zero row)][kInterBlocks]
    uint16_t* sPush;
    uint8_t* sAct;
    int const zeroRow;

    __device__ Fc2Op(Params const& params, uint8_t* smem)
        : p(params)
        , sTable(reinterpret_cast<RouteTable*>(smem))
        , sRoute(&sTable->routing)
        , sPairs(&sTable->pairs)
        , sScale(reinterpret_cast<float*>(smem + align128(sizeof(RouteTable))))
        , sPush(reinterpret_cast<uint16_t*>(
              reinterpret_cast<uint8_t*>(sScale) + align128((RoutePairs::kMax + 1) * kInterBlocks * sizeof(float))))
        , sAct(reinterpret_cast<uint8_t*>(sPush)
              + align128(static_cast<size_t>(kMaxRows) * kTileRows * sizeof(uint16_t)))
        , zeroRow(maxPairs(params))
    {
    }

    __device__ int numItems() const
    {
        return kHidden / kTileRows;
    }

    __device__ int numSubs(int) const
    {
        return sRoute->count;
    }

    // Sub = slot: its expert's 16 output rows. Token column t reads its pair's row (the intermediate already scaled
    // by the routing weight) or the zero row, so every slot accumulates into the same tile.
    __device__ Tile tile(int item, int slot) const
    {
        int const n0 = item * kTileRows;
        int const e = sRoute->slotExpert[slot] - p.routing.localStart;
        Tile d{};
        d.w = static_cast<uint8_t const*>(p.w2) + (static_cast<size_t>(e) * kHidden + n0) * kMoeInter;
        d.wStride = kMoeInter;
        d.rows = kTileRows;
        d.rowBytes = kMoeInter;
        d.kind = kFp8;
        d.in = smemAddr(sAct);
        d.inStride = kActStride;
        d.rowIdx = sPairs->rowIdx + slot * kMaxRows;
        d.ws = p.w2Scale + (static_cast<size_t>(e) * (kHidden / kBlock) + n0 / kBlock) * kInterBlocks;
        d.ws2 = d.ws;
        d.xs = sScale;
        d.xsStride = kInterBlocks;
        return d;
    }

    // The router's routing and each slot's tokens.
    __device__ void prologue(int tid)
    {
        copy16(sTable, p.route, sizeof(RouteTable), tid, true);
    }

    // Each pair's intermediate x its routing weight, requantized to e4m3 per 128 columns (scale = amax / 448), one
    // thread per (pair, block) with its 128 values requested at once; and the zero row.
    __device__ void prologueLate(int tid)
    {
        int const units = sPairs->total * kInterBlocks;
        for (int u = tid; u < units; u += kConsumers)
        {
            int const q = u / kInterBlocks;
            int const blk = u % kInterBlocks;
            float const w = sPairs->weight[q];
            float4 const* src = reinterpret_cast<float4 const*>(p.inter
                + (static_cast<size_t>(sPairs->slot[q]) * kMaxRows + sPairs->token[q]) * kMoeInter + blk * kBlock);
            float4 v[kBlock / 4];
#pragma unroll
            for (int k = 0; k < kBlock / 4; ++k)
            {
                v[k] = src[k];
            }
            float amax = 0.f;
#pragma unroll
            for (int k = 0; k < kBlock / 4; ++k)
            {
                v[k] = make_float4(v[k].x * w, v[k].y * w, v[k].z * w, v[k].w * w);
                amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v[k].x), fabsf(v[k].y)), fmaxf(fabsf(v[k].z), fabsf(v[k].w))));
            }
            float const scale = amax > 0.f ? amax / kFp8Max : 1.f;
            float const inv = 1.f / scale;
            uint8_t* dst = sAct + static_cast<size_t>(q) * kActStride + blk * kBlock;
#pragma unroll
            for (int k = 0; k < kBlock / 4; ++k)
            {
                __nv_fp8x4_e4m3 const e4(make_float4(v[k].x * inv, v[k].y * inv, v[k].z * inv, v[k].w * inv));
                *reinterpret_cast<__nv_fp8x4_e4m3*>(dst + k * 4) = e4;
            }
            sScale[q * kInterBlocks + blk] = scale;
        }
        for (int k = tid; k < kMoeInter / 16; k += kConsumers)
        {
            reinterpret_cast<uint4*>(sAct + static_cast<size_t>(zeroRow) * kActStride)[k] = make_uint4(0, 0, 0, 0);
        }
        if (tid < kInterBlocks)
        {
            sScale[zeroRow * kInterBlocks + tid] = 0.f;
        }
    }

    // The routed sum (bf16) + shared (bf16) -> this rank's MoE partial.
    __device__ void epilogue(int item, int, TileView const& tileOut, int tid)
    {
        if (tid < kTileOutWords)
        {
            int const r = tid / kMaxRows;
            int const t = tid % kMaxRows;
            if (t < p.m)
            {
                int const n0 = item * kTileRows;
                float const shared
                    = bf(static_cast<uint16_t const*>(p.shared)[static_cast<size_t>(t) * kHidden + n0 + r]);
                sPush[t * kTileRows + r] = toBf(roundBf(tileOut[tid]) + shared);
            }
        }
    }

    // -> all-reduce buffer 2.
    __device__ void itemEnd(int item, int tid)
    {
        arPush(p.ar, 2, p.m, item * kTileRows, sPush, tid);
    }
};

struct HeadOp : OpBase
{
    using Params = HeadParams;
    static constexpr bool kEvictFirst = true;
    static constexpr int kChunkMax = 2048;
    static constexpr int kInStride = kHidden * 2 + 16;

    __host__ __device__ static size_t smemBytes(Params const& p)
    {
        return align128(static_cast<size_t>(p.m) * kInStride) + kConsumerWarps * kMaxRows * sizeof(float)
            + kMaxRows * sizeof(unsigned long long) + sizeof(int);
    }

    Params const& p;
    uint8_t* sIn;
    float* sRed;
    unsigned long long* sBest;
    int* sLast;
    unsigned long long best;

    __device__ HeadOp(Params const& params, uint8_t* smem)
        : p(params)
        , sIn(smem)
        , sRed(reinterpret_cast<float*>(smem + align128(static_cast<size_t>(params.m) * kInStride)))
        , sBest(reinterpret_cast<unsigned long long*>(sRed + kConsumerWarps * kMaxRows))
        , sLast(reinterpret_cast<int*>(sBest + kMaxRows))
        , best(0ull)
    {
    }

    __device__ int numItems() const
    {
        return (p.n + kTileRows - 1) / kTileRows;
    }

    __device__ Tile tile(int item, int) const
    {
        int const n0 = item * kTileRows;
        return bf16Tile(static_cast<uint16_t const*>(p.weight) + static_cast<size_t>(n0) * kHidden, kHidden,
            min(kTileRows, p.n - n0), smemAddr(sIn), kInStride);
    }

    __device__ void prologue(int tid)
    {

        if (tid < kMaxRows)
        {
            sBest[tid] = 0ull;
        }
        arReduceNorm(p.ar, 2, p.m, static_cast<uint16_t const*>(p.residualIn), nullptr,
            static_cast<uint16_t const*>(p.gamma), p.eps, sIn, kInStride, static_cast<uint16_t*>(p.hiddenOut),
            p.hiddenStride, sRed, tid);
    }

    __device__ void epilogue(int item, int, TileView const& tileOut, int tid)
    {

        if (tid < kTileOutWords)
        {
            int const r = tid / kMaxRows;
            int const t = tid % kMaxRows;
            int const n = item * kTileRows + r;
            if (t < p.m && n < p.n)
            {
                uint32_t const u = __float_as_uint(tileOut[tid]);
                unsigned long long const key
                    = (static_cast<unsigned long long>((u & 0x80000000u) ? ~u : (u | 0x80000000u)) << 32)
                    | static_cast<unsigned long long>(0xFFFFFFFFu - static_cast<uint32_t>(p.vocabOffset + n));
                best = key > best ? key : best;
            }
        }
    }

    // The CTA's best key per row -> the scratch; the last CTA reduces them and runs the key exchange (the protocol
    // of lm_head_argmax).
    __device__ void finish(int tid)
    {

        namespace lh = lm_head_argmax;
        int const lane = tid & 31;
        uint64_t* parts = p.scratch;
        unsigned int* counter = reinterpret_cast<unsigned int*>(p.scratch + lh::kMaxCtas * lh::kMaxRows);
        uint32_t* state = reinterpret_cast<uint32_t*>(p.scratch + lh::kMaxCtas * lh::kMaxRows + 1);
        if (tid < kTileOutWords && (tid % kMaxRows) < p.m)
        {
            atomicMax(&sBest[tid % kMaxRows], best);
        }
        consumersSync();
        if (tid == 0)
        {
            for (int t = 0; t < p.m; ++t)
            {
                parts[static_cast<size_t>(blockIdx.x) * lh::kMaxRows + t] = sBest[t];
            }
            unsigned int arrived;
            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;\n" : "=r"(arrived) : "l"(counter) : "memory");
            *sLast = arrived == gridDim.x - 1;
        }
        consumersSync();
        if (!*sLast)
        {
            return;
        }
        if (tid == 0)
        {
            *counter = 0u;
        }
        unsigned long long reduced[kMaxRows];
#pragma unroll
        for (int t = 0; t < kMaxRows; ++t)
        {
            reduced[t] = 0ull;
        }
        for (int b = tid; b < static_cast<int>(gridDim.x); b += kConsumers)
        {
#pragma unroll
            for (int t = 0; t < kMaxRows; ++t)
            {
                if (t < p.m)
                {
                    unsigned long long const x = __ldcg(
                        reinterpret_cast<unsigned long long const*>(parts) + static_cast<size_t>(b) * lh::kMaxRows + t);
                    reduced[t] = x > reduced[t] ? x : reduced[t];
                }
            }
        }
#pragma unroll
        for (int t = 0; t < kMaxRows; ++t)
        {
            if (t < p.m)
            {
#pragma unroll
                for (int o = 16; o > 0; o >>= 1)
                {
                    unsigned long long const y = __shfl_xor_sync(0xffffffffu, reduced[t], o);
                    reduced[t] = y > reduced[t] ? y : reduced[t];
                }
                if (lane == 0 && reduced[t])
                {
                    atomicMax(&sBest[t], reduced[t]);
                }
            }
        }
        consumersSync();
        unsigned long long const local = tid < p.m ? sBest[tid] : 0ull;
        if (p.keyRanks <= 1)
        {
            if (tid < p.m)
            {
                p.tokens[tid * p.tokensStride] = static_cast<int32_t>(0xFFFFFFFFu - static_cast<uint32_t>(local));
            }
            return;
        }
        uint32_t const epoch = state[0] + 1;
        int const previousRows = static_cast<int>(state[1]);
        int const cur = epoch % 3;
        int const old = (epoch + 2) % 3;
        size_t const epochSlots = static_cast<size_t>(lh::kMaxRanks) * lh::kMaxRows;
        if (tid < p.m)
        {
            for (int r = 0; r < p.keyRanks; ++r)
            {
                *reinterpret_cast<unsigned long long volatile*>(
                    p.keyPeers[r] + cur * epochSlots + static_cast<size_t>(p.keyRank) * lh::kMaxRows + tid)
                    = local;
            }
            unsigned long long global = 0ull;
            for (int r = 0; r < p.keyRanks; ++r)
            {
                unsigned long long x;
                do
                {
                    x = *reinterpret_cast<unsigned long long volatile*>(
                        p.keyPeers[p.keyRank] + cur * epochSlots + static_cast<size_t>(r) * lh::kMaxRows + tid);
                } while (x == 0ull);
                global = global > x ? global : x;
            }
            p.tokens[tid * p.tokensStride] = static_cast<int32_t>(0xFFFFFFFFu - static_cast<uint32_t>(global));
        }
        for (int i = tid; i < previousRows; i += kConsumers)
        {
            for (int r = 0; r < p.keyRanks; ++r)
            {
                p.keyPeers[p.keyRank][old * epochSlots + static_cast<size_t>(r) * lh::kMaxRows + i] = 0ull;
            }
        }
        consumersSync();
        if (tid == 0)
        {
            state[0] = epoch;
            state[1] = static_cast<uint32_t>(p.m);
        }
    }
};

template <class Op>
void launchOp(typename Op::Params const& p, int items, int ctas, cudaStream_t stream, bool enablePdl)
{
    TLLM_CHECK_WITH_INFO(p.m >= 1 && p.m <= kMaxRows, "glm5_mtp_step: rows must be in [1, %d]", kMaxRows);
    auto kernel = engineKernel<Op>;
    size_t const fixed = align128(Op::smemBytes(p)) + engineFixedBytes();
    size_t const stage = stageBytes<Op>();
    int const ns = static_cast<int>(std::min<size_t>(kMaxStages, (kSmemMax - fixed) / stage));
    TLLM_CHECK_WITH_INFO(ns >= 2, "glm5_mtp_step: shared memory too small for this shape");
    size_t const smem = fixed + static_cast<size_t>(ns) * stage;
    TLLM_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
    cudaLaunchConfig_t config{};
    cudaLaunchAttribute attrs[1];
    config.gridDim = dim3(std::max(1, std::min(ctas, items)));
    config.blockDim = dim3(kThreads);
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = enablePdl ? 1 : 0;
    config.attrs = attrs;
    config.numAttrs = 1;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, p, ns));
}

void checkAr(AllReduce const& ar)
{
    TLLM_CHECK_WITH_INFO(ar.ranks >= 1 && ar.ranks <= kMaxRanks && ar.rank >= 0 && ar.rank < ar.ranks,
        "glm5_mtp_step: all-reduce rank out of range");
}

} // namespace

void launchPre(PreParams const& p, int ctas, cudaStream_t stream, bool enablePdl)
{
    checkAr(p.ar);
    TLLM_CHECK_WITH_INFO(
        p.colStart == 0 || p.colStart == kHidden / 2, "glm5_mtp_step: pre colStart must be 0 or %d", kHidden / 2);
    launchOp<PreOp>(p, kHidden / kTileRows, ctas, stream, enablePdl);
}

void launchQkvA(QkvAParams const& p, int ctas, cudaStream_t stream, bool enablePdl)
{
    checkAr(p.ar);
    TLLM_CHECK_WITH_INFO(p.qkvRows % kTileRows == 0 && p.idxRows % kTileRows == 0, "glm5_mtp_step: qkvA rows");
    launchOp<QkvAOp>(p, (p.qkvRows + p.idxRows) / kTileRows, ctas, stream, enablePdl);
}

void launchQB(QBParams const& p, int ctas, cudaStream_t stream, bool enablePdl)
{
    TLLM_CHECK_WITH_INFO(
        p.qaRank % 512 == 0 && p.kvRank % 256 == 0 && p.qbRows % kTileRows == 0 && p.idxQRows % kTileRows == 0,
        "glm5_mtp_step: qB shape");
    launchOp<QBOp>(p, (p.qbRows + p.idxQRows) / kTileRows, ctas, stream, enablePdl);
}

void launchAbsorb(AbsorbParams const& p, int ctas, cudaStream_t stream, bool enablePdl)
{
    TLLM_CHECK_WITH_INFO(p.nope % 256 == 0 && p.kvRank % kTileRows == 0, "glm5_mtp_step: absorb shape");
    launchOp<AbsorbOp>(p, p.heads * p.kvRank / kTileRows, ctas, stream, enablePdl);
}

void launchVUp(VUpParams const& p, int ctas, cudaStream_t stream, bool enablePdl)
{
    TLLM_CHECK_WITH_INFO(p.kvRank % 512 == 0 && p.vDim % kTileRows == 0, "glm5_mtp_step: vUp shape");
    launchOp<VUpOp>(p, p.heads * p.vDim / kTileRows, ctas, stream, enablePdl);
}

void launchOProj(OProjParams const& p, int ctas, cudaStream_t stream, bool enablePdl)
{
    checkAr(p.ar);
    TLLM_CHECK_WITH_INFO(p.k % 1024 == 0, "glm5_mtp_step: oProj k");
    launchOp<OProjOp>(p, kHidden / kTileRows, ctas, stream, enablePdl);
}

void launchRouter(RouterParams const& p, int ctas, cudaStream_t stream, bool enablePdl)
{
    checkAr(p.ar);
    TLLM_CHECK_WITH_INFO(p.experts <= kMaxExperts && p.sharedInter % 8 == 0, "glm5_mtp_step: router shape");
    TLLM_CHECK_WITH_INFO(p.route != nullptr && p.routing.experts == p.experts && p.routing.topK <= kMaxTopK
            && p.routing.localExperts <= kConsumers,
        "glm5_mtp_step: router routing");
    launchOp<RouterOp>(p, (p.experts + kTileRows - 1) / kTileRows + p.sharedInter / 8, ctas, stream, enablePdl);
}

void launchFc1(Fc1Params const& p, int ctas, cudaStream_t stream, bool enablePdl)
{
    TLLM_CHECK_WITH_INFO(p.routing.experts <= kMaxExperts && p.routing.topK <= kMaxTopK && p.sharedInter % 512 == 0,
        "glm5_mtp_step: fc1 shape");
    // The expert items are known on the device only: size the grid for the most there can be.
    TLLM_CHECK_WITH_INFO(
        p.routing.localExperts <= kConsumers, "glm5_mtp_step: fc1 takes at most %d local experts", kConsumers);
    int const maxSlots = std::min({kMaxSlots, p.routing.localExperts, p.m * p.routing.topK});
    launchOp<Fc1Op>(p, maxSlots * Fc1Op::kItemsPerSlot + kHidden / kTileRows, ctas, stream, enablePdl);
}

void launchFc2(Fc2Params const& p, int ctas, cudaStream_t stream, bool enablePdl)
{
    checkAr(p.ar);
    TLLM_CHECK_WITH_INFO(
        p.routing.experts <= kMaxExperts && p.routing.topK <= kMaxTopK && p.routing.localExperts <= kConsumers,
        "glm5_mtp_step: fc2 shape");
    launchOp<Fc2Op>(p, kHidden / kTileRows, ctas, stream, enablePdl);
}

void launchHead(HeadParams const& p, int ctas, cudaStream_t stream, bool enablePdl)
{
    checkAr(p.ar);
    TLLM_CHECK_WITH_INFO(
        p.keyRanks >= 1 && p.keyRanks <= lm_head_argmax::kMaxRanks && p.keyRank >= 0 && p.keyRank < p.keyRanks,
        "glm5_mtp_step: head rank out of range");
    launchOp<HeadOp>(p, (p.n + kTileRows - 1) / kTileRows, std::min(ctas, lm_head_argmax::kMaxCtas), stream, enablePdl);
}

} // namespace glm5_mtp_step
} // namespace kernels

TRTLLM_NAMESPACE_END
