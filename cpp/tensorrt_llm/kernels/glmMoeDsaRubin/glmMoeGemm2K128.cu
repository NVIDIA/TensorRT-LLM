/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
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

// FC2 of the NVFP4 MoE at intermediate size 128 (K = 128) and tile N 8 (see glmMoeGemm2K128.h for the contract).
//
// Work. A unit is (token tile t, 128-row tile of the output features); the units are split evenly over a persistent
// grid of one CTA per SM, each CTA owning a contiguous run of units (one or two token tiles). A CTA holds every unit of
// its run at once: the weight tiles (8 KB of E2M1, SWIZZLE_64B, by TMA in two-tile boxes) and their 128x4 block scales
// (one bulk copy per token tile) are requested before the grid-dependency wait, the activations of its token tiles
// (8 rows x 64 B) and their 8x4 scales after it.
//
// MMA. tcgen05.mma kind::mxf4nvf4 block16, M = 128 (weight rows), N = 8 (token rows), two K = 64 steps per unit, issued
// by the elected lane of the converged MMA warp, one commit per unit; the weight scales go into TMEM by tcgen05.cp
// ahead of the MMAs (a separate warp, as their bulk copy lands), the token scales after the wait. The TRTLLM-Gen cubin
// computes the same two K steps and two more on the zero padding of its 256-wide K tile; adding +0.0f to the
// accumulator gives its result, bitwise.
//
// Accumulators. Unit k of a token tile's run writes its 8 TMEM columns at base + stride * k, stride = the tile's valid
// rows rounded up to 2, 4 or 8 (accumulators are 2-column aligned): the valid rows of all units of the tile lie in
// stride * units consecutive columns, and the tail columns a unit shares with the next one are padding rows, written
// in issue order. The epilogue reads them in 16-column loads, each once its last unit is complete.
//
// Warps: 0-3 epilogue (TMEM lanes 32 w .. 32 w + 31 = weight rows of each tile), 4 weight loads, 5 TMEM allocation,
// activations and MMA, 6 weight-scale copies.

#include "tensorrt_llm/kernels/glmMoeDsaRubin/glmMoeGemm2K128.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cuda.h>
#include <cuda_bf16.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa::gemm2K128
{
namespace
{

constexpr int kTileRows = 128;
constexpr int kK = 128;
constexpr int kTokens = 8;
constexpr int kTileBytesA = kTileRows * kK / 2;   // 8,192: 128 rows x 64 B, SWIZZLE_64B
constexpr int kTileBytesSf = kTileRows * kK / 16; // 1,024: two 512 B blocks of the 128x4 layout
constexpr int kActBytes = kTokens * kK / 2;       // 512: one SWIZZLE_64B atom
constexpr int kSfbBytes = 512;                    // 32 rows x 16 B: the tcgen05.cp source of a token tile's scales
constexpr int kMaxUnits = 15;                     // units per CTA
constexpr int kMaxSlots = 2;                      // token tiles per CTA (a run of kMaxUnits units spans at most two)
constexpr int kMaxTokenTiles = 64;
constexpr int kThreads = 224;
constexpr uint32_t kTmemCols = 512;
constexpr uint32_t kTmemAcc = 0;   // packed accumulators, at most 256 columns
constexpr uint32_t kTmemSfa = 256; // kMaxUnits x 8 columns: the weight scales of every unit
constexpr uint32_t kTmemSfb = 376; // kMaxSlots x 4 columns: per K step a column of token scales and a zero column
// Instruction descriptor, kind::mxf4nvf4: A/B E2M1, UE4M3 scales, K-major, M = 128, N = 8.
constexpr uint32_t kIdesc = (1u << 7) | (1u << 10) | ((8u >> 3) << 17) | ((128u >> 4) << 24);
// K-major SWIZZLE_64B operand descriptor without its start address: 8-row x 64 B atoms 512 B apart, version 1.
constexpr uint64_t kDescSw64 = (1ull << 16) | (static_cast<uint64_t>(512 >> 4) << 32) | (1ull << 46) | (4ull << 61);
// tcgen05.cp 32x128b source descriptor without its start address: 8-row x 16 B core matrices 128 B apart.
constexpr uint64_t kDescSf = (static_cast<uint64_t>(128 >> 4) << 32) | (1ull << 46);
constexpr int kSmemBytes = kMaxUnits * (kTileBytesA + kTileBytesSf) + kMaxSlots * (kActBytes + kSfbBytes) + 1024;

struct Params
{
    CUtensorMap tmW2; // weights [experts * hidden rows][64 B], box {64, 256}, SWIZZLE_64B
    CUtensorMap tmW1; // the same tensor, box {64, 128}
    uint8_t const* wSf;
    uint8_t const* act;
    uint8_t const* actSf;
    float const* scale;
    int32_t const* numNonExitingCtas;
    int32_t const* ctaIdxXyToBatchIdx;
    int32_t const* ctaIdxXyToMnLimit;
    __nv_bfloat16* out;
    int32_t hidden;
    int32_t maxCtas;
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1100

__device__ __forceinline__ uint32_t smemU32(void const* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbarInit(uint64_t* bar, uint32_t count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smemU32(bar)), "r"(count) : "memory");
}

__device__ __forceinline__ void mbarExpectTx(uint64_t* bar, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smemU32(bar)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbarWait(uint64_t* bar, uint32_t parity)
{
    uint32_t ok = 0;
    while (!ok)
    {
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
            "selp.u32 %0, 1, 0, p;\n"
            "}\n"
            : "=r"(ok)
            : "r"(smemU32(bar)), "r"(parity)
            : "memory");
    }
}

__device__ __forceinline__ void tmaLoad2d(void* dst, void const* map, int x, int y, uint64_t* bar)
{
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(
            smemU32(dst)),
        "l"(map), "r"(x), "r"(y), "r"(smemU32(bar))
        : "memory");
}

__device__ __forceinline__ void bulkLoad(void* dst, void const* src, uint32_t bytes, uint64_t* bar)
{
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(smemU32(dst)),
        "l"(src), "r"(bytes), "r"(smemU32(bar))
        : "memory");
}

__device__ __forceinline__ int ldRelaxed(int32_t const* p)
{
    int v;
    asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void namedBarSync(int id, int threads)
{
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

__device__ __forceinline__ void namedBarArrive(int id, int threads)
{
    asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

__device__ __forceinline__ void fenceProxyAsyncSmem()
{
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void tcFenceBefore()
{
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcFenceAfter()
{
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tmemAlloc(uint32_t* dst, uint32_t cols)
{
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smemU32(dst)), "r"(cols)
                 : "memory");
}

__device__ __forceinline__ void tmemRelinquish()
{
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmemDealloc(uint32_t taddr, uint32_t cols)
{
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(cols) : "memory");
}

// The following three are executed by a whole (converged) warp; its elected lane issues the instruction.
__device__ __forceinline__ void utccpElect(uint32_t taddr, uint64_t sdesc)
{
    asm volatile(
        "{\n"
        ".reg .pred e;\n"
        "elect.sync _|e, 0xffffffff;\n"
        "@e tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;\n"
        "}\n" ::"r"(taddr),
        "l"(sdesc)
        : "memory");
}

__device__ __forceinline__ void mmaElect(uint32_t d, uint64_t a, uint64_t b, uint32_t sfa, uint32_t sfb, uint32_t acc)
{
    asm volatile(
        "{\n"
        ".reg .pred e, p;\n"
        "elect.sync _|e, 0xffffffff;\n"
        "setp.ne.b32 p, %6, 0;\n"
        "@e tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%4], [%5], p;\n"
        "}\n" ::"r"(d),
        "l"(a), "l"(b), "r"(kIdesc), "r"(sfa), "r"(sfb), "r"(acc)
        : "memory");
}

__device__ __forceinline__ void commitElect(uint64_t* bar)
{
    asm volatile(
        "{\n"
        ".reg .pred e;\n"
        "elect.sync _|e, 0xffffffff;\n"
        "@e tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
        "}\n" ::"r"(smemU32(bar))
        : "memory");
}

// Columns [taddr, taddr + 16) of the warp's TMEM lane quarter.
__device__ __forceinline__ void tmemLd16(uint32_t taddr, float (&v)[16])
{
    uint32_t r[16];
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
        "%15}, [%16];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
        : "r"(taddr)
        : "memory");
    asm volatile("tcgen05.wait::ld.sync.aligned;"
                 : "+r"(r[0]), "+r"(r[1]), "+r"(r[2]), "+r"(r[3]), "+r"(r[4]), "+r"(r[5]), "+r"(r[6]), "+r"(r[7]),
                 "+r"(r[8]), "+r"(r[9]), "+r"(r[10]), "+r"(r[11]), "+r"(r[12]), "+r"(r[13]), "+r"(r[14]), "+r"(r[15])
                 :
                 : "memory");
#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        v[i] = __uint_as_float(r[i]);
    }
}

// Stores the valid rows of the units [kBase, kBase + 16 / kStride) of a token tile from 16 packed accumulator columns
// (column kStride * k + r = row r of unit kBase + k). rowPtr: row 0 of the tile at this thread's feature of the run's
// first unit.
template <int kStride>
__device__ __forceinline__ void storeChunk(
    float const (&v)[16], __nv_bfloat16* rowPtr, size_t rowStride, int kBase, int runLen, int nv, float scale)
{
    constexpr int kUnits = 16 / kStride;
#pragma unroll
    for (int k = 0; k < kUnits; ++k)
    {
        if (kBase + k < runLen)
        {
            __nv_bfloat16* o = rowPtr + (kBase + k) * kTileRows;
#pragma unroll
            for (int r = 0; r < kStride; ++r)
            {
                if (r < nv)
                {
                    o[r * rowStride] = __float2bfloat16_rn(__fadd_rn(v[kStride * k + r], 0.0f) * scale);
                }
            }
        }
    }
}

__device__ __forceinline__ int unitStart(int c, int units, int grid)
{
    return static_cast<int>(static_cast<long long>(c) * units / grid);
}

#endif

__global__ void __launch_bounds__(kThreads, 1) gemm2K128Kernel(const __grid_constant__ Params p)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1100
    extern __shared__ __align__(1024) uint8_t smemRaw[];
    __shared__ __align__(8) uint64_t fullBar[kMaxUnits];
    __shared__ __align__(8) uint64_t sfaBar[kMaxSlots];
    __shared__ __align__(8) uint64_t sfaReady[kMaxUnits];
    __shared__ __align__(8) uint64_t accFull[kMaxUnits];
    __shared__ uint32_t sTmem;
    __shared__ int sRange[2];
    __shared__ int sExpert[kMaxSlots];
    __shared__ int sNv[kMaxSlots];
    __shared__ int sStride[kMaxSlots];
    __shared__ int sCol[kMaxSlots];

    uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smemRaw) + 1023) & ~uintptr_t(1023));
    uint8_t* smemA = smem;
    uint8_t* smemSf = smemA + kMaxUnits * kTileBytesA;
    uint8_t* smemAct = smemSf + kMaxUnits * kTileBytesSf;
    uint8_t* smemSfb = smemAct + kMaxSlots * kActBytes;

    int const warp = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
    int const tilesPerSlot = p.hidden / kTileRows;

    if (warp == 4 && lane == 0)
    {
        for (int i = 0; i < kMaxUnits; ++i)
        {
            mbarInit(&fullBar[i], 1);
            mbarInit(&sfaReady[i], 1);
            mbarInit(&accFull[i], 1);
        }
        for (int j = 0; j < kMaxSlots; ++j)
        {
            mbarInit(&sfaBar[j], 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        asm volatile("prefetch.tensormap [%0];" ::"l"(&p.tmW2) : "memory");
        asm volatile("prefetch.tensormap [%0];" ::"l"(&p.tmW1) : "memory");
    }
    if (warp == 5)
    {
        tmemAlloc(&sTmem, kTmemCols);
        tmemRelinquish();
    }
    tcFenceBefore();
    __syncthreads();
    tcFenceAfter();
    uint32_t const tmem = sTmem;

    if (warp == 4)
    {
        // The routing outputs are complete when this kernel starts: its primary, the FC1 cubin, triggers its dependents
        // only after its own grid-dependency wait on the routing kernel. One round trip for the tile count and the
        // experts and valid rows of the token tiles; the weights and scales are constants.
        int const numTiles = ldRelaxed(p.numNonExitingCtas);
        int const e0 = lane < p.maxCtas ? ldRelaxed(p.ctaIdxXyToBatchIdx + lane) : 0;
        int const e1 = lane + 32 < p.maxCtas ? ldRelaxed(p.ctaIdxXyToBatchIdx + lane + 32) : 0;
        int const m0 = lane < p.maxCtas ? ldRelaxed(p.ctaIdxXyToMnLimit + lane) : 0;
        int const m1 = lane + 32 < p.maxCtas ? ldRelaxed(p.ctaIdxXyToMnLimit + lane + 32) : 0;
        int const units = numTiles * tilesPerSlot;
        int const a = unitStart(blockIdx.x, units, gridDim.x);
        int const b = unitStart(blockIdx.x + 1, units, gridDim.x);
        // Publish the run (its experts, valid rows and accumulator columns) before issuing the loads.
        if (b > a)
        {
            int const tA = a / tilesPerSlot;
            int const tB = (b - 1) / tilesPerSlot;
            int col = 0;
            for (int t = tA; t <= tB; ++t)
            {
                int const eLo = __shfl_sync(0xffffffffu, e0, t & 31);
                int const eHi = __shfl_sync(0xffffffffu, e1, t & 31);
                int const mLo = __shfl_sync(0xffffffffu, m0, t & 31);
                int const mHi = __shfl_sync(0xffffffffu, m1, t & 31);
                int const nv = (t < 32 ? mLo : mHi) - kTokens * t;
                int const stride = nv <= 2 ? 2 : (nv <= 4 ? 4 : 8);
                int const runLen = min(b, (t + 1) * tilesPerSlot) - max(a, t * tilesPerSlot);
                if (lane == 0)
                {
                    sExpert[t - tA] = t < 32 ? eLo : eHi;
                    sNv[t - tA] = nv;
                    sStride[t - tA] = stride;
                    sCol[t - tA] = col;
                }
                // The next token tile starts at a 32-column boundary (aligned TMEM loads).
                col = (col + stride * runLen + 31) & ~31;
            }
        }
        if (lane == 0)
        {
            sRange[0] = a;
            sRange[1] = b;
        }
        __syncwarp();
        namedBarArrive(1, kThreads);
        // Per token tile of the run: all its block scales in one bulk copy (first, for the scale-copy warp), then its
        // weight tiles in two-tile boxes (a one-tile box for an odd last one).
        if (lane == 0)
        {
            int i = 0;
            int slot = 0;
            while (a + i < b)
            {
                int const u0 = a + i;
                int const t = u0 / tilesPerSlot;
                int const nt0 = u0 - t * tilesPerSlot;
                int const n = min(b - u0, tilesPerSlot - nt0);
                mbarExpectTx(&sfaBar[slot], static_cast<uint32_t>(n * kTileBytesSf));
                bulkLoad(smemSf + i * kTileBytesSf,
                    p.wSf + (static_cast<size_t>(sExpert[slot]) * tilesPerSlot + nt0) * kTileBytesSf,
                    static_cast<uint32_t>(n * kTileBytesSf), &sfaBar[slot]);
                i += n;
                ++slot;
            }
            i = 0;
            slot = 0;
            while (a + i < b)
            {
                int const u0 = a + i;
                int const t = u0 / tilesPerSlot;
                int const nt0 = u0 - t * tilesPerSlot;
                int const n = min(b - u0, tilesPerSlot - nt0);
                int const row0 = sExpert[slot] * p.hidden + nt0 * kTileRows;
                for (int k = 0; k < n; k += 2)
                {
                    bool const two = k + 1 < n;
                    mbarExpectTx(&fullBar[i + k], two ? 2 * kTileBytesA : kTileBytesA);
                    tmaLoad2d(smemA + (i + k) * kTileBytesA, two ? &p.tmW2 : &p.tmW1, 0, row0 + k * kTileRows,
                        &fullBar[i + k]);
                }
                i += n;
                ++slot;
            }
        }
    }
    else
    {
        namedBarSync(1, kThreads);
    }
    int const a = sRange[0];
    int const b = sRange[1];
    int const n = b - a;
    int const t0 = n > 0 ? a / tilesPerSlot : 0;
    int const nt0 = a - t0 * tilesPerSlot;
    int const numSlots = n > 0 ? (b - 1) / tilesPerSlot - t0 + 1 : 0;

    if (warp == 6)
    {
        // The weight scales of every unit into TMEM as their bulk copy lands (before the grid-dependency wait returns,
        // for a CTA that runs while the previous kernel does); the MMA warp waits on sfaReady[unit].
        int u = 0;
        int nt = nt0;
        for (int j = 0; j < numSlots; ++j)
        {
            int const runLen = min(n - u, tilesPerSlot - nt);
            mbarWait(&sfaBar[j], 0);
            tcFenceAfter();
            for (int k = 0; k < runLen; ++k)
            {
                uint32_t const src = smemU32(smemSf + (u + k) * kTileBytesSf);
                uint32_t const dst = tmem + kTmemSfa + 8 * (u + k);
                utccpElect(dst, kDescSf | ((src & 0x3ffffu) >> 4));
                utccpElect(dst + 4, kDescSf | (((src + 512) & 0x3ffffu) >> 4));
                commitElect(&sfaReady[u + k]);
            }
            u += runLen;
            nt = 0;
        }
    }

    // Everything below reads FC1's output or writes this kernel's output (the weight-load and scale-copy warps read
    // neither). The dependents may launch once every CTA is past its wait.
    if (warp < 4 || warp == 5)
    {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    if (threadIdx.x == 0)
    {
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }

    if (warp < 4)
    {
        // Epilogue, per token tile of the run, in 16-column chunks, each once its last unit is complete.
        uint32_t const tq = tmem + (static_cast<uint32_t>(32 * warp) << 16);
        // The weight rows are shuffled for the 128-row epilogue tile: row 32 w + l of a tile holds its feature
        // 32 w + 4 (l & 7) + l / 8.
        int const feature = 32 * warp + (lane & 7) * 4 + (lane >> 3);
        int runStart = 0;
        for (int j = 0; j < numSlots; ++j)
        {
            int const t = t0 + j;
            int const ntFirst = j == 0 ? nt0 : 0;
            int const runLen = min(n - runStart, tilesPerSlot - ntFirst);
            int const nv = sNv[j];
            int const stride = sStride[j];
            float const scale = __ldg(p.scale + sExpert[j]);
            int const unitsPerChunk = 16 / stride;
            __nv_bfloat16* const rowBase
                = p.out + static_cast<size_t>(t) * kTokens * p.hidden + ntFirst * kTileRows + feature;
            for (int kBase = 0; kBase < runLen; kBase += unitsPerChunk)
            {
                mbarWait(&accFull[runStart + min(kBase + unitsPerChunk, runLen) - 1], 0);
                tcFenceAfter();
                float v[16];
                tmemLd16(tq + kTmemAcc + sCol[j] + stride * kBase, v);
                if (stride == 2)
                {
                    storeChunk<2>(v, rowBase, p.hidden, kBase, runLen, nv, scale);
                }
                else if (stride == 4)
                {
                    storeChunk<4>(v, rowBase, p.hidden, kBase, runLen, nv, scale);
                }
                else
                {
                    storeChunk<8>(v, rowBase, p.hidden, kBase, runLen, nv, scale);
                }
            }
            runStart += runLen;
        }
    }
    else if (warp == 5 && n > 0)
    {
        // Activations of the run's token tiles (8 rows of K = 128 E2M1, SWIZZLE_64B) and their 8x4 block scales, all
        // loaded in one round trip. The scales go through shared memory (rows 0-7 of a 32-row block: per K step a
        // column of 4 scales and a zero column) into every TMEM lane quarter.
        int const r8 = lane >> 2;
        int const cnk = lane & 3;
        uint4 actV[kMaxSlots];
        uint32_t sf0[kMaxSlots];
        uint32_t sf1[kMaxSlots];
#pragma unroll
        for (int j = 0; j < kMaxSlots; ++j)
        {
            actV[j] = make_uint4(0, 0, 0, 0);
            sf0[j] = 0;
            sf1[j] = 0;
            if (j < numSlots)
            {
                actV[j] = __ldcg(reinterpret_cast<uint4 const*>(
                    p.act + (static_cast<size_t>(t0 + j) * kTokens + r8) * (kK / 2) + cnk * 16));
                if (lane < kTokens)
                {
                    sf0[j] = __ldcg(reinterpret_cast<uint32_t const*>(p.actSf) + 2 * (t0 + j) * kTokens + lane);
                    sf1[j] = __ldcg(reinterpret_cast<uint32_t const*>(p.actSf) + (2 * (t0 + j) + 1) * kTokens + lane);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < kMaxSlots; ++j)
        {
            if (j < numSlots)
            {
                *reinterpret_cast<uint4*>(smemAct + j * kActBytes + r8 * 64 + ((cnk ^ ((r8 >> 1) & 3)) << 4)) = actV[j];
                *reinterpret_cast<uint4*>(smemSfb + j * kSfbBytes + lane * 16) = make_uint4(sf0[j], 0u, sf1[j], 0u);
            }
        }
        fenceProxyAsyncSmem();
        __syncwarp();
        for (int j = 0; j < numSlots; ++j)
        {
            utccpElect(tmem + kTmemSfb + 4 * j, kDescSf | ((smemU32(smemSfb + j * kSfbBytes) & 0x3ffffu) >> 4));
        }

        int slot = 0;
        int nt = nt0;
        int runIdx = 0; // index of the unit in its token tile's run
        uint32_t aBase = smemU32(smemA);
        uint32_t bBase = smemU32(smemAct);
        uint32_t sfb = tmem + kTmemSfb;
        uint32_t d = tmem + kTmemAcc + sCol[0];
        uint32_t stride = sStride[0];
        for (int i = 0; i < n; ++i)
        {
            if (nt == tilesPerSlot)
            {
                ++slot;
                nt = 0;
                runIdx = 0;
                bBase += kActBytes;
                sfb += 4;
                d = tmem + kTmemAcc + sCol[slot];
                stride = sStride[slot];
            }
            if ((runIdx & 1) == 0)
            {
                // The first unit of a two-tile box.
                mbarWait(&fullBar[i], 0);
            }
            mbarWait(&sfaReady[i], 0);
            tcFenceAfter();
            uint32_t const sfa = tmem + kTmemSfa + 8 * i;
            uint64_t const da = kDescSw64 | ((aBase & 0x3ffffu) >> 4);
            uint64_t const db = kDescSw64 | ((bBase & 0x3ffffu) >> 4);
            mmaElect(d, da, db, sfa, sfb, 0u);
            mmaElect(d, da + 2, db + 2, sfa + 4, sfb + 2, 1u);
            // A commit per unit, although the epilogue waits only for the last unit of each chunk: with a commit per
            // chunk instead, the issue of a unit takes about 1.4x longer.
            commitElect(&accFull[i]);
            ++nt;
            ++runIdx;
            aBase += kTileBytesA;
            d += stride;
        }
    }

    tcFenceBefore();
    __syncthreads();
    if (warp == 5)
    {
        tcFenceAfter();
        tmemDealloc(tmem, kTmemCols);
    }
#endif
}

void encodeWeightMap(CUtensorMap* map, void const* weights, uint64_t rows, uint32_t boxRows)
{
    cuuint64_t const dims[2] = {64, rows};
    cuuint64_t const strides[1] = {64};
    cuuint32_t const box[2] = {64, boxRows};
    cuuint32_t const elementStrides[2] = {1, 1};
    TLLM_CU_CHECK(tensorrt_llm::common::CUDADriverWrapper::getInstance()->cuTensorMapEncodeTiled(map,
        CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(weights), dims, strides, box, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

} // namespace

bool isSupported(int32_t hiddenSize, int32_t maxNumCtas)
{
    int const sm = tensorrt_llm::common::getSMVersion();
    if (sm < 100 || sm >= 110 || hiddenSize % kTileRows != 0 || hiddenSize / kTileRows < kMaxUnits || maxNumCtas < 1
        || maxNumCtas > kMaxTokenTiles)
    {
        return false;
    }
    int const grid = tensorrt_llm::common::getMultiProcessorCount();
    int const maxUnits = maxNumCtas * (hiddenSize / kTileRows);
    return (maxUnits + grid - 1) / grid <= kMaxUnits;
}

void run(Data const& data, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(isSupported(data.hiddenSize, data.maxNumCtas), "gemm2K128: unsupported shape.");
    TLLM_CUDA_CHECK(cudaFuncSetAttribute(gemm2K128Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    Params p{};
    uint64_t const rows = static_cast<uint64_t>(data.numLocalExperts) * data.hiddenSize;
    encodeWeightMap(&p.tmW2, data.weights, rows, 2 * kTileRows);
    encodeWeightMap(&p.tmW1, data.weights, rows, kTileRows);
    p.wSf = static_cast<uint8_t const*>(data.weightsScale);
    p.act = static_cast<uint8_t const*>(data.act);
    p.actSf = static_cast<uint8_t const*>(data.actScale);
    p.scale = data.outputScale;
    p.numNonExitingCtas = data.numNonExitingCtas;
    p.ctaIdxXyToBatchIdx = data.ctaIdxXyToBatchIdx;
    p.ctaIdxXyToMnLimit = data.ctaIdxXyToMnLimit;
    p.out = static_cast<__nv_bfloat16*>(data.output);
    p.hidden = data.hiddenSize;
    p.maxCtas = data.maxNumCtas;

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(tensorrt_llm::common::getMultiProcessorCount());
    config.blockDim = dim3(kThreads);
    config.dynamicSmemBytes = kSmemBytes;
    config.stream = stream;
    cudaLaunchAttribute attributes[2]{};
    attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attributes[0].val.programmaticStreamSerializationAllowed = data.usePdl ? 1 : 0;
    attributes[1].id = cudaLaunchAttributePreferredSharedMemoryCarveout;
    attributes[1].val.sharedMemCarveout = static_cast<unsigned int>(data.sharedMemCarveout);
    config.attrs = attributes;
    config.numAttrs = data.sharedMemCarveout >= 0 ? 2 : 1;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, gemm2K128Kernel, p));
}

} // namespace kernels::glm_moe_dsa::gemm2K128

TRTLLM_NAMESPACE_END
