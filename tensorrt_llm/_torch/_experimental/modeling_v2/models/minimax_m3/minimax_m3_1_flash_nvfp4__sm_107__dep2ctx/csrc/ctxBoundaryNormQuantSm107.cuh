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

// MiniMax-M3.1 CTX layer boundary (NORMQ_V2 / NORMQ_STAGE): the device code of ctxBoundaryNormQuantSm107.cu.
//
// This is a fragment, not a standalone header: it is included inside a namespace, after <cuda_bf16.h> /
// <cuda_runtime.h> and the quantization helpers it calls unqualified (CVT_ELTS_PER_THREAD, PackedVec<__nv_bfloat16>,
// cvt_warp_fp16_to_mxfp8<__nv_bfloat16, 32>, cvt_warp_fp16_to_fp4<__nv_bfloat16, 16, false>). In TRT-LLM those are
// quantization.cuh's; the microbenchmark includes this file verbatim next to its copies of the same helpers.
//
// Arithmetic: ctxResidualRmsNormMxfp8Kernel's (= rms_norm_rows_kernel's), unchanged: one thread per 8-element chunk
// (hidden / 8 threads per row), the same per-thread accumulate order and the same warp-xor (16..1) -> smem ->
// warp-xor block reduction, the same rsqrtf / multiply order, the same quantize helpers on the same packed input.
// Every output byte is therefore the served op's. What changes is how many rows are in flight:
//  - two CTAs per SM (__launch_bounds__(768, 2)) with grid = resident CTAs x SMs;
//  - a kCbnqStages-deep shared-memory ring per CTA: thread 0 loads the inter (+ residual) rows with cp.async.bulk
//    (1-D TMA, one mbarrier per stage) kCbnqStages rows ahead, so the loads of the next rows overlap this row's
//    reduction instead of waiting behind its __syncthreads;
//  - rows are taken in "quads" (rows m, m + 32, m + 64, m + 96 of one 128-row scale tile) so the MXFP8 scale bytes
//    of a quad fill whole 16-byte groups of the swizzled 128x4 layout: they are staged in shared memory and written
//    with one coalesced 16-byte store per K tile instead of one byte store per 4 threads. Pad rows of the last tile
//    stage zeros (the served kernel's pad loop);
//  - the NVFP4 linear scales of a row are staged the same way and written as hidden / 256 16-byte stores.
// The row-to-CTA map does not enter the arithmetic (one row is always one CTA's 768 threads).

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define CBNQ_HAS_BULK_ASYNC 1
#else
#define CBNQ_HAS_BULK_ASYNC 0
#endif

constexpr int kCbnqPacked = 8;       // BF16 elements per thread (one 16-byte chunk), = CVT_ELTS_PER_THREAD
constexpr int kCbnqWarp = 32;
constexpr int kCbnqStages = 3;       // ring depth (rows in flight per CTA)
constexpr int kCbnqMaxThreads = 768; // hidden / 8 at hidden 6144; the launch bound below
constexpr int kCbnqMinBlocks = 2;    // CTAs per SM the launch bound asks for
constexpr int kCbnqQuadRows = 4;     // rows per quad: one 16-byte group of a swizzled 128x4 scale tile
constexpr int kCbnqTileRows = 128;

//! The kernel's arguments (one launch). Pointers that a flavor does not use may be null.
struct CbnqArgs
{
    __nv_bfloat16* inter;          // [rows, hidden]; becomes inter + residual in place (kAdd), else holds that sum
    __nv_bfloat16 const* residual; // [rows, hidden] (kAdd)
    __nv_bfloat16 const* weight;   // [hidden]
    __nv_bfloat16* normOut;        // [rows, hidden] (kNorm)
    uint64_t* qOut;                // [rows, hidden / 8] E4M3 x 8 (kMx)
    uint8_t* sfOut;                // swizzled 128x4 UE8M0 scales of ceil(rows / 128) * 128 rows (kMx)
    float* f32Out;                 // [rows, hidden] (kF32)
    uint32_t* fp4Out;              // NVFP4 rows, row stride fp4StrideWords 32-bit words (kFp4)
    uint8_t* fp4SfOut;             // linear E4M3 scales, row stride fp4SfStride bytes, 16-byte aligned rows (kFp4)
    float const* fp4GlobalScale;   // [1] (kFp4)
    int32_t* tailIds;              // optional: rows [rows, tailIdsRows) x first tailTopk columns <- -1
    float* tailWeights;            // optional: rows [rows, tailWeightsRows) x first tailTopk columns <- 0
    int64_t fp4StrideWords;
    int64_t fp4SfStride;
    int64_t tailIdsStride;
    int64_t tailWeightsStride;
    int tailIdsRows;
    int tailWeightsRows;
    int tailTopk;
    int rows;
    int hidden;
    float eps;
};

//! Dynamic shared memory of one CTA: the ring (inter (+ residual) rows per stage), two swizzled-scale quad buffers
//! (hidden / 128 K tiles x 16 bytes each), one NVFP4 scale row (hidden / 16 bytes, rounded to 16) and one mbarrier per
//! stage. Every piece starts 16-byte aligned.
__host__ __device__ constexpr int cbnqSmemBytes(int stages, bool add, int hidden)
{
    return stages * (add ? 2 : 1) * hidden * 2 + 2 * (hidden / 128) * 16 + (hidden / 16 + 15) / 16 * 16 + stages * 8;
}

//! Quads of a launch: 32 per 128-row scale tile (the 32 row residues of the tile).
__host__ __device__ constexpr int cbnqNumQuads(int rows)
{
    return (rows + kCbnqTileRows - 1) / kCbnqTileRows * 32;
}

//! Row j (0..3) of quad q: rows q % 32 + 32 j of tile q / 32.
__host__ __device__ constexpr int cbnqQuadRow(int q, int j)
{
    return (q / 32) * kCbnqTileRows + j * 32 + (q % 32);
}

// customAllReduceKernels.cu's reduce_fusion helpers as ctxResidualRmsNormMxfp8Sm103.cu copies them: the same arithmetic
// in the same order.
__device__ __forceinline__ float cbnqWarpReduceSum(float val)
{
    val += __shfl_xor_sync(~0, val, 16);
    val += __shfl_xor_sync(~0, val, 8);
    val += __shfl_xor_sync(~0, val, 4);
    val += __shfl_xor_sync(~0, val, 2);
    val += __shfl_xor_sync(~0, val, 1);
    return val;
}

__device__ __forceinline__ float cbnqBlockReduceSum(float val)
{
    __shared__ float smem[kCbnqWarp];
    int const laneId = threadIdx.x % kCbnqWarp;
    int const warpId = threadIdx.x / kCbnqWarp;
    int const warpNum = blockDim.x / kCbnqWarp;
    val = cbnqWarpReduceSum(val);
    if (laneId == 0)
    {
        smem[warpId] = val;
    }
    __syncthreads();
    val = laneId < warpNum ? smem[laneId] : 0.f;
    val = cbnqWarpReduceSum(val);
    return val;
}

union CbnqPackedBf16
{
    int4 packed;
    __nv_bfloat162 unpacked[4];
};

__device__ __forceinline__ int4 cbnqAdd128b(CbnqPackedBf16& a, CbnqPackedBf16& b)
{
    CbnqPackedBf16 c;
    c.unpacked[0] = a.unpacked[0] + b.unpacked[0];
    c.unpacked[1] = a.unpacked[1] + b.unpacked[1];
    c.unpacked[2] = a.unpacked[2] + b.unpacked[2];
    c.unpacked[3] = a.unpacked[3] + b.unpacked[3];
    return c.packed;
}

__device__ __forceinline__ float cbnqAccumulate(float acc, CbnqPackedBf16& vec)
{
#pragma unroll
    for (int i = 0; i < kCbnqPacked; ++i)
    {
        float v = static_cast<float>(reinterpret_cast<__nv_bfloat16*>(vec.unpacked)[i]);
        acc += v * v;
    }
    return acc;
}

__device__ __forceinline__ int4 cbnqRmsNorm(float denom, CbnqPackedBf16& vec, CbnqPackedBf16& weight)
{
    CbnqPackedBf16 ret;
#pragma unroll
    for (int i = 0; i < kCbnqPacked; ++i)
    {
        float v1 = static_cast<float>(reinterpret_cast<__nv_bfloat16*>(vec.unpacked)[i]);
        float v2 = static_cast<float>(reinterpret_cast<__nv_bfloat16*>(weight.unpacked)[i]);
        reinterpret_cast<__nv_bfloat16*>(ret.unpacked)[i] = static_cast<__nv_bfloat16>(v1 * denom * v2);
    }
    return ret.packed;
}

// --- 1-D TMA ring (cp.async.bulk + mbarrier; sm_90+) ---------------------------------------------------------------
__device__ __forceinline__ uint32_t cbnqSmemAddr(void const* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cbnqBarInit(uint64_t* bar)
{
#if CBNQ_HAS_BULK_ASYNC
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(cbnqSmemAddr(bar)) : "memory");
#else
    __trap(); // the host launches this kernel on sm_10x only
#endif
}

__device__ __forceinline__ void cbnqFenceBarInit()
{
#if CBNQ_HAS_BULK_ASYNC
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
}

//! Orders this thread's generic-proxy reads of a ring stage before the async-proxy refill of that stage.
__device__ __forceinline__ void cbnqFenceProxyAsync()
{
#if CBNQ_HAS_BULK_ASYNC
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
}

//! One row into ring stage `dst`: arm the stage's barrier with the bytes, then the 1-D bulk copies (16-byte multiples,
//! 16-byte aligned on both sides).
__device__ __forceinline__ void cbnqLoadRow(
    uint8_t* dst, __nv_bfloat16 const* inter, __nv_bfloat16 const* residual, uint32_t rowBytes, uint64_t* bar)
{
#if CBNQ_HAS_BULK_ASYNC
    uint32_t const b = cbnqSmemAddr(bar);
    uint32_t const bytes = residual != nullptr ? 2 * rowBytes : rowBytes;
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(bytes) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                     cbnqSmemAddr(dst)),
                 "l"(inter), "r"(rowBytes), "r"(b)
                 : "memory");
    if (residual != nullptr)
    {
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                         cbnqSmemAddr(dst + rowBytes)),
                     "l"(residual), "r"(rowBytes), "r"(b)
                     : "memory");
    }
#else
    __trap();
#endif
}

__device__ __forceinline__ void cbnqWait(uint64_t* bar, uint32_t parity)
{
#if CBNQ_HAS_BULK_ASYNC
    uint32_t const b = cbnqSmemAddr(bar);
    uint32_t done = 0;
    do
    {
        asm volatile(
            "{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, "
            "p;\n\t}\n"
            : "=r"(done)
            : "r"(b), "r"(parity)
            : "memory");
    } while (done == 0);
#else
    __trap();
#endif
}

//! Moves the producer cursor (quad q, row j) forward to the first quad row that exists (row < rows), or past the end.
__device__ __forceinline__ void cbnqSettle(int& q, int& j, int rows, int numQuads)
{
    while (q < numQuads && cbnqQuadRow(q, j) >= rows)
    {
        if (++j == kCbnqQuadRows)
        {
            j = 0;
            q += gridDim.x;
        }
    }
}

__device__ __forceinline__ void cbnqNext(int& q, int& j, int rows, int numQuads)
{
    if (++j == kCbnqQuadRows)
    {
        j = 0;
        q += gridDim.x;
    }
    cbnqSettle(q, j, rows, numQuads);
}

//! kAdd: inter += residual (written back) before the norm (the boundary); else inter already holds that sum and only
//! the norm is recomputed from it (the BF16 norm a dropped-norm boundary left out, bitwise the same rows).
//! kNorm / kMx / kF32 / kFp4: which outputs this launch writes (BF16 norm, MXFP8 + swizzled scales, FP32 widening of
//! the norm, NVFP4 + linear scales). Block = hidden / 8 threads (<= 768).
template <int kStages, bool kAdd, bool kNorm, bool kMx, bool kF32, bool kFp4>
__global__ void __launch_bounds__(kCbnqMaxThreads, kCbnqMinBlocks) ctxBoundaryNormQuantKernel(CbnqArgs const args)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    extern __shared__ __align__(128) uint8_t cbnqSmem[];
    int const rows = args.rows;
    int const hidden = args.hidden;
    int const colIdx = threadIdx.x;
    int const offset = colIdx * kCbnqPacked;
    int const colThreads = hidden / kCbnqPacked;
    int const numKTiles = hidden / 128; // 128x4 scale tiles per 128 rows: hidden / 32 scales, four per tile row
    int const numQuads = cbnqNumQuads(rows);
    uint32_t const rowBytes = static_cast<uint32_t>(hidden) * 2;
    uint32_t const stageBytes = (kAdd ? 2u : 1u) * rowBytes;
    uint8_t* const ring = cbnqSmem;
    uint8_t* const sfStage = ring + kStages * stageBytes;
    uint8_t* const fp4SfStage = sfStage + 2 * numKTiles * 16;
    uint64_t* const bars = reinterpret_cast<uint64_t*>(fp4SfStage + (hidden / 16 + 15) / 16 * 16);

    if (threadIdx.x == 0)
    {
#pragma unroll
        for (int s = 0; s < kStages; ++s)
        {
            cbnqBarInit(&bars[s]);
        }
        cbnqFenceBarInit();
    }
    __syncthreads();

    cudaGridDependencySynchronize();

    // quantize_with_block_size's per-tensor scale (SFScaleVal) for the NVFP4 copy.
    float const fp4SfScale = kFp4 ? args.fp4GlobalScale[0] : 1.0f;
    CbnqPackedBf16 weightVec;
    weightVec.packed = *reinterpret_cast<int4 const*>(args.weight + offset);

    // Producer (thread 0): the first kStages rows of this CTA's sequence (quad-major, j ascending, existing rows).
    int pq = blockIdx.x;
    int pj = 0;
    if (threadIdx.x == 0)
    {
        cbnqSettle(pq, pj, rows, numQuads);
#pragma unroll
        for (int s = 0; s < kStages; ++s)
        {
            if (pq < numQuads)
            {
                int64_t const r = cbnqQuadRow(pq, pj);
                cbnqLoadRow(ring + s * stageBytes, args.inter + r * hidden, kAdd ? args.residual + r * hidden : nullptr,
                    rowBytes, &bars[s]);
                cbnqNext(pq, pj, rows, numQuads);
            }
        }
    }

    int consumed = 0;
    int quadParity = 0;
    for (int q = blockIdx.x; q < numQuads; q += gridDim.x, quadParity ^= 1)
    {
        int const mTile = q / 32;
        int const mResidue = q % 32;
        // Double-buffered by quad: a pad-only quad stages its zeros without a barrier, so it must not write the buffer
        // the previous quad's stores still read.
        uint8_t* const sfBuf = sfStage + quadParity * numKTiles * 16;
#pragma unroll 1
        for (int j = 0; j < kCbnqQuadRows; ++j)
        {
            int const row = cbnqQuadRow(q, j);
            if (row >= rows)
            {
                // The served kernel zero-fills the scale rows that pad the row count to a multiple of 128.
                if constexpr (kMx)
                {
                    if (colIdx % 4 == 0)
                    {
                        int const k = colIdx / 4;
                        sfBuf[(k / 4) * 16 + j * 4 + k % 4] = 0;
                    }
                }
                continue;
            }
            int const s = consumed % kStages;
            uint32_t const parity = static_cast<uint32_t>(consumed / kStages) & 1u;
            ++consumed;
            cbnqWait(&bars[s], parity);
            uint8_t const* const stage = ring + s * stageBytes;
            int64_t const rowOffset = static_cast<int64_t>(row) * hidden + offset;
            CbnqPackedBf16 interVec;
            interVec.packed = *reinterpret_cast<int4 const*>(stage + offset * 2);
            if constexpr (kAdd)
            {
                CbnqPackedBf16 residualVec;
                residualVec.packed = *reinterpret_cast<int4 const*>(stage + rowBytes + offset * 2);
                interVec.packed = cbnqAdd128b(interVec, residualVec);
            }
            cbnqFenceProxyAsync();
            if constexpr (kAdd)
            {
                *reinterpret_cast<int4*>(args.inter + rowOffset) = interVec.packed;
            }
            float acc = cbnqAccumulate(0.f, interVec);
            acc = cbnqBlockReduceSum(acc);
            // Every thread has read stage s (its reads precede the reduction's barrier): refill it kStages rows ahead.
            if (threadIdx.x == 0 && pq < numQuads)
            {
                int64_t const r = cbnqQuadRow(pq, pj);
                cbnqLoadRow(ring + s * stageBytes, args.inter + r * hidden, kAdd ? args.residual + r * hidden : nullptr,
                    rowBytes, &bars[s]);
                cbnqNext(pq, pj, rows, numQuads);
            }
            float const denom = rsqrtf(acc / hidden + args.eps);
            CbnqPackedBf16 outVec;
            outVec.packed = cbnqRmsNorm(denom, interVec, weightVec);
            if constexpr (kNorm)
            {
                *reinterpret_cast<int4*>(args.normOut + rowOffset) = outVec.packed;
            }
            if constexpr (kF32)
            {
                float2 const f0 = __bfloat1622float2(outVec.unpacked[0]);
                float2 const f1 = __bfloat1622float2(outVec.unpacked[1]);
                float2 const f2 = __bfloat1622float2(outVec.unpacked[2]);
                float2 const f3 = __bfloat1622float2(outVec.unpacked[3]);
                reinterpret_cast<float4*>(args.f32Out + rowOffset)[0] = make_float4(f0.x, f0.y, f1.x, f1.y);
                reinterpret_cast<float4*>(args.f32Out + rowOffset)[1] = make_float4(f2.x, f2.y, f3.x, f3.y);
            }
            if constexpr (kMx)
            {
                // cvt_quant_get_sf_out_offset's SWIZZLED byte of (row, k = colIdx / 4) is tile (mTile, k / 4), byte
                // (row % 32) * 16 + j * 4 + k % 4: the quad's rows share row % 32 and fill the 16-byte group.
                tensorrt_llm::kernels::PackedVec<__nv_bfloat16> quantIn;
                *reinterpret_cast<int4*>(&quantIn) = outVec.packed;
                uint8_t* sf = nullptr;
                if (colIdx % 4 == 0)
                {
                    int const k = colIdx / 4;
                    sf = sfBuf + (k / 4) * 16 + j * 4 + k % 4;
                }
                args.qOut[static_cast<int64_t>(row) * colThreads + colIdx]
                    = tensorrt_llm::kernels::cvt_warp_fp16_to_mxfp8<__nv_bfloat16, 32>(quantIn, sf);
            }
            if constexpr (kFp4)
            {
                // fp4_quantize(norm, globalScale, 16, ue8m0 = false, swizzled = false): E2M1 x 8 per thread, one E4M3
                // scale per 16 elements (byte colIdx / 2 of the row's linear scales), staged for 16-byte stores.
                tensorrt_llm::kernels::PackedVec<__nv_bfloat16> fp4In;
                *reinterpret_cast<int4*>(&fp4In) = outVec.packed;
                uint8_t* sf4 = colIdx % 2 == 0 ? fp4SfStage + colIdx / 2 : nullptr;
                args.fp4Out[static_cast<int64_t>(row) * args.fp4StrideWords + colIdx]
                    = tensorrt_llm::kernels::cvt_warp_fp16_to_fp4<__nv_bfloat16, 16, false>(fp4In, fp4SfScale, sf4);
            }
            // cbnqBlockReduceSum's shared memory and the staged NVFP4 scale row are written again by the next row.
            __syncthreads();
            if constexpr (kFp4)
            {
                if (colIdx < hidden / 256)
                {
                    reinterpret_cast<int4*>(args.fp4SfOut + static_cast<int64_t>(row) * args.fp4SfStride)[colIdx]
                        = reinterpret_cast<int4 const*>(fp4SfStage)[colIdx];
                }
            }
        }
        if constexpr (kMx)
        {
            // The quad's 16-byte group of every K tile: tile (mTile, kt) at (mTile * numKTiles + kt) * 512, group
            // mResidue * 16 (get_sf_out_offset_128x4).
            __syncthreads();
            if (colIdx < numKTiles)
            {
                int64_t const dst = (static_cast<int64_t>(mTile) * numKTiles + colIdx) * 512 + mResidue * 16;
                *reinterpret_cast<int4*>(args.sfOut + dst) = reinterpret_cast<int4 const*>(sfBuf)[colIdx];
            }
        }
    }

    // The MegaMoE stage's padded tail (noaux_tc_fp4_quantize_stage_out's): ids -1 and weights 0 in the first tailTopk
    // columns of rows [rows, tail rows).
    if (args.tailTopk > 0)
    {
        int64_t const step = static_cast<int64_t>(gridDim.x) * blockDim.x;
        int64_t const first = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (args.tailIds != nullptr)
        {
            int64_t const n = static_cast<int64_t>(max(args.tailIdsRows - rows, 0)) * args.tailTopk;
            for (int64_t i = first; i < n; i += step)
            {
                args.tailIds[(rows + i / args.tailTopk) * args.tailIdsStride + i % args.tailTopk] = -1;
            }
        }
        if (args.tailWeights != nullptr)
        {
            int64_t const n = static_cast<int64_t>(max(args.tailWeightsRows - rows, 0)) * args.tailTopk;
            for (int64_t i = first; i < n; i += step)
            {
                args.tailWeights[(rows + i / args.tailTopk) * args.tailWeightsStride + i % args.tailTopk] = 0.f;
            }
        }
    }
    // Every thread's stores are visible before any thread reports PDL completion (as in the served kernel).
    __threadfence();
    __syncthreads();
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}
