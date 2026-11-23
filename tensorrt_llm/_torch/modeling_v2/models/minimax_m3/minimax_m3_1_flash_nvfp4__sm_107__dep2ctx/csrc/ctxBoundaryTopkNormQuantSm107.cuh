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

// MiniMax-M3.1 CTX layer boundary with the MegaMoE top-k reduce folded in (DEP2CTX_TOPK_FOLD, K5 U2): the device code
// of invokeCtxBoundaryTopkNormQuant (ctxBoundaryNormQuantSm107.cu).
//
// A fragment like ctxBoundaryNormQuantSm107.cuh, included after it inside the same namespace: it reuses that file's
// ring, quad, reduction and quantize helpers. ctxBoundaryNormQuantSm107.cuh itself is not edited, so its flavors keep
// their SASS.
//
// A MoE layer whose MegaMoE launch deferred its top-k reduce ends today with two kernels per token row:
//   megamoe_topk_reduce_shared_add (megamoe_topk_reduce_tail.py):
//     acc = p[0] * s[0]; acc = fma(p[k], s[k], acc) for k = 1 .. K - 1;  routed = bf16(acc);
//     shared = bf16(f32(shared) + f32(routed))                              (FP32 mul / fma / add, round to nearest)
//   ctxBoundaryNormQuantKernel<kAdd, kNorm, kMx> (the next layer's input norm):
//     shared += residual (bf16x2), RMSNorm, MXFP8 + swizzled scales, written into the next QKV's MXFP8 slot.
// This kernel runs the same statements in the same order in one pass. A ring stage holds one row's K plane rows, its
// shared-expert row, its residual row and its K scores (one cp.async.bulk each; a token's plane rows are contiguous in
// the [T, K, hidden] plane). The reduce runs from shared memory, and from the BF16 moe vector on every statement is
// ctxBoundaryNormQuantKernel's. The moe rows never reach global memory: the shared buffer ends holding
// moe + residual, as after the two kernels. Every output is therefore the two kernels' bit for bit.
//
// Rounding: the reduce's multiply / fused multiply-add / add are written as explicit .rn PTX without .ftz, the
// modifiers CuTe DSL's mul_packed_f32x2 / fma_packed_f32x2 (rnd "rn", ftz unset) emit per lane, so no compile flag
// (--use_fast_math, -fmad) changes them; FP32 -> BF16 is cvt.rn (__float2bfloat16_rn), as the DSL's conversion.
//
// One stage is (K + 2) rows (72 KB at hidden 6144, K 4): only kCbnqStages = 3 stages fit the per-block opt-in, so this
// kernel runs one 768-thread CTA per SM (__launch_bounds__(768, 1)) at the base ring depth.

constexpr int kCbnqTopk = 4;            // plane rows per token the fold reduces (MiniMax-M3.1 top-4)
constexpr int kCbnqTopkScoreBytes = 16; // a row's kCbnqTopk FP32 scores: one 16-byte bulk copy into the stage

//! The fold's arguments (one launch).
struct CbnqTopkArgs
{
    __nv_bfloat16* shared;         // [rows, hidden] shared-expert rows; become moe + residual in place
    __nv_bfloat16 const* plane;    // [rows, kTopk, hidden] unweighted FC2 rows of the deferred launch
    float const* scores;           // [rows, kTopk] top-k weights
    __nv_bfloat16 const* residual; // [rows, hidden]
    __nv_bfloat16 const* weight;   // [hidden]
    __nv_bfloat16* normOut;        // [rows, hidden] (kNorm)
    uint64_t* qOut;                // [rows, hidden / 8] E4M3 x 8
    uint8_t* sfOut;                // swizzled 128x4 UE8M0 scales of ceil(rows / 128) * 128 rows
    int rows;
    int hidden;
    float eps;
};

//! Bytes of one ring stage: [plane rows k = 0 .. topk - 1 | shared row | residual row | scores].
__host__ __device__ constexpr int cbnqTopkStageBytes(int topk, int hidden)
{
    return (topk + 2) * hidden * 2 + kCbnqTopkScoreBytes;
}

//! Dynamic shared memory of one fold CTA: the ring, then cbnqSmemBytes's quad buffers, NVFP4 scale row (unused here,
//! kept so the pieces sit where ctxBoundaryNormQuantKernel puts them) and one mbarrier per stage.
__host__ __device__ constexpr int cbnqTopkSmemBytes(int stages, int topk, int hidden)
{
    return stages * cbnqTopkStageBytes(topk, hidden) + 2 * (hidden / 128) * 16 + (hidden / 16 + 15) / 16 * 16
        + stages * 8;
}

__device__ __forceinline__ float cbnqMulRn(float a, float b)
{
    float d;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

__device__ __forceinline__ float cbnqFmaRn(float a, float b, float c)
{
    float d;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
    return d;
}

__device__ __forceinline__ float cbnqAddRn(float a, float b)
{
    float d;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

//! One 1-D bulk copy into shared memory, completing on `bar` (16-byte multiple, 16-byte aligned on both sides).
__device__ __forceinline__ void cbnqBulkCopy(uint8_t* dst, void const* src, uint32_t bytes, uint32_t bar)
{
#if CBNQ_HAS_BULK_ASYNC
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                     cbnqSmemAddr(dst)),
                 "l"(src), "r"(bytes), "r"(bar)
                 : "memory");
#else
    __trap();
#endif
}

//! One token row into ring stage `dst`: arm the stage's barrier with the stage's bytes, then its four bulk copies.
__device__ __forceinline__ void cbnqLoadTopkRow(uint8_t* dst, __nv_bfloat16 const* plane, __nv_bfloat16 const* shared,
    __nv_bfloat16 const* residual, float const* scores, uint32_t rowBytes, uint32_t topk, uint64_t* bar)
{
#if CBNQ_HAS_BULK_ASYNC
    uint32_t const b = cbnqSmemAddr(bar);
    uint32_t const planeBytes = topk * rowBytes;
    uint32_t const bytes = planeBytes + 2 * rowBytes + kCbnqTopkScoreBytes;
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(bytes) : "memory");
    cbnqBulkCopy(dst, plane, planeBytes, b);
    cbnqBulkCopy(dst + planeBytes, shared, rowBytes, b);
    cbnqBulkCopy(dst + planeBytes + rowBytes, residual, rowBytes, b);
    cbnqBulkCopy(dst + planeBytes + 2 * rowBytes, scores, kCbnqTopkScoreBytes, b);
#else
    __trap();
#endif
}

//! megamoe_topk_reduce_shared_add's statements for this thread's 8 columns of one staged row: the moe vector.
template <int kTopk>
__device__ __forceinline__ int4 cbnqTopkReduceSharedAdd(uint8_t const* stage, int offset, uint32_t rowBytes)
{
    float4 const s4 = *reinterpret_cast<float4 const*>(stage + (kTopk + 2) * rowBytes);
    float const score[kCbnqTopk] = {s4.x, s4.y, s4.z, s4.w};
    CbnqPackedBf16 terms[kTopk];
#pragma unroll
    for (int k = 0; k < kTopk; ++k)
    {
        terms[k].packed = *reinterpret_cast<int4 const*>(stage + k * rowBytes + offset * 2);
    }
    CbnqPackedBf16 sharedVec;
    sharedVec.packed = *reinterpret_cast<int4 const*>(stage + kTopk * rowBytes + offset * 2);
    CbnqPackedBf16 moe;
#pragma unroll
    for (int e = 0; e < kCbnqPacked; ++e)
    {
        // k = 0 is a multiply, not fma(p, s, 0): fma(-x, s, +0) would turn a -0 product into +0.
        float acc = cbnqMulRn(__bfloat162float(reinterpret_cast<__nv_bfloat16 const*>(terms[0].unpacked)[e]), score[0]);
#pragma unroll
        for (int k = 1; k < kTopk; ++k)
        {
            acc = cbnqFmaRn(
                __bfloat162float(reinterpret_cast<__nv_bfloat16 const*>(terms[k].unpacked)[e]), score[k], acc);
        }
        __nv_bfloat16 const routed = __float2bfloat16_rn(acc);
        float const sharedF = __bfloat162float(reinterpret_cast<__nv_bfloat16 const*>(sharedVec.unpacked)[e]);
        reinterpret_cast<__nv_bfloat16*>(moe.unpacked)[e]
            = __float2bfloat16_rn(cbnqAddRn(sharedF, __bfloat162float(routed)));
    }
    return moe.packed;
}

//! The fold: per row, the top-k reduce + shared add, then ctxBoundaryNormQuantKernel<kStages, kAdd = true, kNorm,
//! kMx = true, no FP32, no NVFP4>'s statements on the moe vector (same ring order, quads, scale staging, PDL).
template <int kStages, int kTopk, bool kNorm>
__global__ void __launch_bounds__(kCbnqMaxThreads, 1) ctxBoundaryTopkNormQuantKernel(CbnqTopkArgs const args)
{
    static_assert(kTopk == kCbnqTopk, "a row's scores are one 16-byte vector");
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    extern __shared__ __align__(128) uint8_t cbnqSmem[];
    int const rows = args.rows;
    int const hidden = args.hidden;
    int const colIdx = threadIdx.x;
    int const offset = colIdx * kCbnqPacked;
    int const colThreads = hidden / kCbnqPacked;
    int const numKTiles = hidden / 128;
    int const numQuads = cbnqNumQuads(rows);
    uint32_t const rowBytes = static_cast<uint32_t>(hidden) * 2;
    uint32_t const stageBytes = static_cast<uint32_t>(cbnqTopkStageBytes(kTopk, hidden));
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
                cbnqLoadTopkRow(ring + s * stageBytes, args.plane + r * kTopk * hidden, args.shared + r * hidden,
                    args.residual + r * hidden, args.scores + r * kTopk, rowBytes, kTopk, &bars[s]);
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
        uint8_t* const sfBuf = sfStage + quadParity * numKTiles * 16;
#pragma unroll 1
        for (int j = 0; j < kCbnqQuadRows; ++j)
        {
            int const row = cbnqQuadRow(q, j);
            if (row >= rows)
            {
                if (colIdx % 4 == 0)
                {
                    int const k = colIdx / 4;
                    sfBuf[(k / 4) * 16 + j * 4 + k % 4] = 0;
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
            interVec.packed = cbnqTopkReduceSharedAdd<kTopk>(stage, offset, rowBytes);
            CbnqPackedBf16 residualVec;
            residualVec.packed = *reinterpret_cast<int4 const*>(stage + (kTopk + 1) * rowBytes + offset * 2);
            interVec.packed = cbnqAdd128b(interVec, residualVec);
            cbnqFenceProxyAsync();
            *reinterpret_cast<int4*>(args.shared + rowOffset) = interVec.packed;
            float acc = cbnqAccumulate(0.f, interVec);
            acc = cbnqBlockReduceSum(acc);
            // Every thread has read stage s (its reads precede the reduction's barrier): refill it kStages rows ahead.
            if (threadIdx.x == 0 && pq < numQuads)
            {
                int64_t const r = cbnqQuadRow(pq, pj);
                cbnqLoadTopkRow(ring + s * stageBytes, args.plane + r * kTopk * hidden, args.shared + r * hidden,
                    args.residual + r * hidden, args.scores + r * kTopk, rowBytes, kTopk, &bars[s]);
                cbnqNext(pq, pj, rows, numQuads);
            }
            float const denom = rsqrtf(acc / hidden + args.eps);
            CbnqPackedBf16 outVec;
            outVec.packed = cbnqRmsNorm(denom, interVec, weightVec);
            if constexpr (kNorm)
            {
                *reinterpret_cast<int4*>(args.normOut + rowOffset) = outVec.packed;
            }
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
            // cbnqBlockReduceSum's shared memory is written again by the next row.
            __syncthreads();
        }
        // The quad's 16-byte group of every K tile (get_sf_out_offset_128x4).
        __syncthreads();
        if (colIdx < numKTiles)
        {
            int64_t const dst = (static_cast<int64_t>(mTile) * numKTiles + colIdx) * 512 + mResidue * 16;
            *reinterpret_cast<int4*>(args.sfOut + dst) = reinterpret_cast<int4 const*>(sfBuf)[colIdx];
        }
    }
    // Every thread's stores are visible before any thread reports PDL completion (as in the served kernel).
    __threadfence();
    __syncthreads();
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}
