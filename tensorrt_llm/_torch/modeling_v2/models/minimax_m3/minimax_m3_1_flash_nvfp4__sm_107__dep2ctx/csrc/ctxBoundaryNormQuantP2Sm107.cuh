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

// MiniMax-M3.1 CTX layer boundary at two CTAs per SM on sm_107 (CBNQ_P2, K5 U3): the device code of the P2 launch in
// ctxBoundaryNormQuantSm107.cu.
//
// A fragment like ctxBoundaryNormQuantSm107.cuh, included after it inside the same namespace: it reuses that file's
// ring, quad, accumulate, norm and quantize helpers. ctxBoundaryNormQuantSm107.cuh itself is not edited, so the
// one-chunk kernel keeps its SASS.
//
// Why: an sm_107 SM holds at most 1024 resident threads, so the hidden / 8 = 768-thread ring CTA runs one per SM and
// two rows' serial chains (wait, add, accumulate, block reduce, norm, quantize, __syncthreads) never overlap there.
// Here a CTA has hidden / 16 = 384 threads, each owning two 8-element chunks: thread t, half h owns chunk
// c = t + (hidden / 16) * h, which is the one-chunk kernel's thread c ("virtual thread"). Two CTAs (768 threads) fit
// an SM, so two rows are in their chains at once. Every output byte is the one-chunk kernel's:
//  - per chunk, the same statements on the same 8 elements (add, accumulate, norm, FP32 widening, quantize);
//  - the block reduction: physical warp p holds virtual warps p (h = 0) and p + hidden / 512 (h = 1), lane l holding
//    virtual lane l of both (hidden % 512 == 0). Each half is warp-reduced with the same xor tree (the same 32
//    operands per virtual warp), stored at its virtual warp's shared slot, and the final xor tree reads the virtual
//    warp count (hidden / 256), not blockDim.x / 32: the same operands in the same lanes as the one-chunk kernel;
//  - the MXFP8 / NVFP4 quantize helpers shuffle within lane groups of 4 / 2: for half h, lanes 4g .. 4g + 3 of warp
//    p hold chunks c .. c + 3 of one 32-element block, as the virtual warp's lanes do. Each helper runs once per half
//    (warp-uniform), and the scale bytes are indexed by the chunk c;
//  - the ring (cp.async.bulk + mbarrier, thread 0 producer), the quad order, the scale-tile staging, the stage tail and
//    PDL are the one-chunk kernel's. Which CTA owns a quad changes with the grid; one row is always one CTA.

constexpr int kCbnqP2Chunks = 2;       // 8-element chunks per thread
constexpr int kCbnqP2MaxThreads = 384; // hidden / 16 at hidden 6144; the launch bound below
constexpr int kCbnqP2MinBlocks = 2;    // CTAs per SM: 768 of sm_107's 1024 threads

//! cbnqBlockReduceSum over the virtual 2 x blockDim.x threads: acc0 is virtual thread threadIdx.x's partial sum, acc1
//! virtual thread threadIdx.x + blockDim.x's. Returns the one-chunk kernel's sum bit for bit (blockDim.x % 32 == 0,
//! 2 x blockDim.x / 32 <= 32 virtual warps).
__device__ __forceinline__ float cbnqBlockReduceSumP2(float acc0, float acc1)
{
    __shared__ float smem[kCbnqWarp];
    int const laneId = threadIdx.x % kCbnqWarp;
    int const warpId = threadIdx.x / kCbnqWarp;
    int const physWarps = blockDim.x / kCbnqWarp;
    int const virtWarps = kCbnqP2Chunks * physWarps;
    acc0 = cbnqWarpReduceSum(acc0);
    acc1 = cbnqWarpReduceSum(acc1);
    if (laneId == 0)
    {
        smem[warpId] = acc0;
        smem[warpId + physWarps] = acc1;
    }
    __syncthreads();
    float val = laneId < virtWarps ? smem[laneId] : 0.f;
    val = cbnqWarpReduceSum(val);
    return val;
}

//! The P2 form of ctxBoundaryNormQuantKernel (same template flags, same arguments): block = hidden / 16 threads, two
//! chunks per thread.
template <int kStages, bool kAdd, bool kNorm, bool kMx, bool kF32, bool kFp4>
__global__ void __launch_bounds__(kCbnqP2MaxThreads, kCbnqP2MinBlocks) ctxBoundaryNormQuantP2Kernel(CbnqArgs const args)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    extern __shared__ __align__(128) uint8_t cbnqSmem[];
    int const rows = args.rows;
    int const hidden = args.hidden;
    int const half = blockDim.x; // = hidden / 16: virtual thread c = threadIdx.x + half * h
    int const colThreads = hidden / kCbnqPacked;
    int const numKTiles = hidden / 128;
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

    float const fp4SfScale = kFp4 ? args.fp4GlobalScale[0] : 1.0f;
    CbnqPackedBf16 weightVec[kCbnqP2Chunks];
#pragma unroll
    for (int h = 0; h < kCbnqP2Chunks; ++h)
    {
        weightVec[h].packed = *reinterpret_cast<int4 const*>(args.weight + (threadIdx.x + half * h) * kCbnqPacked);
    }

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
        uint8_t* const sfBuf = sfStage + quadParity * numKTiles * 16;
#pragma unroll 1
        for (int j = 0; j < kCbnqQuadRows; ++j)
        {
            int const row = cbnqQuadRow(q, j);
            if (row >= rows)
            {
                // The served kernel zero-fills the scale rows that pad the row count to a multiple of 128: both halves.
                if constexpr (kMx)
                {
#pragma unroll
                    for (int h = 0; h < kCbnqP2Chunks; ++h)
                    {
                        int const c = threadIdx.x + half * h;
                        if (c % 4 == 0)
                        {
                            int const k = c / 4;
                            sfBuf[(k / 4) * 16 + j * 4 + k % 4] = 0;
                        }
                    }
                }
                continue;
            }
            int const s = consumed % kStages;
            uint32_t const parity = static_cast<uint32_t>(consumed / kStages) & 1u;
            ++consumed;
            cbnqWait(&bars[s], parity);
            uint8_t const* const stage = ring + s * stageBytes;
            CbnqPackedBf16 interVec[kCbnqP2Chunks];
#pragma unroll
            for (int h = 0; h < kCbnqP2Chunks; ++h)
            {
                int const offset = (threadIdx.x + half * h) * kCbnqPacked;
                interVec[h].packed = *reinterpret_cast<int4 const*>(stage + offset * 2);
                if constexpr (kAdd)
                {
                    CbnqPackedBf16 residualVec;
                    residualVec.packed = *reinterpret_cast<int4 const*>(stage + rowBytes + offset * 2);
                    interVec[h].packed = cbnqAdd128b(interVec[h], residualVec);
                }
            }
            cbnqFenceProxyAsync();
            float acc[kCbnqP2Chunks];
#pragma unroll
            for (int h = 0; h < kCbnqP2Chunks; ++h)
            {
                if constexpr (kAdd)
                {
                    int64_t const rowOffset
                        = static_cast<int64_t>(row) * hidden + (threadIdx.x + half * h) * kCbnqPacked;
                    *reinterpret_cast<int4*>(args.inter + rowOffset) = interVec[h].packed;
                }
                acc[h] = cbnqAccumulate(0.f, interVec[h]);
            }
            float const sum = cbnqBlockReduceSumP2(acc[0], acc[1]);
            // Every thread has read stage s (its reads precede the reduction's barrier): refill it kStages rows ahead.
            if (threadIdx.x == 0 && pq < numQuads)
            {
                int64_t const r = cbnqQuadRow(pq, pj);
                cbnqLoadRow(ring + s * stageBytes, args.inter + r * hidden, kAdd ? args.residual + r * hidden : nullptr,
                    rowBytes, &bars[s]);
                cbnqNext(pq, pj, rows, numQuads);
            }
            float const denom = rsqrtf(sum / hidden + args.eps);
#pragma unroll
            for (int h = 0; h < kCbnqP2Chunks; ++h)
            {
                int const c = threadIdx.x + half * h;
                int64_t const rowOffset = static_cast<int64_t>(row) * hidden + c * kCbnqPacked;
                CbnqPackedBf16 outVec;
                outVec.packed = cbnqRmsNorm(denom, interVec[h], weightVec[h]);
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
                    // The one-chunk kernel's staging with colIdx = c (k = c / 4 of the row's 32-element scales).
                    tensorrt_llm::kernels::PackedVec<__nv_bfloat16> quantIn;
                    *reinterpret_cast<int4*>(&quantIn) = outVec.packed;
                    uint8_t* sf = nullptr;
                    if (c % 4 == 0)
                    {
                        int const k = c / 4;
                        sf = sfBuf + (k / 4) * 16 + j * 4 + k % 4;
                    }
                    args.qOut[static_cast<int64_t>(row) * colThreads + c]
                        = tensorrt_llm::kernels::cvt_warp_fp16_to_mxfp8<__nv_bfloat16, 32>(quantIn, sf);
                }
                if constexpr (kFp4)
                {
                    tensorrt_llm::kernels::PackedVec<__nv_bfloat16> fp4In;
                    *reinterpret_cast<int4*>(&fp4In) = outVec.packed;
                    uint8_t* sf4 = c % 2 == 0 ? fp4SfStage + c / 2 : nullptr;
                    args.fp4Out[static_cast<int64_t>(row) * args.fp4StrideWords + c]
                        = tensorrt_llm::kernels::cvt_warp_fp16_to_fp4<__nv_bfloat16, 16, false>(fp4In, fp4SfScale, sf4);
                }
            }
            // cbnqBlockReduceSumP2's shared memory and the staged NVFP4 scale row are written again by the next row.
            __syncthreads();
            if constexpr (kFp4)
            {
                // hidden / 256 <= hidden / 16 threads: one pass.
                if (threadIdx.x < hidden / 256)
                {
                    reinterpret_cast<int4*>(args.fp4SfOut + static_cast<int64_t>(row) * args.fp4SfStride)[threadIdx.x]
                        = reinterpret_cast<int4 const*>(fp4SfStage)[threadIdx.x];
                }
            }
        }
        if constexpr (kMx)
        {
            // hidden / 128 <= hidden / 16 threads: one pass (get_sf_out_offset_128x4's 16-byte group mResidue).
            __syncthreads();
            if (threadIdx.x < numKTiles)
            {
                int64_t const dst = (static_cast<int64_t>(mTile) * numKTiles + threadIdx.x) * 512 + mResidue * 16;
                *reinterpret_cast<int4*>(args.sfOut + dst) = reinterpret_cast<int4 const*>(sfBuf)[threadIdx.x];
            }
        }
    }

    // The MegaMoE stage's padded tail (noaux_tc_fp4_quantize_stage_out's): ids -1 and weights 0 in the first tailTopk
    // columns of rows [rows, tail rows). Any block size covers it (grid-stride over the cells).
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
    __threadfence();
    __syncthreads();
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}
