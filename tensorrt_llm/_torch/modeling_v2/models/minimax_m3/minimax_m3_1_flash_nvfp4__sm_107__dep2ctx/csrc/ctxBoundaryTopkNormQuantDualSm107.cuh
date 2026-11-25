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

// MiniMax-M3.1 CTX top-k fold boundary with two row groups per CTA (CBNQ_TOPK_DUAL): the device code of the
// dual launch in ctxBoundaryNormQuantSm107.cu.
//
// A fragment like ctxBoundaryTopkNormQuantSm107.cuh, included after it inside the same namespace: it reuses that
// file's stage layout, reduce and bulk-copy helpers and ctxBoundaryNormQuantSm107.cuh's ring, quad, accumulate, norm
// and quantize helpers. Neither file is edited, so their kernels keep their SASS.
//
// Why: the fold's (K + 2)-row ring stage (72 KB at hidden 6144, K 4) leaves room for one 768-thread CTA per SM, so one
// row's serial chain (wait, reduce, add, accumulate, block reduce, norm, quantize, __syncthreads) runs at a time on an
// SM: the two-CTA plan the boundary kernels share (CBNQ_P2: two CTAs per SM on sm_107) does not hold for the fold.
// Here the same 768-thread CTA and the same ring run two row groups of hidden / 16 = 384 threads, each owning two
// 8-element chunks per thread (thread t, half h of group g owns chunk c = t + (hidden / 16) * h, the one-chunk fold's
// thread c), so two rows' chains overlap on the SM as two P2 CTAs' do. Every output byte is the one-chunk fold's:
//  - per chunk, the same statements on the same 8 elements (reduce, shared add, residual add, accumulate, norm,
//    quantize) in the same order;
//  - the row reduction is cbnqBlockReduceSumP2's (virtual warps p and p + hidden / 512 of physical warp p, the final
//    xor tree over hidden / 256 slots), on the group's own slots and named barrier;
//  - the MXFP8 helper shuffles in lane groups of 4 of one physical warp, which hold chunks c .. c + 3 of one 32-element
//    block, as in P2; the scale bytes are staged by chunk index in the quad buffer, as in the fold;
//  - rows, their order per CTA (quad-major, j ascending), the grid and the ring depth are the fold's.
//
// Ring: position p of the CTA's row sequence lands in stage p % kStages (parity (p / kStages) & 1) and belongs to
// group p % 2. Only the owner reads a stage, so, as in the fold, the owner's leader refills the stage with position
// p + kStages once its group has read it (fence.proxy.async + its named barrier), then counts the release in a
// shared-memory counter per stage. The other group consumes position p + kStages: it waits until the counter shows
// position p released (the refill armed) before its mbarrier parity wait, so that wait never runs two phases ahead
// of the barrier (the groups skip each other's positions). Thread 0 issues the first kStages positions.

constexpr int kCbnqDualGroups = 2; // row groups per CTA

//! Row of position `p` in this CTA's sequence (quads blockIdx.x + i * gridDim.x, rows j ascending, existing rows
//! only), or -1 past the end. Every quad below the last 128-row tile holds four rows; the last tile's quads hold a
//! prefix j < count of them (rows j * 32 + q % 32 below the tile's row count).
__device__ __forceinline__ int cbnqDualPosRow(int p, int rows, int numQuads)
{
    int const lastTileQuad = (numQuads / 32 - 1) * 32; // first quad of the last tile
    int const first = blockIdx.x;
    int const step = gridDim.x;
    int const fullQuads = first < lastTileQuad ? (lastTileQuad - first + step - 1) / step : 0;
    if (p < 4 * fullQuads)
    {
        return cbnqQuadRow(first + (p / 4) * step, p % 4);
    }
    p -= 4 * fullQuads;
    for (int q = first + fullQuads * step; q < numQuads; q += step)
    {
        int count = 0;
        while (count < kCbnqQuadRows && cbnqQuadRow(q, count) < rows)
        {
            ++count;
        }
        if (p < count)
        {
            return cbnqQuadRow(q, p);
        }
        p -= count;
    }
    return -1;
}

__device__ __forceinline__ void cbnqDualGroupSync(int group, int threads)
{
    asm volatile("bar.sync %0, %1;" ::"r"(1 + group), "r"(threads) : "memory");
}

//! The row sum of cbnqBlockReduceSumP2 over the group's virtual 2 x `threads` threads (acc0: virtual thread t, acc1:
//! virtual thread t + threads), on the group's own slots and barrier.
__device__ __forceinline__ float cbnqDualReduceSum(
    float acc0, float acc1, int tid, int group, int threads, float* slots)
{
    int const laneId = tid % kCbnqWarp;
    int const warpId = tid / kCbnqWarp;
    int const physWarps = threads / kCbnqWarp;
    int const virtWarps = kCbnqP2Chunks * physWarps;
    acc0 = cbnqWarpReduceSum(acc0);
    acc1 = cbnqWarpReduceSum(acc1);
    if (laneId == 0)
    {
        slots[warpId] = acc0;
        slots[warpId + physWarps] = acc1;
    }
    cbnqDualGroupSync(group, threads);
    float val = laneId < virtWarps ? slots[laneId] : 0.f;
    val = cbnqWarpReduceSum(val);
    return val;
}

//! The owner's release of position `p` (stage s, read by its whole group): refill the stage with position
//! p + kStages (arming its mbarrier phase), then count the release. Called by the group's thread 0.
template <int kStages, int kTopk>
__device__ __forceinline__ void cbnqDualRelease(int p, int s, unsigned* released, uint8_t* ring, uint32_t stageBytes,
    uint64_t* bars, CbnqTopkArgs const& args, int numQuads, uint32_t rowBytes)
{
    int const r = cbnqDualPosRow(p + kStages, args.rows, numQuads);
    if (r >= 0)
    {
        int64_t const row = r;
        int const hidden = args.hidden;
        cbnqLoadTopkRow(ring + s * stageBytes, args.plane + row * kTopk * hidden, args.shared + row * hidden,
            args.residual + row * hidden, args.scores + row * kTopk, rowBytes, kTopk, &bars[s]);
    }
    __threadfence_block();
    atomicAdd(&released[s], 1u);
}

//! Waits until stage s counts `need` releases (positions s, s + kStages, ..., the one before the caller's).
__device__ __forceinline__ void cbnqDualWaitReleased(unsigned const* released, int s, unsigned need)
{
    while (*reinterpret_cast<unsigned const volatile*>(&released[s]) < need)
    {
    }
    __threadfence_block();
}

//! The dual-group fold: ctxBoundaryTopkNormQuantKernel's outputs bit for bit, two rows in flight per CTA.
//! Block = hidden / 8 threads (two groups of hidden / 16).
template <int kStages, int kTopk, bool kNorm>
__global__ void __launch_bounds__(kCbnqMaxThreads, 1) ctxBoundaryTopkNormQuantDualKernel(CbnqTopkArgs const args)
{
    static_assert(kTopk == kCbnqTopk, "a row's scores are one 16-byte vector");
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    extern __shared__ __align__(128) uint8_t cbnqSmem[];
    __shared__ float slots[kCbnqDualGroups][kCbnqWarp];
    __shared__ unsigned released[kStages];
    int const rows = args.rows;
    int const hidden = args.hidden;
    int const threads = hidden / (kCbnqP2Chunks * kCbnqPacked); // per group
    int const group = threadIdx.x / threads;
    int const tid = threadIdx.x - group * threads;
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
            released[s] = 0u;
        }
        cbnqFenceBarInit();
    }
    __syncthreads();

    cudaGridDependencySynchronize();

    CbnqPackedBf16 weightVec[kCbnqP2Chunks];
#pragma unroll
    for (int h = 0; h < kCbnqP2Chunks; ++h)
    {
        weightVec[h].packed = *reinterpret_cast<int4 const*>(args.weight + (tid + threads * h) * kCbnqPacked);
    }

    // Thread 0: the first kStages positions.
    if (threadIdx.x == 0)
    {
#pragma unroll
        for (int s = 0; s < kStages; ++s)
        {
            int const r = cbnqDualPosRow(s, rows, numQuads);
            if (r >= 0)
            {
                int64_t const row = r;
                cbnqLoadTopkRow(ring + s * stageBytes, args.plane + row * kTopk * hidden, args.shared + row * hidden,
                    args.residual + row * hidden, args.scores + row * kTopk, rowBytes, kTopk, &bars[s]);
            }
        }
    }

    float* const mySlots = slots[group];
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
                // The served kernel zero-fills the scale rows that pad the row count to a multiple of 128 (group 0).
                if (group == 0)
                {
#pragma unroll
                    for (int h = 0; h < kCbnqP2Chunks; ++h)
                    {
                        int const c = tid + threads * h;
                        if (c % 4 == 0)
                        {
                            int const k = c / 4;
                            sfBuf[(k / 4) * 16 + j * 4 + k % 4] = 0;
                        }
                    }
                }
                continue;
            }
            int const p = consumed++;
            int const s = p % kStages;
            if ((p % kCbnqDualGroups) != group)
            {
                continue; // the other group's row
            }
            uint32_t const parity = static_cast<uint32_t>(p / kStages) & 1u;
            if (p >= kStages)
            {
                cbnqDualWaitReleased(released, s, static_cast<unsigned>(p / kStages));
            }
            cbnqWait(&bars[s], parity);
            uint8_t const* const stage = ring + s * stageBytes;
            CbnqPackedBf16 interVec[kCbnqP2Chunks];
            float acc[kCbnqP2Chunks];
#pragma unroll
            for (int h = 0; h < kCbnqP2Chunks; ++h)
            {
                int const offset = (tid + threads * h) * kCbnqPacked;
                interVec[h].packed = cbnqTopkReduceSharedAdd<kTopk>(stage, offset, rowBytes);
                CbnqPackedBf16 residualVec;
                residualVec.packed = *reinterpret_cast<int4 const*>(stage + (kTopk + 1) * rowBytes + offset * 2);
                interVec[h].packed = cbnqAdd128b(interVec[h], residualVec);
            }
            cbnqFenceProxyAsync();
#pragma unroll
            for (int h = 0; h < kCbnqP2Chunks; ++h)
            {
                int64_t const rowOffset = static_cast<int64_t>(row) * hidden + (tid + threads * h) * kCbnqPacked;
                *reinterpret_cast<int4*>(args.shared + rowOffset) = interVec[h].packed;
                acc[h] = cbnqAccumulate(0.f, interVec[h]);
            }
            float const sum = cbnqDualReduceSum(acc[0], acc[1], tid, group, threads, mySlots);
            // Every thread of the group has read stage s (its reads precede the reduction's barrier): refill it.
            if (tid == 0)
            {
                cbnqDualRelease<kStages, kTopk>(p, s, released, ring, stageBytes, bars, args, numQuads, rowBytes);
            }
            float const denom = rsqrtf(sum / hidden + args.eps);
#pragma unroll
            for (int h = 0; h < kCbnqP2Chunks; ++h)
            {
                int const c = tid + threads * h;
                int64_t const rowOffset = static_cast<int64_t>(row) * hidden + c * kCbnqPacked;
                CbnqPackedBf16 outVec;
                outVec.packed = cbnqRmsNorm(denom, interVec[h], weightVec[h]);
                if constexpr (kNorm)
                {
                    *reinterpret_cast<int4*>(args.normOut + rowOffset) = outVec.packed;
                }
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
            // The group's reduction slots are written again by its next row.
            cbnqDualGroupSync(group, threads);
        }
        // The quad's 16-byte group of every K tile (get_sf_out_offset_128x4): both groups' scale bytes are staged.
        __syncthreads();
        if (threadIdx.x < numKTiles)
        {
            int64_t const dst = (static_cast<int64_t>(mTile) * numKTiles + threadIdx.x) * 512 + mResidue * 16;
            *reinterpret_cast<int4*>(args.sfOut + dst) = reinterpret_cast<int4 const*>(sfBuf)[threadIdx.x];
        }
    }
    // Every thread's stores are visible before any thread reports PDL completion (as in the served kernel).
    __threadfence();
    __syncthreads();
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}
