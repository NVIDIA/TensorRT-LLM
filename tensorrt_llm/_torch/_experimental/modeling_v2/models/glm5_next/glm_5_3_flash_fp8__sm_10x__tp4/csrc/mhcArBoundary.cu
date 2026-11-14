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
// One-shot all-reduce of a decode step's partial rows fused with the mHC boundary that follows it (GLM-5.3-Flash at
// tp4: hidden 4096, hc_mult 4, at most 8 rows), as one cluster of 16 CTAs.
//
// All-reduce: TensorRT-LLM's Lamport one-shot protocol on the all-reduce workspace (the LamportComm layout: three
// rotating receive buffers per rank, the flag, the counter and the clear size), so calls interleave with the
// library's one-shot all-reduce on the same workspace. Each CTA pushes its 256 columns of the rows to every rank
// (-0.0 marks a word not yet arrived, so a -0.0 value is sent as +0.0), clears its share of the previous call's
// buffer, polls until the four ranks' slices are there, sums them in rank order in FP32 and rounds to BF16, like the
// library's one-shot kernel. The layout and rotation are in namespace lamport below.
// mHC boundary (post-mapping of the previous block, pre-mapping of the next one): every CTA computes the new streams
// of its columns, its partial mixer products (TF32 MMA, the streams are BF16 and exact in TF32) and Gram entries;
// the 16 CTAs exchange their sums through distributed shared memory and each adds them in CTA order
// (deterministic). Each CTA then derives the coefficients (the aux warp runs the Sinkhorn normalization) and writes
// its columns of the streams and of the layer input, RMS-normalized through the Gram matrix (no second reduction).
// The CTA's 96 KB slice of the mixer weights is loaded before the grid-dependency wait.
#include "glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/csrc/mhcArBoundary.h"

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

namespace cg = cooperative_groups;

constexpr int kN = kMhcArBoundaryMult, kMix = kMhcArBoundaryMix, kGram = 10,
              kAcc = kMix + kGram;                                                        // 24 + 10 sums per row
constexpr int kH = kMhcArBoundaryHidden, kNH = kN * kH, kRanks = kMhcArBoundaryRanks;
constexpr int kCtas = 16, kCols = kH / kCtas, kWarps = kCols / 32, kThreads = kCols + 32; // + aux warp
constexpr int kMaxRows = kMhcArBoundaryMaxRows;
constexpr int kChunks = kCols / 8;                        // 16-byte chunks of a row slice
constexpr int kSlice = kCols + 8, kTok = kN * kSlice + 8; // padded (bank-conflict free MMA fragment loads)
constexpr int kFnRow = kN * kCols + 4;                    // fn rows (o, j) at o * kFnRow + j * kCols (floats)
constexpr int kFnBytes = kMix * kFnRow * 4;
constexpr int kResBytes = kMaxRows * kTok * 2;            // this CTA's columns of the residual streams
constexpr int kSmem = kResBytes + kWarps * kMaxRows * kAcc * 4 + kMaxRows * kCols * 2 + kMaxRows * 20 * 4 + kFnBytes;

//! The Lamport state of TensorRT-LLM's all-reduce workspace, which this kernel shares with the library's one-shot
//! all-reduce. The library keeps the layout private: LamportComm in
//! cpp/tensorrt_llm/kernels/communicationKernels/allReduceFusionKernels.cu (and its copy in
//! MiniMaxReduceRMSKernel.cu) reads it, and allocate_allreduce_fusion_workspace in
//! tensorrt_llm/_torch/distributed/allreduce_helper.py builds it. Every value here is that code's; the source
//! test test_modeling_v2_glm53_fp8_lamport_layout.py checks them against that code, so changing one side fails CI.
namespace lamport
{
//! The pointer table: kRanks IPC buffers, kRanks barrier flags, kRanks Lamport buffers, then the two below.
constexpr int kBufferSlot = 2 * kRanks;     //!< ws[kBufferSlot + r]: rank r's Lamport buffers (kNumBuffers of them)
constexpr int kFlagSlot = 3 * kRanks;       //!< ws[kFlagSlot]: int32 words [counter, (non-Lamport flag), flag]
constexpr int kLayoutSlot = 3 * kRanks + 1; //!< ws[kLayoutSlot]: int64 words [clear size, buffer size], in bytes
constexpr int kTableSize = 3 * kRanks + 2;  //!< pointers in the table
constexpr int kCounterWord = 0;             //!< CTAs of the current call that have read the flag
constexpr int kFlagWord = 2;                //!< the call's flag; its receive buffer is flag % kNumBuffers
constexpr int kClearSizeWord = 0;           //!< bytes the previous call wrote, which this call clears
constexpr int kCommSizeWord = 1;            //!< bytes of one receive buffer
constexpr int kNumBuffers = 3;              //!< receive buffers per rank, rotated by the flag

//! The receive buffer of the call with flag `flag`.
__device__ __forceinline__ int dataBuffer(int flag)
{
    return flag % kNumBuffers;
}

//! The receive buffer of the call before the one with flag `flag`, which that call clears.
__device__ __forceinline__ int clearBuffer(int flag)
{
    return (flag + kNumBuffers - 1) % kNumBuffers;
}

//! The flag of the call after the one with flag `flag`.
__device__ __forceinline__ int nextFlag(int flag)
{
    return (flag + 1) % kNumBuffers;
}
} // namespace lamport

static_assert(lamport::kTableSize == kMhcArBoundaryWorkspacePointers, "the workspace pointer table the binding checks");

//! The kernel's arguments (MhcArBoundaryArgs, typed).
struct Params
{
    void** workspace;             //!< TensorRT-LLM all-reduce workspace
    int rank;                     //!< this rank in the group
    __nv_bfloat16 const* partial; //!< [M, H] this rank's partial block output
    __nv_bfloat16 const* res;     //!< [M, N, H]
    float const* postPrev;        //!< [M, N]
    float const* combPrev;        //!< [M, N, N]
    float const* fn;              //!< [24, N * H]
    float const* scale;           //!< [3]
    float const* base;            //!< [24]
    __nv_bfloat16 const* w;       //!< [H]
    __nv_bfloat16* resOut;        //!< [M, N, H]
    float* postOut;               //!< [M, N]
    float* combOut;               //!< [M, N, N]
    __nv_bfloat16* liOut;         //!< [M, H]
    int numRows;                  //!< M
    int skIters;                  //!< Sinkhorn iterations
    float normEps;                //!< the mixer's RMS epsilon
    float hcEps;                  //!< added to pre
    float skEps;                  //!< the Sinkhorn epsilon
    float postMult;               //!< post's multiplier
    float liEps;                  //!< the layer input's RMSNorm epsilon
};

__device__ __forceinline__ constexpr int gramIdx(int j, int k)
{
    return j * kN - j * (j - 1) / 2 + (k - j);
}

__device__ __forceinline__ bool isNegZero(uint32_t v)
{
    return v == 0x80000000u;
}

__device__ __forceinline__ uint4 ldVolatile(uint4 const* p)
{
    uint4 v;
    asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ uint32_t toTf32(float v)
{
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(v));
    return r;
}

// D += A (16 x 8, row) * B (8 x 8, col), TF32 inputs, FP32 accumulation
__device__ __forceinline__ void mmaTf32(
    float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t f32Bits(float v)
{
    return __float_as_uint(v);
}

__device__ __forceinline__ uint32_t smemU32(void const* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__global__ void __cluster_dims__(kCtas, 1, 1) __launch_bounds__(kThreads, 1) mhcArKernel(Params p)
{
    extern __shared__ __align__(128) uint8_t smem[];
    __nv_bfloat16* sRes = reinterpret_cast<__nv_bfloat16*>(smem); // [rows][4][256], then the new streams
    float* sWarp = reinterpret_cast<float*>(smem + kResBytes);    // [warps][rows][34]
    __nv_bfloat16* sX
        = reinterpret_cast<__nv_bfloat16*>(smem + kResBytes + kWarps * kMaxRows * kAcc * 4); // [rows][256]
    float* sPc = reinterpret_cast<float*>(smem + kResBytes + kWarps * kMaxRows * kAcc * 4 + kMaxRows * kCols * 2);
    // sPc: postPrev [rows][4], then combPrev [rows][16]
    float* sFn = sPc + kMaxRows * 20;
    __shared__ float sAll[kCtas * kMaxRows * kAcc]; // every CTA's sums, slot c written by CTA c (DSMEM stores)
    __shared__ float sTot[kMaxRows * kAcc];         // the 16 CTAs' totals
    cg::cluster_group cluster = cg::this_cluster();
    __shared__ float sSk[2][kN + kN * kN];          // this CTA's rows' post and comb (aux warp), stored after the wait
    __shared__ __align__(8) uint64_t inBar, fnBar;

    int const cta = blockIdx.x, t = threadIdx.x, warp = t / 32, lane = t % 32;
    int const c0 = cta * kCols, h = c0 + t, numRows = p.numRows;
    bool const colThread = warp < kWarps, aux = warp == kWarps;
    __shared__ float sCoef[kMaxRows][kN]; // layer_input = sum_j coef[m][j] r_j * w

    // before the PDL wait: weights (this column's 96 mixer weights, kept in registers for every row, and its norm
    // weight) and the all-reduce workspace's pointer table and buffer size, which never change
    // mixer K slice of this warp: stream w / 2, columns [128 (w % 2), + 128), 16 k-steps of 8 (B from sFn)
    int const g = lane >> 2, tig = lane & 3;
    int const mj = warp >> 1, mcol = (warp & 1) * 128;
    float const wh = colThread ? __bfloat162float(p.w[h]) : 0.f;
    void** ws = p.workspace;
    int* const flagWords = static_cast<int*>(ws[lamport::kFlagSlot]);
    long long* const layoutWords = static_cast<long long*>(ws[lamport::kLayoutSlot]);
    int* counterPtr = flagWords + lamport::kCounterWord;
    int* flagPtr = flagWords + lamport::kFlagWord;
    long long* clearPtr = layoutWords + lamport::kClearSizeWord;
    long long const commSize = layoutWords[lamport::kCommSizeWord];
    uint8_t* bufs[kRanks];
#pragma unroll
    for (int r = 0; r < kRanks; r++)
    {
        bufs[r] = static_cast<uint8_t*>(ws[lamport::kBufferSlot + r]);
    }
    if (t == kCols)
    {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smemU32(&inBar)));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smemU32(&fnBar)));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        asm volatile(
            "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smemU32(&fnBar)), "r"(kMix * kN * kCols * 4)
            : "memory");
    }
    __syncthreads();
    // the fn slice (weights) before the wait: 96 rows of 1 KB
    if (aux)
    {
        for (int i = lane; i < kMix * kN; i += 32)
        {
            int const o = i / kN, j = i % kN;
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                    smemU32(sFn + o * kFnRow + j * kCols)),
                "l"(p.fn + static_cast<size_t>(o) * kNH + j * kH + c0), "r"(kCols * 4), "r"(smemU32(&fnBar))
                : "memory");
        }
    }

    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); // this CTA has started
    // the producer of the partial rows (and the previous boundary, the last user of the Lamport state) has completed
    cudaGridDependencySynchronize();

    // aux warp: stage this CTA's columns of the residual streams and the rows' post / comb (overlaps the all-reduce)
    if (aux)
    {
        if (lane == 0)
        {
            uint32_t const bytes = numRows * kN * kCols * 2 + numRows * 20 * 4;
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smemU32(&inBar)), "r"(bytes)
                         : "memory");
        }
        __syncwarp();
        for (int i = lane; i < numRows * kN + 2; i += 32)
        {
            bool const resCopy = i < numRows * kN;
            uint32_t const dst = resCopy ? smemU32(sRes + (i / kN) * kTok + (i % kN) * kSlice)
                                         : smemU32(sPc + (i == numRows * kN ? 0 : kMaxRows * 4));
            void const* src = resCopy
                ? static_cast<void const*>(p.res + static_cast<size_t>(i) * kH + c0)
                : (i == numRows * kN ? static_cast<void const*>(p.postPrev) : static_cast<void const*>(p.combPrev));
            uint32_t const bytes = resCopy ? kCols * 2 : (i == numRows * kN ? numRows * 16 : numRows * 64);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
                "l"(src), "r"(bytes), "r"(smemU32(&inBar))
                : "memory");
        }
    }
    // Lamport state (TensorRT-LLM LamportComm): the receive buffers rotate with the flag, and the previous call's is
    // cleared. The clear size is the bytes that call wrote, whatever kernel or dtype it was; it is cleared in whole
    // 16-byte stores, rounded up.
    int const flag = *flagPtr;
    long long const clearSize = *clearPtr;
    if (t == 0)
    {
        atomicAdd(counterPtr, 1);
    }
    uint8_t* dataBufs[kRanks];
#pragma unroll
    for (int r = 0; r < kRanks; r++)
    {
        dataBufs[r] = bufs[r] + lamport::dataBuffer(flag) * commSize;
    }
    uint4* clearBuf = reinterpret_cast<uint4*>(bufs[p.rank] + lamport::clearBuffer(flag) * commSize);
    int const tot = numRows * kH / 8; // 16-byte accesses in one rank's message

    // push this CTA's columns of the partial rows to every rank (-0.0 words sent as +0.0)
    for (int i = t; i < numRows * kChunks; i += kThreads)
    {
        int const m = i / kChunks, q = i % kChunks, idx = (m * kH + c0) / 8 + q;
        uint4 v = reinterpret_cast<uint4 const*>(p.partial)[idx];
        v.x = isNegZero(v.x) ? 0u : v.x;
        v.y = isNegZero(v.y) ? 0u : v.y;
        v.z = isNegZero(v.z) ? 0u : v.z;
        v.w = isNegZero(v.w) ? 0u : v.w;
#pragma unroll
        for (int r = 0; r < kRanks; r++)
        {
            reinterpret_cast<uint4*>(dataBufs[r])[p.rank * tot + idx] = v;
        }
    }
    // clear this CTA's share of the previous call's buffer
    uint4 const sentinel = make_uint4(0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u);
    for (long long i = static_cast<long long>(cta) * kThreads + t; i < (clearSize + 15) / 16;
         i += static_cast<long long>(kCtas) * kThreads)
    {
        clearBuf[i] = sentinel;
    }
    // poll and sum the four ranks' slices in rank order (FP32), rounded to BF16 like the library's all-reduce
    for (int i = t; i < numRows * kChunks; i += kThreads)
    {
        int const m = i / kChunks, q = i % kChunks, idx = (m * kH + c0) / 8 + q;
        uint4 v[kRanks];
        bool done;
        do
        {
            done = true;
#pragma unroll
            for (int r = 0; r < kRanks; r++)
            {
                v[r] = ldVolatile(reinterpret_cast<uint4 const*>(dataBufs[p.rank]) + r * tot + idx);
                done &= !(isNegZero(v[r].x) || isNegZero(v[r].y) || isNegZero(v[r].z) || isNegZero(v[r].w));
            }
        } while (!done);
        float acc[8];
#pragma unroll
        for (int e = 0; e < 8; e++)
        {
            acc[e] = 0.f;
        }
#pragma unroll
        for (int r = 0; r < kRanks; r++)
        {
            __nv_bfloat16 const* bv = reinterpret_cast<__nv_bfloat16 const*>(&v[r]);
#pragma unroll
            for (int e = 0; e < 8; e++)
            {
                acc[e] += __bfloat162float(bv[e]);
            }
        }
        uint4 o;
        __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&o);
#pragma unroll
        for (int e = 0; e < 8; e++)
        {
            ob[e] = __float2bfloat16(acc[e]);
        }
        reinterpret_cast<uint4*>(sX + m * kCols)[q] = o;
    }
    asm volatile(
        "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n @!p bra WAIT_%=;\n}" ::
            "r"(smemU32(&inBar))
        : "memory");
    __syncthreads();

    // post-mapping of column h: the new streams, in place; four rows per group, the group's inputs loaded first
    float4 const* sPost4 = reinterpret_cast<float4 const*>(sPc);
    float4 const* sComb4 = reinterpret_cast<float4 const*>(sPc + kMaxRows * 4);
    if (colThread)
    {
        for (int m0 = 0; m0 < numRows; m0 += 4)
        {
            float xv[4], rp[4][kN];
            float4 po[4], cm[4][kN];
#pragma unroll
            for (int k = 0; k < 4; k++)
            {
                int const m = min(m0 + k, numRows - 1);
                xv[k] = __bfloat162float(sX[m * kCols + t]);
#pragma unroll
                for (int i = 0; i < kN; i++)
                {
                    rp[k][i] = __bfloat162float(sRes[m * kTok + i * kSlice + t]);
                }
                po[k] = sPost4[m];
#pragma unroll
                for (int i = 0; i < kN; i++)
                {
                    cm[k][i] = sComb4[m * kN + i];
                }
            }
#pragma unroll
            for (int k = 0; k < 4; k++)
            {
                int const m = m0 + k;
                if (m < numRows)
                {
                    float const pj[kN] = {po[k].x, po[k].y, po[k].z, po[k].w};
#pragma unroll
                    for (int j = 0; j < kN; j++)
                    {
                        float a = pj[j] * xv[k];
#pragma unroll
                        for (int i = 0; i < kN; i++)
                        {
                            float const c = j == 0 ? cm[k][i].x
                                : j == 1           ? cm[k][i].y
                                : j == 2           ? cm[k][i].z
                                                   : cm[k][i].w;
                            a += c * rp[k][i];
                        }
                        __nv_bfloat16 const bb = __float2bfloat16(a);
                        sRes[m * kTok + j * kSlice + t] = bb;
                    }
                }
            }
        }
    }
    __syncthreads();
    // this warp's partial mixer outputs (K slice above) and Gram entries (columns [32 w, 32 w + 32)) on the
    // tensor cores; A operands from the new streams (BF16, exact in TF32)
    asm volatile(
        "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n @!p bra WAIT_%=;\n}" ::
            "r"(smemU32(&fnBar))
        : "memory");
    if (colThread)
    {
        auto rv = [&](int m, int j, int col) -> uint32_t
        { return m < numRows ? f32Bits(__bfloat162float(sRes[m * kTok + j * kSlice + col])) : 0u; };
        for (int mt = 0; mt * 16 < numRows; mt++)
        {
            float acc[3][4] = {};
            int const r0 = 16 * mt + g, r1 = r0 + 8;
#pragma unroll
            for (int st = 0; st < 16; st++)
            {
                int const col = mcol + 8 * st + tig;
                uint32_t const a0 = rv(r0, mj, col), a1 = rv(r1, mj, col), a2 = rv(r0, mj, col + 4),
                               a3 = rv(r1, mj, col + 4);
#pragma unroll
                for (int nt = 0; nt < 3; nt++)
                {
                    float const* fb = sFn + (nt * 8 + g) * kFnRow + mj * kCols + col;
                    mmaTf32(acc[nt], a0, a1, a2, a3, toTf32(fb[0]), toTf32(fb[4]));
                }
            }
#pragma unroll
            for (int nt = 0; nt < 3; nt++)
            {
                int const n = nt * 8 + 2 * tig;
                if (r0 < numRows)
                {
                    sWarp[(warp * kMaxRows + r0) * kAcc + n] = acc[nt][0];
                    sWarp[(warp * kMaxRows + r0) * kAcc + n + 1] = acc[nt][1];
                }
                if (r1 < numRows)
                {
                    sWarp[(warp * kMaxRows + r1) * kAcc + n] = acc[nt][2];
                    sWarp[(warp * kMaxRows + r1) * kAcc + n + 1] = acc[nt][3];
                }
            }
        }
        // Gram: rows (token 4 mg + r / 4, stream r % 4) x columns (token 4 mg + 2 q + n / 4, stream n % 4)
        for (int mg = 0; mg * 4 < numRows; mg++)
        {
            float ga[2][4] = {};
            int const ta0 = 4 * mg + (g >> 2), ta1 = ta0 + 2, sa = g & 3;
#pragma unroll
            for (int st = 0; st < 4; st++)
            {
                int const col = 32 * warp + 8 * st + tig;
                uint32_t const a0 = rv(ta0, sa, col), a1 = rv(ta1, sa, col), a2 = rv(ta0, sa, col + 4),
                               a3 = rv(ta1, sa, col + 4);
#pragma unroll
                for (int q = 0; q < 2; q++)
                {
                    int const tb = 4 * mg + 2 * q + (g >> 2);
                    mmaTf32(ga[q], a0, a1, a2, a3, rv(tb, sa, col), rv(tb, sa, col + 4));
                }
            }
#pragma unroll
            for (int q = 0; q < 2; q++)
            {
                int const tb = 2 * q + (tig >> 1), kb0 = 2 * (tig & 1);
#pragma unroll
                for (int c = 0; c < 4; c++)
                {
                    int const ta = (c < 2 ? 0 : 2) + (g >> 2), kb = kb0 + (c & 1), token = 4 * mg + ta;
                    if (ta == tb && sa <= kb && token < numRows)
                    {
                        sWarp[(warp * kMaxRows + token) * kAcc + kMix + gramIdx(sa, kb)] = ga[q][c];
                    }
                }
            }
        }
    }
    __syncthreads();
    // this CTA's sums into slot `cta` of every CTA's s_all, then each CTA adds the 16 slots in CTA order
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); // every CTA of the cluster has started
    for (int i = t; i < numRows * kAcc; i += kThreads)
    {
        float s3 = 0.f;
#pragma unroll
        for (int w8 = 0; w8 < kWarps; w8++)
        {
            s3 += sWarp[(w8 * kMaxRows) * kAcc + i];
        }
#pragma unroll
        for (int c = 0; c < kCtas; c++)
        {
            cluster.map_shared_rank(sAll, c)[cta * kMaxRows * kAcc + i] = s3;
        }
    }
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    for (int i = t; i < numRows * kAcc; i += kThreads)
    {
        float s3 = 0.f;
#pragma unroll
        for (int c = 0; c < kCtas; c++)
        {
            s3 += sAll[c * kMaxRows * kAcc + i];
        }
        sTot[i] = s3;
    }
    __syncthreads();
    // per row: the collapse coefficients (pre_j and the RMSNorm scale of the collapsed row from the Gram matrix)
    if (warp == 0 && lane < numRows)
    {
        float const* a = sTot + lane * kAcc;
        float gram[kN][kN];
#pragma unroll
        for (int j = 0; j < kN; j++)
        {
#pragma unroll
            for (int i = j; i < kN; i++)
            {
                float const entry = a[kMix + gramIdx(j, i)];
                gram[j][i] = entry;
                gram[i][j] = entry;
            }
        }
        float const rs
            = rsqrtf((gram[0][0] + gram[1][1] + gram[2][2] + gram[3][3]) / static_cast<float>(kNH) + p.normEps);
        float pre[kN];
#pragma unroll
        for (int j = 0; j < kN; j++)
        {
            pre[j] = 1.f / (1.f + __expf(-(a[j] * rs * p.scale[0] + p.base[j]))) + p.hcEps;
        }
        float q = 0.f;
#pragma unroll
        for (int j = 0; j < kN; j++)
        {
#pragma unroll
            for (int i = 0; i < kN; i++)
            {
                q += pre[j] * pre[i] * gram[j][i];
            }
        }
        float const sc = rsqrtf(q / static_cast<float>(kH) + p.liEps);
#pragma unroll
        for (int j = 0; j < kN; j++)
        {
            sCoef[lane][j] = pre[j] * sc;
        }
    }
    __syncthreads();
    if (colThread)
    {
        // layer_input of column h for every row
#pragma unroll
        for (int m = 0; m < kMaxRows; m++)
        {
            if (m < numRows)
            {
                float y = 0.f;
#pragma unroll
                for (int j = 0; j < kN; j++)
                {
                    y += sCoef[m][j] * __bfloat162float(sRes[m * kTok + j * kSlice + t]);
                }
                sX[m * kCols + t] = __float2bfloat16(y * wh); // s_x (the all-reduced rows) is free again
            }
        }
    }
    else if (lane < (kMaxRows + kCtas - 1) / kCtas && cta + kCtas * lane < numRows)
    {
        // aux warp: post and the Sinkhorn-normalized comb of the rows m with m % 16 == cta, one lane each
        int const m = cta + kCtas * lane;
        float const* a = sTot + m * kAcc;
        float d[kN];
#pragma unroll
        for (int j = 0; j < kN; j++)
        {
            d[j] = a[kMix + gramIdx(j, j)];
        }
        float const rs = rsqrtf((d[0] + d[1] + d[2] + d[3]) / static_cast<float>(kNH) + p.normEps);
#pragma unroll
        for (int j = 0; j < kN; j++)
        {
            sSk[lane][j] = p.postMult / (1.f + __expf(-(a[kN + j] * rs * p.scale[1] + p.base[kN + j])));
        }
        float c[kN][kN];
#pragma unroll
        for (int i = 0; i < kN; i++)
        {
            float mx = -3.4e38f;
#pragma unroll
            for (int j = 0; j < kN; j++)
            {
                c[i][j] = a[2 * kN + i * kN + j] * rs * p.scale[2] + p.base[2 * kN + i * kN + j];
                mx = fmaxf(mx, c[i][j]);
            }
            float s3 = 0.f;
#pragma unroll
            for (int j = 0; j < kN; j++)
            {
                c[i][j] = __expf(c[i][j] - mx);
                s3 += c[i][j];
            }
            float const inv = 1.f / s3;
#pragma unroll
            for (int j = 0; j < kN; j++)
            {
                c[i][j] = c[i][j] * inv + p.skEps;
            }
        }
        for (int it = 0; it < p.skIters; it++)
        {
            if (it > 0)
            {
#pragma unroll
                for (int i = 0; i < kN; i++)
                {
                    float const inv = 1.f / (c[i][0] + c[i][1] + c[i][2] + c[i][3] + p.skEps);
#pragma unroll
                    for (int j = 0; j < kN; j++)
                    {
                        c[i][j] *= inv;
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < kN; j++)
            {
                float const inv = 1.f / (c[0][j] + c[1][j] + c[2][j] + c[3][j] + p.skEps);
#pragma unroll
                for (int i = 0; i < kN; i++)
                {
                    c[i][j] *= inv;
                }
            }
        }
#pragma unroll
        for (int i = 0; i < kN; i++)
        {
#pragma unroll
            for (int j = 0; j < kN; j++)
            {
                sSk[lane][kN + i * kN + j] = c[i][j];
            }
        }
    }
    __syncthreads();
    // 16-byte stores: this CTA's 256 columns of every residual stream row and layer_input row
    for (int i = t; i < numRows * (kN + 1) * kChunks; i += kThreads)
    {
        int const sl = i / kChunks, q = i % kChunks, m = sl / (kN + 1), j = sl % (kN + 1);
        if (j < kN)
        {
            reinterpret_cast<uint4*>(p.resOut + (static_cast<size_t>(m) * kN + j) * kH + c0)[q]
                = reinterpret_cast<uint4 const*>(sRes + m * kTok + j * kSlice)[q];
        }
        else
        {
            reinterpret_cast<uint4*>(p.liOut + static_cast<size_t>(m) * kH + c0)[q]
                = reinterpret_cast<uint4 const*>(sX + m * kCols)[q];
        }
    }
    if (!colThread)
    {
        // aux warp: its rows' post and comb
        if (lane < (kMaxRows + kCtas - 1) / kCtas && cta + kCtas * lane < numRows)
        {
            int const m = cta + kCtas * lane;
#pragma unroll
            for (int j = 0; j < kN; j++)
            {
                p.postOut[m * kN + j] = sSk[lane][j];
            }
#pragma unroll
            for (int i = 0; i < kN * kN; i++)
            {
                p.combOut[m * kN * kN + i] = sSk[lane][kN + i];
            }
        }
    }
    cudaTriggerProgrammaticLaunchCompletion(); // after the outputs: the next kernel's CTAs do not crowd them
    // Lamport rotation (LamportComm::update): once every CTA has counted in
    if (cta == 0 && t == 0)
    {
        while (*static_cast<int volatile*>(counterPtr) != kCtas)
        {
        }
        *flagPtr = lamport::nextFlag(flag);
        // bytes, as LamportComm
        *clearPtr = static_cast<long long>(numRows) * kH * kRanks * static_cast<long long>(sizeof(__nv_bfloat16));
        *counterPtr = 0;
    }
}

} // namespace

cudaError_t mhcArBoundary(MhcArBoundaryArgs const& args, cudaStream_t stream)
{
    if (args.rows < 1 || args.rows > kMaxRows)
    {
        return cudaErrorInvalidValue;
    }
    constexpr int kMaxDevices = 64;
    static bool attributesSet[kMaxDevices] = {};
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
    if (!attributesSet[dev])
    {
        err = cudaFuncSetAttribute(mhcArKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
        if (err == cudaSuccess)
        {
            err = cudaFuncSetAttribute(mhcArKernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
        }
        if (err != cudaSuccess)
        {
            return err;
        }
        attributesSet[dev] = true;
    }
    Params p{args.workspace, args.rank, static_cast<__nv_bfloat16 const*>(args.partial),
        static_cast<__nv_bfloat16 const*>(args.residual), args.postPrev, args.combPrev, args.fn, args.scale, args.base,
        static_cast<__nv_bfloat16 const*>(args.normWeight), static_cast<__nv_bfloat16*>(args.residualOut), args.postOut,
        args.combOut, static_cast<__nv_bfloat16*>(args.layerInput), args.rows, args.sinkhornIters, args.normEps,
        args.hcEps, args.sinkhornEps, args.postMult, args.layerInputEps};
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(kCtas);
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = kSmem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = args.pdl ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, mhcArKernel, p);
}

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
