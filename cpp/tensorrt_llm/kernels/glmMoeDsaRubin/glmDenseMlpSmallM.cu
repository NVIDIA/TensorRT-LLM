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

// Weight-streaming BF16 kernels for the dense (non-MoE) MLP at decode row counts (M <= 8): the gate/up projection
// with SwiGLU, and the down projection. The compiled tilings serve hidden 6144 with a rank-local intermediate size of
// 768, the dense layers 0-2 of GLM-5.2 at TP16; the result is the rank-local partial that the post-MLP all-reduce sums.
//
// Each CTA owns 16-row weight tiles and its warps split the contraction. For every 32-wide contraction step a thread
// takes 16 bytes of two weight rows per tile and of one activation row and feeds them to two m16n8k16 MMAs, the
// weight on the MMA M side and the (at most 8) activation rows on the N side. The contraction index of an MMA k-slot
// depends only on the thread's position inside its quad, so the weight and activation fragments use the same
// permutation of k: the result is the same sum in another FP32 order. The gate/up kernel holds the gate rows and the
// up rows of the same output columns, and the thread that finishes a column applies SwiGLU in registers.
//
// Weights never depend on the preceding kernel: before griddepcontrol.wait each CTA copies all of its weights into
// shared memory with one bulk copy per weight row. After the wait each thread loads its activations from L2 and runs
// its MMAs; the warp sums are added in warp order in shared memory. With kClusterK > 1 the CTAs of a cluster split
// the contraction: each sends its sums to the first CTA of the cluster with st.async, which adds them in rank order.
// The results do not depend on timing.
//
// Programmatic dependent launch: activations are read only after griddepcontrol.wait; the dependents are released at
// entry or after the weight copies are issued (Trigger). Each grid fits in one wave of resident CTAs.

#include "tensorrt_llm/kernels/glmMoeDsaRubin/glmDenseMlpSmallM.h"

#include <cstdint>
#include <tuple>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa::dense_mlp_small_m
{

namespace
{

using Bf16 = __nv_bfloat16;

constexpr int kWarpSize = 32;
//! Weight rows of one MMA tile (MMA M).
constexpr int kTileRows = 16;
//! BF16 values per 16-byte vector load.
constexpr int kVecElems = 8;
//! Threads per MMA quad; a quad covers kQuadThreads * kVecElems contraction elements per step.
constexpr int kQuadThreads = 4;
//! Contraction elements consumed by one warp step (two m16n8k16 MMAs per row tile).
constexpr int kKPerStep = kQuadThreads * kVecElems;
constexpr int kStepBytes = kKPerStep * static_cast<int>(sizeof(Bf16));

//! When the dependents of a grid are released.
enum class Trigger : int
{
    kAtEntry,    //!< first instruction
    kAfterIssue, //!< after the weight copies are issued, before griddepcontrol.wait
};

//! Compile-time tiling of one kernel instance.
//! \tparam kGateUp gate/up + SwiGLU (true) or down projection (false).
//! \tparam kRowTiles 16-row weight tiles per CTA. gate/up: one tile holds 8 gate and the matching 8 up rows; with
//!         more tiles, the first half are gate rows and the second half the up rows of the same columns.
//! \tparam kWarps warps per CTA; each warp owns a contiguous slice of the CTA's contraction.
//! \tparam kSteps 32-wide contraction steps per warp.
//! \tparam kClusterK CTAs of a cluster that split the contraction.
template <bool kGateUp_, int kRowTiles_, int kWarps_, int kSteps_, int kClusterK_, Trigger kTrigger_>
struct Tiling
{
    static constexpr bool kGateUp = kGateUp_;
    static constexpr int kRowTiles = kRowTiles_;
    static constexpr int kWarps = kWarps_;
    static constexpr int kSteps = kSteps_;
    static constexpr int kClusterK = kClusterK_;
    static constexpr Trigger kTrigger = kTrigger_;

    static constexpr int kThreads = kWarps * kWarpSize;
    static constexpr int kRows = kRowTiles * kTileRows;
    static constexpr int kKPerWarp = kSteps * kKPerStep;
    static constexpr int kKPerCta = kWarps * kKPerWarp;
    //! The contraction size this tiling serves.
    static constexpr int kKDim = kKPerCta * kClusterK;
    //! Output columns of a CTA (gate/up: of h, down: of out).
    static constexpr int kOutCols = kGateUp ? kRows / 2 : kRows;
    //! (column, token) values the cluster finishes; each holds kEpiVals sums (gate and up, or one).
    static constexpr int kEpi = kOutCols * kMaxRows;
    static constexpr int kEpiVals = kGateUp ? 2 : 1;

    //! Weight stage [row][kKPerCta]: the bytes of one row copy, rows padded to 64 mod 128 bytes so that the 16-byte
    //! fragment loads of a quarter warp (two rows, four quads) hit distinct banks.
    static constexpr int kRowBytes = kKPerCta * static_cast<int>(sizeof(Bf16));
    static constexpr int kRowStride = kRowBytes % 128 == 0 ? kRowBytes + 64 : kRowBytes;

    // Shared-memory layout (bytes): warp sums | sums received by the cluster leader | weight stage | barriers.
    static constexpr int kRedBytes = kWarps * kRowTiles * 4 * kWarpSize * static_cast<int>(sizeof(float));
    static constexpr int kRecvBytes = (kClusterK - 1) * kEpi * kEpiVals * static_cast<int>(sizeof(float));
    static constexpr int kStageOffset = (kRedBytes + kRecvBytes + 127) / 128 * 128;
    static constexpr int kBarOffset = kStageOffset + kRows * kRowStride;
    static constexpr int kSmemBytes = kBarOffset + 2 * static_cast<int>(sizeof(uint64_t));

    static_assert(!kGateUp_ || kRowTiles_ == 1 || kRowTiles_ % 2 == 0, "gate/up needs one or an even number of tiles");
    static_assert(kRows <= kThreads, "one thread issues the copy of each weight row");
    static_assert(kRowBytes % 64 == 0, "rows are whole contraction steps");
};

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

__device__ __forceinline__ void pdlWait()
{
    asm volatile("griddepcontrol.wait;" ::: "memory");
}

__device__ __forceinline__ void pdlLaunchDependents()
{
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

//! Activation load from L2 (the preceding kernel wrote it; L1 is bypassed).
__device__ __forceinline__ uint4 loadActivation(Bf16 const* ptr)
{
    uint4 r;
    asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "l"(ptr)
                 : "memory");
    return r;
}

__device__ __forceinline__ uint32_t smemAddress(void const* ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

//! One arrival (this one) plus expectBytes of transactions.
__device__ __forceinline__ void mbarrierInit(uint32_t bar, uint32_t expectBytes)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bar) : "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(expectBytes) : "memory");
}

__device__ __forceinline__ void mbarrierWait(uint32_t bar)
{
    asm volatile(
        "{\n"
        ".reg .pred done;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], 0;\n"
        "@!done bra WAIT_%=;\n"
        "}\n" ::"r"(bar)
        : "memory");
}

__device__ __forceinline__ void mbarrierWaitCluster(uint32_t bar)
{
    asm volatile(
        "{\n"
        ".reg .pred done;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 done, [%0], 0;\n"
        "@!done bra WAIT_%=;\n"
        "}\n" ::"r"(bar)
        : "memory");
}

__device__ __forceinline__ void bulkCopy(uint32_t dst, void const* src, uint32_t bytes, uint32_t bar)
{
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
                 "l"(src), "r"(bytes), "r"(bar)
                 : "memory");
}

__device__ __forceinline__ uint32_t clusterCtaRank()
{
    uint32_t rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    return rank;
}

__device__ __forceinline__ uint32_t mapToCta(uint32_t addr, uint32_t rank)
{
    uint32_t peer;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(peer) : "r"(addr), "r"(rank));
    return peer;
}

__device__ __forceinline__ void mmaBf16(
    float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

//! Two MMAs of one step and row tile: MMA A takes contraction elements 0..3 of the thread's 8-element vectors, MMA B
//! elements 4..7. Fragments: a0/a2 = weight row "group", a1/a3 = weight row "group + 8", b0/b1 = activation row
//! "group".
__device__ __forceinline__ void mmaStep(
    float (&accA)[4], float (&accB)[4], uint4 const& wLo, uint4 const& wHi, uint4 const& xv)
{
    mmaBf16(accA, wLo.x, wHi.x, wLo.y, wHi.y, xv.x, xv.y);
    mmaBf16(accB, wLo.z, wHi.z, wLo.w, wHi.w, xv.z, xv.w);
}

__device__ __forceinline__ float roundToBf16(float v)
{
    return __bfloat162float(__float2bfloat16_rn(v));
}

//! SwiGLU of BF16-rounded gate and up sums, as the BF16 GEMM output feeding silu_and_mul: the sigmoid takes the fast
//! exp and division of that kernel.
__device__ __forceinline__ float swiglu(float gateSum, float upSum)
{
    float const gate = roundToBf16(gateSum);
    float const up = roundToBf16(upSum);
    float const sig = __fdividef(1.F, 1.F + __expf(-gate));
    return gate * sig * up;
}

//! Weight row (of the [*, kDim] weight matrix) of row r of row tile t of the CTA of output tile `tile`.
template <typename T>
__device__ __forceinline__ int weightRow(int tile, int t, int r, int inter)
{
    if constexpr (T::kGateUp)
    {
        if constexpr (T::kRowTiles == 1)
        {
            int const col = tile * T::kOutCols + (r & 7);
            return r < 8 ? col : inter + col;
        }
        else
        {
            constexpr int kHalf = T::kRowTiles / 2;
            int const col = tile * T::kOutCols + (t % kHalf) * kTileRows + r;
            return t < kHalf ? col : inter + col;
        }
    }
    else
    {
        return tile * T::kRows + t * kTileRows + r;
    }
}

#endif // __CUDA_ARCH__ >= 900

//! x [numRows, kDim], weight [*, kDim]; out: gate/up h [numRows, outDim] (outDim = inter), down out [numRows, outDim].
template <typename T>
__global__ void __launch_bounds__(T::kThreads, 1) denseMlpSmallMKernel(
    Bf16 const* __restrict__ x, Bf16 const* __restrict__ weight, Bf16* __restrict__ out, int numRows, int outDim)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    constexpr int kDim = T::kKDim;
    extern __shared__ __align__(128) char smem[];
    float* red = reinterpret_cast<float*>(smem);
    [[maybe_unused]] float* recv = reinterpret_cast<float*>(smem + T::kRedBytes);
    char* stage = smem + T::kStageOffset;
    uint32_t const stageBar = smemAddress(smem + T::kBarOffset);
    [[maybe_unused]] uint32_t const clusterBar = stageBar + static_cast<uint32_t>(sizeof(uint64_t));

    if constexpr (T::kTrigger == Trigger::kAtEntry)
    {
        pdlLaunchDependents();
    }

    int const lane = threadIdx.x % kWarpSize;
    int const warp = threadIdx.x / kWarpSize;
    int const group = lane / kQuadThreads;
    int const tig = lane % kQuadThreads;
    int const rank = T::kClusterK > 1 ? static_cast<int>(clusterCtaRank()) : 0;
    int const tile = blockIdx.x / T::kClusterK;
    int const kCta = rank * T::kKPerCta;

    // The weight stage expects all row bytes; the cluster leader also expects the sums of the other CTAs.
    if (threadIdx.x == 0)
    {
        mbarrierInit(stageBar, static_cast<uint32_t>(T::kRows * T::kRowBytes));
        if constexpr (T::kClusterK > 1)
        {
            if (rank == 0)
            {
                mbarrierInit(clusterBar, static_cast<uint32_t>(T::kRecvBytes));
            }
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if constexpr (T::kClusterK > 1)
    {
        // Publishes the leader's barrier; every CTA waits for this phase before it sends.
        asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    }

    // Thread r copies weight row r of the CTA (its contraction slice).
    int const copyRow = static_cast<int>(threadIdx.x);
    if (copyRow < T::kRows)
    {
        int const wRow = weightRow<T>(tile, copyRow / kTileRows, copyRow % kTileRows, outDim);
        bulkCopy(smemAddress(stage + copyRow * T::kRowStride), weight + static_cast<int64_t>(wRow) * kDim + kCta,
            static_cast<uint32_t>(T::kRowBytes), stageBar);
    }

    if constexpr (T::kTrigger == Trigger::kAfterIssue)
    {
        pdlLaunchDependents();
    }
    pdlWait();

    // Activations: row "group" (zero beyond numRows), the thread's 8 contraction elements of every step.
    int const kWarp = kCta + warp * T::kKPerWarp;
    bool const hasRow = group < numRows;
    Bf16 const* xRow = x + static_cast<int64_t>(hasRow ? group : 0) * kDim + kWarp + tig * kVecElems;
    uint4 xv[T::kSteps];
#pragma unroll
    for (int s = 0; s < T::kSteps; ++s)
    {
        xv[s] = hasRow ? loadActivation(xRow + s * kKPerStep) : make_uint4(0U, 0U, 0U, 0U);
    }

    float accA[T::kRowTiles][4];
    float accB[T::kRowTiles][4];
#pragma unroll
    for (int t = 0; t < T::kRowTiles; ++t)
    {
#pragma unroll
        for (int e = 0; e < 4; ++e)
        {
            accA[t][e] = 0.F;
            accB[t][e] = 0.F;
        }
    }
    mbarrierWait(stageBar);
    char const* seg = stage + warp * T::kKPerWarp * static_cast<int>(sizeof(Bf16)) + tig * 16;
#pragma unroll
    for (int s = 0; s < T::kSteps; ++s)
    {
#pragma unroll
        for (int t = 0; t < T::kRowTiles; ++t)
        {
            int const rowLo = t * kTileRows + group;
            uint4 const wLo = *reinterpret_cast<uint4 const*>(seg + rowLo * T::kRowStride + s * kStepBytes);
            uint4 const wHi = *reinterpret_cast<uint4 const*>(seg + (rowLo + 8) * T::kRowStride + s * kStepBytes);
            mmaStep(accA[t], accB[t], wLo, wHi, xv[s]);
        }
    }

    // Warp sums -> shared memory [warp][tile * 4 + element][lane].
#pragma unroll
    for (int t = 0; t < T::kRowTiles; ++t)
    {
#pragma unroll
        for (int e = 0; e < 4; ++e)
        {
            red[(warp * T::kRowTiles * 4 + t * 4 + e) * kWarpSize + lane] = accA[t][e] + accB[t][e];
        }
    }
    __syncthreads();
    if constexpr (T::kClusterK > 1)
    {
        asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    }

    // Finish value j: element (tile t, accumulator e, lane l) holds weight row t * 16 + l / 4 + (e / 2) * 8 of the
    // CTA and token 2 * (l % 4) + e % 2.
    for (int j = static_cast<int>(threadIdx.x); j < T::kEpi; j += T::kThreads)
    {
        int q0;
        [[maybe_unused]] int q1 = 0;
        int col;
        int token;
        int const l = j % kWarpSize;
        if constexpr (T::kGateUp && T::kRowTiles == 1)
        {
            int const e = j / kWarpSize; // 0 or 1: rows l / 4 (gate) and l / 4 + 8 (up)
            q0 = e;
            q1 = e + 2;
            col = l / kQuadThreads;
            token = 2 * (l % kQuadThreads) + e;
        }
        else if constexpr (T::kGateUp)
        {
            int const t = j / (4 * kWarpSize);
            int const e = (j / kWarpSize) % 4;
            q0 = t * 4 + e;
            q1 = (t + T::kRowTiles / 2) * 4 + e;
            col = t * kTileRows + l / kQuadThreads + (e / 2) * 8;
            token = 2 * (l % kQuadThreads) + e % 2;
        }
        else
        {
            int const q = j / kWarpSize;
            q0 = q;
            col = (q / 4) * kTileRows + l / kQuadThreads + ((q % 4) / 2) * 8;
            token = 2 * (l % kQuadThreads) + q % 2;
        }
        // Fixed order over warps keeps the result deterministic.
        float v0 = 0.F;
        [[maybe_unused]] float v1 = 0.F;
#pragma unroll
        for (int w = 0; w < T::kWarps; ++w)
        {
            v0 += red[(w * T::kRowTiles * 4 + q0) * kWarpSize + l];
            if constexpr (T::kGateUp)
            {
                v1 += red[(w * T::kRowTiles * 4 + q1) * kWarpSize + l];
            }
        }
        if constexpr (T::kClusterK > 1)
        {
            if (rank != 0)
            {
                uint32_t const slot = smemAddress(recv + ((rank - 1) * T::kEpi + j) * T::kEpiVals);
                uint32_t const peerSlot = mapToCta(slot, 0);
                uint32_t const peerBar = mapToCta(clusterBar, 0);
                if constexpr (T::kGateUp)
                {
                    asm volatile(
                        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];" ::"r"(
                            peerSlot),
                        "r"(__float_as_uint(v0)), "r"(__float_as_uint(v1)), "r"(peerBar)
                        : "memory");
                }
                else
                {
                    asm volatile(
                        "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(peerSlot),
                        "r"(__float_as_uint(v0)), "r"(peerBar)
                        : "memory");
                }
                continue;
            }
            mbarrierWaitCluster(clusterBar);
            // Rank order.
#pragma unroll
            for (int r = 1; r < T::kClusterK; ++r)
            {
                v0 += recv[((r - 1) * T::kEpi + j) * T::kEpiVals];
                if constexpr (T::kGateUp)
                {
                    v1 += recv[((r - 1) * T::kEpi + j) * T::kEpiVals + 1];
                }
            }
        }
        if (token < numRows)
        {
            int64_t const idx = static_cast<int64_t>(token) * outDim + tile * T::kOutCols + col;
            if constexpr (T::kGateUp)
            {
                out[idx] = __float2bfloat16_rn(swiglu(v0, v1));
            }
            else
            {
                out[idx] = __float2bfloat16_rn(v0);
            }
        }
    }
#endif
}

template <typename T>
cudaError_t launchTiling(
    Bf16 const* in, Bf16 const* weight, Bf16* out, int numRows, int outDim, bool enablePdl, cudaStream_t stream)
{
    auto const kernel = &denseMlpSmallMKernel<T>;
    if constexpr (T::kSmemBytes > 48 * 1024)
    {
        static cudaError_t const attrStatus
            = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, T::kSmemBytes);
        if (attrStatus != cudaSuccess)
        {
            return attrStatus;
        }
    }
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(outDim / T::kOutCols * T::kClusterK);
    config.blockDim = dim3(T::kThreads);
    config.dynamicSmemBytes = T::kSmemBytes;
    config.stream = stream;
    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = enablePdl ? 1 : 0;
    attrs[1].id = cudaLaunchAttributeClusterDimension;
    attrs[1].val.clusterDim.x = T::kClusterK;
    attrs[1].val.clusterDim.y = 1;
    attrs[1].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = T::kClusterK > 1 ? 2 : 1;
    return cudaLaunchKernelEx(&config, kernel, in, weight, out, numRows, outDim);
}

// Tilings: row tiles, warps, steps, cluster size, trigger.
using GateUpTilings = std::tuple<
    // Hidden 6144 (GLM-5.2): 2-CTA clusters split the contraction, 16 warps x 6 steps, 8 columns per cluster; at TP16
    // (inter 768) 192 CTAs with 96 KiB of weights each. The down kernel is released at entry, so its weights are staged
    // during the preceding all-reduce as well.
    Tiling<true, 1, 16, 6, 2, Trigger::kAtEntry>>;

using DownTilings = std::tuple<
    // Inter 768 (GLM-5.2 at TP16): 64 rows per CTA, 8 warps x 3 steps; at hidden 6144 96 CTAs with 96 KiB of weights
    // each.
    Tiling<false, 4, 8, 3, 1, Trigger::kAfterIssue>>;

template <typename Tilings>
constexpr int kNumTilings = static_cast<int>(std::tuple_size_v<Tilings>);

template <typename Tilings, int I = 0>
bool tilingSupported(int tiling, int numRows, int kDim, int outDim)
{
    if constexpr (I < kNumTilings<Tilings>)
    {
        if (tiling == I)
        {
            using T = std::tuple_element_t<I, Tilings>;
            return numRows >= 1 && numRows <= kMaxRows && kDim == T::kKDim && outDim > 0 && outDim % T::kOutCols == 0;
        }
        return tilingSupported<Tilings, I + 1>(tiling, numRows, kDim, outDim);
    }
    else
    {
        return false;
    }
}

template <typename Tilings, int I = 0>
cudaError_t launchTilingIndex(int tiling, Bf16 const* in, Bf16 const* weight, Bf16* out, int numRows, int outDim,
    bool enablePdl, cudaStream_t stream)
{
    if constexpr (I < kNumTilings<Tilings>)
    {
        if (tiling == I)
        {
            return launchTiling<std::tuple_element_t<I, Tilings>>(in, weight, out, numRows, outDim, enablePdl, stream);
        }
        return launchTilingIndex<Tilings, I + 1>(tiling, in, weight, out, numRows, outDim, enablePdl, stream);
    }
    else
    {
        return cudaErrorInvalidValue;
    }
}

//! The tiling a call runs: the requested one if it serves the shape, for kDefaultTiling the first entry that does;
//! -1 if none.
template <typename Tilings>
int resolveTiling(int tiling, int numRows, int kDim, int outDim)
{
    if (tiling != kDefaultTiling)
    {
        return tilingSupported<Tilings>(tiling, numRows, kDim, outDim) ? tiling : -1;
    }
    for (int i = 0; i < kNumTilings<Tilings>; ++i)
    {
        if (tilingSupported<Tilings>(i, numRows, kDim, outDim))
        {
            return i;
        }
    }
    return -1;
}

} // namespace

bool isGateUpSupported(int numRows, int hidden, int inter, int tiling)
{
    return resolveTiling<GateUpTilings>(tiling, numRows, hidden, inter) >= 0;
}

bool isDownSupported(int numRows, int inter, int hidden, int tiling)
{
    return resolveTiling<DownTilings>(tiling, numRows, inter, hidden) >= 0;
}

cudaError_t launchGateUpSwiglu(Bf16 const* x, Bf16 const* gateUpWeight, Bf16* h, int numRows, int hidden, int inter,
    int tiling, bool enablePdl, cudaStream_t stream)
{
    int const resolved = resolveTiling<GateUpTilings>(tiling, numRows, hidden, inter);
    if (resolved < 0)
    {
        return cudaErrorInvalidValue;
    }
    return launchTilingIndex<GateUpTilings>(resolved, x, gateUpWeight, h, numRows, inter, enablePdl, stream);
}

cudaError_t launchDown(Bf16 const* h, Bf16 const* downWeight, Bf16* out, int numRows, int inter, int hidden, int tiling,
    bool enablePdl, cudaStream_t stream)
{
    int const resolved = resolveTiling<DownTilings>(tiling, numRows, inter, hidden);
    if (resolved < 0)
    {
        return cudaErrorInvalidValue;
    }
    return launchTilingIndex<DownTilings>(resolved, h, downWeight, out, numRows, hidden, enablePdl, stream);
}

} // namespace kernels::glm_moe_dsa::dense_mlp_small_m

TRTLLM_NAMESPACE_END
