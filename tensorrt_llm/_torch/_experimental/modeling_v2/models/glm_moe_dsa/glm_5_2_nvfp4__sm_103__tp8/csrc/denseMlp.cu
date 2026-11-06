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

// Weight-streaming BF16 kernels for the dense (non-MoE) MLP at decode row counts (M <= 8).
//
// Both projections are bound by reading the BF16 weight matrix once from HBM. Each CTA owns a
// 16-row weight tile and its warps split the contraction dimension. Every thread loads 16 bytes of
// two weight rows and of one activation row per step and feeds them to two m16n8k16 tensor-core
// MMAs with the weight on the MMA M side and the (at most 8) activation rows on the MMA N side.
// The contraction index assigned to an MMA k-slot depends only on the thread's position inside its
// quad, so weight and activation fragments use the same permutation of k and the sum is exact up to
// FP32 summation order. Each quad therefore reads 64 contiguous bytes of one weight row per step.
//
// The gate/up kernel places 8 gate rows and the matching 8 up rows in one tile, so each thread ends
// with gate and up of the same output column and applies SwiGLU in registers.
//
// Bytes in flight: a thread holds one chunk of steps in registers, and one lane per quad prefetches
// the quad's row segment of the next chunk into L2 with cp.async.bulk.prefetch, which keeps the HBM
// queue full without spending registers.
//
// Programmatic dependent launch: weights never depend on the preceding kernel, so the first chunk
// (registers) and the L2 prefetch of the next one are issued before griddepcontrol.wait; dependents
// are released right after the wait. The tilings keep each grid within one wave of resident CTAs,
// so early-launched dependent CTAs that wait in griddepcontrol.wait cannot starve the primary grid.

#include "glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/csrc/denseMlp.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cstdint>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace dense_mlp
{

namespace
{

using Bf16 = __nv_bfloat16;

constexpr int kWarpSize = 32;
//! Weight rows per CTA tile (MMA M).
constexpr int kTileRows = 16;
//! Output columns per gate/up CTA: 8 gate rows + 8 up rows form one tile.
constexpr int kGateUpCols = kTileRows / 2;
//! BF16 values per 16-byte vector load.
constexpr int kVecElems = 8;
//! Threads per MMA quad; a quad covers kQuadThreads * kVecElems contraction elements per step.
constexpr int kQuadThreads = 4;
//! Contraction elements consumed by one warp step (two m16n8k16 MMAs).
constexpr int kKPerStep = kQuadThreads * kVecElems;

//! Compile-time tiling of one kernel instance.
//! \tparam kWarps warps per CTA; each warp owns a contiguous 1/kWarps slice of the contraction.
//! \tparam kSteps steps per register chunk.
//! \tparam kMinBlocks minimum resident CTAs per SM requested from the compiler (register budget).
//! \tparam kCachedX load activations with ld.global.ca instead of the non-coherent path.
//! \tparam kL2Ahead chunks of each weight row segment prefetched into L2 ahead of the register loads.
template <int kWarps_, int kSteps_, int kMinBlocks_, bool kCachedX_, int kL2Ahead_>
struct Tiling
{
    static constexpr int kWarps = kWarps_;
    static constexpr int kSteps = kSteps_;
    static constexpr int kMinBlocks = kMinBlocks_;
    static constexpr bool kCachedX = kCachedX_;
    static constexpr int kL2Ahead = kL2Ahead_;
    static constexpr int kKPerChunk = kKPerStep * kSteps;
};

__device__ __forceinline__ void pdlWait()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void pdlLaunchDependents()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

//! Streaming weight load: not kept in L1, 256-byte L2 fetch granularity.
__device__ __forceinline__ uint4 loadWeight(Bf16 const* ptr)
{
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "l"(ptr));
    return r;
}

//! Activation load: every CTA reads the same few rows, so keep them in L1.
template <bool kCached>
__device__ __forceinline__ uint4 loadActivation(Bf16 const* ptr)
{
    if constexpr (kCached)
    {
        uint4 r;
        asm volatile("ld.global.ca.v4.u32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                     : "l"(ptr));
        return r;
    }
    else
    {
        return __ldg(reinterpret_cast<uint4 const*>(ptr));
    }
}

__device__ __forceinline__ void prefetchL2(void const* ptr, uint32_t bytes)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(ptr), "r"(bytes) : "memory");
#endif
}

__device__ __forceinline__ void mmaBf16(
    float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

//! Registers of one chunk: kSteps steps of two weight rows and one activation row.
template <int kSteps>
struct Chunk
{
    uint4 w0[kSteps];
    uint4 w1[kSteps];
    uint4 x[kSteps];
};

template <int kSteps>
__device__ __forceinline__ void loadChunkWeights(Chunk<kSteps>& chunk, Bf16 const* w0, Bf16 const* w1, int chunkIdx)
{
    int const offset = chunkIdx * kKPerStep * kSteps;
#pragma unroll
    for (int u = 0; u < kSteps; ++u)
    {
        chunk.w0[u] = loadWeight(w0 + offset + u * kKPerStep);
        chunk.w1[u] = loadWeight(w1 + offset + u * kKPerStep);
    }
}

template <bool kCached, int kSteps>
__device__ __forceinline__ void loadChunkActivations(Chunk<kSteps>& chunk, Bf16 const* x, bool hasRow, int chunkIdx)
{
    int const offset = chunkIdx * kKPerStep * kSteps;
#pragma unroll
    for (int u = 0; u < kSteps; ++u)
    {
        chunk.x[u] = hasRow ? loadActivation<kCached>(x + offset + u * kKPerStep) : make_uint4(0U, 0U, 0U, 0U);
    }
}

//! Two MMAs per step. MMA A uses contraction elements 0..3 of the thread's 8-element vector, MMA B
//! elements 4..7. Fragment registers: a0/a2 = weight row "group", a1/a3 = weight row "group + 8",
//! b0/b1 = activation row "group".
template <int kSteps>
__device__ __forceinline__ void computeChunk(float (&accA)[4], float (&accB)[4], Chunk<kSteps> const& chunk)
{
#pragma unroll
    for (int u = 0; u < kSteps; ++u)
    {
        mmaBf16(accA, chunk.w0[u].x, chunk.w1[u].x, chunk.w0[u].y, chunk.w1[u].y, chunk.x[u].x, chunk.x[u].y);
        mmaBf16(accB, chunk.w0[u].z, chunk.w1[u].z, chunk.w0[u].w, chunk.w1[u].w, chunk.x[u].z, chunk.x[u].w);
    }
}

//! Streams the CTA's two weight rows per thread over the warp's contraction slice and returns the
//! CTA-reduced accumulator fragment in warp 0 (other warps return false).
//! acc[0..1] = tile row "group", activation rows 2 * tig + {0, 1}; acc[2..3] = tile row "group + 8".
template <typename T>
__device__ __forceinline__ bool streamTile(float (&acc)[4], Bf16 const* __restrict__ x, Bf16 const* __restrict__ weight,
    int numRows, int kDim, int row0, int row1)
{
    constexpr int kWarps = T::kWarps;
    constexpr int kSteps = T::kSteps;
    __shared__ float reduceBuf[kWarps][4][kWarpSize];

    int const lane = threadIdx.x % kWarpSize;
    int const warp = threadIdx.x / kWarpSize;
    int const group = lane / kQuadThreads;
    int const tig = lane % kQuadThreads;

    int const kPerWarp = kDim / kWarps;
    int const kBase = warp * kPerWarp + tig * kVecElems;
    Bf16 const* w0 = weight + static_cast<int64_t>(row0) * kDim + kBase;
    Bf16 const* w1 = weight + static_cast<int64_t>(row1) * kDim + kBase;
    bool const hasRow = group < numRows;
    Bf16 const* xRow = x + static_cast<int64_t>(hasRow ? group : 0) * kDim + kBase;
    int const numChunks = kPerWarp / T::kKPerChunk;

    // One lane per quad prefetches the quad's contiguous row segments of a chunk into L2.
    auto prefetchChunkL2 = [&](int chunkIdx)
    {
        if constexpr (T::kL2Ahead > 0)
        {
            if (tig == 0 && chunkIdx < numChunks)
            {
                uint32_t const bytes = T::kKPerChunk * static_cast<uint32_t>(sizeof(Bf16));
                prefetchL2(w0 + chunkIdx * T::kKPerChunk, bytes);
                prefetchL2(w1 + chunkIdx * T::kKPerChunk, bytes);
            }
        }
    };

    float accA[4] = {0.F, 0.F, 0.F, 0.F};
    float accB[4] = {0.F, 0.F, 0.F, 0.F};

    // Weights do not depend on the preceding kernel: request them before the dependency wait.
    Chunk<kSteps> chunk;
    loadChunkWeights(chunk, w0, w1, 0);
#pragma unroll
    for (int a = 1; a <= T::kL2Ahead; ++a)
    {
        prefetchChunkL2(a);
    }
    pdlWait();
    pdlLaunchDependents();
    loadChunkActivations<T::kCachedX>(chunk, xRow, hasRow, 0);

#pragma unroll 1
    for (int c = 0; c < numChunks; ++c)
    {
        if (c > 0)
        {
            loadChunkWeights(chunk, w0, w1, c);
            loadChunkActivations<T::kCachedX>(chunk, xRow, hasRow, c);
        }
        prefetchChunkL2(c + 1 + T::kL2Ahead);
        computeChunk(accA, accB, chunk);
    }

#pragma unroll
    for (int e = 0; e < 4; ++e)
    {
        acc[e] = accA[e] + accB[e];
    }
    if constexpr (kWarps > 1)
    {
#pragma unroll
        for (int e = 0; e < 4; ++e)
        {
            reduceBuf[warp][e][lane] = acc[e];
        }
        __syncthreads();
        if (warp != 0)
        {
            return false;
        }
        // Fixed order over warps keeps the result deterministic.
#pragma unroll
        for (int w = 1; w < kWarps; ++w)
        {
#pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                acc[e] += reduceBuf[w][e][lane];
            }
        }
    }
    return true;
}

__device__ __forceinline__ float roundToBf16(float v)
{
    return __bfloat162float(__float2bfloat16_rn(v));
}

template <typename T>
__global__ void __launch_bounds__(T::kWarps* kWarpSize, T::kMinBlocks) gateUpSwigluKernel(Bf16 const* __restrict__ x,
    Bf16 const* __restrict__ weight, Bf16* __restrict__ h, int numRows, int hidden, int inter)
{
    int const lane = threadIdx.x % kWarpSize;
    int const group = lane / kQuadThreads;
    int const tig = lane % kQuadThreads;
    int const col = blockIdx.x * kGateUpCols + group;

    float acc[4];
    if (!streamTile<T>(acc, x, weight, numRows, hidden, col, inter + col))
    {
        return;
    }
#pragma unroll
    for (int b = 0; b < 2; ++b)
    {
        int const row = 2 * tig + b;
        if (row < numRows)
        {
            // gate and up are rounded to BF16 first, matching a BF16 GEMM output feeding SwiGLU.
            float const gate = roundToBf16(acc[b]);
            float const up = roundToBf16(acc[2 + b]);
            float const sig = 1.F / (1.F + __expf(-gate));
            h[static_cast<int64_t>(row) * inter + col] = __float2bfloat16_rn(gate * sig * up);
        }
    }
}

template <typename T>
__global__ void __launch_bounds__(T::kWarps* kWarpSize, T::kMinBlocks) downKernel(Bf16 const* __restrict__ h,
    Bf16 const* __restrict__ weight, Bf16* __restrict__ out, int numRows, int inter, int hidden)
{
    int const lane = threadIdx.x % kWarpSize;
    int const group = lane / kQuadThreads;
    int const tig = lane % kQuadThreads;
    int const n0 = blockIdx.x * kTileRows + group;

    float acc[4];
    if (!streamTile<T>(acc, h, weight, numRows, inter, n0, n0 + kTileRows / 2))
    {
        return;
    }
#pragma unroll
    for (int b = 0; b < 2; ++b)
    {
        int const row = 2 * tig + b;
        if (row < numRows)
        {
            out[static_cast<int64_t>(row) * hidden + n0] = __float2bfloat16_rn(acc[b]);
            out[static_cast<int64_t>(row) * hidden + n0 + kTileRows / 2] = __float2bfloat16_rn(acc[2 + b]);
        }
    }
}

template <typename T>
cudaError_t launchKernel(void (*kernel)(Bf16 const*, Bf16 const*, Bf16*, int, int, int), int gridSize, Bf16 const* in,
    Bf16 const* weight, Bf16* out, int numRows, int kDim, int outDim, bool enablePdl, cudaStream_t stream)
{
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(gridSize);
    config.blockDim = dim3(T::kWarps * kWarpSize);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = enablePdl ? 1 : 0;
    config.attrs = attrs;
    config.numAttrs = 1;
    return cudaLaunchKernelEx(&config, kernel, in, weight, out, numRows, kDim, outDim);
}

//! Variant table (warps, steps); the template arguments of each index are in launchVariant.
struct VariantDesc
{
    int warps;
    int steps;
};

constexpr VariantDesc kVariants[] = {
    {4, 8}, // 0: down default. L2 prefetch one chunk ahead
    {8, 4}, // 1: gate/up default. Three CTAs/SM register budget, L2 prefetch one chunk ahead
    {8, 4}, // 2: as 1 with ld.global.ca activation loads
    {4, 8}, // 3: as 0 without L2 prefetch
    {8, 8}, // 4: two CTAs/SM, no L2 prefetch
};
constexpr int kNumVariants = sizeof(kVariants) / sizeof(kVariants[0]);

bool isContractionSupported(int kDim, int variant)
{
    VariantDesc const& v = kVariants[variant];
    return kDim > 0 && kDim % (v.warps * kKPerStep * v.steps) == 0;
}

//! Dispatches a variant index to its kernel instance; kGateUp selects the gate/up or the down kernel.
template <bool kGateUp>
cudaError_t launchVariant(int variant, Bf16 const* in, Bf16 const* weight, Bf16* out, int numRows, int kDim, int outDim,
    bool enablePdl, cudaStream_t stream)
{
    auto launch = [&](auto tiling) -> cudaError_t
    {
        using T = decltype(tiling);
        if constexpr (kGateUp)
        {
            return launchKernel<T>(
                gateUpSwigluKernel<T>, outDim / kGateUpCols, in, weight, out, numRows, kDim, outDim, enablePdl, stream);
        }
        else
        {
            return launchKernel<T>(
                downKernel<T>, outDim / kTileRows, in, weight, out, numRows, kDim, outDim, enablePdl, stream);
        }
    };
    switch (variant)
    {
    case 0: return launch(Tiling<4, 8, 1, false, 1>{});
    case 1: return launch(Tiling<8, 4, 3, false, 1>{});
    case 2: return launch(Tiling<8, 4, 3, true, 1>{});
    case 3: return launch(Tiling<4, 8, 1, false, 0>{});
    case 4: return launch(Tiling<8, 8, 2, false, 0>{});
    default: return cudaErrorInvalidValue;
    }
}

} // namespace

int getNumVariants()
{
    return kNumVariants;
}

bool isGateUpSupported(int numRows, int hidden, int inter, int variant)
{
    return variant >= 0 && variant < kNumVariants && numRows >= 1 && numRows <= kMaxRows && inter > 0
        && inter % kGateUpCols == 0 && isContractionSupported(hidden, variant);
}

bool isDownSupported(int numRows, int inter, int hidden, int variant)
{
    return variant >= 0 && variant < kNumVariants && numRows >= 1 && numRows <= kMaxRows && hidden > 0
        && hidden % kTileRows == 0 && isContractionSupported(inter, variant);
}

cudaError_t invokeGateUpSwiglu(Bf16 const* x, Bf16 const* gateUpWeight, Bf16* h, int numRows, int hidden, int inter,
    int variant, bool enablePdl, cudaStream_t stream)
{
    if (!isGateUpSupported(numRows, hidden, inter, variant))
    {
        return cudaErrorInvalidValue;
    }
    return launchVariant<true>(variant, x, gateUpWeight, h, numRows, hidden, inter, enablePdl, stream);
}

cudaError_t invokeDown(Bf16 const* h, Bf16 const* downWeight, Bf16* out, int numRows, int inter, int hidden,
    int variant, bool enablePdl, cudaStream_t stream)
{
    if (!isDownSupported(numRows, inter, hidden, variant))
    {
        return cudaErrorInvalidValue;
    }
    return launchVariant<false>(variant, h, downWeight, out, numRows, inter, hidden, enablePdl, stream);
}

} // namespace dense_mlp

TRTLLM_MV2_NAMESPACE_END
