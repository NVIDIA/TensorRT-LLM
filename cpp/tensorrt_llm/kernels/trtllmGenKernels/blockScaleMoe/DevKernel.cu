/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "DevKernel.h"

#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"
#include <cub/cub.cuh>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include <algorithm>
#include <cstdint>

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function for array conversion
template <class T, class U>
__host__ __device__ constexpr static U arrayConvert(T const& input)
{
    cutlass::NumericArrayConverter<typename U::Element, typename T::Element, U::kElements> converter;
    return converter(input);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace moe::dev
{

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace activation
{

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = batchedGemm::trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float silu(float x)
{
    return x / (1.0f + expf(-x));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void activationKernel(KernelParams params)
{
    using Type = typename KernelParams::Type;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // immediately trigger the secondary kernel when using PDL, then wait on primary
    if constexpr (KernelParams::UsePdl)
    {
        cudaTriggerProgrammaticLaunchCompletion();
        cudaGridDependencySynchronize();
    }
#endif

    // FP8 separate-activation path: swiglu_limit is uniform across experts,
    // passed by value via KernelParams::swigluLimit (gated by hasSwigluLimit).
    // Apply gate.clamp(max=limit) / up.clamp(-limit, limit) before silu/mul.
    // Per-expert non-uniform limits are not supported here.
    bool const hasSwigluLimit = params.hasSwigluLimit;
    float const swigluLimit = params.swigluLimit;

    for (int tokenIdx = blockIdx.z; tokenIdx < params.numTokens; tokenIdx += gridDim.z)
    {
        // Look over experts per token
        for (int k = blockIdx.y; k < params.topK; k += gridDim.y)
        {
            int const expandedIdx = tokenIdx * params.topK + k;
            int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];
            if (permutedIdx == -1)
                continue;

            // Loop over hidden dim
            for (int hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x; hiddenIdx < params.innerDim / 2;
                 hiddenIdx += blockDim.x * gridDim.x)
            {
                // Compute global-memory offsets in int64: under Attention DP + AllGather the
                // permuted token count (permutedIdx up to totalNumPaddedTokens) times a hidden/inner
                // dimension can exceed INT_MAX, so 32-bit index math would overflow. Applies to all
                // permutedIdx/tokenIdx * dim offsets in this file.
                int64_t const baseIdx = static_cast<int64_t>(permutedIdx) * params.innerDim + hiddenIdx;

                float x1 = (float) params.inPtr[baseIdx];                       // up (linear)
                float x2 = (float) params.inPtr[baseIdx + params.innerDim / 2]; // gate (silu input)

                if (hasSwigluLimit)
                {
                    x2 = fminf(x2, swigluLimit);
                    x1 = fmaxf(fminf(x1, swigluLimit), -swigluLimit);
                }

                float act = silu(x2);
                Type out = (Type) (act * x1);

                int64_t const outIdx = static_cast<int64_t>(permutedIdx) * (params.innerDim / 2) + hiddenIdx;
                params.outPtr[outIdx] = out;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Float4Max
{
    __device__ __forceinline__ float4 operator()(float4 const& a, float4 const& b) const
    {
        float4 result;
        result.x = fmaxf(a.x, b.x);
        result.y = fmaxf(a.y, b.y);
        result.z = fmaxf(a.z, b.z);
        result.w = fmaxf(a.w, b.w);
        return result;
    }
};

struct Float2Max
{
    __device__ __forceinline__ float2 operator()(float2 const& a, float2 const& b) const
    {
        float2 result;
        result.x = fmaxf(a.x, b.x);
        result.y = fmaxf(a.y, b.y);
        return result;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename VecType, int size>
__device__ __forceinline__ VecType packedTypeFromArray(float data[size])
{
    return {};
}

template <>
__device__ __forceinline__ float4 packedTypeFromArray<float4, 4>(float data[4])
{
    float4 result;
    result.x = data[0];
    result.y = data[1];
    result.z = data[2];
    result.w = data[3];
    return result;
}

template <>
__device__ __forceinline__ float2 packedTypeFromArray<float2, 2>(float data[2])
{
    float2 result;
    result.x = data[0];
    result.y = data[1];
    return result;
}

template <>
__device__ __forceinline__ float packedTypeFromArray<float, 1>(float data[1])
{
    return data[0];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename PackedType, int size>
__device__ __forceinline__ cutlass::Array<float, size> arrayFromPackedType(PackedType data)
{
    return cutlass::Array<float, size>{};
}

template <>
__device__ __forceinline__ cutlass::Array<float, 4> arrayFromPackedType<float4, 4>(float4 data)
{
    return cutlass::Array<float, 4>{data.x, data.y, data.z, data.w};
}

template <>
__device__ __forceinline__ cutlass::Array<float, 2> arrayFromPackedType<float2, 2>(float2 data)
{
    return cutlass::Array<float, 2>{data.x, data.y};
}

template <>
__device__ __forceinline__ cutlass::Array<float, 1> arrayFromPackedType<float, 1>(float data)
{
    return cutlass::Array<float, 1>{data};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int NUM_TOKENS_PER_CTA>
struct KernelTraits;

template <>
struct KernelTraits<4>
{
    using MaxOp = Float4Max;
    using PackedType = float4;
};

template <>
struct KernelTraits<2>
{
    using MaxOp = Float2Max;
    using PackedType = float2;
};

template <>
struct KernelTraits<1>
{
#if CUDA_VERSION >= 12090
    using MaxOp = cuda::maximum<>;
#else
    using MaxOp = cub::Max;
#endif
    using PackedType = float;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr int DEEP_SEEK_ACTIVATION_NUM_THREADS_PER_CTA = 128;

////////////////////////////////////////////////////////////////////////////////////////////////////

// Permuted-space SwiGLU for the DeepSeek-FP8 separate-activation path.
//
// `activationDeepSeekKernel` below grids over the *expanded* index space
// (numTokens x topK) and discovers work by loading expandedIdxToPermutedIdx,
// skipping entries that map to -1. Under expert parallelism only 1/ep_size of
// those entries are local, so most of the launched CTAs do no memory work at
// all -- yet they still run the unconditional cub::BlockReduce. At a large
// context and a high expert-parallel degree the launched CTA count exceeds the
// permuted rows of real work by the ep_size factor, and the achieved bandwidth
// is a small fraction of what the row count alone would need.
//
// Every memory access in that kernel is addressed by (permutedIdx, hiddenIdx)
// only -- the expanded index exists purely to find the work. So grid directly
// over the permuted rows instead and the indirection, the -1 slots and the
// ep_size-fold CTA inflation all disappear together.
//
// Layout: one warp owns exactly one (permutedRow, 128-element scale block).
// 32 lanes x 4 elements = 128 = one scale block, so the amax reduction is a
// single warp shuffle instead of a shared-memory block reduce, and each lane
// moves 4 bytes per load instead of 1.
//
// totalNumPaddedTokens is only known on the device, so the grid is persistent
// and strides over the row space. This visits the per-expert tile padding that
// the expanded-space kernel skips (~4% extra rows at 32 local experts); those
// rows are dropped by the finalize kernel. The arithmetic below deliberately
// matches activationDeepSeekKernel bit for bit, including its finite all-zero
// block handling.
constexpr int kDsActWarpSize = 32;
constexpr int kDsActEltsPerSf = 128;
constexpr int kDsActEltsPerThread = kDsActEltsPerSf / kDsActWarpSize;
constexpr int kDsActWarpsPerCta = 4;
constexpr int kDsActPermutedNumThreadsPerCta = kDsActWarpSize * kDsActWarpsPerCta;
constexpr float kDsActAmaxEpsilon = 1.0e-10F;

constexpr bool shouldUsePermutedActivation(int innerDim, int numTokens, int topK, int numExperts, int tileTokensDim)
{
    int const outputDim = innerDim / 2;
    bool const layoutEligible = outputDim >= kDsActEltsPerSf && outputDim % kDsActEltsPerSf == 0 && innerDim % 8 == 0;
    int64_t const realRowsPerExpert = numExperts > 0 ? static_cast<int64_t>(numTokens) * topK / numExperts : 0;
    bool const paddingAmortised = tileTokensDim > 0 && realRowsPerExpert >= tileTokensDim;
    return layoutEligible && paddingAmortised;
}

template <typename KernelParams>
__global__ void activationDeepSeekPermutedKernel(KernelParams params)
{
    using Type = typename KernelParams::Type;
    using PackedIo = uint32_t; // kDsActEltsPerThread x 8-bit elements

    static_assert(kDsActEltsPerThread == 4, "PackedIo assumes 4 elements per thread");

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    if constexpr (KernelParams::UsePdl)
    {
        cudaTriggerProgrammaticLaunchCompletion();
        cudaGridDependencySynchronize();
    }
#endif

    float constexpr kE4m3MaxVal{448.F};

    int const totalNumPaddedTokens = params.totalNumPaddedTokens[0];
    int const outputDim = params.innerDim / 2;
    int const numSfBlocks = outputDim / kDsActEltsPerSf;

    bool const hasSwigluLimit = params.hasSwigluLimit;
    float const swigluLimit = params.swigluLimit;

    int const lane = threadIdx.x % kDsActWarpSize;
    int const warpInCta = threadIdx.x / kDsActWarpSize;

    int64_t const numTasks = static_cast<int64_t>(totalNumPaddedTokens) * numSfBlocks;
    int64_t const taskStride = static_cast<int64_t>(gridDim.x) * kDsActWarpsPerCta;

    for (int64_t task = static_cast<int64_t>(blockIdx.x) * kDsActWarpsPerCta + warpInCta; task < numTasks;
         task += taskStride)
    {
        int const permutedIdx = static_cast<int>(task / numSfBlocks);
        int const sfBlock = static_cast<int>(task % numSfBlocks);
        int const hiddenBase = sfBlock * kDsActEltsPerSf + lane * kDsActEltsPerThread;

        // Both scales are uniform across the warp: one per (row, scale block).
        float const scale1 = params.inDqSfsPtr[permutedIdx + totalNumPaddedTokens * sfBlock];
        float const scale2 = params.inDqSfsPtr[permutedIdx + totalNumPaddedTokens * (sfBlock + numSfBlocks)];

        int64_t const baseIdx = static_cast<int64_t>(permutedIdx) * params.innerDim + hiddenBase;
        PackedIo const packed1 = *reinterpret_cast<PackedIo const*>(params.inPtr + baseIdx);
        PackedIo const packed2 = *reinterpret_cast<PackedIo const*>(params.inPtr + baseIdx + outputDim);

        Type const* elts1 = reinterpret_cast<Type const*>(&packed1);
        Type const* elts2 = reinterpret_cast<Type const*>(&packed2);

        float out[kDsActEltsPerThread];
        float aMax = 0.F;
#pragma unroll
        for (int i = 0; i < kDsActEltsPerThread; ++i)
        {
            float x1 = scale1 * static_cast<float>(elts1[i]); // up (linear)
            float x2 = scale2 * static_cast<float>(elts2[i]); // gate (silu input)
            if (hasSwigluLimit)
            {
                x2 = fminf(x2, swigluLimit);
                x1 = fmaxf(fminf(x1, swigluLimit), -swigluLimit);
            }
            out[i] = silu(x2) * x1;
            aMax = fmaxf(aMax, fabsf(out[i]));
        }

#pragma unroll
        for (int offset = kDsActWarpSize / 2; offset > 0; offset >>= 1)
        {
            aMax = fmaxf(aMax, __shfl_xor_sync(0xffffffffu, aMax, offset));
        }

        // Floor aMax so an all-zero block stays finite: without it scaleOut is
        // zero and quantizing evaluates 0 / 0, which is undefined and writes FP8
        // NaNs into that row. Same epsilon as the DeepGEMM FP8 activation
        // quantizer (fp8_utils.py).
        float const scaleOut = fmaxf(aMax, kDsActAmaxEpsilon) / kE4m3MaxVal;

        if (lane == 0)
        {
            params.outDqSfsPtr[permutedIdx + totalNumPaddedTokens * sfBlock] = scaleOut;
        }

        PackedIo packedOut;
        Type* outElts = reinterpret_cast<Type*>(&packedOut);
#pragma unroll
        for (int i = 0; i < kDsActEltsPerThread; ++i)
        {
            // Divide; do NOT hoist a reciprocal. `x / s` and `x * (1/s)` round
            // differently, and an equivalence run showed that single ulp flip a
            // greedy-decoded token. This must match activationDeepSeekKernel
            // bit for bit.
            outElts[i] = static_cast<Type>(out[i] / scaleOut);
        }
        *reinterpret_cast<PackedIo*>(params.outPtr + static_cast<int64_t>(permutedIdx) * outputDim + hiddenBase)
            = packedOut;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Expanded-space SwiGLU for the four-token launch used by medium and large
// decode batches. One warp owns one token and one 128-element scale block, so
// the four warps in a CTA process the same four tokens as the generic kernel
// without a shared-memory block reduction. Each lane moves four adjacent FP8
// elements, matching the packed-I/O layout of the permuted-space kernel.
template <typename KernelParams>
__global__ void activationDeepSeekExpandedPackedKernel(KernelParams params)
{
    using Type = typename KernelParams::Type;
    using PackedIo = uint32_t; // kDsActEltsPerThread x 8-bit elements

    static_assert(kDsActEltsPerThread == 4, "PackedIo assumes 4 elements per thread");

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    if constexpr (KernelParams::UsePdl)
    {
        cudaTriggerProgrammaticLaunchCompletion();
        cudaGridDependencySynchronize();
    }
#endif

    float constexpr kE4m3MaxVal{448.F};

    int const totalNumPaddedTokens = params.totalNumPaddedTokens[0];
    int const outputDim = params.innerDim / 2;
    int const numSfBlocks = outputDim / kDsActEltsPerSf;
    int const lane = threadIdx.x % kDsActWarpSize;
    int const tokenInCta = threadIdx.x / kDsActWarpSize;
    int const hiddenBase = blockIdx.x * kDsActEltsPerSf + lane * kDsActEltsPerThread;
    constexpr int kNumTokensPerCta = KernelParams::NumTokensPerCta;

    // LAUNCH_ACTIVATION instantiates every supported token-count specialization
    // even though this kernel is selected with four at runtime.
    if (tokenInCta >= kNumTokensPerCta)
    {
        return;
    }

    bool const hasSwigluLimit = params.hasSwigluLimit;
    float const swigluLimit = params.swigluLimit;

    for (int k = blockIdx.z; k < params.topK; k += gridDim.z)
    {
        for (int tokenCtaIdx = blockIdx.y * kNumTokensPerCta; tokenCtaIdx < params.numTokens;
             tokenCtaIdx += gridDim.y * kNumTokensPerCta)
        {
            int const tokenIdx = tokenCtaIdx + tokenInCta;
            if (tokenIdx >= params.numTokens)
            {
                continue;
            }

            int const expandedIdx = tokenIdx * params.topK + k;
            int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];
            if (permutedIdx == -1)
            {
                continue;
            }

            int const sfBlock = blockIdx.x;
            float const scale1 = params.inDqSfsPtr[permutedIdx + totalNumPaddedTokens * sfBlock];
            float const scale2 = params.inDqSfsPtr[permutedIdx + totalNumPaddedTokens * (sfBlock + numSfBlocks)];

            int64_t const baseIdx = static_cast<int64_t>(permutedIdx) * params.innerDim + hiddenBase;
            PackedIo const packed1 = *reinterpret_cast<PackedIo const*>(params.inPtr + baseIdx);
            PackedIo const packed2 = *reinterpret_cast<PackedIo const*>(params.inPtr + baseIdx + outputDim);

            Type const* elts1 = reinterpret_cast<Type const*>(&packed1);
            Type const* elts2 = reinterpret_cast<Type const*>(&packed2);

            float out[kDsActEltsPerThread];
            float aMax = 0.F;
#pragma unroll
            for (int i = 0; i < kDsActEltsPerThread; ++i)
            {
                float x1 = scale1 * static_cast<float>(elts1[i]);
                float x2 = scale2 * static_cast<float>(elts2[i]);
                if (hasSwigluLimit)
                {
                    x2 = fminf(x2, swigluLimit);
                    x1 = fmaxf(fminf(x1, swigluLimit), -swigluLimit);
                }
                out[i] = silu(x2) * x1;
                aMax = fmaxf(aMax, fabsf(out[i]));
            }

#pragma unroll
            for (int offset = kDsActWarpSize / 2; offset > 0; offset >>= 1)
            {
                aMax = fmaxf(aMax, __shfl_xor_sync(0xffffffffu, aMax, offset));
            }

            float const scaleOut = fmaxf(aMax, kDsActAmaxEpsilon) / kE4m3MaxVal;
            if (lane == 0)
            {
                params.outDqSfsPtr[permutedIdx + totalNumPaddedTokens * sfBlock] = scaleOut;
            }

            PackedIo packedOut;
            Type* outElts = reinterpret_cast<Type*>(&packedOut);
#pragma unroll
            for (int i = 0; i < kDsActEltsPerThread; ++i)
            {
                outElts[i] = static_cast<Type>(out[i] / scaleOut);
            }
            *reinterpret_cast<PackedIo*>(params.outPtr + static_cast<int64_t>(permutedIdx) * outputDim + hiddenBase)
                = packedOut;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void activationDeepSeekKernel(KernelParams params)
{
    using Type = typename KernelParams::Type;
    int32_t constexpr NumTokensPerCta = KernelParams::NumTokensPerCta;
    using KernelTraits = KernelTraits<NumTokensPerCta>;
    using MaxOp = typename KernelTraits::MaxOp;
    using PackedType = typename KernelTraits::PackedType;
    using BlockReduce = cub::BlockReduce<PackedType, DEEP_SEEK_ACTIVATION_NUM_THREADS_PER_CTA>;

    __shared__ float s_scaleOutArr[NumTokensPerCta];
    __shared__ typename BlockReduce::TempStorage tempStorage;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // immediately trigger the secondary kernel when using PDL, then wait on primary
    if constexpr (KernelParams::UsePdl)
    {
        cudaTriggerProgrammaticLaunchCompletion();
        cudaGridDependencySynchronize();
    }
#endif

    // The largest (finite) value that can be represented using E4m3.
    float constexpr E4m3MaxVal{448.f};

    // FP8 separate-activation path: swiglu_limit is uniform across experts,
    // passed by value via KernelParams::swigluLimit (gated by hasSwigluLimit).
    // Apply gate.clamp(max=limit) / up.clamp(-limit, limit) AFTER
    // dequantization but BEFORE silu/mul. Per-expert non-uniform limits are
    // not supported here.
    bool const hasSwigluLimit = params.hasSwigluLimit;
    float const swigluLimit = params.swigluLimit;

    int const totalNumPaddedTokens = params.totalNumPaddedTokens[0];
    // Loop over tokens
    float scale1Arr[NumTokensPerCta];
    float scale2Arr[NumTokensPerCta];
    float dataX1Arr[NumTokensPerCta];
    float dataX2Arr[NumTokensPerCta];
    float outArr[NumTokensPerCta];
    float absOutArr[NumTokensPerCta];
    int permutedIdxArr[NumTokensPerCta];

    // Loop over tokens
    for (int k = blockIdx.z; k < params.topK; k += gridDim.z)
    {
        for (int tokenCtaIdx = blockIdx.y * NumTokensPerCta; tokenCtaIdx < params.numTokens;
             tokenCtaIdx += gridDim.y * NumTokensPerCta)
        {
            for (int hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x; hiddenIdx < params.innerDim / 2;
                 hiddenIdx += blockDim.x * gridDim.x)
            {
#pragma unroll
                for (int tokenInCtaIdx = 0; tokenInCtaIdx < NumTokensPerCta; tokenInCtaIdx++)
                {
                    scale1Arr[tokenInCtaIdx] = 0.0f;
                    scale2Arr[tokenInCtaIdx] = 0.0f;
                    dataX1Arr[tokenInCtaIdx] = 0.0f;
                    dataX2Arr[tokenInCtaIdx] = 0.0f;
                    outArr[tokenInCtaIdx] = 0.0f;
                    absOutArr[tokenInCtaIdx] = 0.0f;
                }
#pragma unroll
                for (int tokenInCtaIdx = 0; tokenInCtaIdx < NumTokensPerCta; tokenInCtaIdx++)
                {
                    int const tokenIdx = tokenCtaIdx + tokenInCtaIdx;
                    if (tokenIdx >= params.numTokens)
                    {
                        break;
                    }

                    int const expandedIdx = tokenIdx * params.topK + k;
                    int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];
                    permutedIdxArr[tokenInCtaIdx] = permutedIdx;
                    if (permutedIdx == -1)
                    {
                        continue;
                    }

                    // Process blocks for this CTA
                    int64_t const baseIdx = static_cast<int64_t>(permutedIdx) * params.innerDim + hiddenIdx;

                    int const scale1Idx = permutedIdx + totalNumPaddedTokens * (hiddenIdx / 128);
                    int const scale2Idx
                        = permutedIdx + totalNumPaddedTokens * ((hiddenIdx / 128) + (params.innerDim / 2 / 128));

                    scale1Arr[tokenInCtaIdx] = params.inDqSfsPtr[scale1Idx];
                    scale2Arr[tokenInCtaIdx] = params.inDqSfsPtr[scale2Idx];
                    dataX1Arr[tokenInCtaIdx] = static_cast<float>(params.inPtr[baseIdx]);
                    dataX2Arr[tokenInCtaIdx] = static_cast<float>(params.inPtr[baseIdx + params.innerDim / 2]);
                }

#pragma unroll
                for (int tokenInCtaIdx = 0; tokenInCtaIdx < NumTokensPerCta; tokenInCtaIdx++)
                {
                    float x1 = scale1Arr[tokenInCtaIdx] * dataX1Arr[tokenInCtaIdx]; // up (linear)
                    float x2 = scale2Arr[tokenInCtaIdx] * dataX2Arr[tokenInCtaIdx]; // gate (silu input)
                    if (hasSwigluLimit)
                    {
                        x2 = fminf(x2, swigluLimit);
                        x1 = fmaxf(fminf(x1, swigluLimit), -swigluLimit);
                    }
                    float act = silu(x2);
                    float out = act * x1;
                    outArr[tokenInCtaIdx] = out;
                    absOutArr[tokenInCtaIdx] = fabsf(out);
                }

                auto absOutPacked = packedTypeFromArray<PackedType, NumTokensPerCta>(absOutArr);
                auto aMaxPacked = BlockReduce(tempStorage).Reduce(absOutPacked, MaxOp{});
                auto aMaxArr = arrayFromPackedType<PackedType, NumTokensPerCta>(aMaxPacked);

#pragma unroll
                for (int tokenInCtaIdx = 0; tokenInCtaIdx < NumTokensPerCta; tokenInCtaIdx++)
                {
                    if (threadIdx.x == 0)
                    {
                        auto const tokenIdx = tokenCtaIdx + tokenInCtaIdx;
                        if (tokenIdx >= params.numTokens)
                        {
                            break;
                        }
                        int const permutedIdx = permutedIdxArr[tokenInCtaIdx];
                        if (permutedIdx == -1)
                        {
                            continue;
                        }
                        float const scaleOut = fmaxf(aMaxArr[tokenInCtaIdx], kDsActAmaxEpsilon) / E4m3MaxVal;
                        s_scaleOutArr[tokenInCtaIdx] = scaleOut;
                        int const scaleOut_idx
                            = permutedIdxArr[tokenInCtaIdx] + totalNumPaddedTokens * (hiddenIdx / 128);
                        params.outDqSfsPtr[scaleOut_idx] = scaleOut;
                    }
                }
                __syncthreads();

#pragma unroll
                for (int tokenInCtaIdx = 0; tokenInCtaIdx < NumTokensPerCta; tokenInCtaIdx++)
                {
                    auto const tokenIdx = tokenCtaIdx + tokenInCtaIdx;
                    if (tokenIdx >= params.numTokens)
                    {
                        break;
                    }
                    int const permutedIdx = permutedIdxArr[tokenInCtaIdx];
                    if (permutedIdx == -1)
                    {
                        continue;
                    }
                    float const scaleOut = s_scaleOutArr[tokenInCtaIdx];
                    int64_t const outIdx = static_cast<int64_t>(permutedIdx) * (params.innerDim / 2) + hiddenIdx;
                    params.outPtr[outIdx] = static_cast<Type>(outArr[tokenInCtaIdx] / scaleOut);
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void run(Data const& data, void* stream)
{
    if (data.mDtypeElt == tg::Dtype::E2m1)
    {
        // Note: this should be unreachable because the options are checked beforehand.
        // E2m1 requires using higher-precision intermediate data (bf16).
        TLLM_CHECK_WITH_INFO(false, "Activation with E2m1_t isn't supported.");
        return;
    }

    if (data.mUseDeepSeekFp8)
    {
        constexpr int NUM_ELTS_PER_LOAD = 1;
        constexpr int NUM_ELTS_PER_SF = 128;

        int device{-1};
        cudaGetDevice(&device);
        int numSms = 0;
        cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, device);

        // Output dimension is innerDim / 2, and each scale block is 128 elements
        int const outputDim = data.innerDim / 2;
        int const numScaleBlocks = (outputDim + NUM_ELTS_PER_SF - 1) / NUM_ELTS_PER_SF;
        int const gridSizeX = (numScaleBlocks + NUM_ELTS_PER_LOAD - 1) / NUM_ELTS_PER_LOAD;

        auto numCtas = gridSizeX * data.numTokens * data.topK;
        // FIXME: This is heruistic based on very short benchmark.
        int numTokensPerCta = 1;
        if (numCtas > numSms * 32)
        {
            numTokensPerCta = 4;
        }
        else if (numCtas > numSms * 4)
        {
            numTokensPerCta = 2;
        }
        else
        {
            numTokensPerCta = 1;
        }

        int const gridSizeY = std::min(8192, (data.numTokens + numTokensPerCta - 1) / numTokensPerCta);

        const dim3 grid(gridSizeX, gridSizeY, data.topK);

        // The two kernels sweep different spaces, and which one is cheaper flips
        // with batch size.
        //
        // The expanded-space kernel visits numTokens x topK slots and skips the
        // ~(1 - 1/ep_size) of them that are not local, so it never touches the
        // per-expert tile padding. The permuted-space kernel sweeps
        // [0, totalNumPaddedTokens), which *is* padded: each local expert
        // contributes up to tileTokensDim-1 rows of padding that carry no real
        // tokens but cost a full row of load/compute/store.
        //
        // At prefill that padding is noise next to the real rows, and the
        // permuted sweep wins by the ep_size factor. At decode the ratio
        // inverts: a single token leaves well under one real row per expert
        // against the same padding, so the permuted kernel does almost nothing
        // but padding. Getting this wrong costs more on every decode step than
        // the prefill win is worth over a full generation.
        //
        // So gate on real work per expert. tileTokensDim is exactly the padding
        // granularity, which makes it the natural threshold: below it, an
        // expert's real rows do not even fill the tile that must be swept for it.
        if (shouldUsePermutedActivation(data.innerDim, data.numTokens, data.topK, data.numExperts, data.tileTokensDim))
        {
            int64_t const maxTasks = static_cast<int64_t>(data.numTokens) * data.topK * (outputDim / kDsActEltsPerSf);
            int64_t const ctasForAllTasks = (maxTasks + kDsActWarpsPerCta - 1) / kDsActWarpsPerCta;
            // Persistent grid: totalNumPaddedTokens is a device-side value, so the
            // host can only bound it. Cap at a few waves and let the grid stride
            // absorb the difference rather than launching the (numTokens x topK)
            // worst case that the expanded-space kernel pays unconditionally.
            int const numCtas = static_cast<int>(std::min<int64_t>(ctasForAllTasks, int64_t{numSms} * 32));
            dim3 const permutedGrid(std::max(numCtas, 1), 1, 1);

            LAUNCH_ACTIVATION(
                data, activationDeepSeekPermutedKernel, 1, permutedGrid, kDsActPermutedNumThreadsPerCta, 0, stream);
        }
        else
        {
            bool const usePackedExpanded
                = numTokensPerCta == kDsActWarpsPerCta && outputDim % kDsActEltsPerSf == 0 && data.innerDim % 8 == 0;
            if (usePackedExpanded)
            {
                LAUNCH_ACTIVATION(data, activationDeepSeekExpandedPackedKernel, kDsActWarpsPerCta, grid,
                    kDsActPermutedNumThreadsPerCta, 0, stream);
            }
            else
            {
                LAUNCH_ACTIVATION(data, activationDeepSeekKernel, numTokensPerCta, grid,
                    DEEP_SEEK_ACTIVATION_NUM_THREADS_PER_CTA, 0, stream);
            }
        }
    }
    else
    {
        int const numThreads = 256;
        const dim3 grid(data.innerDim / 128, data.topK, std::min(8192, data.numTokens));

        LAUNCH_ACTIVATION(data, activationKernel, 1, grid, numThreads, 0, stream);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace activation

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace convertsf
{

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = batchedGemm::trtllm::gen;

namespace dev
{
// Compute the offset that corresponds to (dataRowIdx, dataBlkColIdx) in the SF tensor where
// dataRowIdx and dataBlkColIdx are the respective indices of the row and the block of 16 elts
// from the K dim in the tensor of data.
inline __device__ int64_t getSfOffset(int32_t dataRowIdx, int32_t dataBlkColIdx, int32_t numDataBlksPerRow)
{

    // The number of rows of SF per block.
    static int32_t constexpr NumRowsPerSfBlock = 128;
    // The number of cols of SF per block.
    static int32_t constexpr NumColsPerSfBlock = 4;
    // The size of each SF block.
    static int32_t constexpr NumBytesPerSfBlock = NumRowsPerSfBlock * NumColsPerSfBlock;

    // The number of rows of data per SF block.
    static int32_t constexpr NumDataRowsPerSfBlock = NumRowsPerSfBlock;
    // The number of cols of blocks of data per SF block.
    static int32_t constexpr NumDataBlkColsPerSfBlock = NumColsPerSfBlock;

    // The row of the SF block in the SF tensor.
    int sfBlkRowIdx = dataRowIdx / NumDataRowsPerSfBlock;
    // The col of the SF block in the SF tensor.
    int sfBlkColIdx = dataBlkColIdx / NumDataBlkColsPerSfBlock;
    // The blocks are stored row-major in the tensor of scaling factors.
    int sfBlkIdx = sfBlkRowIdx * numDataBlksPerRow / NumDataBlkColsPerSfBlock + sfBlkColIdx;

    // Find the row in the SF block.
    int sfRowIdx = (dataRowIdx % 32) * 4 + (dataRowIdx % NumDataRowsPerSfBlock) / 32;
    // Find the col in the SF block.
    int sfColIdx = (dataBlkColIdx % 4);

    // Compute the offset in bytes.
    return sfBlkIdx * NumBytesPerSfBlock + sfRowIdx * NumColsPerSfBlock + sfColIdx;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Given the GMEM address of an output element, compute the offset of the corresponding scaling
// factor in the SF tensor. Optionally, a startTokenIndex can be provided if the first token is not
// the start token in the SF tensor. This is useful when inflight batching is enabled in TRT-LLM,
// where the context and generation output are stored as one output tensor. In this case, the
// generation output may not start with zero offset in the SF output tensor.
template <int32_t NumBitsPerElt>
inline __device__ int64_t getSfOffset(int64_t gmemOffsetInBytes, int32_t hiddenDim, int32_t startTokenIdx = 0)
{
    // The number of elements per sf.
    int32_t constexpr NumEltsPerSf = 16;
    // The GMEM offset of the output element.
    int64_t gmemOffset = gmemOffsetInBytes * 8 /*bits*/ / NumBitsPerElt;
    // The row/col indices of the corresponding SF element.
    int32_t sfRowIdx = gmemOffset / hiddenDim + startTokenIdx;
    int32_t sfColIdx = (gmemOffset % hiddenDim) / NumEltsPerSf;
    // Compute the SF offset.
    return getSfOffset(sfRowIdx, sfColIdx, hiddenDim / NumEltsPerSf);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// TODO(tizheng): Refactor to track gmem offset instead of doing pointer subtraction.
template <int32_t NumBitsPerElt>
inline __device__ int64_t getSfOffset(
    void const* gmemOutPtr, void const* gmemBasePtr, int32_t hiddenDim, int32_t startTokenIdx = 0)
{
    return getSfOffset<NumBitsPerElt>(
        reinterpret_cast<char const*>(gmemOutPtr) - reinterpret_cast<char const*>(gmemBasePtr), hiddenDim,
        startTokenIdx);
}

} // namespace dev

// TODO: it would be nice to move some of that logic to Fp4Utils.h
template <tg::SfLayout Layout>
inline __device__ int32_t getSfOffset(int32_t dataRowIdx, int32_t dataBlkColIdx, int32_t numDataBlksPerRow)
{
    if constexpr (Layout == tg::SfLayout::Linear)
    {
        return numDataBlksPerRow * dataRowIdx + dataBlkColIdx;
    }
    else if constexpr (Layout == tg::SfLayout::R128c4)
    {
        return static_cast<int32_t>(dev::getSfOffset(dataRowIdx, dataBlkColIdx, numDataBlksPerRow));
    }
    else if constexpr (Layout == tg::SfLayout::R8c4 || Layout == tg::SfLayout::R8c16)
    {
        static int32_t constexpr NumRowsPerSfBlock = 8;
        static int32_t constexpr NumColsPerSfBlock = (Layout == tg::SfLayout::R8c4) ? 4 : 16;
        static int32_t constexpr NumBytesPerSfBlock = NumRowsPerSfBlock * NumColsPerSfBlock;
        int sfBlkRowIdx = dataRowIdx / NumRowsPerSfBlock;
        int sfBlkColIdx = dataBlkColIdx / NumColsPerSfBlock;
        int sfBlkIdx = sfBlkRowIdx * numDataBlksPerRow / NumColsPerSfBlock + sfBlkColIdx;
        int sfRowIdx = dataRowIdx % NumRowsPerSfBlock;
        int sfColIdx = dataBlkColIdx % NumColsPerSfBlock;
        return sfBlkIdx * NumBytesPerSfBlock + sfRowIdx * NumColsPerSfBlock + sfColIdx;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <tg::SfLayout LayoutSrc, tg::SfLayout LayoutDst, typename KernelParams>
__device__ void convertSfCommon(KernelParams params)
{
    // Note: it's assumed that the number of scaling factors per row is a multiple of 4.
    constexpr int VecSize = 4;
    using VecType = uint32_t;
    static_assert(sizeof(VecType) == VecSize);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // Immediately trigger the secondary kernel when using PDL, then wait on primary.
    if constexpr (KernelParams::UsePdl)
    {
        cudaTriggerProgrammaticLaunchCompletion();
        cudaGridDependencySynchronize();
    }
#endif

    // TODO: consider optimizing if used in production.
    // This is a naive kernel. It's not doing coalesced loads.

    int const numSfPerRow = params.hiddenDimSf;

    for (int tokenIdx = blockIdx.y; tokenIdx < params.numTokens; tokenIdx += gridDim.y)
    {
        for (int hiddenSfVecIdx = threadIdx.x + blockDim.x * blockIdx.x; hiddenSfVecIdx < numSfPerRow / VecSize;
             hiddenSfVecIdx += blockDim.x * gridDim.x)
        {
            // Index of the first SF in the vector.
            int const hiddenSfIdx = VecSize * hiddenSfVecIdx;

            // Load scale factors.
            int sfIdxIn = getSfOffset<LayoutSrc>(tokenIdx, hiddenSfIdx, numSfPerRow);
            const VecType sfVec = reinterpret_cast<VecType const*>(params.inSfPtr)[sfIdxIn / VecSize];

            // Store scale factors.
            int const sfIdxOut = getSfOffset<LayoutDst>(tokenIdx, hiddenSfIdx, numSfPerRow);
            reinterpret_cast<VecType*>(params.outSfPtr)[sfIdxOut / VecSize] = sfVec;
        }
    }
}

#define CONVERT_FP4_SF_KERNEL(LayoutSrc, LayoutDst)                                                                    \
    template <typename KernelParams>                                                                                   \
    __global__ void convertSf##LayoutSrc##To##LayoutDst##Kernel(KernelParams params)                                   \
    {                                                                                                                  \
        convertSfCommon<tg::SfLayout::LayoutSrc, tg::SfLayout::LayoutDst>(params);                                     \
    }
// We only need a conversion to the linear layout.
CONVERT_FP4_SF_KERNEL(R128c4, Linear);
CONVERT_FP4_SF_KERNEL(R8c4, Linear);
CONVERT_FP4_SF_KERNEL(R8c16, Linear);
#undef CONVERT_FP4_SF_KERNEL

////////////////////////////////////////////////////////////////////////////////////////////////////

void run(Data const& data, void* stream)
{
    constexpr int VecSize = 4;
    int const numThreads = 128;
    int const numBlocksX = (data.hiddenDimSf / VecSize - 1 + numThreads) / numThreads;
    int const numBlocksY = std::min(8192, data.numTokens);
    dim3 numBlocks(numBlocksX, numBlocksY);
#define CONVERT_FP4_SF_LAUNCH(LayoutSrc, LayoutDst)                                                                    \
    if (data.sfLayoutSrc == tg::SfLayout::LayoutSrc && data.sfLayoutDst == tg::SfLayout::LayoutDst)                    \
    {                                                                                                                  \
        LAUNCH_PDL(data, false, cutlass::float_e4m3_t, convertSf##LayoutSrc##To##LayoutDst##Kernel, numBlocks,         \
            numThreads, 0, stream);                                                                                    \
        return;                                                                                                        \
    }
    CONVERT_FP4_SF_LAUNCH(R128c4, Linear);
    CONVERT_FP4_SF_LAUNCH(R8c4, Linear);
    CONVERT_FP4_SF_LAUNCH(R8c16, Linear);
#undef CONVERT_FP4_SF_LAUNCH
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace convertsf

namespace permute
{

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = batchedGemm::trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void permuteKernel(KernelParams params)
{
    using Type = typename KernelParams::Type;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // immediately trigger the secondary kernel when using PDL, then wait on primary
    if constexpr (KernelParams::UsePdl)
    {
        cudaTriggerProgrammaticLaunchCompletion();
        cudaGridDependencySynchronize();
    }
#endif

    for (int tokenIdx = blockIdx.y; tokenIdx < params.numTokens; tokenIdx += gridDim.y)
    {
        // Loop over hidden dim
        for (int hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x; hiddenIdx < params.hiddenDim;
             hiddenIdx += blockDim.x * gridDim.x)
        {

            // Load chunk of token into registers
            const Type data = params.inPtr[static_cast<int64_t>(tokenIdx) * params.hiddenDim + hiddenIdx];

            // Write to topK places
            for (int k = 0; k < params.topK; k++)
            {
                int const expandedIdx = tokenIdx * params.topK + k;
                int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];
                params.outPtr[static_cast<int64_t>(permutedIdx) * params.hiddenDim + hiddenIdx] = data;
            }
        }
        if (params.useDeepSeekFp8)
        {
            for (int scaleIdx = threadIdx.x + blockDim.x * blockIdx.x; scaleIdx < params.hiddenDim / 128;
                 scaleIdx += blockDim.x * gridDim.x)
            {
                for (int k = 0; k < params.topK; k++)
                {
                    int const expandedIdx = tokenIdx * params.topK + k;
                    int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];

                    int const idx_in = tokenIdx + params.numTokens * scaleIdx;
                    int const idx_out = permutedIdx + params.totalNumPaddedTokens[0] * scaleIdx;

                    params.outDqSfsPtr[idx_out] = params.inDqSfsPtr[idx_in];
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void run(Data const& data, void* stream)
{
    int const numThreads = 256;
    int const numBlocksX = (data.hiddenDim - 1 + numThreads) / numThreads;
    int const numBlocksY = std::min(8192, data.numTokens);
    dim3 numBlocks(numBlocksX, numBlocksY);

    LAUNCH(data, permuteKernel, numBlocks, numThreads, 0, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace permute

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace finalize
{

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = batchedGemm::trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void finalizeKernel(KernelParams params)
{
    using Type = typename KernelParams::Type;
    using TypeExpW = typename KernelParams::TypeExpW;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // wait on primary kernel when using PDL
    if constexpr (KernelParams::UsePdl)
    {
        cudaGridDependencySynchronize();
    }
#endif

    for (int tokenIdx = blockIdx.y; tokenIdx < params.numTokens; tokenIdx += gridDim.y)
    {
        // Loop over hidden dim
        for (int hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x; hiddenIdx < params.hiddenDim;
             hiddenIdx += blockDim.x * gridDim.x)
        {

            // Accumulate chunk of token into registers
            float data = 0.0F;

            // Write to topK places
            for (int k = 0; k < params.topK; k++)
            {
                int const expandedIdx = tokenIdx * params.topK + k;
                int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];

                if (permutedIdx == -1)
                {
                    continue;
                }

                if (params.expertWeightsPtr != nullptr)
                {
                    TypeExpW const scale = params.expertWeightsPtr[expandedIdx];
                    data += float{scale}
                        * float{params.inPtr[static_cast<int64_t>(permutedIdx) * params.hiddenDimPadded + hiddenIdx]};
                }
                else
                {
                    data += float{params.inPtr[static_cast<int64_t>(permutedIdx) * params.hiddenDimPadded + hiddenIdx]};
                }
            }

            params.outPtr[static_cast<int64_t>(tokenIdx) * params.hiddenDim + hiddenIdx] = static_cast<Type>(data);
        }
    }
}

constexpr static int FINALIZE_THREADS_PER_BLOCK = 256;
constexpr static int FINALIZE_SINGLE_PASS_BF16_THREADS_PER_BLOCK = 320;

__device__ float4 vectorizedLoadPtx(float4 const* ptr)
{
    float4 ret;
    asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w)
                 : "l"(ptr));
    return ret;
}

// Final kernel to unpermute and scale
// This kernel unpermutes the original data, does the k-way reduction and performs the final skip connection.

template <typename KernelParams>
__global__ void finalizeKernelVecLoad(KernelParams params)
{
    using Type = typename KernelParams::Type;
    using TypeExpW = typename KernelParams::TypeExpW;

    int const hiddenDimPaddedBits = params.hiddenDimPadded * cutlass::sizeof_bits<Type>::value;
    int const hiddenDimBits = params.hiddenDim * cutlass::sizeof_bits<Type>::value;
    assert(hiddenDimPaddedBits % 128 == 0);
    assert(hiddenDimBits % 128 == 0);

    // Load 128-bits per thread, according to the smallest data type we read/write
    constexpr int64_t FINALIZE_ELEM_PER_THREAD = 128 / cutlass::sizeof_bits<Type>::value;
    using InputElem = cutlass::Array<Type, FINALIZE_ELEM_PER_THREAD>;
    using OutputElem = cutlass::Array<Type, FINALIZE_ELEM_PER_THREAD>;
    using ComputeElem = cutlass::Array<float, FINALIZE_ELEM_PER_THREAD>;

    int64_t const hiddenBlockIdx = blockIdx.y;
    int64_t const tokenIdx = blockIdx.x;
    int64_t const startOffset = threadIdx.x + hiddenBlockIdx * params.hiddenDimPerBlock / FINALIZE_ELEM_PER_THREAD;
    int64_t const stride = blockDim.x;
    int64_t const numElemsInPaddedCol = params.hiddenDimPadded / FINALIZE_ELEM_PER_THREAD;
    int64_t const numElemsInColPerBlock = (hiddenBlockIdx + 1) * params.hiddenDimPerBlock / FINALIZE_ELEM_PER_THREAD;

    auto const offset = tokenIdx * params.hiddenDim;
    Type* outputPtr = params.outPtr + offset;
    auto* outElemPtr = reinterpret_cast<OutputElem*>(outputPtr);
    auto const* inElemPtr = reinterpret_cast<InputElem const*>(params.inPtr);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // wait on primary kernel when using PDL
    if constexpr (KernelParams::UsePdl)
    {
        cudaGridDependencySynchronize();
    }
#endif

    for (int elemIndex = startOffset; elemIndex < numElemsInColPerBlock; elemIndex += stride)
    {
        ComputeElem threadOutput;
        threadOutput.fill(0);
        for (int k = 0; k < params.topK; ++k)
        {
            int const expandedIdx = tokenIdx * params.topK + k;
            int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];
            if (permutedIdx == -1)
            {
                continue;
            }

            float const scale
                = (params.expertWeightsPtr != nullptr) ? static_cast<float>(params.expertWeightsPtr[expandedIdx]) : 1.f;

            auto const* inputPermutedPtr = inElemPtr + permutedIdx * numElemsInPaddedCol;

            float4 input = vectorizedLoadPtx(reinterpret_cast<float4 const*>(&inputPermutedPtr[elemIndex]));
            InputElem inputPermutedElem = *reinterpret_cast<InputElem const*>(&input);
            ComputeElem expertResult = arrayConvert<InputElem, ComputeElem>(inputPermutedElem);

            threadOutput = threadOutput + scale * expertResult;
        }

        OutputElem outputElem = arrayConvert<ComputeElem, OutputElem>(threadOutput);
        outElemPtr[elemIndex] = outputElem;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void finalizeDeepSeekKernel(KernelParams params)
{
    using Type = typename KernelParams::Type;
    using BlockReduce = cub::BlockReduce<float, 128>;

    __shared__ float s_scaleOut;
    __shared__ typename BlockReduce::TempStorage temp_storage;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // wait on primary kernel when using PDL
    if constexpr (KernelParams::UsePdl)
    {
        cudaGridDependencySynchronize();
    }
#endif

    for (int tokenIdx = blockIdx.y; tokenIdx < params.numTokens; tokenIdx += gridDim.y)
    {
        // Loop over hidden dim
        for (int hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x; hiddenIdx < params.hiddenDim;
             hiddenIdx += blockDim.x * gridDim.x)
        {

            // Accumulate chunk of token into registers
            float acc = 0.0f;

            for (int k = 0; k < params.topK; k++)
            {
                int const expandedIdx = tokenIdx * params.topK + k;
                int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];
                if (permutedIdx == -1)
                {
                    continue;
                }
                int const totalNumPaddedTokens = params.totalNumPaddedTokens[0];
                int const scaleIdx = permutedIdx + totalNumPaddedTokens * (hiddenIdx / 128);
                float const blockScale = params.inDqSfsPtr ? params.inDqSfsPtr[scaleIdx] : 1;

                float const expertProb = (float) params.expertWeightsPtr[tokenIdx * params.topK + k];

                float const scale = expertProb * blockScale;
                acc += scale
                    * static_cast<float>(
                        params.inPtr[static_cast<int64_t>(permutedIdx) * params.hiddenDimPadded + hiddenIdx]);
            }

            // The largest (finite) value that can be represented using E4m3.
            float constexpr E4m3MaxVal{448.f};

            // Compute the absolute max
            float aMax = BlockReduce(temp_storage).Reduce(fabsf(acc), cuda::maximum<>());

            if (threadIdx.x == 0)
            {
                if (params.outDqSfsPtr)
                {
                    // Same all-zero-block hazard as the activation kernels: without the floor
                    // an all-zero accumulator makes the division below evaluate 0 / 0. This
                    // branch is unreachable today because every thop entry point passes
                    // args.output_scale = nullptr, so nothing observable changes; the floor is
                    // here so the first caller to wire up outDqSfsPtr does not inherit it.
                    float const scaleOut = fmaxf(aMax, activation::kDsActAmaxEpsilon) / E4m3MaxVal;
                    s_scaleOut = scaleOut;
                    int const scaleOut_idx = tokenIdx + hiddenIdx / 128 * params.numTokens;
                    params.outDqSfsPtr[scaleOut_idx] = scaleOut;
                }
                else
                {
                    s_scaleOut = 1.0f;
                }
            }
            __syncthreads();
            float const scaleOut = s_scaleOut;
            __syncthreads();
            params.outPtr[static_cast<int64_t>(tokenIdx) * params.hiddenDim + hiddenIdx] = (Type) (acc / scaleOut);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void run(Data const& data, void* stream)
{
    if (data.mUseDeepSeekFp8)
    {
        int const numThreads = 128;
        int const numBlocksX = (data.hiddenDim - 1 + numThreads) / numThreads;
        // Capped at rather arbitrary 8192 to avoid gridDim exceeding 65535 specified by CUDA.
        int const numBlocksY = std::min(8192, data.numTokens);
        dim3 numBlocks(numBlocksX, numBlocksY);

        LAUNCH_EXPW(data, finalizeDeepSeekKernel, false, numBlocks, numThreads, 0, stream);
    }
    else
    {
        int const numThreads = 256;
        int const numBlocksX = (data.hiddenDim - 1 + numThreads) / numThreads;
        // Capped at rather arbitrary 8192 to avoid gridDim exceeding 65535 specified by CUDA.
        int const numBlocksY = std::min(8192, data.numTokens);

        if (numBlocksX * numBlocksY < 1184)
        {
            // The number 1184 comes from 148 * 8, where 148 is the number of SMs (Streaming Multiprocessors) in the
            // Blackwell architecture,
            // and the value 8 means that each Streaming Multiprocessor (SM) can hold up to 8 blocks for this kernel.
            // This limitation is intended to ensure that when the number of waves is greater than 1, we choose to use
            // the kernel with vectorized loading.
            dim3 numBlocks(numBlocksX, numBlocksY);
            LAUNCH_EXPW(data, finalizeKernel, false, numBlocks, numThreads, 0, stream);
        }
        else
        {
            // A 2560-element BF16 row contains 320 128-bit vectors. Use one
            // thread per vector for this high-occupancy Qwen MoE shape so the
            // finalize kernel does not need a second loop iteration.
            int const vectorThreads = data.mDtypeElt == tg::Dtype::Bfloat16 && data.hiddenDim == 2560 && data.topK == 10
                ? FINALIZE_SINGLE_PASS_BF16_THREADS_PER_BLOCK
                : FINALIZE_THREADS_PER_BLOCK;
            LAUNCH_EXPW(data, finalizeKernelVecLoad, true, /*numBlocks=*/data.numTokens,
                /*numThreads=*/vectorThreads, 0, stream);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace finalize

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace moe::dev
