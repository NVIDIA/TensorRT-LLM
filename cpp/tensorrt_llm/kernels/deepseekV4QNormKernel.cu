/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/deepseekV4QNormKernel.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/envUtils.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <type_traits>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

constexpr int kWarpSize = 32;

// FP8 is one byte per element, so an N-element vector is an N-byte store.
template <int BYTES>
struct Fp8VecStore;

template <>
struct Fp8VecStore<8>
{
    using Type = uint2;
};

template <>
struct Fp8VecStore<16>
{
    using Type = uint4;
};

constexpr int kWarpsPerBlock = 4;
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;

template <typename T>
struct Vec2Traits;

template <>
struct Vec2Traits<half>
{
    using Type = half2;

    __device__ static float2 toFloat2(Type value)
    {
        return __half22float2(value);
    }

    __device__ static Type fromFloat2(float2 value)
    {
        return __floats2half2_rn(value.x, value.y);
    }
};

template <>
struct Vec2Traits<__nv_bfloat16>
{
    using Type = __nv_bfloat162;

    __device__ static float2 toFloat2(Type value)
    {
        return __bfloat1622float2(value);
    }

    __device__ static Type fromFloat2(float2 value)
    {
        return __floats2bfloat162_rn(value.x, value.y);
    }
};

__device__ __forceinline__ float warpReduceSum(float value)
{
    for (int mask = kWarpSize / 2; mask > 0; mask >>= 1)
    {
        value += __shfl_xor_sync(0xFFFFFFFF, value, mask);
    }
    return value;
}

template <typename T, int kHeadDim>
__global__ void deepseekV4QNormKernel(T const* input, T* output, int totalRows, float eps)
{
    static_assert(kHeadDim % (2 * kWarpSize) == 0);
    constexpr int kPairsPerRow = kHeadDim / 2;
    constexpr int kPairsPerLane = kPairsPerRow / kWarpSize;

    using Vec2 = typename Vec2Traits<T>::Type;

    int const warpId = threadIdx.x / kWarpSize;
    int const laneId = threadIdx.x % kWarpSize;
    int const row = blockIdx.x * kWarpsPerBlock + warpId;

    if (row >= totalRows)
    {
        return;
    }

    auto const* inputPair = reinterpret_cast<Vec2 const*>(input + static_cast<int64_t>(row) * kHeadDim);
    auto* outputPair = reinterpret_cast<Vec2*>(output + static_cast<int64_t>(row) * kHeadDim);

    float2 values[kPairsPerLane];
    float sumSquares = 0.0F;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

#pragma unroll
    for (int i = 0; i < kPairsPerLane; ++i)
    {
        int const pairIdx = i * kWarpSize + laneId;
        values[i] = Vec2Traits<T>::toFloat2(inputPair[pairIdx]);
        sumSquares += values[i].x * values[i].x + values[i].y * values[i].y;
    }

    sumSquares = warpReduceSum(sumSquares);
    float const scale = rsqrtf(sumSquares / static_cast<float>(kHeadDim) + eps);

#pragma unroll
    for (int i = 0; i < kPairsPerLane; ++i)
    {
        int const pairIdx = i * kWarpSize + laneId;
        float2 value{values[i].x * scale, values[i].y * scale};
        outputPair[pairIdx] = Vec2Traits<T>::fromFloat2(value);
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename T>
void dispatchDeepseekV4QNorm(
    void const* input, void* output, int totalRows, int headDim, float eps, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(headDim == 512, "deepseek_v4_q_norm only supports head_dim=512, got %d", headDim);

    dim3 const block(kThreadsPerBlock);
    dim3 const grid((totalRows + kWarpsPerBlock - 1) / kWarpsPerBlock);
    tensorrt_llm::common::launchWithPdlWhenEnabled("deepseekV4QNorm", deepseekV4QNormKernel<T, 512>, grid, block, 0,
        stream, static_cast<T const*>(input), static_cast<T*>(output), totalRows, eps);
}

// Fused q-norm + FP8 quant of nope segment. Row layout [nope|rope]; writes FP8
// nope (scaled by inv_rms * quant_scale_qkv) to `quant_q_nope` with per-row
// stride `quantQNopeRowStrideBytes`, and bf16/fp16 rope to `q_pe_out`.
// Requires kHeadDim==512, kNopeDim==448, kRopeDim==64 so each lane's
// (kPairsPerLane-1) iterations cover the nope range and the last iteration
// covers the rope range exactly.

// `kFuseRope`: rotate the rope segment here too and store it FP8 in the same row.
// Register-local -- `kRopePairs == kWarpSize`, so lane `l` owns rope pair `l` and
// needs only one float2 of cos/sin.
//
// Position, by phase (`cu_q_seqlens` selects context):
//   generation: batch = token / seq_len, pos = kv_cache_len[batch] - seq_len + token % seq_len
//   context:    binary search `cu_q_seqlens` (in TOKENS, unlike the generation
//               `seqQOffset` which counts Q rows), pos = local_token + cached_offset
// `row` is warp-uniform, so the search does not diverge.
template <typename T, int kHeadDim, int kNopeDim, bool kFuseRope = false, bool kWideVec = true>
__global__ void deepseekV4QNormFusedKernel(T const* __restrict__ input, __nv_fp8_e4m3* __restrict__ quant_q_nope,
    T* __restrict__ q_pe_out, float const* __restrict__ quant_scale_qkv_ptr, int totalRows,
    int quantQNopeRowStrideBytes, float eps, float2 const* __restrict__ cos_sin_cache = nullptr,
    int const* __restrict__ cache_seq_lens = nullptr, int num_heads = 0, int seq_len = 0,
    int const* __restrict__ cu_q_seqlens = nullptr, int num_seqs = 0, int num_heads_shift = -1, int seq_len_shift = -1)
{
    static_assert(kHeadDim % (2 * kWarpSize) == 0);
    static_assert(kNopeDim > 0 && kNopeDim < kHeadDim);
    constexpr int kRopeDim = kHeadDim - kNopeDim;
    constexpr int kPairsPerRow = kHeadDim / 2;
    constexpr int kPairsPerLane = kPairsPerRow / kWarpSize;
    constexpr int kNopePairs = kNopeDim / 2;
    static_assert(kPairsPerLane >= 2);

    // Row decomposition, per phase:
    //   narrow (kWideVec=false): lane owns kPairsPerLane pairs -> 4B loads, 2B stores
    //   wide   (kWideVec=true):  lane owns kVecsPerLane vectors -> 16B loads, 8B stores
    //
    // Wide costs ~3 registers, dropping occupancy 16 -> 14 blocks/SM. Context is
    // bandwidth-bound (1.6 GB/launch, ~81% HBM SOL) so waves matter more than
    // instructions: forcing it wide measured 248 -> 277 us. Generation moves 6 MB at
    // ~25% SOL, instruction-bound, so wide wins there (3.10 -> 2.77 us).
    constexpr int kEltsPerVec = 16 / sizeof(T);
    constexpr int kPairsPerVec = kEltsPerVec / 2;
    constexpr int kRowVecs = kHeadDim / kEltsPerVec;
    constexpr int kNopeVecs = kNopeDim / kEltsPerVec;
    constexpr int kVecsPerLane = kRowVecs / kWarpSize;
    static_assert(!kWideVec || kHeadDim % kEltsPerVec == 0, "Row must split into whole 16B vectors.");
    static_assert(!kWideVec || kNopeDim % kEltsPerVec == 0, "A 16B vector must not straddle nope/rope.");
    static_assert(!kWideVec || kRowVecs % kWarpSize == 0, "Row vectors must split evenly across a warp.");
    static_assert(kWideVec || kNopePairs == (kPairsPerLane - 1) * kWarpSize,
        "Narrow path assumes the last per-lane iteration covers the rope segment.");
    static_assert(kWideVec || kRopeDim / 2 == kWarpSize, "Narrow path gives each lane one rope pair.");

    using Vec2 = typename Vec2Traits<T>::Type;

    int const warpId = threadIdx.x / kWarpSize;
    int const laneId = threadIdx.x % kWarpSize;
    int const row = blockIdx.x * kWarpsPerBlock + warpId;

    if (row >= totalRows)
    {
        return;
    }

    T const* rowPtr = input + static_cast<int64_t>(row) * kHeadDim;
    // Nope output: row stride is caller-controlled (kNopeDim for packed, kHeadDim
    // when interleaved with the rope segment of a full Q-buffer that RoPE writes).
    auto* nopeOut
        = reinterpret_cast<__nv_fp8_e4m3*>(quant_q_nope) + static_cast<int64_t>(row) * quantQNopeRowStrideBytes;
    auto* nopeOutPair = reinterpret_cast<__nv_fp8x2_e4m3*>(nopeOut);
    auto* ropeOutPair = reinterpret_cast<Vec2*>(q_pe_out + static_cast<int64_t>(row) * kRopeDim);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    float const quantScale = quant_scale_qkv_ptr ? quant_scale_qkv_ptr[0] : 1.0F;

    // Wide keeps RAW vectors (8 regs) and re-converts in pass 2 instead of holding
    // the float expansion (16 regs) across the reduction. The converts are nearly
    // free; the registers are not -- 32 regs/thread is exactly the occupancy ceiling
    // at 128 threads/block, so anything above it costs waves.
    uint4 raw[kWideVec ? kVecsPerLane : 1];
    float2 values[kWideVec ? 1 : kPairsPerLane];
    float sumSquares = 0.0F;

    if constexpr (kWideVec)
    {
        auto const* inputVec = reinterpret_cast<uint4 const*>(rowPtr);
#pragma unroll
        for (int i = 0; i < kVecsPerLane; ++i)
        {
            raw[i] = inputVec[i * kWarpSize + laneId];
            auto const* pairs = reinterpret_cast<Vec2 const*>(&raw[i]);
#pragma unroll
            for (int j = 0; j < kPairsPerVec; ++j)
            {
                float2 const v = Vec2Traits<T>::toFloat2(pairs[j]);
                sumSquares += v.x * v.x + v.y * v.y;
            }
        }
    }
    else
    {
        auto const* inputPair = reinterpret_cast<Vec2 const*>(rowPtr);
#pragma unroll
        for (int i = 0; i < kPairsPerLane; ++i)
        {
            values[i] = Vec2Traits<T>::toFloat2(inputPair[i * kWarpSize + laneId]);
            sumSquares += values[i].x * values[i].x + values[i].y * values[i].y;
        }
    }

    sumSquares = warpReduceSum(sumSquares);
    float const normScale = rsqrtf(sumSquares / static_cast<float>(kHeadDim) + eps);
    float const fp8Scale = normScale * quantScale;

    // Position depends on the token, which every lane of the warp shares, so this is
    // warp-uniform and hoisted out of the store loop.
    int positionId = 0;
    if constexpr (kFuseRope)
    {
        int const token = num_heads_shift >= 0 ? (row >> num_heads_shift) : (row / num_heads);
        if (cu_q_seqlens != nullptr)
        {
            int lo = 0;
            int hi = num_seqs - 1;
            while (lo < hi)
            {
                int const mid = (lo + hi + 1) >> 1;
                if (cu_q_seqlens[mid] <= token)
                {
                    lo = mid;
                }
                else
                {
                    hi = mid - 1;
                }
            }
            int const seqBegin = cu_q_seqlens[lo];
            int const currentSeqLen = cu_q_seqlens[lo + 1] - seqBegin;
            positionId = (token - seqBegin) + (cache_seq_lens[lo] - currentSeqLen);
        }
        else
        {
            int const batchIdx = seq_len_shift >= 0 ? (token >> seq_len_shift) : (token / seq_len);
            int const localToken = seq_len_shift >= 0 ? (token & (seq_len - 1)) : (token % seq_len);
            positionId = cache_seq_lens[batchIdx] - seq_len + localToken;
        }
    }

    if constexpr (kWideVec)
    {
        // One store word per vector, written in place.
        // 16-byte aligned: the non-fused-rope branch below reads this through a
        // `uint4*`, but the members only imply `alignof` 8.
        union __align__(16) StoreWord
        {
            typename Fp8VecStore<kEltsPerVec>::Type packed;
            __nv_fp8x2_e4m3 pairs[kPairsPerVec];
            Vec2 raw_pairs[kPairsPerVec];
        };

#pragma unroll
        for (int i = 0; i < kVecsPerLane; ++i)
        {
            int const vecIdx = i * kWarpSize + laneId;
            auto const* srcPairs = reinterpret_cast<Vec2 const*>(&raw[i]);
            StoreWord out;

            if (vecIdx < kNopeVecs)
            {
#pragma unroll
                for (int j = 0; j < kPairsPerVec; ++j)
                {
                    float2 const v = Vec2Traits<T>::toFloat2(srcPairs[j]);
                    out.pairs[j] = __nv_fp8x2_e4m3(float2{v.x * fp8Scale, v.y * fp8Scale});
                }
                *reinterpret_cast<typename Fp8VecStore<kEltsPerVec>::Type*>(nopeOut + vecIdx * kEltsPerVec)
                    = out.packed;
                continue;
            }

            int const ropePairBase = (vecIdx - kNopeVecs) * kPairsPerVec;
            if constexpr (kFuseRope)
            {
#pragma unroll
                for (int j = 0; j < kPairsPerVec; ++j)
                {
                    float2 const v = Vec2Traits<T>::toFloat2(srcPairs[j]);
                    float2 const normalized{v.x * normScale, v.y * normScale};
                    float2 const coef = cos_sin_cache[static_cast<int64_t>(kRopeDim) * positionId + ropePairBase + j];
                    float2 const rotated{
                        coef.x * normalized.x - coef.y * normalized.y, coef.x * normalized.y + coef.y * normalized.x};
                    out.pairs[j] = __nv_fp8x2_e4m3(float2{rotated.x * quantScale, rotated.y * quantScale});
                }
                *reinterpret_cast<typename Fp8VecStore<kEltsPerVec>::Type*>(nopeOut + vecIdx * kEltsPerVec)
                    = out.packed;
            }
            else
            {
#pragma unroll
                for (int j = 0; j < kPairsPerVec; ++j)
                {
                    float2 const v = Vec2Traits<T>::toFloat2(srcPairs[j]);
                    out.raw_pairs[j] = Vec2Traits<T>::fromFloat2(float2{v.x * normScale, v.y * normScale});
                }
                *reinterpret_cast<uint4*>(ropeOutPair + ropePairBase) = *reinterpret_cast<uint4 const*>(&out);
            }
        }
    }
    else
    {
        // First kPairsPerLane-1 iters land in the nope range -> FP8 STG.
#pragma unroll
        for (int i = 0; i < kPairsPerLane - 1; ++i)
        {
            int const pairIdx = i * kWarpSize + laneId;
            float2 const scaled{values[i].x * fp8Scale, values[i].y * fp8Scale};
            nopeOutPair[pairIdx] = __nv_fp8x2_e4m3(scaled);
        }

        // Last iter is the rope segment.
        {
            constexpr int i = kPairsPerLane - 1;
            int const pairIdx = i * kWarpSize + laneId;   // in [kNopePairs, kPairsPerRow)
            int const ropePairIdx = pairIdx - kNopePairs; // in [0, kRopeDim/2)
            float2 const normalized{values[i].x * normScale, values[i].y * normScale};
            if constexpr (kFuseRope)
            {
                float2 const coef = cos_sin_cache[static_cast<int64_t>(kRopeDim) * positionId + ropePairIdx];
                float2 const rotated{
                    coef.x * normalized.x - coef.y * normalized.y, coef.x * normalized.y + coef.y * normalized.x};
                nopeOutPair[kNopePairs + ropePairIdx]
                    = __nv_fp8x2_e4m3(float2{rotated.x * quantScale, rotated.y * quantScale});
            }
            else
            {
                ropeOutPair[ropePairIdx] = Vec2Traits<T>::fromFloat2(normalized);
            }
        }
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename T>
void dispatchDeepseekV4QNormFused(void const* input, void* quant_q_nope, void* q_pe_out,
    void const* quant_scale_qkv_ptr, int totalRows, int headDim, int nopeDim, int quantQNopeRowStrideBytes, float eps,
    void const* cos_sin_cache, int const* cache_seq_lens, int num_heads, int seq_len, int const* cu_q_seqlens,
    int num_seqs, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(headDim == 512, "deepseekV4QNormFused only supports head_dim=512, got %d", headDim);
    TLLM_CHECK_WITH_INFO(nopeDim == 448, "deepseekV4QNormFused only supports nope_dim=448, got %d", nopeDim);
    TLLM_CHECK_WITH_INFO(quantQNopeRowStrideBytes >= nopeDim,
        "quantQNopeRowStrideBytes (%d) must be at least nope_dim (%d)", quantQNopeRowStrideBytes, nopeDim);

    dim3 const block(kThreadsPerBlock);
    dim3 const grid((totalRows + kWarpsPerBlock - 1) / kWarpsPerBlock);
    // Fusing the RoPE needs the rope slots in this row (interleaved layout) plus
    // positions: context passes cu_q_seqlens, generation passes seq_len.
    // `row / num_heads` and `token / seq_len` divide by runtime values, costing an
    // emulated 32-bit divide per row. Both are powers of two in every DSv4 config
    // (128 heads; seq_len = 1 + MTP depth), so pass a shift; -1 keeps the divide.
    auto const pow2_shift = [](int v) -> int
    {
        if (v <= 0 || (v & (v - 1)) != 0)
        {
            return -1;
        }
        int shift = 0;
        while ((1 << shift) < v)
        {
            ++shift;
        }
        return shift;
    };
    int const num_heads_shift = pow2_shift(num_heads);
    int const seq_len_shift = pow2_shift(seq_len);

    bool const haveRopePositions = cu_q_seqlens != nullptr ? num_seqs > 0 : seq_len > 0;
    bool const fuseRope = cos_sin_cache != nullptr && cache_seq_lens != nullptr && num_heads > 0 && haveRopePositions
        && quantQNopeRowStrideBytes == headDim;
    // Context is bandwidth-bound (narrow, more occupancy), generation is
    // instruction-bound (wide). `cu_q_seqlens` marks context.
    bool const wideVec = cu_q_seqlens == nullptr;

    auto launch = [&](auto fuse_tag, auto wide_tag)
    {
        constexpr bool kFuse = decltype(fuse_tag)::value;
        constexpr bool kWide = decltype(wide_tag)::value;
        if constexpr (kFuse)
        {
            tensorrt_llm::common::launchWithPdlWhenEnabled("deepseekV4QNormFused",
                deepseekV4QNormFusedKernel<T, 512, 448, true, kWide>, grid, block, 0, stream,
                static_cast<T const*>(input), static_cast<__nv_fp8_e4m3*>(quant_q_nope), static_cast<T*>(q_pe_out),
                static_cast<float const*>(quant_scale_qkv_ptr), totalRows, quantQNopeRowStrideBytes, eps,
                static_cast<float2 const*>(cos_sin_cache), cache_seq_lens, num_heads, seq_len, cu_q_seqlens, num_seqs,
                num_heads_shift, seq_len_shift);
        }
        else
        {
            tensorrt_llm::common::launchWithPdlWhenEnabled("deepseekV4QNormFused",
                deepseekV4QNormFusedKernel<T, 512, 448, false, kWide>, grid, block, 0, stream,
                static_cast<T const*>(input), static_cast<__nv_fp8_e4m3*>(quant_q_nope), static_cast<T*>(q_pe_out),
                static_cast<float const*>(quant_scale_qkv_ptr), totalRows, quantQNopeRowStrideBytes, eps,
                static_cast<float2 const*>(nullptr), static_cast<int const*>(nullptr), 0, 0,
                static_cast<int const*>(nullptr), 0, -1, -1);
        }
    };

    if (fuseRope)
    {
        wideVec ? launch(std::true_type{}, std::true_type{}) : launch(std::true_type{}, std::false_type{});
    }
    else
    {
        wideVec ? launch(std::false_type{}, std::true_type{}) : launch(std::false_type{}, std::false_type{});
    }
}

} // namespace

void invokeDeepseekV4QNorm(
    void const* input, void* output, int totalRows, int headDim, bool isBfloat16, float eps, cudaStream_t stream)
{
    if (totalRows == 0)
    {
        return;
    }

    if (isBfloat16)
    {
        dispatchDeepseekV4QNorm<__nv_bfloat16>(input, output, totalRows, headDim, eps, stream);
    }
    else
    {
        dispatchDeepseekV4QNorm<half>(input, output, totalRows, headDim, eps, stream);
    }
}

void invokeDeepseekV4QNormFusedFp8(void const* input, void* quant_q_nope, void* q_pe_out,
    void const* quant_scale_qkv_ptr, int totalRows, int headDim, int nopeDim, int quantQNopeRowStrideBytes,
    bool isBfloat16, float eps, void const* cos_sin_cache, int const* cache_seq_lens, int num_heads, int seq_len,
    int const* cu_q_seqlens, int num_seqs, cudaStream_t stream)
{
    if (totalRows == 0)
    {
        return;
    }

    if (isBfloat16)
    {
        dispatchDeepseekV4QNormFused<__nv_bfloat16>(input, quant_q_nope, q_pe_out, quant_scale_qkv_ptr, totalRows,
            headDim, nopeDim, quantQNopeRowStrideBytes, eps, cos_sin_cache, cache_seq_lens, num_heads, seq_len,
            cu_q_seqlens, num_seqs, stream);
    }
    else
    {
        dispatchDeepseekV4QNormFused<half>(input, quant_q_nope, q_pe_out, quant_scale_qkv_ptr, totalRows, headDim,
            nopeDim, quantQNopeRowStrideBytes, eps, cos_sin_cache, cache_seq_lens, num_heads, seq_len, cu_q_seqlens,
            num_seqs, stream);
    }
}

} // namespace kernels

TRTLLM_NAMESPACE_END
