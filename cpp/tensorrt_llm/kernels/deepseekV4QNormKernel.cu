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

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

constexpr int kWarpSize = 32;
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
}

template <typename T>
void dispatchDeepseekV4QNorm(
    void const* input, void* output, int totalRows, int headDim, float eps, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(headDim == 512, "deepseek_v4_q_norm only supports head_dim=512, got %d", headDim);

    dim3 const block(kThreadsPerBlock);
    dim3 const grid((totalRows + kWarpsPerBlock - 1) / kWarpsPerBlock);
    deepseekV4QNormKernel<T, 512>
        <<<grid, block, 0, stream>>>(static_cast<T const*>(input), static_cast<T*>(output), totalRows, eps);
}

// Fused q-norm + FP8 quant of nope segment. Row layout [nope|rope]; writes FP8
// nope (scaled by inv_rms * quant_scale_qkv) to `quant_q_nope` with per-row
// stride `quantQNopeRowStrideBytes`, and bf16/fp16 rope to `q_pe_out`.
// Requires kHeadDim==512, kNopeDim==448, kRopeDim==64 so each lane's
// (kPairsPerLane-1) iterations cover the nope range and the last iteration
// covers the rope range exactly.

// `kFuseRope`: also apply the Q RoPE here and write the rotated rope segment as
// FP8 into the same row of `quant_q_nope`, instead of handing an un-rotated bf16
// `q_pe` to `applyMLARopeAndAssignQKVKernelGeneration`. The layout already suits
// it -- `kRopePairs == kWarpSize` means lane `l` owns exactly rope pair `l`, so
// the rotation is register-local and needs no extra loads beyond one float2 of
// cos/sin. With the q_nope quantize region already gone, this leaves that kernel
// with nothing to do on the DSv4 decode path.
//
// Positions match whichever RoPE kernel this replaces:
//   generation -- uniform query length, so batch = token / seq_len and
//                 position = kv_cache_len[batch] - seq_len + token % seq_len.
//   context    -- ragged, so the sequence owning a token is found by binary search
//                 over `cu_q_seqlens` (in TOKENS here, unlike the generation
//                 `seqQOffset` which counts Q rows), and
//                 position = local_token + (kv_cache_len[b] - current_seq_len).
//                 The second term is the chunked-prefill cached offset.
// Passing `cu_q_seqlens` selects the context form. Every lane of a warp shares
// `row`, so the search is uniform across the warp and does not diverge.
template <typename T, int kHeadDim, int kNopeDim, bool kFuseRope = false>
__global__ void deepseekV4QNormFusedKernel(T const* __restrict__ input, __nv_fp8_e4m3* __restrict__ quant_q_nope,
    T* __restrict__ q_pe_out, float const* __restrict__ quant_scale_qkv_ptr, int totalRows,
    int quantQNopeRowStrideBytes, float eps, float2 const* __restrict__ cos_sin_cache = nullptr,
    int const* __restrict__ cache_seq_lens = nullptr, int num_heads = 0, int seq_len = 0,
    int64_t const* __restrict__ cu_q_seqlens = nullptr, int num_seqs = 0)
{
    static_assert(kHeadDim % (2 * kWarpSize) == 0);
    static_assert(kNopeDim > 0 && kNopeDim < kHeadDim);
    constexpr int kRopeDim = kHeadDim - kNopeDim;
    constexpr int kPairsPerRow = kHeadDim / 2;
    constexpr int kPairsPerLane = kPairsPerRow / kWarpSize;
    constexpr int kNopePairs = kNopeDim / 2;
    constexpr int kRopePairs = kRopeDim / 2;
    static_assert(kPairsPerLane >= 2);
    static_assert(kNopePairs == (kPairsPerLane - 1) * kWarpSize,
        "Fused kernel assumes the last per-lane iteration covers the rope segment.");
    static_assert(kRopePairs == kWarpSize, "Each lane should own exactly one rope pair.");

    using Vec2 = typename Vec2Traits<T>::Type;

    int const warpId = threadIdx.x / kWarpSize;
    int const laneId = threadIdx.x % kWarpSize;
    int const row = blockIdx.x * kWarpsPerBlock + warpId;

    if (row >= totalRows)
    {
        return;
    }

    auto const* inputPair = reinterpret_cast<Vec2 const*>(input + static_cast<int64_t>(row) * kHeadDim);
    // Nope output: row stride is caller-controlled (kNopeDim for packed, kHeadDim
    // when interleaved with the rope segment of a full Q-buffer that RoPE writes).
    auto* nopeOutPair = reinterpret_cast<__nv_fp8x2_e4m3*>(
        reinterpret_cast<__nv_fp8_e4m3*>(quant_q_nope) + static_cast<int64_t>(row) * quantQNopeRowStrideBytes);
    auto* ropeOutPair = reinterpret_cast<Vec2*>(q_pe_out + static_cast<int64_t>(row) * kRopeDim);

    float const quantScale = quant_scale_qkv_ptr ? quant_scale_qkv_ptr[0] : 1.0F;

    float2 values[kPairsPerLane];
    float sumSquares = 0.0F;

#pragma unroll
    for (int i = 0; i < kPairsPerLane; ++i)
    {
        int const pairIdx = i * kWarpSize + laneId;
        values[i] = Vec2Traits<T>::toFloat2(inputPair[pairIdx]);
        sumSquares += values[i].x * values[i].x + values[i].y * values[i].y;
    }

    sumSquares = warpReduceSum(sumSquares);
    float const normScale = rsqrtf(sumSquares / static_cast<float>(kHeadDim) + eps);
    float const fp8Scale = normScale * quantScale;

    // First kPairsPerLane-1 iters land in the nope range -> FP8 STG.
#pragma unroll
    for (int i = 0; i < kPairsPerLane - 1; ++i)
    {
        int const pairIdx = i * kWarpSize + laneId;
        float2 scaled{values[i].x * fp8Scale, values[i].y * fp8Scale};
        nopeOutPair[pairIdx] = __nv_fp8x2_e4m3(scaled);
    }

    // Last iter is the rope segment.
    {
        constexpr int i = kPairsPerLane - 1;
        int const pairIdx = i * kWarpSize + laneId;   // in [kNopePairs, kPairsPerRow)
        int const ropePairIdx = pairIdx - kNopePairs; // in [0, kRopePairs)
        float2 normalized{values[i].x * normScale, values[i].y * normScale};
        if constexpr (kFuseRope)
        {
            // Rows are (token, head) pairs, and the position depends on the token
            // alone. Same derivation the generation RoPE kernel uses.
            int const token = row / num_heads;
            int positionId;
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
                int const seqBegin = static_cast<int>(cu_q_seqlens[lo]);
                int const currentSeqLen = static_cast<int>(cu_q_seqlens[lo + 1]) - seqBegin;
                positionId = (token - seqBegin) + (cache_seq_lens[lo] - currentSeqLen);
            }
            else
            {
                int const batchIdx = token / seq_len;
                positionId = cache_seq_lens[batchIdx] - seq_len + (token % seq_len);
            }
            // The cos/sin table is float2 (cos, sin) with a stride of kRopeDim per
            // position; GPT-J style rotation pairs adjacent elements, so pair p
            // reads entry p.
            float2 const coef = cos_sin_cache[static_cast<int64_t>(kRopeDim) * positionId + ropePairIdx];
            float2 const rotated{
                coef.x * normalized.x - coef.y * normalized.y, coef.x * normalized.y + coef.y * normalized.x};
            // Straight into the rope slots of the same FP8 row the nope segment
            // just filled, so the whole Q row is complete when this kernel exits.
            float2 const scaled{rotated.x * quantScale, rotated.y * quantScale};
            nopeOutPair[kNopePairs + ropePairIdx] = __nv_fp8x2_e4m3(scaled);
        }
        else
        {
            // bf16/fp16 STG, no extra quant scale; the RoPE kernel rotates it later.
            ropeOutPair[ropePairIdx] = Vec2Traits<T>::fromFloat2(normalized);
        }
    }
}

template <typename T>
void dispatchDeepseekV4QNormFused(void const* input, void* quant_q_nope, void* q_pe_out,
    void const* quant_scale_qkv_ptr, int totalRows, int headDim, int nopeDim, int quantQNopeRowStrideBytes, float eps,
    void const* cos_sin_cache, int const* cache_seq_lens, int num_heads, int seq_len, int64_t const* cu_q_seqlens,
    int num_seqs, cudaStream_t stream)
{
    assert(headDim == 512);
    assert(nopeDim == 448);
    assert(quantQNopeRowStrideBytes >= nopeDim);

    dim3 const block(kThreadsPerBlock);
    dim3 const grid((totalRows + kWarpsPerBlock - 1) / kWarpsPerBlock);
    // Fusing the RoPE needs the rope slots of the same row, so the caller must be
    // using the interleaved layout, and needs positions, which only generation has.
    // Context supplies cu_q_seqlens and needs no uniform seq_len; generation
    // supplies seq_len and no cu_q_seqlens. Either way the rope slots must live in
    // this row, i.e. the interleaved layout.
    bool const haveRopePositions = cu_q_seqlens != nullptr ? num_seqs > 0 : seq_len > 0;
    bool const fuseRope = cos_sin_cache != nullptr && cache_seq_lens != nullptr && num_heads > 0 && haveRopePositions
        && quantQNopeRowStrideBytes == headDim;
    if (fuseRope)
    {
        deepseekV4QNormFusedKernel<T, 512, 448, true><<<grid, block, 0, stream>>>(static_cast<T const*>(input),
            static_cast<__nv_fp8_e4m3*>(quant_q_nope), static_cast<T*>(q_pe_out),
            static_cast<float const*>(quant_scale_qkv_ptr), totalRows, quantQNopeRowStrideBytes, eps,
            static_cast<float2 const*>(cos_sin_cache), cache_seq_lens, num_heads, seq_len, cu_q_seqlens, num_seqs);
    }
    else
    {
        deepseekV4QNormFusedKernel<T, 512, 448, false><<<grid, block, 0, stream>>>(static_cast<T const*>(input),
            static_cast<__nv_fp8_e4m3*>(quant_q_nope), static_cast<T*>(q_pe_out),
            static_cast<float const*>(quant_scale_qkv_ptr), totalRows, quantQNopeRowStrideBytes, eps);
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
    int64_t const* cu_q_seqlens, int num_seqs, cudaStream_t stream)
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
