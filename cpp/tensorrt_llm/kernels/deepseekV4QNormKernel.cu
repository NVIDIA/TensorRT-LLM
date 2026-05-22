#include "tensorrt_llm/kernels/deepseekV4QNormKernel.h"

#include <cassert>
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
    assert(headDim == 512);

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

template <typename T, int kHeadDim, int kNopeDim>
__global__ void deepseekV4QNormFusedKernel(T const* __restrict__ input, __nv_fp8_e4m3* __restrict__ quant_q_nope,
    T* __restrict__ q_pe_out, float const* __restrict__ quant_scale_qkv_ptr, int totalRows,
    int quantQNopeRowStrideBytes, float eps)
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

    // First kPairsPerLane-1 iters land in the nope range -> FP8 STG.
    //
    // We deliberately round the post-norm value to T (bf16 or half) BEFORE
    // applying quantScale so the FP8 cast sees the same input as the legacy
    // two-kernel path (norm-bf16-store → reload → quant-scale → FP8). Skipping
    // this round produces FP8 output that is mathematically more accurate but
    // diverges from the bf16-trained MoE router; the resulting top-K drift
    // costs ~9.5pp on GSM8K for DSv4-Flash and unbalances expert dispatch.
#pragma unroll
    for (int i = 0; i < kPairsPerLane - 1; ++i)
    {
        int const pairIdx = i * kWarpSize + laneId;
        auto const rounded = Vec2Traits<T>::fromFloat2(float2{values[i].x * normScale, values[i].y * normScale});
        float2 reloaded = Vec2Traits<T>::toFloat2(rounded);
        float2 scaled{reloaded.x * quantScale, reloaded.y * quantScale};
        nopeOutPair[pairIdx] = __nv_fp8x2_e4m3(scaled);
    }

    // Last iter is the rope segment -> bf16/fp16 STG (no extra quant scale).
    {
        constexpr int i = kPairsPerLane - 1;
        int const pairIdx = i * kWarpSize + laneId;   // in [kNopePairs, kPairsPerRow)
        int const ropePairIdx = pairIdx - kNopePairs; // in [0, kRopePairs)
        float2 normalized{values[i].x * normScale, values[i].y * normScale};
        ropeOutPair[ropePairIdx] = Vec2Traits<T>::fromFloat2(normalized);
    }
}

template <typename T>
void dispatchDeepseekV4QNormFused(void const* input, void* quant_q_nope, void* q_pe_out,
    void const* quant_scale_qkv_ptr, int totalRows, int headDim, int nopeDim, int quantQNopeRowStrideBytes, float eps,
    cudaStream_t stream)
{
    assert(headDim == 512);
    assert(nopeDim == 448);
    assert(quantQNopeRowStrideBytes >= nopeDim);

    dim3 const block(kThreadsPerBlock);
    dim3 const grid((totalRows + kWarpsPerBlock - 1) / kWarpsPerBlock);
    deepseekV4QNormFusedKernel<T, 512, 448><<<grid, block, 0, stream>>>(static_cast<T const*>(input),
        static_cast<__nv_fp8_e4m3*>(quant_q_nope), static_cast<T*>(q_pe_out),
        static_cast<float const*>(quant_scale_qkv_ptr), totalRows, quantQNopeRowStrideBytes, eps);
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
    bool isBfloat16, float eps, cudaStream_t stream)
{
    if (totalRows == 0)
    {
        return;
    }

    if (isBfloat16)
    {
        dispatchDeepseekV4QNormFused<__nv_bfloat16>(input, quant_q_nope, q_pe_out, quant_scale_qkv_ptr, totalRows,
            headDim, nopeDim, quantQNopeRowStrideBytes, eps, stream);
    }
    else
    {
        dispatchDeepseekV4QNormFused<half>(input, quant_q_nope, q_pe_out, quant_scale_qkv_ptr, totalRows, headDim,
            nopeDim, quantQNopeRowStrideBytes, eps, stream);
    }
}

} // namespace kernels

TRTLLM_NAMESPACE_END
