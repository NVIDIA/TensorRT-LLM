#include "tensorrt_llm/kernels/deepseekV4QNormKernel.h"

#include <cassert>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
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

} // namespace kernels

TRTLLM_NAMESPACE_END
