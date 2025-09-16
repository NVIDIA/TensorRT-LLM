/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
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

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace tensorrt_llm
{
namespace kernels
{

///////////////////////////////////////////////////////////////////////////////////////////////////

// Helper functions to convert float2 to half, bfloat16, and e4m3.

inline __device__ uint32_t convert_float2_to_half(float a, float b)
{
    uint32_t output;
    reinterpret_cast<__half2&>(output) = __float22half2_rn(make_float2(a, b));
    return output;
}

inline __device__ uint32_t convert_float2_to_bfloat16(float a, float b)
{
    uint32_t output;
    reinterpret_cast<__nv_bfloat162&>(output) = __float22bfloat162_rn(make_float2(a, b));
    return output;
}

inline __device__ uint32_t convert_float4_to_e4m3(float a, float b, float c, float d)
{
    uint32_t output;
    reinterpret_cast<__nv_fp8x4_e4m3&>(output) = __nv_fp8x4_e4m3(make_float4(a, b, c, d));
    return output;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// Helper functions for float2 mul and fma.

inline __device__ void mul(float2& c, float2 const& a, float2 const& b)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    asm volatile("mul.f32x2 %0, %1, %2;\n"
                 : "=l"(reinterpret_cast<uint64_t&>(c))
                 : "l"(reinterpret_cast<uint64_t const&>(a)), "l"(reinterpret_cast<uint64_t const&>(b)));
#else
    c.x = a.x * b.x;
    c.y = a.y * b.y;
#endif
}

inline __device__ void fma(float2& d, float2 const& a, float2 const& b, float2 const& c)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    asm volatile("fma.rn.f32x2 %0, %1, %2, %3;\n"
                 : "=l"(reinterpret_cast<uint64_t&>(d))
                 : "l"(reinterpret_cast<uint64_t const&>(a)), "l"(reinterpret_cast<uint64_t const&>(b)),
                 "l"(reinterpret_cast<uint64_t const&>(c)));
#else
    d.x = a.x * b.x + c.x;
    d.y = a.y * b.y + c.y;
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Dtype, int32_t NumElts>
inline __device__ void convertAndStoreToGmem(char* gmemPtr, float (&input)[NumElts])
{
    static_assert(sizeof(Dtype) == 0, "Not implemented.");
}

template <typename Dtype, int32_t NumElts>
inline __device__ void convertAndStoreToGmem(
    char* gmemPtr, char* oSfPtr, float (&input)[NumElts], float sfScale, bool isValidRow)
{
    static_assert(sizeof(Dtype) == 0, "Not implemented.");
}

template <>
inline __device__ void convertAndStoreToGmem<__half, 8>(char* gmemPtr, float (&input)[8])
{
    uint4 output;
    output.x = convert_float2_to_half(input[0], input[1]);
    output.y = convert_float2_to_half(input[2], input[3]);
    output.z = convert_float2_to_half(input[4], input[5]);
    output.w = convert_float2_to_half(input[6], input[7]);
    *reinterpret_cast<uint4*>(gmemPtr) = output;
}

template <>
inline __device__ void convertAndStoreToGmem<__nv_bfloat16, 8>(char* gmemPtr, float (&input)[8])
{
    uint4 output;
    output.x = convert_float2_to_bfloat16(input[0], input[1]);
    output.y = convert_float2_to_bfloat16(input[2], input[3]);
    output.z = convert_float2_to_bfloat16(input[4], input[5]);
    output.w = convert_float2_to_bfloat16(input[6], input[7]);
    *reinterpret_cast<uint4*>(gmemPtr) = output;
}

template <>
inline __device__ void convertAndStoreToGmem<__nv_fp8_e4m3, 8>(char* gmemPtr, float (&input)[8])
{
    uint2 output;
    output.x = convert_float4_to_e4m3(input[0], input[1], input[2], input[3]);
    output.y = convert_float4_to_e4m3(input[4], input[5], input[6], input[7]);
    *reinterpret_cast<uint2*>(gmemPtr) = output;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Dtype, int32_t NumElts>
inline __device__ void convertToFloatAndAccumulate(float (&output)[NumElts], uint4 input, float scale0, float scale1)
{
    static_assert(sizeof(Dtype) == 0, "Not implemented.");
}

template <>
inline __device__ void convertToFloatAndAccumulate<__half, 8>(
    float (&output)[8], uint4 input, float scale0, float scale1)
{
    float2 scales0 = make_float2(scale0, scale0);
    float2 scales1 = make_float2(scale1, scale1);
#pragma unroll
    for (int32_t ii = 0; ii < 4; ++ii)
    {
        float2 a = __half22float2(reinterpret_cast<__half2*>(&input)[ii]);
        float2& c = reinterpret_cast<float2(&)[4]>(output)[ii];
        // FFMA2: output = input * scale1 + output * scale0.
        mul(c, c, scales0);
        fma(c, a, scales1, c);
    }
}

template <>
inline __device__ void convertToFloatAndAccumulate<__nv_bfloat16, 8>(
    float (&output)[8], uint4 input, float scale0, float scale1)
{
    float2 scales0 = make_float2(scale0, scale0);
    float2 scales1 = make_float2(scale1, scale1);
#pragma unroll
    for (int32_t ii = 0; ii < 4; ++ii)
    {
        float2 a = __bfloat1622float2(reinterpret_cast<__nv_bfloat162*>(&input)[ii]);
        float2& c = reinterpret_cast<float2(&)[4]>(output)[ii];
        // FFMA2: output = input * scale1 + output * scale0.
        mul(c, c, scales0);
        fma(c, a, scales1, c);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernels
} // namespace tensorrt_llm
