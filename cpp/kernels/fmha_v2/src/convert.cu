/*
 * SPDX-FileCopyrightText: Copyright (c) 2011-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <fmha/numeric_types.h>
#include <fmha/utils.h>
#include <stdint.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void convert_int32_to_int8_kernel(void* dst, void const* src, size_t n, float scale)
{

    // The step.
    size_t step = (size_t) gridDim.x * blockDim.x;

    // Iterate over the elements.
    for (size_t ii = blockIdx.x * blockDim.x + threadIdx.x; ii < n / 4; ii += step)
    {

        // Load 4 integers.
        int4 tmp = reinterpret_cast<int4 const*>(src)[ii];

        // Convert to float and scale.
        float x = static_cast<float>(tmp.x) * scale;
        float y = static_cast<float>(tmp.y) * scale;
        float z = static_cast<float>(tmp.z) * scale;
        float w = static_cast<float>(tmp.w) * scale;

        // Convert to int8.
        uint32_t a;
        asm volatile("cvt.rni.sat.s8.f32 %0, %1;\n" : "=r"(a) : "f"(x));
        uint32_t b;
        asm volatile("cvt.rni.sat.s8.f32 %0, %1;\n" : "=r"(b) : "f"(y));
        uint32_t c;
        asm volatile("cvt.rni.sat.s8.f32 %0, %1;\n" : "=r"(c) : "f"(z));
        uint32_t d;
        asm volatile("cvt.rni.sat.s8.f32 %0, %1;\n" : "=r"(d) : "f"(w));

        // Compact.
        char4 out;
        out.x = reinterpret_cast<int8_t const&>(a);
        out.y = reinterpret_cast<int8_t const&>(b);
        out.z = reinterpret_cast<int8_t const&>(c);
        out.w = reinterpret_cast<int8_t const&>(d);

        // Store.
        reinterpret_cast<uint32_t*>(dst)[ii] = reinterpret_cast<uint32_t const&>(out);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_conversion_int32_to_int8(void* dst, void const* src, int s, int b, int h, int d, float scale)
{
    size_t n = (size_t) s * b * h * d;
    convert_int32_to_int8_kernel<<<512, 256>>>(dst, src, n, scale);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__device__ inline typename fmha::Uint_from_size_in_bytes<sizeof(T) * 4>::Type pack_float4(float4 const& f);

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
__device__ inline uint2 pack_float4<fmha::fp16_t>(float4 const& f)
{
    return fmha::float4_to_half4(f.x, f.y, f.z, f.w);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
__device__ inline uint2 pack_float4<fmha::bf16_t>(float4 const& f)
{
    return fmha::float4_to_16bit_x4<fmha::bf16_t>(f.x, f.y, f.z, f.w);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
__device__ inline uint32_t pack_float4<fmha::e4m3_t>(float4 const& f)
{
    return fmha::float4_to_e4m3x4(f.x, f.y, f.z, f.w);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <>
__device__ inline uint32_t pack_float4<fmha::e5m2_t>(float4 const& f)
{
    return fmha::float4_to_e5m2x4(f.x, f.y, f.z, f.w);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__global__ void convert_fp32_to_T_kernel(void* dst, void const* src, size_t n, float scale = 1.f)
{

    using Dst = typename fmha::Uint_from_size_in_bytes<sizeof(T) * 4>::Type;

    // The step.
    size_t step = (size_t) gridDim.x * blockDim.x;

    // Iterate over the elements.
    for (size_t ii = blockIdx.x * blockDim.x + threadIdx.x; ii < n / 4; ii += step)
    {

        // Load 4 floats.
        float4 tmp = reinterpret_cast<float4 const*>(src)[ii];
        // Scale.
        tmp.x *= scale;
        tmp.y *= scale;
        tmp.z *= scale;
        tmp.w *= scale;
        // Convert to 4 Ts.
        auto out = pack_float4<T>(tmp);

        // Store.
        reinterpret_cast<Dst*>(dst)[ii] = reinterpret_cast<Dst const&>(out);
    }
}

template <typename T>
__global__ void convert_T_to_fp32_kernel(void* dst, void const* src, size_t n, float scale = 1.f)
{

    using Src = typename fmha::Uint_from_size_in_bytes<sizeof(T) * 4>::Type;

    union
    {
        Src raw;
        T elt[4];
    } data;

    // The step.
    size_t step = (size_t) gridDim.x * blockDim.x;

    // Iterate over the elements.
    for (size_t ii = blockIdx.x * blockDim.x + threadIdx.x; ii < n / 4; ii += step)
    {

        // Load 4 floats.
        data.raw = reinterpret_cast<Src const*>(src)[ii];
        float4 out;
        // Scale.
        out.x = float(data.elt[0]) * scale;
        out.y = float(data.elt[1]) * scale;
        out.z = float(data.elt[2]) * scale;
        out.w = float(data.elt[3]) * scale;

        // Store.
        reinterpret_cast<float4*>(dst)[ii] = reinterpret_cast<float4 const&>(out);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_conversion_fp32_to_fp16(void* dst, void const* src, int s, int b, int h, int d)
{
    // No need to expose the scale factor for FP16/FP32.
    size_t n = (size_t) s * b * h * d;
    convert_fp32_to_T_kernel<fmha::fp16_t><<<512, 256>>>(dst, src, n, 1.f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_conversion_fp32_to_bf16(void* dst, void const* src, int s, int b, int h, int d)
{
    // No need to expose the scale factor for FP16/FP32.
    size_t n = (size_t) s * b * h * d;
    convert_fp32_to_T_kernel<fmha::bf16_t><<<512, 256>>>(dst, src, n, 1.f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_conversion_fp32_to_e4m3(void* dst, void const* src, size_t n, float scale_o)
{
    convert_fp32_to_T_kernel<fmha::e4m3_t><<<512, 256>>>(dst, src, n, scale_o);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_conversion_e4m3_to_fp32(void* dst, void const* src, size_t n, float scale_o)
{
    convert_T_to_fp32_kernel<fmha::e4m3_t><<<512, 256>>>(dst, src, n, scale_o);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_conversion_fp32_to_e4m3(void* dst, void const* src, int s, int b, int h, int d, float scale_o)
{
    run_conversion_fp32_to_e4m3(dst, src, s * b * h * d, scale_o);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_conversion_fp32_to_e5m2(void* dst, void const* src, size_t n, float scale_o)
{
    convert_fp32_to_T_kernel<fmha::e5m2_t><<<512, 256>>>(dst, src, n, scale_o);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_conversion_e5m2_to_fp32(void* dst, void const* src, size_t n, float scale_o)
{
    convert_T_to_fp32_kernel<fmha::e5m2_t><<<512, 256>>>(dst, src, n, scale_o);
}
