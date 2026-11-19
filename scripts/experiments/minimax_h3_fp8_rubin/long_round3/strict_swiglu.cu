// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

namespace ffi = tvm::ffi;

constexpr int D = 14336;
constexpr int ROW_THREADS = 128;
constexpr int ROWS_PER_CTA = 2;
constexpr int THREADS = ROW_THREADS * ROWS_PER_CTA;

__device__ __forceinline__ unsigned short fp8x2(float x0, float x1)
{
    unsigned short packed;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(packed) : "f"(x0), "f"(x1));
    return packed;
}

__device__ __forceinline__ float exp_approx(float x)
{
    float y;
    float const z = x * 0x1.715476p+0f;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(z));
    return y;
}

__device__ __forceinline__ float div_full(float a, float b)
{
    float y;
    asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

__device__ __forceinline__ float clamp_fp8(float x)
{
    float y;
    asm("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(y) : "f"(x), "f"(448.0f));
    return y;
}

__device__ __forceinline__ float swiglu(float a, float b, float scale)
{
    float const sigmoid = div_full(1.0f, 1.0f + exp_approx(-a));
    float const y = (a * sigmoid) * b;
    return clamp_fp8(div_full(y, scale));
}

union Bf162Bits
{
    unsigned int bits;
    __nv_bfloat162 value;
};

__device__ __forceinline__ unsigned short swiglu_pair(unsigned int abits, unsigned int bbits, float scale)
{
    Bf162Bits av{abits};
    Bf162Bits bv{bbits};
    const float2 a = __bfloat1622float2(av.value);
    const float2 b = __bfloat1622float2(bv.value);
    return fp8x2(swiglu(a.x, b.x, scale), swiglu(a.y, b.y, scale));
}

template <int N>
__device__ __forceinline__ void process_stage(uint4 const* __restrict__ gate, uint4 const* __restrict__ up,
    unsigned long long* __restrict__ out, int start, int lane, float scale)
{
    uint4 av[N];
    uint4 bv[N];
#pragma unroll
    for (int i = 0; i < N; ++i)
    {
        int const v = start + lane + i * ROW_THREADS;
        av[i] = gate[v];
        bv[i] = up[v];
    }
#pragma unroll
    for (int i = 0; i < N; ++i)
    {
        unsigned long long const q0 = swiglu_pair(av[i].x, bv[i].x, scale);
        unsigned long long const q1 = swiglu_pair(av[i].y, bv[i].y, scale);
        unsigned long long const q2 = swiglu_pair(av[i].z, bv[i].z, scale);
        unsigned long long const q3 = swiglu_pair(av[i].w, bv[i].w, scale);
        int const v = start + lane + i * ROW_THREADS;
        out[v] = q0 | (q1 << 16) | (q2 << 32) | (q3 << 48);
    }
}

__global__ __launch_bounds__(THREADS, 1024 / THREADS) void swiglu_fp8_rows(
    __nv_bfloat16 const* __restrict__ x, float const* __restrict__ scale_ptr, unsigned char* __restrict__ out, int rows)
{
    int const local_row = threadIdx.x / ROW_THREADS;
    int const lane = threadIdx.x & (ROW_THREADS - 1);
    int const row = blockIdx.x * ROWS_PER_CTA + local_row;
    if (row >= rows)
        return;
    const size_t xoff = static_cast<size_t>(row) * (2 * D);
    const size_t ooff = static_cast<size_t>(row) * D;
    uint4 const* gate = reinterpret_cast<uint4 const*>(x + xoff);
    uint4 const* up = reinterpret_cast<uint4 const*>(x + xoff + D);
    unsigned long long* dst = reinterpret_cast<unsigned long long*>(out + ooff);
    float const scale = scale_ptr[0];
    process_stage<7>(gate, up, dst, 0, lane, scale);
    process_stage<7>(gate, up, dst, 7 * ROW_THREADS, lane, scale);
}

void Run(ffi::TensorView x, ffi::TensorView scale, ffi::TensorView out)
{
    const cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, x.device().device_id));
    int const rows = static_cast<int>(x.shape()[0]);
    swiglu_fp8_rows<<<(rows + ROWS_PER_CTA - 1) / ROWS_PER_CTA, THREADS, 0, stream>>>(
        static_cast<__nv_bfloat16 const*>(x.data_ptr()), static_cast<float const*>(scale.data_ptr()),
        static_cast<unsigned char*>(out.data_ptr()), rows);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, Run);
