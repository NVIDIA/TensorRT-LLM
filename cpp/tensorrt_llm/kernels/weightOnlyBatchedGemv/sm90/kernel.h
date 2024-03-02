/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/sm90/common.h"

namespace tensorrt_llm
{
namespace kernels
{
namespace weight_only
{

struct ConverterI4ToF16
{
    __device__ __forceinline__ static void convert(uint32_t& src, uint4& dst)
    {
        uint32_t* r = reinterpret_cast<uint32_t*>(&dst);
        uint32_t prmt_indices[4] = {0x4040, 0x4141, 0x4242, 0x4343};
#pragma unroll
        for (int ii = 0; ii < 4; ++ii)
        {
            asm volatile(
                "{\n"
                "  prmt.b32 %0, %1, %2, %3;\n"
                "}\n"
                : "=r"(r[ii])
                : "r"(src), "n"(0), "r"(prmt_indices[ii]));
        }
        static constexpr uint32_t xor_mask = 0x64806408;
        static constexpr uint32_t and_mask = 0xFFF0FF0F;
        static constexpr uint32_t immLut = (0xf0 & 0xcc) ^ 0xaa;
#pragma unroll
        for (int ii = 0; ii < 4; ++ii)
        {
            asm volatile(
                "{\n"
                "  lop3.b32 %0, %0, %1, %2, %3;\n"
                "}\n"
                : "+r"(r[ii])
                : "n"(and_mask), "n"(xor_mask), "n"(immLut));
        }
        static constexpr uint32_t hfma_bias_rep = 0xD480E408;
        static constexpr uint32_t hfma_scale_rep = 0x2C003C00;

        const half2& hfma_bias = reinterpret_cast<const half2&>(hfma_bias_rep);
        const half2& hfma_scale = reinterpret_cast<const half2&>(hfma_scale_rep);
#pragma unroll
        for (int ii = 0; ii < 4; ++ii)
        {
            __half2& fp16x2_val = reinterpret_cast<__half2&>(r[ii]);
            fp16x2_val = __hfma2(hfma_scale, fp16x2_val, hfma_bias);
        }
    }

    template <int N>
    __device__ __forceinline__ static void convert(void* src, void* dst)
    {
        static_assert(N == 8 || N == 16);
        convert(reinterpret_cast<uint32_t*>(src)[0], reinterpret_cast<uint4*>(dst)[0]);
        if constexpr (N == 16)
        {
            convert(reinterpret_cast<uint32_t*>(src)[1], reinterpret_cast<uint4*>(dst)[1]);
        }
    }
};

template <typename TVec, int N, bool Enable, typename TSrc>
__device__ __forceinline__ void load(void* dst, TSrc* src, int stride)
{
    if constexpr (Enable)
    {
#pragma unroll
        for (int ii = 0; ii < N; ++ii)
        {
            reinterpret_cast<TVec*>(dst)[ii] = reinterpret_cast<TVec*>(src + ii * stride)[0];
        }
    }
}

template <int M, int K, bool Enable>
__device__ __forceinline__ void apply_scale(void* act, void* act_scale)
{
    static_assert(K % 2 == 0);
    static constexpr int VecK = K / 2;
    if constexpr (Enable)
    {
        half2* pa = reinterpret_cast<half2*>(act);
        half2* pb = reinterpret_cast<half2*>(act_scale);
#pragma unroll
        for (int m = 0; m < M; ++m)
        {
#pragma unroll
            for (int k = 0; k < VecK; ++k)
            {
                pa[m * VecK + k] = __hmul2(pa[m * VecK + k], pb[k]);
            }
        }
    }
}

template <int N, int K, bool EnableZero>
__device__ __forceinline__ void dequantize(void* w, void* quantized_w, void* scales, void* zeros, half alpha)
{
    using Converter = ConverterI4ToF16;
    static_assert(K % 2 == 0);
    static constexpr int VecK = K / 2;
#pragma unroll
    for (int n = 0; n < N; ++n)
    {
        ConverterI4ToF16::convert<K>(
            reinterpret_cast<uint8_t*>(quantized_w) + n * K / 2, reinterpret_cast<half*>(w) + n * K);
        half2 vec_scale = __half2half2(reinterpret_cast<half*>(scales)[n] * alpha);
        half2 vec_zero = __half2half2(__float2half_rn(0.f));
        if constexpr (EnableZero)
        {
            vec_zero = __half2half2(reinterpret_cast<half*>(zeros)[n] * alpha);
        }
#pragma unroll
        for (int k = 0; k < VecK; ++k)
        {
            reinterpret_cast<half2*>(w)[n * VecK + k]
                = __hfma2(reinterpret_cast<half2*>(w)[n * VecK + k], vec_scale, vec_zero);
        }
    }
}

template <int N, int K>
__device__ __forceinline__ void pack_to_vec2(void* dst, void* src)
{
#pragma unroll
    for (int n = 0; n < N; n += 2)
    {
#pragma unroll
        for (int k = 0; k < K; ++k)
        {
            reinterpret_cast<half*>(dst)[n * K + k * 2] = reinterpret_cast<half*>(src)[n * K + k];
            reinterpret_cast<half*>(dst)[n * K + k * 2 + 1] = reinterpret_cast<half*>(src)[(n + 1) * K + k];
        }
    }
}

template <int M, int N, int K>
__device__ __forceinline__ void mma(void* acc, void* w_pack2, void* act)
{
    static_assert(N % 2 == 0);
    static constexpr int VecN = N / 2;
#pragma unroll
    for (int m = 0; m < M; ++m)
    {
#pragma unroll
        for (int n = 0; n < VecN; ++n)
        {
#pragma unroll
            for (int k = 0; k < K; ++k)
            {
                reinterpret_cast<half2*>(acc)[m * VecN + n] = __hfma2(reinterpret_cast<half2*>(w_pack2)[n * K + k],
                    __half2half2(reinterpret_cast<half*>(act)[m * K + k]), reinterpret_cast<half2*>(acc)[m * VecN + n]);
            }
        }
    }
}

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T& val)
{
    val += __shfl_xor_sync(~0, val, 16);
    val += __shfl_xor_sync(~0, val, 8);
    val += __shfl_xor_sync(~0, val, 4);
    val += __shfl_xor_sync(~0, val, 2);
    val += __shfl_xor_sync(~0, val, 1);
    return val;
}

template <int CtaM, int CtaN, int Threads, bool EnableBias>
__device__ __forceinline__ void epilogue(void* out, int stride, void* tile_acc, void* bias)
{
    static constexpr int WarpSize = 32;
    static constexpr int WarpNum = Threads / WarpSize;
    static constexpr int AlignShmemSize = (CtaM * CtaN + 31) / 32 * 32;
    static_assert(Threads % WarpSize == 0);
    __shared__ float shmem[AlignShmemSize * WarpNum];
    int tid = threadIdx.x;
    int warp_id = tid / WarpSize, lane_id = tid % WarpSize;
#pragma unroll
    for (int m = 0; m < CtaM; ++m)
    {
#pragma unroll
        for (int n = 0; n < CtaN; ++n)
        {
            float v = __half2float(reinterpret_cast<half*>(tile_acc)[m * CtaN + n]);
            v = warp_reduce_sum(v);
            if (lane_id == 0)
            {
                shmem[warp_id * AlignShmemSize + m * CtaN + n] = v;
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int ii = tid; ii < CtaM * CtaN; ii += Threads)
    {
        int m = ii / CtaN, n = ii % CtaN;
        float val = 0.f, v_bias = 0.f;
        if constexpr (EnableBias)
        {
            v_bias = static_cast<float>(reinterpret_cast<half*>(bias)[n]);
        }
#pragma unroll
        for (int jj = 0; jj < WarpNum; ++jj)
        {
            val += shmem[jj * AlignShmemSize + ii];
        }
        reinterpret_cast<half*>(out)[m * stride + n] = __float2half_rn(val + v_bias);
    }
}

template <int N>
__device__ __forceinline__ void fill(void* tile, half v)
{
#pragma unroll
    for (int ii = 0; ii < N; ++ii)
    {
        reinterpret_cast<half*>(tile)[ii] = v;
    }
}

struct Fp16Details
{
    using ActDataType = half;
    static constexpr int StepK = 8;
    using AccessTypeAct = float4;
    using AccessTypeActScale = float4;
    using AccessTypeW = float;

    template <int CtaM>
    __device__ __forceinline__ static void load_act(void* dst, void* src, int stride)
    {
        load<AccessTypeAct, CtaM, true>(dst, reinterpret_cast<ActDataType*>(src), stride);
    }
};

struct Fp8Details
{
    using ActDataType = __nv_fp8_e4m3;
    static constexpr int StepK = 8;
    using AccessTypeAct = float2;
    using AccessTypeActScale = float4;
    using AccessTypeW = float;

    __device__ __forceinline__ static void conversion(void* dst, void* src)
    {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#pragma unroll
        for (int ii = 0; ii < StepK / 4; ++ii)
        {
            asm volatile(
                "{\n"
                ".reg .b16 lo, hi;\n"
                "mov.b32 {lo, hi}, %2;\n"
                "cvt.rn.f16x2.e4m3x2 %0, lo;\n"
                "cvt.rn.f16x2.e4m3x2 %1, hi;\n"
                "}\n"
                : "=r"(reinterpret_cast<uint32_t*>(dst)[ii * 2]), "=r"(reinterpret_cast<uint32_t*>(dst)[ii * 2 + 1])
                : "r"(reinterpret_cast<uint32_t*>(src)[ii]));
        }
#else
#pragma unroll
        for (int ii = 0; ii < StepK; ++ii)
        {
            reinterpret_cast<half*>(dst)[ii] = static_cast<half>(reinterpret_cast<ActDataType*>(src)[ii]);
        }
#endif
    }

    template <int CtaM>
    __device__ __forceinline__ static void load_act(void* dst, void* src, int stride)
    {
        ActDataType vec[CtaM * StepK];
        load<AccessTypeAct, CtaM, true>(vec, reinterpret_cast<ActDataType*>(src), stride);
#pragma unroll
        for (int ii = 0; ii < CtaM; ++ii)
        {
            conversion(reinterpret_cast<half*>(dst) + ii * StepK, vec + ii * StepK);
        }
    }
};

template <typename Details, int CtaM, int CtaN, int Threads, int GroupSize, bool EnableActScale, bool EnableZero,
    bool EnableBias>
__global__ void kernel(typename Details::ActDataType* act, half* act_scale, uint8_t* weight, half* scales, half* zeros,
    half* bias, half* out, float alpha, int m, int n, int k)
{
    // ArgType          ArgName          DataType       Shape                   Layout
    //
    // input            act              fp16           [m, k]                  RowMajor
    // input            act_scale        fp16           [1, k]                  RowMajor
    // input            weight           int4b          [k, n]                  ColumnMajor
    // input            scales           fp16           [k / GroupSize, n]      RowMajor
    // input            zeros            fp16           [k / GroupSize, n]      RowMajor
    // input            bias             fp16           [1, n]                  RowMajor
    // output           out              fp16           [m, n]                  RowMajor

    using AccessTypeActScale = typename Details::AccessTypeActScale;
    using AccessTypeW = typename Details::AccessTypeW;
    static constexpr int StepK = Details::StepK;

    static constexpr bool Mandatory = true;
    static constexpr int CtaK = StepK * Threads;
    static_assert(CtaN % 2 == 0);

    const int m_tile_id = blockIdx.x, n_tile_id = blockIdx.y, tid = threadIdx.x;
    const int m_offset = m_tile_id * CtaM, n_offset = n_tile_id * CtaN;

    act += m_offset * k;
    weight += n_offset * k / 2;
    scales += n_offset;
    zeros += n_offset;
    bias += n_offset;
    out += m_offset * n + n_offset;

    half tile_a[StepK * CtaM], tile_w[StepK * CtaN], tile_w_pack2[StepK * CtaN];
    half tile_acc[CtaM * CtaN];
    fill<CtaM * CtaN>(tile_acc, __float2half_rn(0.f));

    for (int idx_k = tid * StepK; idx_k < k; idx_k += CtaK)
    {
        half vec_act_scale[StepK];
        half vec_scale[CtaN], vec_zero[CtaN];
        uint8_t tile_w_quantized[StepK * CtaN / 2];
        // Load Data
        Details::load_act<CtaM>(tile_a, act + idx_k, k);
        load<AccessTypeActScale, 1, EnableActScale>(vec_act_scale, act_scale + idx_k, 0);
        load<AccessTypeW, CtaN, Mandatory>(tile_w_quantized, weight + idx_k / 2, k / 2);
        load<half, CtaN, Mandatory>(vec_scale, scales + idx_k / GroupSize * n, 1);
        load<half, CtaN, EnableZero>(vec_zero, zeros + idx_k / GroupSize * n, 1);
        // Dequantize Data
        // W4A8 checkpoints have larger activation and weight values. In order to prevent the warp-level FP16
        // accumulator from overflow, the multiplication of alpha is moved from epilogue to dequantize
        apply_scale<CtaM, StepK, EnableActScale>(tile_a, vec_act_scale);
        dequantize<CtaN, StepK, EnableZero>(tile_w, tile_w_quantized, vec_scale, vec_zero, __float2half_rn(alpha));
        // Rearrange
        pack_to_vec2<CtaN, StepK>(tile_w_pack2, tile_w);
        // MMA
        mma<CtaM, CtaN, StepK>(tile_acc, tile_w_pack2, tile_a);
    }
    // Epilogue
    epilogue<CtaM, CtaN, Threads, EnableBias>(out, n, tile_acc, bias);
}

template <typename Details, int CtaM, int CtaN, int Threads, int GroupSize, bool EnableActScale, bool EnableZero,
    bool EnableBias>
void exec_kernel(Params& params, cudaStream_t s)
{
    if (params.m % CtaM || params.n % CtaN)
    {
        throw std::runtime_error("launch failed");
    }
    dim3 grid(params.m / CtaM, params.n / CtaN);
    dim3 block(Threads);
    // clang-format off
    kernel<Details, CtaM, CtaN, Threads, GroupSize, EnableActScale, EnableZero, EnableBias><<<grid, block, 0, s>>>(
        reinterpret_cast<typename Details::ActDataType*>(params.act),
        reinterpret_cast<half*>(params.act_scale),
        reinterpret_cast<uint8_t*>(params.weight),
        reinterpret_cast<half*>(params.scales),
        reinterpret_cast<half*>(params.zeros),
        reinterpret_cast<half*>(params.bias),
        reinterpret_cast<half*>(params.out),
        params.alpha,
        params.m, params.n, params.k
    );
    // clang-format on
}

} // namespace weight_only
} // namespace kernels
} // namespace tensorrt_llm
