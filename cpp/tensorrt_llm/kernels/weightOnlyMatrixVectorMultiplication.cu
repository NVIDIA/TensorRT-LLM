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

#include "stdio.h"
#include <cassert>
#include <cmath>
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass_extensions/interleaved_numeric_conversion.h"
#include "tensorrt_llm/kernels/weightOnlyMatrixVectorMultiplication.h"

namespace tensorrt_llm
{
namespace kernels
{

/////////////////////////////////////////////////////////////////////
/* Fast convert from weight only int8/int4 to half */

template <QuantType Type>
struct FastWeightOnlyHalfConverter;

template <>
struct FastWeightOnlyHalfConverter<QuantType::INT8_WEIGHT_ONLY>
{
    using Converter = cutlass::FastInterleavedAndBiasedNumericArrayConverter<cutlass::half_t, uint8_t, 4>;
    static constexpr int kHalfLength = 4;
    static constexpr int kWeightOnlyLength = 4;

    __device__ static inline void convert(half halves[kHalfLength], uint8_t chars[kWeightOnlyLength], half scale)
    {
        *reinterpret_cast<Converter::result_type*>(halves)
            = Converter::convert(*reinterpret_cast<Converter::source_type*>(chars));
#pragma unroll
        for (int i = 0; i < kHalfLength; ++i)
        {
            halves[i] *= scale;
        }
    }
};

template <>
struct FastWeightOnlyHalfConverter<QuantType::PACKED_INT4_WEIGHT_ONLY>
{
    using Converter = cutlass::FastInterleavedAndBiasedNumericArrayConverter<cutlass::half_t, cutlass::uint4b_t, 8>;
    static constexpr int kHalfLength = 8;
    static constexpr int kWeightOnlyLength = 4;

    __device__ static inline void convert(half halves[kHalfLength], uint8_t chars[kWeightOnlyLength], half scale)
    {
        *reinterpret_cast<Converter::result_type*>(halves)
            = Converter::convert(*reinterpret_cast<Converter::source_type*>(chars));
#pragma unroll
        for (int i = 0; i < kHalfLength; ++i)
        {
            halves[i] *= scale;
        }
    }
};

/* Activation */

__forceinline__ __device__ float copysignf_pos(float a, float b)
{
    float r;
    r = __int_as_float(__float_as_int(a) | (__float_as_int(b) & 0x80000000));
    return r;
}

__inline__ __device__ float tanh_opt(float x)
{
#if (__CUDA_ARCH__ >= 750 && CUDART_VERSION >= 11000)
    float r;
    asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
    return r;
#else
    const float exp_val = -1.f * fabs(2 * x);
    return copysignf_pos((1.0f - __expf(exp_val)) / (__expf(exp_val) + 1.0f), x);
#endif
}

template <typename T>
struct GeluActivation
{
    static __device__ __forceinline__ T apply(const T& val)
    {
        const float cdf = 0.5f * (1.0f + tanh_opt((0.7978845608028654f * (val + 0.044715f * val * val * val))));
        return val * cdf;
    }
};

template <typename T>
struct ReluActivation
{
    static __device__ __forceinline__ T apply(const T& val)
    {
        return val > static_cast<T>(0.0f) ? val : static_cast<T>(0.0f);
    }
};

template <typename T>
struct IdentityActivation
{
    static __device__ __forceinline__ T apply(const T& val)
    {
        return val;
    }
};

template <typename VecType, typename T0, typename T1>
__device__ __forceinline__ void load(T0* dst, T1* src, size_t offset = 0)
{
    *reinterpret_cast<VecType*>(dst) = *(reinterpret_cast<const VecType*>(src) + offset);
}

template <typename VecType, typename T0, typename T1>
__device__ __forceinline__ void store(T0* src, T1* dst, size_t offset = 0)
{
    *(reinterpret_cast<VecType*>(dst) + offset) = *reinterpret_cast<const VecType*>(src);
}

template <bool Bias, template <typename T> class Activation, int K = 0>
__global__ void int8_weight_only_gemv_interleave(const int8_t* weight, const half* input, const half* scale_list,
    const half* bias, half* output, const int n, const int k_)
{
    using Converter = FastWeightOnlyHalfConverter<QuantType::INT8_WEIGHT_ONLY>;
    int k = K != 0 ? K : k_;
    uint8_t vec_weight[16];
    half vec_input[16];
    half vec_weight_f16[16];
    int warp_id = threadIdx.x / 32, lane_id = threadIdx.x % 32;
    int tile_id = blockIdx.x * blockDim.x / 32 + warp_id;
    // Every two rows of the original weights are interleaved into a row with stride of 64, so if each thread
    // processes 16 elements(for int8, we can use ldg.128 to load weights), then every group of four adjacent threads
    // will alternately process two different row weights
    // for example
    // every 128 consecutive int8 elements [128*i, 128*(i+1)-1] of row N under interleave layout,
    // the first 64 are from [64*i, 64*(i+1)-1] of row 2N before interleaving,
    // and the last 64 are from [64*i, 64*(i+1)-1] of row 2N+1 before interleaving.
    // So if each thread loads 16 int8 elements, then the elements of the first four and last four threads of each 8
    // consecutive threads will come from row 2N and row 2N+1 respectively before interleaving.
    int row_id = tile_id * 2 + ((lane_id % 8) > 3 ? 1 : 0);
    weight += tile_id * k * 2;
    float v = 0.f, v_bias;
    half scale = scale_list[row_id];
    if (Bias)
    {
        v_bias = __half2float(bias[row_id]);
    }
#pragma unroll
    for (int i = lane_id * 16; i < k * 2; i += 16 * 32)
    {
        load<uint4>(vec_weight, weight + i);
        load<float4>(vec_input, input + i / 128 * 64 + (i % 64));
        load<float4>(vec_input + 8, input + i / 128 * 64 + (i % 64) + 8);
#pragma unroll
        for (int p = 0; p < 16; p += Converter::kHalfLength)
        {
            // The rearrangement here counteracts the effect of cutlass::add_bias_and_interleave_int8s_inplace
            // Input int8 data layout
            //      [elt_3  elt_1  elt_2  elt_0] (each elt occupies 8 bits)
            //
            // Converted fp16 data layout
            //      [elt_3  elt_2  elt_1  elt_0] (each elt occupies 16 bits)
            Converter::convert(vec_weight_f16 + p, vec_weight + p, scale);
        }
#pragma unroll
        for (int p = 0; p < 16; ++p)
        {
            // The index remapping here is to counteracts the effect of cutlass::permute_B_rows_for_mixed_gemm
            // input 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
            // weight 0 1 8 9 2 3 10 11 4 5 12 13 6 7 14 15
            v += __half2float(__hmul(vec_input[p], vec_weight_f16[4 * ((p % 8) / 2) + p % 2 + 2 * (p / 8)]));
        }
    }
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v += __shfl_xor_sync(0xffffffff, v, 8);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    v += __shfl_xor_sync(0xffffffff, v, 1);
    if (lane_id == 0 || lane_id == 4)
    {
        if (Bias)
        {
            output[row_id] = __float2half_rn(Activation<float>::apply(v + v_bias));
        }
        else
        {
            output[row_id] = __float2half_rn(Activation<float>::apply(v));
        }
    }
}

template <bool Bias, template <typename T> class Activation, int K = 0>
__global__ void int4_weight_only_gemv_interleave(const int8_t* weight, const half* input, const half* scale_list,
    const half* bias, half* output, const int n, const int k_)
{
    using Converter = FastWeightOnlyHalfConverter<QuantType::PACKED_INT4_WEIGHT_ONLY>;
    int k = K != 0 ? K : k_;
    uint8_t vec_weight[16];
    half vec_input[32];
    half vec_weight_f16[32];
    int warp_id = threadIdx.x / 32, lane_id = threadIdx.x % 32;
    int tile_id = blockIdx.x * blockDim.x / 32 + warp_id;
    // Every four rows of the original weights are interleaved into a row with stride of 64, so if each thread
    // processes 32 elements(for int4, we can use ldg.128 to load weights), then every group of two adjacent threads
    // will alternately process four different row weights
    // for example
    // every 256 consecutive int4 elements [256*i, 256*(i+1)-1] of row N under interleave layout,
    // the first 64 are from [64*i, 64*(i+1)-1] of row 4N before interleaving,
    // and the second 64 are from [64*i, 64*(i+1)-1] of row 4N+1 before interleaving, and so on.
    // So if each thread loads 32 int4 elements, then the elements of each 2 adjacent threads of each 8
    // consecutive threads will come from row 4N ~ 4N+3 respectively before interleaving.
    int row_id = tile_id * 4 + ((lane_id % 8) / 2);
    weight += tile_id * k / 2 * 4;
    float v = 0.f, v_bias;
    half scale = scale_list[row_id];
    if (Bias)
    {
        v_bias = __half2float(bias[row_id]);
    }
#pragma unroll
    for (int i = lane_id * 32; i < k * 4; i += 32 * 32)
    {
        load<uint4>(vec_weight, weight + i / 2);
        load<float4>(vec_input, input + i / 256 * 64 + (i % 64));
        load<float4>(vec_input + 8, input + i / 256 * 64 + (i % 64) + 8);
        load<float4>(vec_input + 16, input + i / 256 * 64 + (i % 64) + 16);
        load<float4>(vec_input + 24, input + i / 256 * 64 + (i % 64) + 24);
#pragma unroll
        for (int p = 0; p < 32; p += Converter::kHalfLength)
        {
            // The rearrangement here counteracts the effect of cutlass::add_bias_and_interleave_int4s_inplace
            // Input int8 data layout
            //      [elt_7  elt_5  elt_3  elt_1  elt_6  elt_4  elt_2  elt_0] (each elt occupies 4 bits)
            //
            // Converted fp16 data layout
            //      [elt_7  elt_6  elt_5  elt_4  elt_3  elt_2  elt_1  elt_0] (each elt occupies 16 bits)
            Converter::convert(vec_weight_f16 + p, vec_weight + p / 2, scale);
        }
#pragma unroll
        for (int p = 0; p < 32; ++p)
        {
            // The index remapping here is to counteracts the effect of cutlass::permute_B_rows_for_mixed_gemm
            // input 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 ... 31
            // weight 0 1 8 9 16 17 24 25 2 3 10 11 18 19 26 27 4 5 12 13 20 21 28 29 6 7 14 15 22 23 30 31
            v += __half2float(__hmul(vec_input[p], vec_weight_f16[8 * ((p % 8) / 2) + p % 2 + 2 * (p / 8)]));
        }
    }
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v += __shfl_xor_sync(0xffffffff, v, 8);
    v += __shfl_xor_sync(0xffffffff, v, 1);
    if (lane_id == 0 || lane_id == 2 || lane_id == 4 || lane_id == 6)
    {
        if (Bias)
        {
            output[row_id] = __float2half_rn(Activation<float>::apply(v + v_bias));
        }
        else
        {
            output[row_id] = __float2half_rn(Activation<float>::apply(v));
        }
    }
}

template <bool Bias, template <typename T> class Activation, int K = 0>
void weight_only_gemv_kernel_launcher(const int8_t* weight, const half* input, const half* scale_list, const half* bias,
    half* output, const int k, const int n, dim3 grid, dim3 block, QuantType qtype, cudaStream_t stream)
{
    if (qtype == QuantType::PACKED_INT4_WEIGHT_ONLY)
    {
        grid.x /= 2;
        int4_weight_only_gemv_interleave<Bias, Activation, K>
            <<<grid, block, 0, stream>>>(weight, input, scale_list, bias, output, n, k);
    }
    else if (qtype == QuantType::INT8_WEIGHT_ONLY)
    {
        int8_weight_only_gemv_interleave<Bias, Activation, K>
            <<<grid, block, 0, stream>>>(weight, input, scale_list, bias, output, n, k);
    }
}

#define INVOKE_WEIGHT_ONLY_GEMV(ActivationType, K)                                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
        if (bias)                                                                                                      \
        {                                                                                                              \
            weight_only_gemv_kernel_launcher<true, ActivationType, K>(                                                 \
                weight, input, scale_list, bias, output, k, n, grid, block, qtype, stream);                            \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            weight_only_gemv_kernel_launcher<false, ActivationType, K>(                                                \
                weight, input, scale_list, bias, output, k, n, grid, block, qtype, stream);                            \
        }                                                                                                              \
    } while (0);

#define SWITCH_CASE_FOR_DIFFERENT_K(ActivationType, K)                                                                 \
    case K:                                                                                                            \
    {                                                                                                                  \
        INVOKE_WEIGHT_ONLY_GEMV(ActivationType, K);                                                                    \
        break;                                                                                                         \
    }
#define INVOKE_WEIGHT_ONLY_KERNEL_FOR_SPECIFIED_SHAPE(ActivationType)                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        switch (k)                                                                                                     \
        {                                                                                                              \
            SWITCH_CASE_FOR_DIFFERENT_K(ActivationType, 1536)                                                          \
            SWITCH_CASE_FOR_DIFFERENT_K(ActivationType, 2048)                                                          \
            SWITCH_CASE_FOR_DIFFERENT_K(ActivationType, 2560)                                                          \
            SWITCH_CASE_FOR_DIFFERENT_K(ActivationType, 4096)                                                          \
            SWITCH_CASE_FOR_DIFFERENT_K(ActivationType, 4608)                                                          \
            SWITCH_CASE_FOR_DIFFERENT_K(ActivationType, 5120)                                                          \
            SWITCH_CASE_FOR_DIFFERENT_K(ActivationType, 6144)                                                          \
            SWITCH_CASE_FOR_DIFFERENT_K(ActivationType, 7680)                                                          \
            SWITCH_CASE_FOR_DIFFERENT_K(ActivationType, 8192)                                                          \
            SWITCH_CASE_FOR_DIFFERENT_K(ActivationType, 10240)                                                         \
            SWITCH_CASE_FOR_DIFFERENT_K(ActivationType, 12288)                                                         \
            SWITCH_CASE_FOR_DIFFERENT_K(ActivationType, 15360)                                                         \
            SWITCH_CASE_FOR_DIFFERENT_K(ActivationType, 16384)                                                         \
            SWITCH_CASE_FOR_DIFFERENT_K(ActivationType, 20480)                                                         \
        default:                                                                                                       \
        {                                                                                                              \
            INVOKE_WEIGHT_ONLY_GEMV(ActivationType, 0);                                                                \
            break;                                                                                                     \
        }                                                                                                              \
        }                                                                                                              \
    } while (0);
#define INVOKE_WEIGHT_ONLY_KERNEL_FOR_DIFFERENT_ACT()                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        switch (activation)                                                                                            \
        {                                                                                                              \
        case ActivationType::Gelu:                                                                                     \
        {                                                                                                              \
            INVOKE_WEIGHT_ONLY_KERNEL_FOR_SPECIFIED_SHAPE(GeluActivation);                                             \
            break;                                                                                                     \
        }                                                                                                              \
        case ActivationType::Relu:                                                                                     \
        {                                                                                                              \
            INVOKE_WEIGHT_ONLY_KERNEL_FOR_SPECIFIED_SHAPE(ReluActivation);                                             \
            break;                                                                                                     \
        }                                                                                                              \
        case ActivationType::Identity:                                                                                 \
        {                                                                                                              \
            INVOKE_WEIGHT_ONLY_KERNEL_FOR_SPECIFIED_SHAPE(IdentityActivation);                                         \
            break;                                                                                                     \
        }                                                                                                              \
        default:                                                                                                       \
        {                                                                                                              \
            assert(false);                                                                                             \
            break;                                                                                                     \
        }                                                                                                              \
        }                                                                                                              \
    } while (0);

template <>
void weight_only_gemv_launcher(const half* input, const int8_t* weight, const half* scale_list, const half* bias,
    half* output, const int k, const int n, ActivationType activation, QuantType qtype, cudaStream_t stream)
{
    dim3 block(512);
    dim3 grid(n / 32);
    INVOKE_WEIGHT_ONLY_KERNEL_FOR_DIFFERENT_ACT();
}

} // namespace kernels
} // namespace tensorrt_llm
