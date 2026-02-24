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

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaFp8Utils.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include <algorithm>
#include <cstdio>
#include <cuda_fp16.h>
#include <limits>
#include <type_traits>

TRTLLM_NAMESPACE_BEGIN

namespace common
{
#ifdef ENABLE_FP8

constexpr int CTA_SIZE = 256;
constexpr int VEC_SIZE = 8; // elements per vectorized iteration (128-bit load = 8 × bf16)
constexpr int64_t MAX_GRID_DIM = 65536;

inline int64_t scaleMatrixVecGridSize(int64_t numel)
{
    int64_t vecElements = numel / VEC_SIZE;
    int64_t blocks = (vecElements + CTA_SIZE - 1) / CTA_SIZE;
    return std::max(int64_t(1), std::min(blocks, MAX_GRID_DIM));
}

template <bool QUANTIZE>
__inline__ __device__ float scale(float a, float b)
{
    return QUANTIZE ? a / b : a * b;
}

// Vectorized PER_TENSOR kernel for bf16 input → fp8 output.
// Each thread processes 8 bf16 values per iteration via 128-bit loads / 64-bit stores.
// T_S may be float or __nv_bfloat16; the single scale element is converted to float once.
template <bool QUANTIZE, typename T_S>
__global__ void scaleMatrixPerTensorVec(
    __nv_fp8_e4m3* output, T_S const* input_scale, __nv_bfloat16 const* input, int64_t numel)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    float const s = static_cast<float>(input_scale[0]);
    float const factor = QUANTIZE ? (1.0f / s) : s;
    int64_t const vecElements = numel / VEC_SIZE;
    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t vi = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x; vi < vecElements; vi += stride)
    {
        int64_t const base = vi * VEC_SIZE;

        // 128-bit load: 8 × bf16
        float4 raw = *reinterpret_cast<float4 const*>(input + base);
        __nv_bfloat162 const* pairs = reinterpret_cast<__nv_bfloat162 const*>(&raw);

        // Convert 4 × bf16-pair → 4 × float2 → 4 × fp8-pair → pack into 2 × uint32
        __nv_fp8x2_storage_t fp8_pairs[4];
#pragma unroll
        for (int p = 0; p < 4; ++p)
        {
            float2 f2 = __bfloat1622float2(pairs[p]);
            f2.x *= factor;
            f2.y *= factor;
            fp8_pairs[p] = __nv_cvt_float2_to_fp8x2(f2, __NV_SATFINITE, __NV_E4M3);
        }

        // 64-bit store: 8 × fp8
        *reinterpret_cast<uint2*>(output + base) = *reinterpret_cast<uint2 const*>(fp8_pairs);
    }

    // Scalar tail for remaining elements (numel not divisible by 8)
    int64_t const tailStart = vecElements * VEC_SIZE;
    for (int64_t i = tailStart + threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x; i < numel; i += stride)
    {
        output[i] = __nv_fp8_e4m3(static_cast<float>(input[i]) * factor);
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <QuantizeMode QUANTIZE_MODE, bool QUANTIZE, typename T_OUT, typename T_S, typename T_IN>
__global__ void scaleMatrix(T_OUT* output, T_S const* input_scale, T_IN const* input, int64_t numel, int64_t lda)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    if constexpr (QUANTIZE_MODE == QuantizeMode::PER_TENSOR)
    {
        float const s = static_cast<float>(input_scale[0]);
        float const factor = QUANTIZE ? (1.0f / s) : s;
        for (int64_t i = threadIdx.x + blockIdx.x * blockDim.x; i < numel; i += blockDim.x * gridDim.x)
        {
            output[i] = T_OUT(static_cast<float>(input[i]) * factor);
        }
    }
    else
    {
        for (int64_t i = threadIdx.x + blockIdx.x * blockDim.x; i < numel; i += blockDim.x * gridDim.x)
        {
            if constexpr (QUANTIZE_MODE == QuantizeMode::PER_CHANNEL)
            {
                output[i]
                    = T_OUT(scale<QUANTIZE>(static_cast<float>(input[i]), static_cast<float>(input_scale[i % lda])));
            }
            else if constexpr (QUANTIZE_MODE == QuantizeMode::PER_TOKEN)
            {
                output[i]
                    = T_OUT(scale<QUANTIZE>(static_cast<float>(input[i]), static_cast<float>(input_scale[i / lda])));
            }
        }
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// Helper: can we use the vectorized PER_TENSOR kernel for this type combination?
template <typename T_OUT, typename T_S, typename T_IN>
struct CanUseVecPerTensor : std::false_type
{
};

template <>
struct CanUseVecPerTensor<__nv_fp8_e4m3, float, __nv_bfloat16> : std::true_type
{
};

template <>
struct CanUseVecPerTensor<__nv_fp8_e4m3, __nv_bfloat16, __nv_bfloat16> : std::true_type
{
};

template <typename T_OUT, typename T_S, typename T_IN>
void invokeQuantizeMatrix(T_OUT* output, T_S const* input_scale, T_IN const* input, int64_t numel, int64_t lda,
    QuantizeMode quantize_mode, cudaStream_t stream)
{
    // PER_TENSOR + bf16→fp8: use vectorized kernel (128-bit loads, supports float or bf16 scale)
    if constexpr (CanUseVecPerTensor<T_OUT, T_S, T_IN>::value)
    {
        if (quantize_mode == QuantizeMode::PER_TENSOR)
        {
            dim3 grid(static_cast<unsigned int>(scaleMatrixVecGridSize(numel)));
            dim3 block(CTA_SIZE);
            cudaLaunchConfig_t config;
            config.gridDim = grid;
            config.blockDim = block;
            config.dynamicSmemBytes = 0;
            config.stream = stream;
            cudaLaunchAttribute attrs[1];
            attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
            config.numAttrs = 1;
            config.attrs = attrs;
            cudaLaunchKernelEx(&config, scaleMatrixPerTensorVec<true, T_S>, output, input_scale, input, numel);
            sync_check_cuda_error(stream);
            return;
        }
    }

    dim3 grid(1024);
    dim3 block(CTA_SIZE);
    cudaLaunchConfig_t config;
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    if (quantize_mode == QuantizeMode::PER_CHANNEL)
    {
        cudaLaunchKernelEx(&config, scaleMatrix<QuantizeMode::PER_CHANNEL, true, T_OUT, T_S, T_IN>, output, input_scale,
            input, numel, lda);
    }
    else if (quantize_mode == QuantizeMode::PER_TOKEN)
    {
        cudaLaunchKernelEx(&config, scaleMatrix<QuantizeMode::PER_TOKEN, true, T_OUT, T_S, T_IN>, output, input_scale,
            input, numel, lda);
    }
    else if (quantize_mode == QuantizeMode::PER_TENSOR)
    {
        cudaLaunchKernelEx(&config, scaleMatrix<QuantizeMode::PER_TENSOR, true, T_OUT, T_S, T_IN>, output, input_scale,
            input, numel, lda);
    }
    sync_check_cuda_error(stream);
}

template <typename T_OUT, typename T_S, typename T_IN>
void invokeDequantizeMatrix(T_OUT* output, T_S const* input_scale, T_IN const* input, int64_t numel, int64_t lda,
    QuantizeMode quantize_mode, cudaStream_t stream)
{
    dim3 grid(1024);
    dim3 block(CTA_SIZE);
    cudaLaunchConfig_t config;
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    if (quantize_mode == QuantizeMode::PER_CHANNEL)
    {
        cudaLaunchKernelEx(&config, scaleMatrix<QuantizeMode::PER_CHANNEL, false, T_OUT, T_S, T_IN>, output,
            input_scale, input, numel, lda);
    }
    else if (quantize_mode == QuantizeMode::PER_TOKEN)
    {
        cudaLaunchKernelEx(&config, scaleMatrix<QuantizeMode::PER_TOKEN, false, T_OUT, T_S, T_IN>, output, input_scale,
            input, numel, lda);
    }
    else if (quantize_mode == QuantizeMode::PER_TENSOR)
    {
        cudaLaunchKernelEx(&config, scaleMatrix<QuantizeMode::PER_TENSOR, false, T_OUT, T_S, T_IN>, output, input_scale,
            input, numel, lda);
    }
    sync_check_cuda_error(stream);
}

template <typename T_FAKE, typename T_OUT, typename T_IN>
__global__ void fakeQuantize(T_OUT* dst, const T_IN* src, const int64_t numel)
{
    for (int64_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < numel; tid += blockDim.x * gridDim.x)
    {
        T_FAKE tmp = (T_FAKE) (static_cast<float>(src[tid]));
        dst[tid] = (T_OUT) (static_cast<float>(tmp));
    }
}

template <typename T_FAKE, typename T_OUT, typename T_IN>
void invokeFakeQuantize(T_OUT* dst, const T_IN* src, const int64_t numel, cudaStream_t stream)
{
    fakeQuantize<T_FAKE><<<1024, CTA_SIZE, 0, stream>>>(dst, src, numel);
    sync_check_cuda_error(stream);
}

template void invokeFakeQuantize<__nv_fp8_e4m3, float, float>(
    float* dst, float const* src, const int64_t numel, cudaStream_t stream);
template void invokeFakeQuantize<float, float, __nv_fp8_e4m3>(
    float* dst, __nv_fp8_e4m3 const* src, const int64_t numel, cudaStream_t stream);
template void invokeFakeQuantize<__nv_fp8_e4m3, half, half>(
    half* dst, half const* src, const int64_t numel, cudaStream_t stream);
template void invokeFakeQuantize<__nv_fp8_e4m3, __nv_bfloat16, __nv_bfloat16>(
    __nv_bfloat16* dst, __nv_bfloat16 const* src, const int64_t numel, cudaStream_t stream);

template void invokeFakeQuantize<float, half, float>(
    half* dst, float const* src, const int64_t numel, cudaStream_t stream);

__device__ float atomicMaxExtd(float* address, float val)
{
    assert(val >= 0);
    unsigned int* address_as_u = reinterpret_cast<unsigned int*>(address);
    unsigned int old = atomicMax(address_as_u, __float_as_uint(val));
    return __uint_as_float(old);
}

template <typename T>
inline __device__ T atomicMaxExtdV2(T* address, T val)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    static_assert(std::is_same_v<T, half> | std::is_same_v<T, __nv_bfloat16>, "T needs to be either half or bfloat16");
    // The address in 64 bits.
    uint64_t address_u64 = reinterpret_cast<uint64_t const&>(address);

    // Pack the input value into 32 bits.
    union
    {
        T v[2];
        uint16_t u[2];
    } old, tmp = {};

    int const loc = (address_u64 & 0x2) >> 1;
    tmp.v[loc] = val;

    // 4B aligned pointer.
    auto aligned_address = reinterpret_cast<T*>(address_u64 & ~0x3ull);

    if constexpr (std::is_same_v<T, half>)
    {
        asm volatile("atom.global.v2.f16.max.noftz {%0, %1}, [%2], {%3, %4};"
                     : "=h"(old.u[0]), "=h"(old.u[1])
                     : "l"(aligned_address), "h"(tmp.u[0]), "h"(tmp.u[1]));
    }
    if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
        asm volatile("atom.global.v2.bf16.max.noftz {%0, %1}, [%2], {%3, %4};"
                     : "=h"(old.u[0]), "=h"(old.u[1])
                     : "l"(aligned_address), "h"(tmp.u[0]), "h"(tmp.u[1]));
    }

    // Return the correct half.
    return old.v[loc];
#endif
}

__device__ half atomicMaxExtd(half* address, half val)
{
    unsigned short int* address_as_u = reinterpret_cast<unsigned short int*>(address);
    unsigned short int old = *address_as_u, assumed;

    while (val > __ushort_as_half(old))
    {
        assumed = old;
        old = atomicCAS(address_as_u, assumed, __half_as_ushort(val));
    }

    return __ushort_as_half(old);
}

__device__ __nv_bfloat16 atomicMaxExtd(__nv_bfloat16* address, __nv_bfloat16 val)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    unsigned short int* address_as_u = reinterpret_cast<unsigned short int*>(address);
    unsigned short int old = *address_as_u, assumed;

    while (val > __ushort_as_bfloat16(old))
    {
        assumed = old;
        old = atomicCAS(address_as_u, assumed, __bfloat16_as_ushort(val));
    }

    return __ushort_as_bfloat16(old);
#else
    assert(0);
    asm volatile("brkpt;\n" ::);
    return __nv_bfloat16(0);
#endif
}

template <QuantizeMode QUANTIZE_MODE, typename T_S, typename T_W>
__global__ void computeFP8QuantizeScale(T_S* quant_ptr, const T_W* weights, const int64_t size, const int64_t n)
{
    constexpr float min_scaling_factor = 1.0f / (FP8_E4M3_MAX * 512.f);
    if (QUANTIZE_MODE == QuantizeMode::PER_CHANNEL)
    {
        for (int64_t col = threadIdx.x; col < n; col += blockDim.x)
        {
            float max = 0.f;
            for (int64_t i = col + n * blockIdx.x; i < size; i += gridDim.x * n)
            {
                auto val = fabs(static_cast<float>(weights[i]));
                max = max > val ? max : val;
            }
            auto const scale = (T_S) std::max(max / FP8_E4M3_MAX, min_scaling_factor);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
            if constexpr (std::is_same_v<T_S, float>)
            {
                atomicMaxExtd(quant_ptr + col, scale);
            }
            else
            {
                auto const address_u64 = reinterpret_cast<uint64_t>(quant_ptr + col);
                if ((col == 0 && address_u64 % 4 != 0) || (col == n - 1 && address_u64 % 4 == 0))
                    atomicMaxExtd(quant_ptr + col, scale);
                else
                    atomicMaxExtdV2(quant_ptr + col, scale);
            }
#else // Vector atomics require __CUDA_ARCH__ >= 900
            atomicMaxExtd(quant_ptr + col, scale);
#endif
        }
    }
    else if (QUANTIZE_MODE == QuantizeMode::PER_TOKEN)
    {
        auto const nrows = size / n;
        for (int64_t row = blockIdx.x; row < nrows; row += gridDim.x)
        {
            float max = 0.f;
            for (int64_t i = threadIdx.x; i < n; i += blockDim.x)
            {
                auto val = fabs(static_cast<float>(weights[row * n + i]));
                max = max > val ? max : val;
            }
            max = blockReduceMax<float>(max);
            if (threadIdx.x == 0)
            {
                auto const scale = (T_S) std::max(max / FP8_E4M3_MAX, min_scaling_factor);
                quant_ptr[row] = scale;
            }
        }
    }
    else if (QUANTIZE_MODE == QuantizeMode::PER_TENSOR)
    {
        float max = 0.f;
        for (int64_t i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += gridDim.x * blockDim.x)
        {
            auto val = fabs(static_cast<float>(weights[i]));
            max = max > val ? max : val;
        }
        max = blockReduceMax<float>(max);
        if (threadIdx.x == 0)
        {
            auto const scale = (T_S) std::max(max / FP8_E4M3_MAX, min_scaling_factor);
            atomicMaxExtd(quant_ptr, scale);
        }
    }
}

template <typename T_S, typename T_W>
void invokeComputeFP8QuantizeScale(T_S* quant_ptr, const T_W* weights, const int64_t numel, const int64_t lda,
    QuantizeMode quantize_mode, cudaStream_t stream)
{
    if (quantize_mode == QuantizeMode::PER_TOKEN)
    {
        dim3 block(CTA_SIZE);
        dim3 grid(numel / lda);
        computeFP8QuantizeScale<QuantizeMode::PER_TOKEN><<<grid, block, 0, stream>>>(quant_ptr, weights, numel, lda);
    }
    else if (quantize_mode == QuantizeMode::PER_CHANNEL)
    {
        dim3 block(CTA_SIZE);
        dim3 grid((lda + CTA_SIZE - 1) / CTA_SIZE);
        cudaMemsetAsync(quant_ptr, 0, lda * sizeof(T_S), stream);
        sync_check_cuda_error(stream);
        computeFP8QuantizeScale<QuantizeMode::PER_CHANNEL><<<grid, block, 0, stream>>>(quant_ptr, weights, numel, lda);
    }
    else if (quantize_mode == QuantizeMode::PER_TENSOR)
    {
        dim3 block(1024);
        dim3 grid(1024);
        cudaMemsetAsync(quant_ptr, 0, sizeof(T_S), stream);
        sync_check_cuda_error(stream);
        computeFP8QuantizeScale<QuantizeMode::PER_TENSOR><<<grid, block, 0, stream>>>(quant_ptr, weights, numel, lda);
    }
    sync_check_cuda_error(stream);
}

#define DEFINE_INVOKE_COMPUTE_FP8_QUANTIZE_SCALE(type_scale, type_in)                                                  \
    template void invokeComputeFP8QuantizeScale<type_scale, type_in>(type_scale * input_scale, type_in const* weights, \
        int64_t numel, int64_t lda, QuantizeMode quantize_mode, cudaStream_t stream);

DEFINE_INVOKE_COMPUTE_FP8_QUANTIZE_SCALE(half, half);
DEFINE_INVOKE_COMPUTE_FP8_QUANTIZE_SCALE(float, half);
DEFINE_INVOKE_COMPUTE_FP8_QUANTIZE_SCALE(float, float);
#ifdef ENABLE_BF16
DEFINE_INVOKE_COMPUTE_FP8_QUANTIZE_SCALE(__nv_bfloat16, __nv_bfloat16);
DEFINE_INVOKE_COMPUTE_FP8_QUANTIZE_SCALE(float, __nv_bfloat16);
#endif

template <typename T_OUT, typename T_S, typename T_IN>
__global__ void dynamicQuantizeMatrixPerToken(
    T_OUT* output, T_S* quant_ptr, T_IN const* input, int64_t numel, int64_t lda)
{
    extern __shared__ __align__(sizeof(float)) char _shmem[];
    T_IN* shmem = reinterpret_cast<T_IN*>(_shmem);
    constexpr float min_scaling_factor = 1.0f / (FP8_E4M3_MAX * 512.f);
    auto const nrows = numel / lda;
    for (int64_t row = blockIdx.x; row < nrows; row += gridDim.x)
    {
        float max = 0.f;
        for (int64_t i = threadIdx.x; i < lda; i += blockDim.x)
        {
            auto const in = input[row * lda + i];
            shmem[i] = in;
            auto val = fabs(static_cast<float>(in));
            max = max > val ? max : val;
        }
        max = blockAllReduceMax<float>(max); // __syncthreads() called so we can read shmem
        auto const s = (T_S) std::max(max / FP8_E4M3_MAX, min_scaling_factor);
        for (int64_t i = threadIdx.x; i < lda; i += blockDim.x)
        {
            // true means we are quantizing
            output[row * lda + i] = (T_OUT) scale<true>(static_cast<float>(shmem[i]), static_cast<float>(s));
        }
        if (threadIdx.x == 0)
        {
            quant_ptr[row] = s;
        }
    }
}

template <typename T_OUT, typename T_S, typename T_IN>
void invokeComputeScalesAndQuantizeMatrix(T_OUT* output, T_S* quant_ptr, const T_IN* input, const int64_t numel,
    const int64_t lda, QuantizeMode quantize_mode, cudaStream_t stream)
{
    if (quantize_mode == QuantizeMode::PER_TOKEN)
    {
        dim3 grid(numel / lda);
        bool use_shmem = true;
        auto const shmem_size = lda * sizeof(T_IN);
        if (shmem_size >= (48 << 10))
        {
            cudaError_t ret = cudaFuncSetAttribute(dynamicQuantizeMatrixPerToken<T_OUT, T_S, T_IN>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
            use_shmem = ret == cudaSuccess;
        }
        if (use_shmem)
        {
            // ensure the threadblock is as large as possible to increase occupancy
            dim3 block(std::min((lda + 31) / 32 * 32, static_cast<int64_t>(1024)));
            dynamicQuantizeMatrixPerToken<<<grid, block, shmem_size, stream>>>(output, quant_ptr, input, numel, lda);
        }
        else
        {
            dim3 block(CTA_SIZE);
            computeFP8QuantizeScale<QuantizeMode::PER_TOKEN><<<grid, block, 0, stream>>>(quant_ptr, input, numel, lda);
            sync_check_cuda_error(stream);
            invokeQuantizeMatrix(output, quant_ptr, input, numel, lda, quantize_mode, stream);
        }
    }
    else if (quantize_mode == QuantizeMode::PER_CHANNEL)
    {
        dim3 block(CTA_SIZE);
        dim3 grid((lda + CTA_SIZE - 1) / CTA_SIZE);
        cudaMemsetAsync(quant_ptr, 0, lda * sizeof(T_S), stream);
        sync_check_cuda_error(stream);
        computeFP8QuantizeScale<QuantizeMode::PER_CHANNEL><<<grid, block, 0, stream>>>(quant_ptr, input, numel, lda);
        sync_check_cuda_error(stream);
        invokeQuantizeMatrix(output, quant_ptr, input, numel, lda, quantize_mode, stream);
    }
    else if (quantize_mode == QuantizeMode::PER_TENSOR)
    {
        dim3 block(1024);
        dim3 grid(1024);
        cudaMemsetAsync(quant_ptr, 0, sizeof(T_S), stream);
        sync_check_cuda_error(stream);
        computeFP8QuantizeScale<QuantizeMode::PER_TENSOR><<<grid, block, 0, stream>>>(quant_ptr, input, numel, lda);
        sync_check_cuda_error(stream);
        invokeQuantizeMatrix(output, quant_ptr, input, numel, lda, quantize_mode, stream);
    }
    sync_check_cuda_error(stream);
}

#define DEFINE_INVOKE_QUANTIZE_MATRIX(type_out, type_scale, type_in)                                                   \
    template void invokeQuantizeMatrix<type_out, type_scale, type_in>(type_out * output,                               \
        type_scale const* input_scale, type_in const* input, int64_t numel, int64_t lda, QuantizeMode quantize_mode,   \
        cudaStream_t stream);                                                                                          \
    template void invokeDequantizeMatrix<type_out, type_scale, type_in>(type_out * output,                             \
        type_scale const* input_scale, type_in const* input, int64_t numel, int64_t lda, QuantizeMode quantize_mode,   \
        cudaStream_t stream);                                                                                          \
    template void invokeComputeScalesAndQuantizeMatrix<type_out, type_scale, type_in>(type_out * output,               \
        type_scale * input_scale, type_in const* input, int64_t numel, int64_t lda, QuantizeMode quantize_mode,        \
        cudaStream_t stream);

#ifdef ENABLE_FP8
DEFINE_INVOKE_QUANTIZE_MATRIX(__nv_fp8_e4m3, float, float);
DEFINE_INVOKE_QUANTIZE_MATRIX(__nv_fp8_e4m3, float, half);
DEFINE_INVOKE_QUANTIZE_MATRIX(__nv_fp8_e4m3, half, half);
DEFINE_INVOKE_QUANTIZE_MATRIX(half, half, __nv_fp8_e4m3);
DEFINE_INVOKE_QUANTIZE_MATRIX(float, float, __nv_fp8_e4m3);
DEFINE_INVOKE_QUANTIZE_MATRIX(half, float, __nv_fp8_e4m3);
#ifdef ENABLE_BF16
DEFINE_INVOKE_QUANTIZE_MATRIX(__nv_fp8_e4m3, float, __nv_bfloat16);
DEFINE_INVOKE_QUANTIZE_MATRIX(__nv_fp8_e4m3, __nv_bfloat16, __nv_bfloat16);
DEFINE_INVOKE_QUANTIZE_MATRIX(__nv_bfloat16, __nv_bfloat16, __nv_fp8_e4m3);
DEFINE_INVOKE_QUANTIZE_MATRIX(__nv_bfloat16, float, __nv_fp8_e4m3);
#endif
#endif

#endif // ENABLE_FP8
} // namespace common

TRTLLM_NAMESPACE_END
