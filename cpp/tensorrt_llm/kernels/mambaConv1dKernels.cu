/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda_runtime_api.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif

#include "mambaConv1dKernels.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaBf16Fallbacks.cuh"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"

extern "C" __device__ uint32_t __nvvm_get_smem_pointer(void* ptr);

template <int mode_, int line_, typename T_>
__device__ static inline int swizzle(int x_)
{
    return x_ ^ x_ / line_ % (mode_ / 16) * (16 / sizeof(T_));
}

template <int size_, bool aligned_ = true>
__device__ static inline void cp_shared_global(uint32_t s_ptr, void const* g_ptr)
{
    static_assert(size_ == 4 || size_ == 8 || size_ == 16);

    if constexpr (aligned_)
    {
#if __CUDA_ARCH__ >= 800
        asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(s_ptr), "l"(g_ptr), "n"(size_));
#else
        uint32_t tmp[size_ / 4];

        if constexpr (size_ == 16)
        {
            asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(tmp[0]), "=r"(tmp[1]), "=r"(tmp[2]), "=r"(tmp[3])
                         : "l"(g_ptr));
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n" ::"r"(s_ptr), "r"(tmp[0]), "r"(tmp[1]),
                "r"(tmp[2]), "r"(tmp[3]));
        }
        else if constexpr (size_ == 8)
        {
            asm volatile("ld.global.v2.b32 {%0, %1}, [%2];\n" : "=r"(tmp[0]), "=r"(tmp[1]) : "l"(g_ptr));
            asm volatile("st.shared.v2.b32 [%0], {%1, %2};\n" ::"r"(s_ptr), "r"(tmp[0]), "r"(tmp[1]));
        }
        else if constexpr (size_ == 4)
        {
            asm volatile("ld.global.b32 %0, [%1];\n" : "=r"(tmp[0]) : "l"(g_ptr));
            asm volatile("st.shared.b32 [%0], %1;\n" ::"r"(s_ptr), "r"(tmp[0]));
        }
#endif
    }
    else
    {
        uint32_t tmp[size_ / 4];

#pragma unroll
        for (int i = 0; i < size_ / 4; i++)
        {
            asm volatile("ld.global.b32 %0, [%1];\n" : "=r"(tmp[i]) : "l"((int*) g_ptr + i));
            asm volatile("st.shared.b32 [%0], %1;\n" ::"r"(s_ptr + i * 4), "r"(tmp[i]));
        }
    }
}

__device__ static inline void cp_commit_group()
{
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n");
#endif
}

template <int remain_>
__device__ static inline void cp_wait_group()
{
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" ::"n"(remain_));
#endif
}

namespace tensorrt_llm
{
namespace kernels
{

template <typename T, int N>
struct packed_data_type;

template <typename T>
struct packed_data_type<T, 1>
{
    using type = T;
};

template <>
struct packed_data_type<half, 2>
{
    using type = uint32_t;
};

template <>
struct packed_data_type<half, 4>
{
    using type = uint2;
};

template <>
struct packed_data_type<half, 8>
{
    using type = uint4;
};

#ifdef ENABLE_BF16
template <>
struct packed_data_type<__nv_bfloat16, 2>
{
    using type = uint32_t;
};

template <>
struct packed_data_type<__nv_bfloat16, 4>
{
    using type = uint2;
};

template <>
struct packed_data_type<__nv_bfloat16, 8>
{
    using type = uint4;
};
#endif

template <>
struct packed_data_type<float, 2>
{
    using type = float2;
};

template <>
struct packed_data_type<float, 4>
{
    using type = float4;
};

template <typename T, int N>
__device__ __forceinline__ void packed_move(T const* from_ptr, T* to_ptr)
{
    using load_type = typename packed_data_type<T, N>::type;
    *reinterpret_cast<load_type*>(to_ptr) = *reinterpret_cast<load_type const*>(from_ptr);
}

template <typename T, int N>
__device__ __forceinline__ void packed_load_to_float(T const* from_ptr, float* to_ptr)
{
    T tmp_data[N];
    packed_move<T, N>(from_ptr, &tmp_data[0]);
#pragma unroll
    for (int i = 0; i < N; i++)
    {
        to_ptr[i] = tensorrt_llm::common::cuda_cast<float, T>(tmp_data[i]);
    }
}

template <typename T, int N>
__device__ __forceinline__ void packed_store_float_to(float const* from_ptr, T* to_ptr)
{
    T tmp_data[N];
#pragma unroll
    for (int i = 0; i < N; ++i)
    {
        tmp_data[i] = tensorrt_llm::common::cuda_cast<T, float>(from_ptr[i]);
    }
    packed_move<T, N>(&tmp_data[0], to_ptr);
}

template <typename T, int N>
__device__ __forceinline__ void packed_zero(T* to_ptr)
{
    T tmp_data[N];
#pragma unroll
    for (int i = 0; i < N; ++i)
    {
        tmp_data[i] = tensorrt_llm::common::cuda_cast<T, float>(0.0f);
    }
    packed_move<T, N>(&tmp_data[0], to_ptr);
}

template <int K_, int tileL_, int tileD_, int warpL_, int warpD_, int laneD_, int pipe_, bool aligned_, typename T_>
__global__
    std::enable_if_t<std::is_same_v<T_, half>
#ifdef ENABLE_BF16
        || std::is_same_v<T_, __nv_bfloat16>
#endif
        >
    mambaConv1dContextKernel(int B_, int L_, int D_, int S_pre_, int S_post_, T_* g_mxYa_, T_* g_mxYs_,
        T_ const* g_mxXa_, T_ const* g_mxXs_, T_ const* g_mxW_, T_ const* g_mxB_, bool removePadding_, bool applySilu_,
        int const* lastTokenIdsPtr_, int const* stateSlotMappingPtr_ = nullptr)
{
    using namespace tensorrt_llm::common;

    static_assert(laneD_ >= 1 && laneD_ <= 32 && (laneD_ & (laneD_ - 1)) == 0);

    constexpr int laneL = 32 / laneD_;

    float weight[K_][tileD_ / warpD_ / laneD_][8];
    float bias[tileD_ / warpD_ / laneD_][8];

#pragma unroll
    for (int iK = 0; iK < K_; iK++)
#pragma unroll
        for (int iD = 0; iD < tileD_ / (warpD_ * laneD_ * 8); iD++)
        {
            T_ tmp[8];

            *(int4*) &tmp = *(int4*) &g_mxW_[iK * D_ + blockIdx.x * tileD_ + iD * warpD_ * laneD_ * 8
                + threadIdx.y * laneD_ * 8 + threadIdx.x % laneD_ * 8];

#pragma unroll
            for (int i = 0; i < 8; i += 2)
                *(float2*) &weight[iK][iD][i] = std::is_same_v<T_, half> ? __half22float2(*(half2*) &tmp[i])
#ifdef ENABLE_BF16
                                                                         : bf1622float2(*(__nv_bfloat162*) &tmp[i]);
#else
                                                                         : float2{0.f, 0.f};
#endif
        }

#pragma unroll
    for (int iD = 0; iD < tileD_ / (warpD_ * laneD_ * 8); iD++)
    {
        T_ tmp[8];

        *(int4*) &tmp = *(int4*) &g_mxB_[blockIdx.x * tileD_ + iD * warpD_ * laneD_ * 8 + threadIdx.y * laneD_ * 8
            + threadIdx.x % laneD_ * 8];

#pragma unroll
        for (int i = 0; i < 8; i += 2)
            *(float2*) &bias[iD][i] = std::is_same_v<T_, half> ? __half22float2(*(half2*) &tmp[i])
#ifdef ENABLE_BF16
                                                               : bf1622float2(*(__nv_bfloat162*) &tmp[i]);
#else
                                                               : float2{0.f, 0.f};
#endif
    }

    extern __shared__ float smem[];

    T_* s_mxX = (T_*) smem;
    T_* s_mxO = (T_*) smem + warpL_ * laneL * tileD_ * pipe_;

    uint32_t base = __nvvm_get_smem_pointer(smem);

    uint32_t b_mxX = base;
    // uint32_t b_mxO = base + warpL_ * laneL * tileD_ * pipe_ * sizeof(T_);

    int thread = threadIdx.z * 32 * warpD_ + threadIdx.y * 32 + threadIdx.x;
    int STEP = 256 * warpL_ * warpD_;

    int L = L_;
    int DS_ = (D_ + S_pre_ + S_post_);

    long aIStart = long(blockIdx.z) * L_ * DS_;
    long aOStart = long(blockIdx.z) * L_ * D_;
    int sStart = blockIdx.z * (K_ - 1) * D_;
    long lStart = blockIdx.y * tileL_;
    int dStart = blockIdx.x * tileD_;

    if (removePadding_)
    {
        aIStart = blockIdx.z ? lastTokenIdsPtr_[blockIdx.z - 1] : 0;
        L = lastTokenIdsPtr_[blockIdx.z] - aIStart;
        aOStart = aIStart * D_;
        aIStart = aIStart * DS_;
    }
    else
    {
        L = lastTokenIdsPtr_[blockIdx.z];
    }

    if (stateSlotMappingPtr_)
    {
        sStart = stateSlotMappingPtr_[blockIdx.z] * (K_ - 1) * D_;
    }

    if (lStart >= L)
        return;
    else if (lStart)
    {
#pragma unroll
        for (int i = 0; i < (K_ - 1) * tileD_; i += STEP)
            if (i + STEP <= (K_ - 1) * tileD_ || i + thread * 8 < (K_ - 1) * tileD_)
                cp_shared_global<16, aligned_>(b_mxX
                        + 2
                            * swizzle<tileD_ * 2, tileD_, T_>(
                                i + thread * 8 + (warpL_ * laneL * pipe_ + 1 - K_) * tileD_),
                    g_mxXa_ + aIStart + (1 - K_ + lStart + thread * 8 / tileD_) * DS_ + S_pre_ + i * (D_ / tileD_)
                        + dStart + thread * 8 % tileD_);
    }
    else if (g_mxXs_)
    {
#pragma unroll
        for (int i = 0; i < (K_ - 1) * tileD_; i += STEP)
            if (i + STEP <= (K_ - 1) * tileD_ || i + thread * 8 < (K_ - 1) * tileD_)
                cp_shared_global<16, aligned_>(b_mxX
                        + 2
                            * swizzle<tileD_ * 2, tileD_, T_>(
                                i + thread * 8 + (warpL_ * laneL * pipe_ + 1 - K_) * tileD_),
                    g_mxXs_ + sStart + (thread * 8 / tileD_) * D_ + i * (D_ / tileD_) + dStart + thread * 8 % tileD_);
    }
    else
    {
#pragma unroll
        for (int i = 0; i < (K_ - 1) * tileD_; i += STEP)
            if (i + STEP <= (K_ - 1) * tileD_ || i + thread * 8 < (K_ - 1) * tileD_)
                *(int4*) &s_mxX[swizzle<tileD_ * 2, tileD_, T_>(
                    i + thread * 8 + (warpL_ * laneL * pipe_ + 1 - K_) * tileD_)]
                    = int4{0, 0, 0, 0};
    }

    cp_commit_group();

#pragma unroll
    for (int iL = 0; iL < pipe_ - 1; iL++)
    {
        if (lStart + (iL + 1) * warpL_ * laneL <= L)
        {
#pragma unroll
            for (int i = 0; i < warpL_ * laneL * tileD_; i += STEP)
                if (i + STEP <= warpL_ * laneL * tileD_ || i + thread * 8 < warpL_ * laneL * tileD_)
                    cp_shared_global<16, aligned_>(
                        b_mxX + 2 * swizzle<tileD_ * 2, tileD_, T_>(i + thread * 8 + iL * warpL_ * laneL * tileD_),
                        g_mxXa_ + aIStart + iL * warpL_ * laneL * DS_ + (lStart + thread * 8 / tileD_) * DS_ + S_pre_
                            + i * (D_ / tileD_) + dStart + thread * 8 % tileD_);
        }
        else
        {
#pragma unroll
            for (int i = 0; i < warpL_ * laneL * tileD_; i += STEP)
                if (i + thread * 8 < (L - lStart - iL * warpL_ * laneL) * tileD_)
                    cp_shared_global<16, aligned_>(
                        b_mxX + 2 * swizzle<tileD_ * 2, tileD_, T_>(i + thread * 8 + iL * warpL_ * laneL * tileD_),
                        g_mxXa_ + aIStart + iL * warpL_ * laneL * DS_ + (lStart + thread * 8 / tileD_) * DS_ + S_pre_
                            + i * (D_ / tileD_) + dStart + thread * 8 % tileD_);
        }

        cp_commit_group();
    }

#pragma unroll
    for (int iL = 0; iL < tileL_ / (warpL_ * laneL); iL++)
    {
        cp_wait_group<pipe_ - 2>();

        __syncthreads();

#pragma unroll
        for (int iD = 0; iD < tileD_ / (warpD_ * laneD_ * 8); iD++)
        {
            float sum[8];
            T_ tmp[8];

#pragma unroll
            for (int i = 0; i < 8; i += 4)
                *(int4*) &sum[i] = *(int4*) &bias[iD][i];

#pragma unroll
            for (int iK = 0; iK < K_; iK++)
            {
                *(int4*) &tmp = *(int4*) &s_mxX[swizzle<tileD_ * 2, tileD_, T_>(
                    (iL * warpL_ * laneL + threadIdx.z * laneL + threadIdx.x / laneD_ + warpL_ * laneL * pipe_ + 1 - K_
                        + iK)
                        % (warpL_ * laneL * pipe_) * tileD_
                    + iD * warpD_ * laneD_ * 8 + threadIdx.y * laneD_ * 8 + threadIdx.x % laneD_ * 8)];

#pragma unroll
                for (int i = 0; i < 8; i += 2)
                {
                    float2 f32 = std::is_same_v<T_, half> ? __half22float2(*(half2*) &tmp[i])
#ifdef ENABLE_BF16
                                                          : bf1622float2(*(__nv_bfloat162*) &tmp[i]);
#else
                                                          : float2{0.f, 0.f};
#endif

                    sum[i] += f32.x * weight[iK][iD][i];
                    sum[i + 1] += f32.y * weight[iK][iD][i + 1];
                }
            }

            if (applySilu_)
            {
                if (std::is_same_v<T_, half>)
#pragma unroll
                    for (int i = 0; i < 8; i += 2)
                        *(half2*) &tmp[i]
                            = __floats2half2_rn(sum[i] / (1 + exp(-sum[i])), sum[i + 1] / (1 + exp(-sum[i + 1])));
#ifdef ENABLE_BF16
                else
#pragma unroll
                    for (int i = 0; i < 8; i += 2)
                        *(__nv_bfloat162*) &tmp[i]
                            = __floats2bfloat162_rn(sum[i] / (1 + exp(-sum[i])), sum[i + 1] / (1 + exp(-sum[i + 1])));
#endif
            }
            else
            {
                if (std::is_same_v<T_, half>)
#pragma unroll
                    for (int i = 0; i < 8; i += 2)
                        *(half2*) &tmp[i] = __floats2half2_rn(sum[i], sum[i + 1]);
#ifdef ENABLE_BF16
                else
#pragma unroll
                    for (int i = 0; i < 8; i += 2)
                        *(__nv_bfloat162*) &tmp[i] = __floats2bfloat162_rn(sum[i], sum[i + 1]);
#endif
            }

            *(int4*) &s_mxO[swizzle<tileD_ * 2, tileD_, T_>((threadIdx.z * laneL + threadIdx.x / laneD_) * tileD_
                + iD * warpD_ * laneD_ * 8 + threadIdx.y * laneD_ * 8 + threadIdx.x % laneD_ * 8)]
                = *(int4*) &tmp;
        }

        __syncthreads();

        int jL = iL + (pipe_ - 1);

        if (jL < tileL_ / (warpL_ * laneL) && lStart + (jL + 1) * warpL_ * laneL <= L)
        {
#pragma unroll
            for (int i = 0; i < warpL_ * laneL * tileD_; i += STEP)
                if (i + STEP <= warpL_ * laneL * tileD_ || i + thread * 8 < warpL_ * laneL * tileD_)
                    cp_shared_global<16, aligned_>(b_mxX
                            + 2
                                * swizzle<tileD_ * 2, tileD_, T_>(
                                    i + thread * 8 + jL % pipe_ * warpL_ * laneL * tileD_),
                        g_mxXa_ + aIStart + jL * warpL_ * laneL * DS_ + (lStart + thread * 8 / tileD_) * DS_ + S_pre_
                            + i * (D_ / tileD_) + dStart + thread * 8 % tileD_);
        }
        else if (jL < tileL_ / (warpL_ * laneL))
        {
#pragma unroll
            for (int i = 0; i < warpL_ * laneL * tileD_; i += STEP)
                if (i + thread * 8 < (L - lStart - jL * warpL_ * laneL) * tileD_)
                    cp_shared_global<16, aligned_>(b_mxX
                            + 2
                                * swizzle<tileD_ * 2, tileD_, T_>(
                                    i + thread * 8 + jL % pipe_ * warpL_ * laneL * tileD_),
                        g_mxXa_ + aIStart + jL * warpL_ * laneL * DS_ + (lStart + thread * 8 / tileD_) * DS_ + S_pre_
                            + i * (D_ / tileD_) + dStart + thread * 8 % tileD_);
        }

        cp_commit_group();

        int offset
            = aOStart + iL * warpL_ * laneL * D_ + (lStart + thread * 8 / tileD_) * D_ + dStart + thread * 8 % tileD_;

        if (lStart + (iL + 1) * warpL_ * laneL <= L)
        {
#pragma unroll
            for (int i = 0; i < warpL_ * laneL * tileD_; i += STEP)
                if (i + STEP <= warpL_ * laneL * tileD_ || i + thread * 8 < warpL_ * laneL * tileD_)
                    *(int4*) &g_mxYa_[offset + i * (D_ / tileD_)]
                        = *(int4*) &s_mxO[swizzle<tileD_ * 2, tileD_, T_>(i + thread * 8)];
        }
        else
        {
#pragma unroll
            for (int i = 0; i < warpL_ * laneL * tileD_; i += STEP)
                if (i + thread * 8 < (L - lStart - iL * warpL_ * laneL) * tileD_)
                    *(int4*) &g_mxYa_[offset + i * (D_ / tileD_)]
                        = *(int4*) &s_mxO[swizzle<tileD_ * 2, tileD_, T_>(i + thread * 8)];
        }
    }

    cp_wait_group<0>();

    if (lStart + tileL_ == L)
    {
#pragma unroll
        for (int i = 0; i < (K_ - 1) * tileD_; i += STEP)
            if (i + STEP <= (K_ - 1) * tileD_ || i + thread * 8 < (K_ - 1) * tileD_)
                *(int4*) &g_mxYs_[sStart + (thread * 8 / tileD_) * D_ + i * (D_ / tileD_) + dStart
                    + thread * 8 % tileD_]
                    = *(int4*) &s_mxX[swizzle<tileD_ * 2, tileD_, T_>(
                        (tileL_ + 1 - K_) % (warpL_ * laneL * pipe_) * tileD_ + i + thread * 8)];
    }
    else if (lStart + tileL_ > L)
    {
#pragma unroll
        for (int i = 0; i < (K_ - 1) * tileD_; i += STEP)
            if (i + STEP <= (K_ - 1) * tileD_ || i + thread * 8 < (K_ - 1) * tileD_)
                *(int4*) &g_mxYs_[sStart + (thread * 8 / tileD_) * D_ + i * (D_ / tileD_) + dStart
                    + thread * 8 % tileD_]
                    = *(int4*) &s_mxX[swizzle<tileD_ * 2, tileD_, T_>(
                        ((L - lStart + 1 - K_ + warpL_ * laneL * pipe_) * tileD_ + i + thread * 8)
                        % (warpL_ * laneL * pipe_ * tileD_))];
    }
}

template <int K_, int tileL_, int tileD_, int warpL_, int warpD_, int laneD_, int pipe_, bool aligned_, typename T_>
__global__ std::enable_if_t<std::is_same_v<T_, float>> mambaConv1dContextKernel(int B_, int L_, int D_, int S_pre_,
    int S_post_, T_* g_mxYa_, T_* g_mxYs_, T_ const* g_mxXa_, T_ const* g_mxXs_, T_ const* g_mxW_, T_ const* g_mxB_,
    bool removePadding_, bool applySilu_, int const* lastTokenIdsPtr_, int const* stateSlotMappingPtr_ = nullptr)
{
    static_assert(laneD_ >= 1 && laneD_ <= 32 && (laneD_ & (laneD_ - 1)) == 0);

    constexpr int laneL = 32 / laneD_;

    float weight[K_][tileD_ / warpD_ / laneD_][4];
    float bias[tileD_ / warpD_ / laneD_][4];

#pragma unroll
    for (int iK = 0; iK < K_; iK++)
#pragma unroll
        for (int iD = 0; iD < tileD_ / (warpD_ * laneD_ * 4); iD++)
        {
            *(int4*) &weight[iK][iD] = *(int4*) &g_mxW_[iK * D_ + blockIdx.x * tileD_ + iD * warpD_ * laneD_ * 4
                + threadIdx.y * laneD_ * 4 + threadIdx.x % laneD_ * 4];
        }

#pragma unroll
    for (int iD = 0; iD < tileD_ / (warpD_ * laneD_ * 4); iD++)
    {
        *(int4*) &bias[iD] = *(int4*) &g_mxB_[blockIdx.x * tileD_ + iD * warpD_ * laneD_ * 4 + threadIdx.y * laneD_ * 4
            + threadIdx.x % laneD_ * 4];
    }

    extern __shared__ float smem[];

    T_* s_mxX = (T_*) smem;
    T_* s_mxO = (T_*) smem + warpL_ * laneL * tileD_ * pipe_;

    uint32_t base = __nvvm_get_smem_pointer(smem);

    uint32_t b_mxX = base;
    // uint32_t b_mxO = base + warpL_ * laneL * tileD_ * pipe_ * sizeof(T_);

    int thread = threadIdx.z * 32 * warpD_ + threadIdx.y * 32 + threadIdx.x;
    int STEP = 128 * warpL_ * warpD_;

    int L = L_;
    int DS_ = (D_ + S_pre_ + S_post_);

    long aIStart = long(blockIdx.z) * L_ * DS_;
    long aOStart = long(blockIdx.z) * L_ * D_;
    int sStart = blockIdx.z * (K_ - 1) * D_;
    long lStart = blockIdx.y * tileL_;
    int dStart = blockIdx.x * tileD_;

    if (removePadding_)
    {
        aIStart = blockIdx.z ? lastTokenIdsPtr_[blockIdx.z - 1] : 0;
        L = lastTokenIdsPtr_[blockIdx.z] - aIStart;
        aOStart = aIStart * D_;
        aIStart = aIStart * DS_;
    }
    else
    {
        L = lastTokenIdsPtr_[blockIdx.z];
    }

    if (stateSlotMappingPtr_)
    {
        sStart = stateSlotMappingPtr_[blockIdx.z] * (K_ - 1) * D_;
    }

    if (lStart >= L)
        return;
    else if (lStart)
    {
#pragma unroll
        for (int i = 0; i < (K_ - 1) * tileD_; i += STEP)
            if (i + STEP <= (K_ - 1) * tileD_ || i + thread * 4 < (K_ - 1) * tileD_)
                cp_shared_global<16, aligned_>(b_mxX
                        + 4
                            * swizzle<tileD_ * 4, tileD_, T_>(
                                i + thread * 4 + (warpL_ * laneL * pipe_ + 1 - K_) * tileD_),
                    g_mxXa_ + aIStart + (1 - K_ + lStart + thread * 4 / tileD_) * DS_ + S_pre_ + i * (D_ / tileD_)
                        + dStart + thread * 4 % tileD_);
    }
    else if (g_mxXs_)
    {
#pragma unroll
        for (int i = 0; i < (K_ - 1) * tileD_; i += STEP)
            if (i + STEP <= (K_ - 1) * tileD_ || i + thread * 4 < (K_ - 1) * tileD_)
                cp_shared_global<16, aligned_>(b_mxX
                        + 4
                            * swizzle<tileD_ * 4, tileD_, T_>(
                                i + thread * 4 + (warpL_ * laneL * pipe_ + 1 - K_) * tileD_),
                    g_mxXs_ + sStart + (thread * 4 / tileD_) * D_ + i * (D_ / tileD_) + dStart + thread * 4 % tileD_);
    }
    else
    {
#pragma unroll
        for (int i = 0; i < (K_ - 1) * tileD_; i += STEP)
            if (i + STEP <= (K_ - 1) * tileD_ || i + thread * 4 < (K_ - 1) * tileD_)
                *(int4*) &s_mxX[swizzle<tileD_ * 4, tileD_, T_>(
                    i + thread * 4 + (warpL_ * laneL * pipe_ + 1 - K_) * tileD_)]
                    = int4{0, 0, 0, 0};
    }

    cp_commit_group();

#pragma unroll
    for (int iL = 0; iL < pipe_ - 1; iL++)
    {
        if (lStart + (iL + 1) * warpL_ * laneL <= L)
        {
#pragma unroll
            for (int i = 0; i < warpL_ * laneL * tileD_; i += STEP)
                if (i + STEP <= warpL_ * laneL * tileD_ || i + thread * 4 < warpL_ * laneL * tileD_)
                    cp_shared_global<16, aligned_>(
                        b_mxX + 4 * swizzle<tileD_ * 4, tileD_, T_>(i + thread * 4 + iL * warpL_ * laneL * tileD_),
                        g_mxXa_ + aIStart + iL * warpL_ * laneL * DS_ + (lStart + thread * 4 / tileD_) * DS_ + S_pre_
                            + i * (D_ / tileD_) + dStart + thread * 4 % tileD_);
        }
        else
        {
#pragma unroll
            for (int i = 0; i < warpL_ * laneL * tileD_; i += STEP)
                if (i + thread * 4 < (L - lStart - iL * warpL_ * laneL) * tileD_)
                    cp_shared_global<16, aligned_>(
                        b_mxX + 4 * swizzle<tileD_ * 4, tileD_, T_>(i + thread * 4 + iL * warpL_ * laneL * tileD_),
                        g_mxXa_ + aIStart + iL * warpL_ * laneL * DS_ + (lStart + thread * 4 / tileD_) * DS_ + S_pre_
                            + i * (D_ / tileD_) + dStart + thread * 4 % tileD_);
        }

        cp_commit_group();
    }

#pragma unroll
    for (int iL = 0; iL < tileL_ / (warpL_ * laneL); iL++)
    {
        cp_wait_group<pipe_ - 2>();

        __syncthreads();

#pragma unroll
        for (int iD = 0; iD < tileD_ / (warpD_ * laneD_ * 4); iD++)
        {
            float sum[4];
            T_ tmp[4];

#pragma unroll
            for (int i = 0; i < 4; i += 4)
                *(int4*) &sum[i] = *(int4*) &bias[iD][i];

#pragma unroll
            for (int iK = 0; iK < K_; iK++)
            {
                *(int4*) &tmp = *(int4*) &s_mxX[swizzle<tileD_ * 4, tileD_, T_>(
                    (iL * warpL_ * laneL + threadIdx.z * laneL + threadIdx.x / laneD_ + warpL_ * laneL * pipe_ + 1 - K_
                        + iK)
                        % (warpL_ * laneL * pipe_) * tileD_
                    + iD * warpD_ * laneD_ * 4 + threadIdx.y * laneD_ * 4 + threadIdx.x % laneD_ * 4)];

#pragma unroll
                for (int i = 0; i < 4; i++)
                    sum[i] += tmp[i] * weight[iK][iD][i];
            }

            if (applySilu_)
            {
#pragma unroll
                for (int i = 0; i < 4; i++)
                    sum[i] = sum[i] / (1 + exp(-sum[i]));
            }

            *(int4*) &s_mxO[swizzle<tileD_ * 4, tileD_, T_>((threadIdx.z * laneL + threadIdx.x / laneD_) * tileD_
                + iD * warpD_ * laneD_ * 4 + threadIdx.y * laneD_ * 4 + threadIdx.x % laneD_ * 4)]
                = *(int4*) &sum;
        }

        __syncthreads();

        int jL = iL + (pipe_ - 1);

        if (jL < tileL_ / (warpL_ * laneL) && lStart + (jL + 1) * warpL_ * laneL <= L)
        {
#pragma unroll
            for (int i = 0; i < warpL_ * laneL * tileD_; i += STEP)
                if (i + STEP <= warpL_ * laneL * tileD_ || i + thread * 4 < warpL_ * laneL * tileD_)
                    cp_shared_global<16, aligned_>(b_mxX
                            + 4
                                * swizzle<tileD_ * 4, tileD_, T_>(
                                    i + thread * 4 + jL % pipe_ * warpL_ * laneL * tileD_),
                        g_mxXa_ + aIStart + jL * warpL_ * laneL * DS_ + (lStart + thread * 4 / tileD_) * DS_ + S_pre_
                            + i * (D_ / tileD_) + dStart + thread * 4 % tileD_);
        }
        else if (jL < tileL_ / (warpL_ * laneL))
        {
#pragma unroll
            for (int i = 0; i < warpL_ * laneL * tileD_; i += STEP)
                if (i + thread * 4 < (L - lStart - jL * warpL_ * laneL) * tileD_)
                    cp_shared_global<16, aligned_>(b_mxX
                            + 4
                                * swizzle<tileD_ * 4, tileD_, T_>(
                                    i + thread * 4 + jL % pipe_ * warpL_ * laneL * tileD_),
                        g_mxXa_ + aIStart + jL * warpL_ * laneL * DS_ + (lStart + thread * 4 / tileD_) * DS_ + S_pre_
                            + i * (D_ / tileD_) + dStart + thread * 4 % tileD_);
        }

        cp_commit_group();

        int offset
            = aOStart + iL * warpL_ * laneL * D_ + (lStart + thread * 4 / tileD_) * D_ + dStart + thread * 4 % tileD_;

        if (lStart + (iL + 1) * warpL_ * laneL <= L)
        {
#pragma unroll
            for (int i = 0; i < warpL_ * laneL * tileD_; i += STEP)
                if (i + STEP <= warpL_ * laneL * tileD_ || i + thread * 4 < warpL_ * laneL * tileD_)
                    *(int4*) &g_mxYa_[offset + i * (D_ / tileD_)]
                        = *(int4*) &s_mxO[swizzle<tileD_ * 4, tileD_, T_>(i + thread * 4)];
        }
        else
        {
#pragma unroll
            for (int i = 0; i < warpL_ * laneL * tileD_; i += STEP)
                if (i + thread * 4 < (L - lStart - iL * warpL_ * laneL) * tileD_)
                    *(int4*) &g_mxYa_[offset + i * (D_ / tileD_)]
                        = *(int4*) &s_mxO[swizzle<tileD_ * 4, tileD_, T_>(i + thread * 4)];
        }
    }

    cp_wait_group<0>();

    if (lStart + tileL_ == L)
    {
#pragma unroll
        for (int i = 0; i < (K_ - 1) * tileD_; i += STEP)
            if (i + STEP <= (K_ - 1) * tileD_ || i + thread * 4 < (K_ - 1) * tileD_)
                *(int4*) &g_mxYs_[sStart + (thread * 4 / tileD_) * D_ + i * (D_ / tileD_) + dStart
                    + thread * 4 % tileD_]
                    = *(int4*) &s_mxX[swizzle<tileD_ * 4, tileD_, T_>(
                        (tileL_ + 1 - K_) % (warpL_ * laneL * pipe_) * tileD_ + i + thread * 4)];
    }
    else if (lStart + tileL_ > L)
    {
#pragma unroll
        for (int i = 0; i < (K_ - 1) * tileD_; i += STEP)
            if (i + STEP <= (K_ - 1) * tileD_ || i + thread * 4 < (K_ - 1) * tileD_)
                *(int4*) &g_mxYs_[sStart + (thread * 4 / tileD_) * D_ + i * (D_ / tileD_) + dStart
                    + thread * 4 % tileD_]
                    = *(int4*) &s_mxX[swizzle<tileD_ * 4, tileD_, T_>(
                        ((L - lStart + 1 - K_ + warpL_ * laneL * pipe_) * tileD_ + i + thread * 4)
                        % (warpL_ * laneL * pipe_ * tileD_))];
    }
}

template <typename input_t>
void invokeMambaConv1dContext(MambaConv1dParamsBase& params, cudaStream_t stream)
{
    int B = params.batch;
    int L = params.max_seqlen;
    int D = params.dim;
    int K = params.dconv;
    int S_pre = params.pre_stride;
    int S_post = params.post_stride;
    int DS = D + S_pre + S_post;
    bool aligned = DS * sizeof(input_t) % 16 == 0;

    int tileL = 32;
    int tileD = 128;
    int warpL = 1;
    int warpD = 4;
    int laneD = 4;
    int pipe = 4;

    void (*f)(int B_, int L_, int D_, int S_pre_, int S_post_, input_t* g_mxYa_, input_t* g_mxYs_,
        input_t const* g_mxXa_, input_t const* g_mxXs_, input_t const* g_mxW_, input_t const* g_mxB_,
        bool removePadding_, bool applySilu_, int const* lastTokenIdsPtr_, int const* stateSlotMappingPtr_);

    if (std::is_same_v<input_t, float>)
    {
        if (tensorrt_llm::common::getSMVersion() >= 90 && tensorrt_llm::common::getSMVersion() < 120)
        {
            if (B * L * D <= 262144)
            {
                tileD = 32;
                warpD = 8;
                laneD = 1;
                pipe = 2;
                if (aligned)
                    f = mambaConv1dContextKernel<4, 32, 32, 1, 8, 1, 2, true>;
                else
                    f = mambaConv1dContextKernel<4, 32, 32, 1, 8, 1, 2, false>;
            }
            else if (B * L * D <= 524288 || D % 64 == 32)
            {
                tileD = 32;
                warpD = 4;
                laneD = 1;
                pipe = 2;
                if (aligned)
                    f = mambaConv1dContextKernel<4, 32, 32, 1, 4, 1, 2, true>;
                else
                    f = mambaConv1dContextKernel<4, 32, 32, 1, 4, 1, 2, false>;
            }
            else if (B * L * D <= 1048576)
            {
                tileD = 64;
                warpD = 4;
                laneD = 4;
                pipe = 4;
                if (aligned)
                    f = mambaConv1dContextKernel<4, 32, 64, 1, 4, 4, 4, true>;
                else
                    f = mambaConv1dContextKernel<4, 32, 64, 1, 4, 4, 4, false>;
            }
            else if (B * L * D <= 4194304)
            {
                if (D % 128 == 64)
                {
                    tileD = 64;
                    warpD = 4;
                    laneD = 4;
                    pipe = 4;
                    if (aligned)
                        f = mambaConv1dContextKernel<4, 32, 64, 1, 4, 4, 4, true>;
                    else
                        f = mambaConv1dContextKernel<4, 32, 64, 1, 4, 4, 4, false>;
                }
                else
                {
                    tileD = 128;
                    warpD = 4;
                    laneD = 8;
                    pipe = 4;
                    if (aligned)
                        f = mambaConv1dContextKernel<4, 32, 128, 1, 4, 8, 4, true>;
                    else
                        f = mambaConv1dContextKernel<4, 32, 128, 1, 4, 8, 4, false>;
                }
            }
            else
            {
                tileD = 64;
                warpD = 4;
                laneD = 2;
                pipe = 4;
                if (aligned)
                    f = mambaConv1dContextKernel<4, 32, 64, 1, 4, 2, 4, true>;
                else
                    f = mambaConv1dContextKernel<4, 32, 64, 1, 4, 2, 4, false>;
            }
        }
        else if (tensorrt_llm::common::getSMVersion() >= 80)
        {
            if (B * L * D <= 262144)
            {
                tileD = 32;
                warpD = 8;
                laneD = 1;
                pipe = 2;
                if (aligned)
                    f = mambaConv1dContextKernel<4, 32, 32, 1, 8, 1, 2, true>;
                else
                    f = mambaConv1dContextKernel<4, 32, 32, 1, 8, 1, 2, false>;
            }
            else if (B * L * D <= 524288 || D % 64 == 32)
            {
                tileD = 32;
                warpD = 4;
                laneD = 1;
                pipe = 2;
                if (aligned)
                    f = mambaConv1dContextKernel<4, 32, 32, 1, 4, 1, 2, true>;
                else
                    f = mambaConv1dContextKernel<4, 32, 32, 1, 4, 1, 2, false>;
            }
            else if (B * L * D <= 1048576)
            {
                tileD = 64;
                warpD = 4;
                laneD = 4;
                pipe = 4;
                if (aligned)
                    f = mambaConv1dContextKernel<4, 32, 64, 1, 4, 4, 4, true>;
                else
                    f = mambaConv1dContextKernel<4, 32, 64, 1, 4, 4, 4, false>;
            }
            else if (B * L * D <= 4194304)
            {
                if (D % 128 == 64)
                {
                    tileD = 64;
                    warpD = 4;
                    laneD = 4;
                    pipe = 4;
                    if (aligned)
                        f = mambaConv1dContextKernel<4, 32, 64, 1, 4, 4, 4, true>;
                    else
                        f = mambaConv1dContextKernel<4, 32, 64, 1, 4, 4, 4, false>;
                }
                else
                {
                    tileD = 128;
                    warpD = 4;
                    laneD = 8;
                    pipe = 4;
                    if (aligned)
                        f = mambaConv1dContextKernel<4, 32, 128, 1, 4, 8, 4, true>;
                    else
                        f = mambaConv1dContextKernel<4, 32, 128, 1, 4, 8, 4, false>;
                }
            }
            else
            {
                if (D % 128 == 64)
                {
                    tileD = 64;
                    warpD = 4;
                    laneD = 2;
                    pipe = 4;
                    if (aligned)
                        f = mambaConv1dContextKernel<4, 32, 64, 1, 4, 2, 4, true>;
                    else
                        f = mambaConv1dContextKernel<4, 32, 64, 1, 4, 2, 4, false>;
                }
                else
                {
                    tileD = 128;
                    warpD = 4;
                    laneD = 4;
                    pipe = 4;
                    if (aligned)
                        f = mambaConv1dContextKernel<4, 32, 128, 1, 4, 4, 4, true>;
                    else
                        f = mambaConv1dContextKernel<4, 32, 128, 1, 4, 4, 4, false>;
                }
            }
        }
        else if (tensorrt_llm::common::getSMVersion() >= 75)
        {
            if (B * L * D <= 262144)
            {
                tileD = 32;
                warpD = 4;
                laneD = 1;
                pipe = 2;
                if (aligned)
                    f = mambaConv1dContextKernel<4, 32, 32, 1, 4, 1, 2, true>;
                else
                    f = mambaConv1dContextKernel<4, 32, 32, 1, 4, 1, 2, false>;
            }
            else if (B * L * D <= 524288 || D % 64 == 32)
            {
                tileD = 32;
                warpD = 4;
                laneD = 1;
                pipe = 2;
                if (aligned)
                    f = mambaConv1dContextKernel<4, 32, 32, 1, 4, 1, 2, true>;
                else
                    f = mambaConv1dContextKernel<4, 32, 32, 1, 4, 1, 2, false>;
            }
            else if (B * L * D <= 1048576)
            {
                if (D % 128 == 64)
                {
                    tileD = 64;
                    warpD = 4;
                    laneD = 4;
                    pipe = 2;
                    if (aligned)
                        f = mambaConv1dContextKernel<4, 32, 64, 1, 4, 4, 2, true>;
                    else
                        f = mambaConv1dContextKernel<4, 32, 64, 1, 4, 4, 2, false>;
                }
                else
                {
                    tileD = 128;
                    warpD = 4;
                    laneD = 8;
                    pipe = 2;
                    if (aligned)
                        f = mambaConv1dContextKernel<4, 32, 128, 1, 4, 8, 2, true>;
                    else
                        f = mambaConv1dContextKernel<4, 32, 128, 1, 4, 8, 2, false>;
                }
            }
            else
            {
                tileD = 32;
                warpD = 4;
                laneD = 1;
                pipe = 2;
                if (aligned)
                    f = mambaConv1dContextKernel<4, 32, 32, 1, 4, 1, 2, true>;
                else
                    f = mambaConv1dContextKernel<4, 32, 32, 1, 4, 1, 2, false>;
            }
        }
        else
        {
            if (B * L * D <= 262144)
            {
                tileD = 32;
                warpD = 8;
                laneD = 1;
                pipe = 2;
                if (aligned)
                    f = mambaConv1dContextKernel<4, 32, 32, 1, 8, 1, 2, true>;
                else
                    f = mambaConv1dContextKernel<4, 32, 32, 1, 8, 1, 2, false>;
            }
            else if (B * L * D <= 524288 || D % 64 == 32)
            {
                tileD = 32;
                warpD = 4;
                laneD = 2;
                pipe = 2;
                if (aligned)
                    f = mambaConv1dContextKernel<4, 32, 32, 1, 4, 2, 2, true>;
                else
                    f = mambaConv1dContextKernel<4, 32, 32, 1, 4, 2, 2, false>;
            }
            else if (B * L * D <= 1048576)
            {
                tileD = 64;
                warpD = 4;
                laneD = 4;
                pipe = 2;
                if (aligned)
                    f = mambaConv1dContextKernel<4, 32, 64, 1, 4, 4, 2, true>;
                else
                    f = mambaConv1dContextKernel<4, 32, 64, 1, 4, 4, 2, false>;
            }
            else
            {
                tileD = 64;
                warpD = 4;
                laneD = 1;
                pipe = 2;
                if (aligned)
                    f = mambaConv1dContextKernel<4, 32, 64, 1, 4, 1, 2, true>;
                else
                    f = mambaConv1dContextKernel<4, 32, 64, 1, 4, 1, 2, false>;
            }
        }
    }
    else
    {
        if (tensorrt_llm::common::getSMVersion() >= 80)
        {
            if (B * L * D <= 262144 || D % 64 == 32)
            {
                tileD = 32;
                warpD = 4;
                laneD = 1;
                pipe = 4;
                if (aligned)
                    f = mambaConv1dContextKernel<4, 32, 32, 1, 4, 1, 4, true>;
                else
                    f = mambaConv1dContextKernel<4, 32, 32, 1, 4, 1, 4, false>;
            }
            else if (B * L * D <= 1048576)
            {
                tileD = 64;
                warpD = 4;
                laneD = 2;
                pipe = 4;
                if (aligned)
                    f = mambaConv1dContextKernel<4, 32, 64, 1, 4, 2, 4, true>;
                else
                    f = mambaConv1dContextKernel<4, 32, 64, 1, 4, 2, 4, false>;
            }
            else
            {
                if (D % 128 == 64)
                {
                    tileD = 64;
                    warpD = 4;
                    laneD = 2;
                    pipe = 4;
                    if (aligned)
                        f = mambaConv1dContextKernel<4, 32, 64, 1, 4, 2, 4, true>;
                    else
                        f = mambaConv1dContextKernel<4, 32, 64, 1, 4, 2, 4, false>;
                }
                else
                {
                    tileD = 128;
                    warpD = 4;
                    laneD = 4;
                    pipe = 4;
                    if (aligned)
                        f = mambaConv1dContextKernel<4, 32, 128, 1, 4, 4, 4, true>;
                    else
                        f = mambaConv1dContextKernel<4, 32, 128, 1, 4, 4, 4, false>;
                }
            }
        }
        else
        {
            if (B * L * D <= 262144 || D % 64 == 32)
            {
                tileD = 32;
                warpD = 4;
                laneD = 1;
                pipe = 2;
                if (aligned)
                    f = mambaConv1dContextKernel<4, 32, 32, 1, 4, 1, 2, true>;
                else
                    f = mambaConv1dContextKernel<4, 32, 32, 1, 4, 1, 2, false>;
            }
            else if (B * L * D <= 1048576)
            {
                tileD = 64;
                warpD = 4;
                laneD = 2;
                pipe = 2;
                if (aligned)
                    f = mambaConv1dContextKernel<4, 32, 64, 1, 4, 2, 2, true>;
                else
                    f = mambaConv1dContextKernel<4, 32, 64, 1, 4, 2, 2, false>;
            }
            else
            {
                if (D % 128 == 64)
                {
                    tileD = 64;
                    warpD = 4;
                    laneD = 2;
                    pipe = 2;
                    if (aligned)
                        f = mambaConv1dContextKernel<4, 32, 64, 1, 4, 2, 2, true>;
                    else
                        f = mambaConv1dContextKernel<4, 32, 64, 1, 4, 2, 2, false>;
                }
                else
                {
                    tileD = 128;
                    warpD = 4;
                    laneD = 4;
                    pipe = 2;
                    if (aligned)
                        f = mambaConv1dContextKernel<4, 32, 128, 1, 4, 4, 2, true>;
                    else
                        f = mambaConv1dContextKernel<4, 32, 128, 1, 4, 4, 2, false>;
                }
            }
        }
    }

    int shmem = warpL * (32 / laneD) * (pipe + 1) * tileD * 4;

    TLLM_CHECK_WITH_INFO(D % 32 == 0, "Channels should be multiple of 32.");
    TLLM_CHECK_WITH_INFO(K == 4, "Only dconv == 4 is supported.");

    input_t* ya = (input_t*) params.out_ptr;
    input_t* ys = (input_t*) params.state_out_ptr;
    input_t const* xa = (input_t const*) params.in_ptr;
    input_t const* xs = nullptr; // (input_t const*) params.state_in_ptr;
    input_t const* w = (input_t const*) params.weight_ptr;
    input_t const* b = (input_t const*) params.bias_ptr;
    bool rmpd = params.remove_padding;
    bool silu = params.apply_silu;
    int const* ltip = params.last_token_ids_ptr;
    int const* ssmp = params.state_slot_mapping_ptr;

    dim3 blks(D / tileD, (L + tileL - 1) / tileL, B);
    dim3 thds(32, warpD, warpL);

    cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);

    f<<<blks, thds, shmem, stream>>>(B, L, D, S_pre, S_post, ya, ys, xa, xs, w, b, rmpd, silu, ltip, ssmp);
}

template <typename input_t, int DCONV = 4, int CHANNELS_PER_THREAD = 4>
__launch_bounds__(64, 8) __global__
    void mamba_conv1d_generation_kernel(MambaConv1dParamsBase params, int micro_batchsize)
{
    input_t* output = reinterpret_cast<input_t*>(params.out_ptr);
    input_t* input = reinterpret_cast<input_t*>(params.in_ptr);
    input_t* state_in = reinterpret_cast<input_t*>(params.state_in_ptr);
    input_t* state_out = reinterpret_cast<input_t*>(params.state_out_ptr);
    input_t* weight = reinterpret_cast<input_t*>(params.weight_ptr);
    input_t* bias = reinterpret_cast<input_t*>(params.bias_ptr);

    int num_channels = params.dim;
    int const micro_batch = blockIdx.y;
    int const channel = (blockIdx.x * blockDim.x + threadIdx.x) * CHANNELS_PER_THREAD;
    int const num_channels_in = num_channels + params.pre_stride + params.post_stride;

    if (channel >= num_channels)
    {
        return;
    }

    weight += channel;
    bias += channel;
    input += (channel + params.pre_stride);
    output += channel;
    state_in += channel;
    state_out += channel;

    float reg_weight[DCONV][CHANNELS_PER_THREAD];
    float reg_bias[CHANNELS_PER_THREAD];
    float reg_result[CHANNELS_PER_THREAD];
    float reg_input[DCONV][CHANNELS_PER_THREAD];

    // load weights
#pragma unroll
    for (int row = 0; row < DCONV; ++row)
    {
        packed_load_to_float<input_t, CHANNELS_PER_THREAD>(weight + row * params.dim, &reg_weight[row][0]);
    }
    // load bias
    packed_load_to_float<input_t, CHANNELS_PER_THREAD>(bias, &reg_bias[0]);

    for (int sample = micro_batch * micro_batchsize; sample < min((micro_batch + 1) * micro_batchsize, params.batch);
         ++sample)
    {
        int const slot_idx = params.state_slot_mapping_ptr == nullptr ? sample : params.state_slot_mapping_ptr[sample];
        input_t* token_input = input + sample * num_channels_in;
        input_t* token_output = output + sample * params.dim;
        input_t* token_state_in = state_in + slot_idx * (params.dconv - 1) * params.dim;
        input_t* token_state_out = state_out + slot_idx * (params.dconv - 1) * params.dim;
#pragma unroll
        for (int i = 0; i < DCONV - 1; ++i)
        {
            packed_load_to_float<input_t, CHANNELS_PER_THREAD>(token_state_in + i * params.dim, &reg_input[i][0]);
        }
        packed_load_to_float<input_t, CHANNELS_PER_THREAD>(token_input, &reg_input[DCONV - 1][0]);

#pragma unroll
        for (int c = 0; c < CHANNELS_PER_THREAD; ++c)
        {
            reg_result[c] = 0.0f;
        }
        // conv
#pragma unroll
        for (int row = 0; row < DCONV; ++row)
        {
#pragma unroll
            for (int c = 0; c < CHANNELS_PER_THREAD; ++c)
            {
                reg_result[c] += reg_weight[row][c] * reg_input[row][c];
            }
        }
        // add bias
#pragma unroll
        for (int c = 0; c < CHANNELS_PER_THREAD; ++c)
        {
            reg_result[c] += reg_bias[c];
        }
        // Silu
        if (params.apply_silu)
        {
#pragma unroll
            for (int c = 0; c < CHANNELS_PER_THREAD; ++c)
            {
                float sigmoid = reg_result[c] < -20.0 ? 0.0f : 1.0f / (1.0f + __expf(-reg_result[c]));
                reg_result[c] *= sigmoid;
            }
        }
        packed_store_float_to<input_t, CHANNELS_PER_THREAD>(&reg_result[0], token_output);

#pragma unroll
        for (int i = 0; i < DCONV - 1; ++i)
        {
            packed_store_float_to<input_t, CHANNELS_PER_THREAD>(&reg_input[i + 1][0], token_state_out + i * params.dim);
        }
    }
}

template <typename input_t>
void invokeMambaConv1dGeneration(MambaConv1dParamsBase& params, cudaStream_t stream)
{
    int samples = params.batch;
    int channels = params.dim;
    int const threadsPerBlock = 64;
    int microBatchSize = 1;
    int const channelsPerThread = 4;
    int const dConv = 4;
    int const channelsPerBlock = threadsPerBlock * channelsPerThread;
    TLLM_CHECK_WITH_INFO(channels % channelsPerThread == 0, "channels should be multiple of channelsPerThread");
    TLLM_CHECK_WITH_INFO(params.dconv == dConv, "only dconv == 4 is supported now.");
    int blockx = (channels + channelsPerBlock - 1) / channelsPerBlock;
    int blocky = (samples + microBatchSize - 1) / microBatchSize;
    dim3 grid(blockx, blocky, 1);
    mamba_conv1d_generation_kernel<input_t, dConv, channelsPerThread>
        <<<grid, threadsPerBlock, 0, stream>>>(params, microBatchSize);
}

template void invokeMambaConv1dContext<float>(MambaConv1dParamsBase& params, cudaStream_t stream);
template void invokeMambaConv1dContext<half>(MambaConv1dParamsBase& params, cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeMambaConv1dContext<__nv_bfloat16>(MambaConv1dParamsBase& params, cudaStream_t stream);
#endif

template void invokeMambaConv1dGeneration<float>(MambaConv1dParamsBase& params, cudaStream_t stream);
template void invokeMambaConv1dGeneration<half>(MambaConv1dParamsBase& params, cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeMambaConv1dGeneration<__nv_bfloat16>(MambaConv1dParamsBase& params, cudaStream_t stream);
#endif

} // namespace kernels
} // namespace tensorrt_llm
