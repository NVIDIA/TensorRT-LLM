/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/cudaBf16Fallbacks.cuh"
#include "tensorrt_llm/common/cudaBufferUtils.cuh"
#include "tensorrt_llm/common/cudaFp8Utils.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/quantization.cuh"
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <type_traits>

using namespace tensorrt_llm::common;

namespace tensorrt_llm::kernels
{

template <typename T, bool UE8M0_SF = false, typename = void>
struct FP4Converter;

template <typename TIn, bool UE8M0_SF>
struct FP4Converter<TIn, UE8M0_SF, std::enable_if_t<std::is_same_v<TIn, half> || std::is_same_v<TIn, __nv_bfloat16>>>
{

    // // Define a 16 bytes packed data type.
    // struct PackedVec
    // {
    //     half2 elts[4];
    // };

    static constexpr int ELTS_PER_THREAD = 8;
    static constexpr int SF_VEC_SIZE = 16;
    static constexpr int THREADS_PER_WARP = 32;

    // The global scaling factor, which will be applied to the SF.
    // Note SFScale is the same as next GEMM's alpha, which is (448.f / (Alpha_A / 6.f)).
    float const SFScaleVal;
    int const numCols;
    uint32_t* const SFout;
    uint32_t* const out;

    template <typename Param>
    __device__ __forceinline__
    FP4Converter<TIn, UE8M0_SF, std::enable_if_t<std::is_same_v<TIn, half> || std::is_same_v<TIn, __nv_bfloat16>>>(
        Param const& p)
        : SFScaleVal(p.sf_scale == nullptr ? 1.0f : p.sf_scale[0])
        , numCols(p.n)
        , SFout(p.sf_out)
        , out(p.normed_output)
    {
    }

    template <size_t ELTS_PER_THREAD, typename T>
    __device__ __forceinline__ void post_process(int rowIdx, int n_base, T packed_input) const
    {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000

        static_assert(sizeof(T) == sizeof(TIn) * ELTS_PER_THREAD, "Vec size is not matched.");
        static constexpr int NUM_THREADS_PER_SF = SF_VEC_SIZE / ELTS_PER_THREAD;
        static_assert(NUM_THREADS_PER_SF == 2);

        int colIdx = n_base / ELTS_PER_THREAD;

        // Get the input tensor offset.
        // int inOffset = rowIdx * (numCols / ELTS_PER_THREAD) + colIdx;
        // PackedVec vec = reinterpret_cast<PackedVec const*>(in)[inOffset];

        // Get absolute maximum values among the local 8 values.
        auto localMax = __habs2({packed_input.array[0], packed_input.array[1]});

// Local maximum value.
#pragma unroll
        for (int i = 2; i < ELTS_PER_THREAD; i += 2)
        {
            localMax = __hmax2(localMax, __habs2({packed_input.array[i], packed_input.array[i + 1]}));
        }

        // Get the absolute maximum among all 16 values (two threads).
        localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
        // Get the final absolute maximum values.
        float vecMax = float(__hmax(localMax.x, localMax.y));

        // Get the SF (max value of the vector / max value of e2m1).
        // maximum value of e2m1 = 6.0.
        // TODO: use half as compute data type.
        float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
        // 8 bits representation of the SF.
        uint8_t fp8SFVal;
        // Write the SF to global memory (STG.8).
        if constexpr (UE8M0_SF)
        {
            __nv_fp8_e8m0 tmp;
            tmp.__x = __nv_cvt_float_to_e8m0(SFValue, __NV_SATFINITE, cudaRoundPosInf);
            SFValue = static_cast<float>(tmp);
            fp8SFVal = tmp.__x;
        }
        else
        {
            // Here SFValue is always positive, so E4M3 is the same as UE4M3.
            __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
            fp8SFVal = tmp.__x;
            SFValue = static_cast<float>(tmp);
        }

        auto SFOffset = cvt_quant_get_sf_out_offset<uint32_t, NUM_THREADS_PER_SF>(std::nullopt /* batchIdx */, rowIdx,
            colIdx, std::nullopt /* numRows */, numCols / SF_VEC_SIZE, SFout, QuantizationSFLayout::SWIZZLED);
        *SFOffset = fp8SFVal;
        // Get the output scale.
        // Recipe: final_scale = reciprocal(fp32(fp8(SFValue * SFScaleVal))) * reciprocal(SFScaleVal))
        float outputScale = reciprocal_approximate_ftz(SFValue * reciprocal_approximate_ftz(SFScaleVal));

        // Convert the input to float.
        float2 fp2Vals[ELTS_PER_THREAD / 2];

#pragma unroll
        for (int i = 0; i < ELTS_PER_THREAD; i += 2)
        {
            if constexpr (std::is_same<TIn, half>::value)
            {
                fp2Vals[i / 2] = __half22float2({packed_input.array[i], packed_input.array[i + 1]});
            }
            else
            {
                fp2Vals[i / 2] = __bfloat1622float2({packed_input.array[i], packed_input.array[i + 1]});
            }
            fp2Vals[i / 2].x *= outputScale;
            fp2Vals[i / 2].y *= outputScale;
        }

        // Convert to e2m1 values.
        uint32_t e2m1Vec = fp32_vec_to_e2m1(fp2Vals);

        // Get the output tensor offset.
        // Same as inOffset because 8 elements are packed into one uint32_t.
        int64_t outOffset = rowIdx * (numCols / ELTS_PER_THREAD) + colIdx;
        // Write the e2m1 values to global memory.
        out[outOffset] = e2m1Vec;
#else
        printf("FP4 is not supported pre-Blackwell!\n");
#endif
    }
};

template <bool UE8M0_SF>
struct FP4Converter<float, UE8M0_SF>
{

    static constexpr int ELTS_PER_THREAD = 8;
    static constexpr int SF_VEC_SIZE = 16;
    static constexpr int THREADS_PER_WARP = 32;

    // The global scaling factor, which will be applied to the SF.
    // Note SFScale is the same as next GEMM's alpha, which is (448.f / (Alpha_A / 6.f)).
    float const SFScaleVal;
    int const numCols;
    uint32_t* const SFout;
    uint32_t* const out;

    template <typename Param>
    __device__ __forceinline__ FP4Converter<float, UE8M0_SF>(Param const& p)
        : SFScaleVal(p.sf_scale == nullptr ? 1.0f : p.sf_scale[0])
        , numCols(p.n)
        , SFout(p.sf_out)
        , out(p.normed_output)
    {
    }

    template <size_t ELTS_PER_THREAD, typename T>
    __device__ __forceinline__ void post_process(int rowIdx, int n_base, T packed_input) const
    {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000

        static_assert(sizeof(T) == sizeof(float) * ELTS_PER_THREAD, "Vec size is not matched.");
        static constexpr int NUM_THREADS_PER_SF = SF_VEC_SIZE / ELTS_PER_THREAD;
        static_assert(NUM_THREADS_PER_SF == 2);

        int colIdx = n_base / ELTS_PER_THREAD;

        // Get absolute maximum values among the local 8 values.

        float localMax = packed_input.array[0];

// Local maximum value.
#pragma unroll
        for (int i = 1; i < ELTS_PER_THREAD; i++)
        {
            localMax = max(localMax, packed_input.array[i]);
        }

        // Get the absolute maximum among all 16 values (two threads).
        localMax = max(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
        // Get the final absolute maximum values.
        float vecMax = localMax;

        // Get the SF (max value of the vector / max value of e2m1).
        // maximum value of e2m1 = 6.0.
        // TODO: use half as compute data type.
        float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
        // 8 bits representation of the SF.
        uint8_t fp8SFVal;
        // Write the SF to global memory (STG.8).
        if constexpr (UE8M0_SF)
        {
            __nv_fp8_e8m0 tmp;
            tmp.__x = __nv_cvt_float_to_e8m0(SFValue, __NV_SATFINITE, cudaRoundPosInf);
            SFValue = static_cast<float>(tmp);
            fp8SFVal = tmp.__x;
        }
        else
        {
            // Here SFValue is always positive, so E4M3 is the same as UE4M3.
            __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
            fp8SFVal = tmp.__x;
            SFValue = static_cast<float>(tmp);
        }

        auto SFOffset = cvt_quant_get_sf_out_offset<uint32_t, NUM_THREADS_PER_SF>(std::nullopt /* batchIdx */, rowIdx,
            colIdx, std::nullopt /* numRows */, numCols / SF_VEC_SIZE, SFout, QuantizationSFLayout::SWIZZLED);
        float outputScale = reciprocal_approximate_ftz(SFValue * reciprocal_approximate_ftz(SFScaleVal));

        // Convert the input to float.
        float2 fp2Vals[ELTS_PER_THREAD / 2];

#pragma unroll
        for (int i = 0; i < ELTS_PER_THREAD; i += 2)
        {
            fp2Vals[i / 2] = float2{packed_input.array[i], packed_input.array[i + 1]};
            fp2Vals[i / 2] = __fmul2_rn(fp2Vals[i / 2], {outputScale, outputScale});
        }

        // Convert to e2m1 values.
        uint32_t e2m1Vec = fp32_vec_to_e2m1(fp2Vals);

        // Get the output tensor offset.
        // Same as inOffset because 8 elements are packed into one uint32_t.
        int64_t outOffset = rowIdx * (numCols / ELTS_PER_THREAD) + colIdx;
        // Write the e2m1 values to global memory.
        out[outOffset] = e2m1Vec;
#else
        printf("FP4 is not supported pre-Blackwell!\n");
#endif
    }
};

} // namespace tensorrt_llm::kernels
