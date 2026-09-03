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

// Adapted from vLLM (Apache-2.0):
//   https://github.com/vllm-project/vllm/blob/v0.25.0/csrc/quantization/utils.cuh
//   https://github.com/vllm-project/vllm/blob/v0.25.0/csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh
//   https://github.com/vllm-project/vllm/blob/v0.25.0/csrc/quantization/w8a8/fp8/common.cuh
//
// Slim compatibility header: provides the subset of FP8 quantization
// utilities needed by dynamic_per_token_scaled_fp8_quant_kernel_strided.
// Excludes the attention-dtype conversions from quant_utils.cuh that would
// pull in vLLM's attention infrastructure.

#pragma once

#include <c10/util/Float8_e4m3fn.h>
#include <cuda_fp8.h>
#include <limits>
#include <type_traits>

// ---------------------------------------------------------------------------
// From csrc/quantization/utils.cuh: quant_type_max_v, min_scaling_factor
// ---------------------------------------------------------------------------

template <typename T, typename = std::enable_if_t<std::is_same_v<T, c10::Float8_e4m3fn> || std::is_same_v<T, int8_t>>>
struct quant_type_max
{
    __host__ __device__ static constexpr T val()
    {
        return std::numeric_limits<T>::max();
    }
};

template <typename T>
__host__ __device__ static constexpr T quant_type_max_v = quant_type_max<T>::val();

template <typename T, typename = std::enable_if_t<std::is_same_v<T, c10::Float8_e4m3fn> || std::is_same_v<T, int8_t>>>
struct min_scaling_factor
{
    __device__ __forceinline__ static float val()
    {
        return 1.0f / (static_cast<float>(quant_type_max_v<T>) * 512.0f);
    }
};

// ---------------------------------------------------------------------------
// From csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh:
// fp8::vec_conversion<c10::Float8_e4m3fn, float> only.
// The full quant_utils.cuh includes attention_dtypes.h (vLLM-specific types);
// we provide only the conversion needed by the kernel.
// ---------------------------------------------------------------------------

namespace vllm
{
namespace fp8
{

template <typename Tout, typename Tin>
__inline__ __device__ Tout vec_conversion(Tin const& x, const __nv_fp8_interpretation_t fp8_type = __NV_E4M3)
{
    return x;
}

// float -> c10::Float8_e4m3fn
template <>
__inline__ __device__ c10::Float8_e4m3fn vec_conversion<c10::Float8_e4m3fn, float>(
    float const& a, const __nv_fp8_interpretation_t fp8_type)
{
    return c10::Float8_e4m3fn(__nv_cvt_float_to_fp8(a, __NV_SATFINITE, fp8_type), c10::Float8_e4m3fn::from_bits());
}

} // namespace fp8

// ---------------------------------------------------------------------------
// From csrc/quantization/w8a8/fp8/common.cuh: scaled_fp8_conversion
// ---------------------------------------------------------------------------

template <bool is_scale_inverted, typename fp8_type>
__device__ __forceinline__ fp8_type scaled_fp8_conversion(float const val, float const scale)
{
    float x = 0.0f;
    if constexpr (is_scale_inverted)
    {
        x = val * scale;
    }
    else
    {
        x = val / scale;
    }
    float r = fmaxf(
        -static_cast<float>(quant_type_max_v<fp8_type>), fminf(x, static_cast<float>(quant_type_max_v<fp8_type>)));
    return fp8::vec_conversion<fp8_type, float>(r);
}

} // namespace vllm
