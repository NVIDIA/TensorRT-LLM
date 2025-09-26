/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <cstdint>
#include <cuda_runtime_api.h>

#pragma once

#if CUDART_VERSION >= 11080
// TODO Better way?
#define FMHA_CUDA_SUPPORTS_FP8 true
#endif
#include <cuda_bf16.h>
#if FMHA_CUDA_SUPPORTS_FP8
#include <cuda_fp8.h>
#endif
namespace fmha
{

using fp16_t = uint16_t;
using fp32_t = float;
using tf32_t = uint32_t;
using bf16_t = nv_bfloat16;
#if FMHA_CUDA_SUPPORTS_FP8
using e4m3_t = __nv_fp8_e4m3;
using e5m2_t = __nv_fp8_e5m2;
#else
using e4m3_t = char;
using e5m2_t = char;
#endif

static constexpr float MAX_E4M3 = 448.f;   // 0x7E 2^8  * 1.75
static constexpr float MAX_E5M2 = 57344.f; // 0x7B 2^15 * 1.75

template <typename T>
__host__ __device__ constexpr inline float Softmax_fp_quant_scale();

template <>
__host__ __device__ constexpr inline float Softmax_fp_quant_scale<e4m3_t>()
{
    // Softmax has max output of 1.0, therefore we choose fp32-to-fp8 quantization scale as the
    // largest power-of-2 below the e4m3 limit:
    // 2^(floor(log2(E4M3_MAX / amax_exp_p))) = 2^(floor(log2(448 / 1))) = 2 ^ 8
    return 256.f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha
