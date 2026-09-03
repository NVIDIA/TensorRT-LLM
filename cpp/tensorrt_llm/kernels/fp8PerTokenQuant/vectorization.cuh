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
// https://github.com/vllm-project/vllm/blob/v0.25.0/csrc/libtorch_stable/quantization/vectorization.cuh

#pragma once
/**
 * __device__ datatypes vectorized by 4
 */

// Include both AMD and NVIDIA fp8 types to avoid circular import
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e4m3fnuz.h>

namespace vllm
{

// Vectorization containers
template <typename scalar_t, size_t vec_size>
struct __align__(vec_size * sizeof(scalar_t)) vec_n_t
{
    scalar_t val[vec_size];
};

template <typename quant_type_t, size_t vec_size>
struct __align__(vec_size * sizeof(quant_type_t)) q8_n_t
{
    static_assert(std::is_same_v<quant_type_t, int8_t> || std::is_same_v<quant_type_t, c10::Float8_e4m3fn>
        || std::is_same_v<quant_type_t, c10::Float8_e4m3fnuz>);
    quant_type_t val[vec_size];
};

template <typename scalar_t>
using vec4_t = vec_n_t<scalar_t, 4>;
template <typename quant_type_t>
using q8x4_t = q8_n_t<quant_type_t, 4>;

} // namespace vllm
