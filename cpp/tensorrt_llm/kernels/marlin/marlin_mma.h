/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// BF16 m16n8k16 tensor core MMA instructions for Marlin NVFP4 kernels.

#pragma once

#include "marlin_dtypes.cuh"

namespace MARLIN_NAMESPACE_NAME
{

// m16n8k16 tensor core mma: BF16 inputs, FP32 accumulation.
template <typename scalar_t>
__device__ inline void mma(const typename MarlinType<scalar_t>::FragA& a_frag,
    const typename MarlinType<scalar_t>::FragB& frag_b, typename MarlinType<scalar_t>::FragC& frag_c)
{
    uint32_t const* a = reinterpret_cast<uint32_t const*>(&a_frag);
    uint32_t const* b = reinterpret_cast<uint32_t const*>(&frag_b);

    static_assert(std::is_same<scalar_t, nv_bfloat16>::value, "Only BF16 is supported for Marlin NVFP4 MMA");

    float* c = reinterpret_cast<float*>(&frag_c);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// Transposed variant: B is transposed (used for column-major weight loading).
template <typename scalar_t>
__device__ inline void mma_trans(const typename MarlinType<scalar_t>::FragA& a_frag,
    const typename MarlinType<scalar_t>::FragB& frag_b, const typename MarlinType<scalar_t>::FragB& frag_b2,
    typename MarlinType<scalar_t>::FragC& frag_c)
{
    uint32_t const* a = reinterpret_cast<uint32_t const*>(&a_frag);
    uint32_t const* b = reinterpret_cast<uint32_t const*>(&frag_b);
    uint32_t const* b2 = reinterpret_cast<uint32_t const*>(&frag_b2);

    static_assert(std::is_same<scalar_t, nv_bfloat16>::value, "Only BF16 is supported for Marlin NVFP4 MMA");

    float* c = reinterpret_cast<float*>(&frag_c);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(b[0]), "r"(b2[0]), "r"(b[1]), "r"(b2[1]), "r"(a[0]), "r"(a[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]),
        "f"(c[3]));
}

} // namespace MARLIN_NAMESPACE_NAME
