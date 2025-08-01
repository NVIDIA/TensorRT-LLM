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

#pragma once
#include "cuda_hint.cuh"
#include "mha_stdheaders.cuh"
#ifndef __CUDACC__
#include <cuda_runtime.h>
#endif
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// for both a and b, outer-dim is gemm-K and inner-dim is gemm-M or gemm-N
// acc is used as both input and output.
template <typename InputElem>
__device__ inline void mma(float (&acc)[2][2], uint32_t const (&a)[2][2], uint32_t const (&b)[2][1])
{

    static_assert(mha::is_same_v<InputElem, half> || mha::is_same_v<InputElem, __nv_bfloat16>
            || mha::is_same_v<InputElem, __nv_fp8_e4m3>,
        "not implemented");
    if constexpr (mha::is_same_v<InputElem, half>)
    {
        asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \n"
            "    {%0, %1, %2, %3}, \n"
            "    {%4, %5, %6, %7}, \n"
            "    {%8, %9}, \n"
            "    {%0, %1, %2, %3}; \n"
            : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[1][0]), "+f"(acc[1][1])
            : "r"(a[0][0]), "r"(a[0][1]), "r"(a[1][0]), "r"(a[1][1]), "r"(b[0][0]), "r"(b[1][0]));
    }
    else if constexpr (mha::is_same_v<InputElem, __nv_bfloat16>)
    {
        asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 \n"
            "    {%0, %1, %2, %3}, \n"
            "    {%4, %5, %6, %7}, \n"
            "    {%8, %9}, \n"
            "    {%0, %1, %2, %3}; \n"
            : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[1][0]), "+f"(acc[1][1])
            : "r"(a[0][0]), "r"(a[0][1]), "r"(a[1][0]), "r"(a[1][1]), "r"(b[0][0]), "r"(b[1][0]));
    }
    else if constexpr (mha::is_same_v<InputElem, __nv_fp8_e4m3>)
    {
        asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 \n"
            "    {%0, %1, %2, %3}, \n"
            "    {%4, %5, %6, %7}, \n"
            "    {%8, %9}, \n"
            "    {%0, %1, %2, %3}; \n"
            : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[1][0]), "+f"(acc[1][1])
            : "r"(a[0][0]), "r"(a[0][1]), "r"(a[1][0]), "r"(a[1][1]), "r"(b[0][0]), "r"(b[1][0]));
    }
    else
    {
        asm volatile("trap;");
    }
}

__device__ inline void mmaF8_k16(float (&acc)[2][2], uint32_t const (&a)[2], uint32_t const b)
{
    asm("mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32 \n"
        "    {%0, %1, %2, %3}, \n"
        "    {%4, %5}, \n"
        "    {%6}, \n"
        "    {%0, %1, %2, %3}; \n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[1][0]), "+f"(acc[1][1])
        : "r"(a[0]), "r"(a[1]), "r"(b));
}

__device__ inline void mmaF8_k32_2inst(float (&acc)[2][2], uint32_t const (&a)[2][2], uint32_t const (&b)[2][1])
{
    for (uint32_t i = 0; i < 2; i++)
    {
        mmaF8_k16(acc, a[i], b[i][0]);
    }
}

struct mmaShape
{
    uint32_t m;
    uint32_t n;
    uint32_t k;
};

inline constexpr mmaShape qmmaShape = {16, 8, 32};
