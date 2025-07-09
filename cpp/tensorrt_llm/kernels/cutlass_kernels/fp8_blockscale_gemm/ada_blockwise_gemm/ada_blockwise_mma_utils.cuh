/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
#include "cute/atom/mma_atom.hpp"
#include <cute/arch/mma.hpp>
#include <cute/config.hpp>
#include <cute/layout.hpp>
#include <cutlass/arch/mma.h>

// Config
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890))
#define CUTE_ARCH_MMA_F32_SM89_ENABLED
#endif

namespace cute
{

// MMA 16x8x32 TN
struct SM89_16x8x32_F32F8F8F32_TN
{
    using DRegisters = float[4];
    using ARegisters = uint32_t[4];
    using BRegisters = uint32_t[2];
    using CRegisters = float[4];

    CUTE_HOST_DEVICE static void fma(float& d0, float& d1, float& d2, float& d3, uint32_t const& a0, uint32_t const& a1,
        uint32_t const& a2, uint32_t const& a3, uint32_t const& b0, uint32_t const& b1, float const& c0,
        float const& c1, float const& c2, float const& c3)
    {
#if defined(CUTE_ARCH_MMA_F32_SM89_ENABLED)
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0,  %1,  %2,  %3},"
            "{%4,  %5,  %6,  %7},"
            "{%8,  %9},"
            "{%10, %11, %12, %13};\n"
            : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#else
        CUTE_INVALID_CONTROL_PATH(
            "Attempting to use SM89_16x8x32_F32F8F8F32_TN without "
            "CUTE_ARCH_MMA_F32_SM89_ENABLED");
#endif
    }
};

template <>
struct MMA_Traits<SM89_16x8x32_F32F8F8F32_TN>
{
    using ValTypeD = float;
    using ValTypeA = float_e4m3_t;
    using ValTypeB = float_e4m3_t;
    using ValTypeC = float;

    using Shape_MNK = Shape<_16, _8, _32>;
    using ThrID = Layout<_32>;
    using ALayout = Layout<Shape<Shape<_4, _8>, Shape<_4, _2, _2>>, Stride<Stride<_64, _1>, Stride<_16, _8, _256>>>;
    using BLayout = Layout<Shape<Shape<_4, _8>, Shape<_4, _2>>, Stride<Stride<_32, _1>, Stride<_8, _128>>>;
    using CLayout = SM80_16x8_Row;
};

} // namespace cute

namespace ada_blockwise_gemm
{

template <typename Element, typename Arch>
struct DefaultGemm_TensorOp_MMA;

template <>
struct DefaultGemm_TensorOp_MMA<cute::bfloat16_t, cutlass::arch::Sm80>
{
    using ArchTag = cutlass::arch::Sm80;
    using MMA_Atom_Arch = cute::MMA_Atom<cute::SM80_16x8x16_F32BF16BF16F32_TN>;
    using ThreadLayoutMNK = cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>;
    using ValLayoutMNK = cute::Tile<cute::_32, cute::_32, cute::_16>;
    using TiledMma = cute::TiledMMA<MMA_Atom_Arch, ThreadLayoutMNK, ValLayoutMNK>;
};

template <>
struct DefaultGemm_TensorOp_MMA<cute::float_e4m3_t, cutlass::arch::Sm89>
{
    using ArchTag = cutlass::arch::Sm89;
    using MMA_Atom_Arch = cute::MMA_Atom<cute::SM89_16x8x32_F32F8F8F32_TN>;
    using ThreadLayoutMNK = cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>;
    using ValLayoutMNK = cute::Tile<cute::_32, cute::_32, cute::_32>;
    using TiledMma = cute::TiledMMA<MMA_Atom_Arch, ThreadLayoutMNK, ValLayoutMNK>;
};

} // namespace ada_blockwise_gemm
