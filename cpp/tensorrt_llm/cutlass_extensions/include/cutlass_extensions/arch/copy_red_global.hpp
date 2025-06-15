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
#pragma once

#include <cute/config.hpp>

#include <cute/arch/util.hpp>
#include <cute/atom/copy_traits.hpp>
#include <cute/numeric/numeric_types.hpp>

// Config

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700) && (__CUDACC_VER_MAJOR__ >= 10))
#define CUTE_ARCH_RED_F16_SM70_ENABLED
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDACC_VER_MAJOR__ >= 12))
#define CUTE_ARCH_RED_VEC_SM90_ENABLED
#define CUTE_ARCH_RED_BF16_SM90_ENABLED
#endif

namespace cute
{

//////////////////////////////////
// Wrapper around CUDA's atomicAdd
//////////////////////////////////

template <class T>
struct TypedAtomicAdd
{
    using SRegisters = T[1];
    using DRegisters = T[1];

    CUTE_HOST_DEVICE static constexpr void copy(T const& src, T& dst)
    {
        atomicAdd(&dst, src);
    }
};

template <class T>
struct Copy_Traits<TypedAtomicAdd<T>>
{
    // Logical thread id to thread idx (one-thread)
    using ThrID = Layout<_1>;

    // Map from (src-thr,src-val) to bit
    using SrcLayout = Layout<Shape<_1, Int<sizeof_bits<T>::value>>>;
    // Map from (dst-thr,dst-val) to bit
    using DstLayout = Layout<Shape<_1, Int<sizeof_bits<T>::value>>>;

    // Reference map from (thr,val) to bit
    using RefLayout = SrcLayout;
};

//////////////////////////////////
// F16 ADD PTX
//////////////////////////////////

struct SM70_RED_ADD_NOFTZ_F16
{
    using SRegisters = uint16_t[1];
    using DRegisters = uint16_t[1];

    CUTE_HOST_DEVICE static void copy(uint16_t const& src0, uint16_t& gmem_dst)
    {
#if defined(CUTE_ARCH_RED_F16_SM70_ENABLED)
        asm volatile("red.global.add.noftz.f16 [%0], %1;\n" ::"l"(&gmem_dst), "h"(src0));
#else
        CUTE_INVALID_CONTROL_PATH("Trying to use red.global.f16 without CUTE_ARCH_RED_F16_SM70_ENABLED.");
#endif
    }
};

template <>
struct Copy_Traits<SM70_RED_ADD_NOFTZ_F16>
{
    // Logical thread id to thread idx (one-thread)
    using ThrID = Layout<_1>;

    // Map from (src-thr,src-val) to bit
    using SrcLayout = Layout<Shape<_1, _16>>;

    // Map from (dst-thr,dst-val) to bit
    using DstLayout = Layout<Shape<_1, _16>>;

    // Reference map from (thr,val) to bit
    using RefLayout = SrcLayout;
};

struct SM70_RED_ADD_NOFTZ_F16x2
{
    using SRegisters = uint32_t[1];
    using DRegisters = uint32_t[1];

    CUTE_HOST_DEVICE static void copy(uint32_t const& src0, uint32_t& gmem_dst)
    {
#if defined(CUTE_ARCH_RED_F16_SM70_ENABLED)
        asm volatile("red.global.add.noftz.f16x2 [%0], %1;\n" ::"l"(&gmem_dst), "r"(src0));
#else
        CUTE_INVALID_CONTROL_PATH("Trying to use red.global.f16 without CUTE_ARCH_RED_F16_SM70_ENABLED.");
#endif
    }
};

template <>
struct Copy_Traits<SM70_RED_ADD_NOFTZ_F16x2>
{
    // Logical thread id to thread idx (one-thread)
    using ThrID = Layout<_1>;

    // Map from (src-thr,src-val) to bit
    using SrcLayout = Layout<Shape<_1, _32>>;

    // Map from (dst-thr,dst-val) to bit
    using DstLayout = Layout<Shape<_1, _32>>;

    // Reference map from (thr,val) to bit
    using RefLayout = SrcLayout;
};

struct SM90_RED_ADD_NOFTZ_F16x2_V2
{
    using SRegisters = uint32_t[2];
    using DRegisters = uint64_t[1];

    CUTE_HOST_DEVICE static void copy(uint32_t const& src0, uint32_t const& src1, uint64_t& gmem_dst)
    {
#if defined(CUTE_ARCH_RED_VEC_SM90_ENABLED)
        asm volatile("red.global.add.noftz.v2.f16x2 [%0], {%1, %2};\n" ::"l"(&gmem_dst), "r"(src0), "r"(src1));
#else
        CUTE_INVALID_CONTROL_PATH("Trying to use red.global.vX without CUTE_ARCH_RED_VEC_SM90_ENABLED.");
#endif
    }
};

template <>
struct Copy_Traits<SM90_RED_ADD_NOFTZ_F16x2_V2>
{
    // Logical thread id to thread idx (one-thread)
    using ThrID = Layout<_1>;

    // Map from (src-thr,src-val) to bit
    using SrcLayout = Layout<Shape<_1, _64>>;

    // Map from (dst-thr,dst-val) to bit
    using DstLayout = Layout<Shape<_1, _64>>;

    // Reference map from (thr,val) to bit
    using RefLayout = SrcLayout;
};

struct SM90_RED_ADD_NOFTZ_F16x2_V4
{
    using SRegisters = uint32_t[4];
    using DRegisters = uint128_t[1];

    CUTE_HOST_DEVICE static void copy(
        uint32_t const& src0, uint32_t const& src1, uint32_t const& src2, uint32_t const& src3, uint128_t& gmem_dst)
    {
#if defined(CUTE_ARCH_RED_VEC_SM90_ENABLED)
        asm volatile("red.global.add.noftz.v4.f16x2 [%0], {%1, %2, %3, %4};\n" ::"l"(&gmem_dst), "r"(src0), "r"(src1),
            "r"(src2), "r"(src3));
#else
        CUTE_INVALID_CONTROL_PATH("Trying to use red.global.vX without CUTE_ARCH_RED_VEC_SM90_ENABLED.");
#endif
    }
};

template <>
struct Copy_Traits<SM90_RED_ADD_NOFTZ_F16x2_V4>
{
    // Logical thread id to thread idx (one-thread)
    using ThrID = Layout<_1>;

    // Map from (src-thr,src-val) to bit
    using SrcLayout = Layout<Shape<_1, _128>>;

    // Map from (dst-thr,dst-val) to bit
    using DstLayout = Layout<Shape<_1, _128>>;

    // Reference map from (thr,val) to bit
    using RefLayout = SrcLayout;
};

//////////////////////////////////
// BF16 ADD PTX
//////////////////////////////////

struct SM90_RED_ADD_NOFTZ_BF16
{
    using SRegisters = uint16_t[1];
    using DRegisters = uint16_t[1];

    CUTE_HOST_DEVICE static void copy(uint16_t const& src0, uint16_t& gmem_dst)
    {
#if defined(CUTE_ARCH_RED_BF16_SM90_ENABLED)
        asm volatile("red.global.add.noftz.bf16 [%0], %1;\n" ::"l"(&gmem_dst), "h"(src0));
#else
        CUTE_INVALID_CONTROL_PATH("Trying to use red.global.bf16 without CUTE_ARCH_RED_BF16_SM90_ENABLED.");
#endif
    }
};

template <>
struct Copy_Traits<SM90_RED_ADD_NOFTZ_BF16>
{
    // Logical thread id to thread idx (one-thread)
    using ThrID = Layout<_1>;

    // Map from (src-thr,src-val) to bit
    using SrcLayout = Layout<Shape<_1, _16>>;

    // Map from (dst-thr,dst-val) to bit
    using DstLayout = Layout<Shape<_1, _16>>;

    // Reference map from (thr,val) to bit
    using RefLayout = SrcLayout;
};

//////////////////////////////////

struct SM90_RED_ADD_NOFTZ_BF16x2
{
    using SRegisters = uint32_t[1];
    using DRegisters = uint32_t[1];

    CUTE_HOST_DEVICE static void copy(uint32_t const& src0, uint32_t& gmem_dst)
    {
#if defined(CUTE_ARCH_RED_BF16_SM90_ENABLED)
        asm volatile("red.global.add.noftz.bf16x2 [%0], %1;\n" ::"l"(&gmem_dst), "r"(src0));
#else
        CUTE_INVALID_CONTROL_PATH("Trying to use red.global.bf16 without CUTE_ARCH_RED_BF16_SM90_ENABLED.");
#endif
    }
};

template <>
struct Copy_Traits<SM90_RED_ADD_NOFTZ_BF16x2>
{
    // Logical thread id to thread idx (one-thread)
    using ThrID = Layout<_1>;

    // Map from (src-thr,src-val) to bit
    using SrcLayout = Layout<Shape<_1, _32>>;

    // Map from (dst-thr,dst-val) to bit
    using DstLayout = Layout<Shape<_1, _32>>;

    // Reference map from (thr,val) to bit
    using RefLayout = SrcLayout;
};

//////////////////////////////////

struct SM90_RED_ADD_NOFTZ_BF16x2_V2
{
    using SRegisters = uint32_t[2];
    using DRegisters = uint64_t[1];

    CUTE_HOST_DEVICE static void copy(uint32_t const& src0, uint32_t const& src1, uint64_t& gmem_dst)
    {
#if defined(CUTE_ARCH_RED_BF16_SM90_ENABLED)
        asm volatile("red.global.add.noftz.v2.bf16x2 [%0], {%1, %2};\n" ::"l"(&gmem_dst), "r"(src0), "r"(src1));
#else
        CUTE_INVALID_CONTROL_PATH("Trying to use red.global.bf16 without CUTE_ARCH_RED_BF16_SM90_ENABLED.");
#endif
    }
};

template <>
struct Copy_Traits<SM90_RED_ADD_NOFTZ_BF16x2_V2>
{
    // Logical thread id to thread idx (one-thread)
    using ThrID = Layout<_1>;

    // Map from (src-thr,src-val) to bit
    using SrcLayout = Layout<Shape<_1, _64>>;

    // Map from (dst-thr,dst-val) to bit
    using DstLayout = Layout<Shape<_1, _64>>;

    // Reference map from (thr,val) to bit
    using RefLayout = SrcLayout;
};

//////////////////////////////////

struct SM90_RED_ADD_NOFTZ_BF16x2_V4
{
    using SRegisters = uint32_t[4];
    using DRegisters = uint128_t[1];

    CUTE_HOST_DEVICE static void copy(
        uint32_t const& src0, uint32_t const& src1, uint32_t const& src2, uint32_t const& src3, uint128_t& gmem_dst)
    {
#if defined(CUTE_ARCH_RED_BF16_SM90_ENABLED)
        asm volatile("red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4};\n" ::"l"(&gmem_dst), "r"(src0), "r"(src1),
            "r"(src2), "r"(src3));
#else
        CUTE_INVALID_CONTROL_PATH("Trying to use red.global.bf16 without CUTE_ARCH_RED_BF16_SM90_ENABLED.");
#endif
    }
};

template <>
struct Copy_Traits<SM90_RED_ADD_NOFTZ_BF16x2_V4>
{
    // Logical thread id to thread idx (one-thread)
    using ThrID = Layout<_1>;

    // Map from (src-thr,src-val) to bit
    using SrcLayout = Layout<Shape<_1, _128>>;

    // Map from (dst-thr,dst-val) to bit
    using DstLayout = Layout<Shape<_1, _128>>;

    // Reference map from (thr,val) to bit
    using RefLayout = SrcLayout;
};

//////////////////////////////////

} // end namespace cute
