/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <array>
#include <cstdint>

extern "C" __device__ unsigned __nvvm_get_smem_pointer(void* ptr);

template <int mode_, int line_, class T_>
__device__ inline int swizzle(int x_)
{
    return x_ ^ x_ / line_ % (mode_ / 16) * (16 / sizeof(T_));
}

template <class T_>
__device__ inline int swizzle(int x_, int y_)
{
    return x_ ^ y_ * (16 / sizeof(T_));
}

template <int size_>
__device__ inline void cp_shared_global(unsigned s_ptr, void const* g_ptr)
{
    static_assert(size_ == 4 || size_ == 8 || size_ == 16);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if constexpr (size_ == 16)
        asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(s_ptr), "l"(g_ptr), "n"(size_));
    else if constexpr (size_ == 8)
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(s_ptr), "l"(g_ptr), "n"(size_));
    else if constexpr (size_ == 4)
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(s_ptr), "l"(g_ptr), "n"(size_));
#elif defined(__CUDA_ARCH__)
    unsigned tmp[size_ / 4];

    if constexpr (size_ == 16)
    {
        asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(tmp[0]), "=r"(tmp[1]), "=r"(tmp[2]), "=r"(tmp[3])
                     : "l"(g_ptr));
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n" ::"r"(s_ptr), "r"(tmp[0]), "r"(tmp[1]), "r"(tmp[2]),
            "r"(tmp[3]));
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

template <int size_>
__device__ inline void cp_shared_global(unsigned s_ptr, void const* g_ptr, bool valid_)
{
    static_assert(size_ == 4 || size_ == 8 || size_ == 16);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if constexpr (size_ == 16)
        asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"r"(s_ptr), "l"(g_ptr), "n"(size_),
            "r"(valid_ ? size_ : 0));
    else if constexpr (size_ == 8)
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(s_ptr), "l"(g_ptr), "n"(size_),
            "r"(valid_ ? size_ : 0));
    else if constexpr (size_ == 4)
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(s_ptr), "l"(g_ptr), "n"(size_),
            "r"(valid_ ? size_ : 0));
#elif defined(__CUDA_ARCH__)
    unsigned tmp[size_ / 4];

    if constexpr (size_ == 16)
    {
        if (valid_)
        {
            asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(tmp[0]), "=r"(tmp[1]), "=r"(tmp[2]), "=r"(tmp[3])
                         : "l"(g_ptr));
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n" ::"r"(s_ptr), "r"(tmp[0]), "r"(tmp[1]),
                "r"(tmp[2]), "r"(tmp[3]));
        }
        else
        {
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n" ::"r"(s_ptr), "n"(0), "n"(0), "n"(0), "n"(0));
        }
    }
    else if constexpr (size_ == 8)
    {
        if (valid_)
        {
            asm volatile("ld.global.v2.b32 {%0, %1}, [%2];\n" : "=r"(tmp[0]), "=r"(tmp[1]) : "l"(g_ptr));
            asm volatile("st.shared.v2.b32 [%0], {%1, %2};\n" ::"r"(s_ptr), "r"(tmp[0]), "r"(tmp[1]));
        }
        else
        {
            asm volatile("st.shared.v2.b32 [%0], {%1, %2};\n" ::"r"(s_ptr), "n"(0), "n"(0));
        }
    }
    else if constexpr (size_ == 4)
    {
        if (valid_)
        {
            asm volatile("ld.global.b32 %0, [%1];\n" : "=r"(tmp[0]) : "l"(g_ptr));
            asm volatile("st.shared.b32 [%0], %1;\n" ::"r"(s_ptr), "r"(tmp[0]));
        }
        else
        {
            asm volatile("st.shared.b32 [%0], %1;\n" ::"r"(s_ptr), "n"(0));
        }
    }
#endif
}

__device__ inline void cp_commit_group()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n");
#endif
}

template <int remain_>
__device__ inline void cp_wait_group()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" ::"n"(remain_));
#endif
}

template <bool trans_ = false>
__device__ inline void ldmatrix_x4(unsigned& r0_, unsigned& r1_, unsigned& r2_, unsigned& r3_, unsigned addr_)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    if (trans_)
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(r0_), "=r"(r1_), "=r"(r2_), "=r"(r3_)
                     : "r"(addr_));
    else
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(r0_), "=r"(r1_), "=r"(r2_), "=r"(r3_)
                     : "r"(addr_));
#endif
}

template <class Tp_>
__device__ inline void mma(std::array<float, 4>& acc_, std::array<unsigned, 4> a_, std::array<unsigned, 2> b_)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (std::is_same_v<Tp_, half>)
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \n"
            "{%0, %1, %2, %3},\n"
            "{%4, %5, %6, %7},\n"
            "{%8, %9},\n"
            "{%0, %1, %2, %3};\n"
            : "+f"(acc_[0]), "+f"(acc_[1]), "+f"(acc_[2]), "+f"(acc_[3])
            : "r"(a_[0]), "r"(a_[1]), "r"(a_[2]), "r"(a_[3]), "r"(b_[0]), "r"(b_[1]));
    else
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 \n"
            "{%0, %1, %2, %3},\n"
            "{%4, %5, %6, %7},\n"
            "{%8, %9},\n"
            "{%0, %1, %2, %3};\n"
            : "+f"(acc_[0]), "+f"(acc_[1]), "+f"(acc_[2]), "+f"(acc_[3])
            : "r"(a_[0]), "r"(a_[1]), "r"(a_[2]), "r"(a_[3]), "r"(b_[0]), "r"(b_[1]));
#endif
}

template <class Tp_, bool aTrans_ = false, bool bTrans_ = false>
__device__ inline void wgmma(std::array<std::array<float, 4>, 2>& acc_, uint64_t aDesc_, uint64_t bDesc_)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
    if (std::is_same_v<Tp_, half>)
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 \n"
            "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},\n"
            " %8, %9, 1,1,1, %10,%11;\n"
            : "+f"(acc_[0][0]), "+f"(acc_[0][1]), "+f"(acc_[0][2]), "+f"(acc_[0][3]), "+f"(acc_[1][0]),
            "+f"(acc_[1][1]), "+f"(acc_[1][2]), "+f"(acc_[1][3])
            : "l"(aDesc_), "l"(bDesc_), "n"(aTrans_ ? 1 : 0), "n"(bTrans_ ? 1 : 0));
    else
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 \n"
            "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},\n"
            " %8, %9, 1,1,1, %10,%11;\n"
            : "+f"(acc_[0][0]), "+f"(acc_[0][1]), "+f"(acc_[0][2]), "+f"(acc_[0][3]), "+f"(acc_[1][0]),
            "+f"(acc_[1][1]), "+f"(acc_[1][2]), "+f"(acc_[1][3])
            : "l"(aDesc_), "l"(bDesc_), "n"(aTrans_ ? 1 : 0), "n"(bTrans_ ? 1 : 0));
#endif
}

template <class Tp_, bool aTrans_ = false, bool bTrans_ = false>
__device__ inline void wgmma(std::array<std::array<float, 4>, 4>& acc_, uint64_t aDesc_, uint64_t bDesc_)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
    if (std::is_same_v<Tp_, half>)
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 \n"
            "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, \n"
            "  %8,  %9, %10, %11, %12, %13, %14, %15},\n"
            " %16, %17, 1,1,1, %18,%19;\n"
            : "+f"(acc_[0][0]), "+f"(acc_[0][1]), "+f"(acc_[0][2]), "+f"(acc_[0][3]), "+f"(acc_[1][0]),
            "+f"(acc_[1][1]), "+f"(acc_[1][2]), "+f"(acc_[1][3]), "+f"(acc_[2][0]), "+f"(acc_[2][1]), "+f"(acc_[2][2]),
            "+f"(acc_[2][3]), "+f"(acc_[3][0]), "+f"(acc_[3][1]), "+f"(acc_[3][2]), "+f"(acc_[3][3])
            : "l"(aDesc_), "l"(bDesc_), "n"(aTrans_ ? 1 : 0), "n"(bTrans_ ? 1 : 0));
    else
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 \n"
            "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, \n"
            "  %8,  %9, %10, %11, %12, %13, %14, %15},\n"
            " %16, %17, 1,1,1, %18,%19;\n"
            : "+f"(acc_[0][0]), "+f"(acc_[0][1]), "+f"(acc_[0][2]), "+f"(acc_[0][3]), "+f"(acc_[1][0]),
            "+f"(acc_[1][1]), "+f"(acc_[1][2]), "+f"(acc_[1][3]), "+f"(acc_[2][0]), "+f"(acc_[2][1]), "+f"(acc_[2][2]),
            "+f"(acc_[2][3]), "+f"(acc_[3][0]), "+f"(acc_[3][1]), "+f"(acc_[3][2]), "+f"(acc_[3][3])
            : "l"(aDesc_), "l"(bDesc_), "n"(aTrans_ ? 1 : 0), "n"(bTrans_ ? 1 : 0));
#endif
}

template <class Tp_, bool aTrans_ = false, bool bTrans_ = false>
__device__ inline void wgmma(std::array<std::array<float, 4>, 8>& acc_, uint64_t aDesc_, uint64_t bDesc_)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
    if (std::is_same_v<Tp_, half>)
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 \n"
            "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, \n"
            "  %8,  %9, %10, %11, %12, %13, %14, %15, \n"
            " %16, %17, %18, %19, %20, %21, %22, %23, \n"
            " %24, %25, %26, %27, %28, %29, %30, %31},\n"
            " %32, %33, 1,1,1, %34,%35;\n"
            : "+f"(acc_[0][0]), "+f"(acc_[0][1]), "+f"(acc_[0][2]), "+f"(acc_[0][3]), "+f"(acc_[1][0]),
            "+f"(acc_[1][1]), "+f"(acc_[1][2]), "+f"(acc_[1][3]), "+f"(acc_[2][0]), "+f"(acc_[2][1]), "+f"(acc_[2][2]),
            "+f"(acc_[2][3]), "+f"(acc_[3][0]), "+f"(acc_[3][1]), "+f"(acc_[3][2]), "+f"(acc_[3][3]), "+f"(acc_[4][0]),
            "+f"(acc_[4][1]), "+f"(acc_[4][2]), "+f"(acc_[4][3]), "+f"(acc_[5][0]), "+f"(acc_[5][1]), "+f"(acc_[5][2]),
            "+f"(acc_[5][3]), "+f"(acc_[6][0]), "+f"(acc_[6][1]), "+f"(acc_[6][2]), "+f"(acc_[6][3]), "+f"(acc_[7][0]),
            "+f"(acc_[7][1]), "+f"(acc_[7][2]), "+f"(acc_[7][3])
            : "l"(aDesc_), "l"(bDesc_), "n"(aTrans_ ? 1 : 0), "n"(bTrans_ ? 1 : 0));
    else
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 \n"
            "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, \n"
            "  %8,  %9, %10, %11, %12, %13, %14, %15, \n"
            " %16, %17, %18, %19, %20, %21, %22, %23, \n"
            " %24, %25, %26, %27, %28, %29, %30, %31},\n"
            " %32, %33, 1,1,1, %34,%35;\n"
            : "+f"(acc_[0][0]), "+f"(acc_[0][1]), "+f"(acc_[0][2]), "+f"(acc_[0][3]), "+f"(acc_[1][0]),
            "+f"(acc_[1][1]), "+f"(acc_[1][2]), "+f"(acc_[1][3]), "+f"(acc_[2][0]), "+f"(acc_[2][1]), "+f"(acc_[2][2]),
            "+f"(acc_[2][3]), "+f"(acc_[3][0]), "+f"(acc_[3][1]), "+f"(acc_[3][2]), "+f"(acc_[3][3]), "+f"(acc_[4][0]),
            "+f"(acc_[4][1]), "+f"(acc_[4][2]), "+f"(acc_[4][3]), "+f"(acc_[5][0]), "+f"(acc_[5][1]), "+f"(acc_[5][2]),
            "+f"(acc_[5][3]), "+f"(acc_[6][0]), "+f"(acc_[6][1]), "+f"(acc_[6][2]), "+f"(acc_[6][3]), "+f"(acc_[7][0]),
            "+f"(acc_[7][1]), "+f"(acc_[7][2]), "+f"(acc_[7][3])
            : "l"(aDesc_), "l"(bDesc_), "n"(aTrans_ ? 1 : 0), "n"(bTrans_ ? 1 : 0));
#endif
}

template <class Tp_, bool aTrans_ = false, bool bTrans_ = false>
__device__ inline void wgmma(std::array<std::array<float, 4>, 16>& acc_, uint64_t aDesc_, uint64_t bDesc_)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
    if (std::is_same_v<Tp_, half>)
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 \n"
            "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, \n"
            "  %8,  %9, %10, %11, %12, %13, %14, %15, \n"
            " %16, %17, %18, %19, %20, %21, %22, %23, \n"
            " %24, %25, %26, %27, %28, %29, %30, %31, \n"
            " %32, %33, %34, %35, %36, %37, %38, %39, \n"
            " %40, %41, %42, %43, %44, %45, %46, %47, \n"
            " %48, %49, %50, %51, %52, %53, %54, %55, \n"
            " %56, %57, %58, %59, %60, %61, %62, %63},\n"
            " %64, %65, 1,1,1, %66,%67;\n"
            : "+f"(acc_[0][0]), "+f"(acc_[0][1]), "+f"(acc_[0][2]), "+f"(acc_[0][3]), "+f"(acc_[1][0]),
            "+f"(acc_[1][1]), "+f"(acc_[1][2]), "+f"(acc_[1][3]), "+f"(acc_[2][0]), "+f"(acc_[2][1]), "+f"(acc_[2][2]),
            "+f"(acc_[2][3]), "+f"(acc_[3][0]), "+f"(acc_[3][1]), "+f"(acc_[3][2]), "+f"(acc_[3][3]), "+f"(acc_[4][0]),
            "+f"(acc_[4][1]), "+f"(acc_[4][2]), "+f"(acc_[4][3]), "+f"(acc_[5][0]), "+f"(acc_[5][1]), "+f"(acc_[5][2]),
            "+f"(acc_[5][3]), "+f"(acc_[6][0]), "+f"(acc_[6][1]), "+f"(acc_[6][2]), "+f"(acc_[6][3]), "+f"(acc_[7][0]),
            "+f"(acc_[7][1]), "+f"(acc_[7][2]), "+f"(acc_[7][3]), "+f"(acc_[8][0]), "+f"(acc_[8][1]), "+f"(acc_[8][2]),
            "+f"(acc_[8][3]), "+f"(acc_[9][0]), "+f"(acc_[9][1]), "+f"(acc_[9][2]), "+f"(acc_[9][3]), "+f"(acc_[10][0]),
            "+f"(acc_[10][1]), "+f"(acc_[10][2]), "+f"(acc_[10][3]), "+f"(acc_[11][0]), "+f"(acc_[11][1]),
            "+f"(acc_[11][2]), "+f"(acc_[11][3]), "+f"(acc_[12][0]), "+f"(acc_[12][1]), "+f"(acc_[12][2]),
            "+f"(acc_[12][3]), "+f"(acc_[13][0]), "+f"(acc_[13][1]), "+f"(acc_[13][2]), "+f"(acc_[13][3]),
            "+f"(acc_[14][0]), "+f"(acc_[14][1]), "+f"(acc_[14][2]), "+f"(acc_[14][3]), "+f"(acc_[15][0]),
            "+f"(acc_[15][1]), "+f"(acc_[15][2]), "+f"(acc_[15][3])
            : "l"(aDesc_), "l"(bDesc_), "n"(aTrans_ ? 1 : 0), "n"(bTrans_ ? 1 : 0));
    else
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 \n"
            "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, \n"
            "  %8,  %9, %10, %11, %12, %13, %14, %15, \n"
            " %16, %17, %18, %19, %20, %21, %22, %23, \n"
            " %24, %25, %26, %27, %28, %29, %30, %31, \n"
            " %32, %33, %34, %35, %36, %37, %38, %39, \n"
            " %40, %41, %42, %43, %44, %45, %46, %47, \n"
            " %48, %49, %50, %51, %52, %53, %54, %55, \n"
            " %56, %57, %58, %59, %60, %61, %62, %63},\n"
            " %64, %65, 1,1,1, %66,%67;\n"
            : "+f"(acc_[0][0]), "+f"(acc_[0][1]), "+f"(acc_[0][2]), "+f"(acc_[0][3]), "+f"(acc_[1][0]),
            "+f"(acc_[1][1]), "+f"(acc_[1][2]), "+f"(acc_[1][3]), "+f"(acc_[2][0]), "+f"(acc_[2][1]), "+f"(acc_[2][2]),
            "+f"(acc_[2][3]), "+f"(acc_[3][0]), "+f"(acc_[3][1]), "+f"(acc_[3][2]), "+f"(acc_[3][3]), "+f"(acc_[4][0]),
            "+f"(acc_[4][1]), "+f"(acc_[4][2]), "+f"(acc_[4][3]), "+f"(acc_[5][0]), "+f"(acc_[5][1]), "+f"(acc_[5][2]),
            "+f"(acc_[5][3]), "+f"(acc_[6][0]), "+f"(acc_[6][1]), "+f"(acc_[6][2]), "+f"(acc_[6][3]), "+f"(acc_[7][0]),
            "+f"(acc_[7][1]), "+f"(acc_[7][2]), "+f"(acc_[7][3]), "+f"(acc_[8][0]), "+f"(acc_[8][1]), "+f"(acc_[8][2]),
            "+f"(acc_[8][3]), "+f"(acc_[9][0]), "+f"(acc_[9][1]), "+f"(acc_[9][2]), "+f"(acc_[9][3]), "+f"(acc_[10][0]),
            "+f"(acc_[10][1]), "+f"(acc_[10][2]), "+f"(acc_[10][3]), "+f"(acc_[11][0]), "+f"(acc_[11][1]),
            "+f"(acc_[11][2]), "+f"(acc_[11][3]), "+f"(acc_[12][0]), "+f"(acc_[12][1]), "+f"(acc_[12][2]),
            "+f"(acc_[12][3]), "+f"(acc_[13][0]), "+f"(acc_[13][1]), "+f"(acc_[13][2]), "+f"(acc_[13][3]),
            "+f"(acc_[14][0]), "+f"(acc_[14][1]), "+f"(acc_[14][2]), "+f"(acc_[14][3]), "+f"(acc_[15][0]),
            "+f"(acc_[15][1]), "+f"(acc_[15][2]), "+f"(acc_[15][3])
            : "l"(aDesc_), "l"(bDesc_), "n"(aTrans_ ? 1 : 0), "n"(bTrans_ ? 1 : 0));
#endif
}

template <class Tp_, bool aTrans_ = false, bool bTrans_ = false>
__device__ inline void wgmma(std::array<std::array<float, 4>, 32>& acc_, uint64_t aDesc_, uint64_t bDesc_)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
    if (std::is_same_v<Tp_, half>)
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 \n"
            "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, \n"
            "  %8,  %9, %10, %11, %12, %13, %14, %15, \n"
            " %16, %17, %18, %19, %20, %21, %22, %23, \n"
            " %24, %25, %26, %27, %28, %29, %30, %31, \n"
            " %32, %33, %34, %35, %36, %37, %38, %39, \n"
            " %40, %41, %42, %43, %44, %45, %46, %47, \n"
            " %48, %49, %50, %51, %52, %53, %54, %55, \n"
            " %56, %57, %58, %59, %60, %61, %62, %63, \n"
            "  %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71, \n"
            "  %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79, \n"
            "  %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87, \n"
            "  %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95, \n"
            "  %96,  %97,  %98,  %99, %100, %101, %102, %103, \n"
            " %104, %105, %106, %107, %108, %109, %110, %111, \n"
            " %112, %113, %114, %115, %116, %117, %118, %119, \n"
            " %120, %121, %122, %123, %124, %125, %126, %127},\n"
            " %128, %129, 1,1,1, %130,%131;\n"
            : "+f"(acc_[0][0]), "+f"(acc_[0][1]), "+f"(acc_[0][2]), "+f"(acc_[0][3]), "+f"(acc_[1][0]),
            "+f"(acc_[1][1]), "+f"(acc_[1][2]), "+f"(acc_[1][3]), "+f"(acc_[2][0]), "+f"(acc_[2][1]), "+f"(acc_[2][2]),
            "+f"(acc_[2][3]), "+f"(acc_[3][0]), "+f"(acc_[3][1]), "+f"(acc_[3][2]), "+f"(acc_[3][3]), "+f"(acc_[4][0]),
            "+f"(acc_[4][1]), "+f"(acc_[4][2]), "+f"(acc_[4][3]), "+f"(acc_[5][0]), "+f"(acc_[5][1]), "+f"(acc_[5][2]),
            "+f"(acc_[5][3]), "+f"(acc_[6][0]), "+f"(acc_[6][1]), "+f"(acc_[6][2]), "+f"(acc_[6][3]), "+f"(acc_[7][0]),
            "+f"(acc_[7][1]), "+f"(acc_[7][2]), "+f"(acc_[7][3]), "+f"(acc_[8][0]), "+f"(acc_[8][1]), "+f"(acc_[8][2]),
            "+f"(acc_[8][3]), "+f"(acc_[9][0]), "+f"(acc_[9][1]), "+f"(acc_[9][2]), "+f"(acc_[9][3]), "+f"(acc_[10][0]),
            "+f"(acc_[10][1]), "+f"(acc_[10][2]), "+f"(acc_[10][3]), "+f"(acc_[11][0]), "+f"(acc_[11][1]),
            "+f"(acc_[11][2]), "+f"(acc_[11][3]), "+f"(acc_[12][0]), "+f"(acc_[12][1]), "+f"(acc_[12][2]),
            "+f"(acc_[12][3]), "+f"(acc_[13][0]), "+f"(acc_[13][1]), "+f"(acc_[13][2]), "+f"(acc_[13][3]),
            "+f"(acc_[14][0]), "+f"(acc_[14][1]), "+f"(acc_[14][2]), "+f"(acc_[14][3]), "+f"(acc_[15][0]),
            "+f"(acc_[15][1]), "+f"(acc_[15][2]), "+f"(acc_[15][3]), "+f"(acc_[16][0]), "+f"(acc_[16][1]),
            "+f"(acc_[16][2]), "+f"(acc_[16][3]), "+f"(acc_[17][0]), "+f"(acc_[17][1]), "+f"(acc_[17][2]),
            "+f"(acc_[17][3]), "+f"(acc_[18][0]), "+f"(acc_[18][1]), "+f"(acc_[18][2]), "+f"(acc_[18][3]),
            "+f"(acc_[19][0]), "+f"(acc_[19][1]), "+f"(acc_[19][2]), "+f"(acc_[19][3]), "+f"(acc_[20][0]),
            "+f"(acc_[20][1]), "+f"(acc_[20][2]), "+f"(acc_[20][3]), "+f"(acc_[21][0]), "+f"(acc_[21][1]),
            "+f"(acc_[21][2]), "+f"(acc_[21][3]), "+f"(acc_[22][0]), "+f"(acc_[22][1]), "+f"(acc_[22][2]),
            "+f"(acc_[22][3]), "+f"(acc_[23][0]), "+f"(acc_[23][1]), "+f"(acc_[23][2]), "+f"(acc_[23][3]),
            "+f"(acc_[24][0]), "+f"(acc_[24][1]), "+f"(acc_[24][2]), "+f"(acc_[24][3]), "+f"(acc_[25][0]),
            "+f"(acc_[25][1]), "+f"(acc_[25][2]), "+f"(acc_[25][3]), "+f"(acc_[26][0]), "+f"(acc_[26][1]),
            "+f"(acc_[26][2]), "+f"(acc_[26][3]), "+f"(acc_[27][0]), "+f"(acc_[27][1]), "+f"(acc_[27][2]),
            "+f"(acc_[27][3]), "+f"(acc_[28][0]), "+f"(acc_[28][1]), "+f"(acc_[28][2]), "+f"(acc_[28][3]),
            "+f"(acc_[29][0]), "+f"(acc_[29][1]), "+f"(acc_[29][2]), "+f"(acc_[29][3]), "+f"(acc_[30][0]),
            "+f"(acc_[30][1]), "+f"(acc_[30][2]), "+f"(acc_[30][3]), "+f"(acc_[31][0]), "+f"(acc_[31][1]),
            "+f"(acc_[31][2]), "+f"(acc_[31][3])
            : "l"(aDesc_), "l"(bDesc_), "n"(aTrans_ ? 1 : 0), "n"(bTrans_ ? 1 : 0));
    else
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 \n"
            "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, \n"
            "  %8,  %9, %10, %11, %12, %13, %14, %15, \n"
            " %16, %17, %18, %19, %20, %21, %22, %23, \n"
            " %24, %25, %26, %27, %28, %29, %30, %31, \n"
            " %32, %33, %34, %35, %36, %37, %38, %39, \n"
            " %40, %41, %42, %43, %44, %45, %46, %47, \n"
            " %48, %49, %50, %51, %52, %53, %54, %55, \n"
            " %56, %57, %58, %59, %60, %61, %62, %63, \n"
            "  %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71, \n"
            "  %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79, \n"
            "  %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87, \n"
            "  %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95, \n"
            "  %96,  %97,  %98,  %99, %100, %101, %102, %103, \n"
            " %104, %105, %106, %107, %108, %109, %110, %111, \n"
            " %112, %113, %114, %115, %116, %117, %118, %119, \n"
            " %120, %121, %122, %123, %124, %125, %126, %127},\n"
            " %128, %129, 1,1,1, %130,%131;\n"
            : "+f"(acc_[0][0]), "+f"(acc_[0][1]), "+f"(acc_[0][2]), "+f"(acc_[0][3]), "+f"(acc_[1][0]),
            "+f"(acc_[1][1]), "+f"(acc_[1][2]), "+f"(acc_[1][3]), "+f"(acc_[2][0]), "+f"(acc_[2][1]), "+f"(acc_[2][2]),
            "+f"(acc_[2][3]), "+f"(acc_[3][0]), "+f"(acc_[3][1]), "+f"(acc_[3][2]), "+f"(acc_[3][3]), "+f"(acc_[4][0]),
            "+f"(acc_[4][1]), "+f"(acc_[4][2]), "+f"(acc_[4][3]), "+f"(acc_[5][0]), "+f"(acc_[5][1]), "+f"(acc_[5][2]),
            "+f"(acc_[5][3]), "+f"(acc_[6][0]), "+f"(acc_[6][1]), "+f"(acc_[6][2]), "+f"(acc_[6][3]), "+f"(acc_[7][0]),
            "+f"(acc_[7][1]), "+f"(acc_[7][2]), "+f"(acc_[7][3]), "+f"(acc_[8][0]), "+f"(acc_[8][1]), "+f"(acc_[8][2]),
            "+f"(acc_[8][3]), "+f"(acc_[9][0]), "+f"(acc_[9][1]), "+f"(acc_[9][2]), "+f"(acc_[9][3]), "+f"(acc_[10][0]),
            "+f"(acc_[10][1]), "+f"(acc_[10][2]), "+f"(acc_[10][3]), "+f"(acc_[11][0]), "+f"(acc_[11][1]),
            "+f"(acc_[11][2]), "+f"(acc_[11][3]), "+f"(acc_[12][0]), "+f"(acc_[12][1]), "+f"(acc_[12][2]),
            "+f"(acc_[12][3]), "+f"(acc_[13][0]), "+f"(acc_[13][1]), "+f"(acc_[13][2]), "+f"(acc_[13][3]),
            "+f"(acc_[14][0]), "+f"(acc_[14][1]), "+f"(acc_[14][2]), "+f"(acc_[14][3]), "+f"(acc_[15][0]),
            "+f"(acc_[15][1]), "+f"(acc_[15][2]), "+f"(acc_[15][3]), "+f"(acc_[16][0]), "+f"(acc_[16][1]),
            "+f"(acc_[16][2]), "+f"(acc_[16][3]), "+f"(acc_[17][0]), "+f"(acc_[17][1]), "+f"(acc_[17][2]),
            "+f"(acc_[17][3]), "+f"(acc_[18][0]), "+f"(acc_[18][1]), "+f"(acc_[18][2]), "+f"(acc_[18][3]),
            "+f"(acc_[19][0]), "+f"(acc_[19][1]), "+f"(acc_[19][2]), "+f"(acc_[19][3]), "+f"(acc_[20][0]),
            "+f"(acc_[20][1]), "+f"(acc_[20][2]), "+f"(acc_[20][3]), "+f"(acc_[21][0]), "+f"(acc_[21][1]),
            "+f"(acc_[21][2]), "+f"(acc_[21][3]), "+f"(acc_[22][0]), "+f"(acc_[22][1]), "+f"(acc_[22][2]),
            "+f"(acc_[22][3]), "+f"(acc_[23][0]), "+f"(acc_[23][1]), "+f"(acc_[23][2]), "+f"(acc_[23][3]),
            "+f"(acc_[24][0]), "+f"(acc_[24][1]), "+f"(acc_[24][2]), "+f"(acc_[24][3]), "+f"(acc_[25][0]),
            "+f"(acc_[25][1]), "+f"(acc_[25][2]), "+f"(acc_[25][3]), "+f"(acc_[26][0]), "+f"(acc_[26][1]),
            "+f"(acc_[26][2]), "+f"(acc_[26][3]), "+f"(acc_[27][0]), "+f"(acc_[27][1]), "+f"(acc_[27][2]),
            "+f"(acc_[27][3]), "+f"(acc_[28][0]), "+f"(acc_[28][1]), "+f"(acc_[28][2]), "+f"(acc_[28][3]),
            "+f"(acc_[29][0]), "+f"(acc_[29][1]), "+f"(acc_[29][2]), "+f"(acc_[29][3]), "+f"(acc_[30][0]),
            "+f"(acc_[30][1]), "+f"(acc_[30][2]), "+f"(acc_[30][3]), "+f"(acc_[31][0]), "+f"(acc_[31][1]),
            "+f"(acc_[31][2]), "+f"(acc_[31][3])
            : "l"(aDesc_), "l"(bDesc_), "n"(aTrans_ ? 1 : 0), "n"(bTrans_ ? 1 : 0));
#endif
}

typedef __nv_bfloat16 bf16;
typedef __nv_bfloat162 bf162;

template <int mode_ = 128, int line_ = 64>
__device__ int swz(int x_)
{
    return x_ ^ x_ / line_ % (mode_ / 16) * 8;
}

// vim: ts=2 sw=2 sts=2 et sta
