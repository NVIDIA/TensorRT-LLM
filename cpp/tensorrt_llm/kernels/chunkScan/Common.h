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

#if __CUDA_ARCH__ >= 800
    if constexpr (size_ == 16)
        asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(s_ptr), "l"(g_ptr), "n"(size_));
    else if constexpr (size_ == 8)
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(s_ptr), "l"(g_ptr), "n"(size_));
    else if constexpr (size_ == 4)
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(s_ptr), "l"(g_ptr), "n"(size_));
#else
    register unsigned tmp[size_ / 4];

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

#if __CUDA_ARCH__ >= 800
    if constexpr (size_ == 16)
        asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"r"(s_ptr), "l"(g_ptr), "n"(size_),
            "r"(valid_ ? size_ : 0));
    else if constexpr (size_ == 8)
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(s_ptr), "l"(g_ptr), "n"(size_),
            "r"(valid_ ? size_ : 0));
    else if constexpr (size_ == 4)
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(s_ptr), "l"(g_ptr), "n"(size_),
            "r"(valid_ ? size_ : 0));
#else
    register unsigned tmp[size_ / 4];

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
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n");
#endif
}

template <int remain_>
__device__ inline void cp_wait_group()
{
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" ::"n"(remain_));
#endif
}

template <bool trans_ = false>
__device__ inline void ldmatrix(unsigned& r0_, unsigned& r1_, unsigned& r2_, unsigned& r3_, unsigned addr_)
{
    if (trans_)
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(r0_), "=r"(r1_), "=r"(r2_), "=r"(r3_)
                     : "r"(addr_));
    else
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(r0_), "=r"(r1_), "=r"(r2_), "=r"(r3_)
                     : "r"(addr_));
}

typedef __nv_bfloat16 bf16;
typedef __nv_bfloat162 bf162;

template <int mode_ = 128, int line_ = 64>
__device__ int swz(int x_)
{
    return x_ ^ x_ / line_ % (mode_ / 16) * 8;
}

// vim: ts=2 sw=2 sts=2 et sta
