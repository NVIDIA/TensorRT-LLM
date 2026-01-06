// Adapted from https://github.com/nunchaku-tech/nunchaku
// @article{
//   li2024svdquant,
//   title={SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models},
//   author={Li*, Muyang and Lin*, Yujun and Zhang*, Zhekai and Cai, Tianle and Li, Xiuyu and Guo, Junxian and Xie, Enze and Meng, Chenlin and Zhu, Jun-Yan and Han, Song},
//   journal={arXiv preprint arXiv:2411.05007},
//   year={2024}
// }

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>
#include "common.h"

// cuda 12.4- does not support "C" constraint in inline assembly :(
// use explicit specialization for now

namespace nunchaku::kernels {

namespace mma_helper {
struct f32 {
    static constexpr const char value[] = "f32";
};
struct f16 {
    static constexpr const char value[] = "f16";
};
struct bf16 {
    static constexpr const char value[] = "bf16";
};
struct s32 {
    static constexpr const char value[] = "s32";
};
struct s4 {
    static constexpr const char value[] = "s4";
};
struct u4 {
    static constexpr const char value[] = "u4";
};

template<bool is_bf16>
using f16bf16 = std::conditional_t<is_bf16, bf16, f16>;
template<bool is_unsigned>
using s4u4 = std::conditional_t<is_unsigned, u4, s4>;
}; // namespace mma_helper

__device__ __forceinline__ static uint2 mma_m16n8k16_f16f16f16f16(uint4 a, uint2 b, uint2 c) {
    uint2 d;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0,  %1},"
                 "{%2,  %3,  %4,  %5},"
                 "{%6,  %7},"
                 "{%8,  %9};\n"
                 : "=r"(d.x), "=r"(d.y)
                 : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y), "r"(c.x), "r"(c.y));
#else
    asm volatile("{"
                 ".reg .b32 tmp0, tmp1;"
                 "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                 "{tmp0,  tmp1},"
                 "{%2,  %3},"
                 "{%6},"
                 "{%8,  %9};\n"
                 "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                 "{%0,  %1},"
                 "{%4,  %5},"
                 "{%7},"
                 "{tmp0,  tmp1};"
                 "}\n"
                 : "=r"(d.x), "=r"(d.y)
                 : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y), "r"(c.x), "r"(c.y));
#endif
    return d;
}

template<bool is_bf16>
__device__ __forceinline__ static uint4 mma_m16n8k16_f32f16f16f32(uint4 a, uint2 b, uint4 c) = delete;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template<>
__device__ __forceinline__ uint4 mma_m16n8k16_f32f16f16f32<true>(uint4 a, uint2 b, uint4 c) {
    uint4 d;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,  %1,  %2,  %3},"
                 "{%4,  %5,  %6,  %7},"
                 "{%8,  %9},"
                 "{%10,  %11,  %12,  %13};\n"
                 : "=r"(d.x), "=r"(d.y), "=r"(d.z), "=r"(d.w)
                 : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y), "r"(c.x), "r"(c.y), "r"(c.z), "r"(c.w));
    return d;
}
#endif

template<>
__device__ __forceinline__ uint4 mma_m16n8k16_f32f16f16f32<false>(uint4 a, uint2 b, uint4 c) {
    uint4 d;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,  %1,  %2,  %3},"
                 "{%4,  %5,  %6,  %7},"
                 "{%8,  %9},"
                 "{%10,  %11,  %12,  %13};\n"
                 : "=r"(d.x), "=r"(d.y), "=r"(d.z), "=r"(d.w)
                 : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y), "r"(c.x), "r"(c.y), "r"(c.z), "r"(c.w));
#else
    asm volatile("{"
                 ".reg .b32 tmp0, tmp1, tmp2, tmp3;"
                 "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{tmp0,  tmp1,  tmp2,  tmp3},"
                 "{%4,  %5},"
                 "{%8},"
                 "{%10,  %11,  %12,  %13};\n"
                 "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,  %1,  %2,  %3},"
                 "{%6,  %7},"
                 "{%9},"
                 "{tmp0,  tmp1,  tmp2,  tmp3};"
                 "}\n"
                 : "=r"(d.x), "=r"(d.y), "=r"(d.z), "=r"(d.w)
                 : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y), "r"(c.x), "r"(c.y), "r"(c.z), "r"(c.w));
#endif
    return d;
}

template<typename AType, typename BType>
__device__ __forceinline__ static uint4 mma_m16n8kx_s32common(uint4 a, uint2 b, uint4 c) = delete;

template<>
__device__ __forceinline__ uint4 mma_m16n8kx_s32common<mma_helper::s4, mma_helper::s4>(uint4 a, uint2 b, uint4 c) {
    uint4 d;
    static constexpr int K = 64;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k%14.row.col.s32.s4.s4.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10,  %11,  %12,  %13};\n"
        : "=r"(d.x), "=r"(d.y), "=r"(d.z), "=r"(d.w)
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y), "r"(c.x), "r"(c.y), "r"(c.z), "r"(c.w), "n"(K));
#else
    asm volatile("{"
                 ".reg .b32 tmp0, tmp1, tmp2, tmp3;"
                 "mma.sync.aligned.m8n8k%14.row.col.s32.s4.s4.s32 "
                 "{tmp0, tmp1},"
                 "{%4},"
                 "{%8},"
                 "{%10,  %11};\n"
                 "mma.sync.aligned.m8n8k%14.row.col.s32.s4.s4.s32 "
                 "{tmp2, tmp3},"
                 "{%5},"
                 "{%8},"
                 "{%12,  %13};\n"
                 "mma.sync.aligned.m8n8k%14.row.col.s32.s4.s4.s32 "
                 "{%0,  %1},"
                 "{%6},"
                 "{%9},"
                 "{tmp0, tmp1};\n"
                 "mma.sync.aligned.m8n8k%14.row.col.s32.s4.s4.s32 "
                 "{%2,  %3},"
                 "{%7},"
                 "{%9},"
                 "{tmp2, tmp3};\n"
                 "}\n"
                 : "=r"(d.x), "=r"(d.y), "=r"(d.z), "=r"(d.w)
                 : "r"(a.x),
                   "r"(a.y),
                   "r"(a.z),
                   "r"(a.w),
                   "r"(b.x),
                   "r"(b.y),
                   "r"(c.x),
                   "r"(c.y),
                   "r"(c.z),
                   "r"(c.w),
                   "n"(K / 2));
#endif
    return d;
}

template<>
__device__ __forceinline__ uint4 mma_m16n8kx_s32common<mma_helper::u4, mma_helper::s4>(uint4 a, uint2 b, uint4 c) {
    uint4 d;
    static constexpr int K = 64;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k%14.row.col.s32.u4.s4.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10,  %11,  %12,  %13};\n"
        : "=r"(d.x), "=r"(d.y), "=r"(d.z), "=r"(d.w)
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y), "r"(c.x), "r"(c.y), "r"(c.z), "r"(c.w), "n"(K));
#else
    asm volatile("{"
                 ".reg .b32 tmp0, tmp1, tmp2, tmp3;"
                 "mma.sync.aligned.m8n8k%14.row.col.s32.u4.s4.s32 "
                 "{tmp0, tmp1},"
                 "{%4},"
                 "{%8},"
                 "{%10,  %11};\n"
                 "mma.sync.aligned.m8n8k%14.row.col.s32.u4.s4.s32 "
                 "{tmp2, tmp3},"
                 "{%5},"
                 "{%8},"
                 "{%12,  %13};\n"
                 "mma.sync.aligned.m8n8k%14.row.col.s32.u4.s4.s32 "
                 "{%0,  %1},"
                 "{%6},"
                 "{%9},"
                 "{tmp0, tmp1};\n"
                 "mma.sync.aligned.m8n8k%14.row.col.s32.u4.s4.s32 "
                 "{%2,  %3},"
                 "{%7},"
                 "{%9},"
                 "{tmp2, tmp3};\n"
                 "}\n"
                 : "=r"(d.x), "=r"(d.y), "=r"(d.z), "=r"(d.w)
                 : "r"(a.x),
                   "r"(a.y),
                   "r"(a.z),
                   "r"(a.w),
                   "r"(b.x),
                   "r"(b.y),
                   "r"(c.x),
                   "r"(c.y),
                   "r"(c.z),
                   "r"(c.w),
                   "n"(K / 2));
#endif
    return d;
}

}; // namespace nunchaku::kernels
