/*
 * SPDX-FileCopyrightText: Copyright (c) 2011-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <assert.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(__CLANGD__)
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_math.h>
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

// include warpgroup related instructions, used by SM90.
#include <fmha/hopper/utils_warpgroup.h>
// include gmma related instructions, used by SM90.
#include <fmha/hopper/utils_gmma.h>
// include tma related instructions, used by SM90.
#include <fmha/hopper/utils_tma.h>

#include "fmha/numeric_types.h"

#define FP32_I2F_MAGIC_NUMBER 12582912.f
#define FP32_I2F_MAGIC_NUMBER_HEX 0x4b400000

extern "C" __device__ uint32_t __nvvm_get_smem_pointer(void* ptr);

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace introspection
{

template <int... Ns>
struct Unpack;

template <int N>
struct Unpack<N>
{
    // if we simply static_assert(false) then compiler will not emit template params upon failure
    static_assert(N < INT_MIN, "");
    using Type = std::integral_constant<int, N>;
};

template <int N, int... Ns>
struct Unpack<N, Ns...>
{
    using Type = Unpack<N, Ns...>;
    using Unpack_first = typename Unpack<N>::Type;
    using Unpack_remaining = typename Unpack<Ns...>::Type;
};

} // namespace introspection

// Example usage:
//
//   Inspect_ns<(int)USE_LDGSTS_, PRED_REGS, (int)IS_HOPPER> foo;
//
// or
//
//   Inspect_ns<(int)USE_LDGSTS_, PRED_REGS, (int)IS_HOPPER>{}.foo();
//
// Output by nvcc:
//
//   ./src/fmha/gmem_tile_qkv_packed.h(70): error: static assertion failed with ""
//             detected during:
//               instantiation of class "fmha::v2::Unpack<N> [with N=1]"
//   (77): here
//               instantiation of class "fmha::v2::Unpack<N, Ns...> [with N=1, Ns=<2, 0>]"
//   (84): here
//               instantiation of class "fmha::v2::Inspect_ns<Ns...> [with Ns=<1, 2, 0>]"
//   (143): here
template <int... Ns>
struct Inspect_ns
{
    using Type = typename introspection::Unpack<Ns...>::Type;
};

// Can be used alongside with static_assert() to figure out the conditions when assertion failed
// Example:
//
//   Cond_inspect_ns< (int)ROWS >= (int)ROWS_PER_LDG, ROWS, ROWS_PER_LDG> foo;
//
// Output by nvcc (when condition is not met):
//
//   ./src/fmha/utils.h(163): error: static assertion failed with ""
//             detected during:
//               instantiation of class "Cond_inspect_ns<COND, Ns...> [with COND=false, Ns=<32, 64>]"
template <bool COND, int... Ns>
struct Cond_inspect_ns
{
    static_assert(COND, "");
};

// Example:
//
//    Inspect_type<Mma_tile_p>{}.foo();
//
// or
//
//    Inspect_type<Mma_tile_p> foo;
//
// Output by nvcc:
//
//   ./src/fmha/utils.h(189): error: class "fmha::Ampere_hmma_tile<fmha::Cta_tile_<fmha::Ampere, 64, 128, 64, 128, 256,
//   4, 1, 1>, 16>" has no member "Dummy"
//             detected during:
//               instantiation of class "Inspect_type<T> [with T=fmha::Ampere_hmma_tile<fmha::Cta_tile_<fmha::Ampere,
//               64, 128, 64, 128, 256, 4, 1, 1>, 16>]"
template <typename T>
struct Inspect_type
{
    // Purposefully trigger error by referencing non-existent T::Dummy
    using Dummy = typename T::Dummy;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace fmha
{

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Row
{
    static constexpr bool COL = false;
    static constexpr bool ROW = true;
};

struct Col
{
    static constexpr bool COL = true;
    static constexpr bool ROW = false;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int M, int N>
struct Round_up
{
    enum
    {
        VALUE = (M + N - 1) / N * N
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N_, int H_, int W_>
struct Tile_nhw
{
    enum
    {
        N = N_,
        H = H_,
        W = W_
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int M, bool = (M & (M - 1)) == 0>
struct Next_power_of_two
{
};

template <int M>
struct Next_power_of_two<M, true>
{
    enum
    {
        VALUE = M
    };
};

template <>
struct Next_power_of_two<3, false>
{
    enum
    {
        VALUE = 4
    };
};

template <>
struct Next_power_of_two<5, false>
{
    enum
    {
        VALUE = 8
    };
};

template <>
struct Next_power_of_two<6, false>
{
    enum
    {
        VALUE = 8
    };
};

template <>
struct Next_power_of_two<7, false>
{
    enum
    {
        VALUE = 8
    };
};

template <>
struct Next_power_of_two<9, false>
{
    enum
    {
        VALUE = 16
    };
};

template <>
struct Next_power_of_two<10, false>
{
    enum
    {
        VALUE = 16
    };
};

template <>
struct Next_power_of_two<11, false>
{
    enum
    {
        VALUE = 16
    };
};

template <>
struct Next_power_of_two<12, false>
{
    enum
    {
        VALUE = 16
    };
};

template <>
struct Next_power_of_two<13, false>
{
    enum
    {
        VALUE = 16
    };
};

template <>
struct Next_power_of_two<14, false>
{
    enum
    {
        VALUE = 16
    };
};

template <>
struct Next_power_of_two<15, false>
{
    enum
    {
        VALUE = 16
    };
};

template <>
struct Next_power_of_two<24, false>
{
    enum
    {
        VALUE = 32
    };
};

template <>
struct Next_power_of_two<40, false>
{
    enum
    {
        VALUE = 64
    };
};

template <>
struct Next_power_of_two<48, false>
{
    enum
    {
        VALUE = 64
    };
};

template <>
struct Next_power_of_two<72, false>
{
    enum
    {
        VALUE = 128
    };
};

template <>
struct Next_power_of_two<80, false>
{
    enum
    {
        VALUE = 128
    };
};

template <>
struct Next_power_of_two<96, false>
{
    enum
    {
        VALUE = 128
    };
};

template <>
struct Next_power_of_two<104, false>
{
    enum
    {
        VALUE = 128
    };
};

template <>
struct Next_power_of_two<112, false>
{
    enum
    {
        VALUE = 128
    };
};

template <>
struct Next_power_of_two<144, false>
{
    enum
    {
        VALUE = 256
    };
};

template <>
struct Next_power_of_two<160, false>
{
    enum
    {
        VALUE = 256
    };
};

template <>
struct Next_power_of_two<192, false>
{
    enum
    {
        VALUE = 256
    };
};

template <>
struct Next_power_of_two<576, false>
{
    enum
    {
        VALUE = 1024
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N, bool = (N & (N - 1)) == 0>
struct Prev_power_of_two
{
};

template <int N>
struct Prev_power_of_two<N, true>
{
    enum
    {
        VALUE = N
    };
};

template <>
struct Prev_power_of_two<3, false>
{
    enum
    {
        VALUE = 2
    };
};

template <>
struct Prev_power_of_two<5, false>
{
    enum
    {
        VALUE = 4
    };
};

template <>
struct Prev_power_of_two<6, false>
{
    enum
    {
        VALUE = 4
    };
};

template <>
struct Prev_power_of_two<7, false>
{
    enum
    {
        VALUE = 4
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int BYTES_PER_ROW, int SKEW>
struct Compute_skew
{
    // The size of a transaction.
    enum
    {
        BYTES_PER_TRX = 128
    };

    // The remainder of the row without skew.
    enum
    {
        REMAINDER = BYTES_PER_ROW % BYTES_PER_TRX
    };

    // The value.
    enum
    {
        VALUE = REMAINDER <= SKEW ? SKEW - REMAINDER : BYTES_PER_TRX + SKEW - REMAINDER
    };

    // Make sure the math works ;)
    static_assert((BYTES_PER_ROW + VALUE) % BYTES_PER_TRX == SKEW, "");
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int BYTES_PER_ROW>
struct Compute_skew<BYTES_PER_ROW, 128>
{
    // No skew!
    enum
    {
        VALUE = 0
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int M, int N>
struct Div_up
{
    enum
    {
        VALUE = (M + N - 1) / N
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int A, int B>
struct Max
{
    enum
    {
        VALUE = A >= B ? A : B
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int A, int B, int C>
struct Max_3
{
    enum
    {
        VALUE = Max<Max<A, B>::VALUE, C>::VALUE
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int A, int B>
struct Min
{
    enum
    {
        VALUE = A <= B ? A : B
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int SIZE_IN_BYTES>
struct Uint_from_size_in_bytes
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Uint_from_size_in_bytes<1>
{
    using Type = uint8_t;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Uint_from_size_in_bytes<2>
{
    using Type = uint16_t;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Uint_from_size_in_bytes<4>
{
    using Type = uint32_t;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Uint_from_size_in_bytes<8>
{
    using Type = uint2;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Uint_from_size_in_bytes<16>
{
    using Type = uint4;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int WARPS_M, int WARPS_N, int WARPS_K>
struct Warp_masks
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Warp_masks<8, 1, 1>
{
    enum
    {
        M = 0xe0,
        N = 0x00,
        K = 0x00
    };
};

template <>
struct Warp_masks<4, 2, 1>
{
    enum
    {
        M = 0x60,
        N = 0x80,
        K = 0x00
    };
};

template <>
struct Warp_masks<4, 1, 2>
{
    enum
    {
        M = 0x60,
        N = 0x00,
        K = 0x80
    };
};

template <>
struct Warp_masks<4, 1, 1>
{
    enum
    {
        M = 0x60,
        N = 0x00,
        K = 0x00
    };
};

template <>
struct Warp_masks<2, 4, 1>
{
    enum
    {
        M = 0x20,
        N = 0xc0,
        K = 0x00
    };
};

template <>
struct Warp_masks<2, 2, 2>
{
    enum
    {
        M = 0x20,
        N = 0x40,
        K = 0x80
    };
};

template <>
struct Warp_masks<2, 2, 1>
{
    enum
    {
        M = 0x20,
        N = 0x40,
        K = 0x00
    };
};

template <>
struct Warp_masks<2, 1, 2>
{
    enum
    {
        M = 0x20,
        N = 0x00,
        K = 0x40
    };
};

template <>
struct Warp_masks<2, 1, 1>
{
    enum
    {
        M = 0x20,
        N = 0x00,
        K = 0x00
    };
};

template <>
struct Warp_masks<1, 8, 1>
{
    enum
    {
        M = 0x00,
        N = 0xe0,
        K = 0x00
    };
};

template <>
struct Warp_masks<1, 4, 2>
{
    enum
    {
        M = 0x00,
        N = 0x60,
        K = 0x80
    };
};

template <>
struct Warp_masks<1, 4, 1>
{
    enum
    {
        M = 0x00,
        N = 0x60,
        K = 0x00
    };
};

template <>
struct Warp_masks<1, 2, 2>
{
    enum
    {
        M = 0x00,
        N = 0x20,
        K = 0x40
    };
};

template <>
struct Warp_masks<1, 2, 1>
{
    enum
    {
        M = 0x00,
        N = 0x20,
        K = 0x00
    };
};

template <>
struct Warp_masks<1, 1, 4>
{
    enum
    {
        M = 0x00,
        N = 0x00,
        K = 0x60
    };
};

template <>
struct Warp_masks<1, 1, 2>
{
    enum
    {
        M = 0x00,
        N = 0x00,
        K = 0x20
    };
};

template <>
struct Warp_masks<1, 1, 1>
{
    enum
    {
        M = 0x00,
        N = 0x00,
        K = 0x00
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline __device__ __host__ T div_up(T m, T n)
{
    return (m + n - 1) / n;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline int clz(int x)
{
    for (int i = 31; i >= 0; --i)
    {
        if ((1 << i) & x)
        {
            return 31 - i;
        }
    }
    return 32;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline int find_log_2(int x, bool round_up = false)
{
    int a = 31 - clz(x);
    if (round_up)
    {
        a += (x & (x - 1)) ? 1 : 0;
    }
    return a;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline void find_divisor(uint32_t& mul, uint32_t& shr, int x)
{
    assert(x != 0);
    if (x == 1)
    {
        // If dividing by 1, reduced math doesn't work because mul_coeff would need to be 2^32,
        // which doesn't fit into unsigned int.  the div() routine handles this special case
        // separately.
        mul = 0;
        shr = 0;
    }
    else
    {
        // To express the division N/D in terms of a multiplication, what we first
        // imagine is simply N*(1/D).  However, 1/D will always evaluate to 0 (for D>1),
        // so we need another way.  There's nothing that says we have to use exactly
        // the fraction 1/D; instead it could be any X/Y that reduces to 1/D (i.e.,
        // Y=X*D), or at least to "close enough" to it.  If we pick Y that is a power
        // of two, then the N*(X/Y) can be N*X followed by a right-shift by some amount.
        // The power of two we should pick should be at least 2^32, because in the
        // div() routine we'll use umulhi(), which returns only the upper 32 bits --
        // this being equivalent to a right-shift by 32.  But we might want a higher
        // power of two for better accuracy depending on the magnitude of the denominator.
        // Once we've picked Y, then X [our mul_coeff value] is simply Y/D, rounding up,
        // and we save shift_coeff as whatever further shift we have to do beyond
        // what the umulhi() implies.
        uint32_t p = 31 + find_log_2(x, true);
        uint32_t m = (uint32_t) (((1ull << p) + (uint32_t) x - 1) / (uint32_t) x);

        mul = m;
        shr = p - 32;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void fast_divmod(int& div, int& mod, int x, int y, uint32_t mul, uint32_t shr)
{
    if (y == 1)
    {
        div = x;
        mod = 0;
    }
    else
    {
        div = __umulhi((uint32_t) x, mul) >> shr;
        mod = x - div * y;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t hadd2(uint32_t a, uint32_t b)
{
    uint32_t c;
    asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t bfadd2(uint32_t a, uint32_t b)
{
    uint32_t c;
    uint32_t one = 0x3f803f80;
    ;
    asm volatile("fma.rn.bf16x2 %0, %1, %3, %2;\n" : "=r"(c) : "r"(a), "r"(b), "r"(one));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t hmax2(uint32_t a, uint32_t b)
{
    uint32_t c;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("max.f16x2 %0, %1, %2;" : "=r"(c) : "r"(a), "r"(b));
#else
    asm volatile(
        "{\n"
        "\t .reg .f16x2 sela, selb;\n"
        "\n"
        "\t set.ge.f16x2.f16x2 sela, %1, %2;\n"
        "\t set.gt.f16x2.f16x2 selb, %2, %1;\n"
        "\n"
        "\t mul.f16x2 %0, sela, %1;\n"
        "\t fma.rn.f16x2 %0, selb, %2, %0;\n"
        "}\n"
        : "=r"(c)
        : "r"(a), "r"(b));
#endif
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint2 hmax4(uint2 a, uint2 b)
{
    uint2 c;
    c.x = hmax2(a.x, b.x);
    c.y = hmax2(a.y, b.y);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint4 hmax8(uint4 a, uint4 b)
{
    uint4 c;
    c.x = hmax2(a.x, b.x);
    c.y = hmax2(a.y, b.y);
    c.z = hmax2(a.z, b.z);
    c.w = hmax2(a.w, b.w);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t hmin2(uint32_t a, uint32_t b)
{
    uint32_t c;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("min.f16x2 %0, %1, %2;" : "=r"(c) : "r"(a), "r"(b));
#else
    asm volatile(
        "{\n"
        "\t .reg .f16x2 sela, selb;\n"
        "\n"
        "\t set.le.f16x2.f16x2 sela, %1, %2;\n"
        "\t set.lt.f16x2.f16x2 selb, %2, %1;\n"
        "\n"
        "\t mul.f16x2 %0, sela, %1;\n"
        "\t fma.rn.f16x2 %0, selb, %2, %0;\n"
        "}\n"
        : "=r"(c)
        : "r"(a), "r"(b));
#endif
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t hmul2(uint32_t a, uint32_t b)
{
    uint32_t c;
    asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t bfmul2(uint32_t a, uint32_t b)
{
    uint32_t c;
    asm("{.reg .b32 c;\n"
        "  mov.b32 c, 0x80008000U;\n"
        "  fma.rn.bf16x2 %0,%1,%2,c;}\n"
        : "=r"(c)
        : "r"(a), "r"(b));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint2 hmul4(uint2 a, uint2 b)
{
    uint2 c;
    c.x = hmul2(a.x, b.x);
    c.y = hmul2(a.y, b.y);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint4 hmul8(uint4 a, uint4 b)
{
    uint4 c;
    c.x = hmul2(a.x, b.x);
    c.y = hmul2(a.y, b.y);
    c.z = hmul2(a.z, b.z);
    c.w = hmul2(a.w, b.w);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint4 hmul8(uint32_t a, uint4 b)
{
    uint4 c;
    c.x = hmul2(a, b.x);
    c.y = hmul2(a, b.y);
    c.z = hmul2(a, b.z);
    c.w = hmul2(a, b.w);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Template function to support both half and bfloat16
template <typename Data_type>
inline __device__ uint32_t mul2(uint32_t a, uint32_t b)
{
    return hmul2(a, b);
}

template <>
inline __device__ uint32_t mul2<bf16_t>(uint32_t a, uint32_t b)
{
    return bfmul2(a, b);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Template function to support both half and bfloat16
template <typename Data_type>
inline __device__ uint4 mul8(uint32_t a, uint4 b)
{
    uint4 c;
    c.x = hmul2(a, b.x);
    c.y = hmul2(a, b.y);
    c.z = hmul2(a, b.z);
    c.w = hmul2(a, b.w);
    return c;
}

template <>
inline __device__ uint4 mul8<bf16_t>(uint32_t a, uint4 b)
{
    uint4 c;
    c.x = bfmul2(a, b.x);
    c.y = bfmul2(a, b.y);
    c.z = bfmul2(a, b.z);
    c.w = bfmul2(a, b.w);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t hrelu2(uint32_t x)
{
    uint32_t res;
    uint32_t const zero = 0u;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("max.f16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(zero));
#else
    asm volatile(
        "{\n"
        "\t .reg .f16x2 sela;\n"
        "\t set.gtu.u32.f16x2 sela, %1, %2;\n"
        "\t and.b32 %0, sela, %1;\n"
        "}\n"
        : "=r"(res)
        : "r"(x), "r"(zero));
#endif
    return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t bfrelu2(uint32_t x)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint32_t res;
    uint32_t const zero = 0u;
    asm volatile("max.bf16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(zero));
    return res;
#endif
    // not implemented yet
    return x;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Template function to support both half and bfloat16
template <typename Data_type>
inline __device__ uint32_t relu2(uint32_t x)
{
    return hrelu2(x);
}

template <>
inline __device__ uint32_t relu2<bf16_t>(uint32_t x)
{
    return bfrelu2(x);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t habs2(uint32_t x)
{
    uint32_t res;
    asm volatile("abs.f16x2 %0, %1;\n" : "=r"(res) : "r"(x));
    return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// static inline __device__ uint32_t add_bias(uint32_t a, uint32_t bias, bool relu) {
//     uint32_t c;
// #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
//     if( relu ) {
//         uint32_t one = 0x3c003c00u;
//         asm volatile("fma.rn.f16x2.relu %0, %1, %2, %3;" : "=r"(c) : "r"(a), "r"(one), "r"(bias));
//     } else {
//         c = hadd2(a, bias);
//     }
// #else
//     c = hadd2(a, bias);
//     if( relu ) {
//         c = hrelu2(c);
//     }
// #endif
//     return c;
// }

////////////////////////////////////////////////////////////////////////////////////////////////////

// static inline __device__ uint2 add_bias(uint2 a, uint2 bias, bool relu) {
//     uint2 dst;
//     dst.x = add_bias(a.x, bias.x, relu);
//     dst.y = add_bias(a.y, bias.y, relu);
//     return dst;
// }

////////////////////////////////////////////////////////////////////////////////////////////////////

// static inline __device__ uint4 add_bias(uint4 a, uint4 bias, bool relu) {
//     uint4 dst;
//     dst.x = add_bias(a.x, bias.x, relu);
//     dst.y = add_bias(a.y, bias.y, relu);
//     dst.z = add_bias(a.z, bias.z, relu);
//     dst.w = add_bias(a.w, bias.w, relu);
//     return dst;
// }

////////////////////////////////////////////////////////////////////////////////////////////////////

// clamp float +inf/-inf
static inline __device__ float satfinite(float x)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
    // bit representation of maximum value of float
    uint32_t clamp_value = 0x7f7fffffu;
    asm volatile("min.xorsign.abs.f32 %0, %0, %1;" : "+f"(x) : "r"(clamp_value));
    return x;
#else
    // bit representation of maximum and minimum value of float
    uint32_t umax = 0x7f7fffffu;
    uint32_t umin = 0xff7fffffu;
    float out;
    asm volatile("min.f32 %0, %1, %2;" : "=f"(out) : "f"(x), "r"(umax));
    asm volatile("max.f32 %0, %0, %1;" : "+f"(out) : "r"(umin));
    return out;
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// clamp half2 +inf/-inf
static inline __device__ uint32_t satfinite_h2(uint32_t h2)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
    uint32_t out, clamp_value;
    clamp_value = 0x7bff7bffu;
    asm volatile("min.xorsign.abs.f16x2 %0, %1, %2;" : "=r"(out) : "r"(h2), "r"(clamp_value));
    return out;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 800
    // bit representation of maximum and minimum value of half2
    uint32_t umax = 0x7bff7bffu;
    uint32_t umin = 0xfbfffbffu;
    uint32_t out;
    asm volatile("min.f16x2 %0, %1, %2;" : "=r"(out) : "r"(h2), "r"(umax));
    asm volatile("max.f16x2 %0, %0, %1;" : "+r"(out) : "r"(umin));
    return out;
#else
    // Take the absolute value of h2. It should map to |Rx| in SASS.
    uint32_t p2;
    asm volatile("abs.f16x2 %0, %1;" : "=r"(p2) : "r"(h2));

    // Compute a mask for each fp16: 0xffff if +INF and 0x0000 otherwise.
    uint32_t inf2 = 0x7c007c00u;
    uint32_t mask;
    asm volatile("set.eq.u32.f16x2 %0, %1, %2;" : "=r"(mask) : "r"(p2), "r"(inf2));

    // Recreate the new value. 0x7bff is the max value for FP16.
    p2 = (~mask & p2) | (mask & 0x7bff7bff);

    // Simply re-add the sign and we're done.
    return p2 | (h2 & 0x80008000);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
static inline __device__ T clamp(T x, T lb, T ub)
{
    return x < lb ? lb : (x > ub ? ub : x);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float custom_exp2f(float x, float scale, float scaled_max)
{
    float d1, d2;
    asm("fma.rz.ftz.f32 %0, %1, %2, %3;" : "=f"(d1) : "f"(x), "f"(scale), "f"(-scaled_max));
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(d2) : "f"(d1));
    return d2;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint16_t clamp_to_zero(uint16_t x)
{
    uint16_t mask;
    asm volatile("set.gtu %0, %1, 0;" : "=h"(mask) : "h"(x));
    return mask & x;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint16_t float_to_half(float f)
{
    uint16_t h;
    asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ bf16_t float_to_bf16(float f)
{
    return __float2bfloat16(f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t float2_to_half2(float a, float b)
{
    uint32_t c;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(c) : "f"(b), "f"(a));
#else
    uint16_t lo = float_to_half(a);
    uint16_t hi = float_to_half(b);
    asm volatile("mov.b32 %0, {%1, %2};\n" : "=r"(c) : "h"(lo), "h"(hi));
#endif
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t float2_to_bf16_x2(float a, float b)
{
    uint32_t c;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(c) : "f"(b), "f"(a));
#else
    uint16_t* px = reinterpret_cast<uint16_t*>(&a);
    uint16_t* py = reinterpret_cast<uint16_t*>(&b);
    uint16_t value = px[1];
    uint16_t value2 = py[1];

    if (px[0] == 0x8000)
    {
        if ((value & 0x1) == 1)
            value++;
    }
    else if (px[0] > 0x8000)
    {
        value++;
    }

    if (py[0] == 0x8000)
    {
        if ((value2 & 0x1) == 1)
            value2++;
    }
    else if (py[0] > 0x8000)
    {
        value2++;
    }

    uint32_t high = reinterpret_cast<uint32_t&>(value2);
    c = (high << 16) | value;
#endif
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Template function to support both half and bfloat16
template <typename Data_type>
inline __device__ uint32_t float2_to_16bit_2(float a, float b)
{
    return float2_to_half2(a, b);
}

template <>
inline __device__ uint32_t float2_to_16bit_2<bf16_t>(float a, float b)
{
    return float2_to_bf16_x2(a, b);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t float_to_half2(float a)
{
    return float2_to_half2(a, a);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t float2_to_half2(float2 const& f)
{
    return float2_to_half2(f.x, f.y);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t float_to_bf16_2(float a)
{
    return float2_to_bf16_x2(a, a);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint2 float4_to_half4(float x, float y, float z, float w)
{
    uint2 d;
    d.x = float2_to_half2(x, y);
    d.y = float2_to_half2(z, w);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Template function to support both half and bfloat16
template <typename Data_type>
inline __device__ uint2 float4_to_16bit_x4(float x, float y, float z, float w)
{
    uint2 d;
    d.x = float2_to_half2(x, y);
    d.y = float2_to_half2(z, w);
    return d;
}

template <>
inline __device__ uint2 float4_to_16bit_x4<bf16_t>(float x, float y, float z, float w)
{
    uint2 d;
    d.x = float2_to_bf16_x2(x, y);
    d.y = float2_to_bf16_x2(z, w);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t hfma2(uint32_t a, uint32_t b, uint32_t c)
{
    uint32_t d;
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t hfma2_relu(uint32_t a, uint32_t b, uint32_t c)
{
    uint32_t d;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("fma.rn.f16x2.relu %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
#else
    d = hrelu2(hfma2(a, b, c));
#endif
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t h0_h0(uint32_t x)
{
    uint32_t y;
    asm volatile("{.reg .f16 lo, hi; mov.b32 {lo, hi}, %1; mov.b32 %0, {lo, lo};}\n" : "=r"(y) : "r"(x));
    return y;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ float h0_to_float(uint32_t h2)
{
    float f;
    asm volatile(
        "{\n"
        ".reg .f16 lo, hi;\n"
        "mov.b32 {lo, hi}, %1;\n"
        "cvt.f32.f16 %0, lo;\n"
        "}\n"
        : "=f"(f)
        : "r"(h2));
    return f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t h1_h1(uint32_t x)
{
    uint32_t y;
    asm volatile("{.reg .f16 lo, hi; mov.b32 {lo, hi}, %1; mov.b32 %0, {hi, hi};}\n" : "=r"(y) : "r"(x));
    return y;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint16_t hadd(uint16_t a, uint16_t b)
{
    uint16_t d;
    asm volatile("add.f16 %0, %1, %2;" : "=h"(d) : "h"(a), "h"(b));
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t hadd(uint32_t a, uint32_t b)
{
    return hadd2(a, b);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint2 hadd4(uint2 a, uint2 b)
{
    uint2 c;
    c.x = hadd2(a.x, b.x);
    c.y = hadd2(a.y, b.y);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint2 hadd(uint2 a, uint2 b)
{
    return hadd4(a, b);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint4 hadd8(uint4 a, uint4 b)
{
    uint4 c;
    c.x = hadd2(a.x, b.x);
    c.y = hadd2(a.y, b.y);
    c.z = hadd2(a.z, b.z);
    c.w = hadd2(a.w, b.w);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Template function to support both half and bfloat16
template <typename Data_type>
inline __device__ uint4 add8(uint4 a, uint4 b)
{
    return hadd8(a, b);
}

template <>
inline __device__ uint4 add8<bf16_t>(uint4 a, uint4 b)
{
    uint4 c;
    c.x = bfadd2(a.x, b.x);
    c.y = bfadd2(a.y, b.y);
    c.z = bfadd2(a.z, b.z);
    c.w = bfadd2(a.w, b.w);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint4 fadd4(uint4 a, uint4 b)
{
    float4 c;
    c.x = reinterpret_cast<float const&>(a.x) + reinterpret_cast<float const&>(b.x);
    c.y = reinterpret_cast<float const&>(a.y) + reinterpret_cast<float const&>(b.y);
    c.z = reinterpret_cast<float const&>(a.z) + reinterpret_cast<float const&>(b.z);
    c.w = reinterpret_cast<float const&>(a.w) + reinterpret_cast<float const&>(b.w);
    return reinterpret_cast<uint4 const&>(c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint4 hadd(uint4 a, uint4 b)
{
    return hadd8(a, b);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ float half_to_float(uint16_t h)
{
    float f;
    asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(f) : "h"(h));
    return f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ float bf16_to_float(uint16_t h)
{
    float f;
    asm volatile("mov.b32 %0, {0, %1};\n" : "=f"(f) : "h"(h));
    return f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ float2 half2_to_float2(uint32_t x)
{
    uint16_t lo, hi;
    asm volatile("mov.b32 {%0, %1}, %2;\n" : "=h"(lo), "=h"(hi) : "r"(x));
    return make_float2(half_to_float(lo), half_to_float(hi));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ float2 bf16_2_to_float2(uint32_t x)
{
    float2 res;
    asm volatile(
        "{\n"
        "    .reg .b16 lo, hi;\n"
        "    mov.b32 {lo, hi}, %2;\n"
        "    mov.b32 %0, {0, lo};\n"
        "    mov.b32 %1, {0, hi};\n"
        "}\n"
        : "=f"(res.x), "=f"(res.y)
        : "r"(x));
    return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Template function to support both half and bfloat16
template <typename Data_type>
inline __device__ float2 convert_from_16bit_2(uint32_t x)
{
    return half2_to_float2(x);
}

template <>
inline __device__ float2 convert_from_16bit_2<bf16_t>(uint32_t x)
{
    return bf16_2_to_float2(x);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ void half2_to_float2(float& x, float& y, uint32_t h)
{
    float2 tmp = half2_to_float2(h);
    x = tmp.x;
    y = tmp.y;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint16_t hfma(uint16_t a, uint16_t b, uint16_t c)
{
    uint16_t d;
    asm volatile("fma.rn.f16 %0, %1, %2, %3;" : "=h"(d) : "h"(a), "h"(b), "h"(c));
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint16_t hmul(uint16_t a, uint16_t b)
{
    uint16_t d;
    asm volatile("mul.f16 %0, %1, %2;" : "=h"(d) : "h"(a), "h"(b));
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Converted two half2's or bf162's into float, then take their dot product.
template <typename Data_type>
inline __device__ float fma2_in_float(uint32_t const a, uint32_t const b)
{
    float2 af = fmha::convert_from_16bit_2<Data_type>(a);
    float2 bf = fmha::convert_from_16bit_2<Data_type>(b);
    return af.x * bf.x + af.y * bf.y;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Converted two vectors of 8 half's or bf16's into float, then take their dot product.
template <typename Data_type>
inline __device__ float fma8_in_float(uint4 const a, uint4 const b)
{
    float sum;
    sum = fmha::fma2_in_float<Data_type>(a.x, b.x);
    sum += fmha::fma2_in_float<Data_type>(a.y, b.y);
    sum += fmha::fma2_in_float<Data_type>(a.z, b.z);
    sum += fmha::fma2_in_float<Data_type>(a.w, b.w);
    return sum;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ float sigmoid(float x)
{
    return 1.f / (1.f + expf(-x));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void clear(uint16_t& dst)
{
    dst = uint16_t(0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void clear(uint32_t& dst)
{
    dst = 0u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void clear(uint2& dst)
{
    dst = make_uint2(0u, 0u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void clear(uint4& dst)
{
    dst = make_uint4(0u, 0u, 0u, 0u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// P R E D I C A T E   P A C K I N G
//
////////////////////////////////////////////////////////////////////////////////////////////////////

enum
{
    BYTES_PER_REG = 4,
    PREDS_PER_BYTE = 4,
    PREDS_PER_REG = BYTES_PER_REG * PREDS_PER_BYTE
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int LDGS>
struct Compute_number_of_pred_regs
{
    enum
    {
        VALUE = Div_up<LDGS, PREDS_PER_REG>::VALUE
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int M, int N>
inline __device__ void pack_predicates(uint32_t (&preds)[M], uint32_t const (&p)[N])
{

    // Make sure the values match.
    static_assert(Compute_number_of_pred_regs<N>::VALUE == M, "");

    // The number of complete steps (where we use all the predicates in a byte).
    enum
    {
        COMPLETE_BYTES = N / PREDS_PER_BYTE
    };

    // Make sure we allocated enough predicate registers.
    static_assert(Div_up<COMPLETE_BYTES, BYTES_PER_REG>::VALUE <= M, "");

    // The remainder.
    enum
    {
        REMAINDER = N - COMPLETE_BYTES * PREDS_PER_BYTE
    };

    // Make sure we got the math right and the remainder is between 0 and 3.
    static_assert(REMAINDER >= 0 && REMAINDER <= 3, "");

    // The mask to extract the predicates.
    enum
    {
        COMPLETE_MASK = (1 << PREDS_PER_BYTE) - 1
    };

    // Run complete steps.
#pragma unroll
    for (int ii = 0; ii < M; ++ii)
    {

        // The number of complete bytes for that register. Be careful it can be > than 4 ;)
        int const COMPLETE = (N - ii * PREDS_PER_REG) / PREDS_PER_BYTE;

        // Pack the predicates in a register.
        uint32_t reg = 0u;
#pragma unroll
        for (int jj = 0; jj < 4; ++jj)
        {

            // Early exit.
            if (jj >= COMPLETE)
            {
                break;
            }

            // Prepare the array of predicates.
            bool tmp[PREDS_PER_BYTE];
#pragma unroll
            for (int kk = 0; kk < PREDS_PER_BYTE; ++kk)
            {
                tmp[kk] = p[ii * PREDS_PER_REG + jj * PREDS_PER_BYTE + kk] != 0;
            }

            // Store the predicates.
#pragma unroll
            for (int kk = 0; kk < PREDS_PER_BYTE; ++kk)
            {
                if (tmp[kk])
                {
                    reg |= 1u << (jj * 8 + kk);
                }
            }
        }

        // Skip the rest of the code if we do not have a remainder.
        if (COMPLETE < 4 && REMAINDER > 0)
        {

            // The mask to extract the predicates.
            enum
            {
                REMAINDER_MASK = (1 << REMAINDER) - 1
            };

            // Prepare the array of predicates.
            bool tmp[PREDS_PER_BYTE];
#pragma unroll
            for (int jj = 0; jj < REMAINDER; ++jj)
            {
                tmp[jj] = p[COMPLETE_BYTES * PREDS_PER_BYTE + jj] != 0;
            }

            // Store the predicates.
#pragma unroll
            for (int jj = 0; jj < REMAINDER; ++jj)
            {
                if (tmp[jj])
                {
                    reg |= 1u << (COMPLETE * 8 + jj);
                }
            }
        }

        // Store the predicate register.
        preds[ii] = reg;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N>
inline __device__ uint32_t pack_predicates(uint32_t const (&p)[N])
{
    uint32_t tmp[1];
    pack_predicates(tmp, p);
    return tmp[0];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// G E N E R I C   P R E D I C A T E D   L D G S T S
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N, int M, typename Functor>
inline __device__ void ldgsts_(Functor& fct, uint32_t const (&preds)[M])
{

    // The number of complete bytes (where we use all the predicates in a byte).
    enum
    {
        COMPLETE = N / PREDS_PER_BYTE
    };

    // Make sure we did allocate enough predicates.
    static_assert(Div_up<COMPLETE, BYTES_PER_REG>::VALUE <= M, "");

    // The remainder.
    enum
    {
        REMAINDER = N - COMPLETE * PREDS_PER_BYTE
    };

    // Make sure we got the math right and the remainder is between 0 and 3.
    static_assert(REMAINDER >= 0 && REMAINDER <= 3, "");

    // The mask to extract the predicates.
    enum
    {
        COMPLETE_MASK = (1 << PREDS_PER_BYTE) - 1
    };

// Clear the fetch registers.
#pragma unroll
    for (int ii = 0; ii < N; ++ii)
    {
        fct.clear(ii);
    }

    // Run complete steps.
    bool p[PREDS_PER_BYTE];
#pragma unroll
    for (int ii = 0; ii < COMPLETE; ++ii)
    {

        // The predicate.
        uint32_t reg = preds[ii / BYTES_PER_REG];

        // Extract the predicates.
#pragma unroll
        for (int jj = 0; jj < PREDS_PER_BYTE; ++jj)
        {
            uint32_t mask = 1u << (ii % BYTES_PER_REG * 8 + jj);
            p[jj] = (reg & mask) != 0u;
        }

// Issue the loads.
#pragma unroll
        for (int jj = 0; jj < PREDS_PER_BYTE; ++jj)
        {
            fct.ldgsts(ii * PREDS_PER_BYTE + jj, p[jj]);
        }
    }

    // Skip the rest of the code if we do not have a remainder.
    if (REMAINDER > 0)
    {

        // The mask to extract the predicates.
        enum
        {
            REMAINDER_MASK = (1 << REMAINDER) - 1
        };

        // The predicate register.
        uint32_t reg = preds[COMPLETE / BYTES_PER_REG];

        // Extract the predicates.
#pragma unroll
        for (int jj = 0; jj < PREDS_PER_BYTE; ++jj)
        {
            uint32_t mask = 1u << (COMPLETE % BYTES_PER_REG * 8 + jj);
            p[jj] = (reg & mask) != 0u;
        }

// Issue the loads.
#pragma unroll
        for (int ii = 0; ii < REMAINDER; ++ii)
        {
            fct.ldgsts(COMPLETE * PREDS_PER_BYTE + ii, p[ii]);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int M, typename Functor>
inline __device__ void ldgsts_(Functor& fct, uint32_t preds)
{
    uint32_t tmp[1] = {preds};
    ldgsts_<M>(fct, tmp);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// L D G
//
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldg(uint8_t& dst, void const* ptr)
{
    dst = *reinterpret_cast<uint8_t const*>(ptr);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldg(uint16_t& dst, void const* ptr)
{
    dst = *reinterpret_cast<uint16_t const*>(ptr);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldg(uint32_t& dst, void const* ptr)
{
    dst = *reinterpret_cast<uint32_t const*>(ptr);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldg(uint2& dst, void const* ptr)
{
    dst = *reinterpret_cast<uint2 const*>(ptr);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldg(uint4& dst, void const* ptr)
{
    dst = *reinterpret_cast<uint4 const*>(ptr);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Data_type, int N>
struct Ldg_functor
{
    // Ctor.
    inline __device__ Ldg_functor(Data_type (&fetch)[N], void const* (&ptrs)[N])
        : fetch_(fetch)
        , ptrs_(ptrs)
    {
    }

    // Clear the element.
    inline __device__ void clear(int ii)
    {
        fmha::clear(fetch_[ii]);
    }

    // Trigger the loads.
    inline __device__ void ldgsts(int ii, bool p)
    {
        if (p)
        {
            ldg(fetch_[ii], ptrs_[ii]);
        }
    }

    // The fetch registers.
    Data_type (&fetch_)[N];
    // The pointers.
    void const* (&ptrs_)[N];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Data_type, int N, int M>
inline __device__ void ldg_(Data_type (&fetch)[N], void const* (&ptrs)[N], uint32_t (&preds)[M])
{
    Ldg_functor<Data_type, N> fct(fetch, ptrs);
    ldgsts_<N>(fct, preds);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N, int M>
inline __device__ void ldg(uint8_t (&fetch)[N], void const* (&ptrs)[N], uint32_t (&preds)[M])
{
    ldg_<uint8_t, N>(fetch, ptrs, preds);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N, int M>
inline __device__ void ldg(uint16_t (&fetch)[N], void const* (&ptrs)[N], uint32_t (&preds)[M])
{
    ldg_<uint16_t, N>(fetch, ptrs, preds);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N, int M>
inline __device__ void ldg(uint32_t (&fetch)[N], void const* (&ptrs)[N], uint32_t (&preds)[M])
{
    ldg_<uint32_t, N>(fetch, ptrs, preds);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N, int M>
inline __device__ void ldg(uint2 (&fetch)[N], void const* (&ptrs)[N], uint32_t (&preds)[M])
{
    ldg_<uint2, N>(fetch, ptrs, preds);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N, int M>
inline __device__ void ldg(uint4 (&fetch)[N], void const* (&ptrs)[N], uint32_t (&preds)[M])
{
    ldg_<uint4, N>(fetch, ptrs, preds);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool USE_LDGSTS>
inline __device__ void ldgdepbar()
{
    if (USE_LDGSTS)
    {
        asm volatile("cp.async.commit_group;\n" ::);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool USE_LDGSTS, int COUNT = 0>
inline __device__ void depbar_()
{
    if (USE_LDGSTS)
    {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(COUNT));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool USE_LDGSTS, int STAGES>
inline __device__ void depbar()
{
    if (USE_LDGSTS)
    {
        int const VALUE = Max<STAGES - 2, 0>::VALUE;
        asm volatile("cp.async.wait_group %0;\n" ::"n"(VALUE));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldgsts128(uint32_t dst, void const* src, bool p = true)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint32_t m = p ? 16u : 0u;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(m));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N>
struct Ldgsts_functor
{
    // Ctor.
    inline __device__ Ldgsts_functor(uint32_t (&smem_ptrs)[N], void const* (&gmem_ptrs)[N])
        : smem_ptrs_(smem_ptrs)
        , gmem_ptrs_(gmem_ptrs)
    {
    }

    // Does nothing.
    inline __device__ void clear(int ii) {}

    // Trigger the load-store instruction.
    inline __device__ void ldgsts(int ii, bool p)
    {
        ldgsts128(smem_ptrs_[ii], gmem_ptrs_[ii], p);
    }

    // The shared memory pointers.
    uint32_t (&smem_ptrs_)[N];
    // The global memory pointers.
    void const* (&gmem_ptrs_)[N];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N, int M>
inline __device__ void ldgsts(uint32_t (&dst)[N], void const* (&src)[N], uint32_t (&preds)[M])
{
    Ldgsts_functor<N> fct(dst, src);
    ldgsts_<N>(fct, preds);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// L D S
//
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void lds(uint16_t& dst, uint32_t ptr)
{
    asm volatile("ld.shared.b16 %0, [%1];\n" : "=h"(dst) : "r"(ptr));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void lds(uint32_t& dst, uint32_t ptr)
{
    asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(dst) : "r"(ptr));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void lds(uint2& dst, uint32_t ptr)
{
    asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];\n" : "=r"(dst.x), "=r"(dst.y) : "r"(ptr));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void lds(uint4& dst, uint32_t ptr)
{
    asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(dst.x), "=r"(dst.y), "=r"(dst.z), "=r"(dst.w)
                 : "r"(ptr));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// L D S M
//
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldsm(uint32_t& dst, uint32_t ptr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 730
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n" : "=r"(dst) : "r"(ptr));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldsmt(uint32_t& dst, uint32_t ptr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 730
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n" : "=r"(dst) : "r"(ptr));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldsm(uint2& dst, uint32_t ptr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 730
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n" : "=r"(dst.x), "=r"(dst.y) : "r"(ptr));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldsmt(uint2& dst, uint32_t ptr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 730
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(dst.x), "=r"(dst.y)
                 : "r"(ptr));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldsm(uint4& dst, uint32_t ptr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 730
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(dst.x), "=r"(dst.y), "=r"(dst.z), "=r"(dst.w)
                 : "r"(ptr));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldsmt(uint4& dst, uint32_t ptr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 730
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(dst.x), "=r"(dst.y), "=r"(dst.z), "=r"(dst.w)
                 : "r"(ptr));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// S T S M
//
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void stsm(uint32_t ptr, uint32_t const& src)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};\n" ::"r"(ptr), "r"(src));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void stsmt(uint32_t ptr, uint32_t const& src)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("stmatrix.sync.aligned.m8n8.x1.trans.shared.b16 [%0], {%1};\n" ::"r"(ptr), "r"(src));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void stsm(uint32_t ptr, uint2 const& src)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n" ::"r"(ptr), "r"(src.x), "r"(src.y));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void stsmt(uint32_t ptr, uint2 const& src)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n" ::"r"(ptr), "r"(src.x), "r"(src.y));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void stsm(uint32_t ptr, uint4 const& src)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(ptr), "r"(src.x),
        "r"(src.y), "r"(src.z), "r"(src.w));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void stsmt(uint32_t ptr, uint4 const& src)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(ptr), "r"(src.x),
        "r"(src.y), "r"(src.z), "r"(src.w));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// S T G
//
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void stg(void* ptr, float val)
{
    *reinterpret_cast<float*>(ptr) = val;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void stg(void* ptr, uint8_t val)
{
    *reinterpret_cast<uint8_t*>(ptr) = val;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void stg(void* ptr, uint16_t val)
{
    *reinterpret_cast<uint16_t*>(ptr) = val;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void stg(void* ptr, uint32_t val)
{
    *reinterpret_cast<uint32_t*>(ptr) = val;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void stg(void* ptr, uint2 val)
{
    *reinterpret_cast<uint2*>(ptr) = val;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void stg(void* ptr, uint4 val)
{
    *reinterpret_cast<uint4*>(ptr) = val;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// S T S
//
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void sts(uint32_t ptr, uint16_t val)
{
    asm volatile("st.shared.b16 [%0], %1;\n" : : "r"(ptr), "h"(val));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void sts(uint32_t ptr, uint32_t val)
{
    asm volatile("st.shared.b32 [%0], %1;\n" : : "r"(ptr), "r"(val));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void sts(uint32_t ptr, uint2 val)
{
    asm volatile("st.shared.v2.b32 [%0], {%1, %2};\n" : : "r"(ptr), "r"(val.x), "r"(val.y));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void sts(uint32_t ptr, uint4 val)
{
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n"
                 :
                 : "r"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Data_type, int N>
inline __device__ void sts_(uint32_t (&ptrs)[N], Data_type const (&data)[N])
{
#pragma unroll
    for (int ii = 0; ii < N; ++ii)
    {
        sts(ptrs[ii], data[ii]);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N>
inline __device__ void sts(uint32_t (&ptrs)[N], uint16_t const (&data)[N])
{
    sts_<uint16_t, N>(ptrs, data);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N>
inline __device__ void sts(uint32_t (&ptrs)[N], uint32_t const (&data)[N])
{
    sts_<uint32_t, N>(ptrs, data);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N>
inline __device__ void sts(uint32_t (&ptrs)[N], uint2 const (&data)[N])
{
    sts_<uint2, N>(ptrs, data);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N>
inline __device__ void sts(uint32_t (&ptrs)[N], uint4 const (&data)[N])
{
    sts_<uint4, N>(ptrs, data);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
#define __HALF2_TO_UI(var) *(reinterpret_cast<unsigned int*>(&(var)))
#define __HALF2_TO_CUI(var) *(reinterpret_cast<const unsigned int*>(&(var)))

static __device__ __inline__ void atomicAdd_half2(half2* const address, const half2 val)
{
    asm volatile("{ red.global.add.noftz.f16x2 [%0],%1; }\n" ::"l"(address), "r"(__HALF2_TO_CUI(val)) : "memory");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool CAN_BE_NEGATIVE>
static inline __device__ uint32_t float4_to_char4(float x, float y, float z, float w)
{
#if defined(USE_F2I_EMULATION_TRICK)
    // Make sure the float is in the proper range.
    float cx, cy, cz, cw;
    if (CAN_BE_NEGATIVE)
    {
        cx = fmha::clamp(x, -128.f, 127.f);
        cy = fmha::clamp(y, -128.f, 127.f);
        cz = fmha::clamp(z, -128.f, 127.f);
        cw = fmha::clamp(w, -128.f, 127.f);
    }
    else
    {
        cx = fminf(x, 127.f);
        cy = fminf(y, 127.f);
        cz = fminf(z, 127.f);
        cw = fminf(w, 127.f);
    }

    // Re-add the magic number.
    cx += FP32_I2F_MAGIC_NUMBER;
    cy += FP32_I2F_MAGIC_NUMBER;
    cz += FP32_I2F_MAGIC_NUMBER;
    cw += FP32_I2F_MAGIC_NUMBER;

    // We need unsigned ints...
    uint32_t a = reinterpret_cast<uint32_t const&>(cx);
    uint32_t b = reinterpret_cast<uint32_t const&>(cy);
    uint32_t c = reinterpret_cast<uint32_t const&>(cz);
    uint32_t d = reinterpret_cast<uint32_t const&>(cw);

    // Pack the numbers.
    uint32_t dst;
    asm volatile("prmt.b32 %0, %1, %2, 0x0040;\n" : "=r"(dst) : "r"(a), "r"(b));
    asm volatile("prmt.b32 %0, %0, %1, 0x0410;\n" : "+r"(dst) : "r"(c));
    asm volatile("prmt.b32 %0, %0, %1, 0x4210;\n" : "+r"(dst) : "r"(d));
    return dst;
#else
    uint32_t a;
    asm volatile("cvt.rni.sat.s32.f32 %0, %1;\n" : "=r"(a) : "f"(x));
    uint32_t b;
    asm volatile("cvt.rni.sat.s32.f32 %0, %1;\n" : "=r"(b) : "f"(y));
    uint32_t c;
    asm volatile("cvt.rni.sat.s32.f32 %0, %1;\n" : "=r"(c) : "f"(z));
    uint32_t d;
    asm volatile("cvt.rni.sat.s32.f32 %0, %1;\n" : "=r"(d) : "f"(w));

    uint32_t dst;
    asm volatile("cvt.pack.sat.s8.s32.b32 %0, %1, %2,  0;\n" : "=r"(dst) : "r"(d), "r"(c));
    asm volatile("cvt.pack.sat.s8.s32.b32 %0, %1, %2, %0;\n" : "+r"(dst) : "r"(b), "r"(a));
    return dst;
#endif // defined(USE_F2I_EMULATION_TRICK)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ void swizzle_rows(uint32_t& a, uint32_t& b, uint32_t c, uint32_t d)
{
    asm volatile("prmt.b32 %0, %1, %2, 0x6420;\n" : "=r"(a) : "r"(c), "r"(d));
    asm volatile("prmt.b32 %0, %1, %2, 0x7531;\n" : "=r"(b) : "r"(c), "r"(d));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldsm_with_lds(uint2& data, uint32_t smem)
{
    int lane = threadIdx.x % 32;
    data = {0, 0};
    uint4 v = {0, 0, 0, 0};
    uint32_t* a = reinterpret_cast<uint32_t*>(&v);
    if (lane < 16)
    {
        fmha::lds(v, smem);
    }
    int src_row = lane / 4;
    int src_col = lane % 4;
    for (int it = 0; it < 4; it++)
    {
        uint32_t val = a[it];
        uint32_t x = __shfl_sync(uint32_t(-1), val, src_row);
        __syncwarp();
        uint32_t y = __shfl_sync(uint32_t(-1), val, src_row + 8);
        __syncwarp();
        if (it == src_col)
        {
            data.x = x;
            data.y = y;
        }
    }
}

inline __device__ void ldsmt_with_lds(uint2& data, uint32_t smem)
{
    int lane = threadIdx.x % 32;

    uint4 tmp16{0, 0, 0, 0}; // 16B

    if (lane < 16)
    {
        fmha::lds(tmp16, smem);
    }

    uint16_t* tmp16c = reinterpret_cast<uint16_t*>(&tmp16); // 8x2B: we move pairs

    uint16_t* t = reinterpret_cast<uint16_t*>(&data);       // 4x2B

    int const src_col = lane / 4;                           // 0 - 7
    int const src_row = (lane % 4) * 2;

// we have to shuffle the values to distribute them in the warp
#pragma unroll
    for (int it = 0; it < 8; it++)
    {
        uint16_t val, x, y;
        val = tmp16c[it];
        x = __shfl_sync(uint32_t(-1), val, src_row + 0);
        __syncwarp();
        y = __shfl_sync(uint32_t(-1), val, src_row + 1);
        __syncwarp();

        if (src_col == it)
        {
            t[0] = x;
            t[1] = y;
        }
        val = tmp16c[it];
        x = __shfl_sync(uint32_t(-1), val, src_row + 8);
        __syncwarp();
        y = __shfl_sync(uint32_t(-1), val, src_row + 9);
        __syncwarp();

        if (src_col == it)
        {
            t[2] = x;
            t[3] = y;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct MaxOp
{
    __device__ inline T operator()(T const& x, T const& y)
    {
        return x > y ? x : y;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct SumOp
{
    __device__ inline T operator()(T const& x, T const& y)
    {
        return x + y;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int THREADS>
struct Allreduce
{
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);

    template <typename T, typename Operator>
    static __device__ inline T run(T x, Operator& op)
    {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Allreduce<2>
{
    template <typename T, typename Operator>
    static __device__ inline T run(T x, Operator& op)
    {
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
        return x;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Operator, int M>
__device__ inline void quad_reduce(float (&dst)[M], float (&src)[M], Operator& op)
{
#pragma unroll
    for (int mi = 0; mi < M; mi++)
    {
        dst[mi] = src[mi];
        dst[mi] = op(dst[mi], __shfl_down_sync(uint32_t(-1), dst[mi], 2));
        dst[mi] = op(dst[mi], __shfl_down_sync(uint32_t(-1), dst[mi], 1));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Operator, int M>
__device__ inline void quad_reduce(float (&dst)[M], float2 (&src)[M], Operator& op)
{
    float tmp[M];
#pragma unroll
    for (int mi = 0; mi < M; mi++)
    {
        tmp[mi] = op(src[mi].x, src[mi].y);
    }
    quad_reduce(dst, tmp, op);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Operator, int M>
__device__ inline void quad_allreduce(float (&dst)[M], float (&src)[M], Operator& op)
{
#pragma unroll
    for (int mi = 0; mi < M; mi++)
    {
        dst[mi] = src[mi];
        dst[mi] = Allreduce<4>::run(dst[mi], op);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Operator, int M>
__device__ inline void quad_allreduce(float (&dst)[M], float2 (&src)[M], Operator& op)
{
    float tmp[M];
#pragma unroll
    for (int mi = 0; mi < M; mi++)
    {
        tmp[mi] = op(src[mi].x, src[mi].y);
    }
    quad_allreduce(dst, tmp, op);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t elect_one_sync()
{
    uint32_t pred = 0;
#if __CUDA_ARCH__ >= 900
#if !defined(__CUDACC_RTC__)
    uint32_t laneid = 0;
    asm volatile(
        "\n\
    {\n\
        .reg .b32 %rx;\n\
        .reg .pred %px;\n\
        elect.one.sync %rx|%px, %2;\n\
        @%px mov.s32 %1, 1;\n\
        mov.s32 %0, %rx;\n\
    }\n"
        : "+r"(laneid), "+r"(pred)
        : "r"(0xFFFFFFFF));
#else
    pred = threadIdx.x == 0;
#endif
#endif
    return pred;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint16_t float2_to_e4m3x2(float x, float y)
{
#if defined(__CUDA_ARCH__) && ((__CUDA_ARCH__ == 890 && defined(FMHA_ENABLE_SM89_QMMA)) || (__CUDA_ARCH__ >= 900))
    uint16_t res;
    asm volatile("cvt.rn.e4m3x2.f32.satfinite %0, %2, %1;" : "=h"(res) : "f"(x), "f"(y));
    return res;
#else
    assert(false);
    return 0;
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t float4_to_e4m3x4(float x, float y, float z, float w)
{
#if defined(__CUDA_ARCH__) && ((__CUDA_ARCH__ == 890 && defined(FMHA_ENABLE_SM89_QMMA)) || (__CUDA_ARCH__ >= 900))
    uint32_t res;
    asm volatile(
        "{\n"
        ".reg .b16 lo;\n"
        ".reg .b16 hi;\n"
        "cvt.rn.e4m3x2.f32.satfinite   lo, %2, %1;\n"
        "cvt.rn.e4m3x2.f32.satfinite   hi, %4, %3;\n"
        "mov.b32 %0, {lo, hi};\n"
        "}"
        : "=r"(res)
        : "f"(x), "f"(y), "f"(z), "f"(w));
    return res;
#else
    assert(false);
    return 0;
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t float4_to_e5m2x4(float x, float y, float z, float w)
{
#if defined(__CUDA_ARCH__) && ((__CUDA_ARCH__ == 890 && defined(FMHA_ENABLE_SM89_QMMA)) || (__CUDA_ARCH__ >= 900))
    uint32_t res;
    asm volatile(
        "{\n"
        ".reg .b16 lo;\n"
        ".reg .b16 hi;\n"
        "cvt.rn.e5m2x2.f32.satfinite   lo, %2, %1;\n"
        "cvt.rn.e5m2x2.f32.satfinite   hi, %4, %3;\n"
        "mov.b32 %0, {lo, hi};\n"
        "}"
        : "=r"(res)
        : "f"(x), "f"(y), "f"(z), "f"(w));
    return res;
#else
    assert(false);
    return 0;
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t half4_to_e4m3x4(uint32_t const h2_0, uint32_t const h2_1)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890))
    uint32_t res;
    asm volatile(
        "{\n"
        ".reg .b16 lo, hi;\n"
        "cvt.satfinite.rn.e4m3x2.f16x2 lo, %1;\n"
        "cvt.satfinite.rn.e4m3x2.f16x2 hi, %2;\n"
        "mov.b32 %0, {lo, hi};\n"
        "}\n"
        : "=r"(res)
        : "r"(h2_0), "r"(h2_1));
    return res;
#else
    assert(false);
    return 0;
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t half4_to_e5m2x4(uint32_t const h2_0, uint32_t const h2_1)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890))
    uint32_t res;
    asm volatile(
        "{\n"
        ".reg .b16 lo, hi;\n"
        "cvt.satfinite.rn.e5m2x2.f16x2 lo, %1;\n"
        "cvt.satfinite.rn.e5m2x2.f16x2 hi, %2;\n"
        "mov.b32 %0, {lo, hi};\n"
        "}\n"
        : "=r"(res)
        : "r"(h2_0), "r"(h2_1));
    return res;
#else
    assert(false);
    return 0;
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helpers to pack float4 into a destination register with 4 8bit values
template <typename Dst_type>
inline __device__ uint32_t float4_to_8bitx4(float const x, float const y, float const z, float const w)
{
    return float4_to_char4<false>(x, y, z, w);
};

template <>
inline __device__ uint32_t float4_to_8bitx4<e4m3_t>(float const x, float const y, float const z, float const w)
{
    return float4_to_e4m3x4(x, y, z, w);
};

template <>
inline __device__ uint32_t float4_to_8bitx4<e5m2_t>(float const x, float const y, float const z, float const w)
{
    return float4_to_e5m2x4(x, y, z, w);
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline __device__ uint32_t half4_to_fp8x4(uint32_t const h2_0, uint32_t const h2_1);

template <>
inline __device__ uint32_t half4_to_fp8x4<fmha::e4m3_t>(uint32_t const h2_0, uint32_t const h2_1)
{
    return half4_to_e4m3x4(h2_0, h2_1);
}

template <>
inline __device__ uint32_t half4_to_fp8x4<fmha::e5m2_t>(uint32_t const h2_0, uint32_t const h2_1)
{
    return half4_to_e5m2x4(h2_0, h2_1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline __device__ uint32_t float4_to_fp8x4(float const, float const, float const, float const);

template <>
inline __device__ uint32_t float4_to_fp8x4<fmha::e4m3_t>(float const x, float const y, float const z, float const w)
{
    return float4_to_e4m3x4(x, y, z, w);
}

template <>
inline __device__ uint32_t float4_to_fp8x4<fmha::e5m2_t>(float const x, float const y, float const z, float const w)
{
    return float4_to_e5m2x4(x, y, z, w);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void fence_view_async_shared()
{

    // Issue a shared memory fence for async operations (FENCE.VIEW.ASYNC.S)
    // only compiles on sm90+

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("fence.proxy.async.shared::cta;\n");
#else
    assert(false);
#endif
}

inline __device__ void fence_view_async_global()
{

    // Issue a global memory fence for async operations (FENCE.VIEW.ASYNC.G)
    // only compiles on sm90+

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("fence.proxy.async.global::cta;\n");
#else
    assert(false);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ char* align_1024(char* ptr)
{
    uint64_t address_bit = reinterpret_cast<uint64_t>(ptr);
    uint64_t offset = address_bit % 1024;
    if (offset == 0)
    {
        return ptr;
    }
    else
    {
        return ptr + (1024 - offset);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ float atomicMaxFloat(float* addr, float value)
{
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int*) addr, __float_as_int(value)))
                       : __uint_as_float(atomicMin((unsigned int*) addr, __float_as_uint(value)));
    return old;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ float atomicMaxFloatPos_(float* addr, float value)
{
    // VALUE MUST BE POSITIVE! USED ONLY FOR INTERNAL AMAX REDUCTION.
    float old = __int_as_float(atomicMax((int*) addr, __float_as_int(value)));
    return old;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ float max3Pos_(float const a, float const b, float const c)
{
    // VALUE MUST BE POSITIVE! USED ONLY FOR INTERNAL AMAX REDUCTION.
    float res;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    int32_t a_ = reinterpret_cast<int32_t const&>(a);
    int32_t b_ = reinterpret_cast<int32_t const&>(b);
    int32_t c_ = reinterpret_cast<int32_t const&>(c);
    int32_t tmp;
    asm volatile("max.s16x2 %0, %1, %2;\n" : "=r"(tmp) : "r"(a_), "r"(b_));
    asm volatile("max.s16x2 %0, %0, %1;\n" : "+r"(tmp) : "r"(tmp), "r"(c_));
    res = reinterpret_cast<float const&>(tmp);
#else
    res = fmaxf(a, fmaxf(b, c));
#endif
    return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Fast approximate tanh.
static inline __device__ float __tanhf(float x)
{
#if (__CUDA_ARCH__ >= 750)
    float r = x;
    asm("tanh.approx.f32 %0, %0;" : "+f"(r));
    return r;
#else
    return tanhf(x);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha
