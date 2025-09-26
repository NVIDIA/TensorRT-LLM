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
#include <fmha/hopper/fragment.h>

namespace fmha
{

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, int BITS_PER_ELEMENT>
struct Store_accumulator
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits>
struct Store_accumulator<Traits, 16>
{

    // The fragment.
    using Acc = fmha::Fragment_accumulator<Traits>;

    // Store.
    inline __device__ void store(char* ptr, int64_t step_m, int64_t step_n, Acc const& acc, uint32_t scale)
    {
        uint32_t acc_0 = fmha::hmul2(acc.reg(0), scale);
        uint32_t acc_1 = fmha::hmul2(acc.reg(1), scale);
        uint32_t acc_2 = fmha::hmul2(acc.reg(2), scale);
        uint32_t acc_3 = fmha::hmul2(acc.reg(3), scale);

        fmha::stg(ptr + 0 * step_m + 0 * step_n, acc_0);
        fmha::stg(ptr + 1 * step_m + 0 * step_n, acc_1);
        fmha::stg(ptr + 0 * step_m + 1 * step_n, acc_2);
        fmha::stg(ptr + 1 * step_m + 1 * step_n, acc_3);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits>
struct Store_accumulator<Traits, 32>
{

    // The fragment.
    using Accumulator = fmha::Fragment_accumulator<Traits>;

    // Store.
    inline __device__ void store(char* ptr, int64_t step_m, int64_t step_n, Accumulator const& acc, uint32_t)
    {
        int32_t tmp_0 = acc.elt(0);
        int32_t tmp_1 = acc.elt(1);
        int32_t tmp_2 = acc.elt(2);
        int32_t tmp_3 = acc.elt(3);
        int32_t tmp_4 = acc.elt(4);
        int32_t tmp_5 = acc.elt(5);
        int32_t tmp_6 = acc.elt(6);
        int32_t tmp_7 = acc.elt(7);

#if defined(USE_I2F_EMULATION_TRICK)
        tmp_0 -= int32_t(FP32_I2F_MAGIC_NUMBER_HEX);
        tmp_1 -= int32_t(FP32_I2F_MAGIC_NUMBER_HEX);
        tmp_2 -= int32_t(FP32_I2F_MAGIC_NUMBER_HEX);
        tmp_3 -= int32_t(FP32_I2F_MAGIC_NUMBER_HEX);
        tmp_4 -= int32_t(FP32_I2F_MAGIC_NUMBER_HEX);
        tmp_5 -= int32_t(FP32_I2F_MAGIC_NUMBER_HEX);
        tmp_6 -= int32_t(FP32_I2F_MAGIC_NUMBER_HEX);
        tmp_7 -= int32_t(FP32_I2F_MAGIC_NUMBER_HEX);
#endif

        uint32_t acc_0 = reinterpret_cast<uint32_t const&>(tmp_0);
        uint32_t acc_1 = reinterpret_cast<uint32_t const&>(tmp_1);
        uint32_t acc_2 = reinterpret_cast<uint32_t const&>(tmp_2);
        uint32_t acc_3 = reinterpret_cast<uint32_t const&>(tmp_3);
        uint32_t acc_4 = reinterpret_cast<uint32_t const&>(tmp_4);
        uint32_t acc_5 = reinterpret_cast<uint32_t const&>(tmp_5);
        uint32_t acc_6 = reinterpret_cast<uint32_t const&>(tmp_6);
        uint32_t acc_7 = reinterpret_cast<uint32_t const&>(tmp_7);

        fmha::stg(ptr + 0 * step_m + 0 * step_n, make_uint2(acc_0, acc_1));
        fmha::stg(ptr + 1 * step_m + 0 * step_n, make_uint2(acc_4, acc_5));
        fmha::stg(ptr + 0 * step_m + 1 * step_n, make_uint2(acc_2, acc_3));
        fmha::stg(ptr + 1 * step_m + 1 * step_n, make_uint2(acc_6, acc_7));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Store_accumulator<Ampere_hmma_fp32_traits, 32>
{

    // The instruction traits.
    using Traits = Ampere_hmma_fp32_traits;
    // The fragment.
    using Accumulator = fmha::Fragment_accumulator<Traits>;

    // Store.
    inline __device__ void store(char* ptr, int64_t step_m, int64_t step_n, Accumulator const& acc, uint32_t scale)
    {

        float const scalef = reinterpret_cast<float const&>(scale);

        float const tmp_0 = acc.elt(0) * scalef;
        float const tmp_1 = acc.elt(1) * scalef;
        float const tmp_2 = acc.elt(2) * scalef;
        float const tmp_3 = acc.elt(3) * scalef;
        float const tmp_4 = acc.elt(4) * scalef;
        float const tmp_5 = acc.elt(5) * scalef;
        float const tmp_6 = acc.elt(6) * scalef;
        float const tmp_7 = acc.elt(7) * scalef;

        uint32_t acc_0 = reinterpret_cast<uint32_t const&>(tmp_0);
        uint32_t acc_1 = reinterpret_cast<uint32_t const&>(tmp_1);
        uint32_t acc_2 = reinterpret_cast<uint32_t const&>(tmp_2);
        uint32_t acc_3 = reinterpret_cast<uint32_t const&>(tmp_3);
        uint32_t acc_4 = reinterpret_cast<uint32_t const&>(tmp_4);
        uint32_t acc_5 = reinterpret_cast<uint32_t const&>(tmp_5);
        uint32_t acc_6 = reinterpret_cast<uint32_t const&>(tmp_6);
        uint32_t acc_7 = reinterpret_cast<uint32_t const&>(tmp_7);

        fmha::stg(ptr + 0 * step_m + 0 * step_n, make_uint2(acc_0, acc_1));
        fmha::stg(ptr + 1 * step_m + 0 * step_n, make_uint2(acc_2, acc_3));
        fmha::stg(ptr + 0 * step_m + 1 * step_n, make_uint2(acc_4, acc_5));
        fmha::stg(ptr + 1 * step_m + 1 * step_n, make_uint2(acc_6, acc_7));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Store_accumulator<Ampere_hmma_bf16_traits, 32>
{

    // The instruction traits.
    using Traits = Ampere_hmma_bf16_traits;
    // The fragment.
    using Accumulator = fmha::Fragment_accumulator<Traits>;

    // Store.
    inline __device__ void store(char* ptr, int64_t step_m, int64_t step_n, Accumulator const& acc, uint32_t scale)
    {

        float const scalef = reinterpret_cast<float const&>(scale);

        float const tmp_0 = acc.elt(0) * scalef;
        float const tmp_1 = acc.elt(1) * scalef;
        float const tmp_2 = acc.elt(2) * scalef;
        float const tmp_3 = acc.elt(3) * scalef;
        float const tmp_4 = acc.elt(4) * scalef;
        float const tmp_5 = acc.elt(5) * scalef;
        float const tmp_6 = acc.elt(6) * scalef;
        float const tmp_7 = acc.elt(7) * scalef;

        uint32_t acc_0 = reinterpret_cast<uint32_t const&>(tmp_0);
        uint32_t acc_1 = reinterpret_cast<uint32_t const&>(tmp_1);
        uint32_t acc_2 = reinterpret_cast<uint32_t const&>(tmp_2);
        uint32_t acc_3 = reinterpret_cast<uint32_t const&>(tmp_3);
        uint32_t acc_4 = reinterpret_cast<uint32_t const&>(tmp_4);
        uint32_t acc_5 = reinterpret_cast<uint32_t const&>(tmp_5);
        uint32_t acc_6 = reinterpret_cast<uint32_t const&>(tmp_6);
        uint32_t acc_7 = reinterpret_cast<uint32_t const&>(tmp_7);

        fmha::stg(ptr + 0 * step_m + 0 * step_n, make_uint2(acc_0, acc_1));
        fmha::stg(ptr + 1 * step_m + 0 * step_n, make_uint2(acc_2, acc_3));
        fmha::stg(ptr + 0 * step_m + 1 * step_n, make_uint2(acc_4, acc_5));
        fmha::stg(ptr + 1 * step_m + 1 * step_n, make_uint2(acc_6, acc_7));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint16_t pack_char2(uint32_t a, uint32_t b)
{
    uint32_t dst;
    asm volatile("prmt.b32 %0, %1, %2, 0x0040;\n" : "=r"(dst) : "r"(a), "r"(b));
    return reinterpret_cast<uint16_t&>(dst);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits>
struct Store_accumulator<Traits, 8>
{
    // The fragment.
    using Accumulator = fmha::Fragment_accumulator<Traits>;

    // Store.
    inline __device__ void store(char* ptr, int64_t step_m, int64_t step_n, Accumulator const& acc, uint32_t)
    {

        // Pack pairs of values.
        uint16_t tmp_00 = pack_char2(acc.reg(0), acc.reg(1));
        uint16_t tmp_01 = pack_char2(acc.reg(2), acc.reg(3));
        uint16_t tmp_10 = pack_char2(acc.reg(4), acc.reg(5));
        uint16_t tmp_11 = pack_char2(acc.reg(6), acc.reg(7));

        // Store to memory.
        fmha::stg(ptr + 0 * step_m + 0 * step_n, tmp_00);
        fmha::stg(ptr + 1 * step_m + 0 * step_n, tmp_10);
        fmha::stg(ptr + 0 * step_m + 1 * step_n, tmp_01);
        fmha::stg(ptr + 1 * step_m + 1 * step_n, tmp_11);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Store_accumulator<fmha::Ampere_imma_int8_int32_traits, 32>
{

    // The traits.
    using Traits = fmha::Ampere_imma_int8_int32_traits;
    // The fragment.
    using Accumulator = fmha::Fragment_accumulator<Traits>;

    // Store.
    inline __device__ void store(char* ptr, int64_t step_m, int64_t step_n, Accumulator const& acc, uint32_t)
    {

        int32_t tmp_0 = acc.elt(0);
        int32_t tmp_1 = acc.elt(1);
        int32_t tmp_2 = acc.elt(2);
        int32_t tmp_3 = acc.elt(3);
        int32_t tmp_4 = acc.elt(4);
        int32_t tmp_5 = acc.elt(5);
        int32_t tmp_6 = acc.elt(6);
        int32_t tmp_7 = acc.elt(7);

#if defined(USE_I2F_EMULATION_TRICK)
        tmp_0 -= int32_t(FP32_I2F_MAGIC_NUMBER_HEX);
        tmp_1 -= int32_t(FP32_I2F_MAGIC_NUMBER_HEX);
        tmp_2 -= int32_t(FP32_I2F_MAGIC_NUMBER_HEX);
        tmp_3 -= int32_t(FP32_I2F_MAGIC_NUMBER_HEX);
        tmp_4 -= int32_t(FP32_I2F_MAGIC_NUMBER_HEX);
        tmp_5 -= int32_t(FP32_I2F_MAGIC_NUMBER_HEX);
        tmp_6 -= int32_t(FP32_I2F_MAGIC_NUMBER_HEX);
        tmp_7 -= int32_t(FP32_I2F_MAGIC_NUMBER_HEX);
#endif

        uint32_t acc_0 = reinterpret_cast<uint32_t const&>(tmp_0);
        uint32_t acc_1 = reinterpret_cast<uint32_t const&>(tmp_1);
        uint32_t acc_2 = reinterpret_cast<uint32_t const&>(tmp_2);
        uint32_t acc_3 = reinterpret_cast<uint32_t const&>(tmp_3);
        uint32_t acc_4 = reinterpret_cast<uint32_t const&>(tmp_4);
        uint32_t acc_5 = reinterpret_cast<uint32_t const&>(tmp_5);
        uint32_t acc_6 = reinterpret_cast<uint32_t const&>(tmp_6);
        uint32_t acc_7 = reinterpret_cast<uint32_t const&>(tmp_7);

        fmha::stg(ptr + 0 * step_m + 0 * step_n, make_uint2(acc_0, acc_1));
        fmha::stg(ptr + 1 * step_m + 0 * step_n, make_uint2(acc_2, acc_3));
        fmha::stg(ptr + 0 * step_m + 1 * step_n, make_uint2(acc_4, acc_5));
        fmha::stg(ptr + 1 * step_m + 1 * step_n, make_uint2(acc_6, acc_7));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Store_accumulator<fmha::Ampere_imma_int8_int32_traits, 8>
{
    // The traits.
    using Traits = fmha::Ampere_imma_int8_int32_traits;
    // The fragment.
    using Accumulator = fmha::Fragment_accumulator<Traits>;

    // Store.
    inline __device__ void store(char* ptr, int64_t step_m, int64_t step_n, Accumulator const& acc, uint32_t)
    {

        // Pack pairs of values.
        uint16_t tmp_00 = pack_char2(acc.reg(0), acc.reg(1));
        uint16_t tmp_01 = pack_char2(acc.reg(4), acc.reg(5));
        uint16_t tmp_10 = pack_char2(acc.reg(2), acc.reg(3));
        uint16_t tmp_11 = pack_char2(acc.reg(6), acc.reg(7));

        // Store to memory.
        fmha::stg(ptr + 0 * step_m + 0 * step_n, tmp_00);
        fmha::stg(ptr + 1 * step_m + 0 * step_n, tmp_10);
        fmha::stg(ptr + 0 * step_m + 1 * step_n, tmp_01);
        fmha::stg(ptr + 1 * step_m + 1 * step_n, tmp_11);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M, int GMMA_N, int GMMA_K, bool GMMA_A_RF, bool GMMA_B_RF>
struct Store_accumulator<fmha::Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, 16>
{
    // The traits.
    using Traits = fmha::Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>;
    // The fragment.
    using Accumulator = fmha::Fragment_accumulator<Traits>;

    // The number of rows accessed by each thread.
    enum
    {
        ROWS_PER_THREAD = GMMA_M / 8 / 4
    };

    // The number of columns access by each thread.
    // Note there are 2 elements per reg.
    enum
    {
        COLUMNS_PER_THREAD = GMMA_N / 4 / 2
    };

    // The number of accumulator held by each thread, per HGMMA instruction.
    enum
    {
        ELEMENT_PER_THREAD = ROWS_PER_THREAD * COLUMNS_PER_THREAD
    };

    // Store.
    inline __device__ void store(char* ptr, int64_t step_m, int64_t step_n, Accumulator const& acc, uint32_t scale)
    {
#pragma unroll
        for (int col_idx = 0; col_idx < COLUMNS_PER_THREAD; ++col_idx)
        {
#pragma unroll
            for (int row_idx = 0; row_idx < ROWS_PER_THREAD; ++row_idx)
            {
                uint32_t acc_0 = fmha::hmul2(acc.reg(col_idx * ROWS_PER_THREAD + row_idx), scale);
                // float one = 1.f;
                // if(col_idx > 2){
                //   acc_0 = float2_to_half2(one, one);
                // }
                int64_t offset = (int64_t) row_idx * step_m + (int64_t) col_idx * step_n;
                fmha::stg(ptr + offset, acc_0);
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M, int GMMA_N, int GMMA_K, bool GMMA_A_RF, bool GMMA_B_RF, typename Input_type_A,
    typename Input_type_B, typename Output_type>
struct Store_accumulator<fmha::Hopper_qgmma_fp8_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF, Input_type_A,
                             Input_type_B, Output_type>,
    32>
{
    // The traits.
    using Traits = fmha::Hopper_qgmma_fp8_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF, Input_type_A,
        Input_type_B, Output_type>;
    // The fragment.
    using Accumulator = fmha::Fragment_accumulator<Traits>;

    // The number of rows accessed by each thread.
    enum
    {
        ROWS_PER_THREAD = GMMA_M / 8 / 4
    };

    // The number of columns access by each thread.
    // Note there are 2 elements per reg.
    enum
    {
        COLUMNS_PER_THREAD = GMMA_N / 8
    };

    // The number of accumulator held by each thread, per HGMMA instruction.
    enum
    {
        ELEMENT_PER_THREAD = ROWS_PER_THREAD * COLUMNS_PER_THREAD
    };

    // Store.
    inline __device__ void store(char* ptr, int64_t step_m, int64_t step_n, Accumulator const& acc, uint32_t scale)
    {
        float const scalef = reinterpret_cast<float const&>(scale);
#pragma unroll
        for (int col_idx = 0; col_idx < COLUMNS_PER_THREAD; ++col_idx)
        {
#pragma unroll
            for (int row_idx = 0; row_idx < ROWS_PER_THREAD; ++row_idx)
            {
                float const acc_0 = acc.elt((col_idx * ROWS_PER_THREAD + row_idx) * 2 + 0) * scalef;
                float const acc_1 = acc.elt((col_idx * ROWS_PER_THREAD + row_idx) * 2 + 1) * scalef;
                uint2 acc_;
                acc_.x = reinterpret_cast<uint32_t const&>(acc_0);
                acc_.y = reinterpret_cast<uint32_t const&>(acc_1);
                int64_t offset = (int64_t) row_idx * step_m + (int64_t) col_idx * step_n;
                fmha::stg(ptr + offset, acc_);
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M, int GMMA_N, int GMMA_K, bool GMMA_A_RF, bool GMMA_B_RF>
struct Store_accumulator<fmha::Hopper_igmma_int8_int32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, 32>
{
    // The traits.
    using Traits = fmha::Hopper_igmma_int8_int32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>;
    // The fragment.
    using Accumulator = fmha::Fragment_accumulator<Traits>;

    // The number of rows accessed by each thread.
    enum
    {
        ROWS_PER_THREAD = GMMA_M / 8 / 4
    };

    // The number of columns access by each thread.
    // Note there are 2 elements per reg.
    enum
    {
        COLUMNS_PER_THREAD = GMMA_N / 8
    };

    // The number of accumulator held by each thread, per HGMMA instruction.
    enum
    {
        ELEMENT_PER_THREAD = ROWS_PER_THREAD * COLUMNS_PER_THREAD
    };

    // Store.
    inline __device__ void store(char* ptr, int64_t step_m, int64_t step_n, Accumulator const& acc, uint32_t scale)
    {

#pragma unroll
        for (int col_idx = 0; col_idx < COLUMNS_PER_THREAD; ++col_idx)
        {
#pragma unroll
            for (int row_idx = 0; row_idx < ROWS_PER_THREAD; ++row_idx)
            {
                int32_t const acc_0 = acc.elt((col_idx * ROWS_PER_THREAD + row_idx) * 2 + 0);
                int32_t const acc_1 = acc.elt((col_idx * ROWS_PER_THREAD + row_idx) * 2 + 1);
                uint2 acc_;
                acc_.x = reinterpret_cast<uint32_t const&>(acc_0);
                acc_.y = reinterpret_cast<uint32_t const&>(acc_1);
                int64_t offset = (int64_t) row_idx * step_m + (int64_t) col_idx * step_n;
                fmha::stg(ptr + offset, acc_);
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

static __device__ inline uint16_t pack_e4m3x2(float const x, float const y)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    uint16_t storage;
    asm volatile("{cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;}\n" : "=h"(storage) : "f"(x), "f"(y));
    return storage;
#else
    assert(false);
    return 0;
#endif
}

static __device__ inline uint16_t pack_e5m2x2(float const x, float const y)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    uint16_t storage;
    asm volatile("{cvt.rn.satfinite.e5m2x2.f32 %0, %2, %1;}\n" : "=h"(storage) : "f"(x), "f"(y));
    return storage;
#else
    assert(false);
    return 0;
#endif
}

template <typename fp8_t>
__device__ inline uint16_t pack_fp8x2(float const x, float const y);

template <>
__device__ inline uint16_t pack_fp8x2<fmha::e4m3_t>(float const x, float const y)
{
    return pack_e4m3x2(x, y);
}

template <>
__device__ inline uint16_t pack_fp8x2<fmha::e5m2_t>(float const x, float const y)
{
    return pack_e5m2x2(x, y);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M, int GMMA_N, int GMMA_K, bool GMMA_A_RF, bool GMMA_B_RF, typename Input_type_A,
    typename Input_type_B, typename Output_type>
struct Store_accumulator<fmha::Hopper_qgmma_fp8_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF, Input_type_A,
                             Input_type_B, Output_type>,
    8>
{
    // The traits.
    using Traits = fmha::Hopper_qgmma_fp8_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF, Input_type_A,
        Input_type_B, Output_type>;
    // The fragment.
    using Accumulator = fmha::Fragment_accumulator<Traits>;

    // The number of rows accessed by each thread.
    enum
    {
        ROWS_PER_THREAD = GMMA_M / 8 / 4
    };

    // The number of columns access by each thread.
    // Note there are 2 elements per reg.
    enum
    {
        COLUMNS_PER_THREAD = GMMA_N / 8
    };

    // The number of accumulator held by each thread, per HGMMA instruction.
    enum
    {
        ELEMENT_PER_THREAD = ROWS_PER_THREAD * COLUMNS_PER_THREAD
    };

    // Store.
    inline __device__ void store(char* ptr, int64_t step_m, int64_t step_n, Accumulator const& acc, uint32_t)
    {
#pragma unroll
        for (int col_idx = 0; col_idx < COLUMNS_PER_THREAD; ++col_idx)
        {
#pragma unroll
            for (int row_idx = 0; row_idx < ROWS_PER_THREAD; ++row_idx)
            {
                float const acc_0 = acc.elt((col_idx * ROWS_PER_THREAD + row_idx) * 2 + 0);
                float const acc_1 = acc.elt((col_idx * ROWS_PER_THREAD + row_idx) * 2 + 1);
                // uint16_t acc_ = pack_e4m3x2(acc_0, acc_1);
                uint16_t acc_ = pack_fp8x2<Input_type_A>(acc_0, acc_1);
                int64_t offset = (int64_t) row_idx * step_m + (int64_t) col_idx * step_n;
                fmha::stg(ptr + offset, acc_);
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M, int GMMA_N, int GMMA_K, bool GMMA_A_RF, bool GMMA_B_RF>
struct Store_accumulator<fmha::Hopper_igmma_int8_int32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, 8>
{
    // The traits.
    using Traits = fmha::Hopper_igmma_int8_int32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>;
    // The fragment.
    using Accumulator = fmha::Fragment_accumulator<Traits>;

    // The number of rows accessed by each thread.
    enum
    {
        ROWS_PER_THREAD = GMMA_M / 8 / 4
    };

    // The number of columns access by each thread.
    // Note there are 2 elements per reg.
    enum
    {
        COLUMNS_PER_THREAD = GMMA_N / 8
    };

    // The number of accumulator held by each thread, per HGMMA instruction.
    enum
    {
        ELEMENT_PER_THREAD = ROWS_PER_THREAD * COLUMNS_PER_THREAD
    };

    // Store.
    inline __device__ void store(char* ptr, int64_t step_m, int64_t step_n, Accumulator const& acc, uint32_t)
    {
#pragma unroll
        for (int col_idx = 0; col_idx < COLUMNS_PER_THREAD; ++col_idx)
        {
#pragma unroll
            for (int row_idx = 0; row_idx < ROWS_PER_THREAD; ++row_idx)
            {
                uint32_t const acc_0 = acc.reg((col_idx * ROWS_PER_THREAD + row_idx) * 2 + 0);
                uint32_t const acc_1 = acc.reg((col_idx * ROWS_PER_THREAD + row_idx) * 2 + 1);
                uint16_t acc_ = pack_char2(acc_0, acc_1);
                int64_t offset = (int64_t) row_idx * step_m + (int64_t) col_idx * step_n;
                fmha::stg(ptr + offset, acc_);
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, int BITS_PER_ELEMENT>
struct Gmem_tile_ps
{

    // The associated MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    // The number of elements per STG.
    enum
    {
        ELEMENTS_PER_STG = 2
    };

    // The size in bytes of each element.
    enum
    {
        BYTES_PER_ELEMENT = BITS_PER_ELEMENT / 8
    };

    // The size of each STG.
    enum
    {
        BYTES_PER_STG = ELEMENTS_PER_STG * BYTES_PER_ELEMENT
    };

    // The size of a row in bytes.
    enum
    {
        BYTES_PER_ROW = Cta_tile::N * BYTES_PER_ELEMENT
    };

    // // DEBUG.
    // static_assert(BYTES_PER_ROW == 384 || BYTES_PER_ROW == 768 || BYTES_PER_ROW == 1536, "");
    // // END OF DEBUG.

    // Ctor.
    inline __device__ Gmem_tile_ps(
        void* ptr, int64_t const params_stride_in_bytes, uint32_t const params_scale, int tidx, int cta_row_offset = 0)
        : params_stride_in_bytes_(params_stride_in_bytes)
        , params_scale_(params_scale)
        , ptr_(reinterpret_cast<char*>(ptr))
    {

        // For storing P and S, we do not take into account variable sequence length.

        // The block index for the batch.
        int const bidb = blockIdx.y;
        // The block index for the head.
        int const bidh = blockIdx.x;
        // The block index.
        int bidx = bidb * gridDim.x + bidh;

        // Decompose the position of the thread into warp/lane.
        int warp = tidx / Cta_tile::THREADS_PER_WARP;
        int lane = tidx % Cta_tile::THREADS_PER_WARP;

        // Compute the position in the sequence (within the CTA for the moment).
        int row = warp % Cta_tile::WARPS_M * Mma_tile::M_PER_MMA + lane / 4 + cta_row_offset;
        // Compute the position of the thread in the row.
        int col = warp / Cta_tile::WARPS_M * Mma_tile::N_PER_MMA + lane % 4 * ELEMENTS_PER_STG;

        // The offset of the 1st row written by the thread. We store the P matrix interleaved.
        int64_t row_offset = (int64_t) row * params_stride_in_bytes_ + bidx * BYTES_PER_ROW;
        // Finalize the pointer.
        ptr_ += row_offset + col * BYTES_PER_ELEMENT;
    }

    // Store data to memory.
    template <typename Accumulators, int M, int N>
    inline __device__ void store(Accumulators const (&acc)[M][N])
    {

        // A thread holds packet of 2 elements. In 2x2 tile per MMA.
        int64_t const step_m = 8 * params_stride_in_bytes_;
        int64_t const step_n = 8 * BYTES_PER_ELEMENT;

// Store the different accumulators.
#pragma unroll
        for (int mi = 0; mi < M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < N; ++ni)
            {
                int64_t offset = (int64_t) mi * Mma_tile::M_PER_MMA_PER_CTA * params_stride_in_bytes_
                    + ni * Mma_tile::N_PER_MMA_PER_CTA * BYTES_PER_ELEMENT;
                Store_accumulator<Traits, BITS_PER_ELEMENT> delegate;
                delegate.store(ptr_ + offset, step_m, step_n, acc[mi][ni], params_scale_);
            }
        }
    }

    // Move to the next location.
    inline __device__ void move()
    {
        ptr_ += (int64_t) Cta_tile::M * params_stride_in_bytes_;
    }

    inline __device__ void move_n()
    {
        ptr_ += (int64_t) Cta_tile::N * BYTES_PER_ELEMENT;
    }

    // The stride between rows for the QKV matrice.
    int64_t const params_stride_in_bytes_;
    // The scale to apply before storing the element.
    uint32_t const params_scale_;
    // The pointer.
    char* ptr_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Gmem_tile_ps<Volta_hmma_fp16_traits, Cta_tile, 16>
{

    // The traits class.
    using Traits = Volta_hmma_fp16_traits;
    // The associated MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    // The number of elements per STG.
    enum
    {
        ELEMENTS_PER_STG = 4
    };

    // The size in bytes of each element.
    enum
    {
        BYTES_PER_ELEMENT = 2
    };

    // The size of each STG.
    enum
    {
        BYTES_PER_STG = ELEMENTS_PER_STG * BYTES_PER_ELEMENT
    };

    // The size of a row in bytes.
    enum
    {
        BYTES_PER_ROW = Cta_tile::N * BYTES_PER_ELEMENT
    };

    // Ctor.
    inline __device__ Gmem_tile_ps(
        void* ptr, int64_t const params_stride_in_bytes, uint32_t const params_scale, int tidx, int cta_row_offset = 0)
        : params_stride_in_bytes_(params_stride_in_bytes)
        , params_scale_(params_scale)
        , ptr_(reinterpret_cast<char*>(ptr))
    {

        // For storing P and S, we do not take into account variable sequence lengths.

        // The block index for the batch.
        int const bidb = blockIdx.y;
        // The block index for the head.
        int const bidh = blockIdx.x;
        // The block index.
        int bidx = bidb * gridDim.x + bidh;

        // Decompose the position of the thread into warp/lane.
        int warp = tidx / Cta_tile::THREADS_PER_WARP;
        int lane = tidx % Cta_tile::THREADS_PER_WARP;

        // DEBUG.
        static_assert(Mma_tile::M_PER_MMA == 16 && Mma_tile::N_PER_MMA == 16, "");
        // END OF DEBUG.

        // The position of the warp.
        int warp_row = warp % Cta_tile::WARPS_M * Mma_tile::M_PER_MMA;
        int warp_col = warp / Cta_tile::WARPS_M * Mma_tile::N_PER_MMA;

        // Compute the position of the thread (within the CTA for the moment).
        int row = warp_row + (lane & 0x10) / 2 + (lane & 0x07);
        int col = warp_col + (lane & 0x08) / 2;

        // // DEBUG.
        // printf("tidx=%3d row=%3d col=%3d\n", tidx, row, col);
        // // END OF DEBUG.

        // The offset of the 1st row written by the thread. We store the P matrix interleaved.
        int64_t row_offset = (int64_t) row * params_stride_in_bytes_ + bidx * BYTES_PER_ROW + cta_row_offset;

        // Finalize the pointer.
        ptr_ += row_offset + col * BYTES_PER_ELEMENT;
    }

    // Store data to memory.
    template <typename Accumulators, int M, int N>
    inline __device__ void store(Accumulators const (&acc)[M][N])
    {

// Store the different accumulators.
#pragma unroll
        for (int mi = 0; mi < M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < N; ++ni)
            {

                // Scale the accumulators.
                uint32_t acc_0 = fmha::hmul2(acc[mi][ni].reg(0), params_scale_);
                uint32_t acc_1 = fmha::hmul2(acc[mi][ni].reg(1), params_scale_);
                uint32_t acc_2 = fmha::hmul2(acc[mi][ni].reg(2), params_scale_);
                uint32_t acc_3 = fmha::hmul2(acc[mi][ni].reg(3), params_scale_);

                // The offsets.
                int row = mi * Mma_tile::M_PER_MMA_PER_CTA;
                int col = ni * Mma_tile::N_PER_MMA_PER_CTA * BYTES_PER_ELEMENT;

                // The offset in bytes.
                int64_t offset = (int64_t) row * params_stride_in_bytes_ + col;

                // In one MMA, 16 FP16s are interleaved between threads i and i+8 in groups of 4.
                fmha::stg(&ptr_[offset + 0 * BYTES_PER_ELEMENT], make_uint2(acc_0, acc_1));
                fmha::stg(&ptr_[offset + 8 * BYTES_PER_ELEMENT], make_uint2(acc_2, acc_3));
            }
        }
    }

    // Move to the next location.
    inline __device__ void move()
    {
        ptr_ += (int64_t) Cta_tile::M * params_stride_in_bytes_;
    }

    // The stride between rows for the QKV matrice.
    int64_t const params_stride_in_bytes_;
    // The scale to apply before storing the element.
    uint32_t const params_scale_;
    // The pointer.
    char* ptr_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, int BITS_PER_ELEMENT>
struct Gmem_tile_p : public Gmem_tile_ps<Traits, Cta_tile, BITS_PER_ELEMENT>
{

    // The base class.
    using Base = Gmem_tile_ps<Traits, Cta_tile, BITS_PER_ELEMENT>;

    // Ctor.
    inline __device__ Gmem_tile_p(
        void* ptr, int64_t const params_stride_in_bytes, uint32_t const params_scale, int tidx, int cta_row_offset = 0)
        : Base(ptr, params_stride_in_bytes, params_scale, tidx, cta_row_offset)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Not super proud of this. Need to refactor.
template <typename Traits, typename Cta_tile, int BITS_PER_ELEMENT>
struct Gmem_tile_ps_hopper
{
    // The associated MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    // The number of elements per STG.
    enum
    {
        ELEMENTS_PER_STG = 2
    };

    // The size in bytes of each element.
    enum
    {
        BYTES_PER_ELEMENT = BITS_PER_ELEMENT / 8
    };

    // The size of each STG.
    enum
    {
        BYTES_PER_STG = ELEMENTS_PER_STG * BYTES_PER_ELEMENT
    };

    // The size of a row in bytes.
    enum
    {
        BYTES_PER_ROW = Cta_tile::N * BYTES_PER_ELEMENT
    };

    // Ctor.
    inline __device__ Gmem_tile_ps_hopper(void* ptr, int64_t const params_stride_in_bytes, int64_t const bytes_per_row,
        uint32_t const params_scale, int tidx)
        : params_stride_in_bytes_(params_stride_in_bytes)
        , params_scale_(params_scale)
        , ptr_(reinterpret_cast<char*>(ptr))
    {
        // For storing P and S, we do not take into account variable sequence length.

        // The block index for the batch.
        int const bidb = blockIdx.y;
        // The block index for the head.
        int const bidh = blockIdx.x;
        // The block index.
        int bidx = bidb * gridDim.x + bidh;

        // Decompose the position of the thread into warp/lane.
        int warp = tidx / Cta_tile::THREADS_PER_WARP;
        int lane = tidx % Cta_tile::THREADS_PER_WARP;

        int warpgroup_idx = warp / 4;
        int warp_idx_within_warpgroup = warp % 4;

        // Compute the position in the sequence (within the CTA for the moment).
        int row = warp_idx_within_warpgroup * (Mma_tile::M_PER_MMA / 4) + lane / 4;
        // Compute the position of the thread in the row.
        int col = warpgroup_idx * Mma_tile::N_PER_MMA + lane % 4 * ELEMENTS_PER_STG;

        // The offset of the 1st row written by the thread. We store the P matrix interleaved.
        int64_t row_offset = (int64_t) row * params_stride_in_bytes_ + bidx * bytes_per_row;
        // Finalize the pointer.
        ptr_ += row_offset + col * BYTES_PER_ELEMENT;
    }

    // Ctor.
    inline __device__ Gmem_tile_ps_hopper(
        void* ptr, int64_t const params_stride_in_bytes, uint32_t const params_scale, int tidx)
        : Gmem_tile_ps_hopper(ptr, params_stride_in_bytes, BYTES_PER_ROW, params_scale, tidx)
    {
    }

    // Store data to memory.
    template <typename Accumulators, int M, int N>
    inline __device__ void store(Accumulators const (&acc)[M][N])
    {

        // A thread holds packet of 2 elements. In 2x2 tile per MMA.
        // Need to figure out if we need this for hopper.
        int64_t const step_m = 8 * (this->params_stride_in_bytes_);
        int64_t const step_n = 8 * BYTES_PER_ELEMENT;

// Store the different accumulators.
#pragma unroll
        for (int mi = 0; mi < M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < N; ++ni)
            {
                int64_t offset = (int64_t) mi * Mma_tile::M_PER_MMA_PER_CTA * (this->params_stride_in_bytes_)
                    + ni * Mma_tile::N_PER_MMA_PER_CTA * BYTES_PER_ELEMENT;

                Store_accumulator<Traits, BITS_PER_ELEMENT> delegate;
                delegate.store(this->ptr_ + offset, step_m, step_n, acc[mi][ni], this->params_scale_);
            }
        }
    }

    // Move to the next location.
    inline __device__ void move()
    {
        ptr_ += (int64_t) Cta_tile::M * params_stride_in_bytes_;
    }

    // The stride between rows for the QKV matrice.
    int64_t const params_stride_in_bytes_;
    // The scale to apply before storing the element.
    uint32_t const params_scale_;
    // The pointer.
    char* ptr_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, int BITS_PER_ELEMENT>
struct Gmem_tile_s : public Gmem_tile_ps<Traits, Cta_tile, BITS_PER_ELEMENT>
{

    // The base class.
    using Base = Gmem_tile_ps<Traits, Cta_tile, BITS_PER_ELEMENT>;

    // Ctor.
    inline __device__ Gmem_tile_s(
        void* ptr, int64_t const params_stride_in_bytes, uint32_t const params_scale, int tidx)
        : Base(ptr, params_stride_in_bytes, params_scale, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, int BITS_PER_ELEMENT>
struct Gmem_tile_s<Ampere_hmma_fp32_traits, Cta_tile, BITS_PER_ELEMENT>
    : public Gmem_tile_ps<Ampere_hmma_fp16_traits, Cta_tile, BITS_PER_ELEMENT>
{

    // The base class.
    using Base = Gmem_tile_ps<Ampere_hmma_fp16_traits, Cta_tile, BITS_PER_ELEMENT>;

    // Ctor.
    inline __device__ Gmem_tile_s(
        void* ptr, int64_t const params_stride_in_bytes, uint32_t const params_scale, int tidx, int cta_row_offset = 0)
        : Base(ptr, params_stride_in_bytes, float_to_half2(reinterpret_cast<float const&>(params_scale)), tidx,
            cta_row_offset)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha
