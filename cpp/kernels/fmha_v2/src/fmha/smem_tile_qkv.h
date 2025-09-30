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

#include <fmha/smem_tile.h>

namespace fmha
{

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, typename Traits_, int ROWS, int COLS>
struct Smem_tile_qkv_interleaved : public fmha::Smem_tile_without_skews<Cta_tile,
                                       ROWS, // Cta_tile::M * 2,
                                       COLS, // Cta_tile::K / 2,
                                       8,    // bits per element
                                       16,   // bytes per STS
                                       1,    // buffers per tile
                                       0,    // enable lds fast path
                                       2,    // ROWS PER XOR: 2 enough since we have 4 rows of the
                                       // 8x16 LDSM mat in one SMEM row
                                       1 // cols per xor
                                       >
{

    // The traits class.
    using Traits = Traits_;
    // The base class.
    using Base = fmha::Smem_tile_without_skews<Cta_tile,
        ROWS, // Cta_tile::M * 2,
        COLS, // Cta_tile::K / 2,
        8,    // bits per element
        16,   // bytes per STS
        1,    // buffers per tile
        0,    // enable lds fast path
        2,    // ROWS PER XOR: 2 enough since we have 4 rows of
        // the 8x16 LDSM mat in one SMEM row
        1 // cols per xor
        >;

    // The MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    // The fragment.

    // The size of a single LDS in bytes.
    enum
    {
        BYTES_PER_LDS = 16
    };

    enum
    {
        ROWS_PER_WARP = Cta_tile::THREADS_PER_WARP / Base::THREADS_PER_ROW
    };

    using Fragment_a = fmha::Fragment_a<Traits_, fmha::Row>;
    using Fragment_b = fmha::Fragment_b<Traits_, fmha::Col>;

    inline __device__ Smem_tile_qkv_interleaved(char* smem, int tidx)
        : Base(smem, tidx)
    {
    }

    uint32_t offset;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, typename Traits_>
struct Smem_tile_qk_interleaved_a_base : public Smem_tile_qkv_interleaved<Cta_tile, Traits_,
                                             Cta_tile::M * 2, // ROWS
                                             Cta_tile::K / 2  // COLS
                                             >
{

    using Base = Smem_tile_qkv_interleaved<Cta_tile, Traits_,
        Cta_tile::M * 2, // ROWS
        Cta_tile::K / 2  // COLS
        >;

    static_assert(Base::THREADS_PER_ROW == 128 / 16, "");

    enum
    {
        SMEM_ROWS_PER_WARP = Base::ROWS_PER_WARP
    };

    static_assert(SMEM_ROWS_PER_WARP == 4, "");

    using Mma_tile = typename Base::Mma_tile;
    using Fragment = typename Base::Fragment_a;

    inline __device__ Smem_tile_qk_interleaved_a_base(char* smem, int tidx)
        : Base(smem, tidx)
    {

        static_assert(Cta_tile::WARPS_K == 1, "");
        static_assert(Cta_tile::WARPS_M == 1 || Cta_tile::WARPS_M == 2, "");
        static_assert(Cta_tile::WARPS_N == 2 || Cta_tile::WARPS_N == 4, "");

        constexpr int WARPS_M = Cta_tile::WARPS_M;
        constexpr int WARPS_N = Cta_tile::WARPS_N;
        constexpr int WARPS_K = Cta_tile::WARPS_K;

        constexpr int WARP_MASK_M = fmha::Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;
        constexpr int WARP_DIV_M = 1 * 1 * Cta_tile::THREADS_PER_WARP;

        int const warp_m = (tidx & WARP_MASK_M) / WARP_DIV_M;

        /* Read address layout for ldsm:
         * [ 0 16  1 17  2 18  3 19]
         * [20  4 21  5 22  6 23  7]
         * [ 8 24  9 25 10 26 11 27]
         * [28 12 29 13 30 14 31 15]
         */
        int read_row = (tidx & 0x04) / 4 + (tidx & 0x08) / 4 + warp_m * SMEM_ROWS_PER_WARP;
        int read_col = (tidx & 0x03) * 2 + (tidx & 0x10) / 16;
        read_col ^= (read_row & 0x01);

        this->offset = read_row * Base::BYTES_PER_ROW + read_col * Base::BYTES_PER_LDS;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile>
struct Smem_tile_qk_interleaved_a
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Smem_tile_qk_interleaved_a<fmha::Volta_imma_int8_int32_traits, Cta_tile>
    : public Smem_tile_qk_interleaved_a_base<Cta_tile, fmha::Volta_imma_int8_int32_traits>
{

    using Traits = fmha::Volta_imma_int8_int32_traits;
    using Base = Smem_tile_qk_interleaved_a_base<Cta_tile, Traits>;
    using Mma_tile = typename Base::Mma_tile;
    using Fragment = typename Base::Fragment;

    inline __device__ Smem_tile_qk_interleaved_a(char* smem, int tidx)
        : Base(smem, tidx)
    {
    }

    inline __device__ void load(Fragment (&frag)[Mma_tile::MMAS_M], int ki)
    {
        int slice = ki / 2;

#pragma unroll
        for (int mi = 0; mi < Mma_tile::MMAS_M; mi++)
        {
            // the data for the second slice sits below the first slice
            uint32_t read_ptr = this->smem_ + this->offset + slice * Base::ROWS * Base::BYTES_PER_ROW / 2;
            uint2 data;
            ldsm_with_lds(data, read_ptr + mi * Cta_tile::WARPS_M * Base::SMEM_ROWS_PER_WARP * Base::BYTES_PER_ROW);
            static_assert(Fragment::NUM_REGS == 2, "");
            frag[mi].reg(0) = data.x;
            frag[mi].reg(1) = data.y;
        }

        this->offset ^= 16;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Smem_tile_qk_interleaved_a<fmha::Turing_imma_int8_int32_traits, Cta_tile>
    : public Smem_tile_qk_interleaved_a_base<Cta_tile, fmha::Turing_imma_int8_int32_traits>
{

    using Traits = fmha::Turing_imma_int8_int32_traits;
    using Base = Smem_tile_qk_interleaved_a_base<Cta_tile, Traits>;
    using Mma_tile = typename Base::Mma_tile;
    using Fragment = typename Base::Fragment;

    inline __device__ Smem_tile_qk_interleaved_a(char* smem, int tidx)
        : Base(smem, tidx)
    {
    }

    inline __device__ void load(Fragment (&frag)[Mma_tile::MMAS_M], int ki)
    {
        int slice = ki / 2;

#pragma unroll
        for (int mi = 0; mi < Mma_tile::MMAS_M; mi++)
        {
            // the data for the second slice sits below the first slice
            uint32_t read_ptr = this->smem_ + this->offset + slice * Base::ROWS * Base::BYTES_PER_ROW / 2;
            uint2 data;
            fmha::ldsm(data, read_ptr + mi * Cta_tile::WARPS_M * Base::SMEM_ROWS_PER_WARP * Base::BYTES_PER_ROW);
            static_assert(Fragment::NUM_REGS == 2, "");
            frag[mi].reg(0) = data.x;
            frag[mi].reg(1) = data.y;
        }

        this->offset ^= 16;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Smem_tile_qk_interleaved_a<fmha::Ampere_imma_int8_int32_traits, Cta_tile>
    : public Smem_tile_qk_interleaved_a_base<Cta_tile, fmha::Ampere_imma_int8_int32_traits>
{

    using Traits = fmha::Ampere_imma_int8_int32_traits;
    using Base = Smem_tile_qk_interleaved_a_base<Cta_tile, Traits>;
    using Mma_tile = typename Base::Mma_tile;
    using Fragment = typename Base::Fragment;

    inline __device__ Smem_tile_qk_interleaved_a(char* smem, int tidx)
        : Base(smem, tidx)
    {
    }

    inline __device__ void load(Fragment (&frag)[Mma_tile::MMAS_M], int ki)
    {
#pragma unroll
        for (int mi = 0; mi < Mma_tile::MMAS_M; mi++)
        {
            // the data for the second slice sits below the first slice
            uint32_t read_ptr = this->smem_ + this->offset + ki * Base::ROWS * Base::BYTES_PER_ROW / 2;
            uint4 data;
            fmha::ldsm(data, read_ptr + mi * Cta_tile::WARPS_M * Base::SMEM_ROWS_PER_WARP * Base::BYTES_PER_ROW);
            static_assert(Fragment ::NUM_REGS == 4, "");
            frag[mi].reg(0) = data.x;
            frag[mi].reg(1) = data.y;
            frag[mi].reg(2) = data.z;
            frag[mi].reg(3) = data.w;
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, typename Traits_>
struct Smem_tile_qk_interleaved_b_base : public Smem_tile_qkv_interleaved<Cta_tile, Traits_,
                                             Cta_tile::N * 2, // ROWS
                                             Cta_tile::K / 2  // COLS
                                             >
{

    using Base = Smem_tile_qkv_interleaved<Cta_tile, Traits_,
        Cta_tile::N * 2, // ROWS
        Cta_tile::K / 2  // COLS
        >;

    using Mma_tile = typename Base::Mma_tile;
    using Fragment = typename Base::Fragment_b;

    inline __device__ Smem_tile_qk_interleaved_b_base(char* smem, int tidx)
        : Base(smem, tidx)
    {

        constexpr int WARPS_M = Cta_tile::WARPS_M;
        constexpr int WARPS_N = Cta_tile::WARPS_N;
        constexpr int WARPS_K = Cta_tile::WARPS_K;

        // 2x2: 2,2,1 then 2,1,2
        // 1x4: 1,4,1 then 1,1,4
        static_assert(WARPS_K == 1, "");

        constexpr int WARP_MASK_N = fmha::Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;
        constexpr int WARP_DIV_N = WARPS_M * 1 * Cta_tile::THREADS_PER_WARP;

        // Only need to care about warp_n, because if warps_m > 1, both of them should load
        // the same data
        int const warp = (tidx & WARP_MASK_N) / WARP_DIV_N;

        /* transpose the order of the LDSMs: first along K, then along N
         * [ 0  8  1  9  2 10  3 11]
         * [12  4 13  5 14  6 15  7]
         * [16 24 17 25 18 26 19 27]
         * [28 20 29 21 30 22 31 23]
         */
        int read_row = (tidx & 0x04) / 4 + (tidx & 0x10) / 8 + warp * Base::ROWS_PER_WARP;
        int read_col = (tidx & 0x03) * 2 + (tidx & 0x08) / 8;
        read_col ^= (read_row & 0x01);

        this->offset = read_row * Base::BYTES_PER_ROW + read_col * Base::BYTES_PER_LDS;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits_, typename Cta_tile>
struct Smem_tile_qk_interleaved_b
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Smem_tile_qk_interleaved_b<fmha::Volta_imma_int8_int32_traits, Cta_tile>
    : public Smem_tile_qk_interleaved_b_base<Cta_tile, fmha::Volta_imma_int8_int32_traits>
{

    using Traits = fmha::Volta_imma_int8_int32_traits;
    using Base = Smem_tile_qk_interleaved_b_base<Cta_tile, Traits>;

    using Mma_tile = typename Base::Mma_tile;
    using Fragment = typename Base::Fragment;

    inline __device__ Smem_tile_qk_interleaved_b(char* smem, int tidx)
        : Base(smem, tidx)
    {

        constexpr int WARPS_M = Cta_tile::WARPS_M;
        constexpr int WARPS_N = Cta_tile::WARPS_N;
        constexpr int WARPS_K = Cta_tile::WARPS_K;
        constexpr int WARP_MASK_N = fmha::Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;
        constexpr int WARP_DIV_N = WARPS_M * 1 * Cta_tile::THREADS_PER_WARP;

        // Only need to care about warp_n, because if warps_m > 1, both of them should load
        // the same data
        int const warp = (tidx & WARP_MASK_N) / WARP_DIV_N;

        int read_row = (tidx & 0x04) / 4 + (tidx & 0x08) / 4 + warp * Base::ROWS_PER_WARP;
        int read_col = (tidx & 0x03) * 2 + (tidx & 0x10) / 16;
        read_col ^= (read_row & 0x01);

        this->offset = read_row * Base::BYTES_PER_ROW + read_col * Base::BYTES_PER_LDS;
    }

    inline __device__ void load(Fragment (&frag)[Mma_tile::MMAS_N], int ki)
    {
        int slice = ki / 2;
#pragma unroll
        for (int ni = 0; ni < Mma_tile::MMAS_N; ni++)
        {
            uint32_t read_ptr = this->smem_ + this->offset + slice * Base::ROWS * Base::BYTES_PER_ROW / 2;
            uint2 data;
            ldsm_with_lds(data, read_ptr + ni * Base::ROWS_PER_WARP * Cta_tile::WARPS_N * Base::BYTES_PER_ROW);
            static_assert(Fragment ::NUM_REGS == 2, "");
            frag[ni].reg(0) = data.x;
            frag[ni].reg(1) = data.y;
        }
        this->offset ^= 16;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Smem_tile_qk_interleaved_b<fmha::Turing_imma_int8_int32_traits, Cta_tile>
    : public Smem_tile_qk_interleaved_b_base<Cta_tile, fmha::Turing_imma_int8_int32_traits>
{

    using Traits = fmha::Turing_imma_int8_int32_traits;
    using Base = Smem_tile_qk_interleaved_b_base<Cta_tile, Traits>;

    using Mma_tile = typename Base::Mma_tile;
    using Fragment = typename Base::Fragment;

    inline __device__ Smem_tile_qk_interleaved_b(char* smem, int tidx)
        : Base(smem, tidx)
    {

        constexpr int WARPS_M = Cta_tile::WARPS_M;
        constexpr int WARPS_N = Cta_tile::WARPS_N;
        constexpr int WARPS_K = Cta_tile::WARPS_K;
        constexpr int WARP_MASK_N = fmha::Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;
        constexpr int WARP_DIV_N = WARPS_M * 1 * Cta_tile::THREADS_PER_WARP;

        // Only need to care about warp_n, because if warps_m > 1, both of them should load
        // the same data
        int const warp = (tidx & WARP_MASK_N) / WARP_DIV_N;

        int read_row = (tidx & 0x04) / 4 + (tidx & 0x08) / 4 + warp * Base::ROWS_PER_WARP;
        int read_col = (tidx & 0x03) * 2 + (tidx & 0x10) / 16;
        read_col ^= (read_row & 0x01);

        this->offset = read_row * Base::BYTES_PER_ROW + read_col * Base::BYTES_PER_LDS;
    }

    inline __device__ void load(Fragment (&frag)[Mma_tile::MMAS_N], int ki)
    {
        int slice = ki / 2;
#pragma unroll
        for (int ni = 0; ni < Mma_tile::MMAS_N; ni++)
        {
            uint32_t read_ptr = this->smem_ + this->offset + slice * Base::ROWS * Base::BYTES_PER_ROW / 2;
            uint2 data;
            fmha::ldsm(data, read_ptr + ni * Base::ROWS_PER_WARP * Cta_tile::WARPS_N * Base::BYTES_PER_ROW);
            static_assert(Fragment ::NUM_REGS == 2, "");
            frag[ni].reg(0) = data.x;
            frag[ni].reg(1) = data.y;
        }
        this->offset ^= 16;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Smem_tile_qk_interleaved_b<fmha::Ampere_imma_int8_int32_traits, Cta_tile>
    : public Smem_tile_qk_interleaved_b_base<Cta_tile, fmha::Ampere_imma_int8_int32_traits>
{

    using Traits = fmha::Ampere_imma_int8_int32_traits;
    using Base = Smem_tile_qk_interleaved_b_base<Cta_tile, Traits>;

    using Mma_tile = typename Base::Mma_tile;
    using Fragment = typename Base::Fragment;

    inline __device__ Smem_tile_qk_interleaved_b(char* smem, int tidx)
        : Base(smem, tidx)
    {
    }

    inline __device__ void load(Fragment (&frag)[Mma_tile::MMAS_N], int ki)
    {
#pragma unroll
        for (int ni = 0; ni < Mma_tile::MMAS_N; ni++)
        {
            uint32_t read_ptr = this->smem_ + this->offset + ki * Base::ROWS * Base::BYTES_PER_ROW / 2;
            uint4 data;
            fmha::ldsm(data, read_ptr + ni * Base::ROWS_PER_WARP * Cta_tile::WARPS_N * Base::BYTES_PER_ROW);
            static_assert(Fragment ::NUM_REGS == 4, "");
            frag[ni].reg(0) = data.x;
            frag[ni].reg(1) = data.y;
            frag[ni].reg(2) = data.z;
            frag[ni].reg(3) = data.w;
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, typename Traits_>
struct Smem_tile_v_interleaved_b_base : public Smem_tile_qkv_interleaved<Cta_tile, // BMM2: M = STEP, N = d, K = s
                                            Traits_,
                                            Cta_tile::K * 2,                       // ROWS: K is the sequence length
                                            Cta_tile::N / 2                        // COLS: N is the head dimension
                                            >
{

    using Base = Smem_tile_qkv_interleaved<Cta_tile, Traits_,
        Cta_tile::K * 2, // ROWS
        Cta_tile::N / 2  // COLS
        >;

    using Mma_tile = typename Base::Mma_tile;
    // TODO Row or col?
    using Fragment = typename Base::Fragment_b;

    inline __device__ Smem_tile_v_interleaved_b_base(char* smem, int tidx)
        : Base(smem, tidx)
    {

        // // DEBUG.
        // static_assert( Cta_tile::N == 64, "" );
        // // END OF DEBUG.

        constexpr int WARPS_M = Cta_tile::WARPS_M;
        constexpr int WARPS_N = Cta_tile::WARPS_N;
        constexpr int WARPS_K = Cta_tile::WARPS_K;

        // 2x2: 2,2,1 then 2,1,2
        // 1x4: 1,4,1 then 1,1,4
        static_assert(WARPS_N == 1, "");

        // Don't need to consider WARP M. For two warps in M, both would read the same tile
        constexpr int WARP_MASK_K = fmha::Warp_masks<WARPS_M, WARPS_N, WARPS_K>::K;
        constexpr int WARP_DIV_K = WARPS_M * WARPS_N * Cta_tile::THREADS_PER_WARP;

        // the static assert above ensures, that only warp_m or warp_k is non-zero
        int const warp = (tidx & WARP_MASK_K) / WARP_DIV_K;

        /* LDSM.T addresses: warps are split in two to match BMM1-GEMM-N (= BMM2-GEMM-K) register
         * layout
         *    <== GEMM-N = D = 64 ==>
         * [ 0:  0  0  1  0  2  0  3  0]  WARP 0
         * [ 1:  0  4  0  5  0  6  0  7]
         * [ 2:  8  0  9  0 10  0 11  0]
         * [ 3:  0 12  0 13  0 14  0 15]
         * [ 4:  0  0  0  0  0  0  0  0]  WARP 1
         * [ 5:  0  0  0  0  0  0  0  0]
         * [ 6:  0  0  0  0  0  0  0  0]
         * [ 7:  0  0  0  0  0  0  0  0]
         * [ 8:  0  0  0  0  0  0  0  0]  WARP 2
         * [ 9:  0  0  0  0  0  0  0  0]
         * [10:  0  0  0  0  0  0  0  0]
         * [11:  0  0  0  0  0  0  0  0]
         * [12:  0  0  0  0  0  0  0  0]  WARP 3
         * [13:  0  0  0  0  0  0  0  0]
         * [14:  0  0  0  0  0  0  0  0]
         * [15:  0  0  0  0  0  0  0  0]
         * [16: 16  0 17  0 18  0 19  0]  WARP 0
         * [17:  0 20  0 21  0 22  0 23]
         * [18: 24  0 25  0 26  0 27  0]
         * [19:  0 28  0 29  0 30  0 31]
         * etc ...
         */

        // TODO this is a bit misleading, as 4 rows per warp applies to the
        // row-major tiles above. In this smem tile, a warp actually owns 8 rows in
        // SMEM, but we have 4 rows per slice

        // TODO would be good to rename to SMEM_ROWS_PER_WARP to make this clearer
        static_assert(Base::ROWS_PER_WARP == 4, "");

        read_row = ((tidx & 0x0f) / 4) + warp * Base::ROWS_PER_WARP;
        read_col = (tidx & 0x03) * 2;
        read_col ^= (read_row & 0x01);

        // this->offset = read_row * Base::BYTES_PER_ROW + read_col * Base::BYTES_PER_LDS;
    }

    int read_row;
    int read_col;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits_, typename Cta_tile>
struct Smem_tile_v_interleaved_b
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Smem_tile_v_interleaved_b<fmha::Volta_imma_int8_int32_traits, Cta_tile>
    : public Smem_tile_v_interleaved_b_base<Cta_tile, fmha::Volta_imma_int8_int32_traits>
{

    using Traits = fmha::Volta_imma_int8_int32_traits;
    using Base = Smem_tile_v_interleaved_b_base<Cta_tile, Traits>;
    using Mma_tile = typename Base::Mma_tile;
    using Fragment = typename Base::Fragment_b;

    // Ctor.
    inline __device__ Smem_tile_v_interleaved_b(char* smem, int tidx)
        : Base(smem, tidx)
    {
    }

    // Load fragments from shared memory.
    inline __device__ void load(Fragment (&frag)[Mma_tile::MMAS_N], int ki)
    {
        // static_assert(Mma_tile::MMAS_K == 4, "");
        static_assert(Mma_tile::MMAS_N == 4, "");
        static_assert(Base::ROWS_PER_WARP == 4, "");
        // static_assert(Cta_tile::WARPS_K == 2, "");

        int offset_k = ki * Cta_tile::WARPS_K * Base::ROWS_PER_WARP;
#pragma unroll
        for (int ni = 0; ni < Mma_tile::MMAS_N; ni++)
        {
            uint32_t offset
                = (this->read_row + offset_k) * Base::BYTES_PER_ROW + (this->read_col ^ (ni & 1)) * Base::BYTES_PER_LDS;

            // for the next 32B in N, we have to jump down K rows, so K / 4 rows in
            // smem, which stores 4 canonical 32B rows per 128B
            offset += (ni / 2) * Cta_tile::K / 4 * Base::BYTES_PER_ROW;
            uint32_t read_ptr = this->smem_ + offset; // + ki * Base::ROWS * Base::BYTES_PER_ROW / 2;
            uint2 data = {0, 0};
            ldsmt_with_lds(data, read_ptr);
            static_assert(Fragment ::NUM_REGS == 2, "");
            swizzle_rows(frag[ni].reg(0), frag[ni].reg(1), data.x, data.y);
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Smem_tile_v_interleaved_b<fmha::Turing_imma_int8_int32_traits, Cta_tile>
    : public Smem_tile_v_interleaved_b_base<Cta_tile, fmha::Turing_imma_int8_int32_traits>
{

    using Traits = fmha::Turing_imma_int8_int32_traits;
    using Base = Smem_tile_v_interleaved_b_base<Cta_tile, Traits>;
    using Mma_tile = typename Base::Mma_tile;
    using Fragment = typename Base::Fragment_b;

    // Ctor.
    inline __device__ Smem_tile_v_interleaved_b(char* smem, int tidx)
        : Base(smem, tidx)
    {
    }

    // Load fragments from shared memory.
    inline __device__ void load(Fragment (&frag)[Mma_tile::MMAS_N], int ki)
    {
        static_assert(Mma_tile::MMAS_N == 4, "");
        static_assert(Base::ROWS_PER_WARP == 4, "");

        int offset_k = ki * Cta_tile::WARPS_K * Base::ROWS_PER_WARP;
#pragma unroll
        for (int ni = 0; ni < Mma_tile::MMAS_N; ni++)
        {
            uint32_t offset
                = (this->read_row + offset_k) * Base::BYTES_PER_ROW + (this->read_col ^ (ni & 1)) * Base::BYTES_PER_LDS;
            // for the next 32B in N, we have to jump down K rows, so K / 4 rows in
            // smem, which stores 4 canonical 32B rows per 128B
            offset += (ni / 2) * Cta_tile::K / 4 * Base::BYTES_PER_ROW;
            uint32_t read_ptr = this->smem_ + offset; // + ki * Base::ROWS * Base::BYTES_PER_ROW / 2;
            uint2 data = {0, 0};
            fmha::ldsmt(data, read_ptr);
            static_assert(Fragment ::NUM_REGS == 2, "");
            swizzle_rows(frag[ni].reg(0), frag[ni].reg(1), data.x, data.y);
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Smem_tile_v_interleaved_b<fmha::Ampere_imma_int8_int32_traits, Cta_tile>
    : public Smem_tile_v_interleaved_b_base<Cta_tile, fmha::Ampere_imma_int8_int32_traits>
{

    // The instruction traits.
    using Traits = fmha::Ampere_imma_int8_int32_traits;
    // The base class.
    using Base = Smem_tile_v_interleaved_b_base<Cta_tile, Traits>;
    // The tile of MMAs.
    using Mma_tile = typename Base::Mma_tile;
    // The fragment loaded.
    using Fragment = typename Base::Fragment_b;

    // Ctor.
    inline __device__ Smem_tile_v_interleaved_b(char* smem, int tidx)
        : Base(smem, tidx)
    {
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&frag)[Mma_tile::MMAS_N], int ki)
    {

        int offset_k = ki * Cta_tile::WARPS_K * Base::ROWS_PER_WARP * 2;
        static_assert(Cta_tile::K != 192 || Mma_tile::MMAS_K == 2, "");
#pragma unroll
        for (int ni = 0; ni < Mma_tile::MMAS_N; ni++)
        {
            uint32_t offset
                = (this->read_row + offset_k) * Base::BYTES_PER_ROW + (this->read_col ^ (ni & 1)) * Base::BYTES_PER_LDS;

            // For the next 32B in N, we have to jump down K rows, so K / 4 rows in smem, which
            // stores 4 canonical 32B rows per 128B.
            offset += (ni / 2) * Cta_tile::K / 4 * Base::BYTES_PER_ROW;
            uint32_t read_ptr = this->smem_ + offset; // + ki * Base::ROWS * Base::BYTES_PER_ROW / 2;
            uint2 data0 = {0, 0};
            uint2 data1 = {0, 0};
            fmha::ldsmt(data0, read_ptr);

            if (Cta_tile::K != 192 || ki == 0)
            {
                static_assert(Cta_tile::K != 192 || Mma_tile::MMAS_K == 2);
                // For 192, with 4 warps, we need 128 rows of K, so for the second ldsm, we need
                // only 2x instead of 4x.
                int imm = Cta_tile::WARPS_K * Base::ROWS_PER_WARP * Base::BYTES_PER_ROW;
                fmha::ldsmt(data1, read_ptr + imm);
            }

            static_assert(Fragment ::NUM_REGS == 4, "");
            swizzle_rows(frag[ni].reg(0), frag[ni].reg(2), data0.x, data0.y);
            swizzle_rows(frag[ni].reg(1), frag[ni].reg(3), data1.x, data1.y);
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha
