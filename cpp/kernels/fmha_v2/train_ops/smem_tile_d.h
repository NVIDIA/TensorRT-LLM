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

#include <fmha/traits.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile>
struct Smem_tile_mma
{

    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;
    using Fragment = fmha::Fragment_a<Traits, fmha::Col>;

    enum
    {
        COLS = Cta_tile::N
    };

    enum
    {
        BYTES_PER_ELT = 2
    };

    enum
    {
        BYTES_PER_STS = 4
    };

    enum
    {
        BYTES_PER_ROW = COLS * BYTES_PER_ELT
    }; // TODO

    enum
    {
        BYTES_PER_TILE = Cta_tile::M * BYTES_PER_ROW
    };

    enum
    {
        WARPS_M = Cta_tile::WARPS_M
    };

    enum
    {
        WARPS_N = Cta_tile::WARPS_N
    };

    enum
    {
        WARPS_K = Cta_tile::WARPS_K
    };

    static_assert(WARPS_K == 1);

    using Acc = fmha::Fragment_accumulator<Traits>;

    inline __device__ Smem_tile_mma(char* smem, int tidx)
    {
        smem_ = __nvvm_get_smem_pointer(smem);

        int write_col, write_row;
        static_assert(WARPS_M == 1 && (WARPS_N == 4 || WARPS_N == 8) || (WARPS_M == 4 || WARPS_N == 8) || WARPS_N == 1);
        if (WARPS_M == 1 && (WARPS_N == 4 || WARPS_N == 8))
        {
            write_row = (tidx & 0x1c) / 4;
            write_col = (tidx & 0xe0) / 4 + (tidx & 0x03);
        }
        else
        {
            write_row = (tidx & 0xe0) / 2 + (tidx & 0x1c) / 4;
            write_col = (tidx & 0x03);
        }
        write_col ^= (write_row & 0x07) * 4;

        write_offset_ = write_row * BYTES_PER_ROW + write_col * BYTES_PER_STS;
    }

    template <int M, int N>
    inline __device__ void store(uint4 const (&regs)[M][N])
    {
        static_assert(COLS == Cta_tile::N);
#pragma unroll
        for (int mi = 0; mi < M; mi++)
        {
#pragma unroll
            for (int ni = 0; ni < N; ni++)
            {
                size_t offset = write_offset_ + mi * WARPS_M * 16 * BYTES_PER_ROW + ni * WARPS_N * 16 * BYTES_PER_ELT;
                fmha::sts(smem_ + offset + 0 * BYTES_PER_ROW, regs[mi][ni].x);
                fmha::sts(smem_ + offset + 8 * BYTES_PER_ROW, regs[mi][ni].z);
                offset ^= 4 * BYTES_PER_STS;
                fmha::sts(smem_ + offset + 0 * BYTES_PER_ROW, regs[mi][ni].y);
                fmha::sts(smem_ + offset + 8 * BYTES_PER_ROW, regs[mi][ni].w);
            }
        }
    }

    uint32_t smem_;
    uint32_t write_offset_;
    uint32_t warp_m;
    uint32_t warp_n;
    uint32_t lane;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, typename Base = Smem_tile_mma<Traits, Cta_tile>>
struct Smem_tile_mma_transposed : public Base
{
    enum
    {
        BYTES_PER_LDS = 16
    };

    enum
    {
        BYTES_PER_ROW = Base::BYTES_PER_ROW
    };

    enum
    {
        BYTES_PER_ELT = Base::BYTES_PER_ELT
    };

    enum
    {
        WARPS_M = Base::WARPS_M
    };

    enum
    {
        WARPS_N = Base::WARPS_N
    };

    static_assert(WARPS_M == 1 && (WARPS_N == 4 || WARPS_N == 8));
    using Fragment = typename Base::Fragment;

    inline __device__ Smem_tile_mma_transposed(char* smem, int tidx)
        : Base(smem, tidx)
    {

        static_assert(WARPS_M == 1 && (WARPS_N == 4 || WARPS_N == 8));
        int read_row, read_col;
        read_row = (tidx & 0x0f);
        read_col = (tidx & 0xe0) / 16 + (tidx & 0x1c) / 16;

        read_col ^= (read_row & 0x07);
        read_offset_ = read_row * BYTES_PER_ROW + read_col * BYTES_PER_LDS;
    }

    template <int M, int N>
    inline __device__ void load(Fragment (&frag)[M][N])
    {
        static_assert(Base::COLS == Cta_tile::N);
        for (int mi = 0; mi < M; mi++)
        {
            for (int ni = 0; ni < N; ni++)
            {
                size_t offset = read_offset_ + mi * WARPS_M * 16 * BYTES_PER_ROW + ni * WARPS_N * 16 * BYTES_PER_ELT;
                uint4 dst;
                fmha::ldsmt(dst, this->smem_ + offset);
                frag[mi][ni].reg(0) = dst.x;
                frag[mi][ni].reg(1) = dst.z; // Fragment A regs col major!
                frag[mi][ni].reg(2) = dst.y;
                frag[mi][ni].reg(3) = dst.w;
            }
        }
    }

    uint32_t read_offset_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, typename Base = Smem_tile_mma<Traits, Cta_tile>>
struct Smem_tile_mma_epilogue : public Base
{
    enum
    {
        BYTES_PER_LDS = 16
    };

    enum
    {
        BYTES_PER_ROW = Base::BYTES_PER_ROW
    };

    enum
    {
        BYTES_PER_ELT = Base::BYTES_PER_ELT
    };

    enum
    {
        THREADS_PER_ROW = BYTES_PER_ROW / BYTES_PER_LDS
    };

    static_assert(THREADS_PER_ROW * BYTES_PER_LDS == BYTES_PER_ROW);

    enum
    {
        ROWS_PER_LDS = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW
    };

    enum
    {
        NUM_LDS = Cta_tile::M / ROWS_PER_LDS
    };

    static_assert(NUM_LDS * ROWS_PER_LDS == Cta_tile::M);

    enum
    {
        WARPS_M = Base::WARPS_M
    };

    enum
    {
        WARPS_N = Base::WARPS_N
    };

    static_assert((WARPS_M == 4 || WARPS_N == 8) || WARPS_N == 1);

    // The dst data type
    using Dst_type = typename Traits::A_type;

    using Acc = fmha::Fragment_accumulator<Traits>;

    inline __device__ Smem_tile_mma_epilogue(char* smem, int tidx)
        : Base(smem, tidx)
    {

        int const read_row = tidx / THREADS_PER_ROW;
        int read_col = tidx % THREADS_PER_ROW;
        read_col ^= (read_row & 0x07);
        read_offset_ = read_row * BYTES_PER_ROW + read_col * BYTES_PER_LDS;
    }

    inline __device__ void load(uint4 (&data)[NUM_LDS])
    {
        for (int ii = 0; ii < NUM_LDS; ii++)
        {
            size_t offset = read_offset_ + ii * ROWS_PER_LDS * BYTES_PER_ROW;
            fmha::lds(data[ii], this->smem_ + offset);
        }
    }

    template <int M, int N>
    inline __device__ void store(Acc const (&acc)[M][N])
    {
#pragma unroll
        for (int mi = 0; mi < M; mi++)
        {
#pragma unroll
            for (int ni = 0; ni < N; ni++)
            {
                // 1st row - 4 elements per row.
                float tmp00 = acc[mi][ni].elt(0);
                float tmp01 = acc[mi][ni].elt(1);
                float tmp02 = acc[mi][ni].elt(4);
                float tmp03 = acc[mi][ni].elt(5);
                // 2nd row - 4 elements per row.
                float tmp10 = acc[mi][ni].elt(2);
                float tmp11 = acc[mi][ni].elt(3);
                float tmp12 = acc[mi][ni].elt(6);
                float tmp13 = acc[mi][ni].elt(7);

                uint32_t x = fmha::float2_to_16bit_2<Dst_type>(tmp00, tmp01);
                uint32_t y = fmha::float2_to_16bit_2<Dst_type>(tmp02, tmp03);
                uint32_t z = fmha::float2_to_16bit_2<Dst_type>(tmp10, tmp11);
                uint32_t w = fmha::float2_to_16bit_2<Dst_type>(tmp12, tmp13);

                size_t offset = (this->write_offset_ ^ (ni * 32)) + mi * WARPS_M * 16 * BYTES_PER_ROW;
                fmha::sts(this->smem_ + offset + 0 * BYTES_PER_ROW, x);
                fmha::sts(this->smem_ + offset + 8 * BYTES_PER_ROW, z);
                offset ^= 4 * Base::BYTES_PER_STS;
                fmha::sts(this->smem_ + offset + 0 * BYTES_PER_ROW, y);
                fmha::sts(this->smem_ + offset + 8 * BYTES_PER_ROW, w);
            }
        }
    }

    template <int M, int N>
    inline __device__ void store(uint4 const (&regs)[M][N])
    {
        for (int mi = 0; mi < M; mi++)
        {
            for (int ni = 0; ni < N; ni++)
            {
                size_t offset = (this->write_offset_ ^ (ni * 32)) + mi * WARPS_M * 16 * BYTES_PER_ROW;
                fmha::sts(this->smem_ + offset + 0 * BYTES_PER_ROW, regs[mi][ni].x);
                fmha::sts(this->smem_ + offset + 8 * BYTES_PER_ROW, regs[mi][ni].z);
                offset ^= 4 * Base::BYTES_PER_STS;
                fmha::sts(this->smem_ + offset + 0 * BYTES_PER_ROW, regs[mi][ni].y);
                fmha::sts(this->smem_ + offset + 8 * BYTES_PER_ROW, regs[mi][ni].w);
            }
        }
    }

    uint32_t read_offset_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
