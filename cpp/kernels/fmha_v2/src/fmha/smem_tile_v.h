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

#include <fmha/fragment.h>
#include <fmha/hopper/gmma_descriptor.h>
#include <fmha/smem_tile.h>
#include <fmha/traits.h>
#include <fmha/utils.h>

namespace fmha
{

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, int BUFFERS_PER_TILE = 1,
    fmha::Gmma_descriptor_mode desc_mode = fmha::Gmma_descriptor_mode::SWIZZLE_NONE, bool USE_TMA = false>
struct Smem_tile_v
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, template <class, int> class Rows_per_xor_pattern,
    int BUFFERS_PER_TILE = 1>
struct Smem_tile_v_hmma
{
    using Base = Smem_tile_without_skews<Cta_tile, Cta_tile::K, Cta_tile::N, 16, 16, BUFFERS_PER_TILE, 0,
        Rows_per_xor_pattern<Traits, Cta_tile::N>::VALUE, 1>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, int BUFFERS_PER_TILE>
struct Smem_tile_v<Volta_hmma_fp16_16x16x16_traits, Cta_tile, BUFFERS_PER_TILE>
    : public Smem_tile_v_hmma<Volta_hmma_fp16_16x16x16_traits, Cta_tile, Rows_per_xor_pattern_volta_b,
          BUFFERS_PER_TILE>::Base
{

    // The traits class.
    using Traits = fmha::Volta_hmma_fp16_16x16x16_traits;
    // The base class.
    using Base = typename Smem_tile_v_hmma<Traits, Cta_tile, Rows_per_xor_pattern_volta_b, BUFFERS_PER_TILE>::Base;
    // The MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;
    // The fragment.
    using Fragment = fmha::Fragment_b<Traits, fmha::Row>;

    // The size of a single LDS in bytes.
    enum
    {
        BYTES_PER_LDS = 16
    };

    // Ctor.
    inline __device__ Smem_tile_v(void* smem, int tidx)
        : Base(smem, tidx)
    {

        // Warps.
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

        // Determine the config.
        enum
        {
            WARPS_2x1x2 = WARPS_M == 2 && WARPS_N == 1 && WARPS_K == 2
        };

        enum
        {
            WARPS_1x1x8 = WARPS_M == 1 && WARPS_N == 1 && WARPS_K == 8
        };

        enum
        {
            WARPS_1x1x4 = WARPS_M == 1 && WARPS_N == 1 && WARPS_K == 4
        };

        // Flash Attention uses WARPS_4x1x1
        enum
        {
            WARPS_4x1x1 = WARPS_M == 4 && WARPS_N == 1 && WARPS_K == 1
        };

        // The row/col read by the thread.
        int read_row, read_col;

        // SEQLEN == 128 and N == 16.
        if (WARPS_2x1x2 && Cta_tile::N == 16)
        {
            read_row = (tidx & 0x40) / 16 + (tidx & 0x08) / 8;
            read_col = (tidx & 0x10) / 16 + (tidx & 0x03) * 2;

            // SEQLEN == 128 and N == 32.
        }
        else if (WARPS_2x1x2 && Cta_tile::N == 32)
        {
            read_row = (tidx & 0x40) / 8 + (tidx & 0x08) / 4 + (tidx & 0x02) / 2;
            read_col = (tidx & 0x10) / 16 + (tidx & 0x01) * 4;

            // SEQLEN == 128 and N == 64.
        }
        else if (WARPS_2x1x2 && Cta_tile::N == 64)
        {
            read_row = (tidx & 0x40) / 4 + (tidx & 0x08) / 2 + (tidx & 0x03);
            read_col = (tidx & 0x10) / 16;

            // SEQLEN == 256, 512 and N == 16.
        }
        else if ((WARPS_1x1x8 || WARPS_1x1x4) && Cta_tile::N == 16)
        {
            read_row = (tidx & 0xe0) / 8 + (tidx & 0x08) / 8;
            read_col = (tidx & 0x10) / 16 + (tidx & 0x03) * 2;

            // SEQLEN == 256, 512 and N == 32.
        }
        else if ((WARPS_1x1x8 || WARPS_1x1x4) && Cta_tile::N == 32)
        {
            read_row = (tidx & 0xe0) / 4 + (tidx & 0x08) / 4 + (tidx & 0x02) / 2;
            read_col = (tidx & 0x10) / 16 + (tidx & 0x01) * 4;

            // SEQLEN == 256, 384 and 512 and N == 64.
        }
        else if ((WARPS_1x1x8 || WARPS_1x1x4) && (Cta_tile::N == 64 || Cta_tile::N == 128 || Cta_tile::N == 256))
        {
            read_row = (tidx & 0xe0) / 2 + (tidx & 0x08) / 2 + (tidx & 0x03);
            read_col = (tidx & 0x10) / 16;

            // ANY SEQLEN and N == 16.
        }
        else if (WARPS_4x1x1 && Cta_tile::N == 16)
        {
            read_row = (tidx & 0x08) / 8;
            read_col = (tidx & 0x10) / 16 + (tidx & 0x03) * 2;

            // ANY SEQLEN and N == 32.
        }
        else if (WARPS_4x1x1 && Cta_tile::N == 32)
        {
            read_row = (tidx & 0x08) / 4 + (tidx & 0x02) / 2;
            read_col = (tidx & 0x10) / 16 + (tidx & 0x01) * 4;

            // ANY SEQLEN and N == 64/128/256.
        }
        else if (WARPS_4x1x1 && (Cta_tile::N == 64 || Cta_tile::N == 128 || Cta_tile::N == 256))
        {
            read_row = (tidx & 0x08) / 2 + (tidx & 0x03);
            read_col = (tidx & 0x10) / 16;

            // Not supported!
        }
        else
        {
            assert(false);
        }

        // Apply the XOR for the column.
        read_col ^= read_row % Base::ROWS_PER_XOR_PATTERN;

        // The shared memory offset.
        this->smem_read_offset_ = read_row * Base::BYTES_PER_ROW + read_col * BYTES_PER_LDS;
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&b)[Mma_tile::VALID_MMAS_N], int ki)
    {
#pragma unroll
        for (int ni = 0; ni < Mma_tile::MMAS_N; ++ni)
        {

            // The column offset.
            int offset = this->smem_read_offset_ ^ (ni * 2 * BYTES_PER_LDS);

            // Skip N paddings
            if (ni < Mma_tile::VALID_MMAS_N)
            {
                // The rows.
                int row_0 = ki * 16 * Cta_tile::WARPS_K + 0;
                int row_1 = ki * 16 * Cta_tile::WARPS_K + 8;

                // Load the data using 2x LDS.128.
                uint4 tmp;
                fmha::lds(tmp, this->smem_ + offset + row_0 * Base::BYTES_PER_ROW_BEFORE_PACKING);
                b[ni].reg(0) = tmp.x;
                b[ni].reg(1) = tmp.y;
                b[ni].reg(2) = tmp.z;
                b[ni].reg(3) = tmp.w;

                fmha::lds(tmp, this->smem_ + offset + row_1 * Base::BYTES_PER_ROW_BEFORE_PACKING);
                b[ni].reg(4) = tmp.x;
                b[ni].reg(5) = tmp.y;
                b[ni].reg(6) = tmp.z;
                b[ni].reg(7) = tmp.w;
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, int BUFFERS_PER_TILE>
struct Smem_tile_v_turing_hmma
    : public Smem_tile_v_hmma<Traits, Cta_tile, Rows_per_xor_pattern_turing_b, BUFFERS_PER_TILE>::Base
{

    // The base class.
    using Base = typename Smem_tile_v_hmma<Traits, Cta_tile, Rows_per_xor_pattern_turing_b, BUFFERS_PER_TILE>::Base;
    // The MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;
    // The fragment.
    using Fragment = fmha::Fragment_b<Traits, fmha::Col>;

    // The size of a single LDS in bytes.
    enum
    {
        BYTES_PER_LDS = 16
    };

    // Ctor.
    inline __device__ Smem_tile_v_turing_hmma(void* smem, int tidx)
        : Base(smem, tidx)
    {

        // Warps.
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

        // Determine the config.
        enum
        {
            WARPS_2x1x2 = WARPS_M == 2 && WARPS_N == 1 && WARPS_K == 2
        };

        enum
        {
            WARPS_1x1x8 = WARPS_M == 1 && WARPS_N == 1 && WARPS_K == 8
        };

        enum
        {
            WARPS_1x1x4 = WARPS_M == 1 && WARPS_N == 1 && WARPS_K == 4
        };

        // Flash Attention uses WARPS_4x1x1
        enum
        {
            WARPS_4x1x1 = WARPS_M == 4 && WARPS_N == 1 && WARPS_K == 1
        };

        // The row/col read by the thread.
        int read_row, read_col;

        // SEQLEN == 128 and N == 16.
        if (WARPS_2x1x2 && Cta_tile::N == 16)
        {
            read_row = (tidx & 0x40) / 16 + (tidx & 0x04) / 4;
            read_col = (tidx & 0x03) * 2 + (tidx & 0x04) / 4;

            // SEQLEN == 128 and N == 32.
        }
        else if (WARPS_2x1x2 && Cta_tile::N == 32)
        {
            read_row = (tidx & 0x40) / 8 + (tidx & 0x06) / 2;
            read_col = (tidx & 0x01) * 4 + (tidx & 0x06) / 2;

            // SEQLEN == 128 and N == 64.
        }
        else if (WARPS_2x1x2 && Cta_tile::N == 64)
        {
            read_row = (tidx & 0x40) / 4 + (tidx & 0x07);
            read_col = (tidx & 0x07);

            // SEQLEN == 256, 512 and N == 16.
        }
        else if ((WARPS_1x1x8 || WARPS_1x1x4) && Cta_tile::N == 16)
        {
            read_row = (tidx & 0xe0) / 8 + (tidx & 0x04) / 4;
            read_col = (tidx & 0x03) * 2 + (tidx & 0x04) / 4;

            // SEQLEN == 256, 512 and N == 32.
        }
        else if ((WARPS_1x1x8 || WARPS_1x1x4) && Cta_tile::N == 32)
        {
            read_row = (tidx & 0xe0) / 4 + (tidx & 0x06) / 2;
            read_col = (tidx & 0x01) * 4 + (tidx & 0x06) / 2;

            // SEQLEN == 256, 384, 512 and N == 64, 128, 256.
        }
        else if ((WARPS_1x1x8 || WARPS_1x1x4) && (Cta_tile::N == 64 || Cta_tile::N == 128 || Cta_tile::N == 256))
        {
            read_row = (tidx & 0xe0) / 2 + (tidx & 0x07);
            read_col = (tidx & 0x07);

            // ANY SEQLEN and N == 16.
        }
        else if (WARPS_4x1x1 && Cta_tile::N == 16)
        {
            read_row = (tidx & 0x04) / 4;
            read_col = (tidx & 0x03) * 2 + (tidx & 0x04) / 4;

            // ANY SEQLEN and N == 32.
        }
        else if (WARPS_4x1x1 && Cta_tile::N == 32)
        {
            read_row = (tidx & 0x06) / 2;
            read_col = (tidx & 0x01) * 4 + (tidx & 0x06) / 2;

            // ANY SEQLEN and N == 64/128/256.
        }
        else if ((WARPS_4x1x1) && (Cta_tile::N == 64 || Cta_tile::N == 128 || Cta_tile::N == 256))
        {
            read_row = (tidx & 0x07);
            read_col = (tidx & 0x07);

            // Not supported!
        }
        else
        {
            assert(false);
        }

        // The 2nd HMMA.
        read_col ^= (tidx & 0x08) / 8;

        // The shared memory offset.
        this->smem_read_offset_ = read_row * Base::BYTES_PER_ROW + read_col * BYTES_PER_LDS;
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&b)[Mma_tile::VALID_MMAS_N], int ki)
    {
#pragma unroll
        for (int ni = 0; ni < Mma_tile::MMAS_N; ++ni)
        {
            // The amount of row packing.
            enum
            {
                ROW_PACKING = Base::BYTES_PER_ROW / Base::BYTES_PER_ROW_BEFORE_PACKING
            };

            // Skip N paddings
            if (ni < Mma_tile::VALID_MMAS_N)
            {
                // For even values of k value we jump by 16*WARPS_K rows and for odd, we jump by 8 rows.
                int row = (ki / 2) * 16 * Cta_tile::WARPS_K / ROW_PACKING + (ki % 2) * 8 / ROW_PACKING;

                // Load the data using LDSM.MT88.2.
                uint2 tmp;
                fmha::ldsmt(tmp, this->smem_ + this->smem_read_offset_ + row * Base::BYTES_PER_ROW);
                b[ni].reg(0) = tmp.x;
                b[ni].reg(1) = tmp.y;
            }

            // Move to the next N position.
            if (Mma_tile::MMAS_N == 1)
            {
                ;
            }
            else if (Mma_tile::MMAS_N == 2)
            {
                this->smem_read_offset_ ^= BYTES_PER_LDS * 2;
            }
            else if (Mma_tile::MMAS_N == 4)
            {
                this->smem_read_offset_ ^= BYTES_PER_LDS * (ni % 2 == 0 ? 2 : 6);
            }
            else if (Mma_tile::MMAS_N == 8)
            {
                this->smem_read_offset_ ^= BYTES_PER_LDS * ((ni & 1) == 0 ? 2 : ((ni & 3) == 3 ? 14 : 6));
            }
            else if (Mma_tile::MMAS_N == 16)
            {
                this->smem_read_offset_ ^= BYTES_PER_LDS
                    * ((ni & 1) == 0          ? 2
                            : ((ni & 7) == 7) ? 30
                                              : (((ni & 3) == 3) ? 14 : 6));
            }
            else
            {
                assert(false);
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, int BUFFERS_PER_TILE>
struct Smem_tile_v<Turing_hmma_fp16_traits, Cta_tile, BUFFERS_PER_TILE>
    : public Smem_tile_v_turing_hmma<Turing_hmma_fp16_traits, Cta_tile, BUFFERS_PER_TILE>
{

    // The base class.
    using Base = Smem_tile_v_turing_hmma<Turing_hmma_fp16_traits, Cta_tile, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_v(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, int BUFFERS_PER_TILE>
struct Smem_tile_v<Turing_hmma_fp32_traits, Cta_tile, BUFFERS_PER_TILE>
    : public Smem_tile_v_turing_hmma<Turing_hmma_fp32_traits, Cta_tile, BUFFERS_PER_TILE>
{

    // The base class.
    using Base = Smem_tile_v_turing_hmma<Turing_hmma_fp32_traits, Cta_tile, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_v(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, template <class, int> class Rows_per_xor_pattern,
    int BUFFERS_PER_TILE = 1>
struct Smem_tile_v_imma
{
    using Base = Smem_tile_without_skews<Cta_tile, Cta_tile::K, Cta_tile::N, 8, 16, BUFFERS_PER_TILE, 0,
        Rows_per_xor_pattern<Traits, Cta_tile::N>::VALUE, 1>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, int BUFFERS_PER_TILE>
struct Smem_tile_v<Volta_imma_int8_int32_traits, Cta_tile, BUFFERS_PER_TILE>
    : public Smem_tile_v_imma<Volta_imma_int8_int32_traits, Cta_tile, Rows_per_xor_pattern_volta_b,
          BUFFERS_PER_TILE>::Base
{

    // The traits class.
    using Traits = Volta_imma_int8_int32_traits;
    // The base class.
    using Base = typename Smem_tile_v_imma<Traits, Cta_tile, Rows_per_xor_pattern_volta_b, BUFFERS_PER_TILE>::Base;

    // DEBUG.
    static_assert(Base::BYTES_PER_ROW / Base::BYTES_PER_ROW_BEFORE_PACKING == 2, "");
    // END OF DEBUG.

    // The MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;
    // The fragment.
    using Fragment = fmha::Fragment_b<Traits, fmha::Col>;

    // The size of a single LDS in bytes.
    enum
    {
        BYTES_PER_LDS = 16
    };

    // Ctor.
    inline __device__ Smem_tile_v(void* smem, int tidx)
        : Base(smem, tidx)
    {

        // The row/col read by the thread.
        int read_row, read_col;

        // Warps.
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

        // Determine the config.
        enum
        {
            WARPS_2x1x2 = WARPS_M == 2 && WARPS_N == 1 && WARPS_K == 2
        };

        enum
        {
            WARPS_1x1x8 = WARPS_M == 1 && WARPS_N == 1 && WARPS_K == 8
        };

        enum
        {
            WARPS_1x1x4 = WARPS_M == 1 && WARPS_N == 1 && WARPS_K == 4
        };

        // SEQLEN == 128 and N == 16.
        if (WARPS_2x1x2 && Cta_tile::N == 16)
        {
            read_row = (tidx & 0x40) / 32 + (tidx & 0x08) / 8;
            read_col = (tidx & 0x07);

            // SEQLEN == 128 and N == 32.
        }
        else if (WARPS_2x1x2 && Cta_tile::N == 32)
        {
            read_row = (tidx & 0x40) / 16 + (tidx & 0x0c) / 4;
            read_col = (tidx & 0x03) * 2 + (tidx & 0x04) / 4;

            // SEQLEN == 128 and N == 64.
        }
        else if (WARPS_2x1x2 && Cta_tile::N == 64)
        {
            read_row = (tidx & 0x40) / 8 + (tidx & 0x0e) / 2;
            read_col = (tidx & 0x01) * 4 + (tidx & 0x06) / 2;

            // SEQLEN == 256, 512 and N == 16.
        }
        else if ((WARPS_1x1x8 || WARPS_1x1x4) && Cta_tile::N == 16)
        {
            read_row = (tidx & 0xe0) / 16 + (tidx & 0x08) / 8;
            read_col = (tidx & 0x07);

            // SEQLEN == 256, 512 and N == 32.
        }
        else if ((WARPS_1x1x8 || WARPS_1x1x4) && Cta_tile::N == 32)
        {
            read_row = (tidx & 0xe0) / 8 + (tidx & 0x0c) / 4;
            read_col = (tidx & 0x03) * 2 + (tidx & 0x04) / 4;

            // SEQLEN == 256, 384, 512 and N == 64.
        }
        else if ((WARPS_1x1x8 || WARPS_1x1x4) && Cta_tile::N == 64)
        {
            read_row = (tidx & 0xe0) / 4 + (tidx & 0x0e) / 2;
            read_col = (tidx & 0x01) * 4 + (tidx & 0x06) / 2;

            // Not supported.
        }
        else
        {
            assert(false);
        }

        // The shared memory offset.
        this->smem_read_offset_ = read_row * Base::BYTES_PER_ROW + read_col * BYTES_PER_LDS;
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&b)[Mma_tile::VALID_MMAS_N], int ki)
    {
        static_assert(
            Mma_tile::MMAS_K == 2 || Mma_tile::MMAS_K == 3 || Mma_tile::MMAS_K == 4 || Mma_tile::MMAS_K == 6, "");
#pragma unroll
        for (int ni = 0; ni < Mma_tile::MMAS_N; ++ni)
        {

            // The amount of row packing.
            enum
            {
                ROW_PACKING = Base::BYTES_PER_ROW / Base::BYTES_PER_ROW_BEFORE_PACKING
            };

            // Skip N paddings
            if (ni < Mma_tile::VALID_MMAS_N)
            {

                // Jump by 8*16 rows per K but account for packing.
                int row = ki * 16 * Cta_tile::WARPS_K / ROW_PACKING;

                // We emulate the Turing logic, which loads the data using LDSM.MT88.2:
                // uint2 tmp;
                // fmha::ldsmt(tmp, this->smem_ + this->smem_read_offset_ + row * Base::BYTES_PER_ROW);
                // this call fetches two 8x16 matrices, stacked on top of each other

                // we fake LDSM.MT88.2, with 2 LDS.128 and a shuffle:
                // - T 0 - T 7 have the smem addresses of LDSM 0, each should do 16B loads
                // - T 8 - T15 have the smem addresses of LSDM 1, each should do 16B loads
                int const lane = threadIdx.x % Cta_tile::THREADS_PER_WARP;

                uint4 tmp16{0, 0, 0, 0}; // 16B

                if (lane < 16)
                {
                    fmha::lds(tmp16, this->smem_ + this->smem_read_offset_ + row * Base::BYTES_PER_ROW);
                }

                uint16_t* tmp16c = reinterpret_cast<uint16_t*>(&tmp16); // 8x2B: we move pairs

                uint2 tmp;                                              // 2*4B
                uint16_t* t = reinterpret_cast<uint16_t*>(&tmp);        // 4x2B

                int const src_col = lane / 4;                           // 0 - 7
                int const src_row = lane % 4 * 2;

// We have to shuffle the values to distribute them in the warp.
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

                // Repack the elements. With LDSM.T, thread 0 has the following elements in its two
                // regs:
                //
                //   R0 = [(n=0 k=0), (n=1 k=0), (n=0 k=8), (n=1 k=8)]
                //   R1 = [(n=0 k=1), (n=1 k=1), (n=0 k=9), (n=1 k=9)]
                //
                // We want to repack the values as:
                //
                //   R0 = [(n=0 k=0), (n=0 k=1), (n=0 k=8), (n=0 k=9)]
                //   R1 = [(n=1 k=0), (n=1 k=1), (n=1 k=8), (n=1 k=9)]
                //
                // Since that this layout corresponds to the layout of elements in the Fragment_a from
                // P.

                swizzle_rows(b[ni].reg(0), b[ni].reg(1), tmp.x, tmp.y);
            }

            // Move to the next N position.
            if (Mma_tile::MMAS_N == 4)
            {
                this->smem_read_offset_ ^= BYTES_PER_LDS * (ni % 2 == 0 ? 1 : 3);
            }
            else
            {
                assert(false);
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, int BUFFERS_PER_TILE>
struct Smem_tile_v<Turing_imma_int8_int32_traits, Cta_tile, BUFFERS_PER_TILE>
    : public Smem_tile_v_imma<Turing_imma_int8_int32_traits, Cta_tile, Rows_per_xor_pattern_turing_b,
          BUFFERS_PER_TILE>::Base
{

    // The traits class.
    using Traits = Turing_imma_int8_int32_traits;
    // The base class.
    using Base = typename Smem_tile_v_imma<Traits, Cta_tile, Rows_per_xor_pattern_turing_b, BUFFERS_PER_TILE>::Base;
    // The MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;
    // The fragment.
    using Fragment = fmha::Fragment_b<Traits, fmha::Col>;

    // The size of a single LDS in bytes.
    enum
    {
        BYTES_PER_LDS = 16
    };

    // Ctor.
    inline __device__ Smem_tile_v(void* smem, int tidx)
        : Base(smem, tidx)
    {

        // Warps.
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

        // Determine the config.
        enum
        {
            WARPS_2x1x2 = WARPS_M == 2 && WARPS_N == 1 && WARPS_K == 2
        };

        enum
        {
            WARPS_1x1x8 = WARPS_M == 1 && WARPS_N == 1 && WARPS_K == 8
        };

        enum
        {
            WARPS_1x1x4 = WARPS_M == 1 && WARPS_N == 1 && WARPS_K == 4
        };

        // The row/col read by the thread.
        int read_row, read_col;

        // SEQLEN == 128 and N == 32.
        if (WARPS_2x1x2 && Cta_tile::N == 16)
        {
            read_row = (tidx & 0x40) / 32 + (tidx & 0x08) / 8;
            read_col = (tidx & 0x07);

            // SEQLEN == 128 and N == 32.
        }
        else if (WARPS_2x1x2 && Cta_tile::N == 32)
        {
            read_row = (tidx & 0x40) / 16 + (tidx & 0x0c) / 4;
            read_col = (tidx & 0x03) * 2 + (tidx & 0x04) / 4;

            // SEQLEN == 128 and N == 64.
        }
        else if (WARPS_2x1x2 && Cta_tile::N == 64)
        {
            read_row = (tidx & 0x40) / 8 + (tidx & 0x0e) / 2;
            read_col = (tidx & 0x01) * 4 + (tidx & 0x06) / 2;

            // SEQLEN == 256, 512 and N == 16.
        }
        else if ((WARPS_1x1x8 || WARPS_1x1x4) && Cta_tile::N == 16)
        {
            read_row = (tidx & 0xe0) / 16 + (tidx & 0x08) / 8;
            read_col = (tidx & 0x07);

            // SEQLEN == 256, 512 and N == 32.
        }
        else if ((WARPS_1x1x8 || WARPS_1x1x4) && Cta_tile::N == 32)
        {
            read_row = (tidx & 0xe0) / 8 + (tidx & 0x0c) / 4;
            read_col = (tidx & 0x03) * 2 + (tidx & 0x04) / 4;

            // SEQLEN == 256, 384, 512 and N == 64.
        }
        else if ((WARPS_1x1x8 || WARPS_1x1x4) && Cta_tile::N == 64)
        {
            read_row = (tidx & 0xe0) / 4 + (tidx & 0x0e) / 2;
            read_col = (tidx & 0x01) * 4 + (tidx & 0x06) / 2;

            // Not supported.
        }
        else
        {
            assert(false);
        }

        // The shared memory offset.
        this->smem_read_offset_ = read_row * Base::BYTES_PER_ROW + read_col * BYTES_PER_LDS;
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&b)[Mma_tile::VALID_MMAS_N], int ki)
    {
        static_assert(Mma_tile::MMAS_K == 2 || Mma_tile::MMAS_K == 3 || Mma_tile::MMAS_K == 4 || Mma_tile::MMAS_K == 6
                || Mma_tile::MMAS_K == 8,
            "");
#pragma unroll
        for (int ni = 0; ni < Mma_tile::MMAS_N; ++ni)
        {
            // The amount of row packing.
            enum
            {
                ROW_PACKING = Base::BYTES_PER_ROW / Base::BYTES_PER_ROW_BEFORE_PACKING
            };

            // Skip N paddings
            if (ni < Mma_tile::VALID_MMAS_N)
            {
                // Jump by 8*16 rows per K but account for packing.
                int row = ki * 16 * Cta_tile::WARPS_K / ROW_PACKING;

                // Load the data using LDSM.MT88.2.
                uint2 tmp;
                fmha::ldsmt(tmp, this->smem_ + this->smem_read_offset_ + row * Base::BYTES_PER_ROW);

                // Repack the elements. With LDSM.T, thread 0 has the following elements in its two
                // regs:
                //
                //   R0 = [(n=0 k=0), (n=1 k=0), (n=0 k=8), (n=1 k=8)]
                //   R1 = [(n=0 k=1), (n=1 k=1), (n=0 k=9), (n=1 k=9)]
                //
                // We want to repack the values as:
                //
                //   R0 = [(n=0 k=0), (n=0 k=1), (n=0 k=8), (n=0 k=9)]
                //   R1 = [(n=1 k=0), (n=1 k=1), (n=1 k=8), (n=1 k=9)]
                //
                // Since that this layout corresponds to the layout of elements in the Fragment_a from
                // P.

                swizzle_rows(b[ni].reg(0), b[ni].reg(1), tmp.x, tmp.y);

                // b[ni].reg(0) = tmp.x;
                // b[ni].reg(1)=  tmp.y;
            }

            // Move to the next N position.
            if (Mma_tile::MMAS_N == 1)
            {
                // Noop.
            }
            else if (Mma_tile::MMAS_N == 2)
            {
                this->smem_read_offset_ ^= BYTES_PER_LDS;
            }
            else if (Mma_tile::MMAS_N == 4)
            {
                this->smem_read_offset_ ^= BYTES_PER_LDS * (ni % 2 == 0 ? 1 : 3);
            }
            else
            {
                assert(false);
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, int BUFFERS_PER_TILE>
struct Smem_tile_v_ampere_hmma
    : public Smem_tile_v_hmma<Traits, Cta_tile, Rows_per_xor_pattern_ampere_b, BUFFERS_PER_TILE>::Base
{

    // The base class.
    using Base = typename Smem_tile_v_hmma<Traits, Cta_tile, Rows_per_xor_pattern_ampere_b, BUFFERS_PER_TILE>::Base;
    // The MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;
    // The fragment.
    using Fragment = fmha::Fragment_b<Traits, fmha::Col>;

    // The size of a single LDS in bytes.
    enum
    {
        BYTES_PER_LDS = 16
    };

    // Ctor.
    inline __device__ Smem_tile_v_ampere_hmma(void* smem, int tidx)
        : Base(smem, tidx)
    {

        // Warps.
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

        // Determine the config.
        enum
        {
            WARPS_2x1x2 = WARPS_M == 2 && WARPS_N == 1 && WARPS_K == 2
        };

        enum
        {
            WARPS_1x1x8 = WARPS_M == 1 && WARPS_N == 1 && WARPS_K == 8
        };

        enum
        {
            WARPS_1x1x4 = WARPS_M == 1 && WARPS_N == 1 && WARPS_K == 4
        };

        // Flash Attention uses WARPS_4x1x1
        enum
        {
            WARPS_4x1x1 = WARPS_M == 4 && WARPS_N == 1 && WARPS_K == 1
        };

        // The row/col read by the thread.
        int read_row, read_col;

        // SEQLEN == 128 and N == 16.
        if (WARPS_2x1x2 && Cta_tile::N == 16)
        {
            read_row = (tidx & 0x40) / 16 + (tidx & 0x0c) / 4;
            read_col = (tidx & 0x03) * 2 + (tidx & 0x04) / 4;

            // SEQLEN == 128 and N == 32.
        }
        else if (WARPS_2x1x2 && Cta_tile::N == 32)
        {
            read_row = (tidx & 0x40) / 8 + (tidx & 0x0e) / 2;
            read_col = (tidx & 0x01) * 4 + (tidx & 0x06) / 2;

            // SEQLEN == 128 and N == 64/128/256.
        }
        else if (WARPS_2x1x2 && (Cta_tile::N == 64 || Cta_tile::N == 128 || Cta_tile::N == 256))
        {
            read_row = (tidx & 0x40) / 4 + (tidx & 0x0f);
            read_col = (tidx & 0x07);

            // SEQLEN == 256, 512 and N == 16.
        }
        else if ((WARPS_1x1x8 || WARPS_1x1x4) && Cta_tile::N == 16)
        {
            read_row = (tidx & 0xe0) / 8 + (tidx & 0x0c) / 4;
            read_col = (tidx & 0x03) * 2 + (tidx & 0x04) / 4;

            // SEQLEN == 256, 512 and N == 32.
        }
        else if ((WARPS_1x1x8 || WARPS_1x1x4) && Cta_tile::N == 32)
        {
            read_row = (tidx & 0xe0) / 4 + (tidx & 0x0e) / 2;
            read_col = (tidx & 0x01) * 4 + (tidx & 0x06) / 2;

            // SEQLEN == 256, 384, 512 and N == 64/128/256.
        }
        else if ((WARPS_1x1x8 || WARPS_1x1x4) && (Cta_tile::N == 64 || Cta_tile::N == 128 || Cta_tile::N == 256))
        {
            read_row = (tidx & 0xe0) / 2 + (tidx & 0x0f);
            read_col = (tidx & 0x07);

            // ANY SEQLEN and N == 16.
        }
        else if (WARPS_4x1x1 && Cta_tile::N == 16)
        {
            read_row = (tidx & 0x0c) / 4;
            read_col = (tidx & 0x03) * 2 + (tidx & 0x04) / 4;

            // ANY SEQLEN and N == 32.
        }
        else if (WARPS_4x1x1 && Cta_tile::N == 32)
        {
            read_row = (tidx & 0x0e) / 2;
            read_col = (tidx & 0x01) * 4 + (tidx & 0x06) / 2;

            // ANY SEQLEN and N == 64/128/256.
        }
        else if (WARPS_4x1x1 && (Cta_tile::N == 64 || Cta_tile::N == 128 || Cta_tile::N == 256 || Cta_tile::N == 512))
        {
            read_row = (tidx & 0x0f);
            read_col = (tidx & 0x07);

            // Not supported.
        }
        else
        {
            assert(false);
        }

        // The 2nd HMMA.
        read_col ^= (tidx & 0x10) / 16;

        // The shared memory offset.
        this->smem_read_offset_ = read_row * Base::BYTES_PER_ROW + read_col * BYTES_PER_LDS;
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&b)[Mma_tile::VALID_MMAS_N], int ki)
    {
#pragma unroll
        for (int ni = 0; ni < Mma_tile::MMAS_N; ++ni)
        {
            // The amount of row packing.
            enum
            {
                ROW_PACKING = Base::BYTES_PER_ROW / Base::BYTES_PER_ROW_BEFORE_PACKING
            };

            // Jump by 16 * #warps row. Account for the packing.
            int row = ki * 16 * Cta_tile::WARPS_K / ROW_PACKING;

            // Skip N paddings
            if (ni < Mma_tile::VALID_MMAS_N)
            {
                // Jump by 16 * #warps row. Account for the packing.
                int row = ki * 16 * Cta_tile::WARPS_K / ROW_PACKING;

                // Load the data using LDSM.MT88.2.
                uint4 tmp;
                fmha::ldsmt(
                    tmp, this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ + row * Base::BYTES_PER_ROW);
                b[ni].reg(0) = tmp.x;
                b[ni].reg(1) = tmp.y;
                b[ni].reg(2) = tmp.z;
                b[ni].reg(3) = tmp.w;
            }

            // Move the pointer for the next ni. I expect the compiler to not recompute those.
            static_assert(Mma_tile::MMAS_N <= 64, "");
            if (Mma_tile::MMAS_N >= 32 && ni % 16 == 15)
            {
                this->smem_read_offset_ ^= 31 * BYTES_PER_LDS * 2;
            }
            else if (Mma_tile::MMAS_N >= 16 && ni % 8 == 7)
            {
                this->smem_read_offset_ ^= 15 * BYTES_PER_LDS * 2;
            }
            else if (Mma_tile::MMAS_N >= 8 && ni % 4 == 3)
            {
                this->smem_read_offset_ ^= 7 * BYTES_PER_LDS * 2;
            }
            else if (Mma_tile::MMAS_N >= 4 && ni % 2 == 1)
            {
                this->smem_read_offset_ ^= 3 * BYTES_PER_LDS * 2;
            }
            else if (Mma_tile::MMAS_N >= 2)
            {
                this->smem_read_offset_ ^= 1 * BYTES_PER_LDS * 2;
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, int BUFFERS_PER_TILE>
struct Smem_tile_v<Ampere_hmma_fp16_traits, Cta_tile, BUFFERS_PER_TILE>
    : public Smem_tile_v_ampere_hmma<Ampere_hmma_fp16_traits, Cta_tile, BUFFERS_PER_TILE>
{

    // The base class.
    using Base = Smem_tile_v_ampere_hmma<Ampere_hmma_fp16_traits, Cta_tile, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_v(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, int BUFFERS_PER_TILE>
struct Smem_tile_v<Ampere_hmma_fp32_traits, Cta_tile, BUFFERS_PER_TILE>
    : public Smem_tile_v_ampere_hmma<Ampere_hmma_fp32_traits, Cta_tile, BUFFERS_PER_TILE>
{

    // The base class.
    using Base = Smem_tile_v_ampere_hmma<Ampere_hmma_fp32_traits, Cta_tile, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_v(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, int BUFFERS_PER_TILE>
struct Smem_tile_v<Ampere_hmma_bf16_traits, Cta_tile, BUFFERS_PER_TILE>
    : public Smem_tile_v_ampere_hmma<Ampere_hmma_bf16_traits, Cta_tile, BUFFERS_PER_TILE>
{

    // The base class.
    using Base = Smem_tile_v_ampere_hmma<Ampere_hmma_bf16_traits, Cta_tile, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_v(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

template <typename Traits, typename Cta_tile, int BUFFERS_PER_TILE>
struct Smem_tile_v_ampere_8bit_mma
    : public Smem_tile_v_imma<Traits, Cta_tile, Rows_per_xor_pattern_ampere_b, BUFFERS_PER_TILE>::Base
{
    // The base class.
    using Base = typename Smem_tile_v_imma<Traits, Cta_tile, Rows_per_xor_pattern_ampere_b, BUFFERS_PER_TILE>::Base;
    // The MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;
    // The fragment.
    using Fragment = fmha::Fragment_b<Traits, fmha::Col>;

    // The size of a single LDS in bytes.
    enum
    {
        BYTES_PER_LDS = 16
    };

    // Ctor.
    inline __device__ Smem_tile_v_ampere_8bit_mma(void* smem, int tidx)
        : Base(smem, tidx)
    {

        // Warps.
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

        // Determine the config.
        enum
        {
            WARPS_2x1x2 = WARPS_M == 2 && WARPS_N == 1 && WARPS_K == 2
        };

        enum
        {
            WARPS_1x1x8 = WARPS_M == 1 && WARPS_N == 1 && WARPS_K == 8
        };

        enum
        {
            WARPS_1x1x4 = WARPS_M == 1 && WARPS_N == 1 && WARPS_K == 4
        };

        enum
        {
            WARPS_4x1x1 = WARPS_M == 4 && WARPS_N == 1 && WARPS_K == 1
        };

        // The row/col read by the thread.
        int read_row, read_col;

        // SEQLEN == 128 and N == 16.
        if (WARPS_2x1x2 && Cta_tile::N == 16)
        {
            read_row = (tidx & 0x40) / 32 + (tidx & 0x08) / 8;
            read_col = (tidx & 0x07);

            // SEQLEN == 128 and N == 32.
        }
        else if (WARPS_2x1x2 && Cta_tile::N == 32)
        {
            read_row = (tidx & 0x40) / 16 + (tidx & 0x0c) / 4;
            read_col = (tidx & 0x03) * 2 + (tidx & 0x04) / 4;

            // SEQLEN == 128 and N == 64.
        }
        else if (WARPS_2x1x2 && Cta_tile::N == 64)
        {
            read_row = (tidx & 0x40) / 8 + (tidx & 0x0e) / 2;
            read_col = (tidx & 0x01) * 4 + (tidx & 0x06) / 2;

            // SEQLEN == 256, 512 and N == 16.
        }
        else if ((WARPS_1x1x8 || WARPS_1x1x4) && Cta_tile::N == 16)
        {
            read_row = (tidx & 0xe0) / 16 + (tidx & 0x08) / 8;
            read_col = (tidx & 0x07);

            // SEQLEN == 256, 512 and N == 32.
        }
        else if ((WARPS_1x1x8 || WARPS_1x1x4) && Cta_tile::N == 32)
        {
            read_row = (tidx & 0xe0) / 8 + (tidx & 0x0c) / 4;
            read_col = (tidx & 0x03) * 2 + (tidx & 0x04) / 4;

            // SEQLEN == 256, 384, 512 and N == 64.
        }
        else if ((WARPS_1x1x8 || WARPS_1x1x4) && Cta_tile::N == 64)
        {
            read_row = (tidx & 0xe0) / 4 + (tidx & 0x0e) / 2;
            read_col = (tidx & 0x01) * 4 + (tidx & 0x06) / 2;
        }
        else if (WARPS_4x1x1 && Cta_tile::N == 32)
        {
            read_row = (tidx % 32) / 4;
            read_col = read_row % 2 + (tidx % 4) * 2;
        }
        else if (WARPS_4x1x1 && Cta_tile::N == 64)
        {
            read_row = (tidx % 32) / 2;
            read_col = read_row % 4 + (tidx & 0x01) * 4;
        }
        else if (WARPS_4x1x1 && (Cta_tile::N >= 128))
        {
            read_row = tidx % 32;
            read_col = tidx % 8;

            // Not supported.
        }
        else
        {
            assert(false);
        }

        // The shared memory offset.
        this->smem_read_offset_ = read_row * Base::BYTES_PER_ROW + read_col * BYTES_PER_LDS;
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&b)[Mma_tile::VALID_MMAS_N], int ki)
    {
// static_assert(Mma_tile::MMAS_K == 3 || Mma_tile::MMAS_K == 2 || Mma_tile::MMAS_K == 1, "");
#pragma unroll
        for (int ni = 0; ni < Mma_tile::MMAS_N; ++ni)
        {
            // The amount of row packing.
            enum
            {
                ROW_PACKING = Base::BYTES_PER_ROW / Base::BYTES_PER_ROW_BEFORE_PACKING
            };

            // // Make sure we do not end up with weird values :)
            // static_assert(Cta_tile::WARPS_K % ROW_PACKING == 0, "");

            // Skip N paddings
            if (ni < Mma_tile::VALID_MMAS_N)
            {

                // Jump by 8*32 rows per K but account for the fact that we have packing.
                int row_0 = (ki * 32 + 0 * 16) * Cta_tile::WARPS_K / ROW_PACKING;
                int row_1 = (ki * 32 + 1 * 16) * Cta_tile::WARPS_K / ROW_PACKING;

                // Load the data using LDSM.MT88.2.
                uint32_t smem = this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_;
                uint2 tmp_0;
                fmha::ldsmt(tmp_0, smem + row_0 * Base::BYTES_PER_ROW);

                // Load the next two values.
                uint2 tmp_1 = make_uint2(0u, 0u);
                if constexpr (Cta_tile::K > 16)
                {
                    fmha::ldsmt(tmp_1, smem + row_1 * Base::BYTES_PER_ROW);
                }

                // Repack the elements. With LDSM.T, thread 0 has the following elements in its 4 regs:
                //
                //   R0 = [(n=0 k=  0), (n=1 k=  0), (n=0 k=  1), (n=1 k=  1)]
                //   R1 = [(n=0 k=  8), (n=1 k=  8), (n=0 k=  9), (n=1 k=  9)]
                //   R2 = [(n=0 k=128), (n=1 k=128), (n=0 k=129), (n=1 k=129)]
                //   R3 = [(n=0 k=136), (n=1 k=136), (n=0 k=137), (n=1 k=137)]
                //
                // We want to repack the values as:
                //
                //   R0 = [(n=0 k=  0), (n=0 k=  1), (n=0 k=  8), (n=0 k=  9)]
                //   R1 = [(n=0 k=128), (n=0 k=129), (n=0 k=136), (n=0 k=137)]
                //   R2 = [(n=1 k=  0), (n=1 k=  1), (n=1 k=  8), (n=1 k=  9)]
                //   R3 = [(n=1 k=128), (n=1 k=129), (n=1 k=136), (n=1 k=137)]
                //
                // Since this layout corresponds to the layout of elements in the Fragment_a from P.

                swizzle_rows(b[ni].reg(0), b[ni].reg(2), tmp_0.x, tmp_0.y);
                swizzle_rows(b[ni].reg(1), b[ni].reg(3), tmp_1.x, tmp_1.y);
            }

            // Move to the next N position.
            if (Mma_tile::MMAS_N >= 32 && ni % 16 == 15)
            {
                this->smem_read_offset_ ^= 31 * BYTES_PER_LDS;
            }
            else if (Mma_tile::MMAS_N >= 16 && ni % 8 == 7)
            {
                this->smem_read_offset_ ^= 15 * BYTES_PER_LDS;
            }
            else if (Mma_tile::MMAS_N >= 8 && ni % 4 == 3)
            {
                this->smem_read_offset_ ^= 7 * BYTES_PER_LDS;
            }
            else if (Mma_tile::MMAS_N >= 4 && ni % 2 == 1)
            {
                this->smem_read_offset_ ^= 3 * BYTES_PER_LDS;
            }
            else if (Mma_tile::MMAS_N >= 2)
            {
                this->smem_read_offset_ ^= 1 * BYTES_PER_LDS;
            }
            else
            {
                assert(false);
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, int BUFFERS_PER_TILE>
struct Smem_tile_v<Ampere_imma_int8_int32_traits, Cta_tile, BUFFERS_PER_TILE>
    : public Smem_tile_v_ampere_8bit_mma<Ampere_imma_int8_int32_traits, Cta_tile, BUFFERS_PER_TILE>
{

    // The base class.
    using Base = Smem_tile_v_ampere_8bit_mma<Ampere_imma_int8_int32_traits, Cta_tile, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_v(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, int BUFFERS_PER_TILE>
struct Smem_tile_v<Ada_qmma_e4m3_fp32_traits, Cta_tile, BUFFERS_PER_TILE>
    : public Smem_tile_v_ampere_8bit_mma<Ada_qmma_e4m3_fp32_traits, Cta_tile, BUFFERS_PER_TILE>
{

    // The base class.
    using Base = Smem_tile_v_ampere_8bit_mma<Ada_qmma_e4m3_fp32_traits, Cta_tile, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_v(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, int BUFFERS_PER_TILE>
struct Smem_tile_v<Ada_qmma_e4m3_fp16_traits, Cta_tile, BUFFERS_PER_TILE>
    : public Smem_tile_v_ampere_8bit_mma<Ada_qmma_e4m3_fp16_traits, Cta_tile, BUFFERS_PER_TILE>
{

    // The base class.
    using Base = Smem_tile_v_ampere_8bit_mma<Ada_qmma_e4m3_fp16_traits, Cta_tile, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_v(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha
