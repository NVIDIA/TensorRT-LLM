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
#include <fmha/smem_tile.h>
#include <fmha/traits.h>
#include <fmha/utils.h>

namespace fmha
{

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// H M M A
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile>
struct Smem_tile_o
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Smem_tile_o<Volta_hmma_fp16_16x16x16_traits, Cta_tile>
{

    // The instruction traits.
    using Traits = Volta_hmma_fp16_16x16x16_traits;
    // The MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;
    // The accumulators.
    using Accumulator = fmha::Fragment_accumulator<Traits>;
    // The accumulators.
    using Data_type = typename Accumulator::Data_type;

    // The size of each element.
    enum
    {
        BYTES_PER_ELEMENT = sizeof(Data_type)
    };

    // The size of each STS.
    enum
    {
        BYTES_PER_STS = 16
    };

    // The size of each row in shared memory.
    enum
    {
        BYTES_PER_ROW = Cta_tile::N * Cta_tile::WARPS_K * 2 * BYTES_PER_ELEMENT
    };

    // The size of each LDS.
    enum
    {
        BYTES_PER_LDS = 16
    };

    // The number of threads (to produce 16B per LDS).
    enum
    {
        THREADS_PER_ROW = Cta_tile::N * BYTES_PER_ELEMENT / BYTES_PER_LDS
    };

    // The number of rows loaded per LDS.
    enum
    {
        ROWS_PER_LDS = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW
    };

    // The number of rows.
    enum
    {
        ROWS = Cta_tile::M
    };

    // We want at least one output per thread (if possible).
    enum
    {
        ROWS_PER_LOOP_ = ROWS <= 64 ? ROWS : (int) Min<ROWS, ROWS_PER_LDS>::VALUE
    };

    // We also want to have "complete" MMAs.
    enum
    {
        ROWS_PER_LOOP = Max<ROWS_PER_LOOP_, Mma_tile::M_PER_MMA_PER_CTA>::VALUE
    };

    // The number of outer loops.
    enum
    {
        LOOPS = fmha::Div_up<ROWS, ROWS_PER_LOOP>::VALUE
    };

    // Make sure it matches our expectations.
    static_assert(ROWS_PER_LOOP >= (int) Mma_tile::M_PER_MMA_PER_CTA, "");

    // Do we have to guard against partial writes/reads.
    enum
    {
        HAS_INCOMPLETE_LDS = ROWS_PER_LOOP % ROWS_PER_LDS != 0
    };

    // The total number of LDS per loop.
    enum
    {
        LDS_PER_LOOP = fmha::Div_up<ROWS_PER_LOOP, ROWS_PER_LDS>::VALUE
    };

    // The amount of shared memory.
    enum
    {
        BYTES_PER_TILE = ROWS_PER_LOOP * BYTES_PER_ROW
    };

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

    // Ctor.
    inline __device__ Smem_tile_o(void* smem, int tidx)
    {

        // Get a 32-bit value for the shared memory address.
        uint32_t smem_ = __nvvm_get_smem_pointer(smem);

        // The row/col written by the thread.
        int write_row, write_col;

        // SEQLEN == 128. Segments of 128B are written by 2 warps.
        if (WARPS_2x1x2 && Cta_tile::N == 32)
        {
            write_row = (tidx & 0x30) / 2 + (tidx & 0x07);
            write_col = (tidx & 0x0f);
            write_col ^= (tidx & 0x40) / 16;

            // SEQLEN == 128 and N == 64.
        }
        else if (WARPS_2x1x2 && Cta_tile::N == 64)
        {
            write_row = (tidx & 0x30) / 2 + (tidx & 0x07);
            write_col = (tidx & 0x40) / 8 + (tidx & 0x08) * 2 + (tidx & 0x07);

            // SEQLEN == 256, 384 and N == 32. Segments of 128B are written by 2 warps.
        }
        else if (WARPS_1x1x4 && Cta_tile::N == 32)
        {
            write_row = (tidx & 0x10) / 2 + (tidx & 0x07);
            write_col = (tidx & 0x40) / 8 + (tidx & 0x08) * 2 + (tidx & 0x07);
            write_col ^= (tidx & 0x20) / 8;

            // SEQLEN == 256, 384 and N == 64.
        }
        else if (WARPS_1x1x4 && Cta_tile::N == 64)
        {
            write_row = (tidx & 0x10) / 2 + (tidx & 0x07);
            write_col = (tidx & 0x60) / 4 + (tidx & 0x08) * 4 + (tidx & 0x07);

            // SEQLEN == 256, 384, 512 and N == 128.
        }
        else if (WARPS_1x1x4 && Cta_tile::N == 128)
        {
            write_row = (tidx & 0x10) / 2 + (tidx & 0x07);
            write_col = (tidx & 0x60) / 2 + (tidx & 0x08) * 8 + (tidx & 0x07);

            // SEQLEN == 256, 384, 512 and N == 256.
        }
        else if (WARPS_1x1x4 && Cta_tile::N == 256)
        {
            write_row = (tidx & 0x10) / 2 + (tidx & 0x07);
            write_col = (tidx & 0x60) / 1 + (tidx & 0x08) * 16 + (tidx & 0x07);

            // SEQLEN == 256, 384, 512 and N == 32. Segments of 128B are written by 2 warps.
        }
        else if (WARPS_1x1x8 && Cta_tile::N == 32)
        {
            write_row = (tidx & 0x10) / 2 + (tidx & 0x07);
            write_col = (tidx & 0xc0) / 8 + (tidx & 0x08) * 4 + (tidx & 0x07);
            write_col ^= (tidx & 0x20) / 8;

            // SEQLEN == 256, 384, 512 and N == 64.
        }
        else if (WARPS_1x1x8 && Cta_tile::N == 64)
        {
            write_row = (tidx & 0x10) / 2 + (tidx & 0x07);
            write_col = (tidx & 0xe0) / 4 + (tidx & 0x08) * 8 + (tidx & 0x07);

            // ANY SEQLEN and N == 32
        }
        else if (WARPS_4x1x1 && Cta_tile::N == 32)
        {
            write_row = (tidx & 0xf0) / 2 + (tidx & 0x07);
            write_col = (tidx & 0x07);
            write_col ^= (tidx & 0x08) / 2;

            // ANY SEQLEN and N == 64
        }
        else if (WARPS_4x1x1 && Cta_tile::N == 64)
        {
            write_row = (tidx & 0x70) / 2 + (tidx & 0x07);
            write_col = (tidx & 0x0f);

            // ANY SEQLEN and N == 128
        }
        else if (WARPS_4x1x1 && Cta_tile::N == 128)
        {
            write_row = (tidx & 0x70) / 2 + (tidx & 0x07);
            write_col = (tidx & 0x08) + (tidx & 0x0f);

            // ANY SEQLEN and N == 256
        }
        else if (WARPS_4x1x1 && Cta_tile::N == 256)
        {
            write_row = (tidx & 0x70) / 2 + (tidx & 0x07);
            write_col = (tidx & 0x08) * 3 + (tidx & 0x0f);

            // Not supported.
        }
        else
        {
            assert(false);
        }

        // Assemble the write pointer.
        smem_write_ = smem_ + write_row * BYTES_PER_ROW + write_col * BYTES_PER_STS;

        // The element read by each thread.
        int read_row = tidx / THREADS_PER_ROW;
        int read_col = tidx % THREADS_PER_ROW;

        // Take the XOR pattern into account for the column.
        read_col ^= read_row & 0x7;

        // Assemble the read pointer.
        smem_read_ = smem_ + read_row * BYTES_PER_ROW + read_col * BYTES_PER_LDS;

        // Is that thread active on the last LDS?
        if (HAS_INCOMPLETE_LDS)
        {
            is_active_for_last_lds_ = read_row + (LDS_PER_LOOP - 1) * ROWS_PER_LDS < Cta_tile::M;
        }
    }

    // Load the output fragments.
    inline __device__ void load(uint4 (&out)[LDS_PER_LOOP]) const
    {
        uint32_t local_smem_read_ = smem_read_;
#pragma unroll
        for (int ii = 0; ii < LDS_PER_LOOP; ++ii)
        {

            // Apply the XOR pattern if needed. (XOR 8 default)
            if (ROWS_PER_LDS < 8)
            {
                local_smem_read_ = (smem_read_ ^ ((ii * ROWS_PER_LDS) % 8 * BYTES_PER_LDS));
            }

            // Load the elements before the reduction (split-K).
            uint4 tmp[Cta_tile::WARPS_K * 2];
#pragma unroll
            for (int jj = 0; jj < Cta_tile::WARPS_K * 2; ++jj)
            {
                // The immediate.
                int imm = ii * ROWS_PER_LDS * BYTES_PER_ROW;
                if (Cta_tile::N == 256)
                {
                    imm += jj * 512;
                }
                else if (Cta_tile::N == 128)
                {
                    imm += jj * 256;
                }
                else if (Cta_tile::N == 64)
                {
                    imm += jj * 128;
                }
                else if (Cta_tile::N == 32)
                {
                    imm += jj / 2 * 128;
                }
                else
                {
                    assert(false);
                }

                // The XOR mask.
                int smem_read_offset = local_smem_read_;
                if (Cta_tile::N == 32 && (jj % 2) == 1)
                {
                    smem_read_offset ^= 64;
                }

                // Load...
                if (!HAS_INCOMPLETE_LDS || (ii < LDS_PER_LOOP - 1 || is_active_for_last_lds_))
                {
                    fmha::lds(tmp[jj], smem_read_offset + imm);
                }
            }

            // Perform the reduction.
            out[ii] = tmp[0];
#pragma unroll
            for (int jj = 1; jj < Cta_tile::WARPS_K * 2; ++jj)
            {
                out[ii] = fmha::hadd8(out[ii], tmp[jj]);
            }
        }
    }

    // Store the accumulators.
    template <int M, int N>
    inline __device__ void store(Accumulator const (&acc)[M][N], int mi)
    {

        enum
        {
            M_PER_MMA = Mma_tile::M_PER_MMA_PER_CTA
        };

#pragma unroll
        for (int ni = 0; ni < Mma_tile::VALID_MMAS_N; ++ni)
        {

            // The number of MMAs that are stored per loop iteration.
            enum
            {
                MMAS_M_PER_LOOP = Mma_tile::MMAS_M / LOOPS
            };

// Store 1st column of the different MMAs.
#pragma unroll
            for (int mj = 0; mj < MMAS_M_PER_LOOP; ++mj)
            {

                // Assemble the vectors for the stores. See how we swizzle the registers.
                uint4 tmp_0;
                tmp_0.x = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(0);
                tmp_0.y = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(1);
                tmp_0.z = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(4);
                tmp_0.w = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(5);

                uint4 tmp_1;
                tmp_1.x = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(2);
                tmp_1.y = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(3);
                tmp_1.z = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(6);
                tmp_1.w = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(7);

                // Precompute the immediates to jump to the correct row.
                int row = mj * M_PER_MMA * BYTES_PER_ROW;

                // The columns.
                int smem_write_0 = smem_write_ ^ ((2 * ni + 0) * BYTES_PER_STS);
                int smem_write_1 = smem_write_ ^ ((2 * ni + 1) * BYTES_PER_STS);

                // Store.
                fmha::sts(smem_write_0 + row, tmp_0);
                fmha::sts(smem_write_1 + row, tmp_1);
            }
        }
    }

    // The write pointer.
    uint32_t smem_write_;
    // The write pointer.
    uint32_t smem_read_;
    // Is the thread active for the last LDS of the series?
    int is_active_for_last_lds_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// This class converts the FP16/FP32 inputs to FP16x2.

struct Convert_from_fp16
{

    // Convert one pair of fp16 numbers.
    template <typename Accumulators>
    static inline __device__ uint32_t convert(Accumulators const& acc, int ii)
    {

        // Extract the 2x FP16 numbers (packed in a register).
        uint32_t h2 = acc.reg(ii);

        return h2;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Convert_from_fp32
{

    // Convert one pair of fp16 numbers.
    template <typename Accumulators>
    inline __device__ uint32_t convert(Accumulators const& acc, int ii)
    {

        // Extract the 2x floats.
        float f0 = acc.elt(ii * 2 + 0);
        float f1 = acc.elt(ii * 2 + 1);

        // Convert to FP16x2.
        return fmha::float2_to_half2(f0, f1);
    }

    // The bf16 accumulators (convert from fp32 to 2xbf16).
    using Ampere_bf16_Accumulator = fmha::Fragment_accumulator<Ampere_hmma_bf16_traits>;

    static inline __device__ uint32_t convert(Ampere_bf16_Accumulator const& acc, int ii)
    {

        // Extract the 2x floats.
        float f0 = acc.elt(ii * 2 + 0);
        float f1 = acc.elt(ii * 2 + 1);

        // Convert to FP16x2.
        return fmha::float2_to_bf16_x2(f0, f1);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, int BYTES_PER_STS_ = 4>
struct Hmma_smem_tile_o
{

    // The MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;
    // The accumulators.
    using Accumulator = fmha::Fragment_accumulator<Traits>;
    // The data type.
    using Data_type = typename Accumulator::Data_type;
    // The epilogue data type
    using Epilogue_type = typename Traits::Epilogue_type;

    // The size of each element.
    enum
    {
        BYTES_PER_ELEMENT = sizeof(Epilogue_type)
    };

    // The amount of bytes per row (without packing or split-k).
    enum
    {
        BYTES_PER_ROW = Cta_tile::N * BYTES_PER_ELEMENT
    };

    // The size of each STS.
    enum
    {
        BYTES_PER_STS = BYTES_PER_STS_
    };

    // The size of each LDS.
    enum
    {
        BYTES_PER_LDS = 16
    };

    // The number of threads (to produce 16B per LDS).
    enum
    {
        THREADS_PER_ROW = BYTES_PER_ROW / BYTES_PER_LDS
    };

    // The number of rows loaded per LDS.
    enum
    {
        ROWS_PER_LDS = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW
    };

    // The number of rows in shared memory.
    enum
    {
        ROWS = Cta_tile::M
    };

    // We want at least one output per thread (if possible).
    enum
    {
        ROWS_PER_LOOP_ = ROWS <= 64 ? ROWS : (int) Min<ROWS, ROWS_PER_LDS>::VALUE
    };

    // We also want to have "complete" MMAs.
    enum
    {
        ROWS_PER_LOOP = Max<ROWS_PER_LOOP_, Mma_tile::M_PER_MMA_PER_CTA>::VALUE
    };

    // The number of outer loops.
    enum
    {
        LOOPS = fmha::Div_up<ROWS, ROWS_PER_LOOP>::VALUE
    };

    // Make sure it matches our expectations.
    static_assert(ROWS_PER_LOOP >= (int) Mma_tile::M_PER_MMA_PER_CTA, "");

    // Do we have to guard against partial writes/reads.
    enum
    {
        HAS_INCOMPLETE_LDS = ROWS_PER_LOOP % ROWS_PER_LDS != 0
    };

    // The total number of LDS per loop.
    enum
    {
        LDS_PER_LOOP = fmha::Div_up<ROWS_PER_LOOP, ROWS_PER_LDS>::VALUE
    };

    // The amount of shared memory.
    enum
    {
        BYTES_PER_TILE = ROWS_PER_LOOP * BYTES_PER_ROW * Cta_tile::WARPS_K
    };

    // The amount of row packing to make sure we have at least 128B per smem row (without split-k).
    enum
    {
        ROW_PACKING = Max<1, 128 / BYTES_PER_ROW>::VALUE
    };

    // Make sure our row packing is correct
    static_assert(ROWS_PER_LOOP % ROW_PACKING == 0, "");

    // The amount of shared memory per row after packing.
    enum
    {
        BYTES_PER_ROW_WITH_PACKING = BYTES_PER_ROW * ROW_PACKING
    };

    // Make sure we have at least 128B per row after packing.
    static_assert(BYTES_PER_ROW_WITH_PACKING >= 128, "");

    // The number of threads per row after packing.
    enum
    {
        THREADS_PER_ROW_WITH_PACKING = THREADS_PER_ROW * ROW_PACKING
    };

    // Make sure we have at least 8 threads per row after packing.
    static_assert(THREADS_PER_ROW_WITH_PACKING >= 8, "");

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

    enum
    {
        WARPS_4x1x2 = WARPS_M == 4 && WARPS_N == 1 && WARPS_K == 2
    };

    // Ctor.
    inline __device__ Hmma_smem_tile_o(void* smem, int tidx)
    {

        // Get a 32-bit value for the shared memory address.
        uint32_t smem_ = __nvvm_get_smem_pointer(smem);

        // The row/col written by the thread.
        int write_row, write_col;

        // SEQLEN == 128 and HIDDEN_SIZE_PER_HEAD == 16.
        if (WARPS_2x1x2 && Cta_tile::N == 16)
        {
            write_row = (tidx & 0x20) / 8 + (tidx & 0x10) / 16;
            write_col = (tidx & 0x40) / 2 + (tidx & 0x0c) * 2 + (tidx & 0x03);
            write_col ^= (tidx & 0x10) / 4;

            // SEQLEN == 128 and HIDDEN_SIZE_PER_HEAD == 32.
        }
        else if (WARPS_2x1x2 && Cta_tile::N == 32)
        {
            write_row = (tidx & 0x20) / 4 + (tidx & 0x18) / 8;
            write_col = (tidx & 0x40) / 2 + (tidx & 0x04) * 4 + (tidx & 0x03);
            write_col ^= (tidx & 0x18) / 2;

            // SEQLEN == 128 and HIDDEN_SIZE_PER_HEAD == 64.
        }
        else if (WARPS_2x1x2 && Cta_tile::N == 64)
        {
            write_row = (tidx & 0x20) / 2 + (tidx & 0x1c) / 4;
            write_col = (tidx & 0x40) / 2 + (tidx & 0x03);
            write_col ^= (tidx & 0x1c);

            // SEQLEN == 128 and HIDDEN_SIZE_PER_HEAD == 128.
        }
        else if (WARPS_2x1x2 && Cta_tile::N == 128)
        {
            write_row = (tidx & 0x20) / 2 + (tidx & 0x1c) / 4;
            write_col = (tidx & 0x40) / 1 + (tidx & 0x1f);

            // SEQLEN == 256, 384, 512 and HIDDEN_SIZE_PER_HEAD == 16.
        }
        else if ((WARPS_1x1x8 || WARPS_1x1x4) && Cta_tile::N == 16)
        {
            write_row = (tidx & 0x10) / 16;
            write_col = (tidx & 0x0c) * 2 + (tidx & 0xe3);
            write_col ^= (tidx & 0x10) / 4;

            // SEQLEN == 256, 384, 512 and HIDDEN_SIZE_PER_HEAD == 32.
        }
        else if ((WARPS_1x1x8 || WARPS_1x1x4) && Cta_tile::N == 32)
        {
            write_row = (tidx & 0x18) / 8;
            write_col = (tidx & 0x04) * 4 + (tidx & 0xe3);
            write_col ^= (tidx & 0x18) / 2;

            // SEQLEN == 256, 384 and HIDDEN_SIZE_PER_HEAD == 64.
        }
        else if ((WARPS_1x1x8 || WARPS_1x1x4) && Cta_tile::N == 64)
        {
            write_row = (tidx & 0x1c) / 4;
            write_col = (tidx & 0xff);

            // SEQLEN == 256, 384 and HIDDEN_SIZE_PER_HEAD == 128.
        }
        else if ((WARPS_1x1x8 || WARPS_1x1x4) && Cta_tile::N == 128)
        {
            write_row = (tidx & 0x1c) / 4;
            write_col = (tidx & 0xe0) * 2 + (tidx & 0x1f);

            // SEQLEN == 256, 384 and HIDDEN_SIZE_PER_HEAD == 256.
        }
        else if ((WARPS_1x1x8 || WARPS_1x1x4) && Cta_tile::N == 256)
        {
            write_row = (tidx & 0x1c) / 4;
            write_col = (tidx & 0xe0) * 4 + (tidx & 0x1f);

            // ANY SEQLEN and HIDDEN_SIZE_PER_HEAD == 16.
        }
        else if (WARPS_4x1x1 && Cta_tile::N == 16)
        {
            write_row = (tidx & 0xe0) / 8 + (tidx & 0x10) / 16;
            write_col = (tidx & 0x0c) * 2 + (tidx & 0x03);
            write_col ^= (tidx & 0x10) / 4;

            // ANY SEQLEN and HIDDEN_SIZE_PER_HEAD == 32.
        }
        else if (WARPS_4x1x1 && Cta_tile::N == 32)
        {
            write_row = (tidx & 0xe0) / 4 + (tidx & 0x18) / 8;
            write_col = (tidx & 0x04) * 4 + (tidx & 0x03);
            write_col ^= (tidx & 0x18) / 2;

            // ANY SEQLEN and HIDDEN_SIZE_PER_HEAD == 64/128.
        }
        else if (WARPS_4x1x1 && (Cta_tile::N == 64 || Cta_tile::N == 128))
        {
            write_row = (tidx & 0x60) / 2 + (tidx & 0x1c) / 4;
            write_col = (tidx & 0x1f);

            // ANY SEQLEN and HIDDEN_SIZE_PER_HEAD == 256.
        }
        else if (WARPS_4x1x1 && (Cta_tile::N == 256 || Cta_tile::N == 512))
        {
            write_row = (tidx & 0x60) / 2 + (tidx & 0x1c) / 4;
            write_col = (tidx & 0x1f);

            // GMMA: S=284/512 and HIDDEN_SIZE_PER_HEAD == 64.
        }
        else if (WARPS_4x1x2 && Cta_tile::N == 64)
        {
            write_row = (tidx & 0x60) / 2 + (tidx & 0x1c) / 4;
            write_col = (tidx & 0x80) / 4 + (tidx & 0x03);
            write_col ^= (tidx & 0x1c);

            // GMMA: S=284/512 and HIDDEN_SIZE_PER_HEAD == 64.
        }
        else if (WARPS_4x1x2 && Cta_tile::N == 32)
        {
            write_row = (tidx & 0x60) / 4 + (tidx & 0x1c) / 8;
            write_col = (tidx & 0x80) / 4 + (tidx & 0x04) * 4 + (tidx & 0x03);
            write_col ^= (tidx & 0x18) / 2;

            // Not supported.
        }
        else
        {
            assert(false);
        }

        // Assemble the write pointer.
        smem_write_ = smem_ + write_row * BYTES_PER_ROW_WITH_PACKING * Cta_tile::WARPS_K + write_col * BYTES_PER_STS;

        // The element read by each thread.
        int read_row = tidx / THREADS_PER_ROW;
        int read_col = tidx % THREADS_PER_ROW;

        // Is that thread active on the last LDS?
        if (HAS_INCOMPLETE_LDS)
        {
            is_active_for_last_lds_ = read_row + (LDS_PER_LOOP - 1) * ROWS_PER_LDS < ROWS_PER_LOOP;
        }

        // The XOR params.
        int const XOR_MOD = 8 / ROW_PACKING;

        // Take the XOR pattern and the packing into account for the column.
        read_col += read_row % ROW_PACKING * XOR_MOD;
        read_row /= ROW_PACKING;
        read_col ^= read_row % XOR_MOD;

        // Assemble the read pointer.
        smem_read_ = smem_ + read_row * BYTES_PER_ROW_WITH_PACKING * Cta_tile::WARPS_K + read_col * BYTES_PER_LDS;
    }

    // Load the output fragments.
    inline __device__ void load(uint4 (&out)[LDS_PER_LOOP]) const
    {

        uint32_t local_smem_read_ = smem_read_;
#pragma unroll
        for (int ii = 0; ii < LDS_PER_LOOP; ++ii)
        {

            // Apply the XOR pattern if needed. (XOR 8 default)
            if (ROWS_PER_LDS < 8)
            {
                local_smem_read_ = (smem_read_ ^ ((ii * ROWS_PER_LDS) % 8 * BYTES_PER_LDS));
            }

            // Load the elements before the reduction (split-K).
            uint4 tmp[Cta_tile::WARPS_K];
#pragma unroll
            for (int jj = 0; jj < Cta_tile::WARPS_K; ++jj)
            {
                // Note: ROWS_PER_LDS does not take packing into account - hence BYTES_PER_ROW.
                int imm = ii * ROWS_PER_LDS * BYTES_PER_ROW * Cta_tile::WARPS_K + jj * BYTES_PER_ROW_WITH_PACKING;

                // Load...
                if (!HAS_INCOMPLETE_LDS || (ii < LDS_PER_LOOP - 1 || is_active_for_last_lds_))
                {
                    fmha::lds(tmp[jj], local_smem_read_ + imm);
                }
            }

            // Perform the reduction.
            out[ii] = tmp[0];
#pragma unroll
            for (int jj = 1; jj < Cta_tile::WARPS_K; ++jj)
            {
                out[ii] = fmha::add8<Epilogue_type>(out[ii], tmp[jj]);
            }
        }
    }

    // Store the accumulators.
    template <typename Converter, int M, int N, typename Accumulators>
    inline __device__ void store_(Accumulators const (&acc)[M][N], int mi)
    {

        enum
        {
            M_PER_MMA = Mma_tile::M_PER_MMA_PER_CTA
        };

        Converter converter;
#pragma unroll
        for (int ni = 0; ni < Mma_tile::MMAS_N; ++ni)
        {

            // The number of MMAs that are stored per loop iteration.
            enum
            {
                MMAS_M_PER_LOOP = Mma_tile::MMAS_M / LOOPS
            };

            // Store 1st column of the different MMAs.
            // Skip N paddings
            if (ni < Mma_tile::VALID_MMAS_N)
            {
#pragma unroll
                for (int mj = 0; mj < MMAS_M_PER_LOOP; ++mj)
                {

                    // Precompute the immediates to jump between rows.
                    int row_0 = (mj * M_PER_MMA + 0) * BYTES_PER_ROW * Cta_tile::WARPS_K;
                    int row_1 = (mj * M_PER_MMA + 8) * BYTES_PER_ROW * Cta_tile::WARPS_K;

                    // The values (2 halves per register).
                    uint32_t h0 = converter.convert(acc[mi * MMAS_M_PER_LOOP + mj][ni], 0);
                    uint32_t h1 = converter.convert(acc[mi * MMAS_M_PER_LOOP + mj][ni], 1);

                    // Store to shared memory.
                    fmha::sts(smem_write_ + row_0, h0);
                    fmha::sts(smem_write_ + row_1, h1);
                }
            }

            // Swizzle the write pointer using a XOR of 16B.
            smem_write_ ^= 16;

            // Store 2nd column of the different MMAs.
            // Skip N paddings
            if (ni < Mma_tile::VALID_MMAS_N)
            {
#pragma unroll
                for (int mj = 0; mj < MMAS_M_PER_LOOP; ++mj)
                {

                    // Precompute the immediates to jump between rows.
                    int row_0 = (mj * M_PER_MMA + 0) * BYTES_PER_ROW * Cta_tile::WARPS_K;
                    int row_1 = (mj * M_PER_MMA + 8) * BYTES_PER_ROW * Cta_tile::WARPS_K;

                    // The values (2 halves per register).
                    uint32_t h2 = converter.convert(acc[mi * MMAS_M_PER_LOOP + mj][ni], 2);
                    uint32_t h3 = converter.convert(acc[mi * MMAS_M_PER_LOOP + mj][ni], 3);

                    // Store to shared memory.
                    fmha::sts(smem_write_ + row_0, h2);
                    fmha::sts(smem_write_ + row_1, h3);
                }
            }

            // Cancel the previous XOR of 1 + swizzle the write pointer using a XOR of 32B or 64B.
            if (ROW_PACKING == 4)
            {
                smem_write_ ^= 16;
            }
            else if (ROW_PACKING == 2)
            {
                smem_write_ ^= 3 * 16;
            }
            else if (ROW_PACKING == 1)
            {
                //         7
                //       /    \
                //      3      3
                //    /  \    /  \
                //   1    1  1    1
                static_assert(Mma_tile::MMAS_N <= 64, "");
                if (Mma_tile::MMAS_N >= 32 && ni % 16 == 15)
                {
                    smem_write_ ^= 63 * 16;
                }
                else if (Mma_tile::MMAS_N >= 16 && ni % 8 == 7)
                {
                    smem_write_ ^= 31 * 16;
                }
                else if (Mma_tile::MMAS_N >= 8 && ni % 4 == 3)
                {
                    smem_write_ ^= 15 * 16;
                }
                else if (Mma_tile::MMAS_N >= 4 && ni % 2 == 1)
                {
                    smem_write_ ^= 7 * 16;
                }
                else if (Mma_tile::MMAS_N >= 2)
                {
                    smem_write_ ^= 3 * 16;
                }
            }
            else
            {
                assert(false);
            }
        }
    }

    // The write pointer.
    uint32_t smem_write_;
    // The write pointer.
    uint32_t smem_read_;
    // Is the thread active for the last LDS of the series?
    int is_active_for_last_lds_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Smem_tile_o<fmha::Turing_hmma_fp16_traits, Cta_tile>
    : public Hmma_smem_tile_o<fmha::Turing_hmma_fp16_traits, Cta_tile>
{

    // The traits class.
    using Traits = fmha::Turing_hmma_fp16_traits;
    // The base class.
    using Base = Hmma_smem_tile_o<Traits, Cta_tile>;

    // The FP16 accumulators.
    using Accumulators_fp16 = fmha::Fragment_accumulator<fmha::Turing_hmma_fp16_traits>;
    // The FP32 accumulators.
    using Accumulators_fp32 = fmha::Fragment_accumulator<fmha::Turing_hmma_fp32_traits>;

    // Ctor.
    inline __device__ Smem_tile_o(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }

    // Store from FP16 accumulators. That's the default.
    template <int M, int N>
    inline __device__ void store(Accumulators_fp16 const (&acc)[M][N], int mi)
    {
        this->template store_<Convert_from_fp16>(acc, mi);
    }

    // Store from FP32 accumulators. Special trick for the Flash-attention kernel.
    // Convert from fp32 to fp16 before STS
    template <int M, int N>
    inline __device__ void store(Accumulators_fp32 const (&acc)[M][N], int mi)
    {
        this->template store_<Convert_from_fp32>(acc, mi);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Smem_tile_o<fmha::Ampere_hmma_fp16_traits, Cta_tile>
    : public Hmma_smem_tile_o<fmha::Ampere_hmma_fp16_traits, Cta_tile>
{

    // The traits class.
    using Traits = fmha::Ampere_hmma_fp16_traits;
    // The base class.
    using Base = Hmma_smem_tile_o<Traits, Cta_tile>;

    // The FP16 accumulators.
    using Accumulators_fp16 = fmha::Fragment_accumulator<fmha::Ampere_hmma_fp16_traits>;
    // The FP32 accumulators.
    using Accumulators_fp32 = fmha::Fragment_accumulator<fmha::Ampere_hmma_fp32_traits>;

    // Ctor.
    inline __device__ Smem_tile_o(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }

    // Store from FP16 accumulators. That's the default.
    template <int M, int N>
    inline __device__ void store(Accumulators_fp16 const (&acc)[M][N], int mi)
    {
        this->template store_<Convert_from_fp16>(acc, mi);
    }

    // Store from FP32 accumulators. Special trick for the Flash-attention kernel.
    // Convert from fp32 to fp16 before STS
    template <int M, int N>
    inline __device__ void store(Accumulators_fp32 const (&acc)[M][N], int mi)
    {
        this->template store_<Convert_from_fp32>(acc, mi);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Smem_tile_o<fmha::Ampere_hmma_bf16_bf16_traits, Cta_tile>
    : public Hmma_smem_tile_o<fmha::Ampere_hmma_bf16_bf16_traits, Cta_tile>
{

    // The traits class.
    using Traits = fmha::Ampere_hmma_bf16_bf16_traits;
    // The base class.
    using Base = Hmma_smem_tile_o<Traits, Cta_tile>;

    // The FP32 accumulators (only FP32 acc is supported for BF16 MMA).
    using Accumulators_bf16 = fmha::Fragment_accumulator<fmha::Ampere_hmma_bf16_traits>;

    // Ctor.
    inline __device__ Smem_tile_o(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }

    // Store from FP32 accumulators. Special trick for the Flash-attention kernel.
    // Convert from fp32 to bf16 before STS
    template <int M, int N>
    inline __device__ void store(Accumulators_bf16 const (&acc)[M][N], int mi)
    {
        this->template store_<Convert_from_fp32>(acc, mi);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Smem_tile_o<fmha::Ampere_hmma_fp32_traits, Cta_tile>
    : public Hmma_smem_tile_o<fmha::Ampere_hmma_fp32_traits, Cta_tile, 8>
{

    // The traits class.
    using Traits = fmha::Ampere_hmma_fp32_traits;
    // The base class.
    using Base = Hmma_smem_tile_o<Traits, Cta_tile, 8>;
    // The MMA tile.
    using Mma_tile = typename Base::Mma_tile;
    // The accumulators.
    using Accumulator = typename Base::Accumulator;

    // The size of each
    enum
    {
        BYTES_PER_ELEMENT = Base::BYTES_PER_ELEMENT
    };

    // The size of each row in shared memory.
    enum
    {
        BYTES_PER_ROW = Base::BYTES_PER_ROW * Cta_tile::WARPS_K
    };

    // The size of each row in shared memory.
    enum
    {
        BYTES_PER_LDS = Base::BYTES_PER_LDS
    };

    // The number of threads (to produce 16B per LDS).
    enum
    {
        THREADS_PER_ROW = Base::THREADS_PER_ROW
    };

    // The number of outer loops.
    enum
    {
        LOOPS = Base::LOOPS
    };

    // The number of rows loaded per LDS.
    enum
    {
        ROWS_PER_LDS = Base::ROWS_PER_LDS
    };

    // Do we have to guard against partial writes/reads.
    enum
    {
        HAS_INCOMPLETE_LDS = Base::HAS_INCOMPLETE_LDS
    };

    // The total number of LDS per loop.
    enum
    {
        LDS_PER_LOOP = Base::LDS_PER_LOOP
    };

    // Ctor.
    inline __device__ Smem_tile_o(void* smem, int tidx)
        : Base(smem, tidx)
    {

        // Get a 32-bit value for the shared memory address.
        uint32_t smem_ = __nvvm_get_smem_pointer(smem);

        // The element read by each thread.
        int read_row = tidx / THREADS_PER_ROW;
        int read_col = tidx % THREADS_PER_ROW;

        // Take the XOR pattern into account for the column.
        read_col ^= (read_row & 0x7) * 2;

        // Assemble the read pointer.
        this->smem_read_ = smem_ + read_row * BYTES_PER_ROW + read_col * BYTES_PER_LDS;

        // Is that thread active on the last LDS?
        if (HAS_INCOMPLETE_LDS)
        {
            this->is_active_for_last_lds_ = read_row + (LDS_PER_LOOP - 1) * ROWS_PER_LDS < Cta_tile::M;
        }
    }

    // Load the output fragments.
    inline __device__ void load(uint4 (&out)[LDS_PER_LOOP]) const
    {
#pragma unroll
        for (int ii = 0; ii < LDS_PER_LOOP; ++ii)
        {

            // Load the elements before the reduction (split-K).
            uint4 tmp[Cta_tile::WARPS_K];
#pragma unroll
            for (int jj = 0; jj < Cta_tile::WARPS_K; ++jj)
            {
                int imm = ii * ROWS_PER_LDS * BYTES_PER_ROW + jj * Cta_tile::N * BYTES_PER_ELEMENT;
                int is_valid = ii < LDS_PER_LOOP - 1 || this->is_active_for_last_lds_;
                if (!HAS_INCOMPLETE_LDS || is_valid)
                {
                    fmha::lds(tmp[jj], this->smem_read_ + imm);
                }
            }

            // Perform the reduction.
            out[ii] = tmp[0];
#pragma unroll
            for (int jj = 1; jj < Cta_tile::WARPS_K; ++jj)
            {
                out[ii] = fmha::fadd4(out[ii], tmp[jj]);
            }
        }
    }

    // Store the accumulators.
    template <int M, int N>
    inline __device__ void store(Accumulator const (&acc)[M][N], int mi)
    {

        enum
        {
            M_PER_MMA = Mma_tile::M_PER_MMA_PER_CTA
        };

#pragma unroll
        for (int ni = 0; ni < Mma_tile::MMAS_N; ++ni)
        {

            // The number of MMAs that are stored per loop iteration.
            enum
            {
                MMAS_M_PER_LOOP = Mma_tile::MMAS_M / LOOPS
            };

            // Store 1st column of the different MMAs.
            if (ni < Mma_tile::VALID_MMAS_N)
            {
#pragma unroll
                for (int mj = 0; mj < MMAS_M_PER_LOOP; ++mj)
                {

                    // Precompute the immediates to jump between rows.
                    int row_0 = (mj * M_PER_MMA + 0) * BYTES_PER_ROW;
                    int row_1 = (mj * M_PER_MMA + 8) * BYTES_PER_ROW;

                    // Pack vectors.
                    uint2 tmp0;
                    tmp0.x = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(0);
                    tmp0.y = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(1);

                    uint2 tmp1;
                    tmp1.x = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(2);
                    tmp1.y = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(3);

                    // Store.
                    fmha::sts(this->smem_write_ + row_0, tmp0);
                    fmha::sts(this->smem_write_ + row_1, tmp1);
                }
            }

            // Swizzle the write pointer using a XOR of 16B.
            this->smem_write_ ^= 32;

            // Store 2nd column of the different MMAs.
            if (ni < Mma_tile::VALID_MMAS_N)
            {
#pragma unroll
                for (int mj = 0; mj < MMAS_M_PER_LOOP; ++mj)
                {
                    // Precompute the immediates to jump between rows.
                    int row_0 = (mj * M_PER_MMA + 0) * BYTES_PER_ROW;
                    int row_1 = (mj * M_PER_MMA + 8) * BYTES_PER_ROW;

                    uint2 tmp0, tmp1;
                    tmp0.x = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(4);
                    tmp0.y = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(5);

                    tmp1.x = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(6);
                    tmp1.y = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(7);

                    // Store.
                    fmha::sts(this->smem_write_ + row_0, tmp0);
                    fmha::sts(this->smem_write_ + row_1, tmp1);
                }
            }

            // Cancel the previous XOR of 1 + swizzle the write pointer using a XOR of 32B or 64B.
            static_assert(Mma_tile::MMAS_N <= 16, "");
            if (Mma_tile::MMAS_N >= 16 && (ni & 7) == 7)
            {
                this->smem_write_ ^= 31 * 32;
            }
            else if (Mma_tile::MMAS_N >= 8 && (ni & 3) == 3)
            {
                this->smem_write_ ^= 15 * 32;
            }
            else if (Mma_tile::MMAS_N >= 4 && (ni & 1) == 1)
            {
                this->smem_write_ ^= 7 * 32;
            }
            else if ((ni & 1) == 0)
            {
                this->smem_write_ ^= 3 * 32;
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Smem_tile_o<fmha::Ampere_hmma_bf16_traits, Cta_tile>
    : public Hmma_smem_tile_o<fmha::Ampere_hmma_bf16_traits, Cta_tile, 8>
{

    // The traits class.
    using Traits = fmha::Ampere_hmma_bf16_traits;
    // The base class.
    using Base = Hmma_smem_tile_o<Traits, Cta_tile, 8>;
    // The MMA tile.
    using Mma_tile = typename Base::Mma_tile;
    // The accumulators.
    using Accumulator = typename Base::Accumulator;

    // The size of each element.
    enum
    {
        BYTES_PER_ELEMENT = Base::BYTES_PER_ELEMENT
    };

    // The size of each row in shared memory.
    enum
    {
        BYTES_PER_ROW = Base::BYTES_PER_ROW * Cta_tile::WARPS_K
    };

    // The size of each row in shared memory.
    enum
    {
        BYTES_PER_LDS = Base::BYTES_PER_LDS
    };

    // The number of threads (to produce 16B per LDS).
    enum
    {
        THREADS_PER_ROW = Base::THREADS_PER_ROW
    };

    // The number of outer loops.
    enum
    {
        LOOPS = Base::LOOPS
    };

    // The number of rows loaded per LDS.
    enum
    {
        ROWS_PER_LDS = Base::ROWS_PER_LDS
    };

    // Do we have to guard against partial writes/reads.
    enum
    {
        HAS_INCOMPLETE_LDS = Base::HAS_INCOMPLETE_LDS
    };

    // The total number of LDS per loop.
    enum
    {
        LDS_PER_LOOP = Base::LDS_PER_LOOP
    };

    // Ctor.
    inline __device__ Smem_tile_o(void* smem, int tidx)
        : Base(smem, tidx)
    {

        // Get a 32-bit value for the shared memory address.
        uint32_t smem_ = __nvvm_get_smem_pointer(smem);

        // The element read by each thread.
        int read_row = tidx / THREADS_PER_ROW;
        int read_col = tidx % THREADS_PER_ROW;

        // Take the XOR pattern into account for the column.
        read_col ^= (read_row & 0x7) * 2;

        // Assemble the read pointer.
        this->smem_read_ = smem_ + read_row * BYTES_PER_ROW + read_col * BYTES_PER_LDS;

        // Is that thread active on the last LDS?
        if (HAS_INCOMPLETE_LDS)
        {
            this->is_active_for_last_lds_ = read_row + (LDS_PER_LOOP - 1) * ROWS_PER_LDS < Cta_tile::M;
        }
    }

    // Load the output fragments.
    inline __device__ void load(uint4 (&out)[LDS_PER_LOOP]) const
    {
#pragma unroll
        for (int ii = 0; ii < LDS_PER_LOOP; ++ii)
        {

            // Load the elements before the reduction (split-K).
            uint4 tmp[Cta_tile::WARPS_K];
#pragma unroll
            for (int jj = 0; jj < Cta_tile::WARPS_K; ++jj)
            {
                int imm = ii * ROWS_PER_LDS * BYTES_PER_ROW + jj * Cta_tile::N * BYTES_PER_ELEMENT;
                int is_valid = ii < LDS_PER_LOOP - 1 || this->is_active_for_last_lds_;
                if (!HAS_INCOMPLETE_LDS || is_valid)
                {
                    fmha::lds(tmp[jj], this->smem_read_ + imm);
                }
            }

            // Perform the reduction.
            out[ii] = tmp[0];
#pragma unroll
            for (int jj = 1; jj < Cta_tile::WARPS_K; ++jj)
            {
                out[ii] = fmha::fadd4(out[ii], tmp[jj]);
            }
        }
    }

    // Store the accumulators.
    template <int M, int N>
    inline __device__ void store(Accumulator const (&acc)[M][N], int mi)
    {

        enum
        {
            M_PER_MMA = Mma_tile::M_PER_MMA_PER_CTA
        };

#pragma unroll
        for (int ni = 0; ni < Mma_tile::MMAS_N; ++ni)
        {

            // The number of MMAs that are stored per loop iteration.
            enum
            {
                MMAS_M_PER_LOOP = Mma_tile::MMAS_M / LOOPS
            };

            // Store 1st column of the different MMAs.
            if (ni < Mma_tile::VALID_MMAS_N)
            {
#pragma unroll
                for (int mj = 0; mj < MMAS_M_PER_LOOP; ++mj)
                {

                    // Precompute the immediates to jump between rows.
                    int row_0 = (mj * M_PER_MMA + 0) * BYTES_PER_ROW;
                    int row_1 = (mj * M_PER_MMA + 8) * BYTES_PER_ROW;

                    // Pack vectors.
                    uint2 tmp0;
                    tmp0.x = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(0);
                    tmp0.y = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(1);

                    uint2 tmp1;
                    tmp1.x = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(2);
                    tmp1.y = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(3);

                    // Store.
                    fmha::sts(this->smem_write_ + row_0, tmp0);
                    fmha::sts(this->smem_write_ + row_1, tmp1);
                }
            }

            // Swizzle the write pointer using a XOR of 16B.
            this->smem_write_ ^= 32;

            // Store 2nd column of the different MMAs.
            if (ni < Mma_tile::VALID_MMAS_N)
            {
#pragma unroll
                for (int mj = 0; mj < MMAS_M_PER_LOOP; ++mj)
                {
                    // Precompute the immediates to jump between rows.
                    int row_0 = (mj * M_PER_MMA + 0) * BYTES_PER_ROW;
                    int row_1 = (mj * M_PER_MMA + 8) * BYTES_PER_ROW;

                    uint2 tmp0, tmp1;
                    tmp0.x = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(4);
                    tmp0.y = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(5);

                    tmp1.x = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(6);
                    tmp1.y = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(7);

                    // Store.
                    fmha::sts(this->smem_write_ + row_0, tmp0);
                    fmha::sts(this->smem_write_ + row_1, tmp1);
                }
            }

            // Cancel the previous XOR of 1 + swizzle the write pointer using a XOR of 32B or 64B.
            static_assert(Mma_tile::MMAS_N <= 16, "");
            if ((ni & 1) == 0)
            {
                this->smem_write_ ^= 3 * 32;
            }
            else if (Mma_tile::MMAS_N >= 16 && (ni & 7) == 7)
            {
                this->smem_write_ ^= 31 * 32;
            }
            else if (Mma_tile::MMAS_N >= 8 && (ni & 3) == 3)
            {
                this->smem_write_ ^= 15 * 32;
            }
            else if (Mma_tile::MMAS_N >= 4 && (ni & 1) == 1)
            {
                this->smem_write_ ^= 7 * 32;
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// I M M A
//
////////////////////////////////////////////////////////////////////////////////////////////////////

// each thread holds 8 accumulator registers per 16x16 MMA, representing a 2x4 tile
template <typename Traits>
struct Regs_to_rows
{

    template <typename Acc>
    static inline __device__ void extract(Acc const& acc, uint4& row0, uint4& row1)
    {

        // Volta/Turing: row-major
        uint32_t tmp_00 = acc.reg(0);
        uint32_t tmp_01 = acc.reg(2);
        uint32_t tmp_02 = acc.reg(1);
        uint32_t tmp_03 = acc.reg(3);
        uint32_t tmp_10 = acc.reg(4);
        uint32_t tmp_11 = acc.reg(6);
        uint32_t tmp_12 = acc.reg(5);
        uint32_t tmp_13 = acc.reg(7);

        row0.x = tmp_00;
        row0.y = tmp_01;
        row0.z = tmp_02;
        row0.w = tmp_03;

        row1.x = tmp_10;
        row1.y = tmp_11;
        row1.z = tmp_12;
        row1.w = tmp_13;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Regs_to_rows_8bit
{

    template <typename Acc>
    static inline __device__ void extract(Acc const& acc, uint4& row0, uint4& row1)
    {
        // Ampere: col-major
        uint32_t tmp_00 = acc.reg(0);
        uint32_t tmp_01 = acc.reg(4);
        uint32_t tmp_02 = acc.reg(1);
        uint32_t tmp_03 = acc.reg(5);
        uint32_t tmp_10 = acc.reg(2);
        uint32_t tmp_11 = acc.reg(6);
        uint32_t tmp_12 = acc.reg(3);
        uint32_t tmp_13 = acc.reg(7);

        row0.x = tmp_00;
        row0.y = tmp_01;
        row0.z = tmp_02;
        row0.w = tmp_03;

        row1.x = tmp_10;
        row1.y = tmp_11;
        row1.z = tmp_12;
        row1.w = tmp_13;
    }
};

template <>
struct Regs_to_rows<fmha::Ampere_imma_int8_int32_traits> : public Regs_to_rows_8bit
{
};

template <>
struct Regs_to_rows<fmha::Ada_qmma_e4m3_fp32_traits> : public Regs_to_rows_8bit
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Regs_to_rows<fmha::Ada_qmma_e4m3_fp16_traits>
{

    template <typename Acc>
    static inline __device__ void extract(Acc const& acc, uint2& row0, uint2& row1)
    {

        uint16_t* row0_ptr = reinterpret_cast<uint16_t*>(&row0);
        uint16_t* row1_ptr = reinterpret_cast<uint16_t*>(&row1);
        row0_ptr[0] = acc.u16(0);
        row0_ptr[1] = acc.u16(4);
        row0_ptr[2] = acc.u16(1);
        row0_ptr[3] = acc.u16(5);

        row1_ptr[0] = acc.u16(2);
        row1_ptr[1] = acc.u16(6);
        row1_ptr[2] = acc.u16(3);
        row1_ptr[3] = acc.u16(7);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Acc>
inline __device__ void add4(uint4& dst, uint4 const& src)
{
    reinterpret_cast<Acc&>(dst.x) += reinterpret_cast<Acc const&>(src.x);
    reinterpret_cast<Acc&>(dst.y) += reinterpret_cast<Acc const&>(src.y);
    reinterpret_cast<Acc&>(dst.z) += reinterpret_cast<Acc const&>(src.z);
    reinterpret_cast<Acc&>(dst.w) += reinterpret_cast<Acc const&>(src.w);
}

template <typename Acc>
inline __device__ void add_vec(uint4& dst, uint4 const& src)
{
    add4<Acc>(dst, src);
}

template <>
inline __device__ void add_vec<uint16_t>(uint4& dst, uint4 const& src)
{
    dst = fmha::hadd8(dst, src);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// The base class for 32-bit/16-bit accumulator types of imma/qmma.
// TODO Can we port Ampere hmma fp32 to this?
template <typename Traits, typename Cta_tile>
struct Smem_tile_o_base_8bit_mma
{

    // The MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;
    // The accumulators.
    using Accumulator = fmha::Fragment_accumulator<Traits>;

    // The size of each element.
    enum
    {
        BYTES_PER_ELEMENT = sizeof(typename Traits::Accumulator_type)
    };

    // The amount of bytes per row (without packing or split-k).
    enum
    {
        BYTES_PER_ROW = Cta_tile::N * BYTES_PER_ELEMENT
    };

    // The size of each STS.
    enum
    {
        BYTES_PER_STS = BYTES_PER_ELEMENT * 4
    };

    // The STS Packed Data Type
    using Sts_packed_type = typename Uint_from_size_in_bytes<BYTES_PER_STS>::Type;

    // The size of each LDS.
    enum
    {
        BYTES_PER_LDS = 16
    };

    // The number of threads to store a "row" of the matrix. We force it to 16 for SEQLEN=384.
    enum
    {
        THREADS_PER_ROW = BYTES_PER_ROW / BYTES_PER_LDS
    };

    // The number of rows loaded per LDS.
    enum
    {
        ROWS_PER_LDS = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW
    };

    // The STS bytes for one quad of threads
    enum
    {
        BYTES_PER_STS_PER_QUAD = BYTES_PER_STS * 4
    };

    // The xor factor per LDS
    // (4 consecutive threads do 64B swizzle for 16B per sts, 32B swizzle for 8B per sts)
    enum
    {
        XOR_FACTOR = fmha::Div_up<BYTES_PER_STS * 4, BYTES_PER_LDS>::VALUE
    };

    // The smem offset in bytes per MMA_N (2 squad threads)
    enum
    {
        BYTES_OFFSET_PER_MMA_N = BYTES_PER_STS * 8
    };

    // The number of "rows" to process in total.
    enum
    {
        ROWS = Cta_tile::M
    };

    // We want at least one output per thread (if possible).
    enum
    {
        ROWS_PER_LOOP_ = ROWS <= 64 ? ROWS : (int) Min<ROWS, ROWS_PER_LDS>::VALUE
    };

    // We also want to have "complete" MMAs.
    enum
    {
        ROWS_PER_LOOP = Max<ROWS_PER_LOOP_, Mma_tile::M_PER_MMA_PER_CTA>::VALUE
    };

    // The number of outer loops.
    enum
    {
        LOOPS = fmha::Div_up<ROWS, ROWS_PER_LOOP>::VALUE
    };

    // Make sure it matches our expectations.
    static_assert(ROWS_PER_LOOP >= (int) Mma_tile::M_PER_MMA_PER_CTA, "");

    // Do we have to guard against partial writes/reads.
    enum
    {
        HAS_INCOMPLETE_LDS = ROWS_PER_LOOP % ROWS_PER_LDS != 0
    };

    // The total number of LDS per loop.
    enum
    {
        LDS_PER_LOOP = fmha::Div_up<ROWS_PER_LOOP, ROWS_PER_LDS>::VALUE
    };

    // The amount of shared memory.
    enum
    {
        BYTES_PER_TILE = ROWS_PER_LOOP * BYTES_PER_ROW * Cta_tile::WARPS_K
    };

    // The amount of row packing to make sure we have at least 128B per smem row (without split-k).
    enum
    {
        ROW_PACKING = Max<1, 128 / BYTES_PER_ROW>::VALUE
    };

    // Make sure our row packing is correct
    static_assert(ROWS_PER_LOOP % ROW_PACKING == 0, "");

    // The amount of shared memory per row after packing.
    enum
    {
        BYTES_PER_ROW_WITH_PACKING = BYTES_PER_ROW * ROW_PACKING
    };

    // Make sure we have at least 128B per row after packing.
    static_assert(BYTES_PER_ROW_WITH_PACKING >= 128, "");

    // The number of threads per row after packing.
    enum
    {
        THREADS_PER_ROW_WITH_PACKING = THREADS_PER_ROW * ROW_PACKING
    };

    // Make sure we have at least 8 threads per row after packing.
    static_assert(THREADS_PER_ROW_WITH_PACKING >= 8, "");

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

    static_assert(WARPS_K > 1 || std::is_same<Traits, Ada_qmma_e4m3_fp32_traits>::value,
        "Kernel misconfigured. No split-k needed.");

    // Determine the config.
    enum
    {
        WARPS_2x1x2 = WARPS_M == 2 && WARPS_N == 1 && WARPS_K == 2
    };

    enum
    {
        WARPS_4x1x2 = WARPS_M == 4 && WARPS_N == 1 && WARPS_K == 2
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

    // Ctor.
    inline __device__ Smem_tile_o_base_8bit_mma(void* smem, int tidx)
    {

        // Get a 32-bit value for the shared memory address.
        uint32_t smem_ = __nvvm_get_smem_pointer(smem);

        // The row/col written by the thread.
        int write_row, write_col;

        // SEQLEN == 128 and HIDDEN_SIZE_PER_HEAD == 16.
        if (WARPS_2x1x2 && Cta_tile::N == 16)
        {
            write_row = (tidx & 0x20) / 4 + (tidx & 0x1e) / 8;
            write_col = (tidx & 0x40) / 8 + (tidx & 0x07);

            // SEQLEN == 128 and HIDDEN_SIZE_PER_HEAD == 32.
        }
        else if (WARPS_2x1x2 && Cta_tile::N == 32)
        {
            write_row = (tidx & 0x20) / 2 + (tidx & 0x1c) / 4;
            write_col = (tidx & 0x40) / 8 + (tidx & 0x07);

            // SEQLEN == 128 and HIDDEN_SIZE_PER_HEAD == 64.
        }
        else if (WARPS_2x1x2 && Cta_tile::N == 64)
        {
            write_row = (tidx & 0x20) / 2 + (tidx & 0x1c) / 4;
            write_col = (tidx & 0x40) / 4 + (tidx & 0x07);

            // SEQLEN == 256, 384, 512 and HIDDEN_SIZE_PER_HEAD == 16.
        }
        else if ((WARPS_1x1x8 || WARPS_1x1x4) && Cta_tile::N == 16)
        {
            write_row = (tidx & 0x18) / 8;
            write_col = (tidx & 0xe0) / 4 + (tidx & 0x07);

            // SEQLEN == 256, 384, 512 and HIDDEN_SIZE_PER_HEAD == 32.
        }
        else if ((WARPS_1x1x8 || WARPS_1x1x4) && Cta_tile::N == 32)
        {
            write_row = (tidx & 0x1c) / 4;
            write_col = (tidx & 0xe0) / 4 + (tidx & 0x07);

            // SEQLEN == 256, 384 and HIDDEN_SIZE_PER_HEAD == 64.
        }
        else if ((WARPS_1x1x8 || WARPS_1x1x4) && Cta_tile::N == 64)
        {
            write_row = (tidx & 0x1c) / 4;
            write_col = (tidx & 0xe0) / 2 + (tidx & 0x07);

            // GMMA: HIDDEN_SIZE_PER_HEAD == 64.
        }
        else if (WARPS_4x1x2 && Cta_tile::N == 64)
        {
            write_row = (tidx & 0x60) / 2 + (tidx & 0x1c) / 4;
            write_col = (tidx & 0x80) / 8 + (tidx & 0x07);

            // Ada e4m3_fp32
        }
        else if (WARPS_4x1x1)
        {
            write_row = (tidx & 0x60) / 2 + (tidx & 0x1c) / 4;
            write_col = (tidx & 0x80) / 8 + (tidx & 0x07);

            // Not supported.
        }
        else
        {
            assert(false);
        }

        // Assemble the write pointer.
        smem_write_ = smem_ + write_row * BYTES_PER_ROW_WITH_PACKING * Cta_tile::WARPS_K + write_col * BYTES_PER_STS;

        // The element read by each thread.
        int read_row = tidx / THREADS_PER_ROW;
        int read_col = tidx % THREADS_PER_ROW;

        // Is that thread active on the last LDS?
        if (HAS_INCOMPLETE_LDS)
        {
            is_active_for_last_lds_ = read_row + (LDS_PER_LOOP - 1) * ROWS_PER_LDS < ROWS_PER_LOOP;
        }

        // The XOR params.
        constexpr int XOR_MOD = 2 / ROW_PACKING;

        // Take the XOR pattern and the packing into account for the column.
        read_col += read_row % ROW_PACKING * XOR_FACTOR;
        read_row /= ROW_PACKING;
        read_col ^= (read_row % XOR_MOD) * XOR_FACTOR;

        // Assemble the read pointer.
        smem_read_ = smem_ + read_row * BYTES_PER_ROW_WITH_PACKING * Cta_tile::WARPS_K + read_col * BYTES_PER_LDS;
    }

    // Load the output fragments.
    inline __device__ void load(uint4 (&out)[LDS_PER_LOOP]) const
    {
#pragma unroll
        for (int ii = 0; ii < LDS_PER_LOOP; ++ii)
        {

            // Load the elements before the reduction (split-K).
            uint4 tmp[Cta_tile::WARPS_K];
#pragma unroll
            for (int jj = 0; jj < Cta_tile::WARPS_K; ++jj)
            {
                // Note: ROWS_PER_LDS does not take packing into account - hence BYTES_PER_ROW.
                int imm = ii * ROWS_PER_LDS * BYTES_PER_ROW * Cta_tile::WARPS_K + jj * BYTES_PER_ROW_WITH_PACKING;

                // Load...
                if (!HAS_INCOMPLETE_LDS || (ii < LDS_PER_LOOP - 1 || is_active_for_last_lds_))
                {
                    fmha::lds(tmp[jj], smem_read_ + imm);
                }
            }

// Perform the reduction.
#pragma unroll
            for (int jj = 1; jj < Cta_tile::WARPS_K; ++jj)
            {
                add_vec<Traits::Accumulator_type>(tmp[0], tmp[jj]);
            }

            // Write to out.
            out[ii] = tmp[0];
        }
    }

    // Store the accumulators.
    template <int M, int N>
    inline __device__ void store(Accumulator const (&acc)[M][N], int mi)
    {

        enum
        {
            M_PER_MMA = Mma_tile::M_PER_MMA_PER_CTA
        };

        // The number of MMAs that are stored per loop iteration.
        enum
        {
            MMAS_M_PER_LOOP = Mma_tile::MMAS_M / LOOPS
        };

#pragma unroll
        for (int ni = 0; ni < Mma_tile::MMAS_N; ++ni)
        {

#pragma unroll
            for (int mj = 0; mj < MMAS_M_PER_LOOP; ++mj)
            {

                Sts_packed_type row_0, row_1;
                Regs_to_rows<Traits>::extract(acc[mi * MMAS_M_PER_LOOP + mj][ni], row_0, row_1);

                /*
                (32bit acc) Each thread of a quad writes 16B per STS -> 64B per store.
                            Account for 2 -> 128B.
                (16bit acc) Each thread of a quad writes 8B per STS -> 32B per store.
                            Account for 2 -> 64B.
                */
                int imm_0
                    = (mj * M_PER_MMA + 0) * BYTES_PER_ROW * Cta_tile::WARPS_K + (ni / 2) * BYTES_OFFSET_PER_MMA_N;
                int imm_1
                    = (mj * M_PER_MMA + 8) * BYTES_PER_ROW * Cta_tile::WARPS_K + (ni / 2) * BYTES_OFFSET_PER_MMA_N;

                // Store the elements.
                fmha::sts(this->smem_write_ + imm_0, row_0);
                fmha::sts(this->smem_write_ + imm_1, row_1);
            }
            // (32bit acc) Each thread of a quad writes 16B per STS -> 64B per store.
            // (16bit acc) Each thread of a quad writes 8B per STS -> 32B per store.
            if (Mma_tile::MMAS_N == 1)
            {
                // Noop.
            }
            else if (Mma_tile::MMAS_N % 2 == 0)
            {
                this->smem_write_ ^= BYTES_PER_STS_PER_QUAD;
            }
            else
            {
                assert(false && "Unsupported");
            }
        }
    }

    // The write pointer.
    uint32_t smem_write_;
    // The write pointer.
    uint32_t smem_read_;
    // Is the thread active for the last LDS of the series?
    int is_active_for_last_lds_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Smem_tile_o<fmha::Volta_imma_int8_int32_traits, Cta_tile>
    : public Smem_tile_o_base_8bit_mma<fmha::Volta_imma_int8_int32_traits, Cta_tile>
{

    // The traits class.
    using Traits = fmha::Volta_imma_int8_int32_traits;
    // The base class.
    using Base = Smem_tile_o_base_8bit_mma<Traits, Cta_tile>;

    // Ctor.
    inline __device__ Smem_tile_o(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Smem_tile_o<fmha::Turing_imma_int8_int32_traits, Cta_tile>
    : public Smem_tile_o_base_8bit_mma<fmha::Turing_imma_int8_int32_traits, Cta_tile>
{

    // The traits class.
    using Traits = fmha::Turing_imma_int8_int32_traits;
    // The base class.
    using Base = Smem_tile_o_base_8bit_mma<Traits, Cta_tile>;

    // Ctor.
    inline __device__ Smem_tile_o(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Smem_tile_o<fmha::Ampere_imma_int8_int32_traits, Cta_tile>
    : public Smem_tile_o_base_8bit_mma<fmha::Ampere_imma_int8_int32_traits, Cta_tile>
{

    // The traits class.
    using Traits = fmha::Ampere_imma_int8_int32_traits;
    // The base class.
    using Base = Smem_tile_o_base_8bit_mma<Traits, Cta_tile>;

    // Ctor.
    inline __device__ Smem_tile_o(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Smem_tile_o<fmha::Ada_qmma_e4m3_fp32_traits, Cta_tile>
    : public Smem_tile_o_base_8bit_mma<fmha::Ada_qmma_e4m3_fp32_traits, Cta_tile>
{

    // The traits class.
    using Traits = fmha::Ada_qmma_e4m3_fp32_traits;
    // The base class.
    using Base = Smem_tile_o_base_8bit_mma<Traits, Cta_tile>;

    // Ctor.
    inline __device__ Smem_tile_o(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Smem_tile_o<fmha::Ada_qmma_e4m3_fp16_traits, Cta_tile>
    : public Smem_tile_o_base_8bit_mma<fmha::Ada_qmma_e4m3_fp16_traits, Cta_tile>
{

    // The traits class.
    using Traits = fmha::Ada_qmma_e4m3_fp16_traits;
    // The base class.
    using Base = Smem_tile_o_base_8bit_mma<Traits, Cta_tile>;

    // Ctor.
    inline __device__ Smem_tile_o(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile>
struct Smem_tile_o_interleaved
{

    // The MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;
    // The accumulators.
    using Accumulator = fmha::Fragment_accumulator<Traits>;

    enum
    {
        VEC = 32
    };

    enum
    {
        NUM_SLICES = Cta_tile::N / VEC
    };

    static_assert(NUM_SLICES == 1 || NUM_SLICES == 2, "");

    enum
    {
        BYTES_PER_ELEMENT = 4
    };

    enum
    {
        BYTES_PER_STS = 16
    };

    enum
    {
        BYTES_PER_LDS = 16
    };

    enum
    {
        ELTS_PER_STS = BYTES_PER_STS / BYTES_PER_ELEMENT
    };

    static_assert(VEC * BYTES_PER_ELEMENT == 128, "");

    enum
    {
        BYTES_PER_ROW = Cta_tile::WARPS_K * VEC * BYTES_PER_ELEMENT
    };

    // Each row only stores one slice. The other slice starts this many rows below
    enum
    {
        ROWS_PER_SLICE = Cta_tile::WARPS_M * 16
    };

    enum
    {
        TOTAL_ROWS = NUM_SLICES * ROWS_PER_SLICE
    };

    enum
    {
        BYTES_PER_TILE = BYTES_PER_ROW * TOTAL_ROWS
    };

    // LDS
    enum
    {
        THREADS_PER_ROW = 8
    };

    enum
    {
        ROWS_PER_LDS = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW
    };

    enum
    {
        LDS_PER_LOOP = TOTAL_ROWS / ROWS_PER_LDS
    };

    // Ctor.
    inline __device__ Smem_tile_o_interleaved(void* smem, int tidx)
    {

        smem_ = __nvvm_get_smem_pointer(smem);

        constexpr int WARPS_M = Cta_tile::WARPS_M;
        constexpr int WARPS_N = Cta_tile::WARPS_N;
        constexpr int WARPS_K = Cta_tile::WARPS_K;

        // Warp order (fastest to slowest): m => n => k
        // 2x2: 2,2,1 then 2,1,2: mask_m = 0x20, mask_k = 0x40, div_m = 32, div_k = 64
        // 1x4: 1,4,1 then 1,1,4: mask_m = 0x00, mask_k = 0x60, div_m =  X, div_k = 32
        // 1x8: 1,8,1 then 1,1,8: mask_m = 0x00, mask_k = 0xe0, div_m =  X, div_k = 32
        static_assert(WARPS_N == 1, "");

        // A thread holds 4 elts of 4B. One slice of 32 elts has 128B.
        // Two MMAs in N constitute one slice

        // the slice offset that depends on ni and has to be added later
        static_assert(VEC / ELTS_PER_STS == 8, ""); // 8 columns of 4 elements
        if (WARPS_M == 2 && WARPS_N == 1 && WARPS_K == 2)
        {
            write_row = (tidx & 0x1c) / 4 + (tidx & 0x20) / 2; // warp_m * 16 rows
            write_col = (tidx & 0x03) + (tidx & 0x40) / 8;     // warp_k * VEC / ELTS_PER_STS
        }
        else
        {
            write_row = (tidx & 0x1c) / 4;
            write_col = (tidx & 0x03) + (tidx & 0xe0) / 4; // warp_k * VEC / ELTS_PER_STS
        }
        write_col ^= (write_row & 0x01) * 4;               // left or right 64B

        // this->smem_write_ = smem_ + write_row * BYTES_PER_ROW + write_col * BYTES_PER_STS;

        int read_row = tidx / THREADS_PER_ROW;
        int read_col = tidx % THREADS_PER_ROW;
        read_col ^= (read_row & 0x01) * 4;
        this->smem_read_ = smem_ + read_row * BYTES_PER_ROW + read_col * BYTES_PER_LDS;
    }

    // Store the accumulators.
    template <int M, int N>
    inline __device__ void store(Accumulator const (&acc)[M][N], int mi)
    {
#pragma unroll
        for (int ni = 0; ni < Mma_tile::MMAS_N; ++ni)
        {
            int const slice = ni / NUM_SLICES;
            int col = write_col ^ ((ni & 1) * 4);

            uint32_t smem_write_ = smem_ + write_row * BYTES_PER_ROW + col * BYTES_PER_STS;

            // Extract the elements.
            uint4 row_0, row_1;

            Regs_to_rows<Traits>::extract(acc[mi][ni], row_0, row_1);

            // Each thread of a quad writes 16B per STS -> 64B per store. Account for
            // 2 -> 128B.
            int imm_0 = (slice * ROWS_PER_SLICE + 0) * BYTES_PER_ROW;
            int imm_1 = (slice * ROWS_PER_SLICE + 8) * BYTES_PER_ROW;

            // Store the elements.
            fmha::sts(smem_write_ + imm_0, row_0);
            fmha::sts(smem_write_ + imm_1, row_1);
        }
    }

    // Load the output fragments.
    inline __device__ void load(uint4 (&out)[LDS_PER_LOOP]) const
    {
#pragma unroll
        for (int ii = 0; ii < LDS_PER_LOOP; ++ii)
        {

            // Load the elements before the reduction (split-K).
            uint4 tmp[Cta_tile::WARPS_K];
#pragma unroll
            for (int jj = 0; jj < Cta_tile::WARPS_K; ++jj)
            {
                int imm = ii * ROWS_PER_LDS * BYTES_PER_ROW + jj * VEC * BYTES_PER_ELEMENT;
                fmha::lds(tmp[jj], smem_read_ + imm);
            }

// Perform the reduction.
#pragma unroll
            for (int jj = 1; jj < Cta_tile::WARPS_K; ++jj)
            {
                add4<Traits::Accumulator_type>(tmp[0], tmp[jj]);
            }

            // Write to out.
            out[ii] = tmp[0];
        }
    }

    int write_row;
    int write_col;
    uint32_t smem_write_;
    uint32_t smem_read_;
    uint32_t smem_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha
