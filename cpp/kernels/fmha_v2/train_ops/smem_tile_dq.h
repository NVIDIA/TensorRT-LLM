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
#include <fmha/traits.h>
#include <fmha/utils.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

// The shared memory tile is used to reorganize a dQ tile in the Flash Attention training kernel(s)
// and perform a reduction of the partial results computed by the different warps.

template <typename Traits_, typename Cta_tile_>
struct Smem_tile_dq_red_base
{

    // The instruction traits.
    using Traits = Traits_;
    // The CTA descriptor.
    using Cta_tile = Cta_tile_;
    // The associated MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;
    // The accumulators.
    using Accumulators = fmha::Fragment_accumulator<Traits>;

    // DEBUG: Make sure we have only one warp in the M/N dimensions of the tile.
    static_assert(
        Cta_tile::WARPS_M == 1 && Cta_tile::WARPS_N == 1 && (Cta_tile::WARPS_K == 8 || Cta_tile::WARPS_K == 4), "");

    // END OF DEBUG.

    // The number of bytes per element is 4 (fp32).
    enum
    {
        BYTES_PER_ELEMENT = 4
    };

    // The number of elements written per row in shared memory.
    enum
    {
        ELEMENTS_PER_ROW = Cta_tile::N * Cta_tile::WARPS_K
    };

    // The size of a row in bytes.
    enum
    {
        BYTES_PER_ROW = ELEMENTS_PER_ROW * BYTES_PER_ELEMENT
    };

    // The number of rows in smem. We process MMA rows one-by-one to reduce the footprint in smem.
    enum
    {
        ROWS = Mma_tile::M_PER_MMA_PER_CTA
    };

    // The size of the shared memory tile.
    enum
    {
        BYTES_PER_TILE = ROWS * BYTES_PER_ROW
    };

    // DEBUG. Make sure the math is correct for 1x1x8, STEPQ = 16 and D = 64.
    static_assert(BYTES_PER_TILE == 16 * Cta_tile::N * Cta_tile::WARPS_K * 4, "");

    // The number of threads per row when loading from shared memory. We use LDS.32.
    enum
    {
        THREADS_PER_ROW = Cta_tile::N
    };

    // The number of rows per read per LDS.
    enum
    {
        ROWS_PER_LDS = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW
    };

    // Make sure we do not have partial reads.
    static_assert(ROWS % ROWS_PER_LDS == 0, "");

    // The number of LDS per thread/CTA.
    enum
    {
        LDS = ROWS / ROWS_PER_LDS
    };

    // Ctor.
    inline __device__ Smem_tile_dq_red_base(void* smem, int tidx)
        : smem_(__nvvm_get_smem_pointer(smem))
    {

        // Decompose the thread index into the position inside the warp.
        int warp = tidx / Cta_tile::THREADS_PER_WARP;
        int lane = tidx % Cta_tile::THREADS_PER_WARP;

        // The indices below assume 1 warp per CTA in the M dimension.
        static_assert(Cta_tile::WARPS_M == 1, "");

        // The first row/col written by that thread (we use STS.128).
        int write_row = lane / 4;
        int write_col = lane % 4 * 4 + warp * Cta_tile::N;

        // We apply a XOR pattern to avoid bank conflicts inside groups of 8 threads.
        if (Cta_tile::N >= 32)
        {
            write_col ^= (tidx & 0x04) * 4;
        }
        else if (Cta_tile::N == 16)
        {
            write_col ^= (tidx & 0x20) / 2 + (tidx & 0x04) * 4;
        }
        else
        {
            assert(false); // Not implemented!
        }

        write_col_ = write_col;

        // The offset to write to shared memory.
        write_offset_ = write_row * BYTES_PER_ROW + write_col * BYTES_PER_ELEMENT;

        // The first row/col read by each thread.
        int read_row = tidx / THREADS_PER_ROW;
        int read_col = tidx % THREADS_PER_ROW;

        // Take the XOR pattern into account.
        read_col ^= (read_row & 0x1) * 16;

        // The offset to read from shared memory.
        read_offset_ = read_row * BYTES_PER_ROW + read_col * BYTES_PER_ELEMENT;
    }

    // Load one row per thread. Apply the reduction.
    inline __device__ void load(float& out, int ii)
    {
        // The offset to the correct row in shared memory.
        int base_offset = read_offset_ + ii * ROWS_PER_LDS * BYTES_PER_ROW;
        // Apply the XOR pattern if we read a single row per LDS.
        if (ROWS_PER_LDS == 1)
        {
            base_offset ^= (ii & 0x1) * 16 * 4;
        }

        // Load the elements in the row and compute the reduction.
        out = 0.f;
#pragma unroll
        for (int jj = 0; jj < Cta_tile::WARPS_K; ++jj)
        {
            // The offset in shared memory.
            int offset = base_offset + jj * Cta_tile::N * BYTES_PER_ELEMENT;

            // For D == 16, we have to take the XOR pattern into account.
            if (Cta_tile::N >= 32)
            {
                // Nothing to do...
            }
            else if (Cta_tile::N == 16)
            {
                offset ^= (jj & 0x1) * 16 * 4;
            }
            else
            {
                assert(false); // Not implemented.
            }

            // Issue the LDS.32.
            uint32_t data;
            fmha::lds(data, smem_ + offset);

            // Accumulate.
            out += reinterpret_cast<float const&>(data);
        }
    }

    // Store one row of MMAs to shared memory.
    inline __device__ void store(Accumulators const (&acc)[Mma_tile::VALID_MMAS_N], int /*mi*/)
    {

        // The mi parameter is ignored -- TODO: Fix me!
        static_assert(Mma_tile::MMAS_M == 1, "");

        // There must be 8 registers per fragment.
        static_assert(Accumulators::NUM_REGS == 2 /*rows*/ * 4 /*cols*/, "");
#pragma unroll
        for (int ni = 0; ni < Mma_tile::VALID_MMAS_N; ++ni)
        {
            // Compute the write offset in bytes.
            int offset = write_offset_ + (ni / 2) * 2 * 16 * 4;

            // Apply the XOR pattern (in bytes).
            if (Cta_tile::N >= 32)
            {
                offset ^= (ni % 2) * 16 * 4;
            }
            else if (Cta_tile::N == 16)
            {
                // There is only one MMA in the N dimension - no need for a XOR.
            }
            else
            {
                assert(false); // Not implemented.
            }

// Store the 2 rows per MMA.
#pragma unroll
            for (int ii = 0; ii < 2; ++ii)
            {
                // Assemble the vector of elements.
                uint4 tmp;
                tmp.x = acc[ni].reg(ii * 2 + 0);
                tmp.y = acc[ni].reg(ii * 2 + 1);
                tmp.z = acc[ni].reg(ii * 2 + 4);
                tmp.w = acc[ni].reg(ii * 2 + 5);

                // Trigger the STS.128.
                fmha::sts(smem_ + offset + ii * 8 * BYTES_PER_ROW, tmp);
            }
        }
    }

    // The pointer to shared memory.
    uint32_t smem_;
    // The read offset. Reserve 4 offsets if needed.
    int read_offset_;
    // The write offset.
    int write_offset_;
    // write_col
    int write_col_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile>
struct Smem_tile_dq_red
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Smem_tile_dq_red<fmha::Ampere_hmma_fp32_traits, Cta_tile>
    : public Smem_tile_dq_red_base<fmha::Ampere_hmma_fp32_traits, Cta_tile>
{

    // The instruction traits.
    using Traits = fmha::Ampere_hmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_dq_red_base<Traits, Cta_tile>;

    // Ctor.
    inline __device__ Smem_tile_dq_red(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Smem_tile_dq_red<fmha::Ampere_hmma_bf16_traits, Cta_tile>
    : public Smem_tile_dq_red_base<fmha::Ampere_hmma_bf16_traits, Cta_tile>
{

    // The instruction traits.
    using Traits = fmha::Ampere_hmma_bf16_traits;
    // The base class.
    using Base = Smem_tile_dq_red_base<Traits, Cta_tile>;

    // Ctor.
    inline __device__ Smem_tile_dq_red(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
