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

#include <fmha/smem_tile_o.h>

namespace fmha
{

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Smem_tile_o_dummy
{
    enum
    {
        BYTES_PER_TILE = 0
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile>
struct Smem_tile_o_gmma_32bit_8bit : public Smem_tile_o_base_8bit_mma<Traits, Cta_tile>
{

    // The base class.
    using Base = Smem_tile_o_base_8bit_mma<Traits, Cta_tile>;

    using Mma_tile = typename Base::Mma_tile;
    using Accumulator = typename Base::Accumulator;

    enum
    {
        BYTES_PER_ROW = Base::BYTES_PER_ROW,
        BYTES_PER_ROW_WITH_PACKING = Base::BYTES_PER_ROW_WITH_PACKING,
        LOOPS = Base::LOOPS,
        LDS_PER_LOOP = Base::LDS_PER_LOOP,
        ROWS_PER_LDS = Base::ROWS_PER_LDS,
        HAS_INCOMPLETE_LDS = Base::HAS_INCOMPLETE_LDS,
    };

    // Ctor.
    inline __device__ Smem_tile_o_gmma_32bit_8bit(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }

    // Store the accumulators.
    inline __device__ void store(Accumulator const (&acc)[1][1], int mi)
    {

        enum
        {
            M_PER_MMA = Mma_tile::M_PER_MMA_PER_CTA
        };

        static_assert(M_PER_MMA == 64);
        static_assert(Base::WARPS_4x1x2);

        // The number of MMAs that are stored per loop iteration.
        enum
        {
            MMAS_M_PER_LOOP = Mma_tile::MMAS_M / LOOPS
        };

        static_assert(MMAS_M_PER_LOOP == 1);
        static_assert(Mma_tile::MMAS_N == 1);
        static_assert(Mma_tile::CORES_N == 8);
        static_assert(Accumulator::NUM_REGS == Mma_tile::CORES_N / 2 * 8);
        static_assert(BYTES_PER_ROW == 64 * 4);
        static_assert(Cta_tile::WARPS_K == 2);

        static_assert(Mma_tile::CORES_N / 2 == 4);

#pragma unroll
        for (int ni = 0; ni < Mma_tile::CORES_N / 2; ++ni)
        {

#pragma unroll
            for (int mj = 0; mj < MMAS_M_PER_LOOP; ++mj)
            {

                uint4 row_0;
                row_0.x = acc[0][0].reg(ni * 8 + 0); // Even
                row_0.y = acc[0][0].reg(ni * 8 + 4); // Odd
                row_0.z = acc[0][0].reg(ni * 8 + 1); // Even
                row_0.w = acc[0][0].reg(ni * 8 + 5); // Odd
                uint4 row_1;
                row_1.x = acc[0][0].reg(ni * 8 + 2); // Even
                row_1.y = acc[0][0].reg(ni * 8 + 6); // Odd
                row_1.z = acc[0][0].reg(ni * 8 + 3); // Even
                row_1.w = acc[0][0].reg(ni * 8 + 7); // Odd

                // Regs_to_rows<Traits>::extract(acc[mi * MMAS_M_PER_LOOP + mj][ni], row_0, row_1);

                // Each thread of a quad writes 16B per STS -> 64B per store. Account for 2 -> 128B.
                int imm_0 = (mj * M_PER_MMA + 0) * BYTES_PER_ROW * Cta_tile::WARPS_K + (ni / 2) * 128;
                int imm_1 = (mj * M_PER_MMA + 8) * BYTES_PER_ROW * Cta_tile::WARPS_K + (ni / 2) * 128;

                // Store the elements.
                fmha::sts(this->smem_write_ + imm_0, row_0);
                fmha::sts(this->smem_write_ + imm_1, row_1);
            }
            // Each thread of a quad writes 16B per STS -> 64B per store.
            if (Mma_tile::MMAS_N == 1)
            {
                this->smem_write_ ^= 64;
            }
            else
            {
                assert(false && "Unsupported");
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M, // GMMA instruction shape in M dim
    int GMMA_N,       // GMMA instruction shape in N dim
    int GMMA_K,       // GMMA instruction shape in K dim
    bool GMMA_A_RF,   // GMMA A operand coming from RF?
    bool GMMA_B_RF,   // GMMA B operand coming from RF?
    typename Cta_tile>
struct Smem_tile_o<Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile>
    : public Hmma_smem_tile_o<Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile>
{

    // The traits class.
    using Traits = Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>;
    // The base class.
    using Base = Hmma_smem_tile_o<Traits, Cta_tile>;

    using Mma_tile = typename Base::Mma_tile;

    using Accumulator = typename Base::Accumulator;

    enum
    {
        LOOPS = Base::LOOPS,
        ROW_PACKING = Base::ROW_PACKING,
        BYTES_PER_ROW = Base::BYTES_PER_ROW,
    };

    // Ctor.
    inline __device__ Smem_tile_o(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }

    // Store the accumulators.
    inline __device__ void store(Accumulator const (&acc)[1][1], int mi)
    {

        enum
        {
            M_PER_MMA = Mma_tile::M_PER_MMA_PER_CTA
        };

#pragma unroll
        for (int ni = 0; ni < Mma_tile::CORES_N; ++ni)
        {

            // The number of MMAs that are stored per loop iteration.
            enum
            {
                MMAS_M_PER_LOOP = Mma_tile::MMAS_M / LOOPS
            };

            static_assert(MMAS_M_PER_LOOP == 1);
            // inplace multiples seem to be 1, 3, 1, 7, 1, 3, 1,
            auto smem_write = this->smem_write_ ^ (ni * 16);
// Store 1st column of the different MMAs.
#pragma unroll
            for (int mj = 0; mj < MMAS_M_PER_LOOP; ++mj)
            {

                // Precompute the immediates to jump between rows.
                int row_0 = (mj * M_PER_MMA + 0) * BYTES_PER_ROW * Cta_tile::WARPS_K;
                int row_1 = (mj * M_PER_MMA + 8) * BYTES_PER_ROW * Cta_tile::WARPS_K;

                // Store.
                fmha::sts(smem_write + row_0, acc[0][0].reg(ni * 2 + 0));
                fmha::sts(smem_write + row_1, acc[0][0].reg(ni * 2 + 1));
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M, // GMMA instruction shape in M dim
    int GMMA_N,       // GMMA instruction shape in N dim
    int GMMA_K,       // GMMA instruction shape in K dim
    bool GMMA_A_RF,   // GMMA A operand coming from RF?
    bool GMMA_B_RF,   // GMMA B operand coming from RF?
    typename Cta_tile>
struct Smem_tile_o<Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile>
    : public Hmma_smem_tile_o<Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile>
{

    // The traits class.
    using Traits = Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>;
    // The base class.
    using Base = Hmma_smem_tile_o<Traits, Cta_tile>;

    using Mma_tile = typename Base::Mma_tile;

    using Accumulator = typename Base::Accumulator;

    enum
    {
        LOOPS = Base::LOOPS,
        ROW_PACKING = Base::ROW_PACKING,
        BYTES_PER_ROW = Base::BYTES_PER_ROW,
    };

    // Ctor.
    inline __device__ Smem_tile_o(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }

    // Store the accumulators.
    inline __device__ void store(Accumulator const (&acc)[1][1], int mi)
    {

        enum
        {
            M_PER_MMA = Mma_tile::M_PER_MMA_PER_CTA
        };

#pragma unroll
        for (int ni = 0; ni < Mma_tile::CORES_N; ++ni)
        {

            // The number of MMAs that are stored per loop iteration.
            enum
            {
                MMAS_M_PER_LOOP = Mma_tile::MMAS_M / LOOPS
            };

            static_assert(MMAS_M_PER_LOOP == 1);
            // inplace multiples seem to be 1, 3, 1, 7, 1, 3, 1,
            auto smem_write = this->smem_write_ ^ (ni * 16);
// Store 1st column of the different MMAs.
#pragma unroll
            for (int mj = 0; mj < MMAS_M_PER_LOOP; ++mj)
            {

                // Precompute the immediates to jump between rows.
                int row_0 = (mj * M_PER_MMA + 0) * BYTES_PER_ROW * Cta_tile::WARPS_K;
                int row_1 = (mj * M_PER_MMA + 8) * BYTES_PER_ROW * Cta_tile::WARPS_K;

                uint32_t val_0 = float2_to_half2(
                    acc[0][0].elt(2 * ni * Mma_tile::CORES_M + 0), acc[0][0].elt(2 * ni * Mma_tile::CORES_M + 1));

                uint32_t val_1 = float2_to_half2(
                    acc[0][0].elt(2 * ni * Mma_tile::CORES_M + 2), acc[0][0].elt(2 * ni * Mma_tile::CORES_M + 3));

                // Store.
                fmha::sts(smem_write + row_0, val_0);
                fmha::sts(smem_write + row_1, val_1);
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M, // GMMA instruction shape in M dim
    int GMMA_N,       // GMMA instruction shape in N dim
    int GMMA_K,       // GMMA instruction shape in K dim
    bool GMMA_A_RF,   // GMMA A operand coming from RF?
    bool GMMA_B_RF,   // GMMA B operand coming from RF?
    typename Cta_tile>
struct Smem_tile_o<Hopper_hgmma_bf16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile>
    : public Hmma_smem_tile_o<Hopper_hgmma_bf16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile>
{

    // The traits class.
    using Traits = Hopper_hgmma_bf16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>;
    // The base class.
    using Base = Hmma_smem_tile_o<Traits, Cta_tile>;

    using Mma_tile = typename Base::Mma_tile;

    using Accumulator = typename Base::Accumulator;

    enum
    {
        LOOPS = Base::LOOPS,
        ROW_PACKING = Base::ROW_PACKING,
        BYTES_PER_ROW = Base::BYTES_PER_ROW,
    };

    // Ctor.
    inline __device__ Smem_tile_o(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }

    // Convert fp32 to bf16, and store the accumulators.
    inline __device__ void store(Accumulator const (&acc)[1][1], int mi)
    {

        enum
        {
            M_PER_MMA = Mma_tile::M_PER_MMA_PER_CTA
        };

        static_assert(Mma_tile::CORES_M == 2);

#pragma unroll
        for (int ni = 0; ni < Mma_tile::CORES_N; ++ni)
        {

            // The number of MMAs that are stored per loop iteration.
            enum
            {
                MMAS_M_PER_LOOP = Mma_tile::MMAS_M / LOOPS
            };

            static_assert(MMAS_M_PER_LOOP == 1);
            // inplace multiples seem to be 1, 3, 1, 7, 1, 3, 1,
            auto smem_write = this->smem_write_ ^ (ni * 16);
// Store 1st column of the different MMAs.
#pragma unroll
            for (int mj = 0; mj < MMAS_M_PER_LOOP; ++mj)
            {

                // Precompute the immediates to jump between rows.
                int row_0 = (mj * M_PER_MMA + 0) * BYTES_PER_ROW * Cta_tile::WARPS_K;
                int row_1 = (mj * M_PER_MMA + 8) * BYTES_PER_ROW * Cta_tile::WARPS_K;

                uint32_t val_0 = float2_to_bf16_x2(
                    acc[0][0].elt(2 * ni * Mma_tile::CORES_M + 0), acc[0][0].elt(2 * ni * Mma_tile::CORES_M + 1));

                uint32_t val_1 = float2_to_bf16_x2(
                    acc[0][0].elt(2 * ni * Mma_tile::CORES_M + 2), acc[0][0].elt(2 * ni * Mma_tile::CORES_M + 3));

                // Store.
                fmha::sts(smem_write + row_0, val_0);
                fmha::sts(smem_write + row_1, val_1);
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M, // GMMA instruction shape in M dim
    int GMMA_N,       // GMMA instruction shape in N dim
    int GMMA_K,       // GMMA instruction shape in K dim
    bool GMMA_A_RF,   // GMMA A operand coming from RF?
    bool GMMA_B_RF,   // GMMA B operand coming from RF?
    typename Cta_tile>
struct Smem_tile_o<Hopper_qgmma_e4m3_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile>
    : public Smem_tile_o_gmma_32bit_8bit<Hopper_qgmma_e4m3_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
          Cta_tile>
{

    // The traits class.
    using Traits = Hopper_qgmma_e4m3_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>;
    // The base class.
    using Base = Smem_tile_o_gmma_32bit_8bit<Traits, Cta_tile>;

    // Ctor.
    inline __device__ Smem_tile_o(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

template <int GMMA_M, // GMMA instruction shape in M dim
    int GMMA_N,       // GMMA instruction shape in N dim
    int GMMA_K,       // GMMA instruction shape in K dim
    bool GMMA_A_RF,   // GMMA A operand coming from RF?
    bool GMMA_B_RF,   // GMMA B operand coming from RF?
    typename Cta_tile>
struct Smem_tile_o<Hopper_igmma_int8_int32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile>
    : public Smem_tile_o_gmma_32bit_8bit<Hopper_igmma_int8_int32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
          Cta_tile>
{

    // The traits class.
    using Traits = Hopper_igmma_int8_int32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>;
    // The base class.
    using Base = Smem_tile_o_gmma_32bit_8bit<Traits, Cta_tile>;

    // Ctor.
    inline __device__ Smem_tile_o(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha
