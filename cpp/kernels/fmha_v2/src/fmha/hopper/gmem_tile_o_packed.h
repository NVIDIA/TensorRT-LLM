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
#include <fmha/gmem_tile_o_packed.h>
#include <fmha/hopper/fragment.h>

namespace fmha
{

namespace v2
{

template <typename Traits, typename Cta_tile, int WARPS_K>
struct Gmem_tile_o_hopper
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Not super proud of this. Need to refactor.
// A not optimized way of storing tile_O, without SMEM swizzle.
// STG.32 is going to be used.
template <typename Traits, typename Cta_tile>
struct Gmem_tile_o_hopper_16bits
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
        BYTES_PER_ROW = Cta_tile::VALID_N * BYTES_PER_ELEMENT
    };

    // The number of rows accessed by each thread.
    enum
    {
        ROWS_PER_THREAD = Mma_tile::M_PER_MMA / 8 / Cta_tile::WARPS_PER_CTA
    };

    enum
    {
        ROWS = Cta_tile::M
    };

    // The number of columns access by each thread.
    // Note there are 2 elements per reg.
    enum
    {
        COLS_PER_THREAD = Mma_tile::N_PER_MMA / 4 / 2
    };

    // The number of valid columns (stored to GMEM) by each thread.
    enum
    {
        VALID_COLS_PER_THREAD_FOR_LAST_MMA = (Cta_tile::VALID_N % Mma_tile::N_PER_MMA) == 0
            ? COLS_PER_THREAD
            : (Cta_tile::VALID_N % Mma_tile::N_PER_MMA) / 8
    };

    enum
    {
        VALID_MMAS_N = fmha::Div_up<Cta_tile::VALID_N, Mma_tile::N_PER_MMA>::VALUE
    };

    static_assert(Cta_tile::VALID_N % 8 == 0, "The valid head dimension needs to be multiple of 8.");

    // The number of accumulator held by each thread, per HGMMA instruction.
    enum
    {
        ELTS_PER_THREAD = ROWS_PER_THREAD * COLS_PER_THREAD
    };

    // Currently, we assume for o matrix, GMMA M/N shape matches CTA M/N shape.
    static_assert(Mma_tile::M_PER_MMA == Cta_tile::M && Mma_tile::N_PER_MMA * Mma_tile::MMAS_N == Cta_tile::N,
        "Currently, we assume for o matrix, GMMA M shape matches CTA M shape. ");

    // Step N for one quad
    enum
    {
        STEP_N = 8 * BYTES_PER_ELEMENT
    };

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_o_hopper_16bits(
        Params const& params, Block_info const& block_info, int tidx, int cta_row_offset = 0)
        : params_o_stride_in_bytes_(params.o_stride_in_bytes)
        , actual_seqlen_(block_info.actual_seqlen)
        , o_ptr_(reinterpret_cast<char*>(params.o_ptr))
    {
        // Decompose the position of the thread into warp/lane.
        int warp = tidx / Cta_tile::THREADS_PER_WARP;
        int lane = tidx % Cta_tile::THREADS_PER_WARP;

        // int warpgroup_idx = warp / 4;
        int warp_idx_within_warpgroup = warp % 4;

        // Compute the position in the sequence (within the CTA for the moment).
        int row = warp_idx_within_warpgroup * (Mma_tile::M_PER_MMA / 4) + lane / 4;
        // Store the row to update the predicates in load.
        row_ = cta_row_offset + row;
        // Compute the position of the thread in the row.
        int col = lane % 4 * ELEMENTS_PER_STG;

        // The offset of the 1st row written by the thread. We store the P matrix interleaved.
        int64_t row_offset = (int64_t) row_ * params_o_stride_in_bytes_ + block_info.bidx * BYTES_PER_ROW;
        // Finalize the pointer.
        o_ptr_ += row_offset + col * BYTES_PER_ELEMENT;
    }

    // Store data to memory.
    template <typename Accumulators, int M, int N>
    inline __device__ void store(Accumulators const (&acc)[M][N])
    {
        int64_t const step_m = 8 * (this->params_o_stride_in_bytes_);
        // we assume M = 1. some shortcuts.
        static_assert(M == 1);
#pragma unroll
        for (int row_idx = 0; row_idx < ROWS_PER_THREAD; ++row_idx)
        {
            if (row_ + row_idx * 8 >= actual_seqlen_)
            {
                break;
            }
#pragma unroll
            for (int mma_ni = 0; mma_ni < VALID_MMAS_N - 1; ++mma_ni)
            {
#pragma unroll
                for (int col_idx = 0; col_idx < COLS_PER_THREAD; ++col_idx)
                {
                    uint32_t acc_0 = acc[0][mma_ni].reg(col_idx * ROWS_PER_THREAD + row_idx);

                    int64_t offset
                        = (int64_t) row_idx * step_m + (int64_t) (col_idx + mma_ni * COLS_PER_THREAD) * STEP_N;
                    fmha::stg(o_ptr_ + offset, acc_0);
                } // col_idx
            }     // mma_ni

            // The last mma_n may not store full elements back to GMEM.
            int mma_ni = VALID_MMAS_N - 1;
#pragma unroll
            for (int col_idx = 0; col_idx < VALID_COLS_PER_THREAD_FOR_LAST_MMA; ++col_idx)
            {
                uint32_t acc_0 = acc[0][mma_ni].reg(col_idx * ROWS_PER_THREAD + row_idx);

                int64_t offset = (int64_t) row_idx * step_m + (int64_t) (col_idx + mma_ni * COLS_PER_THREAD) * STEP_N;
                fmha::stg(o_ptr_ + offset, acc_0);
            } // col_idx
        }     // row_idx
    }

    // Move to the next location.
    inline __device__ void move()
    {
        row_ += ROWS;
        o_ptr_ += (int64_t) ROWS * params_o_stride_in_bytes_;
    }

    // The stride between rows for the QKV matrice.
    int64_t params_o_stride_in_bytes_;
    // The pointer.
    char* o_ptr_;
    // Is the thread active for the last STG?
    int is_active_for_last_stg_;

    // The row loaded by this thread.
    int row_;
    // The length of the sequence loaded by that CTA.
    int actual_seqlen_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M, int GMMA_N, int GMMA_K, bool GMMA_A_RF, bool GMMA_B_RF, typename Cta_tile>
struct Gmem_tile_o_hopper<fmha::Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile,
    1> // WARPS_K
    : public Gmem_tile_o_hopper_16bits<fmha::Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
          Cta_tile>
{
    using Traits = fmha::Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>;

    using Base = Gmem_tile_o_hopper_16bits<fmha::Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
        Cta_tile>;

    template <typename Params, typename Block_info, typename Shared>
    inline __device__ Gmem_tile_o_hopper(
        Params const& params, Block_info const& block_info, Shared&&, int tidx, int cta_row_offset = 0)
        : Base(params, block_info, tidx, cta_row_offset)
    {
        static_assert(!std::is_same<Shared, int>::value, "Check constructor argument type!");
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M, int GMMA_N, int GMMA_K, bool GMMA_A_RF, bool GMMA_B_RF, typename Cta_tile>
struct Gmem_tile_o_hopper<fmha::Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile,
    1> // WARPS_K
    : public Gmem_tile_o_hopper_16bits<fmha::Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
          Cta_tile>
{
    using Traits = fmha::Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>;

    using Base = Gmem_tile_o_hopper_16bits<fmha::Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
        Cta_tile>;

    using Mma_tile = typename Base::Mma_tile;

    template <typename Params, typename Block_info, typename Shared>
    inline __device__ Gmem_tile_o_hopper(
        Params const& params, Block_info const& block_info, Shared&&, int tidx, int cta_row_offset = 0)
        : Base(params, block_info, tidx, cta_row_offset)
    {
        static_assert(!std::is_same<Shared, int>::value, "Check constructor argument type!");
    }

    // Store data to memory.
    template <typename Accumulators, int M, int N>
    inline __device__ void store(Accumulators const (&acc)[M][N])
    {
        int64_t const step_m = 8 * (this->params_o_stride_in_bytes_);
        // we assume M = 1. some shortcuts.
        static_assert(M == 1);
#pragma unroll
        for (int row_idx = 0; row_idx < Base::ROWS_PER_THREAD; ++row_idx)
        {
            if (this->row_ + row_idx * 8 >= this->actual_seqlen_)
            {
                break;
            }
#pragma unroll
            for (int mma_ni = 0; mma_ni < Base::VALID_MMAS_N - 1; ++mma_ni)
            {
#pragma unroll
                for (int col_idx = 0; col_idx < Base::COLS_PER_THREAD; ++col_idx)
                {
                    // 2 denotes as fp32 --> fp16
                    float reg0 = acc[0][mma_ni].elt(2 * (col_idx * Base::ROWS_PER_THREAD + row_idx));
                    float reg1 = acc[0][mma_ni].elt(2 * (col_idx * Base::ROWS_PER_THREAD + row_idx) + 1);
                    uint32_t out = fmha::float2_to_half2(reg0, reg1);

                    int64_t offset = (int64_t) row_idx * step_m
                        + (int64_t) (col_idx + mma_ni * Base::COLS_PER_THREAD) * Base::STEP_N;
                    fmha::stg(this->o_ptr_ + offset, out);
                } // col_idx
            }     // mma_ni

            // The last mma_n may not store full elements back to GMEM.
            int mma_ni = Base::VALID_MMAS_N - 1;
#pragma unroll
            for (int col_idx = 0; col_idx < Base::VALID_COLS_PER_THREAD_FOR_LAST_MMA; ++col_idx)
            {
                // 2 denotes as fp32 --> fp16
                float reg0 = acc[0][mma_ni].elt(2 * (col_idx * Base::ROWS_PER_THREAD + row_idx));
                float reg1 = acc[0][mma_ni].elt(2 * (col_idx * Base::ROWS_PER_THREAD + row_idx) + 1);
                uint32_t out = fmha::float2_to_half2(reg0, reg1);

                int64_t offset
                    = (int64_t) row_idx * step_m + (int64_t) (col_idx + mma_ni * Base::COLS_PER_THREAD) * Base::STEP_N;
                fmha::stg(this->o_ptr_ + offset, out);
            } // col_idx
        }     // row_idx
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M, int GMMA_N, int GMMA_K, bool GMMA_A_RF, bool GMMA_B_RF, typename Cta_tile>
struct Gmem_tile_o_hopper<fmha::Hopper_hgmma_bf16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile,
    1> // WARPS_K
    : public Gmem_tile_o_hopper_16bits<fmha::Hopper_hgmma_bf16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
          Cta_tile>
{
    using Traits = fmha::Hopper_hgmma_bf16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>;

    using Base = Gmem_tile_o_hopper_16bits<fmha::Hopper_hgmma_bf16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
        Cta_tile>;

    using Mma_tile = typename Base::Mma_tile;

    template <typename Params, typename Block_info, typename Shared>
    inline __device__ Gmem_tile_o_hopper(
        Params const& params, Block_info const& block_info, Shared&&, int tidx, int cta_row_offset = 0)
        : Base(params, block_info, tidx, cta_row_offset)
    {
        static_assert(!std::is_same<Shared, int>::value, "Check constructor argument type!");
    }

    // Store data to memory.
    template <typename Accumulators, int M, int N>
    inline __device__ void store(Accumulators const (&acc)[M][N])
    {
        int64_t const step_m = 8 * (this->params_o_stride_in_bytes_);
        // we assume M = 1. some shortcuts.
        static_assert(M == 1);
#pragma unroll
        for (int row_idx = 0; row_idx < Base::ROWS_PER_THREAD; ++row_idx)
        {
            if (this->row_ + row_idx * 8 >= this->actual_seqlen_)
            {
                break;
            }
#pragma unroll
            for (int mma_ni = 0; mma_ni < Mma_tile::VALID_MMAS_N - 1; ++mma_ni)
            {
#pragma unroll
                for (int col_idx = 0; col_idx < Base::COLS_PER_THREAD; ++col_idx)
                {
                    // 2 denotes as fp32 --> bf16
                    float reg0 = acc[0][mma_ni].elt(2 * (col_idx * Base::ROWS_PER_THREAD + row_idx));
                    float reg1 = acc[0][mma_ni].elt(2 * (col_idx * Base::ROWS_PER_THREAD + row_idx) + 1);
                    uint32_t out = fmha::float2_to_bf16_x2(reg0, reg1);

                    int64_t offset = (int64_t) row_idx * step_m
                        + (int64_t) (col_idx + mma_ni * Base::COLS_PER_THREAD) * Base::STEP_N;
                    fmha::stg(this->o_ptr_ + offset, out);
                } // row_idx
            }     // col_idx

            // The last mma_n may not store full elements back to GMEM.
            int mma_ni = Base::VALID_MMAS_N - 1;
#pragma unroll
            for (int col_idx = 0; col_idx < Base::VALID_COLS_PER_THREAD_FOR_LAST_MMA; ++col_idx)
            {
                // 2 denotes as fp32 --> bf16
                float reg0 = acc[0][mma_ni].elt(2 * (col_idx * Base::ROWS_PER_THREAD + row_idx));
                float reg1 = acc[0][mma_ni].elt(2 * (col_idx * Base::ROWS_PER_THREAD + row_idx) + 1);
                uint32_t out = fmha::float2_to_bf16_x2(reg0, reg1);

                int64_t offset
                    = (int64_t) row_idx * step_m + (int64_t) (col_idx + mma_ni * Base::COLS_PER_THREAD) * Base::STEP_N;
                fmha::stg(this->o_ptr_ + offset, out);
            } // row_idx
        }     // mma_ni
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M, int GMMA_N, int GMMA_K, bool GMMA_A_RF, bool GMMA_B_RF, typename Cta_tile>
struct Gmem_tile_o_hopper<fmha::Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile,
    2> // WARPS_K
    : public fmha::v2::Hmma_gmem_tile_o<fmha::Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
          Cta_tile,
          /*CTAS_PER_HEAD=*/1,
          /*BYTES_PER_STG=*/16>
{
    using Traits = fmha::Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>;
    using Base = fmha::v2::Hmma_gmem_tile_o<Traits, Cta_tile, 1, 16>;

    template <typename Params, typename Block_info, typename Shared>
    inline __device__ Gmem_tile_o_hopper(
        Params const& params, Block_info const& block_info, Shared&&, int tidx, int cta_row_offset = 0)
        : Base(params, block_info, tidx, cta_row_offset)
    {
        static_assert(!std::is_same<Shared, int>::value, "Check constructor argument type!");
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M, int GMMA_N, int GMMA_K, bool GMMA_A_RF, bool GMMA_B_RF, typename Cta_tile>
struct Gmem_tile_o_hopper<fmha::Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile,
    2> // WARPS_K
    : public fmha::v2::Hmma_gmem_tile_o<fmha::Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
          Cta_tile,
          /*CTAS_PER_HEAD=*/1,
          /*BYTES_PER_STG=*/16>
{
    using Traits = fmha::Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>;
    using Base = fmha::v2::Hmma_gmem_tile_o<Traits, Cta_tile, 1, 16>;

    template <typename Params, typename Block_info, typename Shared>
    inline __device__ Gmem_tile_o_hopper(
        Params const& params, Block_info const& block_info, Shared&&, int tidx, int cta_row_offset = 0)
        : Base(params, block_info, tidx, cta_row_offset)
    {
        static_assert(!std::is_same<Shared, int>::value, "Check constructor argument type!");
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M, int GMMA_N, int GMMA_K, bool GMMA_A_RF, bool GMMA_B_RF, typename Cta_tile>
struct Gmem_tile_o_hopper<fmha::Hopper_hgmma_bf16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile,
    2> // WARPS_K
    : public fmha::v2::Hmma_gmem_tile_o<fmha::Hopper_hgmma_bf16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
          Cta_tile,
          /*CTAS_PER_HEAD=*/1,
          /*BYTES_PER_STG=*/16>
{
    using Traits = fmha::Hopper_hgmma_bf16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>;
    using Base = fmha::v2::Hmma_gmem_tile_o<Traits, Cta_tile, 1, 16>;

    template <typename Params, typename Block_info, typename Shared>
    inline __device__ Gmem_tile_o_hopper(
        Params const& params, Block_info const& block_info, Shared&&, int tidx, int cta_row_offset = 0)
        : Base(params, block_info, tidx, cta_row_offset)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M, int GMMA_N, int GMMA_K, bool GMMA_A_RF, bool GMMA_B_RF, typename Cta_tile, int CTAS_PER_HEAD>
struct Gmem_tile_o<fmha::Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile,
    CTAS_PER_HEAD>
    : public Gmem_tile_o_hopper<fmha::Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile,
          Cta_tile::WARPS_K>
{

    // The traits class.
    using Traits = fmha::Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>;

    using Base = Gmem_tile_o_hopper<fmha::Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
        Cta_tile, Cta_tile::WARPS_K>;

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_o(Params const& params, Block_info const& block_info, int tidx, int cta_row_offset = 0)
        : Base(params, block_info, std::nullptr_t{} /* dummy obj */, tidx, cta_row_offset)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M, int GMMA_N, int GMMA_K, bool GMMA_A_RF, bool GMMA_B_RF, typename Cta_tile, int CTAS_PER_HEAD>
struct Gmem_tile_o<fmha::Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile,
    CTAS_PER_HEAD>
    : public Gmem_tile_o_hopper<fmha::Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile,
          Cta_tile::WARPS_K>
{

    // The traits class.
    using Traits = fmha::Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>;

    using Base = Gmem_tile_o_hopper<fmha::Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
        Cta_tile, Cta_tile::WARPS_K>;

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_o(Params const& params, Block_info const& block_info, int tidx, int cta_row_offset = 0)
        : Base(params, block_info, std::nullptr_t{} /* dummy obj */, tidx, cta_row_offset)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M, int GMMA_N, int GMMA_K, bool GMMA_A_RF, bool GMMA_B_RF, typename Cta_tile, int CTAS_PER_HEAD>
struct Gmem_tile_o<fmha::Hopper_hgmma_bf16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile,
    CTAS_PER_HEAD>
    : public Gmem_tile_o_hopper<fmha::Hopper_hgmma_bf16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile,
          Cta_tile::WARPS_K>
{

    // The traits class.
    using Traits = fmha::Hopper_hgmma_bf16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>;

    using Base = Gmem_tile_o_hopper<fmha::Hopper_hgmma_bf16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
        Cta_tile, Cta_tile::WARPS_K>;

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_o(Params const& params, Block_info const& block_info, int tidx, int cta_row_offset = 0)
        : Base(params, block_info, std::nullptr_t{} /* dummy obj */, tidx, cta_row_offset)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, bool USE_TMA_STORE_ = false, int NUM_MATS = 1,
    bool HEADS_INTERLEAVED = false>
struct Gmem_tile_o_gmma_32bit_8bit
{
    static_assert(sizeof(typename Traits::Accumulator_type) == 4);
    static_assert(sizeof(typename Traits::C_type) == 1);
    // This is for non-splitk GMMA BMM2.
    static_assert(Cta_tile::WARPS_K == 1);
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
        BYTES_PER_ELEMENT = 1
    };

    // The size of each STG.
    enum
    {
        BYTES_PER_STG = ELEMENTS_PER_STG * BYTES_PER_ELEMENT
    };

    // The size of a row in bytes.
    enum
    {
        BYTES_PER_ROW = Cta_tile::VALID_N * BYTES_PER_ELEMENT
    };

    enum
    {
        ROWS = Cta_tile::M
    };

    // The number of rows accessed by each thread.
    enum
    {
        ROWS_PER_THREAD = Mma_tile::M_PER_MMA / 8 / Cta_tile::WARPS_M
    };

    static_assert(ROWS_PER_THREAD == 2);
    static_assert(ROWS_PER_THREAD == Mma_tile::ROWS_PER_THREAD);

    // The number of columns access by each thread.
    // The number of core matrices in N.
    enum
    {
        COLS_PER_THREAD = Mma_tile::N_PER_MMA / 4 / 2
    }; // N_PER_MMA = GMMA_N

    static_assert(COLS_PER_THREAD == Mma_tile::COLS_PER_THREAD / 2);
    // Assume there is an even number of core matrices, such that we can pack two
    static_assert(COLS_PER_THREAD % 2 == 0);

    // Number of valid N columns.
    enum
    {
        VALID_N = Cta_tile::VALID_N
    };

    // The number of valid columns (stored to GMEM) by each thread.
    enum
    {
        VALID_COLS_PER_THREAD_FOR_LAST_MMA
        = (VALID_N % Mma_tile::N_PER_MMA) == 0 ? COLS_PER_THREAD : (VALID_N % Mma_tile::N_PER_MMA) / 8
    };

    enum
    {
        VALID_MMAS_N = fmha::Div_up<VALID_N, Mma_tile::N_PER_MMA>::VALUE
    };

    static_assert(VALID_N % 8 == 0, "The valid head dimension needs to be multiple of 8.");

    // The number of N elements must be multiple of 16 in order to pack 4 elements as uint32_t.
    enum
    {
        PACK_4_ELTS = VALID_N % 16 == 0
    };

    // The number of accumulator held by each thread, per HGMMA instruction.
    enum
    {
        ELTS_PER_THREAD = ROWS_PER_THREAD * COLS_PER_THREAD * 2
    };

    // Currently, we assume for o matrix, GMMA M shape matches CTA M shape.
    static_assert(Mma_tile::M_PER_MMA == Cta_tile::M && Mma_tile::N_PER_MMA * Mma_tile::MMAS_N == Cta_tile::N,
        "Currently, we assume for o matrix, GMMA M/N shape matches CTA M/N shape. ");

    // Step N for one quad (pack 4 elements for a thread, so 16 elements for a quad)
    enum
    {
        STEP_N = 16 * BYTES_PER_ELEMENT
    };

    // The number of head_dimension groups.
    enum
    {
        N_GROUPS = fmha::Div_up<Cta_tile::N * BYTES_PER_ELEMENT, 128>::VALUE
    };

    // The head_dimension per group.
    enum
    {
        N_PER_GROUP = Cta_tile::N / N_GROUPS
    };

    static_assert(N_GROUPS * N_PER_GROUP == Cta_tile::N);

    // The head_dimension bytes per group
    enum
    {
        N_BYTES_PER_GROUP = Cta_tile::N * BYTES_PER_ELEMENT / N_GROUPS
    };

    // Pack 2x4 core matrices, use STSMx4
    enum
    {
        STSM_PER_MMA = COLS_PER_THREAD / 4
    };

    // The number of registers per 16x16 block
    enum
    {
        REGS_PER_QUAD = 8
    };

    // Bytes per bank
    enum
    {
        BYTES_PER_BANK = 16
    };

    // The number of banks in N per group
    enum
    {
        N_BANKS_PER_GROUP = N_BYTES_PER_GROUP / BYTES_PER_BANK
    };

    enum
    {
        USE_TMA_STORE = USE_TMA_STORE_
    };

    // Ctor.
    template <typename Params, typename Block_info, typename Shared>
    inline __device__ Gmem_tile_o_gmma_32bit_8bit(
        Params const& params, Block_info const& block_info, Shared& shared, int tidx, int cta_row_offset = 0)
        : Gmem_tile_o_gmma_32bit_8bit(params.o_ptr, params.o_stride_in_bytes, block_info, tidx,
#ifdef GENERATE_CUBIN
            // Specialized for trt-llm generated cubins only.
            params.scale_bmm2_d ? *params.scale_bmm2_d : params.scale_bmm2,
#else
            params.scale_bmm2,
#endif
            cta_row_offset, 0,
            __nvvm_get_smem_pointer(
                reinterpret_cast<void*>(&shared.smem_o[__shfl_sync(0xffffffff, threadIdx.x / 128, 0)][0])),
            &params.tma_desc_o, params.h)
    {
    }

    template <typename Block_info>
    inline __device__ Gmem_tile_o_gmma_32bit_8bit(void* o_ptr, int o_stride_in_bytes, Block_info const& block_info,
        int tidx, uint32_t scale_bmm2, int cta_row_offset = 0, int mat_offset = 0, uint32_t smem_base = 0,
        cudaTmaDesc const* desc_o = nullptr, int head_num = 0)
        : params_o_stride_in_bytes_(o_stride_in_bytes)
        , actual_seqlen_(block_info.actual_seqlen)
        , o_ptr_(reinterpret_cast<char*>(o_ptr))
        , params_scale_bmm2_(scale_bmm2)
        , smem_base_(smem_base)
        , desc_o_(desc_o)
    {

        // Decompose the position of the thread into warp/lane.
        int warp = tidx / Cta_tile::THREADS_PER_WARP;
        int lane = tidx % Cta_tile::THREADS_PER_WARP;

        // int warpgroup_idx = warp / 4;
        int warp_idx_within_warpgroup = warp % 4;

        if (USE_TMA_STORE)
        {
            // The head index
            bidh_ = block_info.bidh;
            // The lane id
            lane_ = lane;
            // The start row index for current batch
            int row_curr_batch = (block_info.bidx - block_info.bidh) / head_num;
            // The row index offset of current warp
            int row_offset_warp = cta_row_offset + warp_idx_within_warpgroup * (Mma_tile::M_PER_MMA / 4);
            // The row index for the current warp
            row_tma_ = row_offset_warp + row_curr_batch;
            // The valid rows for the current warp. Each warp writes from 0 to 16 rows
            num_valid_rows_ = min(Mma_tile::M_PER_MMA / 4, actual_seqlen_ - row_offset_warp);
            num_valid_rows_ = max(num_valid_rows_, 0);
            // WARNING: Without this line, the predicate will not behavior as expected for unknown reason.
            num_valid_rows_ = __shfl_sync(0xffffffff, num_valid_rows_, 0);
            // Compute the smem base for STSM
            smem_base_ += warp_idx_within_warpgroup * (Mma_tile::M_PER_MMA / 4) * Cta_tile::N * BYTES_PER_ELEMENT
                + (warp / 4) * Mma_tile::M_PER_MMA * Cta_tile::N * BYTES_PER_ELEMENT;
            // Compute gmem base for STG in tail case
            o_ptr_ += row_tma_ * params_o_stride_in_bytes_ + bidh_ * BYTES_PER_ROW;
        }
        else
        {
            // Compute the position in the sequence (within the CTA for the moment).
            int row = warp_idx_within_warpgroup * (Mma_tile::M_PER_MMA / 4) + lane / 4;
            // Store the row to update the predicates in load.
            row_ = cta_row_offset + row;
            // Compute the position of the thread in the row.
            col_ = lane % 4 * ELEMENTS_PER_STG;

            // The offset of the 1st row written by the thread. We store the P matrix interleaved.
            int64_t row_offset = (int64_t) row_ * params_o_stride_in_bytes_ + block_info.bidx * BYTES_PER_ROW;
            // Finalize the pointer.
            o_ptr_ += row_offset + col_ * BYTES_PER_ELEMENT;
        }

        // REVIEW: need heads_interleaved option for non-warp-specialized QGMMA + LDGSTS kernels.
        // // The row offset in the batched GEMM. For each seq element, we store QKV in that order.
        // int64_t row_offset = (int64_t) row_ * params_o_stride_in_bytes_;
        // // Add the block index.
        // int64_t idx = block_info.bidx;
        // if(NUM_MATS > 1) {
        //     if( HEADS_INTERLEAVED ) {
        //         idx = block_info.bidx * NUM_MATS + mat_offset;
        //     } else {
        //         idx = (block_info.sum_s * NUM_MATS + mat_offset) * block_info.num_heads + block_info.bidh;
        //     }
        // }
        // // Assemble the final pointer.
        // o_ptr_ += row_offset + idx * BYTES_PER_ROW + col * BYTES_PER_ELEMENT;
    }

    // Store data to memory.
    template <typename Accumulators, int M, int N>
    inline __device__ void store(Accumulators const (&acc)[M][N])
    {

        static_assert(Accumulators::NUM_ELTS == ELTS_PER_THREAD);
        static_assert(COLS_PER_THREAD / 2 * ROWS_PER_THREAD * 4 == ELTS_PER_THREAD);

        // we assume M = N = 1. some shortcuts.
        static_assert(M == 1);

        if (USE_TMA_STORE)
        {
            static_assert(COLS_PER_THREAD % 4 == 0);
            static_assert(ROWS_PER_THREAD == 2);

            int const swizzled_row = (lane_ % 16);
            int const swizzled_col = (lane_ / 16);
            constexpr int max_swizzle_id = N_BYTES_PER_GROUP / 16;
            constexpr int swizzle_row_divider = 128 / N_BYTES_PER_GROUP;

            uint32_t stsm_addr[VALID_MMAS_N][STSM_PER_MMA];
// Compute swizzled smem address
#pragma unroll
            for (int mma_ni = 0; mma_ni < VALID_MMAS_N; ++mma_ni)
            {
#pragma unroll
                for (int ci = 0; ci < STSM_PER_MMA; ++ci)
                {
                    int const col_bank = ((mma_ni) *STSM_PER_MMA + ci) * 2 + swizzled_col;
                    int const di = col_bank / N_BANKS_PER_GROUP;                       // which N group it belongs to
                    stsm_addr[mma_ni][ci] = smem_base_ + di * 16 * N_BYTES_PER_GROUP + // group dimension
                        (((swizzled_row / swizzle_row_divider) % max_swizzle_id) ^ (col_bank % N_BANKS_PER_GROUP))
                            * BYTES_PER_BANK
                        +                                 // column dimension
                        swizzled_row * N_BYTES_PER_GROUP; // row dimension
                }
            }

#pragma unroll
            for (int mma_ni = 0; mma_ni < VALID_MMAS_N; ++mma_ni)
            {
#pragma unroll
                for (int ci = 0; ci < STSM_PER_MMA; ++ci)
                {

                    uint32_t dst[4];
                    uint4 src[4];

                    /*
                     * Each STSMx4 produces a 16x32 block, that is 2x4 core matrices
                     * -----------------
                     * | 0 | 2 | 4 | 6 |
                     * -----------------
                     * | 1 | 3 | 5 | 7 |
                     * -----------------
                     *
                     * Consider the entire warp, src[0] holds matrices 0,2; src[1] holds matrices 1,3;
                     * src[3] holds matrices 4,6; src[4] holds matrices 5,7.
                     */
                    src[0].x = acc[0][mma_ni].reg((ci * 2 + 0) * REGS_PER_QUAD + 0);
                    src[0].y = acc[0][mma_ni].reg((ci * 2 + 0) * REGS_PER_QUAD + 4);
                    src[0].z = acc[0][mma_ni].reg((ci * 2 + 0) * REGS_PER_QUAD + 1);
                    src[0].w = acc[0][mma_ni].reg((ci * 2 + 0) * REGS_PER_QUAD + 5);

                    src[1].x = acc[0][mma_ni].reg((ci * 2 + 0) * REGS_PER_QUAD + 2);
                    src[1].y = acc[0][mma_ni].reg((ci * 2 + 0) * REGS_PER_QUAD + 6);
                    src[1].z = acc[0][mma_ni].reg((ci * 2 + 0) * REGS_PER_QUAD + 3);
                    src[1].w = acc[0][mma_ni].reg((ci * 2 + 0) * REGS_PER_QUAD + 7);

                    src[2].x = acc[0][mma_ni].reg((ci * 2 + 1) * REGS_PER_QUAD + 0);
                    src[2].y = acc[0][mma_ni].reg((ci * 2 + 1) * REGS_PER_QUAD + 4);
                    src[2].z = acc[0][mma_ni].reg((ci * 2 + 1) * REGS_PER_QUAD + 1);
                    src[2].w = acc[0][mma_ni].reg((ci * 2 + 1) * REGS_PER_QUAD + 5);

                    src[3].x = acc[0][mma_ni].reg((ci * 2 + 1) * REGS_PER_QUAD + 2);
                    src[3].y = acc[0][mma_ni].reg((ci * 2 + 1) * REGS_PER_QUAD + 6);
                    src[3].z = acc[0][mma_ni].reg((ci * 2 + 1) * REGS_PER_QUAD + 3);
                    src[3].w = acc[0][mma_ni].reg((ci * 2 + 1) * REGS_PER_QUAD + 7);

                    using Src_type = typename Traits::Accumulator_type;
                    using Dst_type = typename Traits::C_type;
// Packs the 32bit values to 8bit.
// Depending on the type, applies extra scaling with parameter scale_bmm2.
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                    {
#ifdef UNIFIED_EPILOGUE_SCALE
                        dst[i] = Acc_packer<Src_type, Dst_type, false>::run(this, src[i]);
#else
                        dst[i] = Acc_packer<Src_type, Dst_type, true>::run(this, src[i]);
#endif
                    }
                    stsm(stsm_addr[mma_ni][ci], *reinterpret_cast<uint4*>(&dst[0]));
                }
            }

            // TODO: Interleave STSM and UTMASTG of two N groups
            constexpr int MAX_ROWS_PER_WARP = Mma_tile::M_PER_MMA / 4;
            if (num_valid_rows_ == MAX_ROWS_PER_WARP)
            {
                fence_view_async_shared();
#pragma unroll
                for (int di = 0; di < N_GROUPS; ++di)
                {
                    const int32_t coords[3] = {di * N_PER_GROUP, bidh_, row_tma_};
                    fmha::utmastg<3, fmha::cudaTmaDescType::TILED>(
                        desc_o_, smem_base_ + di * 16 * N_BYTES_PER_GROUP, coords);
                }
                tmastg_arrive();
                tmastg_wait();
            }
            else if (num_valid_rows_ > 0)
            {
                // Use LDS.64 + STG.64 to store num_valid_rows_ x N tile
                constexpr int BYTES_PER_THREAD = 8;
                static_assert((VALID_N % BYTES_PER_THREAD) == 0, "VALID_N must be divided by 8 for STG.64");
                // Number of valid rows
                int row_size = num_valid_rows_;
                // Number of threads per row. Each thread read/write 8B (8 elements).
                constexpr int THREADS_PER_ROW = N_BYTES_PER_GROUP / 8;
                // Number of rows read/written by a warp
                static_assert(Cta_tile::THREADS_PER_WARP % THREADS_PER_ROW == 0, "A warp must reads full rows");
                constexpr int ROWS_PER_WARP = Cta_tile::THREADS_PER_WARP / THREADS_PER_ROW;
                // GMEM stride in M dimension
                int64_t const step_m = (this->params_o_stride_in_bytes_);
                // Initial column index
                int const ci = lane_ % THREADS_PER_ROW;
                int const bank_idx = (ci * BYTES_PER_THREAD) / BYTES_PER_BANK;
                int const bank_offset = (ci * BYTES_PER_THREAD) % BYTES_PER_BANK;

#pragma unroll
                for (int di = 0; di < N_GROUPS; ++di)
                {
                    // Detect GMEM index out of bound
                    if ((di * N_BYTES_PER_GROUP + ci * BYTES_PER_THREAD) >= BYTES_PER_ROW)
                    {
                        break;
                    }
#pragma unroll
                    for (int ri = lane_ / THREADS_PER_ROW; ri < row_size; ri += ROWS_PER_WARP)
                    {
                        // Create the swizzled offset
                        uint32_t smem_offset = di * 16 * N_BYTES_PER_GROUP + ri * N_BYTES_PER_GROUP
                            + (((ri / swizzle_row_divider) % max_swizzle_id) ^ bank_idx) * BYTES_PER_BANK + bank_offset;
                        uint2 buffer;
                        fmha::lds(buffer, smem_base_ + smem_offset);
                        int64_t gmem_offset = (int64_t) ri * step_m + di * N_BYTES_PER_GROUP + ci * BYTES_PER_THREAD;
                        fmha::stg(o_ptr_ + gmem_offset, buffer);
                    }
                }
            }
        }
        else
        {
            int64_t const step_m = 8 * (this->params_o_stride_in_bytes_);

#pragma unroll
            for (int ri = 0; ri < ROWS_PER_THREAD; ++ri)
            {
                if (row_ + ri * 8 >= actual_seqlen_)
                {
                    break;
                }

#pragma unroll
                for (int mma_ni = 0; mma_ni < VALID_MMAS_N - 1; ++mma_ni)
                {
// Iterate over 16 columns to pack 4 values per thread.
#pragma unroll
                    for (int ci = 0; ci < COLS_PER_THREAD / 2; ++ci)
                    {
                        // Assuming EVEN,EVEN,ODD,ODD column pattern due to packing of V.
                        uint4 src;
                        src.x = acc[0][mma_ni].reg(((2 * ci + 0) * ROWS_PER_THREAD + ri) * 2 + 0); // 0
                        src.y = acc[0][mma_ni].reg(((2 * ci + 1) * ROWS_PER_THREAD + ri) * 2 + 0); // 4
                        src.z = acc[0][mma_ni].reg(((2 * ci + 0) * ROWS_PER_THREAD + ri) * 2 + 1); // 1
                        src.w = acc[0][mma_ni].reg(((2 * ci + 1) * ROWS_PER_THREAD + ri) * 2 + 1); // 5

                        using Src_type = typename Traits::Accumulator_type;
                        using Dst_type = typename Traits::C_type;
                        // Packs the 32bit values to 8bit.
                        // Depending on the type, applies extra scaling with parameter scale_bmm2.
#ifdef UNIFIED_EPILOGUE_SCALE
                        uint32_t dst = Acc_packer<Src_type, Dst_type, false>::run(this, src);
#else
                        uint32_t dst = Acc_packer<Src_type, Dst_type, true>::run(this, src);
#endif

                        int64_t offset = (int64_t) ri * step_m + (int64_t) (ci + mma_ni * COLS_PER_THREAD / 2) * STEP_N;
                        fmha::stg(o_ptr_ + offset, dst);
                    } // ci
                }     // mma_ni

                if constexpr (PACK_4_ELTS)
                {
                    // The last mma_n may not store full elements back to GMEM.
                    int mma_ni = VALID_MMAS_N - 1;
// Iterate over 16 columns to pack 4 values per thread.
#pragma unroll
                    for (int ci = 0; ci < VALID_COLS_PER_THREAD_FOR_LAST_MMA / 2; ++ci)
                    {
                        // Assuming EVEN,EVEN,ODD,ODD column pattern due to packing of V.
                        uint4 src;
                        src.x = acc[0][mma_ni].reg(((2 * ci + 0) * ROWS_PER_THREAD + ri) * 2 + 0); // 0
                        src.y = acc[0][mma_ni].reg(((2 * ci + 1) * ROWS_PER_THREAD + ri) * 2 + 0); // 4
                        src.z = acc[0][mma_ni].reg(((2 * ci + 0) * ROWS_PER_THREAD + ri) * 2 + 1); // 1
                        src.w = acc[0][mma_ni].reg(((2 * ci + 1) * ROWS_PER_THREAD + ri) * 2 + 1); // 5

                        using Src_type = typename Traits::Accumulator_type;
                        using Dst_type = typename Traits::C_type;
                        // Packs the 32bit values to 8bit.
                        // Depending on the type, applies extra scaling with parameter scale_bmm2.
#ifdef UNIFIED_EPILOGUE_SCALE
                        uint32_t dst = Acc_packer<Src_type, Dst_type, false>::run(this, src);
#else
                        uint32_t dst = Acc_packer<Src_type, Dst_type, true>::run(this, src);
#endif

                        int64_t offset = (int64_t) ri * step_m + (int64_t) (ci + mma_ni * COLS_PER_THREAD / 2) * STEP_N;
                        fmha::stg(o_ptr_ + offset, dst);
                    } // ci
                }
                else
                {

                    // The last mma_n may not store full elements back to GMEM.
                    int mma_ni = VALID_MMAS_N - 1;
// Iterate over 16 columns to pack 4 values per thread (2 uint2).
#pragma unroll
                    for (int ci = 0; ci < fmha::Div_up<VALID_COLS_PER_THREAD_FOR_LAST_MMA, 2>::VALUE; ++ci)
                    {
                        // Assuming EVEN,EVEN,ODD,ODD column pattern due to packing of V.
                        uint2 src0, src1;
                        src0.x = acc[0][mma_ni].reg(((2 * ci + 0) * ROWS_PER_THREAD + ri) * 2 + 0); // 0
                        src0.y = acc[0][mma_ni].reg(((2 * ci + 1) * ROWS_PER_THREAD + ri) * 2 + 0); // 4
                        src1.x = acc[0][mma_ni].reg(((2 * ci + 0) * ROWS_PER_THREAD + ri) * 2 + 1); // 1
                        src1.y = acc[0][mma_ni].reg(((2 * ci + 1) * ROWS_PER_THREAD + ri) * 2 + 1); // 5

                        using Src_type = typename Traits::Accumulator_type;
                        using Dst_type = typename Traits::C_type;
#ifdef UNIFIED_EPILOGUE_SCALE
                        uint16_t dst0 = Acc_packer<Src_type, Dst_type, false>::run(this, src0);
                        uint16_t dst1 = Acc_packer<Src_type, Dst_type, false>::run(this, src1);
#else
                        uint16_t dst0 = Acc_packer<Src_type, Dst_type, true>::run(this, src0);
                        uint16_t dst1 = Acc_packer<Src_type, Dst_type, true>::run(this, src1);
#endif

                        // 4 elements per thread, so 16 elements per loop.
                        int col_idx = (ci + mma_ni * COLS_PER_THREAD / 2) * 16;

                        int64_t offset = (int64_t) ri * step_m + (int64_t) (col_idx) *BYTES_PER_ELEMENT;

                        if (col_idx + col_ < VALID_N)
                        {
                            fmha::stg(o_ptr_ + offset, dst0);
                        }

                        if (col_idx + col_ + 2 < VALID_N)
                        {
                            fmha::stg(o_ptr_ + offset + 2 * BYTES_PER_ELEMENT, dst1);
                        }
                    } // ci
                }
            }         // ri
        }
    }

    // Move to the next location.
    inline __device__ void move()
    {
        row_ += ROWS;
        o_ptr_ += (int64_t) ROWS * params_o_stride_in_bytes_;
    }

    // The stride between rows for the QKV matrice.
    int64_t params_o_stride_in_bytes_;
    // The pointer.
    char* o_ptr_;
    // Is the thread active for the last STG?
    int is_active_for_last_stg_;

    // The row, col loaded by this thread.
    int row_, col_;
    // The length of the sequence loaded by that CTA.
    int actual_seqlen_;

    // Scaling factor; this usually means QKV descale factor in actuality
    uint32_t params_scale_bmm2_;

    // Smem buffer for TMASTG
    uint32_t smem_base_;
    cudaTmaDesc const* desc_o_;

    int lane_;
    int row_tma_;
    int num_valid_rows_;
    int bidh_;

    bool const params_enable_i2f_trick_ = false;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, int WARPS_K, bool USE_TMA_STORE = false>
struct Gmem_tile_o_hopper_32bit_8bit
{
};

template <typename Traits, typename Cta_tile, bool USE_TMA_STORE>
struct Gmem_tile_o_hopper_32bit_8bit<Traits, Cta_tile, 1, USE_TMA_STORE>
    : public Gmem_tile_o_gmma_32bit_8bit<Traits, Cta_tile, USE_TMA_STORE>
{

    // The Base class.
    using Base = Gmem_tile_o_gmma_32bit_8bit<Traits, Cta_tile, USE_TMA_STORE>;

    // Ctor.
    template <typename Params, typename Block_info, typename Shared>
    inline __device__ Gmem_tile_o_hopper_32bit_8bit(
        Params const& params, Block_info const& block_info, Shared& shared, int tidx, int cta_row_offset = 0)
        : Base(params, block_info, shared, tidx, cta_row_offset)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, bool USE_TMA_STORE>
struct Gmem_tile_o_hopper_32bit_8bit<Traits, Cta_tile, 2, USE_TMA_STORE>
    : public Gmem_tile_o_8bit<Traits, Cta_tile, /*CTAS_PER_HEAD=*/1>
{

    // The Base class.
    using Base = Gmem_tile_o_8bit<Traits, Cta_tile, /*CTAS_PER_HEAD=*/1>;

    // Ctor.
    template <typename Params, typename Block_info, typename Shared>
    inline __device__ Gmem_tile_o_hopper_32bit_8bit(
        Params const& params, Block_info const& block_info, Shared& shared, int tidx, int cta_row_offset = 0)
        : Base(params, block_info, shared, tidx, cta_row_offset)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M, int GMMA_N, int GMMA_K, bool GMMA_A_RF, bool GMMA_B_RF, typename Cta_tile, int CTAS_PER_HEAD>
struct Gmem_tile_o_hopper<fmha::Hopper_qgmma_fp8_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile,
    CTAS_PER_HEAD>
    : public Gmem_tile_o_hopper_32bit_8bit<
          fmha::Hopper_qgmma_fp8_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile, Cta_tile::WARPS_K>
{

    // The traits class.
    using Traits = fmha::Hopper_qgmma_fp8_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>;

    using Base = Gmem_tile_o_hopper_32bit_8bit<
        fmha::Hopper_qgmma_fp8_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile, Cta_tile::WARPS_K>;

    // Ctor.
    template <typename Params, typename Block_info, typename Shared>
    inline __device__ Gmem_tile_o_hopper(
        Params const& params, Block_info const& block_info, Shared& shared, int tidx, int cta_row_offset = 0)
        : Base(params, block_info, shared, tidx, cta_row_offset)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M, int GMMA_N, int GMMA_K, bool GMMA_A_RF, bool GMMA_B_RF, typename Cta_tile, int CTAS_PER_HEAD>
struct Gmem_tile_o<fmha::Hopper_igmma_int8_int32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile,
    CTAS_PER_HEAD>
    : public Gmem_tile_o_hopper_32bit_8bit<
          fmha::Hopper_igmma_int8_int32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile,
          Cta_tile::WARPS_K>
{

    // The traits class.
    using Traits = fmha::Hopper_igmma_int8_int32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>;

    using Base = Gmem_tile_o_hopper_32bit_8bit<
        fmha::Hopper_igmma_int8_int32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile,
        Cta_tile::WARPS_K>;

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_o(Params const& params, Block_info const& block_info, int tidx, int cta_row_offset = 0)
        : Base(params, block_info, std::nullptr_t{} /* dummy obj */, tidx, cta_row_offset)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, typename Output_type = bf16_t>
struct Gmem_tile_o_qgmma_fp32_16bits
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
        BYTES_PER_ROW = Cta_tile::VALID_N * BYTES_PER_ELEMENT
    };

    // The number of rows accessed by each thread.
    enum
    {
        ROWS_PER_THREAD = Mma_tile::M_PER_MMA / 8 / Cta_tile::WARPS_PER_CTA
    };

    enum
    {
        ROWS = Cta_tile::M
    };

    // The number of columns access by each thread.
    // Note there are 2 elements per reg.
    enum
    {
        COLS_PER_THREAD = Mma_tile::N_PER_MMA / 4 / 2
    };

    // The number of valid columns (stored to GMEM) by each thread.
    enum
    {
        VALID_COLS_PER_THREAD_FOR_LAST_MMA = (Cta_tile::VALID_N % Mma_tile::N_PER_MMA) == 0
            ? COLS_PER_THREAD
            : (Cta_tile::VALID_N % Mma_tile::N_PER_MMA) / 8
    };

    enum
    {
        VALID_MMAS_N = fmha::Div_up<Cta_tile::VALID_N, Mma_tile::N_PER_MMA>::VALUE
    };

    static_assert(Cta_tile::VALID_N % 8 == 0, "The valid head dimension needs to be multiple of 8.");

    // The number of accumulator held by each thread, per HGMMA instruction.
    enum
    {
        ELTS_PER_THREAD = ROWS_PER_THREAD * COLS_PER_THREAD
    };

    // Currently, we assume for o matrix, GMMA M/N shape matches CTA M/N shape.
    static_assert(Mma_tile::M_PER_MMA == Cta_tile::M && Mma_tile::N_PER_MMA * Mma_tile::MMAS_N == Cta_tile::N,
        "Currently, we assume for o matrix, GMMA M shape matches CTA M shape. ");

    // Step N for one quad
    enum
    {
        STEP_N = 8 * BYTES_PER_ELEMENT
    };

    // Ctor.
    template <typename Params, typename Block_info, typename Shared>
    inline __device__ Gmem_tile_o_qgmma_fp32_16bits(
        Params const& params, Block_info const& block_info, Shared&&, int tidx, int cta_row_offset = 0)
        : params_o_stride_in_bytes_(params.o_stride_in_bytes)
        , params_scale_bmm2_(
#ifdef GENERATE_CUBIN
              // Specialized for trt-llm generated cubins only.
              params.scale_bmm2_d ? *params.scale_bmm2_d : params.scale_bmm2
#else
              params.scale_bmm2
#endif
              )
        , actual_seqlen_(block_info.actual_seqlen)
        , o_ptr_(reinterpret_cast<char*>(params.o_ptr))
    {
        static_assert(!std::is_same<Shared, int>::value, "Check constructor argument type!");
        // Decompose the position of the thread into warp/lane.
        int warp = tidx / Cta_tile::THREADS_PER_WARP;
        int lane = tidx % Cta_tile::THREADS_PER_WARP;

        int warp_idx_within_warpgroup = warp % 4;

        // Compute the position in the sequence (within the CTA for the moment).
        int row = warp_idx_within_warpgroup * (Mma_tile::M_PER_MMA / 4) + lane / 4;
        // Store the row to update the predicates in load.
        row_ = cta_row_offset + row;
        // Compute the position of the thread in the row.
        // echo loop handles 2 cores, so x2 (this is the difference to Gmem_tile_o_hopper_16bits)
        int col = lane % 4 * ELEMENTS_PER_STG * 2;

        // The offset of the 1st row written by the thread. We store the P matrix interleaved.
        int64_t row_offset = (int64_t) row_ * params_o_stride_in_bytes_ + block_info.bidx * BYTES_PER_ROW;
        // Finalize the pointer.
        o_ptr_ += row_offset + col * BYTES_PER_ELEMENT;
    }

    // Store data to memory.
    template <typename Accumulators, int M, int N>
    inline __device__ void store(Accumulators const (&acc)[M][N])
    {
        int64_t const step_m = 8 * params_o_stride_in_bytes_;
#ifdef UNIFIED_EPILOGUE_SCALE
        constexpr bool Scale = false;
#else
        constexpr bool Scale = true;
#endif
#define STORE_COLUMNS()                                                                                                \
    {                                                                                                                  \
        /* we assume M = 1. some shortcuts. */                                                                         \
        static_assert(M == 1);                                                                                         \
        uint4 _src = {                                                                                                 \
            .x = acc[0][mma_ni].reg(((ci + 0) * ROWS_PER_THREAD + ri) * 2),                                            \
            .y = acc[0][mma_ni].reg(((ci + 1) * ROWS_PER_THREAD + ri) * 2),                                            \
            .z = acc[0][mma_ni].reg(((ci + 0) * ROWS_PER_THREAD + ri) * 2 + 1),                                        \
            .w = acc[0][mma_ni].reg(((ci + 1) * ROWS_PER_THREAD + ri) * 2 + 1),                                        \
        };                                                                                                             \
        uint2 _dst = Acc_packer<float, Output_type, Scale>::run(this, _src);                                           \
        int64_t _offset = (int64_t) ri * step_m + (int64_t) (ci + mma_ni * COLS_PER_THREAD) * STEP_N;                  \
        fmha::stg(o_ptr_ + _offset, _dst);                                                                             \
    }

#pragma unroll
        for (int ri = 0; ri < ROWS_PER_THREAD; ri++)
        {
            if (row_ + ri * 8 >= actual_seqlen_)
            {
                break;
            }
#pragma unroll
            for (int mma_ni = 0; mma_ni < VALID_MMAS_N - 1; ++mma_ni)
            {
#pragma unroll
                for (int ci = 0; ci < COLS_PER_THREAD; ci += 2)
                {
                    STORE_COLUMNS()
                }
            }
            // The last mma_n may not store full elements back to GMEM.
            int mma_ni = VALID_MMAS_N - 1;
#pragma unroll
            for (int ci = 0; ci < VALID_COLS_PER_THREAD_FOR_LAST_MMA; ci += 2)
            {
                STORE_COLUMNS()
            }
        }
    }

    // Move to the next location.
    inline __device__ void move()
    {
        row_ += ROWS;
        o_ptr_ += (int64_t) ROWS * params_o_stride_in_bytes_;
    }

    // The stride between rows for the QKV matrice.
    int64_t params_o_stride_in_bytes_;
    // Scaling factor; this usually means QKV descale factor in actuality
    uint32_t params_scale_bmm2_;
    // The pointer.
    char* o_ptr_;
    // The row loaded by this thread.
    int row_;
    // The length of the sequence loaded by that CTA.
    int actual_seqlen_;
};

} // namespace v2

} // namespace fmha
