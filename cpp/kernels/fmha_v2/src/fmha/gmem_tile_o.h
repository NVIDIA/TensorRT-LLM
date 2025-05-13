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
#include <fmha/utils.h>

namespace fmha
{
namespace v1
{

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// H M M A
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, int CTAS_PER_HEAD>
struct Gmem_tile_o
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, int CTAS_PER_HEAD>
struct Hmma_gmem_tile_o
{

    // The mma tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    // The size of each element.
    enum
    {
        BYTES_PER_ELEMENT = 2
    };

    // The size of a row in bytes.
    enum
    {
        BYTES_PER_ROW = Cta_tile::N * BYTES_PER_ELEMENT
    };

    // The size of each STG.
    enum
    {
        BYTES_PER_STG = 16
    };

    // The number of threads to store a "row" of the matrix.
    enum
    {
        THREADS_PER_ROW = BYTES_PER_ROW / BYTES_PER_STG
    };

    // The number of "rows" stored per STG.
    enum
    {
        ROWS_PER_STG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW
    };

    // The number of "rows" stored per iteration of the loop.
    enum
    {
        ROWS = Cta_tile::M
    };

    // We want at least one output per thread (if possible).
    enum
    {
        ROWS_PER_LOOP_ = ROWS <= 64 ? ROWS : (int) Min<ROWS, ROWS_PER_STG>::VALUE
    };

    // We also want to have "complete" MMAs.
    enum
    {
        ROWS_PER_LOOP = Max<ROWS_PER_LOOP_, Mma_tile::M_PER_MMA_PER_CTA>::VALUE
    };

    // The number of outer loop for the stores.
    enum
    {
        LOOPS = fmha::Div_up<ROWS, ROWS_PER_LOOP>::VALUE
    };

    // DEBUG.
    static_assert(ROWS % ROWS_PER_LOOP == 0, "");
    // END OF DEBUG.

    // Make sure the math is correct.
    static_assert(ROWS_PER_LOOP >= (int) Mma_tile::M_PER_MMA_PER_CTA, "");

    // Do we have to guard against partial writes/reads.
    enum
    {
        HAS_INCOMPLETE_STG = Cta_tile::M % ROWS_PER_STG != 0
    };

    // The number of STGs needed to store a chunk of the Q matrix.
    enum
    {
        STGS_PER_LOOP = fmha::Div_up<ROWS_PER_LOOP, ROWS_PER_STG>::VALUE
    };

    // The number of STGs needed to store a chunk of the Q matrix in total.
    enum
    {
        STGS = STGS_PER_LOOP * LOOPS
    };

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Hmma_gmem_tile_o(Params const& params, Block_info const& binfo, int tidx, int cta_row_offset = 0)
        : params_o_stride_in_bytes_(params.o_stride_in_bytes)
        , o_ptr_(reinterpret_cast<char*>(params.o_ptr))
    {

        // Compute the position in the sequence (within the CTA for the moment).
        int row = tidx / THREADS_PER_ROW;
        // Compute the position of the thread in the row.
        int col = tidx % THREADS_PER_ROW;

        // Is that thread active on the last STG?
        if (HAS_INCOMPLETE_STG)
        {
            is_active_for_last_stg_ = row + (STGS - 1) * ROWS_PER_STG < Cta_tile::M;
        }

        // Account for the CTA-wide row offset (no loop mode).
        row += cta_row_offset;

        // The row offset in the batched GEMM.
        int64_t row_offset = (int64_t) row * params.o_stride_in_bytes;
        // Take the batch/head offset into account.
        row_offset += (int64_t) binfo.bidx * BYTES_PER_ROW;
        // Assemble the final pointer.
        o_ptr_ += row_offset + col * BYTES_PER_STG;
    }

    // Load data from global memory.
    inline __device__ void load(uint4 (&dst)[STGS_PER_LOOP], int mi)
    {
        if (blockIdx.x == 0)
        {
#pragma unroll
            for (int ii = 0; ii < STGS_PER_LOOP; ++ii)
            {
                dst[ii] = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        else
        {
#pragma unroll
            for (int ii = 0; ii < STGS_PER_LOOP; ++ii)
            {
                int jj = mi * STGS_PER_LOOP + ii;
                if (!HAS_INCOMPLETE_STG || (jj < STGS - 1 || is_active_for_last_stg_))
                {
                    fmha::ldg(dst[ii], o_ptr_ + jj * ROWS_PER_STG * params_o_stride_in_bytes_);
                }
            }
        }
    }

    // Store data to global memory.
    inline __device__ void store(uint4 const (&src)[STGS_PER_LOOP], int mi)
    {
#pragma unroll
        for (int ii = 0; ii < STGS_PER_LOOP; ++ii)
        {
            int jj = mi * STGS_PER_LOOP + ii;
            if (!HAS_INCOMPLETE_STG || (jj < STGS - 1 || is_active_for_last_stg_))
            {
                fmha::stg(o_ptr_ + jj * ROWS_PER_STG * params_o_stride_in_bytes_, src[ii]);
            }
        }
    }

    // Store data to global memory.
    inline __device__ void store(uint4 const (&src)[STGS_PER_LOOP], uint4 const (&old)[STGS_PER_LOOP], int mi)
    {
        uint4 tmp[STGS_PER_LOOP];
#pragma unroll
        for (int ii = 0; ii < STGS_PER_LOOP; ++ii)
        {
            tmp[ii].x = fmha::hadd2(src[ii].x, old[ii].x);
            tmp[ii].y = fmha::hadd2(src[ii].y, old[ii].y);
            tmp[ii].z = fmha::hadd2(src[ii].z, old[ii].z);
            tmp[ii].w = fmha::hadd2(src[ii].w, old[ii].w);
        }
        this->store(tmp, mi);
    }

    // Move the pointer to the next location.
    inline __device__ void move()
    {
        o_ptr_ += (int64_t) ROWS * params_o_stride_in_bytes_;
    }

    // The stride between rows for the QKV matrice.
    int64_t const params_o_stride_in_bytes_;
    // The pointer.
    char* o_ptr_;
    // Is the thread active for the last STG?
    int is_active_for_last_stg_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, int CTAS_PER_HEAD>
struct Gmem_tile_o<fmha::Volta_hmma_fp16_16x16x16_traits, Cta_tile, CTAS_PER_HEAD>
    : public Hmma_gmem_tile_o<fmha::Volta_hmma_fp16_16x16x16_traits, Cta_tile, CTAS_PER_HEAD>
{

    // The traits.
    using Traits = fmha::Volta_hmma_fp16_16x16x16_traits;
    // The base class.
    using Base = Hmma_gmem_tile_o<Traits, Cta_tile, CTAS_PER_HEAD>;

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_o(Params const& params, Block_info const& binfo, int tidx, int cta_row_offset = 0)
        : Base(params, binfo, tidx, cta_row_offset)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, int CTAS_PER_HEAD>
struct Gmem_tile_o<fmha::Turing_hmma_fp16_traits, Cta_tile, CTAS_PER_HEAD>
    : public Hmma_gmem_tile_o<fmha::Turing_hmma_fp16_traits, Cta_tile, CTAS_PER_HEAD>
{

    // The traits.
    using Traits = fmha::Turing_hmma_fp16_traits;
    // The base class.
    using Base = Hmma_gmem_tile_o<Traits, Cta_tile, CTAS_PER_HEAD>;

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_o(Params const& params, Block_info const& binfo, int tidx, int cta_row_offset = 0)
        : Base(params, binfo, tidx, cta_row_offset)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, int CTAS_PER_HEAD>
struct Gmem_tile_o<fmha::Ampere_hmma_fp16_traits, Cta_tile, CTAS_PER_HEAD>
    : public Hmma_gmem_tile_o<fmha::Ampere_hmma_fp16_traits, Cta_tile, CTAS_PER_HEAD>
{

    // The traits.
    using Traits = fmha::Ampere_hmma_fp16_traits;
    // The base class.
    using Base = Hmma_gmem_tile_o<Traits, Cta_tile, CTAS_PER_HEAD>;

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_o(Params const& params, Block_info const& binfo, int tidx, int cta_row_offset = 0)
        : Base(params, binfo, tidx, cta_row_offset)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// I M M A
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, int CTAS_PER_HEAD>
struct Imma_gmem_tile_o
{

    // The mma tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    // The size of each element.
    enum
    {
        BYTES_PER_ELEMENT = 1
    };

    // The size of a row in bytes.
    enum
    {
        BYTES_PER_ROW = Cta_tile::N * BYTES_PER_ELEMENT
    };

    // The size of each STG.
    enum
    {
        BYTES_PER_STG = 4
    };

    // The number of threads to store a "row" of the matrix.
    enum
    {
        THREADS_PER_ROW = BYTES_PER_ROW / BYTES_PER_STG
    };

    // The number of "rows" stored per STG.
    enum
    {
        ROWS_PER_STG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW
    };

    // The number of "rows" stored per iteration of the loop. The output of 1 MMA.
    enum
    {
        ROWS = Cta_tile::M
    };

    // We want at least one output per thread (if possible).
    enum
    {
        ROWS_PER_LOOP_ = ROWS <= 64 ? ROWS : (int) Min<ROWS, ROWS_PER_STG>::VALUE
    };

    // We also want to have "complete" MMAs.
    enum
    {
        ROWS_PER_LOOP = Max<ROWS_PER_LOOP_, Mma_tile::M_PER_MMA_PER_CTA>::VALUE
    };

    // The number of outer loop for the stores.
    enum
    {
        LOOPS = fmha::Div_up<ROWS, ROWS_PER_LOOP>::VALUE
    };

    // DEBUG.
    static_assert(ROWS % ROWS_PER_LOOP == 0, "");
    // END OF DEBUG.

    // Make sure the math is correct.
    static_assert(ROWS_PER_LOOP >= (int) Mma_tile::M_PER_MMA_PER_CTA, "");

    // Do we have to guard against partial writes/reads (last STG).
    enum
    {
        HAS_INCOMPLETE_STG = Cta_tile::M % ROWS_PER_STG != 0
    };

    // The number of STGs needed to store a chunk of the Q matrix.
    enum
    {
        STGS_PER_LOOP = fmha::Div_up<ROWS_PER_LOOP, ROWS_PER_STG>::VALUE
    };

    // The number of STGs needed to store a chunk of the Q matrix in total.
    enum
    {
        STGS = STGS_PER_LOOP * LOOPS
    };

    // Are all threads active?
    enum
    {
        ALL_THREADS_ACTIVE = ROWS_PER_STG <= ROWS_PER_LOOP
    };

    // The number of active threads.
    enum
    {
        ACTIVE_THREADS_ = Cta_tile::THREADS_PER_CTA * ROWS_PER_LOOP / ROWS_PER_STG
    };

    // The number of active threads.
    enum
    {
        ACTIVE_THREADS = ALL_THREADS_ACTIVE ? Cta_tile::THREADS_PER_CTA : ACTIVE_THREADS_
    };

    // Ctor.
    template <typename Params>
    inline __device__ Imma_gmem_tile_o(Params const& params, int bidx, int tidx, int cta_row_offset)
        : params_o_stride_in_bytes_(params.o_stride_in_bytes)
        , params_scale_bmm2_(params.scale_bmm2)
        , params_enable_i2f_trick_(params.enable_i2f_trick)
        , o_ptr_(reinterpret_cast<char*>(params.o_ptr))
#if USE_DEMO_BERT_PARAMS
        , o_scratch_ptr_(nullptr)
    {
#else
        , o_scratch_ptr_(reinterpret_cast<int32_t*>(params.o_scratch_ptr))
    {
#endif

        // Compute the position in the sequence (within the CTA for the moment).
        int row = tidx / THREADS_PER_ROW;
        // Compute the position of the thread in the row.
        int col = tidx % THREADS_PER_ROW;

        // Is it an active thread?
        is_active_ = ALL_THREADS_ACTIVE || row < ROWS_PER_LOOP;

        // Is that thread active on the last STG?
        if (HAS_INCOMPLETE_STG)
        {
            is_active_for_last_stg_ = row + (STGS - 1) * ROWS_PER_STG < Cta_tile::M;
        }

        // Update the row.
        row += cta_row_offset;

        // The row offset in the batched GEMM.
        int64_t row_offset = (int64_t) row * params.o_stride_in_bytes;
        // Take the batch/head offset into account.
        row_offset += (int64_t) bidx * BYTES_PER_ROW;
        // Assemble the final pointers.
        o_ptr_ += row_offset + col * BYTES_PER_STG;

        // For the scratch space, the pointer has int32 type so it accounts for the *4 factor.
        o_scratch_ptr_ += blockIdx.y * STGS_PER_LOOP * ACTIVE_THREADS + tidx;
    }

    // Load data from global memory.
    inline __device__ void load(uint4 (&dst)[STGS_PER_LOOP], int mi)
    {
        if (blockIdx.x == 0)
        {
#pragma unroll
            for (int ii = 0; ii < STGS_PER_LOOP; ++ii)
            {
                dst[ii] = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        else if (ALL_THREADS_ACTIVE || is_active_)
        {
#pragma unroll
            for (int ii = 0; ii < STGS_PER_LOOP; ++ii)
            {
                fmha::ldg(dst[ii], o_scratch_ptr_ + ii * ACTIVE_THREADS);
            }
        }
    }

    // Store data to global memory.
    inline __device__ void store(uint4 const (&src)[STGS_PER_LOOP], int mi)
    {

        // The scale.
        float const& scale = reinterpret_cast<float const&>(params_scale_bmm2_);
// Iterate over the different STGs.
#pragma unroll
        for (int ii = 0; ii < STGS_PER_LOOP; ++ii)
        {
            // The accumulators are in int32_t.
            int4 const& val = reinterpret_cast<int4 const&>(src[ii]);

            // Extract the floats and scale.
            float f0, f1, f2, f3;
#if defined(USE_I2F_EMULATION_TRICK)
            if (params_enable_i2f_trick_)
            {

                f0 = reinterpret_cast<float const&>(val.x) - FP32_I2F_MAGIC_NUMBER;
                f1 = reinterpret_cast<float const&>(val.y) - FP32_I2F_MAGIC_NUMBER;
                f2 = reinterpret_cast<float const&>(val.z) - FP32_I2F_MAGIC_NUMBER;
                f3 = reinterpret_cast<float const&>(val.w) - FP32_I2F_MAGIC_NUMBER;
            }
            else
#endif // defined(USE_I2F_EMULATION_TRICK)
            {
                f0 = static_cast<float>(val.x);
                f1 = static_cast<float>(val.y);
                f2 = static_cast<float>(val.z);
                f3 = static_cast<float>(val.w);
            }

            // Apply the scaling.
            f0 *= scale;
            f1 *= scale;
            f2 *= scale;
            f3 *= scale;

            // Convert the 4 floats to char4.
            uint32_t dst = float4_to_char4<true>(f0, f1, f2, f3);

            // Store the result.
            int jj = mi * STGS_PER_LOOP + ii;
            if (!HAS_INCOMPLETE_STG || (jj < STGS - 1 || is_active_for_last_stg_))
            {
                fmha::stg(o_ptr_ + jj * ROWS_PER_STG * params_o_stride_in_bytes_, dst);
            }
        }
    }

    // Store data to global memory.
    inline __device__ void store(uint4 const (&src)[STGS_PER_LOOP], uint4 const (&old)[STGS_PER_LOOP], int mi)
    {
        // Do the reduction.
        uint4 tmp[STGS_PER_LOOP];
#pragma unroll
        for (int ii = 0; ii < STGS_PER_LOOP; ++ii)
        {
            int4 const& src_ii = reinterpret_cast<int4 const&>(src[ii]);
            int4 const& old_ii = reinterpret_cast<int4 const&>(old[ii]);

            int32_t x = src_ii.x + old_ii.x;
            int32_t y = src_ii.y + old_ii.y;
            int32_t z = src_ii.z + old_ii.z;
            int32_t w = src_ii.w + old_ii.w;

            tmp[ii].x = reinterpret_cast<uint32_t const&>(x);
            tmp[ii].y = reinterpret_cast<uint32_t const&>(y);
            tmp[ii].z = reinterpret_cast<uint32_t const&>(z);
            tmp[ii].w = reinterpret_cast<uint32_t const&>(w);
        }

        // The last CTA stores INT8 values to the final location.
        if (blockIdx.x == CTAS_PER_HEAD - 1)
        {
            this->store(tmp, mi);

            // Other CTAs store INT32 values to the scratch space.
        }
        else if (ALL_THREADS_ACTIVE || is_active_)
        {
#pragma unroll
            for (int ii = 0; ii < STGS_PER_LOOP; ++ii)
            {
                fmha::stg(o_scratch_ptr_ + ii * ACTIVE_THREADS, tmp[ii]);
            }
        }
    }

    // Move the pointer.
    inline __device__ void move()
    {
        o_ptr_ += (int64_t) ROWS * params_o_stride_in_bytes_;
    }

    // The stride between rows for the QKV matrice.
    int64_t const params_o_stride_in_bytes_;
    // The scaling factor to convert to int8.
    uint32_t const params_scale_bmm2_;
    // Do we enable the i2f trick?
    bool const params_enable_i2f_trick_;
    // The pointer.
    char* o_ptr_;
    // The scratch pointer for 32-bit reductions.
    int32_t* o_scratch_ptr_;

    // Is it an active thread? When ROWS_PER_STG > ROWS_PER_LOOP, some threads do not store.
    int is_active_, is_active_for_last_stg_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, int CTAS_PER_HEAD>
struct Gmem_tile_o<fmha::Turing_imma_int8_int32_traits, Cta_tile, CTAS_PER_HEAD>
    : public Imma_gmem_tile_o<fmha::Turing_imma_int8_int32_traits, Cta_tile, CTAS_PER_HEAD>
{
    // The traits class.
    using Traits = fmha::Turing_imma_int8_int32_traits;
    // The base class.
    using Base = Imma_gmem_tile_o<Traits, Cta_tile, CTAS_PER_HEAD>;

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_o(Params const& params, Block_info const& block_info, int tidx, int cta_row_offset = 0)
        : Base(params, block_info.bidx, tidx, cta_row_offset)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, int CTAS_PER_HEAD>
struct Gmem_tile_o<fmha::Ampere_imma_int8_int32_traits, Cta_tile, CTAS_PER_HEAD>
    : public Imma_gmem_tile_o<fmha::Ampere_imma_int8_int32_traits, Cta_tile, CTAS_PER_HEAD>
{
    // The traits class.
    using Traits = fmha::Ampere_imma_int8_int32_traits;
    // The base class.
    using Base = Imma_gmem_tile_o<Traits, Cta_tile, CTAS_PER_HEAD>;

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_o(Params const& params, Block_info const& block_info, int tidx, int cta_row_offset = 0)
        : Base(params, block_info.bidx, tidx, cta_row_offset)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace v1
} // namespace fmha
