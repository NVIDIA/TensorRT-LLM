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

#include <fmha/numeric_types.h>
#include <fmha/traits.h>
#include <fmha/utils.h>

namespace fmha
{
namespace v2
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

template <typename Traits, typename Cta_tile, int CTAS_PER_HEAD, int BYTES_PER_STG_ = 16, int BYTES_PER_ELEMENT_ = 2>
struct Hmma_gmem_tile_o
{

    // The mma tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    // The size of each element.
    enum
    {
        BYTES_PER_ELEMENT = BYTES_PER_ELEMENT_
    };

    // The size of a row in bytes.
    enum
    {
        BYTES_PER_ROW = Cta_tile::N * BYTES_PER_ELEMENT
    };

    // The valid size of a row in bytes.
    // Note: cross-attention kernels rely on head dim from runtime instead of from compile-time.
    // This approach deviates from self-attention kernels. To explore a unified approach.
    // enum { VALID_BYTES_PER_ROW = Cta_tile::VALID_N * BYTES_PER_ELEMENT };

    // The size of each STG.
    enum
    {
        BYTES_PER_STG = BYTES_PER_STG_
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
    inline __device__ Hmma_gmem_tile_o(
        Params const& params, Block_info const& binfo, int tidx, int cta_row_offset, int cta_col_offset_in_bytes = 0)
        : params_o_stride_in_bytes_(params.o_stride_in_bytes)
        , actual_seqlen_(binfo.actual_q_seqlen)
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

        // Store the row/col to update the predicates in load.
        row_ = cta_row_offset + row;
        col_in_bytes_ = cta_col_offset_in_bytes + col * BYTES_PER_STG;
        init_row_ = row_;

        // The row offset in the batched GEMM.
        int64_t row_offset = (int64_t) row_ * params.o_stride_in_bytes;
        // The amount of bytes per row without padding.
        int const valid_bytes_per_row = params.dv * BYTES_PER_ELEMENT;
        // Take the batch/head offset into account. TODO: Fix me!
        //
        // row_offset += binfo.bidx * VALID_BYTES_PER_ROW;
        //
        row_offset += binfo.bidx * valid_bytes_per_row;

        // Assemble the final pointer.
        o_ptr_ += row_offset + col_in_bytes_;
        init_o_ptr_ = o_ptr_;

        // Do not store if the thread is in the padded area
        active_ = col_in_bytes_ < valid_bytes_per_row;
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
                if (row_ + jj * ROWS_PER_STG >= actual_seqlen_)
                {
                    break;
                }
                if (active_ && (!HAS_INCOMPLETE_STG || (jj < STGS - 1 || is_active_for_last_stg_)))
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
            if (row_ + jj * ROWS_PER_STG >= actual_seqlen_)
            {
                break;
            }
            if (active_ && (!HAS_INCOMPLETE_STG || (jj < STGS - 1 || is_active_for_last_stg_)))
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
    inline __device__ void move(int const steps = 1)
    {
        row_ += ROWS * steps;
        o_ptr_ += (int64_t) ROWS * params_o_stride_in_bytes_ * steps;
    }

    inline __device__ void move_to(int const step)
    {
        row_ = init_row_ + ROWS * step;
        o_ptr_ = init_o_ptr_ + (int64_t) ROWS * params_o_stride_in_bytes_ * step;
    }

    // The stride between rows for the QKV matrice.
    int64_t params_o_stride_in_bytes_;
    // The pointer.
    char* o_ptr_;
    char* init_o_ptr_;
    // Is the thread active for the last STG?
    int is_active_for_last_stg_;

    // The row loaded by this thread.
    int row_, col_in_bytes_;
    int init_row_;
    // The length of the sequence loaded by that CTA.
    int actual_seqlen_;
    // Is that thread active when it comes to loading data?
    int active_;
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
    inline __device__ Gmem_tile_o(Params const& params, Block_info const& block_info, int tidx, int cta_row_offset = 0)
        : Base(params, block_info, tidx, cta_row_offset)
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
    inline __device__ Gmem_tile_o(Params const& params, Block_info const& block_info, int tidx, int cta_row_offset = 0,
        int cta_col_offset_in_bytes = 0)
        : Base(params, block_info, tidx, cta_row_offset, cta_col_offset_in_bytes)
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
    inline __device__ Gmem_tile_o(Params const& params, Block_info const& block_info, int tidx, int cta_row_offset = 0,
        int cta_col_offset_in_bytes = 0)
        : Base(params, block_info, tidx, cta_row_offset, cta_col_offset_in_bytes)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, int CTAS_PER_HEAD>
struct Gmem_tile_o<fmha::Ampere_hmma_bf16_bf16_traits, Cta_tile, CTAS_PER_HEAD>
    : public Hmma_gmem_tile_o<fmha::Ampere_hmma_bf16_bf16_traits, Cta_tile, CTAS_PER_HEAD>
{

    // The traits.
    using Traits = fmha::Ampere_hmma_bf16_bf16_traits;
    // The base class.
    using Base = Hmma_gmem_tile_o<Traits, Cta_tile, CTAS_PER_HEAD>;

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_o(Params const& params, Block_info const& block_info, int tidx, int cta_row_offset = 0,
        int cta_col_offset_in_bytes = 0)
        : Base(params, block_info, tidx, cta_row_offset, cta_col_offset_in_bytes)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, int CTAS_PER_HEAD>
struct Gmem_tile_o<fmha::Ampere_hmma_fp32_traits, Cta_tile, CTAS_PER_HEAD>
    : public Hmma_gmem_tile_o<fmha::Ampere_hmma_fp32_traits, Cta_tile, CTAS_PER_HEAD, 8>
{

    // The traits.
    using Traits = fmha::Ampere_hmma_fp32_traits;
    // The base class.
    using Base = Hmma_gmem_tile_o<Traits, Cta_tile, CTAS_PER_HEAD, 8>;

    // The epilogue data type
    using Epilogue_type = typename Traits::Epilogue_type;

    // DEBUG.
    static_assert((Base::THREADS_PER_ROW == 16 || Base::THREADS_PER_ROW == 32 || Base::THREADS_PER_ROW == 64
                      || Base::THREADS_PER_ROW == 128)
            && Base::BYTES_PER_STG == 8,
        "");

    // END OF DEBUG.

    enum
    {
        STGS_PER_LOOP = Base::STGS_PER_LOOP
    };

    enum
    {
        ROWS_PER_STG = Base::ROWS_PER_STG
    };

    enum
    {
        STGS = Base::STGS
    };

    enum
    {
        HAS_INCOMPLETE_STG = Base::HAS_INCOMPLETE_STG
    };

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_o(Params const& params, Block_info const& block_info, int tidx, int cta_row_offset = 0,
        int cta_col_offset_in_bytes = 0)
        : Base(params, block_info, tidx, cta_row_offset, cta_col_offset_in_bytes)
    {
    }

    // Load data from global memory.
    inline __device__ void load(uint4 const (&dst)[STGS_PER_LOOP], int mi)
    {
        static_assert(CTAS_PER_HEAD == 1, "Not implemented");
    }

    // Store data to global memory.
    inline __device__ void store(uint4 const (&src)[STGS_PER_LOOP], int mi)
    {

#pragma unroll
        for (int ii = 0; ii < STGS_PER_LOOP; ++ii)
        {
            int jj = mi * STGS_PER_LOOP + ii;
            if (this->row_ + jj * ROWS_PER_STG >= this->actual_seqlen_)
            {
                break;
            }

            float x = reinterpret_cast<float const&>(src[ii].x);
            float y = reinterpret_cast<float const&>(src[ii].y);
            float z = reinterpret_cast<float const&>(src[ii].z);
            float w = reinterpret_cast<float const&>(src[ii].w);

            uint2 out = float4_to_16bit_x4<Epilogue_type>(x, y, z, w);
            if (this->active_ && (!HAS_INCOMPLETE_STG || (jj < STGS - 1 || this->is_active_for_last_stg_)))
            {
                fmha::stg(this->o_ptr_ + jj * ROWS_PER_STG * this->params_o_stride_in_bytes_, out);
            }
        }
    }

    // Store data to global memory.
    inline __device__ void store(uint4 const (&src)[STGS_PER_LOOP], uint4 const (&old)[STGS_PER_LOOP], int mi)
    {
        static_assert(CTAS_PER_HEAD == 1, "Not implemented");
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, int CTAS_PER_HEAD>
struct Gmem_tile_o<fmha::Ampere_hmma_bf16_traits, Cta_tile, CTAS_PER_HEAD>
    : public Hmma_gmem_tile_o<fmha::Ampere_hmma_bf16_traits, Cta_tile, CTAS_PER_HEAD, 8>
{

    // The traits.
    using Traits = fmha::Ampere_hmma_bf16_traits;
    // The base class.
    using Base = Hmma_gmem_tile_o<Traits, Cta_tile, CTAS_PER_HEAD, 8>;

    // The epilogue data type
    using Epilogue_type = typename Traits::Epilogue_type;

    // DEBUG.
    static_assert((Base::THREADS_PER_ROW == 16 || Base::THREADS_PER_ROW == 32 || Base::THREADS_PER_ROW == 64
                      || Base::THREADS_PER_ROW == 128)
            && Base::BYTES_PER_STG == 8,
        "");

    // END OF DEBUG.

    enum
    {
        STGS_PER_LOOP = Base::STGS_PER_LOOP
    };

    enum
    {
        ROWS_PER_STG = Base::ROWS_PER_STG
    };

    enum
    {
        STGS = Base::STGS
    };

    enum
    {
        HAS_INCOMPLETE_STG = Base::HAS_INCOMPLETE_STG
    };

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_o(Params const& params, Block_info const& block_info, int tidx, int cta_row_offset = 0,
        int cta_col_offset_in_bytes = 0)
        : Base(params, block_info, tidx, cta_row_offset, cta_col_offset_in_bytes)
    {
    }

    // Load data from global memory.
    inline __device__ void load(uint4 const (&dst)[STGS_PER_LOOP], int mi)
    {
        static_assert(CTAS_PER_HEAD == 1, "Not implemented");
    }

    // Store data to global memory.
    inline __device__ void store(uint4 const (&src)[STGS_PER_LOOP], int mi)
    {

#pragma unroll
        for (int ii = 0; ii < STGS_PER_LOOP; ++ii)
        {
            int jj = mi * STGS_PER_LOOP + ii;
            if (this->row_ + jj * ROWS_PER_STG >= this->actual_seqlen_)
            {
                break;
            }

            float x = reinterpret_cast<float const&>(src[ii].x);
            float y = reinterpret_cast<float const&>(src[ii].y);
            float z = reinterpret_cast<float const&>(src[ii].z);
            float w = reinterpret_cast<float const&>(src[ii].w);

            uint2 out = float4_to_16bit_x4<Epilogue_type>(x, y, z, w);
            if (this->active_ && (!HAS_INCOMPLETE_STG || (jj < STGS - 1 || this->is_active_for_last_stg_)))
            {
                fmha::stg(this->o_ptr_ + jj * ROWS_PER_STG * this->params_o_stride_in_bytes_, out);
            }
        }
    }

    // Store data to global memory.
    inline __device__ void store(uint4 const (&src)[STGS_PER_LOOP], uint4 const (&old)[STGS_PER_LOOP], int mi)
    {
        static_assert(CTAS_PER_HEAD == 1, "Not implemented");
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// I M M A
//
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t quantize(int4 const val, float const scale, bool const params_enable_i2f_trick)
{
    // Extract the floats and scale.
    float f0, f1, f2, f3;
#if defined(USE_I2F_EMULATION_TRICK)
    if (params_enable_i2f_trick)
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

    return dst;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Helpers to pack 4 registers representing a Src_type into a destination register with 4 8bit values
// representing Dst_type.
// Scale factor is assumed to be always FP32 for 32-bit accumulators.
template <typename Src_type, typename Dst_type, bool SCALE = true>
struct Acc_packer
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Signed INT32 => INT8.
template <>
struct Acc_packer<int32_t, int8_t, true>
{
    template <typename This>
    static inline __device__ uint32_t run(This const* this_, uint4 const& src_regs)
    {
        float const& scale = reinterpret_cast<float const&>(this_->params_scale_bmm2_);
        // The accumulators are in int32_t.
        int4 const& val = reinterpret_cast<int4 const&>(src_regs);

        // Quantize...
        uint32_t dst = quantize(val, scale, this_->params_enable_i2f_trick_);
        return dst;
    }
};

template <>
struct Acc_packer<int32_t, int8_t, false>
{
    template <typename This>
    static inline __device__ uint32_t run(This const* this_, uint4 const& src_regs)
    {
        // The accumulators are in int32_t.
        int4 const& val = reinterpret_cast<int4 const&>(src_regs);

        // Quantize...
        uint32_t dst = quantize(val, 1.0f, this_->params_enable_i2f_trick_);
        return dst;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// FP32 => FP8.
template <>
struct Acc_packer<float, fmha::e4m3_t, true>
{

    template <typename This>
    static inline __device__ uint32_t run(This const* this_, uint4 const& src_regs)
    {

        float const scale = reinterpret_cast<float const&>(this_->params_scale_bmm2_);

        float4 const& val = reinterpret_cast<float4 const&>(src_regs);

        uint32_t dst = fmha::float4_to_e4m3x4(val.x * scale, val.y * scale, val.z * scale, val.w * scale);
        return dst;
    }

    template <typename This>
    static inline __device__ uint16_t run(This const* this_, uint2 const& src_regs)
    {

        float const& scale = reinterpret_cast<float const&>(this_->params_scale_bmm2_);

        float2 const& val = reinterpret_cast<float2 const&>(src_regs);

        uint16_t dst = fmha::float2_to_e4m3x2(val.x * scale, val.y * scale);
        return dst;
    }
};

// FP32 => FP8.
template <>
struct Acc_packer<float, fmha::e4m3_t, false>
{

    template <typename This>
    static inline __device__ uint32_t run(This const* this_, uint4 const& src_regs)
    {

        float4 const& val = reinterpret_cast<float4 const&>(src_regs);

        uint32_t dst = fmha::float4_to_e4m3x4(val.x, val.y, val.z, val.w);
        return dst;
    }

    template <typename This>
    static inline __device__ uint16_t run(This const* this_, uint2 const& src_regs)
    {

        float2 const& val = reinterpret_cast<float2 const&>(src_regs);

        uint16_t dst = fmha::float2_to_e4m3x2(val.x, val.y);
        return dst;
    }
};

// FP16 => FP8.
template <>
struct Acc_packer<uint16_t, fmha::e4m3_t, true>
{

    template <typename This>
    static inline __device__ uint2 run(This const* this_, uint4 const& src_regs)
    {

        uint2 dst;
        dst.x = fmha::half4_to_e4m3x4(
            fmha::hmul2(src_regs.x, this_->params_scale_bmm2_), fmha::hmul2(src_regs.y, this_->params_scale_bmm2_));
        dst.y = fmha::half4_to_e4m3x4(
            fmha::hmul2(src_regs.z, this_->params_scale_bmm2_), fmha::hmul2(src_regs.w, this_->params_scale_bmm2_));

        return dst;
    }
};

// FP16 => FP8.
template <>
struct Acc_packer<uint16_t, fmha::e4m3_t, false>
{

    template <typename This>
    static inline __device__ uint2 run(This const* this_, uint4 const& src_regs)
    {

        uint2 dst;
        dst.x = fmha::half4_to_e4m3x4(src_regs.x, src_regs.y);
        dst.y = fmha::half4_to_e4m3x4(src_regs.z, src_regs.w);

        return dst;
    }
};

template <>
struct Acc_packer<float, fmha::e5m2_t, true>
{

    template <typename This>
    static inline __device__ uint32_t run(This const* this_, uint4 const& src_regs)
    {

        float const& scale = reinterpret_cast<float const&>(this_->params_scale_bmm2_);

        float4 const& val = reinterpret_cast<float4 const&>(src_regs);

        uint32_t dst = fmha::float4_to_e5m2x4(val.x * scale, val.y * scale, val.z * scale, val.w * scale);
        return dst;
    }
};

template <>
struct Acc_packer<float, fmha::e5m2_t, false>
{

    template <typename This>
    static inline __device__ uint32_t run(This const* this_, uint4 const& src_regs)
    {

        float4 const& val = reinterpret_cast<float4 const&>(src_regs);

        uint32_t dst = fmha::float4_to_e5m2x4(val.x, val.y, val.z, val.w);
        return dst;
    }
};

template <>
struct Acc_packer<float, uint16_t, false>
{

    template <typename This>
    static inline __device__ uint2 run(This const* this_, uint4 const& src_regs)
    {

        float4 const& val = reinterpret_cast<float4 const&>(src_regs);

        uint2 dst = fmha::float4_to_half4(val.x, val.y, val.z, val.w);
        return dst;
    }
};

template <>
struct Acc_packer<float, uint16_t, true>
{

    template <typename This>
    static inline __device__ uint2 run(This const* this_, uint4 const& src_regs)
    {

        float const& scale = reinterpret_cast<float const&>(this_->params_scale_bmm2_);

        float4 const& val = reinterpret_cast<float4 const&>(src_regs);

        uint2 dst = fmha::float4_to_half4(val.x * scale, val.y * scale, val.z * scale, val.w * scale);
        return dst;
    }
};

template <>
struct Acc_packer<float, nv_bfloat16, false>
{

    template <typename This>
    static inline __device__ uint2 run(This const* this_, uint4 const& src_regs)
    {

        float4 const& val = reinterpret_cast<float4 const&>(src_regs);

        uint2 dst = fmha::float4_to_16bit_x4<bf16_t>(val.x, val.y, val.z, val.w);
        return dst;
    }
};

template <>
struct Acc_packer<float, nv_bfloat16, true>
{

    template <typename This>
    static inline __device__ uint2 run(This const* this_, uint4 const& src_regs)
    {

        float const& scale = reinterpret_cast<float const&>(this_->params_scale_bmm2_);

        float4 const& val = reinterpret_cast<float4 const&>(src_regs);

        uint2 dst = fmha::float4_to_16bit_x4<bf16_t>(val.x * scale, val.y * scale, val.z * scale, val.w * scale);
        return dst;
    }
};

// support both 32 bit accumulationi and 16 bit accumulation (imma and qmma)
template <typename Traits, typename Cta_tile, int CTAS_PER_HEAD>
struct Gmem_tile_o_8bit
{

    // static_assert(sizeof(typename Traits::Accumulator_type) == 4);
    static_assert(sizeof(typename Traits::C_type) == 1);

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

    // The valid size of a row in bytes.
    enum
    {
        VALID_BYTES_PER_ROW = Cta_tile::VALID_N * BYTES_PER_ELEMENT
    };

    // The size of each STG (16B --> 8bit elements).
    enum
    {
        BYTES_PER_STG = fmha::Div_up<16, sizeof(typename Traits::Accumulator_type)>::VALUE
    };

    // The STG packed data type
    using Stg_packed_type = typename Uint_from_size_in_bytes<BYTES_PER_STG>::Type;

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

#if 0
    // The number of "rows" stored per iteration of the loop. The output of 1 MMA.
    enum { ROWS = Cta_tile::M };
    // The number of "rows" stored per iteration of the loop. The output of 1 MMA.
    enum { ROWS_PER_LOOP = Mma_tile::M_PER_MMA_PER_CTA };
    // The number of outer loop for the stores.
    enum { LOOPS = ROWS / ROWS_PER_LOOP };

    // Make sure the math is correct.
    static_assert(LOOPS == (int)Mma_tile::MMAS_M, "");

    // The number of "rows" stored per STG -- for it to be the number of rows per MMA instruction.
    enum { ROWS_PER_STG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW };
    // The number of STGs needed to store a chunk of the Q matrix.
    enum { STGS_PER_LOOP = fmha::Div_up<ROWS_PER_LOOP, ROWS_PER_STG>::VALUE };
#endif

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
    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_o_8bit(Params const& params, Block_info const& block_info, int tidx,
        int cta_row_offset = 0, int cta_col_offset_in_bytes = 0)
        : params_o_stride_in_bytes_(params.o_stride_in_bytes)
        , actual_seqlen_(block_info.actual_q_seqlen)
        , params_scale_bmm2_(params.scale_bmm2_d ? *params.scale_bmm2_d : params.scale_bmm2)
#ifdef GENERATE_CUBIN
        , params_enable_i2f_trick_(false)
#else
        , params_enable_i2f_trick_(params.enable_i2f_trick)
#endif
        , o_ptr_(reinterpret_cast<char*>(params.o_ptr))
#if USE_DEMO_BERT_PARAMS
        , o_scratch_ptr_(nullptr)
    {
#else
        , o_scratch_ptr_(reinterpret_cast<uint4*>(params.o_scratch_ptr))
    {
#endif

        // Compute the position in the sequence (within the CTA for the moment).
        int row = tidx / THREADS_PER_ROW;
        // Compute the position of the thread in the row.
        int col = tidx % THREADS_PER_ROW;

        // Is it an active thread for the very last STG?
        if (HAS_INCOMPLETE_STG)
        {
            is_active_for_last_stg_ = row + (STGS - 1) * ROWS_PER_STG < Cta_tile::M;
        }

        // Store the row to check against the length before loads.
        row_ = cta_row_offset + row;
        col_in_bytes_ = cta_col_offset_in_bytes + col * BYTES_PER_STG;

        // The row offset in the batched GEMM.
        int64_t row_offset = (int64_t) row_ * params.o_stride_in_bytes;
        // The amount of bytes per row without padding (runtime).
        int const valid_bytes_per_row = params.dv * BYTES_PER_ELEMENT;
        // Take the batch/head offset into account.
        row_offset += block_info.bidx * valid_bytes_per_row;
        // Assemble the final pointer.
        o_ptr_ += row_offset + col_in_bytes_;

        // Is it an active thread?
        is_active_ = ALL_THREADS_ACTIVE || (row < ROWS_PER_LOOP && col_in_bytes_ < VALID_BYTES_PER_ROW);

        // Do not store if the thread is in the padded area
        is_active_ = is_active_ && col < valid_bytes_per_row / BYTES_PER_STG;

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

// Iterate over the different STGs.
#pragma unroll
        for (int ii = 0; ii < STGS_PER_LOOP; ++ii)
        {

            // Break early if we exceed s_i...
            int jj = mi * STGS_PER_LOOP + ii;
            if (row_ + jj * ROWS_PER_STG >= actual_seqlen_)
            {
                return;
            }
            using Src_type = typename Traits::Accumulator_type;
            using Dst_type = typename Traits::C_type;
            // Packs the 32bit/16bit values to 8bit.
            // Depending on the type, applies extra scaling with parameter scale_bmm2.
            Stg_packed_type dst = Acc_packer<Src_type, Dst_type>::run(this, src[ii]);
            float const* row_ptr = reinterpret_cast<float const*>(&src[ii]);

            // Store the result.
            if (is_active_ && (!HAS_INCOMPLETE_STG || (jj < STGS - 1 || is_active_for_last_stg_)))
            {
                fmha::stg(o_ptr_ + jj * ROWS_PER_STG * params_o_stride_in_bytes_, dst);
            }
        }
    }

    // Store data to global memory.
    // TODO: 16bit (half)
    inline __device__ void store(uint4 const (&src)[STGS_PER_LOOP], uint4 const (&old)[STGS_PER_LOOP], int mi)
    {
        // Do the reduction.
        uint4 tmp[STGS_PER_LOOP];
#if defined(USE_I2F_EMULATION_TRICK)
        if (params_enable_i2f_trick_)
        {
#pragma unroll
            for (int ii = 0; ii < STGS_PER_LOOP; ++ii)
            {
                float4 const& src_ii = reinterpret_cast<float4 const&>(src[ii]);
                float4 const& old_ii = reinterpret_cast<float4 const&>(old[ii]);

                float x = src_ii.x + old_ii.x;
                float y = src_ii.y + old_ii.y;
                float z = src_ii.z + old_ii.z;
                float w = src_ii.w + old_ii.w;

                tmp[ii].x = reinterpret_cast<uint32_t const&>(x);
                tmp[ii].y = reinterpret_cast<uint32_t const&>(y);
                tmp[ii].z = reinterpret_cast<uint32_t const&>(z);
                tmp[ii].w = reinterpret_cast<uint32_t const&>(w);
            }
        }
        else
#endif
        {
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
        row_ += ROWS;
        o_ptr_ += (int64_t) ROWS * params_o_stride_in_bytes_;
    }

    // The stride between rows for the QKV matrice.
    int64_t params_o_stride_in_bytes_;
    // The scaling factor to convert to int8.
    uint32_t const params_scale_bmm2_;
    // Do we enable the i2f trick?
    bool const params_enable_i2f_trick_;
    // The pointer.
    char* o_ptr_;
    // The pointer to the scratch space to do the reduction (for CTAS_PER_HEAD > 1).
    uint4* o_scratch_ptr_;
    // The row, col stored by this thread (i.e. the position in that sequence).
    int row_, col_in_bytes_;
    // The size of the sequence length computed by that CTA.
    int actual_seqlen_;

    // Is it an active thread?
    int is_active_, is_active_for_last_stg_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, int CTAS_PER_HEAD>
struct Gmem_tile_o<fmha::Volta_imma_int8_int32_traits, Cta_tile, CTAS_PER_HEAD>
    : public Gmem_tile_o_8bit<fmha::Volta_imma_int8_int32_traits, Cta_tile, CTAS_PER_HEAD>
{

    // The traits class.
    using Traits = fmha::Volta_imma_int8_int32_traits;
    // The base class.
    using Base = Gmem_tile_o_8bit<Traits, Cta_tile, CTAS_PER_HEAD>;

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_o(Params const& params, Block_info const& block_info, int tidx, int cta_row_offset = 0,
        int cta_col_offset_in_bytes = 0)
        : Base(params, block_info, tidx, cta_row_offset, cta_col_offset_in_bytes)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, int CTAS_PER_HEAD>
struct Gmem_tile_o<fmha::Turing_imma_int8_int32_traits, Cta_tile, CTAS_PER_HEAD>
    : public Gmem_tile_o_8bit<fmha::Turing_imma_int8_int32_traits, Cta_tile, CTAS_PER_HEAD>
{

    // The traits class.
    using Traits = fmha::Turing_imma_int8_int32_traits;
    // The base class.
    using Base = Gmem_tile_o_8bit<Traits, Cta_tile, CTAS_PER_HEAD>;

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_o(Params const& params, Block_info const& block_info, int tidx, int cta_row_offset = 0,
        int cta_col_offset_in_bytes = 0)
        : Base(params, block_info, tidx, cta_row_offset, cta_col_offset_in_bytes)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, int CTAS_PER_HEAD>
struct Gmem_tile_o<fmha::Ampere_imma_int8_int32_traits, Cta_tile, CTAS_PER_HEAD>
    : public Gmem_tile_o_8bit<fmha::Ampere_imma_int8_int32_traits, Cta_tile, CTAS_PER_HEAD>
{

    // The traits class.
    using Traits = fmha::Ampere_imma_int8_int32_traits;
    // The base class.
    using Base = Gmem_tile_o_8bit<Traits, Cta_tile, CTAS_PER_HEAD>;

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_o(Params const& params, Block_info const& block_info, int tidx, int cta_row_offset = 0,
        int cta_col_offset_in_bytes = 0)
        : Base(params, block_info, tidx, cta_row_offset, cta_col_offset_in_bytes)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, int CTAS_PER_HEAD>
struct Gmem_tile_o<fmha::Ada_qmma_e4m3_fp32_traits, Cta_tile, CTAS_PER_HEAD>
    : public Gmem_tile_o_8bit<fmha::Ada_qmma_e4m3_fp32_traits, Cta_tile, CTAS_PER_HEAD>
{

    // The traits class.
    using Traits = fmha::Ada_qmma_e4m3_fp32_traits;
    // The base class.
    using Base = Gmem_tile_o_8bit<Traits, Cta_tile, CTAS_PER_HEAD>;

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_o(Params const& params, Block_info const& block_info, int tidx, int cta_row_offset = 0,
        int cta_col_offset_in_bytes = 0)
        : Base(params, block_info, tidx, cta_row_offset, cta_col_offset_in_bytes)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, int CTAS_PER_HEAD>
struct Gmem_tile_o<fmha::Ada_qmma_e4m3_fp16_traits, Cta_tile, CTAS_PER_HEAD>
    : public Gmem_tile_o_8bit<fmha::Ada_qmma_e4m3_fp16_traits, Cta_tile, CTAS_PER_HEAD>
{

    // The traits class.
    using Traits = fmha::Ada_qmma_e4m3_fp16_traits;
    // The base class.
    using Base = Gmem_tile_o_8bit<Traits, Cta_tile, CTAS_PER_HEAD>;

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_o(Params const& params, Block_info const& block_info, int tidx, int cta_row_offset = 0,
        int cta_col_offset_in_bytes = 0)
        : Base(params, block_info, tidx, cta_row_offset, cta_col_offset_in_bytes)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, typename Output_type, int CTAS_PER_HEAD>
struct Gmem_tile_o_16bit
{

    // This stores the fp32 accumulators of Ada_qmma_e4m3_fp32_traits as 16bit values to
    // the global memory.

    static_assert(std::is_same<Traits, fmha::Ada_qmma_e4m3_fp32_traits>::value);
    static_assert(std::is_same<Output_type, uint16_t>::value || std::is_same<Output_type, nv_bfloat16>::value);

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

    // The valid size of a row in bytes.
    // Note: cross-attention kernels rely on head dim from runtime instead of from compile-time.
    // This approach deviates from self-attention kernels. To explore a unified approach.
    enum
    {
        VALID_BYTES_PER_ROW = Cta_tile::VALID_N * BYTES_PER_ELEMENT
    };

    // The size of each STG.
    enum
    {
        BYTES_PER_STG = 8
    };

    // The STG packed data type
    using Stg_packed_type = typename Uint_from_size_in_bytes<BYTES_PER_STG>::Type;

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
    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_o_16bit(Params const& params, Block_info const& block_info, int tidx,
        int cta_row_offset = 0, int cta_col_offset_in_bytes = 0)
        : params_o_stride_in_bytes_(params.o_stride_in_bytes)
        , actual_seqlen_(block_info.actual_q_seqlen)
        , params_scale_bmm2_(params.scale_bmm2_d ? *params.scale_bmm2_d : params.scale_bmm2)
#ifdef GENERATE_CUBIN
        , params_enable_i2f_trick_(false)
#else
        , params_enable_i2f_trick_(params.enable_i2f_trick)
#endif
        , o_ptr_(reinterpret_cast<char*>(params.o_ptr))
#if USE_DEMO_BERT_PARAMS
        , o_scratch_ptr_(nullptr)
    {
#else
        , o_scratch_ptr_(reinterpret_cast<uint4*>(params.o_scratch_ptr))
    {
#endif

        // Compute the position in the sequence (within the CTA for the moment).
        int row = tidx / THREADS_PER_ROW;
        // Compute the position of the thread in the row.
        int col = tidx % THREADS_PER_ROW;

        // Is it an active thread for the very last STG?
        if (HAS_INCOMPLETE_STG)
        {
            is_active_for_last_stg_ = row + (STGS - 1) * ROWS_PER_STG < Cta_tile::M;
        }

        // Store the row to check against the length before loads.
        row_ = cta_row_offset + row;
        col_in_bytes_ = cta_col_offset_in_bytes + col * BYTES_PER_STG;

        // The row offset in the batched GEMM.
        int64_t row_offset = (int64_t) row_ * params.o_stride_in_bytes;
        // The amount of bytes per row without padding (runtime).
        int const valid_bytes_per_row = params.dv * BYTES_PER_ELEMENT;
        // Take the batch/head offset into account.
        row_offset += block_info.bidx * valid_bytes_per_row;
        // Assemble the final pointer.
        o_ptr_ += row_offset + col_in_bytes_;

        // Is it an active thread?
        is_active_ = ALL_THREADS_ACTIVE || (row < ROWS_PER_LOOP && col_in_bytes_ < VALID_BYTES_PER_ROW);

        // Do not store if the thread is in the padded area
        is_active_ = is_active_ && col < valid_bytes_per_row / BYTES_PER_STG;

        // For the scratch space, the pointer has int32 type so it accounts for the *4 factor.
        o_scratch_ptr_ += blockIdx.y * STGS_PER_LOOP * ACTIVE_THREADS + tidx;
    }

    // Store data to global memory.
    inline __device__ void store(uint4 const (&src)[STGS_PER_LOOP], int mi)
    {

// Iterate over the different STGs.
#pragma unroll
        for (int ii = 0; ii < STGS_PER_LOOP; ++ii)
        {

            // Break early if we exceed s_i...
            int jj = mi * STGS_PER_LOOP + ii;
            if (row_ + jj * ROWS_PER_STG >= actual_seqlen_)
            {
                return;
            }
            using Src_type = typename Traits::Accumulator_type;
            // Packs the 32bit/16bit values to 16bit.
            // Depending on the type, applies extra scaling with parameter scale_bmm2.
            Stg_packed_type dst = Acc_packer<Src_type, Output_type>::run(this, src[ii]);
            float const* row_ptr = reinterpret_cast<float const*>(&src[ii]);

            // Store the result.
            if (is_active_ && (!HAS_INCOMPLETE_STG || (jj < STGS - 1 || is_active_for_last_stg_)))
            {
                fmha::stg(o_ptr_ + jj * ROWS_PER_STG * params_o_stride_in_bytes_, dst);
            }
        }
    }

    // The stride between rows for the QKV matrice.
    int64_t params_o_stride_in_bytes_;
    // The scaling factor to convert to int8.
    uint32_t const params_scale_bmm2_;
    // Do we enable the i2f trick?
    bool const params_enable_i2f_trick_;
    // The pointer.
    char* o_ptr_;
    // The pointer to the scratch space to do the reduction (for CTAS_PER_HEAD > 1).
    uint4* o_scratch_ptr_;
    // The row, col stored by this thread (i.e. the position in that sequence).
    int row_, col_in_bytes_;
    // The size of the sequence length computed by that CTA.
    int actual_seqlen_;

    // Is it an active thread?
    int is_active_, is_active_for_last_stg_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, int CTAS_PER_HEAD>
struct Gmem_tile_o_uint16 : public Gmem_tile_o_16bit<Traits, Cta_tile, uint16_t, CTAS_PER_HEAD>
{

    using Base = Gmem_tile_o_16bit<Traits, Cta_tile, uint16_t, CTAS_PER_HEAD>;

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_o_uint16(Params const& params, Block_info const& block_info, int tidx,
        int cta_row_offset = 0, int cta_col_offset_in_bytes = 0)
        : Base(params, block_info, tidx, cta_row_offset, cta_col_offset_in_bytes)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, int CTAS_PER_HEAD>
struct Gmem_tile_o_bfloat16 : public Gmem_tile_o_16bit<Traits, Cta_tile, nv_bfloat16, CTAS_PER_HEAD>
{

    using Base = Gmem_tile_o_16bit<Traits, Cta_tile, nv_bfloat16, CTAS_PER_HEAD>;

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_o_bfloat16(Params const& params, Block_info const& block_info, int tidx,
        int cta_row_offset = 0, int cta_col_offset_in_bytes = 0)
        : Base(params, block_info, tidx, cta_row_offset, cta_col_offset_in_bytes)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, int CTAS_PER_HEAD>
struct Imma_gmem_tile_o_interleaved
{

    // The mma tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    enum
    {
        VEC = 32
    };

    enum
    {
        NUM_SLICES = Cta_tile::N / VEC
    };

    // DEBUG.
    static_assert(NUM_SLICES == 1 || NUM_SLICES == 2, "");

    // END OF DEBUG.

    // The size of each element.
    enum
    {
        BYTES_PER_ELEMENT = 1
    };

    // The size of a row in bytes.
    enum
    {
        BYTES_PER_ROW = VEC * BYTES_PER_ELEMENT
    };

    // The size of each STG.
    enum
    {
        BYTES_PER_STG = 4
    };

    // The number of threads to store a "row" of the matrix. We force it to 8
    enum
    {
        THREADS_PER_ROW = BYTES_PER_ROW / BYTES_PER_STG
    };

    // DEBUG.
    static_assert(THREADS_PER_ROW == 8 && BYTES_PER_STG == 4, "");

    // END OF DEBUG.

    // the "logical" number of rows. think of rows per slice
    enum
    {
        ROWS = Cta_tile::M
    };

    // "physical" rows
    enum
    {
        TOTAL_ROWS = ROWS * NUM_SLICES
    };

    // The number of "rows" stored per iteration of the loop. The output of 1 MMA.
    enum
    {
        ROWS_PER_LOOP_PER_SLICE = Mma_tile::M_PER_MMA_PER_CTA
    };

    enum
    {
        ROWS_PER_LOOP = Mma_tile::M_PER_MMA_PER_CTA * NUM_SLICES
    };

    // DEBUG.
    static_assert(ROWS_PER_LOOP == 16 * Cta_tile::WARPS_M * NUM_SLICES, "");

    // END OF DEBUG.

    // The number of outer loop for the stores.
    enum
    {
        LOOPS = TOTAL_ROWS / ROWS_PER_LOOP
    };

    // Make sure the math is correct.
    static_assert(LOOPS == (int) Mma_tile::MMAS_M, "");

    // The number of "rows" stored per STG -- for it to be the number of rows per MMA instruction.
    enum
    {
        ROWS_PER_STG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW
    };

    // The number of STGs needed to store a chunk of the Q matrix.
    enum
    {
        STGS_PER_LOOP = fmha::Div_up<ROWS_PER_LOOP, ROWS_PER_STG>::VALUE
    };

    enum
    {
        STGS_PER_SLICE = STGS_PER_LOOP / NUM_SLICES
    };

    // DEBUG.
    static_assert(
        (Cta_tile::WARPS_M == 1 && STGS_PER_SLICE == 1) || (Cta_tile::WARPS_M == 2 && STGS_PER_SLICE == 2), "");

    // END OF DEBUG.

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Imma_gmem_tile_o_interleaved(
        Params const& params, Block_info const& block_info, int tidx, int cta_row_offset = 0)
        : params_o_stride_in_bytes_(params.o_stride_in_bytes)
        , actual_seqlen_(block_info.actual_seqlen - cta_row_offset)
        , params_scale_bmm2_(params.scale_bmm2)
        , params_enable_i2f_trick_(params.enable_i2f_trick)
        , o_ptr_(reinterpret_cast<char*>(params.o_ptr))
        , total_(params.o_stride_in_bytes)
    {

        int bidh = block_info.bidh;
        int sum_s = block_info.sum_s;

        row_ = tidx / THREADS_PER_ROW;
        int col = tidx % THREADS_PER_ROW;

        // h is N
        // d is H
        // want to save as: h x (d/32) x total x 32 (think 3 x h x (d/32) x b x s x 32)

        int block_offset = bidh * NUM_SLICES * total_ + sum_s; // bidh * GROUPS * B * S + b * S
        int row_offset = (block_offset + cta_row_offset) * BYTES_PER_ROW;

        o_ptr_ += row_offset + col * BYTES_PER_STG;
    }

    // Load data from global memory.
    inline __device__ void load(uint4 (&dst)[STGS_PER_LOOP], int mi)
    {
        static_assert(CTAS_PER_HEAD == 1, "Not implemented");
    }

    // Store data to global memory.
    inline __device__ void store(uint4 const (&src)[STGS_PER_LOOP], int mi)
    {

        int rows_so_far = mi * STGS_PER_LOOP * ROWS_PER_STG;
        int rows_so_far_per_slice = rows_so_far / 2;

        // The scale.
        float const& scale = reinterpret_cast<float const&>(params_scale_bmm2_);

// Iterate over the different STGs.
#pragma unroll
        for (int ii = 0; ii < STGS_PER_LOOP; ++ii)
        {
            // if(ii == 1) return;
            // decompose the iteration into slice
            int slice = ii / STGS_PER_SLICE;
            int si = ii % STGS_PER_SLICE;
            // dbg 256
            //      assert(STGS_PER_SLICE == 1);
            //      assert(STGS_PER_LOOP == 2);
            //      assert(slice == ii);
            // the number of rows one CTA-wide STG writes
            static_assert(ROWS_PER_STG == 16, ""); // only holds for 4 warps/128 threads
            int row_in_slice = row_ + si * ROWS_PER_STG + rows_so_far_per_slice;

            // we cannot return early, because the second half of iterates are
            // responsible for the bottom slice
            if (row_in_slice >= min(actual_seqlen_, ROWS))
            {
                continue;
            }

            int offset = (slice * total_ + row_in_slice) * BYTES_PER_ROW;

            // The accumulators are in int32_t.
            int4 const& val = reinterpret_cast<int4 const&>(src[ii]);

            //      if(threadIdx.x == 96){
            //      printf("mi=%d ii=%d S=%d si=%d sofar=%d row=%d as=%d\n", mi, ii, slice, si,
            //      rows_so_far_per_slice, row_in_slice, actual_seqlen_)  ;
            //      }

            uint32_t dst = quantize(val, scale, params_enable_i2f_trick_);
            // Store the result.
            fmha::stg(o_ptr_ + offset, dst);
        }
    }

    // Store data to global memory.
    inline __device__ void store(uint4 const (&src)[STGS_PER_LOOP], uint4 const (&old)[STGS_PER_LOOP], int mi)
    {
        static_assert(CTAS_PER_HEAD == 1, "Not implemented");
    }

    // Move the pointer.
    inline __device__ void move()
    {
        o_ptr_ += (int64_t) ROWS * BYTES_PER_ROW;
        actual_seqlen_ -= ROWS;
    }

    // The stride between rows for the QKV matrice.
    int64_t const params_o_stride_in_bytes_;
    // The scaling factor to convert to int8.
    uint32_t const params_scale_bmm2_;
    // Do we enable the i2f trick?
    bool const params_enable_i2f_trick_;
    // The pointer.
    char* o_ptr_;
    int row_;
    int actual_seqlen_;
    int total_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace v2
} // namespace fmha
