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

#include "fmha/fragment.h"
#include "fmha/utils.h"
#include <cfloat>

namespace fmha
{

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Sum_
{
    enum
    {
        IS_SUM = 1
    };

    static inline __device__ float apply(float x, float y)
    {
        return x + y;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Max_
{
    enum
    {
        IS_SUM = 0
    };

    static inline __device__ float apply(float x, float y)
    {
        return fmaxf(x, y);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int FMHA_VERSION>
inline __device__ float apply_exp_(float x, float max)
{
    return isinf(x) ? 0.f : __expf(x - max);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ float apply_exp_<2>(float x, float max)
{
    return __expf(x - max);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AlibiParams>
inline __device__ float get_alibi_head_scaling_factor(int const in_head_id, AlibiParams const& params)
{
    int const head_id = params.head_idx_offset + in_head_id;
    if (head_id < params.h_pow_2)
    {
        // 2^(head_id * -8 / h)
        return exp2f((head_id + 1) * 2 * params.alibi_neg4_div_h) * params.scale_after_alibi;
    }
    else
    {
        // 1,3,5... etc
        float const adjusted_head_id = 2 * (head_id - params.h_pow_2) + 1;
        // 2^(adjusted_head_id * -4 / h)
        return exp2f(adjusted_head_id * params.alibi_neg4_div_h) * params.scale_after_alibi;
        ;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int COLS>
struct ReadType
{
    using T = float;
};

template <>
struct ReadType<4>
{
    using T = float;
};

template <>
struct ReadType<8>
{
    using T = float2;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, typename Kernel_traits>
struct Smem_tile_reduce
{
    // Helper class to distribute MMA tiles reduced over rows per warp over quads.

    // The Mma tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    // The number of MMAs in M/N dimensions.
    enum
    {
        MMAS_M = Mma_tile::MMAS_M
    };

    enum
    {
        MMAS_N = Mma_tile::MMAS_N
    };

    enum
    {
        WARPS_M = Cta_tile::WARPS_M
    };

    enum
    {
        WARPS_N = Cta_tile::WARPS_N
    };

    static constexpr int ROWS = WARPS_M * MMAS_M * 16;
    static constexpr int COLS = WARPS_N;
    static constexpr int ROWS_PER_XOR_PATTERN = (COLS == 8) ? 4 : 8;
    static constexpr int BYTES_PER_TILE = ROWS * COLS * sizeof(float);
    static constexpr int ELTS_PER_TILE = ROWS * COLS;

    static constexpr int THREADS_PER_GROUP = Kernel_traits::Gmem_tile_o::THREADS_PER_ROW;
    static constexpr int ROWS_PER_WARP = 32 / THREADS_PER_GROUP;
    static constexpr int LOOPS = Kernel_traits::Gmem_tile_o::LOOPS;

    using read_t = typename ReadType<COLS>::T;

    __device__ inline Smem_tile_reduce(float* smem_, int const tidx)
    {

        int lane = tidx % 32;
        int warp = tidx / 32;

        int warp_m = warp % WARPS_M;
        int warp_n = warp / WARPS_M;

        qid_ = lane % 4;
        int qp = lane / 4;

        // Swizzle the column to avoid 2-fold bank conflicts when we have 8 warps.
        // This won't affect reading as we assume commutative reduction ops.
        int const col = warp_n ^ (qp / ROWS_PER_XOR_PATTERN);
        smem_write_ = &smem_[warp_m * 16 * MMAS_M * WARPS_N + qp * WARPS_N + col];
        smem_read_ = &reinterpret_cast<read_t*>(smem_)[warp_m * 16 * MMAS_M * 4 + qp * 4 + qid_];
    }

    __device__ inline void store(float (&frag)[2 * MMAS_M])
    {
        if (qid_ == 0)
        {
#pragma unroll
            for (int mi = 0; mi < MMAS_M; mi++)
            {
                int offset = mi * 16 * WARPS_N;
                smem_write_[offset + 0 * 8 * WARPS_N] = frag[mi * 2 + 0];
                smem_write_[offset + 1 * 8 * WARPS_N] = frag[mi * 2 + 1];
            }
        }
    }

    __device__ inline void load(read_t (&frag)[2 * MMAS_M])
    {
#pragma unroll
        for (int mi = 0; mi < MMAS_M; mi++)
        {
            int offset = mi * 16 * 4;
            frag[mi * 2 + 0] = smem_read_[offset + 0 * 8 * 4];
            frag[mi * 2 + 1] = smem_read_[offset + 1 * 8 * 4];
        }
    }

    int qid_;
    float* smem_write_;
    read_t* smem_read_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, typename Kernel_traits>
struct Softmax_base
{

    // The Mma tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    // The number of MMAs in M/N dimensions.
    enum
    {
        MMAS_M = Mma_tile::MMAS_M
    };

    enum
    {
        MMAS_N = Mma_tile::MMAS_N
    };

    // The number of groups of warp such that we have at most 4 warps writing consecutive elements.
    enum
    {
        GROUPS = fmha::Div_up<Cta_tile::WARPS_N, 4>::VALUE
    };

    // The number of elements that we are going to store per row.
    enum
    {
        ELEMENTS_PER_ROW = Cta_tile::WARPS_N / GROUPS
    };

    // The number of rows.
    enum
    {
        ROWS = Cta_tile::M * GROUPS
    };

    // The total number of elements.
    enum
    {
        ELEMENTS = ROWS * ELEMENTS_PER_ROW
    };

    // If shared memory is used
    enum
    {
        USE_SHARED_MEMORY = Cta_tile::WARPS_N > 1
    };

    // DEBUG.
    static_assert(ELEMENTS == Cta_tile::M * Cta_tile::WARPS_N, "");

    // END OF DEBUG.

    // The number of rows per thread.
    enum
    {
        ROWS_PER_THREAD = MMAS_M * 2
    };

    // Ctor.
    template <typename Params>
    inline __device__ Softmax_base(Params const& params, void* smem, int bidb, int tidx)
        : smem_(reinterpret_cast<float*>(smem))
        , tidx_(tidx)
    {

        // Extract the position in the warp.
        int warp = tidx / Cta_tile::THREADS_PER_WARP;
        int lane = tidx % Cta_tile::THREADS_PER_WARP;

        // Decompose the warp index into M and N.
        int warp_m = warp % Cta_tile::WARPS_M;
        int warp_n = warp / Cta_tile::WARPS_M;

        // Decompose the warp-n index into group/position-inside-the-group.
        int warp_g = warp_n / ELEMENTS_PER_ROW;
        int warp_i = warp_n % ELEMENTS_PER_ROW;

        // The location written by the threads.
        int write_row = warp_g * Cta_tile::M + warp_m * Mma_tile::M_PER_MMA + lane / 4;
        int write_col = warp_i;

        // Assemble the write pointer.
        smem_write_ = &smem_[write_row * ELEMENTS_PER_ROW + write_col];

        // Assemble the read pointer.
        smem_read_ = &smem_[warp_m * Mma_tile::M_PER_MMA + lane / 4];
    }

    // Apply mask before softmax. Use 1 byte per MMA distributed as 2x4.
    template <typename Mask>
    inline __device__ void apply_mask(Mask const& mask)
    {
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ii = 0; ii < 2; ++ii)
            {
#pragma unroll
                for (int ni = 0; ni < MMAS_N; ++ni)
                {
#pragma unroll
                    for (int jj = 0; jj < 4; ++jj)
                    {
                        if (!mask.is_valid(mi, ni, ii, jj))
                        {
                            elt_[2 * mi + ii][4 * ni + jj] = -FLT_MAX;
                        }
                    }
                }
            }
        }
    }

    template <typename Mask, typename AlibiParams>
    inline __device__ void apply_mask_alibi(Mask const& mask, int head_id, AlibiParams const& alibi_params)
    {
        // 'if constexpr' because ALiBi is only defined for causal masks
        if constexpr (Kernel_traits::CAUSAL_MASK)
        {
            float m = get_alibi_head_scaling_factor<AlibiParams>(head_id, alibi_params);
#pragma unroll
            for (int mi = 0; mi < MMAS_M; ++mi)
            {
#pragma unroll
                for (int ii = 0; ii < 2; ++ii)
                {
#pragma unroll
                    for (int ni = 0; ni < MMAS_N; ++ni)
                    {
#pragma unroll
                        for (int jj = 0; jj < 4; ++jj)
                        {
                            int row, col;
                            mask.get_row_col(row, col, mi, ni, ii, jj);
                            if (mask.is_valid(row, col))
                            {
                                // Since softmax is shift invariant,
                                //  it is sufficient just to use the column as the multiplier
                                elt_[2 * mi + ii][4 * ni + jj]
                                    = elt_[2 * mi + ii][4 * ni + jj] * alibi_params.scale_after_alibi
                                    + m * (col + alibi_params.sequence_pos_offset);
                            }
                            else
                            {
                                elt_[2 * mi + ii][4 * ni + jj] = -FLT_MAX;
                            }
                        }
                    }
                }
            }
        }
        else
        {
            __builtin_unreachable();
        }
    }

    // Apply the mask to unpacked data.
    inline __device__ void apply_mask(uint32_t const (&packed_mask)[MMAS_M])
    {

        // This code works only if we have MMAS_N <= 4.
        static_assert(MMAS_N <= 4, "");

        // Expand the mask.
        int mask[MMAS_M * 2][MMAS_N * 4];
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {
                mask[2 * mi + 0][4 * ni + 0] = packed_mask[mi] & (1u << (8 * ni + 0));
                mask[2 * mi + 0][4 * ni + 1] = packed_mask[mi] & (1u << (8 * ni + 1));
                mask[2 * mi + 1][4 * ni + 0] = packed_mask[mi] & (1u << (8 * ni + 2));
                mask[2 * mi + 1][4 * ni + 1] = packed_mask[mi] & (1u << (8 * ni + 3));
                mask[2 * mi + 0][4 * ni + 2] = packed_mask[mi] & (1u << (8 * ni + 4));
                mask[2 * mi + 0][4 * ni + 3] = packed_mask[mi] & (1u << (8 * ni + 5));
                mask[2 * mi + 1][4 * ni + 2] = packed_mask[mi] & (1u << (8 * ni + 6));
                mask[2 * mi + 1][4 * ni + 3] = packed_mask[mi] & (1u << (8 * ni + 7));
            }
        }

// Apply the mask.
#pragma unroll
        for (int mi = 0; mi < MMAS_M * 2; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N * 4; ++ni)
            {
                if (!mask[mi][ni])
                {
                    elt_[mi][ni] = -FLT_MAX;
                }
            }
        }
    }

    // Mask the elements that are outside the the sequence length.
    inline __device__ void apply_mask(int const actual_seqlen)
    {

        // The warp/lane decomposition.
        int const warp = threadIdx.x / Cta_tile::THREADS_PER_WARP;
        int const lane = threadIdx.x % Cta_tile::THREADS_PER_WARP;

        // The warp in the n dimension.
        int const warp_n = warp / Cta_tile::WARPS_M;
        // The position within a quad.
        int const quad_lane = lane % 4;

#pragma unroll
        for (int mi = 0; mi < MMAS_M * 2; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {
                // Determine the position in the sequence.
                int const offset = ni * Mma_tile::N_PER_MMA_PER_CTA + warp_n * 16;
                if (offset + 0 + 2 * quad_lane >= actual_seqlen)
                {
                    elt_[mi][4 * ni + 0] = -FLT_MAX; // 0
                }
                if (offset + 1 + 2 * quad_lane >= actual_seqlen)
                {
                    elt_[mi][4 * ni + 1] = -FLT_MAX; // 1
                }
                if (offset + 8 + 2 * quad_lane >= actual_seqlen)
                {
                    elt_[mi][4 * ni + 2] = -FLT_MAX; // 8
                }
                if (offset + 9 + 2 * quad_lane >= actual_seqlen)
                {
                    elt_[mi][4 * ni + 3] = -FLT_MAX; // 9
                }
            }
        }
    }

    // Apply the exp to all the elements.
    inline __device__ void apply_exp(float const max)
    {
#pragma unroll
        for (int mi = 0; mi < MMAS_M * 2; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N * 4; ++ni)
            {
                elt_[mi][ni] = apply_exp_<Kernel_traits::VERSION>(elt_[mi][ni], max);
            }
        }
    }

    // Apply the exp to all the elements.
    inline __device__ void apply_scale_exp(float const (&max)[MMAS_M * 2], float scale_bmm1)
    {
#pragma unroll
        for (int mi = 0; mi < MMAS_M * 2; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N * 4; ++ni)
            {
                elt_[mi][ni] = apply_exp_<Kernel_traits::VERSION>(scale_bmm1 * elt_[mi][ni], max[mi]);
            }
        }
    }

    // Apply the exp to all the elements.
    inline __device__ void apply_exp(float const (&max)[MMAS_M * 2])
    {
#pragma unroll
        for (int mi = 0; mi < MMAS_M * 2; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N * 4; ++ni)
            {
                elt_[mi][ni] = apply_exp_<Kernel_traits::VERSION>(elt_[mi][ni], max[mi]);
            }
        }
    }

    // Do a warp-wide reduction.
    template <typename Functor>
    inline __device__ void reduce_Nx1(float (&dst)[MMAS_M * 2])
    {
#if defined(USE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE)
        if (Functor::IS_SUM)
        {
// Apply the summation inside the thread.
#pragma unroll
            for (int mi = 0; mi < MMAS_M * 2; ++mi)
            {
                float tmp[2] = {0.f, 0.f};
#pragma unroll
                for (int ni = 0; ni < MMAS_N; ++ni)
                {
                    tmp[0] += elt_[mi][4 * ni + 0] + elt_[mi][4 * ni + 1];
                    tmp[1] += elt_[mi][4 * ni + 2] + elt_[mi][4 * ni + 3];
                }
                dst[mi] = tmp[0] + tmp[1];
            }
        }
        else
#endif // defined(USE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE)
        {
// Apply the functor for each row inside a thread.
#pragma unroll
            for (int mi = 0; mi < MMAS_M * 2; ++mi)
            {
                dst[mi] = elt_[mi][0];
#pragma unroll
                for (int ni = 1; ni < MMAS_N * 4; ++ni)
                {
                    dst[mi] = Functor::apply(dst[mi], elt_[mi][ni]);
                }
            }
        }

// Apply the functor for each row inside each group of 4 threads.
#pragma unroll
        for (int mi = 0; mi < MMAS_M * 2; ++mi)
        {
            dst[mi] = Functor::apply(dst[mi], __shfl_xor_sync(uint32_t(-1), dst[mi], 1));
            dst[mi] = Functor::apply(dst[mi], __shfl_xor_sync(uint32_t(-1), dst[mi], 2));
        }
    }

    // Do a CTA-wide reduction.
    template <typename Functor>
    inline __device__ float reduce_2x2()
    {
        float dst[MMAS_M * 2];
#if defined(USE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE)
        if (Functor::IS_SUM)
        {
// Apply the summation inside the thread.
#pragma unroll
            for (int mi = 0; mi < MMAS_M * 2; ++mi)
            {

                // Pair-wise adds in the different threads of the reference code (x+y and z+w).
                float a_01 = elt_[mi][0] + elt_[mi][1];
                float a_45 = elt_[mi][4] + elt_[mi][5];

                //// tmp[0/1] += __shfl_xor(2) in the reference code.
                a_01 += elt_[mi][2] + elt_[mi][3];
                a_45 += elt_[mi][6] + elt_[mi][7];

                //// tmp[0/1] += __shfl_xor(8) in the reference code.
                a_01 += a_45;

                if (MMAS_N >= 3)
                {
                    float a_89 = elt_[mi][8] + elt_[mi][9];
                    a_89 += elt_[mi][10] + elt_[mi][11];
                    if (MMAS_N == 4)
                    {
                        float a_cd = elt_[mi][12] + elt_[mi][13];
                        a_cd += elt_[mi][14] + elt_[mi][15];
                        a_89 += a_cd;
                    }
                    a_01 += a_89;
                }
                dst[mi] = a_01;
            }
        }
        else
#endif // defined(USE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE)
        {
// Apply the functor for each row inside a thread.
#pragma unroll
            for (int mi = 0; mi < MMAS_M * 2; ++mi)
            {
                dst[mi] = elt_[mi][0];
#pragma unroll
                for (int ni = 1; ni < MMAS_N * 4; ++ni)
                {
                    dst[mi] = Functor::apply(dst[mi], elt_[mi][ni]);
                }
            }
        }

// Apply the functor for each row inside each group of 4 threads.
#pragma unroll
        for (int mi = 0; mi < MMAS_M * 2; ++mi)
        {
            dst[mi] = Functor::apply(dst[mi], __shfl_xor_sync(uint32_t(-1), dst[mi], 1));
            dst[mi] = Functor::apply(dst[mi], __shfl_xor_sync(uint32_t(-1), dst[mi], 2));
        }

// Store the different values.
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
            if (tidx_ % 4 == 0)
            {
                smem_write_[(mi * Mma_tile::M_PER_MMA_PER_CTA + 0) * ELEMENTS_PER_ROW] = dst[2 * mi + 0];
                smem_write_[(mi * Mma_tile::M_PER_MMA_PER_CTA + 8) * ELEMENTS_PER_ROW] = dst[2 * mi + 1];
            }
        }

        // Make sure the values are in shared memory.
        __syncthreads();

        // Load 2 values (one for each warp).
        float2 tmp = reinterpret_cast<float2 const*>(smem_)[tidx_];

        // Compute the reduction of those 2 values in a binary-tree fashion.
        return Functor::apply(tmp.x, tmp.y);
    }

    // Do a CTA-wide reduction.
    template <typename Functor>
    inline __device__ float reduce_1x4()
    {
        float dst[MMAS_M * 2];
#if defined(USE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE)
        if (Functor::IS_SUM)
        {
// Apply the summation inside the thread.
#pragma unroll
            for (int mi = 0; mi < MMAS_M * 2; ++mi)
            {
                float tmp[2] = {0.f, 0.f};
#pragma unroll
                for (int ni = 0; ni < MMAS_N; ++ni)
                {
                    tmp[0] += elt_[mi][4 * ni + 0] + elt_[mi][4 * ni + 1];
                    tmp[1] += elt_[mi][4 * ni + 2] + elt_[mi][4 * ni + 3];
                }
                dst[mi] = tmp[0] + tmp[1];
            }
        }
        else
#endif // defined(USE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE)
        {
// Apply the functor for each row inside a thread.
#pragma unroll
            for (int mi = 0; mi < MMAS_M * 2; ++mi)
            {
                dst[mi] = elt_[mi][0];
#pragma unroll
                for (int ni = 1; ni < MMAS_N * 4; ++ni)
                {
                    dst[mi] = Functor::apply(dst[mi], elt_[mi][ni]);
                }
            }
        }

// Apply the functor for each row inside each group of 4 threads.
#pragma unroll
        for (int mi = 0; mi < MMAS_M * 2; ++mi)
        {
            dst[mi] = Functor::apply(dst[mi], __shfl_xor_sync(uint32_t(-1), dst[mi], 1));
            dst[mi] = Functor::apply(dst[mi], __shfl_xor_sync(uint32_t(-1), dst[mi], 2));
        }

// Store the different values.
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
            if (tidx_ % 4 == 0)
            {
                smem_write_[(mi * Mma_tile::M_PER_MMA_PER_CTA + 0) * ELEMENTS_PER_ROW] = dst[2 * mi + 0];
                smem_write_[(mi * Mma_tile::M_PER_MMA_PER_CTA + 8) * ELEMENTS_PER_ROW] = dst[2 * mi + 1];
            }
        }

        // Make sure the values are in shared memory.
        __syncthreads();

        // Load 8 values (one for each warp). The /8 corresponds to /(4*2) where 4 is from the
        // float4.
        float4 tmp[1];
        if (tidx_ < Cta_tile::M)
        {
            tmp[0] = reinterpret_cast<float4 const*>(&smem_[0 * ELEMENTS / 2])[tidx_];
        }

        // Compute the reduction of those 8 values in a binary-tree fashion.
        tmp[0].x = Functor::apply(tmp[0].x, tmp[0].y);
        tmp[0].z = Functor::apply(tmp[0].z, tmp[0].w);
        tmp[0].x = Functor::apply(tmp[0].x, tmp[0].z);

        // Return the final reduction.
        return tmp[0].x;
    }

    // Do a CTA-wide reduction.
    template <typename Functor>
    inline __device__ float reduce_1x8()
    {
        float dst[MMAS_M * 2];
#if defined(USE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE)
        if (Functor::IS_SUM)
        {
            // Apply the summation inside the thread.
            float tmp[MMAS_M * 2][2];
#pragma unroll
            for (int mi = 0; mi < MMAS_M * 2; ++mi)
            {
                tmp[mi][0] = 0.f;
                tmp[mi][1] = 0.f;
#pragma unroll
                for (int ni = 0; ni < MMAS_N; ++ni)
                {
                    tmp[mi][0] += elt_[mi][4 * ni + 0];
                    tmp[mi][0] += elt_[mi][4 * ni + 1];
                    tmp[mi][1] += elt_[mi][4 * ni + 2];
                    tmp[mi][1] += elt_[mi][4 * ni + 3];
                }
                dst[mi] = tmp[mi][0] + tmp[mi][1];
            }
        }
        else
#endif // defined(USE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE)
        {
// Apply the functor for each row inside a thread.
#pragma unroll
            for (int mi = 0; mi < MMAS_M * 2; ++mi)
            {
                dst[mi] = elt_[mi][0];
#pragma unroll
                for (int ni = 1; ni < MMAS_N * 4; ++ni)
                {
                    dst[mi] = Functor::apply(dst[mi], elt_[mi][ni]);
                }
            }
        }

// Apply the functor for each row inside each group of 4 threads.
#pragma unroll
        for (int mi = 0; mi < MMAS_M * 2; ++mi)
        {
            dst[mi] = Functor::apply(dst[mi], __shfl_xor_sync(uint32_t(-1), dst[mi], 1));
            dst[mi] = Functor::apply(dst[mi], __shfl_xor_sync(uint32_t(-1), dst[mi], 2));
        }

// Store the different values.
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
            if (tidx_ % 4 == 0)
            {
                smem_write_[(mi * Mma_tile::M_PER_MMA_PER_CTA + 0) * ELEMENTS_PER_ROW] = dst[2 * mi + 0];
                smem_write_[(mi * Mma_tile::M_PER_MMA_PER_CTA + 8) * ELEMENTS_PER_ROW] = dst[2 * mi + 1];
            }
        }

        // Make sure the values are in shared memory.
        __syncthreads();

        // Load 8 values (one for each warp). The /8 corresponds to /(4*2) where 4 is from the
        // float4.
        float4 tmp[2];
        if (tidx_ < Cta_tile::M)
        {
            tmp[0] = reinterpret_cast<float4 const*>(&smem_[0 * ELEMENTS / 2])[tidx_];
            tmp[1] = reinterpret_cast<float4 const*>(&smem_[1 * ELEMENTS / 2])[tidx_];
        }

        // Compute the reduction of those 8 values in a binary-tree fashion.
        tmp[0].x = Functor::apply(tmp[0].x, tmp[0].y);
        tmp[0].z = Functor::apply(tmp[0].z, tmp[0].w);
        tmp[1].x = Functor::apply(tmp[1].x, tmp[1].y);
        tmp[1].z = Functor::apply(tmp[1].z, tmp[1].w);
        tmp[0].x = Functor::apply(tmp[0].x, tmp[0].z);
        tmp[1].x = Functor::apply(tmp[1].x, tmp[1].z);
        tmp[0].x = Functor::apply(tmp[0].x, tmp[1].x);

        // Return the result.
        return tmp[0].x;
    }

    // Do a CTA-wide reduction.
    template <typename Functor>
    inline __device__ float reduce_()
    {

        // The result of the reduction. Threads 0..Cta_tile::M-1 own a single row value.
        float red = 0.f;

        // SEQLEN == 128.
        if (Cta_tile::WARPS_M == 2 && Cta_tile::WARPS_N == 2)
        {
            red = reduce_2x2<Functor>();

            // SEQLEN == 256.
        }
        else if (Cta_tile::WARPS_M == 1 && Cta_tile::WARPS_N == 4)
        {
            red = reduce_1x4<Functor>();

            // SEQLEN == 384.
        }
        else if (Cta_tile::WARPS_M == 1 && Cta_tile::WARPS_N == 8)
        {
            red = reduce_1x8<Functor>();

            // Not supported.
        }
        else
        {
            assert(false);
        }

        return red;
    }

    // Finalize the reduction.
    inline __device__ void shuffle(float (&dst)[MMAS_M * 2], float red)
    {

        // Store the value back to shared memory.
        if (tidx_ < Cta_tile::M)
        {
            smem_[tidx_] = red;
        }

        // Make sure the data is in shared memory.
        __syncthreads();

// Finally read the values.
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
            dst[2 * mi + 0] = smem_read_[mi * Mma_tile::M_PER_MMA_PER_CTA + 0];
            dst[2 * mi + 1] = smem_read_[mi * Mma_tile::M_PER_MMA_PER_CTA + 8];
        }

        // Make sure the data is in shared memory.
        __syncthreads();
    }

    // Do a CTA-wide reduction.
    template <typename Functor>
    inline __device__ void reduce(float (&dst)[MMAS_M * 2])
    {

        // NOTE: 1 warp along reduce direction, no syncs
        if (Cta_tile::WARPS_N == 1)
        {
            reduce_Nx1<Functor>(dst);
        }
        else
        {
            // The result of the reduction. Threads 0..Cta_tile::M-1 own a single row value.
            float red = reduce_<Functor>();

            // Make sure we can write to shared memory.
            __syncthreads();

            // Finalize the reduction.
            shuffle(dst, red);
        }
    }

    // Scale all the elements.
    inline __device__ void scale(float const (&sum)[MMAS_M * 2])
    {
        // Precompute the inverse sum to normalize. Without -use_fast_math, it makes a huge deal.
        float inv_sum[MMAS_M * 2];
#pragma unroll
        for (int mi = 0; mi < MMAS_M * 2; ++mi)
        {
            inv_sum[mi] = (sum[mi] == 0.f || sum[mi] != sum[mi]) ? 1.f : 1.f / sum[mi];
        }

// Update the values.
#pragma unroll
        for (int mi = 0; mi < MMAS_M * 2; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N * 4; ++ni)
            {
                elt_[mi][ni] *= inv_sum[mi];
            }
        }
    }

    // Shared memory for the CTA-wide reduction.
    float *smem_, *smem_write_, *smem_read_;
    // The current thread index.
    int tidx_;
    // The elements.
    float elt_[MMAS_M * 2][MMAS_N * 4];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, typename Kernel_traits>
struct Softmax_hmma : public Softmax_base<Traits, Cta_tile, Kernel_traits>
{

    // The base class.
    using Base = Softmax_base<Traits, Cta_tile, Kernel_traits>;

    // The MMAs.
    enum
    {
        MMAS_M = Base::MMAS_M
    };

    enum
    {
        MMAS_N = Base::MMAS_N
    };

    // Whether we need to skip the softmax due to the sliding-window attention
    // Otherwise, we will get NANs as those tokens are all masked out.
    enum
    {
        SLIDING_WINDOW_ATTENTION = Kernel_traits::SLIDING_WINDOW_ATTENTION
    };

    // Use BMM1 softcapping scale or not.
    enum
    {
        ENABLE_BMM1_SOFTCAPPING_SCALE = Kernel_traits::ENABLE_BMM1_SOFTCAPPING_SCALE
    };

    // The accumulators.
    using Accumulator = fmha::Fragment_accumulator<Traits>;

    // Softmax dst data_type (BMM2 input)
    using Dst_type = typename Traits::A_type;

    // Ctor.
    template <typename Params>
    inline __device__ Softmax_hmma(Params const& params, void* smem, int bidb, int tidx)
        : Base(params, smem, bidb, tidx)
        , params_scale_bmm1_(params.scale_bmm1)
        , params_softcapping_scale_bmm1_(params.softcapping_scale_bmm1)
    {
    }

    // Store the tile after softmax.
    template <typename Gmem_tile>
    inline __device__ void store(Gmem_tile& gmem_tile)
    {
        Accumulator acc[MMAS_M][MMAS_N];
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {

                // The elements.
                float tmp_00 = this->elt_[2 * mi + 0][4 * ni + 0];
                float tmp_01 = this->elt_[2 * mi + 0][4 * ni + 1];
                float tmp_02 = this->elt_[2 * mi + 0][4 * ni + 2];
                float tmp_03 = this->elt_[2 * mi + 0][4 * ni + 3];
                float tmp_10 = this->elt_[2 * mi + 1][4 * ni + 0];
                float tmp_11 = this->elt_[2 * mi + 1][4 * ni + 1];
                float tmp_12 = this->elt_[2 * mi + 1][4 * ni + 2];
                float tmp_13 = this->elt_[2 * mi + 1][4 * ni + 3];

                // Transform to accumulators.
                acc[mi][ni].reg(0) = fmha::float2_to_16bit_2<Dst_type>(tmp_00, tmp_01);
                acc[mi][ni].reg(1) = fmha::float2_to_16bit_2<Dst_type>(tmp_10, tmp_11);
                acc[mi][ni].reg(2) = fmha::float2_to_16bit_2<Dst_type>(tmp_02, tmp_03);
                acc[mi][ni].reg(3) = fmha::float2_to_16bit_2<Dst_type>(tmp_12, tmp_13);
            }
        }

        // Delegate to the gmem tile to store.
        gmem_tile.store(acc);
    }

    // Convert from FP16 fragments to floats.
    inline __device__ void unpack(Accumulator const (&acc)[MMAS_M][MMAS_N])
    {
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {

                // Normalize the values, and clamp to finite half.
                uint32_t acc_0 = satfinite_h2(hmul2(acc[mi][ni].reg(0), params_scale_bmm1_));
                uint32_t acc_1 = satfinite_h2(hmul2(acc[mi][ni].reg(1), params_scale_bmm1_));
                uint32_t acc_2 = satfinite_h2(hmul2(acc[mi][ni].reg(2), params_scale_bmm1_));
                uint32_t acc_3 = satfinite_h2(hmul2(acc[mi][ni].reg(3), params_scale_bmm1_));

                // Extract the values as floats.
                half2_to_float2(this->elt_[2 * mi + 0][4 * ni + 0], this->elt_[2 * mi + 0][4 * ni + 1], acc_0);
                half2_to_float2(this->elt_[2 * mi + 1][4 * ni + 0], this->elt_[2 * mi + 1][4 * ni + 1], acc_1);
                half2_to_float2(this->elt_[2 * mi + 0][4 * ni + 2], this->elt_[2 * mi + 0][4 * ni + 3], acc_2);
                half2_to_float2(this->elt_[2 * mi + 1][4 * ni + 2], this->elt_[2 * mi + 1][4 * ni + 3], acc_3);

                // Attention logit softcapping scale.
                // 1.0f / softcapping_scale has been fused to scale_bmm1.
                if constexpr (ENABLE_BMM1_SOFTCAPPING_SCALE)
                {
                    this->elt_[2 * mi + 0][4 * ni + 0]
                        = params_softcapping_scale_bmm1_ * __tanhf(this->elt_[2 * mi + 0][4 * ni + 0]);
                    this->elt_[2 * mi + 0][4 * ni + 1]
                        = params_softcapping_scale_bmm1_ * __tanhf(this->elt_[2 * mi + 0][4 * ni + 1]);
                    this->elt_[2 * mi + 1][4 * ni + 0]
                        = params_softcapping_scale_bmm1_ * __tanhf(this->elt_[2 * mi + 1][4 * ni + 0]);
                    this->elt_[2 * mi + 1][4 * ni + 1]
                        = params_softcapping_scale_bmm1_ * __tanhf(this->elt_[2 * mi + 1][4 * ni + 1]);
                    this->elt_[2 * mi + 0][4 * ni + 2]
                        = params_softcapping_scale_bmm1_ * __tanhf(this->elt_[2 * mi + 0][4 * ni + 2]);
                    this->elt_[2 * mi + 0][4 * ni + 3]
                        = params_softcapping_scale_bmm1_ * __tanhf(this->elt_[2 * mi + 0][4 * ni + 3]);
                    this->elt_[2 * mi + 1][4 * ni + 2]
                        = params_softcapping_scale_bmm1_ * __tanhf(this->elt_[2 * mi + 1][4 * ni + 2]);
                    this->elt_[2 * mi + 1][4 * ni + 3]
                        = params_softcapping_scale_bmm1_ * __tanhf(this->elt_[2 * mi + 1][4 * ni + 3]);
                }
            }
        }
    }

    // Apply the exp to all the elements.
    // Need to make sure the results are zero when all elts are -FLT_MAX
    //  as it is possible that all tokens are masked out.
    template <bool APPLY_MASK = false>
    inline __device__ void apply_exp_with_mask(float const (&max)[MMAS_M * 2])
    {
#pragma unroll
        for (int mi = 0; mi < MMAS_M * 2; ++mi)
        {
            float max_val = APPLY_MASK && max[mi] == -FLT_MAX ? 0.f : max[mi];
#pragma unroll
            for (int ni = 0; ni < MMAS_N * 4; ++ni)
            {
                this->elt_[mi][ni] = expf(this->elt_[mi][ni] - max_val);
            }
        }
    }

    // The scaling factor.
    uint32_t const params_scale_bmm1_;
    float const params_softcapping_scale_bmm1_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits>
struct Fragment_helper
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Fragment_helper<fmha::Volta_imma_int8_int32_traits>
{

    // The traits.
    using Traits = fmha::Volta_imma_int8_int32_traits;
    // The fragment A.
    using Fragment_a = fmha::Fragment_a<Traits, fmha::Row>;
    // The accumulator.
    using Accumulator = fmha::Fragment_accumulator<Traits>;

    // Load a 2x4 array from registers.
    static inline __device__ void load(int32_t (&dst)[2][4], Accumulator const& src)
    {
        dst[0][0] = src.elt(0);
        dst[0][1] = src.elt(1);
        dst[0][2] = src.elt(2);
        dst[0][3] = src.elt(3);
        dst[1][0] = src.elt(4);
        dst[1][1] = src.elt(5);
        dst[1][2] = src.elt(6);
        dst[1][3] = src.elt(7);
    }

    // Store to an accumulator.
    static inline __device__ void store(Accumulator& dst, uint32_t const (&src)[2][4])
    {
        dst.reg(0) = src[0][0];
        dst.reg(1) = src[0][1];
        dst.reg(2) = src[0][2];
        dst.reg(3) = src[0][3];
        dst.reg(4) = src[1][0];
        dst.reg(5) = src[1][1];
        dst.reg(6) = src[1][2];
        dst.reg(7) = src[1][3];
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Fragment_helper<fmha::Turing_imma_int8_int32_traits>
{

    // The traits.
    using Traits = fmha::Turing_imma_int8_int32_traits;
    // The fragment A.
    using Fragment_a = fmha::Fragment_a<Traits, fmha::Row>;
    // The accumulator.
    using Accumulator = fmha::Fragment_accumulator<Traits>;

    // Load a 2x4 array from registers.
    static inline __device__ void load(int32_t (&dst)[2][4], Accumulator const& src)
    {
        dst[0][0] = src.elt(0);
        dst[0][1] = src.elt(1);
        dst[0][2] = src.elt(2);
        dst[0][3] = src.elt(3);
        dst[1][0] = src.elt(4);
        dst[1][1] = src.elt(5);
        dst[1][2] = src.elt(6);
        dst[1][3] = src.elt(7);
    }

    // Store to an accumulator.
    static inline __device__ void store(Accumulator& dst, uint32_t const (&src)[2][4])
    {
        dst.reg(0) = src[0][0];
        dst.reg(1) = src[0][1];
        dst.reg(2) = src[0][2];
        dst.reg(3) = src[0][3];
        dst.reg(4) = src[1][0];
        dst.reg(5) = src[1][1];
        dst.reg(6) = src[1][2];
        dst.reg(7) = src[1][3];
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Fragment_helper<fmha::Ampere_imma_int8_int32_traits>
{

    // The traits.
    using Traits = fmha::Ampere_imma_int8_int32_traits;
    // The fragment A.
    using Fragment_a = fmha::Fragment_a<Traits, fmha::Row>;
    // The accumulator.
    using Accumulator = fmha::Fragment_accumulator<Traits>;

    // Load a 2x4 array from registers.
    static inline __device__ void load(int32_t (&dst)[2][4], Accumulator const& src)
    {
        dst[0][0] = src.elt(0);
        dst[0][1] = src.elt(1);
        dst[0][2] = src.elt(4);
        dst[0][3] = src.elt(5);
        dst[1][0] = src.elt(2);
        dst[1][1] = src.elt(3);
        dst[1][2] = src.elt(6);
        dst[1][3] = src.elt(7);
    }

    // Store to an accumulator.
    static inline __device__ void store(Accumulator& dst, uint32_t const (&src)[2][4])
    {
        dst.reg(0) = src[0][0];
        dst.reg(1) = src[0][1];
        dst.reg(4) = src[0][2];
        dst.reg(5) = src[0][3];
        dst.reg(2) = src[1][0];
        dst.reg(3) = src[1][1];
        dst.reg(6) = src[1][2];
        dst.reg(7) = src[1][3];
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, typename Kernel_traits>
struct Softmax_imma : public Softmax_base<Traits, Cta_tile, Kernel_traits>
{

    // The base class.
    using Base = Softmax_base<Traits, Cta_tile, Kernel_traits>;
    // The Mma tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    // The MMAs.
    enum
    {
        MMAS_M = Base::MMAS_M
    };

    enum
    {
        MMAS_N = Base::MMAS_N
    };

    // The accumulators.
    using Accumulator = fmha::Fragment_accumulator<Traits>;
    // The fragment.
    using Fragment_a = fmha::Fragment_a<Traits, fmha::Row>;

    // The dst type
    using Dst_type = typename Traits::A_type;

    // Ctor.
    template <typename Params>
    inline __device__ Softmax_imma(Params const& params, void* smem, int bidb, int tidx)
        : Base(params, smem, bidb, tidx)
        , params_scale_bmm1_(params.scale_bmm1)
        , params_scale_softmax_(params.scale_softmax)
    {
    }

    // Store the tile after softmax.
    template <typename Gmem_tile>
    inline __device__ void store(Gmem_tile& gmem_tile)
    {

        float const scale = reinterpret_cast<float const&>(params_scale_softmax_);
        Accumulator acc[MMAS_M][MMAS_N];
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {

                // Scale the FP32 elements.
                uint32_t tmp[2][4];
#pragma unroll
                for (int mj = 0; mj < 2; ++mj)
                {
#pragma unroll
                    for (int nj = 0; nj < 4; ++nj)
                    {
                        float f = this->elt_[2 * mi + mj][4 * ni + nj] * scale;
                        asm volatile("cvt.rni.sat.s8.f32 %0, %1;\n" : "=r"(tmp[mj][nj]) : "f"(f));
                    }
                }

                // Convert to int8 and store.
                Fragment_helper<Traits>::store(acc[mi][ni], tmp);
            }
        }

        // Delegate to the gmem tile to store.
        gmem_tile.store(acc);
    }

    // Convert from accumulators to floats.
    inline __device__ void unpack(Accumulator const (&acc)[MMAS_M][MMAS_N])
    {
        float const scale = reinterpret_cast<float const&>(params_scale_bmm1_);
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {

                // Load the values from the accumulator's registers.
                int32_t tmp[2][4];
                Fragment_helper<Traits>::load(tmp, acc[mi][ni]);

// Convert to FP32 and scale.
#pragma unroll
                for (int mj = 0; mj < 2; ++mj)
                {
#pragma unroll
                    for (int nj = 0; nj < 4; ++nj)
                    {
#if defined(USE_I2F_EMULATION_TRICK)
                        float f = reinterpret_cast<float const&>(tmp[mj][nj]);
                        this->elt_[2 * mi + mj][4 * ni + nj] = (f - FP32_I2F_MAGIC_NUMBER) * scale;
#else
                        this->elt_[2 * mi + mj][4 * ni + nj] = static_cast<float>(tmp[mj][nj]) * scale;
#endif // defined(USE_I2F_EMULATION_TRICK)
                    }
                }
            }
        }
    }

    // Repack. We could use store/load to match the Smem_tile API. (shared by Ampere IMMA and Ada QMMA)
    template <int K, int M, typename Fragment_a_>
    inline __device__ void pack(Fragment_a_ (&dst)[K][M])
    {
        // We pack N 16x16 acc tiles into K 16x32 tiles for A.
        // In the 16x16 tile, a thread owns 4 elts per row (4 regs).
        // In the 16x32 A tile, a thread owns 8 elts per row (2 regs).
        // Hence we have to pack with a 2:1 ratio.
        // For N = 1, K is 1: pack 4 values into dst reg 0. Set reg 1 to 0.
        // For N = 2, K is 1: pack 8 values into dst regs 0, 1.
        // For N = 3, K is 2: pack 12 values into dst regs (0,0), (0,1), (1,0). Set (1,1) to 0.
        // For N = 4, K is 2: pack 16 values into dst regs (0,0), (0,1), (1,0), (1,1)
        // For N = 5, K is 3: pack 20 values into dst regs (0,0), (0,1), (1,0), (1,1), (2,0). Set (2,1) to 0.
        // For N = 6, K is 3: pack 24 values into dst regs (0,0), (0,1), (1,0), (1,1), (2,0), (2,1)

        static_assert(K == 3 || K == 2 || K == 1, "");

        float const scale = reinterpret_cast<float const&>(this->params_scale_softmax_);

#pragma unroll
        for (int mi = 0; mi < M; ++mi)
        {

            // 1st row - 12 elements per row.
            float tmp_00 = this->elt_[2 * mi + 0][0] * scale;
            float tmp_01 = this->elt_[2 * mi + 0][1] * scale;
            float tmp_02 = this->elt_[2 * mi + 0][2] * scale;
            float tmp_03 = this->elt_[2 * mi + 0][3] * scale;
            float tmp_04 = this->elt_[2 * mi + 0][4] * scale;
            float tmp_05 = this->elt_[2 * mi + 0][5] * scale;
            float tmp_06 = this->elt_[2 * mi + 0][6] * scale;
            float tmp_07 = this->elt_[2 * mi + 0][7] * scale;
            float tmp_08 = this->elt_[2 * mi + 0][8] * scale;
            float tmp_09 = this->elt_[2 * mi + 0][9] * scale;
            float tmp_0a = this->elt_[2 * mi + 0][10] * scale;
            float tmp_0b = this->elt_[2 * mi + 0][11] * scale;

            // 2nd row - 12 elements per row.
            float tmp_20 = this->elt_[2 * mi + 1][0] * scale;
            float tmp_21 = this->elt_[2 * mi + 1][1] * scale;
            float tmp_22 = this->elt_[2 * mi + 1][2] * scale;
            float tmp_23 = this->elt_[2 * mi + 1][3] * scale;
            float tmp_24 = this->elt_[2 * mi + 1][4] * scale;
            float tmp_25 = this->elt_[2 * mi + 1][5] * scale;
            float tmp_26 = this->elt_[2 * mi + 1][6] * scale;
            float tmp_27 = this->elt_[2 * mi + 1][7] * scale;
            float tmp_28 = this->elt_[2 * mi + 1][8] * scale;
            float tmp_29 = this->elt_[2 * mi + 1][9] * scale;
            float tmp_2a = this->elt_[2 * mi + 1][10] * scale;
            float tmp_2b = this->elt_[2 * mi + 1][11] * scale;

            // Pack the first 12 elements to 6 registers of 2 fragments.
            dst[0][mi].reg(0) = fmha::float4_to_8bitx4<Dst_type>(tmp_00, tmp_01, tmp_02, tmp_03);
            dst[0][mi].reg(1) = fmha::float4_to_8bitx4<Dst_type>(tmp_20, tmp_21, tmp_22, tmp_23);
            dst[0][mi].reg(2) = fmha::float4_to_8bitx4<Dst_type>(tmp_04, tmp_05, tmp_06, tmp_07);
            dst[0][mi].reg(3) = fmha::float4_to_8bitx4<Dst_type>(tmp_24, tmp_25, tmp_26, tmp_27);
            if (K > 1)
            {
                dst[1][mi].reg(0) = fmha::float4_to_8bitx4<Dst_type>(tmp_08, tmp_09, tmp_0a, tmp_0b);
                dst[1][mi].reg(1) = fmha::float4_to_8bitx4<Dst_type>(tmp_28, tmp_29, tmp_2a, tmp_2b);
            }

            if (Mma_tile::MMAS_N == 6)
            {
                float tmp_0c = this->elt_[2 * mi + 0][12] * scale;
                float tmp_0d = this->elt_[2 * mi + 0][13] * scale;
                float tmp_0e = this->elt_[2 * mi + 0][14] * scale;
                float tmp_0f = this->elt_[2 * mi + 0][15] * scale;
                float tmp_10 = this->elt_[2 * mi + 0][16] * scale;
                float tmp_11 = this->elt_[2 * mi + 0][17] * scale;
                float tmp_12 = this->elt_[2 * mi + 0][18] * scale;
                float tmp_13 = this->elt_[2 * mi + 0][19] * scale;
                float tmp_14 = this->elt_[2 * mi + 0][20] * scale;
                float tmp_15 = this->elt_[2 * mi + 0][21] * scale;
                float tmp_16 = this->elt_[2 * mi + 0][22] * scale;
                float tmp_17 = this->elt_[2 * mi + 0][23] * scale;

                float tmp_2c = this->elt_[2 * mi + 1][12] * scale;
                float tmp_2d = this->elt_[2 * mi + 1][13] * scale;
                float tmp_2e = this->elt_[2 * mi + 1][14] * scale;
                float tmp_2f = this->elt_[2 * mi + 1][15] * scale;
                float tmp_30 = this->elt_[2 * mi + 1][16] * scale;
                float tmp_31 = this->elt_[2 * mi + 1][17] * scale;
                float tmp_32 = this->elt_[2 * mi + 1][18] * scale;
                float tmp_33 = this->elt_[2 * mi + 1][19] * scale;
                float tmp_34 = this->elt_[2 * mi + 1][20] * scale;
                float tmp_35 = this->elt_[2 * mi + 1][21] * scale;
                float tmp_36 = this->elt_[2 * mi + 1][22] * scale;
                float tmp_37 = this->elt_[2 * mi + 1][23] * scale;

                dst[1][mi].reg(2) = fmha::float4_to_8bitx4<Dst_type>(tmp_0c, tmp_0d, tmp_0e, tmp_0f);
                dst[1][mi].reg(3) = fmha::float4_to_8bitx4<Dst_type>(tmp_2c, tmp_2d, tmp_2e, tmp_2f);
                dst[2][mi].reg(0) = fmha::float4_to_8bitx4<Dst_type>(tmp_10, tmp_11, tmp_12, tmp_13);
                dst[2][mi].reg(1) = fmha::float4_to_8bitx4<Dst_type>(tmp_30, tmp_31, tmp_32, tmp_33);
                dst[2][mi].reg(2) = fmha::float4_to_8bitx4<Dst_type>(tmp_14, tmp_15, tmp_16, tmp_17);
                dst[2][mi].reg(3) = fmha::float4_to_8bitx4<Dst_type>(tmp_34, tmp_35, tmp_36, tmp_37);
            }
            else if (Mma_tile::MMAS_N == 4)
            {
                // SEQLEN == 128.
                float tmp_0c = this->elt_[2 * mi + 0][12] * scale;
                float tmp_0d = this->elt_[2 * mi + 0][13] * scale;
                float tmp_0e = this->elt_[2 * mi + 0][14] * scale;
                float tmp_0f = this->elt_[2 * mi + 0][15] * scale;

                float tmp_1c = this->elt_[2 * mi + 1][12] * scale;
                float tmp_1d = this->elt_[2 * mi + 1][13] * scale;
                float tmp_1e = this->elt_[2 * mi + 1][14] * scale;
                float tmp_1f = this->elt_[2 * mi + 1][15] * scale;

                dst[1][mi].reg(2) = fmha::float4_to_8bitx4<Dst_type>(tmp_0c, tmp_0d, tmp_0e, tmp_0f);
                dst[1][mi].reg(3) = fmha::float4_to_8bitx4<Dst_type>(tmp_1c, tmp_1d, tmp_1e, tmp_1f);

                // SEQLEN == 384 or SEQLEN == 256.
            }
            else if (Mma_tile::MMAS_N == 3 || Mma_tile::MMAS_N == 2)
            {

                // TODO added second OR term for ampere imma s=256: correct?
                dst[1][mi].reg(2) = 0u;
                dst[1][mi].reg(3) = 0u;
            }
            else if (Mma_tile::MMAS_N == 1)
            {

                dst[0][mi].reg(2) = 0u;
                dst[0][mi].reg(3) = 0u;

                // Not implemented.
            }
            else
            {
                assert(false);
            }
        }
    }

    // The scaling factors.
    uint32_t const params_scale_bmm1_, params_scale_softmax_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, typename Kernel_traits>
struct Softmax_qmma : public Softmax_imma<Traits, Cta_tile, Kernel_traits>
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, typename Kernel_traits>
struct Softmax_qmma<fmha::Ada_qmma_e4m3_fp32_traits, Cta_tile, Kernel_traits>
    : public Softmax_imma<fmha::Ada_qmma_e4m3_fp32_traits, Cta_tile, Kernel_traits>
{

    // The Traits
    using Traits = fmha::Ada_qmma_e4m3_fp32_traits;
    // The base class.
    using Base = Softmax_imma<Traits, Cta_tile, Kernel_traits>;

    // The MMAs.
    enum
    {
        MMAS_M = Base::MMAS_M
    };

    enum
    {
        MMAS_N = Base::MMAS_N
    };

    // The accumulators.
    using Accumulator = fmha::Fragment_accumulator<Traits>;

    // Ctor.
    template <typename Params>
    inline __device__ Softmax_qmma(Params const& params, void* smem, int bidb, int tidx)
        : Base(params, smem, bidb, tidx)
        , params_scale_bmm1_(params.scale_bmm1_d ? *params.scale_bmm1_d : params.scale_bmm1)
        , params_scale_softmax_(params.scale_softmax)
    {
    }

    // Store the tile after softmax.
    template <typename Gmem_tile>
    inline __device__ void store(Gmem_tile& gmem_tile)
    {

        float const scale = reinterpret_cast<float const&>(params_scale_softmax_);
        Accumulator acc[MMAS_M][MMAS_N];
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {

                // scale
                acc[mi][ni].ele(0) = this->elt_[2 * mi + 0][4 * ni + 0] * scale;
                acc[mi][ni].ele(1) = this->elt_[2 * mi + 0][4 * ni + 1] * scale;
                acc[mi][ni].ele(4) = this->elt_[2 * mi + 0][4 * ni + 2] * scale;
                acc[mi][ni].ele(5) = this->elt_[2 * mi + 0][4 * ni + 3] * scale;
                acc[mi][ni].ele(2) = this->elt_[2 * mi + 1][4 * ni + 0] * scale;
                acc[mi][ni].ele(3) = this->elt_[2 * mi + 1][4 * ni + 1] * scale;
                acc[mi][ni].ele(6) = this->elt_[2 * mi + 1][4 * ni + 2] * scale;
                acc[mi][ni].ele(7) = this->elt_[2 * mi + 1][4 * ni + 3] * scale;
            }
        }

        // Delegate to the gmem tile to store.
        // TODO: need fp32 to fp8 conversion (move this to gmem_tile)
        gmem_tile.store(acc);
    }

    // Convert from accumulators to floats.
    inline __device__ void unpack(Accumulator const (&acc)[MMAS_M][MMAS_N])
    {
        float const scale = reinterpret_cast<float const&>(params_scale_bmm1_);
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {

                // Convert to FP32 and scale.
                this->elt_[2 * mi + 0][4 * ni + 0] = acc[mi][ni].elt(0) * scale;
                this->elt_[2 * mi + 0][4 * ni + 1] = acc[mi][ni].elt(1) * scale;
                this->elt_[2 * mi + 0][4 * ni + 2] = acc[mi][ni].elt(4) * scale;
                this->elt_[2 * mi + 0][4 * ni + 3] = acc[mi][ni].elt(5) * scale;
                this->elt_[2 * mi + 1][4 * ni + 0] = acc[mi][ni].elt(2) * scale;
                this->elt_[2 * mi + 1][4 * ni + 1] = acc[mi][ni].elt(3) * scale;
                this->elt_[2 * mi + 1][4 * ni + 2] = acc[mi][ni].elt(6) * scale;
                this->elt_[2 * mi + 1][4 * ni + 3] = acc[mi][ni].elt(7) * scale;
            }
        }
    }

    template <bool APPLY_MASK = false>
    inline __device__ void apply_exp_with_mask(float const (&max)[MMAS_M * 2])
    {
#pragma unroll
        for (int mi = 0; mi < MMAS_M * 2; ++mi)
        {
            float max_val = APPLY_MASK && max[mi] == -FLT_MAX ? 0.f : (max[mi] - logf(Traits::SOFTMAX_FP_QUANT_SCALE));
#pragma unroll
            for (int ni = 0; ni < MMAS_N * 4; ++ni)
            {
                this->elt_[mi][ni] = expf(this->elt_[mi][ni] - max_val);
            }
        }
    }

    // Pack the data to a fragment for the next GEMM.
    template <typename Fragment_a, int K, int M>
    inline __device__ void pack(Fragment_a (&dst)[K][M]) const
    {
        float const scale = reinterpret_cast<float const&>(this->params_scale_softmax_);

// The canonical layout in K should be R0: [0,1,2,3] R2: [16,17,18,19]
// Note below that this is not possible with the register layout of the accumulator.
#pragma unroll
        for (int mi = 0; mi < M; ++mi)
        {
#pragma unroll
            for (int ki = 0; ki < K; ++ki)
            {
                // 1st row - 8 elements per row.
                float tmp_00 = this->elt_[2 * mi + 0][8 * ki + 0] * scale; // + 0
                float tmp_01 = this->elt_[2 * mi + 0][8 * ki + 1] * scale; // + 1
                float tmp_02 = this->elt_[2 * mi + 0][8 * ki + 2] * scale; // + 8
                float tmp_03 = this->elt_[2 * mi + 0][8 * ki + 3] * scale; // + 9
                float tmp_04 = this->elt_[2 * mi + 0][8 * ki + 4] * scale; // +16
                float tmp_05 = this->elt_[2 * mi + 0][8 * ki + 5] * scale; // +17
                float tmp_06 = this->elt_[2 * mi + 0][8 * ki + 6] * scale; // +24
                float tmp_07 = this->elt_[2 * mi + 0][8 * ki + 7] * scale; // +25

                // 2nd row - 4 elements per row.
                float tmp_10 = this->elt_[2 * mi + 1][8 * ki + 0] * scale; // + 0
                float tmp_11 = this->elt_[2 * mi + 1][8 * ki + 1] * scale; // + 1
                float tmp_12 = this->elt_[2 * mi + 1][8 * ki + 2] * scale; // + 8
                float tmp_13 = this->elt_[2 * mi + 1][8 * ki + 3] * scale; // + 9
                float tmp_14 = this->elt_[2 * mi + 1][8 * ki + 4] * scale; // +16
                float tmp_15 = this->elt_[2 * mi + 1][8 * ki + 5] * scale; // +17
                float tmp_16 = this->elt_[2 * mi + 1][8 * ki + 6] * scale; // +24
                float tmp_17 = this->elt_[2 * mi + 1][8 * ki + 7] * scale; // +25

                // Pack to 4 registers.
                dst[ki][mi].reg(0) = fmha::float4_to_fp8x4<Traits::A_type>(tmp_00, tmp_01, tmp_02, tmp_03);
                dst[ki][mi].reg(1) = fmha::float4_to_fp8x4<Traits::A_type>(tmp_10, tmp_11, tmp_12, tmp_13);
                dst[ki][mi].reg(2) = fmha::float4_to_fp8x4<Traits::A_type>(tmp_04, tmp_05, tmp_06, tmp_07);
                dst[ki][mi].reg(3) = fmha::float4_to_fp8x4<Traits::A_type>(tmp_14, tmp_15, tmp_16, tmp_17);
            }
        }
    }

    // The scaling factors.
    uint32_t const params_scale_bmm1_, params_scale_softmax_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, typename Kernel_traits>
struct Softmax_qmma<fmha::Ada_qmma_e4m3_fp16_traits, Cta_tile, Kernel_traits>
    : public Softmax_imma<fmha::Ada_qmma_e4m3_fp16_traits, Cta_tile, Kernel_traits>
{

    // The Traits
    using Traits = fmha::Ada_qmma_e4m3_fp16_traits;
    // The base class.
    using Base = Softmax_imma<Traits, Cta_tile, Kernel_traits>;

    // The MMAs.
    enum
    {
        MMAS_M = Base::MMAS_M
    };

    enum
    {
        MMAS_N = Base::MMAS_N
    };

    // The accumulators.
    using Accumulator = fmha::Fragment_accumulator<Traits>;

    // Ctor.
    template <typename Params>
    inline __device__ Softmax_qmma(Params const& params, void* smem, int bidb, int tidx)
        : Base(params, smem, bidb, tidx)
        , params_scale_bmm1_(params.scale_bmm1)
        , params_scale_softmax_(params.scale_softmax)
    {
    }

    // Store the tile after softmax.
    template <typename Gmem_tile>
    inline __device__ void store(Gmem_tile& gmem_tile)
    {

        float const scale = reinterpret_cast<float const&>(params_scale_softmax_);
        Accumulator acc[MMAS_M][MMAS_N];
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {

                // scale
                acc[mi][ni].ele(0) = this->elt_[2 * mi + 0][4 * ni + 0] * scale;
                acc[mi][ni].ele(1) = this->elt_[2 * mi + 0][4 * ni + 1] * scale;
                acc[mi][ni].ele(4) = this->elt_[2 * mi + 0][4 * ni + 2] * scale;
                acc[mi][ni].ele(5) = this->elt_[2 * mi + 0][4 * ni + 3] * scale;
                acc[mi][ni].ele(2) = this->elt_[2 * mi + 1][4 * ni + 0] * scale;
                acc[mi][ni].ele(3) = this->elt_[2 * mi + 1][4 * ni + 1] * scale;
                acc[mi][ni].ele(6) = this->elt_[2 * mi + 1][4 * ni + 2] * scale;
                acc[mi][ni].ele(7) = this->elt_[2 * mi + 1][4 * ni + 3] * scale;
            }
        }

        // Delegate to the gmem tile to store.
        // TODO: need fp32 to fp8 conversion (move this to gmem_tile)
        gmem_tile.store(acc);
    }

    // Convert from accumulators to floats.
    inline __device__ void unpack(Accumulator const (&acc)[MMAS_M][MMAS_N])
    {
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {

                // Convert to FP32 and scale.
                float2* elt_ptr0 = reinterpret_cast<float2*>(this->elt_[2 * mi + 0] + 4 * ni);
                float2* elt_ptr1 = reinterpret_cast<float2*>(this->elt_[2 * mi + 1] + 4 * ni);
                elt_ptr0[0] = fmha::half2_to_float2(fmha::hmul2(acc[mi][ni].reg(0), params_scale_bmm1_));
                elt_ptr0[1] = fmha::half2_to_float2(fmha::hmul2(acc[mi][ni].reg(2), params_scale_bmm1_));
                elt_ptr1[0] = fmha::half2_to_float2(fmha::hmul2(acc[mi][ni].reg(1), params_scale_bmm1_));
                elt_ptr1[1] = fmha::half2_to_float2(fmha::hmul2(acc[mi][ni].reg(3), params_scale_bmm1_));
            }
        }
    }

    // The scaling factors.
    uint32_t const params_scale_bmm1_, params_scale_softmax_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, typename Kernel_traits, bool Sage = false>
struct Softmax
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, typename Kernel_traits>
struct Softmax<fmha::Volta_hmma_fp16_traits, Cta_tile, Kernel_traits>
{

    // The traits class.
    using Traits = fmha::Volta_hmma_fp16_traits;
    // The Mma tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;
    // The accumulators.
    using Accumulator = fmha::Fragment_accumulator<Traits>;
    // The fragment.
    using Fragment_a = fmha::Fragment_a<Traits, fmha::Row>;

    // Softmax dst data_type (BMM2 input)
    using Dst_type = typename Traits::A_type;

    // The number of MMAs in M/N dimensions.
    enum
    {
        MMAS_M = Mma_tile::MMAS_M
    };

    enum
    {
        MMAS_N = Mma_tile::MMAS_N
    };

    // The number of groups of warp such that we have at most 2 warps writing consecutive elements.
    enum
    {
        GROUPS = fmha::Div_up<Cta_tile::WARPS_N, 2>::VALUE
    };

    // The number of elements that we are going to store per row.
    enum
    {
        ELEMENTS_PER_ROW = Cta_tile::WARPS_N / GROUPS
    };

    // The number of rows.
    enum
    {
        ROWS = Cta_tile::M * GROUPS
    };

    // The total number of elements.
    enum
    {
        ELEMENTS = ROWS * ELEMENTS_PER_ROW
    };

    // Use BMM1 softcapping scale or not.
    enum
    {
        ENABLE_BMM1_SOFTCAPPING_SCALE = Kernel_traits::ENABLE_BMM1_SOFTCAPPING_SCALE
    };

    // If shared memory is used
    enum
    {
        USE_SHARED_MEMORY = Cta_tile::WARPS_N > 1
    };

    // The number of rows per thread.
    enum
    {
        ROWS_PER_THREAD = MMAS_M
    };

    // DEBUG.
    static_assert(ELEMENTS == Cta_tile::M * Cta_tile::WARPS_N, "");

    // END OF DEBUG.

    // Ctor.
    template <typename Params>
    inline __device__ Softmax(Params const& params, void* smem, int bidb, int tidx)
        : params_scale_bmm1_(params.scale_bmm1)
        , params_softcapping_scale_bmm1_(params.softcapping_scale_bmm1)
        , smem_(reinterpret_cast<float*>(smem))
        , tidx_(tidx)
    {

        // Extract the position in the warp.
        int warp = tidx / Cta_tile::THREADS_PER_WARP;
        int lane = tidx % Cta_tile::THREADS_PER_WARP;

        // Decompose the warp index into M and N.
        int warp_m = warp % Cta_tile::WARPS_M;
        int warp_n = warp / Cta_tile::WARPS_M;

        // Decompose the warp-n index into group/position-inside-the-group.
        int warp_g = warp_n / ELEMENTS_PER_ROW;
        int warp_i = warp_n % ELEMENTS_PER_ROW;

        // The row written/read by the thread (threads i and i+8 are on the same row).
        int row = (lane & 0x10) / 2 + (lane & 0x07);

        // The location written by the threads.
        int write_row = warp_g * Cta_tile::M + warp_m * Mma_tile::M_PER_MMA + row;
        int write_col = warp_i;

        // Assemble the write pointer.
        smem_write_ = &smem_[write_row * ELEMENTS_PER_ROW + write_col];
        // Assemble the read pointer.
        smem_read_ = &smem_[warp_m * Mma_tile::M_PER_MMA + row];
    }

    // Apply mask before softmax. Use 1 byte per MMA distributed as 1x8.
    template <typename Mask>
    inline __device__ void apply_mask(Mask const& mask)
    {
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {
#pragma unroll
                for (int ii = 0; ii < 8; ++ii)
                {
                    if (!mask.is_valid(mi, ni, 0, ii))
                    {
                        elt_[mi][8 * ni + ii] = -FLT_MAX;
                    }
                }
            }
        }
    }

    template <typename Mask, typename AlibiParams>
    inline __device__ void apply_mask_alibi(Mask const& mask, int head_id, AlibiParams const& alibi_params)
    {
        // 'if constexpr' because ALiBi is only defined for causal masks
        if constexpr (Kernel_traits::CAUSAL_MASK)
        {
            float m = get_alibi_head_scaling_factor<AlibiParams>(head_id, alibi_params);
#pragma unroll
            for (int mi = 0; mi < MMAS_M; ++mi)
            {
#pragma unroll
                for (int ni = 0; ni < MMAS_N; ++ni)
                {
#pragma unroll
                    for (int ii = 0; ii < 8; ++ii)
                    {
                        int row, col;
                        mask.get_row_col(row, col, mi, ni, 0, ii);
                        if (mask.is_valid(row, col))
                        {
                            // Since softmax is shift invariant,
                            //  it is sufficient just to use the column as the multiplier
                            elt_[mi][8 * ni + ii] = elt_[mi][8 * ni + ii] * alibi_params.scale_after_alibi
                                + m * (col + alibi_params.sequence_pos_offset);
                        }
                        else
                        {
                            elt_[mi][8 * ni + ii] = -FLT_MAX;
                        }
                    }
                }
            }
        }
        else
        {
            __builtin_unreachable();
        }
    }

    // Apply the mask to unpacked data.
    inline __device__ void apply_mask(uint32_t const (&packed_mask)[MMAS_M])
    {

        // This code works only if we have MMAS_N <= 4.
        static_assert(MMAS_N <= 4, "");

        // Expand the mask.
        int mask[MMAS_M][MMAS_N * 8];
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ii = 0; ii < MMAS_N * 8; ++ii)
            {
                mask[mi][ii] = packed_mask[mi] & (1u << ii);
            }
        }

// Apply the mask.
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N * 8; ++ni)
            {
                if (!mask[mi][ni])
                {
                    elt_[mi][ni] = -FLT_MAX;
                }
            }
        }
    }

    // Mask the elements that are outside the the sequence length.
    inline __device__ void apply_mask(int const seqlen)
    {

        // The warp/lane decomposition.
        int const warp = threadIdx.x / Cta_tile::THREADS_PER_WARP;
        int const lane = threadIdx.x % Cta_tile::THREADS_PER_WARP;

        // The warp in the n dimension.
        int const warp_n = warp / Cta_tile::WARPS_M;
        // The base position within a quad.
        int const offset = warp_n * 16 + (threadIdx.x & 0x08) / 2;

#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {

                // The position in the sequence.
                int pos = offset + ni * Mma_tile::N_PER_MMA_PER_CTA;

                // Determine the position in the sequence.
                if (pos + 0 >= seqlen)
                {
                    elt_[mi][8 * ni + 0] = -FLT_MAX;
                }
                if (pos + 1 >= seqlen)
                {
                    elt_[mi][8 * ni + 1] = -FLT_MAX;
                }
                if (pos + 2 >= seqlen)
                {
                    elt_[mi][8 * ni + 2] = -FLT_MAX;
                }
                if (pos + 3 >= seqlen)
                {
                    elt_[mi][8 * ni + 3] = -FLT_MAX;
                }
                if (pos + 8 >= seqlen)
                {
                    elt_[mi][8 * ni + 4] = -FLT_MAX;
                }
                if (pos + 9 >= seqlen)
                {
                    elt_[mi][8 * ni + 5] = -FLT_MAX;
                }
                if (pos + 10 >= seqlen)
                {
                    elt_[mi][8 * ni + 6] = -FLT_MAX;
                }
                if (pos + 11 >= seqlen)
                {
                    elt_[mi][8 * ni + 7] = -FLT_MAX;
                }
            }
        }
    }

    // Apply the exp to all the elements.
    // Need to make sure the results are zero when all elts are -FLT_MAX
    //  as it is possible that all tokens are masked out.
    template <bool APPLY_MASK = false>
    inline __device__ void apply_exp_with_mask(float const (&max)[MMAS_M])
    {
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
            float max_val = APPLY_MASK && max[mi] == -FLT_MAX ? 0.f : max[mi];
#pragma unroll
            for (int ni = 0; ni < MMAS_N * 8; ++ni)
            {
                this->elt_[mi][ni] = expf(this->elt_[mi][ni] - max_val);
            }
        }
    }

    // Apply the exp to all the elements.
    inline __device__ void apply_exp(float const max)
    {
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N * 8; ++ni)
            {
                elt_[mi][ni] = apply_exp_<Kernel_traits::VERSION>(elt_[mi][ni], max);
            }
        }
    }

    // Apply the exp to all the elements.
    inline __device__ void apply_exp(float const (&max)[MMAS_M])
    {
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N * 8; ++ni)
            {
                elt_[mi][ni] = apply_exp_<Kernel_traits::VERSION>(elt_[mi][ni], max[mi]);
            }
        }
    }

    // Pack the data to a fragment for the next GEMM.
    template <int K, int M>
    inline __device__ void pack(Fragment_a (&dst)[K][M]) const
    {
        static_assert(MMAS_M == M && MMAS_N == K, "");
#pragma unroll
        for (int mi = 0; mi < M; ++mi)
        {
#pragma unroll
            for (int ki = 0; ki < K; ++ki)
            {

                // 8 elements per row.
                float tmp_0 = this->elt_[mi][8 * ki + 0];
                float tmp_1 = this->elt_[mi][8 * ki + 1];
                float tmp_2 = this->elt_[mi][8 * ki + 2];
                float tmp_3 = this->elt_[mi][8 * ki + 3];
                float tmp_4 = this->elt_[mi][8 * ki + 4];
                float tmp_5 = this->elt_[mi][8 * ki + 5];
                float tmp_6 = this->elt_[mi][8 * ki + 6];
                float tmp_7 = this->elt_[mi][8 * ki + 7];

                // Pack to 4 registers.
                dst[ki][mi].reg(0) = fmha::float2_to_16bit_2<Dst_type>(tmp_0, tmp_1);
                dst[ki][mi].reg(1) = fmha::float2_to_16bit_2<Dst_type>(tmp_2, tmp_3);
                dst[ki][mi].reg(2) = fmha::float2_to_16bit_2<Dst_type>(tmp_4, tmp_5);
                dst[ki][mi].reg(3) = fmha::float2_to_16bit_2<Dst_type>(tmp_6, tmp_7);
            }
        }
    }

    // Do a CTA-wide reduction.
    template <typename Functor>
    inline __device__ void reduce_Nx1(float (&dst)[MMAS_M])
    {
#if defined(USE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE)
        if (Functor::IS_SUM)
        {
// Apply the summation inside the thread for each row.
#pragma unroll
            for (int mi = 0; mi < MMAS_M; ++mi)
            {

                // The thread local math in the reference code.
                float sums[MMAS_N * 2];
#pragma unroll
                for (int ii = 0; ii < MMAS_N * 2; ++ii)
                {
                    sums[ii] = elt_[mi][4 * ii + 0];
                    sums[ii] += elt_[mi][4 * ii + 1];
                    sums[ii] += elt_[mi][4 * ii + 2];
                    sums[ii] += elt_[mi][4 * ii + 3];
                }

// Columns 0 and  8: __shfl( 2).
#pragma unroll
                for (int ii = 0; ii < MMAS_N; ++ii)
                {
                    sums[2 * ii] += sums[2 * ii + 1];
                }

// Columns 0 and 32: __shfl( 8).
#pragma unroll
                for (int ii = 0; ii < MMAS_N / 2; ++ii)
                { // MMAS_N / 2 == 0 if MMAS_N <= 1.
                    sums[4 * ii] += sums[4 * ii + 2];
                }

                // Columns 0 and 64: __shfl(16).
                if (MMAS_N == 3)
                {
                    sums[0] += sums[4];
                }
                else if (MMAS_N >= 4)
                {
#pragma unroll
                    for (int ii = 0; ii < MMAS_N / 4; ++ii)
                    { // MMAS_N / 4 == 0 if MMAS_N <= 2.
                        sums[8 * ii] += sums[8 * ii + 4];
                    }
                }

                // Store the final value for that row.
                dst[mi] = sums[0];
            }
        }
        else
#endif // defined(USE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE)
        {
// Apply the functor for each row inside a thread.
#pragma unroll
            for (int mi = 0; mi < MMAS_M; ++mi)
            {
                dst[mi] = elt_[mi][0];
#pragma unroll
                for (int ni = 1; ni < MMAS_N * 8; ++ni)
                {
                    dst[mi] = Functor::apply(dst[mi], elt_[mi][ni]);
                }
            }
        }

// Apply the functor for each row.
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
            dst[mi] = Functor::apply(dst[mi], __shfl_xor_sync(uint32_t(-1), dst[mi], 8));
        }
    }

    // Do a CTA-wide reduction.
    template <typename Functor>
    inline __device__ float reduce_2x2()
    {
        float dst[MMAS_M];
#if defined(USE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE)
        if (Functor::IS_SUM)
        {
// Apply the summation inside the thread for each row.
#pragma unroll
            for (int mi = 0; mi < MMAS_M; ++mi)
            {

                // The thread local math in the reference code.
                float sums[MMAS_N * 2];
#pragma unroll
                for (int ii = 0; ii < MMAS_N * 2; ++ii)
                {
                    sums[ii] = elt_[mi][4 * ii + 0];
                    sums[ii] += elt_[mi][4 * ii + 1];
                    sums[ii] += elt_[mi][4 * ii + 2];
                    sums[ii] += elt_[mi][4 * ii + 3];
                }

// Columns 0 and  8: __shfl( 2).
#pragma unroll
                for (int ii = 0; ii < MMAS_N; ++ii)
                {
                    sums[2 * ii] += sums[2 * ii + 1];
                }

// Columns 0 and 32: __shfl( 8).
#pragma unroll
                for (int ii = 0; ii < MMAS_N / 2; ++ii)
                { // MMAS_N / 2 == 0 if MMAS_N <= 1.
                    sums[4 * ii] += sums[4 * ii + 2];
                }

                // Columns 0 and 64: __shfl(16).
                if (MMAS_N == 3)
                {
                    sums[0] += sums[4];
                }
                else if (MMAS_N >= 4)
                {
#pragma unroll
                    for (int ii = 0; ii < MMAS_N / 4; ++ii)
                    { // MMAS_N / 4 == 0 if MMAS_N <= 2.
                        sums[8 * ii] += sums[8 * ii + 4];
                    }
                }

                // Store the final value for that row.
                dst[mi] = sums[0];
            }
        }
        else
#endif // defined(USE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE)
        {
// Apply the functor for each row inside a thread.
#pragma unroll
            for (int mi = 0; mi < MMAS_M; ++mi)
            {
                dst[mi] = elt_[mi][0];
#pragma unroll
                for (int ni = 1; ni < MMAS_N * 8; ++ni)
                {
                    dst[mi] = Functor::apply(dst[mi], elt_[mi][ni]);
                }
            }
        }

// Apply the functor for each row.
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
            dst[mi] = Functor::apply(dst[mi], __shfl_xor_sync(uint32_t(-1), dst[mi], 8));
        }

// Store the different values to shared memory.
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
            if (tidx_ % 16 < 8)
            {
                smem_write_[mi * Mma_tile::M_PER_MMA_PER_CTA * ELEMENTS_PER_ROW] = dst[mi];
            }
        }

        // Make sure the values are in shared memory.
        __syncthreads();

        // Load 2 values (one for each warp).
        float2 tmp = reinterpret_cast<float2 const*>(smem_)[tidx_];

        // Compute the reduction of those 2 values in a binary-tree fashion.
        return Functor::apply(tmp.x, tmp.y);
    }

    // Do a CTA-wide reduction.
    template <typename Functor>
    inline __device__ float reduce_1x4()
    {
        float dst[MMAS_M];
#if defined(USE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE)
        if (Functor::IS_SUM)
        {
// Apply the summation inside the thread for each row.
#pragma unroll
            for (int mi = 0; mi < MMAS_M; ++mi)
            {

                // The thread local math in the reference code.
                float sums[MMAS_N * 2];
#pragma unroll
                for (int ii = 0; ii < MMAS_N * 2; ++ii)
                {
                    sums[ii] = elt_[mi][4 * ii + 0];
                    sums[ii] += elt_[mi][4 * ii + 1];
                    sums[ii] += elt_[mi][4 * ii + 2];
                    sums[ii] += elt_[mi][4 * ii + 3];
                }

                // Columns 0 and 128 (the ref code uses a step of 128). Not needed if SEQLEN <= 128.
                if (Cta_tile::N > 128)
                {
#pragma unroll
                    for (int ii = 0; ii < MMAS_N; ++ii)
                    {
                        sums[ii] += sums[MMAS_N + ii];
                    }
                }

// Columns 0 and  8: __shfl( 2).
#pragma unroll
                for (int ii = 0; ii < MMAS_N; ++ii)
                {
                    sums[2 * ii] += sums[2 * ii + 1];
                }

// Columns 0 and 64: __shfl(16).
#pragma unroll
                for (int ii = 0; ii < MMAS_N / 2; ++ii)
                { // MMAS_N / 2 == 0 if MMAS_N <= 1.
                    sums[4 * ii] += sums[4 * ii + 2];
                }

                // Store the final value for that row.
                dst[mi] = sums[0];
            }
        }
        else
#endif // defined(USE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE)
        {
// Apply the functor for each row inside a thread.
#pragma unroll
            for (int mi = 0; mi < MMAS_M; ++mi)
            {
                dst[mi] = elt_[mi][0];
#pragma unroll
                for (int ni = 1; ni < MMAS_N * 8; ++ni)
                {
                    dst[mi] = Functor::apply(dst[mi], elt_[mi][ni]);
                }
            }
        }

// Apply the functor for each row.
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
            dst[mi] = Functor::apply(dst[mi], __shfl_xor_sync(uint32_t(-1), dst[mi], 8));
        }

// Store the different values to shared memory.
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
            if (tidx_ % 16 < 8)
            {
                smem_write_[mi * Mma_tile::M_PER_MMA_PER_CTA * ELEMENTS_PER_ROW] = dst[mi];
            }
        }

        // Make sure the values are in shared memory.
        __syncthreads();

        // Load 4 values (one for each warp).
        float2 tmp[2];
        if (tidx_ < Cta_tile::M)
        {
            tmp[0] = reinterpret_cast<float2 const*>(&smem_[0 * ELEMENTS / 2])[tidx_];
            tmp[1] = reinterpret_cast<float2 const*>(&smem_[1 * ELEMENTS / 2])[tidx_];
        }

        // Compute the reduction of those 4 values in a binary-tree fashion.
        tmp[0].x = Functor::apply(tmp[0].x, tmp[0].y);
        tmp[1].x = Functor::apply(tmp[1].x, tmp[1].y);
        tmp[0].x = Functor::apply(tmp[0].x, tmp[1].x);

        // Return the final reduction.
        return tmp[0].x;
    }

    // Do a CTA-wide reduction.
    template <typename Functor>
    inline __device__ float reduce_1x8()
    {
        float dst[MMAS_M];
#if defined(USE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE)
        if (Functor::IS_SUM)
        {
// Apply the summation inside the thread for each row.
#pragma unroll
            for (int mi = 0; mi < MMAS_M; ++mi)
            {

                // The thread local math in the reference code.
                float sums[MMAS_N * 2];
#pragma unroll
                for (int ii = 0; ii < MMAS_N * 2; ++ii)
                {
                    sums[ii] = elt_[mi][4 * ii + 0];
                    sums[ii] += elt_[mi][4 * ii + 1];
                    sums[ii] += elt_[mi][4 * ii + 2];
                    sums[ii] += elt_[mi][4 * ii + 3];
                }

// Columns 0 and 128 (the ref code uses a step of 128). Not needed if SEQLEN <= 128.
#pragma unroll
                for (int ii = 1; ii < MMAS_N; ++ii)
                {
                    sums[0] += sums[2 * ii + 0];
                    sums[1] += sums[2 * ii + 1];
                }

                // Columns 0 and  8: __shfl( 2).
                dst[mi] = sums[0] + sums[1];
            }
        }
        else
#endif // defined(USE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE)
        {
// Apply the functor for each row inside a thread.
#pragma unroll
            for (int mi = 0; mi < MMAS_M; ++mi)
            {
                dst[mi] = elt_[mi][0];
#pragma unroll
                for (int ni = 1; ni < MMAS_N * 8; ++ni)
                {
                    dst[mi] = Functor::apply(dst[mi], elt_[mi][ni]);
                }
            }
        }

// Apply the functor for each row.
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
            dst[mi] = Functor::apply(dst[mi], __shfl_xor_sync(uint32_t(-1), dst[mi], 8));
        }

// Store the different values to shared memory.
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
            if (tidx_ % 16 < 8)
            {
                smem_write_[mi * Mma_tile::M_PER_MMA_PER_CTA * ELEMENTS_PER_ROW] = dst[mi];
            }
        }

        // Make sure the values are in shared memory.
        __syncthreads();

        // Load 8 values (one for each warp).
        float2 tmp[4];
        if (tidx_ < Cta_tile::M)
        {
            tmp[0] = reinterpret_cast<float2 const*>(&smem_[0 * ELEMENTS / 4])[tidx_];
            tmp[1] = reinterpret_cast<float2 const*>(&smem_[1 * ELEMENTS / 4])[tidx_];
            tmp[2] = reinterpret_cast<float2 const*>(&smem_[2 * ELEMENTS / 4])[tidx_];
            tmp[3] = reinterpret_cast<float2 const*>(&smem_[3 * ELEMENTS / 4])[tidx_];
        }

        // // DEBUG.
        // if( tidx_ == 0 ) {
        //     #pragma unroll
        //     for( int ii = 0; ii < 4; ++ii ) {
        //         printf("tidx=%3d tmp[%d]=%8.3f %8.3f\n", tidx_, ii, tmp[ii].x, tmp[ii].y);
        //     }
        // }
        // // END OF DEBUG.

        // Compute the reduction of those 8 values in a binary-tree fashion.
        tmp[0].x = Functor::apply(tmp[0].x, tmp[0].y);
        tmp[1].x = Functor::apply(tmp[1].x, tmp[1].y);
        tmp[2].x = Functor::apply(tmp[2].x, tmp[2].y);
        tmp[3].x = Functor::apply(tmp[3].x, tmp[3].y);

        tmp[0].x = Functor::apply(tmp[0].x, tmp[1].x);
        tmp[2].x = Functor::apply(tmp[2].x, tmp[3].x);

        tmp[0].x = Functor::apply(tmp[0].x, tmp[2].x);

        // Return the final reduction.
        return tmp[0].x;
    }

    // Do a CTA-wide reduction.
    template <typename Functor>
    inline __device__ float reduce_()
    {

        // The final reduction.
        float red = 0.f;

        // SEQLEN == 128.
        if (Cta_tile::WARPS_M == 2 && Cta_tile::WARPS_N == 2)
        {
            red = reduce_2x2<Functor>();

            // SEQLEN == 256.
        }
        else if (Cta_tile::WARPS_M == 1 && Cta_tile::WARPS_N == 4)
        {
            red = reduce_1x4<Functor>();

            // SEQLEN == 256.
        }
        else if (Cta_tile::WARPS_M == 1 && Cta_tile::WARPS_N == 8)
        {
            red = reduce_1x8<Functor>();

            // Not supported.
        }
        else
        {
            assert(false);
        }

        return red;
    }

    // Finalize the reduction.
    inline __device__ void shuffle(float (&dst)[MMAS_M], float red)
    {

        // Store the value back to shared memory.
        if (tidx_ < Cta_tile::M)
        {
            smem_[tidx_] = red;
        }

        // Make sure the data is in shared memory.
        __syncthreads();

// Finally read the values.
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
            dst[mi] = smem_read_[mi * Mma_tile::M_PER_MMA_PER_CTA];
        }

        // Make sure we are done reading shared memory.
        __syncthreads();
    }

    // Do a CTA-wide reduction.
    template <typename Functor>
    inline __device__ void reduce(float (&dst)[MMAS_M])
    {

        // NOTE: 1 warp along reduce direction, no syncs
        if (Cta_tile::WARPS_N == 1)
        {
            reduce_Nx1<Functor>(dst);
        }
        else
        {
            // The result of the reduction. Threads 0..Cta_tile::M-1 own a valid value.
            float red = reduce_<Functor>();

            // Make sure we can write to shared memory.
            __syncthreads();

            // Finalize the reduction.
            shuffle(dst, red);
        }
    }

    // Scale all the elements.
    inline __device__ void scale(float const (&sum)[MMAS_M])
    {
        // Precompute the inverse sum to normalize. Without -use_fast_math, it makes a huge deal.
        float inv_sum[MMAS_M];
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
            inv_sum[mi] = (sum[mi] == 0.f || sum[mi] != sum[mi]) ? 1.f : 1.f / sum[mi];
        }

// Update the values.
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N * 8; ++ni)
            {
                elt_[mi][ni] *= inv_sum[mi];
            }
        }
    }

    // Store the tile after softmax.
    template <typename Gmem_tile>
    inline __device__ void store(Gmem_tile& gmem_tile)
    {
        Accumulator acc[MMAS_M][MMAS_N];
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {

                // The elements.
                float tmp_00 = this->elt_[mi][8 * ni + 0];
                float tmp_01 = this->elt_[mi][8 * ni + 1];
                float tmp_02 = this->elt_[mi][8 * ni + 2];
                float tmp_03 = this->elt_[mi][8 * ni + 3];
                float tmp_04 = this->elt_[mi][8 * ni + 4];
                float tmp_05 = this->elt_[mi][8 * ni + 5];
                float tmp_06 = this->elt_[mi][8 * ni + 6];
                float tmp_07 = this->elt_[mi][8 * ni + 7];

                // Transform to accumulators.
                acc[mi][ni].reg(0) = fmha::float2_to_16bit_2<Dst_type>(tmp_00, tmp_01);
                acc[mi][ni].reg(1) = fmha::float2_to_16bit_2<Dst_type>(tmp_02, tmp_03);
                acc[mi][ni].reg(2) = fmha::float2_to_16bit_2<Dst_type>(tmp_04, tmp_05);
                acc[mi][ni].reg(3) = fmha::float2_to_16bit_2<Dst_type>(tmp_06, tmp_07);
            }
        }

        // Delegate to the gmem tile to store.
        gmem_tile.store(acc);
    }

    // Convert from FP16 fragments to floats.
    inline __device__ void unpack(Accumulator const (&acc)[MMAS_M][MMAS_N])
    {
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {

                // Normalize the values, and clamp to finite half.
                uint32_t acc_0 = satfinite_h2(hmul2(acc[mi][ni].reg(0), params_scale_bmm1_));
                uint32_t acc_1 = satfinite_h2(hmul2(acc[mi][ni].reg(1), params_scale_bmm1_));
                uint32_t acc_2 = satfinite_h2(hmul2(acc[mi][ni].reg(2), params_scale_bmm1_));
                uint32_t acc_3 = satfinite_h2(hmul2(acc[mi][ni].reg(3), params_scale_bmm1_));

                // Extract the values as floats.
                half2_to_float2(this->elt_[mi][8 * ni + 0], this->elt_[mi][8 * ni + 1], acc_0);
                half2_to_float2(this->elt_[mi][8 * ni + 2], this->elt_[mi][8 * ni + 3], acc_1);
                half2_to_float2(this->elt_[mi][8 * ni + 4], this->elt_[mi][8 * ni + 5], acc_2);
                half2_to_float2(this->elt_[mi][8 * ni + 6], this->elt_[mi][8 * ni + 7], acc_3);

                if constexpr (ENABLE_BMM1_SOFTCAPPING_SCALE)
                {
#pragma unroll
                    for (int i = 0; i < 8; i++)
                    {
                        // 1.0f / softcapping_scale has been fused to scale_bmm1.
                        this->elt_[mi][8 * ni + i]
                            = params_softcapping_scale_bmm1_ * __tanhf(this->elt_[mi][8 * ni + i]);
                    }
                }
            }
        }
    }

    // The scaling factor.
    uint32_t const params_scale_bmm1_;
    float const params_softcapping_scale_bmm1_;
    // Shared memory for the CTA-wide reduction.
    float *smem_, *smem_write_, *smem_read_;
    // The current thread index.
    int tidx_;
    // The elements.
    float elt_[MMAS_M][MMAS_N * 8];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, typename Kernel_traits>
struct Softmax<fmha::Turing_hmma_fp16_traits, Cta_tile, Kernel_traits>
    : public Softmax_hmma<fmha::Turing_hmma_fp16_traits, Cta_tile, Kernel_traits>
{

    // The traits.
    using Traits = fmha::Turing_hmma_fp16_traits;
    // The base class.
    using Base = Softmax_hmma<Traits, Cta_tile, Kernel_traits>;
    // The fragment.
    using Fragment_a = fmha::Fragment_a<Traits, fmha::Row>;
    // Softmax dst data_type (BMM2 input)
    using Dst_type = typename Traits::A_type;

    // Ctor.
    template <typename Params>
    inline __device__ Softmax(Params const& params, void* smem, int bidb, int tidx)
        : Base(params, smem, bidb, tidx)
    {
    }

    // Pack the data to a fragment for the next GEMM.
    template <int K, int M>
    inline __device__ void pack(Fragment_a (&dst)[K][M]) const
    {
        static_assert(Base::Mma_tile::MMAS_M == M && Base::Mma_tile::MMAS_N * 4 == K * 2, "");
#pragma unroll
        for (int mi = 0; mi < M; ++mi)
        {
#pragma unroll
            for (int ki = 0; ki < K; ++ki)
            {

                // 1st row - 2 elements per row.
                float tmp_00 = this->elt_[2 * mi + 0][2 * ki + 0];
                float tmp_01 = this->elt_[2 * mi + 0][2 * ki + 1];

                // 2nd row - 2 elements per row.
                float tmp_10 = this->elt_[2 * mi + 1][2 * ki + 0];
                float tmp_11 = this->elt_[2 * mi + 1][2 * ki + 1];

                // Pack to 2 registers.
                dst[ki][mi].reg(0) = fmha::float2_to_16bit_2<Dst_type>(tmp_00, tmp_01);
                dst[ki][mi].reg(1) = fmha::float2_to_16bit_2<Dst_type>(tmp_10, tmp_11);
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, typename Kernel_traits>
struct Softmax<fmha::Volta_imma_int8_int32_traits, Cta_tile, Kernel_traits>
    : public Softmax_imma<fmha::Volta_imma_int8_int32_traits, Cta_tile, Kernel_traits>
{

    // The traits.
    using Traits = fmha::Volta_imma_int8_int32_traits;
    // The base class.
    using Base = Softmax_imma<Traits, Cta_tile, Kernel_traits>;
    // The fragment.
    using Fragment_a = fmha::Fragment_a<Traits, fmha::Row>;

    // Ctor.
    template <typename Params>
    inline __device__ Softmax(Params const& params, void* smem, int bidb, int tidx)
        : Base(params, smem, bidb, tidx)
    {
    }

    // Repack. We could use store/load to match the Smem_tile API.
    template <int K, int M>
    inline __device__ void pack(Fragment_a (&dst)[K][M])
    {
        static_assert(Base::Mma_tile::MMAS_M == M && Base::Mma_tile::MMAS_N == K, "");
        float const scale = reinterpret_cast<float const&>(this->params_scale_softmax_);
#pragma unroll
        for (int mi = 0; mi < M; ++mi)
        {
#pragma unroll
            for (int ki = 0; ki < K; ++ki)
            {

                // 1st row - 4 elements per row.
                float tmp_00 = this->elt_[2 * mi + 0][4 * ki + 0] * scale;
                float tmp_01 = this->elt_[2 * mi + 0][4 * ki + 1] * scale;
                float tmp_02 = this->elt_[2 * mi + 0][4 * ki + 2] * scale;
                float tmp_03 = this->elt_[2 * mi + 0][4 * ki + 3] * scale;

                // 2nd row - 4 elements per row.
                float tmp_10 = this->elt_[2 * mi + 1][4 * ki + 0] * scale;
                float tmp_11 = this->elt_[2 * mi + 1][4 * ki + 1] * scale;
                float tmp_12 = this->elt_[2 * mi + 1][4 * ki + 2] * scale;
                float tmp_13 = this->elt_[2 * mi + 1][4 * ki + 3] * scale;

                // Pack to 2 registers.
                dst[ki][mi].reg(0) = float4_to_char4<false>(tmp_00, tmp_01, tmp_02, tmp_03);
                dst[ki][mi].reg(1) = float4_to_char4<false>(tmp_10, tmp_11, tmp_12, tmp_13);
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, typename Kernel_traits>
struct Softmax<fmha::Turing_imma_int8_int32_traits, Cta_tile, Kernel_traits>
    : public Softmax_imma<fmha::Turing_imma_int8_int32_traits, Cta_tile, Kernel_traits>
{

    // The traits.
    using Traits = fmha::Turing_imma_int8_int32_traits;
    // The base class.
    using Base = Softmax_imma<Traits, Cta_tile, Kernel_traits>;
    // The fragment.
    using Fragment_a = fmha::Fragment_a<Traits, fmha::Row>;

    // Ctor.
    template <typename Params>
    inline __device__ Softmax(Params const& params, void* smem, int bidb, int tidx)
        : Base(params, smem, bidb, tidx)
    {
    }

    // Repack. We could use store/load to match the Smem_tile API.
    template <int K, int M>
    inline __device__ void pack(Fragment_a (&dst)[K][M])
    {
        static_assert(Base::Mma_tile::MMAS_M == M && Base::Mma_tile::MMAS_N == K, "");
        float const scale = reinterpret_cast<float const&>(this->params_scale_softmax_);
#pragma unroll
        for (int mi = 0; mi < M; ++mi)
        {
#pragma unroll
            for (int ki = 0; ki < K; ++ki)
            {

                // 1st row - 4 elements per row.
                float tmp_00 = this->elt_[2 * mi + 0][4 * ki + 0] * scale;
                float tmp_01 = this->elt_[2 * mi + 0][4 * ki + 1] * scale;
                float tmp_02 = this->elt_[2 * mi + 0][4 * ki + 2] * scale;
                float tmp_03 = this->elt_[2 * mi + 0][4 * ki + 3] * scale;

                // 2nd row - 4 elements per row.
                float tmp_10 = this->elt_[2 * mi + 1][4 * ki + 0] * scale;
                float tmp_11 = this->elt_[2 * mi + 1][4 * ki + 1] * scale;
                float tmp_12 = this->elt_[2 * mi + 1][4 * ki + 2] * scale;
                float tmp_13 = this->elt_[2 * mi + 1][4 * ki + 3] * scale;

                // Pack to 2 registers.
                dst[ki][mi].reg(0) = float4_to_char4<false>(tmp_00, tmp_01, tmp_02, tmp_03);
                dst[ki][mi].reg(1) = float4_to_char4<false>(tmp_10, tmp_11, tmp_12, tmp_13);
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, typename Kernel_traits>
struct Softmax<fmha::Ampere_hmma_fp16_traits, Cta_tile, Kernel_traits>
    : public Softmax_hmma<fmha::Ampere_hmma_fp16_traits, Cta_tile, Kernel_traits>
{

    // The traits.
    using Traits = fmha::Ampere_hmma_fp16_traits;
    // The base class.
    using Base = Softmax_hmma<Traits, Cta_tile, Kernel_traits>;
    // The fragment.
    using Fragment_a = fmha::Fragment_a<Traits, fmha::Row>;
    // Softmax dst data_type (BMM2 input)
    using Dst_type = typename Traits::A_type;

    // Ctor.
    template <typename Params>
    inline __device__ Softmax(Params const& params, void* smem, int bidb, int tidx)
        : Base(params, smem, bidb, tidx)
    {
    }

    // Pack the data to a fragment for the next GEMM.
    template <int K, int M>
    inline __device__ void pack(Fragment_a (&dst)[K][M]) const
    {
#pragma unroll
        for (int mi = 0; mi < M; ++mi)
        {
#pragma unroll
            for (int ki = 0; ki < K; ++ki)
            {

                // 1st row - 4 elements per row.
                float tmp_00 = this->elt_[2 * mi + 0][4 * ki + 0];
                float tmp_01 = this->elt_[2 * mi + 0][4 * ki + 1];
                float tmp_02 = this->elt_[2 * mi + 0][4 * ki + 2];
                float tmp_03 = this->elt_[2 * mi + 0][4 * ki + 3];

                // 2nd row - 4 elements per row.
                float tmp_10 = this->elt_[2 * mi + 1][4 * ki + 0];
                float tmp_11 = this->elt_[2 * mi + 1][4 * ki + 1];
                float tmp_12 = this->elt_[2 * mi + 1][4 * ki + 2];
                float tmp_13 = this->elt_[2 * mi + 1][4 * ki + 3];

                // Pack to 4 registers.
                dst[ki][mi].reg(0) = fmha::float2_to_16bit_2<Dst_type>(tmp_00, tmp_01);
                dst[ki][mi].reg(1) = fmha::float2_to_16bit_2<Dst_type>(tmp_10, tmp_11);
                dst[ki][mi].reg(2) = fmha::float2_to_16bit_2<Dst_type>(tmp_02, tmp_03);
                dst[ki][mi].reg(3) = fmha::float2_to_16bit_2<Dst_type>(tmp_12, tmp_13);
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, typename Kernel_traits>
struct Softmax_fp32 : public Softmax_hmma<Traits, Cta_tile, Kernel_traits>
{

    // The base class.
    using Base = Softmax_hmma<Traits, Cta_tile, Kernel_traits>;
    // The fragment.
    using Fragment_a = fmha::Fragment_a<Traits, fmha::Row>;

    // The MMAs.
    enum
    {
        MMAS_M = Base::MMAS_M
    };

    enum
    {
        MMAS_N = Base::MMAS_N
    };

    // The accumulators.
    using Accumulator = fmha::Fragment_accumulator<Traits>;
    // Output accumulators (after conversion).
    using Accumulator_out = fmha::Fragment_accumulator<Ampere_hmma_fp16_traits>;

    // Softmax dst data_type (BMM2 input)
    using Dst_type = typename Traits::A_type;

    // DEBUG.
    static_assert(Accumulator_out::NUM_REGS == 4, "");
    // END OF DEBUG.

    // DEBUG.
    static_assert(std::is_same<typename Accumulator::Data_type, float>::value, "");

    // END OF DEBUG.

    enum
    {
        WARPS_M = Cta_tile::WARPS_M
    };

    enum
    {
        WARPS_N = Cta_tile::WARPS_N
    };

    // Use BMM1 softcapping scale or not.
    enum
    {
        ENABLE_BMM1_SOFTCAPPING_SCALE = Kernel_traits::ENABLE_BMM1_SOFTCAPPING_SCALE
    };

    using Smem_tile_red = Smem_tile_reduce<Traits, Cta_tile, Kernel_traits>;
    static_assert(Smem_tile_red::ELTS_PER_TILE == Cta_tile::M * WARPS_N);

    // Ctor.
    template <typename Params>
    inline __device__ Softmax_fp32(Params const& params, void* smem, int bidb, int tidx)
        : Base(params, smem, bidb, tidx)
        , smem_sum_(static_cast<float*>(smem), tidx)
        , smem_max_(static_cast<float*>(smem) + Smem_tile_red::ELTS_PER_TILE, tidx)
    {
    }

    // Store the tile after softmax.
    template <typename Gmem_tile>
    inline __device__ void store(Gmem_tile& gmem_tile)
    {
        Accumulator_out acc[MMAS_M][MMAS_N];
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {

                // The elements.
                float tmp_00 = this->elt_[2 * mi + 0][4 * ni + 0];
                float tmp_01 = this->elt_[2 * mi + 0][4 * ni + 1];
                float tmp_02 = this->elt_[2 * mi + 0][4 * ni + 2];
                float tmp_03 = this->elt_[2 * mi + 0][4 * ni + 3];
                float tmp_10 = this->elt_[2 * mi + 1][4 * ni + 0];
                float tmp_11 = this->elt_[2 * mi + 1][4 * ni + 1];
                float tmp_12 = this->elt_[2 * mi + 1][4 * ni + 2];
                float tmp_13 = this->elt_[2 * mi + 1][4 * ni + 3];

                // Transform to accumulators.
                acc[mi][ni].reg(0) = fmha::float2_to_16bit_2<Dst_type>(tmp_00, tmp_01);
                acc[mi][ni].reg(1) = fmha::float2_to_16bit_2<Dst_type>(tmp_10, tmp_11);
                acc[mi][ni].reg(2) = fmha::float2_to_16bit_2<Dst_type>(tmp_02, tmp_03);
                acc[mi][ni].reg(3) = fmha::float2_to_16bit_2<Dst_type>(tmp_12, tmp_13);
            }
        }

        // Delegate to the gmem tile to store.
        gmem_tile.store(acc);
    }

    // Pack the data to a fragment for the next GEMM.
    template <int K, int M>
    inline __device__ void pack(Fragment_a (&dst)[K][M]) const
    {
        static_assert(Fragment_a::NUM_REGS == 4, "");
#pragma unroll
        for (int mi = 0; mi < M; ++mi)
        {
#pragma unroll
            for (int ki = 0; ki < K; ++ki)
            {

                // 1st row - 4 elements per row.
                float tmp_00 = this->elt_[2 * mi + 0][4 * ki + 0];
                float tmp_01 = this->elt_[2 * mi + 0][4 * ki + 1];
                float tmp_02 = this->elt_[2 * mi + 0][4 * ki + 2];
                float tmp_03 = this->elt_[2 * mi + 0][4 * ki + 3];

                // 2nd row - 4 elements per row.
                float tmp_10 = this->elt_[2 * mi + 1][4 * ki + 0];
                float tmp_11 = this->elt_[2 * mi + 1][4 * ki + 1];
                float tmp_12 = this->elt_[2 * mi + 1][4 * ki + 2];
                float tmp_13 = this->elt_[2 * mi + 1][4 * ki + 3];

                // Pack to 4 registers.
                dst[ki][mi].reg(0) = fmha::float2_to_16bit_2<Dst_type>(tmp_00, tmp_01);
                dst[ki][mi].reg(1) = fmha::float2_to_16bit_2<Dst_type>(tmp_10, tmp_11);
                dst[ki][mi].reg(2) = fmha::float2_to_16bit_2<Dst_type>(tmp_02, tmp_03);
                dst[ki][mi].reg(3) = fmha::float2_to_16bit_2<Dst_type>(tmp_12, tmp_13);
            }
        }
    }

    // Pack the data to a uint4 for the next operation.
    template <int M, int N>
    inline __device__ void pack(uint4 (&dst)[M][N]) const
    {
#pragma unroll
        for (int mi = 0; mi < M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < N; ++ni)
            {

                // 1st row - 4 elements per row.
                float tmp_00 = this->elt_[2 * mi + 0][4 * ni + 0];
                float tmp_01 = this->elt_[2 * mi + 0][4 * ni + 1];
                float tmp_02 = this->elt_[2 * mi + 0][4 * ni + 2];
                float tmp_03 = this->elt_[2 * mi + 0][4 * ni + 3];

                // 2nd row - 4 elements per row.
                float tmp_10 = this->elt_[2 * mi + 1][4 * ni + 0];
                float tmp_11 = this->elt_[2 * mi + 1][4 * ni + 1];
                float tmp_12 = this->elt_[2 * mi + 1][4 * ni + 2];
                float tmp_13 = this->elt_[2 * mi + 1][4 * ni + 3];

                // Pack to 4 registers.
                dst[mi][ni].x = fmha::float2_to_16bit_2<Dst_type>(tmp_00, tmp_01);
                dst[mi][ni].y = fmha::float2_to_16bit_2<Dst_type>(tmp_02, tmp_03);
                dst[mi][ni].z = fmha::float2_to_16bit_2<Dst_type>(tmp_10, tmp_11);
                dst[mi][ni].w = fmha::float2_to_16bit_2<Dst_type>(tmp_12, tmp_13);
            }
        }
    }

    // Scale FP32 fragments
    inline __device__ void unpack(Accumulator const (&acc)[MMAS_M][MMAS_N])
    {
        float const scalef = reinterpret_cast<float const&>(this->params_scale_bmm1_);

#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {
                // 1st row - 4 elements per row.
                this->elt_[2 * mi + 0][4 * ni + 0] = acc[mi][ni].elt(0) * scalef;
                this->elt_[2 * mi + 0][4 * ni + 1] = acc[mi][ni].elt(1) * scalef;
                this->elt_[2 * mi + 0][4 * ni + 2] = acc[mi][ni].elt(4) * scalef;
                this->elt_[2 * mi + 0][4 * ni + 3] = acc[mi][ni].elt(5) * scalef;
                // 2nd row - 4 elements per row.
                this->elt_[2 * mi + 1][4 * ni + 0] = acc[mi][ni].elt(2) * scalef;
                this->elt_[2 * mi + 1][4 * ni + 1] = acc[mi][ni].elt(3) * scalef;
                this->elt_[2 * mi + 1][4 * ni + 2] = acc[mi][ni].elt(6) * scalef;
                this->elt_[2 * mi + 1][4 * ni + 3] = acc[mi][ni].elt(7) * scalef;

                // Attention logit softcapping scale.
                // 1.0f / softcapping_scale has been fused to scale_bmm1.
                if constexpr (ENABLE_BMM1_SOFTCAPPING_SCALE)
                {
                    this->elt_[2 * mi + 0][4 * ni + 0]
                        = this->params_softcapping_scale_bmm1_ * __tanhf(this->elt_[2 * mi + 0][4 * ni + 0]);
                    this->elt_[2 * mi + 0][4 * ni + 1]
                        = this->params_softcapping_scale_bmm1_ * __tanhf(this->elt_[2 * mi + 0][4 * ni + 1]);
                    this->elt_[2 * mi + 1][4 * ni + 0]
                        = this->params_softcapping_scale_bmm1_ * __tanhf(this->elt_[2 * mi + 1][4 * ni + 0]);
                    this->elt_[2 * mi + 1][4 * ni + 1]
                        = this->params_softcapping_scale_bmm1_ * __tanhf(this->elt_[2 * mi + 1][4 * ni + 1]);
                    this->elt_[2 * mi + 0][4 * ni + 2]
                        = this->params_softcapping_scale_bmm1_ * __tanhf(this->elt_[2 * mi + 0][4 * ni + 2]);
                    this->elt_[2 * mi + 0][4 * ni + 3]
                        = this->params_softcapping_scale_bmm1_ * __tanhf(this->elt_[2 * mi + 0][4 * ni + 3]);
                    this->elt_[2 * mi + 1][4 * ni + 2]
                        = this->params_softcapping_scale_bmm1_ * __tanhf(this->elt_[2 * mi + 1][4 * ni + 2]);
                    this->elt_[2 * mi + 1][4 * ni + 3]
                        = this->params_softcapping_scale_bmm1_ * __tanhf(this->elt_[2 * mi + 1][4 * ni + 3]);
                }
            }
        }
    }

    // Scale FP32 fragments
    inline __device__ void unpack_noscale(Accumulator const (&acc)[MMAS_M][MMAS_N])
    {

#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {
                // 1st row - 4 elements per row.
                this->elt_[2 * mi + 0][4 * ni + 0] = acc[mi][ni].elt(0);
                this->elt_[2 * mi + 0][4 * ni + 1] = acc[mi][ni].elt(1);
                this->elt_[2 * mi + 0][4 * ni + 2] = acc[mi][ni].elt(4);
                this->elt_[2 * mi + 0][4 * ni + 3] = acc[mi][ni].elt(5);
                // 2nd row - 4 elements per row.
                this->elt_[2 * mi + 1][4 * ni + 0] = acc[mi][ni].elt(2);
                this->elt_[2 * mi + 1][4 * ni + 1] = acc[mi][ni].elt(3);
                this->elt_[2 * mi + 1][4 * ni + 2] = acc[mi][ni].elt(6);
                this->elt_[2 * mi + 1][4 * ni + 3] = acc[mi][ni].elt(7);
            }
        }
    }

    template <typename Operator>
    __device__ inline void reduce_(float (&frag)[2 * MMAS_M], Operator& op, Smem_tile_red& smem_red)
    {
#pragma unroll
        for (int mi = 0; mi < 2 * MMAS_M; mi++)
        {
            frag[mi] = this->elt_[mi][0];
#pragma unroll
            for (int ni = 1; ni < 4 * MMAS_N; ni++)
            {
                frag[mi] = op(frag[mi], this->elt_[mi][ni]);
            }
        }
        quad_reduce(frag, frag, op);

        if (WARPS_N > 1)
        {
            smem_red.store(frag);
            __syncthreads();
            typename Smem_tile_red::read_t tmp[2 * MMAS_M];
            smem_red.load(tmp);

            quad_allreduce(frag, tmp, op);
        }
    }

    __device__ inline void reduce_max(float (&frag)[2 * MMAS_M])
    {
        MaxOp<float> max;
        reduce_(frag, max, smem_max_);
    }

    __device__ inline void reduce_sum(float (&frag)[2 * MMAS_M])
    {
        SumOp<float> sum;
        reduce_(frag, sum, smem_sum_);
    }

    __device__ inline float correct(float warp_sum, float warp_max, float max)
    {
        return warp_sum * __expf(warp_max - max);
    }

    __device__ inline float2 correct(float2 warp_sum, float2 warp_max, float max)
    {
        return {correct(warp_sum.x, warp_max.x, max), correct(warp_sum.y, warp_max.y, max)};
    }

    __device__ inline void online_softmax()
    {
        MaxOp<float> maxOp;
        SumOp<float> sumOp;
        float max[2 * MMAS_M];
#pragma unroll
        for (int mi = 0; mi < 2 * MMAS_M; mi++)
        {
            max[mi] = this->elt_[mi][0];
#pragma unroll
            for (int ni = 1; ni < 4 * MMAS_N; ni++)
            {
                max[mi] = maxOp(max[mi], this->elt_[mi][ni]);
            }
        }
        quad_allreduce(max, max, maxOp);
        smem_max_.store(max);
        float sum[2 * MMAS_M];
#pragma unroll
        for (int mi = 0; mi < 2 * MMAS_M; mi++)
        {
            sum[mi] = 0.f;
#pragma unroll
            for (int ni = 0; ni < 4 * MMAS_N; ni++)
            {
                float x = this->elt_[mi][ni];
                this->elt_[mi][ni] = __expf(x - max[mi]);
                sum[mi] += this->elt_[mi][ni];
            }
        }
        quad_allreduce(sum, sum, sumOp);
        smem_sum_.store(sum);

        __syncthreads();

        typename Smem_tile_red::read_t tmp_max[2 * MMAS_M];
        typename Smem_tile_red::read_t tmp_sum[2 * MMAS_M];
        smem_max_.load(tmp_max);
        smem_sum_.load(tmp_sum);
        float full_max[2 * MMAS_M];
        quad_allreduce(full_max, tmp_max, maxOp);
#pragma unroll
        for (int mi = 0; mi < 2 * MMAS_M; mi++)
        {
            tmp_sum[mi] = correct(tmp_sum[mi], tmp_max[mi], full_max[mi]);
        }
        quad_allreduce(sum, tmp_sum, sumOp);
#pragma unroll
        for (int mi = 0; mi < 2 * MMAS_M; mi++)
        {
            float correction = __expf(max[mi] - full_max[mi]) / sum[mi];
#pragma unroll
            for (int ni = 0; ni < 4 * MMAS_N; ni++)
            {
                this->elt_[mi][ni] *= correction;
            }
        }
    }

    Smem_tile_red smem_max_;
    Smem_tile_red smem_sum_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, typename Kernel_traits>
struct Softmax<fmha::Ampere_hmma_fp32_traits, Cta_tile, Kernel_traits>
    : public Softmax_fp32<fmha::Ampere_hmma_fp32_traits, Cta_tile, Kernel_traits>
{

    // The traits.
    using Traits = fmha::Ampere_hmma_fp32_traits;
    // The base class.
    using Base = Softmax_fp32<Traits, Cta_tile, Kernel_traits>;

    // Ctor.
    template <typename Params>
    inline __device__ Softmax(Params const& params, void* smem, int bidb, int tidx)
        : Base(params, smem, bidb, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, typename Kernel_traits>
struct Softmax<fmha::Turing_hmma_fp32_traits, Cta_tile, Kernel_traits>
    : public Softmax_fp32<fmha::Turing_hmma_fp32_traits, Cta_tile, Kernel_traits>
{

    // The traits.
    using Traits = fmha::Turing_hmma_fp32_traits;
    // The base class.
    using Base = Softmax_fp32<Traits, Cta_tile, Kernel_traits>;
    // The fragment.
    using Fragment_a = fmha::Fragment_a<Traits, fmha::Row>;
    // Softmax dst data_type (BMM2 input)
    using Dst_type = typename Traits::A_type;

    // Ctor.
    template <typename Params>
    inline __device__ Softmax(Params const& params, void* smem, int bidb, int tidx)
        : Base(params, smem, bidb, tidx)
    {
    }

    // Pack the data to a fragment for the next GEMM.
    template <int K, int M>
    inline __device__ void pack(Fragment_a (&dst)[K][M]) const
    {
        static_assert(Fragment_a::NUM_REGS == 2, "");
        static_assert(Base::Mma_tile::MMAS_M == M && Base::Mma_tile::MMAS_N * 4 == K * 2, "");
#pragma unroll
        for (int mi = 0; mi < M; ++mi)
        {
#pragma unroll
            for (int ki = 0; ki < K; ++ki)
            {

                // 1st row - 2 elements per row.
                float tmp_00 = this->elt_[2 * mi + 0][2 * ki + 0];
                float tmp_01 = this->elt_[2 * mi + 0][2 * ki + 1];

                // 2nd row - 2 elements per row.
                float tmp_10 = this->elt_[2 * mi + 1][2 * ki + 0];
                float tmp_11 = this->elt_[2 * mi + 1][2 * ki + 1];

                // Pack to 2 registers.
                dst[ki][mi].reg(0) = fmha::float2_to_16bit_2<Dst_type>(tmp_00, tmp_01);
                dst[ki][mi].reg(1) = fmha::float2_to_16bit_2<Dst_type>(tmp_10, tmp_11);
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, typename Kernel_traits>
struct Softmax<fmha::Ampere_hmma_bf16_traits, Cta_tile, Kernel_traits>
    : public Softmax_fp32<fmha::Ampere_hmma_bf16_traits, Cta_tile, Kernel_traits>
{

    // The traits.
    using Traits = fmha::Ampere_hmma_bf16_traits;
    // The base class.
    using Base = Softmax_fp32<Traits, Cta_tile, Kernel_traits>;

    // Ctor.
    template <typename Params>
    inline __device__ Softmax(Params const& params, void* smem, int bidb, int tidx)
        : Base(params, smem, bidb, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, typename Kernel_traits>
struct Softmax<fmha::Ampere_imma_int8_int32_traits, Cta_tile, Kernel_traits>
    : public Softmax_imma<fmha::Ampere_imma_int8_int32_traits, Cta_tile, Kernel_traits>
{

    // The traits.
    using Traits = fmha::Ampere_imma_int8_int32_traits;
    // The base class.
    using Base = Softmax_imma<Traits, Cta_tile, Kernel_traits>;

    // Ctor.
    template <typename Params>
    inline __device__ Softmax(Params const& params, void* smem, int bidb, int tidx)
        : Base(params, smem, bidb, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, typename Kernel_traits>
struct Softmax<fmha::Ada_qmma_e4m3_fp32_traits, Cta_tile, Kernel_traits>
    : public Softmax_qmma<fmha::Ada_qmma_e4m3_fp32_traits, Cta_tile, Kernel_traits>
{

    // The traits.
    using Traits = fmha::Ada_qmma_e4m3_fp32_traits;
    // The base class.
    using Base = Softmax_qmma<Traits, Cta_tile, Kernel_traits>;

    // Ctor.
    template <typename Params>
    inline __device__ Softmax(Params const& params, void* smem, int bidb, int tidx)
        : Base(params, smem, bidb, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, typename Kernel_traits>
struct Softmax<fmha::Ada_qmma_e4m3_fp16_traits, Cta_tile, Kernel_traits>
    : public Softmax_qmma<fmha::Ada_qmma_e4m3_fp16_traits, Cta_tile, Kernel_traits>
{

    // The traits.
    using Traits = fmha::Ada_qmma_e4m3_fp16_traits;
    // The base class.
    using Base = Softmax_qmma<Traits, Cta_tile, Kernel_traits>;

    // Ctor.
    template <typename Params>
    inline __device__ Softmax(Params const& params, void* smem, int bidb, int tidx)
        : Base(params, smem, bidb, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, typename Kernel_traits>
struct Softmax<fmha::Ada_qmma_e4m3_fp32_traits, Cta_tile, Kernel_traits, true>
    : public Softmax_imma<fmha::Ada_qmma_e4m3_fp32_traits, Cta_tile, Kernel_traits>
{

    // The Traits
    using Traits = fmha::Ada_qmma_e4m3_fp32_traits;
    // The base class.
    using Base = Softmax_imma<Traits, Cta_tile, Kernel_traits>;

    // The MMAs.
    enum
    {
        MMAS_M = Base::MMAS_M
    };

    enum
    {
        MMAS_N = Base::MMAS_N
    };

    // The accumulators.
    using Accumulator = fmha::Fragment_accumulator<Traits>;

    // Ctor.
    template <typename Params>
    inline __device__ Softmax(Params const& params, void* smem, int bidb, int tidx)
        : Base(params, smem, bidb, tidx)
        , params_scale_bmm1_(params.scale_bmm1_d ? *params.scale_bmm1_d : params.scale_bmm1)
        , params_scale_softmax_(params.scale_softmax)
    {
    }

    // Store the tile after softmax.
    template <typename Gmem_tile>
    inline __device__ void store(Gmem_tile& gmem_tile)
    {

        float const scale = reinterpret_cast<float const&>(params_scale_softmax_);
        Accumulator acc[MMAS_M][MMAS_N];
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {

                // scale
                acc[mi][ni].ele(0) = this->elt_[2 * mi + 0][4 * ni + 0] * scale;
                acc[mi][ni].ele(1) = this->elt_[2 * mi + 0][4 * ni + 1] * scale;
                acc[mi][ni].ele(4) = this->elt_[2 * mi + 0][4 * ni + 2] * scale;
                acc[mi][ni].ele(5) = this->elt_[2 * mi + 0][4 * ni + 3] * scale;
                acc[mi][ni].ele(2) = this->elt_[2 * mi + 1][4 * ni + 0] * scale;
                acc[mi][ni].ele(3) = this->elt_[2 * mi + 1][4 * ni + 1] * scale;
                acc[mi][ni].ele(6) = this->elt_[2 * mi + 1][4 * ni + 2] * scale;
                acc[mi][ni].ele(7) = this->elt_[2 * mi + 1][4 * ni + 3] * scale;
            }
        }

        // Delegate to the gmem tile to store.
        // TODO: need fp32 to fp8 conversion (move this to gmem_tile)
        gmem_tile.store(acc);
    }

    // Convert from accumulators to floats.
    inline __device__ void unpack(Accumulator const (&acc)[MMAS_M][MMAS_N])
    {

        float const scale = params_scale_q_ * params_scale_k_;
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {

                // Convert to FP32 and scale.
                this->elt_[2 * mi + 0][4 * ni + 0] = acc[mi][ni].elt(0) * scale;
                this->elt_[2 * mi + 0][4 * ni + 1] = acc[mi][ni].elt(1) * scale;
                this->elt_[2 * mi + 0][4 * ni + 2] = acc[mi][ni].elt(4) * scale;
                this->elt_[2 * mi + 0][4 * ni + 3] = acc[mi][ni].elt(5) * scale;
                this->elt_[2 * mi + 1][4 * ni + 0] = acc[mi][ni].elt(2) * scale;
                this->elt_[2 * mi + 1][4 * ni + 1] = acc[mi][ni].elt(3) * scale;
                this->elt_[2 * mi + 1][4 * ni + 2] = acc[mi][ni].elt(6) * scale;
                this->elt_[2 * mi + 1][4 * ni + 3] = acc[mi][ni].elt(7) * scale;
            }
        }
    }

    template <bool APPLY_MASK = false>
    inline __device__ void apply_exp_with_mask(float const (&max)[MMAS_M * 2])
    {
#pragma unroll
        for (int mi = 0; mi < MMAS_M * 2; ++mi)
        {
            float max_val = APPLY_MASK && max[mi] == -FLT_MAX ? 0.f : (max[mi] - logf(Traits::SOFTMAX_FP_QUANT_SCALE));
#pragma unroll
            for (int ni = 0; ni < MMAS_N * 4; ++ni)
            {
                this->elt_[mi][ni] = expf(this->elt_[mi][ni] - max_val);
            }
        }
    }

    // Pack the data to a fragment for the next GEMM.
    template <typename Fragment_a, int K, int M>
    inline __device__ void pack(Fragment_a (&dst)[K][M]) const
    {
        float const scale = reinterpret_cast<float const&>(this->params_scale_softmax_);

// The canonical layout in K should be R0: [0,1,2,3] R2: [16,17,18,19]
// Note below that this is not possible with the register layout of the accumulator.
#pragma unroll
        for (int mi = 0; mi < M; ++mi)
        {
#pragma unroll
            for (int ki = 0; ki < K; ++ki)
            {
                // 1st row - 8 elements per row.
                float tmp_00 = this->elt_[2 * mi + 0][8 * ki + 0] * scale; // + 0
                float tmp_01 = this->elt_[2 * mi + 0][8 * ki + 1] * scale; // + 1
                float tmp_02 = this->elt_[2 * mi + 0][8 * ki + 2] * scale; // + 8
                float tmp_03 = this->elt_[2 * mi + 0][8 * ki + 3] * scale; // + 9
                float tmp_04 = this->elt_[2 * mi + 0][8 * ki + 4] * scale; // +16
                float tmp_05 = this->elt_[2 * mi + 0][8 * ki + 5] * scale; // +17
                float tmp_06 = this->elt_[2 * mi + 0][8 * ki + 6] * scale; // +24
                float tmp_07 = this->elt_[2 * mi + 0][8 * ki + 7] * scale; // +25

                // 2nd row - 4 elements per row.
                float tmp_10 = this->elt_[2 * mi + 1][8 * ki + 0] * scale; // + 0
                float tmp_11 = this->elt_[2 * mi + 1][8 * ki + 1] * scale; // + 1
                float tmp_12 = this->elt_[2 * mi + 1][8 * ki + 2] * scale; // + 8
                float tmp_13 = this->elt_[2 * mi + 1][8 * ki + 3] * scale; // + 9
                float tmp_14 = this->elt_[2 * mi + 1][8 * ki + 4] * scale; // +16
                float tmp_15 = this->elt_[2 * mi + 1][8 * ki + 5] * scale; // +17
                float tmp_16 = this->elt_[2 * mi + 1][8 * ki + 6] * scale; // +24
                float tmp_17 = this->elt_[2 * mi + 1][8 * ki + 7] * scale; // +25

                // Pack to 4 registers.
                dst[ki][mi].reg(0) = fmha::float4_to_fp8x4<Traits::A_type>(tmp_00, tmp_01, tmp_02, tmp_03);
                dst[ki][mi].reg(1) = fmha::float4_to_fp8x4<Traits::A_type>(tmp_10, tmp_11, tmp_12, tmp_13);
                dst[ki][mi].reg(2) = fmha::float4_to_fp8x4<Traits::A_type>(tmp_04, tmp_05, tmp_06, tmp_07);
                dst[ki][mi].reg(3) = fmha::float4_to_fp8x4<Traits::A_type>(tmp_14, tmp_15, tmp_16, tmp_17);
            }
        }
    }

    template <typename Params>
    inline __device__ void move_to_first_block(Params const& params, int bidb, int bidh, int q_loop)
    {
        int scale_q_iter = bidb * params.h * params.sage.q.max_nblock + bidh * params.sage.q.max_nblock + q_loop;
        params_scale_q_ = __ldg(params.sage.q.scales + scale_q_iter);
        params_scale_q_ *= reinterpret_cast<float const&>(params_scale_bmm1_);

        int scale_k_iter = bidb * params.h * params.sage.k.max_nblock + bidh * params.sage.k.max_nblock;
        params_scale_k_iter = reinterpret_cast<float const*>(params.sage.k.scales + scale_k_iter);
        params_scale_k_ = __ldg(params_scale_k_iter);
    }

    inline __device__ void move_to_next_block()
    {
        params_scale_k_iter += 1;
        params_scale_k_ = __ldg(params_scale_k_iter);
    }

    // The scaling factors.
    uint32_t const params_scale_bmm1_, params_scale_softmax_;
    float params_scale_q_, params_scale_k_;
    float const* params_scale_k_iter;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// HOPPER SOFTMAX

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, typename Kernel_traits, int WARPS_N>
struct Softmax_gmma_base
{
};

template <typename Traits_, typename Cta_tile_, typename Kernel_traits_>
struct Softmax_gmma_base<Traits_, Cta_tile_, Kernel_traits_, 1>
{

    // The instruction traits.
    using Traits = Traits_;
    // The Cta_tile.
    using Cta_tile = Cta_tile_;
    // The Kernel traits.
    using Kernel_traits = Kernel_traits_;
    // The accumulators.
    using Accumulator = fmha::Fragment_accumulator<Traits>;
    // The Mma tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    static_assert(Cta_tile::WARPS_M == 4);
    static_assert(Mma_tile::M_PER_MMA_PER_CTA == 64);

    // The number of MMAs in M/N dimensions.
    enum
    {
        MMAS_M = Mma_tile::MMAS_M
    };

    enum
    {
        MMAS_N = Mma_tile::MMAS_N
    };

    // Elements per thread per core matrix.
    enum
    {
        ELTS_PER_THREAD = 2
    };

    // Core matrix is always 8x4.
    enum
    {
        THREADS_PER_ROW = 4
    };

    enum
    {
        SMEM_BYTES = 0
    };

    // The number of rows accessed by each thread.
    enum
    {
        ROWS_PER_THREAD = Traits::GMMA_M / (Cta_tile::THREADS_PER_WARP / THREADS_PER_ROW) / Cta_tile::WARPS_M
    };

    static_assert(ROWS_PER_THREAD == Mma_tile::ROWS_PER_THREAD);

    // The number of columns access by each thread.
    // Note there are 2 elements per reg.
    enum
    {
        COLS_PER_THREAD = Traits::GMMA_N / THREADS_PER_ROW / ELTS_PER_THREAD
    };

    // The number of total elements per thread.
    enum
    {
        TOTAL_ELTS_PER_THREAD = ELTS_PER_THREAD * COLS_PER_THREAD
    };

    template <typename Params>
    inline __device__ Softmax_gmma_base(Params const& params, void*, int const, int const)
        : params_scale_bmm1_(params.scale_bmm1)
        , params_softcapping_scale_bmm1_(params.softcapping_scale_bmm1)
    {
    }

    // Apply mask before softmax. Use 1 byte per MMA distributed as 2x4.
    template <typename Mask>
    inline __device__ void apply_mask(Mask const& mask)
    {
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ii = 0; ii < ROWS_PER_THREAD; ++ii)
            {
#pragma unroll
                for (int ni = 0; ni < MMAS_N; ++ni)
                {
#pragma unroll
                    for (int jj = 0; jj < TOTAL_ELTS_PER_THREAD; ++jj)
                    {
                        if (!mask.is_valid(mi, ni, ii, jj))
                        {
                            this->elt_[ROWS_PER_THREAD * mi + ii][TOTAL_ELTS_PER_THREAD * ni + jj] = -FLT_MAX;
                        }
                    } // jj
                }     // ni
            }         // ii
        }             // mi
    }

    template <typename Mask, typename AlibiParams>
    inline __device__ void apply_mask_alibi(Mask const& mask, int head_id, AlibiParams const& alibi_params)
    {
        // 'if constexpr' because ALiBi is only defined for causal masks
        if constexpr (Kernel_traits::CAUSAL_MASK)
        {
            float m = get_alibi_head_scaling_factor<AlibiParams>(head_id, alibi_params);
#pragma unroll
            for (int mi = 0; mi < MMAS_M; ++mi)
            {
#pragma unroll
                for (int ii = 0; ii < ROWS_PER_THREAD; ++ii)
                {
#pragma unroll
                    for (int ni = 0; ni < MMAS_N; ++ni)
                    {
#pragma unroll
                        for (int jj = 0; jj < TOTAL_ELTS_PER_THREAD; ++jj)
                        {
                            int row, col;
                            mask.get_row_col(row, col, mi, ni, ii, jj);
                            if (mask.is_valid(row, col))
                            {
                                // Since softmax is shift invariant,
                                //  it is sufficient just to use the column as the multiplier
                                elt_[ROWS_PER_THREAD * mi + ii][TOTAL_ELTS_PER_THREAD * ni + jj]
                                    = elt_[ROWS_PER_THREAD * mi + ii][TOTAL_ELTS_PER_THREAD * ni + jj]
                                        * alibi_params.scale_after_alibi
                                    + m * (col + alibi_params.sequence_pos_offset);
                            }
                            else
                            {
                                elt_[ROWS_PER_THREAD * mi + ii][TOTAL_ELTS_PER_THREAD * ni + jj] = -FLT_MAX;
                            }
                        }
                    }
                }
            }
        }
        else
        {
            __builtin_unreachable();
        }
    }

    // Do a CTA-wide reduction.
    template <typename Functor>
    inline __device__ void reduce_4x1(float (&dst)[MMAS_M * ROWS_PER_THREAD])
    {

#if defined(USE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE)
        static_assert(MMAS_N * COLS_PER_THREAD * ELTS_PER_THREAD == MMAS_N * Mma_tile::CORES_N * 2);
        if (Functor::IS_SUM)
        {
// Apply the summation inside the thread.
#pragma unroll
            for (int mi = 0; mi < MMAS_M * ROWS_PER_THREAD; ++mi)
            {
                dst[mi] = (this->elt_[mi][0] + this->elt_[mi][1]);
#pragma unroll
                for (int ni = 1; ni < MMAS_N * Mma_tile::CORES_N; ni++)
                {
                    dst[mi] += (this->elt_[mi][ni * 2 + 0] + this->elt_[mi][ni * 2 + 1]);
                }
            }
        }
        else
#endif // defined(USE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE)
        {
// find the max/sum for each row.
// For hopper, each row is held entirely within 4 threads.
// Apply the functor for each row inside a thread.
#pragma unroll
            for (int mi = 0; mi < MMAS_M * ROWS_PER_THREAD; ++mi)
            {
                dst[mi] = this->elt_[mi][0];
#pragma unroll
                for (int ni = 1; ni < MMAS_N * COLS_PER_THREAD * ELTS_PER_THREAD; ++ni)
                {
                    dst[mi] = Functor::apply(dst[mi], this->elt_[mi][ni]);
                }
            }
        }
// Apply the functor for each row inside each group of 4 threads.
#pragma unroll
        for (int mi = 0; mi < MMAS_M * ROWS_PER_THREAD; ++mi)
        {
            dst[mi] = Functor::apply(dst[mi], __shfl_xor_sync(uint32_t(-1), dst[mi], 1));
            __syncwarp();
            dst[mi] = Functor::apply(dst[mi], __shfl_xor_sync(uint32_t(-1), dst[mi], 2));
            __syncwarp();
        }
    }

    // Do a CTA-wide reduction.
    template <typename Functor>
    inline __device__ void reduce(float (&dst)[MMAS_M * ROWS_PER_THREAD])
    {
        reduce_4x1<Functor>(dst);
    }

    // Apply the exp to all the elements.
    inline __device__ void apply_exp(float const (&max)[MMAS_M * ROWS_PER_THREAD])
    {
#pragma unroll
        for (int mi = 0; mi < MMAS_M * ROWS_PER_THREAD; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N * COLS_PER_THREAD * ELTS_PER_THREAD; ++ni)
            {
                this->elt_[mi][ni] = apply_exp_<Kernel_traits::VERSION>(this->elt_[mi][ni], max[mi]);
            }
        }
    }

    // Scale all the elements.
    inline __device__ void scale(float const (&sum)[MMAS_M * ROWS_PER_THREAD])
    {
        // Precompute the inverse sum to normalize. Without -use_fast_math, it makes a huge deal.
        float inv_sum[MMAS_M * ROWS_PER_THREAD];
#pragma unroll
        for (int mi = 0; mi < MMAS_M * ROWS_PER_THREAD; ++mi)
        {
            inv_sum[mi] = (sum[mi] == 0.f || sum[mi] != sum[mi]) ? 1.f : 1.f / sum[mi];
        }

// Update the values.
#pragma unroll
        for (int mi = 0; mi < MMAS_M * ROWS_PER_THREAD; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N * COLS_PER_THREAD * ELTS_PER_THREAD; ++ni)
            {
                this->elt_[mi][ni] *= inv_sum[mi];
            }
        }
    }

    // The scalig factor. Depens on acc type, e.g. float for 32-bit and fp16x2/bf16x2 for 16-bit.
    uint32_t const params_scale_bmm1_;
    float const params_softcapping_scale_bmm1_;
    // The elements.
    float elt_[MMAS_M * ROWS_PER_THREAD][MMAS_N * COLS_PER_THREAD * ELTS_PER_THREAD];
};

template <typename Traits, typename Cta_tile, typename Kernel_traits>
struct Softmax_gmma_base<Traits, Cta_tile, Kernel_traits, 2>
    : public Softmax_gmma_base<Traits, Cta_tile, Kernel_traits, 1>
{

    using Base = Softmax_gmma_base<Traits, Cta_tile, Kernel_traits, 1>;

    using Mma_tile = typename Base::Mma_tile;

    enum
    {
        BYTES_PER_SMEM = Mma_tile::M_PER_MMA_PER_CTA * Cta_tile::WARPS_N * sizeof(float)
    };

    enum
    {
        ELTS_PER_ROW = 2
    };

    static_assert(Cta_tile::WARPS_N == 2);
    static_assert(Cta_tile::WARPS_M == 4);
    static_assert(Mma_tile::M_PER_MMA_PER_CTA == 64);

    template <typename Params>
    inline __device__ Softmax_gmma_base(Params const& params, void* smem, int const bidb, int const tidx)
        : Base(params, smem, bidb, tidx)
    {

        int const warp = tidx / Cta_tile::THREADS_PER_WARP;
        int const warp_n = warp / 4;
        int const warp_m = warp % 4;
        int const lane = tidx % Cta_tile::THREADS_PER_WARP;
        int const quad = lane / 4;
        is_writer_ = lane % 4 == 0;

        int const col = warp_n;
        int const row = warp_m * 16 + quad;

        smem_write_ = static_cast<float*>(smem) + row * 2 + col;
        smem_read_ = static_cast<float2*>(smem) + row;
    }

    // Do a CTA-wide reduction.
    template <typename Functor>
    inline __device__ void reduce(float (&dst)[2])
    {
        Base::template reduce_4x1<Functor>(dst);
        if (is_writer_)
        {
            smem_write_[0 * ELTS_PER_ROW] = dst[0];
            smem_write_[8 * ELTS_PER_ROW] = dst[1];
        }
        __syncthreads();
        float2 tmp0 = smem_read_[0];
        float2 tmp1 = smem_read_[8];
        dst[0] = Functor::apply(tmp0.x, tmp0.y);
        dst[1] = Functor::apply(tmp1.x, tmp1.y);
    }

    float* smem_write_;
    float2* smem_read_;
    bool is_writer_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M, int GMMA_N, int GMMA_K, bool GMMA_A_RF, bool GMMA_B_RF, typename Cta_tile_,
    typename Kernel_traits_>
struct Softmax<fmha::Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile_, Kernel_traits_>
    : public Softmax_gmma_base<fmha::Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile_,
          Kernel_traits_, Cta_tile_::WARPS_N>
{

    // The traits.
    using Traits = fmha::Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>;
    // Cta_tile.
    using Cta_tile = Cta_tile_;
    // Kernel_traits.
    using Kernel_traits = Kernel_traits_;
    // The Base class.
    using Base = Softmax_gmma_base<Traits, Cta_tile, Kernel_traits, Cta_tile::WARPS_N>;
    // The accumulators.
    using Accumulator = typename Base::Accumulator;
    // The Mma tile.
    using Mma_tile = typename Base::Mma_tile;

    // The number of MMAs in M/N dimensions.
    enum
    {
        MMAS_M = Mma_tile::MMAS_M
    };

    enum
    {
        MMAS_N = Mma_tile::MMAS_N
    };

    // for HGMMA_FP16, there are 2 elements per RF for ACC.
    enum
    {
        ELTS_PER_THREAD = 2
    };

    // for Hopper HGMMA, each row is held within 4 threads.
    enum
    {
        THREADS_PER_ROW = 4
    };

    // The number of rows accessed by each thread.
    enum
    {
        ROWS_PER_THREAD = Traits::GMMA_M / (Cta_tile::THREADS_PER_WARP / THREADS_PER_ROW) / Cta_tile::WARPS_M
    };

    // The number of columns access by each thread.
    // Note there are 2 elements per reg.
    enum
    {
        COLS_PER_THREAD = Traits::GMMA_N / THREADS_PER_ROW / ELTS_PER_THREAD
    };

    // Use BMM1 softcapping scale or not.
    enum
    {
        ENABLE_BMM1_SOFTCAPPING_SCALE = Kernel_traits::ENABLE_BMM1_SOFTCAPPING_SCALE
    };

    // Ctor.
    template <typename Params>
    inline __device__ Softmax(Params const& params, void* smem, int bidb, int tidx)
        : Base(params, smem, bidb, tidx)
    {
    }

    // Convert from FP16 fragments to floats.
    inline __device__ void unpack(Accumulator const (&acc)[MMAS_M][MMAS_N])
    {
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {
#pragma unroll
                for (int col_idx = 0; col_idx < COLS_PER_THREAD; ++col_idx)
                {
#pragma unroll
                    for (int row_idx = 0; row_idx < ROWS_PER_THREAD; ++row_idx)
                    {
                        // the order of the acc rf is we traverse vertically first
                        // then we traverse horizontally.

                        // Normalize the values.
                        uint32_t acc_0 = fmha::hmul2(
                            acc[mi][ni].reg(col_idx * ROWS_PER_THREAD + row_idx), this->params_scale_bmm1_);
                        // Element index.
                        int elt_row_idx = ROWS_PER_THREAD * mi + row_idx;
                        int elt_col_idx = COLS_PER_THREAD * ELTS_PER_THREAD * ni + col_idx * ELTS_PER_THREAD;
                        // Extract the values as floats.
                        half2_to_float2(
                            this->elt_[elt_row_idx][elt_col_idx + 0], this->elt_[elt_row_idx][elt_col_idx + 1], acc_0);
                        // Attention logit softcapping scale.
                        // 1.0f / softcapping_scale has been fused to scale_bmm1.
                        if constexpr (ENABLE_BMM1_SOFTCAPPING_SCALE)
                        {
                            this->elt_[elt_row_idx][elt_col_idx + 0] = this->params_softcapping_scale_bmm1_
                                * __tanhf(this->elt_[elt_row_idx][elt_col_idx + 0]);
                            this->elt_[elt_row_idx][elt_col_idx + 1] = this->params_softcapping_scale_bmm1_
                                * __tanhf(this->elt_[elt_row_idx][elt_col_idx + 1]);
                        }
                    } // row_idx
                }     // col_idx
            }         // ni
        }             // mi
    }

    // Store the tile after softmax.
    template <typename Gmem_tile>
    inline __device__ void store(Gmem_tile& gmem_tile)
    {
        Accumulator acc[MMAS_M][MMAS_N];

#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {
#pragma unroll
                for (int col_idx = 0; col_idx < COLS_PER_THREAD; ++col_idx)
                {
#pragma unroll
                    for (int row_idx = 0; row_idx < ROWS_PER_THREAD; ++row_idx)
                    {
                        // the order of the acc rf is we traverse vertically first
                        // then we traverse horizontally.
                        float tmp_00 = this->elt_[ROWS_PER_THREAD * mi + row_idx][COLS_PER_THREAD * ELTS_PER_THREAD * ni
                            + col_idx * ELTS_PER_THREAD + 0];
                        float tmp_01 = this->elt_[ROWS_PER_THREAD * mi + row_idx][COLS_PER_THREAD * ELTS_PER_THREAD * ni
                            + col_idx * ELTS_PER_THREAD + 1];
                        acc[mi][ni].reg(col_idx * ROWS_PER_THREAD + row_idx) = fmha::float2_to_half2(tmp_00, tmp_01);
                    } // row_idx
                }     // col_idx
            }         // ni
        }             // m

        // Delegate to the gmem tile to store.
        gmem_tile.store(acc);
    }

    // Pack the data to a fragment for the next GEMM.
    template <typename Fragment_a, int K, int M>
    inline __device__ void pack(Fragment_a (&dst)[K][M]) const
    {

// we know the instruction shape is 64xNx16
// Thus for input A matrix, it is of size 64x16 per warpgroup.
// Thus, each threads access 2 rows and 4 columns. contiguous 2 columns are held by 1 RF.
#pragma unroll
        for (int mi = 0; mi < M; ++mi)
        {
#pragma unroll
            for (int ki = 0; ki < K; ++ki)
            {
                // 1st row - 4 elements per row.
                float tmp_00 = this->elt_[2 * mi + 0][4 * ki + 0];
                float tmp_01 = this->elt_[2 * mi + 0][4 * ki + 1];
                float tmp_02 = this->elt_[2 * mi + 0][4 * ki + 2];
                float tmp_03 = this->elt_[2 * mi + 0][4 * ki + 3];

                // 2nd row - 4 elements per row.
                float tmp_10 = this->elt_[2 * mi + 1][4 * ki + 0];
                float tmp_11 = this->elt_[2 * mi + 1][4 * ki + 1];
                float tmp_12 = this->elt_[2 * mi + 1][4 * ki + 2];
                float tmp_13 = this->elt_[2 * mi + 1][4 * ki + 3];

                // Pack to 4 registers.
                dst[ki][mi].reg(0) = fmha::float2_to_half2(tmp_00, tmp_01);
                dst[ki][mi].reg(1) = fmha::float2_to_half2(tmp_10, tmp_11);
                dst[ki][mi].reg(2) = fmha::float2_to_half2(tmp_02, tmp_03);
                dst[ki][mi].reg(3) = fmha::float2_to_half2(tmp_12, tmp_13);
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M, int GMMA_N, int GMMA_K, bool GMMA_A_RF, bool GMMA_B_RF, typename Cta_tile_,
    typename Kernel_traits_>
struct Softmax<fmha::Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile_, Kernel_traits_>
    : public Softmax_gmma_base<fmha::Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile_,
          Kernel_traits_, Cta_tile_::WARPS_N>
{

    // The traits.
    using Traits = fmha::Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>;
    // Cta_tile.
    using Cta_tile = Cta_tile_;
    // Kernel_traits.
    using Kernel_traits = Kernel_traits_;
    // The Base class.
    using Base = Softmax_gmma_base<Traits, Cta_tile, Kernel_traits, Cta_tile::WARPS_N>;
    // The accumulators.
    using Accumulator = typename Base::Accumulator;
    // The Mma tile.
    using Mma_tile = typename Base::Mma_tile;

    // The number of MMAs in M/N dimensions.
    enum
    {
        MMAS_M = Mma_tile::MMAS_M
    };

    enum
    {
        MMAS_N = Mma_tile::MMAS_N
    };

    // for HGMMA_FP16, there are 2 elements per RF for ACC.
    enum
    {
        ELTS_PER_THREAD = 2
    };

    // for Hopper HGMMA, each row is held within 4 threads.
    enum
    {
        THREADS_PER_ROW = 4
    };

    // The number of rows accessed by each thread.
    enum
    {
        ROWS_PER_THREAD = Traits::GMMA_M / (Cta_tile::THREADS_PER_WARP / THREADS_PER_ROW) / Cta_tile::WARPS_M
    };

    // The number of columns access by each thread.
    // Note there are 2 elements per reg.
    enum
    {
        COLS_PER_THREAD = Traits::GMMA_N / THREADS_PER_ROW / ELTS_PER_THREAD
    };

    // Use BMM1 softcapping scale or not.
    enum
    {
        ENABLE_BMM1_SOFTCAPPING_SCALE = Kernel_traits::ENABLE_BMM1_SOFTCAPPING_SCALE
    };

    // Ctor.
    template <typename Params>
    inline __device__ Softmax(Params const& params, void* smem, int bidb, int tidx)
        : Base(params, smem, bidb, tidx)
    {
    }

    // Convert from FP16 fragments to floats.
    inline __device__ void unpack(Accumulator const (&acc)[MMAS_M][MMAS_N])
    {

        float const& scale_f = reinterpret_cast<float const&>(this->params_scale_bmm1_);
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {
#pragma unroll
                for (int col_idx = 0; col_idx < COLS_PER_THREAD; ++col_idx)
                {
#pragma unroll
                    for (int row_idx = 0; row_idx < ROWS_PER_THREAD; ++row_idx)
                    {
                        // the order of the acc rf is we traverse vertically first
                        // then we traverse horizontally.
                        int elt_row = ROWS_PER_THREAD * mi + row_idx;
                        int elt_col = COLS_PER_THREAD * ELTS_PER_THREAD * ni + col_idx * ELTS_PER_THREAD;

                        float elt0 = acc[mi][ni].elt(col_idx * 2 * ROWS_PER_THREAD + 2 * row_idx + 0) * scale_f;
                        float elt1 = acc[mi][ni].elt(col_idx * 2 * ROWS_PER_THREAD + 2 * row_idx + 1) * scale_f;

                        // 1.0f / softcapping_scale has been fused to scale_bmm1.
                        if constexpr (ENABLE_BMM1_SOFTCAPPING_SCALE)
                        {
                            elt0 = this->params_softcapping_scale_bmm1_ * __tanhf(elt0);
                            elt1 = this->params_softcapping_scale_bmm1_ * __tanhf(elt1);
                        }

                        this->elt_[elt_row][elt_col + 0] = elt0;
                        this->elt_[elt_row][elt_col + 1] = elt1;

                    } // row_idx
                }     // col_idx
            }         // ni
        }             // mi
    }

    // Store the tile after softmax.
    template <typename Gmem_tile>
    inline __device__ void store(Gmem_tile& gmem_tile)
    {
        Accumulator acc[MMAS_M][MMAS_N];

#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {
#pragma unroll
                for (int col_idx = 0; col_idx < COLS_PER_THREAD; ++col_idx)
                {
#pragma unroll
                    for (int row_idx = 0; row_idx < ROWS_PER_THREAD; ++row_idx)
                    {
                        // the order of the acc rf is we traverse vertically first
                        // then we traverse horizontally
                        int elt_row = ROWS_PER_THREAD * mi + row_idx;
                        int elt_col = COLS_PER_THREAD * ELTS_PER_THREAD * ni + col_idx * ELTS_PER_THREAD;
                        float elt0 = this->elt_[elt_row][elt_col + 0];
                        float elt1 = this->elt_[elt_row][elt_col + 1];

                        acc[mi][ni].elt(col_idx * 2 * ROWS_PER_THREAD + 2 * row_idx + 0) = elt0;
                        acc[mi][ni].elt(col_idx * 2 * ROWS_PER_THREAD + 2 * row_idx + 1) = elt1;
                    } // row_idx
                }     // col_idx
            }         // ni
        }             // m

        // Delegate to the gmem tile to store.
        gmem_tile.store(acc);
    }

    // Pack the data to a fragment for the next GEMM.
    template <typename Fragment_a, int K, int M>
    inline __device__ void pack(Fragment_a (&dst)[K][M]) const
    {

// we know the instruction shape is 64xNx16
// Thus for input A matrix, it is of size 64x16 per warpgroup.
// Thus, each threads access 2 rows and 4 columns. contiguous 2 columns are held by 1 RF.
#pragma unroll
        for (int mi = 0; mi < M; ++mi)
        {
#pragma unroll
            for (int ki = 0; ki < K; ++ki)
            {
                // 1st row - 4 elements per row.
                float tmp_00 = this->elt_[2 * mi + 0][4 * ki + 0];
                float tmp_01 = this->elt_[2 * mi + 0][4 * ki + 1];
                float tmp_02 = this->elt_[2 * mi + 0][4 * ki + 2];
                float tmp_03 = this->elt_[2 * mi + 0][4 * ki + 3];

                // 2nd row - 4 elements per row.
                float tmp_10 = this->elt_[2 * mi + 1][4 * ki + 0];
                float tmp_11 = this->elt_[2 * mi + 1][4 * ki + 1];
                float tmp_12 = this->elt_[2 * mi + 1][4 * ki + 2];
                float tmp_13 = this->elt_[2 * mi + 1][4 * ki + 3];

                // Pack to 4 registers.
                dst[ki][mi].reg(0) = fmha::float2_to_half2(tmp_00, tmp_01);
                dst[ki][mi].reg(1) = fmha::float2_to_half2(tmp_10, tmp_11);
                dst[ki][mi].reg(2) = fmha::float2_to_half2(tmp_02, tmp_03);
                dst[ki][mi].reg(3) = fmha::float2_to_half2(tmp_12, tmp_13);
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M, int GMMA_N, int GMMA_K, bool GMMA_A_RF, bool GMMA_B_RF, typename Cta_tile_,
    typename Kernel_traits_>
struct Softmax<fmha::Hopper_hgmma_bf16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile_, Kernel_traits_>
    : public Softmax_gmma_base<fmha::Hopper_hgmma_bf16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile_,
          Kernel_traits_, Cta_tile_::WARPS_N>
{

    // The traits.
    using Traits = fmha::Hopper_hgmma_bf16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>;
    // Cta_tile.
    using Cta_tile = Cta_tile_;
    // Kernel_traits.
    using Kernel_traits = Kernel_traits_;
    // The Base class.
    using Base = Softmax_gmma_base<Traits, Cta_tile, Kernel_traits, Cta_tile::WARPS_N>;
    // The accumulators.
    using Accumulator = typename Base::Accumulator;
    // The Mma tile.
    using Mma_tile = typename Base::Mma_tile;

    // The number of MMAs in M/N dimensions.
    enum
    {
        MMAS_M = Mma_tile::MMAS_M
    };

    enum
    {
        MMAS_N = Mma_tile::MMAS_N
    };

    // for HGMMA_FP16, there are 2 elements per RF for ACC.
    enum
    {
        ELTS_PER_THREAD = 2
    };

    // for Hopper HGMMA, each row is held within 4 threads.
    enum
    {
        THREADS_PER_ROW = 4
    };

    // The number of rows accessed by each thread.
    enum
    {
        ROWS_PER_THREAD = Traits::GMMA_M / (Cta_tile::THREADS_PER_WARP / THREADS_PER_ROW) / Cta_tile::WARPS_M
    };

    // The number of columns access by each thread.
    // Note there are 2 elements per reg.
    enum
    {
        COLS_PER_THREAD = Traits::GMMA_N / THREADS_PER_ROW / ELTS_PER_THREAD
    };

    // Use BMM1 softcapping scale or not.
    enum
    {
        ENABLE_BMM1_SOFTCAPPING_SCALE = Kernel_traits::ENABLE_BMM1_SOFTCAPPING_SCALE
    };

    // Ctor.
    template <typename Params>
    inline __device__ Softmax(Params const& params, void* smem, int bidb, int tidx)
        : Base(params, smem, bidb, tidx)
    {
    }

    // Convert from FP16 fragments to floats.
    inline __device__ void unpack(Accumulator const (&acc)[MMAS_M][MMAS_N])
    {

        float const& scale_f = reinterpret_cast<float const&>(this->params_scale_bmm1_);
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {
#pragma unroll
                for (int col_idx = 0; col_idx < COLS_PER_THREAD; ++col_idx)
                {
#pragma unroll
                    for (int row_idx = 0; row_idx < ROWS_PER_THREAD; ++row_idx)
                    {
                        // the order of the acc rf is we traverse vertically first
                        // then we traverse horizontally.
                        int elt_row = ROWS_PER_THREAD * mi + row_idx;
                        int elt_col = COLS_PER_THREAD * ELTS_PER_THREAD * ni + col_idx * ELTS_PER_THREAD;

                        float elt0 = acc[mi][ni].elt(col_idx * 2 * ROWS_PER_THREAD + 2 * row_idx + 0) * scale_f;
                        float elt1 = acc[mi][ni].elt(col_idx * 2 * ROWS_PER_THREAD + 2 * row_idx + 1) * scale_f;

                        if constexpr (ENABLE_BMM1_SOFTCAPPING_SCALE)
                        {
                            elt0 = this->params_softcapping_scale_bmm1_ * __tanhf(elt0);
                            elt1 = this->params_softcapping_scale_bmm1_ * __tanhf(elt1);
                        }

                        this->elt_[elt_row][elt_col + 0] = elt0;
                        this->elt_[elt_row][elt_col + 1] = elt1;

                    } // row_idx
                }     // col_idx
            }         // ni
        }             // mi
    }

    // Store the tile after softmax.
    template <typename Gmem_tile>
    inline __device__ void store(Gmem_tile& gmem_tile)
    {
        Accumulator acc[MMAS_M][MMAS_N];

#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {
#pragma unroll
                for (int col_idx = 0; col_idx < COLS_PER_THREAD; ++col_idx)
                {
#pragma unroll
                    for (int row_idx = 0; row_idx < ROWS_PER_THREAD; ++row_idx)
                    {
                        // the order of the acc rf is we traverse vertically first
                        // then we traverse horizontally.
                        int elt_row = ROWS_PER_THREAD * mi + row_idx;
                        int elt_col = COLS_PER_THREAD * ELTS_PER_THREAD * ni + col_idx * ELTS_PER_THREAD;
                        float elt0 = this->elt_[elt_row][elt_col + 0];
                        float elt1 = this->elt_[elt_row][elt_col + 1];

                        acc[mi][ni].elt(col_idx * 2 * ROWS_PER_THREAD + 2 * row_idx + 0) = elt0;
                        acc[mi][ni].elt(col_idx * 2 * ROWS_PER_THREAD + 2 * row_idx + 1) = elt1;
                    } // row_idx
                }     // col_idx
            }         // ni
        }             // m

        // Delegate to the gmem tile to store.
        gmem_tile.store(acc);
    }

    // Pack the data to a fragment for the next GEMM.
    template <typename Fragment_a, int K, int M>
    inline __device__ void pack(Fragment_a (&dst)[K][M]) const
    {

// we know the instruction shape is 64xNx16
// Thus for input A matrix, it is of size 64x16 per warpgroup.
// Thus, each threads access 2 rows and 4 columns. contiguous 2 columns are held by 1 RF.
#pragma unroll
        for (int mi = 0; mi < M; ++mi)
        {
#pragma unroll
            for (int ki = 0; ki < K; ++ki)
            {
                // 1st row - 4 elements per row.
                float tmp_00 = this->elt_[2 * mi + 0][4 * ki + 0];
                float tmp_01 = this->elt_[2 * mi + 0][4 * ki + 1];
                float tmp_02 = this->elt_[2 * mi + 0][4 * ki + 2];
                float tmp_03 = this->elt_[2 * mi + 0][4 * ki + 3];

                // 2nd row - 4 elements per row.
                float tmp_10 = this->elt_[2 * mi + 1][4 * ki + 0];
                float tmp_11 = this->elt_[2 * mi + 1][4 * ki + 1];
                float tmp_12 = this->elt_[2 * mi + 1][4 * ki + 2];
                float tmp_13 = this->elt_[2 * mi + 1][4 * ki + 3];

                // Pack to 4 registers.
                dst[ki][mi].reg(0) = fmha::float2_to_bf16_x2(tmp_00, tmp_01);
                dst[ki][mi].reg(1) = fmha::float2_to_bf16_x2(tmp_10, tmp_11);
                dst[ki][mi].reg(2) = fmha::float2_to_bf16_x2(tmp_02, tmp_03);
                dst[ki][mi].reg(3) = fmha::float2_to_bf16_x2(tmp_12, tmp_13);
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, typename Kernel_traits>
struct Softmax_gmma_32bit_8bit_base : public Softmax_gmma_base<Traits, Cta_tile, Kernel_traits, Cta_tile::WARPS_N>
{

    // The Base class.
    using Base = Softmax_gmma_base<Traits, Cta_tile, Kernel_traits, Cta_tile::WARPS_N>;
    // The accumulators.
    using Accumulator = typename Base::Accumulator;
    // The Mma tile.
    using Mma_tile = typename Base::Mma_tile;

    // The number of MMAs in M/N dimensions.
    enum
    {
        MMAS_M = Mma_tile::MMAS_M
    };

    enum
    {
        MMAS_N = Mma_tile::MMAS_N
    };

    // TODO these should be general.
    // Two elts per thread per acc core matrix.
    enum
    {
        ELTS_PER_THREAD = 2
    };

    // Number of threads per row of the acc core matrix.
    enum
    {
        THREADS_PER_ROW = 4
    };

    // The number of rows accessed by each thread per GMMA.
    enum
    {
        ROWS_PER_THREAD = Traits::GMMA_M / (Cta_tile::THREADS_PER_WARP / THREADS_PER_ROW) / Cta_tile::WARPS_M
    };

    // The number of columns access by each thread.
    enum
    {
        COLS_PER_THREAD = Traits::GMMA_N / THREADS_PER_ROW / ELTS_PER_THREAD
    };

    // Check the expected number of accumulator elements.
    static_assert(Accumulator::NUM_ELTS == COLS_PER_THREAD * ROWS_PER_THREAD * ELTS_PER_THREAD);

    // Ctor.
    template <typename Params>
    inline __device__ Softmax_gmma_32bit_8bit_base(Params const& params, void* smem, int bidb, int tidx)
        : Base(params, smem, bidb, tidx)
    {
    }

    inline __device__ void unpack(Accumulator const (&acc)[MMAS_M][MMAS_N])
    {
        float const scalef = reinterpret_cast<float const&>(this->params_scale_bmm1_);
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {
#pragma unroll
                for (int ii = 0; ii < COLS_PER_THREAD; ++ii)
                {
                    float tmp_00
                        = acc[mi][ni].elt(ii * ROWS_PER_THREAD * ELTS_PER_THREAD + 0 * ELTS_PER_THREAD + 0) * scalef;
                    float tmp_01
                        = acc[mi][ni].elt(ii * ROWS_PER_THREAD * ELTS_PER_THREAD + 0 * ELTS_PER_THREAD + 1) * scalef;
                    float tmp_10
                        = acc[mi][ni].elt(ii * ROWS_PER_THREAD * ELTS_PER_THREAD + 1 * ELTS_PER_THREAD + 0) * scalef;
                    float tmp_11
                        = acc[mi][ni].elt(ii * ROWS_PER_THREAD * ELTS_PER_THREAD + 1 * ELTS_PER_THREAD + 1) * scalef;
                    int n_offset = ni * COLS_PER_THREAD * ELTS_PER_THREAD + ii * ELTS_PER_THREAD;
                    this->elt_[mi * ROWS_PER_THREAD + 0][n_offset + 0] = tmp_00;
                    this->elt_[mi * ROWS_PER_THREAD + 0][n_offset + 1] = tmp_01;
                    this->elt_[mi * ROWS_PER_THREAD + 1][n_offset + 0] = tmp_10;
                    this->elt_[mi * ROWS_PER_THREAD + 1][n_offset + 1] = tmp_11;
                } // ii
            }     // ni
        }         // mi
    }

    inline __device__ void unpack_noscale(Accumulator const (&acc)[MMAS_M][MMAS_N])
    {
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {
#pragma unroll
                for (int ii = 0; ii < COLS_PER_THREAD; ++ii)
                {
                    float tmp_00 = acc[mi][ni].elt(ii * ROWS_PER_THREAD * ELTS_PER_THREAD + 0 * ELTS_PER_THREAD + 0);
                    float tmp_01 = acc[mi][ni].elt(ii * ROWS_PER_THREAD * ELTS_PER_THREAD + 0 * ELTS_PER_THREAD + 1);
                    float tmp_10 = acc[mi][ni].elt(ii * ROWS_PER_THREAD * ELTS_PER_THREAD + 1 * ELTS_PER_THREAD + 0);
                    float tmp_11 = acc[mi][ni].elt(ii * ROWS_PER_THREAD * ELTS_PER_THREAD + 1 * ELTS_PER_THREAD + 1);
                    int n_offset = ni * COLS_PER_THREAD * ELTS_PER_THREAD + ii * ELTS_PER_THREAD;
                    this->elt_[mi * ROWS_PER_THREAD + 0][n_offset + 0] = tmp_00;
                    this->elt_[mi * ROWS_PER_THREAD + 0][n_offset + 1] = tmp_01;
                    this->elt_[mi * ROWS_PER_THREAD + 1][n_offset + 0] = tmp_10;
                    this->elt_[mi * ROWS_PER_THREAD + 1][n_offset + 1] = tmp_11;
                } // ii
            }     // ni
        }         // mi
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M, int GMMA_N, int GMMA_K, bool GMMA_A_RF, bool GMMA_B_RF, typename Cta_tile, typename Kernel_traits>
struct Softmax<fmha::Hopper_qgmma_e4m3_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile,
    Kernel_traits>
    : public Softmax_gmma_32bit_8bit_base<
          fmha::Hopper_qgmma_e4m3_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile, Kernel_traits>
{

    // The traits.
    using Traits = fmha::Hopper_qgmma_e4m3_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>;
    // The Base class.
    using Base = Softmax_gmma_32bit_8bit_base<Traits, Cta_tile, Kernel_traits>;

    using Accumulator = typename Base::Accumulator;

    enum
    {
        MMAS_M = Base::MMAS_M,
        MMAS_N = Base::MMAS_N,
        ROWS_PER_THREAD = Base::ROWS_PER_THREAD,
        COLS_PER_THREAD = Base::COLS_PER_THREAD,
        ELTS_PER_THREAD = Base::ELTS_PER_THREAD,
    };

    // Ctor.
    template <typename Params>
    inline __device__ Softmax(Params const& params, void* smem, int bidb, int tidx)
        : Base(params, smem, bidb, tidx)
        , params_scale_softmax_(params.scale_softmax)
    {
    }

    // Store the tile after softmax.
    template <typename Gmem_tile>
    inline __device__ void store(Gmem_tile& gmem_tile)
    {

        float const scale = reinterpret_cast<float const&>(this->params_scale_softmax_);

        Accumulator acc[MMAS_M][MMAS_N];

#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {
#pragma unroll
                for (int ii = 0; ii < COLS_PER_THREAD; ++ii)
                {
                    int row = mi * ROWS_PER_THREAD;
                    int col = ni * COLS_PER_THREAD * ELTS_PER_THREAD + ii * ELTS_PER_THREAD;
                    float tmp_00 = this->elt_[row + 0][col + 0] * scale;
                    float tmp_01 = this->elt_[row + 0][col + 1] * scale;
                    float tmp_10 = this->elt_[row + 1][col + 0] * scale;
                    float tmp_11 = this->elt_[row + 1][col + 1] * scale;

                    int elt_idx = ii * ROWS_PER_THREAD * ELTS_PER_THREAD;
                    acc[mi][ni].elt(elt_idx + 0 * ELTS_PER_THREAD + 0) = tmp_00;
                    acc[mi][ni].elt(elt_idx + 0 * ELTS_PER_THREAD + 1) = tmp_01;
                    acc[mi][ni].elt(elt_idx + 1 * ELTS_PER_THREAD + 0) = tmp_10;
                    acc[mi][ni].elt(elt_idx + 1 * ELTS_PER_THREAD + 1) = tmp_11;
                } // ii
            }     // ni
        }         // mi

        // Delegate to the gmem tile to store.
        gmem_tile.store(acc);
    }

    // Pack the data to a fragment for the next GEMM.
    template <typename Fragment_a, int K, int M>
    inline __device__ void pack(Fragment_a (&dst)[K][M]) const
    {
        static_assert(M == 1);
        static_assert(Fragment_a::NUM_REGS == 4);
        static_assert(Fragment_a::NUM_ELTS == 16);
        // Acc per warp: 16 x 256 FP32
        // A is 8 times(in K) 16 x 32 FP8, i.e. 4 registers per thread.

        static_assert(MMAS_N * COLS_PER_THREAD * ELTS_PER_THREAD % 8 == 0);
        static_assert(MMAS_N * COLS_PER_THREAD * ELTS_PER_THREAD == K * Fragment_a::NUM_ELTS / 2);

        float const scale = reinterpret_cast<float const&>(this->params_scale_softmax_);

// The canonical layout in K should be R0: [0,1,2,3] R2: [16,17,18,19]
// Note below that this is not possible with the register layout of the accumulator.
#pragma unroll
        for (int mi = 0; mi < M; ++mi)
        {
#pragma unroll
            for (int ki = 0; ki < K; ++ki)
            {
                // 1st row - 8 elements per row.
                float tmp_00 = this->elt_[2 * mi + 0][8 * ki + 0] * scale; // + 0
                float tmp_01 = this->elt_[2 * mi + 0][8 * ki + 1] * scale; // + 1
                float tmp_02 = this->elt_[2 * mi + 0][8 * ki + 2] * scale; // + 8
                float tmp_03 = this->elt_[2 * mi + 0][8 * ki + 3] * scale; // + 9
                float tmp_04 = this->elt_[2 * mi + 0][8 * ki + 4] * scale; // +16
                float tmp_05 = this->elt_[2 * mi + 0][8 * ki + 5] * scale; // +17
                float tmp_06 = this->elt_[2 * mi + 0][8 * ki + 6] * scale; // +24
                float tmp_07 = this->elt_[2 * mi + 0][8 * ki + 7] * scale; // +25

                // 2nd row - 4 elements per row.
                float tmp_10 = this->elt_[2 * mi + 1][8 * ki + 0] * scale; // + 0
                float tmp_11 = this->elt_[2 * mi + 1][8 * ki + 1] * scale; // + 1
                float tmp_12 = this->elt_[2 * mi + 1][8 * ki + 2] * scale; // + 8
                float tmp_13 = this->elt_[2 * mi + 1][8 * ki + 3] * scale; // + 9
                float tmp_14 = this->elt_[2 * mi + 1][8 * ki + 4] * scale; // +16
                float tmp_15 = this->elt_[2 * mi + 1][8 * ki + 5] * scale; // +17
                float tmp_16 = this->elt_[2 * mi + 1][8 * ki + 6] * scale; // +24
                float tmp_17 = this->elt_[2 * mi + 1][8 * ki + 7] * scale; // +25

                // Pack to 4 registers.
                dst[ki][mi].reg(0) = fmha::float4_to_fp8x4<Traits::A_type>(tmp_00, tmp_01, tmp_02, tmp_03);
                dst[ki][mi].reg(1) = fmha::float4_to_fp8x4<Traits::A_type>(tmp_10, tmp_11, tmp_12, tmp_13);
                dst[ki][mi].reg(2) = fmha::float4_to_fp8x4<Traits::A_type>(tmp_04, tmp_05, tmp_06, tmp_07);
                dst[ki][mi].reg(3) = fmha::float4_to_fp8x4<Traits::A_type>(tmp_14, tmp_15, tmp_16, tmp_17);
            }
        }
    }

    uint32_t const params_scale_softmax_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M, int GMMA_N, int GMMA_K, bool GMMA_A_RF, bool GMMA_B_RF, typename Cta_tile, typename Kernel_traits>
struct Softmax<fmha::Hopper_igmma_int8_int32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile,
    Kernel_traits>
    : public Softmax_gmma_32bit_8bit_base<
          fmha::Hopper_igmma_int8_int32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile, Kernel_traits>
{

    // The traits.
    using Traits = fmha::Hopper_igmma_int8_int32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>;

    // The Base class.
    using Base = Softmax_gmma_32bit_8bit_base<Traits, Cta_tile, Kernel_traits>;

    using Accumulator = typename Base::Accumulator;

    enum
    {
        MMAS_M = Base::MMAS_M,
        MMAS_N = Base::MMAS_N,
        ROWS_PER_THREAD = Base::ROWS_PER_THREAD,
        COLS_PER_THREAD = Base::COLS_PER_THREAD,
        ELTS_PER_THREAD = Base::ELTS_PER_THREAD,
    };

    // Ctor.
    template <typename Params>
    inline __device__ Softmax(Params const& params, void* smem, int bidb, int tidx)
        : Base(params, smem, bidb, tidx)
        , params_scale_softmax_(params.scale_softmax)
    {
    }

    // Store the tile after softmax.
    template <typename Gmem_tile>
    inline __device__ void store(Gmem_tile& gmem_tile)
    {

        float const scale = reinterpret_cast<float const&>(this->params_scale_softmax_);
        Accumulator acc[MMAS_M][MMAS_N];

#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {
#pragma unroll
                for (int ii = 0; ii < COLS_PER_THREAD; ++ii)
                {
                    int n_offset = ni * COLS_PER_THREAD * ELTS_PER_THREAD + ii * ELTS_PER_THREAD;
                    float tmp_00 = this->elt_[mi * ROWS_PER_THREAD + 0][n_offset + 0];
                    float tmp_01 = this->elt_[mi * ROWS_PER_THREAD + 0][n_offset + 1];
                    float tmp_10 = this->elt_[mi * ROWS_PER_THREAD + 1][n_offset + 0];
                    float tmp_11 = this->elt_[mi * ROWS_PER_THREAD + 1][n_offset + 1];

                    int elt_offset = ii * ROWS_PER_THREAD * ELTS_PER_THREAD;
                    acc[mi][ni].elt(elt_offset + 0 * ELTS_PER_THREAD + 0) = tmp_00 * scale;
                    acc[mi][ni].elt(elt_offset + 0 * ELTS_PER_THREAD + 1) = tmp_01 * scale;
                    acc[mi][ni].elt(elt_offset + 1 * ELTS_PER_THREAD + 0) = tmp_10 * scale;
                    acc[mi][ni].elt(elt_offset + 1 * ELTS_PER_THREAD + 1) = tmp_11 * scale;
                } // ii
            }     // ni
        }         // mi

        // Delegate to the gmem tile to store.
        gmem_tile.store(acc);
    }

    // Pack the data to a fragment for the next GEMM.
    template <typename Fragment_a, int K, int M>
    inline __device__ void pack(Fragment_a (&dst)[K][M]) const
    {
        static_assert(M == 1);
        static_assert(Fragment_a::NUM_REGS == 4);
        static_assert(Fragment_a::NUM_ELTS == 16);
        // Acc per warp: 16 x 256 FP32
        // A is 8 times(in K) 16 x 32 FP8, i.e. 4 registers per thread.

        static_assert(MMAS_N * COLS_PER_THREAD * ELTS_PER_THREAD % 8 == 0);
        static_assert(MMAS_N * COLS_PER_THREAD * ELTS_PER_THREAD == K * Fragment_a::NUM_ELTS / 2);

        float const scale = reinterpret_cast<float const&>(this->params_scale_softmax_);
// The canonical layout in K should be R0: [0,1,2,3] R2: [16,17,18,19]
// Note below that this is not possible with the register layout of the accumulator.
#pragma unroll
        for (int mi = 0; mi < M; ++mi)
        {
#pragma unroll
            for (int ki = 0; ki < K; ++ki)
            {
                // 1st row - 8 elements per row.
                float tmp_00 = this->elt_[2 * mi + 0][8 * ki + 0] * scale; // + 0
                float tmp_01 = this->elt_[2 * mi + 0][8 * ki + 1] * scale; // + 1
                float tmp_02 = this->elt_[2 * mi + 0][8 * ki + 2] * scale; // + 8
                float tmp_03 = this->elt_[2 * mi + 0][8 * ki + 3] * scale; // + 9
                float tmp_04 = this->elt_[2 * mi + 0][8 * ki + 4] * scale; // +16
                float tmp_05 = this->elt_[2 * mi + 0][8 * ki + 5] * scale; // +17
                float tmp_06 = this->elt_[2 * mi + 0][8 * ki + 6] * scale; // +24
                float tmp_07 = this->elt_[2 * mi + 0][8 * ki + 7] * scale; // +25

                // 2nd row - 4 elements per row.
                float tmp_10 = this->elt_[2 * mi + 1][8 * ki + 0] * scale; // + 0
                float tmp_11 = this->elt_[2 * mi + 1][8 * ki + 1] * scale; // + 1
                float tmp_12 = this->elt_[2 * mi + 1][8 * ki + 2] * scale; // + 8
                float tmp_13 = this->elt_[2 * mi + 1][8 * ki + 3] * scale; // + 9
                float tmp_14 = this->elt_[2 * mi + 1][8 * ki + 4] * scale; // +16
                float tmp_15 = this->elt_[2 * mi + 1][8 * ki + 5] * scale; // +17
                float tmp_16 = this->elt_[2 * mi + 1][8 * ki + 6] * scale; // +24
                float tmp_17 = this->elt_[2 * mi + 1][8 * ki + 7] * scale; // +25

                // Pack to 4 registers.
                dst[ki][mi].reg(0) = fmha::float4_to_char4<false>(tmp_00, tmp_01, tmp_02, tmp_03);
                dst[ki][mi].reg(1) = fmha::float4_to_char4<false>(tmp_10, tmp_11, tmp_12, tmp_13);
                dst[ki][mi].reg(2) = fmha::float4_to_char4<false>(tmp_04, tmp_05, tmp_06, tmp_07);
                dst[ki][mi].reg(3) = fmha::float4_to_char4<false>(tmp_14, tmp_15, tmp_16, tmp_17);
            }
        }
    }

    uint32_t const params_scale_softmax_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// The softmax normalization statistics used by flash attention (l, m)
template <typename Traits, typename Cta_tile>
struct Softmax_statistics
{

    // The shape of the MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    // The number of MMAs in the M dimension.
    enum
    {
        MMAS_M = Mma_tile::MMAS_M
    };

    // Ctor.
    template <typename Params, typename Binfo>
    inline __device__ Softmax_statistics(Params const& params, void const* ptr, Binfo const& binfo, int tidx)
        : ptr_(reinterpret_cast<int8_t const*>(ptr))
        , seqlen_(binfo.actual_seqlen)
    {

        // The decomposition of the thread index into warp/lane.
        int warp = tidx / Cta_tile::THREADS_PER_WARP;
        int lane = tidx % Cta_tile::THREADS_PER_WARP;

        // The position of the the warp in the CTA.
        int warp_m = warp % Cta_tile::WARPS_M;

        // The position of the thread
        token_ = warp_m * Mma_tile::M_PER_MMA + lane / 4;

        // Compute the offset to the first token of the sequence.
        int64_t offset = binfo.bidb * params.h + binfo.bidh;
        // Move the pointer to the correct position.
        ptr_ += offset * params.lse_stride_in_bytes;
    }

    // Load the bias into registers (and expand).
    inline __device__ void load(int step)
    {
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ii = 0; ii < 2; ++ii)
            {
                // The index of the token.
                int token = token_;
                // At each iteration we jump over STEPQ elements.
                token += step * Cta_tile::M;
                // The extra offset inside the CTA.
                token += mi * Mma_tile::M_PER_MMA_PER_CTA + (ii & 0x1) * 8;

                // Fetch the value if the token is valid.
                float val = 0.0f;
                if (token < seqlen_)
                {
                    val = reinterpret_cast<float const*>(ptr_)[token];
                }
                lm_[2 * mi + ii] = val;
            }
        }
    }

    // The pointer to the bias.
    int8_t const* ptr_;
    // The length of the sequence.
    int const seqlen_;
    // The token that this thread is loading.
    int token_;
    // The bias after expansion.
    float lm_[MMAS_M * 2];
};

} // namespace fmha
