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

#include "fmha/traits.h"

namespace fmha
{

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, int FMHA_VERSION>
struct Mask
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile>
struct Mask<Traits, Cta_tile, 1>
{

    // The shape of the MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    // The number of MMAs in each dimension.
    enum
    {
        MMAS_M = Mma_tile::MMAS_M
    };

    enum
    {
        MMAS_N = Mma_tile::MMAS_N
    };

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Mask(Params const& params, Block_info const& block_info, int tidx)
    {

        // The pointer.
        packed_mask_ptr_ = reinterpret_cast<char const*>(params.packed_mask_ptr);
        // Take the head into account.
        packed_mask_ptr_ += block_info.bidb * params.packed_mask_stride_in_bytes;
        // The thread inside the CTA.
        packed_mask_ptr_ += tidx * sizeof(uint32_t);
    }

    // Load the mask into registers (and expand).
    inline __device__ void load(int it)
    {

        // One 32-bit integer per MMA.
        uint32_t packed_mask[MMAS_M];
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
            int offset = (it * MMAS_M + mi) * Cta_tile::THREADS_PER_CTA * sizeof(uint32_t);
            fmha::ldg(packed_mask[mi], packed_mask_ptr_ + offset);
        }

// Expand the mask.
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {
                mask_[2 * mi + 0][4 * ni + 0] = packed_mask[mi] & (1u << (8 * ni + 0));
                mask_[2 * mi + 0][4 * ni + 1] = packed_mask[mi] & (1u << (8 * ni + 1));
                mask_[2 * mi + 1][4 * ni + 0] = packed_mask[mi] & (1u << (8 * ni + 2));
                mask_[2 * mi + 1][4 * ni + 1] = packed_mask[mi] & (1u << (8 * ni + 3));
                mask_[2 * mi + 0][4 * ni + 2] = packed_mask[mi] & (1u << (8 * ni + 4));
                mask_[2 * mi + 0][4 * ni + 3] = packed_mask[mi] & (1u << (8 * ni + 5));
                mask_[2 * mi + 1][4 * ni + 2] = packed_mask[mi] & (1u << (8 * ni + 6));
                mask_[2 * mi + 1][4 * ni + 3] = packed_mask[mi] & (1u << (8 * ni + 7));
            }
        }
    }

    // Is a given position valid?
    inline __device__ bool is_valid(int mi, int ni, int ii, int jj) const
    {
        return mask_[mi * 2 + ii][ni * 4 + jj];
    }

    // The pointer to the mask.
    char const* packed_mask_ptr_;
    // The mask after expansion.
    bool mask_[MMAS_M * 2][MMAS_N * 4];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Mask<Volta_hmma_fp16_traits, Cta_tile, 1>
{

    // The instruction traits.
    using Traits = Volta_hmma_fp16_traits;
    // The shape of the MMA tile.
    using Mma_tile = typename Traits::Mma_tile<Cta_tile>;

    // The number of MMAs in each dimension.
    enum
    {
        MMAS_M = Mma_tile::MMAS_M
    };

    enum
    {
        MMAS_N = Mma_tile::MMAS_N
    };

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Mask(Params const& params, Block_info const& block_info, int tidx)
    {

        // The pointer.
        packed_mask_ptr_ = reinterpret_cast<char const*>(params.packed_mask_ptr);
        // Take the head into account.
        packed_mask_ptr_ += block_info.bidb * params.packed_mask_stride_in_bytes;
        // The thread inside the CTA.
        packed_mask_ptr_ += tidx * sizeof(uint32_t);
    }

    // Load the mask into registers (and expand).
    inline __device__ void load(int it)
    {

        // One 32-bit integer per MMA.
        uint32_t packed_mask[MMAS_M];
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
            int offset = (it * MMAS_M + mi) * Cta_tile::THREADS_PER_CTA * sizeof(uint32_t);
            fmha::ldg(packed_mask[mi], packed_mask_ptr_ + offset);
        }

// Expand the mask.
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ii = 0; ii < MMAS_N * 8; ++ii)
            {
                mask_[mi][ii] = packed_mask[mi] & (1u << ii);
            }
        }
    }

    // Is a given position valid?
    inline __device__ bool is_valid(int mi, int ni, int, int jj) const
    {
        return mask_[mi][ni * 8 + jj];
    }

    // The pointer to the mask.
    char const* packed_mask_ptr_;
    // The mask after expansion.
    bool mask_[MMAS_M][MMAS_N * 8];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile>
struct Mask<Traits, Cta_tile, 2>
{

    // That implementation works only when WARPS_K is 1.
    static_assert(Cta_tile::WARPS_K == 1, "");

    // The shape of the MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Mask(Params const& params, Block_info const& block_info, int tidx)
        : seqlen_(block_info.actual_seqlen)
        , col_loop_step_(0)
    {

        // The decomposition of the thread index into warp/lane.
        int warp = tidx / Cta_tile::THREADS_PER_WARP;
        int lane = tidx % Cta_tile::THREADS_PER_WARP;

        // The position of the warp.
        int warp_n = warp / Cta_tile::WARPS_M;
        // The position of the thread.
        col_ = block_info.bidn * Cta_tile::N + warp_n * 16 + lane % 4 * 2;
        col_init_ = col_;
    }

    // Is a given position valid?
    inline __device__ bool is_valid(int, int ni, int, int jj) const
    {

        // The position of the thread in the sequence.
        int offset = this->col_ + this->col_loop_step_ * Cta_tile::N + ni * Mma_tile::N_PER_MMA_PER_CTA;
        // The position inside the MMA.
        offset += (jj & 0x02) * 4 + (jj & 0x1);
        // Is it a valid position in the sequence?
        return offset < seqlen_;
    }

    // BERT Mask: if upper left is invalid, none are valid
    inline __device__ bool any_valid(int mi, int ni) const
    {
        return is_valid(mi, ni, 0, 0);
    }

    // Move mask to next tile (flash attention)
    inline __device__ void move()
    {
        this->col_ += Cta_tile::N;
    }

    // Move mask the col by offset (flash attention)
    inline __device__ void move_to_offset(int offset)
    {
        this->col_ = col_init_ + offset;
    }

    // Reset mask to the initial col
    inline __device__ void reset()
    {
        col_ = col_init_;
    }

    // Load the mask... Nothing to do for real.
    inline __device__ void load(int) {}

    // Load the mask... we use it to keep track of to row, col (flash attention).
    inline __device__ void load(int, int col_loop_step)
    {
        col_loop_step_ = col_loop_step;
    }

    // The length of the sequence.
    int seqlen_;
    // The left-most position of the thread in the sequence.
    int col_, col_init_;
    // The current col iteration
    int col_loop_step_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile>
struct Mask<Traits, Cta_tile, 3> : public Mask<Traits, Cta_tile, 2>
{
    // V3 mask is the causal mask (e.g. for GPT) and extends V2 masks (self-attention).
    using Base = Mask<Traits, Cta_tile, 2>;

    // The shape of the MMA tile.
    using Mma_tile = typename Base::Mma_tile;

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Mask(Params const& params, Block_info const& block_info, int tidx)
        : Base(params, block_info, tidx)
        , row_loop_step_(0)
    {

        int warp = tidx / Cta_tile::THREADS_PER_WARP;
        int lane = tidx % Cta_tile::THREADS_PER_WARP;

        // The position of the warp.
        int warp_m = warp % Cta_tile::WARPS_M;
        row_ = warp_m * 16 + lane / 4;
    }

    inline __device__ void get_row_col(int& row, int& col, int mi, int ni, int ii, int jj) const
    {
        // The position of the thread in the sequence.
        row = this->row_ + this->row_loop_step_ + mi * Mma_tile::M_PER_MMA_PER_CTA;
        // The position inside the MMA.
        row += ii * 8;

        // The position of the thread in the sequence.
        col = this->col_ + this->col_loop_step_ * Cta_tile::N + ni * Mma_tile::N_PER_MMA_PER_CTA;
        // The position inside the MMA.
        col += (jj & 0x02) * 4 + (jj & 0x1);
    }

    // Is a given position valid?
    inline __device__ bool is_valid(int mi, int ni, int ii, int jj) const
    {
        int row, col;
        get_row_col(row, col, mi, ni, ii, jj);

        // Is it a valid position in the sequence?
        return is_valid(row, col);
    }

    // Is a given position valid?
    inline __device__ bool is_valid(int row, int col) const
    {
        // Is it a valid position in the sequence, i.e. are we in the lower triangle?
        return (row >= col);
    }

    // GPT Mask: if lower left is invalid, none are valid
    inline __device__ bool any_valid(int mi, int ni) const
    {
        return is_valid(mi, ni, 1, 0);
    }

    // Load the mask... we use it to keep track of to row.
    inline __device__ void load(int row_loop_step)
    {
        row_loop_step_ = row_loop_step;
    }

    // Load the mask... we use it to keep track of to row, col (flash attention).
    inline __device__ void load(int row_loop_step, int col_loop_step)
    {
        row_loop_step_ = row_loop_step;
        this->col_loop_step_ = col_loop_step;
    }

    // The upper-most position of the thread in the sequence.
    int row_;
    // Current row step offset.
    int row_loop_step_;
};

// Specialized mask for MTP (multi-token prediction used in MLA).
template <typename Traits, typename Cta_tile>
struct MtpMask : public Mask<Traits, Cta_tile, 2>
{
    // MTP mask (causal mask) extends from V2 (dense) masks (self-attention).
    using Base = Mask<Traits, Cta_tile, 2>;

    // The shape of the MMA tile.
    using Mma_tile = typename Base::Mma_tile;

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ MtpMask(Params const& params, Block_info const& block_info, int tidx)
        : Base(params, block_info, tidx)
        , num_grouped_heads_(params.num_grouped_heads)
        , row_loop_step_(0)
    {

        // Update the seqlen (excluding all MTP draft tokens).
        this->seqlen_ = this->seqlen_ - (block_info.actual_q_seqlen / params.num_grouped_heads) + 1;

        int warp = tidx / Cta_tile::THREADS_PER_WARP;
        int lane = tidx % Cta_tile::THREADS_PER_WARP;

        // The position of the warp.
        int warp_m = warp % Cta_tile::WARPS_M;
        row_ = warp_m * 16 + lane / 4;
    }

    inline __device__ int get_row(int mi, int ii) const
    {
        // The position of the thread in the sequence.
        int row = this->row_ + this->row_loop_step_ + mi * Mma_tile::M_PER_MMA_PER_CTA;
        // The position inside the MMA.
        row += ii * 8;
        return row;
    }

    inline __device__ int get_col(int ni, int jj) const
    {
        // The position of the thread in the sequence.
        int col = this->col_ + this->col_loop_step_ * Cta_tile::N + ni * Mma_tile::N_PER_MMA_PER_CTA;
        // The position inside the MMA.
        col += (jj & 0x02) * 4 + (jj & 0x1);
        return col;
    }

    inline __device__ void get_row_col(int& row, int& col, int mi, int ni, int ii, int jj) const
    {
        row = get_row(mi, ii);
        col = get_col(ni, jj);
    }

    // Is a given position valid?
    inline __device__ bool is_valid(int mi, int ni, int ii, int jj) const
    {
        int col = get_col(ni, jj);

        // Is it a valid position in the sequence?
        return col < (this->seqlen_ + mtp_token_idx_[mi][ii]);
    }

    // Is a given position valid?
    inline __device__ bool is_valid(int row, int col) const
    {
        // Is it a valid position in the sequence, i.e. are we in the lower triangle?
        return (row >= col);
    }

    // Load the mask... we use it to keep track of to row.
    inline __device__ void load(int row_loop_step)
    {
        row_loop_step_ = row_loop_step;
// Update the MTP token index.
#pragma unroll
        for (int mi = 0; mi < Mma_tile::MMAS_M; ++mi)
        {
#pragma unroll
            for (int ii = 0; ii < 2; ++ii)
            {
                mtp_token_idx_[mi][ii] = get_row(mi, ii) / num_grouped_heads_;
            }
        }
    }

    // The number of grouped heads in the row dimension.
    int num_grouped_heads_;
    // The corresponding MTP token index for each row.
    // FIXME: currently we assume 2 rows per thread (volta/hopper-gmma traits are not supported yet).
    int mtp_token_idx_[Mma_tile::MMAS_M][2];
    // The upper-most position of the thread in the sequence.
    int row_;
    // The current row step offset.
    int row_loop_step_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// The lower triangle attention matrix.
// Assume we only pay attention to past sliding-window-size long sequence.
// v x x x x x x x x
// v v x x x x x x x
// v v v x x x x x x
// v v v v x x x x x
// v v v v v x x x x
// x v v v v v x x x
// x x v v v v v x x
// x x x v v v v v x
// x x x x v v v v v

template <typename Traits, typename Cta_tile>
struct Mask<Traits, Cta_tile, 4> : public Mask<Traits, Cta_tile, 3>
{
    // V4 mask is the causal mask (e.g. for GPT) plus the sliding-window feature.
    using Base = Mask<Traits, Cta_tile, 3>;

    // The shape of the MMA tile.
    using Mma_tile = typename Base::Mma_tile;

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Mask(Params const& params, Block_info const& block_info, int tidx)
        : Base(params, block_info, tidx)
        , sliding_window_size_(params.sliding_window_size)
    {
    }

    // Is a given position valid?
    inline __device__ bool is_valid(int mi, int ni, int ii, int jj) const
    {
        int row, col;
        this->get_row_col(row, col, mi, ni, ii, jj);

        // Is it a valid position in the sequence?
        return is_valid(row, col);
    }

    // Is a given position valid?
    inline __device__ bool is_valid(int row, int col) const
    {
        // Is it a valid position in the sequence, i.e. are we in the lower triangle?
        return (row >= col) && (col >= max(0, row + 1 - sliding_window_size_));
    }

    // The sliding window size.
    int sliding_window_size_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// The custom mask (from global memory).
template <typename Traits, typename Cta_tile>
struct Mask<Traits, Cta_tile, 5> : public Mask<Traits, Cta_tile, 3>
{

    using Base = Mask<Traits, Cta_tile, 3>;

    // The shape of the MMA tile.
    using Mma_tile = typename Base::Mma_tile;

    // The number of MMAs in each dimension.
    enum
    {
        MMAS_M = Mma_tile::MMAS_M
    };

    enum
    {
        MMAS_N = Mma_tile::MMAS_N
    };

    // One 32-bit packed mask holds 4 MMAS_N as one group.
    enum
    {
        MMA_GROUPS_N = fmha::Div_up<MMAS_N, 4>::VALUE
    };

    // The MMAS_N in the group.
    enum
    {
        MMAS_N_IN_GROUP = fmha::Min<MMAS_N, 4>::VALUE
    };

    // MMAS_N uses full 32-bit integer packed masks.
    enum
    {
        FULL_PACKED_MASK = (MMAS_N % 4 == 0)
    };

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Mask(Params const& params, Block_info const& block_info, int tidx)
        : Base(params, block_info, tidx)
        , packed_mask_ptr_(reinterpret_cast<char const*>(params.packed_mask_ptr))
        , params_packed_mask_stride_in_bytes_(params.packed_mask_stride_in_bytes)
        , row_offset_(0)
    {
        // Add the thread offset in bytes.
        packed_mask_ptr_ += (block_info.sum_mask_row * params_packed_mask_stride_in_bytes_ + tidx * sizeof(uint32_t));
    }

    // Load the mask... we use it to keep track of row offset.
    inline __device__ void load(int row_offset)
    {
        row_offset_ = row_offset;
    }

    // Load the mask into registers (and expand).
    inline __device__ void load_mask(int col_offset)
    {

        // The packed_mask_offset in the col(N) dimension.
        int mask_col_offset
            = int(col_offset / (Mma_tile::N_PER_MMA_PER_CTA * 4)) * Cta_tile::THREADS_PER_CTA * sizeof(uint32_t);
        // When MMAS_N < 4, one loaded packed_mask can be expanded to boolean masks
        // of multiple iterations.
        int local_col = FULL_PACKED_MASK ? 0 : (col_offset % (Mma_tile::N_PER_MMA_PER_CTA * 4));
        // The local mma ni if MMAS_N < 4.
        int local_ni = local_col / 16;
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
            // The M dimension offset.
            int offset = (row_offset_ + mi * Mma_tile::M_PER_MMA_PER_CTA) * params_packed_mask_stride_in_bytes_;
            // The N dimension offset.
            offset += mask_col_offset;
            // Set predicate to true only when next 32-bit packed mask is needed.
            bool pred = local_col == 0;
#pragma unroll
            for (int ni = 0; ni < MMA_GROUPS_N; ++ni)
            {
                // The MMAS_N group offset.
                if (pred)
                {
                    fmha::ldg(packed_mask_[mi][ni],
                        packed_mask_ptr_ + offset + ni * Cta_tile::THREADS_PER_CTA * sizeof(uint32_t));
                }
            }
        }

// Expand the mask.
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMA_GROUPS_N; ++ni)
            {
#pragma unroll
                for (int nni = 0; nni < MMAS_N_IN_GROUP; ++nni)
                {
                    mask_[2 * mi + 0][(ni * 4 + nni) * 4 + 0]
                        = packed_mask_[mi][ni] & (1u << (8 * (nni + local_ni) + 0));
                    mask_[2 * mi + 0][(ni * 4 + nni) * 4 + 1]
                        = packed_mask_[mi][ni] & (1u << (8 * (nni + local_ni) + 1));
                    mask_[2 * mi + 1][(ni * 4 + nni) * 4 + 0]
                        = packed_mask_[mi][ni] & (1u << (8 * (nni + local_ni) + 2));
                    mask_[2 * mi + 1][(ni * 4 + nni) * 4 + 1]
                        = packed_mask_[mi][ni] & (1u << (8 * (nni + local_ni) + 3));
                    mask_[2 * mi + 0][(ni * 4 + nni) * 4 + 2]
                        = packed_mask_[mi][ni] & (1u << (8 * (nni + local_ni) + 4));
                    mask_[2 * mi + 0][(ni * 4 + nni) * 4 + 3]
                        = packed_mask_[mi][ni] & (1u << (8 * (nni + local_ni) + 5));
                    mask_[2 * mi + 1][(ni * 4 + nni) * 4 + 2]
                        = packed_mask_[mi][ni] & (1u << (8 * (nni + local_ni) + 6));
                    mask_[2 * mi + 1][(ni * 4 + nni) * 4 + 3]
                        = packed_mask_[mi][ni] & (1u << (8 * (nni + local_ni) + 7));
                }
            }
        }
    }

    // Move mask the col by offset (flash attention)
    inline __device__ void move_to_offset(int col_offset)
    {
        load_mask(col_offset);
    }

    // Is a given position valid?
    inline __device__ bool is_valid(int mi, int ni, int ii, int jj) const
    {
        return mask_[mi * 2 + ii][ni * 4 + jj];
    }

    // Current row step offset.
    int row_offset_;

    // The pointer to the mask.
    char const* packed_mask_ptr_;
    // The stride in the n dimension.
    int64_t const params_packed_mask_stride_in_bytes_;
    // The packed mask (one 32-bit integer per MMA GROUP, MMAS_M * 2 rows, MMA_GROUPS_N * 16 cols).
    uint32_t packed_mask_[MMAS_M][MMA_GROUPS_N];
    // The mask after expansion.
    bool mask_[MMAS_M * 2][MMAS_N * 4];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Mask<Volta_hmma_fp16_traits, Cta_tile, 2>
{

    // The instruction traits.
    using Traits = Volta_hmma_fp16_traits;
    // The shape of the MMA tile.
    using Mma_tile = typename Traits::Mma_tile<Cta_tile>;

    // That implementation works only when WARPS_K is 1.
    static_assert(Cta_tile::WARPS_K == 1, "");

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Mask(Params const& params, Block_info const& block_info, int tidx)
        : seqlen_(block_info.actual_seqlen)
    {

        // The decomposition of the thread index into warp/lane.
        int warp = tidx / Cta_tile::THREADS_PER_WARP;
        int lane = tidx % Cta_tile::THREADS_PER_WARP;

        // The position of the warp.
        int warp_n = warp / Cta_tile::WARPS_M;
        // The position of the thread.
        col_ = block_info.bidn * Cta_tile::N + warp_n * 16 + (lane & 0x08) / 2;
        col_init_ = col_;
    }

    // Is a given position valid?
    inline __device__ bool is_valid(int, int ni, int, int jj) const
    {

        // The position of the thread in the sequence.
        int offset = this->col_ + ni * Mma_tile::N_PER_MMA_PER_CTA;
        // The position inside the MMA.
        offset += (jj & 0x04) * 2 + (jj & 0x03);
        // Is it a valid position in the sequence?
        return offset < seqlen_;
    }

    // Load the mask... Nothing to do for real.
    inline __device__ void load(int) {}

    // Reset mask to the initial col
    inline __device__ void reset()
    {
        col_ = col_init_;
    }

    // Move mask to next tile (flash attention)
    inline __device__ void move()
    {
        this->col_ += Cta_tile::N;
    }

    // Move mask the col by offset (flash attention)
    inline __device__ void move_to_offset(int offset)
    {
        this->col_ = col_init_ + offset;
    }

    // The length of the sequence.
    int const seqlen_;
    // The left-most position of the thread in the sequence.
    int col_, col_init_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Mask<Volta_hmma_fp16_traits, Cta_tile, 3> : public Mask<Volta_hmma_fp16_traits, Cta_tile, 2>
{
    // V3 mask is the causal mask (e.g. for GPT) and extends V2 masks (self-attention).
    using Base = Mask<Volta_hmma_fp16_traits, Cta_tile, 2>;

    // The shape of the MMA tile.
    using Mma_tile = typename Base::Mma_tile;

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Mask(Params const& params, Block_info const& block_info, int tidx)
        : Base(params, block_info, tidx)
        , loop_step_(0)
    {

        int warp = tidx / Cta_tile::THREADS_PER_WARP;
        int lane = tidx % Cta_tile::THREADS_PER_WARP;

        // The position of the warp.
        int warp_m = warp % Cta_tile::WARPS_M;
        row_ = warp_m * 16 + (lane & 0x07) + (lane & 0x10) / 2;
    }

    inline __device__ void get_row_col(int& row, int& col, int mi, int ni, int ii, int jj) const
    {
        // The position of the thread in the sequence.
        row = this->row_ + this->loop_step_ + mi * Mma_tile::M_PER_MMA_PER_CTA;

        // The position of the thread in the sequence.
        col = this->col_ + ni * Mma_tile::N_PER_MMA_PER_CTA;
        // The position inside the MMA.
        col += (jj & 0x04) * 2 + (jj & 0x03);
    }

    // Is a given position valid?
    inline __device__ bool is_valid(int mi, int ni, int ii, int jj) const
    {
        int row, col;
        get_row_col(row, col, mi, ni, ii, jj);

        // Is it a valid position in the sequence?
        return is_valid(row, col);
    }

    // Is a given position valid?
    inline __device__ bool is_valid(int row, int col) const
    {
        // Is it a valid position in the sequence, i.e. are we in the lower triangle?
        return (row >= col) && (col < this->seqlen_);
    }

    // GPT Mask: if lower left is invalid, none are valid
    inline __device__ bool any_valid(int mi, int ni) const
    {
        return is_valid(mi, ni, 0, 0);
    }

    // Load the mask... we use it to keep track of to row.
    inline __device__ void load(int loop_step)
    {
        loop_step_ = loop_step;
    }

    // The upper-most position of the thread in the sequence.
    int row_;
    // Current iteration.
    int loop_step_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, int FMHA_VERSION, bool IS_MTP>
struct Mask_dispatcher
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, int FMHA_VERSION>
struct Mask_dispatcher<Traits, Cta_tile, FMHA_VERSION, false> : public Mask<Traits, Cta_tile, FMHA_VERSION>
{
    using Base = Mask<Traits, Cta_tile, FMHA_VERSION>;

    template <typename Params, typename Block_info>
    inline __device__ Mask_dispatcher(Params const& params, Block_info const& block_info, int tidx)
        : Base(params, block_info, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, int FMHA_VERSION>
struct Mask_dispatcher<Traits, Cta_tile, FMHA_VERSION, true> : public MtpMask<Traits, Cta_tile>
{
    using Base = MtpMask<Traits, Cta_tile>;

    template <typename Params, typename Block_info>
    inline __device__ Mask_dispatcher(Params const& params, Block_info const& block_info, int tidx)
        : Base(params, block_info, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, int FMHA_VERSION>
struct Mask_hopper
{

    // The shape of the MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Mask_hopper(Params const& params, Block_info const& block_info, int tidx)
        : seqlen_(block_info.actual_seqlen)
    {
        // For Hopper the warp distribution is always 4x1 within a warpgroup.
        // So maybe there is some assumptions/optimizations to be made here.

        int warp = tidx / Cta_tile::THREADS_PER_WARP;
        int warp_n = warp / 4;
        int lane = tidx % Cta_tile::THREADS_PER_WARP;
        col_ = warp_n * Mma_tile::N_PER_WARP_GROUP + (lane % 4) * 2;
    }

    // Is a given position valid?
    inline __device__ bool is_valid(int, int ni, int, int jj) const
    {
        // The position of the thread in the sequence.
        int offset = this->col_ + ni * Mma_tile::N_PER_MMA;
        // The position inside the MMA.
        offset += (jj / 2) * 8 + (jj % 2);
        // Is it a valid position in the sequence?
        return offset < seqlen_;
    }

    // Load the mask... Nothing to do for real.
    inline __device__ void load(int) {}

    // The length of the sequence.
    int const seqlen_;
    // The left-most position of the thread in the sequence.
    int col_;
};

template <typename Traits, typename Cta_tile>
struct Mask_hopper<Traits, Cta_tile, 3>
{

    // The shape of the MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Mask_hopper(Params const& params, Block_info const& block_info, int tidx)
    {
        // For Hopper the warp distribution is always 4x1 within a warpgroup.
        // So maybe there is some assumptions/optimizations to be made here.

        int warp = tidx / Cta_tile::THREADS_PER_WARP;
        int warp_n = warp / 4;
        int warp_m = warp % 4;
        int lane = tidx % Cta_tile::THREADS_PER_WARP;
        col_ = warp_n * Mma_tile::N_PER_WARP_GROUP + (lane % 4) * 2;
        row_base_ = warp_m * 16 + lane / 4;
        row_ = row_base_;
    }

    inline __device__ void get_row_col(int& row, int& col, int mi, int ni, int ii, int jj) const
    {
        // The row position of the thread in the sequence.
        row = row_ + mi * Mma_tile::M_PER_MMA + ii * 8;

        // The position of the thread in the sequence.
        col = this->col_ + ni * Mma_tile::N_PER_MMA;
        // The position inside the MMA.
        col += (jj / 2) * 8 + (jj % 2);
    }

    // Is a given position valid?
    inline __device__ bool is_valid(int mi, int ni, int ii, int jj) const
    {
        int row, col;
        get_row_col(row, col, mi, ni, ii, jj);

        // Is it a valid position in the sequence?
        return is_valid(row, col);
    }

    // Is a given position valid?
    inline __device__ bool is_valid(int row, int col) const
    {
        // Is it a valid position in the sequence?
        return col <= row;
    }

    // Load the mask... Nothing to do for real.
    inline __device__ void load(int loop_step)
    {
        row_ = row_base_ + loop_step * Cta_tile::M;
    }

    // The left-most position of the thread in the sequence.
    int row_, row_base_, col_;
};

template <typename Traits, typename Cta_tile>
struct Mask_hopper<Traits, Cta_tile, 4> : public Mask_hopper<Traits, Cta_tile, 3>
{

    // V4 mask is the causal mask (e.g. for GPT) plus the sliding-window feature.
    using Base = Mask_hopper<Traits, Cta_tile, 3>;

    // The shape of the MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Mask_hopper(Params const& params, Block_info const& block_info, int tidx)
        : Base(params, block_info, tidx)
        , sliding_window_size_(params.sliding_window_size)
    {
    }

    // Is a given position valid?
    inline __device__ bool is_valid(int mi, int ni, int ii, int jj) const
    {
        int row, col;
        this->get_row_col(row, col, mi, ni, ii, jj);

        // Is it a valid position in the sequence?
        return is_valid(row, col);
    }

    // Is a given position valid?
    inline __device__ bool is_valid(int row, int col) const
    {
        // Is it a valid position in the sequence?
        return col <= row && col >= max(0, row + 1 - sliding_window_size_);
    }

    // The sliding window size for attention.
    int sliding_window_size_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha
