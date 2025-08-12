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

namespace fmha
{
namespace v1
{

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The number of bits per element.
    int BITS_PER_ELEMENT,
    // The number of rows of Q, K or V loaded by this tile.
    int ROWS_,
    // The number of columns.
    int COLS,
    // The number of valid columns
    int VALID_COLS,
    // Do we use LDGSTS?
    bool USE_LDGSTS_,
    // Are attention heads interleaved?
    bool HEADS_INTERLEAVED,
    // Number of matrices
    int NUM_MATS = 3,
    // Is sliding window attention used ?
    bool SLIDING_WINDOW_ATTENTION = false>
struct Gmem_tile_qkv
{

    // The size of each LDG.
    enum
    {
        BYTES_PER_LDG = 16
    };

    // The size of a row in bytes.
    enum
    {
        BYTES_PER_ROW = COLS * BITS_PER_ELEMENT / 8
    };

    // The number of threads to load a "row" of the matrix.
    enum
    {
        THREADS_PER_ROW = BYTES_PER_ROW / BYTES_PER_LDG
    };

    // The valid size of a row in bytes.
    enum
    {
        VALID_BYTES_PER_ROW = VALID_COLS * BITS_PER_ELEMENT / 8
    };

    // The valid number of threads to load a "row" of the matrix.
    enum
    {
        VALID_THREADS_PER_ROW = VALID_BYTES_PER_ROW / BYTES_PER_LDG
    };

    // The number of "rows" loaded per LDG.
    enum
    {
        ROWS_PER_LDG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW
    };

    // The number of rows.
    enum
    {
        ROWS = ROWS_
    };

    // The number of LDGs needed to load a chunk of the Q matrix.
    enum
    {
        LDGS = fmha::Div_up<ROWS, ROWS_PER_LDG>::VALUE
    };

    // The number of predicate registers.
    enum
    {
        PRED_REGS = fmha::Compute_number_of_pred_regs<LDGS>::VALUE
    };

    // Make sure we use a single register to store predicates.
    static_assert(PRED_REGS == 1, "");

    // We do not use LDGSTS (for the moment).
    enum
    {
        USE_LDGSTS = USE_LDGSTS_
    };

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_qkv(
        Params const& params, int qkv_offset, Block_info const& binfo, int tidx, int cta_row_offset = 0)

        // in PACKED_QKV, q_stride = k_stride = v_stride
        : params_qkv_stride_in_bytes_(params.q_stride_in_bytes)
        , qkv_ptr_(reinterpret_cast<char const*>(params.qkv_ptr))
    {

        // Compute the position in the sequence (within the CTA for the moment).
        int row = tidx / THREADS_PER_ROW;
        // Compute the position of the thread in the row.
        int col = tidx % THREADS_PER_ROW;

        // Prepare predicates.
        uint32_t preds[LDGS];
#pragma unroll
        for (int ii = 0; ii < LDGS; ++ii)
        {
            preds[ii] = row + ii * ROWS_PER_LDG < ROWS;
        }

        // Pack the predicates.
        preds_[0] = fmha::pack_predicates(preds);

        // The row offset in the batched GEMM. For each seq element, we store QKV in that order.
        int64_t row_offset = (int64_t) (row + cta_row_offset) * params_qkv_stride_in_bytes_;
        // Add the block index.
        int idx;
        if (HEADS_INTERLEAVED)
        {
            idx = binfo.bidx * NUM_MATS + qkv_offset;
        }
        else
        {
            idx = (params.b * params.s * NUM_MATS + qkv_offset) * params.h + binfo.bidh;
        }
        // Assemble the final pointer.
        qkv_ptr_ += row_offset + idx * VALID_BYTES_PER_ROW + col * BYTES_PER_LDG;

        // active threads
        is_active_ = col < VALID_THREADS_PER_ROW;
    }

    // Store data to shared memory.
    template <typename Smem_tile>
    inline __device__ void commit(Smem_tile& smem_tile)
    {
        if (!USE_LDGSTS)
        {
            smem_tile.store(fetch_);
        }
    }

    // Load data from memory.
    template <typename Smem_tile>
    inline __device__ void load(Smem_tile& smem_tile)
    {
        void const* ptrs[LDGS];
#pragma unroll
        for (int ii = 0; ii < LDGS; ++ii)
        {
            ptrs[ii] = qkv_ptr_ + (int64_t) ii * ROWS_PER_LDG * params_qkv_stride_in_bytes_;
        }
        if (USE_LDGSTS)
        {
            smem_tile.store(ptrs, preds_);
        }
        else
        {
            fmha::ldg(fetch_, ptrs, preds_);
        }
    }

    // Load data from global memory, shared mem is not needed
    inline __device__ void load()
    {
        void const* ptrs[LDGS];
        if (is_active_)
        {
#pragma unroll
            for (int ii = 0; ii < LDGS; ++ii)
            {
                ptrs[ii] = qkv_ptr_ + (int64_t) ii * ROWS_PER_LDG * params_qkv_stride_in_bytes_;
            }
            fmha::ldg(fetch_, ptrs, preds_);
        }
    }

    // Move the pointer to the next location.
    inline __device__ void move()
    {
        qkv_ptr_ += (int64_t) ROWS * params_qkv_stride_in_bytes_;
    }

    // The stride between rows for the QKV matrice.
    int64_t const params_qkv_stride_in_bytes_;
    // The pointer.
    char const* qkv_ptr_;
    // The register to store predicates.
    uint32_t preds_[PRED_REGS];
    // The fetch registers.
    uint4 fetch_[LDGS];
    // The active LDG threads
    bool is_active_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace v1
} // namespace fmha
