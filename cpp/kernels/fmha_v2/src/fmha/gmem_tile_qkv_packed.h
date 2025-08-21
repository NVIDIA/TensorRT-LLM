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
#include <fused_multihead_attention.h>

namespace fmha
{
namespace v2
{

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int USE_LDGSTS>
struct Ldgsts_helper
{
    template <typename This, typename Smem_tile, int LDGS>
    static inline __device__ void load(
        This* this_, Smem_tile& smem_tile, void const* (&ptrs)[LDGS], uint32_t (&preds)[LDGS])
    {
        fmha::pack_predicates(this_->preds_, preds);
        smem_tile.store(ptrs, this_->preds_);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Ldgsts_helper<0>
{
    template <typename This, typename Smem_tile, int LDGS>
    static inline __device__ void load(
        This* this_, Smem_tile& smem_tile, void const* (&ptrs)[LDGS], uint32_t (&preds)[LDGS])
    {
#if 0
        fmha::pack_predicates(this_->preds_, preds);
        fmha::ldg(this_->fetch_, ptrs, this_->preds_);
#else
#pragma unroll
        for (int ii = 0; ii < LDGS; ++ii)
        {
            this_->fetch_[ii] = make_uint4(0u, 0u, 0u, 0u);
        }
        // not packing predicates removes restrictions (e.g. FP16 384, 4 warps)
        Ldg_functor<uint4, LDGS> fct(this_->fetch_, ptrs);
#pragma unroll
        for (int ii = 0; ii < LDGS; ++ii)
        {
            fct.ldgsts(ii, preds[ii]);
        }
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The number of bits per element.
    int BITS_PER_ELEMENT_,
    // The number of rows of Q, K or V loaded by this tile.
    int ROWS_,
    // The number of columns (padded, e.g 64).
    int COLS,
    // The actual number of columns (unpadded, e.g 40)
    int VALID_COLS_,
    // Do we use LDGSTS?
    bool USE_LDGSTS_,
    // Are attention heads interleaved?
    bool HEADS_INTERLEAVED,
    // The number of matrices
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

    // The number of bits/bytes of element
    enum
    {
        BITS_PER_ELEMENT = BITS_PER_ELEMENT_
    };

    enum
    {
        BYTES_PER_ELEMENT = BITS_PER_ELEMENT_ / 8
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

    // The valid size of a row in bytes (without paddings).
    enum
    {
        VALID_COLS = VALID_COLS_
    };

    // The amount of bytes that are valid per row.
    enum
    {
        VALID_BYTES_PER_ROW = VALID_COLS * BITS_PER_ELEMENT / 8
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

    // Is it Hopper?
    enum
    {
        IS_HOPPER = std::is_same<typename Traits::Gpu_arch, typename fmha::Hopper>::value == true
    };

    // Make sure we use a single register to store predicates. Do not throw for Hopper for now.
    static_assert(!USE_LDGSTS_ || PRED_REGS == 1 || IS_HOPPER, "");

    // We do not use LDGSTS (for the moment).
    enum
    {
        USE_LDGSTS = USE_LDGSTS_
    };

    // Ctor for bert::Fused_multihead_attention_params_v2 class
    template <typename Block_info>
    inline __device__ Gmem_tile_qkv(bert::Fused_multihead_attention_params_v2 const& params, int qkv_offset,
        Block_info const& binfo, int tidx, int cta_row_offset = 0, int cta_col_offset_in_bytes = 0)
        : Gmem_tile_qkv(params.qkv_ptr, params.q_stride_in_bytes, params.d, params.dv, params.h, qkv_offset, binfo,
            tidx, params.h_kv, cta_row_offset, cta_col_offset_in_bytes)
    {
    }

    // Ctor for other param classes (such as Qkv_params in train_ops)
    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_qkv(Params const& params, int qkv_offset, Block_info const& binfo, int tidx,
        int cta_row_offset = 0, int cta_col_offset_in_bytes = 0)
        : Gmem_tile_qkv(params.qkv_ptr, params.q_stride_in_bytes, params.d, params.dv, params.h, qkv_offset, binfo,
            tidx, cta_row_offset, cta_col_offset_in_bytes)
    {
    }

    // Ctor.
    template <typename Block_info>
    inline __device__ Gmem_tile_qkv(void* qkv_ptr, size_t qkv_stride_in_bytes, int d, int dv, int num_heads,
        int qkv_offset, Block_info const& binfo, int tidx, int num_kv_heads = 0, int cta_row_offset = 0,
        int cta_col_offset_in_bytes = 0)
        : params_qkv_stride_in_bytes_(qkv_stride_in_bytes)
        , actual_seqlen_(binfo.actual_seqlen)
        , qkv_ptr_(reinterpret_cast<char*>(qkv_ptr))
    {

        // Compute the position in the sequence (within the CTA for the moment).
        int row = tidx / THREADS_PER_ROW;
        // Compute the position of the thread in the row.
        int col = tidx % THREADS_PER_ROW;

        // We must store the value to update the predicates in "load".
        row_ = row;
        // Do not load/store if the thread is in the padded area
        col_in_bytes_ = cta_col_offset_in_bytes + col * BYTES_PER_LDG;

        // The row offset in the batched GEMM. For each seq element, we store QKV in that order.
        int64_t row_offset = (int64_t) (row + cta_row_offset) * params_qkv_stride_in_bytes_;
        // Add the byte index.
        int64_t idx;

        // Both MQA and GQA will use non HEADS_INTERLEAVED layout
        if (num_kv_heads < num_heads)
        {
            int const head_id = binfo.bidh;
            int const kv_head_id = binfo.bidh / (num_heads / num_kv_heads);
            // QKV layout [b, s, [q_hd, k_h'd, v_h'd]]
            idx = binfo.sum_s * params_qkv_stride_in_bytes_;
            if (qkv_offset == 0)
            { // Q tensor
                idx += head_id * VALID_BYTES_PER_ROW;
            }
            else if (qkv_offset == 1)
            { // K tensor
                idx += (num_heads + kv_head_id) * VALID_BYTES_PER_ROW;
            }
            else if (qkv_offset == 2)
            { // V tensor
                /*  When qkv_offset == 2, this is an instance of Gmem_tile_v defined in Kernel_traits:
                        using Gmem_tile_v = Gmem_tile_v_<Traits_o,
                                Cta_tile_o,
                                Traits_o::BITS_PER_ELEMENT_B,
                                CTA_O_TILE_K,
                                CTA_O_TILE_N,
                                VALID_DV,   // instead of VALID_D
                                USE_LDGSTS_V,
                                HEADS_INTERLEAVED,
                                3, // NUM_MATS
                                SLIDING_WINDOW_ATTENTION>;
                    the 6th template argument is VALID_DV instead of VALID_D.
                    Thus, here VALID_COLS equals VALID_DV, and
                    VALID_BYTES_PER_ROW equals VALID_DV * BYTES_PER_ELEMENT,
                    and `kv_head_id * dv * BYTES_PER_ELEMENT` can be optimized to
                    `kv_head_id * VALID_BYTES_PER_ROW`. */
                idx += (num_heads + num_kv_heads) * d * BYTES_PER_ELEMENT + kv_head_id * VALID_BYTES_PER_ROW;
            }
        }
        else if (HEADS_INTERLEAVED)
        {
            // [b, s, h, [q_d, k_d, v_d]] aka bsh3d
            // bidx = sum_s * params.h + bidh;
            idx = (binfo.bidx * (2 * d + dv) + qkv_offset * d) * BYTES_PER_ELEMENT;
        }
        else
        {
            // [b, s, [q_hd, k_hd, v_hd]] aka bs3hd
            idx = binfo.sum_s * params_qkv_stride_in_bytes_ + qkv_offset * num_heads * d * BYTES_PER_ELEMENT
                + binfo.bidh * VALID_BYTES_PER_ROW;
        }

        // Assemble the final pointer.
        qkv_ptr_ += row_offset + idx + col_in_bytes_;

        // Take the CTA offset to modify the sequence length.
        actual_seqlen_ -= cta_row_offset;

        // Set the initial seq_len and qkv_offset in case of reinterating
        actual_seqlen_init_ = actual_seqlen_;
        qkv_ptr_init_ = qkv_ptr_;
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
        uint32_t preds[LDGS];
#pragma unroll
        for (int ii = 0; ii < LDGS; ++ii)
        {
            preds[ii] = row_ + ii * (int) ROWS_PER_LDG < min((int) ROWS, actual_seqlen_);
            preds[ii] &= col_in_bytes_ < VALID_BYTES_PER_ROW;
        }

        // Prepare the load pointers.
        void const* ptrs[LDGS];
#pragma unroll
        for (int ii = 0; ii < LDGS; ++ii)
        {
            ptrs[ii] = qkv_ptr_ + (int64_t) ii * ROWS_PER_LDG * params_qkv_stride_in_bytes_;
        }

        // Trigger LDGSTS or the LDGs.
        // The predicates protect against out-of-bound access in rows and cols
        Ldgsts_helper<USE_LDGSTS>::load(this, smem_tile, ptrs, preds);
    }

    // Load data from memory.
    inline __device__ void load()
    {
        uint32_t preds[LDGS];
#pragma unroll
        for (int ii = 0; ii < LDGS; ++ii)
        {
            preds[ii] = row_ + ii * (int) ROWS_PER_LDG < min((int) ROWS, actual_seqlen_);
        }

        // Prepare the load pointers.
        void const* ptrs[LDGS];
#pragma unroll
        for (int ii = 0; ii < LDGS; ++ii)
        {
            ptrs[ii] = qkv_ptr_ + (int64_t) ii * ROWS_PER_LDG * params_qkv_stride_in_bytes_;
        }

        // Trigger the LDGs.
        if (col_in_bytes_ < VALID_BYTES_PER_ROW)
        {
            fmha::pack_predicates(preds_, preds);
            fmha::ldg(fetch_, ptrs, preds_);
        }
        else
        {
#pragma unroll
            for (int ii = 0; ii < LDGS; ++ii)
            {
                fetch_[ii] = make_uint4(0u, 0u, 0u, 0u);
            }
        }
    }

    // Move the pointer to the next row location.
    inline __device__ void move(int const steps = 1)
    {
        qkv_ptr_ += (int64_t) ROWS * params_qkv_stride_in_bytes_ * steps;
        actual_seqlen_ -= (int) ROWS * steps;
    }

    // Move the pointer to the next row location by the offset (not step).
    inline __device__ void move_by_offset(int const offset)
    {
        qkv_ptr_ = qkv_ptr_init_ + (int64_t) offset * params_qkv_stride_in_bytes_;
        actual_seqlen_ = actual_seqlen_init_ - (int) offset;
    }

    // Move the pointer to the next column location
    inline __device__ void move_col(int const steps = 1)
    {
        qkv_ptr_ += (int64_t) COLS * (BITS_PER_ELEMENT / 8) * steps;
        // Update col_in_bytes_ to ensure load predicates work
        col_in_bytes_ += THREADS_PER_ROW * BYTES_PER_LDG * steps;
    }

    inline __device__ void reset()
    {
        qkv_ptr_ = qkv_ptr_init_;
        actual_seqlen_ = actual_seqlen_init_;
    }

    // Rewind the pointer back to previous column location
    inline __device__ void rewind_col(int const steps)
    {
        qkv_ptr_ -= COLS * (BITS_PER_ELEMENT / 8) * steps;
        // Update col_in_bytes_ to ensure load predicates work
        col_in_bytes_ -= THREADS_PER_ROW * BYTES_PER_LDG * steps;
    }

    inline __device__ void move_to(int const step)
    {
        qkv_ptr_ = qkv_ptr_init_ + (int64_t) ROWS * params_qkv_stride_in_bytes_ * step;
        actual_seqlen_ = actual_seqlen_init_ - (int) ROWS * step;
    }

    // Store data to memory.
    inline __device__ void store(uint4 const (&data)[LDGS])
    {
#pragma unroll
        for (int ii = 0; ii < LDGS; ++ii)
        {
            char* ptr = qkv_ptr_ + (int64_t) ii * ROWS_PER_LDG * params_qkv_stride_in_bytes_;
            if (((row_ + ii * ROWS_PER_LDG) < min(ROWS, actual_seqlen_))
                && col_in_bytes_ < VALID_BYTES_PER_ROW /*TODO: double check*/)
            {
                fmha::stg(ptr, data[ii]);
            }
        }
    }

    // The stride between rows for the QKV matrice.
    int64_t params_qkv_stride_in_bytes_;
    // The pointer.
    char* qkv_ptr_;
    char* qkv_ptr_init_;
    // The register to store predicates.
    uint32_t preds_[PRED_REGS];
    // The fetch registers.
    uint4 fetch_[LDGS];
    // Keep track of the row and col the thread is processing as we move the tile.
    int row_;
    int col_in_bytes_;
    // The sequence length.
    int actual_seqlen_;
    int actual_seqlen_init_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

// We expect the Q/K/V layout to be [B, S, H, D] with variable sequence length support.
template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The number of bits per element.
    int BITS_PER_ELEMENT_,
    // The number of rows of Q, K or V loaded by this tile.
    int ROWS_,
    // The number of columns (padded, e.g 64).
    int COLS,
    // The actual number of columns (unpadded, e.g 40)
    int VALID_COLS_,
    // Do we use LDGSTS?
    bool USE_LDGSTS_,
    // Are attention heads interleaved? (not used)
    bool HEADS_INTERLEAVED = false,
    // The number of matrices (not used)
    int NUM_MATS = 1,
    // Is sliding window attention used ?
    bool SLIDING_WINDOW_ATTENTION = false>
struct Gmem_tile_q_k_v
{

    // The size of each LDG.
    enum
    {
        BYTES_PER_LDG = 16
    };

    // The number of bits/bytes of element
    enum
    {
        BITS_PER_ELEMENT = BITS_PER_ELEMENT_
    };

    enum
    {
        BYTES_PER_ELEMENT = BITS_PER_ELEMENT_ / 8
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

    // The valid size of a row in bytes (without paddings).
    enum
    {
        VALID_COLS = VALID_COLS_
    };

    // The amount of bytes that are valid per row.
    enum
    {
        VALID_BYTES_PER_ROW = VALID_COLS * BITS_PER_ELEMENT / 8
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

    // Is it Hopper?
    enum
    {
        IS_HOPPER = std::is_same<typename Traits::Gpu_arch, typename fmha::Hopper>::value == true
    };

    // Make sure we use a single register to store predicates. Do not throw for Hopper for now.
    static_assert(!USE_LDGSTS_ || PRED_REGS == 1 || IS_HOPPER, "");

    // We do not use LDGSTS (for the moment).
    enum
    {
        USE_LDGSTS = USE_LDGSTS_
    };

    // Ctor
    // qkv_offset: 0 for Q, 1 for K, 2 for V
    template <typename Block_info>
    inline __device__ Gmem_tile_q_k_v(bert::Fused_multihead_attention_params_v2 const& params, int qkv_offset,
        Block_info const& binfo, int tidx, int cta_row_offset = 0, int cta_col_offset_in_bytes = 0)
    {

        int seq_offset = 0;
        if (qkv_offset == 0)
        {
            // Q tensor
            params_q_k_v_stride_in_bytes_ = params.q_stride_in_bytes;
            q_k_v_ptr_ = reinterpret_cast<char*>(params.q_ptr);
            actual_seqlen_ = binfo.actual_q_seqlen;
            seq_offset = binfo.sum_s;
        }
        else if (qkv_offset == 1)
        {
            // K tensor
            params_q_k_v_stride_in_bytes_ = params.k_stride_in_bytes;
            q_k_v_ptr_ = reinterpret_cast<char*>(params.k_ptr);
            actual_seqlen_ = binfo.actual_kv_seqlen;
            seq_offset = binfo.sum_s_kv;
        }
        else if (qkv_offset == 2)
        {
            // V tensor
            params_q_k_v_stride_in_bytes_ = params.v_stride_in_bytes;
            q_k_v_ptr_ = reinterpret_cast<char*>(params.v_ptr);
            actual_seqlen_ = binfo.actual_kv_seqlen;
            seq_offset = binfo.sum_s_kv;
        }

        // Compute the position in the sequence (within the CTA for the moment).
        int row = tidx / THREADS_PER_ROW;
        // Compute the position of the thread in the row.
        int col = tidx % THREADS_PER_ROW;

        // We must store the value to update the predicates in "load".
        row_ = row;
        // Do not load/store if the thread is in the padded area
        col_in_bytes_ = cta_col_offset_in_bytes + col * BYTES_PER_LDG;

        // The row offset in the batched GEMM, including the sequence offset.
        int64_t row_offset = (int64_t) (row + cta_row_offset + seq_offset) * params_q_k_v_stride_in_bytes_;
        // Add the head index.
        int64_t idx = binfo.bidh;

        // Assemble the final pointer.
        q_k_v_ptr_ += row_offset + idx * VALID_BYTES_PER_ROW + col_in_bytes_;

        // Take the CTA offset to modify the sequence length.
        actual_seqlen_ -= cta_row_offset;

        // Set the initial seq_len and qkv_offset in case of reinterating
        actual_seqlen_init_ = actual_seqlen_;
        q_k_v_ptr_init_ = q_k_v_ptr_;
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
        uint32_t preds[LDGS];
#pragma unroll
        for (int ii = 0; ii < LDGS; ++ii)
        {
            preds[ii] = row_ + ii * (int) ROWS_PER_LDG < min((int) ROWS, actual_seqlen_);
            preds[ii] &= col_in_bytes_ < VALID_BYTES_PER_ROW;
        }

        // Prepare the load pointers.
        void const* ptrs[LDGS];
#pragma unroll
        for (int ii = 0; ii < LDGS; ++ii)
        {
            ptrs[ii] = q_k_v_ptr_ + (int64_t) ii * ROWS_PER_LDG * params_q_k_v_stride_in_bytes_;
        }

        // Trigger LDGSTS or the LDGs.
        // The predicates protect against out-of-bound access in rows and cols
        Ldgsts_helper<USE_LDGSTS>::load(this, smem_tile, ptrs, preds);
    }

    // Move the pointer to the next row location.
    inline __device__ void move(int const steps = 1)
    {
        q_k_v_ptr_ += (int64_t) ROWS * params_q_k_v_stride_in_bytes_ * steps;
        actual_seqlen_ -= (int) ROWS * steps;
    }

    // Move the pointer to the next row location by the offset (not step).
    inline __device__ void move_by_offset(int const offset)
    {
        q_k_v_ptr_ = q_k_v_ptr_init_ + (int64_t) offset * params_q_k_v_stride_in_bytes_;
        actual_seqlen_ = actual_seqlen_init_ - (int) offset;
    }

    // Move the pointer to the next column location
    inline __device__ void move_col()
    {
        q_k_v_ptr_ += (int64_t) COLS * (BITS_PER_ELEMENT / 8);
        // Update col_in_bytes_ to ensure load predicates work
        col_in_bytes_ += THREADS_PER_ROW * BYTES_PER_LDG;
    }

    // Rewind the pointer back to previous column location
    inline __device__ void rewind_col(int const steps)
    {
        q_k_v_ptr_ -= COLS * (BITS_PER_ELEMENT / 8) * steps;
        // Update col_in_bytes_ to ensure load predicates work
        col_in_bytes_ -= THREADS_PER_ROW * BYTES_PER_LDG * steps;
    }

    // Move the pointer to the specified step.
    inline __device__ void move_to(int const step)
    {
        q_k_v_ptr_ = q_k_v_ptr_init_ + (int64_t) ROWS * params_q_k_v_stride_in_bytes_ * step;
        actual_seqlen_ = actual_seqlen_init_ - (int) ROWS * step;
    }

    inline __device__ void reset()
    {
        q_k_v_ptr_ = q_k_v_ptr_init_;
        actual_seqlen_ = actual_seqlen_init_;
    }

    // The stride between rows for the Q/K/V matrice.
    int64_t params_q_k_v_stride_in_bytes_;
    // The pointer.
    char* q_k_v_ptr_;
    char* q_k_v_ptr_init_;
    // The register to store predicates.
    uint32_t preds_[PRED_REGS];
    // The fetch registers.
    uint4 fetch_[LDGS];
    // Keep track of the row and col the thread is processing as we move the tile.
    int row_;
    int64_t col_in_bytes_;
    // The sequence length.
    int actual_seqlen_;
    int actual_seqlen_init_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Shape [B, S, 2, H, D] where S can be variable sequence length.
template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The number of bits per element.
    int BITS_PER_ELEMENT_,
    // The number of rows of Q, K or V loaded by this tile.
    int ROWS_,
    // The number of columns (padded, e.g 64).
    int COLS,
    // The actual number of columns (unpadded, e.g 40)
    int VALID_COLS_,
    // Do we use LDGSTS?
    bool USE_LDGSTS_,
    // Are attention heads interleaved? (Not used)
    bool HEADS_INTERLEAVED,
    // The number of matrices (Not used)
    int NUM_MATS = 2,
    // Is sliding window attention used ?
    bool SLIDING_WINDOW_ATTENTION = false>
struct Gmem_tile_contiguous_kv
{

    // The size of each LDG.
    enum
    {
        BYTES_PER_LDG = 16
    };

    // The number of bits/bytes of element
    enum
    {
        BITS_PER_ELEMENT = BITS_PER_ELEMENT_
    };

    enum
    {
        BYTES_PER_ELEMENT = BITS_PER_ELEMENT_ / 8
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

    // The valid size of a row in bytes (without paddings).
    enum
    {
        VALID_COLS = VALID_COLS_
    };

    // The amount of bytes that are valid per row.
    enum
    {
        VALID_BYTES_PER_ROW = VALID_COLS * BITS_PER_ELEMENT / 8
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

    // Is it Hopper?
    enum
    {
        IS_HOPPER = std::is_same<typename Traits::Gpu_arch, typename fmha::Hopper>::value == true
    };

    // Make sure we use a single register to store predicates. Do not throw for Hopper for now.
    static_assert(!USE_LDGSTS_ || PRED_REGS == 1 || IS_HOPPER, "");

    // We do not use LDGSTS (for the moment).
    enum
    {
        USE_LDGSTS = USE_LDGSTS_
    };

    // Ctor for bert::Fused_multihead_attention_params_v2 class
    template <typename Block_info>
    inline __device__ Gmem_tile_contiguous_kv(bert::Fused_multihead_attention_params_v2 const& params,
        int qkv_offset, // q = 0, k = 1, v = 2.
        Block_info const& binfo, int tidx, int cta_row_offset = 0, int cta_col_offset_in_bytes = 0)
        : Gmem_tile_contiguous_kv(params.kv_ptr, params.k_stride_in_bytes, params.h_kv, params.h_q_per_kv, qkv_offset,
            binfo, tidx, cta_row_offset, cta_col_offset_in_bytes)
    {
    }

    // Ctor.
    template <typename Block_info>
    inline __device__ Gmem_tile_contiguous_kv(void* kv_ptr, size_t kv_stride_in_bytes, int num_kv_heads,
        int head_group_size, int qkv_offset, Block_info const& binfo, int tidx, int cta_row_offset = 0,
        int cta_col_offset_in_bytes = 0)
        : params_kv_stride_in_bytes_(kv_stride_in_bytes)
        , actual_seqlen_(binfo.actual_kv_seqlen)
        , kv_ptr_(reinterpret_cast<char*>(kv_ptr))
    {

        // Compute the position in the sequence (within the CTA for the moment).
        int row = tidx / THREADS_PER_ROW;
        // Compute the position of the thread in the row.
        int col = tidx % THREADS_PER_ROW;

        // We must store the value to update the predicates in "load".
        row_ = row;
        // Do not load/store if the thread is in the padded area
        col_in_bytes_ = cta_col_offset_in_bytes + col * BYTES_PER_LDG;

        // The row offset in the batched GEMM.
        int64_t row_offset = (int64_t) (row + cta_row_offset) * params_kv_stride_in_bytes_;
        // [b, s, 2, h_kv, d].
        int64_t idx = (binfo.sum_s_kv * 2 + qkv_offset - 1) * num_kv_heads + (binfo.bidh / head_group_size);

        // Assemble the final pointer.
        kv_ptr_ += row_offset + idx * VALID_BYTES_PER_ROW + col_in_bytes_;

        // Take the CTA offset to modify the sequence length.
        actual_seqlen_ -= cta_row_offset;

        // Set the initial seq_len and qkv_offset in case of reinterating
        actual_seqlen_init_ = actual_seqlen_;
        kv_ptr_init_ = kv_ptr_;
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
        uint32_t preds[LDGS];
#pragma unroll
        for (int ii = 0; ii < LDGS; ++ii)
        {
            preds[ii] = row_ + ii * (int) ROWS_PER_LDG < min((int) ROWS, actual_seqlen_);
            preds[ii] &= col_in_bytes_ < VALID_BYTES_PER_ROW;
        }

        // Prepare the load pointers.
        void const* ptrs[LDGS];
#pragma unroll
        for (int ii = 0; ii < LDGS; ++ii)
        {
            ptrs[ii] = kv_ptr_ + (int64_t) ii * ROWS_PER_LDG * params_kv_stride_in_bytes_;
        }

        // Trigger LDGSTS or the LDGs.
        // The predicates protect against out-of-bound access in rows and cols
        Ldgsts_helper<USE_LDGSTS>::load(this, smem_tile, ptrs, preds);
    }

    // Load data from memory.
    inline __device__ void load()
    {
        uint32_t preds[LDGS];
#pragma unroll
        for (int ii = 0; ii < LDGS; ++ii)
        {
            preds[ii] = row_ + ii * (int) ROWS_PER_LDG < min((int) ROWS, actual_seqlen_);
        }

        // Prepare the load pointers.
        void const* ptrs[LDGS];
#pragma unroll
        for (int ii = 0; ii < LDGS; ++ii)
        {
            ptrs[ii] = kv_ptr_ + (int64_t) ii * ROWS_PER_LDG * params_kv_stride_in_bytes_;
        }

        // Trigger the LDGs.
        if (col_in_bytes_ < VALID_BYTES_PER_ROW)
        {
            fmha::pack_predicates(preds_, preds);
            fmha::ldg(fetch_, ptrs, preds_);
        }
        else
        {
#pragma unroll
            for (int ii = 0; ii < LDGS; ++ii)
            {
                fetch_[ii] = make_uint4(0u, 0u, 0u, 0u);
            }
        }
    }

    // Move the pointer to the next row location.
    inline __device__ void move(int const steps = 1)
    {
        kv_ptr_ += (int64_t) ROWS * params_kv_stride_in_bytes_ * steps;
        actual_seqlen_ -= (int) ROWS * steps;
    }

    // Move the pointer to the next row location by the offset (not step).
    inline __device__ void move_by_offset(int const offset)
    {
        kv_ptr_ = kv_ptr_init_ + (int64_t) offset * params_kv_stride_in_bytes_;
        actual_seqlen_ = actual_seqlen_init_ - (int) offset;
    }

    // Move the pointer to the next column location
    inline __device__ void move_col(int const steps = 1)
    {
        kv_ptr_ += (int64_t) COLS * (BITS_PER_ELEMENT / 8) * steps;
        // Update col_in_bytes_ to ensure load predicates work
        col_in_bytes_ += THREADS_PER_ROW * BYTES_PER_LDG * steps;
    }

    inline __device__ void reset()
    {
        kv_ptr_ = kv_ptr_init_;
        actual_seqlen_ = actual_seqlen_init_;
    }

    // Rewind the pointer back to previous column location
    inline __device__ void rewind_col(int const steps)
    {
        kv_ptr_ -= COLS * (BITS_PER_ELEMENT / 8) * steps;
        // Update col_in_bytes_ to ensure load predicates work
        col_in_bytes_ -= THREADS_PER_ROW * BYTES_PER_LDG * steps;
    }

    inline __device__ void move_to(int const step)
    {
        kv_ptr_ = kv_ptr_init_ + (int64_t) ROWS * params_kv_stride_in_bytes_ * step;
        actual_seqlen_ = actual_seqlen_init_ - (int) ROWS * step;
    }

    // Store data to memory.
    inline __device__ void store(uint4 const (&data)[LDGS])
    {
#pragma unroll
        for (int ii = 0; ii < LDGS; ++ii)
        {
            char* ptr = kv_ptr_ + (int64_t) ii * ROWS_PER_LDG * params_kv_stride_in_bytes_;
            if (((row_ + ii * ROWS_PER_LDG) < min(ROWS, actual_seqlen_))
                && col_in_bytes_ < VALID_BYTES_PER_ROW /*TODO: double check*/)
            {
                fmha::stg(ptr, data[ii]);
            }
        }
    }

    // The stride between rows for the QKV matrice.
    int64_t params_kv_stride_in_bytes_;
    // The pointer.
    char* kv_ptr_;
    char* kv_ptr_init_;
    // The register to store predicates.
    uint32_t preds_[PRED_REGS];
    // The fetch registers.
    uint4 fetch_[LDGS];
    // Keep track of the row and col the thread is processing as we move the tile.
    int row_;
    int col_in_bytes_;
    // The sequence length.
    int actual_seqlen_;
    int actual_seqlen_init_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// We expect the paged KV layout to be blocks of indices with shape of [B, 2, Blocks_per_Seq],
// and the indice tells the memory distance to the pool ptr in global memory.

template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The number of bits per element.
    int BITS_PER_ELEMENT_,
    // The number of rows of Q, K or V loaded by this tile.
    int ROWS_,
    // The number of columns (padded, e.g 64).
    int COLS,
    // The actual number of columns (unpadded, e.g 40)
    int VALID_COLS_,
    // Do we use LDGSTS?
    bool USE_LDGSTS_,
    // Are attention heads interleaved? (not used)
    bool HEADS_INTERLEAVED = false,
    // The number of matrices (not used)
    int NUM_MATS = 2,
    // Is sliding window attention used ?
    bool SLIDING_WINDOW_ATTENTION_ = false>
struct Gmem_tile_paged_kv
{

    // The size of each LDG.
    enum
    {
        BYTES_PER_LDG = 16
    };

    // The number of bits/bytes of element
    enum
    {
        BITS_PER_ELEMENT = BITS_PER_ELEMENT_
    };

    enum
    {
        BYTES_PER_ELEMENT = BITS_PER_ELEMENT_ / 8
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

    // The valid size of a row in bytes (without paddings).
    enum
    {
        VALID_COLS = VALID_COLS_
    };

    // The amount of bytes that are valid per row.
    enum
    {
        VALID_BYTES_PER_ROW = VALID_COLS * BITS_PER_ELEMENT / 8
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

    // Is sliding window attention used ?
    enum
    {
        SLIDING_WINDOW_ATTENTION = SLIDING_WINDOW_ATTENTION_
    };

    // Is it Hopper?
    enum
    {
        IS_HOPPER = std::is_same<typename Traits::Gpu_arch, typename fmha::Hopper>::value == true
    };

    // Make sure we use a single register to store predicates. Do not throw for Hopper for now.
    static_assert(!USE_LDGSTS_ || PRED_REGS == 1 || IS_HOPPER, "");

    // We do not use LDGSTS (for the moment).
    enum
    {
        USE_LDGSTS = USE_LDGSTS_
    };

    // Ctor.
    template <typename Block_info>
    inline __device__ Gmem_tile_paged_kv(bert::Fused_multihead_attention_params_v2 const& params,
        int qkv_offset, // q = 0, k = 1, v = 2.
        Block_info const& binfo, int tidx, int cta_row_offset = 0, int cta_col_offset_in_bytes = 0)
        : actual_seqlen_(binfo.actual_seqlen)
        , past_seqlen_(binfo.actual_seqlen - binfo.actual_q_seqlen)
        , sliding_window_size_(params.sliding_window_size)
        , paged_kv_log2_block_size_(params.paged_kv_cache.mTokensPerBlockLog2)
        , paged_kv_block_pool_ptr_(reinterpret_cast<char*>(params.paged_kv_cache.mPoolPtr))
        , paged_kv_global_block_offsets_(params.paged_kv_cache.mBlockOffsets)
        , params_kv_block_size_in_bytes_(params.paged_kv_cache.mBytesPerBlock)
    {

        // Handle Paged KV with shape [S, Dh], by offsetting it to the target batch.
        int32_t const paged_kv_block_offset
            = (binfo.bidb * 2 + qkv_offset - 1) * params.paged_kv_cache.mMaxBlocksPerSeq;
        paged_kv_global_block_offsets_ += paged_kv_block_offset;

        // Compute the position in the sequence (within the CTA for the moment).
        int row = tidx / THREADS_PER_ROW;
        // Compute the position of the thread in the row.
        int col = tidx % THREADS_PER_ROW;

        // We must store the value to update the predicates in "load".
        row_ = row;
        // Do not load/store if the thread is in the padded area
        col_in_bytes_ = cta_col_offset_in_bytes + col * BYTES_PER_LDG;

        int64_t kv_stride_in_bytes = qkv_offset == 1 ? params.k_stride_in_bytes : params.v_stride_in_bytes;
        // The head offset.
        head_stride_in_bytes_ = (int64_t) (binfo.bidh / params.h_q_per_kv) * kv_stride_in_bytes;
        // When V is padded (like MLA), we cannot use VALID_BYTES_PER_ROW
        token_stride_in_bytes_ = kv_stride_in_bytes >> paged_kv_log2_block_size_;

        // Take the CTA offset to modify the sequence length.
        // Actually we don't need that for flash attention.
        actual_seqlen_ -= cta_row_offset;
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
        // Prepare the predicates.
        uint32_t preds[LDGS];
        // Prepare the load pointers.
        void const* ptrs[LDGS];

        // Offset for the new paged kv pointer.
        uint64_t const head_col_in_bytes = head_stride_in_bytes_ + col_in_bytes_;

// Update paged_kv ptr for each LDG (reuse is possible).
#pragma unroll
        for (int ii = 0; ii < LDGS; ++ii)
        {
            int row_idx = row_ + ii * (int) ROWS_PER_LDG;
            int paged_kv_block_idx = (row_idx >> paged_kv_log2_block_size_);
            char const* local_kv_ptr = reinterpret_cast<char*>(paged_kv_block_pool_ptr_
                + params_kv_block_size_in_bytes_ * paged_kv_global_block_offsets_[paged_kv_block_idx]);

            // Predicates.
            // TODO: do we need to make sure row_idx < ROWS ?
            preds[ii] = row_idx < actual_seqlen_;
            preds[ii] &= col_in_bytes_ < VALID_BYTES_PER_ROW;

            // Pointers.
            int row_idx_in_block = row_idx & ((1 << paged_kv_log2_block_size_) - 1);
            ptrs[ii] = local_kv_ptr + head_col_in_bytes + (int64_t) row_idx_in_block * token_stride_in_bytes_;
        }

        // Trigger LDGSTS or the LDGs.
        // The predicates protect against out-of-bound access in rows and cols
        Ldgsts_helper<USE_LDGSTS>::load(this, smem_tile, ptrs, preds);
    }

    // Move the pointer to the next row location.
    inline __device__ void move()
    {
        row_ += ROWS;
    }

    // Move the pointer to the next row location by the offset (not step).
    inline __device__ void move_by_offset(int const offset)
    {
        row_ += offset;
    }

    // Move the pointer to the next column location
    inline __device__ void move_col()
    {
        col_in_bytes_ += THREADS_PER_ROW * BYTES_PER_LDG;
    }

    // Rewind the pointer back to previous column location
    inline __device__ void rewind_col(int const steps)
    {
        // Update col_in_bytes_ to ensure load predicates work
        col_in_bytes_ -= THREADS_PER_ROW * BYTES_PER_LDG * steps;
    }

    // The stride between rows for the KV matrice.
    int64_t params_kv_block_size_in_bytes_;
    // The paged cache pool pointer.
    char* paged_kv_block_pool_ptr_;
    // The paged block offsets.
    int32_t* paged_kv_global_block_offsets_;
    // The paged block size.
    int paged_kv_log2_block_size_;
    // The register to store predicates.
    uint32_t preds_[PRED_REGS];
    // The fetch registers.
    uint4 fetch_[LDGS];
    // Keep track of the row and col the thread is processing as we move the tile.
    int row_;
    int64_t col_in_bytes_;
    // Keep track of the head offset.
    int64_t head_stride_in_bytes_;
    // // for DeepSeek MLA, the stride of V tokens != VALID_BYTES_PER_ROW
    int32_t token_stride_in_bytes_;
    // The sequence length.
    int actual_seqlen_;
    // The past sequence length (kv_seqlen - q_seqlen) considering chunked context.
    int past_seqlen_;
    // The sliding attention window size.
    int sliding_window_size_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The number of bits per element.
    int BITS_PER_ELEMENT,
    // The number of rows of Q loaded by this tile.
    int ROWS_,
    // The number of columns.
    int COLS,
    // Do we use LDGSTS?
    bool USE_LDGSTS_,
    // Are attention heads interleaved?
    bool HEADS_INTERLEAVED,
    // The number of matrices
    int NUM_MATS = 1>
struct Gmem_tile_q_kv
{

    // The size of each LDG.
    enum
    {
        BYTES_PER_LDG = 16
    };

    // The padded to the next power of 2 number of columns
    enum
    {
        COLS_PADDED = Next_power_of_two<COLS>::VALUE
    };

    // The padded size of a row in bytes.
    enum
    {
        BYTES_PER_ROW_PADDED = COLS_PADDED * BITS_PER_ELEMENT / 8
    };

    // The size of a row in bytes.
    enum
    {
        BYTES_PER_ROW = COLS * BITS_PER_ELEMENT / 8
    };

    // The number of threads to load a padded "row" of the matrix.
    enum
    {
        THREADS_PER_ROW_PADDED = BYTES_PER_ROW_PADDED / BYTES_PER_LDG
    };

    // The number of threads to load a "row" of the matrix.
    enum
    {
        THREADS_PER_ROW = BYTES_PER_ROW / BYTES_PER_LDG
    };

    // The number of "rows" loaded per LDG.
    enum
    {
        ROWS_PER_LDG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW_PADDED
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

    // Is it Hopper?
    enum
    {
        IS_HOPPER = std::is_same<typename Traits::Gpu_arch, typename fmha::Hopper>::value == true
    };

    // Make sure we use a single register to store predicates. Do not throw for Hopper for now.
    static_assert(!USE_LDGSTS_ || PRED_REGS == 1 || IS_HOPPER, "");

    // We do not use LDGSTS (for the moment).
    enum
    {
        USE_LDGSTS = USE_LDGSTS_
    };

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_q_kv(
        Params const& params, int offset, Block_info const& binfo, int tidx, int cta_row_offset = 0)
        : params_stride_in_bytes_(params.stride_in_bytes)
        , actual_seqlen_(binfo.actual_seqlen)
        , ptr_(reinterpret_cast<char*>(params.ptr))
    {

        // Compute the position in the sequence (within the CTA for the moment).
        int row = tidx / THREADS_PER_ROW_PADDED;
        // Compute the position of the thread in the row.
        int col = tidx % THREADS_PER_ROW_PADDED;

        // We must store the value to update the predicates in "load".
        row_ = row;
        // Mask for predicate if the channels are in the padded area
        int const bytes_per_row_non_padded = params.d * BITS_PER_ELEMENT / 8;
        mask_ = col < bytes_per_row_non_padded / BYTES_PER_LDG;

        // The row offset in the batched GEMM. For each seq element, we store QKV in that order.
        int64_t row_offset = (int64_t) (row + cta_row_offset) * params.stride_in_bytes;
        // Add the block index.
        int64_t idx;
        if (HEADS_INTERLEAVED)
        {
            idx = binfo.bidx * NUM_MATS + offset;
        }
        else
        {
            idx = (binfo.sum_s * NUM_MATS + offset) * params.h + binfo.bidh;
        }
        // Assemble the final pointer.
        ptr_ += row_offset + idx * bytes_per_row_non_padded + col * BYTES_PER_LDG;

        // Take the CTA offset to modify the sequence length.
        actual_seqlen_ -= cta_row_offset;
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
        uint32_t preds[LDGS];
#pragma unroll
        for (int ii = 0; ii < LDGS; ++ii)
        {
            preds[ii] = (row_ + ii * (int) ROWS_PER_LDG < min((int) ROWS, actual_seqlen_)) && mask_;
        }

        // Prepare the load pointers.
        void const* ptrs[LDGS];
#pragma unroll
        for (int ii = 0; ii < LDGS; ++ii)
        {
            ptrs[ii] = ptr_ + (int64_t) ii * ROWS_PER_LDG * params_stride_in_bytes_;
        }

        // Trigger LDGSTS or the LDGs.
        Ldgsts_helper<USE_LDGSTS>::load(this, smem_tile, ptrs, preds);
    }

    inline __device__ void move(int const steps = 1)
    {
        ptr_ += (int64_t) ROWS * params_stride_in_bytes_ * steps;
        actual_seqlen_ -= (int) ROWS * steps;
    }

    // Store data to memory.
    inline __device__ void store(uint4 const (&data)[LDGS])
    {
#pragma unroll
        for (int ii = 0; ii < LDGS; ++ii)
        {
            char* ptr = ptr_ + (int64_t) ii * ROWS_PER_LDG * params_stride_in_bytes_;
            if ((row_ + ii * ROWS_PER_LDG) < min(ROWS, actual_seqlen_))
            {
                fmha::stg(ptr, data[ii]);
            }
        }
    }

    // The stride between rows for the matrix.
    int64_t params_stride_in_bytes_;
    // The pointer.
    char* ptr_;
    // The register to store predicates.
    uint32_t preds_[PRED_REGS];
    // The fetch registers.
    uint4 fetch_[LDGS];
    // Keep track of the row and col the thread is processing as we move the tile.
    int row_;
    // Keep track of predicate state that depends only on the initialization state.
    int mask_;
    // The sequence length.
    int actual_seqlen_;
};

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
    // Do we use LDGSTS?
    bool USE_LDGSTS_>
struct Gmem_tile_qkv_interleaved
{

    // The vectorization width for NC/32HW32.
    enum
    {
        VEC = 32
    };

    // The size of each LDG.
    enum
    {
        BYTES_PER_LDG = 16
    };

    // The size of a row in bytes.
    enum
    {
        BYTES_PER_ROW = VEC * BITS_PER_ELEMENT / 8
    };

    // DEBUG.
    static_assert(BYTES_PER_ROW == 32, "");

    // END OF DEBUG.

    // The number of threads to load a "row" of the matrix.
    enum
    {
        THREADS_PER_ROW = BYTES_PER_ROW / BYTES_PER_LDG
    };

    // DEBUG.
    static_assert(THREADS_PER_ROW == 2, "");

    // END OF DEBUG.

    // The number of "rows" loaded per LDG.
    enum
    {
        ROWS_PER_LDG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW
    };

    // The number of slices. It is either 1 for DIM_PER_HEAD == 32 and 2 for DIM_PER_HEAD == 64.
    enum
    {
        NUM_SLICES = COLS / VEC
    };

    // DEBUG.
    static_assert(NUM_SLICES == 1 || NUM_SLICES == 2, "");

    // END OF DEBUG.

    // The number of rows in a slice.
    enum
    {
        ROWS = ROWS_
    };

    // The number of LDGs needed to load a chunk of the Q matrix.
    enum
    {
        LDGS = fmha::Div_up<ROWS * NUM_SLICES, ROWS_PER_LDG>::VALUE
    };

    // The number of predicate registers.
    enum
    {
        PRED_REGS = fmha::Compute_number_of_pred_regs<LDGS>::VALUE
    };

    // Make sure we use a single register to store predicates.
    static_assert(PRED_REGS == 1, "");

    // Do we use LDGSTS on Ampere?
    enum
    {
        USE_LDGSTS = USE_LDGSTS_
    };

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_qkv_interleaved(
        Params const& params, int qkv_select, Block_info const& block_info, int tidx, int cta_row_offset = 0)
        : actual_seqlen_(block_info.actual_seqlen - cta_row_offset)
        , total_(params.q_stride_in_bytes)
        , kv_ptr_(reinterpret_cast<char const*>(params.qkv_ptr))
    {

        int bidh = block_info.bidh;
        int sum_s = block_info.sum_s;

        // We must keep track of the row to repack predicates in load.
        row_ = tidx / THREADS_PER_ROW;
        // The column.
        int col = tidx % THREADS_PER_ROW;

        // h is N
        // d is H
        // we get the data in as: 3 x h x (d/32) x total x 32 (think 3 x h x (d/32)
        // x b x s x 32)

        // Loading qkv: ignore slice for now.
        int qkv_offset = qkv_select * params.h * NUM_SLICES * total_;
        // bidh * GROUPS * B * S + b * S.
        int block_offset = bidh * NUM_SLICES * total_ + sum_s;
        // The row offset.
        int row_offset = (qkv_offset + block_offset + cta_row_offset) * BYTES_PER_ROW;

        // That's the pointer to load from (see "load").
        kv_ptr_ += row_offset + col * BYTES_PER_LDG;

        init_actual_seqlen_ = actual_seqlen_;
        init_kv_ptr_ = kv_ptr_;
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
        uint32_t preds[LDGS];

// We precompute slice offsets and predicates
#pragma unroll
        for (int ii = 0; ii < LDGS; ii++)
        {
            // the next row
            int row_i = row_ + ii * ROWS_PER_LDG;

            // Decompose the current row in slice and original row
            int slice = row_i / ROWS;
            // The position in the slice.
            int row_in_slice = row_i % ROWS;

            // Update the predicate.
            preds[ii] = row_in_slice < min(actual_seqlen_, ROWS);
            // Compute the pointer.
            ptrs[ii] = &kv_ptr_[(slice * total_ + row_in_slice) * BYTES_PER_ROW];
        }

        // Update the predicate register.
        fmha::pack_predicates(preds_, preds);

        // Trigger the loads.
        if (USE_LDGSTS)
        {
            smem_tile.store(ptrs, preds_);
        }
        else
        {
            fmha::ldg(fetch_, ptrs, preds_);
        }
    }

    // Move the pointer to the next location.
    inline __device__ void move(int const steps = 1)
    {
        kv_ptr_ += (int64_t) ROWS * BYTES_PER_ROW * steps;
        actual_seqlen_ -= ROWS * steps;
    }

    // Reset to the initial location.
    inline __device__ void reset()
    {
        kv_ptr_ = init_kv_ptr_;
        actual_seqlen_ = init_actual_seqlen_;
    }

    // The pointer.
    char const* kv_ptr_;
    char const* init_kv_ptr_;
    // The register to store predicates.
    uint32_t preds_[PRED_REGS];
    // The fetch registers.
    uint4 fetch_[LDGS];
    // keep track of the row the thread is processing as we move the tile
    int row_;
    // The sequence length.
    int actual_seqlen_;
    int init_actual_seqlen_;
    // The number of rows per slice??
    int total_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace v2
} // namespace fmha
